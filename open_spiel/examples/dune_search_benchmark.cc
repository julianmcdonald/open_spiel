#include <algorithm>
#include <atomic>
#include <chrono>
#include <iostream>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <thread>
#include <vector>

#include <torch/torch.h>

#include "dune_search_benchmark_internal.h"
#include "dune_seed_utils.h"
#include "dune_sha256.h"
#include "dune_terminal_vp_report.h"
#include "open_spiel/abseil-cpp/absl/flags/flag.h"
#include "open_spiel/abseil-cpp/absl/flags/parse.h"
#include "open_spiel/abseil-cpp/absl/strings/str_cat.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"

ABSL_FLAG(std::string, model_checkpoint, "", "Path to load search agent model checkpoint.");
ABSL_FLAG(std::string, search_value_checkpoint, "", "Path to load search agent value model checkpoint for SplitPolicyValueEvaluator. If empty, inherits model_checkpoint.");
ABSL_FLAG(bool, force_split_evaluator, false, "Force using SplitPolicyValueEvaluator even when policy and value models share the same checkpoint.");
ABSL_FLAG(std::string, opponent_checkpoint, "", "Path to load opponent model checkpoint. If empty, the search agent model is reused. If 'random', random opponents are used.");
ABSL_FLAG(int, seed, 42, "Seed for deterministic RNG.");
ABSL_FLAG(int, games, 1000, "How many games to play in total.");
ABSL_FLAG(int, start_episode_id, 0, "Episode ID to start the range of games (for resumability).");
ABSL_FLAG(std::string, game_jsonl_path, "", "Path to write structured game-level JSONL outcomes.");
ABSL_FLAG(std::string, search_jsonl_path, "", "Path to write structured search-level JSONL diagnostics.");
ABSL_FLAG(std::string, aggregate_json_path, "", "Path to write structured aggregate JSON outcome.");
ABSL_FLAG(bool, record_market_diagnostics, false, "Record full market diagnostics (cards seen/bought etc.).");
ABSL_FLAG(std::string, market_diag_json_path, "", "Path to write aggregate market diagnostics JSON.");
ABSL_FLAG(std::string, market_diag_csv_path, "", "Path to write aggregate market diagnostics CSV.");
ABSL_FLAG(std::string, market_diag_games_path, "", "Path to write per-game market diagnostics JSONL.");
ABSL_FLAG(std::string, candidate_decision_traces_path, "", "Path to write compact candidate decision traces JSONL.");
ABSL_FLAG(bool, record_candidate_decision_traces, false, "Whether to record per-decision candidate search traces.");
ABSL_FLAG(std::string, output_dir, "", "Directory for output artifacts (market diagnostics, games.jsonl, etc.).");
ABSL_FLAG(int, threads, 8, "How many threads to run.");
ABSL_FLAG(int, hidden_dim, 2048, "Search agent model hidden dimension.");
ABSL_FLAG(int, num_blocks, 8, "Search agent model residual blocks count.");
ABSL_FLAG(int, opp_hidden_dim, -1, "Opponent model hidden dimension. If -1, inherits hidden_dim.");
ABSL_FLAG(int, opp_num_blocks, -1, "Opponent model block count. If -1, inherits num_blocks.");
ABSL_FLAG(int, max_simulations, 50, "MCTS simulation budget per move.");
ABSL_FLAG(double, puct_c, 0.3, "PUCT exploration constant.");
ABSL_FLAG(int, max_world_samples, -1, "Number of cached world samples for determinization (-1 = fresh resampling).");
ABSL_FLAG(double, utility_divisor, 4.0, "Terminal utility divisor (couples with search value normalization).");
ABSL_FLAG(double, temperature, 0.0, "Softmax temperature for final move choice (0.0 = greedy).");
ABSL_FLAG(double, simulated_opponent_temperature, 1.0, "Softmax temperature for simulated opponent action selection inside search (default 1.0).");
ABSL_FLAG(double, external_opponent_temperature, 0.0, "Softmax temperature for external benchmark opponent action selection (default 0.0).");
ABSL_FLAG(double, dirichlet_epsilon, 0.0, "Dirichlet noise weight at root.");
ABSL_FLAG(double, dirichlet_alpha, 0.3, "Dirichlet noise alpha.");
ABSL_FLAG(bool, rotate_seat, true, "Rotate the seat of the search agent across games.");
ABSL_FLAG(bool, use_opponent_model, false, "Whether non-search players in simulation follow the PPO prior policy instead of PUCT.");
// --- WO-LEADER-1A: search-eligible leader drafts (default-inert) --------------
// See LEADER_PICK_SEARCH_WO_2026_08_14.md Part A. Default false reproduces
// today's routing bit-for-bit: a leader draft short-circuits to
// forced_or_bookkeeping with 0 sims. When enabled, the leader node is searched
// at --leader_draft_simulations; if that budget is <=0 the leader node degrades
// to the raw prior with its own "leader_draft_no_budget" fallback reason.
ABSL_FLAG(bool, search_leader_draft, false,
          "WO-LEADER-1A: make kLeaderDraft decisions search-eligible instead of "
          "short-circuiting to forced_or_bookkeeping (default false = today's "
          "routing bit-for-bit).");
ABSL_FLAG(bool, leader_mass_only_coverage, false,
          "Leader-only coverage rule: keep the 0.50 covered-prior-mass "
          "threshold and the min-visit definition, but drop the generic "
          "three-covered-actions clause AT LEADER ROOTS ONLY. Required "
          "whenever --search_leader_draft is set: without it RunSearch rejects "
          "a concentrated 64-simulation Leader policy as 'low_coverage' and "
          "returns the raw prior, so the search runs and is discarded. Default "
          "false preserves historical behaviour exactly.");
ABSL_FLAG(int, leader_draft_simulations, 64,
          "WO-LEADER-1A: simulation budget for a searched leader draft. Consulted "
          "only when --search_leader_draft. Accepts at least 64/200/800; <=0 "
          "starves the leader node to the raw prior.");
// --- WO-PERF-3: shared batched-inference coordinator (Phase-2 integration) ----
// These flags are the permanent, opt-in instrument the deferred GPU sweep will
// drive. Default-inert: --corpus_roots=false leaves the full-game benchmark
// path (below) untouched, and --evaluator_mode=direct reproduces today's
// per-worker DuneNNEvaluator behaviour. See PERF_WORK_ORDERS_2026_08_14.md
// WO-PERF-3 and MCTS_SIMULATION_SPEED_PHASE2_FINDINGS_2026_07_29.md.
ABSL_FLAG(std::string, evaluator_mode, "direct",
          "WO-PERF-3: 'direct' (default; one DuneNNEvaluator per worker, today's "
          "behaviour bit-for-bit) or 'batched' (one shared BatchedEvaluator per "
          "model, one BatchedNNEvaluator per worker, one MCTS session per root). "
          "Batched mode is honoured ONLY in --corpus_roots mode.");
ABSL_FLAG(bool, corpus_roots, false,
          "WO-PERF-3: run the corpus-root parity/telemetry instrument instead of "
          "full games. Loads --corpus_root_path, takes the first "
          "--corpus_root_count roots of --corpus_root_role, and runs one "
          "sequential DuneSearchSession per root across --corpus_workers threads "
          "(sharing the coordinator when --evaluator_mode=batched).");
ABSL_FLAG(std::string, corpus_root_path, "data/dune_diagnostic_corpus.json",
          "WO-PERF-3: corpus JSON for --corpus_roots mode.");
ABSL_FLAG(int, corpus_root_count, 14,
          "WO-PERF-3: number of leading strategic roots to evaluate (<=0 = all).");
ABSL_FLAG(std::string, corpus_root_role, "primary",
          "WO-PERF-3: corpus-root role filter: 'primary', 'purchase', or "
          "'combat'.");
ABSL_FLAG(int, batch_target_rows, 14,
          "WO-PERF-3: target physical batch row count for the shared batcher "
          "(decoupled from --corpus_workers).");
ABSL_FLAG(int, batch_max_queue_wait_ms, 2,
          "WO-PERF-3: maximum queue wait (ms) before the batcher flushes an "
          "under-target physical batch.");
ABSL_FLAG(int, corpus_workers, 14,
          "WO-PERF-3: concurrent search-session worker threads in --corpus_roots "
          "mode (decoupled from --batch_target_rows).");
ABSL_FLAG(std::string, corpus_root_jsonl_path, "",
          "WO-PERF-3: path to write per-root JSONL (selected action + full root "
          "visit vector) for direct/batched parity comparison.");
ABSL_FLAG(std::string, corpus_controller, "session",
          "Corpus driver search controller: 'session' (legacy DuneSearchSession) "
          "or 'fresh' (fresh DunePUCTISMCTSBot with F03 Max-N config per root).");
ABSL_FLAG(int, corpus_warmup_searches, 0,
          "WO-PERF-TIMING-BATCH: run this many discarded warm-up searches "
          "before timing and before batcher telemetry is armed. Each arm is a "
          "SEPARATE PROCESS and inherits nothing from the last, so this does "
          "not amortise setup across arms: it moves THIS process's own lazy "
          "CUDA context creation, cuBLAS/cuDNN autotuning and caching-allocator "
          "growth out of its measured region, identically for every arm, so the "
          "residual is comparable. 0 = no warm-up (default, reproducing "
          "today's behaviour).");
ABSL_FLAG(std::string, batcher_telemetry_json_path, "",
          "WO-PERF-3: path to write batcher telemetry JSON (batched mode only).");
ABSL_FLAG(bool, nonlinear_value_head, false,
          "Use the versioned nonlinear value head for the search model.");
ABSL_FLAG(bool, opponent_nonlinear_value_head, false,
          "Use the versioned nonlinear value head for the external opponent model.");
ABSL_FLAG(bool, verbose_diagnostics, true, "Print IS-MCTS node-reuse and depth diagnostics periodically.");
ABSL_FLAG(bool, check_strategic_state, false, "Whether to bypass MCTS search at non-strategic states.");
// PF-2 Part B. DEFAULT-INERT, and an intentional selection change when ON —
// this is deliberately NOT covered by Part A's neutrality claim. When ON in the
// policy-only session, non-strategic multi-legal decisions sample the raw prior
// with Path-B's exact RNG semantics instead of playing argmax; everything else
// stays greedy raw. Exists so B1 has a matched-hybrid control whose ONLY
// difference from the searched arm is the strategic searched routine.
ABSL_FLAG(bool, matched_fallback_sampling, false,
          "PF-2 Part B: sample the raw prior on non-strategic multi-legal "
          "decisions using Path-B's registered RNG semantics (default off).");
ABSL_FLAG(bool, disable_time_limit, false, "Disable the time limit per move for fixed-simulation evaluation.");
ABSL_FLAG(double, root_prior_temperature, 1.0, "Root prior temperature.");
ABSL_FLAG(int, fixed_continuation_reserve, 0, "Continuation reserve simulations.");
ABSL_FLAG(int, purchase_combat_budget, 16, "Purchase/combat short-window simulation budget.");
ABSL_FLAG(double, live_continuation_reserve_seconds, 10.0, "Live continuation reserve seconds.");
ABSL_FLAG(bool, live_deadline, false, "Use live deadline budget mode instead of fixed simulations.");
ABSL_FLAG(bool, use_session, true, "If true (default), drive the search seat through the persistent DuneSearchSession. If false, reproduce the July-14 reference protocol: a fresh full search per decision via DunePUCTISMCTSBot::Step() (bot->Step(), including its sample-on-fallback selection), with no shared session budget.");
ABSL_FLAG(bool, policy_only, false, "Policy-only control arm: run the session in kPolicyOnly budget mode so the candidate plays the raw network policy on every decision (no search). Pair with --purchase_combat_budget=0 and --temperature=0 for a pure greedy-policy baseline vs the greedy opponents.");
// Audit-only bridge to the Search-PI collector's exact per-root budget mode.
// Default false preserves every historical benchmark invocation. This mode is
// intentionally explicit rather than inferred from the numeric flags: a
// fixed-session run with max_simulations=200 is NOT the training teacher.
ABSL_FLAG(bool, audit_training_policy_iteration, false,
          "Audit-only: use kTrainingPolicyIteration with independent per-root "
          "budgets and wide short-window scope. Normal gameplay defaults are "
          "unchanged when false.");
ABSL_FLAG(int, pi_primary_simulations, 200,
          "Audit-only kTrainingPolicyIteration primary-root budget.");
ABSL_FLAG(int, pi_continuation_simulations, 64,
          "Audit-only kTrainingPolicyIteration continuation-root budget.");
ABSL_FLAG(int, pi_short_window_simulations, 64,
          "Audit-only kTrainingPolicyIteration purchase/combat/other budget.");
ABSL_FLAG(std::string, fresh_search_roles, "",
          "Decomposition arm 4 (Path B only, requires --use_session=false): "
          "comma-separated decision roles that receive a fresh full search; every "
          "OTHER searched-seat decision plays the raw-prior argmax (identical to the "
          "policy-only arm's selection). Empty (default) = search every decision "
          "(unchanged full-search behavior). Recognized tokens: primary, "
          "continuation, purchase, combat, other, leader, forced.");
ABSL_FLAG(int, force_swordmaster_rounds, 0,
          "Swordmaster probe (arm B): if > 0, the searched seat is forced toward "
          "legal Swordmaster acquisition (Task-10 acquisition heuristic) on every "
          "decision while GetCurrentRound() <= this value AND it does not yet own "
          "Swordmaster; afterwards it plays normally. 0 (default) = arm A (no "
          "forcing). Uses a separate rng so paired seeds stay aligned with arm A.");
ABSL_FLAG(int, grant_swordmaster_round, 0,
          "Swordmaster endowment probe (arm B-endow): if > 0, at the searched "
          "seat's FIRST decision of this round (before it places any agent) the "
          "seat is GRANTED Swordmaster free via SetSwordmasterForTesting -- no "
          "solari cost, no forcing of any action. Persistence across later rounds "
          "is handled by the engine (permanent third agent, commit c8b3bf6); the "
          "probe adds NO persistence logic. 0 (default) = off (= arm A).");
ABSL_FLAG(double, relative_time_budget_ms, 52000.0, "Time budget per move (ms).");
// PWO-2: previously hardcoded to 50000 at the config site below, so no caller
// could raise it. A live-deadline search that hits the node ceiling terminates
// on `max_nodes` instead of the deadline (dune_puct_is_mcts.cc:900-902), which
// silently turns "50 seconds of search" into "as much search as 50000 nodes
// buys". Exposing it lets the deadline be PROVEN to be what stopped the search.
ABSL_FLAG(int, max_nodes, 50000,
          "Search tree node ceiling. Default preserves the historical hardcoded "
          "value; PWO-2 runs pin 200000.");
ABSL_FLAG(bool, conservative_override_enabled, false, "Enforce conservative override selection protocol");
ABSL_FLAG(double, conservative_covered_prior_threshold, 0.95, "Minimum covered prior mass to avoid fallback");
ABSL_FLAG(int, conservative_meaningful_visit_threshold, 10, "Minimum visits required for raw and argmax actions");
ABSL_FLAG(double, conservative_q_margin_threshold, 0.03, "Q margin threshold for MCTS to override raw");
ABSL_FLAG(double, conservative_stability_checkpoint_fraction, 0.5, "Fraction of budget at which to check stability");
ABSL_FLAG(bool, conservative_continuation_overrides_disabled, true, "Disable overrides during continuation decisions");
ABSL_FLAG(std::string, action_selection_mode, "historical_sample",
          "Final-action selection mode for fresh search bot: 'historical_sample' "
          "(default F03 behavior) or 'preserve_raw_greedy' (preserves raw greedy U21328 on bypass/fallback).");

// --- WO-1 Phase 2: seat controller topology -------------------------------
// Historically this binary supported exactly ONE searched seat (search_seat =
// rotate_seat ? g%4 : 0); every other seat was driven by --opponent_checkpoint.
// The pace program needs the homogeneous four-copy configuration (the PRIMARY
// endpoint's controller) and an explicitly-labelled mixed slice, so the seat
// topology is now a first-class flag. 'single' reproduces the legacy behavior
// byte-for-byte and is the default.
ABSL_FLAG(std::string, controller_mode, "single",
          "Seat controller topology. 'single' (default, legacy): exactly one "
          "searched seat, the other three driven by --opponent_checkpoint. "
          "'homogeneous': all four seats run the search controller (the "
          "four-copy production-controller configuration). 'mixed': the "
          "rotating candidate seat is searched, the other three run "
          "--mixed_opponent_controller.");
ABSL_FLAG(std::string, mixed_opponent_controller, "random_legal",
          "Controller for the three non-candidate seats when "
          "--controller_mode=mixed. 'random_legal': uniform-random legal "
          "action (slice S1). 'raw_policy': raw network prior from "
          "--opponent_checkpoint (or the candidate checkpoint when unset), "
          "selected at --external_opponent_temperature.");
// Formerly the hard-coded 10.0f passed to every DuneNNEvaluator here. Split
// per-model and named to match dune_population_eval's flags of the same name.
// Default 10.0 preserves prior behavior exactly.
ABSL_FLAG(double, candidate_logit_cap, 10.0,
          "Logit cap for the CANDIDATE / searched-seat evaluator. Default 10.0 "
          "preserves prior behavior (formerly hard-coded).");
ABSL_FLAG(double, opponent_logit_cap, 10.0,
          "Logit cap for OPPONENT-model evaluators. Default 10.0 preserves "
          "prior behavior (formerly hard-coded).");

namespace open_spiel {

// Checkpoint digests, computed ONCE in main() BEFORE any worker thread is
// spawned and only read afterwards. That write-before-spawn ordering is what
// makes these safe to touch from every worker without a lock; do not assign to
// them from inside WorkerThread.
std::string g_candidate_ckpt_sha256;
std::string g_candidate_value_ckpt_sha256;
std::string g_opponent_ckpt_sha256;
bool g_using_split_evaluator = false;

// Terminal VP for REPORTING (vp_margin, search_vp, opponent_vps).
//
// Delegates to the ONE definition in dune_terminal_vp_report.h, which calls the
// engine. This used to be a hand-copied reimplementation of ComputeEndgameVp
// that awarded the all-factions-at-3 +1 without the tech-tile-8 (Memocorders)
// guard the engine applies -- see that header for the full history and the
// measured corpus impact. DO NOT reintroduce a local copy.
//
// Precondition: the state is terminal. All callers below run after
// `while (!state->IsTerminal())` has exited.
int GetTrueFinalVp(const dune_imperium::DuneImperiumState* dune_state, int p) {
  return dune_report::TerminalVpForReporting(dune_state, p);
}

}  // namespace open_spiel

int main(int argc, char* argv[]) {
  auto run_start_time = std::chrono::steady_clock::now();
  absl::ParseCommandLine(argc, argv);

  // Leader search is ADOPTED SEMANTICS, not a set of independent knobs. It was
  // adopted at exactly 64 simulations with mass-only coverage; a benchmark that
  // enabled it at some other budget, or without the coverage rule, would be
  // measuring something the program never adopted -- and in the mass-only case
  // would silently measure the RAW PRIOR, because RunSearch rejects the
  // concentrated Leader policy as "low_coverage" and returns the prior instead.
  // Rejected here rather than honoured quietly. Defaults are false/64/false, so
  // every historical invocation is unaffected.
  if (absl::GetFlag(FLAGS_search_leader_draft)) {
    if (absl::GetFlag(FLAGS_leader_draft_simulations) != 64) {
      open_spiel::SpielFatalError(absl::StrCat(
          "--search_leader_draft is adopted at exactly 64 simulations; "
          "--leader_draft_simulations=",
          absl::GetFlag(FLAGS_leader_draft_simulations),
          " is not an adjustable experimental knob."));
    }
    if (!absl::GetFlag(FLAGS_leader_mass_only_coverage)) {
      open_spiel::SpielFatalError(
          "--search_leader_draft requires --leader_mass_only_coverage=true: "
          "without it the search-result path rejects the concentrated Leader "
          "policy as 'low_coverage' and returns the raw prior, so the benchmark "
          "would report a Leader search it then discarded.");
    }
  }

  // PWO-5 section 10.2: the reserved final-gate base-seed range. No training
  // OR EVALUATION seed may enter [9000000, 9999999]; checked before any work.
  {
    const std::string stop = dune_seed::ReservedFinalGateSeedStop(
        "--seed", static_cast<long long>(absl::GetFlag(FLAGS_seed)));
    if (!stop.empty()) {
      std::cerr << stop << "\n";
      return 1;
    }
  }

  // The training-policy-iteration bridge is deliberately opt-in and
  // fail-closed. In particular, do not silently combine it with the legacy
  // live-deadline or policy-only selectors: those would produce plausible
  // output while no longer representing the registered training teacher.
  if (absl::GetFlag(FLAGS_audit_training_policy_iteration)) {
    if (!absl::GetFlag(FLAGS_use_session)) {
      open_spiel::SpielFatalError(
          "--audit_training_policy_iteration=true requires --use_session=true.");
    }
    if (absl::GetFlag(FLAGS_policy_only)) {
      open_spiel::SpielFatalError(
          "--audit_training_policy_iteration=true cannot be combined with "
          "--policy_only=true.");
    }
    if (absl::GetFlag(FLAGS_live_deadline)) {
      open_spiel::SpielFatalError(
          "--audit_training_policy_iteration=true cannot be combined with "
          "--live_deadline=true.");
    }
    if (absl::GetFlag(FLAGS_pi_primary_simulations) <= 0 ||
        absl::GetFlag(FLAGS_pi_continuation_simulations) <= 0 ||
        absl::GetFlag(FLAGS_pi_short_window_simulations) <= 0) {
      open_spiel::SpielFatalError(
          "kTrainingPolicyIteration audit budgets must all be positive.");
    }
  }

  // Arm-4 (agent-phase decomposition): the --fresh_search_roles filter is only
  // defined for the Path B fresh-search driver. Fail fast on misuse or typos,
  // before any threads spawn.
  if (!absl::GetFlag(FLAGS_fresh_search_roles).empty()) {
    if (absl::GetFlag(FLAGS_use_session)) {
      open_spiel::SpielFatalError(
          "--fresh_search_roles requires --use_session=false (Path B fresh search).");
    }
    (void)open_spiel::ParseFreshSearchRoles(absl::GetFlag(FLAGS_fresh_search_roles));
  }

  // PF-2 Part B: fail-closed configuration guard, over EVERY condition that
  // defines the matched-hybrid controller — not just the ones whose violation
  // is easy to notice. Each clause below has a distinct silent-failure mode,
  // and none of them crashes on its own; the arm would simply stop being the
  // controller B1 registered while still producing a full set of plausible
  // numbers. That is the failure mode this whole work order is built to
  // prevent, so every condition is checked, in one place, fatally.
  if (absl::GetFlag(FLAGS_matched_fallback_sampling)) {
    // (a) The strategic gate. DEFAULTS FALSE in this binary. With it off, the
    // Part-B branch is inert and the matched-hybrid arm silently becomes the
    // greedy-raw arm: three B1 arms collapse to two, identical numbers the
    // only symptom.
    if (!absl::GetFlag(FLAGS_check_strategic_state)) {
      open_spiel::SpielFatalError(
          "--matched_fallback_sampling=true requires --check_strategic_state=true. "
          "With the strategic check off the matched-hybrid arm silently becomes "
          "the greedy-raw arm.");
    }
    // (b) Temperature. The two-phase path passes r_val = 0.0 into
    // SelectControllerAction, inert ONLY on the argmax branch of
    // PickActionRespectingTemperature. At any positive temperature that 0.0
    // becomes a live, wrong draw.
    if (absl::GetFlag(FLAGS_temperature) != 0.0) {
      open_spiel::SpielFatalError(
          "--matched_fallback_sampling=true requires --temperature=0. At a "
          "positive temperature the two-phase path's r_val=0.0 would become a "
          "live, incorrect draw inside SelectControllerAction.");
    }
    // (c) The session. Part B lives entirely on the session branch; with
    // --use_session=false the binary constructs the fresh-search bot instead
    // (:534/:541) and the flag would do nothing at all — the arm would be
    // Path B, i.e. the searched arm, wearing the matched arm's label.
    if (!absl::GetFlag(FLAGS_use_session)) {
      open_spiel::SpielFatalError(
          "--matched_fallback_sampling=true requires --use_session=true. On the "
          "fresh path the flag is inert and the arm is not the matched-hybrid "
          "controller.");
    }
    // (d)+(e) Policy-only with no purchase/combat budget. These define "raw"
    // for this arm. With search budget anywhere in the session the arm is a
    // searched controller, and the primary contrast stops isolating the
    // strategic searched routine — the round-8 confound, reintroduced
    // quietly.
    if (!absl::GetFlag(FLAGS_policy_only)) {
      open_spiel::SpielFatalError(
          "--matched_fallback_sampling=true requires --policy_only=true; "
          "otherwise the matched arm is a searched controller.");
    }
    if (absl::GetFlag(FLAGS_purchase_combat_budget) != 0) {
      open_spiel::SpielFatalError(
          "--matched_fallback_sampling=true requires "
          "--purchase_combat_budget=0; a positive short-window budget gives the "
          "matched arm search at purchase/combat decisions.");
    }
    // (f) Exactly one searched seat. The Part-B ordinal is ONE counter per
    // game, mirroring ONE Path-B bot's search_count_. Path B keeps a separate
    // counter per bot, so with more than one searched seat a single shared
    // ordinal interleaves the seats' decisions into one sequence and every
    // derived sampling seed diverges from the Path-B stream it is contracted
    // to reproduce. The ordinal-sequence test would catch this — but only if
    // it were run in homogeneous mode, so the guard carries it instead.
    if (open_spiel::ParseControllerMode(absl::GetFlag(FLAGS_controller_mode)) !=
        open_spiel::SeatControllerMode::kSingle) {
      open_spiel::SpielFatalError(
          "--matched_fallback_sampling=true requires --controller_mode=single. "
          "The Part-B decision ordinal is one counter per game and mirrors one "
          "Path-B bot; with multiple searched seats the shared counter cannot "
          "reproduce any seat's sampling stream.");
    }
  }

  // Set PyTorch thread limit to 1 to avoid thread contention across CPU-bound forward runs
  at::set_num_threads(1);
  at::set_num_interop_threads(1);

  std::string model_ckpt = absl::GetFlag(FLAGS_model_checkpoint);
  if (model_ckpt.empty()) {
    std::cerr << "Error: --model_checkpoint is required!\n";
    return 1;
  }

  // --- WO-1 Phase 2: validate the seat controller topology ------------------
  const open_spiel::SeatControllerMode cli_controller_mode =
      open_spiel::ParseControllerMode(absl::GetFlag(FLAGS_controller_mode));
  open_spiel::ParseMixedOpponentController(
      absl::GetFlag(FLAGS_mixed_opponent_controller));  // validate early
  if (cli_controller_mode != open_spiel::SeatControllerMode::kSingle &&
      !absl::GetFlag(FLAGS_use_session)) {
    std::cerr << "Error: --controller_mode=" << absl::GetFlag(FLAGS_controller_mode)
              << " requires --use_session=true. Path B (--use_session=false) is "
                 "the single-seat July-14 reference protocol and has no "
                 "multi-seat form; running it here would silently search only "
                 "one seat.\n";
    return 1;
  }
  if (cli_controller_mode == open_spiel::SeatControllerMode::kHomogeneous &&
      absl::GetFlag(FLAGS_opponent_checkpoint) != "random" &&
      !absl::GetFlag(FLAGS_opponent_checkpoint).empty()) {
    std::cerr << "Warning: --controller_mode=homogeneous searches all four "
                 "seats with the candidate model; --opponent_checkpoint="
              << absl::GetFlag(FLAGS_opponent_checkpoint)
              << " is unused and will not appear in seat provenance.\n";
  }

  // Checkpoint digests for per-seat provenance. Computed ONCE here, before any
  // worker thread exists, and only read afterwards -- that ordering is what
  // makes the globals safe without a lock.
  open_spiel::g_candidate_ckpt_sha256 =
      open_spiel::ComputeFileSHA256(model_ckpt);
  {
    const std::string opp = absl::GetFlag(FLAGS_opponent_checkpoint);
    if (opp == "random") {
      open_spiel::g_opponent_ckpt_sha256 = "";
    } else if (opp.empty()) {
      open_spiel::g_opponent_ckpt_sha256 = open_spiel::g_candidate_ckpt_sha256;
    } else {
      open_spiel::g_opponent_ckpt_sha256 = open_spiel::ComputeFileSHA256(opp);
    }
  }

  std::string value_ckpt = absl::GetFlag(FLAGS_search_value_checkpoint);
  bool force_split = absl::GetFlag(FLAGS_force_split_evaluator);
  if (value_ckpt.empty()) {
    value_ckpt = model_ckpt;
  }
  open_spiel::g_candidate_value_ckpt_sha256 = open_spiel::ComputeFileSHA256(value_ckpt);
  open_spiel::g_using_split_evaluator =
      force_split || (!absl::GetFlag(FLAGS_search_value_checkpoint).empty() && value_ckpt != model_ckpt);

  int hidden_dim = absl::GetFlag(FLAGS_hidden_dim);
  int num_blocks = absl::GetFlag(FLAGS_num_blocks);
  int opp_hidden_dim = absl::GetFlag(FLAGS_opp_hidden_dim);
  if (opp_hidden_dim < 0) opp_hidden_dim = hidden_dim;
  int opp_num_blocks = absl::GetFlag(FLAGS_opp_num_blocks);
  if (opp_num_blocks < 0) opp_num_blocks = num_blocks;

  std::shared_ptr<const open_spiel::Game> game = open_spiel::LoadGame("dune_imperium");
  int64_t obs_size = game->InformationStateTensorSize();
  int64_t action_size = game->NumDistinctActions();

  torch::Device device = torch::cuda::is_available() ? torch::Device(torch::kCUDA) : torch::Device(torch::kCPU);

  std::cout << "Loading search policy model weights from: " << model_ckpt << " on device " << device << "\n";
  auto search_model = open_spiel::LoadDuneBenchmarkModel(
      model_ckpt, obs_size, action_size, hidden_dim, num_blocks,
      absl::GetFlag(FLAGS_nonlinear_value_head), device);

  std::shared_ptr<open_spiel::SharedDunePolicyValueNetImpl> search_value_model = nullptr;
  if (open_spiel::g_using_split_evaluator) {
    if (value_ckpt == model_ckpt) {
      search_value_model = search_model;
      std::cout << "Using SplitPolicyValueEvaluator with shared network (aliased policy/value control arm)\n";
    } else {
      std::cout << "Loading search value model weights from: " << value_ckpt << " on device " << device << "\n";
      search_value_model = open_spiel::LoadDuneBenchmarkModel(
          value_ckpt, obs_size, action_size, hidden_dim, num_blocks,
          absl::GetFlag(FLAGS_nonlinear_value_head), device);
    }
  }

  // WO-PERF-3: the corpus-root instrument is an entirely separate driver from
  // the full-game benchmark below. It is entered here, after the (identical)
  // search-model load and before the opponent model / game workers, and returns
  // directly -- so with --corpus_roots=false (default) the game path below is
  // byte-for-byte unchanged (acceptance B1).
  if (absl::GetFlag(FLAGS_corpus_roots)) {
    return open_spiel::RunCorpusRootBenchmark(game, search_model, device);
  }

  std::shared_ptr<open_spiel::SharedDunePolicyValueNetImpl> opponent_model = nullptr;
  std::string opp_ckpt = absl::GetFlag(FLAGS_opponent_checkpoint);
  if (opp_ckpt != "random") {
    if (opp_ckpt.empty()) {
      opp_ckpt = model_ckpt;
    }
    std::cout << "Loading opponent model weights from: " << opp_ckpt << " on device " << device << "\n";
    opponent_model = open_spiel::LoadDuneBenchmarkModel(
        opp_ckpt, obs_size, action_size, opp_hidden_dim, opp_num_blocks,
        absl::GetFlag(FLAGS_opponent_nonlinear_value_head), device);
  } else {
    std::cout << "Running against uniform random opponents.\n";
  }

  int total_games = absl::GetFlag(FLAGS_games);
  int num_threads = absl::GetFlag(FLAGS_threads);
  if (num_threads <= 0) {
    num_threads = std::max(1, (int)std::thread::hardware_concurrency());
  }

  std::cout << "\nStarting evaluation of " << total_games << " games using " << num_threads << " threads...\n\n";

  const bool record_market_diagnostics = absl::GetFlag(FLAGS_record_market_diagnostics);
  open_spiel::ConfigureOutputPathsFromOutputDir(record_market_diagnostics);

  std::vector<open_spiel::dune_market_diag::CompactPerGameMarketDiagnostics> market_results;
  if (record_market_diagnostics) {
    market_results.resize(total_games);
  }

  std::atomic<int> next_game_id{0};
  std::atomic<int> completed_games{0};
  std::mutex log_mutex;
  std::mutex stats_mutex;
  open_spiel::GameStats global_stats;

  std::vector<open_spiel::dune_market_diag::CompactPerGameMarketDiagnostics>* market_results_ptr =
      record_market_diagnostics ? &market_results : nullptr;

  const std::string full_game_eval_mode = absl::GetFlag(FLAGS_evaluator_mode);
  if (full_game_eval_mode != "direct" && full_game_eval_mode != "batched") {
    std::cerr << "Error: --evaluator_mode must be 'direct' or 'batched' (got '"
              << full_game_eval_mode << "')\n";
    return 1;
  }
  const bool batched_mode = (full_game_eval_mode == "batched");

  std::shared_ptr<open_spiel::BatchedEvaluator> search_coord = nullptr;
  std::shared_ptr<open_spiel::BatchedEvaluator> opp_coord = nullptr;
  std::shared_mutex search_coord_mutex;
  std::shared_mutex opp_coord_mutex;

  if (batched_mode) {
    open_spiel::InitSharedBatchedCoordinators(
        search_model, opponent_model, device,
        search_coord_mutex, opp_coord_mutex,
        &search_coord, &opp_coord);
  }

  std::vector<std::thread> worker_threads;
  worker_threads.reserve(num_threads);
  for (int t = 0; t < num_threads; ++t) {
    worker_threads.emplace_back([&, t]() {
      open_spiel::WorkerThread(
          t, game, search_model, search_value_model, opponent_model,
          next_game_id, total_games, completed_games,
          log_mutex, global_stats, stats_mutex,
          record_market_diagnostics, market_results_ptr,
          search_coord, opp_coord);
    });
  }

  for (auto& th : worker_threads) {
    if (th.joinable()) th.join();
  }

  if (batched_mode) {
    open_spiel::PrintBatcherCoordinatorStats(search_coord, opp_coord);
  }

  if (record_market_diagnostics) {
    open_spiel::WriteMarketDiagnosticsReports(market_results, total_games);
  }

  open_spiel::PrintAndWriteFinalBenchmarkReport(
      global_stats, total_games, num_threads, model_ckpt, opp_ckpt,
      opponent_model != nullptr, full_game_eval_mode, batched_mode,
      search_coord, opp_coord, run_start_time);

  return 0;
}
