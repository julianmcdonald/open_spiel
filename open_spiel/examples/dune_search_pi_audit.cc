// Search-PI teacher audit: how strong is the search-PI teacher, at exactly the
// configuration collection runs it?
//
// WHY THIS BINARY EXISTS. Rung 2 left arm D NEUTRAL -- five generations of
// search policy iteration produced no policy stronger than the checkpoint they
// started from -- and the open question is whether the teacher has an edge to
// distil at all (H_plateau) or has none (H_floor). Answering it needs the
// SEARCHED controller's strength, and the repository had no path that measured
// it (rung-3 path document, section 7a):
//
//   * dune_search_benchmark spends a CUMULATIVE session budget, so the primary
//     consumes the pool and every continuation reports
//     fixed_session_limit_exceeded -- and a starved path degrades to raw-prior
//     argmax. It measures a searched primary with raw continuations.
//   * dune_population_eval has no session at all. It IS the raw controller.
//
// Both measure a different controller than collection. So this binary does not
// re-implement the controller: it CALLS SearchPiGenerator::GenerateGeneration,
// the same function the trainer collects with, and the controller-parity smoke
// replays a recorded generation through it and requires the recorded per-role
// budgets and the recorded extended hash chain back. "Same controller as
// collection" is then a reproduced measurement rather than an assertion -- and
// the smoke, not flag exegesis, settles any configuration-semantics question.
//
// THE TWO ARMS. Both play complete games; one designated seat per episode is
// the measured seat and the other three play the frozen raw policy at T=1.
//   * searched   -- the collection controller: 200 primary / 64 continuation
//                   NEW simulations per decision on one persistent placement
//                   session, behaviour temperature 0, epsilon 0, no Leader
//                   search, and raw T=1 for the seat's own off-scope decisions.
//   * raw_argmax -- the MATCHED CONTROL: identical in every one of those
//                   respects except that the primary and continuation actions
//                   come from the raw-prior argmax and no simulation is spent.
// Seat assignment, chance seeds and episode ids are functions of the episode id
// alone, so the two arms are PAIRED and the analysis is a per-episode join.
//
// This binary measures. It decides nothing, and it writes no checkpoint.

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "open_spiel/abseil-cpp/absl/flags/flag.h"
#include "open_spiel/abseil-cpp/absl/flags/parse.h"
#include "open_spiel/abseil-cpp/absl/strings/str_cat.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"
#include "dune_batched_evaluator.h"
#include "dune_evaluator.h"
#include "dune_network.h"
#include "dune_search_pi.h"
#include "dune_seed_utils.h"

// --- What to measure -------------------------------------------------------
ABSL_FLAG(std::string, model_checkpoint, "",
          "Model whose teacher is audited. Loaded once, never written.");
ABSL_FLAG(std::string, arm, "both",
          "searched | raw_argmax | both. 'both' runs the searched arm first so "
          "a run killed part-way still leaves the expensive arm complete.");
ABSL_FLAG(int, games, 16, "Games per arm. Must be a multiple of 4 (seat balance).");
ABSL_FLAG(int64_t, first_episode_id, 0,
          "Episode id of the first game. Seat, chance stream and every seed "
          "derive from it, so two arms sharing this range are paired.");
ABSL_FLAG(int, generation, 1,
          "Generation stamp. NOT cosmetic: it enters the row identity and "
          "therefore both hash chains, so a parity smoke must pass the "
          "generation the reference was collected at.");
ABSL_FLAG(std::string, output_dir, "", "Where the artifacts are written.");

// --- The collection configuration, named exactly as the launcher names it ---
ABSL_FLAG(int, search_pi_primary_simulations, 200, "NEW sims at kAgentPrimary.");
ABSL_FLAG(int, search_pi_continuation_simulations, 64,
          "NEW sims per kAgentContinuation.");
ABSL_FLAG(std::string, search_pi_continuation_target, "total_visits",
          "total_visits | new_visits_only.");
ABSL_FLAG(double, search_pi_behavior_temperature, 0.0, "0 = argmax.");
ABSL_FLAG(double, search_pi_dirichlet_epsilon, 0.0, "Root noise. 0 for this lane.");
ABSL_FLAG(double, search_pi_dirichlet_alpha_total, 10.83, "Inert while eps == 0.");
ABSL_FLAG(double, search_pi_forced_playouts_k, 0.0, "Inert while eps == 0.");
ABSL_FLAG(bool, search_pi_root_noise_fpu_zero, false, "Inert while eps == 0.");
ABSL_FLAG(double, search_pi_root_prior_temperature, 1.0, "Root prior temperature.");
ABSL_FLAG(double, search_pi_target_sharpen_exponent, 1.0, "Inert at 1.0.");
ABSL_FLAG(double, search_pi_puct_c, 0.30, "PUCT exploration constant.");
ABSL_FLAG(double, search_pi_opponent_temperature, 1.0, "Opponent model temperature.");
ABSL_FLAG(double, search_pi_non_search_temperature, 1.0, "Opponent seats, raw T.");
ABSL_FLAG(double, search_pi_unsearched_role_temperature, 1.0,
          "The measured seat's OFF-SCOPE decisions, raw T. Shared by both arms.");
ABSL_FLAG(uint64_t, search_pi_seed_domain, 0, "REQUIRED nonzero.");

// --- Machinery -------------------------------------------------------------
ABSL_FLAG(int, hidden_dim, 2048, "Network hidden dimension.");
ABSL_FLAG(int, num_blocks, 8, "Network residual block count.");
ABSL_FLAG(double, logit_cap, 10.0, "Evaluator logit cap. The trainer's default.");

// dune_ppo_training_utils.cc DECLARES these and dune_ppo_train.cc defines them,
// so a binary that links the former without the latter has to supply them --
// the same block dune_search_pi_test.cc:37-50 carries, for the same reason.
// dune_search_pi.cc needs CenterAndCapLogitsTensor from that translation unit,
// which drags in TrainPpoUpdate, which this binary never calls. Only
// --logit_cap above is read here; the rest exist to resolve at link time.
ABSL_FLAG(int, ppo_minibatch_size, 2048, "");
ABSL_FLAG(int, ppo_update_epochs, 4, "");
ABSL_FLAG(double, ppo_clip_epsilon, 0.2, "");
ABSL_FLAG(bool, normalize_advantages, true, "");
ABSL_FLAG(bool, ppo_clip_value_loss, true, "");
ABSL_FLAG(double, entropy_coef, 0.01, "");
ABSL_FLAG(double, value_coef, 0.5, "");
ABSL_FLAG(double, target_kl, 0.0, "");
ABSL_FLAG(bool, train_amp, true, "");
ABSL_FLAG(double, grad_clip_norm, 0.5, "");
ABSL_FLAG(uint64_t, shaping_start_env_steps, 206830543, "");
ABSL_FLAG(uint64_t, shaping_decay_env_steps, 0, "");
ABSL_FLAG(bool, diagnostics_only, false, "");
ABSL_FLAG(bool, nonlinear_value_head, false, "Nonlinear value head.");
ABSL_FLAG(int, seed, 20276001,
          "Master seed. Feeds torch::manual_seed exactly as the trainer does; "
          "it does not seed the game, which draws from --search_pi_seed_domain.");
ABSL_FLAG(bool, deterministic, true, "Strict LibTorch deterministic algorithms.");
// --- The batched teacher ---------------------------------------------------
//
// EVERY DEFAULT BELOW REPRODUCES THE HISTORICAL UNBATCHED LANE EXACTLY. A run
// that passes none of these flags is the rung-3a instrument bit-for-bit, which
// is what lets the batch-1 parity references stay meaningful references.
ABSL_FLAG(int, batch_target, 0,
          "0 = REFERENCE MODE: one unbatched DuneNNEvaluator per worker, the "
          "historical path, which is what reproduces the rung-2 hash chains "
          "bitwise. >0 = production batched mode: ONE shared BatchedEvaluator "
          "with this target row count, serving every worker. Registered as "
          "4*G for G concurrent games, since a leaf group is 4 rows "
          "(num_players). Routing batch-1 through the unbatched path is the "
          "allowance recorded in proposal §8.1 Q4 -- preserving a reference "
          "mode, not evading the check.");
ABSL_FLAG(int, batcher_timeout_ms, 2,
          "Batcher flush timeout. Phase-2's registered operating point; it "
          "swept {0,1,2,5}. Inert unless --batch_target > 0.");
ABSL_FLAG(int, warmup_games, 0,
          "Discarded warm-up games, run on a THROWAWAY evaluator that is "
          "destroyed before the scored one is constructed. This is not "
          "optional hygiene: EnableBatcherTelemetry() arms the per-batch "
          "sections but does NOT reset the always-on counters, so a warm-up "
          "sharing the scored batcher would leave group_rows above the "
          "leaf-evaluation delta and device_timed_batches permanently below "
          "physical_batches -- making every first batched arm look like an "
          "instrument fault. Warm-up episodes come from the reserved 900,000+ "
          "range so they never replay a scored episode.");
ABSL_FLAG(std::string, batcher_telemetry_json_path, "",
          "Where to write batcher telemetry. Its ABSENCE on a batched arm is a "
          "STOP (Phase-2 S2), so a batched run should always set it.");

ABSL_FLAG(int, search_pi_purchase_combat_budget, 0,
          "NEW simulations at kPurchase / kCombatIntrigue / kOtherOptional -- "
          "one budget covers all three. 0 = the NARROW teacher (agent turns "
          "only), which is what rungs 1/2/3a measured. The registered WIDE arm "
          "is 64: the lane's own continuation budget, with PF-C precedent, at "
          "1.392x the narrow arm's simulation load. The out-of-lane default of "
          "16 is 1.098x -- a token budget that would test 10% more compute "
          "rather than scope. A parameter of the wide teacher's IDENTITY.");
ABSL_FLAG(double, search_pi_relative_time_budget_ms, 10000.0,
          "Per-root wall-clock watchdog. 10,000 ms is the historical inherited "
          "struct default and is kept as the flag default so an unflagged run "
          "matches history. The batched teacher registers 60,000 ms as a FATAL "
          "watchdog with NO shortfall tolerance -- its only function is runaway "
          "protection, and the gate that protects the science is the budget "
          "invariant (completed == requested), not this clock.");

ABSL_FLAG(int, threads, 1,
          "Concurrent game workers. NOT a free throughput knob: every searched "
          "root runs under a live 10,000 ms wall-clock deadline (the lane never "
          "sets DuneSearchConfig::relative_time_budget_ms, so it keeps its "
          "default, and dune_puct_is_mcts.cc:982 breaks the simulation loop on "
          "it). Parallel games share one CUDA stream, per-simulation latency "
          "rises, and a truncated search weakens ONLY the arm that searches -- a "
          "one-sided bias toward 'no teacher edge'. Raise this only after a "
          "pre-measure cell at the same value reports zero short searches with "
          "real headroom; --fail_on_short_search is what enforces it.");
ABSL_FLAG(bool, fail_on_short_search, true,
          "Exit nonzero if any searched root spent less than its configured "
          "budget. This is the searched arm's manipulation check, enforced by "
          "the instrument rather than left to a downstream script: a cell that "
          "silently under-searched is invalid, not weak.");
ABSL_FLAG(int, chunk_games, 4,
          "Games per work unit. A multiple of 4 so every chunk is itself seat "
          "balanced; the generator asserts it.");
ABSL_FLAG(bool, retain_rows, false,
          "Keep every emitted row so both hash chains can be computed over the "
          "whole arm. Required by the parity smoke; costs ~2 MB per game, so "
          "leave it off for a large audit.");
ABSL_FLAG(std::string, expect_extended_chain, "",
          "Parity smoke: the recorded extended chain this run must reproduce. "
          "A mismatch is a nonzero exit -- the runner is wrong, never the "
          "reference.");
ABSL_FLAG(std::string, expect_target_chain, "",
          "Parity smoke: the recorded legacy target chain to reproduce.");

namespace open_spiel {
namespace {

// One arm's merged result. Chunks are merged in EPISODE order, never in
// completion order, so the artifacts and both hash chains are independent of
// how the work happened to be scheduled.
struct ArmResult {
  SearchPiArm arm = SearchPiArm::kSearched;
  std::vector<SearchPiGameOutcome> games;
  SearchPiRoleStats primary;
  SearchPiRoleStats continuation;
  int64_t simulations_by_role[7] = {0, 0, 0, 0, 0, 0, 0};
  int64_t decisions_by_role[7] = {0, 0, 0, 0, 0, 0, 0};
  int64_t leader_rows_emitted = 0;
  int64_t inference_calls = 0;
  int64_t rows_total = 0;
  std::string target_hash_chain;    // empty unless --retain_rows
  std::string extended_hash_chain;  // empty unless --retain_rows
  double wall_time_s = 0.0;
};

void MergeRole(SearchPiRoleStats* dst, const SearchPiRoleStats& src) {
  dst->roots_seen += src.roots_seen;
  dst->searches_run += src.searches_run;
  dst->rows_emitted += src.rows_emitted;
  dst->fallbacks += src.fallbacks;
  dst->simulations_completed += src.simulations_completed;
  dst->simulations_requested += src.simulations_requested;
  dst->searches_short_of_budget += src.searches_short_of_budget;
  dst->simulation_shortfall += src.simulation_shortfall;
  for (int i = 0; i < kSearchPiEarlyExitCount; ++i) {
    dst->early_exit_counts[i] += src.early_exit_counts[i];
  }
  // "First" is the first in EPISODE order, and chunks merge in episode order,
  // so the first chunk that saw one wins. Keeping the earliest rather than the
  // last is what makes the field reproducible under any thread count.
  if (dst->first_short_episode_id < 0 && src.first_short_episode_id >= 0) {
    dst->first_short_episode_id = src.first_short_episode_id;
    dst->first_short_decision_id = src.first_short_decision_id;
    dst->first_short_simulations_completed = src.first_short_simulations_completed;
    dst->first_short_simulations_requested = src.first_short_simulations_requested;
    dst->first_short_bot_reason = src.first_short_bot_reason;
    dst->first_short_session_reason = src.first_short_session_reason;
  }
  dst->max_search_elapsed_ms =
      std::max(dst->max_search_elapsed_ms, src.max_search_elapsed_ms);
  dst->sum_search_elapsed_ms += src.sum_search_elapsed_ms;
  // The deadline is a configuration constant, identical in every chunk; taking
  // the max rather than the last keeps a chunk that ran no searches (and so left
  // it at 0) from erasing it.
  dst->configured_time_limit_ms =
      std::max(dst->configured_time_limit_ms, src.configured_time_limit_ms);
  dst->inherited_visits += src.inherited_visits;
  dst->re_root_hits += src.re_root_hits;
  dst->re_root_misses += src.re_root_misses;
  dst->kept_despite_legacy_gate += src.kept_despite_legacy_gate;
  dst->sum_target_entropy_norm += src.sum_target_entropy_norm;
  dst->sum_raw_entropy_norm += src.sum_raw_entropy_norm;
  dst->sum_kl_target_given_raw += src.sum_kl_target_given_raw;
  dst->target_argmax_overrides += src.target_argmax_overrides;
}

std::string F17(double v) {
  std::ostringstream ss;
  ss << std::setprecision(17) << v;
  return ss.str();
}

void EmitRoleJson(std::ostream& os, const char* prefix,
                  const SearchPiRoleStats& r) {
  os << ",\"" << prefix << "_roots_seen\":" << r.roots_seen
     << ",\"" << prefix << "_searches_run\":" << r.searches_run
     << ",\"" << prefix << "_rows_emitted\":" << r.rows_emitted
     << ",\"" << prefix << "_fallbacks\":" << r.fallbacks
     << ",\"" << prefix << "_simulations_completed\":" << r.simulations_completed
     << ",\"" << prefix << "_simulations_requested\":" << r.simulations_requested
     << ",\"" << prefix << "_searches_short_of_budget\":"
     << r.searches_short_of_budget
     << ",\"" << prefix << "_simulation_shortfall\":" << r.simulation_shortfall
     << ",\"" << prefix << "_inherited_visits\":" << r.inherited_visits
     << ",\"" << prefix << "_re_root_hits\":" << r.re_root_hits
     << ",\"" << prefix << "_re_root_misses\":" << r.re_root_misses
     << ",\"" << prefix << "_kept_despite_legacy_gate\":"
     << r.kept_despite_legacy_gate
     << ",\"" << prefix << "_target_argmax_overrides\":"
     << r.target_argmax_overrides;
  const bool has = r.rows_emitted > 0;
  const double d = has ? static_cast<double>(r.rows_emitted) : 1.0;
  auto mean = [&](double sum) -> std::string {
    return has ? F17(sum / d) : std::string("null");
  };
  os << ",\"" << prefix << "_mean_target_entropy_norm\":"
     << mean(r.sum_target_entropy_norm)
     << ",\"" << prefix << "_mean_raw_entropy_norm\":"
     << mean(r.sum_raw_entropy_norm)
     << ",\"" << prefix << "_mean_kl_target_given_raw\":"
     << mean(r.sum_kl_target_given_raw);
  for (int i = 0; i < kSearchPiEarlyExitCount; ++i) {
    os << ",\"" << prefix << "_early_exit_"
       << SearchPiEarlyExitName(static_cast<SearchPiEarlyExit>(i)) << "\":"
       << r.early_exit_counts[i];
  }
  // The wall-clock deadline and how close the worst root came to it. This is
  // the instrument that makes --threads auditable rather than hopeful.
  os << ",\"" << prefix << "_max_search_elapsed_ms\":"
     << F17(r.max_search_elapsed_ms)
     << ",\"" << prefix << "_mean_search_elapsed_ms\":"
     << (r.searches_run > 0
             ? F17(r.sum_search_elapsed_ms / static_cast<double>(r.searches_run))
             : std::string("null"))
     << ",\"" << prefix << "_configured_time_limit_ms\":"
     << F17(r.configured_time_limit_ms)
     << ",\"" << prefix << "_deadline_headroom_frac\":"
     << (r.configured_time_limit_ms > 0.0
             ? F17(1.0 - r.max_search_elapsed_ms / r.configured_time_limit_ms)
             : std::string("null"));

  const bool short_seen = r.first_short_episode_id >= 0;
  auto or_null = [&](int64_t v) -> std::string {
    return short_seen ? absl::StrCat(v) : std::string("null");
  };
  auto quoted = [](const std::string& s) {
    std::string out = "\"";
    for (char c : s) {
      if (c == '"' || c == '\\') out += '\\';
      out += c;
    }
    return out + "\"";
  };
  os << ",\"" << prefix << "_first_short_episode_id\":"
     << or_null(r.first_short_episode_id)
     << ",\"" << prefix << "_first_short_decision_id\":"
     << or_null(r.first_short_decision_id)
     << ",\"" << prefix << "_first_short_simulations_completed\":"
     << or_null(r.first_short_simulations_completed)
     << ",\"" << prefix << "_first_short_simulations_requested\":"
     << or_null(r.first_short_simulations_requested)
     << ",\"" << prefix << "_first_short_bot_reason\":"
     << (short_seen ? quoted(r.first_short_bot_reason) : std::string("null"))
     << ",\"" << prefix << "_first_short_session_reason\":"
     << (short_seen ? quoted(r.first_short_session_reason) : std::string("null"));
}

SearchPiConfig BuildConfig() {
  SearchPiConfig c;
  c.primary_simulations = absl::GetFlag(FLAGS_search_pi_primary_simulations);
  c.continuation_simulations =
      absl::GetFlag(FLAGS_search_pi_continuation_simulations);
  c.puct_c = absl::GetFlag(FLAGS_search_pi_puct_c);
  c.opponent_model_temperature = absl::GetFlag(FLAGS_search_pi_opponent_temperature);
  c.root_prior_temperature = absl::GetFlag(FLAGS_search_pi_root_prior_temperature);
  c.dirichlet_epsilon = absl::GetFlag(FLAGS_search_pi_dirichlet_epsilon);
  c.dirichlet_alpha_total = absl::GetFlag(FLAGS_search_pi_dirichlet_alpha_total);
  c.forced_playouts_k = absl::GetFlag(FLAGS_search_pi_forced_playouts_k);
  c.root_noise_fpu_zero = absl::GetFlag(FLAGS_search_pi_root_noise_fpu_zero);
  c.target_sharpen_exponent =
      absl::GetFlag(FLAGS_search_pi_target_sharpen_exponent);
  if (!ParseSearchPiContinuationTarget(
          absl::GetFlag(FLAGS_search_pi_continuation_target),
          &c.continuation_target)) {
    SpielFatalError("Unrecognized --search_pi_continuation_target: '" +
                    absl::GetFlag(FLAGS_search_pi_continuation_target) + "'");
  }
  c.behavior_temperature = absl::GetFlag(FLAGS_search_pi_behavior_temperature);
  c.non_search_temperature = absl::GetFlag(FLAGS_search_pi_non_search_temperature);
  c.searched_seat_unsearched_temperature =
      absl::GetFlag(FLAGS_search_pi_unsearched_role_temperature);
  c.seed_domain = absl::GetFlag(FLAGS_search_pi_seed_domain);
  // The scope axis (0 = narrow) and the runaway watchdog. Both default to the
  // historical values, so an unflagged BuildConfig is the rung-3a config.
  c.purchase_combat_budget =
      absl::GetFlag(FLAGS_search_pi_purchase_combat_budget);
  c.relative_time_budget_ms =
      absl::GetFlag(FLAGS_search_pi_relative_time_budget_ms);
  // The Leader teacher is never armed here. The lane's scope is agent-turn
  // decisions, and SearchPiGenerator's constructor fatals if this is true.
  // Widening purchase/combat/optional does NOT widen to leader picks.
  c.search_leader_draft = false;
  return c;
}

ArmResult RunArm(SearchPiArm arm, const std::shared_ptr<const Game>& game,
                 const std::shared_ptr<SharedDunePolicyValueNetImpl>& model,
                 torch::Device device) {
  const int games = absl::GetFlag(FLAGS_games);
  const int chunk = absl::GetFlag(FLAGS_chunk_games);
  const int64_t first = absl::GetFlag(FLAGS_first_episode_id);
  const int generation = absl::GetFlag(FLAGS_generation);
  const bool retain = absl::GetFlag(FLAGS_retain_rows);
  const int num_chunks = games / chunk;
  const auto wall_start = std::chrono::steady_clock::now();

  std::vector<SearchPiGenerationStats> chunk_stats(num_chunks);
  std::vector<std::vector<SearchPiRow>> chunk_rows(retain ? num_chunks : 0);
  std::atomic<int> next_chunk{0};
  std::mutex log_mutex;

  const int nthreads = std::max(1, std::min(absl::GetFlag(FLAGS_threads), num_chunks));

  // One evaluator per worker, all constructed HERE, on this thread, before any
  // worker starts. Following dune_search_benchmark: the evaluator holds no
  // mutable state, allocates its own input tensor per call and opens its own
  // InferenceMode region, so many of them may read one shared model, which is
  // loaded once and never mutated (that is also what makes the autocast weight
  // cache safe -- dune_network.h:313-316). Constructing them up front rather
  // than inside each worker avoids the one write the constructor does perform:
  // DuneNNEvaluator's ctor calls model_->eval() on the SHARED model, which would
  // otherwise race against another worker already inside forward().
  // --- Reference mode vs production batched mode ---------------------------
  //
  // batch_target == 0 keeps the historical arrangement EXACTLY: N independent
  // batch-1 DuneNNEvaluators, which is what reproduces the rung-2 hash chains.
  // batch_target > 0 replaces them with ONE shared asynchronous BatchedEvaluator
  // fronted by a per-worker BatchedNNEvaluator adapter -- so leaf inference from
  // DIFFERENT GAMES coalesces into one physical forward.
  //
  // Cross-game batching is the whole mechanism, and it is worth being precise
  // about why the wrapper alone would buy nothing: BatchedNNEvaluator already
  // submits 4 rows per leaf (one observation per player), but those 4 rows are
  // one leaf of ONE game and are already submitted together today. The speedup
  // comes from many GAMES' leaf groups being resident in the queue at once,
  // which is a property of nthreads > 1 sharing ONE batcher.
  //
  // Per-game search semantics are untouched either way: each worker still runs
  // sequential PUCT, one simulation at a time, and blocks on leaf evaluation
  // exactly where it previously blocked on a batch-1 forward. No virtual loss,
  // no delayed backup, no within-search leaf collection -- that would be design
  // (b), a different teacher.
  const int batch_target = absl::GetFlag(FLAGS_batch_target);
  const bool batched = batch_target > 0;
  const float logit_cap = static_cast<float>(absl::GetFlag(FLAGS_logit_cap));

  // Guards the shared model against the batcher's runner thread. Must outlive
  // every evaluator below, so it is declared before them.
  static std::shared_mutex model_mutex;

  // --- Registered warm-up, on a THROWAWAY evaluator ------------------------
  //
  // Destroyed before the scored evaluator exists. See --warmup_games: sharing
  // the scored batcher would advance its always-on counters before the first
  // scored row and break the Phase-2 B3/S6 crosschecks, making a correct
  // instrument look faulty. Warm-up episodes come from the reserved 900,000+
  // range, disjoint from every scored episode, mirroring Phase-2's
  // seed + 900000 convention.
  const int warmup_games = absl::GetFlag(FLAGS_warmup_games);
  // Validated HERE, before any GPU work, rather than left to fire inside the
  // generator. SearchPiGenerator asserts games_per_generation % 4 == 0 for seat
  // balance (dune_search_pi.cc:553), so an odd --warmup_games would abort the
  // run at the warm-up -- cheap here, but the same flag reaches a multi-hour
  // audit, and a fatal that arrives after the model has loaded reads like an
  // instrument fault rather than a typo.
  if (warmup_games % 4 != 0) {
    SpielFatalError(absl::StrCat(
        "--warmup_games must be a multiple of 4 for seat balance (got ",
        warmup_games, ")"));
  }
  if (warmup_games > 0) {
    torch::InferenceMode guard;
    std::shared_ptr<open_spiel::BatchedEvaluator> warm_batched;
    std::shared_ptr<algorithms::Evaluator> warm_eval;
    if (batched) {
      warm_batched = std::make_shared<open_spiel::BatchedEvaluator>(
          model, batch_target, absl::GetFlag(FLAGS_batcher_timeout_ms), device,
          &model_mutex, /*logit_cap=*/0.0f);
      warm_eval =
          std::make_shared<open_spiel::BatchedNNEvaluator>(warm_batched, logit_cap);
    } else {
      warm_eval = std::make_shared<DuneNNEvaluator>(model, device, logit_cap);
    }
    SearchPiConfig wcfg = BuildConfig();
    wcfg.games_per_generation = warmup_games;
    wcfg.next_episode_id = 900000;
    SearchPiGenerator wgen(wcfg);
    std::vector<SearchPiRow> wrows;
    SearchPiGenerationStats wstats;
    wgen.GenerateGeneration(generation, game, warm_eval, &wrows, &wstats, arm);
    // Explicit teardown BEFORE the scored evaluator is constructed.
    warm_eval.reset();
    warm_batched.reset();
    std::cout << "[audit] " << SearchPiArmName(arm) << " warm-up " << warmup_games
              << " discarded games on a throwaway evaluator" << std::endl;
  }

  // The shared batcher, constructed FRESH after warm-up so every counter it
  // reports starts at zero for the scored run.
  std::shared_ptr<open_spiel::BatchedEvaluator> batched_eval;
  if (batched) {
    batched_eval = std::make_shared<open_spiel::BatchedEvaluator>(
        model, batch_target, absl::GetFlag(FLAGS_batcher_timeout_ms), device,
        &model_mutex, /*logit_cap=*/0.0f);
    batched_eval->EnableBatcherTelemetry();
  }

  std::vector<std::shared_ptr<algorithms::Evaluator>> evaluators;
  evaluators.reserve(nthreads);
  for (int t = 0; t < nthreads; ++t) {
    if (batched) {
      evaluators.push_back(
          std::make_shared<open_spiel::BatchedNNEvaluator>(batched_eval, logit_cap));
    } else {
      evaluators.push_back(
          std::make_shared<DuneNNEvaluator>(model, device, logit_cap));
    }
  }

  auto worker = [&](int slot) {
    torch::InferenceMode guard;
    std::shared_ptr<algorithms::Evaluator> evaluator = evaluators[slot];
    for (;;) {
      const int ci = next_chunk.fetch_add(1);
      if (ci >= num_chunks) break;
      SearchPiConfig cfg = BuildConfig();
      cfg.games_per_generation = chunk;
      cfg.next_episode_id = first + static_cast<int64_t>(ci) * chunk;
      SearchPiGenerator gen(cfg);
      std::vector<SearchPiRow> rows;
      gen.GenerateGeneration(generation, game, evaluator, &rows,
                             &chunk_stats[ci], arm);
      if (retain) chunk_rows[ci] = std::move(rows);
      {
        std::lock_guard<std::mutex> lk(log_mutex);
        std::cout << "[audit] " << SearchPiArmName(arm) << " chunk " << (ci + 1)
                  << "/" << num_chunks << " episodes ["
                  << chunk_stats[ci].first_episode_id << ".."
                  << chunk_stats[ci].next_episode_id << ") rows="
                  << chunk_stats[ci].rows_total << " P="
                  << chunk_stats[ci].primary.simulations_completed << "/"
                  << chunk_stats[ci].primary.simulations_requested << " C="
                  << chunk_stats[ci].continuation.simulations_completed << "/"
                  << chunk_stats[ci].continuation.simulations_requested
                  << " in " << std::fixed << std::setprecision(1)
                  << chunk_stats[ci].collection_wall_time_s << "s" << std::endl;
      }
    }
  };

  std::vector<std::thread> pool;
  for (int t = 1; t < nthreads; ++t) pool.emplace_back(worker, t);
  worker(0);
  for (auto& th : pool) th.join();

  // --- Batcher telemetry ---------------------------------------------------
  //
  // Read ONCE after the workload, per GetBatcherTelemetry()'s contract. Its
  // absence on a batched arm is a Phase-2 S2 STOP, so the path is written
  // whenever one exists. Two self-consistency gates travel with it:
  //
  //   rows_identity_ok      -- every submitted row arrived either as a single
  //                            Evaluate() or inside an EvaluateBatch() group.
  //                            If this is false the telemetry is not
  //                            self-consistent and NO throughput claim may rest
  //                            on it (Phase-2 S7 rule 2).
  //   device_timing_complete -- device_timed_batches == physical_batches. A mean
  //                            over a SUBSET of batches reads exactly like a
  //                            mean over all of them (S6). Only meaningful on
  //                            CUDA; on CPU the device fields are legitimately
  //                            zero, so applicability is reported rather than
  //                            silently passing or silently failing.
  const std::string telemetry_path =
      absl::GetFlag(FLAGS_batcher_telemetry_json_path);
  if (batched && !telemetry_path.empty()) {
    const open_spiel::BatcherTelemetry t = batched_eval->GetBatcherTelemetry();
    const bool rows_identity_ok =
        t.submitted_rows == (t.single_row_calls + t.group_rows);
    const bool device_applicable = device.is_cuda();
    const bool device_timing_complete =
        !device_applicable || (t.device_timed_batches == t.physical_batches);
    std::ofstream o(telemetry_path);
    o << std::setprecision(10);
    o << "{\n";
    o << "  \"arm\": \"" << SearchPiArmName(arm) << "\",\n";
    o << "  \"submitted_rows\": " << t.submitted_rows << ",\n";
    o << "  \"physical_batches\": " << t.physical_batches << ",\n";
    o << "  \"mean_batch_size\": " << t.mean_batch_size << ",\n";
    o << "  \"p50_batch_size\": " << t.p50_batch_size << ",\n";
    o << "  \"p95_batch_size\": " << t.p95_batch_size << ",\n";
    o << "  \"max_batch_size\": " << t.max_batch_size << ",\n";
    o << "  \"target_batch_size\": " << t.target_batch_size << ",\n";
    o << "  \"timeout_ms\": " << t.timeout_ms << ",\n";
    o << "  \"target_occupancy\": " << t.target_occupancy << ",\n";
    o << "  \"timeout_flush_batches\": " << t.timeout_flush_batches << ",\n";
    o << "  \"single_row_calls\": " << t.single_row_calls << ",\n";
    o << "  \"group_calls\": " << t.group_calls << ",\n";
    o << "  \"group_rows\": " << t.group_rows << ",\n";
    o << "  \"leaf_groups_split\": " << t.leaf_groups_split << ",\n";
    o << "  \"single_wait_ms_mean\": " << t.single_wait_ms_mean << ",\n";
    o << "  \"single_wait_ms_max\": " << t.single_wait_ms_max << ",\n";
    o << "  \"group_wait_ms_mean\": " << t.group_wait_ms_mean << ",\n";
    o << "  \"group_wait_ms_max\": " << t.group_wait_ms_max << ",\n";
    o << "  \"h2d_ms\": " << t.h2d_ms << ",\n";
    o << "  \"forward_ms\": " << t.forward_ms << ",\n";
    o << "  \"output_cast_ms\": " << t.output_cast_ms << ",\n";
    o << "  \"d2h_ms\": " << t.d2h_ms << ",\n";
    o << "  \"sync_ms\": " << t.sync_ms << ",\n";
    o << "  \"device_timed_batches\": " << t.device_timed_batches << ",\n";
    o << "  \"device_timing_applicable\": "
      << (device_applicable ? "true" : "false") << ",\n";
    o << "  \"device_timing_complete\": "
      << (device_timing_complete ? "true" : "false") << ",\n";
    o << "  \"rows_identity_ok\": " << (rows_identity_ok ? "true" : "false")
      << "\n";
    o << "}\n";
    o.close();
    std::cout << "[audit] " << SearchPiArmName(arm) << " batcher: rows="
              << t.submitted_rows << " batches=" << t.physical_batches
              << " mean_batch=" << std::fixed << std::setprecision(2)
              << t.mean_batch_size << " occupancy=" << t.target_occupancy
              << " rows_identity_ok=" << (rows_identity_ok ? "1" : "0")
              << " device_timing_complete=" << (device_timing_complete ? "1" : "0")
              << std::endl;
  }

  ArmResult r;
  r.arm = arm;
  for (int ci = 0; ci < num_chunks; ++ci) {
    const SearchPiGenerationStats& s = chunk_stats[ci];
    MergeRole(&r.primary, s.primary);
    MergeRole(&r.continuation, s.continuation);
    for (int i = 0; i < 7; ++i) {
      r.simulations_by_role[i] += s.simulations_by_role[i];
      r.decisions_by_role[i] += s.decisions_by_role[i];
    }
    r.leader_rows_emitted += s.leader_rows_emitted;
    r.inference_calls += s.inference_calls;
    r.rows_total += s.rows_total;
    for (const SearchPiGameOutcome& g : s.games_played) r.games.push_back(g);
  }
  if (retain) {
    // Recomputed over the concatenation in episode order rather than composed
    // from per-chunk chains: a rolling hash does not compose, and the value
    // that has to be reproduced is the one a single undivided generation would
    // have produced.
    std::string tc, ec;
    for (int ci = 0; ci < num_chunks; ++ci) {
      for (const SearchPiRow& row : chunk_rows[ci]) {
        tc = ChainSearchPiTargetHash(tc, row);
        ec = ChainSearchPiExtendedRowHash(ec, row);
      }
    }
    r.target_hash_chain = tc;
    r.extended_hash_chain = ec;
  }
  r.wall_time_s =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - wall_start)
          .count();
  return r;
}

// Returns false if anything could not be written. An instrument that reports
// success while having produced no artifact is the failure mode this lane's
// rules exist to prevent: a mistyped --output_dir used to give zero files, no
// diagnostic and exit 0, and a PARITY OK line on top of it.
bool WriteArm(const ArmResult& r, const std::string& dir) {
  if (dir.empty()) {
    std::cerr << "[audit] --output_dir is empty: nothing was written."
              << std::endl;
    return false;
  }
  std::error_code ec;
  std::filesystem::create_directories(dir, ec);
  if (!std::filesystem::is_directory(dir)) {
    std::cerr << "[audit] --output_dir is not a directory and could not be "
                 "created: " << dir << " (" << ec.message() << ")" << std::endl;
    return false;
  }
  const std::string name = SearchPiArmName(r.arm);

  {
    const std::string p = dir + "/" + name + "_games.jsonl";
    std::ofstream os(p, std::ios::trunc);
    if (!os) {
      std::cerr << "[audit] cannot open " << p << std::endl;
      return false;
    }
    for (const SearchPiGameOutcome& g : r.games) {
      os << "{\"arm\":\"" << name << "\""
         << ",\"episode_id\":" << g.episode_id
         << ",\"searched_seat\":" << static_cast<int>(g.searched_seat)
         << ",\"placement\":" << g.searched_seat_placement
         << ",\"return\":" << F17(g.searched_seat_return) << ",\"returns\":[";
      for (size_t i = 0; i < g.returns.size(); ++i) {
        os << (i ? "," : "") << F17(g.returns[i]);
      }
      os << "]"
         << ",\"searched_seat_decisions\":" << g.searched_seat_decisions
         << ",\"rows_emitted\":" << g.rows_emitted
         << ",\"primary_roots\":" << g.primary_roots
         << ",\"continuation_roots\":" << g.continuation_roots
         << ",\"primary_simulations\":" << g.primary_simulations
         << ",\"continuation_simulations\":" << g.continuation_simulations
         << ",\"fallbacks\":" << g.fallbacks
         << ",\"searches_short_of_budget\":" << g.searches_short_of_budget
         << ",\"off_scope_simulations\":" << g.off_scope_simulations
         // Additive game-shape telemetry (Fable, 2026-08-18). final_round is
         // the round ACTUALLY PLAYED (the engine's counter has already advanced
         // past it at kTerminal); final_vp is FinalScoredVp, the expression
         // Returns() ranks on, NOT the reporting helper that overstates by 1
         // without tech tile 8. Both derivations are argued at
         // SearchPiGameOutcome's declaration.
         << ",\"final_round\":" << g.final_round << ",\"final_vp\":[";
      for (size_t i = 0; i < g.final_vp.size(); ++i) {
        os << (i ? "," : "") << g.final_vp[i];
      }
      os << "]}\n";
    }
    os.flush();
    if (!os) {
      std::cerr << "[audit] write failed for " << p << std::endl;
      return false;
    }
  }

  int64_t place[5] = {0, 0, 0, 0, 0};
  for (const SearchPiGameOutcome& g : r.games) {
    place[g.searched_seat_placement]++;  // index 0 collects off-ladder returns
  }
  const double n = r.games.empty() ? 1.0 : static_cast<double>(r.games.size());

  const std::string sp = dir + "/" + name + "_summary.json";
  std::ofstream os(sp, std::ios::trunc);
  if (!os) {
    std::cerr << "[audit] cannot open " << sp << std::endl;
    return false;
  }
  const SearchPiConfig c = BuildConfig();
  os << "{\"arm\":\"" << name << "\""
     << ",\"model_checkpoint\":\"" << absl::GetFlag(FLAGS_model_checkpoint) << "\""
     << ",\"games\":" << r.games.size()
     << ",\"first_episode_id\":" << absl::GetFlag(FLAGS_first_episode_id)
     << ",\"generation\":" << absl::GetFlag(FLAGS_generation)
     << ",\"threads\":" << absl::GetFlag(FLAGS_threads)
     << ",\"chunk_games\":" << absl::GetFlag(FLAGS_chunk_games)
     << ",\"wall_time_s\":" << F17(r.wall_time_s)
     << ",\"seconds_per_game\":" << F17(r.wall_time_s / n)
     << ",\"rows_total\":" << r.rows_total
     << ",\"inference_calls\":" << r.inference_calls
     << ",\"leader_rows_emitted\":" << r.leader_rows_emitted
     // Placement counts. `placement_offladder` is a return that matched no rung
     // of {+2.25, +0.25, -0.75, -1.75}; it is reported, never snapped.
     << ",\"placement_1st\":" << place[1] << ",\"placement_2nd\":" << place[2]
     << ",\"placement_3rd\":" << place[3] << ",\"placement_4th\":" << place[4]
     << ",\"placement_offladder\":" << place[0]
     << ",\"first_place_rate\":" << F17(static_cast<double>(place[1]) / n);
  EmitRoleJson(os, "primary", r.primary);
  EmitRoleJson(os, "continuation", r.continuation);
  for (int i = 0; i < 7; ++i) {
    os << ",\"sims_role_" << i << "\":" << r.simulations_by_role[i]
       << ",\"decisions_role_" << i << "\":" << r.decisions_by_role[i];
  }
  // The configuration echoed back into the artifact, so a summary is
  // self-describing and a manipulation check reads the run's own claim about
  // itself rather than a launcher's.
  os << ",\"cfg_primary_simulations\":" << c.primary_simulations
     << ",\"cfg_continuation_simulations\":" << c.continuation_simulations
     << ",\"cfg_puct_c\":" << F17(c.puct_c)
     << ",\"cfg_root_prior_temperature\":" << F17(c.root_prior_temperature)
     << ",\"cfg_dirichlet_epsilon\":" << F17(c.dirichlet_epsilon)
     << ",\"cfg_forced_playouts_k\":" << F17(c.forced_playouts_k)
     << ",\"cfg_target_sharpen_exponent\":" << F17(c.target_sharpen_exponent)
     << ",\"cfg_behavior_temperature\":" << F17(c.behavior_temperature)
     << ",\"cfg_non_search_temperature\":" << F17(c.non_search_temperature)
     << ",\"cfg_unsearched_role_temperature\":"
     << F17(c.searched_seat_unsearched_temperature)
     << ",\"cfg_opponent_temperature\":" << F17(c.opponent_model_temperature)
     << ",\"cfg_continuation_target\":\""
     << SearchPiContinuationTargetName(c.continuation_target) << "\""
     << ",\"cfg_search_leader_draft\":"
     << (c.search_leader_draft ? "true" : "false")
     << ",\"cfg_seed_domain\":" << c.seed_domain
     << ",\"cfg_logit_cap\":" << F17(absl::GetFlag(FLAGS_logit_cap))
     << ",\"cfg_hidden_dim\":" << absl::GetFlag(FLAGS_hidden_dim)
     << ",\"cfg_num_blocks\":" << absl::GetFlag(FLAGS_num_blocks)
     // The scope axis and the watchdog. These two decide WHICH TEACHER this
     // arm is -- narrow (0) or wide (64) -- so the three-arm analyzer must be
     // able to read the arm's own claim about its scope rather than infer it
     // from sims_role_4/5/6 being nonzero. An inferred scope cannot tell a
     // correctly-configured wide arm from a narrow arm with a leaked budget.
     << ",\"cfg_purchase_combat_budget\":" << c.purchase_combat_budget
     << ",\"cfg_relative_time_budget_ms\":" << F17(c.relative_time_budget_ms)
     // Batching provenance. batch_target 0 is reference mode (the unbatched
     // path); >0 is production batched. Recorded so a run states which
     // inference mode produced it -- the teachers and the raw control must all
     // report the SAME mode, which is what §8.2 requires of the control.
     << ",\"cfg_batch_target\":" << absl::GetFlag(FLAGS_batch_target)
     << ",\"cfg_batcher_timeout_ms\":" << absl::GetFlag(FLAGS_batcher_timeout_ms)
     << ",\"cfg_warmup_games\":" << absl::GetFlag(FLAGS_warmup_games)
     << ",\"target_hash_chain\":"
     << (r.target_hash_chain.empty()
             ? std::string("null")
             : absl::StrCat("\"", r.target_hash_chain, "\""))
     << ",\"extended_hash_chain\":"
     << (r.extended_hash_chain.empty()
             ? std::string("null")
             : absl::StrCat("\"", r.extended_hash_chain, "\""))
     << "}\n";
  os.flush();
  if (!os) {
    std::cerr << "[audit] write failed for " << sp << std::endl;
    return false;
  }
  return true;
}

// Every check that decides whether an arm MEASURED what it was asked to. Run by
// the binary, not only by the downstream script: an invalid cell should not
// produce a clean exit code and a set of artifacts that look like a result.
bool ArmPassesManipulationCheck(const ArmResult& r) {
  bool ok = true;
  const std::string name = SearchPiArmName(r.arm);
  auto fail = [&](const std::string& msg) {
    std::cerr << "[audit] FAIL [" << name << "] " << msg << std::endl;
    ok = false;
  };

  // Scope, both arms: five of the seven roles run zero simulations
  // STRUCTURALLY, and the searched roles are indices 2 and 3.
  for (int i : {0, 1, 4, 5, 6}) {
    if (r.simulations_by_role[i] != 0) {
      fail(absl::StrCat("off-scope role ", i, " ran ", r.simulations_by_role[i],
                        " simulations"));
    }
  }
  if (r.leader_rows_emitted != 0) fail("Leader rows were emitted");
  for (const SearchPiGameOutcome& g : r.games) {
    if (g.off_scope_simulations != 0) {
      fail(absl::StrCat("episode ", g.episode_id, " ran ",
                        g.off_scope_simulations, " off-scope simulations"));
      break;
    }
    if (g.searched_seat_placement < 1 || g.searched_seat_placement > 4) {
      fail(absl::StrCat("episode ", g.episode_id, " return ",
                        g.searched_seat_return,
                        " sits on no rung of the terminal ladder"));
      break;
    }
  }

  if (r.arm == SearchPiArm::kRawArgmax) {
    // Read simulations_by_role[2] and [3] -- the SEARCHED role indices -- and
    // not the role-stats counters. Those counters are only ever incremented
    // inside the branch this arm skips, so they are structurally zero here and
    // would keep reading zero even if kPolicyOnly stopped short-circuiting and
    // the control started spending real simulations. A check that cannot fail
    // is not a check.
    const int64_t searched_role_sims =
        r.simulations_by_role[2] + r.simulations_by_role[3];
    if (searched_role_sims != 0) {
      fail(absl::StrCat("the raw control ran ", searched_role_sims,
                        " simulations at the searched roles"));
    }
    if (r.rows_total != 0) {
      fail(absl::StrCat("the raw control emitted ", r.rows_total, " rows"));
    }
    if (r.primary.roots_seen <= 0 || r.continuation.roots_seen <= 0) {
      // A control that never reached a continuation lost the session, and role
      // classification with it -- at which point it is not a matched control.
      fail(absl::StrCat("reached ", r.primary.roots_seen, " primary and ",
                        r.continuation.roots_seen,
                        " continuation roots; both must be positive"));
    }
    return ok;
  }

  // --- The searched arm ---
  if (r.primary.fallbacks + r.continuation.fallbacks != 0) {
    fail(absl::StrCat(r.primary.fallbacks + r.continuation.fallbacks,
                      " technical fallbacks"));
  }
  const int prim = absl::GetFlag(FLAGS_search_pi_primary_simulations);
  const int cont = absl::GetFlag(FLAGS_search_pi_continuation_simulations);
  const std::pair<const SearchPiRoleStats*, int> roles[2] = {
      {&r.primary, prim}, {&r.continuation, cont}};
  const char* role_names[2] = {"primary", "continuation"};
  for (int i = 0; i < 2; ++i) {
    const SearchPiRoleStats& rs = *roles[i].first;
    const int64_t want = rs.roots_seen * roles[i].second;
    if (rs.roots_seen <= 0) fail(absl::StrCat("no ", role_names[i], " roots"));
    if (rs.simulations_requested != want) {
      fail(absl::StrCat(role_names[i], ": the session requested ",
                        rs.simulations_requested, " simulations, not ",
                        rs.roots_seen, "x", roles[i].second, " = ", want));
    }
    for (int k = 0; k < kSearchPiEarlyExitCount; ++k) {
      if (static_cast<SearchPiEarlyExit>(k) == SearchPiEarlyExit::kNone) continue;
      if (rs.early_exit_counts[k] != 0) {
        fail(absl::StrCat(
            role_names[i], ": ", rs.early_exit_counts[k], " '",
            SearchPiEarlyExitName(static_cast<SearchPiEarlyExit>(k)),
            "' early exits"));
      }
    }
    if (rs.searches_short_of_budget != 0) {
      // The one-sided contamination. Truncation costs the searched arm strength
      // and cannot touch the control, so it biases the headline toward "no
      // teacher edge" -- the direction a reader is least likely to question.
      std::cerr << "[audit] SHORT SEARCHES [" << name << "] " << role_names[i]
                << ": " << rs.searches_short_of_budget << " roots, "
                << rs.simulation_shortfall << " simulations missing. First at "
                << "episode " << rs.first_short_episode_id << " decision "
                << rs.first_short_decision_id << " ("
                << rs.first_short_simulations_completed << "/"
                << rs.first_short_simulations_requested << ", bot_reason='"
                << rs.first_short_bot_reason << "' session_reason='"
                << rs.first_short_session_reason << "'). Worst search took "
                << rs.max_search_elapsed_ms << " ms against a "
                << rs.configured_time_limit_ms << " ms deadline." << std::endl;
      if (absl::GetFlag(FLAGS_fail_on_short_search)) {
        fail(absl::StrCat(role_names[i], ": ", rs.searches_short_of_budget,
                          " searches short of budget (see above). This cell is "
                          "INVALID, not weak: truncation weakens only the arm "
                          "that searches."));
      }
    }
  }
  return ok;
}

}  // namespace
}  // namespace open_spiel

int main(int argc, char** argv) {
  // Every global torch setting below mirrors dune_ppo_train's main() line for
  // line. They are not incidental: DuneNNEvaluator opens an AutocastGuard on
  // CUDA, so the trainer's TF32 and workspace settings are part of what the
  // frozen network's prior actually is on this box. A mirror that drifts here
  // would move the priors, and the parity smoke would fail for a reason that
  // has nothing to do with the controller.
  setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8", 1);
  absl::ParseCommandLine(argc, argv);
  if (absl::GetFlag(FLAGS_deterministic)) {
    at::globalContext().setDeterministicAlgorithms(true, /*silent=*/true);
  }
  using namespace open_spiel;
  at::set_num_threads(1);
  at::set_num_interop_threads(1);

  const uint64_t master = static_cast<uint64_t>(absl::GetFlag(FLAGS_seed));
  torch::manual_seed(dune_seed::DeriveSeed(master, dune_seed::kStreamModelInit));

  const std::string ckpt = absl::GetFlag(FLAGS_model_checkpoint);
  if (ckpt.empty()) {
    std::cerr << "--model_checkpoint is required." << std::endl;
    return 2;
  }
  if (absl::GetFlag(FLAGS_search_pi_seed_domain) == 0) {
    std::cerr << "--search_pi_seed_domain must be set nonzero (no silent "
                 "default: the seed domain IS the episode stream)." << std::endl;
    return 2;
  }
  const int games = absl::GetFlag(FLAGS_games);
  const int chunk = absl::GetFlag(FLAGS_chunk_games);
  if (games <= 0 || games % 4 != 0) {
    std::cerr << "--games must be a positive multiple of 4 (seat balance)."
              << std::endl;
    return 2;
  }
  if (chunk <= 0 || chunk % 4 != 0 || games % chunk != 0) {
    std::cerr << "--chunk_games must be a positive multiple of 4 that divides "
                 "--games, so every work unit is seat balanced and the chunks "
                 "tile the episode range exactly." << std::endl;
    return 2;
  }
  const std::string arm_flag = absl::GetFlag(FLAGS_arm);
  if (arm_flag != "searched" && arm_flag != "raw_argmax" && arm_flag != "both") {
    std::cerr << "--arm must be searched | raw_argmax | both." << std::endl;
    return 2;
  }

  auto game = open_spiel::LoadGame("dune_imperium");
  const int64_t obs_size = game->GetType().provides_information_state_tensor
                               ? game->InformationStateTensorSize()
                               : game->ObservationTensorSize();
  const int64_t action_size = game->NumDistinctActions();

  torch::Device device(torch::kCPU);
  if (torch::cuda::is_available()) {
    device = torch::Device(torch::kCUDA);
    at::globalContext().setAllowTF32CuBLAS(true);
    at::globalContext().setAllowTF32CuDNN(true);
    at::autocast::set_autocast_dtype(at::kCUDA, at::ScalarType::BFloat16);
  }

  // Constructed WITHOUT the PWO-5 auxiliary heads, exactly as the trainer's
  // inference model is. Module::load reads the module's OWN keys, so a
  // checkpoint that carries the three heads loads here with them simply never
  // visited -- which is why the u15828 archive loads into a head-less model.
  auto model = std::make_shared<SharedDunePolicyValueNetImpl>(
      obs_size, absl::GetFlag(FLAGS_hidden_dim), action_size,
      absl::GetFlag(FLAGS_num_blocks),
      absl::GetFlag(FLAGS_nonlinear_value_head));
  model->to(device);
  try {
    torch::load(model, ckpt, device);
  } catch (const c10::Error& e) {
    std::cerr << "Failed to load " << ckpt << ":\n" << e.msg() << std::endl;
    return 2;
  }
  model->to(device);
  model->eval();

  std::cout << "[audit] model=" << ckpt << " device="
            << (device.is_cuda() ? "cuda" : "cpu") << " obs=" << obs_size
            << " games=" << games << " chunk=" << chunk
            << " threads=" << absl::GetFlag(FLAGS_threads) << " episodes=["
            << absl::GetFlag(FLAGS_first_episode_id) << ".."
            << absl::GetFlag(FLAGS_first_episode_id) + games << ")"
            << " generation=" << absl::GetFlag(FLAGS_generation)
            << " seed_domain=" << absl::GetFlag(FLAGS_search_pi_seed_domain)
            << std::endl;

  const std::string dir = absl::GetFlag(FLAGS_output_dir);
  int rc = 0;

  // The searched arm runs FIRST under --arm=both: it is the expensive one and
  // the one a partial run most wants to have finished.
  if (arm_flag == "searched" || arm_flag == "both") {
    ArmResult r = RunArm(SearchPiArm::kSearched, game, model, device);
    if (!WriteArm(r, dir)) rc = 1;
    if (!ArmPassesManipulationCheck(r)) rc = 1;
    // The deadline margin, printed whether or not it fired. A run that spent
    // its budget with 200 ms to spare and one that spent it with 7 s to spare
    // report the same zero shortfall, and only the second licenses raising
    // --threads.
    std::cout << "[audit] DEADLINE primary worst " << std::fixed
              << std::setprecision(0) << r.primary.max_search_elapsed_ms
              << " ms / continuation worst "
              << r.continuation.max_search_elapsed_ms << " ms against a "
              << r.primary.configured_time_limit_ms << " ms limit ("
              << std::setprecision(1)
              << (r.primary.configured_time_limit_ms > 0.0
                      ? 100.0 * (1.0 - r.primary.max_search_elapsed_ms /
                                           r.primary.configured_time_limit_ms)
                      : 0.0)
              << "% headroom on the worst primary)" << std::endl;
    std::cout << "[audit] SEARCHED first=" << r.games.size() << " games "
              << "1st=" << [&] {
                   int64_t f = 0;
                   for (const auto& g : r.games) f += (g.searched_seat_placement == 1);
                   return f;
                 }()
              << " P_sims=" << r.primary.simulations_completed << "/"
              << r.primary.simulations_requested
              << " C_sims=" << r.continuation.simulations_completed << "/"
              << r.continuation.simulations_requested
              << " fallbacks=" << (r.primary.fallbacks + r.continuation.fallbacks)
              << " short=" << (r.primary.searches_short_of_budget +
                               r.continuation.searches_short_of_budget)
              << " wall=" << std::fixed << std::setprecision(1) << r.wall_time_s
              << "s (" << (r.wall_time_s / std::max<size_t>(1, r.games.size()))
              << " s/game)" << std::endl;
    if (!r.extended_hash_chain.empty()) {
      std::cout << "[audit] target_hash_chain   " << r.target_hash_chain
                << "\n[audit] extended_hash_chain " << r.extended_hash_chain
                << std::endl;
    }
    // The smoke's verdict, decided by the binary rather than by an operator
    // reading two hex strings side by side.
    const std::string want_ext = absl::GetFlag(FLAGS_expect_extended_chain);
    const std::string want_tgt = absl::GetFlag(FLAGS_expect_target_chain);
    if (!want_ext.empty() || !want_tgt.empty()) {
      if (!absl::GetFlag(FLAGS_retain_rows)) {
        std::cerr << "[audit] PARITY FAIL: a chain was expected but "
                     "--retain_rows is false, so none was computed." << std::endl;
        rc = 1;
      }
      if (!want_ext.empty() && want_ext != r.extended_hash_chain) {
        std::cerr << "[audit] PARITY FAIL extended_hash_chain\n  expected "
                  << want_ext << "\n  got      " << r.extended_hash_chain
                  << std::endl;
        rc = 1;
      }
      if (!want_tgt.empty() && want_tgt != r.target_hash_chain) {
        std::cerr << "[audit] PARITY FAIL target_hash_chain\n  expected "
                  << want_tgt << "\n  got      " << r.target_hash_chain
                  << std::endl;
        rc = 1;
      }
      if (rc == 0) std::cout << "[audit] PARITY OK: chains reproduced." << std::endl;
    }
  }

  if (arm_flag == "raw_argmax" || arm_flag == "both") {
    ArmResult r = RunArm(SearchPiArm::kRawArgmax, game, model, device);
    if (!WriteArm(r, dir)) rc = 1;
    if (!ArmPassesManipulationCheck(r)) rc = 1;
    int64_t f = 0;
    for (const auto& g : r.games) f += (g.searched_seat_placement == 1);
    std::cout << "[audit] RAW first=" << r.games.size() << " games 1st=" << f
              << " P_roots=" << r.primary.roots_seen
              << " C_roots=" << r.continuation.roots_seen
              << " sims=" << (r.primary.simulations_completed +
                              r.continuation.simulations_completed)
              << " (must be 0) wall=" << std::fixed << std::setprecision(1)
              << r.wall_time_s << "s" << std::endl;
  }

  if (rc != 0) {
    std::cerr << "[audit] EXITING NONZERO. The artifacts on disk describe a "
                 "cell that did not measure what it was asked to; they are a "
                 "record of the failure, not a result." << std::endl;
  }
  return rc;
}
