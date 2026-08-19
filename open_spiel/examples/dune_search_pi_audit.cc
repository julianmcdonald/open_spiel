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
#include <cmath>
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
#include <utility>
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
#include "dune_search_pi_concurrent.h"
#include "dune_seed_utils.h"
#include "dune_sha256.h"

// --- What to measure -------------------------------------------------------
ABSL_FLAG(std::string, model_checkpoint, "",
          "Model whose teacher is audited. Loaded once, never written.");
// Rung 4A. With a policy-prior checkpoint set, the searched seat runs the
// HYBRID controller: MCTS policy priors come from this model while every leaf
// value still comes from --model_checkpoint. The three opponent seats and the
// searched seat's off-scope decisions stay on --model_checkpoint either way,
// so two audit runs that differ only in this flag are the exact paired
// searched-controller contrast the rung-4A gate needs, played against the same
// frozen opponent panel. Empty restores the single-model controller exactly.
ABSL_FLAG(std::string, search_pi_policy_prior_model_checkpoint, "",
          "Rung 4A candidate supplying MCTS policy priors. Empty means the "
          "audited model supplies both priors and leaf values, as before.");
ABSL_FLAG(std::string, search_pi_policy_prior_model_sha256, "",
          "Required whenever a policy-prior checkpoint is set. Verified "
          "against the file before any game is played.");
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
          "with this target row count, serving every worker. Must be a multiple "
          "of four and no greater than 4*actual_workers; that latter value is "
          "the physical capacity, not an implied operating target. Routing "
          "batch-1 through the unbatched path is the "
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
          "root runs under the wall-clock deadline set by "
          "--search_pi_relative_time_budget_ms, whose FLAG DEFAULT is the "
          "historical 10,000 ms; the batched-teacher audit registers 60,000 ms "
          "and its analyzer FAILS any cell whose artifact does not manifest "
          "that value (dune_puct_is_mcts.cc:982 breaks the simulation loop on "
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

using ArmResult = ConcurrentSearchPiCollectionResult;

// Every leaf group BatchedNNEvaluator submits is exactly NumPlayers rows wide:
// Evaluate() and PriorAndEvaluate() build one observation per seat and hand the
// lot to EvaluateBatch() (dune_batched_evaluator.h:28-33, :100-105). Asserted
// against the loaded game in main rather than trusted.
constexpr int64_t kBatchedLeafGroupRows = kSearchPiBatchedLeafGroupRows;

// --- Batcher telemetry self-consistency ------------------------------------
//
// Two predicates, each a registered STOP, each with its definition here
// rather than inline at a call site so the emitted JSON and the checks that
// read it cannot drift apart.

bool BatcherRowsIdentityOk(const open_spiel::BatcherTelemetry& t) {
  return SearchPiBatcherRowsIdentityOk(t);
}

// NO leaf_rows_crosscheck_ok. Read this before adding one back.
//
// Phase-2's leaf_rows_crosscheck_ok compares the driver's own
// `benchmark_raw_prior_calls` and `implied_opponent_prior_calls` against the
// submitted rows. Those are counters of the BENCHMARK DRIVER; they do not exist
// in this binary. The registration imported the STOP anyway, so it named a
// condition this instrument structurally could not report.
//
// A first attempt to give it a search-PI definition asserted
// `group_rows == group_calls * kBatchedLeafGroupRows`. THAT WAS A TAUTOLOGY and
// is withdrawn: dune_network.h:774-775 increments both counters two lines
// apart, inside one critical section, from the same `gsize`, and the only two
// EvaluateBatch callers reachable here both pass exactly NumPlayers
// observations. No input to this binary can falsify it. Worse, the failure it
// claimed to catch is outside its reach -- the Runner's split accounting
// (dune_network.h:1060-1082) touches only leaf_groups_split_,
// group_batches_seen_ and group_rows_left_, never group_calls_/group_rows_, so
// a split run and an unsplit run emit byte-identical values.
//
// Replacing an unreportable STOP with an unfailable one is worse than either.
// So: no STOP here; `leaf_groups_split` is emitted as the diagnostic that
// actually carries the signal; and the genuinely independent quantity -- the
// SEARCH's own inference count against the BATCHER's call count, two
// accumulators in different objects incremented by different code -- is emitted
// as a pair to be CHARACTERISED at the pre-measure. It is not registered as a
// STOP until it has been observed holding, because this lane does not register
// gates it has never seen bind.

bool BatcherDeviceTimingComplete(const open_spiel::BatcherTelemetry& t,
                                 bool device_applicable) {
  return SearchPiBatcherDeviceTimingComplete(t, device_applicable);
}

void WriteBatcherTelemetryJson(std::ostream& o, SearchPiArm arm,
                               const open_spiel::BatcherTelemetry& t,
                               bool device_applicable,
                               int64_t search_inference_calls,
                               int requested_threads, int actual_workers,
                               int configured_batch_target,
                               int max_inflight_rows) {
  o << std::setprecision(10);
  o << "{\n";
  o << "  \"arm\": \"" << SearchPiArmName(arm) << "\",\n";
  o << "  \"requested_threads\": " << requested_threads << ",\n";
  o << "  \"actual_workers\": " << actual_workers << ",\n";
  o << "  \"configured_batch_target\": " << configured_batch_target << ",\n";
  o << "  \"max_inflight_rows\": " << max_inflight_rows << ",\n";
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
  o << "  \"leaf_group_rows\": " << kBatchedLeafGroupRows << ",\n";
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
    << (BatcherDeviceTimingComplete(t, device_applicable) ? "true" : "false")
    << ",\n";
  o << "  \"rows_identity_ok\": "
    << (BatcherRowsIdentityOk(t) ? "true" : "false") << ",\n";
  // The independent cross-check, REPORTED AND NOT GATED (see the note above).
  // search_inference_calls comes from the MCTS's own counter, evaluator_calls
  // from the batcher's: different objects, different code.
  o << "  \"search_inference_calls\": " << search_inference_calls << ",\n";
  o << "  \"evaluator_calls\": " << (t.group_calls + t.single_row_calls)
    << ",\n";
  o << "  \"leaf_rows_crosscheck_ok\": null\n";
  o << "}\n";
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
     << ",\"" << prefix << "_p50_search_elapsed_ms\":"
     << (r.searches_run > 0
             ? F17(SearchPiLatencyQuantileUpperMs(r, 0.50))
             : std::string("null"))
     << ",\"" << prefix << "_p95_search_elapsed_ms\":"
     << (r.searches_run > 0
             ? F17(SearchPiLatencyQuantileUpperMs(r, 0.95))
             : std::string("null"))
     << ",\"" << prefix << "_mean_search_elapsed_ms\":"
     << (r.searches_run > 0
             ? F17(r.sum_search_elapsed_ms / static_cast<double>(r.searches_run))
             : std::string("null"))
     << ",\"" << prefix << "_configured_time_limit_ms\":"
     << F17(r.configured_time_limit_ms)
     << ",\"" << prefix << "_max_sim_duration_ms\":"
     << F17(r.max_sim_duration_ms)
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
                 torch::Device device,
                 const std::shared_ptr<SharedDunePolicyValueNetImpl>&
                     policy_prior_model = nullptr) {
  ConcurrentSearchPiCollectionConfig config;
  config.search = BuildConfig();
  config.search.next_episode_id = absl::GetFlag(FLAGS_first_episode_id);
  config.generation = absl::GetFlag(FLAGS_generation);
  config.collection_games = absl::GetFlag(FLAGS_games);
  config.chunk_games = absl::GetFlag(FLAGS_chunk_games);
  config.requested_workers = absl::GetFlag(FLAGS_threads);
  config.batch_target = absl::GetFlag(FLAGS_batch_target);
  config.batcher_timeout_ms = absl::GetFlag(FLAGS_batcher_timeout_ms);
  config.warmup_games = absl::GetFlag(FLAGS_warmup_games);
  config.retain_rows = absl::GetFlag(FLAGS_retain_rows);
  config.logit_cap = static_cast<float>(absl::GetFlag(FLAGS_logit_cap));
  config.output_dir = absl::GetFlag(FLAGS_output_dir);
  config.arm = arm;

  // A null policy-prior model is the single-model controller: the collection
  // path takes its historical branch and this audit reproduces exactly what it
  // audited before rung 4A existed.
  ArmResult result =
      CollectSearchPiConcurrent(config, game, model, device, policy_prior_model);

  // Preserve the legacy explicitly requested telemetry destination used by
  // older parity launchers. The per-arm sidecar written by WriteArm remains the
  // authoritative artifact.
  const std::string legacy_path =
      absl::GetFlag(FLAGS_batcher_telemetry_json_path);
  if (result.telemetry_valid && !legacy_path.empty()) {
    std::ofstream out(legacy_path, std::ios::trunc);
    if (out) {
      WriteBatcherTelemetryJson(
          out, arm, result.telemetry, result.telemetry_device_applicable,
          result.inference_calls, result.requested_workers,
          result.actual_workers, result.configured_batch_target,
          result.max_inflight_rows);
    }
  }
  return result;
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
         << ",\"watchdog_fallbacks\":" << g.watchdog_fallbacks
         << ",\"watchdog_timeouts\":" << g.watchdog_timeouts
         << ",\"non_timeout_early_exits\":" << g.non_timeout_early_exits
         << ",\"searches_short_of_budget\":" << g.searches_short_of_budget
         << ",\"off_scope_simulations\":" << g.off_scope_simulations
         // Per-episode budget accounting. The registration's exact invariant
         // has to be assertable over the RETAINED episodes, not over summary
         // totals that still contain the discarded ones -- otherwise a single
         // tolerated watchdog fire makes the totals unequal, the cell reads
         // INVALID, and the discard rule §9 registers can never be reached.
         << ",\"primary_simulations_requested\":"
         << g.primary_simulations_requested
         << ",\"continuation_simulations_requested\":"
         << g.continuation_simulations_requested
         // The wide teacher's three roles, no longer folded into continuation.
         << ",\"wide_roots\":" << g.wide_roots
         << ",\"wide_simulations\":" << g.wide_simulations
         << ",\"wide_simulations_requested\":" << g.wide_simulations_requested
         // The played-action manipulation check: roots where the action
         // actually applied to the state differs from the raw-prior argmax the
         // matched control would have played. A wide arm reading zero here
         // played the control's moves -- the 2026-08-18 defect, which every
         // other counter in this artifact survived.
         << ",\"roots_played_differs_from_raw\":[";
      for (int rr = 0; rr < 7; ++rr) {
        os << (rr ? "," : "") << g.roots_played_differs_from_raw[rr];
      }
      os << "]"
         // Trajectory identity. A divergence detector, NOT a parity reference.
         << ",\"played_action_digest\":\"" << std::hex << std::setw(16)
         << std::setfill('0') << g.played_action_digest << std::dec
         << std::setfill(' ') << "\""
         << ",\"full_trajectory_digest\":\"" << std::hex << std::setw(16)
         << std::setfill('0') << g.full_trajectory_digest << std::dec
         << std::setfill(' ') << "\""
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
  int64_t differs_by_role[7] = {0, 0, 0, 0, 0, 0, 0};
  for (const SearchPiGameOutcome& g : r.games) {
    place[g.searched_seat_placement]++;  // index 0 collects off-ladder returns
    for (int rr = 0; rr < 7; ++rr) {
      differs_by_role[rr] += g.roots_played_differs_from_raw[rr];
    }
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
     << ",\"requested_threads\":" << r.requested_workers
     << ",\"actual_workers\":" << r.actual_workers
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
     << ",\"first_place_rate\":" << F17(static_cast<double>(place[1]) / n)
     // The played-action manipulation check, aggregated. Reported for EVERY
     // arm: the raw controls must read exactly zero (they play the raw-prior
     // argmax at their searched roles by construction, so a nonzero count is a
     // defect in the check or in the control), and a searched arm reading zero
     // at a role class it is registered to search played the control's moves.
     << ",\"played_differs_from_raw_total\":"
     << (differs_by_role[2] + differs_by_role[3] + differs_by_role[4]
         + differs_by_role[5] + differs_by_role[6]);
  for (int rr = 0; rr < 7; ++rr) {
    os << ",\"played_differs_role_" << rr << "\":" << differs_by_role[rr];
  }
  os << "";
  EmitRoleJson(os, "primary", r.primary);
  EmitRoleJson(os, "continuation", r.continuation);
  // The wide teacher's three roles, emitted for EVERY arm. All-zero blocks in a
  // narrow arm are the evidence that narrow searched nothing there; without
  // them the analyzer could only infer scope from sims_role_N, which cannot
  // distinguish a correctly-narrow arm from a wide arm whose searches silently
  // did nothing -- the exact defect the 2026-08-18 review found.
  EmitRoleJson(os, "purchase", r.purchase);
  EmitRoleJson(os, "combat_intrigue", r.combat_intrigue);
  EmitRoleJson(os, "other_optional", r.other_optional);
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
     << ",\"cfg_batch_target\":" << r.configured_batch_target
     << ",\"max_inflight_rows\":" << r.max_inflight_rows
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

  // Stable gate-only row representation. Large audits run with
  // --retain_rows=false, so this file never burdens the registered n=2,344
  // endpoint. Gates 3 and 4 retain >=64 games and join rows by the exact
  // pre-decision serialized-state SHA-256 (plus episode/player/role), never by
  // ordinal decision id after trajectories have diverged.
  if (!r.retained_rows.empty()) {
    const std::string rp = dir + "/" + name + "_rows.jsonl";
    std::ofstream ro(rp, std::ios::trunc);
    if (!ro) {
      std::cerr << "[audit] cannot open " << rp << std::endl;
      return false;
    }
    for (const SearchPiRow& row : r.retained_rows) {
      ro << "{\"episode_id\":" << row.episode_id
         << ",\"decision_id\":" << row.decision_id
         << ",\"state_fingerprint\":\"" << row.state_fingerprint << "\""
         << ",\"player\":" << row.player
         << ",\"role\":" << static_cast<int>(row.role)
         << ",\"chosen_action\":" << row.chosen_action
         << ",\"legal_actions\":[";
      for (size_t i = 0; i < row.legal_actions.size(); ++i) {
        ro << (i ? "," : "") << row.legal_actions[i];
      }
      ro << "],\"target_probs\":[";
      for (size_t i = 0; i < row.target_probs.size(); ++i) {
        ro << (i ? "," : "") << F17(row.target_probs[i]);
      }
      ro << "]}\n";
    }
    ro.flush();
    if (!ro) {
      std::cerr << "[audit] write failed for " << rp << std::endl;
      return false;
    }
  }

  // Batcher telemetry, PER ARM, under the same stem as every other artifact.
  // Its absence on a batched arm is a registered STOP, so it is written where
  // the analyzer looks rather than wherever a launcher flag happened to point.
  //
  // Written LAST on purpose. It used to be written first, so a full disk or a
  // permission problem on this one path returned false BEFORE games.jsonl and
  // summary.json were opened -- discarding a completed multi-hour arm's entire
  // per-episode record to protect a sidecar. The sidecar is a STOP; it is not
  // worth the data.
  if (r.telemetry_valid) {
    const std::string tp = dir + "/" + name + "_batcher_telemetry.json";
    std::ofstream ot(tp, std::ios::trunc);
    if (!ot) {
      std::cerr << "[audit] cannot open " << tp << std::endl;
      return false;
    }
    WriteBatcherTelemetryJson(ot, r.arm, r.telemetry,
                              r.telemetry_device_applicable, r.inference_calls,
                              r.requested_workers, r.actual_workers,
                              r.configured_batch_target,
                              r.max_inflight_rows);
    ot.flush();
    if (!ot) {
      std::cerr << "[audit] write failed for " << tp << std::endl;
      return false;
    }
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

  // Scope, ARM-CONDITIONAL. Which roles may legitimately run simulations is a
  // property of the teacher under test, so a single hard-coded set cannot gate
  // both. Narrow searches {2,3} and must be structurally zero at {0,1,4,5,6};
  // wide searches {2,3,4,5,6} and must be structurally zero at {0,1}. Leader
  // selection and forced/bookkeeping are never searched by either.
  //
  // The previous unconditional {0,1,4,5,6} check would have hard-failed EVERY
  // wide run -- the arm is registered to search exactly the roles it forbade.
  const int pcb = absl::GetFlag(FLAGS_search_pi_purchase_combat_budget);
  const bool wide_scope = pcb > 0;
  for (int i : (wide_scope ? std::vector<int>{0, 1}
                           : std::vector<int>{0, 1, 4, 5, 6})) {
    if (r.simulations_by_role[i] != 0) {
      fail(absl::StrCat("off-scope role ", i, " ran ", r.simulations_by_role[i],
                        " simulations"));
    }
  }
  // And the converse, which is the check that would have caught the defect the
  // review found: a wide arm that runs ZERO simulations at a role it is
  // registered to search is a null arm wearing a wide label. Only asserted for
  // the searched arm -- the wide-matched raw control legitimately runs zero
  // everywhere, and is checked by the arm-consistency gate instead.
  if (wide_scope && r.arm == SearchPiArm::kSearched) {
    for (int i : {4, 5, 6}) {
      if (r.decisions_by_role[i] > 0 && r.simulations_by_role[i] == 0) {
        fail(absl::StrCat("wide arm saw ", r.decisions_by_role[i], " role-", i,
                          " decisions and ran ZERO simulations at it"));
      }
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
    // Read simulations_by_role[] -- the ROLE INDICES -- and not the role-stats
    // counters. Those counters are only ever incremented inside the branch this
    // arm skips, so they are structurally zero here and would keep reading zero
    // even if kPolicyOnly stopped short-circuiting and the control started
    // spending real simulations. A check that cannot fail is not a check.
    //
    // ALL SEVEN roles, not just {2,3}. The wide-matched control (raw at
    // purchase_combat_budget = 64) reaches roles 4/5/6 as SEARCH roles, so the
    // {2,3}-only check left the exact three roles that distinguish it from the
    // narrow-matched control ungated. Fable's ruling is explicit: the
    // arm-consistency gate covers both controls at zero simulations at EVERY
    // role.
    int64_t control_sims = 0;
    for (int i = 0; i < 7; ++i) control_sims += r.simulations_by_role[i];
    if (control_sims != 0) {
      fail(absl::StrCat("the raw control ran ", control_sims,
                        " simulations (must be zero at every role)"));
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
    // A control PLAYS the raw-prior argmax at every role it treats as searched,
    // so any divergence from it is a defect in the control or in the check.
    // Asserted rather than assumed, because this same counter is the floor the
    // wide teacher has to clear and a check that is only ever read in one
    // direction is half a check.
    int64_t differs = 0;
    for (const SearchPiGameOutcome& g : r.games) {
      for (int rr = 0; rr < 7; ++rr) differs += g.roots_played_differs_from_raw[rr];
    }
    if (differs != 0) {
      fail(absl::StrCat("the raw control played something other than the "
                        "raw-prior argmax at ", differs, " searched roots"));
    }
    return ok;
  }

  // --- The searched arm ---
  //
  // Every SEARCHED role bucket, arm-conditionally: {primary, continuation} for
  // narrow, plus {purchase, combat_intrigue, other_optional} for wide. The
  // buckets were added by the wide-arm fix but this loop was not extended to
  // them, so the exact-budget invariant, the early-exit classes and
  // --fail_on_short_search were all blind to two thirds of the wide teacher's
  // roots -- which is the same shape of gap the fix was written to close.
  std::vector<std::pair<const SearchPiRoleStats*, int>> roles = {
      {&r.primary, absl::GetFlag(FLAGS_search_pi_primary_simulations)},
      {&r.continuation, absl::GetFlag(FLAGS_search_pi_continuation_simulations)}};
  std::vector<const char*> role_names = {"primary", "continuation"};
  if (wide_scope) {
    roles.push_back({&r.purchase, pcb});
    roles.push_back({&r.combat_intrigue, pcb});
    roles.push_back({&r.other_optional, pcb});
    role_names.push_back("purchase");
    role_names.push_back("combat_intrigue");
    role_names.push_back("other_optional");
  } else {
    // And the converse for narrow: the three buckets must be untouched. A
    // narrow arm with a leaked budget would otherwise show up nowhere.
    for (const std::pair<const SearchPiRoleStats*, const char*>& b :
         {std::make_pair(&r.purchase, "purchase"),
          std::make_pair(&r.combat_intrigue, "combat_intrigue"),
          std::make_pair(&r.other_optional, "other_optional")}) {
      if (b.first->roots_seen != 0 || b.first->simulations_requested != 0) {
        fail(absl::StrCat("narrow arm touched the ", b.second, " bucket: ",
                          b.first->roots_seen, " roots, ",
                          b.first->simulations_requested, " requested"));
      }
    }
  }
  const bool fail_on_short = absl::GetFlag(FLAGS_fail_on_short_search);
  int64_t all_fallbacks = 0;
  for (const std::pair<const SearchPiRoleStats*, int>& rp : roles) {
    all_fallbacks += rp.first->fallbacks;
  }
  bool fallback_outside_watchdog = false;
  for (const SearchPiGameOutcome& g : r.games) {
    if (g.fallbacks != g.watchdog_fallbacks ||
        g.watchdog_fallbacks > g.searches_short_of_budget) {
      fallback_outside_watchdog = true;
      break;
    }
  }
  if (all_fallbacks != 0 &&
      (fail_on_short || fallback_outside_watchdog)) {
    fail(absl::StrCat(all_fallbacks, " technical fallbacks",
                      fallback_outside_watchdog
                          ? " including a fallback not attributable to a "
                            "named watchdog timeout"
                          : ""));
  }
  // The played-action floor. A searched arm that never played anything other
  // than the raw-prior argmax at a role class it is registered to search did
  // not test that class -- it billed for it. Asserted per class, because the
  // 2026-08-18 defect was exactly a wide arm whose AGENT roles diverged
  // normally while its three wide roles were inert.
  // PER ROLE, and conditioned on the role having been REACHED. The lumped
  // agent/wide pair this replaced asserted over a union: with the observed role
  // frequencies, a wide arm whose kPurchase and kCombatIntrigue searches were
  // entirely inert still showed a healthy count from kOtherOptional alone and
  // passed -- billing for three roles while testing one.
  int64_t differs_by_role[7] = {0, 0, 0, 0, 0, 0, 0};
  for (const SearchPiGameOutcome& g : r.games) {
    for (int rr = 0; rr < 7; ++rr) differs_by_role[rr] += g.roots_played_differs_from_raw[rr];
  }
  const std::vector<int> searched_role_idx =
      wide_scope ? std::vector<int>{2, 3, 4, 5, 6} : std::vector<int>{2, 3};
  for (int rr : searched_role_idx) {
    if (r.simulations_by_role[rr] > 0 && differs_by_role[rr] == 0) {
      fail(absl::StrCat("role ", rr, ": the searched arm ran ",
                        r.simulations_by_role[rr], " simulations and NEVER "
                        "played anything but the raw-prior argmax -- it "
                        "searched this role and discarded the result, which is "
                        "the 2026-08-18 defect"));
    }
  }
  for (int rr = 0; rr < 7; ++rr) {
    const bool searchable =
        std::find(searched_role_idx.begin(), searched_role_idx.end(), rr)
        != searched_role_idx.end();
    if (!searchable && differs_by_role[rr] != 0) {
      fail(absl::StrCat("role ", rr, ": diverged from raw at ",
                        differs_by_role[rr], " roots it must never have "
                        "searched"));
    }
  }
  // Role index per bucket, so a zero-root role can be distinguished from a role
  // the arm never reached.
  std::vector<int> role_idx = {2, 3};
  if (wide_scope) { role_idx.push_back(4); role_idx.push_back(5); role_idx.push_back(6); }
  for (size_t i = 0; i < roles.size(); ++i) {
    const SearchPiRoleStats& rs = *roles[i].first;
    const int64_t want = rs.roots_seen * roles[i].second;
    // A role with ZERO ROOTS is a defect only if the arm REACHED it. Whole
    // generations are observed with zero kCombatIntrigue decisions -- that is a
    // property of the game, not of the measurement, and hard-failing on it
    // would mark a valid multi-hour cell INVALID for a draw. The correctly
    // guarded form already exists above, at the converse scope check.
    if (rs.roots_seen <= 0) {
      if (r.decisions_by_role[role_idx[i]] > 0) {
        fail(absl::StrCat(role_names[i], ": ",
                          r.decisions_by_role[role_idx[i]],
                          " decisions reached this role and NONE became a "
                          "searched root"));
      }
      continue;
    }
    if (rs.simulations_requested != want) {
      fail(absl::StrCat(role_names[i], ": the session requested ",
                        rs.simulations_requested, " simulations, not ",
                        rs.roots_seen, "x", roles[i].second, " = ", want));
    }
    int64_t tolerated_timeouts = 0;
    for (int k = 0; k < kSearchPiEarlyExitCount; ++k) {
      const SearchPiEarlyExit reason = static_cast<SearchPiEarlyExit>(k);
      if (reason == SearchPiEarlyExit::kNone) continue;
      if (SearchPiAuditToleratesEarlyExit(reason, fail_on_short)) {
        tolerated_timeouts += rs.early_exit_counts[k];
        continue;
      }
      if (rs.early_exit_counts[k] != 0) {
        fail(absl::StrCat(
            role_names[i], ": ", rs.early_exit_counts[k], " '",
            SearchPiEarlyExitName(reason),
            "' early exits"));
      }
    }
    if (!fail_on_short && tolerated_timeouts != rs.searches_short_of_budget) {
      fail(absl::StrCat(role_names[i], ": ", tolerated_timeouts,
                        " timeout exits but ", rs.searches_short_of_budget,
                        " searches short of budget; tolerated audit discards "
                        "must be attributable exactly to the watchdog"));
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
      if (fail_on_short) {
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
  const int requested_threads = absl::GetFlag(FLAGS_threads);
  const int available_chunks = games / chunk;
  if (requested_threads <= 0) {
    std::cerr << "--threads must be positive." << std::endl;
    return 2;
  }
  if (requested_threads > available_chunks) {
    std::cerr << "--threads=" << requested_threads << " but only "
              << available_chunks << " scored chunks exist (games=" << games
              << ", chunk_games=" << chunk
              << "); refusing to cap actual workers below requested G."
              << std::endl;
    return 2;
  }
  const int requested_batch_target = absl::GetFlag(FLAGS_batch_target);
  const int max_inflight_rows =
      requested_batch_target > 0
          ? static_cast<int>(kBatchedLeafGroupRows) * requested_threads
          : 0;
  if (requested_batch_target != 0 &&
      (requested_batch_target <= 0 || requested_batch_target % 4 != 0)) {
    std::cerr << "--batch_target must be zero (reference mode) or a positive "
                 "multiple of four."
              << std::endl;
    return 2;
  }
  if (requested_batch_target > max_inflight_rows) {
    std::cerr << "--batch_target=" << requested_batch_target
              << " exceeds max_inflight_rows=4*actual_workers="
              << max_inflight_rows << "." << std::endl;
    return 2;
  }
  const std::string arm_flag = absl::GetFlag(FLAGS_arm);
  if (arm_flag != "searched" && arm_flag != "raw_argmax" && arm_flag != "both") {
    std::cerr << "--arm must be searched | raw_argmax | both." << std::endl;
    return 2;
  }

  auto game = open_spiel::LoadGame("dune_imperium");
  if (game->NumPlayers() != kBatchedLeafGroupRows) {
    // max_inflight_rows=4*G is derived from one in-flight NumPlayers-row
    // evaluation group per game. Keep that capacity identity explicit even
    // though the independently tuned target may be below the maximum.
    std::cerr << "kBatchedLeafGroupRows is " << kBatchedLeafGroupRows
              << " but the game has " << game->NumPlayers()
              << " players; max_inflight_rows=4*G is invalid."
              << std::endl;
    return 2;
  }
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

  // --- Rung 4A hybrid controller ------------------------------------------
  //
  // Loaded and hash-verified BEFORE any game is played, for the same reason the
  // trainer verifies its collector archive up front: a gate arm that discovers
  // it was pointed at the wrong candidate after the games exist has already
  // spent the budget. The SHA here is caller-supplied rather than a pinned
  // constant, because the candidate changes from attempt to attempt while the
  // frozen leaf evaluator does not.
  std::shared_ptr<SharedDunePolicyValueNetImpl> policy_prior_model;
  const std::string prior_ckpt =
      absl::GetFlag(FLAGS_search_pi_policy_prior_model_checkpoint);
  const std::string prior_sha =
      absl::GetFlag(FLAGS_search_pi_policy_prior_model_sha256);
  if (!prior_ckpt.empty() || !prior_sha.empty()) {
    if (prior_ckpt.empty() || prior_sha.empty()) {
      std::cerr << "a policy-prior checkpoint and its sha256 must be given "
                   "together, or neither." << std::endl;
      return 2;
    }
    const std::string actual = open_spiel::ComputeFileSHA256(prior_ckpt);
    if (actual != prior_sha) {
      std::cerr << "policy-prior checkpoint sha256 " << actual << " does not "
                << "match the expected " << prior_sha << "." << std::endl;
      return 2;
    }
    policy_prior_model = std::make_shared<SharedDunePolicyValueNetImpl>(
        obs_size, absl::GetFlag(FLAGS_hidden_dim), action_size,
        absl::GetFlag(FLAGS_num_blocks),
        absl::GetFlag(FLAGS_nonlinear_value_head));
    policy_prior_model->to(device);
    try {
      torch::load(policy_prior_model, prior_ckpt, device);
    } catch (const c10::Error& e) {
      std::cerr << "Failed to load " << prior_ckpt << ":\n" << e.msg()
                << std::endl;
      return 2;
    }
    policy_prior_model->to(device);
    policy_prior_model->eval();
    std::cout << "[audit] hybrid policy prior=" << prior_ckpt << std::endl;
  }

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
    ArmResult r = RunArm(SearchPiArm::kSearched, game, model, device,
                         policy_prior_model);
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
    // Deliberately NOT given the policy-prior model. The raw arm never searches,
    // so its behaviour comes from the generator's behaviour evaluator, which is
    // the frozen model in both controllers -- passing the candidate here would
    // be inert at best and a silent scope change at worst.
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
