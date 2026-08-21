// Shared concurrent Search-PI collection used by both the audit and trainer.
//
// This is deliberately the only implementation of cross-game scheduling,
// shared batching, ordered chunk merging, short-episode streaming, and the
// 512->64 training-subset validation. Search mathematics remains in
// SearchPiGenerator; this helper only orchestrates independent complete games.
#ifndef OPEN_SPIEL_EXAMPLES_DUNE_SEARCH_PI_CONCURRENT_H_
#define OPEN_SPIEL_EXAMPLES_DUNE_SEARCH_PI_CONCURRENT_H_

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "open_spiel/spiel.h"
#include "dune_network.h"
#include "dune_search_pi.h"

namespace open_spiel {

constexpr int64_t kSearchPiBatchedLeafGroupRows = 4;

struct ConcurrentSearchPiCollectionConfig {
  SearchPiConfig search;
  int generation = 1;
  int collection_games = 4;
  int chunk_games = 4;
  int requested_workers = 1;
  int batch_target = 0;       // zero preserves the historical reference path
  int batcher_timeout_ms = 2;
  // Detailed per-row/device telemetry is useful for benchmark runs but adds
  // bookkeeping to every physical batch. Production can disable it while
  // retaining the always-on health counters.
  bool enable_batcher_telemetry = true;
  // The production fast path may use blocking D2H copies, which synchronize
  // only the work needed for that result instead of the whole CUDA device.
  bool device_synchronize = true;
  bool inference_high_priority = false;
  // Number of independent immutable collector-model/batcher lanes. Each lane
  // owns its queue, runner stream(s), staging buffers, and model mutex. The
  // historical path remains one lane; throughput probes may partition workers
  // evenly across multiple lanes.
  int inference_lanes = 1;
  // Optional bounded look-ahead. These games use the current frozen collector
  // but are returned separately so the caller cannot train on them in this
  // generation's update.
  int prefetch_games = 0;
  int prefetch_trigger_running = 32;
  int warmup_games = 0;
  bool retain_rows = true;
  float logit_cap = 10.0f;
  std::string output_dir;     // owns the uncapped short-episode stream
  SearchPiArm arm = SearchPiArm::kSearched;
};

struct ConcurrentSearchPiCollectionResult {
  SearchPiArm arm = SearchPiArm::kSearched;
  int requested_workers = 0;
  int actual_workers = 0;
  int configured_batch_target = 0;
  int configured_batcher_timeout_ms = 0;
  int configured_inference_lanes = 1;
  // Capacity of the FROZEN batcher, in rows: four leaf-group rows per worker,
  // because a worker thread blocks on exactly one evaluator request at a time
  // and its largest request is the four-player leaf group. Unchanged by the
  // hybrid mode below -- leaf values never leave the frozen batcher.
  int max_inflight_rows = 0;
  // Set when a policy-prior model was supplied, i.e. when priors inside the
  // searched seat's tree came from a candidate rather than the frozen model.
  // The RESULT declares this, not the caller, so a validator cannot be talked
  // out of the hybrid-only checks by a contract that predates the mode.
  bool hybrid_policy_prior = false;
  // Capacity of the POLICY batcher, in rows. ONE row per worker, not four: the
  // split evaluator routes only Prior() to the policy source
  // (dune_split_evaluator.h:86-88), and BatchedNNEvaluator::Prior submits a
  // single row (dune_batched_evaluator.h:51). Leaf groups are the frozen
  // batcher's alone. Zero when not hybrid.
  int policy_max_inflight_rows = 0;
  bool policy_telemetry_valid = false;
  BatcherTelemetry policy_telemetry;
  // Canonical digests of the two models a hybrid generation actually ran, so
  // provenance names both. Populated ONLY in hybrid mode: the incumbent path
  // must not acquire a per-generation full-parameter hash it never paid for,
  // and its callers already record the frozen digest themselves.
  std::string frozen_model_digest;
  std::string policy_prior_model_digest;
  std::vector<SearchPiGameOutcome> games;
  SearchPiRoleStats primary;
  SearchPiRoleStats continuation;
  SearchPiRoleStats purchase;
  SearchPiRoleStats combat_intrigue;
  SearchPiRoleStats other_optional;
  int64_t simulations_by_role[7] = {0, 0, 0, 0, 0, 0, 0};
  int64_t decisions_by_role[7] = {0, 0, 0, 0, 0, 0, 0};
  int64_t leader_rows_emitted = 0;
  int64_t inference_calls = 0;
  int64_t rows_total = 0;
  std::string target_hash_chain;
  std::string extended_hash_chain;
  std::vector<SearchPiRow> retained_rows;
  std::vector<SearchPiRow> prefetched_rows;
  std::vector<SearchPiGameOutcome> prefetched_games;
  int prefetch_collector_generation = -1;
  int prefetch_for_generation = -1;
  double wall_time_s = 0.0;
  double main_collection_wall_time_s = 0.0;
  double tail_wall_time_s = 0.0;
  int main_games_completed_before_tail = 0;
  bool prefetch_triggered = false;
  bool telemetry_valid = false;
  bool telemetry_device_applicable = false;
  BatcherTelemetry telemetry;
  std::vector<BatcherTelemetry> lane_telemetry;
  std::vector<int> lane_worker_counts;
};

bool SearchPiBatcherRowsIdentityOk(const BatcherTelemetry& telemetry);
bool SearchPiBatcherDeviceTimingComplete(const BatcherTelemetry& telemetry,
                                         bool device_applicable);

// Returns all games/rows in episode order, independent of worker completion
// order. Invalid geometry is rejected before an evaluator or worker is made.
//
// `policy_prior_model` is the HYBRID switch and defaults to absent. Null means
// the incumbent single-model collection, unchanged in every observable respect:
// one batcher, one evaluator per worker, one telemetry record. Non-null adds a
// SECOND batcher over that model and hands the searched seat's session a
// SplitPolicyValueEvaluator whose priors are the candidate's and whose leaf
// values remain the frozen model's; opponent seats and the searched seat's
// off-scope decisions stay frozen either way (dune_search_pi.cc, the session
// construction comment). The defaulted argument keeps both existing call sites
// -- dune_ppo_train.cc:3654 and dune_search_pi_audit.cc:471 -- on the incumbent
// path without an edit, so the running registered experiment cannot acquire the
// hybrid geometry by recompilation.
ConcurrentSearchPiCollectionResult CollectSearchPiConcurrent(
    const ConcurrentSearchPiCollectionConfig& config,
    const std::shared_ptr<const Game>& game,
    const std::shared_ptr<SharedDunePolicyValueNetImpl>& frozen_model,
    torch::Device device,
    const std::shared_ptr<SharedDunePolicyValueNetImpl>& policy_prior_model =
        nullptr);

struct SearchPiTrainingCollectionContract {
  int64_t first_episode_id = 0;
  int collection_games = 512;
  int training_games = 64;
  int requested_workers = 128;
  int batch_target = 64;
  int batcher_timeout_ms = 2;
  int primary_simulations = 200;
  int continuation_simulations = 64;
  int purchase_combat_budget = 0;
  int max_filler_timeout_episodes = 2;
  // Which collection geometry this contract was written for. Defaults to the
  // incumbent, so a hybrid result submitted against an unmodified contract
  // FAILS rather than being waved through by checks that only ever knew about
  // one batcher. The comparison runs in both directions: an incumbent result
  // cannot satisfy a contract that registered a hybrid run either.
  bool expect_hybrid_policy_prior = false;
};

struct SearchPiTrainingCollectionValidation {
  bool ok = false;
  std::vector<std::string> errors;
  std::vector<SearchPiRow> training_rows;
  std::vector<int64_t> filler_timeout_episode_ids;
  std::string collected_target_hash;
  std::string collected_extended_hash;
  std::string trained_target_hash;
  std::string trained_extended_hash;
  int64_t trained_rows_by_role[7] = {0, 0, 0, 0, 0, 0, 0};
};

// Selects the fixed episode-ID prefix before consulting outcomes or row
// statistics, then validates the complete collection transactionally. A caller
// must not invoke the learner unless `ok` is true.
SearchPiTrainingCollectionValidation ValidateSearchPiTrainingCollection(
    const ConcurrentSearchPiCollectionResult& result,
    const SearchPiTrainingCollectionContract& contract);

// Canonical in-memory identity over parameter AND buffer names, shapes, dtypes,
// and raw bytes. Unlike torch serialization this is stable across save calls.
std::string CanonicalSearchPiModuleDigest(torch::nn::Module& model);

}  // namespace open_spiel

#endif  // OPEN_SPIEL_EXAMPLES_DUNE_SEARCH_PI_CONCURRENT_H_
