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
  int max_inflight_rows = 0;
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
  double wall_time_s = 0.0;
  bool telemetry_valid = false;
  bool telemetry_device_applicable = false;
  BatcherTelemetry telemetry;
};

bool SearchPiBatcherRowsIdentityOk(const BatcherTelemetry& telemetry);
bool SearchPiBatcherDeviceTimingComplete(const BatcherTelemetry& telemetry,
                                         bool device_applicable);

// Returns all games/rows in episode order, independent of worker completion
// order. Invalid geometry is rejected before an evaluator or worker is made.
ConcurrentSearchPiCollectionResult CollectSearchPiConcurrent(
    const ConcurrentSearchPiCollectionConfig& config,
    const std::shared_ptr<const Game>& game,
    const std::shared_ptr<SharedDunePolicyValueNetImpl>& frozen_model,
    torch::Device device);

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
