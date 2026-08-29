#ifndef OPEN_SPIEL_EXAMPLES_DUNE_SEARCH_PI_ASYNC_RESUME_H_
#define OPEN_SPIEL_EXAMPLES_DUNE_SEARCH_PI_ASYNC_RESUME_H_

#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <vector>

#include "dune_search_pi.h"

namespace open_spiel {

// Persisted per-game accounting used both by the live async trainer and by the
// resume repair. A crash may leave a suffix of completed games after the last
// published learner boundary; that suffix is learner data, not merely an ID
// ledger used to suppress recollection.
struct LedgerEntry {
  int64_t episode_id = -1;
  int collector_generation = -1;
  std::string shard_path;
  int64_t rows = 0;
  int64_t requested_simulations = 0;
  int64_t completed_simulations = 0;
  int64_t full_policy_target_rows = 0;
  int64_t trajectory_transitions = 0;
  int start_staleness = 0;
  int64_t fallbacks = 0;
  int64_t watchdog_timeouts = 0;
  int64_t non_timeout_early_exits = 0;
};

struct AsyncResumePendingUpdate {
  int64_t resumed_prefetch_games = 0;
  int64_t collected_games_required = 0;
  std::vector<SearchPiRow> rows;
  std::vector<std::string> shard_paths;
  std::vector<int64_t> episode_ids;
  int64_t trajectory_transitions = 0;
  int64_t requested_simulations = 0;
  int64_t completed_simulations = 0;
  int64_t full_policy_target_rows = 0;
  int64_t fallbacks = 0;
  int64_t watchdog_timeouts = 0;
  int64_t non_timeout_early_exits = 0;
  double collector_generation_sum = 0.0;
  int collector_generation_min = 0;
  int collector_generation_max = 0;
  double collector_start_staleness_sum = 0.0;
  int collector_start_staleness_max = 0;
  double collector_staleness_sum = 0.0;
  int collector_staleness_max = 0;
};

using AsyncResumeShardLoader = std::function<bool(
    const std::string&, std::vector<SearchPiRow>*, std::string*)>;

// An explicit zero is a real lineage origin. Only a missing legacy field may
// inherit the already-parsed published generation.
bool ResolveAsyncInitialGeneration(
    std::optional<int64_t> persisted_initial_generation,
    int published_generation, int* initial_generation, std::string* error);

bool ResolveAsyncFirstEpisodeId(
    std::optional<int64_t> persisted_first_episode_id,
    int64_t next_episode_id, int64_t* first_episode_id, std::string* error);

// Reconstructs the unconsumed completed-ledger suffix for the next learner
// update. Returns false rather than weakening the fixed games-per-update dose
// when any persisted accounting cannot be proved from the shard bytes.
bool ReconstructAsyncResumePendingUpdate(
    const std::vector<LedgerEntry>& completed, int next_update,
    int games_per_update, int learner_generation,
    const AsyncResumeShardLoader& load_shard, AsyncResumePendingUpdate* pending,
    std::string* error);

}  // namespace open_spiel

#endif  // OPEN_SPIEL_EXAMPLES_DUNE_SEARCH_PI_ASYNC_RESUME_H_
