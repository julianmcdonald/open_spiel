#include "dune_search_pi_async_resume.h"

#include <algorithm>
#include <iterator>
#include <limits>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace open_spiel {
namespace {

bool Fail(const std::string& message, std::string* error) {
  if (error != nullptr) *error = message;
  return false;
}

}  // namespace

bool ResolveAsyncInitialGeneration(
    std::optional<int64_t> persisted_initial_generation,
    int published_generation, int* initial_generation, std::string* error) {
  if (initial_generation == nullptr || error == nullptr) return false;
  error->clear();
  if (published_generation < 0) {
    return Fail("async published generation is negative", error);
  }
  const int64_t resolved = persisted_initial_generation.has_value()
                               ? *persisted_initial_generation
                               : published_generation;
  if (resolved < 0 || resolved > std::numeric_limits<int>::max() ||
      resolved > published_generation) {
    return Fail("async initial generation is invalid", error);
  }
  *initial_generation = static_cast<int>(resolved);
  return true;
}

bool ResolveAsyncFirstEpisodeId(
    std::optional<int64_t> persisted_first_episode_id,
    int64_t next_episode_id, int64_t* first_episode_id, std::string* error) {
  if (first_episode_id == nullptr || error == nullptr) return false;
  error->clear();
  if (next_episode_id < 0) {
    return Fail("async next episode id is negative", error);
  }
  const int64_t resolved = persisted_first_episode_id.has_value()
                               ? *persisted_first_episode_id
                               : next_episode_id;
  if (resolved < 0 || resolved > next_episode_id) {
    return Fail("async first episode id is invalid", error);
  }
  *first_episode_id = resolved;
  return true;
}

bool ReconstructAsyncResumePendingUpdate(
    const std::vector<LedgerEntry>& completed, int next_update,
    int games_per_update, int learner_generation,
    const AsyncResumeShardLoader& load_shard, AsyncResumePendingUpdate* pending,
    std::string* error) {
  if (pending == nullptr || error == nullptr) return false;
  *pending = AsyncResumePendingUpdate();
  error->clear();
  if (next_update < 0 || games_per_update <= 0) {
    return Fail("invalid async resume update geometry", error);
  }
  if (static_cast<int64_t>(next_update) >
      std::numeric_limits<int64_t>::max() / games_per_update) {
    return Fail("async resume consumed boundary overflows", error);
  }
  const int64_t consumed_boundary =
      static_cast<int64_t>(next_update) * games_per_update;
  if (static_cast<int64_t>(completed.size()) < consumed_boundary) {
    return Fail("async resume completed ledger is shorter than consumed boundary",
                error);
  }
  const int64_t suffix_games =
      static_cast<int64_t>(completed.size()) - consumed_boundary;
  if (suffix_games >= games_per_update) {
    return Fail("async resume pending suffix reaches or exceeds one full update",
                error);
  }
  pending->resumed_prefetch_games = suffix_games;
  pending->collected_games_required = games_per_update - suffix_games;
  if (suffix_games == 0) return true;

  pending->collector_generation_min = std::numeric_limits<int>::max();
  pending->collector_generation_max = std::numeric_limits<int>::min();
  std::set<int64_t> unique_episode_ids;
  for (size_t index = static_cast<size_t>(consumed_boundary);
       index < completed.size(); ++index) {
    const LedgerEntry& entry = completed[index];
    if (entry.episode_id < 0 ||
        !unique_episode_ids.insert(entry.episode_id).second) {
      return Fail("async resume pending suffix has invalid or duplicate episode id",
                  error);
    }
    if (entry.shard_path.empty()) {
      return Fail("async resume pending suffix has an empty shard path", error);
    }
    if (entry.rows <= 0 || entry.requested_simulations < 0 ||
        entry.completed_simulations < 0 || entry.full_policy_target_rows < 0 ||
        entry.full_policy_target_rows > entry.rows ||
        entry.trajectory_transitions <= 0 || entry.collector_generation < 0 ||
        entry.start_staleness < 0) {
      return Fail("async resume pending ledger metadata is incomplete", error);
    }
    if (entry.requested_simulations != entry.completed_simulations ||
        entry.fallbacks != 0 || entry.watchdog_timeouts != 0 ||
        entry.non_timeout_early_exits != 0) {
      return Fail("async resume pending game violates the health contract", error);
    }

    std::vector<SearchPiRow> rows;
    std::string shard_error;
    if (!load_shard(entry.shard_path, &rows, &shard_error)) {
      return Fail("async resume pending shard load failed: " + shard_error,
                  error);
    }
    if (static_cast<int64_t>(rows.size()) != entry.rows) {
      return Fail("async resume pending shard row count mismatches ledger", error);
    }
    int64_t requested_simulations = 0;
    int64_t completed_simulations = 0;
    int64_t full_policy_target_rows = 0;
    for (const SearchPiRow& row : rows) {
      if (row.episode_id != entry.episode_id ||
          row.simulations_requested < 0 || row.simulations_completed < 0) {
        return Fail("async resume pending shard row metadata mismatches ledger",
                    error);
      }
      requested_simulations += row.simulations_requested;
      completed_simulations += row.simulations_completed;
      if (row.search_budget_class == "full" && row.policy_target_weight > 0.0) {
        ++full_policy_target_rows;
      }
    }
    if (requested_simulations != entry.requested_simulations ||
        completed_simulations != entry.completed_simulations ||
        full_policy_target_rows != entry.full_policy_target_rows) {
      return Fail("async resume pending shard simulation accounting mismatches ledger",
                  error);
    }

    pending->episode_ids.push_back(entry.episode_id);
    pending->shard_paths.push_back(entry.shard_path);
    pending->rows.insert(pending->rows.end(),
                         std::make_move_iterator(rows.begin()),
                         std::make_move_iterator(rows.end()));
    pending->trajectory_transitions += entry.trajectory_transitions;
    pending->requested_simulations += entry.requested_simulations;
    pending->completed_simulations += entry.completed_simulations;
    pending->full_policy_target_rows += entry.full_policy_target_rows;
    pending->fallbacks += entry.fallbacks;
    pending->watchdog_timeouts += entry.watchdog_timeouts;
    pending->non_timeout_early_exits += entry.non_timeout_early_exits;
    pending->collector_generation_sum += entry.collector_generation;
    pending->collector_generation_min =
        std::min(pending->collector_generation_min, entry.collector_generation);
    pending->collector_generation_max =
        std::max(pending->collector_generation_max, entry.collector_generation);
    pending->collector_start_staleness_sum += entry.start_staleness;
    pending->collector_start_staleness_max = std::max(
        pending->collector_start_staleness_max, entry.start_staleness);
    const int staleness =
        std::max(0, learner_generation - entry.collector_generation);
    pending->collector_staleness_sum += staleness;
    pending->collector_staleness_max =
        std::max(pending->collector_staleness_max, staleness);
  }
  if (pending->episode_ids.size() != static_cast<size_t>(suffix_games) ||
      pending->shard_paths.size() != static_cast<size_t>(suffix_games)) {
    return Fail("async resume pending suffix reconstruction is incomplete", error);
  }
  return true;
}

}  // namespace open_spiel
