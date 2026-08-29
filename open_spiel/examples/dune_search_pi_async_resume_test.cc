#include "dune_search_pi_async_resume.h"

#include <cstdint>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace {

void TestInitialGenerationMissingVersusExplicitZero() {
  int initial_generation = -1;
  std::string error;
  SPIEL_CHECK_TRUE(ResolveAsyncInitialGeneration(
      std::nullopt, 115, &initial_generation, &error));
  SPIEL_CHECK_EQ(initial_generation, 115);
  SPIEL_CHECK_TRUE(ResolveAsyncInitialGeneration(
      int64_t{0}, 115, &initial_generation, &error));
  SPIEL_CHECK_EQ(initial_generation, 0);
  SPIEL_CHECK_TRUE(ResolveAsyncInitialGeneration(
      int64_t{105}, 115, &initial_generation, &error));
  SPIEL_CHECK_EQ(initial_generation, 105);
  SPIEL_CHECK_FALSE(ResolveAsyncInitialGeneration(
      int64_t{116}, 115, &initial_generation, &error));
}

void TestFirstEpisodeMissingVersusExplicitValue() {
  int64_t first_episode_id = -1;
  std::string error;
  SPIEL_CHECK_TRUE(ResolveAsyncFirstEpisodeId(
      std::nullopt, 3172576, &first_episode_id, &error));
  SPIEL_CHECK_EQ(first_episode_id, 3172576);
  SPIEL_CHECK_TRUE(ResolveAsyncFirstEpisodeId(
      int64_t{3151584}, 3172576, &first_episode_id, &error));
  SPIEL_CHECK_EQ(first_episode_id, 3151584);
  SPIEL_CHECK_FALSE(ResolveAsyncFirstEpisodeId(
      int64_t{3172577}, 3172576, &first_episode_id, &error));
}

LedgerEntry ValidPendingEntry(int64_t episode_id) {
  LedgerEntry entry;
  entry.episode_id = episode_id;
  entry.collector_generation = 145;
  entry.shard_path = "game_" + std::to_string(episode_id) + ".bin";
  entry.rows = 2;
  entry.requested_simulations = 5;
  entry.completed_simulations = 5;
  entry.full_policy_target_rows = 1;
  entry.trajectory_transitions = 100;
  return entry;
}

bool LoadSyntheticShard(const std::string& path,
                        std::vector<SearchPiRow>* rows, std::string* error) {
  const size_t prefix = path.find("game_");
  const size_t suffix = path.rfind(".bin");
  if (prefix == std::string::npos || suffix == std::string::npos) {
    *error = "synthetic shard path is malformed";
    return false;
  }
  const int64_t episode_id =
      std::stoll(path.substr(prefix + 5, suffix - prefix - 5));
  SearchPiRow first;
  first.episode_id = episode_id;
  first.simulations_requested = 2;
  first.simulations_completed = 2;
  first.search_budget_class = "full";
  first.policy_target_weight = 1.0;
  SearchPiRow second;
  second.episode_id = episode_id;
  second.simulations_requested = 3;
  second.simulations_completed = 3;
  second.search_budget_class = "cheap";
  second.policy_target_weight = 0.0;
  *rows = {std::move(first), std::move(second)};
  return true;
}

void TestFortyFivePlusFourHundredSixtySevenMakesFullCohort() {
  constexpr int kNextUpdate = 40;
  constexpr int kGamesPerUpdate = 512;
  constexpr int kConsumed = kNextUpdate * kGamesPerUpdate;
  std::vector<LedgerEntry> completed(kConsumed);
  for (int i = 0; i < 45; ++i) {
    completed.push_back(ValidPendingEntry(3172032 + i));
  }
  AsyncResumePendingUpdate pending;
  std::string error;
  SPIEL_CHECK_TRUE(ReconstructAsyncResumePendingUpdate(
      completed, kNextUpdate, kGamesPerUpdate, 146, LoadSyntheticShard,
      &pending, &error));
  SPIEL_CHECK_TRUE(error.empty());
  SPIEL_CHECK_EQ(pending.resumed_prefetch_games, 45);
  SPIEL_CHECK_EQ(pending.collected_games_required, 467);
  SPIEL_CHECK_EQ(pending.episode_ids.size(), 45);
  SPIEL_CHECK_EQ(pending.shard_paths.size(), 45);
  SPIEL_CHECK_EQ(pending.rows.size(), 90);
  SPIEL_CHECK_EQ(pending.trajectory_transitions, 4500);
  SPIEL_CHECK_EQ(pending.requested_simulations, 225);
  SPIEL_CHECK_EQ(pending.completed_simulations, 225);

  // This is the production boundary contract: reconstructed and newly
  // completed games share one cohort; a pause cannot turn it into a 467-game
  // cohort and evict 512 historical paths.
  std::vector<std::string> cohort = pending.shard_paths;
  for (int i = 0; i < pending.collected_games_required; ++i) {
    cohort.push_back("new_game_" + std::to_string(i) + ".bin");
  }
  SPIEL_CHECK_EQ(cohort.size(), kGamesPerUpdate);
}

void TestMalformedResumeSuffixFailsClosed() {
  constexpr int kGamesPerUpdate = 512;
  AsyncResumePendingUpdate pending;
  std::string error;
  std::vector<LedgerEntry> too_short(511);
  SPIEL_CHECK_FALSE(ReconstructAsyncResumePendingUpdate(
      too_short, 1, kGamesPerUpdate, 2, LoadSyntheticShard, &pending, &error));

  std::vector<LedgerEntry> full_pending(kGamesPerUpdate);
  for (int i = 0; i < kGamesPerUpdate; ++i) {
    full_pending[i] = ValidPendingEntry(1000 + i);
  }
  SPIEL_CHECK_FALSE(ReconstructAsyncResumePendingUpdate(
      full_pending, 0, kGamesPerUpdate, 1, LoadSyntheticShard, &pending,
      &error));

  std::vector<LedgerEntry> bad_metadata = {ValidPendingEntry(2000)};
  bad_metadata.front().trajectory_transitions = 0;
  SPIEL_CHECK_FALSE(ReconstructAsyncResumePendingUpdate(
      bad_metadata, 0, kGamesPerUpdate, 1, LoadSyntheticShard, &pending,
      &error));

  bad_metadata.front() = ValidPendingEntry(2000);
  bad_metadata.front().rows = 3;
  SPIEL_CHECK_FALSE(ReconstructAsyncResumePendingUpdate(
      bad_metadata, 0, kGamesPerUpdate, 1, LoadSyntheticShard, &pending,
      &error));

  const auto missing_loader = [](const std::string&, std::vector<SearchPiRow>*,
                                 std::string* shard_error) {
    *shard_error = "missing";
    return false;
  };
  bad_metadata.front() = ValidPendingEntry(2000);
  SPIEL_CHECK_FALSE(ReconstructAsyncResumePendingUpdate(
      bad_metadata, 0, kGamesPerUpdate, 1, missing_loader, &pending, &error));
}

}  // namespace
}  // namespace open_spiel

int main() {
  open_spiel::TestInitialGenerationMissingVersusExplicitZero();
  open_spiel::TestFirstEpisodeMissingVersusExplicitValue();
  open_spiel::TestFortyFivePlusFourHundredSixtySevenMakesFullCohort();
  open_spiel::TestMalformedResumeSuffixFailsClosed();
  std::cout << "dune_search_pi_async_resume_test PASS\n";
  return 0;
}
