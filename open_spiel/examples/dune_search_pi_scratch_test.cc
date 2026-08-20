#include "dune_search_pi_scratch.h"

#include <cmath>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include <unistd.h>
#include <vector>

#include "open_spiel/abseil-cpp/absl/flags/flag.h"
#include "open_spiel/spiel_utils.h"
#include "open_spiel/games/dune_imperium/dune_imperium.h"
#include "dune_search_pi_concurrent.h"
#include "dune_search_pi_replay.h"

ABSL_FLAG(int, ppo_minibatch_size, 2048, "");
ABSL_FLAG(int, ppo_update_epochs, 4, "");
ABSL_FLAG(double, ppo_clip_epsilon, 0.2, "");
ABSL_FLAG(bool, normalize_advantages, true, "");
ABSL_FLAG(bool, ppo_clip_value_loss, true, "");
ABSL_FLAG(double, entropy_coef, 0.01, "");
ABSL_FLAG(double, value_coef, 0.5, "");
ABSL_FLAG(double, logit_cap, 10.0, "");
ABSL_FLAG(double, target_kl, 0.0, "");
ABSL_FLAG(bool, train_amp, true, "");
ABSL_FLAG(double, grad_clip_norm, 0.5, "");
ABSL_FLAG(uint64_t, shaping_start_env_steps, 206830543, "");
ABSL_FLAG(uint64_t, shaping_decay_env_steps, 0, "");
ABSL_FLAG(bool, diagnostics_only, false, "");

namespace open_spiel {
namespace {

SearchPiConfig ScratchConfig() {
  SearchPiConfig config;
  config.scratch_q_v1 = true;
  config.seed_domain = 8271001;
  config.purchase_combat_budget = 64;
  config.scratch_full_root_probability = 0.25;
  config.scratch_full_primary_simulations = 200;
  config.scratch_full_other_simulations = 64;
  config.scratch_cheap_primary_simulations = 32;
  config.scratch_cheap_other_simulations = 16;
  return config;
}

SearchPiRow Row(bool full, int64_t episode, int decision,
                std::vector<double> regularized = {0.25, 0.75}) {
  SearchPiRow row;
  row.observation = {0.1f, 0.2f, 0.3f, 0.4f};
  row.observation_is_information_state = true;
  row.player = 2;
  row.role = DuneDecisionRole::kAgentPrimary;
  row.legal_actions = {0, 1};
  row.raw_policy = {0.6, 0.4};
  row.raw_visits = {3, 5};
  row.target_visits = {7, 1};
  row.target_probs = {0.875, 0.125};
  row.chosen_action = 1;
  row.value_target = 0.5625;
  row.value_target_attached = true;
  row.generation = 2;
  row.episode_id = episode;
  row.decision_id = decision;
  row.state_fingerprint = "fixture";
  row.simulations_requested = full ? 200 : 32;
  row.simulations_completed = row.simulations_requested;
  row.q_values = {0.1, 0.2};
  row.root_value = 0.0;
  FillSearchPiRowScalars(&row);
  row.row_schema_version = 2;
  row.target_type = "regularized_q_kl";
  row.search_budget_class = full ? "full" : "cheap";
  row.search_budget_draw = 17;
  row.policy_target_weight = full ? 1.0 : 0.0;
  row.regularized_q_valid = true;
  row.regularized_q_error = "none";
  row.regularized_q_target = std::move(regularized);
  row.regularized_q_beta = 1.0;
  row.regularized_q_kl = 0.01;
  row.regularized_q_prior_expected_q = 0.14;
  row.regularized_q_target_expected_q = 0.175;
  row.regularized_q_q_range = 0.1;
  row.regularized_q_direct_visit_count = 2;
  row.regularized_q_entropy_norm =
      NormalizedEntropy(row.regularized_q_target);
  return row;
}

std::shared_ptr<SharedDunePolicyValueNetImpl> TinyModel(uint64_t seed) {
  torch::manual_seed(seed);
  return std::make_shared<SharedDunePolicyValueNetImpl>(
      4, 16, 4, 1, false);
}

std::unique_ptr<torch::optim::AdamW> Optimizer(
    const std::shared_ptr<SharedDunePolicyValueNetImpl>& model) {
  return std::make_unique<torch::optim::AdamW>(
      model->parameters(), torch::optim::AdamWOptions(1e-3).eps(1e-5));
}

void TestRegularizedQ() {
  const std::vector<Action> actions = {10, 11, 12};
  auto equal = BuildRegularizedQKlTarget(actions, {0.2, 0.3, 0.5},
                                         {1, 1, 1}, {0.7, 0.7, 0.7}, 0.0);
  SPIEL_CHECK_TRUE(equal.ok);
  SPIEL_CHECK_FLOAT_EQ(equal.target[0], 0.2);
  SPIEL_CHECK_FLOAT_EQ(equal.target[1], 0.3);
  SPIEL_CHECK_FLOAT_EQ(equal.target[2], 0.5);

  auto improved = BuildRegularizedQKlTarget(
      actions, {0.6, 0.3, 0.1}, {2, 1, 0}, {0.0, 0.2, 99.0}, 0.1);
  SPIEL_CHECK_TRUE(improved.ok);
  SPIEL_CHECK_GT(improved.target[1] / improved.target[0], 0.3 / 0.6);
  SPIEL_CHECK_GE(improved.target_expected_q + 1e-12,
                 improved.prior_expected_q);
  SPIEL_CHECK_EQ(improved.direct_visit_count, 2);

  auto bound = BuildRegularizedQKlTarget(
      actions, {0.5, 0.25, 0.25}, {1, 1, 1}, {-1.0, 1.0, 1.0}, 0.0);
  SPIEL_CHECK_TRUE(bound.ok);
  SPIEL_CHECK_LE(bound.kl, 0.10 + 1e-12);
  SPIEL_CHECK_LT(bound.beta, 4.0);

  auto zero = BuildRegularizedQKlTarget({1, 2}, {0.0, 1.0}, {1, 1},
                                        {1.0, 0.0}, 0.0);
  SPIEL_CHECK_TRUE(zero.ok);
  SPIEL_CHECK_GT(zero.target[0], 0.0);
  auto one = BuildRegularizedQKlTarget({1}, {0.0}, {0}, {0.0}, 0.25);
  SPIEL_CHECK_TRUE(one.ok);
  SPIEL_CHECK_FLOAT_EQ(one.target[0], 1.0);

  SPIEL_CHECK_FALSE(BuildRegularizedQKlTarget({}, {}, {}, {}, 0.0).ok);
  SPIEL_CHECK_FALSE(BuildRegularizedQKlTarget({1}, {1.0, 0.0}, {1}, {0.0}, 0.0).ok);
  SPIEL_CHECK_FALSE(BuildRegularizedQKlTarget(
      {1}, {std::numeric_limits<double>::quiet_NaN()}, {1}, {0.0}, 0.0).ok);
  SPIEL_CHECK_FALSE(BuildRegularizedQKlTarget(
      {1}, {1.0}, {1}, {std::numeric_limits<double>::infinity()}, 0.0).ok);

  auto shifted = BuildRegularizedQKlTarget(
      actions, {0.6, 0.3, 0.1}, {2, 1, 0}, {5.0, 5.2, 104.0}, 5.1);
  SPIEL_CHECK_TRUE(shifted.ok);
  for (size_t i = 0; i < actions.size(); ++i) {
    SPIEL_CHECK_LE(std::abs(improved.target[i] - shifted.target[i]), 1e-12);
  }
}

void TestBudgets() {
  SearchPiConfig config = ScratchConfig();
  const auto a = AssignScratchSearchBudget(
      config, 44, 12, 3, DuneDecisionRole::kAgentPrimary);
  const auto b = AssignScratchSearchBudget(
      config, 44, 12, 3, DuneDecisionRole::kAgentPrimary);
  SPIEL_CHECK_EQ(a.draw, b.draw);
  SPIEL_CHECK_EQ(a.full, b.full);
  SPIEL_CHECK_TRUE(a.simulations == 200 || a.simulations == 32);
  const auto continuation = AssignScratchSearchBudget(
      config, 44, 12, 3, DuneDecisionRole::kAgentContinuation);
  SPIEL_CHECK_TRUE(continuation.simulations == 64 ||
                   continuation.simulations == 16);
  for (DuneDecisionRole role : {DuneDecisionRole::kPurchase,
                                DuneDecisionRole::kCombatIntrigue,
                                DuneDecisionRole::kOtherOptional}) {
    const auto budget = AssignScratchSearchBudget(config, 1, 2, 0, role);
    SPIEL_CHECK_TRUE(budget.simulations == 64 || budget.simulations == 16);
  }
  SPIEL_CHECK_EQ(AssignScratchSearchBudget(
                     config, 1, 2, 0, DuneDecisionRole::kLeaderSelection)
                     .simulations,
                 0);
}

void TestLearnerMaskingAndTarget() {
  SearchPiLearnerConfig config;
  config.learning_rate = 1e-3;
  config.minibatch_size = 2;
  config.epochs = 1;
  config.value_coef = 0.0;
  auto cheap_model = TinyModel(9);
  auto cheap_optimizer = Optimizer(cheap_model);
  const std::string before = CanonicalSearchPiModuleDigest(*cheap_model);
  auto cheap_stats = RunScratchSearchPiLearner(
      cheap_model, *cheap_optimizer, {Row(false, 1, 1), Row(false, 2, 2)},
      4, 4, torch::kCPU, 99, 1, config);
  SPIEL_CHECK_FALSE(cheap_stats.policy_backward_executed);
  SPIEL_CHECK_FALSE(cheap_stats.value_backward_executed);
  SPIEL_CHECK_EQ(before, CanonicalSearchPiModuleDigest(*cheap_model));

  config.value_coef = 1.0;
  auto value_stats = RunScratchSearchPiLearner(
      cheap_model, *cheap_optimizer, {Row(false, 1, 1), Row(false, 2, 2)},
      4, 4, torch::kCPU, 99, 1, config);
  SPIEL_CHECK_FALSE(value_stats.policy_backward_executed);
  SPIEL_CHECK_TRUE(value_stats.value_backward_executed);
  SPIEL_CHECK_NE(before, CanonicalSearchPiModuleDigest(*cheap_model));

  config.value_coef = 0.0;
  auto first = TinyModel(17);
  auto second = TinyModel(33);
  CopySearchPiModel(first, second);
  auto first_optimizer = Optimizer(first);
  auto second_optimizer = Optimizer(second);
  SearchPiRow row_a = Row(true, 7, 1, {0.1, 0.9});
  SearchPiRow row_b = row_a;
  row_b.target_probs = {0.999, 0.001};
  auto stats_a = RunScratchSearchPiLearner(first, *first_optimizer, {row_a},
                                           4, 4, torch::kCPU, 42, 1, config);
  auto stats_b = RunScratchSearchPiLearner(second, *second_optimizer, {row_b},
                                           4, 4, torch::kCPU, 42, 1, config);
  SPIEL_CHECK_TRUE(stats_a.policy_backward_executed);
  SPIEL_CHECK_EQ(CanonicalSearchPiModuleDigest(*first),
                 CanonicalSearchPiModuleDigest(*second));
  SPIEL_CHECK_FALSE(ScratchCollectorOwnedByOptimizer(second,
                                                     *first_optimizer));
}

void TestV2AndReplay() {
  const std::filesystem::path directory =
      std::filesystem::temp_directory_path() /
      ("searchpi_scratch_test_" + std::to_string(::getpid()));
  std::filesystem::create_directories(directory);
  const std::string path =
      ScratchSearchPiShardPathForGeneration(directory.string(), 2);
  std::vector<SearchPiRow> rows = {Row(true, 1, 1), Row(false, 2, 2)};
  std::string error;
  SPIEL_CHECK_TRUE(WriteScratchSearchPiRowShardV2(path, rows, &error));
  std::vector<SearchPiRow> back;
  SPIEL_CHECK_TRUE(ReadScratchSearchPiRowShardV2(path, &back, &error));
  SPIEL_CHECK_EQ(back.size(), rows.size());
  SPIEL_CHECK_EQ(back[0].regularized_q_target, rows[0].regularized_q_target);
  std::vector<SearchPiRow> legacy;
  SPIEL_CHECK_FALSE(ReadSearchPiRowShard(path, &legacy, &error));

  std::vector<std::vector<SearchPiRow>> window(2);
  for (int i = 0; i < 10; ++i) {
    SearchPiRow row = Row(i < 3, 100 + i, i);
    row.generation = 2;
    window[1].push_back(row);
  }
  for (int i = 0; i < 10; ++i) {
    SearchPiRow row = Row(false, 200 + i, i);
    row.generation = 1;
    window[0].push_back(row);
  }
  auto sample = SampleScratchSearchPiReplayWindow(window, 2, 7, 123, &error);
  SPIEL_CHECK_TRUE(error.empty());
  SPIEL_CHECK_EQ(sample.size(), 7u);
  int current_full = 0;
  for (const SearchPiRow& row : sample) {
    if (row.generation == 2 && row.search_budget_class == "full") {
      ++current_full;
    }
  }
  SPIEL_CHECK_EQ(current_full, 3);

  rows[0].q_values[0] = std::numeric_limits<double>::quiet_NaN();
  SPIEL_CHECK_FALSE(WriteScratchSearchPiRowShardV2(
      (directory / "bad.bin").string(), rows, &error));
  std::filesystem::remove_all(directory);
}

void TestActingPlayerInformationBoundary() {
  auto game = LoadGame("dune_imperium");
  auto base = game->NewInitialState();
  while (base->IsChanceNode()) {
    const ActionsAndProbs outcomes = base->ChanceOutcomes();
    SPIEL_CHECK_FALSE(outcomes.empty());
    base->ApplyAction(outcomes.front().first);
  }
  const Player actor = base->CurrentPlayer();
  SPIEL_CHECK_GE(actor, 0);
  const Player opponent = (actor + 1) % game->NumPlayers();
  auto first = base->Clone();
  auto second = base->Clone();
  auto* first_dune = static_cast<dune_imperium::DuneImperiumState*>(first.get());
  auto* second_dune =
      static_cast<dune_imperium::DuneImperiumState*>(second.get());
  first_dune->SetPlayerHandForTesting(opponent, {1, 2, 3});
  second_dune->SetPlayerHandForTesting(opponent, {4, 5, 6});
  first_dune->SetPlayerDeckForTesting(opponent, {7, 8});
  second_dune->SetPlayerDeckForTesting(opponent, {9, 10});
  const std::vector<float> first_tensor = first->InformationStateTensor(actor);
  const std::vector<float> second_tensor = second->InformationStateTensor(actor);
  SPIEL_CHECK_EQ(first_tensor.size(), second_tensor.size());
  SPIEL_CHECK_EQ(std::memcmp(first_tensor.data(), second_tensor.data(),
                             first_tensor.size() * sizeof(float)),
                 0);
}

void TestScratchOneGameChunksPreserveBalancedGeneration() {
  auto game = LoadGame("dune_imperium");
  auto model = std::make_shared<SharedDunePolicyValueNetImpl>(
      game->InformationStateTensorSize(), 16, game->NumDistinctActions(), 1,
      false);
  model->eval();
  SearchPiConfig search = ScratchConfig();
  search.games_per_generation = 4;
  search.next_episode_id = 4000;
  search.primary_simulations = 2;
  search.continuation_simulations = 1;
  search.purchase_combat_budget = 1;
  search.scratch_full_root_probability = 0.0;
  search.scratch_cheap_primary_simulations = 2;
  search.scratch_cheap_other_simulations = 1;
  search.behavior_temperature = 1.0;
  const std::filesystem::path directory =
      std::filesystem::temp_directory_path() /
      ("searchpi_scratch_chunks_" + std::to_string(::getpid()));
  ConcurrentSearchPiCollectionConfig config;
  config.search = search;
  config.generation = 1;
  config.collection_games = 4;
  config.chunk_games = 1;
  config.requested_workers = 4;
  config.batch_target = 0;
  config.retain_rows = true;
  config.output_dir = directory.string();
  const auto result = CollectSearchPiConcurrent(config, game, model,
                                                 torch::Device(torch::kCPU));
  SPIEL_CHECK_EQ(result.actual_workers, 4);
  SPIEL_CHECK_EQ(result.games.size(), 4u);
  bool seats[4] = {false, false, false, false};
  for (const SearchPiGameOutcome& outcome : result.games) {
    SPIEL_CHECK_GE(outcome.searched_seat, 0);
    SPIEL_CHECK_LT(outcome.searched_seat, 4);
    seats[outcome.searched_seat] = true;
  }
  for (bool seen : seats) SPIEL_CHECK_TRUE(seen);
  std::filesystem::remove_all(directory);
}

}  // namespace
}  // namespace open_spiel

int main() {
  open_spiel::TestRegularizedQ();
  open_spiel::TestBudgets();
  open_spiel::TestLearnerMaskingAndTarget();
  open_spiel::TestV2AndReplay();
  open_spiel::TestActingPlayerInformationBoundary();
  open_spiel::TestScratchOneGameChunksPreserveBalancedGeneration();
  std::cout << "dune_search_pi_scratch_test: PASS\n";
  return 0;
}
