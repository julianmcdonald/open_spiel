// Contract tests for the shared concurrent Search-PI collector boundary.
//
// These tests use synthetic completed collections. They exercise the
// production validator without playing games or touching a GPU, including the
// outcome-blind 512->64 split and the narrowly-scoped D2 filler exception.

#include "dune_search_pi_concurrent.h"

#include <cstdint>
#include <iostream>
#include <string>

#include "open_spiel/abseil-cpp/absl/flags/flag.h"
#include "open_spiel/spiel_utils.h"

// dune_ppo_training_utils.cc declares these flags; the trainer normally
// defines them. This focused test links the utility translation unit and must
// therefore provide the same harmless definitions as dune_search_pi_test.
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

constexpr int64_t kFirst = 600000;

SearchPiRow MakeRow(int64_t episode_id, DuneDecisionRole role, int budget) {
  SearchPiRow row;
  row.observation = {static_cast<float>(episode_id - kFirst), 1.0f};
  row.player = static_cast<Player>(episode_id % 4);
  row.role = role;
  row.legal_actions = {1, 2};
  row.raw_policy = {0.75, 0.25};
  row.raw_visits = {budget, 0};
  row.target_visits = {budget, 0};
  row.target_probs = {1.0, 0.0};
  row.chosen_action = 1;
  row.value_target = 0.25;
  row.value_target_attached = true;
  row.generation = 1;
  row.episode_id = episode_id;
  row.decision_id = 1;
  row.state_fingerprint = "synthetic-state-" + std::to_string(episode_id);
  row.simulations_completed = budget;
  row.simulations_requested = budget;
  return row;
}

ConcurrentSearchPiCollectionResult MakeValidCollection(bool wide) {
  ConcurrentSearchPiCollectionResult result;
  result.requested_workers = 128;
  result.actual_workers = 128;
  result.configured_batch_target = 64;
  result.configured_batcher_timeout_ms = 2;
  result.max_inflight_rows = 512;
  result.telemetry_valid = true;
  result.telemetry_device_applicable = false;
  result.telemetry.submitted_rows = 1024;
  result.telemetry.single_row_calls = 1024;
  result.telemetry.physical_batches = 16;
  result.telemetry.max_batch_size = 64;
  result.telemetry.target_batch_size = 64;
  result.telemetry.timeout_ms = 2;
  result.primary.configured_time_limit_ms = 60000.0;
  result.continuation.configured_time_limit_ms = 60000.0;
  if (wide) {
    result.purchase.configured_time_limit_ms = 60000.0;
    result.combat_intrigue.configured_time_limit_ms = 60000.0;
    result.other_optional.configured_time_limit_ms = 60000.0;
  }

  for (int64_t episode = kFirst; episode < kFirst + 512; ++episode) {
    SearchPiGameOutcome outcome;
    outcome.episode_id = episode;
    outcome.primary_roots = 1;
    outcome.primary_simulations = 200;
    outcome.primary_simulations_requested = 200;
    outcome.continuation_roots = 1;
    outcome.continuation_simulations = 64;
    outcome.continuation_simulations_requested = 64;
    outcome.roots_played_differs_from_raw[2] = 1;
    outcome.roots_played_differs_from_raw[3] = 1;
    if (wide) {
      outcome.wide_roots = 3;
      outcome.wide_simulations = 3 * 64;
      outcome.wide_simulations_requested = 3 * 64;
      outcome.roots_played_differs_from_raw[4] = 1;
      outcome.roots_played_differs_from_raw[5] = 1;
      outcome.roots_played_differs_from_raw[6] = 1;
    }
    result.games.push_back(outcome);
    result.retained_rows.push_back(
        MakeRow(episode, DuneDecisionRole::kAgentPrimary, 200));
    if (wide) {
      result.retained_rows.push_back(
          MakeRow(episode, DuneDecisionRole::kPurchase, 64));
    }
  }
  result.rows_total = result.retained_rows.size();
  result.primary.roots_seen = 512;
  result.primary.simulations_completed = 512 * 200;
  result.primary.simulations_requested = 512 * 200;
  result.continuation.roots_seen = 512;
  result.continuation.simulations_completed = 512 * 64;
  result.continuation.simulations_requested = 512 * 64;
  result.simulations_by_role[2] = 512 * 200;
  result.simulations_by_role[3] = 512 * 64;
  result.decisions_by_role[2] = 512;
  result.decisions_by_role[3] = 512;
  if (wide) {
    result.purchase.roots_seen = 512;
    result.purchase.simulations_completed = 512 * 64;
    result.purchase.simulations_requested = 512 * 64;
    result.combat_intrigue.roots_seen = 512;
    result.combat_intrigue.simulations_completed = 512 * 64;
    result.combat_intrigue.simulations_requested = 512 * 64;
    result.other_optional.roots_seen = 512;
    result.other_optional.simulations_completed = 512 * 64;
    result.other_optional.simulations_requested = 512 * 64;
    result.simulations_by_role[4] = 512 * 64;
    result.simulations_by_role[5] = 512 * 64;
    result.simulations_by_role[6] = 512 * 64;
    result.decisions_by_role[4] = 512;
    result.decisions_by_role[5] = 512;
    result.decisions_by_role[6] = 512;
  }
  return result;
}

SearchPiTrainingCollectionContract Contract(bool wide) {
  SearchPiTrainingCollectionContract contract;
  contract.first_episode_id = kFirst;
  contract.purchase_combat_budget = wide ? 64 : 0;
  return contract;
}

void MarkFillerTimeout(ConcurrentSearchPiCollectionResult* result,
                       int64_t episode_id) {
  SearchPiGameOutcome& outcome =
      result->games.at(static_cast<size_t>(episode_id - kFirst));
  outcome.primary_simulations = 199;
  outcome.watchdog_timeouts = 1;
  outcome.searches_short_of_budget = 1;
  result->primary.simulations_completed--;
  result->primary.searches_short_of_budget++;
  result->primary.simulation_shortfall++;
  result->primary.early_exit_counts[
      static_cast<int>(SearchPiEarlyExit::kTimeout)]++;
  result->simulations_by_role[2]--;
}

void TestOutcomeBlindPrefixAndNarrowScope() {
  auto result = MakeValidCollection(false);
  const auto validation =
      ValidateSearchPiTrainingCollection(result, Contract(false));
  SPIEL_CHECK_TRUE(validation.ok);
  SPIEL_CHECK_EQ(validation.training_rows.size(), 64);
  SPIEL_CHECK_EQ(validation.trained_rows_by_role[2], 64);
  SPIEL_CHECK_EQ(validation.trained_rows_by_role[4], 0);
  for (const SearchPiRow& row : validation.training_rows) {
    SPIEL_CHECK_GE(row.episode_id, kFirst);
    SPIEL_CHECK_LT(row.episode_id, kFirst + 64);
  }
  SPIEL_CHECK_FALSE(validation.collected_target_hash.empty());
  SPIEL_CHECK_FALSE(validation.trained_target_hash.empty());
  SPIEL_CHECK_NE(validation.collected_target_hash,
                 validation.trained_target_hash);
}

void TestWideRoleReachesOnlyRegisteredPrefix() {
  auto result = MakeValidCollection(true);
  const auto validation =
      ValidateSearchPiTrainingCollection(result, Contract(true));
  SPIEL_CHECK_TRUE(validation.ok);
  SPIEL_CHECK_EQ(validation.training_rows.size(), 128);
  SPIEL_CHECK_EQ(validation.trained_rows_by_role[2], 64);
  SPIEL_CHECK_EQ(validation.trained_rows_by_role[4], 64);
}

void TestD2FillerTimeoutCapAndTrainingStop() {
  auto two = MakeValidCollection(false);
  MarkFillerTimeout(&two, kFirst + 64);
  MarkFillerTimeout(&two, kFirst + 511);
  auto validation = ValidateSearchPiTrainingCollection(two, Contract(false));
  SPIEL_CHECK_TRUE(validation.ok);
  SPIEL_CHECK_EQ(validation.filler_timeout_episode_ids.size(), 2);

  auto three = two;
  MarkFillerTimeout(&three, kFirst + 100);
  validation = ValidateSearchPiTrainingCollection(three, Contract(false));
  SPIEL_CHECK_FALSE(validation.ok);

  auto training = MakeValidCollection(false);
  MarkFillerTimeout(&training, kFirst + 63);
  validation = ValidateSearchPiTrainingCollection(training, Contract(false));
  SPIEL_CHECK_FALSE(validation.ok);
}

void TestEveryOtherFailureStopsBeforeLearnerBoundary() {
  auto non_timeout = MakeValidCollection(false);
  non_timeout.games[400].non_timeout_early_exits = 1;
  SPIEL_CHECK_FALSE(
      ValidateSearchPiTrainingCollection(non_timeout, Contract(false)).ok);

  auto worker = MakeValidCollection(false);
  worker.actual_workers = 127;
  SPIEL_CHECK_FALSE(
      ValidateSearchPiTrainingCollection(worker, Contract(false)).ok);

  auto batcher = MakeValidCollection(false);
  batcher.telemetry.leaf_groups_split = 1;
  SPIEL_CHECK_FALSE(
      ValidateSearchPiTrainingCollection(batcher, Contract(false)).ok);

  auto order = MakeValidCollection(false);
  order.games[12].episode_id++;
  SPIEL_CHECK_FALSE(
      ValidateSearchPiTrainingCollection(order, Contract(false)).ok);

  auto scope = MakeValidCollection(false);
  scope.simulations_by_role[4] = 64;
  SPIEL_CHECK_FALSE(
      ValidateSearchPiTrainingCollection(scope, Contract(false)).ok);
}

void TestCanonicalCollectorDigest() {
  torch::manual_seed(7);
  SharedDunePolicyValueNetImpl model(/*input_size=*/8, /*hidden_size=*/8,
                                     /*output_size=*/4, /*num_blocks=*/1);
  const std::string before = CanonicalSearchPiModuleDigest(model);
  const std::string again = CanonicalSearchPiModuleDigest(model);
  SPIEL_CHECK_EQ(before, again);
  {
    torch::NoGradGuard no_grad;
    model.input_layer->weight.add_(1.0);
  }
  SPIEL_CHECK_NE(before, CanonicalSearchPiModuleDigest(model));
}

}  // namespace
}  // namespace open_spiel

int main() {
  open_spiel::TestOutcomeBlindPrefixAndNarrowScope();
  open_spiel::TestWideRoleReachesOnlyRegisteredPrefix();
  open_spiel::TestD2FillerTimeoutCapAndTrainingStop();
  open_spiel::TestEveryOtherFailureStopsBeforeLearnerBoundary();
  open_spiel::TestCanonicalCollectorDigest();
  std::cout << "All shared concurrent Search-PI contract tests passed!\n";
  return 0;
}
