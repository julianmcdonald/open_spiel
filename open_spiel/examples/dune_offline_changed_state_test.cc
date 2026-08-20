#include "dune_offline_changed_state.h"

#include <cmath>
#include <iostream>
#include <string>
#include <vector>

#include "open_spiel/abseil-cpp/absl/flags/flag.h"
#include "open_spiel/spiel_utils.h"

#include "dune_search_pi.h"

// Link-only definitions required by dune_ppo_training_utils.cc.
ABSL_FLAG(int, ppo_minibatch_size, 2048, "INERT test link flag");
ABSL_FLAG(int, ppo_update_epochs, 4, "INERT test link flag");
ABSL_FLAG(double, ppo_clip_epsilon, 0.2, "INERT test link flag");
ABSL_FLAG(bool, normalize_advantages, true, "INERT test link flag");
ABSL_FLAG(bool, ppo_clip_value_loss, true, "INERT test link flag");
ABSL_FLAG(double, entropy_coef, 0.01, "INERT test link flag");
ABSL_FLAG(double, value_coef, 0.5, "INERT test link flag");
ABSL_FLAG(double, target_kl, 0.0, "INERT test link flag");
ABSL_FLAG(bool, train_amp, true, "INERT test link flag");
ABSL_FLAG(double, grad_clip_norm, 0.5, "INERT test link flag");
ABSL_FLAG(double, logit_cap, 10.0, "INERT test link flag");
ABSL_FLAG(uint64_t, shaping_start_env_steps, 206830543,
          "INERT test link flag");
ABSL_FLAG(uint64_t, shaping_decay_env_steps, 0, "INERT test link flag");
ABSL_FLAG(bool, diagnostics_only, false, "INERT test link flag");

namespace open_spiel {
namespace {

SearchPiRow MakeRow(int64_t input_dim, int64_t episode,
                    DuneDecisionRole role, bool changed) {
  SearchPiRow row;
  row.observation.resize(input_dim);
  for (int64_t i = 0; i < input_dim; ++i) {
    row.observation[i] = static_cast<float>((episode + i) % 11) / 10.0f;
  }
  row.observation_is_information_state = true;
  row.player = 0;
  row.role = role;
  row.legal_actions = {0, 1, 2, 3};
  row.raw_policy = {0.4, 0.3, 0.2, 0.1};
  row.raw_visits = {4, 3, 2, 1};
  row.target_visits = changed ? std::vector<int>{1, 7, 1, 1}
                              : std::vector<int>{7, 1, 1, 1};
  row.target_probs = changed ? std::vector<double>{0.1, 0.7, 0.1, 0.1}
                             : std::vector<double>{0.7, 0.1, 0.1, 0.1};
  row.chosen_action = changed ? 1 : 0;
  row.value_target = 0.0625;
  row.value_target_attached = true;
  row.generation = 1;
  row.episode_id = episode;
  row.decision_id = 0;
  row.state_fingerprint = std::string(64, changed ? 'a' : 'b');
  row.simulations_completed = 200;
  row.simulations_requested = 200;
  row.target_argmax_differs_from_raw = changed;
  FillSearchPiRowScalars(&row);
  return row;
}

void TestRegisteredCountsAndWeights() {
  const auto& registered = RegisteredOfflineRoleSpecs();
  int64_t total = 0, changed = 0, preserved = 0;
  for (const OfflineRoleSpec& role : registered) {
    total += role.total;
    changed += role.changed;
    preserved += role.preserved;
  }
  SPIEL_CHECK_EQ(total, kOfflineChangedStateRows);
  SPIEL_CHECK_EQ(changed, 9720);
  SPIEL_CHECK_EQ(preserved, 82659);

  const std::vector<OfflineRoleSpec> small = {
      {DuneDecisionRole::kAgentPrimary, "agent_primary", 3, 1, 2},
      {DuneDecisionRole::kPurchase, "purchase", 2, 1, 1},
  };
  std::vector<SearchPiRow> rows = {
      MakeRow(8, 0, DuneDecisionRole::kAgentPrimary, true),
      MakeRow(8, 1, DuneDecisionRole::kAgentPrimary, false),
      MakeRow(8, 2, DuneDecisionRole::kAgentPrimary, false),
      MakeRow(8, 3, DuneDecisionRole::kPurchase, true),
      MakeRow(8, 4, DuneDecisionRole::kPurchase, false),
  };
  OfflineWeightPlan plan;
  std::string error;
  SPIEL_CHECK_TRUE(BuildOfflineWeightPlan(rows, small, &plan, &error));
  SPIEL_CHECK_EQ(plan.control.size(), rows.size());
  SPIEL_CHECK_EQ(plan.treatment.size(), rows.size());
  SPIEL_CHECK_FLOAT_EQ(plan.treatment[0], 1.5);
  SPIEL_CHECK_FLOAT_EQ(plan.treatment[1], 0.75);
  SPIEL_CHECK_FLOAT_EQ(plan.treatment[2], 0.75);
  SPIEL_CHECK_FLOAT_EQ(plan.treatment[3], 1.0);
  SPIEL_CHECK_FLOAT_EQ(plan.treatment[4], 1.0);
  SPIEL_CHECK_FLOAT_EQ(plan.treatment_mass, 5.0);
  SPIEL_CHECK_FLOAT_EQ(plan.roles[0].changed_loss_mass, 1.5);
  SPIEL_CHECK_FLOAT_EQ(plan.roles[0].preserved_loss_mass, 1.5);
  rows[0].role = DuneDecisionRole::kOtherOptional;
  SPIEL_CHECK_FALSE(BuildOfflineWeightPlan(rows, small, &plan, &error));
}

void TestOrderHashesAndLastMinibatch() {
  std::vector<SearchPiRow> rows = {
      MakeRow(8, 10, DuneDecisionRole::kAgentPrimary, true),
      MakeRow(8, 11, DuneDecisionRole::kAgentPrimary, false),
  };
  std::string target_a, extended_a;
  for (const SearchPiRow& row : rows) {
    target_a = ChainSearchPiTargetHash(target_a, row);
    extended_a = ChainSearchPiExtendedRowHash(extended_a, row);
  }
  std::reverse(rows.begin(), rows.end());
  std::string target_b, extended_b;
  for (const SearchPiRow& row : rows) {
    target_b = ChainSearchPiTargetHash(target_b, row);
    extended_b = ChainSearchPiExtendedRowHash(extended_b, row);
  }
  SPIEL_CHECK_NE(target_a, target_b);
  SPIEL_CHECK_NE(extended_a, extended_b);

  const auto boundaries = BuildOfflineBatchBoundaries(92379, 256);
  SPIEL_CHECK_EQ(boundaries.size(), 361);
  SPIEL_CHECK_EQ(boundaries.front().start, 0);
  SPIEL_CHECK_EQ(boundaries.front().count, 256);
  SPIEL_CHECK_EQ(boundaries.back().start, 92160);
  SPIEL_CHECK_EQ(boundaries.back().count, 219);
  const std::string digest =
      OfflinePresentationDigest(extended_a, boundaries, 8260001);
  SPIEL_CHECK_EQ(digest,
                 OfflinePresentationDigest(extended_a, boundaries, 8260001));
  SPIEL_CHECK_NE(digest,
                 OfflinePresentationDigest(extended_b, boundaries, 8260001));
  SPIEL_CHECK_NE(digest,
                 OfflinePresentationDigest(extended_a, boundaries, 8260002));
}

void TestFreshOptimizerBoundaryAndTraining() {
  torch::manual_seed(1234);
  auto control = std::make_shared<SharedDunePolicyValueNetImpl>(
      8, 16, 4, 1, /*use_nonlinear=*/false);
  torch::manual_seed(1234);
  auto treatment = std::make_shared<SharedDunePolicyValueNetImpl>(
      8, 16, 4, 1, /*use_nonlinear=*/false);
  auto control_optimizer = MakeOfflineFreshAdamW(control, 1e-2, 1e-5, 0, 0);
  auto treatment_optimizer =
      MakeOfflineFreshAdamW(treatment, 1e-2, 1e-5, 0, 0);
  const std::string start_digest = CanonicalOfflineModuleDigest(*control);
  std::string error;
  SPIEL_CHECK_TRUE(ValidateFreshOfflineBoundary(
      *control, *control_optimizer, *treatment, *treatment_optimizer,
      start_digest, &error));
  SPIEL_CHECK_TRUE(control_optimizer->state().empty());
  SPIEL_CHECK_TRUE(treatment_optimizer->state().empty());
  SPIEL_CHECK_TRUE(RefuseOfflineOptimizerCheckpoint("", &error));
  SPIEL_CHECK_FALSE(RefuseOfflineOptimizerCheckpoint("retained_lr0.pt", &error));

  std::vector<SearchPiRow> rows = {
      MakeRow(8, 0, DuneDecisionRole::kAgentPrimary, true),
      MakeRow(8, 1, DuneDecisionRole::kAgentPrimary, false),
      MakeRow(8, 2, DuneDecisionRole::kAgentPrimary, true),
  };
  const std::vector<double> weights(rows.size(), 1.0);
  const auto boundaries = BuildOfflineBatchBoundaries(rows.size(), 256);
  const OfflineTrainingStats stats = TrainOfflineChangedStateArm(
      control, *control_optimizer, rows, weights, boundaries, 8, 4,
      torch::Device(torch::kCPU), 8260001, 0.5, 10.0);
  SPIEL_CHECK_EQ(stats.presentations, 3);
  SPIEL_CHECK_EQ(stats.minibatches, 1);
  SPIEL_CHECK_EQ(stats.final_minibatch_size, 3);
  SPIEL_CHECK_FALSE(control_optimizer->state().empty());
  SPIEL_CHECK_TRUE(treatment_optimizer->state().empty());
  SPIEL_CHECK_NE(CanonicalOfflineModuleDigest(*control), start_digest);
  SPIEL_CHECK_EQ(CanonicalOfflineModuleDigest(*treatment), start_digest);
}

void TestOfflineGate() {
  std::vector<SearchPiRow> rows;
  OfflineModelMetrics parent, control, treatment;
  int64_t episode = 0;
  for (const OfflineRoleSpec& spec : RegisteredOfflineRoleSpecs()) {
    for (int i = 0; i < 3; ++i) {
      rows.push_back(MakeRow(8, episode++, spec.role, true));
      parent.ce.push_back(1.3);
      control.ce.push_back(1.0);
      treatment.ce.push_back(0.8);
      parent.adopted.push_back(0);
      control.adopted.push_back(0);
      treatment.adopted.push_back(1);
    }
    for (int i = 0; i < 2; ++i) {
      rows.push_back(MakeRow(8, episode++, spec.role, false));
      parent.ce.push_back(0.4);
      control.ce.push_back(0.3);
      treatment.ce.push_back(0.301);
      parent.adopted.push_back(1);
      control.adopted.push_back(1);
      treatment.adopted.push_back(1);
    }
  }
  const OfflineGateResult pass = ApplyOfflineChangedStateGate(
      rows, parent, control, treatment, true);
  SPIEL_CHECK_TRUE(pass.condition_1_changed_fit_and_adoption);
  SPIEL_CHECK_TRUE(pass.condition_2_all_roles_lower_changed_ce);
  SPIEL_CHECK_TRUE(pass.condition_3_preserved_noninferiority);
  SPIEL_CHECK_TRUE(pass.condition_4_integrity_assertions);
  SPIEL_CHECK_TRUE(pass.passed);
  SPIEL_CHECK_LT(std::abs(OneSidedPairedExactP(5, 0) - 0.03125), 1e-12);
  SPIEL_CHECK_FLOAT_EQ(OneSidedPairedExactP(0, 0), 1.0);

  // A role guard is independent of the pooled changed-row result.
  treatment.ce[0] = 1.2;
  treatment.ce[1] = 1.2;
  treatment.ce[2] = 1.2;
  const OfflineGateResult fail = ApplyOfflineChangedStateGate(
      rows, parent, control, treatment, true);
  SPIEL_CHECK_FALSE(fail.condition_2_all_roles_lower_changed_ce);
  SPIEL_CHECK_FALSE(fail.passed);
}

}  // namespace
}  // namespace open_spiel

int main() {
  open_spiel::TestRegisteredCountsAndWeights();
  open_spiel::TestOrderHashesAndLastMinibatch();
  open_spiel::TestFreshOptimizerBoundaryAndTraining();
  open_spiel::TestOfflineGate();
  std::cout << "dune_offline_changed_state_test passed\n";
  return 0;
}
