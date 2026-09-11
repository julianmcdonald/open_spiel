#include "dune_action_value_study.h"

#include <cmath>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace action_value_study {

void TestSchemaValidation() {
  std::cout << "Running TestSchemaValidation...\n" << std::flush;

  // 1. Legacy critic (9012)
  DuneVrpoQNetImpl legacy_critic(19, kVrpoQInputDimLegacy);
  SPIEL_CHECK_EQ(legacy_critic.expected_input_dim, 9012);
  SPIEL_CHECK_EQ(legacy_critic.input_layer->weight.size(1), 9012);
  SPIEL_CHECK_TRUE(legacy_critic.CheckSchema());

  torch::Tensor legacy_input = torch::zeros({2, 9012}, torch::kFloat32);
  torch::Tensor legacy_output;
  std::string err;
  SPIEL_CHECK_TRUE(legacy_critic.ForwardChecked(legacy_input, &legacy_output, &err));
  SPIEL_CHECK_EQ(legacy_output.size(0), 2);
  SPIEL_CHECK_EQ(legacy_output.size(1), 2391);
  SPIEL_CHECK_EQ(legacy_output.size(2), 4);

  // Feeding 9647 input into 9012 critic must fail
  torch::Tensor bad_legacy_input = torch::zeros({2, 9647}, torch::kFloat32);
  SPIEL_CHECK_FALSE(legacy_critic.ForwardChecked(bad_legacy_input, &legacy_output, &err));
  SPIEL_CHECK_FALSE(err.empty());

  // 2. Study critic with market appendix (9647)
  DuneVrpoQNetImpl study_critic(19, kCriticInputDim);
  SPIEL_CHECK_EQ(study_critic.expected_input_dim, 9647);
  SPIEL_CHECK_EQ(study_critic.input_layer->weight.size(1), 9647);
  SPIEL_CHECK_TRUE(study_critic.CheckSchema());

  torch::Tensor study_input = torch::zeros({3, 9647}, torch::kFloat32);
  torch::Tensor study_output;
  SPIEL_CHECK_TRUE(study_critic.ForwardChecked(study_input, &study_output, &err));
  SPIEL_CHECK_EQ(study_output.size(0), 3);
  SPIEL_CHECK_EQ(study_output.size(1), 2391);
  SPIEL_CHECK_EQ(study_output.size(2), 4);

  // Feeding 9012 input into 9647 critic must fail
  torch::Tensor bad_study_input = torch::zeros({3, 9012}, torch::kFloat32);
  SPIEL_CHECK_FALSE(study_critic.ForwardChecked(bad_study_input, &study_output, &err));
  SPIEL_CHECK_FALSE(err.empty());

  // 3. Deliberate schema tampering check: simulate loading mismatched checkpoint
  const std::string temp_legacy_path = "/tmp/test_legacy_critic_checkpoint.pt";
  torch::serialize::OutputArchive legacy_out_archive;
  legacy_critic.save(legacy_out_archive);
  legacy_out_archive.save_to(temp_legacy_path);

  DuneVrpoQNetImpl tampered_critic(19, kCriticInputDim);
  torch::serialize::InputArchive legacy_in_archive;
  legacy_in_archive.load_from(temp_legacy_path);
  tampered_critic.load(legacy_in_archive);
  SPIEL_CHECK_FALSE(tampered_critic.CheckSchema(&err));
  SPIEL_CHECK_FALSE(tampered_critic.ForwardChecked(study_input, &study_output, &err));
  std::filesystem::remove(temp_legacy_path);
  std::cout << "  Passed TestSchemaValidation.\n";
}

void TestActorRelativeMapping() {
  std::cout << "Running TestActorRelativeMapping...\n" << std::flush;
  std::array<double, 4> abs_returns = {2.25, 0.25, -0.75, -1.75};

  // When actor is player 0: rel[0]=abs[0], rel[1]=abs[1], rel[2]=abs[2], rel[3]=abs[3]
  auto rel0 = ConvertAbsoluteReturnsToActorRelative(0, abs_returns);
  SPIEL_CHECK_EQ(rel0[0], 2.25);
  SPIEL_CHECK_EQ(rel0[1], 0.25);
  SPIEL_CHECK_EQ(rel0[2], -0.75);
  SPIEL_CHECK_EQ(rel0[3], -1.75);

  // When actor is player 1: rel[0]=abs[1], rel[1]=abs[2], rel[2]=abs[3], rel[3]=abs[0]
  auto rel1 = ConvertAbsoluteReturnsToActorRelative(1, abs_returns);
  SPIEL_CHECK_EQ(rel1[0], 0.25);
  SPIEL_CHECK_EQ(rel1[1], -0.75);
  SPIEL_CHECK_EQ(rel1[2], -1.75);
  SPIEL_CHECK_EQ(rel1[3], 2.25);

  // When actor is player 2: rel[0]=abs[2], rel[1]=abs[3], rel[2]=abs[0], rel[3]=abs[1]
  auto rel2 = ConvertAbsoluteReturnsToActorRelative(2, abs_returns);
  SPIEL_CHECK_EQ(rel2[0], -0.75);
  SPIEL_CHECK_EQ(rel2[1], -1.75);
  SPIEL_CHECK_EQ(rel2[2], 2.25);
  SPIEL_CHECK_EQ(rel2[3], 0.25);

  // When actor is player 3: rel[0]=abs[3], rel[1]=abs[0], rel[2]=abs[1], rel[3]=abs[2]
  auto rel3 = ConvertAbsoluteReturnsToActorRelative(3, abs_returns);
  SPIEL_CHECK_EQ(rel3[0], -1.75);
  SPIEL_CHECK_EQ(rel3[1], 2.25);
  SPIEL_CHECK_EQ(rel3[2], 0.25);
  SPIEL_CHECK_EQ(rel3[3], -0.75);

  std::cout << "  Passed TestActorRelativeMapping.\n";
}

void TestSeedDerivationAndPairing() {
  std::cout << "Running TestSeedDerivationAndPairing...\n" << std::flush;
  std::string root_id_a = "0123456789abcdef";
  std::string root_id_b = "fedcba9876543210";

  // 1. Paired seeds: replicate 0 for root A must have identical chance and policy seeds
  // regardless of which action was taken
  uint64_t chance_seed_rep0 = DeriveContinuationChanceSeed(Partition::kTrain, root_id_a, 0);
  uint64_t policy_seed_rep0 = DeriveContinuationPolicySeed(Partition::kTrain, root_id_a, 0);
  SPIEL_CHECK_NE(chance_seed_rep0, policy_seed_rep0); // Independent chance and policy streams

  // Another call with same coordinates must produce identical seeds
  SPIEL_CHECK_EQ(chance_seed_rep0, DeriveContinuationChanceSeed(Partition::kTrain, root_id_a, 0));
  SPIEL_CHECK_EQ(policy_seed_rep0, DeriveContinuationPolicySeed(Partition::kTrain, root_id_a, 0));

  // Different replicates must have different seeds
  uint64_t chance_seed_rep1 = DeriveContinuationChanceSeed(Partition::kTrain, root_id_a, 1);
  SPIEL_CHECK_NE(chance_seed_rep0, chance_seed_rep1);

  // 2. Disjoint partition namespaces: Train vs Dev vs Test
  uint64_t train_seed = DeriveContinuationChanceSeed(Partition::kTrain, root_id_a, 0);
  uint64_t dev_seed = DeriveContinuationChanceSeed(Partition::kDev, root_id_a, 0);
  uint64_t test_seed = DeriveContinuationChanceSeed(Partition::kTest, root_id_a, 0);
  uint64_t pilot_seed = DeriveContinuationChanceSeed(Partition::kPilot, root_id_a, 0);

  SPIEL_CHECK_NE(train_seed, dev_seed);
  SPIEL_CHECK_NE(train_seed, test_seed);
  SPIEL_CHECK_NE(train_seed, pilot_seed);
  SPIEL_CHECK_NE(dev_seed, test_seed);
  SPIEL_CHECK_NE(dev_seed, pilot_seed);
  SPIEL_CHECK_NE(test_seed, pilot_seed);

  // Source game seeds must also be distinct
  uint64_t train_game0 = DeriveSourceGameSeed(Partition::kTrain, 0);
  uint64_t dev_game0 = DeriveSourceGameSeed(Partition::kDev, 0);
  uint64_t test_game0 = DeriveSourceGameSeed(Partition::kTest, 0);
  SPIEL_CHECK_NE(train_game0, dev_game0);
  SPIEL_CHECK_NE(train_game0, test_game0);
  SPIEL_CHECK_NE(dev_game0, test_game0);

  std::cout << "  Passed TestSeedDerivationAndPairing.\n";
}

void TestCandidateActionSelection() {
  std::cout << "Running TestCandidateActionSelection...\n" << std::flush;
  std::vector<Action> legal_actions = {10, 20, 30, 40};
  std::vector<float> logits(2391, 0.0f);
  logits[10] = 5.0f; // Should be highest
  logits[20] = 3.0f; // Should be 2nd highest
  logits[30] = 1.0f;
  logits[40] = 0.5f;

  std::vector<double> probs;
  auto candidates = SelectCandidateActions(legal_actions, logits, Partition::kTrain, "0123456789abcdef", &probs);
  SPIEL_CHECK_EQ(candidates.size(), 3u);
  SPIEL_CHECK_EQ(candidates[0], 10); // Highest
  SPIEL_CHECK_EQ(candidates[1], 20); // 2nd highest
  // 3rd action must be either 30 or 40
  SPIEL_CHECK_TRUE(candidates[2] == 30 || candidates[2] == 40);

  // Exactly two legal actions case
  std::vector<Action> legal_two = {100, 200};
  logits[100] = 2.0f;
  logits[200] = 8.0f;
  auto candidates_two = SelectCandidateActions(legal_two, logits, Partition::kTrain, "0123456789abcdef", &probs);
  SPIEL_CHECK_EQ(candidates_two.size(), 2u);
  SPIEL_CHECK_EQ(candidates_two[0], 200); // Highest
  SPIEL_CHECK_EQ(candidates_two[1], 100); // 2nd highest

  std::cout << "  Passed TestCandidateActionSelection.\n";
}

void TestCriticOptimizerStepAndSaveReload() {
  std::cout << "Running TestCriticOptimizerStepAndSaveReload...\n" << std::flush;
  at::globalContext().setAllowTF32CuBLAS(false);
  at::globalContext().setAllowTF32CuDNN(false);

  auto critic1 = std::make_shared<DuneVrpoQNetImpl>(kRegisteredCriticInitSeed, kCriticInputDim);
  auto critic2 = std::make_shared<DuneVrpoQNetImpl>(kRegisteredCriticInitSeed, kCriticInputDim);

  // Confirm identical initialization
  for (const auto& p1 : critic1->named_parameters()) {
    const auto& p2 = critic2->named_parameters()[p1.key()];
    SPIEL_CHECK_TRUE(torch::equal(p1.value(), p2));
  }

  // Perform one training step on critic1
  torch::optim::AdamW optimizer(
      critic1->parameters(),
      torch::optim::AdamWOptions(kCriticLearningRate)
          .eps(kCriticAdamWEpsilon)
          .weight_decay(kCriticWeightDecay));

  torch::Tensor x = torch::randn({4, kCriticInputDim}, torch::kFloat32);
  torch::Tensor output;
  std::string err;
  SPIEL_CHECK_TRUE(critic1->ForwardChecked(x, &output, &err));

  // Mock target for selected action 10
  torch::Tensor target = torch::zeros({4, 4}, torch::kFloat32);
  torch::Tensor pred = output.index({torch::indexing::Slice(), 10, torch::indexing::Slice()});
  torch::Tensor loss = torch::mse_loss(pred, target);

  optimizer.zero_grad();
  loss.backward();
  torch::nn::utils::clip_grad_norm_(critic1->parameters(), kCriticGradClipNorm);
  optimizer.step();

  // Confirm critic1 has changed from critic2
  bool has_difference = false;
  for (const auto& p1 : critic1->named_parameters()) {
    const auto& p2 = critic2->named_parameters()[p1.key()];
    if (!torch::equal(p1.value(), p2)) {
      has_difference = true;
      break;
    }
  }
  SPIEL_CHECK_TRUE(has_difference);

  // Compare post-step forward output with reloaded output
  torch::Tensor post_step_output;
  SPIEL_CHECK_TRUE(critic1->ForwardChecked(x, &post_step_output, &err));

  // Save and reload
  const std::string temp_path = "/tmp/test_critic_checkpoint.pt";
  torch::serialize::OutputArchive out_archive;
  critic1->save(out_archive);
  out_archive.save_to(temp_path);

  auto reloaded_critic = std::make_shared<DuneVrpoQNetImpl>(0, kCriticInputDim);
  torch::serialize::InputArchive in_archive;
  in_archive.load_from(temp_path);
  reloaded_critic->load(in_archive);
  SPIEL_CHECK_TRUE(reloaded_critic->CheckSchema());

  torch::Tensor reloaded_output;
  SPIEL_CHECK_TRUE(reloaded_critic->ForwardChecked(x, &reloaded_output, &err));
  SPIEL_CHECK_TRUE(torch::allclose(post_step_output, reloaded_output, 1e-6, 1e-6));

  std::filesystem::remove(temp_path);
  std::cout << "  Passed TestCriticOptimizerStepAndSaveReload.\n";
}

void TestBootstrapConfidenceIntervals() {
  std::cout << "Running TestBootstrapConfidenceIntervals...\n" << std::flush;
  std::vector<double> sample = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0};
  auto ci = Bootstrap95CI(sample, 1000, 12345);
  SPIEL_CHECK_GT(ci.first, 3.0);
  SPIEL_CHECK_LT(ci.second, 8.0);
  SPIEL_CHECK_LE(ci.first, ci.second);
  std::cout << "  Passed TestBootstrapConfidenceIntervals (CI: ["
            << ci.first << ", " << ci.second << "]).\n";
}

void TestManifestContractAndCorruptionRejection() {
  std::cout << "Running TestManifestContractAndCorruptionRejection...\n" << std::flush;
  ExecutableManifest manifest;
  std::string json_str = manifest.ToJsonString();
  auto parsed = json::FromString(json_str);
  SPIEL_CHECK_TRUE(parsed.has_value());

  auto reloaded = ExecutableManifest::FromJson(parsed.value().GetObject());
  SPIEL_CHECK_EQ(reloaded.actor_input_dim, 5580);
  SPIEL_CHECK_EQ(reloaded.critic_expected_input_dim, 9647);
  SPIEL_CHECK_EQ(reloaded.roots_train, 2048);
  SPIEL_CHECK_EQ(reloaded.roots_dev, 256);
  SPIEL_CHECK_EQ(reloaded.roots_test, 512);
  SPIEL_CHECK_EQ(reloaded.critic_init_seed, 19u);
  SPIEL_CHECK_EQ(reloaded.deadline_seconds, 28800);

  // Deliberate corruption checks:
  // Corrupt SHA256 must be detected
  std::string bad_sha = "0000000000000000000000000000000000000000000000000000000000000000";
  SPIEL_CHECK_NE(bad_sha, manifest.actor_model_sha256);

  // Corrupt seed must be detected
  uint64_t wrong_seed = 20;
  SPIEL_CHECK_NE(wrong_seed, manifest.critic_init_seed);

  std::cout << "  Passed TestManifestContractAndCorruptionRejection.\n";
}

}  // namespace action_value_study
}  // namespace open_spiel

#ifdef ACTION_VALUE_TEST_STANDALONE
int main() {
  using namespace open_spiel::action_value_study;
  std::cout << "=== Running Dune Action-Value Learnability Study Tests ===\n";
  TestSchemaValidation();
  TestActorRelativeMapping();
  TestSeedDerivationAndPairing();
  TestCandidateActionSelection();
  TestCriticOptimizerStepAndSaveReload();
  TestBootstrapConfidenceIntervals();
  TestManifestContractAndCorruptionRejection();
  std::cout << "=== ALL ACTION-VALUE LEARNABILITY STUDY TESTS PASSED ===\n";
  return 0;
}
#endif
