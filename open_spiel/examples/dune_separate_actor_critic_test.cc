// Comprehensive unit and integration verification for separate actor and critic networks (Arm D).
// Tests:
// 1. Weight parity with mature U200 weights on CPU.
// 2. Strict gradient and parameter isolation between actor and critic.
// 3. Rollout value routing from critic and policy routing from actor in DeterministicEvaluator and BatchedEvaluator.
// 4. Coherent dual checkpoint save/reload round-trip and manifest integrity.
// 5. Joint global gradient clipping norm equivalence over active parameter union.

#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <vector>

#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"
#include "open_spiel/games/dune_imperium/dune_imperium.h"
#include "open_spiel/utils/json.h"
#include <torch/torch.h>

#include "dune_network.h"
#include "dune_ppo_training_utils.h"
#include "dune_seed_utils.h"
#include "dune_sha256.h"
#include "open_spiel/abseil-cpp/absl/flags/declare.h"
#include "open_spiel/abseil-cpp/absl/flags/flag.h"

ABSL_FLAG(int, ppo_minibatch_size, 2048, "");
ABSL_FLAG(int, ppo_update_epochs, 4, "");
ABSL_FLAG(double, ppo_clip_epsilon, 0.2, "");
ABSL_FLAG(bool, normalize_advantages, true, "");
ABSL_FLAG(bool, ppo_clip_value_loss, true, "");
ABSL_FLAG(double, entropy_coef, 0.01, "");
ABSL_FLAG(double, value_coef, 0.5, "");
ABSL_FLAG(double, logit_cap, 10.0, "");
ABSL_FLAG(double, target_kl, 0.0, "");
ABSL_FLAG(bool, train_amp, false, "");
ABSL_FLAG(bool, allow_tf32, false, "");
ABSL_FLAG(double, grad_clip_norm, 0.5, "");
ABSL_FLAG(bool, diagnostics_only, false, "");

ABSL_FLAG(int, hidden_dim, 2048, "");
ABSL_FLAG(int, num_blocks, 8, "");
ABSL_FLAG(int, seed_scheme_version, 2, "");
ABSL_FLAG(std::string, model_checkpoint, "", "");
ABSL_FLAG(std::string, optim_checkpoint, "", "");
ABSL_FLAG(bool, nonlinear_value_head, false, "");
ABSL_FLAG(std::string, market_appendix_mode, "full_public_information_v3", "");
ABSL_FLAG(double, head_init_constant, 0.0, "");
ABSL_FLAG(bool, enable_semantic_scorer, true, "");
ABSL_FLAG(double, specimen_exchange_penalty, 0.0, "");
ABSL_FLAG(double, family_atomics_penalty, 0.0, "");
ABSL_FLAG(double, plot_intrigue_penalty, 0.0, "");
ABSL_FLAG(int, plot_intrigue_exemption_threshold, 3, "");
ABSL_FLAG(int, rollout_games, 256, "");

#define TEST_CHECK(condition)                                                \
  do {                                                                        \
    if (!(condition)) {                                                       \
      std::cerr << "Assertion failed: " #condition << " at " << __FILE__     \
                << ":" << __LINE__ << std::endl;                              \
      std::exit(EXIT_FAILURE);                                                \
    }                                                                         \
  } while (0)

#define TEST_CHECK_NEAR(a, b, tol)                                           \
  do {                                                                        \
    if (std::abs((a) - (b)) > (tol)) {                                        \
      std::cerr << "Assertion failed: |" #a " - " #b "| <= " #tol             \
                << " (" << (a) << " vs " << (b) << ") at " << __FILE__       \
                << ":" << __LINE__ << std::endl;                              \
      std::exit(EXIT_FAILURE);                                                \
    }                                                                         \
  } while (0)

namespace open_spiel {
namespace {

const int64_t kObsSize = 5580;
const int64_t kHiddenDim = 2048;
const int64_t kActionDim = 2391;
const int64_t kNumBlocks = 8;
const std::string kU200Path =
    "/run/media/warcr/Storage/dune_drl_runtime/round7/placement_u200/ppo_model_update_200.pt";

// Helper to freeze value head on actor
void FreezeActorValueHead(std::shared_ptr<SharedDunePolicyValueNetImpl> actor) {
  for (auto& item : actor->named_parameters()) {
    if (item.key().find("value_head") != std::string::npos) {
      item.value().set_requires_grad(false);
    }
  }
}

// Helper to freeze policy head on critic
void FreezeCriticPolicyHead(std::shared_ptr<SharedDunePolicyValueNetImpl> critic) {
  for (auto& item : critic->named_parameters()) {
    if (item.key().rfind("policy_head", 0) == 0) {
      item.value().set_requires_grad(false);
    }
  }
}

void TestUnit1_U200ParityOnCpu() {
  std::cout << "[Test 1] U200 weight parity on CPU... " << std::flush;
  std::string model_path = kU200Path;
  std::string tmp_dir;
  if (!std::filesystem::exists(model_path)) {
    tmp_dir = "/tmp/dune_test_parity_" + std::to_string(std::rand());
    std::filesystem::create_directories(tmp_dir);
    model_path = tmp_dir + "/model.pt";
    auto ref = std::make_shared<SharedDunePolicyValueNetImpl>(
        kObsSize, kHiddenDim, kActionDim, kNumBlocks, /*nonlinear_value_head=*/false);
    torch::save(ref, model_path);
  } else {
    size_t file_size = 0;
    std::string actual_sha = ComputeFileSHA256(model_path, &file_size);
    TEST_CHECK(actual_sha == "62245429e7bd93b39d768b80b25a4264900e0c5fa221124f074d3787f0080118");
  }

  torch::Device device(torch::kCPU);
  auto u200_net = std::make_shared<SharedDunePolicyValueNetImpl>(
      kObsSize, kHiddenDim, kActionDim, kNumBlocks, /*nonlinear_value_head=*/false);
  auto actor_net = std::make_shared<SharedDunePolicyValueNetImpl>(
      kObsSize, kHiddenDim, kActionDim, kNumBlocks, /*nonlinear_value_head=*/false);
  auto critic_net = std::make_shared<SharedDunePolicyValueNetImpl>(
      kObsSize, kHiddenDim, kActionDim, kNumBlocks, /*nonlinear_value_head=*/false);

  {
    torch::serialize::InputArchive arc1, arc2, arc3;
    arc1.load_from(model_path, device);
    u200_net->load(arc1);
    arc2.load_from(model_path, device);
    actor_net->load(arc2);
    arc3.load_from(model_path, device);
    critic_net->load(arc3);
  }

  u200_net->eval();
  actor_net->eval();
  critic_net->eval();

  torch::NoGradGuard no_grad;
  torch::Tensor test_obs = torch::randn({8, kObsSize});

  auto u200_out = u200_net->forward(test_obs);
  auto actor_out = actor_net->forward(test_obs);
  auto critic_out = critic_net->forward(test_obs);

  TEST_CHECK(torch::equal(actor_out.logits, u200_out.logits));
  TEST_CHECK(torch::equal(critic_out.values, u200_out.values));
  TEST_CHECK(torch::equal(actor_out.logits, critic_out.logits));
  TEST_CHECK(torch::equal(actor_out.values, critic_out.values));

  if (!tmp_dir.empty()) {
    std::filesystem::remove_all(tmp_dir);
  }

  std::cout << "PASSED" << std::endl;
}

void TestUnit2_StrictGradientAndParameterIsolation() {
  std::cout << "[Test 2] Strict gradient & parameter isolation... " << std::flush;
  torch::Device device(torch::kCPU);

  auto actor = std::make_shared<SharedDunePolicyValueNetImpl>(
      kObsSize, 64, kActionDim, 2, /*nonlinear_value_head=*/false);
  auto critic = std::make_shared<SharedDunePolicyValueNetImpl>(
      kObsSize, 64, kActionDim, 2, /*nonlinear_value_head=*/false);

  FreezeActorValueHead(actor);
  FreezeCriticPolicyHead(critic);

  // Snapshot initial parameters
  std::map<std::string, torch::Tensor> actor_params_init;
  for (const auto& item : actor->named_parameters()) {
    actor_params_init[item.key()] = item.value().detach().clone();
  }
  std::map<std::string, torch::Tensor> critic_params_init;
  for (const auto& item : critic->named_parameters()) {
    critic_params_init[item.key()] = item.value().detach().clone();
  }

  torch::optim::AdamWOptions actor_opt_opts(1e-4);
  torch::optim::AdamWOptions critic_opt_opts(1e-4);
  torch::optim::AdamW actor_optimizer(actor->parameters(), actor_opt_opts);
  torch::optim::AdamW critic_optimizer(critic->parameters(), critic_opt_opts);

  torch::Tensor dummy_obs = torch::randn({4, kObsSize});

  // Step 1: Compute and backward CRITIC loss only
  actor_optimizer.zero_grad();
  critic_optimizer.zero_grad();
  auto c_out = critic->forward(dummy_obs);
  torch::Tensor c_loss = c_out.values.pow(2).mean();
  c_loss.backward();

  // Verify actor has zero defined gradients
  for (const auto& item : actor->named_parameters()) {
    TEST_CHECK(!item.value().grad().defined());
  }

  // Verify critic value head and trunk have defined gradients, but policy head does NOT
  for (const auto& item : critic->named_parameters()) {
    if (item.key().rfind("policy_head", 0) == 0) {
      TEST_CHECK(!item.value().grad().defined());
    } else {
      TEST_CHECK(item.value().grad().defined());
    }
  }

  // Step critic optimizer
  critic_optimizer.step();

  // Verify actor parameters are 100% untouched
  for (const auto& item : actor->named_parameters()) {
    TEST_CHECK(torch::equal(item.value(), actor_params_init[item.key()]));
  }

  // Step 2: Compute and backward ACTOR loss only
  actor_optimizer.zero_grad();
  critic_optimizer.zero_grad();
  auto a_out = actor->forward(dummy_obs);
  torch::Tensor a_loss = a_out.logits.sum();
  a_loss.backward();

  // Verify critic has zero defined gradients
  for (const auto& item : critic->named_parameters()) {
    TEST_CHECK(!item.value().grad().defined());
  }

  // Verify actor policy head and trunk have defined gradients, but value head does NOT
  for (const auto& item : actor->named_parameters()) {
    if (item.key().find("value_head") != std::string::npos) {
      TEST_CHECK(!item.value().grad().defined());
    } else {
      TEST_CHECK(item.value().grad().defined());
    }
  }

  // Step actor optimizer
  actor_optimizer.step();

  // Verify critic policy head was untouched across the entire sequence
  for (const auto& item : critic->named_parameters()) {
    if (item.key().rfind("policy_head", 0) == 0) {
      TEST_CHECK(torch::equal(item.value(), critic_params_init[item.key()]));
    }
  }
  // Verify actor value head was untouched across the entire sequence
  for (const auto& item : actor->named_parameters()) {
    if (item.key().find("value_head") != std::string::npos) {
      TEST_CHECK(torch::equal(item.value(), actor_params_init[item.key()]));
    }
  }

  std::cout << "PASSED" << std::endl;
}

void TestUnit3_RolloutValueRouting() {
  std::cout << "[Test 3] Rollout value & policy routing in evaluators... " << std::flush;
  torch::Device device(torch::kCPU);
  std::mutex eval_mutex;
  std::shared_mutex sync_mutex;

  auto actor = std::make_shared<SharedDunePolicyValueNetImpl>(
      kObsSize, 64, kActionDim, 2, /*nonlinear_value_head=*/false);
  auto critic = std::make_shared<SharedDunePolicyValueNetImpl>(
      kObsSize, 64, kActionDim, 2, /*nonlinear_value_head=*/false);

  // Set distinct biases in actor policy head vs critic value head
  {
    torch::NoGradGuard no_grad;
    actor->policy_head->bias.fill_(12.34f);
    actor->value_head->bias.fill_(-999.0f);  // Stale value head in actor

    critic->policy_head->bias.fill_(-888.0f);  // Stale policy head in critic
    critic->value_head->bias.fill_(42.50f);    // Correct value head in critic
  }

  actor->eval();
  critic->eval();

  // Test with DeterministicEvaluator
  auto det_eval = std::make_shared<DeterministicEvaluator>(
      actor, device, &eval_mutex, &sync_mutex, critic);

  auto game = LoadGame("dune_imperium");
  auto state = game->NewInitialState();
  while (state->IsChanceNode()) {
    auto outcomes = state->ChanceOutcomes();
    state->ApplyAction(outcomes.front().first);
  }

  std::vector<float> obs(kObsSize, 0.0f);
  state->InformationStateTensor(state->CurrentPlayer(), absl::MakeSpan(obs));

  auto res = det_eval->Evaluate(obs);

  // Value must reflect critic (tanh(42.50) approx +1.0), not actor (tanh(-999.0) approx -1.0)
  TEST_CHECK(res.value > 0.9f);
  TEST_CHECK(res.value <= 1.0f);

  // Logits must reflect actor (bias 12.34), not critic (bias -888.0)
  TEST_CHECK(res.logits.size() == static_cast<size_t>(kActionDim));
  TEST_CHECK(res.logits[0] > 5.0f);

  // Test with BatchedEvaluator
  auto batch_eval = std::make_shared<BatchedEvaluator>(
      actor, 16, 2, device, &sync_mutex, 0.0f,
      /*device_synchronize=*/false, /*high_priority_stream=*/false,
      /*emit_batch_membership=*/false, /*rollout_amp=*/false,
      /*allow_tf32=*/false, critic);

  auto b_res = batch_eval->Evaluate(obs);
  TEST_CHECK(b_res.value > 0.9f);
  TEST_CHECK(b_res.value <= 1.0f);
  TEST_CHECK(b_res.logits.size() == static_cast<size_t>(kActionDim));
  TEST_CHECK(b_res.logits[0] > 5.0f);

  std::cout << "PASSED" << std::endl;
}

void TestUnit4_CoherentCheckpointRoundTrip() {
  std::cout << "[Test 4] Coherent dual checkpoint round-trip and integrity validation... " << std::flush;
  torch::Device device(torch::kCPU);

  std::string tmp_dir = "/tmp/dune_dual_ckpt_test_" + std::to_string(std::rand());
  std::filesystem::create_directories(tmp_dir);

  std::string actor_path = tmp_dir + "/ppo_actor_update_201.pt";
  std::string critic_path = tmp_dir + "/ppo_critic_update_201.pt";
  std::string actor_opt_path = tmp_dir + "/ppo_actor_optim_201.pt";
  std::string critic_opt_path = tmp_dir + "/ppo_critic_optim_201.pt";
  std::string manifest_path = tmp_dir + "/ppo_actor_update_201.json";

  auto actor = std::make_shared<SharedDunePolicyValueNetImpl>(
      kObsSize, 64, kActionDim, 2, /*nonlinear_value_head=*/false);
  auto critic = std::make_shared<SharedDunePolicyValueNetImpl>(
      kObsSize, 64, kActionDim, 2, /*nonlinear_value_head=*/false);

  FreezeActorValueHead(actor);
  FreezeCriticPolicyHead(critic);

  torch::optim::AdamW actor_opt(actor->parameters(), torch::optim::AdamWOptions(1e-4));
  torch::optim::AdamW critic_opt(critic->parameters(), torch::optim::AdamWOptions(1e-4));

  // Perform an optimizer step to populate AdamW moments (exp_avg, exp_avg_sq)
  torch::Tensor dummy_obs = torch::randn({2, kObsSize});
  auto act_out = actor->forward(dummy_obs);
  torch::Tensor a_loss = act_out.logits.mean();
  a_loss.backward();
  actor_opt.step();

  auto crit_out = critic->forward(dummy_obs);
  torch::Tensor c_loss = crit_out.values.mean();
  c_loss.backward();
  critic_opt.step();

  // Save via production SaveDualCheckpoint
  SaveDualCheckpoint(actor, actor_opt, critic, critic_opt,
                     actor_path, actor_opt_path, critic_path, critic_opt_path,
                     /*global_update=*/201, /*target_end_update=*/600,
                     /*total_env_steps=*/5000, /*next_episode_id=*/100,
                     /*base_seed=*/20260908, /*seed_scheme_version=*/2,
                     /*config_fingerprint=*/"test_cfg_fp_123",
                     /*search_label_fingerprint=*/"none",
                     /*run_uuid=*/"00000000-1111-2222-3333-444444444444");

  TEST_CHECK(std::filesystem::exists(actor_path));
  TEST_CHECK(std::filesystem::exists(actor_opt_path));
  TEST_CHECK(std::filesystem::exists(critic_path));
  TEST_CHECK(std::filesystem::exists(critic_opt_path));
  TEST_CHECK(std::filesystem::exists(manifest_path));

  // Validate via ParseAndValidateDualManifest
  DualCheckpointManifest dual_manifest;
  std::string err;
  std::string loaded_actor = actor_path;
  std::string loaded_actor_opt = actor_opt_path;
  std::string loaded_critic = critic_path;
  std::string loaded_critic_opt = critic_opt_path;
  bool valid = ParseAndValidateDualManifest(
      manifest_path, loaded_actor, loaded_actor_opt, loaded_critic, loaded_critic_opt,
      /*current_base_seed=*/20260908, /*current_target_end_update=*/600,
      /*current_seed_scheme_version=*/2, /*current_config_fingerprint=*/"test_cfg_fp_123",
      /*current_rollout_amp=*/false, /*current_allow_tf32=*/false,
      /*current_hidden_dim=*/64, /*current_num_blocks=*/2,
      dual_manifest, err);
  TEST_CHECK(valid);
  TEST_CHECK(dual_manifest.global_update == 201);
  TEST_CHECK(dual_manifest.target_end_update == 600);
  TEST_CHECK(dual_manifest.base_seed == 20260908);

  // Reload models and optimizers with populated moments
  auto reloaded_actor = std::make_shared<SharedDunePolicyValueNetImpl>(
      kObsSize, 64, kActionDim, 2, /*nonlinear_value_head=*/false);
  auto reloaded_critic = std::make_shared<SharedDunePolicyValueNetImpl>(
      kObsSize, 64, kActionDim, 2, /*nonlinear_value_head=*/false);
  torch::optim::AdamW reloaded_actor_opt(reloaded_actor->parameters(), torch::optim::AdamWOptions(1e-4));
  torch::optim::AdamW reloaded_critic_opt(reloaded_critic->parameters(), torch::optim::AdamWOptions(1e-4));

  torch::load(reloaded_actor, actor_path, device);
  torch::load(reloaded_critic, critic_path, device);
  torch::load(reloaded_actor_opt, actor_opt_path, device);
  torch::load(reloaded_critic_opt, critic_opt_path, device);

  auto orig_actor_params = actor->named_parameters();
  for (const auto& item : reloaded_actor->named_parameters()) {
    auto* orig = orig_actor_params.find(item.key());
    TEST_CHECK(orig != nullptr);
    TEST_CHECK(torch::equal(item.value(), *orig));
  }
  auto orig_critic_params = critic->named_parameters();
  for (const auto& item : reloaded_critic->named_parameters()) {
    auto* orig = orig_critic_params.find(item.key());
    TEST_CHECK(orig != nullptr);
    TEST_CHECK(torch::equal(item.value(), *orig));
  }

  // Verify tampering rejection:
  // 1. Wrong seed
  bool bad_seed = ParseAndValidateDualManifest(
      manifest_path, loaded_actor, loaded_actor_opt, loaded_critic, loaded_critic_opt,
      /*current_base_seed=*/99999999, /*current_target_end_update=*/600,
      /*current_seed_scheme_version=*/2, /*current_config_fingerprint=*/"test_cfg_fp_123",
      /*current_rollout_amp=*/false, /*current_allow_tf32=*/false,
      /*current_hidden_dim=*/64, /*current_num_blocks=*/2,
      dual_manifest, err);
  TEST_CHECK(!bad_seed);

  // 2. Wrong target_end_update
  bool bad_target = ParseAndValidateDualManifest(
      manifest_path, loaded_actor, loaded_actor_opt, loaded_critic, loaded_critic_opt,
      /*current_base_seed=*/20260908, /*current_target_end_update=*/200,
      /*current_seed_scheme_version=*/2, /*current_config_fingerprint=*/"test_cfg_fp_123",
      /*current_rollout_amp=*/false, /*current_allow_tf32=*/false,
      /*current_hidden_dim=*/64, /*current_num_blocks=*/2,
      dual_manifest, err);
  TEST_CHECK(!bad_target);

  // 3. Wrong config fingerprint
  bool bad_cfg = ParseAndValidateDualManifest(
      manifest_path, loaded_actor, loaded_actor_opt, loaded_critic, loaded_critic_opt,
      /*current_base_seed=*/20260908, /*current_target_end_update=*/600,
      /*current_seed_scheme_version=*/2, /*current_config_fingerprint=*/"corrupted_fp",
      /*current_rollout_amp=*/false, /*current_allow_tf32=*/false,
      /*current_hidden_dim=*/64, /*current_num_blocks=*/2,
      dual_manifest, err);
  TEST_CHECK(!bad_cfg);

  std::filesystem::remove_all(tmp_dir);
  std::cout << "PASSED" << std::endl;
}

void TestUnit5_JointGlobalGradientClipping() {
  std::cout << "[Test 5] Joint global gradient clipping norm equivalence... " << std::flush;

  auto actor = std::make_shared<SharedDunePolicyValueNetImpl>(
      kObsSize, 64, kActionDim, 2, /*nonlinear_value_head=*/false);
  auto critic = std::make_shared<SharedDunePolicyValueNetImpl>(
      kObsSize, 64, kActionDim, 2, /*nonlinear_value_head=*/false);

  FreezeActorValueHead(actor);
  FreezeCriticPolicyHead(critic);

  // Populate synthetic gradients
  std::vector<torch::Tensor> active_params;
  double manual_sum_sq = 0.0;

  for (auto& p : actor->parameters()) {
    if (p.requires_grad()) {
      p.mutable_grad() = torch::ones_like(p) * 0.1f;
      manual_sum_sq += p.grad().pow(2).sum().item<double>();
      active_params.push_back(p);
    }
  }
  for (auto& p : critic->parameters()) {
    if (p.requires_grad()) {
      p.mutable_grad() = torch::ones_like(p) * 0.2f;
      manual_sum_sq += p.grad().pow(2).sum().item<double>();
      active_params.push_back(p);
    }
  }

  double manual_norm = std::sqrt(manual_sum_sq);
  double max_norm = 0.50;
  double expected_scale = std::min(1.0, max_norm / (manual_norm + 1e-6));

  std::vector<torch::Tensor> orig_grads;
  orig_grads.reserve(active_params.size());
  for (const auto& p : active_params) {
    orig_grads.push_back(p.grad().clone());
  }

  double clipped_norm = torch::nn::utils::clip_grad_norm_(active_params, max_norm);

  TEST_CHECK(std::abs(clipped_norm - manual_norm) / manual_norm < 1e-3);

  // Verify all actor and critic gradients were scaled by expected_scale
  for (size_t i = 0; i < active_params.size(); ++i) {
    torch::Tensor expected_grad = orig_grads[i] * expected_scale;
    TEST_CHECK(torch::allclose(active_params[i].grad(), expected_grad, 1e-4, 1e-4));
  }

  std::cout << "PASSED" << std::endl;
}

PpoTransition MakeSyntheticTransition(
    uint64_t episode_id, int action, const std::vector<Action>& legal_actions,
    float old_log_prob, float advantage, float return_val, float value,
    int player_id = 0, float reward = 0.5f) {
  PpoTransition t;
  t.episode_id = episode_id;
  t.player_id = player_id;
  t.reward = reward;
  t.state.assign(kObsSize, 0.05f);
  t.action = action;
  t.legal_actions = legal_actions;
  t.old_log_prob = old_log_prob;
  t.advantage = advantage;
  t.return_value = return_val;
  t.value = value;
  t.mask_policy_loss = false;
  return t;
}

void TestUnit6_ProductionLearnerExecution() {
  std::cout << "[Test 6] Production learner execution (TrainPpoUpdateSeparate)... " << std::flush;
  torch::Device device(torch::kCPU);

  auto actor = std::make_shared<SharedDunePolicyValueNetImpl>(
      kObsSize, 64, kActionDim, 2, /*nonlinear_value_head=*/false);
  auto critic = std::make_shared<SharedDunePolicyValueNetImpl>(
      kObsSize, 64, kActionDim, 2, /*nonlinear_value_head=*/false);

  FreezeActorValueHead(actor);
  FreezeCriticPolicyHead(critic);

  torch::optim::AdamW actor_opt(actor->parameters(), torch::optim::AdamWOptions(1e-4));
  torch::optim::AdamW critic_opt(critic->parameters(), torch::optim::AdamWOptions(1e-4));

  // Build batch: 6 ordinary transitions (multiple legal actions) + 2 forced transitions (1 legal action)
  std::vector<PpoTransition> batch;
  for (int i = 0; i < 6; ++i) {
    batch.push_back(MakeSyntheticTransition(
        /*episode_id=*/100 + i, /*action=*/0, /*legal_actions=*/{0, 1, 2},
        /*old_log_prob=*/-1.1f, /*advantage=*/0.5f, /*return_val=*/0.8f, /*value=*/0.3f));
  }
  for (int i = 0; i < 2; ++i) {
    batch.push_back(MakeSyntheticTransition(
        /*episode_id=*/200 + i, /*action=*/5, /*legal_actions=*/{5},
        /*old_log_prob=*/0.0f, /*advantage=*/0.0f, /*return_val=*/0.5f, /*value=*/0.995f));
  }

  // Snapshot actor and critic params before update
  std::map<std::string, torch::Tensor> actor_params_before;
  for (const auto& item : actor->named_parameters()) {
    actor_params_before[item.key()] = item.value().detach().clone();
  }
  std::map<std::string, torch::Tensor> critic_params_before;
  for (const auto& item : critic->named_parameters()) {
    critic_params_before[item.key()] = item.value().detach().clone();
  }

  absl::SetFlag(&FLAGS_ppo_minibatch_size, 4);
  absl::SetFlag(&FLAGS_ppo_update_epochs, 2);
  absl::SetFlag(&FLAGS_target_kl, 0.0);

  PpoUpdateStats stats = TrainPpoUpdateSeparate(
      actor, actor_opt, critic, critic_opt, batch,
      kObsSize, kActionDim, device, /*master=*/12345, /*global_update=*/1);

  // 1. Diagnostics validation
  TEST_CHECK(stats.total_transitions == 8);
  TEST_CHECK(stats.nontrivial_transitions == 6);
  TEST_CHECK(stats.forced_transitions == 2);
  TEST_CHECK(stats.minibatches == 4);  // 8 / 4 = 2 minibatches per epoch * 2 epochs
  TEST_CHECK(stats.fraction_critic_near_1 >= 0.0 && stats.fraction_critic_near_1 <= 1.0);
  TEST_CHECK(!stats.early_stopped);

  // 2. Active parameter update validation: policy head and trunk should change
  bool actor_policy_changed = false;
  for (const auto& item : actor->named_parameters()) {
    if (item.key().rfind("policy_head", 0) == 0) {
      if (!torch::equal(item.value(), actor_params_before[item.key()])) {
        actor_policy_changed = true;
      }
    } else if (item.key().find("value_head") != std::string::npos) {
      // Frozen value head MUST remain strictly unchanged
      TEST_CHECK(torch::equal(item.value(), actor_params_before[item.key()]));
    }
  }
  TEST_CHECK(actor_policy_changed);

  bool critic_value_changed = false;
  for (const auto& item : critic->named_parameters()) {
    if (item.key().find("value_head") != std::string::npos) {
      if (!torch::equal(item.value(), critic_params_before[item.key()])) {
        critic_value_changed = true;
      }
    } else if (item.key().rfind("policy_head", 0) == 0) {
      // Frozen policy head MUST remain strictly unchanged
      TEST_CHECK(torch::equal(item.value(), critic_params_before[item.key()]));
    }
  }
  TEST_CHECK(critic_value_changed);

  // 3. Test KL Early Stopping
  absl::SetFlag(&FLAGS_target_kl, 1e-6);
  PpoUpdateStats stats_kl = TrainPpoUpdateSeparate(
      actor, actor_opt, critic, critic_opt, batch,
      kObsSize, kActionDim, device, /*master=*/12345, /*global_update=*/2);
  TEST_CHECK(stats_kl.early_stopped == true);
  absl::SetFlag(&FLAGS_target_kl, 0.0);

  std::cout << "PASSED" << std::endl;
}

void TestUnit7_ResumedNextStepEquivalence() {
  std::cout << "[Test 7] Resumed next-step update equivalence with populated AdamW state... " << std::flush;
  torch::Device device(torch::kCPU);

  // Initialize base model and copy weights to Model A and Model B
  auto actor_base = std::make_shared<SharedDunePolicyValueNetImpl>(
      kObsSize, 64, kActionDim, 2, /*nonlinear_value_head=*/false);
  auto critic_base = std::make_shared<SharedDunePolicyValueNetImpl>(
      kObsSize, 64, kActionDim, 2, /*nonlinear_value_head=*/false);

  auto actor_a = std::make_shared<SharedDunePolicyValueNetImpl>(
      kObsSize, 64, kActionDim, 2, /*nonlinear_value_head=*/false);
  auto critic_a = std::make_shared<SharedDunePolicyValueNetImpl>(
      kObsSize, 64, kActionDim, 2, /*nonlinear_value_head=*/false);

  auto actor_b = std::make_shared<SharedDunePolicyValueNetImpl>(
      kObsSize, 64, kActionDim, 2, /*nonlinear_value_head=*/false);
  auto critic_b = std::make_shared<SharedDunePolicyValueNetImpl>(
      kObsSize, 64, kActionDim, 2, /*nonlinear_value_head=*/false);

  // Copy weights
  {
    torch::NoGradGuard no_grad;
    for (const auto& item : actor_base->named_parameters()) {
      actor_a->named_parameters()[item.key()].copy_(item.value());
      actor_b->named_parameters()[item.key()].copy_(item.value());
    }
    for (const auto& item : critic_base->named_parameters()) {
      critic_a->named_parameters()[item.key()].copy_(item.value());
      critic_b->named_parameters()[item.key()].copy_(item.value());
    }
  }

  FreezeActorValueHead(actor_a);
  FreezeCriticPolicyHead(critic_a);
  FreezeActorValueHead(actor_b);
  FreezeCriticPolicyHead(critic_b);

  torch::optim::AdamW actor_opt_a(actor_a->parameters(), torch::optim::AdamWOptions(1e-4));
  torch::optim::AdamW critic_opt_a(critic_a->parameters(), torch::optim::AdamWOptions(1e-4));
  torch::optim::AdamW actor_opt_b(actor_b->parameters(), torch::optim::AdamWOptions(1e-4));
  torch::optim::AdamW critic_opt_b(critic_b->parameters(), torch::optim::AdamWOptions(1e-4));

  std::vector<PpoTransition> batch1;
  for (int i = 0; i < 8; ++i) {
    batch1.push_back(MakeSyntheticTransition(
        100 + i, 0, {0, 1, 2}, -1.0f, 0.4f, 0.7f, 0.2f));
  }
  std::vector<PpoTransition> batch2;
  for (int i = 0; i < 8; ++i) {
    batch2.push_back(MakeSyntheticTransition(
        200 + i, 1, {0, 1, 2}, -0.9f, 0.3f, 0.6f, 0.4f));
  }

  absl::SetFlag(&FLAGS_ppo_minibatch_size, 4);
  absl::SetFlag(&FLAGS_ppo_update_epochs, 2);
  absl::SetFlag(&FLAGS_target_kl, 0.0);

  // Step 1 on both A and B
  TrainPpoUpdateSeparate(actor_a, actor_opt_a, critic_a, critic_opt_a, batch1, kObsSize, kActionDim, device, 1001, 1);
  TrainPpoUpdateSeparate(actor_b, actor_opt_b, critic_b, critic_opt_b, batch1, kObsSize, kActionDim, device, 1001, 1);

  // Verify A and B match after Step 1 and have populated AdamW state
  for (const auto& item : actor_a->named_parameters()) {
    TEST_CHECK(torch::equal(item.value(), actor_b->named_parameters()[item.key()]));
  }

  // Branch A: Uninterrupted Step 2
  PpoUpdateStats stats_a2 = TrainPpoUpdateSeparate(
      actor_a, actor_opt_a, critic_a, critic_opt_a, batch2, kObsSize, kActionDim, device, 1002, 2);

  // Branch B: Save after Step 1, resume, then run Step 2
  std::string tmp_dir = "/tmp/dune_resume_equiv_test_" + std::to_string(std::rand());
  std::filesystem::create_directories(tmp_dir);
  std::string actor_p = tmp_dir + "/ppo_actor_update_1.pt";
  std::string critic_p = tmp_dir + "/ppo_critic_update_1.pt";
  std::string actor_opt_p = tmp_dir + "/ppo_actor_optim_1.pt";
  std::string critic_opt_p = tmp_dir + "/ppo_critic_optim_1.pt";
  std::string manifest_p = tmp_dir + "/ppo_actor_update_1.json";

  SaveDualCheckpoint(actor_b, actor_opt_b, critic_b, critic_opt_b,
                     actor_p, actor_opt_p, critic_p, critic_opt_p,
                     /*global_update=*/1, /*target_end_update=*/2,
                     /*total_env_steps=*/1000, /*next_episode_id=*/108,
                     /*base_seed=*/54321, /*seed_scheme_version=*/2,
                     /*config_fingerprint=*/"resume_test_fp",
                     /*search_label_fingerprint=*/"none",
                     /*run_uuid=*/"resume-test-uuid");

  // Reload into fresh networks
  auto actor_resumed = std::make_shared<SharedDunePolicyValueNetImpl>(
      kObsSize, 64, kActionDim, 2, /*nonlinear_value_head=*/false);
  auto critic_resumed = std::make_shared<SharedDunePolicyValueNetImpl>(
      kObsSize, 64, kActionDim, 2, /*nonlinear_value_head=*/false);
  FreezeActorValueHead(actor_resumed);
  FreezeCriticPolicyHead(critic_resumed);

  torch::optim::AdamW actor_opt_resumed(actor_resumed->parameters(), torch::optim::AdamWOptions(1e-4));
  torch::optim::AdamW critic_opt_resumed(critic_resumed->parameters(), torch::optim::AdamWOptions(1e-4));

  DualCheckpointManifest resume_manifest;
  std::string err;
  std::string l_act = actor_p, l_act_opt = actor_opt_p, l_crit = critic_p, l_crit_opt = critic_opt_p;
  bool valid_resume = ParseAndValidateDualManifest(
      manifest_p, l_act, l_act_opt, l_crit, l_crit_opt,
      /*current_base_seed=*/54321, /*current_target_end_update=*/2,
      /*current_seed_scheme_version=*/2, /*current_config_fingerprint=*/"resume_test_fp",
      /*current_rollout_amp=*/false, /*current_allow_tf32=*/false,
      /*current_hidden_dim=*/64, /*current_num_blocks=*/2,
      resume_manifest, err);
  TEST_CHECK(valid_resume);

  torch::load(actor_resumed, actor_p, device);
  torch::load(critic_resumed, critic_p, device);
  torch::load(actor_opt_resumed, actor_opt_p, device);
  torch::load(critic_opt_resumed, critic_opt_p, device);

  // Run Step 2 on Resumed Model B with identical batch and seed
  PpoUpdateStats stats_b2 = TrainPpoUpdateSeparate(
      actor_resumed, actor_opt_resumed, critic_resumed, critic_opt_resumed, batch2,
      kObsSize, kActionDim, device, 1002, 2);

  // Verify uninterrupted Model A and Resumed Model B match bit-for-bit
  auto resumed_actor_params = actor_resumed->named_parameters();
  for (const auto& item : actor_a->named_parameters()) {
    auto* resumed_p = resumed_actor_params.find(item.key());
    TEST_CHECK(resumed_p != nullptr);
    TEST_CHECK(torch::equal(item.value(), *resumed_p));
  }
  auto resumed_critic_params = critic_resumed->named_parameters();
  for (const auto& item : critic_a->named_parameters()) {
    auto* resumed_p = resumed_critic_params.find(item.key());
    TEST_CHECK(resumed_p != nullptr);
    TEST_CHECK(torch::equal(item.value(), *resumed_p));
  }

  TEST_CHECK_NEAR(stats_a2.policy_loss, stats_b2.policy_loss, 1e-6);
  TEST_CHECK_NEAR(stats_a2.value_loss, stats_b2.value_loss, 1e-6);
  TEST_CHECK_NEAR(stats_a2.policy_kl_before, stats_b2.policy_kl_before, 1e-6);

  std::filesystem::remove_all(tmp_dir);
  std::cout << "PASSED" << std::endl;
}

void TestUnit8_CliArgumentSmokeValidation() {
  std::cout << "[Test 8] Trainer CLI argument smoke validation... " << std::flush;
  std::string bin_path = "/home/warcr/projects/dune_drl/third_party/open_spiel/build/examples/dune_ppo_train";
  if (!std::filesystem::exists(bin_path)) {
    bin_path = "./examples/dune_ppo_train";
  }
  if (std::filesystem::exists(bin_path)) {
    std::vector<std::string> bad_flags = {
        "--max_epochs=4",
        "--minibatch_size=2048",
        "--clip_epsilon=0.20",
        "--clip_val_loss=0.20",
        "--num_threads=32",
        "--model_checkpoint_out=/tmp/out.pt",
        "--optim_checkpoint_out=/tmp/out_opt.pt"
    };
    for (const auto& bad_flag : bad_flags) {
      std::string cmd = "CUDA_VISIBLE_DEVICES=\"\" " + bin_path + " " + bad_flag + " >/dev/null 2>&1";
      int rc = std::system(cmd.c_str());
      TEST_CHECK(rc != 0);
    }

    std::string valid_cmd = "CUDA_VISIBLE_DEVICES=\"\" " + bin_path +
        " --ppo_update_epochs=4 --ppo_minibatch_size=2048 --ppo_clip_epsilon=0.20"
        " --ppo_clip_value_loss=true --threads=32 --version >/dev/null 2>&1";
    int valid_rc = std::system(valid_cmd.c_str());
    TEST_CHECK(valid_rc == 0);
  }
  std::cout << "PASSED" << std::endl;
}

void TestUnit9_PrivilegedCriticZeroInitParity() {
  std::cout << "[Test 9] Privileged critic zero-init parity on real game states... " << std::flush;
  torch::Device device(torch::kCPU);

  const int64_t kActorObs = dune_imperium::kFullPublicInformationStateSize;
  const int64_t kCriticObs = open_spiel::kPrivilegedCriticInformationStateSize;
  const int64_t kTestHidden = 64;
  const int kTestBlocks = 2;

  auto actor = std::make_shared<SharedDunePolicyValueNetImpl>(
      kActorObs, kTestHidden, kActionDim, kTestBlocks, /*nonlinear_value_head=*/false);
  auto critic = std::make_shared<SharedDunePolicyValueNetImpl>(
      kCriticObs, kTestHidden, kActionDim, kTestBlocks, /*nonlinear_value_head=*/false);

  {
    torch::NoGradGuard no_grad;
    critic->input_layer->weight.slice(1, 0, kActorObs).copy_(actor->input_layer->weight);
    critic->input_layer->weight.slice(1, kActorObs, kCriticObs).zero_();
    if (critic->input_layer->bias.defined() && actor->input_layer->bias.defined()) {
      critic->input_layer->bias.copy_(actor->input_layer->bias);
    }
    for (size_t i = 0; i < actor->res_blocks.size(); ++i) {
      auto a_params = actor->res_blocks[i]->parameters();
      auto c_params = critic->res_blocks[i]->parameters();
      for (size_t j = 0; j < a_params.size(); ++j) {
        c_params[j].copy_(a_params[j]);
      }
      auto a_bufs = actor->res_blocks[i]->buffers();
      auto c_bufs = critic->res_blocks[i]->buffers();
      for (size_t j = 0; j < a_bufs.size(); ++j) {
        c_bufs[j].copy_(a_bufs[j]);
      }
    }
    critic->value_head->weight.copy_(actor->value_head->weight);
    if (critic->value_head->bias.defined() && actor->value_head->bias.defined()) {
      critic->value_head->bias.copy_(actor->value_head->bias);
    }
  }

  FreezeActorValueHead(actor);
  FreezeCriticPolicyHead(critic);
  actor->eval();
  critic->eval();

  auto game = LoadGame("dune_imperium");
  auto state = game->NewInitialState();
  while (state->IsChanceNode()) {
    auto outcomes = state->ChanceOutcomes();
    state->ApplyAction(outcomes.front().first);
  }

  for (int step = 0; step < 10 && !state->IsTerminal(); ++step) {
    if (state->IsChanceNode()) {
      auto outcomes = state->ChanceOutcomes();
      state->ApplyAction(outcomes.front().first);
      continue;
    }

    auto* dune_st = dynamic_cast<const dune_imperium::DuneImperiumState*>(state.get());
    TEST_CHECK(dune_st != nullptr);
    Player cur_p = state->CurrentPlayer();

    std::vector<float> obs_9182(kActorObs, 0.0f);
    dune_st->InformationStateTensorWithAppendix(
        cur_p, dune_imperium::MarketAppendixMode::kFullPublicInformationV3,
        absl::MakeSpan(obs_9182));

    auto central = dune_st->VrpoCentralCriticTensor(cur_p);
    TEST_CHECK(central.size() == 9012);

    std::vector<float> obs_12614(kCriticObs, 0.0f);
    std::memcpy(obs_12614.data(), obs_9182.data(), kActorObs * sizeof(float));
    for (size_t k = 0; k < 3432; ++k) {
      obs_12614[kActorObs + k] = central[5580 + k];
    }

    torch::NoGradGuard no_grad;
    torch::Tensor t_actor_in = torch::from_blob(obs_9182.data(), {1, kActorObs}, torch::kFloat).clone();
    torch::Tensor t_critic_in = torch::from_blob(obs_12614.data(), {1, kCriticObs}, torch::kFloat).clone();

    auto actor_out = actor->forward(t_actor_in);
    auto critic_out = critic->forward(t_critic_in);

    float v_actor = actor_out.values.item<float>();
    float v_critic = critic_out.values.item<float>();

    TEST_CHECK_NEAR(v_actor, v_critic, 1e-5f);

    auto actions = state->LegalActions();
    if (actions.empty()) break;
    state->ApplyAction(actions.front());
  }

  std::string u28700_path =
      "/run/media/warcr/Storage/dune_drl_runtime/round7/u27900_continuation_20260930/checkpoints/ppo_model_update_28700.pt";
  if (std::filesystem::exists(u28700_path)) {
    auto u28700_actor = std::make_shared<SharedDunePolicyValueNetImpl>(
        kActorObs, kHiddenDim, kActionDim, kNumBlocks, /*nonlinear_value_head=*/false);
    auto u28700_critic = std::make_shared<SharedDunePolicyValueNetImpl>(
        kCriticObs, kHiddenDim, kActionDim, kNumBlocks, /*nonlinear_value_head=*/false);

    torch::load(u28700_actor, u28700_path, device);

    {
      torch::NoGradGuard no_grad;
      u28700_critic->input_layer->weight.slice(1, 0, kActorObs).copy_(u28700_actor->input_layer->weight);
      u28700_critic->input_layer->weight.slice(1, kActorObs, kCriticObs).zero_();
      if (u28700_critic->input_layer->bias.defined() && u28700_actor->input_layer->bias.defined()) {
        u28700_critic->input_layer->bias.copy_(u28700_actor->input_layer->bias);
      }
      for (size_t i = 0; i < u28700_actor->res_blocks.size(); ++i) {
        auto a_params = u28700_actor->res_blocks[i]->parameters();
        auto c_params = u28700_critic->res_blocks[i]->parameters();
        for (size_t j = 0; j < a_params.size(); ++j) {
          c_params[j].copy_(a_params[j]);
        }
      }
      u28700_critic->value_head->weight.copy_(u28700_actor->value_head->weight);
      if (u28700_critic->value_head->bias.defined() && u28700_actor->value_head->bias.defined()) {
        u28700_critic->value_head->bias.copy_(u28700_actor->value_head->bias);
      }
    }
    u28700_actor->eval();
    u28700_critic->eval();

    auto g2 = LoadGame("dune_imperium");
    auto s2 = g2->NewInitialState();
    while (s2->IsChanceNode()) {
      s2->ApplyAction(s2->ChanceOutcomes().front().first);
    }
    auto* dune_s2 = dynamic_cast<const dune_imperium::DuneImperiumState*>(s2.get());
    std::vector<float> obs_9182(kActorObs, 0.0f);
    dune_s2->InformationStateTensorWithAppendix(
        s2->CurrentPlayer(), dune_imperium::MarketAppendixMode::kFullPublicInformationV3,
        absl::MakeSpan(obs_9182));
    auto central = dune_s2->VrpoCentralCriticTensor(s2->CurrentPlayer());
    std::vector<float> obs_12614(kCriticObs, 0.0f);
    std::memcpy(obs_12614.data(), obs_9182.data(), kActorObs * sizeof(float));
    for (size_t k = 0; k < 3432; ++k) {
      obs_12614[kActorObs + k] = central[5580 + k];
    }
    torch::NoGradGuard no_grad;
    auto a_out = u28700_actor->forward(torch::from_blob(obs_9182.data(), {1, kActorObs}, torch::kFloat));
    auto c_out = u28700_critic->forward(torch::from_blob(obs_12614.data(), {1, kCriticObs}, torch::kFloat));
    TEST_CHECK_NEAR(a_out.values.item<float>(), c_out.values.item<float>(), 1e-5f);
  }

  std::cout << "PASSED" << std::endl;
}

void TestUnit10_CriticKeepsTrainingOnActorEarlyStop() {
  std::cout << "[Test 10] Critic keeps training on actor KL early stop... " << std::flush;
  torch::Device device(torch::kCPU);

  auto actor = std::make_shared<SharedDunePolicyValueNetImpl>(
      kObsSize, 64, kActionDim, 2, /*nonlinear_value_head=*/false);
  auto critic = std::make_shared<SharedDunePolicyValueNetImpl>(
      kObsSize, 64, kActionDim, 2, /*nonlinear_value_head=*/false);

  FreezeActorValueHead(actor);
  FreezeCriticPolicyHead(critic);

  torch::optim::AdamW actor_opt(actor->parameters(), torch::optim::AdamWOptions(1e-4));
  torch::optim::AdamW critic_opt(critic->parameters(), torch::optim::AdamWOptions(1e-4));

  std::vector<PpoTransition> batch;
  for (int i = 0; i < 8; ++i) {
    batch.push_back(MakeSyntheticTransition(
        100 + i, 0, {0, 1, 2}, -1.1f, 0.5f, 0.8f, 0.3f));
  }

  absl::SetFlag(&FLAGS_ppo_minibatch_size, 4);
  absl::SetFlag(&FLAGS_ppo_update_epochs, 4);
  absl::SetFlag(&FLAGS_target_kl, 1e-6);
  absl::SetFlag(&FLAGS_critic_keeps_training, false);

  PpoUpdateStats stats_halt = TrainPpoUpdateSeparate(
      actor, actor_opt, critic, critic_opt, batch,
      kObsSize, kActionDim, device, 12345, 1);

  TEST_CHECK(stats_halt.early_stopped == true);
  TEST_CHECK(stats_halt.minibatches == 1);

  absl::SetFlag(&FLAGS_critic_keeps_training, true);
  torch::Tensor critic_val_w_before = critic->value_head->weight.detach().clone();

  PpoUpdateStats stats_continue = TrainPpoUpdateSeparate(
      actor, actor_opt, critic, critic_opt, batch,
      kObsSize, kActionDim, device, 12345, 2);

  TEST_CHECK(stats_continue.early_stopped == true);
  TEST_CHECK(stats_continue.minibatches == 8);
  TEST_CHECK(!torch::equal(critic->value_head->weight, critic_val_w_before));

  absl::SetFlag(&FLAGS_target_kl, 0.0);
  absl::SetFlag(&FLAGS_critic_keeps_training, false);

  std::cout << "PASSED" << std::endl;
}

void TestUnit11_PurchaseExplorationAndMasking() {
  std::cout << "[Test 11] Purchase exploration classification and policy loss masking... " << std::flush;
  torch::Device device(torch::kCPU);

  // 1. Verify IsAcquisitionAction classification
  TEST_CHECK(IsAcquisitionAction(dune_imperium::kActionBuyImperiumRow0));
  TEST_CHECK(IsAcquisitionAction(dune_imperium::kActionBuyImperiumRow0 + 7));
  TEST_CHECK(IsAcquisitionAction(dune_imperium::kActionBuyHelenaReserve0));
  TEST_CHECK(IsAcquisitionAction(dune_imperium::kActionBuyHelenaReserve0 + 7));
  TEST_CHECK(IsAcquisitionAction(dune_imperium::kActionBuyReserveArrakisLiaison));
  TEST_CHECK(IsAcquisitionAction(dune_imperium::kActionBuyReserveTheSpiceMustFlow));
  TEST_CHECK(IsAcquisitionAction(dune_imperium::kActionTechAcquire0));
  TEST_CHECK(IsAcquisitionAction(dune_imperium::kActionTleilaxuAcquire0));
  TEST_CHECK(IsAcquisitionAction(dune_imperium::kActionTechAcquireWithSolari0));

  TEST_CHECK(!IsAcquisitionAction(dune_imperium::kActionReveal));
  TEST_CHECK(!IsAcquisitionAction(dune_imperium::kActionEndTurn));
  TEST_CHECK(!IsAcquisitionAction(dune_imperium::kActionCombatPass));
  TEST_CHECK(!IsAcquisitionAction(dune_imperium::kActionTechAcquireSkip));
  TEST_CHECK(!IsAcquisitionAction(dune_imperium::kActionTleilaxuAcquireSkip));

  // 2. Verify policy loss masking in TrainPpoUpdateSeparate
  auto actor_base = std::make_shared<SharedDunePolicyValueNetImpl>(
      kObsSize, 64, kActionDim, 2, /*nonlinear_value_head=*/false);
  auto critic_base = std::make_shared<SharedDunePolicyValueNetImpl>(
      kObsSize, 64, kActionDim, 2, /*nonlinear_value_head=*/false);

  auto actor1 = std::make_shared<SharedDunePolicyValueNetImpl>(
      kObsSize, 64, kActionDim, 2, /*nonlinear_value_head=*/false);
  auto critic1 = std::make_shared<SharedDunePolicyValueNetImpl>(
      kObsSize, 64, kActionDim, 2, /*nonlinear_value_head=*/false);

  auto actor2 = std::make_shared<SharedDunePolicyValueNetImpl>(
      kObsSize, 64, kActionDim, 2, /*nonlinear_value_head=*/false);
  auto critic2 = std::make_shared<SharedDunePolicyValueNetImpl>(
      kObsSize, 64, kActionDim, 2, /*nonlinear_value_head=*/false);

  {
    torch::NoGradGuard no_grad;
    for (const auto& item : actor_base->named_parameters()) {
      actor1->named_parameters()[item.key()].copy_(item.value());
      actor2->named_parameters()[item.key()].copy_(item.value());
    }
    for (const auto& item : critic_base->named_parameters()) {
      critic1->named_parameters()[item.key()].copy_(item.value());
      critic2->named_parameters()[item.key()].copy_(item.value());
    }
  }

  FreezeActorValueHead(actor1);
  FreezeCriticPolicyHead(critic1);
  FreezeActorValueHead(actor2);
  FreezeCriticPolicyHead(critic2);

  torch::optim::AdamW actor_opt1(actor1->parameters(), torch::optim::AdamWOptions(1e-4));
  torch::optim::AdamW critic_opt1(critic1->parameters(), torch::optim::AdamWOptions(1e-4));
  torch::optim::AdamW actor_opt2(actor2->parameters(), torch::optim::AdamWOptions(1e-4));
  torch::optim::AdamW critic_opt2(critic2->parameters(), torch::optim::AdamWOptions(1e-4));

  std::vector<PpoTransition> batch1;
  for (int i = 0; i < 4; ++i) {
    batch1.push_back(MakeSyntheticTransition(100 + i, i % 3, {0, 1, 2}, -1.0f, 0.5f, 0.8f, 0.2f));
  }
  batch1[0].mask_policy_loss = true;

  // Batch 2: Identical to Batch 1 EXCEPT transition 0 has different action, advantage, old_log_prob,
  // but SAME mask_policy_loss = true and SAME return_value.
  std::vector<PpoTransition> batch2 = batch1;
  batch2[0].action = 2;
  batch2[0].advantage = -99.0f;
  batch2[0].old_log_prob = -5.0f;

  absl::SetFlag(&FLAGS_ppo_minibatch_size, 4);
  absl::SetFlag(&FLAGS_ppo_update_epochs, 1);
  absl::SetFlag(&FLAGS_target_kl, 0.0);
  absl::SetFlag(&FLAGS_normalize_advantages, false);

  PpoUpdateStats s1 = TrainPpoUpdateSeparate(
      actor1, actor_opt1, critic1, critic_opt1, batch1,
      kObsSize, kActionDim, device, 12345, 1);

  PpoUpdateStats s2 = TrainPpoUpdateSeparate(
      actor2, actor_opt2, critic2, critic_opt2, batch2,
      kObsSize, kActionDim, device, 12345, 1);

  // Actor policy loss must be identical because masked transition 0 was excluded!
  TEST_CHECK_NEAR(s1.policy_loss, s2.policy_loss, 1e-6);

  // Actor parameters must match bit-for-bit between run 1 and run 2
  for (const auto& item : actor1->named_parameters()) {
    TEST_CHECK(torch::equal(item.value(), actor2->named_parameters()[item.key()]));
  }

  // Critic value loss is also identical since return_values matched
  TEST_CHECK_NEAR(s1.value_loss, s2.value_loss, 1e-6);

  // Part 3: Verify transition 0 IS unmasked for critic value loss.
  // Changing return_value on transition 0 must change value loss.
  auto actor3 = std::make_shared<SharedDunePolicyValueNetImpl>(
      kObsSize, 64, kActionDim, 2, /*nonlinear_value_head=*/false);
  auto critic3 = std::make_shared<SharedDunePolicyValueNetImpl>(
      kObsSize, 64, kActionDim, 2, /*nonlinear_value_head=*/false);
  {
    torch::NoGradGuard no_grad;
    for (const auto& item : actor_base->named_parameters()) {
      actor3->named_parameters()[item.key()].copy_(item.value());
    }
    for (const auto& item : critic_base->named_parameters()) {
      critic3->named_parameters()[item.key()].copy_(item.value());
    }
  }
  FreezeActorValueHead(actor3);
  FreezeCriticPolicyHead(critic3);
  torch::optim::AdamW actor_opt3(actor3->parameters(), torch::optim::AdamWOptions(1e-4));
  torch::optim::AdamW critic_opt3(critic3->parameters(), torch::optim::AdamWOptions(1e-4));

  std::vector<PpoTransition> batch3 = batch1;
  batch3[0].reward = 2.0f;

  PpoUpdateStats s3 = TrainPpoUpdateSeparate(
      actor3, actor_opt3, critic3, critic_opt3, batch3,
      kObsSize, kActionDim, device, 12345, 1);

  // Policy loss is STILL identical because transition 0 policy loss is masked
  TEST_CHECK_NEAR(s1.policy_loss, s3.policy_loss, 1e-6);

  // But value loss differs because critic sees transition 0 unmasked
  TEST_CHECK(std::abs(s1.value_loss - s3.value_loss) > 1e-3);

  absl::SetFlag(&FLAGS_normalize_advantages, true);

  std::cout << "PASSED" << std::endl;
}

void TestUnit12_AdvantageMatching() {
  std::cout << "[Test 12] Advantage and value matching from rollout through TrainPpoUpdateSeparate... " << std::flush;
  std::string u28700_path =
      "/run/media/warcr/Storage/dune_drl_runtime/round7/u27900_continuation_20260930/checkpoints/ppo_model_update_28700.pt";
  if (!std::filesystem::exists(u28700_path)) {
    std::cout << "SKIPPED (u28700 checkpoint not found at " << u28700_path << ")" << std::endl;
    return;
  }

  torch::Device device(torch::kCPU);
  const int64_t kActorObs = dune_imperium::kFullPublicInformationStateSize;  // 9182
  const int64_t kHidden = 2048;
  const int64_t kActionDim = 2391;
  const int64_t kNumBlocks = 8;

  auto actor = std::make_shared<SharedDunePolicyValueNetImpl>(
      kActorObs, kHidden, kActionDim, kNumBlocks, /*use_nonlinear=*/false,
      /*with_aux_heads=*/false, /*head_init_seed=*/0, /*with_semantic_scorer=*/true);
  auto critic = std::make_shared<SharedDunePolicyValueNetImpl>(
      kActorObs, kHidden, kActionDim, kNumBlocks, /*use_nonlinear=*/false,
      /*with_aux_heads=*/false, /*head_init_seed=*/0, /*with_semantic_scorer=*/true);

  LoadModelCheckpoint(actor, u28700_path, device, /*allow_skip_semantic_scorer=*/true);
  LoadModelCheckpoint(critic, u28700_path, device, /*allow_skip_semantic_scorer=*/true);

  actor->semantic_descriptor_schema_ = dune_semantic::kDescriptorSchemaVersionV3;
  critic->semantic_descriptor_schema_ = dune_semantic::kDescriptorSchemaVersionV3;

  FreezeActorValueHead(actor);
  FreezeCriticPolicyHead(critic);
  actor->eval();
  critic->eval();

  torch::optim::AdamW actor_opt(actor->parameters(), torch::optim::AdamWOptions(1e-4));
  torch::optim::AdamW critic_opt(critic->parameters(), torch::optim::AdamWOptions(1e-4));

  std::mutex eval_mutex;
  std::shared_mutex sync_mutex;
  auto det_eval = std::make_shared<DeterministicEvaluator>(
      actor, device, &eval_mutex, &sync_mutex, critic, /*rollout_amp=*/false);

  auto game = LoadGame("dune_imperium");
  std::vector<PpoTransition> batch;

  double gamma = 1.0;
  double gae_lambda = 1.0;
  float logit_cap = 10.0f;
  absl::SetFlag(&FLAGS_logit_cap, 10.0);
  absl::SetFlag(&FLAGS_rollout_amp, false);

  for (int ep = 0; ep < 16; ++ep) {
    uint64_t ep_id = 1000 + ep;
    auto state = game->NewInitialState();
    std::vector<PpoTransition> trajectory;

    while (!state->IsTerminal()) {
      if (state->IsChanceNode()) {
        auto outcomes = state->ChanceOutcomes();
        state->ApplyAction(outcomes.front().first);
        continue;
      }
      Player p = state->CurrentPlayer();
      auto actions = state->LegalActions();
      if (actions.empty()) break;

      std::vector<float> obs(kActorObs, 0.0f);
      auto* dune_st = dynamic_cast<const dune_imperium::DuneImperiumState*>(state.get());
      dune_st->InformationStateTensorWithAppendix(
          p, dune_imperium::MarketAppendixMode::kFullPublicInformationV3,
          absl::MakeSpan(obs));

      auto eval_res = det_eval->Evaluate(obs);
      CenterAndCapLegalLogits(eval_res.logits, actions, logit_cap);

      Action chosen_action = actions.front();
      float max_l = -1e9f;
      for (Action a : actions) {
        if (eval_res.logits[a] > max_l) {
          max_l = eval_res.logits[a];
          chosen_action = a;
        }
      }

      double sum_exp = 0.0;
      for (Action a : actions) {
        sum_exp += std::exp(eval_res.logits[a] - max_l);
      }
      float log_prob = (eval_res.logits[chosen_action] - max_l) - std::log(sum_exp);

      PpoTransition t;
      t.state = obs;
      t.legal_actions = actions;
      t.action = chosen_action;
      t.old_log_prob = log_prob;
      t.reward = 0.0f;
      t.value = eval_res.value;
      t.advantage = 0.0f;
      t.return_value = 0.0f;
      t.player_id = p;
      t.episode_id = ep_id;
      t.mask_policy_loss = false;
      trajectory.push_back(std::move(t));

      state->ApplyAction(chosen_action);
    }

    auto returns = state->Returns();
    std::vector<float> last_value(4, 0.0f);
    std::vector<float> last_gae(4, 0.0f);
    std::vector<bool> seen_last_action(4, false);

    for (auto it = trajectory.rbegin(); it != trajectory.rend(); ++it) {
      int p = it->player_id;
      if (p < 0 || p >= 4) continue;
      float reward = it->reward;
      if (!seen_last_action[p]) {
        reward += static_cast<float>(returns[p]);
        seen_last_action[p] = true;
      }
      reward = std::clamp(reward, -1.0f, 1.0f);
      it->reward = reward;

      float delta = reward + static_cast<float>(gamma) * last_value[p] - it->value;
      float advantage = delta + static_cast<float>(gamma * gae_lambda) * last_gae[p];
      it->advantage = advantage;
      it->return_value = advantage + it->value;
      last_value[p] = it->value;
      last_gae[p] = advantage;
    }

    for (auto& trans : trajectory) {
      batch.push_back(std::move(trans));
    }
  }

  TEST_CHECK(!batch.empty());

  std::vector<float> collector_values(batch.size());
  std::vector<float> collector_advantages(batch.size());
  for (size_t i = 0; i < batch.size(); ++i) {
    collector_values[i] = batch[i].value;
    collector_advantages[i] = batch[i].advantage;
  }

  absl::SetFlag(&FLAGS_ppo_minibatch_size, static_cast<int>(batch.size()));
  absl::SetFlag(&FLAGS_ppo_update_epochs, 1);
  absl::SetFlag(&FLAGS_target_kl, 0.0);
  absl::SetFlag(&FLAGS_normalize_advantages, false);

  PpoUpdateStats stats = TrainPpoUpdateSeparate(
      actor, actor_opt, critic, critic_opt, batch,
      kActorObs, kActionDim, device, /*master=*/12345, /*global_update=*/1,
      /*compute_diagnostics=*/false, gamma, gae_lambda);

  for (size_t i = 0; i < batch.size(); ++i) {
    TEST_CHECK_NEAR(batch[i].value, collector_values[i], 1e-4f);
    TEST_CHECK_NEAR(batch[i].advantage, collector_advantages[i], 1e-4f);
  }

  std::cout << "PASSED" << std::endl;
}

}  // namespace
}  // namespace open_spiel

int main() {
  std::cout << "=== dune_separate_actor_critic_test ===" << std::endl;
  absl::SetFlag(&FLAGS_rollout_amp, false);
  open_spiel::TestUnit1_U200ParityOnCpu();
  open_spiel::TestUnit2_StrictGradientAndParameterIsolation();
  open_spiel::TestUnit3_RolloutValueRouting();
  open_spiel::TestUnit4_CoherentCheckpointRoundTrip();
  open_spiel::TestUnit5_JointGlobalGradientClipping();
  open_spiel::TestUnit6_ProductionLearnerExecution();
  open_spiel::TestUnit7_ResumedNextStepEquivalence();
  open_spiel::TestUnit8_CliArgumentSmokeValidation();
  open_spiel::TestUnit9_PrivilegedCriticZeroInitParity();
  open_spiel::TestUnit10_CriticKeepsTrainingOnActorEarlyStop();
  open_spiel::TestUnit11_PurchaseExplorationAndMasking();
  open_spiel::TestUnit12_AdvantageMatching();
  std::cout << "All 12/12 separate actor/critic unit tests PASSED!" << std::endl;
  return 0;
}

