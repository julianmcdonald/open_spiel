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
ABSL_FLAG(bool, rollout_amp, false, "");
ABSL_FLAG(bool, allow_tf32, false, "");
ABSL_FLAG(double, grad_clip_norm, 0.5, "");
ABSL_FLAG(bool, diagnostics_only, false, "");

ABSL_FLAG(int, hidden_dim, 2048, "");
ABSL_FLAG(int, num_blocks, 8, "");
ABSL_FLAG(int, seed_scheme_version, 2, "");
ABSL_FLAG(std::string, model_checkpoint, "", "");
ABSL_FLAG(std::string, optim_checkpoint, "", "");

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
  TEST_CHECK(std::filesystem::exists(kU200Path));
  size_t file_size = 0;
  std::string actual_sha = ComputeFileSHA256(kU200Path, &file_size);
  TEST_CHECK(actual_sha == "62245429e7bd93b39d768b80b25a4264900e0c5fa221124f074d3787f0080118");

  torch::Device device(torch::kCPU);
  auto u200_net = std::make_shared<SharedDunePolicyValueNetImpl>(
      kObsSize, kHiddenDim, kActionDim, kNumBlocks, /*nonlinear_value_head=*/false);
  auto actor_net = std::make_shared<SharedDunePolicyValueNetImpl>(
      kObsSize, kHiddenDim, kActionDim, kNumBlocks, /*nonlinear_value_head=*/false);
  auto critic_net = std::make_shared<SharedDunePolicyValueNetImpl>(
      kObsSize, kHiddenDim, kActionDim, kNumBlocks, /*nonlinear_value_head=*/false);

  {
    torch::serialize::InputArchive arc1, arc2, arc3;
    arc1.load_from(kU200Path, device);
    u200_net->load(arc1);
    arc2.load_from(kU200Path, device);
    actor_net->load(arc2);
    arc3.load_from(kU200Path, device);
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
    float old_log_prob, float advantage, float return_val, float value) {
  PpoTransition t;
  t.episode_id = episode_id;
  t.state.assign(kObsSize, 0.05f);
  t.action = action;
  t.legal_actions = legal_actions;
  t.old_log_prob = old_log_prob;
  t.advantage = advantage;
  t.return_value = return_val;
  t.value = value;
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

}  // namespace
}  // namespace open_spiel

int main() {
  std::cout << "=== dune_separate_actor_critic_test ===" << std::endl;
  open_spiel::TestUnit1_U200ParityOnCpu();
  open_spiel::TestUnit2_StrictGradientAndParameterIsolation();
  open_spiel::TestUnit3_RolloutValueRouting();
  open_spiel::TestUnit4_CoherentCheckpointRoundTrip();
  open_spiel::TestUnit5_JointGlobalGradientClipping();
  open_spiel::TestUnit6_ProductionLearnerExecution();
  open_spiel::TestUnit7_ResumedNextStepEquivalence();
  open_spiel::TestUnit8_CliArgumentSmokeValidation();
  std::cout << "All 8/8 separate actor/critic unit tests PASSED!" << std::endl;
  return 0;
}
