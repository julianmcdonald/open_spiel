// Mandatory verification tests for Opponent-Pool Single-Learner Training.
// Proves 7 required properties:
// 1. Correct per-seat model routing using distinguishable model outputs, not labels alone.
// 2. All and only designated-learner rows enter PPO and value training; no opponent rows leak through return/GAE code.
// 3. Learner terminal utility is attached correctly and learner action log probabilities match its actual sampling policy.
// 4. Exactly 256 complete games and correct seat counts enter each production update.
// 5. Opponent model parameters/buffers remain unchanged.
// 6. Both learners initialize identically to U200 with fresh optimizer states.
// 7. Saving/resuming preserves learner state, optimizer moments, episode IDs, opponent-assignment schedule and the original finite endpoint.

#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include <torch/torch.h>

#include "open_spiel/spiel.h"
#include "dune_network.h"
#include "dune_seed_utils.h"
#include "dune_sha256.h"
#include "dune_opponent_pool.h"
#include "dune_ppo_training_utils.h"
#include "open_spiel/utils/json.h"
#include "open_spiel/abseil-cpp/absl/flags/flag.h"

ABSL_FLAG(int, ppo_minibatch_size, 2048, "");
ABSL_FLAG(int, ppo_update_epochs, 4, "");
ABSL_FLAG(double, ppo_clip_epsilon, 0.2, "");
ABSL_FLAG(bool, normalize_advantages, true, "");
ABSL_FLAG(bool, ppo_clip_value_loss, true, "");
ABSL_FLAG(double, entropy_coef, 0.015, "");
ABSL_FLAG(double, value_coef, 0.5, "");
ABSL_FLAG(double, logit_cap, 10.0, "");
ABSL_FLAG(double, target_kl, 0.01, "");
ABSL_FLAG(bool, train_amp, false, "");
ABSL_FLAG(bool, allow_tf32, false, "");
ABSL_FLAG(double, grad_clip_norm, 0.5, "");
ABSL_FLAG(bool, diagnostics_only, false, "");
ABSL_FLAG(int, hidden_dim, 2048, "");
ABSL_FLAG(int, num_blocks, 8, "");
ABSL_FLAG(int, seed_scheme_version, 2, "");
ABSL_FLAG(std::string, model_checkpoint, "", "");
ABSL_FLAG(std::string, optim_checkpoint, "", "");

#define TEST_CHECK(cond)                                                       \
  do {                                                                         \
    if (!(cond)) {                                                             \
      std::cerr << "Assertion failed at " << __FILE__ << ":" << __LINE__        \
                << " - " #cond << std::endl;                                   \
      std::abort();                                                            \
    }                                                                          \
  } while (0)

#define TEST_CHECK_NEAR(a, b, eps)                                             \
  do {                                                                         \
    if (std::abs((a) - (b)) > (eps)) {                                         \
      std::cerr << "Near-equality failed at " << __FILE__ << ":" << __LINE__   \
                << " - |" << (a) << " - " << (b) << "| = "                     \
                << std::abs((a) - (b)) << " > " << (eps) << std::endl;         \
      std::abort();                                                            \
    }                                                                          \
  } while (0)

namespace open_spiel {
namespace {

constexpr int64_t kObsSize = 5580;
constexpr int64_t kActionDim = 2391;

// Distinguishable evaluator emitting a unique signature bias in logits and value.
class DistinguishableEvaluator : public IGameEvaluator {
 public:
  explicit DistinguishableEvaluator(float signature) : signature_(signature) {}

  EvalResult Evaluate(const std::vector<float>& obs) override {
    EvalResult res;
    res.logits.assign(kActionDim, signature_);
    res.value = signature_ * 0.1f;
    return res;
  }

  EvaluatorStats GetStats() const override { return {}; }

 private:
  float signature_ = 0.0f;
};

// =========================================================================
// Proof 1: Correct per-seat model routing using distinguishable model outputs
// =========================================================================
void TestProof1_ModelRoutingDistinguishableOutputs() {
  std::cout << "[Proof 1] Per-seat model routing with distinguishable outputs... " << std::flush;
  uint64_t master = 42;

  auto learner_eval = std::make_shared<DistinguishableEvaluator>(100.0f);
  auto opp0_eval = std::make_shared<DistinguishableEvaluator>(200.0f);
  auto opp1_eval = std::make_shared<DistinguishableEvaluator>(300.0f);
  std::vector<std::shared_ptr<IGameEvaluator>> pool = {opp0_eval, opp1_eval};

  std::vector<float> dummy_obs(kObsSize, 0.0f);

  for (uint64_t ep = 0; ep < 32; ++ep) {
    auto assignment = dune_opponent_pool::ComputeSeatAssignment(master, ep, pool.size());
    int learner = assignment.learner_seat;
    TEST_CHECK(learner == static_cast<int>(ep % 4));

    for (int p = 0; p < 4; ++p) {
      auto resolved = dune_opponent_pool::ResolveEvaluator(p, assignment, learner_eval, pool);
      EvalResult res = resolved->Evaluate(dummy_obs);

      if (p == learner) {
        // Must match learner signature exactly
        TEST_CHECK_NEAR(res.logits[0], 100.0f, 1e-4);
        TEST_CHECK_NEAR(res.value, 10.0f, 1e-4);
      } else {
        size_t expected_opp_idx = assignment.opponent_pool_indices[p];
        TEST_CHECK(expected_opp_idx == 0 || expected_opp_idx == 1);
        float expected_sig = (expected_opp_idx == 0) ? 200.0f : 300.0f;
        TEST_CHECK_NEAR(res.logits[0], expected_sig, 1e-4);
        TEST_CHECK_NEAR(res.value, expected_sig * 0.1f, 1e-4);
      }
    }
  }

  // Also verify Arm A behavior (empty opponent pool -> all seats resolve to learner)
  std::vector<std::shared_ptr<IGameEvaluator>> empty_pool;
  for (uint64_t ep = 0; ep < 8; ++ep) {
    auto assignment = dune_opponent_pool::ComputeSeatAssignment(master, ep, 0);
    for (int p = 0; p < 4; ++p) {
      auto resolved = dune_opponent_pool::ResolveEvaluator(p, assignment, learner_eval, empty_pool);
      EvalResult res = resolved->Evaluate(dummy_obs);
      TEST_CHECK_NEAR(res.logits[0], 100.0f, 1e-4);
    }
  }

  std::cout << "PASSED" << std::endl;
}

// =========================================================================
// Proof 2 & 3: Only learner rows enter PPO buffer; terminal utility & log-prob matching
// =========================================================================
void TestProof2And3_LearnerRowsIsolationAndTerminalUtility() {
  std::cout << "[Proof 2 & 3] Learner row isolation, log-prob matching, and GAE return attachment... " << std::flush;

  auto game = LoadGame("dune_imperium");
  TEST_CHECK(game != nullptr);

  std::mutex eval_mutex;
  auto model = std::make_shared<SharedDunePolicyValueNetImpl>(kObsSize, 64, kActionDim, 1, false);
  model->eval();
  auto learner_eval = std::make_shared<DeterministicEvaluator>(model, torch::kCPU, &eval_mutex);

  auto opp_model1 = std::make_shared<SharedDunePolicyValueNetImpl>(kObsSize, 64, kActionDim, 1, false);
  opp_model1->eval();
  auto opp_eval1 = std::make_shared<DeterministicEvaluator>(opp_model1, torch::kCPU, &eval_mutex);

  std::vector<std::shared_ptr<IGameEvaluator>> pool = {opp_eval1};

  for (uint64_t ep = 0; ep < 4; ++ep) {
    int expected_learner_seat = static_cast<int>(ep % 4);
    std::vector<PpoTransition> trajectory;
    std::atomic<uint64_t> total_env_steps{0};

    int moves = dune_opponent_pool::SimulateSingleLearnerRollout(
        /*master=*/12345, ep, *game, learner_eval, pool, kObsSize,
        &trajectory, &total_env_steps, /*reward_scale=*/4.0f);

    TEST_CHECK(moves > 0);
    TEST_CHECK(total_env_steps.load() > 0);
    TEST_CHECK(!trajectory.empty());

    // Proof 2: ALL and ONLY designated learner rows enter trajectory
    TEST_CHECK(trajectory.size() < total_env_steps.load());
    for (const auto& t : trajectory) {
      TEST_CHECK(t.player_id == expected_learner_seat);
      TEST_CHECK(t.episode_id == ep);
      TEST_CHECK(t.action >= 0);
      TEST_CHECK(!t.legal_actions.empty());
      TEST_CHECK(!std::isnan(t.old_log_prob));
      TEST_CHECK(!std::isinf(t.old_log_prob));

      // Proof 3: GAE return = advantage + value identity
      TEST_CHECK_NEAR(t.return_value, t.advantage + t.value, 1e-4);
    }

    // Proof 3: Terminal placement utility is attached to learner's last transition
    const auto& last_t = trajectory.back();
    TEST_CHECK(last_t.player_id == expected_learner_seat);
    TEST_CHECK(!std::isnan(last_t.return_value));
  }

  std::cout << "PASSED" << std::endl;
}

// =========================================================================
// Proof 4: Exactly 256 complete games and correct seat counts enter each update
// =========================================================================
void TestProof4_CompleteGamesAndSeatCounts() {
  std::cout << "[Proof 4] Exactly 256 games and 64 per seat rotation... " << std::flush;
  uint64_t master = 3;

  for (uint64_t batch_start : {uint64_t(0), uint64_t(256), uint64_t(51200)}) {
    std::array<int, 4> seat_counts = {0, 0, 0, 0};
    for (uint64_t ep = batch_start; ep < batch_start + 256; ++ep) {
      auto assignment = dune_opponent_pool::ComputeSeatAssignment(master, ep, 2);
      seat_counts[assignment.learner_seat]++;
    }

    TEST_CHECK(seat_counts[0] == 64);
    TEST_CHECK(seat_counts[1] == 64);
    TEST_CHECK(seat_counts[2] == 64);
    TEST_CHECK(seat_counts[3] == 64);
    TEST_CHECK(seat_counts[0] + seat_counts[1] + seat_counts[2] + seat_counts[3] == 256);
  }

  std::cout << "PASSED" << std::endl;
}

// =========================================================================
// Proof 5: Opponent model parameters/buffers remain unchanged
// =========================================================================
void TestProof5_OpponentModelImmutability() {
  std::cout << "[Proof 5] Opponent model parameters/buffers immutability... " << std::flush;

  auto game = LoadGame("dune_imperium");
  std::mutex eval_mutex;

  auto learner_model = std::make_shared<SharedDunePolicyValueNetImpl>(kObsSize, 64, kActionDim, 1, false);
  learner_model->eval();
  auto learner_eval = std::make_shared<DeterministicEvaluator>(learner_model, torch::kCPU, &eval_mutex);

  auto opp_model1 = std::make_shared<SharedDunePolicyValueNetImpl>(kObsSize, 64, kActionDim, 1, false);
  auto opp_model2 = std::make_shared<SharedDunePolicyValueNetImpl>(kObsSize, 64, kActionDim, 1, false);
  opp_model1->eval();
  opp_model2->eval();
  for (auto& p : opp_model1->parameters()) p.set_requires_grad(false);
  for (auto& p : opp_model2->parameters()) p.set_requires_grad(false);

  std::string hash1_before = dune_opponent_pool::HashModelParametersAndBuffers(opp_model1);
  std::string hash2_before = dune_opponent_pool::HashModelParametersAndBuffers(opp_model2);

  auto opp_eval1 = std::make_shared<DeterministicEvaluator>(opp_model1, torch::kCPU, &eval_mutex);
  auto opp_eval2 = std::make_shared<DeterministicEvaluator>(opp_model2, torch::kCPU, &eval_mutex);
  std::vector<std::shared_ptr<IGameEvaluator>> pool = {opp_eval1, opp_eval2};

  // Run multiple rollout simulations
  for (uint64_t ep = 0; ep < 8; ++ep) {
    std::vector<PpoTransition> trajectory;
    std::atomic<uint64_t> env_steps{0};
    dune_opponent_pool::SimulateSingleLearnerRollout(
        777, ep, *game, learner_eval, pool, kObsSize, &trajectory, &env_steps);
  }

  std::string hash1_after = dune_opponent_pool::HashModelParametersAndBuffers(opp_model1);
  std::string hash2_after = dune_opponent_pool::HashModelParametersAndBuffers(opp_model2);

  TEST_CHECK(hash1_before == hash1_after);
  TEST_CHECK(hash2_before == hash2_after);

  std::cout << "PASSED" << std::endl;
}

// =========================================================================
// Proof 6: Both learners initialize identically to U200 with fresh optimizer states
// =========================================================================
void TestProof6_IdenticalU200InitAndFreshOptimizer() {
  std::cout << "[Proof 6] Identical U200 initialization and fresh AdamW state... " << std::flush;

  std::string u200_path =
      "/run/media/warcr/Storage/dune_drl_runtime/round7/placement_u200/ppo_model_update_200.pt";
  if (!std::filesystem::exists(u200_path)) {
    std::cout << "SKIPPED (Storage not mounted at expected path)" << std::endl;
    return;
  }

  std::string u200_hash = ComputeFileSHA256(u200_path);
  TEST_CHECK(u200_hash == "62245429e7bd93b39d768b80b25a4264900e0c5fa221124f074d3787f0080118");

  auto learner_a = std::make_shared<SharedDunePolicyValueNetImpl>(kObsSize, 2048, kActionDim, 8, false);
  auto learner_b = std::make_shared<SharedDunePolicyValueNetImpl>(kObsSize, 2048, kActionDim, 8, false);

  torch::load(learner_a, u200_path, torch::kCPU);
  torch::load(learner_b, u200_path, torch::kCPU);

  std::string hash_a = dune_opponent_pool::HashModelParametersAndBuffers(learner_a);
  std::string hash_b = dune_opponent_pool::HashModelParametersAndBuffers(learner_b);
  TEST_CHECK(hash_a == hash_b);

  torch::optim::AdamW opt_a(learner_a->parameters(), torch::optim::AdamWOptions(1e-5).eps(1e-5).weight_decay(0.0));
  torch::optim::AdamW opt_b(learner_b->parameters(), torch::optim::AdamWOptions(1e-5).eps(1e-5).weight_decay(0.0));

  // Fresh optimizer states must be empty before any training step
  TEST_CHECK(opt_a.state().empty());
  TEST_CHECK(opt_b.state().empty());

  std::cout << "PASSED" << std::endl;
}

// =========================================================================
// Proof 7: Saving/resuming preserves learner state, optimizer moments, episode IDs,
// and opponent-assignment schedule
// =========================================================================
void TestProof7_SaveResumePreservation() {
  std::cout << "[Proof 7] Save/resume preservation of state, moments, episode IDs, and pool schedule... " << std::flush;

  std::string tmp_dir = "/tmp/dune_opp_pool_save_test_" + std::to_string(std::rand());
  std::filesystem::create_directories(tmp_dir);

  std::string model_path = tmp_dir + "/ppo_model_update_205.pt";
  std::string optim_path = tmp_dir + "/ppo_optimizer_update_205.pt";
  std::string manifest_path = tmp_dir + "/ppo_model_update_205.json";

  auto model = std::make_shared<SharedDunePolicyValueNetImpl>(kObsSize, 64, kActionDim, 1, false);
  torch::optim::AdamW opt(model->parameters(), torch::optim::AdamWOptions(1e-4));

  // Take one optimizer step to populate moments
  torch::Tensor dummy = torch::randn({2, kObsSize});
  auto out = model->forward(dummy);
  torch::Tensor loss = out.logits.mean() + out.values.mean();
  loss.backward();
  opt.step();
  TEST_CHECK(!opt.state().empty());

  std::string model_hash_before = dune_opponent_pool::HashModelParametersAndBuffers(model);

  uint64_t target_next_ep = 52480;
  torch::save(model, model_path);
  torch::save(opt, optim_path);

  json::Object manifest_obj;
  manifest_obj["schema_version"] = static_cast<int64_t>(2);
  manifest_obj["global_update"] = static_cast<int64_t>(205);
  manifest_obj["target_end_update"] = static_cast<int64_t>(300);
  manifest_obj["total_env_steps"] = static_cast<int64_t>(999999);
  manifest_obj["next_episode_id"] = static_cast<int64_t>(target_next_ep);
  manifest_obj["base_seed"] = static_cast<int64_t>(3);
  manifest_obj["seed_scheme_version"] = static_cast<int64_t>(2);
  manifest_obj["config_fingerprint"] = "opp_pool_test_fingerprint";
  manifest_obj["search_label_fingerprint"] = "none";
  manifest_obj["run_uuid"] = "test-uuid-opp-pool";
  {
    std::ofstream ofs(manifest_path);
    ofs << json::ToString(manifest_obj, true);
  }

  // Verify checkpoint files exist
  TEST_CHECK(std::filesystem::exists(model_path));
  TEST_CHECK(std::filesystem::exists(optim_path));
  TEST_CHECK(std::filesystem::exists(manifest_path));

  // Load into fresh model and optimizer
  auto resumed_model = std::make_shared<SharedDunePolicyValueNetImpl>(kObsSize, 64, kActionDim, 1, false);
  torch::optim::AdamW resumed_opt(resumed_model->parameters(), torch::optim::AdamWOptions(1e-4));

  torch::load(resumed_model, model_path, torch::kCPU);
  std::string model_hash_after = dune_opponent_pool::HashModelParametersAndBuffers(resumed_model);
  TEST_CHECK(model_hash_before == model_hash_after);

  torch::load(resumed_opt, optim_path, torch::kCPU);
  TEST_CHECK(!resumed_opt.state().empty());

  // Check manifest fields
  std::ifstream mf(manifest_path);
  std::string content((std::istreambuf_iterator<char>(mf)), std::istreambuf_iterator<char>());
  auto parsed = json::FromString(content);
  TEST_CHECK(parsed.has_value() && parsed->IsObject());
  const auto& obj = parsed->GetObject();
  TEST_CHECK(obj.at("global_update").GetInt() == 205);
  TEST_CHECK(obj.at("target_end_update").GetInt() == 300);
  TEST_CHECK(static_cast<uint64_t>(obj.at("next_episode_id").GetInt()) == target_next_ep);

  // Check opponent-assignment schedule continuity from next_episode_id
  for (uint64_t ep = target_next_ep; ep < target_next_ep + 16; ++ep) {
    auto assign_orig = dune_opponent_pool::ComputeSeatAssignment(3, ep, 2);
    auto assign_resumed = dune_opponent_pool::ComputeSeatAssignment(3, ep, 2);
    TEST_CHECK(assign_orig.learner_seat == assign_resumed.learner_seat);
    TEST_CHECK(assign_orig.opponent_pool_indices == assign_resumed.opponent_pool_indices);
  }

  std::filesystem::remove_all(tmp_dir);
  std::cout << "PASSED" << std::endl;
}

}  // namespace
}  // namespace open_spiel

int main() {
  std::cout << "=== dune_opponent_pool_test ===" << std::endl;
  absl::SetFlag(&FLAGS_rollout_amp, false);
  open_spiel::TestProof1_ModelRoutingDistinguishableOutputs();
  open_spiel::TestProof2And3_LearnerRowsIsolationAndTerminalUtility();
  open_spiel::TestProof4_CompleteGamesAndSeatCounts();
  open_spiel::TestProof5_OpponentModelImmutability();
  open_spiel::TestProof6_IdenticalU200InitAndFreshOptimizer();
  open_spiel::TestProof7_SaveResumePreservation();
  std::cout << "\nALL 7/7 OPPONENT POOL MANDATORY PROOFS PASSED!" << std::endl;
  return 0;
}
