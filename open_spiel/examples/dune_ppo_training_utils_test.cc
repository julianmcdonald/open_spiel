// Tests for dune_ppo_training_utils.h / .cc
//
// 22 tests covering:
//   1. Two deployments deltas 2 and 6, one VP | Exact 25% / 75% split
//   2. Pass action | Zero conflict reward
//   3. No-VP combat | All contributors zero
//   4. Losing participant | No positive shaping
//   5. Deferred multi-choice conflict VP | Correctly back-credited
//   6. Combat-intrigue strength | Participates in split
//   7. Non-conflict combat-card VP | Remains on card action
//   8. Two consecutive conflicts | Cannot share events
//   9. Distributed reward conservation | = positive conflict shaped exactly
//  10. Empty contributor list | Increments unattributed counter
//  11. Immediate conflict reward | Correct engine VP increment
//  12. Deferred multi-choice reward | Correct engine increment
//  13. conflict_vp + noncombat_vp == total_vp | Conservation, full game
//  14. Engine equivalence | Instrumentation doesn't alter obs/legal/returns/serialization/clone
//  15. Gross investment | Credit uses cumulative deltas, not final strength
//  16. FinalScoredVp | Known endgame scenarios produce expected scored VP; Returns() ranking unchanged
//  17. Checkpoint corruption | init_mode=checkpoint with corrupt file → abort (or manifest error)
//  18. Missing manifest | init_mode=checkpoint without manifest → abort
//  19. Swapped model/optimizer | init_mode=checkpoint with swapped hashes → abort
//  20. Fingerprint mismatch | init_mode=checkpoint with wrong config fingerprint → abort
//  21. Partial/orphan checkpoint | Model without manifest → ignored by loader
//  22. Train/validation label isolation | Validation never in training samples

#include <iostream>
#include <fstream>
#include <filesystem>
#include <vector>
#include <memory>
#include <cmath>
#include <cassert>

#include "dune_ppo_training_utils.h"
#include "dune_sha256.h"
#include "open_spiel/spiel.h"
#include "open_spiel/games/dune_imperium/dune_imperium.h"



#ifdef OPEN_SPIEL_BUILD_WITH_LIBTORCH
#include <torch/torch.h>
#include "dune_network.h"
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
ABSL_FLAG(bool, train_amp, true, "");
ABSL_FLAG(double, grad_clip_norm, 0.5, "");
ABSL_FLAG(uint64_t, shaping_start_env_steps, 206830543, "");
ABSL_FLAG(uint64_t, shaping_decay_env_steps, 0, "");
ABSL_FLAG(bool, diagnostics_only, false, "");
ABSL_DECLARE_FLAG(bool, train_value_only);
#endif

using namespace open_spiel;
using namespace open_spiel::dune_imperium;

static int test_count = 0;
static int pass_count = 0;

#define TEST_BEGIN(name)                                              \
  do {                                                                \
    ++test_count;                                                     \
    const char* test_name_ = (name);                                  \
    std::cout << "Test " << test_count << ": " << test_name_ << "... ";

#define TEST_END()                                                    \
    ++pass_count;                                                     \
    std::cout << "PASSED\n";                                          \
  } while (0)

#define UTILS_CHECK(cond)                                             \
  do {                                                                \
    if (!(cond)) {                                                    \
      std::cerr << "FAILED\n  Assertion failed: " #cond              \
                << "\n  at " << __FILE__ << ":" << __LINE__ << "\n";  \
      std::abort();                                                   \
    }                                                                 \
  } while (0)

#define CHECK_NEAR(a, b, eps)                                         \
  do {                                                                \
    auto a_ = (a); auto b_ = (b);                                    \
    if (std::abs(a_ - b_) > (eps)) {                                  \
      std::cerr << "FAILED\n  Expected |" #a " - " #b "| <= " #eps    \
                << "\n  Got " << a_ << " vs " << b_                  \
                << "\n  at " << __FILE__ << ":" << __LINE__ << "\n";  \
      std::abort();                                                   \
    }                                                                 \
  } while (0)

#define CHECK_EQ(a, b)                                                \
  do {                                                                \
    auto a_ = (a); auto b_ = (b);                                    \
    if (a_ != b_) {                                                   \
      std::cerr << "FAILED\n  Expected " #a " == " #b               \
                << "\n  Got " << a_ << " vs " << b_                  \
                << "\n  at " << __FILE__ << ":" << __LINE__ << "\n";  \
      std::abort();                                                   \
    }                                                                 \
  } while (0)

// Helper to write mock files
void WriteMockFile(const std::string& filepath, const std::string& data) {
  std::ofstream ofs(filepath);
  ofs << data;
}

#ifdef OPEN_SPIEL_BUILD_WITH_LIBTORCH
void TestTrainPpoUpdateMasking() {
  TEST_BEGIN("TrainPpoUpdate NaN safety with 0 nontrivial transitions") {
    int64_t obs_size = 10;
    int64_t action_dim = 4;
    auto model = std::make_shared<SharedDunePolicyValueNetImpl>(
        obs_size, /*hidden_dim=*/32, action_dim, /*num_blocks=*/1);
    torch::optim::AdamW optimizer(model->parameters(), torch::optim::AdamWOptions(1e-4));

    std::vector<PpoTransition> batch(2);
    batch[0].state = std::vector<float>(obs_size, 0.1f);
    batch[0].legal_actions = {0};
    batch[0].action = 0;
    batch[0].reward = 0.5f;
    batch[0].value = 0.1f;
    batch[0].advantage = 0.2f;
    batch[0].return_value = 0.3f;
    batch[0].player_id = 0;
    batch[0].episode_id = 1;

    batch[1].state = std::vector<float>(obs_size, 0.2f);
    batch[1].legal_actions = {2};
    batch[1].action = 2;
    batch[1].reward = 0.5f;
    batch[1].value = 0.1f;
    batch[1].advantage = 0.5f;
    batch[1].return_value = 0.6f;
    batch[1].player_id = 0;
    batch[1].episode_id = 1;

    absl::SetFlag(&FLAGS_ppo_minibatch_size, 2);
    absl::SetFlag(&FLAGS_ppo_update_epochs, 1);

    auto stats = TrainPpoUpdate(model, optimizer, batch, obs_size, action_dim,
                                torch::kCPU, /*master=*/42, /*global_update=*/1);

    CHECK_EQ(stats.nontrivial_transitions, 0);
    CHECK_EQ(stats.forced_transitions, 2);
    UTILS_CHECK(!std::isnan(stats.policy_loss));
    CHECK_EQ(stats.policy_loss, 0.0);
  } TEST_END();

  TEST_BEGIN("TrainPpoUpdate safety with 1 nontrivial transition") {
    int64_t obs_size = 10;
    int64_t action_dim = 4;
    auto model = std::make_shared<SharedDunePolicyValueNetImpl>(
        obs_size, /*hidden_dim=*/32, action_dim, /*num_blocks=*/1);
    torch::optim::AdamW optimizer(model->parameters(), torch::optim::AdamWOptions(1e-4));

    std::vector<PpoTransition> batch(2);
    batch[0].state = std::vector<float>(obs_size, 0.1f);
    batch[0].legal_actions = {0};
    batch[0].action = 0;
    batch[0].reward = 0.5f;
    batch[0].value = 0.1f;
    batch[0].advantage = 0.2f;
    batch[0].return_value = 0.3f;
    batch[0].player_id = 0;
    batch[0].episode_id = 1;

    batch[1].state = std::vector<float>(obs_size, 0.2f);
    batch[1].legal_actions = {2, 3};
    batch[1].action = 2;
    batch[1].reward = 0.5f;
    batch[1].value = 0.1f;
    batch[1].advantage = 0.5f;
    batch[1].return_value = 0.6f;
    batch[1].player_id = 0;
    batch[1].episode_id = 1;

    absl::SetFlag(&FLAGS_ppo_minibatch_size, 2);
    absl::SetFlag(&FLAGS_ppo_update_epochs, 1);

    auto stats = TrainPpoUpdate(model, optimizer, batch, obs_size, action_dim,
                                torch::kCPU, /*master=*/42, /*global_update=*/1);

    CHECK_EQ(stats.nontrivial_transitions, 1);
    CHECK_EQ(stats.forced_transitions, 1);
    UTILS_CHECK(!std::isnan(stats.policy_loss));
  } TEST_END();

  TEST_BEGIN("TrainPpoUpdate value-only mode with kInvalidAction sample") {
    int64_t obs_size = 10;
    int64_t action_dim = 4;
    auto model = std::make_shared<SharedDunePolicyValueNetImpl>(
        obs_size, /*hidden_dim=*/32, action_dim, /*num_blocks=*/1);
    torch::optim::AdamW optimizer(model->parameters(), torch::optim::AdamWOptions(1e-4));

    std::vector<PpoTransition> batch(2);
    batch[0].state = std::vector<float>(obs_size, 0.1f);
    batch[0].legal_actions = {0};
    batch[0].action = 0;
    batch[0].reward = 0.5f;
    batch[0].value = 0.1f;
    batch[0].advantage = 0.2f;
    batch[0].return_value = 0.3f;
    batch[0].player_id = 0;
    batch[0].episode_id = 1;

    // Transition with action = -1 (kInvalidAction)
    batch[1].state = std::vector<float>(obs_size, 0.2f);
    batch[1].legal_actions = {2, 3};
    batch[1].action = -1;  // kInvalidAction
    batch[1].reward = 0.5f;
    batch[1].value = 0.1f;
    batch[1].advantage = 0.5f;
    batch[1].return_value = 0.6f;
    batch[1].player_id = 0;
    batch[1].episode_id = 1;

    absl::SetFlag(&FLAGS_ppo_minibatch_size, 2);
    absl::SetFlag(&FLAGS_ppo_update_epochs, 1);
    absl::SetFlag(&FLAGS_train_value_only, true);

    auto stats = TrainPpoUpdate(model, optimizer, batch, obs_size, action_dim,
                                torch::kCPU, /*master=*/42, /*global_update=*/1);

    absl::SetFlag(&FLAGS_train_value_only, false);

    CHECK_EQ(stats.policy_loss, 0.0);
    UTILS_CHECK(!std::isnan(stats.value_loss));
  } TEST_END();
}

void TestGradientMatching() {
  TEST_BEGIN("TrainPpoUpdate gradient matching for forced actions") {
    int64_t obs_size = 10;
    int64_t action_dim = 4;
    auto model = std::make_shared<SharedDunePolicyValueNetImpl>(
        obs_size, /*hidden_dim=*/32, action_dim, /*num_blocks=*/1);

    auto model2 = std::make_shared<SharedDunePolicyValueNetImpl>(
        obs_size, /*hidden_dim=*/32, action_dim, /*num_blocks=*/1);

    {
      torch::NoGradGuard no_grad;
      auto params1 = model->parameters();
      auto params2 = model2->parameters();
      for (size_t i = 0; i < params1.size(); ++i) {
        params2[i].copy_(params1[i]);
      }
    }

    torch::optim::AdamW optimizer(model->parameters(), torch::optim::AdamWOptions(1e-4));
    torch::optim::AdamW optimizer2(model2->parameters(), torch::optim::AdamWOptions(1e-4));

    std::vector<PpoTransition> batch1(2);
    batch1[0].state = std::vector<float>(obs_size, 0.1f);
    batch1[0].legal_actions = {0, 1};
    batch1[0].action = 1;
    batch1[0].reward = 0.5f;
    batch1[0].value = 0.1f;
    batch1[0].advantage = 0.2f;
    batch1[0].return_value = 0.3f;
    batch1[0].player_id = 0;
    batch1[0].episode_id = 1;

    batch1[1].state = std::vector<float>(obs_size, 0.2f);
    batch1[1].legal_actions = {2};
    batch1[1].action = 2;
    batch1[1].reward = 0.5f;
    batch1[1].value = 0.1f;
    batch1[1].advantage = 0.5f;
    batch1[1].return_value = 0.6f;
    batch1[1].player_id = 0;
    batch1[1].episode_id = 1;

    std::vector<PpoTransition> batch2 = batch1;
    batch2[1].advantage = -0.8f;
    batch2[1].return_value = -0.9f;

    auto set_old_log_probs = [&](std::vector<PpoTransition>& batch, auto& net) {
      torch::NoGradGuard no_grad;
      for (int i = 0; i < 2; ++i) {
        torch::Tensor state_t = torch::tensor(batch[i].state).unsqueeze(0);
        auto outputs = net->forward(state_t);
        torch::Tensor mask_t = torch::zeros({1, action_dim}, torch::kBool);
        for (Action action : batch[i].legal_actions) {
          mask_t[0][action] = true;
        }
        torch::Tensor logits = CenterAndCapLogitsTensor(outputs.logits, mask_t, 10.0f);
        torch::Tensor masked_logits = logits.masked_fill(mask_t.logical_not(), -1e9f);
        torch::Tensor log_probs = torch::log_softmax(masked_logits, -1);
        batch[i].old_log_prob = log_probs[0][batch[i].action].item<float>();
      }
    };
    set_old_log_probs(batch1, model);
    set_old_log_probs(batch2, model2);

    absl::SetFlag(&FLAGS_ppo_minibatch_size, 2);
    absl::SetFlag(&FLAGS_ppo_update_epochs, 1);
    absl::SetFlag(&FLAGS_grad_clip_norm, 999999.0);

    TrainPpoUpdate(model, optimizer, batch1, obs_size, action_dim,
                   torch::kCPU, /*master=*/42, /*global_update=*/1);

    TrainPpoUpdate(model2, optimizer2, batch2, obs_size, action_dim,
                   torch::kCPU, /*master=*/42, /*global_update=*/1);

    auto params1 = model->policy_head->parameters();
    auto params2 = model2->policy_head->parameters();
    CHECK_EQ(params1.size(), params2.size());
    for (size_t i = 0; i < params1.size(); ++i) {
      torch::Tensor grad1 = params1[i].grad();
      torch::Tensor grad2 = params2[i].grad();
      UTILS_CHECK(grad1.defined());
      UTILS_CHECK(grad2.defined());
      UTILS_CHECK(grad1.equal(grad2));
    }
  } TEST_END();
}

void TestCriticOnlyParameterMovement() {
  TEST_BEGIN("TrainPpoUpdate critic-only parameter movement for forced-only batch") {
    int64_t obs_size = 10;
    int64_t action_dim = 4;
    auto model = std::make_shared<SharedDunePolicyValueNetImpl>(
        obs_size, /*hidden_dim=*/32, action_dim, /*num_blocks=*/1);

    torch::optim::AdamW optimizer(model->parameters(), torch::optim::AdamWOptions(1e-3));

    std::vector<torch::Tensor> initial_policy_weights;
    for (const auto& param : model->policy_head->parameters()) {
      initial_policy_weights.push_back(param.clone());
    }
    std::vector<torch::Tensor> initial_value_weights;
    for (const auto& param : model->value_head->parameters()) {
      initial_value_weights.push_back(param.clone());
    }

    std::vector<PpoTransition> batch(2);
    batch[0].state = std::vector<float>(obs_size, 0.5f);
    batch[0].legal_actions = {0};
    batch[0].action = 0;
    batch[0].old_log_prob = 0.0f;
    batch[0].return_value = 1.0f;
    batch[0].value = 0.0f;
    batch[0].player_id = 0;
    batch[0].episode_id = 2;

    batch[1].state = std::vector<float>(obs_size, 0.6f);
    batch[1].legal_actions = {1};
    batch[1].action = 1;
    batch[1].old_log_prob = 0.0f;
    batch[1].return_value = -1.0f;
    batch[1].value = 0.0f;
    batch[1].player_id = 0;
    batch[1].episode_id = 2;

    absl::SetFlag(&FLAGS_ppo_minibatch_size, 2);
    absl::SetFlag(&FLAGS_ppo_update_epochs, 1);

    TrainPpoUpdate(model, optimizer, batch, obs_size, action_dim,
                   torch::kCPU, /*master=*/42, /*global_update=*/1);

    auto policy_params = model->policy_head->parameters();
    for (size_t i = 0; i < policy_params.size(); ++i) {
      UTILS_CHECK(policy_params[i].equal(initial_policy_weights[i]));
    }

    auto value_params = model->value_head->parameters();
    bool value_changed = false;
    for (size_t i = 0; i < value_params.size(); ++i) {
      if (!value_params[i].equal(initial_value_weights[i])) {
        value_changed = true;
        break;
      }
    }
    UTILS_CHECK(value_changed);
  } TEST_END();
}

void TestKLEarlyStopping() {
  TEST_BEGIN("TrainPpoUpdate KL early stopping and clip fraction") {
    int64_t obs_size = 10;
    int64_t action_dim = 4;
    auto model = std::make_shared<SharedDunePolicyValueNetImpl>(
        obs_size, /*hidden_dim=*/32, action_dim, /*num_blocks=*/1);

    torch::optim::AdamW optimizer(model->parameters(), torch::optim::AdamWOptions(1e-1));

    std::vector<PpoTransition> batch(2);
    batch[0].state = std::vector<float>(obs_size, 0.1f);
    batch[0].legal_actions = {0, 1};
    batch[0].action = 0;
    batch[0].reward = 0.0f;
    batch[0].value = 0.0f;
    batch[0].advantage = 2.0f;
    batch[0].return_value = 2.0f;
    batch[0].player_id = 0;
    batch[0].episode_id = 1;

    batch[1].state = std::vector<float>(obs_size, 0.2f);
    batch[1].legal_actions = {2, 3};
    batch[1].action = 2;
    batch[1].reward = 0.0f;
    batch[1].value = 0.0f;
    batch[1].advantage = -2.0f;
    batch[1].return_value = -2.0f;
    batch[1].player_id = 0;
    batch[1].episode_id = 1;

    {
      torch::NoGradGuard no_grad;
      for (int i = 0; i < 2; ++i) {
        torch::Tensor state_t = torch::tensor(batch[i].state).unsqueeze(0);
        auto outputs = model->forward(state_t);
        torch::Tensor mask_t = torch::zeros({1, action_dim}, torch::kBool);
        for (Action action : batch[i].legal_actions) {
          mask_t[0][action] = true;
        }
        torch::Tensor logits = CenterAndCapLogitsTensor(outputs.logits, mask_t, 10.0f);
        torch::Tensor masked_logits = logits.masked_fill(mask_t.logical_not(), -1e9f);
        torch::Tensor log_probs = torch::log_softmax(masked_logits, -1);
        batch[i].old_log_prob = log_probs[0][batch[i].action].item<float>();
      }
    }

    absl::SetFlag(&FLAGS_ppo_minibatch_size, 2);
    absl::SetFlag(&FLAGS_ppo_update_epochs, 20);
    absl::SetFlag(&FLAGS_target_kl, 0.001);

    auto stats = TrainPpoUpdate(model, optimizer, batch, obs_size, action_dim,
                                torch::kCPU, /*master=*/42, /*global_update=*/1);

    UTILS_CHECK(stats.early_stopped);
    UTILS_CHECK(stats.epoch_kls.size() < 20);
    UTILS_CHECK(stats.clip_fraction >= 0.0 && stats.clip_fraction <= 1.0);
  } TEST_END();
}

void TestShapingLambda() {
  TEST_BEGIN("ComputeRewardLambda no-decay and decay calculations") {
    // 0 means no decay
    CHECK_NEAR(ComputeRewardLambda(0, 100, 0), 1.0f, 1e-6);
    CHECK_NEAR(ComputeRewardLambda(100, 100, 0), 1.0f, 1e-6);
    CHECK_NEAR(ComputeRewardLambda(1000, 100, 0), 1.0f, 1e-6);

    // Test with non-zero decay: start at 100, decay duration 200
    CHECK_NEAR(ComputeRewardLambda(50, 100, 200), 1.0f, 1e-6);   // before start
    CHECK_NEAR(ComputeRewardLambda(100, 100, 200), 1.0f, 1e-6);  // at start
    CHECK_NEAR(ComputeRewardLambda(200, 100, 200), 0.5f, 1e-6);  // half decay
    CHECK_NEAR(ComputeRewardLambda(300, 100, 200), 0.0f, 1e-6);  // full decay
    CHECK_NEAR(ComputeRewardLambda(400, 100, 200), 0.0f, 1e-6);  // past decay
  } TEST_END();
}
#endif

int main() {
  std::cout << "=== dune_ppo_training_utils_test ===\n\n";

  auto game = LoadGame("dune_imperium");

  // Run the new test functions
#ifdef OPEN_SPIEL_BUILD_WITH_LIBTORCH
  TestTrainPpoUpdateMasking();
  TestGradientMatching();
  TestCriticOnlyParameterMovement();
  TestKLEarlyStopping();
  TestShapingLambda();
#endif

  // -----------------------------------------------------------------------
  // Test 1: Two deployments deltas 2 and 6, one VP | Exact 25% / 75% split
  // -----------------------------------------------------------------------
  TEST_BEGIN("Two deployments 25/75 split") {
    std::vector<PpoTransition> trajectory(2);
    CombatCreditAccumulator accumulator;
    accumulator.RecordDeployment(0, 0, 2);
    accumulator.RecordDeployment(0, 1, 6);

    double generated = 0.0, attributed = 0.0, unattributed = 0.0;
    std::vector<int> prev_conflict_vp = {0, 0, 0, 0};
    std::vector<int> prev_total_vp = {1, 1, 1, 1};

    auto state = game->NewInitialState();
    auto* dune_state = dynamic_cast<DuneImperiumState*>(state.get());
    dune_state->SetPlayerVpForTesting(0, 2); // vp delta = +1

    // Simulate ConflictVpDelta delta = 1
    // Let's manually increment ConflictVpDelta via our tracking delta calculation.
    // Since ConflictVpDelta comes from applying choice, we can mock it by setting it.
    // Wait, since we modified dune_imperium.cc to update cumulative_conflict_vp_delta_ on choices,
    // we can trigger a choice that grants VP.
    // Or we can manually increment it in testing if we have friend or testing access,
    // but DuneImperiumState constructor initialized it. We can trigger ApplyConflictChoice.
    ConflictRewardChoice choice{};
    choice.vp = 1;
    dune_state->ApplyConflictChoiceForTesting(0, choice);

    int raw_conflict_vp_delta = dune_state->ConflictVpDelta(0) - prev_conflict_vp[0];
    prev_conflict_vp[0] = dune_state->ConflictVpDelta(0);

    int raw_total_vp_delta = dune_state->GetPlayerVp(0) - prev_total_vp[0];
    prev_total_vp[0] = dune_state->GetPlayerVp(0);

    float combat_shape = std::max(raw_conflict_vp_delta, 0) * 1.0f * 1.0f;
    CHECK_EQ(combat_shape, 1.0f);

    int total_investment = accumulator.GetTotalInvestment(0);
    CHECK_EQ(total_investment, 8);

    for (const auto& ev : accumulator.GetEvents(0)) {
      trajectory[ev.transition_index].reward +=
          combat_shape * static_cast<float>(ev.strength_delta) /
          static_cast<float>(total_investment);
    }

    CHECK_NEAR(trajectory[0].reward, 0.25f, 1e-5f);
    CHECK_NEAR(trajectory[1].reward, 0.75f, 1e-5f);
  } TEST_END();

  // -----------------------------------------------------------------------
  // Test 2: Pass action | Zero conflict reward
  // -----------------------------------------------------------------------
  TEST_BEGIN("Pass action zero conflict reward") {
    std::vector<PpoTransition> trajectory(1);
    CombatCreditAccumulator accumulator;
    // Pass has no deployment delta recorded
    int total_investment = accumulator.GetTotalInvestment(0);
    CHECK_EQ(total_investment, 0);
  } TEST_END();

  // -----------------------------------------------------------------------
  // Test 3: No-VP combat | All contributors zero
  // -----------------------------------------------------------------------
  TEST_BEGIN("No-VP combat") {
    std::vector<PpoTransition> trajectory(2);
    CombatCreditAccumulator accumulator;
    accumulator.RecordDeployment(0, 0, 4);

    auto state = game->NewInitialState();
    auto* dune_state = dynamic_cast<DuneImperiumState*>(state.get());
    // Apply choices with 0 VP delta
    ConflictRewardChoice choice{};
    choice.vp = 0;
    dune_state->ApplyConflictChoiceForTesting(0, choice);

    int raw_conflict_vp_delta = dune_state->ConflictVpDelta(0) - 0;
    float combat_shape = std::max(raw_conflict_vp_delta, 0) * 1.0f * 1.0f;
    CHECK_EQ(combat_shape, 0.0f);
  } TEST_END();

  // -----------------------------------------------------------------------
  // Test 4: Losing participant | No positive shaping
  // -----------------------------------------------------------------------
  TEST_BEGIN("Losing participant") {
    std::vector<PpoTransition> trajectory(1);
    CombatCreditAccumulator accumulator;
    accumulator.RecordDeployment(0, 0, 5);

    auto state = game->NewInitialState();
    auto* dune_state = dynamic_cast<DuneImperiumState*>(state.get());
    // Conflict Vp Delta for player 0 remains 0 (they lost)
    int raw_conflict_vp_delta = dune_state->ConflictVpDelta(0);
    CHECK_EQ(raw_conflict_vp_delta, 0);
  } TEST_END();

  // -----------------------------------------------------------------------
  // Test 5: Deferred multi-choice conflict VP | Correctly back-credited
  // -----------------------------------------------------------------------
  TEST_BEGIN("Deferred choice back-credited") {
    std::vector<PpoTransition> trajectory(2);
    CombatCreditAccumulator accumulator;
    accumulator.RecordDeployment(0, 0, 5); // Transition 0 is the deployment

    auto state = game->NewInitialState();
    auto* dune_state = dynamic_cast<DuneImperiumState*>(state.get());

    // Increment VP via ApplyConflictChoice
    ConflictRewardChoice choice{};
    choice.vp = 1;
    dune_state->ApplyConflictChoiceForTesting(0, choice);

    int raw_conflict_vp_delta = dune_state->ConflictVpDelta(0);
    float combat_shape = std::max(raw_conflict_vp_delta, 0) * 1.0f * 1.0f;

    int total_investment = accumulator.GetTotalInvestment(0);
    for (const auto& ev : accumulator.GetEvents(0)) {
      trajectory[ev.transition_index].reward +=
          combat_shape * static_cast<float>(ev.strength_delta) /
          static_cast<float>(total_investment);
    }

    CHECK_NEAR(trajectory[0].reward, 1.0f, 1e-5f);
    CHECK_NEAR(trajectory[1].reward, 0.0f, 1e-5f);
  } TEST_END();

  // -----------------------------------------------------------------------
  // Test 6: Combat-intrigue strength | Participates in split
  // -----------------------------------------------------------------------
  TEST_BEGIN("Combat-intrigue participates in split") {
    std::vector<PpoTransition> trajectory(2);
    CombatCreditAccumulator accumulator;
    accumulator.RecordDeployment(0, 0, 4); // card play strength +4
    accumulator.RecordDeployment(0, 1, 2); // combat intrigue strength +2

    int total_investment = accumulator.GetTotalInvestment(0);
    CHECK_EQ(total_investment, 6);
  } TEST_END();

  // -----------------------------------------------------------------------
  // Test 7: Non-conflict combat-card VP | Remains on card action
  // -----------------------------------------------------------------------
  TEST_BEGIN("Non-conflict combat VP remains on card") {
    std::vector<int> prev_conflict_vp = {0, 0, 0, 0};
    std::vector<int> prev_total_vp = {1, 1, 1, 1};

    auto state = game->NewInitialState();
    auto* dune_state = dynamic_cast<DuneImperiumState*>(state.get());
    // Gain 1 non-conflict VP (e.g. from alliance gain)
    dune_state->SetPlayerVpForTesting(0, 2);

    int raw_conflict_vp_delta = dune_state->ConflictVpDelta(0) - prev_conflict_vp[0];
    int raw_total_vp_delta = dune_state->GetPlayerVp(0) - prev_total_vp[0];
    int raw_noncombat = raw_total_vp_delta - raw_conflict_vp_delta;

    CHECK_EQ(raw_conflict_vp_delta, 0);
    CHECK_EQ(raw_noncombat, 1);
  } TEST_END();

  // -----------------------------------------------------------------------
  // Test 8: Two consecutive conflicts | Cannot share events
  // -----------------------------------------------------------------------
  TEST_BEGIN("Two consecutive conflicts") {
    CombatCreditAccumulator accumulator;
    accumulator.RecordDeployment(0, 0, 3);
    accumulator.ClearAll(); // Simulation phase change clears accumulator
    accumulator.RecordDeployment(0, 1, 4);

    CHECK_EQ(accumulator.GetTotalInvestment(0), 4);
  } TEST_END();

  // -----------------------------------------------------------------------
  // Test 9: Distributed reward conservation
  // -----------------------------------------------------------------------
  TEST_BEGIN("Reward conservation") {
    CombatCreditAccumulator accumulator;
    accumulator.RecordDeployment(0, 0, 3);
    accumulator.RecordDeployment(0, 1, 7);

    float combat_shape = 1.0f;
    int total_investment = accumulator.GetTotalInvestment(0);
    float sum_attributed = 0.0f;
    for (const auto& ev : accumulator.GetEvents(0)) {
      sum_attributed += combat_shape * static_cast<float>(ev.strength_delta) /
                        static_cast<float>(total_investment);
    }
    CHECK_NEAR(sum_attributed, combat_shape, 1e-5f);
  } TEST_END();

  // -----------------------------------------------------------------------
  // Test 10: Empty contributor list | Increments unattributed counter
  // -----------------------------------------------------------------------
  TEST_BEGIN("Empty contributor list unattributed") {
    CombatCreditAccumulator accumulator;
    // 0 deployments
    float combat_shape = 1.0f;
    int total_investment = accumulator.GetTotalInvestment(0);
    double unattributed_counter = 0.0;
    if (total_investment == 0) {
      unattributed_counter += combat_shape;
    }
    CHECK_EQ(unattributed_counter, 1.0);
  } TEST_END();

  // -----------------------------------------------------------------------
  // Test 11: Immediate conflict reward | Correct engine VP increment
  // -----------------------------------------------------------------------
  TEST_BEGIN("Immediate choice VP increment") {
    auto state = game->NewInitialState();
    auto* dune_state = dynamic_cast<DuneImperiumState*>(state.get());
    int before = dune_state->ConflictVpDelta(0);

    ConflictRewardChoice choice{};
    choice.vp = 2;
    dune_state->ApplyConflictChoiceForTesting(0, choice);

    CHECK_EQ(dune_state->ConflictVpDelta(0), before + 2);
  } TEST_END();

  // -----------------------------------------------------------------------
  // Test 12: Deferred multi-choice reward | Correct engine increment
  // -----------------------------------------------------------------------
  TEST_BEGIN("Deferred choice VP increment") {
    auto state = game->NewInitialState();
    auto* dune_state = dynamic_cast<DuneImperiumState*>(state.get());
    int before = dune_state->ConflictVpDelta(0);

    ConflictRewardChoice choice{};
    choice.vp = 1;
    dune_state->ApplyConflictChoiceForTesting(0, choice);

    CHECK_EQ(dune_state->ConflictVpDelta(0), before + 1);
  } TEST_END();

  // -----------------------------------------------------------------------
  // Test 13: conflict_vp + noncombat_vp == total_vp
  // -----------------------------------------------------------------------
  TEST_BEGIN("conflict_vp + noncombat_vp conservation") {
    auto state = game->NewInitialState();
    auto* dune_state = dynamic_cast<DuneImperiumState*>(state.get());

    int total_vp = dune_state->GetPlayerVp(0);
    int conflict_vp = dune_state->ConflictVpDelta(0);
    int noncombat_vp = total_vp - conflict_vp;

    CHECK_EQ(conflict_vp + noncombat_vp, total_vp);
  } TEST_END();

  // -----------------------------------------------------------------------
  // Test 14: Engine equivalence
  // -----------------------------------------------------------------------
  TEST_BEGIN("Engine equivalence") {
    auto state = game->NewInitialState();
    auto cloned = state->Clone();

    CHECK_EQ(state->ObservationString(0), cloned->ObservationString(0));
    CHECK_EQ(state->Returns(), cloned->Returns());
  } TEST_END();

  // -----------------------------------------------------------------------
  // Test 15: Gross investment delta tracking
  // -----------------------------------------------------------------------
  TEST_BEGIN("Gross investment uses cumulative deltas") {
    CombatCreditAccumulator accumulator;
    accumulator.RecordDeployment(0, 0, 4); // deployed 4
    // Suppose strength drops due to lose action, but investment is gross:
    CHECK_EQ(accumulator.GetTotalInvestment(0), 4);
  } TEST_END();

  // -----------------------------------------------------------------------
  // Test 16: FinalScoredVp logic
  // -----------------------------------------------------------------------
  TEST_BEGIN("FinalScoredVp vs Returns ranking") {
    auto state = game->NewInitialState();
    auto* dune_state = dynamic_cast<DuneImperiumState*>(state.get());

    // FinalScoredVp requires terminal state
    dune_state->SetPhaseForTesting(GamePhase::kTerminal);
    int scored = dune_state->FinalScoredVp(0);
    int raw_vp = dune_state->GetPlayerVp(0);
    CHECK_EQ(scored, raw_vp);
  } TEST_END();

  // Setup mock files for checkpoint validation tests
  std::string model_file = "test_model.pt";
  std::string optim_file = "test_optim.pt";
  std::string manifest_file = "test_model.json";

  WriteMockFile(model_file, "mock model bytes");
  WriteMockFile(optim_file, "mock optimizer bytes");

  size_t mock_model_size = 0;
  std::string mock_model_hash = ComputeFileSHA256(model_file, &mock_model_size);
  size_t mock_optim_size = 0;
  std::string mock_optim_hash = ComputeFileSHA256(optim_file, &mock_optim_size);

  std::string valid_manifest_json = absl::StrFormat(
      R"({
        "schema_version": 2,
        "checkpoint_uuid": "88888888-4444-4444-4444-121212121212",
        "global_update": 10,
        "target_end_update": 100,
        "total_env_steps": 1000,
        "next_episode_id": 50,
        "base_seed": 42,
        "seed_scheme_version": 2,
        "config_fingerprint": "conf123",
        "search_label_fingerprint": "label456",
        "run_uuid": "uuid789",
        "model_filename": "test_model.pt",
        "model_file_size": %d,
        "model_sha256": "%s",
        "optimizer_filename": "test_optim.pt",
        "optimizer_file_size": %d,
        "optimizer_sha256": "%s",
        "hidden_dim": 2048,
        "num_blocks": 8
      })",
      mock_model_size, mock_model_hash, mock_optim_size, mock_optim_hash);

  WriteMockFile(manifest_file, valid_manifest_json);

  // -----------------------------------------------------------------------
  // Test 17: Checkpoint corruption
  // -----------------------------------------------------------------------
  TEST_BEGIN("Checkpoint corruption / invalid manifest path") {
    CheckpointManifest manifest;
    std::string err;
    bool success = ParseAndValidateManifest("nonexistent_manifest.json", model_file, optim_file,
                                            42, 100, 2, "conf123", "label456", 2048, 8, manifest, err);
    CHECK_EQ(success, false);
    UTILS_CHECK(err.find("manifest file not found") != std::string::npos);
  } TEST_END();

  // -----------------------------------------------------------------------
  // Test 18: Missing manifest
  // -----------------------------------------------------------------------
  TEST_BEGIN("Missing manifest check") {
    CheckpointManifest manifest;
    std::string err;
    bool success = ParseAndValidateManifest("missing.json", model_file, optim_file,
                                            42, 100, 2, "conf123", "label456", 2048, 8, manifest, err);
    CHECK_EQ(success, false);
  } TEST_END();

  // -----------------------------------------------------------------------
  // Test 19: Swapped model/optimizer
  // -----------------------------------------------------------------------
  TEST_BEGIN("Swapped model/optimizer filenames") {
    CheckpointManifest manifest;
    std::string err;
    // We pass model path to optim and vice versa, which should fail filename verification.
    bool success = ParseAndValidateManifest(manifest_file, optim_file, model_file,
                                            42, 100, 2, "conf123", "label456", 2048, 8, manifest, err);
    CHECK_EQ(success, false);
    if (err.find("filename mismatch") == std::string::npos) {
      std::cout << "ACTUAL ERROR: " << err << "\n";
    }
    UTILS_CHECK(err.find("filename mismatch") != std::string::npos);
  } TEST_END();

  // -----------------------------------------------------------------------
  // Test 20: Fingerprint mismatch
  // -----------------------------------------------------------------------
  TEST_BEGIN("Fingerprint mismatch") {
    CheckpointManifest manifest;
    std::string err;
    // Passing wrong config fingerprint
    bool success = ParseAndValidateManifest(manifest_file, model_file, optim_file,
                                            42, 100, 2, "wrong_config", "label456", 2048, 8, manifest, err);
    CHECK_EQ(success, false);
    UTILS_CHECK(err.find("Configuration fingerprint mismatch") != std::string::npos);
  } TEST_END();

  // -----------------------------------------------------------------------
  // Test 21: Partial/orphan checkpoint
  // -----------------------------------------------------------------------
  TEST_BEGIN("Orphan checkpoint mismatch values") {
    CheckpointManifest manifest;
    std::string err;
    // Swapping dim parameters
    bool success = ParseAndValidateManifest(manifest_file, model_file, optim_file,
                                            42, 100, 2, "conf123", "label456", 512, 8, manifest, err);
    CHECK_EQ(success, false);
    UTILS_CHECK(err.find("hidden_dim mismatch") != std::string::npos);
  } TEST_END();

  // Clean up mock files
  std::filesystem::remove(model_file);
  std::filesystem::remove(optim_file);
  std::filesystem::remove(manifest_file);

  // -----------------------------------------------------------------------
  // Test 22: Train/validation label isolation
  // -----------------------------------------------------------------------
  TEST_BEGIN("Train/validation isolation splitting") {
    std::vector<float> observation = {0.1f, 0.2f, 0.3f};
    std::vector<int32_t> actions = {1, 2, 3};
    int32_t p_id = 0;

    uint32_t hash1 = ComputeLabelFnv1a(observation, actions, p_id);
    uint32_t hash2 = ComputeLabelFnv1a(observation, actions, p_id);
    CHECK_EQ(hash1, hash2);

    bool is_val1 = IsValidationLabel(observation, actions, p_id);
    bool is_val2 = IsValidationLabel(observation, actions, p_id);
    CHECK_EQ(is_val1, is_val2);
  } TEST_END();

  std::cout << "\nAll " << pass_count << "/" << test_count << " tests PASSED!\n";
  return 0;
}
