// Offline verification test for Search Auxiliary Supervision Redesign
// Validates all corrections and invariants required:
// 1. Post-PPO frozen model reference with exact tie preservation and outside action preservation.
// 2. Near-tie stability demonstrating temperature floor prevents KL budget exhaustion.
// 3. Strict KL ceiling enforcement with step rejection & bitwise rollback.
// 4. Bitwise value-head parameter immutability with populated Adam optimizer state & critic shift tracking.
// 5. Explicit null-supervision phase detection and bitwise zero drift.
// 6. Partial rollback vs one-accepted-step control comparing complete optimizer tensors and counters.
// 7. Deployed semantic policy parity on real Dune states within declared tolerances.

#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include <torch/torch.h>

#include "open_spiel/abseil-cpp/absl/flags/flag.h"
#include "open_spiel/abseil-cpp/absl/flags/parse.h"
#include "open_spiel/abseil-cpp/absl/strings/str_format.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"

#include "dune_network.h"
#include "dune_ppo_training_utils.h"
#include "dune_compound_turn_search.h"
#include "dune_semantic_action_scorer.h"
#include "dune_batched_evaluator.h"
#include <shared_mutex>

ABSL_FLAG(std::string, checkpoint_path,
          "/home/warcr/dune_drl_runtime/round7/u22263_search_wholeturn_continuation_20260922_120600/checkpoints/ppo_model_update_22263.pt",
          "Path to saved model checkpoint for offline testing.");
ABSL_FLAG(std::string, optimizer_path,
          "/home/warcr/dune_drl_runtime/round7/u22263_search_wholeturn_continuation_20260922_120600/checkpoints/ppo_optimizer_update_22263.pt",
          "Path to saved optimizer checkpoint for offline testing.");

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
ABSL_FLAG(bool, allow_tf32, true, "");
ABSL_FLAG(double, grad_clip_norm, 0.5, "");
ABSL_FLAG(uint64_t, shaping_start_env_steps, 206830543, "");
ABSL_FLAG(uint64_t, shaping_decay_env_steps, 0, "");
ABSL_FLAG(bool, diagnostics_only, false, "");

#define OFFLINE_CHECK(expr)                                                    \
  do {                                                                         \
    if (!(expr)) {                                                             \
      std::cerr << "FAILED: " #expr " at " << __FILE__ << ":" << __LINE__     \
                << std::endl;                                                  \
      std::exit(1);                                                            \
    }                                                                          \
  } while (0)

#define OFFLINE_CHECK_NEAR(a, b, eps)                                         \
  do {                                                                         \
    if (std::abs((a) - (b)) > (eps)) {                                         \
      std::cerr << "FAILED: " #a " (" << (a) << ") vs " #b " (" << (b)        \
                << "), eps=" << (eps) << " at " << __FILE__ << ":"            \
                << __LINE__ << std::endl;                                      \
      std::exit(1);                                                            \
    }                                                                          \
  } while (0)

namespace open_spiel {
namespace {

void CopyModelWeights(std::shared_ptr<SharedDunePolicyValueNetImpl> source,
                      std::shared_ptr<SharedDunePolicyValueNetImpl> target) {
  torch::NoGradGuard no_grad;
  auto source_params = source->named_parameters();
  auto source_buffers = source->named_buffers();
  for (auto& item : target->named_parameters()) {
    auto* src = source_params.find(item.key());
    if (src != nullptr) {
      item.value().copy_(src->to(item.value().device()));
    }
  }
  for (auto& item : target->named_buffers()) {
    auto* src = source_buffers.find(item.key());
    if (src != nullptr) {
      item.value().copy_(src->to(item.value().device()));
    }
  }
  target->market_appendix_mode_ = source->market_appendix_mode_;
  target->semantic_descriptor_schema_ = source->semantic_descriptor_schema_;
}

std::vector<SearchTrainingExample> CreateMockExamples(int n, int64_t obs_size, bool with_candidates) {
  std::vector<SearchTrainingExample> examples(n);
  for (int i = 0; i < n; ++i) {
    examples[i].observation = std::vector<float>(obs_size, 0.05f * (i % 10 + 1));
    examples[i].legal_actions = {0, 1, 2, 3, 4};
    examples[i].normalized_visits = {0.05, 0.05, 0.05, 0.05, 0.80};
    if (with_candidates) {
      examples[i].candidate_data.actions = {0, 1, 2, 3, 4};
      examples[i].candidate_data.features.assign(5 * dune_semantic::kSemanticFeatDim, 0.1f * (i % 5 + 1));
      examples[i].candidate_data.card_ids.assign(5, 0);
      examples[i].candidate_data.space_ids.assign(5, 0);
      examples[i].candidate_data.supported.assign(5, 1);
      examples[i].candidate_data.roles.assign(5, dune_semantic::ActionRole::kPrimaryCard);
    }
  }
  return examples;
}

std::vector<PpoTransition> CreateMockRefRollout(int n, int64_t obs_size, bool with_candidates) {
  std::vector<PpoTransition> ref_rollout(n);
  for (int i = 0; i < n; ++i) {
    ref_rollout[i].state = std::vector<float>(obs_size, 0.03f * (i % 7 + 1));
    ref_rollout[i].legal_actions = {0, 1, 2, 3, 4};
    ref_rollout[i].action = 0;
    if (with_candidates) {
      ref_rollout[i].candidate_data.actions = {0, 1, 2, 3, 4};
      ref_rollout[i].candidate_data.features.assign(5 * dune_semantic::kSemanticFeatDim, 0.1f * (i % 5 + 1));
      ref_rollout[i].candidate_data.card_ids.assign(5, 0);
      ref_rollout[i].candidate_data.space_ids.assign(5, 0);
      ref_rollout[i].candidate_data.supported.assign(5, 1);
      ref_rollout[i].candidate_data.roles.assign(5, dune_semantic::ActionRole::kPrimaryCard);
    }
  }
  return ref_rollout;
}

void Test1_Correction1_ReferencePolicyAndTiePreservation() {
  std::cout << "\n=== Test 1: Correction 1 - Frozen Reference Policy & Tie-Preservation ===" << std::endl;
  std::vector<Action> legal_actions = {10, 20, 30, 40, 50};
  std::vector<std::pair<Action, double>> ref_pi = {
      {10, 0.45}, {20, 0.25}, {30, 0.15}, {40, 0.10}, {50, 0.05}};

  std::vector<std::pair<Action, double>> tied_utilities = {
      {10, 0.70}, {20, 0.70}, {30, 0.70}};
  double realized_kl = -1.0;
  double realized_tau = -1.0;
  std::vector<double> target = dune_imperium::ComputeMpoRelativeTarget(
      ref_pi, legal_actions, tied_utilities,
      /*base_temperature=*/0.50, /*max_target_kl=*/0.05,
      &realized_kl, &realized_tau);

  OFFLINE_CHECK(target.size() == 5);
  OFFLINE_CHECK_NEAR(realized_kl, 0.0, 1e-12);
  OFFLINE_CHECK_NEAR(realized_tau, 0.50, 1e-9);

  for (size_t i = 0; i < legal_actions.size(); ++i) {
    OFFLINE_CHECK_NEAR(target[i], ref_pi[i].second, 1e-12);
  }

  double h_ref = 0.0, h_target = 0.0;
  for (size_t i = 0; i < legal_actions.size(); ++i) {
    h_ref -= ref_pi[i].second * std::log(ref_pi[i].second);
    h_target -= target[i] * std::log(target[i]);
  }
  OFFLINE_CHECK_NEAR(h_ref, h_target, 1e-12);

  std::vector<std::pair<Action, double>> unequal_utilities = {
      {10, 0.20}, {20, 0.85}, {30, -0.40}};
  target = dune_imperium::ComputeMpoRelativeTarget(
      ref_pi, legal_actions, unequal_utilities,
      /*base_temperature=*/0.50, /*max_target_kl=*/0.05,
      &realized_kl, &realized_tau);

  double cand_target_mass = target[0] + target[1] + target[2];
  OFFLINE_CHECK_NEAR(cand_target_mass, 0.85, 1e-12);
  OFFLINE_CHECK_NEAR(target[3], ref_pi[3].second, 1e-12);
  OFFLINE_CHECK_NEAR(target[4], ref_pi[4].second, 1e-12);
  OFFLINE_CHECK(target[1] > ref_pi[1].second);

  std::cout << "[PASS] Reference policy tie-preservation and outside mass invariant verified." << std::endl;
}

void Test2_Correction4_NearTieStabilityAndTemperatureFloor() {
  std::cout << "\n=== Test 2: Correction 4 - Near-Tie Stability & Temperature Floor ===" << std::endl;
  std::vector<Action> legal_actions = {10, 20, 30};
  std::vector<std::pair<Action, double>> ref_pi = {
      {10, 0.50}, {20, 0.30}, {30, 0.20}};

  std::vector<std::pair<Action, double>> near_tie_utilities = {
      {10, 0.5000}, {20, 0.5005}, {30, 0.5000}};
  double r_kl = 0.0;
  double r_tau = 0.0;
  std::vector<double> target = dune_imperium::ComputeMpoRelativeTarget(
      ref_pi, legal_actions, near_tie_utilities,
      /*base_temperature=*/0.50, /*max_target_kl=*/0.05,
      &r_kl, &r_tau);

  OFFLINE_CHECK_NEAR(r_tau, 0.50, 1e-9);
  OFFLINE_CHECK(r_kl < 0.0001);
  OFFLINE_CHECK(r_kl >= 0.0);
  std::cout << absl::StrFormat("  Near-tie (Delta Q=0.0005): realized_kl=%.8f (budget ceiling=0.05) tau=%.4f\n",
                               r_kl, r_tau);

  std::vector<std::pair<Action, double>> extreme_utilities = {
      {10, 0.0}, {20, 10.0}, {30, 0.0}};
  target = dune_imperium::ComputeMpoRelativeTarget(
      ref_pi, legal_actions, extreme_utilities,
      /*base_temperature=*/0.50, /*max_target_kl=*/0.05,
      &r_kl, &r_tau);
  OFFLINE_CHECK(r_tau > 0.50);
  OFFLINE_CHECK(r_kl <= 0.05001);
  std::cout << absl::StrFormat("  Extreme (Delta Q=10.0):    realized_kl=%.8f (budget ceiling=0.05) tau=%.4f\n",
                               r_kl, r_tau);

  std::cout << "[PASS] Temperature floor invariant verified: near-ties do not exhaust KL budget." << std::endl;
}

void Test3_Correction2_StrictKLEnforcementAndRollback(
    std::shared_ptr<SharedDunePolicyValueNetImpl> model,
    torch::optim::AdamW& optimizer,
    int64_t obs_size, int64_t action_dim, torch::Device device) {
  std::cout << "\n=== Test 3: Correction 2 - Strict KL Enforcement, Step Rejection & State Rollback ===" << std::endl;

  std::vector<torch::Tensor> initial_params;
  for (const auto& p : model->parameters()) {
    initial_params.push_back(p.detach().clone());
  }

  const int num_examples = 64;
  auto search_examples = CreateMockExamples(num_examples, obs_size, /*with_candidates=*/true);
  auto ref_rollout = CreateMockRefRollout(32, obs_size, /*with_candidates=*/true);

  // 1. Test rejection on Step 0 when budget is tight (max_cumulative_kl = 1e-9)
  SearchAuxPhaseResult tight_res = TrainSearchAuxiliaryPhase(
      model, optimizer, search_examples, obs_size, action_dim, device,
      /*max_steps=*/4, /*batch_size=*/32, /*max_cumulative_kl=*/1e-9,
      /*search_loss_coef=*/0.50, ref_rollout);

  OFFLINE_CHECK(tight_res.step_rejected_on_kl == true);
  OFFLINE_CHECK(tight_res.phase_rejected == true);
  OFFLINE_CHECK(tight_res.steps_accepted == 0);

  // Invariant: Model parameters after step 0 rejection must be bitwise IDENTICAL to initial parameters
  size_t p_idx = 0;
  for (const auto& p : model->parameters()) {
    OFFLINE_CHECK(torch::equal(p, initial_params[p_idx++]));
  }
  std::cout << "  Step 0 rejection: 0 accepted steps, phase_rejected=true, parameters rolled back bit-for-bit." << std::endl;
  std::cout << "[PASS] Strict KL enforcement and bitwise rollback verified." << std::endl;
}

void Test4_Correction3_ValueHeadBitwiseImmutabilityWithPopulatedOptimizer(
    std::shared_ptr<SharedDunePolicyValueNetImpl> model,
    torch::optim::AdamW& optimizer,
    int64_t obs_size, int64_t action_dim, torch::Device device) {
  std::cout << "\n=== Test 4: Correction 3 - Value-Head Immutability with Populated Optimizer & Critic Probe ===" << std::endl;

  std::vector<torch::Tensor> v_params_before;
  for (const auto& p : model->value_head->parameters()) {
    v_params_before.push_back(p.detach().clone());
  }
  if (model->use_nonlinear_value_head_ && model->value_head2) {
    for (const auto& p : model->value_head2->parameters()) {
      v_params_before.push_back(p.detach().clone());
    }
  }

  // Populate Adam optimizer state with nonzero first and second moments
  for (auto& group : optimizer.param_groups()) {
    for (auto& param : group.params()) {
      auto key = param.unsafeGetTensorImpl();
      auto it = optimizer.state().find(key);
      if (it == optimizer.state().end()) {
        auto state = std::make_unique<torch::optim::AdamWParamState>();
        state->step(100);
        state->exp_avg(torch::full_like(param, 0.05f));
        state->exp_avg_sq(torch::full_like(param, 0.01f));
        optimizer.state()[key] = std::move(state);
      }
    }
  }
  OFFLINE_CHECK(optimizer.state().size() > 0);

  const int num_examples = 64;
  auto search_examples = CreateMockExamples(num_examples, obs_size, /*with_candidates=*/true);
  auto ref_rollout = CreateMockRefRollout(32, obs_size, /*with_candidates=*/true);

  SearchAuxPhaseResult aux_res = TrainSearchAuxiliaryPhase(
      model, optimizer, search_examples, obs_size, action_dim, device,
      /*max_steps=*/4, /*batch_size=*/32, /*max_cumulative_kl=*/1.0,
      /*search_loss_coef=*/0.20, ref_rollout);

  OFFLINE_CHECK(aux_res.steps_accepted == 4);
  OFFLINE_CHECK(aux_res.value_head_immutable == true);

  // Bit-for-bit check on all value head parameters
  size_t vi = 0;
  for (const auto& p : model->value_head->parameters()) {
    OFFLINE_CHECK(torch::equal(p, v_params_before[vi++]));
  }
  if (model->use_nonlinear_value_head_ && model->value_head2) {
    for (const auto& p : model->value_head2->parameters()) {
      OFFLINE_CHECK(torch::equal(p, v_params_before[vi++]));
    }
  }
  std::cout << "  Value head parameters: 100% bitwise identical after auxiliary phase." << std::endl;
  OFFLINE_CHECK(aux_res.critic_probe_mse_shift >= 0.0);
  OFFLINE_CHECK(std::isfinite(aux_res.critic_probe_mse_shift));

  std::cout << "[PASS] Value head immutability and critic probe shift verified." << std::endl;
}

void Test5_NullSupervisionDetection(
    std::shared_ptr<SharedDunePolicyValueNetImpl> model,
    torch::optim::AdamW& optimizer,
    int64_t obs_size, int64_t action_dim, torch::Device device) {
  std::cout << "\n=== Test 5: Explicit Null-Supervision Detection & Zero Drift ===" << std::endl;

  std::vector<torch::Tensor> initial_params;
  for (const auto& p : model->parameters()) {
    initial_params.push_back(p.detach().clone());
  }
  size_t initial_opt_states = optimizer.state().size();

  // Create examples where target distribution is computed from model's own deployed forward
  const int n = 32;
  auto examples = CreateMockExamples(n, obs_size, /*with_candidates=*/true);
  torch::Tensor s_batch = torch::zeros({n, obs_size}, torch::kFloat32);
  for (int i = 0; i < n; ++i) {
    std::memcpy(s_batch[i].data_ptr<float>(), examples[i].observation.data(), obs_size * sizeof(float));
  }
  s_batch = s_batch.to(device);
  std::vector<const dune_semantic::CandidateActionData*> cands(n);
  for (int i = 0; i < n; ++i) cands[i] = &examples[i].candidate_data;

  torch::Tensor mask = torch::zeros({n, action_dim}, torch::kBool).to(device);
  for (int i = 0; i < n; ++i) {
    for (Action a : examples[i].legal_actions) mask[i][a] = true;
  }

  torch::Tensor probs;
  {
    torch::NoGradGuard no_grad;
    AutocastGuard autocast_guard(c10::DeviceType::CUDA, device.is_cuda() && absl::GetFlag(FLAGS_train_amp));
    auto ao = model->forward(s_batch);
    if (model->with_semantic_scorer_ && model->semantic_scorer_) {
      dune_semantic::ApplySemanticScorerBatch(
          model->semantic_scorer_, ao.trunk, cands, ao.logits, device);
    }
    torch::Tensor logits = CenterAndCapLogitsTensor(ao.logits, mask, 10.0);
    torch::Tensor masked = logits.masked_fill(mask.logical_not(), -1e9f);
    probs = torch::softmax(masked, -1).cpu();
  }

  for (int i = 0; i < n; ++i) {
    for (size_t a_idx = 0; a_idx < examples[i].legal_actions.size(); ++a_idx) {
      Action a = examples[i].legal_actions[a_idx];
      examples[i].normalized_visits[a_idx] = probs[i][a].item<double>();
    }
  }

  SearchAuxPhaseResult null_res = TrainSearchAuxiliaryPhase(
      model, optimizer, examples, obs_size, action_dim, device,
      /*max_steps=*/4, /*batch_size=*/32, /*max_cumulative_kl=*/0.05,
      /*search_loss_coef=*/0.10, /*reference_rollout_sample=*/{});

  OFFLINE_CHECK(null_res.null_supervision_skipped == true);
  OFFLINE_CHECK(null_res.steps_accepted == 0);
  OFFLINE_CHECK_NEAR(null_res.max_prob_movement, 0.0, 1e-9);

  // Assert bitwise identical parameters
  size_t p_idx = 0;
  for (const auto& p : model->parameters()) {
    OFFLINE_CHECK(torch::equal(p, initial_params[p_idx++]));
  }
  OFFLINE_CHECK(optimizer.state().size() == initial_opt_states);
  std::cout << "  Null supervision phase explicitly skipped; bitwise zero movement verified." << std::endl;
  std::cout << "[PASS] Explicit null supervision detection verified." << std::endl;
}

void Test6_PartialRollbackVsControl(
    std::shared_ptr<SharedDunePolicyValueNetImpl> model,
    torch::optim::AdamW& optimizer,
    int64_t obs_size, int64_t action_dim, torch::Device device) {
  std::cout << "\n=== Test 6: Partial Rollback vs 1-Accepted-Step Control ===" << std::endl;

  // Snapshot starting model parameters
  std::vector<torch::Tensor> starting_params;
  for (const auto& p : model->parameters()) {
    starting_params.push_back(p.detach().clone());
  }

  // Create isolated auxiliary optimizer
  std::vector<torch::Tensor> aux_params;
  for (auto& p : model->parameters()) {
    if (p.requires_grad()) aux_params.push_back(p);
  }
  std::vector<torch::optim::OptimizerParamGroup> groups;
  groups.emplace_back(aux_params);
  torch::optim::AdamW aux_opt(groups, torch::optim::AdamWOptions(2.5e-4).eps(1e-5).weight_decay(0.0));

  const int num_examples = 64;
  auto search_examples = CreateMockExamples(num_examples, obs_size, /*with_candidates=*/true);
  auto ref_rollout = CreateMockRefRollout(32, obs_size, /*with_candidates=*/true);

  // 1. Run CONTROL: exactly 1 step accepted
  SearchAuxPhaseResult ctrl_res = TrainSearchAuxiliaryPhase(
      model, aux_opt, search_examples, obs_size, action_dim, device,
      /*max_steps=*/1, /*batch_size=*/32, /*max_cumulative_kl=*/1.0,
      /*search_loss_coef=*/0.10, ref_rollout);
  OFFLINE_CHECK(ctrl_res.steps_accepted == 1);
  double single_step_kl = ctrl_res.cumulative_kl_searched;

  // Snapshot control model and complete optimizer state
  std::vector<torch::Tensor> ctrl_model_params;
  for (const auto& p : model->parameters()) {
    ctrl_model_params.push_back(p.detach().clone());
  }

  struct FullOptState {
    int64_t step;
    torch::Tensor exp_avg;
    torch::Tensor exp_avg_sq;
  };
  std::unordered_map<void*, FullOptState> ctrl_opt_states;
  for (auto& group : aux_opt.param_groups()) {
    for (auto& param : group.params()) {
      auto key = param.unsafeGetTensorImpl();
      auto it = aux_opt.state().find(key);
      if (it != aux_opt.state().end()) {
        auto* s = dynamic_cast<torch::optim::AdamWParamState*>(it->second.get());
        if (s) {
          ctrl_opt_states[key] = {s->step(), s->exp_avg().clone(), s->exp_avg_sq().clone()};
        }
      }
    }
  }
  OFFLINE_CHECK(ctrl_opt_states.size() > 0);

  // 2. Reset model and optimizer to starting state
  size_t p_idx = 0;
  for (auto& p : model->parameters()) {
    p.data().copy_(starting_params[p_idx++]);
  }
  aux_opt.state().clear();

  // 3. Run TREATMENT: max_steps = 2, threshold calibrated to accept step 0 and reject step 1
  double threshold = single_step_kl * 1.5;
  SearchAuxPhaseResult treat_res = TrainSearchAuxiliaryPhase(
      model, aux_opt, search_examples, obs_size, action_dim, device,
      /*max_steps=*/2, /*batch_size=*/32, /*max_cumulative_kl=*/threshold,
      /*search_loss_coef=*/0.10, ref_rollout);

  OFFLINE_CHECK(treat_res.steps_accepted == 1);
  OFFLINE_CHECK(treat_res.step_rejected_on_kl == true);
  OFFLINE_CHECK(treat_res.phase_rejected == false);

  // Compare model parameters: 100% BITWISE EQUAL to control!
  p_idx = 0;
  for (const auto& p : model->parameters()) {
    OFFLINE_CHECK(torch::equal(p, ctrl_model_params[p_idx++]));
  }
  std::cout << "  Model parameters: 100% bitwise equal to 1-accepted-step control." << std::endl;

  // Compare complete optimizer state tensors and counters
  OFFLINE_CHECK(aux_opt.state().size() == ctrl_opt_states.size());
  for (const auto& pair : ctrl_opt_states) {
    auto it = aux_opt.state().find(pair.first);
    OFFLINE_CHECK(it != aux_opt.state().end());
    auto* s_treat = dynamic_cast<torch::optim::AdamWParamState*>(it->second.get());
    OFFLINE_CHECK(s_treat != nullptr);
    OFFLINE_CHECK(s_treat->step() == pair.second.step);
    OFFLINE_CHECK(torch::equal(s_treat->exp_avg(), pair.second.exp_avg));
    OFFLINE_CHECK(torch::equal(s_treat->exp_avg_sq(), pair.second.exp_avg_sq));
  }
  std::cout << "  Optimizer state: complete tensors (step, exp_avg, exp_avg_sq) bitwise equal to control." << std::endl;
  std::cout << "[PASS] Partial rollback vs one-accepted-step control verified." << std::endl;

  // Cleanly restore model to starting state
  p_idx = 0;
  for (auto& p : model->parameters()) {
    p.data().copy_(starting_params[p_idx++]);
  }
  aux_opt.state().clear();
}

void Test7_DeployedSemanticPolicyParityOnRealDuneStates(
    std::shared_ptr<SharedDunePolicyValueNetImpl> model,
    int64_t obs_size, int64_t action_dim, torch::Device device) {
  std::cout << "\n=== Test 7: Deployed Semantic Policy Parity on Real Dune States with Active Corrections ===" << std::endl;

  std::shared_ptr<const Game> game = LoadGame("dune_imperium");
  std::unique_ptr<State> state = game->NewInitialState();

  std::shared_mutex mutex;
  auto coord = std::make_shared<BatchedEvaluator>(
      model, 128, 1, device, &mutex, /*logit_cap=*/10.0f,
      /*device_synchronize=*/false, /*high_priority_stream=*/false,
      /*emit_batch_membership=*/false,
      /*rollout_amp=*/(device.is_cuda() && absl::GetFlag(FLAGS_train_amp)),
      /*allow_tf32=*/absl::GetFlag(FLAGS_allow_tf32));
  BatchedNNEvaluator teacher(coord, 10.0f);

  std::mt19937_64 rng(20260923ULL);
  double max_diff_supported = 0.0;
  int supported_states_evaluated = 0;
  SearchTrainingExample worst_supported_example;

  for (int step = 0; step < 600 && supported_states_evaluated < 16 && !state->IsTerminal(); ++step) {
    if (state->IsChanceNode()) {
      auto outcomes = state->ChanceOutcomes();
      double u = std::generate_canonical<double, 53>(rng);
      Action a = SampleAction(outcomes, u).first;
      state->ApplyAction(a);
      continue;
    }

    auto legals = state->LegalActions();
    if (legals.empty()) break;
    auto prior = teacher.Prior(*state);
    if (prior.empty()) break;

    if (legals.size() > 1) {
      auto* ds = dynamic_cast<dune_imperium::DuneImperiumState*>(state.get());
      auto obs = teacher.GetConsumedObservation(*state, state->CurrentPlayer());
      dune_semantic::CandidateActionData cands;
      dune_semantic::ExtractCandidateDescriptors(
          *ds, legals, &cands, teacher.SemanticDescriptorSchema());
      bool supported = std::any_of(cands.supported.begin(), cands.supported.end(), [](uint8_t x) { return x != 0; });

      if (supported) {
        supported_states_evaluated++;
        torch::NoGradGuard guard;
        AutocastGuard autocast(c10::DeviceType::CUDA, device.is_cuda() && absl::GetFlag(FLAGS_train_amp));
        auto x = torch::tensor(obs).unsqueeze(0).to(device);
        auto output = model->forward(x);
        dune_semantic::ApplySemanticScorerBatch(
            model->semantic_scorer_, output.trunk, {&cands}, output.logits, device);
        auto mask = torch::zeros({1, action_dim}, torch::TensorOptions().dtype(torch::kBool).device(device));
        for (Action a : legals) mask[0][a] = true;
        auto probs = torch::softmax(CenterAndCapLogitsTensor(output.logits, mask, 10.0).masked_fill(mask.logical_not(), -1e9f), -1).cpu();

        double diff = 0.0;
        for (const auto& ap : prior) {
          diff = std::max(diff, std::abs(ap.second - probs[0][ap.first].item<double>()));
        }
        OFFLINE_CHECK(diff < 1e-5);
        if (diff > max_diff_supported) {
          max_diff_supported = diff;
          worst_supported_example.observation = obs;
          worst_supported_example.legal_actions = legals;
          worst_supported_example.candidate_data = cands;
          worst_supported_example.player = state->CurrentPlayer();
          std::vector<std::pair<Action, double>> ties;
          for (Action a : legals) ties.push_back({a, 0.0});
          worst_supported_example.normalized_visits = dune_imperium::ComputeMpoRelativeTarget(
              prior, legals, ties, 0.50, 0.05);
        }
      }
    }
    state->ApplyAction(dune_imperium::PickGreedyAction(prior, legals));
  }

  OFFLINE_CHECK(supported_states_evaluated > 0);
  std::cout << absl::StrFormat(
      "  Evaluated %d supported states with active semantic corrections; max diff vs BatchedEvaluator = %.2e (< 1e-5).\n",
      supported_states_evaluated, max_diff_supported);

  std::vector<torch::Tensor> aux_params;
  for (auto& p : model->parameters()) {
    if (p.requires_grad()) aux_params.push_back(p);
  }
  std::vector<torch::optim::OptimizerParamGroup> groups;
  groups.emplace_back(aux_params);
  torch::optim::AdamW aux_opt(groups, torch::optim::AdamWOptions(1e-5).eps(1e-5).weight_decay(0.0));

  SearchAuxPhaseResult r = TrainSearchAuxiliaryPhase(
      model, aux_opt, {worst_supported_example}, obs_size, action_dim, device,
      /*max_steps=*/1, /*batch_size=*/1, /*max_cumulative_kl=*/0.01,
      /*search_loss_coef=*/0.10, /*reference_rollout_sample=*/{});

  OFFLINE_CHECK(r.null_supervision_skipped == true);
  OFFLINE_CHECK(r.steps_accepted == 0);
  OFFLINE_CHECK_NEAR(r.mean_aux_grad_norm, 0.0, 1e-9);
  OFFLINE_CHECK_NEAR(r.max_prob_movement, 0.0, 1e-6);
  std::cout << "  Worst supported state with exactly tied Q values produced zero gradient norm and was skipped." << std::endl;
  std::cout << "[PASS] Deployed semantic policy parity verified on real Dune states with active semantic actions." << std::endl;
}

void Test8_NanInjectionRejectionAndRollback(
    std::shared_ptr<SharedDunePolicyValueNetImpl> model,
    int64_t obs_size, int64_t action_dim, torch::Device device) {
  std::cout << "\n=== Test 8: Nonfinite Rejection & Parameter Rollback ===" << std::endl;
  std::vector<torch::Tensor> initial_params;
  for (const auto& p : model->parameters()) {
    initial_params.push_back(p.detach().clone());
  }

  std::vector<torch::Tensor> aux_params;
  for (auto& p : model->parameters()) {
    if (p.requires_grad()) aux_params.push_back(p);
  }
  std::vector<torch::optim::OptimizerParamGroup> groups;
  groups.emplace_back(aux_params);
  torch::optim::AdamW aux_opt(groups, torch::optim::AdamWOptions(1e-5).eps(1e-5).weight_decay(0.0));

  auto examples = CreateMockExamples(4, obs_size, /*with_candidates=*/true);
  // Inject NaN into target visits
  examples[0].normalized_visits[0] = std::numeric_limits<double>::quiet_NaN();

  SearchAuxPhaseResult res = TrainSearchAuxiliaryPhase(
      model, aux_opt, examples, obs_size, action_dim, device,
      /*max_steps=*/2, /*batch_size=*/4, /*max_cumulative_kl=*/0.05,
      /*search_loss_coef=*/0.10, /*reference_rollout_sample=*/{});

  OFFLINE_CHECK(res.step_rejected_nonfinite == true);
  OFFLINE_CHECK(res.phase_rejected == true);
  OFFLINE_CHECK(res.steps_accepted == 0);

  size_t p_idx = 0;
  for (const auto& p : model->parameters()) {
    OFFLINE_CHECK(torch::equal(p, initial_params[p_idx++]));
  }
  std::cout << "  Injected NaN target correctly triggered step_rejected_nonfinite and bitwise rollback." << std::endl;
  std::cout << "[PASS] Nonfinite rejection and parameter rollback verified." << std::endl;
}

void Test9_ResumeParityVerification(
    std::shared_ptr<SharedDunePolicyValueNetImpl> model,
    int64_t obs_size, int64_t action_dim, torch::Device device) {
  std::cout << "\n=== Test 9: Resume Parity Verification (Per-Phase Reset Invariant) ===" << std::endl;

  // Clone two identical models A and B
  auto model_a = std::make_shared<SharedDunePolicyValueNetImpl>(
      obs_size, 2048, action_dim, 8, false, false, 0, true);
  auto model_b = std::make_shared<SharedDunePolicyValueNetImpl>(
      obs_size, 2048, action_dim, 8, false, false, 0, true);
  model_a->to(device);
  model_b->to(device);

  CopyModelWeights(model, model_a);
  CopyModelWeights(model, model_b);

  auto batch1 = CreateMockExamples(32, obs_size, /*with_candidates=*/true);
  auto batch2 = CreateMockExamples(32, obs_size, /*with_candidates=*/true);
  // Perturb batch targets slightly away from self to ensure nonzero steps
  for (auto& ex : batch1) {
    ex.normalized_visits = {0.80, 0.05, 0.05, 0.05, 0.05};
  }
  for (auto& ex : batch2) {
    ex.normalized_visits = {0.05, 0.80, 0.05, 0.05, 0.05};
  }
  auto ref_rollout = CreateMockRefRollout(16, obs_size, /*with_candidates=*/true);

  // Setup optimizer A (uninterrupted runner)
  std::vector<torch::Tensor> aux_params_a;
  for (auto& p : model_a->parameters()) {
    if (p.requires_grad()) aux_params_a.push_back(p);
  }
  std::vector<torch::optim::OptimizerParamGroup> groups_a;
  groups_a.emplace_back(aux_params_a);
  torch::optim::AdamW opt_a(groups_a, torch::optim::AdamWOptions(1e-4).eps(1e-5).weight_decay(0.0));

  // Run Phase 1 Uninterrupted
  opt_a.state().clear();
  TrainSearchAuxiliaryPhase(model_a, opt_a, batch1, obs_size, action_dim, device,
                            1, 32, 1.0, 0.10, ref_rollout);

  // Run Phase 2 Uninterrupted (clear moments at phase boundary as per dune_ppo_train.cc)
  opt_a.state().clear();
  TrainSearchAuxiliaryPhase(model_a, opt_a, batch2, obs_size, action_dim, device,
                            1, 32, 1.0, 0.10, ref_rollout);

  // Run Phase 1 Resumed Model B
  std::vector<torch::Tensor> aux_params_b;
  for (auto& p : model_b->parameters()) {
    if (p.requires_grad()) aux_params_b.push_back(p);
  }
  std::vector<torch::optim::OptimizerParamGroup> groups_b1;
  groups_b1.emplace_back(aux_params_b);
  torch::optim::AdamW opt_b1(groups_b1, torch::optim::AdamWOptions(1e-4).eps(1e-5).weight_decay(0.0));
  opt_b1.state().clear();
  TrainSearchAuxiliaryPhase(model_b, opt_b1, batch1, obs_size, action_dim, device,
                            1, 32, 1.0, 0.10, ref_rollout);

  // Simulate Checkpoint boundary: save model_b only (optimizer is destroyed).
  // On resume, a fresh optimizer opt_b2 is constructed with empty state.
  std::vector<torch::optim::OptimizerParamGroup> groups_b2;
  groups_b2.emplace_back(aux_params_b);
  torch::optim::AdamW opt_b2(groups_b2, torch::optim::AdamWOptions(1e-4).eps(1e-5).weight_decay(0.0));
  // Run Phase 2 Resumed
  TrainSearchAuxiliaryPhase(model_b, opt_b2, batch2, obs_size, action_dim, device,
                            1, 32, 1.0, 0.10, ref_rollout);

  // Assert 100% BITWISE EQUALITY between uninterrupted and resumed models
  auto params_a = model_a->parameters();
  auto params_b = model_b->parameters();
  OFFLINE_CHECK(params_a.size() == params_b.size());
  OFFLINE_CHECK(params_a.size() > 0);
  for (size_t i = 0; i < params_a.size(); ++i) {
    OFFLINE_CHECK(torch::equal(params_a[i], params_b[i]));
  }
  std::cout << absl::StrFormat("  Uninterrupted vs resumed runs: %zu parameter tensors are 100%% bitwise identical.\n", params_a.size());
  std::cout << "[PASS] Resume parity verified under per-phase reset rule." << std::endl;
}

}  // namespace
}  // namespace open_spiel

int main(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);

  std::cout << "===========================================================" << std::endl;
  std::cout << "   Dune DRL Search Auxiliary Supervision Offline Test Suite" << std::endl;
  std::cout << "===========================================================" << std::endl;

  // Invariant tests 1 and 2
  open_spiel::Test1_Correction1_ReferencePolicyAndTiePreservation();
  open_spiel::Test2_Correction4_NearTieStabilityAndTemperatureFloor();

  at::globalContext().setDeterministicAlgorithms(true, /*silent=*/true);
  if (torch::cuda::is_available()) {
    at::globalContext().setAllowTF32CuBLAS(absl::GetFlag(FLAGS_allow_tf32));
    at::globalContext().setAllowTF32CuDNN(absl::GetFlag(FLAGS_allow_tf32));
    at::autocast::set_autocast_dtype(at::kCUDA, at::ScalarType::BFloat16);
  }
  torch::manual_seed(20260923);

  // Model & Optimizer setup
  const int64_t obs_size = 9182;
  const int64_t action_dim = 2391;
  const int hidden_dim = 2048;
  const int num_blocks = 8;
  torch::Device device = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
  std::cout << "\nUsing device: " << (device.is_cuda() ? "CUDA" : "CPU") << std::endl;

  auto model = std::make_shared<open_spiel::SharedDunePolicyValueNetImpl>(
      obs_size, hidden_dim, action_dim, num_blocks,
      /*use_nonlinear=*/false, /*with_aux_heads=*/false, /*head_init_seed=*/0,
      /*with_semantic_scorer=*/true);
  model->to(device);

  const std::string ckpt_path = absl::GetFlag(FLAGS_checkpoint_path);
  const std::string opt_path = absl::GetFlag(FLAGS_optimizer_path);

  // Explicit failure on missing artifacts
  if (!std::filesystem::exists(ckpt_path)) {
    std::cerr << "FATAL: Checkpoint artifact missing at " << ckpt_path << std::endl;
    std::exit(1);
  }
  std::cout << "Loading real saved checkpoint: " << ckpt_path << std::endl;
  torch::load(model, ckpt_path, device);
  std::cout << "Checkpoint loaded successfully." << std::endl;

  std::vector<torch::Tensor> policy_params;
  std::vector<torch::Tensor> other_params;
  auto policy_params_set = model->policy_head->parameters();
  for (auto& param : model->parameters()) {
    bool is_policy = false;
    for (auto& policy_param : policy_params_set) {
      if (param.is_same(policy_param)) {
        is_policy = true;
        break;
      }
    }
    if (is_policy) policy_params.push_back(param);
    else other_params.push_back(param);
  }
  std::vector<torch::optim::OptimizerParamGroup> groups;
  groups.emplace_back(policy_params);
  groups.emplace_back(other_params);
  torch::optim::AdamW optimizer(groups, torch::optim::AdamWOptions(2.5e-4).eps(1e-5));

  if (!std::filesystem::exists(opt_path)) {
    std::cerr << "FATAL: Optimizer artifact missing at " << opt_path << std::endl;
    std::exit(1);
  }
  std::cout << "Loading real saved optimizer: " << opt_path << std::endl;
  if (!open_spiel::LoadOptimizerCheckpointMigrating(model, optimizer, opt_path, device)) {
    try {
      torch::load(optimizer, opt_path, device);
      std::cout << "Optimizer state loaded successfully." << std::endl;
    } catch (const std::exception& e) {
      std::cerr << "FATAL: Optimizer load exception: " << e.what() << std::endl;
      std::exit(1);
    }
  } else {
    std::cout << "Optimizer checkpoint loaded and migrated successfully." << std::endl;
  }

  // Run comprehensive offline test suite
  open_spiel::Test3_Correction2_StrictKLEnforcementAndRollback(
      model, optimizer, obs_size, action_dim, device);

  open_spiel::Test4_Correction3_ValueHeadBitwiseImmutabilityWithPopulatedOptimizer(
      model, optimizer, obs_size, action_dim, device);

  open_spiel::Test5_NullSupervisionDetection(
      model, optimizer, obs_size, action_dim, device);

  open_spiel::Test6_PartialRollbackVsControl(
      model, optimizer, obs_size, action_dim, device);

  open_spiel::Test7_DeployedSemanticPolicyParityOnRealDuneStates(
      model, obs_size, action_dim, device);

  open_spiel::Test8_NanInjectionRejectionAndRollback(
      model, obs_size, action_dim, device);

  open_spiel::Test9_ResumeParityVerification(
      model, obs_size, action_dim, device);

  std::cout << "\n===========================================================" << std::endl;
  std::cout << "   ALL 9 TEST SUITES VERIFIED SUCCESSFULLY!" << std::endl;
  std::cout << "===========================================================" << std::endl;
  return 0;
}
