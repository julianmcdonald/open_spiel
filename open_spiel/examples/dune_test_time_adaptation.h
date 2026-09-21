#ifndef OPEN_SPIEL_EXAMPLES_DUNE_TEST_TIME_ADAPTATION_H_
#define OPEN_SPIEL_EXAMPLES_DUNE_TEST_TIME_ADAPTATION_H_

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <memory>
#include <cmath>
#include <chrono>
#include <random>
#include <algorithm>
#include <atomic>
#include <shared_mutex>
#include <future>
#include <sstream>
#include <iomanip>

#include <torch/torch.h>

#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"
#include "open_spiel/utils/json.h"
#include "open_spiel/games/dune_imperium/dune_imperium.h"
#include "open_spiel/games/dune_imperium/dune_imperium_common.h"

#include "dune_network.h"
#include "dune_evaluator.h"
#include "dune_batched_evaluator.h"
#include "dune_semantic_action_scorer.h"
#include "dune_ppo_training_utils.h"
#include "dune_eval_action_selection.h"
#include "dune_seed_utils.h"

namespace open_spiel {
namespace dune_tt_adapt {

// ---------------------------------------------------------------------------
// Error & Fallback Codes
// ---------------------------------------------------------------------------
enum class FallbackReason {
  kNone = 0,
  kInsufficientData,
  kNonFiniteLoss,
  kNonFiniteGrad,
  kTimeout,
  kExcessiveKL,
  kInvalidState,
  kInvalidAction,
  kBudgetExhausted
};

inline std::ostream& operator<<(std::ostream& os, FallbackReason r) {
  return os << static_cast<int>(r);
}

inline std::string FallbackReasonToString(FallbackReason r) {
  switch (r) {
    case FallbackReason::kNone: return "NONE";
    case FallbackReason::kInsufficientData: return "INSUFFICIENT_DATA";
    case FallbackReason::kNonFiniteLoss: return "NON_FINITE_LOSS";
    case FallbackReason::kNonFiniteGrad: return "NON_FINITE_GRAD";
    case FallbackReason::kTimeout: return "TIMEOUT";
    case FallbackReason::kExcessiveKL: return "EXCESSIVE_KL";
    case FallbackReason::kInvalidState: return "INVALID_STATE";
    case FallbackReason::kInvalidAction: return "INVALID_ACTION";
    case FallbackReason::kBudgetExhausted: return "BUDGET_EXHAUSTED";
    default: return "UNKNOWN";
  }
}

// ---------------------------------------------------------------------------
// Configuration & Telemetry Types
// ---------------------------------------------------------------------------
struct AdaptationConfig {
  int k_train_worlds = 8;
  int m_eval_worlds = 16;
  int comparator_worlds = 8;
  int gradient_steps = 1;
  float learning_rate = 1e-4f;
  float eta_kl = 0.20f;
  double timeout_seconds = 10.0;
  double max_kl = 0.50;
  int min_rollouts = 4;
  bool rollout_amp = true;
  bool allow_tf32 = false;
  bool zero_update_mode = false;
};

struct OperationalBudget {
  uint64_t max_physical_batches = 32000;
  uint64_t max_logical_queries = 350000;
  std::atomic<uint64_t> physical_batches{0};
  std::atomic<uint64_t> logical_queries{0};

  bool CanDispatch(uint64_t est_batches = 1, uint64_t est_queries = 1) const {
    if (physical_batches.load(std::memory_order_relaxed) + est_batches > max_physical_batches) {
      return false;
    }
    if (logical_queries.load(std::memory_order_relaxed) + est_queries > max_logical_queries) {
      return false;
    }
    return true;
  }

  bool TryReserveLogicalQueries(uint64_t num = 1) {
    uint64_t cur = logical_queries.load(std::memory_order_relaxed);
    while (true) {
      if (cur + num > max_logical_queries) return false;
      if (logical_queries.compare_exchange_weak(cur, cur + num,
                                                std::memory_order_relaxed,
                                                std::memory_order_relaxed)) {
        return true;
      }
    }
  }

  bool TryReservePhysicalBatches(uint64_t num = 1) {
    uint64_t cur = physical_batches.load(std::memory_order_relaxed);
    while (true) {
      if (cur + num > max_physical_batches) return false;
      if (physical_batches.compare_exchange_weak(cur, cur + num,
                                                 std::memory_order_relaxed,
                                                 std::memory_order_relaxed)) {
        return true;
      }
    }
  }

  void ReleaseLogicalQueries(uint64_t num = 1) {
    logical_queries.fetch_sub(num, std::memory_order_relaxed);
  }

  void ReleasePhysicalBatches(uint64_t num = 1) {
    physical_batches.fetch_sub(num, std::memory_order_relaxed);
  }

  void RecordQuery(uint64_t queries = 1) {
    logical_queries.fetch_add(queries, std::memory_order_relaxed);
  }

  void RecordBatch(uint64_t batches = 1) {
    physical_batches.fetch_add(batches, std::memory_order_relaxed);
  }
};

struct RolloutSample {
  Action root_action = kInvalidAction;
  float scalar_return = 0.0f;
};

struct AdaptationResult {
  Action action = kInvalidAction;
  bool fallback_triggered = false;
  FallbackReason fallback_reason = FallbackReason::kNone;
  double kl_divergence = 0.0;
  double elapsed_seconds = 0.0;
  float loss_value = 0.0f;
  float policy_grad_norm = 0.0f;
  std::vector<double> legal_probabilities;
  std::vector<Action> legal_actions;
};

class AdaptationClock;

struct AdaptationHooks {
  std::function<std::unique_ptr<State>(
      const State& base_state,
      Player player,
      uint64_t seed)> resample_override = nullptr;

  std::function<ActionsAndProbs(const State& state, Player player)> prior_override = nullptr;

  std::function<AdaptationResult(
      const State& root_state,
      Player player,
      const std::vector<RolloutSample>& samples,
      Action precomputed_raw_action,
      const ActionsAndProbs* precomputed_raw_prior,
      OperationalBudget* budget,
      const AdaptationClock* clock,
      const std::chrono::steady_clock::time_point& start_time)> adapt_override = nullptr;

  int max_rollout_steps = -1;
};

// ---------------------------------------------------------------------------
// Clock Abstractions
// ---------------------------------------------------------------------------
class AdaptationClock {
 public:
  virtual ~AdaptationClock() = default;
  virtual double ElapsedSeconds(
      const std::chrono::steady_clock::time_point& start, int stage) const {
    return std::chrono::duration<double>(
        std::chrono::steady_clock::now() - start).count();
  }
};

class SimulatedTimeoutClock : public AdaptationClock {
 public:
  explicit SimulatedTimeoutClock(int timeout_at_stage, double expired_time = 10.01)
      : timeout_stage_(timeout_at_stage), expired_time_(expired_time) {}

  double ElapsedSeconds(
      const std::chrono::steady_clock::time_point& start, int stage) const override {
    if (stage >= timeout_stage_) {
      return expired_time_;
    }
    return 0.05 * stage;
  }

 private:
  int timeout_stage_;
  double expired_time_;
};

// ---------------------------------------------------------------------------
// Model Deep Copy Utility
// ---------------------------------------------------------------------------
inline void DeepCopyModelParameters(
    const std::shared_ptr<SharedDunePolicyValueNetImpl>& source,
    std::shared_ptr<SharedDunePolicyValueNetImpl>& destination) {
  torch::NoGradGuard no_grad;
  auto src_params = source->named_parameters();
  auto dst_params = destination->named_parameters();
  for (const auto& item : src_params) {
    auto* dst_p = dst_params.find(item.key());
    if (!dst_p) {
      SpielFatalError("DeepCopyModelParameters: parameter missing in destination: " + item.key());
    }
    dst_p->copy_(item.value().detach().clone());
  }
  auto src_buffers = source->named_buffers();
  auto dst_buffers = destination->named_buffers();
  for (const auto& item : src_buffers) {
    auto* dst_b = dst_buffers.find(item.key());
    if (!dst_b) {
      SpielFatalError("DeepCopyModelParameters: buffer missing in destination: " + item.key());
    }
    dst_b->copy_(item.value().detach().clone());
  }
}

// ---------------------------------------------------------------------------
// Action Selection Helpers
// ---------------------------------------------------------------------------
inline Action PickGreedyFromPrior(const ActionsAndProbs& prior, const std::vector<Action>& legal_actions) {
  if (prior.empty()) {
    return legal_actions.empty() ? kInvalidAction : legal_actions.front();
  }
  Action best_action = prior.front().first;
  double best_prob = -1.0;
  for (const auto& ap : prior) {
    if (ap.second > best_prob) {
      best_prob = ap.second;
      best_action = ap.first;
    }
  }
  return best_action;
}

template <typename Rng>
inline Action SampleActionFromPrior(const ActionsAndProbs& prior, Rng& rng) {
  if (prior.empty()) return kInvalidAction;
  double u = std::generate_canonical<double, 53>(rng);
  double cumulative = 0.0;
  for (const auto& ap : prior) {
    cumulative += ap.second;
    if (u <= cumulative) {
      return ap.first;
    }
  }
  return prior.back().first;
}

// ---------------------------------------------------------------------------
// Test-Time Adaptation Controller
// ---------------------------------------------------------------------------
class TestTimeAdaptationController {
 public:
  TestTimeAdaptationController(
      std::shared_ptr<SharedDunePolicyValueNetImpl> blueprint,
      std::shared_ptr<BatchedNNEvaluator> production_evaluator,
      torch::Device device,
      AdaptationConfig config = AdaptationConfig())
      : blueprint_(blueprint),
        evaluator_(production_evaluator),
        device_(device),
        config_(config) {}

  AdaptationResult Adapt(
      const State& root_state,
      Player player,
      const std::vector<RolloutSample>& rollout_samples,
      bool inject_nan_loss = false,
      bool inject_post_step_nan = false,
      bool inject_controlled_excessive_kl = false,
      int force_timeout_stage = 0,
      int override_valid_worlds = -1,
      const AdaptationClock* clock = nullptr,
      std::chrono::steady_clock::time_point root_start_time = std::chrono::steady_clock::now(),
      Action precomputed_raw_action = kInvalidAction,
      const ActionsAndProbs* precomputed_raw_prior = nullptr,
      OperationalBudget* budget = nullptr) {

    AdaptationResult result;
    AdaptationClock real_clock;
    SimulatedTimeoutClock legacy_sim_clock(force_timeout_stage, 10.01);
    const AdaptationClock* active_clock = clock;
    if (!active_clock) {
      if (force_timeout_stage > 0) {
        active_clock = &legacy_sim_clock;
      } else {
        active_clock = &real_clock;
      }
    }

    // 1. True raw incumbent greedy action via production evaluator
    Action raw_action = precomputed_raw_action;
    ActionsAndProbs raw_prior;
    if (precomputed_raw_prior != nullptr) {
      raw_prior = *precomputed_raw_prior;
      if (raw_action == kInvalidAction) {
        raw_action = PickGreedyFromPrior(raw_prior, root_state.LegalActions());
      }
    } else {
      if (budget && !budget->TryReserveLogicalQueries(1)) {
        result.fallback_triggered = true;
        result.fallback_reason = FallbackReason::kBudgetExhausted;
        result.action = root_state.LegalActions().empty() ? kInvalidAction : root_state.LegalActions().front();
        return result;
      }
      raw_prior = evaluator_->Prior(root_state);
      raw_action = PickGreedyFromPrior(raw_prior, root_state.LegalActions());
    }
    result.action = raw_action;

    const auto* dune = dynamic_cast<const dune_imperium::DuneImperiumState*>(&root_state);
    if (!dune) {
      result.fallback_triggered = true;
      result.fallback_reason = FallbackReason::kInvalidState;
      return result;
    }

    std::vector<Action> legal_actions = root_state.LegalActions();
    if (legal_actions.empty()) {
      return result;
    }
    result.legal_actions = legal_actions;

    std::vector<float> obs = dune->InformationStateTensorWithAppendix(
        player, dune_imperium::MarketAppendixMode::kFullPublicInformationV3);
    torch::Tensor obs_tensor = torch::from_blob(
        obs.data(), {1, static_cast<int64_t>(obs.size())}, torch::kFloat32).to(device_);
    torch::Tensor mask_tensor = torch::zeros({1, 2391}, torch::TensorOptions().dtype(torch::kBool).device(device_));
    for (Action a : legal_actions) {
      mask_tensor[0][a] = true;
    }

    dune_semantic::CandidateActionData cand_data;
    dune_semantic::ExtractCandidateDescriptors(
        *dune, legal_actions, &cand_data, dune_semantic::kDescriptorSchemaVersionV3);
    std::vector<const dune_semantic::CandidateActionData*> batch_cands = {&cand_data};

    // Blueprint prior reference for KL calculation under internal precision settings
    torch::Tensor bp_probs;
    {
      if (budget) {
        if (!budget->TryReserveLogicalQueries(1)) {
          result.fallback_triggered = true;
          result.fallback_reason = FallbackReason::kBudgetExhausted;
          result.action = raw_action;
          return result;
        }
        if (!budget->TryReservePhysicalBatches(1)) {
          budget->ReleaseLogicalQueries(1);
          result.fallback_triggered = true;
          result.fallback_reason = FallbackReason::kBudgetExhausted;
          result.action = raw_action;
          return result;
        }
      }
      AutocastGuard guard(device_.type(), device_.is_cuda() && config_.rollout_amp);
      torch::NoGradGuard no_grad;
      auto bp_out = blueprint_->forward(obs_tensor);
      dune_semantic::ApplySemanticScorerBatch(
          blueprint_->semantic_scorer_, bp_out.trunk, batch_cands, bp_out.logits, device_);
      torch::Tensor bp_capped = CenterAndCapLogitsTensor(bp_out.logits, mask_tensor, 10.0f);
      torch::Tensor bp_masked = bp_capped.masked_fill(mask_tensor.logical_not(), -1e9f);
      bp_probs = torch::softmax(bp_masked, -1);
    }

    // Zero-update mode: evaluate candidate directly under internal precision settings
    if (config_.zero_update_mode) {
      if (budget) {
        if (!budget->TryReserveLogicalQueries(1)) {
          result.fallback_triggered = true;
          result.fallback_reason = FallbackReason::kBudgetExhausted;
          result.action = raw_action;
          return result;
        }
        if (!budget->TryReservePhysicalBatches(1)) {
          budget->ReleaseLogicalQueries(1);
          result.fallback_triggered = true;
          result.fallback_reason = FallbackReason::kBudgetExhausted;
          result.action = raw_action;
          return result;
        }
      }
      AutocastGuard guard(device_.type(), device_.is_cuda() && config_.rollout_amp);
      auto cand_model = std::make_shared<SharedDunePolicyValueNetImpl>(
          dune_imperium::kFullPublicInformationStateSize, 2048, 2391, 8,
          /*nonlinear_value_head=*/false, /*with_aux_heads=*/false,
          /*head_init_seed=*/0, /*with_semantic_scorer=*/true);
      cand_model->to(device_);
      DeepCopyModelParameters(blueprint_, cand_model);
      cand_model->eval();

      auto zero_out = cand_model->forward(obs_tensor);
      dune_semantic::ApplySemanticScorerBatch(
          cand_model->semantic_scorer_, zero_out.trunk, batch_cands, zero_out.logits, device_);
      torch::Tensor zero_capped = CenterAndCapLogitsTensor(zero_out.logits, mask_tensor, 10.0f);
      torch::Tensor zero_masked = zero_capped.masked_fill(mask_tensor.logical_not(), -1e9f);
      torch::Tensor zero_probs = torch::softmax(zero_masked, -1);

      result.legal_probabilities.clear();
      Action zero_action = kInvalidAction;
      double max_p = -1.0;
      for (Action a : legal_actions) {
        double p = zero_probs[0][a].item<double>();
        result.legal_probabilities.push_back(p);
        if (p > max_p) {
          max_p = p;
          zero_action = a;
        }
      }
      result.action = zero_action;
      result.fallback_triggered = false;
      result.fallback_reason = FallbackReason::kNone;
      result.kl_divergence = 0.0;
      return result;
    }

    // 2. Data requirement check (strictly enforced)
    int valid_worlds = (override_valid_worlds >= 0)
                           ? override_valid_worlds
                           : static_cast<int>(rollout_samples.size());
    if (valid_worlds < config_.min_rollouts) {
      result.fallback_triggered = true;
      result.fallback_reason = FallbackReason::kInsufficientData;
      result.action = raw_action;
      return result;
    }

    // 3. Deep-copy candidate model for temporary adaptation
    auto cand_model = std::make_shared<SharedDunePolicyValueNetImpl>(
        dune_imperium::kFullPublicInformationStateSize, 2048, 2391, 8,
        /*nonlinear_value_head=*/false, /*with_aux_heads=*/false,
        /*head_init_seed=*/0, /*with_semantic_scorer=*/true);
    cand_model->to(device_);
    DeepCopyModelParameters(blueprint_, cand_model);
    cand_model->train();

    // Freeze critic heads
    for (auto& p : cand_model->value_head->parameters()) p.set_requires_grad(false);
    if (cand_model->value_head2) {
      for (auto& p : cand_model->value_head2->parameters()) p.set_requires_grad(false);
    }

    // Setup optimizer
    torch::optim::AdamW optimizer(
        cand_model->parameters(),
        torch::optim::AdamWOptions(config_.learning_rate).betas({0.9, 0.999}).eps(1e-8).weight_decay(0.0));

    // Deadline Check 1 (Pre-step)
    double elapsed = active_clock->ElapsedSeconds(root_start_time, 1);
    result.elapsed_seconds = elapsed;
    if (elapsed > config_.timeout_seconds) {
      result.fallback_triggered = true;
      result.fallback_reason = FallbackReason::kTimeout;
      result.action = raw_action;
      return result;
    }

    // 4. Candidate forward pass under internal precision
    torch::Tensor cand_probs;
    torch::Tensor cand_log_probs;
    {
      if (budget) {
        if (!budget->TryReserveLogicalQueries(1)) {
          result.fallback_triggered = true;
          result.fallback_reason = FallbackReason::kBudgetExhausted;
          result.action = raw_action;
          return result;
        }
        if (!budget->TryReservePhysicalBatches(1)) {
          budget->ReleaseLogicalQueries(1);
          result.fallback_triggered = true;
          result.fallback_reason = FallbackReason::kBudgetExhausted;
          result.action = raw_action;
          return result;
        }
      }
      AutocastGuard guard(device_.type(), device_.is_cuda() && config_.rollout_amp);
      auto cand_out = cand_model->forward(obs_tensor);
      dune_semantic::ApplySemanticScorerBatch(
          cand_model->semantic_scorer_, cand_out.trunk, batch_cands, cand_out.logits, device_);
      torch::Tensor cand_capped = CenterAndCapLogitsTensor(cand_out.logits, mask_tensor, 10.0f);
      torch::Tensor cand_masked = cand_capped.masked_fill(mask_tensor.logical_not(), -1e9f);
      cand_log_probs = torch::log_softmax(cand_masked, -1);
      cand_probs = torch::softmax(cand_masked, -1);
    }

    // Baseline subtraction across batch
    float mean_return = 0.0f;
    for (const auto& s : rollout_samples) {
      mean_return += s.scalar_return;
    }
    mean_return /= static_cast<float>(rollout_samples.size());

    int num_steps = std::max(1, config_.gradient_steps);
    float last_loss = 0.0f;
    float last_p_grad_norm = 0.0f;

    for (int step_idx = 0; step_idx < num_steps; ++step_idx) {
      // 4. Candidate forward pass under internal precision
      torch::Tensor cand_probs;
      torch::Tensor cand_log_probs;
      {
        if (budget) {
          if (!budget->TryReserveLogicalQueries(1)) {
            result.fallback_triggered = true;
            result.fallback_reason = FallbackReason::kBudgetExhausted;
            result.action = raw_action;
            return result;
          }
          if (!budget->TryReservePhysicalBatches(1)) {
            budget->ReleaseLogicalQueries(1);
            result.fallback_triggered = true;
            result.fallback_reason = FallbackReason::kBudgetExhausted;
            result.action = raw_action;
            return result;
          }
        }
        AutocastGuard guard(device_.type(), device_.is_cuda() && config_.rollout_amp);
        auto cand_out = cand_model->forward(obs_tensor);
        dune_semantic::ApplySemanticScorerBatch(
            cand_model->semantic_scorer_, cand_out.trunk, batch_cands, cand_out.logits, device_);
        torch::Tensor cand_capped = CenterAndCapLogitsTensor(cand_out.logits, mask_tensor, 10.0f);
        torch::Tensor cand_masked = cand_capped.masked_fill(mask_tensor.logical_not(), -1e9f);
        cand_log_probs = torch::log_softmax(cand_masked, -1);
        cand_probs = torch::softmax(cand_masked, -1);
      }

      // Policy gradient loss over sampled actions
      torch::Tensor pg_loss = torch::zeros({}, torch::TensorOptions().device(device_));
      for (const auto& s : rollout_samples) {
        float advantage = s.scalar_return - mean_return;
        pg_loss = pg_loss - (advantage * cand_log_probs[0][s.root_action]);
      }
      pg_loss = pg_loss / static_cast<float>(rollout_samples.size());

      // Root-only KL regularization: D_KL(pi_cand || pi_bp)
      torch::Tensor kl_term = torch::zeros({}, torch::TensorOptions().device(device_));
      for (Action a : legal_actions) {
        torch::Tensor p = cand_probs[0][a];
        torch::Tensor log_p = cand_log_probs[0][a];
        torch::Tensor log_q = torch::log(bp_probs[0][a] + 1e-12f);
        kl_term = kl_term + (p * (log_p - log_q));
      }
      torch::Tensor total_loss = pg_loss + (config_.eta_kl * kl_term);

      // Injected fault: NaN loss
      if (inject_nan_loss) {
        total_loss = total_loss * std::numeric_limits<float>::quiet_NaN();
      }

      // Check finite loss
      if (!torch::isfinite(total_loss).item<bool>()) {
        result.fallback_triggered = true;
        result.fallback_reason = FallbackReason::kNonFiniteLoss;
        result.action = raw_action;
        return result;
      }
      last_loss = total_loss.item<float>();

      // Backward pass
      optimizer.zero_grad();
      total_loss.backward();

      // Check finite gradients & compute policy gradient norm
      float p_grad_norm = 0.0f;
      bool grad_ok = true;
      for (const auto& p : cand_model->parameters()) {
        if (p.grad().defined()) {
          if (!torch::isfinite(p.grad()).all().item<bool>()) {
            grad_ok = false;
            break;
          }
          p_grad_norm += p.grad().norm().item<float>();
        }
      }
      if (!grad_ok) {
        result.fallback_triggered = true;
        result.fallback_reason = FallbackReason::kNonFiniteGrad;
        result.action = raw_action;
        return result;
      }
      last_p_grad_norm = p_grad_norm;

      // Optimizer step
      optimizer.step();

      // Hook: Post-step NaN injection
      if (inject_post_step_nan && step_idx == 0) {
        torch::NoGradGuard g;
        cand_model->input_layer->weight.fill_(std::numeric_limits<float>::quiet_NaN());
      }

      // Hook: Controlled finite excessive KL
      if (inject_controlled_excessive_kl && step_idx == 0) {
        torch::NoGradGuard g;
        Action target_a = (legal_actions.size() > 1) ? legal_actions.back() : legal_actions[0];
        cand_model->policy_head->weight[target_a].fill_(5.0f);
      }

      // Check parameter finiteness after step
      bool params_finite = true;
      for (const auto& p : cand_model->parameters()) {
        if (!torch::isfinite(p).all().item<bool>()) {
          params_finite = false;
          break;
        }
      }
      if (!params_finite) {
        result.fallback_triggered = true;
        result.fallback_reason = FallbackReason::kNonFiniteGrad;
        result.action = raw_action;
        return result;
      }

      // Deadline Check 2 (Post-step)
      elapsed = active_clock->ElapsedSeconds(root_start_time, 2);
      result.elapsed_seconds = elapsed;
      if (elapsed > config_.timeout_seconds) {
        result.fallback_triggered = true;
        result.fallback_reason = FallbackReason::kTimeout;
        result.action = raw_action;
        return result;
      }
    }
    result.loss_value = last_loss;
    result.policy_grad_norm = last_p_grad_norm;

    // Check KL divergence and post-step distributions
    {
      if (budget) {
        if (!budget->TryReserveLogicalQueries(1)) {
          result.fallback_triggered = true;
          result.fallback_reason = FallbackReason::kBudgetExhausted;
          result.action = raw_action;
          return result;
        }
        if (!budget->TryReservePhysicalBatches(1)) {
          budget->ReleaseLogicalQueries(1);
          result.fallback_triggered = true;
          result.fallback_reason = FallbackReason::kBudgetExhausted;
          result.action = raw_action;
          return result;
        }
      }
      AutocastGuard guard(device_.type(), device_.is_cuda() && config_.rollout_amp);
      torch::NoGradGuard no_grad;
      auto post_out = cand_model->forward(obs_tensor);
      dune_semantic::ApplySemanticScorerBatch(
          cand_model->semantic_scorer_, post_out.trunk, batch_cands, post_out.logits, device_);
      torch::Tensor post_capped = CenterAndCapLogitsTensor(post_out.logits, mask_tensor, 10.0f);
      torch::Tensor post_masked = post_capped.masked_fill(mask_tensor.logical_not(), -1e9f);
      torch::Tensor post_probs = torch::softmax(post_masked, -1);
      torch::Tensor post_log_probs = torch::log_softmax(post_masked, -1);

      if (!torch::isfinite(post_probs).all().item<bool>() ||
          !torch::isfinite(post_log_probs).all().item<bool>()) {
        result.fallback_triggered = true;
        result.fallback_reason = FallbackReason::kNonFiniteGrad;
        result.action = raw_action;
        return result;
      }

      double measured_kl = 0.0;
      for (Action a : legal_actions) {
        double p = post_probs[0][a].item<double>();
        double log_p = post_log_probs[0][a].item<double>();
        double log_q = std::log(bp_probs[0][a].item<double>() + 1e-12);
        measured_kl += p * (log_p - log_q);
      }
      result.kl_divergence = measured_kl;

      if (std::isnan(measured_kl) || std::isinf(measured_kl) || measured_kl > config_.max_kl) {
        result.fallback_triggered = true;
        result.fallback_reason = FallbackReason::kExcessiveKL;
        result.action = raw_action;
        return result;
      }

      // Deadline Check 3 (Post-inference)
      elapsed = active_clock->ElapsedSeconds(root_start_time, 3);
      result.elapsed_seconds = elapsed;
      if (elapsed > config_.timeout_seconds) {
        result.fallback_triggered = true;
        result.fallback_reason = FallbackReason::kTimeout;
        result.action = raw_action;
        return result;
      }

      // Greedily pick adapted action & save probabilities
      result.legal_probabilities.clear();
      Action adapted_action = kInvalidAction;
      double max_p = -1.0;
      for (Action a : legal_actions) {
        double p = post_probs[0][a].item<double>();
        result.legal_probabilities.push_back(p);
        if (p > max_p) {
          max_p = p;
          adapted_action = a;
        }
      }
      if (adapted_action == kInvalidAction) {
        result.fallback_triggered = true;
        result.fallback_reason = FallbackReason::kInvalidAction;
        result.action = raw_action;
        return result;
      }
      result.action = adapted_action;
    }

    return result;
  }

 private:
  std::shared_ptr<SharedDunePolicyValueNetImpl> blueprint_;
  std::shared_ptr<BatchedNNEvaluator> evaluator_;
  torch::Device device_;
  AdaptationConfig config_;
};

// ---------------------------------------------------------------------------
// Native Rollout Continuation Engine
// ---------------------------------------------------------------------------
// Executes a trajectory continuation from a state where an action may already
// have been applied, stepping chance nodes with true transition probabilities
// and players under frozen greedy policies to game end.
inline bool RunRolloutContinuation(
    State* state,
    Player candidate_player,
    BatchedNNEvaluator* cand_eval,
    BatchedNNEvaluator* opp_eval,
    uint64_t seed_base,
    int stream_id,
    OperationalBudget* budget,
    const AdaptationClock* clock,
    const std::chrono::steady_clock::time_point& root_start_time,
    double timeout_seconds,
    int clock_stage,
    double* out_return,
    const AdaptationHooks* hooks = nullptr) {

  // Check phase deadline upfront
  if (clock) {
    double elapsed = clock->ElapsedSeconds(root_start_time, clock_stage);
    if (elapsed > timeout_seconds) {
      return false;
    }
  }

  int depth = 0;
  int max_depth = (hooks && hooks->max_rollout_steps > 0) ? hooks->max_rollout_steps : 2000;
  while (!state->IsTerminal()) {
    if (depth >= max_depth) {
      return false;
    }
    // 1. Check phase deadline
    if (clock) {
      double elapsed = clock->ElapsedSeconds(root_start_time, clock_stage);
      if (elapsed > timeout_seconds) {
        return false;
      }
    }

    if (state->IsChanceNode()) {
      auto chance_outcomes = state->ChanceOutcomes();
      if (chance_outcomes.empty()) {
        return false;
      }
      uint64_t cseed = dune_seed::DeriveSeed(seed_base, dune_seed::kStreamChance, stream_id, depth);
      std::mt19937 crng(cseed);
      double u = std::generate_canonical<double, 53>(crng);
      Action caction = open_spiel::SampleAction(chance_outcomes, u).first;
      state->ApplyAction(caction);
      depth++;
      continue;
    }

    std::vector<Action> legals = state->LegalActions();
    if (legals.empty()) {
      return false;
    }
    if (legals.size() == 1) {
      state->ApplyAction(legals.front());
      depth++;
      continue;
    }

    Player cur = state->CurrentPlayer();
    ActionsAndProbs prior;
    if (hooks && hooks->prior_override) {
      if (budget && !budget->TryReserveLogicalQueries(1)) {
        return false;
      }
      prior = hooks->prior_override(*state, cur);
    } else {
      if (budget && !budget->TryReserveLogicalQueries(1)) {
        return false;
      }
      auto* active_eval = (cur == candidate_player) ? cand_eval : opp_eval;
      if (!active_eval) {
        return false;
      }
      prior = active_eval->Prior(*state);
    }

    if (prior.empty()) {
      return false;
    }

    Action chosen = PickGreedyFromPrior(prior, legals);
    if (chosen == kInvalidAction) {
      chosen = legals.front();
    }
    state->ApplyAction(chosen);
    depth++;
  }

  std::vector<double> returns = state->Returns();
  if (static_cast<size_t>(candidate_player) >= returns.size()) {
    return false;
  }
  *out_return = returns[candidate_player];
  return true;
}

// ---------------------------------------------------------------------------
// Pilot Decision Root Definition
// ---------------------------------------------------------------------------
struct DecisionRoot {
  int root_id = -1;
  int episode_id = -1;
  Player player = -1;
  int round = -1;
  std::string phase;
  std::string stratum;
  std::vector<Action> legal_actions;
  int num_legal = 0;
  std::vector<Action> history;
  uint64_t master_seed = 0;
  uint64_t train_seed_base = 0;
  uint64_t eval_seed_base = 0;
  std::string source_bank;
};

// ---------------------------------------------------------------------------
// Per-Root Execution Result & Per-World Receipt
// ---------------------------------------------------------------------------
struct RootExecutionResult {
  int root_id = -1;
  int episode_id = -1;
  Player player = -1;
  int round = -1;
  std::string stratum;
  Action raw_action = kInvalidAction;
  Action adapt_action = kInvalidAction;
  Action comp_action = kInvalidAction;
  bool fallback_triggered = false;
  FallbackReason fallback_reason = FallbackReason::kNone;
  double kl_divergence = 0.0;
  float loss_value = 0.0f;
  float policy_grad_norm = 0.0f;
  double elapsed_seconds = 0.0;

  // 16 held-out evaluation world returns
  std::vector<double> raw_returns;
  std::vector<double> adapt_returns;
  std::vector<double> comp_returns;

  // Root mean summaries
  double mean_raw = 0.0;
  double mean_adapt = 0.0;
  double mean_comp = 0.0;
  double paired_diff = 0.0;
  double comp_diff = 0.0;

  bool evaluation_completed = false;
};

// ---------------------------------------------------------------------------
// Full Pilot Pipeline Execution for One Root
// ---------------------------------------------------------------------------
inline bool ExecuteDecisionRoot(
    const std::shared_ptr<const Game>& game,
    const DecisionRoot& root,
    std::shared_ptr<SharedDunePolicyValueNetImpl> blueprint,
    std::shared_ptr<BatchedNNEvaluator> cand_evaluator,
    std::shared_ptr<BatchedNNEvaluator> opp_evaluator,
    TestTimeAdaptationController* controller,
    const AdaptationConfig& config,
    OperationalBudget* budget,
    const AdaptationClock* clock,
    RootExecutionResult* out_result,
    const AdaptationHooks* hooks = nullptr) {

  auto root_start_time = std::chrono::steady_clock::now();
  out_result->root_id = root.root_id;
  out_result->episode_id = root.episode_id;
  out_result->player = root.player;
  out_result->round = root.round;
  out_result->stratum = root.stratum;
  out_result->evaluation_completed = false;

  // 1. Reconstruct root state
  std::unique_ptr<State> root_state = game->NewInitialState();
  for (Action a : root.history) {
    root_state->ApplyAction(a);
  }
  const auto* dune = dynamic_cast<const dune_imperium::DuneImperiumState*>(root_state.get());
  if (!dune || root_state->IsTerminal() || root_state->CurrentPlayer() != root.player) {
    out_result->fallback_triggered = true;
    out_result->fallback_reason = FallbackReason::kInvalidState;
    return false;
  }

  // 2. Branch 1: Raw incumbent greedy action
  ActionsAndProbs raw_prior;
  if (hooks && hooks->prior_override) {
    if (budget && !budget->TryReserveLogicalQueries(1)) {
      out_result->fallback_triggered = true;
      out_result->fallback_reason = FallbackReason::kBudgetExhausted;
      out_result->raw_action = root.legal_actions.empty() ? kInvalidAction : root.legal_actions.front();
      out_result->adapt_action = out_result->raw_action;
      out_result->comp_action = out_result->raw_action;
      return false;
    }
    raw_prior = hooks->prior_override(*root_state, root.player);
  } else {
    if (budget && !budget->TryReserveLogicalQueries(1)) {
      out_result->fallback_triggered = true;
      out_result->fallback_reason = FallbackReason::kBudgetExhausted;
      out_result->raw_action = root.legal_actions.empty() ? kInvalidAction : root.legal_actions.front();
      out_result->adapt_action = out_result->raw_action;
      out_result->comp_action = out_result->raw_action;
      return false;
    }
    raw_prior = cand_evaluator->Prior(*root_state);
  }
  if (raw_prior.empty()) {
    out_result->fallback_triggered = true;
    out_result->fallback_reason = FallbackReason::kBudgetExhausted;
    out_result->raw_action = root.legal_actions.empty() ? kInvalidAction : root.legal_actions.front();
    out_result->adapt_action = out_result->raw_action;
    out_result->comp_action = out_result->raw_action;
    return false;
  }

  Action raw_action = PickGreedyFromPrior(raw_prior, root.legal_actions);
  out_result->raw_action = raw_action;

  // 3. Branch 2: Collect K=8 training rollouts on resampled worlds
  std::vector<RolloutSample> train_samples;
  train_samples.reserve(config.k_train_worlds);

  struct TrainRolloutTask {
    std::unique_ptr<State> state;
    Action sampled_root_a;
    uint64_t wseed;
    int k;
  };
  std::vector<TrainRolloutTask> tasks;
  tasks.reserve(config.k_train_worlds);

  for (int k = 0; k < config.k_train_worlds; ++k) {
    uint64_t wseed = root.train_seed_base + k;
    std::unique_ptr<State> resampled = nullptr;
    if (hooks && hooks->resample_override) {
      resampled = hooks->resample_override(*dune, root.player, wseed);
    } else {
      std::mt19937 wrng(wseed);
      auto rng_func = [&wrng]() {
        return std::generate_canonical<double, 53>(wrng);
      };
      resampled = dune->ResampleFromInfostate(root.player, rng_func);
    }
    if (!resampled) continue;

    std::mt19937 action_rng(wseed + 100);
    Action sampled_root_a = SampleActionFromPrior(raw_prior, action_rng);
    if (sampled_root_a == kInvalidAction) {
      sampled_root_a = raw_action;
    }
    resampled->ApplyAction(sampled_root_a);
    tasks.push_back({std::move(resampled), sampled_root_a, wseed, k});
  }

  // Concurrently execute training rollouts
  std::vector<std::future<std::pair<bool, double>>> train_futures;
  train_futures.reserve(tasks.size());

  for (size_t i = 0; i < tasks.size(); ++i) {
    auto* task_state = tasks[i].state.get();
    uint64_t wseed = tasks[i].wseed;
    int k = tasks[i].k;
    train_futures.push_back(std::async(
        std::launch::async,
        [task_state, &root, cand_evaluator, opp_evaluator, wseed, k, budget, clock, root_start_time, &config, hooks]() {
          double s_ret = 0.0;
          bool ok = RunRolloutContinuation(
              task_state, root.player, cand_evaluator.get(), opp_evaluator.get(),
              wseed, k, budget, clock, root_start_time, config.timeout_seconds,
              /*clock_stage=*/4, &s_ret, hooks);
          return std::make_pair(ok, s_ret);
        }));
  }

  bool training_budget_exhausted = false;
  bool training_timed_out = false;
  for (size_t i = 0; i < train_futures.size(); ++i) {
    auto res = train_futures[i].get();
    if (res.first) {
      train_samples.push_back({tasks[i].sampled_root_a, static_cast<float>(res.second)});
    } else {
      if (clock && clock->ElapsedSeconds(root_start_time, 4) > config.timeout_seconds) {
        training_timed_out = true;
      } else if (budget && !budget->CanDispatch(1, 1)) {
        training_budget_exhausted = true;
      }
    }
  }

  // Strict requirement: If budget exhausted during training, abort immediately even if >= 4 samples completed!
  if (training_budget_exhausted || (budget && !budget->CanDispatch(1, 1))) {
    out_result->fallback_triggered = true;
    out_result->fallback_reason = FallbackReason::kBudgetExhausted;
    out_result->adapt_action = raw_action;
    out_result->comp_action = raw_action;
    return false;
  }
  if (training_timed_out) {
    out_result->fallback_triggered = true;
    out_result->fallback_reason = FallbackReason::kTimeout;
    out_result->adapt_action = raw_action;
    out_result->comp_action = raw_action;
    return false;
  }

  if (clock) {
    std::cout << "    [Root " << root.root_id << " Stage 4 Train: "
              << std::fixed << std::setprecision(2) << clock->ElapsedSeconds(root_start_time, 4)
              << "s, samples=" << train_samples.size() << "]\n" << std::flush;
  }

  // Run adaptation update
  AdaptationResult adapt_res;
  if (hooks && hooks->adapt_override) {
    adapt_res = hooks->adapt_override(
        *root_state, root.player, train_samples,
        raw_action, &raw_prior, budget, clock, root_start_time);
  } else if (controller) {
    adapt_res = controller->Adapt(
        *root_state, root.player, train_samples,
        /*inject_nan_loss=*/false, /*inject_post_step_nan=*/false,
        /*inject_controlled_excessive_kl=*/false, /*force_timeout_stage=*/0,
        /*override_valid_worlds=*/-1, clock, root_start_time,
        raw_action, &raw_prior, budget);
  } else {
    adapt_res.action = raw_action;
    adapt_res.fallback_triggered = true;
    adapt_res.fallback_reason = FallbackReason::kInvalidAction;
  }

  if (adapt_res.action == kInvalidAction) {
    adapt_res.action = raw_action;
    adapt_res.fallback_triggered = true;
    if (adapt_res.fallback_reason == FallbackReason::kNone) {
      adapt_res.fallback_reason = FallbackReason::kInvalidAction;
    }
  }

  out_result->adapt_action = adapt_res.action;
  out_result->fallback_triggered = adapt_res.fallback_triggered;
  out_result->fallback_reason = adapt_res.fallback_reason;
  out_result->kl_divergence = adapt_res.kl_divergence;
  out_result->loss_value = adapt_res.loss_value;
  out_result->policy_grad_norm = adapt_res.policy_grad_norm;
  out_result->elapsed_seconds = adapt_res.elapsed_seconds;

  // 4. Branch 3: Comparator Selection (top 2 actions by prior, 4 selection rollouts each = 8 rollouts)
  Action top1 = raw_action;
  Action top2 = kInvalidAction;
  double top2_p = -1.0;
  for (const auto& ap : raw_prior) {
    if (ap.first != top1 && ap.second > top2_p) {
      top2_p = ap.second;
      top2 = ap.first;
    }
  }
  if (top2 == kInvalidAction && root.legal_actions.size() > 1) {
    for (Action a : root.legal_actions) {
      if (a != top1) {
        top2 = a;
        break;
      }
    }
  }

  Action comp_choice = raw_action;
  if (top2 != kInvalidAction) {
    std::vector<Action> candidate_pool = {top1, top2};
    std::vector<double> action_scores(2, 0.0);
    std::vector<int> action_counts(2, 0);

    struct CompTask {
      std::unique_ptr<State> state;
      int act_idx;
      uint64_t comp_seed;
      int stream_id;
    };
    std::vector<CompTask> comp_tasks;
    comp_tasks.reserve(8);

    for (int act_idx = 0; act_idx < 2; ++act_idx) {
      Action candidate_a = candidate_pool[act_idx];
      for (int c_roll = 0; c_roll < 4; ++c_roll) {
        uint64_t comp_seed = root.train_seed_base + 800 + act_idx * 4 + c_roll;
        std::unique_ptr<State> resampled = nullptr;
        if (hooks && hooks->resample_override) {
          resampled = hooks->resample_override(*dune, root.player, comp_seed);
        } else {
          std::mt19937 crng(comp_seed);
          auto crng_func = [&crng]() {
            return std::generate_canonical<double, 53>(crng);
          };
          resampled = dune->ResampleFromInfostate(root.player, crng_func);
        }
        if (!resampled) continue;
        resampled->ApplyAction(candidate_a);
        comp_tasks.push_back({std::move(resampled), act_idx, comp_seed, act_idx * 4 + c_roll});
      }
    }

    std::vector<std::future<std::pair<bool, double>>> comp_futures;
    comp_futures.reserve(comp_tasks.size());
    for (size_t i = 0; i < comp_tasks.size(); ++i) {
      auto* t_state = comp_tasks[i].state.get();
      uint64_t cseed = comp_tasks[i].comp_seed;
      int sid = comp_tasks[i].stream_id;
      comp_futures.push_back(std::async(
          std::launch::async,
          [t_state, &root, cand_evaluator, opp_evaluator, cseed, sid, budget, clock, root_start_time, &config, hooks]() {
            double c_ret = 0.0;
            bool ok = RunRolloutContinuation(
                t_state, root.player, cand_evaluator.get(), opp_evaluator.get(),
                cseed, sid, budget, clock, root_start_time, config.timeout_seconds,
                /*clock_stage=*/5, &c_ret, hooks);
            return std::make_pair(ok, c_ret);
          }));
    }

    bool comp_aborted = false;
    for (size_t i = 0; i < comp_futures.size(); ++i) {
      auto res = comp_futures[i].get();
      if (res.first) {
        action_scores[comp_tasks[i].act_idx] += res.second;
        action_counts[comp_tasks[i].act_idx]++;
      } else {
        comp_aborted = true;
      }
    }

    if (!comp_aborted && action_counts[0] > 0 && action_counts[1] > 0) {
      double mean0 = action_scores[0] / action_counts[0];
      double mean1 = action_scores[1] / action_counts[1];
      if (mean1 > mean0) {
        comp_choice = top2;
      } else {
        comp_choice = top1;
      }
    } else {
      comp_choice = raw_action;
    }
  }
  out_result->comp_action = comp_choice;
  if (clock) {
    std::cout << "    [Root " << root.root_id << " Stage 5 Comp: "
              << std::fixed << std::setprecision(2) << clock->ElapsedSeconds(root_start_time, 5)
              << "s, raw=" << raw_action << " adapt=" << out_result->adapt_action
              << " comp=" << comp_choice << "]\n" << std::flush;
  }

  // 5. Held-Out Evaluation: M=16 paired worlds across all 3 branches
  out_result->raw_returns.assign(config.m_eval_worlds, 0.0);
  out_result->adapt_returns.assign(config.m_eval_worlds, 0.0);
  out_result->comp_returns.assign(config.m_eval_worlds, 0.0);

  struct EvalRolloutTask {
    int m;
    int branch_idx; // 0=raw, 1=adapt, 2=comp
    std::unique_ptr<State> state;
    uint64_t eseed;
  };

  std::vector<EvalRolloutTask> eval_tasks;
  eval_tasks.reserve(config.m_eval_worlds * 3);

  bool adapt_needs_eval = (out_result->adapt_action != out_result->raw_action);
  bool comp_needs_eval = (out_result->comp_action != out_result->raw_action &&
                          out_result->comp_action != out_result->adapt_action);

  for (int m = 0; m < config.m_eval_worlds; ++m) {
    uint64_t eseed = root.eval_seed_base + m;
    std::unique_ptr<State> base_world = nullptr;
    if (hooks && hooks->resample_override) {
      base_world = hooks->resample_override(*dune, root.player, eseed);
    } else {
      std::mt19937 erng(eseed);
      auto erng_func = [&erng]() {
        return std::generate_canonical<double, 53>(erng);
      };
      base_world = dune->ResampleFromInfostate(root.player, erng_func);
    }
    if (!base_world) continue;

    // Branch 0: Raw
    std::unique_ptr<State> w0 = base_world->Clone();
    w0->ApplyAction(out_result->raw_action);
    eval_tasks.push_back({m, 0, std::move(w0), eseed});

    // Branch 1: Adapted (only if distinct action)
    if (adapt_needs_eval) {
      std::unique_ptr<State> w1 = base_world->Clone();
      w1->ApplyAction(out_result->adapt_action);
      eval_tasks.push_back({m, 1, std::move(w1), eseed});
    }

    // Branch 2: Comparator (only if distinct action)
    if (comp_needs_eval) {
      std::unique_ptr<State> w2 = base_world->Clone();
      w2->ApplyAction(out_result->comp_action);
      eval_tasks.push_back({m, 2, std::move(w2), eseed});
    }
  }

  std::vector<std::future<std::pair<bool, double>>> eval_futures;
  eval_futures.reserve(eval_tasks.size());

  for (size_t i = 0; i < eval_tasks.size(); ++i) {
    auto* t_state = eval_tasks[i].state.get();
    uint64_t eseed = eval_tasks[i].eseed;
    int m = eval_tasks[i].m;
    eval_futures.push_back(std::async(
        std::launch::async,
        [t_state, &root, cand_evaluator, opp_evaluator, eseed, m, budget, clock, root_start_time, &config, hooks]() {
          double ret = 0.0;
          bool ok = RunRolloutContinuation(
              t_state, root.player, cand_evaluator.get(), opp_evaluator.get(),
              eseed, m, budget, clock, root_start_time, config.timeout_seconds,
              /*clock_stage=*/6, &ret, hooks);
          return std::make_pair(ok, ret);
        }));
  }

  bool eval_all_ok = (eval_tasks.size() >= static_cast<size_t>(config.m_eval_worlds));
  for (size_t i = 0; i < eval_futures.size(); ++i) {
    auto res = eval_futures[i].get();
    if (!res.first) {
      eval_all_ok = false;
    } else {
      int m = eval_tasks[i].m;
      int b = eval_tasks[i].branch_idx;
      if (b == 0) out_result->raw_returns[m] = res.second;
      else if (b == 1) out_result->adapt_returns[m] = res.second;
      else if (b == 2) out_result->comp_returns[m] = res.second;
    }
  }

  // Route counterfactual identities if branches were skipped
  for (int m = 0; m < config.m_eval_worlds; ++m) {
    if (!adapt_needs_eval) {
      out_result->adapt_returns[m] = out_result->raw_returns[m];
    }
    if (!comp_needs_eval) {
      if (out_result->comp_action == out_result->raw_action) {
        out_result->comp_returns[m] = out_result->raw_returns[m];
      } else {
        out_result->comp_returns[m] = out_result->adapt_returns[m];
      }
    }
  }

  if (!eval_all_ok) {
    return false;
  }

  // Compute root summaries
  double sum_raw = 0.0, sum_adapt = 0.0, sum_comp = 0.0;
  for (int m = 0; m < config.m_eval_worlds; ++m) {
    sum_raw += out_result->raw_returns[m];
    sum_adapt += out_result->adapt_returns[m];
    sum_comp += out_result->comp_returns[m];
  }
  out_result->mean_raw = sum_raw / config.m_eval_worlds;
  out_result->mean_adapt = sum_adapt / config.m_eval_worlds;
  out_result->mean_comp = sum_comp / config.m_eval_worlds;
  out_result->paired_diff = out_result->mean_adapt - out_result->mean_raw;
  out_result->comp_diff = out_result->mean_comp - out_result->mean_raw;

  out_result->evaluation_completed = true;
  return true;
}

// ---------------------------------------------------------------------------
// Statistical Aggregation Across 32 Root Means
// ---------------------------------------------------------------------------
struct PilotAggregateStats {
  double mean_raw_utility = 0.0;
  double mean_adapt_utility = 0.0;
  double mean_comp_utility = 0.0;
  double delta_u_bar = 0.0;
  double s_root = 0.0;
  double se_delta_u_bar = 0.0;
  double ci_95_lower = 0.0;
  double ci_95_upper = 0.0;

  double comp_delta_u_bar = 0.0;
  double comp_s_root = 0.0;
  double comp_se_delta = 0.0;
  double comp_ci_95_lower = 0.0;
  double comp_ci_95_upper = 0.0;

  bool gate1_passed = false;
  bool gate2_passed = false;
  bool gate3_passed = false;
  bool gate4_passed = false;
  bool all_gates_passed = false;
};

inline PilotAggregateStats ComputePilotAggregateStats(
    const std::vector<RootExecutionResult>& results,
    uint64_t total_physical_batches,
    uint64_t total_logical_queries = 0,
    bool blueprint_preserved = true,
    const std::vector<int>& registered_root_ids = {}) {

  PilotAggregateStats stats;
  int n = results.size();
  if (n == 0) return stats;

  // Validate registered root IDs if provided
  if (!registered_root_ids.empty()) {
    if (results.size() != registered_root_ids.size()) {
      return stats;
    }
    for (size_t i = 0; i < results.size(); ++i) {
      if (results[i].root_id != registered_root_ids[i]) {
        return stats;
      }
    }
  }

  // Precondition checks: All results must have completed evaluation and 16 finite returns
  for (const auto& r : results) {
    if (!r.evaluation_completed) return stats;
    if (r.raw_returns.size() != 16 || r.adapt_returns.size() != 16 || r.comp_returns.size() != 16) {
      return stats;
    }
    for (int m = 0; m < 16; ++m) {
      if (!std::isfinite(r.raw_returns[m]) ||
          !std::isfinite(r.adapt_returns[m]) ||
          !std::isfinite(r.comp_returns[m])) {
        return stats;
      }
    }
  }

  // Recompute root means directly from raw return arrays, completely ignoring cached fields
  std::vector<double> root_diffs(n, 0.0);
  std::vector<double> root_comp_diffs(n, 0.0);
  double sum_raw = 0.0, sum_adapt = 0.0, sum_comp = 0.0;
  double sum_diff = 0.0, sum_comp_diff = 0.0;

  for (int i = 0; i < n; ++i) {
    const auto& r = results[i];
    double r_raw = 0.0, r_adapt = 0.0, r_comp = 0.0;
    for (int m = 0; m < 16; ++m) {
      r_raw += r.raw_returns[m];
      r_adapt += r.adapt_returns[m];
      r_comp += r.comp_returns[m];
    }
    double m_raw = r_raw / 16.0;
    double m_adapt = r_adapt / 16.0;
    double m_comp = r_comp / 16.0;
    double p_diff = m_adapt - m_raw;
    double c_diff = m_comp - m_raw;

    root_diffs[i] = p_diff;
    root_comp_diffs[i] = c_diff;
    sum_raw += m_raw;
    sum_adapt += m_adapt;
    sum_comp += m_comp;
    sum_diff += p_diff;
    sum_comp_diff += c_diff;
  }

  stats.mean_raw_utility = sum_raw / n;
  stats.mean_adapt_utility = sum_adapt / n;
  stats.mean_comp_utility = sum_comp / n;
  stats.delta_u_bar = sum_diff / n;
  stats.comp_delta_u_bar = sum_comp_diff / n;

  if (n > 1) {
    double var_diff = 0.0;
    double var_comp_diff = 0.0;
    for (int i = 0; i < n; ++i) {
      double d = root_diffs[i] - stats.delta_u_bar;
      var_diff += d * d;
      double cd = root_comp_diffs[i] - stats.comp_delta_u_bar;
      var_comp_diff += cd * cd;
    }
    stats.s_root = std::sqrt(var_diff / (n - 1));
    stats.se_delta_u_bar = stats.s_root / std::sqrt(n);
    stats.ci_95_lower = stats.delta_u_bar - 1.96 * stats.se_delta_u_bar;
    stats.ci_95_upper = stats.delta_u_bar + 1.96 * stats.se_delta_u_bar;

    stats.comp_s_root = std::sqrt(var_comp_diff / (n - 1));
    stats.comp_se_delta = stats.comp_s_root / std::sqrt(n);
    stats.comp_ci_95_lower = stats.comp_delta_u_bar - 1.96 * stats.comp_se_delta;
    stats.comp_ci_95_upper = stats.comp_delta_u_bar + 1.96 * stats.comp_se_delta;
  }

  // Gate evaluation:
  stats.gate1_passed = (stats.delta_u_bar > 0.0);
  stats.gate2_passed = (stats.ci_95_lower > 0.0);
  stats.gate3_passed = (stats.delta_u_bar >= stats.comp_delta_u_bar);
  stats.gate4_passed = (total_physical_batches <= 32000 &&
                        (total_logical_queries == 0 || total_logical_queries <= 350000) &&
                        blueprint_preserved);
  stats.all_gates_passed = (stats.gate1_passed && stats.gate2_passed && stats.gate3_passed && stats.gate4_passed);

  return stats;
}

}  // namespace dune_tt_adapt
}  // namespace open_spiel

#endif  // OPEN_SPIEL_EXAMPLES_DUNE_TEST_TIME_ADAPTATION_H_
