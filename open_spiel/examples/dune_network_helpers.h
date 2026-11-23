#ifndef OPEN_SPIEL_EXAMPLES_DUNE_NETWORK_HELPERS_H_
#define OPEN_SPIEL_EXAMPLES_DUNE_NETWORK_HELPERS_H_

#include <ATen/autocast_mode.h>
#include <torch/torch.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "dune_semantic_action_scorer.h"
#include "open_spiel/spiel.h"

namespace open_spiel {

struct AutocastGuard {
    c10::DeviceType device_type_;
    bool previous_state_;
    at::ScalarType previous_dtype_;
    bool clear_cache_on_exit_;
    AutocastGuard(c10::DeviceType device_type, bool enabled, bool clear_cache_on_exit = true)
        : device_type_(device_type), clear_cache_on_exit_(clear_cache_on_exit) {
        if (device_type_ == c10::DeviceType::CUDA) {
            previous_state_ = at::autocast::is_autocast_enabled(at::kCUDA);
            previous_dtype_ = at::autocast::get_autocast_dtype(at::kCUDA);
            at::autocast::set_autocast_dtype(at::kCUDA, at::ScalarType::BFloat16);
            at::autocast::set_autocast_enabled(at::kCUDA, enabled);
        } else if (device_type_ == c10::DeviceType::CPU) {
            previous_state_ = at::autocast::is_autocast_enabled(at::kCPU);
            previous_dtype_ = at::autocast::get_autocast_dtype(at::kCPU);
            at::autocast::set_autocast_enabled(at::kCPU, enabled);
        } else {
            previous_state_ = false;
            previous_dtype_ = at::ScalarType::Undefined;
        }
    }
    ~AutocastGuard() {
        // PWO-5 section 7.5. Autocast caches the lower-precision cast of every
        // fp32 LEAF parameter that requires grad, keyed by TensorImpl*, and that
        // cache is NOT scoped to the region -- `set_autocast_enabled(false)`
        // does not clear it. Upstream's own context manager clears it on region
        // exit; this guard sets the flags directly and so never did.
        //
        // The consequence is not a slow path, it is a WRONG one: the optimizer
        // mutates parameters IN PLACE, which leaves the TensorImpl* unchanged,
        // so the next region's forward silently reuses the bf16 cast of the
        // weights as they were on this process's FIRST autocast region. Every
        // PPO minibatch after the first therefore differentiated a stale
        // linearization point, and a run that executed update 1 in-process
        // computed a different update 2 than a run that restored the identical
        // state from a checkpoint -- which is exactly the section 7.5 bitwise
        // continuation failure, deterministic in both directions.
        //
        // Clearing on scope exit restores the upstream contract: a region's
        // cache does not outlive the region. It cannot change a value that was
        // not already stale -- the only effect is to force `arg.to(bf16)` to be
        // recomputed, and a cast is deterministic.
        //
        // MEASURED, not assumed (scratch probe, this box, the linked torch
        // 2.12.1+cu130 that `ldd dune_ppo_train` reports -- NOT the unused
        // vendored 2.3.0 tree): staleness is gated on `requires_grad` alone and
        // is independent of grad mode -- requires_grad=true goes stale under
        // both `NoGradGuard` and grad-enabled; requires_grad=false never
        // caches. `inference_model` is `eval()`-ed but its parameters are NOT
        // set_requires_grad(false), so it is cache-eligible and
        // `CopyModelWeights`' in-place `copy_` does leave a stale entry behind
        // on whichever thread holds it (`cached_casts` is thread_local).
        //
        // The evaluation binaries are unaffected for a STRUCTURAL reason, not
        // because of a requires_grad setting: a cached cast can only be wrong
        // if the parameter is mutated in place between regions, and they load a
        // checkpoint once and never mutate it.
        if (device_type_ == c10::DeviceType::CUDA) {
            at::autocast::set_autocast_enabled(at::kCUDA, previous_state_);
            at::autocast::set_autocast_dtype(at::kCUDA, previous_dtype_);
            if (clear_cache_on_exit_) {
                at::autocast::clear_cache();
            }
        } else if (device_type_ == c10::DeviceType::CPU) {
            at::autocast::set_autocast_enabled(at::kCPU, previous_state_);
            at::autocast::set_autocast_dtype(at::kCPU, previous_dtype_);
            if (clear_cache_on_exit_) {
                at::autocast::clear_cache();
            }
        }
    }
};

struct EvalResult {
    std::vector<float> logits;
    float value;
    bool ok = true;
    bool budget_exhausted = false;
    // Opt-in physical-batch membership for read-only numerical diagnostics.
    // Default sentinels preserve every existing evaluator consumer and make an
    // uninstrumented result mechanically distinguishable from batch 0/row 0.
    int64_t physical_batch_id = -1;
    int32_t physical_batch_size = -1;
    int32_t physical_batch_row = -1;
};

struct CompactEvalResult {
    std::vector<Action> actions;
    std::vector<float> probabilities;
    float value = 0.0f;
    bool ok = true;
    bool budget_exhausted = false;
    int64_t physical_batch_id = -1;
    int32_t physical_batch_size = -1;
    int32_t physical_batch_row = -1;
};

struct GreedyEvalResult {
    Action action = kInvalidAction;
    bool ok = true;
    bool budget_exhausted = false;
};

struct LogitCapApplicationStats {
  double pre_max_abs = 0.0;
  double post_max_abs = 0.0;
  int legal_count = 0;
  int saturated_count = 0;
};

// Lifetime counters for one BatchedEvaluator. Async telemetry records the
// delta since the prior generation publication, so the scope of a reported
// cap statistic is explicit and never confused with a cumulative total.
struct BatchedLogitCapAggregate {
  uint64_t decisions = 0;
  uint64_t legal_logits = 0;
  uint64_t saturated = 0;
  double pre_max_abs = 0.0;
  double post_max_abs = 0.0;
};

inline LogitCapApplicationStats CenterAndCapLegalLogitsWithStats(
    std::vector<float>& logits, const std::vector<Action>& legal_actions,
    float logit_cap) {
  LogitCapApplicationStats stats;
  if (logits.empty() || legal_actions.empty()) return stats;

    double legal_sum = 0.0;
    int legal_count = 0;
    for (Action action : legal_actions) {
        if (action >= 0 && static_cast<size_t>(action) < logits.size()) {
            legal_sum += logits[action];
            ++legal_count;
        }
    }
  if (legal_count == 0) return stats;

    const float legal_mean = static_cast<float>(legal_sum / legal_count);
    // Only legal logits are consumed by the softmax. Transforming all 2,391
    // outputs performed thousands of unnecessary tanh calls per game.
    for (Action action : legal_actions) {
    if (action < 0 || static_cast<size_t>(action) >= logits.size()) continue;
    float& logit = logits[action];
    logit -= legal_mean;
    stats.pre_max_abs =
        std::max(stats.pre_max_abs, std::abs(static_cast<double>(logit)));
    ++stats.legal_count;
    if (logit_cap > 0.0f) {
      logit = logit_cap * std::tanh(logit / logit_cap);
    }
    stats.post_max_abs =
        std::max(stats.post_max_abs, std::abs(static_cast<double>(logit)));
    if (logit_cap > 0.0f &&
        std::abs(static_cast<double>(logit)) >= 0.99 * logit_cap) {
      ++stats.saturated_count;
    }
  }
  return stats;
}

inline void CenterAndCapLegalLogits(std::vector<float>& logits,
                                    const std::vector<Action>& legal_actions,
                                    float logit_cap) {
  CenterAndCapLegalLogitsWithStats(logits, legal_actions, logit_cap);
}

// Samples one legal action from `logits` (already legal-centered and capped by
// CenterAndCapLegalLogits) and returns it together with the log-probability of
// the distribution that was ACTUALLY sampled from, so a stored `old_log_prob`
// always describes the sampling law.
//
// Weights are the exact legal-softmax numerators exp(logit - max_legal_logit).
// An action whose exponent underflows to zero stays at zero: it must not
// inherit exp(0) == 1, which is the weight of the argmax. Only when the whole
// legal distribution carries no usable mass — no legal action has a finite
// logit, or every weight underflowed — do we fall back to uniform, and the
// returned log-probability then describes that same uniform law.
//
// Shared by the PPO rollout and the swordmaster planner rollout so the two
// cannot drift apart in their policy transform.
struct PolicyDistributionSample {
    Action action = kInvalidAction;
    size_t chosen_index = 0;
    float chosen_log_probability = 0.0f;
    bool leader_draft_overridden = false;
};

namespace policy_sampling_internal {

template <typename RngType>
inline std::pair<Action, float> SamplePolicyDistributionImpl(
    RngType* rng, const std::vector<float>& logits,
    const std::vector<Action>& legal_actions,
    std::vector<double>* ordered_legal_probabilities,
    size_t* chosen_index, int64_t* probability_output_writes) {
    SPIEL_CHECK_FALSE(legal_actions.empty());

    float max_logit = -std::numeric_limits<float>::infinity();
    for (Action legal_action : legal_actions) {
        if (legal_action >= 0 &&
            static_cast<size_t>(legal_action) < logits.size()) {
            max_logit = std::max(max_logit, logits[legal_action]);
        }
    }

    std::vector<double> weights;
    weights.reserve(legal_actions.size());
    double total_weight = 0.0;
    for (Action legal_action : legal_actions) {
        // Zero, not one: an action with no logit of its own carries no mass.
        double weight = 0.0;
        if (legal_action >= 0 &&
            static_cast<size_t>(legal_action) < logits.size() &&
            std::isfinite(max_logit)) {
            weight = std::exp(static_cast<double>(logits[legal_action] - max_logit));
            if (!std::isfinite(weight) || weight < 0.0) weight = 0.0;
        }
        weights.push_back(weight);
        total_weight += weight;
    }

    if (!(total_weight > 0.0) || !std::isfinite(total_weight)) {
        weights.assign(legal_actions.size(), 1.0);
        total_weight = static_cast<double>(legal_actions.size());
    }

    std::discrete_distribution<size_t> dist(weights.begin(), weights.end());
    size_t sampled_index = dist(*rng);
    // A zero-weight element is never drawn, so the sampled probability is
    // strictly positive and its log is exact. Deliberately NOT floored: with the
    // cap disabled a legal action can sit ~30 nats below the max, and clamping
    // its probability up to 1e-12 would hand PPO an old_log_prob the sampler
    // never used — while the update recomputes the true log_softmax, unfloored.
    double prob = weights[sampled_index] / total_weight;
    SPIEL_CHECK_GT(prob, 0.0);
    if (chosen_index != nullptr) *chosen_index = sampled_index;
    if (ordered_legal_probabilities != nullptr) {
      ordered_legal_probabilities->clear();
      ordered_legal_probabilities->reserve(weights.size());
      for (double weight : weights) {
        ordered_legal_probabilities->push_back(weight / total_weight);
        if (probability_output_writes != nullptr) {
          ++*probability_output_writes;
        }
      }
    }
    return {legal_actions[sampled_index],
            static_cast<float>(std::log(prob))};
}

}  // namespace policy_sampling_internal

template <typename RngType>
inline PolicyDistributionSample SamplePolicyDistribution(
    RngType* rng, const std::vector<float>& logits,
    const std::vector<Action>& legal_actions,
    std::vector<double>* ordered_legal_probabilities) {
    PolicyDistributionSample result;
    const auto pair = policy_sampling_internal::SamplePolicyDistributionImpl(
        rng, logits, legal_actions, ordered_legal_probabilities,
        &result.chosen_index, nullptr);
    result.action = pair.first;
    result.chosen_log_probability = pair.second;
    return result;
}

template <typename RngType>
inline std::pair<Action, float> SamplePolicyAction(
    RngType* rng, const std::vector<float>& logits,
    const std::vector<Action>& legal_actions) {
    return policy_sampling_internal::SamplePolicyDistributionImpl(
        rng, logits, legal_actions,
        /*ordered_legal_probabilities=*/nullptr,
        /*chosen_index=*/nullptr,
        /*probability_output_writes=*/nullptr);
}

struct EvaluatorStats {
    uint64_t batches = 0;
    uint64_t requests = 0;
    uint64_t max_batch_size = 0;
    double avg_batch_size = 0.0;
};

// Observation-size contract shared by the evaluators (WO-02, search finding 6).
// An observation is accepted only if its length exactly matches the model input
// dimension, or forms the single supported cross-layout pair (5580 <-> 5584).
// This mirrors DuneNNEvaluator::CheckObsSize exactly; any other length -- even a
// near miss such as 5581 or 5579 -- is a genuine model/observation mismatch and
// is rejected here rather than being silently zero-padded or truncated in the
// copy loops (which would otherwise drop or fabricate feature columns).
inline void CheckEvalObsSize(size_t obs_size, int64_t model_input_dim) {
    const int64_t size = static_cast<int64_t>(obs_size);
    const bool ok = (size == model_input_dim) ||
                    (size == 5580 && model_input_dim == 5584) ||
                    (size == 5584 && model_input_dim == 5580) ||
                    (size == 6215 && model_input_dim == 5580) ||
                    (size == 5580 && model_input_dim == 6215) ||
                    (size == 6215 && model_input_dim == 5584) ||
                    (size == 5584 && model_input_dim == 6215);
    if (!ok) {
        SpielFatalError("Evaluator observation size " +
                        std::to_string(obs_size) +
                        " is incompatible with model input dim " +
                        std::to_string(model_input_dim) +
                        " (expected an exact match or supported cross-layout pair).");
    }
}

class IGameEvaluator {
public:
    virtual ~IGameEvaluator() = default;
    virtual EvalResult Evaluate(const std::vector<float>& obs) = 0;
    virtual EvalResult EvaluateWithActions(
        const std::vector<float>& obs,
        const dune_semantic::CandidateActionData* action_data) {
        return Evaluate(obs);
    }
    virtual std::vector<EvalResult> EvaluateBatch(
        const std::vector<std::vector<float>>& observations) {
        std::vector<EvalResult> results;
        results.reserve(observations.size());
        for (const auto& obs : observations) {
            results.push_back(Evaluate(obs));
        }
        return results;
    }

    virtual EvaluatorStats GetStats() const = 0;
    virtual bool HasSemanticScorer() const { return false; }
    virtual std::string SemanticDescriptorSchema() const {
        return dune_semantic::kDescriptorSchemaVersionV3;
    }
    virtual dune_imperium::DeckPoolSemantics DeckPoolSemantics() const {
        return dune_imperium::DeckPoolSemantics::kLegacy;
    }
    virtual dune_imperium::MarketAppendixMode MarketMode() const {
        return dune_imperium::MarketAppendixMode::kNone;
    }
    virtual void AdjustActiveRollouts(int delta) {}
};

}  // namespace open_spiel

#endif  // OPEN_SPIEL_EXAMPLES_DUNE_NETWORK_HELPERS_H_
