#pragma once

#include <torch/torch.h>
#include <c10/cuda/CUDAFunctions.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <array>
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>
#include <deque>
#include <unordered_map>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <chrono>
#include <random>
#include <shared_mutex>
#include <utility>
#include <ATen/autocast_mode.h>
#include "open_spiel/examples/dune_hotspot_profile.h"

#include "open_spiel/spiel.h"
#include "dune_seed_utils.h"  // MakeTorchCPUGenerator, for the PWO-5 head init

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

namespace open_spiel {

// 1. Residual Block Definition
struct ResBlockImpl : torch::nn::Module {
  torch::nn::Linear fc1{nullptr};
  torch::nn::Linear fc2{nullptr};
  torch::nn::LayerNorm ln1{nullptr};
  torch::nn::LayerNorm ln2{nullptr};

  ResBlockImpl(int64_t dim) {
    fc1 = register_module("fc1", torch::nn::Linear(dim, dim));
    fc2 = register_module("fc2", torch::nn::Linear(dim, dim));
    ln1 = register_module("ln1", torch::nn::LayerNorm(torch::nn::LayerNormOptions({dim})));
    ln2 = register_module("ln2", torch::nn::LayerNorm(torch::nn::LayerNormOptions({dim})));
  }

  torch::Tensor forward(torch::Tensor x) {
    torch::Tensor residual = x;
    x = torch::relu(ln1->forward(fc1->forward(x)));
    x = ln2->forward(fc2->forward(x));
    return torch::relu(x + residual);
  }
};
TORCH_MODULE(ResBlock);

// Multiplier applied to the value head's scalar output layer at construction.
// See the rationale block in the constructor below.
inline constexpr double kValueHeadOutputInitScale = 0.01;

// PWO-5 section 7.2: the same x0.01 applied to the three auxiliary heads'
// output weights, for the same reason and with the same precedent.
inline constexpr double kAuxHeadOutputInitScale = 0.01;

// 2. Main Dual-Headed Policy-Value Network Definition
struct SharedDunePolicyValueNetImpl : torch::nn::Module {
  torch::nn::Linear input_layer{nullptr};
  std::vector<std::shared_ptr<ResBlockImpl>> res_blocks;
  torch::nn::Linear policy_head{nullptr};
  torch::nn::Linear value_head{nullptr};
  torch::nn::Linear value_head2{nullptr};
  bool use_nonlinear_value_head_ = false;

  // --- PWO-5 section 7.2: the three auxiliary heads. -----------------------
  //
  // Constructed ONLY when `with_aux_heads` is true, which ONLY dune_ppo_train
  // passes. That default is load-bearing three times over:
  //
  //   * every evaluation binary keeps the pre-migration module set, so it can
  //     still load the frozen Branch-A u2450 opponent checkpoint, which has no
  //     head tensors;
  //   * a candidate checkpoint that DOES have them still loads into a
  //     head-less module, because torch's Module::load walks the MODULE's own
  //     members and ignores extra archive entries;
  //   * it makes section 9.5's "auxiliary outputs unused at runtime" a
  //     STRUCTURAL fact rather than a convention -- the evaluator cannot
  //     compute a head output because the head does not exist in its process.
  //
  // The heads are NEVER computed by forward(). Section 8.6 registers a
  // "dedicated auxiliary forward path": ForwardAux() below is the only entry
  // point that touches them, and it is called only from the auxiliary slice of
  // the PPO minibatch loop, only when a head coefficient is nonzero. So a
  // head-off arm (T, P) never computes a head output at all, which is what
  // makes the head-off claim exact rather than a multiply-by-zero.
  torch::nn::Linear final_vp_head{nullptr};        // 1 output
  torch::nn::Linear terminal_round_head{nullptr};  // 3 classes: <=8 / 9 / 10
  torch::nn::Linear next_own_action_head{nullptr}; // action_dim classes
  bool has_aux_heads_ = false;

  SharedDunePolicyValueNetImpl(int64_t input_dim, int64_t hidden_dim = 2048, int64_t action_dim = 2391, int num_blocks = 8, bool use_nonlinear = false, bool with_aux_heads = false, uint64_t head_init_seed = 0) {
    use_nonlinear_value_head_ = use_nonlinear;
    input_layer = register_module("input_layer", torch::nn::Linear(input_dim, hidden_dim));

    // Register custom submodules dynamically (res1, res2, etc.)
    for (int i = 0; i < num_blocks; ++i) {
      auto block = std::make_shared<ResBlockImpl>(hidden_dim);
      res_blocks.push_back(block);
      register_module("res" + std::to_string(i + 1), block);
    }

    policy_head = register_module("policy_head", torch::nn::Linear(hidden_dim, action_dim));
    if (use_nonlinear) {
      value_head = register_module("value_head", torch::nn::Linear(hidden_dim, hidden_dim / 2));
      value_head2 = register_module("value_head2", torch::nn::Linear(hidden_dim / 2, 1));
    } else {
      value_head = register_module("value_head", torch::nn::Linear(hidden_dim, 1));
    }

    // Small-init on the scalar-producing value layer (canonical PPO practice).
    //
    // Without it a from-scratch run drives the tanh at :forward() into hard
    // saturation on the FIRST optimizer step and stays pinned there for
    // hundreds of updates. The trap is not a step-size problem, so neither
    // grad_clip_norm nor a lower learning rate escapes it:
    //
    //   - the trunk output x is post-ReLU, so every component is >= 0 (~80%
    //     strictly positive, mean ~1.8, sum_i mean(x_i) ~ 3.6e3 at hidden=2048);
    //   - AdamW's first step is lr * m_hat/sqrt(v_hat) = lr * sign(g) for every
    //     coordinate, independent of |g|. Clipping rescales g and is therefore
    //     a no-op on the step actually taken;
    //   - dL/dx = delta * w, so with default init (|w|_2 ~ 0.57) the whole
    //     shared trunk is sign-stepped coherently in the one direction that
    //     moves the value output, and ~6.7e7 trunk coordinates each contribute
    //     lr to the same scalar. The resulting first-step displacement of the
    //     pre-tanh z is ~45 — two orders of magnitude past the [-0.4375,
    //     +0.5625] target range — so |v| lands at ~1.0 and 1 - v^2 then
    //     throttles the value gradient by ~50x while AdamW's beta2 memory still
    //     carries the pre-saturation second moment.
    //
    // Scaling this layer by 0.01 shrinks both z and dL/dx by the same factor
    // and keeps the critic inside the linear region of the tanh, where the
    // gradient it needs to fit the placement targets survives.
    //
    // This runs at CONSTRUCTION only. Every checkpoint path constructs and then
    // overwrites via torch::load, so transfer runs are bitwise unaffected. The
    // single exception is the nonlinear-head partial-copy path
    // (dune_ppo_train.cc LoadModelCheckpoint), which deliberately does not copy
    // the value head — there the fresh head is exactly the scratch-critic case
    // this guards, so the smaller init applies by design.
    {
      torch::NoGradGuard no_grad;
      torch::nn::Linear& value_out = use_nonlinear ? value_head2 : value_head;
      value_out->weight.mul_(kValueHeadOutputInitScale);
      if (value_out->bias.defined()) {
        value_out->bias.mul_(kValueHeadOutputInitScale);
      }
    }

    // --- PWO-5 section 7.2: deterministic auxiliary-head initialization. ----
    //
    // Registered recipe: linear from the trunk, "weights x 0.01 of the standard
    // init, bias 0", derived from a FIXED REGISTERED CONSTANT and NOT from the
    // run seed, so that the initial head parameters are BYTE-IDENTICAL ACROSS
    // ALL SIX ARMS (section 9's matching table checks this rather than assuming
    // it). Deriving from the triplet seed would give T1/P1/H1 one set and
    // T2/P2/H2 another, which satisfies the ablation's within-triplet
    // requirement but NOT the stronger all-six property the work order states.
    //
    // The x0.01 mirrors this repo's verified critic-init fix (submodule
    // b48a35ee): a standard-init tanh value head saturated at update 2 and
    // dwelt there ~837 updates. A near-zero-output init also keeps the
    // auxiliary gradients small at the start, so H's early trajectory is not
    // dominated by head transients.
    //
    // Determinism: torch::nn::Linear's constructor above already drew from the
    // global generator, so we OVERWRITE both tensors from a private,
    // seed-derived CPU generator instead of relying on that draw. Whatever the
    // ambient global RNG state is, these three heads come out identical.
    if (with_aux_heads) {
      has_aux_heads_ = true;
      final_vp_head =
          register_module("final_vp_head", torch::nn::Linear(hidden_dim, 1));
      terminal_round_head =
          register_module("terminal_round_head", torch::nn::Linear(hidden_dim, 3));
      next_own_action_head = register_module(
          "next_own_action_head", torch::nn::Linear(hidden_dim, action_dim));

      torch::NoGradGuard no_grad;
      at::Generator head_gen = dune_seed::MakeTorchCPUGenerator(head_init_seed);
      // kaiming_uniform_ with a=sqrt(5) is what torch::nn::Linear's own
      // reset_parameters() uses, so "the standard init" is reproduced exactly
      // and then scaled -- rather than a different distribution scaled to look
      // similar. bias is set to EXACTLY zero, per the registered recipe.
      auto init_head = [&](torch::nn::Linear& head) {
        const double bound =
            std::sqrt(6.0 / ((1.0 + 5.0) * head->weight.size(1)));
        head->weight.uniform_(-bound, bound, head_gen);
        head->weight.mul_(kAuxHeadOutputInitScale);
        if (head->bias.defined()) head->bias.zero_();
      };
      init_head(final_vp_head);
      init_head(terminal_round_head);
      init_head(next_own_action_head);
    }
  }

  struct ModelOutputs {
    torch::Tensor logits;
    torch::Tensor values;
  };

  // The three auxiliary head outputs. Produced ONLY by ForwardAux().
  struct AuxOutputs {
    torch::Tensor final_vp;        // [B, 1]
    torch::Tensor terminal_round;  // [B, 3]  logits
    torch::Tensor next_own_action; // [B, action_dim]  logits
  };

  // Shared trunk. Split out of forward() so ForwardAux() reuses exactly the
  // same computation rather than a parallel copy of it.
  torch::Tensor Trunk(torch::Tensor x) {
    x = torch::relu(input_layer->forward(x));
    for (auto& block : res_blocks) {
      x = block->forward(x);
    }
    return x;
  }

  ModelOutputs forward(torch::Tensor x) {
    x = Trunk(x);

    torch::Tensor logits = policy_head->forward(x);
    torch::Tensor values;
    if (use_nonlinear_value_head_) {
      auto h = torch::relu(value_head->forward(x));
      values = torch::tanh(value_head2->forward(h));
    } else {
      values = torch::tanh(value_head->forward(x));
    }
    return {logits, values};
  }

  // PWO-5 section 8.6's DEDICATED AUXILIARY FORWARD PATH.
  //
  // The three heads are computed here and NOWHERE ELSE -- never in the PPO
  // forward, never in the rollout/inference forward, never in an evaluation
  // forward. Callers must check the coefficients first; this asserts rather
  // than silently returning empty tensors, because a head-off arm reaching this
  // function at all is a bug in the caller, not a no-op.
  AuxOutputs ForwardAux(torch::Tensor x) {
    if (!has_aux_heads_) {
      throw std::runtime_error(
          "ForwardAux called on a model constructed without auxiliary heads. "
          "The heads are a dedicated forward path (PWO-5 section 8.6); a "
          "head-off arm must never reach this call.");
    }
    torch::Tensor h = Trunk(x);
    return {final_vp_head->forward(h), terminal_round_head->forward(h),
            next_own_action_head->forward(h)};
  }
};
TORCH_MODULE(SharedDunePolicyValueNet);

struct AutocastGuard {
    c10::DeviceType device_type_;
    bool previous_state_;
    at::ScalarType previous_dtype_;
    AutocastGuard(c10::DeviceType device_type, bool enabled)
        : device_type_(device_type) {
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
            at::autocast::clear_cache();
        } else if (device_type_ == c10::DeviceType::CPU) {
            at::autocast::set_autocast_enabled(at::kCPU, previous_state_);
            at::autocast::set_autocast_dtype(at::kCPU, previous_dtype_);
            at::autocast::clear_cache();
        }
    }
};

struct EvalResult {
    std::vector<float> logits;
    float value;
};

struct CompactEvalResult {
    std::vector<Action> actions;
    std::vector<float> probabilities;
    float value = 0.0f;
};

struct LogitCapApplicationStats {
  double pre_max_abs = 0.0;
  double post_max_abs = 0.0;
  int legal_count = 0;
  int saturated_count = 0;
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
template <typename RngType>
inline std::pair<Action, float> SamplePolicyAction(
    RngType* rng, const std::vector<float>& logits,
    const std::vector<Action>& legal_actions) {
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
    return {legal_actions[sampled_index], static_cast<float>(std::log(prob))};
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
                    (size == 5584 && model_input_dim == 5580);
    if (!ok) {
        SpielFatalError("Evaluator observation size " +
                        std::to_string(obs_size) +
                        " is incompatible with model input dim " +
                        std::to_string(model_input_dim) +
                        " (expected an exact match or the 5580/5584 pair).");
    }
}

class IGameEvaluator {
public:
    virtual ~IGameEvaluator() = default;
    virtual EvalResult Evaluate(const std::vector<float>& obs) = 0;
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
};

class DeterministicEvaluator : public IGameEvaluator {
public:
    DeterministicEvaluator(std::shared_ptr<SharedDunePolicyValueNetImpl> model,
                           torch::Device device,
                           std::mutex* mutex,
                           std::shared_mutex* sync_mutex = nullptr)
        : model_(model), device_(device), mutex_(mutex), sync_mutex_(sync_mutex) {
        model_input_dim_ = model_->input_layer->weight.size(1);
        action_dim_ = model_->policy_head->weight.size(0);
    }

    EvalResult Evaluate(const std::vector<float>& obs) override {
        torch::NoGradGuard no_grad;
        EvalResult result;
        result.logits.resize(action_dim_);

        // Reject observation sizes incompatible with the model input rather than
        // silently truncating/over-reading them (WO-02 finding 6).
        CheckEvalObsSize(obs.size(), model_input_dim_);

        // Per-instance staging buffers (NOT thread_local). Previously these lived
        // in a thread_local map keyed by `this`, so at end of run all 64 worker
        // threads destroyed their pinned host tensor simultaneously; the pinned
        // free path (CachingHostAllocator::free -> CUDAEvent::record ->
        // cudaStreamIsCapturing) ran concurrently across the exiting threads and
        // segfaulted libcuda inside __call_tls_dtors, right after the final game.
        // Making them per-instance members frees each buffer exactly once, in the
        // main thread at evaluator destruction, after the workers have joined.
        //
        // Staging now runs inside the global eval mutex because the buffers are
        // shared across threads. `*mutex_` already serializes every Evaluate call
        // (one mutex for the model + all opponents), so serializing the tiny
        // memcpy/memset stage as well leaves throughput and numerics unchanged.
        torch::Tensor pred_logits;
        torch::Tensor pred_values;
        {
            std::shared_lock<std::shared_mutex> sync_lock;
            if (sync_mutex_ != nullptr) {
                sync_lock = std::shared_lock<std::shared_mutex>(*sync_mutex_);
            }
            std::lock_guard<std::mutex> lock(*mutex_);

            if (!input_tensor_.defined()) {
                auto options = torch::TensorOptions().dtype(torch::kFloat32);
                if (device_.is_cuda()) {
                    options = options.pinned_memory(true);
                }
                input_tensor_ = torch::empty({1, model_input_dim_}, options);
                if (device_.is_cuda()) {
                    device_tensor_ = torch::empty({1, model_input_dim_}, torch::TensorOptions().dtype(torch::kFloat32).device(device_));
                }
            }

            float* data_ptr = input_tensor_.data_ptr<float>();
            int64_t copy_size = std::min<int64_t>(model_input_dim_, obs.size());
            std::memcpy(data_ptr, obs.data(), copy_size * sizeof(float));
            if (copy_size < model_input_dim_) {
                std::memset(data_ptr + copy_size, 0, (model_input_dim_ - copy_size) * sizeof(float));
            }

            SharedDunePolicyValueNetImpl::ModelOutputs outputs;
            if (device_.is_cuda()) {
                // Blocking H2D (finding 1): the shared pinned input_tensor_ must
                // not be reused by the next thread until this copy's DMA has
                // finished reading it. non_blocking=false makes copy_ host-
                // synchronous (Torch inserts a stream sync), so the pinned
                // staging buffer is free the moment copy_ returns.
                device_tensor_.copy_(input_tensor_, /*non_blocking=*/false);
                AutocastGuard autocast_guard(c10::DeviceType::CUDA, true);
                outputs = model_->forward(device_tensor_);
            } else {
                outputs = model_->forward(input_tensor_);
            }
            // Blocking device->host copy INSIDE the lock drains the stream before
            // the mutex is released, so the next thread cannot overwrite the
            // shared staging buffers while this request's async work may still be
            // reading them (belt-and-suspenders for the finding-1 race). No
            // numeric change: identical ops, just serialized under the lock.
            pred_logits = outputs.logits.to(torch::kCPU).to(torch::kFloat32);
            pred_values = outputs.values.to(torch::kCPU).to(torch::kFloat32);
        }

        std::memcpy(result.logits.data(), pred_logits.data_ptr<float>(), action_dim_ * sizeof(float));
        result.value = pred_values.data_ptr<float>()[0];

        requests_.fetch_add(1, std::memory_order_relaxed);
        return result;
    }

    EvaluatorStats GetStats() const override {
        EvaluatorStats s;
        s.requests = requests_.load(std::memory_order_relaxed);
        s.batches = s.requests;
        s.max_batch_size = 1;
        s.avg_batch_size = 1.0;
        return s;
    }

private:
    std::shared_ptr<SharedDunePolicyValueNetImpl> model_;
    torch::Device device_;
    std::mutex* mutex_;
    std::shared_mutex* sync_mutex_;
    int64_t model_input_dim_;
    int64_t action_dim_;
    // Per-instance staging buffers, allocated lazily under mutex_ and freed once
    // by the owning (main) thread at destruction. Replaces the thread_local map
    // whose 64-way concurrent pinned-tensor free at thread exit segfaulted libcuda.
    torch::Tensor input_tensor_;   // CPU staging (pinned host memory when CUDA)
    torch::Tensor device_tensor_;  // device-side input buffer (CUDA only)
    mutable std::atomic<uint64_t> requests_{0};
};

// WO-PERF-3: permanent, opt-in telemetry for the shared batched-inference
// coordinator. Every field here is DERIVED from counters the runner/actor
// threads increment purely as a side effect; none of it changes which requests
// land in which physical batch, so it is behaviour-neutral for every existing
// consumer of BatchedEvaluator. `submitted_rows`/`physical_batches`/
// `max_batch_size` duplicate EvaluatorStats so the batcher can be read through
// one struct. The logical-call decomposition is BY SUBMISSION METHOD: a
// one-row `Evaluate()` (opponent Prior or an out-of-MCTS raw prior) increments
// `single_row_calls`; a `EvaluateBatch()` of N rows (a searched-leaf
// PriorAndEvaluate/Evaluate group) increments `group_calls` and adds N to
// `group_rows`. Hence the identity `submitted_rows == single_row_calls +
// group_rows` holds by construction (Phase-2 "rows are not logical calls").
// Device-side timings (H2D/D2H/forward/sync, utilisation) are intentionally
// ABSENT under RC-1 (no GPU); a GPU follow-up WO adds them in the runner.
struct BatcherTelemetry {
    // Row/batch level.
    uint64_t submitted_rows = 0;          // total observation rows enqueued
    uint64_t physical_batches = 0;        // model forwards actually run
    uint64_t max_batch_size = 0;          // largest realised physical batch
    double mean_batch_size = 0.0;
    uint64_t p50_batch_size = 0;
    uint64_t p95_batch_size = 0;
    int target_batch_size = 0;            // configured target row count
    int timeout_ms = 0;                   // configured max queue wait
    double target_occupancy = 0.0;        // mean_batch_size / target_batch_size
    uint64_t timeout_flush_batches = 0;   // batches dispatched under-target on deadline
    // Logical calls by request class.
    uint64_t single_row_calls = 0;        // Evaluate() invocations (1 row each)
    uint64_t group_calls = 0;             // EvaluateBatch() invocations (leaf groups)
    uint64_t group_rows = 0;              // sum of rows across EvaluateBatch() calls
    uint64_t leaf_groups_split = 0;       // leaf groups whose rows spanned >1 batch
    // Queue-wait by request class (per row, milliseconds).
    double single_wait_ms_mean = 0.0;
    double single_wait_ms_max = 0.0;
    uint64_t single_wait_n = 0;
    double group_wait_ms_mean = 0.0;
    double group_wait_ms_max = 0.0;
    uint64_t group_wait_n = 0;
    // --- Device-side time (WO-PERF-TIMING-BATCH, 2026-08-16) ----------------
    //
    // Cumulative milliseconds across physical batches, measured with CUDA
    // events on the compute stream. Zero on CPU and zero unless
    // EnableBatcherTelemetry() armed the batcher, exactly like the sections
    // above.
    //
    // These exist because the Phase-2 promotion gate asks for the H2D /
    // forward / D2H / synchronisation split, and the original WO-PERF-3
    // telemetry deliberately omitted it: that work was done under a no-GPU
    // constraint, so it was deferred as a "GPU follow-up deliverable". Without
    // these fields the gate cannot be satisfied, and reporting the split as
    // "unavailable" is not the same as measuring it.
    //
    // Read them the way the trainer's phase timer documents: a CUDA event pair
    // spans the interval between two markers EXECUTING on the stream, so it
    // includes stream idle. `sync_ms` is host wall time, because a
    // synchronise is a host-blocking call and host time is the honest measure
    // of it.
    // THESE INTERVALS OVERLAP AND MUST NOT BE NORMALISED INTO ONE ADDITIVE
    // PERCENTAGE SPLIT. h2d/forward/output_cast/d2h are CUDA-event intervals on
    // the compute stream; sync_ms is HOST wall time spent blocked in
    // torch::cuda::synchronize(). The host sync overlaps the device work it is
    // waiting for, so (h2d + forward + output_cast + d2h + sync) is NOT a total
    // and a percentage of that sum is meaningless. Report sync_ms separately.
    double h2d_ms = 0.0;
    double forward_ms = 0.0;
    double output_cast_ms = 0.0;  // AMP FP16/BF16 -> FP32 cast, device-side
    double d2h_ms = 0.0;          // the device->host transfer ALONE
    double sync_ms = 0.0;         // HOST wall time blocked; overlaps the above
    uint64_t device_timed_batches = 0;   // batches contributing to the above
};

class BatchedEvaluator : public IGameEvaluator {
public:
    BatchedEvaluator(std::shared_ptr<SharedDunePolicyValueNetImpl> model,
                     int target_batch_size,
                     int timeout_ms,
                     torch::Device device,
                     std::shared_mutex* sync_mutex,
                     float logit_cap = 0.0f,
                     bool device_synchronize = true,
                     bool high_priority_stream = false)
        : model_(model), target_batch_size_(target_batch_size),
          timeout_ms_(timeout_ms), device_(device), sync_mutex_(sync_mutex),
          logit_cap_(logit_cap), device_synchronize_(device_synchronize),
          high_priority_stream_(high_priority_stream),
          stop_(false) {

        // Dynamically get the input layer dimension from the model's weights
        model_input_dim_ = model_->input_layer->weight.size(1);
        model_action_dim_ = model_->policy_head->weight.size(0);

        // Enable TF32 for Ada Lovelace (RTX 4080 Super) speedup
        if (device_.is_cuda()) {
            at::globalContext().setAllowTF32CuBLAS(true);
            at::globalContext().setAllowTF32CuDNN(true);
        }

        model_->eval(); // Defensive hygiene: ResBlocks use LayerNorm so this is a no-op, but protects future Dropout additions.
        runner_thread_ = std::thread(&BatchedEvaluator::Runner, this);
    }

    ~BatchedEvaluator() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stop_ = true;
        }
        cv_.notify_all();
        if (runner_thread_.joinable()) runner_thread_.join();
    }

    // Called by the actor threads
    EvalResult Evaluate(const std::vector<float>& obs) override {
        // Validate size in the caller thread (finding 6) so the runner never
        // over-reads/truncates and never faults on an unsupported request.
        CheckEvalObsSize(obs.size(), model_input_dim_);
        EvalResult result; // Stack allocated!
        std::atomic<bool> ready{false};

        {
            std::lock_guard<std::mutex> lock(mutex_);
            // WO-PERF-3: single now() serves both the existing first-request
            // stamp (semantics unchanged: assigned only when the queue was
            // empty) and this row's per-request enqueue stamp.
            auto now = std::chrono::steady_clock::now();
            if (requests_.empty()) {
                first_request_ts_ = now;
            }
            Request req;
            req.obs = &obs;
            req.result_dest = &result;
            req.ready_flag = &ready;
            req.wants_logits = true;
            req.group_id = next_group_id_.fetch_add(1, std::memory_order_relaxed);
            req.group_size = 1;
            req.is_group = false;
            req.enqueue_ts = now;
            requests_.push_back(req);
            single_row_calls_.fetch_add(1, std::memory_order_relaxed);

            // Only wake the Runner on the first arrival or when the batch is full
            if (requests_.size() == 1 || requests_.size() >= (size_t)target_batch_size_) {
                cv_.notify_one();
            }
        }

        // HYBRID WAIT: Spin briefly to catch ultra-fast GPU responses, then park.
        int spin_count = 0;
        while (!ready.load(std::memory_order_acquire)) {
            if (spin_count < 4000) {
                if (spin_count < 4000) {
#if defined(__x86_64__) || defined(_M_X64)
                    _mm_pause(); // Emits PAUSE instruction to prevent pipeline flushing
#else
                    std::this_thread::yield();
#endif
                    spin_count++;
                }
            } else {
                // Park safely (Zero CPU burn)
                std::unique_lock<std::mutex> park_lock(park_mutex_);
                park_cv_.wait(park_lock, [&] { return ready.load(std::memory_order_acquire); });
            }
        }

        return result; // Move semantics
    }

    // Production leaf path: values are returned without allocating or copying
    // the 2,391-action policy vector to host memory.
    EvalResult EvaluateValues(const std::vector<float>& obs) {
        CheckEvalObsSize(obs.size(), model_input_dim_);
        EvalResult result;
        std::atomic<bool> ready{false};
        {
            std::lock_guard<std::mutex> lock(mutex_);
            auto now = std::chrono::steady_clock::now();
            if (requests_.empty()) first_request_ts_ = now;
            Request req;
            req.obs = &obs;
            req.result_dest = &result;
            req.ready_flag = &ready;
            req.wants_logits = false;
            req.group_id = next_group_id_.fetch_add(1, std::memory_order_relaxed);
            req.group_size = 1;
            req.is_group = false;
            req.enqueue_ts = now;
            requests_.push_back(req);
            single_row_calls_.fetch_add(1, std::memory_order_relaxed);
            if (requests_.size() == 1 ||
                requests_.size() >= static_cast<size_t>(target_batch_size_)) {
                cv_.notify_one();
            }
        }
        int spin_count = 0;
        while (!ready.load(std::memory_order_acquire)) {
            if (spin_count++ < 4000) {
#if defined(__x86_64__) || defined(_M_X64)
                _mm_pause();
#else
                std::this_thread::yield();
#endif
            } else {
                std::unique_lock<std::mutex> park_lock(park_mutex_);
                park_cv_.wait(park_lock, [&] {
                    return ready.load(std::memory_order_acquire);
                });
            }
        }
        return result;
    }

    CompactEvalResult EvaluateCompact(
        const std::vector<float>& obs,
        const std::vector<Action>& legal_actions) {
        if (!device_.is_cuda() || device_synchronize_) {
            EvalResult full = Evaluate(obs);
            CompactEvalResult compact;
            compact.actions = legal_actions;
            std::vector<float> logits = full.logits;
            CenterAndCapLegalLogitsWithStats(logits, legal_actions, logit_cap_);
            double max_logit = -std::numeric_limits<double>::infinity();
            for (Action action : legal_actions) {
                max_logit = std::max(max_logit,
                                     static_cast<double>(logits[action]));
            }
            double total = 0.0;
            for (Action action : legal_actions) {
                total += std::exp(static_cast<double>(logits[action]) - max_logit);
            }
            compact.probabilities.reserve(legal_actions.size());
            for (Action action : legal_actions) {
                compact.probabilities.push_back(static_cast<float>(
                    std::exp(static_cast<double>(logits[action]) - max_logit) /
                    total));
            }
            compact.value = full.value;
            return compact;
        }
        CheckEvalObsSize(obs.size(), model_input_dim_);
        CompactEvalResult result;
        std::atomic<bool> ready{false};
        {
            std::lock_guard<std::mutex> lock(mutex_);
            auto now = std::chrono::steady_clock::now();
            if (requests_.empty()) first_request_ts_ = now;
            Request req;
            req.obs = &obs;
            req.ready_flag = &ready;
            req.compact_dest = &result;
            req.legal_actions = legal_actions;
            req.wants_logits = false;
            req.group_id = next_group_id_.fetch_add(1, std::memory_order_relaxed);
            req.group_size = 1;
            req.is_group = false;
            req.enqueue_ts = now;
            requests_.push_back(std::move(req));
            single_row_calls_.fetch_add(1, std::memory_order_relaxed);
            if (requests_.size() == 1 ||
                requests_.size() >= static_cast<size_t>(target_batch_size_)) {
                cv_.notify_one();
            }
        }
        int spin_count = 0;
        while (!ready.load(std::memory_order_acquire)) {
            if (spin_count++ < 4000) {
#if defined(__x86_64__) || defined(_M_X64)
                _mm_pause();
#else
                std::this_thread::yield();
#endif
            } else {
                std::unique_lock<std::mutex> park_lock(park_mutex_);
                park_cv_.wait(park_lock, [&] {
                    return ready.load(std::memory_order_acquire);
                });
            }
        }
        return result;
    }

    std::vector<EvalResult> EvaluateBatch(
        const std::vector<std::vector<float>>& observations) override {
        std::vector<EvalResult> results(observations.size());
        if (observations.empty()) return results;
        // A logical leaf group larger than the configured physical target used
        // to be admitted whole. That silently violated max-batch provenance and
        // became unsafe once the trainer joined the audit as a general
        // production consumer. Reject it in the class that owns the contract.
        if (observations.size() > static_cast<size_t>(target_batch_size_)) {
            SpielFatalError("BatchedEvaluator::EvaluateBatch group size " +
                            std::to_string(observations.size()) +
                            " exceeds configured target " +
                            std::to_string(target_batch_size_));
        }
        for (const auto& obs : observations) {
            CheckEvalObsSize(obs.size(), model_input_dim_);
        }

        auto ready = std::make_unique<std::atomic<bool>[]>(observations.size());
        {
            std::lock_guard<std::mutex> lock(mutex_);
            auto now = std::chrono::steady_clock::now();
            if (requests_.empty()) {
                first_request_ts_ = now;
            }
            // WO-PERF-3: one group id for the whole leaf group so the runner can
            // detect when its rows straddle two physical batches.
            const uint64_t gid = next_group_id_.fetch_add(1, std::memory_order_relaxed);
            const uint32_t gsize = static_cast<uint32_t>(observations.size());
            for (size_t i = 0; i < observations.size(); ++i) {
                ready[i].store(false, std::memory_order_relaxed);
                Request req;
                req.obs = &observations[i];
                req.result_dest = &results[i];
                req.ready_flag = &ready[i];
                req.group_id = gid;
                req.group_size = gsize;
                req.is_group = true;
                req.enqueue_ts = now;
                requests_.push_back(req);
            }
            group_calls_.fetch_add(1, std::memory_order_relaxed);
            group_rows_.fetch_add(gsize, std::memory_order_relaxed);
            // Submit all player observations atomically so the runner can form
            // a useful batch without four serial queue/wait cycles.
            cv_.notify_one();
        }

        auto all_ready = [&]() {
            for (size_t i = 0; i < observations.size(); ++i) {
                if (!ready[i].load(std::memory_order_acquire)) return false;
            }
            return true;
        };
        int spin_count = 0;
        while (!all_ready()) {
            if (spin_count < 4000) {
#if defined(__x86_64__) || defined(_M_X64)
                _mm_pause();
#else
                std::this_thread::yield();
#endif
                ++spin_count;
            } else {
                std::unique_lock<std::mutex> park_lock(park_mutex_);
                park_cv_.wait(park_lock, all_ready);
            }
        }
        return results;
    }

    EvaluatorStats GetStats() const override {
        uint64_t batches = total_batches_.load(std::memory_order_relaxed);
        uint64_t requests = total_requests_.load(std::memory_order_relaxed);
        uint64_t max_batch = max_batch_size_seen_.load(std::memory_order_relaxed);
        double avg_batch = batches > 0 ? static_cast<double>(requests) / batches : 0.0;
        EvaluatorStats s;
        s.batches = batches;
        s.requests = requests;
        s.max_batch_size = max_batch;
        s.avg_batch_size = avg_batch;
        return s;
    }

    std::vector<EvalResult> EvaluateBatchValues(
        const std::vector<std::vector<float>>& observations) {
        std::vector<EvalResult> results(observations.size());
        if (observations.empty()) return results;
        if (observations.size() > static_cast<size_t>(target_batch_size_)) {
            SpielFatalError("BatchedEvaluator::EvaluateBatchValues group exceeds target");
        }
        for (const auto& obs : observations) CheckEvalObsSize(obs.size(), model_input_dim_);
        auto ready = std::make_unique<std::atomic<bool>[]>(observations.size());
        {
            std::lock_guard<std::mutex> lock(mutex_);
            auto now = std::chrono::steady_clock::now();
            if (requests_.empty()) first_request_ts_ = now;
            const uint64_t gid = next_group_id_.fetch_add(1, std::memory_order_relaxed);
            const uint32_t gsize = static_cast<uint32_t>(observations.size());
            for (size_t i = 0; i < observations.size(); ++i) {
                ready[i].store(false, std::memory_order_relaxed);
                Request req;
                req.obs = &observations[i];
                req.result_dest = &results[i];
                req.ready_flag = &ready[i];
                req.wants_logits = false;
                req.group_id = gid;
                req.group_size = gsize;
                req.is_group = true;
                req.enqueue_ts = now;
                requests_.push_back(req);
            }
            group_calls_.fetch_add(1, std::memory_order_relaxed);
            group_rows_.fetch_add(gsize, std::memory_order_relaxed);
            cv_.notify_one();
        }
        auto all_ready = [&]() {
            for (size_t i = 0; i < observations.size(); ++i) {
                if (!ready[i].load(std::memory_order_acquire)) return false;
            }
            return true;
        };
        int spin_count = 0;
        while (!all_ready()) {
            if (spin_count++ < 4000) {
#if defined(__x86_64__) || defined(_M_X64)
                _mm_pause();
#else
                std::this_thread::yield();
#endif
            } else {
                std::unique_lock<std::mutex> park_lock(park_mutex_);
                park_cv_.wait(park_lock, all_ready);
            }
        }
        return results;
    }

    std::pair<std::vector<EvalResult>, CompactEvalResult>
    EvaluateBatchValuesWithCompactPrior(
        const std::vector<std::vector<float>>& observations,
        size_t prior_index, const std::vector<Action>& legal_actions) {
        std::vector<EvalResult> results(observations.size());
        CompactEvalResult compact;
        if (observations.empty() || prior_index >= observations.size()) {
            return {std::move(results), std::move(compact)};
        }
        for (const auto& obs : observations) CheckEvalObsSize(obs.size(), model_input_dim_);
        auto ready = std::make_unique<std::atomic<bool>[]>(observations.size());
        {
            std::lock_guard<std::mutex> lock(mutex_);
            auto now = std::chrono::steady_clock::now();
            if (requests_.empty()) first_request_ts_ = now;
            const uint64_t gid = next_group_id_.fetch_add(1, std::memory_order_relaxed);
            const uint32_t gsize = static_cast<uint32_t>(observations.size());
            for (size_t i = 0; i < observations.size(); ++i) {
                ready[i].store(false, std::memory_order_relaxed);
                Request req;
                req.obs = &observations[i];
                req.result_dest = &results[i];
                req.ready_flag = &ready[i];
                req.wants_logits = false;
                if (i == prior_index) {
                    req.compact_dest = &compact;
                    req.legal_actions = legal_actions;
                }
                req.group_id = gid;
                req.group_size = gsize;
                req.is_group = true;
                req.enqueue_ts = now;
                requests_.push_back(std::move(req));
            }
            group_calls_.fetch_add(1, std::memory_order_relaxed);
            group_rows_.fetch_add(gsize, std::memory_order_relaxed);
            cv_.notify_one();
        }
        auto all_ready = [&]() {
            for (size_t i = 0; i < observations.size(); ++i) {
                if (!ready[i].load(std::memory_order_acquire)) return false;
            }
            return true;
        };
        int spin_count = 0;
        while (!all_ready()) {
            if (spin_count++ < 4000) {
#if defined(__x86_64__) || defined(_M_X64)
                _mm_pause();
#else
                std::this_thread::yield();
#endif
            } else {
                std::unique_lock<std::mutex> park_lock(park_mutex_);
                park_cv_.wait(park_lock, all_ready);
            }
        }
        return {std::move(results), std::move(compact)};
    }

    // R2 (WO-3 review F1): arms the per-batch bookkeeping below (batch-size
    // record, per-row queue-wait accounting, split-leaf-group maps). Call it
    // BEFORE the workload starts; without it those sections are skipped and
    // GetBatcherTelemetry() reports zeros for the affected fields while the
    // always-on counters (rows, batches, class counts, timeout flushes) stay
    // live. Existing consumers that never opt in are unaffected by design.
    void EnableBatcherTelemetry() {
        batcher_telemetry_enabled_.store(true, std::memory_order_relaxed);
    }

    // WO-PERF-3: rich, opt-in telemetry for the Phase-2 "Batcher" list — the
    // detailed sections require EnableBatcherTelemetry() first. Safe to
    // call from another thread while the runner is live (all reads are atomic or
    // guarded by telemetry_mutex_); intended to be read once after the workload
    // drains. Percentiles are computed from the recorded per-batch sizes.
    BatcherTelemetry GetBatcherTelemetry() const {
        BatcherTelemetry t;
        t.submitted_rows = total_requests_.load(std::memory_order_relaxed);
        t.physical_batches = total_batches_.load(std::memory_order_relaxed);
        t.max_batch_size = max_batch_size_seen_.load(std::memory_order_relaxed);
        t.target_batch_size = target_batch_size_;
        t.timeout_ms = timeout_ms_;
        t.timeout_flush_batches = timeout_flush_batches_.load(std::memory_order_relaxed);
        t.single_row_calls = single_row_calls_.load(std::memory_order_relaxed);
        t.group_calls = group_calls_.load(std::memory_order_relaxed);
        t.group_rows = group_rows_.load(std::memory_order_relaxed);
        t.leaf_groups_split = leaf_groups_split_.load(std::memory_order_relaxed);
        // WO-PERF-TIMING-BATCH: device-side split, microseconds -> milliseconds.
        t.h2d_ms = h2d_us_.load(std::memory_order_relaxed) / 1000.0;
        t.forward_ms = forward_us_.load(std::memory_order_relaxed) / 1000.0;
        t.output_cast_ms = output_cast_us_.load(std::memory_order_relaxed) / 1000.0;
        t.d2h_ms = d2h_us_.load(std::memory_order_relaxed) / 1000.0;
        t.sync_ms = sync_us_.load(std::memory_order_relaxed) / 1000.0;
        t.device_timed_batches =
            device_timed_batches_.load(std::memory_order_relaxed);
        // Lightweight production telemetry: these counters are always live,
        // so mean batch size remains available even when detailed per-batch
        // telemetry and CUDA event timing are disabled.
        if (t.physical_batches > 0) {
            t.mean_batch_size =
                static_cast<double>(t.submitted_rows) /
                static_cast<double>(t.physical_batches);
        }

        std::vector<uint32_t> sizes;
        {
            std::lock_guard<std::mutex> lk(telemetry_mutex_);
            sizes = physical_batch_sizes_;
        }
        if (!sizes.empty()) {
            std::sort(sizes.begin(), sizes.end());
            double sum = 0.0;
            for (uint32_t s : sizes) sum += s;
            t.mean_batch_size = sum / sizes.size();
            auto pct = [&sizes](double p) -> uint64_t {
                size_t idx = static_cast<size_t>(std::round(p * (sizes.size() - 1)));
                return static_cast<uint64_t>(sizes[idx]);
            };
            t.p50_batch_size = pct(0.50);
            t.p95_batch_size = pct(0.95);
        }
        t.target_occupancy =
            target_batch_size_ > 0 ? t.mean_batch_size / target_batch_size_ : 0.0;

        t.single_wait_n = single_wait_n_.load(std::memory_order_relaxed);
        t.group_wait_n = group_wait_n_.load(std::memory_order_relaxed);
        t.single_wait_ms_mean =
            t.single_wait_n > 0
                ? (single_wait_ns_sum_.load(std::memory_order_relaxed) / 1e6) / t.single_wait_n
                : 0.0;
        t.group_wait_ms_mean =
            t.group_wait_n > 0
                ? (group_wait_ns_sum_.load(std::memory_order_relaxed) / 1e6) / t.group_wait_n
                : 0.0;
        t.single_wait_ms_max = single_wait_ns_max_.load(std::memory_order_relaxed) / 1e6;
        t.group_wait_ms_max = group_wait_ns_max_.load(std::memory_order_relaxed) / 1e6;
        return t;
    }

private:
    struct Request {
        const std::vector<float>* obs = nullptr;
        EvalResult* result_dest = nullptr;
        std::atomic<bool>* ready_flag = nullptr;
        bool wants_logits = true;
        CompactEvalResult* compact_dest = nullptr;
        std::vector<Action> legal_actions;
        // WO-PERF-3 telemetry tags (no effect on dispatch). group_id is unique
        // per logical call; a one-row Evaluate() is its own group. is_group is
        // true only for EvaluateBatch() (searched-leaf) rows. enqueue_ts stamps
        // when the row entered the queue so the runner can measure queue wait.
        uint64_t group_id = 0;
        uint32_t group_size = 1;
        bool is_group = false;
        std::chrono::steady_clock::time_point enqueue_ts{};
    };

    std::mutex park_mutex_;
    std::condition_variable park_cv_;

    std::shared_ptr<SharedDunePolicyValueNetImpl> model_;
    int target_batch_size_;
    int timeout_ms_;
    torch::Device device_;
    std::shared_mutex* sync_mutex_;
    float logit_cap_;
    bool device_synchronize_;
    bool high_priority_stream_;
    int64_t model_input_dim_;
    int64_t model_action_dim_;

    std::mutex mutex_;
    std::condition_variable cv_;
    std::deque<Request> requests_;
    std::chrono::steady_clock::time_point first_request_ts_;

    bool stop_;
    std::thread runner_thread_;
    std::atomic<uint64_t> total_batches_{0};
    std::atomic<uint64_t> total_requests_{0};
    std::atomic<uint64_t> max_batch_size_seen_{0};

    // --- WO-PERF-3 telemetry (additive; does not influence dispatch) ---------
    std::atomic<uint64_t> next_group_id_{1};
    std::atomic<uint64_t> single_row_calls_{0};   // Evaluate() invocations
    std::atomic<uint64_t> group_calls_{0};        // EvaluateBatch() invocations
    std::atomic<uint64_t> group_rows_{0};         // rows across EvaluateBatch() calls
    std::atomic<uint64_t> timeout_flush_batches_{0};
    std::atomic<uint64_t> leaf_groups_split_{0};
    std::atomic<uint64_t> single_wait_ns_sum_{0};
    std::atomic<uint64_t> single_wait_ns_max_{0};
    std::atomic<uint64_t> single_wait_n_{0};
    std::atomic<uint64_t> group_wait_ns_sum_{0};
    std::atomic<uint64_t> group_wait_ns_max_{0};
    std::atomic<uint64_t> group_wait_n_{0};
    // R2 (WO-3 review F1): per-batch bookkeeping is OPT-IN. Consumers that
    // never call EnableBatcherTelemetry() pay one relaxed load per physical
    // batch and hold no growing state.
    std::atomic<bool> batcher_telemetry_enabled_{false};
    mutable std::mutex telemetry_mutex_;          // guards physical_batch_sizes_
    std::vector<uint32_t> physical_batch_sizes_;  // one entry per physical batch

    // WO-PERF-TIMING-BATCH: cumulative device-side microseconds. Integer
    // microseconds rather than double milliseconds so the accumulation is a
    // plain atomic add with no read-modify-write race and no float ordering
    // question across dispatch threads.
    std::atomic<uint64_t> h2d_us_{0};
    std::atomic<uint64_t> forward_us_{0};
    std::atomic<uint64_t> output_cast_us_{0};
    std::atomic<uint64_t> d2h_us_{0};
    std::atomic<uint64_t> sync_us_{0};
    std::atomic<uint64_t> device_timed_batches_{0};
    // Runner-thread-only maps for split-leaf-group detection. A leaf group's
    // rows are contiguous in the deque, so within one physical batch a group is
    // a single run; a group counted in a second physical batch is a split.
    std::unordered_map<uint64_t, int> group_batches_seen_;  // gid -> # batches
    std::unordered_map<uint64_t, int> group_rows_left_;     // gid -> rows undispatched

    // Production CUDA path: two reusable slots let the runner assemble the
    // next CPU batch while the previous slot is executing. Completion waits
    // on one CUDA event for this stream, never on the whole device.
    struct AsyncBatchSlot {
        std::vector<Request> batch;
        torch::Tensor host_obs;
        torch::Tensor device_obs;
        torch::Tensor device_logits;
        torch::Tensor device_values;
        torch::Tensor host_logits;
        torch::Tensor host_values;
        std::unique_ptr<c10::cuda::CUDAStream> stream;
        std::unique_ptr<c10::Event> complete;
        bool wants_logits = false;
        int compact_max_legal = 0;
        std::vector<size_t> compact_request_indices;
        torch::Tensor compact_device_ids;
        torch::Tensor compact_device_mask;
        torch::Tensor compact_device_rows;
        torch::Tensor compact_host_probs;
        bool pending = false;
    };

    void RunnerAsync() {
        torch::InferenceMode inference_guard;
        const c10::DeviceIndex device_index =
            device_.has_index() ? device_.index() : c10::cuda::current_device();
        std::array<AsyncBatchSlot, 2> slots;
        for (AsyncBatchSlot& slot : slots) {
            slot.stream = std::make_unique<c10::cuda::CUDAStream>(
                c10::cuda::getStreamFromPool(high_priority_stream_,
                                             device_index));
        }
        int pending_slot = -1;
        int next_slot = 0;

        auto take_batch = [&](bool* timeout_flush) {
            std::vector<Request> batch;
            *timeout_flush = false;
            {
                std::unique_lock<std::mutex> lock(mutex_);
                cv_.wait(lock, [this]() { return stop_ || !requests_.empty(); });
                if (stop_ && requests_.empty()) return batch;
                const auto deadline =
                    first_request_ts_ + std::chrono::milliseconds(timeout_ms_);
                cv_.wait_until(lock, deadline, [this, deadline]() {
                    return stop_ || requests_.size() >=
                                      static_cast<size_t>(target_batch_size_) ||
                           std::chrono::steady_clock::now() >= deadline;
                });
                if (stop_ && requests_.empty()) return batch;
                *timeout_flush = !stop_ &&
                                 requests_.size() <
                                     static_cast<size_t>(target_batch_size_);
                size_t actual_batch =
                    std::min(static_cast<size_t>(target_batch_size_),
                             requests_.size());
                if (actual_batch > 0 && actual_batch < requests_.size() &&
                    requests_[actual_batch - 1].is_group &&
                    requests_[actual_batch].is_group &&
                    requests_[actual_batch - 1].group_id ==
                        requests_[actual_batch].group_id) {
                    const uint64_t boundary_gid =
                        requests_[actual_batch].group_id;
                    size_t group_start = actual_batch;
                    while (group_start > 0 &&
                           requests_[group_start - 1].group_id == boundary_gid) {
                        --group_start;
                    }
                    if (group_start > 0) {
                        actual_batch = group_start;
                    } else {
                        while (actual_batch < requests_.size() &&
                               requests_[actual_batch].group_id ==
                                   boundary_gid) {
                            ++actual_batch;
                        }
                    }
                }
                batch.reserve(actual_batch);
                for (size_t i = 0; i < actual_batch; ++i) {
                    batch.push_back(std::move(requests_.front()));
                    requests_.pop_front();
                }
                if (!requests_.empty()) {
                    first_request_ts_ = std::chrono::steady_clock::now();
                }
            }
            return batch;
        };

        auto record_batch_telemetry = [&](const std::vector<Request>& batch,
                                          bool timeout_flush) {
            const size_t batch_size = batch.size();
            total_batches_.fetch_add(1, std::memory_order_relaxed);
            total_requests_.fetch_add(static_cast<uint64_t>(batch_size),
                                       std::memory_order_relaxed);
            uint64_t observed_max =
                max_batch_size_seen_.load(std::memory_order_relaxed);
            while (batch_size > observed_max &&
                   !max_batch_size_seen_.compare_exchange_weak(
                       observed_max, static_cast<uint64_t>(batch_size),
                       std::memory_order_relaxed,
                       std::memory_order_relaxed)) {}
            if (timeout_flush) {
                timeout_flush_batches_.fetch_add(1,
                                                  std::memory_order_relaxed);
            }
            if (!batcher_telemetry_enabled_.load(std::memory_order_relaxed)) {
                return;
            }
            {
                std::lock_guard<std::mutex> lk(telemetry_mutex_);
                physical_batch_sizes_.push_back(
                    static_cast<uint32_t>(batch_size));
            }
            const auto dispatch_ts = std::chrono::steady_clock::now();
            std::unordered_map<uint64_t, std::pair<int, uint32_t>> groups_here;
            for (const Request& request : batch) {
                const uint64_t wait_ns = static_cast<uint64_t>(std::max<int64_t>(
                    0, std::chrono::duration_cast<std::chrono::nanoseconds>(
                           dispatch_ts - request.enqueue_ts)
                           .count()));
                if (request.is_group) {
                    group_wait_ns_sum_.fetch_add(wait_ns,
                                                 std::memory_order_relaxed);
                    group_wait_n_.fetch_add(1, std::memory_order_relaxed);
                    uint64_t max_wait =
                        group_wait_ns_max_.load(std::memory_order_relaxed);
                    while (wait_ns > max_wait &&
                           !group_wait_ns_max_.compare_exchange_weak(
                               max_wait, wait_ns, std::memory_order_relaxed,
                               std::memory_order_relaxed)) {}
                    auto& entry = groups_here[request.group_id];
                    entry.first += 1;
                    entry.second = request.group_size;
                } else {
                    single_wait_ns_sum_.fetch_add(wait_ns,
                                                  std::memory_order_relaxed);
                    single_wait_n_.fetch_add(1, std::memory_order_relaxed);
                    uint64_t max_wait =
                        single_wait_ns_max_.load(std::memory_order_relaxed);
                    while (wait_ns > max_wait &&
                           !single_wait_ns_max_.compare_exchange_weak(
                               max_wait, wait_ns, std::memory_order_relaxed,
                               std::memory_order_relaxed)) {}
                }
            }
            for (const auto& item : groups_here) {
                const uint64_t group_id = item.first;
                const int rows_here = item.second.first;
                const uint32_t group_size = item.second.second;
                const int seen = ++group_batches_seen_[group_id];
                if (seen == 2) {
                    leaf_groups_split_.fetch_add(1,
                                                  std::memory_order_relaxed);
                }
                int& rows_left = group_rows_left_
                    .try_emplace(group_id, static_cast<int>(group_size))
                    .first->second;
                rows_left -= rows_here;
                if (rows_left <= 0) {
                    group_batches_seen_.erase(group_id);
                    group_rows_left_.erase(group_id);
                }
            }
        };

        auto finish_slot = [&](AsyncBatchSlot& slot) {
            if (!slot.pending) return;
            slot.complete->synchronize();
            const float* values = slot.host_values.data_ptr<float>();
            const size_t action_dim = static_cast<size_t>(model_action_dim_);
            dune_hotspot::ScopedTimer hotspot_timer(
                dune_hotspot::Kind::kBatcherResultDistribution);
            for (size_t i = 0; i < slot.batch.size(); ++i) {
                Request& request = slot.batch[i];
                if (request.result_dest != nullptr) {
                    request.result_dest->value = values[i];
                    if (request.wants_logits) {
                        const float* logits = slot.host_logits.data_ptr<float>();
                        request.result_dest->logits.assign(
                            logits + i * action_dim,
                            logits + (i + 1) * action_dim);
                    }
                }
                if (request.compact_dest != nullptr) {
                    const auto it = std::find(slot.compact_request_indices.begin(),
                                              slot.compact_request_indices.end(), i);
                    const size_t compact_index = static_cast<size_t>(
                        std::distance(slot.compact_request_indices.begin(), it));
                    const float* probs = slot.compact_host_probs.data_ptr<float>();
                    request.compact_dest->actions = request.legal_actions;
                    request.compact_dest->probabilities.assign(
                        probs + compact_index * slot.compact_max_legal,
                        probs + compact_index * slot.compact_max_legal +
                            request.legal_actions.size());
                    request.compact_dest->value = values[i];
                }
                slot.batch[i].ready_flag->store(true,
                                                std::memory_order_release);
            }
            {
                std::lock_guard<std::mutex> park_lock(park_mutex_);
            }
            park_cv_.notify_all();
            slot.batch.clear();
            slot.pending = false;
        };

        auto launch_slot = [&](AsyncBatchSlot& slot) {
            const size_t batch_size = slot.batch.size();
            slot.wants_logits = false;
            slot.compact_request_indices.clear();
            slot.compact_max_legal = 0;
            for (size_t i = 0; i < batch_size; ++i) {
                if (slot.batch[i].wants_logits) slot.wants_logits = true;
                if (slot.batch[i].compact_dest != nullptr) {
                    slot.compact_request_indices.push_back(i);
                    slot.compact_max_legal = std::max(
                        slot.compact_max_legal,
                        static_cast<int>(slot.batch[i].legal_actions.size()));
                }
            }
            if (!slot.device_obs.defined()) {
                auto host_options = torch::TensorOptions()
                    .dtype(torch::kFloat32).device(torch::kCPU)
                    .pinned_memory(true);
                auto device_options = torch::TensorOptions()
                    .dtype(torch::kFloat32).device(device_);
                slot.host_obs = torch::empty(
                    {target_batch_size_, model_input_dim_}, host_options);
                slot.device_obs = torch::empty(
                    {target_batch_size_, model_input_dim_}, device_options);
                slot.device_values = torch::empty(
                    {target_batch_size_}, device_options);
                slot.host_values = torch::empty(
                    {target_batch_size_}, host_options);
                slot.complete = std::make_unique<c10::Event>(
                    c10::DeviceType::CUDA, c10::EventFlag::BACKEND_DEFAULT);
            }
            if (slot.wants_logits && !slot.device_logits.defined()) {
                auto host_options = torch::TensorOptions()
                    .dtype(torch::kFloat32).device(torch::kCPU)
                    .pinned_memory(true);
                auto device_options = torch::TensorOptions()
                    .dtype(torch::kFloat32).device(device_);
                slot.device_logits = torch::empty(
                    {target_batch_size_, model_action_dim_}, device_options);
                slot.host_logits = torch::empty(
                    {target_batch_size_, model_action_dim_}, host_options);
            }
            {
                c10::cuda::CUDAStreamGuard stream_guard(*slot.stream);
                auto host_obs = slot.host_obs.narrow(0, 0, batch_size);
                auto device_obs = slot.device_obs.narrow(0, 0, batch_size);
                device_obs.copy_(host_obs, /*non_blocking=*/true);
                SharedDunePolicyValueNetImpl::ModelOutputs outputs;
                {
                    std::shared_lock<std::shared_mutex> lock(*sync_mutex_);
                    AutocastGuard autocast_guard(c10::DeviceType::CUDA, true);
                    outputs = model_->forward(device_obs);
                }
                auto device_values = slot.device_values.narrow(
                    0, 0, batch_size);
                device_values.copy_(outputs.values.reshape(
                                        {static_cast<int64_t>(batch_size)}),
                                    /*non_blocking=*/true);
                if (slot.wants_logits) {
                    auto device_logits = slot.device_logits.narrow(
                        0, 0, batch_size);
                    device_logits.copy_(outputs.logits,
                                        /*non_blocking=*/true);
                    slot.host_logits.narrow(0, 0, batch_size).copy_(
                        device_logits, /*non_blocking=*/true);
                }
                if (!slot.compact_request_indices.empty()) {
                    const int compact_count =
                        static_cast<int>(slot.compact_request_indices.size());
                    torch::Tensor ids_cpu = torch::zeros(
                        {compact_count, slot.compact_max_legal},
                        torch::TensorOptions().dtype(torch::kInt64));
                    torch::Tensor mask_cpu = torch::zeros(
                        {compact_count, slot.compact_max_legal},
                        torch::TensorOptions().dtype(torch::kBool));
                    int64_t* ids = ids_cpu.data_ptr<int64_t>();
                    bool* mask = mask_cpu.data_ptr<bool>();
                    for (int k = 0; k < compact_count; ++k) {
                        const Request& request = slot.batch[
                            slot.compact_request_indices[k]];
                        for (size_t j = 0; j < request.legal_actions.size(); ++j) {
                            ids[k * slot.compact_max_legal + j] =
                                request.legal_actions[j];
                            mask[k * slot.compact_max_legal + j] = true;
                        }
                    }
                    slot.compact_device_ids = ids_cpu.to(
                        device_, /*non_blocking=*/true);
                    slot.compact_device_mask = mask_cpu.to(
                        device_, /*non_blocking=*/true);
                    std::vector<int64_t> compact_row_ids;
                    compact_row_ids.reserve(compact_count);
                    for (size_t row_index : slot.compact_request_indices) {
                        compact_row_ids.push_back(static_cast<int64_t>(row_index));
                    }
                    slot.compact_device_rows = torch::tensor(
                        compact_row_ids,
                        torch::TensorOptions().dtype(torch::kInt64))
                        .to(device_, /*non_blocking=*/true);
                    auto compact_logits = outputs.logits.index_select(
                        0, slot.compact_device_rows)
                        .gather(1, slot.compact_device_ids)
                        .to(torch::kFloat32);
                    const torch::Tensor legal =
                        slot.compact_device_mask.to(torch::kFloat32);
                    const torch::Tensor legal_count =
                        legal.sum(1, true).clamp_min(1.0);
                    compact_logits = compact_logits -
                        (compact_logits * legal).sum(1, true) / legal_count;
                    if (logit_cap_ > 0.0f) {
                        compact_logits = logit_cap_ *
                            torch::tanh(compact_logits / logit_cap_);
                    }
                    auto compact_probs = torch::softmax(
                        compact_logits.masked_fill(
                            slot.compact_device_mask.logical_not(), -1e9f),
                        1);
                    slot.compact_host_probs = compact_probs.to(
                        torch::kCPU, /*non_blocking=*/true);
                }
                slot.host_values.narrow(0, 0, batch_size).copy_(
                    device_values, /*non_blocking=*/true);
                slot.complete->record(*slot.stream);
            }
            slot.pending = true;
        };

        for (;;) {
            bool timeout_flush = false;
            std::vector<Request> batch = take_batch(&timeout_flush);
            if (batch.empty()) break;
            const int slot_index = next_slot;
            AsyncBatchSlot& slot = slots[slot_index];
            if (slot.pending) finish_slot(slot);
            slot.batch = std::move(batch);
            record_batch_telemetry(slot.batch, timeout_flush);
            float* destination = slot.host_obs.defined()
                ? slot.host_obs.data_ptr<float>()
                : nullptr;
            if (destination == nullptr) {
                auto host_options = torch::TensorOptions()
                    .dtype(torch::kFloat32).device(torch::kCPU)
                    .pinned_memory(true);
                slot.host_obs = torch::empty(
                    {target_batch_size_, model_input_dim_}, host_options);
                destination = slot.host_obs.data_ptr<float>();
            }
            for (size_t i = 0; i < slot.batch.size(); ++i) {
                const int64_t input_size =
                    static_cast<int64_t>(slot.batch[i].obs->size());
                const int64_t copy_size =
                    std::min<int64_t>(model_input_dim_, input_size);
                std::memcpy(destination + i * model_input_dim_,
                            slot.batch[i].obs->data(),
                            copy_size * sizeof(float));
                if (copy_size < model_input_dim_) {
                    std::memset(destination + i * model_input_dim_ + copy_size,
                                0,
                                (model_input_dim_ - copy_size) * sizeof(float));
                }
            }
            // Complete the prior stream before launching the next shared-model
            // forward. The two slots still overlap CPU batch assembly with the
            // prior GPU batch, while the event wait protects host outputs and
            // avoids whole-device synchronization. Concurrent forwards on the
            // shared LibTorch module are deliberately not enabled.
            if (pending_slot >= 0) finish_slot(slots[pending_slot]);
            launch_slot(slot);
            bool queue_empty = false;
            {
                std::lock_guard<std::mutex> queue_lock(mutex_);
                queue_empty = requests_.empty();
            }
            if (queue_empty) {
                // Every worker may be blocked on this batch. Publish its
                // event before returning to an empty-queue wait, otherwise
                // the runner and all actors wait on one another forever.
                finish_slot(slot);
                pending_slot = -1;
            } else {
                pending_slot = slot_index;
            }
            next_slot = 1 - next_slot;
        }
        if (pending_slot >= 0) finish_slot(slots[pending_slot]);
    }

    void Runner() {
        if (device_.is_cuda() && !device_synchronize_) {
            RunnerAsync();
            return;
        }
        torch::InferenceMode inference_guard;
        torch::Tensor pinned_stacked_obs;
        bool pinned_allocated = false;

        while (true) {
            std::vector<Request> batch;
            bool timeout_flush = false;  // WO-PERF-3: dispatched under-target on deadline

            {
                std::unique_lock<std::mutex> lock(mutex_);

                // 1. Wait until we have at least one request or we are shutting down
                cv_.wait(lock, [this]() { return stop_ || !requests_.empty(); });
                if (stop_ && requests_.empty()) break;

                // 2. We have requests. Calculate the absolute timeout deadline
                auto deadline = first_request_ts_ + std::chrono::milliseconds(timeout_ms_);

                // 3. Wait until the batch is full OR the timeout expires (Deadlock fix!)
                cv_.wait_until(lock, deadline, [this, deadline]() {
                    return stop_ || requests_.size() >= (size_t)target_batch_size_ ||
                           std::chrono::steady_clock::now() >= deadline;
                });

                if (stop_ && requests_.empty()) break;

                // WO-PERF-3: a batch scooped with fewer than target rows (and not
                // during shutdown) was released by the deadline, not by filling.
                timeout_flush = !stop_ &&
                                requests_.size() < (size_t)target_batch_size_;

                // 4. Scoop up the batch. EvaluateBatch() inserts every logical
                // leaf group atomically and contiguously. Do not let the target
                // boundary cut one of those groups across two physical
                // forwards: besides wasting a second launch, batch-size-
                // dependent kernels could otherwise give the four players in
                // one leaf slightly different numerical contexts. Back off to
                // the start of the boundary group. (A group larger than the
                // configured target is rejected by EvaluateBatch() before it
                // reaches this queue.)
                size_t actual_batch =
                    std::min((size_t)target_batch_size_, requests_.size());
                if (actual_batch > 0 && actual_batch < requests_.size() &&
                    requests_[actual_batch - 1].is_group &&
                    requests_[actual_batch].is_group &&
                    requests_[actual_batch - 1].group_id ==
                        requests_[actual_batch].group_id) {
                    const uint64_t boundary_gid = requests_[actual_batch].group_id;
                    size_t group_start = actual_batch;
                    while (group_start > 0 &&
                           requests_[group_start - 1].group_id == boundary_gid) {
                        --group_start;
                    }
                    if (group_start > 0) {
                        actual_batch = group_start;
                    } else {
                        while (actual_batch < requests_.size() &&
                               requests_[actual_batch].group_id == boundary_gid) {
                            ++actual_batch;
                        }
                    }
                }
                batch.reserve(actual_batch);
                for (size_t i = 0; i < actual_batch; ++i) {
                    batch.push_back(std::move(requests_.front()));
                    requests_.pop_front();
                }

                // Reset the timer if there are leftovers
                if (!requests_.empty()) {
                    first_request_ts_ = std::chrono::steady_clock::now();
                }
            }

            if (batch.empty()) continue;

            size_t batch_size = batch.size();
            total_batches_.fetch_add(1, std::memory_order_relaxed);
            total_requests_.fetch_add(static_cast<uint64_t>(batch_size), std::memory_order_relaxed);
            uint64_t observed_max = max_batch_size_seen_.load(std::memory_order_relaxed);
            while (batch_size > observed_max &&
                   !max_batch_size_seen_.compare_exchange_weak(
                       observed_max, static_cast<uint64_t>(batch_size),
                       std::memory_order_relaxed, std::memory_order_relaxed)) {
            }

            // --- WO-PERF-3 telemetry (behaviour-neutral bookkeeping) -----------
            // Runner is the sole writer of the split-detection maps, so they need
            // no lock. Everything else is atomic or guarded by telemetry_mutex_.
            if (timeout_flush) {
                timeout_flush_batches_.fetch_add(1, std::memory_order_relaxed);
            }
            // R2 (WO-3 review F1): the per-batch size record, per-row
            // queue-wait accounting, and split-group maps below run ONLY for a
            // caller that opted in via EnableBatcherTelemetry(). The teacher,
            // calibration, and fidelity-gate consumers never opt in, so they
            // hold no growing state and do no per-row telemetry work.
            if (batcher_telemetry_enabled_.load(std::memory_order_relaxed)) {
            {
                std::lock_guard<std::mutex> lk(telemetry_mutex_);
                physical_batch_sizes_.push_back(static_cast<uint32_t>(batch_size));
            }
            // Per-row queue wait, split by request class.
            const auto dispatch_ts = std::chrono::steady_clock::now();
            // Distinct leaf groups present in THIS physical batch: rows counted
            // and group_size captured for split accounting.
            std::unordered_map<uint64_t, std::pair<int, uint32_t>> groups_here;
            for (const Request& r : batch) {
                const uint64_t wait_ns = static_cast<uint64_t>(std::max<int64_t>(
                    0, std::chrono::duration_cast<std::chrono::nanoseconds>(
                           dispatch_ts - r.enqueue_ts)
                           .count()));
                if (r.is_group) {
                    group_wait_ns_sum_.fetch_add(wait_ns, std::memory_order_relaxed);
                    group_wait_n_.fetch_add(1, std::memory_order_relaxed);
                    uint64_t gmax = group_wait_ns_max_.load(std::memory_order_relaxed);
                    while (wait_ns > gmax &&
                           !group_wait_ns_max_.compare_exchange_weak(
                               gmax, wait_ns, std::memory_order_relaxed,
                               std::memory_order_relaxed)) {}
                    auto& e = groups_here[r.group_id];
                    e.first += 1;
                    e.second = r.group_size;
                } else {
                    single_wait_ns_sum_.fetch_add(wait_ns, std::memory_order_relaxed);
                    single_wait_n_.fetch_add(1, std::memory_order_relaxed);
                    uint64_t smax = single_wait_ns_max_.load(std::memory_order_relaxed);
                    while (wait_ns > smax &&
                           !single_wait_ns_max_.compare_exchange_weak(
                               smax, wait_ns, std::memory_order_relaxed,
                               std::memory_order_relaxed)) {}
                }
            }
            for (const auto& kv : groups_here) {
                const uint64_t gid = kv.first;
                const int rows_here = kv.second.first;
                const uint32_t gsize = kv.second.second;
                const int seen = ++group_batches_seen_[gid];
                if (seen == 2) {
                    // First time this group appears in a SECOND physical batch.
                    leaf_groups_split_.fetch_add(1, std::memory_order_relaxed);
                }
                int& left =
                    group_rows_left_.try_emplace(gid, static_cast<int>(gsize)).first->second;
                left -= rows_here;
                if (left <= 0) {
                    group_batches_seen_.erase(gid);
                    group_rows_left_.erase(gid);
                }
            }
            }  // if (batcher_telemetry_enabled_) — R2 opt-in gate

            // A. PINNED MEMORY: Allocate exactly ONCE outside the hot loop based on model_input_dim_
            if (!pinned_allocated) {
                if (device_.is_cuda()) {
                    auto options = torch::TensorOptions().dtype(torch::kFloat32).pinned_memory(true);
                    pinned_stacked_obs = torch::empty({(long)target_batch_size_, (long)model_input_dim_}, options);
                } else {
                    pinned_stacked_obs = torch::empty({(long)target_batch_size_, (long)model_input_dim_}, torch::kFloat32);
                }
                pinned_allocated = true;
            }

            float* dest_ptr = pinned_stacked_obs.data_ptr<float>();
            for (size_t i = 0; i < batch_size; ++i) {
                // Per-request copy length (finding 6): sizing from batch[0] would
                // over-read a shorter request's vector or truncate a longer one
                // when the batch mixes the two engine observation layouts.
                const int64_t this_obs_size = static_cast<int64_t>(batch[i].obs->size());
                const int64_t copy_size = std::min<int64_t>(model_input_dim_, this_obs_size);
                std::memcpy(dest_ptr + i * model_input_dim_, batch[i].obs->data(), copy_size * sizeof(float));
                if (copy_size < model_input_dim_) {
                    std::memset(dest_ptr + i * model_input_dim_ + copy_size, 0, (model_input_dim_ - copy_size) * sizeof(float));
                }
            }

            // WO-PERF-TIMING-BATCH: arm the device-side split for this batch.
            // Opt-in with the rest of the batcher telemetry, CUDA only, so the
            // default path creates no events and reads no clock.
            const bool time_device_ =
                device_.is_cuda() &&
                batcher_telemetry_enabled_.load(std::memory_order_relaxed);
            // thread_local so the events are created once per dispatch thread
            // and reused, rather than created and destroyed per physical batch.
            // Reuse is safe: every pair is read below, before the next batch on
            // this thread can re-record it.
            thread_local std::vector<c10::Event> dev_ev;
            if (time_device_ && dev_ev.empty()) {
                for (int e = 0; e < 8; ++e) {
                    dev_ev.emplace_back(c10::DeviceType::CUDA,
                                        c10::EventFlag::BACKEND_DEFAULT);
                }
            }
            // Constructed ONLY when timing is armed. An unconditional
            // VirtualGuardImpl on the default path would touch the CUDA guard
            // registry on every physical batch of every run, including CPU runs
            // and runs with telemetry off.
            std::optional<c10::impl::VirtualGuardImpl> dev_impl;
            if (time_device_) {
                dev_impl.emplace(c10::DeviceType::CUDA);
                dev_ev[0].record(dev_impl->getStream(device_));
            }

            // Non-blocking H2D transfer using .slice() for partial batches
            torch::Tensor device_obs = device_.is_cuda()
                ? pinned_stacked_obs.slice(0, 0, batch_size).to(device_, /*non_blocking=*/true)
                : pinned_stacked_obs.slice(0, 0, batch_size);

            if (time_device_) dev_ev[1].record(dev_impl->getStream(device_));

            // B. FORWARD PASS WITH AMP (FP16/TF32) AND WRITE PROTECTION
            torch::Tensor pred_logits, pred_values;
            {
                std::shared_lock<std::shared_mutex> lock(*sync_mutex_);
                if (device_.is_cuda()) {
                    AutocastGuard autocast_guard(c10::DeviceType::CUDA, true);
                    auto outputs = model_->forward(device_obs);
                    pred_logits = outputs.logits;
                    pred_values = outputs.values;
                } else {
                    auto outputs = model_->forward(device_obs);
                    pred_logits = outputs.logits;
                    pred_values = outputs.values;
                }
            }
            if (time_device_) dev_ev[2].record(dev_impl->getStream(device_));

            // Return raw logits. The tanh soft-cap must be applied after
            // legal-action centering, which requires the State's legal actions.

            // C. NON-BLOCKING D2H TRANSFER (cast AMP FP16 outputs back to Float32)
            if (device_.is_cuda()) {
                pred_logits = pred_logits.to(torch::kFloat32);
                pred_values = pred_values.to(torch::kFloat32);
                // Boundary AFTER the cast: without it, "d2h" silently included
                // the AMP->FP32 conversion, which is device compute, not a
                // transfer.
                if (time_device_) dev_ev[3].record(dev_impl->getStream(device_));
                std::chrono::steady_clock::time_point sync_t0, sync_t1;
                if (device_synchronize_) {
                    pred_logits = pred_logits.to(torch::kCPU, /*non_blocking=*/true).contiguous();
                    pred_values = pred_values.to(torch::kCPU, /*non_blocking=*/true).contiguous();
                    if (time_device_) dev_ev[4].record(dev_impl->getStream(device_));

                    // Wait for asynchronous transfers to complete before reading.
                    sync_t0 = std::chrono::steady_clock::now();
                    torch::cuda::synchronize();
                    sync_t1 = std::chrono::steady_clock::now();
                } else {
                    pred_logits = pred_logits.to(torch::kCPU, /*non_blocking=*/false).contiguous();
                    pred_values = pred_values.to(torch::kCPU, /*non_blocking=*/false).contiguous();
                    if (time_device_) dev_ev[4].record(dev_impl->getStream(device_));
                    sync_t0 = sync_t1 = std::chrono::steady_clock::now();
                }
                if (time_device_) {
                    // Every event above is complete by here: either
                    // torch::cuda::synchronize() drained the stream, or the D2H
                    // was blocking. So no additional synchronise is inserted --
                    // the instrument must not add the very stalls it measures.
                    dev_ev[4].synchronize();
                    auto us = [](double ms) {
                        return static_cast<uint64_t>(ms * 1000.0 + 0.5);
                    };
                    h2d_us_.fetch_add(us(dev_ev[0].elapsedTime(dev_ev[1])),
                                      std::memory_order_relaxed);
                    forward_us_.fetch_add(us(dev_ev[1].elapsedTime(dev_ev[2])),
                                          std::memory_order_relaxed);
                    output_cast_us_.fetch_add(us(dev_ev[2].elapsedTime(dev_ev[3])),
                                              std::memory_order_relaxed);
                    d2h_us_.fetch_add(us(dev_ev[3].elapsedTime(dev_ev[4])),
                                      std::memory_order_relaxed);
                    sync_us_.fetch_add(
                        static_cast<uint64_t>(
                            std::chrono::duration_cast<std::chrono::microseconds>(
                                sync_t1 - sync_t0).count()),
                        std::memory_order_relaxed);
                    device_timed_batches_.fetch_add(1, std::memory_order_relaxed);
                }
            } else {
                pred_logits = pred_logits.contiguous();
                pred_values = pred_values.contiguous();
            }

            // --- DISTRIBUTE RESULTS & WAKE THREADS ---
            float* logits_ptr = pred_logits.data_ptr<float>();
            float* values_ptr = pred_values.data_ptr<float>();
            size_t action_dim = pred_logits.size(1);

            for (size_t i = 0; i < batch_size; ++i) {
                batch[i].result_dest->logits.assign(logits_ptr + i * action_dim, logits_ptr + (i + 1) * action_dim);
                batch[i].result_dest->value = values_ptr[i];

                // Set the atomic flag (fast spinning threads will catch this instantly)
                batch[i].ready_flag->store(true, std::memory_order_release);
            }

            // Memory barrier to guarantee parked threads see the ready flag,
            // followed by a SINGLE broadcast syscall (instead of 64 individual futex wakes)
            { std::lock_guard<std::mutex> park_lock(park_mutex_); }
            park_cv_.notify_all();
        }
    }
};

} // namespace open_spiel
