#pragma once

#include <torch/torch.h>
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>
#include <deque>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <chrono>
#include <random>
#include <shared_mutex>
#include <utility>
#include <ATen/autocast_mode.h>

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
        if (device_type_ == c10::DeviceType::CUDA) {
            at::autocast::set_autocast_enabled(at::kCUDA, previous_state_);
            at::autocast::set_autocast_dtype(at::kCUDA, previous_dtype_);
        } else if (device_type_ == c10::DeviceType::CPU) {
            at::autocast::set_autocast_enabled(at::kCPU, previous_state_);
            at::autocast::set_autocast_dtype(at::kCPU, previous_dtype_);
        }
    }
};

struct EvalResult {
    std::vector<float> logits;
    float value;
};

inline void CenterAndCapLegalLogits(std::vector<float>& logits,
                                    const std::vector<Action>& legal_actions,
                                    float logit_cap) {
    if (logits.empty() || legal_actions.empty()) return;

    double legal_sum = 0.0;
    int legal_count = 0;
    for (Action action : legal_actions) {
        if (action >= 0 && static_cast<size_t>(action) < logits.size()) {
            legal_sum += logits[action];
            ++legal_count;
        }
    }
    if (legal_count == 0) return;

    const float legal_mean = static_cast<float>(legal_sum / legal_count);
    // Only legal logits are consumed by the softmax. Transforming all 2,391
    // outputs performed thousands of unnecessary tanh calls per game.
    for (Action action : legal_actions) {
        if (action < 0 || static_cast<size_t>(action) >= logits.size()) continue;
        float& logit = logits[action];
        logit -= legal_mean;
        if (logit_cap > 0.0f) {
            logit = logit_cap * std::tanh(logit / logit_cap);
        }
    }
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

class BatchedEvaluator : public IGameEvaluator {
public:
    BatchedEvaluator(std::shared_ptr<SharedDunePolicyValueNetImpl> model,
                     int target_batch_size,
                     int timeout_ms,
                     torch::Device device,
                     std::shared_mutex* sync_mutex,
                     float logit_cap = 0.0f,
                     bool device_synchronize = true)
        : model_(model), target_batch_size_(target_batch_size),
          timeout_ms_(timeout_ms), device_(device), sync_mutex_(sync_mutex),
          logit_cap_(logit_cap), device_synchronize_(device_synchronize),
          stop_(false) {

        // Dynamically get the input layer dimension from the model's weights
        model_input_dim_ = model_->input_layer->weight.size(1);

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
            if (requests_.empty()) {
                first_request_ts_ = std::chrono::steady_clock::now();
            }
            requests_.push_back({&obs, &result, &ready});

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

    std::vector<EvalResult> EvaluateBatch(
        const std::vector<std::vector<float>>& observations) override {
        std::vector<EvalResult> results(observations.size());
        if (observations.empty()) return results;
        for (const auto& obs : observations) {
            CheckEvalObsSize(obs.size(), model_input_dim_);
        }

        auto ready = std::make_unique<std::atomic<bool>[]>(observations.size());
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (requests_.empty()) {
                first_request_ts_ = std::chrono::steady_clock::now();
            }
            for (size_t i = 0; i < observations.size(); ++i) {
                ready[i].store(false, std::memory_order_relaxed);
                requests_.push_back({&observations[i], &results[i], &ready[i]});
            }
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

private:
    struct Request {
        const std::vector<float>* obs;
        EvalResult* result_dest;
        std::atomic<bool>* ready_flag;
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
    int64_t model_input_dim_;

    std::mutex mutex_;
    std::condition_variable cv_;
    std::deque<Request> requests_;
    std::chrono::steady_clock::time_point first_request_ts_;

    bool stop_;
    std::thread runner_thread_;
    std::atomic<uint64_t> total_batches_{0};
    std::atomic<uint64_t> total_requests_{0};
    std::atomic<uint64_t> max_batch_size_seen_{0};

    void Runner() {
        torch::InferenceMode inference_guard;
        torch::Tensor pinned_stacked_obs;
        bool pinned_allocated = false;

        while (true) {
            std::vector<Request> batch;

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

                // 4. Scoop up the batch
                size_t actual_batch = std::min((size_t)target_batch_size_, requests_.size());
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

            // Non-blocking H2D transfer using .slice() for partial batches
            torch::Tensor device_obs = device_.is_cuda()
                ? pinned_stacked_obs.slice(0, 0, batch_size).to(device_, /*non_blocking=*/true)
                : pinned_stacked_obs.slice(0, 0, batch_size);

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

            // Return raw logits. The tanh soft-cap must be applied after
            // legal-action centering, which requires the State's legal actions.

            // C. NON-BLOCKING D2H TRANSFER (cast AMP FP16 outputs back to Float32)
            if (device_.is_cuda()) {
                pred_logits = pred_logits.to(torch::kFloat32);
                pred_values = pred_values.to(torch::kFloat32);
                if (device_synchronize_) {
                    pred_logits = pred_logits.to(torch::kCPU, /*non_blocking=*/true).contiguous();
                    pred_values = pred_values.to(torch::kCPU, /*non_blocking=*/true).contiguous();

                    // Wait for asynchronous transfers to complete before reading.
                    torch::cuda::synchronize();
                } else {
                    pred_logits = pred_logits.to(torch::kCPU, /*non_blocking=*/false).contiguous();
                    pred_values = pred_values.to(torch::kCPU, /*non_blocking=*/false).contiguous();
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
