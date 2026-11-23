#ifndef OPEN_SPIEL_EXAMPLES_DUNE_NETWORK_EVAL_BATCHED_H_
#define OPEN_SPIEL_EXAMPLES_DUNE_NETWORK_EVAL_BATCHED_H_

#include <ATen/autocast_mode.h>
#include <ATen/cuda/CUDAGraph.h>
#include <c10/cuda/CUDAFunctions.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/torch.h>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include "dune_network_helpers.h"
#include "dune_network_model.h"
#include "dune_semantic_action_scorer.h"
#include "open_spiel/spiel.h"

namespace open_spiel {

class DeterministicEvaluator : public IGameEvaluator {
public:
    DeterministicEvaluator(std::shared_ptr<SharedDunePolicyValueNetImpl> model,
                           torch::Device device,
                           std::mutex* mutex,
                           std::shared_mutex* sync_mutex = nullptr,
                           std::shared_ptr<SharedDunePolicyValueNetImpl> critic_model = nullptr,
                           bool rollout_amp = true)
        : model_(model), critic_model_(critic_model), device_(device), mutex_(mutex),
          sync_mutex_(sync_mutex), rollout_amp_(rollout_amp) {
        if (model_) {
            deck_pool_semantics_ = model_->deck_pool_semantics_;
        }
        model_input_dim_ = model_->input_layer->weight.size(1);
        action_dim_ = model_->policy_head->weight.size(0);
        if (dune_imperium::IsFullPublicInformationStateSize(model_input_dim_)) {
            if (!model_->with_semantic_scorer_ || !model_->semantic_scorer_ ||
                (model_->semantic_descriptor_schema_ != dune_semantic::kDescriptorSchemaVersionV3 &&
                 model_->semantic_descriptor_schema_ != dune_semantic::kDescriptorSchemaVersionV4 &&
                 model_->semantic_descriptor_schema_ != dune_semantic::kDescriptorSchemaVersionV5)) {
                SpielFatalError("DeterministicEvaluator: Full public information actor requires an active semantic scorer with schema v3, v4, or v5");
            }
        }
    }

    bool HasSemanticScorer() const override {
        return model_ != nullptr && model_->with_semantic_scorer_;
    }
    std::string SemanticDescriptorSchema() const override {
        if (model_ != nullptr) return model_->semantic_descriptor_schema_;
        return dune_semantic::kDescriptorSchemaVersionV3;
    }

    bool RolloutAmpForTesting() const { return rollout_amp_; }

    EvalResult Evaluate(const std::vector<float>& obs) override {
        return EvaluateWithActions(obs, nullptr);
    }

    EvalResult EvaluateWithActions(
        const std::vector<float>& obs,
        const dune_semantic::CandidateActionData* action_data) override {
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
            SharedDunePolicyValueNetImpl::ModelOutputs critic_outputs;
            if (device_.is_cuda()) {
                // Blocking H2D (finding 1): the shared pinned input_tensor_ must
                // not be reused by the next thread until this copy's DMA has
                // finished reading it. non_blocking=false makes copy_ host-
                // synchronous (Torch inserts a stream sync), so the pinned
                // staging buffer is free the moment copy_ returns.
                device_tensor_.copy_(input_tensor_, /*non_blocking=*/false);
                AutocastGuard autocast_guard(c10::DeviceType::CUDA, rollout_amp_);
                outputs = model_->forward(device_tensor_);
                critic_outputs = (critic_model_ != nullptr) ? critic_model_->forward(device_tensor_) : outputs;
                if (model_->with_semantic_scorer_ && model_->semantic_scorer_ &&
                    action_data != nullptr && !action_data->empty()) {
                    std::vector<const dune_semantic::CandidateActionData*> single_batch = {action_data};
                    dune_semantic::ApplySemanticScorerBatch(
                        model_->semantic_scorer_, outputs.trunk, single_batch, outputs.logits, device_);
                }
            } else {
                outputs = model_->forward(input_tensor_);
                critic_outputs = (critic_model_ != nullptr) ? critic_model_->forward(input_tensor_) : outputs;
                if (model_->with_semantic_scorer_ && model_->semantic_scorer_ &&
                    action_data != nullptr && !action_data->empty()) {
                    std::vector<const dune_semantic::CandidateActionData*> single_batch = {action_data};
                    dune_semantic::ApplySemanticScorerBatch(
                        model_->semantic_scorer_, outputs.trunk, single_batch, outputs.logits, device_);
                }
            }

            // Blocking device->host copy INSIDE the lock drains the stream before
            // the mutex is released, so the next thread cannot overwrite the
            // shared staging buffers while this request's async work may still be
            // reading them (belt-and-suspenders for the finding-1 race). No
            // numeric change: identical ops, just serialized under the lock.
            pred_logits = outputs.logits.to(torch::kCPU).to(torch::kFloat32);
            pred_values = critic_outputs.values.to(torch::kCPU).to(torch::kFloat32);
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

    dune_imperium::DeckPoolSemantics DeckPoolSemantics() const override {
        return deck_pool_semantics_;
    }
    dune_imperium::MarketAppendixMode MarketMode() const override {
        if (model_ != nullptr) return model_->market_appendix_mode_;
        return dune_imperium::MarketAppendixMode::kNone;
    }

private:
    std::shared_ptr<SharedDunePolicyValueNetImpl> model_;
    std::shared_ptr<SharedDunePolicyValueNetImpl> critic_model_{nullptr};
    torch::Device device_;
    std::mutex* mutex_;
    std::shared_mutex* sync_mutex_;
    int64_t model_input_dim_;
    int64_t action_dim_;
    bool rollout_amp_{true};
    dune_imperium::DeckPoolSemantics deck_pool_semantics_ = dune_imperium::DeckPoolSemantics::kLegacy;
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
    // Step 1: Stage-1 Telemetry, Inter-Slot GPU Idle Gap & CPU stage timings
    double trunk_ms = 0.0;
    double scorer_ms = 0.0;
    double post_cap_softmax_ms = 0.0;
    double inter_slot_idle_gap_ms = 0.0;
    uint64_t inter_slot_gap_count = 0;
    double cpu_wait_ms = 0.0;
    double cpu_pack_ms = 0.0;
    double cpu_distribute_ms = 0.0;
    uint64_t wants_logits_batches = 0;
    uint64_t wants_logits_rows = 0;
    uint64_t d2h_bytes = 0;
    uint64_t greedy_action_calls = 0;
    uint64_t greedy_action_ties = 0;
    uint64_t full_batches = 0;
    uint64_t ge128_batches = 0;
    uint64_t ge64_batches = 0;
    uint64_t under64_batches = 0;
};

class BatchedEvaluator : public IGameEvaluator {
public:
    enum WaitState : uint32_t {
        kWaitSpinning = 0,
        kWaitParked = 1,
        kWaitNotifying = 2,
        kWaitReady = 3,
    };

    template <typename BeforeNotifyHook = void (*)()>
    static void SignalReadyFlagWithHook(
        std::atomic<uint32_t>* flag,
        BeforeNotifyHook before_notify = []() {}) {
        if (flag == nullptr) return;
        uint32_t expected = kWaitSpinning;
        if (flag->compare_exchange_strong(expected, kWaitReady,
                                          std::memory_order_acq_rel,
                                          std::memory_order_acquire)) {
            return;
        }
        // Caller transitioned to kWaitParked. Transition to kWaitNotifying so
        // WaitOnReadyFlag cannot return (or exit its thread and destroy *flag)
        // until notify_one() completes and kWaitReady is published as the
        // final access to *flag.
        flag->store(kWaitNotifying, std::memory_order_release);
        before_notify();
        flag->notify_one();
        flag->store(kWaitReady, std::memory_order_release);
    }

    static void SignalReadyFlag(std::atomic<uint32_t>* flag) {
        SignalReadyFlagWithHook(flag);
    }

    template <typename BeforeWaitHook = void (*)()>
    static void WaitOnReadyFlagWithHook(
        std::atomic<uint32_t>& flag,
        BeforeWaitHook before_wait = []() {}) {
        int spin_count = 0;
        while (flag.load(std::memory_order_acquire) != kWaitReady) {
            if (spin_count++ < 4000) {
#if defined(__x86_64__) || defined(_M_X64)
                _mm_pause();
#else
                std::this_thread::yield();
#endif
            } else {
                uint32_t expected = kWaitSpinning;
                if (flag.compare_exchange_strong(expected, kWaitParked,
                                                 std::memory_order_acq_rel,
                                                 std::memory_order_acquire)) {
                    before_wait();
                    flag.wait(kWaitParked, std::memory_order_acquire);
                } else if (expected == kWaitParked) {
                    flag.wait(kWaitParked, std::memory_order_acquire);
                } else if (expected == kWaitNotifying) {
                    std::this_thread::yield();
                }
            }
        }
    }

    static void WaitOnReadyFlag(std::atomic<uint32_t>& flag) {
        WaitOnReadyFlagWithHook(flag);
    }

    BatchedEvaluator(std::shared_ptr<SharedDunePolicyValueNetImpl> model,
                     int target_batch_size,
                     int timeout_ms,
                     torch::Device device,
                     std::shared_mutex* sync_mutex,
                     float logit_cap = 0.0f,
                     bool device_synchronize = true,
                     bool high_priority_stream = false,
                     bool emit_batch_membership = false,
                     bool rollout_amp = true,
                     bool allow_tf32 = true,
                     std::shared_ptr<SharedDunePolicyValueNetImpl> critic_model = nullptr,
                     bool clear_autocast_cache_per_batch = true);

    ~BatchedEvaluator();

    bool EmitsBatchMembershipForTesting() const {
        return emit_batch_membership_;
    }
    bool RolloutAmpForTesting() const { return rollout_amp_; }
    bool AllowTf32ForTesting() const { return allow_tf32_; }
    int64_t ModelInputDim() const { return model_input_dim_; }
    std::shared_ptr<SharedDunePolicyValueNetImpl> Model() const { return model_; }
    dune_imperium::DeckPoolSemantics DeckPoolSemantics() const override {
        return deck_pool_semantics_;
    }
    dune_imperium::MarketAppendixMode MarketMode() const override {
        if (model_ != nullptr) return model_->market_appendix_mode_;
        return dune_imperium::MarketAppendixMode::kNone;
    }
    bool HasSemanticScorer() const override {
        return model_ != nullptr && model_->with_semantic_scorer_;
    }
    std::string SemanticDescriptorSchema() const override {
        if (model_ != nullptr) return model_->semantic_descriptor_schema_;
        return dune_semantic::kDescriptorSchemaVersionV3;
    }

    void SetDispatchBudgetHook(
        std::function<bool(size_t batch_size)> hook,
        std::function<void(size_t rows)> on_rejection_hook = nullptr);

    void AdjustActiveRollouts(int delta) override;

    size_t EffectiveTargetBatchSize() const;

    EvalResult Evaluate(const std::vector<float>& obs) override;

    // Called by the actor threads
    EvalResult EvaluateWithActions(
        const std::vector<float>& obs,
        const dune_semantic::CandidateActionData* action_data) override;

    // Production leaf path: values are returned without allocating or copying
    // the 2,391-action policy vector to host memory.
    EvalResult EvaluateValues(const std::vector<float>& obs);

    CompactEvalResult EvaluateCompact(
        const std::vector<float>& obs,
        const std::vector<Action>& legal_actions,
        const dune_semantic::CandidateActionData* action_data = nullptr);

    GreedyEvalResult EvaluateGreedyAction(
        const std::vector<float>& obs,
        const std::vector<Action>& legal_actions,
        const dune_semantic::CandidateActionData* action_data = nullptr);

    std::vector<EvalResult> EvaluateBatch(
        const std::vector<std::vector<float>>& observations) override;

    EvaluatorStats GetStats() const override;

    std::vector<EvalResult> EvaluateBatchValues(
        const std::vector<std::vector<float>>& observations);

    std::pair<std::vector<EvalResult>, CompactEvalResult>
    EvaluateBatchValuesWithCompactPrior(
        const std::vector<std::vector<float>>& observations,
        size_t prior_index, const std::vector<Action>& legal_actions,
        const dune_semantic::CandidateActionData* action_data = nullptr);

    void EnableAsyncStageTimers(bool enable = true);

    void ResetBatcherStageTelemetry();

    void WarmupCudaGraphs(const std::vector<int>& batch_sizes);

    void InvalidateCudaGraphs();

    void SetEnableFp8Trunk(bool enable);

    // R2 (WO-3 review F1): arms the per-batch bookkeeping below (batch-size
    // record, per-row queue-wait accounting, split-leaf-group maps). Call it
    // BEFORE the workload starts; without it those sections are skipped and
    // GetBatcherTelemetry() reports zeros for the affected fields while the
    // always-on counters (rows, batches, class counts, timeout flushes) stay
    // live. Existing consumers that never opt in are unaffected by design.
    void EnableBatcherTelemetry();

    // WO-PERF-3: rich, opt-in telemetry for the Phase-2 "Batcher" list — the
    // detailed sections require EnableBatcherTelemetry() first. Safe to
    // call from another thread while the runner is live (all reads are atomic or
    // guarded by telemetry_mutex_); intended to be read once after the workload
    // drains. Percentiles are computed from the recorded per-batch sizes.
    BatcherTelemetry GetBatcherTelemetry() const;

    BatchedLogitCapAggregate GetLogitCapStats() const;

private:
    struct Request {
        const std::vector<float>* obs = nullptr;
        const dune_semantic::CandidateActionData* action_data = nullptr;
        EvalResult* result_dest = nullptr;
        std::atomic<uint32_t>* ready_flag = nullptr;
        bool wants_logits = true;
        CompactEvalResult* compact_dest = nullptr;
        Action* greedy_action_dest = nullptr;
        GreedyEvalResult* greedy_result_dest = nullptr;
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

    std::shared_ptr<SharedDunePolicyValueNetImpl> model_;
    std::shared_ptr<SharedDunePolicyValueNetImpl> critic_model_{nullptr};
    dune_imperium::DeckPoolSemantics deck_pool_semantics_{dune_imperium::DeckPoolSemantics::kLegacy};
    int target_batch_size_;
    int timeout_ms_;
    torch::Device device_;
    std::shared_mutex* sync_mutex_;
    float logit_cap_;
    bool device_synchronize_;
    bool high_priority_stream_;
    bool emit_batch_membership_;
    bool rollout_amp_;
    bool allow_tf32_;
    bool clear_autocast_cache_per_batch_{true};
    int64_t model_input_dim_;
    int64_t model_action_dim_;

    std::mutex mutex_;
    std::condition_variable cv_;
    std::deque<Request> requests_;
    std::chrono::steady_clock::time_point first_request_ts_;

    bool stop_;
    std::thread runner_thread_;
    std::atomic<int> active_rollouts_{0};
    std::atomic<uint64_t> total_batches_{0};
    std::atomic<uint64_t> total_requests_{0};
    std::atomic<uint64_t> max_batch_size_seen_{0};
    std::atomic<int64_t> next_membership_batch_id_{0};
    std::function<bool(size_t batch_size)> dispatch_budget_hook_{nullptr};
    std::function<void(size_t rows)> dispatch_rejection_hook_{nullptr};

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
    std::atomic<bool> async_stage_timers_enabled_{false};
    std::condition_variable warmup_cv_;
    bool warmup_pending_{false};
    std::vector<int> warmup_requested_sizes_;
    std::atomic<uint64_t> graph_generation_{0};

    std::atomic<uint64_t> wants_logits_batches_{0};
    std::atomic<uint64_t> wants_logits_rows_{0};
    std::atomic<uint64_t> d2h_bytes_{0};
    std::atomic<uint64_t> greedy_action_calls_{0};
    std::atomic<uint64_t> greedy_action_ties_{0};
    std::atomic<uint64_t> full_batches_{0};
    std::atomic<uint64_t> ge128_batches_{0};
    std::atomic<uint64_t> ge64_batches_{0};
    std::atomic<uint64_t> under64_batches_{0};

    std::atomic<uint64_t> trunk_us_{0};
    std::atomic<uint64_t> scorer_us_{0};
    std::atomic<uint64_t> post_cap_softmax_us_{0};
    std::atomic<uint64_t> inter_slot_idle_gap_us_{0};
    std::atomic<uint64_t> inter_slot_gap_count_{0};
    std::atomic<uint64_t> cpu_wait_us_{0};
    std::atomic<uint64_t> cpu_pack_us_{0};
    std::atomic<uint64_t> cpu_distribute_us_{0};
    std::atomic<uint64_t> logit_cap_decisions_{0};
    std::atomic<uint64_t> logit_cap_legal_logits_{0};
    std::atomic<uint64_t> logit_cap_saturated_{0};
    std::atomic<double> logit_cap_pre_max_{0.0};
    std::atomic<double> logit_cap_post_max_{0.0};
    // Runner-thread-only maps for split-leaf-group detection. A leaf group's
    // rows are contiguous in the deque, so within one physical batch a group is
    // a single run; a group counted in a second physical batch is a split.
    std::unordered_map<uint64_t, int> group_batches_seen_;  // gid -> # batches
    std::unordered_map<uint64_t, int> group_rows_left_;     // gid -> rows undispatched

    static void AtomicMax(std::atomic<double>* target, double value);

    void RecordLogitCapStats(const LogitCapApplicationStats& stats);

    void RecordLogitCapStats(double pre_max_abs, double post_max_abs,
                             int64_t legal_count, int64_t saturated_count);

    struct CapturedForwardGraph {
        std::unique_ptr<at::cuda::CUDAGraph> graph;
        SharedDunePolicyValueNetImpl::ModelOutputs outputs;
        SharedDunePolicyValueNetImpl::ModelOutputs critic_outputs;
        torch::Tensor projected_trunk;
        torch::Tensor values_f32;
    };

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
        torch::Tensor compact_host_ids;
        torch::Tensor compact_host_mask;
        torch::Tensor compact_host_rows;
        torch::Tensor compact_device_ids;
        torch::Tensor compact_device_mask;
        torch::Tensor compact_device_rows;
        torch::Tensor compact_host_probs;
        // Logit-cap telemetry for the compact path, copied to the host with the
        // results and recorded in finish_slot, so launching never waits on it.
        torch::Tensor host_cap_stats;
        bool has_cap_stats = false;
        bool pending = false;
        int64_t membership_batch_id = -1;
        at::cuda::MempoolId_t graph_pool;
        std::vector<CapturedForwardGraph> captured_graphs;

        uint64_t seq = 0;
        bool has_stage_timing = false;
        bool h2d_launched = false;
        bool selective_logits_active = false;
        std::vector<size_t> logits_request_indices;
        torch::Tensor logits_host_rows;
        torch::Tensor logits_device_rows;
        std::vector<size_t> greedy_request_indices;
        int greedy_max_legal = 0;
        int greedy_k_top = 0;
        torch::Tensor greedy_host_ids;
        torch::Tensor greedy_host_mask;
        torch::Tensor greedy_host_rows;
        torch::Tensor greedy_device_ids;
        torch::Tensor greedy_device_mask;
        torch::Tensor greedy_device_rows;
        torch::Tensor greedy_host_top_vals;
        torch::Tensor greedy_host_top_idx;
        torch::Tensor greedy_host_tie_probs;
        torch::Tensor greedy_last_logits;
        torch::Tensor greedy_last_mask;

        std::unique_ptr<c10::Event> ev_h2d_start;
        std::unique_ptr<c10::Event> ev_h2d_end;
        std::unique_ptr<c10::Event> ev_fwd_start;
        std::unique_ptr<c10::Event> ev_trunk_end;
        std::unique_ptr<c10::Event> ev_scorer_end;
        std::unique_ptr<c10::Event> ev_post_cap_end;
        std::unique_ptr<c10::Event> ev_d2h_end;
    };

    void CaptureForwardGraphIfNeeded(CapturedForwardGraph& cg,
                                     const torch::Tensor& device_obs,
                                     size_t batch_size,
                                     at::cuda::MempoolId_t graph_pool,
                                     bool enable_tail_graph);

    std::vector<Request> TakeBatchFromQueue(bool* timeout_flush,
                                            bool has_pending_slot,
                                            bool check_warmup,
                                            bool stage_timers_enabled);

    bool HandleBudgetRejection(std::vector<Request>& batch);

    void RecordBatchDispatchTelemetry(const std::vector<Request>& batch,
                                      bool timeout_flush);

    void WaitAsyncSlotGpu(
        AsyncBatchSlot& slot,
        std::vector<std::unique_ptr<c10::Event>>& batch_end_events);

    void DistributeAsyncSlotCpu(AsyncBatchSlot& slot);

    void EnsureAsyncSlotStageEvents(
        AsyncBatchSlot& slot,
        std::vector<std::unique_ptr<c10::Event>>& batch_end_events);

    void LaunchAsyncSlot(
        AsyncBatchSlot& slot,
        bool stage_timers_enabled,
        bool enable_selective_logits_d2h,
        bool enable_scorer_inplace,
        bool enable_tail_graph,
        uint64_t cap_stats_stride,
        uint64_t& next_batch_seq,
        uint64_t& compact_stats_counter,
        std::vector<std::unique_ptr<c10::Event>>& batch_end_events);

    void RunnerAsync();

    void Runner();
};

inline void BatchedEvaluator::WarmupCudaGraphs(const std::vector<int>& batch_sizes) {
    if (!device_.is_cuda() || device_synchronize_ || batch_sizes.empty()) {
        return;
    }
    std::vector<int> valid_sizes;
    valid_sizes.reserve(batch_sizes.size());
    for (int b : batch_sizes) {
        if (b >= 1 && b <= target_batch_size_) {
            valid_sizes.push_back(b);
        }
    }
    if (valid_sizes.empty()) return;
    std::sort(valid_sizes.begin(), valid_sizes.end());
    valid_sizes.erase(std::unique(valid_sizes.begin(), valid_sizes.end()), valid_sizes.end());

    std::unique_lock<std::mutex> lock(mutex_);
    warmup_requested_sizes_ = std::move(valid_sizes);
    warmup_pending_ = true;
    cv_.notify_one();
    warmup_cv_.wait(lock, [this]() { return !warmup_pending_ || stop_; });
}

inline void BatchedEvaluator::InvalidateCudaGraphs() {
    graph_generation_.fetch_add(1, std::memory_order_acq_rel);
}

inline void BatchedEvaluator::SetEnableFp8Trunk(bool enable) {
    if (sync_mutex_ != nullptr) {
        std::unique_lock<std::shared_mutex> sync_lock(*sync_mutex_);
        if (model_) model_->set_use_fp8_trunk(enable);
        if (critic_model_) critic_model_->set_use_fp8_trunk(enable);
    } else {
        if (model_) model_->set_use_fp8_trunk(enable);
        if (critic_model_) critic_model_->set_use_fp8_trunk(enable);
    }
    InvalidateCudaGraphs();
}

inline void BatchedEvaluator::CaptureForwardGraphIfNeeded(
    CapturedForwardGraph& cg,
    const torch::Tensor& device_obs,
    size_t batch_size,
    at::cuda::MempoolId_t graph_pool,
    bool enable_tail_graph) {
    if (cg.graph) {
        return;
    }
    if (rollout_amp_ && device_.is_cuda()) {
        if (model_) {
            if (model_->use_fp8_trunk()) {
                model_->PrepareFp8Cache();
            } else {
                model_->RefreshBf16InferenceCache();
            }
        }
        if (critic_model_) {
            if (critic_model_->use_fp8_trunk()) {
                critic_model_->PrepareFp8Cache();
            } else {
                critic_model_->RefreshBf16InferenceCache();
            }
        }
    }
    SharedDunePolicyValueNetImpl::ModelOutputs outputs;
    SharedDunePolicyValueNetImpl::ModelOutputs critic_outputs;
    {
        AutocastGuard autocast_guard(c10::DeviceType::CUDA,
                                     rollout_amp_,
                                     /*clear_cache_on_exit=*/true);
        outputs = model_->forward(device_obs);
        critic_outputs = (critic_model_ != nullptr)
            ? critic_model_->forward(device_obs)
            : outputs;
        if (enable_tail_graph && model_->with_semantic_scorer_ && model_->semantic_scorer_) {
            (void)model_->semantic_scorer_->ProjectTrunk(outputs.trunk);
        }
    }
    at::autocast::clear_cache();
    {
        AutocastGuard autocast_guard(c10::DeviceType::CUDA,
                                     rollout_amp_,
                                     /*clear_cache_on_exit=*/true);
        cg.graph = std::make_unique<at::cuda::CUDAGraph>();
        cg.graph->capture_begin(graph_pool, cudaStreamCaptureModeThreadLocal);
        cg.outputs = model_->forward(device_obs);
        cg.critic_outputs = (critic_model_ != nullptr)
            ? critic_model_->forward(device_obs)
            : cg.outputs;
        if (enable_tail_graph) {
            if (model_->with_semantic_scorer_ && model_->semantic_scorer_) {
                cg.projected_trunk = model_->semantic_scorer_->ProjectTrunk(cg.outputs.trunk);
            }
            cg.values_f32 = cg.critic_outputs.values.reshape({static_cast<int64_t>(batch_size)}).to(torch::kFloat32);
        }
        cg.graph->capture_end();
    }
}

}  // namespace open_spiel

#endif  // OPEN_SPIEL_EXAMPLES_DUNE_NETWORK_EVAL_BATCHED_H_
