#ifndef OPEN_SPIEL_EXAMPLES_DUNE_NETWORK_EVAL_BATCHED_IMPL_H_
#define OPEN_SPIEL_EXAMPLES_DUNE_NETWORK_EVAL_BATCHED_IMPL_H_

#include <torch/torch.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "dune_network_eval_batched.h"

namespace open_spiel {

inline BatchedEvaluator::BatchedEvaluator(
    std::shared_ptr<SharedDunePolicyValueNetImpl> model,
    int target_batch_size,
    int timeout_ms,
    torch::Device device,
    std::shared_mutex* sync_mutex,
    float logit_cap,
    bool device_synchronize,
    bool high_priority_stream,
    bool emit_batch_membership,
    bool rollout_amp,
    bool allow_tf32,
    std::shared_ptr<SharedDunePolicyValueNetImpl> critic_model,
    bool clear_autocast_cache_per_batch)
    : model_(model), critic_model_(critic_model), target_batch_size_(target_batch_size),
      timeout_ms_(timeout_ms), device_(device), sync_mutex_(sync_mutex),
      logit_cap_(logit_cap), device_synchronize_(device_synchronize),
      high_priority_stream_(high_priority_stream),
      emit_batch_membership_(emit_batch_membership),
      rollout_amp_(rollout_amp), allow_tf32_(allow_tf32),
      clear_autocast_cache_per_batch_(clear_autocast_cache_per_batch),
      async_stage_timers_enabled_((std::getenv("DUNE_ENABLE_ASYNC_STAGE_TIMERS") != nullptr)),
      stop_(false) {
    if (model_) {
        deck_pool_semantics_ = model_->deck_pool_semantics_;
    }

    // Dynamically get the input layer dimension from the model's weights
    model_input_dim_ = model_->input_layer->weight.size(1);
    model_action_dim_ = model_->policy_head->weight.size(0);
    if (dune_imperium::IsFullPublicInformationStateSize(model_input_dim_)) {
        if (!model_->with_semantic_scorer_ || !model_->semantic_scorer_ ||
            (model_->semantic_descriptor_schema_ != dune_semantic::kDescriptorSchemaVersionV3 &&
             model_->semantic_descriptor_schema_ != dune_semantic::kDescriptorSchemaVersionV4 &&
             model_->semantic_descriptor_schema_ != dune_semantic::kDescriptorSchemaVersionV5)) {
            SpielFatalError("BatchedEvaluator: Full public information actor requires an active semantic scorer with schema v3, v4, or v5");
        }
    }

    // Enable TF32 for Ada Lovelace (RTX 4080 Super) speedup
    if (device_.is_cuda()) {
        at::globalContext().setAllowTF32CuBLAS(allow_tf32_);
        at::globalContext().setAllowTF32CuDNN(allow_tf32_);
    }

    model_->eval(); // Defensive hygiene: ResBlocks use LayerNorm so this is a no-op, but protects future Dropout additions.
    if (critic_model_) {
        critic_model_->eval();
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
        clear_autocast_cache_per_batch_ = false;
    }

    runner_thread_ = std::thread(&BatchedEvaluator::Runner, this);
}

inline BatchedEvaluator::~BatchedEvaluator() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        stop_ = true;
    }
    cv_.notify_all();
    warmup_cv_.notify_all();
    if (runner_thread_.joinable()) runner_thread_.join();
}

inline void BatchedEvaluator::SetDispatchBudgetHook(
    std::function<bool(size_t batch_size)> hook,
    std::function<void(size_t rows)> on_rejection_hook) {
    std::lock_guard<std::mutex> lock(mutex_);
    dispatch_budget_hook_ = std::move(hook);
    dispatch_rejection_hook_ = std::move(on_rejection_hook);
}

inline void BatchedEvaluator::AdjustActiveRollouts(int delta) {
    int prev = active_rollouts_.fetch_add(delta, std::memory_order_relaxed);
    if (delta < 0 && prev + delta > 0) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!requests_.empty() &&
            requests_.size() >= EffectiveTargetBatchSize()) {
            cv_.notify_one();
        }
    }
}

inline size_t BatchedEvaluator::EffectiveTargetBatchSize() const {
    size_t eff = static_cast<size_t>(target_batch_size_);
    int act = active_rollouts_.load(std::memory_order_relaxed);
    if (act > 0) {
        size_t cap = static_cast<size_t>(act);
        eff = std::min(eff, cap);
    }
    return eff;
}

inline EvalResult BatchedEvaluator::Evaluate(const std::vector<float>& obs) {
    return EvaluateWithActions(obs, nullptr);
}

// Called by the actor threads
inline EvalResult BatchedEvaluator::EvaluateWithActions(
    const std::vector<float>& obs,
    const dune_semantic::CandidateActionData* action_data) {
    // Validate size in the caller thread (finding 6) so the runner never
    // over-reads/truncates and never faults on an unsupported request.
    CheckEvalObsSize(obs.size(), model_input_dim_);
    EvalResult result; // Stack allocated!
    thread_local std::atomic<uint32_t> tls_ready{kWaitSpinning};
    tls_ready.store(kWaitSpinning, std::memory_order_relaxed);

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
        req.action_data = action_data;
        req.result_dest = &result;
        req.ready_flag = &tls_ready;
        req.wants_logits = true;
        req.group_id = next_group_id_.fetch_add(1, std::memory_order_relaxed);
        req.group_size = 1;
        req.is_group = false;
        req.enqueue_ts = now;
        requests_.push_back(req);
        single_row_calls_.fetch_add(1, std::memory_order_relaxed);

        // Only wake the Runner on the first arrival or when the batch is full
        if (requests_.size() == 1 || requests_.size() >= EffectiveTargetBatchSize()) {
            cv_.notify_one();
        }
    }

    WaitOnReadyFlag(tls_ready);
    return result; // Move semantics
}

// Production leaf path: values are returned without allocating or copying
// the 2,391-action policy vector to host memory.
inline EvalResult BatchedEvaluator::EvaluateValues(const std::vector<float>& obs) {
    CheckEvalObsSize(obs.size(), model_input_dim_);
    EvalResult result;
    thread_local std::atomic<uint32_t> tls_ready{kWaitSpinning};
    tls_ready.store(kWaitSpinning, std::memory_order_relaxed);
    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto now = std::chrono::steady_clock::now();
        if (requests_.empty()) first_request_ts_ = now;
        Request req;
        req.obs = &obs;
        req.result_dest = &result;
        req.ready_flag = &tls_ready;
        req.wants_logits = false;
        req.group_id = next_group_id_.fetch_add(1, std::memory_order_relaxed);
        req.group_size = 1;
        req.is_group = false;
        req.enqueue_ts = now;
        requests_.push_back(req);
        single_row_calls_.fetch_add(1, std::memory_order_relaxed);
        if (requests_.size() == 1 ||
            requests_.size() >= EffectiveTargetBatchSize()) {
            cv_.notify_one();
        }
    }
    WaitOnReadyFlag(tls_ready);
    return result;
}

inline CompactEvalResult BatchedEvaluator::EvaluateCompact(
    const std::vector<float>& obs,
    const std::vector<Action>& legal_actions,
    const dune_semantic::CandidateActionData* action_data) {
    if (!device_.is_cuda() || device_synchronize_) {
        EvalResult full = EvaluateWithActions(obs, action_data);
        CompactEvalResult compact;
        if (!full.ok) {
            compact.ok = false;
            compact.budget_exhausted = full.budget_exhausted;
            return compact;
        }
        compact.actions = legal_actions;
        std::vector<float> logits = full.logits;
        const LogitCapApplicationStats cap_stats =
            CenterAndCapLegalLogitsWithStats(logits, legal_actions,
                                             logit_cap_);
        RecordLogitCapStats(cap_stats);
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
        compact.physical_batch_id = full.physical_batch_id;
        compact.physical_batch_size = full.physical_batch_size;
        compact.physical_batch_row = full.physical_batch_row;
        return compact;
    }
    CheckEvalObsSize(obs.size(), model_input_dim_);
    CompactEvalResult result;
    thread_local std::atomic<uint32_t> tls_ready{kWaitSpinning};
    tls_ready.store(kWaitSpinning, std::memory_order_relaxed);
    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto now = std::chrono::steady_clock::now();
        if (requests_.empty()) first_request_ts_ = now;
        Request req;
        req.obs = &obs;
        req.action_data = action_data;
        req.ready_flag = &tls_ready;
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
            requests_.size() >= EffectiveTargetBatchSize()) {
            cv_.notify_one();
        }
    }
    WaitOnReadyFlag(tls_ready);
    return result;
}

inline GreedyEvalResult BatchedEvaluator::EvaluateGreedyAction(
    const std::vector<float>& obs,
    const std::vector<Action>& legal_actions,
    const dune_semantic::CandidateActionData* action_data) {
    if (legal_actions.empty()) {
        return {kInvalidAction, true, false};
    }
    CheckEvalObsSize(obs.size(), model_input_dim_);
    GreedyEvalResult result;
    thread_local std::atomic<uint32_t> tls_ready{kWaitSpinning};
    tls_ready.store(kWaitSpinning, std::memory_order_relaxed);
    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto now = std::chrono::steady_clock::now();
        if (requests_.empty()) first_request_ts_ = now;
        Request req;
        req.obs = &obs;
        req.action_data = action_data;
        req.ready_flag = &tls_ready;
        req.result_dest = nullptr;
        req.compact_dest = nullptr;
        req.greedy_action_dest = &result.action;
        req.greedy_result_dest = &result;
        req.legal_actions = legal_actions;
        req.wants_logits = false;
        req.group_id = next_group_id_.fetch_add(1, std::memory_order_relaxed);
        req.group_size = 1;
        req.is_group = false;
        req.enqueue_ts = now;
        requests_.push_back(std::move(req));
        single_row_calls_.fetch_add(1, std::memory_order_relaxed);
        if (requests_.size() == 1 ||
            requests_.size() >= EffectiveTargetBatchSize()) {
            cv_.notify_one();
        }
    }
    WaitOnReadyFlag(tls_ready);
    return result;
}

inline std::vector<EvalResult> BatchedEvaluator::EvaluateBatch(
    const std::vector<std::vector<float>>& observations) {
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

    thread_local std::vector<std::unique_ptr<std::atomic<uint32_t>>> tls_group_ready;
    while (tls_group_ready.size() < observations.size()) {
        tls_group_ready.push_back(std::make_unique<std::atomic<uint32_t>>(kWaitSpinning));
    }
    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto now = std::chrono::steady_clock::now();
        const bool was_empty = requests_.empty();
        if (was_empty) {
            first_request_ts_ = now;
        }
        // WO-PERF-3: one group id for the whole leaf group so the runner can
        // detect when its rows straddle two physical batches.
        const uint64_t gid = next_group_id_.fetch_add(1, std::memory_order_relaxed);
        const uint32_t gsize = static_cast<uint32_t>(observations.size());
        for (size_t i = 0; i < observations.size(); ++i) {
            tls_group_ready[i]->store(kWaitSpinning, std::memory_order_relaxed);
            Request req;
            req.obs = &observations[i];
            req.result_dest = &results[i];
            req.ready_flag = tls_group_ready[i].get();
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
        if (was_empty || requests_.size() >= EffectiveTargetBatchSize()) {
            cv_.notify_one();
        }
    }

    for (size_t i = 0; i < observations.size(); ++i) {
        WaitOnReadyFlag(*tls_group_ready[i]);
    }
    return results;
}

inline EvaluatorStats BatchedEvaluator::GetStats() const {
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

inline std::vector<EvalResult> BatchedEvaluator::EvaluateBatchValues(
    const std::vector<std::vector<float>>& observations) {
    std::vector<EvalResult> results(observations.size());
    if (observations.empty()) return results;
    if (observations.size() > static_cast<size_t>(target_batch_size_)) {
        SpielFatalError("BatchedEvaluator::EvaluateBatchValues group exceeds target");
    }
    for (const auto& obs : observations) CheckEvalObsSize(obs.size(), model_input_dim_);
    thread_local std::vector<std::unique_ptr<std::atomic<uint32_t>>> tls_group_ready;
    while (tls_group_ready.size() < observations.size()) {
        tls_group_ready.push_back(std::make_unique<std::atomic<uint32_t>>(kWaitSpinning));
    }
    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto now = std::chrono::steady_clock::now();
        const bool was_empty = requests_.empty();
        if (was_empty) first_request_ts_ = now;
        const uint64_t gid = next_group_id_.fetch_add(1, std::memory_order_relaxed);
        const uint32_t gsize = static_cast<uint32_t>(observations.size());
        for (size_t i = 0; i < observations.size(); ++i) {
            tls_group_ready[i]->store(kWaitSpinning, std::memory_order_relaxed);
            Request req;
            req.obs = &observations[i];
            req.result_dest = &results[i];
            req.ready_flag = tls_group_ready[i].get();
            req.wants_logits = false;
            req.group_id = gid;
            req.group_size = gsize;
            req.is_group = true;
            req.enqueue_ts = now;
            requests_.push_back(req);
        }
        group_calls_.fetch_add(1, std::memory_order_relaxed);
        group_rows_.fetch_add(gsize, std::memory_order_relaxed);
        if (was_empty || requests_.size() >= EffectiveTargetBatchSize()) {
            cv_.notify_one();
        }
    }
    for (size_t i = 0; i < observations.size(); ++i) {
        WaitOnReadyFlag(*tls_group_ready[i]);
    }
    return results;
}

inline std::pair<std::vector<EvalResult>, CompactEvalResult>
BatchedEvaluator::EvaluateBatchValuesWithCompactPrior(
    const std::vector<std::vector<float>>& observations,
    size_t prior_index, const std::vector<Action>& legal_actions,
    const dune_semantic::CandidateActionData* action_data) {
    std::vector<EvalResult> results(observations.size());
    CompactEvalResult compact;
    if (observations.empty()) {
        return {std::move(results), std::move(compact)};
    }
    if (observations.size() > static_cast<size_t>(target_batch_size_)) {
        SpielFatalError("BatchedEvaluator::EvaluateBatchValuesWithCompactPrior group size " +
                        std::to_string(observations.size()) +
                        " exceeds configured target " +
                        std::to_string(target_batch_size_));
    }
    if (prior_index >= observations.size()) {
        return {std::move(results), std::move(compact)};
    }
    for (const auto& obs : observations) CheckEvalObsSize(obs.size(), model_input_dim_);
    thread_local std::vector<std::unique_ptr<std::atomic<uint32_t>>> tls_group_ready;
    while (tls_group_ready.size() < observations.size()) {
        tls_group_ready.push_back(std::make_unique<std::atomic<uint32_t>>(kWaitSpinning));
    }
    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto now = std::chrono::steady_clock::now();
        if (requests_.empty()) first_request_ts_ = now;
        const uint64_t gid = next_group_id_.fetch_add(1, std::memory_order_relaxed);
        const uint32_t gsize = static_cast<uint32_t>(observations.size());
        for (size_t i = 0; i < observations.size(); ++i) {
            tls_group_ready[i]->store(kWaitSpinning, std::memory_order_relaxed);
            Request req;
            req.obs = &observations[i];
            req.result_dest = &results[i];
            req.ready_flag = tls_group_ready[i].get();
            req.wants_logits = false;
            if (i == prior_index) {
                req.compact_dest = &compact;
                req.legal_actions = legal_actions;
                req.action_data = action_data;
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
    for (size_t i = 0; i < observations.size(); ++i) {
        WaitOnReadyFlag(*tls_group_ready[i]);
    }
    return {std::move(results), std::move(compact)};
}

inline void BatchedEvaluator::EnableAsyncStageTimers(bool enable) {
    async_stage_timers_enabled_.store(enable, std::memory_order_relaxed);
}

inline void BatchedEvaluator::ResetBatcherStageTelemetry() {
    wants_logits_batches_.store(0, std::memory_order_relaxed);
    wants_logits_rows_.store(0, std::memory_order_relaxed);
    d2h_bytes_.store(0, std::memory_order_relaxed);
    greedy_action_calls_.store(0, std::memory_order_relaxed);
    greedy_action_ties_.store(0, std::memory_order_relaxed);
    full_batches_.store(0, std::memory_order_relaxed);
    ge128_batches_.store(0, std::memory_order_relaxed);
    ge64_batches_.store(0, std::memory_order_relaxed);
    under64_batches_.store(0, std::memory_order_relaxed);
    h2d_us_.store(0, std::memory_order_relaxed);
    forward_us_.store(0, std::memory_order_relaxed);
    output_cast_us_.store(0, std::memory_order_relaxed);
    d2h_us_.store(0, std::memory_order_relaxed);
    sync_us_.store(0, std::memory_order_relaxed);
    device_timed_batches_.store(0, std::memory_order_relaxed);
    trunk_us_.store(0, std::memory_order_relaxed);
    scorer_us_.store(0, std::memory_order_relaxed);
    post_cap_softmax_us_.store(0, std::memory_order_relaxed);
    inter_slot_idle_gap_us_.store(0, std::memory_order_relaxed);
    inter_slot_gap_count_.store(0, std::memory_order_relaxed);
    cpu_wait_us_.store(0, std::memory_order_relaxed);
    cpu_pack_us_.store(0, std::memory_order_relaxed);
    cpu_distribute_us_.store(0, std::memory_order_relaxed);
}

inline void BatchedEvaluator::EnableBatcherTelemetry() {
    batcher_telemetry_enabled_.store(true, std::memory_order_relaxed);
}

inline BatcherTelemetry BatchedEvaluator::GetBatcherTelemetry() const {
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
    t.trunk_ms = trunk_us_.load(std::memory_order_relaxed) / 1000.0;
    t.scorer_ms = scorer_us_.load(std::memory_order_relaxed) / 1000.0;
    t.post_cap_softmax_ms = post_cap_softmax_us_.load(std::memory_order_relaxed) / 1000.0;
    t.inter_slot_idle_gap_ms = inter_slot_idle_gap_us_.load(std::memory_order_relaxed) / 1000.0;
    t.inter_slot_gap_count = inter_slot_gap_count_.load(std::memory_order_relaxed);
    t.cpu_wait_ms = cpu_wait_us_.load(std::memory_order_relaxed) / 1000.0;
    t.cpu_pack_ms = cpu_pack_us_.load(std::memory_order_relaxed) / 1000.0;
    t.cpu_distribute_ms = cpu_distribute_us_.load(std::memory_order_relaxed) / 1000.0;
    t.wants_logits_batches = wants_logits_batches_.load(std::memory_order_relaxed);
    t.wants_logits_rows = wants_logits_rows_.load(std::memory_order_relaxed);
    t.d2h_bytes = d2h_bytes_.load(std::memory_order_relaxed);
    t.greedy_action_calls = greedy_action_calls_.load(std::memory_order_relaxed);
    t.greedy_action_ties = greedy_action_ties_.load(std::memory_order_relaxed);
    t.full_batches = full_batches_.load(std::memory_order_relaxed);
    t.ge128_batches = ge128_batches_.load(std::memory_order_relaxed);
    t.ge64_batches = ge64_batches_.load(std::memory_order_relaxed);
    t.under64_batches = under64_batches_.load(std::memory_order_relaxed);
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

inline BatchedLogitCapAggregate BatchedEvaluator::GetLogitCapStats() const {
    return {logit_cap_decisions_.load(std::memory_order_relaxed),
            logit_cap_legal_logits_.load(std::memory_order_relaxed),
            logit_cap_saturated_.load(std::memory_order_relaxed),
            logit_cap_pre_max_.load(std::memory_order_relaxed),
            logit_cap_post_max_.load(std::memory_order_relaxed)};
}

inline void BatchedEvaluator::AtomicMax(std::atomic<double>* target, double value) {
    double current = target->load(std::memory_order_relaxed);
    while (value > current &&
           !target->compare_exchange_weak(
               current, value, std::memory_order_relaxed,
               std::memory_order_relaxed)) {
    }
}

inline void BatchedEvaluator::RecordLogitCapStats(const LogitCapApplicationStats& stats) {
    RecordLogitCapStats(stats.pre_max_abs, stats.post_max_abs,
                        stats.legal_count, stats.saturated_count);
}

inline void BatchedEvaluator::RecordLogitCapStats(double pre_max_abs, double post_max_abs,
                                                  int64_t legal_count, int64_t saturated_count) {
    logit_cap_decisions_.fetch_add(1, std::memory_order_relaxed);
    logit_cap_legal_logits_.fetch_add(
        static_cast<uint64_t>(std::max<int64_t>(0, legal_count)),
        std::memory_order_relaxed);
    logit_cap_saturated_.fetch_add(
        static_cast<uint64_t>(std::max<int64_t>(0, saturated_count)),
        std::memory_order_relaxed);
    AtomicMax(&logit_cap_pre_max_, pre_max_abs);
    AtomicMax(&logit_cap_post_max_, post_max_abs);
}

}  // namespace open_spiel

#endif  // OPEN_SPIEL_EXAMPLES_DUNE_NETWORK_EVAL_BATCHED_IMPL_H_
