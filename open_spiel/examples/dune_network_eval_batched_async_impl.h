#ifndef OPEN_SPIEL_EXAMPLES_DUNE_NETWORK_EVAL_BATCHED_ASYNC_IMPL_H_
#define OPEN_SPIEL_EXAMPLES_DUNE_NETWORK_EVAL_BATCHED_ASYNC_IMPL_H_

#include <ATen/autocast_mode.h>
#include <ATen/cuda/CUDAGraph.h>
#include <c10/cuda/CUDAFunctions.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/torch.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <unordered_map>
#include <utility>
#include <vector>

#include "dune_network_eval_batched.h"
#include "dune_network_eval_batched_async_slot_impl.h"
#include "dune_network_helpers.h"
#include "dune_network_model.h"
#include "open_spiel/spiel.h"

namespace open_spiel {

inline std::vector<BatchedEvaluator::Request> BatchedEvaluator::TakeBatchFromQueue(
    bool* timeout_flush,
    bool has_pending_slot,
    bool check_warmup,
    bool stage_timers_enabled) {
    std::vector<Request> batch;
    *timeout_flush = false;
    auto t_wait_start = stage_timers_enabled ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, [this, check_warmup]() {
            return stop_ || !requests_.empty() || (check_warmup && warmup_pending_);
        });
        if (check_warmup && warmup_pending_) return batch;
        if (stop_ && requests_.empty()) return batch;
        if (!has_pending_slot) {
            const auto deadline =
                first_request_ts_ + std::chrono::milliseconds(timeout_ms_);
            cv_.wait_until(lock, deadline, [this, deadline, check_warmup]() {
                return stop_ || requests_.size() >= EffectiveTargetBatchSize() ||
                       (check_warmup && warmup_pending_) ||
                       std::chrono::steady_clock::now() >= deadline;
            });
        }
        if (check_warmup && warmup_pending_) return batch;
        if (stop_ && requests_.empty()) return batch;
        *timeout_flush = !stop_ &&
                         requests_.size() <
                             static_cast<size_t>(target_batch_size_);
        // Scoop up the batch. EvaluateBatch() inserts every logical
        // leaf group atomically and contiguously. Do not let the target
        // boundary cut one of those groups across two physical
        // forwards: besides wasting a second launch, batch-size-
        // dependent kernels could otherwise give the four players in
        // one leaf slightly different numerical contexts. Back off to
        // the start of the boundary group. (A group larger than the
        // configured target is rejected by EvaluateBatch() before it
        // reaches this queue.)
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
    if (stage_timers_enabled) {
        auto t_wait_end = std::chrono::steady_clock::now();
        cpu_wait_us_.fetch_add(
            static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(
                t_wait_end - t_wait_start).count()),
            std::memory_order_relaxed);
    }
    return batch;
}

inline bool BatchedEvaluator::HandleBudgetRejection(std::vector<Request>& batch) {
    if (!dispatch_budget_hook_) {
        return false;
    }
    if (dispatch_budget_hook_(batch.size())) {
        return false;
    }
    if (dispatch_rejection_hook_) {
        dispatch_rejection_hook_(batch.size());
    }
    for (auto& req : batch) {
        if (req.result_dest) {
            req.result_dest->ok = false;
            req.result_dest->budget_exhausted = true;
            req.result_dest->logits.clear();
        }
        if (req.compact_dest) {
            req.compact_dest->ok = false;
            req.compact_dest->budget_exhausted = true;
            req.compact_dest->actions.clear();
            req.compact_dest->probabilities.clear();
        }
        if (req.greedy_result_dest) {
            req.greedy_result_dest->ok = false;
            req.greedy_result_dest->budget_exhausted = true;
        }
        if (req.greedy_action_dest) {
            *req.greedy_action_dest = kInvalidAction;
        }
        SignalReadyFlag(req.ready_flag);
    }
    return true;
}

inline void BatchedEvaluator::RecordBatchDispatchTelemetry(
    const std::vector<Request>& batch,
    bool timeout_flush) {
    const size_t batch_size = batch.size();
    total_batches_.fetch_add(1, std::memory_order_relaxed);
    total_requests_.fetch_add(static_cast<uint64_t>(batch_size),
                               std::memory_order_relaxed);
    if (batch_size >= static_cast<size_t>(target_batch_size_)) {
        full_batches_.fetch_add(1, std::memory_order_relaxed);
    } else if (batch_size >= 128) {
        ge128_batches_.fetch_add(1, std::memory_order_relaxed);
    } else if (batch_size >= 64) {
        ge64_batches_.fetch_add(1, std::memory_order_relaxed);
    } else {
        under64_batches_.fetch_add(1, std::memory_order_relaxed);
    }
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
}

inline void BatchedEvaluator::RunnerAsync() {
    torch::InferenceMode inference_guard;
    const bool enable_slot_reorder = (std::getenv("DUNE_DISABLE_SLOT_REORDER") == nullptr);
    const bool enable_prewait_h2d = (std::getenv("DUNE_DISABLE_PREWAIT_H2D") == nullptr);
    const uint64_t cap_stats_stride = (std::getenv("DUNE_DISABLE_CAP_SUBSAMPLE") != nullptr) ? 1ULL : 64ULL;
    const bool enable_selective_logits_d2h = (std::getenv("DUNE_DISABLE_SELECTIVE_LOGITS_D2H") == nullptr);
    const bool enable_scorer_inplace = (std::getenv("DUNE_DISABLE_SCORER_INPLACE") == nullptr);
    const bool enable_tail_graph = (std::getenv("DUNE_DISABLE_TAIL_GRAPH") == nullptr);
    const c10::DeviceIndex device_index =
        device_.has_index() ? device_.index() : c10::cuda::current_device();
    std::array<AsyncBatchSlot, 2> slots;
    for (AsyncBatchSlot& slot : slots) {
        slot.stream = std::make_unique<c10::cuda::CUDAStream>(
            c10::cuda::getStreamFromPool(high_priority_stream_,
                                         device_index));
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
        slot.graph_pool = at::cuda::graph_pool_handle();
        slot.captured_graphs.resize(target_batch_size_ + 1);
    }
    int pending_slot = -1;
    int next_slot = 0;
    std::vector<std::unique_ptr<c10::Event>> batch_end_events;
    uint64_t next_batch_seq = 0;
    uint64_t compact_stats_counter = 0;
    uint64_t local_graph_gen = graph_generation_.load(std::memory_order_acquire);

    auto finish_slot = [&](AsyncBatchSlot& slot) {
        if (!slot.pending) return;
        WaitAsyncSlotGpu(slot, batch_end_events);
        DistributeAsyncSlotCpu(slot);
    };

    for (;;) {
        const uint64_t target_graph_gen = graph_generation_.load(std::memory_order_acquire);
        if (local_graph_gen != target_graph_gen) {
            if (pending_slot >= 0) {
                finish_slot(slots[pending_slot]);
                pending_slot = -1;
            }
            for (AsyncBatchSlot& s : slots) {
                if (s.stream) s.stream->synchronize();
                s.captured_graphs.clear();
                s.graph_pool = at::cuda::graph_pool_handle();
                s.captured_graphs.resize(target_batch_size_ + 1);
            }
            local_graph_gen = target_graph_gen;
        }

        if (warmup_pending_) {
            if (pending_slot >= 0) {
                finish_slot(slots[pending_slot]);
                pending_slot = -1;
            }
            std::vector<int> sizes_to_warmup;
            {
                std::lock_guard<std::mutex> lock(mutex_);
                sizes_to_warmup = warmup_requested_sizes_;
                warmup_requested_sizes_.clear();
            }
            for (AsyncBatchSlot& s : slots) {
                c10::cuda::CUDAStreamGuard stream_guard(*s.stream);
                for (int b : sizes_to_warmup) {
                    if (b >= static_cast<int>(s.captured_graphs.size())) {
                        s.captured_graphs.resize(b + 1);
                    }
                    if (b > s.device_obs.size(0)) {
                        auto dev_opts = torch::TensorOptions().dtype(torch::kFloat32).device(device_);
                        s.device_obs = torch::empty({static_cast<int64_t>(b), model_input_dim_}, dev_opts);
                        s.captured_graphs.clear();
                        s.graph_pool = at::cuda::graph_pool_handle();
                        s.captured_graphs.resize(b + 1);
                    }
                    auto& cg = s.captured_graphs[b];
                    if (!cg.graph) {
                        auto device_obs = s.device_obs.narrow(0, 0, b);
                        device_obs.zero_();
                        std::shared_lock<std::shared_mutex> lock(*sync_mutex_);
                        CaptureForwardGraphIfNeeded(cg, device_obs, b, s.graph_pool, enable_tail_graph);
                    }
                }
                s.stream->synchronize();
            }
            {
                std::lock_guard<std::mutex> lock(mutex_);
                warmup_pending_ = false;
            }
            warmup_cv_.notify_all();
            continue;
        }

        if (enable_slot_reorder && pending_slot >= 0) {
            bool drain_pending_now = false;
            {
                std::unique_lock<std::mutex> lock(mutex_);
                while (!stop_ && !warmup_pending_ &&
                       requests_.size() < EffectiveTargetBatchSize() &&
                       !slots[pending_slot].complete->query()) {
                    cv_.wait_for(lock, std::chrono::microseconds(50));
                }
                drain_pending_now =
                    !stop_ && !warmup_pending_ && requests_.size() < EffectiveTargetBatchSize();
            }
            if (drain_pending_now) {
                finish_slot(slots[pending_slot]);
                pending_slot = -1;
            }
        }
        bool timeout_flush = false;
        const bool stage_timers_enabled = async_stage_timers_enabled_.load(std::memory_order_relaxed);
        std::vector<Request> batch = TakeBatchFromQueue(
            &timeout_flush, /*has_pending_slot=*/(pending_slot >= 0),
            /*check_warmup=*/true, stage_timers_enabled);
        if (batch.empty()) {
            if (warmup_pending_) continue;
            break;
        }
        if (HandleBudgetRejection(batch)) {
            continue;
        }
        const int slot_index = next_slot;
        AsyncBatchSlot& slot = slots[slot_index];
        if (slot.pending) finish_slot(slot);
        slot.batch = std::move(batch);
        if (slot.batch.size() > static_cast<size_t>(target_batch_size_)) {
            SpielFatalError("BatchedEvaluator::RunnerAsync physical batch size " +
                            std::to_string(slot.batch.size()) +
                            " exceeds configured target " +
                            std::to_string(target_batch_size_));
        }
        if (emit_batch_membership_) {
            slot.membership_batch_id =
                next_membership_batch_id_.fetch_add(
                    1, std::memory_order_relaxed);
        }
        RecordBatchDispatchTelemetry(slot.batch, timeout_flush);
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
        const bool pack_timers_enabled = async_stage_timers_enabled_.load(std::memory_order_relaxed);
        auto t_pack_start = pack_timers_enabled ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
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
        if (pack_timers_enabled) {
            auto t_pack_end = std::chrono::steady_clock::now();
            cpu_pack_us_.fetch_add(
                static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(
                    t_pack_end - t_pack_start).count()),
                std::memory_order_relaxed);
        }
        // Complete the prior stream before launching the next shared-model
        // forward. The two slots still overlap CPU batch assembly with the
        // prior GPU batch, while the event wait protects host outputs and
        // avoids whole-device synchronization. Concurrent forwards on the
        // shared LibTorch module are deliberately not enabled.
        if (enable_slot_reorder) {
            if (enable_prewait_h2d) {
                const size_t batch_size = slot.batch.size();
                if (!slot.device_obs.defined() || batch_size > static_cast<size_t>(slot.device_obs.size(0))) {
                    auto dev_opts = torch::TensorOptions().dtype(torch::kFloat32).device(device_);
                    const int64_t alloc_rows = std::max<int64_t>(target_batch_size_, batch_size);
                    slot.device_obs = torch::empty({alloc_rows, model_input_dim_}, dev_opts);
                    slot.captured_graphs.clear();
                    slot.captured_graphs.resize(batch_size + 1);
                }
                if (pack_timers_enabled) {
                    EnsureAsyncSlotStageEvents(slot, batch_end_events);
                }
                {
                    c10::cuda::CUDAStreamGuard stream_guard(*slot.stream);
                    if (pack_timers_enabled && slot.ev_h2d_start) {
                        slot.ev_h2d_start->record(*slot.stream);
                    }
                    slot.device_obs.narrow(0, 0, batch_size).copy_(
                        slot.host_obs.narrow(0, 0, batch_size), /*non_blocking=*/true);
                    if (pack_timers_enabled && slot.ev_h2d_end) {
                        slot.ev_h2d_end->record(*slot.stream);
                    }
                }
                slot.h2d_launched = true;
            }
            if (pending_slot >= 0) {
                WaitAsyncSlotGpu(slots[pending_slot], batch_end_events);
            }
            try {
                LaunchAsyncSlot(slot, pack_timers_enabled,
                                enable_selective_logits_d2h,
                                enable_scorer_inplace, enable_tail_graph,
                                cap_stats_stride, next_batch_seq,
                                compact_stats_counter, batch_end_events);
            } catch (...) {
                slot.h2d_launched = false;
                if (pending_slot >= 0) {
                    DistributeAsyncSlotCpu(slots[pending_slot]);
                    pending_slot = -1;
                }
                throw;
            }
            if (pending_slot >= 0) {
                DistributeAsyncSlotCpu(slots[pending_slot]);
                pending_slot = -1;
            }
        } else {
            if (pending_slot >= 0) {
                finish_slot(slots[pending_slot]);
                pending_slot = -1;
            }
            LaunchAsyncSlot(slot,
                            async_stage_timers_enabled_.load(std::memory_order_relaxed),
                            enable_selective_logits_d2h,
                            enable_scorer_inplace, enable_tail_graph,
                            cap_stats_stride, next_batch_seq,
                            compact_stats_counter, batch_end_events);
        }
        if (enable_slot_reorder) {
            pending_slot = slot_index;
        } else {
            bool has_ready_next_batch = false;
            {
                std::lock_guard<std::mutex> queue_lock(mutex_);
                has_ready_next_batch =
                    !requests_.empty() &&
                    requests_.size() >= EffectiveTargetBatchSize();
            }
            if (!has_ready_next_batch) {
                // Unless a full next batch is already queued to overlap CPU
                // staging with the GPU forward pass, finish this slot as soon
                // as its CUDA event completes so blocked rollout threads wake
                // immediately instead of stalling behind take_batch() timeout.
                finish_slot(slot);
                pending_slot = -1;
            } else {
                pending_slot = slot_index;
            }
        }
        next_slot = 1 - next_slot;
    }
    if (pending_slot >= 0) finish_slot(slots[pending_slot]);
    if (!clear_autocast_cache_per_batch_) {
        at::autocast::clear_cache();
    }
}

}  // namespace open_spiel

#endif  // OPEN_SPIEL_EXAMPLES_DUNE_NETWORK_EVAL_BATCHED_ASYNC_IMPL_H_
