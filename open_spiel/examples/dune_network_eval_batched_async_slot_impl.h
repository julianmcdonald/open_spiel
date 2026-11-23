#ifndef OPEN_SPIEL_EXAMPLES_DUNE_NETWORK_EVAL_BATCHED_ASYNC_SLOT_IMPL_H_
#define OPEN_SPIEL_EXAMPLES_DUNE_NETWORK_EVAL_BATCHED_ASYNC_SLOT_IMPL_H_

#include <ATen/cuda/CUDAGraph.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/torch.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <limits>
#include <memory>
#include <shared_mutex>
#include <vector>

#include "dune_hotspot_profile.h"
#include "dune_network_eval_batched.h"
#include "dune_network_helpers.h"
#include "dune_network_model.h"
#include "dune_semantic_action_scorer.h"

namespace open_spiel {

inline void BatchedEvaluator::WaitAsyncSlotGpu(
    AsyncBatchSlot& slot,
    std::vector<std::unique_ptr<c10::Event>>& batch_end_events) {
    if (!slot.pending) return;
    const bool stage_timers_enabled = async_stage_timers_enabled_.load(std::memory_order_relaxed);
    auto t_sync_start = stage_timers_enabled ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    slot.complete->synchronize();
    if (stage_timers_enabled) {
        auto t_sync_end = std::chrono::steady_clock::now();
        sync_us_.fetch_add(
            static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(
                t_sync_end - t_sync_start).count()),
            std::memory_order_relaxed);
    }
    if (slot.has_cap_stats) {
        const double* cap = slot.host_cap_stats.data_ptr<double>();
        RecordLogitCapStats(cap[0], cap[1], static_cast<int64_t>(cap[2]),
                            static_cast<int64_t>(cap[3]));
        slot.has_cap_stats = false;
    }
    if (slot.has_stage_timing) {
        slot.has_stage_timing = false;
        if (slot.seq > 0 && batch_end_events.size() == 4) {
            const size_t prev_idx = (slot.seq - 1) % 4;
            if (batch_end_events[prev_idx] && slot.ev_fwd_start &&
                batch_end_events[prev_idx]->was_marked_for_recording() &&
                slot.ev_fwd_start->was_marked_for_recording()) {
                batch_end_events[prev_idx]->synchronize();
                double gap_ms = std::max(0.0, static_cast<double>(
                    batch_end_events[prev_idx]->elapsedTime(*slot.ev_fwd_start)));
                inter_slot_idle_gap_us_.fetch_add(
                    static_cast<uint64_t>(gap_ms * 1000.0 + 0.5), std::memory_order_relaxed);
                inter_slot_gap_count_.fetch_add(1, std::memory_order_relaxed);
            }
        }
        if (slot.ev_h2d_start && slot.ev_h2d_end &&
            slot.ev_fwd_start && slot.ev_trunk_end &&
            slot.ev_scorer_end && slot.ev_post_cap_end && slot.ev_d2h_end &&
            slot.ev_h2d_start->was_marked_for_recording() &&
            slot.ev_h2d_end->was_marked_for_recording() &&
            slot.ev_fwd_start->was_marked_for_recording() &&
            slot.ev_trunk_end->was_marked_for_recording() &&
            slot.ev_scorer_end->was_marked_for_recording() &&
            slot.ev_post_cap_end->was_marked_for_recording() &&
            slot.ev_d2h_end->was_marked_for_recording()) {
            slot.ev_d2h_end->synchronize();
            double h2d_ms = std::max(0.0, static_cast<double>(slot.ev_h2d_start->elapsedTime(*slot.ev_h2d_end)));
            double trunk_ms = std::max(0.0, static_cast<double>(slot.ev_fwd_start->elapsedTime(*slot.ev_trunk_end)));
            double scorer_ms = std::max(0.0, static_cast<double>(slot.ev_trunk_end->elapsedTime(*slot.ev_scorer_end)));
            double fwd_ms = std::max(0.0, static_cast<double>(slot.ev_fwd_start->elapsedTime(*slot.ev_scorer_end)));
            double post_cap_ms = std::max(0.0, static_cast<double>(slot.ev_scorer_end->elapsedTime(*slot.ev_post_cap_end)));
            double d2h_ms = std::max(0.0, static_cast<double>(slot.ev_post_cap_end->elapsedTime(*slot.ev_d2h_end)));

            auto us = [](double ms) {
                return static_cast<uint64_t>(ms * 1000.0 + 0.5);
            };
            h2d_us_.fetch_add(us(h2d_ms), std::memory_order_relaxed);
            trunk_us_.fetch_add(us(trunk_ms), std::memory_order_relaxed);
            scorer_us_.fetch_add(us(scorer_ms), std::memory_order_relaxed);
            forward_us_.fetch_add(us(fwd_ms), std::memory_order_relaxed);
            post_cap_softmax_us_.fetch_add(us(post_cap_ms), std::memory_order_relaxed);
            d2h_us_.fetch_add(us(d2h_ms), std::memory_order_relaxed);
            device_timed_batches_.fetch_add(1, std::memory_order_relaxed);
        }
    }
}

inline void BatchedEvaluator::DistributeAsyncSlotCpu(AsyncBatchSlot& slot) {
    if (!slot.pending) return;
    const bool stage_timers_enabled = async_stage_timers_enabled_.load(std::memory_order_relaxed);
    auto t_dist_start = stage_timers_enabled ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    const float* values = slot.host_values.data_ptr<float>();
    const size_t action_dim = static_cast<size_t>(model_action_dim_);
    dune_hotspot::ScopedTimer hotspot_timer(
        dune_hotspot::Kind::kBatcherResultDistribution);
    size_t compact_cursor = 0;
    size_t logits_cursor = 0;
    const float* logits = slot.host_logits.defined() ? slot.host_logits.data_ptr<float>() : nullptr;

    for (size_t i = 0; i < slot.batch.size(); ++i) {
        Request& request = slot.batch[i];
        if (request.result_dest != nullptr) {
            request.result_dest->value = values[i];
            if (request.wants_logits && logits != nullptr) {
                const size_t offset = slot.selective_logits_active
                    ? (logits_cursor++ * action_dim)
                    : (i * action_dim);
                request.result_dest->logits.assign(
                    logits + offset,
                    logits + offset + action_dim);
            }
        }
        if (request.compact_dest != nullptr) {
            const size_t compact_index = compact_cursor++;
            const float* probs = slot.compact_host_probs.data_ptr<float>();
            request.compact_dest->actions = request.legal_actions;
            request.compact_dest->probabilities.assign(
                probs + compact_index * slot.compact_max_legal,
                probs + compact_index * slot.compact_max_legal +
                    request.legal_actions.size());
            request.compact_dest->value = values[i];
        }
        if (emit_batch_membership_) {
            if (request.result_dest != nullptr) {
                request.result_dest->physical_batch_id =
                    slot.membership_batch_id;
                request.result_dest->physical_batch_size =
                    static_cast<int32_t>(slot.batch.size());
                request.result_dest->physical_batch_row =
                    static_cast<int32_t>(i);
            }
            if (request.compact_dest != nullptr) {
                request.compact_dest->physical_batch_id =
                    slot.membership_batch_id;
                request.compact_dest->physical_batch_size =
                    static_cast<int32_t>(slot.batch.size());
                request.compact_dest->physical_batch_row =
                    static_cast<int32_t>(i);
            }
        }
    }

    const int G = static_cast<int>(slot.greedy_request_indices.size());
    if (G > 0) {
        greedy_action_calls_.fetch_add(G, std::memory_order_relaxed);
        const int k_top = slot.greedy_k_top;
        const float* vals = slot.greedy_host_top_vals.data_ptr<float>();
        const int64_t* idxs = slot.greedy_host_top_idx.data_ptr<int64_t>();
        std::vector<int> tied_g_indices;
        for (int g = 0; g < G; ++g) {
            Request& req = slot.batch[slot.greedy_request_indices[g]];
            if (req.legal_actions.size() <= 1) {
                Action act = req.legal_actions.empty() ? kInvalidAction : req.legal_actions.front();
                if (req.greedy_action_dest != nullptr) *req.greedy_action_dest = act;
                if (req.greedy_result_dest != nullptr) {
                    req.greedy_result_dest->action = act;
                    req.greedy_result_dest->ok = true;
                }
            } else {
                const float z1 = vals[g * k_top + 0];
                const float z2 = vals[g * k_top + 1];
                const int64_t idx1 = idxs[g * k_top + 0];
                const float gap = z1 - z2;
                if (gap > 1e-5f && std::isfinite(z1) && idx1 >= 0 &&
                    static_cast<size_t>(idx1) < req.legal_actions.size()) {
                    Action act = req.legal_actions[static_cast<size_t>(idx1)];
                    if (req.greedy_action_dest != nullptr) *req.greedy_action_dest = act;
                    if (req.greedy_result_dest != nullptr) {
                        req.greedy_result_dest->action = act;
                        req.greedy_result_dest->ok = true;
                    }
                } else {
                    tied_g_indices.push_back(g);
                }
            }
        }
        if (!tied_g_indices.empty()) {
            const size_t T = tied_g_indices.size();
            greedy_action_ties_.fetch_add(T, std::memory_order_relaxed);
            c10::cuda::CUDAStreamGuard stream_guard(*slot.stream);
            torch::InferenceMode inf_guard;
            std::vector<int64_t> tied_rows_vec(T);
            for (size_t t = 0; t < T; ++t) {
                tied_rows_vec[t] = tied_g_indices[t];
            }
            auto tied_rows_dev = torch::tensor(tied_rows_vec, torch::TensorOptions().dtype(torch::kInt64).device(device_));
            auto tied_logits = slot.greedy_last_logits.index_select(0, tied_rows_dev);
            auto tied_mask = slot.greedy_last_mask.index_select(0, tied_rows_dev);
            auto tied_probs = torch::softmax(tied_logits.masked_fill(tied_mask.logical_not(), -1e9f), 1);
            const int64_t tie_total = static_cast<int64_t>(T) * slot.greedy_max_legal;
            if (!slot.greedy_host_tie_probs.defined() || slot.greedy_host_tie_probs.numel() < tie_total) {
                auto host_f32_opts = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU).pinned_memory(true);
                slot.greedy_host_tie_probs = torch::empty({tie_total}, host_f32_opts);
            }
            auto host_tie_slice = slot.greedy_host_tie_probs.narrow(0, 0, tie_total).view({static_cast<int64_t>(T), slot.greedy_max_legal});
            host_tie_slice.copy_(tied_probs, /*non_blocking=*/true);
            slot.stream->synchronize();
            d2h_bytes_.fetch_add(T * slot.greedy_max_legal * sizeof(float), std::memory_order_relaxed);
            const float* tie_probs_ptr = host_tie_slice.data_ptr<float>();
            for (size_t t = 0; t < T; ++t) {
                int g = tied_g_indices[t];
                Request& req = slot.batch[slot.greedy_request_indices[g]];
                const float* row_probs = tie_probs_ptr + t * slot.greedy_max_legal;
                float best_p = -1.0f;
                Action best_a = req.legal_actions.front();
                for (size_t j = 0; j < req.legal_actions.size(); ++j) {
                    if (row_probs[j] > best_p) {
                        best_p = row_probs[j];
                        best_a = req.legal_actions[j];
                    }
                }
                if (req.greedy_action_dest != nullptr) *req.greedy_action_dest = best_a;
                if (req.greedy_result_dest != nullptr) {
                    req.greedy_result_dest->action = best_a;
                    req.greedy_result_dest->ok = true;
                }
            }
        }
    }

    for (size_t i = 0; i < slot.batch.size(); ++i) {
        SignalReadyFlag(slot.batch[i].ready_flag);
    }
    slot.batch.clear();
    slot.pending = false;
    slot.membership_batch_id = -1;
    slot.greedy_last_logits = torch::Tensor();
    slot.greedy_last_mask = torch::Tensor();
    if (stage_timers_enabled) {
        auto t_dist_end = std::chrono::steady_clock::now();
        cpu_distribute_us_.fetch_add(
            static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(
                t_dist_end - t_dist_start).count()),
            std::memory_order_relaxed);
    }
}

inline void BatchedEvaluator::EnsureAsyncSlotStageEvents(
    AsyncBatchSlot& s,
    std::vector<std::unique_ptr<c10::Event>>& batch_end_events) {
    if (!s.ev_h2d_start) {
        s.ev_h2d_start = std::make_unique<c10::Event>(c10::DeviceType::CUDA, c10::EventFlag::BACKEND_DEFAULT);
        s.ev_h2d_end = std::make_unique<c10::Event>(c10::DeviceType::CUDA, c10::EventFlag::BACKEND_DEFAULT);
        s.ev_fwd_start = std::make_unique<c10::Event>(c10::DeviceType::CUDA, c10::EventFlag::BACKEND_DEFAULT);
        s.ev_trunk_end = std::make_unique<c10::Event>(c10::DeviceType::CUDA, c10::EventFlag::BACKEND_DEFAULT);
        s.ev_scorer_end = std::make_unique<c10::Event>(c10::DeviceType::CUDA, c10::EventFlag::BACKEND_DEFAULT);
        s.ev_post_cap_end = std::make_unique<c10::Event>(c10::DeviceType::CUDA, c10::EventFlag::BACKEND_DEFAULT);
        s.ev_d2h_end = std::make_unique<c10::Event>(c10::DeviceType::CUDA, c10::EventFlag::BACKEND_DEFAULT);
    }
    if (batch_end_events.empty()) {
        for (int e = 0; e < 4; ++e) {
            batch_end_events.push_back(std::make_unique<c10::Event>(
                c10::DeviceType::CUDA, c10::EventFlag::BACKEND_DEFAULT));
        }
    }
}

inline void BatchedEvaluator::LaunchAsyncSlot(
    AsyncBatchSlot& slot,
    bool stage_timers_enabled,
    bool enable_selective_logits_d2h,
    bool enable_scorer_inplace,
    bool enable_tail_graph,
    uint64_t cap_stats_stride,
    uint64_t& next_batch_seq,
    uint64_t& compact_stats_counter,
    std::vector<std::unique_ptr<c10::Event>>& batch_end_events) {
    const size_t batch_size = slot.batch.size();
    if (stage_timers_enabled) {
        EnsureAsyncSlotStageEvents(slot, batch_end_events);
        slot.seq = next_batch_seq++;
        slot.has_stage_timing = true;
    } else {
        slot.has_stage_timing = false;
    }

    slot.wants_logits = false;
    slot.compact_request_indices.clear();
    slot.compact_max_legal = 0;
    slot.greedy_request_indices.clear();
    slot.greedy_max_legal = 0;
    slot.logits_request_indices.clear();

    for (size_t i = 0; i < batch_size; ++i) {
        if (slot.batch[i].wants_logits) {
            slot.wants_logits = true;
            if (slot.batch[i].result_dest != nullptr) {
                slot.logits_request_indices.push_back(i);
            }
        }
        if (slot.batch[i].compact_dest != nullptr) {
            slot.compact_request_indices.push_back(i);
            slot.compact_max_legal = std::max(
                slot.compact_max_legal,
                static_cast<int>(slot.batch[i].legal_actions.size()));
        }
        if (slot.batch[i].greedy_action_dest != nullptr) {
            slot.greedy_request_indices.push_back(i);
            slot.greedy_max_legal = std::max(
                slot.greedy_max_legal,
                static_cast<int>(slot.batch[i].legal_actions.size()));
        }
    }
    const size_t k_logits = slot.logits_request_indices.size();
    if (k_logits > 0) {
        wants_logits_batches_.fetch_add(1, std::memory_order_relaxed);
        wants_logits_rows_.fetch_add(k_logits, std::memory_order_relaxed);
    }

    if (!slot.device_obs.defined()) {
        auto host_options = torch::TensorOptions()
            .dtype(torch::kFloat32).device(torch::kCPU)
            .pinned_memory(true);
        auto device_options = torch::TensorOptions()
            .dtype(torch::kFloat32).device(device_);
        if (!slot.host_obs.defined()) {
            slot.host_obs = torch::empty(
                {target_batch_size_, model_input_dim_}, host_options);
        }
        slot.device_obs = torch::empty(
            {target_batch_size_, model_input_dim_}, device_options);
        slot.device_values = torch::empty(
            {target_batch_size_}, device_options);
        slot.host_values = torch::empty(
            {target_batch_size_}, host_options);
        slot.complete = std::make_unique<c10::Event>(
            c10::DeviceType::CUDA, c10::EventFlag::BACKEND_DEFAULT);
    }
    if (batch_size >= slot.captured_graphs.size()) {
        slot.captured_graphs.resize(batch_size + 1);
    }
    if (batch_size > static_cast<size_t>(slot.device_obs.size(0))) {
        auto device_options = torch::TensorOptions()
            .dtype(torch::kFloat32).device(device_);
        slot.device_obs = torch::empty(
            {static_cast<int64_t>(batch_size), model_input_dim_}, device_options);
        slot.captured_graphs.clear();
        slot.graph_pool = at::cuda::graph_pool_handle();
        slot.captured_graphs.resize(batch_size + 1);
    }
    {
        c10::cuda::CUDAStreamGuard stream_guard(*slot.stream);
        if (!slot.h2d_launched) {
            if (stage_timers_enabled && slot.ev_h2d_start) {
                slot.ev_h2d_start->record(*slot.stream);
            }
            auto host_obs = slot.host_obs.narrow(0, 0, batch_size);
            auto device_obs = slot.device_obs.narrow(0, 0, batch_size);
            device_obs.copy_(host_obs, /*non_blocking=*/true);
            if (stage_timers_enabled && slot.ev_h2d_end) {
                slot.ev_h2d_end->record(*slot.stream);
            }
        }
        slot.h2d_launched = false;

        if (stage_timers_enabled && slot.ev_fwd_start) {
            slot.ev_fwd_start->record(*slot.stream);
        }

        auto device_obs = slot.device_obs.narrow(0, 0, batch_size);
        SharedDunePolicyValueNetImpl::ModelOutputs outputs;
        SharedDunePolicyValueNetImpl::ModelOutputs critic_outputs;
        torch::Tensor values_f32_out;
        {
            std::shared_lock<std::shared_mutex> lock(*sync_mutex_);
            auto& cg = slot.captured_graphs[batch_size];
            CaptureForwardGraphIfNeeded(cg, device_obs, batch_size, slot.graph_pool, enable_tail_graph);
            cg.graph->replay();
            outputs = cg.outputs;
            critic_outputs = cg.critic_outputs;
            values_f32_out = cg.values_f32;

            if (stage_timers_enabled && slot.ev_trunk_end) {
                slot.ev_trunk_end->record(*slot.stream);
            }

            if (model_->with_semantic_scorer_ && model_->semantic_scorer_) {
                std::vector<const dune_semantic::CandidateActionData*> batch_actions(batch_size);
                bool has_any_actions = false;
                for (size_t i = 0; i < batch_size; ++i) {
                    batch_actions[i] = slot.batch[i].action_data;
                    if (slot.batch[i].action_data != nullptr && !slot.batch[i].action_data->empty()) {
                        has_any_actions = true;
                    }
                }
                if (has_any_actions) {
                    AutocastGuard autocast_guard(c10::DeviceType::CUDA,
                                                 rollout_amp_,
                                                 clear_autocast_cache_per_batch_);
                    dune_semantic::ApplySemanticScorerBatch(
                        model_->semantic_scorer_, outputs.trunk, batch_actions, outputs.logits, device_,
                        /*workspace=*/nullptr,
                        /*allow_inplace=*/enable_scorer_inplace,
                        /*precomputed_trunk_proj=*/(enable_tail_graph ? cg.projected_trunk : torch::Tensor()));
                }
            }
            if (stage_timers_enabled && slot.ev_scorer_end) {
                slot.ev_scorer_end->record(*slot.stream);
            }
        }

        if (enable_tail_graph && values_f32_out.defined()) {
            slot.host_values.narrow(0, 0, batch_size).copy_(
                values_f32_out, /*non_blocking=*/true);
        } else {
            auto device_values = slot.device_values.narrow(
                0, 0, batch_size);
            device_values.copy_(critic_outputs.values.reshape(
                                    {static_cast<int64_t>(batch_size)}),
                                /*non_blocking=*/true);
            slot.host_values.narrow(0, 0, batch_size).copy_(
                device_values, /*non_blocking=*/true);
        }

        if (k_logits > 0) {
            if (enable_selective_logits_d2h && k_logits < batch_size) {
                if (!slot.logits_host_rows.defined() ||
                    slot.logits_host_rows.numel() < static_cast<int64_t>(k_logits)) {
                    const int64_t alloc_k = std::max<int64_t>(target_batch_size_, k_logits);
                    auto host_i64 = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU).pinned_memory(true);
                    auto dev_i64 = torch::TensorOptions().dtype(torch::kInt64).device(device_);
                    slot.logits_host_rows = torch::empty({alloc_k}, host_i64);
                    slot.logits_device_rows = torch::empty({alloc_k}, dev_i64);
                }
                int64_t* row_ptrs = slot.logits_host_rows.data_ptr<int64_t>();
                for (size_t k = 0; k < k_logits; ++k) {
                    row_ptrs[k] = static_cast<int64_t>(slot.logits_request_indices[k]);
                }
                auto dev_rows = slot.logits_device_rows.narrow(0, 0, k_logits);
                dev_rows.copy_(slot.logits_host_rows.narrow(0, 0, k_logits), /*non_blocking=*/true);

                auto selected_logits = outputs.logits.index_select(0, dev_rows);
                if (!slot.device_logits.defined() || slot.device_logits.size(0) < static_cast<int64_t>(k_logits)) {
                    const int64_t alloc_rows = std::max<int64_t>(target_batch_size_, k_logits);
                    auto dev_f32 = torch::TensorOptions().dtype(torch::kFloat32).device(device_);
                    auto host_f32 = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU).pinned_memory(true);
                    slot.device_logits = torch::empty({alloc_rows, model_action_dim_}, dev_f32);
                    slot.host_logits = torch::empty({alloc_rows, model_action_dim_}, host_f32);
                }
                auto dev_log_slice = slot.device_logits.narrow(0, 0, k_logits);
                dev_log_slice.copy_(selected_logits, /*non_blocking=*/true);
                slot.host_logits.narrow(0, 0, k_logits).copy_(dev_log_slice, /*non_blocking=*/true);
                slot.selective_logits_active = true;
            } else {
                if (!slot.device_logits.defined() || slot.device_logits.size(0) < static_cast<int64_t>(batch_size)) {
                    const int64_t alloc_rows = std::max<int64_t>(target_batch_size_, batch_size);
                    auto dev_f32 = torch::TensorOptions().dtype(torch::kFloat32).device(device_);
                    auto host_f32 = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU).pinned_memory(true);
                    slot.device_logits = torch::empty({alloc_rows, model_action_dim_}, dev_f32);
                    slot.host_logits = torch::empty({alloc_rows, model_action_dim_}, host_f32);
                }
                auto dev_log_slice = slot.device_logits.narrow(0, 0, batch_size);
                dev_log_slice.copy_(outputs.logits, /*non_blocking=*/true);
                slot.host_logits.narrow(0, 0, batch_size).copy_(dev_log_slice, /*non_blocking=*/true);
                slot.selective_logits_active = false;
            }
        }

        if (!slot.compact_request_indices.empty()) {
            const int compact_count =
                static_cast<int>(slot.compact_request_indices.size());
            const int max_legal = slot.compact_max_legal;
            const int64_t total_elems = static_cast<int64_t>(compact_count) * max_legal;
            if (!slot.compact_host_ids.defined() ||
                slot.compact_host_ids.numel() < total_elems ||
                !slot.compact_host_rows.defined() ||
                slot.compact_host_rows.numel() < compact_count) {
                const int64_t alloc_rows = std::max<int64_t>(target_batch_size_, compact_count);
                const int64_t alloc_cols = std::max<int64_t>(max_legal, 128);
                const int64_t alloc_total = alloc_rows * alloc_cols;
                auto host_i64_opts = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU).pinned_memory(true);
                auto host_bool_opts = torch::TensorOptions().dtype(torch::kBool).device(torch::kCPU).pinned_memory(true);
                auto host_f32_opts = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU).pinned_memory(true);
                auto dev_i64_opts = torch::TensorOptions().dtype(torch::kInt64).device(device_);
                auto dev_bool_opts = torch::TensorOptions().dtype(torch::kBool).device(device_);

                slot.compact_host_ids = torch::empty({alloc_total}, host_i64_opts);
                slot.compact_host_mask = torch::empty({alloc_total}, host_bool_opts);
                slot.compact_host_rows = torch::empty({alloc_rows}, host_i64_opts);
                slot.compact_host_probs = torch::empty({alloc_total}, host_f32_opts);
                slot.compact_device_ids = torch::empty({alloc_total}, dev_i64_opts);
                slot.compact_device_mask = torch::empty({alloc_total}, dev_bool_opts);
                slot.compact_device_rows = torch::empty({alloc_rows}, dev_i64_opts);
            }
            auto ids_cpu = slot.compact_host_ids.narrow(0, 0, total_elems).view({compact_count, max_legal});
            auto mask_cpu = slot.compact_host_mask.narrow(0, 0, total_elems).view({compact_count, max_legal});
            auto rows_cpu = slot.compact_host_rows.narrow(0, 0, compact_count);
            ids_cpu.zero_();
            mask_cpu.zero_();
            int64_t* ids = ids_cpu.data_ptr<int64_t>();
            bool* mask = mask_cpu.data_ptr<bool>();
            int64_t* rows = rows_cpu.data_ptr<int64_t>();
            for (int k = 0; k < compact_count; ++k) {
                rows[k] = static_cast<int64_t>(slot.compact_request_indices[k]);
                const Request& request = slot.batch[slot.compact_request_indices[k]];
                for (size_t j = 0; j < request.legal_actions.size(); ++j) {
                    ids[k * max_legal + j] = request.legal_actions[j];
                    mask[k * max_legal + j] = true;
                }
            }
            auto dev_ids = slot.compact_device_ids.narrow(0, 0, total_elems).view({compact_count, max_legal});
            auto dev_mask = slot.compact_device_mask.narrow(0, 0, total_elems).view({compact_count, max_legal});
            auto dev_rows = slot.compact_device_rows.narrow(0, 0, compact_count);
            dev_ids.copy_(ids_cpu, /*non_blocking=*/true);
            dev_mask.copy_(mask_cpu, /*non_blocking=*/true);
            dev_rows.copy_(rows_cpu, /*non_blocking=*/true);

            auto compact_logits = outputs.logits.index_select(
                0, dev_rows)
                .gather(1, dev_ids)
                .to(torch::kFloat32);
            const torch::Tensor legal = dev_mask.to(torch::kFloat32);
            const torch::Tensor legal_count =
                legal.sum(1, true).clamp_min(1.0);
            compact_logits = compact_logits -
                (compact_logits * legal).sum(1, true) / legal_count;

            const bool sample_cap = ((compact_stats_counter++ % cap_stats_stride) == 0);
            if (sample_cap) {
                const torch::Tensor pre_max = compact_logits.abs().masked_fill(
                    dev_mask.logical_not(), 0.0).amax(1);
                torch::Tensor post_capped = compact_logits;
                if (logit_cap_ > 0.0f) {
                    post_capped = logit_cap_ * torch::tanh(compact_logits / logit_cap_);
                }
                const torch::Tensor post_max = post_capped.abs().masked_fill(
                    dev_mask.logical_not(), 0.0).amax(1);
                const torch::Tensor saturated =
                    ((post_capped.abs() >=
                      0.99 * static_cast<float>(logit_cap_)) &
                     dev_mask).sum();
                const torch::Tensor cap_stats = torch::stack(
                    {pre_max.max().to(torch::kFloat64),
                     post_max.max().to(torch::kFloat64),
                     legal.sum().to(torch::kFloat64),
                     saturated.to(torch::kFloat64)});
                if (!slot.host_cap_stats.defined()) {
                    slot.host_cap_stats = torch::empty(
                        {4}, torch::TensorOptions().dtype(torch::kFloat64)
                                 .pinned_memory(true));
                }
                slot.host_cap_stats.copy_(cap_stats, /*non_blocking=*/true);
                slot.has_cap_stats = true;
                if (logit_cap_ > 0.0f) {
                    compact_logits = post_capped;
                }
            } else {
                if (logit_cap_ > 0.0f) {
                    compact_logits = logit_cap_ * torch::tanh(compact_logits / logit_cap_);
                }
            }
            auto compact_probs = torch::softmax(
                compact_logits.masked_fill(
                    dev_mask.logical_not(), -1e9f),
                1);
            auto host_probs_slice = slot.compact_host_probs.narrow(0, 0, total_elems).view({compact_count, max_legal});
            host_probs_slice.copy_(compact_probs, /*non_blocking=*/true);
        }

        if (!slot.greedy_request_indices.empty()) {
            const int greedy_count = static_cast<int>(slot.greedy_request_indices.size());
            const int g_max_legal = std::max(1, slot.greedy_max_legal);
            slot.greedy_max_legal = g_max_legal;
            const int64_t g_total_elems = static_cast<int64_t>(greedy_count) * g_max_legal;
            if (!slot.greedy_host_ids.defined() ||
                slot.greedy_host_ids.numel() < g_total_elems ||
                !slot.greedy_host_rows.defined() ||
                slot.greedy_host_rows.numel() < greedy_count) {
                const int64_t alloc_rows = std::max<int64_t>(target_batch_size_, greedy_count);
                const int64_t alloc_cols = std::max<int64_t>(g_max_legal, 128);
                const int64_t alloc_total = alloc_rows * alloc_cols;
                auto host_i64_opts = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU).pinned_memory(true);
                auto host_bool_opts = torch::TensorOptions().dtype(torch::kBool).device(torch::kCPU).pinned_memory(true);
                auto host_f32_opts = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU).pinned_memory(true);
                auto dev_i64_opts = torch::TensorOptions().dtype(torch::kInt64).device(device_);
                auto dev_bool_opts = torch::TensorOptions().dtype(torch::kBool).device(device_);

                slot.greedy_host_ids = torch::empty({alloc_total}, host_i64_opts);
                slot.greedy_host_mask = torch::empty({alloc_total}, host_bool_opts);
                slot.greedy_host_rows = torch::empty({alloc_rows}, host_i64_opts);
                slot.greedy_device_ids = torch::empty({alloc_total}, dev_i64_opts);
                slot.greedy_device_mask = torch::empty({alloc_total}, dev_bool_opts);
                slot.greedy_device_rows = torch::empty({alloc_rows}, dev_i64_opts);
                slot.greedy_host_top_vals = torch::empty({alloc_rows * 2}, host_f32_opts);
                slot.greedy_host_top_idx = torch::empty({alloc_rows * 2}, host_i64_opts);
                slot.greedy_host_tie_probs = torch::empty({alloc_total}, host_f32_opts);
            }
            auto g_ids_cpu = slot.greedy_host_ids.narrow(0, 0, g_total_elems).view({greedy_count, g_max_legal});
            auto g_mask_cpu = slot.greedy_host_mask.narrow(0, 0, g_total_elems).view({greedy_count, g_max_legal});
            auto g_rows_cpu = slot.greedy_host_rows.narrow(0, 0, greedy_count);
            g_ids_cpu.zero_();
            g_mask_cpu.zero_();
            int64_t* g_ids = g_ids_cpu.data_ptr<int64_t>();
            bool* g_mask = g_mask_cpu.data_ptr<bool>();
            int64_t* g_rows = g_rows_cpu.data_ptr<int64_t>();
            for (int k = 0; k < greedy_count; ++k) {
                g_rows[k] = static_cast<int64_t>(slot.greedy_request_indices[k]);
                const Request& req = slot.batch[slot.greedy_request_indices[k]];
                for (size_t j = 0; j < req.legal_actions.size(); ++j) {
                    g_ids[k * g_max_legal + j] = req.legal_actions[j];
                    g_mask[k * g_max_legal + j] = true;
                }
            }
            auto g_dev_ids = slot.greedy_device_ids.narrow(0, 0, g_total_elems).view({greedy_count, g_max_legal});
            auto g_dev_mask = slot.greedy_device_mask.narrow(0, 0, g_total_elems).view({greedy_count, g_max_legal});
            auto g_dev_rows = slot.greedy_device_rows.narrow(0, 0, greedy_count);
            g_dev_ids.copy_(g_ids_cpu, /*non_blocking=*/true);
            g_dev_mask.copy_(g_mask_cpu, /*non_blocking=*/true);
            g_dev_rows.copy_(g_rows_cpu, /*non_blocking=*/true);

            auto greedy_logits = outputs.logits.index_select(0, g_dev_rows)
                .gather(1, g_dev_ids)
                .to(torch::kFloat32);
            const torch::Tensor g_legal = g_dev_mask.to(torch::kFloat32);
            const torch::Tensor g_legal_count = g_legal.sum(1, true).clamp_min(1.0);
            greedy_logits = greedy_logits - (greedy_logits * g_legal).sum(1, true) / g_legal_count;
            if (logit_cap_ > 0.0f) {
                greedy_logits = logit_cap_ * torch::tanh(greedy_logits / logit_cap_);
            }
            slot.greedy_last_logits = greedy_logits;
            slot.greedy_last_mask = g_dev_mask;

            auto masked_top = greedy_logits.masked_fill(g_dev_mask.logical_not(), -std::numeric_limits<float>::infinity());
            const int k_top = std::min<int>(2, g_max_legal);
            slot.greedy_k_top = k_top;
            auto top_res = torch::topk(masked_top, k_top, /*dim=*/1, /*largest=*/true, /*sorted=*/true);
            auto top_vals = std::get<0>(top_res);
            auto top_idx = std::get<1>(top_res);

            auto host_top_vals_slice = slot.greedy_host_top_vals.narrow(0, 0, greedy_count * k_top).view({greedy_count, k_top});
            auto host_top_idx_slice = slot.greedy_host_top_idx.narrow(0, 0, greedy_count * k_top).view({greedy_count, k_top});
            host_top_vals_slice.copy_(top_vals, /*non_blocking=*/true);
            host_top_idx_slice.copy_(top_idx, /*non_blocking=*/true);
        }

        if (stage_timers_enabled && slot.ev_post_cap_end) {
            slot.ev_post_cap_end->record(*slot.stream);
        }

        uint64_t cur_d2h_bytes = batch_size * sizeof(float);
        if (k_logits > 0) {
            const size_t rows_copied = slot.selective_logits_active ? k_logits : batch_size;
            cur_d2h_bytes += rows_copied * model_action_dim_ * sizeof(float);
        }
        if (!slot.compact_request_indices.empty()) {
            cur_d2h_bytes += static_cast<uint64_t>(slot.compact_request_indices.size()) * slot.compact_max_legal * sizeof(float);
        }
        if (!slot.greedy_request_indices.empty()) {
            cur_d2h_bytes += static_cast<uint64_t>(slot.greedy_request_indices.size()) * slot.greedy_k_top * (sizeof(float) + sizeof(int64_t));
        }
        if (slot.has_cap_stats) {
            cur_d2h_bytes += 4 * sizeof(double);
        }
        d2h_bytes_.fetch_add(cur_d2h_bytes, std::memory_order_relaxed);

        if (stage_timers_enabled && slot.ev_d2h_end) {
            slot.ev_d2h_end->record(*slot.stream);
            if (batch_end_events.size() == 4) {
                batch_end_events[slot.seq % 4]->record(*slot.stream);
            }
        }
        slot.complete->record(*slot.stream);
    }
    slot.pending = true;
}

}  // namespace open_spiel

#endif  // OPEN_SPIEL_EXAMPLES_DUNE_NETWORK_EVAL_BATCHED_ASYNC_SLOT_IMPL_H_
