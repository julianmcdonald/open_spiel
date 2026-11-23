#ifndef OPEN_SPIEL_EXAMPLES_DUNE_NETWORK_EVAL_BATCHED_RUNNER_IMPL_H_
#define OPEN_SPIEL_EXAMPLES_DUNE_NETWORK_EVAL_BATCHED_RUNNER_IMPL_H_

#include <ATen/autocast_mode.h>
#include <ATen/cuda/CUDAGraph.h>
#include <c10/cuda/CUDAFunctions.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/torch.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <optional>
#include <shared_mutex>
#include <vector>

#include "dune_network_eval_batched.h"
#include "dune_network_eval_batched_async_impl.h"
#include "dune_network_helpers.h"
#include "dune_network_model.h"
#include "dune_semantic_action_scorer.h"
#include "open_spiel/spiel.h"

namespace open_spiel {

inline void BatchedEvaluator::Runner() {
    if (device_.is_cuda() && !device_synchronize_) {
        RunnerAsync();
        return;
    }
    const bool enable_scorer_inplace = (std::getenv("DUNE_DISABLE_SCORER_INPLACE") == nullptr);
    const bool enable_tail_graph = (std::getenv("DUNE_DISABLE_TAIL_GRAPH") == nullptr);
    torch::InferenceMode inference_guard;
    torch::Tensor pinned_stacked_obs;
    torch::Tensor pinned_logits_out;
    torch::Tensor pinned_values_out;
    bool pinned_allocated = false;

    torch::Tensor graph_device_obs;
    at::cuda::MempoolId_t graph_pool;
    std::vector<CapturedForwardGraph> captured_graphs;
    uint64_t local_graph_gen = graph_generation_.load(std::memory_order_acquire);
    std::unique_ptr<c10::cuda::CUDAStream> sync_runner_stream;
    if (device_.is_cuda()) {
        const c10::DeviceIndex dev_idx = device_.index();
        sync_runner_stream = std::make_unique<c10::cuda::CUDAStream>(
            c10::cuda::getStreamFromPool(high_priority_stream_, dev_idx));
        graph_device_obs = torch::empty(
            {target_batch_size_, model_input_dim_},
            torch::TensorOptions().dtype(torch::kFloat32).device(device_));
        graph_pool = at::cuda::graph_pool_handle();
        captured_graphs.resize(target_batch_size_ + 1);
    }

    while (true) {
        bool timeout_flush = false;  // WO-PERF-3: dispatched under-target on deadline
        std::vector<Request> batch = TakeBatchFromQueue(
            &timeout_flush, /*has_pending_slot=*/false,
            /*check_warmup=*/false, /*stage_timers_enabled=*/false);
        if (batch.empty()) {
            bool should_stop = false;
            {
                std::lock_guard<std::mutex> lock(mutex_);
                should_stop = stop_ && requests_.empty();
            }
            if (should_stop) break;
            continue;
        }

        if (HandleBudgetRejection(batch)) {
            continue;
        }

        size_t batch_size = batch.size();
        const int64_t membership_batch_id = emit_batch_membership_
            ? next_membership_batch_id_.fetch_add(
                  1, std::memory_order_relaxed)
            : -1;
        RecordBatchDispatchTelemetry(batch, timeout_flush);

        // A. PINNED MEMORY: Allocate exactly ONCE outside the hot loop based on model_input_dim_
        if (!pinned_allocated) {
            if (device_.is_cuda()) {
                auto options = torch::TensorOptions().dtype(torch::kFloat32).pinned_memory(true);
                pinned_stacked_obs = torch::empty({(long)target_batch_size_, (long)model_input_dim_}, options);
                pinned_logits_out = torch::empty({(long)target_batch_size_, (long)model_action_dim_}, options);
                pinned_values_out = torch::empty({(long)target_batch_size_}, options);
            } else {
                pinned_stacked_obs = torch::empty({(long)target_batch_size_, (long)model_input_dim_}, torch::kFloat32);
            }
            pinned_allocated = true;
        }

        if (batch.size() > static_cast<size_t>(target_batch_size_)) {
            SpielFatalError("BatchedEvaluator::Runner physical batch size " +
                            std::to_string(batch.size()) +
                            " exceeds target " +
                            std::to_string(target_batch_size_));
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
        std::optional<c10::cuda::CUDAStreamGuard> stream_guard;
        if (device_.is_cuda()) {
            stream_guard.emplace(*sync_runner_stream);
        }
        std::optional<c10::impl::VirtualGuardImpl> dev_impl;
        if (time_device_) {
            dev_impl.emplace(c10::DeviceType::CUDA);
            dev_ev[0].record(dev_impl->getStream(device_));
        }

        if (device_.is_cuda()) {
            const uint64_t target_graph_gen = graph_generation_.load(std::memory_order_acquire);
            if (local_graph_gen != target_graph_gen) {
                if (sync_runner_stream) sync_runner_stream->synchronize();
                captured_graphs.clear();
                graph_pool = at::cuda::graph_pool_handle();
                local_graph_gen = target_graph_gen;
            }

            if (batch_size >= captured_graphs.size()) {
                captured_graphs.resize(batch_size + 1);
            }
            if (batch_size > static_cast<size_t>(graph_device_obs.size(0))) {
                graph_device_obs = torch::empty(
                    {static_cast<int64_t>(batch_size), model_input_dim_},
                    torch::TensorOptions().dtype(torch::kFloat32).device(device_));
                captured_graphs.clear();
                graph_pool = at::cuda::graph_pool_handle();
                captured_graphs.resize(batch_size + 1);
            }
        }

        // Non-blocking H2D transfer using .slice() for partial batches
        torch::Tensor device_obs;
        if (device_.is_cuda()) {
            auto host_slice = pinned_stacked_obs.slice(0, 0, batch_size);
            device_obs = graph_device_obs.narrow(0, 0, batch_size);
            device_obs.copy_(host_slice, /*non_blocking=*/true);
        } else {
            device_obs = pinned_stacked_obs.slice(0, 0, batch_size);
        }

        if (time_device_) dev_ev[1].record(dev_impl->getStream(device_));

        // B. FORWARD PASS WITH AMP (FP16/TF32) AND WRITE PROTECTION
        torch::Tensor pred_logits, pred_values;
        {
            std::shared_lock<std::shared_mutex> lock(*sync_mutex_);
            SharedDunePolicyValueNetImpl::ModelOutputs outputs;
            SharedDunePolicyValueNetImpl::ModelOutputs critic_outputs;
            torch::Tensor values_f32_out;
            if (device_.is_cuda()) {
                auto& cg = captured_graphs[batch_size];
                CaptureForwardGraphIfNeeded(cg, device_obs, batch_size, graph_pool, enable_tail_graph);
                cg.graph->replay();
                outputs = cg.outputs;
                critic_outputs = cg.critic_outputs;
                values_f32_out = cg.values_f32;
            } else {
                AutocastGuard autocast_guard(c10::DeviceType::CUDA,
                                             false,
                                             clear_autocast_cache_per_batch_);
                outputs = model_->forward(device_obs);
                critic_outputs = (critic_model_ != nullptr) ? critic_model_->forward(device_obs) : outputs;
            }
            if (model_->with_semantic_scorer_ && model_->semantic_scorer_) {
                std::vector<const dune_semantic::CandidateActionData*> batch_actions(batch_size);
                bool has_any_actions = false;
                for (size_t i = 0; i < batch_size; ++i) {
                    batch_actions[i] = batch[i].action_data;
                    if (batch[i].action_data != nullptr && !batch[i].action_data->empty()) {
                        has_any_actions = true;
                    }
                }
                if (has_any_actions) {
                    AutocastGuard autocast_guard(c10::DeviceType::CUDA,
                                                 device_.is_cuda() && rollout_amp_,
                                                 clear_autocast_cache_per_batch_);
                    torch::Tensor precomputed_trunk = (device_.is_cuda() && enable_tail_graph)
                        ? captured_graphs[batch_size].projected_trunk
                        : torch::Tensor();
                    dune_semantic::ApplySemanticScorerBatch(
                        model_->semantic_scorer_, outputs.trunk, batch_actions, outputs.logits, device_,
                        /*workspace=*/nullptr,
                        /*allow_inplace=*/enable_scorer_inplace,
                        /*precomputed_trunk_proj=*/precomputed_trunk);
                }
            }
            pred_logits = outputs.logits;
            pred_values = critic_outputs.values;
        }
        if (time_device_) dev_ev[2].record(dev_impl->getStream(device_));

        // Return raw logits. The tanh soft-cap must be applied after
        // legal-action centering, which requires the State's legal actions.

        // C. NON-BLOCKING D2H TRANSFER (cast AMP FP16 outputs back to Float32)
        if (device_.is_cuda()) {
            pred_logits = pred_logits.to(torch::kFloat32);
            pred_values = pred_values.to(torch::kFloat32).reshape({static_cast<int64_t>(batch_size)});
            // Boundary AFTER the cast: without it, "d2h" silently included
            // the AMP->FP32 conversion, which is device compute, not a
            // transfer.
            if (time_device_) dev_ev[3].record(dev_impl->getStream(device_));
            std::chrono::steady_clock::time_point sync_t0, sync_t1;
            auto host_logits_slice = pinned_logits_out.slice(0, 0, batch_size);
            auto host_values_slice = pinned_values_out.slice(0, 0, batch_size);
            host_logits_slice.copy_(pred_logits, /*non_blocking=*/device_synchronize_);
            host_values_slice.copy_(pred_values, /*non_blocking=*/device_synchronize_);
            if (time_device_) dev_ev[4].record(dev_impl->getStream(device_));
            if (device_synchronize_) {
                // Wait for asynchronous transfers to complete before reading.
                sync_t0 = std::chrono::steady_clock::now();
                sync_runner_stream->synchronize();
                sync_t1 = std::chrono::steady_clock::now();
            } else {
                sync_t0 = sync_t1 = std::chrono::steady_clock::now();
            }
            pred_logits = host_logits_slice;
            pred_values = host_values_slice;
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
            if (batch[i].result_dest != nullptr) {
                batch[i].result_dest->logits.assign(logits_ptr + i * action_dim, logits_ptr + (i + 1) * action_dim);
                batch[i].result_dest->value = values_ptr[i];

                if (emit_batch_membership_) {
                    batch[i].result_dest->physical_batch_id =
                        membership_batch_id;
                    batch[i].result_dest->physical_batch_size =
                        static_cast<int32_t>(batch_size);
                    batch[i].result_dest->physical_batch_row =
                        static_cast<int32_t>(i);
                }
            }

            if (batch[i].compact_dest != nullptr) {
                batch[i].compact_dest->actions = batch[i].legal_actions;
                std::vector<float> row_logits(
                    logits_ptr + i * action_dim,
                    logits_ptr + (i + 1) * action_dim);
                CenterAndCapLegalLogits(row_logits, batch[i].legal_actions, logit_cap_);
                double max_logit = -std::numeric_limits<double>::infinity();
                for (Action a : batch[i].legal_actions) {
                    if (a >= 0 && static_cast<size_t>(a) < row_logits.size()) {
                        max_logit = std::max(max_logit, static_cast<double>(row_logits[a]));
                    }
                }
                double sum_exp = 0.0;
                for (Action a : batch[i].legal_actions) {
                    if (a >= 0 && static_cast<size_t>(a) < row_logits.size()) {
                        sum_exp += std::exp(static_cast<double>(row_logits[a]) - max_logit);
                    }
                }
                batch[i].compact_dest->probabilities.clear();
                batch[i].compact_dest->probabilities.reserve(batch[i].legal_actions.size());
                for (Action a : batch[i].legal_actions) {
                    double prob = (sum_exp > 0.0)
                                      ? (std::exp(static_cast<double>(row_logits[a]) - max_logit) / sum_exp)
                                      : (1.0 / batch[i].legal_actions.size());
                    batch[i].compact_dest->probabilities.push_back(static_cast<float>(prob));
                }
                batch[i].compact_dest->value = values_ptr[i];
                if (emit_batch_membership_) {
                    batch[i].compact_dest->physical_batch_id =
                        membership_batch_id;
                    batch[i].compact_dest->physical_batch_size =
                        static_cast<int32_t>(batch_size);
                    batch[i].compact_dest->physical_batch_row =
                        static_cast<int32_t>(i);
                }
            }

            if (batch[i].greedy_action_dest != nullptr || batch[i].greedy_result_dest != nullptr) {
                greedy_action_calls_.fetch_add(1, std::memory_order_relaxed);
                Action best_a = kInvalidAction;
                if (!batch[i].legal_actions.empty()) {
                    best_a = batch[i].legal_actions.front();
                    if (batch[i].legal_actions.size() > 1) {
                        std::vector<float> row_logits(
                            logits_ptr + i * action_dim,
                            logits_ptr + (i + 1) * action_dim);
                        CenterAndCapLegalLogits(row_logits, batch[i].legal_actions, logit_cap_);
                        float z_max1 = -std::numeric_limits<float>::infinity();
                        float z_max2 = -std::numeric_limits<float>::infinity();
                        Action act1 = batch[i].legal_actions.front();
                        for (Action a : batch[i].legal_actions) {
                            if (a >= 0 && static_cast<size_t>(a) < row_logits.size()) {
                                float val = row_logits[a];
                                if (val > z_max1) {
                                    z_max2 = z_max1;
                                    z_max1 = val;
                                    act1 = a;
                                } else if (val > z_max2) {
                                    z_max2 = val;
                                }
                            }
                        }
                        if (z_max1 - z_max2 > 1e-5f) {
                            best_a = act1;
                        } else {
                            greedy_action_ties_.fetch_add(1, std::memory_order_relaxed);
                            double max_logit = -std::numeric_limits<double>::infinity();
                            for (Action a : batch[i].legal_actions) {
                                if (a >= 0 && static_cast<size_t>(a) < row_logits.size()) {
                                    max_logit = std::max(max_logit, static_cast<double>(row_logits[a]));
                                }
                            }
                            double sum_exp = 0.0;
                            for (Action a : batch[i].legal_actions) {
                                if (a >= 0 && static_cast<size_t>(a) < row_logits.size()) {
                                    sum_exp += std::exp(static_cast<double>(row_logits[a]) - max_logit);
                                }
                            }
                            float best_p = -1.0f;
                            for (Action a : batch[i].legal_actions) {
                                double prob = (sum_exp > 0.0 && !std::isnan(sum_exp))
                                                  ? (std::exp(static_cast<double>(row_logits[a]) - max_logit) / sum_exp)
                                                  : (1.0 / batch[i].legal_actions.size());
                                float fprob = static_cast<float>(prob);
                                if (fprob > best_p) {
                                    best_p = fprob;
                                    best_a = a;
                                }
                            }
                        }
                    }
                }
                if (batch[i].greedy_action_dest != nullptr) {
                    *batch[i].greedy_action_dest = best_a;
                }
                if (batch[i].greedy_result_dest != nullptr) {
                    batch[i].greedy_result_dest->action = best_a;
                    batch[i].greedy_result_dest->ok = true;
                }
            }

            // Signal the atomic flag (fast spinning threads will catch this instantly,
            // parked threads will be notified via notify_one)
            SignalReadyFlag(batch[i].ready_flag);
        }
    }
    if (device_.is_cuda() && !clear_autocast_cache_per_batch_) {
        at::autocast::clear_cache();
    }
}

}  // namespace open_spiel

#endif  // OPEN_SPIEL_EXAMPLES_DUNE_NETWORK_EVAL_BATCHED_RUNNER_IMPL_H_
