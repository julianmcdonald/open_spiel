#include "dune_ppo_training_utils_update_internal.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

#ifdef OPEN_SPIEL_BUILD_WITH_LIBTORCH
#include <ATen/Parallel.h>
#include <ATen/ops/_fused_adamw.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include "open_spiel/abseil-cpp/absl/flags/flag.h"

namespace open_spiel::ppo_utils_internal {

void StageAndUploadMinibatch(
    TrainPpoUpdateContext& ctx,
    const int64_t* mb_indices,
    int64_t start_seq,
    int64_t mb_len,
    bool full_ppo_fields,
    torch::Tensor& out_states,
    torch::Tensor& out_masks,
    torch::Tensor& out_actions,
    torch::Tensor& out_old_log_probs,
    torch::Tensor& out_nontrivial,
    std::vector<const dune_semantic::CandidateActionData*>* out_cands,
    torch::Tensor* out_advantages,
    torch::Tensor* out_returns,
    torch::Tensor* out_old_values,
    int64_t* out_num_nontrivial,
    torch::Tensor* out_off_policy,
    int64_t* out_num_onpolicy_nontrivial) {
  const int s = ctx.staging_slot;
  ctx.staging_slot ^= 1;

  if (ctx.staging_upload_done[s] && ctx.staging_slot_recorded[s]) {
    ctx.staging_upload_done[s]->synchronize();
  }

  float* staging_states_ptr = ctx.staging_states_cpu[s].data_ptr<float>();
  bool* staging_masks_ptr = ctx.staging_masks_cpu[s].data_ptr<bool>();
  int64_t* staging_actions_ptr = ctx.staging_actions_cpu[s].data_ptr<int64_t>();
  float* staging_old_log_probs_ptr =
      ctx.staging_old_log_probs_cpu[s].data_ptr<float>();
  float* staging_advantages_ptr =
      ctx.staging_advantages_cpu[s].data_ptr<float>();
  float* staging_returns_ptr = ctx.staging_returns_cpu[s].data_ptr<float>();
  float* staging_old_values_ptr =
      ctx.staging_old_values_cpu[s].data_ptr<float>();
  bool* staging_nontrivial_ptr =
      ctx.staging_nontrivial_cpu[s].data_ptr<bool>();
  bool* staging_off_policy_ptr =
      ctx.staging_off_policy_cpu[s].data_ptr<bool>();

  if (out_cands) {
    out_cands->resize(mb_len);
  }

  const int64_t obs_size = ctx.obs_size;
  const int64_t action_dim = ctx.action_dim;
  const auto& train_batch = *ctx.train_batch_ptr;

  at::parallel_for(0, mb_len, 64, [&](int64_t begin, int64_t end) {
    for (int64_t i = begin; i < end; ++i) {
      std::memset(staging_masks_ptr + i * action_dim, 0,
                  action_dim * sizeof(bool));
      int64_t idx = mb_indices ? mb_indices[i] : (start_seq + i);
      const PpoTransition& transition = train_batch[idx];
      std::memcpy(staging_states_ptr + i * obs_size, transition.state.data(),
                  obs_size * sizeof(float));
      for (Action action : transition.legal_actions) {
        if (action >= 0 && action < action_dim) {
          staging_masks_ptr[i * action_dim + action] = true;
        }
      }
      staging_actions_ptr[i] = transition.action;
      staging_old_log_probs_ptr[i] = transition.old_log_prob;
      const bool is_nt =
          (transition.legal_actions.size() > 1 && !transition.mask_policy_loss);
      staging_nontrivial_ptr[i] = is_nt;
      staging_off_policy_ptr[i] = transition.off_policy_sampled;
      if (out_cands) {
        (*out_cands)[i] = &transition.candidate_data;
      }
      if (full_ppo_fields) {
        staging_advantages_ptr[i] = transition.advantage;
        staging_returns_ptr[i] = transition.return_value;
        staging_old_values_ptr[i] = transition.value;
      }
    }
  });

  if (out_num_nontrivial != nullptr ||
      out_num_onpolicy_nontrivial != nullptr) {
    int64_t num_nontrivial = 0;
    int64_t num_onpolicy_nontrivial = 0;
    for (int64_t i = 0; i < mb_len; ++i) {
      if (staging_nontrivial_ptr[i]) {
        ++num_nontrivial;
        if (!staging_off_policy_ptr[i]) {
          ++num_onpolicy_nontrivial;
        }
      }
    }
    if (out_num_nontrivial != nullptr) *out_num_nontrivial = num_nontrivial;
    if (out_num_onpolicy_nontrivial != nullptr) {
      *out_num_onpolicy_nontrivial = num_onpolicy_nontrivial;
    }
  }

  if (::absl::GetFlag(::FLAGS_verify_packing_alignment)) {
    std::string err;
    if (!VerifyPackedMinibatchAlignment(
            train_batch, mb_indices, start_seq, mb_len, obs_size, action_dim,
            staging_states_ptr, staging_masks_ptr, staging_actions_ptr,
            out_cands
                ? *out_cands
                : std::vector<const dune_semantic::CandidateActionData*>(),
            &err)) {
      SpielFatalError("Packed minibatch alignment verification failed: " + err);
    }
  }

  const bool use_pinned = ctx.use_pinned;
  const torch::Device device = ctx.device;
  // Narrow buffers to [0, mb_len] on CPU before upload so only active rows
  // are uploaded and consumed by PyTorch.
  out_states = ctx.staging_states_cpu[s].narrow(0, 0, mb_len).to(device, /*non_blocking=*/use_pinned);
  out_masks = ctx.staging_masks_cpu[s].narrow(0, 0, mb_len).to(device, /*non_blocking=*/use_pinned);
  out_actions = ctx.staging_actions_cpu[s].narrow(0, 0, mb_len).to(device, /*non_blocking=*/use_pinned);
  out_old_log_probs = ctx.staging_old_log_probs_cpu[s].narrow(0, 0, mb_len).to(device, /*non_blocking=*/use_pinned);
  out_nontrivial = ctx.staging_nontrivial_cpu[s].narrow(0, 0, mb_len).to(device, /*non_blocking=*/use_pinned);
  if (out_off_policy != nullptr) {
    *out_off_policy = ctx.staging_off_policy_cpu[s].narrow(0, 0, mb_len).to(device, /*non_blocking=*/use_pinned);
  }
  if (full_ppo_fields) {
    if (out_advantages) *out_advantages = ctx.staging_advantages_cpu[s].narrow(0, 0, mb_len).to(device, /*non_blocking=*/use_pinned);
    if (out_returns) *out_returns = ctx.staging_returns_cpu[s].narrow(0, 0, mb_len).to(device, /*non_blocking=*/use_pinned);
    if (out_old_values) *out_old_values = ctx.staging_old_values_cpu[s].narrow(0, 0, mb_len).to(device, /*non_blocking=*/use_pinned);
  }
  if (ctx.staging_upload_done[s]) {
    ctx.staging_upload_done[s]->record(c10::cuda::getCurrentCUDAStream());
    ctx.staging_slot_recorded[s] = true;
  }
}

void ExecuteOptimizerStep(TrainPpoUpdateContext& ctx) {
  auto& model = ctx.model;
  auto& optimizer = *ctx.optimizer;
  const torch::Device device = ctx.device;

  const double v4_lr_mult = ::absl::GetFlag(::FLAGS_v4_appendix_lr_mult);
  const double v5_lr_mult = ::absl::GetFlag(::FLAGS_v5_appendix_lr_mult);
  const int64_t in_cols =
      (model->input_layer && model->input_layer->weight.defined())
          ? model->input_layer->weight.size(1)
          : 0;
  const bool split_v4_v5 = (v5_lr_mult > 0.0 && in_cols > 9941);
  const int64_t v4_len = split_v4_v5 ? (9941 - 9182) : (in_cols - 9182);
  const bool scale_v4 =
      (v4_lr_mult >= 0.0 && std::abs(v4_lr_mult - 1.0) > 1e-12 &&
       in_cols >= 9941 && v4_len > 0);
  const bool scale_v5 =
      (split_v4_v5 && std::abs(v5_lr_mult - 1.0) > 1e-12);

  double input_group_lr = 0.0;
  double input_group_wd = 0.0;
  if (scale_v4 || scale_v5) {
    const auto* target_impl =
        model->input_layer->weight.unsafeGetTensorImpl();
    for (auto& group : optimizer.param_groups()) {
      bool found = false;
      for (const auto& p : group.params()) {
        if (p.unsafeGetTensorImpl() == target_impl) {
          found = true;
          break;
        }
      }
      if (found) {
        const auto& opts =
            static_cast<const torch::optim::AdamWOptions&>(group.options());
        input_group_lr = opts.lr();
        input_group_wd = opts.weight_decay();
        break;
      }
    }
  }

  torch::Tensor v4_w_before;
  torch::Tensor v5_w_before;
  if (scale_v4) {
    v4_w_before =
        model->input_layer->weight.narrow(1, 9182, v4_len).clone();
  }
  if (scale_v5) {
    v5_w_before =
        model->input_layer->weight.narrow(1, 9941, in_cols - 9941).clone();
  }

  ctx.phase_timer->Begin(kPhaseOptimizerStep);
  if (ctx.reuse_foreach_norms) {
    auto& param_groups = optimizer.param_groups();
    if (ctx.fused_step_caches.size() < param_groups.size()) {
      ctx.fused_step_caches.resize(param_groups.size());
    }
    for (size_t g_idx = 0; g_idx < param_groups.size(); ++g_idx) {
      auto& group = param_groups[g_idx];
      auto& options = static_cast<torch::optim::AdamWOptions&>(group.options());
      const double lr = options.lr();
      const auto [beta1, beta2] = options.betas();
      const double weight_decay = options.weight_decay();
      const double eps = options.eps();
      const bool amsgrad = options.amsgrad();

      std::vector<torch::Tensor> g_params;
      std::vector<torch::Tensor> g_grads;
      std::vector<torch::Tensor> g_exp_avgs;
      std::vector<torch::Tensor> g_exp_avg_sqs;
      std::vector<torch::Tensor> g_max_exp_avg_sqs;
      std::vector<const c10::TensorImpl*> current_keys;
      std::vector<float> steps_cpu;

      for (auto& p : group.params()) {
        if (!p.grad().defined()) continue;
        auto key = p.unsafeGetTensorImpl();
        current_keys.push_back(key);
        auto it = optimizer.state().find(key);
        if (it == optimizer.state().end()) {
          auto new_state = std::make_unique<torch::optim::AdamWParamState>();
          new_state->step(0);
          new_state->exp_avg(torch::zeros_like(p));
          new_state->exp_avg_sq(torch::zeros_like(p));
          if (amsgrad) {
            new_state->max_exp_avg_sq(torch::zeros_like(p));
          }
          it = optimizer.state().emplace(key, std::move(new_state)).first;
        }
        auto* state = dynamic_cast<torch::optim::AdamWParamState*>(it->second.get());
        state->step(state->step() + 1);
        if (!state->exp_avg().defined()) {
          state->exp_avg(torch::zeros_like(p));
        }
        if (!state->exp_avg_sq().defined()) {
          state->exp_avg_sq(torch::zeros_like(p));
        }
        if (amsgrad && !state->max_exp_avg_sq().defined()) {
          state->max_exp_avg_sq(torch::zeros_like(p));
        }

        g_params.push_back(p);
        g_grads.push_back(p.grad());
        g_exp_avgs.push_back(state->exp_avg());
        g_exp_avg_sqs.push_back(state->exp_avg_sq());
        if (amsgrad) {
          g_max_exp_avg_sqs.push_back(state->max_exp_avg_sq());
        }
        steps_cpu.push_back(static_cast<float>(state->step()));
      }

      if (!g_params.empty()) {
        auto& cache = ctx.fused_step_caches[g_idx];
        if (cache.active_keys == current_keys && cache.step_buf_dev.defined()) {
          cache.step_buf_dev.add_(1.0f);
        } else {
          cache.active_keys = current_keys;
          cache.step_buf_dev = torch::tensor(
              steps_cpu,
              torch::TensorOptions().dtype(torch::kFloat32).device(device));
          cache.state_steps_views.clear();
          cache.state_steps_views.reserve(steps_cpu.size());
          for (size_t idx = 0; idx < steps_cpu.size(); ++idx) {
            cache.state_steps_views.push_back(cache.step_buf_dev.narrow(0, idx, 1));
          }
        }

        at::_fused_adamw_(
            g_params, g_grads, g_exp_avgs, g_exp_avg_sqs, g_max_exp_avg_sqs,
            cache.state_steps_views, lr, beta1, beta2, weight_decay, eps,
            amsgrad, /*maximize=*/false);
      }
    }
  } else if (!::absl::GetFlag(::FLAGS_use_legacy_unoptimized_reads) &&
             !::absl::GetFlag(::FLAGS_use_wave1_ppo_update) &&
             device.is_cuda()) {
    for (auto& group : optimizer.param_groups()) {
      auto& options = static_cast<torch::optim::AdamWOptions&>(group.options());
      const double lr = options.lr();
      const auto [beta1, beta2] = options.betas();
      const double weight_decay = options.weight_decay();
      const double eps = options.eps();
      const bool amsgrad = options.amsgrad();

      std::vector<torch::Tensor> g_params;
      std::vector<torch::Tensor> g_grads;
      std::vector<torch::Tensor> g_exp_avgs;
      std::vector<torch::Tensor> g_exp_avg_sqs;
      std::vector<torch::Tensor> g_max_exp_avg_sqs;
      std::vector<float> steps_cpu;

      for (auto& p : group.params()) {
        if (!p.grad().defined()) continue;
        auto key = p.unsafeGetTensorImpl();
        auto it = optimizer.state().find(key);
        if (it == optimizer.state().end()) {
          auto new_state = std::make_unique<torch::optim::AdamWParamState>();
          new_state->step(0);
          new_state->exp_avg(torch::zeros_like(p));
          new_state->exp_avg_sq(torch::zeros_like(p));
          if (amsgrad) {
            new_state->max_exp_avg_sq(torch::zeros_like(p));
          }
          it = optimizer.state().emplace(key, std::move(new_state)).first;
        }
        auto* state = dynamic_cast<torch::optim::AdamWParamState*>(it->second.get());
        state->step(state->step() + 1);
        if (!state->exp_avg().defined()) {
          state->exp_avg(torch::zeros_like(p));
        }
        if (!state->exp_avg_sq().defined()) {
          state->exp_avg_sq(torch::zeros_like(p));
        }
        if (amsgrad && !state->max_exp_avg_sq().defined()) {
          state->max_exp_avg_sq(torch::zeros_like(p));
        }

        g_params.push_back(p);
        g_grads.push_back(p.grad());
        g_exp_avgs.push_back(state->exp_avg());
        g_exp_avg_sqs.push_back(state->exp_avg_sq());
        if (amsgrad) {
          g_max_exp_avg_sqs.push_back(state->max_exp_avg_sq());
        }
        steps_cpu.push_back(static_cast<float>(state->step()));
      }

      if (!g_params.empty()) {
        torch::Tensor step_buf_dev = torch::from_blob(
            steps_cpu.data(), {static_cast<int64_t>(steps_cpu.size())},
            torch::kFloat32).to(device, /*non_blocking=*/false);
        std::vector<torch::Tensor> g_state_steps;
        g_state_steps.reserve(steps_cpu.size());
        for (size_t idx = 0; idx < steps_cpu.size(); ++idx) {
          g_state_steps.push_back(step_buf_dev.narrow(0, idx, 1));
        }

        at::_fused_adamw_(
            g_params, g_grads, g_exp_avgs, g_exp_avg_sqs, g_max_exp_avg_sqs,
            g_state_steps, lr, beta1, beta2, weight_decay, eps,
            amsgrad, /*maximize=*/false);
      }
    }
  } else {
    optimizer.step();
  }
  if (scale_v4 || scale_v5) {
    torch::NoGradGuard no_grad;
    const double wd_keep = 1.0 - input_group_lr * input_group_wd;
    if (scale_v4) {
      torch::Tensor v4_slice =
          model->input_layer->weight.narrow(1, 9182, v4_len);
      torch::Tensor decayed_before =
          v4_w_before * static_cast<float>(wd_keep);
      torch::Tensor adam_delta = v4_slice - decayed_before;
      v4_slice.copy_(
          decayed_before + static_cast<float>(v4_lr_mult) * adam_delta);
    }
    if (scale_v5) {
      torch::Tensor v5_slice =
          model->input_layer->weight.narrow(1, 9941, in_cols - 9941);
      torch::Tensor decayed_before =
          v5_w_before * static_cast<float>(wd_keep);
      torch::Tensor adam_delta = v5_slice - decayed_before;
      v5_slice.copy_(
          decayed_before + static_cast<float>(v5_lr_mult) * adam_delta);
    }
  }
  ctx.phase_timer->End(kPhaseOptimizerStep);
}

void RecordT1SliceDeltaLogOdds(
    TrainPpoUpdateContext& ctx,
    const torch::Tensor& slice_states,
    const torch::Tensor& slice_masks,
    const torch::Tensor& slice_masks_cpu,
    const torch::Tensor& slice_targets_cpu,
    const torch::Tensor& slice_logp_before_cpu,
    const std::vector<const dune_semantic::CandidateActionData*>& slice_cands,
    const std::vector<std::pair<Action, Action>>& slice_pairs,
    const std::vector<int>& slice_ages) {
  if (!slice_states.defined() || !slice_logp_before_cpu.defined() || slice_states.size(0) == 0) {
    return;
  }
  auto& model = ctx.model;
  const torch::Device device = ctx.device;
  torch::Tensor alogp_after;
  {
    torch::NoGradGuard no_grad;
    auto compute_after = [&]() {
      auto ao_after = model->forward(slice_states);
      if (model->with_semantic_scorer_ && model->semantic_scorer_) {
        dune_semantic::ApplySemanticScorerBatch(
            model->semantic_scorer_, ao_after.trunk, slice_cands,
            ao_after.logits, device);
      }
      torch::Tensor alogits_after =
          CenterAndCapLogitsTensor(ao_after.logits, slice_masks, ctx.logit_cap);
      torch::Tensor amasked_after =
          alogits_after.masked_fill(slice_masks.logical_not(), -1e9f);
      alogp_after = torch::log_softmax(amasked_after, -1);
    };
    if (device.is_cuda() && ::absl::GetFlag(::FLAGS_train_amp)) {
      AutocastGuard autocast_guard(c10::DeviceType::CUDA, true);
      compute_after();
    } else {
      compute_after();
    }
  }
  torch::Tensor alogp_after_cpu = alogp_after.detach().to(torch::kFloat32).cpu();
  const int64_t n_rows = slice_states.size(0);
  const int64_t act_dim = slice_logp_before_cpu.size(1);
  for (int64_t r = 0; r < n_rows; ++r) {
    if (r < static_cast<int64_t>(slice_ages.size()) && slice_ages[r] != 0) continue;
    const float* targets_row = slice_targets_cpu[r].data_ptr<float>();
    const float* logp_before_row = slice_logp_before_cpu[r].data_ptr<float>();
    const float* logp_after_row = alogp_after_cpu[r].data_ptr<float>();
    const bool* masks_row = slice_masks_cpu[r].data_ptr<bool>();

    int64_t a_star = -1;
    int64_t a_ref = -1;

    if (r < static_cast<int64_t>(slice_pairs.size())) {
      Action p_star = slice_pairs[r].first;
      Action p_ref = slice_pairs[r].second;
      if (p_star >= 0 && p_star < act_dim &&
          p_ref >= 0 && p_ref < act_dim &&
          p_star != p_ref &&
          masks_row[p_star] && masks_row[p_ref]) {
        a_star = p_star;
        a_ref = p_ref;
      }
    }

    if (a_star < 0 || a_ref < 0) {
      float best_target = -1e9f;
      int64_t a_second_target = -1;
      float second_best_target = -1e9f;
      int64_t a_prev_top = -1;
      float best_logp_before = -1e9f;
      int64_t a_second_prev = -1;
      float second_best_logp_before = -1e9f;

      for (int64_t c = 0; c < act_dim; ++c) {
        if (!masks_row[c]) continue;
        float tgt = targets_row[c];
        if (tgt > best_target) {
          second_best_target = best_target;
          a_second_target = a_star;
          best_target = tgt;
          a_star = c;
        } else if (tgt > second_best_target) {
          second_best_target = tgt;
          a_second_target = c;
        }
        float lp_b = logp_before_row[c];
        if (lp_b > best_logp_before) {
          second_best_logp_before = best_logp_before;
          a_second_prev = a_prev_top;
          best_logp_before = lp_b;
          a_prev_top = c;
        } else if (lp_b > second_best_logp_before) {
          second_best_logp_before = lp_b;
          a_second_prev = c;
        }
      }

      if (a_star < 0 || best_target <= 0.0f) continue;
      a_ref = -1;
      if (a_star != a_prev_top && a_prev_top >= 0) {
        a_ref = a_prev_top;
      } else if (a_second_target >= 0 && second_best_target > 0.0f) {
        a_ref = a_second_target;
      } else if (a_second_prev >= 0) {
        a_ref = a_second_prev;
      }
    }

    if (a_ref >= 0 && a_star >= 0 && a_ref != a_star) {
      double delta_after = static_cast<double>(
          logp_after_row[a_star] - logp_after_row[a_ref]);
      double delta_before = static_cast<double>(
          logp_before_row[a_star] - logp_before_row[a_ref]);
      ctx.stats.aux_t1_delta_log_odds_sum += (delta_after - delta_before);
      ctx.stats.aux_t1_measured_rows++;
    }
  }
}

std::vector<float> ComputeForcedLogProbs(
    const TrainPpoUpdateContext& ctx,
    const std::vector<size_t>& indices) {
  const int64_t num_forced = static_cast<int64_t>(indices.size());
  if (num_forced == 0) return {};

  const auto& batch = *ctx.batch;
  const auto& model = ctx.model;
  const int64_t obs_size = ctx.obs_size;
  const int64_t action_dim = ctx.action_dim;
  const torch::Device device = ctx.device;

  auto cpu_float = torch::TensorOptions().dtype(torch::kFloat32);
  auto cpu_bool = torch::TensorOptions().dtype(torch::kBool);
  auto cpu_long = torch::TensorOptions().dtype(torch::kInt64);

  torch::Tensor f_states = torch::empty({num_forced, obs_size}, cpu_float);
  torch::Tensor f_masks = torch::zeros({num_forced, action_dim}, cpu_bool);
  torch::Tensor f_actions = torch::empty({num_forced}, cpu_long);

  float* f_states_ptr = f_states.data_ptr<float>();
  bool* f_masks_ptr = f_masks.data_ptr<bool>();
  int64_t* f_actions_ptr = f_actions.data_ptr<int64_t>();

  std::vector<const dune_semantic::CandidateActionData*> f_candidate_actions(num_forced, nullptr);

  for (int64_t i = 0; i < num_forced; ++i) {
    const auto& trans = batch[indices[i]];
    std::memcpy(f_states_ptr + i * obs_size, trans.state.data(), obs_size * sizeof(float));
    for (Action a : trans.legal_actions) {
      if (a >= 0 && a < action_dim) {
        f_masks_ptr[i * action_dim + a] = true;
      }
    }
    f_actions_ptr[i] = trans.action;
    f_candidate_actions[i] = &trans.candidate_data;
  }

  torch::Tensor dev_states = f_states.to(device);
  torch::Tensor dev_masks = f_masks.to(device);
  torch::Tensor dev_actions = f_actions.to(device);

  torch::NoGradGuard no_grad;
  bool was_training = model->is_training();
  model->eval();

  AutocastGuard autocast_guard(c10::DeviceType::CUDA, device.is_cuda() && ::absl::GetFlag(::FLAGS_train_amp));

  auto outputs = model->forward(dev_states);
  if (model->with_semantic_scorer_ && model->semantic_scorer_) {
    dune_semantic::ApplySemanticScorerBatch(
        model->semantic_scorer_, outputs.trunk, f_candidate_actions, outputs.logits, device);
  }

  torch::Tensor logits = CenterAndCapLogitsTensor(outputs.logits, dev_masks, ctx.logit_cap);
  torch::Tensor masked_logits = logits.masked_fill(dev_masks.logical_not(), -1e9f);
  torch::Tensor log_probs = torch::log_softmax(masked_logits, -1);
  torch::Tensor selected_log_probs =
      log_probs.gather(1, dev_actions.unsqueeze(1)).squeeze(1).to(torch::kCPU);

  if (was_training) {
    model->train();
  }

  std::vector<float> result(num_forced);
  std::memcpy(result.data(), selected_log_probs.data_ptr<float>(), num_forced * sizeof(float));
  return result;
}

double ComputeValueMse(
    const TrainPpoUpdateContext& ctx,
    const std::vector<int64_t>& indices) {
  if (indices.empty()) return 0.0;
  const auto& batch = *ctx.batch;
  const auto& model = ctx.model;
  const int64_t obs_size = ctx.obs_size;
  const int64_t minibatch_size = ctx.minibatch_size;
  const torch::Device device = ctx.device;
  auto cpu_float = torch::TensorOptions().dtype(torch::kFloat32);

  torch::NoGradGuard no_grad;
  bool was_training = model->is_training();
  model->eval();
  double total_sq_err = 0.0;
  int64_t count = 0;
  for (size_t start = 0; start < indices.size(); start += minibatch_size) {
    size_t end = std::min(start + static_cast<size_t>(minibatch_size), indices.size());
    int64_t chunk_len = end - start;
    torch::Tensor s_dev = torch::empty({chunk_len, obs_size}, cpu_float);
    torch::Tensor t_dev = torch::empty({chunk_len}, cpu_float);
    float* sp = s_dev.data_ptr<float>();
    float* tp = t_dev.data_ptr<float>();
    for (int64_t i = 0; i < chunk_len; ++i) {
      int64_t idx = indices[start + i];
      std::memcpy(sp + i * obs_size, batch[idx].state.data(), obs_size * sizeof(float));
      tp[i] = batch[idx].return_value;
    }
    torch::Tensor dev_s = s_dev.to(device);
    torch::Tensor dev_t = t_dev.to(device);
    auto out = model->forward(dev_s);
    torch::Tensor v = out.values.squeeze(1);
    torch::Tensor sq_err = (v - dev_t).pow(2);
    total_sq_err += sq_err.sum().item<double>();
    count += chunk_len;
  }
  if (was_training) model->train();
  return count > 0 ? (total_sq_err / count) : 0.0;
}

}  // namespace open_spiel::ppo_utils_internal

#endif  // OPEN_SPIEL_BUILD_WITH_LIBTORCH
