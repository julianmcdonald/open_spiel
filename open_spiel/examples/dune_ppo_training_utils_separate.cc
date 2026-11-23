#include "dune_ppo_training_utils_internal.h"
#include "dune_targeted_buy.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <unordered_set>
#include <vector>

#ifdef OPEN_SPIEL_BUILD_WITH_LIBTORCH
#include <c10/cuda/CUDACachingAllocator.h>
#include "dune_seed_utils.h"
#include "open_spiel/abseil-cpp/absl/flags/flag.h"

namespace open_spiel::ppo_utils_internal {

SeparateStagingBuffers AllocateSeparateStagingBuffers(
    int64_t max_mb, int64_t obs_size, int64_t action_dim, bool is_privileged) {
  auto cpu_float = torch::TensorOptions().dtype(torch::kFloat32);
  auto cpu_bool = torch::TensorOptions().dtype(torch::kBool);
  auto cpu_long = torch::TensorOptions().dtype(torch::kInt64);
  auto cpu_fp16 = torch::TensorOptions().dtype(torch::kFloat16);

  SeparateStagingBuffers buf;
  buf.states_cpu = torch::empty({max_mb, obs_size}, cpu_float);
  if (is_privileged) {
    buf.appendix_cpu = torch::empty({max_mb, open_spiel::kPrivilegedCriticAppendixSize}, cpu_fp16);
  }
  buf.masks_cpu = torch::empty({max_mb, action_dim}, cpu_bool);
  buf.actions_cpu = torch::empty({max_mb}, cpu_long);
  buf.old_log_probs_cpu = torch::empty({max_mb}, cpu_float);
  buf.advantages_cpu = torch::empty({max_mb}, cpu_float);
  buf.returns_cpu = torch::empty({max_mb}, cpu_float);
  buf.old_values_cpu = torch::empty({max_mb}, cpu_float);
  buf.nontrivial_cpu = torch::empty({max_mb}, cpu_bool);
  buf.policy_mask_cpu = torch::empty({max_mb}, cpu_bool);
  buf.off_policy_cpu = torch::empty({max_mb}, cpu_bool);

  buf.states_ptr = buf.states_cpu.data_ptr<float>();
  buf.appendix_ptr = is_privileged ? reinterpret_cast<c10::Half*>(buf.appendix_cpu.data_ptr()) : nullptr;
  buf.masks_ptr = buf.masks_cpu.data_ptr<bool>();
  buf.actions_ptr = buf.actions_cpu.data_ptr<int64_t>();
  buf.old_log_probs_ptr = buf.old_log_probs_cpu.data_ptr<float>();
  buf.advantages_ptr = buf.advantages_cpu.data_ptr<float>();
  buf.returns_ptr = buf.returns_cpu.data_ptr<float>();
  buf.old_values_ptr = buf.old_values_cpu.data_ptr<float>();
  buf.nontrivial_ptr = buf.nontrivial_cpu.data_ptr<bool>();
  buf.policy_mask_ptr = buf.policy_mask_cpu.data_ptr<bool>();
  buf.off_policy_ptr = buf.off_policy_cpu.data_ptr<bool>();
  return buf;
}

void RecomputeSeparateValuesAndGae(
    const std::shared_ptr<SharedDunePolicyValueNetImpl>& critic_model,
    SeparateStagingBuffers& staging,
    std::vector<PpoTransition>& batch,
    int64_t obs_size,
    torch::Device device,
    double gamma,
    double gae_lambda,
    int64_t max_mb,
    bool is_privileged,
    torch::Tensor* out_recomputed_values_cpu,
    torch::Tensor* out_returns_cpu) {
  const int64_t n = static_cast<int64_t>(batch.size());
  auto cpu_float = torch::TensorOptions().dtype(torch::kFloat32);
  // Recompute every row's value with the training critic (eval mode, no grad) using CPU staging
  torch::Tensor recomputed_values_cpu = torch::empty({n}, cpu_float);
  float* recomputed_values_ptr = recomputed_values_cpu.data_ptr<float>();
  {
    torch::NoGradGuard no_grad;
    bool critic_was_training = critic_model->is_training();
    critic_model->eval();
    int64_t eval_mb = max_mb;
    for (int64_t start_idx = 0; start_idx < n; start_idx += eval_mb) {
      int64_t mb_len = std::min(eval_mb, n - start_idx);
      for (int64_t i = 0; i < mb_len; ++i) {
        int64_t idx = start_idx + i;
        std::memcpy(staging.states_ptr + i * obs_size, batch[idx].state.data(),
                    obs_size * sizeof(float));
        if (is_privileged) {
          if (!batch[idx].privileged_appendix.empty()) {
            const size_t copy_len = std::min<size_t>(open_spiel::kPrivilegedCriticAppendixSize,
                                                     batch[idx].privileged_appendix.size());
            for (size_t k = 0; k < copy_len; ++k) {
              staging.appendix_ptr[i * open_spiel::kPrivilegedCriticAppendixSize + k] =
                  c10::Half(batch[idx].privileged_appendix[k], c10::Half::from_bits());
            }
          } else {
            std::memset(staging.appendix_ptr + i * open_spiel::kPrivilegedCriticAppendixSize, 0,
                        open_spiel::kPrivilegedCriticAppendixSize * sizeof(c10::Half));
          }
        }
      }
      torch::Tensor mb_states = staging.states_cpu.narrow(0, 0, mb_len).to(device);
      torch::Tensor mb_critic_in;
      if (is_privileged) {
        torch::Tensor mb_app = staging.appendix_cpu.narrow(0, 0, mb_len).to(device).to(torch::kFloat32);
        mb_critic_in = torch::cat({mb_states, mb_app}, 1);
      } else {
        mb_critic_in = mb_states;
      }
      AutocastGuard autocast_guard(
          c10::DeviceType::CUDA,
          device.is_cuda() && ::absl::GetFlag(::FLAGS_train_amp));
      auto critic_out = critic_model->forward(mb_critic_in);
      torch::Tensor mb_vals =
          critic_out.values.squeeze(1).to(torch::kFloat32).to(torch::kCPU);
      std::memcpy(recomputed_values_ptr + start_idx, mb_vals.data_ptr<float>(), mb_len * sizeof(float));
    }
    if (critic_was_training) {
      critic_model->train();
    }
  }

  // Assign recomputed values to batch and recompute GAE backwards per (episode, player) from batch[i].reward
  for (int64_t i = 0; i < n; ++i) {
    batch[i].value = recomputed_values_ptr[i];
  }

  int64_t ep_start = 0;
  while (ep_start < n) {
    uint64_t ep_id = batch[ep_start].episode_id;
    int64_t ep_end = ep_start + 1;
    while (ep_end < n && batch[ep_end].episode_id == ep_id) {
      ++ep_end;
    }

    std::vector<float> last_value(4, 0.0f);
    std::vector<float> last_gae(4, 0.0f);

    for (int64_t i = ep_end - 1; i >= ep_start; --i) {
      int p = batch[i].player_id;
      if (p < 0 || p >= 4) continue;
      float reward = batch[i].reward;
      float delta = reward + static_cast<float>(gamma) * last_value[p] - batch[i].value;
      float advantage = delta + static_cast<float>(gamma * gae_lambda) * last_gae[p];
      batch[i].advantage = advantage;
      batch[i].return_value = advantage + batch[i].value;
      last_value[p] = batch[i].value;
      last_gae[p] = advantage;
    }

    ep_start = ep_end;
  }

  torch::Tensor returns_cpu = torch::empty({n}, cpu_float);
  float* returns_ptr = returns_cpu.data_ptr<float>();
  for (int64_t i = 0; i < n; ++i) {
    returns_ptr[i] = batch[i].return_value;
  }
  torch::Tensor old_values_cpu = recomputed_values_cpu;
  *out_recomputed_values_cpu = old_values_cpu;
  *out_returns_cpu = returns_cpu;
}

void StageAndUploadSeparateMinibatch(
    const std::vector<PpoTransition>& batch,
    SeparateStagingBuffers& staging,
    const int64_t* mb_indices,
    int64_t start_seq,
    int64_t mb_len,
    int64_t obs_size,
    int64_t action_dim,
    bool is_privileged,
    bool full_ppo_fields,
    torch::Device device,
    torch::Tensor& out_states,
    torch::Tensor& out_critic_states,
    torch::Tensor& out_masks,
    torch::Tensor& out_actions,
    torch::Tensor& out_old_log_probs,
    torch::Tensor& out_nontrivial,
    torch::Tensor& out_policy_mask,
    std::vector<const dune_semantic::CandidateActionData*>* out_cands,
    torch::Tensor* out_advantages,
    torch::Tensor* out_returns,
    torch::Tensor* out_old_values,
    torch::Tensor* out_off_policy) {
    std::memset(staging.masks_ptr, 0, mb_len * action_dim * sizeof(bool));
    if (out_cands) out_cands->resize(mb_len);

    for (int64_t i = 0; i < mb_len; ++i) {
      int64_t idx = mb_indices ? mb_indices[i] : (start_seq + i);
      const PpoTransition& transition = batch[idx];
      std::memcpy(staging.states_ptr + i * obs_size, transition.state.data(),
                  obs_size * sizeof(float));
      if (is_privileged) {
        if (!transition.privileged_appendix.empty()) {
          const size_t copy_len = std::min<size_t>(open_spiel::kPrivilegedCriticAppendixSize,
                                                   transition.privileged_appendix.size());
          for (size_t k = 0; k < copy_len; ++k) {
            staging.appendix_ptr[i * open_spiel::kPrivilegedCriticAppendixSize + k] =
                c10::Half(transition.privileged_appendix[k], c10::Half::from_bits());
          }
        } else {
          std::memset(staging.appendix_ptr + i * open_spiel::kPrivilegedCriticAppendixSize, 0,
                      open_spiel::kPrivilegedCriticAppendixSize * sizeof(c10::Half));
        }
      }
      for (Action action : transition.legal_actions) {
        if (action >= 0 && action < action_dim) {
          staging.masks_ptr[i * action_dim + action] = true;
        }
      }
      staging.actions_ptr[i] = transition.action;
      staging.old_log_probs_ptr[i] = transition.old_log_prob;
      staging.nontrivial_ptr[i] = (transition.legal_actions.size() > 1 && !transition.mask_policy_loss);
      staging.policy_mask_ptr[i] = (transition.legal_actions.size() > 1 && !transition.mask_policy_loss);
      staging.off_policy_ptr[i] = transition.off_policy_sampled;
      if (out_cands) {
        (*out_cands)[i] = &transition.candidate_data;
      }
      if (full_ppo_fields) {
        staging.advantages_ptr[i] = transition.advantage;
        staging.returns_ptr[i] = transition.return_value;
        staging.old_values_ptr[i] = transition.value;
      }
    }

    out_states = staging.states_cpu.narrow(0, 0, mb_len).to(device);
    if (is_privileged) {
      torch::Tensor mb_app = staging.appendix_cpu.narrow(0, 0, mb_len).to(device).to(torch::kFloat32);
      out_critic_states = torch::cat({out_states, mb_app}, 1);
    } else {
      out_critic_states = out_states;
    }
    out_masks = staging.masks_cpu.narrow(0, 0, mb_len).to(device);
    out_actions = staging.actions_cpu.narrow(0, 0, mb_len).to(device);
    out_old_log_probs = staging.old_log_probs_cpu.narrow(0, 0, mb_len).to(device);
    out_nontrivial = staging.nontrivial_cpu.narrow(0, 0, mb_len).to(device);
    out_policy_mask = staging.policy_mask_cpu.narrow(0, 0, mb_len).to(device);
    if (out_off_policy != nullptr) {
      *out_off_policy = staging.off_policy_cpu.narrow(0, 0, mb_len).to(device);
    }
    if (full_ppo_fields) {
      if (out_advantages) *out_advantages = staging.advantages_cpu.narrow(0, 0, mb_len).to(device);
      if (out_returns) *out_returns = staging.returns_cpu.narrow(0, 0, mb_len).to(device);
      if (out_old_values) *out_old_values = staging.old_values_cpu.narrow(0, 0, mb_len).to(device);
    }
}

void ComputeSeparatePrepassDiagnostics(
    const std::shared_ptr<SharedDunePolicyValueNetImpl>& actor_model,
    const std::shared_ptr<SharedDunePolicyValueNetImpl>& critic_model,
    const std::vector<PpoTransition>& batch,
    SeparateStagingBuffers& staging,
    const torch::Tensor& recomputed_values_cpu,
    const torch::Tensor& returns_cpu,
    int64_t obs_size,
    int64_t action_dim,
    int64_t minibatch_size,
    bool is_privileged,
    float logit_cap,
    bool compute_diagnostics,
    bool prepass_cadenced,
    bool expensive_checks_now,
    torch::Device device,
    PhaseTimer* phase_timer,
    PpoUpdateStats* stats) {
  const int64_t n = static_cast<int64_t>(batch.size());
  const float* recomputed_values_ptr = recomputed_values_cpu.data_ptr<float>();
  const float* returns_ptr = returns_cpu.data_ptr<float>();
  // 1. Episode-ID Uniqueness
  std::unordered_set<uint64_t> seen_episodes;
  bool episode_ids_unique = true;
  if (!batch.empty()) {
    uint64_t last_ep_id = batch[0].episode_id;
    seen_episodes.insert(last_ep_id);
    for (const auto& trans : batch) {
      if (trans.episode_id != last_ep_id) {
        last_ep_id = trans.episode_id;
        if (seen_episodes.find(last_ep_id) != seen_episodes.end()) {
          episode_ids_unique = false;
        }
        seen_episodes.insert(last_ep_id);
      }
    }
  }
  stats->episode_ids_unique = episode_ids_unique;

  // 2. Nontrivial Mask and Transition Counts
  int64_t nontrivial_count = 0;
  for (int64_t i = 0; i < n; ++i) {
    if (batch[i].legal_actions.size() > 1 && !batch[i].mask_policy_loss) {
      nontrivial_count++;
    }
  }
  stats->total_transitions = n;
  stats->nontrivial_transitions = nontrivial_count;
  stats->forced_transitions = n - nontrivial_count;

  if (compute_diagnostics) {
    // 3. Prepass: Policy KL Before (from actor_model) and Critic Saturation (from critic_model)
    phase_timer->Begin(kPhaseDiagPrepass);
    if (!prepass_cadenced) {
      bool was_training = actor_model->is_training();
      bool critic_was_training = critic_model->is_training();
      actor_model->eval();
      critic_model->eval();
      double kl_before_sum = 0.0;
      int64_t kl_before_nontrivial_count = 0;
      double value_near_one_count = 0.0;
      {
        torch::NoGradGuard no_grad;
        for (int64_t start = 0; start < n; start += minibatch_size) {
          int64_t end = std::min(start + minibatch_size, n);
          int64_t mb_len = end - start;
          torch::Tensor mb_states, mb_critic_states, mb_masks, mb_actions, mb_old_log_probs, mb_nontrivial, mb_policy_mask;
          std::vector<const dune_semantic::CandidateActionData*> mb_cands;
          StageAndUploadSeparateMinibatch(batch, staging, /*mb_indices=*/nullptr, start, mb_len, obs_size, action_dim, is_privileged, /*full_ppo_fields=*/false, device,
                           mb_states, mb_critic_states, mb_masks, mb_actions, mb_old_log_probs,
                           mb_nontrivial, mb_policy_mask,
                           (actor_model->with_semantic_scorer_ && actor_model->semantic_scorer_) ? &mb_cands : nullptr);
          auto outputs = actor_model->forward(mb_states);
          if (actor_model->with_semantic_scorer_ && actor_model->semantic_scorer_) {
            dune_semantic::ApplySemanticScorerBatch(
                actor_model->semantic_scorer_, outputs.trunk, mb_cands, outputs.logits, device);
          }
          torch::Tensor logits = CenterAndCapLogitsTensor(outputs.logits, mb_masks, logit_cap);
          torch::Tensor masked_logits = logits.masked_fill(mb_masks.logical_not(), -1e9f);
          torch::Tensor log_probs = torch::log_softmax(masked_logits, -1);
          torch::Tensor selected_log_probs = log_probs.gather(1, mb_actions.unsqueeze(1)).squeeze(1);

          torch::Tensor log_ratio = selected_log_probs - mb_old_log_probs;
          torch::Tensor ratio = torch::exp(log_ratio);
          torch::Tensor approx_kl = (ratio - 1.0f) - log_ratio;

          torch::Tensor masked_kl = approx_kl.masked_select(mb_policy_mask);
          if (masked_kl.numel() > 0) {
            kl_before_sum += masked_kl.sum().item<double>();
            kl_before_nontrivial_count += masked_kl.numel();
          }

          {
            AutocastGuard autocast_guard(
                c10::DeviceType::CUDA,
                device.is_cuda() && ::absl::GetFlag(::FLAGS_train_amp));
            auto critic_outputs = critic_model->forward(mb_critic_states);
            torch::Tensor mb_values =
                critic_outputs.values.squeeze(1).to(torch::kFloat32);
            torch::Tensor mb_near_one = mb_values.abs() >= 0.99f;
            value_near_one_count += mb_near_one.sum().item<double>();
          }
        }
      }
      if (was_training) {
        actor_model->train();
      }
      if (critic_was_training) {
        critic_model->train();
      }
      stats->policy_kl_before = (kl_before_nontrivial_count > 0)
                                   ? (kl_before_sum / kl_before_nontrivial_count)
                                   : 0.0;
      stats->measured_transitions = kl_before_nontrivial_count;
      stats->fraction_critic_near_1 = (n > 0) ? (value_near_one_count / n) : 0.0;
    } else {
      int64_t stored_near_one_count = 0;
      for (int64_t i = 0; i < n; ++i) {
        if (std::abs(recomputed_values_ptr[i]) >= 0.99f) ++stored_near_one_count;
      }
      stats->fraction_critic_near_1 = (n > 0) ? (static_cast<double>(stored_near_one_count) / n) : 0.0;

      double kl_before_sum = 0.0;
      int64_t kl_before_nontrivial_count = 0;
      const bool kl_measured_this_update = expensive_checks_now;
      if (kl_measured_this_update) {
        bool was_training = actor_model->is_training();
        actor_model->eval();
        {
          torch::NoGradGuard no_grad;
          AutocastGuard autocast_guard(
              device.type(), device.is_cuda() && ::absl::GetFlag(::FLAGS_rollout_amp));
          for (int64_t start = 0; start < n; start += minibatch_size) {
            int64_t end = std::min(start + minibatch_size, n);
            int64_t mb_len = end - start;
            torch::Tensor mb_states, mb_critic_states, mb_masks, mb_actions, mb_old_log_probs, mb_nontrivial, mb_policy_mask;
            std::vector<const dune_semantic::CandidateActionData*> mb_cands;
            StageAndUploadSeparateMinibatch(batch, staging, /*mb_indices=*/nullptr, start, mb_len, obs_size, action_dim, is_privileged, /*full_ppo_fields=*/false, device,
                             mb_states, mb_critic_states, mb_masks, mb_actions, mb_old_log_probs,
                             mb_nontrivial, mb_policy_mask,
                             (actor_model->with_semantic_scorer_ && actor_model->semantic_scorer_) ? &mb_cands : nullptr);
            auto outputs = actor_model->forward(mb_states);
            if (actor_model->with_semantic_scorer_ && actor_model->semantic_scorer_) {
              dune_semantic::ApplySemanticScorerBatch(
                  actor_model->semantic_scorer_, outputs.trunk, mb_cands, outputs.logits, device);
            }
            torch::Tensor logits = CenterAndCapLogitsTensor(outputs.logits, mb_masks, logit_cap);
            torch::Tensor masked_logits = logits.masked_fill(mb_masks.logical_not(), -1e9f);
            torch::Tensor log_probs = torch::log_softmax(masked_logits, -1);
            torch::Tensor selected_log_probs = log_probs.gather(1, mb_actions.unsqueeze(1)).squeeze(1);

            torch::Tensor log_ratio = selected_log_probs - mb_old_log_probs;
            torch::Tensor ratio = torch::exp(log_ratio);
            torch::Tensor approx_kl = (ratio - 1.0f) - log_ratio;

            torch::Tensor masked_kl = approx_kl.masked_select(mb_policy_mask);
            if (masked_kl.numel() > 0) {
              kl_before_sum += masked_kl.sum().item<double>();
              kl_before_nontrivial_count += masked_kl.numel();
            }
          }
        }
        if (was_training) {
          actor_model->train();
        }
      }
      stats->policy_kl_before = (kl_before_nontrivial_count > 0)
                                   ? (kl_before_sum / kl_before_nontrivial_count)
                                   : 0.0;
      stats->measured_transitions =
          kl_measured_this_update ? kl_before_nontrivial_count : int64_t{-1};
    }
    phase_timer->End(kPhaseDiagPrepass);

    // 4. Return Calibration Diagnostics
    std::vector<float> returns_vec(returns_ptr, returns_ptr + n);
    std::sort(returns_vec.begin(), returns_vec.end());
    stats->return_min = returns_vec.front();
    stats->return_max = returns_vec.back();
    stats->return_p50 = returns_vec[n * 50 / 100];
    stats->return_p95 = returns_vec[n * 95 / 100];
    stats->return_p99 = returns_vec[n * 99 / 100];

    std::vector<float> abs_returns_vec(n);
    int64_t count_outside = 0;
    for (int64_t i = 0; i < n; ++i) {
      abs_returns_vec[i] = std::abs(returns_ptr[i]);
      if (returns_ptr[i] < -1.0f || returns_ptr[i] > 1.0f) {
        count_outside++;
      }
    }
    std::sort(abs_returns_vec.begin(), abs_returns_vec.end());
    stats->abs_return_p99 = abs_returns_vec[n * 99 / 100];
    stats->fraction_targets_outside_1 = static_cast<double>(count_outside) / n;
  } else {
    stats->measured_transitions = int64_t{-1};
  }
}

}  // namespace open_spiel::ppo_utils_internal

namespace open_spiel {

PpoUpdateStats TrainPpoUpdateSeparate(
    std::shared_ptr<SharedDunePolicyValueNetImpl> actor_model,
    torch::optim::AdamW& actor_optimizer,
    std::shared_ptr<SharedDunePolicyValueNetImpl> critic_model,
    torch::optim::AdamW& critic_optimizer,
    std::vector<PpoTransition>& batch,
    int64_t obs_size, int64_t action_dim, torch::Device device,
    uint64_t master, int global_update,
    bool compute_diagnostics,
    double gamma,
    double gae_lambda) {
  if (actor_model != nullptr) {
    SPIEL_CHECK_FALSE(actor_model->bf16_inference_cache_ready());
  }
  if (critic_model != nullptr) {
    SPIEL_CHECK_FALSE(critic_model->bf16_inference_cache_ready());
  }
  PpoUpdateStats stats;
  if (batch.empty()) return stats;
  const bool r755_hybrid = ::absl::GetFlag(::FLAGS_r755_hybrid_mode);
  if (device.is_cuda() && r755_hybrid) {
    c10::cuda::CUDACachingAllocator::emptyCache();
  }

  const std::string phase_timing_mode =
      ::absl::GetFlag(::FLAGS_phase_timing_mode);
  const bool phase_timing_on = (phase_timing_mode == "phases");
  if (!phase_timing_on && phase_timing_mode != "off") {
    SpielFatalError("Unknown --phase_timing_mode: '" + phase_timing_mode +
                    "' (expected 'off' or 'phases').");
  }

  const std::string diag_prepass_mode =
      ::absl::GetFlag(::FLAGS_diag_prepass_mode);
  const bool prepass_cadenced = (diag_prepass_mode == "cadenced");
  if (!prepass_cadenced && diag_prepass_mode != "full") {
    SpielFatalError("Unknown --diag_prepass_mode: '" + diag_prepass_mode +
                    "' (expected 'full' or 'cadenced').");
  }
  const int prepass_interval = ::absl::GetFlag(::FLAGS_diag_prepass_interval);
  const bool first_update_for_model =
      prepass_cadenced && ppo_utils_internal::ConsumeFirstPrepassForModel(static_cast<const void*>(actor_model.get()));
  const bool expensive_checks_now =
      !prepass_cadenced || first_update_for_model ||
      (prepass_interval > 0 && global_update % prepass_interval == 0);

  if (compute_diagnostics) {
    if (expensive_checks_now) {
      const auto hash_start = std::chrono::steady_clock::now();
      stats.rollout_hash = ComputeRolloutHash(batch);
      if (phase_timing_on) {
        stats.phase_timings.rollout_hash_host_s =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - hash_start).count();
      }
    } else {
      stats.rollout_hash = "unmeasured";
      stats.phase_timings.rollout_hash_host_s = 0.0;
    }
  } else {
    stats.rollout_hash.clear();
    stats.phase_timings.rollout_hash_host_s = 0.0;
  }

  std::unordered_set<uint64_t> checked_episodes;
  uint64_t cur_ep = std::numeric_limits<uint64_t>::max();
  for (size_t i = 0; i < batch.size(); ++i) {
    SPIEL_CHECK_GE(batch[i].player_id, 0);
    SPIEL_CHECK_LT(batch[i].player_id, 4);
    SPIEL_CHECK_GT(batch[i].episode_id, 0);
    if (batch[i].episode_id != cur_ep) {
      SPIEL_CHECK_EQ(checked_episodes.count(batch[i].episode_id), 0);
      checked_episodes.insert(batch[i].episode_id);
      cur_ep = batch[i].episode_id;
    }
  }

  const int64_t critic_obs_size = critic_model->input_layer->weight.size(1);
  const bool is_privileged = (critic_obs_size > obs_size);

  PhaseTimer phase_timer(phase_timing_on, device);

  const int64_t n = static_cast<int64_t>(batch.size());

  int64_t minibatch_size =
      std::min<int64_t>(::absl::GetFlag(::FLAGS_ppo_minibatch_size), n);
  int64_t max_mb = minibatch_size > 0 ? minibatch_size : 2048;


  auto staging = ppo_utils_internal::AllocateSeparateStagingBuffers(
      max_mb, obs_size, action_dim, is_privileged);

  int update_epochs = std::max(1, ::absl::GetFlag(::FLAGS_ppo_update_epochs));
  float clip_epsilon = static_cast<float>(::absl::GetFlag(::FLAGS_ppo_clip_epsilon));
  bool normalize_advantages = ::absl::GetFlag(::FLAGS_normalize_advantages);
  bool clip_value_loss = ::absl::GetFlag(::FLAGS_ppo_clip_value_loss);
  float entropy_coef = static_cast<float>(::absl::GetFlag(::FLAGS_entropy_coef));
  float value_coef = static_cast<float>(::absl::GetFlag(::FLAGS_value_coef));
  float logit_cap = static_cast<float>(::absl::GetFlag(::FLAGS_logit_cap));
  double target_kl = ::absl::GetFlag(::FLAGS_target_kl);


  phase_timer.Begin(kPhaseTensorPackH2D);
  torch::Tensor recomputed_values_cpu;
  torch::Tensor returns_cpu;
  ppo_utils_internal::RecomputeSeparateValuesAndGae(
      critic_model, staging, batch, obs_size, device, gamma, gae_lambda,
      max_mb, is_privileged, &recomputed_values_cpu, &returns_cpu);
  torch::Tensor old_values_cpu = recomputed_values_cpu;
  phase_timer.End(kPhaseTensorPackH2D);

  ppo_utils_internal::ComputeSeparatePrepassDiagnostics(
      actor_model, critic_model, batch, staging, recomputed_values_cpu,
      returns_cpu, obs_size, action_dim, minibatch_size, is_privileged,
      logit_cap, compute_diagnostics, prepass_cadenced, expensive_checks_now,
      device, &phase_timer, &stats);

  if (absl::GetFlag(FLAGS_diagnostics_only)) {
    phase_timer.Finalize(&stats.phase_timings);
    return stats;
  }

  // --- PPO Loss Loop ---
  actor_model->train();
  critic_model->train();

  double weighted_policy_loss_sum = 0.0;
  double weighted_entropy_sum = 0.0;
  double weighted_kl_sum = 0.0;
  double weighted_clip_fraction_sum = 0.0;
  int64_t total_nontrivial_count = 0;
  int64_t total_onpolicy_nontrivial_count = 0;
  double value_loss_sum = 0.0;
  double value_clip_frac_sum = 0.0;

  bool actor_early_stopped = false;

  for (int epoch = 0; epoch < update_epochs; ++epoch) {
    uint64_t perm_seed = dune_seed::DeriveSeed(
        master, dune_seed::kDomainTrain, global_update, epoch,
        dune_seed::kStreamPPOPermutation);
    at::Generator gen = dune_seed::MakeTorchCPUGenerator(perm_seed);
    torch::Tensor permutation_cpu =
        torch::randperm(n, gen,
                        torch::TensorOptions()
                            .device(torch::kCPU)
                            .dtype(torch::kInt64));
    const int64_t* perm_ptr = permutation_cpu.data_ptr<int64_t>();

    double epoch_kl_sum = 0.0;
    int64_t epoch_kl_nontrivial_count = 0;

    for (int64_t start = 0; start < n; start += minibatch_size) {
      int64_t end = std::min(start + minibatch_size, n);
      const int64_t mb_len = end - start;
      const int64_t mb_index = start / minibatch_size;
      const int64_t* mb_indices = perm_ptr + start;

      phase_timer.Begin(kPhaseTensorPackH2D);
      torch::Tensor mb_states, mb_critic_states, mb_masks, mb_actions, mb_old_log_probs, mb_nontrivial, mb_policy_mask;
      torch::Tensor mb_advantages, mb_returns, mb_old_values, mb_off_policy;
      std::vector<const dune_semantic::CandidateActionData*> mb_candidate_actions;

      ppo_utils_internal::StageAndUploadSeparateMinibatch(
          batch, staging, mb_indices, /*start_seq=*/0, mb_len,
          obs_size, action_dim, is_privileged, /*full_ppo_fields=*/true, device,
          mb_states, mb_critic_states, mb_masks, mb_actions, mb_old_log_probs,
          mb_nontrivial, mb_policy_mask,
          actor_model->with_semantic_scorer_ ? &mb_candidate_actions : nullptr,
          &mb_advantages, &mb_returns, &mb_old_values, &mb_off_policy);
      phase_timer.End(kPhaseTensorPackH2D);

      int64_t mb_num_nontrivial = mb_nontrivial.sum().item<int64_t>();
      int64_t mb_num_policy = mb_policy_mask.sum().item<int64_t>();
      torch::Tensor mb_onpolicy_mask = mb_policy_mask & mb_off_policy.logical_not();
      int64_t mb_num_onpolicy = mb_onpolicy_mask.sum().item<int64_t>();

      actor_optimizer.zero_grad();
      critic_optimizer.zero_grad();

      torch::Tensor critic_loss;
      torch::Tensor value_loss;
      torch::Tensor value_clip_frac_t;

      auto compute_critic_loss = [&]() {
        auto critic_outputs = critic_model->forward(mb_critic_states);
        torch::Tensor new_values = critic_outputs.values.squeeze(1);
        if (clip_value_loss) {
          value_clip_frac_t =
              ((new_values - mb_old_values).abs() > clip_epsilon)
                  .to(torch::kFloat32)
                  .mean();
          torch::Tensor value_loss_unclipped = (new_values - mb_returns).pow(2);
          torch::Tensor value_clipped =
              mb_old_values +
              (new_values - mb_old_values).clamp(-clip_epsilon, clip_epsilon);
          torch::Tensor value_loss_clipped = (value_clipped - mb_returns).pow(2);
          value_loss = 0.5f * torch::max(value_loss_unclipped, value_loss_clipped).mean();
        } else {
          value_loss = 0.5f * (new_values - mb_returns).pow(2).mean();
        }
        critic_loss = value_coef * value_loss;
      };

      torch::Tensor actor_loss;
      torch::Tensor policy_loss, entropy, approx_kl, clip_fraction;

      auto compute_actor_loss = [&]() {
        if (actor_early_stopped) {
          policy_loss = torch::tensor(
              0.0f, torch::TensorOptions().device(device).dtype(torch::kFloat32));
          entropy = torch::tensor(
              0.0f, torch::TensorOptions().device(device).dtype(torch::kFloat32));
          approx_kl = torch::tensor(
              0.0f, torch::TensorOptions().device(device).dtype(torch::kFloat32));
          clip_fraction = torch::tensor(
              0.0f, torch::TensorOptions().device(device).dtype(torch::kFloat32));
          actor_loss = torch::tensor(
              0.0f, torch::TensorOptions().device(device).dtype(torch::kFloat32));
          return;
        }

        auto actor_outputs = actor_model->forward(mb_states);
        if (actor_model->with_semantic_scorer_ && actor_model->semantic_scorer_) {
          dune_semantic::ApplySemanticScorerBatch(
              actor_model->semantic_scorer_, actor_outputs.trunk, mb_candidate_actions, actor_outputs.logits, device);
        }
        torch::Tensor logits =
            CenterAndCapLogitsTensor(actor_outputs.logits, mb_masks, logit_cap);
        torch::Tensor masked_logits =
            logits.masked_fill(mb_masks.logical_not(), -1e9f);
        torch::Tensor log_probs = torch::log_softmax(masked_logits, -1);
        const bool use_fast_probs =
            (device.is_cuda() &&
             !::absl::GetFlag(::FLAGS_use_legacy_unoptimized_reads) &&
             !::absl::GetFlag(::FLAGS_use_wave1_ppo_update));
        torch::Tensor probs = use_fast_probs
            ? log_probs.exp()
            : torch::softmax(masked_logits, -1);
        torch::Tensor selected_log_probs =
            log_probs.gather(1, mb_actions.unsqueeze(1)).squeeze(1);

        torch::Tensor log_ratio = selected_log_probs - mb_old_log_probs;
        torch::Tensor ratio = torch::exp(log_ratio);

        torch::Tensor mb_adv = mb_advantages;
        if (normalize_advantages) {
          if (mb_num_policy > 1) {
            torch::Tensor nontrivial_adv =
                mb_advantages.masked_select(mb_policy_mask);
            torch::Tensor mean = nontrivial_adv.mean();
            torch::Tensor std = nontrivial_adv.std(/*unbiased=*/false) + 1e-8f;
            mb_adv = (mb_advantages - mean) / std;

            float mb_mean_val = mean.item<float>();
            float mb_std_val = std.item<float>();
            for (int64_t i = 0; i < mb_len; ++i) {
              auto& trans = batch[mb_indices[i]];
              if (trans.targeted_buy_forced) {
                if (epoch == 0) {
                  trans.norm_advantage = (trans.advantage - mb_mean_val) / mb_std_val;
                }
                open_spiel::dune_targeted_buy::RecordForcedNormalization(
                    global_update, epoch, mb_index,
                    trans.targeted_buy_card, trans.targeted_buy_round,
                    trans.gae_advantage, trans.advantage, trans.mc_return, trans.value,
                    mb_mean_val, mb_std_val);
              }
            }
          }
        }
        mb_adv = mb_adv.detach();

        constexpr float kMaxOffPolicyRatio = 3.0f;
        torch::Tensor eff_ratio = torch::where(
            mb_off_policy & (mb_adv < 0.0f),
            ratio.clamp_max(kMaxOffPolicyRatio), ratio);
        torch::Tensor pg_loss1 = -mb_adv * eff_ratio;
        torch::Tensor pg_loss2 =
            -mb_adv * ratio.clamp(1.0f - clip_epsilon, 1.0f + clip_epsilon);
        torch::Tensor pg_loss = torch::max(pg_loss1, pg_loss2);

        if (mb_num_policy > 0) {
          policy_loss = pg_loss.masked_select(mb_policy_mask).mean();
          entropy =
              -(probs * log_probs).sum(-1).masked_select(mb_policy_mask).mean();
          if (mb_num_onpolicy > 0) {
            approx_kl = ((ratio - 1.0f) - log_ratio)
                            .masked_select(mb_onpolicy_mask)
                            .mean();
          } else {
            approx_kl = torch::tensor(
                0.0f, torch::TensorOptions().device(device).dtype(torch::kFloat32));
          }
          clip_fraction = ((ratio - 1.0f).abs() > clip_epsilon)
                              .to(torch::kFloat32)
                              .masked_select(mb_policy_mask)
                              .mean();
          actor_loss = policy_loss - entropy_coef * entropy;
          double logit_penalty_c = ::absl::GetFlag(::FLAGS_logit_penalty_coef);
          if (logit_penalty_c > 0.0) {
            torch::Tensor masked_raw = actor_outputs.logits.masked_fill(mb_masks.logical_not(), 0.0f);
            torch::Tensor legal_counts = mb_masks.sum(-1, /*keepdim=*/true).clamp_min(1);
            torch::Tensor legal_means = masked_raw.sum(-1, /*keepdim=*/true) / legal_counts;
            torch::Tensor z = (actor_outputs.logits - legal_means).abs();
            torch::Tensor pen = torch::relu(z - 20.0f).pow(2);
            torch::Tensor masked_pen = pen.masked_select(mb_masks);
            if (masked_pen.numel() > 0) {
              actor_loss = actor_loss + static_cast<float>(logit_penalty_c) * masked_pen.mean();
            }
          }
        } else {
          policy_loss = torch::tensor(
              0.0f, torch::TensorOptions().device(device).dtype(torch::kFloat32));
          entropy = torch::tensor(
              0.0f, torch::TensorOptions().device(device).dtype(torch::kFloat32));
          approx_kl = torch::tensor(
              0.0f, torch::TensorOptions().device(device).dtype(torch::kFloat32));
          clip_fraction = torch::tensor(
              0.0f, torch::TensorOptions().device(device).dtype(torch::kFloat32));
          actor_loss = torch::tensor(
              0.0f, torch::TensorOptions().device(device).dtype(torch::kFloat32));
        }
      };

      phase_timer.Begin(kPhasePpoForwardLoss);
      if (device.is_cuda() && ::absl::GetFlag(::FLAGS_train_amp)) {
        AutocastGuard autocast_guard(c10::DeviceType::CUDA, true);
        compute_critic_loss();
        compute_actor_loss();
      } else {
        compute_critic_loss();
        compute_actor_loss();
      }
      phase_timer.End(kPhasePpoForwardLoss);

      phase_timer.Begin(kPhaseBackward);
      critic_loss.backward();
      if (!actor_early_stopped && mb_num_policy > 0) {
        actor_loss.backward();
      }
      phase_timer.End(kPhaseBackward);

      // Pre-clip gradient telemetry
      if (compute_diagnostics) {
        PhaseScope phase_scope_telemetry(&phase_timer, kPhaseGradTelemetry);
        double policy_sq = 0.0, actor_trunk_sq = 0.0;
        if (!actor_early_stopped) {
          for (const auto& named : actor_model->named_parameters()) {
            const torch::Tensor& g = named.value().grad();
            if (!g.defined()) continue;
            const double sq = g.pow(2).sum().item<double>();
            if (named.key().rfind("policy_head", 0) == 0) {
              policy_sq += sq;
            } else {
              actor_trunk_sq += sq;
            }
          }
        }

        double value_sq = 0.0, critic_trunk_sq = 0.0;
        for (const auto& named : critic_model->named_parameters()) {
          const torch::Tensor& g = named.value().grad();
          if (!g.defined()) continue;
          const double sq = g.pow(2).sum().item<double>();
          if (named.key().find("value_head") != std::string::npos) {
            value_sq += sq;
          } else {
            critic_trunk_sq += sq;
          }
        }

        double trunk_sq = actor_trunk_sq + critic_trunk_sq;
        stats.policy_head_grad_norm_sum += std::sqrt(policy_sq);
        stats.value_head_grad_norm_sum += std::sqrt(value_sq);
        stats.trunk_grad_norm_sum += std::sqrt(trunk_sq);
        stats.head_grad_norm_count += 1;
      }

      // Independent gradient clipping for separate actor and critic networks
      std::vector<torch::Tensor> active_actor_params;
      if (!actor_early_stopped) {
        for (const auto& p : actor_model->parameters()) {
          if (p.requires_grad() && p.grad().defined()) active_actor_params.push_back(p);
        }
      }
      std::vector<torch::Tensor> active_critic_params;
      for (const auto& p : critic_model->parameters()) {
        if (p.requires_grad() && p.grad().defined()) active_critic_params.push_back(p);
      }

      phase_timer.Begin(kPhaseGradClip);
      const double max_grad_norm = ::absl::GetFlag(::FLAGS_grad_clip_norm);
      double actor_grad_norm = active_actor_params.empty()
          ? 0.0
          : torch::nn::utils::clip_grad_norm_(active_actor_params, max_grad_norm);
      double critic_grad_norm = active_critic_params.empty()
          ? 0.0
          : torch::nn::utils::clip_grad_norm_(active_critic_params, max_grad_norm);
      double grad_norm = std::hypot(actor_grad_norm, critic_grad_norm);
      phase_timer.End(kPhaseGradClip);

      if (!std::isfinite(actor_grad_norm) || !std::isfinite(critic_grad_norm) ||
          std::isnan(grad_norm) || std::isinf(grad_norm)) {
        stats.nonfinite_abort = true;
        std::cerr << "Fatal PPO gradient norm: " << grad_norm << "\n";
        std::exit(EXIT_FAILURE);
      }
      stats.grad_norm_sum += grad_norm;
      stats.grad_norm_count += 1;
      if (grad_norm > stats.grad_norm_max) stats.grad_norm_max = grad_norm;

      phase_timer.Begin(kPhaseOptimizerStep);
      if (!actor_early_stopped) {
        actor_optimizer.step();
      }
      critic_optimizer.step();
      phase_timer.End(kPhaseOptimizerStep);

      phase_timer.Begin(kPhaseScalarReads);
      double kl = approx_kl.item<double>();
      value_loss_sum += value_loss.item<double>();
      if (value_clip_frac_t.defined()) {
        value_clip_frac_sum += value_clip_frac_t.item<double>();
      }
      stats.minibatches += 1;

      if (!actor_early_stopped && mb_num_policy > 0) {
        weighted_policy_loss_sum += policy_loss.item<double>() * mb_num_policy;
        weighted_entropy_sum += entropy.item<double>() * mb_num_policy;
        weighted_clip_fraction_sum +=
            clip_fraction.item<double>() * mb_num_policy;
        total_nontrivial_count += mb_num_policy;

        if (mb_num_onpolicy > 0) {
          weighted_kl_sum += kl * mb_num_onpolicy;
          total_onpolicy_nontrivial_count += mb_num_onpolicy;
          epoch_kl_sum += kl * mb_num_onpolicy;
          epoch_kl_nontrivial_count += mb_num_onpolicy;
        }
      }
      phase_timer.End(kPhaseScalarReads);

      if (!actor_early_stopped && target_kl > 0.0 && kl > target_kl) {
        stats.early_stopped = true;
        stats.kl_early_stop_epoch = epoch;
        actor_early_stopped = true;
        if (!::absl::GetFlag(::FLAGS_critic_keeps_training)) {
          break;
        }
      }
    }

    double ep_kl = (epoch_kl_nontrivial_count > 0)
                       ? (epoch_kl_sum / epoch_kl_nontrivial_count)
                       : 0.0;
    stats.epoch_kls.push_back(ep_kl);

    if (stats.early_stopped && !::absl::GetFlag(::FLAGS_critic_keeps_training)) break;
  }

  if (stats.minibatches > 0) {
    stats.value_loss = value_loss_sum / stats.minibatches;
    stats.value_clip_fraction = value_clip_frac_sum / stats.minibatches;
    if (total_nontrivial_count > 0) {
      stats.policy_loss = weighted_policy_loss_sum / total_nontrivial_count;
      stats.entropy = weighted_entropy_sum / total_nontrivial_count;
      stats.approx_kl = (total_onpolicy_nontrivial_count > 0)
                            ? (weighted_kl_sum / total_onpolicy_nontrivial_count)
                            : 0.0;
      stats.clip_fraction = weighted_clip_fraction_sum / total_nontrivial_count;
    } else {
      stats.policy_loss = 0.0;
      stats.entropy = 0.0;
      stats.approx_kl = 0.0;
      stats.clip_fraction = 0.0;
    }
  }

  torch::Tensor returns_cpu_flat = returns_cpu;
  torch::Tensor old_values_cpu_flat = old_values_cpu;
  double return_var = returns_cpu_flat.var(/*unbiased=*/false).item<double>();
  if (return_var > 1e-12) {
    double residual_var =
        (returns_cpu_flat - old_values_cpu_flat)
            .var(/*unbiased=*/false)
            .item<double>();
    stats.explained_variance = 1.0 - residual_var / return_var;
  } else {
    stats.explained_variance = 0.0;
  }

  phase_timer.Finalize(&stats.phase_timings);
  stats.phase_timings.total_host_attributed_s +=
      stats.phase_timings.rollout_hash_host_s;
  return stats;
}

}  // namespace open_spiel

#endif  // OPEN_SPIEL_BUILD_WITH_LIBTORCH
