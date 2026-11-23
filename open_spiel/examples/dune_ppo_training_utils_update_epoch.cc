#include "dune_ppo_training_utils_update_internal.h"
#include "dune_targeted_buy.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <optional>
#include <string>
#include <vector>

#ifdef OPEN_SPIEL_BUILD_WITH_LIBTORCH
#include <ATen/ops/_foreach_mul.h>
#include <ATen/ops/_foreach_norm.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include "dune_seed_utils.h"
#include "open_spiel/abseil-cpp/absl/flags/flag.h"

namespace open_spiel::ppo_utils_internal {

namespace {

void RunNonMicrobatchPpoForwardBackward(
    TrainPpoUpdateContext& ctx,
    MinibatchPpoBuffers& mb) {
  auto& model = ctx.model;
  const torch::Device device = ctx.device;
  torch::Tensor total_loss;
  auto compute_loss = [&]() {
    auto outputs = model->forward(mb.mb_states);
    if (model->with_semantic_scorer_ && model->semantic_scorer_) {
      dune_semantic::ApplySemanticScorerBatch(
          model->semantic_scorer_, outputs.trunk, mb.mb_candidate_actions, outputs.logits, device);
    }

    // 3. Critic loss (Retain all samples)
    torch::Tensor new_values = outputs.values.squeeze(1);
    if (ctx.clip_value_loss) {
      mb.value_clip_frac_t =
          ((new_values - mb.mb_old_values).abs() > ctx.clip_epsilon)
              .to(torch::kFloat32)
              .mean();
      torch::Tensor value_loss_unclipped =
          (new_values - mb.mb_returns).pow(2);
      torch::Tensor value_clipped =
          mb.mb_old_values +
          (new_values - mb.mb_old_values)
              .clamp(-ctx.clip_epsilon, ctx.clip_epsilon);
      torch::Tensor value_loss_clipped =
          (value_clipped - mb.mb_returns).pow(2);
      mb.value_loss =
          0.5f * torch::max(value_loss_unclipped, value_loss_clipped).mean();
    } else {
      mb.value_loss = 0.5f * (new_values - mb.mb_returns).pow(2).mean();
    }

    if (::absl::GetFlag(::FLAGS_train_value_only)) {
      mb.policy_loss = torch::tensor(0.0f, torch::TensorOptions().device(device).dtype(torch::kFloat32));
      mb.entropy = torch::tensor(0.0f, torch::TensorOptions().device(device).dtype(torch::kFloat32));
      mb.approx_kl = torch::tensor(0.0f, torch::TensorOptions().device(device).dtype(torch::kFloat32));
      mb.clip_fraction = torch::tensor(0.0f, torch::TensorOptions().device(device).dtype(torch::kFloat32));
      total_loss = ctx.value_coef * mb.value_loss;
      if (ctx.anchor_model) {
        auto anchor_outputs = ctx.anchor_model->forward(mb.mb_states);
        if (ctx.anchor_model->with_semantic_scorer_ && ctx.anchor_model->semantic_scorer_) {
          dune_semantic::ApplySemanticScorerBatch(
              ctx.anchor_model->semantic_scorer_, anchor_outputs.trunk, mb.mb_candidate_actions, anchor_outputs.logits, device);
        }
        torch::Tensor anchor_logits = CenterAndCapLogitsTensor(anchor_outputs.logits, mb.mb_masks, ctx.logit_cap);
        torch::Tensor masked_anchor_logits = anchor_logits.masked_fill(mb.mb_masks.logical_not(), -1e9f);
        torch::Tensor anchor_probs = torch::softmax(masked_anchor_logits, -1);
        torch::Tensor log_anchor_probs = torch::log_softmax(masked_anchor_logits, -1);

        torch::Tensor logits = CenterAndCapLogitsTensor(outputs.logits, mb.mb_masks, ctx.logit_cap);
        torch::Tensor masked_logits = logits.masked_fill(mb.mb_masks.logical_not(), -1e9f);
        torch::Tensor log_probs = torch::log_softmax(masked_logits, -1);

        torch::Tensor kl_loss = anchor_probs * (log_anchor_probs - log_probs);
        torch::Tensor mean_kl;
        if (mb.mb_num_nontrivial > 0) {
          mean_kl = kl_loss.sum(-1).masked_select(mb.mb_nontrivial).mean();
        } else {
          mean_kl = torch::tensor(0.0f, torch::TensorOptions().device(device).dtype(torch::kFloat32));
        }
        double kl_anchor_coeff = ::absl::GetFlag(::FLAGS_policy_kl_anchor_coeff);
        total_loss += kl_anchor_coeff * mean_kl;
      }
      return;
    }

    torch::Tensor logits =
        CenterAndCapLogitsTensor(outputs.logits, mb.mb_masks, ctx.logit_cap);
    torch::Tensor masked_logits =
        logits.masked_fill(mb.mb_masks.logical_not(), -1e9f);
    torch::Tensor log_probs = torch::log_softmax(masked_logits, -1);
    const bool use_fast_probs =
        (device.is_cuda() &&
         !::absl::GetFlag(::FLAGS_use_legacy_unoptimized_reads) &&
         !::absl::GetFlag(::FLAGS_use_wave1_ppo_update));
    torch::Tensor probs = use_fast_probs
        ? log_probs.exp()
        : torch::softmax(masked_logits, -1);
    torch::Tensor selected_log_probs =
        log_probs.gather(1, mb.mb_actions.unsqueeze(1)).squeeze(1);

    torch::Tensor log_ratio = selected_log_probs - mb.mb_old_log_probs;
    torch::Tensor ratio = torch::exp(log_ratio);

    // 1. Advantage Normalization (per-minibatch)
    torch::Tensor mb_adv = mb.mb_advantages;
    if (ctx.normalize_advantages) {
      if (mb.mb_num_nontrivial > 1) {
        torch::Tensor nontrivial_adv = mb.mb_advantages.masked_select(mb.mb_nontrivial);
        torch::Tensor mean = nontrivial_adv.mean();
        torch::Tensor std = nontrivial_adv.std(/*unbiased=*/false) + 1e-8f;
        mb_adv = (mb.mb_advantages - mean) / std;

        if (!ctx.forced_indices.empty() || ::absl::GetFlag(::FLAGS_use_legacy_unoptimized_reads)) {
          float mb_mean_val = mean.item<float>();
          float mb_std_val = std.item<float>();
          for (int64_t i = 0; i < mb.mb_len; ++i) {
            auto& trans = (*ctx.train_batch_ptr)[mb.mb_indices[i]];
            if (trans.targeted_buy_forced) {
              if (mb.epoch == 0) {
                trans.norm_advantage = (trans.advantage - mb_mean_val) / mb_std_val;
              }
              open_spiel::dune_targeted_buy::RecordForcedNormalization(
                  ctx.global_update, mb.epoch, mb.mb_index,
                  trans.targeted_buy_card, trans.targeted_buy_round,
                  trans.gae_advantage, trans.advantage, trans.mc_return, trans.value,
                  mb_mean_val, mb_std_val);
            }
          }
        }
      }
    }
    mb_adv = mb_adv.detach();

    // 2. PPO surrogate loss
    const double acpo_clip = ::absl::GetFlag(::FLAGS_acpo_surrogate_clip);
    const double acpo_clip_neg = ::absl::GetFlag(::FLAGS_acpo_surrogate_clip_neg);
    const double acpo_clip_pos = ::absl::GetFlag(::FLAGS_acpo_surrogate_clip_pos);
    const double eff_neg = (acpo_clip_neg > 0.0) ? acpo_clip_neg : acpo_clip;
    const double eff_pos = (acpo_clip_pos > 0.0) ? acpo_clip_pos : acpo_clip;
    torch::Tensor mb_adv_surrogate = mb_adv;
    if (eff_neg > 0.0 && eff_pos > 0.0) {
      mb_adv_surrogate = mb_adv.clamp(static_cast<float>(-eff_neg), static_cast<float>(eff_pos));
    } else if (eff_neg > 0.0) {
      mb_adv_surrogate = mb_adv.clamp_min(static_cast<float>(-eff_neg));
    } else if (eff_pos > 0.0) {
      mb_adv_surrogate = mb_adv.clamp_max(static_cast<float>(eff_pos));
    }
    constexpr float kMaxOffPolicyRatio = 3.0f;
    torch::Tensor eff_ratio = torch::where(
        mb.mb_off_policy & (mb_adv_surrogate < 0.0f),
        ratio.clamp_max(kMaxOffPolicyRatio), ratio);
    torch::Tensor pg_loss1 = -mb_adv_surrogate * eff_ratio;
    torch::Tensor pg_loss2 =
        -mb_adv_surrogate * ratio.clamp(1.0f - ctx.clip_epsilon,
                                        1.0f + ctx.clip_epsilon);
    torch::Tensor pg_loss = torch::max(pg_loss1, pg_loss2);

    // 4. Actor metrics and entropy
    if (mb.mb_num_nontrivial > 0) {
      if (ctx.do_filtering && ctx.actor_only_filtering) {
        torch::Tensor actor_active_mask =
            mb.mb_nontrivial & (mb_adv.abs() >= static_cast<float>(ctx.actor_adv_threshold));
        torch::Tensor active_f = actor_active_mask.to(pg_loss.dtype());
        mb.policy_loss = (pg_loss * active_f).sum() / active_f.sum().clamp_min(1.0f);
      } else {
        mb.policy_loss = pg_loss.masked_select(mb.mb_nontrivial).mean();
      }
      mb.entropy = -(probs * log_probs).sum(-1).masked_select(mb.mb_nontrivial).mean();
      if (mb.mb_num_onpolicy_nontrivial > 0) {
        torch::Tensor mb_onpolicy_nontrivial =
            mb.mb_nontrivial & mb.mb_off_policy.logical_not();
        mb.approx_kl = ApproxKlK3Tensor(log_ratio)
                           .masked_select(mb_onpolicy_nontrivial)
                           .mean();
      } else {
        mb.approx_kl = torch::zeros(
            {}, torch::TensorOptions().device(device).dtype(torch::kFloat32));
      }
      mb.clip_fraction =
          ((ratio - 1.0f).abs() > ctx.clip_epsilon).to(torch::kFloat32).masked_select(mb.mb_nontrivial).mean();

      total_loss = mb.policy_loss + ctx.value_coef * mb.value_loss -
                   ctx.entropy_coef * mb.entropy;
      if (ctx.reverse_kl_active) {
        torch::Tensor log_diff = log_probs - mb.mb_coll_lp;
        torch::Tensor kl_elem = probs * log_diff;
        torch::Tensor legal_kl =
            ::absl::GetFlag(::FLAGS_use_wave1_ppo_update)
                ? torch::where(mb.mb_masks, kl_elem, torch::zeros_like(kl_elem))
                : torch::where(mb.mb_masks, kl_elem, 0.0f);
        torch::Tensor decision_kl = legal_kl.sum(-1);
        mb.reverse_kl_penalty_t = decision_kl.masked_select(mb.mb_nontrivial).mean();
        total_loss = total_loss + static_cast<float>(ctx.reverse_kl_coef) * mb.reverse_kl_penalty_t;
      }
      if (ctx.ema_magnet_active) {
        torch::Tensor ema_log_probs;
        torch::Tensor ema_probs;
        if (::absl::GetFlag(::FLAGS_use_wave1_ppo_update)) {
          torch::Tensor ema_logits;
          if (ctx.use_ema_cache && mb.epoch > 0) {
            torch::Tensor mb_idx_dev = mb.permutation_device.narrow(0, mb.start, mb.mb_len);
            ema_logits = ctx.ema_log_probs_cache.index_select(0, mb_idx_dev);
          } else {
            torch::NoGradGuard no_grad;
            auto ema_out = ctx.ema_model->forward(mb.mb_states);
            if (ctx.ema_model->with_semantic_scorer_ && ctx.ema_model->semantic_scorer_) {
              dune_semantic::ApplySemanticScorerBatch(
                  ctx.ema_model->semantic_scorer_, ema_out.trunk, mb.mb_candidate_actions,
                  ema_out.logits, device);
            }
            ema_logits = CenterAndCapLogitsTensor(ema_out.logits, mb.mb_masks, ctx.logit_cap);
            if (ctx.use_ema_cache && mb.epoch == 0) {
              if (!ctx.ema_log_probs_cache.defined()) {
                ctx.ema_log_probs_cache =
                    torch::zeros({ctx.n_train, ctx.action_dim}, ema_logits.options());
              }
              torch::Tensor mb_idx_dev = mb.permutation_device.narrow(0, mb.start, mb.mb_len);
              ctx.ema_log_probs_cache.index_copy_(0, mb_idx_dev, ema_logits);
            }
          }
          torch::Tensor masked_ema_logits =
              ema_logits.masked_fill(mb.mb_masks.logical_not(), -1e9f);
          ema_log_probs = torch::log_softmax(masked_ema_logits, -1);
          ema_probs = torch::softmax(masked_ema_logits, -1);
        } else {
          if (ctx.use_ema_cache && mb.epoch > 0) {
            torch::Tensor mb_idx_dev = mb.permutation_device.narrow(0, mb.start, mb.mb_len);
            ema_log_probs = ctx.ema_log_probs_cache.index_select(0, mb_idx_dev);
          } else {
            torch::NoGradGuard no_grad;
            auto ema_out = ctx.ema_model->forward(mb.mb_states);
            if (ctx.ema_model->with_semantic_scorer_ && ctx.ema_model->semantic_scorer_) {
              dune_semantic::ApplySemanticScorerBatch(
                  ctx.ema_model->semantic_scorer_, ema_out.trunk, mb.mb_candidate_actions,
                  ema_out.logits, device);
            }
            torch::Tensor ema_logits = CenterAndCapLogitsTensor(ema_out.logits, mb.mb_masks, ctx.logit_cap);
            torch::Tensor masked_ema_logits =
                ema_logits.masked_fill(mb.mb_masks.logical_not(), -1e9f);
            ema_log_probs = torch::log_softmax(masked_ema_logits, -1);
            if (ctx.use_ema_cache && mb.epoch == 0) {
              if (!ctx.ema_log_probs_cache.defined()) {
                ctx.ema_log_probs_cache =
                    torch::zeros({ctx.n_train, ctx.action_dim}, ema_log_probs.options());
              }
              torch::Tensor mb_idx_dev = mb.permutation_device.narrow(0, mb.start, mb.mb_len);
              ctx.ema_log_probs_cache.index_copy_(0, mb_idx_dev, ema_log_probs);
            }
          }
          ema_probs = ema_log_probs.exp();
        }

        torch::Tensor log_diff = ema_log_probs - log_probs;
        torch::Tensor kl_elem = ema_probs * log_diff;
        torch::Tensor legal_kl =
            ::absl::GetFlag(::FLAGS_use_wave1_ppo_update)
                ? torch::where(mb.mb_masks, kl_elem, torch::zeros_like(kl_elem))
                : torch::where(mb.mb_masks, kl_elem, 0.0f);
        torch::Tensor decision_kl = legal_kl.sum(-1);
        mb.ema_magnet_kl_t = decision_kl.masked_select(mb.mb_nontrivial).mean();
        total_loss = total_loss +
                     static_cast<float>(ctx.ema_magnet_kl_coef) * mb.ema_magnet_kl_t;
      }
    } else {
      // 0 nontrivial transitions: train critic only
      mb.policy_loss = torch::tensor(0.0f, torch::TensorOptions().device(device).dtype(torch::kFloat32));
      mb.entropy = torch::tensor(0.0f, torch::TensorOptions().device(device).dtype(torch::kFloat32));
      mb.approx_kl = torch::tensor(0.0f, torch::TensorOptions().device(device).dtype(torch::kFloat32));
      mb.clip_fraction = torch::tensor(0.0f, torch::TensorOptions().device(device).dtype(torch::kFloat32));

      total_loss = ctx.value_coef * mb.value_loss;
    }
  };

  ctx.phase_timer->Begin(kPhasePpoForwardLoss);
  if (device.is_cuda() && ::absl::GetFlag(::FLAGS_train_amp)) {
    AutocastGuard autocast_guard(c10::DeviceType::CUDA, true);
    compute_loss();
  } else {
    compute_loss();
  }
  ctx.phase_timer->End(kPhasePpoForwardLoss);

  ctx.phase_timer->Begin(kPhaseBackward);
  total_loss.backward();
  ctx.phase_timer->End(kPhaseBackward);
}

void ClipAndMeasureGradNorms(
    TrainPpoUpdateContext& ctx,
    torch::Tensor& total_norm_t) {
  auto& model = ctx.model;
  auto& stats = ctx.stats;
  const torch::Device device = ctx.device;
  const double max_norm = ::absl::GetFlag(::FLAGS_grad_clip_norm);

  if (ctx.reuse_foreach_norms) {
    std::vector<torch::Tensor> grads_to_clip;
    std::vector<const c10::TensorImpl*> current_param_keys;
    std::vector<int64_t> group_indices_cpu;
    auto named_params = model->named_parameters();
    grads_to_clip.reserve(named_params.size());
    current_param_keys.reserve(named_params.size());
    group_indices_cpu.reserve(named_params.size());

    for (const auto& named : named_params) {
      const torch::Tensor& g = named.value().grad();
      if (!g.defined()) continue;
      grads_to_clip.push_back(g);
      current_param_keys.push_back(named.value().unsafeGetTensorImpl());
      const std::string& nm = named.key();
      int64_t group_idx;
      if (nm.rfind("final_vp_head", 0) == 0) {
        group_idx = 3;
      } else if (nm.rfind("terminal_round_head", 0) == 0) {
        group_idx = 4;
      } else if (nm.rfind("next_own_action_head", 0) == 0) {
        group_idx = 5;
      } else if (nm.rfind("policy_head", 0) == 0) {
        group_idx = 0;
      } else if (nm.find("value_head") != std::string::npos) {
        group_idx = 1;
      } else {
        group_idx = 2;
      }
      group_indices_cpu.push_back(group_idx);
    }

    ctx.phase_timer->Begin(kPhaseGradClip);
    std::vector<torch::Tensor> precomputed_norms;
    torch::Tensor all_norms;
    if (!grads_to_clip.empty()) {
      precomputed_norms = at::_foreach_norm(grads_to_clip, 2.0);
      all_norms = (precomputed_norms.size() == 1)
          ? precomputed_norms[0].reshape({1})
          : torch::stack(precomputed_norms);
    }
    ctx.phase_timer->End(kPhaseGradClip);

    if (ctx.compute_diagnostics) {
      PhaseScope phase_scope_telemetry(ctx.phase_timer, kPhaseGradTelemetry);
      if (ctx.grad_telemetry_accumulated) {
        if (ctx.cached_telemetry_keys != current_param_keys ||
            !ctx.cached_group_indices_dev.defined()) {
          ctx.cached_telemetry_keys = current_param_keys;
          ctx.cached_group_indices_dev = torch::tensor(
              group_indices_cpu,
              torch::TensorOptions().dtype(torch::kInt64).device(device));
        }
        auto f64_opts =
            torch::TensorOptions().dtype(torch::kFloat64).device(device);
        torch::Tensor group_sq = torch::zeros({6}, f64_opts);
        if (all_norms.defined() && all_norms.numel() > 0) {
          torch::Tensor sq_vec = all_norms.to(torch::kFloat64).pow(2);
          group_sq.index_add_(0, ctx.cached_group_indices_dev, sq_vec);
        }
        ctx.head_norm_accum += group_sq.sqrt();
        stats.head_grad_norm_count += 1;
      } else {
        double policy_sq = 0.0, value_sq = 0.0, trunk_sq = 0.0;
        double final_vp_sq = 0.0, terminal_round_sq = 0.0, next_action_sq = 0.0;
        for (const auto& named : model->named_parameters()) {
          const torch::Tensor& g = named.value().grad();
          if (!g.defined()) continue;
          const double sq = g.pow(2).sum().item<double>();
          const std::string& nm = named.key();
          if (nm.rfind("final_vp_head", 0) == 0) {
            final_vp_sq += sq;
          } else if (nm.rfind("terminal_round_head", 0) == 0) {
            terminal_round_sq += sq;
          } else if (nm.rfind("next_own_action_head", 0) == 0) {
            next_action_sq += sq;
          } else if (nm.rfind("policy_head", 0) == 0) {
            policy_sq += sq;
          } else if (nm.find("value_head") != std::string::npos) {
            value_sq += sq;
          } else {
            trunk_sq += sq;
          }
        }
        stats.policy_head_grad_norm_sum += std::sqrt(policy_sq);
        stats.value_head_grad_norm_sum += std::sqrt(value_sq);
        stats.trunk_grad_norm_sum += std::sqrt(trunk_sq);
        stats.final_vp_head_grad_norm_sum += std::sqrt(final_vp_sq);
        stats.terminal_round_head_grad_norm_sum += std::sqrt(terminal_round_sq);
        stats.next_own_action_head_grad_norm_sum += std::sqrt(next_action_sq);
        stats.head_grad_norm_count += 1;
      }
    }

    ctx.phase_timer->Begin(kPhaseGradClip);
    if (!grads_to_clip.empty()) {
      total_norm_t = (precomputed_norms.size() == 1)
          ? precomputed_norms[0]
          : all_norms.norm(2.0);
      auto clip_coef = torch::clamp(
          static_cast<float>(max_norm) / (total_norm_t + 1e-6f),
          std::nullopt, 1.0f);
      at::_foreach_mul_(grads_to_clip, clip_coef);
    } else {
      total_norm_t = torch::zeros(
          {1}, torch::TensorOptions().dtype(torch::kFloat64).device(device));
    }
    ctx.phase_timer->End(kPhaseGradClip);
  } else {
    if (ctx.compute_diagnostics) {
      PhaseScope phase_scope_telemetry(ctx.phase_timer, kPhaseGradTelemetry);
      if (!ctx.grad_telemetry_accumulated) {
        double policy_sq = 0.0, value_sq = 0.0, trunk_sq = 0.0;
        double final_vp_sq = 0.0, terminal_round_sq = 0.0, next_action_sq = 0.0;
        for (const auto& named : model->named_parameters()) {
          const torch::Tensor& g = named.value().grad();
          if (!g.defined()) continue;
          const double sq = g.pow(2).sum().item<double>();
          const std::string& nm = named.key();
          if (nm.rfind("final_vp_head", 0) == 0) {
            final_vp_sq += sq;
          } else if (nm.rfind("terminal_round_head", 0) == 0) {
            terminal_round_sq += sq;
          } else if (nm.rfind("next_own_action_head", 0) == 0) {
            next_action_sq += sq;
          } else if (nm.rfind("policy_head", 0) == 0) {
            policy_sq += sq;
          } else if (nm.find("value_head") != std::string::npos) {
            value_sq += sq;
          } else {
            trunk_sq += sq;
          }
        }
        stats.policy_head_grad_norm_sum += std::sqrt(policy_sq);
        stats.value_head_grad_norm_sum += std::sqrt(value_sq);
        stats.trunk_grad_norm_sum += std::sqrt(trunk_sq);
        stats.final_vp_head_grad_norm_sum += std::sqrt(final_vp_sq);
        stats.terminal_round_head_grad_norm_sum += std::sqrt(terminal_round_sq);
        stats.next_own_action_head_grad_norm_sum += std::sqrt(next_action_sq);
        stats.head_grad_norm_count += 1;
      } else {
        auto f64_opts =
            torch::TensorOptions().dtype(torch::kFloat64).device(device);
        torch::Tensor group_sq = torch::zeros({6}, f64_opts);
        for (const auto& named : model->named_parameters()) {
          const torch::Tensor& g = named.value().grad();
          if (!g.defined()) continue;
          torch::Tensor sq = g.pow(2).sum().to(torch::kFloat64);
          const std::string& nm = named.key();
          int group_idx;
          if (nm.rfind("final_vp_head", 0) == 0) {
            group_idx = 3;
          } else if (nm.rfind("terminal_round_head", 0) == 0) {
            group_idx = 4;
          } else if (nm.rfind("next_own_action_head", 0) == 0) {
            group_idx = 5;
          } else if (nm.rfind("policy_head", 0) == 0) {
            group_idx = 0;
          } else if (nm.find("value_head") != std::string::npos) {
            group_idx = 1;
          } else {
            group_idx = 2;
          }
          group_sq[group_idx] += sq;
        }
        ctx.head_norm_accum += group_sq.sqrt();
        stats.head_grad_norm_count += 1;
      }
    }

    ctx.phase_timer->Begin(kPhaseGradClip);
    std::vector<torch::Tensor> grads_to_clip;
    auto model_params = model->parameters();
    grads_to_clip.reserve(model_params.size());
    for (const auto& p : model_params) {
      if (p.grad().defined()) {
        grads_to_clip.push_back(p.grad());
      }
    }
    if (!::absl::GetFlag(::FLAGS_use_legacy_unoptimized_reads) &&
        !::absl::GetFlag(::FLAGS_use_wave1_ppo_update) &&
        device.is_cuda() && !grads_to_clip.empty()) {
      auto norms = at::_foreach_norm(grads_to_clip, 2.0);
      total_norm_t = (norms.size() == 1) ? norms[0] : torch::stack(norms).norm(2.0);
      auto clip_coef = torch::clamp(static_cast<float>(max_norm) / (total_norm_t + 1e-6f), std::nullopt, 1.0f);
      at::_foreach_mul_(grads_to_clip, clip_coef);
    } else {
      double grad_norm_legacy = torch::nn::utils::clip_grad_norm_(
          model_params, max_norm);
      total_norm_t = torch::tensor(grad_norm_legacy, torch::TensorOptions().dtype(torch::kFloat64).device(device));
    }
    ctx.phase_timer->End(kPhaseGradClip);
  }
}

}  // namespace

void RunTrainPpoUpdateEpochLoop(TrainPpoUpdateContext& ctx) {
  auto& model = ctx.model;
  auto& optimizer = *ctx.optimizer;
  auto& stats = ctx.stats;
  const torch::Device device = ctx.device;
  const int64_t n_train = ctx.n_train;
  const int64_t minibatch_size = ctx.minibatch_size;

  for (int epoch = 0; epoch < ctx.update_epochs; ++epoch) {
    uint64_t perm_seed = dune_seed::DeriveSeed(
        ctx.master, dune_seed::kDomainTrain, ctx.global_update, epoch,
        dune_seed::kStreamPPOPermutation);
    at::Generator gen = dune_seed::MakeTorchCPUGenerator(perm_seed);
    torch::Tensor permutation_cpu =
        torch::randperm(n_train, gen, torch::TensorOptions().device(torch::kCPU).dtype(torch::kInt64));
    torch::Tensor permutation_device;
    if (ctx.reverse_kl_active || ctx.use_ema_cache) {
      permutation_device = permutation_cpu.to(device);
    }
    const int64_t* perm_ptr = permutation_cpu.data_ptr<int64_t>();

    double epoch_kl_sum = 0.0;
    int64_t epoch_kl_nontrivial_count = 0;

    for (int64_t start = 0; start < n_train; start += minibatch_size) {
      int64_t end = std::min(start + minibatch_size, n_train);
      const int64_t mb_len = end - start;
      const int64_t mb_index = start / minibatch_size;
      const int64_t* mb_indices = perm_ptr + start;

      MinibatchPpoBuffers mb;
      mb.epoch = epoch;
      mb.start = start;
      mb.mb_index = mb_index;
      mb.mb_len = mb_len;
      mb.mb_indices = mb_indices;
      mb.permutation_device = permutation_device;

      ctx.phase_timer->Begin(kPhaseTensorPackH2D);
      StageAndUploadMinibatch(
          ctx, mb_indices, /*start_seq=*/0, mb_len, /*full_ppo_fields=*/true,
          mb.mb_states, mb.mb_masks, mb.mb_actions, mb.mb_old_log_probs, mb.mb_nontrivial,
          model->with_semantic_scorer_ ? &mb.mb_candidate_actions : nullptr,
          &mb.mb_advantages, &mb.mb_returns, &mb.mb_old_values,
          &mb.mb_num_nontrivial, &mb.mb_off_policy,
          &mb.mb_num_onpolicy_nontrivial);

      if (ctx.reverse_kl_active) {
        torch::Tensor mb_idx_dev = permutation_device.narrow(0, start, end - start);
        mb.mb_coll_lp = ctx.collection_log_probs.index_select(0, mb_idx_dev);
      }
      ctx.phase_timer->End(kPhaseTensorPackH2D);

      if (::absl::GetFlag(::FLAGS_use_legacy_unoptimized_reads)) {
        mb.mb_num_nontrivial = mb.mb_nontrivial.sum().item<int64_t>();
        mb.mb_num_onpolicy_nontrivial =
            (mb.mb_nontrivial & mb.mb_off_policy.logical_not()).sum().item<int64_t>();
      }

      optimizer.zero_grad();

      MinibatchAuxState mb_aux;
      RunMinibatchAuxStep(ctx, mb_index, mb_len, mb_aux);

      const int64_t cfg_micro_mb = ::absl::GetFlag(::FLAGS_ppo_microbatch_size);
      const bool use_microbatch =
          (cfg_micro_mb > 0 &&
           !::absl::GetFlag(::FLAGS_train_value_only) &&
           !::absl::GetFlag(::FLAGS_use_wave1_ppo_update) &&
           !::absl::GetFlag(::FLAGS_use_legacy_unoptimized_reads));

      if (!use_microbatch) {
        RunNonMicrobatchPpoForwardBackward(ctx, mb);
      } else {
        RunMicrobatchPpoForwardBackward(ctx, cfg_micro_mb, mb);
      }

      AccumulateAuxPpoGradStats(ctx, mb_aux);
      mb_aux.aux_grad_snapshot.clear();

      torch::Tensor total_norm_t;
      ClipAndMeasureGradNorms(ctx, total_norm_t);

      double kl = 0.0;
      if (!::absl::GetFlag(::FLAGS_use_legacy_unoptimized_reads)) {
        ctx.phase_timer->Begin(kPhaseScalarReads);
        torch::Tensor zero_dev;
        auto ensure_zero = [&]() -> const torch::Tensor& {
          if (!zero_dev.defined()) {
            zero_dev = torch::zeros({1}, device);
          }
          return zero_dev;
        };

        torch::Tensor s_gn = total_norm_t.to(torch::kFloat64).reshape({1});
        torch::Tensor s_kl = mb.approx_kl.to(torch::kFloat64).reshape({1});
        torch::Tensor s_val = mb.value_loss.to(torch::kFloat64).reshape({1});
        torch::Tensor s_vc = mb.value_clip_frac_t.defined() ? mb.value_clip_frac_t.to(torch::kFloat64).reshape({1}) : ensure_zero();
        torch::Tensor s_pi = (mb.mb_num_nontrivial > 0 && mb.policy_loss.defined()) ? mb.policy_loss.to(torch::kFloat64).reshape({1}) : ensure_zero();
        torch::Tensor s_ent = (mb.mb_num_nontrivial > 0 && mb.entropy.defined()) ? mb.entropy.to(torch::kFloat64).reshape({1}) : ensure_zero();
        torch::Tensor s_cf = (mb.mb_num_nontrivial > 0 && mb.clip_fraction.defined()) ? mb.clip_fraction.to(torch::kFloat64).reshape({1}) : ensure_zero();
        torch::Tensor s_rkl = (mb.mb_num_nontrivial > 0 && ctx.reverse_kl_active && mb.reverse_kl_penalty_t.defined()) ? mb.reverse_kl_penalty_t.to(torch::kFloat64).reshape({1}) : ensure_zero();
        torch::Tensor s_mag = (mb.mb_num_nontrivial > 0 && ctx.ema_magnet_active && mb.ema_magnet_kl_t.defined()) ? mb.ema_magnet_kl_t.to(torch::kFloat64).reshape({1}) : ensure_zero();

        torch::Tensor packed_scalars = torch::cat({s_gn, s_kl, s_val, s_vc, s_pi, s_ent, s_cf, s_rkl, s_mag}).to(torch::kCPU);
        const double* sc = packed_scalars.data_ptr<double>();
        double grad_norm = sc[0];
        kl = sc[1];
        double v_loss = sc[2];
        double v_clip = sc[3];
        double p_loss = sc[4];
        double ent = sc[5];
        double c_frac = sc[6];
        double r_kl = sc[7];
        double mag_kl = sc[8];

        if (std::isnan(grad_norm) || std::isinf(grad_norm)) {
          stats.nonfinite_abort = true;
          std::cerr << "Fatal PPO gradient norm: " << grad_norm << "\n";
          std::exit(EXIT_FAILURE);
        }
        stats.grad_norm_sum += grad_norm;
        stats.grad_norm_count += 1;
        if (grad_norm > stats.grad_norm_max) stats.grad_norm_max = grad_norm;

        ctx.value_loss_sum += v_loss;
        if (mb.value_clip_frac_t.defined()) {
          ctx.value_clip_frac_sum += v_clip;
        }
        stats.minibatches += 1;

        if (mb.mb_num_nontrivial > 0) {
          ctx.weighted_policy_loss_sum += p_loss * mb.mb_num_nontrivial;
          ctx.weighted_entropy_sum += ent * mb.mb_num_nontrivial;
          ctx.weighted_clip_fraction_sum += c_frac * mb.mb_num_nontrivial;
          ctx.total_nontrivial_count += mb.mb_num_nontrivial;
          if (mb.mb_num_onpolicy_nontrivial > 0) {
            ctx.weighted_kl_sum += kl * mb.mb_num_onpolicy_nontrivial;
            ctx.total_onpolicy_nontrivial_count += mb.mb_num_onpolicy_nontrivial;
            epoch_kl_sum += kl * mb.mb_num_onpolicy_nontrivial;
            epoch_kl_nontrivial_count += mb.mb_num_onpolicy_nontrivial;
          }
          if (ctx.reverse_kl_active && mb.reverse_kl_penalty_t.defined()) {
            ctx.weighted_reverse_kl_sum += r_kl * mb.mb_num_nontrivial;
          }
          if (ctx.ema_magnet_active && mb.ema_magnet_kl_t.defined()) {
            ctx.weighted_ema_magnet_kl_sum += mag_kl * mb.mb_num_nontrivial;
            ctx.ema_magnet_nontrivial_count += mb.mb_num_nontrivial;
          }
        }
        ctx.phase_timer->End(kPhaseScalarReads);
      }

      ExecuteOptimizerStep(ctx);

      if (ctx.r755_hybrid && mb_aux.this_mb_has_aux &&
          !::absl::GetFlag(::FLAGS_r765_absorption_diag_jsonl).empty() &&
          mb_aux.t1_diag_alogp_before.defined() && mb_aux.t1_diag_a_states.defined()) {
        RecordT1SliceDeltaLogOdds(
            ctx, mb_aux.t1_diag_a_states, mb_aux.t1_diag_a_masks, mb_aux.t1_diag_a_masks_cpu,
            mb_aux.t1_diag_a_targets, mb_aux.t1_diag_alogp_before, mb_aux.t1_diag_a_cands,
            mb_aux.t1_diag_a_pairs, mb_aux.t1_diag_a_ages);
      }

      if (::absl::GetFlag(::FLAGS_use_legacy_unoptimized_reads)) {
        double grad_norm = total_norm_t.item<double>();
        if (std::isnan(grad_norm) || std::isinf(grad_norm)) {
          stats.nonfinite_abort = true;
          std::cerr << "Fatal PPO gradient norm: " << grad_norm << "\n";
          std::exit(EXIT_FAILURE);
        }
        stats.grad_norm_sum += grad_norm;
        stats.grad_norm_count += 1;
        if (grad_norm > stats.grad_norm_max) stats.grad_norm_max = grad_norm;

        ctx.phase_timer->Begin(kPhaseScalarReads);
        kl = mb.approx_kl.item<double>();
        ctx.value_loss_sum += mb.value_loss.item<double>();
        if (mb.value_clip_frac_t.defined()) {
          ctx.value_clip_frac_sum += mb.value_clip_frac_t.item<double>();
        }
        stats.minibatches += 1;

        if (mb.mb_num_nontrivial > 0) {
          ctx.weighted_policy_loss_sum += mb.policy_loss.item<double>() * mb.mb_num_nontrivial;
          ctx.weighted_entropy_sum += mb.entropy.item<double>() * mb.mb_num_nontrivial;
          ctx.weighted_clip_fraction_sum += mb.clip_fraction.item<double>() * mb.mb_num_nontrivial;
          ctx.total_nontrivial_count += mb.mb_num_nontrivial;
          if (mb.mb_num_onpolicy_nontrivial > 0) {
            ctx.weighted_kl_sum += kl * mb.mb_num_onpolicy_nontrivial;
            ctx.total_onpolicy_nontrivial_count += mb.mb_num_onpolicy_nontrivial;
            epoch_kl_sum += kl * mb.mb_num_onpolicy_nontrivial;
            epoch_kl_nontrivial_count += mb.mb_num_onpolicy_nontrivial;
          }
          if (ctx.reverse_kl_active && mb.reverse_kl_penalty_t.defined()) {
            ctx.weighted_reverse_kl_sum += mb.reverse_kl_penalty_t.item<double>() * mb.mb_num_nontrivial;
          }
          if (ctx.ema_magnet_active && mb.ema_magnet_kl_t.defined()) {
            ctx.weighted_ema_magnet_kl_sum += mb.ema_magnet_kl_t.item<double>() * mb.mb_num_nontrivial;
            ctx.ema_magnet_nontrivial_count += mb.mb_num_nontrivial;
          }
        }
        ctx.phase_timer->End(kPhaseScalarReads);
      }

      if (ctx.target_kl > 0.0 && kl > ctx.target_kl) {
        stats.early_stopped = true;
        stats.kl_early_stop_epoch = epoch;
        break;
      }
    }

    double ep_kl = (epoch_kl_nontrivial_count > 0) ? (epoch_kl_sum / epoch_kl_nontrivial_count) : 0.0;
    stats.epoch_kls.push_back(ep_kl);

    if (stats.early_stopped) break;
  }
  ctx.ema_log_probs_cache = torch::Tensor();
  if (device.is_cuda() && ctx.r755_hybrid) {
    c10::cuda::CUDACachingAllocator::emptyCache();
  }
}

}  // namespace open_spiel::ppo_utils_internal

#endif  // OPEN_SPIEL_BUILD_WITH_LIBTORCH
