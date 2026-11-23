#include "dune_ppo_training_utils_update_internal.h"
#include "dune_targeted_buy.h"

#include <algorithm>
#include <cmath>
#include <vector>

#ifdef OPEN_SPIEL_BUILD_WITH_LIBTORCH
#include <ATen/ops/_foreach_norm.h>
#include "open_spiel/abseil-cpp/absl/flags/flag.h"

namespace open_spiel::ppo_utils_internal {

void RunMinibatchAuxStep(
    TrainPpoUpdateContext& ctx,
    int64_t mb_index,
    int64_t mb_len,
    MinibatchAuxState& mb_aux) {
  if (!ctx.aux_active) {
    return;
  }
  auto& model = ctx.model;
  auto& stats = ctx.stats;
  const torch::Device device = ctx.device;
  const bool r755_hybrid = ctx.r755_hybrid;

  torch::Tensor a_states, a_masks, a_targets, a_values;
  int64_t as = 0;
  int64_t al = 0;
  if (::absl::GetFlag(::FLAGS_aux_full_batch_presentation)) {
    mb_aux.this_mb_has_aux = true;
    a_states = ctx.aux_states;
    a_masks = ctx.aux_masks;
    a_targets = ctx.aux_targets;
    a_values = ctx.aux_values;
    al = ctx.aux_n;
  } else {
    as = ctx.aux_slice_start[mb_index];
    al = ctx.aux_slice_len[mb_index];
    if (al > 0) {
      mb_aux.this_mb_has_aux = true;
      a_states = ctx.aux_states.narrow(0, as, al);
      a_masks = ctx.aux_masks.narrow(0, as, al);
      a_targets = ctx.aux_targets.narrow(0, as, al);
      a_values = ctx.aux_values.narrow(0, as, al);
    }
  }
  if (!mb_aux.this_mb_has_aux) {
    return;
  }

  std::vector<const dune_semantic::CandidateActionData*> a_cands;
  if (model->with_semantic_scorer_) {
    if (::absl::GetFlag(::FLAGS_aux_full_batch_presentation)) {
      a_cands = ctx.all_aux_cands;
    } else {
      a_cands.reserve(al);
      for (int64_t k = 0; k < al; ++k) {
        a_cands.push_back(ctx.all_aux_cands[as + k]);
      }
    }
  }
  std::vector<double> slice_raw_w;
  std::vector<int> slice_ages;
  if (::absl::GetFlag(::FLAGS_aux_full_batch_presentation)) {
    slice_raw_w = ctx.all_aux_row_weights;
    slice_ages = ctx.all_aux_ages;
  } else {
    slice_raw_w.reserve(al);
    slice_ages.reserve(al);
    for (int64_t k = 0; k < al; ++k) {
      slice_raw_w.push_back(ctx.all_aux_row_weights[as + k]);
      slice_ages.push_back(ctx.all_aux_ages[as + k]);
    }
  }
  std::vector<float> eff_w;
  bool slice_cap_active = false;
  torch::Tensor scaled_aux;
  double ce_val = 0.0, vmse_val = 0.0;
  auto compute_aux = [&]() {
    auto ao = model->forward(a_states);
    if (model->with_semantic_scorer_ && model->semantic_scorer_) {
      dune_semantic::ApplySemanticScorerBatch(
          model->semantic_scorer_, ao.trunk, a_cands, ao.logits, device);
    }
    torch::Tensor alogits =
        CenterAndCapLogitsTensor(ao.logits, a_masks, ctx.logit_cap);
    torch::Tensor amasked =
        alogits.masked_fill(a_masks.logical_not(), -1e9f);
    torch::Tensor alogp = torch::log_softmax(amasked, -1);
    if (r755_hybrid && !::absl::GetFlag(::FLAGS_r765_absorption_diag_jsonl).empty()) {
      mb_aux.t1_diag_alogp_before = alogp.detach().to(torch::kFloat32).cpu();
    }
    if (r755_hybrid) {
      torch::Tensor ce_per_row = -(a_targets * alogp).sum(-1);
      ce_val = ce_per_row.mean().item<double>();
      vmse_val = 0.0;
      bool cap_active = false;
      eff_w = ComputeWeightedHybridAuxRowWeights(
          slice_raw_w, mb_len,
          ::absl::GetFlag(::FLAGS_r765_aux_norm_mode),
          ::absl::GetFlag(::FLAGS_r765_aux_max_slice_coef),
          &cap_active);
      slice_cap_active = cap_active;
      torch::Tensor eff_w_t = torch::from_blob(
          eff_w.data(), {static_cast<int64_t>(eff_w.size())},
          torch::TensorOptions().dtype(torch::kFloat32))
          .to(ce_per_row.device(), ce_per_row.dtype());
      scaled_aux = (ce_per_row * eff_w_t).sum();
    } else {
      torch::Tensor ce = -(a_targets * alogp).sum(-1).mean();
      torch::Tensor av = ao.values.squeeze(1);
      torch::Tensor vmse = (av - a_values).pow(2).mean();
      ce_val = ce.item<double>();
      vmse_val = vmse.item<double>();
      double aux_v_coef = ::absl::GetFlag(::FLAGS_aux_value_coef);
      if (aux_v_coef < 0.0) aux_v_coef = ctx.value_coef;
      scaled_aux =
          static_cast<float>(ctx.search_loss_coef) * (ce + static_cast<float>(aux_v_coef) * vmse);
    }
  };
  if (device.is_cuda() && ::absl::GetFlag(::FLAGS_train_amp)) {
    AutocastGuard autocast_guard(c10::DeviceType::CUDA, true);
    compute_aux();
  } else {
    compute_aux();
  }
  scaled_aux.backward();
  if (r755_hybrid) {
    stats.aux_total_slices++;
    if (slice_cap_active) stats.aux_cap_active_slices++;
    for (size_t k = 0; k < slice_ages.size(); ++k) {
      int age = slice_ages[k];
      if (age >= 0 && age < PpoUpdateStats::kMaxR765ReplayAges) {
        stats.aux_presentations_by_age[age]++;
        if (k < eff_w.size()) {
          stats.aux_eff_weight_sum_by_age[age] += eff_w[k];
        }
      }
    }
  }
  if (r755_hybrid && !::absl::GetFlag(::FLAGS_r765_absorption_diag_jsonl).empty()) {
    mb_aux.t1_diag_a_states = a_states;
    mb_aux.t1_diag_a_masks = a_masks;
    mb_aux.t1_diag_a_masks_cpu = a_masks.detach().to(torch::kCPU);
    mb_aux.t1_diag_a_targets = a_targets.detach().to(torch::kFloat32).cpu();
    mb_aux.t1_diag_a_cands = a_cands;
    mb_aux.t1_diag_a_ages = slice_ages;
    if (::absl::GetFlag(::FLAGS_aux_full_batch_presentation)) {
      mb_aux.t1_diag_a_pairs = ctx.all_aux_mpo_pairs;
    } else {
      mb_aux.t1_diag_a_pairs.clear();
      mb_aux.t1_diag_a_pairs.reserve(al);
      for (int64_t k = 0; k < al; ++k) {
        mb_aux.t1_diag_a_pairs.push_back(ctx.all_aux_mpo_pairs[as + k]);
      }
    }
  }
  // Snapshot aux-only grads (clone) and their flattened 2-norm.
  const bool need_full_aux_grad = (!r755_hybrid || ctx.abort_grad_norm_ratio > 0.0);
  if (need_full_aux_grad || !ctx.purchase_param_indices.empty() || !ctx.action_head_param_indices.empty()) {
    auto params = model->parameters();
    mb_aux.aux_grad_snapshot.resize(params.size());
    std::vector<torch::Tensor> defined_aux_grads;
    if (need_full_aux_grad) {
      defined_aux_grads.reserve(params.size());
    }
    for (size_t pi = 0; pi < params.size(); ++pi) {
      if (!need_full_aux_grad &&
          (pi >= ctx.need_snapshot_param.size() || !ctx.need_snapshot_param[pi])) {
        continue;
      }
      auto g = params[pi].grad();
      if (g.defined()) {
        mb_aux.aux_grad_snapshot[pi] = g.detach().clone();
        if (need_full_aux_grad) {
          defined_aux_grads.push_back(mb_aux.aux_grad_snapshot[pi]);
        }
      }
    }
    if (need_full_aux_grad) {
      double aux_sq = 0.0;
      if (!defined_aux_grads.empty()) {
        auto norms = at::_foreach_norm(defined_aux_grads, 2.0);
        aux_sq = torch::stack(norms).to(torch::kFloat64).pow(2).sum().item<double>();
      }
      mb_aux.this_aux_norm = std::sqrt(aux_sq);
    }
  }
  ctx.aux_ce_sum += ce_val;
  ctx.aux_vmse_sum += vmse_val;
  stats.aux_total_presentations += a_states.size(0);
  if (::absl::GetFlag(::FLAGS_aux_full_batch_presentation)) {
    std::fill(ctx.aux_visited_mask.begin(), ctx.aux_visited_mask.end(), static_cast<uint8_t>(1));
  } else {
    int64_t mask_sz = static_cast<int64_t>(ctx.aux_visited_mask.size());
    int64_t start_idx = std::max<int64_t>(0, std::min(as, mask_sz));
    int64_t end_idx = std::max<int64_t>(start_idx, std::min(as + al, mask_sz));
    std::fill(ctx.aux_visited_mask.begin() + start_idx, ctx.aux_visited_mask.begin() + end_idx, static_cast<uint8_t>(1));
  }
  ++ctx.aux_slice_uses;
}

void RunMicrobatchPpoForwardBackward(
    TrainPpoUpdateContext& ctx,
    int64_t cfg_micro_mb,
    MinibatchPpoBuffers& mb) {
  auto& model = ctx.model;
  const torch::Device device = ctx.device;
  const bool train_amp =
      (device.is_cuda() && ::absl::GetFlag(::FLAGS_train_amp));

  // 1. Advantage Normalization (per-minibatch, computed once over [0, mb_len))
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

  // 2. PPO surrogate advantage clipping (computed once over [0, mb_len))
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

  torch::Tensor full_actor_active_mask;
  torch::Tensor actor_denom;
  if (mb.mb_num_nontrivial > 0 && ctx.do_filtering && ctx.actor_only_filtering) {
    full_actor_active_mask =
        mb.mb_nontrivial & (mb_adv.abs() >= static_cast<float>(ctx.actor_adv_threshold));
    actor_denom =
        full_actor_active_mask.to(torch::kFloat32).sum().clamp_min(1.0f);
  }

  auto fp32_dev_opts = torch::TensorOptions().device(device).dtype(torch::kFloat32);
  torch::Tensor acc_val_loss = torch::zeros({}, fp32_dev_opts);
  torch::Tensor acc_pol_loss = torch::zeros({}, fp32_dev_opts);
  torch::Tensor acc_entropy = torch::zeros({}, fp32_dev_opts);
  torch::Tensor acc_approx_kl = torch::zeros({}, fp32_dev_opts);
  torch::Tensor acc_clip_frac = torch::zeros({}, fp32_dev_opts);
  torch::Tensor acc_val_clip_frac;
  if (ctx.clip_value_loss) {
    acc_val_clip_frac = torch::zeros({}, fp32_dev_opts);
  }
  torch::Tensor acc_rkl;
  if (mb.mb_num_nontrivial > 0 && ctx.reverse_kl_active) {
    acc_rkl = torch::zeros({}, fp32_dev_opts);
  }
  torch::Tensor acc_ema_kl;
  if (mb.mb_num_nontrivial > 0 && ctx.ema_magnet_active) {
    acc_ema_kl = torch::zeros({}, fp32_dev_opts);
  }

  for (int64_t sub_s = 0; sub_s < mb.mb_len; sub_s += cfg_micro_mb) {
    const int64_t sub_len = std::min(cfg_micro_mb, mb.mb_len - sub_s);
    torch::Tensor sub_states = mb.mb_states.narrow(0, sub_s, sub_len);
    torch::Tensor sub_masks = mb.mb_masks.narrow(0, sub_s, sub_len);
    torch::Tensor sub_actions = mb.mb_actions.narrow(0, sub_s, sub_len);
    torch::Tensor sub_old_lp = mb.mb_old_log_probs.narrow(0, sub_s, sub_len);
    torch::Tensor sub_nt = mb.mb_nontrivial.narrow(0, sub_s, sub_len);
    torch::Tensor sub_off_policy = mb.mb_off_policy.narrow(0, sub_s, sub_len);
    torch::Tensor sub_returns = mb.mb_returns.narrow(0, sub_s, sub_len);
    torch::Tensor sub_old_vals = mb.mb_old_values.narrow(0, sub_s, sub_len);
    torch::Tensor sub_adv_surr = mb_adv_surrogate.narrow(0, sub_s, sub_len);
    std::vector<const dune_semantic::CandidateActionData*> sub_cands;
    if (!mb.mb_candidate_actions.empty()) {
      sub_cands.assign(mb.mb_candidate_actions.begin() + sub_s,
                       mb.mb_candidate_actions.begin() + sub_s + sub_len);
    }

    ctx.phase_timer->Begin(kPhasePpoForwardLoss);

    // Step A (BEFORE learner forward): scoped EMA forward / cache read.
    torch::Tensor sub_ema_log_probs;
    torch::Tensor sub_ema_probs;
    if (mb.mb_num_nontrivial > 0 && ctx.ema_magnet_active) {
      if (ctx.use_ema_cache && mb.epoch > 0) {
        torch::Tensor sub_idx_dev =
            mb.permutation_device.narrow(0, mb.start + sub_s, sub_len);
        sub_ema_log_probs = ctx.ema_log_probs_cache.index_select(0, sub_idx_dev);
      } else {
        torch::NoGradGuard no_grad;
        auto run_ema_sub = [&]() {
          auto ema_out = ctx.ema_model->forward(sub_states);
          if (ctx.ema_model->with_semantic_scorer_ && ctx.ema_model->semantic_scorer_) {
            dune_semantic::ApplySemanticScorerBatch(
                ctx.ema_model->semantic_scorer_, ema_out.trunk, sub_cands,
                ema_out.logits, device);
          }
          torch::Tensor ema_logits =
              CenterAndCapLogitsTensor(ema_out.logits, sub_masks, ctx.logit_cap);
          torch::Tensor masked_ema_logits =
              ema_logits.masked_fill(sub_masks.logical_not(), -1e9f);
          sub_ema_log_probs = torch::log_softmax(masked_ema_logits, -1);
          if (ctx.use_ema_cache && mb.epoch == 0) {
            if (!ctx.ema_log_probs_cache.defined()) {
              ctx.ema_log_probs_cache =
                  torch::zeros({ctx.n_train, ctx.action_dim}, sub_ema_log_probs.options());
            }
            torch::Tensor sub_idx_dev =
                mb.permutation_device.narrow(0, mb.start + sub_s, sub_len);
            ctx.ema_log_probs_cache.index_copy_(0, sub_idx_dev, sub_ema_log_probs);
          }
        };
        if (train_amp) {
          AutocastGuard ema_autocast(c10::DeviceType::CUDA, true);
          run_ema_sub();
        } else {
          run_ema_sub();
        }
      }
      sub_ema_probs = sub_ema_log_probs.exp();
    }

    // Step B: learner forward and loss inside its own autocast scope.
    torch::Tensor sub_total_loss;
    auto compute_sub_loss = [&]() {
      auto outputs = model->forward(sub_states);
      if (model->with_semantic_scorer_ && model->semantic_scorer_) {
        dune_semantic::ApplySemanticScorerBatch(
            model->semantic_scorer_, outputs.trunk, sub_cands,
            outputs.logits, device);
      }

      torch::Tensor new_values = outputs.values.squeeze(1);
      torch::Tensor sub_val_loss;
      if (ctx.clip_value_loss) {
        acc_val_clip_frac +=
            ((new_values - sub_old_vals).abs() > ctx.clip_epsilon)
                .to(torch::kFloat32)
                .sum()
                .detach() /
            static_cast<float>(mb.mb_len);
        torch::Tensor value_loss_unclipped =
            (new_values - sub_returns).pow(2);
        torch::Tensor value_clipped =
            sub_old_vals +
            (new_values - sub_old_vals).clamp(-ctx.clip_epsilon, ctx.clip_epsilon);
        torch::Tensor value_loss_clipped =
            (value_clipped - sub_returns).pow(2);
        sub_val_loss =
            0.5f * torch::max(value_loss_unclipped, value_loss_clipped).sum() /
            static_cast<float>(mb.mb_len);
      } else {
        sub_val_loss =
            0.5f * (new_values - sub_returns).pow(2).sum() /
            static_cast<float>(mb.mb_len);
      }
      acc_val_loss += sub_val_loss.detach().to(torch::kFloat32);

      if (mb.mb_num_nontrivial > 0) {
        const float nt_denom = static_cast<float>(mb.mb_num_nontrivial);
        torch::Tensor logits =
            CenterAndCapLogitsTensor(outputs.logits, sub_masks, ctx.logit_cap);
        torch::Tensor masked_logits =
            logits.masked_fill(sub_masks.logical_not(), -1e9f);
        torch::Tensor log_probs = torch::log_softmax(masked_logits, -1);
        const bool use_fast_probs = device.is_cuda();
        torch::Tensor probs = use_fast_probs
            ? log_probs.exp()
            : torch::softmax(masked_logits, -1);
        torch::Tensor selected_lp =
            log_probs.gather(1, sub_actions.unsqueeze(1)).squeeze(1);

        torch::Tensor log_ratio = selected_lp - sub_old_lp;
        torch::Tensor ratio = torch::exp(log_ratio);

        constexpr float kMaxOffPolicyRatio = 3.0f;
        torch::Tensor eff_ratio = torch::where(
            sub_off_policy & (sub_adv_surr < 0.0f),
            ratio.clamp_max(kMaxOffPolicyRatio), ratio);
        torch::Tensor pg_loss1 = -sub_adv_surr * eff_ratio;
        torch::Tensor pg_loss2 =
            -sub_adv_surr * ratio.clamp(1.0f - ctx.clip_epsilon,
                                        1.0f + ctx.clip_epsilon);
        torch::Tensor pg_loss = torch::max(pg_loss1, pg_loss2);

        torch::Tensor sub_pol_loss;
        if (ctx.do_filtering && ctx.actor_only_filtering) {
          torch::Tensor sub_active_mask =
              full_actor_active_mask.narrow(0, sub_s, sub_len);
          torch::Tensor sub_active_f = sub_active_mask.to(pg_loss.dtype());
          sub_pol_loss =
              (pg_loss * sub_active_f).sum() /
              actor_denom.to(pg_loss.dtype());
        } else {
          sub_pol_loss =
              pg_loss.masked_select(sub_nt).sum() / nt_denom;
        }
        torch::Tensor sub_ent =
            -(probs * log_probs).sum(-1).masked_select(sub_nt).sum() /
            nt_denom;

        acc_pol_loss += sub_pol_loss.detach().to(torch::kFloat32);
        acc_entropy += sub_ent.detach().to(torch::kFloat32);
        if (mb.mb_num_onpolicy_nontrivial > 0) {
          torch::Tensor sub_onpolicy_nt =
              sub_nt & sub_off_policy.logical_not();
          acc_approx_kl +=
              (ApproxKlK3Tensor(log_ratio)
                   .masked_select(sub_onpolicy_nt)
                   .sum() /
               static_cast<float>(mb.mb_num_onpolicy_nontrivial))
                  .detach()
                  .to(torch::kFloat32);
        }
        acc_clip_frac +=
            (((ratio - 1.0f).abs() > ctx.clip_epsilon)
                 .to(torch::kFloat32)
                 .masked_select(sub_nt)
                 .sum() /
             nt_denom)
                .detach();

        sub_total_loss =
            sub_pol_loss + ctx.value_coef * sub_val_loss - ctx.entropy_coef * sub_ent;
        if (ctx.reverse_kl_active) {
          torch::Tensor sub_coll_lp = mb.mb_coll_lp.narrow(0, sub_s, sub_len);
          torch::Tensor log_diff = log_probs - sub_coll_lp;
          torch::Tensor kl_elem = probs * log_diff;
          torch::Tensor legal_kl = torch::where(sub_masks, kl_elem, 0.0f);
          torch::Tensor decision_kl = legal_kl.sum(-1);
          torch::Tensor sub_rkl =
              decision_kl.masked_select(sub_nt).sum() / nt_denom;
          acc_rkl += sub_rkl.detach().to(torch::kFloat32);
          sub_total_loss =
              sub_total_loss + static_cast<float>(ctx.reverse_kl_coef) * sub_rkl;
        }
        if (ctx.ema_magnet_active) {
          torch::Tensor log_diff = sub_ema_log_probs - log_probs;
          torch::Tensor kl_elem = sub_ema_probs * log_diff;
          torch::Tensor legal_kl = torch::where(sub_masks, kl_elem, 0.0f);
          torch::Tensor decision_kl = legal_kl.sum(-1);
          torch::Tensor sub_ema_kl =
              decision_kl.masked_select(sub_nt).sum() / nt_denom;
          acc_ema_kl += sub_ema_kl.detach().to(torch::kFloat32);
          sub_total_loss =
              sub_total_loss +
              static_cast<float>(ctx.ema_magnet_kl_coef) * sub_ema_kl;
        }
      } else {
        sub_total_loss = ctx.value_coef * sub_val_loss;
      }
    };

    if (train_amp) {
      AutocastGuard autocast_guard(c10::DeviceType::CUDA, true);
      compute_sub_loss();
    } else {
      compute_sub_loss();
    }
    ctx.phase_timer->End(kPhasePpoForwardLoss);

    ctx.phase_timer->Begin(kPhaseBackward);
    sub_total_loss.backward();
    ctx.phase_timer->End(kPhaseBackward);
  }

  mb.value_loss = acc_val_loss;
  if (ctx.clip_value_loss) {
    mb.value_clip_frac_t = acc_val_clip_frac;
  }
  mb.policy_loss = acc_pol_loss;
  mb.entropy = acc_entropy;
  mb.approx_kl = acc_approx_kl;
  mb.clip_fraction = acc_clip_frac;
  if (mb.mb_num_nontrivial > 0) {
    if (ctx.reverse_kl_active) {
      mb.reverse_kl_penalty_t = acc_rkl;
    }
    if (ctx.ema_magnet_active) {
      mb.ema_magnet_kl_t = acc_ema_kl;
    }
  }
}

void AccumulateAuxPpoGradStats(
    TrainPpoUpdateContext& ctx,
    const MinibatchAuxState& mb_aux) {
  auto& model = ctx.model;
  auto& stats = ctx.stats;
  const bool r755_hybrid = ctx.r755_hybrid;

  if (mb_aux.this_mb_has_aux && (!r755_hybrid || ctx.abort_grad_norm_ratio > 0.0)) {
    auto params = model->parameters();
    std::vector<torch::Tensor> defined_ppo_grads;
    defined_ppo_grads.reserve(params.size());
    for (size_t pi = 0; pi < params.size(); ++pi) {
      auto tot = params[pi].grad();
      if (!tot.defined()) continue;
      torch::Tensor ppo_g =
          (pi < mb_aux.aux_grad_snapshot.size() && mb_aux.aux_grad_snapshot[pi].defined())
              ? (tot - mb_aux.aux_grad_snapshot[pi])
              : tot;
      defined_ppo_grads.push_back(std::move(ppo_g));
    }
    double ppo_sq = 0.0;
    if (!defined_ppo_grads.empty()) {
      auto norms = at::_foreach_norm(defined_ppo_grads, 2.0);
      ppo_sq = torch::stack(norms).to(torch::kFloat64).pow(2).sum().item<double>();
    }
    ctx.aux_norm_sum += mb_aux.this_aux_norm;
    ctx.ppo_only_norm_sum += std::sqrt(ppo_sq);
    ++ctx.aux_norm_count;
  }

  // Compute purchase logit gradient cosine between PPO policy loss grads and MPO aux loss grads on scorer output parameters (Gate 2).
  if (mb_aux.this_mb_has_aux && !ctx.purchase_param_indices.empty()) {
    auto params = model->parameters();
    double dot_prod = 0.0;
    double aux_sq = 0.0;
    double ppo_sq = 0.0;
    for (size_t pi : ctx.purchase_param_indices) {
      if (pi >= params.size()) continue;
      auto tot = params[pi].grad();
      if (!tot.defined()) continue;
      torch::Tensor g_aux =
          (pi < mb_aux.aux_grad_snapshot.size() && mb_aux.aux_grad_snapshot[pi].defined())
              ? mb_aux.aux_grad_snapshot[pi]
              : torch::zeros_like(tot);
      torch::Tensor g_ppo = tot - g_aux;
      dot_prod += (g_aux * g_ppo).sum().to(torch::kFloat64).item<double>();
      aux_sq += (g_aux * g_aux).sum().to(torch::kFloat64).item<double>();
      ppo_sq += (g_ppo * g_ppo).sum().to(torch::kFloat64).item<double>();
    }
    if (aux_sq > 1e-12 && ppo_sq > 1e-12) {
      double cosine_sim = dot_prod / (std::sqrt(aux_sq) * std::sqrt(ppo_sq));
      stats.purchase_grad_cosine_sum += cosine_sim;
      stats.purchase_grad_cosine_count++;
    }
  }

  if (mb_aux.this_mb_has_aux && !ctx.action_head_param_indices.empty()) {
    auto params = model->parameters();
    double dot_prod = 0.0;
    double aux_sq = 0.0;
    double ppo_sq = 0.0;
    for (size_t pi : ctx.action_head_param_indices) {
      if (pi >= params.size()) continue;
      auto tot = params[pi].grad();
      if (!tot.defined()) continue;
      torch::Tensor g_aux =
          (pi < mb_aux.aux_grad_snapshot.size() && mb_aux.aux_grad_snapshot[pi].defined())
              ? mb_aux.aux_grad_snapshot[pi]
              : torch::zeros_like(tot);
      torch::Tensor g_ppo = tot - g_aux;
      dot_prod += (g_aux * g_ppo).sum().to(torch::kFloat64).item<double>();
      aux_sq += (g_aux * g_aux).sum().to(torch::kFloat64).item<double>();
      ppo_sq += (g_ppo * g_ppo).sum().to(torch::kFloat64).item<double>();
    }
    if (aux_sq > 1e-12 && ppo_sq > 1e-12) {
      double cosine_sim = dot_prod / (std::sqrt(aux_sq) * std::sqrt(ppo_sq));
      stats.action_head_grad_cosine_sum += cosine_sim;
      stats.action_head_grad_cosine_count++;
    }
  }
}

}  // namespace open_spiel::ppo_utils_internal

#endif  // OPEN_SPIEL_BUILD_WITH_LIBTORCH
