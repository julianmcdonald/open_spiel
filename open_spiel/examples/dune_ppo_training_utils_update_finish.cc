#include "dune_ppo_training_utils_update_internal.h"
#include "dune_targeted_buy.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <optional>
#include <vector>

#ifdef OPEN_SPIEL_BUILD_WITH_LIBTORCH
#include <ATen/ops/_foreach_mul.h>
#include <ATen/ops/_foreach_norm.h>
#include "dune_seed_utils.h"
#include "open_spiel/abseil-cpp/absl/flags/flag.h"

namespace open_spiel::ppo_utils_internal {

void RunTrainPpoUpdateFinishPass(TrainPpoUpdateContext& ctx) {
  auto& batch = *ctx.batch;
  auto& model = ctx.model;
  auto& optimizer = *ctx.optimizer;
  auto& stats = ctx.stats;
  const torch::Device device = ctx.device;
  const int64_t n = ctx.n;
  const int64_t n_train = ctx.n_train;
  const int64_t obs_size = ctx.obs_size;
  const int64_t action_dim = ctx.action_dim;
  const int64_t minibatch_size = ctx.minibatch_size;
  const int64_t aux_n = ctx.aux_n;
  const bool r755_hybrid = ctx.r755_hybrid;

  auto cpu_float = torch::TensorOptions().dtype(torch::kFloat32);
  auto cpu_bool = torch::TensorOptions().dtype(torch::kBool);
  auto cpu_long = torch::TensorOptions().dtype(torch::kInt64);

  // --- Opt-in Learner Unvisited Aux Finish Pass (--r755_finish_unvisited_aux_on_kl_stop) ---
  if (::absl::GetFlag(::FLAGS_r755_finish_unvisited_aux_on_kl_stop) && ctx.aux_active && stats.early_stopped) {
    int64_t visited_count = 0;
    for (uint8_t b : ctx.aux_visited_mask) {
      if (b != 0) ++visited_count;
    }
    if (visited_count < aux_n) {
      std::vector<int64_t> unvisited_indices;
      unvisited_indices.reserve(aux_n - visited_count);
      for (int64_t i = 0; i < aux_n; ++i) {
        if (ctx.aux_visited_mask[i] == 0) {
          unvisited_indices.push_back(i);
        }
      }
      const int64_t num_mb_train =
          std::max<int64_t>(1, (n_train + minibatch_size - 1) / minibatch_size);
      const int64_t normal_slice = (aux_n + num_mb_train - 1) / num_mb_train;
      const int64_t target_chunk_size =
          std::max<int64_t>(1, std::min<int64_t>(64, normal_slice));
      const int64_t total_unvisited =
          static_cast<int64_t>(unvisited_indices.size());
      int64_t finish_step = 0;

      for (int64_t offset = 0; offset < total_unvisited;
           offset += target_chunk_size, ++finish_step) {
        const int64_t this_chunk_len =
            std::min(target_chunk_size, total_unvisited - offset);

        bool is_contiguous =
            (unvisited_indices[offset + this_chunk_len - 1] -
             unvisited_indices[offset] == this_chunk_len - 1);
        torch::Tensor a_states, a_masks, a_targets, a_values;
        if (is_contiguous) {
          int64_t as = unvisited_indices[offset];
          a_states = ctx.aux_states.narrow(0, as, this_chunk_len);
          a_masks = ctx.aux_masks.narrow(0, as, this_chunk_len);
          a_targets = ctx.aux_targets.narrow(0, as, this_chunk_len);
          a_values = ctx.aux_values.narrow(0, as, this_chunk_len);
        } else {
          std::vector<int64_t> chunk_idx_vec;
          chunk_idx_vec.reserve(this_chunk_len);
          for (int64_t k = 0; k < this_chunk_len; ++k) {
            chunk_idx_vec.push_back(unvisited_indices[offset + k]);
          }
          torch::Tensor idx_t = torch::tensor(
              chunk_idx_vec,
              torch::TensorOptions().dtype(torch::kInt64).device(device));
          a_states = ctx.aux_states.index_select(0, idx_t);
          a_masks = ctx.aux_masks.index_select(0, idx_t);
          a_targets = ctx.aux_targets.index_select(0, idx_t);
          a_values = ctx.aux_values.index_select(0, idx_t);
        }

        std::vector<const dune_semantic::CandidateActionData*> a_cands;
        if (model->with_semantic_scorer_) {
          a_cands.reserve(this_chunk_len);
          for (int64_t k = 0; k < this_chunk_len; ++k) {
            a_cands.push_back(ctx.all_aux_cands[unvisited_indices[offset + k]]);
          }
        }

        const int64_t anchor_len = std::min(minibatch_size, n_train);
        int64_t anchor_start = 0;
        if (n_train > 0 && anchor_len > 0) {
          anchor_start = (finish_step * anchor_len) % n_train;
          if (anchor_start + anchor_len > n_train) {
            anchor_start = 0;
          }
        }

        torch::Tensor mb_states, mb_masks, dummy_actions, dummy_olp, mb_nontrivial;
        std::vector<const dune_semantic::CandidateActionData*> mb_cands;
        int64_t mb_num_nontrivial = 0;
        if (ctx.ema_magnet_active && anchor_len > 0) {
          StageAndUploadMinibatch(
              ctx, /*mb_indices=*/nullptr, anchor_start, anchor_len,
              /*full_ppo_fields=*/false, mb_states, mb_masks, dummy_actions,
              dummy_olp, mb_nontrivial,
              model->with_semantic_scorer_ ? &mb_cands : nullptr,
              /*out_advantages=*/nullptr, /*out_returns=*/nullptr,
              /*out_old_values=*/nullptr, &mb_num_nontrivial);
        }

        torch::Tensor f_logp_before_cpu;
        torch::Tensor f_masks_cpu;
        torch::Tensor f_targets_cpu;
        std::vector<std::pair<Action, Action>> f_pairs;
        if (r755_hybrid && !::absl::GetFlag(::FLAGS_r765_absorption_diag_jsonl).empty()) {
          f_masks_cpu = a_masks.detach().to(torch::kCPU);
          f_targets_cpu = a_targets.detach().to(torch::kFloat32).cpu();
          f_pairs.reserve(this_chunk_len);
          for (int64_t k = 0; k < this_chunk_len; ++k) {
            f_pairs.push_back(ctx.all_aux_mpo_pairs[unvisited_indices[offset + k]]);
          }
        }

        optimizer.zero_grad();

        std::vector<double> f_slice_raw_w;
        std::vector<int> f_slice_ages;
        f_slice_raw_w.reserve(this_chunk_len);
        f_slice_ages.reserve(this_chunk_len);
        for (int64_t k = 0; k < this_chunk_len; ++k) {
          int64_t row_idx = unvisited_indices[offset + k];
          f_slice_raw_w.push_back(ctx.all_aux_row_weights[row_idx]);
          f_slice_ages.push_back(ctx.all_aux_ages[row_idx]);
        }
        std::vector<float> f_eff_w;
        bool f_slice_cap_active = false;

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
            f_logp_before_cpu = alogp.detach().to(torch::kFloat32).cpu();
          }
          if (r755_hybrid) {
            torch::Tensor ce_per_row = -(a_targets * alogp).sum(-1);
            ce_val = ce_per_row.mean().item<double>();
            vmse_val = 0.0;
            const int64_t denom_rows = anchor_len > 0 ? anchor_len : minibatch_size;
            bool cap_active = false;
            f_eff_w = ComputeWeightedHybridAuxRowWeights(
                f_slice_raw_w, denom_rows,
                ::absl::GetFlag(::FLAGS_r765_aux_norm_mode),
                ::absl::GetFlag(::FLAGS_r765_aux_max_slice_coef),
                &cap_active);
            f_slice_cap_active = cap_active;
            torch::Tensor eff_w_t = torch::from_blob(
                f_eff_w.data(), {static_cast<int64_t>(f_eff_w.size())},
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
            scaled_aux = static_cast<float>(ctx.search_loss_coef) *
                         (ce + static_cast<float>(aux_v_coef) * vmse);
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
          if (f_slice_cap_active) stats.aux_cap_active_slices++;
          for (size_t k = 0; k < f_slice_ages.size(); ++k) {
            int age = f_slice_ages[k];
            if (age >= 0 && age < PpoUpdateStats::kMaxR765ReplayAges) {
              stats.aux_presentations_by_age[age]++;
              if (k < f_eff_w.size()) {
                stats.aux_eff_weight_sum_by_age[age] += f_eff_w[k];
              }
            }
          }
        }

        if (ctx.ema_magnet_active && anchor_len > 0) {
          const int64_t cfg_micro_mb = ::absl::GetFlag(::FLAGS_ppo_microbatch_size);
          if (cfg_micro_mb > 0 && !::absl::GetFlag(::FLAGS_use_wave1_ppo_update)) {
            if (mb_num_nontrivial > 0) {
              const bool train_amp =
                  (device.is_cuda() && ::absl::GetFlag(::FLAGS_train_amp));
              const float nt_denom = static_cast<float>(mb_num_nontrivial);
              for (int64_t sub_s = 0; sub_s < anchor_len; sub_s += cfg_micro_mb) {
                const int64_t sub_len = std::min(cfg_micro_mb, anchor_len - sub_s);
                torch::Tensor sub_states = mb_states.narrow(0, sub_s, sub_len);
                torch::Tensor sub_masks = mb_masks.narrow(0, sub_s, sub_len);
                torch::Tensor sub_nt = mb_nontrivial.narrow(0, sub_s, sub_len);
                std::vector<const dune_semantic::CandidateActionData*> sub_cands;
                if (!mb_cands.empty()) {
                  sub_cands.assign(mb_cands.begin() + sub_s,
                                   mb_cands.begin() + sub_s + sub_len);
                }

                torch::Tensor sub_ema_log_probs;
                torch::Tensor sub_ema_probs;
                {
                  torch::NoGradGuard no_grad;
                  auto run_ema_anchor = [&]() {
                    auto ema_out = ctx.ema_model->forward(sub_states);
                    if (ctx.ema_model->with_semantic_scorer_ &&
                        ctx.ema_model->semantic_scorer_) {
                      dune_semantic::ApplySemanticScorerBatch(
                          ctx.ema_model->semantic_scorer_, ema_out.trunk, sub_cands,
                          ema_out.logits, device);
                    }
                    torch::Tensor ema_logits = CenterAndCapLogitsTensor(
                        ema_out.logits, sub_masks, ctx.logit_cap);
                    torch::Tensor masked_ema_logits =
                        ema_logits.masked_fill(sub_masks.logical_not(), -1e9f);
                    sub_ema_log_probs = torch::log_softmax(masked_ema_logits, -1);
                    sub_ema_probs = sub_ema_log_probs.exp();
                  };
                  if (train_amp) {
                    AutocastGuard ema_autocast(c10::DeviceType::CUDA, true);
                    run_ema_anchor();
                  } else {
                    run_ema_anchor();
                  }
                }

                torch::Tensor sub_ema_loss;
                auto compute_sub_ema_anchor = [&]() {
                  auto outputs = model->forward(sub_states);
                  if (model->with_semantic_scorer_ && model->semantic_scorer_) {
                    dune_semantic::ApplySemanticScorerBatch(
                        model->semantic_scorer_, outputs.trunk, sub_cands,
                        outputs.logits, device);
                  }
                  torch::Tensor logits = CenterAndCapLogitsTensor(
                      outputs.logits, sub_masks, ctx.logit_cap);
                  torch::Tensor masked_logits =
                      logits.masked_fill(sub_masks.logical_not(), -1e9f);
                  torch::Tensor log_probs = torch::log_softmax(masked_logits, -1);

                  torch::Tensor log_diff = sub_ema_log_probs - log_probs;
                  torch::Tensor kl_elem = sub_ema_probs * log_diff;
                  torch::Tensor legal_kl = torch::where(sub_masks, kl_elem, 0.0f);
                  torch::Tensor decision_kl = legal_kl.sum(-1);
                  torch::Tensor sub_masked_kl = decision_kl.masked_select(sub_nt);
                  if (sub_masked_kl.numel() > 0) {
                    torch::Tensor sub_ema_kl = sub_masked_kl.sum() / nt_denom;
                    sub_ema_loss =
                        static_cast<float>(ctx.ema_magnet_kl_coef) * sub_ema_kl;
                  }
                };
                if (train_amp) {
                  AutocastGuard autocast_guard(c10::DeviceType::CUDA, true);
                  compute_sub_ema_anchor();
                } else {
                  compute_sub_ema_anchor();
                }
                if (sub_ema_loss.defined()) {
                  sub_ema_loss.backward();
                }
              }
            }
          } else {
            torch::Tensor scaled_ema_loss;
            auto compute_ema_anchor = [&]() {
              auto outputs = model->forward(mb_states);
              if (model->with_semantic_scorer_ && model->semantic_scorer_) {
                dune_semantic::ApplySemanticScorerBatch(
                    model->semantic_scorer_, outputs.trunk, mb_cands, outputs.logits,
                    device);
              }
              torch::Tensor logits =
                  CenterAndCapLogitsTensor(outputs.logits, mb_masks, ctx.logit_cap);
              torch::Tensor masked_logits =
                  logits.masked_fill(mb_masks.logical_not(), -1e9f);
              torch::Tensor log_probs = torch::log_softmax(masked_logits, -1);

              torch::Tensor ema_log_probs;
              torch::Tensor ema_probs;
              {
                torch::NoGradGuard no_grad;
                auto ema_out = ctx.ema_model->forward(mb_states);
                if (ctx.ema_model->with_semantic_scorer_ && ctx.ema_model->semantic_scorer_) {
                  dune_semantic::ApplySemanticScorerBatch(
                      ctx.ema_model->semantic_scorer_, ema_out.trunk, mb_cands,
                      ema_out.logits, device);
                }
                torch::Tensor ema_logits =
                    CenterAndCapLogitsTensor(ema_out.logits, mb_masks, ctx.logit_cap);
                torch::Tensor masked_ema_logits =
                    ema_logits.masked_fill(mb_masks.logical_not(), -1e9f);
                ema_log_probs = torch::log_softmax(masked_ema_logits, -1);
                ema_probs = ema_log_probs.exp();
              }

              torch::Tensor log_diff = ema_log_probs - log_probs;
              torch::Tensor kl_elem = ema_probs * log_diff;
              torch::Tensor legal_kl =
                  ::absl::GetFlag(::FLAGS_use_wave1_ppo_update)
                      ? torch::where(mb_masks, kl_elem, torch::zeros_like(kl_elem))
                      : torch::where(mb_masks, kl_elem, 0.0f);
              torch::Tensor decision_kl = legal_kl.sum(-1);
              torch::Tensor masked_kl = decision_kl.masked_select(mb_nontrivial);
              if (masked_kl.numel() > 0) {
                torch::Tensor ema_magnet_kl_t = masked_kl.mean();
                scaled_ema_loss =
                    static_cast<float>(ctx.ema_magnet_kl_coef) * ema_magnet_kl_t;
              }
            };
            if (device.is_cuda() && ::absl::GetFlag(::FLAGS_train_amp)) {
              AutocastGuard autocast_guard(c10::DeviceType::CUDA, true);
              compute_ema_anchor();
            } else {
              compute_ema_anchor();
            }
            if (scaled_ema_loss.defined()) {
              scaled_ema_loss.backward();
            }
          }
        }

        const double max_norm = ::absl::GetFlag(::FLAGS_grad_clip_norm);
        std::vector<torch::Tensor> grads_to_clip;
        auto model_params = model->parameters();
        grads_to_clip.reserve(model_params.size());
        for (const auto& p : model_params) {
          if (p.grad().defined()) {
            grads_to_clip.push_back(p.grad());
          }
        }
        torch::Tensor total_norm_t;
        ctx.phase_timer->Begin(kPhaseGradClip);
        if (!::absl::GetFlag(::FLAGS_use_legacy_unoptimized_reads) &&
            !::absl::GetFlag(::FLAGS_use_wave1_ppo_update) &&
            device.is_cuda() && !grads_to_clip.empty()) {
          auto norms = at::_foreach_norm(grads_to_clip, 2.0);
          total_norm_t =
              (norms.size() == 1) ? norms[0] : torch::stack(norms).norm(2.0);
          auto clip_coef = torch::clamp(
              static_cast<float>(max_norm) / (total_norm_t + 1e-6f),
              std::nullopt, 1.0f);
          at::_foreach_mul_(grads_to_clip, clip_coef);
        } else {
          double grad_norm_legacy = torch::nn::utils::clip_grad_norm_(
              model_params, max_norm);
          total_norm_t = torch::tensor(
              grad_norm_legacy,
              torch::TensorOptions().dtype(torch::kFloat64).device(device));
        }
        ctx.phase_timer->End(kPhaseGradClip);

        double grad_norm = total_norm_t.item<double>();
        if (std::isnan(grad_norm) || std::isinf(grad_norm)) {
          stats.nonfinite_abort = true;
          std::cerr << "Fatal PPO gradient norm in aux finish pass: " << grad_norm
                    << "\n";
          std::exit(EXIT_FAILURE);
        }
        stats.grad_norm_sum += grad_norm;
        stats.grad_norm_count += 1;
        if (grad_norm > stats.grad_norm_max) stats.grad_norm_max = grad_norm;

        ExecuteOptimizerStep(ctx);

        if (r755_hybrid && !::absl::GetFlag(::FLAGS_r765_absorption_diag_jsonl).empty() &&
            f_logp_before_cpu.defined()) {
          RecordT1SliceDeltaLogOdds(
              ctx, a_states, a_masks, f_masks_cpu, f_targets_cpu,
              f_logp_before_cpu, a_cands, f_pairs, f_slice_ages);
        }

        for (int64_t k = 0; k < this_chunk_len; ++k) {
          ctx.aux_visited_mask[unvisited_indices[offset + k]] = 1;
        }
        ctx.aux_ce_sum += ce_val;
        ctx.aux_vmse_sum += vmse_val;
        stats.aux_total_presentations += this_chunk_len;
        ++ctx.aux_slice_uses;
        ++stats.aux_finish_pass_steps;
        stats.aux_finish_pass_rows += this_chunk_len;
      }
    }
  }

  // Step 3 update-path telemetry: evaluate post-update log probabilities and record telemetry.
  if (!ctx.forced_indices.empty()) {
    std::vector<float> log_pi_after = ComputeForcedLogProbs(ctx, ctx.forced_indices);
    for (size_t i = 0; i < ctx.forced_indices.size(); ++i) {
      const auto& trans = batch[ctx.forced_indices[i]];
      float lp_before = ctx.log_pi_before[i];
      float lp_after = log_pi_after[i];
      float ratio_pre = std::exp(lp_before - trans.old_log_prob);
      float adv_used_raw = trans.advantage;
      float adv_used_norm = trans.norm_advantage;
      float gae_adv = trans.gae_advantage;
      float full_return_minus_val = trans.mc_return - trans.value;

      open_spiel::dune_targeted_buy::RecordForcedUpdateTelemetry(
          ctx.global_update, trans.targeted_buy_card, trans.targeted_buy_round,
          trans.old_log_prob, lp_before, lp_after, ratio_pre,
          adv_used_raw, adv_used_norm, gae_adv, full_return_minus_val);
    }
  }

  // WO-PERF-1 Part B: the one host readback per update.
  if (ctx.compute_diagnostics && ctx.grad_telemetry_accumulated && stats.head_grad_norm_count > 0) {
    torch::Tensor head_norms_cpu = ctx.head_norm_accum.to(torch::kCPU);
    auto head_norms = head_norms_cpu.accessor<double, 1>();
    stats.policy_head_grad_norm_sum += head_norms[0];
    stats.value_head_grad_norm_sum += head_norms[1];
    stats.trunk_grad_norm_sum += head_norms[2];
    stats.final_vp_head_grad_norm_sum += head_norms[3];
    stats.terminal_round_head_grad_norm_sum += head_norms[4];
    stats.next_own_action_head_grad_norm_sum += head_norms[5];
  }

  // Phase 18B combined-optimization diagnostics + per-update abort decision.
  if (ctx.aux_slice_uses > 0) {
    stats.aux_ce = ctx.aux_ce_sum / static_cast<double>(ctx.aux_slice_uses);
    stats.search_aux_loss = stats.aux_ce;
    stats.aux_value_mse = ctx.aux_vmse_sum / static_cast<double>(ctx.aux_slice_uses);
  }
  if (ctx.aux_active) {
    int64_t dv = 0;
    for (uint8_t b : ctx.aux_visited_mask) dv += (b != 0 ? 1 : 0);
    stats.aux_distinct_visited = dv;
    stats.aux_visited_mask = std::move(ctx.aux_visited_mask);
  }
  if (ctx.aux_norm_count > 0) {
    stats.aux_grad_norm_mean =
        ctx.aux_norm_sum / static_cast<double>(ctx.aux_norm_count);
    stats.ppo_grad_norm_mean =
        ctx.ppo_only_norm_sum / static_cast<double>(ctx.aux_norm_count);
    stats.aux_ppo_norm_ratio =
        (stats.ppo_grad_norm_mean > 0.0)
            ? (stats.aux_grad_norm_mean / stats.ppo_grad_norm_mean)
            : 0.0;
    double effective_abort_ratio = stats.aux_ppo_norm_ratio;
    if (r755_hybrid &&
        ::absl::GetFlag(::FLAGS_r765_aux_norm_mode) == "per_row_capped") {
      const int64_t num_mb_epoch =
          std::max<int64_t>(1, (n_train + minibatch_size - 1) / minibatch_size);
      const int64_t update_mb_denom =
          std::max<int64_t>({ctx.aux_norm_count, stats.minibatches, num_mb_epoch});
      const double update_wide_aux_mean =
          ctx.aux_norm_sum / static_cast<double>(update_mb_denom);
      effective_abort_ratio =
          (stats.ppo_grad_norm_mean > 0.0)
              ? (update_wide_aux_mean / stats.ppo_grad_norm_mean)
              : 0.0;
    }
    if (ctx.abort_grad_norm_ratio > 0.0 &&
        effective_abort_ratio > ctx.abort_grad_norm_ratio) {
      stats.aux_ratio_abort = true;
    }
  }

  if (stats.purchase_grad_cosine_count > 0) {
    stats.purchase_grad_cosine =
        stats.purchase_grad_cosine_sum / static_cast<double>(stats.purchase_grad_cosine_count);
  }
  stats.action_head_grad_cosine = stats.action_head_grad_cosine_count > 0
      ? stats.action_head_grad_cosine_sum / static_cast<double>(stats.action_head_grad_cosine_count)
      : 0.0;

  if (stats.minibatches > 0) {
    stats.value_loss = ctx.value_loss_sum / stats.minibatches;
    stats.value_clip_fraction = ctx.value_clip_frac_sum / stats.minibatches;
    if (ctx.total_nontrivial_count > 0) {
      stats.policy_loss = ctx.weighted_policy_loss_sum / ctx.total_nontrivial_count;
      stats.entropy = ctx.weighted_entropy_sum / ctx.total_nontrivial_count;
      stats.approx_kl = (ctx.total_onpolicy_nontrivial_count > 0)
                            ? (ctx.weighted_kl_sum / ctx.total_onpolicy_nontrivial_count)
                            : 0.0;
      stats.clip_fraction = ctx.weighted_clip_fraction_sum / ctx.total_nontrivial_count;
    } else {
      stats.policy_loss = 0.0;
      stats.entropy = 0.0;
      stats.approx_kl = 0.0;
      stats.clip_fraction = 0.0;
    }
    if (ctx.reverse_kl_active && ctx.total_nontrivial_count > 0) {
      stats.reverse_kl = ctx.weighted_reverse_kl_sum / ctx.total_nontrivial_count;
    } else {
      stats.reverse_kl = 0.0;
    }
    if (ctx.ema_magnet_active && ctx.ema_magnet_nontrivial_count > 0) {
      stats.ema_magnet_kl = ctx.weighted_ema_magnet_kl_sum / ctx.ema_magnet_nontrivial_count;
    } else {
      stats.ema_magnet_kl = 0.0;
    }
  }

  stats.optimizer_minibatches = stats.minibatches;

  if (ctx.study_readouts) {
    stats.single_action_value_mse_after = ComputeValueMse(ctx, ctx.sa_indices);

    // Uniform sample of unfiltered nontrivial decisions for KL diagnostic
    std::vector<int64_t> nontrivial_indices;
    nontrivial_indices.reserve(ctx.nontrivial_count);
    for (int64_t i = 0; i < n; ++i) {
      if (batch[i].legal_actions.size() > 1 && !batch[i].mask_policy_loss) {
        nontrivial_indices.push_back(i);
      }
    }
    if (!nontrivial_indices.empty()) {
      if (nontrivial_indices.size() > 1024) {
        uint64_t kl_seed = dune_seed::DeriveSeed(
            ctx.master, dune_seed::kDomainTrain, ctx.global_update, 0,
            dune_seed::kStreamUnfilteredKlDiag);
        std::mt19937_64 kl_rng(kl_seed);
        std::shuffle(nontrivial_indices.begin(), nontrivial_indices.end(), kl_rng);
        nontrivial_indices.resize(1024);
        std::sort(nontrivial_indices.begin(), nontrivial_indices.end());
      }
      stats.unfiltered_kl_sample_size = static_cast<int64_t>(nontrivial_indices.size());

      torch::NoGradGuard no_grad;
      bool was_training = model->is_training();
      model->eval();

      double kl_sum = 0.0;
      double ema_kl_sum = 0.0;
      int64_t kl_count = 0;

      for (size_t start = 0; start < nontrivial_indices.size(); start += minibatch_size) {
        size_t end = std::min(start + static_cast<size_t>(minibatch_size), nontrivial_indices.size());
        int64_t chunk_len = end - start;

        torch::Tensor s_dev = torch::empty({chunk_len, obs_size}, cpu_float);
        torch::Tensor m_dev = torch::zeros({chunk_len, action_dim}, cpu_bool);
        torch::Tensor a_dev = torch::empty({chunk_len}, cpu_long);
        torch::Tensor olp_dev = torch::empty({chunk_len}, cpu_float);

        float* sp = s_dev.data_ptr<float>();
        bool* mp = m_dev.data_ptr<bool>();
        int64_t* ap = a_dev.data_ptr<int64_t>();
        float* olpp = olp_dev.data_ptr<float>();

        std::vector<const dune_semantic::CandidateActionData*> cands(chunk_len, nullptr);

        for (int64_t i = 0; i < chunk_len; ++i) {
          int64_t idx = nontrivial_indices[start + i];
          const auto& trans = batch[idx];
          std::memcpy(sp + i * obs_size, trans.state.data(), obs_size * sizeof(float));
          for (Action a : trans.legal_actions) {
            if (a >= 0 && a < action_dim) {
              mp[i * action_dim + a] = true;
            }
          }
          ap[i] = trans.action;
          olpp[i] = trans.old_log_prob;
          cands[i] = &trans.candidate_data;
        }

        torch::Tensor ds = s_dev.to(device);
        torch::Tensor dm = m_dev.to(device);
        torch::Tensor da = a_dev.to(device);
        torch::Tensor dolp = olp_dev.to(device);

        auto outputs = model->forward(ds);
        if (model->with_semantic_scorer_ && model->semantic_scorer_) {
          dune_semantic::ApplySemanticScorerBatch(
              model->semantic_scorer_, outputs.trunk, cands, outputs.logits, device);
        }
        torch::Tensor logits = CenterAndCapLogitsTensor(outputs.logits, dm, ctx.logit_cap);
        torch::Tensor masked_logits = logits.masked_fill(dm.logical_not(), -1e9f);
        torch::Tensor log_probs = torch::log_softmax(masked_logits, -1);
        torch::Tensor selected_lp = log_probs.gather(1, da.unsqueeze(1)).squeeze(1);

        torch::Tensor log_ratio = selected_lp - dolp;
        torch::Tensor ratio = torch::exp(log_ratio);
        torch::Tensor approx_kl = (ratio - 1.0f) - log_ratio;

        kl_sum += approx_kl.sum().item<double>();
        if (ctx.ema_model != nullptr) {
          auto ema_out = ctx.ema_model->forward(ds);
          if (ctx.ema_model->with_semantic_scorer_ && ctx.ema_model->semantic_scorer_) {
            dune_semantic::ApplySemanticScorerBatch(
                ctx.ema_model->semantic_scorer_, ema_out.trunk, cands, ema_out.logits, device);
          }
          torch::Tensor ema_logits = CenterAndCapLogitsTensor(ema_out.logits, dm, ctx.logit_cap);
          torch::Tensor masked_ema_logits = ema_logits.masked_fill(dm.logical_not(), -1e9f);
          torch::Tensor ema_lp = torch::log_softmax(masked_ema_logits, -1);
          torch::Tensor ema_p = torch::softmax(masked_ema_logits, -1);

          torch::Tensor log_diff = ema_lp - log_probs;
          torch::Tensor kl_elem = ema_p * log_diff;
          torch::Tensor legal_kl = torch::where(dm, kl_elem, torch::zeros_like(kl_elem));
          ema_kl_sum += legal_kl.sum(-1).sum().item<double>();
        }
        kl_count += chunk_len;
      }

      if (was_training) model->train();
      stats.unfiltered_kl = kl_count > 0 ? (kl_sum / kl_count) : 0.0;
      if (ctx.ema_model != nullptr && kl_count > 0) {
        stats.ema_magnet_kl_post = ema_kl_sum / kl_count;
        if (!ctx.ema_magnet_active) {
          stats.ema_magnet_kl = stats.ema_magnet_kl_post;
        }
      }
    }
  }

  if (!ctx.study_readouts && ctx.ema_model != nullptr && ctx.compute_diagnostics) {
    std::vector<int64_t> nontrivial_indices;
    nontrivial_indices.reserve(ctx.nontrivial_count);
    for (int64_t i = 0; i < n; ++i) {
      if (batch[i].legal_actions.size() > 1 && !batch[i].mask_policy_loss) {
        nontrivial_indices.push_back(i);
      }
    }
    if (!nontrivial_indices.empty()) {
      if (nontrivial_indices.size() > 1024) {
        uint64_t kl_seed = dune_seed::DeriveSeed(
            ctx.master, dune_seed::kDomainTrain, ctx.global_update, 0,
            dune_seed::kStreamUnfilteredKlDiag);
        std::mt19937_64 kl_rng(kl_seed);
        std::shuffle(nontrivial_indices.begin(), nontrivial_indices.end(), kl_rng);
        nontrivial_indices.resize(1024);
        std::sort(nontrivial_indices.begin(), nontrivial_indices.end());
      }
      torch::NoGradGuard no_grad;
      bool was_training = model->is_training();
      model->eval();
      double ema_kl_sum = 0.0;
      int64_t kl_count = 0;

      for (size_t start = 0; start < nontrivial_indices.size(); start += minibatch_size) {
        size_t end = std::min(start + static_cast<size_t>(minibatch_size), nontrivial_indices.size());
        int64_t chunk_len = end - start;

        torch::Tensor s_dev = torch::empty({chunk_len, obs_size}, cpu_float);
        torch::Tensor m_dev = torch::zeros({chunk_len, action_dim}, cpu_bool);
        float* sp = s_dev.data_ptr<float>();
        bool* mp = m_dev.data_ptr<bool>();
        std::vector<const dune_semantic::CandidateActionData*> cands(chunk_len, nullptr);

        for (int64_t i = 0; i < chunk_len; ++i) {
          int64_t idx = nontrivial_indices[start + i];
          const auto& trans = batch[idx];
          std::memcpy(sp + i * obs_size, trans.state.data(), obs_size * sizeof(float));
          for (Action a : trans.legal_actions) {
            if (a >= 0 && a < action_dim) {
              mp[i * action_dim + a] = true;
            }
          }
          cands[i] = &trans.candidate_data;
        }

        torch::Tensor ds = s_dev.to(device);
        torch::Tensor dm = m_dev.to(device);

        auto outputs = model->forward(ds);
        if (model->with_semantic_scorer_ && model->semantic_scorer_) {
          dune_semantic::ApplySemanticScorerBatch(
              model->semantic_scorer_, outputs.trunk, cands, outputs.logits, device);
        }
        torch::Tensor logits = CenterAndCapLogitsTensor(outputs.logits, dm, ctx.logit_cap);
        torch::Tensor masked_logits = logits.masked_fill(dm.logical_not(), -1e9f);
        torch::Tensor log_probs = torch::log_softmax(masked_logits, -1);

        auto ema_out = ctx.ema_model->forward(ds);
        if (ctx.ema_model->with_semantic_scorer_ && ctx.ema_model->semantic_scorer_) {
          dune_semantic::ApplySemanticScorerBatch(
              ctx.ema_model->semantic_scorer_, ema_out.trunk, cands, ema_out.logits, device);
        }
        torch::Tensor ema_logits = CenterAndCapLogitsTensor(ema_out.logits, dm, ctx.logit_cap);
        torch::Tensor masked_ema_logits = ema_logits.masked_fill(dm.logical_not(), -1e9f);
        torch::Tensor ema_lp = torch::log_softmax(masked_ema_logits, -1);
        torch::Tensor ema_p = torch::softmax(masked_ema_logits, -1);

        torch::Tensor log_diff = ema_lp - log_probs;
        torch::Tensor kl_elem = ema_p * log_diff;
        torch::Tensor legal_kl = torch::where(dm, kl_elem, torch::zeros_like(kl_elem));
        ema_kl_sum += legal_kl.sum(-1).sum().item<double>();
        kl_count += chunk_len;
      }
      if (was_training) model->train();
      if (ctx.ema_model != nullptr && kl_count > 0) {
        stats.ema_magnet_kl_post = ema_kl_sum / kl_count;
        if (!ctx.ema_magnet_active) {
          stats.ema_magnet_kl = stats.ema_magnet_kl_post;
        }
      }
    }
  }

  torch::Tensor returns_cpu_flat = ctx.returns_cpu;
  torch::Tensor old_values_cpu_flat = ctx.old_values_cpu;
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
  // The single end-of-update synchronise and the one read of every event pair.
  ctx.phase_timer->Finalize(&stats.phase_timings);
  stats.phase_timings.total_host_attributed_s +=
      stats.phase_timings.rollout_hash_host_s;
}

}  // namespace open_spiel::ppo_utils_internal

#endif  // OPEN_SPIEL_BUILD_WITH_LIBTORCH
