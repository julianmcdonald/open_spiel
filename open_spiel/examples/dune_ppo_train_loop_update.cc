#include "dune_ppo_train_run_context.h"

#include <future>

namespace open_spiel::ppo_train_internal {

#ifdef OPEN_SPIEL_BUILD_WITH_LIBTORCH

std::vector<std::vector<double>> EvalModelLegalProbs(
    const PpoTrainRunContext& ctx,
    const std::shared_ptr<SharedDunePolicyValueNetImpl>& eval_m,
    const std::vector<const std::vector<float>*>& obs_ptrs,
    const std::vector<const dune_semantic::CandidateActionData*>& cads,
    const std::vector<const std::vector<Action>*>& legals_list,
    size_t max_chunk_size) {
  std::vector<std::vector<double>> batch_probs;
  const size_t M = obs_ptrs.size();
  if (M == 0 || !eval_m) return batch_probs;
  batch_probs.reserve(M);
  if (max_chunk_size == 0) max_chunk_size = 2048;

  for (size_t chunk_start = 0; chunk_start < M; chunk_start += max_chunk_size) {
    const size_t chunk_end = std::min(M, chunk_start + max_chunk_size);
    const size_t chunk_size = chunk_end - chunk_start;

    torch::NoGradGuard no_grad;
    open_spiel::AutocastGuard autocast_guard(
        c10::DeviceType::CUDA, ctx.device.is_cuda() && absl::GetFlag(FLAGS_train_amp));
    auto obs_cpu = torch::empty(
        {static_cast<int64_t>(chunk_size), static_cast<int64_t>(ctx.obs_size)},
        torch::kFloat32);
    float* obs_ptr = obs_cpu.data_ptr<float>();
    for (size_t b = 0; b < chunk_size; ++b) {
      const auto& obs = *obs_ptrs[chunk_start + b];
      const size_t copy_n = std::min(obs.size(), static_cast<size_t>(ctx.obs_size));
      std::memcpy(obs_ptr + b * static_cast<size_t>(ctx.obs_size), obs.data(),
                  copy_n * sizeof(float));
      if (copy_n < static_cast<size_t>(ctx.obs_size)) {
        std::memset(obs_ptr + b * static_cast<size_t>(ctx.obs_size) + copy_n, 0,
                    (static_cast<size_t>(ctx.obs_size) - copy_n) * sizeof(float));
      }
    }
    auto obs_t = obs_cpu.to(ctx.device);
    auto out = eval_m->forward(obs_t);
    if (eval_m->with_semantic_scorer_ && eval_m->semantic_scorer_) {
      std::vector<const dune_semantic::CandidateActionData*> chunk_cads(
          cads.begin() + chunk_start, cads.begin() + chunk_end);
      dune_semantic::ApplySemanticScorerBatch(
          eval_m->semantic_scorer_, out.trunk, chunk_cads, out.logits, ctx.device);
    }
    auto logits_cpu = out.logits.to(torch::kCPU, torch::kFloat32).contiguous();
    const float* logits_ptr = logits_cpu.data_ptr<float>();
    const float logit_cap_f = static_cast<float>(absl::GetFlag(FLAGS_logit_cap));
    std::vector<float> row_logits(ctx.action_size);
    for (size_t b = 0; b < chunk_size; ++b) {
      std::memcpy(row_logits.data(),
                  logits_ptr + b * static_cast<size_t>(ctx.action_size),
                  static_cast<size_t>(ctx.action_size) * sizeof(float));
      const auto& legals = *legals_list[chunk_start + b];
      CenterAndCapLegalLogits(row_logits, legals, logit_cap_f);
      std::vector<double> probs;
      probs.reserve(legals.size());
      double max_l = -1e9;
      for (Action a : legals) {
        if (a >= 0 && a < ctx.action_size) {
          max_l = std::max(max_l, static_cast<double>(row_logits[a]));
        }
      }
      double sum_e = 0.0;
      for (Action a : legals) {
        double e = (a >= 0 && a < ctx.action_size)
                       ? std::exp(row_logits[a] - max_l)
                       : 0.0;
        probs.push_back(e);
        sum_e += e;
      }
      if (sum_e > 0.0) {
        for (double& p : probs) p /= sum_e;
      }
      batch_probs.push_back(std::move(probs));
    }
  }
  return batch_probs;
}

void RunLoopUpdatePhase(PpoTrainRunContext& ctx, PpoTrainStepState& step) {
  const int update = step.update;

  if (absl::GetFlag(FLAGS_anneal_lr)) {
    double frac = 1.0 - static_cast<double>(update - 1) /
                            std::max(1, ctx.target_end_update);
    double current_lr = std::max(1e-8, frac * ctx.base_lr);
    double current_new_block_lr =
        (absl::GetFlag(FLAGS_new_block_lr) > 0.0)
            ? std::max(1e-8, frac * absl::GetFlag(FLAGS_new_block_lr))
            : 0.0;
    SetOptimizerLearningRate(*ctx.optimizer, current_lr, current_new_block_lr);
    if (absl::GetFlag(FLAGS_separate_actor_critic) && ctx.critic_optimizer != nullptr) {
      SetOptimizerLearningRate(*ctx.critic_optimizer, current_lr);
    }
  }

  step.wall_start = std::chrono::high_resolution_clock::now();

  // Launch background collection for the next update (pipelined).
  step.have_bg = ctx.pipeline && update < ctx.target_end_update;
  if (step.have_bg) {
    float next_reward_lambda = ComputeRewardLambda(ctx.total_env_steps.load(),
                                                   absl::GetFlag(FLAGS_shaping_start_env_steps),
                                                   absl::GetFlag(FLAGS_shaping_decay_env_steps));
    step.bg_collect_thread = std::thread([&ctx, &step, update, next_reward_lambda]() {
      step.next_collect = CollectRollout(
          ctx.game.get(), ctx.evaluator, ctx.obs_size, &ctx.total_env_steps,
          ctx.num_threads, &ctx.next_episode_id, ctx.rollout_games,
          next_reward_lambda, ctx.opponent_evaluators,
          /*global_update=*/update + 1);
      // Phase 18B: aux for the NEXT update, from the same frozen snapshot, run
      // sequentially AFTER the rollout (design: collect PPO first, then aux).
      ctx.RunAuxCollection(update + 1, step.next_collect);
    });
  }

  step.pairing_mode = absl::GetFlag(FLAGS_search_pairing)
      ? open_spiel::dune_imperium::SearchPairingMode::kPaired
      : open_spiel::dune_imperium::SearchPairingMode::kUnpaired;
  step.truncation_mode = absl::GetFlag(FLAGS_search_truncation)
      ? open_spiel::dune_imperium::SearchTruncationMode::kNextRoundSearchPlayer
      : open_spiel::dune_imperium::SearchTruncationMode::kTerminal;

  step.ppo_start = std::chrono::high_resolution_clock::now();
  if (absl::GetFlag(FLAGS_search_in_training) &&
      !absl::GetFlag(FLAGS_search_aux_decoupled) &&
      !ctx.current_collect.search_snapshots.empty()) {
    uint64_t search_seed = dune_seed::DeriveSeed(
        ctx.master, dune_seed::kDomainTrain, update, /*stream=*/0x53555045ULL);
    step.search_sup_res = open_spiel::dune_imperium::BatchCompoundSearchSupervision(
        ctx.current_collect.search_snapshots,
        ctx.search_teacher_evaluators,
        absl::GetFlag(FLAGS_search_top_k),
        absl::GetFlag(FLAGS_search_rollouts_per_action),
        absl::GetFlag(FLAGS_search_override_margin),
        absl::GetFlag(FLAGS_search_temperature),
        absl::GetFlag(FLAGS_search_blend_eta),
        search_seed,
        absl::GetFlag(FLAGS_search_threads),
        absl::GetFlag(FLAGS_search_full_turn_supervision),
        /*use_mpo_target=*/true,
        /*max_target_kl=*/0.05,
        step.pairing_mode,
        step.truncation_mode,
        absl::GetFlag(FLAGS_search_target_mode),
        absl::GetFlag(FLAGS_search_override_eta));
    ctx.current_collect.aux_examples = std::move(step.search_sup_res.examples);
  }
  const double r755_coef_override = absl::GetFlag(FLAGS_r755_search_loss_coef);
  const double hybrid_coef = (r755_coef_override >= 0.0)
      ? r755_coef_override
      : absl::GetFlag(FLAGS_search_loss_coef);
  step.this_search_coef =
      absl::GetFlag(FLAGS_r755_hybrid_mode)
          ? hybrid_coef
          : (absl::GetFlag(FLAGS_search_in_training)
                 ? absl::GetFlag(FLAGS_search_loss_lambda)
                 : (ctx.online_search_collection
                        ? open_spiel::SearchLossCoefForUpdate(
                              update, ctx.aux_search_loss_coef_target,
                              ctx.aux_search_loss_warmup)
                        : 0.0));
  const bool compute_diagnostics = !absl::GetFlag(FLAGS_diagnostics_path).empty();

  R755Gate2Diagnostics gate2_diag;
  const bool do_gate2 = !absl::GetFlag(FLAGS_r755_gate2_output).empty();
  std::vector<std::vector<double>> gate2_aux_pi_old;
  std::vector<std::vector<double>> gate2_non_override_pi_old;
  std::vector<size_t> gate2_non_override_indices;
  std::vector<const std::vector<float>*> gate2_aux_obs_ptrs;
  std::vector<const dune_semantic::CandidateActionData*> gate2_aux_cads;
  std::vector<const std::vector<Action>*> gate2_aux_legals;
  std::vector<const std::vector<float>*> gate2_non_ov_obs_ptrs;
  std::vector<const dune_semantic::CandidateActionData*> gate2_non_ov_cads;
  std::vector<const std::vector<Action>*> gate2_non_ov_legals;

  auto eval_collector_probs = [&](const std::vector<float>& obs,
                                  const dune_semantic::CandidateActionData* cad,
                                  const std::vector<Action>& legals) -> std::vector<double> {
    EvalResult res = ctx.evaluator->EvaluateWithActions(obs, cad);
    CenterAndCapLegalLogits(res.logits, legals, static_cast<float>(absl::GetFlag(FLAGS_logit_cap)));
    std::vector<double> probs;
    probs.reserve(legals.size());
    double max_l = -1e9;
    for (Action a : legals) {
      if (a >= 0 && a < ctx.action_size) max_l = std::max(max_l, static_cast<double>(res.logits[a]));
    }
    double sum_e = 0.0;
    for (Action a : legals) {
      double e = (a >= 0 && a < ctx.action_size) ? std::exp(res.logits[a] - max_l) : 0.0;
      probs.push_back(e);
      sum_e += e;
    }
    if (sum_e > 0.0) {
      for (double& p : probs) p /= sum_e;
    }
    return probs;
  };

  auto eval_collector_batch_probs = [&](const std::vector<const std::vector<float>*>& obs_ptrs,
                                        const std::vector<const dune_semantic::CandidateActionData*>& cads,
                                        const std::vector<const std::vector<Action>*>& legals_list)
      -> std::vector<std::vector<double>> {
    const size_t M = obs_ptrs.size();
    std::vector<std::vector<double>> batch_probs(M);
    if (M == 0) return batch_probs;

    ctx.evaluator->AdjustActiveRollouts(static_cast<int>(M));

    std::vector<std::future<void>> futures;
    futures.reserve(M);
    for (size_t i = 0; i < M; ++i) {
      futures.push_back(open_spiel::dune_imperium::SearchRolloutThreadPool::Instance().Submit([&, i]() {
        open_spiel::dune_imperium::ActiveRolloutReleaseGuard release_guard(ctx.evaluator.get(), 1);
        batch_probs[i] = eval_collector_probs(*obs_ptrs[i], cads[i], *legals_list[i]);
      }));
    }
    for (auto& f : futures) {
      f.get();
    }
    return batch_probs;
  };

  auto kl_div = [](const std::vector<double>& p, const std::vector<double>& q) -> double {
    double kl = 0.0;
    for (size_t i = 0; i < p.size(); ++i) {
      if (p[i] > 1e-12) {
        double denom = (i < q.size() && q[i] > 1e-12) ? q[i] : 1e-12;
        kl += p[i] * std::log(p[i] / denom);
      }
    }
    return std::max(0.0, kl);
  };

  const int r765_replay_k = absl::GetFlag(FLAGS_r765_replay_window_updates);
  if (absl::GetFlag(FLAGS_r755_hybrid_mode) && r765_replay_k > 1) {
    uint64_t aux_shuf_seed = dune_seed::DeriveSeed(
        ctx.master, dune_seed::kDomainTrain, update, 0,
        dune_seed::kStreamR758AuxShuffle);
    ctx.r765_replay_mgr.AdvanceAndBuildUpdateBatch(
        update, ctx.current_collect.aux_examples, step.this_search_coef,
        absl::GetFlag(FLAGS_r765_replay_row_coef), r765_replay_k,
        aux_shuf_seed, absl::GetFlag(FLAGS_r755_shuffle_aux_examples),
        &step.train_aux_examples);
  } else {
    if (absl::GetFlag(FLAGS_r755_hybrid_mode) &&
        absl::GetFlag(FLAGS_r755_shuffle_aux_examples) &&
        ctx.current_collect.aux_examples.size() > 1) {
      uint64_t aux_shuf_seed = dune_seed::DeriveSeed(
          ctx.master, dune_seed::kDomainTrain, update, 0,
          dune_seed::kStreamR758AuxShuffle);
      std::mt19937_64 aux_shuf_rng(aux_shuf_seed);
      std::shuffle(ctx.current_collect.aux_examples.begin(),
                   ctx.current_collect.aux_examples.end(), aux_shuf_rng);
    }
    step.train_aux_examples = ctx.current_collect.aux_examples;
    if (absl::GetFlag(FLAGS_r755_hybrid_mode) && !absl::GetFlag(FLAGS_r765_absorption_diag_jsonl).empty()) {
      std::vector<open_spiel::SearchTrainingExample> dummy_comb;
      ctx.r765_replay_mgr.AdvanceAndBuildUpdateBatch(
          update, ctx.current_collect.aux_examples, step.this_search_coef,
          absl::GetFlag(FLAGS_r765_replay_row_coef), 1, 0, false, &dummy_comb);
    }
  }

  if (do_gate2) {
    gate2_diag.update = update;
    gate2_diag.total_transitions = static_cast<int64_t>(ctx.current_collect.rollout.size());
    for (const auto& t : ctx.current_collect.rollout) {
      if (std::isnan(t.advantage) || std::isinf(t.advantage) ||
          std::isnan(t.return_value) || std::isinf(t.return_value)) {
        gate2_diag.nan_inf_count++;
      }
      if (t.legal_actions.size() <= 1) {
        gate2_diag.single_action_transitions++;
      } else {
        gate2_diag.multi_action_transitions++;
        if (t.mask_policy_loss) {
          gate2_diag.override_masked_transitions++;
        } else {
          gate2_diag.ppo_multi_action_transitions++;
        }
      }
    }
    gate2_diag.mask_identity_holds =
        (gate2_diag.ppo_multi_action_transitions + gate2_diag.override_masked_transitions ==
         gate2_diag.multi_action_transitions);

    gate2_aux_obs_ptrs.reserve(ctx.current_collect.aux_examples.size());
    gate2_aux_cads.reserve(ctx.current_collect.aux_examples.size());
    gate2_aux_legals.reserve(ctx.current_collect.aux_examples.size());
    for (const auto& ex : ctx.current_collect.aux_examples) {
      gate2_aux_obs_ptrs.push_back(&ex.observation);
      gate2_aux_cads.push_back(ex.candidate_data.empty() ? nullptr : &ex.candidate_data);
      gate2_aux_legals.push_back(&ex.legal_actions);
    }
    const size_t learner_eval_chunk = static_cast<size_t>(std::max(1, absl::GetFlag(FLAGS_eval_batch_size)));
    gate2_aux_pi_old = EvalModelLegalProbs(
        ctx, ctx.training_model, gate2_aux_obs_ptrs, gate2_aux_cads,
        gate2_aux_legals, learner_eval_chunk);

    std::vector<size_t> all_non_ov_eligible;
    all_non_ov_eligible.reserve(ctx.current_collect.rollout.size());
    for (size_t i = 0; i < ctx.current_collect.rollout.size(); ++i) {
      const auto& t = ctx.current_collect.rollout[i];
      if (t.legal_actions.size() > 1 && !t.mask_policy_loss) {
        all_non_ov_eligible.push_back(i);
      }
    }
    if (all_non_ov_eligible.size() > 2048) {
      uint64_t non_ov_seed = dune_seed::DeriveSeed(
          ctx.master, dune_seed::kDomainTrain, update, 0,
          dune_seed::kStreamGate2NonOverrideDiag);
      std::mt19937_64 non_ov_rng(non_ov_seed);
      std::shuffle(all_non_ov_eligible.begin(), all_non_ov_eligible.end(), non_ov_rng);
      all_non_ov_eligible.resize(2048);
      std::sort(all_non_ov_eligible.begin(), all_non_ov_eligible.end());
    }
    gate2_non_override_indices = std::move(all_non_ov_eligible);
    gate2_non_ov_obs_ptrs.reserve(gate2_non_override_indices.size());
    gate2_non_ov_cads.reserve(gate2_non_override_indices.size());
    gate2_non_ov_legals.reserve(gate2_non_override_indices.size());
    for (size_t idx : gate2_non_override_indices) {
      const auto& t = ctx.current_collect.rollout[idx];
      gate2_non_ov_obs_ptrs.push_back(&t.state);
      gate2_non_ov_cads.push_back(t.candidate_data.empty() ? nullptr : &t.candidate_data);
      gate2_non_ov_legals.push_back(&t.legal_actions);
    }
    gate2_non_override_pi_old = EvalModelLegalProbs(
        ctx, ctx.training_model, gate2_non_ov_obs_ptrs, gate2_non_ov_cads,
        gate2_non_ov_legals, learner_eval_chunk);

    double max_abs_diff = 0.0;
    double sum_coll_vs_learn_kl = 0.0;
    size_t num_coll_vs_learn_states = 0;
    std::vector<double> coll_vs_learn_diffs;

    auto accumulate_coll_vs_learn = [&](const std::vector<double>& p_coll,
                                        const std::vector<double>& p_learn) {
      if (p_coll.size() != p_learn.size() || p_coll.empty()) return;
      sum_coll_vs_learn_kl += kl_div(p_coll, p_learn);
      num_coll_vs_learn_states++;
      for (size_t a = 0; a < p_coll.size(); ++a) {
        double logp_coll = std::log(std::max(1e-12, p_coll[a]));
        double logp_learn = std::log(std::max(1e-12, p_learn[a]));
        coll_vs_learn_diffs.push_back(std::abs(logp_learn - logp_coll));
      }
    };

    const size_t coll_check_n = std::min<size_t>(128, gate2_non_override_indices.size());
    std::vector<const std::vector<float>*> coll_check_obs(
        gate2_non_ov_obs_ptrs.begin(), gate2_non_ov_obs_ptrs.begin() + coll_check_n);
    std::vector<const dune_semantic::CandidateActionData*> coll_check_cads(
        gate2_non_ov_cads.begin(), gate2_non_ov_cads.begin() + coll_check_n);
    std::vector<const std::vector<Action>*> coll_check_legals(
        gate2_non_ov_legals.begin(), gate2_non_ov_legals.begin() + coll_check_n);

    auto gate2_non_ov_coll_probs =
        eval_collector_batch_probs(coll_check_obs, coll_check_cads, coll_check_legals);
    for (size_t j = 0; j < coll_check_n; ++j) {
      size_t idx = gate2_non_override_indices[j];
      const auto& t = ctx.current_collect.rollout[idx];
      const auto& p_coll = gate2_non_ov_coll_probs[j];
      accumulate_coll_vs_learn(p_coll, gate2_non_override_pi_old[j]);
      for (size_t a = 0; a < t.legal_actions.size(); ++a) {
        if (t.legal_actions[a] == t.action && p_coll[a] > 0.0) {
          double recomputed_logp = std::log(p_coll[a]);
          max_abs_diff = std::max(
              max_abs_diff, std::abs(recomputed_logp - static_cast<double>(t.old_log_prob)));
          break;
        }
      }
    }
    auto gate2_aux_coll_probs =
        eval_collector_batch_probs(gate2_aux_obs_ptrs, gate2_aux_cads, gate2_aux_legals);
    for (size_t i = 0; i < ctx.current_collect.aux_examples.size(); ++i) {
      accumulate_coll_vs_learn(gate2_aux_coll_probs[i], gate2_aux_pi_old[i]);
    }

    gate2_diag.logprob_recompute_max_abs_diff = max_abs_diff;
    gate2_diag.collector_vs_learner_kl =
        (num_coll_vs_learn_states > 0)
            ? (sum_coll_vs_learn_kl / static_cast<double>(num_coll_vs_learn_states))
            : 0.0;
    if (!coll_vs_learn_diffs.empty()) {
      std::sort(coll_vs_learn_diffs.begin(), coll_vs_learn_diffs.end());
      double sum_diff = 0.0;
      for (double d : coll_vs_learn_diffs) sum_diff += d;
      gate2_diag.collector_vs_learner_mean_diff =
          sum_diff / static_cast<double>(coll_vs_learn_diffs.size());
      size_t p99_idx = std::min(
          coll_vs_learn_diffs.size() - 1,
          static_cast<size_t>(std::floor(0.99 * static_cast<double>(coll_vs_learn_diffs.size() - 1))));
      gate2_diag.collector_vs_learner_p99_diff = coll_vs_learn_diffs[p99_idx];
      gate2_diag.collector_vs_learner_max_diff = coll_vs_learn_diffs.back();
    }

    // Production Pre-Step Minibatch Comparison
    const size_t pre_step_cap = static_cast<size_t>(std::max(1, absl::GetFlag(FLAGS_ppo_minibatch_size)));
    std::vector<const std::vector<float>*> pre_step_obs_ptrs;
    std::vector<const dune_semantic::CandidateActionData*> pre_step_cads;
    std::vector<const std::vector<Action>*> pre_step_legals;
    std::vector<const PpoTransition*> pre_step_transitions;

    pre_step_obs_ptrs.reserve(pre_step_cap);
    pre_step_cads.reserve(pre_step_cap);
    pre_step_legals.reserve(pre_step_cap);
    pre_step_transitions.reserve(pre_step_cap);

    for (const auto& t : ctx.current_collect.rollout) {
      if (pre_step_transitions.size() >= pre_step_cap) break;
      if (t.legal_actions.size() > 1 && !t.mask_policy_loss) {
        pre_step_transitions.push_back(&t);
        pre_step_obs_ptrs.push_back(&t.state);
        pre_step_cads.push_back(t.candidate_data.empty() ? nullptr : &t.candidate_data);
        pre_step_legals.push_back(&t.legal_actions);
      }
    }

    if (!pre_step_transitions.empty()) {
      auto pre_step_probs = EvalModelLegalProbs(
          ctx, ctx.training_model, pre_step_obs_ptrs, pre_step_cads,
          pre_step_legals, pre_step_cap);

      std::vector<double> abs_diffs;
      std::vector<double> ratios;
      abs_diffs.reserve(pre_step_transitions.size());
      ratios.reserve(pre_step_transitions.size());

      const double clip_eps = absl::GetFlag(FLAGS_ppo_clip_epsilon);
      int64_t clipped_count = 0;
      double sum_abs_diff = 0.0;

      for (size_t i = 0; i < pre_step_transitions.size(); ++i) {
        const auto& t = *pre_step_transitions[i];
        const auto& p_learn = pre_step_probs[i];
        size_t a_idx = p_learn.size();
        for (size_t a = 0; a < t.legal_actions.size(); ++a) {
          if (t.legal_actions[a] == t.action) {
            a_idx = a;
            break;
          }
        }
        if (a_idx < p_learn.size()) {
          double learner_logp = std::log(std::max(1e-12, p_learn[a_idx]));
          double diff = learner_logp - static_cast<double>(t.old_log_prob);
          double abs_diff = std::abs(diff);
          double ratio = std::exp(diff);

          abs_diffs.push_back(abs_diff);
          ratios.push_back(ratio);
          sum_abs_diff += abs_diff;

          if (ratio < 1.0 - clip_eps || ratio > 1.0 + clip_eps) {
            clipped_count++;
          }
        }
      }

      const size_t N_eval = abs_diffs.size();
      if (N_eval > 0) {
        std::sort(abs_diffs.begin(), abs_diffs.end());
        std::sort(ratios.begin(), ratios.end());

        auto percentile = [](const std::vector<double>& sorted_v, double p) -> double {
          if (sorted_v.empty()) return 0.0;
          double idx_f = p * static_cast<double>(sorted_v.size() - 1);
          size_t idx = static_cast<size_t>(std::clamp<double>(std::round(idx_f), 0.0, static_cast<double>(sorted_v.size() - 1)));
          return sorted_v[idx];
        };

        gate2_diag.pre_step_logprob_mean_abs_diff = sum_abs_diff / static_cast<double>(N_eval);
        gate2_diag.pre_step_logprob_p99_abs_diff = percentile(abs_diffs, 0.99);
        gate2_diag.pre_step_logprob_max_abs_diff = abs_diffs.back();

        gate2_diag.pre_step_ratio_min = ratios.front();
        gate2_diag.pre_step_ratio_p01 = percentile(ratios, 0.01);
        gate2_diag.pre_step_ratio_p05 = percentile(ratios, 0.05);
        gate2_diag.pre_step_ratio_p50 = percentile(ratios, 0.50);
        gate2_diag.pre_step_ratio_p95 = percentile(ratios, 0.95);
        gate2_diag.pre_step_ratio_p99 = percentile(ratios, 0.99);
        gate2_diag.pre_step_ratio_max = ratios.back();

        gate2_diag.pre_step_clip_frac = static_cast<double>(clipped_count) / static_cast<double>(N_eval);

        gate2_diag.logprob_recompute_max_abs_diff = gate2_diag.pre_step_logprob_max_abs_diff;
      }
    }
  }

  step.stats =
      (absl::GetFlag(FLAGS_separate_actor_critic) && ctx.training_critic != nullptr)
          ? open_spiel::TrainPpoUpdateSeparate(
                ctx.training_model, *ctx.optimizer, ctx.training_critic, *ctx.critic_optimizer,
                ctx.current_collect.rollout, ctx.obs_size, ctx.action_size, ctx.device,
                ctx.master, update, compute_diagnostics,
                absl::GetFlag(FLAGS_gamma),
                (absl::GetFlag(FLAGS_train_value_only) ? 1.0 : absl::GetFlag(FLAGS_gae_lambda)))
          : open_spiel::TrainPpoUpdate(
                ctx.training_model, *ctx.optimizer, ctx.current_collect.rollout,
                ctx.obs_size, ctx.action_size, ctx.device, ctx.master, update,
                ctx.anchor_model, step.train_aux_examples, step.this_search_coef,
                (ctx.online_search_collection || absl::GetFlag(FLAGS_r755_hybrid_mode))
                    ? ctx.aux_abort_ratio : 0.0,
                ctx.collection_model, absl::GetFlag(FLAGS_reverse_kl_coef),
                compute_diagnostics,
                ctx.ema_model, absl::GetFlag(FLAGS_ema_magnet_kl_coef));
  step.stats.episode_ids_unique = ctx.current_collect.episode_ids_unique;
  step.stats.forced_buys = ctx.current_collect.forced_buys;
  AttachPrecapAbszStats(&step.stats, ctx.current_collect);
  AttachCanaryStats(&step.stats, ctx.current_collect);

  if (ctx.v5_slot_residual_state.active) {
    const int completed_v5_updates =
        (ctx.v5_slot_residual_state.N > 0)
            ? (update - ctx.v5_slot_residual_state.start_global_update)
            : (update - ctx.start_update + 1);
    ctx.v5_slot_residual_state.Apply(ctx.training_model, completed_v5_updates);
  }

  if (do_gate2) {
    const size_t learner_eval_chunk = static_cast<size_t>(std::max(1, absl::GetFlag(FLAGS_eval_batch_size)));
    auto gate2_aux_pi_new = EvalModelLegalProbs(
        ctx, ctx.training_model, gate2_aux_obs_ptrs, gate2_aux_cads,
        gate2_aux_legals, learner_eval_chunk);
    auto gate2_non_override_pi_new = EvalModelLegalProbs(
        ctx, ctx.training_model, gate2_non_ov_obs_ptrs, gate2_non_ov_cads,
        gate2_non_ov_legals, learner_eval_chunk);

    const std::vector<uint8_t> fresh_visited_mask =
        (absl::GetFlag(FLAGS_r755_hybrid_mode) && r765_replay_k > 1)
            ? ctx.r765_replay_mgr.FreshCohortVisitedMask(
                  update, ctx.current_collect.aux_examples.size(),
                  step.stats.aux_visited_mask)
            : step.stats.aux_visited_mask;

    double sum_student_kl = 0.0, sum_target_kl = 0.0;
    double sum_dp_target = 0.0, sum_dp_student = 0.0;
    double vis_dp_target = 0.0, vis_dp_student = 0.0;
    double unvis_dp_target = 0.0, unvis_dp_student = 0.0;
    int64_t visited_aux_count = 0, unvisited_aux_count = 0;
    for (size_t i = 0; i < ctx.current_collect.aux_examples.size(); ++i) {
      const auto& ex = ctx.current_collect.aux_examples[i];
      const auto& p_old = gate2_aux_pi_old[i];
      const auto& p_new = gate2_aux_pi_new[i];
      sum_student_kl += kl_div(p_old, p_new);
      sum_target_kl += kl_div(ex.normalized_visits, p_old);
      const bool is_vis = (i < fresh_visited_mask.size() && fresh_visited_mask[i] != 0);
      if (is_vis) {
        ++visited_aux_count;
      } else {
        ++unvisited_aux_count;
      }
      int best_idx = -1;
      double max_diff = -1e9;
      for (size_t a = 0; a < ex.normalized_visits.size(); ++a) {
        double diff = ex.normalized_visits[a] - p_old[a];
        if (diff > max_diff) { max_diff = diff; best_idx = static_cast<int>(a); }
      }
      if (best_idx >= 0) {
        double dp_t = ex.normalized_visits[best_idx] - p_old[best_idx];
        double dp_s = p_new[best_idx] - p_old[best_idx];
        sum_dp_target += dp_t;
        sum_dp_student += dp_s;
        if (is_vis) {
          vis_dp_target += dp_t;
          vis_dp_student += dp_s;
        } else {
          unvis_dp_target += dp_t;
          unvis_dp_student += dp_s;
        }
      }
    }
    size_t n_aux = ctx.current_collect.aux_examples.size();
    if (n_aux > 0) {
      gate2_diag.mean_student_kl = sum_student_kl / n_aux;
      gate2_diag.mean_target_kl = sum_target_kl / n_aux;
      gate2_diag.mean_delta_p_target = sum_dp_target / n_aux;
      gate2_diag.mean_delta_p_student = sum_dp_student / n_aux;
      gate2_diag.student_to_target_ratio = (gate2_diag.mean_delta_p_target > 1e-8)
          ? (gate2_diag.mean_delta_p_student / gate2_diag.mean_delta_p_target) : 0.0;
    }
    gate2_diag.aux_distinct_visited = step.stats.aux_distinct_visited;
    gate2_diag.aux_total_presentations = step.stats.aux_total_presentations;
    gate2_diag.visited_aux_count = visited_aux_count;
    gate2_diag.unvisited_aux_count = unvisited_aux_count;
    gate2_diag.sum_delta_p_target = sum_dp_target;
    gate2_diag.sum_delta_p_student = sum_dp_student;
    gate2_diag.visited_sum_delta_p_target = vis_dp_target;
    gate2_diag.visited_sum_delta_p_student = vis_dp_student;
    gate2_diag.unvisited_sum_delta_p_target = unvis_dp_target;
    gate2_diag.unvisited_sum_delta_p_student = unvis_dp_student;
    gate2_diag.visited_student_to_target_ratio =
        (vis_dp_target > 1e-8) ? (vis_dp_student / vis_dp_target) : 0.0;
    gate2_diag.unvisited_student_to_target_ratio =
        (unvis_dp_target > 1e-8) ? (unvis_dp_student / unvis_dp_target) : 0.0;

    double sum_non_override_kl = 0.0;
    for (size_t j = 0; j < gate2_non_override_indices.size(); ++j) {
      const auto& p_old = gate2_non_override_pi_old[j];
      const auto& p_new = gate2_non_override_pi_new[j];
      sum_non_override_kl += kl_div(p_old, p_new);
    }
    if (!gate2_non_override_indices.empty()) {
      gate2_diag.mean_non_override_kl = sum_non_override_kl / gate2_non_override_indices.size();
    }
    for (int c = 0; c < 6; ++c) {
      gate2_diag.searched_by_cat[c] = ctx.current_collect.r755_searched_by_cat[c];
      gate2_diag.executed_overrides_by_cat[c] = ctx.current_collect.r755_executed_overrides_by_cat[c];
    }
    gate2_diag.cat3_aux_distinct_visited_fraction =
        ctx.current_collect.aux_examples.empty()
            ? 0.0
            : static_cast<double>(step.stats.aux_distinct_visited) /
                  static_cast<double>(ctx.current_collect.aux_examples.size());
    gate2_diag.imperium_buy_rate =
        ctx.current_collect.r755_reveal_turns > 0
            ? static_cast<double>(ctx.current_collect.r755_imperium_buys) /
                  static_cast<double>(ctx.current_collect.r755_reveal_turns)
            : 0.0;
    gate2_diag.tleilaxu_buy_rate =
        ctx.current_collect.r755_reveal_turns > 0
            ? static_cast<double>(ctx.current_collect.r755_tleilaxu_buys) /
                  static_cast<double>(ctx.current_collect.r755_reveal_turns)
            : 0.0;
    gate2_diag.tech_buy_rate =
        ctx.current_collect.r755_reveal_turns > 0
            ? static_cast<double>(ctx.current_collect.r755_tech_buys) /
                  static_cast<double>(ctx.current_collect.r755_reveal_turns)
            : 0.0;
    gate2_diag.mean_unspent_persuasion =
        ctx.current_collect.r755_reveal_turns > 0
            ? ctx.current_collect.r755_unspent_persuasion_sum /
                  static_cast<double>(ctx.current_collect.r755_reveal_turns)
            : 0.0;
    gate2_diag.purchase_grad_cosine = step.stats.purchase_grad_cosine;

    gate2_diag.cat3_i_avg.assign(9, 0.0);
    gate2_diag.cat3_ii_avg.assign(9, 0.0);
    gate2_diag.cat3_iii_avg.assign(9, 0.0);
    gate2_diag.cat3_bucket_counts.assign(9, 0);
    for (int b = 0; b < 9; ++b) {
      int64_t count = ctx.current_collect.r755_cat3_bucket_count[b];
      gate2_diag.cat3_bucket_counts[b] = count;
      if (count > 0) {
        gate2_diag.cat3_i_avg[b] = ctx.current_collect.r755_cat3_i_sum[b] / static_cast<double>(count);
        gate2_diag.cat3_ii_avg[b] = ctx.current_collect.r755_cat3_ii_sum[b] / static_cast<double>(count);
        gate2_diag.cat3_iii_avg[b] = ctx.current_collect.r755_cat3_iii_sum[b] / static_cast<double>(count);
      }
    }
    g_gate2_diagnostics.push_back(gate2_diag);
    WriteR755Gate2Json(absl::GetFlag(FLAGS_r755_gate2_output), g_gate2_diagnostics);
    std::cout << absl::StrFormat(
        "[R755-LIVE-CHECK] update=%d nan_inf_count=%lld mask_identity_holds=%d "
        "override_masked=%lld cat1_exec=%lld cat2_exec=%lld cat3_exec=%lld "
        "cat4_exec=%lld cat5_exec=%lld student_to_target_ratio=%.6f "
        "visited_ratio=%.6f unvisited_ratio=%.6f aux_vis=%lld/%zu aux_pres=%lld\n",
        update, static_cast<long long>(gate2_diag.nan_inf_count),
        gate2_diag.mask_identity_holds ? 1 : 0,
        static_cast<long long>(gate2_diag.override_masked_transitions),
        static_cast<long long>(gate2_diag.executed_overrides_by_cat[1]),
        static_cast<long long>(gate2_diag.executed_overrides_by_cat[2]),
        static_cast<long long>(gate2_diag.executed_overrides_by_cat[3]),
        static_cast<long long>(gate2_diag.executed_overrides_by_cat[4]),
        static_cast<long long>(gate2_diag.executed_overrides_by_cat[5]),
        gate2_diag.student_to_target_ratio,
        gate2_diag.visited_student_to_target_ratio,
        gate2_diag.unvisited_student_to_target_ratio,
        static_cast<long long>(gate2_diag.aux_distinct_visited),
        ctx.current_collect.aux_examples.size(),
        static_cast<long long>(gate2_diag.aux_total_presentations)) << std::flush;
    std::cout << absl::StrFormat(
        "[R755-CAT3-DIAG] update=%d buy_rates=[imp=%.4f tle=%.4f tch=%.4f] "
        "unspent_persuasion=%.4f grad_cos=%.4f aux_vis_frac=%.4f\n",
        update, gate2_diag.imperium_buy_rate, gate2_diag.tleilaxu_buy_rate,
        gate2_diag.tech_buy_rate, gate2_diag.mean_unspent_persuasion,
        gate2_diag.purchase_grad_cosine,
        gate2_diag.cat3_aux_distinct_visited_fraction);
    std::cout << "[R755-CAT3-DIAG] (iii) G-(V+A) by bucket: ";
    for (int b = 0; b < 9; ++b) {
      std::cout << absl::StrFormat("B%d[n=%lld]=%.4f ", b,
          static_cast<long long>(gate2_diag.cat3_bucket_counts[b]),
          gate2_diag.cat3_iii_avg[b]);
    }
    std::cout << "\n" << std::flush;
  }

  // Phase 18B clean abort at the latest valid checkpoint (plan §18C).
  if (step.stats.aux_ratio_abort) {
    if (step.have_bg && step.bg_collect_thread.joinable()) step.bg_collect_thread.join();
    std::string abort_diagnostics_path = absl::GetFlag(FLAGS_diagnostics_path);
    if (!abort_diagnostics_path.empty()) {
      open_spiel::WriteDiagnostics(
          abort_diagnostics_path, update, step.stats,
          ctx.current_collect.conflict_vp_generated,
          ctx.current_collect.conflict_vp_attributed,
          ctx.current_collect.conflict_vp_unattributed, ctx.master, ctx.run_uuid,
          absl::GetFlag(FLAGS_run_prefix), ctx.config_fingerprint,
          ctx.current_collect.raw_conflict_vp, ctx.current_collect.raw_noncombat_vp,
          ctx.current_collect.raw_total_vp,
          ctx.search_buffer.ComputeValidationKL(
              ctx.training_model, ctx.device,
              static_cast<float>(absl::GetFlag(FLAGS_logit_cap))),
          absl::GetFlag(FLAGS_emit_canary_columns));
    }
    std::cerr << absl::StrFormat(
        "ABORT[18B grad-ratio]: update %d aux/PPO norm ratio %.4f > %.4f "
        "(aux_norm=%.4f ppo_norm=%.4f). Keeping last valid checkpoint; this "
        "update is NOT saved.\n",
        update, step.stats.aux_ppo_norm_ratio, ctx.aux_abort_ratio,
        step.stats.aux_grad_norm_mean, step.stats.ppo_grad_norm_mean);
    std::exit(3);
  }

  ctx.AccountAuxConsumed(ctx.current_collect);

  if (ctx.ema_model != nullptr) {
    UpdateEmaWeights(ctx.training_model, ctx.ema_model, ctx.ema_decay);
    ctx.ema_updates_absorbed++;
  }

  double hs_pre_share = (ctx.current_collect.nontrivial_transitions > 0)
      ? (static_cast<double>(ctx.current_collect.high_stakes_pre_transitions) /
         ctx.current_collect.nontrivial_transitions)
      : 0.0;
  double hs_post_share = (ctx.current_collect.nontrivial_transitions > 0)
      ? (static_cast<double>(ctx.current_collect.high_stakes_post_transitions) /
         ctx.current_collect.nontrivial_transitions)
      : 0.0;
  std::cout << absl::StrFormat(
      "[HIGH-STAKES-SHARE] update %d: nontrivial=%ld | pre_action (fixed)=%.4f (%ld) | post_action (buggy)=%.4f (%ld)\n",
      update, ctx.current_collect.nontrivial_transitions,
      hs_pre_share, ctx.current_collect.high_stakes_pre_transitions,
      hs_post_share, ctx.current_collect.high_stakes_post_transitions);

  {
    int64_t n_total = 0;
    double sum_err = 0.0;
    double sum_sq_err = 0.0;
    int64_t n_thirds[3] = {0, 0, 0};
    double sum_err_thirds[3] = {0.0, 0.0, 0.0};

    for (const auto& t : ctx.current_collect.rollout) {
      if (t.trajectory_third >= 0 && t.trajectory_third < 3) {
        double v_diff = static_cast<double>(t.value) - static_cast<double>(t.mc_return);
        sum_err += v_diff;
        sum_sq_err += v_diff * v_diff;
        n_total++;
        n_thirds[t.trajectory_third]++;
        sum_err_thirds[t.trajectory_third] += v_diff;
      }
    }

    double mean_err = n_total > 0 ? (sum_err / n_total) : 0.0;
    double rmse = n_total > 0 ? std::sqrt(sum_sq_err / n_total) : 0.0;
    double mean_first = n_thirds[0] > 0 ? (sum_err_thirds[0] / n_thirds[0]) : 0.0;
    double mean_mid = n_thirds[1] > 0 ? (sum_err_thirds[1] / n_thirds[1]) : 0.0;
    double mean_last = n_thirds[2] > 0 ? (sum_err_thirds[2] / n_thirds[2]) : 0.0;

    std::cout << absl::StrFormat(
        "[VALUE-CALIBRATION] update %d: n=%ld, mean(V-G_MC)=%.4f, rmse=%.4f | first_third=%.4f (n=%ld), mid_third=%.4f (n=%ld), last_third=%.4f (n=%ld)\n",
        update, n_total, mean_err, rmse, mean_first, n_thirds[0], mean_mid, n_thirds[1], mean_last, n_thirds[2]);
  }
}

#endif  // OPEN_SPIEL_BUILD_WITH_LIBTORCH

}  // namespace open_spiel::ppo_train_internal
