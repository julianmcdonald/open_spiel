#include "dune_ppo_train_run_context.h"

namespace open_spiel::ppo_train_internal {

#ifdef OPEN_SPIEL_BUILD_WITH_LIBTORCH

namespace {

void RunDecoupledSearchAuxPhase(PpoTrainRunContext& ctx, PpoTrainStepState& step) {
  const int update = step.update;
  if (!(absl::GetFlag(FLAGS_search_in_training) &&
        absl::GetFlag(FLAGS_search_aux_decoupled) &&
        !ctx.current_collect.search_snapshots.empty())) {
    return;
  }

  for (int lag : {1, 10, 25}) {
    int target_u = update - lag;
    for (const auto& rec : ctx.kept_labels_history_25u) {
      if (rec.update_id == target_u && !rec.examples.empty()) {
        auto m = open_spiel::EvaluateSearchExamplesMetrics(
            ctx.training_model, rec.examples, ctx.obs_size, ctx.action_size, ctx.device);
        double ce_before = rec.ce_before;
        double ce_after = rec.ce_after;
        double ce_now = m.mean_ce;
        double p_search = m.mean_p_search_move;
        double denom = ce_before - ce_after;
        double retention = (std::abs(denom) > 1e-6) ? ((ce_before - ce_now) / denom) : 1.0;
        std::cout << absl::StrFormat(
            "[SEARCH-RETENTION] update %d: lag %du (target update %d, N=%zu): P(search)=%.4f, CE=%.4f (CE_before=%.4f, CE_after=%.4f), retention=%.2f%%\n",
            update, lag, target_u, rec.examples.size(), p_search, ce_now, ce_before, ce_after, retention * 100.0);
        break;
      }
    }
  }

  // 1. Freeze post-PPO updated learner and synchronize to search teacher models (Correction 1)
  for (size_t e = 0; e < ctx.search_teacher_models.size(); ++e) {
    CopyModelWeights(ctx.training_model, ctx.search_teacher_models[e]);
  }

  // 2. Generate search targets using post-PPO snapshot and its no-search continuation (Correction 1 & 4)
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
      absl::GetFlag(FLAGS_search_target_max_kl),
      step.pairing_mode,
      step.truncation_mode,
      absl::GetFlag(FLAGS_search_target_mode),
      absl::GetFlag(FLAGS_search_override_eta));

  std::cout << absl::StrFormat(
      "[SEARCH-SUPERVISION] update %d: states_searched=%d, override_labels_kept=%d (%.2f%%), mean_target_prob_search_move=%.4f (elapsed=%.2fs)\n",
      update, step.search_sup_res.total_decisions_evaluated,
      static_cast<int>(step.search_sup_res.examples.size()),
      step.search_sup_res.total_decisions_evaluated > 0
          ? (100.0 * step.search_sup_res.examples.size() / step.search_sup_res.total_decisions_evaluated)
          : 0.0,
      step.search_sup_res.mean_target_prob_search_move,
      step.search_sup_res.elapsed_seconds);

  if (absl::GetFlag(FLAGS_search_research_diagnostic) && !step.search_sup_res.examples.empty()) {
    std::vector<const open_spiel::SearchTrainingExample*> diag_cands;
    for (const auto& ex : step.search_sup_res.examples) {
      if (ex.state_snapshot && ex.raw_action != open_spiel::kInvalidAction && ex.best_action != open_spiel::kInvalidAction) {
        diag_cands.push_back(&ex);
      }
    }
    if (!diag_cands.empty()) {
      std::mt19937_64 diag_rng(dune_seed::DeriveSeed(ctx.master, dune_seed::kDomainTrain, update, 0x52455352ULL));
      std::shuffle(diag_cands.begin(), diag_cands.end(), diag_rng);
      size_t n_diag = std::min<size_t>(16, diag_cands.size());
      int original_still_best_count = 0;
      int still_beats_margin_count = 0;
      double sum_advantage_best_over_raw = 0.0;
      for (size_t di = 0; di < n_diag; ++di) {
        const auto* ex = diag_cands[di];
        uint64_t sample_seed = diag_rng();
        auto res_res = open_spiel::dune_imperium::CompoundRolloutSearchDecision(
            *ex->state_snapshot,
            ex->player,
            /*turn_tree=*/nullptr,
            ctx.search_teacher_evaluators,
            absl::GetFlag(FLAGS_search_top_k),
            absl::GetFlag(FLAGS_search_rollouts_per_action),
            absl::GetFlag(FLAGS_search_override_margin),
            open_spiel::dune_imperium::TreeReuseMode::kTargetFill,
            sample_seed,
            step.pairing_mode,
            step.truncation_mode);

        double raw_u = res_res.raw_mean_utility;
        double best_u = raw_u;
        for (const auto& cu : res_res.candidate_utilities) {
          if (cu.first == ex->best_action) {
            best_u = cu.second;
            break;
          }
        }
        double adv = best_u - raw_u;
        sum_advantage_best_over_raw += adv;
        if (res_res.action == ex->best_action) {
          original_still_best_count++;
        }
        if (adv >= absl::GetFlag(FLAGS_search_override_margin) - 1e-6) {
          still_beats_margin_count++;
        }
      }
      double share_still_best = 100.0 * original_still_best_count / n_diag;
      double share_still_margin = 100.0 * still_beats_margin_count / n_diag;
      double mean_adv = sum_advantage_best_over_raw / n_diag;
      std::cout << absl::StrFormat(
          "[SEARCH-RESEARCH-DIAG] update %d: evaluated %zu override labels | original best still best=%.1f%%, still beats raw by >= margin=%.1f%%, mean re-searched advantage=%.4f\n",
          update, n_diag, share_still_best, share_still_margin, mean_adv);
    }
  }

  // 3. Subsample reference rollout transitions for rollout KL ceiling and critic probe (Correction 2 & 3)
  std::vector<open_spiel::PpoTransition> ref_rollout_sample;
  const size_t sample_size = std::min<size_t>(256, ctx.current_collect.rollout.size());
  if (sample_size > 0) {
    ref_rollout_sample.reserve(sample_size);
    size_t stride = std::max<size_t>(1, ctx.current_collect.rollout.size() / sample_size);
    for (size_t i = 0; i < ctx.current_collect.rollout.size() && ref_rollout_sample.size() < sample_size; i += stride) {
      ref_rollout_sample.push_back(ctx.current_collect.rollout[i]);
    }
  }

  // Deliberately reset auxiliary optimizer state every phase
  if (ctx.search_aux_optimizer == nullptr) {
    std::vector<torch::Tensor> aux_trainable_params;
    for (auto& p : ctx.training_model->parameters()) {
      if (p.requires_grad()) {
        aux_trainable_params.push_back(p);
      }
    }
    std::vector<torch::optim::OptimizerParamGroup> groups;
    groups.emplace_back(aux_trainable_params);
    ctx.search_aux_optimizer = std::make_unique<torch::optim::AdamW>(
        groups, torch::optim::AdamWOptions(absl::GetFlag(FLAGS_learning_rate)).eps(1e-5));
    static_cast<torch::optim::AdamWOptions&>(
        ctx.search_aux_optimizer->param_groups()[0].options())
        .weight_decay(0.0);
  } else {
    ctx.search_aux_optimizer->state().clear();
  }

  // 4. Run bounded auxiliary optimization phase with isolated aux optimizer
  const uint64_t aux_phase_seed = dune_seed::DeriveSeed(
      ctx.master, dune_seed::kDomainTrain, update, /*stream=*/0x41555850ULL);
  int aux_steps = absl::GetFlag(FLAGS_search_aux_steps);
  if (absl::GetFlag(FLAGS_search_aux_epochs) > 1 && aux_steps == 4) {
    aux_steps = 0;
  }
  if (!step.search_sup_res.examples.empty()) {
    auto fresh_m = open_spiel::EvaluateSearchExamplesMetrics(
        ctx.training_model, step.search_sup_res.examples, ctx.obs_size, ctx.action_size, ctx.device);
    std::cout << absl::StrFormat(
        "[SEARCH-FRESH-CHECK] update %d: fresh kept-labels N=%zu, mean P(search)=%.4f, greedy agreement=%.2f%%, CE=%.4f\n",
        update, step.search_sup_res.examples.size(),
        fresh_m.mean_p_search_move, fresh_m.greedy_agreement_pct, fresh_m.mean_ce);
  }

  open_spiel::SearchAuxAnchoringConfig anchoring_config;
  anchoring_config.enabled = absl::GetFlag(FLAGS_search_aux_anchoring);
  anchoring_config.beta = absl::GetFlag(FLAGS_search_aux_beta);
  anchoring_config.alpha = absl::GetFlag(FLAGS_search_aux_alpha);
  anchoring_config.rail_kl_target = absl::GetFlag(FLAGS_search_aux_rail_kl_target);
  anchoring_config.logit_cap = absl::GetFlag(FLAGS_logit_cap);

  open_spiel::SearchAuxPhaseResult aux_res = open_spiel::TrainSearchAuxiliaryPhase(
      ctx.training_model, *ctx.search_aux_optimizer,
      step.search_sup_res.examples,
      ctx.obs_size, ctx.action_size, ctx.device,
      aux_steps,
      absl::GetFlag(FLAGS_search_aux_batch_size),
      absl::GetFlag(FLAGS_search_aux_target_kl),
      step.this_search_coef,
      ref_rollout_sample,
      aux_phase_seed,
      absl::GetFlag(FLAGS_search_aux_epochs),
      step.search_sup_res.agreement_examples,
      ctx.current_collect.rollout,
      anchoring_config);

  // 5. Update stats and log telemetry
  step.stats.aux_examples_used = static_cast<int>(step.search_sup_res.examples.size());
  step.stats.aux_search_loss_coef = step.this_search_coef;
  step.stats.aux_ce = aux_res.final_aux_ce;
  step.stats.search_aux_loss = step.stats.aux_ce;
  step.stats.aux_grad_norm_mean = aux_res.mean_aux_grad_norm;

  double this_aux_gain = aux_res.initial_aux_ce - aux_res.final_aux_ce;
  std::cout << absl::StrFormat(
      "[SEARCH-AUX] update %d: %d/%d steps accepted (rejected_on_kl=%d, rejected_nonfinite=%d, phase_rejected=%d, null_skipped=%d) "
      "| ce_before=%.4f ce_after=%.4f (aux_gain=%.4f) "
      "| kl_searched=%.6f kl_rollout=%.6f (ceil=%.4f) | val_immut=%d | critic_shift_mse=%.6f\n",
      update, aux_res.steps_accepted, aux_res.steps_budgeted,
      aux_res.step_rejected_on_kl ? 1 : 0,
      aux_res.step_rejected_nonfinite ? 1 : 0,
      aux_res.phase_rejected ? 1 : 0,
      aux_res.null_supervision_skipped ? 1 : 0,
      aux_res.initial_aux_ce, aux_res.final_aux_ce, this_aux_gain,
      aux_res.cumulative_kl_searched, aux_res.cumulative_kl_rollout,
      absl::GetFlag(FLAGS_search_aux_target_kl),
      aux_res.value_head_immutable ? 1 : 0, aux_res.critic_probe_mse_shift);

  if (anchoring_config.enabled) {
    std::cout << absl::StrFormat(
        "[SEARCH-AUX-ANCHORED] update %d: %d/%d steps accepted, %d override presentations "
        "| override CE %.4f -> %.4f (gain=%.4f), override KL=%.6f "
        "| rail KL=%.6f (agr=%.6f, non_hs=%.6f, target=%.4f) "
        "| delta entropy: rail=%.6f, hs=%.6f "
        "| critic shift MSE: anchors=%.6f, rail=%.6f "
        "| loss terms: CE=%.4f, beta*KL=%.4f, alpha*MSE=%.4f\n",
        update, aux_res.steps_accepted, aux_res.steps_budgeted,
        aux_res.accepted_override_presentations,
        aux_res.initial_aux_ce, aux_res.final_aux_ce, this_aux_gain,
        aux_res.override_kl,
        aux_res.rail_kl_total, aux_res.rail_kl_agreement, aux_res.rail_kl_non_high_stakes,
        anchoring_config.rail_kl_target,
        aux_res.delta_entropy_rail, aux_res.delta_entropy_high_stakes,
        aux_res.critic_probe_mse_shift_anchors, aux_res.critic_probe_mse_shift_rail,
        aux_res.mean_override_ce_loss, aux_res.mean_anchor_kl_loss, aux_res.mean_anchor_value_mse_loss);
  }

  if (!step.search_sup_res.examples.empty()) {
    KeptLabelsRecord rec;
    rec.update_id = update;
    rec.examples = step.search_sup_res.examples;
    rec.ce_before = aux_res.initial_aux_ce;
    rec.ce_after = aux_res.final_aux_ce;
    ctx.kept_labels_history_25u.push_back(std::move(rec));
    while (ctx.kept_labels_history_25u.size() > 25) {
      ctx.kept_labels_history_25u.pop_front();
    }
  }

  if (aux_res.step_rejected_nonfinite) {
    std::cerr << absl::StrFormat(
        "ABORT[SEARCH-AUX-NONFINITE]: update %d search auxiliary phase encountered nonfinite loss, gradient, or parameter values!\n",
        update);
    SpielFatalError(absl::StrFormat("Search auxiliary phase nonfinite invariant violation at update %d", update));
  }
}

void EvalAbsorptionLagProbes(
    const PpoTrainRunContext& ctx,
    const R765AbsorptionUpdateRecord& hist_rec,
    int64_t* out_roots,
    double* out_mean_dlo) {
  std::unordered_set<uint64_t> roots;
  std::vector<const std::vector<float>*> h_obs;
  std::vector<const dune_semantic::CandidateActionData*> h_cads;
  std::vector<const std::vector<Action>*> h_legals;
  for (const auto& pr : hist_rec.probes) {
    roots.insert(pr.root_uid);
    h_obs.push_back(&pr.observation);
    h_cads.push_back(&pr.candidate_data);
    h_legals.push_back(&pr.legal_actions);
  }
  *out_roots = static_cast<int64_t>(roots.size());
  auto h_probs = EvalModelLegalProbs(ctx, ctx.training_model, h_obs, h_cads, h_legals);
  double sum_dlo = 0.0;
  size_t valid_n = 0;
  for (size_t i = 0; i < hist_rec.probes.size(); ++i) {
    const auto& pr = hist_rec.probes[i];
    int s_idx = -1, r_idx = -1;
    for (size_t a = 0; a < pr.legal_actions.size(); ++a) {
      if (pr.legal_actions[a] == pr.a_star) s_idx = static_cast<int>(a);
      if (pr.legal_actions[a] == pr.a_ref) r_idx = static_cast<int>(a);
    }
    if (s_idx >= 0 && r_idx >= 0 && i < h_probs.size() &&
        s_idx < static_cast<int>(h_probs[i].size()) &&
        r_idx < static_cast<int>(h_probs[i].size())) {
      double cur_lo = std::log(std::max(1e-12, h_probs[i][s_idx])) -
                      std::log(std::max(1e-12, h_probs[i][r_idx]));
      sum_dlo += (cur_lo - pr.t0_log_odds);
      valid_n++;
    }
  }
  if (valid_n > 0) {
    *out_mean_dlo = sum_dlo / valid_n;
  }
}

void RunR765AbsorptionDiag(PpoTrainRunContext& ctx, PpoTrainStepState& step) {
  const int update = step.update;
  if (absl::GetFlag(FLAGS_r765_absorption_diag_jsonl).empty()) {
    return;
  }

  const size_t n_ov = ctx.current_collect.r765_override_probes.size();
  std::vector<const std::vector<float>*> cur_obs_ptrs;
  std::vector<const dune_semantic::CandidateActionData*> cur_cads;
  std::vector<const std::vector<Action>*> cur_legals;
  cur_obs_ptrs.reserve(n_ov);
  cur_cads.reserve(n_ov);
  cur_legals.reserve(n_ov);
  for (const auto& probe : ctx.current_collect.r765_override_probes) {
    cur_obs_ptrs.push_back(&probe.observation);
    cur_cads.push_back(&probe.candidate_data);
    cur_legals.push_back(&probe.legal_actions);
  }

  auto pi_t2_live = EvalModelLegalProbs(ctx, ctx.training_model, cur_obs_ptrs, cur_cads, cur_legals);
  auto pi_t2_ema = (ctx.ema_model != nullptr)
      ? EvalModelLegalProbs(ctx, ctx.ema_model, cur_obs_ptrs, cur_cads, cur_legals)
      : std::vector<std::vector<double>>{};

  std::unordered_set<uint64_t> distinct_roots;
  std::map<std::string, int64_t> cat_counts = {
      {"card", 0}, {"same_card_space", 0}, {"cross_card_space", 0},
      {"graft", 0}, {"cat3_purchase", 0}};
  double sum_delta_q = 0.0;
  double sum_realized_kl = 0.0;
  double sum_realized_temp = 0.0;
  int64_t kl_cap_bound_count = 0;
  int64_t target_flips_argmax_count = 0;
  double sum_delta_log_odds_target = 0.0;
  double sum_delta_log_odds_t2_live = 0.0;
  double sum_delta_log_odds_t2_ema = 0.0;

  for (size_t i = 0; i < n_ov; ++i) {
    auto& probe = ctx.current_collect.r765_override_probes[i];
    distinct_roots.insert(probe.root_uid);
    cat_counts[probe.category]++;
    sum_delta_q += probe.delta_q;
    sum_realized_kl += probe.realized_kl;
    sum_realized_temp += probe.realized_temp;
    if (probe.kl_cap_bound) kl_cap_bound_count++;
    if (probe.target_flips_argmax) target_flips_argmax_count++;
    sum_delta_log_odds_target += (probe.target_log_odds - probe.t0_log_odds);

    int s_idx = -1, r_idx = -1;
    for (size_t a = 0; a < probe.legal_actions.size(); ++a) {
      if (probe.legal_actions[a] == probe.a_star) s_idx = static_cast<int>(a);
      if (probe.legal_actions[a] == probe.a_ref) r_idx = static_cast<int>(a);
    }
    if (s_idx >= 0 && r_idx >= 0 && i < pi_t2_live.size() &&
        s_idx < static_cast<int>(pi_t2_live[i].size()) &&
        r_idx < static_cast<int>(pi_t2_live[i].size())) {
      double p_star = pi_t2_live[i][s_idx];
      double p_ref = pi_t2_live[i][r_idx];
      probe.t2_live_log_odds = std::log(std::max(1e-12, p_star)) - std::log(std::max(1e-12, p_ref));
      sum_delta_log_odds_t2_live += (probe.t2_live_log_odds - probe.t0_log_odds);
    }
    if (s_idx >= 0 && r_idx >= 0 && i < pi_t2_ema.size() &&
        s_idx < static_cast<int>(pi_t2_ema[i].size()) &&
        r_idx < static_cast<int>(pi_t2_ema[i].size())) {
      double p_star = pi_t2_ema[i][s_idx];
      double p_ref = pi_t2_ema[i][r_idx];
      probe.t2_ema_log_odds = std::log(std::max(1e-12, p_star)) - std::log(std::max(1e-12, p_ref));
      sum_delta_log_odds_t2_ema += (probe.t2_ema_log_odds - probe.t0_log_odds);
    }
  }

  double mean_delta_q = n_ov > 0 ? (sum_delta_q / n_ov) : 0.0;
  double mean_realized_kl = n_ov > 0 ? (sum_realized_kl / n_ov) : 0.0;
  double mean_realized_temp = n_ov > 0 ? (sum_realized_temp / n_ov) : 0.0;
  double kl_cap_bound_frac = n_ov > 0 ? (static_cast<double>(kl_cap_bound_count) / n_ov) : 0.0;
  double target_flips_argmax_frac = n_ov > 0 ? (static_cast<double>(target_flips_argmax_count) / n_ov) : 0.0;
  double mean_delta_log_odds_target = n_ov > 0 ? (sum_delta_log_odds_target / n_ov) : 0.0;
  double mean_delta_log_odds_t1 = step.stats.aux_t1_measured_rows > 0
      ? (step.stats.aux_t1_delta_log_odds_sum / step.stats.aux_t1_measured_rows) : 0.0;
  double mean_delta_log_odds_t2_live = n_ov > 0 ? (sum_delta_log_odds_t2_live / n_ov) : 0.0;
  double mean_delta_log_odds_t2_ema = n_ov > 0 ? (sum_delta_log_odds_t2_ema / n_ov) : 0.0;
  double ratio_t1 = (mean_delta_log_odds_target > 1e-8)
      ? (mean_delta_log_odds_t1 / mean_delta_log_odds_target) : 0.0;
  double ratio_t2_live = (mean_delta_log_odds_target > 1e-8)
      ? (mean_delta_log_odds_t2_live / mean_delta_log_odds_target) : 0.0;

  std::vector<open_spiel::R765DiagProbeItem> fresh_diag_items;
  fresh_diag_items.reserve(ctx.current_collect.r765_override_probes.size());
  for (const auto& pr : ctx.current_collect.r765_override_probes) {
    open_spiel::R765DiagProbeItem item;
    item.root_uid = pr.root_uid;
    item.observation = pr.observation;
    item.legal_actions = pr.legal_actions;
    item.candidate_data = pr.candidate_data;
    item.a_star = pr.a_star;
    item.a_ref = pr.a_ref;
    item.t0_log_odds = pr.t0_log_odds;
    fresh_diag_items.push_back(std::move(item));
  }
  auto post_evict = ctx.r765_replay_mgr.FinalizeUpdateAndEvaluateRetention(
      update, std::max(1, absl::GetFlag(FLAGS_r765_replay_window_updates)),
      fresh_diag_items, step.stats.aux_visited_mask,
      [&](const auto& o, const auto& c, const auto& l) {
        return EvalModelLegalProbs(ctx, ctx.training_model, o, c, l);
      });

  int64_t agreed_preserved_count = 0;
  int64_t agreed_eligible_count = static_cast<int64_t>(ctx.current_collect.r765_agreed_probes.size());
  double agreed_greedy_preserve_rate = (agreed_eligible_count == 0) ? -1.0 : 1.0;
  if (agreed_eligible_count > 0) {
    std::vector<const std::vector<float>*> ag_obs_ptrs;
    std::vector<const dune_semantic::CandidateActionData*> ag_cads;
    std::vector<const std::vector<Action>*> ag_legals;
    ag_obs_ptrs.reserve(ctx.current_collect.r765_agreed_probes.size());
    ag_cads.reserve(ctx.current_collect.r765_agreed_probes.size());
    ag_legals.reserve(ctx.current_collect.r765_agreed_probes.size());
    for (const auto& ap : ctx.current_collect.r765_agreed_probes) {
      ag_obs_ptrs.push_back(&ap.observation);
      ag_cads.push_back(&ap.candidate_data);
      ag_legals.push_back(&ap.legal_actions);
    }
    auto ag_probs = EvalModelLegalProbs(ctx, ctx.training_model, ag_obs_ptrs, ag_cads, ag_legals);
    for (size_t i = 0; i < ctx.current_collect.r765_agreed_probes.size(); ++i) {
      const auto& ap = ctx.current_collect.r765_agreed_probes[i];
      if (i < ag_probs.size() && !ag_probs[i].empty()) {
        size_t best_idx = std::max_element(ag_probs[i].begin(), ag_probs[i].end()) - ag_probs[i].begin();
        if (best_idx < ap.legal_actions.size() && ap.legal_actions[best_idx] == ap.greedy_action) {
          agreed_preserved_count++;
        }
      }
    }
    agreed_greedy_preserve_rate = static_cast<double>(agreed_preserved_count) /
                                  static_cast<double>(agreed_eligible_count);
  }

  bool panel1024_evaluated = false;
  double panel1024_agree_all = -1.0;
  double panel1024_agree_hp65 = -1.0;
  if (!ctx.r765_collateral_panel.empty()) {
    const int check_interval = std::max(1, absl::GetFlag(FLAGS_r765_collateral_check_interval));
    if ((update - ctx.start_update + 1) % check_interval == 0 || update == ctx.target_end_update) {
      panel1024_evaluated = true;
      std::vector<const std::vector<float>*> p_obs;
      std::vector<const dune_semantic::CandidateActionData*> p_cads;
      std::vector<const std::vector<Action>*> p_legals;
      p_obs.reserve(ctx.r765_collateral_panel.size());
      p_cads.reserve(ctx.r765_collateral_panel.size());
      p_legals.reserve(ctx.r765_collateral_panel.size());
      for (const auto& item : ctx.r765_collateral_panel) {
        p_obs.push_back(&item.obs);
        p_cads.push_back(item.candidate_data.empty() ? nullptr : &item.candidate_data);
        p_legals.push_back(&item.legal_actions);
      }
      auto p_probs = EvalModelLegalProbs(ctx, ctx.training_model, p_obs, p_cads, p_legals);
      int64_t agree_all_cnt = 0;
      int64_t agree_hp65_cnt = 0;
      int64_t hp65_tot = 0;
      for (size_t i = 0; i < ctx.r765_collateral_panel.size(); ++i) {
        const auto& item = ctx.r765_collateral_panel[i];
        if (item.is_hp65) hp65_tot++;
        if (i < p_probs.size() && !p_probs[i].empty()) {
          const auto& probs = p_probs[i];
          double best_p = -1e9;
          Action best_a = open_spiel::kInvalidAction;
          for (size_t a_idx = 0; a_idx < item.legal_actions.size(); ++a_idx) {
            if (a_idx >= probs.size()) continue;
            double pf = probs[a_idx];
            Action ai = item.legal_actions[a_idx];
            if (pf > best_p + 1e-12 || (std::abs(pf - best_p) <= 1e-12 && (best_a < 0 || ai < best_a))) {
              best_p = pf;
              best_a = ai;
            }
          }
          if (best_a == item.baseline_greedy_action) {
            agree_all_cnt++;
            if (item.is_hp65) agree_hp65_cnt++;
          }
        }
      }
      panel1024_agree_all = ctx.r765_collateral_panel.empty()
          ? 1.0 : (static_cast<double>(agree_all_cnt) / ctx.r765_collateral_panel.size());
      panel1024_agree_hp65 = hp65_tot == 0
          ? 1.0 : (static_cast<double>(agree_hp65_cnt) / static_cast<double>(hp65_tot));
    }
  }

  bool all_metrics_finite =
      std::isfinite(step.stats.policy_loss) &&
      std::isfinite(step.stats.value_loss) &&
      std::isfinite(step.stats.entropy) &&
      std::isfinite(step.stats.approx_kl) &&
      std::isfinite(mean_delta_q) &&
      std::isfinite(mean_realized_kl) &&
      std::isfinite(mean_realized_temp) &&
      std::isfinite(mean_delta_log_odds_target) &&
      std::isfinite(mean_delta_log_odds_t1) &&
      std::isfinite(mean_delta_log_odds_t2_live) &&
      std::isfinite(mean_delta_log_odds_t2_ema) &&
      std::isfinite(post_evict.post_evict_5u_ret) &&
      std::isfinite(post_evict.post_evict_10u_ret);
  if (panel1024_evaluated) {
    all_metrics_finite = all_metrics_finite &&
        std::isfinite(panel1024_agree_all) &&
        std::isfinite(panel1024_agree_hp65);
  }

  double pooled_5u_preserve_rate = -1.0;
  int64_t pooled_5u_preserved = 0;
  int64_t pooled_5u_eligible = 0;
  std::string preserve_abort_reason;
  bool preserve_ok = ctx.r765_preserve_guard.PushAndCheck(
      agreed_preserved_count, agreed_eligible_count,
      absl::GetFlag(FLAGS_r765_preserve_5u_min_rate),
      all_metrics_finite, &pooled_5u_preserve_rate,
      &pooled_5u_preserved, &pooled_5u_eligible, &preserve_abort_reason);

  bool coverage_ok = true;
  if (absl::GetFlag(FLAGS_r765_replay_window_updates) > 1 &&
      absl::GetFlag(FLAGS_r755_finish_unvisited_aux_on_kl_stop) &&
      post_evict.has_final_age_cohort_this_update &&
      !post_evict.final_age_full_coverage) {
    coverage_ok = false;
  }

  bool panel_ok = true;
  if (panel1024_evaluated) {
    double min_all = absl::GetFlag(FLAGS_r765_collateral_min_all_agree);
    double min_hp65 = absl::GetFlag(FLAGS_r765_collateral_min_hp65_agree);
    if (min_all > 0.0 && panel1024_agree_all < min_all) panel_ok = false;
    if (min_hp65 > 0.0 && panel1024_agree_hp65 < min_hp65) panel_ok = false;
  }

  int64_t t3_5u_roots = 0;
  double mean_delta_log_odds_t3_5u = 0.0;
  int64_t t3_10u_roots = 0;
  double mean_delta_log_odds_t3_10u = 0.0;
  double ret_10u = 0.0;

  for (const auto& hist_rec : ctx.r765_absorption_history) {
    if (hist_rec.update == update - 5 && !hist_rec.probes.empty()) {
      EvalAbsorptionLagProbes(ctx, hist_rec, &t3_5u_roots, &mean_delta_log_odds_t3_5u);
    }
    if (hist_rec.update == update - 10 && !hist_rec.probes.empty()) {
      EvalAbsorptionLagProbes(ctx, hist_rec, &t3_10u_roots, &mean_delta_log_odds_t3_10u);
      if (hist_rec.mean_t2_live_delta > 1e-8) {
        ret_10u = mean_delta_log_odds_t3_10u / hist_rec.mean_t2_live_delta;
      }
    }
  }

  R765AbsorptionUpdateRecord cur_rec;
  cur_rec.update = update;
  cur_rec.probes = ctx.current_collect.r765_override_probes;
  cur_rec.mean_t2_live_delta = mean_delta_log_odds_t2_live;
  ctx.r765_absorption_history.push_back(std::move(cur_rec));
  while (ctx.r765_absorption_history.size() > 10) {
    ctx.r765_absorption_history.pop_front();
  }

  std::string diag_jsonl_path = absl::GetFlag(FLAGS_r765_absorption_diag_jsonl);
  bool jsonl_write_ok = true;
  if (!diag_jsonl_path.empty()) {
    std::ofstream out(diag_jsonl_path, std::ios::app);
    if (out.is_open()) {
      const int replay_k = absl::GetFlag(FLAGS_r765_replay_window_updates);
      const int n_ages = std::min(16, std::max(1, replay_k));
      const double cap_frac = step.stats.aux_total_slices > 0
          ? (static_cast<double>(step.stats.aux_cap_active_slices) /
             static_cast<double>(step.stats.aux_total_slices))
          : 0.0;
      out << "{"
          << "\"global_update\":" << update
          << ",\"distinct_override_roots\":" << distinct_roots.size()
          << ",\"emitted_aux_examples\":" << ctx.current_collect.aux_examples.size()
          << ",\"total_aux_rows_in_update\":" << step.train_aux_examples.size()
          << ",\"active_replay_cohorts\":" << post_evict.active_train_cohorts
          << ",\"aux_cap_active_slices\":" << step.stats.aux_cap_active_slices
          << ",\"aux_total_slices\":" << step.stats.aux_total_slices
          << ",\"aux_cap_active_frac\":" << cap_frac
          << ",\"aux_presentations_by_age\":[";
      for (int a = 0; a < n_ages; ++a) {
        if (a > 0) out << ",";
        out << step.stats.aux_presentations_by_age[a];
      }
      out << "],\"aux_mean_eff_weight_by_age\":[";
      for (int a = 0; a < n_ages; ++a) {
        if (a > 0) out << ",";
        double mean_w = step.stats.aux_presentations_by_age[a] > 0
            ? (step.stats.aux_eff_weight_sum_by_age[a] /
               static_cast<double>(step.stats.aux_presentations_by_age[a]))
            : 0.0;
        out << mean_w;
      }
      out << "],\"final_age_cohort_matured\":" << (post_evict.has_final_age_cohort_this_update ? 1 : 0)
          << ",\"final_age_full_coverage\":" << (post_evict.final_age_full_coverage ? 1 : 0)
          << ",\"post_evict_5u_roots\":" << post_evict.post_evict_5u_roots
          << ",\"post_evict_5u_d_final_denom\":" << post_evict.post_evict_5u_d_final_denom
          << ",\"post_evict_5u_d_cur\":" << post_evict.post_evict_5u_d_cur << ",\"post_evict_5u_ret\":" << post_evict.post_evict_5u_ret
          << ",\"post_evict_10u_roots\":" << post_evict.post_evict_10u_roots
          << ",\"post_evict_10u_d_final_denom\":" << post_evict.post_evict_10u_d_final_denom
          << ",\"post_evict_10u_d_cur\":" << post_evict.post_evict_10u_d_cur << ",\"post_evict_10u_ret\":" << post_evict.post_evict_10u_ret
          << ",\"agreed_preserved_count\":" << agreed_preserved_count << ",\"agreed_eligible_count\":" << agreed_eligible_count
          << ",\"pooled_5u_preserved\":" << pooled_5u_preserved << ",\"pooled_5u_eligible\":" << pooled_5u_eligible
          << ",\"pooled_5u_preserve_rate\":" << pooled_5u_preserve_rate
          << ",\"panel1024_evaluated\":" << (panel1024_evaluated ? 1 : 0)
          << ",\"panel1024_agree_all\":" << panel1024_agree_all << ",\"panel1024_agree_hp65\":" << panel1024_agree_hp65
          << ",\"category_counts\":{\"card\":" << cat_counts["card"]
          << ",\"same_card_space\":" << cat_counts["same_card_space"]
          << ",\"cross_card_space\":" << cat_counts["cross_card_space"]
          << ",\"graft\":" << cat_counts["graft"]
          << ",\"cat3_purchase\":" << cat_counts["cat3_purchase"] << "}"
          << ",\"mean_delta_q\":" << mean_delta_q << ",\"mean_realized_kl\":" << mean_realized_kl
          << ",\"mean_realized_temp\":" << mean_realized_temp << ",\"kl_cap_bound_frac\":" << kl_cap_bound_frac
          << ",\"target_flips_argmax_frac\":" << target_flips_argmax_frac
          << ",\"mean_delta_log_odds_target\":" << mean_delta_log_odds_target
          << ",\"mean_delta_log_odds_t1\":" << mean_delta_log_odds_t1
          << ",\"mean_delta_log_odds_t2_live\":" << mean_delta_log_odds_t2_live
          << ",\"mean_delta_log_odds_t2_ema\":" << mean_delta_log_odds_t2_ema
          << ",\"ratio_t1\":" << ratio_t1 << ",\"ratio_t2_live\":" << ratio_t2_live
          << ",\"t3_5u_roots\":" << t3_5u_roots << ",\"mean_delta_log_odds_t3_5u\":" << mean_delta_log_odds_t3_5u
          << ",\"t3_10u_roots\":" << t3_10u_roots << ",\"mean_delta_log_odds_t3_10u\":" << mean_delta_log_odds_t3_10u
          << ",\"ret_10u\":" << ret_10u << ",\"agreed_greedy_preserve_rate\":" << agreed_greedy_preserve_rate
          << ",\"v5_dynamic_grad_cosine\":" << step.stats.purchase_grad_cosine
          << ",\"action_head_grad_cosine\":" << step.stats.action_head_grad_cosine << "}\n";
      out.flush();
      jsonl_write_ok = out.good();
    } else {
      jsonl_write_ok = false;
    }
  }

  if (!preserve_ok || !all_metrics_finite || !jsonl_write_ok || !coverage_ok || !panel_ok) {
    if (step.have_bg && step.bg_collect_thread.joinable()) step.bg_collect_thread.join();
    std::string abort_msg = "ABORT[R765-GUARD]:";
    if (!jsonl_write_ok) abort_msg += " jsonl_write_failed";
    if (!all_metrics_finite) abort_msg += " nonfinite_telemetry";
    if (!preserve_ok) abort_msg += " " + preserve_abort_reason;
    if (!coverage_ok) abort_msg += " final_age_coverage_failed";
    if (!panel_ok) {
      abort_msg += absl::StrFormat(" collateral_panel_floor_breached (all=%.4f < %.4f or hp65=%.4f < %.4f)",
                                  panel1024_agree_all, absl::GetFlag(FLAGS_r765_collateral_min_all_agree),
                                  panel1024_agree_hp65, absl::GetFlag(FLAGS_r765_collateral_min_hp65_agree));
    }
    std::cerr << abort_msg << "\n";
    std::cout << abort_msg << "\n";
    std::exit(4);
  }
}

}  // namespace

void RunLoopAuxPhase(PpoTrainRunContext& ctx, PpoTrainStepState& step) {
  const int update = step.update;

  // Decoupled Standalone Auxiliary Search Supervision Phase (PPG-style)
  RunDecoupledSearchAuxPhase(ctx, step);

  if (absl::GetFlag(FLAGS_train_value_only)) {
    std::string current_non_value_hash = HashNonValueParameters(ctx.training_model);
    if (current_non_value_hash != ctx.initial_non_value_hash) {
      SpielFatalError(absl::StrFormat("Policy/trunk parameters changed during value-only training! Expected: %s, Got: %s",
                                      ctx.initial_non_value_hash, current_non_value_hash));
    }
  }

  RunR765AbsorptionDiag(ctx, step);

  step.ppo_elapsed = std::chrono::duration<double>(
      std::chrono::high_resolution_clock::now() - step.ppo_start).count();

  // Log PPO updates to diagnostics CSV
  std::string diagnostics_path = absl::GetFlag(FLAGS_diagnostics_path);
  if (!diagnostics_path.empty()) {
    double val_kl = ctx.search_buffer.ComputeValidationKL(
        ctx.training_model, ctx.device, static_cast<float>(absl::GetFlag(FLAGS_logit_cap)));
    open_spiel::WriteDiagnostics(diagnostics_path, update, step.stats,
                                 ctx.current_collect.conflict_vp_generated,
                                 ctx.current_collect.conflict_vp_attributed,
                                 ctx.current_collect.conflict_vp_unattributed,
                                 ctx.master, ctx.run_uuid, absl::GetFlag(FLAGS_run_prefix),
                                 ctx.config_fingerprint,
                                 ctx.current_collect.raw_conflict_vp,
                                 ctx.current_collect.raw_noncombat_vp,
                                 ctx.current_collect.raw_total_vp,
                                 val_kl,
                                 absl::GetFlag(FLAGS_emit_canary_columns));
  }

  // WO-PERF-TIMING
  std::string sidecar_timing_path = absl::GetFlag(FLAGS_phase_timing_path);
  if (sidecar_timing_path.empty()) {
    if (!diagnostics_path.empty()) {
      sidecar_timing_path = open_spiel::PhaseTimingPath(diagnostics_path);
    } else if (absl::GetFlag(FLAGS_phase_timing_mode) == "phases") {
      std::string run_pfx = absl::GetFlag(FLAGS_run_prefix);
      if (!run_pfx.empty()) {
        std::filesystem::path p(run_pfx);
        sidecar_timing_path = (p.parent_path() / "phase_timing.jsonl").string();
      }
    }
  }
  if (!sidecar_timing_path.empty()) {
    open_spiel::WritePhaseTimingToPath(sidecar_timing_path, update, step.stats,
                                       step.ppo_elapsed, ctx.run_uuid,
                                       absl::GetFlag(FLAGS_run_prefix));
  }

  ctx.WriteVerificationSidecarIfEnabled(update, ctx.current_collect, step.stats);

  // --- Search auxiliary distillation steps ---
  step.search_kl_sum = 0.0;
  step.search_grad_sum = 0.0;
  step.search_lambda = absl::GetFlag(FLAGS_search_lambda);
  step.search_minibatches_per_update = absl::GetFlag(FLAGS_search_minibatches_per_update);
  int search_minibatch_size = absl::GetFlag(FLAGS_search_minibatch_size);
  float logit_cap = static_cast<float>(absl::GetFlag(FLAGS_logit_cap));

  // A ROLE-AWARE dataset is CLOSED at load time; LoadNewFiles is skipped when role_aware().
  if (!ctx.search_label_dir.empty() && !ctx.search_buffer.role_aware()) {
    ctx.search_buffer.LoadNewFiles(ctx.search_label_dir);
  }

  if (step.search_lambda > 0.0 && ctx.search_buffer.Size() > 0) {
    auto cpu_float = torch::TensorOptions().dtype(torch::kFloat32);
    auto cpu_bool = torch::TensorOptions().dtype(torch::kBool);

    for (int aux = 0; aux < step.search_minibatches_per_update; ++aux) {
      uint64_t aux_seed = dune_seed::DeriveSeed(
          ctx.master, dune_seed::kDomainTrain, update, aux, dune_seed::kStreamSearchSampling);
      std::mt19937 aux_rng = dune_seed::MakeRng32(aux_seed);
      std::vector<SearchLabel> search_batch = ctx.search_buffer.Sample(search_minibatch_size, &aux_rng);
      if (search_batch.empty()) continue;

      int64_t sb_size = search_batch.size();
      torch::Tensor search_states_cpu = torch::empty({sb_size, ctx.obs_size}, cpu_float);
      torch::Tensor search_masks_cpu = torch::zeros({sb_size, ctx.action_size}, cpu_bool);
      torch::Tensor search_teacher_probs_cpu = torch::zeros({sb_size, ctx.action_size}, cpu_float);

      float* states_ptr = search_states_cpu.data_ptr<float>();
      bool* masks_ptr = search_masks_cpu.data_ptr<bool>();
      float* teacher_ptr = search_teacher_probs_cpu.data_ptr<float>();

      for (int64_t i = 0; i < sb_size; ++i) {
        const auto& label = search_batch[i];
        std::memcpy(states_ptr + i * ctx.obs_size, label.state.data(), ctx.obs_size * sizeof(float));
        for (const auto& ap : label.teacher_probs) {
          int64_t action_id = ap.first;
          float prob = ap.second;
          if (action_id >= 0 && action_id < ctx.action_size) {
            masks_ptr[i * ctx.action_size + action_id] = true;
            teacher_ptr[i * ctx.action_size + action_id] = prob;
          }
        }
      }

      torch::Tensor search_states = search_states_cpu.to(ctx.device);
      torch::Tensor search_masks = search_masks_cpu.to(ctx.device);
      torch::Tensor search_teacher_probs = search_teacher_probs_cpu.to(ctx.device);

      ctx.optimizer->zero_grad();

      torch::Tensor mean_kl;
      auto compute_distill_loss = [&]() {
        SPIEL_CHECK_FALSE(ctx.training_model->with_semantic_scorer_);
        auto outputs = ctx.training_model->forward(search_states);
        torch::Tensor logits = CenterAndCapLogitsTensor(outputs.logits, search_masks, logit_cap);
        torch::Tensor masked_logits = logits.masked_fill(search_masks.logical_not(), -1e9f);
        torch::Tensor log_probs = torch::log_softmax(masked_logits, -1);

        torch::Tensor log_teacher = torch::log(search_teacher_probs.clamp_min(1e-12f));
        torch::Tensor kl_loss = search_teacher_probs * (log_teacher - log_probs);
        mean_kl = kl_loss.sum(-1).mean();
      };

      if (ctx.device.is_cuda() && absl::GetFlag(FLAGS_train_amp)) {
        AutocastGuard autocast_guard(c10::DeviceType::CUDA, true);
        compute_distill_loss();
      } else {
        compute_distill_loss();
      }

      torch::Tensor search_loss = step.search_lambda * mean_kl;
      search_loss.backward();

      double search_grad_norm = torch::nn::utils::clip_grad_norm_(
          ctx.training_model->parameters(), std::numeric_limits<double>::infinity());
      if (!std::isfinite(search_grad_norm) || !std::isfinite(mean_kl.item<double>())) {
        std::cerr << "Warning: Non-finite search distillation gradient (" << search_grad_norm
                  << ") or loss. Skipping optimizer step.\n";
        ctx.optimizer->zero_grad();
        continue;
      }
      torch::nn::utils::clip_grad_norm_(
          ctx.training_model->parameters(), absl::GetFlag(FLAGS_grad_clip_norm));
      ctx.optimizer->step();

      step.search_kl_sum += mean_kl.item<double>();
      step.search_grad_sum += search_grad_norm;
    }
  }
}

#endif  // OPEN_SPIEL_BUILD_WITH_LIBTORCH

}  // namespace open_spiel::ppo_train_internal
