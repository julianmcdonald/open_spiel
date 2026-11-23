#include "dune_ppo_train_sim_internal.h"

namespace open_spiel::ppo_train_internal {

#ifdef OPEN_SPIEL_BUILD_WITH_LIBTORCH

namespace {

int FindActionIndex(const std::vector<Action>& actions, Action target) {
  for (size_t i = 0; i < actions.size(); ++i) {
    if (actions[i] == target) return static_cast<int>(i);
  }
  return -1;
}

Action FindGreedyAction(const std::vector<Action>& actions,
                        const std::vector<double>& legal_probabilities) {
  Action best = actions[0];
  double max_p = -1.0;
  for (size_t i = 0; i < actions.size(); ++i) {
    double p = (i < legal_probabilities.size()) ? legal_probabilities[i] : 0.0;
    if (p > max_p) {
      max_p = p;
      best = actions[i];
    }
  }
  return best;
}

open_spiel::ActionsAndProbs BuildRawPolicyPairs(
    const std::vector<Action>& actions,
    const std::vector<double>& legal_probabilities) {
  open_spiel::ActionsAndProbs pairs;
  pairs.reserve(actions.size());
  for (size_t i = 0; i < actions.size(); ++i) {
    double p = (i < legal_probabilities.size()) ? legal_probabilities[i] : 0.0;
    pairs.push_back({actions[i], p});
  }
  return pairs;
}

bool ApplyOverrideAction(const std::vector<Action>& actions,
                         const std::vector<double>& legal_probabilities,
                         Action target_action, SimStepDecision* step) {
  int idx = FindActionIndex(actions, target_action);
  if (idx < 0) return false;
  step->action = target_action;
  if (idx < static_cast<int>(legal_probabilities.size())) {
    step->old_log_prob = static_cast<float>(
        std::log(std::max(1e-12, legal_probabilities[idx])));
  }
  step->r755_override_accepted = true;
  return true;
}

open_spiel::dune_imperium::SearchRolloutCrnMode ParseCrnMode(
    const std::string& crn_str, const char* flag_name) {
  if (crn_str == "sequential") {
    return open_spiel::dune_imperium::SearchRolloutCrnMode::kSequential;
  }
  if (crn_str == "deck_order") {
    return open_spiel::dune_imperium::SearchRolloutCrnMode::kDeckOrderCrn;
  }
  if (crn_str == "event_keyed") {
    return open_spiel::dune_imperium::SearchRolloutCrnMode::kEventKeyedCrn;
  }
  open_spiel::SpielFatalError(absl::StrCat("Unknown ", flag_name, ": ", crn_str));
  return open_spiel::dune_imperium::SearchRolloutCrnMode::kSequential;
}

open_spiel::dune_imperium::SearchSnapshot BuildSearchSnapshot(
    const PpoSimContext& ctx, const SimStepDecision& step) {
  open_spiel::dune_imperium::SearchSnapshot snap;
  snap.sim_state = ctx.state->Clone();
  snap.player = step.current_player;
  snap.observation = step.obs;
  snap.legal_actions = step.actions;
  snap.chosen_action = step.action;
  snap.behavior_log_prob = step.old_log_prob;
  snap.root_id = (ctx.episode_id << 16) |
                 static_cast<uint64_t>(ctx.trajectory ? ctx.trajectory->size() : 0);
  snap.candidate_data = step.cand_data;
  snap.raw_policy = BuildRawPolicyPairs(step.actions, step.legal_probabilities);
  return snap;
}

void EvaluateModelAndCanaries(PpoSimContext* ctx, SimStepDecision* step) {
  const Player current_player = step->current_player;
  const std::vector<Action>& actions = step->actions;

  std::shared_ptr<IGameEvaluator> active_evaluator =
      dune_opponent_pool::ResolveEvaluator(current_player, ctx->seat_assignment,
                                           ctx->evaluator, *ctx->opponent_evaluators);

  dune_imperium::MarketAppendixMode eval_market_mode =
      active_evaluator ? active_evaluator->MarketMode()
                       : dune_imperium::ParseMarketAppendixMode(absl::GetFlag(FLAGS_market_appendix_mode));
  dune_imperium::DeckPoolSemantics eval_deck_semantics =
      active_evaluator ? active_evaluator->DeckPoolSemantics()
                       : dune_imperium::ParseDeckPoolSemanticsOrDie(absl::GetFlag(FLAGS_deck_pool_semantics));

  const int active_obs_size =
      dune_imperium::GetInformationStateSchema(eval_market_mode, eval_deck_semantics).size;
  step->obs.assign(active_obs_size, 0.0f);
  if (ctx->dune_state != nullptr) {
    ctx->dune_state->InformationStateTensorWithAppendix(
        current_player, eval_market_mode, absl::MakeSpan(step->obs), eval_deck_semantics);
  } else if (ctx->provides_info_state_tensor && current_player >= 0) {
    ctx->state->InformationStateTensor(current_player, absl::MakeSpan(step->obs));
  } else if (ctx->provides_observations_tensor && current_player >= 0) {
    ctx->state->ObservationTensor(current_player, absl::MakeSpan(step->obs));
  }
  step->p_cand_data = nullptr;
  if (ctx->dune_state != nullptr) {
    const std::string cand_schema =
        active_evaluator ? active_evaluator->SemanticDescriptorSchema()
                         : absl::GetFlag(FLAGS_semantic_descriptor_schema);
    dune_semantic::ExtractCandidateDescriptors(*ctx->dune_state, actions, &step->cand_data, cand_schema);
    step->p_cand_data = &step->cand_data;
  }
  step->eval_result = active_evaluator->EvaluateWithActions(step->obs, step->p_cand_data);
  std::vector<float>& logits = step->eval_result.logits;

  step->parity_capture =
      (!absl::GetFlag(FLAGS_numerical_parity_output).empty()) && step->is_learner;
  if (step->parity_capture) {
    const int64_t decision_index = static_cast<int64_t>(ctx->trajectory->size());
    const std::string identity = absl::StrFormat(
        "episode=%llu decision=%lld",
        static_cast<unsigned long long>(ctx->episode_id),
        static_cast<long long>(decision_index));
    ++ctx->local_stats->parity_decisions;
    const std::vector<int64_t> legal_ids(actions.begin(), actions.end());
    std::string validation_error;
    const PpoParityPrecapCaptureValidation validation =
        ValidateAndCapturePpoParityPrecap(
            logits, ctx->game->NumDistinctActions(), legal_ids,
            &step->parity_raw_legal_logits, &validation_error);
    if (validation.full_width_ok) {
      ++ctx->local_stats->parity_full_width_ok;
    } else {
      ctx->local_stats->parity_capture_errors.push_back(absl::StrFormat(
          "%s: evaluator width %d != action_dim %d", identity,
          static_cast<int>(logits.size()), ctx->game->NumDistinctActions()));
    }
    if (validation.full_finite_ok) {
      ++ctx->local_stats->parity_full_finite_ok;
    } else {
      ctx->local_stats->parity_capture_errors.push_back(
          identity + ": evaluator emitted a nonfinite full logit");
    }
    if (validation.legal_ids_unique_in_range) {
      ++ctx->local_stats->parity_legal_ids_unique_in_range;
    } else {
      ctx->local_stats->parity_capture_errors.push_back(
          identity + ": legal action IDs are duplicated or out of range");
    }
  }

  // --- Phase A canary: pre-cap legal-centered |z| for this decision ---
  // Measured HERE because CenterAndCapLegalLogits below mutates `logits` in
  // place, destroying the raw values. Observational only: no RNG draw, no
  // tensor op, nothing read back into the acting path.
  if (ctx->local_stats != nullptr && actions.size() >= 2 && step->is_learner) {
    double legal_sum = 0.0;
    int legal_count = 0;
    for (Action a : actions) {
      if (a >= 0 && static_cast<size_t>(a) < logits.size()) {
        legal_sum += logits[a];
        ++legal_count;
      }
    }
    if (legal_count > 0) {
      const float legal_mean = static_cast<float>(legal_sum / legal_count);
      float max_absz = 0.0f;
      for (Action a : actions) {
        if (a < 0 || static_cast<size_t>(a) >= logits.size()) continue;
        max_absz = std::max(max_absz, std::abs(logits[a] - legal_mean));
      }
      ctx->local_stats->precap_absz_all.push_back(max_absz);
      const DuneDecisionRole decision_role = ClassifyDuneDecisionRole(
          *ctx->state, current_player, /*has_active_session=*/false);
      switch (decision_role) {
        case DuneDecisionRole::kAgentPrimary:
          ctx->local_stats->precap_absz_primary.push_back(max_absz);
          break;
        case DuneDecisionRole::kAgentContinuation:
          ctx->local_stats->precap_absz_continuation.push_back(max_absz);
          break;
        case DuneDecisionRole::kPurchase:
          ctx->local_stats->precap_absz_purchase.push_back(max_absz);
          break;
        default:
          break;
      }

      // --- PWO-5 section 14.1b: SAT, the PRE-cap half. --------------------
      // Must be PRE-cap against the configured cap (`c * tanh(z / c)` keeps
      // post-cap |y| < c strictly in exact arithmetic).
      if (ctx->emit_canary_columns) {
        for (Action a : actions) {
          if (a < 0 || static_cast<size_t>(a) >= logits.size()) continue;
          ++ctx->local_stats->canary_sat_logits;
          if (std::abs(logits[a] - legal_mean) >= ctx->logit_cap) {
            ++ctx->local_stats->canary_sat_ge_cap;
          }
        }
        ctx->canary_role_this_decision = static_cast<int>(decision_role);
        ctx->canary_measure_this_decision = true;
      }
    }
  }

  CenterAndCapLegalLogits(logits, actions, ctx->logit_cap);

  // Numerical-parity gate retains the opt-in behavior categorical; normal
  // training stores only the sampled action's scalar log-probability.
  step->parity_decision_role = static_cast<int>(ClassifyDuneDecisionRole(
      *ctx->state, current_player, /*has_active_session=*/false));
  if (step->parity_capture) {
    float max_logit = -std::numeric_limits<float>::infinity();
    for (Action a : actions) {
      if (a >= 0 && static_cast<size_t>(a) < logits.size()) {
        max_logit = std::max(max_logit, logits[a]);
      }
    }
    std::vector<double> weights(actions.size(), 0.0);
    double total_weight = 0.0;
    for (size_t i = 0; i < actions.size(); ++i) {
      const Action a = actions[i];
      if (a >= 0 && static_cast<size_t>(a) < logits.size() && std::isfinite(max_logit)) {
        weights[i] = std::exp(static_cast<double>(logits[a] - max_logit));
        if (!std::isfinite(weights[i]) || weights[i] < 0.0) weights[i] = 0.0;
      }
      total_weight += weights[i];
    }
    if (!(total_weight > 0.0) || !std::isfinite(total_weight)) {
      weights.assign(actions.size(), 1.0);
      total_weight = static_cast<double>(actions.size());
    }
    step->parity_behavior_log_probs.reserve(actions.size());
    for (double weight : weights) {
      step->parity_behavior_log_probs.push_back(
          weight > 0.0
              ? static_cast<float>(std::log(weight / total_weight))
              : -std::numeric_limits<float>::infinity());
    }
  }

  // --- PWO-5 section 14.1b: NE and max_action_prob, the POST-cap half. --
  if (ctx->canary_measure_this_decision && step->is_learner) {
    float max_logit = -std::numeric_limits<float>::infinity();
    for (Action a : actions) {
      if (a >= 0 && static_cast<size_t>(a) < logits.size()) {
        max_logit = std::max(max_logit, logits[a]);
      }
    }
    double total = 0.0;
    std::vector<double> probs;
    probs.reserve(actions.size());
    for (Action a : actions) {
      double w = 0.0;
      if (a >= 0 && static_cast<size_t>(a) < logits.size() && std::isfinite(max_logit)) {
        w = std::exp(static_cast<double>(logits[a]) - static_cast<double>(max_logit));
      }
      probs.push_back(w);
      total += w;
    }
    if (total > 0.0) {
      double h = 0.0;
      double max_p = 0.0;
      for (double& w : probs) {
        w /= total;
        if (w > 0.0) h -= w * std::log(w);
        max_p = std::max(max_p, w);
      }
      const double ne = h / std::log(static_cast<double>(actions.size()));
      ctx->local_stats->canary_ne_sum += ne;
      ++ctx->local_stats->canary_ne_n;
      ctx->local_stats->canary_ne_sum_role[ctx->canary_role_this_decision] += ne;
      ++ctx->local_stats->canary_ne_n_role[ctx->canary_role_this_decision];
      ctx->local_stats->canary_maxprob_sum += max_p;
    }
    ctx->canary_measure_this_decision = false;
  }
}

void SampleAndExploreAction(PpoSimContext* ctx, SimStepDecision* step) {
  const Player current_player = step->current_player;
  const std::vector<Action>& actions = step->actions;

  const PolicyDistributionSample policy_sample =
      SampleTrainingPolicyDistribution(*ctx->state, &ctx->policy_rng[current_player],
                                       step->eval_result.logits, actions,
                                       &step->legal_probabilities);
  step->action = policy_sample.action;
  step->old_log_prob = policy_sample.chosen_log_probability;
  step->leader_draft_overridden = policy_sample.leader_draft_overridden;

  step->is_forced_buy = false;
  if (ctx->do_purchase_exploration &&
      current_player == ctx->purchase_exploration_seat &&
      ctx->dune_state != nullptr &&
      ctx->dune_state->IsPlayerInRevealTurn(current_player)) {
    int cur_round = ctx->dune_state->GetCurrentRound();
    bool round_eligible =
        (cur_round == ctx->forced_round_1 && !ctx->forced_round_1_done) ||
        (cur_round == ctx->forced_round_2 && !ctx->forced_round_2_done);
    if (round_eligible) {
      std::vector<Action> legal_acquisitions;
      for (Action a : actions) {
        if (open_spiel::IsAcquisitionAction(a)) legal_acquisitions.push_back(a);
      }
      if (legal_acquisitions.size() >= 2) {
        std::uniform_int_distribution<size_t> buy_dist(0, legal_acquisitions.size() - 1);
        step->action = legal_acquisitions[buy_dist(ctx->explore_rng)];
        step->is_forced_buy = true;
        if (cur_round == ctx->forced_round_1) ctx->forced_round_1_done = true;
        else if (cur_round == ctx->forced_round_2) ctx->forced_round_2_done = true;

        int act_i = FindActionIndex(actions, step->action);
        if (act_i >= 0 && static_cast<size_t>(act_i) < step->legal_probabilities.size()) {
          step->old_log_prob = static_cast<float>(
              std::log(std::max(1e-12, step->legal_probabilities[act_i])));
        }
        if (ctx->local_stats != nullptr) ++ctx->local_stats->forced_buys;
      }
    }
  }

  step->targeted_buy_card_id = 0;
  step->is_targeted_forced_buy = false;
  if (ctx->dune_state != nullptr) {
    int cur_round = ctx->dune_state->GetCurrentRound();
    Action buy_act_11 = open_spiel::kInvalidAction;
    int slot_11 = -1;
    bool legal_11 = open_spiel::dune_targeted_buy::FindLegalTleilaxuBuyAction(
        *ctx->dune_state, actions,
        open_spiel::dune_targeted_buy::kScientificBreakthroughTleilaxuId,
        &slot_11, &buy_act_11);

    Action buy_act_13 = open_spiel::kInvalidAction;
    int slot_13 = -1;
    bool legal_13 = open_spiel::dune_targeted_buy::FindLegalTleilaxuBuyAction(
        *ctx->dune_state, actions,
        open_spiel::dune_targeted_buy::kStitchedHorrorTleilaxuId,
        &slot_13, &buy_act_13);

    auto forcing_decision = open_spiel::dune_targeted_buy::DecideTargetedForcedBuy(
        open_spiel::dune_targeted_buy::g_targeted_buy_tracker,
        *ctx->dune_state, current_player, ctx->designated_explore_seat,
        cur_round, actions, step->legal_probabilities,
        ctx->episode_forced_card11, ctx->episode_forced_card13,
        ctx->policy_rng[current_player]);

    if (forcing_decision.forced) {
      step->action = forcing_decision.forced_action;
      step->is_forced_buy = true;
      step->is_targeted_forced_buy = true;
      step->targeted_buy_card_id = forcing_decision.forced_card_tleilaxu_id;
      step->old_log_prob = forcing_decision.old_log_prob;
    }

    if (ctx->local_stats != nullptr && current_player == ctx->designated_explore_seat) {
      int r_idx = std::clamp(cur_round, 1, 10);
      auto record_unforced_opp = [&](int c_idx, int card_id, Action buy_act) {
        int act_i = FindActionIndex(actions, buy_act);
        double prob = (act_i >= 0 && static_cast<size_t>(act_i) < step->legal_probabilities.size())
                          ? step->legal_probabilities[act_i]
                          : 0.0;
        ctx->local_stats->targeted_buy_telemetry.unforced_opps[c_idx][r_idx]++;
        ctx->local_stats->targeted_buy_telemetry.unforced_sum_prob[c_idx][r_idx] += prob;
        if (step->action == buy_act) {
          ctx->local_stats->targeted_buy_telemetry.unforced_buys[c_idx][r_idx]++;
          if (!step->is_targeted_forced_buy) step->targeted_buy_card_id = card_id;
        }
      };
      if (legal_11 && (!step->is_targeted_forced_buy ||
                       step->targeted_buy_card_id !=
                           open_spiel::dune_targeted_buy::kScientificBreakthroughTleilaxuId)) {
        record_unforced_opp(0, open_spiel::dune_targeted_buy::kScientificBreakthroughTleilaxuId,
                            buy_act_11);
      }
      if (legal_13 && (!step->is_targeted_forced_buy ||
                       step->targeted_buy_card_id !=
                           open_spiel::dune_targeted_buy::kStitchedHorrorTleilaxuId)) {
        record_unforced_opp(1, open_spiel::dune_targeted_buy::kStitchedHorrorTleilaxuId,
                            buy_act_13);
      }
    }
  }
}

void RunR755HybridSearch(PpoSimContext* ctx, SimStepDecision* step) {
  const Player current_player = step->current_player;
  const std::vector<Action>& actions = step->actions;
  step->r755_override_accepted = false;

  if (!ctx->r755_hybrid || !step->is_learner || current_player < 0 ||
      current_player >= 4 || ctx->r755_search_evals.empty() ||
      step->is_forced_buy) {
    return;
  }

  const double teacher_win_speed_eps = open_spiel::ResolveOnlineTeacherWinSpeedEps(
      absl::GetFlag(FLAGS_r765_teacher_win_speed_tiebreak_eps),
      absl::GetFlag(FLAGS_win_speed_bonus_per_round));
  const bool teacher_unbiased_s2 = absl::GetFlag(FLAGS_r765_teacher_unbiased_stage2_gate);
  const bool teacher_ceiling_aware_margin =
      absl::GetFlag(FLAGS_r765_teacher_ceiling_aware_margin);

  if (absl::GetFlag(FLAGS_r765_joint_agent_search) &&
      ctx->r765_pending_joint[current_player].active) {
    Action step_raw_greedy_action = FindGreedyAction(actions, step->legal_probabilities);
    int cur_hand = open_spiel::dune_imperium::ComputeAgentTurnHandPlusPlayed(
        ctx->dune_state, current_player);
    int cur_intr = ctx->dune_state
                       ? static_cast<int>(ctx->dune_state->GetPlayerIntrigues(current_player).size())
                       : -1;
    auto follow = open_spiel::dune_imperium::AdvancePendingJointAgentOverride(
        &ctx->r765_pending_joint[current_player], actions, step_raw_greedy_action,
        cur_hand, cur_intr, ctx->r765_saw_chance_since_root[current_player],
        /*min_s2_worlds=*/24, teacher_win_speed_eps, teacher_ceiling_aware_margin);

    if (follow.execute_override) {
      int act_i = FindActionIndex(actions, follow.forced_action);
      step->action = follow.forced_action;
      if (act_i >= 0 && static_cast<size_t>(act_i) < step->legal_probabilities.size()) {
        step->old_log_prob = static_cast<float>(
            std::log(std::max(1e-12, step->legal_probabilities[act_i])));
      }
      step->r755_override_accepted = true;

      if (follow.emit_aux_target && ctx->r755_aux_examples != nullptr) {
        auto raw_policy_pairs = BuildRawPolicyPairs(actions, step->legal_probabilities);
        const double follow_ref_q = follow.challenger_mean_q;
        const double follow_star_q =
            (teacher_unbiased_s2 && follow.shared_s2_worlds >= 24)
                ? (follow.challenger_mean_q + follow.mean_s2_diff)
                : follow.chosen_mean_q;
        const char* probe_cat =
            follow.is_graft_step
                ? "graft"
                : (follow.same_card_as_raw ? "same_card_space" : "cross_card_space");
        EmitR765PairwiseMpoAuxAndProbe(ctx, *step, follow.root_uid, probe_cat,
                                       raw_policy_pairs, follow.challenger_action,
                                       follow_ref_q, follow.forced_action, follow_star_q);
      }
    }
  }

  if (step->r755_override_accepted) return;

  double max_raw_p = 0.0;
  double p_2 = 0.0;
  for (double p : step->legal_probabilities) {
    if (p > max_raw_p) {
      p_2 = max_raw_p;
      max_raw_p = p;
    } else if (p > p_2) {
      p_2 = p;
    }
  }
  auto cat = open_spiel::dune_imperium::ClassifyR755SearchCategory(
      *ctx->state, current_player, ctx->r755_space_placed_this_turn[current_player],
      absl::GetFlag(FLAGS_r755_category_mask),
      absl::GetFlag(FLAGS_r755_cat3_strict_card_tech_only));

  bool coin_active = false;
  if (cat == open_spiel::dune_imperium::R755SearchCategory::kAgentCardOrStartPlot ||
      cat == open_spiel::dune_imperium::R755SearchCategory::kImmediateSpacePlacement) {
    coin_active = ctx->r755_turn_coin_active[current_player];
  } else if (cat != open_spiel::dune_imperium::R755SearchCategory::kIneligible) {
    double search_prob = absl::GetFlag(FLAGS_r755_search_prob);
    if (cat == open_spiel::dune_imperium::R755SearchCategory::kRevealPurchase) {
      double cat3_prob = absl::GetFlag(FLAGS_r755_cat3_search_prob);
      if (cat3_prob >= 0.0) search_prob = cat3_prob;
      double boost_prob = absl::GetFlag(FLAGS_r755_cat3_low_prior_boost_prob);
      if (boost_prob >= 0.0) {
        const double boost_thresh = absl::GetFlag(FLAGS_r755_cat3_low_prior_threshold);
        for (size_t a_idx = 0; a_idx < actions.size(); ++a_idx) {
          if (open_spiel::dune_imperium::IsCardOrTechAcquisitionAction(actions[a_idx])) {
            double p = (a_idx < step->legal_probabilities.size())
                           ? step->legal_probabilities[a_idx]
                           : 0.0;
            if (p < boost_thresh) {
              search_prob = std::max(search_prob, boost_prob);
              break;
            }
          }
        }
      }
    }
    uint64_t coin_seed = dune_seed::DeriveSeed(
        ctx->master, dune_seed::kDomainTrain, ctx->episode_id,
        ctx->r755_decision_index, 0x52373535ULL);
    std::mt19937_64 coin_rng(coin_seed);
    coin_active = (std::uniform_real_distribution<double>(0.0, 1.0)(coin_rng) < search_prob);
  }

  double max_prior_gate = absl::GetFlag(FLAGS_r755_max_raw_prior);
  if ((cat == open_spiel::dune_imperium::R755SearchCategory::kAgentCardOrStartPlot ||
       cat == open_spiel::dune_imperium::R755SearchCategory::kImmediateSpacePlacement) &&
      absl::GetFlag(FLAGS_r765_agent_max_raw_prior) >= 0.0) {
    max_prior_gate = absl::GetFlag(FLAGS_r765_agent_max_raw_prior);
  }
  const bool prior_gated = (max_prior_gate < 1.0 && max_raw_p >= max_prior_gate);
  if (cat == open_spiel::dune_imperium::R755SearchCategory::kIneligible ||
      !coin_active || prior_gated) {
    return;
  }

  int cat_idx = static_cast<int>(cat);
  if (ctx->local_stats != nullptr && cat_idx >= 1 && cat_idx <= 5) {
    ctx->local_stats->r755_searched_by_cat[cat_idx]++;
  }
  uint64_t search_seed = dune_seed::DeriveSeed(
      ctx->master, dune_seed::kDomainTrain, ctx->episode_id,
      ctx->r755_decision_index, 0x52373553ULL);
  open_spiel::dune_imperium::TurnSearchTree local_r755_tree;
  auto crn_mode = ParseCrnMode(absl::GetFlag(FLAGS_r755_crn_mode), "r755_crn_mode");
  open_spiel::ActionsAndProbs raw_policy_pairs =
      BuildRawPolicyPairs(actions, step->legal_probabilities);
  const uint64_t root_uid =
      (ctx->episode_id << 16) | static_cast<uint64_t>(ctx->r755_decision_index);

  if (cat == open_spiel::dune_imperium::R755SearchCategory::kAgentCardOrStartPlot &&
      absl::GetFlag(FLAGS_r765_joint_agent_search)) {
    open_spiel::dune_imperium::SearchV2TuningConfig tuning{};
    tuning.unclocked_c96_3arm_mode = true;
    auto joint_res = open_spiel::dune_imperium::SearchV2JointPlacementDecision(
        *ctx->state, current_player, &local_r755_tree, ctx->r755_search_evals,
        absl::GetFlag(FLAGS_r765_agent_ref_rollouts), /*min_override_margin=*/0.15,
        search_seed, crn_mode, /*max_time_ms=*/-1.0, /*max_inflight_worlds=*/192,
        /*prior_log_ratio_weight=*/0.02, max_prior_gate, &raw_policy_pairs,
        /*est_ms_per_sim=*/3.5, /*max_candidates=*/8, tuning,
        teacher_win_speed_eps, teacher_ceiling_aware_margin);

    if (joint_res.base.overridden && absl::GetFlag(FLAGS_r755_execute_overrides)) {
      if (ctx->local_stats != nullptr) ctx->local_stats->r755_executed_overrides_by_cat[1]++;
      if (ApplyOverrideAction(actions, step->legal_probabilities,
                              joint_res.chosen_first_action, step)) {
        if (joint_res.chosen_first_action != joint_res.raw_first_action &&
            ctx->r755_aux_examples != nullptr) {
          const double joint_ref_q = joint_res.base.raw_mean_utility;
          const double joint_star_q =
              (teacher_unbiased_s2 && joint_res.base.stage2_kept_worlds >= 16)
                  ? (joint_res.base.raw_mean_utility + joint_res.base.mean_d_stage2)
                  : joint_res.base.chosen_mean_utility;
          EmitR765PairwiseMpoAuxAndProbe(ctx, *step, root_uid, "card", raw_policy_pairs,
                                         joint_res.raw_first_action, joint_ref_q,
                                         joint_res.chosen_first_action, joint_star_q);
        }
        if (joint_res.chosen_graft_action != open_spiel::kInvalidAction ||
            joint_res.chosen_second_action != open_spiel::kInvalidAction) {
          auto& pj = ctx->r765_pending_joint[current_player];
          pj.active = true;
          pj.root_uid = root_uid;
          pj.stage1_kept_worlds = joint_res.base.stage1_kept_worlds;
          pj.root_hand_size = open_spiel::dune_imperium::ComputeAgentTurnHandPlusPlayed(
              ctx->dune_state, current_player);
          pj.root_intrigue_count =
              ctx->dune_state
                  ? static_cast<int>(ctx->dune_state->GetPlayerIntrigues(current_player).size())
                  : -1;
          pj.chosen_first_action = joint_res.chosen_first_action;
          pj.executed_graft_action = open_spiel::kInvalidAction;
          pj.expected_graft_action = joint_res.chosen_graft_action;
          pj.expected_second_action = joint_res.chosen_second_action;
          pj.chosen_candidate_index = joint_res.chosen_candidate_index;
          pj.joint_candidates = std::move(joint_res.joint_candidates);
          pj.per_candidate_rollouts = std::move(joint_res.per_candidate_rollouts);
          ctx->r765_saw_chance_since_root[current_player] = false;
        }
      }
    } else if (!joint_res.base.overridden) {
      MaybeRecordR765AgreedProbe(ctx, *step, FindGreedyAction(actions, step->legal_probabilities));
    }
    return;
  }

  if ((cat == open_spiel::dune_imperium::R755SearchCategory::kAgentCardOrStartPlot ||
       cat == open_spiel::dune_imperium::R755SearchCategory::kImmediateSpacePlacement) &&
      absl::GetFlag(FLAGS_r765_seq_v1_agent_search)) {
    Action raw_a = FindGreedyAction(actions, step->legal_probabilities);
    auto v1_res = open_spiel::dune_imperium::SuccessiveHalvingRolloutSearchDecision(
        *ctx->state, current_player, &ctx->r755_turn_trees[current_player],
        ctx->r755_search_evals, absl::GetFlag(FLAGS_r765_agent_ref_rollouts),
        /*min_override_margin=*/0.15, search_seed, crn_mode,
        /*unbiased_stage2_gate=*/teacher_unbiased_s2, /*max_time_ms=*/-1.0,
        /*fixed_top_k=*/5, /*max_inflight_worlds=*/192,
        /*use_bounded_pool_when_unclocked=*/false, /*full_clock_stage2=*/false,
        /*prior_log_ratio_weight=*/0.02, max_prior_gate, &raw_policy_pairs,
        /*all_buy_candidates=*/false, teacher_win_speed_eps, teacher_ceiling_aware_margin);

    if (v1_res.overridden && v1_res.action != raw_a &&
        absl::GetFlag(FLAGS_r755_execute_overrides)) {
      if (ApplyOverrideAction(actions, step->legal_probabilities, v1_res.action, step)) {
        if (ctx->local_stats != nullptr && cat_idx >= 1 && cat_idx <= 5) {
          ctx->local_stats->r755_executed_overrides_by_cat[cat_idx]++;
        }
        double raw_q = v1_res.raw_mean_utility;
        double chosen_q = v1_res.chosen_mean_utility;
        for (const auto& cu : v1_res.candidate_utilities) {
          if (cu.first == raw_a) raw_q = cu.second;
          if (cu.first == v1_res.action) chosen_q = cu.second;
        }
        if (teacher_unbiased_s2 && v1_res.stage2_kept_worlds >= 16) {
          chosen_q = raw_q + v1_res.mean_d_stage2;
        }
        const char* probe_cat =
            (cat == open_spiel::dune_imperium::R755SearchCategory::kAgentCardOrStartPlot)
                ? "card"
                : "same_card_space";
        EmitR765PairwiseMpoAuxAndProbe(ctx, *step, root_uid, probe_cat, raw_policy_pairs,
                                       raw_a, raw_q, v1_res.action, chosen_q);
      }
    } else if (!v1_res.overridden &&
               cat == open_spiel::dune_imperium::R755SearchCategory::kAgentCardOrStartPlot) {
      MaybeRecordR765AgreedProbe(ctx, *step, raw_a);
    }
    return;
  }

  open_spiel::ActionsAndProbs collector_raw_prior;
  open_spiel::dune_imperium::CompoundSearchOptions opts;
  if (absl::GetFlag(FLAGS_r755_reuse_root_prior)) {
    collector_raw_prior = raw_policy_pairs;
    opts.precomputed_raw_prior = &collector_raw_prior;
  }
  opts.dedup_paired_worlds = absl::GetFlag(FLAGS_r755_dedup_paired_worlds);
  opts.win_speed_tiebreak_eps = teacher_win_speed_eps;
  opts.ceiling_aware_margin = teacher_ceiling_aware_margin;

  int top_k = 3;
  int rollouts_per_action = 16;
  auto truncation_mode =
      open_spiel::dune_imperium::SearchTruncationMode::kNextRoundSearchPlayer;

  if (cat == open_spiel::dune_imperium::R755SearchCategory::kRevealPurchase) {
    opts.all_buy_candidates = true;
    opts.symmetric_stage2_confirmation = absl::GetFlag(FLAGS_r755_cat3_symmetric_confirmation);
    opts.stage1_rollouts = absl::GetFlag(FLAGS_r755_cat3_stage1_rollouts);
    opts.stage1_min_mean_diff = absl::GetFlag(FLAGS_r755_stage1_min_mean_diff);
    opts.stage2_independent_rollouts = absl::GetFlag(FLAGS_r755_cat3_stage2_indep_rollouts);
    rollouts_per_action = opts.stage1_rollouts + opts.stage2_independent_rollouts;
    top_k = 12;

    const std::string cat3_trunc_str = absl::GetFlag(FLAGS_r755_cat3_truncation_mode);
    if (cat3_trunc_str == "next_round" || cat3_trunc_str == "next_round_search_player") {
      truncation_mode = open_spiel::dune_imperium::SearchTruncationMode::kNextRoundSearchPlayer;
    } else if (cat3_trunc_str == "plus2_rounds" || cat3_trunc_str == "plus_2_rounds") {
      truncation_mode = open_spiel::dune_imperium::SearchTruncationMode::kPlus2Rounds;
    } else if (cat3_trunc_str == "terminal" || cat3_trunc_str == "none") {
      truncation_mode = open_spiel::dune_imperium::SearchTruncationMode::kTerminal;
    } else {
      open_spiel::SpielFatalError(absl::StrCat("Unknown r755_cat3_truncation_mode: ", cat3_trunc_str));
    }

    const std::string cat3_crn_override = absl::GetFlag(FLAGS_r755_cat3_crn_mode);
    if (!cat3_crn_override.empty()) {
      crn_mode = ParseCrnMode(cat3_crn_override, "r755_cat3_crn_mode");
    }
  } else {
    opts.stage1_rollouts = absl::GetFlag(FLAGS_r755_stage1_rollouts);
    opts.stage1_min_mean_diff = absl::GetFlag(FLAGS_r755_stage1_min_mean_diff);
    opts.stage1_subwave_rollouts = absl::GetFlag(FLAGS_r755_stage1_subwave_rollouts);
    opts.stage1_subwave_min_diff = absl::GetFlag(FLAGS_r755_stage1_subwave_min_diff);
    opts.center_4seat_leaf_values = absl::GetFlag(FLAGS_r755_leaf_value_centering);
  }

  auto search_res = open_spiel::dune_imperium::CompoundRolloutSearchDecision(
      *ctx->state, current_player, &local_r755_tree, ctx->r755_search_evals,
      top_k, rollouts_per_action, /*min_override_margin=*/0.15,
      open_spiel::dune_imperium::TreeReuseMode::kTargetFill, search_seed,
      open_spiel::dune_imperium::SearchPairingMode::kPaired, truncation_mode,
      crn_mode, /*world_sampler=*/nullptr, absl::GetFlag(FLAGS_r755_min_t_stat), opts);

  DumpR755RootJsonl(ctx->episode_id, ctx->r755_decision_index, cat_idx,
                    search_res, max_raw_p, p_2);

  const bool execute_decision =
      search_res.teacher_accepted
          ? absl::GetFlag(FLAGS_r755_execute_overrides)
          : (search_res.overridden && absl::GetFlag(FLAGS_r755_execute_overrides));
  const Action target_action =
      search_res.teacher_accepted ? search_res.teacher_action : search_res.chosen_action;

  if (execute_decision &&
      ApplyOverrideAction(actions, step->legal_probabilities, target_action, step)) {
    if (ctx->local_stats != nullptr && cat_idx >= 1 && cat_idx <= 5) {
      ctx->local_stats->r755_executed_overrides_by_cat[cat_idx]++;
    }
    EmitR755CompoundAuxAndProbe(ctx, *step, root_uid, cat, raw_policy_pairs, search_res);
  }
}

void SampleSearchReservoir(PpoSimContext* ctx, const SimStepDecision& step) {
  const bool is_turn_start = (step.current_player != ctx->last_player ||
                              ctx->last_action == dune_imperium::kActionEndTurn ||
                              ctx->last_player == kInvalidPlayer);
  const bool search_full_turn = absl::GetFlag(FLAGS_search_full_turn_supervision);
  const bool high_stakes_only = absl::GetFlag(FLAGS_search_high_stakes_only);
  bool passes_stakes = true;
  if (high_stakes_only) {
    passes_stakes = dune_imperium::IsHighStakesStrategicState(*ctx->state, step.current_player);
  }
  const bool eligible_decision = search_full_turn
      ? (is_turn_start && step.is_learner && step.actions.size() > 1 && passes_stakes)
      : (step.is_learner && step.actions.size() > 1 && passes_stakes);

  if (eligible_decision && ctx->local_stats != nullptr) {
    ctx->local_stats->search_eligible_decisions++;
  }
  if (ctx->local_snapshots != nullptr && ctx->decision_count != nullptr &&
      ctx->reservoir_rng != nullptr && eligible_decision) {
    (*ctx->decision_count)++;
    uint64_t m = *ctx->decision_count;
    int budget = absl::GetFlag(FLAGS_search_budget_per_update);
    if (ctx->local_snapshots->size() < static_cast<size_t>(budget)) {
      ctx->local_snapshots->push_back(BuildSearchSnapshot(*ctx, step));
    } else {
      std::uniform_int_distribution<uint64_t> dist(0, m - 1);
      uint64_t r_idx = dist(*ctx->reservoir_rng);
      if (r_idx < static_cast<uint64_t>(budget)) {
        (*ctx->local_snapshots)[r_idx] = BuildSearchSnapshot(*ctx, step);
      }
    }
  }
}

}  // namespace

void EvaluateAndSelectStepAction(PpoSimContext* ctx, SimStepDecision* step) {
  EvaluateModelAndCanaries(ctx, step);
  SampleAndExploreAction(ctx, step);
  RunR755HybridSearch(ctx, step);
  SampleSearchReservoir(ctx, *step);
}

void RecordLearnerTransition(PpoSimContext* ctx, SimStepDecision* step) {
  if (!step->is_learner) return;
  const Player current_player = step->current_player;
  const size_t n_actions = step->actions.size();

  PpoTransition transition;
  transition.state = std::move(step->obs);
  transition.legal_actions = std::move(step->actions);
  transition.action = step->action;
  transition.old_log_prob = step->old_log_prob;
  transition.reward = 0.0f;
  transition.value = step->eval_result.value;
  transition.advantage = 0.0f;
  transition.return_value = 0.0f;
  transition.player_id = current_player;
  transition.episode_id = ctx->episode_id;
  transition.behavior_legal_log_probs = std::move(step->parity_behavior_log_probs);
  transition.behavior_raw_legal_logits = std::move(step->parity_raw_legal_logits);
  transition.decision_role = step->parity_decision_role;
  transition.behavior_physical_batch_id = step->eval_result.physical_batch_id;
  transition.behavior_physical_batch_size = step->eval_result.physical_batch_size;
  transition.behavior_physical_batch_row = step->eval_result.physical_batch_row;
  transition.candidate_data = std::move(step->cand_data);
  const bool is_uniform_purchase_explore =
      step->is_forced_buy && !step->is_targeted_forced_buy;
  transition.mask_policy_loss =
      step->r755_override_accepted || is_uniform_purchase_explore;
  transition.off_policy_sampled =
      step->is_forced_buy || step->leader_draft_overridden;
  transition.targeted_buy_card = step->targeted_buy_card_id;
  transition.targeted_buy_round =
      (ctx->dune_state != nullptr) ? ctx->dune_state->GetCurrentRound() : 0;
  transition.round =
      (ctx->dune_state != nullptr) ? ctx->dune_state->GetCurrentRound() : 0;
  transition.targeted_buy_forced = step->is_targeted_forced_buy;
  transition.privileged_appendix = std::move(step->pre_action_privileged_appendix);
  transition.is_high_stakes = step->pre_action_is_high_stakes;
  ctx->trajectory->push_back(std::move(transition));

  if (n_actions > 1 && ctx->local_stats != nullptr) {
    ctx->local_stats->nontrivial_transitions++;
    if (step->pre_action_is_high_stakes) ctx->local_stats->high_stakes_pre_transitions++;
    if (step->post_action_is_high_stakes) ctx->local_stats->high_stakes_post_transitions++;
  }

  if (current_player >= 0 && current_player < ctx->game->NumPlayers()) {
    ctx->last_transition_index[current_player] =
        static_cast<int>(ctx->trajectory->size() - 1);
  }
}

#endif  // OPEN_SPIEL_BUILD_WITH_LIBTORCH

}  // namespace open_spiel::ppo_train_internal
