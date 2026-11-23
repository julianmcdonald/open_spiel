#include "dune_search_session_internal.h"

#include <algorithm>

namespace open_spiel {
namespace dune_search_session_internal {

int ComputeEffectiveRoundBucket(const State& state, Player p) {
  const auto& dune_state =
      static_cast<const dune_imperium::DuneImperiumState&>(state);
  int eff_r = dune_state.GetCurrentRound();
  if (dune_state.phase() == dune_imperium::GamePhase::kCombat ||
      dune_state.phase() == dune_imperium::GamePhase::kRevealTurns ||
      dune_state.IsPlayerInRevealTurn(p)) {
    eff_r += 1;
  }
  return std::clamp(eff_r, 1, 11);
}

void SyncAdaptiveTurnTree(
    const State& state,
    const dune_imperium::DuneImperiumState& dune_state,
    Player p,
    dune_imperium::TurnSearchTree& tree,
    bool has_active_session,
    Player last_player,
    int last_round,
    int last_hand,
    int last_intrigues,
    const std::vector<Action>& last_history,
    double& turn_elapsed_search_time_ms,
    Action& pending_macro_space_hint) {
  if (!has_active_session || p != last_player ||
      dune_state.GetCurrentRound() != last_round) {
    tree.Reset();
    turn_elapsed_search_time_ms = 0.0;
    pending_macro_space_hint = kInvalidAction;
  } else if (state.History().size() > last_history.size() &&
             std::equal(last_history.begin(), last_history.end(),
                        state.History().begin())) {
    if (dune_state.GetPlayerCardsInHand(p) > last_hand ||
        static_cast<int>(dune_state.GetPlayerIntrigues(p).size()) >
            last_intrigues) {
      tree.Reset();
      turn_elapsed_search_time_ms = 0.0;
      pending_macro_space_hint = kInvalidAction;
    } else {
      for (size_t i = last_history.size(); i < state.History().size(); ++i) {
        Action a = state.History()[i];
        tree.Advance(a);
        if (a == dune_imperium::kActionEndTurn) {
          tree.Reset();
          turn_elapsed_search_time_ms = 0.0;
          pending_macro_space_hint = kInvalidAction;
        }
      }
    }
  } else if (state.History() != last_history) {
    tree.Reset();
    turn_elapsed_search_time_ms = 0.0;
    pending_macro_space_hint = kInvalidAction;
  }
  tree.BindPublicCardKnowledge(state);
}

void UpdateAdaptiveRolloutRateEma(
    const State& state,
    Player p,
    bool is_leader,
    double remaining_time_ms,
    double elapsed_ms,
    const DuneSearchResult& result,
    bool search_v2,
    bool superhuman_mode,
    double& ms_per_rollout_ema,
    std::array<double, 12>& ms_per_rollout_by_round) {
  if (remaining_time_ms >= 0.0 && result.simulations_completed >= 6 &&
      !result.used_fallback && elapsed_ms > 0.0) {
    double current_rate = elapsed_ms / result.simulations_completed;
    ms_per_rollout_ema = 0.8 * ms_per_rollout_ema + 0.2 * current_rate;
    if (result.simulations_completed >=
        ((search_v2 || superhuman_mode) ? 24 : 160)) {
      int r_bucket = is_leader ? 0 : ComputeEffectiveRoundBucket(state, p);
      double clamped_rate = std::clamp(current_rate, 0.4, 35.0);
      ms_per_rollout_by_round[r_bucket] =
          0.8 * ms_per_rollout_by_round[r_bucket] + 0.2 * clamped_rate;
    }
  }
}

BranchRemResult CountBranchSearchedWeighted(
    const State& state,
    Player p,
    BatchedNNEvaluator* prior_eval,
    int hand0,
    int int0,
    Action first_action,
    Action graft_action,
    Action second_action,
    int max_count,
    double stage1_high_conf_prior_skip,
    bool allow_non_hs_subsearch) {
  auto sim = state.Clone();
  sim->ApplyAction(first_action);
  if (graft_action != kInvalidAction) {
    if (!sim->IsTerminal() && !sim->IsChanceNode() &&
        sim->CurrentPlayer() == p) {
      const auto leg = sim->LegalActions();
      if (std::find(leg.begin(), leg.end(), graft_action) != leg.end()) {
        sim->ApplyAction(graft_action);
      }
    }
  }
  if (second_action != kInvalidAction) {
    if (!sim->IsTerminal() && !sim->IsChanceNode() &&
        sim->CurrentPlayer() == p) {
      const auto leg = sim->LegalActions();
      if (std::find(leg.begin(), leg.end(), second_action) != leg.end()) {
        sim->ApplyAction(second_action);
      }
    }
  }
  bool tree_broken = false;
  double weighted_rem = 0.0;
  int raw_rem_count = 0;
  for (int step = 0; step < 16 && !sim->IsTerminal(); ++step) {
    if (sim->IsChanceNode()) {
      tree_broken = true;
      auto outs = sim->ChanceOutcomes();
      if (outs.empty()) break;
      sim->ApplyAction(outs[0].first);
      continue;
    }
    if (sim->CurrentPlayer() != p) break;
    const auto& sim_d =
        static_cast<const dune_imperium::DuneImperiumState&>(*sim);
    if (sim_d.phase() != dune_imperium::GamePhase::kAgentTurns ||
        sim_d.IsPlayerInRevealTurn(p)) {
      break;
    }
    if (sim_d.GetPlayerCardsInHand(p) > hand0 ||
        static_cast<int>(sim_d.GetPlayerIntrigues(p).size()) > int0) {
      tree_broken = true;
    }
    std::vector<Action> nl = sim->LegalActions();
    if (nl.empty()) break;
    if (nl.size() == 1) {
      if (nl[0] == dune_imperium::kActionEndTurn ||
          nl[0] == dune_imperium::kActionReveal) {
        break;
      }
      sim->ApplyAction(nl[0]);
      continue;
    }
    bool has_end_or_reveal = false;
    int non_spec = 0;
    for (Action na : nl) {
      if (na == dune_imperium::kActionEndTurn ||
          na == dune_imperium::kActionReveal) {
        has_end_or_reveal = true;
      }
      if (!dune_imperium::IsConvertSpecimenAction(na)) {
        non_spec++;
      }
    }
    if (has_end_or_reveal && non_spec < 2) {
      break;
    }
    bool hs_or_buy = dune_imperium::IsHighStakesStrategicState(*sim, p) ||
                     dune_imperium::IsPurchaseDecisionState(*sim);
    ActionsAndProbs sub_prior = prior_eval->Prior(*sim);
    bool sub_search = hs_or_buy;
    if (!sub_search && allow_non_hs_subsearch) {
      if (!has_end_or_reveal && tree_broken) {
        sub_search = true;
      } else if (has_end_or_reveal &&
                 dune_imperium::IsFullStrategicState(*sim, p, sub_prior,
                                                     0.020)) {
        sub_search = true;
      }
    }
    if (sub_search) {
      double sub_max_p = 0.0;
      for (const auto& ap : sub_prior) {
        if (std::find(nl.begin(), nl.end(), ap.first) != nl.end()) {
          sub_max_p = std::max(sub_max_p, ap.second);
        }
      }
      raw_rem_count++;
      weighted_rem += (stage1_high_conf_prior_skip > 0.0 &&
                       sub_max_p >= stage1_high_conf_prior_skip)
                          ? 0.18
                          : 1.0;
      if (raw_rem_count >= max_count) break;
    }
    Action next_a = dune_imperium::PickGreedyAction(sub_prior, nl);
    if (next_a == dune_imperium::kActionEndTurn ||
        next_a == dune_imperium::kActionReveal) {
      break;
    }
    sim->ApplyAction(next_a);
  }
  return {weighted_rem, raw_rem_count};
}

void PopulateRolloutStageDiagnostics(
    DuneSearchResult& result,
    const dune_imperium::CompoundSearchDecisionResult& c_res) {
  result.simulations_completed = c_res.new_rollouts_executed;
  result.diagnostics.newly_completed_simulations = c_res.new_rollouts_executed;
  result.diagnostics.inherited_root_visits = c_res.prior_rollouts_reused;
  result.diagnostics.stage1_planned_worlds = c_res.stage1_planned_worlds;
  result.diagnostics.stage1_kept_worlds = c_res.stage1_kept_worlds;
  result.diagnostics.stage2_planned_worlds = c_res.stage2_planned_worlds;
  result.diagnostics.stage2_kept_worlds = c_res.stage2_kept_worlds;
  result.diagnostics.stage1_rollouts_executed = c_res.stage1_rollouts_executed;
  result.diagnostics.stage2_rollouts_executed = c_res.stage2_rollouts_executed;
  result.diagnostics.stage1_elapsed_ms = c_res.stage1_elapsed_ms;
  result.diagnostics.stage2_elapsed_ms = c_res.stage2_elapsed_ms;
  result.diagnostics.termination_reason = c_res.termination_reason;
}

void PopulateCandidateDiagnosticsByAction(
    DuneSearchResult& result,
    const std::vector<Action>& legal,
    const dune_imperium::CompoundSearchDecisionResult& c_res) {
  for (const auto& cu : c_res.candidate_utilities) {
    for (size_t i = 0; i < legal.size(); ++i) {
      if (legal[i] == cu.first) {
        result.diagnostics.q_values[i] = cu.second;
        break;
      }
    }
  }
  for (const auto& cru : c_res.candidate_rollout_utilities) {
    for (size_t i = 0; i < legal.size(); ++i) {
      if (legal[i] == cru.first) {
        result.diagnostics.visit_counts[i] =
            static_cast<int>(cru.second.size());
        break;
      }
    }
  }
}

void PopulateJointCandidateDiagnostics(
    DuneSearchResult& result,
    const std::vector<Action>& legal,
    const std::vector<dune_imperium::JointAgentCandidate>& joint_cands,
    const dune_imperium::CompoundSearchDecisionResult& c_res) {
  for (size_t i = 0; i < legal.size(); ++i) {
    Action a = legal[i];
    double max_q = -1e9;
    int max_v = 0;
    bool found = false;
    for (size_t j = 0; j < joint_cands.size(); ++j) {
      if (joint_cands[j].first_action == a) {
        found = true;
        if (j < c_res.candidate_utilities.size()) {
          max_q = std::max(max_q, c_res.candidate_utilities[j].second);
        }
        if (j < c_res.candidate_rollout_utilities.size()) {
          max_v = std::max(
              max_v, static_cast<int>(
                         c_res.candidate_rollout_utilities[j].second.size()));
        }
      }
    }
    if (found) {
      result.diagnostics.q_values[i] = max_q;
      result.diagnostics.visit_counts[i] = max_v;
    }
  }
}

void CheckLiveDeadlineFallback(
    DuneSearchResult& result,
    const dune_imperium::CompoundSearchDecisionResult& c_res,
    double remaining_time_ms,
    bool is_leader,
    bool search_v2,
    int num_cands) {
  const int retained_pairs = c_res.candidate_rollout_utilities.empty()
      ? 0
      : static_cast<int>(c_res.candidate_rollout_utilities[0].second.size());
  bool timed_out = false;
  if (remaining_time_ms >= 0.0) {
    if (is_leader) {
      timed_out = search_v2
          ? (remaining_time_ms <= 50.0 && retained_pairs < 2)
          : (retained_pairs < 2 || c_res.new_rollouts_executed < 2 * num_cands);
    } else {
      timed_out = (retained_pairs < 2);
    }
  }
  if (timed_out) {
    result.used_fallback = true;
    result.fallback_reason = "live_deadline_reached";
    result.diagnostics.fallback_reason = "live_deadline_reached";
  }
}

void RunAdaptiveLeaderRolloutSearch(
    const State& state,
    Player p,
    const std::vector<Action>& legal,
    const ActionsAndProbs& raw_prior,
    const std::vector<std::shared_ptr<BatchedNNEvaluator>>& evaluators,
    int effective_leader_rollouts,
    double leader_deadline_ms,
    int leader_inflight,
    uint64_t leader_seed,
    bool use_event_keyed_crn,
    double win_speed_tiebreak_eps,
    bool ceiling_aware_margin,
    double remaining_time_ms,
    bool search_v2,
    DuneSearchResult& result) {
  const int num_cands = std::max(1, static_cast<int>(legal.size()));
  const auto c_res = dune_imperium::LeaderPickRolloutSearchDecision(
      state, p, evaluators, effective_leader_rollouts, 0.10, leader_seed,
      use_event_keyed_crn ? dune_imperium::SearchRolloutCrnMode::kEventKeyedCrn
                          : dune_imperium::SearchRolloutCrnMode::kSequential,
      /*sample_opponent_leader_draft=*/false,
      /*max_time_ms=*/leader_deadline_ms, leader_inflight,
      /*precomputed_raw_prior=*/&raw_prior, win_speed_tiebreak_eps,
      ceiling_aware_margin);
  result.diagnostics.selected_action = c_res.action;
  result.diagnostics.mcts_proposed_action = c_res.action;
  result.diagnostics.mcts_overrode_raw = c_res.overridden;
  PopulateRolloutStageDiagnostics(result, c_res);
  result.policy = raw_prior;
  PopulateCandidateDiagnosticsByAction(result, legal, c_res);
  CheckLiveDeadlineFallback(result, c_res, remaining_time_ms,
                            /*is_leader=*/true, search_v2, num_cands);
}

void RunAdaptiveV2JointRolloutSearch(
    const State& state,
    const AdaptiveRolloutDispatchParams& params,
    const std::vector<Action>& legal,
    Action raw_action,
    const ActionsAndProbs& raw_prior,
    dune_imperium::TurnSearchTree* tree,
    const std::vector<std::shared_ptr<BatchedNNEvaluator>>& evaluators,
    const std::vector<dune_imperium::JointAgentCandidate>& joint_cands,
    DuneSearchResult& result,
    bool& out_cached_followups_valid,
    int& out_chosen_c_idx,
    Action& out_chosen_first,
    Action& out_chosen_graft,
    Action& out_chosen_second,
    double& out_chosen_q,
    int& out_chosen_visits) {
  std::vector<Action> ordered_cands;
  std::vector<Action> graft_actions;
  std::vector<Action> second_actions;
  std::vector<double> cand_priors;
  ordered_cands.reserve(joint_cands.size());
  graft_actions.reserve(joint_cands.size());
  second_actions.reserve(joint_cands.size());
  cand_priors.reserve(joint_cands.size());
  for (const auto& jc : joint_cands) {
    ordered_cands.push_back(jc.first_action);
    graft_actions.push_back(jc.graft_action);
    second_actions.push_back(jc.second_action);
    cand_priors.push_back(jc.joint_prior);
  }
  int chosen_c_idx = 0;
  const auto c_res = dune_imperium::SearchV2SuccessiveHalvingDecision(
      state, params.player, tree, evaluators, ordered_cands, cand_priors,
      params.ref_rollouts, 0.15, params.decision_seed,
      (params.use_event_keyed_crn
           ? dune_imperium::SearchRolloutCrnMode::kEventKeyedCrn
           : dune_imperium::SearchRolloutCrnMode::kSequential),
      params.target_ms, params.max_inflight_worlds,
      params.prior_log_ratio_weight, params.stage1_high_conf_prior_skip,
      &second_actions, &graft_actions, params.round_ms_est, &chosen_c_idx,
      params.search_v2_tuning, params.win_speed_tiebreak_eps,
      params.ceiling_aware_margin);

  Action chosen_first = ordered_cands[chosen_c_idx];
  Action chosen_graft = graft_actions[chosen_c_idx];
  Action chosen_second = second_actions[chosen_c_idx];

  result.diagnostics.selected_action = chosen_first;
  result.diagnostics.mcts_proposed_action = chosen_first;
  result.diagnostics.mcts_overrode_raw =
      (chosen_first != raw_action) || c_res.overridden;
  PopulateRolloutStageDiagnostics(result, c_res);
  result.policy = raw_prior;
  PopulateJointCandidateDiagnostics(result, legal, joint_cands, c_res);
  CheckLiveDeadlineFallback(result, c_res, params.remaining_time_ms,
                            /*is_leader=*/false, /*search_v2=*/true,
                            static_cast<int>(legal.size()));

  out_cached_followups_valid = !result.used_fallback;
  out_chosen_c_idx = chosen_c_idx;
  out_chosen_first = chosen_first;
  out_chosen_graft = chosen_graft;
  out_chosen_second = chosen_second;
  out_chosen_q = c_res.chosen_mean_utility;
  out_chosen_visits =
      (chosen_c_idx < static_cast<int>(c_res.candidate_rollout_utilities.size()))
          ? static_cast<int>(
                c_res.candidate_rollout_utilities[chosen_c_idx].second.size())
          : 0;
}

void RunAdaptiveV2SingleRolloutSearch(
    const State& state,
    const AdaptiveRolloutDispatchParams& params,
    const std::vector<Action>& legal,
    const ActionsAndProbs& raw_prior,
    dune_imperium::TurnSearchTree* tree,
    const std::vector<std::shared_ptr<BatchedNNEvaluator>>& evaluators,
    DuneSearchResult& result) {
  const int min_top5_budget = std::min(params.top5_min_budget, 96);
  const bool use_top5 =
      params.fixed_top5_candidates && 3 * params.ref_rollouts >= min_top5_budget;
  const int effective_top_k = use_top5 ? 5 : 0;
  const bool include_all_buys =
      params.all_buy_candidates && dune_imperium::IsBuyDecisionState(state) &&
      (params.target_ms < 0.0 || params.target_ms >= 1500.0) &&
      (use_top5 || 3 * params.ref_rollouts >= 48);
  auto [ordered_cands, cand_priors] =
      dune_imperium::SelectOrderedCandidatesAndPriors(
          raw_prior, legal, effective_top_k, include_all_buys);

  const auto c_res = dune_imperium::SearchV2SuccessiveHalvingDecision(
      state, params.player, tree, evaluators, ordered_cands, cand_priors,
      params.ref_rollouts, 0.15, params.decision_seed,
      (params.use_event_keyed_crn
           ? dune_imperium::SearchRolloutCrnMode::kEventKeyedCrn
           : dune_imperium::SearchRolloutCrnMode::kSequential),
      params.target_ms, params.max_inflight_worlds,
      params.prior_log_ratio_weight, params.stage1_high_conf_prior_skip,
      /*second_actions=*/nullptr, /*graft_actions=*/nullptr,
      params.round_ms_est, /*chosen_candidate_index=*/nullptr,
      params.search_v2_tuning, params.win_speed_tiebreak_eps,
      params.ceiling_aware_margin);

  result.diagnostics.selected_action = c_res.action;
  result.diagnostics.mcts_proposed_action = c_res.action;
  result.diagnostics.mcts_overrode_raw = c_res.overridden;
  PopulateRolloutStageDiagnostics(result, c_res);
  result.policy = raw_prior;
  PopulateCandidateDiagnosticsByAction(result, legal, c_res);
  CheckLiveDeadlineFallback(result, c_res, params.remaining_time_ms,
                            /*is_leader=*/false, /*search_v2=*/true,
                            static_cast<int>(legal.size()));
}

void RunAdaptiveV1RolloutSearch(
    const State& state,
    const AdaptiveRolloutDispatchParams& params,
    const std::vector<Action>& legal,
    const ActionsAndProbs& raw_prior,
    dune_imperium::TurnSearchTree* tree,
    const std::vector<std::shared_ptr<BatchedNNEvaluator>>& evaluators,
    BatchedNNEvaluator* prior_eval,
    DuneSearchResult& result,
    Action& out_chosen_second_action) {
  const int worker_s2_cap = 128 * std::max(1, params.num_evaluators);
  const int min_top5_budget =
      params.full_search ? std::min(params.top5_min_budget, 96)
                         : params.top5_min_budget;
  const bool use_top5 =
      params.fixed_top5_candidates && 3 * params.ref_rollouts >= min_top5_budget;
  const int effective_top_k = use_top5 ? 5 : 0;
  const bool include_all_buys =
      params.all_buy_candidates && dune_imperium::IsBuyDecisionState(state) &&
      (use_top5 || (params.full_search && 3 * params.ref_rollouts >= 48));
  const int eff_inflight = std::min(params.max_inflight_worlds, worker_s2_cap);
  const bool eff_unbiased_s2 =
      params.unbiased_stage2_gate &&
      (!params.superhuman_mode || params.remaining_time_ms < 0.0 ||
       params.target_ms >= 2000.0);
  const bool eff_rescue_abilities =
      params.rescue_optional_abilities &&
      (params.remaining_time_ms < 0.0 || params.target_ms >= 2000.0);

  dune_imperium::MacroSpaceConfig macro_cfg;
  macro_cfg.enabled = params.enable_macro_space_candidates;
  macro_cfg.min_round = params.macro_space_min_round;

  dune_imperium::EndgamePly2Config ply2_cfg;
  ply2_cfg.enabled = params.enable_endgame_ply2;
  ply2_cfg.stage2_only = params.endgame_ply2_stage2_only;
  ply2_cfg.inner_worlds = params.endgame_ply2_worlds;
  ply2_cfg.margin = params.endgame_ply2_margin;

  dune_imperium::IntrigueBeliefConfig belief_cfg;
  belief_cfg.enabled = params.enable_intrigue_belief;
  belief_cfg.held_age_combat_log_slope = params.intrigue_belief_age_combat_slope;
  belief_cfg.held_age_endgame_log_slope =
      params.intrigue_belief_age_endgame_slope;
  belief_cfg.close_pass_combat_log_penalty =
      params.intrigue_belief_close_pass_combat_penalty;
  if (belief_cfg.enabled) {
    auto ev = dune_imperium::ComputeOpponentIntrigueEvidenceFromHistory(
        state, params.player);
    belief_cfg.mean_age = ev.mean_age;
    belief_cfg.close_passes = ev.close_passes;
  }

  const auto c_res = params.use_successive_halving
      ? dune_imperium::SuccessiveHalvingRolloutSearchDecision(
            state, params.player, tree, evaluators, params.ref_rollouts, 0.15,
            params.decision_seed,
            (params.use_event_keyed_crn
                 ? dune_imperium::SearchRolloutCrnMode::kEventKeyedCrn
                 : dune_imperium::SearchRolloutCrnMode::kSequential),
            eff_unbiased_s2, params.target_ms, effective_top_k, eff_inflight,
            /*use_bounded_pool_when_unclocked=*/false,
            /*full_clock_stage2=*/params.full_search,
            params.prior_log_ratio_weight, params.stage1_high_conf_prior_skip,
            /*precomputed_raw_prior=*/&raw_prior, include_all_buys,
            params.win_speed_tiebreak_eps, params.ceiling_aware_margin,
            macro_cfg, params.forced_include_action,
            /*prior_eval_override=*/prior_eval, &ply2_cfg,
            belief_cfg.enabled ? &belief_cfg : nullptr, eff_rescue_abilities)
      : dune_imperium::AdaptiveRolloutSearchDecision(
            state, params.player, tree, evaluators, params.ref_rollouts, 0.15,
            params.decision_seed,
            (params.use_event_keyed_crn
                 ? dune_imperium::SearchRolloutCrnMode::kEventKeyedCrn
                 : dune_imperium::SearchRolloutCrnMode::kSequential),
            params.target_ms, eff_inflight,
            /*precomputed_raw_prior=*/&raw_prior, params.win_speed_tiebreak_eps,
            params.ceiling_aware_margin);

  out_chosen_second_action = c_res.chosen_second_action;
  result.diagnostics.selected_action = c_res.action;
  result.diagnostics.mcts_proposed_action = c_res.action;
  result.diagnostics.mcts_overrode_raw = c_res.overridden;
  PopulateRolloutStageDiagnostics(result, c_res);
  result.policy = raw_prior;
  PopulateCandidateDiagnosticsByAction(result, legal, c_res);
  CheckLiveDeadlineFallback(result, c_res, params.remaining_time_ms,
                            /*is_leader=*/false, /*search_v2=*/false,
                            static_cast<int>(legal.size()));
}

void FinalizeTreeReuseOrFallback(
    dune_imperium::TurnSearchTree* tree,
    const std::vector<Action>& legal,
    Action raw_action,
    const ActionsAndProbs& raw_prior,
    bool search_v2,
    bool live_deadline_starved,
    double remaining_time_ms,
    double available_after_reserve_ms,
    DuneSearchResult& result) {
  Action chosen = raw_action;
  int inherited_visits = 0;
  int covered_in_tree = 0;
  if (tree->current_node() && !tree->current_node()->children.empty()) {
    for (size_t i = 0; i < legal.size(); ++i) {
      Action a = legal[i];
      if (tree->HasPriorData(a)) {
        int v = tree->GetVisits(a);
        double q = tree->GetMeanUtility(a);
        result.diagnostics.visit_counts[i] = v;
        result.diagnostics.q_values[i] = q;
        inherited_visits += v;
        covered_in_tree++;
      }
    }
    chosen = tree->BestConfirmedAction(raw_action, raw_prior, legal);
  }

  if (covered_in_tree > 0) {
    result.diagnostics.selected_action = chosen;
    result.diagnostics.mcts_proposed_action = chosen;
    result.diagnostics.mcts_overrode_raw = (chosen != raw_action);
    result.used_fallback = false;
    result.fallback_reason = "none";
    result.diagnostics.fallback_reason = "none";
    result.diagnostics.termination_reason = "tree_reuse";
    result.diagnostics.inherited_root_visits = inherited_visits;
    result.policy = raw_prior;
  } else {
    result.diagnostics.selected_action = raw_action;
    result.diagnostics.mcts_proposed_action = raw_action;
    result.used_fallback = true;
    std::string reason;
    if (search_v2) {
      reason = (remaining_time_ms >= 0.0 && remaining_time_ms <= 50.0)
                   ? "live_deadline_reached"
                   : "policy_only";
    } else {
      reason = (live_deadline_starved ||
                (remaining_time_ms >= 0.0 && available_after_reserve_ms <= 0.0))
                   ? "live_deadline_reached"
                   : "policy_only";
    }
    result.fallback_reason = reason;
    result.diagnostics.fallback_reason = reason;
    result.diagnostics.termination_reason = reason;
    result.policy = raw_prior;
  }
}

void PlanV1SearchBudget(
    const State& state,
    const dune_imperium::DuneImperiumState& dune_state,
    Player p,
    const std::vector<Action>& legal,
    const ActionsAndProbs& raw_prior,
    bool is_agent_turn_phase,
    bool full_search,
    bool superhuman_mode,
    bool is_cuda,
    double reserve_ms,
    int reference_rollouts_per_action,
    double available_after_reserve_ms,
    double remaining_time_ms,
    double round_ms_est,
    BatchedNNEvaluator* prior_eval,
    DuneSearchResult& result,
    double& target_ms,
    int& ref_rollouts) {
  double usable_ms = full_search
      ? ((is_agent_turn_phase ? 0.980 : 0.980) * available_after_reserve_ms)
      : available_after_reserve_ms;
  double frac = is_agent_turn_phase
      ? 0.60
      : (superhuman_mode ? ((reserve_ms > 0.0) ? 0.96 : 0.92)
                         : (full_search ? 1.0 : 0.75));
  double cap = is_agent_turn_phase ? 25000.0 : 30000.0;
  if (full_search && is_agent_turn_phase) {
    const int hand0 = dune_state.GetPlayerCardsInHand(p);
    const int int0 =
        static_cast<int>(dune_state.GetPlayerIntrigues(p).size());
    const bool sub_blitz_hs_only = superhuman_mode &&
                                   remaining_time_ms >= 0.0 &&
                                   available_after_reserve_ms <= 4500.0;

    int rem_0 = 0;
    int max_other_rem = 0;
    const auto sorted_cands =
        dune_imperium::GetSortedCandidatesByPrior(raw_prior, legal);
    if (!sorted_cands.empty()) {
      rem_0 = CountBranchSearchedWeighted(
                  state, p, prior_eval, hand0, int0, sorted_cands[0].first,
                  kInvalidAction, kInvalidAction, 3, 0.0, !sub_blitz_hs_only)
                  .raw_rem_count;
    }
    const size_t k_check = std::min<size_t>(5, sorted_cands.size());
    for (size_t ci = 1; ci < k_check; ++ci) {
      int r_ci = CountBranchSearchedWeighted(
                     state, p, prior_eval, hand0, int0, sorted_cands[ci].first,
                     kInvalidAction, kInvalidAction, 1, 0.0,
                     !sub_blitz_hs_only)
                     .raw_rem_count;
      max_other_rem = std::max(max_other_rem, r_ci);
      if (max_other_rem >= 1) break;
    }

    cap = 49200.0;
    if (superhuman_mode) {
      if (rem_0 == 0) {
        frac = (max_other_rem == 0) ? ((reserve_ms > 0.0) ? 0.96 : 0.90) : 0.82;
      } else if (rem_0 == 1) {
        frac = (max_other_rem == 0) ? 0.54 : 0.50;
      } else if (rem_0 == 2) {
        frac = 0.44;
      } else {
        frac = 0.36;
      }
    } else {
      if (rem_0 == 0) {
        frac = (max_other_rem == 0) ? 1.0 : 0.94;
      } else if (rem_0 == 1) {
        if (usable_ms >= 38000.0) {
          frac = 0.50;
        } else {
          frac = (max_other_rem == 0) ? 0.68 : 0.65;
        }
      } else if (rem_0 == 2) {
        frac = (usable_ms >= 38000.0) ? 0.45 : 0.54;
      } else {
        frac = 0.40;
      }
    }
  }
  target_ms = std::min(cap, frac * usable_ms);
  result.diagnostics.hard_time_limit_ms = target_ms;
  result.diagnostics.soft_time_limit_ms = remaining_time_ms;
  double plan_ms = full_search ? (0.82 * target_ms) : target_ms;
  int total_rollout_budget = static_cast<int>(plan_ms / round_ms_est);
  ref_rollouts = superhuman_mode
      ? std::min(320, total_rollout_budget / 3)
      : (full_search
             ? (total_rollout_budget / 3)
             : std::min(4 * reference_rollouts_per_action,
                        total_rollout_budget / 3));
  static constexpr double kMinCudaWaveMs[11] = {
      700.0, 620.0, 520.0, 430.0, 340.0, 260.0, 190.0, 150.0, 120.0, 100.0, 90.0};
  const int cur_r_clamped = std::clamp(dune_state.GetCurrentRound(), 1, 10);
  const double cuda_wave_floor_ms =
      (superhuman_mode && is_cuda) ? kMinCudaWaveMs[cur_r_clamped] : 240.0;
  const double min_subdecision_ms = superhuman_mode
      ? std::max(cuda_wave_floor_ms, 10.0 * round_ms_est)
      : std::max(1800.0, 24.0 * round_ms_est);
  if ((full_search && is_agent_turn_phase && usable_ms < min_subdecision_ms) ||
      (superhuman_mode && is_cuda && target_ms < cuda_wave_floor_ms)) {
    ref_rollouts = 0;
  } else if (superhuman_mode && remaining_time_ms > cuda_wave_floor_ms &&
             target_ms >= cuda_wave_floor_ms && total_rollout_budget >= 6 &&
             ref_rollouts < 4) {
    ref_rollouts = 4;
  }
}

}  // namespace dune_search_session_internal
}  // namespace open_spiel
