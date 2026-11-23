#include "dune_search_session_internal.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <numeric>

namespace open_spiel {
namespace {

dune_imperium::JointSearchV2Result RunJointPlacementWithPrimaryPriors(
    const State& state,
    Player search_player,
    dune_imperium::TurnSearchTree* turn_tree,
    const std::vector<std::shared_ptr<BatchedNNEvaluator>>& prior_evaluators,
    const std::vector<std::shared_ptr<BatchedNNEvaluator>>& rollout_evaluators,
    int reference_rollouts_per_action,
    double min_override_margin,
    uint64_t decision_seed,
    dune_imperium::SearchRolloutCrnMode crn_mode,
    double max_time_ms,
    int max_inflight_worlds,
    double prior_log_ratio_weight,
    double stage1_high_conf_prior_skip,
    double est_ms_per_sim,
    size_t max_candidates,
    const dune_imperium::SearchV2TuningConfig& tuning,
    double win_speed_tiebreak_eps,
    bool ceiling_aware_margin) {
  if (turn_tree != nullptr) turn_tree->BindPublicCardKnowledge(state);
  const auto legal = state.LegalActions();
  if (legal.empty()) {
    dune_imperium::JointSearchV2Result res;
    res.base = {kInvalidAction, false, 0.0, 0.0, 0, 0, {}, {}};
    return res;
  }
  const ActionsAndProbs raw_prior =
      prior_evaluators.empty() ? ActionsAndProbs{}
                               : prior_evaluators.front()->Prior(state);
  std::vector<dune_imperium::JointAgentCandidate> joint_cands =
      dune_imperium::EnumerateJointAgentCandidates(
          state, search_player, prior_evaluators, raw_prior, max_candidates);
  if (joint_cands.empty()) {
    Action raw_a = dune_imperium::PickGreedyAction(raw_prior, legal);
    dune_imperium::JointSearchV2Result res;
    res.base = {raw_a, false, 0.0, 0.0, 0, 0, {}, {}};
    res.chosen_first_action = raw_a;
    res.chosen_graft_action = kInvalidAction;
    res.chosen_second_action = kInvalidAction;
    res.raw_first_action = raw_a;
    res.raw_graft_action = kInvalidAction;
    res.raw_second_action = kInvalidAction;
    return res;
  }
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
  dune_imperium::CompoundSearchDecisionResult base_res =
      dune_imperium::SearchV2SuccessiveHalvingDecision(
          state, search_player, turn_tree, rollout_evaluators, ordered_cands,
          cand_priors, reference_rollouts_per_action, min_override_margin,
          decision_seed, crn_mode, max_time_ms, max_inflight_worlds,
          prior_log_ratio_weight, stage1_high_conf_prior_skip, &second_actions,
          &graft_actions, est_ms_per_sim, &chosen_c_idx, tuning,
          win_speed_tiebreak_eps, ceiling_aware_margin);
  dune_imperium::JointSearchV2Result res;
  res.base = base_res;
  res.raw_first_action = joint_cands[0].first_action;
  res.raw_graft_action = joint_cands[0].graft_action;
  res.raw_second_action = joint_cands[0].second_action;
  res.chosen_candidate_index = chosen_c_idx;
  if (chosen_c_idx >= 0 && chosen_c_idx < static_cast<int>(joint_cands.size())) {
    res.chosen_first_action = joint_cands[chosen_c_idx].first_action;
    res.chosen_graft_action = joint_cands[chosen_c_idx].graft_action;
    res.chosen_second_action = joint_cands[chosen_c_idx].second_action;
  } else {
    res.chosen_first_action = base_res.action;
    res.chosen_graft_action = kInvalidAction;
    res.chosen_second_action = kInvalidAction;
  }
  for (const auto& kv : base_res.candidate_rollout_utilities) {
    res.per_candidate_rollouts.push_back(kv.second);
  }
  res.joint_candidates = std::move(joint_cands);
  return res;
}

}  // namespace

PairedActionEvaluationResult DuneAdaptiveRolloutSession::EvaluateActionPairTerminal(
    const State& state, Action a_first, Action a_second,
    int num_worlds, uint64_t seed) {
  PairedActionEvaluationResult result;
  if (num_worlds <= 0 || !impl_ || impl_->evaluators.empty()) return result;

  const auto* dune = dynamic_cast<const dune_imperium::DuneImperiumState*>(&state);
  const Player search_player = state.CurrentPlayer();
  const std::vector<Action> ordered_candidates = {a_first, a_second};
  const std::vector<size_t> candidate_indices = {0, 1};

  std::vector<std::vector<double>> per_candidate_rollouts(2);
  int new_rollouts_executed = 0;
  auto deadline_tp = std::chrono::steady_clock::now() + std::chrono::hours(24);

  int kept_worlds = dune_imperium::RunClockedPairedRolloutStage(
      state,
      dune,
      search_player,
      ActiveRolloutEvaluators(),
      ordered_candidates,
      candidate_indices,
      num_worlds,
      /*world_seed_offset=*/0,
      seed,
      use_event_keyed_crn_ ? dune_imperium::SearchRolloutCrnMode::kEventKeyedCrn
                           : dune_imperium::SearchRolloutCrnMode::kSequential,
      /*sample_opponent_leader_draft=*/false,
      deadline_tp,
      /*active_tree=*/nullptr,
      per_candidate_rollouts,
      new_rollouts_executed,
      max_inflight_worlds_,
      /*resample_from_infostate=*/true,
      /*second_actions=*/nullptr,
      /*graft_actions=*/nullptr,
      win_speed_tiebreak_eps_);

  result.num_worlds = kept_worlds;
  if (kept_worlds > 0) {
    double sum0 = 0.0;
    double sum1 = 0.0;
    for (int r = 0; r < kept_worlds; ++r) {
      sum0 += per_candidate_rollouts[0][r];
      sum1 += per_candidate_rollouts[1][r];
    }
    result.u_first = sum0 / kept_worlds;
    result.u_second = sum1 / kept_worlds;
    result.diff_mean = result.u_first - result.u_second;

    double sum_sq_diff = 0.0;
    for (int r = 0; r < kept_worlds; ++r) {
      double d_r = per_candidate_rollouts[0][r] - per_candidate_rollouts[1][r];
      double diff = d_r - result.diff_mean;
      sum_sq_diff += diff * diff;
    }
    double var_d = (kept_worlds > 1) ? (sum_sq_diff / (kept_worlds - 1)) : 0.0;
    result.diff_se = (kept_worlds > 1) ? std::sqrt(var_d / kept_worlds) : 0.0;
  }
  return result;
}

DirectSearchV2RootResult DuneAdaptiveRolloutSession::RunDirectSearchV2Root(
    const State& state, double target_budget_ms, uint64_t decision_seed,
    bool is_purchase, int override_ref_rollouts) {
  DirectSearchV2RootResult res;
  if (!impl_ || impl_->evaluators.empty()) {
    return res;
  }
  const auto& dune_state = static_cast<const dune_imperium::DuneImperiumState&>(state);
  Player p = state.CurrentPlayer();
  const int r_bucket =
      dune_search_session_internal::ComputeEffectiveRoundBucket(state, p);
  const double est_ms_per_sim = ms_per_rollout_by_round_[r_bucket];
  res.est_ms_per_sim_used = est_ms_per_sim;
  const int ref_rollouts = (override_ref_rollouts > 0)
                               ? override_ref_rollouts
                               : std::max(4, static_cast<int>((0.82 * target_budget_ms) / est_ms_per_sim) / 3);
  const auto& rollout_evals = ActiveRolloutEvaluators();
  const int worker_s2_cap =
      128 * std::max(1, static_cast<int>(rollout_evals.size()));
  const int inflight = (target_budget_ms < 0.0)
                           ? std::min(worker_s2_cap, std::max(192, max_inflight_worlds_))
                           : std::min(worker_s2_cap, max_inflight_worlds_);
  auto eff_v2_tuning = search_v2_tuning_;
  eff_v2_tuning.s2_inflight_cap =
      std::min(eff_v2_tuning.s2_inflight_cap, worker_s2_cap);

  dune_imperium::TurnSearchTree local_tree;
  const auto t0 = std::chrono::steady_clock::now();

  if (!is_purchase) {
    const size_t max_cands = (target_budget_ms >= 0.0 && target_budget_ms < 2500.0) ? 5 : 8;
    auto joint_res = RunJointPlacementWithPrimaryPriors(
        state, p, &local_tree, impl_->evaluators, rollout_evals, ref_rollouts,
        0.15, decision_seed,
        (use_event_keyed_crn_ ? dune_imperium::SearchRolloutCrnMode::kEventKeyedCrn
                              : dune_imperium::SearchRolloutCrnMode::kSequential),
        target_budget_ms, inflight, prior_log_ratio_weight_,
        stage1_high_conf_prior_skip_, est_ms_per_sim, max_cands, eff_v2_tuning,
        win_speed_tiebreak_eps_, ceiling_aware_margin_);

    const auto t1 = std::chrono::steady_clock::now();
    res.elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    res.stage1_elapsed_ms = joint_res.base.stage1_elapsed_ms;
    res.stage2_elapsed_ms = joint_res.base.stage2_elapsed_ms;
    res.new_rollouts_executed = joint_res.base.new_rollouts_executed;
    res.stage1_rollouts_executed = joint_res.base.stage1_rollouts_executed;
    res.stage2_rollouts_executed = joint_res.base.stage2_rollouts_executed;
    res.stage1_planned_worlds = joint_res.base.stage1_planned_worlds;
    res.stage1_kept_worlds = joint_res.base.stage1_kept_worlds;
    res.stage2_planned_worlds = joint_res.base.stage2_planned_worlds;
    res.stage2_kept_worlds = joint_res.base.stage2_kept_worlds;
    res.termination_reason = joint_res.base.termination_reason;
    res.overridden = joint_res.base.overridden;
    res.chosen_first_action = joint_res.chosen_first_action;
    res.chosen_graft_action = joint_res.chosen_graft_action;
    res.chosen_second_action = joint_res.chosen_second_action;
    res.raw_first_action = joint_res.raw_first_action;
    res.raw_graft_action = joint_res.raw_graft_action;
    res.raw_second_action = joint_res.raw_second_action;
    res.chosen_candidate_index = joint_res.chosen_candidate_index;
    res.num_candidates = static_cast<int>(joint_res.joint_candidates.size());

    if (joint_res.base.overridden) {
      if (joint_res.chosen_first_action != joint_res.raw_first_action) {
        res.usable_override = true;
      } else if (joint_res.chosen_graft_action != kInvalidAction ||
                 joint_res.chosen_second_action != kInvalidAction) {
        dune_imperium::PendingJointAgentOverride pj;
        pj.active = true;
        pj.root_uid = 1;
        pj.stage1_kept_worlds = joint_res.base.stage1_kept_worlds;
        pj.root_hand_size = dune_imperium::ComputeAgentTurnHandPlusPlayed(&dune_state, p);
        pj.root_intrigue_count = static_cast<int>(dune_state.GetPlayerIntrigues(p).size());
        pj.chosen_first_action = joint_res.chosen_first_action;
        pj.executed_graft_action = kInvalidAction;
        pj.expected_graft_action = joint_res.chosen_graft_action;
        pj.expected_second_action = joint_res.chosen_second_action;
        pj.chosen_candidate_index = joint_res.chosen_candidate_index;
        pj.joint_candidates = joint_res.joint_candidates;
        pj.per_candidate_rollouts = joint_res.per_candidate_rollouts;

        auto sim = state.Clone();
        sim->ApplyAction(joint_res.chosen_first_action);
        for (int step = 0; step < 8 && !sim->IsTerminal() && !sim->IsChanceNode() && sim->CurrentPlayer() == p; ++step) {
          auto nl = sim->LegalActions();
          if (nl.empty()) break;
          if (nl.size() == 1) {
            if (nl[0] == dune_imperium::kActionEndTurn || nl[0] == dune_imperium::kActionReveal) break;
            sim->ApplyAction(nl[0]);
            continue;
          }
          auto sub_prior = impl_->evaluators.front()->Prior(*sim);
          Action step_greedy = dune_imperium::PickGreedyAction(sub_prior, nl);
          const auto* d_sim = dynamic_cast<const dune_imperium::DuneImperiumState*>(sim.get());
          int cur_hand = d_sim ? dune_imperium::ComputeAgentTurnHandPlusPlayed(d_sim, p) : -1;
          int cur_intr = d_sim ? static_cast<int>(d_sim->GetPlayerIntrigues(p).size()) : -1;
          auto follow = dune_imperium::AdvancePendingJointAgentOverride(
              &pj, nl, step_greedy, cur_hand, cur_intr, /*saw_chance_node=*/false);
          if (follow.emit_aux_target) {
            res.usable_override = true;
          }
          Action act_to_apply = follow.execute_override ? follow.forced_action : step_greedy;
          if (act_to_apply == dune_imperium::kActionEndTurn || act_to_apply == dune_imperium::kActionReveal) break;
          sim->ApplyAction(act_to_apply);
          if (!pj.active) break;
        }
      }
    }
  } else {
    const auto legal = state.LegalActions();
    const auto raw_prior = impl_->evaluators.front()->Prior(state);
    const int effective_top_k = fixed_top5_candidates_ ? 5 : 0;
    const bool include_all_buys =
        all_buy_candidates_ && dune_imperium::IsBuyDecisionState(state) &&
        (fixed_top5_candidates_ || 3 * ref_rollouts >= 48);
    auto [ordered_cands, cand_priors] =
        dune_imperium::SelectOrderedCandidatesAndPriors(
            raw_prior, legal, effective_top_k, include_all_buys);

    int chosen_c_idx = 0;
    const auto c_res = dune_imperium::SearchV2SuccessiveHalvingDecision(
        state, p, &local_tree, rollout_evals, ordered_cands, cand_priors,
        ref_rollouts, 0.15, decision_seed,
        (use_event_keyed_crn_ ? dune_imperium::SearchRolloutCrnMode::kEventKeyedCrn
                              : dune_imperium::SearchRolloutCrnMode::kSequential),
        target_budget_ms, inflight, prior_log_ratio_weight_,
        stage1_high_conf_prior_skip_, /*second_actions=*/nullptr,
        /*graft_actions=*/nullptr, est_ms_per_sim, &chosen_c_idx,
        eff_v2_tuning, win_speed_tiebreak_eps_, ceiling_aware_margin_);

    const auto t1 = std::chrono::steady_clock::now();
    res.elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    res.stage1_elapsed_ms = c_res.stage1_elapsed_ms;
    res.stage2_elapsed_ms = c_res.stage2_elapsed_ms;
    res.new_rollouts_executed = c_res.new_rollouts_executed;
    res.stage1_rollouts_executed = c_res.stage1_rollouts_executed;
    res.stage2_rollouts_executed = c_res.stage2_rollouts_executed;
    res.stage1_planned_worlds = c_res.stage1_planned_worlds;
    res.stage1_kept_worlds = c_res.stage1_kept_worlds;
    res.stage2_planned_worlds = c_res.stage2_planned_worlds;
    res.stage2_kept_worlds = c_res.stage2_kept_worlds;
    res.termination_reason = c_res.termination_reason;
    res.overridden = c_res.overridden;
    res.chosen_first_action = c_res.action;
    res.chosen_graft_action = kInvalidAction;
    res.chosen_second_action = kInvalidAction;
    res.raw_first_action = ordered_cands.empty() ? kInvalidAction : ordered_cands[0];
    res.raw_graft_action = kInvalidAction;
    res.raw_second_action = kInvalidAction;
    res.chosen_candidate_index = chosen_c_idx;
    res.num_candidates = static_cast<int>(ordered_cands.size());
    res.usable_override = c_res.overridden && (res.chosen_first_action != res.raw_first_action);
  }
  return res;
}

DirectSearchV1SequentialRootResult DuneAdaptiveRolloutSession::RunDirectSearchV1SequentialRoot(
    const State& state, double target_budget_ms, uint64_t decision_seed,
    int override_ref_rollouts) {
  DirectSearchV1SequentialRootResult res;
  if (!impl_ || impl_->evaluators.empty()) {
    return res;
  }
  Player p = state.CurrentPlayer();
  const int r_bucket =
      dune_search_session_internal::ComputeEffectiveRoundBucket(state, p);
  const double est_ms_per_sim = ms_per_rollout_by_round_[r_bucket];

  const auto legal = state.LegalActions();
  if (legal.empty()) return res;
  const auto raw_prior = impl_->evaluators.front()->Prior(state);
  Action raw_first_action = dune_imperium::PickGreedyAction(raw_prior, legal);
  Action raw_graft_action = kInvalidAction;
  Action raw_second_action = kInvalidAction;

  auto is_graft_action = [](Action a) {
    return dune_imperium::IsSelectGraftPartnerAction(a);
  };
  auto is_space_action = [](Action a) {
    return a >= dune_imperium::kActionAgentSpaceConspire &&
           a <= dune_imperium::kActionAgentSpaceSietchTabr;
  };

  auto raw_sim = state.Clone();
  raw_sim->ApplyAction(raw_first_action);
  for (int step = 0; step < 8 && !raw_sim->IsTerminal() && !raw_sim->IsChanceNode() && raw_sim->CurrentPlayer() == p; ++step) {
    auto r_nl = raw_sim->LegalActions();
    if (r_nl.empty()) break;
    Action step_a = kInvalidAction;
    if (r_nl.size() == 1) {
      step_a = r_nl[0];
    } else {
      auto r_prior = impl_->evaluators.front()->Prior(*raw_sim);
      step_a = dune_imperium::PickGreedyAction(r_prior, r_nl);
    }
    if (step_a == dune_imperium::kActionEndTurn || step_a == dune_imperium::kActionReveal) break;
    if (is_graft_action(step_a)) {
      raw_graft_action = step_a;
    } else if (is_space_action(step_a)) {
      raw_second_action = step_a;
      break;
    }
    raw_sim->ApplyAction(step_a);
  }

  dune_imperium::TurnSearchTree local_tree;
  const auto t0 = std::chrono::steady_clock::now();

  double root_budget_ms = -1.0;
  int root_ref = 0;
  int inflight = 0;
  const auto& rollout_evals = ActiveRolloutEvaluators();
  const int worker_s2_cap =
      128 * std::max(1, static_cast<int>(rollout_evals.size()));
  if (target_budget_ms >= 0.0) {
    root_budget_ms = 0.60 * target_budget_ms;
    root_ref = (override_ref_rollouts > 0)
                   ? override_ref_rollouts
                   : std::max(4, static_cast<int>((0.82 * root_budget_ms) / est_ms_per_sim) / 3);
    inflight = std::min(worker_s2_cap, max_inflight_worlds_);
  } else {
    root_budget_ms = -1.0;
    root_ref = (override_ref_rollouts > 0) ? override_ref_rollouts : reference_rollouts_per_action_;
    inflight = std::min(worker_s2_cap, std::max(192, max_inflight_worlds_));
  }

  const auto crn_mode = (use_event_keyed_crn_ ? dune_imperium::SearchRolloutCrnMode::kEventKeyedCrn
                                              : dune_imperium::SearchRolloutCrnMode::kSequential);

  auto s1_res = dune_imperium::SuccessiveHalvingRolloutSearchDecision(
      state, p, &local_tree, rollout_evals, root_ref, /*min_override_margin=*/0.15,
      decision_seed, crn_mode, unbiased_stage2_gate_, root_budget_ms,
      /*fixed_top_k=*/5, inflight, /*use_bounded_pool_when_unclocked=*/false,
      /*full_clock_stage2=*/false, prior_log_ratio_weight_,
      stage1_high_conf_prior_skip_, &raw_prior, /*all_buy_candidates=*/false,
      win_speed_tiebreak_eps_, ceiling_aware_margin_);

  Action chosen_first_action = s1_res.action;
  int total_rollouts = s1_res.new_rollouts_executed;
  bool any_ovr = (s1_res.overridden && s1_res.action != raw_first_action);
  Action chosen_graft_action = kInvalidAction;
  Action chosen_second_action = kInvalidAction;

  auto sim = state.Clone();
  local_tree.Advance(chosen_first_action);
  sim->ApplyAction(chosen_first_action);

  for (int step = 0; step < 8 && !sim->IsTerminal() && !sim->IsChanceNode() && sim->CurrentPlayer() == p; ++step) {
    auto nl = sim->LegalActions();
    if (nl.empty()) break;
    if (nl.size() == 1) {
      if (is_graft_action(nl[0])) chosen_graft_action = nl[0];
      if (is_space_action(nl[0])) {
        chosen_second_action = nl[0];
        break;
      }
      if (nl[0] == dune_imperium::kActionEndTurn || nl[0] == dune_imperium::kActionReveal) break;
      local_tree.Advance(nl[0]);
      sim->ApplyAction(nl[0]);
      continue;
    }

    auto sub_prior = impl_->evaluators.front()->Prior(*sim);
    Action sub_raw = dune_imperium::PickGreedyAction(sub_prior, nl);

    bool has_graft = false, has_space = false;
    for (Action a : nl) {
      if (is_graft_action(a)) has_graft = true;
      if (is_space_action(a)) has_space = true;
    }

    if (has_graft || has_space) {
      const double elapsed_ms_since_t0 =
          std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
      double sub_budget_ms = -1.0;
      if (target_budget_ms >= 0.0) {
        double rem = std::max(0.0, target_budget_ms - elapsed_ms_since_t0);
        sub_budget_ms = has_graft ? (0.50 * rem) : (1.0 * rem);
      }
      int sub_ref = (override_ref_rollouts > 0)
                        ? override_ref_rollouts
                        : (target_budget_ms >= 0.0
                               ? std::max(4, static_cast<int>((0.82 * sub_budget_ms) / est_ms_per_sim) / 3)
                               : reference_rollouts_per_action_);

      Action sub_chosen = sub_raw;
      if (target_budget_ms < 0.0 || sub_budget_ms >= 5.0) {
        uint64_t sub_seed = decision_seed ^ (has_graft ? 0x517CC1B727220A95ULL : 0x9E3779B97F4A7C15ULL);
        auto sub_res = dune_imperium::SuccessiveHalvingRolloutSearchDecision(
            *sim, p, &local_tree, rollout_evals, sub_ref, 0.15, sub_seed,
            crn_mode, unbiased_stage2_gate_, sub_budget_ms, 5, inflight, false, false,
            prior_log_ratio_weight_, stage1_high_conf_prior_skip_, &sub_prior,
            /*all_buy_candidates=*/false, win_speed_tiebreak_eps_,
            ceiling_aware_margin_);
        sub_chosen = sub_res.action;
        total_rollouts += sub_res.new_rollouts_executed;
        if (sub_res.overridden && sub_chosen != sub_raw) any_ovr = true;
      } else if (local_tree.current_node() && !local_tree.current_node()->children.empty()) {
        sub_chosen = local_tree.BestConfirmedAction(sub_raw, sub_prior, nl);
        if (sub_chosen != sub_raw) any_ovr = true;
      }

      if (has_graft) chosen_graft_action = sub_chosen;
      if (has_space) {
        chosen_second_action = sub_chosen;
        break;
      }
      local_tree.Advance(sub_chosen);
      sim->ApplyAction(sub_chosen);
    } else {
      if (sub_raw == dune_imperium::kActionEndTurn || sub_raw == dune_imperium::kActionReveal) break;
      local_tree.Advance(sub_raw);
      sim->ApplyAction(sub_raw);
    }
  }

  const auto t1 = std::chrono::steady_clock::now();
  res.elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
  res.new_rollouts_executed = total_rollouts;
  res.overridden = any_ovr;
  res.usable_override = any_ovr;
  res.chosen_first_action = chosen_first_action;
  res.chosen_graft_action = chosen_graft_action;
  res.chosen_second_action = chosen_second_action;
  res.raw_first_action = raw_first_action;
  res.raw_graft_action = raw_graft_action;
  res.raw_second_action = raw_second_action;
  return res;
}

EvaluateJointCandidatesResult DuneAdaptiveRolloutSession::EvaluateJointCandidatesPaired(
    const State& state,
    const std::vector<Action>& first_actions,
    const std::vector<Action>& graft_actions,
    const std::vector<Action>& second_actions,
    int num_worlds,
    uint64_t decision_seed) {
  EvaluateJointCandidatesResult res;
  if (!impl_ || impl_->evaluators.empty() || first_actions.empty()) {
    return res;
  }
  const size_t K = first_actions.size();
  const auto* dune = dynamic_cast<const dune_imperium::DuneImperiumState*>(&state);
  Player p = state.CurrentPlayer();
  std::vector<size_t> candidate_indices(K);
  std::iota(candidate_indices.begin(), candidate_indices.end(), 0);

  std::vector<std::vector<double>> per_candidate_rollouts(K);
  for (size_t c = 0; c < K; ++c) {
    per_candidate_rollouts[c].reserve(num_worlds);
  }
  int rollouts_executed = 0;
  dune_imperium::TurnSearchTree local_tree;

  const auto& rollout_evals = ActiveRolloutEvaluators();
  const int worker_s2_cap =
      128 * std::max(1, static_cast<int>(rollout_evals.size()));
  const int inflight = std::clamp(
      num_worlds, 2, std::min(worker_s2_cap, std::max(2, 512 / static_cast<int>(K))));

  const auto t0 = std::chrono::steady_clock::now();
  dune_imperium::RunClockedPairedRolloutStage(
      state, dune, p, rollout_evals, first_actions, candidate_indices,
      num_worlds, /*world_seed_offset=*/0, decision_seed,
      dune_imperium::SearchRolloutCrnMode::kEventKeyedCrn,
      /*sample_opponent_leader_draft=*/false,
      std::chrono::steady_clock::time_point::max(), &local_tree,
      per_candidate_rollouts, rollouts_executed, inflight,
      /*resample_from_infostate=*/true,
      &second_actions, &graft_actions, win_speed_tiebreak_eps_);
  const auto t1 = std::chrono::steady_clock::now();

  res.elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
  res.rollouts_executed = rollouts_executed;
  res.per_candidate_rollouts = per_candidate_rollouts;
  res.mean_utilities.resize(K, 0.0);
  for (size_t c = 0; c < K; ++c) {
    if (!per_candidate_rollouts[c].empty()) {
      double sum = 0.0;
      for (double v : per_candidate_rollouts[c]) sum += v;
      res.mean_utilities[c] = sum / per_candidate_rollouts[c].size();
    }
  }
  return res;
}

}  // namespace open_spiel
