#include "dune_search_session_internal.h"

#include <algorithm>
#include <chrono>

#include "dune_seed_utils.h"

namespace open_spiel {
namespace {

bool IsAgentPrimaryJointState(
    const State& state,
    const dune_imperium::DuneImperiumState& dune_state,
    Player p,
    const std::vector<Action>& legal) {
  if (dune_state.phase() != dune_imperium::GamePhase::kAgentTurns ||
      dune_state.IsPlayerInRevealTurn(p)) {
    return false;
  }
  for (Action a1 : legal) {
    if (dune_imperium::IsConvertSpecimenAction(a1)) continue;
    auto s1 = state.Clone();
    s1->ApplyAction(a1);
    if (!s1->IsTerminal() && !s1->IsChanceNode() && s1->CurrentPlayer() == p) {
      const auto l2 = s1->LegalActions();
      if (!l2.empty() &&
          (std::all_of(l2.begin(), l2.end(), [](Action a) {
             return dune_imperium::IsBoardSpaceAction(a);
           }) ||
           std::all_of(l2.begin(), l2.end(), [](Action a) {
             return dune_imperium::IsGraftSelectionAction(a);
           }))) {
        return true;
      }
    }
  }
  return false;
}

bool ShouldSearchAdaptiveState(
    const State& state,
    const dune_imperium::DuneImperiumState& dune_state,
    Player p,
    const std::vector<Action>& legal,
    const ActionsAndProbs& raw_prior,
    Action raw_action,
    const dune_imperium::TurnSearchTree& tree,
    bool search_v2,
    bool full_search,
    bool superhuman_mode,
    bool is_agent_primary_joint,
    bool enable_macro_space_candidates,
    bool is_board_space_step,
    Action forced_include_action,
    double remaining_time_ms,
    double available_after_reserve_ms) {
  bool should_search = false;
  if (search_v2) {
    if (is_agent_primary_joint) {
      should_search = true;
    } else if (dune_imperium::IsPurchaseDecisionState(state)) {
      bool has_buy = false;
      for (Action a : legal) {
        if (dune_imperium::IsCardOrTechAcquisitionAction(a) ||
            a == dune_imperium::kActionTleilaxuAcquire0) {
          has_buy = true;
          break;
        }
      }
      if (has_buy || dune_imperium::IsHighStakesStrategicState(state, p)) {
        should_search = true;
      }
    } else if (dune_imperium::IsHighStakesStrategicState(state, p)) {
      should_search = true;
    } else if (state.CurrentPlayer() == p && legal.size() >= 2) {
      if (remaining_time_ms < 0.0 || available_after_reserve_ms >= 900.0) {
        if (dune_state.phase() == dune_imperium::GamePhase::kCombat) {
          should_search = true;
        } else if (dune_imperium::IsFullStrategicState(state, p, raw_prior,
                                                       0.025)) {
          should_search = true;
        } else if (dune_state.phase() == dune_imperium::GamePhase::kAgentTurns &&
                   !dune_state.IsPlayerInRevealTurn(p)) {
          const bool has_tree_reuse = tree.HasConfirmedSubtree(raw_action, 16);
          int non_spec = 0;
          for (Action a : legal) {
            if (!dune_imperium::IsConvertSpecimenAction(a)) ++non_spec;
          }
          if (!has_tree_reuse && non_spec >= 2) {
            should_search = true;
          }
        }
      }
    }
    if (!is_agent_primary_joint && remaining_time_ms >= 0.0 &&
        available_after_reserve_ms < 320.0) {
      should_search = false;
    }
  } else {
    should_search = dune_imperium::IsHighStakesStrategicState(state, p);
    const bool blitz_high_stakes_only =
        superhuman_mode && remaining_time_ms >= 0.0 &&
        available_after_reserve_ms <= 4500.0;
    if (full_search && !should_search && state.CurrentPlayer() == p &&
        legal.size() >= 2) {
      if (dune_imperium::IsPurchaseDecisionState(state) ||
          dune_state.phase() == dune_imperium::GamePhase::kCombat) {
        should_search = true;
      } else if (!blitz_high_stakes_only) {
        bool has_end_or_reveal = false;
        for (Action a : legal) {
          if (a == dune_imperium::kActionEndTurn ||
              a == dune_imperium::kActionReveal) {
            has_end_or_reveal = true;
            break;
          }
        }
        const bool has_tree_reuse = tree.HasConfirmedSubtree(raw_action, 16);
        if (!has_end_or_reveal && !has_tree_reuse) {
          should_search = true;
        } else if (has_end_or_reveal &&
                   dune_imperium::IsFullStrategicState(state, p, raw_prior,
                                                       0.020)) {
          should_search = true;
        }
      }
    }
    if (enable_macro_space_candidates && is_board_space_step &&
        forced_include_action != kInvalidAction &&
        forced_include_action != raw_action) {
      should_search = true;
    }
  }
  return should_search;
}

void PlanV2SearchBudget(
    const State& state,
    const dune_imperium::DuneImperiumState& dune_state,
    Player p,
    const ActionsAndProbs& raw_prior,
    Action raw_action,
    bool is_agent_primary_joint,
    bool is_agent_turn_phase,
    double available_after_reserve_ms,
    double remaining_time_ms,
    double round_ms_est,
    double stage1_high_conf_prior_skip,
    const dune_imperium::SearchV2TuningConfig& search_v2_tuning,
    const std::vector<std::shared_ptr<BatchedNNEvaluator>>& evaluators,
    std::vector<dune_imperium::JointAgentCandidate>& joint_cands,
    DuneSearchResult& result,
    double& target_ms,
    int& ref_rollouts) {
  BatchedNNEvaluator* prior_eval = evaluators.front().get();
  if (is_agent_primary_joint) {
    const size_t max_cands =
        (available_after_reserve_ms < 2500.0) ? 5 : 8;
    joint_cands = dune_imperium::EnumerateJointAgentCandidates(
        state, p, evaluators, raw_prior, max_cands);

    const int hand0 = dune_state.GetPlayerCardsInHand(p);
    const int int0 =
        static_cast<int>(dune_state.GetPlayerIntrigues(p).size());

    if (search_v2_tuning.enable_rem0_turn_reservation) {
      double rem_0 = 0.0;
      double max_rem_sub = 0.0;
      int max_raw_rem = 0;
      for (size_t i = 0; i < joint_cands.size(); ++i) {
        auto b_res = dune_search_session_internal::CountBranchSearchedWeighted(
            state, p, prior_eval, hand0, int0, joint_cands[i].first_action,
            joint_cands[i].graft_action, joint_cands[i].second_action, 3,
            stage1_high_conf_prior_skip, /*allow_non_hs_subsearch=*/true);
        if (i == 0) {
          rem_0 = b_res.weighted_rem;
        }
        max_rem_sub = std::max(max_rem_sub, b_res.weighted_rem);
        max_raw_rem = std::max(max_raw_rem, b_res.raw_rem_count);
      }
      const double p_0 =
          joint_cands.empty() ? 1.0 : joint_cands[0].joint_prior;
      const double eff_rem =
          (p_0 >= 0.70) ? rem_0 : (0.5 * rem_0 + 0.5 * max_rem_sub);

      result.diagnostics.pred_rem_0 = rem_0;
      result.diagnostics.pred_max_rem_sub = max_rem_sub;
      result.diagnostics.pred_eff_rem = eff_rem;
      result.diagnostics.pred_max_raw_rem = max_raw_rem;

      double frac = 0.48;
      if (eff_rem <= 0.05) {
        frac = (max_raw_rem == 0) ? 0.94 : 0.88;
      } else if (eff_rem <= 0.40) {
        frac = (max_raw_rem <= 1) ? 0.84 : 0.78;
      } else if (eff_rem <= 0.85) {
        frac = 0.72;
      } else if (eff_rem <= 1.35) {
        frac = 0.60;
      } else {
        frac = 0.48;
      }
      const double hard_cap_ms = (max_raw_rem == 0) ? 47500.0 : 44500.0;
      target_ms = std::min(hard_cap_ms, frac * available_after_reserve_ms);
    } else {
      const size_t num_check = std::min<size_t>(4, joint_cands.size());
      int max_rem_sub = 0;
      for (size_t i = 0; i < num_check; ++i) {
        int rem = dune_search_session_internal::CountBranchSearchedWeighted(
                      state, p, prior_eval, hand0, int0,
                      joint_cands[i].first_action, joint_cands[i].graft_action,
                      joint_cands[i].second_action, 3,
                      stage1_high_conf_prior_skip,
                      /*allow_non_hs_subsearch=*/true)
                      .raw_rem_count;
        max_rem_sub = std::max(max_rem_sub, rem);
      }
      double frac = 0.85;
      if (max_rem_sub == 1) {
        frac = 0.62;
      } else if (max_rem_sub >= 2) {
        frac = 0.48;
      }
      target_ms = std::min(46000.0, frac * available_after_reserve_ms);
    }
  } else if (is_agent_turn_phase) {
    const int hand0 = dune_state.GetPlayerCardsInHand(p);
    const int int0 =
        static_cast<int>(dune_state.GetPlayerIntrigues(p).size());

    if (search_v2_tuning.enable_rem0_turn_reservation) {
      auto b_res = dune_search_session_internal::CountBranchSearchedWeighted(
          state, p, prior_eval, hand0, int0, raw_action, kInvalidAction,
          kInvalidAction, 2, stage1_high_conf_prior_skip,
          /*allow_non_hs_subsearch=*/true);
      result.diagnostics.pred_rem_0 = b_res.weighted_rem;
      result.diagnostics.pred_max_rem_sub = b_res.weighted_rem;
      result.diagnostics.pred_eff_rem = b_res.weighted_rem;
      result.diagnostics.pred_max_raw_rem = b_res.raw_rem_count;
      double frac = (b_res.weighted_rem >= 0.85)
                        ? 0.58
                        : (b_res.weighted_rem > 0.05 ? 0.78 : 0.90);
      double intra_cap_ms = (b_res.raw_rem_count == 0) ? 44000.0 : 32000.0;
      target_ms = std::min(intra_cap_ms, frac * available_after_reserve_ms);
    } else {
      int rem_sub = dune_search_session_internal::CountBranchSearchedWeighted(
                        state, p, prior_eval, hand0, int0, raw_action,
                        kInvalidAction, kInvalidAction, 2,
                        stage1_high_conf_prior_skip,
                        /*allow_non_hs_subsearch=*/true)
                        .raw_rem_count;
      double frac = (rem_sub >= 1) ? 0.58 : 0.88;
      target_ms = std::min(28000.0, frac * available_after_reserve_ms);
    }
  } else {
    double frac = 0.90;
    target_ms = std::min(30000.0, frac * available_after_reserve_ms);
  }
  result.diagnostics.hard_time_limit_ms = target_ms;
  result.diagnostics.soft_time_limit_ms = remaining_time_ms;
  double plan_ms = 0.82 * target_ms;
  int total_rollout_budget = static_cast<int>(plan_ms / round_ms_est);
  ref_rollouts = std::max(4, total_rollout_budget / 3);
}

}  // namespace

DuneSearchResult DuneAdaptiveRolloutSession::Search(const State& state,
                                                   double remaining_time_ms) {
  if (has_pending_commit_) {
    SpielFatalError("DuneAdaptiveRolloutSession::Search called while a commit is still pending!");
  }
  const auto& dune_state = static_cast<const dune_imperium::DuneImperiumState&>(state);
  Player p = state.CurrentPlayer();

  dune_search_session_internal::SyncAdaptiveTurnTree(
      state, dune_state, p, impl_->tree, has_active_session_, last_player_,
      last_round_, last_hand_, last_intrigues_, last_history_,
      turn_elapsed_search_time_ms_, pending_macro_space_hint_);

  has_active_session_ = true;
  last_player_ = p;
  last_round_ = dune_state.GetCurrentRound();
  last_hand_ = (p >= 0 && p < 4) ? dune_state.GetPlayerCardsInHand(p) : 0;
  last_intrigues_ = (p >= 0 && p < 4) ? static_cast<int>(dune_state.GetPlayerIntrigues(p).size()) : 0;
  last_history_ = state.History();

  const auto t_start = std::chrono::steady_clock::now();
  auto [raw_prior, values] = impl_->evaluators.front()->PriorAndEvaluate(state);
  std::vector<Action> legal = state.LegalActions();
  Action raw_action = legal.empty() ? kInvalidAction : dune_imperium::PickGreedyAction(raw_prior, legal);

  bool is_board_space_step = !legal.empty() &&
      std::all_of(legal.begin(), legal.end(), dune_imperium::IsBoardSpaceAction);
  Action forced_include_action = kInvalidAction;
  if (is_board_space_step && pending_macro_space_hint_ != kInvalidAction) {
    forced_include_action = pending_macro_space_hint_;
    pending_macro_space_hint_ = kInvalidAction;
  }

  DuneSearchResult result;
  result.raw_policy = raw_prior;
  result.diagnostics.protocol_version = "adaptive_rollout";
  result.diagnostics.searched_seat = p;
  result.diagnostics.round = dune_state.GetCurrentRound();
  result.diagnostics.phase = dune_search_session_internal::LocalPhaseToString(dune_state.phase());
  result.diagnostics.actions = legal;
  result.diagnostics.priors.assign(legal.size(), 0.0);
  for (size_t i = 0; i < legal.size(); ++i) {
    Action a = legal[i];
    for (const auto& ap : raw_prior) {
      if (ap.first == a) {
        result.diagnostics.priors[i] = ap.second;
        break;
      }
    }
  }
  result.diagnostics.raw_priors = result.diagnostics.priors;
  result.diagnostics.root_value = (p >= 0 && p < static_cast<int>(values.size())) ? values[p] : 0.0;
  result.diagnostics.raw_reference_action = raw_action;
  result.diagnostics.visit_counts.assign(legal.size(), 0);
  result.diagnostics.q_values.assign(legal.size(), result.diagnostics.root_value);

  if (!cached_joint_followups_.empty()) {
    if (p != cached_joint_followup_player_ ||
        dune_state.GetCurrentRound() != cached_joint_followup_round_) {
      cached_joint_followups_.clear();
    } else {
      const auto& cur_hist = state.History();
      cached_joint_followups_.erase(
          std::remove_if(cached_joint_followups_.begin(), cached_joint_followups_.end(),
                         [&](const CachedJointFollowupStep& step) {
                           if (step.expected_history.size() < cur_hist.size()) return true;
                           for (size_t i = 0; i < cur_hist.size(); ++i) {
                             if (step.expected_history[i] != cur_hist[i]) return true;
                           }
                           return false;
                         }),
          cached_joint_followups_.end());
    }
  }

  if (search_v2_ && !cached_joint_followups_.empty()) {
    const auto& cur_hist = state.History();
    for (auto it = cached_joint_followups_.begin(); it != cached_joint_followups_.end(); ++it) {
      if (it->expected_history == cur_hist) {
        auto leg_it = std::find(legal.begin(), legal.end(), it->action);
        if (leg_it != legal.end()) {
          Action hit_action = it->action;
          double hit_q = it->q;
          int hit_visits = it->visits;
          cached_joint_followups_.erase(it);

          result.diagnostics.selected_action = hit_action;
          result.diagnostics.mcts_proposed_action = hit_action;
          result.diagnostics.mcts_overrode_raw = (hit_action != raw_action);
          result.diagnostics.search_accepted = (hit_action != raw_action);
          result.simulations_completed = 0;
          result.used_fallback = false;
          result.fallback_reason = "none";
          result.diagnostics.fallback_reason = "none";
          result.diagnostics.termination_reason = "joint_followup_cache";
          result.policy.clear();
          result.policy.reserve(legal.size());
          for (Action la : legal) {
            result.policy.push_back({la, (la == hit_action) ? 1.0 : 0.0});
          }

          size_t chosen_idx = std::distance(legal.begin(), leg_it);
          result.diagnostics.q_values[chosen_idx] = hit_q;
          result.diagnostics.visit_counts[chosen_idx] = hit_visits;

          result.diagnostics.total_root_visits = hit_visits;
          result.diagnostics.num_covered_actions = 1;

          const auto t_end = std::chrono::steady_clock::now();
          double elapsed_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
          result.elapsed_time_ms = elapsed_ms;
          result.diagnostics.elapsed_search_time_ms = elapsed_ms;
          session_elapsed_time_ms_ += elapsed_ms;
          turn_elapsed_search_time_ms_ += elapsed_ms;

          last_search_state_ = state.Clone();
          last_search_result_ = result;
          has_pending_commit_ = true;
          return result;
        }
      }
    }
  }

  bool is_leader = (dune_state.phase() == dune_imperium::GamePhase::kLeaderDraft ||
                    dune_state.phase() == dune_imperium::GamePhase::kLeaderOfferChance);

  if (legal.size() <= 1) {
    result.diagnostics.selected_action = raw_action;
    result.diagnostics.mcts_proposed_action = raw_action;
    result.used_fallback = true;
    result.fallback_reason = (remaining_time_ms == 0.0) ? "live_deadline_reached" : "forced_or_bookkeeping";
    result.diagnostics.fallback_reason = result.fallback_reason;
    result.diagnostics.termination_reason = result.diagnostics.fallback_reason;
    result.policy = raw_prior;
  } else if (is_leader) {
    const double eff_reserve_ms = (remaining_time_ms >= 0.0)
        ? (search_v2_ ? std::min(reserve_ms_, 0.15 * remaining_time_ms)
                      : ((superhuman_mode_ && reserve_ms_ < 8000.0)
                             ? std::min(reserve_ms_, 0.12 * remaining_time_ms)
                             : reserve_ms_))
        : 0.0;
    const double available_after_reserve_ms = (remaining_time_ms >= 0.0)
        ? (remaining_time_ms - eff_reserve_ms)
        : -1.0;

    int effective_leader_rollouts = leader_rollouts_;
    double target_ms = -1.0;
    bool deadline_reached = false;

    if (remaining_time_ms >= 0.0) {
      if (search_v2_ ? (remaining_time_ms <= 50.0 || available_after_reserve_ms <= 0.0)
                     : (available_after_reserve_ms <= 0.0)) {
        deadline_reached = true;
      } else {
        double usable_leader_ms = superhuman_mode_
            ? ((reserve_ms_ > 0.0) ? (0.975 * available_after_reserve_ms)
                                   : (0.985 * remaining_time_ms))
            : ((search_v2_ || full_search_)
                   ? (0.965 * available_after_reserve_ms)
                   : available_after_reserve_ms);
        target_ms = std::min(25000.0, usable_leader_ms);
        int num_cands = std::max(1, static_cast<int>(legal.size()));
        double leader_ms_est = (search_v2_ || full_search_)
            ? std::max(2.0, ms_per_rollout_by_round_[0])
            : std::max(2.0, ms_per_rollout_ema_);
        double plan_ms = (search_v2_ || full_search_) ? (0.975 * target_ms) : target_ms;
        int total_leader_rollout_budget = static_cast<int>(plan_ms / leader_ms_est);
        if (superhuman_mode_) {
          if (remaining_time_ms <= 1600.0) {
            effective_leader_rollouts = (remaining_time_ms > 120.0) ? 2 : 0;
          } else {
            effective_leader_rollouts = std::clamp(
                static_cast<int>((0.45 * plan_ms) / (leader_ms_est * num_cands)),
                2, 48);
          }
        } else {
          effective_leader_rollouts = (search_v2_ || full_search_)
              ? (total_leader_rollout_budget / num_cands)
              : std::min(2 * leader_rollouts_, total_leader_rollout_budget / num_cands);
        }
        if (search_v2_ && target_ms < 4000.0) {
          effective_leader_rollouts = std::clamp(effective_leader_rollouts, 2, 8);
        }
        if (effective_leader_rollouts < 2) {
          if (!search_v2_ || remaining_time_ms <= 50.0) {
            deadline_reached = true;
          } else {
            effective_leader_rollouts = 2;
          }
        }
      }
    }
    result.diagnostics.hard_time_limit_ms = target_ms;
    result.diagnostics.soft_time_limit_ms = remaining_time_ms;

    if (deadline_reached) {
      result.diagnostics.selected_action = raw_action;
      result.diagnostics.mcts_proposed_action = raw_action;
      result.used_fallback = true;
      result.fallback_reason = "live_deadline_reached";
      result.diagnostics.fallback_reason = "live_deadline_reached";
      result.diagnostics.termination_reason = result.diagnostics.fallback_reason;
      result.policy = raw_prior;
    } else {
      int num_cands = std::max(1, static_cast<int>(legal.size()));
      const int worker_inflight_cap = 128 * std::max(1, num_evaluators_);
      const bool small_caller_cap = (max_inflight_worlds_ > 0 && max_inflight_worlds_ < 32);
      int leader_inflight = search_v2_
          ? std::clamp(effective_leader_rollouts, 2, std::max(2, 192 / num_cands))
          : (superhuman_mode_
                 ? (small_caller_cap
                        ? max_inflight_worlds_
                        : std::min(worker_inflight_cap, std::max(2, 256 / num_cands)))
                 : (full_search_
                        ? std::min(worker_inflight_cap, std::max(max_inflight_worlds_, 384 / num_cands))
                        : max_inflight_worlds_));
      double leader_deadline_ms = superhuman_mode_
          ? target_ms
          : (((search_v2_ || full_search_) && target_ms > 0.0) ? (1.008 * target_ms) : target_ms);
      const uint64_t leader_seed = dune_seed::DeriveSeed(
          seed_, 0x4C454144ULL, static_cast<uint64_t>(p), search_seed_counter_++);
      dune_search_session_internal::RunAdaptiveLeaderRolloutSearch(
          state, p, legal, raw_prior, ActiveRolloutEvaluators(),
          effective_leader_rollouts, leader_deadline_ms, leader_inflight,
          leader_seed, use_event_keyed_crn_, win_speed_tiebreak_eps_,
          ceiling_aware_margin_, remaining_time_ms, search_v2_, result);
    }
  } else {
    const double eff_reserve_ms = (remaining_time_ms >= 0.0)
        ? (search_v2_ ? std::min(reserve_ms_, 0.15 * remaining_time_ms)
                      : ((superhuman_mode_ && reserve_ms_ < 8000.0)
                             ? std::min(reserve_ms_, 0.12 * remaining_time_ms)
                             : reserve_ms_))
        : 0.0;
    const double available_after_reserve_ms = (remaining_time_ms >= 0.0)
        ? (remaining_time_ms - eff_reserve_ms)
        : -1.0;

    bool ran_high_stakes = false;
    bool live_deadline_starved = false;

    const bool is_agent_primary_joint =
        search_v2_ && IsAgentPrimaryJointState(state, dune_state, p, legal);
    const bool should_search = ShouldSearchAdaptiveState(
        state, dune_state, p, legal, raw_prior, raw_action, impl_->tree,
        search_v2_, full_search_, superhuman_mode_, is_agent_primary_joint,
        enable_macro_space_candidates_, is_board_space_step,
        forced_include_action, remaining_time_ms, available_after_reserve_ms);

    if (should_search) {
      if (remaining_time_ms >= 0.0 &&
          (search_v2_ ? (remaining_time_ms <= 50.0 || available_after_reserve_ms <= 0.0)
                      : (available_after_reserve_ms <= 0.0))) {
        live_deadline_starved = true;
      } else {
        int ref_rollouts = reference_rollouts_per_action_;
        double target_ms = -1.0;
        const bool is_agent_turn_phase =
            (dune_state.phase() == dune_imperium::GamePhase::kAgentTurns &&
             !dune_state.IsPlayerInRevealTurn(p));

        const int r_bucket =
            dune_search_session_internal::ComputeEffectiveRoundBucket(state, p);
        const double round_ms_est = (search_v2_ || full_search_)
            ? std::max(0.4, ms_per_rollout_by_round_[r_bucket])
            : std::max(2.0, ms_per_rollout_ema_);

        std::vector<dune_imperium::JointAgentCandidate> joint_cands;

        if (search_v2_ && remaining_time_ms >= 0.0) {
          PlanV2SearchBudget(
              state, dune_state, p, raw_prior, raw_action,
              is_agent_primary_joint, is_agent_turn_phase,
              available_after_reserve_ms, remaining_time_ms, round_ms_est,
              stage1_high_conf_prior_skip_, search_v2_tuning_, impl_->evaluators,
              joint_cands, result, target_ms, ref_rollouts);
        } else if (!search_v2_ && remaining_time_ms >= 0.0) {
          dune_search_session_internal::PlanV1SearchBudget(
              state, dune_state, p, legal, raw_prior, is_agent_turn_phase,
              full_search_, superhuman_mode_,
              impl_ && impl_->device.is_cuda(), reserve_ms_,
              reference_rollouts_per_action_, available_after_reserve_ms,
              remaining_time_ms, round_ms_est, impl_->evaluators.front().get(),
              result, target_ms, ref_rollouts);
        }

        if (!search_v2_ && !superhuman_mode_ && remaining_time_ms >= 0.0 && ref_rollouts < 4) {
          live_deadline_starved = true;
        } else if (!search_v2_ && superhuman_mode_ && remaining_time_ms >= 0.0 && ref_rollouts < 4) {
          live_deadline_starved = (remaining_time_ms <= 50.0 || available_after_reserve_ms <= 0.0);
        }
        if (search_v2_ || remaining_time_ms < 0.0 || ref_rollouts >= 4) {
          ran_high_stakes = true;
          const uint64_t decision_seed =
              dune_seed::DeriveSeed(seed_, static_cast<uint64_t>(p), search_seed_counter_++);
          const int worker_s2_cap = 128 * std::max(1, num_evaluators_);

          dune_search_session_internal::AdaptiveRolloutDispatchParams params;
          params.player = p;
          params.remaining_time_ms = remaining_time_ms;
          params.target_ms = target_ms;
          params.ref_rollouts = ref_rollouts;
          params.round_ms_est = round_ms_est;
          params.decision_seed = decision_seed;
          params.max_inflight_worlds = max_inflight_worlds_;
          params.num_evaluators = num_evaluators_;
          params.top5_min_budget = top5_min_budget_;
          params.full_search = full_search_;
          params.superhuman_mode = superhuman_mode_;
          params.fixed_top5_candidates = fixed_top5_candidates_;
          params.all_buy_candidates = all_buy_candidates_;
          params.use_event_keyed_crn = use_event_keyed_crn_;
          params.use_successive_halving = use_successive_halving_;
          params.unbiased_stage2_gate = unbiased_stage2_gate_;
          params.rescue_optional_abilities = rescue_optional_abilities_;
          params.ceiling_aware_margin = ceiling_aware_margin_;
          params.win_speed_tiebreak_eps = win_speed_tiebreak_eps_;
          params.prior_log_ratio_weight = prior_log_ratio_weight_;
          params.stage1_high_conf_prior_skip = stage1_high_conf_prior_skip_;
          params.enable_macro_space_candidates = enable_macro_space_candidates_;
          params.macro_space_min_round = macro_space_min_round_;
          params.forced_include_action = forced_include_action;
          params.enable_endgame_ply2 = enable_endgame_ply2_;
          params.endgame_ply2_stage2_only = endgame_ply2_stage2_only_;
          params.endgame_ply2_worlds = endgame_ply2_worlds_;
          params.endgame_ply2_margin = endgame_ply2_margin_;
          params.enable_intrigue_belief = enable_intrigue_belief_;
          params.intrigue_belief_age_combat_slope = intrigue_belief_age_combat_slope_;
          params.intrigue_belief_age_endgame_slope = intrigue_belief_age_endgame_slope_;
          params.intrigue_belief_close_pass_combat_penalty = intrigue_belief_close_pass_combat_penalty_;
          params.search_v2_tuning = search_v2_tuning_;
          params.search_v2_tuning.s2_inflight_cap =
              std::min(params.search_v2_tuning.s2_inflight_cap, worker_s2_cap);

          if (search_v2_) {
            if (is_agent_primary_joint) {
              if (joint_cands.empty()) {
                const size_t max_cands =
                    (remaining_time_ms >= 0.0 && available_after_reserve_ms < 2500.0) ? 5 : 8;
                joint_cands = dune_imperium::EnumerateJointAgentCandidates(
                    state, p, impl_->evaluators, raw_prior, max_cands);
              }
              bool cached_followups_valid = false;
              int chosen_c_idx = 0;
              Action chosen_first = kInvalidAction;
              Action chosen_graft = kInvalidAction;
              Action chosen_second = kInvalidAction;
              double chosen_q = 0.0;
              int chosen_visits = 0;
              dune_search_session_internal::RunAdaptiveV2JointRolloutSearch(
                  state, params, legal, raw_action, raw_prior, &(impl_->tree),
                  ActiveRolloutEvaluators(), joint_cands, result,
                  cached_followups_valid, chosen_c_idx, chosen_first,
                  chosen_graft, chosen_second, chosen_q, chosen_visits);
              if (cached_followups_valid) {
                cached_joint_followup_round_ = dune_state.GetCurrentRound();
                cached_joint_followup_player_ = p;
                cached_joint_followups_.clear();
                const auto base_hist = state.History();
                if (chosen_graft != kInvalidAction) {
                  auto hist_step1 = base_hist;
                  hist_step1.push_back(chosen_first);
                  cached_joint_followups_.push_back(
                      {hist_step1, chosen_graft, chosen_q, chosen_visits});
                  if (chosen_second != kInvalidAction) {
                    auto hist_step2 = hist_step1;
                    hist_step2.push_back(chosen_graft);
                    cached_joint_followups_.push_back(
                        {hist_step2, chosen_second, chosen_q, chosen_visits});
                  }
                } else if (chosen_second != kInvalidAction) {
                  auto hist_step1 = base_hist;
                  hist_step1.push_back(chosen_first);
                  cached_joint_followups_.push_back(
                      {hist_step1, chosen_second, chosen_q, chosen_visits});
                }
              }
            } else {
              dune_search_session_internal::RunAdaptiveV2SingleRolloutSearch(
                  state, params, legal, raw_prior, &(impl_->tree),
                  ActiveRolloutEvaluators(), result);
            }
          } else {
            Action chosen_second_action = kInvalidAction;
            dune_search_session_internal::RunAdaptiveV1RolloutSearch(
                state, params, legal, raw_prior, &(impl_->tree),
                ActiveRolloutEvaluators(), impl_->evaluators.front().get(),
                result, chosen_second_action);
            bool is_select_card_step = !legal.empty() &&
                std::any_of(legal.begin(), legal.end(),
                            dune_imperium::IsSelectAgentCardAction);
            if (enable_macro_space_candidates_ && is_select_card_step &&
                chosen_second_action != kInvalidAction) {
              pending_macro_space_hint_ = chosen_second_action;
            }
          }
        }
      }
    }

    if (!ran_high_stakes) {
      dune_search_session_internal::FinalizeTreeReuseOrFallback(
          &(impl_->tree), legal, raw_action, raw_prior, search_v2_,
          live_deadline_starved, remaining_time_ms, available_after_reserve_ms,
          result);
    }
  }

  int covered = 0;
  int total_visits = 0;
  for (int v : result.diagnostics.visit_counts) {
    total_visits += v;
    if (v > 0) covered++;
  }
  result.diagnostics.total_root_visits = total_visits;
  result.diagnostics.num_covered_actions = covered;
  result.diagnostics.search_accepted = result.diagnostics.mcts_overrode_raw;

  if (kl_policy_tau_ > 0.0 && !result.used_fallback && total_visits > 0 &&
      !legal.empty()) {
    result.policy = dune_imperium::ComputeKlRegularizedSearchPolicy(
        raw_prior, legal, raw_action, result.diagnostics.visit_counts,
        result.diagnostics.q_values, kl_policy_tau_);
  }

  const auto t_end = std::chrono::steady_clock::now();
  double elapsed_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
  result.elapsed_time_ms = elapsed_ms;
  result.diagnostics.elapsed_search_time_ms = elapsed_ms;
  session_elapsed_time_ms_ += elapsed_ms;
  turn_elapsed_search_time_ms_ += elapsed_ms;
  session_new_simulations_completed_ += result.simulations_completed;
  dune_search_session_internal::UpdateAdaptiveRolloutRateEma(
      state, p, is_leader, remaining_time_ms, elapsed_ms, result, search_v2_,
      superhuman_mode_, ms_per_rollout_ema_, ms_per_rollout_by_round_);

  last_search_state_ = state.Clone();
  last_search_result_ = result;
  has_pending_commit_ = true;
  return result;
}

}  // namespace open_spiel
