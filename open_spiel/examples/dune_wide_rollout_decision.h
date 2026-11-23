#ifndef OPEN_SPIEL_EXAMPLES_DUNE_WIDE_ROLLOUT_DECISION_H_
#define OPEN_SPIEL_EXAMPLES_DUNE_WIDE_ROLLOUT_DECISION_H_

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <future>
#include <memory>
#include <numeric>
#include <random>
#include <utility>
#include <vector>

#include "open_spiel/examples/dune_compound_turn_search.h"
#include "open_spiel/examples/dune_seed_utils.h"
#include "open_spiel/examples/dune_wide_rollout_helpers.h"
#include "open_spiel/examples/dune_wide_rollout_types.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace dune_imperium {

inline CompoundSearchDecisionResult WideRolloutSearchDecision(
    const State& state, Player search_player, TurnSearchTree* turn_tree,
    const std::vector<std::shared_ptr<BatchedNNEvaluator>>& evaluators,
    int reference_rollouts_per_action, double min_override_margin,
    uint64_t decision_seed,
    SearchRolloutCrnMode crn_mode = SearchRolloutCrnMode::kSequential,
    double win_speed_tiebreak_eps = 0.0,
    bool ceiling_aware_margin = true) {
  CompoundSearchOptions opts;
  opts.win_speed_tiebreak_eps = win_speed_tiebreak_eps;
  opts.ceiling_aware_margin = ceiling_aware_margin;
  const auto legal = state.LegalActions();
  if (legal.size() <= 3) {
    return CompoundRolloutSearchDecision(
        state, search_player, turn_tree, evaluators, 3,
        reference_rollouts_per_action, min_override_margin,
        TreeReuseMode::kAdditive, decision_seed, SearchPairingMode::kUnpaired,
        SearchTruncationMode::kTerminal, crn_mode, nullptr, 0.0, opts);
  }
  const int top_k = std::min(6, static_cast<int>(legal.size()));
  const int rollouts_per_action = 3 * reference_rollouts_per_action / top_k;
  SPIEL_CHECK_GE(rollouts_per_action, 1);
  return CompoundRolloutSearchDecision(
      state, search_player, turn_tree, evaluators, top_k, rollouts_per_action,
      min_override_margin, TreeReuseMode::kAdditive, decision_seed,
      SearchPairingMode::kPaired, SearchTruncationMode::kTerminal, crn_mode,
      nullptr, 0.0, opts);
}

inline CompoundSearchDecisionResult PairedPriorSortedRolloutSearchDecision(
    const State& state,
    Player search_player,
    TurnSearchTree* turn_tree,
    const std::vector<std::shared_ptr<BatchedNNEvaluator>>& evaluators,
    const std::vector<Action>& ordered_candidates,
    int rollouts_per_action,
    double min_override_margin,
    double min_t_stat,
    uint64_t decision_seed,
    SearchRolloutCrnMode crn_mode = SearchRolloutCrnMode::kSequential,
    bool sample_opponent_leader_draft = false,
    double max_time_ms = -1.0,
    int max_inflight_worlds = 32,
    bool resample_from_infostate = true,
    bool use_bounded_pool_when_unclocked = false,
    double win_speed_tiebreak_eps = 0.0,
    bool ceiling_aware_margin = true) {
  if (turn_tree != nullptr) turn_tree->BindPublicCardKnowledge(state);
  if (ordered_candidates.empty()) {
    return {kInvalidAction, false, 0.0, 0.0, 0, 0, {}, {}};
  }
  if (ordered_candidates.size() == 1 || rollouts_per_action <= 0) {
    return {ordered_candidates.front(), false, 0.0, 0.0, 0, 0, {}, {}};
  }

  const Action raw_action = ordered_candidates.front();
  const auto* dune = dynamic_cast<const dune_imperium::DuneImperiumState*>(&state);

  int prior_rollouts_reused = 0;
  if (turn_tree != nullptr) {
    for (Action cand_a : ordered_candidates) {
      prior_rollouts_reused += turn_tree->GetVisits(cand_a);
    }
  }

  TurnSearchTree local_tree;
  TurnSearchTree* active_tree = (turn_tree != nullptr) ? turn_tree : &local_tree;
  const size_t K = ordered_candidates.size();

  std::vector<std::vector<double>> per_candidate_rollouts(K);
  for (size_t c_idx = 0; c_idx < K; ++c_idx) {
    per_candidate_rollouts[c_idx].reserve(rollouts_per_action);
  }

  int new_rollouts_executed = 0;

  const auto s1_t0 = std::chrono::steady_clock::now();
  if (max_time_ms <= 0.0 && !use_bounded_pool_when_unclocked) {
    std::vector<RolloutTask> tasks;
    tasks.reserve(K * rollouts_per_action);

    std::vector<std::unique_ptr<State>> base_worlds(rollouts_per_action);
    for (int r = 0; r < rollouts_per_action; ++r) {
      uint64_t wseed = dune_seed::DeriveSeed(decision_seed, r);
      if (dune && resample_from_infostate) {
        std::mt19937_64 wrng(wseed);
        auto rng_func = [&wrng]() {
          return std::generate_canonical<double, 53>(wrng);
        };
        base_worlds[r] = dune->ResampleFromInfostate(search_player, rng_func);
      }
      if (!base_worlds[r]) {
        base_worlds[r] = state.Clone();
      }
    }

    for (size_t c_idx = 0; c_idx < K; ++c_idx) {
      const Action cand_a = ordered_candidates[c_idx];
      const bool is_last_cand = (c_idx + 1 == K);
      for (int r = 0; r < rollouts_per_action; ++r) {
        uint64_t wseed = dune_seed::DeriveSeed(decision_seed, r);
        std::unique_ptr<State> sim_state =
            is_last_cand ? std::move(base_worlds[r]) : base_worlds[r]->Clone();
        sim_state->ApplyAction(cand_a);
        tasks.push_back({c_idx, r, cand_a, std::move(sim_state), wseed + 999});
      }
    }

    new_rollouts_executed = static_cast<int>(tasks.size());

    if (!tasks.empty()) {
      std::vector<std::future<RolloutTrajectory>> futures;
      futures.reserve(tasks.size());

      for (size_t i = 0; i < tasks.size(); ++i) {
        auto* active_evaluator = evaluators.empty() ? nullptr : evaluators[i % evaluators.size()].get();
        if (active_evaluator != nullptr) {
          active_evaluator->AdjustActiveRollouts(1);
        }
      }

      for (size_t i = 0; i < tasks.size(); ++i) {
        auto* s_ptr = tasks[i].sim_state.get();
        uint64_t s_seed = tasks[i].wseed;
        auto* active_evaluator = evaluators.empty() ? nullptr : evaluators[i % evaluators.size()].get();
        futures.push_back(SearchRolloutThreadPool::Instance().Submit(
            [s_ptr, search_player, active_evaluator, s_seed, crn_mode,
             sample_opponent_leader_draft, win_speed_tiebreak_eps]() {
              ActiveRolloutReleaseGuard guard{active_evaluator};
              return SimulateRolloutWithTurnTrajectory(
                  s_ptr, search_player, active_evaluator, s_seed, SearchTruncationMode::kTerminal,
                  crn_mode, sample_opponent_leader_draft, /*deadline=*/nullptr,
                  /*out_steps=*/nullptr, /*out_nn_evals=*/nullptr,
                  /*center_4seat_leaf_values=*/false, win_speed_tiebreak_eps);
            }));
      }

      for (size_t c_idx = 0; c_idx < K; ++c_idx) {
        per_candidate_rollouts[c_idx].resize(rollouts_per_action, 0.0);
      }
      for (size_t i = 0; i < futures.size(); ++i) {
        RolloutTrajectory traj = futures[i].get();
        active_tree->RecordRollout(tasks[i].action, traj.intra_turn_actions, traj.terminal_utility);
        if (tasks[i].c_idx < per_candidate_rollouts.size() &&
            tasks[i].rollout_idx >= 0 &&
            static_cast<size_t>(tasks[i].rollout_idx) < per_candidate_rollouts[tasks[i].c_idx].size()) {
          per_candidate_rollouts[tasks[i].c_idx][tasks[i].rollout_idx] = traj.terminal_utility;
        }
      }
    }
  } else {
    RunPairedDecisionClockedWaves(
        state, dune, search_player, evaluators, ordered_candidates,
        rollouts_per_action, decision_seed, crn_mode,
        sample_opponent_leader_draft, max_time_ms, max_inflight_worlds,
        resample_from_infostate, win_speed_tiebreak_eps, s1_t0, active_tree,
        per_candidate_rollouts, new_rollouts_executed);
  }
  const auto s1_t1 = std::chrono::steady_clock::now();
  const double stage1_elapsed_ms =
      std::chrono::duration<double, std::milli>(s1_t1 - s1_t0).count();

  const int r_completed = per_candidate_rollouts.empty() ? 0 : static_cast<int>(per_candidate_rollouts[0].size());

  double raw_mean_u = 0.0;
  if (r_completed > 0) {
    for (int r = 0; r < r_completed; ++r) {
      raw_mean_u += per_candidate_rollouts[0][r];
    }
    raw_mean_u /= r_completed;
  }

  Action best_action = raw_action;
  double best_mean_u = raw_mean_u;
  bool overridden = false;
  const double eff_override_margin = ComputeEffectiveOverrideMargin(
      min_override_margin, raw_mean_u, win_speed_tiebreak_eps,
      ceiling_aware_margin);

  for (size_t c_idx = 1; c_idx < K; ++c_idx) {
    Action cand_a = ordered_candidates[c_idx];
    double sum_u = 0.0;
    for (int r = 0; r < r_completed; ++r) {
      sum_u += per_candidate_rollouts[c_idx][r];
    }
    double mean_u = (r_completed > 0) ? (sum_u / r_completed) : 0.0;
    const PairedEbDiffStats eb = (r_completed >= 2)
        ? ComputePairedEbDiffStats(
              per_candidate_rollouts[c_idx].data(),
              per_candidate_rollouts[0].data(),
              r_completed)
        : PairedEbDiffStats{};
    const double mean_d = mean_u - raw_mean_u;
    const double t_stat = (r_completed >= 2) ? eb.t_stat : 0.0;

    if (mean_d > eff_override_margin &&
        t_stat >= min_t_stat &&
        PassesSingleFlipGuard(r_completed, min_t_stat, eb.positive_diff_count) &&
        mean_u > best_mean_u) {
      best_mean_u = mean_u;
      best_action = cand_a;
      overridden = true;
    }
  }

  std::vector<double> candidate_means(K, 0.0);
  for (size_t c_idx = 0; c_idx < K; ++c_idx) {
    double sum = 0.0;
    for (int r = 0; r < r_completed; ++r) {
      sum += per_candidate_rollouts[c_idx][r];
    }
    candidate_means[c_idx] = (r_completed > 0) ? (sum / r_completed) : 0.0;
  }

  return BuildCompoundSearchResult(
      best_action, overridden, raw_mean_u, best_mean_u,
      prior_rollouts_reused, new_rollouts_executed,
      ordered_candidates, candidate_means, per_candidate_rollouts,
      std::max(rollouts_per_action, r_completed), r_completed,
      /*stage2_planned_worlds=*/0, /*stage2_kept_worlds=*/0,
      /*stage1_rollouts_executed=*/0, /*stage2_rollouts_executed=*/0,
      stage1_elapsed_ms, /*stage2_elapsed_ms=*/0.0,
      (r_completed < 2) ? "stage1_starved" : (overridden ? "override" : "gate_failed"));
}

inline CompoundSearchDecisionResult LeaderPickRolloutSearchDecision(
    const State& state,
    Player search_player,
    const std::vector<std::shared_ptr<BatchedNNEvaluator>>& evaluators,
    int rollouts_per_leader,
    double min_override_margin,
    uint64_t decision_seed,
    SearchRolloutCrnMode crn_mode = SearchRolloutCrnMode::kDeckOrderCrn,
    bool sample_opponent_leader_draft = true,
    double max_time_ms = -1.0,
    int max_inflight_worlds = 32,
    const ActionsAndProbs* precomputed_raw_prior = nullptr,
    double win_speed_tiebreak_eps = 0.0,
    bool ceiling_aware_margin = true) {
  const auto legal = state.LegalActions();
  if (legal.empty()) return {kInvalidAction, false, 0.0, 0.0, 0, 0, {}, {}};
  if (legal.size() == 1) return {legal.front(), false, 0.0, 0.0, 0, 0, {}, {}};
  if (evaluators.empty()) {
    return {legal.front(), false, 0.0, 0.0, 0, 0, {}, {}};
  }

  const ActionsAndProbs raw_prior =
      (precomputed_raw_prior != nullptr && !precomputed_raw_prior->empty())
          ? *precomputed_raw_prior
          : evaluators.front()->Prior(state);
  const auto sorted_candidates = GetSortedCandidatesByPrior(raw_prior, legal);
  std::vector<Action> ordered_candidates;
  ordered_candidates.reserve(sorted_candidates.size());
  for (const auto& item : sorted_candidates) {
    ordered_candidates.push_back(item.first);
  }

  return PairedPriorSortedRolloutSearchDecision(
      state, search_player, /*turn_tree=*/nullptr, evaluators,
      ordered_candidates, rollouts_per_leader, min_override_margin,
      /*min_t_stat=*/1.0, decision_seed, crn_mode, sample_opponent_leader_draft,
      max_time_ms, max_inflight_worlds, /*resample_from_infostate=*/true,
      /*use_bounded_pool_when_unclocked=*/false, win_speed_tiebreak_eps,
      ceiling_aware_margin);
}

inline CompoundSearchDecisionResult AdaptiveRolloutSearchDecision(
    const State& state,
    Player search_player,
    TurnSearchTree* turn_tree,
    const std::vector<std::shared_ptr<BatchedNNEvaluator>>& evaluators,
    int reference_rollouts_per_action,
    double min_override_margin,
    uint64_t decision_seed,
    SearchRolloutCrnMode crn_mode = SearchRolloutCrnMode::kSequential,
    double max_time_ms = -1.0,
    int max_inflight_worlds = 32,
    const ActionsAndProbs* precomputed_raw_prior = nullptr,
    double win_speed_tiebreak_eps = 0.0,
    bool ceiling_aware_margin = true) {
  const auto legal = state.LegalActions();
  if (legal.empty()) return {kInvalidAction, false, 0.0, 0.0, 0, 0, {}, {}};
  if (legal.size() == 1) return {legal.front(), false, 0.0, 0.0, 0, 0, {}, {}};
  if (evaluators.empty()) {
    return {legal.front(), false, 0.0, 0.0, 0, 0, {}, {}};
  }

  const ActionsAndProbs raw_prior =
      (precomputed_raw_prior != nullptr && !precomputed_raw_prior->empty())
          ? *precomputed_raw_prior
          : evaluators.front()->Prior(state);
  const auto sorted_by_prior = GetSortedCandidatesByPrior(raw_prior, legal);
  if (sorted_by_prior.size() <= 1) {
    Action raw_action = sorted_by_prior.empty() ? legal.front() : sorted_by_prior.front().first;
    return {raw_action, false, 0.0, 0.0, 0, 0, {}, {}};
  }

  std::vector<Action> ordered_candidates;
  ordered_candidates.reserve(4);
  ordered_candidates.push_back(sorted_by_prior[0].first);
  ordered_candidates.push_back(sorted_by_prior[1].first);

  if (sorted_by_prior.size() >= 3 && sorted_by_prior[2].second >= 0.025) {
    ordered_candidates.push_back(sorted_by_prior[2].first);
    if (sorted_by_prior.size() >= 4 && sorted_by_prior[3].second >= 0.08) {
      ordered_candidates.push_back(sorted_by_prior[3].first);
    }
  }

  const int rollouts_per_action =
      (3 * reference_rollouts_per_action) / static_cast<int>(ordered_candidates.size());
  SPIEL_CHECK_GE(rollouts_per_action, 1);

  return PairedPriorSortedRolloutSearchDecision(
      state, search_player, turn_tree, evaluators, ordered_candidates,
      rollouts_per_action, min_override_margin, /*min_t_stat=*/1.0, decision_seed,
      crn_mode, /*sample_opponent_leader_draft=*/false, max_time_ms, max_inflight_worlds,
      /*resample_from_infostate=*/true, /*use_bounded_pool_when_unclocked=*/false,
      win_speed_tiebreak_eps, ceiling_aware_margin);
}

inline CompoundSearchDecisionResult SuccessiveHalvingRolloutSearchDecision(
    const State& state,
    Player search_player,
    TurnSearchTree* turn_tree,
    const std::vector<std::shared_ptr<BatchedNNEvaluator>>& evaluators,
    int reference_rollouts_per_action,
    double min_override_margin,
    uint64_t decision_seed,
    SearchRolloutCrnMode crn_mode = SearchRolloutCrnMode::kSequential,
    bool unbiased_stage2_gate = false,
    double max_time_ms = -1.0,
    int fixed_top_k = 0,
    int max_inflight_worlds = 32,
    bool use_bounded_pool_when_unclocked = false,
    bool full_clock_stage2 = false,
    double prior_log_ratio_weight = 0.0,
    double stage1_high_conf_prior_skip = 0.0,
    const ActionsAndProbs* precomputed_raw_prior = nullptr,
    bool all_buy_candidates = false,
    double win_speed_tiebreak_eps = 0.0,
    bool ceiling_aware_margin = true,
    const MacroSpaceConfig& macro_cfg = {},
    Action forced_include_action = kInvalidAction,
    BatchedNNEvaluator* prior_eval_override = nullptr,
    const EndgamePly2Config* ply2_cfg = nullptr,
    const IntrigueBeliefConfig* belief_cfg = nullptr,
    bool rescue_optional_abilities = false) {
  if (turn_tree != nullptr) turn_tree->BindPublicCardKnowledge(state);
  if (fixed_top_k == 1) {
    SpielFatalError("SuccessiveHalvingRolloutSearchDecision: fixed_top_k cannot be 1 (requires at least 2 candidates).");
  }
  const auto legal = state.LegalActions();
  if (legal.empty()) return {kInvalidAction, false, 0.0, 0.0, 0, 0, {}, {}};
  if (legal.size() == 1) return {legal.front(), false, 0.0, 0.0, 0, 0, {}, {}};
  if (evaluators.empty()) {
    return {legal.front(), false, 0.0, 0.0, 0, 0, {}, {}};
  }

  const ActionsAndProbs raw_prior =
      (precomputed_raw_prior != nullptr && !precomputed_raw_prior->empty())
          ? *precomputed_raw_prior
          : evaluators.front()->Prior(state);
  const auto sorted_by_prior = GetSortedCandidatesByPrior(raw_prior, legal);
  if (sorted_by_prior.size() <= 1) {
    Action raw_action = sorted_by_prior.empty() ? legal.front() : sorted_by_prior.front().first;
    return {raw_action, false, 0.0, 0.0, 0, 0, {}, {}};
  }

  auto [ordered_candidates, candidate_priors] =
      SelectOrderedCandidatesAndPriorsFromSorted(
          sorted_by_prior, fixed_top_k,
          all_buy_candidates && IsBuyDecisionState(state));

  if (rescue_optional_abilities && fixed_top_k > 0) {
    const bool is_buy = IsBuyDecisionState(state);
    int rescued_count = 0;
    for (const auto& [act, prob] : sorted_by_prior) {
      if (rescued_count >= 3) break;
      if (IsRescuableOptionalAbilityAction(act, is_buy) &&
          std::find(ordered_candidates.begin(), ordered_candidates.end(), act) ==
              ordered_candidates.end()) {
        ordered_candidates.push_back(act);
        candidate_priors.push_back(prob);
        ++rescued_count;
      }
    }
  }

  if (forced_include_action != kInvalidAction &&
      std::find(legal.begin(), legal.end(), forced_include_action) != legal.end()) {
    if (std::find(ordered_candidates.begin(), ordered_candidates.end(), forced_include_action) ==
        ordered_candidates.end()) {
      double f_prob = 0.0;
      for (const auto& [act, prob] : raw_prior) {
        if (act == forced_include_action) {
          f_prob = prob;
          break;
        }
      }
      ordered_candidates.push_back(forced_include_action);
      candidate_priors.push_back(f_prob);
    }
    if (forced_include_action != ordered_candidates[0]) {
      stage1_high_conf_prior_skip = 0.0;
    }
  }

  const size_t K_base = ordered_candidates.size();
  std::vector<Action> graft_actions(K_base, kInvalidAction);
  std::vector<Action> second_actions(K_base, kInvalidAction);

  BatchedNNEvaluator* prior_eval = (prior_eval_override != nullptr)
      ? prior_eval_override
      : (evaluators.empty() ? nullptr : evaluators.front().get());

  const auto* dune = dynamic_cast<const DuneImperiumState*>(&state);
  ExpandMacroSpaceCandidates(
      state, dune, search_player, prior_eval, macro_cfg,
      ordered_candidates, candidate_priors, graft_actions, second_actions);

  const size_t K = ordered_candidates.size();
  const Action raw_action = ordered_candidates[0];

  std::vector<double> prior_penalty(K, 0.0);
  for (size_t c_idx = 1; c_idx < K; ++c_idx) {
    if (prior_log_ratio_weight > 0.0 && !candidate_priors.empty()) {
      double log_ratio = std::log((candidate_priors[0] + 1e-6) / (candidate_priors[c_idx] + 1e-6));
      prior_penalty[c_idx] = prior_log_ratio_weight * std::clamp(log_ratio, 0.0, 4.0);
    }
  }

  int prior_rollouts_reused = 0;
  if (turn_tree != nullptr) {
    for (Action cand_a : ordered_candidates) {
      prior_rollouts_reused += turn_tree->GetVisits(cand_a);
    }
  }

  TurnSearchTree local_tree;
  TurnSearchTree* active_tree = (turn_tree != nullptr) ? turn_tree : &local_tree;

  const int total_budget = 3 * reference_rollouts_per_action;
  const int stage1_budget = total_budget / 2;
  const int r_stage1 = std::max(2, stage1_budget / static_cast<int>(K));

  std::vector<std::vector<double>> per_candidate_rollouts(K);
  for (size_t c_idx = 0; c_idx < K; ++c_idx) {
    per_candidate_rollouts[c_idx].reserve(r_stage1);
  }

  int new_rollouts_executed = 0;
  const bool has_deadline = (max_time_ms > 0.0);
  const bool use_bounded_stage = has_deadline || use_bounded_pool_when_unclocked;
  const auto start_time = std::chrono::steady_clock::now();
  const double s1_deadline_frac = full_clock_stage2
      ? ((has_deadline && max_time_ms <= 2500.0) ? 0.62 : 0.46)
      : 0.50;
  const double final_deadline_frac = 1.00;
  const auto stage1_deadline = has_deadline
      ? (start_time +
         std::chrono::duration_cast<std::chrono::steady_clock::duration>(
             std::chrono::duration<double, std::milli>(s1_deadline_frac * max_time_ms)))
      : (use_bounded_pool_when_unclocked
             ? std::chrono::steady_clock::time_point::max()
             : std::chrono::steady_clock::time_point{});
  const auto final_deadline = has_deadline
      ? (start_time +
         std::chrono::duration_cast<std::chrono::steady_clock::duration>(
             std::chrono::duration<double, std::milli>(final_deadline_frac * max_time_ms)))
      : (use_bounded_pool_when_unclocked
             ? std::chrono::steady_clock::time_point::max()
             : std::chrono::steady_clock::time_point{});

  const EndgamePly2Config* eff_ply2_s1 = nullptr;
  const EndgamePly2Config* eff_ply2_s2 = nullptr;
  if (ply2_cfg != nullptr && ply2_cfg->enabled && dune != nullptr) {
    int max_vp = 0;
    for (int i = 0; i < 4; ++i) {
      max_vp = std::max(max_vp, dune->GetPlayerVp(i));
    }
    if (dune->GetCurrentRound() >= ply2_cfg->min_round || max_vp >= ply2_cfg->min_table_vp) {
      if (!ply2_cfg->stage2_only && (!has_deadline || max_time_ms >= 5000.0)) {
        eff_ply2_s1 = ply2_cfg;
      }
      if (!has_deadline || max_time_ms >= 2500.0) {
        eff_ply2_s2 = ply2_cfg;
      }
    }
  }

  int stage1_planned_worlds = r_stage1;
  int stage1_kept_worlds = 0;
  int s1_waves_completed = 0;
  const auto s1_t0 = std::chrono::steady_clock::now();

  // --- STAGE 1: Broad Screen ---
  if (!use_bounded_stage) {
    std::vector<size_t> candidate_indices(K);
    std::iota(candidate_indices.begin(), candidate_indices.end(), 0);
    RunUnboundedPairedRolloutTasks(
        state, dune, search_player, evaluators, ordered_candidates,
        candidate_indices, r_stage1, /*world_seed_offset=*/0, decision_seed,
        crn_mode, active_tree, per_candidate_rollouts, new_rollouts_executed,
        second_actions, graft_actions, win_speed_tiebreak_eps,
        eff_ply2_s1, belief_cfg);
  } else {
    s1_waves_completed = RunSHStage1ClockedWaves(
        state, dune, search_player, evaluators, ordered_candidates,
        r_stage1, decision_seed, crn_mode, has_deadline, full_clock_stage2,
        max_time_ms, s1_deadline_frac, start_time, stage1_deadline,
        final_deadline, s1_t0, max_inflight_worlds, active_tree,
        per_candidate_rollouts, new_rollouts_executed, second_actions,
        graft_actions, win_speed_tiebreak_eps, eff_ply2_s1, belief_cfg);
  }

  const auto s1_t1 = std::chrono::steady_clock::now();
  const double stage1_elapsed_ms =
      std::chrono::duration<double, std::milli>(s1_t1 - s1_t0).count();
  const int stage1_rollouts_executed = new_rollouts_executed;

  const int r_stage1_done = per_candidate_rollouts.empty() ? 0 : static_cast<int>(per_candidate_rollouts[0].size());
  stage1_kept_worlds = r_stage1_done;

  if (r_stage1_done < 2) {
    double raw_mean = (r_stage1_done > 0) ? per_candidate_rollouts[0][0] : 0.0;
    std::vector<double> candidate_means(K, 0.0);
    if (r_stage1_done > 0) {
      for (size_t c_idx = 0; c_idx < K; ++c_idx) {
        candidate_means[c_idx] = per_candidate_rollouts[c_idx][0];
      }
    }
    return BuildCompoundSearchResult(
        raw_action, false, raw_mean, raw_mean,
        prior_rollouts_reused, new_rollouts_executed,
        ordered_candidates, candidate_means, per_candidate_rollouts,
        stage1_planned_worlds, stage1_kept_worlds,
        /*stage2_planned_worlds=*/0, /*stage2_kept_worlds=*/0,
        stage1_rollouts_executed, /*stage2_rollouts_executed=*/0,
        stage1_elapsed_ms, /*stage2_elapsed_ms=*/0.0,
        "stage1_starved", graft_actions[0], second_actions[0]);
  }

  double raw_mean_1 = 0.0;
  for (int r = 0; r < r_stage1_done; ++r) {
    raw_mean_1 += per_candidate_rollouts[0][r];
  }
  raw_mean_1 /= r_stage1_done;

  std::vector<double> mean_1(K, 0.0);
  std::vector<double> mean_d_1(K, 0.0);
  mean_1[0] = raw_mean_1;
  mean_d_1[0] = 0.0;

  size_t best_c_idx = 1;
  double highest_penalized_adv_1 = -1e9;

  for (size_t c_idx = 1; c_idx < K; ++c_idx) {
    double sum_u = 0.0;
    for (int r = 0; r < r_stage1_done; ++r) {
      sum_u += per_candidate_rollouts[c_idx][r];
    }
    mean_1[c_idx] = sum_u / r_stage1_done;
    mean_d_1[c_idx] = mean_1[c_idx] - raw_mean_1;
    const double penalized_adv_1 = mean_d_1[c_idx] - prior_penalty[c_idx];
    if (penalized_adv_1 > highest_penalized_adv_1) {
      highest_penalized_adv_1 = penalized_adv_1;
      best_c_idx = c_idx;
    }
  }

  // Stage 1 Early-Stop Gate: If no challenger beat raw_action in Stage 1, stop early.
  const double best_challenger_adv_1 = mean_d_1[best_c_idx] - prior_penalty[best_c_idx];
  const bool high_conf_skip = (stage1_high_conf_prior_skip > 0.0 &&
                               !candidate_priors.empty() &&
                               candidate_priors[0] >= stage1_high_conf_prior_skip &&
                               best_challenger_adv_1 <= 0.05);
  const bool standard_early_stop = (!full_clock_stage2 && best_challenger_adv_1 <= 0.0);
  const double est_s2_one_wave_ms =
      (s1_waves_completed > 0)
          ? (0.60 * (stage1_elapsed_ms / static_cast<double>(s1_waves_completed)))
          : 0.0;
  const bool tight_clock_futile_stop =
      (full_clock_stage2 && has_deadline && max_time_ms < 4800.0 &&
       ((r_stage1_done >= 16 && best_challenger_adv_1 <= -0.10) ||
        (stage1_elapsed_ms >= 0.68 * max_time_ms && best_challenger_adv_1 <= 0.0) ||
        (est_s2_one_wave_ms > 0.0 &&
         (max_time_ms - stage1_elapsed_ms) < est_s2_one_wave_ms)));

  if (standard_early_stop || high_conf_skip || tight_clock_futile_stop) {
    return BuildCompoundSearchResult(
        raw_action, false, raw_mean_1, raw_mean_1,
        prior_rollouts_reused, new_rollouts_executed,
        ordered_candidates, mean_1, per_candidate_rollouts,
        stage1_planned_worlds, stage1_kept_worlds,
        /*stage2_planned_worlds=*/0, /*stage2_kept_worlds=*/0,
        stage1_rollouts_executed, /*stage2_rollouts_executed=*/0,
        stage1_elapsed_ms, /*stage2_elapsed_ms=*/0.0,
        high_conf_skip ? "stage1_high_conf_skip" : "early_stop",
        graft_actions[0], second_actions[0]);
  }

  // --- STAGE 2: Deep Finale ---
  const int stage2_budget = std::max(2, total_budget - static_cast<int>(K) * r_stage1);
  int r_stage2 = std::max(2, stage2_budget / 2);
  if (full_clock_stage2 && has_deadline && stage1_elapsed_ms > 0.0) {
    const double s1_ms_per_sim =
        stage1_elapsed_ms / std::max(1.0, static_cast<double>(K * r_stage1_done));
    const double rem_stage2_ms = std::max(0.0, 0.968 * max_time_ms - stage1_elapsed_ms);
    const double est_s2_sims = rem_stage2_ms / std::max(0.1, s1_ms_per_sim);
    r_stage2 = std::max(2, static_cast<int>(est_s2_sims / 2.0));
  }
  const std::array<size_t, 2> finalists = {0, best_c_idx};

  int stage2_planned_worlds = r_stage2;
  int stage2_kept_worlds = 0;
  const auto s2_t0 = std::chrono::steady_clock::now();

  if (!use_bounded_stage) {
    std::vector<size_t> stage2_cands(finalists.begin(), finalists.end());
    RunUnboundedPairedRolloutTasks(
        state, dune, search_player, evaluators, ordered_candidates,
        stage2_cands, r_stage2, /*world_seed_offset=*/r_stage1, decision_seed,
        crn_mode, active_tree, per_candidate_rollouts, new_rollouts_executed,
        second_actions, graft_actions, win_speed_tiebreak_eps,
        eff_ply2_s2, belief_cfg);
  } else {
    RunSHStage2ClockedWaves(
        state, dune, search_player, evaluators, ordered_candidates,
        best_c_idx, r_stage1, r_stage1_done, r_stage2, stage1_elapsed_ms,
        raw_mean_1, min_override_margin, prior_penalty[best_c_idx],
        decision_seed, crn_mode, has_deadline, full_clock_stage2, max_time_ms,
        start_time, final_deadline, max_inflight_worlds, active_tree,
        per_candidate_rollouts, new_rollouts_executed, stage2_planned_worlds,
        second_actions, graft_actions, win_speed_tiebreak_eps,
        ceiling_aware_margin, eff_ply2_s2, belief_cfg);
  }

  const auto s2_t1 = std::chrono::steady_clock::now();
  const double stage2_elapsed_ms =
      std::chrono::duration<double, std::milli>(s2_t1 - s2_t0).count();
  const int stage2_rollouts_executed = new_rollouts_executed - stage1_rollouts_executed;

  const int r_stage2_done = static_cast<int>(per_candidate_rollouts[0].size()) - r_stage1_done;
  stage2_kept_worlds = r_stage2_done;

  if (r_stage2_done < 2) {
    const int R_tot = static_cast<int>(per_candidate_rollouts[0].size());
    double raw_sum = 0.0;
    for (int r = 0; r < R_tot; ++r) {
      raw_sum += per_candidate_rollouts[0][r];
    }
    const double raw_mean_tot = (R_tot > 0) ? (raw_sum / R_tot) : 0.0;

    std::vector<double> candidate_means(K, 0.0);
    for (size_t c_idx = 0; c_idx < K; ++c_idx) {
      if (c_idx == 0) {
        candidate_means[c_idx] = raw_mean_tot;
      } else if (c_idx == best_c_idx && !per_candidate_rollouts[best_c_idx].empty()) {
        double b_sum = 0.0;
        for (double v : per_candidate_rollouts[best_c_idx]) b_sum += v;
        candidate_means[c_idx] = b_sum / per_candidate_rollouts[best_c_idx].size();
      } else {
        candidate_means[c_idx] = mean_1[c_idx];
      }
    }
    return BuildCompoundSearchResult(
        raw_action, false, raw_mean_tot, raw_mean_tot,
        prior_rollouts_reused, new_rollouts_executed,
        ordered_candidates, candidate_means, per_candidate_rollouts,
        stage1_planned_worlds, stage1_kept_worlds,
        stage2_planned_worlds, stage2_kept_worlds,
        stage1_rollouts_executed, stage2_rollouts_executed,
        stage1_elapsed_ms, stage2_elapsed_ms,
        "stage2_starved", graft_actions[0], second_actions[0]);
  }

  const int R_total = r_stage1_done + r_stage2_done;

  // Stage 2 paired difference with Empirical-Bayes variance floor:
  const PairedEbDiffStats s2_stats = ComputePairedEbDiffStats(
      per_candidate_rollouts[best_c_idx].data() + r_stage1_done,
      per_candidate_rollouts[0].data() + r_stage1_done,
      r_stage2_done);
  const double mean_d_stage2 = s2_stats.mean_d;
  const double se_d_stage2 = s2_stats.se_eb;
  const double t_stat_stage2 = s2_stats.t_stat;

  // Total statistics across all R_total rollouts:
  double sum_raw_total = 0.0;
  double sum_best_total = 0.0;
  for (int r = 0; r < R_total; ++r) {
    sum_raw_total += per_candidate_rollouts[0][r];
    sum_best_total += per_candidate_rollouts[best_c_idx][r];
  }
  const double raw_mean_total = sum_raw_total / R_total;
  const double best_cand_mean_total = sum_best_total / R_total;
  const PairedEbDiffStats tot_stats = ComputePairedEbDiffStats(
      per_candidate_rollouts[best_c_idx].data(),
      per_candidate_rollouts[0].data(),
      R_total);
  const double mean_d_total = tot_stats.mean_d;
  const double t_stat_total = tot_stats.t_stat;

  // Two-Stage Confirmation Override Gate:
  Action best_action = raw_action;
  double best_mean_u = raw_mean_total;
  bool overridden = false;

  const double base_eff_margin = ComputeEffectiveOverrideMargin(
      min_override_margin, raw_mean_total, win_speed_tiebreak_eps,
      ceiling_aware_margin);
  const double effective_margin = base_eff_margin + prior_penalty[best_c_idx];
  bool should_override = false;
  if (!unbiased_stage2_gate) {
    should_override =
        (mean_d_stage2 > 0.0 &&
         mean_d_total > effective_margin &&
         t_stat_total >= 1.25 &&
         PassesSingleFlipGuard(R_total, 1.25, tot_stats.positive_diff_count));
  } else {
    const int stage2_min_confirm =
        std::min<int>(8, std::max(2, stage2_planned_worlds));
    if (K >= 2 && r_stage2_done >= stage2_min_confirm) {
      should_override =
          (mean_d_stage2 >= effective_margin &&
           t_stat_stage2 >= 1.25 &&
           PassesSingleFlipGuard(r_stage2_done, 1.25, s2_stats.positive_diff_count));
    } else {
      should_override =
          (mean_d_stage2 > 0.0 &&
           mean_d_total >= effective_margin &&
           t_stat_total >= 1.25 &&
           PassesSingleFlipGuard(R_total, 1.25, tot_stats.positive_diff_count));
    }
  }

  Action chosen_graft = graft_actions[0];
  Action chosen_second = second_actions[0];
  if (should_override) {
    best_action = ordered_candidates[best_c_idx];
    best_mean_u = best_cand_mean_total;
    overridden = true;
    chosen_graft = graft_actions[best_c_idx];
    chosen_second = second_actions[best_c_idx];
  }

  std::vector<double> candidate_means(K, 0.0);
  for (size_t c_idx = 0; c_idx < K; ++c_idx) {
    if (c_idx == 0) {
      candidate_means[c_idx] = raw_mean_total;
    } else if (c_idx == best_c_idx) {
      candidate_means[c_idx] = best_cand_mean_total;
    } else {
      candidate_means[c_idx] = mean_1[c_idx];
    }
  }

  return BuildCompoundSearchResult(
      best_action, overridden, raw_mean_total, best_mean_u,
      prior_rollouts_reused, new_rollouts_executed,
      ordered_candidates, candidate_means, per_candidate_rollouts,
      stage1_planned_worlds, stage1_kept_worlds,
      stage2_planned_worlds, stage2_kept_worlds,
      stage1_rollouts_executed, stage2_rollouts_executed,
      stage1_elapsed_ms, stage2_elapsed_ms,
      overridden ? "override" : "stage2_no_override",
      chosen_graft, chosen_second, mean_d_stage2, se_d_stage2);
}

}  // namespace dune_imperium
}  // namespace open_spiel

#endif  // OPEN_SPIEL_EXAMPLES_DUNE_WIDE_ROLLOUT_DECISION_H_
