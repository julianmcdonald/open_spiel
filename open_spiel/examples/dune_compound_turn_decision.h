// Copyright 2026 DeepMind Technologies Limited.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef OPEN_SPIEL_EXAMPLES_DUNE_COMPOUND_TURN_DECISION_H_
#define OPEN_SPIEL_EXAMPLES_DUNE_COMPOUND_TURN_DECISION_H_

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <memory>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include "open_spiel/examples/dune_compound_turn_decision_helpers.h"

namespace open_spiel {
namespace dune_imperium {

// Tree reuse strategies
enum class TreeReuseMode {
  kTargetFill,  // Top up candidate visits to reach target (maximizes GPU efficiency)
  kAdditive     // Always run target rollouts and add to existing tree visits (deepens search)
};

// Optional search configuration options for CompoundRolloutSearchDecision
struct CompoundSearchOptions {
  const ActionsAndProbs* precomputed_raw_prior = nullptr;
  bool dedup_paired_worlds = false;
  int stage1_rollouts = 0;
  double stage1_min_mean_diff = -1e9;
  // Step 8: 4-seat leaf value centering (default OFF)
  bool center_4seat_leaf_values = false;
  // Phase 3 Cat 3: all buy candidates (default OFF)
  bool all_buy_candidates = false;
  // Step 4: 4+4 stage-1 subwave pruning (default OFF)
  int stage1_subwave_rollouts = 0;
  double stage1_subwave_min_diff = -0.05;
  // Phase 3 Q4a: Symmetric Multi-Candidate Stage-2 Confirmation (default OFF)
  bool symmetric_stage2_confirmation = false;
  double confirmation_max_raw_prior = 0.85;
  double confirmation_rival_cum_mass = 0.90;
  int confirmation_max_rivals = 3;
  int stage2_independent_rollouts = 0;
  // Terminal win-speed tie-breaker and ceiling-aware margin near +2.25
  double win_speed_tiebreak_eps = 0.0;
  bool ceiling_aware_margin = true;
};

using CompoundSearchConfig = CompoundSearchOptions;

// Decision result from CompoundRolloutSearch
struct CompoundSearchDecisionResult {
  Action action = kInvalidAction;
  bool overridden = false;
  double raw_mean_utility = 0.0;
  double chosen_mean_utility = 0.0;
  int prior_rollouts_reused = 0;
  int new_rollouts_executed = 0;
  std::vector<std::pair<Action, double>> candidate_utilities = {};
  std::vector<std::pair<Action, std::vector<double>>> candidate_rollout_utilities = {};
  int stage1_planned_worlds = 0;
  int stage1_kept_worlds = 0;
  int stage2_planned_worlds = 0;
  int stage2_kept_worlds = 0;
  int stage1_rollouts_executed = 0;
  int stage2_rollouts_executed = 0;
  double stage1_elapsed_ms = 0.0;
  double stage2_elapsed_ms = 0.0;
  std::string termination_reason = "none";
  double mean_d_stage2 = 0.0;
  double se_d_stage2 = 0.0;
  // R7-55 hybrid search diagnostics
  Action raw_action = kInvalidAction;
  Action best_action = kInvalidAction;
  Action chosen_action = kInvalidAction;
  double best_vs_raw_mean_diff = 0.0;
  double best_vs_raw_t_stat = 0.0;
  int rollout_steps = 0;
  int rollout_nn_evals = 0;
  // R7-58 options and diagnostics
  std::vector<std::pair<Action, double>> candidate_priors = {};
  int resample_fallbacks = 0;
  bool stopped_stage1 = false;
  // Phase 3 Q4a confirmation and teacher execution
  bool confirmed = false;
  bool teacher_accepted = false;
  Action teacher_action = kInvalidAction;
  std::vector<std::pair<Action, double>> stage2_candidate_utilities = {};
  Action chosen_graft_action = kInvalidAction;
  Action chosen_second_action = kInvalidAction;
};

using CompoundRolloutSearchResult = CompoundSearchDecisionResult;

// Intra-turn compound rollout search decision with tree preservation
inline CompoundSearchDecisionResult CompoundRolloutSearchDecision(
    const State& state,
    Player search_player,
    TurnSearchTree* turn_tree,
    const std::vector<std::shared_ptr<BatchedNNEvaluator>>& evaluators,
    int top_k,
    int rollouts_per_action,
    double min_override_margin,
    TreeReuseMode reuse_mode,
    uint64_t decision_seed,
    SearchPairingMode pairing_mode = SearchPairingMode::kUnpaired,
    SearchTruncationMode truncation_mode = SearchTruncationMode::kTerminal,
    SearchRolloutCrnMode crn_mode = SearchRolloutCrnMode::kSequential,
    const SearchWorldSampler& world_sampler = nullptr,
    double min_paired_t_stat = 0.0,
    const CompoundSearchOptions& opts = {}) {
  if (turn_tree != nullptr) turn_tree->BindPublicCardKnowledge(state);
  std::vector<Action> legal_actions = state.LegalActions();
  if (legal_actions.empty()) return {kInvalidAction, false, 0.0, 0.0, 0, 0, {}, {}};
  if (legal_actions.size() == 1) {
    CompoundSearchDecisionResult r;
    r.action = legal_actions.front();
    r.chosen_action = r.action;
    r.raw_action = r.action;
    r.best_action = r.action;
    if (opts.precomputed_raw_prior != nullptr) {
      for (const auto& ap : *opts.precomputed_raw_prior) {
        if (ap.first == r.action) {
          r.candidate_priors.push_back(ap);
          break;
        }
      }
    }
    if (r.candidate_priors.empty()) {
      r.candidate_priors.push_back({r.action, 1.0});
    }
    return r;
  }

  int total_steps = 0;
  int total_nn_evals = 0;
  ActionsAndProbs raw_prior;
  if (opts.precomputed_raw_prior != nullptr) {
    raw_prior = *opts.precomputed_raw_prior;
  } else {
    if (!evaluators.empty()) ++total_nn_evals;
    raw_prior = evaluators.front()->Prior(state);
  }
  Action raw_action = PickGreedyAction(raw_prior, legal_actions);

  std::vector<Action> candidates = SelectCompoundSearchCandidates(
      raw_prior, legal_actions, raw_action, top_k, opts.all_buy_candidates,
      opts.symmetric_stage2_confirmation, opts.confirmation_rival_cum_mass,
      opts.confirmation_max_rivals);

  if (candidates.size() <= 1) {
    CompoundSearchDecisionResult r;
    r.action = raw_action;
    r.chosen_action = raw_action;
    r.raw_action = raw_action;
    r.best_action = raw_action;
    r.teacher_action = raw_action;
    r.teacher_accepted = false;
    r.confirmed = false;
    r.rollout_nn_evals = total_nn_evals;
    for (Action a : candidates) {
      double p = 0.0;
      for (const auto& ap : raw_prior) {
        if (ap.first == a) { p = ap.second; break; }
      }
      r.candidate_priors.push_back({a, p});
    }
    return r;
  }

  std::vector<std::pair<Action, double>> candidate_priors =
      AlignRawCandidateAndPriors(
          &candidates, raw_prior, raw_action, top_k,
          opts.all_buy_candidates, opts.symmetric_stage2_confirmation);

  const auto* dune = dynamic_cast<const dune_imperium::DuneImperiumState*>(&state);

  int prior_rollouts_reused = 0;
  for (size_t c_idx = 0; c_idx < candidates.size(); ++c_idx) {
    Action cand_a = candidates[c_idx];
    int existing_visits = (turn_tree != nullptr) ? turn_tree->GetVisits(cand_a) : 0;
    if (turn_tree != nullptr && reuse_mode == TreeReuseMode::kTargetFill) {
      prior_rollouts_reused += std::min(existing_visits, rollouts_per_action);
    } else if (turn_tree != nullptr) {
      prior_rollouts_reused += existing_visits;
    }
  }

  const bool use_two_stage =
      (opts.stage1_rollouts > 0 &&
       (opts.stage1_rollouts < rollouts_per_action || opts.stage2_independent_rollouts > 0) &&
       pairing_mode == SearchPairingMode::kPaired &&
       prior_rollouts_reused == 0);

  const bool can_dedup =
      (opts.dedup_paired_worlds &&
       pairing_mode == SearchPairingMode::kPaired &&
       !world_sampler &&
       dune != nullptr &&
       prior_rollouts_reused == 0);

  std::atomic<int> resample_fallbacks{0};
  int new_rollouts_executed = 0;

  TurnSearchTree local_tree;
  TurnSearchTree* active_tree = (turn_tree != nullptr) ? turn_tree : &local_tree;

  const int R1 = opts.stage1_rollouts;
  const int R2_indep = (opts.stage2_independent_rollouts > 0)
      ? opts.stage2_independent_rollouts
      : std::max(0, rollouts_per_action - R1);
  const int max_rollouts_capacity = std::max(rollouts_per_action, R1 + R2_indep);
  std::vector<std::vector<double>> per_candidate_rollouts(
      candidates.size(), std::vector<double>(max_rollouts_capacity, 0.0));
  std::vector<std::vector<double>> cand_rollout_returns(candidates.size());

  const State* root_state_ptr = &state;

  auto build_tasks = [&](const std::vector<size_t>& active_c_indices,
                         int r_begin, int r_end, bool use_dedup) {
    return BuildCompoundRolloutTasks(
        candidates, active_c_indices, r_begin, r_end, use_dedup, pairing_mode,
        decision_seed, search_player, root_state_ptr, dune, world_sampler,
        &resample_fallbacks);
  };

  auto execute_tasks = [&](std::vector<CompoundRolloutTask>& batch_tasks) {
    ExecuteCompoundRolloutTasks(
        batch_tasks, evaluators, root_state_ptr, dune, search_player,
        truncation_mode, crn_mode, opts.center_4seat_leaf_values,
        opts.win_speed_tiebreak_eps, active_tree, &resample_fallbacks,
        &new_rollouts_executed, &total_steps, &total_nn_evals,
        &per_candidate_rollouts, &cand_rollout_returns);
  };

  bool stopped_stage1 = false;
  size_t best_s1_c = 0;
  double best_s1_mean_d = -1e9;
  bool confirmed = false;
  bool teacher_accepted = false;
  Action teacher_action = raw_action;
  std::vector<std::pair<Action, double>> stage2_candidate_utilities;
  Action chosen_action = raw_action;
  Action best_action = raw_action;
  double best_mean_u = 0.0;
  size_t best_c_idx = 0;
  double best_vs_raw_mean_diff = 0.0;
  double best_vs_raw_t_stat = 0.0;
  bool overridden = false;
  double raw_mean_u = 0.0;

  if (use_two_stage) {
    const int W1 = opts.symmetric_stage2_confirmation ? 0 : opts.stage1_subwave_rollouts;
    if (W1 > 0 && W1 < R1) {
      std::vector<size_t> all_c_indices(candidates.size());
      std::iota(all_c_indices.begin(), all_c_indices.end(), 0);
      auto s1_tasks = build_tasks(all_c_indices, 0, W1, can_dedup);
      execute_tasks(s1_tasks);

      std::vector<size_t> surviving_challengers;
      size_t best_w1_c = 1;
      double best_w1_mean_d = -1e9;
      for (size_t c_idx = 1; c_idx < candidates.size(); ++c_idx) {
        double sum_d = 0.0;
        for (int r = 0; r < W1; ++r) {
          sum_d += (per_candidate_rollouts[c_idx][r] - per_candidate_rollouts[0][r]);
        }
        double mean_d = sum_d / static_cast<double>(W1);
        if (c_idx == 1 || mean_d > best_w1_mean_d) {
          best_w1_mean_d = mean_d;
          best_w1_c = c_idx;
        }
        if (mean_d >= opts.stage1_subwave_min_diff) {
          surviving_challengers.push_back(c_idx);
        }
      }

      if (surviving_challengers.empty()) {
        stopped_stage1 = true;
        best_s1_c = best_w1_c;
        best_s1_mean_d = best_w1_mean_d;
      } else {
        std::vector<size_t> subwave2_indices = {0};
        subwave2_indices.insert(subwave2_indices.end(),
                                surviving_challengers.begin(), surviving_challengers.end());
        auto sw2_tasks = build_tasks(subwave2_indices, W1, R1, can_dedup);
        execute_tasks(sw2_tasks);

        best_s1_c = surviving_challengers.front();
        best_s1_mean_d = -1e9;
        for (size_t c_idx : surviving_challengers) {
          double sum_d = 0.0;
          for (int r = 0; r < R1; ++r) {
            sum_d += (per_candidate_rollouts[c_idx][r] - per_candidate_rollouts[0][r]);
          }
          double mean_d = sum_d / static_cast<double>(R1);
          if (mean_d > best_s1_mean_d) {
            best_s1_mean_d = mean_d;
            best_s1_c = c_idx;
          }
        }
        if (best_s1_mean_d < opts.stage1_min_mean_diff) {
          stopped_stage1 = true;
        }
      }
    } else {
      std::vector<size_t> all_c_indices(candidates.size());
      std::iota(all_c_indices.begin(), all_c_indices.end(), 0);
      auto s1_tasks = build_tasks(all_c_indices, 0, R1, can_dedup);
      execute_tasks(s1_tasks);

      best_s1_c = 1;
      best_s1_mean_d = -1e9;
      for (size_t c_idx = 1; c_idx < candidates.size(); ++c_idx) {
        double sum_d = 0.0;
        for (int r = 0; r < R1; ++r) {
          sum_d += (per_candidate_rollouts[c_idx][r] - per_candidate_rollouts[0][r]);
        }
        double mean_d = sum_d / static_cast<double>(R1);
        if (c_idx == 1 || mean_d > best_s1_mean_d) {
          best_s1_mean_d = mean_d;
          best_s1_c = c_idx;
        }
      }

      if (best_s1_c == 0 || best_s1_mean_d < opts.stage1_min_mean_diff) {
        stopped_stage1 = true;
      }
    }

    if (opts.symmetric_stage2_confirmation) {
      SymmetricStage2Params s2_params;
      s2_params.r1 = R1;
      s2_params.r2_indep = R2_indep;
      s2_params.can_dedup = can_dedup;
      s2_params.stage1_min_mean_diff = opts.stage1_min_mean_diff;
      s2_params.min_override_margin = min_override_margin;
      s2_params.min_paired_t_stat = min_paired_t_stat;
      s2_params.confirmation_max_raw_prior = opts.confirmation_max_raw_prior;
      s2_params.confirmation_rival_cum_mass = opts.confirmation_rival_cum_mass;
      s2_params.confirmation_max_rivals = opts.confirmation_max_rivals;
      s2_params.win_speed_tiebreak_eps = opts.win_speed_tiebreak_eps;
      s2_params.ceiling_aware_margin = opts.ceiling_aware_margin;

      auto s2_out = EvaluateSymmetricStage2Confirmation(
          s2_params, candidates, raw_action, raw_prior, cand_rollout_returns,
          per_candidate_rollouts, build_tasks, execute_tasks);
      stopped_stage1 = s2_out.stopped_stage1;
      best_s1_c = s2_out.best_s1_c;
      best_s1_mean_d = s2_out.best_s1_mean_d;
      confirmed = s2_out.confirmed;
      teacher_accepted = s2_out.teacher_accepted;
      teacher_action = s2_out.teacher_action;
      stage2_candidate_utilities = std::move(s2_out.stage2_candidate_utilities);
      chosen_action = s2_out.chosen_action;
      best_action = s2_out.best_action;
      best_mean_u = s2_out.best_mean_u;
      best_c_idx = s2_out.best_c_idx;
      best_vs_raw_mean_diff = s2_out.best_vs_raw_mean_diff;
      best_vs_raw_t_stat = s2_out.best_vs_raw_t_stat;
      overridden = s2_out.overridden;
      raw_mean_u = s2_out.raw_mean_u;
    } else {
      if (stopped_stage1 || best_s1_c == 0 || best_s1_mean_d < opts.stage1_min_mean_diff) {
        stopped_stage1 = true;
      } else {
        std::vector<size_t> stage2_c_indices = {0, best_s1_c};
        auto s2_tasks = build_tasks(stage2_c_indices, R1, rollouts_per_action, can_dedup);
        execute_tasks(s2_tasks);
      }
    }

    for (size_t c_idx = 0; c_idx < candidates.size(); ++c_idx) {
      per_candidate_rollouts[c_idx].resize(cand_rollout_returns[c_idx].size());
    }
  } else if (prior_rollouts_reused == 0) {
    std::vector<size_t> all_c_indices(candidates.size());
    std::iota(all_c_indices.begin(), all_c_indices.end(), 0);
    auto tasks = build_tasks(all_c_indices, 0, rollouts_per_action, can_dedup);
    execute_tasks(tasks);
  } else {
    std::vector<CompoundRolloutTask> tasks;
    for (size_t c_idx = 0; c_idx < candidates.size(); ++c_idx) {
      Action cand_a = candidates[c_idx];
      int existing_visits = (turn_tree != nullptr) ? turn_tree->GetVisits(cand_a) : 0;
      int needed_rollouts = rollouts_per_action;
      if (turn_tree != nullptr && reuse_mode == TreeReuseMode::kTargetFill) {
        needed_rollouts = std::max(0, rollouts_per_action - existing_visits);
      }
      auto c_tasks = build_tasks({c_idx}, 0, needed_rollouts, /*use_dedup=*/false);
      tasks.insert(tasks.end(), std::make_move_iterator(c_tasks.begin()),
                   std::make_move_iterator(c_tasks.end()));
    }
    execute_tasks(tasks);
  }

  if (!(use_two_stage && opts.symmetric_stage2_confirmation)) {
    raw_mean_u = active_tree->GetMeanUtility(raw_action);
    best_action = raw_action;
    best_mean_u = raw_mean_u;
    best_c_idx = 0;
    best_vs_raw_mean_diff = 0.0;
    best_vs_raw_t_stat = 0.0;
    overridden = false;

    if (use_two_stage) {
      if (!stopped_stage1) {
        double cand_mean_u = active_tree->GetMeanUtility(candidates[best_s1_c]);
        if (cand_mean_u > raw_mean_u) {
          best_action = candidates[best_s1_c];
          best_mean_u = cand_mean_u;
          best_c_idx = best_s1_c;
          best_vs_raw_mean_diff = best_mean_u - raw_mean_u;

          const auto& raw_returns = cand_rollout_returns[0];
          const auto& best_returns = cand_rollout_returns[best_c_idx];
          const size_t R = std::min(raw_returns.size(), best_returns.size());
          int pos_cnt = 0;
          if (R >= 2) {
            const auto eb = ComputePairedEbDiffStats(
                best_returns.data(), raw_returns.data(), static_cast<int>(R));
            best_vs_raw_t_stat = eb.t_stat;
            pos_cnt = eb.positive_diff_count;
          } else {
            best_vs_raw_t_stat = (min_paired_t_stat <= 0.0) ? 10.0 : 0.0;
          }

          const double eff_margin = ComputeEffectiveOverrideMargin(
              min_override_margin, raw_mean_u, opts.win_speed_tiebreak_eps,
              opts.ceiling_aware_margin);
          if (best_mean_u > raw_mean_u + eff_margin &&
              best_vs_raw_t_stat >= min_paired_t_stat &&
              PassesSingleFlipGuard(static_cast<int>(R), min_paired_t_stat, pos_cnt)) {
            overridden = true;
          }
        }
      } else {
        double cand_mean_u = active_tree->GetMeanUtility(candidates[best_s1_c]);
        if (cand_mean_u > raw_mean_u) {
          best_action = candidates[best_s1_c];
          best_mean_u = cand_mean_u;
          best_c_idx = best_s1_c;
          best_vs_raw_mean_diff = best_mean_u - raw_mean_u;

          const auto& raw_returns = cand_rollout_returns[0];
          const auto& best_returns = cand_rollout_returns[best_c_idx];
          const size_t R = std::min(raw_returns.size(), best_returns.size());
          if (R >= 2) {
            best_vs_raw_t_stat = ComputePairedEbDiffStats(
                best_returns.data(), raw_returns.data(), static_cast<int>(R)).t_stat;
          }
        }
        overridden = false;
      }
    } else {
      for (size_t c_idx = 1; c_idx < candidates.size(); ++c_idx) {
        Action cand_a = candidates[c_idx];
        double mean_u = active_tree->GetMeanUtility(cand_a);
        if (mean_u > best_mean_u) {
          best_mean_u = mean_u;
          best_action = cand_a;
          best_c_idx = c_idx;
        }
      }

      int pos_cnt = 0;
      size_t paired_n = 0;
      if (best_action != raw_action) {
        best_vs_raw_mean_diff = best_mean_u - raw_mean_u;
        const auto& raw_returns = cand_rollout_returns[0];
        const auto& best_returns = cand_rollout_returns[best_c_idx];
        const size_t R = std::min(raw_returns.size(), best_returns.size());

        if (pairing_mode == SearchPairingMode::kPaired && R >= 2) {
          const auto eb = ComputePairedEbDiffStats(
              best_returns.data(), raw_returns.data(), static_cast<int>(R));
          best_vs_raw_t_stat = eb.t_stat;
          pos_cnt = eb.positive_diff_count;
          paired_n = R;
        } else if (raw_returns.size() >= 2 && best_returns.size() >= 2) {
          double m0 = 0.0, m1 = 0.0;
          for (double v : raw_returns) m0 += v;
          m0 /= raw_returns.size();
          for (double v : best_returns) m1 += v;
          m1 /= best_returns.size();
          double v0 = 0.0, v1 = 0.0;
          for (double v : raw_returns) v0 += (v - m0) * (v - m0);
          v0 /= (raw_returns.size() - 1);
          for (double v : best_returns) v1 += (v - m1) * (v - m1);
          v1 /= (best_returns.size() - 1);
          const double n0 = static_cast<double>(raw_returns.size());
          const double n1 = static_cast<double>(best_returns.size());
          double se = std::sqrt(
              (v0 + kDefaultEbVarFloor / n0) / n0 +
              (v1 + kDefaultEbVarFloor / n1) / n1);
          best_vs_raw_t_stat = (se > 0.0) ? ((m1 - m0) / se) : 0.0;
        } else {
          best_vs_raw_t_stat = (min_paired_t_stat <= 0.0) ? 10.0 : 0.0;
        }
      }

      const double eff_margin = ComputeEffectiveOverrideMargin(
          min_override_margin, raw_mean_u, opts.win_speed_tiebreak_eps,
          opts.ceiling_aware_margin);
      if (best_action != raw_action &&
          best_mean_u > raw_mean_u + eff_margin &&
          best_vs_raw_t_stat >= min_paired_t_stat &&
          (pairing_mode != SearchPairingMode::kPaired ||
           PassesSingleFlipGuard(static_cast<int>(paired_n), min_paired_t_stat, pos_cnt))) {
        overridden = true;
      }
    }
  }

  std::vector<std::pair<Action, double>> candidate_utilities;
  candidate_utilities.reserve(candidates.size());
  for (Action cand_a : candidates) {
    double mean_u = active_tree->GetMeanUtility(cand_a);
    candidate_utilities.push_back({cand_a, mean_u});
  }

  std::vector<std::pair<Action, std::vector<double>>> candidate_rollout_utilities;
  candidate_rollout_utilities.reserve(candidates.size());
  for (size_t c_idx = 0; c_idx < candidates.size(); ++c_idx) {
    candidate_rollout_utilities.push_back({candidates[c_idx], per_candidate_rollouts[c_idx]});
  }

  CompoundSearchDecisionResult result;
  if (opts.symmetric_stage2_confirmation) {
    result.action = teacher_action;
    result.chosen_action = teacher_action;
    result.best_action = best_action;
    result.raw_action = raw_action;
    result.overridden = overridden;
    result.confirmed = confirmed;
    result.teacher_accepted = teacher_accepted;
    result.teacher_action = teacher_action;
    result.stage2_candidate_utilities = std::move(stage2_candidate_utilities);
  } else {
    result.action = overridden ? best_action : raw_action;
    result.chosen_action = result.action;
    result.best_action = best_action;
    result.raw_action = raw_action;
    result.overridden = overridden;
    result.confirmed = false;
    result.teacher_accepted = overridden;
    result.teacher_action = overridden ? best_action : raw_action;
  }
  result.stopped_stage1 = stopped_stage1;
  result.resample_fallbacks = resample_fallbacks.load();
  result.raw_mean_utility = raw_mean_u;
  result.chosen_mean_utility = (overridden || confirmed) ? best_mean_u : raw_mean_u;
  result.prior_rollouts_reused = prior_rollouts_reused;
  result.new_rollouts_executed = new_rollouts_executed;
  result.candidate_priors = std::move(candidate_priors);
  result.candidate_utilities = std::move(candidate_utilities);
  result.candidate_rollout_utilities = std::move(candidate_rollout_utilities);
  result.best_vs_raw_mean_diff = best_vs_raw_mean_diff;
  result.best_vs_raw_t_stat = best_vs_raw_t_stat;
  result.rollout_steps = total_steps;
  result.rollout_nn_evals = total_nn_evals;

  return result;
}

inline CompoundSearchDecisionResult CompoundRolloutSearchDecision(
    const State& state,
    Player search_player,
    TurnSearchTree* turn_tree,
    const std::vector<std::shared_ptr<BatchedNNEvaluator>>& evaluators,
    int top_k,
    int rollouts_per_action,
    double min_override_margin,
    TreeReuseMode reuse_mode,
    uint64_t decision_seed,
    SearchPairingMode pairing_mode,
    SearchTruncationMode truncation_mode,
    bool /*full_turn_supervision*/,
    double min_paired_t_stat = 0.0) {
  return CompoundRolloutSearchDecision(
      state, search_player, turn_tree, evaluators, top_k, rollouts_per_action,
      min_override_margin, reuse_mode, decision_seed, pairing_mode,
      truncation_mode, SearchRolloutCrnMode::kSequential, nullptr, min_paired_t_stat);
}

}  // namespace dune_imperium
}  // namespace open_spiel

#endif  // OPEN_SPIEL_EXAMPLES_DUNE_COMPOUND_TURN_DECISION_H_
