#ifndef OPEN_SPIEL_EXAMPLES_DUNE_WIDE_ROLLOUT_SEARCH_H_
#define OPEN_SPIEL_EXAMPLES_DUNE_WIDE_ROLLOUT_SEARCH_H_

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <future>
#include <memory>
#include <utility>
#include <vector>

#include "open_spiel/examples/dune_compound_turn_search.h"
#include "open_spiel/examples/dune_seed_utils.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace dune_imperium {

inline CompoundSearchDecisionResult WideRolloutSearchDecision(
    const State& state, Player search_player, TurnSearchTree* turn_tree,
    const std::vector<std::shared_ptr<BatchedNNEvaluator>>& evaluators,
    int reference_rollouts_per_action, double min_override_margin,
    uint64_t decision_seed) {
  const auto legal = state.LegalActions();
  if (legal.size() <= 3) {
    return CompoundRolloutSearchDecision(
        state, search_player, turn_tree, evaluators, 3,
        reference_rollouts_per_action, min_override_margin,
        TreeReuseMode::kAdditive, decision_seed, SearchPairingMode::kUnpaired,
        SearchTruncationMode::kTerminal);
  }
  const int top_k = std::min(6, static_cast<int>(legal.size()));
  const int rollouts_per_action = 3 * reference_rollouts_per_action / top_k;
  SPIEL_CHECK_GE(rollouts_per_action, 1);
  return CompoundRolloutSearchDecision(
      state, search_player, turn_tree, evaluators, top_k, rollouts_per_action,
      min_override_margin, TreeReuseMode::kAdditive, decision_seed,
      SearchPairingMode::kPaired, SearchTruncationMode::kTerminal);
}

inline std::vector<std::pair<Action, double>> GetSortedCandidatesByPrior(
    const ActionsAndProbs& prior,
    const std::vector<Action>& legal_actions) {
  std::vector<std::pair<double, Action>> scored;
  scored.reserve(legal_actions.size());
  for (Action a : legal_actions) {
    double p = 0.0;
    for (const auto& ap : prior) {
      if (ap.first == a) {
        p = ap.second;
        break;
      }
    }
    scored.push_back({p, a});
  }
  std::stable_sort(scored.begin(), scored.end(),
                   [](const auto& lhs, const auto& rhs) {
                     if (lhs.first != rhs.first) return lhs.first > rhs.first;
                     return lhs.second < rhs.second;
                   });
  std::vector<std::pair<Action, double>> result;
  result.reserve(scored.size());
  for (const auto& item : scored) {
    result.push_back({item.second, item.first});
  }
  return result;
}

struct RolloutTask {
  size_t c_idx;
  int rollout_idx;
  Action action;
  std::unique_ptr<State> sim_state;
  uint64_t wseed;
};

inline CompoundSearchDecisionResult PairedPriorSortedRolloutSearchDecision(
    const State& state,
    Player search_player,
    TurnSearchTree* turn_tree,
    const std::vector<std::shared_ptr<BatchedNNEvaluator>>& evaluators,
    const std::vector<Action>& ordered_candidates,
    int rollouts_per_action,
    double min_override_margin,
    double min_t_stat,
    uint64_t decision_seed) {
  if (ordered_candidates.empty()) {
    return {kInvalidAction, false, 0.0, 0.0, 0, 0, {}, {}};
  }
  if (ordered_candidates.size() == 1 || rollouts_per_action <= 0) {
    return {ordered_candidates.front(), false, 0.0, 0.0, 0, 0, {}, {}};
  }

  const Action raw_action = ordered_candidates.front();
  const auto* dune = dynamic_cast<const dune_imperium::DuneImperiumState*>(&state);

  std::vector<RolloutTask> tasks;
  tasks.reserve(ordered_candidates.size() * rollouts_per_action);

  int prior_rollouts_reused = 0;
  if (turn_tree != nullptr) {
    for (Action cand_a : ordered_candidates) {
      prior_rollouts_reused += turn_tree->GetVisits(cand_a);
    }
  }

  for (size_t c_idx = 0; c_idx < ordered_candidates.size(); ++c_idx) {
    const Action cand_a = ordered_candidates[c_idx];
    for (int r = 0; r < rollouts_per_action; ++r) {
      uint64_t wseed = dune_seed::DeriveSeed(decision_seed, r);
      std::unique_ptr<State> sim_state = nullptr;
      if (dune) {
        std::mt19937 wrng(wseed);
        auto rng_func = [&wrng]() {
          return std::generate_canonical<double, 53>(wrng);
        };
        sim_state = dune->ResampleFromInfostate(search_player, rng_func);
      }
      if (!sim_state) {
        sim_state = state.Clone();
      }
      sim_state->ApplyAction(cand_a);
      tasks.push_back({c_idx, r, cand_a, std::move(sim_state), wseed + 999});
    }
  }

  int new_rollouts_executed = static_cast<int>(tasks.size());

  TurnSearchTree local_tree;
  TurnSearchTree* active_tree = (turn_tree != nullptr) ? turn_tree : &local_tree;

  std::vector<std::vector<double>> per_candidate_rollouts(
      ordered_candidates.size(), std::vector<double>(rollouts_per_action, 0.0));

  if (!tasks.empty()) {
    std::vector<std::future<RolloutTrajectory>> futures;
    futures.reserve(tasks.size());

    for (size_t i = 0; i < tasks.size(); ++i) {
      auto* s_ptr = tasks[i].sim_state.get();
      uint64_t s_seed = tasks[i].wseed;
      auto* active_evaluator = evaluators.empty() ? nullptr : evaluators[i % evaluators.size()].get();
      futures.push_back(std::async(
          std::launch::async,
          [s_ptr, search_player, active_evaluator, s_seed]() {
            return SimulateRolloutWithTurnTrajectory(
                s_ptr, search_player, active_evaluator, s_seed, SearchTruncationMode::kTerminal);
          }));
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

  double raw_mean_u = 0.0;
  for (int r = 0; r < rollouts_per_action; ++r) {
    raw_mean_u += per_candidate_rollouts[0][r];
  }
  if (rollouts_per_action > 0) {
    raw_mean_u /= rollouts_per_action;
  }

  Action best_action = raw_action;
  double best_mean_u = raw_mean_u;
  bool overridden = false;

  for (size_t c_idx = 1; c_idx < ordered_candidates.size(); ++c_idx) {
    Action cand_a = ordered_candidates[c_idx];
    double sum_u = 0.0;
    for (int r = 0; r < rollouts_per_action; ++r) {
      sum_u += per_candidate_rollouts[c_idx][r];
    }
    double mean_u = (rollouts_per_action > 0) ? (sum_u / rollouts_per_action) : 0.0;
    double mean_d = mean_u - raw_mean_u;

    double sum_sq_diff = 0.0;
    for (int r = 0; r < rollouts_per_action; ++r) {
      double d_r = per_candidate_rollouts[c_idx][r] - per_candidate_rollouts[0][r];
      double diff = d_r - mean_d;
      sum_sq_diff += diff * diff;
    }
    double var_d = (rollouts_per_action > 1) ? (sum_sq_diff / (rollouts_per_action - 1)) : 0.0;
    double se_d = (rollouts_per_action > 1) ? std::sqrt(var_d / rollouts_per_action) : 0.0;
    double t_stat = (se_d > 1e-9) ? (mean_d / se_d) : (mean_d > 0.0 ? 100.0 : 0.0);

    if (mean_d > min_override_margin && t_stat >= min_t_stat && mean_u > best_mean_u) {
      best_mean_u = mean_u;
      best_action = cand_a;
      overridden = true;
    }
  }

  std::vector<std::pair<Action, double>> candidate_utilities;
  candidate_utilities.reserve(ordered_candidates.size());
  for (size_t c_idx = 0; c_idx < ordered_candidates.size(); ++c_idx) {
    double sum = 0.0;
    for (int r = 0; r < rollouts_per_action; ++r) {
      sum += per_candidate_rollouts[c_idx][r];
    }
    double m = (rollouts_per_action > 0) ? (sum / rollouts_per_action) : 0.0;
    candidate_utilities.push_back({ordered_candidates[c_idx], m});
  }

  std::vector<std::pair<Action, std::vector<double>>> candidate_rollout_utilities;
  candidate_rollout_utilities.reserve(ordered_candidates.size());
  for (size_t c_idx = 0; c_idx < ordered_candidates.size(); ++c_idx) {
    candidate_rollout_utilities.push_back({ordered_candidates[c_idx], per_candidate_rollouts[c_idx]});
  }

  return {best_action, overridden, raw_mean_u, best_mean_u, prior_rollouts_reused,
          new_rollouts_executed, candidate_utilities, candidate_rollout_utilities};
}

inline CompoundSearchDecisionResult LeaderPickRolloutSearchDecision(
    const State& state,
    Player search_player,
    const std::vector<std::shared_ptr<BatchedNNEvaluator>>& evaluators,
    int rollouts_per_leader,
    double min_override_margin,
    uint64_t decision_seed) {
  const auto legal = state.LegalActions();
  if (legal.empty()) return {kInvalidAction, false, 0.0, 0.0, 0, 0, {}, {}};
  if (legal.size() == 1) return {legal.front(), false, 0.0, 0.0, 0, 0, {}, {}};
  if (evaluators.empty()) {
    return {legal.front(), false, 0.0, 0.0, 0, 0, {}, {}};
  }

  const ActionsAndProbs raw_prior = evaluators.front()->Prior(state);
  const auto sorted_candidates = GetSortedCandidatesByPrior(raw_prior, legal);
  std::vector<Action> ordered_candidates;
  ordered_candidates.reserve(sorted_candidates.size());
  for (const auto& item : sorted_candidates) {
    ordered_candidates.push_back(item.first);
  }

  return PairedPriorSortedRolloutSearchDecision(
      state, search_player, /*turn_tree=*/nullptr, evaluators,
      ordered_candidates, rollouts_per_leader, min_override_margin,
      /*min_t_stat=*/1.0, decision_seed);
}

inline CompoundSearchDecisionResult AdaptiveRolloutSearchDecision(
    const State& state,
    Player search_player,
    TurnSearchTree* turn_tree,
    const std::vector<std::shared_ptr<BatchedNNEvaluator>>& evaluators,
    int reference_rollouts_per_action,
    double min_override_margin,
    uint64_t decision_seed) {
  const auto legal = state.LegalActions();
  if (legal.empty()) return {kInvalidAction, false, 0.0, 0.0, 0, 0, {}, {}};
  if (legal.size() == 1) return {legal.front(), false, 0.0, 0.0, 0, 0, {}, {}};
  if (evaluators.empty()) {
    return {legal.front(), false, 0.0, 0.0, 0, 0, {}, {}};
  }

  const ActionsAndProbs raw_prior = evaluators.front()->Prior(state);
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
      rollouts_per_action, min_override_margin, /*min_t_stat=*/1.0, decision_seed);
}

inline CompoundSearchDecisionResult SuccessiveHalvingRolloutSearchDecision(
    const State& state,
    Player search_player,
    TurnSearchTree* turn_tree,
    const std::vector<std::shared_ptr<BatchedNNEvaluator>>& evaluators,
    int reference_rollouts_per_action,
    double min_override_margin,
    uint64_t decision_seed) {
  const auto legal = state.LegalActions();
  if (legal.empty()) return {kInvalidAction, false, 0.0, 0.0, 0, 0, {}, {}};
  if (legal.size() == 1) return {legal.front(), false, 0.0, 0.0, 0, 0, {}, {}};
  if (evaluators.empty()) {
    return {legal.front(), false, 0.0, 0.0, 0, 0, {}, {}};
  }

  const ActionsAndProbs raw_prior = evaluators.front()->Prior(state);
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

  const size_t K = ordered_candidates.size();
  const Action raw_action = ordered_candidates[0];

  int prior_rollouts_reused = 0;
  if (turn_tree != nullptr) {
    for (Action cand_a : ordered_candidates) {
      prior_rollouts_reused += turn_tree->GetVisits(cand_a);
    }
  }

  const auto* dune = dynamic_cast<const DuneImperiumState*>(&state);
  TurnSearchTree local_tree;
  TurnSearchTree* active_tree = (turn_tree != nullptr) ? turn_tree : &local_tree;

  const int total_budget = 3 * reference_rollouts_per_action;
  const int stage1_budget = total_budget / 2;
  const int r_stage1 = std::max(1, stage1_budget / static_cast<int>(K));

  // --- STAGE 1: Broad Screen ---
  std::vector<RolloutTask> tasks_stage1;
  tasks_stage1.reserve(K * r_stage1);

  for (size_t c_idx = 0; c_idx < K; ++c_idx) {
    const Action cand_a = ordered_candidates[c_idx];
    for (int r = 0; r < r_stage1; ++r) {
      uint64_t wseed = dune_seed::DeriveSeed(decision_seed, static_cast<uint64_t>(r));
      std::unique_ptr<State> sim_state = nullptr;
      if (dune) {
        std::mt19937 wrng(wseed);
        auto rng_func = [&wrng]() {
          return std::generate_canonical<double, 53>(wrng);
        };
        sim_state = dune->ResampleFromInfostate(search_player, rng_func);
      }
      if (!sim_state) {
        sim_state = state.Clone();
      }
      sim_state->ApplyAction(cand_a);
      tasks_stage1.push_back({c_idx, r, cand_a, std::move(sim_state), wseed + 999});
    }
  }

  std::vector<std::vector<double>> per_candidate_rollouts(K, std::vector<double>(r_stage1, 0.0));

  if (!tasks_stage1.empty()) {
    std::vector<std::future<RolloutTrajectory>> futures;
    futures.reserve(tasks_stage1.size());

    for (size_t i = 0; i < tasks_stage1.size(); ++i) {
      auto* s_ptr = tasks_stage1[i].sim_state.get();
      uint64_t s_seed = tasks_stage1[i].wseed;
      auto* active_evaluator = evaluators.empty() ? nullptr : evaluators[i % evaluators.size()].get();
      futures.push_back(std::async(
          std::launch::async,
          [s_ptr, search_player, active_evaluator, s_seed]() {
            return SimulateRolloutWithTurnTrajectory(
                s_ptr, search_player, active_evaluator, s_seed, SearchTruncationMode::kTerminal);
          }));
    }

    for (size_t i = 0; i < futures.size(); ++i) {
      RolloutTrajectory traj = futures[i].get();
      active_tree->RecordRollout(tasks_stage1[i].action, traj.intra_turn_actions, traj.terminal_utility);
      per_candidate_rollouts[tasks_stage1[i].c_idx][tasks_stage1[i].rollout_idx] = traj.terminal_utility;
    }
  }

  double raw_mean_1 = 0.0;
  for (int r = 0; r < r_stage1; ++r) {
    raw_mean_1 += per_candidate_rollouts[0][r];
  }
  raw_mean_1 /= r_stage1;

  std::vector<double> mean_1(K, 0.0);
  std::vector<double> mean_d_1(K, 0.0);
  mean_1[0] = raw_mean_1;
  mean_d_1[0] = 0.0;

  size_t best_c_idx = 1;
  double highest_mean_1 = -1e9;

  for (size_t c_idx = 1; c_idx < K; ++c_idx) {
    double sum_u = 0.0;
    for (int r = 0; r < r_stage1; ++r) {
      sum_u += per_candidate_rollouts[c_idx][r];
    }
    mean_1[c_idx] = sum_u / r_stage1;
    mean_d_1[c_idx] = mean_1[c_idx] - raw_mean_1;
    if (mean_1[c_idx] > highest_mean_1) {
      highest_mean_1 = mean_1[c_idx];
      best_c_idx = c_idx;
    }
  }

  // Stage 1 Early-Stop Gate: If no challenger beat raw_action in Stage 1, stop early.
  if (mean_d_1[best_c_idx] <= 0.0) {
    std::vector<std::pair<Action, double>> candidate_utilities;
    candidate_utilities.reserve(K);
    for (size_t c_idx = 0; c_idx < K; ++c_idx) {
      candidate_utilities.push_back({ordered_candidates[c_idx], mean_1[c_idx]});
    }
    std::vector<std::pair<Action, std::vector<double>>> candidate_rollout_utilities;
    candidate_rollout_utilities.reserve(K);
    for (size_t c_idx = 0; c_idx < K; ++c_idx) {
      candidate_rollout_utilities.push_back({ordered_candidates[c_idx], per_candidate_rollouts[c_idx]});
    }

    return {raw_action, /*overridden=*/false, raw_mean_1, raw_mean_1, prior_rollouts_reused,
            static_cast<int>(tasks_stage1.size()), candidate_utilities, candidate_rollout_utilities};
  }

  // --- STAGE 2: Deep Finale ---
  const int stage2_budget = std::max(2, total_budget - static_cast<int>(K) * r_stage1);
  const int r_stage2 = std::max(1, stage2_budget / 2);

  std::vector<RolloutTask> tasks_stage2;
  tasks_stage2.reserve(2 * r_stage2);

  const std::array<size_t, 2> finalists = {0, best_c_idx};

  for (size_t f_idx : finalists) {
    const Action cand_a = ordered_candidates[f_idx];
    for (int r = 0; r < r_stage2; ++r) {
      uint64_t wseed = dune_seed::DeriveSeed(decision_seed, static_cast<uint64_t>(r_stage1 + r));
      std::unique_ptr<State> sim_state = nullptr;
      if (dune) {
        std::mt19937 wrng(wseed);
        auto rng_func = [&wrng]() {
          return std::generate_canonical<double, 53>(wrng);
        };
        sim_state = dune->ResampleFromInfostate(search_player, rng_func);
      }
      if (!sim_state) {
        sim_state = state.Clone();
      }
      sim_state->ApplyAction(cand_a);
      tasks_stage2.push_back({f_idx, r_stage1 + r, cand_a, std::move(sim_state), wseed + 999});
    }
  }

  for (size_t f_idx : finalists) {
    per_candidate_rollouts[f_idx].resize(r_stage1 + r_stage2, 0.0);
  }

  if (!tasks_stage2.empty()) {
    std::vector<std::future<RolloutTrajectory>> futures;
    futures.reserve(tasks_stage2.size());

    for (size_t i = 0; i < tasks_stage2.size(); ++i) {
      auto* s_ptr = tasks_stage2[i].sim_state.get();
      uint64_t s_seed = tasks_stage2[i].wseed;
      auto* active_evaluator = evaluators.empty() ? nullptr : evaluators[i % evaluators.size()].get();
      futures.push_back(std::async(
          std::launch::async,
          [s_ptr, search_player, active_evaluator, s_seed]() {
            return SimulateRolloutWithTurnTrajectory(
                s_ptr, search_player, active_evaluator, s_seed, SearchTruncationMode::kTerminal);
          }));
    }

    for (size_t i = 0; i < futures.size(); ++i) {
      RolloutTrajectory traj = futures[i].get();
      active_tree->RecordRollout(tasks_stage2[i].action, traj.intra_turn_actions, traj.terminal_utility);
      per_candidate_rollouts[tasks_stage2[i].c_idx][tasks_stage2[i].rollout_idx] = traj.terminal_utility;
    }
  }

  const int R_total = r_stage1 + r_stage2;

  // Stage 2 paired difference:
  double sum_d_stage2 = 0.0;
  for (int r = 0; r < r_stage2; ++r) {
    sum_d_stage2 += (per_candidate_rollouts[best_c_idx][r_stage1 + r] - per_candidate_rollouts[0][r_stage1 + r]);
  }
  const double mean_d_stage2 = sum_d_stage2 / r_stage2;

  // Total statistics across all R_total rollouts:
  double sum_raw_total = 0.0;
  double sum_best_total = 0.0;
  for (int r = 0; r < R_total; ++r) {
    sum_raw_total += per_candidate_rollouts[0][r];
    sum_best_total += per_candidate_rollouts[best_c_idx][r];
  }
  const double raw_mean_total = sum_raw_total / R_total;
  const double best_cand_mean_total = sum_best_total / R_total;
  const double mean_d_total = best_cand_mean_total - raw_mean_total;

  double sum_sq_diff = 0.0;
  for (int r = 0; r < R_total; ++r) {
    double d_r = per_candidate_rollouts[best_c_idx][r] - per_candidate_rollouts[0][r];
    double diff = d_r - mean_d_total;
    sum_sq_diff += diff * diff;
  }
  const double var_d = (R_total > 1) ? (sum_sq_diff / (R_total - 1)) : 0.0;
  const double se_d = (R_total > 1) ? std::sqrt(var_d / R_total) : 0.0;
  const double t_stat_total = (se_d > 1e-9) ? (mean_d_total / se_d) : (mean_d_total > 0.0 ? 100.0 : 0.0);

  // Two-Stage Confirmation Override Gate:
  Action best_action = raw_action;
  double best_mean_u = raw_mean_total;
  bool overridden = false;

  if (mean_d_stage2 > 0.0 && mean_d_total > min_override_margin && t_stat_total >= 1.25) {
    best_action = ordered_candidates[best_c_idx];
    best_mean_u = best_cand_mean_total;
    overridden = true;
  }

  std::vector<std::pair<Action, double>> candidate_utilities;
  candidate_utilities.reserve(K);
  for (size_t c_idx = 0; c_idx < K; ++c_idx) {
    double u_val = 0.0;
    if (c_idx == 0) {
      u_val = raw_mean_total;
    } else if (c_idx == best_c_idx) {
      u_val = best_cand_mean_total;
    } else {
      u_val = mean_1[c_idx];
    }
    candidate_utilities.push_back({ordered_candidates[c_idx], u_val});
  }

  std::vector<std::pair<Action, std::vector<double>>> candidate_rollout_utilities;
  candidate_rollout_utilities.reserve(K);
  for (size_t c_idx = 0; c_idx < K; ++c_idx) {
    candidate_rollout_utilities.push_back({ordered_candidates[c_idx], per_candidate_rollouts[c_idx]});
  }

  const int new_rollouts_executed = static_cast<int>(tasks_stage1.size() + tasks_stage2.size());

  return {best_action, overridden, raw_mean_total, best_mean_u, prior_rollouts_reused,
          new_rollouts_executed, candidate_utilities, candidate_rollout_utilities};
}

}  // namespace dune_imperium
}  // namespace open_spiel

#endif  // OPEN_SPIEL_EXAMPLES_DUNE_WIDE_ROLLOUT_SEARCH_H_
