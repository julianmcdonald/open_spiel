#ifndef OPEN_SPIEL_EXAMPLES_DUNE_WIDE_ROLLOUT_SEARCH_H_
#define OPEN_SPIEL_EXAMPLES_DUNE_WIDE_ROLLOUT_SEARCH_H_

#include <algorithm>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <future>
#include <memory>
#include <mutex>
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
    uint64_t decision_seed,
    SearchRolloutCrnMode crn_mode = SearchRolloutCrnMode::kSequential) {
  const auto legal = state.LegalActions();
  if (legal.size() <= 3) {
    return CompoundRolloutSearchDecision(
        state, search_player, turn_tree, evaluators, 3,
        reference_rollouts_per_action, min_override_margin,
        TreeReuseMode::kAdditive, decision_seed, SearchPairingMode::kUnpaired,
        SearchTruncationMode::kTerminal, crn_mode);
  }
  const int top_k = std::min(6, static_cast<int>(legal.size()));
  const int rollouts_per_action = 3 * reference_rollouts_per_action / top_k;
  SPIEL_CHECK_GE(rollouts_per_action, 1);
  return CompoundRolloutSearchDecision(
      state, search_player, turn_tree, evaluators, top_k, rollouts_per_action,
      min_override_margin, TreeReuseMode::kAdditive, decision_seed,
      SearchPairingMode::kPaired, SearchTruncationMode::kTerminal, crn_mode);
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

inline bool IsConvertSpecimenAction(Action a) {
  return a >= kActionConvertSpecimenToTroop0 + 1 &&
         a <= kActionConvertSpecimenToTroop0 + 12;
}

inline bool IsFullStrategicState(
    const State& state,
    Player searched_player,
    const ActionsAndProbs& prior,
    double min_second_prior = 0.020) {
  if (state.CurrentPlayer() != searched_player) return false;
  const std::vector<Action> legal = state.LegalActions();
  if (legal.size() < 2) return false;
  int non_specimen_count = 0;
  for (Action a : legal) {
    if (!IsConvertSpecimenAction(a)) {
      ++non_specimen_count;
    }
  }
  if (non_specimen_count < 2) return false;
  const auto sorted = GetSortedCandidatesByPrior(prior, legal);
  if (sorted.size() < 2 || sorted[1].second < min_second_prior) {
    return false;
  }
  if (IsConvertSpecimenAction(sorted[1].first) &&
      (sorted.size() < 3 || sorted[2].second < min_second_prior)) {
    return false;
  }
  return true;
}

struct PairedRolloutCorrelationStats {
  double sum_corr = 0.0;
  int corr_pair_count = 0;
  double sum_diff_var = 0.0;
  int diff_var_pair_count = 0;
  int pair_count = 0;
  double mean_paired_corr = 0.0;
  double mean_paired_diff_var = 0.0;
};

inline PairedRolloutCorrelationStats ComputeMeanPairedCorrelation(
    const std::vector<std::vector<double>>& per_candidate_rollouts) {
  PairedRolloutCorrelationStats stats;
  if (per_candidate_rollouts.size() < 2) return stats;
  const auto& u_0 = per_candidate_rollouts[0];
  if (u_0.size() < 2) return stats;

  double sum_corr = 0.0;
  int corr_pair_count = 0;
  double sum_diff_var = 0.0;
  int diff_var_pair_count = 0;

  for (size_t c = 1; c < per_candidate_rollouts.size(); ++c) {
    const auto& u_c = per_candidate_rollouts[c];
    const size_t R = std::min(u_0.size(), u_c.size());
    if (R < 2) continue;

    double sum_0 = 0.0;
    double sum_c = 0.0;
    for (size_t i = 0; i < R; ++i) {
      sum_0 += u_0[i];
      sum_c += u_c[i];
    }
    const double mean_0 = sum_0 / static_cast<double>(R);
    const double mean_c = sum_c / static_cast<double>(R);

    double s00 = 0.0;
    double scc = 0.0;
    double s0c = 0.0;
    double sum_sq_diff = 0.0;
    const double mean_diff = mean_c - mean_0;

    for (size_t i = 0; i < R; ++i) {
      const double d0 = u_0[i] - mean_0;
      const double dc = u_c[i] - mean_c;
      s00 += d0 * d0;
      scc += dc * dc;
      s0c += d0 * dc;

      const double diff = (u_c[i] - u_0[i]) - mean_diff;
      sum_sq_diff += diff * diff;
    }

    const double var_d = sum_sq_diff / static_cast<double>(R - 1);
    sum_diff_var += var_d;
    diff_var_pair_count++;

    if (s00 > 1e-12 && scc > 1e-12) {
      const double r = std::clamp(s0c / std::sqrt(s00 * scc), -1.0, 1.0);
      sum_corr += r;
      corr_pair_count++;
    }
  }

  stats.sum_corr = sum_corr;
  stats.corr_pair_count = corr_pair_count;
  stats.sum_diff_var = sum_diff_var;
  stats.diff_var_pair_count = diff_var_pair_count;
  stats.pair_count = corr_pair_count;
  stats.mean_paired_corr = (corr_pair_count > 0) ? (sum_corr / static_cast<double>(corr_pair_count)) : 0.0;
  stats.mean_paired_diff_var = (diff_var_pair_count > 0) ? (sum_diff_var / static_cast<double>(diff_var_pair_count)) : 0.0;
  return stats;
}

inline PairedRolloutCorrelationStats ComputeMeanPairedCorrelation(
    const std::vector<std::pair<Action, std::vector<double>>>& candidate_rollouts) {
  if (candidate_rollouts.size() < 2) return {};
  std::vector<std::vector<double>> matrix;
  matrix.reserve(candidate_rollouts.size());
  for (const auto& pair : candidate_rollouts) {
    matrix.push_back(pair.second);
  }
  return ComputeMeanPairedCorrelation(matrix);
}

struct RolloutTask {
  size_t c_idx;
  int rollout_idx;
  Action action;
  std::unique_ptr<State> sim_state;
  uint64_t wseed;
};

inline int RunClockedPairedRolloutStage(
    const State& state,
    const DuneImperiumState* dune,
    Player search_player,
    const std::vector<std::shared_ptr<BatchedNNEvaluator>>& evaluators,
    const std::vector<Action>& ordered_candidates,
    const std::vector<size_t>& candidate_indices,
    int planned_worlds,
    int world_seed_offset,
    uint64_t decision_seed,
    SearchRolloutCrnMode crn_mode,
    bool sample_opponent_leader_draft,
    std::chrono::steady_clock::time_point stage_deadline_tp,
    TurnSearchTree* active_tree,
    std::vector<std::vector<double>>& per_candidate_rollouts,
    int& new_rollouts_executed,
    int max_inflight_worlds = 32) {
  const size_t num_cands = candidate_indices.size();
  if (planned_worlds <= 0 || num_cands == 0) return 0;
  const int W = std::max(1, max_inflight_worlds);

  std::vector<std::vector<RolloutTrajectory>> trajs(
      planned_worlds, std::vector<RolloutTrajectory>(num_cands));
  std::vector<int> world_remaining(planned_worlds, static_cast<int>(num_cands));
  std::vector<std::future<void>> futures;
  futures.reserve(planned_worlds * num_cands);

  std::mutex cv_m;
  std::condition_variable cv;
  int active_worlds = 0;
  std::atomic<bool> any_aborted{false};

  int launched_worlds = 0;
  for (int w = 0; w < planned_worlds; ++w) {
    {
      std::unique_lock<std::mutex> lock(cv_m);
      cv.wait(lock, [&]() {
        return active_worlds < W || any_aborted.load(std::memory_order_relaxed);
      });
      if (any_aborted.load(std::memory_order_relaxed) ||
          std::chrono::steady_clock::now() >= stage_deadline_tp) {
        break;
      }
      ++active_worlds;
    }
    launched_worlds = w + 1;

    const uint64_t wseed = dune_seed::DeriveSeed(
        decision_seed, static_cast<uint64_t>(world_seed_offset + w));
    const uint64_t s_seed = wseed + 999;

    for (size_t slot = 0; slot < num_cands; ++slot) {
      const size_t c_idx = candidate_indices[slot];
      const Action cand_a = ordered_candidates[c_idx];
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
      auto* s_ptr = sim_state.get();
      size_t eval_idx = (w * num_cands + slot) % (evaluators.empty() ? 1 : evaluators.size());
      auto* active_evaluator = evaluators.empty() ? nullptr : evaluators[eval_idx].get();

      futures.push_back(std::async(
          std::launch::async,
          [s_ptr, sim_state = std::move(sim_state), search_player, active_evaluator,
           s_seed, crn_mode, sample_opponent_leader_draft, stage_deadline_tp,
           &trajs, w, slot, &world_remaining, &active_worlds, &any_aborted,
           &cv_m, &cv]() mutable {
            trajs[w][slot] = SimulateRolloutWithTurnTrajectory(
                s_ptr, search_player, active_evaluator, s_seed,
                SearchTruncationMode::kTerminal, crn_mode,
                sample_opponent_leader_draft, &stage_deadline_tp);
            if (trajs[w][slot].aborted_by_deadline) {
              any_aborted.store(true, std::memory_order_relaxed);
            }
            {
              std::lock_guard<std::mutex> lock(cv_m);
              if (--world_remaining[w] == 0) {
                --active_worlds;
              }
              cv.notify_all();
            }
          }));
    }
  }

  for (auto& f : futures) {
    f.get();
  }

  int kept_worlds = 0;
  for (int w = 0; w < launched_worlds; ++w) {
    bool world_ok = true;
    for (size_t slot = 0; slot < num_cands; ++slot) {
      if (trajs[w][slot].aborted_by_deadline) {
        world_ok = false;
        break;
      }
    }
    if (!world_ok) {
      break;
    }
    for (size_t slot = 0; slot < num_cands; ++slot) {
      const size_t c_idx = candidate_indices[slot];
      if (active_tree) {
        active_tree->RecordRollout(ordered_candidates[c_idx],
                                   trajs[w][slot].intra_turn_actions,
                                   trajs[w][slot].terminal_utility);
      }
      per_candidate_rollouts[c_idx].push_back(trajs[w][slot].terminal_utility);
    }
    ++kept_worlds;
  }

  new_rollouts_executed += kept_worlds * static_cast<int>(num_cands);
  return kept_worlds;
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
    int max_inflight_worlds = 32) {
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
  if (max_time_ms <= 0.0) {
    std::vector<RolloutTask> tasks;
    tasks.reserve(K * rollouts_per_action);

    for (size_t c_idx = 0; c_idx < K; ++c_idx) {
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

    new_rollouts_executed = static_cast<int>(tasks.size());

    if (!tasks.empty()) {
      std::vector<std::future<RolloutTrajectory>> futures;
      futures.reserve(tasks.size());

      for (size_t i = 0; i < tasks.size(); ++i) {
        auto* s_ptr = tasks[i].sim_state.get();
        uint64_t s_seed = tasks[i].wseed;
        auto* active_evaluator = evaluators.empty() ? nullptr : evaluators[i % evaluators.size()].get();
        futures.push_back(std::async(
            std::launch::async,
            [s_ptr, search_player, active_evaluator, s_seed, crn_mode, sample_opponent_leader_draft]() {
              return SimulateRolloutWithTurnTrajectory(
                  s_ptr, search_player, active_evaluator, s_seed, SearchTruncationMode::kTerminal,
                  crn_mode, sample_opponent_leader_draft);
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
    const auto deadline_tp = std::chrono::steady_clock::now() +
        std::chrono::duration_cast<std::chrono::steady_clock::duration>(
            std::chrono::duration<double, std::milli>(max_time_ms));

    std::vector<size_t> candidate_indices(K);
    std::iota(candidate_indices.begin(), candidate_indices.end(), 0);

    RunClockedPairedRolloutStage(
        state, dune, search_player, evaluators, ordered_candidates,
        candidate_indices, rollouts_per_action, /*world_seed_offset=*/0,
        decision_seed, crn_mode, sample_opponent_leader_draft,
        deadline_tp, active_tree, per_candidate_rollouts,
        new_rollouts_executed, max_inflight_worlds);
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

  for (size_t c_idx = 1; c_idx < K; ++c_idx) {
    Action cand_a = ordered_candidates[c_idx];
    double sum_u = 0.0;
    for (int r = 0; r < r_completed; ++r) {
      sum_u += per_candidate_rollouts[c_idx][r];
    }
    double mean_u = (r_completed > 0) ? (sum_u / r_completed) : 0.0;
    double mean_d = mean_u - raw_mean_u;

    double sum_sq_diff = 0.0;
    for (int r = 0; r < r_completed; ++r) {
      double d_r = per_candidate_rollouts[c_idx][r] - per_candidate_rollouts[0][r];
      double diff = d_r - mean_d;
      sum_sq_diff += diff * diff;
    }
    double var_d = (r_completed > 1) ? (sum_sq_diff / (r_completed - 1)) : 0.0;
    double se_d = (r_completed > 1) ? std::sqrt(var_d / r_completed) : 0.0;
    const double t_stat = (r_completed >= 2) ? ((se_d > 1e-9) ? (mean_d / se_d) : (mean_d > 0.0 ? 100.0 : 0.0)) : 0.0;

    if (mean_d > min_override_margin && t_stat >= min_t_stat && mean_u > best_mean_u) {
      best_mean_u = mean_u;
      best_action = cand_a;
      overridden = true;
    }
  }

  std::vector<std::pair<Action, double>> candidate_utilities;
  candidate_utilities.reserve(K);
  for (size_t c_idx = 0; c_idx < K; ++c_idx) {
    double sum = 0.0;
    for (int r = 0; r < r_completed; ++r) {
      sum += per_candidate_rollouts[c_idx][r];
    }
    double m = (r_completed > 0) ? (sum / r_completed) : 0.0;
    candidate_utilities.push_back({ordered_candidates[c_idx], m});
  }

  std::vector<std::pair<Action, std::vector<double>>> candidate_rollout_utilities;
  candidate_rollout_utilities.reserve(K);
  for (size_t c_idx = 0; c_idx < K; ++c_idx) {
    candidate_rollout_utilities.push_back({ordered_candidates[c_idx], per_candidate_rollouts[c_idx]});
  }

  CompoundSearchDecisionResult res;
  res.action = best_action;
  res.overridden = overridden;
  res.raw_mean_utility = raw_mean_u;
  res.chosen_mean_utility = best_mean_u;
  res.prior_rollouts_reused = prior_rollouts_reused;
  res.new_rollouts_executed = new_rollouts_executed;
  res.candidate_utilities = std::move(candidate_utilities);
  res.candidate_rollout_utilities = std::move(candidate_rollout_utilities);
  res.stage1_planned_worlds = rollouts_per_action;
  res.stage1_kept_worlds = r_completed;
  res.stage2_planned_worlds = 0;
  res.stage2_kept_worlds = 0;
  res.stage1_elapsed_ms = stage1_elapsed_ms;
  res.stage2_elapsed_ms = 0.0;
  res.termination_reason = (r_completed < 2) ? "stage1_starved" : (overridden ? "override" : "gate_failed");
  return res;
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
    int max_inflight_worlds = 32) {
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
      /*min_t_stat=*/1.0, decision_seed, crn_mode, sample_opponent_leader_draft,
      max_time_ms, max_inflight_worlds);
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
    int max_inflight_worlds = 32) {
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
      rollouts_per_action, min_override_margin, /*min_t_stat=*/1.0, decision_seed,
      crn_mode, /*sample_opponent_leader_draft=*/false, max_time_ms, max_inflight_worlds);
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
    bool fixed_top5_candidates = false,
    int max_inflight_worlds = 32) {
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
  if (!fixed_top5_candidates) {
    ordered_candidates.reserve(4);
    ordered_candidates.push_back(sorted_by_prior[0].first);
    ordered_candidates.push_back(sorted_by_prior[1].first);

    if (sorted_by_prior.size() >= 3 && sorted_by_prior[2].second >= 0.025) {
      ordered_candidates.push_back(sorted_by_prior[2].first);
      if (sorted_by_prior.size() >= 4 && sorted_by_prior[3].second >= 0.08) {
        ordered_candidates.push_back(sorted_by_prior[3].first);
      }
    }
  } else {
    const size_t k_max = std::min<size_t>(5, sorted_by_prior.size());
    ordered_candidates.reserve(k_max);
    for (size_t i = 0; i < k_max; ++i) {
      ordered_candidates.push_back(sorted_by_prior[i].first);
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

  std::vector<std::vector<double>> per_candidate_rollouts(K);
  for (size_t c_idx = 0; c_idx < K; ++c_idx) {
    per_candidate_rollouts[c_idx].reserve(r_stage1);
  }

  int new_rollouts_executed = 0;
  const bool has_deadline = (max_time_ms > 0.0);
  const auto start_time = std::chrono::steady_clock::now();
  const auto stage1_deadline = has_deadline
      ? (start_time +
         std::chrono::duration_cast<std::chrono::steady_clock::duration>(
             std::chrono::duration<double, std::milli>(0.5 * max_time_ms)))
      : std::chrono::steady_clock::time_point{};
  const auto final_deadline = has_deadline
      ? (start_time +
         std::chrono::duration_cast<std::chrono::steady_clock::duration>(
             std::chrono::duration<double, std::milli>(max_time_ms)))
      : std::chrono::steady_clock::time_point{};

  int stage1_planned_worlds = r_stage1;
  int stage1_kept_worlds = 0;
  const auto s1_t0 = std::chrono::steady_clock::now();

  // --- STAGE 1: Broad Screen ---
  if (!has_deadline) {
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

    new_rollouts_executed += static_cast<int>(tasks_stage1.size());

    if (!tasks_stage1.empty()) {
      std::vector<std::future<RolloutTrajectory>> futures;
      futures.reserve(tasks_stage1.size());

      for (size_t i = 0; i < tasks_stage1.size(); ++i) {
        auto* s_ptr = tasks_stage1[i].sim_state.get();
        uint64_t s_seed = tasks_stage1[i].wseed;
        auto* active_evaluator = evaluators.empty() ? nullptr : evaluators[i % evaluators.size()].get();
        futures.push_back(std::async(
            std::launch::async,
            [s_ptr, search_player, active_evaluator, s_seed, crn_mode]() {
              return SimulateRolloutWithTurnTrajectory(
                  s_ptr, search_player, active_evaluator, s_seed, SearchTruncationMode::kTerminal,
                  crn_mode);
            }));
      }

      for (size_t c_idx = 0; c_idx < K; ++c_idx) {
        per_candidate_rollouts[c_idx].resize(r_stage1, 0.0);
      }
      for (size_t i = 0; i < futures.size(); ++i) {
        RolloutTrajectory traj = futures[i].get();
        active_tree->RecordRollout(tasks_stage1[i].action, traj.intra_turn_actions, traj.terminal_utility);
        per_candidate_rollouts[tasks_stage1[i].c_idx][tasks_stage1[i].rollout_idx] = traj.terminal_utility;
      }
    }
  } else {
    std::vector<size_t> candidate_indices(K);
    std::iota(candidate_indices.begin(), candidate_indices.end(), 0);

    RunClockedPairedRolloutStage(
        state, dune, search_player, evaluators, ordered_candidates,
        candidate_indices, r_stage1, /*world_seed_offset=*/0,
        decision_seed, crn_mode, /*sample_opponent_leader_draft=*/false,
        stage1_deadline, active_tree, per_candidate_rollouts,
        new_rollouts_executed, max_inflight_worlds);
  }

  const auto s1_t1 = std::chrono::steady_clock::now();
  const double stage1_elapsed_ms =
      std::chrono::duration<double, std::milli>(s1_t1 - s1_t0).count();

  const int r_stage1_done = per_candidate_rollouts.empty() ? 0 : static_cast<int>(per_candidate_rollouts[0].size());
  stage1_kept_worlds = r_stage1_done;

  if (r_stage1_done < 2) {
    double raw_mean = 0.0;
    if (r_stage1_done > 0) {
      raw_mean = per_candidate_rollouts[0][0];
    }
    std::vector<std::pair<Action, double>> candidate_utilities;
    candidate_utilities.reserve(K);
    for (size_t c_idx = 0; c_idx < K; ++c_idx) {
      double u_val = (r_stage1_done > 0) ? per_candidate_rollouts[c_idx][0] : 0.0;
      candidate_utilities.push_back({ordered_candidates[c_idx], u_val});
    }
    std::vector<std::pair<Action, std::vector<double>>> candidate_rollout_utilities;
    candidate_rollout_utilities.reserve(K);
    for (size_t c_idx = 0; c_idx < K; ++c_idx) {
      candidate_rollout_utilities.push_back({ordered_candidates[c_idx], per_candidate_rollouts[c_idx]});
    }

    CompoundSearchDecisionResult res;
    res.action = raw_action;
    res.overridden = false;
    res.raw_mean_utility = raw_mean;
    res.chosen_mean_utility = raw_mean;
    res.prior_rollouts_reused = prior_rollouts_reused;
    res.new_rollouts_executed = new_rollouts_executed;
    res.candidate_utilities = std::move(candidate_utilities);
    res.candidate_rollout_utilities = std::move(candidate_rollout_utilities);
    res.stage1_planned_worlds = stage1_planned_worlds;
    res.stage1_kept_worlds = stage1_kept_worlds;
    res.stage2_planned_worlds = 0;
    res.stage2_kept_worlds = 0;
    res.stage1_elapsed_ms = stage1_elapsed_ms;
    res.stage2_elapsed_ms = 0.0;
    res.termination_reason = "stage1_starved";
    return res;
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
  double highest_mean_1 = -1e9;

  for (size_t c_idx = 1; c_idx < K; ++c_idx) {
    double sum_u = 0.0;
    for (int r = 0; r < r_stage1_done; ++r) {
      sum_u += per_candidate_rollouts[c_idx][r];
    }
    mean_1[c_idx] = sum_u / r_stage1_done;
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

    CompoundSearchDecisionResult res;
    res.action = raw_action;
    res.overridden = false;
    res.raw_mean_utility = raw_mean_1;
    res.chosen_mean_utility = raw_mean_1;
    res.prior_rollouts_reused = prior_rollouts_reused;
    res.new_rollouts_executed = new_rollouts_executed;
    res.candidate_utilities = std::move(candidate_utilities);
    res.candidate_rollout_utilities = std::move(candidate_rollout_utilities);
    res.stage1_planned_worlds = stage1_planned_worlds;
    res.stage1_kept_worlds = stage1_kept_worlds;
    res.stage2_planned_worlds = 0;
    res.stage2_kept_worlds = 0;
    res.stage1_elapsed_ms = stage1_elapsed_ms;
    res.stage2_elapsed_ms = 0.0;
    res.termination_reason = "early_stop";
    return res;
  }

  // --- STAGE 2: Deep Finale ---
  const int stage2_budget = std::max(2, total_budget - static_cast<int>(K) * r_stage1);
  const int r_stage2 = std::max(1, stage2_budget / 2);
  const std::array<size_t, 2> finalists = {0, best_c_idx};

  int stage2_planned_worlds = r_stage2;
  int stage2_kept_worlds = 0;
  const auto s2_t0 = std::chrono::steady_clock::now();

  if (!has_deadline) {
    std::vector<RolloutTask> tasks_stage2;
    tasks_stage2.reserve(2 * r_stage2);

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

    new_rollouts_executed += static_cast<int>(tasks_stage2.size());

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
            [s_ptr, search_player, active_evaluator, s_seed, crn_mode]() {
              return SimulateRolloutWithTurnTrajectory(
                  s_ptr, search_player, active_evaluator, s_seed, SearchTruncationMode::kTerminal,
                  crn_mode);
            }));
      }

      for (size_t i = 0; i < futures.size(); ++i) {
        RolloutTrajectory traj = futures[i].get();
        active_tree->RecordRollout(tasks_stage2[i].action, traj.intra_turn_actions, traj.terminal_utility);
        per_candidate_rollouts[tasks_stage2[i].c_idx][tasks_stage2[i].rollout_idx] = traj.terminal_utility;
      }
    }
  } else {
    std::vector<size_t> stage2_cands = {0, best_c_idx};
    RunClockedPairedRolloutStage(
        state, dune, search_player, evaluators, ordered_candidates,
        stage2_cands, r_stage2, /*world_seed_offset=*/r_stage1,
        decision_seed, crn_mode, /*sample_opponent_leader_draft=*/false,
        final_deadline, active_tree, per_candidate_rollouts,
        new_rollouts_executed, max_inflight_worlds);
  }

  const auto s2_t1 = std::chrono::steady_clock::now();
  const double stage2_elapsed_ms =
      std::chrono::duration<double, std::milli>(s2_t1 - s2_t0).count();

  const int r_stage2_done = static_cast<int>(per_candidate_rollouts[0].size()) - r_stage1_done;
  stage2_kept_worlds = r_stage2_done;

  if (r_stage2_done < 2) {
    const int R_tot = static_cast<int>(per_candidate_rollouts[0].size());
    double raw_sum = 0.0;
    for (int r = 0; r < R_tot; ++r) {
      raw_sum += per_candidate_rollouts[0][r];
    }
    const double raw_mean_tot = (R_tot > 0) ? (raw_sum / R_tot) : 0.0;

    std::vector<std::pair<Action, double>> candidate_utilities;
    candidate_utilities.reserve(K);
    for (size_t c_idx = 0; c_idx < K; ++c_idx) {
      double u_val = 0.0;
      if (c_idx == 0) {
        u_val = raw_mean_tot;
      } else if (c_idx == best_c_idx && !per_candidate_rollouts[best_c_idx].empty()) {
        double b_sum = 0.0;
        for (double v : per_candidate_rollouts[best_c_idx]) b_sum += v;
        u_val = b_sum / per_candidate_rollouts[best_c_idx].size();
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

    CompoundSearchDecisionResult res;
    res.action = raw_action;
    res.overridden = false;
    res.raw_mean_utility = raw_mean_tot;
    res.chosen_mean_utility = raw_mean_tot;
    res.prior_rollouts_reused = prior_rollouts_reused;
    res.new_rollouts_executed = new_rollouts_executed;
    res.candidate_utilities = std::move(candidate_utilities);
    res.candidate_rollout_utilities = std::move(candidate_rollout_utilities);
    res.stage1_planned_worlds = stage1_planned_worlds;
    res.stage1_kept_worlds = stage1_kept_worlds;
    res.stage2_planned_worlds = stage2_planned_worlds;
    res.stage2_kept_worlds = stage2_kept_worlds;
    res.stage1_elapsed_ms = stage1_elapsed_ms;
    res.stage2_elapsed_ms = stage2_elapsed_ms;
    res.termination_reason = "stage2_starved";
    return res;
  }

  const int R_total = r_stage1_done + r_stage2_done;

  // Stage 2 paired difference:
  double sum_d_stage2 = 0.0;
  for (int r = 0; r < r_stage2_done; ++r) {
    sum_d_stage2 += (per_candidate_rollouts[best_c_idx][r_stage1_done + r] - per_candidate_rollouts[0][r_stage1_done + r]);
  }
  const double mean_d_stage2 = sum_d_stage2 / r_stage2_done;

  double sum_sq_diff_stage2 = 0.0;
  for (int r = 0; r < r_stage2_done; ++r) {
    double d_r = per_candidate_rollouts[best_c_idx][r_stage1_done + r] -
                 per_candidate_rollouts[0][r_stage1_done + r];
    double diff = d_r - mean_d_stage2;
    sum_sq_diff_stage2 += diff * diff;
  }
  const double var_d_stage2 =
      (r_stage2_done > 1) ? (sum_sq_diff_stage2 / (r_stage2_done - 1)) : 0.0;
  const double se_d_stage2 =
      (r_stage2_done > 1) ? std::sqrt(var_d_stage2 / r_stage2_done) : 0.0;
  const double t_stat_stage2 = (r_stage2_done >= 2)
                                   ? ((se_d_stage2 > 1e-9) ? (mean_d_stage2 / se_d_stage2)
                                                           : (mean_d_stage2 > 0.0 ? 100.0 : 0.0))
                                   : 0.0;

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
  const double t_stat_total = (R_total >= 2)
                                  ? ((se_d > 1e-9) ? (mean_d_total / se_d)
                                                   : (mean_d_total > 0.0 ? 100.0 : 0.0))
                                  : 0.0;

  // Two-Stage Confirmation Override Gate:
  Action best_action = raw_action;
  double best_mean_u = raw_mean_total;
  bool overridden = false;

  bool should_override = false;
  if (!unbiased_stage2_gate) {
    should_override =
        (mean_d_stage2 > 0.0 && mean_d_total > min_override_margin && t_stat_total >= 1.25);
  } else {
    if (K > 2) {
      should_override =
          (mean_d_stage2 >= min_override_margin && t_stat_stage2 >= 1.25);
    } else {
      should_override =
          (mean_d_stage2 > 0.0 && mean_d_total >= min_override_margin && t_stat_total >= 1.25);
    }
  }

  if (should_override) {
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

  CompoundSearchDecisionResult res;
  res.action = best_action;
  res.overridden = overridden;
  res.raw_mean_utility = raw_mean_total;
  res.chosen_mean_utility = best_mean_u;
  res.prior_rollouts_reused = prior_rollouts_reused;
  res.new_rollouts_executed = new_rollouts_executed;
  res.candidate_utilities = std::move(candidate_utilities);
  res.candidate_rollout_utilities = std::move(candidate_rollout_utilities);
  res.stage1_planned_worlds = stage1_planned_worlds;
  res.stage1_kept_worlds = stage1_kept_worlds;
  res.stage2_planned_worlds = stage2_planned_worlds;
  res.stage2_kept_worlds = stage2_kept_worlds;
  res.stage1_elapsed_ms = stage1_elapsed_ms;
  res.stage2_elapsed_ms = stage2_elapsed_ms;
  res.termination_reason = overridden ? "override" : "stage2_no_override";
  return res;
}

}  // namespace dune_imperium
}  // namespace open_spiel

#endif  // OPEN_SPIEL_EXAMPLES_DUNE_WIDE_ROLLOUT_SEARCH_H_
