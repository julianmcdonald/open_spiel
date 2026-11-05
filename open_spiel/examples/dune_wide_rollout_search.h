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

struct ClockedStageSharedState {
  ClockedStageSharedState(int worlds, size_t cands)
      : trajs(worlds, std::vector<RolloutTrajectory>(cands)),
        world_remaining(worlds, static_cast<int>(cands)) {}
  std::vector<std::vector<RolloutTrajectory>> trajs;
  std::vector<int> world_remaining;
  std::mutex cv_m;
  std::condition_variable cv;
  int active_worlds = 0;
  std::atomic<bool> any_aborted{false};
};

struct SlotReleaseGuard {
  std::shared_ptr<ClockedStageSharedState> state;
  int world_idx;
  BatchedNNEvaluator* evaluator = nullptr;
  bool normal_finish = false;
  ~SlotReleaseGuard() {
    if (evaluator) {
      evaluator->AdjustActiveRollouts(-1);
    }
    if (!normal_finish) {
      state->any_aborted.store(true, std::memory_order_relaxed);
    }
    std::lock_guard<std::mutex> lock(state->cv_m);
    if (--state->world_remaining[world_idx] == 0) {
      --state->active_worlds;
    }
    state->cv.notify_all();
  }
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
    int max_inflight_worlds = 32,
    bool resample_from_infostate = true,
    const std::vector<Action>* second_actions = nullptr,
    const std::vector<Action>* graft_actions = nullptr) {
  const size_t num_cands = candidate_indices.size();
  if (planned_worlds <= 0 || num_cands == 0) return 0;
  const int W = std::max(1, max_inflight_worlds);

  auto shared = std::make_shared<ClockedStageSharedState>(planned_worlds, num_cands);
  std::vector<std::future<void>> futures;
  futures.reserve(planned_worlds * num_cands);

  std::exception_ptr first_exception = nullptr;
  int launched_worlds = 0;
  const int initial_burst_worlds = std::min(planned_worlds, W);
  for (int w = 0; w < initial_burst_worlds; ++w) {
    for (size_t slot = 0; slot < num_cands; ++slot) {
      size_t eval_idx = (w * num_cands + slot) % (evaluators.empty() ? 1 : evaluators.size());
      auto* active_evaluator = evaluators.empty() ? nullptr : evaluators[eval_idx].get();
      if (active_evaluator) {
        active_evaluator->AdjustActiveRollouts(1);
      }
    }
  }

  for (int w = 0; w < planned_worlds; ++w) {
    const bool in_initial_burst = (w < initial_burst_worlds);
    {
      std::unique_lock<std::mutex> lock(shared->cv_m);
      shared->cv.wait(lock, [&]() {
        return shared->active_worlds < W || shared->any_aborted.load(std::memory_order_relaxed);
      });
      if (shared->any_aborted.load(std::memory_order_relaxed) ||
          std::chrono::steady_clock::now() >= stage_deadline_tp) {
        if (in_initial_burst) {
          for (int rem_w = w; rem_w < initial_burst_worlds; ++rem_w) {
            for (size_t slot = 0; slot < num_cands; ++slot) {
              size_t eval_idx = (rem_w * num_cands + slot) % (evaluators.empty() ? 1 : evaluators.size());
              auto* active_evaluator = evaluators.empty() ? nullptr : evaluators[eval_idx].get();
              if (active_evaluator) {
                active_evaluator->AdjustActiveRollouts(-1);
              }
            }
          }
        }
        break;
      }
      ++shared->active_worlds;
    }
    launched_worlds = w + 1;

    const uint64_t wseed = dune_seed::DeriveSeed(
        decision_seed, static_cast<uint64_t>(world_seed_offset + w));
    const uint64_t s_seed = wseed + 999;

    std::unique_ptr<State> base_world_state = nullptr;
    if (dune && resample_from_infostate) {
      std::mt19937 wrng(wseed);
      auto rng_func = [&wrng]() {
        return std::generate_canonical<double, 53>(wrng);
      };
      base_world_state = dune->ResampleFromInfostate(search_player, rng_func);
    }
    if (!base_world_state) {
      base_world_state = state.Clone();
    }

    bool slot_launch_failed = false;
    for (size_t slot = 0; slot < num_cands; ++slot) {
      BatchedNNEvaluator* adjusted_evaluator = nullptr;
      size_t eval_idx = (w * num_cands + slot) % (evaluators.empty() ? 1 : evaluators.size());
      auto* active_evaluator = evaluators.empty() ? nullptr : evaluators[eval_idx].get();
      if (in_initial_burst) {
        adjusted_evaluator = active_evaluator;
      }
      try {
        const size_t c_idx = candidate_indices[slot];
        const Action cand_a = ordered_candidates[c_idx];
        std::unique_ptr<State> sim_state =
            (slot + 1 == num_cands) ? std::move(base_world_state) : base_world_state->Clone();
        sim_state->ApplyAction(cand_a);
        if (graft_actions != nullptr && c_idx < graft_actions->size()) {
          const Action graft_a = (*graft_actions)[c_idx];
          if (graft_a != kInvalidAction) {
            const auto leg_g = sim_state->LegalActions();
            if (std::find(leg_g.begin(), leg_g.end(), graft_a) != leg_g.end()) {
              sim_state->ApplyAction(graft_a);
            }
          }
        }
        if (second_actions != nullptr && c_idx < second_actions->size()) {
          const Action second_a = (*second_actions)[c_idx];
          if (second_a != kInvalidAction) {
            const auto leg2 = sim_state->LegalActions();
            if (std::find(leg2.begin(), leg2.end(), second_a) != leg2.end()) {
              sim_state->ApplyAction(second_a);
            }
          }
        }
        auto* s_ptr = sim_state.get();
        if (!in_initial_burst && active_evaluator) {
          active_evaluator->AdjustActiveRollouts(1);
          adjusted_evaluator = active_evaluator;
        }

        futures.push_back(SearchRolloutThreadPool::Instance().Submit(
            [shared, s_ptr, sim_state = std::move(sim_state), search_player, active_evaluator,
             s_seed, crn_mode, sample_opponent_leader_draft, stage_deadline_tp,
             w, slot]() mutable {
              SlotReleaseGuard guard{shared, w, active_evaluator, false};
              shared->trajs[w][slot] = SimulateRolloutWithTurnTrajectory(
                  s_ptr, search_player, active_evaluator, s_seed,
                  SearchTruncationMode::kTerminal, crn_mode,
                  sample_opponent_leader_draft, &stage_deadline_tp);
              if (shared->trajs[w][slot].aborted_by_deadline) {
                shared->any_aborted.store(true, std::memory_order_relaxed);
              }
              guard.normal_finish = true;
            }));
      } catch (...) {
        if (adjusted_evaluator) {
          adjusted_evaluator->AdjustActiveRollouts(-1);
        }
        if (in_initial_burst) {
          for (size_t rem_slot = slot + 1; rem_slot < num_cands; ++rem_slot) {
            size_t r_idx = (w * num_cands + rem_slot) % (evaluators.empty() ? 1 : evaluators.size());
            auto* r_eval = evaluators.empty() ? nullptr : evaluators[r_idx].get();
            if (r_eval) r_eval->AdjustActiveRollouts(-1);
          }
          for (int rem_w = w + 1; rem_w < initial_burst_worlds; ++rem_w) {
            for (size_t rem_slot = 0; rem_slot < num_cands; ++rem_slot) {
              size_t r_idx = (rem_w * num_cands + rem_slot) % (evaluators.empty() ? 1 : evaluators.size());
              auto* r_eval = evaluators.empty() ? nullptr : evaluators[r_idx].get();
              if (r_eval) r_eval->AdjustActiveRollouts(-1);
            }
          }
        }
        if (!first_exception) {
          first_exception = std::current_exception();
        }
        shared->any_aborted.store(true, std::memory_order_relaxed);
        {
          std::lock_guard<std::mutex> lock(shared->cv_m);
          shared->world_remaining[w] -= static_cast<int>(num_cands - slot);
          if (shared->world_remaining[w] == 0) {
            --shared->active_worlds;
          }
          shared->cv.notify_all();
        }
        slot_launch_failed = true;
        break;
      }
    }
    if (slot_launch_failed) {
      break;
    }
  }

  for (auto& f : futures) {
    try {
      f.get();
    } catch (...) {
      if (!first_exception) {
        first_exception = std::current_exception();
      }
    }
  }

  if (first_exception) {
    std::rethrow_exception(first_exception);
  }

  int kept_worlds = 0;
  for (int w = 0; w < launched_worlds; ++w) {
    bool world_ok = true;
    for (size_t slot = 0; slot < num_cands; ++slot) {
      if (shared->trajs[w][slot].aborted_by_deadline) {
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
        std::vector<Action> intra = shared->trajs[w][slot].intra_turn_actions;
        if (second_actions != nullptr && c_idx < second_actions->size()) {
          const Action second_a = (*second_actions)[c_idx];
          if (second_a != kInvalidAction) {
            intra.insert(intra.begin(), second_a);
          }
        }
        if (graft_actions != nullptr && c_idx < graft_actions->size()) {
          const Action graft_a = (*graft_actions)[c_idx];
          if (graft_a != kInvalidAction) {
            intra.insert(intra.begin(), graft_a);
          }
        }
        active_tree->RecordRollout(ordered_candidates[c_idx],
                                   intra,
                                   shared->trajs[w][slot].terminal_utility);
      }
      per_candidate_rollouts[c_idx].push_back(shared->trajs[w][slot].terminal_utility);
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
    int max_inflight_worlds = 32,
    bool resample_from_infostate = true,
    bool use_bounded_pool_when_unclocked = false) {
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
        std::mt19937 wrng(wseed);
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
            [s_ptr, search_player, active_evaluator, s_seed, crn_mode, sample_opponent_leader_draft]() {
              ActiveRolloutReleaseGuard guard{active_evaluator};
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
    const auto deadline_tp = (max_time_ms > 0.0)
        ? (s1_t0 +
           std::chrono::duration_cast<std::chrono::steady_clock::duration>(
               std::chrono::duration<double, std::milli>(max_time_ms)))
        : std::chrono::steady_clock::time_point::max();

    std::vector<size_t> candidate_indices(K);
    std::iota(candidate_indices.begin(), candidate_indices.end(), 0);

    int seed_offset = 0;
    int first_wave = (max_time_ms > 0.0) ? std::min(rollouts_per_action, 512) : rollouts_per_action;
    const auto w0_t0 = std::chrono::steady_clock::now();
    RunClockedPairedRolloutStage(
        state, dune, search_player, evaluators, ordered_candidates,
        candidate_indices, first_wave, /*world_seed_offset=*/seed_offset,
        decision_seed, crn_mode, sample_opponent_leader_draft,
        deadline_tp, active_tree, per_candidate_rollouts,
        new_rollouts_executed, max_inflight_worlds,
        resample_from_infostate);
    seed_offset += first_wave;

    if (max_time_ms > 0.0 && !per_candidate_rollouts.empty()) {
      int done_w = static_cast<int>(per_candidate_rollouts[0].size());
      double w_ms = std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - w0_t0).count();
      double cur_ms_per_sim = (done_w > 0 && w_ms > 0.0)
          ? (w_ms / static_cast<double>(K * done_w))
          : 10.0;
      while (done_w >= first_wave) {
        const double total_elapsed_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - s1_t0).count();
        if (total_elapsed_ms >= 0.935 * max_time_ms) break;
        const double rem_ms = std::max(0.0, 0.968 * max_time_ms - total_elapsed_ms);
        const double est_sims = rem_ms / std::max(0.05, cur_ms_per_sim);
        int wave_worlds = std::clamp(
            static_cast<int>(est_sims / static_cast<double>(K)), 2, 512);
        const int before_w = static_cast<int>(per_candidate_rollouts[0].size());
        const auto wt0 = std::chrono::steady_clock::now();
        RunClockedPairedRolloutStage(
            state, dune, search_player, evaluators, ordered_candidates,
            candidate_indices, wave_worlds, /*world_seed_offset=*/seed_offset,
            decision_seed, crn_mode, sample_opponent_leader_draft,
            deadline_tp, active_tree, per_candidate_rollouts,
            new_rollouts_executed, max_inflight_worlds,
            resample_from_infostate);
        const int kept_w = static_cast<int>(per_candidate_rollouts[0].size()) - before_w;
        const double wt_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - wt0).count();
        seed_offset += wave_worlds;
        if (kept_w < wave_worlds || kept_w <= 0) break;
        if (wt_ms > 0.0) {
          cur_ms_per_sim = 0.3 * cur_ms_per_sim +
                           0.7 * (wt_ms / static_cast<double>(K * kept_w));
        }
      }
    }
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
  res.stage1_planned_worlds = std::max(rollouts_per_action, r_completed);
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
    int max_inflight_worlds = 32,
    const ActionsAndProbs* precomputed_raw_prior = nullptr) {
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
    int max_inflight_worlds = 32,
    const ActionsAndProbs* precomputed_raw_prior = nullptr) {
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
    int fixed_top_k = 0,
    int max_inflight_worlds = 32,
    bool use_bounded_pool_when_unclocked = false,
    bool full_clock_stage2 = false,
    double prior_log_ratio_weight = 0.0,
    double stage1_high_conf_prior_skip = 0.0,
    const ActionsAndProbs* precomputed_raw_prior = nullptr) {
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

  std::vector<Action> ordered_candidates;
  std::vector<double> candidate_priors;
  if (fixed_top_k <= 0) {
    ordered_candidates.reserve(4);
    candidate_priors.reserve(4);
    ordered_candidates.push_back(sorted_by_prior[0].first);
    candidate_priors.push_back(sorted_by_prior[0].second);
    ordered_candidates.push_back(sorted_by_prior[1].first);
    candidate_priors.push_back(sorted_by_prior[1].second);

    if (sorted_by_prior.size() >= 3 && sorted_by_prior[2].second >= 0.025) {
      ordered_candidates.push_back(sorted_by_prior[2].first);
      candidate_priors.push_back(sorted_by_prior[2].second);
      if (sorted_by_prior.size() >= 4 && sorted_by_prior[3].second >= 0.08) {
        ordered_candidates.push_back(sorted_by_prior[3].first);
        candidate_priors.push_back(sorted_by_prior[3].second);
      }
    }
  } else {
    const size_t k_max = std::min<size_t>(static_cast<size_t>(fixed_top_k), sorted_by_prior.size());
    ordered_candidates.reserve(k_max);
    candidate_priors.reserve(k_max);
    for (size_t i = 0; i < k_max; ++i) {
      ordered_candidates.push_back(sorted_by_prior[i].first);
      candidate_priors.push_back(sorted_by_prior[i].second);
    }
  }

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

  const auto* dune = dynamic_cast<const DuneImperiumState*>(&state);
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
  const double s1_deadline_frac = full_clock_stage2 ? 0.56 : 0.50;
  const double final_deadline_frac = full_clock_stage2 ? 1.008 : 1.00;
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

  int stage1_planned_worlds = r_stage1;
  int stage1_kept_worlds = 0;
  const auto s1_t0 = std::chrono::steady_clock::now();

  // --- STAGE 1: Broad Screen ---
  if (!use_bounded_stage) {
    std::vector<RolloutTask> tasks_stage1;
    tasks_stage1.reserve(K * r_stage1);

    for (size_t c_idx = 0; c_idx < K; ++c_idx) {
      const Action cand_a = ordered_candidates[c_idx];
      for (int r = 0; r < r_stage1; ++r) {
        uint64_t wseed = dune_seed::DeriveSeed(decision_seed, static_cast<uint64_t>(r));
        std::unique_ptr<State> prebuilt;
        if (dune == nullptr) {
          prebuilt = state.Clone();
          prebuilt->ApplyAction(cand_a);
        }
        tasks_stage1.push_back({c_idx, r, cand_a, std::move(prebuilt), wseed});
      }
    }

    new_rollouts_executed += static_cast<int>(tasks_stage1.size());

    if (!tasks_stage1.empty()) {
      std::vector<std::future<RolloutTrajectory>> futures;
      futures.reserve(tasks_stage1.size());

      for (size_t i = 0; i < tasks_stage1.size(); ++i) {
        auto* active_evaluator = evaluators.empty() ? nullptr : evaluators[i % evaluators.size()].get();
        if (active_evaluator != nullptr) {
          active_evaluator->AdjustActiveRollouts(1);
        }
      }

      const State* root_state_ptr = &state;
      for (size_t i = 0; i < tasks_stage1.size(); ++i) {
        auto* prebuilt_ptr = tasks_stage1[i].sim_state.get();
        uint64_t wseed = tasks_stage1[i].wseed;
        Action cand_a = tasks_stage1[i].action;
        auto* active_evaluator = evaluators.empty() ? nullptr : evaluators[i % evaluators.size()].get();
        futures.push_back(SearchRolloutThreadPool::Instance().Submit(
            [prebuilt_ptr, root_state_ptr, dune, cand_a, wseed, search_player, active_evaluator, crn_mode]() {
              ActiveRolloutReleaseGuard guard{active_evaluator};
              std::unique_ptr<State> sim_state;
              State* s_ptr = prebuilt_ptr;
              if (s_ptr == nullptr) {
                if (dune != nullptr) {
                  std::mt19937 wrng(wseed);
                  auto rng_func = [&wrng]() {
                    return std::generate_canonical<double, 53>(wrng);
                  };
                  sim_state = dune->ResampleFromInfostate(search_player, rng_func);
                }
                if (!sim_state) {
                  sim_state = root_state_ptr->Clone();
                }
                sim_state->ApplyAction(cand_a);
                s_ptr = sim_state.get();
              }
              return SimulateRolloutWithTurnTrajectory(
                  s_ptr, search_player, active_evaluator, wseed + 999,
                  SearchTruncationMode::kTerminal, crn_mode);
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
    const int s1_inflight = full_clock_stage2
        ? std::max(max_inflight_worlds, 384 / static_cast<int>(K))
        : max_inflight_worlds;

    RunClockedPairedRolloutStage(
        state, dune, search_player, evaluators, ordered_candidates,
        candidate_indices, r_stage1, /*world_seed_offset=*/0,
        decision_seed, crn_mode, /*sample_opponent_leader_draft=*/false,
        stage1_deadline, active_tree, per_candidate_rollouts,
        new_rollouts_executed, s1_inflight);
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

  if (standard_early_stop || high_conf_skip) {
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
    res.termination_reason = high_conf_skip ? "stage1_high_conf_skip" : "early_stop";
    return res;
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
    std::vector<RolloutTask> tasks_stage2;
    tasks_stage2.reserve(2 * r_stage2);

    for (size_t f_i = 0; f_i < finalists.size(); ++f_i) {
      size_t f_idx = finalists[f_i];
      const Action cand_a = ordered_candidates[f_idx];
      for (int r = 0; r < r_stage2; ++r) {
        uint64_t wseed = dune_seed::DeriveSeed(decision_seed, static_cast<uint64_t>(r_stage1 + r));
        std::unique_ptr<State> prebuilt;
        if (dune == nullptr) {
          prebuilt = state.Clone();
          prebuilt->ApplyAction(cand_a);
        }
        tasks_stage2.push_back({f_idx, r_stage1 + r, cand_a, std::move(prebuilt), wseed});
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
        auto* active_evaluator = evaluators.empty() ? nullptr : evaluators[i % evaluators.size()].get();
        if (active_evaluator != nullptr) {
          active_evaluator->AdjustActiveRollouts(1);
        }
      }

      const State* root_state_ptr = &state;
      for (size_t i = 0; i < tasks_stage2.size(); ++i) {
        auto* prebuilt_ptr = tasks_stage2[i].sim_state.get();
        uint64_t wseed = tasks_stage2[i].wseed;
        Action cand_a = tasks_stage2[i].action;
        auto* active_evaluator = evaluators.empty() ? nullptr : evaluators[i % evaluators.size()].get();
        futures.push_back(SearchRolloutThreadPool::Instance().Submit(
            [prebuilt_ptr, root_state_ptr, dune, cand_a, wseed, search_player, active_evaluator, crn_mode]() {
              ActiveRolloutReleaseGuard guard{active_evaluator};
              std::unique_ptr<State> sim_state;
              State* s_ptr = prebuilt_ptr;
              if (s_ptr == nullptr) {
                if (dune != nullptr) {
                  std::mt19937 wrng(wseed);
                  auto rng_func = [&wrng]() {
                    return std::generate_canonical<double, 53>(wrng);
                  };
                  sim_state = dune->ResampleFromInfostate(search_player, rng_func);
                }
                if (!sim_state) {
                  sim_state = root_state_ptr->Clone();
                }
                sim_state->ApplyAction(cand_a);
                s_ptr = sim_state.get();
              }
              return SimulateRolloutWithTurnTrajectory(
                  s_ptr, search_player, active_evaluator, wseed + 999,
                  SearchTruncationMode::kTerminal, crn_mode);
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
    const int s2_inflight = full_clock_stage2 ? std::max(max_inflight_worlds, 256) : std::max(max_inflight_worlds, 194);
    if (!full_clock_stage2 || !has_deadline) {
      RunClockedPairedRolloutStage(
          state, dune, search_player, evaluators, ordered_candidates,
          stage2_cands, r_stage2, /*world_seed_offset=*/r_stage1,
          decision_seed, crn_mode, /*sample_opponent_leader_draft=*/false,
          final_deadline, active_tree, per_candidate_rollouts,
          new_rollouts_executed, s2_inflight);
    } else {
      stage2_planned_worlds = 0;
      int seed_offset = r_stage1;
      double cur_ms_per_sim =
          stage1_elapsed_ms / std::max(1.0, static_cast<double>(K * r_stage1_done));
      while (true) {
        const double total_elapsed_ms =
            std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - start_time).count();
        const int s2_done_now =
            static_cast<int>(per_candidate_rollouts[0].size()) - r_stage1_done;
        if (s2_done_now >= 2 && total_elapsed_ms >= 0.935 * max_time_ms) {
          break;
        }
        const double rem_ms = std::max(0.0, 0.968 * max_time_ms - total_elapsed_ms);
        const double wave_target_ms = (s2_done_now == 0 && rem_ms < 4500.0)
            ? (0.65 * rem_ms)
            : rem_ms;
        const double est_sims = wave_target_ms / std::max(0.05, cur_ms_per_sim);
        const int wave_worlds = std::clamp(
            static_cast<int>(est_sims / 2.0), 2, 1024);
        stage2_planned_worlds += wave_worlds;
        const int before_size = static_cast<int>(per_candidate_rollouts[0].size());
        const auto wt0 = std::chrono::steady_clock::now();
        RunClockedPairedRolloutStage(
            state, dune, search_player, evaluators, ordered_candidates,
            stage2_cands, wave_worlds, /*world_seed_offset=*/seed_offset,
            decision_seed, crn_mode, /*sample_opponent_leader_draft=*/false,
            final_deadline, active_tree, per_candidate_rollouts,
            new_rollouts_executed, s2_inflight);
        const int wave_kept =
            static_cast<int>(per_candidate_rollouts[0].size()) - before_size;
        const double wt_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - wt0).count();
        seed_offset += wave_worlds;
        if (wave_kept < wave_worlds || wave_kept <= 0) {
          break;
        }
        if (wt_ms > 0.0) {
          cur_ms_per_sim = 0.25 * cur_ms_per_sim +
                           0.75 * (wt_ms / static_cast<double>(2 * wave_kept));
        }
      }
    }
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

  const double effective_margin = min_override_margin + prior_penalty[best_c_idx];
  bool should_override = false;
  if (!unbiased_stage2_gate) {
    should_override =
        (mean_d_stage2 > 0.0 && mean_d_total > effective_margin && t_stat_total >= 1.25);
  } else {
    if (K > 2) {
      should_override =
          (mean_d_stage2 >= effective_margin && t_stat_stage2 >= 1.25);
    } else {
      should_override =
          (mean_d_stage2 > 0.0 && mean_d_total >= effective_margin && t_stat_total >= 1.25);
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
  res.mean_d_stage2 = mean_d_stage2;
  res.se_d_stage2 = se_d_stage2;
  return res;
}

inline constexpr int kNumBoardSpaces = 22;

inline bool IsBoardSpaceAction(Action a) {
  return a >= kActionAgentSpaceConspire && a <= kActionAgentSpaceSietchTabr;
}

inline bool IsGraftSelectionAction(Action a) {
  return (a >= kActionSelectGraftPartner0 &&
          a < kActionSelectGraftPartner0 + kNumImperiumCardIds) ||
         a == kActionPlayAgentSolo;
}

struct JointAgentCandidate {
  Action first_action = kInvalidAction;
  Action graft_action = kInvalidAction;
  Action second_action = kInvalidAction;  // kInvalidAction for standalone 1-step root actions (e.g. Reveal, PlotIntrigue)
  double joint_prior = 0.0;
};

struct JointSearchV2Result {
  CompoundSearchDecisionResult base;
  Action chosen_first_action = kInvalidAction;
  Action chosen_graft_action = kInvalidAction;
  Action chosen_second_action = kInvalidAction;
  Action raw_first_action = kInvalidAction;
  Action raw_graft_action = kInvalidAction;
  Action raw_second_action = kInvalidAction;
  std::vector<JointAgentCandidate> joint_candidates;
};

inline std::vector<JointAgentCandidate> EnumerateJointAgentCandidates(
    const State& state,
    Player search_player,
    const std::vector<std::shared_ptr<BatchedNNEvaluator>>& evaluators,
    const ActionsAndProbs& root_prior,
    size_t max_candidates = 8) {
  const std::vector<Action> legal = state.LegalActions();
  if (legal.empty()) return {};

  auto get_prior_p = [](const ActionsAndProbs& prior, Action a) -> double {
    for (const auto& ap : prior) {
      if (ap.first == a) return ap.second;
    }
    return 0.0;
  };

  const Action a1_raw = PickGreedyAction(root_prior, legal);

  int non_spec_count = 0;
  for (Action a : legal) {
    if (!IsConvertSpecimenAction(a)) {
      ++non_spec_count;
    }
  }
  const bool expand_all = (non_spec_count < 4);

  struct CandRecord {
    Action a1 = kInvalidAction;
    Action a_graft = kInvalidAction;
    Action a2 = kInvalidAction;
    double joint_p = 0.0;
    double p_s1 = 0.0;
    bool is_space = false;
  };

  std::vector<CandRecord> pool;
  ActionsAndProbs prior_s1_raw;
  std::vector<Action> l2_raw;
  bool a1_raw_is_space = false;
  Action a_graft_raw = kInvalidAction;
  Action a2_raw = kInvalidAction;
  double raw_joint_p = get_prior_p(root_prior, a1_raw);

  for (Action a1 : legal) {
    if (IsConvertSpecimenAction(a1) && a1 != a1_raw) {
      continue;
    }
    const double p_a1 = get_prior_p(root_prior, a1);
    std::unique_ptr<State> s1 = state.Clone();
    s1->ApplyAction(a1);

    if (!s1->IsTerminal() && !s1->IsChanceNode() && s1->CurrentPlayer() == search_player) {
      const auto l2 = s1->LegalActions();
      if (!l2.empty() && std::all_of(l2.begin(), l2.end(), [](Action a) { return IsBoardSpaceAction(a); })) {
        if (l2.size() == 1) {
          pool.push_back({a1, kInvalidAction, l2[0], p_a1, 1.0, true});
          if (a1 == a1_raw) {
            a_graft_raw = kInvalidAction;
            a2_raw = l2[0];
            raw_joint_p = p_a1;
            l2_raw = l2;
            a1_raw_is_space = true;
          }
        } else if (l2.size() >= 2) {
          if (a1 == a1_raw || p_a1 >= 0.005 || expand_all) {
            ActionsAndProbs prior_s1 = evaluators.empty() ? ActionsAndProbs{} : evaluators.front()->Prior(*s1);
            if (a1 == a1_raw) {
              a_graft_raw = kInvalidAction;
              prior_s1_raw = prior_s1;
              l2_raw = l2;
              a1_raw_is_space = true;
              a2_raw = PickGreedyAction(prior_s1, l2);
              raw_joint_p = p_a1 * get_prior_p(prior_s1, a2_raw);
            }
            for (Action a2 : l2) {
              const double p_a2 = get_prior_p(prior_s1, a2);
              pool.push_back({a1, kInvalidAction, a2, p_a1 * p_a2, p_a2, true});
            }
          }
        }
      } else if (!l2.empty() && std::all_of(l2.begin(), l2.end(), [](Action a) { return IsGraftSelectionAction(a); })) {
        if (a1 == a1_raw || p_a1 >= 0.005 || expand_all) {
          ActionsAndProbs prior_graft = evaluators.empty() ? ActionsAndProbs{} : evaluators.front()->Prior(*s1);
          Action a_graft_0 = PickGreedyAction(prior_graft, l2);
          std::vector<Action> chosen_grafts = {a_graft_0};
          Action alt_graft = kInvalidAction;
          double alt_graft_p = -1.0;
          for (Action ag : l2) {
            if (ag == a_graft_0) continue;
            double pag = get_prior_p(prior_graft, ag);
            if (pag >= 0.05 && pag > alt_graft_p) {
              alt_graft_p = pag;
              alt_graft = ag;
            }
          }
          if (alt_graft != kInvalidAction) {
            chosen_grafts.push_back(alt_graft);
          }

          for (Action a_graft : chosen_grafts) {
            const double p_graft = get_prior_p(prior_graft, a_graft);
            auto s_g = s1->Clone();
            s_g->ApplyAction(a_graft);
            if (!s_g->IsTerminal() && !s_g->IsChanceNode() && s_g->CurrentPlayer() == search_player) {
              const auto l_space = s_g->LegalActions();
              if (!l_space.empty() && std::all_of(l_space.begin(), l_space.end(), [](Action a) { return IsBoardSpaceAction(a); })) {
                if (l_space.size() == 1) {
                  pool.push_back({a1, a_graft, l_space[0], p_a1 * p_graft, 1.0, true});
                  if (a1 == a1_raw && a_graft == a_graft_0) {
                    a_graft_raw = a_graft_0;
                    a2_raw = l_space[0];
                    raw_joint_p = p_a1 * p_graft;
                    l2_raw = l_space;
                    a1_raw_is_space = true;
                  }
                } else {
                  ActionsAndProbs prior_space = evaluators.empty() ? ActionsAndProbs{} : evaluators.front()->Prior(*s_g);
                  if (a1 == a1_raw && a_graft == a_graft_0) {
                    a_graft_raw = a_graft_0;
                    a2_raw = PickGreedyAction(prior_space, l_space);
                    raw_joint_p = p_a1 * p_graft * get_prior_p(prior_space, a2_raw);
                    prior_s1_raw = prior_space;
                    l2_raw = l_space;
                    a1_raw_is_space = true;
                  }
                  for (Action a2 : l_space) {
                    const double p_a2 = get_prior_p(prior_space, a2);
                    pool.push_back({a1, a_graft, a2, p_a1 * p_graft * p_a2, p_a2, true});
                  }
                }
              } else {
                pool.push_back({a1, a_graft, kInvalidAction, p_a1 * p_graft, 1.0, false});
                if (a1 == a1_raw && a_graft == a_graft_0) {
                  a_graft_raw = a_graft_0;
                  a2_raw = kInvalidAction;
                  raw_joint_p = p_a1 * p_graft;
                }
              }
            }
          }
        } else {
          pool.push_back({a1, kInvalidAction, kInvalidAction, p_a1, 1.0, false});
        }
      } else {
        pool.push_back({a1, kInvalidAction, kInvalidAction, p_a1, 1.0, false});
      }
    } else {
      pool.push_back({a1, kInvalidAction, kInvalidAction, p_a1, 1.0, false});
    }
  }

  std::vector<JointAgentCandidate> result;
  result.reserve(max_candidates);

  auto already_included = [&](Action a1, Action a_graft, Action a2) {
    for (const auto& c : result) {
      if (c.first_action == a1 && c.graft_action == a_graft && c.second_action == a2) return true;
    }
    return false;
  };

  // 1. Index 0 MUST be exact sequential greedy baseline (a1_raw, a_graft_raw, a2_raw)
  result.push_back({a1_raw, a_graft_raw, a2_raw, raw_joint_p});

  // 2. Guarantee diversity across cards + spaces before filling by joint_prior:
  // 2a. Second-highest-prior space for a1_raw if p(a2_second | s1_raw) >= 0.025
  if (a1_raw_is_space && l2_raw.size() >= 2) {
    std::vector<std::pair<double, Action>> sorted_s1_spaces;
    sorted_s1_spaces.reserve(l2_raw.size());
    for (Action a2 : l2_raw) {
      sorted_s1_spaces.push_back({get_prior_p(prior_s1_raw, a2), a2});
    }
    std::stable_sort(sorted_s1_spaces.begin(), sorted_s1_spaces.end(),
                     [](const auto& lhs, const auto& rhs) {
                       if (lhs.first != rhs.first) return lhs.first > rhs.first;
                       return lhs.second < rhs.second;
                     });
    if (sorted_s1_spaces.size() >= 2 && sorted_s1_spaces[1].first >= 0.025) {
      Action a2_second = sorted_s1_spaces[1].second;
      if (!already_included(a1_raw, a_graft_raw, a2_second) && result.size() < max_candidates) {
        double p_g = (a_graft_raw != kInvalidAction)
            ? (raw_joint_p / std::max(1e-12, get_prior_p(root_prior, a1_raw) * get_prior_p(prior_s1_raw, a2_raw)))
            : 1.0;
        double j_p = get_prior_p(root_prior, a1_raw) * p_g * sorted_s1_spaces[1].first;
        result.push_back({a1_raw, a_graft_raw, a2_second, j_p});
      }
    }
  }

  // 2b. Include top-prior follow-up for up to 4 distinct non-specimen root actions a1_alt != a1_raw
  // (provided p(a1_alt | state) >= 0.015)
  std::vector<std::pair<double, Action>> sorted_alt_roots;
  for (Action a1 : legal) {
    if (a1 == a1_raw || IsConvertSpecimenAction(a1)) continue;
    const double p_a1 = get_prior_p(root_prior, a1);
    if (p_a1 >= 0.015) {
      sorted_alt_roots.push_back({p_a1, a1});
    }
  }
  std::stable_sort(sorted_alt_roots.begin(), sorted_alt_roots.end(),
                   [](const auto& lhs, const auto& rhs) {
                     if (lhs.first != rhs.first) return lhs.first > rhs.first;
                     return lhs.second < rhs.second;
                   });

  const size_t max_alts = std::min<size_t>(4, sorted_alt_roots.size());
  for (size_t i = 0; i < max_alts; ++i) {
    if (result.size() >= max_candidates) break;
    const Action a1_alt = sorted_alt_roots[i].second;
    const CandRecord* best_rec = nullptr;
    for (const auto& rec : pool) {
      if (rec.a1 == a1_alt) {
        if (!best_rec || rec.joint_p > best_rec->joint_p) {
          best_rec = &rec;
        }
      }
    }
    if (best_rec && !already_included(best_rec->a1, best_rec->a_graft, best_rec->a2)) {
      result.push_back({best_rec->a1, best_rec->a_graft, best_rec->a2, best_rec->joint_p});
    }
  }

  // 2c. Fill remaining slots up to max_candidates in descending order of joint_prior
  // (filtering joint_prior >= 0.008 unless needed to reach at least 2 candidates)
  std::vector<CandRecord> rem_pool = pool;
  std::stable_sort(rem_pool.begin(), rem_pool.end(),
                   [](const auto& lhs, const auto& rhs) {
                     if (lhs.joint_p != rhs.joint_p) return lhs.joint_p > rhs.joint_p;
                     if (lhs.a1 != rhs.a1) return lhs.a1 < rhs.a1;
                     if (lhs.a_graft != rhs.a_graft) return lhs.a_graft < rhs.a_graft;
                     return lhs.a2 < rhs.a2;
                   });

  for (const auto& rec : rem_pool) {
    if (result.size() >= max_candidates) break;
    if (rec.joint_p < 0.008 && result.size() >= 2) break;
    if (!already_included(rec.a1, rec.a_graft, rec.a2)) {
      result.push_back({rec.a1, rec.a_graft, rec.a2, rec.joint_p});
    }
  }

  return result;
}

inline std::pair<std::vector<Action>, std::vector<double>>
SelectOrderedCandidatesAndPriors(
    const ActionsAndProbs& raw_prior,
    const std::vector<Action>& legal,
    int fixed_top_k = 0) {
  const auto sorted_by_prior = GetSortedCandidatesByPrior(raw_prior, legal);
  std::vector<Action> ordered_candidates;
  std::vector<double> candidate_priors;
  if (sorted_by_prior.empty()) return {ordered_candidates, candidate_priors};
  if (fixed_top_k <= 0) {
    ordered_candidates.reserve(4);
    candidate_priors.reserve(4);
    ordered_candidates.push_back(sorted_by_prior[0].first);
    candidate_priors.push_back(sorted_by_prior[0].second);
    if (sorted_by_prior.size() >= 2) {
      ordered_candidates.push_back(sorted_by_prior[1].first);
      candidate_priors.push_back(sorted_by_prior[1].second);
    }
    if (sorted_by_prior.size() >= 3 && sorted_by_prior[2].second >= 0.025) {
      ordered_candidates.push_back(sorted_by_prior[2].first);
      candidate_priors.push_back(sorted_by_prior[2].second);
      if (sorted_by_prior.size() >= 4 && sorted_by_prior[3].second >= 0.08) {
        ordered_candidates.push_back(sorted_by_prior[3].first);
        candidate_priors.push_back(sorted_by_prior[3].second);
      }
    }
  } else {
    const size_t k_max = std::min<size_t>(static_cast<size_t>(fixed_top_k), sorted_by_prior.size());
    ordered_candidates.reserve(k_max);
    candidate_priors.reserve(k_max);
    for (size_t i = 0; i < k_max; ++i) {
      ordered_candidates.push_back(sorted_by_prior[i].first);
      candidate_priors.push_back(sorted_by_prior[i].second);
    }
  }
  return {ordered_candidates, candidate_priors};
}

inline CompoundSearchDecisionResult SearchV2SuccessiveHalvingDecision(
    const State& state,
    Player search_player,
    TurnSearchTree* turn_tree,
    const std::vector<std::shared_ptr<BatchedNNEvaluator>>& evaluators,
    const std::vector<Action>& ordered_candidates,
    const std::vector<double>& candidate_priors,
    int reference_rollouts_per_action,
    double min_override_margin,
    uint64_t decision_seed,
    SearchRolloutCrnMode crn_mode = SearchRolloutCrnMode::kSequential,
    double max_time_ms = -1.0,
    int max_inflight_worlds = 32,
    double prior_log_ratio_weight = 0.02,
    double stage1_high_conf_prior_skip = 0.95,
    const std::vector<Action>* second_actions = nullptr,
    const std::vector<Action>* graft_actions = nullptr,
    double est_ms_per_sim = 20.0,
    int* chosen_candidate_index = nullptr) {
  if (chosen_candidate_index != nullptr) {
    *chosen_candidate_index = 0;
  }
  if (turn_tree != nullptr) turn_tree->BindPublicCardKnowledge(state);
  if (ordered_candidates.empty()) {
    return {kInvalidAction, false, 0.0, 0.0, 0, 0, {}, {}};
  }
  if (ordered_candidates.size() == 1 || evaluators.empty()) {
    return {ordered_candidates.front(), false, 0.0, 0.0, 0, 0, {}, {}};
  }

  const size_t K = ordered_candidates.size();
  const Action raw_action = ordered_candidates[0];

  std::vector<double> prior_penalty(K, 0.0);
  for (size_t c_idx = 1; c_idx < K; ++c_idx) {
    if (prior_log_ratio_weight > 0.0 && !candidate_priors.empty() && c_idx < candidate_priors.size()) {
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

  const auto* dune = dynamic_cast<const DuneImperiumState*>(&state);
  TurnSearchTree local_tree;
  TurnSearchTree* active_tree = (turn_tree != nullptr) ? turn_tree : &local_tree;

  const int total_budget = 3 * reference_rollouts_per_action;
  const bool has_deadline = (max_time_ms > 0.0);
  const auto start_time = std::chrono::steady_clock::now();

  // 1. Fast Capped Stage-1 Screen (max_stage1_worlds = 16)
  int r_stage1 = 2;
  if (!has_deadline) {
    r_stage1 = std::clamp(total_budget / (2 * static_cast<int>(K)), 2, 16);
  } else {
    r_stage1 = std::clamp(
        static_cast<int>((0.35 * max_time_ms) / std::max(0.5, static_cast<double>(K) * est_ms_per_sim)),
        2, 16);
  }

  const auto stage1_deadline = has_deadline
      ? (start_time +
         std::chrono::duration_cast<std::chrono::steady_clock::duration>(
             std::chrono::duration<double, std::milli>(0.58 * max_time_ms)))
      : std::chrono::steady_clock::time_point::max();

  const auto final_deadline = has_deadline
      ? (start_time +
         std::chrono::duration_cast<std::chrono::steady_clock::duration>(
             std::chrono::duration<double, std::milli>(0.985 * max_time_ms)))
      : std::chrono::steady_clock::time_point::max();

  // CRITICAL ANTI-STARVATION FIX:
  // Cap Stage-1 s1_inflight worlds by std::clamp(r_stage1, 2, std::max(2, std::min(max_inflight_worlds, 192 / static_cast<int>(K))))
  const int s1_cap = std::max(2, std::min(max_inflight_worlds, 192 / static_cast<int>(K)));
  const int s1_inflight = std::clamp(r_stage1, 2, s1_cap);

  std::vector<size_t> candidate_indices(K);
  std::iota(candidate_indices.begin(), candidate_indices.end(), 0);

  std::vector<std::vector<double>> per_candidate_rollouts(K);
  for (size_t c = 0; c < K; ++c) {
    per_candidate_rollouts[c].reserve(r_stage1);
  }
  int new_rollouts_executed = 0;
  const auto s1_t0 = std::chrono::steady_clock::now();

  if (has_deadline) {
    // Sub-wave 1A (min_worlds = 2): ALWAYS runs first with min_s1_deadline = start_time + 0.92 * max_time_ms and inflight = 2
    const auto min_s1_deadline = start_time +
        std::chrono::duration_cast<std::chrono::steady_clock::duration>(
            std::chrono::duration<double, std::milli>(0.92 * max_time_ms));
    RunClockedPairedRolloutStage(
        state, dune, search_player, evaluators, ordered_candidates,
        candidate_indices, 2, /*world_seed_offset=*/0,
        decision_seed, crn_mode, /*sample_opponent_leader_draft=*/false,
        min_s1_deadline, active_tree, per_candidate_rollouts,
        new_rollouts_executed, /*max_inflight_worlds=*/2, /*resample_from_infostate=*/true,
        second_actions, graft_actions);

    const double elapsed_1a = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - s1_t0).count();
    const int rem_w = r_stage1 - 2;

    // Sub-wave 1B (rem_w = r_stage1 - 2): runs ONLY if rem_w > 0, 2 worlds completed in 1A, and elapsed_1a < 0.40 * max_time_ms
    if (rem_w > 0 && !per_candidate_rollouts.empty() &&
        per_candidate_rollouts[0].size() == 2 && elapsed_1a < 0.40 * max_time_ms) {
      RunClockedPairedRolloutStage(
          state, dune, search_player, evaluators, ordered_candidates,
          candidate_indices, rem_w, /*world_seed_offset=*/2,
          decision_seed, crn_mode, /*sample_opponent_leader_draft=*/false,
          stage1_deadline, active_tree, per_candidate_rollouts,
          new_rollouts_executed, s1_inflight, /*resample_from_infostate=*/true,
          second_actions, graft_actions);
    }
  } else {
    RunClockedPairedRolloutStage(
        state, dune, search_player, evaluators, ordered_candidates,
        candidate_indices, r_stage1, /*world_seed_offset=*/0,
        decision_seed, crn_mode, /*sample_opponent_leader_draft=*/false,
        stage1_deadline, active_tree, per_candidate_rollouts,
        new_rollouts_executed, s1_inflight, /*resample_from_infostate=*/true,
        second_actions, graft_actions);
  }

  const auto s1_t1 = std::chrono::steady_clock::now();
  const double stage1_elapsed_ms =
      std::chrono::duration<double, std::milli>(s1_t1 - s1_t0).count();

  const int r_stage1_done = per_candidate_rollouts.empty() ? 0 : static_cast<int>(per_candidate_rollouts[0].size());

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
    res.stage1_planned_worlds = r_stage1;
    res.stage1_kept_worlds = r_stage1_done;
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

  // 2. Smart Stage-1 Early-Stop
  const double best_challenger_adv_1 = mean_d_1[best_c_idx] - prior_penalty[best_c_idx];
  const bool high_conf_skip =
      (stage1_high_conf_prior_skip > 0.0 &&
       !candidate_priors.empty() &&
       candidate_priors[0] >= stage1_high_conf_prior_skip &&
       best_challenger_adv_1 <= 0.05);
  const bool blowout_skip = (r_stage1_done >= 12 && best_challenger_adv_1 < -0.20);
  const bool tight_clock_skip =
      (has_deadline && stage1_elapsed_ms >= 0.78 * max_time_ms && best_challenger_adv_1 <= 0.0);

  if (high_conf_skip || blowout_skip || tight_clock_skip) {
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
    res.stage1_planned_worlds = r_stage1;
    res.stage1_kept_worlds = r_stage1_done;
    res.stage2_planned_worlds = 0;
    res.stage2_kept_worlds = 0;
    res.stage1_elapsed_ms = stage1_elapsed_ms;
    res.stage2_elapsed_ms = 0.0;
    res.termination_reason = high_conf_skip ? "stage1_high_conf_skip" : "early_stop";
    return res;
  }

  // 3. Adaptive Stage-2 Wave Ramp + High-Confidence Convergence Stop
  const std::vector<size_t> stage2_cands = {0, best_c_idx};
  int stage2_planned_worlds = 0;
  int stage2_kept_worlds = 0;
  const auto s2_t0 = std::chrono::steady_clock::now();

  const double effective_margin = min_override_margin + prior_penalty[best_c_idx];

  if (!has_deadline) {
    int r_stage2 = std::max(16, (total_budget - static_cast<int>(K) * r_stage1_done) / 2);
    stage2_planned_worlds = r_stage2;
    int s2_inflight = std::min(r_stage2, std::min(max_inflight_worlds, 96));
    RunClockedPairedRolloutStage(
        state, dune, search_player, evaluators, ordered_candidates,
        stage2_cands, r_stage2, /*world_seed_offset=*/r_stage1_done,
        decision_seed, crn_mode, /*sample_opponent_leader_draft=*/false,
        final_deadline, active_tree, per_candidate_rollouts,
        new_rollouts_executed, s2_inflight, /*resample_from_infostate=*/true,
        second_actions, graft_actions);
  } else {
    int seed_offset = r_stage1_done;
    double cur_ms_per_sim = (r_stage1_done > 0 && stage1_elapsed_ms > 0.0)
        ? (stage1_elapsed_ms / static_cast<double>(K * r_stage1_done))
        : est_ms_per_sim;
    while (true) {
      const double total_elapsed_ms = std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - start_time).count();
      const int s2_done_now = static_cast<int>(per_candidate_rollouts[0].size()) - r_stage1_done;
      if (s2_done_now >= 2 && total_elapsed_ms >= 0.935 * max_time_ms) {
        break;
      }
      const double rem_ms = std::max(0.0, 0.985 * max_time_ms - total_elapsed_ms);
      if (rem_ms <= 0.0) break;

      int wave_worlds = 0;
      if (s2_done_now == 0) {
        int max_w0 = (rem_ms < 4000.0 ? 12 : 48);
        wave_worlds = std::clamp(static_cast<int>((0.45 * rem_ms) / std::max(0.1, 2.0 * cur_ms_per_sim)), 2, max_w0);
      } else {
        wave_worlds = std::clamp(static_cast<int>(rem_ms / std::max(0.1, 2.0 * cur_ms_per_sim)), 2, 128);
      }
      int s2_inflight = std::min(wave_worlds, std::min(max_inflight_worlds, 96));

      stage2_planned_worlds += wave_worlds;
      const int before_size = static_cast<int>(per_candidate_rollouts[0].size());
      const auto wt0 = std::chrono::steady_clock::now();
      RunClockedPairedRolloutStage(
          state, dune, search_player, evaluators, ordered_candidates,
          stage2_cands, wave_worlds, /*world_seed_offset=*/seed_offset,
          decision_seed, crn_mode, /*sample_opponent_leader_draft=*/false,
          final_deadline, active_tree, per_candidate_rollouts,
          new_rollouts_executed, s2_inflight, /*resample_from_infostate=*/true,
          second_actions, graft_actions);
      const int wave_kept = static_cast<int>(per_candidate_rollouts[0].size()) - before_size;
      const double wt_ms = std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - wt0).count();
      seed_offset += wave_worlds;
      if (wave_kept < wave_worlds || wave_kept <= 0) {
        break;
      }
      if (wt_ms > 0.0) {
        cur_ms_per_sim = 0.25 * cur_ms_per_sim +
                         0.75 * (wt_ms / static_cast<double>(2 * wave_kept));
      }

      const int cur_s2_done = static_cast<int>(per_candidate_rollouts[0].size()) - r_stage1_done;
      if (cur_s2_done >= 128) {
        double sum_d_s2 = 0.0;
        for (int r = 0; r < cur_s2_done; ++r) {
          sum_d_s2 += (per_candidate_rollouts[best_c_idx][r_stage1_done + r] -
                       per_candidate_rollouts[0][r_stage1_done + r]);
        }
        const double mean_d_s2 = sum_d_s2 / cur_s2_done;
        double sq_diff_s2 = 0.0;
        for (int r = 0; r < cur_s2_done; ++r) {
          double diff = (per_candidate_rollouts[best_c_idx][r_stage1_done + r] -
                         per_candidate_rollouts[0][r_stage1_done + r]) - mean_d_s2;
          sq_diff_s2 += diff * diff;
        }
        const double var_d_s2 = sq_diff_s2 / (cur_s2_done - 1);
        const double se_d_s2 = std::sqrt(var_d_s2 / cur_s2_done);
        if (se_d_s2 > 1e-9 &&
            (mean_d_s2 < effective_margin - 3.5 * se_d_s2 ||
             (mean_d_s2 > effective_margin + 3.5 * se_d_s2 && mean_d_s2 / se_d_s2 >= 2.5))) {
          break;
        }
      }
    }
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
    res.stage1_planned_worlds = r_stage1;
    res.stage1_kept_worlds = r_stage1_done;
    res.stage2_planned_worlds = stage2_planned_worlds;
    res.stage2_kept_worlds = stage2_kept_worlds;
    res.stage1_elapsed_ms = stage1_elapsed_ms;
    res.stage2_elapsed_ms = stage2_elapsed_ms;
    res.termination_reason = "stage2_starved";
    return res;
  }

  const int R_total = r_stage1_done + r_stage2_done;

  // Stage 2 paired difference
  double sum_d_stage2 = 0.0;
  for (int r = 0; r < r_stage2_done; ++r) {
    sum_d_stage2 += (per_candidate_rollouts[best_c_idx][r_stage1_done + r] -
                     per_candidate_rollouts[0][r_stage1_done + r]);
  }
  const double mean_d_stage2 = sum_d_stage2 / r_stage2_done;

  double sum_sq_diff_stage2 = 0.0;
  for (int r = 0; r < r_stage2_done; ++r) {
    double diff = (per_candidate_rollouts[best_c_idx][r_stage1_done + r] -
                   per_candidate_rollouts[0][r_stage1_done + r]) - mean_d_stage2;
    sum_sq_diff_stage2 += diff * diff;
  }
  const double var_d_stage2 = (r_stage2_done > 1) ? (sum_sq_diff_stage2 / (r_stage2_done - 1)) : 0.0;
  const double se_d_stage2 = (r_stage2_done > 1) ? std::sqrt(var_d_stage2 / r_stage2_done) : 0.0;
  const double t_stat_stage2 = (r_stage2_done >= 2)
      ? ((se_d_stage2 > 1e-9) ? (mean_d_stage2 / se_d_stage2) : (mean_d_stage2 > 0.0 ? 100.0 : 0.0))
      : 0.0;

  // Total statistics across all R_total rollouts
  double sum_raw_total = 0.0;
  double sum_best_total = 0.0;
  for (int r = 0; r < R_total; ++r) {
    sum_raw_total += per_candidate_rollouts[0][r];
    sum_best_total += per_candidate_rollouts[best_c_idx][r];
  }
  const double raw_mean_total = sum_raw_total / R_total;
  const double best_cand_mean_total = sum_best_total / R_total;
  const double mean_d_total = best_cand_mean_total - raw_mean_total;

  double sum_sq_diff_total = 0.0;
  for (int r = 0; r < R_total; ++r) {
    double diff = (per_candidate_rollouts[best_c_idx][r] - per_candidate_rollouts[0][r]) - mean_d_total;
    sum_sq_diff_total += diff * diff;
  }
  const double var_d_total = (R_total > 1) ? (sum_sq_diff_total / (R_total - 1)) : 0.0;
  const double se_d_total = (R_total > 1) ? std::sqrt(var_d_total / R_total) : 0.0;
  const double t_stat_total = (R_total >= 2)
      ? ((se_d_total > 1e-9) ? (mean_d_total / se_d_total) : (mean_d_total > 0.0 ? 100.0 : 0.0))
      : 0.0;

  // 4. Unbiased Stage-2 Override Gate
  bool should_override = false;
  if (r_stage2_done >= 16) {
    should_override = (mean_d_stage2 >= effective_margin &&
                       t_stat_stage2 >= 1.50 &&
                       mean_d_total >= 0.75 * effective_margin);
  } else {
    should_override = (mean_d_stage2 > 0.0 &&
                       mean_d_total >= effective_margin &&
                       t_stat_total >= 1.50);
  }

  Action best_action = raw_action;
  double best_mean_u = raw_mean_total;
  bool overridden = false;
  if (should_override) {
    best_action = ordered_candidates[best_c_idx];
    best_mean_u = best_cand_mean_total;
    overridden = true;
    if (chosen_candidate_index != nullptr) {
      *chosen_candidate_index = static_cast<int>(best_c_idx);
    }
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
  res.stage1_planned_worlds = r_stage1;
  res.stage1_kept_worlds = r_stage1_done;
  res.stage2_planned_worlds = stage2_planned_worlds;
  res.stage2_kept_worlds = stage2_kept_worlds;
  res.stage1_elapsed_ms = stage1_elapsed_ms;
  res.stage2_elapsed_ms = stage2_elapsed_ms;
  res.termination_reason = overridden ? "override" : "stage2_no_override";
  res.mean_d_stage2 = mean_d_stage2;
  res.se_d_stage2 = se_d_stage2;
  return res;
}

inline CompoundSearchDecisionResult SearchV2SuccessiveHalvingDecision(
    const State& state,
    Player search_player,
    TurnSearchTree* turn_tree,
    const std::vector<std::shared_ptr<BatchedNNEvaluator>>& evaluators,
    int reference_rollouts_per_action,
    double min_override_margin,
    uint64_t decision_seed,
    SearchRolloutCrnMode crn_mode = SearchRolloutCrnMode::kSequential,
    double max_time_ms = -1.0,
    int fixed_top_k = 0,
    int max_inflight_worlds = 32,
    double prior_log_ratio_weight = 0.02,
    double stage1_high_conf_prior_skip = 0.95,
    const ActionsAndProbs* precomputed_raw_prior = nullptr,
    double est_ms_per_sim = 20.0,
    const std::vector<Action>* second_actions = nullptr,
    int* chosen_candidate_index = nullptr) {
  if (turn_tree != nullptr) turn_tree->BindPublicCardKnowledge(state);
  const auto legal = state.LegalActions();
  if (legal.empty()) return {kInvalidAction, false, 0.0, 0.0, 0, 0, {}, {}};
  if (legal.size() == 1) return {legal.front(), false, 0.0, 0.0, 0, 0, {}, {}};
  if (evaluators.empty()) return {legal.front(), false, 0.0, 0.0, 0, 0, {}, {}};

  const ActionsAndProbs raw_prior =
      (precomputed_raw_prior != nullptr && !precomputed_raw_prior->empty())
          ? *precomputed_raw_prior
          : evaluators.front()->Prior(state);

  auto [ordered_candidates, candidate_priors] =
      SelectOrderedCandidatesAndPriors(raw_prior, legal, fixed_top_k);

  return SearchV2SuccessiveHalvingDecision(
      state, search_player, turn_tree, evaluators, ordered_candidates,
      candidate_priors, reference_rollouts_per_action, min_override_margin,
      decision_seed, crn_mode, max_time_ms, max_inflight_worlds,
      prior_log_ratio_weight, stage1_high_conf_prior_skip, second_actions,
      /*graft_actions=*/nullptr, est_ms_per_sim, chosen_candidate_index);
}

inline JointSearchV2Result SearchV2JointPlacementDecision(
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
    double prior_log_ratio_weight = 0.02,
    double stage1_high_conf_prior_skip = 0.95,
    const ActionsAndProbs* precomputed_raw_prior = nullptr,
    double est_ms_per_sim = 20.0,
    size_t max_candidates = 8) {
  if (turn_tree != nullptr) turn_tree->BindPublicCardKnowledge(state);
  const auto legal = state.LegalActions();
  if (legal.empty()) {
    JointSearchV2Result res;
    res.base = {kInvalidAction, false, 0.0, 0.0, 0, 0, {}, {}};
    return res;
  }
  const ActionsAndProbs raw_prior =
      (precomputed_raw_prior != nullptr && !precomputed_raw_prior->empty())
          ? *precomputed_raw_prior
          : (evaluators.empty() ? ActionsAndProbs{} : evaluators.front()->Prior(state));

  std::vector<JointAgentCandidate> joint_cands = EnumerateJointAgentCandidates(
      state, search_player, evaluators, raw_prior, max_candidates);

  if (joint_cands.empty()) {
    Action raw_a = PickGreedyAction(raw_prior, legal);
    JointSearchV2Result res;
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
  CompoundSearchDecisionResult base_res = SearchV2SuccessiveHalvingDecision(
      state, search_player, turn_tree, evaluators, ordered_cands, cand_priors,
      reference_rollouts_per_action, min_override_margin, decision_seed,
      crn_mode, max_time_ms, max_inflight_worlds, prior_log_ratio_weight,
      stage1_high_conf_prior_skip, &second_actions, &graft_actions, est_ms_per_sim,
      &chosen_c_idx);

  JointSearchV2Result res;
  res.base = base_res;
  res.raw_first_action = joint_cands[0].first_action;
  res.raw_graft_action = joint_cands[0].graft_action;
  res.raw_second_action = joint_cands[0].second_action;
  if (chosen_c_idx >= 0 && chosen_c_idx < static_cast<int>(joint_cands.size())) {
    res.chosen_first_action = joint_cands[chosen_c_idx].first_action;
    res.chosen_graft_action = joint_cands[chosen_c_idx].graft_action;
    res.chosen_second_action = joint_cands[chosen_c_idx].second_action;
  } else {
    res.chosen_first_action = base_res.action;
    res.chosen_graft_action = kInvalidAction;
    res.chosen_second_action = kInvalidAction;
  }
  res.joint_candidates = std::move(joint_cands);
  return res;
}

}  // namespace dune_imperium
}  // namespace open_spiel

#endif  // OPEN_SPIEL_EXAMPLES_DUNE_WIDE_ROLLOUT_SEARCH_H_
