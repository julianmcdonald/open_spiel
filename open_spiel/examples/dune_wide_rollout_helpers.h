#ifndef OPEN_SPIEL_EXAMPLES_DUNE_WIDE_ROLLOUT_HELPERS_H_
#define OPEN_SPIEL_EXAMPLES_DUNE_WIDE_ROLLOUT_HELPERS_H_

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <future>
#include <memory>
#include <mutex>
#include <numeric>
#include <random>
#include <utility>
#include <vector>

#include "open_spiel/examples/dune_compound_turn_search.h"
#include "open_spiel/examples/dune_seed_utils.h"
#include "open_spiel/examples/dune_wide_rollout_types.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace dune_imperium {

inline void ApplyCandidateFollowUpActions(
    State* sim_state, Action graft_a, Action second_a) {
  if (graft_a != kInvalidAction) {
    const auto leg_g = sim_state->LegalActions();
    if (std::find(leg_g.begin(), leg_g.end(), graft_a) != leg_g.end()) {
      sim_state->ApplyAction(graft_a);
    }
  }
  if (second_a != kInvalidAction) {
    const auto leg2 = sim_state->LegalActions();
    if (std::find(leg2.begin(), leg2.end(), second_a) != leg2.end()) {
      sim_state->ApplyAction(second_a);
    }
  }
}

inline void ExpandMacroSpaceCandidates(
    const State& state,
    const DuneImperiumState* dune,
    Player search_player,
    BatchedNNEvaluator* prior_eval,
    const MacroSpaceConfig& macro_cfg,
    std::vector<Action>& ordered_candidates,
    std::vector<double>& candidate_priors,
    std::vector<Action>& graft_actions,
    std::vector<Action>& second_actions) {
  if (!(macro_cfg.enabled && dune != nullptr &&
        dune->phase() == GamePhase::kAgentTurns &&
        !dune->IsPlayerInRevealTurn(search_player) &&
        dune->GetCurrentRound() >= macro_cfg.min_round &&
        prior_eval != nullptr)) {
    return;
  }
  const size_t K_base = ordered_candidates.size();
  int num_extra_macros = 0;
  const size_t check_k = std::min<size_t>(3, K_base);
  for (size_t c = 0; c < check_k; ++c) {
    Action a_c = ordered_candidates[c];
    auto s1 = state.Clone();
    s1->ApplyAction(a_c);
    if (s1->IsTerminal() || s1->IsChanceNode() || s1->CurrentPlayer() != search_player) {
      continue;
    }
    auto l1 = s1->LegalActions();
    if (l1.empty()) continue;

    std::unique_ptr<State> s_space = nullptr;
    Action g_0 = kInvalidAction;
    std::vector<Action> l_space;

    if (std::all_of(l1.begin(), l1.end(), IsBoardSpaceAction)) {
      s_space = std::move(s1);
      l_space = std::move(l1);
    } else if (std::all_of(l1.begin(), l1.end(), IsGraftSelectionAction)) {
      ActionsAndProbs prior_graft = prior_eval->Prior(*s1);
      g_0 = PickGreedyAction(prior_graft, l1);
      s1->ApplyAction(g_0);
      if (!s1->IsTerminal() && !s1->IsChanceNode() && s1->CurrentPlayer() == search_player) {
        auto l2 = s1->LegalActions();
        if (!l2.empty() && std::all_of(l2.begin(), l2.end(), IsBoardSpaceAction)) {
          s_space = std::move(s1);
          l_space = std::move(l2);
        }
      }
    }

    if (s_space != nullptr && l_space.size() >= 2) {
      ActionsAndProbs prior_space = prior_eval->Prior(*s_space);
      auto sorted_spaces = GetSortedCandidatesByPrior(prior_space, l_space);
      if (!sorted_spaces.empty()) {
        Action space_greedy = sorted_spaces[0].first;
        graft_actions[c] = g_0;
        second_actions[c] = space_greedy;

        if (c == 0 && sorted_spaces.size() >= 2 && sorted_spaces[1].second >= 0.08) {
          candidate_priors[0] = std::min(candidate_priors[0], 0.90);
        }

        if (num_extra_macros < 2 && ordered_candidates.size() < 7 && sorted_spaces.size() >= 2) {
          Action space_alt = sorted_spaces[1].first;
          double p_space_alt = sorted_spaces[1].second;
          if (p_space_alt >= 0.06 && (candidate_priors[c] * p_space_alt) >= 0.015) {
            ordered_candidates.push_back(a_c);
            graft_actions.push_back(g_0);
            second_actions.push_back(space_alt);
            candidate_priors.push_back(candidate_priors[c] * p_space_alt);
            num_extra_macros++;
          }
        }
      }
    }
  }
}

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
    const std::vector<Action>* graft_actions = nullptr,
    double win_speed_tiebreak_eps = 0.0,
    const EndgamePly2Config* ply2_cfg = nullptr,
    const IntrigueBeliefConfig* belief_cfg = nullptr) {
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
      std::mt19937_64 wrng(wseed);
      auto rng_func = [&wrng]() {
        return std::generate_canonical<double, 53>(wrng);
      };
      base_world_state = dune->ResampleFromInfostate(search_player, rng_func);
      if (base_world_state && belief_cfg != nullptr && belief_cfg->enabled) {
        auto* sampled_dune = dynamic_cast<DuneImperiumState*>(base_world_state.get());
        if (sampled_dune != nullptr &&
            !sampled_dune->SteamIntrigueSimulationForSearch() &&
            sampled_dune->GetPendingHundroSetupPlayer() < 0 &&
            sampled_dune->GetHundroPlayerForTesting() < 0) {
          ApplyOpponentIntrigueBeliefReweighting(*sampled_dune, search_player, *belief_cfg, wseed);
        }
      }
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
        const Action graft_a = (graft_actions != nullptr && c_idx < graft_actions->size())
            ? (*graft_actions)[c_idx] : kInvalidAction;
        const Action second_a = (second_actions != nullptr && c_idx < second_actions->size())
            ? (*second_actions)[c_idx] : kInvalidAction;
        ApplyCandidateFollowUpActions(sim_state.get(), graft_a, second_a);
        auto* s_ptr = sim_state.get();
        if (!in_initial_burst && active_evaluator) {
          active_evaluator->AdjustActiveRollouts(1);
          adjusted_evaluator = active_evaluator;
        }

        futures.push_back(SearchRolloutThreadPool::Instance().Submit(
            [shared, s_ptr, sim_state = std::move(sim_state), search_player, active_evaluator,
             s_seed, crn_mode, sample_opponent_leader_draft, stage_deadline_tp,
             win_speed_tiebreak_eps, ply2_cfg, w, slot]() mutable {
              SlotReleaseGuard guard{shared, w, active_evaluator, false};
              shared->trajs[w][slot] = SimulateRolloutWithTurnTrajectory(
                  s_ptr, search_player, active_evaluator, s_seed,
                  SearchTruncationMode::kTerminal, crn_mode,
                  sample_opponent_leader_draft, &stage_deadline_tp,
                  /*out_steps=*/nullptr, /*out_nn_evals=*/nullptr,
                  /*center_4seat_leaf_values=*/false, win_speed_tiebreak_eps,
                  ply2_cfg);
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

inline void RunUnboundedPairedRolloutTasks(
    const State& state,
    const DuneImperiumState* dune,
    Player search_player,
    const std::vector<std::shared_ptr<BatchedNNEvaluator>>& evaluators,
    const std::vector<Action>& ordered_candidates,
    const std::vector<size_t>& candidate_indices,
    int num_worlds,
    int world_seed_offset,
    uint64_t decision_seed,
    SearchRolloutCrnMode crn_mode,
    TurnSearchTree* active_tree,
    std::vector<std::vector<double>>& per_candidate_rollouts,
    int& new_rollouts_executed,
    const std::vector<Action>& second_actions,
    const std::vector<Action>& graft_actions,
    double win_speed_tiebreak_eps,
    const EndgamePly2Config* eff_ply2,
    const IntrigueBeliefConfig* belief_cfg) {
  std::vector<RolloutTask> tasks;
  tasks.reserve(candidate_indices.size() * num_worlds);

  for (size_t c_idx : candidate_indices) {
    const Action cand_a = ordered_candidates[c_idx];
    const Action graft_a = (c_idx < graft_actions.size()) ? graft_actions[c_idx] : kInvalidAction;
    const Action second_a = (c_idx < second_actions.size()) ? second_actions[c_idx] : kInvalidAction;
    for (int r = 0; r < num_worlds; ++r) {
      uint64_t wseed = dune_seed::DeriveSeed(decision_seed, static_cast<uint64_t>(world_seed_offset + r));
      std::unique_ptr<State> prebuilt;
      if (dune == nullptr) {
        prebuilt = state.Clone();
        prebuilt->ApplyAction(cand_a);
        ApplyCandidateFollowUpActions(prebuilt.get(), graft_a, second_a);
      }
      tasks.push_back({c_idx, world_seed_offset + r, cand_a, std::move(prebuilt), wseed});
    }
  }

  new_rollouts_executed += static_cast<int>(tasks.size());
  for (size_t c_idx : candidate_indices) {
    per_candidate_rollouts[c_idx].resize(world_seed_offset + num_worlds, 0.0);
  }
  if (tasks.empty()) return;

  std::vector<std::future<RolloutTrajectory>> futures;
  futures.reserve(tasks.size());

  for (size_t i = 0; i < tasks.size(); ++i) {
    auto* active_evaluator = evaluators.empty() ? nullptr : evaluators[i % evaluators.size()].get();
    if (active_evaluator != nullptr) {
      active_evaluator->AdjustActiveRollouts(1);
    }
  }

  const State* root_state_ptr = &state;
  for (size_t i = 0; i < tasks.size(); ++i) {
    auto* prebuilt_ptr = tasks[i].sim_state.get();
    uint64_t wseed = tasks[i].wseed;
    Action cand_a = tasks[i].action;
    size_t c_idx = tasks[i].c_idx;
    Action graft_a = (c_idx < graft_actions.size()) ? graft_actions[c_idx] : kInvalidAction;
    Action second_a = (c_idx < second_actions.size()) ? second_actions[c_idx] : kInvalidAction;
    auto* active_evaluator = evaluators.empty() ? nullptr : evaluators[i % evaluators.size()].get();
    futures.push_back(SearchRolloutThreadPool::Instance().Submit(
        [prebuilt_ptr, root_state_ptr, dune, cand_a, graft_a, second_a, wseed, search_player,
         active_evaluator, crn_mode, win_speed_tiebreak_eps, eff_ply2, belief_cfg]() {
          ActiveRolloutReleaseGuard guard{active_evaluator};
          std::unique_ptr<State> sim_state;
          State* s_ptr = prebuilt_ptr;
          if (s_ptr == nullptr) {
            if (dune != nullptr) {
              std::mt19937_64 wrng(wseed);
              auto rng_func = [&wrng]() {
                return std::generate_canonical<double, 53>(wrng);
              };
              sim_state = dune->ResampleFromInfostate(search_player, rng_func);
              if (sim_state && belief_cfg != nullptr && belief_cfg->enabled) {
                auto* sampled_dune = dynamic_cast<DuneImperiumState*>(sim_state.get());
                if (sampled_dune != nullptr &&
                    !sampled_dune->SteamIntrigueSimulationForSearch() &&
                    sampled_dune->GetPendingHundroSetupPlayer() < 0 &&
                    sampled_dune->GetHundroPlayerForTesting() < 0) {
                  ApplyOpponentIntrigueBeliefReweighting(*sampled_dune, search_player, *belief_cfg, wseed);
                }
              }
            }
            if (!sim_state) {
              sim_state = root_state_ptr->Clone();
            }
            sim_state->ApplyAction(cand_a);
            ApplyCandidateFollowUpActions(sim_state.get(), graft_a, second_a);
            s_ptr = sim_state.get();
          }
          return SimulateRolloutWithTurnTrajectory(
              s_ptr, search_player, active_evaluator, wseed + 999,
              SearchTruncationMode::kTerminal, crn_mode,
              /*sample_opponent_leader_draft=*/false, /*deadline=*/nullptr,
              /*out_steps=*/nullptr, /*out_nn_evals=*/nullptr,
              /*center_4seat_leaf_values=*/false, win_speed_tiebreak_eps,
              eff_ply2);
        }));
  }

  for (size_t i = 0; i < futures.size(); ++i) {
    RolloutTrajectory traj = futures[i].get();
    size_t c_idx = tasks[i].c_idx;
    std::vector<Action> intra = traj.intra_turn_actions;
    if (c_idx < second_actions.size() && second_actions[c_idx] != kInvalidAction) {
      intra.insert(intra.begin(), second_actions[c_idx]);
    }
    if (c_idx < graft_actions.size() && graft_actions[c_idx] != kInvalidAction) {
      intra.insert(intra.begin(), graft_actions[c_idx]);
    }
    active_tree->RecordRollout(tasks[i].action, intra, traj.terminal_utility);
    per_candidate_rollouts[tasks[i].c_idx][tasks[i].rollout_idx] = traj.terminal_utility;
  }
}

inline int RunSHStage1ClockedWaves(
    const State& state,
    const DuneImperiumState* dune,
    Player search_player,
    const std::vector<std::shared_ptr<BatchedNNEvaluator>>& evaluators,
    const std::vector<Action>& ordered_candidates,
    int r_stage1,
    uint64_t decision_seed,
    SearchRolloutCrnMode crn_mode,
    bool has_deadline,
    bool full_clock_stage2,
    double max_time_ms,
    double s1_deadline_frac,
    std::chrono::steady_clock::time_point start_time,
    std::chrono::steady_clock::time_point stage1_deadline,
    std::chrono::steady_clock::time_point final_deadline,
    std::chrono::steady_clock::time_point s1_t0,
    int max_inflight_worlds,
    TurnSearchTree* active_tree,
    std::vector<std::vector<double>>& per_candidate_rollouts,
    int& new_rollouts_executed,
    const std::vector<Action>& second_actions,
    const std::vector<Action>& graft_actions,
    double win_speed_tiebreak_eps,
    const EndgamePly2Config* eff_ply2_s1,
    const IntrigueBeliefConfig* belief_cfg) {
  const size_t K = ordered_candidates.size();
  std::vector<size_t> candidate_indices(K);
  std::iota(candidate_indices.begin(), candidate_indices.end(), 0);
  const bool caller_small_cap =
      (max_inflight_worlds > 0 && max_inflight_worlds < 64);
  const int s1_inflight = caller_small_cap
      ? max_inflight_worlds
      : (full_clock_stage2
             ? std::min(128, std::max(max_inflight_worlds, 256 / static_cast<int>(K)))
             : max_inflight_worlds);

  int s1_waves_completed = 0;
  if (!full_clock_stage2 || !has_deadline) {
    RunClockedPairedRolloutStage(
        state, dune, search_player, evaluators, ordered_candidates,
        candidate_indices, r_stage1, /*world_seed_offset=*/0,
        decision_seed, crn_mode, /*sample_opponent_leader_draft=*/false,
        stage1_deadline, active_tree, per_candidate_rollouts,
        new_rollouts_executed, s1_inflight, /*resample_from_infostate=*/true,
        &second_actions, &graft_actions,
        win_speed_tiebreak_eps, eff_ply2_s1, belief_cfg);
    return s1_waves_completed;
  }

  const int w0 = std::min(
      r_stage1,
      (max_time_ms <= 1600.0)
          ? 2
          : ((max_time_ms <= 3500.0) ? 4 : std::clamp(r_stage1 / 2, 4, 32)));
  RunClockedPairedRolloutStage(
      state, dune, search_player, evaluators, ordered_candidates,
      candidate_indices, w0, /*world_seed_offset=*/0,
      decision_seed, crn_mode, /*sample_opponent_leader_draft=*/false,
      final_deadline, active_tree, per_candidate_rollouts,
      new_rollouts_executed, std::min(s1_inflight, w0),
      /*resample_from_infostate=*/true,
      &second_actions, &graft_actions,
      win_speed_tiebreak_eps, eff_ply2_s1, belief_cfg);

  int s1_done = per_candidate_rollouts.empty()
      ? 0
      : static_cast<int>(per_candidate_rollouts[0].size());
  if (s1_done >= w0) {
    s1_waves_completed = 1;
  }
  int s1_seed_offset = w0;
  if (s1_done >= w0 && s1_done < r_stage1) {
    const double w0_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - s1_t0).count();
    double s1_cur_ms = std::max(
        0.05, w0_ms / static_cast<double>(K * std::max(1, s1_done)));
    const double s1_stop_frac = (max_time_ms <= 2500.0) ? 0.45 : 0.38;
    while (s1_done < r_stage1) {
      const double s1_el = std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - start_time).count();
      if (s1_el >= s1_stop_frac * max_time_ms ||
          (s1_waves_completed > 0 &&
           (s1_deadline_frac * max_time_ms - s1_el) <
               0.85 * (s1_el / s1_waves_completed))) {
        break;
      }
      const double s1_rem_ms =
          0.76 * std::max(0.0, s1_deadline_frac * max_time_ms - s1_el);
      const int next_w = std::min(
          r_stage1 - s1_done,
          static_cast<int>(s1_rem_ms / (static_cast<double>(K) * s1_cur_ms)));
      if (next_w < 2) break;
      const int before_s1 = s1_done;
      const auto s1_wt0 = std::chrono::steady_clock::now();
      RunClockedPairedRolloutStage(
          state, dune, search_player, evaluators, ordered_candidates,
          candidate_indices, next_w, /*world_seed_offset=*/s1_seed_offset,
          decision_seed, crn_mode, /*sample_opponent_leader_draft=*/false,
          stage1_deadline, active_tree, per_candidate_rollouts,
          new_rollouts_executed, std::min(s1_inflight, next_w),
          /*resample_from_infostate=*/true,
          &second_actions, &graft_actions,
          win_speed_tiebreak_eps, eff_ply2_s1, belief_cfg);
      s1_done = static_cast<int>(per_candidate_rollouts[0].size());
      const int kept_s1 = s1_done - before_s1;
      const double s1_wt_ms = std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - s1_wt0).count();
      s1_seed_offset += next_w;
      if (kept_s1 < next_w || kept_s1 <= 0) break;
      s1_waves_completed++;
      if (s1_wt_ms > 0.0) {
        s1_cur_ms = 0.3 * s1_cur_ms +
                    0.7 * (s1_wt_ms / static_cast<double>(K * kept_s1));
      }
    }
  }
  return s1_waves_completed;
}

inline void RunSHStage2ClockedWaves(
    const State& state,
    const DuneImperiumState* dune,
    Player search_player,
    const std::vector<std::shared_ptr<BatchedNNEvaluator>>& evaluators,
    const std::vector<Action>& ordered_candidates,
    size_t best_c_idx,
    int r_stage1,
    int r_stage1_done,
    int r_stage2,
    double stage1_elapsed_ms,
    double raw_mean_1,
    double min_override_margin,
    double prior_penalty_best,
    uint64_t decision_seed,
    SearchRolloutCrnMode crn_mode,
    bool has_deadline,
    bool full_clock_stage2,
    double max_time_ms,
    std::chrono::steady_clock::time_point start_time,
    std::chrono::steady_clock::time_point final_deadline,
    int max_inflight_worlds,
    TurnSearchTree* active_tree,
    std::vector<std::vector<double>>& per_candidate_rollouts,
    int& new_rollouts_executed,
    int& stage2_planned_worlds,
    const std::vector<Action>& second_actions,
    const std::vector<Action>& graft_actions,
    double win_speed_tiebreak_eps,
    bool ceiling_aware_margin,
    const EndgamePly2Config* eff_ply2_s2,
    const IntrigueBeliefConfig* belief_cfg) {
  const size_t K = ordered_candidates.size();
  std::vector<size_t> stage2_cands = {0, best_c_idx};
  const int worker_s2_cap =
      128 * std::max<int>(1, static_cast<int>(evaluators.size()));
  const bool caller_small_cap =
      (max_inflight_worlds > 0 && max_inflight_worlds < 64);
  const int s2_inflight = std::min(
      worker_s2_cap,
      caller_small_cap ? max_inflight_worlds : std::max(max_inflight_worlds, 128));
  if (!full_clock_stage2 || !has_deadline) {
    RunClockedPairedRolloutStage(
        state, dune, search_player, evaluators, ordered_candidates,
        stage2_cands, r_stage2, /*world_seed_offset=*/r_stage1,
        decision_seed, crn_mode, /*sample_opponent_leader_draft=*/false,
        final_deadline, active_tree, per_candidate_rollouts,
        new_rollouts_executed, s2_inflight, /*resample_from_infostate=*/true,
        &second_actions, &graft_actions,
        win_speed_tiebreak_eps, eff_ply2_s2, belief_cfg);
    return;
  }

  stage2_planned_worlds = 0;
  int seed_offset = r_stage1;
  double cur_ms_per_sim =
      stage1_elapsed_ms / std::max(1.0, static_cast<double>(K * r_stage1_done));
  const double s2_margin_est = ComputeEffectiveOverrideMargin(
      min_override_margin, raw_mean_1, win_speed_tiebreak_eps,
      ceiling_aware_margin) + prior_penalty_best;
  while (true) {
    const double total_elapsed_ms =
        std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - start_time).count();
    const int s2_done_now =
        static_cast<int>(per_candidate_rollouts[0].size()) - r_stage1_done;
    if (s2_done_now >= 2 && total_elapsed_ms >= 0.90 * max_time_ms) {
      break;
    }
    if (s2_done_now >= 320) {
      const PairedEbDiffStats cur_s2_stats = ComputePairedEbDiffStats(
          per_candidate_rollouts[best_c_idx].data() + r_stage1_done,
          per_candidate_rollouts[0].data() + r_stage1_done,
          s2_done_now);
      if (cur_s2_stats.se_eb > 0.0 &&
          (cur_s2_stats.mean_d + 3.5 * cur_s2_stats.se_eb < s2_margin_est ||
           cur_s2_stats.mean_d - 4.0 * cur_s2_stats.se_eb > s2_margin_est)) {
        break;
      }
    }
    const double rem_ms = std::max(0.0, 0.955 * max_time_ms - total_elapsed_ms);
    if (s2_done_now >= 2 && rem_ms < std::max(60.0, 4.5 * cur_ms_per_sim)) {
      break;
    }
    const double wave_target_ms = (s2_done_now == 0)
        ? (0.55 * rem_ms)
        : (0.76 * rem_ms);
    const int max_wave_w = (s2_done_now == 0) ? 128 : 256;
    const double est_sims = wave_target_ms / std::max(0.05, cur_ms_per_sim);
    const int wave_worlds = std::clamp(
        static_cast<int>(est_sims / 2.0), 2, max_wave_w);
    stage2_planned_worlds += wave_worlds;
    const int before_size = static_cast<int>(per_candidate_rollouts[0].size());
    const auto wt0 = std::chrono::steady_clock::now();
    RunClockedPairedRolloutStage(
        state, dune, search_player, evaluators, ordered_candidates,
        stage2_cands, wave_worlds, /*world_seed_offset=*/seed_offset,
        decision_seed, crn_mode, /*sample_opponent_leader_draft=*/false,
        final_deadline, active_tree, per_candidate_rollouts,
        new_rollouts_executed, std::min(s2_inflight, wave_worlds),
        /*resample_from_infostate=*/true,
        &second_actions, &graft_actions,
        win_speed_tiebreak_eps, eff_ply2_s2, belief_cfg);
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

inline void RunPairedDecisionClockedWaves(
    const State& state,
    const DuneImperiumState* dune,
    Player search_player,
    const std::vector<std::shared_ptr<BatchedNNEvaluator>>& evaluators,
    const std::vector<Action>& ordered_candidates,
    int rollouts_per_action,
    uint64_t decision_seed,
    SearchRolloutCrnMode crn_mode,
    bool sample_opponent_leader_draft,
    double max_time_ms,
    int max_inflight_worlds,
    bool resample_from_infostate,
    double win_speed_tiebreak_eps,
    std::chrono::steady_clock::time_point s1_t0,
    TurnSearchTree* active_tree,
    std::vector<std::vector<double>>& per_candidate_rollouts,
    int& new_rollouts_executed) {
  const size_t K = ordered_candidates.size();
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
      resample_from_infostate, /*second_actions=*/nullptr,
      /*graft_actions=*/nullptr, win_speed_tiebreak_eps);
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
      if (total_elapsed_ms >= 0.88 * max_time_ms) break;
      const double rem_ms =
          0.72 * std::max(0.0, 0.955 * max_time_ms - total_elapsed_ms);
      if (rem_ms < 2.0 * static_cast<double>(K) * std::max(0.2, cur_ms_per_sim)) {
        break;
      }
      const double est_sims = rem_ms / std::max(0.05, cur_ms_per_sim);
      int wave_worlds = std::clamp(
          static_cast<int>(est_sims / static_cast<double>(K)), 2, 256);
      const int before_w = static_cast<int>(per_candidate_rollouts[0].size());
      const auto wt0 = std::chrono::steady_clock::now();
      RunClockedPairedRolloutStage(
          state, dune, search_player, evaluators, ordered_candidates,
          candidate_indices, wave_worlds, /*world_seed_offset=*/seed_offset,
          decision_seed, crn_mode, sample_opponent_leader_draft,
          deadline_tp, active_tree, per_candidate_rollouts,
          new_rollouts_executed, max_inflight_worlds,
          resample_from_infostate, /*second_actions=*/nullptr,
          /*graft_actions=*/nullptr, win_speed_tiebreak_eps);
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

}  // namespace dune_imperium
}  // namespace open_spiel

#endif  // OPEN_SPIEL_EXAMPLES_DUNE_WIDE_ROLLOUT_HELPERS_H_
