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

#ifndef OPEN_SPIEL_EXAMPLES_DUNE_COMPOUND_TURN_DECISION_HELPERS_H_
#define OPEN_SPIEL_EXAMPLES_DUNE_COMPOUND_TURN_DECISION_HELPERS_H_

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <future>
#include <memory>
#include <random>
#include <utility>
#include <vector>

#include "open_spiel/examples/dune_compound_turn_rollout.h"

namespace open_spiel {
namespace dune_imperium {

// Optional source of search worlds, keyed by the world seed. Returning nullptr
// (or leaving it empty) uses ResampleFromInfostate as before. A sampler must be
// deterministic in its seed so paired candidates share worlds.
using SearchWorldSampler = std::function<std::unique_ptr<State>(uint64_t world_seed)>;

struct CompoundRolloutTask {
  size_t c_idx;
  int rollout_idx;
  Action action;
  std::unique_ptr<State> sim_state;
  uint64_t wseed;
};

inline std::vector<Action> SelectCompoundSearchCandidates(
    const ActionsAndProbs& raw_prior,
    const std::vector<Action>& legal_actions,
    Action raw_action,
    int top_k,
    bool all_buy_candidates,
    bool symmetric_stage2_confirmation,
    double confirmation_rival_cum_mass,
    int confirmation_max_rivals) {
  std::vector<Action> candidates;
  if (all_buy_candidates) {
    candidates.push_back(raw_action);
    for (Action a : legal_actions) {
      if (IsCardOrTechAcquisitionAction(a) && a != raw_action) {
        candidates.push_back(a);
      }
    }
    std::vector<Action> top_actions = GetTopKActions(raw_prior, legal_actions, top_k);
    for (Action a : top_actions) {
      if (std::find(candidates.begin(), candidates.end(), a) == candidates.end()) {
        candidates.push_back(a);
      }
    }
  } else {
    candidates = GetTopKActions(raw_prior, legal_actions, top_k);
  }

  if (symmetric_stage2_confirmation) {
    std::vector<std::pair<double, Action>> sorted_prior;
    for (const auto& ap : raw_prior) {
      if (std::find(legal_actions.begin(), legal_actions.end(), ap.first) != legal_actions.end()) {
        sorted_prior.push_back({ap.second, ap.first});
      }
    }
    std::sort(sorted_prior.begin(), sorted_prior.end(), [](const auto& a, const auto& b) {
      return a.first > b.first;
    });
    double cum_mass = 0.0;
    for (const auto& sp : sorted_prior) {
      if (sp.second == raw_action) {
        cum_mass += sp.first;
        break;
      }
    }
    int rival_count = 0;
    for (const auto& sp : sorted_prior) {
      if (sp.second == raw_action) continue;
      if (std::find(candidates.begin(), candidates.end(), sp.second) == candidates.end()) {
        candidates.push_back(sp.second);
      }
      cum_mass += sp.first;
      ++rival_count;
      if (cum_mass >= confirmation_rival_cum_mass || rival_count >= confirmation_max_rivals) {
        break;
      }
    }
  }
  return candidates;
}

inline std::vector<std::pair<Action, double>> AlignRawCandidateAndPriors(
    std::vector<Action>* candidates,
    const ActionsAndProbs& raw_prior,
    Action raw_action,
    int top_k,
    bool all_buy_candidates,
    bool symmetric_stage2_confirmation) {
  // Ensure candidate 0 is always the baseline raw_action
  if (candidates->front() != raw_action) {
    auto it = std::find(candidates->begin(), candidates->end(), raw_action);
    if (it != candidates->end()) {
      std::swap(candidates->front(), *it);
    } else {
      candidates->insert(candidates->begin(), raw_action);
      if (!all_buy_candidates && !symmetric_stage2_confirmation &&
          candidates->size() > static_cast<size_t>(top_k)) {
        candidates->pop_back();
      }
    }
  }

  std::vector<std::pair<Action, double>> candidate_priors;
  candidate_priors.reserve(candidates->size());
  for (Action cand_a : *candidates) {
    double p = 0.0;
    for (const auto& ap : raw_prior) {
      if (ap.first == cand_a) {
        p = ap.second;
        break;
      }
    }
    candidate_priors.push_back({cand_a, p});
  }
  return candidate_priors;
}

inline std::vector<CompoundRolloutTask> BuildCompoundRolloutTasks(
    const std::vector<Action>& candidates,
    const std::vector<size_t>& active_c_indices,
    int r_begin,
    int r_end,
    bool use_dedup,
    SearchPairingMode pairing_mode,
    uint64_t decision_seed,
    Player search_player,
    const State* root_state_ptr,
    const DuneImperiumState* dune,
    const SearchWorldSampler& world_sampler,
    std::atomic<int>* resample_fallbacks) {
  std::vector<CompoundRolloutTask> batch_tasks;
  if (r_begin >= r_end || active_c_indices.empty()) return batch_tasks;
  const int num_worlds = r_end - r_begin;

  std::vector<std::unique_ptr<State>> base_worlds;
  if (use_dedup) {
    base_worlds.resize(num_worlds);
    for (int w = 0; w < num_worlds; ++w) {
      uint64_t wseed = dune_seed::DeriveSeed(decision_seed, r_begin + w);
      std::mt19937_64 wrng(wseed);
      auto rng_func = [&wrng]() {
        return std::generate_canonical<double, 53>(wrng);
      };
      std::unique_ptr<State> base_state = dune->ResampleFromInfostate(search_player, rng_func);
      if (!base_state) {
        ++(*resample_fallbacks);
        base_state = root_state_ptr->Clone();
      }
      base_worlds[w] = std::move(base_state);
    }
  }

  batch_tasks.reserve(active_c_indices.size() * num_worlds);
  for (size_t c_idx : active_c_indices) {
    Action cand_a = candidates[c_idx];
    for (int r = r_begin; r < r_end; ++r) {
      uint64_t wseed = (pairing_mode == SearchPairingMode::kPaired)
          ? dune_seed::DeriveSeed(decision_seed, r)
          : dune_seed::DeriveSeed(decision_seed, c_idx, r);
      std::unique_ptr<State> sim_state = nullptr;
      if (use_dedup) {
        sim_state = base_worlds[r - r_begin]->Clone();
        sim_state->ApplyAction(cand_a);
      } else if (world_sampler || dune == nullptr) {
        if (world_sampler) {
          sim_state = world_sampler(wseed);
          if (sim_state) {
            const auto sim_legal = sim_state->LegalActions();
            if (std::find(sim_legal.begin(), sim_legal.end(), cand_a) == sim_legal.end()) {
              sim_state = nullptr;
            }
          }
        }
        if (!sim_state && dune) {
          std::mt19937_64 wrng(wseed);
          auto rng_func = [&wrng]() {
            return std::generate_canonical<double, 53>(wrng);
          };
          sim_state = dune->ResampleFromInfostate(search_player, rng_func);
          if (!sim_state) {
            ++(*resample_fallbacks);
          }
        }
        if (!sim_state) {
          sim_state = root_state_ptr->Clone();
        }
        sim_state->ApplyAction(cand_a);
      }
      batch_tasks.push_back({c_idx, r, cand_a, std::move(sim_state), wseed});
    }
  }
  return batch_tasks;
}

inline void ExecuteCompoundRolloutTasks(
    std::vector<CompoundRolloutTask>& batch_tasks,
    const std::vector<std::shared_ptr<BatchedNNEvaluator>>& evaluators,
    const State* root_state_ptr,
    const DuneImperiumState* dune,
    Player search_player,
    SearchTruncationMode truncation_mode,
    SearchRolloutCrnMode crn_mode,
    bool center_4seat_leaf_values,
    double win_speed_tiebreak_eps,
    TurnSearchTree* active_tree,
    std::atomic<int>* resample_fallbacks,
    int* new_rollouts_executed,
    int* total_steps,
    int* total_nn_evals,
    std::vector<std::vector<double>>* per_candidate_rollouts,
    std::vector<std::vector<double>>* cand_rollout_returns) {
  if (batch_tasks.empty()) return;
  *new_rollouts_executed += static_cast<int>(batch_tasks.size());

  std::vector<std::future<RolloutTrajectory>> futures;
  futures.reserve(batch_tasks.size());

  for (size_t i = 0; i < batch_tasks.size(); ++i) {
    auto* active_evaluator = evaluators[i % evaluators.size()].get();
    if (active_evaluator != nullptr) {
      active_evaluator->AdjustActiveRollouts(1);
    }
  }

  for (size_t i = 0; i < batch_tasks.size(); ++i) {
    auto* prebuilt_ptr = batch_tasks[i].sim_state.get();
    uint64_t wseed = batch_tasks[i].wseed;
    Action cand_a = batch_tasks[i].action;
    auto* active_evaluator = evaluators[i % evaluators.size()].get();
    futures.push_back(SearchRolloutThreadPool::Instance().Submit(
        [prebuilt_ptr, root_state_ptr, dune, cand_a, wseed, search_player,
         active_evaluator, truncation_mode, crn_mode, resample_fallbacks,
         center_4seat_leaf_values, win_speed_tiebreak_eps]() {
          ActiveRolloutReleaseGuard guard{active_evaluator};
          std::unique_ptr<State> local_state;
          State* s_ptr = prebuilt_ptr;
          if (s_ptr == nullptr) {
            if (dune != nullptr) {
              std::mt19937_64 wrng(wseed);
              auto rng_func = [&wrng]() {
                return std::generate_canonical<double, 53>(wrng);
              };
              local_state = dune->ResampleFromInfostate(search_player, rng_func);
            }
            if (!local_state) {
              ++(*resample_fallbacks);
              local_state = root_state_ptr->Clone();
            }
            local_state->ApplyAction(cand_a);
            s_ptr = local_state.get();
          }
          return SimulateRolloutWithTurnTrajectory(
              s_ptr, search_player, active_evaluator, wseed + 999, truncation_mode, crn_mode,
              /*sample_opponent_leader_draft=*/false, /*deadline=*/nullptr,
              /*out_steps=*/nullptr, /*out_nn_evals=*/nullptr,
              center_4seat_leaf_values, win_speed_tiebreak_eps);
        }));
  }

  for (size_t i = 0; i < futures.size(); ++i) {
    RolloutTrajectory traj = futures[i].get();
    *total_steps += (traj.steps + 1);
    *total_nn_evals += traj.nn_evals;
    active_tree->RecordRollout(batch_tasks[i].action, traj.intra_turn_actions, traj.terminal_utility);
    if (batch_tasks[i].c_idx < per_candidate_rollouts->size() &&
        batch_tasks[i].rollout_idx >= 0 &&
        static_cast<size_t>(batch_tasks[i].rollout_idx) < (*per_candidate_rollouts)[batch_tasks[i].c_idx].size()) {
      (*per_candidate_rollouts)[batch_tasks[i].c_idx][batch_tasks[i].rollout_idx] = traj.terminal_utility;
    }
    if (batch_tasks[i].c_idx < cand_rollout_returns->size()) {
      (*cand_rollout_returns)[batch_tasks[i].c_idx].push_back(traj.terminal_utility);
    }
  }
}

struct SymmetricStage2Params {
  int r1 = 0;
  int r2_indep = 0;
  bool can_dedup = false;
  double stage1_min_mean_diff = -1e9;
  double min_override_margin = 0.0;
  double min_paired_t_stat = 0.0;
  double confirmation_max_raw_prior = 0.85;
  double confirmation_rival_cum_mass = 0.90;
  int confirmation_max_rivals = 3;
  double win_speed_tiebreak_eps = 0.0;
  bool ceiling_aware_margin = true;
};

struct SymmetricStage2Outcome {
  bool stopped_stage1 = false;
  size_t best_s1_c = 0;
  double best_s1_mean_d = -1e9;
  bool confirmed = false;
  bool teacher_accepted = false;
  Action teacher_action = kInvalidAction;
  std::vector<std::pair<Action, double>> stage2_candidate_utilities;
  Action chosen_action = kInvalidAction;
  Action best_action = kInvalidAction;
  double best_mean_u = 0.0;
  size_t best_c_idx = 0;
  double best_vs_raw_mean_diff = 0.0;
  double best_vs_raw_t_stat = 0.0;
  bool overridden = false;
  double raw_mean_u = 0.0;
};

template <typename BuildTasksFn, typename ExecuteTasksFn>
inline SymmetricStage2Outcome EvaluateSymmetricStage2Confirmation(
    const SymmetricStage2Params& params,
    const std::vector<Action>& candidates,
    Action raw_action,
    const ActionsAndProbs& raw_prior,
    const std::vector<std::vector<double>>& cand_rollout_returns,
    const std::vector<std::vector<double>>& per_candidate_rollouts,
    BuildTasksFn&& build_tasks,
    ExecuteTasksFn&& execute_tasks) {
  SymmetricStage2Outcome out;
  out.teacher_action = raw_action;
  out.chosen_action = raw_action;
  out.best_action = raw_action;

  const int R1 = params.r1;
  const int R2_indep = params.r2_indep;
  const bool can_dedup = params.can_dedup;

  double s1_mean_0 = 0.0;
  for (int r = 0; r < R1; ++r) {
    s1_mean_0 += per_candidate_rollouts[0][r];
  }
  s1_mean_0 /= R1;

  out.best_s1_c = 0;
  out.best_s1_mean_d = -1e9;
  for (size_t c_idx = 1; c_idx < candidates.size(); ++c_idx) {
    if (cand_rollout_returns[c_idx].size() < static_cast<size_t>(R1)) continue;
    double sum_c = 0.0;
    for (int r = 0; r < R1; ++r) {
      sum_c += per_candidate_rollouts[c_idx][r];
    }
    double mean_d = (sum_c / R1) - s1_mean_0;
    if (out.best_s1_c == 0 || mean_d > out.best_s1_mean_d) {
      out.best_s1_mean_d = mean_d;
      out.best_s1_c = c_idx;
    }
  }

  out.stopped_stage1 = false;

  // Case A: best_s1_c != 0 && best_s1_mean_d >= stage1_min_mean_diff
  if (out.best_s1_c != 0 && out.best_s1_mean_d >= params.stage1_min_mean_diff) {
    std::vector<size_t> s2_c_indices = {0, out.best_s1_c};
    auto s2_tasks = build_tasks(s2_c_indices, R1, R1 + R2_indep, can_dedup);
    execute_tasks(s2_tasks);

    const PairedEbDiffStats s2_eb = ComputePairedEbDiffStats(
        per_candidate_rollouts[out.best_s1_c].data() + R1,
        per_candidate_rollouts[0].data() + R1,
        R2_indep);
    const double mean_d = s2_eb.mean_d;
    const double paired_t_s2 = s2_eb.t_stat;

    double q_s2_0 = 0.0, q_s2_best = 0.0;
    for (int r = R1; r < R1 + R2_indep; ++r) {
      q_s2_0 += per_candidate_rollouts[0][r];
      q_s2_best += per_candidate_rollouts[out.best_s1_c][r];
    }
    q_s2_0 /= R2_indep;
    q_s2_best /= R2_indep;

    out.best_action = candidates[out.best_s1_c];
    out.best_mean_u = q_s2_best;
    out.raw_mean_u = q_s2_0;
    out.best_c_idx = out.best_s1_c;
    out.best_vs_raw_mean_diff = mean_d;
    out.best_vs_raw_t_stat = paired_t_s2;

    const double eff_margin_s2 = ComputeEffectiveOverrideMargin(
        params.min_override_margin, q_s2_0, params.win_speed_tiebreak_eps,
        params.ceiling_aware_margin);
    if (q_s2_best - q_s2_0 >= eff_margin_s2 &&
        paired_t_s2 >= params.min_paired_t_stat &&
        PassesSingleFlipGuard(R2_indep, params.min_paired_t_stat, s2_eb.positive_diff_count)) {
      out.overridden = true;
      out.teacher_accepted = true;
      out.teacher_action = candidates[out.best_s1_c];
      out.chosen_action = out.teacher_action;
      out.stage2_candidate_utilities.push_back({candidates[0], q_s2_0});
      out.stage2_candidate_utilities.push_back({candidates[out.best_s1_c], q_s2_best});
    } else {
      out.overridden = false;
      out.teacher_accepted = false;
      out.teacher_action = raw_action;
      out.chosen_action = raw_action;
      out.stage2_candidate_utilities.clear();
    }
  }
  // Case B: best_s1_mean_d <= 0.0 (candidates[0] won Stage 1)
  else if (out.best_s1_mean_d <= 0.0) {
    double pi_raw_0 = 0.0;
    for (const auto& ap : raw_prior) {
      if (ap.first == raw_action) { pi_raw_0 = ap.second; break; }
    }
    if (pi_raw_0 < params.confirmation_max_raw_prior && candidates.size() >= 2) {
      std::vector<std::pair<double, size_t>> rival_pairs;
      for (size_t c = 1; c < candidates.size(); ++c) {
        double p = 0.0;
        for (const auto& ap : raw_prior) {
          if (ap.first == candidates[c]) { p = ap.second; break; }
        }
        rival_pairs.push_back({p, c});
      }
      std::sort(rival_pairs.begin(), rival_pairs.end(), [](const auto& a, const auto& b) {
        return a.first > b.first;
      });
      double cum_p = pi_raw_0;
      std::vector<size_t> selected_rivals;
      for (const auto& rp : rival_pairs) {
        selected_rivals.push_back(rp.second);
        cum_p += rp.first;
        if (cum_p >= params.confirmation_rival_cum_mass ||
            static_cast<int>(selected_rivals.size()) >= params.confirmation_max_rivals) {
          break;
        }
      }
      if (selected_rivals.empty() && candidates.size() >= 2) {
        selected_rivals.push_back(1);
      }

      std::vector<size_t> s2_actions = {0};
      s2_actions.insert(s2_actions.end(), selected_rivals.begin(), selected_rivals.end());
      auto s2_tasks = build_tasks(s2_actions, R1, R1 + R2_indep, can_dedup);
      execute_tasks(s2_tasks);

      std::vector<double> q_s2(candidates.size(), 0.0);
      for (size_t c_idx : s2_actions) {
        double sum_q = 0.0;
        for (int r = R1; r < R1 + R2_indep; ++r) {
          sum_q += per_candidate_rollouts[c_idx][r];
        }
        q_s2[c_idx] = sum_q / R2_indep;
      }
      out.raw_mean_u = q_s2[0];

      auto paired_stats = [&](size_t a_idx, size_t b_idx) -> PairedEbDiffStats {
        return ComputePairedEbDiffStats(
            per_candidate_rollouts[a_idx].data() + R1,
            per_candidate_rollouts[b_idx].data() + R1,
            R2_indep);
      };

      size_t rival_1 = selected_rivals[0];
      auto stats_0_vs_r1 = paired_stats(0, rival_1);
      const double eff_margin_confirm = ComputeEffectiveOverrideMargin(
          params.min_override_margin, q_s2[rival_1], params.win_speed_tiebreak_eps,
          params.ceiling_aware_margin);
      if (stats_0_vs_r1.mean_d >= eff_margin_confirm &&
          stats_0_vs_r1.t_stat >= params.min_paired_t_stat &&
          PassesSingleFlipGuard(R2_indep, params.min_paired_t_stat, stats_0_vs_r1.positive_diff_count)) {
        out.confirmed = true;
        out.teacher_accepted = true;
        out.teacher_action = candidates[0];
        out.chosen_action = candidates[0];
        out.best_action = candidates[0];
        out.best_mean_u = q_s2[0];
        out.best_vs_raw_mean_diff = stats_0_vs_r1.mean_d;
        out.best_vs_raw_t_stat = stats_0_vs_r1.t_stat;
        out.overridden = false;
        for (size_t c_idx : s2_actions) {
          out.stage2_candidate_utilities.push_back({candidates[c_idx], q_s2[c_idx]});
        }
      } else {
        size_t best_r = 0;
        double best_r_q = -1e9;
        double best_r_d = -1e9;
        double best_r_t = -1e9;
        const double eff_margin_r = ComputeEffectiveOverrideMargin(
            params.min_override_margin, q_s2[0], params.win_speed_tiebreak_eps,
            params.ceiling_aware_margin);
        for (size_t r_c : selected_rivals) {
          auto stats_r_vs_0 = paired_stats(r_c, 0);
          if (stats_r_vs_0.mean_d >= eff_margin_r &&
              stats_r_vs_0.t_stat >= params.min_paired_t_stat &&
              PassesSingleFlipGuard(R2_indep, params.min_paired_t_stat, stats_r_vs_0.positive_diff_count)) {
            if (q_s2[r_c] > best_r_q) {
              best_r_q = q_s2[r_c];
              best_r = r_c;
              best_r_d = stats_r_vs_0.mean_d;
              best_r_t = stats_r_vs_0.t_stat;
            }
          }
        }
        if (best_r != 0) {
          out.overridden = true;
          out.teacher_accepted = true;
          out.teacher_action = candidates[best_r];
          out.chosen_action = candidates[best_r];
          out.best_action = candidates[best_r];
          out.best_mean_u = best_r_q;
          out.best_c_idx = best_r;
          out.best_vs_raw_mean_diff = best_r_d;
          out.best_vs_raw_t_stat = best_r_t;
          for (size_t c_idx : s2_actions) {
            out.stage2_candidate_utilities.push_back({candidates[c_idx], q_s2[c_idx]});
          }
        } else {
          out.teacher_accepted = false;
          out.teacher_action = raw_action;
          out.chosen_action = raw_action;
          out.best_action = raw_action;
          out.best_mean_u = q_s2[0];
          out.best_c_idx = 0;
          out.best_vs_raw_mean_diff = 0.0;
          out.best_vs_raw_t_stat = 0.0;
          out.overridden = false;
          out.confirmed = false;
          out.stage2_candidate_utilities.clear();
        }
      }
    } else {
      // pi_raw_0 >= confirmation_max_raw_prior: no Stage 2 runs
      out.stopped_stage1 = true;
      out.raw_mean_u = s1_mean_0;
      out.best_mean_u = s1_mean_0;
      out.best_action = raw_action;
      out.teacher_accepted = false;
      out.teacher_action = raw_action;
      out.chosen_action = raw_action;
      out.overridden = false;
      out.confirmed = false;
      out.stage2_candidate_utilities.clear();
    }
  }
  // Case C: 0.0 < best_s1_mean_d < stage1_min_mean_diff (challenger edged 0 by < min diff)
  else {
    out.stopped_stage1 = true;
    out.raw_mean_u = s1_mean_0;
    out.best_mean_u = s1_mean_0;
    out.best_action = raw_action;
    out.teacher_accepted = false;
    out.teacher_action = raw_action;
    out.chosen_action = raw_action;
    out.overridden = false;
    out.confirmed = false;
    out.stage2_candidate_utilities.clear();
  }
  return out;
}

}  // namespace dune_imperium
}  // namespace open_spiel

#endif  // OPEN_SPIEL_EXAMPLES_DUNE_COMPOUND_TURN_DECISION_HELPERS_H_
