#ifndef OPEN_SPIEL_EXAMPLES_DUNE_WIDE_ROLLOUT_TYPES_H_
#define OPEN_SPIEL_EXAMPLES_DUNE_WIDE_ROLLOUT_TYPES_H_

#include <algorithm>
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <utility>
#include <vector>

#include "open_spiel/examples/dune_compound_turn_search.h"

namespace open_spiel {
namespace dune_imperium {

struct WideRolloutSearchConfig {
  double win_speed_tiebreak_eps = 0.0;
  bool ceiling_aware_margin = true;
  bool unbiased_stage2_gate = false;
  bool all_buy_candidates = false;
};

struct PairedRolloutCorrelationStats {
  double sum_corr = 0.0;
  int corr_pair_count = 0;
  double sum_diff_var = 0.0;
  int diff_var_pair_count = 0;
  int pair_count = 0;
  double mean_paired_corr = 0.0;
  double mean_paired_diff_var = 0.0;
};

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

inline constexpr int kNumBoardSpaces = 22;

inline bool IsBoardSpaceAction(Action a) {
  return a >= kActionAgentSpaceConspire && a <= kActionAgentSpaceSietchTabr;
}

inline bool IsGraftSelectionAction(Action a) {
  return (a >= kActionSelectGraftPartner0 &&
          a < kActionSelectGraftPartner0 + kNumImperiumCardIds) ||
         a == kActionPlayAgentSolo;
}

inline bool IsSelectGraftPartnerAction(Action a) {
  return IsGraftSelectionAction(a);
}

inline bool IsSelectAgentCardAction(Action a) {
  return a >= kActionSelectAgentCard0 && a < kActionSelectAgentCard0 + 256;
}

struct MacroSpaceConfig {
  bool enabled = false;
  int min_round = 1;
};

inline bool IsRescuableOptionalAbilityAction(Action a, bool is_buy_state) {
  if (is_buy_state) {
    return a == kActionFamilyAtomics;
  }
  return a == kActionPlayKwisatzHaderach ||
         (a >= 130 && a <= 149) ||
         (a >= 630 && a <= 635) ||
         (a >= 2320 && a <= 2390);
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
  int chosen_candidate_index = 0;
  std::vector<std::vector<double>> per_candidate_rollouts;
};

struct SearchV2TuningConfig {
  // Step 2A batching & wave sizing controls
  double subwave_1a_max_budget_ms = 3000.0;  // Set to 1e9 for legacy always-Subwave-1A; sweep {0.0, 3000.0, 4500.0}
  bool legacy_s1_inflight_cap = false;       // true => min(max_inflight_worlds, 192 / K); false => 512 / K (respecting small CPU test max_inflight_worlds < 96)
  bool seed_s2_ms_with_min_prior = true;     // true => min(s1_ms_per_sim, est_ms_per_sim) initially, then direct measured after wave 0
  int s2_max_w0 = 192;                       // Legacy: 48; Step 2A winner: 192
  int s2_max_wave_worlds = 1024;             // Legacy: 128; Step 2A winner: 1024
  int s2_inflight_cap = 256;                 // Legacy: 96; sweep {96, 256}
  bool legacy_clamp_session_inflight = false;// true => also clamp s2_inflight by max_inflight_worlds (96)

  // Step 2B screening & retention controls (Winner: C96_3arm)
  int r_stage1_cap = 96;                     // Legacy/Step 2A: 16; Step 2B winner: 96
  double s1_budget_frac = 0.30;              // Legacy/Step 2A: 0.35; Step 2B winner: 0.30
  bool enable_conditional_3arm_stage2 = true; // Legacy/Step 2A: false; Step 2B winner: true
  double three_arm_min_rem_ms = 6000.0;
  double three_arm_max_s1_gap = 0.06;
  double three_arm_prune_se = 1.5;
  bool unclocked_c96_3arm_mode = false;
  int unclocked_3arm_min_s2_worlds = 24;

  // Step 3.2 turn reservation controls
  bool enable_rem0_turn_reservation = false;
};

struct PendingJointAgentOverride {
  bool active = false;
  uint64_t root_uid = 0;
  int stage1_kept_worlds = 0;
  int root_hand_size = -1;
  int root_intrigue_count = -1;
  Action chosen_first_action = kInvalidAction;
  Action executed_graft_action = kInvalidAction;
  Action expected_graft_action = kInvalidAction;
  Action expected_second_action = kInvalidAction;
  int chosen_candidate_index = -1;
  std::vector<JointAgentCandidate> joint_candidates;
  std::vector<std::vector<double>> per_candidate_rollouts;

  void Clear() { *this = PendingJointAgentOverride{}; }
};

struct PendingJointFollowUpDecision {
  bool execute_override = false;
  Action forced_action = kInvalidAction;
  bool emit_aux_target = false;
  bool is_graft_step = false;
  uint64_t root_uid = 0;
  bool same_card_as_raw = false;
  Action challenger_action = kInvalidAction;
  double challenger_mean_q = 0.0;
  double chosen_mean_q = 0.0;
  int shared_s2_worlds = 0;
  double mean_s2_diff = 0.0;
};

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

inline std::pair<std::vector<Action>, std::vector<double>>
SelectOrderedCandidatesAndPriorsFromSorted(
    const std::vector<std::pair<Action, double>>& sorted_by_prior,
    int fixed_top_k = 0,
    bool all_buy_candidates = false) {
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
  if (all_buy_candidates) {
    for (const auto& [act, prob] : sorted_by_prior) {
      if (IsCardOrTechAcquisitionAction(act) &&
          std::find(ordered_candidates.begin(), ordered_candidates.end(), act) ==
              ordered_candidates.end()) {
        ordered_candidates.push_back(act);
        candidate_priors.push_back(prob);
      }
    }
  }
  return {ordered_candidates, candidate_priors};
}

inline std::pair<std::vector<Action>, std::vector<double>>
SelectOrderedCandidatesAndPriors(
    const ActionsAndProbs& raw_prior,
    const std::vector<Action>& legal,
    int fixed_top_k = 0,
    bool all_buy_candidates = false) {
  const auto sorted_by_prior = GetSortedCandidatesByPrior(raw_prior, legal);
  return SelectOrderedCandidatesAndPriorsFromSorted(
      sorted_by_prior, fixed_top_k, all_buy_candidates);
}

inline CompoundSearchDecisionResult BuildCompoundSearchResult(
    Action action,
    bool overridden,
    double raw_mean_utility,
    double chosen_mean_utility,
    int prior_rollouts_reused,
    int new_rollouts_executed,
    const std::vector<Action>& ordered_candidates,
    const std::vector<double>& candidate_means,
    const std::vector<std::vector<double>>& per_candidate_rollouts,
    int stage1_planned_worlds,
    int stage1_kept_worlds,
    int stage2_planned_worlds,
    int stage2_kept_worlds,
    int stage1_rollouts_executed,
    int stage2_rollouts_executed,
    double stage1_elapsed_ms,
    double stage2_elapsed_ms,
    const char* termination_reason,
    Action chosen_graft_action = kInvalidAction,
    Action chosen_second_action = kInvalidAction,
    double mean_d_stage2 = 0.0,
    double se_d_stage2 = 0.0) {
  const size_t K = ordered_candidates.size();
  std::vector<std::pair<Action, double>> candidate_utilities;
  candidate_utilities.reserve(K);
  for (size_t c_idx = 0; c_idx < K; ++c_idx) {
    candidate_utilities.push_back({ordered_candidates[c_idx], candidate_means[c_idx]});
  }
  std::vector<std::pair<Action, std::vector<double>>> candidate_rollout_utilities;
  candidate_rollout_utilities.reserve(K);
  for (size_t c_idx = 0; c_idx < K; ++c_idx) {
    candidate_rollout_utilities.push_back({ordered_candidates[c_idx], per_candidate_rollouts[c_idx]});
  }
  CompoundSearchDecisionResult res;
  res.action = action;
  res.overridden = overridden;
  res.raw_mean_utility = raw_mean_utility;
  res.chosen_mean_utility = chosen_mean_utility;
  res.prior_rollouts_reused = prior_rollouts_reused;
  res.new_rollouts_executed = new_rollouts_executed;
  res.candidate_utilities = std::move(candidate_utilities);
  res.candidate_rollout_utilities = std::move(candidate_rollout_utilities);
  res.stage1_planned_worlds = stage1_planned_worlds;
  res.stage1_kept_worlds = stage1_kept_worlds;
  res.stage2_planned_worlds = stage2_planned_worlds;
  res.stage2_kept_worlds = stage2_kept_worlds;
  res.stage1_rollouts_executed = stage1_rollouts_executed;
  res.stage2_rollouts_executed = stage2_rollouts_executed;
  res.stage1_elapsed_ms = stage1_elapsed_ms;
  res.stage2_elapsed_ms = stage2_elapsed_ms;
  res.termination_reason = termination_reason;
  res.chosen_graft_action = chosen_graft_action;
  res.chosen_second_action = chosen_second_action;
  res.mean_d_stage2 = mean_d_stage2;
  res.se_d_stage2 = se_d_stage2;
  return res;
}

}  // namespace dune_imperium
}  // namespace open_spiel

#endif  // OPEN_SPIEL_EXAMPLES_DUNE_WIDE_ROLLOUT_TYPES_H_
