#ifndef OPEN_SPIEL_EXAMPLES_DUNE_WIDE_ROLLOUT_V2_H_
#define OPEN_SPIEL_EXAMPLES_DUNE_WIDE_ROLLOUT_V2_H_

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <memory>
#include <numeric>
#include <tuple>
#include <utility>
#include <vector>

#include "open_spiel/examples/dune_compound_turn_search.h"
#include "open_spiel/examples/dune_wide_rollout_helpers.h"
#include "open_spiel/examples/dune_wide_rollout_types.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace dune_imperium {

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
    int* chosen_candidate_index = nullptr,
    const SearchV2TuningConfig& tuning = SearchV2TuningConfig{},
    double win_speed_tiebreak_eps = 0.0,
    bool ceiling_aware_margin = true) {
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

  const bool caller_small_cap = (max_inflight_worlds > 0 && max_inflight_worlds < 96);

  // 1. Capped Stage-1 Screen
  int r_stage1 = 2;
  if (!has_deadline) {
    if (!tuning.unclocked_c96_3arm_mode) {
      r_stage1 = std::clamp(total_budget / (2 * static_cast<int>(K)), 2, tuning.r_stage1_cap);
    } else {
      const int raw_s1 = static_cast<int>(std::llround(
          (tuning.s1_budget_frac * static_cast<double>(total_budget)) / static_cast<double>(K)));
      r_stage1 = std::clamp(raw_s1, 2, tuning.r_stage1_cap);
    }
  } else {
    r_stage1 = std::clamp(
        static_cast<int>((tuning.s1_budget_frac * max_time_ms) / std::max(0.5, static_cast<double>(K) * est_ms_per_sim)),
        2, tuning.r_stage1_cap);
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

  int s1_pool_cap = std::max(2, 512 / static_cast<int>(K));
  if (tuning.legacy_s1_inflight_cap) {
    s1_pool_cap = std::max(2, std::min(max_inflight_worlds, 192 / static_cast<int>(K)));
  } else if (caller_small_cap) {
    s1_pool_cap = std::max(2, std::min(max_inflight_worlds, s1_pool_cap));
  }
  const int s1_inflight = std::clamp(r_stage1, 2, s1_pool_cap);

  std::vector<size_t> candidate_indices(K);
  std::iota(candidate_indices.begin(), candidate_indices.end(), 0);

  std::vector<std::vector<double>> per_candidate_rollouts(K);
  for (size_t c = 0; c < K; ++c) {
    per_candidate_rollouts[c].reserve(r_stage1);
  }
  int new_rollouts_executed = 0;
  const auto s1_t0 = std::chrono::steady_clock::now();

  if (has_deadline && max_time_ms <= tuning.subwave_1a_max_budget_ms) {
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
        second_actions, graft_actions, win_speed_tiebreak_eps);

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
          second_actions, graft_actions, win_speed_tiebreak_eps);
    }
  } else {
    RunClockedPairedRolloutStage(
        state, dune, search_player, evaluators, ordered_candidates,
        candidate_indices, r_stage1, /*world_seed_offset=*/0,
        decision_seed, crn_mode, /*sample_opponent_leader_draft=*/false,
        stage1_deadline, active_tree, per_candidate_rollouts,
        new_rollouts_executed, s1_inflight, /*resample_from_infostate=*/true,
        second_actions, graft_actions, win_speed_tiebreak_eps);
  }

  const auto s1_t1 = std::chrono::steady_clock::now();
  const double stage1_elapsed_ms =
      std::chrono::duration<double, std::milli>(s1_t1 - s1_t0).count();
  const int stage1_rollouts_executed = new_rollouts_executed;

  const int r_stage1_done = per_candidate_rollouts.empty() ? 0 : static_cast<int>(per_candidate_rollouts[0].size());

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
        r_stage1, r_stage1_done,
        /*stage2_planned_worlds=*/0, /*stage2_kept_worlds=*/0,
        stage1_rollouts_executed, /*stage2_rollouts_executed=*/0,
        stage1_elapsed_ms, /*stage2_elapsed_ms=*/0.0,
        "stage1_starved");
  }

  double raw_mean_1 = 0.0;
  for (int r = 0; r < r_stage1_done; ++r) {
    raw_mean_1 += per_candidate_rollouts[0][r];
  }
  raw_mean_1 /= r_stage1_done;

  std::vector<double> mean_1(K, 0.0);
  std::vector<double> mean_d_1(K, 0.0);
  std::vector<double> penalized_adv_1(K, 0.0);
  mean_1[0] = raw_mean_1;
  mean_d_1[0] = 0.0;
  penalized_adv_1[0] = 0.0;

  size_t best_c_idx = 1;
  double highest_penalized_adv_1 = -1e9;

  for (size_t c_idx = 1; c_idx < K; ++c_idx) {
    double sum_u = 0.0;
    for (int r = 0; r < r_stage1_done; ++r) {
      sum_u += per_candidate_rollouts[c_idx][r];
    }
    mean_1[c_idx] = sum_u / r_stage1_done;
    mean_d_1[c_idx] = mean_1[c_idx] - raw_mean_1;
    penalized_adv_1[c_idx] = mean_d_1[c_idx] - prior_penalty[c_idx];
    if (penalized_adv_1[c_idx] > highest_penalized_adv_1) {
      highest_penalized_adv_1 = penalized_adv_1[c_idx];
      best_c_idx = c_idx;
    }
  }

  size_t second_best_c_idx = 0;
  double second_highest_penalized_adv_1 = -1e9;
  if (K >= 3) {
    for (size_t c_idx = 1; c_idx < K; ++c_idx) {
      if (c_idx == best_c_idx) continue;
      if (penalized_adv_1[c_idx] > second_highest_penalized_adv_1) {
        second_highest_penalized_adv_1 = penalized_adv_1[c_idx];
        second_best_c_idx = c_idx;
      }
    }
  }

  // 2. Smart Stage-1 Early-Stop
  const double best_challenger_adv_1 = penalized_adv_1[best_c_idx];
  const bool high_conf_skip =
      (stage1_high_conf_prior_skip > 0.0 &&
       !candidate_priors.empty() &&
       candidate_priors[0] >= stage1_high_conf_prior_skip &&
       best_challenger_adv_1 <= 0.05);
  const bool blowout_skip = (r_stage1_done >= 12 && best_challenger_adv_1 < -0.20);
  const bool tight_clock_skip =
      (has_deadline && stage1_elapsed_ms >= 0.78 * max_time_ms && best_challenger_adv_1 <= 0.0);

  if (high_conf_skip || blowout_skip || tight_clock_skip) {
    return BuildCompoundSearchResult(
        raw_action, false, raw_mean_1, raw_mean_1,
        prior_rollouts_reused, new_rollouts_executed,
        ordered_candidates, mean_1, per_candidate_rollouts,
        r_stage1, r_stage1_done,
        /*stage2_planned_worlds=*/0, /*stage2_kept_worlds=*/0,
        stage1_rollouts_executed, /*stage2_rollouts_executed=*/0,
        stage1_elapsed_ms, /*stage2_elapsed_ms=*/0.0,
        high_conf_skip ? "stage1_high_conf_skip" : "early_stop");
  }

  // 3. Adaptive Stage-2 Wave Ramp + High-Confidence Convergence Stop
  std::vector<size_t> active_stage2_cands = {0, best_c_idx};
  if (tuning.enable_conditional_3arm_stage2 && K >= 3 && second_best_c_idx != 0) {
    if (has_deadline) {
      const double entry_elapsed_ms = std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - start_time).count();
      const double entry_rem_ms = std::max(0.0, 0.985 * max_time_ms - entry_elapsed_ms);
      if (entry_rem_ms >= tuning.three_arm_min_rem_ms &&
          (penalized_adv_1[best_c_idx] - penalized_adv_1[second_best_c_idx]) <= tuning.three_arm_max_s1_gap) {
        active_stage2_cands = {0, best_c_idx, second_best_c_idx};
      }
    } else if (tuning.unclocked_c96_3arm_mode) {
      const int rem_budget = std::max(0, total_budget - static_cast<int>(K) * r_stage1_done);
      if ((rem_budget / 3) >= tuning.unclocked_3arm_min_s2_worlds &&
          (penalized_adv_1[best_c_idx] - penalized_adv_1[second_best_c_idx]) <= tuning.three_arm_max_s1_gap) {
        active_stage2_cands = {0, best_c_idx, second_best_c_idx};
      }
    }
  }

  int stage2_planned_worlds = 0;
  int stage2_kept_worlds = 0;
  const auto s2_t0 = std::chrono::steady_clock::now();

  const int worker_s2_cap =
      128 * std::max<int>(1, static_cast<int>(evaluators.size()));
  if (!has_deadline) {
    const int s2_divisor = tuning.unclocked_c96_3arm_mode
        ? std::max(1, static_cast<int>(active_stage2_cands.size()))
        : 2;
    int r_stage2 = std::max(16, (total_budget - static_cast<int>(K) * r_stage1_done) / s2_divisor);
    stage2_planned_worlds = r_stage2;
    const int num_active_s2 = static_cast<int>(active_stage2_cands.size());
    int s2_pool_cap = std::max(2, 512 / num_active_s2);
    int s2_cap = (tuning.legacy_clamp_session_inflight || caller_small_cap)
        ? std::min(max_inflight_worlds, std::min(tuning.s2_inflight_cap, worker_s2_cap))
        : std::min(s2_pool_cap, std::min(tuning.s2_inflight_cap, worker_s2_cap));
    int s2_inflight = std::min(r_stage2, std::max(2, s2_cap));
    RunClockedPairedRolloutStage(
        state, dune, search_player, evaluators, ordered_candidates,
        active_stage2_cands, r_stage2, /*world_seed_offset=*/r_stage1_done,
        decision_seed, crn_mode, /*sample_opponent_leader_draft=*/false,
        final_deadline, active_tree, per_candidate_rollouts,
        new_rollouts_executed, s2_inflight, /*resample_from_infostate=*/true,
        second_actions, graft_actions, win_speed_tiebreak_eps);
  } else {
    int seed_offset = r_stage1_done;
    const double s1_ms_per_sim = (r_stage1_done > 0 && stage1_elapsed_ms > 0.0)
        ? (stage1_elapsed_ms / static_cast<double>(K * r_stage1_done))
        : est_ms_per_sim;
    double cur_ms_per_sim = (tuning.seed_s2_ms_with_min_prior && est_ms_per_sim > 0.0)
        ? std::min(s1_ms_per_sim, est_ms_per_sim)
        : s1_ms_per_sim;

    while (true) {
      const double total_elapsed_ms = std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - start_time).count();
      const int s2_done_now = static_cast<int>(per_candidate_rollouts[0].size()) - r_stage1_done;
      if (s2_done_now >= 2 && total_elapsed_ms >= 0.935 * max_time_ms) {
        break;
      }
      const double rem_ms = std::max(0.0, 0.985 * max_time_ms - total_elapsed_ms);
      if (rem_ms <= 0.0) break;

      const int num_active_s2 = static_cast<int>(active_stage2_cands.size());
      const double arms_d = static_cast<double>(num_active_s2);
      int wave_worlds = 0;
      if (s2_done_now == 0) {
        int max_w0 = (rem_ms < 4000.0 ? 12 : tuning.s2_max_w0);
        wave_worlds = std::clamp(
            static_cast<int>((0.45 * rem_ms) / std::max(0.1, arms_d * cur_ms_per_sim)),
            2, max_w0);
      } else {
        wave_worlds = std::clamp(
            static_cast<int>(rem_ms / std::max(0.1, arms_d * cur_ms_per_sim)),
            2, tuning.s2_max_wave_worlds);
      }

      const int s2_pool_cap = std::max(2, 512 / num_active_s2);
      const int s2_cap = (tuning.legacy_clamp_session_inflight || caller_small_cap)
          ? std::min(max_inflight_worlds, std::min(tuning.s2_inflight_cap, worker_s2_cap))
          : std::min(s2_pool_cap, std::min(tuning.s2_inflight_cap, worker_s2_cap));
      int s2_inflight = std::min(wave_worlds, std::max(2, s2_cap));

      stage2_planned_worlds += wave_worlds;
      const int before_size = static_cast<int>(per_candidate_rollouts[0].size());
      const auto wt0 = std::chrono::steady_clock::now();
      RunClockedPairedRolloutStage(
          state, dune, search_player, evaluators, ordered_candidates,
          active_stage2_cands, wave_worlds, /*world_seed_offset=*/seed_offset,
          decision_seed, crn_mode, /*sample_opponent_leader_draft=*/false,
          final_deadline, active_tree, per_candidate_rollouts,
          new_rollouts_executed, s2_inflight, /*resample_from_infostate=*/true,
          second_actions, graft_actions, win_speed_tiebreak_eps);
      const int wave_kept = static_cast<int>(per_candidate_rollouts[0].size()) - before_size;
      const double wt_ms = std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - wt0).count();
      seed_offset += wave_worlds;
      if (wave_kept < wave_worlds || wave_kept <= 0) {
        break;
      }
      if (wt_ms > 0.0) {
        const double measured_ms = wt_ms / static_cast<double>(num_active_s2 * wave_kept);
        if (tuning.seed_s2_ms_with_min_prior && s2_done_now == 0) {
          cur_ms_per_sim = measured_ms;
        } else {
          cur_ms_per_sim = 0.25 * cur_ms_per_sim + 0.75 * measured_ms;
        }
      }

      const int cur_s2_done = static_cast<int>(per_candidate_rollouts[0].size()) - r_stage1_done;

      if (active_stage2_cands.size() == 3 && cur_s2_done >= 128) {
        const size_t c1 = active_stage2_cands[1];
        const size_t c2 = active_stage2_cands[2];
        double sum_c1 = 0.0, sum_c2 = 0.0;
        for (int r = 0; r < cur_s2_done; ++r) {
          sum_c1 += per_candidate_rollouts[c1][r_stage1_done + r];
          sum_c2 += per_candidate_rollouts[c2][r_stage1_done + r];
        }
        const double mean_s2_c1 = sum_c1 / cur_s2_done;
        const double mean_s2_c2 = sum_c2 / cur_s2_done;
        const double mean_diff_pair = mean_s2_c1 - mean_s2_c2;
        double sq_diff_pair = 0.0;
        for (int r = 0; r < cur_s2_done; ++r) {
          double diff = (per_candidate_rollouts[c1][r_stage1_done + r] -
                         per_candidate_rollouts[c2][r_stage1_done + r]) - mean_diff_pair;
          sq_diff_pair += diff * diff;
        }
        const double var_pair = sq_diff_pair / (cur_s2_done - 1);
        const double se_pair = std::sqrt(var_pair / cur_s2_done);
        const double score1 = mean_s2_c1 - prior_penalty[c1];
        const double score2 = mean_s2_c2 - prior_penalty[c2];
        const size_t lead_c = (score1 >= score2) ? c1 : c2;
        const double gap = std::abs(score1 - score2);
        if ((se_pair > 1e-9 && gap >= tuning.three_arm_prune_se * se_pair) ||
            (se_pair <= 1e-9 && gap > 0.0)) {
          active_stage2_cands = {0, lead_c};
          best_c_idx = lead_c;
        }
      }

      if (cur_s2_done >= 128) {
        const int tot_done_now = r_stage1_done + cur_s2_done;
        double sum_raw_now = 0.0;
        for (int r = 0; r < tot_done_now; ++r) {
          sum_raw_now += per_candidate_rollouts[0][r];
        }
        const double raw_mean_now =
            (tot_done_now > 0) ? (sum_raw_now / tot_done_now) : 0.0;
        const double base_eff_m_now = ComputeEffectiveOverrideMargin(
            min_override_margin, raw_mean_now, win_speed_tiebreak_eps,
            ceiling_aware_margin);
        if (active_stage2_cands.size() == 2) {
          const size_t cur_best = active_stage2_cands[1];
          const double eff_m = base_eff_m_now + prior_penalty[cur_best];
          double sum_d_s2 = 0.0;
          for (int r = 0; r < cur_s2_done; ++r) {
            sum_d_s2 += (per_candidate_rollouts[cur_best][r_stage1_done + r] -
                         per_candidate_rollouts[0][r_stage1_done + r]);
          }
          const double mean_d_s2 = sum_d_s2 / cur_s2_done;
          double sq_diff_s2 = 0.0;
          for (int r = 0; r < cur_s2_done; ++r) {
            double diff = (per_candidate_rollouts[cur_best][r_stage1_done + r] -
                           per_candidate_rollouts[0][r_stage1_done + r]) - mean_d_s2;
            sq_diff_s2 += diff * diff;
          }
          const double var_d_s2 = sq_diff_s2 / (cur_s2_done - 1);
          const double se_d_s2 = std::sqrt(var_d_s2 / cur_s2_done);
          if (se_d_s2 > 1e-9 &&
              (mean_d_s2 < eff_m - 3.5 * se_d_s2 ||
               (mean_d_s2 > eff_m + 3.5 * se_d_s2 && mean_d_s2 / se_d_s2 >= 2.5))) {
            break;
          }
        } else if (active_stage2_cands.size() == 3) {
          const size_t c1 = active_stage2_cands[1];
          const size_t c2 = active_stage2_cands[2];
          const double eff_m1 = base_eff_m_now + prior_penalty[c1];
          const double eff_m2 = base_eff_m_now + prior_penalty[c2];

          double sum_d1 = 0.0, sum_d2 = 0.0;
          for (int r = 0; r < cur_s2_done; ++r) {
            sum_d1 += (per_candidate_rollouts[c1][r_stage1_done + r] -
                       per_candidate_rollouts[0][r_stage1_done + r]);
            sum_d2 += (per_candidate_rollouts[c2][r_stage1_done + r] -
                       per_candidate_rollouts[0][r_stage1_done + r]);
          }
          const double mean_d1 = sum_d1 / cur_s2_done;
          const double mean_d2 = sum_d2 / cur_s2_done;
          double sq_diff1 = 0.0, sq_diff2 = 0.0;
          for (int r = 0; r < cur_s2_done; ++r) {
            double diff1 = (per_candidate_rollouts[c1][r_stage1_done + r] -
                            per_candidate_rollouts[0][r_stage1_done + r]) - mean_d1;
            double diff2 = (per_candidate_rollouts[c2][r_stage1_done + r] -
                            per_candidate_rollouts[0][r_stage1_done + r]) - mean_d2;
            sq_diff1 += diff1 * diff1;
            sq_diff2 += diff2 * diff2;
          }
          const double se1 = std::sqrt((sq_diff1 / (cur_s2_done - 1)) / cur_s2_done);
          const double se2 = std::sqrt((sq_diff2 / (cur_s2_done - 1)) / cur_s2_done);

          const bool both_lose = (se1 > 1e-9 && mean_d1 < eff_m1 - 3.5 * se1) &&
                                 (se2 > 1e-9 && mean_d2 < eff_m2 - 3.5 * se2);
          if (both_lose) {
            break;
          }
        }
      }
    }
  }

  const auto s2_t1 = std::chrono::steady_clock::now();
  const double stage2_elapsed_ms =
      std::chrono::duration<double, std::milli>(s2_t1 - s2_t0).count();
  const int stage2_rollouts_executed = new_rollouts_executed - stage1_rollouts_executed;

  const int r_stage2_done = static_cast<int>(per_candidate_rollouts[0].size()) - r_stage1_done;
  stage2_kept_worlds = r_stage2_done;

  auto compute_all_candidate_means = [&]() {
    std::vector<double> means(K, 0.0);
    for (size_t c_idx = 0; c_idx < K; ++c_idx) {
      double u_sum = 0.0;
      for (double val : per_candidate_rollouts[c_idx]) u_sum += val;
      means[c_idx] = per_candidate_rollouts[c_idx].empty()
          ? 0.0
          : (u_sum / per_candidate_rollouts[c_idx].size());
    }
    return means;
  };

  if (r_stage2_done < 2) {
    const std::vector<double> candidate_means = compute_all_candidate_means();
    return BuildCompoundSearchResult(
        raw_action, false, candidate_means[0], candidate_means[0],
        prior_rollouts_reused, new_rollouts_executed,
        ordered_candidates, candidate_means, per_candidate_rollouts,
        r_stage1, r_stage1_done,
        stage2_planned_worlds, stage2_kept_worlds,
        stage1_rollouts_executed, stage2_rollouts_executed,
        stage1_elapsed_ms, stage2_elapsed_ms,
        "stage2_starved");
  }

  const int R_total = r_stage1_done + r_stage2_done;

  auto compute_candidate_stats = [&](size_t c_idx) {
    const PairedEbDiffStats s2_eb = ComputePairedEbDiffStats(
        per_candidate_rollouts[c_idx].data() + r_stage1_done,
        per_candidate_rollouts[0].data() + r_stage1_done,
        r_stage2_done);
    const double mean_d_s2 = s2_eb.mean_d;
    const double se_d_s2 = s2_eb.se_eb;
    const double t_stat_s2 = s2_eb.t_stat;

    double sum_raw = 0.0;
    for (int r = 0; r < R_total; ++r) {
      sum_raw += per_candidate_rollouts[0][r];
    }
    const double mean_raw = sum_raw / R_total;
    const PairedEbDiffStats tot_eb = ComputePairedEbDiffStats(
        per_candidate_rollouts[c_idx].data(),
        per_candidate_rollouts[0].data(),
        R_total);
    const double mean_d_tot = tot_eb.mean_d;
    const double t_stat_tot = tot_eb.t_stat;

    const double base_eff_m = ComputeEffectiveOverrideMargin(
        min_override_margin, mean_raw, win_speed_tiebreak_eps,
        ceiling_aware_margin);
    const double eff_m = base_eff_m + prior_penalty[c_idx];
    bool passes = false;
    if (r_stage2_done >= 16) {
      passes = (mean_d_s2 >= eff_m &&
                t_stat_s2 >= 1.50 &&
                mean_d_tot >= 0.75 * eff_m &&
                PassesSingleFlipGuard(r_stage2_done, 1.50, s2_eb.positive_diff_count));
    } else {
      passes = (mean_d_s2 > 0.0 &&
                mean_d_tot >= eff_m &&
                t_stat_tot >= 1.50 &&
                PassesSingleFlipGuard(R_total, 1.50, tot_eb.positive_diff_count));
    }
    return std::make_tuple(passes, mean_d_s2, se_d_s2, mean_d_s2 - prior_penalty[c_idx]);
  };

  bool should_override = false;
  double final_mean_d_s2 = 0.0;
  double final_se_d_s2 = 0.0;

  if (active_stage2_cands.size() == 2) {
    best_c_idx = active_stage2_cands[1];
    auto [passes, m_d, se_d, score] = compute_candidate_stats(best_c_idx);
    should_override = passes;
    final_mean_d_s2 = m_d;
    final_se_d_s2 = se_d;
  } else if (active_stage2_cands.size() == 3) {
    const size_t c1 = active_stage2_cands[1];
    const size_t c2 = active_stage2_cands[2];
    auto [p1, m_d1, se_d1, score1] = compute_candidate_stats(c1);
    auto [p2, m_d2, se_d2, score2] = compute_candidate_stats(c2);

    if (p1 && !p2) {
      best_c_idx = c1;
      should_override = true;
      final_mean_d_s2 = m_d1;
      final_se_d_s2 = se_d1;
    } else if (!p1 && p2) {
      best_c_idx = c2;
      should_override = true;
      final_mean_d_s2 = m_d2;
      final_se_d_s2 = se_d2;
    } else if (p1 && p2) {
      best_c_idx = (score1 >= score2) ? c1 : c2;
      should_override = true;
      final_mean_d_s2 = (best_c_idx == c1) ? m_d1 : m_d2;
      final_se_d_s2 = (best_c_idx == c1) ? se_d1 : se_d2;
    } else {
      best_c_idx = (score1 >= score2) ? c1 : c2;
      should_override = false;
      final_mean_d_s2 = (best_c_idx == c1) ? m_d1 : m_d2;
      final_se_d_s2 = (best_c_idx == c1) ? se_d1 : se_d2;
    }
  }

  Action best_action = raw_action;
  bool overridden = false;
  if (should_override) {
    best_action = ordered_candidates[best_c_idx];
    overridden = true;
    if (chosen_candidate_index != nullptr) {
      *chosen_candidate_index = static_cast<int>(best_c_idx);
    }
  }

  const std::vector<double> candidate_means = compute_all_candidate_means();
  return BuildCompoundSearchResult(
      best_action, overridden, candidate_means[0],
      overridden ? candidate_means[best_c_idx] : candidate_means[0],
      prior_rollouts_reused, new_rollouts_executed,
      ordered_candidates, candidate_means, per_candidate_rollouts,
      r_stage1, r_stage1_done,
      stage2_planned_worlds, stage2_kept_worlds,
      stage1_rollouts_executed, stage2_rollouts_executed,
      stage1_elapsed_ms, stage2_elapsed_ms,
      overridden ? "override" : "stage2_no_override",
      kInvalidAction, kInvalidAction,
      final_mean_d_s2, final_se_d_s2);
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
    int* chosen_candidate_index = nullptr,
    const SearchV2TuningConfig& tuning = SearchV2TuningConfig{},
    bool all_buy_candidates = false,
    double win_speed_tiebreak_eps = 0.0,
    bool ceiling_aware_margin = true) {
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
      SelectOrderedCandidatesAndPriors(
          raw_prior, legal, fixed_top_k,
          all_buy_candidates && IsBuyDecisionState(state));

  return SearchV2SuccessiveHalvingDecision(
      state, search_player, turn_tree, evaluators, ordered_candidates,
      candidate_priors, reference_rollouts_per_action, min_override_margin,
      decision_seed, crn_mode, max_time_ms, max_inflight_worlds,
      prior_log_ratio_weight, stage1_high_conf_prior_skip, second_actions,
      /*graft_actions=*/nullptr, est_ms_per_sim, chosen_candidate_index, tuning,
      win_speed_tiebreak_eps, ceiling_aware_margin);
}

}  // namespace dune_imperium
}  // namespace open_spiel

#endif  // OPEN_SPIEL_EXAMPLES_DUNE_WIDE_ROLLOUT_V2_H_
