#ifndef OPEN_SPIEL_EXAMPLES_DUNE_WIDE_ROLLOUT_JOINT_H_
#define OPEN_SPIEL_EXAMPLES_DUNE_WIDE_ROLLOUT_JOINT_H_

#include <algorithm>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "open_spiel/examples/dune_compound_turn_search.h"
#include "open_spiel/examples/dune_wide_rollout_helpers.h"
#include "open_spiel/examples/dune_wide_rollout_types.h"
#include "open_spiel/examples/dune_wide_rollout_v2.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace dune_imperium {

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
    size_t max_candidates = 8,
    const SearchV2TuningConfig& tuning = SearchV2TuningConfig{},
    double win_speed_tiebreak_eps = 0.0,
    bool ceiling_aware_margin = true) {
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
      &chosen_c_idx, tuning, win_speed_tiebreak_eps, ceiling_aware_margin);

  JointSearchV2Result res;
  res.base = base_res;
  res.raw_first_action = joint_cands[0].first_action;
  res.raw_graft_action = joint_cands[0].graft_action;
  res.raw_second_action = joint_cands[0].second_action;
  res.chosen_candidate_index = chosen_c_idx;
  if (chosen_c_idx >= 0 && chosen_c_idx < static_cast<int>(joint_cands.size())) {
    res.chosen_first_action = joint_cands[chosen_c_idx].first_action;
    res.chosen_graft_action = joint_cands[chosen_c_idx].graft_action;
    res.chosen_second_action = joint_cands[chosen_c_idx].second_action;
  } else {
    res.chosen_first_action = base_res.action;
    res.chosen_graft_action = kInvalidAction;
    res.chosen_second_action = kInvalidAction;
  }
  for (const auto& kv : base_res.candidate_rollout_utilities) {
    res.per_candidate_rollouts.push_back(kv.second);
  }
  res.joint_candidates = std::move(joint_cands);
  return res;
}

inline int ComputeAgentTurnHandPlusPlayed(
    const DuneImperiumState* dune_state, Player p) {
  if (dune_state == nullptr || p < 0 || p >= kNumPlayers) return -1;
  return dune_state->GetPlayerCardsInHand(p) +
         static_cast<int>(dune_state->GetPlayedAgentCardsForTesting(p).size());
}

inline bool HasStage2ConfirmedJointAdvantage(
    const std::vector<std::vector<double>>& rollouts,
    int chosen_c,
    int alt_idx,
    int stage1_kept_worlds,
    double min_s2_margin,
    double min_total_margin = 0.10,
    int min_s2_worlds = 24,
    double* out_alt_mean = nullptr,
    double* out_chosen_mean = nullptr,
    int* out_s2_worlds = nullptr,
    double* out_s2_diff = nullptr,
    double win_speed_tiebreak_eps = 0.0,
    bool ceiling_aware_margin = true) {
  if (chosen_c < 0 || alt_idx < 0 || chosen_c == alt_idx) return false;
  if (chosen_c >= static_cast<int>(rollouts.size()) ||
      alt_idx >= static_cast<int>(rollouts.size())) {
    return false;
  }
  if (stage1_kept_worlds < 0) return false;

  const int n_shared = static_cast<int>(
      std::min(rollouts[chosen_c].size(), rollouts[alt_idx].size()));
  const int n_s2 = n_shared - stage1_kept_worlds;
  if (n_s2 < min_s2_worlds) return false;

  double sum_s2_chosen = 0.0;
  double sum_s2_alt = 0.0;
  for (int i = stage1_kept_worlds; i < n_shared; ++i) {
    sum_s2_chosen += rollouts[chosen_c][i];
    sum_s2_alt += rollouts[alt_idx][i];
  }
  const double mean_s2_chosen = sum_s2_chosen / static_cast<double>(n_s2);
  const double mean_s2_alt = sum_s2_alt / static_cast<double>(n_s2);

  double sum_all_chosen = 0.0;
  double sum_all_alt = 0.0;
  for (int i = 0; i < n_shared; ++i) {
    sum_all_chosen += rollouts[chosen_c][i];
    sum_all_alt += rollouts[alt_idx][i];
  }
  const double mean_all_chosen = sum_all_chosen / static_cast<double>(n_shared);
  const double mean_all_alt = sum_all_alt / static_cast<double>(n_shared);

  if (out_alt_mean != nullptr) *out_alt_mean = mean_all_alt;
  if (out_chosen_mean != nullptr) *out_chosen_mean = mean_all_chosen;
  if (out_s2_worlds != nullptr) *out_s2_worlds = n_s2;
  if (out_s2_diff != nullptr) *out_s2_diff = mean_s2_chosen - mean_s2_alt;

  const double eff_s2_margin = ComputeEffectiveOverrideMargin(
      min_s2_margin, mean_s2_alt, win_speed_tiebreak_eps, ceiling_aware_margin);
  const double eff_total_margin = ComputeEffectiveOverrideMargin(
      min_total_margin, mean_all_alt, win_speed_tiebreak_eps,
      ceiling_aware_margin);

  return (mean_s2_chosen - mean_s2_alt) >= eff_s2_margin &&
         (mean_all_chosen - mean_all_alt) >= eff_total_margin;
}

inline PendingJointFollowUpDecision AdvancePendingJointAgentOverride(
    PendingJointAgentOverride* pending,
    const std::vector<Action>& legal_actions,
    Action step_raw_greedy_action,
    int cur_hand_size,
    int cur_intrigue_count,
    bool saw_chance_node,
    int min_s2_worlds = 24,
    double win_speed_tiebreak_eps = 0.0,
    bool ceiling_aware_margin = true) {
  if (!pending || !pending->active || legal_actions.empty()) {
    return {};
  }

  // Stochastic Tree-Break Guard: if a chance node intervened or private hand/intrigue count changed
  if (saw_chance_node ||
      (pending->root_hand_size >= 0 && cur_hand_size >= 0 && cur_hand_size != pending->root_hand_size) ||
      (pending->root_intrigue_count >= 0 && cur_intrigue_count >= 0 && cur_intrigue_count != pending->root_intrigue_count)) {
    pending->Clear();
    return {};
  }

  const bool all_graft = std::all_of(
      legal_actions.begin(), legal_actions.end(),
      [](Action a) { return IsGraftSelectionAction(a); });

  if (all_graft && pending->expected_graft_action != kInvalidAction) {
    auto it = std::find(legal_actions.begin(), legal_actions.end(), pending->expected_graft_action);
    if (it == legal_actions.end()) {
      pending->Clear();
      return {};
    }
    PendingJointFollowUpDecision out;
    out.execute_override = true;
    out.forced_action = pending->expected_graft_action;
    out.is_graft_step = true;
    out.root_uid = pending->root_uid;
    out.same_card_as_raw = (!pending->joint_candidates.empty() &&
        pending->chosen_first_action == pending->joint_candidates[0].first_action);
    pending->executed_graft_action = pending->expected_graft_action;
    pending->expected_graft_action = kInvalidAction;

    if (legal_actions.size() >= 2 && step_raw_greedy_action != out.forced_action) {
      int best_alt_idx = -1;
      double best_prior = -1.0;
      for (size_t i = 0; i < pending->joint_candidates.size(); ++i) {
        const auto& c = pending->joint_candidates[i];
        if (c.first_action == pending->chosen_first_action &&
            c.graft_action == step_raw_greedy_action) {
          if (c.joint_prior > best_prior) {
            best_prior = c.joint_prior;
            best_alt_idx = static_cast<int>(i);
          }
        }
      }
      if (best_alt_idx >= 0) {
        if (HasStage2ConfirmedJointAdvantage(
                pending->per_candidate_rollouts,
                pending->chosen_candidate_index,
                best_alt_idx,
                pending->stage1_kept_worlds,
                /*min_s2_margin=*/0.10,
                /*min_total_margin=*/0.10,
                min_s2_worlds,
                &out.challenger_mean_q,
                &out.chosen_mean_q,
                &out.shared_s2_worlds,
                &out.mean_s2_diff,
                win_speed_tiebreak_eps,
                ceiling_aware_margin)) {
          out.emit_aux_target = true;
          out.challenger_action = step_raw_greedy_action;
        }
      }
    }
    if (pending->expected_second_action == kInvalidAction) {
      pending->Clear();
    }
    return out;
  }

  const bool all_space = std::all_of(
      legal_actions.begin(), legal_actions.end(),
      [](Action a) { return IsBoardSpaceAction(a); });

  if (all_space && pending->expected_second_action != kInvalidAction) {
    auto it = std::find(legal_actions.begin(), legal_actions.end(), pending->expected_second_action);
    if (it == legal_actions.end()) {
      pending->Clear();
      return {};
    }
    PendingJointFollowUpDecision out;
    out.execute_override = true;
    out.forced_action = pending->expected_second_action;
    out.is_graft_step = false;
    out.root_uid = pending->root_uid;
    out.same_card_as_raw = (!pending->joint_candidates.empty() &&
        pending->chosen_first_action == pending->joint_candidates[0].first_action);

    if (legal_actions.size() >= 2 && step_raw_greedy_action != out.forced_action) {
      int best_alt_idx = -1;
      double best_prior = -1.0;
      for (size_t i = 0; i < pending->joint_candidates.size(); ++i) {
        const auto& c = pending->joint_candidates[i];
        if (c.first_action == pending->chosen_first_action &&
            c.graft_action == pending->executed_graft_action &&
            c.second_action == step_raw_greedy_action) {
          if (c.joint_prior > best_prior) {
            best_prior = c.joint_prior;
            best_alt_idx = static_cast<int>(i);
          }
        }
      }
      if (best_alt_idx >= 0) {
        if (HasStage2ConfirmedJointAdvantage(
                pending->per_candidate_rollouts,
                pending->chosen_candidate_index,
                best_alt_idx,
                pending->stage1_kept_worlds,
                /*min_s2_margin=*/0.05,
                /*min_total_margin=*/0.10,
                min_s2_worlds,
                &out.challenger_mean_q,
                &out.chosen_mean_q,
                &out.shared_s2_worlds,
                &out.mean_s2_diff,
                win_speed_tiebreak_eps,
                ceiling_aware_margin)) {
          out.emit_aux_target = true;
          out.challenger_action = step_raw_greedy_action;
        }
      }
    }
    pending->Clear();
    return out;
  }

  pending->Clear();
  return {};
}

}  // namespace dune_imperium
}  // namespace open_spiel

#endif  // OPEN_SPIEL_EXAMPLES_DUNE_WIDE_ROLLOUT_JOINT_H_
