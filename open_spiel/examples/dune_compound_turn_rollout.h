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

#ifndef OPEN_SPIEL_EXAMPLES_DUNE_COMPOUND_TURN_ROLLOUT_H_
#define OPEN_SPIEL_EXAMPLES_DUNE_COMPOUND_TURN_ROLLOUT_H_

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "open_spiel/examples/dune_compound_turn_stats.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace dune_imperium {

inline Action PickGreedyAction(const ActionsAndProbs& prior, const std::vector<Action>& legal_actions);

// Simulate a rollout to terminal or truncation cutoff while recording any actions
// taken by search_player in the active agent turn before kActionEndTurn.
inline RolloutTrajectory SimulateRolloutWithTurnTrajectory(
    State* sim_state,
    Player search_player,
    BatchedNNEvaluator* evaluator,
    uint64_t rollout_seed,
    SearchTruncationMode truncation_mode = SearchTruncationMode::kTerminal,
    SearchRolloutCrnMode crn_mode = SearchRolloutCrnMode::kSequential,
    bool sample_opponent_leader_draft = false,
    const std::chrono::steady_clock::time_point* deadline = nullptr,
    int* out_steps = nullptr,
    int* out_nn_evals = nullptr,
    bool center_4seat_leaf_values = false,
    double win_speed_tiebreak_eps = 0.0,
    const EndgamePly2Config* ply2_cfg = nullptr) {
  std::mt19937_64 rng(rollout_seed);
  int steps = 0;
  const int max_steps = 2500;
  std::vector<Action> intra_turn_actions;
  bool active_turn = true;
  bool ply2_done = false;
  int loc_steps = 0;
  int loc_nn_evals = 0;

  auto apply_action = [&](Action act) {
    sim_state->ApplyAction(act);
    ++loc_steps;
    if (out_steps) ++(*out_steps);
  };
  auto count_nn_eval = [&](int count = 1) {
    loc_nn_evals += count;
    if (out_nn_evals) *out_nn_evals += count;
  };

  const auto* cur_dune = dynamic_cast<const dune_imperium::DuneImperiumState*>(sim_state);
  int start_round = cur_dune ? cur_dune->GetCurrentRound() : 1;

  std::array<int, 15> epoch_draw_ordinal{};
  std::array<int, 15> last_deck_size;
  last_deck_size.fill(-1);
  std::array<int, 15> reshuffle_epoch{};
  std::array<bool, 15> just_reshuffled{};
  std::array<int, 15> last_effective_epoch;
  last_effective_epoch.fill(-1);
  std::array<EpochCardDrawCounter, 15> epoch_card_draws{};

  while (!sim_state->IsTerminal() && steps < max_steps) {
    if (deadline != nullptr && std::chrono::steady_clock::now() >= *deadline) {
      return {0.0, std::move(intra_turn_actions), /*aborted_by_deadline=*/true, loc_steps, loc_nn_evals};
    }
    steps++;
    if (sim_state->IsChanceNode()) {
      auto chance_outcomes = sim_state->ChanceOutcomes();
      if (crn_mode == SearchRolloutCrnMode::kSequential) {
        double u = std::generate_canonical<double, 53>(rng);
        Action ca = open_spiel::SampleAction(chance_outcomes, u).first;
        sim_state->ApplyAction(ca);
      } else if (crn_mode == SearchRolloutCrnMode::kDeckOrderCrn) {
        const int s_id = cur_dune ? cur_dune->CurrentChanceStreamId() : 14;
        if (cur_dune != nullptr) {
          const int rem_size = cur_dune->CurrentChanceDeckRemainingSize();
          const bool will_reshuffle = cur_dune->CurrentChanceWillReshuffle();
          const bool is_tracked_deck =
              (s_id == 0 || s_id == 1 || s_id == 2 || (s_id >= 3 && s_id <= 5) ||
               s_id == 8 || (s_id >= 10 && s_id <= 13));
          if (is_tracked_deck) {
            if (will_reshuffle) {
              reshuffle_epoch[s_id]++;
              epoch_draw_ordinal[s_id] = 0;
              just_reshuffled[s_id] = true;
            } else {
              if (!just_reshuffled[s_id] && last_deck_size[s_id] >= 0 &&
                  rem_size > last_deck_size[s_id]) {
                reshuffle_epoch[s_id]++;
                epoch_draw_ordinal[s_id] = 0;
              }
              just_reshuffled[s_id] = false;
            }
            last_deck_size[s_id] = rem_size;
          }
        }
        const uint64_t k = (s_id == 0 && cur_dune != nullptr)
                               ? static_cast<uint64_t>(cur_dune->GetCurrentRound())
                               : ((static_cast<uint64_t>(reshuffle_epoch[s_id]) << 32) |
                                  static_cast<uint64_t>(epoch_draw_ordinal[s_id]++));
        const uint64_t draw_seed = dune_seed::DeriveSeed(
            rollout_seed, 0x43524E00ULL + static_cast<uint64_t>(s_id), k);
        const double u =
            static_cast<double>(draw_seed >> 11) * (1.0 / 9007199254740992.0);
        Action ca = open_spiel::SampleAction(chance_outcomes, u).first;
        sim_state->ApplyAction(ca);
      } else {
        // SearchRolloutCrnMode::kEventKeyedCrn
        const int s_id = cur_dune ? cur_dune->CurrentChanceStreamId() : 14;
        const bool is_tracked_deck =
            (s_id == 0 || s_id == 1 || s_id == 2 || (s_id >= 3 && s_id <= 5) ||
             s_id == 8 || (s_id >= 10 && s_id <= 13));
        if (cur_dune != nullptr && is_tracked_deck) {
          const int rem_size = cur_dune->CurrentChanceDeckRemainingSize();
          const bool will_reshuffle = cur_dune->CurrentChanceWillReshuffle();
          if (will_reshuffle) {
            reshuffle_epoch[s_id]++;
            epoch_draw_ordinal[s_id] = 0;
            just_reshuffled[s_id] = true;
          } else {
            if (!just_reshuffled[s_id] && last_deck_size[s_id] >= 0 &&
                rem_size > last_deck_size[s_id]) {
              reshuffle_epoch[s_id]++;
              epoch_draw_ordinal[s_id] = 0;
            }
            just_reshuffled[s_id] = false;
          }
          last_deck_size[s_id] = rem_size;
        }

        int effective_epoch = reshuffle_epoch[s_id];
        if (s_id == 0) {
          effective_epoch = (cur_dune != nullptr)
                                ? ((cur_dune->GetCurrentRound() <= 3)
                                       ? 0
                                       : ((cur_dune->GetCurrentRound() <= 6) ? 1 : 2))
                                : reshuffle_epoch[0];
        } else if (s_id >= 10 && s_id <= 13 && cur_dune != nullptr) {
          const int p = s_id - 10;
          const int engine_reshuffles =
              cur_dune->deck_reshuffle_count(p) +
              (cur_dune->CurrentChanceWillReshuffle() ? 1 : 0);
          effective_epoch = std::max(reshuffle_epoch[s_id], engine_reshuffles);
        }

        if (effective_epoch != last_effective_epoch[s_id]) {
          epoch_card_draws[s_id].clear();
          last_effective_epoch[s_id] = effective_epoch;
        }

        if (chance_outcomes.size() == 1) {
          const Action only_ca = chance_outcomes.front().first;
          if (is_tracked_deck) {
            epoch_card_draws[s_id].increment(only_ca);
          }
          sim_state->ApplyAction(only_ca);
          continue;
        }

        if (is_tracked_deck) {
          const int pool_size =
              (cur_dune != nullptr) ? cur_dune->CurrentChanceSourcePoolSize() : 0;
          Action chosen_action = chance_outcomes.front().first;
          uint64_t min_key = std::numeric_limits<uint64_t>::max();
          for (const auto& [cand_action, prob] : chance_outcomes) {
            const int m =
                (pool_size > 0)
                    ? std::max(1, static_cast<int>(std::llround(
                                      prob * static_cast<double>(pool_size))))
                    : 1;
            const int d = epoch_card_draws[s_id].get(cand_action);
            const int total_copies = d + m;
            const uint64_t StreamEpochTag =
                0x4F534343ULL + (static_cast<uint64_t>(s_id) << 16) +
                static_cast<uint64_t>(effective_epoch);
            std::array<uint64_t, 16> copy_keys;
            const int capped_total = std::min(total_copies, 16);
            const int capped_d = std::min(d, capped_total - 1);
            for (int j = 0; j < capped_total; ++j) {
              copy_keys[j] = dune_seed::DeriveSeed(
                  rollout_seed, StreamEpochTag,
                  (static_cast<uint64_t>(cand_action) << 8) +
                      static_cast<uint64_t>(j));
            }
            std::nth_element(copy_keys.begin(), copy_keys.begin() + capped_d,
                             copy_keys.begin() + capped_total);
            const uint64_t cand_key = copy_keys[capped_d];
            if (cand_key < min_key) {
              min_key = cand_key;
              chosen_action = cand_action;
            }
          }
          epoch_card_draws[s_id].increment(chosen_action);
          apply_action(chosen_action);
        } else {
          // Fall back to kDeckOrderCrn formula for non-deck chance streams (6, 7, 9, 14)
          const uint64_t k = (static_cast<uint64_t>(reshuffle_epoch[s_id]) << 32) |
                             static_cast<uint64_t>(epoch_draw_ordinal[s_id]++);
          const uint64_t draw_seed = dune_seed::DeriveSeed(
              rollout_seed, 0x43524E00ULL + static_cast<uint64_t>(s_id), k);
          const double u =
              static_cast<double>(draw_seed >> 11) * (1.0 / 9007199254740992.0);
          Action ca = open_spiel::SampleAction(chance_outcomes, u).first;
          apply_action(ca);
        }
      }
      continue;
    }

    Player cur = sim_state->CurrentPlayer();
    int cur_round = cur_dune ? cur_dune->GetCurrentRound() : 1;

    // Truncation check
    bool should_truncate = false;
    if (truncation_mode == SearchTruncationMode::kNextRoundSearchPlayer &&
        cur == search_player && cur_round > start_round) {
      should_truncate = true;
    } else if (truncation_mode == SearchTruncationMode::kPlus2Rounds &&
               cur == search_player && cur_round >= start_round + 2) {
      should_truncate = true;
    }

    if (should_truncate) {
      double leaf_u = 0.0;
      if (evaluator) {
        if (center_4seat_leaf_values) {
          count_nn_eval(4);
          auto vals = evaluator->Evaluate(*sim_state);
          double mean_v = 0.0;
          if (vals.size() >= 4) {
            mean_v = 0.25 * (vals[0] + vals[1] + vals[2] + vals[3]);
            leaf_u = TruncatedLeafUtility(vals[search_player] - mean_v);
          } else if (!vals.empty()) {
            for (double v : vals) mean_v += v;
            mean_v /= vals.size();
            leaf_u = TruncatedLeafUtility(vals[search_player] - mean_v);
          }
        } else {
          count_nn_eval(1);
          double raw_val = evaluator->EvaluatePlayerValue(*sim_state, search_player);
          leaf_u = TruncatedLeafUtility(raw_val);
        }
      }
      return {leaf_u, std::move(intra_turn_actions), /*aborted_by_deadline=*/false, loc_steps, loc_nn_evals};
    }

    std::vector<Action> legals = sim_state->LegalActions();
    if (legals.empty()) break;

    Action a = kInvalidAction;
    if (legals.size() == 1) {
      a = legals.front();
    } else if (sample_opponent_leader_draft && cur_dune != nullptr &&
               cur_dune->phase() == dune_imperium::GamePhase::kLeaderDraft &&
               cur != search_player) {
      if (evaluator) count_nn_eval();
      ActionsAndProbs prior = evaluator ? evaluator->PriorWithLegals(*sim_state, legals) : ActionsAndProbs{};
      std::vector<double> prior_probs(legals.size(), 0.0);
      double prior_sum = 0.0;
      if (prior.size() == legals.size()) {
        bool aligned = true;
        for (size_t i = 0; i < legals.size(); ++i) {
          if (prior[i].first != legals[i]) {
            aligned = false;
            break;
          }
          prior_probs[i] = std::max(0.0, prior[i].second);
          prior_sum += prior_probs[i];
        }
        if (!aligned) {
          std::fill(prior_probs.begin(), prior_probs.end(), 0.0);
          prior_sum = 0.0;
          for (size_t i = 0; i < legals.size(); ++i) {
            for (const auto& ap : prior) {
              if (ap.first == legals[i]) {
                prior_probs[i] = std::max(0.0, ap.second);
                prior_sum += prior_probs[i];
                break;
              }
            }
          }
        }
      } else {
        for (size_t i = 0; i < legals.size(); ++i) {
          for (const auto& ap : prior) {
            if (ap.first == legals[i]) {
              prior_probs[i] = std::max(0.0, ap.second);
              prior_sum += prior_probs[i];
              break;
            }
          }
        }
      }
      const double uniform_prob = 1.0 / static_cast<double>(legals.size());
      ActionsAndProbs mixture_outcomes;
      mixture_outcomes.reserve(legals.size());
      for (size_t i = 0; i < legals.size(); ++i) {
        double p_prior = (prior_sum > 1e-12) ? (prior_probs[i] / prior_sum) : uniform_prob;
        double p_mix = 0.5 * p_prior + 0.5 * uniform_prob;
        mixture_outcomes.push_back({legals[i], p_mix});
      }
      uint64_t lead_seed = dune_seed::DeriveSeed(rollout_seed, 0x4C445246ULL, static_cast<uint64_t>(cur));
      double u_lead = static_cast<double>(lead_seed >> 11) * (1.0 / 9007199254740992.0);
      a = open_spiel::SampleAction(mixture_outcomes, u_lead).first;
    } else if (ply2_cfg != nullptr && ply2_cfg->enabled && !ply2_done &&
               !active_turn && evaluator != nullptr && cur == search_player &&
               cur_round == start_round && legals.size() >= 2) {
      std::vector<Action> non_spec_legals;
      non_spec_legals.reserve(legals.size());
      for (Action act : legals) {
        if (!IsConvertSpecimenAction(act)) {
          non_spec_legals.push_back(act);
        }
      }
      if (non_spec_legals.size() < 2) {
        count_nn_eval();
        a = evaluator->GreedyActionWithLegals(*sim_state, legals);
      } else {
        count_nn_eval();
        ActionsAndProbs prior2 = evaluator->PriorWithLegals(*sim_state, legals);
        auto sorted_non_spec = GetSortedCandidatesByPrior(prior2, non_spec_legals);
        Action b0 = sorted_non_spec[0].first;
        double p0 = sorted_non_spec[0].second;
        Action b1 = sorted_non_spec[1].first;
        double p1 = sorted_non_spec[1].second;

        if (p0 < ply2_cfg->max_top1_prior && p1 >= ply2_cfg->min_top2_prior) {
          ply2_done = true;
          std::string info_str = sim_state->InformationStateString(search_player);
          size_t info_hash = std::hash<std::string>{}(info_str);
          uint64_t ply2_seed = dune_seed::DeriveSeed(
              0x504C5932ULL, static_cast<uint64_t>(info_hash), static_cast<uint64_t>(steps));
          const int M = std::clamp(ply2_cfg->inner_worlds, 2, 8);
          double sum_diff = 0.0;
          int pos_cnt = 0;
          int completed_pairs = 0;
          for (int m = 0; m < M; ++m) {
            if (deadline != nullptr && std::chrono::steady_clock::now() >= *deadline) {
              break;
            }
            std::mt19937_64 mrng(dune_seed::DeriveSeed(ply2_seed, static_cast<uint64_t>(m)));
            auto rng_func = [&mrng]() {
              return std::generate_canonical<double, 53>(mrng);
            };
            std::unique_ptr<State> s_resamp = nullptr;
            if (cur_dune != nullptr) {
              s_resamp = cur_dune->ResampleFromInfostate(search_player, rng_func);
            }
            if (!s_resamp) {
              s_resamp = sim_state->Clone();
            }
            auto s_b0 = s_resamp->Clone();
            auto s_b1 = std::move(s_resamp);
            s_b0->ApplyAction(b0);
            s_b1->ApplyAction(b1);

            uint64_t inner_seed = dune_seed::DeriveSeed(ply2_seed, 1000ULL + static_cast<uint64_t>(m));
            int inner_steps0 = 0, inner_evals0 = 0;
            int inner_steps1 = 0, inner_evals1 = 0;
            auto traj0 = SimulateRolloutWithTurnTrajectory(
                s_b0.get(), search_player, evaluator, inner_seed,
                truncation_mode, crn_mode, sample_opponent_leader_draft,
                deadline, &inner_steps0, &inner_evals0, center_4seat_leaf_values,
                win_speed_tiebreak_eps, /*ply2_cfg=*/nullptr);
            if (traj0.aborted_by_deadline) break;

            auto traj1 = SimulateRolloutWithTurnTrajectory(
                s_b1.get(), search_player, evaluator, inner_seed,
                truncation_mode, crn_mode, sample_opponent_leader_draft,
                deadline, &inner_steps1, &inner_evals1, center_4seat_leaf_values,
                win_speed_tiebreak_eps, /*ply2_cfg=*/nullptr);
            if (traj1.aborted_by_deadline) break;

            count_nn_eval(inner_evals0 + inner_evals1);
            if (out_steps) *out_steps += (inner_steps0 + inner_steps1);

            double diff = traj1.terminal_utility - traj0.terminal_utility;
            sum_diff += diff;
            if (diff > 1e-9) {
              pos_cnt++;
            }
            completed_pairs++;
          }
          if (completed_pairs == M) {
            double mean_d = sum_diff / static_cast<double>(M);
            if (mean_d >= ply2_cfg->margin && pos_cnt >= 2) {
              a = b1;
            } else {
              a = b0;
            }
          } else {
            a = b0;
          }
        } else {
          a = PickGreedyAction(prior2, legals);
        }
      }
    } else {
      if (evaluator) count_nn_eval();
      a = evaluator ? evaluator->GreedyActionWithLegals(*sim_state, legals) : legals.front();
    }

    if (active_turn) {
      if (cur == search_player) {
        intra_turn_actions.push_back(a);
        if (a == dune_imperium::kActionEndTurn) {
          active_turn = false;
        }
      } else {
        // Another player acted, turn has completed
        active_turn = false;
      }
    }

    apply_action(a);
  }

  // In kNextRoundSearchPlayer or kPlus2Rounds mode, if loop exits without terminal, evaluate value head
  if ((truncation_mode == SearchTruncationMode::kNextRoundSearchPlayer ||
       truncation_mode == SearchTruncationMode::kPlus2Rounds) && !sim_state->IsTerminal()) {
    double leaf_u = 0.0;
    if (evaluator) {
      if (center_4seat_leaf_values) {
        count_nn_eval(4);
        auto vals = evaluator->Evaluate(*sim_state);
        double mean_v = 0.0;
        if (vals.size() >= 4) {
          mean_v = 0.25 * (vals[0] + vals[1] + vals[2] + vals[3]);
          leaf_u = TruncatedLeafUtility(vals[search_player] - mean_v);
        } else if (!vals.empty()) {
          for (double v : vals) mean_v += v;
          mean_v /= vals.size();
          leaf_u = TruncatedLeafUtility(vals[search_player] - mean_v);
        }
      } else {
        count_nn_eval(1);
        double raw_val = evaluator->EvaluatePlayerValue(*sim_state, search_player);
        leaf_u = TruncatedLeafUtility(raw_val);
      }
    }
    return {leaf_u, std::move(intra_turn_actions), /*aborted_by_deadline=*/false, loc_steps, loc_nn_evals};
  }

  // Original fallback: terminal return with optional win-speed tie-breaker
  std::vector<double> ret = sim_state->Returns();
  double u = (static_cast<size_t>(search_player) < ret.size()) ? ret[search_player] : 0.0;
  if (win_speed_tiebreak_eps > 0.0 && u >= 2.24 && sim_state->IsTerminal()) {
    const auto* term_dune =
        (cur_dune != nullptr)
            ? cur_dune
            : dynamic_cast<const dune_imperium::DuneImperiumState*>(sim_state);
    const int actual_rounds = term_dune ? term_dune->GetCurrentRound() : 10;
    u = ApplyTerminalWinSpeedTiebreak(u, actual_rounds, win_speed_tiebreak_eps);
  }
  return {u, std::move(intra_turn_actions), /*aborted_by_deadline=*/false, loc_steps, loc_nn_evals};
}

inline RolloutTrajectory SimulateRemainingTurn(
    State* sim_state,
    Player search_player,
    BatchedNNEvaluator* evaluator,
    uint64_t rollout_seed,
    SearchTruncationMode truncation_mode = SearchTruncationMode::kTerminal,
    SearchRolloutCrnMode crn_mode = SearchRolloutCrnMode::kSequential,
    bool sample_opponent_leader_draft = false,
    const std::chrono::steady_clock::time_point* deadline = nullptr,
    int* out_steps = nullptr,
    int* out_nn_evals = nullptr,
    bool center_4seat_leaf_values = false,
    double win_speed_tiebreak_eps = 0.0) {
  return SimulateRolloutWithTurnTrajectory(
      sim_state, search_player, evaluator, rollout_seed, truncation_mode,
      crn_mode, sample_opponent_leader_draft, deadline, out_steps, out_nn_evals,
      center_4seat_leaf_values, win_speed_tiebreak_eps);
}

// Pick greedy action from prior over legal actions
inline Action PickGreedyAction(const ActionsAndProbs& prior, const std::vector<Action>& legal_actions) {
  if (legal_actions.empty()) return kInvalidAction;
  if (prior.empty()) return legal_actions.front();
  if (prior.size() == legal_actions.size()) {
    bool aligned = true;
    double best_p = -1.0;
    Action best_a = legal_actions.front();
    for (size_t i = 0; i < legal_actions.size(); ++i) {
      if (prior[i].first != legal_actions[i]) {
        aligned = false;
        break;
      }
      if (prior[i].second > best_p) {
        best_p = prior[i].second;
        best_a = prior[i].first;
      }
    }
    if (aligned) return best_a;
  }
  double best_p = -1.0;
  Action best_a = legal_actions.front();
  for (Action a : legal_actions) {
    for (const auto& ap : prior) {
      if (ap.first == a) {
        if (ap.second > best_p) {
          best_p = ap.second;
          best_a = a;
        }
        break;
      }
    }
  }
  return best_a;
}

// Get top-K candidate actions sorted by prior probability
inline std::vector<Action> GetTopKActions(
    const ActionsAndProbs& prior,
    const std::vector<Action>& legal_actions,
    int k) {
  if (legal_actions.size() <= static_cast<size_t>(k)) {
    return legal_actions;
  }
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
  std::sort(scored.rbegin(), scored.rend());
  std::vector<Action> top_actions;
  top_actions.reserve(k);
  for (int i = 0; i < k; ++i) {
    top_actions.push_back(scored[i].second);
  }
  return top_actions;
}

}  // namespace dune_imperium
}  // namespace open_spiel

#endif  // OPEN_SPIEL_EXAMPLES_DUNE_COMPOUND_TURN_ROLLOUT_H_
