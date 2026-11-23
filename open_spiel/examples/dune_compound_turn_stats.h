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

#ifndef OPEN_SPIEL_EXAMPLES_DUNE_COMPOUND_TURN_STATS_H_
#define OPEN_SPIEL_EXAMPLES_DUNE_COMPOUND_TURN_STATS_H_

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <limits>
#include <memory>
#include <random>
#include <utility>
#include <vector>

#include "open_spiel/examples/dune_compound_turn_tree.h"
#include "open_spiel/examples/dune_seed_utils.h"
#include "open_spiel/games/dune_imperium/dune_imperium_cards.h"
#include "open_spiel/games/dune_imperium/dune_imperium_common.h"
#include "open_spiel/games/dune_imperium/dune_imperium_util.h"

namespace open_spiel {
namespace dune_imperium {

struct EpochCardDrawCounter {
  std::array<std::pair<Action, int>, 32> inline_counts{};
  int inline_size = 0;
  std::vector<std::pair<Action, int>> overflow;

  void clear() {
    inline_size = 0;
    overflow.clear();
  }
  int get(Action a) const {
    for (int i = 0; i < inline_size; ++i) {
      if (inline_counts[i].first == a) return inline_counts[i].second;
    }
    for (const auto& kv : overflow) {
      if (kv.first == a) return kv.second;
    }
    return 0;
  }
  void increment(Action a) {
    for (int i = 0; i < inline_size; ++i) {
      if (inline_counts[i].first == a) {
        ++inline_counts[i].second;
        return;
      }
    }
    if (inline_size < 32) {
      inline_counts[inline_size++] = {a, 1};
      return;
    }
    for (auto& kv : overflow) {
      if (kv.first == a) {
        ++kv.second;
        return;
      }
    }
    overflow.push_back({a, 1});
  }
};

// Result of a rollout tracking both terminal utility and the actions taken within the active turn.
struct RolloutTrajectory {
  double terminal_utility = 0.0;
  std::vector<Action> intra_turn_actions;
  bool aborted_by_deadline = false;
  int steps = 0;
  int nn_evals = 0;
};

inline constexpr double kDefaultEbVarFloor = 0.04;

inline double ApplyTerminalWinSpeedTiebreak(
    double ret, int actual_rounds, double win_speed_tiebreak_eps) {
  if (win_speed_tiebreak_eps > 0.0 && ret >= 2.24) {
    return ret + win_speed_tiebreak_eps *
                     static_cast<double>(std::clamp(10 - actual_rounds, 0, 5));
  }
  return ret;
}

inline double ComputeEffectiveOverrideMargin(
    double min_advantage_margin,
    double q_raw,
    double win_speed_tiebreak_eps = 0.0,
    bool ceiling_aware_margin = true) {
  if (!ceiling_aware_margin || min_advantage_margin <= 0.0 ||
      min_advantage_margin >= 1.0) {
    return min_advantage_margin;
  }
  const double max_win_return =
      2.25 + (win_speed_tiebreak_eps > 0.0 ? 3.0 * win_speed_tiebreak_eps : 0.0);
  if (q_raw > std::max(1.50, max_win_return - 2.0 * min_advantage_margin)) {
    return std::min(
        min_advantage_margin,
        std::max(0.03, 0.45 * std::max(0.0, max_win_return - q_raw)));
  }
  return min_advantage_margin;
}

struct PairedEbDiffStats {
  double mean_d = 0.0;
  double sample_var_d = 0.0;
  double se_eb = 0.0;
  double t_stat = 0.0;
  int positive_diff_count = 0;
};

inline PairedEbDiffStats ComputePairedEbDiffStats(
    const double* cand_u,
    const double* raw_u,
    int n,
    double eb_var_floor = kDefaultEbVarFloor) {
  PairedEbDiffStats st;
  if (n <= 0 || cand_u == nullptr || raw_u == nullptr) return st;
  double sum_d = 0.0;
  int pos_cnt = 0;
  for (int i = 0; i < n; ++i) {
    const double d = cand_u[i] - raw_u[i];
    sum_d += d;
    if (d > 1e-9) ++pos_cnt;
  }
  st.mean_d = sum_d / static_cast<double>(n);
  st.positive_diff_count = pos_cnt;
  if (n >= 2) {
    double sum_sq = 0.0;
    for (int i = 0; i < n; ++i) {
      const double diff = (cand_u[i] - raw_u[i]) - st.mean_d;
      sum_sq += diff * diff;
    }
    st.sample_var_d = sum_sq / static_cast<double>(n - 1);
    st.se_eb = std::sqrt(
        (st.sample_var_d + eb_var_floor / static_cast<double>(n)) /
        static_cast<double>(n));
    st.t_stat = (st.se_eb > 0.0) ? (st.mean_d / st.se_eb) : 0.0;
  }
  return st;
}

inline bool PassesSingleFlipGuard(
    int n, double min_t_stat, int positive_diff_count) {
  if (n >= 8 && min_t_stat >= 1.0) {
    return positive_diff_count >= 2;
  }
  return true;
}

inline ActionsAndProbs ComputeKlRegularizedSearchPolicy(
    const ActionsAndProbs& raw_prior,
    const std::vector<Action>& legal_actions,
    Action raw_action,
    const std::vector<int>& visit_counts,
    const std::vector<double>& q_values,
    double tau,
    double shrink_n0 = 8.0) {
  if (tau <= 0.0 || legal_actions.empty()) {
    return raw_prior;
  }
  const size_t n_leg = legal_actions.size();
  std::vector<double> p_raw_vec(n_leg, 0.0);
  double p_sum = 0.0;
  for (size_t i = 0; i < n_leg; ++i) {
    for (const auto& [a, p] : raw_prior) {
      if (a == legal_actions[i]) {
        p_raw_vec[i] = std::max(0.0, p);
        break;
      }
    }
    p_sum += p_raw_vec[i];
  }
  if (p_sum > 1e-12) {
    for (double& p : p_raw_vec) p /= p_sum;
  } else {
    std::fill(p_raw_vec.begin(), p_raw_vec.end(), 1.0 / static_cast<double>(n_leg));
  }

  bool any_visits = false;
  double q_raw = 0.0;
  bool has_q_raw = false;
  double weighted_q_sum = 0.0;
  double weighted_p_sum = 0.0;
  for (size_t i = 0; i < n_leg; ++i) {
    const int v = (i < visit_counts.size()) ? visit_counts[i] : 0;
    const double q = (i < q_values.size()) ? q_values[i] : 0.0;
    if (v > 0) {
      any_visits = true;
      weighted_q_sum += p_raw_vec[i] * q;
      weighted_p_sum += p_raw_vec[i];
      if (legal_actions[i] == raw_action) {
        q_raw = q;
        has_q_raw = true;
      }
    }
  }
  if (!any_visits) {
    return raw_prior;
  }
  if (!has_q_raw && weighted_p_sum > 1e-12) {
    q_raw = weighted_q_sum / weighted_p_sum;
  }

  std::vector<double> shifts(n_leg, 0.0);
  double max_shift = -std::numeric_limits<double>::infinity();
  for (size_t i = 0; i < n_leg; ++i) {
    const int v = (i < visit_counts.size()) ? visit_counts[i] : 0;
    if (v > 0 && i < q_values.size()) {
      const double adv = q_values[i] - q_raw;
      const double shrunk_adv =
          (static_cast<double>(v) / (static_cast<double>(v) + std::max(0.0, shrink_n0))) * adv;
      shifts[i] = std::clamp(shrunk_adv / tau, -20.0, 20.0);
    }
    max_shift = std::max(max_shift, shifts[i]);
  }

  ActionsAndProbs out;
  out.reserve(n_leg);
  double w_sum = 0.0;
  for (size_t i = 0; i < n_leg; ++i) {
    const double p_floor = std::max(1e-8, p_raw_vec[i]);
    const double w = p_floor * std::exp(shifts[i] - max_shift);
    out.emplace_back(legal_actions[i], w);
    w_sum += w;
  }
  if (w_sum > 0.0) {
    for (auto& [a, w] : out) {
      w /= w_sum;
    }
  }
  return out;
}

inline bool IsConvertSpecimenAction(Action a) {
  return a >= kActionConvertSpecimenToTroop0 + 1 &&
         a <= kActionConvertSpecimenToTroop0 + 12;
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

struct EndgamePly2Config {
  bool enabled = false;
  bool stage2_only = true;
  int min_round = 6;
  int min_table_vp = 7;
  int inner_worlds = 4;
  double margin = 0.12;
  double max_top1_prior = 0.85;
  double min_top2_prior = 0.08;
};

struct IntrigueBeliefConfig {
  bool enabled = false;
  double held_age_combat_log_slope = 0.25;      // Per-round held log-weight boost for kCombat intrigues
  double held_age_endgame_log_slope = 0.30;     // Per-round held log-weight boost for kEndgame intrigues
  double close_pass_combat_log_penalty = -1.20; // Log-weight penalty on kCombat intrigues per close-combat pass while holding intrigues
  std::array<double, 4> mean_age{};
  std::array<int, 4> close_passes{};
};

struct OpponentIntrigueEvidence {
  std::array<double, 4> mean_age{};
  std::array<int, 4> close_passes{};
};

inline OpponentIntrigueEvidence ComputeOpponentIntrigueEvidenceFromHistory(
    const State& target_state, Player observer) {
  OpponentIntrigueEvidence evidence;
  std::unique_ptr<State> replay = target_state.GetGame()->NewInitialState();
  std::array<std::deque<int>, 4> acq_rounds_fifo;
  std::array<int, 4> prev_hand_size = {0, 0, 0, 0};
  std::array<int, 4> close_passes = {0, 0, 0, 0};

  for (Action act : target_state.History()) {
    const auto* d = dynamic_cast<const dune_imperium::DuneImperiumState*>(replay.get());
    if (d != nullptr && d->phase() == dune_imperium::GamePhase::kCombat &&
        act == dune_imperium::kActionCombatPass) {
      Player p = replay->CurrentPlayer();
      if (p >= 0 && p < 4 && p != observer) {
        if (d->GetIntrigueHandForTesting(p).size() >= 1 &&
            (d->TroopsInCombat(p) > 0 || d->DreadnoughtsInCombat(p) > 0)) {
          int p_str = dune_imperium::CombatStrength(*d, p);
          int max_opp_str = 0;
          for (int opp = 0; opp < 4; ++opp) {
            if (opp != p &&
                (d->TroopsInCombat(opp) > 0 || d->DreadnoughtsInCombat(opp) > 0)) {
              max_opp_str = std::max(max_opp_str, dune_imperium::CombatStrength(*d, opp));
            }
          }
          if (max_opp_str >= p_str && (max_opp_str - p_str) <= 4) {
            close_passes[p]++;
          }
        }
      }
    }

    replay->ApplyAction(act);
    const auto* post_d = dynamic_cast<const dune_imperium::DuneImperiumState*>(replay.get());
    const int cur_round = post_d ? post_d->GetCurrentRound() : 1;
    for (int p = 0; p < 4; ++p) {
      const int cur_size = post_d ? static_cast<int>(post_d->GetIntrigueHandForTesting(p).size()) : 0;
      const int delta = cur_size - prev_hand_size[p];
      if (delta > 0) {
        for (int k = 0; k < delta; ++k) {
          acq_rounds_fifo[p].push_back(cur_round);
        }
      } else if (delta < 0) {
        const int pop_count = std::min<int>(-delta, static_cast<int>(acq_rounds_fifo[p].size()));
        for (int k = 0; k < pop_count; ++k) {
          acq_rounds_fifo[p].pop_front();
        }
      }
      prev_hand_size[p] = cur_size;
    }
  }

  const auto* final_d = dynamic_cast<const dune_imperium::DuneImperiumState*>(&target_state);
  const int target_round = final_d ? final_d->GetCurrentRound() : 1;
  for (int p = 0; p < 4; ++p) {
    if (!acq_rounds_fifo[p].empty()) {
      double sum_age = 0.0;
      for (int acq_r : acq_rounds_fifo[p]) {
        sum_age += std::max(0, target_round - acq_r);
      }
      evidence.mean_age[p] = sum_age / static_cast<double>(acq_rounds_fifo[p].size());
    } else {
      evidence.mean_age[p] = 0.0;
    }
    evidence.close_passes[p] = close_passes[p];
  }
  return evidence;
}

inline void ApplyOpponentIntrigueBeliefReweighting(
    dune_imperium::DuneImperiumState& sampled_dune,
    Player search_player,
    const IntrigueBeliefConfig& belief_cfg,
    uint64_t seed) {
  std::vector<int> pool;
  std::array<int, 4> needed_hand_sizes{};
  for (int p = 0; p < 4; ++p) {
    if (p != search_player) {
      const auto& h = sampled_dune.GetIntrigueHandForTesting(p);
      needed_hand_sizes[p] = static_cast<int>(h.size());
      pool.insert(pool.end(), h.begin(), h.end());
    }
  }
  const auto& deck = sampled_dune.GetIntrigueDrawDeckForTesting();
  pool.insert(pool.end(), deck.begin(), deck.end());

  if (pool.empty()) return;

  uint64_t belief_seed = dune_seed::DeriveSeed(seed, 0x42454C49ULL);
  std::mt19937_64 rng(belief_seed);

  for (int p = 0; p < 4; ++p) {
    if (p == search_player) continue;
    const int k = needed_hand_sizes[p];
    if (k <= 0) {
      sampled_dune.SetPlayerIntrigueHandForTesting(p, {});
      continue;
    }
    std::vector<int> new_hand;
    new_hand.reserve(k);

    for (int draw = 0; draw < k && !pool.empty(); ++draw) {
      std::vector<double> weights;
      weights.reserve(pool.size());
      double total_w = 0.0;
      for (int cid : pool) {
        const auto* ic = dune_imperium::FindIntrigueCardById(cid);
        double log_w = 0.0;
        if (ic != nullptr) {
          if ((ic->phase_mask & dune_imperium::kIntriguePhaseCombatMask) != 0) {
            log_w += belief_cfg.mean_age[p] * belief_cfg.held_age_combat_log_slope +
                     static_cast<double>(belief_cfg.close_passes[p]) * belief_cfg.close_pass_combat_log_penalty;
          }
          if ((ic->phase_mask & dune_imperium::kIntriguePhaseEndgameMask) != 0) {
            log_w += belief_cfg.mean_age[p] * belief_cfg.held_age_endgame_log_slope;
          }
        }
        log_w = std::clamp(log_w, -3.0, 3.0);
        double w = std::exp(log_w);
        weights.push_back(w);
        total_w += w;
      }
      std::uniform_real_distribution<double> dist(0.0, total_w);
      double roll = dist(rng);
      double cum = 0.0;
      size_t chosen_idx = pool.size() - 1;
      for (size_t i = 0; i < pool.size(); ++i) {
        cum += weights[i];
        if (roll <= cum) {
          chosen_idx = i;
          break;
        }
      }
      new_hand.push_back(pool[chosen_idx]);
      pool.erase(pool.begin() + chosen_idx);
    }
    sampled_dune.SetPlayerIntrigueHandForTesting(p, new_hand);
  }
  sampled_dune.SetIntrigueDrawDeckForTesting(pool);
}

}  // namespace dune_imperium
}  // namespace open_spiel

#endif  // OPEN_SPIEL_EXAMPLES_DUNE_COMPOUND_TURN_STATS_H_
