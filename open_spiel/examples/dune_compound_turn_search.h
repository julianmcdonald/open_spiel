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

#ifndef OPEN_SPIEL_EXAMPLES_DUNE_COMPOUND_TURN_SEARCH_H_
#define OPEN_SPIEL_EXAMPLES_DUNE_COMPOUND_TURN_SEARCH_H_

#include <memory>
#include <random>
#include <unordered_map>
#include <vector>
#include <algorithm>
#include <cmath>
#include <future>
#include <set>

#include "open_spiel/examples/dune_batched_evaluator.h"
#include "open_spiel/examples/dune_online_search_collector.h"
#include "open_spiel/examples/dune_seed_utils.h"
#include "open_spiel/games/dune_imperium/dune_imperium.h"
#include "open_spiel/games/dune_imperium/dune_imperium_common.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace dune_imperium {

// A node in the intra-turn search tree tracking visits and accumulated utility.
struct TurnSearchNode {
  int visits = 0;
  double utility_sum = 0.0;
  std::unordered_map<Action, std::unique_ptr<TurnSearchNode>> children;

  double MeanUtility() const {
    return visits > 0 ? (utility_sum / static_cast<double>(visits)) : 0.0;
  }

  TurnSearchNode* GetOrCreateChild(Action a) {
    auto it = children.find(a);
    if (it == children.end()) {
      auto new_node = std::make_unique<TurnSearchNode>();
      TurnSearchNode* ptr = new_node.get();
      children[a] = std::move(new_node);
      return ptr;
    }
    return it->second.get();
  }

  void RecordOutcome(double u) {
    visits++;
    utility_sum += u;
  }
};

// Search tree preserved across sequential actions within a single player turn.
// Clears when the player turn ends (kActionEndTurn).
class TurnSearchTree {
 public:
  TurnSearchTree() { Reset(); }

  void Reset() {
    root_ = std::make_unique<TurnSearchNode>();
    current_node_ = root_.get();
  }

  // Advance the current root to child 'action' when that action is taken in the real game.
  void Advance(Action action) {
    if (!current_node_) {
      Reset();
      return;
    }
    current_node_ = current_node_->GetOrCreateChild(action);
  }

  // Record a simulated rollout path and its terminal payoff down the active subtree.
  void RecordRollout(Action first_action, const std::vector<Action>& subsequent_turn_actions, double terminal_u) {
    if (!current_node_) return;

    TurnSearchNode* node = current_node_->GetOrCreateChild(first_action);
    node->RecordOutcome(terminal_u);

    for (Action a : subsequent_turn_actions) {
      node = node->GetOrCreateChild(a);
      node->RecordOutcome(terminal_u);
    }
  }

  // Check if an action has preserved rollouts from earlier in the turn.
  bool HasPriorData(Action a) const {
    if (!current_node_) return false;
    auto it = current_node_->children.find(a);
    return (it != current_node_->children.end() && it->second->visits > 0);
  }

  int GetVisits(Action a) const {
    if (!current_node_) return 0;
    auto it = current_node_->children.find(a);
    return (it != current_node_->children.end()) ? it->second->visits : 0;
  }

  double GetUtilitySum(Action a) const {
    if (!current_node_) return 0.0;
    auto it = current_node_->children.find(a);
    return (it != current_node_->children.end()) ? it->second->utility_sum : 0.0;
  }

  double GetMeanUtility(Action a) const {
    if (!current_node_) return 0.0;
    auto it = current_node_->children.find(a);
    return (it != current_node_->children.end()) ? it->second->MeanUtility() : 0.0;
  }

  TurnSearchNode* current_node() const { return current_node_; }

 private:
  std::unique_ptr<TurnSearchNode> root_;
  TurnSearchNode* current_node_ = nullptr;
};

// Result of a rollout tracking both terminal utility and the actions taken within the active turn.
struct RolloutTrajectory {
  double terminal_utility = 0.0;
  std::vector<Action> intra_turn_actions;
};

// Simulate a rollout to terminal while recording any actions taken by search_player
// in the active agent turn before kActionEndTurn.
inline RolloutTrajectory SimulateRolloutWithTurnTrajectory(
    State* sim_state,
    Player search_player,
    BatchedNNEvaluator* evaluator,
    uint64_t rollout_seed) {
  std::mt19937 rng(rollout_seed);
  int steps = 0;
  const int max_steps = 2500;
  std::vector<Action> intra_turn_actions;
  bool active_turn = true;

  while (!sim_state->IsTerminal() && steps < max_steps) {
    steps++;
    if (sim_state->IsChanceNode()) {
      auto chance_outcomes = sim_state->ChanceOutcomes();
      double u = std::generate_canonical<double, 53>(rng);
      Action ca = open_spiel::SampleAction(chance_outcomes, u).first;
      sim_state->ApplyAction(ca);
      continue;
    }

    Player cur = sim_state->CurrentPlayer();
    std::vector<Action> legals = sim_state->LegalActions();
    if (legals.empty()) break;

    Action a = kInvalidAction;
    if (legals.size() == 1) {
      a = legals.front();
    } else {
      ActionsAndProbs prior = evaluator->Prior(*sim_state);
      // Pick greedy action
      double best_p = -1.0;
      a = legals.front();
      for (Action cand_a : legals) {
        for (const auto& ap : prior) {
          if (ap.first == cand_a && ap.second > best_p) {
            best_p = ap.second;
            a = cand_a;
            break;
          }
        }
      }
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

    sim_state->ApplyAction(a);
  }

  std::vector<double> ret = sim_state->Returns();
  double u = (static_cast<size_t>(search_player) < ret.size()) ? ret[search_player] : 0.0;
  return {u, std::move(intra_turn_actions)};
}

// Pick greedy action from prior over legal actions
inline Action PickGreedyAction(const ActionsAndProbs& prior, const std::vector<Action>& legal_actions) {
  if (legal_actions.empty()) return kInvalidAction;
  if (prior.empty()) return legal_actions.front();
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

// Tree reuse strategies
enum class TreeReuseMode {
  kTargetFill,  // Top up candidate visits to reach target (maximizes GPU efficiency)
  kAdditive     // Always run target rollouts and add to existing tree visits (deepens search)
};

// Decision result from CompoundRolloutSearch
struct CompoundSearchDecisionResult {
  Action action = kInvalidAction;
  bool overridden = false;
  double raw_mean_utility = 0.0;
  double chosen_mean_utility = 0.0;
  int prior_rollouts_reused = 0;
  int new_rollouts_executed = 0;
  std::vector<std::pair<Action, double>> candidate_utilities = {};
};

// Intra-turn compound rollout search decision with tree preservation
inline CompoundSearchDecisionResult CompoundRolloutSearchDecision(
    const State& state,
    Player search_player,
    TurnSearchTree* turn_tree,
    const std::vector<std::shared_ptr<BatchedNNEvaluator>>& evaluators,
    int top_k,
    int rollouts_per_action,
    double min_override_margin,
    TreeReuseMode reuse_mode,
    uint64_t decision_seed) {
  std::vector<Action> legal_actions = state.LegalActions();
  if (legal_actions.empty()) return {kInvalidAction, false, 0.0, 0.0, 0, 0};
  if (legal_actions.size() == 1) return {legal_actions.front(), false, 0.0, 0.0, 0, 0};

  ActionsAndProbs raw_prior = evaluators.front()->Prior(state);
  Action raw_action = PickGreedyAction(raw_prior, legal_actions);

  std::vector<Action> candidates = GetTopKActions(raw_prior, legal_actions, top_k);
  if (candidates.size() <= 1) {
    return {raw_action, false, 0.0, 0.0, 0, 0};
  }

  // Ensure candidate 0 is always the baseline raw_action
  if (candidates.front() != raw_action) {
    auto it = std::find(candidates.begin(), candidates.end(), raw_action);
    if (it != candidates.end()) {
      std::swap(candidates.front(), *it);
    } else {
      candidates.insert(candidates.begin(), raw_action);
      if (candidates.size() > static_cast<size_t>(top_k)) {
        candidates.pop_back();
      }
    }
  }

  const auto* dune = dynamic_cast<const dune_imperium::DuneImperiumState*>(&state);

  struct RolloutTask {
    size_t c_idx;
    Action action;
    std::unique_ptr<State> sim_state;
    uint64_t wseed;
  };

  std::vector<RolloutTask> tasks;
  int prior_rollouts_reused = 0;

  for (size_t c_idx = 0; c_idx < candidates.size(); ++c_idx) {
    Action cand_a = candidates[c_idx];
    int existing_visits = (turn_tree != nullptr) ? turn_tree->GetVisits(cand_a) : 0;
    int needed_rollouts = rollouts_per_action;

    if (turn_tree != nullptr && reuse_mode == TreeReuseMode::kTargetFill) {
      prior_rollouts_reused += std::min(existing_visits, rollouts_per_action);
      needed_rollouts = std::max(0, rollouts_per_action - existing_visits);
    } else if (turn_tree != nullptr) {
      prior_rollouts_reused += existing_visits;
    }

    for (int r = 0; r < needed_rollouts; ++r) {
      uint64_t wseed = dune_seed::DeriveSeed(decision_seed, c_idx, r);
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
      tasks.push_back({c_idx, cand_a, std::move(sim_state), wseed + 999});
    }
  }

  int new_rollouts_executed = static_cast<int>(tasks.size());

  TurnSearchTree local_tree;
  TurnSearchTree* active_tree = (turn_tree != nullptr) ? turn_tree : &local_tree;

  // Concurrently step all rollouts if any are needed
  if (!tasks.empty()) {
    std::vector<std::future<RolloutTrajectory>> futures;
    futures.reserve(tasks.size());

    for (size_t i = 0; i < tasks.size(); ++i) {
      auto* s_ptr = tasks[i].sim_state.get();
      uint64_t s_seed = tasks[i].wseed;
      auto* active_evaluator = evaluators[i % evaluators.size()].get();
      futures.push_back(std::async(
          std::launch::async,
          [s_ptr, search_player, active_evaluator, s_seed]() {
            return SimulateRolloutWithTurnTrajectory(s_ptr, search_player, active_evaluator, s_seed);
          }));
    }

    for (size_t i = 0; i < futures.size(); ++i) {
      RolloutTrajectory traj = futures[i].get();
      active_tree->RecordRollout(tasks[i].action, traj.intra_turn_actions, traj.terminal_utility);
    }
  }

  // Calculate candidate statistics
  double raw_mean_u = active_tree->GetMeanUtility(raw_action);

  Action best_action = raw_action;
  double best_mean_u = raw_mean_u;
  bool overridden = false;

  for (size_t c_idx = 1; c_idx < candidates.size(); ++c_idx) {
    Action cand_a = candidates[c_idx];
    double mean_u = active_tree->GetMeanUtility(cand_a);
    if (mean_u > raw_mean_u + min_override_margin && mean_u > best_mean_u) {
      best_mean_u = mean_u;
      best_action = cand_a;
      overridden = true;
    }
  }

  std::vector<std::pair<Action, double>> candidate_utilities;
  candidate_utilities.reserve(candidates.size());
  for (Action cand_a : candidates) {
    double mean_u = active_tree->GetMeanUtility(cand_a);
    candidate_utilities.push_back({cand_a, mean_u});
  }

  return {best_action, overridden, raw_mean_u, best_mean_u, prior_rollouts_reused, new_rollouts_executed, candidate_utilities};
}

// Compute softmax target distribution over evaluated actions at temperature T
inline std::vector<std::pair<Action, double>> ComputeSoftmaxTarget(
    const std::vector<std::pair<Action, double>>& action_utilities,
    const std::vector<Action>& legal_actions,
    double temperature) {
  std::vector<std::pair<Action, double>> probs;
  if (action_utilities.empty() || legal_actions.empty()) return probs;

  double max_u = -1e9;
  for (const auto& au : action_utilities) {
    if (au.second > max_u) max_u = au.second;
  }

  double sum_exp = 0.0;
  std::unordered_map<Action, double> exp_map;
  for (const auto& au : action_utilities) {
    double e = std::exp((au.second - max_u) / temperature);
    exp_map[au.first] = e;
    sum_exp += e;
  }

  probs.reserve(legal_actions.size());
  for (Action a : legal_actions) {
    auto it = exp_map.find(a);
    double p = (it != exp_map.end() && sum_exp > 0.0) ? (it->second / sum_exp) : 0.0;
    probs.push_back({a, p});
  }
  return probs;
}

// Blend blueprint prior mu and search target q_search: (1 - eta) * mu + eta * q_search
inline std::vector<std::pair<Action, double>> BlendTargets(
    const std::vector<std::pair<Action, double>>& blueprint_prior,
    const std::vector<std::pair<Action, double>>& search_target,
    double eta) {
  std::unordered_map<Action, double> q_map;
  for (const auto& st : search_target) q_map[st.first] = st.second;

  std::vector<std::pair<Action, double>> blended;
  blended.reserve(blueprint_prior.size());
  for (const auto& bp : blueprint_prior) {
    Action a = bp.first;
    double mu = bp.second;
    auto it = q_map.find(a);
    double q = (it != q_map.end()) ? it->second : 0.0;
    blended.push_back({a, (1.0 - eta) * mu + eta * q});
  }
  return blended;
}

// Check if state_after gained new private information (card drawn or intrigue drawn)
// compared to state_before for player.
inline bool HasNewInformation(
    const DuneImperiumState& state_before,
    const DuneImperiumState& state_after,
    Player player) {
  // 1. Intrigue card drawn
  const auto& int_before = state_before.GetIntrigueHandForTesting(player);
  const auto& int_after = state_after.GetIntrigueHandForTesting(player);
  if (int_after.size() > int_before.size()) return true;

  // 2. Card drawn from deck
  const auto& draw_before = state_before.GetPlayerDrawDeckForTesting(player);
  const auto& draw_after = state_after.GetPlayerDrawDeckForTesting(player);
  if (draw_after.size() != draw_before.size()) return true;

  return false;
}

// Check for high-stakes decision points: Agent Placement, Combat Deployment, Reveal Purchase
inline bool IsHighStakesStrategicState(const State& state, Player searched_player) {
  if (state.CurrentPlayer() != searched_player) return false;
  std::vector<Action> legal_actions = state.LegalActions();
  if (legal_actions.size() < 2) return false;

  const auto* dune_state = dynamic_cast<const dune_imperium::DuneImperiumState*>(&state);
  if (!dune_state) return false;

  for (Action a : legal_actions) {
    std::string s = dune_state->ActionToString(searched_player, a);
    if (s.rfind("PlaceAgent", 0) == 0 ||
        s.rfind("SelectAgentCard", 0) == 0 ||
        s.rfind("Deploy ", 0) == 0 ||
        s.rfind("CombatCommit", 0) == 0 ||
        s.rfind("Buy", 0) == 0) {
      return true;
    }
  }
  return false;
}

// Owned immutable snapshot of a pre-action decision visited by the learner
struct SearchSnapshot {
  std::unique_ptr<State> sim_state;
  Player player = kInvalidPlayer;
  std::vector<float> observation;
  std::vector<Action> legal_actions;
  std::vector<std::pair<Action, double>> raw_policy;
  Action chosen_action = kInvalidAction;
  float behavior_log_prob = 0.0f;
  uint64_t root_id = 0;
  // Candidate action descriptors for the deployed semantic action scorer.
  dune_semantic::CandidateActionData candidate_data;
};

// Audited Fallback-Aligned Target Rule:
// q = (1 - eta) * mu + eta * q_search if overridden or if D(q_search) == a_mu;
// q = mu if not overridden and D(q_search) != a_mu.
inline std::vector<double> ComputeFallbackAlignedTarget(
    const std::vector<std::pair<Action, double>>& raw_policy,
    const std::vector<Action>& legal_actions,
    const CompoundSearchDecisionResult& search_res,
    Action raw_action,
    double softmax_temperature,
    double eta_blend,
    bool* fell_back_to_mu) {
  auto q_search_probs = ComputeSoftmaxTarget(search_res.candidate_utilities, legal_actions, softmax_temperature);

  Action search_argmax = raw_action;
  double max_q = -1.0;
  for (const auto& qp : q_search_probs) {
    if (qp.second > max_q) {
      max_q = qp.second;
      search_argmax = qp.first;
    }
  }

  bool fallback = (!search_res.overridden) && (search_argmax != raw_action);
  if (fell_back_to_mu != nullptr) {
    *fell_back_to_mu = fallback;
  }

  std::vector<double> target_distribution(legal_actions.size(), 0.0);
  if (fallback) {
    for (size_t i = 0; i < legal_actions.size(); ++i) {
      Action a = legal_actions[i];
      for (const auto& ap : raw_policy) {
        if (ap.first == a) {
          target_distribution[i] = ap.second;
          break;
        }
      }
    }
  } else {
    auto blended = BlendTargets(raw_policy, q_search_probs, eta_blend);
    for (size_t i = 0; i < legal_actions.size(); ++i) {
      Action a = legal_actions[i];
      for (const auto& ap : blended) {
        if (ap.first == a) {
          target_distribution[i] = ap.second;
          break;
        }
      }
    }
  }
  return target_distribution;
}

// MPO-Style Policy-Relative Reweighting with Temperature Floor and KL Ceiling:
// Outside candidate set C: q(a | s) = pi_ref(a | s).
// Inside candidate set C: q(a | s) = P_C * (pi_ref(a | s) * exp((Q(s, a) - Q_max) / tau)) / sum_{c in C} (pi_ref(c | s) * exp((Q(s, c) - Q_max) / tau)),
// where P_C = sum_{c in C} pi_ref(c | s).
// Base temperature tau_base acts as a floor. If KL(q || pi_ref) <= max_kl, tau_base is kept.
// If KL(q || pi_ref) > max_kl, tau is increased via bisection until KL(q || pi_ref) <= max_kl.
inline std::vector<double> ComputeMpoRelativeTarget(
    const std::vector<std::pair<Action, double>>& raw_policy,
    const std::vector<Action>& legal_actions,
    const std::vector<std::pair<Action, double>>& candidate_utilities,
    double base_temperature = 0.50,
    double max_target_kl = 0.05,
    double* out_realized_kl = nullptr,
    double* out_realized_temperature = nullptr) {
  std::unordered_map<Action, double> pi_map;
  for (const auto& ap : raw_policy) {
    pi_map[ap.first] = std::max(0.0, ap.second);
  }

  std::vector<double> fallback_dist(legal_actions.size(), 0.0);
  for (size_t i = 0; i < legal_actions.size(); ++i) {
    auto it = pi_map.find(legal_actions[i]);
    fallback_dist[i] = (it != pi_map.end()) ? it->second : 0.0;
  }

  if (candidate_utilities.size() <= 1 || legal_actions.empty()) {
    if (out_realized_kl) *out_realized_kl = 0.0;
    if (out_realized_temperature) *out_realized_temperature = base_temperature;
    return fallback_dist;
  }

  std::vector<std::pair<Action, double>> valid_cands;
  double p_c = 0.0;
  double q_max = -1e9;
  for (const auto& cu : candidate_utilities) {
    auto it = pi_map.find(cu.first);
    if (it != pi_map.end() && it->second > 0.0) {
      valid_cands.push_back(cu);
      p_c += it->second;
      if (cu.second > q_max) q_max = cu.second;
    }
  }

  if (valid_cands.size() <= 1 || p_c <= 1e-12) {
    if (out_realized_kl) *out_realized_kl = 0.0;
    if (out_realized_temperature) *out_realized_temperature = base_temperature;
    return fallback_dist;
  }

  auto eval_cand_dist_and_kl = [&](double tau, std::vector<double>& cand_q) -> double {
    double sum_w = 0.0;
    std::vector<double> weights(valid_cands.size(), 0.0);
    for (size_t i = 0; i < valid_cands.size(); ++i) {
      Action a = valid_cands[i].first;
      double pi_a = pi_map[a];
      double u = valid_cands[i].second;
      double w = pi_a * std::exp((u - q_max) / tau);
      weights[i] = w;
      sum_w += w;
    }
    if (sum_w <= 1e-12) {
      for (size_t i = 0; i < valid_cands.size(); ++i) {
        cand_q[i] = pi_map[valid_cands[i].first];
      }
      return 0.0;
    }
    double kl = 0.0;
    for (size_t i = 0; i < valid_cands.size(); ++i) {
      Action a = valid_cands[i].first;
      double pi_a = pi_map[a];
      double q_a = p_c * (weights[i] / sum_w);
      cand_q[i] = q_a;
      if (q_a > 1e-15 && pi_a > 1e-15) {
        kl += q_a * std::log(q_a / pi_a);
      }
    }
    return kl;
  };

  double tau = std::max(1e-4, base_temperature);
  std::vector<double> cand_q(valid_cands.size(), 0.0);
  double current_kl = eval_cand_dist_and_kl(tau, cand_q);

  if (current_kl > max_target_kl) {
    double tau_low = tau;
    double tau_high = 1000.0;
    for (int iter = 0; iter < 25; ++iter) {
      double tau_mid = 0.5 * (tau_low + tau_high);
      double kl_mid = eval_cand_dist_and_kl(tau_mid, cand_q);
      if (kl_mid > max_target_kl) {
        tau_low = tau_mid;
      } else {
        tau_high = tau_mid;
      }
    }
    tau = tau_high;
    current_kl = eval_cand_dist_and_kl(tau, cand_q);
  }

  if (out_realized_kl) *out_realized_kl = current_kl;
  if (out_realized_temperature) *out_realized_temperature = tau;

  std::unordered_map<Action, double> cand_q_map;
  for (size_t i = 0; i < valid_cands.size(); ++i) {
    cand_q_map[valid_cands[i].first] = cand_q[i];
  }

  std::vector<double> target_distribution(legal_actions.size(), 0.0);
  for (size_t i = 0; i < legal_actions.size(); ++i) {
    Action a = legal_actions[i];
    auto it = cand_q_map.find(a);
    if (it != cand_q_map.end()) {
      target_distribution[i] = it->second;
    } else {
      auto pit = pi_map.find(a);
      target_distribution[i] = (pit != pi_map.end()) ? pit->second : 0.0;
    }
  }

  return target_distribution;
}

inline std::vector<float> ExtractObservationTensor(
    const State& state,
    Player player,
    dune_imperium::MarketAppendixMode mode = dune_imperium::MarketAppendixMode::kFullPublicInformationV3) {
  const auto* dune = dynamic_cast<const dune_imperium::DuneImperiumState*>(&state);
  if (dune && mode != dune_imperium::MarketAppendixMode::kNone) {
    return dune->InformationStateTensorWithAppendix(player, mode);
  }
  return state.InformationStateTensor(player);
}

struct SearchSupervisionResult {
  std::vector<open_spiel::SearchTrainingExample> examples;
  int roots_evaluated = 0;
  int overrides = 0;
  int fallbacks_to_mu = 0;
  double elapsed_seconds = 0.0;
  double mean_override_rate = 0.0;
  int total_decisions_evaluated = 0;
};

inline SearchSupervisionResult BatchCompoundSearchSupervision(
    std::vector<SearchSnapshot>& snapshots,
    const std::vector<std::shared_ptr<BatchedNNEvaluator>>& evaluators,
    int top_k = 3,
    int rollouts_per_action = 64,
    double min_override_margin = 0.15,
    double softmax_temperature = 0.50,
    double eta_blend = 0.80,
    uint64_t search_seed = 0,
    int num_workers = 1,
    bool full_turn_supervision = true,
    bool use_mpo_target = true,
    double max_target_kl = 0.05) {
  auto start_time = std::chrono::steady_clock::now();
  SearchSupervisionResult result;
  if (snapshots.empty() || evaluators.empty()) return result;

  struct TurnResult {
    bool valid = false;
    int overrides = 0;
    int fallbacks = 0;
    int rollouts = 0;
    int decisions_evaluated = 0;
    std::vector<open_spiel::SearchTrainingExample> examples;
  };
  std::vector<TurnResult> turn_results(snapshots.size());

  auto process_root = [&](size_t r_idx) {
    auto& snap = snapshots[r_idx];
    if (!snap.sim_state) return;

    uint64_t root_seed = dune_seed::DeriveSeed(search_seed, r_idx);
    TurnResult& tr = turn_results[r_idx];
    tr.valid = true;

    if (!full_turn_supervision) {
      // Single-decision supervision at the root state
      auto current_legals = snap.sim_state->LegalActions();
      SPIEL_CHECK_EQ(current_legals.size(), snap.legal_actions.size());
      for (size_t i = 0; i < current_legals.size(); ++i) {
        SPIEL_CHECK_EQ(current_legals[i], snap.legal_actions[i]);
      }

      CompoundSearchDecisionResult search_res = CompoundRolloutSearchDecision(
          *snap.sim_state,
          snap.player,
          /*turn_tree=*/nullptr,
          evaluators,
          top_k,
          rollouts_per_action,
          min_override_margin,
          TreeReuseMode::kTargetFill,
          root_seed);

      std::vector<std::pair<Action, double>> mu;
      if (!use_mpo_target && !snap.raw_policy.empty()) {
        mu = snap.raw_policy;
      } else if (!evaluators.empty()) {
        ActionsAndProbs raw_prior = evaluators.front()->Prior(*snap.sim_state);
        mu.reserve(snap.legal_actions.size());
        for (Action a : snap.legal_actions) {
          double p = 0.0;
          for (const auto& ap : raw_prior) {
            if (ap.first == a) { p = ap.second; break; }
          }
          mu.push_back({a, p});
        }
      } else {
        mu = snap.raw_policy;
      }
      Action raw_action = PickGreedyAction(mu, snap.legal_actions);

      bool fell_back = false;
      std::vector<double> target_dist;
      if (use_mpo_target) {
        double r_kl = 0.0;
        double r_tau = softmax_temperature;
        target_dist = ComputeMpoRelativeTarget(
            mu, snap.legal_actions, search_res.candidate_utilities,
            softmax_temperature, max_target_kl, &r_kl, &r_tau);
      } else {
        target_dist = ComputeFallbackAlignedTarget(
            mu, snap.legal_actions, search_res, raw_action,
            softmax_temperature, eta_blend, &fell_back);
      }

      open_spiel::SearchTrainingExample ex;
      ex.observation = snap.observation;
      ex.player = snap.player;
      ex.legal_actions = snap.legal_actions;
      ex.normalized_visits = std::move(target_dist);
      ex.value_target = 0.0;
      ex.value_target_attached = false;
      ex.simulations_completed = search_res.new_rollouts_executed;
      ex.candidate_data = snap.candidate_data;

      tr.decisions_evaluated = 1;
      tr.rollouts = search_res.new_rollouts_executed;
      if (search_res.overridden) tr.overrides = 1;
      if (fell_back) tr.fallbacks = 1;
      tr.examples.push_back(std::move(ex));
      return;
    }

    // Full-turn supervision: search takes over from turn start to kActionEndTurn
    TurnSearchTree turn_tree;
    int intra_step = 0;
    const int max_turn_steps = 20;

    // Use a persistent per-root RNG across consecutive chance outcomes
    uint64_t chance_root_seed = dune_seed::DeriveSeed(root_seed, /*stream=*/0x4348414EULL);
    std::mt19937_64 chance_rng(chance_root_seed);

    while (!snap.sim_state->IsTerminal() &&
           (snap.sim_state->CurrentPlayer() == snap.player || snap.sim_state->IsChanceNode()) &&
           intra_step < max_turn_steps) {
      if (snap.sim_state->IsChanceNode()) {
        auto chance_outcomes = snap.sim_state->ChanceOutcomes();
        double u = std::generate_canonical<double, 53>(chance_rng);
        Action ca = open_spiel::SampleAction(chance_outcomes, u).first;
        snap.sim_state->ApplyAction(ca);
        // Random chance event (card draw/market refill) invalidates cached search tree
        turn_tree.Reset();
        continue;
      }

      std::vector<Action> current_legals = snap.sim_state->LegalActions();
      if (current_legals.empty()) break;

      if (current_legals.size() == 1) {
        Action forced_a = current_legals.front();
        turn_tree.Advance(forced_a);
        const auto* dune_before = dynamic_cast<const dune_imperium::DuneImperiumState*>(snap.sim_state.get());
        std::unique_ptr<dune_imperium::DuneImperiumState> dune_snap_before =
            dune_before ? std::unique_ptr<dune_imperium::DuneImperiumState>(
                              dynamic_cast<dune_imperium::DuneImperiumState*>(dune_before->Clone().release()))
                        : nullptr;
        snap.sim_state->ApplyAction(forced_a);
        if (dune_snap_before) {
          const auto* dune_after = dynamic_cast<const dune_imperium::DuneImperiumState*>(snap.sim_state.get());
          if (dune_after && HasNewInformation(*dune_snap_before, *dune_after, snap.player)) {
            turn_tree.Reset();
          }
        }
        if (forced_a == dune_imperium::kActionEndTurn) {
          break;
        }
        continue;
      }

      uint64_t step_seed = dune_seed::DeriveSeed(root_seed, intra_step);
      intra_step++;

      CompoundSearchDecisionResult search_res = CompoundRolloutSearchDecision(
          *snap.sim_state,
          snap.player,
          &turn_tree,
          evaluators,
          top_k,
          rollouts_per_action,
          min_override_margin,
          TreeReuseMode::kTargetFill,
          step_seed);

      // Raw policy prior mu for fallback target / reference policy
      std::vector<std::pair<Action, double>> mu;
      if (!use_mpo_target && intra_step == 1 && !snap.raw_policy.empty() && current_legals.size() == snap.legal_actions.size()) {
        mu = snap.raw_policy;
      } else if (!evaluators.empty()) {
        ActionsAndProbs raw_prior = evaluators.front()->Prior(*snap.sim_state);
        mu.reserve(current_legals.size());
        for (Action a : current_legals) {
          double p = 0.0;
          for (const auto& ap : raw_prior) {
            if (ap.first == a) { p = ap.second; break; }
          }
          mu.push_back({a, p});
        }
      } else {
        mu = snap.raw_policy;
      }
      Action raw_action = PickGreedyAction(mu, current_legals);

      bool fell_back = false;
      std::vector<double> target_dist;
      if (use_mpo_target) {
        double r_kl = 0.0;
        double r_tau = softmax_temperature;
        target_dist = ComputeMpoRelativeTarget(
            mu, current_legals, search_res.candidate_utilities,
            softmax_temperature, max_target_kl, &r_kl, &r_tau);
      } else {
        target_dist = ComputeFallbackAlignedTarget(
            mu, current_legals, search_res, raw_action,
            softmax_temperature, eta_blend, &fell_back);
      }

      open_spiel::SearchTrainingExample ex;
      if (intra_step == 1 && !snap.observation.empty()) {
        ex.observation = snap.observation;
        ex.candidate_data = snap.candidate_data;
      } else {
        ex.observation = ExtractObservationTensor(*snap.sim_state, snap.player);
        const auto* dune_s = dynamic_cast<const dune_imperium::DuneImperiumState*>(snap.sim_state.get());
        if (dune_s) {
          const std::string cand_schema = (!evaluators.empty() && evaluators.front())
                                              ? evaluators.front()->SemanticDescriptorSchema()
                                              : dune_semantic::kDescriptorSchemaVersion;
          dune_semantic::ExtractCandidateDescriptors(*dune_s, current_legals, &ex.candidate_data, cand_schema);
        }
      }
      ex.player = snap.player;
      ex.legal_actions = current_legals;
      ex.normalized_visits = std::move(target_dist);
      ex.value_target = 0.0;
      ex.value_target_attached = false;
      ex.simulations_completed = search_res.new_rollouts_executed;

      tr.decisions_evaluated++;
      tr.rollouts += search_res.new_rollouts_executed;
      if (search_res.overridden) tr.overrides++;
      if (fell_back) tr.fallbacks++;
      tr.examples.push_back(std::move(ex));

      Action chosen_action = search_res.action;
      turn_tree.Advance(chosen_action);
      const auto* dune_before = dynamic_cast<const dune_imperium::DuneImperiumState*>(snap.sim_state.get());
      std::unique_ptr<dune_imperium::DuneImperiumState> dune_snap_before =
          dune_before ? std::unique_ptr<dune_imperium::DuneImperiumState>(
                            dynamic_cast<dune_imperium::DuneImperiumState*>(dune_before->Clone().release()))
                      : nullptr;
      snap.sim_state->ApplyAction(chosen_action);
      if (dune_snap_before) {
        const auto* dune_after = dynamic_cast<const dune_imperium::DuneImperiumState*>(snap.sim_state.get());
        if (dune_after && HasNewInformation(*dune_snap_before, *dune_after, snap.player)) {
          turn_tree.Reset();
        }
      }

      if (chosen_action == dune_imperium::kActionEndTurn) {
        break;
      }
    }
  };

  int actual_workers = std::min(num_workers, static_cast<int>(snapshots.size()));
  if (actual_workers <= 1) {
    for (size_t r_idx = 0; r_idx < snapshots.size(); ++r_idx) {
      process_root(r_idx);
    }
  } else {
    std::atomic<size_t> next_root{0};
    std::vector<std::thread> workers;
    workers.reserve(actual_workers);
    for (int w = 0; w < actual_workers; ++w) {
      workers.emplace_back([&]() {
        while (true) {
          size_t r_idx = next_root.fetch_add(1);
          if (r_idx >= snapshots.size()) break;
          process_root(r_idx);
        }
      });
    }
    for (auto& worker : workers) {
      worker.join();
    }
  }

  for (size_t r_idx = 0; r_idx < snapshots.size(); ++r_idx) {
    if (!turn_results[r_idx].valid) continue;
    result.roots_evaluated++;
    result.overrides += turn_results[r_idx].overrides;
    result.fallbacks_to_mu += turn_results[r_idx].fallbacks;
    result.total_decisions_evaluated += turn_results[r_idx].decisions_evaluated;
    for (auto& ex : turn_results[r_idx].examples) {
      result.examples.push_back(std::move(ex));
    }
  }

  auto end_time = std::chrono::steady_clock::now();
  result.elapsed_seconds = std::chrono::duration<double>(end_time - start_time).count();
  if (result.total_decisions_evaluated > 0) {
    result.mean_override_rate = static_cast<double>(result.overrides) / result.total_decisions_evaluated;
  }
  return result;
}

}  // namespace dune_imperium
}  // namespace open_spiel

#endif  // OPEN_SPIEL_EXAMPLES_DUNE_COMPOUND_TURN_SEARCH_H_
