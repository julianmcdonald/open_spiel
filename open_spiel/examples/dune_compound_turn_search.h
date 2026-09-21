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
#include <future>
#include <set>

#include "open_spiel/examples/dune_batched_evaluator.h"
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
      if (turn_tree) {
        turn_tree->RecordRollout(tasks[i].action, traj.intra_turn_actions, traj.terminal_utility);
      }
    }
  }

  // Calculate candidate statistics
  double raw_mean_u = 0.0;
  if (turn_tree) {
    raw_mean_u = turn_tree->GetMeanUtility(raw_action);
  }

  Action best_action = raw_action;
  double best_mean_u = raw_mean_u;
  bool overridden = false;

  for (size_t c_idx = 1; c_idx < candidates.size(); ++c_idx) {
    Action cand_a = candidates[c_idx];
    double mean_u = (turn_tree != nullptr) ? turn_tree->GetMeanUtility(cand_a) : 0.0;
    if (mean_u > raw_mean_u + min_override_margin && mean_u > best_mean_u) {
      best_mean_u = mean_u;
      best_action = cand_a;
      overridden = true;
    }
  }

  return {best_action, overridden, raw_mean_u, best_mean_u, prior_rollouts_reused, new_rollouts_executed};
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

}  // namespace dune_imperium
}  // namespace open_spiel

#endif  // OPEN_SPIEL_EXAMPLES_DUNE_COMPOUND_TURN_SEARCH_H_
