#pragma once

#include <vector>
#include <memory>
#include <algorithm>
#include <limits>
#include <cmath>

#include "open_spiel/spiel.h"
#include "open_spiel/games/dune_imperium/dune_imperium.h"
#include "open_spiel/games/dune_imperium/dune_imperium_common.h"
#include "open_spiel/games/dune_imperium/dune_imperium_board.h"
#include "dune_network.h"
#include <random>

namespace open_spiel {

inline void DrainForcedNodesForPlanner(State* state) {
  int safety = 0;
  while (!state->IsTerminal()) {
    ++safety;
    if (safety > 150) break;
    if (state->IsChanceNode()) {
      auto outcomes = state->ChanceOutcomes();
      if (outcomes.empty()) break;
      state->ApplyAction(outcomes.front().first);
      continue;
    }
    auto legal = state->LegalActions();
    if (legal.empty()) break;
    if (legal.size() == 1) {
      state->ApplyAction(legal[0]);
      continue;
    }
    if (legal.size() > 1) {
      bool has_ack = false;
      for (Action a : legal) {
        if (a == dune_imperium::kActionAcknowledgeChance) {
          state->ApplyAction(a);
          has_ack = true;
          break;
        }
      }
      if (has_ack) continue;
    }
    break;
  }
}

// Overload for std::vector<float> logits (used by dune_ppo_train.cc)
inline Action FindActionOrCardPathToSpace(
    const State& state,
    Player owner,
    Action target_space,
    const std::vector<float>& logits) {
  std::vector<Action> legal_actions = state.LegalActions();
  if (std::find(legal_actions.begin(), legal_actions.end(), target_space) != legal_actions.end()) {
    return target_space;
  }

  std::vector<Action> candidate_cards;
  for (Action a : legal_actions) {
    if (a >= dune_imperium::kActionSelectAgentCard0 &&
        a < dune_imperium::kActionSelectAgentCard0 + 256) {
      candidate_cards.push_back(a);
    }
  }
  if (candidate_cards.empty()) {
    return kInvalidAction;
  }

  std::vector<Action> compatible_cards;
  for (Action card_action : candidate_cards) {
    std::unique_ptr<State> clone = state.Clone();
    clone->ApplyAction(card_action);
    DrainForcedNodesForPlanner(clone.get());
    std::vector<Action> clone_legal = clone->LegalActions();
    if (std::find(clone_legal.begin(), clone_legal.end(), target_space) != clone_legal.end()) {
      compatible_cards.push_back(card_action);
    }
  }

  if (compatible_cards.empty()) {
    return kInvalidAction;
  }

  Action best_card = compatible_cards[0];
  float best_logit = -std::numeric_limits<float>::infinity();
  for (Action card_action : compatible_cards) {
    if (static_cast<size_t>(card_action) < logits.size()) {
      float l = logits[card_action];
      if (l > best_logit) {
        best_logit = l;
        best_card = card_action;
      }
    }
  }
  return best_card;
}

// Overload for ActionsAndProbs (used by dune_search_teacher.cc)
inline Action FindActionOrCardPathToSpace(
    const State& state,
    Player owner,
    Action target_space,
    const ActionsAndProbs& prior) {
  std::vector<Action> legal_actions = state.LegalActions();
  if (std::find(legal_actions.begin(), legal_actions.end(), target_space) != legal_actions.end()) {
    return target_space;
  }

  std::vector<Action> candidate_cards;
  for (Action a : legal_actions) {
    if (a >= dune_imperium::kActionSelectAgentCard0 &&
        a < dune_imperium::kActionSelectAgentCard0 + 256) {
      candidate_cards.push_back(a);
    }
  }
  if (candidate_cards.empty()) {
    return kInvalidAction;
  }

  std::vector<Action> compatible_cards;
  for (Action card_action : candidate_cards) {
    std::unique_ptr<State> clone = state.Clone();
    clone->ApplyAction(card_action);
    DrainForcedNodesForPlanner(clone.get());
    std::vector<Action> clone_legal = clone->LegalActions();
    if (std::find(clone_legal.begin(), clone_legal.end(), target_space) != clone_legal.end()) {
      compatible_cards.push_back(card_action);
    }
  }

  if (compatible_cards.empty()) {
    return kInvalidAction;
  }

  Action best_card = compatible_cards[0];
  double best_prob = -1.0;
  for (Action card_action : compatible_cards) {
    double p = 0.0;
    for (const auto& ap : prior) {
      if (ap.first == card_action) {
        p = ap.second;
        break;
      }
    }
    if (p > best_prob) {
      best_prob = p;
      best_card = card_action;
    }
  }
  return best_card;
}

struct SwordmasterPlannerConfig {
  int deadline_round = 4;
  int max_depth = 64;
  int rollouts_per_action = 2;
  double success_score = 10000.0;
  double round_bonus = 500.0;
  double solari_deficit_weight = 80.0;
  bool use_policy_for_opponents = false;
  bool use_policy_for_owner = false;
  bool block_aware_opponents = false;
};

// Stochastic draining of chance/simultaneous nodes for rollout
inline void DrainForcedNodesForRollout(State* state, std::mt19937* rng) {
  int safety = 0;
  while (!state->IsTerminal()) {
    ++safety;
    if (safety > 150) break;
    if (state->IsChanceNode()) {
      auto outcomes = state->ChanceOutcomes();
      if (outcomes.empty()) break;
      double r = std::generate_canonical<double, 10>(*rng);
      Action action = outcomes.front().first;
      double sum = 0.0;
      for (const auto& outcome : outcomes) {
        sum += outcome.second;
        if (r <= sum) {
          action = outcome.first;
          break;
        }
      }
      state->ApplyAction(action);
      continue;
    }
    if (state->CurrentPlayer() == kSimultaneousPlayerId) {
      std::vector<Action> joint_action;
      for (int p = 0; p < state->NumPlayers(); ++p) {
        std::vector<Action> actions = state->LegalActions(p);
        if (actions.empty()) {
          joint_action.push_back(0);
        } else {
          std::uniform_int_distribution<int> dis(0, actions.size() - 1);
          joint_action.push_back(actions[dis(*rng)]);
        }
      }
      state->ApplyActions(joint_action);
      continue;
    }
    auto legal = state->LegalActions();
    if (legal.empty()) break;
    if (legal.size() == 1) {
      state->ApplyAction(legal[0]);
      continue;
    }
    if (legal.size() > 1) {
      bool has_ack = false;
      for (Action a : legal) {
        if (a == dune_imperium::kActionAcknowledgeChance) {
          state->ApplyAction(a);
          has_ack = true;
          break;
        }
      }
      if (has_ack) continue;
    }
    break;
  }
}

// No-prior overload of FindActionOrCardPathToSpace selecting compatible cards stochastically
inline Action FindActionOrCardPathToSpace(
    const State& state,
    Player owner,
    Action target_space,
    std::mt19937* rng) {
  std::vector<Action> legal_actions = state.LegalActions();
  if (std::find(legal_actions.begin(), legal_actions.end(), target_space) != legal_actions.end()) {
    return target_space;
  }

  std::vector<Action> candidate_cards;
  for (Action a : legal_actions) {
    if (a >= dune_imperium::kActionSelectAgentCard0 &&
        a < dune_imperium::kActionSelectAgentCard0 + 256) {
      candidate_cards.push_back(a);
    }
  }
  if (candidate_cards.empty()) {
    return kInvalidAction;
  }

  std::vector<Action> compatible_cards;
  for (Action card_action : candidate_cards) {
    std::unique_ptr<State> clone = state.Clone();
    clone->ApplyAction(card_action);
    DrainForcedNodesForPlanner(clone.get());
    std::vector<Action> clone_legal = clone->LegalActions();
    if (std::find(clone_legal.begin(), clone_legal.end(), target_space) != clone_legal.end()) {
      compatible_cards.push_back(card_action);
    }
  }

  if (compatible_cards.empty()) {
    return kInvalidAction;
  }

  std::uniform_int_distribution<int> dis(0, compatible_cards.size() - 1);
  return compatible_cards[dis(*rng)];
}

// Heuristic goal-biased action selection for owner when policy is bypassed
inline Action ChooseHeuristicAcquisitionAction(
    const State& state,
    const std::vector<Action>& actions,
    Player owner,
    std::mt19937* rng) {
  
  if (std::find(actions.begin(), actions.end(), dune_imperium::kActionAgentSpaceSwordmaster) != actions.end()) {
    return dune_imperium::kActionAgentSpaceSwordmaster;
  }
  
  Action sm_card = FindActionOrCardPathToSpace(state, owner, dune_imperium::kActionAgentSpaceSwordmaster, rng);
  if (sm_card != kInvalidAction && std::find(actions.begin(), actions.end(), sm_card) != actions.end()) {
    return sm_card;
  }

  if (std::find(actions.begin(), actions.end(), dune_imperium::kActionAgentSpaceSmuggling) != actions.end()) {
    return dune_imperium::kActionAgentSpaceSmuggling;
  }

  Action smuggling_card = FindActionOrCardPathToSpace(state, owner, dune_imperium::kActionAgentSpaceSmuggling, rng);
  if (smuggling_card != kInvalidAction && std::find(actions.begin(), actions.end(), smuggling_card) != actions.end()) {
    return smuggling_card;
  }

  if (std::find(actions.begin(), actions.end(), dune_imperium::kActionAgentRewardShipping) != actions.end()) {
    return dune_imperium::kActionAgentRewardShipping;
  }

  if (std::find(actions.begin(), actions.end(), dune_imperium::kActionShippingAdvance) != actions.end()) {
    return dune_imperium::kActionShippingAdvance;
  }

  if (std::find(actions.begin(), actions.end(), dune_imperium::kActionShippingRecall) != actions.end()) {
    return dune_imperium::kActionShippingRecall;
  }

  if (std::find(actions.begin(), actions.end(), dune_imperium::kActionShippingLevel1Dividends) != actions.end()) {
    return dune_imperium::kActionShippingLevel1Dividends;
  }

  std::uniform_int_distribution<int> dis(0, actions.size() - 1);
  return actions[dis(*rng)];
}

// Compute the raw heuristic rollout score at the deadline
inline double ScoreSwordmasterRaceState(
    const State& state,
    Player owner,
    const SwordmasterPlannerConfig& cfg) {
  
  const auto* dune_state =
      dynamic_cast<const dune_imperium::DuneImperiumState*>(&state);
  if (dune_state == nullptr) return 0.0;

  // Penalize another player owning Swordmaster mildly
  for (int p = 0; p < state.NumPlayers(); ++p) {
    if (p != owner && dune_state->HasSwordmaster(p)) {
      return -150.0;
    }
  }

  double score = 0.0;

  // 1. Solari Progress
  int solari = dune_state->GetPlayerSolari(owner);
  bool is_leto = (dune_state->PlayerLeader(owner) == 4);
  int target_solari = is_leto ? 7 : 8;
  int deficit = std::max(0, target_solari - solari);
  score -= deficit * cfg.solari_deficit_weight;

  // 2. Shipping track level bonus
  int shipping = dune_state->PlayerShippingLevelForTesting(owner);
  score += shipping * 150.0;

  // 3. Occupancy bitmask checks
  uint8_t owner_bit = dune_imperium::PlayerSpaceBit(owner);

  const auto* sm_space = dune_state->AgentSpaceForActionForTesting(610);
  if (sm_space != nullptr) {
    uint8_t occupancy = dune_state->GetAgentSpaceOwnerForTesting(sm_space->board_index);
    if (occupancy != 0 && (occupancy & ~owner_bit) != 0) {
      score -= 100.0;
    }
  }

  const auto* smuggling_space = dune_state->AgentSpaceForActionForTesting(614);
  if (smuggling_space != nullptr) {
    uint8_t occupancy = dune_state->GetAgentSpaceOwnerForTesting(smuggling_space->board_index);
    if (occupancy != 0 && (occupancy & ~owner_bit) != 0) {
      score -= 50.0;
    }
  }

  return score;
}

// Replicate SamplePolicyAction in helpers to keep it independent
inline Action SampleActionFromLogits(
    std::mt19937* rng, const std::vector<float>& logits,
    const std::vector<Action>& legal_actions) {
  float max_logit = -std::numeric_limits<float>::infinity();
  for (Action legal_action : legal_actions) {
    if (legal_action >= 0 &&
        static_cast<size_t>(legal_action) < logits.size()) {
      max_logit = std::max(max_logit, logits[legal_action]);
    }
  }

  std::vector<double> weights;
  weights.reserve(legal_actions.size());
  for (Action legal_action : legal_actions) {
    double weight = 1.0;
    if (legal_action >= 0 &&
        static_cast<size_t>(legal_action) < logits.size() &&
        std::isfinite(max_logit)) {
      weight = std::exp(static_cast<double>(logits[legal_action] - max_logit));
    }
    if (!std::isfinite(weight) || weight <= 0.0) weight = 1.0;
    weights.push_back(weight);
  }

  std::discrete_distribution<size_t> dist(weights.begin(), weights.end());
  return legal_actions[dist(*rng)];
}

// Execute a single simulated rollout to score an action path
inline double RolloutSwordmasterRace(
    const State& start_state,
    Player owner,
    const SwordmasterPlannerConfig& cfg,
    std::mt19937* rng,
    BatchedEvaluator* evaluator,
    uint64_t* eval_requests_out = nullptr) {

  std::unique_ptr<State> state = start_state.Clone();
  const auto* dune_state =
      dynamic_cast<const dune_imperium::DuneImperiumState*>(state.get());

  int depth = 0;

  while (!state->IsTerminal()) {
    DrainForcedNodesForRollout(state.get(), rng);

    if (state->IsTerminal()) {
      break;
    }

    if (dune_state != nullptr) {
      if (dune_state->HasSwordmaster(owner)) {
        int purchase_round = dune_state->GetCurrentRound();
        double score = cfg.success_score + (cfg.deadline_round - purchase_round) * cfg.round_bonus;
        return score;
      }
      if (dune_state->GetCurrentRound() > cfg.deadline_round) {
        break; // Passed deadline
      }
    }

    if (depth >= cfg.max_depth) {
      break;
    }
    depth++;

    Player cur_player = state->CurrentPlayer();
    std::vector<Action> actions = state->LegalActions();
    if (actions.empty()) break;

    Action action = kInvalidAction;

    if (cur_player == owner) {
      if (cfg.use_policy_for_owner && evaluator != nullptr) {
        action = FindActionOrCardPathToSpace(*state, owner, dune_imperium::kActionAgentSpaceSwordmaster, rng);
        if (action == kInvalidAction) {
          auto game = state->GetGame();
          int64_t obs_size = game->GetType().provides_information_state_tensor
                                 ? game->InformationStateTensorSize()
                                 : game->ObservationTensorSize();
          std::vector<float> obs(obs_size, 0.0f);
          state->ObservationTensor(cur_player, absl::MakeSpan(obs));
          if (eval_requests_out) (*eval_requests_out)++;
          EvalResult res = evaluator->Evaluate(obs);
          std::vector<float> logits = std::move(res.logits);
          CenterAndCapLegalLogits(logits, actions, 10.0f);
          action = SampleActionFromLogits(rng, logits, actions);
        }
      } else {
        action = ChooseHeuristicAcquisitionAction(*state, actions, owner, rng);
      }
    } else {
      bool block_taken = false;
      if (cfg.block_aware_opponents && dune_state != nullptr) {
        int owner_solari = dune_state->GetPlayerSolari(owner);
        bool owner_is_leto = (dune_state->PlayerLeader(owner) == 4);
        int needed_solari = owner_is_leto ? 7 : 8;
        if (owner_solari >= needed_solari - 2) {
          std::uniform_real_distribution<double> dist_u(0.0, 1.0);
          if (dist_u(*rng) < 0.25) {
            if (std::find(actions.begin(), actions.end(), dune_imperium::kActionAgentSpaceSwordmaster) != actions.end()) {
              action = dune_imperium::kActionAgentSpaceSwordmaster;
              block_taken = true;
            } else if (std::find(actions.begin(), actions.end(), dune_imperium::kActionAgentSpaceSmuggling) != actions.end()) {
              action = dune_imperium::kActionAgentSpaceSmuggling;
              block_taken = true;
            }
          }
        }
      }
      if (!block_taken) {
        if (cfg.use_policy_for_opponents && evaluator != nullptr) {
          auto game = state->GetGame();
          int64_t obs_size = game->GetType().provides_information_state_tensor
                                 ? game->InformationStateTensorSize()
                                 : game->ObservationTensorSize();
          std::vector<float> obs(obs_size, 0.0f);
          state->ObservationTensor(cur_player, absl::MakeSpan(obs));
          if (eval_requests_out) (*eval_requests_out)++;
          EvalResult res = evaluator->Evaluate(obs);
          std::vector<float> logits = std::move(res.logits);
          CenterAndCapLegalLogits(logits, actions, 10.0f);
          action = SampleActionFromLogits(rng, logits, actions);
        } else {
          std::uniform_int_distribution<int> dis(0, actions.size() - 1);
          action = actions[dis(*rng)];
        }
      }
    }

    state->ApplyAction(action);
  }

  if (dune_state != nullptr) {
    if (dune_state->HasSwordmaster(owner)) {
      int purchase_round = dune_state->GetCurrentRound();
      return cfg.success_score + (cfg.deadline_round - purchase_round) * cfg.round_bonus;
    }
    return ScoreSwordmasterRaceState(*state, owner, cfg);
  }
  return 0.0;
}

} // namespace open_spiel
