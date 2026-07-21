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

// Heuristic goal-biased action selection for owner when policy is bypassed.
//
// `return_invalid_on_no_acquisition` (default false preserves Task-10 behavior):
// when true this is the SWORDMASTER PROBE ladder; the caller (arm B) MUST fall
// through to normal play/search on kInvalidAction. It exists ONLY to fund and
// reach Swordmaster, and is fully probe-gated so Task-10 call sites are unchanged:
//   * Tier 1: buy Swordmaster (direct or card path) whenever reachable.
//   * While the seat CANNOT afford it (solari < 8, or 7 for Leto): climb the
//     funding ladder — Interstellar Shipping (2 freighter moves) > Smuggling
//     (1 move) > Wealth (+2 solari); resolve the shipping track by advancing
//     while >1 movement remains (or at level 0), else recalling; force 5 solari
//     (Dividends) at the tier-1 recall reward and leave tier-2+ rewards to
//     policy. EXCEPTION: when exactly 2 solari short AND at freighter level 0,
//     take Wealth first (its immediate +2 beats an unfinished freighter path).
//   * Once the seat CAN afford it but Swordmaster is not reachable this decision
//     (blocked / no access): if a 2nd agent turn remains, draw a card via
//     Arrakeen (different access than Swordmaster) to fish for access; otherwise
//     stop chasing solari and play the best move.
// At any decision with no forced move, return kInvalidAction instead of a
// uniform-random legal action — a random fallback would randomize unrelated
// purchase/combat/reveal decisions (the Phase-18A uniform-policy poison class)
// and misattribute that damage to the swordmaster hypothesis.
inline Action ChooseHeuristicAcquisitionAction(
    const State& state,
    const std::vector<Action>& actions,
    Player owner,
    std::mt19937* rng,
    bool return_invalid_on_no_acquisition = false) {

  if (std::find(actions.begin(), actions.end(), dune_imperium::kActionAgentSpaceSwordmaster) != actions.end()) {
    return dune_imperium::kActionAgentSpaceSwordmaster;
  }
  
  Action sm_card = FindActionOrCardPathToSpace(state, owner, dune_imperium::kActionAgentSpaceSwordmaster, rng);
  if (sm_card != kInvalidAction && std::find(actions.begin(), actions.end(), sm_card) != actions.end()) {
    return sm_card;
  }

  if (return_invalid_on_no_acquisition) {
    // ===================== SWORDMASTER PROBE ladder =====================
    // Everything below exists ONLY to fund and reach Swordmaster. Tier 1 above
    // already buys it when reachable. Any decision with no forced move returns
    // kInvalidAction so the arm-B driver routes it to the normal search path.
    const auto* ds = dynamic_cast<const dune_imperium::DuneImperiumState*>(&state);
    if (ds == nullptr) return kInvalidAction;

    auto has = [&](Action a) {
      return std::find(actions.begin(), actions.end(), a) != actions.end();
    };
    auto reach_space = [&](Action space) -> Action {
      if (has(space)) return space;
      Action c = FindActionOrCardPathToSpace(state, owner, space, rng);
      if (c != kInvalidAction && has(c)) return c;
      return kInvalidAction;
    };

    const int solari = ds->GetPlayerSolari(owner);
    const int cost = (ds->PlayerLeader(owner) == 4) ? 7 : 8;  // Leto 7, else 8

    if (solari < cost) {
      // ---- Cannot afford yet: climb the funding ladder. ----
      const int level = ds->PlayerShippingLevelForTesting(owner);

      // Freighter-move spaces (Interstellar > Smuggling) outrank Wealth, EXCEPT
      // when the seat is exactly 2 solari short AND the freighter is at level 0:
      // from level 0 a freighter path needs >=2 movements to yield solari (so it
      // may not complete), whereas Wealth's immediate +2 secures the buy. Then
      // prefer Wealth. Interstellar grants 2 movements, Smuggling 1.
      const bool wealth_first = (solari == cost - 2) && (level == 0);
      Action a;
      if (wealth_first) {
        a = reach_space(dune_imperium::kActionAgentSpaceWealth);               if (a != kInvalidAction) return a;
        a = reach_space(dune_imperium::kActionAgentSpaceInterstellarShipping); if (a != kInvalidAction) return a;
        a = reach_space(dune_imperium::kActionAgentSpaceSmuggling);            if (a != kInvalidAction) return a;
      } else {
        a = reach_space(dune_imperium::kActionAgentSpaceInterstellarShipping); if (a != kInvalidAction) return a;
        a = reach_space(dune_imperium::kActionAgentSpaceSmuggling);            if (a != kInvalidAction) return a;
        a = reach_space(dune_imperium::kActionAgentSpaceWealth);               if (a != kInvalidAction) return a;
      }

      // Shipping-track resolution (spend queued freighter movements for solari).
      // (a) Start the next queued movement.
      if (has(dune_imperium::kActionAgentRewardShipping)) return dune_imperium::kActionAgentRewardShipping;
      // (b) Advance while more than one movement remains (or forced at level 0);
      // else recall to bank the reward. pending_reward_shipping()[owner] counts
      // movements queued AFTER the current one, so >=1 means "more than one
      // remaining". Recalling from a higher level also collects the lower
      // levels' rewards on the way back to 0.
      const bool can_advance = has(dune_imperium::kActionShippingAdvance);
      const bool can_recall = has(dune_imperium::kActionShippingRecall);
      if (can_advance || can_recall) {
        const int queued = ds->pending_reward_shipping()[owner];
        if (level == 0 && can_advance) return dune_imperium::kActionShippingAdvance;
        if (queued >= 1 && can_advance) return dune_imperium::kActionShippingAdvance;
        if (can_recall) return dune_imperium::kActionShippingRecall;
        if (can_advance) return dune_imperium::kActionShippingAdvance;
      }
      // (c) Tier-1 recall reward: FORCE 5 solari (Dividends), never 2 spice.
      // Tier-2+ reward choices are left to policy (fall through to kInvalidAction).
      if (has(dune_imperium::kActionShippingLevel1Dividends)) return dune_imperium::kActionShippingLevel1Dividends;

      return kInvalidAction;
    }

    // ---- Can afford Swordmaster, but Tier 1 could not reach it this decision. ----
    // Blocked, or the seat lacks Swordmaster access. If the seat still has a 2nd
    // agent turn AFTER this placement, draw a card now (it may draw access): use
    // Arrakeen, a card-draw space whose access DIFFERS from Swordmaster's (Mentat
    // is excluded — it gates the same as Swordmaster, so it would be unreachable
    // for the same reason). GetPlayerAgentsRemainingForTesting counts the agent
    // about to be placed, so >= 2 means "a 2nd agent turn remains after this".
    // With only one agent left, or if Arrakeen is unavailable, play the best move
    // and take Swordmaster next round. Rare (the RL agent almost never races it).
    if (ds->GetPlayerAgentsRemainingForTesting(owner) >= 2 &&
        has(dune_imperium::kActionAgentSpaceArrakeen)) {
      return dune_imperium::kActionAgentSpaceArrakeen;
    }
    // Other card-draw mechanics the seat may hold (Paul's signet ring, a
    // card-drawing intrigue) are left to policy for now (see the probe design).
    return kInvalidAction;
  }

  // ===================== Task-10 legacy ladder (UNCHANGED) =====================
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
