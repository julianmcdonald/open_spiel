#include "dune_search_routing.h"
#include "open_spiel/games/dune_imperium/dune_imperium.h"
#include <algorithm>

namespace open_spiel {

using dune_imperium::GamePhase;

DuneDecisionRole ClassifyDuneDecisionRole(const State& state, Player searched_player, bool has_active_session) {
  // 1. Forced/bookkeeping
  if (searched_player == kChancePlayerId) {
    return DuneDecisionRole::kForcedOrBookkeeping;
  }
  std::vector<Action> legal_actions = state.LegalActions();
  if (legal_actions.size() <= 1) {
    return DuneDecisionRole::kForcedOrBookkeeping;
  }

  // Cast to DuneImperiumState
  const dune_imperium::DuneImperiumState& dune_state =
      static_cast<const dune_imperium::DuneImperiumState&>(state);

  GamePhase phase = dune_state.phase();

  // 2. Leader Draft / Selection
  if (phase == GamePhase::kLeaderDraft || phase == GamePhase::kLeaderOfferChance) {
    return DuneDecisionRole::kLeaderSelection;
  }

  // 3. Purchase: Buy Imperium card, Tech tile, Tleilaxu card, or reservation.
  bool has_purchase = false;
  bool all_purchase_or_end_turn = true;
  for (Action a : legal_actions) {
    bool is_purchase_action = 
        (a >= dune_imperium::kActionBuyImperiumRow0 && a < dune_imperium::kActionBuyImperiumRow0 + 8) ||
        (a >= dune_imperium::kActionBuyHelenaReserve0 && a < dune_imperium::kActionBuyHelenaReserve0 + 8) ||
        (a == dune_imperium::kActionBuyReserveArrakisLiaison) ||
        (a == dune_imperium::kActionBuyReserveTheSpiceMustFlow) ||
        (a >= dune_imperium::kActionTechAcquire0 && a <= dune_imperium::kActionTechAcquireSkip) ||
        (a >= dune_imperium::kActionTleilaxuAcquire0 && a <= dune_imperium::kActionTleilaxuAcquireSkip) ||
        (a >= dune_imperium::kActionTechAcquireWithSolari0 && a <= dune_imperium::kActionTechAcquireWithSolariSkip);
    if (is_purchase_action) {
      has_purchase = true;
    }
    if (!is_purchase_action && a != dune_imperium::kActionEndTurn) {
      all_purchase_or_end_turn = false;
      break;
    }
  }
  if (has_purchase && all_purchase_or_end_turn) {
    return DuneDecisionRole::kPurchase;
  }

  // 4. Combat Intrigue: plays and pass in combat phase.
  {
    bool has_combat_action = false;
    for (Action a : legal_actions) {
      bool is_combat = (a == dune_imperium::kActionCombatPass) ||
                       (a >= dune_imperium::kActionPlayIntrigueCombatCard0 && a < dune_imperium::kActionPlayIntrigueCombatCard0 + dune_imperium::kMaxIntrigueCards) ||
                       (a >= dune_imperium::kActionCombatCommit0 && a < dune_imperium::kActionCombatCommit0 + 48);
      if (is_combat) {
        has_combat_action = true;
        break;
      }
    }
    if (has_combat_action) {
      return DuneDecisionRole::kCombatIntrigue;
    }
  }

  // 5. Agent Primary vs Agent Continuation
  if (phase == GamePhase::kAgentTurns) {
    int agents_remaining = dune_state.GetPlayerAgentsRemainingForTesting(searched_player);
    if (agents_remaining > 0) {
      bool has_agent_card_play = false;
      bool has_space_place = false;
      for (Action a : legal_actions) {
        if ((a >= dune_imperium::kActionSelectAgentCard0 && a < dune_imperium::kActionSelectAgentCard0 + 256) ||
            a == dune_imperium::kActionPlayKwisatzHaderach) {
          has_agent_card_play = true;
        }
        if (a >= dune_imperium::kActionAgentSpaceConspire && a <= dune_imperium::kActionAgentSpaceSietchTabr) {
          has_space_place = true;
        }
      }
      if (has_agent_card_play && !has_space_place) {
        return DuneDecisionRole::kAgentPrimary;
      }
      if (has_space_place || has_active_session) {
        return DuneDecisionRole::kAgentContinuation;
      }
    } else {
      bool has_intrigue_plot = false;
      bool has_reveal = false;
      for (Action a : legal_actions) {
        if (a >= dune_imperium::kActionPlayIntriguePlotCard0 && a < dune_imperium::kActionPlayIntriguePlotCard0 + dune_imperium::kMaxIntrigueCards) {
          has_intrigue_plot = true;
        }
        if (a == dune_imperium::kActionReveal) {
          has_reveal = true;
        }
      }
      if (has_intrigue_plot && has_reveal) {
        return DuneDecisionRole::kOtherOptional;
      }
    }
  }

  return DuneDecisionRole::kOtherOptional;
}

} // namespace open_spiel
