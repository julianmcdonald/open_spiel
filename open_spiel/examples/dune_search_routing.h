#ifndef OPEN_SPIEL_EXAMPLES_DUNE_SEARCH_ROUTING_H_
#define OPEN_SPIEL_EXAMPLES_DUNE_SEARCH_ROUTING_H_

#include "open_spiel/spiel.h"

namespace open_spiel {

enum class DuneDecisionRole {
  kForcedOrBookkeeping,
  kLeaderSelection,
  kAgentPrimary,
  kAgentContinuation,
  kPurchase,
  kCombatIntrigue,
  kOtherOptional,
};

DuneDecisionRole ClassifyDuneDecisionRole(const State& state, Player searched_player, bool has_active_session);

}  // namespace open_spiel

#endif  // OPEN_SPIEL_EXAMPLES_DUNE_SEARCH_ROUTING_H_
