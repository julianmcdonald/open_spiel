#ifndef OPEN_SPIEL_EXAMPLES_DUNE_WIDE_ROLLOUT_SEARCH_H_
#define OPEN_SPIEL_EXAMPLES_DUNE_WIDE_ROLLOUT_SEARCH_H_

#include <algorithm>
#include <cstdint>
#include <memory>
#include <vector>

#include "open_spiel/examples/dune_compound_turn_search.h"

namespace open_spiel {
namespace dune_imperium {

inline CompoundSearchDecisionResult WideRolloutSearchDecision(
    const State& state, Player search_player, TurnSearchTree* turn_tree,
    const std::vector<std::shared_ptr<BatchedNNEvaluator>>& evaluators,
    int reference_rollouts_per_action, double min_override_margin,
    uint64_t decision_seed) {
  const auto legal = state.LegalActions();
  if (legal.size() <= 3) {
    return CompoundRolloutSearchDecision(
        state, search_player, turn_tree, evaluators, 3,
        reference_rollouts_per_action, min_override_margin,
        TreeReuseMode::kAdditive, decision_seed, SearchPairingMode::kUnpaired,
        SearchTruncationMode::kTerminal);
  }
  const int top_k = std::min(6, static_cast<int>(legal.size()));
  const int rollouts_per_action = 3 * reference_rollouts_per_action / top_k;
  SPIEL_CHECK_GE(rollouts_per_action, 1);
  return CompoundRolloutSearchDecision(
      state, search_player, turn_tree, evaluators, top_k, rollouts_per_action,
      min_override_margin, TreeReuseMode::kAdditive, decision_seed,
      SearchPairingMode::kPaired, SearchTruncationMode::kTerminal);
}

}  // namespace dune_imperium
}  // namespace open_spiel

#endif  // OPEN_SPIEL_EXAMPLES_DUNE_WIDE_ROLLOUT_SEARCH_H_
