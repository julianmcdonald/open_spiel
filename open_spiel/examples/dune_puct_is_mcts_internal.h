#ifndef OPEN_SPIEL_EXAMPLES_DUNE_PUCT_IS_MCTS_INTERNAL_H_
#define OPEN_SPIEL_EXAMPLES_DUNE_PUCT_IS_MCTS_INTERNAL_H_

#include <cstdint>
#include <string>
#include <vector>

#include "dune_puct_is_mcts.h"

namespace open_spiel {

Action GetRootArgmaxAction(const DuneISMCTSNode* root_node,
                           const std::vector<Action>& legal_actions);

namespace dune_puct_internal {

// FNV-1a over a state key gives a stable identity for a position that is cheap
// to compute and does not depend on iteration order or pointer values.
uint64_t FnvHash64(const std::string& s);

// PWO-4 Amendment 2 section 5.1. Root-child visits in legal-action order.
// Pure read: no RNG, no tree mutation, no allocation into the tree. Actions with
// no child yet read 0, which is what a cold root's children legitimately are.
void SnapshotRootVisits(const DuneISMCTSNode* root_node,
                        const std::vector<Action>& legal_actions,
                        std::vector<int>* out);

}  // namespace dune_puct_internal
}  // namespace open_spiel

#endif  // OPEN_SPIEL_EXAMPLES_DUNE_PUCT_IS_MCTS_INTERNAL_H_
