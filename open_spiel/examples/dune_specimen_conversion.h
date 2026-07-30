#ifndef OPEN_SPIEL_EXAMPLES_DUNE_SPECIMEN_CONVERSION_H_
#define OPEN_SPIEL_EXAMPLES_DUNE_SPECIMEN_CONVERSION_H_

// The ConvertSpecimenToTroop action range, in ONE place.
//
// PWO-5 gate 2 item (c) (docs/PWO5_PILOT_REGISTRATION.md section 16).
//
// THE REAL RANGE IS 741-752, NOT 740-752.
//
// `kActionConvertSpecimenToTroop0 == 740` is an UNUSED BASE CONSTANT: the amount
// converted is `action_id - kActionConvertSpecimenToTroop0`, so 740 would mean
// "convert zero specimens" and is never enqueued and never legal. The engine's
// own live decode agrees -- `dune_imperium.cc:517-518` and `:8457-8458` both
// test `base + 1 .. base + 12` -- and the PWO-2 final report section 12.4, the
// PWO-4 analyzer (`scripts/eval/pwo4_trajectory_analysis.py`) and its test
// (`tests/test_pwo4_trajectory_analysis.py:912-931`) all pin the same boundary.
//
// A range written as 740..752 is the NATURAL MISTAKE, because 740 is the base
// the amount is measured from. Three C++ sites had made it, each independently:
//
//   * `dune_ppo_train.cc`          -- the specimen-exchange shaping guard
//   * `dune_search_benchmark.cc`   -- the `specimen_conversions` telemetry counter
//   * `dune_population_eval.cc`    -- the same telemetry counter, evaluation side
//
// All three now call the single predicate below, so the boundary cannot drift
// apart again.
//
// NOT A BEHAVIOUR CHANGE, and that is checkable rather than asserted: 740 is
// absent from all 10,903 conversion-legal rows of the frozen PWO-4 stream, so no
// committed number moves. It is correctness hygiene on a latent trap.
//
// The engine is FROZEN (tree a67f1925305d35d9a2f30fd9658b9f01fe21bdd6) and is
// NOT touched by this fix. `dune_imperium_content.h:315-316`'s span table entry
// `{base, base + 12}  // 740-752` is an action-CATEGORY span, not a legality
// claim, and it stays exactly as it is: editing it would change the engine tree
// hash, which is STOP condition 9.

#include "open_spiel/spiel.h"  // Action
#include "open_spiel/games/dune_imperium/dune_imperium_content.h"

namespace open_spiel {
namespace dune_shaping {

// True iff `action` is a real ConvertSpecimenToTroop action (ids 741-752).
inline constexpr bool IsSpecimenConversionAction(Action action) {
  return action >= dune_imperium::kActionConvertSpecimenToTroop0 + 1 &&
         action <= dune_imperium::kActionConvertSpecimenToTroop0 + 12;
}

// The number of specimens a conversion action converts. Only meaningful when
// IsSpecimenConversionAction(action) is true; returns 0 otherwise, which is the
// same value the unused base constant would decode to and is never a real
// conversion amount.
inline constexpr int SpecimenConversionAmount(Action action) {
  return IsSpecimenConversionAction(action)
             ? static_cast<int>(action -
                                dune_imperium::kActionConvertSpecimenToTroop0)
             : 0;
}

}  // namespace dune_shaping
}  // namespace open_spiel

#endif  // OPEN_SPIEL_EXAMPLES_DUNE_SPECIMEN_CONVERSION_H_
