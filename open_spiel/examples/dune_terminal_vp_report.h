#ifndef OPEN_SPIEL_EXAMPLES_DUNE_TERMINAL_VP_REPORT_H_
#define OPEN_SPIEL_EXAMPLES_DUNE_TERMINAL_VP_REPORT_H_

// THE ONE definition of "a seat's final VP, for reporting".
//
// WHY THIS HEADER EXISTS. `dune_pwo4_trajectory.cc` and
// `dune_search_benchmark.cc` each carried their own hand-written copy of the
// engine's endgame-VP arithmetic, both named GetTrueFinalVp, one copied from
// the other. Both copies awarded the "all four factions at influence >= 3" +1
// UNCONDITIONALLY, while the engine's `ComputeEndgameVp` awards it only to a
// seat owning TECH TILE 8 (Memocorders) -- dune_imperium.cc:2303-2310. The
// copies therefore OVERSTATED by exactly 1 for any seat at >= 3 influence in
// all four factions without Memocorders.
//
// Measured on the frozen PWO-4 corpus: 212 of 1,600 (game, seat) `final_vp`
// values are inflated, 59 of them on searched seats. That corpus is preserved
// unrewritten as historical evidence; see
// docs/PWO5_AMENDMENT_1_TARGET_EXPOSURE_TELEMETRY_2026_07_31.md section 10.
//
// SCOPE. This was a REPORTING defect, never a reward defect. `Returns()` ranks
// placements on `vp_[p] + ComputeEndgameVp(p)` (dune_imperium.cc:2357), and
// `FinalScoredVp` returns that same expression (:2403) -- so placements,
// rewards and every committed `returns` array were always correct.
//
// THE RULE THIS HEADER ENFORCES: there is exactly one reporting definition, it
// is a call into the engine, and it is tested. DO NOT reimplement endgame
// scoring anywhere. If a reporter needs a seat's final VP, include this header.
//
// The single sanctioned exception is `dune_pwo5_prepare.cc`'s
// `Pwo4LegacyReportedVp`, which is a FROZEN HISTORICAL REPRODUCTION of the old
// arithmetic used solely to audit the already-frozen corpus against the replay.
// It is not a scoring function and it is not this one.

#include "open_spiel/games/dune_imperium/dune_imperium.h"

namespace open_spiel {
namespace dune_report {

// A seat's final scored VP, as the engine scores it.
//
// Precondition: `state` is terminal. `FinalScoredVp` asserts it.
inline int TerminalVpForReporting(
    const dune_imperium::DuneImperiumState* state, int player) {
  return state->FinalScoredVp(player);
}

}  // namespace dune_report
}  // namespace open_spiel

#endif  // OPEN_SPIEL_EXAMPLES_DUNE_TERMINAL_VP_REPORT_H_
