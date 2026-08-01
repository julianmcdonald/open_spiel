// PF-2 Part B — the matched-hybrid controller's frozen decision rules.
//
// WHY THIS HEADER EXISTS
//
// These four rules used to live inline inside dune_search_benchmark.cc's
// worker loop, where nothing outside that translation unit could reach them.
// The conformance tests therefore re-expressed each rule and tested the
// re-expression — which proves that two copies of a rule agree, not that the
// binary implements it. Sol's review named the failure precisely: the tests
// "test reference functions rather than the production matched-hybrid
// branch," and one of them compared the same expression to itself.
//
// So the rules moved HERE, the benchmark now CALLS them, and the tests call
// the SAME functions. A test that passes now says something about the binary.
//
// Everything in this header is a pure function of its arguments. Nothing
// reads a searched arm's output: the matched arm and the searched arm share a
// RULE, never a decision instance, and a search.jsonl replay implementation is
// forbidden by the work order.

#ifndef OPEN_SPIEL_EXAMPLES_DUNE_PF2_MATCHED_FALLBACK_H_
#define OPEN_SPIEL_EXAMPLES_DUNE_PF2_MATCHED_FALLBACK_H_

#include <cstdint>
#include <random>
#include <string>

#include "open_spiel/abseil-cpp/absl/random/distributions.h"
#include "open_spiel/games/dune_imperium/dune_imperium.h"
#include "open_spiel/spiel.h"
#include "dune_puct_is_mcts.h"
#include "dune_seed_utils.h"

namespace open_spiel {
namespace pf2 {

// ---------------------------------------------------------------------------
// WHERE the matched arm samples. The frozen eligibility predicate.
//
// TWO clauses, not one. RunSearch() returns early on single-legal decisions
// BEFORE the strategic check (dune_puct_is_mcts.cc:809 precedes :818), so a
// single-legal decision never produces a fallback. !IsStrategicState alone
// returns TRUE for them (the predicate is false below 2 legals) and would
// sweep in every forced action — inflating the matched class roughly
// eightfold and letting a count gate pass on decisions that were never
// choices.
//
// IsStrategicState, NOT the role-based `is_strategic` telemetry field: they
// disagree on whole classes. Purchase and combat-intrigue decisions are
// STRATEGIC under the gate that actually produces Path B's fallbacks and
// non-strategic under the roles, so routing by role would put the matched arm
// at decisions the searched arm searches.
// ---------------------------------------------------------------------------
inline bool MatchedFallbackEligible(const State& state, Player searched_player) {
  return state.LegalActions().size() > 1 &&
         !IsStrategicState(state, searched_player);
}

// ---------------------------------------------------------------------------
// WHEN in the stream it draws. The frozen seed derivation.
//
// Mirrors DunePUCTISMCTSBot::Step (dune_puct_is_mcts.cc:1179-1180):
// Combine(config.seed, kStreamBlueprint, ordinal). The ordinal is 1-based —
// Path B increments BEFORE deriving, so the first decision derives with 1.
// ---------------------------------------------------------------------------
inline uint64_t MatchedFallbackStepSeed(uint64_t config_seed, uint64_t ordinal) {
  return dune_seed::Combine(config_seed, dune_seed::kStreamBlueprint, ordinal);
}

inline double MatchedFallbackRVal(uint64_t step_seed) {
  std::mt19937 step_rng(step_seed);
  return absl::Uniform(step_rng, 0.0, 1.0);
}

// ---------------------------------------------------------------------------
// WHAT it draws. OpenSpiel's SampleAction, which is what Path B's Step() uses
// — NOT the session's SampleActionFromPrior. The session's own comment
// (dune_search_session.cc:44) records that the two differ at cumulative
// boundaries, and revision 21 of the source of record froze the wrong one.
// ---------------------------------------------------------------------------
inline Action MatchedFallbackSelect(const ActionsAndProbs& policy, double r_val) {
  return SampleAction(policy, r_val).first;
}

// ---------------------------------------------------------------------------
// PF-2 Part A. Duplicated from the anonymous-namespace helper in
// dune_search_session.cc:18, which has internal linkage; lifting that one
// would modify a translation unit other binaries link, which Part A's
// flag-OFF neutrality projection must not do. The two must stay in step: the
// fresh path's `phase` string is only comparable to the session path's if
// they map identically.
// ---------------------------------------------------------------------------
inline std::string PathBPhaseToString(dune_imperium::GamePhase phase) {
  switch (phase) {
    case dune_imperium::GamePhase::kLeaderOfferChance: return "kLeaderOfferChance";
    case dune_imperium::GamePhase::kLeaderDraft: return "kLeaderDraft";
    case dune_imperium::GamePhase::kDeal: return "kDeal";
    case dune_imperium::GamePhase::kRoundStart: return "kRoundStart";
    case dune_imperium::GamePhase::kAgentTurns: return "kAgentTurns";
    case dune_imperium::GamePhase::kRevealTurns: return "kRevealTurns";
    case dune_imperium::GamePhase::kCombat: return "kCombat";
    case dune_imperium::GamePhase::kMakers: return "kMakers";
    case dune_imperium::GamePhase::kRecall: return "kRecall";
    case dune_imperium::GamePhase::kTerminal: return "kTerminal";
    default: return "Unknown";
  }
}

// Part A's field population for the fresh path, as one callable unit so the
// field regression test exercises what the binary runs. Telemetry only: no
// control-flow, RNG or action-selection effect.
template <typename DiagnosticsT>
inline void PopulatePathBDiagnostics(DiagnosticsT& diag, const State& state,
                                     int decision_role_int) {
  const auto* dune_state =
      dynamic_cast<const dune_imperium::DuneImperiumState*>(&state);
  if (dune_state != nullptr) {
    diag.round = dune_state->GetCurrentRound();
    diag.phase = PathBPhaseToString(dune_state->phase());
  }
  diag.decision_role = std::to_string(decision_role_int);
}

}  // namespace pf2
}  // namespace open_spiel

#endif  // OPEN_SPIEL_EXAMPLES_DUNE_PF2_MATCHED_FALLBACK_H_
