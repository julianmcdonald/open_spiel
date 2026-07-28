// Shared primitives for the PWO-3 tools (enrichment strata, teacher audit,
// rollout oracle).
//
// Everything here is frozen by docs/PWO3_REGISTRATION.md. PWO-2's derivations are
// INHERITED unchanged via dune_pwo2_common.h -- deliberately, because that
// inheritance is what makes this WO's fixed_800 rows on the 192 original roots a
// reproduction tripwire for the rebuild (registration section 6.1.1) and what keeps
// the cached oracle cells paired with every new one.
//
// Only NEW registered quantities live here.

#ifndef OPEN_SPIEL_EXAMPLES_DUNE_PWO3_COMMON_H_
#define OPEN_SPIEL_EXAMPLES_DUNE_PWO3_COMMON_H_

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

#include "open_spiel/spiel.h"

#include "dune_pwo2_common.h"
#include "dune_seed_utils.h"

namespace pwo3 {

// ---------------------------------------------------------------------------
// Registration section 3.2: ONE new-selection seed, with an enumerated tag table.
// No other tag exists.
// ---------------------------------------------------------------------------
constexpr uint64_t kNewSelectionSeed = 20260728;

enum PwoTag : uint64_t {
  // 300 + arm_index: extension-episode half assignment (section 5.4).
  kTagExtensionHalfBase = 300,
  kTagConversionRank = 401,
  kTagSmRank = 402,
  kTagLiveAuditSubsample = 501,
  kTagReplicationSubset = 502,
  kTagConversionSanityRoots = 503,
  kTagValidationLiveSample = 504,
  kTagConversionSanityActionsBase = 505,
};

// Per-root rank key: DeriveSeed(DeriveSeed(20260728, tag), HexPrefix64(hash)).
// A function of the root's identity alone -- not of when in the game it occurred,
// not of which seat acted, not of any search or outcome information. This shape
// mirrors PWO-2's CorpusRankKey, whose predecessor (ranking by ascending
// decision_index) caused the Amendment-1 premise failure.
inline uint64_t RankKey(uint64_t tag, const std::string& history_hash) {
  return dune_seed::DeriveSeed(dune_seed::DeriveSeed(kNewSelectionSeed, tag),
                               pwo2::HexPrefix64(history_hash));
}

inline uint64_t SubsampleSeed(uint64_t tag) {
  return dune_seed::DeriveSeed(kNewSelectionSeed, tag);
}

// ---------------------------------------------------------------------------
// Conversion and Swordmaster action identities.
//
// kActionConvertSpecimenToTroop0 = 740 is an UNUSED BASE CONSTANT / sentinel:
// the engine names conversions only over base+1 .. base+12 (dune_imperium.cc:
// 517-520), every legal-action site pushes base + i for a conversion AMOUNT
// (:6556, :6716, :6732, :6870), and the apply handler accepts only base+1..base+12
// (:8457-8459). 740 is never legal and is NOT a "zero/skip" action -- PWO-2 final
// report section 12.4. The real range is 741..752 = convert 1..12.
// ---------------------------------------------------------------------------
constexpr open_spiel::Action kConversionBase = 740;
constexpr open_spiel::Action kConversionMin = 741;
constexpr open_spiel::Action kConversionMax = 752;
// kActionAgentSpaceSwordmaster, games/dune_imperium/dune_imperium_content.h:63.
constexpr open_spiel::Action kSwordmasterAction = 610;

inline bool IsConversion(open_spiel::Action a) {
  return a >= kConversionMin && a <= kConversionMax;
}

// Conversion amount, or -1 when the action is not a conversion. Never called on
// 740, which IsConversion rejects.
inline int ConversionAmount(open_spiel::Action a) {
  return IsConversion(a) ? static_cast<int>(a - kConversionBase) : -1;
}

inline std::vector<open_spiel::Action> LegalConversionActions(
    const std::vector<open_spiel::Action>& legal) {
  std::vector<open_spiel::Action> out;
  for (open_spiel::Action a : legal)
    if (IsConversion(a)) out.push_back(a);
  return out;
}

inline bool HasConversion(const std::vector<open_spiel::Action>& legal) {
  return !LegalConversionActions(legal).empty();
}

inline bool HasSwordmaster(const std::vector<open_spiel::Action>& legal) {
  return std::find(legal.begin(), legal.end(), kSwordmasterAction) != legal.end();
}

// ---------------------------------------------------------------------------
// Registration section 14.2: PWO-3 registers the LOWEST-ACTION-ID tie-break
// everywhere. The repo carries two rules -- GetRootArgmaxAction and
// ArgmaxVisitAction break ties toward the first action in LEGAL ORDER, while
// ArgmaxPolicy and the PWO-2 analyzer break toward the lowest ID. Verified at
// registration that legal_actions is ascending on all 192 corpus roots and
// root_actions on all 2,688 PWO-2 rows, so the rules COINCIDE; every emitter
// asserts ascending order so they cannot silently diverge.
// ---------------------------------------------------------------------------
inline bool IsAscending(const std::vector<open_spiel::Action>& v) {
  return std::is_sorted(v.begin(), v.end()) &&
         std::adjacent_find(v.begin(), v.end()) == v.end();
}

template <typename T>
open_spiel::Action ArgmaxLowestId(const std::vector<open_spiel::Action>& actions,
                                  const std::vector<T>& weights) {
  SPIEL_CHECK_EQ(actions.size(), weights.size());
  SPIEL_CHECK_GT(actions.size(), 0u);
  SPIEL_CHECK_TRUE(IsAscending(actions));
  size_t best = 0;
  for (size_t i = 1; i < weights.size(); ++i)
    if (weights[i] > weights[best]) best = i;
  return actions[best];
}

// Registration section 5.2 / PWO-2 Amendment 1 section 3.6-A: round buckets.
inline int RoundBucket(int round) {
  if (round <= 2) return 0;
  if (round <= 4) return 1;
  if (round <= 6) return 2;
  return 3;
}

// The coverage gate, hardcoded at dune_puct_is_mcts.cc:942-943. PWO-3 AUDITS its
// effect and NEVER modifies it; this is a read-only mirror so the emitter can
// record `coverage_gate_fired` from the gate's own INPUTS rather than keying it on
// `fallback_reason` -- on 120 of PWO-2's 384 live rows the gate genuinely fired
// while fallback_reason still read "timeout" (registration section 14.4).
constexpr int kCoverageReqActionsCap = 3;
constexpr double kCoveragePriorThreshold = 0.50;

inline bool CoverageGateFires(int num_covered_actions, double covered_prior_mass,
                              int n_legal) {
  const int req = std::min(kCoverageReqActionsCap, n_legal);
  return num_covered_actions < req || covered_prior_mass < kCoveragePriorThreshold;
}

}  // namespace pwo3

#endif  // OPEN_SPIEL_EXAMPLES_DUNE_PWO3_COMMON_H_
