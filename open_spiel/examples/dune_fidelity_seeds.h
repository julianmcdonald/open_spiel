// Seed derivation for the critic fidelity gate (dune_critic_fidelity_gate.cc).
//
// Every rollout seed the gate draws is defined here so the pairing contract of
// each family is stated once, next to the arithmetic that implements it.
//
// Two families, deliberately governed by different rules:
//
//   GATE families -- Gate 1 rollouts, the Gate 2 paired raw-vs-search
//   rollouts, the Gate 2 forced-action continuations, the controller step
//   draw, and the MCTS bot seed. These values are load-bearing: every recorded
//   gate verdict was produced under them. They keep their historical
//   fixed-width arithmetic `base + 1000 * state + replicate` verbatim, locked
//   by exact-value tests in dune_fidelity_seeds_test.cc so a future refactor
//   cannot silently re-roll a gate metric. Injectivity in (state, replicate)
//   holds only while replicate < kMaxRollouts, which the binary now enforces
//   up front instead of assuming.
//
//   DIAGNOSTIC families -- the successor and leaf raw-policy rollouts, which
//   feed only the reporting metrics successor_rmse, successor_mean_bias,
//   leaf_mean_bias, leaf_rmse and choice_rank_correlation. Their previous
//   scheme `base + 1000 * state + 100 * index + replicate` reused seeds
//   whenever a root had more than ~8 actions (100 * index overflowed the
//   1000-wide state block) or whenever --diagnostic_rollouts exceeded the
//   100-wide index spacing -- and the default is 256, so the collision was
//   live in every standard run. These are now derived from domain-separated
//   ordered tuples via dune_seed::DeriveSeed, which carries no width
//   assumption at any coordinate.
//
// WO-18 scope note: the gate families were intentionally NOT re-derived.
// Their collisions are latent outside the newly enforced rollout range, and
// re-deriving them would change gate metrics and invalidate historical
// comparisons and caches for no correctness gain. A full protocol reseed
// remains available as separate future work.

#pragma once

#include <cstdint>

#include "dune_seed_utils.h"

namespace dune_fidelity_seeds {

// Widest replicate index the fixed-width gate families can represent without
// running into the next state's block: `base + 1000 * state + replicate` is
// injective in (state, replicate) only while replicate < 1000. Replicates are
// 1-based, so --rollouts must satisfy 1 <= rollouts < kMaxRollouts.
inline constexpr int kMaxRollouts = 1000;

// Historical bases. These are literals in the gate arithmetic rather than
// flags; --rollout_seed_base / --chance_seed_base are threaded in explicitly
// where the original code read them.
inline constexpr uint64_t kGate1RolloutBase    = 300000;
inline constexpr uint64_t kGate2PairedBase     = 100000;
inline constexpr uint64_t kControllerStepBase  = 300000;
inline constexpr uint64_t kSearchBotBase       = 200000;

inline constexpr uint64_t kStateBlockWidth = 1000;

// ---------------------------------------------------------------------------
// Gate families -- historical arithmetic, preserved exactly.
// ---------------------------------------------------------------------------

// Gate 1: one raw-policy rollout stream per (state, replicate).
inline uint64_t Gate1RolloutSeed(uint64_t state_index, uint64_t replicate) {
  return kGate1RolloutBase + kStateBlockWidth * state_index + replicate;
}

// Gate 2 paired raw-vs-search rollouts. The raw controller rollout and the
// search controller rollout for a given replicate receive this same seed on
// purpose: the common random number is the variance reduction behind
// mean_paired_advantage. Splitting these streams would widen the paired
// interval, so the shared value is a contract, not an oversight.
inline uint64_t Gate2PairedRolloutSeed(uint64_t state_index,
                                       uint64_t replicate) {
  return kGate2PairedBase + kStateBlockWidth * state_index + replicate;
}

// Gate 2 forced-action continuations (--choice_only_gate2). The action index
// is deliberately absent from both seeds so that every root action replays
// replicate k under identical chance outcomes and identical policy draws --
// again common random numbers, this time across actions rather than across
// arms.
inline uint64_t Gate2ForcedChanceSeed(uint64_t chance_seed_base,
                                      uint64_t state_index,
                                      uint64_t replicate) {
  return chance_seed_base + kStateBlockWidth * state_index + replicate;
}

inline uint64_t Gate2ForcedRolloutSeed(uint64_t rollout_seed_base,
                                       uint64_t state_index,
                                       uint64_t replicate) {
  return rollout_seed_base + kStateBlockWidth * state_index + replicate;
}

// Single controller-selection draw per Gate 2 state.
inline uint64_t ControllerStepSeed(uint64_t raw_policy_seed,
                                   uint64_t state_index) {
  return raw_policy_seed + kControllerStepBase +
         kStateBlockWidth * state_index;
}

// MCTS bot seed for a Gate 2 state.
inline uint64_t SearchBotSeed(uint64_t search_seed, uint64_t state_index) {
  return search_seed + kSearchBotBase + kStateBlockWidth * state_index;
}

// ---------------------------------------------------------------------------
// Diagnostic families -- domain-separated ordered tuples, no width assumption.
// ---------------------------------------------------------------------------

// Chance advance for one sampled successor replicate.
inline uint64_t SuccessorChanceSeed(uint64_t raw_policy_seed,
                                    uint64_t state_index,
                                    uint64_t action_index,
                                    uint64_t replicate) {
  return dune_seed::DeriveSeed(dune_seed::kDomainFidelityGate,
                               dune_seed::kStreamFidelitySuccessorChance,
                               raw_policy_seed, state_index, action_index,
                               replicate);
}

// Raw-policy rollout for one sampled successor replicate. Distinct from the
// chance stream above: the previous code seeded both from a single value, so
// the chance draw and the rollout that consumed its result shared a stream.
inline uint64_t SuccessorRolloutSeed(uint64_t raw_policy_seed,
                                     uint64_t state_index,
                                     uint64_t action_index,
                                     uint64_t replicate) {
  return dune_seed::DeriveSeed(dune_seed::kDomainFidelityGate,
                               dune_seed::kStreamFidelitySuccessorRollout,
                               raw_policy_seed, state_index, action_index,
                               replicate);
}

// Raw-policy rollout for one sampled leaf replicate.
inline uint64_t LeafRolloutSeed(uint64_t raw_policy_seed,
                                uint64_t state_index, uint64_t leaf_index,
                                uint64_t replicate) {
  return dune_seed::DeriveSeed(dune_seed::kDomainFidelityGate,
                               dune_seed::kStreamFidelityLeafRollout,
                               raw_policy_seed, state_index, leaf_index,
                               replicate);
}

}  // namespace dune_fidelity_seeds
