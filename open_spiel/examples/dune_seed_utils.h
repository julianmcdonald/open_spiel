// Shared seeding utility for deterministic, domain-separated RNG derivation.
//
// Provides:
//   - SplitMix64: bijective finalizer for seed mixing
//   - Combine: position-tagged sequential combiner (breaks commutativity)
//   - DeriveSeed: variadic seed derivation from an arbitrary tuple of uint64_t
//   - Domain and stream constants for canonical tuple specifications
//   - RNG construction helpers for mt19937_64, mt19937 (seed_seq), and Torch
//
// All functions are header-only (inline/constexpr) — no .cc file needed.

#pragma once

#include <cstdint>
#include <random>

#include <torch/torch.h>
#include <ATen/CPUGeneratorImpl.h>

namespace dune_seed {

// ---------------------------------------------------------------------------
// SplitMix64 finalizer (bijective on uint64_t)
// ---------------------------------------------------------------------------
inline uint64_t SplitMix64(uint64_t z) {
  z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
  z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
  return z ^ (z >> 31);
}

// ---------------------------------------------------------------------------
// Position-tagged combine. Breaks commutativity:
//   Combine(s, v, 0) ≠ Combine(s, v, 1)
//   Combine(0, a, b, p0) ≠ Combine(0, b, a, p0)
// kPositionSalt ensures position 0 ≠ identity.
// ---------------------------------------------------------------------------
constexpr uint64_t kPositionSalt = 0x517cc1b727220a95ULL;

inline uint64_t Combine(uint64_t state, uint64_t value, uint64_t position) {
  return SplitMix64((state ^ (kPositionSalt + position)) + value);
}

// ---------------------------------------------------------------------------
// Variadic: DeriveSeed(v0, v1, v2, ...)
// Each value is combined at its argument position (0, 1, 2, ...).
// ---------------------------------------------------------------------------
namespace detail {

inline uint64_t DeriveImpl(uint64_t state, uint64_t /*pos*/) {
  return state;
}

template<typename... Args>
inline uint64_t DeriveImpl(uint64_t state, uint64_t pos,
                            uint64_t next, Args... rest) {
  return DeriveImpl(Combine(state, next, pos), pos + 1, rest...);
}

}  // namespace detail

template<typename... Args>
inline uint64_t DeriveSeed(Args... args) {
  return detail::DeriveImpl(0, 0, static_cast<uint64_t>(args)...);
}

// ===========================================================================
// Domain constants
// ===========================================================================
constexpr uint64_t kDomainTrain         = 0x0001;
constexpr uint64_t kDomainEvalBaseline  = 0x0002;
constexpr uint64_t kDomainEvalDev       = 0x0003;
constexpr uint64_t kDomainEvalFinal     = 0x0004;
constexpr uint64_t kDomainSearchTeacher = 0x0005;
constexpr uint64_t kDomainFidelityGate  = 0x0006;
// Legacy dune_eval_1000. Kept distinct from the EVAL_* domains so the legacy
// evaluator and dune_population_eval never draw correlated game streams from
// the same base seed.
constexpr uint64_t kDomainEvalLegacy    = 0x0007;

// ===========================================================================
// Stream constants
// ===========================================================================
constexpr uint64_t kStreamChance          = 0x0010;
constexpr uint64_t kStreamPolicyPlayer0   = 0x0020;
constexpr uint64_t kStreamPolicyPlayer1   = 0x0021;
constexpr uint64_t kStreamPolicyPlayer2   = 0x0022;
constexpr uint64_t kStreamPolicyPlayer3   = 0x0023;
constexpr uint64_t kStreamOpponentAssign  = 0x0030;
constexpr uint64_t kStreamPPOPermutation  = 0x0040;
constexpr uint64_t kStreamSearchSampling  = 0x0050;
constexpr uint64_t kStreamMCTS            = 0x0060;
constexpr uint64_t kStreamModelInit       = 0x0070;
constexpr uint64_t kStreamBlueprint       = 0x0080;
constexpr uint64_t kStreamSearchGate      = 0x0090;

// Fidelity-gate secondary diagnostics. Each sampled successor replicate draws
// two independent streams (chance advance, then raw-policy rollout); sampled
// leaves draw one.
constexpr uint64_t kStreamFidelitySuccessorChance  = 0x00A0;
constexpr uint64_t kStreamFidelitySuccessorRollout = 0x00A1;
constexpr uint64_t kStreamFidelityLeafRollout      = 0x00A2;

// ===========================================================================
// RNG construction helpers
// ===========================================================================

// Primary: mt19937_64
inline std::mt19937_64 MakeRng64(uint64_t seed) {
  return std::mt19937_64(seed);
}

// For APIs requiring mt19937: seed_seq from both 32-bit halves.
// This ensures the full 64-bit seed influences the RNG state, avoiding
// truncation to 32 bits.
inline std::mt19937 MakeRng32(uint64_t seed) {
  uint32_t lo = static_cast<uint32_t>(seed);
  uint32_t hi = static_cast<uint32_t>(seed >> 32);
  std::seed_seq seq{lo, hi};
  return std::mt19937(seq);
}

// ===========================================================================
// Torch generator
// ===========================================================================

// Explicit local CPU Generator for randperm and similar ops.
// Move resulting indices to target device afterward.
// Do NOT use torch::manual_seed (global state) except for init_mode=random.
inline at::Generator MakeTorchCPUGenerator(uint64_t seed) {
  auto gen = at::detail::createCPUGenerator();
  gen.set_current_seed(seed);
  return gen;
}

}  // namespace dune_seed
