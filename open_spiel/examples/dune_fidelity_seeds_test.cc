// Tests for dune_fidelity_seeds.h
//
// Three groups:
//
//   Collision  -- the diagnostic families must be injective over the full
//                 supported (state, action-or-leaf, replicate) grid, at counts
//                 that broke the previous fixed-width scheme.
//   Pairing    -- the families that intentionally share a random stream must
//                 keep sharing it, and only along the intended coordinate.
//   Lock       -- the gate families must still produce their historical seed
//                 sequences bit-for-bit, so recorded gate verdicts stay
//                 comparable and a later refactor cannot silently re-roll them.

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <map>
#include <set>
#include <string>
#include <tuple>

#include "dune_fidelity_seeds.h"

using namespace dune_fidelity_seeds;

static int test_count = 0;
static int pass_count = 0;
static bool current_failed = false;

#define TEST_BEGIN(name)                                              \
  do {                                                                \
    ++test_count;                                                     \
    current_failed = false;                                           \
    const char* test_name_ = (name);                                  \
    std::cout << "Test " << test_count << ": " << test_name_ << "... ";

#define TEST_END()                                                    \
    if (!current_failed) {                                            \
      ++pass_count;                                                   \
      std::cout << "PASSED\n";                                        \
    }                                                                 \
  } while (0)

#define CHECK_TRUE(cond)                                              \
  do {                                                                \
    if (!(cond)) {                                                    \
      if (!current_failed) std::cerr << "FAILED\n";                   \
      current_failed = true;                                          \
      std::cerr << "  Assertion failed: " #cond " (line "             \
                << __LINE__ << ")\n";                                 \
    }                                                                 \
  } while (0)

#define CHECK_EQ(a, b)                                                \
  do {                                                                \
    if ((a) != (b)) {                                                 \
      if (!current_failed) std::cerr << "FAILED\n";                   \
      current_failed = true;                                          \
      std::cerr << "  Expected equal: " #a " == " #b " (line "        \
                << __LINE__ << ")\n    lhs = " << (a)                 \
                << "\n    rhs = " << (b) << "\n";                     \
    }                                                                 \
  } while (0)

#define CHECK_NE(a, b)                                                \
  do {                                                                \
    if ((a) == (b)) {                                                 \
      if (!current_failed) std::cerr << "FAILED\n";                   \
      current_failed = true;                                          \
      std::cerr << "  Expected different: " #a " != " #b " (line "    \
                << __LINE__ << ")\n    both = " << (a) << "\n";       \
    }                                                                 \
  } while (0)

namespace {

// Coordinate ceilings for the collision sweeps. --diagnostic_rollouts defaults
// to 256, which is what made the old 100-wide replicate spacing collide; the
// action/leaf counts comfortably exceed the ~8 actions at which the old
// 1000-wide state block overflowed.
constexpr uint64_t kStates = 32;      // strict-v2 opportunity-state count
constexpr uint64_t kActions = 40;
constexpr uint64_t kLeaves = 40;
constexpr uint64_t kReplicates = 256;

struct Coord {
  uint64_t state;
  uint64_t index;
  uint64_t replicate;
  std::string family;
};

// Records seed -> first coordinate that produced it, and reports the colliding
// pair rather than just a count, so a failure is directly actionable.
class CollisionSet {
 public:
  void Insert(uint64_t seed, const Coord& c) {
    auto it = seen_.find(seed);
    if (it == seen_.end()) {
      seen_.emplace(seed, c);
      return;
    }
    if (collisions_ == 0) {
      first_a_ = it->second;
      first_b_ = c;
    }
    ++collisions_;
  }

  size_t collisions() const { return collisions_; }
  size_t distinct() const { return seen_.size(); }

  std::string Describe() const {
    if (collisions_ == 0) return "";
    return "  first collision: " + Fmt(first_a_) + " vs " + Fmt(first_b_);
  }

 private:
  static std::string Fmt(const Coord& c) {
    return c.family + "(state=" + std::to_string(c.state) +
           ", index=" + std::to_string(c.index) +
           ", replicate=" + std::to_string(c.replicate) + ")";
  }

  std::map<uint64_t, Coord> seen_;
  size_t collisions_ = 0;
  Coord first_a_{};
  Coord first_b_{};
};

}  // namespace

int main() {
  std::cout << "=== dune_fidelity_seeds tests ===\n\n";

  const uint64_t raw_policy_seed = 42;

  // =======================================================================
  // Collision tests
  // =======================================================================

  // -----------------------------------------------------------------------
  // Test 1: successor diagnostics injective over the full coordinate grid.
  // -----------------------------------------------------------------------
  TEST_BEGIN("Successor seeds distinct over 32 states x 40 actions x 256 reps") {
    CollisionSet set;
    for (uint64_t s = 0; s < kStates; ++s) {
      for (uint64_t a = 0; a < kActions; ++a) {
        for (uint64_t r = 1; r <= kReplicates; ++r) {
          set.Insert(SuccessorRolloutSeed(raw_policy_seed, s, a, r),
                     Coord{s, a, r, "successor_rollout"});
        }
      }
    }
    CHECK_EQ(set.collisions(), size_t{0});
    CHECK_EQ(set.distinct(), size_t{kStates * kActions * kReplicates});
    if (set.collisions() > 0) std::cerr << set.Describe() << "\n";
  } TEST_END();

  // -----------------------------------------------------------------------
  // Test 2: leaf diagnostics injective over the full coordinate grid.
  // -----------------------------------------------------------------------
  TEST_BEGIN("Leaf seeds distinct over 32 states x 40 leaves x 256 reps") {
    CollisionSet set;
    for (uint64_t s = 0; s < kStates; ++s) {
      for (uint64_t l = 0; l < kLeaves; ++l) {
        for (uint64_t r = 1; r <= kReplicates; ++r) {
          set.Insert(LeafRolloutSeed(raw_policy_seed, s, l, r),
                     Coord{s, l, r, "leaf_rollout"});
        }
      }
    }
    CHECK_EQ(set.collisions(), size_t{0});
    CHECK_EQ(set.distinct(), size_t{kStates * kLeaves * kReplicates});
    if (set.collisions() > 0) std::cerr << set.Describe() << "\n";
  } TEST_END();

  // -----------------------------------------------------------------------
  // Test 3: the three diagnostic streams do not alias each other. Pooling all
  // of them must stay collision-free, or a successor rollout could replay a
  // leaf rollout's stream.
  // -----------------------------------------------------------------------
  TEST_BEGIN("Successor-chance, successor-rollout and leaf streams disjoint") {
    CollisionSet set;
    for (uint64_t s = 0; s < kStates; ++s) {
      for (uint64_t i = 0; i < kActions; ++i) {
        for (uint64_t r = 1; r <= kReplicates; ++r) {
          set.Insert(SuccessorChanceSeed(raw_policy_seed, s, i, r),
                     Coord{s, i, r, "successor_chance"});
          set.Insert(SuccessorRolloutSeed(raw_policy_seed, s, i, r),
                     Coord{s, i, r, "successor_rollout"});
          set.Insert(LeafRolloutSeed(raw_policy_seed, s, i, r),
                     Coord{s, i, r, "leaf_rollout"});
        }
      }
    }
    CHECK_EQ(set.collisions(), size_t{0});
    if (set.collisions() > 0) std::cerr << set.Describe() << "\n";
  } TEST_END();

  // -----------------------------------------------------------------------
  // Test 4: the exact coordinates the previous scheme aliased.
  // Old rule: 400000 + 1000*state + 100*index + replicate.
  //   (a) replicate spacing 100 < 256 rollouts: (a=0, r=101) == (a=1, r=1)
  //   (b) state block 1000 overflowed past ~8 actions: (a=10, r=1) landed in
  //       the next state's block, colliding with (state+1, a=0, r=1).
  // -----------------------------------------------------------------------
  TEST_BEGIN("Historical successor alias pairs now produce distinct seeds") {
    // (a) replicate bleeding into the next action's sub-block
    CHECK_EQ(400000 + 1000 * 5 + 100 * 0 + 101, 400000 + 1000 * 5 + 100 * 1 + 1);
    CHECK_NE(SuccessorRolloutSeed(raw_policy_seed, 5, 0, 101),
             SuccessorRolloutSeed(raw_policy_seed, 5, 1, 1));

    // (b) action index overflowing the state block
    CHECK_EQ(400000 + 1000 * 5 + 100 * 10 + 1, 400000 + 1000 * 6 + 100 * 0 + 1);
    CHECK_NE(SuccessorRolloutSeed(raw_policy_seed, 5, 10, 1),
             SuccessorRolloutSeed(raw_policy_seed, 6, 0, 1));

    // Same two aliases in the leaf family (old base 500000).
    CHECK_NE(LeafRolloutSeed(raw_policy_seed, 5, 0, 101),
             LeafRolloutSeed(raw_policy_seed, 5, 1, 1));
    CHECK_NE(LeafRolloutSeed(raw_policy_seed, 5, 10, 1),
             LeafRolloutSeed(raw_policy_seed, 6, 0, 1));
  } TEST_END();

  // -----------------------------------------------------------------------
  // Test 5: diagnostics respond to --raw_policy_seed. The old 400000/500000
  // blocks ignored every seed flag, so re-running with a different seed
  // reproduced identical diagnostic rollouts.
  // -----------------------------------------------------------------------
  TEST_BEGIN("Diagnostic seeds vary with raw_policy_seed") {
    CHECK_NE(SuccessorRolloutSeed(42, 3, 2, 7), SuccessorRolloutSeed(43, 3, 2, 7));
    CHECK_NE(LeafRolloutSeed(42, 3, 2, 7), LeafRolloutSeed(43, 3, 2, 7));
    // Determinism: same coordinates reproduce.
    CHECK_EQ(SuccessorRolloutSeed(42, 3, 2, 7), SuccessorRolloutSeed(42, 3, 2, 7));
    CHECK_EQ(LeafRolloutSeed(42, 3, 2, 7), LeafRolloutSeed(42, 3, 2, 7));
  } TEST_END();

  // =======================================================================
  // Pairing tests -- intentional common random numbers must survive.
  // =======================================================================

  // -----------------------------------------------------------------------
  // Test 6: Gate 2 raw and search arms share one stream per replicate. The
  // call site draws a single Gate2PairedRolloutSeed and hands it to both
  // rollouts, so the property to hold here is that the seed depends only on
  // (state, replicate) -- there is no arm coordinate that could split them.
  // -----------------------------------------------------------------------
  TEST_BEGIN("Gate 2 paired seed depends only on (state, replicate)") {
    for (uint64_t s = 0; s < kStates; ++s) {
      for (uint64_t k = 1; k < 64; ++k) {
        CHECK_EQ(Gate2PairedRolloutSeed(s, k), Gate2PairedRolloutSeed(s, k));
      }
    }
    // Distinct replicates must still be distinct streams.
    CHECK_NE(Gate2PairedRolloutSeed(3, 1), Gate2PairedRolloutSeed(3, 2));
    CHECK_NE(Gate2PairedRolloutSeed(3, 1), Gate2PairedRolloutSeed(4, 1));
  } TEST_END();

  // -----------------------------------------------------------------------
  // Test 7: forced-action continuations share a stream ACROSS actions. Every
  // root action must replay replicate k under identical chance outcomes, so
  // the seed must carry no action coordinate at all.
  // -----------------------------------------------------------------------
  TEST_BEGIN("Gate 2 forced-action seeds are common across actions") {
    const uint64_t chance_base = 200000;
    const uint64_t rollout_base = 100000;
    for (uint64_t s = 0; s < 8; ++s) {
      for (uint64_t k = 1; k < 33; ++k) {
        uint64_t chance_ref = Gate2ForcedChanceSeed(chance_base, s, k);
        uint64_t rollout_ref = Gate2ForcedRolloutSeed(rollout_base, s, k);
        // Whatever action index the caller is on, replicate k resolves to the
        // same pair of streams.
        for (uint64_t a = 0; a < kActions; ++a) {
          CHECK_EQ(Gate2ForcedChanceSeed(chance_base, s, k), chance_ref);
          CHECK_EQ(Gate2ForcedRolloutSeed(rollout_base, s, k), rollout_ref);
        }
      }
    }
    // Chance and rollout streams are distinct from each other at their
    // configured bases.
    CHECK_NE(Gate2ForcedChanceSeed(200000, 3, 5),
             Gate2ForcedRolloutSeed(100000, 3, 5));
  } TEST_END();

  // =======================================================================
  // Lock tests -- gate families keep their historical values.
  // =======================================================================

  // -----------------------------------------------------------------------
  // Test 8: gate seed sequences reproduce the pre-WO-18 arithmetic exactly.
  // If one of these fails, previously recorded gate verdicts are no longer
  // comparable with a fresh run.
  // -----------------------------------------------------------------------
  TEST_BEGIN("Gate families reproduce historical seed arithmetic exactly") {
    for (uint64_t s = 0; s < kStates; ++s) {
      for (uint64_t k = 1; k < static_cast<uint64_t>(kMaxRollouts); ++k) {
        CHECK_EQ(Gate1RolloutSeed(s, k), 300000 + 1000 * s + k);
        CHECK_EQ(Gate2PairedRolloutSeed(s, k), 100000 + 1000 * s + k);
        CHECK_EQ(Gate2ForcedChanceSeed(200000, s, k), 200000 + 1000 * s + k);
        CHECK_EQ(Gate2ForcedRolloutSeed(100000, s, k), 100000 + 1000 * s + k);
      }
      CHECK_EQ(ControllerStepSeed(42, s), 42 + 300000 + 1000 * s);
      CHECK_EQ(SearchBotSeed(42, s), 42 + 200000 + 1000 * s);
    }
    // Non-default bases still thread through unchanged.
    CHECK_EQ(Gate2ForcedChanceSeed(777000, 4, 9), 777000 + 1000 * 4 + 9);
    CHECK_EQ(Gate2ForcedRolloutSeed(555000, 4, 9), 555000 + 1000 * 4 + 9);
  } TEST_END();

  // -----------------------------------------------------------------------
  // Test 9: within the enforced range 1 <= replicate < kMaxRollouts, the gate
  // families stay injective in (state, replicate) -- which is exactly what the
  // binary's --rollouts bound buys. Run at the maximum supported replicate
  // count, the boundary the bound protects.
  // -----------------------------------------------------------------------
  TEST_BEGIN("Gate families injective at the maximum supported rollout count") {
    const uint64_t max_rep = static_cast<uint64_t>(kMaxRollouts) - 1;
    CollisionSet gate1;
    CollisionSet gate2;
    for (uint64_t s = 0; s < kStates; ++s) {
      for (uint64_t k = 1; k <= max_rep; ++k) {
        gate1.Insert(Gate1RolloutSeed(s, k), Coord{s, 0, k, "gate1"});
        gate2.Insert(Gate2PairedRolloutSeed(s, k), Coord{s, 0, k, "gate2_paired"});
      }
    }
    CHECK_EQ(gate1.collisions(), size_t{0});
    CHECK_EQ(gate2.collisions(), size_t{0});
    CHECK_EQ(gate1.distinct(), size_t{kStates * max_rep});
    CHECK_EQ(gate2.distinct(), size_t{kStates * max_rep});
    if (gate1.collisions() > 0) std::cerr << gate1.Describe() << "\n";
    if (gate2.collisions() > 0) std::cerr << gate2.Describe() << "\n";
  } TEST_END();

  // -----------------------------------------------------------------------
  // Test 10: the bound is necessary, not decorative -- one past it, the
  // historical arithmetic really does alias the next state. This is the
  // collision --rollouts >= kMaxRollouts is rejected to prevent.
  // -----------------------------------------------------------------------
  TEST_BEGIN("Replicate == kMaxRollouts would alias the next state's block") {
    const uint64_t over = static_cast<uint64_t>(kMaxRollouts);
    CHECK_EQ(Gate1RolloutSeed(5, over), Gate1RolloutSeed(6, 0));
    CHECK_EQ(Gate2PairedRolloutSeed(5, over), Gate2PairedRolloutSeed(6, 0));
    CHECK_EQ(Gate2ForcedChanceSeed(200000, 5, over),
             Gate2ForcedChanceSeed(200000, 6, 0));
  } TEST_END();

  // -----------------------------------------------------------------------
  // Summary
  // -----------------------------------------------------------------------
  std::cout << "\n=== " << pass_count << "/" << test_count
            << " tests passed ===\n";

  return (pass_count == test_count) ? 0 : 1;
}
