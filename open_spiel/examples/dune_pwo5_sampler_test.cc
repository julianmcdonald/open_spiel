// PWO-5 gate 3: the auxiliary sampler's EXACT algorithm.
//
// docs/PWO5_PILOT_REGISTRATION.md section 8.6 pins the draw bit for bit,
// because "uniformly without replacement admits many non-equivalent
// implementations and they do not agree bit for bit", and because the
// registration requires an update-1 sampler digest in every arm's manifest --
// so two arms sharing a triplet seed can be checked to have drawn identically.
// "Reproducible in principle" is not enough for that; it has to be pinned.
//
// What is tested here:
//   1. the partial Fisher-Yates is EXACTLY as written, verified against an
//      independent transcription of the registered pseudocode;
//   2. `rng() % (N - i)` is what is used -- NOT std::uniform_int_distribution,
//      whose generator-to-range mapping is implementation-defined;
//   3. draw ORDER is preserved (the batch partition reads it);
//   4. the draw is a permutation prefix: no duplicates, all drawn elements come
//      from the input;
//   5. same seed -> same draw; different seed -> different draw.

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <map>
#include <random>
#include <set>
#include <string>
#include <vector>

#include "dune_pwo5_aux.h"
#include "dune_seed_utils.h"

namespace {

int failures = 0;

void Check(bool ok, const std::string& what) {
  if (ok) {
    std::cout << "  PASS  " << what << "\n";
  } else {
    std::cout << "  FAIL  " << what << "\n";
    ++failures;
  }
}

// An INDEPENDENT transcription of the registered pseudocode, section 8.6:
//
//   SampleWithoutReplacement(A[0..N-1], k, rng):
//       for i in 0 .. k-1:
//           j = i + (rng() % (N - i))
//           swap(A[i], A[j])
//       return A[0 .. k-1]
std::vector<int32_t> RegisteredReference(std::vector<int32_t> a, int k,
                                         std::mt19937_64* rng) {
  const int n = static_cast<int>(a.size());
  for (int i = 0; i < k; ++i) {
    const int j = i + static_cast<int>((*rng)() % static_cast<uint64_t>(n - i));
    std::swap(a[i], a[j]);
  }
  a.resize(k);
  return a;
}

std::vector<int32_t> Iota(int n, int base) {
  std::vector<int32_t> v;
  for (int i = 0; i < n; ++i) v.push_back(base + i);
  return v;
}

void TestMatchesTheRegisteredPseudocode() {
  std::cout << "[1] the implementation IS the registered pseudocode\n";
  // The registered shape: 340 training games, draw 64.
  const std::vector<int32_t> games = Iota(340, 1000);
  int mismatches = 0;
  for (int update = 1; update <= 300; ++update) {
    const uint64_t seed = dune_seed::DeriveSeed(
        20260801ULL, dune_seed::kDomainTrain, update, 0,
        dune_seed::kStreamAuxSampling);
    std::mt19937_64 r1(seed), r2(seed);
    const auto impl =
        open_spiel::pwo5::SampleWithoutReplacement(games, 64, &r1);
    const auto ref = RegisteredReference(games, 64, &r2);
    if (impl != ref) ++mismatches;
    if (r1 != r2) ++mismatches;  // same number of generator draws
  }
  Check(mismatches == 0,
        "identical draw AND identical RNG consumption over all 300 updates");
}

void TestNotUniformIntDistribution() {
  std::cout << "[2] std::uniform_int_distribution is NOT used\n";
  // If the implementation used uniform_int_distribution, its draws would differ
  // from `rng() % span` on this libstdc++ for at least some (seed, span). This
  // demonstrates the two are genuinely different mappings -- which is exactly
  // why the registration forbids the distribution: the difference is
  // implementation-defined and would not survive a toolchain change.
  int differing = 0;
  for (uint64_t seed = 1; seed <= 200; ++seed) {
    std::mt19937_64 a(seed), b(seed);
    const int span = 340;
    const int via_mod = static_cast<int>(a() % static_cast<uint64_t>(span));
    std::uniform_int_distribution<int> d(0, span - 1);
    const int via_dist = d(b);
    if (via_mod != via_dist) ++differing;
  }
  Check(differing > 0,
        "the two mappings differ on " + std::to_string(differing) +
            "/200 seeds -- so the choice is observable and had to be pinned");

  // And the implementation follows the modulo mapping.
  std::mt19937_64 r(12345), ref(12345);
  const auto v = open_spiel::pwo5::SampleWithoutReplacement(Iota(340, 0), 1, &r);
  const int expected = static_cast<int>(ref() % 340ULL);
  Check(v.size() == 1 && v[0] == expected,
        "the first drawn element is A[rng() % N], the modulo mapping");
}

void TestDrawOrderAndDistinctness() {
  std::cout << "[3] draw order is preserved and the draw is distinct\n";
  const std::vector<int32_t> games = Iota(340, 1000);
  std::mt19937_64 r(20260801);
  const auto drawn = open_spiel::pwo5::SampleWithoutReplacement(games, 64, &r);

  Check(drawn.size() == 64, "exactly 64 drawn");
  std::set<int32_t> uniq(drawn.begin(), drawn.end());
  Check(uniq.size() == 64, "all 64 are DISTINCT (without replacement)");

  bool all_from_input = true;
  const std::set<int32_t> input(games.begin(), games.end());
  for (int32_t g : drawn) {
    if (input.count(g) == 0) all_from_input = false;
  }
  Check(all_from_input, "every drawn element came from the input array");

  // Draw order is significant: the batch partition is [0..31] and [32..63], so
  // a sorted result would silently change which games share a batch.
  std::vector<int32_t> sorted = drawn;
  std::sort(sorted.begin(), sorted.end());
  Check(drawn != sorted,
        "the result is in DRAW order, not sorted (the partition reads it)");
}

void TestSeedSensitivity() {
  std::cout << "[4] seed determinism and separation\n";
  const std::vector<int32_t> games = Iota(340, 1000);

  // Same (master, update) -> same draw. This is what lets a T arm and its
  // matched H arm be checked to have drawn identically from their manifests.
  const uint64_t s1 = dune_seed::DeriveSeed(20260801ULL, dune_seed::kDomainTrain,
                                            1, 0, dune_seed::kStreamAuxSampling);
  std::mt19937_64 a(s1), b(s1);
  Check(open_spiel::pwo5::SampleWithoutReplacement(games, 64, &a) ==
            open_spiel::pwo5::SampleWithoutReplacement(games, 64, &b),
        "same derived seed -> identical draw");

  // Different triplet -> different draw. The sampler digest is matched WITHIN a
  // triplet and expected to DIFFER between triplets.
  const uint64_t s2 = dune_seed::DeriveSeed(20260802ULL, dune_seed::kDomainTrain,
                                            1, 0, dune_seed::kStreamAuxSampling);
  std::mt19937_64 c(s1), d(s2);
  Check(open_spiel::pwo5::SampleWithoutReplacement(games, 64, &c) !=
            open_spiel::pwo5::SampleWithoutReplacement(games, 64, &d),
        "triplet 1 and triplet 2 draw differently");

  // Different update -> different draw.
  const uint64_t s3 = dune_seed::DeriveSeed(20260801ULL, dune_seed::kDomainTrain,
                                            2, 0, dune_seed::kStreamAuxSampling);
  std::mt19937_64 e(s1), f(s3);
  Check(open_spiel::pwo5::SampleWithoutReplacement(games, 64, &e) !=
            open_spiel::pwo5::SampleWithoutReplacement(games, 64, &f),
        "update 1 and update 2 draw differently");

  // The auxiliary stream must not collide with the distillation stream.
  Check(dune_seed::kStreamAuxSampling != dune_seed::kStreamSearchSampling,
        "kStreamAuxSampling != kStreamSearchSampling");
  const uint64_t aux = dune_seed::DeriveSeed(20260801ULL, dune_seed::kDomainTrain,
                                             7, 0, dune_seed::kStreamAuxSampling);
  const uint64_t dist = dune_seed::DeriveSeed(
      20260801ULL, dune_seed::kDomainTrain, 7, 0,
      dune_seed::kStreamSearchSampling);
  Check(aux != dist,
        "the two samplers derive different seeds at the same (master, update)");
}

void TestUniformityIsAcceptable() {
  std::cout << "[5] the accepted modulo bias is negligible at N <= 340\n";
  // The registration accepts a modulo bias bounded by (N-i)/2^64 < 2e-17,
  // buying exact reproducibility. This checks the draw is not grossly biased --
  // i.e. that the accepted bias really is the only one.
  const int kGames = 340;
  const int kDraw = 64;
  const int kTrials = 20000;
  std::vector<int64_t> counts(kGames, 0);
  const std::vector<int32_t> games = Iota(kGames, 0);
  for (int t = 0; t < kTrials; ++t) {
    std::mt19937_64 r(0x9E3779B97F4A7C15ULL ^ static_cast<uint64_t>(t));
    for (int32_t g : open_spiel::pwo5::SampleWithoutReplacement(games, kDraw, &r)) {
      ++counts[g];
    }
  }
  const double expected =
      static_cast<double>(kTrials) * kDraw / static_cast<double>(kGames);
  double worst = 0.0;
  for (int64_t c : counts) {
    worst = std::max(worst, std::abs(static_cast<double>(c) - expected) / expected);
  }
  // Sampling noise alone at ~3765 expected counts is ~1.6% for 1 sigma, so a
  // 10% band is loose enough not to be flaky and tight enough to catch a real
  // bias (e.g. an off-by-one that never draws the last element).
  Check(worst < 0.10,
        "every game's inclusion rate is within 10% of uniform (worst " +
            std::to_string(worst * 100.0) + "%)");

  bool all_reachable = true;
  for (int64_t c : counts) {
    if (c == 0) all_reachable = false;
  }
  Check(all_reachable, "every game is reachable -- no off-by-one at the ends");
}

}  // namespace

int main(int argc, char** argv) {
  std::cout << "dune_pwo5_sampler_test -- PWO-5 gate 3, section 8.6\n\n";
  TestMatchesTheRegisteredPseudocode();
  TestNotUniformIntDistribution();
  TestDrawOrderAndDistinctness();
  TestSeedSensitivity();
  TestUniformityIsAcceptable();
  std::cout << "\n";
  if (failures == 0) {
    std::cout << "ALL PASS\n";
    return 0;
  }
  std::cout << failures << " FAILED\n";
  return 1;
}
