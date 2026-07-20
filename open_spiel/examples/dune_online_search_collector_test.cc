// Unit tests for the deterministic/pure logic of the 18B online collector.
// (CollectUpdate's search wiring is covered separately once implemented.)

#include "open_spiel/examples/dune_online_search_collector.h"

#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

namespace open_spiel {
namespace {

OnlineSearchConfig DefaultConfig() {
  OnlineSearchConfig c;  // defaults encode baseline/uncapped + 64/8
  return c;
}

void TestSeatRotation() {
  std::cout << "Running TestSeatRotation...\n";
  for (int64_t ep = 0; ep < 12; ++ep) {
    assert(OnlineSearchCollector::SearchedSeatForEpisode(ep, 4) ==
           static_cast<Player>(ep % 4));
  }
  // Every seat is covered within any window of 4 consecutive episodes.
  for (int64_t base = 0; base < 8; ++base) {
    std::vector<bool> seen(4, false);
    for (int64_t k = 0; k < 4; ++k) {
      seen[OnlineSearchCollector::SearchedSeatForEpisode(base + k, 4)] = true;
    }
    for (int p = 0; p < 4; ++p) assert(seen[p]);
  }
  std::cout << "TestSeatRotation Passed!\n\n";
}

void TestDeterministicBernoulliRate() {
  std::cout << "Running TestDeterministicBernoulliRate...\n";
  OnlineSearchCollector col(DefaultConfig(), "hash");
  // Determinism: identical inputs -> identical decision.
  assert(col.ShouldSearchAtRoot(7, 3) == col.ShouldSearchAtRoot(7, 3));
  // Empirical rate ~= 0.25 over a large sample.
  long selected = 0, total = 0;
  for (int64_t ep = 0; ep < 200; ++ep) {
    for (int d = 0; d < 200; ++d) {
      if (col.ShouldSearchAtRoot(ep, d)) ++selected;
      ++total;
    }
  }
  const double rate = static_cast<double>(selected) / total;
  std::cout << "  empirical search rate = " << rate << " (target 0.25)\n";
  assert(std::abs(rate - 0.25) < 0.01);

  // Domain separation: a different seed domain yields a different stream.
  OnlineSearchConfig other = DefaultConfig();
  other.auxiliary_search_seed_domain = 0xABCDEF01ULL;
  OnlineSearchCollector col2(other, "hash");
  int differ = 0;
  for (int64_t ep = 0; ep < 50; ++ep) {
    for (int d = 0; d < 50; ++d) {
      if (col.ShouldSearchAtRoot(ep, d) != col2.ShouldSearchAtRoot(ep, d)) {
        ++differ;
      }
    }
  }
  assert(differ > 0);  // streams are not identical
  std::cout << "TestDeterministicBernoulliRate Passed!\n\n";
}

void TestAcceptance() {
  std::cout << "Running TestAcceptance...\n";
  OnlineSearchCollector col(DefaultConfig(), "hash");
  int covered = 0;
  double mass = 0.0;

  // 3 covered actions (>=2 visits), covered prior mass 0.6 >= 0.50 -> accept.
  {
    std::vector<int> visits = {5, 3, 2, 1, 0};
    std::vector<double> priors = {0.30, 0.20, 0.10, 0.25, 0.15};
    bool ok = col.AcceptSearch(visits, priors, /*legal_count=*/5, &covered, &mass);
    assert(covered == 3);
    assert(std::abs(mass - 0.60) < 1e-9);
    assert(ok);
  }
  // Coverage fail: only 2 actions with >=2 visits (need min(3,5)=3).
  {
    std::vector<int> visits = {5, 2, 1, 1, 0};
    std::vector<double> priors = {0.40, 0.40, 0.10, 0.05, 0.05};
    bool ok = col.AcceptSearch(visits, priors, 5, &covered, &mass);
    assert(covered == 2);
    assert(!ok);
  }
  // Mass fail: 3 covered (>=2 visits) but covered prior mass 0.30 < 0.50 -> reject.
  {
    std::vector<int> visits = {2, 2, 2, 1, 1};
    std::vector<double> priors = {0.10, 0.10, 0.10, 0.35, 0.35};
    bool ok = col.AcceptSearch(visits, priors, 5, &covered, &mass);
    assert(covered == 3);
    assert(std::abs(mass - 0.30) < 1e-9);
    assert(!ok);  // coverage met (3 >= min(3,5)) but mass 0.30 < 0.50
  }
  // Small legal set: required coverage clamps to legal_count.
  {
    std::vector<int> visits = {3, 2};
    std::vector<double> priors = {0.5, 0.5};
    bool ok = col.AcceptSearch(visits, priors, /*legal_count=*/2, &covered, &mass);
    assert(covered == 2);
    assert(ok);  // min(3,2)=2 covered, mass 1.0
  }
  std::cout << "TestAcceptance Passed!\n\n";
}

void TestLossCoefWarmup() {
  std::cout << "Running TestLossCoefWarmup...\n";
  const double target = 0.10;
  const int warmup = 25;
  assert(std::abs(SearchLossCoefForUpdate(1, target, warmup) - 0.0) < 1e-12);
  assert(std::abs(SearchLossCoefForUpdate(25, target, warmup) - target) < 1e-12);
  assert(std::abs(SearchLossCoefForUpdate(100, target, warmup) - target) < 1e-12);
  // Midpoint (update 13): 12/24 * target = 0.05.
  assert(std::abs(SearchLossCoefForUpdate(13, target, warmup) - 0.05) < 1e-12);
  // Monotonic non-decreasing across the warmup.
  double prev = -1.0;
  for (int u = 1; u <= 25; ++u) {
    double c = SearchLossCoefForUpdate(u, target, warmup);
    assert(c >= prev - 1e-12);
    prev = c;
  }
  std::cout << "TestLossCoefWarmup Passed!\n\n";
}

}  // namespace
}  // namespace open_spiel

int main() {
  open_spiel::TestSeatRotation();
  open_spiel::TestDeterministicBernoulliRate();
  open_spiel::TestAcceptance();
  open_spiel::TestLossCoefWarmup();
  std::cout << "All Dune online search collector tests passed!\n";
  return 0;
}
