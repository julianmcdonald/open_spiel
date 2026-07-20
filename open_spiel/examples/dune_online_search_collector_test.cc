// Unit tests for the deterministic/pure logic of the 18B online collector.
// (CollectUpdate's search wiring is covered separately once implemented.)

#include "open_spiel/examples/dune_online_search_collector.h"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "open_spiel/algorithms/mcts.h"  // algorithms::Evaluator (mock base)
#include "open_spiel/spiel.h"

namespace open_spiel {
namespace {

// Deterministic mock inference snapshot: uniform policy over legal actions and a
// zero value for each player. Never touches torch/CUDA, so CollectUpdate tests
// run entirely on CPU.
class UniformMockEvaluator : public algorithms::Evaluator {
 public:
  explicit UniformMockEvaluator(int num_players) : num_players_(num_players) {}
  std::vector<double> Evaluate(const State& state) override {
    return std::vector<double>(num_players_, 0.0);
  }
  ActionsAndProbs Prior(const State& state) override {
    std::vector<Action> legal = state.LegalActions();
    ActionsAndProbs out;
    out.reserve(legal.size());
    const double p = legal.empty() ? 0.0 : 1.0 / static_cast<double>(legal.size());
    for (Action a : legal) out.push_back({a, p});
    return out;
  }

 private:
  const int num_players_;
};

// Small, fast frozen-guard-satisfying config for CollectUpdate tests. Keeps the
// uncapped-depth / baseline-critic / %4 / sims>reserve invariants the ctor
// enforces, but shrinks the sim budget and game count so tests stay quick.
OnlineSearchConfig FastCollectConfig() {
  OnlineSearchConfig c;
  c.auxiliary_games = 4;
  c.max_simulations = 12;             // 8 primary after an 8/… -> here reserve 4
  c.fixed_continuation_reserve = 4;   // -> 8 primary simulations
  return c;
}

// Compact, timing-independent signature of a collection run for equality checks.
std::string RunSignature(const std::vector<SearchTrainingExample>& ex,
                         const OnlineSearchCollectionStats& s) {
  auto q = [](double d) -> int64_t {
    return static_cast<int64_t>(std::llround(d * 1e6));
  };
  std::ostringstream os;
  os << "S|" << s.strategic_roots_seen << ',' << s.searches_selected << ','
     << s.accepted_targets << ',' << s.rejected_incomplete << ','
     << s.fallback_raw_policy << ',' << s.first_episode_id << ','
     << s.next_episode_id << ',' << s.inference_calls << ';';
  for (const auto& e : ex) {
    os << "E|" << e.episode_id << ',' << e.decision_id << ',' << e.player << ','
       << q(e.value_target) << ',' << (e.value_target_attached ? 1 : 0) << ','
       << e.simulations_completed << ',' << e.legal_actions.size() << '|';
    for (Action a : e.legal_actions) os << a << '.';
    os << '|';
    for (double v : e.normalized_visits) os << q(v) << '.';
    os << ';';
  }
  return os.str();
}

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

// --- CollectUpdate wiring tests (mock evaluator, real dune_imperium game). ---

// Accept path: forces a search at every strategic root with relaxed acceptance,
// then checks the emitted examples are well-formed and terminal-value-attached,
// the executed seat matches the rotation, and the cursor advances.
void TestCollectUpdateAcceptPath() {
  std::cout << "Running TestCollectUpdateAcceptPath...\n";
  std::shared_ptr<const Game> game = LoadGame("dune_imperium");
  const int np = game->NumPlayers();
  auto eval = std::make_shared<UniformMockEvaluator>(np);

  OnlineSearchConfig c = FastCollectConfig();
  c.search_probability = 1.0;        // search every strategic root
  c.min_coverage = 1;                // relaxed: any visited action -> accept
  c.min_visits_per_action = 1;
  c.min_prior_mass = 0.0;

  OnlineSearchCollector col(c, "ckpt-accept");
  std::vector<SearchTrainingExample> out;
  OnlineSearchCollectionStats stats;
  col.CollectUpdate(/*update_id=*/7, game, eval, &out, &stats);

  assert(stats.strategic_roots_seen > 0);
  assert(stats.searches_selected == stats.strategic_roots_seen);  // prob 1.0
  assert(stats.accepted_targets > 0);
  assert(out.size() == static_cast<size_t>(stats.accepted_targets));
  assert(stats.first_episode_id == 0);
  assert(stats.next_episode_id == c.auxiliary_games);
  assert(stats.mean_simulations_completed > 0.0);

  for (const auto& e : out) {
    assert(e.update_id == 7);
    assert(e.checkpoint_hash == "ckpt-accept");
    assert(e.episode_id >= 0 && e.episode_id < c.auxiliary_games);
    // Emitting seat == searched seat for that episode.
    assert(e.player ==
           OnlineSearchCollector::SearchedSeatForEpisode(e.episode_id, np));
    // Well-formed CE target aligned to legal actions and summing to 1.
    assert(!e.legal_actions.empty());
    assert(e.normalized_visits.size() == e.legal_actions.size());
    double sum = 0.0;
    for (double v : e.normalized_visits) {
      assert(v >= 0.0);
      sum += v;
    }
    assert(std::abs(sum - 1.0) < 1e-6);
    // Terminal value target attached.
    assert(e.value_target_attached);
    assert(std::isfinite(e.value_target));
    assert(!e.observation.empty());
    assert(e.simulations_completed > 0);
    assert(e.num_covered_actions >= 1);
  }
  std::cout << "  games=" << c.auxiliary_games
            << " roots=" << stats.strategic_roots_seen
            << " accepted=" << stats.accepted_targets << " obs_dim="
            << (out.empty() ? 0 : static_cast<int>(out.front().observation.size()))
            << "\n";
  std::cout << "TestCollectUpdateAcceptPath Passed!\n\n";
}

// Reject path: impossible acceptance threshold -> every search rejected, no
// examples emitted, fallback-to-raw-policy counted, cursor still advances.
void TestCollectUpdateRejectPath() {
  std::cout << "Running TestCollectUpdateRejectPath...\n";
  std::shared_ptr<const Game> game = LoadGame("dune_imperium");
  auto eval = std::make_shared<UniformMockEvaluator>(game->NumPlayers());

  OnlineSearchConfig c = FastCollectConfig();
  c.search_probability = 1.0;
  c.min_visits_per_action = 1000000;  // unreachable -> 0 covered actions

  OnlineSearchCollector col(c, "h");
  std::vector<SearchTrainingExample> out;
  OnlineSearchCollectionStats stats;
  col.CollectUpdate(/*update_id=*/1, game, eval, &out, &stats);

  assert(stats.strategic_roots_seen > 0);
  assert(stats.searches_selected == stats.strategic_roots_seen);
  assert(stats.accepted_targets == 0);
  assert(out.empty());
  assert(stats.rejected_incomplete == stats.searches_selected);
  assert(stats.fallback_raw_policy == stats.rejected_incomplete);
  assert(stats.next_episode_id == c.auxiliary_games);
  std::cout << "  rejected=" << stats.rejected_incomplete << " (all searched roots)\n";
  std::cout << "TestCollectUpdateRejectPath Passed!\n\n";
}

// Determinism (no dependence on any external/global RNG) and seed-domain
// isolation (a different domain yields a different collection stream).
void TestCollectUpdateDeterminismAndSeedDomain() {
  std::cout << "Running TestCollectUpdateDeterminismAndSeedDomain...\n";
  std::shared_ptr<const Game> game = LoadGame("dune_imperium");
  auto eval = std::make_shared<UniformMockEvaluator>(game->NumPlayers());

  OnlineSearchConfig c = FastCollectConfig();
  c.search_probability = 0.5;  // a mix of searched and skipped roots
  c.min_coverage = 1;
  c.min_visits_per_action = 1;
  c.min_prior_mass = 0.0;

  std::vector<SearchTrainingExample> o1, o2;
  OnlineSearchCollectionStats s1, s2;
  OnlineSearchCollector(c, "h").CollectUpdate(3, game, eval, &o1, &s1);
  OnlineSearchCollector(c, "h").CollectUpdate(3, game, eval, &o2, &s2);
  assert(RunSignature(o1, s1) == RunSignature(o2, s2));  // byte-identical reruns

  OnlineSearchConfig c2 = c;
  c2.auxiliary_search_seed_domain = 0xDEADBEEFULL;
  std::vector<SearchTrainingExample> o3;
  OnlineSearchCollectionStats s3;
  OnlineSearchCollector(c2, "h").CollectUpdate(3, game, eval, &o3, &s3);
  assert(RunSignature(o3, s3) != RunSignature(o1, s1));  // domains are isolated

  std::cout << "  run1 accepted=" << s1.accepted_targets
            << " altdomain accepted=" << s3.accepted_targets << "\n";
  std::cout << "TestCollectUpdateDeterminismAndSeedDomain Passed!\n\n";
}

// PPO-only (search_probability 0): strategic roots are seen but never searched,
// zero examples leak out, and the cursor advances across successive updates.
void TestCollectUpdatePpoOnlyNoLeakage() {
  std::cout << "Running TestCollectUpdatePpoOnlyNoLeakage...\n";
  std::shared_ptr<const Game> game = LoadGame("dune_imperium");
  auto eval = std::make_shared<UniformMockEvaluator>(game->NumPlayers());

  OnlineSearchConfig c = FastCollectConfig();
  c.search_probability = 0.0;  // never search

  OnlineSearchCollector col(c, "h");
  std::vector<SearchTrainingExample> out;
  OnlineSearchCollectionStats s1;
  col.CollectUpdate(5, game, eval, &out, &s1);
  assert(s1.strategic_roots_seen > 0);
  assert(s1.searches_selected == 0);
  assert(s1.accepted_targets == 0);
  assert(s1.rejected_incomplete == 0);
  assert(s1.inference_calls == 0);
  assert(out.empty());
  assert(s1.first_episode_id == 0);
  assert(s1.next_episode_id == c.auxiliary_games);

  // Second update on the same collector continues the episode cursor.
  OnlineSearchCollectionStats s2;
  col.CollectUpdate(6, game, eval, &out, &s2);
  assert(s2.first_episode_id == c.auxiliary_games);
  assert(s2.next_episode_id == 2 * c.auxiliary_games);
  assert(out.empty());
  std::cout << "TestCollectUpdatePpoOnlyNoLeakage Passed!\n\n";
}

}  // namespace
}  // namespace open_spiel

int main() {
  open_spiel::TestSeatRotation();
  open_spiel::TestDeterministicBernoulliRate();
  open_spiel::TestAcceptance();
  open_spiel::TestLossCoefWarmup();
  open_spiel::TestCollectUpdateAcceptPath();
  open_spiel::TestCollectUpdateRejectPath();
  open_spiel::TestCollectUpdateDeterminismAndSeedDomain();
  open_spiel::TestCollectUpdatePpoOnlyNoLeakage();
  std::cout << "All Dune online search collector tests passed!\n";
  return 0;
}
