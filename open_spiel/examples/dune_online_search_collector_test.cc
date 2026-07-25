// Unit tests for the deterministic/pure logic of the 18B online collector.
// (CollectUpdate's search wiring is covered separately once implemented.)

#include "open_spiel/examples/dune_online_search_collector.h"

#include <algorithm>
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

// WO-20, half 1 of 2 (the predicate): the SAME search accepts against the raw
// network prior and rejects against the post-noise tree prior, so which vector
// the rule is handed is a real decision, not bookkeeping. The teacher that froze
// the 0.50 threshold pins dirichlet_epsilon=0 (dune_search_teacher.cc:465), so
// only the raw column reproduces the contract the threshold was calibrated on.
void TestAcceptancePriorSourceMatters() {
  std::cout << "Running TestAcceptancePriorSourceMatters...\n";
  OnlineSearchCollector col(DefaultConfig(), "hash");

  // Same root as TestAcceptance's accept case: 3 actions with >= 2 visits
  // carrying 0.60 of the raw prior.
  const std::vector<int> visits = {5, 3, 2, 1, 0};
  const std::vector<double> raw = {0.30, 0.20, 0.10, 0.25, 0.15};

  // The tree prior the search ran with at a noised root: 0.75 * raw + 0.25 *
  // Dirichlet, here drawn entirely on the two barely-visited actions (a legal
  // draw, and the worst case for the covered actions). Noise costs the covered
  // set a flat 25% of its mass: 0.60 -> 0.45.
  const std::vector<double> noise = {0.0, 0.0, 0.0, 0.5, 0.5};
  std::vector<double> tree(raw.size());
  double tree_sum = 0.0;
  for (size_t i = 0; i < raw.size(); ++i) {
    tree[i] = 0.75 * raw[i] + 0.25 * noise[i];
    tree_sum += tree[i];
  }
  assert(std::abs(tree_sum - 1.0) < 1e-12);  // still a distribution

  int covered_raw = 0, covered_tree = 0;
  double mass_raw = 0.0, mass_tree = 0.0;
  const bool ok_raw =
      col.AcceptSearch(visits, raw, /*legal_count=*/5, &covered_raw, &mass_raw);
  const bool ok_tree =
      col.AcceptSearch(visits, tree, /*legal_count=*/5, &covered_tree, &mass_tree);

  // Coverage (the visit half) is identical — only the measured mass moves.
  assert(covered_raw == 3 && covered_tree == 3);
  assert(std::abs(mass_raw - 0.60) < 1e-9);
  assert(std::abs(mass_tree - 0.45) < 1e-9);
  assert(ok_raw);    // frozen contract: 0.60 >= 0.50
  assert(!ok_tree);  // post-noise view: 0.45 < 0.50 -- the same search, rejected

  // The names are the ones that get logged/persisted; keep them stable.
  assert(std::string(AcceptancePriorSourceName(
             AcceptancePriorSource::kRawNetworkPrior)) == "raw_network_prior");
  assert(std::string(AcceptancePriorSourceName(
             AcceptancePriorSource::kTreePrior)) == "tree_prior");
  std::cout << "TestAcceptancePriorSourceMatters Passed!\n\n";
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

// KataGo visit-target pruning (Item 2d). Hand-computed expected outcomes.
void TestPruneForcedPlayouts() {
  std::cout << "Running TestPruneForcedPlayouts...\n";

  // n <= 1 is an identity.
  assert((PruneForcedPlayouts({7}, {1.0}, {0.2}, 7, 0.3, 2.0) == std::vector<int>{7}));

  // A: c* (index 0) is never pruned; two weak, forced children collapse to a
  // single visit and are then dropped. Verified by hand against the formula:
  //   n_forced=floor(sqrt(k*P*N)); remove while PUCT(c,v-1) < PUCT(c*); drop v==1.
  {
    std::vector<int> pruned = PruneForcedPlayouts(
        /*visits=*/{40, 5, 3}, /*priors=*/{0.6, 0.3, 0.1},
        /*q=*/{0.5, -0.5, -0.8}, /*N=*/48, /*puct_c=*/0.3, /*k=*/2.0);
    assert((pruned == std::vector<int>{40, 0, 0}));
  }

  // B: a genuinely competitive runner-up (high Q, PUCT already >= c*'s at full
  // visits) is protected from pruning; only the weak child is dropped.
  {
    std::vector<int> pruned = PruneForcedPlayouts(
        /*visits=*/{25, 20, 3}, /*priors=*/{0.4, 0.4, 0.2},
        /*q=*/{0.30, 0.40, -0.5}, /*N=*/48, /*puct_c=*/0.3, /*k=*/2.0);
    assert((pruned == std::vector<int>{25, 20, 0}));
  }

  std::cout << "TestPruneForcedPlayouts Passed!\n\n";
}

// CE-target sharpening (Phase 18C consolidation). Hand-computed + invariants.
void TestTargetSharpen() {
  std::cout << "Running TestTargetSharpen...\n";

  auto entropy = [](const std::vector<double>& v) {
    double h = 0.0;
    for (double p : v) if (p > 0.0) h -= p * std::log(p);
    return h;
  };
  auto argmax = [](const std::vector<double>& v) {
    size_t m = 0;
    for (size_t i = 1; i < v.size(); ++i) if (v[i] > v[m]) m = i;
    return m;
  };

  // alpha == 1.0 is inert: returns the input byte-for-byte (identical to before).
  {
    std::vector<double> t = {0.5, 0.3, 0.2};
    assert(SharpenVisitTarget(t, 1.0) == t);  // exact equality, not approximate
  }

  // alpha == 2.0 on a known visit vector 5:3:2 -> target [0.5, 0.3, 0.2]:
  //   squares [0.25, 0.09, 0.04], sum 0.38 -> [0.25/0.38, 0.09/0.38, 0.04/0.38].
  {
    std::vector<double> out = SharpenVisitTarget({0.5, 0.3, 0.2}, 2.0);
    assert(out.size() == 3);
    assert(std::abs(out[0] - 0.25 / 0.38) < 1e-12);
    assert(std::abs(out[1] - 0.09 / 0.38) < 1e-12);
    assert(std::abs(out[2] - 0.04 / 0.38) < 1e-12);
    assert(std::abs((out[0] + out[1] + out[2]) - 1.0) < 1e-12);  // still a distribution
  }

  // alpha == 2.0 preserves argmax and STRICTLY reduces entropy (non-uniform input).
  {
    std::vector<double> t = {0.5, 0.3, 0.2};
    std::vector<double> s = SharpenVisitTarget(t, 2.0);
    assert(argmax(s) == argmax(t));
    assert(entropy(s) < entropy(t) - 1e-9);
  }

  std::cout << "TestTargetSharpen Passed!\n\n";
}

// Integration smoke: the collector runs end-to-end with the exploration package
// (noise + forced playouts + FPU=0 + pruning) enabled, emits well-formed pruned
// CE targets, and stays deterministic.
void TestCollectUpdateExplorationPackage() {
  std::cout << "Running TestCollectUpdateExplorationPackage...\n";
  auto game = LoadGame("dune_imperium");
  auto eval = std::make_shared<UniformMockEvaluator>(game->NumPlayers());

  OnlineSearchConfig c = FastCollectConfig();
  c.dirichlet_epsilon = 0.25;        // activate the package
  c.dirichlet_alpha_total = 10.83;
  c.forced_playouts_k = 2.0;
  c.root_noise_fpu_zero = true;

  OnlineSearchCollector col1(c, "ckpt");
  std::vector<SearchTrainingExample> ex1;
  OnlineSearchCollectionStats s1;
  col1.CollectUpdate(1, game, eval, &ex1, &s1);

  // Every emitted CE target is a valid distribution over its legal actions.
  for (const auto& e : ex1) {
    assert(e.normalized_visits.size() == e.legal_actions.size());
    double sum = 0.0;
    for (double v : e.normalized_visits) {
      assert(v >= 0.0);
      sum += v;
    }
    assert(std::abs(sum - 1.0) < 1e-9);
  }

  // Determinism: an identical config/seed reproduces the run bit-for-bit.
  OnlineSearchCollector col2(c, "ckpt");
  std::vector<SearchTrainingExample> ex2;
  OnlineSearchCollectionStats s2;
  col2.CollectUpdate(1, game, eval, &ex2, &s2);
  assert(RunSignature(ex1, s1) == RunSignature(ex2, s2));

  // Wiring: target_sharpen_exponent=2.0 flows config -> emission. The executed
  // action samples the RAW distribution (independent of alpha), so the run emits
  // the SAME examples 1:1 with only the CE targets peaked. Each target keeps its
  // argmax; every multi-support target has strictly lower entropy.
  OnlineSearchConfig cs = c;
  cs.target_sharpen_exponent = 2.0;
  OnlineSearchCollector col3(cs, "ckpt");
  std::vector<SearchTrainingExample> ex3;
  OnlineSearchCollectionStats s3;
  col3.CollectUpdate(1, game, eval, &ex3, &s3);
  assert(ex3.size() == ex1.size());
  auto ent = [](const std::vector<double>& v) {
    double h = 0.0;
    for (double p : v) if (p > 0.0) h -= p * std::log(p);
    return h;
  };
  auto amax = [](const std::vector<double>& v) {
    size_t m = 0;
    for (size_t i = 1; i < v.size(); ++i) if (v[i] > v[m]) m = i;
    return m;
  };
  bool saw_change = false;
  for (size_t i = 0; i < ex1.size(); ++i) {
    const auto& a = ex1[i].normalized_visits;   // alpha=1.0
    const auto& b = ex3[i].normalized_visits;   // alpha=2.0
    assert(b.size() == a.size());
    assert(amax(b) == amax(a));                 // monotone sharpening preserves argmax
    assert(ent(b) <= ent(a) + 1e-9);            // sharpening never raises entropy
    // Strict reduction only for NON-uniform support (a uniform target such as
    // [0.5,0.5] is a power-map fixed point -- its entropy is unchanged).
    if (ent(a) - ent(b) > 1e-9) saw_change = true;
  }
  assert(saw_change);  // >= 1 non-uniform target is visibly peaked (knob is wired)

  std::cout << "  accepted=" << s1.accepted_targets
            << " emitted=" << ex1.size() << " (package on)\n";
  std::cout << "TestCollectUpdateExplorationPackage Passed!\n\n";
}

// WO-20, half 2 of 2 (the wiring): which prior CollectUpdate hands the predicate.
// The mock prior is uniform over the legal actions, so the raw covered mass is
// exactly num_covered/legal_count at every root — an identity the Dirichlet
// mixture cannot satisfy. That makes the recorded mass a direct read-out of the
// source actually used.
//
// Noise OFF: the two sources are the same vector, so the runs agree exactly.
// Noise ON: the default (raw) run still satisfies the identity, the kTreePrior
// run does not — the acceptance decisions are held identical (thresholds
// relaxed) so the ONLY difference is the quantity measured.
void TestCollectUpdateAcceptancePriorSource() {
  std::cout << "Running TestCollectUpdateAcceptancePriorSource...\n";
  auto game = LoadGame("dune_imperium");
  auto eval = std::make_shared<UniformMockEvaluator>(game->NumPlayers());

  OnlineSearchConfig base = FastCollectConfig();
  base.search_probability = 1.0;    // search every target root
  base.min_coverage = 1;            // acceptance turns on the VISIT half only,
  base.min_visits_per_action = 1;   // so both sources accept the same roots and
  base.min_prior_mass = 0.0;        // the recorded mass is the isolated variable

  auto run = [&](const OnlineSearchConfig& c,
                 std::vector<SearchTrainingExample>* ex,
                 OnlineSearchCollectionStats* s) {
    OnlineSearchCollector(c, "ckpt").CollectUpdate(1, game, eval, ex, s);
  };
  // Deviation of a run's recorded masses from the uniform-prior identity.
  auto max_identity_error = [](const std::vector<SearchTrainingExample>& ex) {
    double worst = 0.0;
    for (const auto& e : ex) {
      assert(!e.legal_actions.empty());
      const double expect = static_cast<double>(e.num_covered_actions) /
                            static_cast<double>(e.legal_actions.size());
      worst = std::max(worst, std::abs(e.covered_prior_mass - expect));
    }
    return worst;
  };

  // --- Noise OFF: the tree prior IS the network prior; the source is inert. ---
  OnlineSearchConfig off_raw = base;  // dirichlet_epsilon defaults to 0.0
  OnlineSearchConfig off_tree = off_raw;
  off_tree.acceptance_prior_source = AcceptancePriorSource::kTreePrior;
  std::vector<SearchTrainingExample> ex_off_raw, ex_off_tree;
  OnlineSearchCollectionStats s_off_raw, s_off_tree;
  run(off_raw, &ex_off_raw, &s_off_raw);
  run(off_tree, &ex_off_tree, &s_off_tree);
  assert(!ex_off_raw.empty());
  assert(RunSignature(ex_off_raw, s_off_raw) ==
         RunSignature(ex_off_tree, s_off_tree));
  assert(ex_off_raw.size() == ex_off_tree.size());
  for (size_t i = 0; i < ex_off_raw.size(); ++i) {
    assert(std::abs(ex_off_raw[i].covered_prior_mass -
                    ex_off_tree[i].covered_prior_mass) < 1e-12);
  }
  assert(max_identity_error(ex_off_raw) < 1e-9);   // uniform prior, either way
  assert(max_identity_error(ex_off_tree) < 1e-9);
  assert(std::abs(s_off_raw.mean_covered_prior_mass -
                  s_off_tree.mean_covered_prior_mass) < 1e-12);

  // --- Noise ON: the sources come apart. -------------------------------------
  OnlineSearchConfig on_raw = base;
  on_raw.dirichlet_epsilon = 0.25;   // pilot exploration package
  on_raw.dirichlet_alpha_total = 10.83;
  on_raw.forced_playouts_k = 2.0;
  on_raw.root_noise_fpu_zero = true;
  OnlineSearchConfig on_tree = on_raw;
  on_tree.acceptance_prior_source = AcceptancePriorSource::kTreePrior;
  std::vector<SearchTrainingExample> ex_on_raw, ex_on_tree;
  OnlineSearchCollectionStats s_on_raw, s_on_tree;
  run(on_raw, &ex_on_raw, &s_on_raw);
  run(on_tree, &ex_on_tree, &s_on_tree);
  assert(!ex_on_raw.empty());

  // Same accepted set, same targets, same executed actions: only the measured
  // mass may differ (RunSignature covers everything except that mass).
  assert(RunSignature(ex_on_raw, s_on_raw) == RunSignature(ex_on_tree, s_on_tree));

  // The default keeps the frozen, noise-independent coverage contract...
  const double raw_err = max_identity_error(ex_on_raw);
  assert(raw_err < 1e-9);
  // ...while the tree prior measures the flattened distribution instead. The
  // Dirichlet draw moves covered mass in EITHER direction (here the tree mean
  // comes out higher), so assert on the magnitude of the deviation, not a sign.
  const double tree_err = max_identity_error(ex_on_tree);
  std::cout << "  identity error: raw=" << raw_err << " tree=" << tree_err
            << " (noise on)\n";
  assert(tree_err > 1e-6);

  // The run reports which source it used — no inferring it from the noise knob.
  assert(s_on_raw.acceptance_prior_source ==
         AcceptancePriorSource::kRawNetworkPrior);
  assert(s_on_tree.acceptance_prior_source == AcceptancePriorSource::kTreePrior);
  assert(s_off_raw.acceptance_prior_source ==
         AcceptancePriorSource::kRawNetworkPrior);

  // Regression pin for the finding itself: with noise ON, the pre-WO-20
  // behavior (tree prior) reports a DIFFERENT mean covered mass than the rule
  // was frozen against, so the two arms were never comparable.
  std::cout << "  mean covered mass: raw=" << s_on_raw.mean_covered_prior_mass
            << " tree=" << s_on_tree.mean_covered_prior_mass << "\n";
  assert(std::abs(s_on_raw.mean_covered_prior_mass -
                  s_on_tree.mean_covered_prior_mass) > 1e-6);
  std::cout << "TestCollectUpdateAcceptancePriorSource Passed!\n\n";
}

// Item 1 (surface extension) + Item 4 (per-role telemetry): the collector now
// searches primary, continuation, AND purchase roots (the strategic-state gate
// does NOT bypass continuation/purchase), and the per-role counters + KL are
// populated and partition the aggregates exactly.
void TestCollectUpdateSurfaceAndTelemetry() {
  std::cout << "Running TestCollectUpdateSurfaceAndTelemetry...\n";
  auto game = LoadGame("dune_imperium");
  auto eval = std::make_shared<UniformMockEvaluator>(game->NumPlayers());
  OnlineSearchConfig c = FastCollectConfig();
  c.search_probability = 1.0;  // search every target root
  OnlineSearchCollector col(c, "ckpt");
  std::vector<SearchTrainingExample> ex;
  OnlineSearchCollectionStats s;
  col.CollectUpdate(3, game, eval, &ex, &s);

  // All three surface roles are actually searched (empirically confirms the
  // strategic-state gate covers continuation + purchase, not just primary).
  assert(s.by_role[0].searches > 0);  // primary
  assert(s.by_role[1].searches > 0);  // continuation
  assert(s.by_role[2].searches > 0);  // purchase

  // Per-role counters partition the aggregates exactly.
  int sum_searches = s.by_role[0].searches + s.by_role[1].searches + s.by_role[2].searches;
  int sum_accept = s.by_role[0].accepted + s.by_role[1].accepted + s.by_role[2].accepted;
  assert(sum_searches == s.searches_selected);
  assert(sum_accept == s.accepted_targets);

  // KL non-negative + finite; override in [0, searches]; accepted/searches sane.
  for (int r = 0; r < 3; ++r) {
    assert(s.by_role[r].sum_kl >= -1e-9);
    assert(std::isfinite(s.by_role[r].sum_kl));
    assert(s.by_role[r].prior_argmax_overrides >= 0);
    assert(s.by_role[r].prior_argmax_overrides <= s.by_role[r].searches);
    assert(s.by_role[r].accepted <= s.by_role[r].searches);
    assert(s.by_role[r].searches <= s.by_role[r].roots_seen);
  }
  std::cout << "  searched primary/cont/purchase = " << s.by_role[0].searches << "/"
            << s.by_role[1].searches << "/" << s.by_role[2].searches
            << "  accepted = " << s.by_role[0].accepted << "/" << s.by_role[1].accepted
            << "/" << s.by_role[2].accepted << "\n";
  std::cout << "TestCollectUpdateSurfaceAndTelemetry Passed!\n\n";
}

// --- Swordmaster endowment curriculum (Phase 18B follow-on to arm B-endow). ---

// fraction=1.0 selects every game; round=2 grants the searched seat a free
// Swordmaster at its first round-2 decision. Every selected game that reaches
// round 2 grants once, organic stays 0 (a fired grant suppresses the organic
// count), and search still accepts targets (the grant does not break search).
void TestCollectUpdateSwordmasterGrantAll() {
  std::cout << "Running TestCollectUpdateSwordmasterGrantAll...\n";
  auto game = LoadGame("dune_imperium");
  auto eval = std::make_shared<UniformMockEvaluator>(game->NumPlayers());

  OnlineSearchConfig c = FastCollectConfig();  // auxiliary_games = 4 (%4 guard)
  c.search_probability = 1.0;                  // exercise search at every root
  c.min_coverage = 1;                          // relaxed acceptance
  c.min_visits_per_action = 1;
  c.min_prior_mass = 0.0;
  c.swordmaster_grant_fraction = 1.0;          // every game selected
  c.swordmaster_grant_round = 2;

  OnlineSearchCollector col(c, "ckpt-sm");
  std::vector<SearchTrainingExample> out;
  OnlineSearchCollectionStats s;
  col.CollectUpdate(/*update_id=*/9, game, eval, &out, &s);

  std::cout << "  granted=" << s.swordmaster_granted_games
            << " organic=" << s.swordmaster_organic_games
            << " accepted=" << s.accepted_targets << std::endl;
  // auxiliary_games is %4-constrained by the ctor guard, so this window is 4
  // (the spec's "~3" is illustrative); at fraction 1.0 every game reaches its
  // round-2 decision and grants exactly once. Seed-deterministic: observed once,
  // then hard-coded.
  assert(s.swordmaster_granted_games == 4);
  assert(s.swordmaster_organic_games == 0);  // a fired grant suppresses organic
  assert(s.accepted_targets > 0);            // grant did not break search
  std::cout << "TestCollectUpdateSwordmasterGrantAll Passed!\n\n";
}

// fraction=0.0 is structurally inert for the GRANT path: no grant fires, and
// running with two different grant_round values but everything else identical
// yields byte-identical example streams AND an identical measured organic count
// (the round knob does nothing at fraction 0). NOTE: after the step-0 ungate the
// organic counter is measured for EVERY game, so at fraction 0 it is a real
// uniform-mock baseline that need NOT be 0 (random play sometimes buys a
// Swordmaster) — we assert it is inert to grant_round, not that it is zero.
void TestCollectUpdateSwordmasterInert() {
  std::cout << "Running TestCollectUpdateSwordmasterInert...\n";
  auto game = LoadGame("dune_imperium");
  auto eval = std::make_shared<UniformMockEvaluator>(game->NumPlayers());

  OnlineSearchConfig c = FastCollectConfig();
  c.search_probability = 0.5;  // a mix of searched/skipped roots -> real stream
  c.min_coverage = 1;
  c.min_visits_per_action = 1;
  c.min_prior_mass = 0.0;
  c.swordmaster_grant_fraction = 0.0;  // inert
  c.swordmaster_grant_round = 2;

  std::vector<SearchTrainingExample> o1;
  OnlineSearchCollectionStats s1;
  OnlineSearchCollector(c, "h").CollectUpdate(4, game, eval, &o1, &s1);
  assert(s1.swordmaster_granted_games == 0);  // fraction 0 -> no grant fires

  // Same config but a different grant_round: at fraction 0 the knob is inert, so
  // the emitted example stream (episode/decision/legal_actions/visits) and stats
  // are byte-identical.
  OnlineSearchConfig c2 = c;
  c2.swordmaster_grant_round = 7;
  std::vector<SearchTrainingExample> o2;
  OnlineSearchCollectionStats s2;
  OnlineSearchCollector(c2, "h").CollectUpdate(4, game, eval, &o2, &s2);
  assert(s2.swordmaster_granted_games == 0);
  // Organic is inert to grant_round at fraction 0 (measured, not necessarily 0).
  assert(s1.swordmaster_organic_games == s2.swordmaster_organic_games);
  assert(RunSignature(o1, s1) == RunSignature(o2, s2));  // round inert @ frac 0

  std::cout << "  frac0 granted=" << s1.swordmaster_granted_games
            << " organic=" << s1.swordmaster_organic_games
            << " (round 2 vs 7 streams identical)" << std::endl;
  std::cout << "TestCollectUpdateSwordmasterInert Passed!\n\n";
}

// fraction=0.5: the per-episode selection stream is deterministic, so two runs
// with identical config produce identical grant counts and identical example
// streams. The exact granted count is seed-deterministic (hard-coded below).
void TestCollectUpdateSwordmasterPartial() {
  std::cout << "Running TestCollectUpdateSwordmasterPartial...\n";
  auto game = LoadGame("dune_imperium");
  auto eval = std::make_shared<UniformMockEvaluator>(game->NumPlayers());

  OnlineSearchConfig c = FastCollectConfig();
  c.auxiliary_games = 8;  // %4 guard; a larger window for the 0.5 selection
  c.search_probability = 0.5;
  c.min_coverage = 1;
  c.min_visits_per_action = 1;
  c.min_prior_mass = 0.0;
  c.swordmaster_grant_fraction = 0.5;
  c.swordmaster_grant_round = 2;

  std::vector<SearchTrainingExample> o1, o2;
  OnlineSearchCollectionStats s1, s2;
  OnlineSearchCollector(c, "h").CollectUpdate(2, game, eval, &o1, &s1);
  OnlineSearchCollector(c, "h").CollectUpdate(2, game, eval, &o2, &s2);

  std::cout << "  granted=" << s1.swordmaster_granted_games
            << " organic=" << s1.swordmaster_organic_games
            << " (of " << c.auxiliary_games << " games)" << std::endl;
  assert(s1.swordmaster_granted_games == s2.swordmaster_granted_games);
  assert(s1.swordmaster_organic_games == s2.swordmaster_organic_games);
  assert(RunSignature(o1, s1) == RunSignature(o2, s2));  // deterministic stream
  // Exact count is seed-deterministic (the 0.5 selection draw over episodes
  // 0..7 selects 5): observed once, then hard-coded.
  assert(s1.swordmaster_granted_games == 5);
  // Measured organic baseline (step-0 ungate): 0 here — none of the 3 non-granted
  // games' searched seats hold a Swordmaster at terminal. Seed-deterministic.
  assert(s1.swordmaster_organic_games == 0);
  std::cout << "TestCollectUpdateSwordmasterPartial Passed!\n\n";
}

}  // namespace
}  // namespace open_spiel

int main() {
  open_spiel::TestSeatRotation();
  open_spiel::TestDeterministicBernoulliRate();
  open_spiel::TestAcceptance();
  open_spiel::TestAcceptancePriorSourceMatters();
  open_spiel::TestLossCoefWarmup();
  open_spiel::TestCollectUpdateAcceptPath();
  open_spiel::TestCollectUpdateRejectPath();
  open_spiel::TestCollectUpdateDeterminismAndSeedDomain();
  open_spiel::TestCollectUpdatePpoOnlyNoLeakage();
  open_spiel::TestPruneForcedPlayouts();
  open_spiel::TestTargetSharpen();
  open_spiel::TestCollectUpdateExplorationPackage();
  open_spiel::TestCollectUpdateAcceptancePriorSource();
  open_spiel::TestCollectUpdateSurfaceAndTelemetry();
  open_spiel::TestCollectUpdateSwordmasterGrantAll();
  open_spiel::TestCollectUpdateSwordmasterInert();
  open_spiel::TestCollectUpdateSwordmasterPartial();
  std::cout << "All Dune online search collector tests passed!\n";
  return 0;
}
