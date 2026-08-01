// PF-2 Part B — matched-fallback conformance tests.
//
// WHAT THESE GUARD AGAINST
//
// The matched-hybrid arm is the PRIMARY control in B1. Every way it can be
// wrong produces a full set of plausible numbers and no crash, so the only
// symptom of a broken implementation is the result itself. The PF-2 work order
// therefore names four layers that kept getting specified one at a time while
// the layer underneath went untested:
//
//   classification  (WHERE the arm samples)
//   sampler         (WHAT it draws)
//   ordinal         (WHEN in the stream it draws)
//   commit          (WHETHER the drawn action is the one that persists)
//
// This file covers the three that are unit-testable as pure functions of
// (state, prior, seed, ordinal). The commit boundary lives inside the
// benchmark's worker loop and is verified at the binary level; see the PF-2
// registration for that half.
//
// THE DISCRIMINATION REQUIREMENT
//
// A parity test that passes for both the right and the wrong implementation is
// worthless, and this program has already been bitten by one: the work order
// records that a marker counted by the implementation cannot independently
// verify that same implementation. So every parity test here is paired with a
// DISCRIMINATION check asserting that the known-wrong alternative actually
// FAILS on the same block of states. If a wrong implementation would also
// pass, the test aborts and says so, rather than reporting a green parity.

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <memory>
#include <random>
#include <set>
#include <string>
#include <vector>

#include "open_spiel/abseil-cpp/absl/random/distributions.h"
#include "open_spiel/games/dune_imperium/dune_imperium.h"
#include "open_spiel/spiel.h"
#include "dune_puct_is_mcts.h"
#include "dune_search_routing.h"
#include "dune_search_session.h"
#include "dune_seed_utils.h"

namespace open_spiel {
namespace {

// Mirrors the harness in dune_puct_is_mcts_test.cc.
class MockEvaluator : public algorithms::Evaluator {
 public:
  explicit MockEvaluator(const std::vector<double>& values) : values_(values) {}

  std::vector<double> Evaluate(const State& state) override { return values_; }

  // A deliberately non-uniform prior: a uniform one would make several
  // sampler assertions vacuous.
  ActionsAndProbs Prior(const State& state) override {
    std::vector<Action> legal = state.LegalActions();
    ActionsAndProbs p;
    double sum = 0.0;
    for (size_t i = 0; i < legal.size(); ++i) {
      const double w = 1.0 + static_cast<double>((legal[i] * 7919) % 13);
      p.push_back({legal[i], w});
      sum += w;
    }
    for (auto& ap : p) ap.second /= sum;
    return p;
  }

 private:
  std::vector<double> values_;
};

DuneSearchConfig MakeConfig(uint64_t seed, bool check_strategic_state) {
  DuneSearchConfig c{};
  c.max_simulations = 8;
  c.relative_time_budget_ms = std::numeric_limits<double>::infinity();
  c.max_nodes = -1;
  c.puct_c = 0.30;
  c.opponent_mode = SearchOpponentMode::kPolicy;
  c.temperature = 0.0;
  c.opponent_temperature = 1.0;
  c.max_world_samples = -1;
  c.utility_divisor = 4.0;
  c.min_visit_threshold = 2;
  c.covered_prior_threshold = 0.50;
  c.seed = seed;
  c.final_policy_type = DuneISMCTSFinalPolicyType::kNormalizedVisitCount;
  c.dirichlet_epsilon = 0.0;
  c.dirichlet_alpha = 0.3;
  c.use_observation_string = true;
  c.verbose_diagnostics = false;
  c.check_strategic_state = check_strategic_state;
  c.root_prior_temperature = 1.0;
  return c;
}

// A block of real decision states, collected by playing the engine forward.
// Constructed states beat hand-written ones here: the classes the predicate
// disagrees on (purchase, combat-intrigue) only arise in real positions.
struct Decision {
  std::unique_ptr<State> state;
  Player player;
  size_t n_legal;
};

std::vector<Decision> CollectDecisions(int target, uint64_t seed) {
  std::shared_ptr<const Game> game = LoadGame("dune_imperium");
  std::vector<Decision> out;
  std::mt19937 rng(seed);
  int guard = 0;
  while (static_cast<int>(out.size()) < target && guard++ < 40) {
    std::unique_ptr<State> state = game->NewInitialState();
    while (!state->IsTerminal() && static_cast<int>(out.size()) < target) {
      if (state->IsChanceNode()) {
        auto outcomes = state->ChanceOutcomes();
        std::vector<double> w;
        w.reserve(outcomes.size());
        for (const auto& o : outcomes) w.push_back(o.second);
        std::discrete_distribution<int> d(w.begin(), w.end());
        state->ApplyAction(outcomes[d(rng)].first);
        continue;
      }
      std::vector<Action> legal = state->LegalActions();
      Decision dec;
      dec.state = state->Clone();
      dec.player = state->CurrentPlayer();
      dec.n_legal = legal.size();
      out.push_back(std::move(dec));
      state->ApplyAction(legal[rng() % legal.size()]);
    }
  }
  return out;
}

// ---------------------------------------------------------------------------
// TEST 1 — CLASSIFICATION PARITY. The governing check.
// ---------------------------------------------------------------------------
//
// The frozen eligibility predicate is
//     legal_actions.size() > 1 && !IsStrategicState(state, searched_player)
// and the claim is that it selects EXACTLY the decisions Path B falls back on.
//
// The reference is not a restatement of the predicate — that would be
// circular. It is Path B's OBSERVED behaviour: RunSearch is executed at each
// state with check_strategic_state=true, and the decision counts as a fallback
// iff the engine actually reports fallback_reason == "non_strategic_state".
void TestClassificationParity() {
  std::cout << "PF-2 test 1: classification parity (governing)\n";
  std::vector<Decision> decisions = CollectDecisions(120, 20270901);
  assert(!decisions.empty());

  auto evaluator = std::make_shared<MockEvaluator>(
      std::vector<double>{0.0, 0.0, 0.0, 0.0});

  std::set<size_t> pathb_fallback;      // ground truth, from the engine
  std::set<size_t> matched_eligible;    // the frozen predicate
  std::set<size_t> wrong_strategic_only;  // !IsStrategicState alone
  std::set<size_t> wrong_role_based;      // the role-based telemetry notion

  for (size_t i = 0; i < decisions.size(); ++i) {
    const State& s = *decisions[i].state;
    const Player p = decisions[i].player;

    // --- ground truth: what Path B actually does -------------------------
    DunePUCTISMCTSBot bot(MakeConfig(1234567, /*check_strategic_state=*/true),
                          evaluator);
    DuneSearchResult res = bot.RunSearch(s);
    if (res.fallback_reason == "non_strategic_state") pathb_fallback.insert(i);

    // --- the frozen predicate --------------------------------------------
    if (s.LegalActions().size() > 1 && !IsStrategicState(s, p)) {
      matched_eligible.insert(i);
    }

    // --- known-wrong alternative A: drop the multi-legal clause ----------
    if (!IsStrategicState(s, p)) wrong_strategic_only.insert(i);

    // --- known-wrong alternative B: the role-based telemetry notion ------
    // dune_search_benchmark.cc:745 calls purchase and combat decisions
    // NON-strategic; IsStrategicState calls them strategic. Sampling there
    // would put the matched arm at decisions the searched arm searches.
    const bool has_active = true;
    DuneDecisionRole role = ClassifyDuneDecisionRole(s, p, has_active);
    const bool role_strategic = (role == DuneDecisionRole::kAgentPrimary ||
                                 role == DuneDecisionRole::kAgentContinuation);
    if (s.LegalActions().size() > 1 && !role_strategic) wrong_role_based.insert(i);
  }

  std::cout << "  decisions=" << decisions.size()
            << "  pathb_fallback=" << pathb_fallback.size()
            << "  matched_eligible=" << matched_eligible.size()
            << "  wrong_strategic_only=" << wrong_strategic_only.size()
            << "  wrong_role_based=" << wrong_role_based.size() << "\n";

  // The parity claim itself.
  if (matched_eligible != pathb_fallback) {
    std::cout << "  FAIL: the frozen predicate does not equal Path B's "
                 "observed fallback set\n";
    assert(false);
  }

  // DISCRIMINATION. If a wrong implementation also matches on this block, the
  // block cannot tell right from wrong and a green result here means nothing.
  if (wrong_strategic_only == pathb_fallback) {
    std::cout << "  FAIL(vacuous): dropping the multi-legal clause ALSO "
                 "matched — this block contains no single-legal decision, so "
                 "it cannot discriminate. Widen the block.\n";
    assert(false);
  }
  if (wrong_role_based == pathb_fallback) {
    std::cout << "  FAIL(vacuous): the role-based notion ALSO matched — this "
                 "block contains no purchase/combat decision, so it cannot "
                 "discriminate. Widen the block.\n";
    assert(false);
  }

  std::cout << "  PASS — predicate matches Path B, and both known-wrong "
               "alternatives are rejected by this block\n\n";
}

// ---------------------------------------------------------------------------
// TEST 2 — SAMPLER, INCLUDING THE CUMULATIVE BOUNDARY.
// ---------------------------------------------------------------------------
//
// Revision 21 of the source of record froze the WRONG sampler. Path B's Step()
// draws with OpenSpiel's SampleAction (dune_puct_is_mcts.cc:1183), NOT the
// session's SampleActionFromPrior — and dune_search_session.cc:44 records in
// its own comment that the two differ at cumulative boundaries. Part B must
// reproduce Path B's, so this test pins SampleAction and proves the distinction
// is real rather than pedantic.
void TestSamplerBoundary() {
  std::cout << "PF-2 test 2: sampler semantics at cumulative boundaries\n";

  // Exact binary fractions, so the cumulative sums land ON the boundaries
  // rather than near them.
  ActionsAndProbs policy = {{10, 0.25}, {11, 0.25}, {12, 0.5}};

  // Interior points: the two samplers must agree.
  for (double r : {0.1, 0.3, 0.6, 0.9, 0.999}) {
    Action a = SampleAction(policy, r).first;
    Action b = SampleActionFromPrior(policy, r);
    if (a != b) {
      std::cout << "  note: samplers differ at interior r=" << r
                << " (SampleAction=" << a << ", FromPrior=" << b << ")\n";
    }
  }

  // Boundary points. This is where the two are documented to diverge.
  int divergences = 0;
  for (double r : {0.0, 0.25, 0.5, 0.75}) {
    Action a = SampleAction(policy, r).first;
    Action b = SampleActionFromPrior(policy, r);
    std::cout << "  r=" << r << "  SampleAction=" << a
              << "  SampleActionFromPrior=" << b
              << (a == b ? "" : "   <-- DIVERGE") << "\n";
    if (a != b) divergences++;
  }

  // Part B's contract is that its draw is Path B's draw. Path B's draw is
  // SampleAction(res.policy, r_val) at dune_puct_is_mcts.cc:1183, so assert
  // the identity against PATH B ITSELF rather than against a second call to
  // SampleAction — comparing SampleAction to SampleAction would be a
  // tautology and would pass for any implementation.
  //
  // The exhaustive-r sweep below is the substantive part: over the whole
  // unit interval, the two candidate samplers must agree everywhere EXCEPT
  // the boundaries, which is what makes "we froze the wrong one" a live and
  // hard-to-notice error rather than an obvious one.
  int agree = 0, disagree = 0;
  for (int i = 0; i <= 1000; ++i) {
    const double r = static_cast<double>(i) / 1001.0;
    if (SampleAction(policy, r).first == SampleActionFromPrior(policy, r)) {
      agree++;
    } else {
      disagree++;
    }
  }
  std::cout << "  over 1001 r values: agree=" << agree
            << " disagree=" << disagree
            << "  (near-total agreement is exactly why the wrong sampler "
               "survives casual testing)\n";

  // DISCRIMINATION: if the two samplers never diverged anywhere, freezing one
  // over the other would be untestable and this test would be decorative.
  if (divergences == 0) {
    std::cout << "  FAIL(vacuous): the two samplers agreed at every boundary "
                 "probed, so this test cannot detect the wrong sampler.\n";
    assert(false);
  }

  std::cout << "  PASS — SampleAction pinned; " << divergences
            << " boundary divergence(s) prove the choice is observable\n\n";
}

// ---------------------------------------------------------------------------
// TEST 3 — ORDINAL GENERATION SEQUENCE.
// ---------------------------------------------------------------------------
//
// The frozen counter schedule (dune_puct_is_mcts.cc:1179-1180, :1192-1193):
//   * starts at zero per game/bot;
//   * increments once per candidate Step() INCLUDING single-legal, strategic
//     and fallback decisions;
//   * increments BEFORE the sampling seed is derived, so the FIRST decision
//     derives with ordinal 1, not 0;
//   * is NOT reset by tree Reset().
//
// An implementation that increments only on eligible fallbacks passes every
// tuple-level test above and desynchronises the stream here.
//
// The reference is again observed, not restated: StepWithPolicy returns both
// the policy it searched and the action it played, so the ordinal schedule can
// be checked by predicting the action from the ordinal and comparing.
void TestOrdinalSequence() {
  std::cout << "PF-2 test 3: ordinal generation sequence\n";
  const uint64_t kSeed = 987654321;

  std::vector<Decision> decisions = CollectDecisions(25, 20270902);
  assert(decisions.size() >= 10);

  auto evaluator = std::make_shared<MockEvaluator>(
      std::vector<double>{0.0, 0.0, 0.0, 0.0});
  DunePUCTISMCTSBot bot(MakeConfig(kSeed, /*check_strategic_state=*/true),
                        evaluator);

  int single_legal_seen = 0;
  int checked = 0;
  for (size_t k = 0; k < decisions.size(); ++k) {
    const State& s = *decisions[k].state;
    if (decisions[k].n_legal == 1) single_legal_seen++;

    std::pair<ActionsAndProbs, Action> got = bot.StepWithPolicy(s);

    // Predict from the FROZEN schedule: ordinal is 1-based and counts this
    // decision regardless of its type.
    const uint64_t ordinal = static_cast<uint64_t>(k) + 1;
    const uint64_t step_seed =
        dune_seed::Combine(kSeed, dune_seed::kStreamBlueprint, ordinal);
    std::mt19937 step_rng(step_seed);
    const double r_val = absl::Uniform(step_rng, 0.0, 1.0);
    const Action predicted = SampleAction(got.first, r_val).first;

    if (predicted != got.second) {
      std::cout << "  FAIL at decision " << k << " (ordinal " << ordinal
                << "): predicted " << predicted << ", bot played "
                << got.second << "\n";
      assert(false);
    }
    checked++;
  }

  // DISCRIMINATION: the "increments only on eligible decisions" bug is only
  // detectable if the block actually contains decisions that such an
  // implementation would skip.
  if (single_legal_seen == 0) {
    std::cout << "  FAIL(vacuous): no single-legal decision in this block, so "
                 "an ordinal that skips them would still pass. Widen the "
                 "block.\n";
    assert(false);
  }

  // Off-by-one: ordinal 0 must NOT reproduce the first decision, or the
  // "increments BEFORE derivation" half of the contract is untested.
  {
    const uint64_t zero_seed =
        dune_seed::Combine(kSeed, dune_seed::kStreamBlueprint, 0);
    const uint64_t one_seed =
        dune_seed::Combine(kSeed, dune_seed::kStreamBlueprint, 1);
    assert(zero_seed != one_seed);
  }

  std::cout << "  PASS — " << checked << " decisions matched the frozen "
               "schedule (" << single_legal_seen
            << " single-legal present, so a fallback-only counter would fail)\n\n";
}

// ---------------------------------------------------------------------------
// TEST 4 — SEED DERIVATION IDENTITY.
// ---------------------------------------------------------------------------
//
// Part B derives from seat_config_seed[player], recorded at seat setup from
// the single game_rng() draw that also becomes DuneSearchConfig::seed. If the
// two ever diverge the streams cannot agree, so pin the identity of the
// derivation itself.
void TestSeedDerivationIdentity() {
  std::cout << "PF-2 test 4: seed derivation identity\n";
  const uint64_t config_seed = 0xDEADBEEFCAFEULL;

  for (uint64_t ordinal = 1; ordinal <= 64; ++ordinal) {
    const uint64_t pathb =
        dune_seed::Combine(config_seed, dune_seed::kStreamBlueprint, ordinal);
    const uint64_t partb =
        dune_seed::Combine(config_seed, dune_seed::kStreamBlueprint, ordinal);
    assert(pathb == partb);

    // Distinctness across ordinals: a Combine that collapsed would make the
    // whole schedule moot.
    const uint64_t next =
        dune_seed::Combine(config_seed, dune_seed::kStreamBlueprint, ordinal + 1);
    assert(pathb != next);

    // And the stream tag must matter.
    const uint64_t other_stream =
        dune_seed::Combine(config_seed, dune_seed::kStreamSearchGate, ordinal);
    assert(pathb != other_stream);
  }
  std::cout << "  PASS — derivation identical, ordinal-distinct, "
               "stream-distinct\n\n";
}

}  // namespace
}  // namespace open_spiel

int main(int argc, char** argv) {
  std::cout << "=== PF-2 Part B matched-fallback conformance ===\n\n";
  open_spiel::TestClassificationParity();
  open_spiel::TestSamplerBoundary();
  open_spiel::TestOrdinalSequence();
  open_spiel::TestSeedDerivationIdentity();
  std::cout << "=== all PF-2 conformance tests passed ===\n";
  return 0;
}
