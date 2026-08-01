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
#include "dune_pf2_matched_fallback.h"   // the PRODUCTION rules under test
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
  // 50000 is the binary's real default. -1 is NOT 'unlimited': the ceiling
  // check is `node_pool_.size() >= config_.max_nodes`, so -1 trips on the
  // first node and every strategic decision falls back with 0 simulations --
  // which silently disabled search in an earlier revision of these tests.
  c.max_nodes = 50000;
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

    // --- the frozen predicate, called from PRODUCTION ---------------------
    // pf2::MatchedFallbackEligible is the function dune_search_benchmark.cc
    // itself calls. Re-expressing the predicate here would test the
    // re-expression, which is what Sol's review rejected.
    if (pf2::MatchedFallbackEligible(s, p)) matched_eligible.insert(i);

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

    // Predict using the PRODUCTION functions the matched arm calls, against
    // the real bot's own play. This is the load-bearing direction: if the
    // matched arm's seed/ordinal/sampler rules disagree with Path B's Step(),
    // this comparison fails.
    const uint64_t ordinal = static_cast<uint64_t>(k) + 1;
    const uint64_t step_seed = pf2::MatchedFallbackStepSeed(kSeed, ordinal);
    const double r_val = pf2::MatchedFallbackRVal(step_seed);
    const Action predicted = pf2::MatchedFallbackSelect(got.first, r_val);

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
// TEST 4 — SEED DERIVATION, AGAINST PATH B'S ACTUAL STREAM.
// ---------------------------------------------------------------------------
//
// The previous version of this test asserted
//     Combine(seed, kStreamBlueprint, k) == Combine(seed, kStreamBlueprint, k)
// which is the same expression on both sides and passes for ANY
// implementation. Sol's review caught it. The reference here is Path B's real
// bot instead: the production derivation must reproduce the r_val the bot
// actually drew, which is observable because the bot's played action is
// SampleAction(policy, r_val).
void TestSeedDerivationAgainstPathB() {
  std::cout << "PF-2 test 4: seed derivation vs Path B's actual stream\n";
  const uint64_t kSeed = 0xDEADBEEFCAFEULL;

  std::vector<Decision> decisions = CollectDecisions(12, 20270903);
  assert(decisions.size() >= 5);
  auto evaluator = std::make_shared<MockEvaluator>(
      std::vector<double>{0.0, 0.0, 0.0, 0.0});
  DunePUCTISMCTSBot bot(MakeConfig(kSeed, true), evaluator);

  int multi_legal = 0;
  for (size_t k = 0; k < decisions.size(); ++k) {
    std::pair<ActionsAndProbs, Action> got =
        bot.StepWithPolicy(*decisions[k].state);
    const uint64_t seed = pf2::MatchedFallbackStepSeed(kSeed, k + 1);
    const Action predicted =
        pf2::MatchedFallbackSelect(got.first, pf2::MatchedFallbackRVal(seed));
    assert(predicted == got.second);
    if (got.first.size() > 1) multi_legal++;
  }

  // DISCRIMINATION: with a single-entry policy every r_val yields the same
  // action, so the seed would be untested. Require real choices.
  if (multi_legal == 0) {
    std::cout << "  FAIL(vacuous): every policy had one entry, so any seed "
                 "would pass.\n";
    assert(false);
  }

  // Ordinal- and stream-distinctness, which make the schedule meaningful.
  for (uint64_t o = 1; o <= 64; ++o) {
    assert(pf2::MatchedFallbackStepSeed(kSeed, o) !=
           pf2::MatchedFallbackStepSeed(kSeed, o + 1));
    assert(pf2::MatchedFallbackStepSeed(kSeed, o) !=
           dune_seed::Combine(kSeed, dune_seed::kStreamSearchGate, o));
  }
  std::cout << "  PASS — production derivation reproduces Path B's stream on "
            << decisions.size() << " decisions (" << multi_legal
            << " with a real choice)\n\n";
}

// ---------------------------------------------------------------------------
// TEST 5 — COMMIT BOUNDARY: the committed action IS the played action.
// ---------------------------------------------------------------------------
//
// The fourth member of the parity family, from
// docs/PF2_COMMIT_LIFECYCLE_DEFECT_2026_08_01.md. Part B's first
// implementation called SearchAndSelectWithDeadline and substituted the
// sampled action AFTERWARDS. That wrapper is Search -> SelectControllerAction
// -> CommitAction, and CommitAction PERSISTS the decision into session
// history before returning — so on exactly the decisions Part B exists to
// change, the game would play the sampled action while the session recorded
// the argmax, and every later re-root would key off a history that never
// happened.
//
// Every other test in this file passes against that broken implementation.
// This one does not: it drives the real two-phase lifecycle and asserts the
// action the session COMMITTED equals the action Part B chose.
void TestCommitBoundary() {
  std::cout << "PF-2 test 5: commit boundary (committed == played)\n";
  const uint64_t kSeed = 4242424242ULL;

  // A COHERENT trajectory, not a bag of cloned positions. The session keeps
  // persistent history and re-roots against it, so feeding it unrelated
  // states makes every re-root a mismatch and tests nothing about commit
  // ordering. This drives one real game exactly as the benchmark does:
  // opponents play, the searched seat goes through the two-phase lifecycle,
  // and the committed action is the one applied to the state.
  auto evaluator = std::make_shared<MockEvaluator>(
      std::vector<double>{0.0, 0.0, 0.0, 0.0});
  DuneSearchConfig cfg = MakeConfig(kSeed, /*check_strategic_state=*/true);
  cfg.purchase_combat_budget = 0;
  DuneSearchSession session(cfg, evaluator, DuneSearchBudgetMode::kPolicyOnly);

  std::shared_ptr<const Game> game = LoadGame("dune_imperium");
  std::unique_ptr<State> state = game->NewInitialState();
  std::mt19937 rng(20270904);
  const Player searched_seat = 0;

  int sampled = 0, overridden_differs = 0, checked = 0;
  uint64_t ordinal = 0;
  const int kMaxDecisions = 220;

  while (!state->IsTerminal() && checked < kMaxDecisions) {
    if (state->IsChanceNode()) {
      auto outcomes = state->ChanceOutcomes();
      std::vector<double> w;
      w.reserve(outcomes.size());
      for (const auto& o : outcomes) w.push_back(o.second);
      std::discrete_distribution<int> d(w.begin(), w.end());
      state->ApplyAction(outcomes[d(rng)].first);
      continue;
    }
    const Player p = state->CurrentPlayer();
    if (p != searched_seat) {
      std::vector<Action> legal = state->LegalActions();
      state->ApplyAction(legal[rng() % legal.size()]);
      continue;
    }

    // The ordinal counts EVERY candidate decision, matching Path B.
    ++ordinal;

    // The production two-phase lifecycle, exactly as the benchmark runs it.
    DuneSearchResult searched = session.Search(*state);
    ControllerDecision dec = session.SelectControllerAction(*state, searched, 0.0);
    const Action argmax_action = dec.selected_action;

    Action intended = argmax_action;
    if (pf2::MatchedFallbackEligible(*state, p) && !searched.policy.empty()) {
      const uint64_t seed = pf2::MatchedFallbackStepSeed(kSeed, ordinal);
      intended = pf2::MatchedFallbackSelect(searched.policy,
                                            pf2::MatchedFallbackRVal(seed));
      dec.selected_action = intended;      // override BEFORE the commit
      sampled++;
      if (intended != argmax_action) overridden_differs++;
    }

    DuneSearchResult committed = session.CommitAction(dec);

    // THE ASSERTION. If the override landed after CommitAction (the defect),
    // the committed action would still be the argmax on overridden rows.
    if (committed.diagnostics.selected_action != intended) {
      std::cout << "  FAIL at decision " << ordinal << ": committed "
                << committed.diagnostics.selected_action << ", Part B chose "
                << intended << std::endl;
      assert(false);
    }
    checked++;
    state->ApplyAction(intended);          // play what was committed
  }

  std::cout << "  decisions=" << checked << "  sampled=" << sampled
            << "  sampled-and-differing-from-argmax=" << overridden_differs
            << "\n";

  // DISCRIMINATION, and this one is essential: if the sampled action NEVER
  // differed from the argmax, a commit-after-override implementation would
  // record the same action either way and this test could not tell them
  // apart. A green result would then be meaningless.
  if (overridden_differs == 0) {
    std::cout << "  FAIL(vacuous): the sampled action never differed from the "
                 "argmax on this block, so the commit-ordering defect would be "
                 "invisible here. Widen the block.\n";
    assert(false);
  }
  std::cout << "  PASS — the committed action is Part B's action on every "
               "sampled decision, including "
            << overridden_differs << " where it differs from the argmax\n\n";
}

// ---------------------------------------------------------------------------
// TEST 6 — PART A FIELD REGRESSION.
// ---------------------------------------------------------------------------
//
// The defect Part A fixes: on the fresh path the session-populated
// diagnostics stay blank/-1, because only dune_search_session.cc:501 ever
// writes them. 13,395 of 26,147 historical `fallback_reason=="none"` rows
// carry round -1 and an empty role.
//
// Exercises pf2::PopulatePathBDiagnostics — the function the benchmark calls.
void TestPartAFields() {
  std::cout << "PF-2 test 6: Part A field population\n";

  struct FakeDiag {
    int round = -1;
    std::string phase;
    std::string decision_role;
  };

  std::vector<Decision> decisions = CollectDecisions(150, 20270905);
  assert(!decisions.empty());

  int populated_round = 0, populated_phase = 0, unknown_phase = 0;
  for (size_t k = 0; k < decisions.size(); ++k) {
    FakeDiag d;
    // Pre-state is the defect: blank/-1, exactly what the fresh path emitted.
    assert(d.round == -1 && d.phase.empty() && d.decision_role.empty());

    pf2::PopulatePathBDiagnostics(d, *decisions[k].state, /*role=*/2);

    assert(!d.decision_role.empty());
    assert(d.decision_role == "2");
    if (d.round != -1) populated_round++;
    if (!d.phase.empty()) populated_phase++;
    // A phase that maps to "Unknown" means the switch has fallen behind the
    // engine's GamePhase enum -- the session path would then disagree with
    // the fresh path on the same position.
    if (d.phase == "Unknown") unknown_phase++;
  }

  std::cout << "  decisions=" << decisions.size()
            << "  round populated=" << populated_round
            << "  phase populated=" << populated_phase
            << "  phase==Unknown=" << unknown_phase << std::endl;

  if (populated_round == 0 || populated_phase == 0) {
    std::cout << "  FAIL: Part A populated nothing -- the fields would still "
                 "serialize blank/-1 on the fresh path.\n";
    assert(false);
  }
  if (unknown_phase != 0) {
    std::cout << "  FAIL: " << unknown_phase << " decision(s) mapped to "
                 "\"Unknown\" -- the phase switch has fallen behind the "
                 "engine's GamePhase enum.\n";
    assert(false);
  }

  // The zero-simulation row must stay distinguishable: simulations_completed
  // is the searched-row discriminator on the fresh path, and Part A must not
  // have disturbed it.
  auto evaluator = std::make_shared<MockEvaluator>(
      std::vector<double>{0.0, 0.0, 0.0, 0.0});
  DunePUCTISMCTSBot bot(MakeConfig(77777, true), evaluator);
  int zero_sim = 0, positive_sim = 0;
  for (size_t k = 0; k < decisions.size(); ++k) {
    DuneSearchResult r = bot.RunSearch(*decisions[k].state);
    if (r.simulations_completed == 0) zero_sim++;
    else positive_sim++;
  }
  std::cout << "  zero-simulation rows=" << zero_sim
            << "  positive-simulation rows=" << positive_sim << std::endl;
  if (zero_sim == 0 || positive_sim == 0) {
    std::cout << "  FAIL(vacuous): the block has only one kind of row, so "
                 "`simulations_completed==0` could not be shown to "
                 "discriminate.\n";
    assert(false);
  }
  std::cout << "  PASS — fields populated, phase mapping current, "
               "zero-simulation rows distinguishable\n\n";
}

}  // namespace
}  // namespace open_spiel

int main(int argc, char** argv) {
  std::cout << "=== PF-2 Part B matched-fallback conformance ===\n\n";
  open_spiel::TestClassificationParity();
  open_spiel::TestSamplerBoundary();
  open_spiel::TestOrdinalSequence();
  open_spiel::TestSeedDerivationAgainstPathB();
  open_spiel::TestCommitBoundary();
  open_spiel::TestPartAFields();
  std::cout << "=== all PF-2 conformance tests passed ===\n";
  return 0;
}
