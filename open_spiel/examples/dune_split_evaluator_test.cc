// Acceptance tests for SplitPolicyValueEvaluator (dune_split_evaluator.h).
//
// The class under test is four forwarding lines, which is exactly why it needs
// a test. Swap its two members, or "optimise" PriorAndEvaluate into a single
// fused call the way both production evaluators do, and the result still
// compiles, still runs, still returns a well-formed prior and a well-formed
// value vector, and still trips no assertion anywhere downstream. The only
// thing that changes is WHICH model's value head steers the search -- which is
// the entire point of the class and is invisible in its outputs.
//
// So these tests do not check that the outputs are sane. They check which
// source produced them, and they do it two independent ways:
//
//   BY VALUE, using two mock models whose priors and whose per-player values
//   are pairwise distinguishable, so "this came from A and not B" is decidable
//   from the returned numbers alone; and
//
//   BY CALL COUNT, which is the only evidence that separates a correct unfused
//   implementation from a fused one that happens to agree on this state. A
//   fused implementation would show a value-head call on the policy source; the
//   correct one shows zero.
//
// Mock evaluators only: no torch, no CUDA, no checkpoint. LoadGame is used
// solely because Evaluator's interface takes a State and one has to come from
// somewhere; nothing here depends on the game's rules.

#include "dune_split_evaluator.h"

#include <iostream>
#include <memory>
#include <utility>
#include <vector>

#include "open_spiel/algorithms/mcts.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace {

// --- Fixtures --------------------------------------------------------------

// Head mass for each mock's prior. The two mocks differ in BOTH the action that
// carries the head and the size of the head, so the two priors stay
// distinguishable even in a state with only two legal actions, where "head on
// the first action" and "head on the last action" would otherwise be the same
// shape read from opposite ends.
constexpr double kCandidateHead = 0.80;
constexpr double kFrozenHead = 0.60;

// Per-player leaf values, written as functions rather than constants so the
// assertions can invoke the identical arithmetic the mocks do. That makes the
// comparisons exact instead of approximately right, which matters because the
// split evaluator is supposed to pass its source's vector through UNCHANGED --
// a tolerance would hide exactly the kind of transformation being ruled out.
//
// The two curves slope in opposite directions and do not meet at any player
// index a real game has, so "this value came from the frozen model, not the
// candidate" is decidable per player rather than only in aggregate.
double CandidateValueFor(int player) { return 0.75 - 0.10 * player; }
double FrozenValueFor(int player) { return -0.50 + 0.10 * player; }

// Mock A -- the CANDIDATE. The model whose policy we want to search with, and
// whose value head must never reach the tree. Its prior puts the head on the
// FIRST legal action; its values are the recognisable ones that constitute the
// failure signal if they ever show up in an Evaluate() result.
//
// Modelled on PeakedMockEvaluator in dune_search_pi_test.cc, with call counters
// added: the counters, not the numbers, are what make an accidental fusion
// detectable.
class CandidateMockEvaluator : public algorithms::Evaluator {
 public:
  explicit CandidateMockEvaluator(int num_players) : num_players_(num_players) {}

  std::vector<double> Evaluate(const State& state) override {
    ++evaluate_calls_;
    std::vector<double> out(num_players_, 0.0);
    for (int p = 0; p < num_players_; ++p) out[p] = CandidateValueFor(p);
    return out;
  }

  ActionsAndProbs Prior(const State& state) override {
    ++prior_calls_;
    std::vector<Action> legal = state.LegalActions();
    ActionsAndProbs out;
    out.reserve(legal.size());
    if (legal.empty()) return out;
    const double tail =
        legal.size() > 1
            ? (1.0 - kCandidateHead) / static_cast<double>(legal.size() - 1)
            : 0.0;
    for (size_t i = 0; i < legal.size(); ++i) {
      out.push_back({legal[i], i == 0 ? kCandidateHead : tail});
    }
    return out;
  }

  // Counted, but otherwise left exactly as the base class wrote it, so
  // overriding it changes no behaviour. The split evaluator must never reach
  // this: if it delegated the fused entry point downwards instead of making its
  // own two narrow calls, every other assertion in this file would still pass
  // and this counter would be the only thing that noticed.
  std::pair<ActionsAndProbs, std::vector<double>> PriorAndEvaluate(
      const State& state) override {
    ++prior_and_evaluate_calls_;
    return algorithms::Evaluator::PriorAndEvaluate(state);
  }

  int prior_calls() const { return prior_calls_; }
  int evaluate_calls() const { return evaluate_calls_; }
  int prior_and_evaluate_calls() const { return prior_and_evaluate_calls_; }

 private:
  const int num_players_;
  int prior_calls_ = 0;
  int evaluate_calls_ = 0;
  int prior_and_evaluate_calls_ = 0;
};

// Mock B -- the FROZEN model. The one whose values are supposed to drive the
// search and whose policy must never be used. Its prior puts the head on the
// LAST legal action, and its values are the recognisable ones that a correct
// Evaluate() must return.
class FrozenMockEvaluator : public algorithms::Evaluator {
 public:
  explicit FrozenMockEvaluator(int num_players) : num_players_(num_players) {}

  std::vector<double> Evaluate(const State& state) override {
    ++evaluate_calls_;
    std::vector<double> out(num_players_, 0.0);
    for (int p = 0; p < num_players_; ++p) out[p] = FrozenValueFor(p);
    return out;
  }

  ActionsAndProbs Prior(const State& state) override {
    ++prior_calls_;
    std::vector<Action> legal = state.LegalActions();
    ActionsAndProbs out;
    out.reserve(legal.size());
    if (legal.empty()) return out;
    const size_t head_index = legal.size() - 1;
    const double tail =
        legal.size() > 1
            ? (1.0 - kFrozenHead) / static_cast<double>(legal.size() - 1)
            : 0.0;
    for (size_t i = 0; i < legal.size(); ++i) {
      out.push_back({legal[i], i == head_index ? kFrozenHead : tail});
    }
    return out;
  }

  std::pair<ActionsAndProbs, std::vector<double>> PriorAndEvaluate(
      const State& state) override {
    ++prior_and_evaluate_calls_;
    return algorithms::Evaluator::PriorAndEvaluate(state);
  }

  int prior_calls() const { return prior_calls_; }
  int evaluate_calls() const { return evaluate_calls_; }
  int prior_and_evaluate_calls() const { return prior_and_evaluate_calls_; }

 private:
  const int num_players_;
  int prior_calls_ = 0;
  int evaluate_calls_ = 0;
  int prior_and_evaluate_calls_ = 0;
};

// Exact equality, deliberately. The split evaluator hands back its source's
// container untouched, so any difference at all is a defect; comparing with a
// tolerance would tolerate precisely the silent transformation being excluded.
bool SamePolicy(const ActionsAndProbs& a, const ActionsAndProbs& b) {
  if (a.size() != b.size()) return false;
  for (size_t i = 0; i < a.size(); ++i) {
    if (a[i].first != b[i].first) return false;
    if (a[i].second != b[i].second) return false;
  }
  return true;
}

bool SameValues(const std::vector<double>& a, const std::vector<double>& b) {
  if (a.size() != b.size()) return false;
  for (size_t i = 0; i < a.size(); ++i) {
    if (a[i] != b[i]) return false;
  }
  return true;
}

// The first real decision state, reached the way the rest of this directory
// reaches one (dune_puct_is_mcts_test.cc walks the setup chance nodes and
// stops). Nothing here needs a particular state -- only one that is not
// terminal and offers at least two legal actions, so that the two mock priors
// are actually different objects rather than accidentally equal. The bounded
// loop keeps a pathological game from hanging the test; callers assert the
// postcondition rather than assuming it, so a state that fails to qualify fails
// the test loudly instead of passing it vacuously.
std::unique_ptr<State> FirstBranchingDecisionState(
    const std::shared_ptr<const Game>& game) {
  std::unique_ptr<State> state = game->NewInitialState();
  for (int guard = 0; guard < 1024; ++guard) {
    while (state->IsChanceNode()) {
      state->ApplyAction(state->ChanceOutcomes().front().first);
    }
    if (state->IsTerminal()) break;
    const std::vector<Action> legal = state->LegalActions();
    if (legal.size() >= 2 || legal.empty()) break;
    state->ApplyAction(legal.front());
  }
  return state;
}

// Asserts the fixture is capable of discriminating before any test relies on
// it. Without this, a single-action state would make "A's prior differs from
// B's prior" trivially false and every comparison below meaningless.
void CheckFixtureIsDiscriminating(const State& state) {
  SPIEL_CHECK_FALSE(state.IsTerminal());
  SPIEL_CHECK_GE(static_cast<int>(state.LegalActions().size()), 2);
}

// --- 1. Prior() reads the POLICY source, and only it ----------------------
void Test1_PriorComesFromThePolicySource(
    const std::shared_ptr<const Game>& game) {
  std::cout << "Running Test1_PriorComesFromThePolicySource...\n";
  const int num_players = game->NumPlayers();
  auto candidate = std::make_shared<CandidateMockEvaluator>(num_players);
  auto frozen = std::make_shared<FrozenMockEvaluator>(num_players);
  SplitPolicyValueEvaluator split(candidate, frozen);

  // The wiring itself, by identity. Both sources have the same static type, so
  // a swapped constructor call is otherwise undetectable at this level.
  SPIEL_CHECK_TRUE(split.policy_source().get() == candidate.get());
  SPIEL_CHECK_TRUE(split.value_source().get() == frozen.get());

  std::unique_ptr<State> state = FirstBranchingDecisionState(game);
  CheckFixtureIsDiscriminating(*state);

  // References computed on FRESH mock instances so that producing the expected
  // values does not disturb the counters of the two wired-in sources.
  CandidateMockEvaluator ref_candidate(num_players);
  FrozenMockEvaluator ref_frozen(num_players);
  const ActionsAndProbs expect_candidate = ref_candidate.Prior(*state);
  const ActionsAndProbs expect_frozen = ref_frozen.Prior(*state);
  SPIEL_CHECK_FALSE(SamePolicy(expect_candidate, expect_frozen));

  const ActionsAndProbs got = split.Prior(*state);
  SPIEL_CHECK_TRUE(SamePolicy(got, expect_candidate));
  SPIEL_CHECK_FALSE(SamePolicy(got, expect_frozen));

  // The same conclusion in a human-legible form: the head sits on the first
  // legal action at the candidate's mass, which is the candidate's signature.
  SPIEL_CHECK_EQ(got.size(), state->LegalActions().size());
  SPIEL_CHECK_FLOAT_EQ(got.front().second, kCandidateHead);
  SPIEL_CHECK_TRUE(got.back().second != kFrozenHead);
  std::cout << "  prior head " << got.front().second << " on action "
            << got.front().first << "; frozen head mass absent\n";
  std::cout << "Test1 Passed!\n\n";
}

// --- 2. Evaluate() reads the VALUE source, and only it --------------------
void Test2_EvaluateComesFromTheValueSource(
    const std::shared_ptr<const Game>& game) {
  std::cout << "Running Test2_EvaluateComesFromTheValueSource...\n";
  const int num_players = game->NumPlayers();
  auto candidate = std::make_shared<CandidateMockEvaluator>(num_players);
  auto frozen = std::make_shared<FrozenMockEvaluator>(num_players);
  SplitPolicyValueEvaluator split(candidate, frozen);

  std::unique_ptr<State> state = FirstBranchingDecisionState(game);
  CheckFixtureIsDiscriminating(*state);

  const std::vector<double> got = split.Evaluate(*state);
  SPIEL_CHECK_EQ(static_cast<int>(got.size()), num_players);

  // Per player, not just in aggregate: a partial copy or an off-by-one splice
  // would survive a whole-vector summary.
  for (int p = 0; p < num_players; ++p) {
    SPIEL_CHECK_FLOAT_EQ(got[p], FrozenValueFor(p));
    SPIEL_CHECK_TRUE(got[p] != CandidateValueFor(p));
  }
  std::cout << "  values[0]=" << got[0] << " (frozen "
            << FrozenValueFor(0) << ", candidate " << CandidateValueFor(0)
            << ")\n";
  std::cout << "Test2 Passed!\n\n";
}

// --- 3. PriorAndEvaluate() returns A's policy and B's values together -----
//
// This is the criterion the class exists for. Tests 1 and 2 exercise entry
// points MCTS does not call at a leaf; this one exercises the entry point it
// does, which is where a fused implementation would take both heads off the
// candidate.
void Test3_PriorAndEvaluateSplitsAcrossBothSources(
    const std::shared_ptr<const Game>& game) {
  std::cout << "Running Test3_PriorAndEvaluateSplitsAcrossBothSources...\n";
  const int num_players = game->NumPlayers();
  auto candidate = std::make_shared<CandidateMockEvaluator>(num_players);
  auto frozen = std::make_shared<FrozenMockEvaluator>(num_players);
  SplitPolicyValueEvaluator split(candidate, frozen);

  std::unique_ptr<State> state = FirstBranchingDecisionState(game);
  CheckFixtureIsDiscriminating(*state);

  CandidateMockEvaluator ref_candidate(num_players);
  FrozenMockEvaluator ref_frozen(num_players);
  const ActionsAndProbs expect_policy = ref_candidate.Prior(*state);
  const ActionsAndProbs frozen_policy = ref_frozen.Prior(*state);
  const std::vector<double> expect_values = ref_frozen.Evaluate(*state);
  const std::vector<double> candidate_values = ref_candidate.Evaluate(*state);
  SPIEL_CHECK_FALSE(SamePolicy(expect_policy, frozen_policy));
  SPIEL_CHECK_FALSE(SameValues(expect_values, candidate_values));

  const std::pair<ActionsAndProbs, std::vector<double>> out =
      split.PriorAndEvaluate(*state);

  // Both halves, from one call, from different models.
  SPIEL_CHECK_TRUE(SamePolicy(out.first, expect_policy));
  SPIEL_CHECK_FALSE(SamePolicy(out.first, frozen_policy));
  SPIEL_CHECK_TRUE(SameValues(out.second, expect_values));
  SPIEL_CHECK_FALSE(SameValues(out.second, candidate_values));

  // And the two halves agree with the narrow entry points, so the fused path
  // is not a second, divergent implementation of either.
  SPIEL_CHECK_TRUE(SamePolicy(out.first, split.Prior(*state)));
  SPIEL_CHECK_TRUE(SameValues(out.second, split.Evaluate(*state)));
  std::cout << "  one call yielded candidate prior head "
            << out.first.front().second << " with frozen value "
            << out.second[0] << "\n";
  std::cout << "Test3 Passed!\n\n";
}

// --- 4. Call counters prove the two sources are exercised independently ---
//
// The decisive test. Values can coincide; call counts cannot. A fused
// implementation reading both heads off the policy source would show
// candidate Evaluate calls; a delegated one would show candidate or frozen
// PriorAndEvaluate calls; a swapped one would show the counts on the wrong
// objects. All three are excluded below.
void Test4_CallCountersProveIndependentSources(
    const std::shared_ptr<const Game>& game) {
  std::cout << "Running Test4_CallCountersProveIndependentSources...\n";
  const int num_players = game->NumPlayers();
  auto candidate = std::make_shared<CandidateMockEvaluator>(num_players);
  auto frozen = std::make_shared<FrozenMockEvaluator>(num_players);
  SplitPolicyValueEvaluator split(candidate, frozen);

  std::unique_ptr<State> state = FirstBranchingDecisionState(game);
  CheckFixtureIsDiscriminating(*state);

  // Nothing has touched either source yet; from here the split evaluator is the
  // ONLY caller, so every increment below is attributable to it.
  SPIEL_CHECK_EQ(candidate->prior_calls(), 0);
  SPIEL_CHECK_EQ(candidate->evaluate_calls(), 0);
  SPIEL_CHECK_EQ(frozen->prior_calls(), 0);
  SPIEL_CHECK_EQ(frozen->evaluate_calls(), 0);

  split.PriorAndEvaluate(*state);

  // Exactly one prior, from the policy source. Zero values from it: this is the
  // assertion that catches an accidental fusion.
  SPIEL_CHECK_EQ(candidate->prior_calls(), 1);
  SPIEL_CHECK_EQ(candidate->evaluate_calls(), 0);
  // Exactly one value, from the value source. Zero priors from it.
  SPIEL_CHECK_EQ(frozen->evaluate_calls(), 1);
  SPIEL_CHECK_EQ(frozen->prior_calls(), 0);
  // And the fusion was not simply pushed one level down.
  SPIEL_CHECK_EQ(candidate->prior_and_evaluate_calls(), 0);
  SPIEL_CHECK_EQ(frozen->prior_and_evaluate_calls(), 0);
  std::cout << "  after PriorAndEvaluate: candidate prior=1 evaluate=0, "
            << "frozen prior=0 evaluate=1\n";

  // The narrow entry points route the same way, and each touches one source
  // only -- the counters that must not move are checked, not just the ones that
  // must.
  split.Prior(*state);
  SPIEL_CHECK_EQ(candidate->prior_calls(), 2);
  SPIEL_CHECK_EQ(candidate->evaluate_calls(), 0);
  SPIEL_CHECK_EQ(frozen->prior_calls(), 0);
  SPIEL_CHECK_EQ(frozen->evaluate_calls(), 1);

  split.Evaluate(*state);
  SPIEL_CHECK_EQ(frozen->evaluate_calls(), 2);
  SPIEL_CHECK_EQ(frozen->prior_calls(), 0);
  SPIEL_CHECK_EQ(candidate->prior_calls(), 2);
  SPIEL_CHECK_EQ(candidate->evaluate_calls(), 0);
  std::cout << "  candidate evaluate_calls stayed 0 across all three entry "
            << "points\n";
  std::cout << "Test4 Passed!\n\n";
}

// --- 5. Degenerate wiring: one evaluator supplied as both sources ---------
//
// The control arm. Aliasing must be legal and must reduce to the wrapped
// evaluator's own behaviour, so that a split-vs-unsplit comparison isolates the
// change of MODEL rather than confounding it with a change of call pattern.
void Test5_AliasedSourcesDegenerateToOneModel(
    const std::shared_ptr<const Game>& game) {
  std::cout << "Running Test5_AliasedSourcesDegenerateToOneModel...\n";
  const int num_players = game->NumPlayers();
  auto both = std::make_shared<CandidateMockEvaluator>(num_players);
  SplitPolicyValueEvaluator split(both, both);
  SPIEL_CHECK_TRUE(split.policy_source().get() == split.value_source().get());

  std::unique_ptr<State> state = FirstBranchingDecisionState(game);
  CheckFixtureIsDiscriminating(*state);

  const std::pair<ActionsAndProbs, std::vector<double>> out =
      split.PriorAndEvaluate(*state);

  // Still two separate calls, now landing on the same object: aliasing removes
  // the second MODEL, not the second CALL.
  SPIEL_CHECK_EQ(both->prior_calls(), 1);
  SPIEL_CHECK_EQ(both->evaluate_calls(), 1);
  SPIEL_CHECK_EQ(both->prior_and_evaluate_calls(), 0);

  // The results are that evaluator's own, compared against a fresh instance so
  // the reference is independent of anything the split evaluator did.
  CandidateMockEvaluator ref(num_players);
  const ActionsAndProbs ref_policy = ref.Prior(*state);
  const std::vector<double> ref_values = ref.Evaluate(*state);
  SPIEL_CHECK_TRUE(SamePolicy(out.first, ref_policy));
  SPIEL_CHECK_TRUE(SameValues(out.second, ref_values));
  SPIEL_CHECK_TRUE(SamePolicy(split.Prior(*state), ref_policy));
  SPIEL_CHECK_TRUE(SameValues(split.Evaluate(*state), ref_values));
  std::cout << "  aliased sources reproduced the wrapped evaluator exactly\n";
  std::cout << "Test5 Passed!\n\n";
}

}  // namespace
}  // namespace open_spiel

int main(int argc, char** argv) {
  using namespace open_spiel;
  std::shared_ptr<const Game> game = LoadGame("dune_imperium");

  Test1_PriorComesFromThePolicySource(game);
  Test2_EvaluateComesFromTheValueSource(game);
  Test3_PriorAndEvaluateSplitsAcrossBothSources(game);
  Test4_CallCountersProveIndependentSources(game);
  Test5_AliasedSourcesDegenerateToOneModel(game);

  std::cout << "All dune split-evaluator tests passed!\n";
  return 0;
}
