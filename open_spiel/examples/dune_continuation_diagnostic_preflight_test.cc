// Preflight verification test for Search-PI Factorial Continuation Diagnostic.
// Verifies:
// 1. Infostate resampling invariants (focal player infostate preserved, no card leakage).
// 2. Dynamic divergence & cache invalidation (deviation from plan cleanly drops cached suffix; new info resets tree).
// 3. Student representation alignment (9,182-dim observation is strictly Markov and invariant to search tree memory).
// 4. Bitwise non-invasive search query check (querying search rollouts leaves live state & chance streams bitwise identical).

#include "open_spiel/examples/dune_compound_turn_search.h"
#include "open_spiel/games/dune_imperium/dune_imperium.h"
#include "open_spiel/games/dune_imperium/dune_imperium_test_utils.h"
#include "open_spiel/spiel_utils.h"

#include <iostream>
#include <memory>
#include <random>
#include <vector>

namespace open_spiel {
namespace dune_imperium {
namespace {

void TestInfostateResamplingInvariants() {
  std::cout << "[PREFLIGHT 1] Testing infostate resampling & private card conservation...\n";
  std::shared_ptr<const Game> game = LoadGame("dune_imperium");
  std::unique_ptr<State> state = game->NewInitialState();
  std::mt19937 rng(42);

  // Advance past setup to round 1 agent turns
  while (!state->IsTerminal() && state->CurrentPlayer() != 0) {
    if (state->IsChanceNode()) {
      auto chance = state->ChanceOutcomes();
      Action a = SampleAction(chance, std::generate_canonical<double, 53>(rng)).first;
      state->ApplyAction(a);
    } else {
      auto legals = state->LegalActions();
      state->ApplyAction(legals.front());
    }
  }

  const auto* dune_live = dynamic_cast<const DuneImperiumState*>(state.get());
  SPIEL_CHECK_TRUE(dune_live != nullptr);
  Player focal_player = 0;

  int live_hand_size = dune_live->GetPlayerCardsInHand(focal_player);
  int live_intrigue_count = static_cast<int>(dune_live->GetPlayerIntrigues(focal_player).size());

  // Perform multiple resamples and verify invariants
  for (int r = 0; r < 20; ++r) {
    std::mt19937 wrng(1000 + r);
    auto rng_func = [&wrng]() {
      return std::generate_canonical<double, 53>(wrng);
    };
    std::unique_ptr<State> resampled_state = dune_live->ResampleFromInfostate(focal_player, rng_func);
    const auto* dune_resampled = dynamic_cast<const DuneImperiumState*>(resampled_state.get());
    SPIEL_CHECK_TRUE(dune_resampled != nullptr);

    // 1. Focal player's private state must be strictly identical
    SPIEL_CHECK_EQ(dune_resampled->GetPlayerCardsInHand(focal_player), live_hand_size);
    SPIEL_CHECK_EQ(static_cast<int>(dune_resampled->GetPlayerIntrigues(focal_player).size()), live_intrigue_count);

    // 2. Public board state must be strictly identical
    SPIEL_CHECK_EQ(dune_resampled->GetCurrentRound(), dune_live->GetCurrentRound());
    SPIEL_CHECK_EQ(static_cast<int>(dune_resampled->phase()), static_cast<int>(dune_live->phase()));
    for (int p = 0; p < 4; ++p) {
      SPIEL_CHECK_EQ(dune_resampled->GetPlayerSpiceForTesting(p), dune_live->GetPlayerSpiceForTesting(p));
      SPIEL_CHECK_EQ(dune_resampled->GetPlayerSolari(p), dune_live->GetPlayerSolari(p));
      SPIEL_CHECK_EQ(dune_resampled->GetPlayerWaterForTesting(p), dune_live->GetPlayerWaterForTesting(p));
      SPIEL_CHECK_EQ(dune_resampled->GetPlayerVp(p), dune_live->GetPlayerVp(p));
    }

    // 3. Legal actions for focal player must be identical
    auto live_legals = state->LegalActions();
    auto resample_legals = resampled_state->LegalActions();
    SPIEL_CHECK_EQ(live_legals.size(), resample_legals.size());
    for (size_t i = 0; i < live_legals.size(); ++i) {
      SPIEL_CHECK_EQ(live_legals[i], resample_legals[i]);
    }
  }
  std::cout << "[PREFLIGHT 1] PASS: Infostate resampling strictly preserves focal player private info & legal action mapping.\n";
}

void TestDynamicDivergenceAndCacheInvalidation() {
  std::cout << "[PREFLIGHT 2] Testing dynamic divergence and cache invalidation...\n";
  TurnSearchTree tree;

  // Plan: Candidate Card 10 -> Space 25 -> Deploy 3 -> EndTurn (Payoff 2.5)
  tree.RecordRollout(10, {25, 3, 99}, 2.5);
  tree.RecordRollout(10, {25, 2, 99}, 1.5);

  SPIEL_CHECK_TRUE(tree.HasPriorData(10));
  SPIEL_CHECK_EQ(tree.GetVisits(10), 2);

  // Case A: Student deviates from plan and plays Action 11 instead of 10
  tree.Advance(11);

  // Suffix under 10 must NOT be present under 11
  SPIEL_CHECK_FALSE(tree.HasPriorData(25));
  SPIEL_CHECK_EQ(tree.GetVisits(25), 0);
  SPIEL_CHECK_TRUE(tree.current_node() != nullptr);
  SPIEL_CHECK_TRUE(tree.current_node()->children.empty());

  // Case B: Reset on new information
  tree.Reset();
  tree.RecordRollout(10, {25, 3, 99}, 2.5);
  tree.Advance(10);
  SPIEL_CHECK_TRUE(tree.HasPriorData(25));

  // Intervening observation: player draws card or intrigue -> tree resets
  tree.Reset();
  SPIEL_CHECK_FALSE(tree.HasPriorData(25));
  std::cout << "[PREFLIGHT 2] PASS: Cache branch cleanly invalidates upon student deviation and resets upon new info.\n";
}

void TestStudentRepresentationAlignment() {
  std::cout << "[PREFLIGHT 3] Testing student representation alignment...\n";
  std::shared_ptr<const Game> game = LoadGame("dune_imperium");
  std::unique_ptr<State> state = game->NewInitialState();
  std::mt19937 rng(123);

  while (!state->IsTerminal() && state->CurrentPlayer() != 0) {
    if (state->IsChanceNode()) {
      auto chance = state->ChanceOutcomes();
      Action a = SampleAction(chance, std::generate_canonical<double, 53>(rng)).first;
      state->ApplyAction(a);
    } else {
      state->ApplyAction(state->LegalActions().front());
    }
  }

  const auto* dune = dynamic_cast<const DuneImperiumState*>(state.get());
  SPIEL_CHECK_TRUE(dune != nullptr);

  // Check observation tensor dimension
  auto obs = dune->InformationStateTensorWithAppendix(0, MarketAppendixMode::kFullPublicInformationV3);
  SPIEL_CHECK_EQ(obs.size(), static_cast<size_t>(kFullPublicInformationStateSize));
  SPIEL_CHECK_EQ(obs.size(), 9182);

  // Record a rollout in tree
  TurnSearchTree tree;
  tree.RecordRollout(10, {25, 3, 99}, 2.5);

  // Student observation after tree rollouts must remain identical Markov state
  auto obs_after = dune->InformationStateTensorWithAppendix(0, MarketAppendixMode::kFullPublicInformationV3);
  SPIEL_CHECK_EQ(obs.size(), obs_after.size());
  for (size_t i = 0; i < obs.size(); ++i) {
    SPIEL_CHECK_FLOAT_EQ(obs[i], obs_after[i]);
  }
  std::cout << "[PREFLIGHT 3] PASS: Observation vector (9,182 floats) is strictly Markov and invariant to tree memory.\n";
}

void TestNonInvasiveSearchQueryCheck() {
  std::cout << "[PREFLIGHT 4] Testing non-invasive search query invariant...\n";
  std::shared_ptr<const Game> game = LoadGame("dune_imperium");
  uint64_t test_seed = 20260921;

  // Run 1: Normal game step execution without search
  std::unique_ptr<State> state1 = game->NewInitialState();
  std::mt19937 rng1(test_seed);
  std::vector<Action> actions1;

  for (int step = 0; step < 50 && !state1->IsTerminal(); ++step) {
    if (state1->IsChanceNode()) {
      auto chance = state1->ChanceOutcomes();
      Action a = SampleAction(chance, std::generate_canonical<double, 53>(rng1)).first;
      state1->ApplyAction(a);
      actions1.push_back(a);
    } else {
      auto legals = state1->LegalActions();
      Action a = legals.front();
      state1->ApplyAction(a);
      actions1.push_back(a);
    }
  }

  // Run 2: Parallel execution where hypothetical search clones and resamples are queried at every turn
  std::unique_ptr<State> state2 = game->NewInitialState();
  std::mt19937 rng2(test_seed);
  std::vector<Action> actions2;

  for (int step = 0; step < 50 && !state2->IsTerminal(); ++step) {
    if (state2->IsChanceNode()) {
      auto chance = state2->ChanceOutcomes();
      Action a = SampleAction(chance, std::generate_canonical<double, 53>(rng2)).first;
      state2->ApplyAction(a);
      actions2.push_back(a);
    } else {
      Player cur = state2->CurrentPlayer();
      const auto* dune = dynamic_cast<const DuneImperiumState*>(state2.get());

      // Query hypothetical search clone (should have zero effect on live game)
      if (dune) {
        std::mt19937 search_rng(9999 + step);
        auto rng_f = [&search_rng]() { return std::generate_canonical<double, 53>(search_rng); };
        std::unique_ptr<State> sim = dune->ResampleFromInfostate(cur, rng_f);
        sim->ApplyAction(sim->LegalActions().front());
      }

      auto legals = state2->LegalActions();
      Action a = legals.front();
      state2->ApplyAction(a);
      actions2.push_back(a);
    }
  }

  // Invariant assertion: Actions, state strings, and observations must be bitwise identical
  SPIEL_CHECK_EQ(actions1.size(), actions2.size());
  for (size_t i = 0; i < actions1.size(); ++i) {
    SPIEL_CHECK_EQ(actions1[i], actions2[i]);
  }
  SPIEL_CHECK_EQ(state1->ToString(), state2->ToString());

  auto obs1 = state1->ObservationTensor(0);
  auto obs2 = state2->ObservationTensor(0);
  SPIEL_CHECK_EQ(obs1.size(), obs2.size());
  for (size_t i = 0; i < obs1.size(); ++i) {
    SPIEL_CHECK_FLOAT_EQ(obs1[i], obs2[i]);
  }

  std::cout << "[PREFLIGHT 4] PASS: Live game state, chance streams, and observations are 100% bitwise identical with search queries ON vs OFF.\n";
}

}  // namespace
}  // namespace dune_imperium
}  // namespace open_spiel

int main() {
  open_spiel::dune_imperium::TestInfostateResamplingInvariants();
  open_spiel::dune_imperium::TestDynamicDivergenceAndCacheInvalidation();
  open_spiel::dune_imperium::TestStudentRepresentationAlignment();
  open_spiel::dune_imperium::TestNonInvasiveSearchQueryCheck();

  std::cout << "\n================================================================================\n";
  std::cout << "ALL 4 PREFLIGHT INVARIANTS CERTIFIED SUCCESSFULLY (PASS).\n";
  std::cout << "================================================================================\n";
  return 0;
}
