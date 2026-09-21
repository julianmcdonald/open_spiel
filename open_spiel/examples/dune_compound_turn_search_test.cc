// Unit test for TurnSearchTree and compound turn search

#include "open_spiel/examples/dune_compound_turn_search.h"
#include "open_spiel/spiel_utils.h"

int main() {
  using namespace open_spiel;
  using namespace open_spiel::dune_imperium;

  TurnSearchTree tree;

  // Test 1: Multi-sample tree accumulation and branching
  // Candidate card 10: 2 rollouts
  // Rollout 1: Space 25 -> Deploy 3 -> EndTurn (Payoff +2.0)
  // Rollout 2: Space 25 -> Deploy 2 -> EndTurn (Payoff +3.0)
  // Rollout 3: Space 26 -> Deploy 0 -> EndTurn (Payoff -1.0)
  // Candidate card 11: 1 rollout
  // Rollout 4: Space 28 -> Deploy 0 -> EndTurn (Payoff -2.0)
  tree.RecordRollout(10, {25, 3, 99}, 2.0);
  tree.RecordRollout(10, {25, 2, 99}, 3.0);
  tree.RecordRollout(10, {26, 0, 99}, -1.0);
  tree.RecordRollout(11, {28, 0, 99}, -2.0);

  SPIEL_CHECK_TRUE(tree.HasPriorData(10));
  SPIEL_CHECK_EQ(tree.GetVisits(10), 3);
  SPIEL_CHECK_FLOAT_EQ(tree.GetUtilitySum(10), 4.0);
  SPIEL_CHECK_FLOAT_EQ(tree.GetMeanUtility(10), 4.0 / 3.0);

  SPIEL_CHECK_TRUE(tree.HasPriorData(11));
  SPIEL_CHECK_EQ(tree.GetVisits(11), 1);
  SPIEL_CHECK_FLOAT_EQ(tree.GetMeanUtility(11), -2.0);

  // Player commits to Card 10 in real game
  tree.Advance(10);

  // Now at Space decision point:
  // Space 25 should have 2 visits with mean utility +2.5
  SPIEL_CHECK_TRUE(tree.HasPriorData(25));
  SPIEL_CHECK_EQ(tree.GetVisits(25), 2);
  SPIEL_CHECK_FLOAT_EQ(tree.GetUtilitySum(25), 5.0);
  SPIEL_CHECK_FLOAT_EQ(tree.GetMeanUtility(25), 2.5);

  // Space 26 should have 1 visit with mean utility -1.0
  SPIEL_CHECK_TRUE(tree.HasPriorData(26));
  SPIEL_CHECK_EQ(tree.GetVisits(26), 1);
  SPIEL_CHECK_FLOAT_EQ(tree.GetMeanUtility(26), -1.0);

  // Space 28 was under card 11, so it should NOT be present under card 10!
  SPIEL_CHECK_FALSE(tree.HasPriorData(28));
  SPIEL_CHECK_EQ(tree.GetVisits(28), 0);

  // Simulate additional rollout topping up Space 25
  tree.RecordRollout(25, {3, 99}, 1.0);
  SPIEL_CHECK_EQ(tree.GetVisits(25), 3);
  SPIEL_CHECK_FLOAT_EQ(tree.GetUtilitySum(25), 6.0);
  SPIEL_CHECK_FLOAT_EQ(tree.GetMeanUtility(25), 2.0);

  // Player commits to Space 25 in real game
  tree.Advance(25);

  // Now at Deploy decision point:
  // Deploy 3 had 1 visit (+2.0) from card rollouts + 1 visit (+1.0) from space rollouts = 2 visits, mean 1.5
  SPIEL_CHECK_TRUE(tree.HasPriorData(3));
  SPIEL_CHECK_EQ(tree.GetVisits(3), 2);
  SPIEL_CHECK_FLOAT_EQ(tree.GetUtilitySum(3), 3.0);
  SPIEL_CHECK_FLOAT_EQ(tree.GetMeanUtility(3), 1.5);

  // Deploy 2 had 1 visit (+3.0)
  SPIEL_CHECK_TRUE(tree.HasPriorData(2));
  SPIEL_CHECK_EQ(tree.GetVisits(2), 1);
  SPIEL_CHECK_FLOAT_EQ(tree.GetMeanUtility(2), 3.0);

  // Deploy 0 was under space 26, NOT space 25
  SPIEL_CHECK_FALSE(tree.HasPriorData(0));

  // Player commits to Deploy 3
  tree.Advance(3);

  // End of turn (kActionEndTurn), tree resets
  tree.Reset();
  SPIEL_CHECK_FALSE(tree.HasPriorData(10));
  SPIEL_CHECK_FALSE(tree.HasPriorData(25));
  SPIEL_CHECK_FALSE(tree.HasPriorData(3));

  // Test 2: Prior selection utilities
  ActionsAndProbs test_prior = {{10, 0.7}, {11, 0.2}, {12, 0.1}};
  std::vector<Action> legal_acts = {11, 10, 12};
  SPIEL_CHECK_EQ(PickGreedyAction(test_prior, legal_acts), 10);

  // Test 3: HasNewInformation detection
  std::shared_ptr<const Game> game = LoadGame("dune_imperium");
  std::unique_ptr<State> initial_state = game->NewInitialState();
  auto* dune_base = dynamic_cast<DuneImperiumState*>(initial_state.get());
  SPIEL_CHECK_TRUE(dune_base != nullptr);

  auto s_before = std::unique_ptr<DuneImperiumState>(
      static_cast<DuneImperiumState*>(dune_base->Clone().release()));
  auto s_after = std::unique_ptr<DuneImperiumState>(
      static_cast<DuneImperiumState*>(dune_base->Clone().release()));

  // Initially identical: no new information
  SPIEL_CHECK_FALSE(HasNewInformation(*s_before, *s_after, 0));

  // Case 1: Draw a card from deck
  s_after->GetPlayerDrawDeckForTesting(0).pop_back();
  SPIEL_CHECK_TRUE(HasNewInformation(*s_before, *s_after, 0));
  // Player 1 did not draw: should still be false for Player 1
  SPIEL_CHECK_FALSE(HasNewInformation(*s_before, *s_after, 1));

  // Reset s_after to match s_before
  s_after = std::unique_ptr<DuneImperiumState>(
      static_cast<DuneImperiumState*>(s_before->Clone().release()));
  SPIEL_CHECK_FALSE(HasNewInformation(*s_before, *s_after, 0));

  // Case 2: Gain an intrigue card
  s_after->SetPlayerIntrigueHandForTesting(0, {42});
  SPIEL_CHECK_TRUE(HasNewInformation(*s_before, *s_after, 0));

  // Reset s_after
  s_after = std::unique_ptr<DuneImperiumState>(
      static_cast<DuneImperiumState*>(s_before->Clone().release()));

  // Case 3: Play a card from hand without drawing (draw deck unchanged)
  if (!s_after->GetPlayerHandForTesting(0).empty()) {
    s_after->GetPlayerHandForTesting(0).pop_back();
    SPIEL_CHECK_FALSE(HasNewInformation(*s_before, *s_after, 0));
  }

  std::cout << "All TurnSearchTree and CompoundTurnSearch unit tests PASSED successfully!\n";
  return 0;
}
