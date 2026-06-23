#include <iostream>
#include <string>
#include <vector>
#include <memory>
#include <cmath>
#include <cassert>
#include <algorithm>

#include "open_spiel/spiel.h"
#include "open_spiel/games/dune_imperium/dune_imperium.h"
#include "dune_puct_is_mcts.h"
#include "open_spiel/abseil-cpp/absl/random/distributions.h"

namespace open_spiel {

// Defining TestBotAccessor inside open_spiel namespace as a friend of DunePUCTISMCTSBot
struct TestBotAccessor {
  static ActionsAndProbs RunFilterPriors(DunePUCTISMCTSBot& b, DuneISMCTSNode* node, const std::vector<Action>& legals) {
    return b.FilterAndNormalizePriors(node, legals);
  }
  static Action RunSelectTree(DunePUCTISMCTSBot& b, DuneISMCTSNode* node, const std::vector<Action>& legals) {
    return b.SelectActionTreePolicy(node, legals);
  }
  static int GetNodeCount(DunePUCTISMCTSBot& b) {
    return b.node_pool_.size();
  }
  static DuneISMCTSNode* GetRootNode(DunePUCTISMCTSBot& b) {
    return b.root_node_;
  }
  static std::vector<double> Simulate(DunePUCTISMCTSBot& b, State* s) {
    return b.RunSimulation(s);
  }
  static double CallRandomNumber(DunePUCTISMCTSBot& b) {
    return b.RandomNumber();
  }
  static std::mt19937& GetRng(DunePUCTISMCTSBot& b) {
    return b.rng_;
  }
};

namespace {

void AssertAlmostEqual(double a, double b, double tol = 1e-5) {
  if (std::abs(a - b) > tol) {
    std::cerr << "Assertion failed: " << a << " != " << b << " (diff: " << std::abs(a - b) << ")\n";
    std::abort();
  }
}

class MockEvaluator : public algorithms::Evaluator {
 public:
  MockEvaluator(const ActionsAndProbs& priors, const std::vector<double>& values)
      : priors_(priors), values_(values) {}
  
  std::vector<double> Evaluate(const State& state) override {
    return values_;
  }
  
  ActionsAndProbs Prior(const State& state) override {
    ActionsAndProbs filtered_priors;
    std::vector<Action> legal_actions = state.LegalActions();
    for (const auto& action_prob : priors_) {
      if (std::find(legal_actions.begin(), legal_actions.end(), action_prob.first) != legal_actions.end()) {
        filtered_priors.push_back(action_prob);
      }
    }
    double sum = 0.0;
    for (const auto& ap : filtered_priors) sum += ap.second;
    if (sum > 0.0) {
      for (auto& ap : filtered_priors) ap.second /= sum;
    }
    return filtered_priors;
  }
 private:
  ActionsAndProbs priors_;
  std::vector<double> values_;
};

// Test 1: Key Uniqueness and history size presence
void TestKeyUniqueness() {
  std::cout << "Running Test 1: Key Uniqueness...\n";
  std::shared_ptr<const Game> game = LoadGame("dune_imperium");
  std::unique_ptr<State> state = game->NewInitialState();

  // Walk through setup chance nodes to first decision state
  while (state->IsChanceNode()) {
    auto outcomes = state->ChanceOutcomes();
    state->ApplyAction(outcomes.front().first);
  }

  std::string obs1 = state->ObservationString(state->CurrentPlayer());
  assert(obs1.find("|history_size=") != std::string::npos);

  // Take a legal action and observe change in history size
  std::vector<Action> actions = state->LegalActions();
  state->ApplyAction(actions.front());

  std::string obs2 = state->ObservationString(state->CurrentPlayer());
  assert(obs2.find("|history_size=") != std::string::npos);
  assert(obs1 != obs2);

  std::cout << "Test 1 Passed!\n\n";
}

// Test 2: Deterministic PUCT Selection
void TestDeterministicPUCT() {
  std::cout << "Running Test 2: Deterministic PUCT Selection...\n";
  std::shared_ptr<const Game> game = LoadGame("dune_imperium");
  std::unique_ptr<State> state = game->NewInitialState();
  while (state->IsChanceNode()) {
    state->ApplyAction(state->ChanceOutcomes().front().first);
  }

  std::vector<Action> legal_actions = state->LegalActions();
  assert(legal_actions.size() > 1);

  // Mock priors: bias heavily towards the first action
  ActionsAndProbs mock_priors;
  mock_priors.push_back({legal_actions[0], 0.9});
  for (size_t i = 1; i < legal_actions.size(); ++i) {
    mock_priors.push_back({legal_actions[i], 0.1 / (legal_actions.size() - 1)});
  }
  std::vector<double> mock_values = {1.0, 0.0, 0.0, 0.0};

  auto evaluator = std::make_shared<MockEvaluator>(mock_priors, mock_values);
  DunePUCTISMCTSBot bot(42, evaluator, 1.0, 50, -1, 1.0, 0.0, 0.3, 4.0);

  DuneISMCTSNode node;
  node.child_info[legal_actions[0]] = DuneChildInfo{0, 0.0, 0.9};
  node.child_info[legal_actions[1]] = DuneChildInfo{0, 0.0, 0.1};
  node.priors_initialized = true;

  // Unvisited prioritization: SelectActionTreePolicy should return the unvisited action with the highest prior
  Action selected = TestBotAccessor::RunSelectTree(bot, &node, {legal_actions[0], legal_actions[1]});
  assert(selected == legal_actions[0]);

  // Give a visit to legal_actions[0]
  node.total_visits = 1;
  node.child_info[legal_actions[0]].visits = 1;
  node.child_info[legal_actions[0]].return_sum = 0.5;

  // Now legal_actions[1] is unvisited (visits = 0), so it should be prioritized even though it has a lower prior than the visited one
  selected = TestBotAccessor::RunSelectTree(bot, &node, {legal_actions[0], legal_actions[1]});
  assert(selected == legal_actions[1]);

  // Now visit both
  node.total_visits = 2;
  node.child_info[legal_actions[1]].visits = 1;
  node.child_info[legal_actions[1]].return_sum = 0.1;

  // Standard PUCT formula applies. Let's verify selection.
  // value_0 = 0.5 / 1 = 0.5. PUCT_0 = 0.5 + 1.0 * 0.9 * sqrt(2) / (1 + 1) = 0.5 + 0.9 * 1.414 / 2 = 0.5 + 0.636 = 1.136
  // value_1 = 0.1 / 1 = 0.1. PUCT_1 = 0.1 + 1.0 * 0.1 * sqrt(2) / (1 + 1) = 0.1 + 0.1 * 1.414 / 2 = 0.1 + 0.070 = 0.170
  selected = TestBotAccessor::RunSelectTree(bot, &node, {legal_actions[0], legal_actions[1]});
  assert(selected == legal_actions[0]);

  std::cout << "Test 2 Passed!\n\n";
}

// Test 3: Prior Renormalization
void TestPriorRenormalization() {
  std::cout << "Running Test 3: Prior Renormalization...\n";
  std::shared_ptr<const Game> game = LoadGame("dune_imperium");
  std::unique_ptr<State> state = game->NewInitialState();
  while (state->IsChanceNode()) {
    state->ApplyAction(state->ChanceOutcomes().front().first);
  }

  std::vector<Action> legal_actions = state->LegalActions();
  assert(legal_actions.size() > 1);

  ActionsAndProbs mock_priors = {{legal_actions[0], 0.6}, {legal_actions[1], 0.3}};
  auto evaluator = std::make_shared<MockEvaluator>(mock_priors, std::vector<double>{0.0, 0.0, 0.0, 0.0});
  DunePUCTISMCTSBot bot(42, evaluator, 1.0, 10);

  DuneISMCTSNode node;
  node.child_info[legal_actions[0]] = DuneChildInfo{0, 0.0, 0.6};
  node.child_info[legal_actions[1]] = DuneChildInfo{0, 0.0, 0.3};
  node.priors_initialized = true;

  // Filter so only legal_actions[0] and legal_actions[1] are tested
  ActionsAndProbs filtered = TestBotAccessor::RunFilterPriors(bot, &node, {legal_actions[0], legal_actions[1]});
  assert(filtered.size() == 2);
  AssertAlmostEqual(filtered[0].second, 2.0 / 3.0);
  AssertAlmostEqual(filtered[1].second, 1.0 / 3.0);

  // Filter with a missing action: should fall back to 1e-5 prior and re-normalize
  ActionsAndProbs filtered2 = TestBotAccessor::RunFilterPriors(bot, &node, {legal_actions[0], 9999});
  assert(filtered2.size() == 2);
  assert(filtered2[0].first == legal_actions[0]);
  assert(filtered2[1].first == 9999);
  assert(filtered2[0].second > 0.99);
  assert(filtered2[1].second < 0.01);

  std::cout << "Test 3 Passed!\n\n";
}

// Test 4: Value Backpropagation (Max-N backup)
void TestValueBackpropagation() {
  std::cout << "Running Test 4: Value Backpropagation (Max-N backup)...\n";
  std::shared_ptr<const Game> game = LoadGame("dune_imperium");
  std::unique_ptr<State> state = game->NewInitialState();
  while (state->IsChanceNode()) {
    state->ApplyAction(state->ChanceOutcomes().front().first);
  }

  std::vector<Action> legal_actions = state->LegalActions();
  Player current_player = state->CurrentPlayer();

  // Mock priors and values
  ActionsAndProbs mock_priors;
  for (Action a : legal_actions) {
    mock_priors.push_back({a, 1.0 / legal_actions.size()});
  }
  
  // Evaluator returns a custom vector where the acting player gets a high value
  std::vector<double> mock_values(4, 0.0);
  mock_values[current_player] = 0.8;

  auto evaluator = std::make_shared<MockEvaluator>(mock_priors, mock_values);
  DunePUCTISMCTSBot bot(42, evaluator, 1.0, 10, -1, 1.0, 0.0, 0.3, 1.0);

  ActionsAndProbs policy = bot.RunSearch(*state);
  assert(!policy.empty());

  DuneISMCTSNode* root = TestBotAccessor::GetRootNode(bot);
  assert(root != nullptr);
  assert(root->total_visits > 0);

  // Check that the sum of returns matches the evaluation from the evaluator.
  // Note: Out of 10 simulations, the first is spent evaluating the root node itself
  // (which doesn't update any child). The remaining 9 simulations visit and update children.
  // 9 * 0.8 = 7.2
  double sum_q = 0.0;
  for (const auto& action_child : root->child_info) {
    sum_q += action_child.second.return_sum;
  }
  AssertAlmostEqual(sum_q, 7.2, 1e-4);

  std::cout << "Test 4 Passed!\n\n";
}

// Test 5: Single Legal Action Bypass
void TestSingleActionBypass() {
  std::cout << "Running Test 5: Single Legal Action Bypass...\n";
  std::shared_ptr<const Game> game = LoadGame("dune_imperium");
  std::unique_ptr<State> state = game->NewInitialState();

  // We need to construct a state that has exactly 1 legal action
  // Setup chance nodes have exactly 1 outcome (deterministic setups)
  assert(state->IsChanceNode());
  std::vector<Action> legal_actions = state->LegalActions();
  if (legal_actions.size() == 1) {
    auto evaluator = std::make_shared<MockEvaluator>(ActionsAndProbs{}, std::vector<double>{0.0, 0.0, 0.0, 0.0});
    DunePUCTISMCTSBot bot(42, evaluator, 1.0, 10);
    ActionsAndProbs policy = bot.GetPolicy(*state);
    assert(policy.size() == 1);
    assert(policy[0].first == legal_actions[0]);
    AssertAlmostEqual(policy[0].second, 1.0);
  }

  std::cout << "Test 5 Passed!\n\n";
}

// Test 6: Chance Node Traversal
void TestChanceNodeTraversal() {
  std::cout << "Running Test 6: Chance Node Traversal...\n";
  std::shared_ptr<const Game> game = LoadGame("dune_imperium");
  std::unique_ptr<State> state = game->NewInitialState();

  // Root state is a chance node
  assert(state->IsChanceNode());

  // Evaluator mock
  ActionsAndProbs mock_priors;
  std::vector<double> mock_values = {0.2, 0.3, 0.4, 0.1};
  auto evaluator = std::make_shared<MockEvaluator>(mock_priors, mock_values);

  // Bot setup
  DunePUCTISMCTSBot bot(42, evaluator, 1.0, 10, -1, 1.0, 0.0, 0.3, 1.0);

  // Run simulation starting directly on the chance node state
  std::unique_ptr<State> sim_state = state->Clone();
  std::vector<double> values = TestBotAccessor::Simulate(bot, sim_state.get());

  // It should successfully traverse the setup chance nodes and reach a decision leaf node
  // evaluating to mock_values
  assert(values.size() == 4);
  AssertAlmostEqual(values[0], 0.2);
  AssertAlmostEqual(values[1], 0.3);
  AssertAlmostEqual(values[2], 0.4);
  AssertAlmostEqual(values[3], 0.1);

  std::cout << "Test 6 Passed!\n\n";
}

// Test 7: End-to-End Game Rollouts
void TestEndToEndRollouts() {
  std::cout << "Running Test 7: End-to-End Game Rollouts...\n";
  std::shared_ptr<const Game> game = LoadGame("dune_imperium");

  ActionsAndProbs mock_priors;
  for (int i = 0; i < 2391; ++i) {
    mock_priors.push_back({i, 1.0 / 2391.0});
  }
  std::vector<double> mock_values = {0.0, 0.0, 0.0, 0.0};
  auto evaluator = std::make_shared<MockEvaluator>(mock_priors, mock_values);

  for (int g = 0; g < 5; ++g) {
    std::unique_ptr<State> state = game->NewInitialState();
    DunePUCTISMCTSBot mcts_bot(42 + g, evaluator, 1.0, 5, -1, 1.0, 0.0, 0.3, 1.0);

    while (!state->IsTerminal()) {
      if (state->IsChanceNode()) {
        state->ApplyAction(SampleAction(state->ChanceOutcomes(), TestBotAccessor::CallRandomNumber(mcts_bot)).first);
      } else {
        Player p = state->CurrentPlayer();
        if (p == 0) {
          Action action = mcts_bot.Step(*state);
          state->ApplyAction(action);
        } else {
          auto actions = state->LegalActions();
          Action action = actions[absl::Uniform(TestBotAccessor::GetRng(mcts_bot), 0u, actions.size())];
          state->ApplyAction(action);
        }
      }
    }
  }

  std::cout << "Test 7 Passed!\n\n";
}

} // namespace
} // namespace open_spiel

int main() {
  open_spiel::TestKeyUniqueness();
  open_spiel::TestDeterministicPUCT();
  open_spiel::TestPriorRenormalization();
  open_spiel::TestValueBackpropagation();
  open_spiel::TestSingleActionBypass();
  open_spiel::TestChanceNodeTraversal();
  open_spiel::TestEndToEndRollouts();
  std::cout << "All Dune PUCT IS-MCTS tests completed successfully!\n";
  return 0;
}
