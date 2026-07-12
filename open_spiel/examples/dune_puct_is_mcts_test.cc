#include <iostream>
#include <string>
#include <vector>
#include <memory>
#include <cmath>
#include <cassert>
#include <algorithm>
#include <thread>
#include <chrono>

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
    return b.RunSimulation(s, 0, 0);
  }
  static double CallRandomNumber(DunePUCTISMCTSBot& b) {
    return b.RandomNumber();
  }
  static std::mt19937& GetRng(DunePUCTISMCTSBot& b) {
    return b.rng_;
  }
  static DuneSearchConfig& GetConfig(DunePUCTISMCTSBot& b) {
    return b.config_;
  }
  static void RunInitialize(DunePUCTISMCTSBot& b, DuneISMCTSNode* n, const State& s) {
    b.InitializePriorsAndValue(n, s);
  }
  static int GetOpponentPriorCacheSize(DunePUCTISMCTSBot& b) {
    return b.opponent_prior_cache_.size();
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
  MockEvaluator(const ActionsAndProbs& priors, const std::vector<double>& values, int sleep_ms = 0)
      : priors_(priors), values_(values), sleep_ms_(sleep_ms) {}
  
  std::vector<double> Evaluate(const State& state) override {
    if (sleep_ms_ > 0) {
      std::this_thread::sleep_for(std::chrono::milliseconds(sleep_ms_));
    }
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
  int sleep_ms_;
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
  DunePUCTISMCTSBot bot(42, evaluator, 1.0, 50, -1, 1.0, 0.0, 0.3, 1.0);

  DuneISMCTSNode node;
  node.child_info[legal_actions[0]] = DuneChildInfo{0, 0.0, 0.9};
  node.child_info[legal_actions[1]] = DuneChildInfo{0, 0.0, 0.1};
  node.priors_initialized = true;

  // Initial selection: both unvisited, choose the one with the highest prior
  Action selected = TestBotAccessor::RunSelectTree(bot, &node, {legal_actions[0], legal_actions[1]});
  assert(selected == legal_actions[0]);

  // Give a visit to legal_actions[0]. Under FPU, since legal_actions[1] has a very low prior (0.1)
  // and legal_actions[0] has a high prior (0.9), it should continue to select legal_actions[0]
  // if its value is decent.
  node.total_visits = 1;
  node.child_info[legal_actions[0]].visits = 1;
  node.child_info[legal_actions[0]].return_sum = 0.5;

  selected = TestBotAccessor::RunSelectTree(bot, &node, {legal_actions[0], legal_actions[1]});
  assert(selected == legal_actions[0]);

  // Now let's change priors so the unvisited action has a high prior.
  node.child_info[legal_actions[0]].prior = 0.1;
  node.child_info[legal_actions[1]].prior = 0.9;
  selected = TestBotAccessor::RunSelectTree(bot, &node, {legal_actions[0], legal_actions[1]});
  assert(selected == legal_actions[1]); // Chooses unvisited because of its high prior

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

  ActionsAndProbs policy = bot.RunSearch(*state).policy;
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

// Test 8: Canonical Observation and InfoState Sorting
void TestCanonicalObservationSorting() {
  std::cout << "Running Test 8: Canonical Observation Sorting...\n";
  std::shared_ptr<const Game> game = LoadGame("dune_imperium");
  std::unique_ptr<State> state1 = game->NewInitialState();
  while (state1->IsChanceNode()) {
    state1->ApplyAction(state1->ChanceOutcomes().front().first);
  }
  std::unique_ptr<State> state2 = state1->Clone();

  auto* dstate1 = static_cast<dune_imperium::DuneImperiumState*>(state1.get());
  auto* dstate2 = static_cast<dune_imperium::DuneImperiumState*>(state2.get());

  // Setup hands with different orderings but same elements
  dstate1->SetPlayerHandForTesting(0, {1, 2, 3});
  dstate2->SetPlayerHandForTesting(0, {3, 1, 2});

  dstate1->SetPlayerDeckForTesting(0, {10, 20});
  dstate2->SetPlayerDeckForTesting(0, {20, 10});

  dstate1->SetPlayerIntriguesForTesting(0, {5, 6});
  dstate2->SetPlayerIntriguesForTesting(0, {6, 5});

  // Observations and InfoState strings must be exactly identical
  assert(dstate1->ObservationString(0) == dstate2->ObservationString(0));
  assert(dstate1->InformationStateString(0) == dstate2->InformationStateString(0));

  std::cout << "Test 8 Passed!\n\n";
}

// Test 9: Opponent Model Path MCTS Search
void TestOpponentModelPath() {
  std::cout << "Running Test 9: Opponent Model Path...\n";
  std::shared_ptr<const Game> game = LoadGame("dune_imperium");
  std::unique_ptr<State> state = game->NewInitialState();
  while (state->IsChanceNode()) {
    state->ApplyAction(state->ChanceOutcomes().front().first);
  }

  std::vector<Action> legal_actions = state->LegalActions();
  assert(legal_actions.size() > 1);

  // Mock priors and values
  ActionsAndProbs mock_priors;
  for (Action a : legal_actions) {
    mock_priors.push_back({a, 1.0 / legal_actions.size()});
  }
  std::vector<double> mock_values = {0.2, 0.3, 0.4, 0.1};

  auto evaluator = std::make_shared<MockEvaluator>(mock_priors, mock_values);
  // Instantiate MCTS bot with use_opponent_model = true
  DunePUCTISMCTSBot bot(
      /*seed=*/42,
      evaluator,
      /*puct_c=*/1.0,
      /*max_simulations=*/10,
      /*max_world_samples=*/-1,
      /*temperature=*/1.0,
      /*dirichlet_epsilon=*/0.0,
      /*dirichlet_alpha=*/0.3,
      /*utility_divisor=*/1.0,
      /*use_observation_string=*/true,
      DuneISMCTSFinalPolicyType::kNormalizedVisitCount,
      /*use_opponent_model=*/true,
      /*opponent_temperature=*/0.0,
      /*verbose_diagnostics=*/false
  );

  ActionsAndProbs policy = bot.RunSearch(*state).policy;
  assert(!policy.empty());

  std::cout << "Test 9 Passed!\n\n";
}

// Test 10: Hundro Coherence Resampling
void TestHundroCoherenceResampling() {
  std::cout << "Running Test 10: Hundro Coherence Resampling...\n";
  std::shared_ptr<const Game> game = LoadGame("dune_imperium");
  std::unique_ptr<State> state = game->NewInitialState();
  while (state->IsChanceNode()) {
    state->ApplyAction(state->ChanceOutcomes().front().first);
  }

  auto* dstate = static_cast<dune_imperium::DuneImperiumState*>(state.get());

  // Setup Hundro player to Player 1, and search player (player_id) to Player 0
  dstate->SetHundroPlayerForTesting(1);
  dstate->SetLeaderForTesting(1, static_cast<int>(dune_imperium::LeaderId::kHundroMoritani));

  std::mt19937 rng(42);
  auto sampler = [&rng]() { return absl::Uniform(rng, 0.0, 1.0); };

  // Resample from Player 0's perspective (searcher who is NOT Hundro)
  auto resampled = dstate->ResampleFromInfostate(0, sampler);
  auto* dresampled = static_cast<dune_imperium::DuneImperiumState*>(resampled.get());

  // Since the searcher is Player 0 (not Hundro), all entries in hundro_known_drawn_intrigue_ must be cleared to kInvalidCard
  for (int p = 0; p < 4; ++p) {
    assert(dresampled->GetHundroKnownDrawnIntrigueForTesting(p) == dune_imperium::kInvalidCard);
  }

  std::cout << "Test 10 Passed!\n\n";
}

// Test 11: PUCT + FPU Action Pruning (Regression test)
void TestPUCTFPUActionPruning() {
  std::cout << "Running Test 11: PUCT + FPU Action Pruning...\n";
  std::shared_ptr<const Game> game = LoadGame("dune_imperium");
  std::unique_ptr<State> state = game->NewInitialState();
  while (state->IsChanceNode()) {
    state->ApplyAction(state->ChanceOutcomes().front().first);
  }

  std::vector<Action> legal_actions = state->LegalActions();
  assert(legal_actions.size() > 2);

  ActionsAndProbs mock_priors;
  for (Action a : legal_actions) {
    mock_priors.push_back({a, 1e-5}); // default low prior
  }
  // Make action 0 and action 1 have custom priors
  mock_priors[0] = {legal_actions[0], 0.2};
  mock_priors[1] = {legal_actions[1], 0.8};
  std::vector<double> mock_values = {0.0, 0.0, 0.0, 0.0};

  auto evaluator = std::make_shared<MockEvaluator>(mock_priors, mock_values);
  DunePUCTISMCTSBot bot(42, evaluator, 1.0, 50, -1, 1.0, 0.0, 0.3, 1.0);

  DuneISMCTSNode node;
  node.priors_initialized = true;
  for (Action a : legal_actions) {
    node.child_info[a] = DuneChildInfo{0, 0.0, 1e-5};
  }
  node.child_info[legal_actions[0]] = DuneChildInfo{0, 0.0, 0.2};
  node.child_info[legal_actions[1]] = DuneChildInfo{0, 0.0, 0.8};

  // Step 1: Initial visit is done (node total visits is 0)
  // Run SelectTree. It should select legal_actions[1] because it has the highest prior (0.8).
  Action selected = TestBotAccessor::RunSelectTree(bot, &node, legal_actions);
  assert(selected == legal_actions[1]);

  // Step 2: Now we simulate that legal_actions[1] has been visited once and has return 1.0.
  // Total visits is 1.
  node.total_visits = 1;
  node.child_info[legal_actions[1]].visits = 1;
  node.child_info[legal_actions[1]].return_sum = 1.0;

  // Under PUCT + FPU:
  // fpu_val = 1.0 / 1 = 1.0.
  // For unvisited actions: q_val = fpu_val = 1.0.
  // For legal_actions[0] (unvisited, prior 0.2): puct = 1.0 + 1.0 * 0.2 * sqrt(1) / 1 = 1.2.
  // For legal_actions[1] (visited once, prior 0.8, value 1.0): puct = 1.0 + 1.0 * 0.8 * sqrt(1) / 2 = 1.4.
  // For other low-prior unvisited actions (prior 1e-5): puct = 1.0 + 1e-5.
  // So legal_actions[1] should be selected again! The low-prior unvisited actions are pruned!
  selected = TestBotAccessor::RunSelectTree(bot, &node, legal_actions);
  assert(selected == legal_actions[1]);

  std::cout << "Test 11 Passed!\n\n";
}

void TestFPUCaching() {
  std::cout << "Running Test 12: FPU Value Caching...\n";
  std::shared_ptr<const Game> game = LoadGame("dune_imperium");
  std::unique_ptr<State> state = game->NewInitialState();
  while (state->IsChanceNode()) {
    state->ApplyAction(state->ChanceOutcomes().front().first);
  }

  std::vector<Action> legal_actions = state->LegalActions();

  ActionsAndProbs mock_priors;
  for (Action a : legal_actions) {
    mock_priors.push_back({a, 1.0 / legal_actions.size()});
  }
  std::vector<double> mock_values(state->NumPlayers(), 0.5);
  mock_values[state->CurrentPlayer()] = 0.75;

  auto evaluator = std::make_shared<MockEvaluator>(mock_priors, mock_values);
  DunePUCTISMCTSBot bot(42, evaluator, 1.0, 50, -1, 1.0, 0.0, 0.3, 1.0);

  bot.RunSearch(*state);
  
  DuneISMCTSNode* root_node = TestBotAccessor::GetRootNode(bot);
  assert(root_node != nullptr);
  AssertAlmostEqual(root_node->cached_value, 0.75);

  std::cout << "Test 12 Passed!\n\n";
}

void TestSearchDiagnostics() {
  std::cout << "Running Test 13: Search Diagnostics...\n";
  std::shared_ptr<const Game> game = LoadGame("dune_imperium");
  std::unique_ptr<State> state = game->NewInitialState();
  while (state->IsChanceNode()) {
    state->ApplyAction(state->ChanceOutcomes().front().first);
  }

  std::vector<Action> legal_actions = state->LegalActions();
  ActionsAndProbs mock_priors;
  for (Action a : legal_actions) {
    mock_priors.push_back({a, 1.0 / legal_actions.size()});
  }
  std::vector<double> mock_values(state->NumPlayers(), 0.5);

  auto evaluator = std::make_shared<MockEvaluator>(mock_priors, mock_values);
  DunePUCTISMCTSBot bot(42, evaluator, 1.0, 50, -1, 1.0, 0.0, 0.3, 1.0);

  bot.RunSearch(*state);
  SearchDiagnostics diag = bot.GetRootDiagnostics(*state, 2);

  assert(diag.actions.size() == legal_actions.size());
  assert(diag.visit_counts.size() == legal_actions.size());
  assert(diag.q_values.size() == legal_actions.size());
  assert(diag.priors.size() == legal_actions.size());
  AssertAlmostEqual(diag.root_value, 0.5);

  std::cout << "Test 13 Passed!\n\n";
}

void TestHiddenInformationInvariance() {
  std::cout << "Running Test 14: Hidden Information Invariance...\n";
  std::shared_ptr<const Game> game = LoadGame("dune_imperium");
  std::unique_ptr<State> state1 = game->NewInitialState();
  while (state1->IsChanceNode()) {
    state1->ApplyAction(state1->ChanceOutcomes().front().first);
  }
  std::unique_ptr<State> state2 = state1->Clone();

  auto* dstate1 = static_cast<dune_imperium::DuneImperiumState*>(state1.get());
  auto* dstate2 = static_cast<dune_imperium::DuneImperiumState*>(state2.get());

  // Player 0 (searcher) has same state
  dstate1->SetPlayerHandForTesting(0, {1, 2, 3});
  dstate2->SetPlayerHandForTesting(0, {1, 2, 3});

  // Opponents have different private information
  dstate1->SetPlayerHandForTesting(1, {4, 5});
  dstate2->SetPlayerHandForTesting(1, {6, 7});

  dstate1->SetPlayerIntriguesForTesting(2, {10});
  dstate2->SetPlayerIntriguesForTesting(2, {20});

  // Assert that player 0 information states are identical
  assert(dstate1->InformationStateString(0) == dstate2->InformationStateString(0));

  std::vector<Action> legal_actions = state1->LegalActions();
  ActionsAndProbs mock_priors;
  for (Action a : legal_actions) {
    mock_priors.push_back({a, 1.0 / legal_actions.size()});
  }
  std::vector<double> mock_values = {0.5, 0.5, 0.5, 0.5};
  auto evaluator = std::make_shared<MockEvaluator>(mock_priors, mock_values);

  // Set up bots with the same config seed
  DuneSearchConfig config;
  config.seed = 999;
  config.max_simulations = 20;

  DunePUCTISMCTSBot bot1(config, evaluator);
  DunePUCTISMCTSBot bot2(config, evaluator);

  DuneSearchResult res1 = bot1.RunSearch(*state1);
  DuneSearchResult res2 = bot2.RunSearch(*state2);

  // Assert that the resulting policy and visits are exactly identical!
  assert(res1.policy.size() == res2.policy.size());
  for (size_t i = 0; i < res1.policy.size(); ++i) {
    assert(res1.policy[i].first == res2.policy[i].first);
    AssertAlmostEqual(res1.policy[i].second, res2.policy[i].second);
  }

  assert(res1.diagnostics.visit_counts == res2.diagnostics.visit_counts);
  assert(res1.diagnostics.q_values == res2.diagnostics.q_values);
  assert(res1.diagnostics.root_value == res2.diagnostics.root_value);

  std::cout << "Test 14 Passed!\n\n";
}

void TestCoverageFallback() {
  std::cout << "Running Test 15: Coverage Fallback...\n";
  std::shared_ptr<const Game> game = LoadGame("dune_imperium");
  std::unique_ptr<State> state = game->NewInitialState();
  while (state->IsChanceNode()) {
    state->ApplyAction(state->ChanceOutcomes().front().first);
  }

  std::vector<Action> legal_actions = state->LegalActions();
  ActionsAndProbs mock_priors;
  for (Action a : legal_actions) {
    mock_priors.push_back({a, 1.0 / legal_actions.size()});
  }
  std::vector<double> mock_values = {0.5, 0.5, 0.5, 0.5};
  auto evaluator = std::make_shared<MockEvaluator>(mock_priors, mock_values);

  DuneSearchConfig config;
  config.max_simulations = 2; // Very few simulations, won't cover min(3, legal_actions) actions with visits >= 2
  config.min_visit_threshold = 2;
  config.covered_prior_threshold = 0.5;

  DunePUCTISMCTSBot bot(config, evaluator);
  DuneSearchResult res = bot.RunSearch(*state);

  // Verify that fallback is triggered
  assert(res.fallback_reason == "low_coverage");
  // Policy should be the fallback prior policy (equal to mock_priors here)
  assert(res.policy.size() == legal_actions.size());
  for (size_t i = 0; i < res.policy.size(); ++i) {
    AssertAlmostEqual(res.policy[i].second, mock_priors[i].second);
  }

  std::cout << "Test 15 Passed!\n\n";
}

void TestTimeoutNodeLimits() {
  std::cout << "Running Test 16: Timeout and Node Limits...\n";
  std::shared_ptr<const Game> game = LoadGame("dune_imperium");
  std::unique_ptr<State> state = game->NewInitialState();
  while (state->IsChanceNode()) {
    state->ApplyAction(state->ChanceOutcomes().front().first);
  }

  std::vector<Action> legal_actions = state->LegalActions();
  ActionsAndProbs mock_priors;
  for (Action a : legal_actions) {
    mock_priors.push_back({a, 1.0 / legal_actions.size()});
  }
  std::vector<double> mock_values = {0.5, 0.5, 0.5, 0.5};
  auto evaluator = std::make_shared<MockEvaluator>(mock_priors, mock_values);

  // 1. Timeout Check
  DuneSearchConfig config_timeout;
  config_timeout.max_simulations = 1000;
  config_timeout.relative_time_budget_ms = 0.0; // Instant timeout!

  DunePUCTISMCTSBot bot_timeout(config_timeout, evaluator);
  DuneSearchResult res_timeout = bot_timeout.RunSearch(*state);
  assert(res_timeout.timeout_status == true);
  assert(res_timeout.fallback_reason == "timeout" || res_timeout.fallback_reason == "low_coverage");

  // 2. Node Limit Check
  DuneSearchConfig config_nodes;
  config_nodes.max_simulations = 1000;
  config_nodes.max_nodes = 1; // Limit pool size to 1 node

  DunePUCTISMCTSBot bot_nodes(config_nodes, evaluator);
  DuneSearchResult res_nodes = bot_nodes.RunSearch(*state);
  assert(res_nodes.fallback_reason == "max_nodes" || res_nodes.fallback_reason == "low_coverage");

  std::cout << "Test 16 Passed!\n\n";
}

void TestLeafTerminalScaleEquivalence() {
  std::cout << "Running Test 17: Leaf and Terminal Scale Equivalence...\n";
  std::shared_ptr<const Game> game = LoadGame("dune_imperium");
  std::unique_ptr<State> state = game->NewInitialState();
  while (state->IsChanceNode()) {
    state->ApplyAction(state->ChanceOutcomes().front().first);
  }

  std::vector<Action> legal_actions = state->LegalActions();
  ActionsAndProbs mock_priors;
  for (Action a : legal_actions) {
    mock_priors.push_back({a, 1.0 / legal_actions.size()});
  }
  std::vector<double> mock_values = {-0.5, 0.2, 0.8, -0.1};
  auto evaluator = std::make_shared<MockEvaluator>(mock_priors, mock_values);

  DuneSearchConfig config;
  config.max_simulations = 10;
  config.utility_divisor = 4.0;

  DunePUCTISMCTSBot bot(config, evaluator);

  // 1. Verify Leaf evaluations are unscaled (remain in [-1, 1] range as evaluated by network)
  DuneISMCTSNode node;
  TestBotAccessor::RunInitialize(bot, &node, *state);
  for (int p = 0; p < 4; ++p) {
    AssertAlmostEqual(node.cached_values[p], mock_values[p]);
  }

  // 2. Verify Terminal returns are scaled (divided by utility_divisor)
  std::unique_ptr<State> term_state = game->NewInitialState();
  std::mt19937 rng(42);
  while (!term_state->IsTerminal()) {
    if (term_state->IsChanceNode()) {
      term_state->ApplyAction(term_state->ChanceOutcomes().front().first);
    } else {
      std::vector<Action> actions = term_state->LegalActions();
      term_state->ApplyAction(actions[absl::Uniform(rng, 0u, actions.size())]);
    }
  }

  std::vector<double> sim_returns = TestBotAccessor::Simulate(bot, term_state.get());
  std::vector<double> real_returns = term_state->Returns();
  for (size_t i = 0; i < real_returns.size(); ++i) {
    AssertAlmostEqual(sim_returns[i], real_returns[i] / config.utility_divisor);
  }

  std::cout << "Test 17 Passed!\n\n";
}

class RoutingMockEvaluator : public MockEvaluator {
 public:
  using MockEvaluator::MockEvaluator;

  std::pair<ActionsAndProbs, std::vector<double>> PriorAndEvaluate(const State& state) override {
    std::vector<double> prior_vals = Evaluate(state);
    for (double& v : prior_vals) {
      if (v != 0.0) v += 100.0;
    }
    return {Prior(state), prior_vals};
  }
};

void TestPerSeatEvaluatorRouting() {
  std::cout << "Running Test 18: Per-Seat Evaluator Routing...\n";
  std::shared_ptr<const Game> game = LoadGame("dune_imperium");
  std::unique_ptr<State> state = game->NewInitialState();
  while (state->IsChanceNode()) {
    state->ApplyAction(state->ChanceOutcomes().front().first);
  }

  std::vector<Action> legal_actions = state->LegalActions();
  ActionsAndProbs mock_priors;
  for (Action a : legal_actions) {
    mock_priors.push_back({a, 1.0 / legal_actions.size()});
  }

  std::vector<std::shared_ptr<algorithms::Evaluator>> evaluators(4);
  for (int p = 0; p < 4; ++p) {
    std::vector<double> vals(4, 0.0);
    vals[p] = 10.0 + p;
    evaluators[p] = std::make_shared<RoutingMockEvaluator>(mock_priors, vals);
  }

  DuneSearchConfig config;
  config.max_simulations = 10;
  DunePUCTISMCTSBot bot(config, evaluators);

  DuneISMCTSNode node;
  TestBotAccessor::RunInitialize(bot, &node, *state);

  Player cur_player = state->CurrentPlayer();
  for (int p = 0; p < 4; ++p) {
    if (p == cur_player) {
      AssertAlmostEqual(node.cached_values[p], 110.0 + p);
    } else {
      AssertAlmostEqual(node.cached_values[p], 10.0 + p);
    }
  }

  std::cout << "Test 18 Passed!\n\n";
}

class CachingTestMockEvaluator : public MockEvaluator {
 public:
  mutable int prior_calls = 0;
  CachingTestMockEvaluator(const ActionsAndProbs& priors, const std::vector<double>& values)
      : MockEvaluator(priors, values) {}

  ActionsAndProbs Prior(const State& state) override {
    prior_calls++;
    return MockEvaluator::Prior(state);
  }
};

void TestOpponentPriorCache() {
  std::cout << "Running Test 19: Opponent Prior Cache...\n";
  std::shared_ptr<const Game> game = LoadGame("dune_imperium");
  std::unique_ptr<State> state = game->NewInitialState();
  while (state->IsChanceNode()) {
    state->ApplyAction(state->ChanceOutcomes().front().first);
  }

  std::vector<Action> legal_actions = state->LegalActions();
  ActionsAndProbs mock_priors;
  for (Action a : legal_actions) {
    mock_priors.push_back({a, 1.0 / legal_actions.size()});
  }
  std::vector<double> mock_values = {0.1, 0.2, 0.3, 0.4};

  auto evaluator = std::make_shared<CachingTestMockEvaluator>(mock_priors, mock_values);

  DuneSearchConfig config;
  config.max_simulations = 50;
  config.opponent_mode = SearchOpponentMode::kPolicy;
  config.opponent_temperature = 1.0;
  config.check_strategic_state = false; // Force search

  DunePUCTISMCTSBot bot(config, evaluator);

  // Run search
  bot.RunSearch(*state);

  // Verify that the evaluator's Prior was called
  int initial_calls = evaluator->prior_calls;
  assert(initial_calls > 0);

  // The cache size should match the number of unique opponent state-player pairs encountered
  int cache_size = TestBotAccessor::GetOpponentPriorCacheSize(bot);
  assert(cache_size > 0);

  // Since MCTS did 50 simulations, some opponent states would be encountered repeatedly.
  // The cache should have prevented redundant neural network Prior evaluations, so:
  // prior_calls should equal the number of cache entries.
  assert(evaluator->prior_calls == cache_size);

  // Now, run search again on the same state.
  // The cache should be cleared at the start of RunSearch() to prevent stale information.
  bot.RunSearch(*state);

  // The cache should have been cleared and repopulated, so:
  int new_cache_size = TestBotAccessor::GetOpponentPriorCacheSize(bot);
  assert(new_cache_size > 0);

  std::cout << "Test 19 Passed!\n\n";
}

void TestTimeoutWithAdequateCoverage() {
  std::cout << "Running Test 20: Timeout with Adequate Coverage...\n";
  std::shared_ptr<const Game> game = LoadGame("dune_imperium");
  std::unique_ptr<State> state = game->NewInitialState();
  while (state->IsChanceNode()) {
    state->ApplyAction(state->ChanceOutcomes().front().first);
  }

  std::vector<Action> legal_actions = state->LegalActions();
  ActionsAndProbs mock_priors;
  for (Action a : legal_actions) {
    mock_priors.push_back({a, 1.0 / legal_actions.size()});
  }
  std::vector<double> mock_values = {0.5, 0.5, 0.5, 0.5};

  // Set up evaluator that sleeps for 20 ms per inference query
  auto evaluator = std::make_shared<MockEvaluator>(mock_priors, mock_values, 20);

  DuneSearchConfig config;
  config.max_simulations = 100;
  config.relative_time_budget_ms = 50.0; // allows time for about 2 simulations
  config.min_visit_threshold = 1;
  config.covered_prior_threshold = 0.1; // very easy to cover

  DunePUCTISMCTSBot bot(config, evaluator);
  DuneSearchResult res = bot.RunSearch(*state);

  assert(res.timeout_status == true);
  assert(res.fallback_reason == "timeout");
  assert(res.used_fallback == false); // coverage passed, so we did NOT fall back to prior!
  assert(res.policy.size() == legal_actions.size());

  std::cout << "Test 20 Passed!\n\n";
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
  open_spiel::TestCanonicalObservationSorting();
  open_spiel::TestOpponentModelPath();
  open_spiel::TestHundroCoherenceResampling();
  open_spiel::TestPUCTFPUActionPruning();
  open_spiel::TestFPUCaching();
  open_spiel::TestSearchDiagnostics();
  open_spiel::TestHiddenInformationInvariance();
  open_spiel::TestCoverageFallback();
  open_spiel::TestTimeoutNodeLimits();
  open_spiel::TestLeafTerminalScaleEquivalence();
  open_spiel::TestPerSeatEvaluatorRouting();
  open_spiel::TestOpponentPriorCache();
  open_spiel::TestTimeoutWithAdequateCoverage();
  std::cout << "All Dune PUCT IS-MCTS tests completed successfully!\n";
  return 0;
}
