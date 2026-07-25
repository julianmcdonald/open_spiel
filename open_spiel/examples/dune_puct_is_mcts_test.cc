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
#include "dune_search_routing.h"
#include "dune_search_session.h"
#include "open_spiel/abseil-cpp/absl/random/distributions.h"
#include "open_spiel/utils/json.h"

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
    return b.RunSimulation(s, 0, 0, 0);
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
  static const absl::flat_hash_map<std::pair<Player, std::string>, DuneISMCTSNode*>& GetNodes(DunePUCTISMCTSBot& b) {
    return b.nodes_;
  }
  static DuneISMCTSNode* CallLookupNode(DunePUCTISMCTSBot& b, const State& s) {
    return b.LookupNode(s);
  }
  static void InjectMockNode(DunePUCTISMCTSBot& b, const State& s, int total_visits) {
    auto key = b.GetStateKey(s);
    auto node = std::make_unique<DuneISMCTSNode>();
    node->total_visits = total_visits;
    b.nodes_[key] = node.get();
    b.node_pool_.push_back(std::move(node));
  }
  static void ClearNodes(DunePUCTISMCTSBot& b) {
    b.nodes_.clear();
    b.node_pool_.clear();
  }
  static ActionsAndProbs RunGetFinalPolicy(DunePUCTISMCTSBot& b, const State& s,
                                           DuneISMCTSNode* n) {
    return b.GetFinalPolicy(s, n);
  }
  static ActionsAndProbs RunFilterRawPriors(DunePUCTISMCTSBot& b, DuneISMCTSNode* node,
                                            const std::vector<Action>& legals) {
    return b.FilterAndNormalizeRawPriors(node, legals);
  }
};

namespace {

DuneSearchResult RunSessionSearch(DuneSearchSession& session, const State& state, double remaining_time_ms = -1.0) {
  DuneSearchResult res = session.Search(state, remaining_time_ms);
  double r_val = 0.5;
  ControllerDecision dec = session.SelectControllerAction(state, res, r_val);
  return session.CommitAction(dec);
}

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

// Test 1: Key uniqueness, and that the key carries NO path-derived field.
//
// This test used to assert that ObservationString CONTAINS "|history_size=".
// That pinned the wrong contract: history length is a property of the path
// taken, not of the position, so two identical frozen positions reached in a
// different number of moves received different transposition keys and never
// shared search statistics. WO-31 removed it; the assertion is now inverted.
// Key uniqueness itself — the real point of this test — is unchanged and does
// not depend on history_size: applying an action moves the position.
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
  assert(obs1.find("history_size") == std::string::npos);

  // Take a legal action; the position — and therefore the key — must change.
  std::vector<Action> actions = state->LegalActions();
  state->ApplyAction(actions.front());

  std::string obs2 = state->ObservationString(state->CurrentPlayer());
  assert(obs2.find("history_size") == std::string::npos);
  assert(obs1 != obs2);

  // The InformationStateString is NOT the transposition key and legitimately
  // keeps history_size in its body, outside the shared |obs= tail.
  const std::string info = state->InformationStateString(state->CurrentPlayer());
  assert(info.find("|history_size=") != std::string::npos);

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

  DuneSearchResult result = bot.RunSearch(*state);
  assert(!result.policy.empty());

  DuneISMCTSNode* root = TestBotAccessor::GetRootNode(bot);
  assert(root != nullptr);
  assert(root->total_visits > 0);

  // Check that the sum of returns matches the evaluation from the evaluator.
  //
  // WO-15 / search finding 7: ALL 10 simulations descend, so 10 * 0.8 = 8.0.
  // This used to be 7.2, because the root was left unexpanded (total_visits
  // == -1) until simulation 0 re-reached it, flipped it to 0 and returned the
  // cached value without descending — a silent budget-1. The advertised budget
  // and the work actually done now agree.
  double sum_q = 0.0;
  for (const auto& action_child : root->child_info) {
    sum_q += action_child.second.return_sum;
  }
  AssertAlmostEqual(sum_q, 8.0, 1e-4);

  // The same accounting, stated directly: an N-simulation cold root performs N
  // root descents, and every one of them is reported.
  assert(result.simulations_completed == 10);
  assert(root->total_visits == 10);
  int summed_child_visits = 0;
  for (const auto& action_child : root->child_info) {
    summed_child_visits += action_child.second.visits;
  }
  assert(summed_child_visits == 10);

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

  DuneSearchResult search_result = bot.RunSearch(*state);
  SearchDiagnostics diag = bot.GetRootDiagnostics(*state, 2);

  assert(diag.actions.size() == legal_actions.size());
  assert(diag.visit_counts.size() == legal_actions.size());
  assert(diag.q_values.size() == legal_actions.size());
  assert(diag.priors.size() == legal_actions.size());
  AssertAlmostEqual(diag.root_value, 0.5);
  assert(!search_result.diagnostics.sampled_leaf_states.empty());
  assert(search_result.diagnostics.sampled_leaf_states.size() <= 16);
  for (const auto& leaf : search_result.diagnostics.sampled_leaf_states) {
    assert(leaf != nullptr);
    if (!leaf->IsTerminal() && !leaf->IsChanceNode()) {
      assert(!leaf->LegalActions().empty());
    }
  }

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

  auto search_evaluator = std::make_shared<CachingTestMockEvaluator>(mock_priors, mock_values);
  auto opponent_evaluator = std::make_shared<CachingTestMockEvaluator>(mock_priors, mock_values);
  std::vector<std::shared_ptr<algorithms::Evaluator>> evaluators(4, opponent_evaluator);
  evaluators[state->CurrentPlayer()] = search_evaluator;

  DuneSearchConfig config;
  config.max_simulations = 50;
  config.opponent_mode = SearchOpponentMode::kPolicy;
  config.opponent_temperature = 1.0;
  config.check_strategic_state = false; // Force search

  DunePUCTISMCTSBot bot(config, evaluators);

  // Run search
  bot.RunSearch(*state);

  // Verify that the evaluator's Prior was called
  int initial_calls = opponent_evaluator->prior_calls;
  assert(initial_calls > 0);

  // The cache size should match the number of unique opponent state-player pairs encountered
  int cache_size = TestBotAccessor::GetOpponentPriorCacheSize(bot);
  assert(cache_size > 0);

  // Since MCTS did 50 simulations, some opponent states would be encountered repeatedly.
  // The cache should have prevented redundant neural network Prior evaluations, so:
  assert(opponent_evaluator->prior_calls == cache_size);

  // Now, run search again on the same state.
  // The cache should be cleared at the start of RunSearch() to prevent stale information.
  opponent_evaluator->prior_calls = 0;
  bot.RunSearch(*state);

  // The cache should have been cleared and repopulated, so:
  int new_cache_size = TestBotAccessor::GetOpponentPriorCacheSize(bot);
  assert(new_cache_size > 0);
  assert(opponent_evaluator->prior_calls == new_cache_size);

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

  // Set up evaluator that sleeps for 10 ms per inference query
  auto evaluator = std::make_shared<MockEvaluator>(mock_priors, mock_values, 10);

  DuneSearchConfig config;
  config.max_simulations = 100;
  config.relative_time_budget_ms = 50.0; // allows time for about 4-5 simulations
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

class StateDependentMockEvaluator : public MockEvaluator {
 public:
  StateDependentMockEvaluator(const ActionsAndProbs& priors, const std::vector<double>& values)
      : MockEvaluator(priors, values) {}

  std::vector<double> Evaluate(const State& state) override {
    double sum = 0.0;
    for (Action a : state.History()) {
      sum += a;
    }
    std::vector<double> vals(4, 0.0);
    for (int p = 0; p < 4; ++p) {
      vals[p] = (sum - 30000.0) * 0.0001;
    }
    return vals;
  }
};

void TestOpponentTemperatureIsolation() {
  std::cout << "Running Test 21: Opponent Temperature Isolation...\n";
  std::shared_ptr<const Game> game = LoadGame("dune_imperium");
  std::unique_ptr<State> state = game->NewInitialState();
  while (state->IsChanceNode()) {
    state->ApplyAction(state->ChanceOutcomes().front().first);
  }

  ActionsAndProbs mock_priors;
  int num_actions = game->NumDistinctActions();
  for (int i = 0; i < num_actions; ++i) {
    mock_priors.push_back({i, 1.0 / num_actions});
  }
  std::vector<double> mock_values = {0.1, 0.2, 0.3, 0.4};

  auto evaluator = std::make_shared<StateDependentMockEvaluator>(mock_priors, mock_values);

  DuneSearchConfig config1;
  config1.max_simulations = 40;
  config1.puct_c = 5.0; // Encourage exploration to visit all actions
  config1.opponent_mode = SearchOpponentMode::kPolicy;
  config1.opponent_temperature = 0.0; // Greedy simulated opponents
  config1.check_strategic_state = false;
  config1.seed = 42;

  DunePUCTISMCTSBot bot1(config1, evaluator);
  DuneSearchConfig& c1 = TestBotAccessor::GetConfig(bot1);
  assert(c1.opponent_temperature == 0.0);
  DuneSearchResult res1 = bot1.RunSearch(*state);

  DuneSearchConfig config2;
  config2.max_simulations = 40;
  config2.puct_c = 5.0; // Encourage exploration to visit all actions
  config2.opponent_mode = SearchOpponentMode::kPolicy;
  config2.opponent_temperature = 5.0; // High temperature (stochastic)
  config2.check_strategic_state = false;
  config2.seed = 42; // Same seed!

  DunePUCTISMCTSBot bot2(config2, evaluator);
  DuneSearchConfig& c2 = TestBotAccessor::GetConfig(bot2);
  assert(c2.opponent_temperature == 5.0);
  DuneSearchResult res2 = bot2.RunSearch(*state);

  // Since greedy vs stochastic opponents explore different tree search paths,
  // the resulting visit distributions/policies should differ.
  bool policies_differ = false;
  for (size_t i = 0; i < res1.policy.size(); ++i) {
    if (std::abs(res1.policy[i].second - res2.policy[i].second) > 1e-4) {
      policies_differ = true;
      break;
    }
  }
  assert(policies_differ);

  std::cout << "Test 21 Passed!\n\n";
}

void TestCorrectedTerminalRounds() {
  std::cout << "Running Test 22: Corrected Terminal Rounds...\n";
  std::shared_ptr<const Game> game = LoadGame("dune_imperium");
  std::unique_ptr<State> state = game->NewInitialState();
  std::mt19937 rng(42);
  int last_round = 1;
  while (!state->IsTerminal()) {
    const dune_imperium::DuneImperiumState* dune_state =
        static_cast<const dune_imperium::DuneImperiumState*>(state.get());
    last_round = dune_state->GetCurrentRound();
    if (state->IsChanceNode()) {
      auto outcomes = state->ChanceOutcomes();
      state->ApplyAction(outcomes.front().first);
    } else {
      auto legal_actions = state->LegalActions();
      state->ApplyAction(legal_actions[absl::Uniform(rng, 0u, legal_actions.size())]);
    }
  }

  const dune_imperium::DuneImperiumState* dune_state =
      static_cast<const dune_imperium::DuneImperiumState*>(state.get());

  int raw_round = dune_state->GetCurrentRound();
  int corrected_round = raw_round - 1; // since the game is terminal

  // Verify that the engine's raw round has been incremented by one
  assert(raw_round == last_round + 1);
  // Verify that our corrected round perfectly matches the final active round
  assert(corrected_round == last_round);
  assert(corrected_round >= 1 && corrected_round <= 10);

  std::cout << "Test 22 Passed!\n\n";
}

void TestFixedSimulationCompleteness() {
  std::cout << "Running Test 23: Fixed Simulation Completeness Fallback...\n";
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

  DuneSearchConfig config_nodes;
  config_nodes.max_simulations = 100;
  config_nodes.max_nodes = 5; // extremely low node limit
  config_nodes.check_strategic_state = false;

  DunePUCTISMCTSBot bot_nodes(config_nodes, evaluator);
  DuneSearchResult res_nodes = bot_nodes.RunSearch(*state);

  assert(res_nodes.fallback_reason == "max_nodes");
  assert(res_nodes.simulations_completed < 100);
  assert(res_nodes.used_fallback == true);

  std::cout << "Test 23 Passed!\n\n";
}

void TestRootPriorTemperatureScaling() {
  std::cout << "Running Test 24: Root Prior Temperature Scaling...\n";
  std::shared_ptr<const Game> game = LoadGame("dune_imperium");
  std::unique_ptr<State> state = game->NewInitialState();
  while (state->IsChanceNode()) {
    state->ApplyAction(state->ChanceOutcomes().front().first);
  }

  std::vector<Action> legal_actions = state->LegalActions();
  assert(legal_actions.size() >= 2);

  // Set distinct priors
  ActionsAndProbs mock_priors;
  double sum = 0.0;
  for (size_t i = 0; i < legal_actions.size(); ++i) {
    double p = 0.1 + 0.1 * i;
    mock_priors.push_back({legal_actions[i], p});
    sum += p;
  }
  for (auto& ap : mock_priors) {
    ap.second /= sum;
  }

  std::vector<double> mock_values = {0.5, 0.5, 0.5, 0.5};
  auto evaluator = std::make_shared<MockEvaluator>(mock_priors, mock_values);

  DuneSearchConfig config;
  config.max_simulations = 2;
  config.root_prior_temperature = 2.0; // scale with temperature 2.0
  config.check_strategic_state = false;

  DunePUCTISMCTSBot bot(config, evaluator);
  DuneSearchResult res = bot.RunSearch(*state);

  DuneISMCTSNode* root = TestBotAccessor::GetRootNode(bot);
  assert(root != nullptr);

  // Calculate expected scaled priors dynamically based on what the evaluator actually returns for the state
  ActionsAndProbs raw_priors = evaluator->Prior(*state);
  double expected_sum = 0.0;
  std::vector<double> expected_p_T;
  for (const auto& ap : raw_priors) {
    double p = std::sqrt(ap.second);
    expected_p_T.push_back(p);
    expected_sum += p;
  }
  for (double& p : expected_p_T) {
    p /= expected_sum;
  }

  // Verify root priors match expectations
  for (size_t i = 0; i < raw_priors.size(); ++i) {
    auto child_it = root->child_info.find(raw_priors[i].first);
    assert(child_it != root->child_info.end());
    double actual_prior = child_it->second.prior;
    AssertAlmostEqual(actual_prior, expected_p_T[i]);
  }

  // Verify that child nodes do NOT have their priors scaled
  DuneISMCTSNode non_root;
  non_root.priors_initialized = false;
  TestBotAccessor::RunInitialize(bot, &non_root, *state);

  for (size_t i = 0; i < raw_priors.size(); ++i) {
    auto child_it = non_root.child_info.find(raw_priors[i].first);
    assert(child_it != non_root.child_info.end());
    AssertAlmostEqual(child_it->second.prior, raw_priors[i].second);
  }

  std::cout << "Test 24 Passed!\n\n";
}

class TestRoutingState : public dune_imperium::DuneImperiumState {
 public:
  TestRoutingState(std::shared_ptr<const Game> game)
      : dune_imperium::DuneImperiumState(game),
        mock_current_player_(0),
        mock_root_history_size_(0) {
    SetPhaseForTesting(dune_imperium::GamePhase::kAgentTurns);
    SetPlayerAgentsRemainingForTesting(0, 2);
    SetPendingImperiumSlotForTesting(dune_imperium::kInvalidCard);
    SetImperiumRowForTesting({5, 6, 7, 8, 9});
    SetPlayerSolariForTesting(0, 10);
    SetPlayerSpiceForTesting(0, 10);
    SetPlayerPersuasionForTesting(0, 10);
    for (int p = 0; p < 4; ++p) {
      SetPlayerHandForTesting(p, {0, 1, 2, 3, 4});
    }
  }

  std::vector<Action> LegalActions() const override {
    if (CurrentPlayer() == mock_current_player_ && History().size() == mock_root_history_size_) {
      return mock_legal_actions_;
    }
    return dune_imperium::DuneImperiumState::LegalActions();
  }

  void SetMockLegalActions(const std::vector<Action>& actions) {
    mock_legal_actions_ = actions;
    mock_root_history_size_ = History().size();
  }

  void SetMockCurrentPlayer(Player p) {
    mock_current_player_ = p;
    SetCurrentPlayerForTesting(p);
  }

  std::unique_ptr<State> Clone() const override {
    return std::make_unique<TestRoutingState>(*this);
  }

  std::unique_ptr<State> ResampleFromInfostate(Player player, std::function<double()> rng_prob) const override {
    return Clone();
  }

 private:
  std::vector<Action> mock_legal_actions_;
  Player mock_current_player_;
  int mock_root_history_size_;
};

void TestClassifierSearchRouting() {
  std::cout << "Running Test: Classifier Search Routing...\n";
  auto game = LoadGame("dune_imperium");
  TestRoutingState state(game);

  // 1. Forced/bookkeeping (size <= 1)
  state.SetMockLegalActions({1});
  assert(ClassifyDuneDecisionRole(state, 0, false) == DuneDecisionRole::kForcedOrBookkeeping);

  // 2. LeaderDraft phase
  state.SetPhaseForTesting(dune_imperium::GamePhase::kLeaderDraft);
  state.SetMockLegalActions({1, 2});
  assert(ClassifyDuneDecisionRole(state, 0, false) == DuneDecisionRole::kLeaderSelection);

  // 3. Purchase phase
  state.SetPhaseForTesting(dune_imperium::GamePhase::kAgentTurns);
  state.SetMockLegalActions({
    dune_imperium::kActionBuyImperiumRow0,
    dune_imperium::kActionBuyReserveArrakisLiaison
  });
  assert(ClassifyDuneDecisionRole(state, 0, false) == DuneDecisionRole::kPurchase);

  // 4. Combat Intrigue
  state.SetPhaseForTesting(dune_imperium::GamePhase::kCombat);
  state.SetMockLegalActions({
    dune_imperium::kActionCombatPass,
    dune_imperium::kActionPlayIntrigueCombatCard0 + 5
  });
  assert(ClassifyDuneDecisionRole(state, 0, false) == DuneDecisionRole::kCombatIntrigue);

  // 5. Agent Primary vs Continuation
  state.SetPhaseForTesting(dune_imperium::GamePhase::kAgentTurns);
  state.SetPlayerAgentsRemainingForTesting(0, 1);
  state.SetMockLegalActions({
    dune_imperium::kActionSelectAgentCard0 + 10,
    dune_imperium::kActionReveal
  });
  assert(ClassifyDuneDecisionRole(state, 0, false) == DuneDecisionRole::kAgentPrimary);

  state.SetMockLegalActions({
    dune_imperium::kActionAgentSpaceConspire,
    dune_imperium::kActionAgentSpaceHighCouncil
  });
  assert(ClassifyDuneDecisionRole(state, 0, true) == DuneDecisionRole::kAgentContinuation);

  // 6. Other Optional
  state.SetPlayerAgentsRemainingForTesting(0, 0);
  state.SetMockLegalActions({
    dune_imperium::kActionPlayIntriguePlotCard0 + 2,
    dune_imperium::kActionReveal
  });
  assert(ClassifyDuneDecisionRole(state, 0, false) == DuneDecisionRole::kOtherOptional);

  std::cout << "Test Classifier Search Routing Passed!\n\n";
}

void TestSearchSessionControls() {
  std::cout << "Running Test: Search Session Controls...\n";
  std::shared_ptr<const Game> game = LoadGame("dune_imperium");
  std::unique_ptr<State> state = game->NewInitialState();
  while (!state->IsTerminal()) {
    if (state->IsChanceNode()) {
      state->ApplyAction(state->ChanceOutcomes().front().first);
    } else {
      const dune_imperium::DuneImperiumState& dune_state =
          static_cast<const dune_imperium::DuneImperiumState&>(*state);
      if (dune_state.phase() == dune_imperium::GamePhase::kAgentTurns && state->LegalActions().size() > 1) {
        break;
      }
      state->ApplyAction(state->LegalActions().front());
    }
  }

  // Setup evaluator
  std::vector<Action> legal_actions = state->LegalActions();
  ActionsAndProbs mock_priors;
  for (Action a : legal_actions) {
    mock_priors.push_back({a, 1.0 / legal_actions.size()});
  }
  std::vector<double> mock_values = {0.5, 0.5, 0.5, 0.5};
  auto evaluator = std::make_shared<MockEvaluator>(mock_priors, mock_values);

  DuneSearchConfig config;
  config.max_simulations = 10;

  // 1. PolicyOnly mode
  {
    DuneSearchSession session(config, evaluator, DuneSearchBudgetMode::kPolicyOnly);
    DuneSearchResult res = RunSessionSearch(session, *state);
    std::cout << "DEBUG PolicyOnly fallback: " << res.fallback_reason << std::endl;
    assert(res.simulations_completed == 0);
    assert(res.used_fallback == true);
    assert(res.fallback_reason == "policy_only_mode");
  }

  // 2. FixedSessionSimulations mode
  {
    DuneSearchSession session(config, evaluator, DuneSearchBudgetMode::kFixedSessionSimulations);
    DuneSearchResult res = RunSessionSearch(session, *state);
    assert(session.HasActiveSession());
    assert(session.session_new_simulations_completed() > 0);
    int first_search_sims = session.session_new_simulations_completed();

    Action play_card_action = res.diagnostics.selected_action;
    assert(play_card_action != kInvalidAction);
    state->ApplyAction(play_card_action);

    // Now call Search on the continuation state
    DuneSearchResult res2 = RunSessionSearch(session, *state);
    assert(session.session_new_simulations_completed() == first_search_sims + res2.simulations_completed);
    assert(session.session_new_simulations_completed() <= 200);

    // Verify long-agent cumulative time is tracked and reset correctly
    double t1 = res.diagnostics.long_agent_session_cumulative_time_ms;
    double t2 = res2.diagnostics.long_agent_session_cumulative_time_ms;
    assert(t2 >= t1);

    session.ResetSession("test_reset");
    assert(session.session_elapsed_time_ms() == 0.0);

    std::unique_ptr<State> state_new = game->NewInitialState();
    while (state_new->IsChanceNode()) {
      state_new->ApplyAction(state_new->ChanceOutcomes().front().first);
    }
    DuneSearchResult res3 = RunSessionSearch(session, *state_new);
    assert(res3.diagnostics.long_agent_session_cumulative_time_ms == res3.elapsed_time_ms);
  }

  std::cout << "Test Search Session Controls Passed!\n\n";
}

void TestTransactionalMCTSAndTreeFallback() {
  std::cout << "Running Test 27: Transactional MCTS & Tree Fallback...\n";
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

  auto evaluator = std::make_shared<MockEvaluator>(mock_priors, mock_values, 20); // 20 ms latency
  DuneSearchConfig config;
  config.max_simulations = 100;
  config.relative_time_budget_ms = 35.0; // 35 ms budget

  DunePUCTISMCTSBot bot(config, evaluator);
  DuneSearchResult res = bot.RunSearch(*state);

  DuneISMCTSNode* root = TestBotAccessor::CallLookupNode(bot, *state);

  assert(res.timeout_status == true);
  assert(root != nullptr);
  // Transactional accounting: a simulation that aborts on the deadline rolls its
  // visits back, so every simulation REPORTED is a root descent that actually
  // happened. On a cold root that makes the two exactly equal (WO-15 / search
  // finding 7 — it used to be off by one, because the root's own expansion
  // consumed simulation 0 without descending). Asserted as an invariant rather
  // than as a fixed count, which only held for one particular interleaving of
  // the 20 ms mock latency against the 35 ms budget.
  assert(res.simulations_completed >= 1);
  assert(root->total_visits == res.simulations_completed);
  int child_visit_sum = 0;
  for (const auto& action_child : root->child_info) {
    child_visit_sum += action_child.second.visits;
  }
  assert(child_visit_sum == res.simulations_completed);

  // 2. Tree Fallback:
  DuneSearchConfig config2;
  config2.max_simulations = 10;
  config2.relative_time_budget_ms = 10000.0;
  config2.min_visit_threshold = 1;
  config2.covered_prior_threshold = 0.01;

  DunePUCTISMCTSBot bot2(config2, evaluator);
  DuneSearchResult res2 = bot2.RunSearch(*state);
  assert(res2.used_fallback == false);

  DuneSearchResult res3 = bot2.RunSearch(*state, 0, 10000.0);
  assert(res3.simulations_completed == 0);
  assert(res3.used_fallback == false); // retrieved from the tree!
  assert(res3.policy.size() == legal_actions.size());

  std::cout << "Test 27 Passed!\n\n";
}

void TestInvariantGatesAndScenarios() {
  std::cout << "Running TestInvariantGatesAndScenarios...\n";
  auto game = open_spiel::LoadGame("dune_imperium");
  auto state = game->NewInitialState();
  while (!state->IsTerminal()) {
    if (state->IsChanceNode()) {
      state->ApplyAction(state->ChanceOutcomes().front().first);
    } else {
      const dune_imperium::DuneImperiumState& dune_state =
          static_cast<const dune_imperium::DuneImperiumState&>(*state);
      if (dune_state.phase() == dune_imperium::GamePhase::kAgentTurns && state->LegalActions().size() > 1) {
        break;
      }
      state->ApplyAction(state->LegalActions().front());
    }
  }

  ActionsAndProbs priors;
  std::vector<Action> legal_actions = state->LegalActions();
  for (Action a : legal_actions) {
    priors.push_back({a, 1.0 / legal_actions.size()});
  }
  std::vector<double> mock_vals(4, 0.0);
  auto evaluator = std::make_shared<MockEvaluator>(priors, mock_vals);

  // 1. Branch hits/misses & Reset boundary checks
  {
    DuneSearchConfig config;
    config.max_simulations = 10;
    config.fixed_session_limit = 10;
    config.fixed_continuation_reserve = 0;
    config.purchase_combat_budget = 16;
    DuneSearchSession session(config, evaluator, DuneSearchBudgetMode::kFixedSessionSimulations);

    DuneSearchResult res1 = RunSessionSearch(session, *state);
    assert(session.last_re_root_status() == "none");

    // Test branch hit (apply the chosen/selected action)
    auto next_state_hit = state->Clone();
    Action act = res1.diagnostics.selected_action;
    next_state_hit->ApplyAction(act);
    DuneSearchResult res2 = RunSessionSearch(session, *next_state_hit);
    assert(session.last_re_root_status() == "hit");

    // Test branch miss (search on a sibling state that is not a descendant of current session root)
    DuneSearchSession session_miss(config, evaluator, DuneSearchBudgetMode::kFixedSessionSimulations);
    RunSessionSearch(session_miss, *state);

    auto state_a = state->Clone();
    state_a->ApplyAction(act);
    RunSessionSearch(session_miss, *state_a);
    assert(session_miss.last_re_root_status() == "hit");

    Action other_card_action = kInvalidAction;
    for (Action a : state->LegalActions()) {
      if (a >= dune_imperium::kActionSelectAgentCard0 && a < dune_imperium::kActionSelectAgentCard0 + 256 && a != act) {
        auto test_state = state->Clone();
        test_state->ApplyAction(a);
        while (test_state->IsChanceNode()) {
          test_state->ApplyAction(test_state->ChanceOutcomes().front().first);
        }
        if (test_state->LegalActions().size() > 1) {
          other_card_action = a;
          break;
        }
      }
    }
    assert(other_card_action != kInvalidAction);
    auto state_b = state->Clone();
    state_b->ApplyAction(other_card_action);
    while (state_b->IsChanceNode()) {
      state_b->ApplyAction(state_b->ChanceOutcomes().front().first);
    }

    RunSessionSearch(session_miss, *state_b);
    assert(session_miss.last_re_root_status() == "miss");

    // Test seat boundary boundary reset
    DuneSearchSession session_seat(config, evaluator, DuneSearchBudgetMode::kFixedSessionSimulations);
    DuneSearchResult r_primary = RunSessionSearch(session_seat, *state);

    auto state_seat = state->Clone();
    Action act_card = r_primary.diagnostics.selected_action;
    state_seat->ApplyAction(act_card);
    while (state_seat->IsChanceNode()) {
      state_seat->ApplyAction(state_seat->ChanceOutcomes().front().first);
    }

    DuneSearchResult r_cont = RunSessionSearch(session_seat, *state_seat);
    Action act_space = r_cont.diagnostics.selected_action;
    if (act_space != kInvalidAction) {
      state_seat->ApplyAction(act_space);
    }

    // Advance state until another real player's turn is reached (descendant history preserved)
    while (!state_seat->IsTerminal()) {
      if (state_seat->IsChanceNode()) {
        state_seat->ApplyAction(state_seat->ChanceOutcomes().front().first);
      } else {
        if (state_seat->CurrentPlayer() != state->CurrentPlayer()) {
          break;
        }
        state_seat->ApplyAction(state_seat->LegalActions().front());
      }
    }

    if (!state_seat->IsTerminal()) {
      DuneSearchResult res_seat = RunSessionSearch(session_seat, *state_seat);
      assert(res_seat.diagnostics.reset_reason == "seat_or_round_boundary");
    }
  }

  // 2. Exact 52-second deadlines & live continuation reserve
  {
    DuneSearchConfig config;
    config.max_simulations = 20;
    config.live_continuation_reserve_seconds = 10.0;

    Action act = kInvalidAction;
    for (Action a : state->LegalActions()) {
      if (a >= dune_imperium::kActionSelectAgentCard0 && a < dune_imperium::kActionSelectAgentCard0 + 256) {
        auto test_state = state->Clone();
        test_state->ApplyAction(a);
        while (test_state->IsChanceNode()) {
          test_state->ApplyAction(test_state->ChanceOutcomes().front().first);
        }
        if (test_state->LegalActions().size() > 1) {
          act = a;
          break;
        }
      }
    }
    assert(act != kInvalidAction);

    ActionsAndProbs live_priors;
    for (Action a : state->LegalActions()) {
      live_priors.push_back({a, a == act ? 1.0 : 0.0});
    }
    auto live_evaluator = std::make_shared<MockEvaluator>(live_priors, mock_vals);
    DuneSearchSession session(config, live_evaluator, DuneSearchBudgetMode::kLiveDeadline);

    DuneSearchResult res1 = RunSessionSearch(session, *state, 52000.0);
    assert(res1.diagnostics.hard_time_limit_ms == 52000.0);
    assert(res1.diagnostics.soft_time_limit_ms <= 42050.0);
    assert(res1.diagnostics.soft_time_limit_ms >= 41500.0);

    auto state_a = state->Clone();
    state_a->ApplyAction(res1.diagnostics.selected_action);
    while (state_a->IsChanceNode()) {
      state_a->ApplyAction(state_a->ChanceOutcomes().front().first);
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    DuneSearchResult res2 = RunSessionSearch(session, *state_a, -1.0);
    // Soft limit should subtract elapsed time (approx 100ms + search latency)
    assert(res2.diagnostics.soft_time_limit_ms < 51950.0);
    assert(res2.diagnostics.soft_time_limit_ms > 40000.0);
    // Continuation has zero reserve, so soft limit is much higher than primary (which subtracted 10s reserve)
    assert(res2.diagnostics.soft_time_limit_ms > 45000.0);
  }

  // 3. Full/fast selection & dynamic configuration updates
  {
    DuneSearchConfig config;
    config.fixed_continuation_reserve = 2;
    config.root_prior_temperature = 1.5;
    config.training_root_prior_temperature = 1.5;

    bool rolled_full = false;
    bool rolled_fast = false;

    for (int i = 0; i < 50; ++i) {
      DuneSearchSession session(config, evaluator, DuneSearchBudgetMode::kTrainingFullFast);
      session.SetEpisodeId(i);
      DuneSearchResult res = RunSessionSearch(session, *state);
      if (res.diagnostics.hard_sim_limit == 64) {
        rolled_full = true;
        assert(session.GetBot()->GetConfig().temperature == 1.0);
        assert(session.GetBot()->GetConfig().dirichlet_epsilon == 0.25);
        assert(session.GetBot()->GetConfig().root_prior_temperature == 1.5);
      } else if (res.diagnostics.hard_sim_limit == 8) {
        rolled_fast = true;
        assert(session.GetBot()->GetConfig().temperature == 0.0);
        assert(session.GetBot()->GetConfig().dirichlet_epsilon == 0.0);
        assert(session.GetBot()->GetConfig().root_prior_temperature == 1.0);
      }
    }
    assert(rolled_full && rolled_fast);
  }

  // 4. Tiny window budget sharing
  {
    DuneSearchConfig config;
    config.max_simulations = 100;
    config.purchase_combat_budget = 16;

    DuneSearchSession session(config, evaluator, DuneSearchBudgetMode::kFixedSessionSimulations);

    // Advance to reveal turns phase (where purchases are made)
    auto state_combat = state->Clone();
    while (!state_combat->IsTerminal()) {
      const dune_imperium::DuneImperiumState& ds = static_cast<const dune_imperium::DuneImperiumState&>(*state_combat);
      if (ds.phase() == dune_imperium::GamePhase::kRevealTurns) {
        break;
      }
      if (state_combat->IsChanceNode()) {
        state_combat->ApplyAction(state_combat->ChanceOutcomes().front().first);
      } else {
        Action chosen = state_combat->LegalActions().front();
        for (Action a : state_combat->LegalActions()) {
          if (a >= dune_imperium::kActionSelectAgentCard0 && a < dune_imperium::kActionSelectAgentCard0 + 256) {
            chosen = a;
            break;
          }
        }
        state_combat->ApplyAction(chosen);
      }
    }
    assert(!state_combat->IsTerminal());

    // First search in combat intrigue (gets full tiny budget = 16)
    DuneSearchResult res1 = RunSessionSearch(session, *state_combat);
    assert(res1.diagnostics.hard_sim_limit == 16);
    assert(res1.simulations_completed == 16);

    // Advance state to next combat action for the same player (we construct state descendant history)
    auto state_combat2 = state_combat->Clone();
    state_combat2->ApplyAction(res1.diagnostics.selected_action);
    while (state_combat2->IsChanceNode()) {
      state_combat2->ApplyAction(state_combat2->ChanceOutcomes().front().first);
    }



    // Test that intermediate forced/bookkeeping search (a state with 1 legal action) does NOT renew the budget
    // Test that intermediate forced/bookkeeping search (a state with 1 legal action) does NOT renew the budget
    // Second search in combat intrigue (descendant of state_combat) has exactly 1 legal action, so it is forced/bookkeeping
    // and should have 0 simulations completed.
    DuneSearchResult res2 = RunSessionSearch(session, *state_combat2);
    assert(res2.diagnostics.hard_sim_limit == 0);
    assert(res2.simulations_completed == 0);
  }

  // 5. Reset reasons
  {
    DuneSearchConfig config;
    DuneSearchSession session(config, evaluator, DuneSearchBudgetMode::kFixedSessionSimulations);

    RunSessionSearch(session, *state);
    session.ResetSession("manual_reset_test");
    DuneSearchResult res = RunSessionSearch(session, *state);
    assert(res.diagnostics.reset_reason == "manual_reset_test");
  }

  std::cout << "TestInvariantGatesAndScenarios Passed!\n\n";
}

class BifurcationMockEvaluator : public MockEvaluator {
 public:
  using MockEvaluator::MockEvaluator;

  ActionsAndProbs Prior(const State& state) override {
    const std::vector<Action>& history = state.History();
    std::vector<Action> legals = state.LegalActions();
    if (legals.empty()) return {};

    bool has_a = false;
    if (history.size() > deal_history_index) {
      has_a = (history[deal_history_index] == action_a);
    }

    ActionsAndProbs priors;
    if (has_a) {
      priors.push_back({legals.front(), 1.0});
      for (size_t i = 1; i < legals.size(); ++i) {
        priors.push_back({legals[i], 0.0});
      }
    } else {
      priors.push_back({legals.back(), 1.0});
      for (size_t i = 0; i < legals.size() - 1; ++i) {
        priors.push_back({legals[i], 0.0});
      }
    }
    return priors;
  }

  Action action_a = -1;
  Action action_b = -1;
  size_t deal_history_index = 0;
};

class BifurcationMockState : public TestRoutingState {
 public:
  BifurcationMockState(std::shared_ptr<const Game> game)
      : TestRoutingState(game), is_chance_(false), mock_current_player_(0), is_terminal_(false) {}

  bool IsChanceNode() const override { return is_chance_; }

  std::string ObservationString(Player player) const override {
    std::string obs = TestRoutingState::ObservationString(player);
    for (Action a : History()) {
      if (a == 100 || a == 101) {
        absl::StrAppend(&obs, "|draw=", a);
      }
    }
    return obs;
  }

  std::string InformationStateString(Player player) const override {
    std::string inf = TestRoutingState::InformationStateString(player);
    for (Action a : History()) {
      if (a == 100 || a == 101) {
        absl::StrAppend(&inf, "|draw=", a);
      }
    }
    return inf;
  }

  Player CurrentPlayer() const override {
    if (is_chance_) return kChancePlayerId;
    return mock_current_player_;
  }

  ActionsAndProbs ChanceOutcomes() const override {
    if (is_chance_) {
      return {{100, 0.5}, {101, 0.5}};
    }
    return {};
  }

  std::vector<Action> LegalActions() const override {
    if (is_chance_) return {100, 101};
    return mock_legal_actions_;
  }

  bool IsTerminal() const override {
    return is_terminal_ || dune_imperium::DuneImperiumState::IsTerminal();
  }

  std::unique_ptr<State> Clone() const override {
    return std::make_unique<BifurcationMockState>(*this);
  }

  void DoApplyAction(Action action_id) override {
    if (action_id >= dune_imperium::kActionSelectAgentCard0 &&
        action_id < dune_imperium::kActionSelectAgentCard0 + 256) {
      is_chance_ = true;
    } else if (action_id == 100 || action_id == 101) {
      is_chance_ = false;
      mock_current_player_ = 0;
      mock_legal_actions_ = {
        dune_imperium::kActionAgentSpaceConspire,
        dune_imperium::kActionAgentSpaceHighCouncil
      };
    } else {
      is_terminal_ = true;
      mock_legal_actions_ = {};
    }
  }

  void SetChance(bool c) { is_chance_ = c; }
  void SetMockCurrentPlayer(Player p) { mock_current_player_ = p; }
  void SetMockLegalActions(const std::vector<Action>& actions) { mock_legal_actions_ = actions; }

 private:
  bool is_chance_;
  Player mock_current_player_;
  std::vector<Action> mock_legal_actions_;
  bool is_terminal_;
};

void TestFidelityGateScenarios() {
  std::cout << "Running TestFidelityGateScenarios...\n";
  auto game = open_spiel::LoadGame("dune_imperium");
  auto state = game->NewInitialState();
  while (!state->IsTerminal()) {
    if (state->IsChanceNode()) {
      state->ApplyAction(state->ChanceOutcomes().front().first);
    } else {
      const auto* dune_state = dynamic_cast<const dune_imperium::DuneImperiumState*>(state.get());
      SPIEL_CHECK_TRUE(dune_state != nullptr);
      if (dune_state->phase() == dune_imperium::GamePhase::kAgentTurns && state->LegalActions().size() > 1) {
        break;
      }
      state->ApplyAction(state->LegalActions().front());
    }
  }

  ActionsAndProbs priors;
  std::vector<Action> legal_actions = state->LegalActions();
  for (Action a : legal_actions) {
    priors.push_back({a, 1.0 / legal_actions.size()});
  }
  std::vector<double> mock_vals(4, 0.0);
  auto evaluator = std::make_shared<MockEvaluator>(priors, mock_vals);

  DuneSearchConfig config;
  config.max_simulations = 200;
  config.fixed_session_limit = 200;
  config.fixed_continuation_reserve = 50;
  config.purchase_combat_budget = 0; // Short decisions budget = 0

  DuneSearchSession session(config, evaluator, DuneSearchBudgetMode::kFixedSessionSimulations);

  // 1. First Searched Role classification
  DuneDecisionRole first_role = ClassifyDuneDecisionRole(*state, state->CurrentPlayer(), false);
  assert(first_role == DuneDecisionRole::kAgentPrimary);

  // 2. Budget Accounting
  auto root_state = state->Clone();
  DuneSearchResult res = RunSessionSearch(session, *root_state);
  assert(session.HasActiveSession());
  assert(res.simulations_completed <= 150);
  int first_sims = res.simulations_completed;

  // Apply action
  Action act = res.diagnostics.selected_action;
  root_state->ApplyAction(act);

  // 3. Continuation Budget Accounting
  DuneSearchResult res_cont = RunSessionSearch(session, *root_state);
  assert(session.session_new_simulations_completed() == first_sims + res_cont.simulations_completed);
  assert(session.session_new_simulations_completed() <= 200);

  // 4. Policy-only short budget handling (combat/purchase)
  std::unique_ptr<State> purchase_state = game->NewInitialState();
  while (!purchase_state->IsTerminal()) {
    if (purchase_state->IsChanceNode()) {
      purchase_state->ApplyAction(purchase_state->ChanceOutcomes().front().first);
    } else {
      DuneDecisionRole role = ClassifyDuneDecisionRole(*purchase_state, purchase_state->CurrentPlayer(), false);
      if (role == DuneDecisionRole::kPurchase || role == DuneDecisionRole::kCombatIntrigue) {
        break;
      }
      purchase_state->ApplyAction(purchase_state->LegalActions().front());
    }
  }
  if (!purchase_state->IsTerminal()) {
    DuneSearchResult res_purchase = RunSessionSearch(session, *purchase_state);
    assert(res_purchase.simulations_completed == 0);
    assert(res_purchase.used_fallback);
    assert(res_purchase.fallback_reason == "policy_only_purchase_combat");
  }

  // 5. Determinism: same seed produces identical results
  DuneSearchConfig config_det = config;
  config_det.seed = 12345;
  DuneSearchSession session_det1(config_det, evaluator, DuneSearchBudgetMode::kFixedSessionSimulations);
  DuneSearchSession session_det2(config_det, evaluator, DuneSearchBudgetMode::kFixedSessionSimulations);

  auto state_det1 = game->NewInitialState();
  auto state_det2 = game->NewInitialState();
  while (state_det1->IsChanceNode()) {
    state_det1->ApplyAction(state_det1->ChanceOutcomes().front().first);
    state_det2->ApplyAction(state_det2->ChanceOutcomes().front().first);
  }
  DuneSearchResult r1 = RunSessionSearch(session_det1, *state_det1);
  DuneSearchResult r2 = RunSessionSearch(session_det2, *state_det2);
  assert(r1.diagnostics.selected_action == r2.diagnostics.selected_action);
  assert(r1.simulations_completed == r2.simulations_completed);

  // 6. Card draw with two possible observations
  auto card_draw_state = game->NewInitialState();
  while (true) {
    const auto* ds = dynamic_cast<const dune_imperium::DuneImperiumState*>(card_draw_state.get());
    if (ds && ds->phase() == dune_imperium::GamePhase::kDeal && card_draw_state->IsChanceNode()) {
      auto outcomes = card_draw_state->ChanceOutcomes();
      if (outcomes.size() >= 2) {
        break;
      }
    }
    if (card_draw_state->IsChanceNode()) {
      card_draw_state->ApplyAction(card_draw_state->ChanceOutcomes().front().first);
    } else {
      card_draw_state->ApplyAction(card_draw_state->LegalActions().front());
    }
  }
  size_t deal_history_index = card_draw_state->History().size();
  auto outcomes = card_draw_state->ChanceOutcomes();
  assert(outcomes.size() >= 2);
  auto obs_state1 = card_draw_state->Clone();
  obs_state1->ApplyAction(outcomes[0].first);
  auto obs_state2 = card_draw_state->Clone();
  obs_state2->ApplyAction(outcomes[1].first);

  auto advance_to_agent_turn = [](std::unique_ptr<State>& s) {
    while (!s->IsTerminal()) {
      const auto* ds = dynamic_cast<const dune_imperium::DuneImperiumState*>(s.get());
      if (ds && ds->phase() == dune_imperium::GamePhase::kAgentTurns && s->CurrentPlayer() == 0 && s->LegalActions().size() > 1) {
        break;
      }
      if (s->IsChanceNode()) {
        s->ApplyAction(s->ChanceOutcomes().front().first);
      } else {
        s->ApplyAction(s->LegalActions().front());
      }
    }
  };

  advance_to_agent_turn(obs_state1);
  advance_to_agent_turn(obs_state2);
  assert(obs_state1->InformationStateTensor(0) != obs_state2->InformationStateTensor(0));

  auto bif_eval = std::make_shared<BifurcationMockEvaluator>(priors, mock_vals);
  bif_eval->action_a = outcomes[0].first;
  bif_eval->action_b = outcomes[1].first;
  bif_eval->deal_history_index = deal_history_index;

  DuneSearchConfig config_bif = config;
  config_bif.max_simulations = 10;
  config_bif.fixed_session_limit = 10;
  DuneSearchSession session_bif1(config_bif, bif_eval, DuneSearchBudgetMode::kFixedSessionSimulations);
  DuneSearchSession session_bif2(config_bif, bif_eval, DuneSearchBudgetMode::kFixedSessionSimulations);

  DuneSearchResult r_bif1 = RunSessionSearch(session_bif1, *obs_state1);
  DuneSearchResult r_bif2 = RunSessionSearch(session_bif2, *obs_state2);
  assert(r_bif1.diagnostics.selected_action != r_bif2.diagnostics.selected_action);

  // 6b. Card draw persistence & bifurcation: verify session remains active across draw branches, but mismatches trigger miss
  {
    auto bif_eval_draw = std::make_shared<BifurcationMockEvaluator>(priors, mock_vals);
    bif_eval_draw->action_a = 100;
    bif_eval_draw->action_b = 101;

    DuneSearchSession session_draw(config, bif_eval_draw, DuneSearchBudgetMode::kFixedSessionSimulations);

    // Root state: card selection (kAgentPrimary)
    BifurcationMockState state_root(game);
    state_root.SetMockLegalActions({
      dune_imperium::kActionSelectAgentCard0,
      dune_imperium::kActionSelectAgentCard0 + 1
    });

    bif_eval_draw->deal_history_index = state_root.History().size() + 1;

    DuneSearchResult r_root = RunSessionSearch(session_draw, state_root);
    assert(session_draw.HasActiveSession());

    // Apply primary action
    Action selected = r_root.diagnostics.selected_action;
    if (selected == kInvalidAction) selected = dune_imperium::kActionSelectAgentCard0;

    auto state_chance = state_root;
    state_chance.ApplyAction(selected);

    // Branch A: apply chance outcome 100
    auto state_branch_a = state_chance;
    state_branch_a.ApplyAction(100);

    DuneSearchResult r_branch_a = RunSessionSearch(session_draw, state_branch_a);
    assert(session_draw.HasActiveSession());
    assert(session_draw.last_re_root_status() == "hit");
    assert(r_branch_a.diagnostics.selected_action == dune_imperium::kActionAgentSpaceConspire);

    // Branch B: apply chance outcome 101 (sibling branch)
    auto state_branch_b = state_chance;
    state_branch_b.ApplyAction(101);

    DuneSearchResult r_branch_b = RunSessionSearch(session_draw, state_branch_b);
    assert(session_draw.HasActiveSession());
    assert(session_draw.last_re_root_status() == "miss");
    assert(r_branch_b.diagnostics.selected_action == dune_imperium::kActionAgentSpaceHighCouncil);

    // Verify follow-up choices differ between the two observed branches
    assert(r_branch_a.diagnostics.selected_action != r_branch_b.diagnostics.selected_action);

    // Verify budget is accumulated correctly
    assert(session_draw.session_new_simulations_completed() ==
           r_root.simulations_completed + r_branch_a.simulations_completed + r_branch_b.simulations_completed);
  }

  // 7. Re-root miss check
  DuneSearchSession session_miss(config, evaluator, DuneSearchBudgetMode::kFixedSessionSimulations);
  DuneSearchResult r_miss_root = RunSessionSearch(session_miss, *state); // First search (kAgentPrimary)
  auto sibling_state = state->Clone();
  Action other_act = kInvalidAction;
  for (Action a : legal_actions) {
    if (a != act) {
      other_act = a;
      break;
    }
  }
  if (other_act != kInvalidAction) {
    sibling_state->ApplyAction(other_act);
    DuneSearchResult r_sibling = RunSessionSearch(session_miss, *sibling_state);
    assert(session_miss.last_re_root_status() == "miss");
    // Verify fail-closed session budget tracking: budget was accumulated, not reset
    assert(session_miss.session_new_simulations_completed() == r_miss_root.simulations_completed + r_sibling.simulations_completed);
  }

  // 8. Fail-closed check
  DuneSearchSession empty_session(config, evaluator, DuneSearchBudgetMode::kFixedSessionSimulations);
  assert(!empty_session.HasActiveSession());

  // 9. Stable JSON ordering
  open_spiel::json::Object json_obj;
  json_obj["b"] = 2;
  json_obj["a"] = 1;
  std::string json_str = open_spiel::json::ToString(json_obj, false);
  assert(json_str == "{\"a\": 1, \"b\": 2}");

  std::cout << "TestFidelityGateScenarios Passed!\n\n";
}

void TestMoreRoutingAndScenarioInvariants() {
  std::shared_ptr<const Game> game = LoadGame("dune_imperium");
  std::shared_ptr<algorithms::Evaluator> evaluator =
      std::make_shared<MockEvaluator>(ActionsAndProbs{}, std::vector<double>{0.0, 0.0, 0.0, 0.0});

  DuneSearchConfig config;
  config.max_simulations = 100;
  config.check_strategic_state = false;

  // 1. Duplicate calls at one primary root & Descendant Continuation
  {
    DuneSearchSession session(config, evaluator, DuneSearchBudgetMode::kFixedSessionSimulations);

    TestRoutingState state(game);
    state.SetMockLegalActions({
      dune_imperium::kActionSelectAgentCard0,
      dune_imperium::kActionSelectAgentCard0 + 1
    });

    DuneSearchResult res1 = RunSessionSearch(session, state);
    std::cout << "res1.diagnostics.reset_reason: " << res1.diagnostics.reset_reason << "\n";
    std::string first_session_id = res1.diagnostics.session_id;
    assert(res1.diagnostics.reset_reason == "new_placement_activation");

    DuneSearchResult res2 = RunSessionSearch(session, state);
    assert(res2.diagnostics.session_id == first_session_id);
    assert(res2.diagnostics.reset_reason == "none");

    // Simulate descendant continuation
    auto state_next = state;
    if (res2.diagnostics.selected_action != kInvalidAction) {
      state_next.ApplyAction(res2.diagnostics.selected_action);
    } else {
      state_next.ApplyAction(dune_imperium::kActionSelectAgentCard0);
    }
    state_next.SetMockLegalActions({
      dune_imperium::kActionAgentSpaceConspire,
      dune_imperium::kActionAgentSpaceArrakeen
    });

    std::cout << "res1.selected_action: " << res1.diagnostics.selected_action << "\n";
    std::cout << "first_session_id: " << first_session_id << "\n";
    DuneSearchResult res3 = RunSessionSearch(session, state_next);
    std::cout << "res3.diagnostics.reset_reason: " << res3.diagnostics.reset_reason << "\n";
    std::cout << "res3.diagnostics.session_id: " << res3.diagnostics.session_id << "\n";
    assert(res3.diagnostics.session_id == first_session_id);
    assert(res3.diagnostics.reset_reason == "none");
  }

  // 2. Multiple placement activations & Early Reveal
  {
    DuneSearchSession session(config, evaluator, DuneSearchBudgetMode::kFixedSessionSimulations);

    // Scenario A: 2 Normal Activations
    {
      TestRoutingState state1(game);
      state1.SetPlayerAgentsRemainingForTesting(0, 2);
      state1.SetMockLegalActions({dune_imperium::kActionSelectAgentCard0, dune_imperium::kActionSelectAgentCard0 + 1});
      DuneSearchResult res1 = RunSessionSearch(session, state1);
      assert(res1.diagnostics.reset_reason == "new_placement_activation");

      // Apply card select and space placement to advance history
      state1.ApplyAction(dune_imperium::kActionSelectAgentCard0);
      state1.SetMockLegalActions({dune_imperium::kActionAgentSpaceConspire});
      state1.ApplyAction(dune_imperium::kActionAgentSpaceConspire);
      while (state1.IsChanceNode()) {
        state1.ApplyAction(state1.LegalActions()[0]);
      }

      // Activation 2: 1 agent remaining
      TestRoutingState state2 = state1;
      state2.SetPlayerAgentsRemainingForTesting(0, 1);
      state2.SetMockCurrentPlayer(0);
      state2.SetMockLegalActions({dune_imperium::kActionSelectAgentCard0 + 1, dune_imperium::kActionSelectAgentCard0 + 2});
      DuneSearchResult res2 = RunSessionSearch(session, state2);
      assert(res2.diagnostics.reset_reason == "new_placement_activation");
    }

    // Scenario B: 3 Activations (Swordmaster)
    {
      TestRoutingState state1(game);
      state1.SetPlayerAgentsRemainingForTesting(0, 3);
      state1.SetMockLegalActions({dune_imperium::kActionSelectAgentCard0, dune_imperium::kActionSelectAgentCard0 + 1});
      DuneSearchResult res1 = RunSessionSearch(session, state1);
      assert(res1.diagnostics.reset_reason == "new_placement_activation");

      // Play 1st agent (plays card 0)
      state1.ApplyAction(dune_imperium::kActionSelectAgentCard0);
      state1.SetMockLegalActions({dune_imperium::kActionAgentSpaceConspire});
      state1.ApplyAction(dune_imperium::kActionAgentSpaceConspire);
      while (state1.IsChanceNode()) {
        state1.ApplyAction(state1.LegalActions()[0]);
      }

      // 2nd activation: 2 agents remaining
      TestRoutingState state2 = state1;
      state2.SetPlayerAgentsRemainingForTesting(0, 2);
      state2.SetMockCurrentPlayer(0);
      state2.SetMockLegalActions({dune_imperium::kActionSelectAgentCard0 + 1, dune_imperium::kActionSelectAgentCard0 + 2});
      DuneSearchResult res2 = RunSessionSearch(session, state2);
      assert(res2.diagnostics.reset_reason == "new_placement_activation");

      // Play 2nd agent (plays card 1)
      state2.ApplyAction(dune_imperium::kActionSelectAgentCard0 + 1);
      state2.SetMockLegalActions({dune_imperium::kActionAgentSpaceConspire});
      state2.ApplyAction(dune_imperium::kActionAgentSpaceConspire);
      while (state2.IsChanceNode()) {
        state2.ApplyAction(state2.LegalActions()[0]);
      }

      // 3rd activation: 1 agent remaining
      TestRoutingState state3 = state2;
      state3.SetPlayerAgentsRemainingForTesting(0, 1);
      state3.SetMockCurrentPlayer(0);
      state3.SetMockLegalActions({dune_imperium::kActionSelectAgentCard0 + 2, dune_imperium::kActionSelectAgentCard0 + 3});
      DuneSearchResult res3 = RunSessionSearch(session, state3);
      assert(res3.diagnostics.reset_reason == "new_placement_activation");
    }

    // Scenario C: 4 Activations (Swordmaster + Mentat)
    {
      TestRoutingState state(game);
      state.SetPlayerAgentsRemainingForTesting(0, 4);

      for (int a = 4; a >= 1; --a) {
        state.SetPlayerAgentsRemainingForTesting(0, a);
        state.SetMockCurrentPlayer(0);
        int card_offset = 4 - a; // 0, 1, 2, 3
        state.SetMockLegalActions({dune_imperium::kActionSelectAgentCard0 + card_offset, dune_imperium::kActionSelectAgentCard0 + card_offset + 1});
        DuneSearchResult res = RunSessionSearch(session, state);
        assert(res.diagnostics.reset_reason == "new_placement_activation");

        if (a > 1) {
          state.ApplyAction(dune_imperium::kActionSelectAgentCard0 + card_offset);
          state.SetMockLegalActions({dune_imperium::kActionAgentSpaceConspire});
          state.ApplyAction(dune_imperium::kActionAgentSpaceConspire);
          while (state.IsChanceNode()) {
            state.ApplyAction(state.LegalActions()[0]);
          }
        }
      }
    }

    // Scenario D: Early Reveal Suppression
    {
      // 1st placement activation
      TestRoutingState state1(game);
      state1.SetPlayerAgentsRemainingForTesting(0, 2);
      state1.SetMockLegalActions({dune_imperium::kActionSelectAgentCard0, dune_imperium::kActionSelectAgentCard0 + 1});
      DuneSearchResult res1 = RunSessionSearch(session, state1);
      assert(res1.diagnostics.reset_reason == "new_placement_activation");

      // Play 1st agent card and space
      state1.ApplyAction(dune_imperium::kActionSelectAgentCard0);
      state1.SetMockLegalActions({dune_imperium::kActionAgentSpaceConspire});
      state1.ApplyAction(dune_imperium::kActionAgentSpaceConspire);
      while (state1.IsChanceNode()) {
        state1.ApplyAction(state1.LegalActions()[0]);
      }

      // Player does early reveal: plays Reveal action (agents remaining -> 0)
      TestRoutingState state2 = state1;
      state2.SetMockCurrentPlayer(0);
      state2.SetMockLegalActions({dune_imperium::kActionReveal});
      state2.ApplyAction(dune_imperium::kActionReveal);
      while (state2.IsChanceNode()) {
        state2.ApplyAction(state2.LegalActions()[0]);
      }

      // Transition player state to have 0 agents remaining and show purchase/end actions
      state2.SetPlayerAgentsRemainingForTesting(0, 0);
      state2.SetMockLegalActions({dune_imperium::kActionBuyImperiumRow0, dune_imperium::kActionEndTurn});

      DuneSearchResult res2 = RunSessionSearch(session, state2);
      // Since agents remaining == 0, the decision role is classified as kPurchase or kOtherOptional (not kAgentPrimary).
      // So reset_reason must NOT be new_placement_activation!
      assert(res2.diagnostics.reset_reason != "new_placement_activation");
    }
  }

  // Case 1: Descendant + existing node (hit)
  {
    DuneSearchConfig config_case1 = config;
    config_case1.fixed_session_limit = 200;
    config_case1.fixed_continuation_reserve = 50;
    DuneSearchSession session(config_case1, evaluator, DuneSearchBudgetMode::kFixedSessionSimulations);
    
    // Start session
    TestRoutingState state1(game);
    state1.SetMockLegalActions({dune_imperium::kActionSelectAgentCard0, dune_imperium::kActionSelectAgentCard0 + 1});
    DuneSearchResult res1 = RunSessionSearch(session, state1);
    assert(session.HasActiveSession());

    // Create descendant state
    auto state2 = state1;
    Action act = res1.diagnostics.selected_action;
    state2.ApplyAction(act);
    state2.SetMockLegalActions({dune_imperium::kActionAgentSpaceConspire, dune_imperium::kActionAgentSpaceHighCouncil});

    // Manually inject a node for state2 to guarantee it is in the cache with visits
    TestBotAccessor::ClearNodes(*session.GetBot());
    TestBotAccessor::InjectMockNode(*session.GetBot(), state2, 15);
    assert(session.GetBot()->nodes().size() == 1);

    // Run search on state2
    DuneSearchResult res2 = RunSessionSearch(session, state2);
    assert(session.intermediate_re_root_status() == "hit");
    assert(session.last_re_root_status() == "hit");
    assert(res2.diagnostics.re_root_status == "hit");
    assert(res2.diagnostics.inherited_root_visits == 15);
    assert(session.GetBot()->nodes().size() > 1);
  }

  // Case 1.5: Non-descendant + Cache Presence (Impossibility Test)
  {
    DuneSearchSession session(config, evaluator, DuneSearchBudgetMode::kFixedSessionSimulations);
    
    // First, start a session with a primary search
    TestRoutingState state1(game);
    state1.SetMockLegalActions({dune_imperium::kActionSelectAgentCard0, dune_imperium::kActionSelectAgentCard0 + 1});
    DuneSearchResult res1 = RunSessionSearch(session, state1);
    assert(session.HasActiveSession());

    // Construct a non-descendant state2 (same player & round, but different history)
    TestRoutingState state2(game);
    state2.SetMockCurrentPlayer(0);
    // Give it a non-primary, continuation role (e.g. kAgentContinuation)
    Action act = res1.diagnostics.selected_action;
    Action other_act = (act == dune_imperium::kActionSelectAgentCard0) 
                       ? (dune_imperium::kActionSelectAgentCard0 + 1)
                       : dune_imperium::kActionSelectAgentCard0;
    state2.SetMockLegalActions({dune_imperium::kActionSelectAgentCard0, dune_imperium::kActionSelectAgentCard0 + 1});
    state2.ApplyAction(other_act);
    state2.SetMockLegalActions({dune_imperium::kActionAgentSpaceConspire, dune_imperium::kActionAgentSpaceHighCouncil});
    
    // Manually inject a cache entry for state2 and another state
    TestBotAccessor::ClearNodes(*session.GetBot());
    auto other_state = state1;
    TestBotAccessor::InjectMockNode(*session.GetBot(), other_state, 5);
    auto other_key = session.GetBot()->GetStateKey(other_state);

    TestBotAccessor::InjectMockNode(*session.GetBot(), state2, 10);
    assert(session.GetBot()->nodes().size() == 2);

    // Now run search on state2.
    // Pass 1 mismatch check will call HandleReRootMismatch, resetting the bot and nodes.
    DuneSearchResult res2 = RunSessionSearch(session, state2);
    assert(session.intermediate_re_root_status() == "miss");
    assert(session.last_re_root_status() == "miss");
    assert(session.GetBot()->nodes().find(other_key) == session.GetBot()->nodes().end());
  }

  // Case 2: Descendant + absent node (soft miss - Continuation/Optional Role)
  {
    DuneSearchSession session(config, evaluator, DuneSearchBudgetMode::kFixedSessionSimulations);
    
    // Start session
    TestRoutingState state1(game);
    state1.SetMockLegalActions({dune_imperium::kActionSelectAgentCard0, dune_imperium::kActionSelectAgentCard0 + 1});
    DuneSearchResult res1 = RunSessionSearch(session, state1);
    assert(session.HasActiveSession());

    // Create descendant state
    auto state2 = state1;
    Action act = res1.diagnostics.selected_action;
    state2.ApplyAction(act);
    // Make role kAgentContinuation
    state2.SetMockLegalActions({dune_imperium::kActionAgentSpaceConspire, dune_imperium::kActionAgentSpaceHighCouncil});

    // We manually clear the bot's nodes so the node is guaranteed absent.
    TestBotAccessor::ClearNodes(*session.GetBot());

    // Inject a dummy node for ANOTHER state so that the nodes map is NOT empty initially.
    auto other_state = state1;
    Action other_act = (act == dune_imperium::kActionSelectAgentCard0) 
                       ? (dune_imperium::kActionSelectAgentCard0 + 1)
                       : dune_imperium::kActionSelectAgentCard0;
    other_state.ApplyAction(other_act);
    TestBotAccessor::InjectMockNode(*session.GetBot(), other_state, 5);
    assert(session.GetBot()->nodes().size() == 1);

    // Run search on state2
    DuneSearchResult res2 = RunSessionSearch(session, state2);
    assert(session.intermediate_re_root_status() == "hit");
    assert(session.last_re_root_status() == "miss");
    assert(res2.diagnostics.re_root_status == "miss");
    assert(res2.diagnostics.post_chance_branch_miss == true);
    // Verify nodes was NOT reset (meaning our injected other_state node is still there, or new nodes were added)
    assert(!session.GetBot()->nodes().empty());
  }

  // Case 2.5: Descendant + absent node (soft miss - Primary Role)
  {
    DuneSearchSession session(config, evaluator, DuneSearchBudgetMode::kFixedSessionSimulations);

    // Start session
    TestRoutingState state1(game);
    state1.SetMockLegalActions({dune_imperium::kActionSelectAgentCard0, dune_imperium::kActionSelectAgentCard0 + 1});
    DuneSearchResult res1 = RunSessionSearch(session, state1);
    assert(session.HasActiveSession());

    // Create descendant state with primary role (agents remaining = 1)
    auto state2 = state1;
    Action act = res1.diagnostics.selected_action;
    state2.ApplyAction(act);
    state2.SetMockLegalActions({dune_imperium::kActionAgentSpaceConspire});
    state2.ApplyAction(dune_imperium::kActionAgentSpaceConspire);
    while (state2.IsChanceNode()) {
      state2.ApplyAction(state2.LegalActions()[0]);
    }
    state2.SetMockCurrentPlayer(0);
    state2.SetPlayerAgentsRemainingForTesting(0, 1);
    state2.SetMockLegalActions({dune_imperium::kActionSelectAgentCard0 + 2, dune_imperium::kActionSelectAgentCard0 + 3});
    // Make sure nodes are absent
    TestBotAccessor::ClearNodes(*session.GetBot());
    // Inject a dummy node for verification
    auto other_state = state1;
    TestBotAccessor::InjectMockNode(*session.GetBot(), other_state, 5);
    auto other_key = session.GetBot()->GetStateKey(other_state);

    DuneSearchResult res2 = RunSessionSearch(session, state2);
    assert(session.intermediate_re_root_status() == "hit");
    assert(session.last_re_root_status() == "none");
    assert(res2.diagnostics.re_root_status == "none");
    assert(session.GetBot()->nodes().find(other_key) == session.GetBot()->nodes().end());
  }

  // Case 2.7: Role transition (Placement to Short-Window) with the fixed
  // session budget already consumed by the primary (reserve == 0). The short
  // window is therefore correctly routed to the policy-only fallback before the
  // search-path re-root matching check runs, so last_re_root_status keeps the
  // "hit" set by the descendant-validation step rather than being overwritten
  // to the short-bot cache "miss". The exhausted-budget fallback itself is
  // covered in depth by TestShortWindowExhaustedBudgetFallsBackToPrior.
  {
    DuneSearchSession session(config, evaluator, DuneSearchBudgetMode::kFixedSessionSimulations);

    // Search on placement state
    TestRoutingState state1(game);
    state1.SetMockLegalActions({dune_imperium::kActionSelectAgentCard0, dune_imperium::kActionSelectAgentCard0 + 1});
    DuneSearchResult res1 = RunSessionSearch(session, state1);

    // Transition to descendant other optional role (short-window bot)
    auto state2 = state1;
    Action act = res1.diagnostics.selected_action;
    state2.ApplyAction(act);
    state2.SetMockLegalActions({dune_imperium::kActionAgentSpaceConspire});
    state2.ApplyAction(dune_imperium::kActionAgentSpaceConspire);
    while (state2.IsChanceNode()) {
      state2.ApplyAction(state2.LegalActions()[0]);
    }
    state2.SetMockCurrentPlayer(0);
    state2.SetPlayerAgentsRemainingForTesting(0, 0);
    state2.SetMockLegalActions({dune_imperium::kActionPlayIntriguePlotCard0 + 2, dune_imperium::kActionReveal});

    DuneSearchResult res2 = RunSessionSearch(session, state2);
    assert(session.intermediate_re_root_status() == "hit");
    // The primary consumed the whole fixed session budget (reserve == 0), so
    // the short window is exhausted and returns the policy-only fallback before
    // the search-path re-root matching check. last_re_root_status therefore
    // keeps the descendant-validation "hit" instead of the short-bot "miss".
    assert(session.last_re_root_status() == "hit");
    assert(res2.fallback_reason == "short_window_budget_exceeded");
  }

  // Case 3: Non-descendant history mismatch (Continuation/Optional Role)
  {
    DuneSearchSession session(config, evaluator, DuneSearchBudgetMode::kFixedSessionSimulations);

    // Search on placement state
    TestRoutingState state1(game);
    state1.SetMockLegalActions({dune_imperium::kActionSelectAgentCard0, dune_imperium::kActionSelectAgentCard0 + 1});
    DuneSearchResult res1 = RunSessionSearch(session, state1);

    // Create a non-descendant state (same player & round, but different history)
    TestRoutingState state2(game);
    state2.SetMockCurrentPlayer(0);
    Action act = res1.diagnostics.selected_action;
    Action other_act = (act == dune_imperium::kActionSelectAgentCard0) 
                       ? (dune_imperium::kActionSelectAgentCard0 + 1)
                       : dune_imperium::kActionSelectAgentCard0;
    state2.SetMockLegalActions({dune_imperium::kActionSelectAgentCard0, dune_imperium::kActionSelectAgentCard0 + 1});
    state2.ApplyAction(other_act);
    state2.SetMockLegalActions({dune_imperium::kActionAgentSpaceConspire, dune_imperium::kActionAgentSpaceHighCouncil});

    // Populate placement_bot_'s cache
    TestBotAccessor::ClearNodes(*session.GetBot());
    auto other_state = state1;
    TestBotAccessor::InjectMockNode(*session.GetBot(), other_state, 5);
    auto other_key = session.GetBot()->GetStateKey(other_state);

    DuneSearchResult res2 = RunSessionSearch(session, state2);
    assert(session.intermediate_re_root_status() == "miss");
    assert(session.last_re_root_status() == "miss");
    assert(res2.diagnostics.re_root_status == "miss");
    assert(res2.diagnostics.post_chance_branch_miss == true);
    assert(session.GetBot()->nodes().find(other_key) == session.GetBot()->nodes().end());
  }

  // Case 3.5: Non-descendant history mismatch (Primary Role)
  {
    DuneSearchSession session(config, evaluator, DuneSearchBudgetMode::kFixedSessionSimulations);

    // Search on placement state
    TestRoutingState state1(game);
    state1.SetMockLegalActions({dune_imperium::kActionSelectAgentCard0, dune_imperium::kActionSelectAgentCard0 + 1});
    DuneSearchResult res1 = RunSessionSearch(session, state1);

    // Create a non-descendant primary role state (same player & round, different history)
    TestRoutingState state2(game);
    state2.SetMockCurrentPlayer(0);
    Action act = res1.diagnostics.selected_action;
    Action other_act = (act == dune_imperium::kActionSelectAgentCard0) 
                       ? (dune_imperium::kActionSelectAgentCard0 + 1)
                       : dune_imperium::kActionSelectAgentCard0;
    state2.SetMockLegalActions({dune_imperium::kActionSelectAgentCard0, dune_imperium::kActionSelectAgentCard0 + 1});
    state2.ApplyAction(other_act);
    state2.SetPlayerAgentsRemainingForTesting(0, 1);
    state2.SetMockLegalActions({dune_imperium::kActionSelectAgentCard0 + 2, dune_imperium::kActionSelectAgentCard0 + 3});

    // Populate cache
    TestBotAccessor::ClearNodes(*session.GetBot());
    auto other_state = state1;
    TestBotAccessor::InjectMockNode(*session.GetBot(), other_state, 5);
    auto other_key = session.GetBot()->GetStateKey(other_state);

    DuneSearchResult res2 = RunSessionSearch(session, state2);
    assert(session.intermediate_re_root_status() == "none");
    assert(session.last_re_root_status() == "none");
    assert(res2.diagnostics.re_root_status == "none");
    assert(session.GetBot()->nodes().find(other_key) == session.GetBot()->nodes().end());
  }

  std::cout << "TestMoreRoutingAndScenarioInvariants Passed!\n\n";
}

void TestPersistentSessionScenarios() {
  std::cout << "Running TestPersistentSessionScenarios...\n";
  std::shared_ptr<const Game> game = LoadGame("dune_imperium");
  std::shared_ptr<algorithms::Evaluator> evaluator =
      std::make_shared<algorithms::RandomRolloutEvaluator>(20, 42);

  DuneSearchConfig config;
  config.max_simulations = 10;
  config.fixed_session_limit = 10;

  // 1. Test incorrect roles: calling search on kForcedOrBookkeeping role should not activate session
  DuneSearchSession session(config, evaluator, DuneSearchBudgetMode::kFixedSessionSimulations);

  auto state = game->NewInitialState();
  DuneDecisionRole role = ClassifyDuneDecisionRole(*state, state->CurrentPlayer(), false);
  assert(role == DuneDecisionRole::kLeaderSelection || role == DuneDecisionRole::kForcedOrBookkeeping);

  DuneSearchResult r = RunSessionSearch(session, *state);
  assert(!session.HasActiveSession()); // Session must NOT be activated!
  assert(session.session_new_simulations_completed() == 0); // Bypassed search, simulations are 0

  std::cout << "TestPersistentSessionScenarios Passed!\n\n";
}

void TestFallbackDeterministic() {
  std::cout << "Running TestFallbackDeterministic...\n";
  std::shared_ptr<const Game> game = LoadGame("dune_imperium");
  std::shared_ptr<algorithms::Evaluator> evaluator =
      std::make_shared<algorithms::RandomRolloutEvaluator>(20, 42);

  auto state = game->NewInitialState();
  // Ensure we are at a player decision node with multiple legal actions (not chance/forced)
  while (state->IsChanceNode() || state->LegalActions().size() <= 1) {
    state->ApplyAction(state->LegalActions()[0]);
  }

  // Case A (Zero-Search Fallback): max_simulations = 0 produces identical actions.
  {
    DuneSearchConfig config;
    config.max_simulations = 0;
    DuneSearchSession session1(config, evaluator, DuneSearchBudgetMode::kFixedSessionSimulations);
    DuneSearchSession session2(config, evaluator, DuneSearchBudgetMode::kFixedSessionSimulations);

    double r_val = 0.55; // selects a specific action from prior stochastically
    DuneSearchResult res1 = session1.Search(*state);
    ControllerDecision dec1 = session1.SelectControllerAction(*state, res1, r_val);
    res1 = session1.CommitAction(dec1);

    DuneSearchResult res2 = session2.Search(*state);
    ControllerDecision dec2 = session2.SelectControllerAction(*state, res2, r_val);
    res2 = session2.CommitAction(dec2);

    assert(res1.used_fallback);
    assert(res2.used_fallback);
    assert(res1.diagnostics.selected_action == res2.diagnostics.selected_action);
    assert(res1.diagnostics.selected_action != kInvalidAction);
  }

  // Case B (Low-Coverage Fallback): max_simulations = 5 triggers fallback on a state with many legal actions
  {
    DuneSearchConfig config;
    config.max_simulations = 5;
    config.min_visit_threshold = 10; // high threshold to guarantee fallback
    DuneSearchSession session1(config, evaluator, DuneSearchBudgetMode::kFixedSessionSimulations);
    DuneSearchSession session2(config, evaluator, DuneSearchBudgetMode::kFixedSessionSimulations);

    double r_val = 0.25;
    DuneSearchResult res1 = session1.Search(*state);
    ControllerDecision dec1 = session1.SelectControllerAction(*state, res1, r_val);
    res1 = session1.CommitAction(dec1);

    DuneSearchResult res2 = session2.Search(*state);
    ControllerDecision dec2 = session2.SelectControllerAction(*state, res2, r_val);
    res2 = session2.CommitAction(dec2);

    assert(res1.used_fallback);
    assert(res2.used_fallback);
    assert(res1.diagnostics.selected_action == res2.diagnostics.selected_action);
    assert(res1.diagnostics.selected_action != kInvalidAction);
  }

  // Case C (Timeout Fallback): very low time limit triggers fallback
  {
    DuneSearchConfig config;
    config.relative_time_budget_ms = 0.001; // extremely low to trigger timeout
    DuneSearchSession session1(config, evaluator, DuneSearchBudgetMode::kFixedSessionSimulations);
    DuneSearchSession session2(config, evaluator, DuneSearchBudgetMode::kFixedSessionSimulations);

    double r_val = 0.85;
    DuneSearchResult res1 = session1.Search(*state);
    ControllerDecision dec1 = session1.SelectControllerAction(*state, res1, r_val);
    res1 = session1.CommitAction(dec1);

    DuneSearchResult res2 = session2.Search(*state);
    ControllerDecision dec2 = session2.SelectControllerAction(*state, res2, r_val);
    res2 = session2.CommitAction(dec2);

    assert(res1.used_fallback);
    assert(res2.used_fallback);
    assert(res1.diagnostics.selected_action == res2.diagnostics.selected_action);
    assert(res1.diagnostics.selected_action != kInvalidAction);
  }

  // Case D (Raw vs. Search Parity on Fallback): asserts fallback matches raw action.
  {
    DuneSearchConfig config;
    config.max_simulations = 0; // force zero sims (direct fallback)
    config.conservative_override_enabled = true;
    DuneSearchSession session(config, evaluator, DuneSearchBudgetMode::kFixedSessionSimulations);

    ActionsAndProbs prior = evaluator->Prior(*state);
    Action argmax_raw_action = kInvalidAction;
    double max_prob = -1.0;
    for (const auto& ap : prior) {
      if (ap.second > max_prob) {
        max_prob = ap.second;
        argmax_raw_action = ap.first;
      }
    }

    bool found_non_argmax = false;
    // Test across multiple r_vals to check stochastic mapping
    for (double r_val : {0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0}) {
      Action raw_act = SampleActionFromPrior(prior, r_val);
      if (raw_act != argmax_raw_action) {
        found_non_argmax = true;
      }

      DuneSearchResult res = session.Search(*state);
      ControllerDecision dec = session.SelectControllerAction(*state, res, r_val);
      res = session.CommitAction(dec);

      assert(res.used_fallback || dec.confidence_fallback);
      assert(dec.raw_reference_action == raw_act);
      assert(dec.selected_action == raw_act);
      assert(res.diagnostics.raw_reference_action == raw_act);
      assert(res.diagnostics.selected_action == raw_act);
      assert(res.diagnostics.confidence_fallback);
    }
    assert(found_non_argmax);
  }

  std::cout << "TestFallbackDeterministic Passed!\n\n";
}

void TestConservativeOverrideCriteria() {
  std::cout << "Running TestConservativeOverrideCriteria...\n";
  std::shared_ptr<const Game> game = LoadGame("dune_imperium");
  TestRoutingState state(game);

  state.SetPhaseForTesting(dune_imperium::GamePhase::kAgentTurns);
  state.SetPlayerAgentsRemainingForTesting(0, 1);
  state.SetMockLegalActions({
    dune_imperium::kActionSelectAgentCard0,
    dune_imperium::kActionSelectAgentCard0 + 1,
    dune_imperium::kActionSelectAgentCard0 + 2
  });

  std::shared_ptr<algorithms::Evaluator> evaluator =
      std::make_shared<algorithms::RandomRolloutEvaluator>(20, 42);

  DuneSearchConfig config;
  config.conservative_override_enabled = true;
  config.conservative_covered_prior_threshold = 0.95;
  config.conservative_meaningful_visit_threshold = 10;
  config.conservative_q_margin_threshold = 0.03;
  config.conservative_stability_checkpoint_fraction = 0.5;
  config.conservative_continuation_overrides_disabled = true;

  DuneSearchSession session(config, evaluator, DuneSearchBudgetMode::kFixedSessionSimulations);
  session.SetLastRequestedMaxSimsForTesting(200);

  DuneSearchResult res;
  res.simulations_completed = 200;
  res.timeout_status = false;
  res.fallback_reason = "none";

  res.diagnostics.actions = {
    dune_imperium::kActionSelectAgentCard0,
    dune_imperium::kActionSelectAgentCard0 + 1,
    dune_imperium::kActionSelectAgentCard0 + 2
  };
  res.diagnostics.num_covered_actions = 3;
  res.diagnostics.covered_prior_mass = 0.98;
  res.diagnostics.visit_counts = {100, 15, 5};
  res.diagnostics.q_values = {0.8, 0.5, 0.1};
  res.diagnostics.priors = {0.9, 0.07, 0.03};
  res.diagnostics.stability_checkpoint_reached = true;
  res.diagnostics.stability_agreement = true;

  double r_val = 0.5;
  {
    ControllerDecision dec = session.SelectControllerAction(state, res, r_val);
    assert(dec.selected_action == dec.mcts_proposed_action);
    assert(!dec.confidence_fallback);
  }

  // 1. Test prior mass check failure
  {
    DuneSearchResult res_fail = res;
    res_fail.diagnostics.covered_prior_mass = 0.90;
    ControllerDecision dec = session.SelectControllerAction(state, res_fail, r_val);
    assert(!dec.pass_prior_mass);
    assert(dec.selected_action == dec.raw_reference_action);
    assert(dec.confidence_fallback);
  }

  // 2. Test min actions check failure
  {
    DuneSearchResult res_fail = res;
    res_fail.diagnostics.num_covered_actions = 2;
    ControllerDecision dec = session.SelectControllerAction(state, res_fail, r_val);
    assert(!dec.pass_min_actions);
    assert(dec.selected_action == dec.raw_reference_action);
  }

  // 3. Test meaningful visits check failure
  {
    DuneSearchResult res_fail = res;
    ControllerDecision dec = session.SelectControllerAction(state, res_fail, 0.99);
    assert(!dec.pass_meaningful_visits);
    assert(dec.selected_action == dec.raw_reference_action);
  }

  // 4. Test Q margin check failure
  {
    DuneSearchResult res_fail = res;
    res_fail.diagnostics.q_values = {0.8, 0.79, 0.1};
    res_fail.diagnostics.visit_counts = {100, 15, 5};
    ControllerDecision dec = session.SelectControllerAction(state, res_fail, r_val);
    assert(!dec.pass_q_margin);
    assert(dec.selected_action == dec.raw_reference_action);
  }

  // 5. Test stability check failure
  {
    DuneSearchResult res_fail = res;
    res_fail.diagnostics.stability_agreement = false;
    ControllerDecision dec = session.SelectControllerAction(state, res_fail, r_val);
    assert(!dec.pass_stability);
    assert(dec.selected_action == dec.raw_reference_action);
  }

  std::cout << "TestConservativeOverrideCriteria Passed!\n\n";
}

void TestCommitLifecycle() {
  std::cout << "Running TestCommitLifecycle...\n";
  std::shared_ptr<const Game> game = LoadGame("dune_imperium");
  std::shared_ptr<algorithms::Evaluator> evaluator =
      std::make_shared<algorithms::RandomRolloutEvaluator>(20, 42);

  DuneSearchConfig config;
  DuneSearchSession session(config, evaluator, DuneSearchBudgetMode::kFixedSessionSimulations);
  auto state = game->NewInitialState();

  DuneSearchResult res = session.Search(*state);
  double r_val = 0.5;
  ControllerDecision decision = session.SelectControllerAction(*state, res, r_val);

  // Test diagnostic reflection
  DuneSearchResult committed_res = session.CommitAction(decision);
  assert(committed_res.diagnostics.selected_action == decision.selected_action);
  assert(committed_res.diagnostics.raw_reference_action == decision.raw_reference_action);

  std::cout << "TestCommitLifecycle Passed!\n\n";
}

void TestShortWindowRNGSeedContinuity() {
  std::cout << "Running TestShortWindowRNGSeedContinuity...\n";
  auto game = open_spiel::LoadGame("dune_imperium");
  TestRoutingState state(game);

  // Set to a short window role (e.g. kPurchase role: agents remaining = 0)
  state.SetPhaseForTesting(dune_imperium::GamePhase::kRevealTurns);
  state.SetPlayerAgentsRemainingForTesting(0, 0);
  state.SetMockLegalActions({
    dune_imperium::kActionBuyImperiumRow0,
    dune_imperium::kActionEndTurn
  });

  ActionsAndProbs priors;
  for (Action a : state.LegalActions()) {
    priors.push_back({a, 1.0 / state.LegalActions().size()});
  }
  std::vector<double> mock_vals(4, 0.0);
  auto evaluator = std::make_shared<MockEvaluator>(priors, mock_vals);

  DuneSearchConfig config;
  config.max_simulations = 10;
  config.fixed_session_limit = 100;
  config.purchase_combat_budget = 16;

  DuneSearchSession session(config, evaluator, DuneSearchBudgetMode::kFixedSessionSimulations);

  // 1. Initial State: short_cumulative_counter should be 0
  assert(session.short_cumulative_counter() == 0);
  assert(session.short_sims_completed() == 0);

  // 2. Perform search 1 (under kPurchase role)
  DuneSearchResult res1 = RunSessionSearch(session, state);
  int after1 = session.short_cumulative_counter();
  assert(after1 > 0);
  assert(session.short_sims_completed() == after1);
  assert(res1.diagnostics.short_window_cumulative_simulations == after1);

  // 3. Perform search 2 (same role, no transition)
  DuneSearchResult res2 = RunSessionSearch(session, state);
  int after2 = session.short_cumulative_counter();
  assert(after2 == after1 + res2.simulations_completed);
  assert(session.short_sims_completed() == after2);
  assert(res2.diagnostics.short_window_cumulative_simulations == after2);

  // 4. Role transition: from kPurchase to kCombat (another short-window role)
  state.SetPhaseForTesting(dune_imperium::GamePhase::kCombat);
  state.SetPlayerIntriguesForTesting(0, {0});
  state.SetMockLegalActions({
    dune_imperium::kActionPlayIntrigueCombatCard0,
    dune_imperium::kActionCombatPass
  });

  DuneSearchResult res3 = RunSessionSearch(session, state);
  int after3 = session.short_cumulative_counter();
  assert(after3 == after2 + res3.simulations_completed);
  // Verify that short_sims_completed() reset to just res3's completed simulations,
  // but short_cumulative_counter() grew monotonically.
  assert(session.short_sims_completed() == res3.simulations_completed);
  assert(res3.diagnostics.short_window_cumulative_simulations == res3.simulations_completed);
  assert(session.short_cumulative_counter() > session.short_sims_completed());

  std::cout << "TestShortWindowRNGSeedContinuity Passed!\n\n";
}

// Regression test for the Phase 18A Step 2 short-window uniform-policy defect.
// When the fixed session budget is already exhausted, a multi-action
// short-window decision (kPurchase/kCombatIntrigue/kOtherOptional) must NOT be
// routed into RunSearch with zeroed limits (max_time_ms == 0 there returned a
// UNIFORM policy). It must degrade to the raw network prior on the true current
// state, mark used_fallback, and — at temperature 0 — select the prior argmax,
// NOT the lowest legal action id that a uniform policy collapses to under argmax.
void TestShortWindowExhaustedBudgetFallsBackToPrior() {
  std::cout << "Running TestShortWindowExhaustedBudgetFallsBackToPrior...\n";
  auto game = open_spiel::LoadGame("dune_imperium");
  TestRoutingState state(game);

  // A multi-action kPurchase short-window state (all purchase actions).
  state.SetPhaseForTesting(dune_imperium::GamePhase::kRevealTurns);
  state.SetPlayerAgentsRemainingForTesting(0, 0);
  const Action a0 = dune_imperium::kActionBuyImperiumRow0 + 0;
  const Action a1 = dune_imperium::kActionBuyImperiumRow0 + 1;
  const Action a2 = dune_imperium::kActionBuyImperiumRow0 + 2;
  state.SetMockLegalActions({a0, a1, a2});
  assert(ClassifyDuneDecisionRole(state, 0, false) == DuneDecisionRole::kPurchase);

  // Deliberately non-uniform prior whose argmax is the HIGHEST action id, so a
  // uniform-policy bug (argmax -> lowest id a0) is distinguishable from the
  // correct prior argmax (a2).
  ActionsAndProbs priors = {{a0, 0.2}, {a1, 0.3}, {a2, 0.5}};
  std::vector<double> mock_vals(4, 0.0);
  auto evaluator = std::make_shared<MockEvaluator>(priors, mock_vals);

  DuneSearchConfig config;
  config.temperature = 0.0;  // temp 0 == greedy/argmax on candidate paths
  config.purchase_combat_budget = 16;
  config.fixed_continuation_reserve = 0;
  // Zero session budget == "the primary already consumed the whole budget":
  // every subsequent short-window decision computes overall_remaining == 0.
  config.fixed_session_limit = 0;

  DuneSearchSession session(config, evaluator,
                            DuneSearchBudgetMode::kFixedSessionSimulations);

  DuneSearchResult res = session.Search(state);

  // 1. No search ran; this is an explicit fallback with the budget reason.
  assert(res.simulations_completed == 0);
  assert(res.used_fallback == true);
  assert(res.fallback_reason == "short_window_budget_exceeded");

  // 2. Returned policy is the evaluator prior over legal actions, NOT uniform.
  ActionsAndProbs expected_prior = evaluator->Prior(state);
  assert(res.policy.size() == expected_prior.size());
  assert(res.policy.size() == 3u);
  bool is_uniform = true;
  const double uniform_p = 1.0 / res.policy.size();
  for (size_t i = 0; i < res.policy.size(); ++i) {
    assert(res.policy[i].first == expected_prior[i].first);
    AssertAlmostEqual(res.policy[i].second, expected_prior[i].second);
    if (std::abs(res.policy[i].second - uniform_p) > 1e-9) is_uniform = false;
  }
  assert(!is_uniform);  // guards against the RunSearch(0 sims, 0 ms) uniform poison

  // 3. At temperature 0, selection is the prior argmax (a2), not the lowest
  //    legal action id (a0) that a uniform policy collapses to under argmax.
  double r_val = 0.5;
  ControllerDecision dec = session.SelectControllerAction(state, res, r_val);
  assert(dec.raw_reference_action == a2);
  assert(dec.selected_action == a2);
  assert(dec.selected_action != a0);
  assert(dec.confidence_fallback == true);  // used_fallback propagated

  // Close the commit lifecycle cleanly.
  DuneSearchResult committed = session.CommitAction(dec);
  assert(committed.diagnostics.selected_action == a2);

  std::cout << "TestShortWindowExhaustedBudgetFallsBackToPrior Passed!\n\n";
}

// Companion regression test for the telemetry-honesty half of the fix. On a
// PRIMARY decision whose fixed session budget is exhausted, max_sims is clamped
// to 0 and RunSearch degrades to its own low-coverage prior fallback. The
// session must NOT clobber RunSearch's fallback_reason; it records the session
// limit separately in budget_limit_reason and keeps used_fallback set.
void TestExhaustedPrimaryPreservesRunSearchFallbackReason() {
  std::cout << "Running TestExhaustedPrimaryPreservesRunSearchFallbackReason...\n";
  auto game = open_spiel::LoadGame("dune_imperium");
  TestRoutingState state(game);

  // A multi-action kAgentPrimary state.
  state.SetPhaseForTesting(dune_imperium::GamePhase::kAgentTurns);
  state.SetPlayerAgentsRemainingForTesting(0, 1);
  const Action card = dune_imperium::kActionSelectAgentCard0 + 10;
  const Action reveal = dune_imperium::kActionReveal;
  state.SetMockLegalActions({card, reveal});
  assert(ClassifyDuneDecisionRole(state, 0, false) ==
         DuneDecisionRole::kAgentPrimary);

  ActionsAndProbs priors = {{card, 0.7}, {reveal, 0.3}};
  std::vector<double> mock_vals(4, 0.0);
  auto evaluator = std::make_shared<MockEvaluator>(priors, mock_vals);

  DuneSearchConfig config;
  config.temperature = 0.0;
  config.fixed_continuation_reserve = 0;
  config.fixed_session_limit = 0;  // exhausted before any simulation runs

  DuneSearchSession session(config, evaluator,
                            DuneSearchBudgetMode::kFixedSessionSimulations);
  DuneSearchResult res = session.Search(state);

  assert(res.simulations_completed == 0);
  assert(res.used_fallback == true);
  // RunSearch's own reason is preserved (0 covered actions -> low_coverage),
  // NOT overwritten by the session-level limit.
  assert(res.fallback_reason == "low_coverage");
  assert(res.diagnostics.fallback_reason == "low_coverage");
  // The session limit is recorded in the dedicated field instead.
  assert(res.diagnostics.budget_limit_reason == "fixed_session_limit_exceeded");

  session.DiscardPendingAction();
  std::cout << "TestExhaustedPrimaryPreservesRunSearchFallbackReason Passed!\n\n";
}

std::unique_ptr<State> FirstMultiActionDecision(
    const std::shared_ptr<const Game>& game) {
  std::unique_ptr<State> state = game->NewInitialState();
  while (state->IsChanceNode() || state->LegalActions().size() <= 1) {
    if (state->IsChanceNode()) {
      state->ApplyAction(state->ChanceOutcomes().front().first);
    } else {
      state->ApplyAction(state->LegalActions().front());
    }
  }
  return state;
}

std::shared_ptr<MockEvaluator> MakeUniformMockEvaluator(const State& state) {
  ActionsAndProbs priors;
  for (Action action : state.LegalActions()) {
    priors.push_back({action, 1.0 / state.LegalActions().size()});
  }
  return std::make_shared<MockEvaluator>(
      priors, std::vector<double>(state.NumPlayers(), 0.0));
}

void TestDefaultDecisionDepthIsUnchanged() {
  std::cout << "Running TestDefaultDecisionDepthIsUnchanged...\n";
  std::shared_ptr<const Game> game = LoadGame("dune_imperium");
  std::unique_ptr<State> state = FirstMultiActionDecision(game);

  DuneSearchConfig default_config;
  default_config.max_simulations = 30;
  default_config.relative_time_budget_ms =
      std::numeric_limits<double>::infinity();
  DuneSearchConfig explicit_uncapped = default_config;
  explicit_uncapped.max_search_decision_depth = -1;

  DunePUCTISMCTSBot default_bot(default_config,
                                MakeUniformMockEvaluator(*state));
  DunePUCTISMCTSBot explicit_bot(explicit_uncapped,
                                 MakeUniformMockEvaluator(*state));
  DuneSearchResult default_result = default_bot.RunSearch(*state);
  DuneSearchResult explicit_result = explicit_bot.RunSearch(*state);

  assert(default_config.max_search_decision_depth == -1);
  assert(default_result.simulations_completed ==
         explicit_result.simulations_completed);
  assert(default_result.diagnostics.visit_counts ==
         explicit_result.diagnostics.visit_counts);
  assert(default_result.diagnostics.q_values ==
         explicit_result.diagnostics.q_values);
  std::cout << "TestDefaultDecisionDepthIsUnchanged Passed!\n\n";
}

void TestDecisionDepthCapsOneAndTwo() {
  std::cout << "Running TestDecisionDepthCapsOneAndTwo...\n";
  std::shared_ptr<const Game> game = LoadGame("dune_imperium");
  std::unique_ptr<State> state = FirstMultiActionDecision(game);

  for (int cap : {1, 2}) {
    DuneSearchConfig config;
    config.max_simulations = 40;
    config.relative_time_budget_ms =
        std::numeric_limits<double>::infinity();
    config.max_search_decision_depth = cap;
    DunePUCTISMCTSBot bot(config, MakeUniformMockEvaluator(*state));
    DuneSearchResult result = bot.RunSearch(*state);
    assert(result.simulations_completed == config.max_simulations);
    assert(result.diagnostics.max_decision_depth <= cap);
    assert(result.diagnostics.mean_decision_depth <= cap);
  }
  std::cout << "TestDecisionDepthCapsOneAndTwo Passed!\n\n";
}

void TestDecisionDepthSimulationAccounting() {
  std::cout << "Running TestDecisionDepthSimulationAccounting...\n";
  std::shared_ptr<const Game> game = LoadGame("dune_imperium");
  std::unique_ptr<State> state = FirstMultiActionDecision(game);

  DuneSearchConfig config;
  config.max_simulations = 200;
  config.relative_time_budget_ms = std::numeric_limits<double>::infinity();
  config.max_search_decision_depth = 2;
  DunePUCTISMCTSBot bot(config, MakeUniformMockEvaluator(*state));
  DuneSearchResult result = bot.RunSearch(*state);
  assert(result.simulations_completed <= 200);
  assert(result.simulations_completed == config.max_simulations);
  std::cout << "TestDecisionDepthSimulationAccounting Passed!\n\n";
}

// Item 2a: per-root Dirichlet-noise seed. Distinct roots -> distinct seeds (the
// bug was a constant position argument that seeded every root identically);
// identical inputs -> identical seed; each input dimension perturbs the stream.
void TestDeriveRootNoiseSeed() {
  std::cout << "Running TestDeriveRootNoiseSeed...\n";
  const uint64_t s = 12345;
  assert(DeriveRootNoiseSeed(s, 0, 0, "rootA") == DeriveRootNoiseSeed(s, 0, 0, "rootA"));
  // The fix: same seed/counter/player, DIFFERENT state key -> different stream.
  assert(DeriveRootNoiseSeed(s, 0, 0, "rootA") != DeriveRootNoiseSeed(s, 0, 0, "rootB"));
  assert(DeriveRootNoiseSeed(s, 0, 0, "rootA") != DeriveRootNoiseSeed(s, 1, 0, "rootA"));      // search_count
  assert(DeriveRootNoiseSeed(s, 0, 0, "rootA") != DeriveRootNoiseSeed(s, 0, 1, "rootA"));      // player
  assert(DeriveRootNoiseSeed(s, 0, 0, "rootA") != DeriveRootNoiseSeed(s + 1, 0, 0, "rootA"));  // config seed
  std::cout << "TestDeriveRootNoiseSeed Passed!\n\n";
}

// Item 1b (telemetry comparability): SearchDiagnostics must expose the
// UNTRANSFORMED network prior in `raw_priors` — the baseline item-4 KL is
// defined against — while `priors` carries whatever the tree actually searched
// with. WO-15 widened this: `raw_priors` is now populated on every searched
// root and strips root_prior_temperature as well as Dirichlet noise. It used to
// appear only under noise and to snapshot the POST-temperature prior, so a
// non-unit root_prior_temperature made the "raw" baseline not raw.
void TestRawPriorsCapturedUnderNoise() {
  std::cout << "Running TestRawPriorsCapturedUnderNoise...\n";
  std::shared_ptr<const Game> game = LoadGame("dune_imperium");
  std::unique_ptr<State> state = game->NewInitialState();
  while (state->IsChanceNode()) {
    state->ApplyAction(state->ChanceOutcomes().front().first);
  }
  std::vector<Action> legal_actions = state->LegalActions();
  assert(legal_actions.size() >= 2);

  // Distinct, normalized network priors over exactly the legal actions, so the
  // pre-noise baseline is unambiguous and differs from any post-noise mixture.
  ActionsAndProbs mock_priors;
  double sum = 0.0;
  for (size_t i = 0; i < legal_actions.size(); ++i) {
    double p = 0.1 + 0.1 * i;
    mock_priors.push_back({legal_actions[i], p});
    sum += p;
  }
  for (auto& ap : mock_priors) ap.second /= sum;
  std::vector<double> mock_values = {0.5, 0.5, 0.5, 0.5};
  auto evaluator = std::make_shared<MockEvaluator>(mock_priors, mock_values);

  // --- Noise ON: pilot exploration package. ---
  DuneSearchConfig noised;
  noised.seed = 7;
  noised.max_simulations = 32;
  noised.dirichlet_epsilon = 0.25;
  noised.dirichlet_alpha_total = 10.83;
  noised.check_strategic_state = false;
  DunePUCTISMCTSBot bot_noised(noised, evaluator);
  SearchDiagnostics d = bot_noised.RunSearch(*state).diagnostics;

  // raw_priors is populated, aligned to actions, and a valid distribution.
  assert(d.raw_priors.size() == d.actions.size());
  assert(d.priors.size() == d.actions.size());
  double raw_sum = 0.0;
  for (double rp : d.raw_priors) { assert(rp >= 0.0); raw_sum += rp; }
  assert(std::abs(raw_sum - 1.0) < 1e-6);

  // raw_priors equals the noise-free network prior (what `priors` would be with
  // noise off), NOT the post-noise tree prior.
  for (size_t i = 0; i < d.actions.size(); ++i) {
    assert(std::abs(d.raw_priors[i] - mock_priors[i].second) < 1e-6);
  }
  // The tree prior actually WAS perturbed (so measuring KL against `priors`
  // would be the bug this fix corrects): L1(priors, raw_priors) is non-trivial.
  double l1 = 0.0;
  for (size_t i = 0; i < d.actions.size(); ++i) {
    l1 += std::abs(d.priors[i] - d.raw_priors[i]);
  }
  assert(l1 > 1e-6);

  // --- Noise OFF, temperature 1: raw_priors and priors both equal the network
  // prior. Consumers using `raw_priors.empty() ? priors : raw_priors` therefore
  // read exactly what they read before this change. ---
  DuneSearchConfig plain;
  plain.seed = 7;
  plain.max_simulations = 32;
  plain.check_strategic_state = false;  // dirichlet_epsilon defaults to 0.0
  DunePUCTISMCTSBot bot_plain(plain, evaluator);
  SearchDiagnostics d0 = bot_plain.RunSearch(*state).diagnostics;
  assert(d0.raw_priors.size() == d0.actions.size());
  for (size_t i = 0; i < d0.actions.size(); ++i) {
    assert(std::abs(d0.priors[i] - mock_priors[i].second) < 1e-6);
    assert(std::abs(d0.raw_priors[i] - d0.priors[i]) < 1e-12);
  }

  // --- Noise OFF but root_prior_temperature != 1: this is the case the old
  // contract got wrong. The tree prior is flattened, yet `raw_priors` must
  // still be the untransformed network prior, or the item-4 KL baseline is
  // measured against a reshaped distribution. ---
  DuneSearchConfig tempered;
  tempered.seed = 7;
  tempered.max_simulations = 32;
  tempered.root_prior_temperature = 2.0;
  tempered.check_strategic_state = false;
  DunePUCTISMCTSBot bot_tempered(tempered, evaluator);
  SearchDiagnostics dt = bot_tempered.RunSearch(*state).diagnostics;
  assert(dt.raw_priors.size() == dt.actions.size());
  double raw_sum_t = 0.0;
  for (double rp : dt.raw_priors) raw_sum_t += rp;
  assert(std::abs(raw_sum_t - 1.0) < 1e-6);
  double l1_t = 0.0;
  for (size_t i = 0; i < dt.actions.size(); ++i) {
    assert(std::abs(dt.raw_priors[i] - mock_priors[i].second) < 1e-6);
    l1_t += std::abs(dt.priors[i] - dt.raw_priors[i]);
  }
  assert(l1_t > 1e-6);  // the tree prior really was reshaped

  std::cout << "  noised L1(priors,raw)=" << l1
            << "  tempered L1(priors,raw)=" << l1_t
            << "  (raw_priors matches the network prior in both)\n";
  std::cout << "TestRawPriorsCapturedUnderNoise Passed!\n\n";
}

// The companion to the test above: that one proves `raw_priors` is CORRECT,
// this one proves the four raw-named diagnostics actually CONSUME it. They were
// all computed from `diag.priors`, so with the exploration package on they
// reported against the post-Dirichlet / temperature-scaled tree prior:
// `raw_to_search_policy_kl` measured the wrong divergence, and the prior
// rank/argmax that `chosen_action_raw_prior_rank` and
// `action_changed_vs_raw_argmax` derive from moved with the noise draw.
void TestRawDiagnosticsUseRawPriorBaseline() {
  std::cout << "Running TestRawDiagnosticsUseRawPriorBaseline...\n";
  std::shared_ptr<const Game> game = LoadGame("dune_imperium");
  std::unique_ptr<State> state = game->NewInitialState();
  while (state->IsChanceNode()) {
    state->ApplyAction(state->ChanceOutcomes().front().first);
  }
  std::vector<Action> legal_actions = state->LegalActions();
  assert(legal_actions.size() >= 2);

  // Strictly increasing network prior: the raw argmax is unambiguously the last
  // legal action and every raw rank is distinct, so a baseline swap moves the
  // rank/argmax fields rather than landing on a tie.
  ActionsAndProbs mock_priors;
  double sum = 0.0;
  for (size_t i = 0; i < legal_actions.size(); ++i) {
    double p = 0.1 + 0.1 * i;
    mock_priors.push_back({legal_actions[i], p});
    sum += p;
  }
  for (auto& ap : mock_priors) ap.second /= sum;
  std::vector<double> mock_values = {0.5, 0.5, 0.5, 0.5};
  auto evaluator = std::make_shared<MockEvaluator>(mock_priors, mock_values);

  // Recompute a diagnostic set from an explicit baseline, mirroring
  // GetRootDiagnostics. Called with `raw_priors` (the contract) and with
  // `priors` (the defect) to prove the assertions below discriminate.
  struct Expected {
    double kl;
    double chosen_prob;
    int rank;
    bool changed_vs_argmax;
  };
  auto recompute = [](const SearchDiagnostics& d,
                      const std::vector<double>& baseline) {
    const size_t n = d.actions.size();
    double total_visits = 0.0;
    for (int v : d.visit_counts) total_visits += v;
    std::vector<double> search_probs(n, 1.0 / n);
    if (total_visits > 0.0) {
      for (size_t i = 0; i < n; ++i) search_probs[i] = d.visit_counts[i] / total_visits;
    }
    Expected e;
    e.kl = 0.0;
    for (size_t i = 0; i < n; ++i) {
      if (search_probs[i] > 0.0) {
        e.kl += search_probs[i] *
                std::log(search_probs[i] / std::max(baseline[i], 1e-12));
      }
    }
    // RunSearch reports with chosen_action unset, so the chosen index is the
    // visit argmax.
    size_t chosen_idx = 0;
    double max_prob = -1.0;
    for (size_t i = 0; i < n; ++i) {
      if (search_probs[i] > max_prob) { max_prob = search_probs[i]; chosen_idx = i; }
    }
    e.chosen_prob = baseline[chosen_idx];
    e.rank = 1;
    for (size_t i = 0; i < n; ++i) {
      if (baseline[i] > e.chosen_prob) e.rank++;
    }
    size_t argmax_idx = 0;
    double max_baseline = -1.0;
    for (size_t i = 0; i < n; ++i) {
      if (baseline[i] > max_baseline) { max_baseline = baseline[i]; argmax_idx = i; }
    }
    e.changed_vs_argmax = (chosen_idx != argmax_idx);
    return e;
  };

  // Each arm is a transformation that makes the tree prior differ from the raw
  // prior. epsilon=1.0 replaces the tree prior with pure noise, which reorders
  // it (temperature scaling is monotone, so it can only move the value-valued
  // fields).
  struct Arm { const char* name; double epsilon; double temperature; };
  const std::vector<Arm> arms = {
      {"dirichlet_eps=0.25", 0.25, 1.0},
      {"dirichlet_eps=1.0", 1.0, 1.0},
      {"root_prior_temperature=2.0", 0.0, 2.0},
      {"root_prior_temperature=0.5", 0.0, 0.5},
  };

  bool any_rank_discriminated = false;
  bool any_argmax_discriminated = false;

  for (const Arm& arm : arms) {
    // Several seeds per arm: the noise draw is what moves the prior ordering,
    // so a single seed could pass by landing on an unreordered draw.
    for (int seed : {7, 11, 13, 17, 23}) {
      DuneSearchConfig cfg;
      cfg.seed = seed;
      cfg.max_simulations = 32;
      cfg.dirichlet_epsilon = arm.epsilon;
      if (arm.epsilon > 0.0) cfg.dirichlet_alpha_total = 10.83;
      cfg.root_prior_temperature = arm.temperature;
      cfg.check_strategic_state = false;
      DunePUCTISMCTSBot bot(cfg, evaluator);
      SearchDiagnostics d = bot.RunSearch(*state).diagnostics;

      assert(d.raw_priors.size() == d.actions.size());
      assert(d.priors.size() == d.actions.size());

      // The tree prior really was transformed; otherwise this arm proves
      // nothing and the assertions below would hold under the defect too.
      double l1 = 0.0;
      for (size_t i = 0; i < d.actions.size(); ++i) {
        l1 += std::abs(d.priors[i] - d.raw_priors[i]);
      }
      assert(l1 > 1e-6);

      const Expected want = recompute(d, d.raw_priors);
      const Expected defect = recompute(d, d.priors);

      // The contract: all four fields are computed against the raw prior.
      assert(std::abs(d.raw_to_search_policy_kl - want.kl) < 1e-9);
      assert(std::abs(d.chosen_action_raw_prior_probability - want.chosen_prob) < 1e-12);
      assert(d.chosen_action_raw_prior_rank == want.rank);
      assert(d.action_changed_vs_raw_argmax == want.changed_vs_argmax);

      // The chosen-action prior probability is read straight off the baseline,
      // so it separates the two on every arm.
      assert(std::abs(want.chosen_prob - defect.chosen_prob) > 1e-9);
      // KL against a reshaped prior is a different number on every arm too.
      assert(std::abs(want.kl - defect.kl) > 1e-9);

      if (want.rank != defect.rank) any_rank_discriminated = true;
      if (want.changed_vs_argmax != defect.changed_vs_argmax) any_argmax_discriminated = true;
    }
  }

  // The rank/argmax fields only diverge when the transformation REORDERS the
  // prior, which temperature scaling cannot do. Assert some arm/seed actually
  // exercised that, so these two fields are covered rather than incidentally
  // passing on ties.
  assert(any_rank_discriminated);
  assert(any_argmax_discriminated);

  std::cout << "  4 fields x " << arms.size()
            << " transformation arms x 5 seeds; rank and argmax both"
               " discriminated raw vs tree prior\n";
  std::cout << "TestRawDiagnosticsUseRawPriorBaseline Passed!\n\n";
}

// ---------------------------------------------------------------------------
// WO-15 regression tests (search findings 2, 3, 4, 5, 7, 9)
// ---------------------------------------------------------------------------

// A player decision with at least two legal actions, plus a normalized,
// deliberately NON-uniform mock prior over exactly those actions. Non-uniform
// matters: the historical starvation bug returned the uniform policy, and a
// uniform mock prior would make the bug indistinguishable from the fix.
struct SkewedPriorFixture {
  std::shared_ptr<const Game> game;
  std::unique_ptr<State> state;
  std::vector<Action> legal_actions;
  ActionsAndProbs priors;
  std::shared_ptr<MockEvaluator> evaluator;

  SkewedPriorFixture() {
    game = LoadGame("dune_imperium");
    state = game->NewInitialState();
    while (state->IsChanceNode()) {
      state->ApplyAction(state->ChanceOutcomes().front().first);
    }
    legal_actions = state->LegalActions();
    assert(legal_actions.size() >= 3);
    double sum = 0.0;
    for (size_t i = 0; i < legal_actions.size(); ++i) {
      double p = 1.0 + static_cast<double>(i);  // strictly increasing
      priors.push_back({legal_actions[i], p});
      sum += p;
    }
    for (auto& ap : priors) ap.second /= sum;
    evaluator = std::make_shared<MockEvaluator>(
        priors, std::vector<double>{0.5, 0.5, 0.5, 0.5});
  }
};

bool IsUniformPolicy(const ActionsAndProbs& policy) {
  if (policy.empty()) return false;
  double u = 1.0 / policy.size();
  for (const auto& ap : policy) {
    if (std::abs(ap.second - u) > 1e-9) return false;
  }
  return true;
}

// Search finding 2: the legacy seed-first constructor used positional
// braced-init, so Phase 18B's three inserted KataGo fields shifted the trailing
// bools into `dirichlet_alpha_total` / `forced_playouts_k`. Every field is
// inspected here — both the ones the signature carries and the ones it must
// leave at their declared defaults — so a future field insertion fails loudly
// instead of silently rebinding someone's value.
void TestLegacyConstructorFieldAlignment() {
  std::cout << "Running TestLegacyConstructorFieldAlignment...\n";
  SkewedPriorFixture f;
  const DuneSearchConfig defaults;

  for (bool obs_string : {true, false}) {
    for (bool verbose : {true, false}) {
      DunePUCTISMCTSBot bot(
          /*seed=*/98765, f.evaluator, /*puct_c=*/0.77, /*max_simulations=*/37,
          /*max_world_samples=*/5, /*temperature=*/0.6,
          /*dirichlet_epsilon=*/0.11, /*dirichlet_alpha=*/0.42,
          /*utility_divisor=*/3.5, obs_string,
          DuneISMCTSFinalPolicyType::kMaxValue,
          /*use_opponent_model=*/true, /*opponent_temperature=*/0.25, verbose);
      const DuneSearchConfig& c = bot.GetConfig();

      // Fields the legacy signature carries.
      assert(c.max_simulations == 37);
      assert(c.puct_c == 0.77);
      assert(c.opponent_mode == SearchOpponentMode::kPolicy);
      assert(c.temperature == 0.6);
      assert(c.opponent_temperature == 0.25);
      assert(c.max_world_samples == 5);
      assert(c.utility_divisor == 3.5);
      assert(c.seed == 98765ULL);
      assert(c.final_policy_type == DuneISMCTSFinalPolicyType::kMaxValue);
      assert(c.dirichlet_epsilon == 0.11);
      assert(c.dirichlet_alpha == 0.42);
      assert(c.use_observation_string == obs_string);
      assert(c.verbose_diagnostics == verbose);

      // Fields the legacy signature fixes.
      assert(c.relative_time_budget_ms == 10000.0);
      assert(c.max_nodes == 50000);
      assert(c.min_visit_threshold == 2);
      assert(c.covered_prior_threshold == 0.50);

      // THE REGRESSION: the Phase 18B KataGo package must keep its defaults.
      // Under the old positional init, use_observation_string=true landed here
      // as dirichlet_alpha_total=1.0, silently switching on inverse-legal-count
      // alpha scaling for anyone who enabled noise on this path.
      assert(c.dirichlet_alpha_total == defaults.dirichlet_alpha_total);
      assert(c.dirichlet_alpha_total == 0.0);
      assert(c.forced_playouts_k == defaults.forced_playouts_k);
      assert(c.forced_playouts_k == 0.0);
      assert(c.root_noise_fpu_zero == defaults.root_noise_fpu_zero);
      assert(c.root_noise_fpu_zero == false);

      // Every remaining field must equal its declared default.
      assert(c.check_strategic_state == defaults.check_strategic_state);
      assert(c.root_prior_temperature == defaults.root_prior_temperature);
      assert(c.training_root_prior_temperature ==
             defaults.training_root_prior_temperature);
      assert(c.fixed_continuation_reserve == defaults.fixed_continuation_reserve);
      assert(c.purchase_combat_budget == defaults.purchase_combat_budget);
      assert(c.live_continuation_reserve_seconds ==
             defaults.live_continuation_reserve_seconds);
      assert(c.fixed_session_limit == defaults.fixed_session_limit);
      assert(c.model_checkpoint_path == defaults.model_checkpoint_path);
      assert(c.conservative_override_enabled ==
             defaults.conservative_override_enabled);
      assert(c.conservative_covered_prior_threshold ==
             defaults.conservative_covered_prior_threshold);
      assert(c.conservative_meaningful_visit_threshold ==
             defaults.conservative_meaningful_visit_threshold);
      assert(c.conservative_q_margin_threshold ==
             defaults.conservative_q_margin_threshold);
      assert(c.conservative_stability_checkpoint_fraction ==
             defaults.conservative_stability_checkpoint_fraction);
      assert(c.conservative_continuation_overrides_disabled ==
             defaults.conservative_continuation_overrides_disabled);
      assert(c.max_search_decision_depth == defaults.max_search_decision_depth);
    }
  }
  std::cout << "TestLegacyConstructorFieldAlignment Passed!\n\n";
}

// Search findings 3 and 5: min_visit_threshold <= 0 used to let a zero-visit
// root pass the coverage gate trivially, which returned the UNIFORM policy
// (the historical starvation bug) and evaluated return_sum/visits as 0/0.
void TestZeroVisitAndThresholdZeroContract() {
  std::cout << "Running TestZeroVisitAndThresholdZeroContract...\n";

  // The contract itself, stated once.
  assert(EffectiveMinVisitThreshold(0) == 1);
  assert(EffectiveMinVisitThreshold(-7) == 1);
  assert(EffectiveMinVisitThreshold(1) == 1);
  assert(EffectiveMinVisitThreshold(2) == 2);

  SkewedPriorFixture f;
  for (int threshold : {0, -1, 1, 2}) {
    DuneSearchConfig config;
    config.max_simulations = 0;  // root expanded, zero visits anywhere
    config.min_visit_threshold = threshold;
    config.check_strategic_state = false;
    DunePUCTISMCTSBot bot(config, f.evaluator);
    DuneSearchResult res = bot.RunSearch(*f.state);

    // A zero-visit root is starved regardless of how the threshold is set.
    assert(res.used_fallback);
    assert(res.diagnostics.num_covered_actions == 0);
    AssertAlmostEqual(res.diagnostics.covered_prior_mass, 0.0);

    // The policy is the raw network prior — finite, normalized, NOT uniform.
    assert(res.policy.size() == f.legal_actions.size());
    double sum = 0.0;
    for (const auto& ap : res.policy) {
      assert(std::isfinite(ap.second));
      sum += ap.second;
    }
    AssertAlmostEqual(sum, 1.0);
    assert(!IsUniformPolicy(res.policy));
    for (size_t i = 0; i < res.policy.size(); ++i) {
      assert(res.policy[i].first == f.priors[i].first);
      AssertAlmostEqual(res.policy[i].second, f.priors[i].second);
    }

    // No NaN anywhere in the diagnostics the collector and the forced-playout
    // pruner read. A NaN Q compares false against everything, which silently
    // disables PruneForcedPlayouts for that child.
    for (double q : res.diagnostics.q_values) assert(std::isfinite(q));
    for (double p : res.diagnostics.priors) assert(std::isfinite(p));
    for (double p : res.diagnostics.raw_priors) assert(std::isfinite(p));
  }

  // The public diagnostics entry point clamps too: a caller passing 0 directly
  // must not be able to reach 0/0.
  DuneSearchConfig config;
  config.max_simulations = 0;
  config.check_strategic_state = false;
  DunePUCTISMCTSBot bot(config, f.evaluator);
  bot.RunSearch(*f.state);
  for (int threshold : {0, -3}) {
    SearchDiagnostics diag = bot.GetRootDiagnostics(*f.state, threshold);
    assert(diag.num_covered_actions == 0);
    for (double q : diag.q_values) assert(std::isfinite(q));
  }

  // Defense in depth at the exact former bug site: GetFinalPolicy on a
  // zero-visit node returns the raw prior, never uniform.
  DuneISMCTSNode* root = TestBotAccessor::GetRootNode(bot);
  assert(root != nullptr);
  ActionsAndProbs final_policy =
      TestBotAccessor::RunGetFinalPolicy(bot, *f.state, root);
  assert(!IsUniformPolicy(final_policy));
  for (size_t i = 0; i < final_policy.size(); ++i) {
    AssertAlmostEqual(final_policy[i].second, f.priors[i].second);
  }

  std::cout << "TestZeroVisitAndThresholdZeroContract Passed!\n\n";
}

// Search finding 4: at a noised root the stored tree prior is a 25% Dirichlet
// mixture (and may also carry root_prior_temperature flattening). The
// low-coverage fallback must degrade to the RAW network prior, or a consumer
// that plays the fallback policy executes exploration noise as if it were the
// policy.
void TestNoisedLowCoverageFallsBackToRawPrior() {
  std::cout << "Running TestNoisedLowCoverageFallsBackToRawPrior...\n";
  SkewedPriorFixture f;

  // Pilot exploration package, starved so the coverage gate always trips.
  DuneSearchConfig noised;
  noised.seed = 7;
  noised.max_simulations = 2;
  noised.min_visit_threshold = 2;
  noised.dirichlet_epsilon = 0.25;
  noised.dirichlet_alpha_total = 10.83;
  noised.check_strategic_state = false;
  DunePUCTISMCTSBot bot(noised, f.evaluator);
  DuneSearchResult res = bot.RunSearch(*f.state);

  assert(res.used_fallback);
  assert(res.fallback_reason == "low_coverage");

  DuneISMCTSNode* root = TestBotAccessor::GetRootNode(bot);
  assert(root != nullptr);
  assert(root->dirichlet_noise_applied);  // the premise: noise really ran

  // The fallback policy is the raw network prior...
  assert(res.policy.size() == f.legal_actions.size());
  for (size_t i = 0; i < res.policy.size(); ++i) {
    assert(res.policy[i].first == f.priors[i].first);
    AssertAlmostEqual(res.policy[i].second, f.priors[i].second);
  }
  // ...and that is a real distinction: the tree prior it used to return is
  // measurably different.
  ActionsAndProbs tree_prior =
      TestBotAccessor::RunFilterPriors(bot, root, f.legal_actions);
  double l1 = 0.0;
  for (size_t i = 0; i < tree_prior.size(); ++i) {
    l1 += std::abs(tree_prior[i].second - res.policy[i].second);
  }
  assert(l1 > 1e-6);

  // Same requirement under root-prior temperature, with noise off: the
  // fallback strips the temperature reshaping too.
  DuneSearchConfig tempered;
  tempered.seed = 7;
  tempered.max_simulations = 2;
  tempered.min_visit_threshold = 2;
  tempered.root_prior_temperature = 2.0;
  tempered.check_strategic_state = false;
  DunePUCTISMCTSBot bot_t(tempered, f.evaluator);
  DuneSearchResult res_t = bot_t.RunSearch(*f.state);
  assert(res_t.used_fallback);
  assert(res_t.fallback_reason == "low_coverage");
  for (size_t i = 0; i < res_t.policy.size(); ++i) {
    AssertAlmostEqual(res_t.policy[i].second, f.priors[i].second);
  }
  DuneISMCTSNode* root_t = TestBotAccessor::GetRootNode(bot_t);
  ActionsAndProbs tree_prior_t =
      TestBotAccessor::RunFilterPriors(bot_t, root_t, f.legal_actions);
  double l1_t = 0.0;
  for (size_t i = 0; i < tree_prior_t.size(); ++i) {
    l1_t += std::abs(tree_prior_t[i].second - res_t.policy[i].second);
  }
  assert(l1_t > 1e-6);

  std::cout << "  noised L1(tree,fallback)=" << l1
            << "  tempered L1(tree,fallback)=" << l1_t << "\n";
  std::cout << "TestNoisedLowCoverageFallsBackToRawPrior Passed!\n\n";
}

// Search finding 7: a cold N-simulation root must perform N root descents. It
// used to perform N-1 — the root's own expansion consumed simulation 0 — while
// still reporting N against the session budget.
void TestColdRootPerformsFullBudget() {
  std::cout << "Running TestColdRootPerformsFullBudget...\n";
  SkewedPriorFixture f;

  for (int budget : {1, 2, 7, 16, 64}) {
    DuneSearchConfig config;
    config.max_simulations = budget;
    config.relative_time_budget_ms = std::numeric_limits<double>::infinity();
    config.check_strategic_state = false;
    DunePUCTISMCTSBot bot(config, f.evaluator);
    DuneSearchResult res = bot.RunSearch(*f.state);

    assert(res.simulations_completed == budget);
    DuneISMCTSNode* root = TestBotAccessor::GetRootNode(bot);
    assert(root != nullptr);
    // Every advertised simulation is a root descent that actually happened.
    assert(root->total_visits == budget);
    int summed = 0;
    for (const auto& child : root->child_info) summed += child.second.visits;
    assert(summed == budget);
    // ...and the visit counts the collector reads agree.
    int diag_summed = 0;
    for (int v : res.diagnostics.visit_counts) diag_summed += v;
    assert(diag_summed == budget);
  }

  // A continuation re-enters an already-expanded root and keeps accumulating;
  // it must not pay the expansion cost a second time.
  DuneSearchConfig config;
  config.max_simulations = 8;
  config.relative_time_budget_ms = std::numeric_limits<double>::infinity();
  config.check_strategic_state = false;
  DunePUCTISMCTSBot bot(config, f.evaluator);
  bot.RunSearch(*f.state, /*max_sims=*/8, /*max_time_ms=*/-1.0,
                /*start_sim_index=*/0);
  DuneSearchResult cont =
      bot.RunSearch(*f.state, /*max_sims=*/5, /*max_time_ms=*/-1.0,
                    /*start_sim_index=*/8);
  assert(cont.simulations_completed == 5);
  DuneISMCTSNode* root = TestBotAccessor::GetRootNode(bot);
  assert(root->total_visits == 13);

  std::cout << "TestColdRootPerformsFullBudget Passed!\n\n";
}

// Search finding 9: the PUCT tie-break stream was keyed only on
// (seed, kStreamMCTS, total_visits), so every node at the same local visit
// count drew the identical tie-break — a correlated repeating pattern across
// the whole tree, worst at fresh nodes where uniform priors make ties the norm.
void TestTieBreakSeedIncludesNodeIdentity() {
  std::cout << "Running TestTieBreakSeedIncludesNodeIdentity...\n";

  // Unit level: each input dimension perturbs the stream, and equal inputs
  // reproduce (deterministic replay).
  assert(DeriveNodeKeyHash(0, "keyA") == DeriveNodeKeyHash(0, "keyA"));
  assert(DeriveNodeKeyHash(0, "keyA") != DeriveNodeKeyHash(0, "keyB"));
  assert(DeriveNodeKeyHash(0, "keyA") != DeriveNodeKeyHash(1, "keyA"));
  assert(DeriveTieBreakSeed(42, 111, 7) == DeriveTieBreakSeed(42, 111, 7));
  assert(DeriveTieBreakSeed(42, 111, 7) != DeriveTieBreakSeed(42, 222, 7));  // node
  assert(DeriveTieBreakSeed(42, 111, 7) != DeriveTieBreakSeed(42, 111, 8));  // visits
  assert(DeriveTieBreakSeed(43, 111, 7) != DeriveTieBreakSeed(42, 111, 7));  // seed

  // Behavioral: two distinct nodes, identical tied children, identical local
  // visit counts. They must not be forced into the same draw at every count.
  SkewedPriorFixture f;
  DuneSearchConfig config;
  config.puct_c = 1.0;
  config.check_strategic_state = false;
  DunePUCTISMCTSBot bot(config, f.evaluator);

  std::vector<Action> tied = {f.legal_actions[0], f.legal_actions[1]};
  auto make_tied_node = [&](uint64_t key_hash) {
    DuneISMCTSNode node;
    node.priors_initialized = true;
    node.key_hash = key_hash;
    for (Action a : tied) {
      DuneChildInfo& child = node.child_info[a];
      child.prior = 0.5;      // exact PUCT tie between two unvisited children
      child.raw_prior = 0.5;
    }
    return node;
  };
  DuneISMCTSNode node_a = make_tied_node(DeriveNodeKeyHash(0, "nodeA"));
  DuneISMCTSNode node_b = make_tied_node(DeriveNodeKeyHash(0, "nodeB"));

  int disagreements = 0;
  for (int v = 0; v <= 31; ++v) {
    node_a.total_visits = v;
    node_b.total_visits = v;
    Action pick_a = TestBotAccessor::RunSelectTree(bot, &node_a, tied);
    Action pick_b = TestBotAccessor::RunSelectTree(bot, &node_b, tied);
    // Determinism is preserved: the same node at the same count replays.
    assert(TestBotAccessor::RunSelectTree(bot, &node_a, tied) == pick_a);
    if (pick_a != pick_b) ++disagreements;
  }
  assert(disagreements > 0);

  // Deterministic replay end to end: two bots with the same seed and config
  // produce byte-identical policies and visit counts.
  DuneSearchConfig replay;
  replay.max_simulations = 24;
  replay.seed = 4242;
  replay.relative_time_budget_ms = std::numeric_limits<double>::infinity();
  replay.check_strategic_state = false;
  DunePUCTISMCTSBot bot1(replay, f.evaluator);
  DunePUCTISMCTSBot bot2(replay, f.evaluator);
  DuneSearchResult r1 = bot1.RunSearch(*f.state);
  DuneSearchResult r2 = bot2.RunSearch(*f.state);
  assert(r1.policy == r2.policy);
  assert(r1.diagnostics.visit_counts == r2.diagnostics.visit_counts);
  assert(r1.diagnostics.q_values == r2.diagnostics.q_values);
  assert(r1.simulations_completed == r2.simulations_completed);

  std::cout << "  tie-break disagreements across 32 visit counts: "
            << disagreements << "/32\n";
  std::cout << "TestTieBreakSeedIncludesNodeIdentity Passed!\n\n";
}

}  // namespace

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
  open_spiel::TestOpponentTemperatureIsolation();
  open_spiel::TestCorrectedTerminalRounds();
  open_spiel::TestFixedSimulationCompleteness();
  open_spiel::TestRootPriorTemperatureScaling();
  open_spiel::TestClassifierSearchRouting();
  open_spiel::TestSearchSessionControls();
  open_spiel::TestTransactionalMCTSAndTreeFallback();
  open_spiel::TestInvariantGatesAndScenarios();
  open_spiel::TestMoreRoutingAndScenarioInvariants();
  open_spiel::TestPersistentSessionScenarios();
  open_spiel::TestFidelityGateScenarios();
  open_spiel::TestFallbackDeterministic();
  open_spiel::TestCommitLifecycle();
  open_spiel::TestConservativeOverrideCriteria();
  open_spiel::TestShortWindowRNGSeedContinuity();
  open_spiel::TestShortWindowExhaustedBudgetFallsBackToPrior();
  open_spiel::TestExhaustedPrimaryPreservesRunSearchFallbackReason();
  open_spiel::TestDefaultDecisionDepthIsUnchanged();
  open_spiel::TestDecisionDepthCapsOneAndTwo();
  open_spiel::TestDecisionDepthSimulationAccounting();
  open_spiel::TestDeriveRootNoiseSeed();
  open_spiel::TestRawPriorsCapturedUnderNoise();
  open_spiel::TestRawDiagnosticsUseRawPriorBaseline();
  // WO-15 (search findings 2, 3, 4, 5, 7, 9)
  open_spiel::TestLegacyConstructorFieldAlignment();
  open_spiel::TestZeroVisitAndThresholdZeroContract();
  open_spiel::TestNoisedLowCoverageFallsBackToRawPrior();
  open_spiel::TestColdRootPerformsFullBudget();
  open_spiel::TestTieBreakSeedIncludesNodeIdentity();
  std::cout << "All Dune PUCT IS-MCTS tests completed successfully!\n";
  return 0;
}
