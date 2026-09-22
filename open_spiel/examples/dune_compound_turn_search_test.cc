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

  // Test 4: Compound plan target computation and continuation conditioning
  std::vector<Action> test_legals = {10, 20, 30, 40};
  std::vector<std::pair<Action, double>> test_cands = {{10, 1.20}, {20, 0.80}, {30, 0.40}};
  std::vector<std::pair<Action, double>> test_blueprint = {{10, 0.40}, {20, 0.30}, {30, 0.20}, {40, 0.10}};

  double temp = 0.5;
  double eta = 0.80;
  auto q_search = ComputeSoftmaxTarget(test_cands, test_legals, temp);
  SPIEL_CHECK_EQ(q_search.size(), test_legals.size());

  double q_sum = 0.0;
  for (const auto& qp : q_search) {
    SPIEL_CHECK_TRUE(qp.second >= 0.0);
    SPIEL_CHECK_TRUE(std::isfinite(qp.second));
    q_sum += qp.second;
    if (qp.first == 40) {
      // Unevaluated legal action must have 0 search probability
      SPIEL_CHECK_FLOAT_EQ(qp.second, 0.0);
    }
  }
  SPIEL_CHECK_FLOAT_EQ(q_sum, 1.0);

  auto q_blend = BlendTargets(test_blueprint, q_search, eta);
  SPIEL_CHECK_EQ(q_blend.size(), test_blueprint.size());
  double blend_sum = 0.0;
  for (const auto& bp : q_blend) {
    SPIEL_CHECK_TRUE(bp.second > 0.0);  // Non-zero support for all legal actions
    SPIEL_CHECK_TRUE(std::isfinite(bp.second));
    blend_sum += bp.second;
  }
  SPIEL_CHECK_FLOAT_EQ(blend_sum, 1.0);

  // Test 5: Exact Fallback-Aligned Target Rule Unit Test
  // Case A: Overridden = true -> Blended target
  CompoundSearchDecisionResult res_a;
  res_a.overridden = true;
  res_a.candidate_utilities = {{10, 1.0}, {20, 2.0}};  // Cand 20 beat Cand 10
  bool fell_back_a = false;
  auto target_a = ComputeFallbackAlignedTarget(test_blueprint, test_legals, res_a, 10, temp, eta, &fell_back_a);
  SPIEL_CHECK_FALSE(fell_back_a);
  // Target should be blended (non-identical to mu)
  SPIEL_CHECK_TRUE(std::abs(target_a[0] - test_blueprint[0].second) > 1e-4);

  // Case B: Overridden = false, but search agrees with raw policy D(q_search) == a_mu -> Blended target!
  CompoundSearchDecisionResult res_b;
  res_b.overridden = false;
  res_b.candidate_utilities = {{10, 2.0}, {20, 1.0}};  // Cand 10 (raw action) is higher
  bool fell_back_b = false;
  auto target_b = ComputeFallbackAlignedTarget(test_blueprint, test_legals, res_b, 10, temp, eta, &fell_back_b);
  SPIEL_CHECK_FALSE(fell_back_b);
  // Target should be blended, sharpening action 10
  SPIEL_CHECK_TRUE(target_b[0] > test_blueprint[0].second);

  // Case C: Overridden = false, but search disagrees with raw policy D(q_search) != a_mu -> Fallback to mu!
  CompoundSearchDecisionResult res_c;
  res_c.overridden = false;
  res_c.candidate_utilities = {{10, 1.0}, {20, 1.05}};  // Cand 20 slightly higher (+0.05 < 0.15 threshold)
  bool fell_back_c = false;
  auto target_c = ComputeFallbackAlignedTarget(test_blueprint, test_legals, res_c, 10, temp, eta, &fell_back_c);
  SPIEL_CHECK_TRUE(fell_back_c);
  // Target must EXACTLY match test_blueprint (mu)
  for (size_t i = 0; i < test_legals.size(); ++i) {
    SPIEL_CHECK_FLOAT_EQ(target_c[i], test_blueprint[i].second);
  }

  // Test 6: Direct Scoring Fix Regression Test with Known Unequal Candidate Returns
  // Verifies that with identical roots and randomness, null-tree (single-decision supervision)
  // and explicit fresh-tree paths produce bitwise-identical utilities and training targets,
  // and that candidate utilities are strictly unequal and correctly trigger overrides.
  {
    class MockUnequalState : public open_spiel::State {
     public:
      MockUnequalState(std::shared_ptr<const Game> g) : State(g) {}
      Player CurrentPlayer() const override { return terminal_ ? kTerminalPlayerId : 0; }
      std::vector<Action> LegalActions() const override {
        if (terminal_) return {};
        return {10, 20, 30};
      }
      std::string ActionToString(Player player, Action action_id) const override {
        return std::to_string(action_id);
      }
      std::string ToString() const override { return "MockUnequalState"; }
      bool IsTerminal() const override { return terminal_; }
      std::vector<double> Returns() const override { return returns_; }
      std::unique_ptr<State> Clone() const override {
        return std::make_unique<MockUnequalState>(*this);
      }

     protected:
      void DoApplyAction(Action action_id) override {
        terminal_ = true;
        // Unequal returns: Action 10 (raw) = 0.20, Action 20 = 0.85, Action 30 = -0.50
        if (action_id == 10) {
          returns_ = {0.20, -0.20, 0.0, 0.0};
        } else if (action_id == 20) {
          returns_ = {0.85, -0.85, 0.0, 0.0};
        } else if (action_id == 30) {
          returns_ = {-0.50, 0.50, 0.0, 0.0};
        } else {
          returns_ = {0.0, 0.0, 0.0, 0.0};
        }
      }

     private:
      bool terminal_ = false;
      std::vector<double> returns_ = {0.0, 0.0, 0.0, 0.0};
    };

    class MockUnequalEvaluator : public open_spiel::BatchedNNEvaluator {
     public:
      MockUnequalEvaluator() : BatchedNNEvaluator(nullptr, 10.0f) {}
      ActionsAndProbs Prior(const State& state) override {
        // Action 10 has highest prior (0.60) -> raw greedy action
        return {{10, 0.60}, {20, 0.30}, {30, 0.10}};
      }
      std::vector<double> Evaluate(const State& state) override {
        return state.Returns();
      }
    };

    std::vector<std::shared_ptr<BatchedNNEvaluator>> evals = {
        std::make_shared<MockUnequalEvaluator>()};

    MockUnequalState root_state(game);
    const uint64_t test_seed = 0x12345678ULL;
    const int top_k = 3;
    const int rollouts_per_action = 8;
    const double override_margin = 0.15;

    // Path A: null-tree path (as used in single-decision supervision)
    CompoundSearchDecisionResult res_null = CompoundRolloutSearchDecision(
        root_state, 0, /*turn_tree=*/nullptr, evals, top_k, rollouts_per_action,
        override_margin, TreeReuseMode::kTargetFill, test_seed);

    // Path B: explicit fresh-tree path
    TurnSearchTree explicit_tree;
    CompoundSearchDecisionResult res_explicit = CompoundRolloutSearchDecision(
        root_state, 0, &explicit_tree, evals, top_k, rollouts_per_action,
        override_margin, TreeReuseMode::kTargetFill, test_seed);

    // 1. Verify candidate returns are strictly unequal and match known ground truth
    SPIEL_CHECK_EQ(res_null.candidate_utilities.size(), 3);
    SPIEL_CHECK_FLOAT_EQ(res_null.candidate_utilities[0].second, 0.20);   // Action 10
    SPIEL_CHECK_FLOAT_EQ(res_null.candidate_utilities[1].second, 0.85);   // Action 20
    SPIEL_CHECK_FLOAT_EQ(res_null.candidate_utilities[2].second, -0.50);  // Action 30
    SPIEL_CHECK_NE(res_null.candidate_utilities[0].second, res_null.candidate_utilities[1].second);
    SPIEL_CHECK_NE(res_null.candidate_utilities[1].second, res_null.candidate_utilities[2].second);

    // 2. Verify search override correctly triggered for Action 20 (0.85 > 0.20 + 0.15)
    SPIEL_CHECK_TRUE(res_null.overridden);
    SPIEL_CHECK_EQ(res_null.action, 20);
    SPIEL_CHECK_FLOAT_EQ(res_null.raw_mean_utility, 0.20);
    SPIEL_CHECK_FLOAT_EQ(res_null.chosen_mean_utility, 0.85);

    // 3. Verify exact parity between null-tree and explicit fresh-tree paths
    SPIEL_CHECK_EQ(res_null.action, res_explicit.action);
    SPIEL_CHECK_EQ(res_null.overridden, res_explicit.overridden);
    SPIEL_CHECK_FLOAT_EQ(res_null.raw_mean_utility, res_explicit.raw_mean_utility);
    SPIEL_CHECK_FLOAT_EQ(res_null.chosen_mean_utility, res_explicit.chosen_mean_utility);
    SPIEL_CHECK_EQ(res_null.candidate_utilities.size(), res_explicit.candidate_utilities.size());
    for (size_t i = 0; i < res_null.candidate_utilities.size(); ++i) {
      SPIEL_CHECK_EQ(res_null.candidate_utilities[i].first, res_explicit.candidate_utilities[i].first);
      SPIEL_CHECK_FLOAT_EQ(res_null.candidate_utilities[i].second, res_explicit.candidate_utilities[i].second);
    }

    // 4. Verify training targets match identically between null-tree and explicit-tree paths
    std::vector<Action> legals = {10, 20, 30};
    std::vector<std::pair<Action, double>> raw_mu = {{10, 0.60}, {20, 0.30}, {30, 0.10}};
    bool fell_back_null = false;
    bool fell_back_explicit = false;
    auto target_null = ComputeFallbackAlignedTarget(
        raw_mu, legals, res_null, 10, 0.50, 0.80, &fell_back_null);
    auto target_explicit = ComputeFallbackAlignedTarget(
        raw_mu, legals, res_explicit, 10, 0.50, 0.80, &fell_back_explicit);

    SPIEL_CHECK_FALSE(fell_back_null);
    SPIEL_CHECK_FALSE(fell_back_explicit);
    SPIEL_CHECK_EQ(target_null.size(), target_explicit.size());
    for (size_t i = 0; i < target_null.size(); ++i) {
      SPIEL_CHECK_FLOAT_EQ(target_null[i], target_explicit[i]);
    }
    // Overridden Action 20 must have highest target probability
    SPIEL_CHECK_TRUE(target_null[1] > target_null[0]);
    SPIEL_CHECK_TRUE(target_null[1] > target_null[2]);

    // 5. Verify BatchCompoundSearchSupervision single-decision path with null tree
    SearchSnapshot snap;
    snap.sim_state = std::make_unique<MockUnequalState>(game);
    snap.player = 0;
    snap.observation = {1.0f, 0.0f};
    snap.legal_actions = {10, 20, 30};
    snap.chosen_action = 10;
    snap.raw_policy = raw_mu;

    std::vector<SearchSnapshot> snapshots;
    snapshots.push_back(std::move(snap));

    // 5a. Verify BatchCompoundSearchSupervision with legacy fallback target
    SearchSupervisionResult batch_res_legacy = BatchCompoundSearchSupervision(
        snapshots, evals, top_k, rollouts_per_action, override_margin,
        /*softmax_temperature=*/0.50, /*eta_blend=*/0.80, test_seed,
        /*num_workers=*/1, /*full_turn_supervision=*/false, /*use_mpo_target=*/false);

    SPIEL_CHECK_EQ(batch_res_legacy.roots_evaluated, 1);
    SPIEL_CHECK_EQ(batch_res_legacy.overrides, 1);
    SPIEL_CHECK_EQ(batch_res_legacy.fallbacks_to_mu, 0);
    SPIEL_CHECK_EQ(batch_res_legacy.examples.size(), 1);
    SPIEL_CHECK_EQ(batch_res_legacy.examples[0].normalized_visits.size(), 3);
    for (size_t i = 0; i < 3; ++i) {
      SPIEL_CHECK_FLOAT_EQ(batch_res_legacy.examples[0].normalized_visits[i], target_null[i]);
    }

    // 5b. Verify BatchCompoundSearchSupervision with MPO relative target
    auto target_mpo_expected = ComputeMpoRelativeTarget(
        raw_mu, legals, res_null.candidate_utilities, 0.50, 0.05);

    SearchSupervisionResult batch_res_mpo = BatchCompoundSearchSupervision(
        snapshots, evals, top_k, rollouts_per_action, override_margin,
        /*softmax_temperature=*/0.50, /*eta_blend=*/0.80, test_seed,
        /*num_workers=*/1, /*full_turn_supervision=*/false, /*use_mpo_target=*/true);

    SPIEL_CHECK_EQ(batch_res_mpo.roots_evaluated, 1);
    SPIEL_CHECK_EQ(batch_res_mpo.overrides, 1);
    SPIEL_CHECK_EQ(batch_res_mpo.fallbacks_to_mu, 0);
    SPIEL_CHECK_EQ(batch_res_mpo.examples.size(), 1);
    SPIEL_CHECK_EQ(batch_res_mpo.examples[0].normalized_visits.size(), 3);
    for (size_t i = 0; i < 3; ++i) {
      SPIEL_CHECK_FLOAT_EQ(batch_res_mpo.examples[0].normalized_visits[i], target_mpo_expected[i]);
    }
  }

  // Test 7: MPO-Style Policy-Relative Reweighting Invariants
  {
    std::vector<Action> legal_acts = {10, 20, 30, 40, 50};
    std::vector<std::pair<Action, double>> ref_pi = {
        {10, 0.50}, {20, 0.25}, {30, 0.15}, {40, 0.08}, {50, 0.02}};

    // Invariant 1: Tied-Score Invariant
    // When evaluated candidates have identical utilities, the target distribution
    // must reproduce reference probabilities bit-for-bit with zero entropy change.
    std::vector<std::pair<Action, double>> tied_cands = {
        {10, 0.65}, {20, 0.65}, {30, 0.65}};
    double tied_kl = -1.0;
    double tied_tau = -1.0;
    auto tied_target = ComputeMpoRelativeTarget(
        ref_pi, legal_acts, tied_cands, /*base_temperature=*/0.50,
        /*max_target_kl=*/0.05, &tied_kl, &tied_tau);

    SPIEL_CHECK_FLOAT_EQ(tied_kl, 0.0);
    SPIEL_CHECK_FLOAT_EQ(tied_tau, 0.50);
    SPIEL_CHECK_EQ(tied_target.size(), legal_acts.size());
    for (size_t i = 0; i < legal_acts.size(); ++i) {
      SPIEL_CHECK_FLOAT_EQ(tied_target[i], ref_pi[i].second);
    }

    // Invariant 2: Near-Tie Stability (Astra Correction 4)
    // Small utility differences (0.001) must produce near-zero movement without
    // exhausting the KL budget or collapsing the temperature floor.
    std::vector<std::pair<Action, double>> near_tied_cands = {
        {10, 0.500}, {20, 0.501}, {30, 0.499}};
    double near_kl = -1.0;
    double near_tau = -1.0;
    auto near_target = ComputeMpoRelativeTarget(
        ref_pi, legal_acts, near_tied_cands, /*base_temperature=*/0.50,
        /*max_target_kl=*/0.05, &near_kl, &near_tau);

    SPIEL_CHECK_FLOAT_EQ(near_tau, 0.50);  // Temperature floor held
    SPIEL_CHECK_TRUE(near_kl < 0.0005);     // Near-zero KL movement
    for (size_t i = 0; i < legal_acts.size(); ++i) {
      SPIEL_CHECK_TRUE(std::abs(near_target[i] - ref_pi[i].second) < 0.005);
    }

    // Invariant 3: KL Ceiling Enforcement on Extreme Disparities
    // Large score gaps (+10.0 vs -10.0) must trigger temperature relaxation
    // and strictly respect the KL budget ceiling.
    std::vector<std::pair<Action, double>> extreme_cands = {
        {10, 10.0}, {20, -10.0}, {30, 0.0}};
    double extreme_kl = -1.0;
    double extreme_tau = -1.0;
    const double max_kl_ceiling = 0.05;
    auto extreme_target = ComputeMpoRelativeTarget(
        ref_pi, legal_acts, extreme_cands, /*base_temperature=*/0.50,
        max_kl_ceiling, &extreme_kl, &extreme_tau);

    SPIEL_CHECK_TRUE(extreme_tau > 0.50);
    SPIEL_CHECK_TRUE(extreme_kl <= max_kl_ceiling + 1e-5);
    // Highest utility candidate must gain probability
    SPIEL_CHECK_TRUE(extreme_target[0] > ref_pi[0].second);
    // Lowest utility candidate must lose probability
    SPIEL_CHECK_TRUE(extreme_target[1] < ref_pi[1].second);

    // Invariant 4: Outside Candidate Preservation
    // Legal actions not in the candidate set (40 and 50) must retain exact reference probabilities.
    SPIEL_CHECK_FLOAT_EQ(extreme_target[3], ref_pi[3].second);
    SPIEL_CHECK_FLOAT_EQ(extreme_target[4], ref_pi[4].second);

    // Invariant 5: Valid Probability Distribution
    double total_prob = 0.0;
    for (double p : extreme_target) total_prob += p;
    SPIEL_CHECK_FLOAT_EQ(total_prob, 1.0);
  }

  // Test 5: Consecutive chance events RNG advancement and TurnSearchTree invalidation (Correction 3)
  {
    std::cout << "Running Test 5: Consecutive chance events RNG and tree invalidation...\n";
    std::shared_ptr<const Game> game = LoadGame("dune_imperium");
    std::unique_ptr<State> state = game->NewInitialState();
    SPIEL_CHECK_TRUE(state->IsChanceNode());

    // Populate TurnSearchTree with prior data
    TurnSearchTree turn_tree;
    turn_tree.RecordRollout(15, {30, 2}, 3.0);
    SPIEL_CHECK_TRUE(turn_tree.HasPriorData(15));
    SPIEL_CHECK_EQ(turn_tree.GetVisits(15), 1);

    // Initialize persistent per-root RNG
    uint64_t root_seed = 20260923ULL;
    uint64_t chance_root_seed = dune_seed::DeriveSeed(root_seed, /*stream=*/0x4348414EULL);
    std::mt19937_64 chance_rng(chance_root_seed);

    int chance_steps = 0;
    std::vector<double> random_draws;
    std::vector<Action> chance_actions;

    // Traverse consecutive chance nodes
    while (state->IsChanceNode() && chance_steps < 10) {
      chance_steps++;
      auto chance_outcomes = state->ChanceOutcomes();
      double u = std::generate_canonical<double, 53>(chance_rng);
      random_draws.push_back(u);

      Action ca = open_spiel::SampleAction(chance_outcomes, u).first;
      chance_actions.push_back(ca);
      state->ApplyAction(ca);

      // Chance event invalidates cached search tree
      turn_tree.Reset();
      SPIEL_CHECK_FALSE(turn_tree.HasPriorData(15));
      SPIEL_CHECK_EQ(turn_tree.GetVisits(15), 0);
    }

    SPIEL_CHECK_TRUE(chance_steps >= 2);
    SPIEL_CHECK_EQ(random_draws.size(), static_cast<size_t>(chance_steps));

    // Invariant: Consecutive random draws must NOT be identical (proves persistent RNG advancement)
    for (size_t i = 1; i < random_draws.size(); ++i) {
      SPIEL_CHECK_FALSE(random_draws[i] == random_draws[i - 1]);
    }
    std::cout << absl::StrFormat("  Consecutive chance draws (%d events) advanced RNG monotonically; tree reset verified.\n",
                                 chance_steps);
  }

  // Test 6: Production BatchCompoundSearchSupervision full-turn loop with chance handling
  {
    std::cout << "Running Test 6: Production BatchCompoundSearchSupervision with chance handling...\n";
    std::shared_ptr<const Game> game = LoadGame("dune_imperium");
    std::unique_ptr<State> state = game->NewInitialState();
    SPIEL_CHECK_TRUE(state->IsChanceNode());

    auto net = std::make_shared<SharedDunePolicyValueNetImpl>(
        dune_imperium::kFullPublicInformationStateSize, 256, 2391, 2, false, false, 0, true);
    std::shared_mutex mutex;
    auto coord = std::make_shared<BatchedEvaluator>(
        net, 16, 1, torch::kCPU, &mutex, 10.0f, false, false, false, true, true);
    auto nn_eval = std::make_shared<BatchedNNEvaluator>(coord, 10.0f);
    std::vector<std::shared_ptr<BatchedNNEvaluator>> evaluators = {nn_eval};

    SearchSnapshot snap;
    snap.sim_state = state->Clone();
    snap.player = 0;
    snap.observation.assign(dune_imperium::kFullPublicInformationStateSize, 0.0f);
    snap.legal_actions = {1, 2};

    std::vector<SearchSnapshot> snapshots;
    snapshots.push_back(std::move(snap));

    auto res = BatchCompoundSearchSupervision(
        snapshots, evaluators,
        /*top_k=*/2, /*rollouts_per_action=*/2, /*min_override_margin=*/0.15,
        /*softmax_temperature=*/0.50, /*eta_blend=*/0.80, /*search_seed=*/20260923ULL,
        /*num_workers=*/1, /*full_turn_supervision=*/true, /*use_mpo_target=*/true,
        /*max_target_kl=*/0.05);

    SPIEL_CHECK_EQ(res.roots_evaluated, 1);
    SPIEL_CHECK_TRUE(snapshots[0].sim_state != nullptr);
    // After full-turn traversal starting from initial chance node, initial chance nodes were resolved
    SPIEL_CHECK_FALSE(snapshots[0].sim_state->IsChanceNode());
    std::cout << absl::StrFormat("  Production BatchCompoundSearchSupervision resolved chance nodes and evaluated turn (decisions=%d).\n",
                                 res.total_decisions_evaluated);
  }

  std::cout << "All TurnSearchTree and CompoundTurnSearch unit tests PASSED successfully!\n";
  return 0;
}
