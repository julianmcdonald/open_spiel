#include "open_spiel/examples/dune_wide_rollout_search.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <memory>
#include <thread>
#include <vector>

#include "open_spiel/games/dune_imperium/dune_imperium.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace dune_imperium {
namespace {

void CheckSameResult(const CompoundSearchDecisionResult& lhs,
                     const CompoundSearchDecisionResult& rhs) {
  SPIEL_CHECK_EQ(lhs.action, rhs.action);
  SPIEL_CHECK_EQ(lhs.overridden, rhs.overridden);
  SPIEL_CHECK_EQ(lhs.raw_mean_utility, rhs.raw_mean_utility);
  SPIEL_CHECK_EQ(lhs.chosen_mean_utility, rhs.chosen_mean_utility);
  SPIEL_CHECK_EQ(lhs.prior_rollouts_reused, rhs.prior_rollouts_reused);
  SPIEL_CHECK_EQ(lhs.new_rollouts_executed, rhs.new_rollouts_executed);
  SPIEL_CHECK_TRUE(lhs.candidate_utilities == rhs.candidate_utilities);
  SPIEL_CHECK_TRUE(lhs.candidate_rollout_utilities ==
                   rhs.candidate_rollout_utilities);
}

void CheckSameTree(const TurnSearchNode& lhs, const TurnSearchNode& rhs) {
  SPIEL_CHECK_EQ(lhs.visits, rhs.visits);
  SPIEL_CHECK_EQ(lhs.utility_sum, rhs.utility_sum);
  SPIEL_CHECK_EQ(lhs.children.size(), rhs.children.size());
  for (const auto& [action, child] : lhs.children) {
    auto it = rhs.children.find(action);
    SPIEL_CHECK_TRUE(it != rhs.children.end());
    CheckSameTree(*child, *it->second);
  }
}

std::vector<Action> Actions(int count) {
  std::vector<Action> actions;
  for (int i = 0; i < count; ++i) actions.push_back(10 + i);
  return actions;
}

class PayoffState : public State {
 public:
  PayoffState(std::shared_ptr<const Game> game, std::vector<Action> actions,
              Action boosted_action = kInvalidAction, double advantage = 1.0,
              double noise_size = 10.0)
      : State(game), actions_(std::move(actions)),
        boosted_action_(boosted_action), advantage_(advantage),
        noise_size_(noise_size) {}

  Player CurrentPlayer() const override {
    return phase_ == 3 ? kTerminalPlayerId : (phase_ == 1 ? kChancePlayerId : 0);
  }
  std::vector<Action> LegalActions() const override {
    if (phase_ == 0) return actions_;
    if (phase_ == 1) return {0, 1};
    if (phase_ == 2) return {99};
    return {};
  }
  ActionsAndProbs ChanceOutcomes() const override { return {{0, 0.5}, {1, 0.5}}; }
  std::string ActionToString(Player, Action action) const override {
    return std::to_string(action);
  }
  std::string ToString() const override { return "WidePayoffState"; }
  bool IsTerminal() const override { return phase_ == 3; }
  std::vector<double> Returns() const override {
    const double value = noise_ + (selected_ == boosted_action_ ? advantage_ : 0.0);
    return {value, -value, 0.0, 0.0};
  }
  std::unique_ptr<State> Clone() const override {
    return std::make_unique<PayoffState>(*this);
  }

 protected:
  void DoApplyAction(Action action) override {
    if (phase_ == 0) selected_ = action;
    if (phase_ == 1) noise_ = action == 0 ? -noise_size_ : noise_size_;
    ++phase_;
  }

 private:
  std::vector<Action> actions_;
  Action boosted_action_;
  double advantage_;
  double noise_size_;
  int phase_ = 0;
  Action selected_ = kInvalidAction;
  double noise_ = 0.0;
};

class FixedPriorEvaluator : public BatchedNNEvaluator {
 public:
  explicit FixedPriorEvaluator(bool tied = false)
      : BatchedNNEvaluator(nullptr, 10.0f), tied_(tied) {}
  ActionsAndProbs Prior(const State& state) override {
    ActionsAndProbs prior{{999, 1.0}};
    for (Action action : state.LegalActions()) {
      prior.emplace_back(action, tied_ ? 0.1 : 1.0 / action);
    }
    return prior;
  }

 private:
  bool tied_;
};

void SeedTree(TurnSearchTree* tree, const std::vector<Action>& actions) {
  if (!actions.empty()) tree->RecordRollout(actions[0], {99}, 10000.0);
  if (actions.size() > 1) tree->RecordRollout(actions[1], {99}, -1000.0);
}

void NativeLegacyDelegationTest(const std::shared_ptr<const Game>& game) {
  const std::vector<std::shared_ptr<BatchedNNEvaluator>> evaluators = {
      std::make_shared<FixedPriorEvaluator>()};
  for (int count : {0, 1, 2, 3}) {
    const auto actions = Actions(count);
    PayoffState state(game, actions);
    for (int rollouts : {1, 7}) {
      TurnSearchTree reference_tree;
      TurnSearchTree wide_tree;
      SeedTree(&reference_tree, actions);
      SeedTree(&wide_tree, actions);
      const auto expected = CompoundRolloutSearchDecision(
          state, 0, &reference_tree, evaluators, 3, rollouts, 0.15,
          TreeReuseMode::kAdditive, 314, SearchPairingMode::kUnpaired,
          SearchTruncationMode::kTerminal);
      const auto actual = WideRolloutSearchDecision(
          state, 0, &wide_tree, evaluators, rollouts, 0.15, 314);
      CheckSameResult(expected, actual);
      CheckSameTree(*reference_tree.current_node(), *wide_tree.current_node());
    }
  }
}

void NativeWideBudgetAndPairingTest(const std::shared_ptr<const Game>& game) {
  const std::vector<std::shared_ptr<BatchedNNEvaluator>> evaluators = {
      std::make_shared<FixedPriorEvaluator>()};
  for (int count : {4, 5, 6, 9}) {
    const auto actions = Actions(count);
    const int candidate_count = std::min(count, 6);
    const int expected_per_action = count == 4 ? 48 : count == 5 ? 38 : 32;
    const Action winner = actions[candidate_count - 1];
    PayoffState state(game, actions, winner);
    TurnSearchTree tree;
    const auto result = WideRolloutSearchDecision(
        state, 0, &tree, evaluators, 64, 0.15, 27);
    SPIEL_CHECK_EQ(result.action, winner);
    SPIEL_CHECK_TRUE(result.overridden);
    SPIEL_CHECK_EQ(result.new_rollouts_executed,
                   candidate_count * expected_per_action);
    SPIEL_CHECK_EQ(result.prior_rollouts_reused, 0);
    SPIEL_CHECK_EQ(result.candidate_utilities.size(), candidate_count);
    SPIEL_CHECK_FLOAT_EQ(result.chosen_mean_utility - result.raw_mean_utility, 1.0);
    const auto& raw_rollouts = result.candidate_rollout_utilities[0].second;
    int visits = 0;
    for (int i = 0; i < count; ++i) {
      const int expected_visits = i < candidate_count ? expected_per_action : 0;
      SPIEL_CHECK_EQ(tree.GetVisits(actions[i]), expected_visits);
      visits += tree.GetVisits(actions[i]);
      if (i >= candidate_count) continue;
      const auto& [action, rollouts] = result.candidate_rollout_utilities[i];
      SPIEL_CHECK_EQ(action, actions[i]);
      SPIEL_CHECK_EQ(rollouts.size(), expected_per_action);
      for (int r = 0; r < expected_per_action; ++r) {
        SPIEL_CHECK_EQ(rollouts[r] - raw_rollouts[r], action == winner ? 1.0 : 0.0);
      }
    }
    SPIEL_CHECK_EQ(visits, result.new_rollouts_executed);
    tree.Advance(result.action);
    SPIEL_CHECK_EQ(tree.GetVisits(99), expected_per_action);
  }
}

void NativeWideCompoundParityTest(const std::shared_ptr<const Game>& game) {
  const std::vector<std::shared_ptr<BatchedNNEvaluator>> evaluators = {
      std::make_shared<FixedPriorEvaluator>()};
  for (int count : {4, 5, 6, 9}) {
    const auto actions = Actions(count);
    PayoffState state(game, actions, actions[std::min(count, 6) - 1]);
    for (int reference_rollouts : {2, 7, 64}) {
      const int top_k = std::min(count, 6);
      const int per_action = 3 * reference_rollouts / top_k;
      TurnSearchTree reference_tree;
      TurnSearchTree wide_tree;
      SeedTree(&reference_tree, actions);
      SeedTree(&wide_tree, actions);
      const auto expected = CompoundRolloutSearchDecision(
          state, 0, &reference_tree, evaluators, top_k, per_action, 0.15,
          TreeReuseMode::kAdditive, 27, SearchPairingMode::kPaired,
          SearchTruncationMode::kTerminal);
      const auto actual = WideRolloutSearchDecision(
          state, 0, &wide_tree, evaluators, reference_rollouts, 0.15, 27);
      CheckSameResult(expected, actual);
      CheckSameTree(*reference_tree.current_node(), *wide_tree.current_node());
      SPIEL_CHECK_EQ(actual.prior_rollouts_reused, 2);
      SPIEL_CHECK_EQ(actual.new_rollouts_executed, top_k * per_action);
      SPIEL_CHECK_EQ(actual.action, actions.front());
    }
  }
}

void RawRetentionAndMarginTest(const std::shared_ptr<const Game>& game) {
  const std::vector<std::shared_ptr<BatchedNNEvaluator>> evaluators = {
      std::make_shared<FixedPriorEvaluator>(true)};
  const auto actions = Actions(9);
  PayoffState winner_state(game, actions, 18);
  const auto winner = WideRolloutSearchDecision(
      winner_state, 0, nullptr, evaluators, 64, 0.15, 17);
  SPIEL_CHECK_EQ(winner.action, 18);
  SPIEL_CHECK_TRUE(winner.overridden);
  const std::vector<Action> expected_candidates{10, 18, 17, 16, 15, 14};
  SPIEL_CHECK_EQ(winner.candidate_utilities.size(), expected_candidates.size());
  for (size_t i = 0; i < expected_candidates.size(); ++i) {
    SPIEL_CHECK_EQ(winner.candidate_utilities[i].first, expected_candidates[i]);
  }
  PayoffState tied_state(game, actions);
  const auto tied = WideRolloutSearchDecision(
      tied_state, 0, nullptr, evaluators, 64, 0.15, 17);
  SPIEL_CHECK_EQ(tied.action, 10);
  SPIEL_CHECK_FALSE(tied.overridden);
  PayoffState margin_state(game, actions, 18, 0.25, 0.0);
  const auto at_margin = WideRolloutSearchDecision(
      margin_state, 0, nullptr, evaluators, 64, 0.25, 17);
  SPIEL_CHECK_EQ(at_margin.action, 10);
  SPIEL_CHECK_FALSE(at_margin.overridden);

  const std::vector<std::shared_ptr<BatchedNNEvaluator>> ranked_evaluators = {
      std::make_shared<FixedPriorEvaluator>()};
  const auto excluded = WideRolloutSearchDecision(
      winner_state, 0, nullptr, ranked_evaluators, 64, 0.15, 17);
  SPIEL_CHECK_EQ(excluded.action, 10);
  SPIEL_CHECK_FALSE(excluded.overridden);
  for (const auto& [action, utility] : excluded.candidate_utilities) {
    SPIEL_CHECK_NE(action, 18);
  }
}

void NativeSuccessiveHalvingTest(const std::shared_ptr<const Game>& game) {
  const std::vector<std::shared_ptr<BatchedNNEvaluator>> evaluators = {
      std::make_shared<FixedPriorEvaluator>()};
  const auto actions = Actions(6);

  // Case 1: Raw action is already best -> Early stop after Stage 1.
  {
    PayoffState raw_best_state(game, actions, actions[0], /*advantage=*/1.0, /*noise_size=*/0.0);
    TurnSearchTree tree;
    const auto result = SuccessiveHalvingRolloutSearchDecision(
        raw_best_state, 0, &tree, evaluators, 64, 0.15, 42);
    SPIEL_CHECK_EQ(result.action, actions[0]);
    SPIEL_CHECK_FALSE(result.overridden);
    // K=6, total_budget=192, stage1_budget=96, r_stage1=16 -> 96 rollouts executed
    SPIEL_CHECK_EQ(result.new_rollouts_executed, 96);
  }

  // Case 2: Challenger is best and beats margin -> Two-stage confirmation.
  {
    const Action winner = actions[2];
    PayoffState winner_state(game, actions, winner, /*advantage=*/1.0, /*noise_size=*/0.0);
    TurnSearchTree tree;
    const auto result = SuccessiveHalvingRolloutSearchDecision(
        winner_state, 0, &tree, evaluators, 64, 0.15, 42);
    SPIEL_CHECK_EQ(result.action, winner);
    SPIEL_CHECK_TRUE(result.overridden);
    // Stage 1: 96 rollouts. Stage 2: finalists={0, 2}, stage2_budget=96, r_stage2=48. 2*48=96.
    // Total = 96 + 96 = 192 rollouts executed.
    SPIEL_CHECK_EQ(result.new_rollouts_executed, 192);
  }

  // Case 3: Challenger is best, unbiased_stage2_gate = true confirms override.
  {
    const Action winner = actions[2];
    PayoffState winner_state(game, actions, winner, /*advantage=*/1.0, /*noise_size=*/0.0);
    TurnSearchTree tree;
    const auto result = SuccessiveHalvingRolloutSearchDecision(
        winner_state, 0, &tree, evaluators, 64, 0.15, 42,
        SearchRolloutCrnMode::kSequential, /*unbiased_stage2_gate=*/true);
    SPIEL_CHECK_EQ(result.action, winner);
    SPIEL_CHECK_TRUE(result.overridden);
    SPIEL_CHECK_EQ(result.new_rollouts_executed, 192);
  }

  // Case 4: K = 2 candidate confirmation with unbiased_stage2_gate = true.
  {
    const auto actions2 = Actions(2);
    const Action winner2 = actions2[1];
    PayoffState winner2_state(game, actions2, winner2, /*advantage=*/1.0, /*noise_size=*/0.0);
    TurnSearchTree tree2;
    const auto result2 = SuccessiveHalvingRolloutSearchDecision(
        winner2_state, 0, &tree2, evaluators, 64, 0.15, 42,
        SearchRolloutCrnMode::kSequential, /*unbiased_stage2_gate=*/true);
    SPIEL_CHECK_EQ(result2.action, winner2);
    SPIEL_CHECK_TRUE(result2.overridden);
    SPIEL_CHECK_EQ(result2.new_rollouts_executed, 192);
  }
}

void ComputeMeanPairedCorrelationTest() {
  // Test 1: Empty or single-candidate returns default
  {
    const auto stats0 = ComputeMeanPairedCorrelation(std::vector<std::vector<double>>{});
    SPIEL_CHECK_EQ(stats0.pair_count, 0);
    SPIEL_CHECK_FLOAT_EQ(stats0.sum_corr, 0.0);
    SPIEL_CHECK_FLOAT_EQ(stats0.mean_paired_corr, 0.0);
    SPIEL_CHECK_FLOAT_EQ(stats0.sum_diff_var, 0.0);
    SPIEL_CHECK_FLOAT_EQ(stats0.mean_paired_diff_var, 0.0);

    const auto stats1 = ComputeMeanPairedCorrelation(std::vector<std::vector<double>>{{1.0, 2.0}});
    SPIEL_CHECK_EQ(stats1.pair_count, 0);
  }

  // Test 2: Perfect positive correlation (r = 1.0, diff_var = 0.0)
  {
    const std::vector<double> c0 = {1.0, 2.0, 3.0, 4.0};
    const std::vector<double> c1 = {11.0, 12.0, 13.0, 14.0};
    const auto stats = ComputeMeanPairedCorrelation(std::vector<std::vector<double>>{c0, c1});
    SPIEL_CHECK_EQ(stats.pair_count, 1);
    SPIEL_CHECK_FLOAT_EQ(stats.sum_corr, 1.0);
    SPIEL_CHECK_FLOAT_EQ(stats.mean_paired_corr, 1.0);
    SPIEL_CHECK_FLOAT_EQ(stats.sum_diff_var, 0.0);
    SPIEL_CHECK_FLOAT_EQ(stats.mean_paired_diff_var, 0.0);
  }

  // Test 3: Anti-correlation (r = -1.0)
  {
    const std::vector<double> c0 = {1.0, 2.0, 3.0, 4.0};
    const std::vector<double> c1 = {4.0, 3.0, 2.0, 1.0};
    const auto stats = ComputeMeanPairedCorrelation(std::vector<std::vector<double>>{c0, c1});
    SPIEL_CHECK_EQ(stats.pair_count, 1);
    SPIEL_CHECK_FLOAT_EQ(stats.sum_corr, -1.0);
    SPIEL_CHECK_FLOAT_EQ(stats.mean_paired_corr, -1.0);
    SPIEL_CHECK_FLOAT_EQ(stats.mean_paired_diff_var, 20.0 / 3.0);
  }

  // Test 4: Multi-candidate pairing and overload with vector<pair<Action, vector<double>>>
  {
    const std::vector<double> c0 = {1.0, 2.0, 3.0, 4.0};
    const std::vector<double> c1 = {11.0, 12.0, 13.0, 14.0};
    const std::vector<double> c2 = {4.0, 3.0, 2.0, 1.0};
    const std::vector<std::pair<Action, std::vector<double>>> candidate_rollouts = {
        {10, c0}, {20, c1}, {30, c2}};
    const auto stats = ComputeMeanPairedCorrelation(candidate_rollouts);
    SPIEL_CHECK_EQ(stats.pair_count, 2);
    SPIEL_CHECK_FLOAT_EQ(stats.sum_corr, 0.0);
    SPIEL_CHECK_FLOAT_EQ(stats.mean_paired_corr, 0.0);
    SPIEL_CHECK_FLOAT_EQ(stats.sum_diff_var, 20.0 / 3.0);
    SPIEL_CHECK_FLOAT_EQ(stats.mean_paired_diff_var, 10.0 / 3.0);
  }

  // Test 5: Zero-variance candidate excluded from corr_pair_count and mean_paired_corr,
  // but included in diff_var_pair_count and mean_paired_diff_var.
  {
    const std::vector<double> c0 = {1.0, 2.0, 3.0, 4.0};
    const std::vector<double> c1 = {11.0, 12.0, 13.0, 14.0};
    const std::vector<double> c2 = {5.0, 5.0, 5.0, 5.0};
    const std::vector<std::vector<double>> candidate_rollouts = {c0, c1, c2};
    const auto stats = ComputeMeanPairedCorrelation(candidate_rollouts);
    SPIEL_CHECK_EQ(stats.pair_count, 1);
    SPIEL_CHECK_EQ(stats.corr_pair_count, 1);
    SPIEL_CHECK_EQ(stats.diff_var_pair_count, 2);
    SPIEL_CHECK_FLOAT_EQ(stats.sum_corr, 1.0);
    SPIEL_CHECK_FLOAT_EQ(stats.mean_paired_corr, 1.0);
    SPIEL_CHECK_FLOAT_EQ(stats.sum_diff_var, 5.0 / 3.0);
    SPIEL_CHECK_FLOAT_EQ(stats.mean_paired_diff_var, (5.0 / 3.0) / 2.0);
  }
}

void CurrentChanceStreamIdTest(const std::shared_ptr<const Game>& game) {
  auto state = game->NewInitialState();
  auto* dune_state = dynamic_cast<DuneImperiumState*>(state.get());
  SPIEL_CHECK_TRUE(dune_state != nullptr);

  // 1. Initial setup refills: 5 Imperium row slots (stream 1)
  for (int slot = 0; slot < 5; ++slot) {
    SPIEL_CHECK_EQ(dune_state->CurrentChanceStreamId(), 1);
    SPIEL_CHECK_GT(dune_state->CurrentChanceDeckRemainingSize(), 0);
    const auto outcomes = dune_state->ChanceOutcomes();
    SPIEL_CHECK_FALSE(outcomes.empty());
    dune_state->ApplyAction(outcomes.front().first);
  }

  // 2. Initial setup refills: 2 Tleilaxu row slots (stream 2)
  for (int slot = 0; slot < 2; ++slot) {
    SPIEL_CHECK_EQ(dune_state->CurrentChanceStreamId(), 2);
    SPIEL_CHECK_GT(dune_state->CurrentChanceDeckRemainingSize(), 0);
    const auto outcomes = dune_state->ChanceOutcomes();
    SPIEL_CHECK_FALSE(outcomes.empty());
    dune_state->ApplyAction(outcomes.front().first);
  }

  // 3. Initial setup refills: 3 Tech stack slots (streams 3, 4, 5)
  for (int stack = 0; stack < 3; ++stack) {
    SPIEL_CHECK_EQ(dune_state->CurrentChanceStreamId(), 3 + stack);
    SPIEL_CHECK_GT(dune_state->CurrentChanceDeckRemainingSize(), 0);
    const auto outcomes = dune_state->ChanceOutcomes();
    SPIEL_CHECK_FALSE(outcomes.empty());
    dune_state->ApplyAction(outcomes.front().first);
  }

  // 4. Now initial setup is complete, phase is kLeaderOfferChance -> stream 6
  SPIEL_CHECK_TRUE(dune_state->phase() == GamePhase::kLeaderOfferChance);
  SPIEL_CHECK_EQ(dune_state->CurrentChanceStreamId(), 6);

  // 5. In kRoundStart, if conflict not yet revealed -> stream 0
  dune_state->SetPhaseForTesting(GamePhase::kRoundStart);
  dune_state->SetRoundStartConflictDoneForTesting(false);
  SPIEL_CHECK_EQ(dune_state->CurrentChanceStreamId(), 0);
  SPIEL_CHECK_GT(dune_state->CurrentChanceDeckRemainingSize(), 0);

  // 6. If conflict is revealed and imperium slot refill is pending -> stream 1
  dune_state->SetRoundStartConflictDoneForTesting(true);
  dune_state->SetPendingImperiumSlotForTesting(1);
  SPIEL_CHECK_EQ(dune_state->CurrentChanceStreamId(), 1);
  SPIEL_CHECK_GT(dune_state->CurrentChanceDeckRemainingSize(), 0);

  // 7. Clear imperium slot, set pending player draw player for seats 0..3 -> streams 10..13
  dune_state->SetPendingImperiumSlotForTesting(kInvalidCard);
  for (int p = 0; p < kNumPlayers; ++p) {
    dune_state->SetPendingPlayerDrawPlayerForTesting(p);
    SPIEL_CHECK_EQ(dune_state->CurrentChanceStreamId(), 10 + p);
  }
}

class CrnTestDuneState : public DuneImperiumState {
 public:
  explicit CrnTestDuneState(const std::shared_ptr<const Game>& game)
      : DuneImperiumState(game) {
    SetPendingImperiumSlotForTesting(kInvalidCard);
  }

  std::unique_ptr<State> Clone() const override {
    return std::make_unique<CrnTestDuneState>(*this);
  }

  void SetBranchB(bool is_branch_b) {
    is_branch_b_ = is_branch_b;
  }

  bool IsTerminal() const override {
    return done_ || DuneImperiumState::IsTerminal();
  }

  Player CurrentPlayer() const override {
    if (done_) return kTerminalPlayerId;
    return DuneImperiumState::CurrentPlayer();
  }

  ActionsAndProbs ChanceOutcomes() const override {
    const int stream = CurrentChanceStreamId();
    if (stream == 0) {
      return {{1, 0.5}, {2, 0.5}};
    }
    if (stream >= 10 && stream <= 13) {
      return {{10, 0.25}, {11, 0.25}, {12, 0.25}, {13, 0.25}};
    }
    return DuneImperiumState::ChanceOutcomes();
  }

 protected:
  void DoApplyAction(Action action) override {
    history_actions_.push_back(action);
    step_++;
    if (!is_branch_b_) {
      if (step_ == 1) {
        SetRoundStartConflictDoneForTesting(true);
        SetPendingPlayerDrawPlayerForTesting(1);
      } else if (step_ >= 2) {
        done_ = true;
      }
    } else {
      if (step_ == 1) {
        SetRoundStartConflictDoneForTesting(false);
        SetPendingPlayerDrawPlayerForTesting(-1);
      } else if (step_ == 2) {
        SetRoundStartConflictDoneForTesting(true);
        SetPendingPlayerDrawPlayerForTesting(1);
      } else if (step_ >= 3) {
        done_ = true;
      }
    }
  }

 public:
  const std::vector<Action>& history_actions() const { return history_actions_; }

 private:
  bool is_branch_b_ = false;
  int step_ = 0;
  bool done_ = false;
  std::vector<Action> history_actions_;
};

void DeckOrderCrnAlignmentAndParityTest(const std::shared_ptr<const Game>& game) {
  int crn_matches = 0;
  int seq_mismatches = 0;
  const int num_seeds = 50;

  for (uint64_t seed = 1; seed <= num_seeds; ++seed) {
    // 1. Test kDeckOrderCrn:
    CrnTestDuneState state_a_crn(game);
    state_a_crn.SetPhaseForTesting(GamePhase::kRoundStart);
    state_a_crn.SetRoundStartConflictDoneForTesting(false);
    state_a_crn.SetPendingPlayerDrawPlayerForTesting(-1);
    state_a_crn.SetRoundNumberForTesting(1);
    state_a_crn.SetBranchB(false);

    CrnTestDuneState state_b_crn(game);
    state_b_crn.SetPhaseForTesting(GamePhase::kRoundStart);
    state_b_crn.SetRoundStartConflictDoneForTesting(true);
    state_b_crn.SetPendingPlayerDrawPlayerForTesting(0);
    state_b_crn.SetRoundNumberForTesting(1);
    state_b_crn.SetBranchB(true);

    SimulateRolloutWithTurnTrajectory(
        &state_a_crn, 0, nullptr, seed,
        SearchTruncationMode::kTerminal, SearchRolloutCrnMode::kDeckOrderCrn);
    SimulateRolloutWithTurnTrajectory(
        &state_b_crn, 0, nullptr, seed,
        SearchTruncationMode::kTerminal, SearchRolloutCrnMode::kDeckOrderCrn);

    SPIEL_CHECK_EQ(state_a_crn.history_actions().size(), 2);
    SPIEL_CHECK_EQ(state_b_crn.history_actions().size(), 3);
    SPIEL_CHECK_EQ(state_a_crn.history_actions()[0], state_b_crn.history_actions()[1]);
    SPIEL_CHECK_EQ(state_a_crn.history_actions()[1], state_b_crn.history_actions()[2]);
    crn_matches++;

    // 2. Test kSequential:
    CrnTestDuneState state_a_seq(game);
    state_a_seq.SetPhaseForTesting(GamePhase::kRoundStart);
    state_a_seq.SetRoundStartConflictDoneForTesting(false);
    state_a_seq.SetPendingPlayerDrawPlayerForTesting(-1);
    state_a_seq.SetRoundNumberForTesting(1);
    state_a_seq.SetBranchB(false);

    CrnTestDuneState state_b_seq(game);
    state_b_seq.SetPhaseForTesting(GamePhase::kRoundStart);
    state_b_seq.SetRoundStartConflictDoneForTesting(true);
    state_b_seq.SetPendingPlayerDrawPlayerForTesting(0);
    state_b_seq.SetRoundNumberForTesting(1);
    state_b_seq.SetBranchB(true);

    SimulateRolloutWithTurnTrajectory(
        &state_a_seq, 0, nullptr, seed,
        SearchTruncationMode::kTerminal, SearchRolloutCrnMode::kSequential);
    SimulateRolloutWithTurnTrajectory(
        &state_b_seq, 0, nullptr, seed,
        SearchTruncationMode::kTerminal, SearchRolloutCrnMode::kSequential);

    if (state_a_seq.history_actions()[0] != state_b_seq.history_actions()[1] ||
        state_a_seq.history_actions()[1] != state_b_seq.history_actions()[2]) {
      seq_mismatches++;
    }
  }

  SPIEL_CHECK_EQ(crn_matches, num_seeds);
  SPIEL_CHECK_GT(seq_mismatches, 0);
}

class NativeReshuffleCrnTestState : public DuneImperiumState {
 public:
  NativeReshuffleCrnTestState(const std::shared_ptr<const Game>& game, bool is_branch_b)
      : DuneImperiumState(game), is_branch_b_(is_branch_b) {
    SetPhaseForTesting(GamePhase::kRoundStart);
    SetRoundStartConflictDoneForTesting(true);
    SetPendingImperiumSlotForTesting(kInvalidCard);
    SetPendingPlayerDrawPlayerForTesting(0);
    // Real native discard pile with distinct cards for player 0:
    SetPlayerDiscardForTesting(0, {10, 11, 12, 13});
    if (!is_branch_b_) {
      // Branch A: 1 card in draw deck, hand size 2
      SetPlayerDrawDeckForTesting(0, {9});
      SetPlayerHandForTesting(0, {1, 2});
    } else {
      // Branch B: 0 cards in draw deck (already drew card 9), hand size 3
      SetPlayerDrawDeckForTesting(0, {});
      SetPlayerHandForTesting(0, {1, 2, 9});
    }
    initial_history_size_ = History().size();
  }

  std::unique_ptr<State> Clone() const override {
    return std::make_unique<NativeReshuffleCrnTestState>(*this);
  }

  bool is_done() const {
    return static_cast<int>(History().size() - initial_history_size_) >=
           (is_branch_b_ ? 2 : 3);
  }

  bool IsTerminal() const override {
    return is_done() || DuneImperiumState::IsTerminal();
  }

  Player CurrentPlayer() const override {
    if (is_done()) return kTerminalPlayerId;
    return DuneImperiumState::CurrentPlayer();
  }

  std::vector<Action> history_actions() const {
    const std::vector<Action> hist = History();
    return std::vector<Action>(hist.begin() + initial_history_size_, hist.end());
  }

 private:
  bool is_branch_b_ = false;
  size_t initial_history_size_ = 0;
};

void ReshuffleEpochCrnTest(const std::shared_ptr<const Game>& game) {
  int crn_matches = 0;
  int seq_mismatches = 0;
  const int num_seeds = 50;

  for (uint64_t seed = 1; seed <= num_seeds; ++seed) {
    // 1. Test kDeckOrderCrn:
    NativeReshuffleCrnTestState state_a_crn(game, /*is_branch_b=*/false);
    NativeReshuffleCrnTestState state_b_crn(game, /*is_branch_b=*/true);

    SimulateRolloutWithTurnTrajectory(
        &state_a_crn, 0, nullptr, seed,
        SearchTruncationMode::kTerminal, SearchRolloutCrnMode::kDeckOrderCrn);
    SimulateRolloutWithTurnTrajectory(
        &state_b_crn, 0, nullptr, seed,
        SearchTruncationMode::kTerminal, SearchRolloutCrnMode::kDeckOrderCrn);

    SPIEL_CHECK_EQ(state_a_crn.history_actions().size(), 3);
    SPIEL_CHECK_EQ(state_b_crn.history_actions().size(), 2);
    // Draw 0 in Branch A was card 9 from draw deck before reshuffle:
    SPIEL_CHECK_EQ(state_a_crn.history_actions()[0], 9);
    // Under CRN, draw 1 of Branch A (first reshuffled card) must match draw 0 of Branch B!
    SPIEL_CHECK_EQ(state_a_crn.history_actions()[1], state_b_crn.history_actions()[0]);
    // Draw 2 of Branch A (second reshuffled card) must match draw 1 of Branch B!
    SPIEL_CHECK_EQ(state_a_crn.history_actions()[2], state_b_crn.history_actions()[1]);
    crn_matches++;

    // 2. Test kSequential:
    NativeReshuffleCrnTestState state_a_seq(game, /*is_branch_b=*/false);
    NativeReshuffleCrnTestState state_b_seq(game, /*is_branch_b=*/true);

    SimulateRolloutWithTurnTrajectory(
        &state_a_seq, 0, nullptr, seed,
        SearchTruncationMode::kTerminal, SearchRolloutCrnMode::kSequential);
    SimulateRolloutWithTurnTrajectory(
        &state_b_seq, 0, nullptr, seed,
        SearchTruncationMode::kTerminal, SearchRolloutCrnMode::kSequential);

    if (state_a_seq.history_actions()[1] != state_b_seq.history_actions()[0] ||
        state_a_seq.history_actions()[2] != state_b_seq.history_actions()[1]) {
      seq_mismatches++;
    }
  }

  SPIEL_CHECK_EQ(crn_matches, num_seeds);
  SPIEL_CHECK_GT(seq_mismatches, 0);
}

void SmallBudgetGuardTest(const std::shared_ptr<const Game>& game) {
  const std::vector<std::shared_ptr<BatchedNNEvaluator>> evaluators = {
      std::make_shared<FixedPriorEvaluator>()};
  const auto actions = Actions(2);

  // Challenger (actions[1]) has huge advantage (+10.0), but r_completed <= 1
  // or r_stage2 <= 1 must guard t_stat and prevent override.
  PayoffState state(game, actions, actions[1], /*advantage=*/10.0, /*noise_size=*/0.0);

  // 1. PairedPriorSortedRolloutSearchDecision with rollouts_per_action = 1 -> r_completed = 1
  {
    TurnSearchTree tree;
    const auto result = PairedPriorSortedRolloutSearchDecision(
        state, 0, &tree, evaluators, actions, /*rollouts_per_action=*/1,
        /*min_override_margin=*/0.15, /*min_t_stat=*/1.645, 42);
    SPIEL_CHECK_EQ(result.action, actions[0]);
    SPIEL_CHECK_FALSE(result.overridden);
  }

  // 2. SuccessiveHalvingRolloutSearchDecision with reference_rollouts_per_action = 1 -> r_stage2 <= 1
  {
    TurnSearchTree tree;
    const auto result = SuccessiveHalvingRolloutSearchDecision(
        state, 0, &tree, evaluators, /*reference_rollouts_per_action=*/1,
        /*min_override_margin=*/0.15, 42);
    SPIEL_CHECK_EQ(result.action, actions[0]);
    SPIEL_CHECK_FALSE(result.overridden);
  }

  // 3. PairedPriorSortedRolloutSearchDecision with rollouts_per_action = 2 -> r_completed = 2 confirms override
  {
    TurnSearchTree tree;
    const auto result = PairedPriorSortedRolloutSearchDecision(
        state, 0, &tree, evaluators, actions, /*rollouts_per_action=*/2,
        /*min_override_margin=*/0.15, /*min_t_stat=*/1.645, 42);
    SPIEL_CHECK_EQ(result.action, actions[1]);
    SPIEL_CHECK_TRUE(result.overridden);
  }
}

class SlowDeadlineState : public State {
 public:
  SlowDeadlineState(std::shared_ptr<const Game> game, std::vector<Action> actions)
      : State(game), actions_(std::move(actions)) {}

  Player CurrentPlayer() const override {
    return step_ >= 4 ? kTerminalPlayerId : (step_ % 2 == 1 ? kChancePlayerId : 0);
  }
  std::vector<Action> LegalActions() const override {
    if (step_ >= 4) return {};
    return actions_;
  }
  ActionsAndProbs ChanceOutcomes() const override {
    return {{0, 0.5}, {1, 0.5}};
  }
  std::string ActionToString(Player, Action action) const override {
    return std::to_string(action);
  }
  std::string ToString() const override { return "SlowDeadlineState"; }
  bool IsTerminal() const override { return step_ >= 4; }
  std::vector<double> Returns() const override { return {1.0, -1.0, 0.0, 0.0}; }
  std::unique_ptr<State> Clone() const override {
    return std::make_unique<SlowDeadlineState>(*this);
  }

 protected:
  void DoApplyAction(Action /*action*/) override {
    ++step_;
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
  }

 private:
  std::vector<Action> actions_;
  int step_ = 0;
};

void DeadlineEnforcementTest(const std::shared_ptr<const Game>& game) {
  const std::vector<std::shared_ptr<BatchedNNEvaluator>> evaluators = {
      std::make_shared<FixedPriorEvaluator>()};
  const auto actions = Actions(2);
  SlowDeadlineState slow_state(game, actions);

  // 1. PairedPriorSortedRolloutSearchDecision with tiny deadline (0.001 ms).
  // Must return quickly without hanging and retain only paired completed worlds.
  {
    TurnSearchTree tree;
    const auto start_time = std::chrono::steady_clock::now();
    const auto result = PairedPriorSortedRolloutSearchDecision(
        slow_state, 0, &tree, evaluators, actions, /*rollouts_per_action=*/16,
        /*min_override_margin=*/0.15, /*min_t_stat=*/1.645, 42,
        SearchRolloutCrnMode::kSequential, /*sample_opponent_leader_draft=*/false,
        /*max_time_ms=*/0.001);
    const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - start_time).count();

    SPIEL_CHECK_LT(elapsed, 500);
    SPIEL_CHECK_EQ(result.action, actions[0]);
    SPIEL_CHECK_FALSE(result.overridden);
    SPIEL_CHECK_EQ(result.new_rollouts_executed, 0);
    SPIEL_CHECK_EQ(result.candidate_rollout_utilities.size(), 2);
    SPIEL_CHECK_EQ(result.candidate_rollout_utilities[0].second.size(),
                   result.candidate_rollout_utilities[1].second.size());
  }

  // 2. SuccessiveHalvingRolloutSearchDecision with tiny deadline (0.001 ms).
  {
    TurnSearchTree tree;
    const auto start_time = std::chrono::steady_clock::now();
    const auto result = SuccessiveHalvingRolloutSearchDecision(
        slow_state, 0, &tree, evaluators, /*reference_rollouts_per_action=*/16,
        /*min_override_margin=*/0.15, 42,
        SearchRolloutCrnMode::kSequential, /*unbiased_stage2_gate=*/false,
        /*max_time_ms=*/0.001);
    const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - start_time).count();

    SPIEL_CHECK_LT(elapsed, 500);
    SPIEL_CHECK_EQ(result.action, actions[0]);
    SPIEL_CHECK_FALSE(result.overridden);
    SPIEL_CHECK_EQ(result.new_rollouts_executed, 0);
    SPIEL_CHECK_EQ(result.candidate_rollout_utilities.size(), 2);
    SPIEL_CHECK_EQ(result.candidate_rollout_utilities[0].second.size(),
                   result.candidate_rollout_utilities[1].second.size());
  }

  // 3. Fast state with generous deadline (5000.0 ms) completes normally and paired.
  {
    PayoffState fast_state(game, actions, actions[1], /*advantage=*/1.0, /*noise_size=*/0.0);
    TurnSearchTree tree;
    const auto result = PairedPriorSortedRolloutSearchDecision(
        fast_state, 0, &tree, evaluators, actions, /*rollouts_per_action=*/8,
        /*min_override_margin=*/0.15, /*min_t_stat=*/1.645, 42,
        SearchRolloutCrnMode::kSequential, /*sample_opponent_leader_draft=*/false,
        /*max_time_ms=*/5000.0);
    SPIEL_CHECK_EQ(result.action, actions[1]);
    SPIEL_CHECK_TRUE(result.overridden);
    SPIEL_CHECK_EQ(result.new_rollouts_executed, 16);
    SPIEL_CHECK_EQ(result.candidate_rollout_utilities.size(), 2);
    SPIEL_CHECK_EQ(result.candidate_rollout_utilities[0].second.size(), 8);
    SPIEL_CHECK_EQ(result.candidate_rollout_utilities[1].second.size(), 8);
  }
}

}  // namespace
}  // namespace dune_imperium
}  // namespace open_spiel

int main() {
  using namespace open_spiel::dune_imperium;
  const auto game = open_spiel::LoadGame("dune_imperium");
  NativeLegacyDelegationTest(game);
  NativeWideBudgetAndPairingTest(game);
  NativeWideCompoundParityTest(game);
  RawRetentionAndMarginTest(game);
  NativeSuccessiveHalvingTest(game);
  ComputeMeanPairedCorrelationTest();
  CurrentChanceStreamIdTest(game);
  DeckOrderCrnAlignmentAndParityTest(game);
  ReshuffleEpochCrnTest(game);
  SmallBudgetGuardTest(game);
  DeadlineEnforcementTest(game);
}
