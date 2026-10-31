#include "open_spiel/examples/dune_wide_rollout_search.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <memory>
#include <thread>
#include <vector>

#include <ATen/Parallel.h>

#include "open_spiel/examples/dune_network.h"
#include "open_spiel/examples/dune_search_session.h"
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

class InsertedCardEventKeyedCrnTestState : public DuneImperiumState {
 public:
  InsertedCardEventKeyedCrnTestState(const std::shared_ptr<const Game>& game,
                                     bool is_branch_a)
      : DuneImperiumState(game), is_branch_a_(is_branch_a) {
    SetPhaseForTesting(GamePhase::kRoundStart);
    SetRoundStartConflictDoneForTesting(true);
    SetPendingImperiumSlotForTesting(kInvalidCard);
    SetPendingPlayerDrawPlayerForTesting(0);
    SetPlayerDrawDeckForTesting(0, {});
    if (is_branch_a_) {
      // Branch A: extra card 45 inserted into discard pile
      SetPlayerDiscardForTesting(0, {10, 11, 12, 13, 45});
      SetPlayerHandForTesting(0, {});  // draws 5 cards (target hand size 5)
    } else {
      // Branch B: 4 cards in discard pile
      SetPlayerDiscardForTesting(0, {10, 11, 12, 13});
      SetPlayerHandForTesting(0, {1});  // draws 4 cards (target hand size 5)
    }
    initial_history_size_ = History().size();
  }

  std::unique_ptr<State> Clone() const override {
    return std::make_unique<InsertedCardEventKeyedCrnTestState>(*this);
  }

  bool is_done() const {
    return static_cast<int>(History().size() - initial_history_size_) >=
           (is_branch_a_ ? 5 : 4);
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
  bool is_branch_a_ = true;
  size_t initial_history_size_ = 0;
};

class MidRoundReshuffleCrnTestState : public DuneImperiumState {
 public:
  MidRoundReshuffleCrnTestState(const std::shared_ptr<const Game>& game,
                                bool is_branch_b)
      : DuneImperiumState(game), is_branch_b_(is_branch_b) {
    SetPendingImperiumSlotForTesting(kInvalidCard);
    SetPlayerDiscardForTesting(0, {10, 11, 12, 13});
    SetPlayerDrawDeckForTesting(0, {});
    SetPlayerHandForTesting(0, {1, 2, 3});

    if (!is_branch_b_) {
      // Branch A: triggers mid-round reshuffle via DrawPlayerCards before rollout starts
      SetPhaseForTesting(GamePhase::kAgentTurns);
      DrawPlayerCardsForTesting(0, 2);
      SPIEL_CHECK_EQ(deck_reshuffle_count(0), 1);
    } else {
      // Branch B: reshuffles at kRoundStart during rollout
      SetPhaseForTesting(GamePhase::kRoundStart);
      SetRoundStartConflictDoneForTesting(true);
      SetPendingPlayerDrawPlayerForTesting(0);
      SPIEL_CHECK_EQ(deck_reshuffle_count(0), 0);
    }
    initial_history_size_ = History().size();
  }

  std::unique_ptr<State> Clone() const override {
    return std::make_unique<MidRoundReshuffleCrnTestState>(*this);
  }

  bool is_done() const {
    return static_cast<int>(History().size() - initial_history_size_) >= 2;
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

void EventKeyedCrnExtraDrawAndInsertedCardTest(
    const std::shared_ptr<const Game>& game) {
  const int num_seeds = 50;

  // 1. Inserted card test:
  // Branch A has card 45 inserted into discard {10, 11, 12, 13, 45}.
  // Branch B has discard {10, 11, 12, 13}.
  // Relative order of the 4 shared cards {10, 11, 12, 13} must be 100% identical under kEventKeyedCrn.
  int event_keyed_matches_part1 = 0;
  int deck_order_mismatches_part1 = 0;
  for (uint64_t seed = 1; seed <= num_seeds; ++seed) {
    InsertedCardEventKeyedCrnTestState state_a_evk(game, /*is_branch_a=*/true);
    InsertedCardEventKeyedCrnTestState state_b_evk(game, /*is_branch_a=*/false);

    SimulateRolloutWithTurnTrajectory(
        &state_a_evk, 0, nullptr, seed,
        SearchTruncationMode::kTerminal, SearchRolloutCrnMode::kEventKeyedCrn);
    SimulateRolloutWithTurnTrajectory(
        &state_b_evk, 0, nullptr, seed,
        SearchTruncationMode::kTerminal, SearchRolloutCrnMode::kEventKeyedCrn);

    SPIEL_CHECK_EQ(state_a_evk.history_actions().size(), 5);
    SPIEL_CHECK_EQ(state_b_evk.history_actions().size(), 4);

    std::vector<Action> shared_a_evk;
    for (Action a : state_a_evk.history_actions()) {
      if (a != 45) shared_a_evk.push_back(a);
    }
    SPIEL_CHECK_EQ(shared_a_evk.size(), 4);
    if (shared_a_evk == state_b_evk.history_actions()) {
      event_keyed_matches_part1++;
    }

    // Compare against kDeckOrderCrn (which suffers desync from the extra card draw)
    InsertedCardEventKeyedCrnTestState state_a_doc(game, /*is_branch_a=*/true);
    InsertedCardEventKeyedCrnTestState state_b_doc(game, /*is_branch_a=*/false);

    SimulateRolloutWithTurnTrajectory(
        &state_a_doc, 0, nullptr, seed,
        SearchTruncationMode::kTerminal, SearchRolloutCrnMode::kDeckOrderCrn);
    SimulateRolloutWithTurnTrajectory(
        &state_b_doc, 0, nullptr, seed,
        SearchTruncationMode::kTerminal, SearchRolloutCrnMode::kDeckOrderCrn);

    std::vector<Action> shared_a_doc;
    for (Action a : state_a_doc.history_actions()) {
      if (a != 45) shared_a_doc.push_back(a);
    }
    if (shared_a_doc != state_b_doc.history_actions()) {
      deck_order_mismatches_part1++;
    }
  }
  SPIEL_CHECK_EQ(event_keyed_matches_part1, num_seeds);
  SPIEL_CHECK_GT(deck_order_mismatches_part1, 0);

  // 2. Mid-round reshuffle synchronization test:
  // Branch A reshuffles mid-round via DrawPlayerCards before rollout starts (deck_reshuffle_count(0) == 1).
  // Branch B reshuffles at kRoundStart during rollout.
  // Both synchronize to effective_epoch == 1 and draw the exact same sequence of cards.
  int event_keyed_matches_part2 = 0;
  int deck_order_mismatches_part2 = 0;
  for (uint64_t seed = 1; seed <= num_seeds; ++seed) {
    MidRoundReshuffleCrnTestState state_a_evk(game, /*is_branch_b=*/false);
    MidRoundReshuffleCrnTestState state_b_evk(game, /*is_branch_b=*/true);

    SimulateRolloutWithTurnTrajectory(
        &state_a_evk, 0, nullptr, seed,
        SearchTruncationMode::kTerminal, SearchRolloutCrnMode::kEventKeyedCrn);
    SimulateRolloutWithTurnTrajectory(
        &state_b_evk, 0, nullptr, seed,
        SearchTruncationMode::kTerminal, SearchRolloutCrnMode::kEventKeyedCrn);

    SPIEL_CHECK_EQ(state_a_evk.history_actions().size(), 2);
    SPIEL_CHECK_EQ(state_b_evk.history_actions().size(), 2);
    if (state_a_evk.history_actions() == state_b_evk.history_actions()) {
      event_keyed_matches_part2++;
    }

    // Compare against kDeckOrderCrn (which desynchronizes because Branch A's reshuffle occurred before rollout)
    MidRoundReshuffleCrnTestState state_a_doc(game, /*is_branch_b=*/false);
    MidRoundReshuffleCrnTestState state_b_doc(game, /*is_branch_b=*/true);

    SimulateRolloutWithTurnTrajectory(
        &state_a_doc, 0, nullptr, seed,
        SearchTruncationMode::kTerminal, SearchRolloutCrnMode::kDeckOrderCrn);
    SimulateRolloutWithTurnTrajectory(
        &state_b_doc, 0, nullptr, seed,
        SearchTruncationMode::kTerminal, SearchRolloutCrnMode::kDeckOrderCrn);

    if (state_a_doc.history_actions() != state_b_doc.history_actions()) {
      deck_order_mismatches_part2++;
    }
  }
  SPIEL_CHECK_EQ(event_keyed_matches_part2, num_seeds);
  SPIEL_CHECK_GT(deck_order_mismatches_part2, 0);
}

void CrnDiagnostic30RootsBenchmark(const std::shared_ptr<const Game>& game) {
  std::cout << "\n=== Running 30-Root CRN Diagnostic Benchmark ===" << std::endl;
  const std::vector<std::shared_ptr<BatchedNNEvaluator>> evaluators = {
      std::make_shared<FixedPriorEvaluator>()};

  // 1. Generate 30 mid-game high-stakes root states from real games:
  std::vector<std::unique_ptr<State>> roots;
  uint64_t game_seed = 20261030ULL;
  while (roots.size() < 30) {
    auto state = game->NewInitialState();
    std::mt19937_64 rng(game_seed++);
    while (!state->IsTerminal()) {
      if (state->IsChanceNode()) {
        auto outcomes = state->ChanceOutcomes();
        double u = std::generate_canonical<double, 53>(rng);
        Action a = open_spiel::SampleAction(outcomes, u).first;
        state->ApplyAction(a);
        continue;
      }
      const auto* dune = dynamic_cast<const DuneImperiumState*>(state.get());
      if (dune && dune->GetCurrentRound() >= 3 &&
          dune->phase() == GamePhase::kAgentTurns) {
        auto legals = state->LegalActions();
        if (legals.size() >= 3) {
          roots.push_back(state->Clone());
          if (roots.size() >= 30) break;
        }
      }
      auto legals = state->LegalActions();
      if (legals.empty()) break;
      std::uniform_int_distribution<size_t> dist(0, legals.size() - 1);
      state->ApplyAction(legals[dist(rng)]);
    }
  }
  SPIEL_CHECK_EQ(roots.size(), 30);
  std::cout << "Successfully sampled " << roots.size()
            << " mid-game roots (Round >= 3, kAgentTurns, >= 3 legal actions)."
            << std::endl;

  struct ModeStats {
    std::string name;
    double mean_paired_corr = 0.0;
    double mean_paired_diff_var = 0.0;
    int pair_count = 0;
    double elapsed_ms = 0.0;
  };

  std::vector<ModeStats> results;
  const std::vector<SearchRolloutCrnMode> modes = {
      SearchRolloutCrnMode::kSequential,
      SearchRolloutCrnMode::kDeckOrderCrn,
      SearchRolloutCrnMode::kEventKeyedCrn};

  for (auto mode : modes) {
    const auto t0 = std::chrono::steady_clock::now();
    double total_sum_corr = 0.0;
    int total_corr_pairs = 0;
    double total_sum_diff_var = 0.0;
    int total_diff_var_pairs = 0;

    for (size_t r = 0; r < roots.size(); ++r) {
      const uint64_t dec_seed = 10007ULL + r * 137ULL;
      const auto legals = roots[r]->LegalActions();
      const std::vector<Action> candidates(
          legals.begin(), legals.begin() + std::min<size_t>(3, legals.size()));
      const Player seat = roots[r]->CurrentPlayer();
      const auto result = PairedPriorSortedRolloutSearchDecision(
          *roots[r], seat, /*turn_tree=*/nullptr, evaluators,
          candidates, /*rollouts_per_action=*/16,
          /*min_override_margin=*/0.15, /*min_t_stat=*/1.0, dec_seed,
          mode, /*sample_opponent_leader_draft=*/false,
          /*max_time_ms=*/-1.0, /*max_inflight_worlds=*/32,
          /*resample_from_infostate=*/false);
      const auto stats =
          ComputeMeanPairedCorrelation(result.candidate_rollout_utilities);
      total_sum_corr += stats.sum_corr;
      total_corr_pairs += stats.corr_pair_count;
      total_sum_diff_var += stats.sum_diff_var;
      total_diff_var_pairs += stats.diff_var_pair_count;
    }

    const double elapsed = std::chrono::duration<double, std::milli>(
                               std::chrono::steady_clock::now() - t0)
                               .count();
    std::string mode_str = (mode == SearchRolloutCrnMode::kSequential)
                               ? "sequential"
                               : (mode == SearchRolloutCrnMode::kDeckOrderCrn)
                                     ? "deck_order"
                                     : "event_keyed";
    const double mean_corr =
        total_corr_pairs > 0 ? (total_sum_corr / total_corr_pairs) : 0.0;
    const double mean_diff_var = total_diff_var_pairs > 0
                                     ? (total_sum_diff_var / total_diff_var_pairs)
                                     : 0.0;
    results.push_back(
        {mode_str, mean_corr, mean_diff_var, total_corr_pairs, elapsed});
  }

  std::cout << "\n--- Diagnostic Results (30 roots, 16 rollouts/candidate, K=3) ---"
            << std::endl;
  for (const auto& r : results) {
    std::cout << "Mode: " << r.name
              << " | mean_paired_corr: " << r.mean_paired_corr
              << " | mean_paired_diff_var: " << r.mean_paired_diff_var
              << " | pairs: " << r.pair_count
              << " | time: " << r.elapsed_ms << " ms" << std::endl;
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

class SleepingPayoffState : public State {
 public:
  SleepingPayoffState(std::shared_ptr<const Game> game, std::vector<Action> actions,
                      Action boosted_action = kInvalidAction, double advantage = 1.0,
                      int sleep_ms_per_step = 2)
      : State(game), actions_(std::move(actions)),
        boosted_action_(boosted_action), advantage_(advantage),
        sleep_ms_(sleep_ms_per_step) {}

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
  std::string ToString() const override { return "SleepingPayoffState"; }
  bool IsTerminal() const override { return phase_ == 3; }
  std::vector<double> Returns() const override {
    const double value = (selected_ == boosted_action_ ? advantage_ : 0.0);
    return {value, -value, 0.0, 0.0};
  }
  std::unique_ptr<State> Clone() const override {
    return std::make_unique<SleepingPayoffState>(*this);
  }

 protected:
  void DoApplyAction(Action action) override {
    if (sleep_ms_ > 0) {
      std::this_thread::sleep_for(std::chrono::milliseconds(sleep_ms_));
    }
    if (phase_ == 0) selected_ = action;
    ++phase_;
  }

 private:
  std::vector<Action> actions_;
  Action boosted_action_;
  double advantage_;
  int sleep_ms_;
  int phase_ = 0;
  Action selected_ = kInvalidAction;
};

void Stage1OverAllocationStage2ReservationTest(const std::shared_ptr<const Game>& game) {
  const std::vector<std::shared_ptr<BatchedNNEvaluator>> evaluators = {
      std::make_shared<FixedPriorEvaluator>()};
  const auto actions = Actions(2);
  const Action raw = actions[0];
  const Action challenger = actions[1];

  SleepingPayoffState state(game, actions, challenger, /*advantage=*/1.0, /*sleep_ms_per_step=*/2);
  TurnSearchTree tree;
  const auto result = SuccessiveHalvingRolloutSearchDecision(
      state, 0, &tree, evaluators, /*reference_rollouts_per_action=*/400,
      /*min_override_margin=*/0.15, /*decision_seed=*/42,
      SearchRolloutCrnMode::kSequential, /*unbiased_stage2_gate=*/false,
      /*max_time_ms=*/120.0);

  SPIEL_CHECK_TRUE(result.overridden);
  SPIEL_CHECK_EQ(result.action, challenger);
  SPIEL_CHECK_EQ(result.termination_reason, "override");
  SPIEL_CHECK_GE(result.stage1_kept_worlds, 2);
  SPIEL_CHECK_LT(result.stage1_kept_worlds, result.stage1_planned_worlds);
  SPIEL_CHECK_GE(result.stage2_kept_worlds, 2);
}

class PrefixState : public State {
 public:
  PrefixState(std::shared_ptr<const Game> game, std::vector<Action> actions,
              int world_id = 0)
      : State(game), actions_(std::move(actions)), world_id_(world_id) {}

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
  std::string ToString() const override { return "PrefixState"; }
  bool IsTerminal() const override { return phase_ == 3; }
  std::vector<double> Returns() const override {
    const double val = (world_id_ >= 2) ? 9999.0 : 1.0;
    return {val, -val, 0.0, 0.0};
  }
  std::unique_ptr<State> Clone() const override {
    auto copy = std::make_unique<PrefixState>(*this);
    // In PairedPriorSortedRolloutSearchDecision with 2 candidates, each world launches 2 rollouts
    copy->world_id_ = (global_clone_counter_++) / 2;
    return copy;
  }

  static void ResetCloneCounter() { global_clone_counter_ = 0; }

 protected:
  void DoApplyAction(Action action) override {
    if (world_id_ == 1 && phase_ > 0) {
      // Sleep across steps past the 40ms deadline
      std::this_thread::sleep_for(std::chrono::milliseconds(60));
    }
    if (phase_ == 0) selected_ = action;
    ++phase_;
  }

 private:
  std::vector<Action> actions_;
  int world_id_;
  int phase_ = 0;
  Action selected_ = kInvalidAction;
  inline static std::atomic<int> global_clone_counter_{0};
};

void ContiguousPrefixOrderingTest(const std::shared_ptr<const Game>& game) {
  const std::vector<std::shared_ptr<BatchedNNEvaluator>> evaluators = {
      std::make_shared<FixedPriorEvaluator>()};
  const auto actions = Actions(2);

  PrefixState::ResetCloneCounter();
  PrefixState state(game, actions);
  TurnSearchTree tree;

  const auto result = PairedPriorSortedRolloutSearchDecision(
      state, 0, &tree, evaluators, actions, /*rollouts_per_action=*/8,
      /*min_override_margin=*/0.15, /*min_t_stat=*/1.0, /*decision_seed=*/42,
      SearchRolloutCrnMode::kSequential, /*sample_opponent_leader_draft=*/false,
      /*max_time_ms=*/40.0, /*max_inflight_worlds=*/32);

  SPIEL_CHECK_EQ(result.stage1_kept_worlds, 1);
  SPIEL_CHECK_EQ(result.candidate_rollout_utilities[0].second.size(), 1);
  SPIEL_CHECK_EQ(result.candidate_rollout_utilities[0].second[0], 1.0);
  SPIEL_CHECK_EQ(tree.GetVisits(actions[0]), 1);
  SPIEL_CHECK_EQ(tree.GetVisits(actions[1]), 1);
}

void DepthGuardTest() {
  const auto temp_dir = std::filesystem::temp_directory_path() / "depth_guard_test";
  std::filesystem::remove_all(temp_dir);
  std::filesystem::create_directories(temp_dir);

  open_spiel::SetErrorHandler([](const std::string& err) {
    throw std::runtime_error(err);
  });

  try {
    // 1. Deep checkpoint (13 blocks) into shallow model (12 blocks) triggers SpielFatalError
    const auto deep_pt = (temp_dir / "model_13.pt").string();
    auto deep_model = std::make_shared<SharedDunePolicyValueNetImpl>(
        /*input_dim=*/5000, /*hidden_dim=*/16, /*action_dim=*/100, /*num_blocks=*/13);
    torch::save(deep_model, deep_pt);

    auto shallow_model = std::make_shared<SharedDunePolicyValueNetImpl>(
        /*input_dim=*/5000, /*hidden_dim=*/16, /*action_dim=*/100, /*num_blocks=*/12);

    bool caught_deep = false;
    try {
      LoadModelCheckpointRobust(shallow_model, deep_pt, torch::kCPU);
    } catch (const std::runtime_error& e) {
      caught_deep = true;
      SPIEL_CHECK_TRUE(std::string(e.what()).find("exceeding model num_blocks") != std::string::npos);
    }
    SPIEL_CHECK_TRUE(caught_deep);

    // 2. Mismatched sidecar JSON (declared 16 != model 12) triggers SpielFatalError
    const auto shallow_pt = (temp_dir / "model_12.pt").string();
    const auto shallow_json = (temp_dir / "model_12.json").string();
    torch::save(shallow_model, shallow_pt);
    {
      std::ofstream out(shallow_json);
      out << "{\"num_blocks\": 16}\n";
    }

    bool caught_json_mismatch = false;
    try {
      LoadModelCheckpointRobust(shallow_model, shallow_pt, torch::kCPU);
    } catch (const std::runtime_error& e) {
      caught_json_mismatch = true;
      SPIEL_CHECK_TRUE(std::string(e.what()).find("does not match model num_blocks") != std::string::npos);
    }
    SPIEL_CHECK_TRUE(caught_json_mismatch);

    // 3. Symlink to pt with mismatched sidecar JSON triggers SpielFatalError
    const auto symlink_pt = (temp_dir / "symlink_12.pt").string();
    std::filesystem::create_symlink(shallow_pt, symlink_pt);

    bool caught_symlink_mismatch = false;
    try {
      LoadModelCheckpointRobust(shallow_model, symlink_pt, torch::kCPU);
    } catch (const std::runtime_error& e) {
      caught_symlink_mismatch = true;
      SPIEL_CHECK_TRUE(std::string(e.what()).find("does not match model num_blocks") != std::string::npos);
    }
    SPIEL_CHECK_TRUE(caught_symlink_mismatch);

    // 4. Matching sidecar JSON (declared 12 == model 12) loads cleanly without error
    {
      std::ofstream out(shallow_json);
      out << "{\"num_blocks\": 12}\n";
    }
    LoadModelCheckpointRobust(shallow_model, shallow_pt, torch::kCPU);
    LoadModelCheckpointRobust(shallow_model, symlink_pt, torch::kCPU);

    // 5. Matching deep model (13 blocks into 13 blocks) loads cleanly with matching sidecar
    const auto deep_json = (temp_dir / "model_13.json").string();
    {
      std::ofstream out(deep_json);
      out << "{\"num_blocks\": 13}\n";
    }
    LoadModelCheckpointRobust(deep_model, deep_pt, torch::kCPU);
  } catch (...) {
    open_spiel::SetErrorHandler(nullptr);
    std::filesystem::remove_all(temp_dir);
    throw;
  }

  open_spiel::SetErrorHandler(nullptr);
  std::filesystem::remove_all(temp_dir);
}

void AdaptivePublicKnowledgeCorrectionTest() {
  auto state = LoadGame("dune_imperium")->NewInitialState();
  auto* dune = static_cast<DuneImperiumState*>(state.get());
  for (int steps = 0;; ++steps) {
    SPIEL_CHECK_LT(steps, 1000);
    if (!state->IsChanceNode() && dune->phase() == GamePhase::kAgentTurns) break;
    state->ApplyAction(state->IsChanceNode() ? state->ChanceOutcomes().front().first
                                             : state->LegalActions().front());
  }
  const Player mover = state->CurrentPlayer();
  const Player opponent = (mover + 1) % kNumPlayers;
  const auto opponent_hand = dune->GetPlayerHandForTesting(opponent);
  SPIEL_CHECK_GT(opponent_hand.size(), 1);
  SPIEL_CHECK_TRUE(dune->SynchronizePublicHandKnowledge(opponent, {opponent_hand.front()}));

  Action prefix = kInvalidAction;
  std::unique_ptr<State> descendant;
  for (Action action : state->LegalActions()) {
    if (action < kActionSelectAgentCard0 ||
        action >= kActionSelectAgentCard0 + kNumImperiumCardIds) continue;
    auto next = state->Clone();
    next->ApplyAction(action);
    if (!next->IsChanceNode() && next->CurrentPlayer() == mover &&
        next->LegalActions().size() > 1) {
      prefix = action;
      descendant = std::move(next);
      break;
    }
  }
  SPIEL_CHECK_TRUE(descendant != nullptr);
  auto corrected_descendant = descendant->Clone();
  auto* corrected = static_cast<DuneImperiumState*>(corrected_descendant.get());
  SPIEL_CHECK_TRUE(corrected->SynchronizePublicHandKnowledge(opponent, opponent_hand));
  SPIEL_CHECK_NE(static_cast<DuneImperiumState*>(descendant.get())
      ->PublicCardKnowledgeRecordDigest(), corrected->PublicCardKnowledgeRecordDigest());

  const auto temp_dir = std::filesystem::temp_directory_path() /
      ("dune_adaptive_public_card_" + std::to_string(
          std::chrono::steady_clock::now().time_since_epoch().count()));
  std::filesystem::create_directories(temp_dir);
  const int previous_cpu_threads = at::get_num_threads();
  at::set_num_threads(1);
  try {
    const auto checkpoint = (temp_dir / "tiny.pt").string();
    auto model = std::make_shared<SharedDunePolicyValueNetImpl>(
        kLegacyInformationStateSize, /*hidden_dim=*/16,
        kNumDistinctPlayerActions, /*num_blocks=*/1);
    {
      torch::NoGradGuard no_grad;
      for (auto& parameter : model->parameters()) parameter.zero_();
      model->policy_head->bias.index_put_({prefix}, 10.0);
      model->policy_head->bias.index_put_({kActionReveal}, 9.0);
    }
    torch::save(model, checkpoint);
    {
      const auto schema = GetInformationStateSchema(MarketAppendixMode::kNone);
      std::ofstream metadata(temp_dir / "tiny.json");
      metadata << "{\"observation_dim\":" << kLegacyInformationStateSize
               << ",\"hidden_dim\":16,\"num_blocks\":1,"
               << "\"market_appendix_mode\":\"none\","
               << "\"feature_schema_sha256\":\"" << schema.sha256 << "\","
               << "\"enable_semantic_scorer\":false}\n";
    }
    DuneAdaptiveRolloutSession session(
        checkpoint, "cpu", /*hidden_dim=*/16, /*num_blocks=*/1,
        /*reference_rollouts_per_action=*/4, /*leader_rollouts=*/2,
        /*target_batch_size=*/16, /*seed=*/1138,
        /*use_successive_halving=*/false, /*fixed_top5_candidates=*/false,
        /*max_inflight_worlds=*/1, /*num_evaluators=*/1);
    const auto initial = session.Search(*state);
    session.DiscardPendingAction();
    SPIEL_CHECK_GT(initial.simulations_completed, 0);

    auto recaptured = state->Clone();
    SPIEL_CHECK_TRUE(static_cast<DuneImperiumState*>(recaptured.get())
        ->SynchronizePublicHandKnowledge(opponent, {opponent_hand.front()}));
    const double before_recapture = session.turn_elapsed_search_time_ms();
    const auto repeated = session.Search(*recaptured, /*remaining_time_ms=*/0.0);
    session.DiscardPendingAction();
    SPIEL_CHECK_EQ(repeated.diagnostics.termination_reason, "tree_reuse");
    SPIEL_CHECK_GT(repeated.diagnostics.inherited_root_visits, 0);
    SPIEL_CHECK_EQ(repeated.simulations_completed, 0);
    SPIEL_CHECK_FLOAT_EQ(session.turn_elapsed_search_time_ms(),
                         before_recapture + repeated.elapsed_time_ms);

    const double before_action = session.turn_elapsed_search_time_ms();
    const auto ordinary = session.Search(*descendant, /*remaining_time_ms=*/0.0);
    session.DiscardPendingAction();
    SPIEL_CHECK_EQ(ordinary.diagnostics.termination_reason, "tree_reuse");
    SPIEL_CHECK_GT(ordinary.diagnostics.inherited_root_visits, 0);
    SPIEL_CHECK_EQ(ordinary.simulations_completed, 0);
    SPIEL_CHECK_FLOAT_EQ(session.turn_elapsed_search_time_ms(),
                         before_action + ordinary.elapsed_time_ms);

    const double before_same_history = session.turn_elapsed_search_time_ms();
    const auto same_history = session.Search(*corrected_descendant, /*remaining_time_ms=*/0.0);
    session.DiscardPendingAction();
    SPIEL_CHECK_EQ(same_history.diagnostics.inherited_root_visits, 0);
    SPIEL_CHECK_EQ(same_history.simulations_completed, 0);
    SPIEL_CHECK_EQ(same_history.fallback_reason, "live_deadline_reached");
    SPIEL_CHECK_FLOAT_EQ(session.turn_elapsed_search_time_ms(),
                         before_same_history + same_history.elapsed_time_ms);

    session.ResetSession("descendant correction fixture");
    session.Search(*state);
    session.DiscardPendingAction();
    const double before_correction = session.turn_elapsed_search_time_ms();
    const auto advanced_correction = session.Search(
        *corrected_descendant, /*remaining_time_ms=*/0.0);
    session.DiscardPendingAction();
    SPIEL_CHECK_EQ(advanced_correction.diagnostics.inherited_root_visits, 0);
    SPIEL_CHECK_EQ(advanced_correction.simulations_completed, 0);
    SPIEL_CHECK_EQ(advanced_correction.fallback_reason, "live_deadline_reached");
    SPIEL_CHECK_FLOAT_EQ(session.turn_elapsed_search_time_ms(),
                         before_correction + advanced_correction.elapsed_time_ms);
  } catch (...) {
    std::filesystem::remove_all(temp_dir);
    at::set_num_threads(previous_cpu_threads);
    throw;
  }
  std::filesystem::remove_all(temp_dir);
  at::set_num_threads(previous_cpu_threads);
}

class ThrowingRolloutState : public State {
 public:
  ThrowingRolloutState(std::shared_ptr<const Game> game, std::vector<Action> actions,
                       int throw_world_id = 1, int sleep_ms = 2)
      : State(game), actions_(std::move(actions)),
        throw_world_id_(throw_world_id), sleep_ms_(sleep_ms) {}

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
  std::string ToString() const override { return "ThrowingRolloutState"; }
  bool IsTerminal() const override { return phase_ == 3; }
  std::vector<double> Returns() const override {
    return {1.0, -1.0, 0.0, 0.0};
  }
  std::unique_ptr<State> Clone() const override {
    auto copy = std::make_unique<ThrowingRolloutState>(*this);
    copy->world_id_ = (global_clone_counter_++) / num_cands_;
    return copy;
  }

  static void ResetCloneCounter(int num_cands = 2) {
    global_clone_counter_ = 0;
    num_cands_ = num_cands;
  }

 protected:
  void DoApplyAction(Action action) override {
    if (world_id_ == throw_world_id_ && phase_ > 0) {
      throw std::runtime_error("simulated rollout exception");
    }
    if (sleep_ms_ > 0 && phase_ > 0) {
      std::this_thread::sleep_for(std::chrono::milliseconds(sleep_ms_));
    }
    if (phase_ == 0) selected_ = action;
    ++phase_;
  }

 private:
  std::vector<Action> actions_;
  int throw_world_id_;
  int sleep_ms_;
  int world_id_ = 0;
  int phase_ = 0;
  Action selected_ = kInvalidAction;
  inline static std::atomic<int> global_clone_counter_{0};
  inline static int num_cands_{2};
};

void ClockedExceptionSafetyTest(const std::shared_ptr<const Game>& game) {
  const std::vector<std::shared_ptr<BatchedNNEvaluator>> evaluators = {
      std::make_shared<FixedPriorEvaluator>()};
  const auto actions = Actions(2);

  // 1. SuccessiveHalvingRolloutSearchDecision (50 iterations)
  for (int i = 0; i < 50; ++i) {
    const auto t0 = std::chrono::steady_clock::now();
    bool caught = false;
    try {
      ThrowingRolloutState::ResetCloneCounter(2);
      ThrowingRolloutState state(game, actions, /*throw_world_id=*/1, /*sleep_ms=*/2);
      TurnSearchTree tree;
      SuccessiveHalvingRolloutSearchDecision(
          state, 0, &tree, evaluators, /*reference_rollouts_per_action=*/16,
          /*min_override_margin=*/0.15, /*decision_seed=*/42 + i,
          SearchRolloutCrnMode::kSequential, /*unbiased_stage2_gate=*/false,
          /*max_time_ms=*/500.0);
    } catch (const std::runtime_error& e) {
      caught = true;
      SPIEL_CHECK_EQ(std::string(e.what()), "simulated rollout exception");
    }
    const double elapsed_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t0).count();
    SPIEL_CHECK_TRUE(caught);
    SPIEL_CHECK_LT(elapsed_ms, 1000.0);
  }

  // 2. PairedPriorSortedRolloutSearchDecision (50 iterations)
  for (int i = 0; i < 50; ++i) {
    const auto t0 = std::chrono::steady_clock::now();
    bool caught = false;
    try {
      ThrowingRolloutState::ResetCloneCounter(2);
      ThrowingRolloutState state(game, actions, /*throw_world_id=*/1, /*sleep_ms=*/2);
      TurnSearchTree tree;
      PairedPriorSortedRolloutSearchDecision(
          state, 0, &tree, evaluators, actions, /*rollouts_per_action=*/16,
          /*min_override_margin=*/0.15, /*min_t_stat=*/1.0, /*decision_seed=*/42 + i,
          SearchRolloutCrnMode::kSequential, /*sample_opponent_leader_draft=*/false,
          /*max_time_ms=*/500.0);
    } catch (const std::runtime_error& e) {
      caught = true;
      SPIEL_CHECK_EQ(std::string(e.what()), "simulated rollout exception");
    }
    const double elapsed_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t0).count();
    SPIEL_CHECK_TRUE(caught);
    SPIEL_CHECK_LT(elapsed_ms, 1000.0);
  }
}

}  // namespace
}  // namespace dune_imperium
}  // namespace open_spiel

int main(int argc, char** argv) {
  using namespace open_spiel::dune_imperium;
  if (argc > 1 && std::string(argv[1]) == "public-card-adaptive") {
    AdaptivePublicKnowledgeCorrectionTest();
    return 0;
  }
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
  EventKeyedCrnExtraDrawAndInsertedCardTest(game);
  CrnDiagnostic30RootsBenchmark(game);
  Stage1OverAllocationStage2ReservationTest(game);
  ContiguousPrefixOrderingTest(game);
  DepthGuardTest();
  AdaptivePublicKnowledgeCorrectionTest();
  ClockedExceptionSafetyTest(game);
}
