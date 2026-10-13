#include "open_spiel/examples/dune_wide_rollout_search.h"

#include <algorithm>
#include <memory>
#include <vector>

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
}
