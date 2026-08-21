#pragma once

#include <memory>
#include <vector>
#include <cmath>
#include <limits>

#include "open_spiel/spiel.h"
#include "open_spiel/algorithms/mcts.h"
#include "dune_network.h"
#include "dune_evaluator.h"

namespace open_spiel {

class BatchedNNEvaluator : public algorithms::Evaluator {
 public:
  BatchedNNEvaluator(
      std::shared_ptr<open_spiel::BatchedEvaluator> batched_eval,
      float logit_cap = 10.0f)
      : batched_eval_(batched_eval),
        logit_cap_(logit_cap) {
    obs_size_ = 5580;
  }

  std::vector<double> Evaluate(const State& state) override {
    int num_players = state.NumPlayers();
    std::vector<double> values(num_players, 0.0);
    std::vector<std::vector<float>> observations;
    observations.reserve(num_players);
    for (int p = 0; p < num_players; ++p) {
      observations.push_back(state.InformationStateTensor(p));
    }
    auto results = batched_eval_->EvaluateBatchValues(observations);
    for (int p = 0; p < num_players; ++p) {
      double val = results[p].value;
      values[p] = val;
      DuneNNEvaluator::RecordLeafValue(val);
    }
    return values;
  }

  ActionsAndProbs Prior(const State& state) override {
    if (state.IsTerminal()) {
      return {};
    }
    Player current_player = state.CurrentPlayer();
    if (current_player < 0 || current_player >= state.NumPlayers()) {
      return {};
    }
    std::vector<Action> legal_actions = state.LegalActions();
    if (legal_actions.empty()) {
      return {};
    }
    std::vector<float> obs = state.InformationStateTensor(current_player);
    open_spiel::CompactEvalResult result =
        batched_eval_->EvaluateCompact(obs, legal_actions);

    ActionsAndProbs policy;
    policy.reserve(result.actions.size());
    for (size_t i = 0; i < result.actions.size(); ++i) {
      policy.push_back({result.actions[i], result.probabilities[i]});
    }
    return policy;
  }

  std::pair<ActionsAndProbs, std::vector<double>> PriorAndEvaluate(const State& state) override {
    int num_players = state.NumPlayers();
    std::vector<double> values(num_players, 0.0);
    ActionsAndProbs policy;

    if (state.IsTerminal()) {
      return {policy, values};
    }
    Player current_player = state.CurrentPlayer();
    if (current_player < 0 || current_player >= num_players) {
      return {policy, values};
    }

    std::vector<std::vector<float>> observations;
    observations.reserve(num_players);
    for (int p = 0; p < num_players; ++p) {
      observations.push_back(state.InformationStateTensor(p));
    }
    std::vector<Action> legal_actions;
    if (current_player >= 0 && current_player < num_players) {
      legal_actions = state.LegalActions();
    }
    auto value_and_prior = batched_eval_->EvaluateBatchValuesWithCompactPrior(
        observations, static_cast<size_t>(current_player),
        legal_actions);
    auto& results = value_and_prior.first;
    const auto& compact_prior = value_and_prior.second;

    for (int p = 0; p < num_players; ++p) {
      const open_spiel::EvalResult& result = results[p];
      double val = result.value;
      values[p] = val;
      DuneNNEvaluator::RecordLeafValue(val);

    }
    policy.reserve(compact_prior.actions.size());
    for (size_t i = 0; i < compact_prior.actions.size(); ++i) {
      policy.push_back({compact_prior.actions[i], compact_prior.probabilities[i]});
    }
    return {policy, values};
  }

 private:
  std::shared_ptr<open_spiel::BatchedEvaluator> batched_eval_;
  float logit_cap_;
  int64_t obs_size_;
};

} // namespace open_spiel
