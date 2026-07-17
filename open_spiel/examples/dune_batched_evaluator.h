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
    auto results = batched_eval_->EvaluateBatch(observations);
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
    std::vector<float> obs = state.InformationStateTensor(current_player);
    open_spiel::EvalResult result = batched_eval_->Evaluate(obs);

    std::vector<Action> legal_actions = state.LegalActions();
    if (legal_actions.empty()) {
      return {};
    }

    std::vector<float> logits(result.logits.begin(), result.logits.end());
    open_spiel::CenterAndCapLegalLogits(logits, legal_actions, logit_cap_);

    double max_logit = -std::numeric_limits<double>::infinity();
    for (Action action : legal_actions) {
      if (action >= 0 && static_cast<size_t>(action) < logits.size()) {
        max_logit = std::max(max_logit, static_cast<double>(logits[action]));
      }
    }

    double sum_exp = 0.0;
    std::vector<double> exps(legal_actions.size());
    for (size_t i = 0; i < legal_actions.size(); ++i) {
      Action action = legal_actions[i];
      if (action >= 0 && static_cast<size_t>(action) < logits.size()) {
        exps[i] = std::exp(logits[action] - max_logit);
        sum_exp += exps[i];
      } else {
        exps[i] = 0.0;
      }
    }

    ActionsAndProbs policy;
    policy.reserve(legal_actions.size());
    for (size_t i = 0; i < legal_actions.size(); ++i) {
      double prob = (sum_exp > 0.0) ? (exps[i] / sum_exp) : (1.0 / legal_actions.size());
      policy.push_back({legal_actions[i], prob});
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

    std::vector<std::vector<float>> observations;
    observations.reserve(num_players);
    for (int p = 0; p < num_players; ++p) {
      observations.push_back(state.InformationStateTensor(p));
    }
    auto results = batched_eval_->EvaluateBatch(observations);

    for (int p = 0; p < num_players; ++p) {
      const open_spiel::EvalResult& result = results[p];
      double val = result.value;
      values[p] = val;
      DuneNNEvaluator::RecordLeafValue(val);

      if (p == current_player) {
        std::vector<Action> legal_actions = state.LegalActions();
        if (!legal_actions.empty()) {
          std::vector<float> logits(result.logits.begin(), result.logits.end());
          open_spiel::CenterAndCapLegalLogits(logits, legal_actions, logit_cap_);

          double max_logit = -std::numeric_limits<double>::infinity();
          for (Action action : legal_actions) {
            if (action >= 0 && static_cast<size_t>(action) < logits.size()) {
              max_logit = std::max(max_logit, static_cast<double>(logits[action]));
            }
          }

          double sum_exp = 0.0;
          std::vector<double> exps(legal_actions.size());
          for (size_t i = 0; i < legal_actions.size(); ++i) {
            Action action = legal_actions[i];
            if (action >= 0 && static_cast<size_t>(action) < logits.size()) {
              exps[i] = std::exp(logits[action] - max_logit);
              sum_exp += exps[i];
            } else {
              exps[i] = 0.0;
            }
          }

          policy.reserve(legal_actions.size());
          for (size_t i = 0; i < legal_actions.size(); ++i) {
            double prob = (sum_exp > 0.0) ? (exps[i] / sum_exp) : (1.0 / legal_actions.size());
            policy.push_back({legal_actions[i], prob});
          }
        }
      }
    }
    return {policy, values};
  }

 private:
  std::shared_ptr<open_spiel::BatchedEvaluator> batched_eval_;
  float logit_cap_;
  int64_t obs_size_;
};

} // namespace open_spiel
