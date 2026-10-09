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

inline std::atomic<uint64_t> g_total_network_evaluations{0};

inline uint64_t GetTotalNetworkEvaluations() {
  return g_total_network_evaluations.load(std::memory_order_relaxed);
}

inline void ResetTotalNetworkEvaluations() {
  g_total_network_evaluations.store(0, std::memory_order_relaxed);
}

class BatchedNNEvaluator : public algorithms::Evaluator {
 public:
  BatchedNNEvaluator(
      std::shared_ptr<open_spiel::BatchedEvaluator> batched_eval,
      float logit_cap = 10.0f,
      dune_imperium::MarketAppendixMode market_mode = dune_imperium::MarketAppendixMode::kNone)
      : batched_eval_(batched_eval),
        logit_cap_(logit_cap),
        market_mode_(market_mode) {
    if (batched_eval_) {
      obs_size_ = batched_eval_->ModelInputDim();
      if (obs_size_ == dune_imperium::kFullPublicInformationStateSize) {
        if (!batched_eval_->HasSemanticScorer() ||
            (batched_eval_->SemanticDescriptorSchema() != dune_semantic::kDescriptorSchemaVersionV3 &&
             batched_eval_->SemanticDescriptorSchema() != dune_semantic::kDescriptorSchemaVersionV4)) {
          SpielFatalError("BatchedNNEvaluator: 9182 actor requires an active semantic scorer with schema v3 or v4");
        }
      }
      if (market_mode_ == dune_imperium::MarketAppendixMode::kNone &&
          batched_eval_->MarketMode() != dune_imperium::MarketAppendixMode::kNone) {
        market_mode_ = batched_eval_->MarketMode();
      }
      if (market_mode_ == dune_imperium::MarketAppendixMode::kNone) {
        if (obs_size_ == dune_imperium::kFullPublicInformationStateSize) {
          market_mode_ = dune_imperium::MarketAppendixMode::kFullPublicInformationV3;
        } else if (obs_size_ == dune_imperium::kOrderedCardSlotsInformationStateSize) {
          market_mode_ = dune_imperium::MarketAppendixMode::kOrderedCardSlotsV2;
        } else if (obs_size_ == dune_imperium::kExpandedInformationStateSize) {
          SpielFatalError("Ambiguous 6,215 model requires explicit market_appendix_mode (cannot default to kZeros).");
        }
      }
    } else {
      obs_size_ = 5580;
    }
  }

  std::vector<float> GetConsumedObservation(const State& state, Player player) const {
    return ModelObservation(state, player);
  }

  std::string SemanticDescriptorSchema() const {
    return batched_eval_ ? batched_eval_->SemanticDescriptorSchema() : dune_semantic::kDescriptorSchemaVersionV3;
  }

  std::vector<double> Evaluate(const State& state) override {
    int num_players = state.NumPlayers();
    std::vector<double> values(num_players, 0.0);
    std::vector<std::vector<float>> observations;
    observations.reserve(num_players);
    for (int p = 0; p < num_players; ++p) {
      observations.push_back(ModelObservation(state, p));
    }
    g_total_network_evaluations.fetch_add(num_players, std::memory_order_relaxed);
    auto results = batched_eval_->EvaluateBatchValues(observations);
    for (int p = 0; p < num_players; ++p) {
      double val = results[p].value;
      values[p] = val;
      DuneNNEvaluator::RecordLeafValue(val);
    }
    return values;
  }

  double EvaluatePlayerValue(const State& state, Player player) {
    if (!batched_eval_) return 0.0;
    std::vector<std::vector<float>> observations = {ModelObservation(state, player)};
    g_total_network_evaluations.fetch_add(1, std::memory_order_relaxed);
    auto results = batched_eval_->EvaluateBatchValues(observations);
    double val = results.empty() ? 0.0 : results[0].value;
    DuneNNEvaluator::RecordLeafValue(val);
    return val;
  }

  open_spiel::CompactEvalResult PriorWithDetails(const State& state) {
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
    std::vector<float> obs = ModelObservation(state, current_player);
    dune_semantic::CandidateActionData cand_data;
    const auto* dune = dynamic_cast<const dune_imperium::DuneImperiumState*>(&state);
    if (dune != nullptr) {
      const std::string schema = batched_eval_ ? batched_eval_->SemanticDescriptorSchema() : dune_semantic::kDescriptorSchemaVersionV3;
      dune_semantic::ExtractCandidateDescriptors(*dune, legal_actions, &cand_data, schema);
    }
    g_total_network_evaluations.fetch_add(1, std::memory_order_relaxed);
    return batched_eval_->EvaluateCompact(obs, legal_actions, &cand_data);
  }

  ActionsAndProbs Prior(const State& state) override {
    open_spiel::CompactEvalResult result = PriorWithDetails(state);
    if (!result.ok || result.probabilities.empty()) {
      return {};
    }
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
      observations.push_back(ModelObservation(state, p));
    }
    std::vector<Action> legal_actions;
    dune_semantic::CandidateActionData cand_data;
    if (current_player >= 0 && current_player < num_players) {
      legal_actions = state.LegalActions();
      const auto* dune = dynamic_cast<const dune_imperium::DuneImperiumState*>(&state);
      if (dune != nullptr) {
        const std::string schema = batched_eval_ ? batched_eval_->SemanticDescriptorSchema() : dune_semantic::kDescriptorSchemaVersionV3;
        dune_semantic::ExtractCandidateDescriptors(*dune, legal_actions, &cand_data, schema);
      }
    }
    g_total_network_evaluations.fetch_add(num_players, std::memory_order_relaxed);
    auto value_and_prior = batched_eval_->EvaluateBatchValuesWithCompactPrior(
        observations, static_cast<size_t>(current_player),
        legal_actions, &cand_data);
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
  std::vector<float> ModelObservation(const State& state, Player player) const {
    if (market_mode_ != dune_imperium::MarketAppendixMode::kNone) {
      const auto* dune = dynamic_cast<const dune_imperium::DuneImperiumState*>(&state);
      SPIEL_CHECK_TRUE(dune != nullptr);
      return dune->InformationStateTensorWithAppendix(player, market_mode_);
    }
    return state.InformationStateTensor(player);
  }

  std::shared_ptr<open_spiel::BatchedEvaluator> batched_eval_;
  float logit_cap_;
  int64_t obs_size_;
  dune_imperium::MarketAppendixMode market_mode_{dune_imperium::MarketAppendixMode::kNone};
};

} // namespace open_spiel
