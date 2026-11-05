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
      dune_imperium::MarketAppendixMode market_mode = dune_imperium::MarketAppendixMode::kNone,
      bool forward_active_rollouts = true)
      : batched_eval_(batched_eval),
        logit_cap_(logit_cap),
        market_mode_(market_mode),
        forward_active_rollouts_(forward_active_rollouts) {
    if (batched_eval_) {
      obs_size_ = batched_eval_->ModelInputDim();
      if (dune_imperium::IsFullPublicInformationStateSize(obs_size_)) {
        if (!batched_eval_->HasSemanticScorer() ||
            (batched_eval_->SemanticDescriptorSchema() != dune_semantic::kDescriptorSchemaVersionV3 &&
             batched_eval_->SemanticDescriptorSchema() != dune_semantic::kDescriptorSchemaVersionV4 &&
             batched_eval_->SemanticDescriptorSchema() != dune_semantic::kDescriptorSchemaVersionV5)) {
          SpielFatalError("BatchedNNEvaluator: Full public information actor requires an active semantic scorer with schema v3, v4, or v5");
        }
      }
      if (market_mode_ == dune_imperium::MarketAppendixMode::kNone &&
          batched_eval_->MarketMode() != dune_imperium::MarketAppendixMode::kNone) {
        market_mode_ = batched_eval_->MarketMode();
      }
      if (market_mode_ == dune_imperium::MarketAppendixMode::kNone) {
        if (obs_size_ == dune_imperium::kFullPublicInformationV5StateSize) {
          market_mode_ = dune_imperium::MarketAppendixMode::kFullPublicInformationV5;
        } else if (obs_size_ == dune_imperium::kFullPublicInformationV4StateSize) {
          market_mode_ = dune_imperium::MarketAppendixMode::kFullPublicInformationV4;
        } else if (obs_size_ == dune_imperium::kFullPublicInformationStateSize) {
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

  dune_imperium::DeckPoolSemantics DeckPoolSemantics() const {
    return batched_eval_ ? batched_eval_->DeckPoolSemantics() : dune_imperium::DeckPoolSemantics::kLegacy;
  }

  dune_imperium::MarketAppendixMode MarketMode() const {
    return market_mode_;
  }

  void AdjustActiveRollouts(int delta) {
    if (forward_active_rollouts_ && batched_eval_) {
      batched_eval_->AdjustActiveRollouts(delta);
    }
  }

  void SetFastGreedyEnabled(bool enabled) { enable_fast_greedy_ = enabled; }
  bool FastGreedyEnabled() const { return enable_fast_greedy_; }

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
    thread_local std::vector<std::vector<float>> tls_obs(1);
    FillModelObservation(state, player, &tls_obs[0]);
    g_total_network_evaluations.fetch_add(1, std::memory_order_relaxed);
    auto results = batched_eval_->EvaluateBatchValues(tls_obs);
    double val = results.empty() ? 0.0 : results[0].value;
    DuneNNEvaluator::RecordLeafValue(val);
    return val;
  }

  open_spiel::CompactEvalResult PriorWithDetails(
      const State& state, const std::vector<Action>& legal_actions) {
    if (!batched_eval_ || state.IsTerminal() || legal_actions.empty()) {
      return {};
    }
    Player current_player = state.CurrentPlayer();
    if (current_player < 0 || current_player >= state.NumPlayers()) {
      return {};
    }
    thread_local std::vector<float> tls_obs;
    FillModelObservation(state, current_player, &tls_obs);
    thread_local dune_semantic::CandidateActionData tls_cand_data;
    tls_cand_data.clear();
    const auto* dune = dynamic_cast<const dune_imperium::DuneImperiumState*>(&state);
    if (dune != nullptr) {
      const std::string schema = batched_eval_->SemanticDescriptorSchema();
      dune_semantic::ExtractCandidateDescriptors(*dune, legal_actions, &tls_cand_data, schema);
    }
    g_total_network_evaluations.fetch_add(1, std::memory_order_relaxed);
    return batched_eval_->EvaluateCompact(tls_obs, legal_actions, &tls_cand_data);
  }

  Action GreedyActionWithLegals(
      const State& state, const std::vector<Action>& legal_actions) {
    if (legal_actions.empty()) return kInvalidAction;
    if (batched_eval_ != nullptr && enable_fast_greedy_) {
      if (state.IsTerminal()) return legal_actions.front();
      Player current_player = state.CurrentPlayer();
      if (current_player < 0 || current_player >= state.NumPlayers()) return legal_actions.front();
      thread_local std::vector<float> tls_obs;
      FillModelObservation(state, current_player, &tls_obs);
      thread_local dune_semantic::CandidateActionData tls_cand_data;
      tls_cand_data.clear();
      const auto* dune = dynamic_cast<const dune_imperium::DuneImperiumState*>(&state);
      if (dune != nullptr) {
        const std::string schema = batched_eval_->SemanticDescriptorSchema();
        dune_semantic::ExtractCandidateDescriptors(*dune, legal_actions, &tls_cand_data, schema);
      }
      g_total_network_evaluations.fetch_add(1, std::memory_order_relaxed);
      open_spiel::GreedyEvalResult res = batched_eval_->EvaluateGreedyAction(tls_obs, legal_actions, &tls_cand_data);
      return (!res.ok || res.action == kInvalidAction) ? legal_actions.front() : res.action;
    }
    if (!batched_eval_) {
      ActionsAndProbs prior = Prior(state);
      if (prior.empty()) return legal_actions.front();
      double best_p = -1.0;
      Action best_a = legal_actions.front();
      for (Action a : legal_actions) {
        for (const auto& ap : prior) {
          if (ap.first == a) {
            if (ap.second > best_p) {
              best_p = ap.second;
              best_a = a;
            }
            break;
          }
        }
      }
      return best_a;
    }
    open_spiel::CompactEvalResult result = PriorWithDetails(state, legal_actions);
    if (!result.ok || result.probabilities.empty()) {
      return legal_actions.front();
    }
    float best_p = -1.0f;
    Action best_a = legal_actions.front();
    const size_t n = std::min(result.actions.size(), result.probabilities.size());
    for (size_t i = 0; i < n; ++i) {
      if (result.probabilities[i] > best_p) {
        best_p = result.probabilities[i];
        best_a = result.actions[i];
      }
    }
    return best_a;
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
    return PriorWithDetails(state, legal_actions);
  }

  ActionsAndProbs PriorWithLegals(
      const State& state, const std::vector<Action>& legal_actions) {
    if (!batched_eval_) {
      return Prior(state);
    }
    open_spiel::CompactEvalResult result = PriorWithDetails(state, legal_actions);
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
  void FillModelObservation(const State& state, Player player, std::vector<float>* out) const {
    const auto* dune = dynamic_cast<const dune_imperium::DuneImperiumState*>(&state);
    if (dune != nullptr) {
      const auto sem = DeckPoolSemantics();
      const size_t expected_size =
          (market_mode_ == dune_imperium::MarketAppendixMode::kNone)
              ? static_cast<size_t>(dune_imperium::kLegacyInformationStateSize)
              : static_cast<size_t>(
                    dune_imperium::GetInformationStateSchema(market_mode_, sem).size);
      out->resize(expected_size);
      dune->InformationStateTensorWithAppendix(
          player, market_mode_, absl::MakeSpan(*out), sem);
      return;
    }
    *out = state.InformationStateTensor(player);
  }

  std::vector<float> ModelObservation(const State& state, Player player) const {
    std::vector<float> out;
    FillModelObservation(state, player, &out);
    return out;
  }

  std::shared_ptr<open_spiel::BatchedEvaluator> batched_eval_;
  float logit_cap_;
  int64_t obs_size_;
  dune_imperium::MarketAppendixMode market_mode_{dune_imperium::MarketAppendixMode::kNone};
  bool forward_active_rollouts_{true};
  bool enable_fast_greedy_{std::getenv("DUNE_ENABLE_FAST_GREEDY") != nullptr &&
                           std::getenv("DUNE_DISABLE_FAST_GREEDY") == nullptr};
};

} // namespace open_spiel
