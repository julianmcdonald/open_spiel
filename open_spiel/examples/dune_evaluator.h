#pragma once

#include <memory>
#include <vector>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <limits>

#include <atomic>
#include <iostream>

#include "open_spiel/spiel.h"
#include "open_spiel/algorithms/mcts.h"
#include <torch/torch.h>

#include "dune_network.h"

namespace open_spiel {

class DuneNNEvaluator : public algorithms::Evaluator {
 public:
  inline static std::atomic<double> global_abs_leaf_value_sum{0.0};
  inline static std::atomic<uint64_t> global_num_leaf_evaluations{0};

  static void RecordLeafValue(double val) {
    double current = global_abs_leaf_value_sum.load(std::memory_order_relaxed);
    while (!global_abs_leaf_value_sum.compare_exchange_weak(
        current, current + std::abs(val),
        std::memory_order_relaxed, std::memory_order_relaxed)) {}
    global_num_leaf_evaluations.fetch_add(1, std::memory_order_relaxed);
  }

  DuneNNEvaluator(
      std::shared_ptr<SharedDunePolicyValueNetImpl> model,
      torch::Device device,
      double value_scale = 1.0,
      float logit_cap = 10.0f)
      : model_(model),
        device_(device),
        value_scale_(value_scale),
        logit_cap_(logit_cap) {
    model_->eval(); // Guard against BatchNorm/Dropout updates

    // Dynamically retrieve the expected observation input size from the model
    obs_size_ = model_->input_layer->weight.size(1);
  }

  std::vector<double> Evaluate(const State& state) override {
    torch::InferenceMode guard; // Fast ungrad inference
    AutocastGuard autocast_guard(device_.type(), device_.is_cuda()); // Match training precision

    int num_players = state.NumPlayers();
    std::vector<double> values(num_players, 0.0);

    // Pre-allocate a CPU tensor locally to ensure thread-safety
    auto options = torch::TensorOptions().dtype(torch::kFloat32);
    if (device_.is_cuda()) {
      options = options.pinned_memory(true);
    }
    torch::Tensor input_tensor = torch::zeros({num_players, obs_size_}, options);

    // Note: IS-MCTS normally handles chance states recursively without calling
    // Evaluate() on them. However, if defensively called on a chance node,
    // state.CurrentPlayer() will return kChancePlayerId (-1). The batched
    // evaluation below safely processes player 0..3 observations, avoiding crashes.

    // Stack all players' observations
    for (int p = 0; p < num_players; ++p) {
      std::vector<float> obs = state.InformationStateTensor(p);
      CheckObsSize(obs.size());
      
      std::memcpy(input_tensor.data_ptr<float>() + p * obs_size_, obs.data(), std::min<size_t>(obs.size(), obs_size_) * sizeof(float));
    }
    
    torch::Tensor device_tensor = device_.is_cuda() ? input_tensor.to(device_, /*non_blocking=*/true) : input_tensor;
    auto outputs = model_->forward(device_tensor);
    
    torch::Tensor values_cpu = outputs.values.to(torch::kCPU).to(torch::kDouble).contiguous();
    const double* values_data = values_cpu.data_ptr<double>();
    for (int p = 0; p < num_players; ++p) {
      double val = values_data[p] * value_scale_;
      values[p] = val;
      RecordLeafValue(val);
    }
    return values;
  }

  ActionsAndProbs Prior(const State& state) override {
    torch::InferenceMode guard;
    AutocastGuard autocast_guard(device_.type(), device_.is_cuda());

    if (state.IsTerminal()) {
      return {};
    }
    Player current_player = state.CurrentPlayer();
    if (current_player < 0 || current_player >= state.NumPlayers()) {
      return {};
    }

    std::vector<float> obs = state.InformationStateTensor(current_player);
    CheckObsSize(obs.size());

    // Pre-allocate a CPU tensor locally to ensure thread-safety
    auto options = torch::TensorOptions().dtype(torch::kFloat32);
    if (device_.is_cuda()) {
      options = options.pinned_memory(true);
    }
    torch::Tensor input_tensor = torch::zeros({1, obs_size_}, options);

    std::memcpy(input_tensor.data_ptr<float>(), obs.data(), std::min<size_t>(obs.size(), obs_size_) * sizeof(float));
    torch::Tensor device_tensor = device_.is_cuda() ? input_tensor.to(device_, /*non_blocking=*/true) : input_tensor;

    auto outputs = model_->forward(device_tensor);

    std::vector<Action> legal_actions = state.LegalActions();
    if (legal_actions.empty()) {
      return {};
    }

    // Cast AMP FP16 outputs back to Float32 and move to CPU host
    torch::Tensor logits_tensor = outputs.logits.squeeze(0).to(torch::kFloat32).to(torch::kCPU);
    float* logits_ptr = logits_tensor.data_ptr<float>();
    int64_t action_dim = logits_tensor.size(0);
    std::vector<float> logits(logits_ptr, logits_ptr + action_dim);

    CenterAndCapLegalLogits(logits, legal_actions, logit_cap_);

    // CPU-based softmax over legal actions
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

 private:
  void CheckObsSize(size_t size) const {
    SPIEL_CHECK_TRUE(size == static_cast<size_t>(obs_size_) || 
                     (size == 5580 && obs_size_ == 5584) ||
                     (size == 5584 && obs_size_ == 5580));
  }

  std::shared_ptr<SharedDunePolicyValueNetImpl> model_;
  torch::Device device_;
  double value_scale_;
  float logit_cap_;
  int64_t obs_size_;
};

} // namespace open_spiel
