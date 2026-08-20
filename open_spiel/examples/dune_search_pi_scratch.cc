#include "dune_search_pi_scratch.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <numeric>
#include <utility>

#include "open_spiel/spiel_utils.h"
#include "dune_ppo_training_utils.h"
#include "dune_seed_utils.h"

namespace open_spiel {
namespace {

bool IsTrunkParameter(const std::string& name) {
  return name.rfind("policy_head", 0) != 0 &&
         name.find("value_head") == std::string::npos;
}

void MeanSd(const torch::Tensor& values, double* mean, double* sd) {
  torch::Tensor x = values.detach().to(torch::kCPU).to(torch::kDouble).flatten();
  if (x.numel() == 0) {
    *mean = 0.0;
    *sd = 0.0;
    return;
  }
  *mean = x.mean().item<double>();
  *sd = x.numel() > 1
            ? std::sqrt((x - *mean).pow(2).mean().item<double>())
            : 0.0;
}

double SaturationRate(const torch::Tensor& values) {
  if (values.numel() == 0) return 0.0;
  return values.detach().abs().ge(0.99).to(torch::kDouble).mean().item<double>();
}

double GradNorm(const std::vector<torch::Tensor>& grads) {
  double sum_sq = 0.0;
  for (const torch::Tensor& grad : grads) {
    if (grad.defined()) sum_sq += grad.pow(2).sum().item<double>();
  }
  return std::sqrt(sum_sq);
}

}  // namespace

void CopySearchPiModel(
    const std::shared_ptr<SharedDunePolicyValueNetImpl>& source,
    const std::shared_ptr<SharedDunePolicyValueNetImpl>& destination) {
  SPIEL_CHECK_TRUE(source != nullptr);
  SPIEL_CHECK_TRUE(destination != nullptr);
  torch::NoGradGuard guard;
  auto source_params = source->named_parameters(true);
  auto destination_params = destination->named_parameters(true);
  SPIEL_CHECK_EQ(source_params.size(), destination_params.size());
  for (const auto& item : source_params) {
    SPIEL_CHECK_TRUE(destination_params.contains(item.key()));
    destination_params[item.key()].copy_(item.value());
  }
  auto source_buffers = source->named_buffers(true);
  auto destination_buffers = destination->named_buffers(true);
  SPIEL_CHECK_EQ(source_buffers.size(), destination_buffers.size());
  for (const auto& item : source_buffers) {
    SPIEL_CHECK_TRUE(destination_buffers.contains(item.key()));
    destination_buffers[item.key()].copy_(item.value());
  }
}

bool ScratchCollectorOwnedByOptimizer(
    const std::shared_ptr<SharedDunePolicyValueNetImpl>& collector,
    const torch::optim::Optimizer& optimizer) {
  if (collector == nullptr) return false;
  for (const auto& collector_param : collector->parameters()) {
    for (const auto& group : optimizer.param_groups()) {
      for (const torch::Tensor& optimized : group.params()) {
        if (collector_param.unsafeGetTensorImpl() ==
            optimized.unsafeGetTensorImpl()) {
          return true;
        }
      }
    }
  }
  return false;
}

ScratchSearchPiLearnerStats RunScratchSearchPiLearner(
    const std::shared_ptr<SharedDunePolicyValueNetImpl>& model,
    torch::optim::Optimizer& optimizer, const std::vector<SearchPiRow>& rows,
    int64_t obs_size, int64_t action_dim, torch::Device device,
    uint64_t master_seed, int generation, const SearchPiLearnerConfig& cfg) {
  ScratchSearchPiLearnerStats stats;
  stats.distinct_rows = static_cast<int64_t>(rows.size());
  if (rows.empty()) return stats;
  SPIEL_CHECK_TRUE(model != nullptr);
  SPIEL_CHECK_GT(cfg.minibatch_size, 0);
  SPIEL_CHECK_GT(cfg.epochs, 0);

  const int64_t n = static_cast<int64_t>(rows.size());
  auto f32 = torch::TensorOptions().dtype(torch::kFloat32);
  auto bool_options = torch::TensorOptions().dtype(torch::kBool);
  torch::Tensor states = torch::zeros({n, obs_size}, f32);
  torch::Tensor masks = torch::zeros({n, action_dim}, bool_options);
  torch::Tensor targets = torch::zeros({n, action_dim}, f32);
  torch::Tensor weights = torch::zeros({n}, f32);
  torch::Tensor values = torch::zeros({n}, f32);
  float* state_data = states.data_ptr<float>();
  bool* mask_data = masks.data_ptr<bool>();
  float* target_data = targets.data_ptr<float>();
  float* weight_data = weights.data_ptr<float>();
  float* value_data = values.data_ptr<float>();

  for (int64_t i = 0; i < n; ++i) {
    const SearchPiRow& row = rows[i];
    SPIEL_CHECK_EQ(row.row_schema_version, 2);
    SPIEL_CHECK_EQ(row.target_type, "regularized_q_kl");
    SPIEL_CHECK_TRUE(row.value_target_attached);
    SPIEL_CHECK_TRUE(std::isfinite(row.value_target));
    SPIEL_CHECK_TRUE(row.policy_target_weight == 0.0 ||
                     row.policy_target_weight == 1.0);
    if (row.policy_target_weight > 0.0) {
      SPIEL_CHECK_TRUE(row.regularized_q_valid);
      SPIEL_CHECK_EQ(row.regularized_q_target.size(), row.legal_actions.size());
      ++stats.policy_weight_rows;
    }
    const int64_t copy = std::min<int64_t>(
        obs_size, static_cast<int64_t>(row.observation.size()));
    if (copy > 0) {
      std::memcpy(state_data + i * obs_size, row.observation.data(),
                  copy * sizeof(float));
    }
    for (size_t j = 0; j < row.legal_actions.size(); ++j) {
      const Action action = row.legal_actions[j];
      SPIEL_CHECK_GE(action, 0);
      SPIEL_CHECK_LT(action, action_dim);
      mask_data[i * action_dim + action] = true;
      if (row.policy_target_weight > 0.0) {
        const double probability = row.regularized_q_target[j];
        SPIEL_CHECK_TRUE(std::isfinite(probability));
        SPIEL_CHECK_GE(probability, 0.0);
        target_data[i * action_dim + action] = static_cast<float>(probability);
      }
    }
    weight_data[i] = static_cast<float>(row.policy_target_weight);
    value_data[i] = static_cast<float>(row.value_target);
  }

  states = states.to(device);
  masks = masks.to(device);
  targets = targets.to(device);
  weights = weights.to(device);
  values = values.to(device);
  MeanSd(values, &stats.value_target_mean, &stats.value_target_sd);

  {
    torch::NoGradGuard guard;
    torch::Tensor prediction = model->forward(states).values.squeeze(1);
    MeanSd(prediction, &stats.critic_pred_mean_pre,
           &stats.critic_pred_sd_pre);
    stats.critic_saturation_pre = SaturationRate(prediction);
  }

  auto named_parameters = model->named_parameters(true);
  std::vector<torch::Tensor> parameters;
  std::vector<bool> trunk;
  parameters.reserve(named_parameters.size());
  trunk.reserve(named_parameters.size());
  for (const auto& item : named_parameters) {
    parameters.push_back(item.value());
    trunk.push_back(IsTrunkParameter(item.key()));
  }

  double ce_numerator = 0.0;
  double ce_denominator = 0.0;
  double mse_sum = 0.0;
  double policy_norm_sum = 0.0;
  double value_norm_sum = 0.0;
  double cosine_sum = 0.0;
  int64_t cosine_count = 0;

  model->train();
  for (int epoch = 0; epoch < cfg.epochs; ++epoch) {
    at::Generator generator = dune_seed::MakeTorchCPUGenerator(
        dune_seed::DeriveSeed(master_seed, dune_seed::kDomainTrain,
                              generation, epoch, uint64_t{0x5009}));
    torch::Tensor permutation =
        torch::randperm(n, generator,
                        torch::TensorOptions().dtype(torch::kInt64))
            .to(device);
    for (int64_t start = 0; start < n; start += cfg.minibatch_size) {
      const int64_t length =
          std::min<int64_t>(cfg.minibatch_size, n - start);
      torch::Tensor index = permutation.narrow(0, start, length);
      torch::Tensor mb_states = states.index_select(0, index);
      torch::Tensor mb_masks = masks.index_select(0, index);
      torch::Tensor mb_targets = targets.index_select(0, index);
      torch::Tensor mb_weights = weights.index_select(0, index);
      torch::Tensor mb_values = values.index_select(0, index);

      auto output = model->forward(mb_states);
      torch::Tensor logits = CenterAndCapLogitsTensor(
          output.logits, mb_masks, static_cast<float>(cfg.logit_cap));
      torch::Tensor log_probability = torch::log_softmax(
          logits.masked_fill(mb_masks.logical_not(), -1e9f), -1);
      torch::Tensor row_ce = -(mb_targets * log_probability).sum(-1);
      torch::Tensor weight_sum = mb_weights.sum();
      const double weight_sum_value = weight_sum.item<double>();
      const bool policy_active = cfg.policy_coef != 0.0 &&
                                 weight_sum_value > 0.0;
      torch::Tensor ce;
      if (policy_active) {
        ce = (row_ce * mb_weights).sum() / weight_sum;
      }
      torch::Tensor value_mse =
          (output.values.squeeze(1) - mb_values).pow(2).mean();
      const bool value_active = cfg.value_coef != 0.0;

      optimizer.zero_grad();
      std::vector<torch::Tensor> policy_grads(parameters.size());
      if (policy_active) {
        torch::Tensor policy_term = cfg.policy_coef == 1.0
                                        ? ce
                                        : cfg.policy_coef * ce;
        policy_term.backward(torch::Tensor(), value_active);
        for (size_t i = 0; i < parameters.size(); ++i) {
          if (parameters[i].grad().defined()) {
            policy_grads[i] = parameters[i].grad().detach().clone();
          }
        }
        optimizer.zero_grad();
        stats.policy_backward_executed = true;
        ++stats.policy_backward_minibatches;
        ce_numerator += ce.item<double>() * weight_sum_value;
        ce_denominator += weight_sum_value;
      }

      std::vector<torch::Tensor> value_grads(parameters.size());
      if (value_active) {
        torch::Tensor value_term = cfg.value_coef * value_mse;
        value_term.backward();
        for (size_t i = 0; i < parameters.size(); ++i) {
          if (parameters[i].grad().defined()) {
            value_grads[i] = parameters[i].grad().detach().clone();
          }
        }
        optimizer.zero_grad();
        stats.value_backward_executed = true;
        ++stats.value_backward_minibatches;
      }

      policy_norm_sum += GradNorm(policy_grads);
      value_norm_sum += GradNorm(value_grads);
      double dot = 0.0;
      double policy_sq = 0.0;
      double value_sq = 0.0;
      for (size_t i = 0; i < parameters.size(); ++i) {
        if (!trunk[i] || !policy_grads[i].defined() ||
            !value_grads[i].defined()) {
          continue;
        }
        dot += (policy_grads[i] * value_grads[i]).sum().item<double>();
        policy_sq += policy_grads[i].pow(2).sum().item<double>();
        value_sq += value_grads[i].pow(2).sum().item<double>();
      }
      if (policy_sq > 0.0 && value_sq > 0.0) {
        cosine_sum += dot / (std::sqrt(policy_sq) * std::sqrt(value_sq));
        ++cosine_count;
      }

      for (size_t i = 0; i < parameters.size(); ++i) {
        const bool have_policy = policy_grads[i].defined();
        const bool have_value = value_grads[i].defined();
        if (have_policy && have_value) {
          parameters[i].mutable_grad() = policy_grads[i] + value_grads[i];
        } else if (have_policy) {
          parameters[i].mutable_grad() = policy_grads[i];
        } else if (have_value) {
          parameters[i].mutable_grad() = value_grads[i];
        }
      }
      if (cfg.grad_clip_norm > 0.0) {
        torch::nn::utils::clip_grad_norm_(model->parameters(),
                                          cfg.grad_clip_norm);
      }
      if (policy_active || value_active) optimizer.step();
      mse_sum += value_mse.item<double>();
      stats.presentations += length;
      ++stats.minibatches;
    }
  }

  if (ce_denominator > 0.0) stats.policy_ce = ce_numerator / ce_denominator;
  if (stats.minibatches > 0) {
    stats.value_mse = mse_sum / stats.minibatches;
    stats.policy_grad_norm = policy_norm_sum / stats.minibatches;
    stats.value_grad_norm = value_norm_sum / stats.minibatches;
  }
  if (cosine_count > 0) {
    stats.shared_trunk_cosine = cosine_sum / cosine_count;
    stats.shared_trunk_cosine_defined = true;
  }

  model->eval();
  {
    torch::NoGradGuard guard;
    torch::Tensor prediction = model->forward(states).values.squeeze(1);
    MeanSd(prediction, &stats.critic_pred_mean_post,
           &stats.critic_pred_sd_post);
    stats.critic_saturation_post = SaturationRate(prediction);
  }
  auto finite = [](double value) { return std::isfinite(value); };
  SPIEL_CHECK_TRUE(finite(stats.policy_ce));
  SPIEL_CHECK_TRUE(finite(stats.value_mse));
  SPIEL_CHECK_TRUE(finite(stats.critic_pred_mean_post));
  SPIEL_CHECK_TRUE(finite(stats.critic_pred_sd_post));
  return stats;
}

}  // namespace open_spiel
