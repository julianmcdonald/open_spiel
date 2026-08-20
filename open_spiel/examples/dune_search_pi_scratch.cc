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

RegularizedQTargetResult BuildNormalizedCmpoTarget(
    const std::vector<Action>& legal_actions,
    const std::vector<double>& raw_prior, const std::vector<int>& visits,
    const std::vector<double>& q_values, double root_value, double sigma,
    double max_kl) {
  RegularizedQTargetResult out;
  auto fail = [&out](const char* name) {
    out.error = name;
    return out;
  };
  const size_t n = legal_actions.size();
  if (n == 0 || raw_prior.size() != n || visits.size() != n ||
      q_values.size() != n) {
    return fail("vector_alignment");
  }
  if (!std::isfinite(root_value) || !std::isfinite(sigma) ||
      !(sigma > 0.0) || !std::isfinite(max_kl) || !(max_kl >= 0.0)) {
    return fail("invalid_normalization_parameters");
  }

  constexpr double kPriorFloor = 1e-12;
  std::vector<double> prior(n);
  std::vector<double> completed_q(n);
  double prior_sum = 0.0;
  for (size_t i = 0; i < n; ++i) {
    if (!std::isfinite(raw_prior[i]) || raw_prior[i] < 0.0) {
      return fail("invalid_prior");
    }
    if (visits[i] < 0 || !std::isfinite(q_values[i])) {
      return fail("invalid_q_inputs");
    }
    prior[i] = std::max(kPriorFloor, raw_prior[i]);
    prior_sum += prior[i];
    completed_q[i] = visits[i] > 0 ? q_values[i] : root_value;
    if (visits[i] > 0) ++out.direct_visit_count;
  }
  if (!(prior_sum > 0.0) || !std::isfinite(prior_sum)) {
    return fail("invalid_prior_sum");
  }
  for (double& value : prior) value /= prior_sum;

  double baseline = 0.0;
  for (size_t i = 0; i < n; ++i) baseline += prior[i] * completed_q[i];
  if (!std::isfinite(baseline)) return fail("nonfinite_baseline");
  out.prior_expected_q = baseline;
  const auto mm = std::minmax_element(completed_q.begin(), completed_q.end());
  out.q_range = *mm.second - *mm.first;
  out.normalized_sigma = sigma;

  std::vector<double> z(n);
  int clipped_low = 0;
  int clipped_high = 0;
  for (size_t i = 0; i < n; ++i) {
    const double unbounded = (completed_q[i] - baseline) / sigma;
    if (!std::isfinite(unbounded)) return fail("nonfinite_normalized_advantage");
    if (unbounded <= -1.0) ++clipped_low;
    if (unbounded >= 1.0) ++clipped_high;
    z[i] = std::clamp(unbounded, -1.0, 1.0);
  }
  out.normalized_clip_low_fraction =
      static_cast<double>(clipped_low) / static_cast<double>(n);
  out.normalized_clip_high_fraction =
      static_cast<double>(clipped_high) / static_cast<double>(n);

  auto make_target = [&](double scale, std::vector<double>* target,
                         double* kl) -> bool {
    target->resize(n);
    double max_log_weight = -std::numeric_limits<double>::infinity();
    for (size_t i = 0; i < n; ++i) {
      const double log_weight = std::log(prior[i]) + scale * z[i];
      (*target)[i] = log_weight;
      max_log_weight = std::max(max_log_weight, log_weight);
    }
    double normalizer = 0.0;
    for (double& log_weight : *target) {
      log_weight = std::exp(log_weight - max_log_weight);
      normalizer += log_weight;
    }
    if (!(normalizer > 0.0) || !std::isfinite(normalizer)) return false;
    *kl = 0.0;
    for (size_t i = 0; i < n; ++i) {
      (*target)[i] /= normalizer;
      const double probability = (*target)[i];
      if (!std::isfinite(probability) || probability < 0.0) return false;
      if (probability > 0.0) {
        *kl += probability * std::log(probability / prior[i]);
      }
    }
    return std::isfinite(*kl);
  };

  std::vector<double> target;
  double kl = 0.0;
  if (!make_target(1.0, &target, &kl)) return fail("target_failure");
  double scale = 1.0;
  if (kl > max_kl) {
    double low = 0.0;
    double high = 1.0;
    for (int iteration = 0; iteration < 48; ++iteration) {
      const double mid = low + (high - low) * 0.5;
      std::vector<double> candidate;
      double candidate_kl = 0.0;
      if (!make_target(mid, &candidate, &candidate_kl)) {
        return fail("bisection_failure");
      }
      if (candidate_kl <= max_kl) {
        low = mid;
      } else {
        high = mid;
      }
    }
    scale = low;
    if (!make_target(scale, &target, &kl)) return fail("target_failure");
  }

  double target_sum = 0.0;
  double target_expected_q = 0.0;
  for (size_t i = 0; i < n; ++i) {
    target_sum += target[i];
    target_expected_q += target[i] * completed_q[i];
  }
  if (!std::isfinite(target_sum) || std::abs(target_sum - 1.0) > 1e-12 ||
      kl > max_kl + 1e-12 || !std::isfinite(target_expected_q)) {
    return fail("target_postcondition");
  }
  out.ok = true;
  out.error = "none";
  out.target = std::move(target);
  out.normalized_scale = scale;
  out.beta = scale;
  out.kl = kl;
  out.target_expected_q = target_expected_q;
  return out;
}

bool ApplyNormalizedCmpoTargets(
    std::vector<SearchPiRow>* rows, NormalizedCmpoGenerationStats* stats,
    std::string* error) {
  if (error != nullptr) error->clear();
  if (rows == nullptr || stats == nullptr) {
    if (error != nullptr) *error = "null_normalized_cmpo_output";
    return false;
  }
  *stats = NormalizedCmpoGenerationStats();
  double variance_sum = 0.0;
  int64_t full_count = 0;
  for (const SearchPiRow& row : *rows) {
    if (row.search_budget_class == "cheap") {
      ++stats->cheap_rows;
      continue;
    }
    if (row.search_budget_class != "full") {
      if (error != nullptr) *error = "unknown_budget_class";
      return false;
    }
    ++stats->full_rows;
    const size_t n = row.legal_actions.size();
    if (n == 0 || row.raw_policy.size() != n || row.raw_visits.size() != n ||
        row.q_values.size() != n || !std::isfinite(row.root_value)) {
      if (error != nullptr) *error = "unaligned_full_row";
      return false;
    }
    std::vector<double> prior(n);
    double prior_sum = 0.0;
    for (size_t i = 0; i < n; ++i) {
      if (!std::isfinite(row.raw_policy[i]) || row.raw_policy[i] < 0.0 ||
          row.raw_visits[i] < 0 || !std::isfinite(row.q_values[i])) {
        if (error != nullptr) *error = "invalid_full_row_inputs";
        return false;
      }
      prior[i] = std::max(1e-12, row.raw_policy[i]);
      prior_sum += prior[i];
    }
    if (!(prior_sum > 0.0) || !std::isfinite(prior_sum)) {
      if (error != nullptr) *error = "invalid_full_row_prior";
      return false;
    }
    for (double& value : prior) value /= prior_sum;
    double baseline = 0.0;
    for (size_t i = 0; i < n; ++i) {
      const double q = row.raw_visits[i] > 0 ? row.q_values[i]
                                             : row.root_value;
      baseline += prior[i] * q;
    }
    double variance = 0.0;
    for (size_t i = 0; i < n; ++i) {
      const double q = row.raw_visits[i] > 0 ? row.q_values[i]
                                             : row.root_value;
      variance += prior[i] * (q - baseline) * (q - baseline);
    }
    variance_sum += variance;
    ++full_count;
  }
  if (full_count > 0) {
    stats->sigma = std::sqrt(std::max(0.0, variance_sum /
                                               static_cast<double>(full_count)));
    if (!std::isfinite(stats->sigma) || stats->sigma < 1e-6) {
      stats->sigma = 1e-6;
    }
  }

  double prior_sum = 0.0;
  double target_sum = 0.0;
  double q_range_sum = 0.0;
  double direct_visits_sum = 0.0;
  double raw_entropy_sum = 0.0;
  double target_entropy_sum = 0.0;
  for (SearchPiRow& row : *rows) {
    auto result = BuildNormalizedCmpoTarget(
        row.legal_actions, row.raw_policy, row.raw_visits, row.q_values,
        row.root_value, stats->sigma);
    if (!result.ok) {
      if (error != nullptr) *error = result.error;
      return false;
    }
    row.target_type = "normalized_cmpo";
    row.regularized_q_valid = true;
    row.regularized_q_error = "none";
    row.regularized_q_target = std::move(result.target);
    row.regularized_q_beta = result.normalized_scale;
    row.regularized_q_kl = result.kl;
    row.regularized_q_prior_expected_q = result.prior_expected_q;
    row.regularized_q_target_expected_q = result.target_expected_q;
    row.regularized_q_q_range = result.q_range;
    row.regularized_q_direct_visit_count = result.direct_visit_count;
    row.regularized_q_entropy_norm = NormalizedEntropy(row.regularized_q_target);
    if (row.search_budget_class == "full") {
      stats->advantage_count += static_cast<int64_t>(row.legal_actions.size());
      stats->clipped_low += static_cast<int64_t>(
          std::round(result.normalized_clip_low_fraction * row.legal_actions.size()));
      stats->clipped_high += static_cast<int64_t>(
          std::round(result.normalized_clip_high_fraction * row.legal_actions.size()));
    }
    if (row.search_budget_class == "full") {
      stats->target_kls.push_back(result.kl);
      prior_sum += result.prior_expected_q;
      target_sum += result.target_expected_q;
      q_range_sum += result.q_range;
      direct_visits_sum += result.direct_visit_count;
      raw_entropy_sum += NormalizedEntropy(row.raw_policy);
      target_entropy_sum += row.regularized_q_entropy_norm;
    }
  }
  const double denominator =
      std::max<int64_t>(1, stats->full_rows);
  stats->prior_expected_q_mean = prior_sum / denominator;
  stats->target_expected_q_mean = target_sum / denominator;
  stats->q_range_mean = q_range_sum / denominator;
  stats->direct_visit_count_mean = direct_visits_sum / denominator;
  stats->raw_entropy_mean = raw_entropy_sum / denominator;
  stats->target_entropy_mean = target_entropy_sum / denominator;
  return true;
}

std::vector<std::vector<SearchPiRow>> FilterNormalizedCmpoReplayWindow(
    const std::vector<std::vector<SearchPiRow>>& replay_window) {
  std::vector<std::vector<SearchPiRow>> full_only(replay_window.size());
  for (size_t cohort = 0; cohort < replay_window.size(); ++cohort) {
    for (const SearchPiRow& row : replay_window[cohort]) {
      if (row.search_budget_class == "full" &&
          row.policy_target_weight > 0.0) {
        full_only[cohort].push_back(row);
      }
    }
  }
  return full_only;
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
    SPIEL_CHECK_TRUE(row.target_type == "regularized_q_kl" ||
                     row.target_type == "normalized_cmpo");
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
