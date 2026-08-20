#include "dune_offline_changed_state.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <limits>
#include <map>
#include <numeric>
#include <set>
#include <sstream>
#include <tuple>
#include <utility>

#include "open_spiel/spiel_utils.h"

#include "dune_ppo_training_utils.h"
#include "dune_sha256.h"

namespace open_spiel {
namespace {

constexpr double kZ95 = 1.959963984540054;

std::string ShapeString(const torch::Tensor& tensor) {
  std::ostringstream out;
  out << '[';
  for (int64_t i = 0; i < tensor.dim(); ++i) {
    if (i) out << ',';
    out << tensor.size(i);
  }
  return out.str() + ']';
}

void SetError(std::string* error, const std::string& value) {
  if (error != nullptr) *error = value;
}

int FindSpec(const std::vector<OfflineRoleSpec>& specs,
             DuneDecisionRole role) {
  for (size_t i = 0; i < specs.size(); ++i) {
    if (specs[i].role == role) return static_cast<int>(i);
  }
  return -1;
}

struct BatchTensors {
  torch::Tensor states;
  torch::Tensor masks;
  torch::Tensor targets;
  torch::Tensor weights;
};

BatchTensors MakeBatch(const std::vector<SearchPiRow>& rows,
                       const std::vector<double>* weights, int64_t start,
                       int64_t count, int64_t obs_size, int64_t action_dim,
                       torch::Device device) {
  const auto f32 = torch::TensorOptions().dtype(torch::kFloat32);
  const auto boolopt = torch::TensorOptions().dtype(torch::kBool);
  BatchTensors batch;
  batch.states = torch::zeros({count, obs_size}, f32);
  batch.masks = torch::zeros({count, action_dim}, boolopt);
  batch.targets = torch::zeros({count, action_dim}, f32);
  if (weights != nullptr) batch.weights = torch::zeros({count}, f32);
  float* states_ptr = batch.states.data_ptr<float>();
  bool* masks_ptr = batch.masks.data_ptr<bool>();
  float* targets_ptr = batch.targets.data_ptr<float>();
  float* weights_ptr = weights == nullptr ? nullptr
                                          : batch.weights.data_ptr<float>();
  for (int64_t bi = 0; bi < count; ++bi) {
    const int64_t row_index = start + bi;
    SPIEL_CHECK_GE(row_index, 0);
    SPIEL_CHECK_LT(row_index, static_cast<int64_t>(rows.size()));
    const SearchPiRow& row = rows[static_cast<size_t>(row_index)];
    if (row.observation.empty() ||
        static_cast<int64_t>(row.observation.size()) > obs_size ||
        row.legal_actions.empty() ||
        row.legal_actions.size() != row.target_probs.size()) {
      SpielFatalError("malformed offline row at presentation index " +
                      std::to_string(row_index));
    }
    std::copy(row.observation.begin(), row.observation.end(),
              states_ptr + bi * obs_size);
    double target_mass = 0.0;
    for (size_t j = 0; j < row.legal_actions.size(); ++j) {
      const Action action = row.legal_actions[j];
      if (action < 0 || action >= action_dim) {
        SpielFatalError("out-of-range legal action in offline row " +
                        std::to_string(row_index));
      }
      const double probability = row.target_probs[j];
      if (!std::isfinite(probability) || probability < 0.0) {
        SpielFatalError("invalid target probability in offline row " +
                        std::to_string(row_index));
      }
      masks_ptr[bi * action_dim + action] = true;
      targets_ptr[bi * action_dim + action] =
          static_cast<float>(probability);
      target_mass += probability;
    }
    if (std::abs(target_mass - 1.0) > 1e-9) {
      SpielFatalError("target mass is not one in offline row " +
                      std::to_string(row_index));
    }
    if (weights_ptr != nullptr) {
      const double weight = (*weights)[static_cast<size_t>(row_index)];
      if (!std::isfinite(weight) || weight <= 0.0) {
        SpielFatalError("invalid offline row weight at presentation index " +
                        std::to_string(row_index));
      }
      weights_ptr[bi] = static_cast<float>(weight);
    }
  }
  batch.states = batch.states.to(device);
  batch.masks = batch.masks.to(device);
  batch.targets = batch.targets.to(device);
  if (weights != nullptr) batch.weights = batch.weights.to(device);
  return batch;
}

OfflineMeanInterval MeanInterval(const std::vector<double>& values) {
  OfflineMeanInterval out;
  out.n = static_cast<int64_t>(values.size());
  if (values.empty()) return out;
  const long double sum = std::accumulate(
      values.begin(), values.end(), static_cast<long double>(0.0));
  out.mean = static_cast<double>(sum / values.size());
  if (values.size() > 1) {
    long double squared = 0.0;
    for (double value : values) {
      const long double delta = value - out.mean;
      squared += delta * delta;
    }
    out.sd = std::sqrt(static_cast<double>(
        squared / static_cast<long double>(values.size() - 1)));
    out.se = out.sd / std::sqrt(static_cast<double>(values.size()));
  }
  out.lower = out.mean - kZ95 * out.se;
  out.upper = out.mean + kZ95 * out.se;
  return out;
}

template <typename Predicate>
OfflineMetricLevel MetricLevel(const std::vector<SearchPiRow>& rows,
                               const OfflineModelMetrics& metrics,
                               Predicate include) {
  OfflineMetricLevel out;
  long double ce_sum = 0.0;
  for (size_t i = 0; i < rows.size(); ++i) {
    if (!include(rows[i])) continue;
    ++out.n;
    ce_sum += metrics.ce[i];
    out.adopted += metrics.adopted[i] != 0;
  }
  if (out.n > 0) {
    out.ce_mean = static_cast<double>(ce_sum / out.n);
    out.adoption = static_cast<double>(out.adopted) / out.n;
  }
  return out;
}

template <typename Predicate>
OfflinePairedComparison PairedComparison(
    const std::vector<SearchPiRow>& rows,
    const OfflineModelMetrics& control,
    const OfflineModelMetrics& treatment, Predicate include) {
  OfflinePairedComparison out;
  std::vector<double> ce_delta;
  std::vector<double> adoption_delta;
  ce_delta.reserve(rows.size());
  adoption_delta.reserve(rows.size());
  for (size_t i = 0; i < rows.size(); ++i) {
    if (!include(rows[i])) continue;
    ce_delta.push_back(treatment.ce[i] - control.ce[i]);
    const int c = control.adopted[i] != 0;
    const int t = treatment.adopted[i] != 0;
    adoption_delta.push_back(static_cast<double>(t - c));
    out.adoption_gained += t && !c;
    out.adoption_lost += c && !t;
  }
  out.adoption_discordant = out.adoption_gained + out.adoption_lost;
  out.ce_t_minus_c = MeanInterval(ce_delta);
  out.adoption_t_minus_c = MeanInterval(adoption_delta);
  out.adoption_one_sided_exact_p =
      OneSidedPairedExactP(out.adoption_gained, out.adoption_lost);
  return out;
}

}  // namespace

const std::array<OfflineRoleSpec, 5>& RegisteredOfflineRoleSpecs() {
  static const std::array<OfflineRoleSpec, 5> specs = {{
      {DuneDecisionRole::kAgentContinuation, "agent_continuation", 39522,
       3790, 35732},
      {DuneDecisionRole::kAgentPrimary, "agent_primary", 16233, 2746, 13487},
      {DuneDecisionRole::kCombatIntrigue, "combat_intrigue", 10852, 630,
       10222},
      {DuneDecisionRole::kOtherOptional, "other_optional", 17415, 1274,
       16141},
      {DuneDecisionRole::kPurchase, "purchase", 8357, 1280, 7077},
  }};
  return specs;
}

const char* OfflineRoleName(DuneDecisionRole role) {
  switch (role) {
    case DuneDecisionRole::kForcedOrBookkeeping:
      return "forced_or_bookkeeping";
    case DuneDecisionRole::kLeaderSelection:
      return "leader_selection";
    case DuneDecisionRole::kAgentPrimary:
      return "agent_primary";
    case DuneDecisionRole::kAgentContinuation:
      return "agent_continuation";
    case DuneDecisionRole::kPurchase:
      return "purchase";
    case DuneDecisionRole::kCombatIntrigue:
      return "combat_intrigue";
    case DuneDecisionRole::kOtherOptional:
      return "other_optional";
  }
  return "invalid";
}

bool BuildOfflineWeightPlan(const std::vector<SearchPiRow>& rows,
                            const std::vector<OfflineRoleSpec>& specs,
                            OfflineWeightPlan* out, std::string* error,
                            double treatment_lambda) {
  if (out == nullptr) {
    SetError(error, "weight plan output is null");
    return false;
  }
  *out = OfflineWeightPlan();
  if (specs.empty()) {
    SetError(error, "role specifications are empty");
    return false;
  }
  if (!std::isfinite(treatment_lambda) || treatment_lambda < 0.0 ||
      treatment_lambda > 1.0) {
    SetError(error, "treatment lambda must be finite and in [0, 1]");
    return false;
  }
  out->treatment_lambda = treatment_lambda;
  std::set<int> roles;
  int64_t expected_rows = 0;
  out->roles.reserve(specs.size());
  for (const OfflineRoleSpec& spec : specs) {
    if (!roles.insert(static_cast<int>(spec.role)).second) {
      SetError(error, "duplicate role specification");
      return false;
    }
    if (spec.total <= 0 || spec.changed <= 0 || spec.preserved <= 0 ||
        spec.changed + spec.preserved != spec.total) {
      SetError(error, std::string("invalid role specification for ") +
                          spec.name);
      return false;
    }
    expected_rows += spec.total;
    OfflineRoleWeightSummary summary;
    summary.spec = spec;
    summary.balanced_changed_weight =
        static_cast<double>(spec.total) / (2.0 * spec.changed);
    summary.balanced_preserved_weight =
        static_cast<double>(spec.total) / (2.0 * spec.preserved);
    summary.changed_weight =
        1.0 + treatment_lambda * (summary.balanced_changed_weight - 1.0);
    summary.preserved_weight =
        1.0 + treatment_lambda * (summary.balanced_preserved_weight - 1.0);
    out->roles.push_back(summary);
  }
  if (expected_rows != static_cast<int64_t>(rows.size())) {
    SetError(error, "parsed row count does not equal registered role total");
    return false;
  }

  out->control.assign(rows.size(), 1.0);
  out->treatment.resize(rows.size());
  for (size_t i = 0; i < rows.size(); ++i) {
    const int spec_index = FindSpec(specs, rows[i].role);
    if (spec_index < 0) {
      SetError(error, std::string("unregistered role at row ") +
                          std::to_string(i) + ": " +
                          OfflineRoleName(rows[i].role));
      return false;
    }
    OfflineRoleWeightSummary& summary = out->roles[spec_index];
    ++summary.parsed_total;
    if (rows[i].target_argmax_differs_from_raw) {
      ++summary.parsed_changed;
      out->treatment[i] = summary.changed_weight;
    } else {
      ++summary.parsed_preserved;
      out->treatment[i] = summary.preserved_weight;
    }
  }

  out->control_mass = static_cast<double>(rows.size());
  double role_mass = 0.0;
  for (OfflineRoleWeightSummary& summary : out->roles) {
    if (summary.parsed_total != summary.spec.total ||
        summary.parsed_changed != summary.spec.changed ||
        summary.parsed_preserved != summary.spec.preserved) {
      std::ostringstream message;
      message << "role-count mismatch for " << summary.spec.name
              << ": parsed " << summary.parsed_total << '/'
              << summary.parsed_changed << '/' << summary.parsed_preserved
              << " expected " << summary.spec.total << '/'
              << summary.spec.changed << '/' << summary.spec.preserved;
      SetError(error, message.str());
      return false;
    }
    summary.changed_loss_mass =
        summary.changed_weight * summary.parsed_changed;
    summary.preserved_loss_mass =
        summary.preserved_weight * summary.parsed_preserved;
    summary.total_loss_mass =
        summary.changed_loss_mass + summary.preserved_loss_mass;
    const double expected_changed_mass =
        (1.0 - treatment_lambda) * summary.spec.changed +
        treatment_lambda * summary.spec.total / 2.0;
    const double expected_preserved_mass =
        (1.0 - treatment_lambda) * summary.spec.preserved +
        treatment_lambda * summary.spec.total / 2.0;
    if (std::abs(summary.changed_loss_mass - expected_changed_mass) > 1e-8 ||
        std::abs(summary.preserved_loss_mass - expected_preserved_mass) >
            1e-8 ||
        std::abs(summary.total_loss_mass - summary.spec.total) > 1e-8) {
      SetError(error, std::string("loss-mass mismatch for ") +
                          summary.spec.name);
      return false;
    }
    role_mass += summary.total_loss_mass;
  }
  const long double row_mass = std::accumulate(
      out->treatment.begin(), out->treatment.end(),
      static_cast<long double>(0.0));
  out->treatment_mass = role_mass;
  out->extended_precision_sequential_treatment_mass =
      static_cast<double>(row_mass);
  out->binary64_sequential_treatment_mass =
      std::accumulate(out->treatment.begin(), out->treatment.end(), 0.0);
  if (std::abs(row_mass - static_cast<long double>(role_mass)) > 1e-10L ||
      std::abs(row_mass - static_cast<long double>(rows.size())) > 1e-10L) {
    SetError(error, "treatment corpus mean weight is not exactly one");
    return false;
  }
  return true;
}

std::vector<OfflineBatchBoundary> BuildOfflineBatchBoundaries(
    int64_t row_count, int64_t batch_size) {
  SPIEL_CHECK_GE(row_count, 0);
  SPIEL_CHECK_GT(batch_size, 0);
  std::vector<OfflineBatchBoundary> out;
  for (int64_t start = 0; start < row_count; start += batch_size) {
    out.push_back({start, std::min(batch_size, row_count - start)});
  }
  return out;
}

std::string OfflinePresentationDigest(
    const std::string& extended_row_hash,
    const std::vector<OfflineBatchBoundary>& boundaries, uint64_t seed) {
  std::ostringstream canonical;
  canonical << "offline_changed_state_presentation_v1\n"
            << "extended_row_hash=" << extended_row_hash << '\n'
            << "seed=" << seed << '\n';
  for (const OfflineBatchBoundary& boundary : boundaries) {
    canonical << boundary.start << '\t' << boundary.count << '\n';
  }
  return ComputeStringSHA256(canonical.str());
}

std::string CanonicalOfflineModuleDigest(torch::nn::Module& model) {
  struct Entry {
    std::string category;
    std::string name;
    torch::Tensor tensor;
  };
  std::vector<Entry> entries;
  for (const auto& item : model.named_parameters(/*recurse=*/true)) {
    entries.push_back({"parameter", item.key(), item.value()});
  }
  for (const auto& item : model.named_buffers(/*recurse=*/true)) {
    entries.push_back({"buffer", item.key(), item.value()});
  }
  std::sort(entries.begin(), entries.end(), [](const Entry& lhs,
                                                const Entry& rhs) {
    return std::tie(lhs.category, lhs.name) <
           std::tie(rhs.category, rhs.name);
  });
  std::string canonical = "search_pi_module_digest_v1\n";
  for (const Entry& entry : entries) {
    torch::Tensor tensor = entry.tensor.detach().to(torch::kCPU).contiguous();
    canonical += entry.category + "\t" + entry.name + "\t" +
                 ShapeString(tensor) + "\t" +
                 std::string(c10::toString(tensor.scalar_type())) + "\t" +
                 std::to_string(tensor.numel() * tensor.element_size()) +
                 "\n";
    const size_t bytes =
        static_cast<size_t>(tensor.numel()) * tensor.element_size();
    canonical.append(static_cast<const char*>(tensor.data_ptr()), bytes);
    canonical.push_back('\n');
  }
  return ComputeStringSHA256(canonical);
}

std::unique_ptr<torch::optim::AdamW> MakeOfflineFreshAdamW(
    const std::shared_ptr<SharedDunePolicyValueNetImpl>& model,
    double learning_rate, double epsilon, double policy_weight_decay,
    double other_weight_decay) {
  SPIEL_CHECK_TRUE(model != nullptr);
  std::vector<torch::Tensor> policy_params;
  std::vector<torch::Tensor> other_params;
  const std::vector<torch::Tensor> policy_set = model->policy_head->parameters();
  for (torch::Tensor& parameter : model->parameters()) {
    bool is_policy = false;
    for (const torch::Tensor& candidate : policy_set) {
      if (parameter.is_same(candidate)) {
        is_policy = true;
        break;
      }
    }
    (is_policy ? policy_params : other_params).push_back(parameter);
  }
  SPIEL_CHECK_FALSE(policy_params.empty());
  SPIEL_CHECK_FALSE(other_params.empty());
  std::vector<torch::optim::OptimizerParamGroup> groups;
  groups.emplace_back(policy_params);
  groups.emplace_back(other_params);
  auto optimizer = std::make_unique<torch::optim::AdamW>(
      groups, torch::optim::AdamWOptions(learning_rate).eps(epsilon));
  static_cast<torch::optim::AdamWOptions&>(
      optimizer->param_groups()[0].options())
      .weight_decay(policy_weight_decay);
  static_cast<torch::optim::AdamWOptions&>(
      optimizer->param_groups()[1].options())
      .weight_decay(other_weight_decay);
  return optimizer;
}

OfflineAdamWContract CaptureOfflineAdamWContract(
    const SharedDunePolicyValueNetImpl& model,
    const torch::optim::AdamW& optimizer) {
  OfflineAdamWContract out;
  auto& mutable_model = const_cast<SharedDunePolicyValueNetImpl&>(model);
  const auto named = mutable_model.named_parameters(/*recurse=*/true);
  for (const torch::optim::OptimizerParamGroup& group :
       optimizer.param_groups()) {
    OfflineAdamWGroupContract captured;
    for (const torch::Tensor& parameter : group.params()) {
      std::string name;
      for (const auto& item : named) {
        if (parameter.is_same(item.value())) {
          name = item.key();
          break;
        }
      }
      if (name.empty()) {
        SpielFatalError("offline optimizer contains an unnamed parameter");
      }
      captured.parameter_names.push_back(name);
    }
    const auto& options =
        static_cast<const torch::optim::AdamWOptions&>(group.options());
    captured.learning_rate = options.lr();
    captured.beta1 = std::get<0>(options.betas());
    captured.beta2 = std::get<1>(options.betas());
    captured.epsilon = options.eps();
    captured.weight_decay = options.weight_decay();
    captured.amsgrad = options.amsgrad();
    out.groups.push_back(std::move(captured));
  }
  return out;
}

bool OfflineAdamWContractsEqual(const OfflineAdamWContract& lhs,
                                const OfflineAdamWContract& rhs,
                                std::string* error) {
  if (lhs.groups.size() != rhs.groups.size()) {
    SetError(error, "AdamW parameter-group count differs between arms");
    return false;
  }
  for (size_t i = 0; i < lhs.groups.size(); ++i) {
    const OfflineAdamWGroupContract& a = lhs.groups[i];
    const OfflineAdamWGroupContract& b = rhs.groups[i];
    if (a.parameter_names != b.parameter_names) {
      SetError(error, "AdamW parameter membership/order differs in group " +
                          std::to_string(i));
      return false;
    }
    if (a.learning_rate != b.learning_rate || a.beta1 != b.beta1 ||
        a.beta2 != b.beta2 || a.epsilon != b.epsilon ||
        a.weight_decay != b.weight_decay || a.amsgrad != b.amsgrad) {
      SetError(error, "AdamW options differ in group " + std::to_string(i));
      return false;
    }
  }
  return true;
}

bool ValidateFreshOfflineBoundary(
    SharedDunePolicyValueNetImpl& control_model,
    const torch::optim::AdamW& control_optimizer,
    SharedDunePolicyValueNetImpl& treatment_model,
    const torch::optim::AdamW& treatment_optimizer,
    const std::string& expected_start_digest, std::string* error) {
  if (expected_start_digest.empty()) {
    SetError(error, "registered canonical start-model digest is empty");
    return false;
  }
  const std::string control_digest =
      CanonicalOfflineModuleDigest(control_model);
  const std::string treatment_digest =
      CanonicalOfflineModuleDigest(treatment_model);
  if (control_digest != treatment_digest ||
      control_digest != expected_start_digest) {
    SetError(error, "canonical start-model digest mismatch");
    return false;
  }
  if (!control_optimizer.state().empty() ||
      !treatment_optimizer.state().empty()) {
    SetError(error, "fresh AdamW state map is not empty");
    return false;
  }
  return OfflineAdamWContractsEqual(
      CaptureOfflineAdamWContract(control_model, control_optimizer),
      CaptureOfflineAdamWContract(treatment_model, treatment_optimizer),
      error);
}

bool RefuseOfflineOptimizerCheckpoint(const std::string& path,
                                      std::string* error) {
  if (!path.empty()) {
    SetError(error, "optimizer checkpoint loading is forbidden in the frozen "
                    "offline changed-state experiment");
    return false;
  }
  return true;
}

OfflineTrainingStats TrainOfflineChangedStateArm(
    const std::shared_ptr<SharedDunePolicyValueNetImpl>& model,
    torch::optim::AdamW& optimizer, const std::vector<SearchPiRow>& rows,
    const std::vector<double>& weights,
    const std::vector<OfflineBatchBoundary>& boundaries, int64_t obs_size,
    int64_t action_dim, torch::Device device, uint64_t seed,
    double gradient_clip, double logit_cap) {
  SPIEL_CHECK_TRUE(model != nullptr);
  SPIEL_CHECK_EQ(rows.size(), weights.size());
  const std::vector<OfflineBatchBoundary> expected_boundaries =
      BuildOfflineBatchBoundaries(rows.size(),
                                  kOfflineChangedStateBatchSize);
  SPIEL_CHECK_EQ(boundaries.size(), expected_boundaries.size());
  for (size_t i = 0; i < boundaries.size(); ++i) {
    SPIEL_CHECK_EQ(boundaries[i].start, expected_boundaries[i].start);
    SPIEL_CHECK_EQ(boundaries[i].count, expected_boundaries[i].count);
  }
  torch::manual_seed(seed);
  model->train();
  OfflineTrainingStats stats;
  for (const OfflineBatchBoundary& boundary : boundaries) {
    BatchTensors batch = MakeBatch(rows, &weights, boundary.start,
                                   boundary.count, obs_size, action_dim,
                                   device);
    auto outputs = model->forward(batch.states);
    torch::Tensor logits = CenterAndCapLogitsTensor(
        outputs.logits, batch.masks, static_cast<float>(logit_cap));
    torch::Tensor masked =
        logits.masked_fill(batch.masks.logical_not(), -1e9f);
    torch::Tensor per_row_ce =
        -(batch.targets * torch::log_softmax(masked, -1)).sum(-1);
    torch::Tensor numerator = (batch.weights * per_row_ce).sum();
    torch::Tensor denominator = batch.weights.sum();
    torch::Tensor loss = numerator / denominator;
    optimizer.zero_grad();
    loss.backward();
    if (gradient_clip > 0.0) {
      torch::nn::utils::clip_grad_norm_(model->parameters(), gradient_clip);
    }
    optimizer.step();

    stats.presentations += boundary.count;
    ++stats.minibatches;
    stats.final_minibatch_size = boundary.count;
    stats.weighted_ce_sum += numerator.item<double>();
  }
  stats.presentation_weight_mass = static_cast<double>(std::accumulate(
      weights.begin(), weights.end(), static_cast<long double>(0.0)));
  if (stats.presentation_weight_mass > 0.0) {
    stats.mean_presented_weighted_ce =
        stats.weighted_ce_sum / stats.presentation_weight_mass;
  }
  return stats;
}

OfflineModelMetrics EvaluateOfflineChangedStateRows(
    const std::shared_ptr<SharedDunePolicyValueNetImpl>& model,
    const std::vector<SearchPiRow>& rows,
    const std::vector<OfflineBatchBoundary>& boundaries, int64_t obs_size,
    int64_t action_dim, torch::Device device, double logit_cap) {
  SPIEL_CHECK_TRUE(model != nullptr);
  model->eval();
  torch::InferenceMode inference_guard;
  OfflineModelMetrics out;
  out.ce.reserve(rows.size());
  out.adopted.reserve(rows.size());
  for (const OfflineBatchBoundary& boundary : boundaries) {
    BatchTensors batch = MakeBatch(rows, nullptr, boundary.start,
                                   boundary.count, obs_size, action_dim,
                                   device);
    const auto outputs = model->forward(batch.states);
    const torch::Tensor centered = CenterAndCapLogitsTensor(
        outputs.logits, batch.masks, static_cast<float>(logit_cap));
    const torch::Tensor masked =
        centered.masked_fill(batch.masks.logical_not(), -1e9f);
    const torch::Tensor logp = torch::log_softmax(masked, -1);
    const torch::Tensor ce = -(batch.targets * logp).sum(-1);
    const torch::Tensor adopted =
        torch::argmax(masked, -1).eq(torch::argmax(batch.targets, -1));
    const torch::Tensor ce_cpu = ce.to(
        torch::TensorOptions().device(torch::kCPU).dtype(torch::kDouble));
    const torch::Tensor adopted_cpu = adopted.to(
        torch::TensorOptions().device(torch::kCPU).dtype(torch::kUInt8));
    const double* ce_ptr = ce_cpu.data_ptr<double>();
    const uint8_t* adopted_ptr = adopted_cpu.data_ptr<uint8_t>();
    for (int64_t bi = 0; bi < boundary.count; ++bi) {
      out.ce.push_back(ce_ptr[bi]);
      out.adopted.push_back(adopted_ptr[bi]);
    }
  }
  return out;
}

double OneSidedPairedExactP(int64_t gained, int64_t lost) {
  SPIEL_CHECK_GE(gained, 0);
  SPIEL_CHECK_GE(lost, 0);
  const int64_t n = gained + lost;
  if (n == 0) return 1.0;
  std::vector<long double> log_terms;
  log_terms.reserve(static_cast<size_t>(n - gained + 1));
  long double max_log = -std::numeric_limits<long double>::infinity();
  const long double log_two = std::log(static_cast<long double>(2.0));
  for (int64_t k = gained; k <= n; ++k) {
    const long double log_term =
        std::lgamma(static_cast<long double>(n + 1)) -
        std::lgamma(static_cast<long double>(k + 1)) -
        std::lgamma(static_cast<long double>(n - k + 1)) - n * log_two;
    log_terms.push_back(log_term);
    max_log = std::max(max_log, log_term);
  }
  long double scaled_sum = 0.0;
  for (long double value : log_terms) {
    scaled_sum += std::exp(value - max_log);
  }
  const long double probability = std::exp(max_log) * scaled_sum;
  return static_cast<double>(std::min(static_cast<long double>(1.0),
                                      probability));
}

OfflineGateResult ApplyOfflineChangedStateGate(
    const std::vector<SearchPiRow>& rows,
    const OfflineModelMetrics& parent,
    const OfflineModelMetrics& control_child,
    const OfflineModelMetrics& treatment_child,
    bool integrity_assertions_passed) {
  const size_t n = rows.size();
  SPIEL_CHECK_EQ(parent.ce.size(), n);
  SPIEL_CHECK_EQ(parent.adopted.size(), n);
  SPIEL_CHECK_EQ(control_child.ce.size(), n);
  SPIEL_CHECK_EQ(control_child.adopted.size(), n);
  SPIEL_CHECK_EQ(treatment_child.ce.size(), n);
  SPIEL_CHECK_EQ(treatment_child.adopted.size(), n);
  const auto all = [](const SearchPiRow&) { return true; };
  const auto changed = [](const SearchPiRow& row) {
    return row.target_argmax_differs_from_raw;
  };
  const auto preserved = [](const SearchPiRow& row) {
    return !row.target_argmax_differs_from_raw;
  };
  OfflineGateResult out;
  out.parent_all = MetricLevel(rows, parent, all);
  out.control_all = MetricLevel(rows, control_child, all);
  out.treatment_all = MetricLevel(rows, treatment_child, all);
  out.parent_changed = MetricLevel(rows, parent, changed);
  out.control_changed = MetricLevel(rows, control_child, changed);
  out.treatment_changed = MetricLevel(rows, treatment_child, changed);
  out.parent_preserved = MetricLevel(rows, parent, preserved);
  out.control_preserved = MetricLevel(rows, control_child, preserved);
  out.treatment_preserved = MetricLevel(rows, treatment_child, preserved);
  out.changed = PairedComparison(rows, control_child, treatment_child, changed);
  out.preserved =
      PairedComparison(rows, control_child, treatment_child, preserved);
  out.condition_1_changed_fit_and_adoption =
      out.changed.ce_t_minus_c.upper < 0.0 &&
      out.changed.adoption_one_sided_exact_p < 0.05;

  out.condition_2_all_roles_lower_changed_ce = true;
  for (const OfflineRoleSpec& spec : RegisteredOfflineRoleSpecs()) {
    OfflineRoleGateResult role;
    role.role = spec.name;
    role.changed = PairedComparison(
        rows, control_child, treatment_child,
        [&](const SearchPiRow& row) {
          return row.role == spec.role &&
                 row.target_argmax_differs_from_raw;
        });
    role.lower_changed_ce = role.changed.ce_t_minus_c.mean < 0.0;
    out.condition_2_all_roles_lower_changed_ce &= role.lower_changed_ce;
    out.roles.push_back(std::move(role));
  }
  out.condition_3_preserved_noninferiority =
      out.preserved.ce_t_minus_c.upper < 0.01 &&
      out.preserved.adoption_t_minus_c.lower > -0.01;
  out.condition_4_integrity_assertions = integrity_assertions_passed;
  out.passed = out.condition_1_changed_fit_and_adoption &&
               out.condition_2_all_roles_lower_changed_ce &&
               out.condition_3_preserved_noninferiority &&
               out.condition_4_integrity_assertions;
  return out;
}

}  // namespace open_spiel
