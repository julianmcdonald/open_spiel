#ifndef OPEN_SPIEL_EXAMPLES_DUNE_VRPO_TRAINING_H_
#define OPEN_SPIEL_EXAMPLES_DUNE_VRPO_TRAINING_H_

// Phase 4d: isolated, CPU-testable mechanics for one registered PPO/VRPO
// update. This header deliberately has no production trainer call site. It
// consumes the phase-4 arm contract and the phase-1 global Expected-SARSA
// reference from dune_vrpo.h instead of maintaining parallel definitions.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "dune_vrpo.h"

#ifdef OPEN_SPIEL_BUILD_WITH_LIBTORCH
#include <torch/torch.h>

namespace open_spiel {

inline constexpr int kVrpoTrainingMinibatchesPerEpoch = 16;
inline constexpr double kVrpoTrainingClipEpsilon = 0.2;
inline constexpr double kVrpoTrainingEntropyCoefficient = 0.01;
inline constexpr double kVrpoTrainingValueCoefficient = 0.5;
inline constexpr double kVrpoTrainingGradientClipNorm = 0.5;

struct VrpoTrainingRow {
  uint64_t row_id = 0;
  uint64_t episode_id = 0;
  int64_t step_index = -1;
  Player actor = kInvalidPlayer;
  torch::Tensor actor_input;
  torch::Tensor q_input;
  std::vector<Action> legal_actions;
  int chosen_index = -1;
  Action chosen_action = kInvalidAction;
  std::vector<double> old_legal_probabilities;
  double old_chosen_log_probability = 0.0;
  double ppo_advantage = 0.0;
  double ppo_return = 0.0;
  double ppo_old_value = 0.0;
  VrpoSeatValues rewards = {0.0, 0.0, 0.0, 0.0};
  bool terminal_after = false;
};

struct VrpoTrainingEpisode {
  uint64_t episode_id = 0;
  std::vector<VrpoTrainingRow> rows;
};

struct VrpoEpisodePartition {
  std::vector<size_t> episode_indices;
  int64_t row_count = 0;
};

struct VrpoEpisodePartitionPlan {
  uint64_t epoch_seed = 0;
  std::array<VrpoEpisodePartition, kVrpoTrainingMinibatchesPerEpoch>
      minibatches;
  std::string canonical_sha256;
};

struct VrpoActorTrainingOutput {
  torch::Tensor logits;  // [rows, action_dim]
  torch::Tensor values;  // [rows] or [rows,1]
};

using VrpoActorForward =
    std::function<VrpoActorTrainingOutput(const torch::Tensor&)>;
using VrpoQForward = std::function<torch::Tensor(const torch::Tensor&)>;

struct VrpoTrainingTargetRow {
  uint64_t row_id = 0;
  uint64_t episode_id = 0;
  int64_t step_index = -1;
  Player actor = kInvalidPlayer;
  double actor_advantage = 0.0;
  VrpoSeatValues q_target_absolute = {0.0, 0.0, 0.0, 0.0};
  VrpoSeatValues q_target_actor_relative = {0.0, 0.0, 0.0, 0.0};
};

struct VrpoTrainingTargetBundle {
  std::vector<VrpoTrainingTargetRow> rows;
  std::string actor_values_sha256;
  std::string q_values_sha256;
  std::string target_values_sha256;
  std::string canonical_sha256;
};

struct VrpoTrainingUpdateStats {
  bool success = false;
  int64_t actor_optimizer_steps = 0;
  int64_t q_optimizer_steps = 0;
  int64_t actor_backward_calls = 0;
  int64_t q_backward_calls = 0;
  int64_t actor_rows_seen = 0;
  int64_t q_rows_seen = 0;
  int64_t target_recomputations_after_actor = 0;
  bool complete_episode_partitions = false;
  bool q_frozen_during_actor = false;
  bool advantages_detached = false;
  bool targets_recomputed_after_actor = false;
  bool current_rollout_only = false;
  double actor_loss_mean = 0.0;
  double q_loss_mean = 0.0;
  double max_abs_advantage = 0.0;
  double min_ratio = std::numeric_limits<double>::infinity();
  double max_ratio = 0.0;
  double max_full_legal_kl = 0.0;
  double max_actor_grad_norm = 0.0;
  double max_q_grad_norm = 0.0;
  double max_value_head_grad_norm = 0.0;
  std::string actor_values_before_sha256;
  std::string actor_values_after_sha256;
  std::string q_values_before_sha256;
  std::string q_values_after_sha256;
  std::string value_head_before_sha256;
  std::string value_head_after_sha256;
  std::string pre_actor_target_values_sha256;
  std::string post_actor_target_values_sha256;
  std::string post_actor_target_bundle_sha256;
  std::array<std::string, 4> actor_epoch_partition_sha256;
  std::array<std::string, 4> q_epoch_partition_sha256;
  std::string deterministic_summary_sha256;
};

namespace vrpo_training_internal {

inline uint64_t SplitMix64(uint64_t value) {
  value += 0x9e3779b97f4a7c15ULL;
  value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
  value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
  return value ^ (value >> 31);
}

template <typename T>
inline void AppendPod(std::string* payload, const T& value) {
  payload->append(reinterpret_cast<const char*>(&value), sizeof(value));
}

inline bool FiniteTensor(const torch::Tensor& tensor) {
  return tensor.defined() && tensor.numel() > 0 &&
         torch::isfinite(tensor).all().item<bool>();
}

inline bool ModuleValueSha256(torch::nn::Module& module,
                              const std::string& prefix,
                              std::string* output, std::string* error) {
  auto fail = [&](const std::string& message) {
    if (error != nullptr) *error = message;
    if (output != nullptr) output->clear();
    return false;
  };
  if (output == nullptr) return fail("null module-value hash output");
  output->clear();
  std::vector<VrpoNamedParameterIdentity> identities;
  for (const auto& item : module.named_parameters()) {
    if (!prefix.empty() && item.key().rfind(prefix, 0) != 0) continue;
    torch::Tensor value =
        item.value().detach().contiguous().cpu().to(torch::kFloat32);
    if (!FiniteTensor(value)) {
      return fail("module parameter is empty or nonfinite: " + item.key());
    }
    std::string bytes(reinterpret_cast<const char*>(value.data_ptr<float>()),
                      value.numel() * sizeof(float));
    identities.push_back(
        {item.key(),
         std::vector<int64_t>(value.sizes().begin(), value.sizes().end()),
         ComputeStringSHA256(bytes)});
  }
  if (identities.empty()) {
    return fail("module-value hash prefix selected no parameters: " + prefix);
  }
  *output = VrpoNamedParameterIdentitySha256(identities, true);
  return true;
}

inline bool GradientNorm(torch::nn::Module& module,
                         const std::string& prefix, double* norm,
                         bool* any_gradient, std::string* error) {
  if (norm == nullptr || any_gradient == nullptr) {
    if (error != nullptr) *error = "null gradient summary output";
    return false;
  }
  *norm = 0.0;
  *any_gradient = false;
  long double sum_squares = 0.0;
  bool selected = false;
  for (const auto& item : module.named_parameters()) {
    if (!prefix.empty() && item.key().rfind(prefix, 0) != 0) continue;
    selected = true;
    const torch::Tensor gradient = item.value().grad();
    if (!gradient.defined()) continue;
    *any_gradient = true;
    if (!FiniteTensor(gradient)) {
      if (error != nullptr) *error = "parameter gradient is nonfinite";
      return false;
    }
    const double squared = gradient.detach().to(torch::kFloat64)
                               .pow(2)
                               .sum()
                               .item<double>();
    if (!std::isfinite(squared)) {
      if (error != nullptr) *error = "gradient norm is nonfinite";
      return false;
    }
    sum_squares += squared;
  }
  if (!selected) {
    if (error != nullptr) *error = "gradient prefix selected no parameters";
    return false;
  }
  *norm = std::sqrt(static_cast<double>(sum_squares));
  if (!std::isfinite(*norm)) {
    if (error != nullptr) *error = "gradient norm is nonfinite";
    return false;
  }
  return true;
}

inline bool ValidateOptimizerCoverage(torch::optim::Optimizer& optimizer,
                                      torch::nn::Module& expected,
                                      torch::nn::Module& forbidden,
                                      std::string* error) {
  std::set<c10::TensorImpl*> expected_parameters;
  std::set<c10::TensorImpl*> forbidden_parameters;
  for (const auto& item : expected.named_parameters()) {
    expected_parameters.insert(item.value().unsafeGetTensorImpl());
  }
  for (const auto& item : forbidden.named_parameters()) {
    forbidden_parameters.insert(item.value().unsafeGetTensorImpl());
  }
  if (expected_parameters.empty()) {
    if (error != nullptr) *error = "optimizer expected module is empty";
    return false;
  }
  std::set<c10::TensorImpl*> actual;
  for (const auto& group : optimizer.param_groups()) {
    for (const auto& parameter : group.params()) {
      c10::TensorImpl* identity = parameter.unsafeGetTensorImpl();
      if (!actual.insert(identity).second ||
          forbidden_parameters.count(identity) != 0) {
        if (error != nullptr) {
          *error = "optimizer has duplicate or cross-module parameter";
        }
        return false;
      }
    }
  }
  if (actual != expected_parameters) {
    if (error != nullptr) *error = "optimizer module coverage is not exact";
    return false;
  }
  return true;
}

inline bool OptimizerStateIsFresh(torch::optim::Optimizer& optimizer,
                                  std::string* error) {
  for (const auto& entry : optimizer.state()) {
    const auto* state = dynamic_cast<const torch::optim::AdamWParamState*>(
        entry.second.get());
    if (state == nullptr || state->step() != 0) {
      if (error != nullptr) *error = "optimizer state is not fresh step zero";
      return false;
    }
    if ((state->exp_avg().defined() &&
         (!FiniteTensor(state->exp_avg()) ||
          state->exp_avg().abs().max().item<double>() != 0.0)) ||
        (state->exp_avg_sq().defined() &&
         (!FiniteTensor(state->exp_avg_sq()) ||
          state->exp_avg_sq().abs().max().item<double>() != 0.0))) {
      if (error != nullptr) *error = "optimizer moments are not exactly zero";
      return false;
    }
  }
  return true;
}

inline bool Float32DomainFinite(double value) {
  return std::isfinite(value) &&
         std::isfinite(static_cast<float>(value));
}

inline bool ValidatePlannedActorMinibatches(
    const std::vector<VrpoTrainingEpisode>& episodes,
    const std::array<VrpoEpisodePartitionPlan, 4>& plans,
    std::string* error) {
  auto fail = [&](const std::string& message) {
    if (error != nullptr) *error = message;
    return false;
  };
  for (size_t epoch = 0; epoch < plans.size(); ++epoch) {
    for (size_t minibatch_index = 0;
         minibatch_index < plans[epoch].minibatches.size();
         ++minibatch_index) {
      const auto& minibatch = plans[epoch].minibatches[minibatch_index];
      bool has_nontrivial_row = false;
      for (size_t episode_index : minibatch.episode_indices) {
        if (episode_index >= episodes.size()) {
          return fail("actor plan episode index is out of range");
        }
        for (const VrpoTrainingRow& row : episodes[episode_index].rows) {
          if (row.legal_actions.size() > 1) has_nontrivial_row = true;
          if (row.chosen_index < 0 ||
              row.chosen_index >= static_cast<int>(row.legal_actions.size()) ||
              row.legal_actions[row.chosen_index] != row.chosen_action) {
            return fail("planned actor row chosen/legal alignment is invalid");
          }
          if (!Float32DomainFinite(row.old_chosen_log_probability) ||
              !Float32DomainFinite(row.ppo_advantage) ||
              !Float32DomainFinite(row.ppo_return) ||
              !Float32DomainFinite(row.ppo_old_value)) {
            return fail("planned PPO scalar is not finite in Float32 domain");
          }
          for (double probability : row.old_legal_probabilities) {
            if (!Float32DomainFinite(probability)) {
              return fail(
                  "planned behavior probability is not finite in Float32 domain");
            }
          }
          for (double reward : row.rewards) {
            if (!Float32DomainFinite(reward)) {
              return fail("planned reward is not finite in Float32 domain");
            }
          }
        }
      }
      if (!has_nontrivial_row) {
        return fail("planned actor minibatch has no nontrivial legal decision");
      }
    }
  }
  return true;
}

inline bool ValidateTargetBundleFloat32Domain(
    const VrpoTrainingTargetBundle& bundle, std::string* error) {
  if (bundle.rows.empty()) {
    if (error != nullptr) *error = "training target bundle is empty";
    return false;
  }
  for (const auto& row : bundle.rows) {
    if (!Float32DomainFinite(row.actor_advantage)) {
      if (error != nullptr) {
        *error = "VRPO actor advantage is not finite in Float32 domain";
      }
      return false;
    }
    for (double value : row.q_target_absolute) {
      if (!Float32DomainFinite(value)) {
        if (error != nullptr) {
          *error = "VRPO absolute Q target is not finite in Float32 domain";
        }
        return false;
      }
    }
    for (double value : row.q_target_actor_relative) {
      if (!Float32DomainFinite(value)) {
        if (error != nullptr) {
          *error = "VRPO relative Q target is not finite in Float32 domain";
        }
        return false;
      }
    }
  }
  return true;
}

class TrainingTransaction {
 public:
  bool Capture(torch::nn::Module& actor_model, torch::nn::Module& q_model,
               torch::optim::Optimizer& actor_optimizer,
               torch::optim::Optimizer& q_optimizer, std::string* error) {
    if (active_) {
      if (error != nullptr) *error = "training transaction is already active";
      return false;
    }
    actor_model_ = &actor_model;
    q_model_ = &q_model;
    actor_optimizer_ = &actor_optimizer;
    q_optimizer_ = &q_optimizer;
    actor_training_ = actor_model.is_training();
    q_training_ = q_model.is_training();
    for (const auto& item : actor_model.named_parameters()) {
      actor_requires_grad_.push_back(item.value().requires_grad());
    }
    for (const auto& item : q_model.named_parameters()) {
      q_requires_grad_.push_back(item.value().requires_grad());
    }
    try {
      actor_module_bytes_ = SaveModule(actor_model);
      q_module_bytes_ = SaveModule(q_model);
      actor_optimizer_bytes_ = SaveOptimizer(actor_optimizer);
      q_optimizer_bytes_ = SaveOptimizer(q_optimizer);
    } catch (const std::exception& exception) {
      if (error != nullptr) {
        *error = std::string("training transaction capture failed: ") +
                 exception.what();
      }
      Clear();
      return false;
    }
    active_ = true;
    return true;
  }

  bool Rollback(std::string* error) noexcept {
    if (!active_) return true;
    active_ = false;
    try {
      LoadModule(*actor_model_, actor_module_bytes_);
      LoadModule(*q_model_, q_module_bytes_);
      actor_optimizer_->state().clear();
      q_optimizer_->state().clear();
      LoadOptimizer(*actor_optimizer_, actor_optimizer_bytes_);
      LoadOptimizer(*q_optimizer_, q_optimizer_bytes_);
      actor_optimizer_->zero_grad();
      q_optimizer_->zero_grad();
      RestoreRequiresGrad(*actor_model_, actor_requires_grad_);
      RestoreRequiresGrad(*q_model_, q_requires_grad_);
      actor_model_->train(actor_training_);
      q_model_->train(q_training_);
      Clear();
      return true;
    } catch (const std::exception& exception) {
      if (error != nullptr) {
        *error = std::string("training transaction rollback failed: ") +
                 exception.what();
      }
      Clear();
      return false;
    }
  }

  void Commit() {
    active_ = false;
    Clear();
  }

  ~TrainingTransaction() {
    std::string ignored;
    Rollback(&ignored);
  }

 private:
  static std::string SaveModule(torch::nn::Module& module) {
    torch::serialize::OutputArchive archive;
    module.save(archive);
    std::ostringstream stream(std::ios::out | std::ios::binary);
    archive.save_to(stream);
    return stream.str();
  }

  static std::string SaveOptimizer(torch::optim::Optimizer& optimizer) {
    torch::serialize::OutputArchive archive;
    optimizer.save(archive);
    std::ostringstream stream(std::ios::out | std::ios::binary);
    archive.save_to(stream);
    return stream.str();
  }

  static void LoadModule(torch::nn::Module& module,
                         const std::string& bytes) {
    std::istringstream stream(bytes, std::ios::in | std::ios::binary);
    torch::serialize::InputArchive archive;
    archive.load_from(stream, torch::kCPU);
    module.load(archive);
  }

  static void LoadOptimizer(torch::optim::Optimizer& optimizer,
                            const std::string& bytes) {
    std::istringstream stream(bytes, std::ios::in | std::ios::binary);
    torch::serialize::InputArchive archive;
    archive.load_from(stream, torch::kCPU);
    optimizer.load(archive);
  }

  static void RestoreRequiresGrad(torch::nn::Module& module,
                                  const std::vector<bool>& values) {
    size_t index = 0;
    for (auto& item : module.named_parameters()) {
      if (index >= values.size()) {
        throw std::runtime_error("transaction parameter count changed");
      }
      item.value().set_requires_grad(values[index++]);
    }
    if (index != values.size()) {
      throw std::runtime_error("transaction parameter count changed");
    }
  }

  void Clear() {
    actor_model_ = nullptr;
    q_model_ = nullptr;
    actor_optimizer_ = nullptr;
    q_optimizer_ = nullptr;
    actor_requires_grad_.clear();
    q_requires_grad_.clear();
    actor_module_bytes_.clear();
    q_module_bytes_.clear();
    actor_optimizer_bytes_.clear();
    q_optimizer_bytes_.clear();
  }

  bool active_ = false;
  bool actor_training_ = false;
  bool q_training_ = false;
  torch::nn::Module* actor_model_ = nullptr;
  torch::nn::Module* q_model_ = nullptr;
  torch::optim::Optimizer* actor_optimizer_ = nullptr;
  torch::optim::Optimizer* q_optimizer_ = nullptr;
  std::vector<bool> actor_requires_grad_;
  std::vector<bool> q_requires_grad_;
  std::string actor_module_bytes_;
  std::string q_module_bytes_;
  std::string actor_optimizer_bytes_;
  std::string q_optimizer_bytes_;
};

inline torch::Tensor CenterAndCapLegalLogits(const torch::Tensor& legal_logits,
                                             double logit_cap) {
  torch::Tensor centered = legal_logits - legal_logits.mean();
  if (logit_cap <= 0.0) return centered;
  return logit_cap * torch::tanh(centered / logit_cap);
}

inline bool LegalPolicy(const torch::Tensor& row_logits,
                        const VrpoTrainingRow& row, double logit_cap,
                        torch::Tensor* log_probabilities,
                        torch::Tensor* probabilities, std::string* error) {
  auto fail = [&](const std::string& message) {
    if (error != nullptr) *error = message;
    if (log_probabilities != nullptr) *log_probabilities = torch::Tensor();
    if (probabilities != nullptr) *probabilities = torch::Tensor();
    return false;
  };
  if (log_probabilities == nullptr || probabilities == nullptr) {
    return fail("null legal-policy output");
  }
  if (!row_logits.defined() || row_logits.dim() != 1 ||
      row_logits.scalar_type() != torch::kFloat32 ||
      !FiniteTensor(row_logits)) {
    return fail("actor row logits must be finite Float32 rank one");
  }
  std::vector<int64_t> actions(row.legal_actions.begin(),
                               row.legal_actions.end());
  for (int64_t action : actions) {
    if (action < 0 || action >= row_logits.size(0)) {
      return fail("legal action exceeds actor output width");
    }
  }
  torch::Tensor indices = torch::tensor(
      actions, torch::TensorOptions().dtype(torch::kInt64).device(
                   row_logits.device()));
  torch::Tensor legal_logits = row_logits.index_select(0, indices);
  torch::Tensor transformed = CenterAndCapLegalLogits(legal_logits, logit_cap);
  *log_probabilities = torch::log_softmax(transformed, 0);
  *probabilities = torch::softmax(transformed, 0);
  if (!FiniteTensor(*log_probabilities) || !FiniteTensor(*probabilities)) {
    return fail("derived legal policy is nonfinite");
  }
  return true;
}

struct FlatBatch {
  std::vector<const VrpoTrainingRow*> rows;
  std::vector<std::pair<size_t, size_t>> episode_ranges;
};

inline FlatBatch FlattenEpisodes(const std::vector<VrpoTrainingEpisode>& episodes,
                                 const std::vector<size_t>& indices) {
  FlatBatch flat;
  for (size_t index : indices) {
    const size_t begin = flat.rows.size();
    for (const auto& row : episodes[index].rows) flat.rows.push_back(&row);
    flat.episode_ranges.push_back({begin, flat.rows.size()});
  }
  return flat;
}

inline torch::Tensor StackInputs(const std::vector<const VrpoTrainingRow*>& rows,
                                 bool actor_input) {
  std::vector<torch::Tensor> tensors;
  tensors.reserve(rows.size());
  for (const VrpoTrainingRow* row : rows) {
    tensors.push_back(actor_input ? row->actor_input : row->q_input);
  }
  return torch::stack(tensors);
}

inline bool ValidateActorOutput(const VrpoActorTrainingOutput& output,
                                int64_t rows, std::string* error) {
  const bool values_shape = output.values.defined() &&
      ((output.values.dim() == 1 && output.values.size(0) == rows) ||
       (output.values.dim() == 2 && output.values.size(0) == rows &&
        output.values.size(1) == 1));
  if (!output.logits.defined() || output.logits.dim() != 2 ||
      output.logits.size(0) != rows || output.logits.size(1) <= 0 ||
      output.logits.scalar_type() != torch::kFloat32 || !values_shape ||
      output.values.scalar_type() != torch::kFloat32 ||
      !FiniteTensor(output.logits) || !FiniteTensor(output.values)) {
    if (error != nullptr) *error = "actor output shape/dtype/finiteness invalid";
    return false;
  }
  return true;
}

inline bool ValidateQOutput(const torch::Tensor& output, int64_t rows,
                            std::string* error) {
  if (!output.defined() || output.dim() != 3 || output.size(0) != rows ||
      output.size(1) <= 0 || output.size(2) != kVrpoNumSeats ||
      output.scalar_type() != torch::kFloat32 || !FiniteTensor(output)) {
    if (error != nullptr) *error = "Q output must be finite Float32 [rows,A,4]";
    return false;
  }
  return true;
}

inline bool BuildReferences(
    const VrpoPhase4ArmConfig& config, const FlatBatch& flat,
    const std::vector<std::vector<double>>& probabilities,
    const torch::Tensor& q_output,
    std::vector<double>* advantages,
    std::vector<VrpoTrainingTargetRow>* targets, std::string* error) {
  auto fail = [&](const std::string& message) {
    if (error != nullptr) *error = message;
    if (advantages != nullptr) advantages->clear();
    if (targets != nullptr) targets->clear();
    return false;
  };
  if (advantages == nullptr || targets == nullptr ||
      probabilities.size() != flat.rows.size()) {
    return fail("reference builder input widths differ");
  }
  advantages->assign(flat.rows.size(), 0.0);
  targets->assign(flat.rows.size(), VrpoTrainingTargetRow{});
  torch::Tensor q_cpu = q_output.detach().contiguous().cpu();
  const auto q = q_cpu.accessor<float, 3>();
  for (const auto& range : flat.episode_ranges) {
    std::vector<VrpoTimelineRow> timeline;
    timeline.reserve(range.second - range.first);
    for (size_t flat_index = range.first; flat_index < range.second;
         ++flat_index) {
      const VrpoTrainingRow& source = *flat.rows[flat_index];
      VrpoTimelineRow row;
      row.episode_id = source.episode_id;
      row.actor = source.actor;
      row.legal_actions = source.legal_actions;
      row.chosen_index = source.chosen_index;
      row.chosen_action = source.chosen_action;
      row.legal_probabilities = probabilities[flat_index];
      row.rewards = source.rewards;
      row.terminal_after = source.terminal_after;
      std::vector<VrpoActorRelativeSeatValues> relative;
      relative.reserve(source.legal_actions.size());
      for (Action action : source.legal_actions) {
        if (action < 0 || action >= q_output.size(1)) {
          return fail("legal action exceeds Q output width");
        }
        VrpoActorRelativeSeatValues values;
        for (int slot = 0; slot < kVrpoNumSeats; ++slot) {
          values.slots[slot] = q[flat_index][action][slot];
        }
        relative.push_back(values);
      }
      std::string adapter_error;
      if (!VrpoActorRelativeQToAbsolute(
              source.actor, relative, &row.legal_q_values,
              &adapter_error)) {
        return fail("legal Q seat conversion failed: " + adapter_error);
      }
      timeline.push_back(std::move(row));
    }
    VrpoReferenceTrace trace;
    std::string reference_error;
    if (!ComputeVrpoExpectedSarsaLambdaReference(
            timeline, config.gamma, config.lambda, &trace,
            &reference_error, kVrpoMaxProbabilityTolerance)) {
      return fail("global Expected-SARSA reference failed: " +
                  reference_error);
    }
    for (size_t local = 0; local < trace.rows.size(); ++local) {
      const size_t flat_index = range.first + local;
      const VrpoTrainingRow& source = *flat.rows[flat_index];
      const VrpoReferenceRow& reference = trace.rows[local];
      (*advantages)[flat_index] = reference.actor_advantage;
      VrpoTrainingTargetRow target;
      target.row_id = source.row_id;
      target.episode_id = source.episode_id;
      target.step_index = source.step_index;
      target.actor = source.actor;
      target.actor_advantage = reference.actor_advantage;
      target.q_target_absolute = reference.q_target;
      for (int slot = 0; slot < kVrpoNumSeats; ++slot) {
        target.q_target_actor_relative[slot] =
            reference.q_target[(source.actor + slot) % kVrpoNumSeats];
      }
      (*targets)[flat_index] = target;
    }
  }
  return true;
}

inline std::string TargetValuesSha256(
    const std::vector<VrpoTrainingTargetRow>& rows) {
  std::string payload = "dune_vrpo_phase4d_target_values_v1";
  const uint64_t count = rows.size();
  AppendPod(&payload, count);
  for (const auto& row : rows) {
    AppendPod(&payload, row.row_id);
    AppendPod(&payload, row.episode_id);
    AppendPod(&payload, row.step_index);
    AppendPod(&payload, row.actor);
    AppendPod(&payload, row.actor_advantage);
    for (double value : row.q_target_absolute) AppendPod(&payload, value);
    for (double value : row.q_target_actor_relative) AppendPod(&payload, value);
  }
  return ComputeStringSHA256(payload);
}

inline bool BuildLegalPolicies(
    const VrpoActorTrainingOutput& actor_output, const FlatBatch& flat,
    double logit_cap, std::vector<torch::Tensor>* log_probabilities,
    std::vector<torch::Tensor>* probabilities,
    std::vector<std::vector<double>>* detached_probabilities,
    std::string* error) {
  log_probabilities->clear();
  probabilities->clear();
  detached_probabilities->clear();
  for (size_t index = 0; index < flat.rows.size(); ++index) {
    torch::Tensor log_probs;
    torch::Tensor probs;
    if (!LegalPolicy(actor_output.logits[index], *flat.rows[index], logit_cap,
                     &log_probs, &probs, error)) {
      return false;
    }
    log_probabilities->push_back(log_probs);
    probabilities->push_back(probs);
    torch::Tensor cpu = probs.detach().contiguous().cpu().to(torch::kFloat64);
    std::vector<double> values(cpu.numel());
    std::memcpy(values.data(), cpu.data_ptr<double>(),
                values.size() * sizeof(double));
    detached_probabilities->push_back(std::move(values));
  }
  return true;
}

inline bool ValidateAllData(const std::vector<VrpoTrainingEpisode>& episodes,
                            std::string* error) {
  auto fail = [&](const std::string& message) {
    if (error != nullptr) *error = message;
    return false;
  };
  if (episodes.size() < kVrpoTrainingMinibatchesPerEpoch) {
    return fail("phase4d requires at least 16 complete episodes");
  }
  std::set<uint64_t> episode_ids;
  std::set<uint64_t> row_ids;
  int64_t actor_width = -1;
  int64_t q_width = -1;
  for (const auto& episode : episodes) {
    if (!episode_ids.insert(episode.episode_id).second) {
      return fail("duplicate episode ID");
    }
    if (episode.rows.empty()) return fail("empty episode");
    for (size_t index = 0; index < episode.rows.size(); ++index) {
      const VrpoTrainingRow& row = episode.rows[index];
      if (row.episode_id != episode.episode_id ||
          row.step_index != static_cast<int64_t>(index)) {
        return fail("episode row identity/order is malformed");
      }
      if (!row_ids.insert(row.row_id).second) return fail("duplicate row ID");
      if (row.actor < 0 || row.actor >= kVrpoNumSeats) {
        return fail("row actor is out of range");
      }
      if (row.terminal_after != (index + 1 == episode.rows.size())) {
        return fail("terminal marker is not exactly the episode final row");
      }
      if (!row.actor_input.defined() || row.actor_input.dim() != 1 ||
          row.actor_input.scalar_type() != torch::kFloat32 ||
          !FiniteTensor(row.actor_input) || !row.q_input.defined() ||
          row.q_input.dim() != 1 ||
          row.q_input.scalar_type() != torch::kFloat32 ||
          !FiniteTensor(row.q_input)) {
        return fail("training input is empty, malformed, or nonfinite");
      }
      if (actor_width < 0) actor_width = row.actor_input.size(0);
      if (q_width < 0) q_width = row.q_input.size(0);
      if (row.actor_input.size(0) != actor_width ||
          row.q_input.size(0) != q_width) {
        return fail("training input widths differ");
      }
      if (row.legal_actions.empty() ||
          row.old_legal_probabilities.size() != row.legal_actions.size() ||
          row.chosen_index < 0 ||
          row.chosen_index >= static_cast<int>(row.legal_actions.size()) ||
          row.chosen_action != row.legal_actions[row.chosen_index]) {
        return fail("legal/chosen action alignment is malformed");
      }
      std::set<Action> legal;
      double probability_sum = 0.0;
      for (size_t action = 0; action < row.legal_actions.size(); ++action) {
        if (row.legal_actions[action] < 0 ||
            row.legal_actions[action] >= kVrpoDuneActionDim ||
            !legal.insert(row.legal_actions[action]).second) {
          return fail("legal actions are duplicate or out of range");
        }
        const double probability = row.old_legal_probabilities[action];
        if (!std::isfinite(probability) || probability < 0.0 ||
            probability > 1.0) {
          return fail("old legal probability is invalid");
        }
        probability_sum += probability;
      }
      const double chosen_probability =
          row.old_legal_probabilities[row.chosen_index];
      if (std::abs(probability_sum - 1.0) > 1e-6 ||
          chosen_probability <= 0.0 ||
          !std::isfinite(row.old_chosen_log_probability) ||
          std::abs(std::log(chosen_probability) -
                   row.old_chosen_log_probability) > 1e-6 ||
          !std::isfinite(row.ppo_advantage) ||
          !std::isfinite(row.ppo_return) ||
          !std::isfinite(row.ppo_old_value)) {
        return fail("stored PPO behavior/GAE/return data is invalid");
      }
      for (double reward : row.rewards) {
        if (!std::isfinite(reward)) return fail("row reward is nonfinite");
      }
    }
  }
  return true;
}

inline std::string StatsCanonicalPayload(const VrpoTrainingUpdateStats& stats) {
  std::string payload = "dune_vrpo_phase4d_update_stats_v1";
  AppendPod(&payload, stats.actor_optimizer_steps);
  AppendPod(&payload, stats.q_optimizer_steps);
  AppendPod(&payload, stats.actor_rows_seen);
  AppendPod(&payload, stats.q_rows_seen);
  AppendPod(&payload, stats.actor_loss_mean);
  AppendPod(&payload, stats.q_loss_mean);
  AppendPod(&payload, stats.max_abs_advantage);
  AppendPod(&payload, stats.min_ratio);
  AppendPod(&payload, stats.max_ratio);
  AppendPod(&payload, stats.max_full_legal_kl);
  AppendPod(&payload, stats.max_actor_grad_norm);
  AppendPod(&payload, stats.max_q_grad_norm);
  AppendPod(&payload, stats.max_value_head_grad_norm);
  payload.append(stats.actor_values_after_sha256);
  payload.append(stats.q_values_after_sha256);
  payload.append(stats.value_head_after_sha256);
  payload.append(stats.post_actor_target_values_sha256);
  payload.append(stats.post_actor_target_bundle_sha256);
  for (const auto& digest : stats.actor_epoch_partition_sha256) {
    payload.append(digest);
  }
  for (const auto& digest : stats.q_epoch_partition_sha256) {
    payload.append(digest);
  }
  return payload;
}

inline std::string StatsSha256(const VrpoTrainingUpdateStats& stats) {
  return ComputeStringSHA256(StatsCanonicalPayload(stats));
}

}  // namespace vrpo_training_internal

inline bool BuildVrpoEpisodePartitionPlan(
    const std::vector<VrpoTrainingEpisode>& episodes, uint64_t epoch_seed,
    VrpoEpisodePartitionPlan* output, std::string* error) {
  auto fail = [&](const std::string& message) {
    if (error != nullptr) *error = message;
    if (output != nullptr) *output = VrpoEpisodePartitionPlan{};
    return false;
  };
  if (output == nullptr) return fail("null partition-plan output");
  *output = VrpoEpisodePartitionPlan{};
  if (epoch_seed == 0) return fail("partition seed must be nonzero");
  if (!vrpo_training_internal::ValidateAllData(episodes, error)) return false;

  std::vector<size_t> order(episodes.size());
  for (size_t index = 0; index < order.size(); ++index) order[index] = index;
  std::stable_sort(order.begin(), order.end(), [&](size_t first, size_t second) {
    if (episodes[first].rows.size() != episodes[second].rows.size()) {
      return episodes[first].rows.size() > episodes[second].rows.size();
    }
    const uint64_t first_key = vrpo_training_internal::SplitMix64(
        epoch_seed ^ episodes[first].episode_id);
    const uint64_t second_key = vrpo_training_internal::SplitMix64(
        epoch_seed ^ episodes[second].episode_id);
    if (first_key != second_key) return first_key < second_key;
    return episodes[first].episode_id < episodes[second].episode_id;
  });

  VrpoEpisodePartitionPlan plan;
  plan.epoch_seed = epoch_seed;
  for (size_t episode_index : order) {
    size_t destination = 0;
    for (size_t bin = 1; bin < plan.minibatches.size(); ++bin) {
      const auto& candidate = plan.minibatches[bin];
      const auto& current = plan.minibatches[destination];
      if (candidate.row_count < current.row_count ||
          (candidate.row_count == current.row_count &&
           vrpo_training_internal::SplitMix64(epoch_seed ^ bin) <
               vrpo_training_internal::SplitMix64(epoch_seed ^ destination))) {
        destination = bin;
      }
    }
    plan.minibatches[destination].episode_indices.push_back(episode_index);
    plan.minibatches[destination].row_count +=
        static_cast<int64_t>(episodes[episode_index].rows.size());
  }
  std::set<size_t> covered;
  std::string payload = "dune_vrpo_phase4d_whole_episode_partition_v1";
  vrpo_training_internal::AppendPod(&payload, epoch_seed);
  for (auto& minibatch : plan.minibatches) {
    if (minibatch.episode_indices.empty() || minibatch.row_count <= 0) {
      return fail("whole-episode partition produced an empty minibatch");
    }
    std::stable_sort(
        minibatch.episode_indices.begin(), minibatch.episode_indices.end(),
        [&](size_t first, size_t second) {
          const uint64_t first_key = vrpo_training_internal::SplitMix64(
              epoch_seed + episodes[first].episode_id);
          const uint64_t second_key = vrpo_training_internal::SplitMix64(
              epoch_seed + episodes[second].episode_id);
          if (first_key != second_key) return first_key < second_key;
          return episodes[first].episode_id < episodes[second].episode_id;
        });
    vrpo_training_internal::AppendPod(&payload, minibatch.row_count);
    const uint64_t count = minibatch.episode_indices.size();
    vrpo_training_internal::AppendPod(&payload, count);
    for (size_t episode_index : minibatch.episode_indices) {
      if (!covered.insert(episode_index).second) {
        return fail("partition assigned an episode more than once");
      }
      vrpo_training_internal::AppendPod(
          &payload, episodes[episode_index].episode_id);
      const uint64_t rows = episodes[episode_index].rows.size();
      vrpo_training_internal::AppendPod(&payload, rows);
      for (const auto& row : episodes[episode_index].rows) {
        vrpo_training_internal::AppendPod(&payload, row.row_id);
      }
    }
  }
  if (covered.size() != episodes.size()) {
    return fail("partition omitted one or more episodes");
  }
  plan.canonical_sha256 = ComputeStringSHA256(payload);
  *output = std::move(plan);
  return true;
}

inline bool VrpoTrainingLegalProbabilities(
    const torch::Tensor& row_logits, const VrpoTrainingRow& row,
    double logit_cap, std::vector<double>* output, std::string* error) {
  if (output == nullptr) {
    if (error != nullptr) *error = "null legal probability output";
    return false;
  }
  output->clear();
  torch::NoGradGuard no_grad;
  torch::Tensor log_probs;
  torch::Tensor probabilities;
  if (!vrpo_training_internal::LegalPolicy(
          row_logits, row, logit_cap, &log_probs, &probabilities, error)) {
    return false;
  }
  torch::Tensor cpu =
      probabilities.contiguous().cpu().to(torch::kFloat64);
  output->resize(cpu.numel());
  std::memcpy(output->data(), cpu.data_ptr<double>(),
              output->size() * sizeof(double));
  return true;
}

inline bool ComputeVrpoTrainingTargets(
    const VrpoPhase4ArmConfig& config,
    const std::vector<VrpoTrainingEpisode>& episodes,
    torch::nn::Module& actor_model, torch::nn::Module& q_model,
    const VrpoActorForward& actor_forward, const VrpoQForward& q_forward,
    VrpoTrainingTargetBundle* output, std::string* error) {
  auto fail = [&](const std::string& message) {
    if (error != nullptr) *error = message;
    if (output != nullptr) *output = VrpoTrainingTargetBundle{};
    return false;
  };
  if (output == nullptr || !actor_forward || !q_forward) {
    return fail("target computation callbacks/output are incomplete");
  }
  *output = VrpoTrainingTargetBundle{};
  std::string contract_error;
  if (!ValidateVrpoPhase4ArmConfig(config, &contract_error)) {
    return fail(contract_error);
  }
  if (!vrpo_training_internal::ValidateAllData(episodes, error)) return false;
  std::vector<size_t> all(episodes.size());
  for (size_t index = 0; index < all.size(); ++index) all[index] = index;
  const auto flat = vrpo_training_internal::FlattenEpisodes(episodes, all);
  torch::NoGradGuard no_grad;
  VrpoActorTrainingOutput actor_output;
  torch::Tensor q_output;
  try {
    actor_output = actor_forward(
        vrpo_training_internal::StackInputs(flat.rows, true));
    q_output = q_forward(vrpo_training_internal::StackInputs(flat.rows, false));
  } catch (const std::exception& exception) {
    return fail(std::string("target forward failed: ") + exception.what());
  }
  if (!vrpo_training_internal::ValidateActorOutput(
          actor_output, flat.rows.size(), error) ||
      !vrpo_training_internal::ValidateQOutput(
          q_output, flat.rows.size(), error)) {
    return false;
  }
  std::vector<torch::Tensor> log_probabilities;
  std::vector<torch::Tensor> probabilities;
  std::vector<std::vector<double>> detached_probabilities;
  if (!vrpo_training_internal::BuildLegalPolicies(
          actor_output, flat, config.logit_cap, &log_probabilities,
          &probabilities, &detached_probabilities, error)) {
    return false;
  }
  std::vector<double> advantages;
  std::vector<VrpoTrainingTargetRow> targets;
  if (!vrpo_training_internal::BuildReferences(
          config, flat, detached_probabilities, q_output, &advantages,
          &targets, error)) {
    return false;
  }
  VrpoTrainingTargetBundle bundle;
  bundle.rows = std::move(targets);
  if (!vrpo_training_internal::ModuleValueSha256(
          actor_model, "", &bundle.actor_values_sha256, error) ||
      !vrpo_training_internal::ModuleValueSha256(
          q_model, "", &bundle.q_values_sha256, error)) {
    return false;
  }
  bundle.target_values_sha256 =
      vrpo_training_internal::TargetValuesSha256(bundle.rows);
  std::string payload = "dune_vrpo_phase4d_target_bundle_v1";
  payload.append(config.arm_id);
  payload.push_back('\0');
  payload.append(bundle.actor_values_sha256);
  payload.append(bundle.q_values_sha256);
  payload.append(bundle.target_values_sha256);
  bundle.canonical_sha256 = ComputeStringSHA256(payload);
  *output = std::move(bundle);
  return true;
}

inline bool ValidateVrpoTrainingTargetsFresh(
    const VrpoTrainingTargetBundle& bundle, torch::nn::Module& actor_model,
    torch::nn::Module& q_model, std::string* error) {
  std::string actor_hash;
  std::string q_hash;
  if (!vrpo_training_internal::ModuleValueSha256(
          actor_model, "", &actor_hash, error) ||
      !vrpo_training_internal::ModuleValueSha256(
          q_model, "", &q_hash, error)) {
    return false;
  }
  if (bundle.rows.empty() || bundle.actor_values_sha256 != actor_hash ||
      bundle.q_values_sha256 != q_hash ||
      bundle.target_values_sha256 !=
          vrpo_training_internal::TargetValuesSha256(bundle.rows)) {
    if (error != nullptr) *error = "VRPO targets are stale or malformed";
    return false;
  }
  return true;
}

inline bool RunVrpoPhase4dOneUpdate(
    const VrpoPhase4ArmConfig& config,
    const std::vector<VrpoTrainingEpisode>& episodes, uint64_t update_seed,
    torch::nn::Module& actor_model, torch::nn::Module& q_model,
    torch::optim::Optimizer& actor_optimizer,
    torch::optim::Optimizer& q_optimizer,
    const VrpoActorForward& actor_forward, const VrpoQForward& q_forward,
    VrpoTrainingUpdateStats* output, std::string* error) {
  auto fail = [&](const std::string& message) {
    if (error != nullptr) *error = message;
    if (output != nullptr) *output = VrpoTrainingUpdateStats{};
    return false;
  };
  if (output == nullptr || !actor_forward || !q_forward || update_seed == 0) {
    return fail("phase4d update inputs are incomplete");
  }
  *output = VrpoTrainingUpdateStats{};
  std::string contract_error;
  if (!ValidateVrpoPhase4ArmConfig(config, &contract_error)) {
    return fail(contract_error);
  }
  if (config.minibatches_per_epoch != kVrpoTrainingMinibatchesPerEpoch ||
      config.actor_epochs != 4 || config.q_epochs != 4 ||
      config.critic_replay_rollouts != 1 || !config.complete_game_batches) {
    return fail("phase4d update config violates the registered mechanics");
  }
  if (!vrpo_training_internal::ValidateAllData(episodes, error) ||
      !vrpo_training_internal::ValidateOptimizerCoverage(
          actor_optimizer, actor_model, q_model, error) ||
      !vrpo_training_internal::ValidateOptimizerCoverage(
          q_optimizer, q_model, actor_model, error) ||
      !vrpo_training_internal::OptimizerStateIsFresh(actor_optimizer, error) ||
      !vrpo_training_internal::OptimizerStateIsFresh(q_optimizer, error)) {
    return false;
  }

  std::array<VrpoEpisodePartitionPlan, 4> actor_plans;
  std::array<VrpoEpisodePartitionPlan, 4> q_plans;
  for (int epoch = 0; epoch < 4; ++epoch) {
    if (!BuildVrpoEpisodePartitionPlan(
            episodes,
            vrpo_training_internal::SplitMix64(update_seed + epoch + 1),
            &actor_plans[epoch], error) ||
        !BuildVrpoEpisodePartitionPlan(
            episodes,
            vrpo_training_internal::SplitMix64(update_seed + 0x10000 +
                                               epoch + 1),
            &q_plans[epoch], error)) {
      return false;
    }
  }
  if (!vrpo_training_internal::ValidatePlannedActorMinibatches(
          episodes, actor_plans, error)) {
    return false;
  }

  VrpoTrainingUpdateStats stats;
  stats.complete_episode_partitions = true;
  stats.current_rollout_only = true;
  stats.advantages_detached = true;
  stats.q_frozen_during_actor =
      config.algorithm == VrpoPhase4Algorithm::kVrpo;
  if (!vrpo_training_internal::ModuleValueSha256(
          actor_model, "", &stats.actor_values_before_sha256, error) ||
      !vrpo_training_internal::ModuleValueSha256(
          q_model, "", &stats.q_values_before_sha256, error) ||
      !vrpo_training_internal::ModuleValueSha256(
          actor_model, "value_head", &stats.value_head_before_sha256,
          error)) {
    return false;
  }

  // Complete no-gradient preflight over every row occurs before the first
  // optimizer step. It catches malformed output shapes, NaNs, bad legal-action
  // widths, or reference failures without partially mutating either module.
  VrpoTrainingTargetBundle pre_actor_targets;
  if (!ComputeVrpoTrainingTargets(
          config, episodes, actor_model, q_model, actor_forward, q_forward,
          &pre_actor_targets, error)) {
    return false;
  }
  if (!vrpo_training_internal::ValidateTargetBundleFloat32Domain(
          pre_actor_targets, error)) {
    return false;
  }
  stats.pre_actor_target_values_sha256 =
      pre_actor_targets.target_values_sha256;

  // Model/optimizer state is now fully captured. Any model-dependent failure
  // that can arise only after an earlier step (for example a later forward or
  // gradient becoming nonfinite) returns through this scope and rolls back
  // parameters, buffers, requires-grad/training modes, and AdamW moments.
  vrpo_training_internal::TrainingTransaction transaction;
  if (!transaction.Capture(actor_model, q_model, actor_optimizer, q_optimizer,
                           error)) {
    return false;
  }

  double actor_loss_sum = 0.0;
  int64_t actor_loss_count = 0;
  std::vector<bool> q_requires_grad;
  q_requires_grad.reserve(q_model.named_parameters().size());
  for (auto& item : q_model.named_parameters()) {
    q_requires_grad.push_back(item.value().requires_grad());
    item.value().set_requires_grad(false);
  }
  const bool q_was_training = q_model.is_training();
  q_model.eval();

  auto restore_q = [&]() {
    size_t index = 0;
    for (auto& item : q_model.named_parameters()) {
      item.value().set_requires_grad(q_requires_grad[index++]);
    }
    q_model.train(q_was_training);
  };

  for (int epoch = 0; epoch < config.actor_epochs; ++epoch) {
    stats.actor_epoch_partition_sha256[epoch] =
        actor_plans[epoch].canonical_sha256;
    for (const auto& minibatch : actor_plans[epoch].minibatches) {
      const auto flat = vrpo_training_internal::FlattenEpisodes(
          episodes, minibatch.episode_indices);
      actor_optimizer.zero_grad();
      q_optimizer.zero_grad();
      VrpoActorTrainingOutput actor_output;
      torch::Tensor q_output;
      try {
        actor_output = actor_forward(
            vrpo_training_internal::StackInputs(flat.rows, true));
        {
          torch::NoGradGuard no_grad;
          q_output = q_forward(
              vrpo_training_internal::StackInputs(flat.rows, false));
        }
      } catch (const std::exception& exception) {
        restore_q();
        return fail(std::string("actor-phase forward failed: ") +
                    exception.what());
      }
      if (!vrpo_training_internal::ValidateActorOutput(
              actor_output, flat.rows.size(), error) ||
          !vrpo_training_internal::ValidateQOutput(
              q_output, flat.rows.size(), error)) {
        restore_q();
        return false;
      }
      std::vector<torch::Tensor> legal_log_probabilities;
      std::vector<torch::Tensor> legal_probabilities;
      std::vector<std::vector<double>> detached_probabilities;
      if (!vrpo_training_internal::BuildLegalPolicies(
              actor_output, flat, config.logit_cap,
              &legal_log_probabilities, &legal_probabilities,
              &detached_probabilities, error)) {
        restore_q();
        return false;
      }
      std::vector<double> advantages(flat.rows.size());
      if (config.algorithm == VrpoPhase4Algorithm::kPpo) {
        for (size_t index = 0; index < flat.rows.size(); ++index) {
          advantages[index] = flat.rows[index]->ppo_advantage;
        }
      } else {
        std::vector<VrpoTrainingTargetRow> ignored_targets;
        if (!vrpo_training_internal::BuildReferences(
                config, flat, detached_probabilities, q_output, &advantages,
                &ignored_targets, error)) {
          restore_q();
          return false;
        }
      }

      // Preserve the current PPO learner's per-minibatch normalization in all
      // four cells. Forced one-action decisions are excluded from the moments
      // and actor/entropy loss exactly as they are in the PPO path.
      std::vector<double> loss_advantages = advantages;
      std::vector<double> nontrivial_advantages;
      for (size_t index = 0; index < flat.rows.size(); ++index) {
        if (flat.rows[index]->legal_actions.size() > 1) {
          nontrivial_advantages.push_back(advantages[index]);
        }
      }
      if (nontrivial_advantages.size() > 1) {
        double mean = 0.0;
        for (double value : nontrivial_advantages) mean += value;
        mean /= nontrivial_advantages.size();
        double variance = 0.0;
        for (double value : nontrivial_advantages) {
          variance += (value - mean) * (value - mean);
        }
        variance /= nontrivial_advantages.size();
        const double scale = std::sqrt(variance) + 1e-8;
        for (double& value : loss_advantages) value = (value - mean) / scale;
      }

      std::vector<torch::Tensor> policy_terms;
      std::vector<torch::Tensor> entropy_terms;
      std::vector<torch::Tensor> value_losses;
      for (size_t index = 0; index < flat.rows.size(); ++index) {
        const VrpoTrainingRow& row = *flat.rows[index];
        const torch::Tensor& log_probs = legal_log_probabilities[index];
        const torch::Tensor& probs = legal_probabilities[index];
        torch::Tensor chosen_log_probability = log_probs[row.chosen_index];
        torch::Tensor ratio = torch::exp(
            chosen_log_probability - row.old_chosen_log_probability);
        const double ratio_value = ratio.detach().item<double>();
        const double advantage = advantages[index];
        if (!std::isfinite(ratio_value) || !std::isfinite(advantage)) {
          restore_q();
          return fail("actor ratio or advantage is nonfinite before step");
        }
        const double loss_advantage = loss_advantages[index];
        if (row.legal_actions.size() > 1) {
          torch::Tensor first = ratio * loss_advantage;
          torch::Tensor second = torch::clamp(
              ratio, 1.0 - kVrpoTrainingClipEpsilon,
              1.0 + kVrpoTrainingClipEpsilon) * loss_advantage;
          policy_terms.push_back(-torch::minimum(first, second));
          entropy_terms.push_back(-(probs * log_probs).sum());
        }
        if (config.algorithm == VrpoPhase4Algorithm::kPpo) {
          torch::Tensor value = actor_output.values.dim() == 2
              ? actor_output.values[index][0]
              : actor_output.values[index];
          torch::Tensor unclipped = torch::pow(value - row.ppo_return, 2);
          torch::Tensor clipped_value = row.ppo_old_value + torch::clamp(
              value - row.ppo_old_value, -kVrpoTrainingClipEpsilon,
              kVrpoTrainingClipEpsilon);
          torch::Tensor clipped =
              torch::pow(clipped_value - row.ppo_return, 2);
          value_losses.push_back(0.5 * torch::maximum(unclipped, clipped));
        }

        torch::Tensor old = torch::tensor(
            row.old_legal_probabilities,
            torch::TensorOptions().dtype(torch::kFloat32).device(
                log_probs.device()));
        torch::Tensor positive = old > 0.0;
        torch::Tensor safe_old = torch::clamp_min(old, 1e-30);
        const double full_kl = torch::where(
            positive, old * (torch::log(safe_old) - log_probs),
            torch::zeros_like(old)).sum().detach().item<double>();
        if (!std::isfinite(full_kl) || full_kl < -1e-5) {
          restore_q();
          return fail("full legal-distribution KL is invalid before step");
        }
        stats.min_ratio = std::min(stats.min_ratio, ratio_value);
        stats.max_ratio = std::max(stats.max_ratio, ratio_value);
        stats.max_full_legal_kl =
            std::max(stats.max_full_legal_kl, std::max(0.0, full_kl));
        stats.max_abs_advantage =
            std::max(stats.max_abs_advantage, std::abs(advantage));
      }
      if (policy_terms.empty()) {
        restore_q();
        return fail("actor minibatch has no nontrivial legal decision");
      }
      torch::Tensor policy_loss = torch::stack(policy_terms).mean();
      torch::Tensor entropy = torch::stack(entropy_terms).mean();
      torch::Tensor total_loss =
          policy_loss - kVrpoTrainingEntropyCoefficient * entropy;
      if (config.algorithm == VrpoPhase4Algorithm::kPpo) {
        total_loss = total_loss + kVrpoTrainingValueCoefficient *
            torch::stack(value_losses).mean();
      }
      if (!vrpo_training_internal::FiniteTensor(total_loss)) {
        restore_q();
        return fail("actor loss is nonfinite before step");
      }
      total_loss.backward();
      ++stats.actor_backward_calls;
      double actor_grad_norm = 0.0;
      double q_grad_norm = 0.0;
      double value_grad_norm = 0.0;
      bool actor_grad = false;
      bool q_grad = false;
      bool value_grad = false;
      if (!vrpo_training_internal::GradientNorm(
              actor_model, "", &actor_grad_norm, &actor_grad, error) ||
          !vrpo_training_internal::GradientNorm(
              q_model, "", &q_grad_norm, &q_grad, error) ||
          !vrpo_training_internal::GradientNorm(
              actor_model, "value_head", &value_grad_norm, &value_grad,
              error)) {
        restore_q();
        return false;
      }
      if (!actor_grad || actor_grad_norm <= 0.0 || q_grad ||
          q_grad_norm != 0.0 ||
          (config.algorithm == VrpoPhase4Algorithm::kVrpo &&
           (value_grad || value_grad_norm != 0.0)) ||
          (config.algorithm == VrpoPhase4Algorithm::kPpo && !value_grad)) {
        restore_q();
        return fail("actor-phase gradient/module gate failed before step");
      }
      stats.max_actor_grad_norm =
          std::max(stats.max_actor_grad_norm, actor_grad_norm);
      stats.max_value_head_grad_norm =
          std::max(stats.max_value_head_grad_norm, value_grad_norm);
      torch::nn::utils::clip_grad_norm_(actor_model.parameters(),
                                       kVrpoTrainingGradientClipNorm);
      actor_optimizer.step();
      ++stats.actor_optimizer_steps;
      stats.actor_rows_seen += flat.rows.size();
      actor_loss_sum += total_loss.detach().item<double>();
      ++actor_loss_count;
    }
  }
  restore_q();
  actor_optimizer.zero_grad();
  q_optimizer.zero_grad();

  VrpoTrainingTargetBundle post_actor_targets;
  if (!ComputeVrpoTrainingTargets(
          config, episodes, actor_model, q_model, actor_forward, q_forward,
          &post_actor_targets, error)) {
    return false;
  }
  if (!vrpo_training_internal::ValidateTargetBundleFloat32Domain(
          post_actor_targets, error)) {
    return false;
  }
  ++stats.target_recomputations_after_actor;
  stats.targets_recomputed_after_actor = true;
  stats.post_actor_target_values_sha256 =
      post_actor_targets.target_values_sha256;
  stats.post_actor_target_bundle_sha256 =
      post_actor_targets.canonical_sha256;
  if (!ValidateVrpoTrainingTargetsFresh(
          post_actor_targets, actor_model, q_model, error)) {
    return false;
  }

  double q_loss_sum = 0.0;
  int64_t q_loss_count = 0;
  std::map<uint64_t, VrpoTrainingTargetRow> target_by_row;
  for (const auto& target : post_actor_targets.rows) {
    if (!target_by_row.emplace(target.row_id, target).second) {
      return fail("post-actor target bundle repeats a row");
    }
  }
  if (config.algorithm == VrpoPhase4Algorithm::kVrpo) {
    const bool actor_was_training = actor_model.is_training();
    actor_model.eval();
    for (int epoch = 0; epoch < config.q_epochs; ++epoch) {
      stats.q_epoch_partition_sha256[epoch] =
          q_plans[epoch].canonical_sha256;
      for (const auto& minibatch : q_plans[epoch].minibatches) {
        const auto flat = vrpo_training_internal::FlattenEpisodes(
            episodes, minibatch.episode_indices);
        actor_optimizer.zero_grad();
        q_optimizer.zero_grad();
        torch::Tensor q_output;
        try {
          q_output = q_forward(
              vrpo_training_internal::StackInputs(flat.rows, false));
        } catch (const std::exception& exception) {
          actor_model.train(actor_was_training);
          return fail(std::string("Q-phase forward failed: ") +
                      exception.what());
        }
        if (!vrpo_training_internal::ValidateQOutput(
                q_output, flat.rows.size(), error)) {
          actor_model.train(actor_was_training);
          return false;
        }
        std::vector<torch::Tensor> predicted;
        std::vector<torch::Tensor> targets;
        for (size_t index = 0; index < flat.rows.size(); ++index) {
          const VrpoTrainingRow& row = *flat.rows[index];
          const auto found = target_by_row.find(row.row_id);
          if (found == target_by_row.end() || row.chosen_action < 0 ||
              row.chosen_action >= q_output.size(1)) {
            actor_model.train(actor_was_training);
            return fail("Q regression target/action lookup failed");
          }
          predicted.push_back(q_output[index][row.chosen_action]);
          targets.push_back(torch::tensor(
              std::vector<double>(
                  found->second.q_target_actor_relative.begin(),
                  found->second.q_target_actor_relative.end()),
              torch::TensorOptions().dtype(torch::kFloat32).device(
                  q_output.device())));
        }
        torch::Tensor q_loss = 0.5 * torch::mse_loss(
            torch::stack(predicted), torch::stack(targets));
        if (!vrpo_training_internal::FiniteTensor(q_loss)) {
          actor_model.train(actor_was_training);
          return fail("Q loss is nonfinite before step");
        }
        q_loss.backward();
        ++stats.q_backward_calls;
        double actor_grad_norm = 0.0;
        double q_grad_norm = 0.0;
        bool actor_grad = false;
        bool q_grad = false;
        if (!vrpo_training_internal::GradientNorm(
                actor_model, "", &actor_grad_norm, &actor_grad, error) ||
            !vrpo_training_internal::GradientNorm(
                q_model, "", &q_grad_norm, &q_grad, error)) {
          actor_model.train(actor_was_training);
          return false;
        }
        if (actor_grad || actor_grad_norm != 0.0 || !q_grad ||
            q_grad_norm <= 0.0) {
          actor_model.train(actor_was_training);
          return fail("Q-phase gradient/module gate failed before step");
        }
        stats.max_q_grad_norm =
            std::max(stats.max_q_grad_norm, q_grad_norm);
        torch::nn::utils::clip_grad_norm_(q_model.parameters(),
                                         kVrpoTrainingGradientClipNorm);
        q_optimizer.step();
        ++stats.q_optimizer_steps;
        stats.q_rows_seen += flat.rows.size();
        q_loss_sum += q_loss.detach().item<double>();
        ++q_loss_count;
      }
    }
    actor_model.train(actor_was_training);
  }
  actor_optimizer.zero_grad();
  q_optimizer.zero_grad();

  if (!vrpo_training_internal::ModuleValueSha256(
          actor_model, "", &stats.actor_values_after_sha256, error) ||
      !vrpo_training_internal::ModuleValueSha256(
          q_model, "", &stats.q_values_after_sha256, error) ||
      !vrpo_training_internal::ModuleValueSha256(
          actor_model, "value_head", &stats.value_head_after_sha256,
          error)) {
    return false;
  }
  const int64_t expected_actor_steps =
      config.actor_epochs * kVrpoTrainingMinibatchesPerEpoch;
  const int64_t expected_q_steps =
      config.algorithm == VrpoPhase4Algorithm::kVrpo
          ? config.q_epochs * kVrpoTrainingMinibatchesPerEpoch
          : 0;
  const bool movement_valid =
      stats.actor_values_before_sha256 != stats.actor_values_after_sha256 &&
      (config.algorithm == VrpoPhase4Algorithm::kPpo
           ? stats.q_values_before_sha256 == stats.q_values_after_sha256 &&
                 stats.value_head_before_sha256 !=
                     stats.value_head_after_sha256
           : stats.q_values_before_sha256 != stats.q_values_after_sha256 &&
                 stats.value_head_before_sha256 ==
                     stats.value_head_after_sha256);
  if (stats.actor_optimizer_steps != expected_actor_steps ||
      stats.q_optimizer_steps != expected_q_steps || !movement_valid ||
      actor_loss_count != expected_actor_steps ||
      (config.algorithm == VrpoPhase4Algorithm::kVrpo &&
       q_loss_count != expected_q_steps)) {
    return fail("phase4d step-count or module-movement postcondition failed");
  }
  stats.actor_loss_mean = actor_loss_sum / actor_loss_count;
  stats.q_loss_mean = q_loss_count == 0 ? 0.0 : q_loss_sum / q_loss_count;
  if (!std::isfinite(stats.actor_loss_mean) ||
      !std::isfinite(stats.q_loss_mean) ||
      !std::isfinite(stats.min_ratio) || !std::isfinite(stats.max_ratio) ||
      !std::isfinite(stats.max_full_legal_kl) ||
      !std::isfinite(stats.max_actor_grad_norm) ||
      !std::isfinite(stats.max_q_grad_norm) ||
      !std::isfinite(stats.max_value_head_grad_norm)) {
    return fail("phase4d diagnostics are nonfinite");
  }
  stats.success = true;
  stats.deterministic_summary_sha256 =
      vrpo_training_internal::StatsSha256(stats);
  transaction.Commit();
  *output = std::move(stats);
  return true;
}

}  // namespace open_spiel

#endif  // OPEN_SPIEL_BUILD_WITH_LIBTORCH

#endif  // OPEN_SPIEL_EXAMPLES_DUNE_VRPO_TRAINING_H_
