#ifndef OPEN_SPIEL_EXAMPLES_DUNE_VRPO_CHECKPOINT_H_
#define OPEN_SPIEL_EXAMPLES_DUNE_VRPO_CHECKPOINT_H_

#include <cerrno>
#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include <torch/torch.h>

#include "dune_network.h"
#include "dune_vrpo.h"

namespace open_spiel {

inline constexpr char kVrpoExpandedCheckpointSchema[] =
    "dune_vrpo_phase4b_expanded_checkpoint_v1";

struct VrpoExpandedCheckpointPaths {
  std::filesystem::path directory;
  std::filesystem::path actor_model;
  std::filesystem::path q_model;
  std::filesystem::path actor_optimizer;
  std::filesystem::path q_optimizer;
  std::filesystem::path manifest;
};

struct VrpoExpandedExpectedLayout {
  std::string label;
  bool test_fixture = false;
  int64_t actor_observation_dim = 0;
  int64_t actor_hidden_dim = 0;
  int64_t actor_action_dim = 0;
  int64_t actor_residual_blocks = 0;
  std::string actor_names_shapes_sha256;
  std::string q_names_shapes_sha256;
};

inline VrpoExpandedCheckpointPaths VrpoExpandedPaths(
    const std::filesystem::path& directory) {
  return {directory,
          directory / "actor_model.pt",
          directory / "q_model.pt",
          directory / "actor_optimizer.pt",
          directory / "q_optimizer.pt",
          directory / "manifest.json"};
}

enum class VrpoCheckpointFailurePoint {
  kNone,
  kAfterActorTemp,
  kAfterQTemp,
  kAfterActorOptimizerTemp,
  kAfterQOptimizerTemp,
  kAfterManifestTemp,
  kAfterActorRename,
  kAfterQRename,
  kAfterActorOptimizerRename,
  kAfterQOptimizerRename,
};

inline bool VrpoFsyncFile(const std::filesystem::path& path,
                          std::string* error) {
  const int fd = ::open(path.c_str(), O_RDONLY);
  if (fd < 0) {
    if (error != nullptr) *error = "cannot open file for fsync: " + path.string();
    return false;
  }
  const int result = ::fsync(fd);
  ::close(fd);
  if (result != 0) {
    if (error != nullptr) *error = "fsync failed: " + path.string();
    return false;
  }
  return true;
}

inline bool VrpoFsyncDirectory(const std::filesystem::path& path,
                               std::string* error) {
  const int fd = ::open(path.c_str(), O_RDONLY | O_DIRECTORY);
  if (fd < 0) {
    if (error != nullptr) *error = "cannot open directory for fsync";
    return false;
  }
  const int result = ::fsync(fd);
  ::close(fd);
  if (result != 0) {
    if (error != nullptr) *error = "directory fsync failed";
    return false;
  }
  return true;
}

inline void CleanupVrpoExpandedDirectory(
    const VrpoExpandedCheckpointPaths& paths) {
  const std::array<std::filesystem::path, 10> files = {
      paths.actor_model, paths.q_model, paths.actor_optimizer,
      paths.q_optimizer, paths.manifest,
      std::filesystem::path(paths.actor_model.string() + ".tmp"),
      std::filesystem::path(paths.q_model.string() + ".tmp"),
      std::filesystem::path(paths.actor_optimizer.string() + ".tmp"),
      std::filesystem::path(paths.q_optimizer.string() + ".tmp"),
      std::filesystem::path(paths.manifest.string() + ".tmp")};
  std::error_code ec;
  for (const auto& file : files) std::filesystem::remove(file, ec);
  std::filesystem::remove(paths.directory, ec);
  const auto parent = paths.directory.parent_path();
  if (!parent.empty() && std::filesystem::is_directory(parent)) {
    std::string ignored;
    VrpoFsyncDirectory(parent, &ignored);
  }
}

struct VrpoFreshOptimizers {
  std::unique_ptr<torch::optim::AdamW> actor;
  std::unique_ptr<torch::optim::AdamW> q;
  std::vector<std::string> actor_policy_names;
  std::vector<std::string> actor_trunk_value_names;
  std::vector<std::string> q_names;
  std::vector<std::string> actor_group_names;
  std::vector<std::string> q_group_names;
};

inline void MaterializeVrpoZeroAdamWState(torch::optim::AdamW& optimizer) {
  torch::NoGradGuard no_grad;
  for (auto& group : optimizer.param_groups()) {
    for (auto& parameter : group.params()) {
      auto key = parameter.unsafeGetTensorImpl();
      auto state = std::make_unique<torch::optim::AdamWParamState>();
      state->step(0);
      state->exp_avg(torch::zeros_like(
          parameter, torch::MemoryFormat::Preserve));
      state->exp_avg_sq(torch::zeros_like(
          parameter, torch::MemoryFormat::Preserve));
      optimizer.state()[key] = std::move(state);
    }
  }
}

inline bool MakeVrpoFreshOptimizers(
    SharedDunePolicyValueNetImpl& actor_model, torch::nn::Module& q_model,
    VrpoFreshOptimizers* output, std::string* error) {
  auto fail = [&](const std::string& message) {
    if (error != nullptr) *error = message;
    if (output != nullptr) *output = VrpoFreshOptimizers{};
    return false;
  };
  if (output == nullptr) return fail("null fresh optimizer output");
  *output = VrpoFreshOptimizers{};
  std::vector<torch::Tensor> policy;
  std::vector<torch::Tensor> trunk_value;
  std::set<void*> covered;
  for (const auto& item : actor_model.named_parameters()) {
    const bool is_policy = item.key().rfind("policy_head", 0) == 0;
    auto* identity = static_cast<void*>(item.value().unsafeGetTensorImpl());
    if (!covered.insert(identity).second) {
      return fail("actor parameter is duplicated");
    }
    if (is_policy) {
      policy.push_back(item.value());
      output->actor_policy_names.push_back(item.key());
    } else {
      trunk_value.push_back(item.value());
      output->actor_trunk_value_names.push_back(item.key());
    }
  }
  if (policy.empty() || trunk_value.empty() ||
      covered.size() != actor_model.named_parameters().size()) {
    return fail("actor optimizer coverage is incomplete");
  }
  std::vector<torch::Tensor> q_parameters;
  std::set<void*> q_covered;
  for (const auto& item : q_model.named_parameters()) {
    auto* identity = static_cast<void*>(item.value().unsafeGetTensorImpl());
    if (!q_covered.insert(identity).second || covered.count(identity)) {
      return fail("Q parameter coverage overlaps or duplicates actor");
    }
    q_parameters.push_back(item.value());
    output->q_names.push_back(item.key());
  }
  if (q_parameters.empty() ||
      q_covered.size() != q_model.named_parameters().size()) {
    return fail("Q optimizer coverage is incomplete");
  }
  const auto specs = CanonicalVrpoPhase4OptimizerGroups();
  output->actor_group_names = {specs[0].group_name, specs[1].group_name};
  output->q_group_names = {specs[2].group_name};
  std::vector<torch::optim::OptimizerParamGroup> actor_groups;
  actor_groups.emplace_back(policy);
  actor_groups.emplace_back(trunk_value);
  output->actor = std::make_unique<torch::optim::AdamW>(
      actor_groups, torch::optim::AdamWOptions(specs[0].learning_rate)
                        .betas({specs[0].beta1, specs[0].beta2})
                        .eps(specs[0].epsilon));
  for (size_t group = 0; group < actor_groups.size(); ++group) {
    auto& options = static_cast<torch::optim::AdamWOptions&>(
        output->actor->param_groups()[group].options());
    options.lr(specs[group].learning_rate);
    options.betas({specs[group].beta1, specs[group].beta2});
    options.eps(specs[group].epsilon);
    options.weight_decay(specs[group].weight_decay);
  }
  output->q = std::make_unique<torch::optim::AdamW>(
      q_parameters, torch::optim::AdamWOptions(specs[2].learning_rate)
                        .betas({specs[2].beta1, specs[2].beta2})
                        .eps(specs[2].epsilon)
                        .weight_decay(specs[2].weight_decay));
  MaterializeVrpoZeroAdamWState(*output->actor);
  MaterializeVrpoZeroAdamWState(*output->q);
  return true;
}

inline bool ValidateVrpoOptimizerGroupsAndZeroState(
    const VrpoFreshOptimizers& optimizers,
    SharedDunePolicyValueNetImpl& actor_model, torch::nn::Module& q_model,
    std::string* zero_state_identity, std::string* error) {
  auto fail = [&](const std::string& message) {
    if (error != nullptr) *error = message;
    if (zero_state_identity != nullptr) zero_state_identity->clear();
    return false;
  };
  if (optimizers.actor == nullptr || optimizers.q == nullptr ||
      zero_state_identity == nullptr) {
    return fail("optimizer validation input is incomplete");
  }
  const auto specs = CanonicalVrpoPhase4OptimizerGroups();
  if (optimizers.actor->param_groups().size() != 2 ||
      optimizers.q->param_groups().size() != 1) {
    return fail("optimizer group count differs from contract");
  }
  auto check_options = [&](const torch::optim::OptimizerParamGroup& group,
                           const VrpoOptimizerGroupSpec& spec) {
    const auto& options = static_cast<const torch::optim::AdamWOptions&>(
        group.options());
    return options.lr() == spec.learning_rate &&
        std::get<0>(options.betas()) == spec.beta1 &&
        std::get<1>(options.betas()) == spec.beta2 &&
        options.eps() == spec.epsilon &&
        options.weight_decay() == spec.weight_decay;
  };
  if (!check_options(optimizers.actor->param_groups()[0], specs[0]) ||
      !check_options(optimizers.actor->param_groups()[1], specs[1]) ||
      !check_options(optimizers.q->param_groups()[0], specs[2])) {
    return fail("optimizer hyperparameters differ from contract");
  }
  if (optimizers.actor_group_names !=
          std::vector<std::string>({specs[0].group_name,
                                    specs[1].group_name}) ||
      optimizers.q_group_names !=
          std::vector<std::string>({specs[2].group_name})) {
    return fail("optimizer group names/order differ from contract");
  }
  std::map<c10::TensorImpl*, std::string> actor_names;
  std::map<c10::TensorImpl*, std::string> q_names;
  std::vector<std::string> expected_policy;
  std::vector<std::string> expected_trunk_value;
  std::vector<std::string> expected_q;
  for (const auto& item : actor_model.named_parameters()) {
    actor_names[item.value().unsafeGetTensorImpl()] = item.key();
    if (item.key().rfind("policy_head", 0) == 0) {
      expected_policy.push_back(item.key());
    } else {
      expected_trunk_value.push_back(item.key());
    }
  }
  for (const auto& item : q_model.named_parameters()) {
    q_names[item.value().unsafeGetTensorImpl()] = item.key();
    expected_q.push_back(item.key());
  }
  if (optimizers.actor_policy_names != expected_policy ||
      optimizers.actor_trunk_value_names != expected_trunk_value ||
      optimizers.q_names != expected_q) {
    return fail("optimizer declared parameter memberships differ from modules");
  }
  auto group_names = [&](const torch::optim::OptimizerParamGroup& group,
                         const std::map<c10::TensorImpl*, std::string>& names,
                         std::set<c10::TensorImpl*>* seen,
                         std::vector<std::string>* actual) {
    actual->clear();
    for (const auto& parameter : group.params()) {
      c10::TensorImpl* identity = parameter.unsafeGetTensorImpl();
      const auto name = names.find(identity);
      if (name == names.end() || !seen->insert(identity).second) return false;
      actual->push_back(name->second);
    }
    return true;
  };
  std::set<c10::TensorImpl*> actor_seen;
  std::set<c10::TensorImpl*> q_seen;
  std::vector<std::string> actual;
  if (!group_names(optimizers.actor->param_groups()[0], actor_names,
                   &actor_seen, &actual) ||
      actual != expected_policy ||
      !group_names(optimizers.actor->param_groups()[1], actor_names,
                   &actor_seen, &actual) ||
      actual != expected_trunk_value || actor_seen.size() != actor_names.size() ||
      !group_names(optimizers.q->param_groups()[0], q_names,
                   &q_seen, &actual) ||
      actual != expected_q || q_seen.size() != q_names.size()) {
    return fail("optimizer actual tensor membership/order/coverage is invalid");
  }
  for (c10::TensorImpl* identity : actor_seen) {
    if (q_seen.count(identity)) {
      return fail("actor and Q optimizer memberships overlap");
    }
  }
  auto check_zero = [&](const torch::optim::AdamW& optimizer,
                        const std::set<c10::TensorImpl*>& expected_keys) {
    std::set<c10::TensorImpl*> actual_keys;
    for (const auto& entry : optimizer.state()) {
      actual_keys.insert(static_cast<c10::TensorImpl*>(entry.first));
    }
    if (actual_keys != expected_keys) return false;
    for (c10::TensorImpl* key : expected_keys) {
      const auto it = optimizer.state().find(key);
      if (it == optimizer.state().end()) return false;
      const auto* state = dynamic_cast<const torch::optim::AdamWParamState*>(
          it->second.get());
      if (state == nullptr || state->step() != 0 ||
          !state->exp_avg().defined() || !state->exp_avg_sq().defined() ||
          state->exp_avg().sizes() != key->sizes() ||
          state->exp_avg_sq().sizes() != key->sizes() ||
          state->exp_avg().abs().max().item<double>() != 0.0 ||
          state->exp_avg_sq().abs().max().item<double>() != 0.0) {
        return false;
      }
    }
    return true;
  };
  if (!check_zero(*optimizers.actor, actor_seen) ||
      !check_zero(*optimizers.q, q_seen)) {
    return fail("optimizer state key set/zero values are invalid");
  }
  std::vector<VrpoNamedParameterIdentity> actor_identities;
  std::vector<VrpoNamedParameterIdentity> q_identities;
  if (!VrpoNamedParameterIdentities(
          actor_model, nullptr, &actor_identities, error) ||
      !VrpoNamedParameterIdentities(q_model, nullptr, &q_identities, error)) {
    return false;
  }
  *zero_state_identity = VrpoOptimizerZeroStateIdentitySha256(
      specs, actor_identities, q_identities);
  return true;
}

struct VrpoExpandedFileIdentity {
  std::string filename;
  int64_t size = 0;
  std::string sha256;
};

struct VrpoExpandedArchiveIdentity {
  std::array<VrpoExpandedFileIdentity, 5> files;
  std::string combined_sha256;
};

inline VrpoExpandedFileIdentity VrpoFileIdentity(
    const std::filesystem::path& path) {
  size_t size = 0;
  const std::string digest = ComputeFileSHA256(path.string(), &size);
  return {path.filename().string(), static_cast<int64_t>(size), digest};
}

inline bool ComputeVrpoExpandedArchiveIdentity(
    const std::filesystem::path& directory,
    VrpoExpandedArchiveIdentity* output, std::string* error) {
  if (output == nullptr) {
    if (error != nullptr) *error = "null expanded archive identity output";
    return false;
  }
  *output = VrpoExpandedArchiveIdentity{};
  const auto paths = VrpoExpandedPaths(directory);
  const std::array<std::filesystem::path, 5> files = {
      paths.actor_model, paths.q_model, paths.actor_optimizer,
      paths.q_optimizer, paths.manifest};
  std::string payload = "dune_vrpo_phase4b_archive_identity_v1";
  try {
    for (size_t index = 0; index < files.size(); ++index) {
      if (!std::filesystem::is_regular_file(files[index]) ||
          std::filesystem::is_symlink(files[index])) {
        if (error != nullptr) *error = "expanded archive file is missing";
        return false;
      }
      output->files[index] = VrpoFileIdentity(files[index]);
      payload.append(output->files[index].filename);
      payload.push_back('\0');
      vrpo_capture_internal::AppendPod(
          &payload, output->files[index].size);
      payload.append(output->files[index].sha256);
    }
  } catch (const std::exception& exception) {
    if (error != nullptr) *error = exception.what();
    return false;
  }
  output->combined_sha256 = ComputeStringSHA256(payload);
  return true;
}

inline void SaveVrpoModule(torch::nn::Module& module,
                           const std::filesystem::path& path) {
  torch::serialize::OutputArchive archive;
  module.save(archive);
  archive.save_to(path.string());
}

inline void LoadVrpoModule(torch::nn::Module& module,
                           const std::filesystem::path& path) {
  torch::serialize::InputArchive archive;
  archive.load_from(path.string(), torch::kCPU);
  module.load(archive);
}

inline json::Object BuildVrpoExpandedManifest(
    const VrpoPhase4ArmConfig& arm,
    const VrpoPhase4ManifestBinding& binding,
    const VrpoExpandedExpectedLayout& serialized_layout,
    const std::string& checkpoint_uuid, int64_t global_update,
    uint64_t next_episode_id,
    const VrpoExpandedFileIdentity& actor_model,
    const VrpoExpandedFileIdentity& q_model,
    const VrpoExpandedFileIdentity& actor_optimizer,
    const VrpoExpandedFileIdentity& q_optimizer,
    const std::string& actor_values_sha256,
    const std::string& q_values_sha256,
    const std::string& optimizer_zero_state_sha256) {
  json::Object out;
  out["schema"] = kVrpoExpandedCheckpointSchema;
  out["phase4_contract"] = json::Value(BuildVrpoPhase4Manifest(arm, binding));
  out["checkpoint_uuid"] = checkpoint_uuid;
  out["global_update"] = global_update;
  out["next_episode_id"] = static_cast<int64_t>(next_episode_id);
  out["source_optimizer_moments_loaded"] = false;
  out["actor_values_sha256"] = actor_values_sha256;
  out["q_values_sha256"] = q_values_sha256;
  out["optimizer_zero_state_sha256"] = optimizer_zero_state_sha256;
  out["serialized_layout_label"] = serialized_layout.label;
  out["serialized_layout_test_fixture"] = serialized_layout.test_fixture;
  out["serialized_actor_observation_dim"] =
      serialized_layout.actor_observation_dim;
  out["serialized_actor_hidden_dim"] = serialized_layout.actor_hidden_dim;
  out["serialized_actor_action_dim"] = serialized_layout.actor_action_dim;
  out["serialized_actor_residual_blocks"] =
      serialized_layout.actor_residual_blocks;
  out["serialized_actor_names_shapes_sha256"] =
      serialized_layout.actor_names_shapes_sha256;
  out["serialized_q_names_shapes_sha256"] =
      serialized_layout.q_names_shapes_sha256;
  const std::array<std::pair<std::string, VrpoExpandedFileIdentity>, 4> files = {{
      {"actor_model", actor_model}, {"q_model", q_model},
      {"actor_optimizer", actor_optimizer}, {"q_optimizer", q_optimizer}}};
  for (const auto& entry : files) {
    out[entry.first + "_filename"] = entry.second.filename;
    out[entry.first + "_size"] = entry.second.size;
    out[entry.first + "_sha256"] = entry.second.sha256;
  }
  return out;
}

inline bool VrpoExpandedManifestStrictEqual(
    const json::Object& actual, const json::Object& expected,
    std::string* error) {
  if (actual.size() != expected.size()) {
    if (error != nullptr) *error = "expanded manifest missing/extra fields";
    return false;
  }
  for (const auto& field : expected) {
    const auto it = actual.find(field.first);
    if (it == actual.end() ||
        !VrpoJsonValuesExactlyEqual(it->second, field.second)) {
      if (error != nullptr) {
        *error = "expanded manifest mismatch: " + field.first;
      }
      return false;
    }
  }
  return true;
}

inline bool VrpoCheckpointUuidValid(const std::string& uuid) {
  if (uuid.size() != 36) return false;
  for (size_t i = 0; i < uuid.size(); ++i) {
    if (i == 8 || i == 13 || i == 18 || i == 23) {
      if (uuid[i] != '-') return false;
    } else if (!((uuid[i] >= '0' && uuid[i] <= '9') ||
                 (uuid[i] >= 'a' && uuid[i] <= 'f'))) {
      return false;
    }
  }
  return true;
}

inline bool ValidateVrpoExpandedLiveLayout(
    SharedDunePolicyValueNetImpl& actor_model, torch::nn::Module& q_model,
    const VrpoExpandedExpectedLayout& expected,
    const VrpoPhase4ManifestBinding& binding, std::string* error) {
  auto fail = [&](const std::string& message) {
    if (error != nullptr) *error = message;
    return false;
  };
  std::vector<VrpoNamedParameterIdentity> actor_identities;
  std::vector<VrpoNamedParameterIdentity> q_identities;
  if (!VrpoNamedParameterIdentities(
          actor_model, nullptr, &actor_identities, error) ||
      !VrpoNamedParameterIdentities(q_model, nullptr, &q_identities, error)) {
    return false;
  }
  const std::string actor_layout =
      VrpoNamedParameterIdentitySha256(actor_identities, false);
  const std::string q_layout =
      VrpoNamedParameterIdentitySha256(q_identities, false);
  const bool actor_shape =
      actor_model.input_layer->weight.size(1) ==
          expected.actor_observation_dim &&
      actor_model.input_layer->weight.size(0) == expected.actor_hidden_dim &&
      actor_model.policy_head->weight.size(0) == expected.actor_action_dim &&
      static_cast<int64_t>(actor_model.res_blocks.size()) ==
          expected.actor_residual_blocks;
  if (expected.label.empty() || !actor_shape ||
      actor_layout != expected.actor_names_shapes_sha256 ||
      q_layout != expected.q_names_shapes_sha256) {
    return fail("serialized live module layout differs from expected layout");
  }
  if (binding.actor_names_shapes_sha256 != actor_layout ||
      binding.q_names_shapes_sha256 != q_layout) {
    return fail("serialized live module layout differs from registered binding");
  }
  if (!expected.test_fixture &&
      (expected.label != "production_dune_vrpo_layout_v1" ||
       expected.actor_observation_dim != kVrpoDuneInformationStateSize ||
       expected.actor_hidden_dim != 2048 ||
       expected.actor_action_dim != kVrpoDuneActionDim ||
       expected.actor_residual_blocks != 8 ||
       binding.actor_observation_dim != expected.actor_observation_dim ||
       binding.actor_hidden_dim != expected.actor_hidden_dim ||
       binding.actor_action_dim != expected.actor_action_dim ||
       binding.actor_residual_blocks != expected.actor_residual_blocks)) {
    return fail("production archive does not match pinned Dune actor layout");
  }
  if (expected.test_fixture && expected.label != "tiny_test_fixture_v1") {
    return fail("test fixture layout label is invalid");
  }
  return true;
}

inline bool SaveVrpoExpandedCheckpointAtomic(
    const std::filesystem::path& directory,
    const VrpoPhase4ArmConfig& arm,
    const VrpoPhase4ManifestBinding& binding,
    const VrpoExpandedExpectedLayout& expected_layout,
    const std::string& checkpoint_uuid, int64_t global_update,
    uint64_t next_episode_id,
    std::shared_ptr<SharedDunePolicyValueNetImpl> actor_model,
    std::shared_ptr<torch::nn::Module> q_model,
    VrpoFreshOptimizers& optimizers,
    VrpoCheckpointFailurePoint failure_point, std::string* error,
    VrpoExpandedArchiveIdentity* archive_identity = nullptr) {
  auto fail = [&](const std::string& message,
                  const VrpoExpandedCheckpointPaths& paths) {
    if (error != nullptr) *error = message;
    CleanupVrpoExpandedDirectory(paths);
    return false;
  };
  auto current_error = [&](const std::string& fallback) {
    return error != nullptr && !error->empty() ? *error : fallback;
  };
  const VrpoExpandedCheckpointPaths paths = VrpoExpandedPaths(directory);
  if (std::filesystem::exists(directory)) {
    if (error != nullptr) *error = "expanded checkpoint directory already exists";
    return false;
  }
  if (actor_model == nullptr || q_model == nullptr ||
      optimizers.actor == nullptr || optimizers.q == nullptr ||
      !VrpoCheckpointUuidValid(checkpoint_uuid) || global_update != 0 ||
      next_episode_id != binding.start_episode_id) {
    if (error != nullptr) *error = "expanded checkpoint save input is invalid";
    return false;
  }
  std::string contract_error;
  if (!ValidateVrpoPhase4ArmConfig(arm, &contract_error) ||
      !ValidateVrpoPhase4ManifestBinding(binding, &contract_error) ||
      !ValidateVrpoExpandedLiveLayout(
          *actor_model, *q_model, expected_layout, binding,
          &contract_error)) {
    if (error != nullptr) *error = contract_error;
    return false;
  }
  std::error_code ec;
  if (!std::filesystem::create_directories(directory, ec) || ec) {
    if (error != nullptr) *error = "cannot create fresh checkpoint directory";
    return false;
  }
  if (!directory.parent_path().empty() &&
      !VrpoFsyncDirectory(directory.parent_path(), error)) {
    return fail(current_error("parent directory fsync after creation failed"),
                paths);
  }
  const auto actor_tmp = std::filesystem::path(paths.actor_model.string() + ".tmp");
  const auto q_tmp = std::filesystem::path(paths.q_model.string() + ".tmp");
  const auto actor_optimizer_tmp =
      std::filesystem::path(paths.actor_optimizer.string() + ".tmp");
  const auto q_optimizer_tmp =
      std::filesystem::path(paths.q_optimizer.string() + ".tmp");
  const auto manifest_tmp =
      std::filesystem::path(paths.manifest.string() + ".tmp");
  try {
    SaveVrpoModule(*actor_model, actor_tmp);
    if (!VrpoFsyncFile(actor_tmp, error)) {
      return fail(current_error("actor temp fsync failed"), paths);
    }
    if (failure_point == VrpoCheckpointFailurePoint::kAfterActorTemp) {
      return fail("injected failure after actor temp", paths);
    }
    SaveVrpoModule(*q_model, q_tmp);
    if (!VrpoFsyncFile(q_tmp, error)) {
      return fail(current_error("Q temp fsync failed"), paths);
    }
    if (failure_point == VrpoCheckpointFailurePoint::kAfterQTemp) {
      return fail("injected failure after Q temp", paths);
    }
    torch::save(*optimizers.actor, actor_optimizer_tmp.string());
    if (!VrpoFsyncFile(actor_optimizer_tmp, error)) {
      return fail(current_error("actor optimizer temp fsync failed"), paths);
    }
    if (failure_point == VrpoCheckpointFailurePoint::kAfterActorOptimizerTemp) {
      return fail("injected failure after actor optimizer temp", paths);
    }
    torch::save(*optimizers.q, q_optimizer_tmp.string());
    if (!VrpoFsyncFile(q_optimizer_tmp, error)) {
      return fail(current_error("Q optimizer temp fsync failed"), paths);
    }
    if (failure_point == VrpoCheckpointFailurePoint::kAfterQOptimizerTemp) {
      return fail("injected failure after Q optimizer temp", paths);
    }

    std::vector<VrpoNamedParameterIdentity> actor_identities;
    std::vector<VrpoNamedParameterIdentity> q_identities;
    if (!VrpoNamedParameterIdentities(
            *actor_model, nullptr, &actor_identities, error) ||
        !VrpoNamedParameterIdentities(
            *q_model, nullptr, &q_identities, error)) {
      return fail(error == nullptr ? "parameter identity failed" : *error,
                  paths);
    }
    const std::string q_parameter_hash =
        VrpoNamedParameterIdentitySha256(q_identities, true);
    std::string zero_state_hash;
    if (!ValidateVrpoOptimizerGroupsAndZeroState(
            optimizers, *actor_model, *q_model, &zero_state_hash, error)) {
      return fail(error == nullptr ? "optimizer identity failed" : *error,
                  paths);
    }
    const std::string actor_value_hash =
        VrpoNamedParameterIdentitySha256(actor_identities, true);
    const std::string actor_layout_hash =
        VrpoNamedParameterIdentitySha256(actor_identities, false);
    const std::string q_layout_hash =
        VrpoNamedParameterIdentitySha256(q_identities, false);
    std::string layout_payload = "dune_vrpo_phase4_module_layout_v1";
    layout_payload.push_back('\0');
    layout_payload.append(actor_layout_hash);
    layout_payload.append(q_layout_hash);
    const std::string module_layout_hash = ComputeStringSHA256(layout_payload);
    const std::string optimizer_groups_hash =
        VrpoOptimizerGroupSpecSha256(CanonicalVrpoPhase4OptimizerGroups());
    std::string binding_error;
    if (binding.actor_subset_sha256 != actor_value_hash ||
        binding.actor_names_shapes_sha256 != actor_layout_hash ||
        binding.q_init_sha256 != q_parameter_hash ||
        binding.q_names_shapes_sha256 != q_layout_hash ||
        binding.module_layout_sha256 != module_layout_hash ||
        binding.optimizer_groups_sha256 != optimizer_groups_hash ||
        binding.optimizer_zero_state_sha256 != zero_state_hash) {
      return fail("live module/optimizer identities differ from registered binding",
                  paths);
    }
    if (!ValidateVrpoExpandedLiveLayout(
            *actor_model, *q_model, expected_layout, binding,
            &binding_error)) {
      return fail(binding_error, paths);
    }
    if (!ValidateVrpoPhase4ManifestBinding(binding, &binding_error)) {
      return fail(binding_error, paths);
    }
    VrpoExpandedFileIdentity actor_file = VrpoFileIdentity(actor_tmp);
    VrpoExpandedFileIdentity q_file = VrpoFileIdentity(q_tmp);
    VrpoExpandedFileIdentity actor_optimizer_file =
        VrpoFileIdentity(actor_optimizer_tmp);
    VrpoExpandedFileIdentity q_optimizer_file =
        VrpoFileIdentity(q_optimizer_tmp);
    actor_file.filename = paths.actor_model.filename().string();
    q_file.filename = paths.q_model.filename().string();
    actor_optimizer_file.filename =
        paths.actor_optimizer.filename().string();
    q_optimizer_file.filename = paths.q_optimizer.filename().string();
    const json::Object manifest = BuildVrpoExpandedManifest(
        arm, binding, expected_layout, checkpoint_uuid, global_update,
        next_episode_id,
        actor_file, q_file, actor_optimizer_file, q_optimizer_file,
        binding.actor_subset_sha256, binding.q_init_sha256,
        zero_state_hash);
    {
      std::ofstream stream(manifest_tmp, std::ios::trunc);
      if (!stream) return fail("cannot write expanded manifest temp", paths);
      stream << json::ToString(manifest, true) << "\n";
      stream.flush();
      if (!stream) return fail("expanded manifest flush failed", paths);
    }
    if (!VrpoFsyncFile(manifest_tmp, error)) {
      return fail(current_error("manifest temp fsync failed"), paths);
    }
    if (failure_point == VrpoCheckpointFailurePoint::kAfterManifestTemp) {
      return fail("injected failure after manifest temp", paths);
    }

    std::filesystem::rename(actor_tmp, paths.actor_model);
    if (failure_point == VrpoCheckpointFailurePoint::kAfterActorRename) {
      return fail("injected failure after actor rename", paths);
    }
    std::filesystem::rename(q_tmp, paths.q_model);
    if (failure_point == VrpoCheckpointFailurePoint::kAfterQRename) {
      return fail("injected failure after Q rename", paths);
    }
    std::filesystem::rename(actor_optimizer_tmp, paths.actor_optimizer);
    if (failure_point ==
        VrpoCheckpointFailurePoint::kAfterActorOptimizerRename) {
      return fail("injected failure after actor optimizer rename", paths);
    }
    std::filesystem::rename(q_optimizer_tmp, paths.q_optimizer);
    if (failure_point == VrpoCheckpointFailurePoint::kAfterQOptimizerRename) {
      return fail("injected failure after Q optimizer rename", paths);
    }
    if (!VrpoFsyncDirectory(directory, error)) {
      return fail(current_error("checkpoint directory fsync failed"), paths);
    }
    if (!directory.parent_path().empty() &&
        !VrpoFsyncDirectory(directory.parent_path(), error)) {
      return fail(current_error("parent directory publish fsync failed"),
                  paths);
    }
    std::filesystem::rename(manifest_tmp, paths.manifest);
    if (!VrpoFsyncDirectory(directory, error)) {
      return fail(current_error("checkpoint directory commit fsync failed"),
                  paths);
    }
    if (!directory.parent_path().empty() &&
        !VrpoFsyncDirectory(directory.parent_path(), error)) {
      return fail(current_error("parent directory commit fsync failed"),
                  paths);
    }
    if (archive_identity != nullptr &&
        !ComputeVrpoExpandedArchiveIdentity(
            directory, archive_identity, error)) {
      return fail(current_error("archive identity failed"), paths);
    }
    return true;
  } catch (const std::exception& exception) {
    return fail(std::string("expanded checkpoint save failed: ") +
                    exception.what(),
                paths);
  }
}

inline bool ReadVrpoExpandedManifest(
    const std::filesystem::path& directory, json::Object* manifest,
    std::string* error) {
  if (manifest == nullptr) {
    if (error != nullptr) *error = "null expanded manifest output";
    return false;
  }
  manifest->clear();
  const auto paths = VrpoExpandedPaths(directory);
  if (!std::filesystem::is_regular_file(paths.manifest)) {
    if (error != nullptr) *error = "expanded checkpoint has no accepted manifest";
    return false;
  }
  std::ifstream stream(paths.manifest);
  std::string text((std::istreambuf_iterator<char>(stream)),
                   std::istreambuf_iterator<char>());
  auto parsed = json::FromString(text);
  if (!parsed.has_value() || !parsed->IsObject()) {
    if (error != nullptr) *error = "expanded manifest is malformed JSON";
    return false;
  }
  *manifest = parsed->GetObject();
  return true;
}

inline bool ValidateVrpoExpandedManifestSet(
    const std::array<json::Object, 4>& expanded,
    const std::array<VrpoPhase4ArmConfig, 4>& arms,
    const VrpoPhase4ManifestBinding& binding, std::string* error) {
  std::array<json::Object, 4> phase4;
  for (size_t arm = 0; arm < expanded.size(); ++arm) {
    const auto contract = expanded[arm].find("phase4_contract");
    if (contract == expanded[arm].end() || !contract->second.IsObject()) {
      if (error != nullptr) *error = "expanded set missing phase4 contract";
      return false;
    }
    phase4[arm] = contract->second.GetObject();
    if (!ValidateVrpoPhase4ManifestStrict(
            phase4[arm], arms[arm], binding, error)) {
      return false;
    }
  }
  return ValidateVrpoPhase4ManifestSetMatched(phase4, error);
}

inline bool LoadAndValidateVrpoExpandedCheckpoint(
    const std::filesystem::path& directory,
    const VrpoPhase4ArmConfig& expected_arm,
    const VrpoPhase4ManifestBinding& expected_binding,
    const VrpoExpandedExpectedLayout& expected_layout,
    std::shared_ptr<SharedDunePolicyValueNetImpl> actor_model,
    std::shared_ptr<torch::nn::Module> q_model,
    VrpoFreshOptimizers& optimizers, json::Object* loaded_manifest,
    std::string* error,
    VrpoExpandedArchiveIdentity* archive_identity = nullptr) {
  auto fail = [&](const std::string& message) {
    if (error != nullptr) *error = message;
    if (loaded_manifest != nullptr) loaded_manifest->clear();
    return false;
  };
  if (actor_model == nullptr || q_model == nullptr ||
      optimizers.actor == nullptr || optimizers.q == nullptr ||
      loaded_manifest == nullptr) {
    return fail("expanded load target is incomplete");
  }
  const auto paths = VrpoExpandedPaths(directory);
  const std::set<std::string> expected_names = {
      paths.actor_model.filename().string(), paths.q_model.filename().string(),
      paths.actor_optimizer.filename().string(),
      paths.q_optimizer.filename().string(), paths.manifest.filename().string()};
  std::set<std::string> actual_names;
  std::error_code ec;
  if (!std::filesystem::is_directory(directory)) {
    return fail("expanded checkpoint directory is missing");
  }
  for (const auto& entry : std::filesystem::directory_iterator(directory)) {
    if (!entry.is_regular_file() || entry.is_symlink()) {
      return fail("expanded checkpoint contains a non-regular file");
    }
    actual_names.insert(entry.path().filename().string());
  }
  if (actual_names != expected_names) {
    return fail("expanded checkpoint file set is missing, extra, or partial");
  }
  VrpoExpandedArchiveIdentity actual_archive_identity;
  if (!ComputeVrpoExpandedArchiveIdentity(
          directory, &actual_archive_identity, error)) {
    return false;
  }
  json::Object manifest;
  if (!ReadVrpoExpandedManifest(directory, &manifest, error)) return false;
  const auto contract = manifest.find("phase4_contract");
  if (contract == manifest.end() || !contract->second.IsObject()) {
    return fail("expanded phase4 contract is missing or malformed");
  }
  auto require_string = [&](const std::string& key,
                            const std::string& expected) {
    const auto it = manifest.find(key);
    return it != manifest.end() && it->second.IsString() &&
        it->second.GetString() == expected;
  };
  auto require_int = [&](const std::string& key, int64_t expected) {
    const auto it = manifest.find(key);
    return it != manifest.end() && it->second.IsInt() &&
        it->second.GetInt() == expected;
  };
  const auto uuid_it = manifest.find("checkpoint_uuid");
  const auto next_it = manifest.find("next_episode_id");
  const auto source_moments_it =
      manifest.find("source_optimizer_moments_loaded");
  if (!require_string("schema", kVrpoExpandedCheckpointSchema) ||
      !require_int("global_update", 0) || uuid_it == manifest.end() ||
      !uuid_it->second.IsString() ||
      !VrpoCheckpointUuidValid(uuid_it->second.GetString()) ||
      next_it == manifest.end() || !next_it->second.IsInt() ||
      next_it->second.GetInt() !=
          static_cast<int64_t>(expected_binding.start_episode_id) ||
      source_moments_it == manifest.end() ||
      !source_moments_it->second.IsBool() ||
      source_moments_it->second.GetBool()) {
    return fail("expanded checkpoint update/UUID/next-ID/source-moment contract is invalid");
  }
  const std::array<std::pair<std::string, std::filesystem::path>, 4> files = {{
      {"actor_model", paths.actor_model}, {"q_model", paths.q_model},
      {"actor_optimizer", paths.actor_optimizer},
      {"q_optimizer", paths.q_optimizer}}};
  for (const auto& entry : files) {
    const VrpoExpandedFileIdentity identity = VrpoFileIdentity(entry.second);
    if (!require_string(entry.first + "_filename", identity.filename) ||
        !require_int(entry.first + "_size", identity.size) ||
        !require_string(entry.first + "_sha256", identity.sha256)) {
      return fail("expanded checkpoint file identity mismatch: " + entry.first);
    }
  }
  try {
    LoadVrpoModule(*actor_model, paths.actor_model);
    LoadVrpoModule(*q_model, paths.q_model);
    torch::load(*optimizers.actor, paths.actor_optimizer.string());
    torch::load(*optimizers.q, paths.q_optimizer.string());
  } catch (const std::exception& exception) {
    return fail(std::string("expanded checkpoint deserialize failed: ") +
                exception.what());
  }
  std::vector<VrpoNamedParameterIdentity> actor_identities;
  std::vector<VrpoNamedParameterIdentity> q_identities;
  if (!VrpoNamedParameterIdentities(
          *actor_model, nullptr, &actor_identities, error) ||
      !VrpoNamedParameterIdentities(
          *q_model, nullptr, &q_identities, error)) {
    return fail(error == nullptr ? "expanded parameter identity failed" : *error);
  }
  const std::string q_values_hash =
      VrpoNamedParameterIdentitySha256(q_identities, true);
  const std::string actor_values_hash =
      VrpoNamedParameterIdentitySha256(actor_identities, true);
  std::string zero_state_hash;
  if (!ValidateVrpoOptimizerGroupsAndZeroState(
          optimizers, *actor_model, *q_model, &zero_state_hash, error)) {
    return fail(error == nullptr ? "expanded optimizer validation failed" : *error);
  }
  const std::string actual_optimizer_groups_hash =
      VrpoOptimizerGroupSpecSha256(CanonicalVrpoPhase4OptimizerGroups());
  if (actor_values_hash != expected_binding.actor_subset_sha256 ||
      q_values_hash != expected_binding.q_init_sha256 ||
      zero_state_hash != expected_binding.optimizer_zero_state_sha256 ||
      actual_optimizer_groups_hash !=
          expected_binding.optimizer_groups_sha256) {
    return fail("loaded identities differ from external registered binding");
  }
  if (!ValidateVrpoExpandedLiveLayout(
          *actor_model, *q_model, expected_layout, expected_binding,
          error)) {
    return false;
  }
  if (!ValidateVrpoPhase4ManifestStrict(
          contract->second.GetObject(), expected_arm, expected_binding,
          error)) {
    return fail(error != nullptr && !error->empty()
                    ? *error
                    : "expanded phase4 contract is invalid");
  }
  if (!require_string("actor_values_sha256", actor_values_hash) ||
      !require_string("q_values_sha256", q_values_hash) ||
      !require_string("optimizer_zero_state_sha256", zero_state_hash)) {
    return fail("expanded loaded parameter/optimizer identity mismatch");
  }
  const VrpoExpandedFileIdentity actor_file = VrpoFileIdentity(paths.actor_model);
  const VrpoExpandedFileIdentity q_file = VrpoFileIdentity(paths.q_model);
  const VrpoExpandedFileIdentity actor_optimizer_file =
      VrpoFileIdentity(paths.actor_optimizer);
  const VrpoExpandedFileIdentity q_optimizer_file =
      VrpoFileIdentity(paths.q_optimizer);
  const json::Object expected = BuildVrpoExpandedManifest(
      expected_arm, expected_binding, expected_layout,
      uuid_it->second.GetString(), 0,
      static_cast<uint64_t>(next_it->second.GetInt()),
      actor_file, q_file, actor_optimizer_file, q_optimizer_file,
      actor_values_hash, q_values_hash, zero_state_hash);
  if (!VrpoExpandedManifestStrictEqual(manifest, expected, error)) {
    return false;
  }
  *loaded_manifest = std::move(manifest);
  if (archive_identity != nullptr) {
    *archive_identity = std::move(actual_archive_identity);
  }
  return true;
}

}  // namespace open_spiel

#endif  // OPEN_SPIEL_EXAMPLES_DUNE_VRPO_CHECKPOINT_H_
