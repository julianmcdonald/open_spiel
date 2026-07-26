#include "dune_ppo_training_utils.h"
#include "dune_sha256.h"
#include "open_spiel/abseil-cpp/absl/strings/str_format.h"
#include "open_spiel/utils/json.h"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>

#ifdef OPEN_SPIEL_BUILD_WITH_LIBTORCH
#include "open_spiel/abseil-cpp/absl/flags/declare.h"
#include "open_spiel/abseil-cpp/absl/flags/flag.h"
#include "dune_seed_utils.h"
#include <unordered_set>
#include <cmath>


ABSL_DECLARE_FLAG(int, ppo_minibatch_size);
ABSL_DECLARE_FLAG(int, ppo_update_epochs);
ABSL_FLAG(double, policy_kl_anchor_coeff, 0.0, "Coefficient for policy KL anchor penalty to prevent policy drift when unfreezing trunk in value-only training");
ABSL_DECLARE_FLAG(double, ppo_clip_epsilon);
ABSL_DECLARE_FLAG(bool, normalize_advantages);
ABSL_DECLARE_FLAG(bool, ppo_clip_value_loss);
ABSL_DECLARE_FLAG(double, entropy_coef);
ABSL_DECLARE_FLAG(double, value_coef);
ABSL_DECLARE_FLAG(double, logit_cap);
ABSL_DECLARE_FLAG(double, target_kl);
ABSL_DECLARE_FLAG(bool, train_amp);
ABSL_DECLARE_FLAG(double, grad_clip_norm);
ABSL_DECLARE_FLAG(bool, diagnostics_only);
ABSL_FLAG(bool, train_value_only, false, "Train only the value head parameters.");
#endif

namespace open_spiel {

std::string ComputeRolloutHash(const std::vector<PpoTransition>& batch) {
  std::string data;
  for (const auto& t : batch) {
    data.append(reinterpret_cast<const char*>(&t.episode_id), sizeof(t.episode_id));
    data.append(reinterpret_cast<const char*>(&t.player_id), sizeof(t.player_id));
    data.append(reinterpret_cast<const char*>(&t.action), sizeof(t.action));
    data.append(reinterpret_cast<const char*>(&t.old_log_prob), sizeof(t.old_log_prob));
    data.append(reinterpret_cast<const char*>(&t.advantage), sizeof(t.advantage));
    data.append(reinterpret_cast<const char*>(&t.return_value), sizeof(t.return_value));
    data.append(reinterpret_cast<const char*>(&t.value), sizeof(t.value));
    data.append(reinterpret_cast<const char*>(t.state.data()), t.state.size() * sizeof(float));
    for (Action a : t.legal_actions) {
      data.append(reinterpret_cast<const char*>(&a), sizeof(a));
    }
  }
  return open_spiel::ComputeStringSHA256(data);
}

bool ParseAndValidateManifest(const std::string& manifest_path,
                              const std::string& model_path,
                              const std::string& optim_path,
                              uint64_t current_base_seed,
                              int current_target_end_update,
                              int current_seed_scheme_version,
                              const std::string& current_config_fingerprint,
                              const std::string& current_search_label_fingerprint,
                              int current_hidden_dim,
                              int current_num_blocks,
                              CheckpointManifest& out_manifest,
                              std::string& error_msg,
                              const std::string& current_legacy_config_fingerprint) {
  if (!std::filesystem::exists(manifest_path)) {
    error_msg = "manifest file not found at: " + manifest_path;
    return false;
  }

  std::ifstream ifs(manifest_path);
  if (!ifs) {
    error_msg = "Could not open manifest file at: " + manifest_path;
    return false;
  }
  std::string content((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
  auto val_opt = open_spiel::json::FromString(content);
  if (!val_opt.has_value() || !val_opt->IsObject()) {
    error_msg = "Manifest file is malformed JSON at: " + manifest_path;
    return false;
  }
  const auto& manifest_obj = val_opt->GetObject();

  auto get_string_field = [&](const std::string& key, std::string& val) -> bool {
    auto it = manifest_obj.find(key);
    if (it == manifest_obj.end() || !it->second.IsString()) {
      error_msg = "Manifest missing required string field: " + key;
      return false;
    }
    val = it->second.GetString();
    return true;
  };

  auto get_int_field = [&](const std::string& key, int64_t& val) -> bool {
    auto it = manifest_obj.find(key);
    if (it == manifest_obj.end() || !it->second.IsInt()) {
      error_msg = "Manifest missing required int field: " + key;
      return false;
    }
    val = it->second.GetInt();
    return true;
  };

  int64_t schema_version = 0;
  int64_t global_update = 0;
  int64_t target_end_update = 0;
  int64_t total_env_steps = 0;
  int64_t next_episode_id = 0;
  int64_t base_seed = 0;
  int64_t seed_scheme_version = 0;
  int64_t model_file_size = 0;
  int64_t optimizer_file_size = 0;
  int64_t hidden_dim = 0;
  int64_t num_blocks = 0;

  if (!get_int_field("schema_version", schema_version) ||
      !get_string_field("checkpoint_uuid", out_manifest.checkpoint_uuid) ||
      !get_int_field("global_update", global_update) ||
      !get_int_field("target_end_update", target_end_update) ||
      !get_int_field("total_env_steps", total_env_steps) ||
      !get_int_field("next_episode_id", next_episode_id) ||
      !get_int_field("base_seed", base_seed) ||
      !get_int_field("seed_scheme_version", seed_scheme_version) ||
      !get_string_field("config_fingerprint", out_manifest.config_fingerprint) ||
      !get_string_field("search_label_fingerprint", out_manifest.search_label_fingerprint) ||
      !get_string_field("run_uuid", out_manifest.run_uuid) ||
      !get_string_field("model_filename", out_manifest.model_filename) ||
      !get_int_field("model_file_size", model_file_size) ||
      !get_string_field("model_sha256", out_manifest.model_sha256) ||
      !get_string_field("optimizer_filename", out_manifest.optimizer_filename) ||
      !get_int_field("optimizer_file_size", optimizer_file_size) ||
      !get_string_field("optimizer_sha256", out_manifest.optimizer_sha256) ||
      !get_int_field("hidden_dim", hidden_dim) ||
      !get_int_field("num_blocks", num_blocks)) {
    return false;
  }

  if (schema_version != 2) {
    error_msg = absl::StrFormat("schema_version mismatch. Expected: 2, Got: %d", schema_version);
    return false;
  }

  auto is_valid_uuid = [](const std::string& s) -> bool {
    if (s.length() != 36) return false;
    for (int i = 0; i < 36; ++i) {
      if (i == 8 || i == 13 || i == 18 || i == 23) {
        if (s[i] != '-') return false;
      } else {
        char c = s[i];
        if (!((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F'))) {
          return false;
        }
      }
    }
    return true;
  };
  if (!is_valid_uuid(out_manifest.checkpoint_uuid)) {
    error_msg = "Invalid checkpoint_uuid format: " + out_manifest.checkpoint_uuid;
    return false;
  }

  out_manifest.schema_version = schema_version;
  out_manifest.global_update = global_update;
  out_manifest.target_end_update = target_end_update;
  out_manifest.total_env_steps = total_env_steps;
  out_manifest.next_episode_id = next_episode_id;
  out_manifest.base_seed = base_seed;
  out_manifest.seed_scheme_version = seed_scheme_version;
  out_manifest.model_file_size = model_file_size;
  out_manifest.optimizer_file_size = optimizer_file_size;
  out_manifest.hidden_dim = hidden_dim;
  out_manifest.num_blocks = num_blocks;

  if (!std::filesystem::exists(model_path)) {
    error_msg = "Model file not found: " + model_path;
    return false;
  }
  if (!std::filesystem::exists(optim_path)) {
    error_msg = "Optimizer file not found: " + optim_path;
    return false;
  }

  if (std::filesystem::path(model_path).filename().string() != out_manifest.model_filename) {
    error_msg = "Model filename mismatch. Manifest: " + out_manifest.model_filename + ", Current: " + std::filesystem::path(model_path).filename().string();
    return false;
  }
  if (std::filesystem::path(optim_path).filename().string() != out_manifest.optimizer_filename) {
    error_msg = "Optimizer filename mismatch. Manifest: " + out_manifest.optimizer_filename + ", Current: " + std::filesystem::path(optim_path).filename().string();
    return false;
  }

  size_t actual_model_size = 0;
  std::string actual_model_hash = open_spiel::ComputeFileSHA256(model_path, &actual_model_size);
  if (actual_model_size != out_manifest.model_file_size) {
    error_msg = absl::StrFormat("Model file size mismatch. Manifest: %d, Actual: %d", out_manifest.model_file_size, actual_model_size);
    return false;
  }
  if (actual_model_hash != out_manifest.model_sha256) {
    error_msg = "Model SHA-256 hash mismatch. Manifest: " + out_manifest.model_sha256 + ", Actual: " + actual_model_hash;
    return false;
  }

  size_t actual_optim_size = 0;
  std::string actual_optim_hash = open_spiel::ComputeFileSHA256(optim_path, &actual_optim_size);
  if (actual_optim_size != out_manifest.optimizer_file_size) {
    error_msg = absl::StrFormat("Optimizer file size mismatch. Manifest: %d, Actual: %d", out_manifest.optimizer_file_size, actual_optim_size);
    return false;
  }
  if (actual_optim_hash != out_manifest.optimizer_sha256) {
    error_msg = "Optimizer SHA-256 hash mismatch. Manifest: " + out_manifest.optimizer_sha256 + ", Actual: " + actual_optim_hash;
    return false;
  }

  if (current_base_seed != out_manifest.base_seed) {
    error_msg = absl::StrFormat("Base seed mismatch. Current: %d, Manifest: %d", current_base_seed, out_manifest.base_seed);
    return false;
  }
  if (current_target_end_update != out_manifest.target_end_update) {
    error_msg = absl::StrFormat("Target end update mismatch. Current: %d, Manifest: %d", current_target_end_update, out_manifest.target_end_update);
    return false;
  }
  if (current_seed_scheme_version != out_manifest.seed_scheme_version) {
    error_msg = absl::StrFormat("Seed scheme version mismatch. Current: %d, Manifest: %d", current_seed_scheme_version, out_manifest.seed_scheme_version);
    return false;
  }

  if (current_config_fingerprint != out_manifest.config_fingerprint) {
    if (current_legacy_config_fingerprint.empty() || current_legacy_config_fingerprint != out_manifest.config_fingerprint) {
      error_msg = "Configuration fingerprint mismatch. Effective hyperparameters changed.\n  Expected: " + out_manifest.config_fingerprint + "\n  Got:      " + current_config_fingerprint;
      return false;
    }
    std::cout << "Resuming checkpoint with legacy config fingerprint format.\n";
  }
  if (current_search_label_fingerprint != out_manifest.search_label_fingerprint) {
    error_msg = "Search label fingerprint mismatch.\n  Expected: " + out_manifest.search_label_fingerprint + "\n  Got:      " + current_search_label_fingerprint;
    return false;
  }

  if (current_hidden_dim != out_manifest.hidden_dim) {
    error_msg = absl::StrFormat("hidden_dim mismatch. Flags: %d, Manifest: %d", current_hidden_dim, out_manifest.hidden_dim);
    return false;
  }
  if (current_num_blocks != out_manifest.num_blocks) {
    error_msg = absl::StrFormat("num_blocks mismatch. Flags: %d, Manifest: %d", current_num_blocks, out_manifest.num_blocks);
    return false;
  }

  return true;
}

std::vector<std::pair<int64_t, int64_t>> ComputeAuxSlices(
    int64_t num_examples, int64_t num_minibatches) {
  std::vector<std::pair<int64_t, int64_t>> out;
  if (num_minibatches <= 0) return out;
  int64_t base = num_examples / num_minibatches;
  int64_t rem = num_examples % num_minibatches;
  int64_t cursor = 0;
  for (int64_t k = 0; k < num_minibatches; ++k) {
    int64_t len = base + (k < rem ? 1 : 0);
    out.push_back({cursor, len});
    cursor += len;
  }
  return out;
}

void WriteOnlineCollectionState(json::Object& manifest_obj,
                                const OnlineCollectionState& st) {
  json::Object o;
  o["auxiliary_games"] = static_cast<int64_t>(st.auxiliary_games);
  o["auxiliary_search_seed_domain"] =
      static_cast<int64_t>(st.auxiliary_search_seed_domain);
  o["collector_dirichlet_epsilon"] = st.collector_dirichlet_epsilon;
  o["swordmaster_grant_fraction"] = st.swordmaster_grant_fraction;
  o["swordmaster_grant_round"] = static_cast<int64_t>(st.swordmaster_grant_round);
  o["search_loss_coef_target"] = st.search_loss_coef_target;
  o["search_loss_warmup_update"] =
      static_cast<int64_t>(st.search_loss_warmup_update);
  o["abort_grad_norm_ratio"] = st.abort_grad_norm_ratio;
  o["target_sharpen_exponent"] = st.target_sharpen_exponent;
  o["acceptance_prior_source"] = st.acceptance_prior_source;
  o["next_auxiliary_episode_id"] =
      static_cast<int64_t>(st.next_auxiliary_episode_id);
  o["cum_accepted"] = static_cast<int64_t>(st.cum_accepted);
  o["cum_rejected"] = static_cast<int64_t>(st.cum_rejected);
  json::Array rs, ra;
  for (int i = 0; i < 3; ++i) {
    rs.push_back(static_cast<int64_t>(st.cum_role_searches[i]));
    ra.push_back(static_cast<int64_t>(st.cum_role_accepted[i]));
  }
  o["cum_role_searches"] = rs;
  o["cum_role_accepted"] = ra;
  o["cum_granted"] = static_cast<int64_t>(st.cum_granted);
  o["cum_organic"] = static_cast<int64_t>(st.cum_organic);
  o["accepted_hash_chain"] = st.accepted_hash_chain;
  manifest_obj["online_collection"] = o;
}

bool ReadOnlineCollectionState(const std::string& manifest_path,
                               OnlineCollectionState& out,
                               std::string& error_msg) {
  if (!std::filesystem::exists(manifest_path)) {
    error_msg = "manifest file not found: " + manifest_path;
    return false;
  }
  std::ifstream ifs(manifest_path);
  if (!ifs) {
    error_msg = "could not open manifest: " + manifest_path;
    return false;
  }
  std::string content((std::istreambuf_iterator<char>(ifs)),
                      std::istreambuf_iterator<char>());
  auto val_opt = json::FromString(content);
  if (!val_opt.has_value() || !val_opt->IsObject()) {
    error_msg = "malformed manifest JSON: " + manifest_path;
    return false;
  }
  const auto& mo = val_opt->GetObject();
  auto it = mo.find("online_collection");
  if (it == mo.end()) {  // absent is fine (collection was off)
    out.present = false;
    return true;
  }
  if (!it->second.IsObject()) {
    error_msg = "manifest online_collection is not an object";
    return false;
  }
  const auto& o = it->second.GetObject();
  auto gi = [&](const char* k, int64_t def) -> int64_t {
    auto f = o.find(k);
    return (f != o.end() && f->second.IsInt()) ? f->second.GetInt() : def;
  };
  auto gd = [&](const char* k, double def) -> double {
    auto f = o.find(k);
    if (f == o.end()) return def;
    if (f->second.IsDouble()) return f->second.GetDouble();
    if (f->second.IsInt()) return static_cast<double>(f->second.GetInt());
    return def;
  };
  auto ga = [&](const char* k, int64_t* arr) {
    auto f = o.find(k);
    if (f != o.end() && f->second.IsArray()) {
      const auto& a = f->second.GetArray();
      for (int i = 0; i < 3 && i < static_cast<int>(a.size()); ++i) {
        if (a[i].IsInt()) arr[i] = a[i].GetInt();
      }
    }
  };
  out.auxiliary_games = static_cast<int>(gi("auxiliary_games", 0));
  out.auxiliary_search_seed_domain =
      static_cast<uint64_t>(gi("auxiliary_search_seed_domain", 0));
  out.collector_dirichlet_epsilon = gd("collector_dirichlet_epsilon", 0.0);
  out.swordmaster_grant_fraction = gd("swordmaster_grant_fraction", 0.0);
  out.swordmaster_grant_round = static_cast<int>(gi("swordmaster_grant_round", 0));
  out.search_loss_coef_target = gd("search_loss_coef_target", 0.0);
  out.search_loss_warmup_update =
      static_cast<int>(gi("search_loss_warmup_update", 0));
  out.abort_grad_norm_ratio = gd("abort_grad_norm_ratio", 0.0);
  out.target_sharpen_exponent = gd("target_sharpen_exponent", 1.0);
  // Absent (pre-WO-20 manifest) stays EMPTY on purpose: "unknown", not a
  // default. The resume guard distinguishes the two.
  auto aps = o.find("acceptance_prior_source");
  if (aps != o.end() && aps->second.IsString()) {
    out.acceptance_prior_source = aps->second.GetString();
  }
  out.next_auxiliary_episode_id =
      static_cast<uint64_t>(gi("next_auxiliary_episode_id", 0));
  out.cum_accepted = gi("cum_accepted", 0);
  out.cum_rejected = gi("cum_rejected", 0);
  ga("cum_role_searches", out.cum_role_searches);
  ga("cum_role_accepted", out.cum_role_accepted);
  out.cum_granted = gi("cum_granted", 0);
  out.cum_organic = gi("cum_organic", 0);
  auto fs = o.find("accepted_hash_chain");
  if (fs != o.end() && fs->second.IsString()) {
    out.accepted_hash_chain = fs->second.GetString();
  }
  out.present = true;
  return true;
}

#ifdef OPEN_SPIEL_BUILD_WITH_LIBTORCH
float ComputeRewardLambda(uint64_t env_steps, uint64_t start_steps, uint64_t decay_steps) {
  if (decay_steps == 0) {
    return 1.0f;
  }
  if (env_steps < start_steps) {
    return 1.0f;
  }
  double elapsed = static_cast<double>(env_steps - start_steps);
  double lambda = std::clamp(1.0 - elapsed / static_cast<double>(decay_steps), 0.0, 1.0);
  return static_cast<float>(lambda);
}

torch::Tensor LegalLogitMean(const torch::Tensor& logits,
                             const torch::Tensor& legal_mask) {
  torch::Tensor mask_f = legal_mask.to(logits.dtype());
  torch::Tensor legal_counts = mask_f.sum(1, true).clamp_min(1.0);
  return (logits * mask_f).sum(1, true) / legal_counts;
}

torch::Tensor ApplyLogitCapTensor(const torch::Tensor& logits,
                                  float logit_cap) {
  if (logit_cap <= 0.0f) return logits;
  return logit_cap * torch::tanh(logits / logit_cap);
}

torch::Tensor CenterAndCapLogitsTensor(const torch::Tensor& logits,
                                       const torch::Tensor& legal_mask,
                                       float logit_cap) {
  return ApplyLogitCapTensor(logits - LegalLogitMean(logits, legal_mask),
                             logit_cap);
}

PpoUpdateStats TrainPpoUpdate(
    std::shared_ptr<SharedDunePolicyValueNetImpl> model,
    torch::optim::AdamW& optimizer, std::vector<PpoTransition>& batch,
    int64_t obs_size, int64_t action_dim, torch::Device device,
    uint64_t master, int global_update,
    std::shared_ptr<SharedDunePolicyValueNetImpl> anchor_model,
    const std::vector<SearchTrainingExample>& search_examples,
    double search_loss_coef, double abort_grad_norm_ratio) {
  PpoUpdateStats stats;
  if (batch.empty()) return stats;
  stats.rollout_hash = ComputeRolloutHash(batch);

  int64_t n = static_cast<int64_t>(batch.size());
  auto cpu_float = torch::TensorOptions().dtype(torch::kFloat32);
  auto cpu_bool = torch::TensorOptions().dtype(torch::kBool);
  auto cpu_long = torch::TensorOptions().dtype(torch::kInt64);

  torch::Tensor states_cpu = torch::empty({n, obs_size}, cpu_float);
  torch::Tensor masks_cpu = torch::zeros({n, action_dim}, cpu_bool);
  torch::Tensor actions_cpu = torch::empty({n}, cpu_long);
  torch::Tensor old_log_probs_cpu = torch::empty({n}, cpu_float);
  torch::Tensor advantages_cpu = torch::empty({n}, cpu_float);
  torch::Tensor returns_cpu = torch::empty({n}, cpu_float);
  torch::Tensor old_values_cpu = torch::empty({n}, cpu_float);

  float* states_ptr = states_cpu.data_ptr<float>();
  bool* masks_ptr = masks_cpu.data_ptr<bool>();
  int64_t* actions_ptr = actions_cpu.data_ptr<int64_t>();
  float* old_log_probs_ptr = old_log_probs_cpu.data_ptr<float>();
  float* advantages_ptr = advantages_cpu.data_ptr<float>();
  float* returns_ptr = returns_cpu.data_ptr<float>();
  float* old_values_ptr = old_values_cpu.data_ptr<float>();

  for (int64_t i = 0; i < n; ++i) {
    const PpoTransition& transition = batch[i];
    std::memcpy(states_ptr + i * obs_size, transition.state.data(),
                obs_size * sizeof(float));
    for (Action action : transition.legal_actions) {
      if (action >= 0 && action < action_dim) {
        masks_ptr[i * action_dim + action] = true;
      }
    }
    actions_ptr[i] = transition.action;
    old_log_probs_ptr[i] = transition.old_log_prob;
    advantages_ptr[i] = transition.advantage;
    returns_ptr[i] = transition.return_value;
    old_values_ptr[i] = transition.value;
  }

  torch::Tensor states = states_cpu.to(device);
  torch::Tensor masks = masks_cpu.to(device);
  torch::Tensor actions = actions_cpu.to(device);
  torch::Tensor old_log_probs = old_log_probs_cpu.to(device);
  torch::Tensor advantages = advantages_cpu.to(device);
  torch::Tensor returns = returns_cpu.to(device);
  torch::Tensor old_values = old_values_cpu.to(device);

  int64_t minibatch_size =
      std::min<int64_t>(::absl::GetFlag(::FLAGS_ppo_minibatch_size), n);
  int update_epochs = std::max(1, ::absl::GetFlag(::FLAGS_ppo_update_epochs));
  float clip_epsilon = static_cast<float>(::absl::GetFlag(::FLAGS_ppo_clip_epsilon));
  bool normalize_advantages = ::absl::GetFlag(::FLAGS_normalize_advantages);
  bool clip_value_loss = ::absl::GetFlag(::FLAGS_ppo_clip_value_loss);
  float entropy_coef = static_cast<float>(::absl::GetFlag(::FLAGS_entropy_coef));
  float value_coef = static_cast<float>(::absl::GetFlag(::FLAGS_value_coef));
  float logit_cap = static_cast<float>(::absl::GetFlag(::FLAGS_logit_cap));
  double target_kl = ::absl::GetFlag(::FLAGS_target_kl);

  // --- Phase 5 Diagnostics Pre-pass ---

  // 1. Episode-ID Uniqueness
  std::unordered_set<uint64_t> seen_episodes;
  bool episode_ids_unique = true;
  if (!batch.empty()) {
    uint64_t last_ep_id = batch[0].episode_id;
    seen_episodes.insert(last_ep_id);
    for (const auto& trans : batch) {
      if (trans.episode_id != last_ep_id) {
        last_ep_id = trans.episode_id;
        if (seen_episodes.find(last_ep_id) != seen_episodes.end()) {
          episode_ids_unique = false;
        }
        seen_episodes.insert(last_ep_id);
      }
    }
  }
  stats.episode_ids_unique = episode_ids_unique;

  // 2. Nontrivial Mask
  torch::Tensor nontrivial_mask_cpu = torch::zeros({n}, cpu_bool);
  bool* nontrivial_mask_ptr = nontrivial_mask_cpu.data_ptr<bool>();
  int64_t nontrivial_count = 0;
  for (int64_t i = 0; i < n; ++i) {
    bool is_nontrivial = (batch[i].legal_actions.size() > 1);
    nontrivial_mask_ptr[i] = is_nontrivial;
    if (is_nontrivial) {
      nontrivial_count++;
    }
  }
  stats.total_transitions = n;
  stats.nontrivial_transitions = nontrivial_count;
  stats.forced_transitions = n - nontrivial_count;

  torch::Tensor nontrivial_mask = nontrivial_mask_cpu.to(device);

  // 3. Policy KL Before & Saturation Checks (merged forward pre-pass)
  double kl_before_sum = 0.0;
  int64_t kl_before_nontrivial_count = 0;
  double value_near_one_count = 0;

  bool was_training = model->is_training();
  model->eval();
  {
    torch::NoGradGuard no_grad;
    for (int64_t start = 0; start < n; start += minibatch_size) {
      int64_t end = std::min(start + minibatch_size, n);
      torch::Tensor mb_states = states.narrow(0, start, end - start);
      torch::Tensor mb_masks = masks.narrow(0, start, end - start);
      torch::Tensor mb_actions = actions.narrow(0, start, end - start);
      torch::Tensor mb_old_log_probs = old_log_probs.narrow(0, start, end - start);
      torch::Tensor mb_nontrivial = nontrivial_mask.narrow(0, start, end - start);

      auto outputs = model->forward(mb_states);
      if (!::absl::GetFlag(::FLAGS_train_value_only)) {
        torch::Tensor logits = CenterAndCapLogitsTensor(outputs.logits, mb_masks, logit_cap);
        torch::Tensor masked_logits = logits.masked_fill(mb_masks.logical_not(), -1e9f);
        torch::Tensor log_probs = torch::log_softmax(masked_logits, -1);
        torch::Tensor selected_log_probs = log_probs.gather(1, mb_actions.unsqueeze(1)).squeeze(1);

        torch::Tensor log_ratio = selected_log_probs - mb_old_log_probs;
        torch::Tensor ratio = torch::exp(log_ratio);
        torch::Tensor approx_kl = (ratio - 1.0f) - log_ratio;

        torch::Tensor masked_kl = approx_kl.masked_select(mb_nontrivial);
        if (masked_kl.numel() > 0) {
          kl_before_sum += masked_kl.sum().item<double>();
          kl_before_nontrivial_count += masked_kl.numel();
        }
      }

      torch::Tensor mb_values = outputs.values.squeeze(1);
      torch::Tensor mb_near_one = mb_values.abs() >= 0.99f;
      value_near_one_count += mb_near_one.sum().item<double>();
    }
  }
  if (was_training) {
    model->train();
  }
  stats.policy_kl_before = (kl_before_nontrivial_count > 0) ? (kl_before_sum / kl_before_nontrivial_count) : 0.0;
  stats.fraction_critic_near_1 = value_near_one_count / n;

  // 4. Return Calibration Diagnostics
  std::vector<float> returns_vec(returns_ptr, returns_ptr + n);
  std::sort(returns_vec.begin(), returns_vec.end());
  stats.return_min = returns_vec.front();
  stats.return_max = returns_vec.back();
  stats.return_p50 = returns_vec[n * 50 / 100];
  stats.return_p95 = returns_vec[n * 95 / 100];
  stats.return_p99 = returns_vec[n * 99 / 100];

  std::vector<float> abs_returns_vec(n);
  int64_t count_outside = 0;
  for (int64_t i = 0; i < n; ++i) {
    abs_returns_vec[i] = std::abs(returns_ptr[i]);
    if (returns_ptr[i] < -1.0f || returns_ptr[i] > 1.0f) {
      count_outside++;
    }
  }
  std::sort(abs_returns_vec.begin(), abs_returns_vec.end());
  stats.abs_return_p99 = abs_returns_vec[n * 99 / 100];
  stats.fraction_targets_outside_1 = static_cast<double>(count_outside) / n;

  if (absl::GetFlag(FLAGS_diagnostics_only)) {
    return stats;
  }

  // --- Phase 18B online-search auxiliary examples (combined optimization) ---
  // Active only with non-empty examples AND a positive coefficient (and not in
  // value-only mode, whose whole point is a frozen policy). When INACTIVE every
  // line below is skipped and the update is numerically identical to today.
  const bool aux_active = !search_examples.empty() && search_loss_coef > 0.0 &&
                          !::absl::GetFlag(::FLAGS_train_value_only);
  stats.aux_search_loss_coef = search_loss_coef;
  const int64_t aux_n =
      aux_active ? static_cast<int64_t>(search_examples.size()) : 0;
  torch::Tensor aux_states, aux_masks, aux_targets, aux_values;
  std::vector<int64_t> aux_slice_start, aux_slice_len;  // by minibatch index/epoch
  double aux_ce_sum = 0.0, aux_vmse_sum = 0.0;
  int64_t aux_slice_uses = 0;                 // (epoch,minibatch) slices actually run
  double aux_norm_sum = 0.0, ppo_only_norm_sum = 0.0;
  int64_t aux_norm_count = 0;
  if (aux_active) {
    stats.aux_examples_used = static_cast<int>(aux_n);
    auto af = torch::TensorOptions().dtype(torch::kFloat32);
    auto ab = torch::TensorOptions().dtype(torch::kBool);
    torch::Tensor s = torch::zeros({aux_n, obs_size}, af);
    torch::Tensor m = torch::zeros({aux_n, action_dim}, ab);
    torch::Tensor t = torch::zeros({aux_n, action_dim}, af);
    torch::Tensor v = torch::zeros({aux_n}, af);
    float* sp = s.data_ptr<float>();
    bool* mp = m.data_ptr<bool>();
    float* tp = t.data_ptr<float>();
    float* vp = v.data_ptr<float>();
    for (int64_t i = 0; i < aux_n; ++i) {
      const SearchTrainingExample& ex = search_examples[i];
      int64_t copy = std::min<int64_t>(
          obs_size, static_cast<int64_t>(ex.observation.size()));
      if (copy > 0) {
        std::memcpy(sp + i * obs_size, ex.observation.data(),
                    copy * sizeof(float));
      }
      // Legal mask + normalized-visit CE targets, aligned to legal_actions.
      for (size_t j = 0; j < ex.legal_actions.size(); ++j) {
        Action a = ex.legal_actions[j];
        if (a >= 0 && a < action_dim) {
          mp[i * action_dim + a] = true;
          tp[i * action_dim + a] = static_cast<float>(
              (j < ex.normalized_visits.size()) ? ex.normalized_visits[j] : 0.0);
        }
      }
      vp[i] = static_cast<float>(ex.value_target);
    }
    aux_states = s.to(device);
    aux_masks = m.to(device);
    aux_targets = t.to(device);
    aux_values = v.to(device);
    // Deterministic contiguous partition of the E examples across the
    // num_minibatches PPO steps of one epoch (base each; remainder spread to the
    // first `rem`). Identical every epoch => each example is used exactly once
    // per PPO epoch that fully runs (KL early-stop may cut the tail — expected).
    int64_t num_mb = (n + minibatch_size - 1) / minibatch_size;
    auto slices = ComputeAuxSlices(aux_n, num_mb);
    aux_slice_start.assign(num_mb, 0);
    aux_slice_len.assign(num_mb, 0);
    for (int64_t k = 0; k < num_mb && k < static_cast<int64_t>(slices.size()); ++k) {
      aux_slice_start[k] = slices[k].first;
      aux_slice_len[k] = slices[k].second;
    }
  }

  // --- PPO Loss Loop ---
  double weighted_policy_loss_sum = 0.0;
  double weighted_entropy_sum = 0.0;
  double weighted_kl_sum = 0.0;
  double weighted_clip_fraction_sum = 0.0;
  int64_t total_nontrivial_count = 0;
  double value_loss_sum = 0.0;
  // WO-1 Phase 5: accumulated over executed minibatches, same denominator as
  // value_loss_sum.
  double value_clip_frac_sum = 0.0;

  for (int epoch = 0; epoch < update_epochs; ++epoch) {
    uint64_t perm_seed = dune_seed::DeriveSeed(master, dune_seed::kDomainTrain, global_update, epoch, dune_seed::kStreamPPOPermutation);
    at::Generator gen = dune_seed::MakeTorchCPUGenerator(perm_seed);
    torch::Tensor permutation =
        torch::randperm(n, gen, torch::TensorOptions().device(torch::kCPU).dtype(torch::kInt64)).to(device);

    double epoch_kl_sum = 0.0;
    int64_t epoch_kl_nontrivial_count = 0;

    for (int64_t start = 0; start < n; start += minibatch_size) {
      int64_t end = std::min(start + minibatch_size, n);
      const int64_t mb_index = start / minibatch_size;  // minibatch # within epoch
      torch::Tensor mb_idx = permutation.narrow(0, start, end - start);

      torch::Tensor mb_states = states.index_select(0, mb_idx);
      torch::Tensor mb_masks = masks.index_select(0, mb_idx);
      torch::Tensor mb_actions = actions.index_select(0, mb_idx);
      torch::Tensor mb_old_log_probs = old_log_probs.index_select(0, mb_idx);
      torch::Tensor mb_advantages = advantages.index_select(0, mb_idx);
      torch::Tensor mb_returns = returns.index_select(0, mb_idx);
      torch::Tensor mb_old_values = old_values.index_select(0, mb_idx);
      torch::Tensor mb_nontrivial = nontrivial_mask.index_select(0, mb_idx);

      int64_t mb_num_nontrivial = mb_nontrivial.sum().item<int64_t>();

      optimizer.zero_grad();

      // --- Phase 18B auxiliary search loss (disjoint graph via a separate
      // forward). Backward the SCALED aux loss FIRST so the aux-only grads can be
      // snapshotted; the PPO backward below then accumulates on top, and
      // ppo_grad = total_grad - aux_grad exactly (grads add). Entirely skipped
      // when aux is inactive or this minibatch's slice is empty -> identical to
      // today. Aux terms never touch PPO ratios/clip/entropy/adv/KL. ---
      std::vector<torch::Tensor> aux_grad_snapshot;
      bool this_mb_has_aux = false;
      double this_aux_norm = 0.0;
      if (aux_active) {
        const int64_t as = aux_slice_start[mb_index];
        const int64_t al = aux_slice_len[mb_index];
        if (al > 0) {
          this_mb_has_aux = true;
          torch::Tensor a_states = aux_states.narrow(0, as, al);
          torch::Tensor a_masks = aux_masks.narrow(0, as, al);
          torch::Tensor a_targets = aux_targets.narrow(0, as, al);
          torch::Tensor a_values = aux_values.narrow(0, as, al);
          torch::Tensor scaled_aux;
          double ce_val = 0.0, vmse_val = 0.0;
          auto compute_aux = [&]() {
            auto ao = model->forward(a_states);
            torch::Tensor alogits =
                CenterAndCapLogitsTensor(ao.logits, a_masks, logit_cap);
            torch::Tensor amasked =
                alogits.masked_fill(a_masks.logical_not(), -1e9f);
            torch::Tensor alogp = torch::log_softmax(amasked, -1);
            // Legal-action cross-entropy: targets are 0 at illegal actions, so
            // 0 * (finite masked logprob) = 0 there (no NaN).
            torch::Tensor ce = -(a_targets * alogp).sum(-1).mean();
            torch::Tensor av = ao.values.squeeze(1);
            torch::Tensor vmse = (av - a_values).pow(2).mean();
            ce_val = ce.item<double>();
            vmse_val = vmse.item<double>();
            scaled_aux =
                static_cast<float>(search_loss_coef) * (ce + value_coef * vmse);
          };
          if (device.is_cuda() && ::absl::GetFlag(::FLAGS_train_amp)) {
            AutocastGuard autocast_guard(c10::DeviceType::CUDA, true);
            compute_aux();
          } else {
            compute_aux();
          }
          scaled_aux.backward();
          // Snapshot aux-only grads (clone) and their flattened 2-norm.
          auto params = model->parameters();
          aux_grad_snapshot.resize(params.size());
          double aux_sq = 0.0;
          for (size_t pi = 0; pi < params.size(); ++pi) {
            auto g = params[pi].grad();
            if (g.defined()) {
              aux_grad_snapshot[pi] = g.detach().clone();
              aux_sq += aux_grad_snapshot[pi].pow(2).sum().item<double>();
            }
          }
          this_aux_norm = std::sqrt(aux_sq);
          aux_ce_sum += ce_val;
          aux_vmse_sum += vmse_val;
          ++aux_slice_uses;
        }
      }

      torch::Tensor policy_loss, value_loss, entropy, approx_kl, clip_fraction;
      torch::Tensor total_loss;
      // Left undefined when --ppo_clip_value_loss=false: nothing is clipped, so
      // the fraction contributes 0 rather than a fabricated value.
      torch::Tensor value_clip_frac_t;

      auto compute_loss = [&]() {
        auto outputs = model->forward(mb_states);

        // 3. Critic loss (Retain all samples)
        torch::Tensor new_values = outputs.values.squeeze(1);
        if (clip_value_loss) {
          // WO-1 Phase 5: how often the value clip actually bound. Computed
          // inside the autocast region but only read out where the other
          // metrics are, so no extra device sync is added to the hot path.
          value_clip_frac_t =
              ((new_values - mb_old_values).abs() > clip_epsilon)
                  .to(torch::kFloat32)
                  .mean();
          torch::Tensor value_loss_unclipped =
              (new_values - mb_returns).pow(2);
          torch::Tensor value_clipped =
              mb_old_values +
              (new_values - mb_old_values)
                  .clamp(-clip_epsilon, clip_epsilon);
          torch::Tensor value_loss_clipped =
              (value_clipped - mb_returns).pow(2);
          value_loss =
              0.5f * torch::max(value_loss_unclipped, value_loss_clipped).mean();
        } else {
          value_loss = 0.5f * (new_values - mb_returns).pow(2).mean();
        }

        if (::absl::GetFlag(::FLAGS_train_value_only)) {
          policy_loss = torch::tensor(0.0f, torch::TensorOptions().device(device).dtype(torch::kFloat32));
          entropy = torch::tensor(0.0f, torch::TensorOptions().device(device).dtype(torch::kFloat32));
          approx_kl = torch::tensor(0.0f, torch::TensorOptions().device(device).dtype(torch::kFloat32));
          clip_fraction = torch::tensor(0.0f, torch::TensorOptions().device(device).dtype(torch::kFloat32));
          total_loss = value_coef * value_loss;
          if (anchor_model) {
            auto anchor_outputs = anchor_model->forward(mb_states);
            torch::Tensor anchor_logits = CenterAndCapLogitsTensor(anchor_outputs.logits, mb_masks, logit_cap);
            torch::Tensor masked_anchor_logits = anchor_logits.masked_fill(mb_masks.logical_not(), -1e9f);
            torch::Tensor anchor_probs = torch::softmax(masked_anchor_logits, -1);
            torch::Tensor log_anchor_probs = torch::log_softmax(masked_anchor_logits, -1);

            torch::Tensor logits = CenterAndCapLogitsTensor(outputs.logits, mb_masks, logit_cap);
            torch::Tensor masked_logits = logits.masked_fill(mb_masks.logical_not(), -1e9f);
            torch::Tensor log_probs = torch::log_softmax(masked_logits, -1);

            torch::Tensor kl_loss = anchor_probs * (log_anchor_probs - log_probs);
            torch::Tensor mean_kl;
            if (mb_num_nontrivial > 0) {
              mean_kl = kl_loss.sum(-1).masked_select(mb_nontrivial).mean();
            } else {
              mean_kl = torch::tensor(0.0f, torch::TensorOptions().device(device).dtype(torch::kFloat32));
            }
            double kl_anchor_coeff = ::absl::GetFlag(::FLAGS_policy_kl_anchor_coeff);
            total_loss += kl_anchor_coeff * mean_kl;
          }
          return;
        }

        torch::Tensor logits =
            CenterAndCapLogitsTensor(outputs.logits, mb_masks, logit_cap);
        torch::Tensor masked_logits =
            logits.masked_fill(mb_masks.logical_not(), -1e9f);
        torch::Tensor log_probs = torch::log_softmax(masked_logits, -1);
        torch::Tensor probs = torch::softmax(masked_logits, -1);
        torch::Tensor selected_log_probs =
            log_probs.gather(1, mb_actions.unsqueeze(1)).squeeze(1);

        torch::Tensor log_ratio = selected_log_probs - mb_old_log_probs;
        torch::Tensor ratio = torch::exp(log_ratio);

        // 1. Advantage Normalization (per-minibatch)
        torch::Tensor mb_adv = mb_advantages;
        if (normalize_advantages) {
          if (mb_num_nontrivial > 1) {
            torch::Tensor nontrivial_adv = mb_advantages.masked_select(mb_nontrivial);
            torch::Tensor mean = nontrivial_adv.mean();
            torch::Tensor std = nontrivial_adv.std(/*unbiased=*/false) + 1e-8f;
            mb_adv = (mb_advantages - mean) / std;
          }
        }
        mb_adv = mb_adv.detach();

        // 2. PPO surrogate loss
        torch::Tensor pg_loss1 = -mb_adv * ratio;
        torch::Tensor pg_loss2 =
            -mb_adv * ratio.clamp(1.0f - clip_epsilon,
                                  1.0f + clip_epsilon);
        torch::Tensor pg_loss = torch::max(pg_loss1, pg_loss2);

        // 4. Actor metrics and entropy
        if (mb_num_nontrivial > 0) {
          policy_loss = pg_loss.masked_select(mb_nontrivial).mean();
          entropy = -(probs * log_probs).sum(-1).masked_select(mb_nontrivial).mean();
          approx_kl = ((ratio - 1.0f) - log_ratio).masked_select(mb_nontrivial).mean();
          clip_fraction =
              ((ratio - 1.0f).abs() > clip_epsilon).to(torch::kFloat32).masked_select(mb_nontrivial).mean();

          total_loss = policy_loss + value_coef * value_loss -
                       entropy_coef * entropy;
        } else {
          // 0 nontrivial transitions: train critic only
          policy_loss = torch::tensor(0.0f, torch::TensorOptions().device(device).dtype(torch::kFloat32));
          entropy = torch::tensor(0.0f, torch::TensorOptions().device(device).dtype(torch::kFloat32));
          approx_kl = torch::tensor(0.0f, torch::TensorOptions().device(device).dtype(torch::kFloat32));
          clip_fraction = torch::tensor(0.0f, torch::TensorOptions().device(device).dtype(torch::kFloat32));

          total_loss = value_coef * value_loss;
        }
      };

      if (device.is_cuda() && ::absl::GetFlag(::FLAGS_train_amp)) {
        AutocastGuard autocast_guard(c10::DeviceType::CUDA, true);
        compute_loss();
      } else {
        compute_loss();
      }

      total_loss.backward();

      // Phase 18B unclipped norm accounting (before the shared clip): the
      // accumulated grad now holds aux+ppo; ppo-only = total - aux (snapshot).
      if (this_mb_has_aux) {
        auto params = model->parameters();
        double ppo_sq = 0.0;
        for (size_t pi = 0; pi < params.size(); ++pi) {
          auto tot = params[pi].grad();
          if (!tot.defined()) continue;
          torch::Tensor ppo_g =
              (pi < aux_grad_snapshot.size() && aux_grad_snapshot[pi].defined())
                  ? (tot - aux_grad_snapshot[pi])
                  : tot;
          ppo_sq += ppo_g.pow(2).sum().item<double>();
        }
        aux_norm_sum += this_aux_norm;
        ppo_only_norm_sum += std::sqrt(ppo_sq);
        ++aux_norm_count;
      }

      // WO-1 Phase 5: per-module pre-clip gradient norms. MUST be measured here,
      // before clip_grad_norm_ rescales the gradients in place. Grouped by
      // parameter-name prefix; the `value_head` test uses find() rather than a
      // prefix match so `value_head2` (the nonlinear head's second layer) lands
      // in the value group instead of the trunk.
      {
        double policy_sq = 0.0, value_sq = 0.0, trunk_sq = 0.0;
        for (const auto& named : model->named_parameters()) {
          const torch::Tensor& g = named.value().grad();
          if (!g.defined()) continue;
          const double sq = g.pow(2).sum().item<double>();
          const std::string& nm = named.key();
          if (nm.rfind("policy_head", 0) == 0) {
            policy_sq += sq;
          } else if (nm.find("value_head") != std::string::npos) {
            value_sq += sq;
          } else {
            trunk_sq += sq;
          }
        }
        stats.policy_head_grad_norm_sum += std::sqrt(policy_sq);
        stats.value_head_grad_norm_sum += std::sqrt(value_sq);
        stats.trunk_grad_norm_sum += std::sqrt(trunk_sq);
        stats.head_grad_norm_count += 1;
      }

      double grad_norm =
          torch::nn::utils::clip_grad_norm_(
              model->parameters(), ::absl::GetFlag(::FLAGS_grad_clip_norm));
      if (std::isnan(grad_norm) || std::isinf(grad_norm)) {
        stats.nonfinite_abort = true;
        std::cerr << "Fatal PPO gradient norm: " << grad_norm << "\n";
        std::exit(EXIT_FAILURE);
      }
      stats.grad_norm_sum += grad_norm;
      stats.grad_norm_count += 1;
      // clip_grad_norm_ returns the norm as measured BEFORE clipping, so this
      // max tracks the same unclipped quantity grad_norm_sum accumulates.
      if (grad_norm > stats.grad_norm_max) stats.grad_norm_max = grad_norm;
      optimizer.step();

      double kl = approx_kl.item<double>();
      value_loss_sum += value_loss.item<double>();
      if (value_clip_frac_t.defined()) {
        value_clip_frac_sum += value_clip_frac_t.item<double>();
      }
      stats.minibatches += 1;

      if (mb_num_nontrivial > 0) {
        weighted_policy_loss_sum += policy_loss.item<double>() * mb_num_nontrivial;
        weighted_entropy_sum += entropy.item<double>() * mb_num_nontrivial;
        weighted_kl_sum += kl * mb_num_nontrivial;
        weighted_clip_fraction_sum += clip_fraction.item<double>() * mb_num_nontrivial;
        total_nontrivial_count += mb_num_nontrivial;

        epoch_kl_sum += kl * mb_num_nontrivial;
        epoch_kl_nontrivial_count += mb_num_nontrivial;
      }

      if (target_kl > 0.0 && kl > target_kl) {
        stats.early_stopped = true;
        stats.kl_early_stop_epoch = epoch;
        break;
      }
    }

    double ep_kl = (epoch_kl_nontrivial_count > 0) ? (epoch_kl_sum / epoch_kl_nontrivial_count) : 0.0;
    stats.epoch_kls.push_back(ep_kl);

    if (stats.early_stopped) break;
  }

  // Phase 18B combined-optimization diagnostics + per-update abort decision.
  if (aux_slice_uses > 0) {
    stats.aux_ce = aux_ce_sum / static_cast<double>(aux_slice_uses);
    stats.aux_value_mse = aux_vmse_sum / static_cast<double>(aux_slice_uses);
  }
  if (aux_norm_count > 0) {
    stats.aux_grad_norm_mean =
        aux_norm_sum / static_cast<double>(aux_norm_count);
    stats.ppo_grad_norm_mean =
        ppo_only_norm_sum / static_cast<double>(aux_norm_count);
    stats.aux_ppo_norm_ratio =
        (stats.ppo_grad_norm_mean > 0.0)
            ? (stats.aux_grad_norm_mean / stats.ppo_grad_norm_mean)
            : 0.0;
    // The caller (trainer) performs the clean abort at the latest valid
    // checkpoint; TrainPpoUpdate only flags it (so tests can assert the path).
    if (abort_grad_norm_ratio > 0.0 &&
        stats.aux_ppo_norm_ratio > abort_grad_norm_ratio) {
      stats.aux_ratio_abort = true;
    }
  }

  if (stats.minibatches > 0) {
    stats.value_loss = value_loss_sum / stats.minibatches;
    stats.value_clip_fraction = value_clip_frac_sum / stats.minibatches;
    if (total_nontrivial_count > 0) {
      stats.policy_loss = weighted_policy_loss_sum / total_nontrivial_count;
      stats.entropy = weighted_entropy_sum / total_nontrivial_count;
      stats.approx_kl = weighted_kl_sum / total_nontrivial_count;
      stats.clip_fraction = weighted_clip_fraction_sum / total_nontrivial_count;
    } else {
      stats.policy_loss = 0.0;
      stats.entropy = 0.0;
      stats.approx_kl = 0.0;
      stats.clip_fraction = 0.0;
    }
  }

  torch::Tensor returns_cpu_flat = returns_cpu;
  torch::Tensor old_values_cpu_flat = old_values_cpu;
  double return_var = returns_cpu_flat.var(/*unbiased=*/false).item<double>();
  if (return_var > 1e-12) {
    double residual_var =
        (returns_cpu_flat - old_values_cpu_flat)
            .var(/*unbiased=*/false)
            .item<double>();
    stats.explained_variance = 1.0 - residual_var / return_var;
  } else {
    stats.explained_variance = 0.0;
  }
  return stats;
}

// Linear-interpolated percentile over a sorted vector, matching the convention
// the frozen scripts/eval/logit_cap_audit.py uses so the two agree on the same
// data. `sorted` must be non-empty and ascending.
double SortedPercentile(const std::vector<float>& sorted, double q) {
  if (sorted.empty()) return 0.0;
  if (sorted.size() == 1) return sorted[0];
  const double pos = (q / 100.0) * (static_cast<double>(sorted.size()) - 1.0);
  const size_t lo = static_cast<size_t>(std::floor(pos));
  const size_t hi = static_cast<size_t>(std::ceil(pos));
  if (lo == hi) return sorted[lo];
  const double frac = pos - static_cast<double>(lo);
  return sorted[lo] * (1.0 - frac) + sorted[hi] * frac;
}

PrecapAbszStats SummarizePrecapAbsz(std::vector<float> values) {
  PrecapAbszStats out;
  out.decisions = static_cast<int64_t>(values.size());
  if (values.empty()) return out;
  std::sort(values.begin(), values.end());
  out.p95 = SortedPercentile(values, 95.0);
  out.p99 = SortedPercentile(values, 99.0);
  out.max = values.back();
  int64_t ge10 = 0;
  for (float v : values) {
    if (v >= 10.0f) ++ge10;
  }
  out.frac_ge10 = static_cast<double>(ge10) / static_cast<double>(values.size());
  return out;
}

// Diagnostics CSV schema v3 (Phase A, 2026-07 screens): v2 plus the trailing
// durable PPO-side canary columns. v1 -> v2 (WO-17) added the eight aux
// columns; v2 -> v3 adds only PPO-side columns and changes no existing one, so
// column meanings are stable across versions and only the tail grows.
//
// Appending is only safe against a file carrying this exact header, so a
// resumed run pointed at a v1 or v2 file fails loudly instead of writing rows
// the header cannot describe. JSONL is self-describing per line and needs no
// such gate.
//
// Deliberately NOT reusing the WO-17 column `ppo_grad_norm_mean` as the PPO
// grad-norm canary: that column is populated only when online collection fed
// aux examples and reads 0.0 with collection off. `grad_norm_mean` /
// `grad_norm_max` below are always populated.
const char* const kDiagnosticsCsvHeader =
    "seed,run_uuid,run_prefix,config_fingerprint,update,rollout_hash,episode_ids_unique,policy_kl_before,return_min,return_max,return_p50,"
    "return_p95,return_p99,abs_return_p99,fraction_targets_outside_1,fraction_critic_near_1,"
    "total_transitions,nontrivial_transitions,forced_transitions,epoch_kls,"
    "conflict_vp_generated,conflict_vp_attributed,conflict_vp_unattributed,"
    "raw_conflict_vp,raw_noncombat_vp,raw_total_vp,validation_kl,"
    "aux_examples_used,aux_search_loss_coef,aux_ce,aux_value_mse,"
    "aux_grad_norm_mean,ppo_grad_norm_mean,aux_ppo_norm_ratio,aux_ratio_abort,"
    "entropy,approx_kl,kl_early_stop,kl_early_stop_epoch,"
    "grad_norm_mean,grad_norm_max,value_loss,nonfinite_abort,"
    "precap_absz_n,precap_absz_p95,precap_absz_p99,precap_absz_max,frac_decisions_absz_ge10,"
    "precap_absz_n_primary,precap_absz_p95_primary,precap_absz_p99_primary,"
    "precap_absz_max_primary,frac_decisions_absz_ge10_primary,"
    "precap_absz_n_continuation,precap_absz_p95_continuation,precap_absz_p99_continuation,"
    "precap_absz_max_continuation,frac_decisions_absz_ge10_continuation,"
    "precap_absz_n_purchase,precap_absz_p95_purchase,precap_absz_p99_purchase,"
    "precap_absz_max_purchase,frac_decisions_absz_ge10_purchase,"
    // --- v4: WO-1 Phase 5 scratch-anomaly bundle (log-only) ---
    "value_clip_fraction,explained_variance,policy_head_grad_norm,"
    "value_head_grad_norm,trunk_grad_norm,minibatches_executed";

// Returns true if `filepath` already holds a header line, and sets `out_header`
// to it (trailing CR/LF stripped). An absent or zero-length file has no header.
bool ReadDiagnosticsCsvHeader(const std::string& filepath,
                              std::string* out_header) {
  std::ifstream ifs(filepath);
  if (!ifs) return false;
  std::string line;
  if (!std::getline(ifs, line)) return false;
  while (!line.empty() && (line.back() == '\r' || line.back() == '\n')) {
    line.pop_back();
  }
  *out_header = line;
  return true;
}

void WriteDiagnostics(const std::string& filepath, int update, const PpoUpdateStats& stats,
                      double conflict_vp_generated, double conflict_vp_attributed, double conflict_vp_unattributed,
                      uint64_t seed, const std::string& run_uuid, const std::string& run_prefix, const std::string& config_fingerprint,
                      double raw_conflict_vp, double raw_noncombat_vp, double raw_total_vp,
                      double validation_kl) {
  if (filepath.empty()) return;
  bool is_csv = (filepath.size() >= 4 && filepath.substr(filepath.size() - 4) == ".csv");

  if (is_csv) {
    std::string existing_header;
    const bool has_header = ReadDiagnosticsCsvHeader(filepath, &existing_header);
    if (has_header && existing_header != kDiagnosticsCsvHeader) {
      SpielFatalError(
          "Diagnostics CSV schema mismatch at: " + filepath +
          "\n  Existing header: " + existing_header +
          "\n  This binary writes: " + std::string(kDiagnosticsCsvHeader) +
          "\n  Appending would produce rows the header cannot describe. Move "
          "the old file aside or point --diagnostics_path at a new file.");
    }
    std::ofstream ofs(filepath, std::ios::app);
    if (!ofs) {
      SpielFatalError("Could not open diagnostics path: " + filepath);
    }
    if (!has_header) {
      ofs << kDiagnosticsCsvHeader << "\n";
    }
    std::string epoch_kls_str;
    for (size_t i = 0; i < stats.epoch_kls.size(); ++i) {
      epoch_kls_str += std::to_string(stats.epoch_kls[i]);
      if (i + 1 < stats.epoch_kls.size()) epoch_kls_str += ";";
    }
    ofs << seed << ","
        << run_uuid << ","
        << run_prefix << ","
        << config_fingerprint << ","
        << update << ","
        << stats.rollout_hash << ","
        << (stats.episode_ids_unique ? 1 : 0) << ","
        << stats.policy_kl_before << ","
        << stats.return_min << ","
        << stats.return_max << ","
        << stats.return_p50 << ","
        << stats.return_p95 << ","
        << stats.return_p99 << ","
        << stats.abs_return_p99 << ","
        << stats.fraction_targets_outside_1 << ","
        << stats.fraction_critic_near_1 << ","
        << stats.total_transitions << ","
        << stats.nontrivial_transitions << ","
        << stats.forced_transitions << ","
        << epoch_kls_str << ","
        << conflict_vp_generated << ","
        << conflict_vp_attributed << ","
        << conflict_vp_unattributed << ","
        << raw_conflict_vp << ","
        << raw_noncombat_vp << ","
        << raw_total_vp << ","
        << validation_kl << ","
        << stats.aux_examples_used << ","
        << stats.aux_search_loss_coef << ","
        << stats.aux_ce << ","
        << stats.aux_value_mse << ","
        << stats.aux_grad_norm_mean << ","
        << stats.ppo_grad_norm_mean << ","
        << stats.aux_ppo_norm_ratio << ","
        << (stats.aux_ratio_abort ? 1 : 0) << ","
        // --- v3: durable PPO-side canaries (Phase A) ---
        << stats.entropy << ","
        << stats.approx_kl << ","
        << (stats.early_stopped ? 1 : 0) << ","
        << stats.kl_early_stop_epoch << ","
        << stats.GradNormMean() << ","
        << stats.grad_norm_max << ","
        << stats.value_loss << ","
        << (stats.nonfinite_abort ? 1 : 0) << ","
        << stats.precap_absz_all.decisions << ","
        << stats.precap_absz_all.p95 << ","
        << stats.precap_absz_all.p99 << ","
        << stats.precap_absz_all.max << ","
        << stats.precap_absz_all.frac_ge10 << ","
        << stats.precap_absz_primary.decisions << ","
        << stats.precap_absz_primary.p95 << ","
        << stats.precap_absz_primary.p99 << ","
        << stats.precap_absz_primary.max << ","
        << stats.precap_absz_primary.frac_ge10 << ","
        << stats.precap_absz_continuation.decisions << ","
        << stats.precap_absz_continuation.p95 << ","
        << stats.precap_absz_continuation.p99 << ","
        << stats.precap_absz_continuation.max << ","
        << stats.precap_absz_continuation.frac_ge10 << ","
        << stats.precap_absz_purchase.decisions << ","
        << stats.precap_absz_purchase.p95 << ","
        << stats.precap_absz_purchase.p99 << ","
        << stats.precap_absz_purchase.max << ","
        << stats.precap_absz_purchase.frac_ge10 << ","
        // --- v4: WO-1 Phase 5 scratch-anomaly bundle ---
        << stats.value_clip_fraction << ","
        << stats.explained_variance << ","
        << stats.PolicyHeadGradNormMean() << ","
        << stats.ValueHeadGradNormMean() << ","
        << stats.TrunkGradNormMean() << ","
        // Minibatches actually executed, i.e. BEFORE the KL early stop cut the
        // update short. Pairs with kl_early_stop/kl_early_stop_epoch: a short
        // count with early_stop=0 means something else ended the update.
        << stats.minibatches << "\n";
    ofs.flush();
    if (!ofs) {
      SpielFatalError("Failed to write diagnostics CSV data to: " + filepath);
    }
    ofs.close();
    if (!ofs) {
      SpielFatalError("Failed to close diagnostics CSV file at: " + filepath);
    }
  } else {
    // JSONL format
    std::ofstream ofs(filepath, std::ios::app);
    if (!ofs) {
      SpielFatalError("Could not open diagnostics path: " + filepath);
    }
    open_spiel::json::Object obj;
    obj["seed"] = static_cast<int64_t>(seed);
    obj["run_uuid"] = run_uuid;
    obj["run_prefix"] = run_prefix;
    obj["config_fingerprint"] = config_fingerprint;
    obj["update"] = update;
    obj["rollout_hash"] = stats.rollout_hash;
    obj["episode_ids_unique"] = stats.episode_ids_unique;
    obj["policy_kl_before"] = stats.policy_kl_before;
    obj["return_min"] = stats.return_min;
    obj["return_max"] = stats.return_max;
    obj["return_p50"] = stats.return_p50;
    obj["return_p95"] = stats.return_p95;
    obj["return_p99"] = stats.return_p99;
    obj["abs_return_p99"] = stats.abs_return_p99;
    obj["fraction_targets_outside_1"] = stats.fraction_targets_outside_1;
    obj["fraction_critic_near_1"] = stats.fraction_critic_near_1;
    obj["total_transitions"] = static_cast<int64_t>(stats.total_transitions);
    obj["nontrivial_transitions"] = static_cast<int64_t>(stats.nontrivial_transitions);
    obj["forced_transitions"] = static_cast<int64_t>(stats.forced_transitions);

    open_spiel::json::Array kls_arr;
    for (double kl : stats.epoch_kls) kls_arr.push_back(kl);
    obj["epoch_kls"] = kls_arr;

    obj["conflict_vp_generated"] = conflict_vp_generated;
    obj["conflict_vp_attributed"] = conflict_vp_attributed;
    obj["conflict_vp_unattributed"] = conflict_vp_unattributed;
    obj["raw_conflict_vp"] = raw_conflict_vp;
    obj["raw_noncombat_vp"] = raw_noncombat_vp;
    obj["raw_total_vp"] = raw_total_vp;
    obj["validation_kl"] = validation_kl;

    // Phase 18B auxiliary-loss trajectory (WO-17): same names as the CSV
    // columns, so the two formats stay analysable interchangeably.
    obj["aux_examples_used"] = static_cast<int64_t>(stats.aux_examples_used);
    obj["aux_search_loss_coef"] = stats.aux_search_loss_coef;
    obj["aux_ce"] = stats.aux_ce;
    obj["aux_value_mse"] = stats.aux_value_mse;
    obj["aux_grad_norm_mean"] = stats.aux_grad_norm_mean;
    obj["ppo_grad_norm_mean"] = stats.ppo_grad_norm_mean;
    obj["aux_ppo_norm_ratio"] = stats.aux_ppo_norm_ratio;
    obj["aux_ratio_abort"] = stats.aux_ratio_abort;

    // Durable PPO-side canaries (Phase A): same names as the CSV columns, so
    // the two formats stay analysable interchangeably.
    obj["entropy"] = stats.entropy;
    obj["approx_kl"] = stats.approx_kl;
    obj["kl_early_stop"] = stats.early_stopped;
    obj["kl_early_stop_epoch"] = stats.kl_early_stop_epoch;
    obj["grad_norm_mean"] = stats.GradNormMean();
    obj["grad_norm_max"] = stats.grad_norm_max;
    obj["value_loss"] = stats.value_loss;
    obj["nonfinite_abort"] = stats.nonfinite_abort;
    // WO-1 Phase 5: same names as the CSV columns, so the two formats stay
    // analysable interchangeably.
    obj["value_clip_fraction"] = stats.value_clip_fraction;
    obj["explained_variance"] = stats.explained_variance;
    obj["policy_head_grad_norm"] = stats.PolicyHeadGradNormMean();
    obj["value_head_grad_norm"] = stats.ValueHeadGradNormMean();
    obj["trunk_grad_norm"] = stats.TrunkGradNormMean();
    obj["minibatches_executed"] = stats.minibatches;
    const auto add_absz = [&obj](const std::string& suffix,
                                 const PrecapAbszStats& z) {
      obj["precap_absz_n" + suffix] = static_cast<int64_t>(z.decisions);
      obj["precap_absz_p95" + suffix] = z.p95;
      obj["precap_absz_p99" + suffix] = z.p99;
      obj["precap_absz_max" + suffix] = z.max;
      obj["frac_decisions_absz_ge10" + suffix] = z.frac_ge10;
    };
    add_absz("", stats.precap_absz_all);
    add_absz("_primary", stats.precap_absz_primary);
    add_absz("_continuation", stats.precap_absz_continuation);
    add_absz("_purchase", stats.precap_absz_purchase);

    ofs << open_spiel::json::ToString(obj, false) << "\n";
    ofs.flush();
    if (!ofs) {
      SpielFatalError("Failed to write diagnostics JSONL data to: " + filepath);
    }
    ofs.close();
    if (!ofs) {
      SpielFatalError("Failed to close diagnostics JSONL file at: " + filepath);
    }
  }
}
#endif

} // namespace open_spiel
