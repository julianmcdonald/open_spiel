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
    uint64_t master, int global_update) {
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

  // --- PPO Loss Loop ---
  double weighted_policy_loss_sum = 0.0;
  double weighted_entropy_sum = 0.0;
  double weighted_kl_sum = 0.0;
  double weighted_clip_fraction_sum = 0.0;
  int64_t total_nontrivial_count = 0;
  double value_loss_sum = 0.0;

  for (int epoch = 0; epoch < update_epochs; ++epoch) {
    uint64_t perm_seed = dune_seed::DeriveSeed(master, dune_seed::kDomainTrain, global_update, epoch, dune_seed::kStreamPPOPermutation);
    at::Generator gen = dune_seed::MakeTorchCPUGenerator(perm_seed);
    torch::Tensor permutation =
        torch::randperm(n, gen, torch::TensorOptions().device(torch::kCPU).dtype(torch::kInt64)).to(device);

    double epoch_kl_sum = 0.0;
    int64_t epoch_kl_nontrivial_count = 0;

    for (int64_t start = 0; start < n; start += minibatch_size) {
      int64_t end = std::min(start + minibatch_size, n);
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
      torch::Tensor policy_loss, value_loss, entropy, approx_kl, clip_fraction;
      torch::Tensor total_loss;

      auto compute_loss = [&]() {
        auto outputs = model->forward(mb_states);
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
          // If mb_num_nontrivial <= 1, standard deviation is undefined or zero, use raw.
        }
        mb_adv = mb_adv.detach();

        // 2. PPO surrogate loss
        torch::Tensor pg_loss1 = -mb_adv * ratio;
        torch::Tensor pg_loss2 =
            -mb_adv * ratio.clamp(1.0f - clip_epsilon,
                                  1.0f + clip_epsilon);
        torch::Tensor pg_loss = torch::max(pg_loss1, pg_loss2);

        // 3. Critic loss (Retain all samples)
        torch::Tensor new_values = outputs.values.squeeze(1);
        if (clip_value_loss) {
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
      double grad_norm =
          torch::nn::utils::clip_grad_norm_(
              model->parameters(), ::absl::GetFlag(::FLAGS_grad_clip_norm));
      if (std::isnan(grad_norm) || std::isinf(grad_norm)) {
        std::cerr << "Fatal PPO gradient norm: " << grad_norm << "\n";
        std::exit(EXIT_FAILURE);
      }
      stats.grad_norm_sum += grad_norm;
      stats.grad_norm_count += 1;
      optimizer.step();

      double kl = approx_kl.item<double>();
      value_loss_sum += value_loss.item<double>();
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
        break;
      }
    }

    double ep_kl = (epoch_kl_nontrivial_count > 0) ? (epoch_kl_sum / epoch_kl_nontrivial_count) : 0.0;
    stats.epoch_kls.push_back(ep_kl);

    if (stats.early_stopped) break;
  }

  if (stats.minibatches > 0) {
    stats.value_loss = value_loss_sum / stats.minibatches;
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

void WriteDiagnostics(const std::string& filepath, int update, const PpoUpdateStats& stats,
                      double conflict_vp_generated, double conflict_vp_attributed, double conflict_vp_unattributed,
                      uint64_t seed, const std::string& run_uuid, const std::string& run_prefix, const std::string& config_fingerprint,
                      double raw_conflict_vp, double raw_noncombat_vp, double raw_total_vp) {
  if (filepath.empty()) return;
  bool is_csv = (filepath.size() >= 4 && filepath.substr(filepath.size() - 4) == ".csv");

  if (is_csv) {
    bool file_exists = std::filesystem::exists(filepath);
    std::ofstream ofs(filepath, std::ios::app);
    if (!ofs) {
      SpielFatalError("Could not open diagnostics path: " + filepath);
    }
    if (!file_exists) {
      ofs << "seed,run_uuid,run_prefix,config_fingerprint,update,rollout_hash,episode_ids_unique,policy_kl_before,return_min,return_max,return_p50,"
             "return_p95,return_p99,abs_return_p99,fraction_targets_outside_1,fraction_critic_near_1,"
             "total_transitions,nontrivial_transitions,forced_transitions,epoch_kls,"
             "conflict_vp_generated,conflict_vp_attributed,conflict_vp_unattributed,"
             "raw_conflict_vp,raw_noncombat_vp,raw_total_vp\n";
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
        << raw_total_vp << "\n";
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
