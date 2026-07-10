#include "dune_ppo_training_utils.h"
#include "dune_sha256.h"
#include "open_spiel/abseil-cpp/absl/strings/str_format.h"

namespace open_spiel {

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
                              std::string& error_msg) {
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

  if (!get_int_field("global_update", global_update) ||
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
    error_msg = "Configuration fingerprint mismatch. Effective hyperparameters changed.\n  Expected: " + out_manifest.config_fingerprint + "\n  Got:      " + current_config_fingerprint;
    return false;
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

} // namespace open_spiel
