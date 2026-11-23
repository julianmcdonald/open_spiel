#include "dune_ppo_training_utils_internal.h"
#include "dune_sha256.h"
#include "open_spiel/abseil-cpp/absl/flags/flag.h"
#include "open_spiel/abseil-cpp/absl/strings/str_format.h"
#include "open_spiel/games/dune_imperium/dune_imperium.h"
#include "open_spiel/utils/json.h"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace open_spiel {

bool ParseAndValidateDualManifest(
    const std::string& manifest_path,
    std::string& actor_model_path,
    std::string& actor_optim_path,
    std::string& critic_model_path,
    std::string& critic_optim_path,
    uint64_t current_base_seed,
    int current_target_end_update,
    int current_seed_scheme_version,
    const std::string& current_config_fingerprint,
    bool current_rollout_amp,
    bool current_allow_tf32,
    int current_hidden_dim,
    int current_num_blocks,
    DualCheckpointManifest& out_manifest,
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

  auto get_bool_field = [&](const std::string& key, bool& val) -> bool {
    auto it = manifest_obj.find(key);
    if (it == manifest_obj.end() || !it->second.IsBool()) {
      error_msg = "Manifest missing required bool field: " + key;
      return false;
    }
    val = it->second.GetBool();
    return true;
  };

  int64_t schema_version = 0;
  int64_t global_update = 0;
  int64_t target_end_update = 0;
  int64_t total_env_steps = 0;
  int64_t next_episode_id = 0;
  int64_t base_seed = 0;
  int64_t seed_scheme_version = 0;
  int64_t actor_model_file_size = 0;
  int64_t actor_optimizer_file_size = 0;
  int64_t critic_model_file_size = 0;
  int64_t critic_optimizer_file_size = 0;
  int64_t hidden_dim = 0;
  int64_t num_blocks = 0;

  if (!get_int_field("schema_version", schema_version) ||
      !get_string_field("checkpoint_uuid", out_manifest.checkpoint_uuid) ||
      !get_string_field("architecture", out_manifest.architecture) ||
      !get_bool_field("separate_actor_critic", out_manifest.separate_actor_critic) ||
      !get_int_field("global_update", global_update) ||
      !get_int_field("target_end_update", target_end_update) ||
      !get_int_field("total_env_steps", total_env_steps) ||
      !get_int_field("next_episode_id", next_episode_id) ||
      !get_int_field("base_seed", base_seed) ||
      !get_int_field("seed_scheme_version", seed_scheme_version) ||
      !get_string_field("config_fingerprint", out_manifest.config_fingerprint) ||
      !get_string_field("run_uuid", out_manifest.run_uuid) ||
      !get_string_field("model_filename", out_manifest.actor_model_filename) ||
      !get_int_field("model_file_size", actor_model_file_size) ||
      !get_string_field("model_sha256", out_manifest.actor_model_sha256) ||
      !get_string_field("optimizer_filename", out_manifest.actor_optimizer_filename) ||
      !get_int_field("optimizer_file_size", actor_optimizer_file_size) ||
      !get_string_field("optimizer_sha256", out_manifest.actor_optimizer_sha256) ||
      !get_string_field("critic_model_filename", out_manifest.critic_model_filename) ||
      !get_int_field("critic_model_file_size", critic_model_file_size) ||
      !get_string_field("critic_model_sha256", out_manifest.critic_model_sha256) ||
      !get_string_field("critic_optimizer_filename", out_manifest.critic_optimizer_filename) ||
      !get_int_field("critic_optimizer_file_size", critic_optimizer_file_size) ||
      !get_string_field("critic_optimizer_sha256", out_manifest.critic_optimizer_sha256) ||
      !get_int_field("hidden_dim", hidden_dim) ||
      !get_int_field("num_blocks", num_blocks)) {
    return false;
  }

  if (schema_version != 2) {
    error_msg = absl::StrFormat("schema_version mismatch. Expected: 2, Got: %d", schema_version);
    return false;
  }

  if (out_manifest.architecture != "separate_actor_critic" || !out_manifest.separate_actor_critic) {
    error_msg = "Manifest architecture is not separate_actor_critic.";
    return false;
  }

  out_manifest.schema_version = schema_version;
  out_manifest.global_update = global_update;
  out_manifest.target_end_update = target_end_update;
  out_manifest.total_env_steps = total_env_steps;
  out_manifest.next_episode_id = next_episode_id;
  out_manifest.base_seed = base_seed;
  out_manifest.seed_scheme_version = seed_scheme_version;
  out_manifest.actor_model_file_size = actor_model_file_size;
  out_manifest.actor_optimizer_file_size = actor_optimizer_file_size;
  out_manifest.critic_model_file_size = critic_model_file_size;
  out_manifest.critic_optimizer_file_size = critic_optimizer_file_size;
  out_manifest.hidden_dim = hidden_dim;
  out_manifest.num_blocks = num_blocks;

  std::filesystem::path manifest_dir = std::filesystem::path(manifest_path).parent_path();
  if (actor_model_path.empty()) {
    actor_model_path = (manifest_dir / out_manifest.actor_model_filename).string();
  }
  if (actor_optim_path.empty()) {
    actor_optim_path = (manifest_dir / out_manifest.actor_optimizer_filename).string();
  }
  if (critic_model_path.empty()) {
    critic_model_path = (manifest_dir / out_manifest.critic_model_filename).string();
  }
  if (critic_optim_path.empty()) {
    critic_optim_path = (manifest_dir / out_manifest.critic_optimizer_filename).string();
  }

  if (std::filesystem::path(actor_model_path).filename().string() != out_manifest.actor_model_filename) {
    error_msg = "Actor model filename mismatch. Manifest: " + out_manifest.actor_model_filename + ", Actual: " + std::filesystem::path(actor_model_path).filename().string();
    return false;
  }
  if (std::filesystem::path(actor_optim_path).filename().string() != out_manifest.actor_optimizer_filename) {
    error_msg = "Actor optimizer filename mismatch. Manifest: " + out_manifest.actor_optimizer_filename + ", Actual: " + std::filesystem::path(actor_optim_path).filename().string();
    return false;
  }
  if (std::filesystem::path(critic_model_path).filename().string() != out_manifest.critic_model_filename) {
    error_msg = "Critic model filename mismatch. Manifest: " + out_manifest.critic_model_filename + ", Actual: " + std::filesystem::path(critic_model_path).filename().string();
    return false;
  }
  if (std::filesystem::path(critic_optim_path).filename().string() != out_manifest.critic_optimizer_filename) {
    error_msg = "Critic optimizer filename mismatch. Manifest: " + out_manifest.critic_optimizer_filename + ", Actual: " + std::filesystem::path(critic_optim_path).filename().string();
    return false;
  }

  if (!std::filesystem::exists(actor_model_path)) {
    error_msg = "Actor model file not found: " + actor_model_path;
    return false;
  }
  if (!std::filesystem::exists(actor_optim_path)) {
    error_msg = "Actor optimizer file not found: " + actor_optim_path;
    return false;
  }
  if (!std::filesystem::exists(critic_model_path)) {
    error_msg = "Critic model file not found: " + critic_model_path;
    return false;
  }
  if (!std::filesystem::exists(critic_optim_path)) {
    error_msg = "Critic optimizer file not found: " + critic_optim_path;
    return false;
  }

  size_t actual_actor_size = 0;
  std::string actual_actor_hash = open_spiel::ComputeFileSHA256(actor_model_path, &actual_actor_size);
  if (actual_actor_size != out_manifest.actor_model_file_size) {
    error_msg = absl::StrFormat("Actor model size mismatch. Manifest: %d, Actual: %d", out_manifest.actor_model_file_size, actual_actor_size);
    return false;
  }
  if (actual_actor_hash != out_manifest.actor_model_sha256) {
    error_msg = "Actor model SHA-256 mismatch. Manifest: " + out_manifest.actor_model_sha256 + ", Actual: " + actual_actor_hash;
    return false;
  }

  size_t actual_actor_opt_size = 0;
  std::string actual_actor_opt_hash = open_spiel::ComputeFileSHA256(actor_optim_path, &actual_actor_opt_size);
  if (actual_actor_opt_size != out_manifest.actor_optimizer_file_size) {
    error_msg = absl::StrFormat("Actor optimizer size mismatch. Manifest: %d, Actual: %d", out_manifest.actor_optimizer_file_size, actual_actor_opt_size);
    return false;
  }
  if (actual_actor_opt_hash != out_manifest.actor_optimizer_sha256) {
    error_msg = "Actor optimizer SHA-256 mismatch. Manifest: " + out_manifest.actor_optimizer_sha256 + ", Actual: " + actual_actor_opt_hash;
    return false;
  }

  size_t actual_critic_size = 0;
  std::string actual_critic_hash = open_spiel::ComputeFileSHA256(critic_model_path, &actual_critic_size);
  if (actual_critic_size != out_manifest.critic_model_file_size) {
    error_msg = absl::StrFormat("Critic model size mismatch. Manifest: %d, Actual: %d", out_manifest.critic_model_file_size, actual_critic_size);
    return false;
  }
  if (actual_critic_hash != out_manifest.critic_model_sha256) {
    error_msg = "Critic model SHA-256 mismatch. Manifest: " + out_manifest.critic_model_sha256 + ", Actual: " + actual_critic_hash;
    return false;
  }

  size_t actual_critic_opt_size = 0;
  std::string actual_critic_opt_hash = open_spiel::ComputeFileSHA256(critic_optim_path, &actual_critic_opt_size);
  if (actual_critic_opt_size != out_manifest.critic_optimizer_file_size) {
    error_msg = absl::StrFormat("Critic optimizer size mismatch. Manifest: %d, Actual: %d", out_manifest.critic_optimizer_file_size, actual_critic_opt_size);
    return false;
  }
  if (actual_critic_opt_hash != out_manifest.critic_optimizer_sha256) {
    error_msg = "Critic optimizer SHA-256 mismatch. Manifest: " + out_manifest.critic_optimizer_sha256 + ", Actual: " + actual_critic_opt_hash;
    return false;
  }

  if (current_base_seed != 0 && current_base_seed != static_cast<uint64_t>(out_manifest.base_seed)) {
    error_msg = absl::StrFormat("Base seed mismatch. Current: %llu, Manifest: %llu",
                                static_cast<unsigned long long>(current_base_seed),
                                static_cast<unsigned long long>(out_manifest.base_seed));
    return false;
  }
  if (current_target_end_update <= out_manifest.global_update) {
    error_msg = absl::StrFormat("Target end update (%d) must be greater than manifest global update (%d)", current_target_end_update, out_manifest.global_update);
    return false;
  }
  if (current_seed_scheme_version != out_manifest.seed_scheme_version) {
    error_msg = absl::StrFormat("Seed scheme version mismatch. Current: %d, Manifest: %d", current_seed_scheme_version, out_manifest.seed_scheme_version);
    return false;
  }
  if (current_config_fingerprint != out_manifest.config_fingerprint) {
    error_msg = "Config fingerprint mismatch.\n  Expected: " + out_manifest.config_fingerprint + "\n  Got:      " + current_config_fingerprint;
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

  bool manifest_rollout_amp = false, manifest_allow_tf32 = false;
  if (get_bool_field("rollout_amp", manifest_rollout_amp) && manifest_rollout_amp != current_rollout_amp) {
    error_msg = absl::StrFormat("rollout_amp mismatch. Flags: %d, Manifest: %d", current_rollout_amp, manifest_rollout_amp);
    return false;
  }
  if (get_bool_field("allow_tf32", manifest_allow_tf32) && manifest_allow_tf32 != current_allow_tf32) {
    error_msg = absl::StrFormat("allow_tf32 mismatch. Flags: %d, Manifest: %d", current_allow_tf32, manifest_allow_tf32);
    return false;
  }

  return true;
}

void SaveDualCheckpoint(
    std::shared_ptr<SharedDunePolicyValueNetImpl> actor_model,
    torch::optim::AdamW& actor_optimizer,
    std::shared_ptr<SharedDunePolicyValueNetImpl> critic_model,
    torch::optim::AdamW& critic_optimizer,
    const std::string& actor_model_path,
    const std::string& actor_optim_path,
    const std::string& critic_model_path,
    const std::string& critic_optim_path,
    int global_update,
    int target_end_update,
    uint64_t total_env_steps,
    uint64_t next_episode_id,
    uint64_t base_seed,
    int seed_scheme_version,
    const std::string& config_fingerprint,
    const std::string& search_label_fingerprint,
    const std::string& run_uuid,
    const std::string& input_model_to_protect,
    const std::string& input_optim_to_protect) {
  if (actor_model_path.empty() || critic_model_path.empty() ||
      actor_optim_path.empty() || critic_optim_path.empty()) {
    SpielFatalError("SaveDualCheckpoint called with empty path.");
  }

  // Prevent overwriting input model or optimizer checkpoints
  if (!input_model_to_protect.empty()) {
    std::error_code ec;
    std::filesystem::path in_model = std::filesystem::absolute(input_model_to_protect, ec);
    std::filesystem::path out_actor = std::filesystem::absolute(actor_model_path, ec);
    std::filesystem::path out_critic = std::filesystem::absolute(critic_model_path, ec);
    if (in_model == out_actor || in_model == out_critic) {
      SpielFatalError("CRITICAL BASELINE OVERWRITE ERROR: SaveDualCheckpoint attempted to overwrite input model checkpoint: " + input_model_to_protect);
    }
    if (std::filesystem::exists(input_model_to_protect)) {
      if ((std::filesystem::exists(actor_model_path) && std::filesystem::equivalent(actor_model_path, input_model_to_protect, ec)) ||
          (std::filesystem::exists(critic_model_path) && std::filesystem::equivalent(critic_model_path, input_model_to_protect, ec))) {
        SpielFatalError("CRITICAL BASELINE OVERWRITE ERROR: SaveDualCheckpoint attempted to overwrite input model checkpoint: " + input_model_to_protect);
      }
    }
  }
  if (!input_optim_to_protect.empty()) {
    std::error_code ec;
    std::filesystem::path in_optim = std::filesystem::absolute(input_optim_to_protect, ec);
    std::filesystem::path out_actor_opt = std::filesystem::absolute(actor_optim_path, ec);
    std::filesystem::path out_critic_opt = std::filesystem::absolute(critic_optim_path, ec);
    if (in_optim == out_actor_opt || in_optim == out_critic_opt) {
      SpielFatalError("CRITICAL BASELINE OVERWRITE ERROR: SaveDualCheckpoint attempted to overwrite input optimizer checkpoint: " + input_optim_to_protect);
    }
    if (std::filesystem::exists(input_optim_to_protect)) {
      if ((std::filesystem::exists(actor_optim_path) && std::filesystem::equivalent(actor_optim_path, input_optim_to_protect, ec)) ||
          (std::filesystem::exists(critic_optim_path) && std::filesystem::equivalent(critic_optim_path, input_optim_to_protect, ec))) {
        SpielFatalError("CRITICAL BASELINE OVERWRITE ERROR: SaveDualCheckpoint attempted to overwrite input optimizer checkpoint: " + input_optim_to_protect);
      }
    }
  }

  std::string actor_tmp = actor_model_path + ".tmp";
  std::string actor_optim_tmp = actor_optim_path + ".tmp";
  std::string critic_tmp = critic_model_path + ".tmp";
  std::string critic_optim_tmp = critic_optim_path + ".tmp";

  std::filesystem::path manifest_path = actor_model_path;
  manifest_path.replace_extension(".json");
  std::string manifest_path_str = manifest_path.string();
  std::string manifest_tmp = manifest_path_str + ".tmp";

  try {
    torch::save(actor_model, actor_tmp);
    torch::save(actor_optimizer, actor_optim_tmp);
    torch::save(critic_model, critic_tmp);
    torch::save(critic_optimizer, critic_optim_tmp);

    size_t actor_size = 0;
    std::string actor_hash = ComputeFileSHA256(actor_tmp, &actor_size);
    size_t actor_opt_size = 0;
    std::string actor_opt_hash = ComputeFileSHA256(actor_optim_tmp, &actor_opt_size);
    size_t critic_size = 0;
    std::string critic_hash = ComputeFileSHA256(critic_tmp, &critic_size);
    size_t critic_opt_size = 0;
    std::string critic_opt_hash = ComputeFileSHA256(critic_optim_tmp, &critic_opt_size);

    std::string checkpoint_uuid = GenerateUUID();
    json::Object manifest_obj;
    manifest_obj["schema_version"] = static_cast<int64_t>(2);
    manifest_obj["checkpoint_uuid"] = checkpoint_uuid;
    manifest_obj["global_update"] = static_cast<int64_t>(global_update);
    manifest_obj["target_end_update"] = static_cast<int64_t>(target_end_update);
    manifest_obj["total_env_steps"] = static_cast<int64_t>(total_env_steps);
    manifest_obj["next_episode_id"] = static_cast<int64_t>(next_episode_id);
    manifest_obj["base_seed"] = static_cast<int64_t>(base_seed);
    manifest_obj["seed_scheme_version"] = static_cast<int64_t>(seed_scheme_version);
    manifest_obj["config_fingerprint"] = config_fingerprint;
    WritePpoPrecisionManifestFields(
        manifest_obj, absl::GetFlag(FLAGS_rollout_amp),
        absl::GetFlag(FLAGS_allow_tf32));
    manifest_obj["search_label_fingerprint"] = search_label_fingerprint;
    manifest_obj["run_uuid"] = run_uuid;

    manifest_obj["model_filename"] = std::filesystem::path(actor_model_path).filename().string();
    manifest_obj["model_file_size"] = static_cast<int64_t>(actor_size);
    manifest_obj["model_sha256"] = actor_hash;
    manifest_obj["optimizer_filename"] = std::filesystem::path(actor_optim_path).filename().string();
    manifest_obj["optimizer_file_size"] = static_cast<int64_t>(actor_opt_size);
    manifest_obj["optimizer_sha256"] = actor_opt_hash;

    manifest_obj["hidden_dim"] = static_cast<int64_t>(actor_model->input_layer->weight.size(0));
    manifest_obj["num_blocks"] = static_cast<int64_t>(actor_model->res_blocks.size());
    manifest_obj["observation_dim"] = static_cast<int64_t>(actor_model->input_layer->weight.size(1));
    manifest_obj["market_appendix_mode"] = absl::GetFlag(FLAGS_market_appendix_mode);
    const auto resolved_deck_pool_semantics = dune_imperium::ParseDeckPoolSemanticsOrDie(absl::GetFlag(FLAGS_deck_pool_semantics));
    manifest_obj["deck_pool_semantics"] = dune_imperium::DeckPoolSemanticsToString(resolved_deck_pool_semantics);
    const auto schema = dune_imperium::GetInformationStateSchema(
        dune_imperium::ParseMarketAppendixMode(absl::GetFlag(FLAGS_market_appendix_mode)),
        resolved_deck_pool_semantics);
    if (static_cast<int64_t>(actor_model->input_layer->weight.size(1)) == schema.size) {
      manifest_obj["feature_schema_label"] = std::string(schema.label);
      manifest_obj["feature_schema_sha256"] = std::string(schema.sha256);
    }
    manifest_obj["enable_semantic_scorer"] = absl::GetFlag(FLAGS_enable_semantic_scorer);
    if (absl::GetFlag(FLAGS_enable_semantic_scorer)) {
      manifest_obj["semantic_descriptor_schema"] = actor_model->semantic_descriptor_schema_;
      if (actor_model->semantic_scorer_ != nullptr && actor_model->semantic_scorer_->slot_indexed_f_) {
        manifest_obj["slot_indexed_out_layer_f"] = true;
      }
    }
    manifest_obj["enable_zone_encoder"] = actor_model->enable_zone_encoder_;
    manifest_obj["specimen_exchange_penalty"] = absl::GetFlag(FLAGS_specimen_exchange_penalty);
    manifest_obj["family_atomics_penalty"] = absl::GetFlag(FLAGS_family_atomics_penalty);
    manifest_obj["plot_intrigue_penalty"] = absl::GetFlag(FLAGS_plot_intrigue_penalty);
    manifest_obj["plot_intrigue_exemption_threshold"] = static_cast<int64_t>(absl::GetFlag(FLAGS_plot_intrigue_exemption_threshold));
    manifest_obj["rollout_games"] = static_cast<int64_t>(absl::GetFlag(FLAGS_rollout_games));
    manifest_obj["random_leader_draft"] = absl::GetFlag(FLAGS_random_leader_draft);

    if (!g_reward_transition_source_fingerprint.empty()) {
      manifest_obj["reward_transition_source_fingerprint"] = g_reward_transition_source_fingerprint;
      if (g_reward_transition_source_update >= 0) {
        manifest_obj["reward_transition_source_update"] = g_reward_transition_source_update;
      }
    }
    if (!g_gae_transition_source_fingerprint.empty()) {
      manifest_obj["gae_transition_source_fingerprint"] = g_gae_transition_source_fingerprint;
      if (g_gae_transition_source_update >= 0) {
        manifest_obj["gae_transition_source_update"] = g_gae_transition_source_update;
      }
    }
    if (!g_learning_rate_transition_source_fingerprint.empty()) {
      manifest_obj["learning_rate_transition_source_fingerprint"] = g_learning_rate_transition_source_fingerprint;
      if (g_learning_rate_transition_source_update >= 0) {
        manifest_obj["learning_rate_transition_source_update"] = g_learning_rate_transition_source_update;
      }
    }
    if (!g_target_kl_transition_source_fingerprint.empty()) {
      manifest_obj["target_kl_transition_source_fingerprint"] = g_target_kl_transition_source_fingerprint;
      if (g_target_kl_transition_source_update >= 0) {
        manifest_obj["target_kl_transition_source_update"] = g_target_kl_transition_source_update;
      }
    }
    if (!g_single_learner_transition_source_fingerprint.empty()) {
      manifest_obj["single_learner_transition_source_fingerprint"] = g_single_learner_transition_source_fingerprint;
      if (g_single_learner_transition_source_update >= 0) {
        manifest_obj["single_learner_transition_source_update"] = g_single_learner_transition_source_update;
      }
    }
    if (!g_ema_magnet_transition_source_fingerprint.empty()) {
      manifest_obj["ema_magnet_transition_source_fingerprint"] = g_ema_magnet_transition_source_fingerprint;
      if (g_ema_magnet_transition_source_update >= 0) {
        manifest_obj["ema_magnet_transition_source_update"] = g_ema_magnet_transition_source_update;
      }
    }
    if (!g_market_appendix_transition_source_fingerprint.empty()) {
      manifest_obj["market_appendix_transition_source_fingerprint"] = g_market_appendix_transition_source_fingerprint;
      if (g_market_appendix_transition_source_update >= 0) {
        manifest_obj["market_appendix_transition_source_update"] = g_market_appendix_transition_source_update;
      }
    }
    if (!g_deck_pool_semantics_transition_source_fingerprint.empty()) {
      manifest_obj["deck_pool_semantics_transition_source_fingerprint"] = g_deck_pool_semantics_transition_source_fingerprint;
      if (g_deck_pool_semantics_transition_source_update >= 0) {
        manifest_obj["deck_pool_semantics_transition_source_update"] = g_deck_pool_semantics_transition_source_update;
      }
    }
    if (!g_tf32_transition_source_fingerprint.empty()) {
      manifest_obj["tf32_transition_source_fingerprint"] = g_tf32_transition_source_fingerprint;
      if (g_tf32_transition_source_update >= 0) {
        manifest_obj["tf32_transition_source_update"] = g_tf32_transition_source_update;
      }
    }
    if (!g_semantic_schema_transition_source_fingerprint.empty()) {
      manifest_obj["semantic_schema_transition_source_fingerprint"] = g_semantic_schema_transition_source_fingerprint;
      if (g_semantic_schema_transition_source_update >= 0) {
        manifest_obj["semantic_schema_transition_source_update"] = g_semantic_schema_transition_source_update;
      }
    }
    if (absl::GetFlag(FLAGS_ema_magnet_kl_coef) > 0.0) {
      manifest_obj["ema_magnet_kl_coef"] = absl::GetFlag(FLAGS_ema_magnet_kl_coef);
    }

    manifest_obj["architecture"] = "separate_actor_critic";
    manifest_obj["separate_actor_critic"] = true;
    manifest_obj["value_head_status"] = "actor_value_head_detached_and_stale";
    manifest_obj["evaluation_role"] = "actor_only_no_value_head";
    manifest_obj["warning"] =
        "CRITICAL: Value head in this actor checkpoint is detached and was never updated after U200. Evaluators requiring leaf values must load the paired critic checkpoint, not this actor.";

    manifest_obj["critic_model_filename"] = std::filesystem::path(critic_model_path).filename().string();
    manifest_obj["critic_model_file_size"] = static_cast<int64_t>(critic_size);
    manifest_obj["critic_model_sha256"] = critic_hash;
    manifest_obj["critic_optimizer_filename"] = std::filesystem::path(critic_optim_path).filename().string();
    manifest_obj["critic_optimizer_file_size"] = static_cast<int64_t>(critic_opt_size);
    manifest_obj["critic_optimizer_sha256"] = critic_opt_hash;
    manifest_obj["critic_observation_dim"] = static_cast<int64_t>(critic_model->input_layer->weight.size(1));

    {
      std::ofstream ofs(manifest_tmp);
      if (!ofs) {
        throw std::runtime_error("Could not open manifest temp file for writing: " + manifest_tmp);
      }
      ofs << json::ToString(manifest_obj, true);
    }

    std::filesystem::rename(actor_tmp, actor_model_path);
    std::filesystem::rename(actor_optim_tmp, actor_optim_path);
    std::filesystem::rename(critic_tmp, critic_model_path);
    std::filesystem::rename(critic_optim_tmp, critic_optim_path);
    std::filesystem::rename(manifest_tmp, manifest_path);

    std::cout << "Saved dual checkpoint successfully:\n"
              << "  Actor Model: " << actor_model_path << " (" << actor_size << " bytes, sha256=" << actor_hash << ")\n"
              << "  Actor Optimizer: " << actor_optim_path << " (" << actor_opt_size << " bytes, sha256=" << actor_opt_hash << ")\n"
              << "  Critic Model: " << critic_model_path << " (" << critic_size << " bytes, sha256=" << critic_hash << ")\n"
              << "  Critic Optimizer: " << critic_optim_path << " (" << critic_opt_size << " bytes, sha256=" << critic_opt_hash << ")\n"
              << "  Manifest: " << manifest_path << "\n";
  } catch (const std::exception& e) {
    std::cerr << "CRITICAL ERROR: SaveDualCheckpoint failed: " << e.what() << "\n";
    if (std::filesystem::exists(actor_tmp)) std::filesystem::remove(actor_tmp);
    if (std::filesystem::exists(actor_optim_tmp)) std::filesystem::remove(actor_optim_tmp);
    if (std::filesystem::exists(critic_tmp)) std::filesystem::remove(critic_tmp);
    if (std::filesystem::exists(critic_optim_tmp)) std::filesystem::remove(critic_optim_tmp);
    if (std::filesystem::exists(manifest_tmp)) std::filesystem::remove(manifest_tmp);
    SpielFatalError("SaveDualCheckpoint failed, aborting training to prevent corrupted states.");
  }
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

void WriteCheckpointMetadataFile(const std::string& json_path,
                                 const CheckpointMetadataSpec& spec) {
  json::Object manifest_obj;
  manifest_obj["observation_dim"] = spec.observation_dim;
  manifest_obj["hidden_dim"] = spec.hidden_dim;
  manifest_obj["num_blocks"] = spec.num_blocks;
  manifest_obj["market_appendix_mode"] = spec.market_appendix_mode;
  manifest_obj["deck_pool_semantics"] = dune_imperium::DeckPoolSemanticsToString(spec.deck_pool_semantics);
  const auto schema = dune_imperium::GetInformationStateSchema(
      dune_imperium::ParseMarketAppendixMode(spec.market_appendix_mode),
      spec.deck_pool_semantics);
  manifest_obj["feature_schema_label"] = spec.feature_schema_label.empty()
      ? std::string(schema.label)
      : spec.feature_schema_label;
  manifest_obj["feature_schema_sha256"] = spec.feature_schema_sha256.empty()
      ? std::string(schema.sha256)
      : spec.feature_schema_sha256;
  manifest_obj["enable_semantic_scorer"] = spec.enable_semantic_scorer;
  if (spec.enable_semantic_scorer && !spec.semantic_descriptor_schema.empty()) {
    manifest_obj["semantic_descriptor_schema"] = spec.semantic_descriptor_schema;
  }
  manifest_obj["enable_zone_encoder"] = spec.enable_zone_encoder;
  std::ofstream ofs(json_path);
  if (!ofs) {
    SpielFatalError("WriteCheckpointMetadataFile: Failed to open " + json_path);
  }
  ofs << json::ToString(manifest_obj, true);
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
  for (int i = 0; i < OnlineCollectionState::kNumSearchRoles; ++i) {
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
      // Bounded by BOTH the array we are filling and the array on disk, so a
      // three-entry pre-Leader manifest reads cleanly and leaves leader at zero.
      for (int i = 0;
           i < OnlineCollectionState::kNumSearchRoles &&
           i < static_cast<int>(a.size());
           ++i) {
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

}  // namespace open_spiel
