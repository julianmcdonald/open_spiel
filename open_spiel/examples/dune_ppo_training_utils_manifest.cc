#include "dune_ppo_training_utils_internal.h"
#include "dune_sha256.h"
#include "open_spiel/abseil-cpp/absl/flags/flag.h"
#include "open_spiel/abseil-cpp/absl/strings/str_format.h"
#include "open_spiel/games/dune_imperium/dune_imperium.h"
#include "open_spiel/utils/json.h"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

namespace open_spiel {

std::string ComputePrePrecisionConfigFingerprint(
    const json::Object& pre_precision_config) {
  return ComputeStringSHA256(json::ToString(pre_precision_config));
}

std::string ComputePrecisionConfigFingerprint(
    const json::Object& pre_precision_config, bool rollout_amp,
    bool allow_tf32) {
  json::Object current = pre_precision_config;
  current["rollout_amp"] = rollout_amp;
  current["allow_tf32"] = allow_tf32;
  return ComputeStringSHA256(json::ToString(current));
}

void WritePpoPrecisionManifestFields(json::Object& manifest_obj,
                                     bool rollout_amp, bool allow_tf32) {
  manifest_obj["rollout_amp"] = rollout_amp;
  manifest_obj["allow_tf32"] = allow_tf32;
}

bool ValidatePpoPrecisionManifestCompatibility(
    const json::Object& manifest_obj,
    const std::string& stored_config_fingerprint,
    const std::string& current_config_fingerprint,
    const std::string& current_pre_precision_config_fingerprint,
    const std::string& current_legacy_config_fingerprint,
    bool current_rollout_amp, bool current_allow_tf32,
    PpoPrecisionManifestCompatibility* out, std::string* error,
    const std::string& permitted_transition_source_fingerprint) {
  auto fail = [&](const std::string& message) {
    if (error != nullptr) *error = message;
    return false;
  };
  if (out == nullptr) return fail("null precision compatibility output");
  *out = PpoPrecisionManifestCompatibility{};
  const auto rollout_it = manifest_obj.find("rollout_amp");
  const auto tf32_it = manifest_obj.find("allow_tf32");
  const bool rollout_present = rollout_it != manifest_obj.end();
  const bool tf32_present = tf32_it != manifest_obj.end();
  if (rollout_present != tf32_present) {
    return fail(
        "Precision manifest fields are partial: rollout_amp and allow_tf32 "
        "must either both be present or both be absent");
  }
  if (rollout_present) {
    if (!rollout_it->second.IsBool() || !tf32_it->second.IsBool()) {
      return fail(
          "Precision manifest fields rollout_amp and allow_tf32 must both be booleans");
    }
    out->fields_present = true;
    out->rollout_amp = rollout_it->second.GetBool();
    out->allow_tf32 = tf32_it->second.GetBool();
    if (out->rollout_amp != current_rollout_amp) {
      if (permitted_transition_source_fingerprint.empty() ||
          stored_config_fingerprint != permitted_transition_source_fingerprint) {
        return fail(absl::StrFormat(
            "rollout_amp mismatch. Current: %s, Manifest: %s",
            current_rollout_amp ? "true" : "false",
            out->rollout_amp ? "true" : "false"));
      }
    }
    if (out->allow_tf32 != current_allow_tf32) {
      if (permitted_transition_source_fingerprint.empty() ||
          stored_config_fingerprint != permitted_transition_source_fingerprint) {
        return fail(absl::StrFormat(
            "allow_tf32 mismatch. Current: %s, Manifest: %s",
            current_allow_tf32 ? "true" : "false",
            out->allow_tf32 ? "true" : "false"));
      }
    }
    if (stored_config_fingerprint != current_config_fingerprint) {
      if (!permitted_transition_source_fingerprint.empty() &&
          stored_config_fingerprint == permitted_transition_source_fingerprint) {
        out->reward_transition = true;
        out->reward_transition_source_fingerprint = stored_config_fingerprint;
        return true;
      }
      return fail(
          "Configuration fingerprint mismatch for precision-aware manifest.\n  Expected: " +
          stored_config_fingerprint + "\n  Got:      " +
          current_config_fingerprint);
    }
    return true;
  }

  if (!current_rollout_amp || !current_allow_tf32) {
    return fail(
        "Precision fields are absent, but migration is permitted only with "
        "default rollout_amp=true and allow_tf32=true");
  }
  out->legacy_precision_migration = true;
  if (!permitted_transition_source_fingerprint.empty() &&
      stored_config_fingerprint == permitted_transition_source_fingerprint) {
    out->reward_transition = true;
    out->reward_transition_source_fingerprint = stored_config_fingerprint;
    return true;
  }
  if (!current_pre_precision_config_fingerprint.empty() &&
      stored_config_fingerprint == current_pre_precision_config_fingerprint) {
    out->migration_source = "pre_precision_fingerprint";
    return true;
  }
  if (!current_legacy_config_fingerprint.empty() &&
      stored_config_fingerprint == current_legacy_config_fingerprint) {
    out->migration_source = "older_legacy_fingerprint";
    return true;
  }
  return fail(
      "Configuration fingerprint mismatch for field-absent precision manifest.\n"
      "  Stored:        " + stored_config_fingerprint +
      "\n  Pre-precision: " + current_pre_precision_config_fingerprint +
      (current_legacy_config_fingerprint.empty()
           ? std::string("\n  Older legacy:  disabled")
           : std::string("\n  Older legacy:  ") +
                 current_legacy_config_fingerprint));
}

std::string ValidateResumeDeckPoolSemanticsOrDie(
    const json::Object& manifest_obj,
    const std::string& flag_semantics,
    const std::string& context_label,
    bool allow_transition,
    int64_t source_update) {
  std::string manifest_sem = "legacy";
  auto sem_it = manifest_obj.find("deck_pool_semantics");
  if (sem_it != manifest_obj.end() && sem_it->second.IsString()) {
    manifest_sem = sem_it->second.GetString();
  }
  (void)dune_imperium::ParseDeckPoolSemanticsOrDie(manifest_sem);
  (void)dune_imperium::ParseDeckPoolSemanticsOrDie(flag_semantics);

  bool effective_allow =
      allow_transition || absl::GetFlag(FLAGS_allow_deck_pool_semantics_transition);

  int64_t effective_source_update = source_update;
  if (effective_source_update < 0) {
    effective_source_update = absl::GetFlag(FLAGS_deck_pool_semantics_transition_source_update);
  }
  if (effective_source_update < 0 && absl::GetFlag(FLAGS_start_update) > 0) {
    effective_source_update = absl::GetFlag(FLAGS_start_update);
  }

  if (!effective_allow) {
    if (manifest_sem != flag_semantics) {
      SpielFatalError(absl::StrFormat(
          "deck_pool_semantics mismatch on %s. Manifest: '%s', Flag: '%s'",
          context_label, manifest_sem, flag_semantics));
    }
    return manifest_sem;
  }

  // effective_allow is true. Validate one-time transition on segment 1.
  if (manifest_obj.find("deck_pool_semantics_transition_source_update") != manifest_obj.end() ||
      manifest_obj.find("deck_pool_semantics_transition_source_fingerprint") != manifest_obj.end()) {
    SpielFatalError(absl::StrFormat(
        "deck_pool_semantics transition on %s is invalid: deck_pool_semantics transition already occurred in a prior segment",
        context_label));
  }

  int64_t manifest_update = -1;
  auto up_it = manifest_obj.find("global_update");
  if (up_it != manifest_obj.end() && up_it->second.IsInt()) {
    manifest_update = up_it->second.GetInt();
  }

  if (effective_source_update < 0 || manifest_update < 0 ||
      manifest_update != effective_source_update) {
    SpielFatalError(absl::StrFormat(
        "deck_pool_semantics transition on %s is only valid when manifest global_update (%lld) equals source_update (%lld)",
        context_label, static_cast<long long>(manifest_update),
        static_cast<long long>(effective_source_update)));
  }

  if (manifest_sem == flag_semantics) {
    SpielFatalError(absl::StrFormat(
        "deck_pool_semantics transition flag set on %s at update %lld, but manifest semantics ('%s') already match flag ('%s')",
        context_label, static_cast<long long>(manifest_update),
        manifest_sem, flag_semantics));
  }

  return manifest_sem;
}

bool ParseAndValidateManifest(const std::string& manifest_path,
                              const std::string& model_path,
                              const std::string& optim_path,
                              uint64_t current_base_seed,
                              int current_target_end_update,
                              int current_seed_scheme_version,
                              const std::string& current_config_fingerprint,
                              const std::string& current_pre_precision_config_fingerprint,
                              bool current_rollout_amp,
                              bool current_allow_tf32,
                              const std::string& current_search_label_fingerprint,
                              int current_hidden_dim,
                              int current_num_blocks,
                              CheckpointManifest& out_manifest,
                              std::string& error_msg,
                              const std::string& current_legacy_config_fingerprint,
                              const std::string& permitted_transition_source_fingerprint) {
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
  if (current_target_end_update <= out_manifest.global_update) {
    error_msg = absl::StrFormat("Target end update (%d) must be greater than manifest global update (%d)", current_target_end_update, out_manifest.global_update);
    return false;
  }
  if (current_seed_scheme_version != out_manifest.seed_scheme_version) {
    error_msg = absl::StrFormat("Seed scheme version mismatch. Current: %d, Manifest: %d", current_seed_scheme_version, out_manifest.seed_scheme_version);
    return false;
  }

  PpoPrecisionManifestCompatibility precision;
  if (!ValidatePpoPrecisionManifestCompatibility(
          manifest_obj, out_manifest.config_fingerprint,
          current_config_fingerprint,
          current_pre_precision_config_fingerprint,
          current_legacy_config_fingerprint, current_rollout_amp,
          current_allow_tf32, &precision, &error_msg,
          permitted_transition_source_fingerprint)) {
    return false;
  }
  out_manifest.precision_fields_present = precision.fields_present;
  out_manifest.rollout_amp = precision.rollout_amp;
  out_manifest.allow_tf32 = precision.allow_tf32;
  out_manifest.legacy_precision_migration =
      precision.legacy_precision_migration;
  out_manifest.precision_migration_source = precision.migration_source;
  out_manifest.reward_transition = precision.reward_transition;
  out_manifest.reward_transition_source_fingerprint =
      precision.reward_transition_source_fingerprint;
  if (precision.legacy_precision_migration) {
    std::cout
        << "[precision-manifest] Accepted legacy-precision migration from "
        << precision.migration_source
        << " under required defaults rollout_amp=true allow_tf32=true. "
           "The next checkpoint will write both precision fields and the "
           "current fingerprint.\n";
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

bool CheckBundleIntegrity(const std::string& manifest_path,
                          const std::string& model_path,
                          const std::string& optim_path,
                          std::string* error_msg) {
  std::error_code ec;
  if (!std::filesystem::exists(manifest_path, ec) || ec) {
    if (error_msg) *error_msg = "manifest file not found at: " + manifest_path;
    return false;
  }

  std::ifstream ifs(manifest_path);
  if (!ifs) {
    if (error_msg) *error_msg = "Could not open manifest file at: " + manifest_path;
    return false;
  }
  std::string content((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
  auto val_opt = open_spiel::json::FromString(content);
  if (!val_opt.has_value() || !val_opt->IsObject()) {
    if (error_msg) *error_msg = "Manifest file is malformed JSON at: " + manifest_path;
    return false;
  }
  const auto& manifest_obj = val_opt->GetObject();

  auto it_m_size = manifest_obj.find("model_file_size");
  auto it_m_sha = manifest_obj.find("model_sha256");
  if (it_m_size == manifest_obj.end() || !it_m_size->second.IsInt() ||
      it_m_sha == manifest_obj.end() || !it_m_sha->second.IsString()) {
    if (error_msg) *error_msg = "Manifest missing model_file_size or model_sha256: " + manifest_path;
    return false;
  }
  int64_t exp_model_size = it_m_size->second.GetInt();
  std::string exp_model_sha = it_m_sha->second.GetString();

  if (!std::filesystem::exists(model_path, ec) || ec) {
    if (error_msg) *error_msg = "Model file not found: " + model_path;
    return false;
  }
  uintmax_t actual_model_size = std::filesystem::file_size(model_path, ec);
  if (ec || static_cast<int64_t>(actual_model_size) != exp_model_size) {
    if (error_msg) {
      *error_msg = absl::StrFormat("Model file size mismatch. Manifest: %d, Actual: %d",
                                   exp_model_size, actual_model_size);
    }
    return false;
  }
  size_t bytes_read = 0;
  std::string actual_model_sha = open_spiel::ComputeFileSHA256(model_path, &bytes_read);
  if (actual_model_sha != exp_model_sha) {
    if (error_msg) {
      *error_msg = "Model SHA-256 hash mismatch. Manifest: " + exp_model_sha + ", Actual: " + actual_model_sha;
    }
    return false;
  }

  // Check optimizer if tracked in manifest and optim_path is provided
  auto it_o_size = manifest_obj.find("optimizer_file_size");
  auto it_o_sha = manifest_obj.find("optimizer_sha256");
  if (it_o_size != manifest_obj.end() && it_o_size->second.IsInt() &&
      it_o_sha != manifest_obj.end() && it_o_sha->second.IsString() &&
      !it_o_sha->second.GetString().empty() && it_o_size->second.GetInt() > 0) {
    int64_t exp_optim_size = it_o_size->second.GetInt();
    std::string exp_optim_sha = it_o_sha->second.GetString();

    if (optim_path.empty() || !std::filesystem::exists(optim_path, ec) || ec) {
      if (error_msg) *error_msg = "Optimizer file not found: " + optim_path;
      return false;
    }
    uintmax_t actual_optim_size = std::filesystem::file_size(optim_path, ec);
    if (ec || static_cast<int64_t>(actual_optim_size) != exp_optim_size) {
      if (error_msg) {
        *error_msg = absl::StrFormat("Optimizer file size mismatch. Manifest: %d, Actual: %d",
                                     exp_optim_size, actual_optim_size);
      }
      return false;
    }
    std::string actual_optim_sha = open_spiel::ComputeFileSHA256(optim_path, &bytes_read);
    if (actual_optim_sha != exp_optim_sha) {
      if (error_msg) {
        *error_msg = "Optimizer SHA-256 hash mismatch. Manifest: " + exp_optim_sha + ", Actual: " + actual_optim_sha;
      }
      return false;
    }
  }

  return true;
}

void RecoverInterruptedBundleIfNeeded(const std::string& model_path,
                                      const std::string& optim_path,
                                      const std::string& manifest_path) {
  std::string manifest_bak = manifest_path + ".bak";
  std::string model_bak = model_path + ".bak";
  std::string optim_bak = optim_path.empty() ? "" : (optim_path + ".bak");

  std::error_code ec;
  bool any_bak = std::filesystem::exists(manifest_bak, ec) ||
                 std::filesystem::exists(model_bak, ec) ||
                 (!optim_bak.empty() && std::filesystem::exists(optim_bak, ec));
  if (!any_bak) {
    return;
  }

  std::string active_err;
  bool active_ok = CheckBundleIntegrity(manifest_path, model_path, optim_path, &active_err);
  if (active_ok) {
    // Active bundle is fully committed and self-consistent!
    // Never roll back a valid committed new generation merely because leftover backups exist.
    // Clean up stale leftover .bak files.
    if (std::filesystem::exists(manifest_bak, ec)) std::filesystem::remove(manifest_bak, ec);
    if (std::filesystem::exists(model_bak, ec)) std::filesystem::remove(model_bak, ec);
    if (!optim_bak.empty() && std::filesystem::exists(optim_bak, ec)) {
      std::filesystem::remove(optim_bak, ec);
    }
    std::cout << "[RECOVERY] Fully committed active bundle verified; cleaned up stale .bak files.\n";
    return;
  }

  // Active bundle is incomplete or mixed (e.g. interruption mid-publication).
  // Determine backup sources. If a previous partial recovery already restored
  // model or optimizer to the active path, allow that verified active file to serve
  // as the source rather than refusing recovery because model.bak was consumed.
  std::string model_src = std::filesystem::exists(model_bak, ec) ? model_bak : model_path;
  std::string optim_src = (!optim_bak.empty() && std::filesystem::exists(optim_bak, ec)) ? optim_bak : optim_path;
  std::string manifest_src = std::filesystem::exists(manifest_bak, ec) ? manifest_bak : manifest_path;

  std::string backup_err;
  bool backup_ok = CheckBundleIntegrity(manifest_src, model_src, optim_src, &backup_err);
  if (!backup_ok) {
    SpielFatalError(absl::StrFormat(
        "Interrupted publication detected: active bundle is incomplete/corrupted (%s), "
        "and backup bundle is also invalid or missing (%s). Cannot recover automatically.",
        active_err, backup_err));
  }

  // Backup bundle is valid! Restore prior bundle without consuming source backups.
  // We copy each artifact into place, verify the complete active bundle, and ONLY
  // then delete the backups.
  std::cout << "[RECOVERY] Interrupted publication detected: mixed active bundle restored from valid .bak files.\n";

  auto safe_copy_restore = [&](const std::string& src, const std::string& dst, const std::string& label) {
    if (src.empty() || dst.empty() || src == dst) return;
    std::error_code eq_ec;
    if (std::filesystem::exists(dst, eq_ec) && std::filesystem::equivalent(src, dst, eq_ec)) {
      return;  // Already identical inode (e.g. hardlink)
    }
    std::error_code cp_ec;
    std::filesystem::copy_file(src, dst, std::filesystem::copy_options::overwrite_existing, cp_ec);
    if (cp_ec) {
      SpielFatalError("Failed to restore " + label + " from backup (" + src + " -> " + dst + "): " + cp_ec.message());
    }
  };

  safe_copy_restore(model_src, model_path, "model");
  safe_copy_restore(optim_src, optim_path, "optimizer");
  safe_copy_restore(manifest_src, manifest_path, "manifest");

  // Verify the newly restored active bundle before deleting any backups!
  std::string restored_verify_err;
  if (!CheckBundleIntegrity(manifest_path, model_path, optim_path, &restored_verify_err)) {
    SpielFatalError("Restored active bundle failed integrity verification: " + restored_verify_err);
  }

  // Now that the restored active bundle is fully verified, clean up .bak and .tmp files.
  if (std::filesystem::exists(model_bak, ec)) std::filesystem::remove(model_bak, ec);
  if (!optim_bak.empty() && std::filesystem::exists(optim_bak, ec)) std::filesystem::remove(optim_bak, ec);
  if (std::filesystem::exists(manifest_bak, ec)) std::filesystem::remove(manifest_bak, ec);

  // Clean up any leftover .tmp files from the interrupted save
  std::string model_tmp = model_path + ".tmp";
  std::string optim_tmp = optim_path.empty() ? "" : (optim_path + ".tmp");
  std::string manifest_tmp = manifest_path + ".tmp";
  if (std::filesystem::exists(model_tmp, ec)) std::filesystem::remove(model_tmp, ec);
  if (!optim_tmp.empty() && std::filesystem::exists(optim_tmp, ec)) std::filesystem::remove(optim_tmp, ec);
  if (std::filesystem::exists(manifest_tmp, ec)) std::filesystem::remove(manifest_tmp, ec);
}

dune_imperium::DeckPoolSemantics ReadCheckpointDeckPoolSemantics(
    const std::string& checkpoint_path) {
  std::filesystem::path metadata_path(checkpoint_path);
  metadata_path.replace_extension(".json");
  if (!std::filesystem::exists(metadata_path)) {
    SpielFatalError("Missing checkpoint sidecar JSON: " + metadata_path.string());
  }
  std::ifstream input(metadata_path);
  if (!input) {
    SpielFatalError("Missing checkpoint sidecar JSON: " + metadata_path.string());
  }
  const std::string content((std::istreambuf_iterator<char>(input)), {});
  const auto metadata = open_spiel::json::FromString(content);
  if (!metadata || !metadata->IsObject()) {
    SpielFatalError("Invalid checkpoint sidecar JSON: " + metadata_path.string());
  }
  const auto& object = metadata->GetObject();
  dune_imperium::DeckPoolSemantics sem = dune_imperium::DeckPoolSemantics::kLegacy;
  auto it = object.find("deck_pool_semantics");
  if (it != object.end() && it->second.IsString()) {
    sem = dune_imperium::ParseDeckPoolSemanticsOrDie(it->second.GetString());
  }
  auto mode_it = object.find("market_appendix_mode");
  auto hash_it = object.find("feature_schema_sha256");
  if (mode_it != object.end() && mode_it->second.IsString() &&
      hash_it != object.end() && hash_it->second.IsString()) {
    const auto mode = dune_imperium::ParseMarketAppendixMode(mode_it->second.GetString());
    const auto schema = dune_imperium::GetInformationStateSchema(mode, sem);
    if (hash_it->second.GetString() != schema.sha256) {
      SpielFatalError("ReadCheckpointDeckPoolSemantics: Checkpoint " + checkpoint_path +
                      " feature_schema_sha256 (" + hash_it->second.GetString() +
                      ") does not match schema (" + std::string(schema.sha256) + ")");
    }
  }
  return sem;
}

dune_imperium::MarketAppendixMode ReadCheckpointInputMode(
    const std::string& checkpoint_path, int input_dim) {
  std::filesystem::path metadata_path(checkpoint_path);
  metadata_path.replace_extension(".json");
  if (!std::filesystem::exists(metadata_path)) {
    SpielFatalError("Missing card-slot checkpoint metadata: " + metadata_path.string());
  }
  std::ifstream input(metadata_path);
  if (!input) {
    SpielFatalError("Missing card-slot checkpoint metadata: " + metadata_path.string());
  }
  const std::string content((std::istreambuf_iterator<char>(input)), {});
  const auto metadata = open_spiel::json::FromString(content);
  if (!metadata || !metadata->IsObject()) {
    SpielFatalError("Invalid card-slot checkpoint metadata: " + metadata_path.string());
  }
  const auto& object = metadata->GetObject();

  if (input_dim == dune_imperium::kLegacyInformationStateSize) {
    dune_imperium::DeckPoolSemantics sem = dune_imperium::DeckPoolSemantics::kLegacy;
    auto sem_entry = object.find("deck_pool_semantics");
    if (sem_entry != object.end() && sem_entry->second.IsString()) {
      sem = dune_imperium::ParseDeckPoolSemanticsOrDie(sem_entry->second.GetString());
    }
    auto hash_entry = object.find("feature_schema_sha256");
    if (hash_entry != object.end() && hash_entry->second.IsString()) {
      const auto schema = dune_imperium::GetInformationStateSchema(dune_imperium::MarketAppendixMode::kNone, sem);
      if (hash_entry->second.GetString() != schema.sha256) {
        SpielFatalError("ReadCheckpointInputMode: Schema hash mismatch for legacy input");
      }
    }
    return dune_imperium::MarketAppendixMode::kNone;
  }

  SPIEL_CHECK_TRUE(dune_imperium::IsExtendedInformationStateSize(input_dim));
  auto mode_entry = object.find("market_appendix_mode");
  auto dim_entry = object.find("observation_dim");
  auto hash_entry = object.find("feature_schema_sha256");
  if (mode_entry == object.end() || !mode_entry->second.IsString() ||
      dim_entry == object.end() || !dim_entry->second.IsInt() ||
      hash_entry == object.end() || !hash_entry->second.IsString()) {
    SpielFatalError("Incomplete card-slot checkpoint schema: " + metadata_path.string());
  }
  const auto mode = dune_imperium::ParseMarketAppendixMode(mode_entry->second.GetString());
  dune_imperium::DeckPoolSemantics sem = dune_imperium::DeckPoolSemantics::kLegacy;
  auto sem_entry = object.find("deck_pool_semantics");
  if (sem_entry != object.end() && sem_entry->second.IsString()) {
    sem = dune_imperium::ParseDeckPoolSemanticsOrDie(sem_entry->second.GetString());
  }
  const auto schema = dune_imperium::GetInformationStateSchema(mode, sem);
  SPIEL_CHECK_EQ(schema.size, input_dim);
  SPIEL_CHECK_EQ(dim_entry->second.GetInt(), input_dim);
  SPIEL_CHECK_EQ(hash_entry->second.GetString(), std::string(schema.sha256));
  return mode;
}

}  // namespace open_spiel
