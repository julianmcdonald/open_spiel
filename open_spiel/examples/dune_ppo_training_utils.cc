#include "dune_ppo_training_utils.h"
#include "dune_sha256.h"
#include "dune_search_routing.h"  // DuneDecisionRole, for the PWO-5 canary role order
#include "open_spiel/abseil-cpp/absl/strings/str_format.h"
#include "open_spiel/utils/json.h"
#include "open_spiel/games/dune_imperium/dune_imperium.h"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>

#ifdef OPEN_SPIEL_BUILD_WITH_LIBTORCH
#include "open_spiel/abseil-cpp/absl/flags/declare.h"
#include "open_spiel/abseil-cpp/absl/flags/flag.h"
#include "dune_seed_utils.h"
#include <c10/core/Event.h>
#include <c10/core/impl/VirtualGuardImpl.h>
#include <atomic>
#include <chrono>
#include <mutex>
#include <optional>
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
// --- WO-PERF-1 (2026-08-14): diagnostics-only performance flags. Both
// defaults reproduce today's behavior bit-for-bit; neither statistic has any
// consumer outside diagnostics (PPO Phase-0 findings, 2026-07-29).
ABSL_FLAG(std::string, diag_prepass_mode, "full",
          "Phase 5 diagnostics pre-pass mode. 'full' (default): today's "
          "behavior, an FP32 no-grad forward over the whole rollout every "
          "update. 'cadenced': fraction_critic_near_1 is computed exactly "
          "from each transition's stored rollout value (no forward), and "
          "policy_kl_before is measured over the full rollout only at process "
          "startup/resume and every --diag_prepass_interval updates, in the "
          "same autocast precision as rollout inference when on CUDA.");
ABSL_FLAG(int, diag_prepass_interval, 25,
          "With --diag_prepass_mode=cadenced, measure policy_kl_before on "
          "updates where global_update %% this interval == 0 (absolute, like "
          "--checkpoint_interval), plus always on the first update of the "
          "process. <= 0 means startup/resume only.");
ABSL_FLAG(std::string, grad_telemetry_mode, "per_param",
          "Per-module gradient-norm telemetry. 'per_param' (default): today's "
          "behavior, one device-to-host .item() read per parameter tensor per "
          "minibatch (70 reads/minibatch at production size). 'accumulated': "
          "the six group sums stay on-device in float64 and are read back "
          "once per update; values are identical on CPU by construction.");
// --- WO-PERF-TIMING (2026-08-16): per-phase PPO wall attribution. -----------
// Declared here, beside the WO-PERF-1 flags, so it registers in every binary
// linking this TU -- the same pattern for the same reason.
ABSL_FLAG(std::string, phase_timing_mode, "off",
          "off|phases -- per-phase PPO wall attribution. Default 'off' is "
          "bit-for-bit today's behavior: no timers armed, no CUDA events "
          "created, no clock read, no sidecar written. 'phases' brackets the "
          "eight PPO phases with CUDA events on the compute stream (read once "
          "after a single end-of-update synchronise) and writes a "
          "phase_timing.jsonl sidecar beside --diagnostics_path. The mode "
          "perturbs what it measures; the sidecar reports its own sync cost, "
          "its event count and the trainer's PPO figure so the discrepancy is "
          "visible. NOT part of ComputeConfigFingerprint(), so a checkpoint "
          "manifest cannot detect a mid-run flip: every timed arm must be a "
          "FRESH run, never a resume. Rejected with --pipeline=true, whose "
          "background collection thread shares the compute stream.");
#endif

namespace open_spiel {

// PWO-5 gate 2 item (b). The term is SUBTRACTED, so a positive penalty lowers
// the reward. The range predicate is the shared 741-752 one, never an inline
// `base .. base + 12`. See dune_specimen_conversion.h for why 740 is excluded.
// `reward_lambda` is a FLOAT at the one call site, and the arithmetic below is
// written in float for that reason: the original in-line statement was
// `reward -= static_cast<float>(penalty) * reward_lambda`, all-float, and a
// double intermediate here would be a last-ulp behaviour change smuggled in
// under a refactor.
float ApplySpecimenExchangeShaping(float reward, Action action,
                                   double specimen_exchange_penalty,
                                   float reward_lambda) {
  if (specimen_exchange_penalty == 0.0) return reward;
  if (!dune_shaping::IsSpecimenConversionAction(action)) return reward;
  return reward - static_cast<float>(specimen_exchange_penalty) * reward_lambda;
}

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
    PpoPrecisionManifestCompatibility* out, std::string* error) {
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
      return fail(absl::StrFormat(
          "rollout_amp mismatch. Current: %s, Manifest: %s",
          current_rollout_amp ? "true" : "false",
          out->rollout_amp ? "true" : "false"));
    }
    if (out->allow_tf32 != current_allow_tf32) {
      return fail(absl::StrFormat(
          "allow_tf32 mismatch. Current: %s, Manifest: %s",
          current_allow_tf32 ? "true" : "false",
          out->allow_tf32 ? "true" : "false"));
    }
    if (stored_config_fingerprint != current_config_fingerprint) {
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

  PpoPrecisionManifestCompatibility precision;
  if (!ValidatePpoPrecisionManifestCompatibility(
          manifest_obj, out_manifest.config_fingerprint,
          current_config_fingerprint,
          current_pre_precision_config_fingerprint,
          current_legacy_config_fingerprint, current_rollout_amp,
          current_allow_tf32, &precision, &error_msg)) {
    return false;
  }
  out_manifest.precision_fields_present = precision.fields_present;
  out_manifest.rollout_amp = precision.rollout_amp;
  out_manifest.allow_tf32 = precision.allow_tf32;
  out_manifest.legacy_precision_migration =
      precision.legacy_precision_migration;
  out_manifest.precision_migration_source = precision.migration_source;
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

#ifdef OPEN_SPIEL_BUILD_WITH_LIBTORCH
namespace {
// WO-PERF-1 (R2 review fix F2): the "first TrainPpoUpdate in this process"
// marker is keyed PER MODEL (module identity), not process-global, so a
// process that trains two models (e.g. a pilot arm and a live arm) force-
// measures each model's own first update. A resume is always a new process,
// so first-call-per-model-in-process is exactly the "startup/resume"
// condition the cadenced pre-pass must always measure on. Only touched in
// --diag_prepass_mode=cadenced; full mode never reaches it.
std::mutex g_diag_prepass_seen_mu;
std::unordered_set<const void*> g_diag_prepass_seen_models;
// Returns true exactly once per model identity per process (per reset).
bool ConsumeFirstPrepassForModel(const void* model_identity) {
  std::lock_guard<std::mutex> lock(g_diag_prepass_seen_mu);
  return g_diag_prepass_seen_models.insert(model_identity).second;
}
}  // namespace

void ResetDiagPrepassStateForTesting() {
  std::lock_guard<std::mutex> lock(g_diag_prepass_seen_mu);
  g_diag_prepass_seen_models.clear();
}

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

// --- WO-PERF-TIMING: the phase timer. ---------------------------------------
//
// Ordered exactly as the work order's §3.2 table. This array is the single
// source of truth for the order; the sidecar's field list is generated from
// it, so a reordering cannot silently mislabel a column.
const char* const kPhaseTimingNames[kPhaseTimingNumPhases] = {
    "tensor_pack_h2d", "diag_prepass",  "ppo_forward_loss", "backward",
    "grad_telemetry",  "grad_clip",     "optimizer_step",   "scalar_reads",
};

namespace {

// The CUDA event pool, REUSED across updates.
//
// PhaseTimer is a per-update local, so a pool owned by the timer would create
// and destroy every event on every update -- at the production shape that is
// ~2,100 cudaEventCreateWithFlags/cudaEventDestroy pairs per update, paid
// forever, for no benefit. Hoisting the pool here makes the reuse the earlier
// comment claimed but did not implement: events are created once, then
// re-recorded each update. Re-recording is well defined (cudaEventRecord
// overwrites) and safe here because Finalize reads every pair's elapsed time
// before the update returns, so no pair is ever re-recorded between its record
// and its read.
//
// thread_local, not a plain static: TrainPpoUpdate is called on one thread, but
// a pool shared across threads would be a data race waiting for the first
// caller that changes that.
std::vector<c10::Event>& PhaseEventPool() {
  static thread_local std::vector<c10::Event> pool;
  return pool;
}

}  // namespace

PhaseTimer::PhaseTimer(bool enabled, torch::Device device)
    : enabled_(enabled),
      cuda_(enabled && device.is_cuda()),
      device_(device) {
  for (int i = 0; i < kPhaseTimingNumPhases; ++i) open_event_[i] = kNoEvent;
  if (cuda_) impl_.emplace(device.type());
}

size_t PhaseTimer::AcquirePair() {
  auto& pool = PhaseEventPool();
  while (next_free_ + 1 >= pool.size()) {
    pool.emplace_back(device_.type(), c10::EventFlag::BACKEND_DEFAULT);
    pool.emplace_back(device_.type(), c10::EventFlag::BACKEND_DEFAULT);
  }
  const size_t idx = next_free_;
  next_free_ += 2;
  return idx;
}

void PhaseTimer::Begin(int phase) {
  if (!enabled_) return;
  host_begin_[phase] = std::chrono::steady_clock::now();
  if (!cuda_) return;
  const size_t idx = AcquirePair();
  open_event_[phase] = idx;
  PhaseEventPool()[idx].record(impl_->getStream(device_));
  ++events_recorded_;
}

void PhaseTimer::End(int phase) {
  if (!enabled_) return;
  if (cuda_) {
    const size_t idx = open_event_[phase];
    // An End without a matching Begin would otherwise index the pool blind.
    // Dropping the bracket is the safe failure: it lands in the reported
    // residual instead of fabricating an interval.
    if (idx == kNoEvent) return;
    PhaseEventPool()[idx + 1].record(impl_->getStream(device_));
    ++events_recorded_;
    pairs_.push_back({phase, idx});
    last_recorded_ = idx + 1;
    open_event_[phase] = kNoEvent;
  }
  host_s_[phase] += std::chrono::duration<double>(
                        std::chrono::steady_clock::now() - host_begin_[phase])
                        .count();
  ++bracket_count_;
}

void PhaseTimer::Finalize(PpoPhaseTimings* out) {
  if (!enabled_) return;
  out->enabled = true;
  out->cuda = cuda_;
  out->bracket_count = bracket_count_;
  out->events_recorded = events_recorded_;

  if (cuda_ && last_recorded_ != kNoEvent) {
    auto& pool = PhaseEventPool();
    // Every event was recorded on the stream returned by getStream() for this
    // device, and stream events complete in stream order, so waiting on the
    // LAST one is sufficient for all of them. One synchronise, never one per
    // phase -- a per-phase sync would serialise the pipeline and inflate the
    // very total this is trying to attribute (work order §3.3).
    const auto sync_begin = std::chrono::steady_clock::now();
    pool[last_recorded_].synchronize();
    out->sync_s = std::chrono::duration<double>(
                      std::chrono::steady_clock::now() - sync_begin)
                      .count();
    for (const auto& p : pairs_) {
      // elapsedTime is milliseconds on every backend that implements it.
      out->device_s[p.phase] +=
          pool[p.start].elapsedTime(pool[p.start + 1]) * 1e-3;
    }
  }

  for (int i = 0; i < kPhaseTimingNumPhases; ++i) {
    out->host_s[i] = host_s_[i];
    // On CPU there is no second clock to consult: steady_clock IS the
    // measurement, and the async hazard that motivates events does not arise.
    // Populating the field keeps the sidecar schema uniform across devices
    // instead of emitting nulls a consumer must special-case.
    if (!cuda_) out->device_s[i] = host_s_[i];
    out->total_attributed_s += out->device_s[i];
    out->total_host_attributed_s += out->host_s[i];
  }
}

PpoUpdateStats TrainPpoUpdate(
    std::shared_ptr<SharedDunePolicyValueNetImpl> model,
    torch::optim::AdamW& optimizer, std::vector<PpoTransition>& batch,
    int64_t obs_size, int64_t action_dim, torch::Device device,
    uint64_t master, int global_update,
    std::shared_ptr<SharedDunePolicyValueNetImpl> anchor_model,
    const std::vector<SearchTrainingExample>& search_examples,
    double search_loss_coef, double abort_grad_norm_ratio,
    const Pwo5AuxBatch& pwo5_aux, const Pwo5AuxConfig& pwo5_cfg) {
  PpoUpdateStats stats;
  if (batch.empty()) return stats;
  stats.rollout_hash = ComputeRolloutHash(batch);

  int64_t n = static_cast<int64_t>(batch.size());
  auto cpu_float = torch::TensorOptions().dtype(torch::kFloat32);
  auto cpu_bool = torch::TensorOptions().dtype(torch::kBool);
  auto cpu_long = torch::TensorOptions().dtype(torch::kInt64);

  // WO-PERF-TIMING. Validated on EVERY path, including the default, so a typo
  // is fatal at the first update rather than silently timing nothing for a
  // whole run. Construction of a disabled timer allocates nothing.
  const std::string phase_timing_mode =
      ::absl::GetFlag(::FLAGS_phase_timing_mode);
  const bool phase_timing_on = (phase_timing_mode == "phases");
  if (!phase_timing_on && phase_timing_mode != "off") {
    SpielFatalError("Unknown --phase_timing_mode: '" + phase_timing_mode +
                    "' (expected 'off' or 'phases').");
  }
  // NOTE: the --pipeline incompatibility is enforced at startup in
  // dune_ppo_train.cc, where both flags are defined. It cannot live here --
  // FLAGS_pipeline is defined in the trainer's own TU, and six other targets
  // link this file without it.
  PhaseTimer phase_timer(phase_timing_on, device);

  phase_timer.Begin(kPhaseTensorPackH2D);
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
  phase_timer.End(kPhaseTensorPackH2D);

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

  // 3. Policy KL Before & Saturation Checks
  //
  // WO-PERF-1: two modes. 'full' is the pre-WO merged FP32 forward pre-pass,
  // bit-for-bit. 'cadenced' takes the full-rollout forward off the hot path:
  // the saturation metric comes exactly from the stored rollout values (the
  // actual behavior-policy statistic rather than an FP32 replay of it), and
  // the policy_kl_before canary is measured only at process startup/resume
  // and on the --diag_prepass_interval cadence, in rollout-inference autocast
  // precision on CUDA. Neither statistic has any consumer outside
  // diagnostics; stats.measured_transitions records how many nontrivial
  // transitions the KL was actually measured over (-1 = "not measured this
  // update", 0 = "measured, but zero nontrivial transitions"; either way an
  // unmeasured update can never be read as KL == 0).
  const std::string diag_prepass_mode =
      ::absl::GetFlag(::FLAGS_diag_prepass_mode);
  const bool prepass_cadenced = (diag_prepass_mode == "cadenced");
  if (!prepass_cadenced && diag_prepass_mode != "full") {
    SpielFatalError("Unknown --diag_prepass_mode: '" + diag_prepass_mode +
                    "' (expected 'full' or 'cadenced').");
  }

  double kl_before_sum = 0.0;
  int64_t kl_before_nontrivial_count = 0;
  // R2 review fix F1: distinguishes "KL not measured this update" (emitted as
  // measured_transitions = -1) from "measured over zero nontrivial
  // transitions" (0). Full mode measures every update, so it stays true there.
  bool kl_measured_this_update = true;

  phase_timer.Begin(kPhaseDiagPrepass);
  if (!prepass_cadenced) {
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
  } else {
    // (1) Saturation exactly from the stored rollout-time values -- the same
    // |v| >= 0.99f predicate the full mode applies to its replayed values,
    // applied to PpoTransition.value (already copied into old_values_cpu).
    int64_t stored_near_one_count = 0;
    for (int64_t i = 0; i < n; ++i) {
      if (std::abs(old_values_ptr[i]) >= 0.99f) ++stored_near_one_count;
    }
    stats.fraction_critic_near_1 = static_cast<double>(stored_near_one_count) / n;

    // (2) The policy_kl_before canary, at startup/resume and on the cadence.
    const int prepass_interval = ::absl::GetFlag(::FLAGS_diag_prepass_interval);
    const bool first_update_for_model =
        ConsumeFirstPrepassForModel(static_cast<const void*>(model.get()));
    const bool measure_kl_now =
        first_update_for_model ||
        (prepass_interval > 0 && global_update % prepass_interval == 0);
    kl_measured_this_update =
        measure_kl_now && !::absl::GetFlag(::FLAGS_train_value_only);
    if (kl_measured_this_update) {
      bool was_training = model->is_training();
      model->eval();
      {
        torch::NoGradGuard no_grad;
        // Same autocast setup as rollout inference (dune_evaluator.h):
        // BF16 on CUDA, disabled (plain FP32) on CPU. A NEW scope -- the
        // PWO-5 section 7.5 cache-clearing guard, not a restructuring of the
        // existing training autocast scopes below.
        AutocastGuard autocast_guard(device.type(), device.is_cuda());
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
        }
      }
      if (was_training) {
        model->train();
      }
    }
    stats.policy_kl_before = (kl_before_nontrivial_count > 0) ? (kl_before_sum / kl_before_nontrivial_count) : 0.0;
  }
  phase_timer.End(kPhaseDiagPrepass);
  stats.measured_transitions =
      kl_measured_this_update ? kl_before_nontrivial_count : int64_t{-1};

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
    // Populates phases 1-2 for a caller that wants them. NOTE: the only
    // --diagnostics_only call site (dune_ppo_train.cc) writes diagnostics and
    // then exits WITHOUT calling WritePhaseTiming, so no sidecar record is
    // emitted on this path. Stated plainly rather than left to look like
    // coverage that does not exist.
    phase_timer.Finalize(&stats.phase_timings);
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

  // --- PWO-5 section 8.6: slice the update's 1,024 auxiliary rows across the
  // PPO minibatches, with the SAME ComputeAuxSlices pattern the 18B path uses.
  //
  // The slices are RE-CONSUMED EACH EPOCH (this partition sits outside the
  // epoch loop and mb_index is the index within an epoch), so the realized
  // presentation count is 1,024 x (epochs actually executed) <= 4,096 --
  // "<=" because --target_kl can truncate the epoch loop. Both the realized
  // presentation count and the realized DISTINCT-row count are emitted per
  // update so the exposure is measured, not assumed.
  const bool pwo5_active = pwo5_aux.valid && pwo5_cfg.AnyActive();
  const int64_t pwo5_n =
      pwo5_active ? pwo5_aux.obs.size(0) : static_cast<int64_t>(0);
  std::vector<int64_t> pwo5_slice_start, pwo5_slice_len;
  if (pwo5_active) {
    const int64_t num_mb_p = (n + minibatch_size - 1) / minibatch_size;
    auto pslices = ComputeAuxSlices(pwo5_n, num_mb_p);
    pwo5_slice_start.assign(num_mb_p, 0);
    pwo5_slice_len.assign(num_mb_p, 0);
    for (int64_t k = 0; k < num_mb_p && k < (int64_t)pslices.size(); ++k) {
      pwo5_slice_start[k] = pslices[k].first;
      pwo5_slice_len[k] = pslices[k].second;
    }
    stats.pwo5_distinct_rows = pwo5_n;
  }
  double pwo5_final_vp_sum = 0.0, pwo5_terminal_round_sum = 0.0,
         pwo5_next_action_sum = 0.0;
  int64_t pwo5_final_vp_games = 0, pwo5_terminal_round_games = 0,
          pwo5_next_action_games = 0;
  int64_t pwo5_slice_uses = 0;
  int64_t pwo5_presentations = 0;

  // WO-PERF-1 Part B: per-module gradient-telemetry mode. 'per_param' is
  // today's behavior (one .item() host read per parameter tensor per
  // minibatch). 'accumulated' keeps the six group sums on-device in float64
  // and reads them back once per update, after the epoch loop. Group order in
  // the accumulator: [0]=policy, [1]=value, [2]=trunk, [3]=final_vp,
  // [4]=terminal_round, [5]=next_own_action.
  const std::string grad_telemetry_mode =
      ::absl::GetFlag(::FLAGS_grad_telemetry_mode);
  const bool grad_telemetry_accumulated =
      (grad_telemetry_mode == "accumulated");
  if (!grad_telemetry_accumulated && grad_telemetry_mode != "per_param") {
    SpielFatalError("Unknown --grad_telemetry_mode: '" + grad_telemetry_mode +
                    "' (expected 'per_param' or 'accumulated').");
  }
  torch::Tensor head_norm_accum;
  if (grad_telemetry_accumulated) {
    head_norm_accum = torch::zeros(
        {6}, torch::TensorOptions().dtype(torch::kFloat64).device(device));
  }

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
      // --- PWO-5 section 8.6: the DEDICATED AUXILIARY FORWARD. ------------
      //
      // The three heads are computed HERE and nowhere else -- never in the PPO
      // forward below, never in the rollout/inference forward, never in an
      // evaluation forward. Reached only when a coefficient is nonzero, so a
      // head-off arm (T, P) never computes a head output at all.
      //
      // Summed into the PPO loss (not a separate optimizer step), so an H arm
      // executes the IDENTICAL number of optimizer steps as its matched T arm
      // and the ablation differs in exactly the three coefficients.
      //
      // All three losses are TRAJECTORY-WEIGHTED: the within-game mean, then
      // the mean over the games present. final_vp and terminal_round targets
      // are CONSTANT within a trajectory (one searched seat per game), so for
      // them trajectory weighting is not merely defensible but correct, and row
      // weighting was the biased choice.
      if (pwo5_active) {
        const int64_t ps = pwo5_slice_start[mb_index];
        const int64_t pl = pwo5_slice_len[mb_index];
        if (pl > 0) {
          torch::Tensor p_obs = pwo5_aux.obs.narrow(0, ps, pl);
          torch::Tensor p_final_vp = pwo5_aux.final_vp.narrow(0, ps, pl);
          torch::Tensor p_round = pwo5_aux.terminal_round.narrow(0, ps, pl);
          torch::Tensor p_next = pwo5_aux.next_action.narrow(0, ps, pl);
          torch::Tensor p_game = pwo5_aux.game_id.narrow(0, ps, pl);

          torch::Tensor scaled_pwo5;
          double fv_val = 0.0, tr_val = 0.0, na_val = 0.0;
          int64_t fv_games = 0, tr_games = 0, na_games = 0;

          // Mean over games of the within-game mean, via the SHARED definition
          // in pwo5loss -- the same one the update-300 whole-dataset evaluator
          // and the numeric tests call.
          auto trajectory_mean = [&](const torch::Tensor& per_row,
                                     const torch::Tensor& mask,
                                     int64_t* out_games) -> torch::Tensor {
            return pwo5loss::TrajectoryMean(per_row, mask, p_game,
                                            pwo5_aux.num_games, out_games);
          };

          auto compute_pwo5 = [&]() {
            auto ao = model->ForwardAux(p_obs);
            torch::Tensor total = torch::zeros({}, p_final_vp.options());

            if (pwo5_cfg.final_vp_coef != 0.0) {
              // Huber on the /20 scale, delta 0.10. The framework default of
              // 1.0 would be pure squared error over the whole reachable range
              // [0.25, 0.75] and would make "Huber" a misnomer.
              //
              // The range is [0.25, 0.75], NOT [0.25, 0.80]: amendment 1
              // ruling 1 registers the target as FinalScoredVp/20, whose
              // measured max over the 400 games is 15 VP. The 16-VP figure
              // that produced 0.80 came from the legacy GetTrueFinalVp
              // reporting helper, which awarded the all-factions-at-3 bonus
              // without the tech-tile-8 guard.
              torch::Tensor pred = ao.final_vp.squeeze(1);
              torch::Tensor per_row = pwo5loss::HuberPerRow(
                  pred, p_final_vp, pwo5_cfg.huber_delta);
              torch::Tensor ones = torch::ones_like(per_row, torch::kBool);
              torch::Tensor l = trajectory_mean(per_row, ones, &fv_games);
              fv_val = l.item<double>();
              total = total + static_cast<float>(pwo5_cfg.final_vp_coef) * l;
            }
            if (pwo5_cfg.terminal_round_coef != 0.0) {
              torch::Tensor per_row =
                  pwo5loss::CrossEntropyPerRow(ao.terminal_round, p_round);
              torch::Tensor ones = torch::ones_like(per_row, torch::kBool);
              torch::Tensor l = trajectory_mean(per_row, ones, &tr_games);
              tr_val = l.item<double>();
              total = total + static_cast<float>(pwo5_cfg.terminal_round_coef) * l;
            }
            if (pwo5_cfg.next_own_action_coef != 0.0) {
              // Full 2,391-way softmax, NO MASK. The current state's legal mask
              // is not valid for a FUTURE action: the target is illegal at the
              // predicting state on 74.57% of rows, so masking would make the
              // correct answer unreachable on three rows in four.
              torch::Tensor mask = p_next >= 0;
              // The -1 rows are masked OUT of the trajectory mean, but gather
              // still needs a valid index for them, so they are pointed at 0
              // and their contribution is discarded by the mask.
              torch::Tensor safe_idx = torch::where(
                  mask, p_next, torch::zeros_like(p_next));
              torch::Tensor per_row = pwo5loss::CrossEntropyPerRow(
                  ao.next_own_action, safe_idx);
              torch::Tensor l = trajectory_mean(per_row, mask, &na_games);
              na_val = l.item<double>();
              total =
                  total + static_cast<float>(pwo5_cfg.next_own_action_coef) * l;
            }
            scaled_pwo5 = total;
          };

          if (device.is_cuda() && ::absl::GetFlag(::FLAGS_train_amp)) {
            AutocastGuard autocast_guard(c10::DeviceType::CUDA, true);
            compute_pwo5();
          } else {
            compute_pwo5();
          }
          scaled_pwo5.backward();

          pwo5_final_vp_sum += fv_val;
          pwo5_terminal_round_sum += tr_val;
          pwo5_next_action_sum += na_val;
          pwo5_final_vp_games += fv_games;
          pwo5_terminal_round_games += tr_games;
          pwo5_next_action_games += na_games;
          pwo5_presentations += pl;
          ++pwo5_slice_uses;
        }
      }

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

      phase_timer.Begin(kPhasePpoForwardLoss);
      if (device.is_cuda() && ::absl::GetFlag(::FLAGS_train_amp)) {
        AutocastGuard autocast_guard(c10::DeviceType::CUDA, true);
        compute_loss();
      } else {
        compute_loss();
      }
      phase_timer.End(kPhasePpoForwardLoss);

      phase_timer.Begin(kPhaseBackward);
      total_loss.backward();
      phase_timer.End(kPhaseBackward);

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
        // PWO-5 section 16 gate 3 item 6 requires per-component gradient norms
        // for "policy, value, trunk, AND EACH AUXILIARY HEAD".
        //
        // The `else` branch below was a real trap, not a gap: before PWO-5 it
        // swept every non-policy, non-value parameter into `trunk_sq`, so the
        // three new heads would have been silently counted as trunk and the
        // trunk_grad_norm column would have been corrupted for exactly the arms
        // (H) whose trunk gradient the ablation is about. The auxiliary tests
        // come FIRST for that reason.
        //
        // Ordering note: `next_own_action_head` does not contain the substring
        // "value_head", but a future head that did would have been captured by
        // the value branch. Testing the auxiliary names first makes the
        // classification independent of the other branches' patterns.
        PhaseScope phase_scope_telemetry(&phase_timer, kPhaseGradTelemetry);
        if (!grad_telemetry_accumulated) {
          double policy_sq = 0.0, value_sq = 0.0, trunk_sq = 0.0;
          double final_vp_sq = 0.0, terminal_round_sq = 0.0, next_action_sq = 0.0;
          for (const auto& named : model->named_parameters()) {
            const torch::Tensor& g = named.value().grad();
            if (!g.defined()) continue;
            const double sq = g.pow(2).sum().item<double>();
            const std::string& nm = named.key();
            if (nm.rfind("final_vp_head", 0) == 0) {
              final_vp_sq += sq;
            } else if (nm.rfind("terminal_round_head", 0) == 0) {
              terminal_round_sq += sq;
            } else if (nm.rfind("next_own_action_head", 0) == 0) {
              next_action_sq += sq;
            } else if (nm.rfind("policy_head", 0) == 0) {
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
          stats.final_vp_head_grad_norm_sum += std::sqrt(final_vp_sq);
          stats.terminal_round_head_grad_norm_sum += std::sqrt(terminal_round_sq);
          stats.next_own_action_head_grad_norm_sum += std::sqrt(next_action_sq);
          stats.head_grad_norm_count += 1;
        } else {
          // WO-PERF-1 Part B: same classification, same accumulation order,
          // same double-precision arithmetic -- but on-device, so a minibatch
          // contributes ZERO device-to-host synchronization points here. Each
          // per-parameter float32 sum is widened to float64 exactly (as
          // .item<double>() did), group sums accumulate left-to-right in
          // parameter order (as `policy_sq += sq` did), and sqrt in float64
          // is correctly rounded on both paths, so the values read back after
          // the epoch loop are bit-identical to per_param on CPU.
          auto f64_opts =
              torch::TensorOptions().dtype(torch::kFloat64).device(device);
          torch::Tensor group_sq = torch::zeros({6}, f64_opts);
          for (const auto& named : model->named_parameters()) {
            const torch::Tensor& g = named.value().grad();
            if (!g.defined()) continue;
            torch::Tensor sq = g.pow(2).sum().to(torch::kFloat64);
            const std::string& nm = named.key();
            int group_idx;
            if (nm.rfind("final_vp_head", 0) == 0) {
              group_idx = 3;
            } else if (nm.rfind("terminal_round_head", 0) == 0) {
              group_idx = 4;
            } else if (nm.rfind("next_own_action_head", 0) == 0) {
              group_idx = 5;
            } else if (nm.rfind("policy_head", 0) == 0) {
              group_idx = 0;
            } else if (nm.find("value_head") != std::string::npos) {
              group_idx = 1;
            } else {
              group_idx = 2;
            }
            group_sq[group_idx] += sq;
          }
          head_norm_accum += group_sq.sqrt();
          stats.head_grad_norm_count += 1;
        }
      }

      phase_timer.Begin(kPhaseGradClip);
      double grad_norm =
          torch::nn::utils::clip_grad_norm_(
              model->parameters(), ::absl::GetFlag(::FLAGS_grad_clip_norm));
      phase_timer.End(kPhaseGradClip);
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
      phase_timer.Begin(kPhaseOptimizerStep);
      optimizer.step();
      phase_timer.End(kPhaseOptimizerStep);

      phase_timer.Begin(kPhaseScalarReads);
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
      phase_timer.End(kPhaseScalarReads);

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

  // WO-PERF-1 Part B: the one host readback per update. Adding the whole
  // accumulated total into the zero-initialized sums is exact (0.0 + x == x),
  // so the resulting sums match per_param's minibatch-by-minibatch host
  // accumulation bit-for-bit on CPU.
  if (grad_telemetry_accumulated && stats.head_grad_norm_count > 0) {
    torch::Tensor head_norms_cpu = head_norm_accum.to(torch::kCPU);
    auto head_norms = head_norms_cpu.accessor<double, 1>();
    stats.policy_head_grad_norm_sum += head_norms[0];
    stats.value_head_grad_norm_sum += head_norms[1];
    stats.trunk_grad_norm_sum += head_norms[2];
    stats.final_vp_head_grad_norm_sum += head_norms[3];
    stats.terminal_round_head_grad_norm_sum += head_norms[4];
    stats.next_own_action_head_grad_norm_sum += head_norms[5];
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

  // --- PWO-5 section 16 gate 3 item 6: per-head loss, per-head denominator,
  // and the REALIZED exposure. Emitted per update so a silent denominator
  // change is visible in diagnostics.csv rather than inferred, and so the
  // exposure schedule is MEASURED rather than assumed.
  stats.pwo5_slice_uses = pwo5_slice_uses;
  stats.pwo5_presentations = pwo5_presentations;
  if (pwo5_slice_uses > 0) {
    stats.pwo5_final_vp_loss = pwo5_final_vp_sum / pwo5_slice_uses;
    stats.pwo5_terminal_round_loss = pwo5_terminal_round_sum / pwo5_slice_uses;
    stats.pwo5_next_action_loss = pwo5_next_action_sum / pwo5_slice_uses;
    // The REGISTERED denominators come from the DRAW, not from summing the
    // per-slice game counts. Slices are re-consumed once per executed epoch, so
    // a slice-sum counts a game up to ppo_update_epochs times and its value is
    // a function of how many epochs --target_kl allowed -- which is exactly the
    // "silent denominator change" section 8.2 emits a denominator to prevent.
    stats.pwo5_sampled_games = pwo5_aux.sampled_games;
    stats.pwo5_final_vp_games = pwo5_aux.final_vp_games;
    stats.pwo5_terminal_round_games = pwo5_aux.terminal_round_games;
    stats.pwo5_next_action_games = pwo5_aux.next_own_action_games;
    // Retained as a separate, differently-named diagnostic so the loss mean's
    // own denominator (slice-uses) is visible and is never mistaken for the
    // trajectory denominator.
    stats.pwo5_slice_game_sum_final_vp = pwo5_final_vp_games;
    stats.pwo5_slice_game_sum_terminal_round = pwo5_terminal_round_games;
    stats.pwo5_slice_game_sum_next_action = pwo5_next_action_games;
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
  // The single end-of-update synchronise and the one read of every event pair.
  phase_timer.Finalize(&stats.phase_timings);
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

// --- PWO-5 section 14.1c: schema v5 = v4 (69 cols) + 17 canary cols 70-86. --
//
// The names, the ORDER and the pairing of every role mean with its own `_n`
// denominator are all frozen by section 14.1c. The six role denominators are
// what makes the partition identity AUDITABLE; revision 7 of the registration
// registered the identity while emitting no counts at all, which made it
// decoration rather than a check.
//
// Role coverage is a COMPLETE PARTITION of NE's domain at the registered
// measurement site, not a convenient subset: `DuneDecisionRole` has SEVEN
// members and kOtherOptional ALONE is 45.82% of the PWO-4 stream, so the
// existing precap_absz convention of primary/continuation/purchase would leave
// the canary blind to nearly half of all decisions. The seventh role,
// kForcedOrBookkeeping, is excluded by proof: the classifier returns it iff the
// searched player is the chance player OR n_legal <= 1; chance nodes `continue`
// in the rollout loop BEFORE any policy evaluation or role classification, so
// the first disjunct is unsatisfiable at this site; and NE's population already
// requires n_legal >= 2. A norm_entropy_forced column would be identically
// empty by construction.
const char* const kDiagnosticsCsvHeaderV5Suffix =
    ",norm_entropy,norm_entropy_n,"
    "norm_entropy_primary,norm_entropy_n_primary,"
    "norm_entropy_continuation,norm_entropy_n_continuation,"
    "norm_entropy_purchase,norm_entropy_n_purchase,"
    "norm_entropy_combat_intrigue,norm_entropy_n_combat_intrigue,"
    "norm_entropy_other_optional,norm_entropy_n_other_optional,"
    "norm_entropy_leader_selection,norm_entropy_n_leader_selection,"
    "max_action_prob,frac_legal_absz_ge_cap,frac_legal_absz_ge_cap_n";

// --- v6 (WO-PERF-1): one column, gated on --diag_prepass_mode=cadenced. ----
//
// Gated, not unconditional, for the same reason v5 is gated on
// --emit_canary_columns: the v4/69 and v5/86 shapes are pinned character by
// character (dune_pwo5_schema_test; PWO-5 registration section 17.5 item 14
// makes any other header a STOP for runs under that registration), so a mode
// that reproduces today's behavior must also reproduce today's header. In
// cadenced mode the column is what makes a skipped KL measurement (-1)
// distinguishable from a measured KL of zero (>= 0; 0 itself means "measured
// over zero nontrivial transitions"). Appended AFTER the v5 canary
// block so the frozen positions 70-86 never move.
const char* const kDiagnosticsCsvHeaderV6Suffix = ",measured_transitions";

std::string DiagnosticsCsvHeader(bool emit_canary_columns) {
  std::string header(kDiagnosticsCsvHeader);
  if (emit_canary_columns) header += kDiagnosticsCsvHeaderV5Suffix;
  if (::absl::GetFlag(::FLAGS_diag_prepass_mode) == "cadenced") {
    header += kDiagnosticsCsvHeaderV6Suffix;
  }
  return header;
}

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

// ---------------------------------------------------------------------------
// The three PWO-5 head loss forms. ONE definition, called by the trainer, the
// update-300 whole-dataset evaluator and the numeric tests.
// ---------------------------------------------------------------------------
namespace pwo5loss {

torch::Tensor TrajectoryMean(const torch::Tensor& per_row,
                             const torch::Tensor& mask,
                             const torch::Tensor& game_id, int64_t num_games,
                             int64_t* out_games) {
  torch::Tensor w = mask.to(per_row.dtype());
  torch::Tensor num = torch::zeros(
      {num_games},
      torch::TensorOptions().dtype(per_row.dtype()).device(per_row.device()));
  torch::Tensor den = torch::zeros_like(num);
  num = num.index_add(0, game_id, per_row * w);
  den = den.index_add(0, game_id, w);
  torch::Tensor present = den > 0;
  const int64_t games = present.sum().item<int64_t>();
  if (out_games != nullptr) *out_games = games;
  if (games == 0) return torch::zeros({}, per_row.options());
  // A game with no contributing row is out of the GAME denominator; the
  // where() only avoids a divide-by-zero on that game's (masked-out) entry.
  torch::Tensor safe_den = torch::where(present, den, torch::ones_like(den));
  torch::Tensor per_game = num / safe_den;
  return (per_game * present.to(per_game.dtype())).sum() /
         static_cast<double>(games);
}

torch::Tensor HuberPerRow(const torch::Tensor& pred, const torch::Tensor& target,
                          double delta) {
  torch::Tensor diff = pred - target;
  torch::Tensor absd = diff.abs();
  return torch::where(absd <= delta, 0.5 * diff * diff,
                      delta * (absd - 0.5 * delta));
}

torch::Tensor CrossEntropyPerRow(const torch::Tensor& logits,
                                 const torch::Tensor& target_index) {
  torch::Tensor logp = torch::log_softmax(logits, -1);
  return -logp.gather(1, target_index.unsqueeze(1)).squeeze(1);
}

}  // namespace pwo5loss

// ---------------------------------------------------------------------------
// PWO-5 head telemetry sidecar. See the header for ruling 6's four registered
// properties.
// ---------------------------------------------------------------------------
namespace {

// The registered field order. A consumer reads names from the header record, so
// this array IS the contract; changing it changes the schema.
constexpr const char* kPwo5TelemetryFields[] = {
    "pilot_local_update",
    "global_update",
    "final_vp_head_loss",
    "terminal_round_head_loss",
    "next_own_action_head_loss",
    "final_vp_head_games",
    "terminal_round_head_games",
    "next_own_action_head_games",
    "aux_sampled_games",
    "aux_slice_uses",
    "aux_slice_game_sum_final_vp",
    "aux_slice_game_sum_terminal_round",
    "aux_slice_game_sum_next_action",
    "aux_distinct_rows",
    "aux_row_presentations",
    "policy_grad_norm",
    "value_grad_norm",
    "trunk_grad_norm",
    "final_vp_head_grad_norm",
    "terminal_round_head_grad_norm",
    "next_own_action_head_grad_norm",
};

// %.17g, built by hand. NOT open_spiel::json -- its writer emits doubles as
// `%f` at six decimals, so any per-head loss below 5e-7 would serialize as
// exactly 0.0 and a head converging to zero would be indistinguishable from a
// head that was never computed.
std::string F17(double v) {
  char buf[64];
  std::snprintf(buf, sizeof(buf), "%.17g", v);
  return std::string(buf);
}

}  // namespace

std::string Pwo5HeadTelemetryPath(const std::string& diagnostics_path) {
  if (diagnostics_path.empty()) return std::string();
  std::filesystem::path p(diagnostics_path);
  std::filesystem::path dir = p.parent_path();
  return (dir / "pwo5_head_telemetry.jsonl").string();
}

std::string Pwo5HeadTelemetryContract() {
  std::string s =
      "{\"record\":\"header\",\"schema\":\"pwo5_head_telemetry.v1\","
      "\"float_format\":\"%.17g\",\"global_update_base\":" +
      std::to_string(kPwo5GlobalUpdateBase) + ",\"fields\":[";
  const int n = static_cast<int>(sizeof(kPwo5TelemetryFields) /
                                 sizeof(kPwo5TelemetryFields[0]));
  for (int i = 0; i < n; ++i) {
    if (i) s += ",";
    s += "\"";
    s += kPwo5TelemetryFields[i];
    s += "\"";
  }
  s += "]";
  return s;
}

void WritePwo5HeadTelemetry(const std::string& diagnostics_path,
                            int pilot_local_update,
                            const PpoUpdateStats& stats,
                            const std::string& run_uuid,
                            const std::string& config_fingerprint,
                            uint64_t base_seed) {
  const std::string path = Pwo5HeadTelemetryPath(diagnostics_path);
  if (path.empty()) return;

  const std::string contract = Pwo5HeadTelemetryContract();
  bool needs_header = true;
  {
    std::ifstream in(path);
    std::string first;
    if (in && std::getline(in, first) && !first.empty()) {
      needs_header = false;
      if (first.compare(0, contract.size(), contract) != 0) {
        SpielFatalError(
            "PWO-5 head telemetry schema mismatch at: " + path +
            "\n  Existing header contract: " + first.substr(0, contract.size()) +
            "\n  This binary writes:       " + contract +
            "\n  Appending would produce records the header cannot describe.");
      }
    }
  }

  std::ofstream out(path, std::ios::app);
  if (!out) {
    SpielFatalError("Could not open PWO-5 head telemetry sidecar: " + path);
  }
  if (needs_header) {
    out << contract << ",\"run_uuid\":\"" << run_uuid
        << "\",\"config_fingerprint\":\"" << config_fingerprint
        << "\",\"base_seed\":" << base_seed << "}\n";
  }

  // Section 11.1: BOTH conventions, as explicit separate fields, so no
  // consumer has to infer the 2450 offset. The u175 / update_25 trap is the
  // standing precedent for what inferring it costs.
  out << "{\"record\":\"update\""
      << ",\"pilot_local_update\":" << pilot_local_update
      << ",\"global_update\":" << (kPwo5GlobalUpdateBase + pilot_local_update)
      << ",\"final_vp_head_loss\":" << F17(stats.pwo5_final_vp_loss)
      << ",\"terminal_round_head_loss\":" << F17(stats.pwo5_terminal_round_loss)
      << ",\"next_own_action_head_loss\":" << F17(stats.pwo5_next_action_loss)
      << ",\"final_vp_head_games\":" << stats.pwo5_final_vp_games
      << ",\"terminal_round_head_games\":" << stats.pwo5_terminal_round_games
      << ",\"next_own_action_head_games\":" << stats.pwo5_next_action_games
      << ",\"aux_sampled_games\":" << stats.pwo5_sampled_games
      << ",\"aux_slice_uses\":" << stats.pwo5_slice_uses
      << ",\"aux_slice_game_sum_final_vp\":"
      << stats.pwo5_slice_game_sum_final_vp
      << ",\"aux_slice_game_sum_terminal_round\":"
      << stats.pwo5_slice_game_sum_terminal_round
      << ",\"aux_slice_game_sum_next_action\":"
      << stats.pwo5_slice_game_sum_next_action
      << ",\"aux_distinct_rows\":" << stats.pwo5_distinct_rows
      << ",\"aux_row_presentations\":" << stats.pwo5_presentations
      << ",\"policy_grad_norm\":" << F17(stats.PolicyHeadGradNormMean())
      << ",\"value_grad_norm\":" << F17(stats.ValueHeadGradNormMean())
      << ",\"trunk_grad_norm\":" << F17(stats.TrunkGradNormMean())
      << ",\"final_vp_head_grad_norm\":" << F17(stats.FinalVpHeadGradNormMean())
      << ",\"terminal_round_head_grad_norm\":"
      << F17(stats.TerminalRoundHeadGradNormMean())
      << ",\"next_own_action_head_grad_norm\":"
      << F17(stats.NextOwnActionHeadGradNormMean())
      << "}\n";
}

// --- WO-PERF-TIMING sidecar ------------------------------------------------

std::string PhaseTimingPath(const std::string& diagnostics_path) {
  if (diagnostics_path.empty()) return std::string();
  std::filesystem::path p(diagnostics_path);
  std::filesystem::path dir = p.parent_path();
  return (dir / "phase_timing.jsonl").string();
}

std::string PhaseTimingContract() {
  std::string s =
      "{\"record\":\"header\",\"schema\":\"phase_timing.v1\","
      "\"float_format\":\"%.17g\",\"phases\":[";
  for (int i = 0; i < kPhaseTimingNumPhases; ++i) {
    if (i) s += ",";
    s += "\"";
    s += kPhaseTimingNames[i];
    s += "\"";
  }
  s += "]";
  return s;
}

// %.17g of a non-finite double emits `inf`/`nan`, which is NOT valid JSON --
// `json.loads('{"x":inf}')` rejects it. Gate T3 asks for finite values; this is
// what makes a violation visible as data rather than as an unparseable file.
std::string F17Finite(double v, bool* saw_nonfinite) {
  if (!std::isfinite(v)) {
    *saw_nonfinite = true;
    return "null";
  }
  return F17(v);
}

// run_prefix and run_uuid are caller-supplied and land inside JSON strings.
std::string JsonEscape(const std::string& in) {
  std::string out;
  out.reserve(in.size() + 8);
  for (char c : in) {
    switch (c) {
      case '"':  out += "\\\""; break;
      case '\\': out += "\\\\"; break;
      case '\n': out += "\\n"; break;
      case '\r': out += "\\r"; break;
      case '\t': out += "\\t"; break;
      default:
        if (static_cast<unsigned char>(c) < 0x20) {
          char buf[8];
          std::snprintf(buf, sizeof(buf), "\\u%04x", c);
          out += buf;
        } else {
          out += c;
        }
    }
  }
  return out;
}

void WritePhaseTiming(const std::string& diagnostics_path, int update,
                      const PpoUpdateStats& stats, double ppo_elapsed_s,
                      const std::string& run_uuid,
                      const std::string& run_prefix) {
  if (!stats.phase_timings.enabled) return;
  const std::string path = PhaseTimingPath(diagnostics_path);
  if (path.empty()) return;

  const std::string contract = PhaseTimingContract();
  bool needs_header = true;
  {
    std::ifstream in(path);
    std::string first;
    if (in && std::getline(in, first) && !first.empty()) {
      needs_header = false;
      if (first.compare(0, contract.size(), contract) != 0) {
        SpielFatalError(
            "Phase-timing schema mismatch at: " + path +
            "\n  Existing header contract: " + first.substr(0, contract.size()) +
            "\n  This binary writes:       " + contract +
            "\n  Appending would produce records the header cannot describe.");
      }
    }
  }

  std::ofstream out(path, std::ios::app);
  if (!out) {
    SpielFatalError("Could not open phase-timing sidecar: " + path);
  }
  if (needs_header) {
    out << contract << ",\"run_uuid\":\"" << JsonEscape(run_uuid)
        << "\",\"run_prefix\":\"" << JsonEscape(run_prefix) << "\"}\n";
  }

  const PpoPhaseTimings& t = stats.phase_timings;
  bool nonfinite = false;

  // run_uuid is repeated on EVERY record, not just the header. The header is
  // written once per FILE, so a resume or a second run pointed at the same
  // --diagnostics_path directory would otherwise append its updates under the
  // first run's header with no per-record identity -- and update numbers repeat
  // across runs. WriteDiagnostics puts run_uuid on every CSV row for exactly
  // this reason. It matters more here: --phase_timing_mode is deliberately
  // absent from ComputeConfigFingerprint(), so a manifest cannot detect a
  // mid-run mode flip and this field is the only provenance a record carries.
  out << "{\"record\":\"update\",\"update\":" << update
      << ",\"run_uuid\":\"" << JsonEscape(run_uuid) << "\""
      << ",\"mode\":\"" << JsonEscape(::absl::GetFlag(::FLAGS_phase_timing_mode))
      << "\",\"timer\":\"" << (t.cuda ? "cuda_event" : "steady_clock") << "\"";

  out << ",\"device_s\":{";
  for (int i = 0; i < kPhaseTimingNumPhases; ++i) {
    if (i) out << ",";
    out << "\"" << kPhaseTimingNames[i] << "\":"
        << F17Finite(t.device_s[i], &nonfinite);
  }
  out << "}";

  out << ",\"host_s\":{";
  for (int i = 0; i < kPhaseTimingNumPhases; ++i) {
    if (i) out << ",";
    out << "\"" << kPhaseTimingNames[i] << "\":"
        << F17Finite(t.host_s[i], &nonfinite);
  }
  out << "}";

  // The metadata that lets a reader see the attribution discrepancy directly
  // rather than infer it (work order §3.3, gate T4).
  //
  // T4 asks for total_attributed_s to be within a declared tolerance of the
  // trainer's PPO figure. DECLARED DEVIATION: on CUDA, total_attributed_s is
  // sum(device_s), which deliberately excludes host-bound time and therefore
  // CANNOT approach ppo_elapsed_s. The quantity that is comparable is
  // total_host_attributed_s, and `unattributed_host_s` is its signed residual
  // against the trainer's own figure. Declared tolerance: the residual is
  // expected to be POSITIVE and to account for the unbracketed work named in
  // the registration (the auxiliary-loss sections, the permutation draws, the
  // per-minibatch index_select gathers and mb_nontrivial sync, the epoch
  // bookkeeping, and this instrument's own sync). A NEGATIVE residual would
  // mean a phase was double-counted and is a defect.
  out << ",\"total_attributed_s\":" << F17Finite(t.total_attributed_s, &nonfinite)
      << ",\"total_host_attributed_s\":"
      << F17Finite(t.total_host_attributed_s, &nonfinite)
      << ",\"sync_s\":" << F17Finite(t.sync_s, &nonfinite)
      << ",\"ppo_elapsed_s\":" << F17Finite(ppo_elapsed_s, &nonfinite)
      << ",\"unattributed_host_s\":"
      << F17Finite(ppo_elapsed_s - t.total_host_attributed_s, &nonfinite)
      << ",\"bracket_count\":" << t.bracket_count
      << ",\"events_recorded\":" << t.events_recorded
      << ",\"minibatches\":" << stats.minibatches
      << ",\"nonfinite\":" << (nonfinite ? "true" : "false") << "}\n";

  if (!out.good()) {
    SpielFatalError("Failed to write phase-timing sidecar to: " + path);
  }
  out.close();
  if (out.fail()) {
    SpielFatalError("Failed to close phase-timing sidecar at: " + path);
  }
}

void WriteDiagnostics(const std::string& filepath, int update, const PpoUpdateStats& stats,
                      double conflict_vp_generated, double conflict_vp_attributed, double conflict_vp_unattributed,
                      uint64_t seed, const std::string& run_uuid, const std::string& run_prefix, const std::string& config_fingerprint,
                      double raw_conflict_vp, double raw_noncombat_vp, double raw_total_vp,
                      double validation_kl, bool emit_canary_columns) {
  if (filepath.empty()) return;
  bool is_csv = (filepath.size() >= 4 && filepath.substr(filepath.size() - 4) == ".csv");

  if (is_csv) {
    std::string existing_header;
    const bool has_header = ReadDiagnosticsCsvHeader(filepath, &existing_header);
    const std::string expected_header = DiagnosticsCsvHeader(emit_canary_columns);
    if (has_header && existing_header != expected_header) {
      SpielFatalError(
          "Diagnostics CSV schema mismatch at: " + filepath +
          "\n  Existing header: " + existing_header +
          "\n  This binary writes: " + expected_header +
          "\n  Appending would produce rows the header cannot describe. Move "
          "the old file aside or point --diagnostics_path at a new file.");
    }
    std::ofstream ofs(filepath, std::ios::app);
    if (!ofs) {
      SpielFatalError("Could not open diagnostics path: " + filepath);
    }
    if (!has_header) {
      ofs << expected_header << "\n";
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
        << stats.minibatches;
    if (emit_canary_columns) {
      // --- v5: PWO-5 section 14.1c, columns 70-86 in registered order. -----
      //
      // Round-trip precision, per section 13.5's rule as applied to CSV: a
      // normalized entropy of 3e-7 must not print as 0.000000. The stream's
      // default precision is 6 SIGNIFICANT digits (not 6 decimals), which
      // switches to scientific notation rather than flushing a small value to
      // zero -- but 6 significant digits still loses information, so the
      // canary columns are written at max_digits10 and restored afterwards.
      const std::streamsize prev_precision = ofs.precision(
          std::numeric_limits<double>::max_digits10);
      // Role order is the registered column order, which is the section 8.2
      // strategic set (primary, continuation, purchase, combat_intrigue,
      // other_optional) followed by leader_selection -- NOT the enum's
      // declaration order.
      const int kRoleColumnOrder[6] = {
          static_cast<int>(DuneDecisionRole::kAgentPrimary),
          static_cast<int>(DuneDecisionRole::kAgentContinuation),
          static_cast<int>(DuneDecisionRole::kPurchase),
          static_cast<int>(DuneDecisionRole::kCombatIntrigue),
          static_cast<int>(DuneDecisionRole::kOtherOptional),
          static_cast<int>(DuneDecisionRole::kLeaderSelection)};
      ofs << "," << stats.norm_entropy << "," << stats.norm_entropy_n;
      for (int i = 0; i < 6; ++i) {
        const int r = kRoleColumnOrder[i];
        ofs << "," << stats.norm_entropy_role[r]
            << "," << stats.norm_entropy_n_role[r];
      }
      ofs << "," << stats.max_action_prob
          << "," << stats.frac_legal_absz_ge_cap
          << "," << stats.frac_legal_absz_ge_cap_n;
      ofs.precision(prev_precision);
    }
    // --- v6 (WO-PERF-1): the KL-measurement denominator, cadenced mode only
    // (matching DiagnosticsCsvHeader's gate so header and rows never diverge).
    if (::absl::GetFlag(::FLAGS_diag_prepass_mode) == "cadenced") {
      ofs << "," << stats.measured_transitions;
    }
    ofs << "\n";
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

    // --- v6 (WO-PERF-1): same name and the same cadenced-mode gate as the
    // CSV column, so the two formats stay analysable interchangeably.
    if (::absl::GetFlag(::FLAGS_diag_prepass_mode) == "cadenced") {
      obj["measured_transitions"] =
          static_cast<int64_t>(stats.measured_transitions);
    }

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
std::vector<double> ComputeTerminalReturns(
    const State& state,
    const std::string& reward_mode,
    double round7_speed_bonus) {
  std::vector<double> terminal_returns = state.Returns();
  if (reward_mode == "first_place") {
    for (int p = 0; p < state.NumPlayers(); ++p) {
      if (terminal_returns[p] == 2.25) {
        terminal_returns[p] = 3.0;
      } else {
        terminal_returns[p] = -1.0;
      }
    }
  } else if (reward_mode != "placement") {
    SpielFatalError("Unknown terminal_reward_mode: " + reward_mode);
  }

  if (round7_speed_bonus != 0.0) {
    const auto* dune_state = dynamic_cast<const dune_imperium::DuneImperiumState*>(&state);
    // At terminal, GetCurrentRound() has already been incremented by 1 at the
    // end of the round (e.g., a game decided on round 7 reports GetCurrentRound() == 8).
    // Therefore, the actual rounds played is GetCurrentRound() - 1.
    const int rounds_played = (dune_state != nullptr) ? (dune_state->GetCurrentRound() - 1) : 0;
    if (dune_state != nullptr && rounds_played <= 7) {
      const double max_ret = *std::max_element(terminal_returns.begin(), terminal_returns.end());
      for (int p = 0; p < state.NumPlayers(); ++p) {
        if (terminal_returns[p] == max_ret) {
          terminal_returns[p] += round7_speed_bonus;
        }
      }
    }
  }
  return terminal_returns;
}

#endif

} // namespace open_spiel
