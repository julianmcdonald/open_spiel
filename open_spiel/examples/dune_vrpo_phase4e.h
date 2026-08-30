#ifndef OPEN_SPIEL_EXAMPLES_DUNE_VRPO_PHASE4E_H_
#define OPEN_SPIEL_EXAMPLES_DUNE_VRPO_PHASE4E_H_

// Phase 4e: terminal production integration for exactly one registered
// four-arm PPO/VRPO update.  Collection remains in dune_ppo_train.cc; this
// file owns strict archive binding, capture/rollout pairing, phase-4d dispatch,
// rollback, update-1 persistence, strict reload, and status-last publication.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <optional>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "dune_ppo_training_utils.h"
#include "dune_vrpo_checkpoint.h"
#include "dune_vrpo_training.h"

namespace open_spiel {

inline constexpr char kVrpoPhase4eResultSchema[] =
    "dune_vrpo_phase4e_one_update_result_v1";
inline constexpr int kVrpoPhase4eGames = 16;

inline std::string GenerateVrpoPhase4eCheckpointUuid() {
  std::random_device device;
  std::mt19937_64 generator(device());
  const uint64_t first = generator();
  const uint64_t second = generator();
  std::ostringstream stream;
  stream << std::hex << std::setfill('0') << std::nouppercase
         << std::setw(8) << static_cast<uint32_t>(first >> 32) << '-'
         << std::setw(4) << static_cast<uint16_t>(first >> 16) << '-'
         << std::setw(4)
         << static_cast<uint16_t>((first & 0x0fffU) | 0x4000U) << '-'
         << std::setw(4)
         << static_cast<uint16_t>(((second >> 48) & 0x3fffU) | 0x8000U)
         << '-' << std::setw(12) << (second & 0xffffffffffffULL);
  return stream.str();
}

inline std::vector<std::string> VrpoPhase4eSourceRelativePaths() {
  // This is the whole production semantic surface for the one-update path:
  // trainer, optimizer/trace modules, and every Dune engine TU/header linked
  // into game rollout. Search-only sources are intentionally excluded because
  // the startup gate structurally disables all search/distillation paths.
  return {
      "open_spiel/examples/dune_ppo_train.cc",
      "open_spiel/examples/dune_ppo_training_utils.cc",
      "open_spiel/examples/dune_ppo_training_utils.h",
      "open_spiel/examples/dune_network.h",
      "open_spiel/examples/dune_seed_utils.h",
      "open_spiel/examples/dune_sha256.h",
      "open_spiel/examples/dune_vrpo.h",
      "open_spiel/examples/dune_vrpo_checkpoint.h",
      "open_spiel/examples/dune_vrpo_training.h",
      "open_spiel/examples/dune_vrpo_phase4e.h",
      "open_spiel/games/dune_imperium/dune_imperium.cc",
      "open_spiel/games/dune_imperium/dune_imperium.h",
      "open_spiel/games/dune_imperium/dune_imperium_board.cc",
      "open_spiel/games/dune_imperium/dune_imperium_board.h",
      "open_spiel/games/dune_imperium/dune_imperium_cards.cc",
      "open_spiel/games/dune_imperium/dune_imperium_cards.h",
      "open_spiel/games/dune_imperium/dune_imperium_common.cc",
      "open_spiel/games/dune_imperium/dune_imperium_common.h",
      "open_spiel/games/dune_imperium/dune_imperium_content.h",
      "open_spiel/games/dune_imperium/dune_imperium_content_generated.h",
      "open_spiel/games/dune_imperium/dune_imperium_state_agent.cc",
      "open_spiel/games/dune_imperium/dune_imperium_state_combat.cc",
      "open_spiel/games/dune_imperium/dune_imperium_state_intrigue.cc",
      "open_spiel/games/dune_imperium/dune_imperium_state_reveal.cc",
      "open_spiel/games/dune_imperium/dune_imperium_state_tech.cc",
      "open_spiel/games/dune_imperium/dune_imperium_state_util.cc",
      "open_spiel/games/dune_imperium/dune_imperium_util.cc",
      "open_spiel/games/dune_imperium/dune_imperium_util.h"};
}

struct VrpoPhase4eSourceIdentity {
  std::filesystem::path canonical_root;
  std::vector<std::string> relative_paths;
  std::string combined_sha256;
};

inline bool LoadVrpoPhase4eSourceIdentity(
    const std::filesystem::path& source_root,
    const std::string& registered_sha256, VrpoPhase4eSourceIdentity* output,
    std::string* error) {
  const auto lower_hex64 = [](const std::string& value) {
    return value.size() == 64 &&
        std::all_of(value.begin(), value.end(), [](char c) {
          return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
        });
  };
  auto fail = [&](const std::string& message) {
    if (error != nullptr) *error = message;
    if (output != nullptr) *output = VrpoPhase4eSourceIdentity{};
    return false;
  };
  if (output == nullptr || !lower_hex64(registered_sha256)) {
    return fail("phase4e source identity output/registration is invalid");
  }
  std::error_code ec;
  const std::filesystem::path canonical_root =
      std::filesystem::canonical(source_root, ec);
  if (ec || !std::filesystem::is_directory(canonical_root)) {
    return fail("phase4e source root is not a readable directory");
  }
  const std::vector<std::string> paths = VrpoPhase4eSourceRelativePaths();
  std::set<std::string> unique_paths;
  std::string payload;
  for (const std::string& relative : paths) {
    const std::filesystem::path rel_path(relative);
    if (rel_path.empty() || rel_path.is_absolute() ||
        relative.find("..") != std::string::npos ||
        !unique_paths.insert(relative).second) {
      return fail("phase4e source path list is noncanonical or duplicated");
    }
    const std::filesystem::path path = canonical_root / rel_path;
    if (!std::filesystem::is_regular_file(path) ||
        std::filesystem::is_symlink(path)) {
      return fail("phase4e required source is missing/non-regular: " +
                  path.string());
    }
    size_t size = 0;
    const std::string digest = ComputeFileSHA256(path.string(), &size);
    if (!lower_hex64(digest) || size == 0) {
      return fail("phase4e required source could not be hashed: " +
                  path.string());
    }
    payload.append(relative);
    payload.push_back('\0');
    payload.append(digest);
    payload.push_back('\n');
  }
  const std::string observed = ComputeStringSHA256(payload);
  if (observed != registered_sha256) {
    return fail("phase4e registered source SHA-256 mismatch");
  }
  output->canonical_root = canonical_root;
  output->relative_paths = paths;
  output->combined_sha256 = observed;
  return true;
}

struct VrpoPhase4eStartupConfig {
  std::string game;
  std::string registration_id;
  std::string selected_arm_id;
  std::filesystem::path input_archive;
  std::filesystem::path output_root;
  std::string source_root;
  std::string source_code_sha256;
  std::string executed_binary_sha256;
  int64_t executed_binary_size = 0;
  std::string input_archive_sha256;
  std::string profile;
  int rollout_games = 0;
  int threads = 0;
  bool diagnostics_only = false;
  std::string init_mode;
  bool rollout_amp = true;
  bool train_amp = true;
  bool allow_tf32 = true;
  bool pipeline = false;
  bool online_search_collection = false;
  bool search_pi_mode = false;
  bool train_value_only = false;
  bool sample_counterfactual_states = false;
  bool has_search_label_dir = false;
  bool ordinary_checkpoint_writes_enabled = false;
  double shaped_reward_weight = 0.0;
  double tleilaxu_breadcrumb_weight = 0.0;
  double tleilaxu_level7_breadcrumb_weight = 0.0;
  double specimen_exchange_penalty = 0.0;
  double reward_scale = 4.0;
  double gamma = 1.0;
  double lambda = 1.0;
  // The actual runtime transform, supplied from FLAGS_logit_cap. It must be
  // exactly the registered selected-arm value before Phase 4e can load,
  // collect, or step.
  double logit_cap = std::numeric_limits<double>::quiet_NaN();
  double probability_tolerance = kVrpoRegisteredProbabilityTolerance;
};

inline const VrpoPhase4ArmConfig* FindCanonicalVrpoPhase4Arm(
    const std::string& arm_id) {
  static const auto arms = CanonicalVrpoPhase4Arms();
  for (const auto& arm : arms) {
    if (arm.arm_id == arm_id) return &arm;
  }
  return nullptr;
}

inline bool ValidateVrpoPhase4eSelectedArmLogitCap(
    const VrpoPhase4eStartupConfig& config, std::string* error) {
  const VrpoPhase4ArmConfig* selected_arm =
      FindCanonicalVrpoPhase4Arm(config.selected_arm_id);
  if (selected_arm == nullptr || !std::isfinite(config.logit_cap) ||
      config.logit_cap != selected_arm->logit_cap) {
    if (error != nullptr) {
      *error = "phase4e runtime logit cap does not exactly match selected arm";
    }
    return false;
  }
  return true;
}

inline bool VrpoPhase4eLowerHex64(const std::string& value) {
  return value.size() == 64 &&
      std::all_of(value.begin(), value.end(), [](char c) {
        return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
      });
}

inline bool VrpoPhase4ePathContainedBy(const std::filesystem::path& candidate,
                                       const std::filesystem::path& parent) {
  auto candidate_it = candidate.begin();
  auto parent_it = parent.begin();
  for (; parent_it != parent.end(); ++parent_it, ++candidate_it) {
    if (candidate_it == candidate.end() || *candidate_it != *parent_it) {
      return false;
    }
  }
  return true;
}

struct VrpoPhase4eResolvedPaths {
  std::filesystem::path input_archive;
  std::filesystem::path output_root;
};

inline bool ResolveVrpoPhase4ePaths(const std::filesystem::path& input,
                                    const std::filesystem::path& output,
                                    VrpoPhase4eResolvedPaths* resolved,
                                    std::string* error) {
  auto fail = [&](const std::string& message) {
    if (error != nullptr) *error = message;
    if (resolved != nullptr) *resolved = VrpoPhase4eResolvedPaths{};
    return false;
  };
  if (resolved == nullptr || input.empty() || output.empty()) {
    return fail("phase4e path resolution input is incomplete");
  }
  std::error_code ec;
  const std::filesystem::path canonical_input =
      std::filesystem::canonical(input, ec);
  if (ec || !std::filesystem::is_directory(canonical_input)) {
    return fail("phase4e input archive cannot be canonicalized as a directory");
  }
  ec.clear();
  const std::filesystem::path canonical_output =
      std::filesystem::weakly_canonical(output, ec);
  if (ec || canonical_output.empty()) {
    return fail("phase4e output root cannot be weakly canonicalized");
  }
  if (VrpoPhase4ePathContainedBy(canonical_output, canonical_input) ||
      VrpoPhase4ePathContainedBy(canonical_input, canonical_output)) {
    return fail("phase4e input/output paths alias or contain one another");
  }
  resolved->input_archive = canonical_input;
  resolved->output_root = canonical_output;
  return true;
}

inline bool ValidateVrpoPhase4eStartupConfig(
    const VrpoPhase4eStartupConfig& config, std::string* error) {
  auto fail = [&](const std::string& message) {
    if (error != nullptr) *error = message;
    return false;
  };
  const VrpoPhase4ArmConfig* selected_arm =
      FindCanonicalVrpoPhase4Arm(config.selected_arm_id);
  if (config.game != "dune_imperium" || config.registration_id.empty() ||
      selected_arm == nullptr || config.input_archive.empty() ||
      config.output_root.empty() || config.source_root.empty() ||
      !VrpoPhase4eLowerHex64(config.source_code_sha256) ||
      !VrpoPhase4eLowerHex64(config.executed_binary_sha256) ||
      config.executed_binary_size <= 0 ||
      !VrpoPhase4eLowerHex64(config.input_archive_sha256)) {
    return fail("phase4e identity/game/path/arm contract is invalid");
  }
  if (!ValidateVrpoPhase4eSelectedArmLogitCap(config, error)) {
    return false;
  }
  VrpoPhase4eResolvedPaths resolved;
  if (!ResolveVrpoPhase4ePaths(config.input_archive, config.output_root,
                               &resolved, error)) {
    return false;
  }
  if ((config.profile != "smoke16" && config.profile != "M16") ||
      config.rollout_games != kVrpoPhase4eGames || config.threads <= 0) {
    return fail("phase4e requires profile in {smoke16,M16} and exactly 16 games");
  }
  if (config.diagnostics_only || config.init_mode != "vrpo_one_update" ||
      config.rollout_amp || config.train_amp || !config.allow_tf32) {
    return fail("phase4e requires training-authorized FP32/TF32 one-update mode");
  }
  if (config.pipeline || config.online_search_collection ||
      config.search_pi_mode || config.train_value_only ||
      config.sample_counterfactual_states || config.has_search_label_dir ||
      config.ordinary_checkpoint_writes_enabled) {
    return fail("phase4e forbids legacy/search/checkpoint side paths");
  }
  for (double shaping :
       {config.shaped_reward_weight, config.tleilaxu_breadcrumb_weight,
        config.tleilaxu_level7_breadcrumb_weight,
        config.specimen_exchange_penalty}) {
    if (!std::isfinite(shaping) || shaping != 0.0) {
      return fail("phase4e shaping coefficients must be exactly zero");
    }
  }
  if (config.reward_scale != 4.0 || config.gamma != 1.0 ||
      config.lambda != 1.0 ||
      config.probability_tolerance != kVrpoRegisteredProbabilityTolerance) {
    return fail("phase4e reward/GAE/probability contract is invalid");
  }
  if (std::filesystem::exists(resolved.output_root)) {
    return fail("phase4e output root must be fresh and absent");
  }
  return true;
}

inline bool ReadVrpoPhase4eExternalBinding(
    const std::filesystem::path& input_archive,
    const VrpoPhase4ArmConfig& expected_arm,
    VrpoPhase4ManifestBinding* binding,
    VrpoExpandedExpectedLayout* layout,
    json::Object* manifest,
    VrpoExpandedArchiveIdentity* archive_identity, std::string* error) {
  auto fail = [&](const std::string& message) {
    if (error != nullptr) *error = message;
    if (binding != nullptr) *binding = VrpoPhase4ManifestBinding{};
    if (layout != nullptr) *layout = VrpoExpandedExpectedLayout{};
    if (manifest != nullptr) manifest->clear();
    if (archive_identity != nullptr) {
      *archive_identity = VrpoExpandedArchiveIdentity{};
    }
    return false;
  };
  if (binding == nullptr || layout == nullptr || manifest == nullptr ||
      archive_identity == nullptr) {
    return fail("phase4e external-binding outputs are incomplete");
  }
  if (!ComputeVrpoExpandedArchiveIdentity(
          input_archive, archive_identity, error) ||
      !ReadVrpoExpandedManifest(input_archive, manifest, error)) {
    return false;
  }
  const auto contract_it = manifest->find("phase4_contract");
  if (contract_it == manifest->end() || !contract_it->second.IsObject()) {
    return fail("phase4e input has no strict phase4 contract");
  }
  const auto& contract = contract_it->second.GetObject();
  auto string_field = [&](const json::Object& object, const char* key,
                          std::string* destination) {
    const auto it = object.find(key);
    if (it == object.end() || !it->second.IsString()) return false;
    *destination = it->second.GetString();
    return true;
  };
  auto int_field = [&](const json::Object& object, const char* key,
                       int64_t* destination) {
    const auto it = object.find(key);
    if (it == object.end() || !it->second.IsInt()) return false;
    *destination = it->second.GetInt();
    return true;
  };
  std::string arm_id;
  int64_t q_seed = 0;
  int64_t base_seed = 0;
  int64_t start_episode = 0;
  int64_t end_episode = 0;
  if (!string_field(contract, "arm_id", &arm_id) ||
      arm_id != expected_arm.arm_id ||
      !string_field(contract, "source_actor_model_sha256",
                    &binding->source_actor_model_sha256) ||
      !string_field(contract, "source_actor_manifest_sha256",
                    &binding->source_actor_manifest_sha256) ||
      !string_field(contract, "source_code_sha256",
                    &binding->source_code_sha256) ||
      !string_field(contract, "actor_subset_sha256",
                    &binding->actor_subset_sha256) ||
      !string_field(contract, "actor_names_shapes_sha256",
                    &binding->actor_names_shapes_sha256) ||
      !string_field(contract, "q_init_sha256", &binding->q_init_sha256) ||
      !string_field(contract, "q_names_shapes_sha256",
                    &binding->q_names_shapes_sha256) ||
      !string_field(contract, "module_layout_sha256",
                    &binding->module_layout_sha256) ||
      !string_field(contract, "optimizer_zero_state_sha256",
                    &binding->optimizer_zero_state_sha256) ||
      !string_field(contract, "optimizer_groups_sha256",
                    &binding->optimizer_groups_sha256) ||
      !string_field(contract, "experiment_uuid", &binding->experiment_uuid) ||
      !int_field(contract, "actor_observation_dim",
                 &binding->actor_observation_dim) ||
      !int_field(contract, "actor_hidden_dim", &binding->actor_hidden_dim) ||
      !int_field(contract, "actor_action_dim", &binding->actor_action_dim) ||
      !int_field(contract, "actor_residual_blocks",
                 &binding->actor_residual_blocks) ||
      !int_field(contract, "q_init_seed", &q_seed) || q_seed <= 0 ||
      !int_field(contract, "base_seed", &base_seed) || base_seed <= 0 ||
      !int_field(contract, "start_episode_id", &start_episode) ||
      start_episode <= 0 ||
      !int_field(contract, "end_episode_id_inclusive", &end_episode) ||
      end_episode < start_episode) {
    return fail("phase4e input contract binding is missing or malformed");
  }
  binding->q_init_seed = static_cast<uint64_t>(q_seed);
  binding->base_seed = static_cast<uint64_t>(base_seed);
  binding->start_episode_id = static_cast<uint64_t>(start_episode);
  binding->end_episode_id_inclusive = static_cast<uint64_t>(end_episode);

  int64_t fixture = 0;
  const auto fixture_it = manifest->find("serialized_layout_test_fixture");
  if (!string_field(*manifest, "serialized_layout_label", &layout->label) ||
      fixture_it == manifest->end() || !fixture_it->second.IsBool() ||
      !int_field(*manifest, "serialized_actor_observation_dim",
                 &layout->actor_observation_dim) ||
      !int_field(*manifest, "serialized_actor_hidden_dim",
                 &layout->actor_hidden_dim) ||
      !int_field(*manifest, "serialized_actor_action_dim",
                 &layout->actor_action_dim) ||
      !int_field(*manifest, "serialized_actor_residual_blocks",
                 &layout->actor_residual_blocks) ||
      !string_field(*manifest, "serialized_actor_names_shapes_sha256",
                    &layout->actor_names_shapes_sha256) ||
      !string_field(*manifest, "serialized_q_names_shapes_sha256",
                    &layout->q_names_shapes_sha256)) {
    (void)fixture;
    return fail("phase4e serialized layout is missing or malformed");
  }
  layout->test_fixture = fixture_it->second.GetBool();
  std::string binding_error;
  if (!ValidateVrpoPhase4ManifestBinding(*binding, &binding_error) ||
      !ValidateVrpoPhase4ManifestStrict(
          contract, expected_arm, *binding, &binding_error)) {
    return fail("phase4e external phase4 binding is not strict: " +
                binding_error);
  }
  return true;
}

struct VrpoPhase4ePairingStats {
  int64_t episodes = 0;
  int64_t capture_rows = 0;
  int64_t rollout_rows = 0;
  int64_t paired_rows = 0;
  std::array<int64_t, kVrpoNumSeats> actor_rows = {0, 0, 0, 0};
  std::string canonical_sha256;
};

inline bool BuildVrpoPhase4eTrainingEpisodes(
    const std::vector<VrpoCapturedEpisode>& captures,
    const std::vector<PpoTransition>& rollout, uint64_t expected_start_id,
    int expected_games, std::vector<VrpoTrainingEpisode>* output,
    VrpoPhase4ePairingStats* stats, std::string* error) {
  auto fail = [&](const std::string& message) {
    if (error != nullptr) *error = message;
    if (output != nullptr) output->clear();
    if (stats != nullptr) *stats = VrpoPhase4ePairingStats{};
    return false;
  };
  if (output == nullptr || stats == nullptr || expected_games != 16 ||
      captures.size() != static_cast<size_t>(expected_games)) {
    return fail("phase4e capture/game count is incomplete");
  }
  output->clear();
  *stats = VrpoPhase4ePairingStats{};
  std::map<uint64_t, std::vector<const PpoTransition*>> rollout_by_episode;
  for (const auto& row : rollout) {
    if (row.episode_id < expected_start_id ||
        row.episode_id >= expected_start_id +
                              static_cast<uint64_t>(expected_games)) {
      return fail("phase4e rollout has an episode outside the registered range");
    }
    rollout_by_episode[row.episode_id].push_back(&row);
  }
  if (rollout_by_episode.size() != captures.size()) {
    return fail("phase4e rollout/capture episode sets differ");
  }
  uint64_t next_row_id = 1;
  std::string payload = "dune_vrpo_phase4e_capture_rollout_pairing_v1";
  for (size_t episode_index = 0; episode_index < captures.size();
       ++episode_index) {
    const auto& capture = captures[episode_index];
    if (capture.episode_id != expected_start_id + episode_index) {
      return fail("phase4e capture IDs are not the exact registered range");
    }
    std::string capture_error;
    if (!ValidateVrpoCapturedEpisode(capture, &capture_error)) {
      return fail("phase4e sealed capture is invalid: " + capture_error);
    }
    const auto found = rollout_by_episode.find(capture.episode_id);
    if (found == rollout_by_episode.end() ||
        found->second.size() != capture.rows.size()) {
      return fail("phase4e rollout/capture row counts differ");
    }
    VrpoTrainingEpisode episode;
    episode.episode_id = capture.episode_id;
    episode.rows.reserve(capture.rows.size());
    std::array<std::vector<size_t>, kVrpoNumSeats> own_rows;
    for (size_t row_index = 0; row_index < capture.rows.size(); ++row_index) {
      const auto& captured = capture.rows[row_index];
      const PpoTransition& paired = *found->second[row_index];
      if (paired.episode_id != captured.episode_id ||
          paired.player_id != captured.actor ||
          paired.action != captured.chosen_action ||
          paired.legal_actions != captured.legal_actions ||
          paired.state.size() != captured.actor_observation.size() ||
          std::memcmp(paired.state.data(), captured.actor_observation.data(),
                      paired.state.size() * sizeof(float)) != 0 ||
          !std::isfinite(paired.value) ||
          !std::isfinite(paired.old_log_prob)) {
        return fail("phase4e rollout/capture row identity mismatch");
      }
      const double chosen_probability =
          captured.legal_behavior_probabilities[captured.chosen_index];
      if (!(chosen_probability > 0.0) ||
          static_cast<float>(std::log(chosen_probability)) !=
              paired.old_log_prob) {
        return fail("phase4e rollout/capture chosen probability mismatch");
      }
      VrpoRolloutPairingView pairing;
      pairing.episode_id = paired.episode_id;
      pairing.actor = paired.player_id;
      pairing.actor_observation = &paired.state;
      pairing.legal_actions = &paired.legal_actions;
      pairing.action = paired.action;
      pairing.chosen_log_probability = paired.old_log_prob;
      std::string pairing_error;
      if (!ValidateVrpoCaptureRolloutPairing(
              captured, pairing, &pairing_error)) {
        return fail("phase4e row pairing failed: " + pairing_error);
      }
      VrpoTrainingRow row;
      row.row_id = next_row_id++;
      row.episode_id = captured.episode_id;
      row.step_index = captured.global_row_index;
      row.actor = captured.actor;
      row.actor_input = torch::from_blob(
          const_cast<float*>(captured.actor_observation.data()),
          {static_cast<int64_t>(captured.actor_observation.size())},
          torch::kFloat32).clone();
      row.q_input = torch::from_blob(
          const_cast<float*>(captured.central_tensor.data()),
          {static_cast<int64_t>(captured.central_tensor.size())},
          torch::kFloat32).clone();
      row.legal_actions = captured.legal_actions;
      row.chosen_index = captured.chosen_index;
      row.chosen_action = captured.chosen_action;
      row.old_legal_probabilities = captured.legal_behavior_probabilities;
      row.old_chosen_log_probability = paired.old_log_prob;
      row.ppo_old_value = paired.value;
      row.rewards = captured.rewards;
      row.terminal_after = captured.terminal_after;
      own_rows[row.actor].push_back(episode.rows.size());
      ++stats->actor_rows[row.actor];
      episode.rows.push_back(std::move(row));
      payload.append(captured.row_sha256);
      vrpo_capture_internal::AppendPod(&payload, paired.value);
    }

    // Registered PPO control target: GAE on each absolute player's own-action
    // subsequence, with the same zero-shaping terminal outcome as the sealed
    // global capture. No source PPO reward/advantage/return is trusted.
    for (int seat = 0; seat < kVrpoNumSeats; ++seat) {
      double next_value = 0.0;
      double next_gae = 0.0;
      bool terminal_assigned = false;
      for (auto it = own_rows[seat].rbegin(); it != own_rows[seat].rend(); ++it) {
        auto& row = episode.rows[*it];
        const double reward = terminal_assigned
            ? 0.0
            : std::clamp(capture.terminal_returns[seat] /
                             capture.reward_scale,
                         -1.0, 1.0);
        terminal_assigned = true;
        const double delta = reward + capture.gamma * next_value -
                             row.ppo_old_value;
        row.ppo_advantage =
            delta + capture.gamma * capture.lambda * next_gae;
        row.ppo_return = row.ppo_advantage + row.ppo_old_value;
        next_value = row.ppo_old_value;
        next_gae = row.ppo_advantage;
      }
      if (!terminal_assigned) {
        return fail("phase4e episode has a seat with no actor row");
      }
    }
    stats->capture_rows += capture.rows.size();
    output->push_back(std::move(episode));
  }
  stats->episodes = output->size();
  stats->rollout_rows = rollout.size();
  stats->paired_rows = stats->capture_rows;
  if (stats->rollout_rows != stats->capture_rows) {
    return fail("phase4e did not pair every rollout row");
  }
  stats->canonical_sha256 = ComputeStringSHA256(payload);
  std::string data_error;
  if (!vrpo_training_internal::ValidateAllData(*output, &data_error)) {
    return fail("phase4e converted training data is invalid: " + data_error);
  }
  return true;
}

struct VrpoPhase4eCanary {
  torch::Tensor actor_input;
  torch::Tensor q_input;
  std::string input_sha256;
};

inline std::string VrpoPhase4eTensorSha256(const torch::Tensor& input,
                                           const std::string& label) {
  const torch::Tensor value =
      input.detach().contiguous().cpu().to(torch::kFloat32);
  std::string payload = "dune_vrpo_phase4e_tensor_v1";
  payload.append(label);
  vrpo_capture_internal::AppendPod(&payload, value.dim());
  for (int64_t size : value.sizes()) {
    vrpo_capture_internal::AppendPod(&payload, size);
  }
  payload.append(reinterpret_cast<const char*>(value.data_ptr<float>()),
                 value.numel() * sizeof(float));
  return ComputeStringSHA256(payload);
}

inline VrpoPhase4eCanary MakeVrpoPhase4eCanary(
    const VrpoExpandedExpectedLayout& layout) {
  auto make = [](int64_t columns, int64_t salt) {
    torch::Tensor tensor = torch::empty({2, columns}, torch::kFloat32);
    float* values = tensor.data_ptr<float>();
    for (int64_t index = 0; index < tensor.numel(); ++index) {
      values[index] = static_cast<float>(
          ((index * 37 + salt) % 257) - 128) / 257.0f;
    }
    return tensor;
  };
  VrpoPhase4eCanary out;
  out.actor_input = make(layout.actor_observation_dim, 11);
  const int64_t q_columns = layout.test_fixture
      ? layout.actor_observation_dim
      : dune_imperium::kVrpoCentralCriticTensorSize;
  out.q_input = make(q_columns, 29);
  out.input_sha256 = ComputeStringSHA256(
      VrpoPhase4eTensorSha256(out.actor_input, "actor_input") +
      VrpoPhase4eTensorSha256(out.q_input, "q_input"));
  return out;
}

enum class VrpoPhase4eFailurePoint {
  kNone,
  kCapture,
  kPreflight,
  kLateUpdate,
  kSave,
  kReload,
};

inline void CleanupVrpoPhase4eOutput(
    const std::filesystem::path& output_root) {
  std::error_code ec;
  std::filesystem::remove_all(output_root, ec);
  const auto parent = output_root.parent_path();
  if (!parent.empty() && std::filesystem::is_directory(parent)) {
    std::string ignored;
    VrpoFsyncDirectory(parent, &ignored);
  }
}

inline bool WriteVrpoPhase4eOneUpdate(
    const VrpoPhase4eStartupConfig& startup,
    const VrpoPhase4ArmConfig& arm,
    const VrpoPhase4ManifestBinding& binding,
    const VrpoExpandedExpectedLayout& layout,
    const VrpoExpandedArchiveIdentity& input_identity,
    const std::vector<VrpoTrainingEpisode>& episodes,
    const VrpoPhase4ePairingStats& pairing,
    std::shared_ptr<SharedDunePolicyValueNetImpl> actor,
    std::shared_ptr<torch::nn::Module> q, VrpoFreshOptimizers* optimizers,
    const VrpoActorForward& actor_forward, const VrpoQForward& q_forward,
    VrpoPhase4eFailurePoint failure_point, json::Object* result,
    std::string* error) {
  VrpoPhase4eResolvedPaths resolved_paths;
  if (!ResolveVrpoPhase4ePaths(startup.input_archive, startup.output_root,
                               &resolved_paths, error)) {
    if (result != nullptr) result->clear();
    return false;
  }
  VrpoPhase4eStartupConfig effective_startup = startup;
  effective_startup.input_archive = resolved_paths.input_archive;
  effective_startup.output_root = resolved_paths.output_root;
  bool owns_output_root = false;
  auto fail = [&](const std::string& message) {
    if (error != nullptr) *error = message;
    if (result != nullptr) result->clear();
    if (owns_output_root) CleanupVrpoPhase4eOutput(effective_startup.output_root);
    return false;
  };
  if (result == nullptr || actor == nullptr || q == nullptr ||
      optimizers == nullptr || optimizers->actor == nullptr ||
      optimizers->q == nullptr || pairing.episodes != kVrpoPhase4eGames ||
      episodes.size() != kVrpoPhase4eGames) {
    return fail("phase4e one-update input is incomplete");
  }
  result->clear();
  std::string startup_error;
  if (!ValidateVrpoPhase4eStartupConfig(effective_startup, &startup_error)) {
    return fail(startup_error);
  }
  // This redundant writer-local check is intentionally before transaction
  // capture and RunVrpoPhase4dOneUpdate: a caller cannot bypass the startup
  // gate and collect/step under a cap other than its registered arm.
  if (!std::isfinite(effective_startup.logit_cap) ||
      effective_startup.logit_cap != arm.logit_cap) {
    return fail("phase4e runtime logit cap differs from selected arm before update");
  }
  const VrpoPhase4eLogitCapBinding cap_binding{
      effective_startup.logit_cap, arm.logit_cap, true};
  if (arm.arm_id != effective_startup.selected_arm_id ||
      input_identity.combined_sha256 != effective_startup.input_archive_sha256 ||
      binding.start_episode_id + kVrpoPhase4eGames - 1 >
          binding.end_episode_id_inclusive) {
    return fail("phase4e selected arm/input/range binding differs");
  }
  if (failure_point == VrpoPhase4eFailurePoint::kCapture) {
    return fail("injected phase4e capture failure");
  }
  const uint64_t next_episode_id =
      binding.start_episode_id + kVrpoPhase4eGames;
  const uint64_t update_seed = vrpo_training_internal::SplitMix64(
      binding.base_seed ^ binding.start_episode_id ^ 0x4eULL);
  if (failure_point == VrpoPhase4eFailurePoint::kPreflight) {
    return fail("injected phase4e preflight failure");
  }

  vrpo_training_internal::ModuleRuntimePlacement actor_placement;
  vrpo_training_internal::ModuleRuntimePlacement q_placement;
  if (!vrpo_training_internal::CapturePhase4dRuntimePlacements(
          *actor, *q, &actor_placement, &q_placement, error)) {
    return fail(error != nullptr ? *error
                                 : "phase4e module placement is invalid");
  }
  vrpo_training_internal::TrainingTransaction transaction;
  if (!transaction.Capture(*actor, *q, *optimizers->actor,
                           *optimizers->q, error)) {
    return fail(error != nullptr ? *error : "phase4e rollback capture failed");
  }
  VrpoTrainingUpdateStats update;
  if (!RunVrpoPhase4dOneUpdate(
          arm, episodes, update_seed, *actor, *q, *optimizers->actor,
          *optimizers->q, actor_forward, q_forward, &update, error)) {
    return fail(error != nullptr ? *error : "phase4e mechanics failed");
  }
  if (failure_point == VrpoPhase4eFailurePoint::kLateUpdate) {
    return fail("injected phase4e late-update failure");
  }

  VrpoPhase4eCanary canary = MakeVrpoPhase4eCanary(layout);
  torch::Tensor actor_canary_input;
  torch::Tensor q_canary_input;
  if (!vrpo_training_internal::MoveTensorToPlacement(
          canary.actor_input, actor_placement, "phase4e actor canary",
          &actor_canary_input, error) ||
      !vrpo_training_internal::MoveTensorToPlacement(
          canary.q_input, q_placement, "phase4e Q canary",
          &q_canary_input, error)) {
    return fail(error != nullptr ? *error
                                 : "phase4e canary placement failed");
  }
  torch::Tensor actor_logits_before;
  torch::Tensor q_values_before;
  auto evaluate_canary = [&](torch::Tensor* actor_logits,
                             torch::Tensor* q_values) {
    try {
      torch::NoGradGuard no_grad;
      const VrpoActorTrainingOutput actor_output =
          actor_forward(actor_canary_input);
      const torch::Tensor q_output = q_forward(q_canary_input);
      if (!vrpo_training_internal::ValidateActorOutput(
              actor_output, actor_canary_input.size(0), actor_placement,
              error) ||
          !vrpo_training_internal::ValidateQOutput(
              q_output, q_canary_input.size(0), q_placement, error)) {
        return false;
      }
      *actor_logits = actor_output.logits.detach().contiguous().cpu();
      *q_values = q_output.detach().contiguous().cpu();
      return vrpo_training_internal::FiniteTensor(*actor_logits) &&
          vrpo_training_internal::FiniteTensor(*q_values);
    } catch (const std::exception& exception) {
      if (error != nullptr) {
        *error = std::string("phase4e canary forward failed: ") +
            exception.what();
      }
      return false;
    }
  };
  if (!evaluate_canary(&actor_logits_before, &q_values_before)) {
    return fail(error != nullptr && !error->empty()
                    ? *error : "phase4e pre-save canary failed");
  }
  const std::string actor_canary_before =
      VrpoPhase4eTensorSha256(actor_logits_before, "actor_logits");
  const std::string q_canary_before =
      VrpoPhase4eTensorSha256(q_values_before, "q_outputs");
  const std::string actor_optimizer_before =
      [&]() {
        std::string a, q_hash, ignored;
        if (!ValidateVrpoOptimizerGroupsAndFiniteState(
                *optimizers, *actor, *q, &a, &q_hash, &ignored)) return std::string();
        return a;
      }();
  const std::string q_optimizer_before =
      [&]() {
        std::string a, q_hash, ignored;
        if (!ValidateVrpoOptimizerGroupsAndFiniteState(
                *optimizers, *actor, *q, &a, &q_hash, &ignored)) return std::string();
        return q_hash;
      }();
  if (actor_optimizer_before.empty() || q_optimizer_before.empty()) {
    return fail("phase4e post-update optimizer state is invalid");
  }

  std::error_code ec;
  if (!std::filesystem::create_directories(effective_startup.output_root, ec) || ec ||
      (!effective_startup.output_root.parent_path().empty() &&
       !VrpoFsyncDirectory(effective_startup.output_root.parent_path(), error))) {
    return fail("phase4e cannot create/fsync fresh output root");
  }
  owns_output_root = true;
  VrpoExpandedArchiveIdentity output_identity;
  const VrpoCheckpointFailurePoint save_failure =
      failure_point == VrpoPhase4eFailurePoint::kSave
          ? VrpoCheckpointFailurePoint::kAfterQOptimizerTemp
          : VrpoCheckpointFailurePoint::kNone;
  const auto archive_path = effective_startup.output_root / "archive";
  if (!SaveVrpoExpandedCheckpointAtomic(
          archive_path, arm, binding, layout,
          GenerateVrpoPhase4eCheckpointUuid(), 1,
          next_episode_id, actor, q, *optimizers, save_failure, error,
          &output_identity, cap_binding)) {
    return fail(error != nullptr ? *error : "phase4e update archive save failed");
  }
  if (failure_point == VrpoPhase4eFailurePoint::kReload) {
    return fail("injected phase4e reload failure");
  }
  json::Object reloaded_manifest;
  VrpoExpandedArchiveIdentity reloaded_identity;
  if (!LoadAndValidateVrpoExpandedCheckpoint(
          archive_path, arm, binding, layout, actor, q, *optimizers,
          &reloaded_manifest, error, &reloaded_identity, 1,
          next_episode_id, cap_binding)) {
    return fail(error != nullptr ? *error : "phase4e strict reload failed");
  }
  torch::Tensor actor_logits_after;
  torch::Tensor q_values_after;
  if (!evaluate_canary(&actor_logits_after, &q_values_after)) {
    return fail(error != nullptr && !error->empty()
                    ? *error : "phase4e post-reload canary failed");
  }
  const std::string actor_canary_after =
      VrpoPhase4eTensorSha256(actor_logits_after, "actor_logits");
  const std::string q_canary_after =
      VrpoPhase4eTensorSha256(q_values_after, "q_outputs");
  std::string actor_optimizer_after;
  std::string q_optimizer_after;
  if (actor_canary_after != actor_canary_before ||
      q_canary_after != q_canary_before ||
      reloaded_identity.combined_sha256 != output_identity.combined_sha256 ||
      !ValidateVrpoOptimizerGroupsAndFiniteState(
          *optimizers, *actor, *q, &actor_optimizer_after,
          &q_optimizer_after, error) ||
      actor_optimizer_after != actor_optimizer_before ||
      q_optimizer_after != q_optimizer_before) {
    return fail("phase4e strict reload/canary/optimizer reproduction failed");
  }
  VrpoExpandedArchiveIdentity input_after;
  if (!ComputeVrpoExpandedArchiveIdentity(
          effective_startup.input_archive, &input_after, error) ||
      input_after.combined_sha256 != input_identity.combined_sha256) {
    return fail("phase4e input archive changed during the update");
  }

  json::Object root;
  root["schema"] = kVrpoPhase4eResultSchema;
  root["status"] = "PASS";
  root["classification"] = "VALID_ONE_UPDATE";
  root["registration_id"] = effective_startup.registration_id;
  root["selected_arm_id"] = arm.arm_id;
  root["algorithm"] = VrpoPhase4AlgorithmName(arm.algorithm);
  root["logit_cap"] = arm.logit_cap;
  root["startup_logit_cap"] = effective_startup.logit_cap;
  root["arm_logit_cap"] = arm.logit_cap;
  root["logit_cap_matched"] = true;
  root["profile"] = effective_startup.profile;
  root["source_code_sha256"] = effective_startup.source_code_sha256;
  root["executed_binary_sha256"] = effective_startup.executed_binary_sha256;
  root["executed_binary_size"] = effective_startup.executed_binary_size;
  root["input_archive_sha256"] = input_identity.combined_sha256;
  root["output_archive_sha256"] = output_identity.combined_sha256;
  root["experiment_uuid"] = binding.experiment_uuid;
  root["source_actor_model_sha256"] = binding.source_actor_model_sha256;
  root["source_actor_manifest_sha256"] =
      binding.source_actor_manifest_sha256;
  root["module_layout_sha256"] = binding.module_layout_sha256;
  root["optimizer_groups_sha256"] = binding.optimizer_groups_sha256;
  root["optimizer_zero_state_sha256"] =
      binding.optimizer_zero_state_sha256;
  root["source_optimizer_moments_loaded"] = false;
  root["rollout_amp"] = false;
  root["train_amp"] = false;
  root["allow_tf32"] = true;
  root["zero_shaping"] = true;
  root["global_update"] = int64_t{1};
  root["start_episode_id"] =
      static_cast<int64_t>(binding.start_episode_id);
  root["next_episode_id"] = static_cast<int64_t>(next_episode_id);
  root["rollout_games"] = int64_t{kVrpoPhase4eGames};
  root["episodes_paired"] = pairing.episodes;
  root["capture_rows"] = pairing.capture_rows;
  root["rollout_rows"] = pairing.rollout_rows;
  root["paired_rows"] = pairing.paired_rows;
  root["pairing_sha256"] = pairing.canonical_sha256;
  json::Array actor_rows;
  for (int64_t count : pairing.actor_rows) actor_rows.emplace_back(count);
  root["actor_rows_by_absolute_seat"] = std::move(actor_rows);
  root["actor_optimizer_steps"] = update.actor_optimizer_steps;
  root["q_optimizer_steps"] = update.q_optimizer_steps;
  root["actor_backward_calls"] = update.actor_backward_calls;
  root["q_backward_calls"] = update.q_backward_calls;
  root["actor_rows_seen"] = update.actor_rows_seen;
  root["q_rows_seen"] = update.q_rows_seen;
  root["complete_episode_partitions"] = update.complete_episode_partitions;
  root["current_rollout_only"] = update.current_rollout_only;
  root["advantages_detached"] = update.advantages_detached;
  root["q_frozen_during_actor"] = update.q_frozen_during_actor;
  root["targets_recomputed_after_actor"] =
      update.targets_recomputed_after_actor;
  root["target_recomputations_after_actor"] =
      update.target_recomputations_after_actor;
  root["pre_actor_target_values_sha256"] =
      update.pre_actor_target_values_sha256;
  root["post_actor_target_values_sha256"] =
      update.post_actor_target_values_sha256;
  root["post_actor_target_bundle_sha256"] =
      update.post_actor_target_bundle_sha256;
  root["actor_values_before_sha256"] =
      update.actor_values_before_sha256;
  root["actor_values_after_sha256"] = update.actor_values_after_sha256;
  root["q_values_before_sha256"] = update.q_values_before_sha256;
  root["q_values_after_sha256"] = update.q_values_after_sha256;
  root["value_head_before_sha256"] = update.value_head_before_sha256;
  root["value_head_after_sha256"] = update.value_head_after_sha256;
  root["actor_loss_mean"] = update.actor_loss_mean;
  root["q_loss_mean"] = update.q_loss_mean;
  root["max_abs_advantage"] = update.max_abs_advantage;
  root["min_ratio"] = update.min_ratio;
  root["max_ratio"] = update.max_ratio;
  root["max_full_legal_kl"] = update.max_full_legal_kl;
  root["max_actor_grad_norm"] = update.max_actor_grad_norm;
  root["max_q_grad_norm"] = update.max_q_grad_norm;
  root["max_value_head_grad_norm"] = update.max_value_head_grad_norm;
  root["update_deterministic_summary_sha256"] =
      update.deterministic_summary_sha256;
  root["actor_optimizer_state_sha256"] = actor_optimizer_after;
  root["q_optimizer_state_sha256"] = q_optimizer_after;
  root["canary_input_sha256"] = canary.input_sha256;
  root["actor_canary_sha256"] = actor_canary_after;
  root["q_canary_sha256"] = q_canary_after;
  root["strict_reload_passed"] = true;
  root["input_archive_untouched"] = true;
  root["rollback_state"] = "not_required_committed";
  root["multi_update_loop_present"] = false;

  const auto status_path = effective_startup.output_root / "UPDATE_RESULT.json";
  const auto status_tmp = effective_startup.output_root / ".UPDATE_RESULT.json.tmp";
  {
    std::ofstream stream(status_tmp, std::ios::trunc);
    if (!stream) return fail("phase4e cannot write result temp");
    stream << json::ToString(root, true) << "\n";
    stream.flush();
    if (!stream) return fail("phase4e result temp flush failed");
  }
  if (!VrpoFsyncFile(status_tmp, error)) {
    return fail(error != nullptr ? *error : "phase4e result fsync failed");
  }
  std::filesystem::rename(status_tmp, status_path);
  if (!VrpoFsyncDirectory(effective_startup.output_root, error) ||
      (!effective_startup.output_root.parent_path().empty() &&
       !VrpoFsyncDirectory(effective_startup.output_root.parent_path(), error))) {
    return fail(error != nullptr ? *error : "phase4e status commit fsync failed");
  }
  transaction.Commit();
  *result = std::move(root);
  return true;
}

}  // namespace open_spiel

#endif  // OPEN_SPIEL_EXAMPLES_DUNE_VRPO_PHASE4E_H_
