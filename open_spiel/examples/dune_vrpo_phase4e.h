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
#include <locale>
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
    "dune_vrpo_phase4e_one_update_result_v2";
inline constexpr char kVrpoPhase4eEpisodeEvidenceSchema[] =
    "dune_vrpo_phase4e_episode_pairing_evidence_v2";
inline constexpr char kVrpoPhase4eOutcomeConvention[] =
    "absolute_seat_0_1_2_3; terminal_return=unscaled_game_return; "
    "registered_reward_scale=4_exact; "
    "zero_shaping_terminal_reward=terminal_return/4; "
    "Dune_terminal_returns_are_unsaturated_in_registered_range; "
    "reward_assigned_once_to_each_absolute_seat_on_final_training_row";
inline constexpr int kVrpoPhase4eGames = 16;

inline std::string VrpoPhase4eCanonicalDouble(double value) {
  if (!std::isfinite(value)) return {};
  std::ostringstream stream;
  stream.imbue(std::locale::classic());
  stream << std::setprecision(std::numeric_limits<double>::max_digits10)
         << std::defaultfloat << value;
  return stream.str();
}

inline bool ParseVrpoPhase4eCanonicalDouble(
    const std::string& text, double* output, std::string* error) {
  if (output == nullptr || text.empty()) {
    if (error != nullptr) *error = "phase4e exact double output/text is empty";
    return false;
  }
  std::istringstream stream(text);
  stream.imbue(std::locale::classic());
  stream >> std::noskipws >> *output;
  if (!stream || !stream.eof() || !std::isfinite(*output) ||
      VrpoPhase4eCanonicalDouble(*output) != text) {
    if (error != nullptr) *error = "phase4e exact double is not canonical";
    return false;
  }
  return true;
}

inline bool CheckedVrpoPhase4eEpisodeRange(
    uint64_t start_episode_id, int64_t rollout_games,
    uint64_t* end_episode_id_inclusive, uint64_t* next_episode_id,
    int64_t* start_episode_id_json, int64_t* next_episode_id_json,
    std::string* error) {
  auto fail = [&](const std::string& message) {
    if (error != nullptr) *error = message;
    if (end_episode_id_inclusive != nullptr) *end_episode_id_inclusive = 0;
    if (next_episode_id != nullptr) *next_episode_id = 0;
    if (start_episode_id_json != nullptr) *start_episode_id_json = 0;
    if (next_episode_id_json != nullptr) *next_episode_id_json = 0;
    return false;
  };
  if (end_episode_id_inclusive == nullptr || next_episode_id == nullptr ||
      start_episode_id_json == nullptr || next_episode_id_json == nullptr ||
      start_episode_id == 0 || rollout_games <= 0) {
    return fail("phase4e episode range arguments are invalid");
  }
  const uint64_t games = static_cast<uint64_t>(rollout_games);
  if (start_episode_id > std::numeric_limits<uint64_t>::max() - games) {
    return fail("phase4e episode range overflows uint64 next ID");
  }
  const uint64_t next = start_episode_id + games;
  if (start_episode_id >
          static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) ||
      next > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
    return fail("phase4e episode range cannot be represented in result int64");
  }
  *end_episode_id_inclusive = next - 1;
  *next_episode_id = next;
  *start_episode_id_json = static_cast<int64_t>(start_episode_id);
  *next_episode_id_json = static_cast<int64_t>(next);
  return true;
}

inline bool CheckedVrpoPhase4eSignedEpisodeRange(
    int64_t start_episode_id, int64_t rollout_games,
    int64_t* next_episode_id, std::string* error) {
  if (next_episode_id == nullptr || start_episode_id <= 0 ||
      rollout_games <= 0 ||
      start_episode_id >
          std::numeric_limits<int64_t>::max() - rollout_games) {
    if (error != nullptr) {
      *error = "phase4e signed episode range overflows or is invalid";
    }
    if (next_episode_id != nullptr) *next_episode_id = 0;
    return false;
  }
  *next_episode_id = start_episode_id + rollout_games;
  return true;
}

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
  uint64_t required_end = 0;
  uint64_t required_next = 0;
  int64_t required_start_json = 0;
  int64_t required_next_json = 0;
  std::string range_error;
  if (!CheckedVrpoPhase4eEpisodeRange(
          binding->start_episode_id, kVrpoPhase4eGames, &required_end,
          &required_next, &required_start_json, &required_next_json,
          &range_error) ||
      required_end > binding->end_episode_id_inclusive) {
    return fail("phase4e external episode range is invalid: " + range_error);
  }

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

struct VrpoPhase4eEpisodeEvidence {
  uint64_t episode_id = 0;
  int64_t capture_rows = 0;
  int64_t rollout_rows = 0;
  int64_t paired_rows = 0;
  std::array<int64_t, kVrpoNumSeats> actor_rows = {0, 0, 0, 0};
  double reward_scale = 4.0;
  VrpoSeatValues terminal_returns_absolute = {0.0, 0.0, 0.0, 0.0};
  VrpoSeatValues zero_shaping_terminal_rewards_absolute =
      {0.0, 0.0, 0.0, 0.0};
  std::string pairing_sha256;
};

struct VrpoPhase4ePairingStats {
  int64_t episodes = 0;
  int64_t capture_rows = 0;
  int64_t rollout_rows = 0;
  int64_t paired_rows = 0;
  std::array<int64_t, kVrpoNumSeats> actor_rows = {0, 0, 0, 0};
  std::vector<VrpoPhase4eEpisodeEvidence> episode_evidence;
  std::string canonical_sha256;
};

inline std::string VrpoPhase4eEpisodePairingSha256(
    const VrpoTrainingEpisode& episode) {
  std::string payload = "dune_vrpo_phase4e_episode_row_pairing_v1";
  vrpo_capture_internal::AppendPod(&payload, episode.episode_id);
  vrpo_capture_internal::AppendPod(
      &payload, static_cast<uint64_t>(episode.rows.size()));
  for (size_t order = 0; order < episode.rows.size(); ++order) {
    const VrpoTrainingRow& row = episode.rows[order];
    vrpo_capture_internal::AppendPod(&payload, static_cast<uint64_t>(order));
    vrpo_capture_internal::AppendPod(&payload, row.row_id);
    vrpo_capture_internal::AppendPod(&payload, row.episode_id);
    vrpo_capture_internal::AppendPod(&payload, row.step_index);
    vrpo_capture_internal::AppendPod(&payload, row.actor);
    const torch::Tensor observation =
        row.actor_input.detach().contiguous().cpu().to(torch::kFloat32);
    vrpo_capture_internal::AppendPod(&payload, observation.numel());
    payload.append(reinterpret_cast<const char*>(observation.data_ptr<float>()),
                   observation.numel() * sizeof(float));
    const torch::Tensor central =
        row.q_input.detach().contiguous().cpu().to(torch::kFloat32);
    vrpo_capture_internal::AppendPod(&payload, central.numel());
    payload.append(reinterpret_cast<const char*>(central.data_ptr<float>()),
                   central.numel() * sizeof(float));
    vrpo_capture_internal::AppendPod(
        &payload, static_cast<uint64_t>(row.legal_actions.size()));
    for (Action action : row.legal_actions) {
      vrpo_capture_internal::AppendPod(&payload, action);
    }
    vrpo_capture_internal::AppendPod(&payload, row.chosen_index);
    vrpo_capture_internal::AppendPod(&payload, row.chosen_action);
    vrpo_capture_internal::AppendPod(
        &payload, static_cast<uint64_t>(row.old_legal_probabilities.size()));
    for (double probability : row.old_legal_probabilities) {
      vrpo_capture_internal::AppendPod(&payload, probability);
    }
    vrpo_capture_internal::AppendPod(&payload,
                                     row.old_chosen_log_probability);
    vrpo_capture_internal::AppendPod(&payload, row.ppo_old_value);
    vrpo_capture_internal::AppendPod(&payload, row.ppo_advantage);
    vrpo_capture_internal::AppendPod(&payload, row.ppo_return);
    for (double reward : row.rewards) {
      vrpo_capture_internal::AppendPod(&payload, reward);
    }
    vrpo_capture_internal::AppendPod(&payload, row.terminal_after);
  }
  return ComputeStringSHA256(payload);
}

inline bool ValidateVrpoPhase4eEpisodeEvidence(
    const std::vector<VrpoPhase4eEpisodeEvidence>& evidence,
    std::string* error) {
  auto fail = [&](const std::string& message) {
    if (error != nullptr) *error = message;
    return false;
  };
  if (evidence.size() != kVrpoPhase4eGames || evidence.front().episode_id == 0) {
    return fail("phase4e episode evidence must contain exactly 16 games");
  }
  const uint64_t first_id = evidence.front().episode_id;
  if (first_id > std::numeric_limits<uint64_t>::max() -
                     (kVrpoPhase4eGames - 1) ||
      first_id > static_cast<uint64_t>(
                     std::numeric_limits<int64_t>::max() -
                     kVrpoPhase4eGames)) {
    return fail("phase4e episode evidence ID range overflows");
  }
  std::set<uint64_t> ids;
  for (size_t index = 0; index < evidence.size(); ++index) {
    const auto& episode = evidence[index];
    if (episode.episode_id != first_id + index ||
        !ids.insert(episode.episode_id).second || episode.capture_rows <= 0 ||
        episode.capture_rows != episode.rollout_rows ||
        episode.capture_rows != episode.paired_rows ||
        !VrpoPhase4eLowerHex64(episode.pairing_sha256)) {
      return fail("phase4e episode evidence ID/count/digest contract is invalid");
    }
    if (episode.reward_scale != 4.0 ||
        VrpoPhase4eCanonicalDouble(episode.reward_scale) != "4") {
      return fail("phase4e episode reward scale is not registered exact 4");
    }
    int64_t actor_sum = 0;
    for (int seat = 0; seat < kVrpoNumSeats; ++seat) {
      if (episode.actor_rows[seat] <= 0 ||
          !std::isfinite(episode.terminal_returns_absolute[seat]) ||
          !std::isfinite(
              episode.zero_shaping_terminal_rewards_absolute[seat]) ||
          episode.zero_shaping_terminal_rewards_absolute[seat] < -1.0 ||
          episode.zero_shaping_terminal_rewards_absolute[seat] > 1.0 ||
          episode.terminal_returns_absolute[seat] !=
              episode.zero_shaping_terminal_rewards_absolute[seat] * 4.0 ||
          VrpoPhase4eCanonicalDouble(
              episode.terminal_returns_absolute[seat]).empty() ||
          VrpoPhase4eCanonicalDouble(
              episode.zero_shaping_terminal_rewards_absolute[seat]).empty()) {
        return fail("phase4e episode actor/outcome evidence is invalid");
      }
      actor_sum += episode.actor_rows[seat];
    }
    if (actor_sum != episode.paired_rows) {
      return fail("phase4e episode actor rows do not sum to paired rows");
    }
  }
  return true;
}

inline bool ComputeVrpoPhase4eEpisodeEvidenceAggregateSha256(
    const std::vector<VrpoPhase4eEpisodeEvidence>& evidence,
    std::string* output, std::string* error) {
  if (output == nullptr) {
    if (error != nullptr) *error = "null phase4e evidence aggregate output";
    return false;
  }
  output->clear();
  if (!ValidateVrpoPhase4eEpisodeEvidence(evidence, error)) return false;
  std::string payload = kVrpoPhase4eEpisodeEvidenceSchema;
  payload.push_back('\0');
  payload.append(kVrpoPhase4eOutcomeConvention);
  vrpo_capture_internal::AppendPod(
      &payload, static_cast<uint64_t>(evidence.size()));
  for (const auto& episode : evidence) {
    vrpo_capture_internal::AppendPod(&payload, episode.episode_id);
    vrpo_capture_internal::AppendPod(&payload, episode.capture_rows);
    vrpo_capture_internal::AppendPod(&payload, episode.rollout_rows);
    vrpo_capture_internal::AppendPod(&payload, episode.paired_rows);
    for (int64_t count : episode.actor_rows) {
      vrpo_capture_internal::AppendPod(&payload, count);
    }
    payload.append(VrpoPhase4eCanonicalDouble(episode.reward_scale));
    payload.push_back('\0');
    for (double value : episode.terminal_returns_absolute) {
      payload.append(VrpoPhase4eCanonicalDouble(value));
      payload.push_back('\0');
    }
    for (double value : episode.zero_shaping_terminal_rewards_absolute) {
      payload.append(VrpoPhase4eCanonicalDouble(value));
      payload.push_back('\0');
    }
    payload.append(episode.pairing_sha256);
    payload.push_back('\0');
  }
  *output = ComputeStringSHA256(payload);
  return true;
}

inline bool ValidateVrpoPhase4ePairingStats(
    const VrpoPhase4ePairingStats& stats, std::string* error) {
  std::string aggregate;
  if (!ComputeVrpoPhase4eEpisodeEvidenceAggregateSha256(
          stats.episode_evidence, &aggregate, error)) {
    return false;
  }
  int64_t capture_rows = 0;
  int64_t rollout_rows = 0;
  int64_t paired_rows = 0;
  std::array<int64_t, kVrpoNumSeats> actor_rows = {0, 0, 0, 0};
  for (const auto& episode : stats.episode_evidence) {
    capture_rows += episode.capture_rows;
    rollout_rows += episode.rollout_rows;
    paired_rows += episode.paired_rows;
    for (int seat = 0; seat < kVrpoNumSeats; ++seat) {
      actor_rows[seat] += episode.actor_rows[seat];
    }
  }
  if (stats.episodes != kVrpoPhase4eGames ||
      stats.capture_rows != capture_rows || stats.rollout_rows != rollout_rows ||
      stats.paired_rows != paired_rows || stats.actor_rows != actor_rows ||
      stats.canonical_sha256 != aggregate) {
    if (error != nullptr) {
      *error = "phase4e aggregate pairing stats differ from episode evidence";
    }
    return false;
  }
  return true;
}

inline bool ValidateVrpoPhase4ePairingStatsAgainstEpisodes(
    const VrpoPhase4ePairingStats& stats,
    const std::vector<VrpoTrainingEpisode>& episodes, std::string* error) {
  if (!ValidateVrpoPhase4ePairingStats(stats, error)) return false;
  if (episodes.size() != stats.episode_evidence.size()) {
    if (error != nullptr) *error = "phase4e evidence/training episode count differs";
    return false;
  }
  for (size_t index = 0; index < episodes.size(); ++index) {
    const auto& episode = episodes[index];
    const auto& evidence = stats.episode_evidence[index];
    std::array<int64_t, kVrpoNumSeats> actor_rows = {0, 0, 0, 0};
    for (const auto& row : episode.rows) {
      if (row.episode_id != episode.episode_id || row.actor < 0 ||
          row.actor >= kVrpoNumSeats) {
        if (error != nullptr) *error = "phase4e training episode identity is invalid";
        return false;
      }
      ++actor_rows[row.actor];
    }
    if (episode.rows.empty() || episode.episode_id != evidence.episode_id ||
        evidence.capture_rows != static_cast<int64_t>(episode.rows.size()) ||
        evidence.rollout_rows != static_cast<int64_t>(episode.rows.size()) ||
        evidence.paired_rows != static_cast<int64_t>(episode.rows.size()) ||
        evidence.actor_rows != actor_rows ||
        evidence.pairing_sha256 !=
            VrpoPhase4eEpisodePairingSha256(episode) ||
        evidence.zero_shaping_terminal_rewards_absolute !=
            episode.rows.back().rewards) {
      if (error != nullptr) {
        *error = "phase4e evidence does not match exhaustive training rows";
      }
      return false;
    }
    for (int seat = 0; seat < kVrpoNumSeats; ++seat) {
      if (evidence.reward_scale != 4.0 ||
          evidence.terminal_returns_absolute[seat] !=
              episode.rows.back().rewards[seat] * 4.0) {
        if (error != nullptr) {
          *error = "phase4e terminal return differs from final row reward x4";
        }
        return false;
      }
    }
  }
  return true;
}

inline json::Object VrpoPhase4eEpisodeEvidenceJson(
    const VrpoPhase4eEpisodeEvidence& evidence) {
  json::Object out;
  out["episode_id"] = static_cast<int64_t>(evidence.episode_id);
  out["capture_rows"] = evidence.capture_rows;
  out["rollout_rows"] = evidence.rollout_rows;
  out["paired_rows"] = evidence.paired_rows;
  json::Array actor_rows;
  json::Array terminal_returns;
  json::Array terminal_rewards;
  json::Array terminal_returns_exact;
  json::Array terminal_rewards_exact;
  for (int seat = 0; seat < kVrpoNumSeats; ++seat) {
    actor_rows.emplace_back(evidence.actor_rows[seat]);
    terminal_returns.emplace_back(evidence.terminal_returns_absolute[seat]);
    terminal_rewards.emplace_back(
        evidence.zero_shaping_terminal_rewards_absolute[seat]);
    terminal_returns_exact.emplace_back(VrpoPhase4eCanonicalDouble(
        evidence.terminal_returns_absolute[seat]));
    terminal_rewards_exact.emplace_back(VrpoPhase4eCanonicalDouble(
        evidence.zero_shaping_terminal_rewards_absolute[seat]));
  }
  out["actor_rows_by_absolute_seat"] = std::move(actor_rows);
  out["reward_scale"] = evidence.reward_scale;
  out["reward_scale_exact"] =
      VrpoPhase4eCanonicalDouble(evidence.reward_scale);
  out["terminal_returns_by_absolute_seat"] = std::move(terminal_returns);
  out["terminal_returns_by_absolute_seat_exact"] =
      std::move(terminal_returns_exact);
  out["zero_shaping_terminal_rewards_by_absolute_seat"] =
      std::move(terminal_rewards);
  out["zero_shaping_terminal_rewards_by_absolute_seat_exact"] =
      std::move(terminal_rewards_exact);
  out["pairing_sha256"] = evidence.pairing_sha256;
  return out;
}

inline bool ParseVrpoPhase4eEpisodeEvidenceJson(
    const json::Value& value, VrpoPhase4eEpisodeEvidence* output,
    std::string* error) {
  auto fail = [&](const std::string& message) {
    if (error != nullptr) *error = message;
    if (output != nullptr) *output = VrpoPhase4eEpisodeEvidence{};
    return false;
  };
  if (output == nullptr || !value.IsObject()) {
    return fail("phase4e episode evidence is not an object");
  }
  const auto& object = value.GetObject();
  if (object.size() != 12) {
    return fail("phase4e episode evidence field set is not strict");
  }
  auto require_int = [&](const char* key, int64_t* destination) {
    const auto it = object.find(key);
    if (it == object.end() || !it->second.IsInt()) return false;
    *destination = it->second.GetInt();
    return true;
  };
  auto require_double = [&](const char* key, double* destination) {
    const auto it = object.find(key);
    if (it == object.end() || !it->second.IsDouble()) return false;
    *destination = it->second.GetDouble();
    return true;
  };
  int64_t episode_id = 0;
  if (!require_int("episode_id", &episode_id) || episode_id <= 0 ||
      !require_int("capture_rows", &output->capture_rows) ||
      !require_int("rollout_rows", &output->rollout_rows) ||
      !require_int("paired_rows", &output->paired_rows) ||
      !require_double("reward_scale", &output->reward_scale)) {
    return fail("phase4e episode scalar evidence is missing/malformed");
  }
  output->episode_id = static_cast<uint64_t>(episode_id);
  const auto reward_scale_exact = object.find("reward_scale_exact");
  double parsed_reward_scale = 0.0;
  if (reward_scale_exact == object.end() ||
      !reward_scale_exact->second.IsString() ||
      !ParseVrpoPhase4eCanonicalDouble(
          reward_scale_exact->second.GetString(), &parsed_reward_scale,
          error) ||
      parsed_reward_scale != output->reward_scale ||
      VrpoPhase4eCanonicalDouble(output->reward_scale) !=
          reward_scale_exact->second.GetString()) {
    return fail("phase4e exact reward scale is missing/malformed");
  }
  auto parse_four = [&](const char* key, auto* destination,
                        bool integers) {
    const auto it = object.find(key);
    if (it == object.end() || !it->second.IsArray() ||
        it->second.GetArray().size() != kVrpoNumSeats) {
      return false;
    }
    for (int seat = 0; seat < kVrpoNumSeats; ++seat) {
      const auto& item = it->second.GetArray()[seat];
      if (integers) {
        if (!item.IsInt()) return false;
        (*destination)[seat] = item.GetInt();
      } else {
        if (!item.IsDouble()) return false;
        (*destination)[seat] = item.GetDouble();
      }
    }
    return true;
  };
  if (!parse_four("actor_rows_by_absolute_seat", &output->actor_rows, true) ||
      !parse_four("terminal_returns_by_absolute_seat",
                  &output->terminal_returns_absolute, false) ||
      !parse_four("zero_shaping_terminal_rewards_by_absolute_seat",
                  &output->zero_shaping_terminal_rewards_absolute, false)) {
    return fail("phase4e episode vector evidence is missing/malformed");
  }
  auto parse_four_exact = [&](const char* key,
                              const VrpoSeatValues& numeric_values) {
    const auto it = object.find(key);
    if (it == object.end() || !it->second.IsArray() ||
        it->second.GetArray().size() != kVrpoNumSeats) {
      return false;
    }
    for (int seat = 0; seat < kVrpoNumSeats; ++seat) {
      const auto& item = it->second.GetArray()[seat];
      double exact_value = 0.0;
      if (!item.IsString() || !ParseVrpoPhase4eCanonicalDouble(
              item.GetString(), &exact_value, error) ||
          exact_value != numeric_values[seat] ||
          VrpoPhase4eCanonicalDouble(numeric_values[seat]) !=
              item.GetString()) {
        return false;
      }
    }
    return true;
  };
  if (!parse_four_exact("terminal_returns_by_absolute_seat_exact",
                        output->terminal_returns_absolute) ||
      !parse_four_exact(
          "zero_shaping_terminal_rewards_by_absolute_seat_exact",
          output->zero_shaping_terminal_rewards_absolute)) {
    return fail("phase4e exact episode outcome vectors are missing/malformed");
  }
  const auto digest = object.find("pairing_sha256");
  if (digest == object.end() || !digest->second.IsString()) {
    return fail("phase4e episode pairing digest is missing/malformed");
  }
  output->pairing_sha256 = digest->second.GetString();
  return true;
}

inline bool RecomputeVrpoPhase4eEpisodeEvidenceAggregateFromResult(
    const json::Object& result, std::string* output, std::string* error,
    std::vector<VrpoPhase4eEpisodeEvidence>* parsed = nullptr) {
  auto fail = [&](const std::string& message) {
    if (error != nullptr) *error = message;
    if (output != nullptr) output->clear();
    if (parsed != nullptr) parsed->clear();
    return false;
  };
  if (output == nullptr) return fail("null phase4e result evidence digest");
  const auto schema = result.find("episode_pairing_evidence_schema");
  const auto convention = result.find("terminal_outcome_convention");
  const auto evidence = result.find("episode_pairing_evidence");
  if (schema == result.end() || !schema->second.IsString() ||
      schema->second.GetString() != kVrpoPhase4eEpisodeEvidenceSchema ||
      convention == result.end() || !convention->second.IsString() ||
      convention->second.GetString() != kVrpoPhase4eOutcomeConvention ||
      evidence == result.end() || !evidence->second.IsArray()) {
    return fail("phase4e result episode evidence schema/convention is invalid");
  }
  std::vector<VrpoPhase4eEpisodeEvidence> values;
  values.reserve(evidence->second.GetArray().size());
  for (const auto& item : evidence->second.GetArray()) {
    VrpoPhase4eEpisodeEvidence episode;
    if (!ParseVrpoPhase4eEpisodeEvidenceJson(item, &episode, error)) {
      return false;
    }
    values.push_back(std::move(episode));
  }
  if (!ComputeVrpoPhase4eEpisodeEvidenceAggregateSha256(
          values, output, error)) {
    return false;
  }
  if (parsed != nullptr) *parsed = std::move(values);
  return true;
}

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
  uint64_t expected_end_id = 0;
  uint64_t expected_next_id = 0;
  int64_t expected_start_json = 0;
  int64_t expected_next_json = 0;
  std::string range_error;
  if (!CheckedVrpoPhase4eEpisodeRange(
          expected_start_id, expected_games, &expected_end_id,
          &expected_next_id, &expected_start_json, &expected_next_json,
          &range_error)) {
    return fail("phase4e requested rollout range is invalid: " + range_error);
  }
  std::map<uint64_t, std::vector<const PpoTransition*>> rollout_by_episode;
  for (const auto& row : rollout) {
    if (row.episode_id < expected_start_id ||
        row.episode_id >= expected_next_id) {
      return fail("phase4e rollout has an episode outside the registered range");
    }
    rollout_by_episode[row.episode_id].push_back(&row);
  }
  if (rollout_by_episode.size() != captures.size()) {
    return fail("phase4e rollout/capture episode sets differ");
  }
  uint64_t next_row_id = 1;
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
    VrpoPhase4eEpisodeEvidence episode_evidence;
    episode_evidence.episode_id = capture.episode_id;
    episode_evidence.capture_rows = capture.rows.size();
    episode_evidence.rollout_rows = found->second.size();
    episode_evidence.paired_rows = capture.rows.size();
    episode_evidence.reward_scale = capture.reward_scale;
    episode_evidence.terminal_returns_absolute = capture.terminal_returns;
    for (int seat = 0; seat < kVrpoNumSeats; ++seat) {
      episode_evidence.zero_shaping_terminal_rewards_absolute[seat] =
          std::clamp(capture.terminal_returns[seat] / capture.reward_scale,
                     -1.0, 1.0);
    }
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
      ++episode_evidence.actor_rows[row.actor];
      episode.rows.push_back(std::move(row));
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
    episode_evidence.pairing_sha256 =
        VrpoPhase4eEpisodePairingSha256(episode);
    stats->episode_evidence.push_back(std::move(episode_evidence));
    stats->capture_rows += capture.rows.size();
    output->push_back(std::move(episode));
  }
  stats->episodes = output->size();
  stats->rollout_rows = rollout.size();
  stats->paired_rows = stats->capture_rows;
  if (stats->rollout_rows != stats->capture_rows) {
    return fail("phase4e did not pair every rollout row");
  }
  if (!ComputeVrpoPhase4eEpisodeEvidenceAggregateSha256(
          stats->episode_evidence, &stats->canonical_sha256, error)) {
    return fail(error != nullptr ? *error
                                 : "phase4e evidence aggregation failed");
  }
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

inline bool RecomputeVrpoPhase4eUpdateDeterministicSummarySha256(
    const json::Object& result, std::string* output, std::string* error) {
  auto fail = [&](const std::string& message) {
    if (error != nullptr) *error = message;
    if (output != nullptr) output->clear();
    return false;
  };
  if (output == nullptr) return fail("null phase4e update-summary output");
  auto require_int = [&](const char* key, int64_t* destination) {
    const auto it = result.find(key);
    if (it == result.end() || !it->second.IsInt()) return false;
    *destination = it->second.GetInt();
    return true;
  };
  auto require_exact_double = [&](const char* key, double* destination) {
    const auto it = result.find(key);
    const auto exact = result.find(std::string(key) + "_exact");
    return it != result.end() && it->second.IsDouble() &&
        std::isfinite(it->second.GetDouble()) && exact != result.end() &&
        exact->second.IsString() && ParseVrpoPhase4eCanonicalDouble(
            exact->second.GetString(), destination, error);
  };
  auto require_string = [&](const char* key, std::string* destination) {
    const auto it = result.find(key);
    if (it == result.end() || !it->second.IsString()) return false;
    *destination = it->second.GetString();
    return true;
  };
  VrpoTrainingUpdateStats stats;
  if (!require_int("actor_optimizer_steps", &stats.actor_optimizer_steps) ||
      !require_int("q_optimizer_steps", &stats.q_optimizer_steps) ||
      !require_int("actor_rows_seen", &stats.actor_rows_seen) ||
      !require_int("q_rows_seen", &stats.q_rows_seen) ||
      !require_exact_double("actor_loss_mean", &stats.actor_loss_mean) ||
      !require_exact_double("q_loss_mean", &stats.q_loss_mean) ||
      !require_exact_double("max_abs_advantage", &stats.max_abs_advantage) ||
      !require_exact_double("min_ratio", &stats.min_ratio) ||
      !require_exact_double("max_ratio", &stats.max_ratio) ||
      !require_exact_double("max_full_legal_kl", &stats.max_full_legal_kl) ||
      !require_exact_double("max_actor_grad_norm",
                            &stats.max_actor_grad_norm) ||
      !require_exact_double("max_q_grad_norm", &stats.max_q_grad_norm) ||
      !require_exact_double("max_value_head_grad_norm",
                            &stats.max_value_head_grad_norm) ||
      !require_string("actor_values_after_sha256",
                      &stats.actor_values_after_sha256) ||
      !require_string("q_values_after_sha256",
                      &stats.q_values_after_sha256) ||
      !require_string("value_head_after_sha256",
                      &stats.value_head_after_sha256) ||
      !require_string("post_actor_target_values_sha256",
                      &stats.post_actor_target_values_sha256) ||
      !require_string("post_actor_target_bundle_sha256",
                      &stats.post_actor_target_bundle_sha256)) {
    return fail("phase4e update-summary scalar/hash field is missing/malformed");
  }
  const auto actor_partitions = result.find("actor_epoch_partition_sha256");
  const auto q_partitions = result.find("q_epoch_partition_sha256");
  const auto q_applicable = result.find("q_epoch_partitions_applicable");
  const auto algorithm = result.find("algorithm");
  if (actor_partitions == result.end() ||
      !actor_partitions->second.IsArray() ||
      actor_partitions->second.GetArray().size() != 4 ||
      q_partitions == result.end() || !q_partitions->second.IsArray() ||
      q_partitions->second.GetArray().size() != 4 ||
      q_applicable == result.end() || !q_applicable->second.IsBool() ||
      algorithm == result.end() || !algorithm->second.IsString()) {
    return fail("phase4e epoch partition evidence is missing/malformed");
  }
  for (int epoch = 0; epoch < 4; ++epoch) {
    const auto& actor_value = actor_partitions->second.GetArray()[epoch];
    const auto& q_value = q_partitions->second.GetArray()[epoch];
    if (!actor_value.IsString() || !q_value.IsString()) {
      return fail("phase4e epoch partition entry has wrong type");
    }
    stats.actor_epoch_partition_sha256[epoch] = actor_value.GetString();
    stats.q_epoch_partition_sha256[epoch] = q_value.GetString();
    if (!VrpoPhase4eLowerHex64(stats.actor_epoch_partition_sha256[epoch])) {
      return fail("phase4e actor epoch partition digest is invalid");
    }
  }
  const bool is_vrpo = algorithm->second.GetString() == "vrpo";
  const bool is_ppo = algorithm->second.GetString() == "ppo";
  if (!is_vrpo && !is_ppo) return fail("phase4e algorithm is invalid");
  if (q_applicable->second.GetBool() != is_vrpo) {
    return fail("phase4e Q partition applicability differs from algorithm");
  }
  const auto q_reason = result.find("q_epoch_partition_not_applicable_reason");
  for (const std::string& digest : stats.q_epoch_partition_sha256) {
    if ((is_vrpo && !VrpoPhase4eLowerHex64(digest)) ||
        (is_ppo && !digest.empty())) {
      return fail("phase4e Q epoch partition digest/N/A rule is invalid");
    }
  }
  if ((is_ppo &&
       (stats.q_optimizer_steps != 0 || stats.q_rows_seen != 0 ||
        q_reason == result.end() || !q_reason->second.IsString() ||
        q_reason->second.GetString() != "PPO_HAS_NO_Q_OPTIMIZER_STEPS")) ||
      (is_vrpo && q_reason != result.end())) {
    return fail("phase4e Q partition N/A metadata is invalid");
  }
  for (double value :
       {stats.actor_loss_mean, stats.q_loss_mean, stats.max_abs_advantage,
        stats.min_ratio, stats.max_ratio, stats.max_full_legal_kl,
        stats.max_actor_grad_norm, stats.max_q_grad_norm,
        stats.max_value_head_grad_norm}) {
    if (!std::isfinite(value)) {
      return fail("phase4e update-summary numeric field is nonfinite");
    }
  }
  *output = vrpo_training_internal::StatsSha256(stats);
  return true;
}

inline bool ValidateVrpoPhase4eResultEvidence(
    const json::Object& result, std::string* error) {
  auto fail = [&](const std::string& message) {
    if (error != nullptr) *error = message;
    return false;
  };
  const auto schema = result.find("schema");
  if (schema == result.end() || !schema->second.IsString() ||
      schema->second.GetString() != kVrpoPhase4eResultSchema) {
    return fail("phase4e result schema is invalid");
  }
  std::string aggregate;
  std::vector<VrpoPhase4eEpisodeEvidence> episodes;
  if (!RecomputeVrpoPhase4eEpisodeEvidenceAggregateFromResult(
          result, &aggregate, error, &episodes)) {
    return false;
  }
  const auto start_id = result.find("start_episode_id");
  const auto next_id = result.find("next_episode_id");
  const auto rollout_games = result.find("rollout_games");
  int64_t checked_next_id = 0;
  if (start_id == result.end() || !start_id->second.IsInt() ||
      next_id == result.end() || !next_id->second.IsInt() ||
      rollout_games == result.end() || !rollout_games->second.IsInt() ||
      rollout_games->second.GetInt() != kVrpoPhase4eGames ||
      !CheckedVrpoPhase4eSignedEpisodeRange(
          start_id->second.GetInt(), rollout_games->second.GetInt(),
          &checked_next_id, error) ||
      next_id->second.GetInt() != checked_next_id ||
      episodes.front().episode_id !=
          static_cast<uint64_t>(start_id->second.GetInt()) ||
      episodes.back().episode_id !=
          static_cast<uint64_t>(checked_next_id - 1)) {
    return fail("phase4e ordered evidence range differs from root range");
  }
  const auto reward_scale = result.find("registered_reward_scale");
  const auto reward_scale_exact =
      result.find("registered_reward_scale_exact");
  double parsed_reward_scale = 0.0;
  if (reward_scale == result.end() || !reward_scale->second.IsDouble() ||
      reward_scale->second.GetDouble() != 4.0 ||
      reward_scale_exact == result.end() ||
      !reward_scale_exact->second.IsString() ||
      reward_scale_exact->second.GetString() != "4" ||
      !ParseVrpoPhase4eCanonicalDouble(
          reward_scale_exact->second.GetString(), &parsed_reward_scale,
          error) ||
      parsed_reward_scale != reward_scale->second.GetDouble()) {
    return fail("phase4e root registered reward scale is not canonical exact 4");
  }
  const auto aggregate_field =
      result.find("episode_pairing_aggregate_sha256");
  const auto pairing_field = result.find("pairing_sha256");
  if (aggregate_field == result.end() ||
      !aggregate_field->second.IsString() ||
      aggregate_field->second.GetString() != aggregate ||
      pairing_field == result.end() || !pairing_field->second.IsString() ||
      pairing_field->second.GetString() != aggregate) {
    return fail("phase4e episode aggregate/pairing digest mismatch");
  }
  int64_t capture_rows = 0;
  int64_t rollout_rows = 0;
  int64_t paired_rows = 0;
  std::array<int64_t, kVrpoNumSeats> actor_rows = {0, 0, 0, 0};
  for (const auto& episode : episodes) {
    capture_rows += episode.capture_rows;
    rollout_rows += episode.rollout_rows;
    paired_rows += episode.paired_rows;
    for (int seat = 0; seat < kVrpoNumSeats; ++seat) {
      actor_rows[seat] += episode.actor_rows[seat];
    }
  }
  auto require_int_equal = [&](const char* key, int64_t expected) {
    const auto it = result.find(key);
    return it != result.end() && it->second.IsInt() &&
        it->second.GetInt() == expected;
  };
  if (!require_int_equal("episodes_paired", kVrpoPhase4eGames) ||
      !require_int_equal("capture_rows", capture_rows) ||
      !require_int_equal("rollout_rows", rollout_rows) ||
      !require_int_equal("paired_rows", paired_rows)) {
    return fail("phase4e aggregate row counts differ from episode evidence");
  }
  const auto actor_field = result.find("actor_rows_by_absolute_seat");
  if (actor_field == result.end() || !actor_field->second.IsArray() ||
      actor_field->second.GetArray().size() != kVrpoNumSeats) {
    return fail("phase4e aggregate actor rows are missing/malformed");
  }
  for (int seat = 0; seat < kVrpoNumSeats; ++seat) {
    const auto& item = actor_field->second.GetArray()[seat];
    if (!item.IsInt() || item.GetInt() != actor_rows[seat]) {
      return fail("phase4e aggregate actor rows differ from episode evidence");
    }
  }
  std::string update_summary;
  if (!RecomputeVrpoPhase4eUpdateDeterministicSummarySha256(
          result, &update_summary, error)) {
    return false;
  }
  const auto summary_field =
      result.find("update_deterministic_summary_sha256");
  if (summary_field == result.end() || !summary_field->second.IsString() ||
      summary_field->second.GetString() != update_summary) {
    return fail("phase4e update deterministic summary mismatch");
  }
  return true;
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
  uint64_t required_end_episode_id = 0;
  uint64_t next_episode_id = 0;
  int64_t start_episode_id_json = 0;
  int64_t next_episode_id_json = 0;
  std::string range_error;
  if (!CheckedVrpoPhase4eEpisodeRange(
          binding.start_episode_id, kVrpoPhase4eGames,
          &required_end_episode_id, &next_episode_id,
          &start_episode_id_json, &next_episode_id_json, &range_error)) {
    return fail("phase4e selected rollout range is invalid: " + range_error);
  }
  if (arm.arm_id != effective_startup.selected_arm_id ||
      input_identity.combined_sha256 != effective_startup.input_archive_sha256 ||
      required_end_episode_id > binding.end_episode_id_inclusive) {
    return fail("phase4e selected arm/input/range binding differs");
  }
  std::string pairing_error;
  if (!ValidateVrpoPhase4ePairingStatsAgainstEpisodes(
          pairing, episodes, &pairing_error)) {
    return fail("phase4e episode evidence rejected before update: " +
                pairing_error);
  }
  if (pairing.episode_evidence.front().episode_id !=
          binding.start_episode_id ||
      pairing.episode_evidence.back().episode_id !=
          required_end_episode_id) {
    return fail("phase4e episode evidence IDs differ from bound rollout range");
  }
  if (failure_point == VrpoPhase4eFailurePoint::kCapture) {
    return fail("injected phase4e capture failure");
  }
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
  root["start_episode_id"] = start_episode_id_json;
  root["next_episode_id"] = next_episode_id_json;
  root["rollout_games"] = int64_t{kVrpoPhase4eGames};
  root["registered_reward_scale"] = effective_startup.reward_scale;
  root["registered_reward_scale_exact"] =
      VrpoPhase4eCanonicalDouble(effective_startup.reward_scale);
  root["episodes_paired"] = pairing.episodes;
  root["capture_rows"] = pairing.capture_rows;
  root["rollout_rows"] = pairing.rollout_rows;
  root["paired_rows"] = pairing.paired_rows;
  root["pairing_sha256"] = pairing.canonical_sha256;
  root["episode_pairing_evidence_schema"] =
      kVrpoPhase4eEpisodeEvidenceSchema;
  root["terminal_outcome_convention"] = kVrpoPhase4eOutcomeConvention;
  root["episode_pairing_aggregate_sha256"] = pairing.canonical_sha256;
  json::Array episode_evidence;
  for (const auto& episode : pairing.episode_evidence) {
    episode_evidence.emplace_back(VrpoPhase4eEpisodeEvidenceJson(episode));
  }
  root["episode_pairing_evidence"] = std::move(episode_evidence);
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
  root["actor_loss_mean_exact"] =
      VrpoPhase4eCanonicalDouble(update.actor_loss_mean);
  root["q_loss_mean"] = update.q_loss_mean;
  root["q_loss_mean_exact"] = VrpoPhase4eCanonicalDouble(update.q_loss_mean);
  root["max_abs_advantage"] = update.max_abs_advantage;
  root["max_abs_advantage_exact"] =
      VrpoPhase4eCanonicalDouble(update.max_abs_advantage);
  root["min_ratio"] = update.min_ratio;
  root["min_ratio_exact"] = VrpoPhase4eCanonicalDouble(update.min_ratio);
  root["max_ratio"] = update.max_ratio;
  root["max_ratio_exact"] = VrpoPhase4eCanonicalDouble(update.max_ratio);
  root["max_full_legal_kl"] = update.max_full_legal_kl;
  root["max_full_legal_kl_exact"] =
      VrpoPhase4eCanonicalDouble(update.max_full_legal_kl);
  root["max_actor_grad_norm"] = update.max_actor_grad_norm;
  root["max_actor_grad_norm_exact"] =
      VrpoPhase4eCanonicalDouble(update.max_actor_grad_norm);
  root["max_q_grad_norm"] = update.max_q_grad_norm;
  root["max_q_grad_norm_exact"] =
      VrpoPhase4eCanonicalDouble(update.max_q_grad_norm);
  root["max_value_head_grad_norm"] = update.max_value_head_grad_norm;
  root["max_value_head_grad_norm_exact"] =
      VrpoPhase4eCanonicalDouble(update.max_value_head_grad_norm);
  json::Array actor_partitions;
  json::Array q_partitions;
  for (int epoch = 0; epoch < 4; ++epoch) {
    actor_partitions.emplace_back(update.actor_epoch_partition_sha256[epoch]);
    q_partitions.emplace_back(update.q_epoch_partition_sha256[epoch]);
  }
  root["actor_epoch_partition_sha256"] = std::move(actor_partitions);
  root["q_epoch_partition_sha256"] = std::move(q_partitions);
  const bool q_partitions_applicable =
      arm.algorithm == VrpoPhase4Algorithm::kVrpo;
  root["q_epoch_partitions_applicable"] = q_partitions_applicable;
  if (!q_partitions_applicable) {
    root["q_epoch_partition_not_applicable_reason"] =
        "PPO_HAS_NO_Q_OPTIMIZER_STEPS";
  }
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

  std::string result_evidence_error;
  if (!ValidateVrpoPhase4eResultEvidence(root, &result_evidence_error)) {
    return fail("phase4e result evidence rejected before PASS: " +
                result_evidence_error);
  }
  root["status"] = "PASS";
  const std::string serialized_result = json::ToString(root, true) + "\n";
  const auto parsed_result = json::FromString(serialized_result);
  if (!parsed_result.has_value() || !parsed_result->IsObject() ||
      !ValidateVrpoPhase4eResultEvidence(
          parsed_result->GetObject(), &result_evidence_error)) {
    return fail("phase4e serialized result evidence rejected before commit: " +
                result_evidence_error);
  }

  const auto status_path = effective_startup.output_root / "UPDATE_RESULT.json";
  const auto status_tmp = effective_startup.output_root / ".UPDATE_RESULT.json.tmp";
  {
    std::ofstream stream(status_tmp, std::ios::trunc);
    if (!stream) return fail("phase4e cannot write result temp");
    stream << serialized_result;
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
  *result = parsed_result->GetObject();
  return true;
}

}  // namespace open_spiel

#endif  // OPEN_SPIEL_EXAMPLES_DUNE_VRPO_PHASE4E_H_
