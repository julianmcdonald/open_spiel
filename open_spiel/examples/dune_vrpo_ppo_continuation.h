#ifndef OPEN_SPIEL_EXAMPLES_DUNE_VRPO_PPO_CONTINUATION_H_
#define OPEN_SPIEL_EXAMPLES_DUNE_VRPO_PPO_CONTINUATION_H_

// Bounded actor-only PPO continuation selected only after the four-update
// pilot and its fresh 256-game noninferiority confirmation. This path is
// deliberately separate from the pilot, phase4e, and ordinary PPO.

#include <chrono>
#include <filesystem>
#include <fstream>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "dune_vrpo_ppo_pilot.h"

#ifdef OPEN_SPIEL_BUILD_WITH_LIBTORCH
#include <torch/torch.h>

namespace open_spiel {

inline constexpr char kVrpoPpoContinuationProfile[] =
    "ppo_cap10_lr5e6_continue4_u5_u8_v1";
inline constexpr char kVrpoPpoContinuationRegistrationId[] =
    "VRPO_U15828_PPO_CAP10_LR5E6_CONTINUE4_U5_U8_20260902";
inline constexpr char kVrpoPpoContinuationResultSchema[] =
    "dune_vrpo_ppo_continuation_result_v1";
inline constexpr char kVrpoPpoContinuationUpdateSchema[] =
    "dune_vrpo_ppo_continuation_update_v1";
inline constexpr char kVrpoPpoContinuationEvidenceSchema[] =
    "dune_vrpo_ppo_continuation_episode_evidence_v1";
inline constexpr char kVrpoPpoContinuationChecksumsSchema[] =
    "dune_vrpo_ppo_continuation_checksums_v1";
inline constexpr char kVrpoPpoContinuationQRole[] =
    "SOURCE_ARCHIVE_PROVENANCE_ONLY";
inline constexpr uint64_t kVrpoPpoContinuationBaseSeed = 8304001;
inline constexpr uint64_t kVrpoPpoContinuationStartEpisodeId = 1200200064;
inline constexpr uint64_t kVrpoPpoContinuationEndEpisodeIdInclusive =
    1200200127;
inline constexpr int kVrpoPpoContinuationFirstGlobalUpdate = 5;
inline constexpr int kVrpoPpoContinuationLastGlobalUpdate = 8;
inline constexpr int kVrpoPpoContinuationNewUpdates = 4;
inline constexpr int kVrpoPpoContinuationGamesPerUpdate = 16;
inline constexpr int kVrpoPpoContinuationActorEpochs = 1;
inline constexpr int64_t kVrpoPpoContinuationStepsPerUpdate = 16;
inline constexpr int64_t kVrpoPpoContinuationPriorActorSteps = 64;
inline constexpr int64_t kVrpoPpoContinuationFinalActorSteps = 128;
inline constexpr double kVrpoPpoContinuationLearningRate = 5.0e-6;
inline constexpr uint64_t kVrpoPpoContinuationQSeed = 2026083001;
inline constexpr char kVrpoPpoContinuationPriorActorValuesSha256[] =
    "8604ee82e76e86591b123c69ebab0a85e6b4aabd3bc088b48f922a35590a988d";
inline constexpr char kVrpoPpoContinuationPriorOptimizerStateSha256[] =
    "29eaec9f1bd5bb2102ad40e0dc230d1617c54b53bbc9021dcb37394a4df747fd";
inline constexpr char kVrpoPpoContinuationSourceQValuesSha256[] =
    "ff77fe5fd6b5b557206405ca6c12aa157baa94095560fbbbadbe2b72ef1b2c26";
inline constexpr char kVrpoPpoContinuationFrozenU1PayloadSha256[] =
    "ac6758343c81d3575159b28c825e8a0f097f99be464f9ede8fb269f0f015abd3";
inline constexpr int64_t kVrpoPpoContinuationRetainedByteCeiling =
    8LL * 1024 * 1024 * 1024;
inline constexpr int64_t kVrpoPpoContinuationPeakByteCeiling =
    10LL * 1024 * 1024 * 1024;

class VrpoPpoContinuationDeadline {
 public:
  static VrpoPpoContinuationDeadline Start(
      std::chrono::steady_clock::time_point start,
      std::chrono::seconds limit = std::chrono::seconds(1800)) {
    VrpoPpoContinuationDeadline out;
    out.active_ = true;
    out.start_ = start;
    out.deadline_ = start + limit;
    return out;
  }

  static VrpoPpoContinuationDeadline ExpireAfterChecksForTest(int checks) {
    VrpoPpoContinuationDeadline out = Start(
        std::chrono::steady_clock::now(), std::chrono::hours(1));
    out.test_checks_remaining_ = checks;
    return out;
  }

  bool Check(const std::string& stage, std::string* error) const {
    if (test_checks_remaining_ >= 0) {
      if (test_checks_remaining_ == 0) {
        if (error != nullptr) {
          *error = "injected PPO-continuation deadline exceeded at " + stage;
        }
        return false;
      }
      --test_checks_remaining_;
    }
    if (!active_ || std::chrono::steady_clock::now() < deadline_) return true;
    if (error != nullptr) {
      *error = "PPO-continuation 1800-second deadline exceeded at " + stage;
    }
    return false;
  }

  double ElapsedSeconds() const {
    if (!active_) return 0.0;
    return std::chrono::duration<double>(
               std::chrono::steady_clock::now() - start_)
        .count();
  }

 private:
  bool active_ = false;
  std::chrono::steady_clock::time_point start_{};
  std::chrono::steady_clock::time_point deadline_{};
  mutable int test_checks_remaining_ = -1;
};

inline std::vector<VrpoPpoPilotEvidenceFileSpec>
VrpoPpoContinuationCanonicalEvidenceFiles() {
  return {
      {"pilot_registration",
       "registrations/ppo_pilot4x16_lr5e6_u15828_20260902/registration.json",
       "f4ae3e987016c8038d16e8cc140d489dbf759e358aa42c6321d63bed8fd1452e"},
      {"pilot_result",
       "ppo_pilot4x16_lr5e6_u15828_20260902/PILOT_RESULT.json",
       "29d56508c5ab09e06245f68053352dca07f4f49bdbbaf605bba58d311402c8a6"},
      {"pilot_global_validation",
       "registrations/ppo_pilot4x16_lr5e6_u15828_20260902/validation.json",
       "c0f6b3880f55a64db6037785a2e7ede20bbaf78a5d1bbb09414fac0f8c1d56fb"},
      {"pilot_update1_validation",
       "registrations/ppo_pilot4x16_lr5e6_u15828_20260902/update_1.validation.json",
       "52c521474f0f47a717b24dbef6e98f0067f2263d88ca8adcd98089ff649d13b4"},
      {"pilot_update2_validation",
       "registrations/ppo_pilot4x16_lr5e6_u15828_20260902/update_2.validation.json",
       "0a8668d2b0d5e27596ae57581a2161fe353e3fee17b004942b356d4ab224fe4f"},
      {"pilot_update3_validation",
       "registrations/ppo_pilot4x16_lr5e6_u15828_20260902/update_3.validation.json",
       "57425ac695fc452a07c22d2473c462e262c03bb85326c9a455cc8f1d3e9f8088"},
      {"pilot_update4_validation",
       "registrations/ppo_pilot4x16_lr5e6_u15828_20260902/update_4.validation.json",
       "a59c0613a231273c72b498f5094b43b16e39e3254a4d2d4cd7fb88f4ff39853e"},
      {"pilot_sidecars",
       "registrations/ppo_pilot4x16_lr5e6_u15828_20260902/sidecars.sha256",
       "7c1c11382d0dbbd3e5ad2c15e27b9b87f6ffffa101bd23f267a9a992382cd162"},
      {"pilot_update4_actor",
       "ppo_pilot4x16_lr5e6_u15828_20260902/update_4/actor_model.pt",
       "8b3d4e5061ed8db611a266880d249b2064268d00f4c280d1e695c265f7ed46fd"},
      {"pilot_update4_optimizer",
       "ppo_pilot4x16_lr5e6_u15828_20260902/update_4/actor_optimizer.pt",
       "0ec288cd2145ed6448c36bbe79edaf02aac2d4d2a89363e9f14281f8096e82b8"},
      {"pilot_update4_checksums",
       "ppo_pilot4x16_lr5e6_u15828_20260902/update_4/CHECKSUMS.json",
       "60058138142f932d9b11397e50cb699507ab32e036b9452c33a944eb9a3ab6ed"},
      {"pilot_update4_result",
       "ppo_pilot4x16_lr5e6_u15828_20260902/update_4/UPDATE_RESULT.json",
       "bbbda80c821547e2a4149f1d2e31a5ab41b4bc4f66a70e7cf8663fb54b57ec3a"},
      {"pilot_update1_corpus",
       "ppo_pilot4x16_lr5e6_u15828_20260902/update_1/ACTOR_CORPUS.bin",
       "2f5e8efad96d738c13dce56e0bb26662d2d790602951bed3ffde494c618ced16"},
      {"raw128_manifest",
       "raw_strength_screen_ppo_pilot_u4_u15828_128_20260902/SCREEN_MANIFEST.json",
       "50a862688379558bd335516b1d2fb6480c9e856a765bac41a4e3e9e12e49aa56"},
      {"raw128_result",
       "raw_strength_screen_ppo_pilot_u4_u15828_128_20260902/SCREEN_RESULT.json",
       "37b7288b8fbf9c79ce7d3e01ca146149688b48ce86420e52ae0fa350f2318054"},
      {"raw128_validation",
       "raw_strength_screen_ppo_pilot_u4_u15828_128_20260902/validation.json",
       "c250feb8c86f42a3a9cae732f8cc91529ebc0f7a52d252cc0163cb4085e9b914"},
      {"raw128_sidecars",
       "raw_strength_screen_ppo_pilot_u4_u15828_128_20260902/sidecars.sha256",
       "57c8fc33ad9cc2b051d533cc46368c2d85e6f5c6ca0b929fc186cef0c7a482be"},
      {"confirm256_manifest",
       "raw_strength_confirm_ppo_pilot_u4_u15828_256_20260902/CONFIRM_MANIFEST.json",
       "abaebe9599d81155815598bdae0effd1e14df4ee5f5e79bc5d82c1a2b1b5c1cf"},
      {"confirm256_result",
       "raw_strength_confirm_ppo_pilot_u4_u15828_256_20260902/CONFIRM_RESULT.json",
       "19935d91ca60978ff483b4e9863130d70c10cea1512dc4b52f94ab24f84e92bf"},
      {"confirm256_validation",
       "raw_strength_confirm_ppo_pilot_u4_u15828_256_20260902/validation.json",
       "99c9ad6bbe968b129e638d632404b8ba6d7072c4d035763d03d5c83f1ebb556d"},
      {"confirm256_sidecars",
       "raw_strength_confirm_ppo_pilot_u4_u15828_256_20260902/sidecars.sha256",
       "b1162b2050292d1ec357ad2a02d7f208d832572eaa44e0fecb6883a4b39af6ce"},
  };
}

struct VrpoPpoContinuationPriorEvidence {
  std::vector<VrpoPpoPilotEvidenceObservation> observations;
  json::Array prior_update_summaries;
  std::vector<VrpoTrainingEpisode> frozen_update1_episodes;
  std::string frozen_update1_payload_sha256;
  std::filesystem::path actor_path;
  std::filesystem::path optimizer_path;
};

namespace vrpo_ppo_continuation_internal {

inline bool ReadJsonObject(const std::filesystem::path& path,
                           json::Object* output, std::string* error) {
  if (output == nullptr || !std::filesystem::is_regular_file(path) ||
      std::filesystem::is_symlink(path)) {
    if (error != nullptr) *error = "continuation JSON file is unavailable";
    return false;
  }
  std::ifstream stream(path, std::ios::binary);
  const std::string text((std::istreambuf_iterator<char>(stream)),
                         std::istreambuf_iterator<char>());
  const auto parsed = json::FromString(text);
  if (!parsed.has_value() || !parsed->IsObject()) {
    if (error != nullptr) *error = "continuation JSON file is malformed";
    return false;
  }
  *output = parsed->GetObject();
  return true;
}

inline bool StringIs(const json::Object& object, const std::string& key,
                     const std::string& expected) {
  const auto it = object.find(key);
  return it != object.end() && it->second.IsString() &&
      it->second.GetString() == expected;
}

inline bool IntIs(const json::Object& object, const std::string& key,
                  int64_t expected) {
  const auto it = object.find(key);
  return it != object.end() && it->second.IsInt() &&
      it->second.GetInt() == expected;
}

inline bool BoolIs(const json::Object& object, const std::string& key,
                   bool expected) {
  const auto it = object.find(key);
  return it != object.end() && it->second.IsBool() &&
      it->second.GetBool() == expected;
}

inline bool ValidatePriorSemantics(
    const std::filesystem::path& root,
    VrpoPpoContinuationPriorEvidence* output, std::string* error) {
  auto fail = [&](const std::string& message) {
    if (error != nullptr) *error = message;
    if (output != nullptr) *output = VrpoPpoContinuationPriorEvidence{};
    return false;
  };
  if (output == nullptr) return fail("null continuation prior output");
  const auto specs = VrpoPpoContinuationCanonicalEvidenceFiles();
  std::vector<std::string> compiled;
  compiled.reserve(specs.size());
  for (const auto& spec : specs) compiled.push_back(spec.expected_sha256);
  VrpoPpoContinuationPriorEvidence prior;
  if (!ValidateVrpoPpoPilotEvidenceFiles(
          root, specs, compiled, &prior.observations, error)) {
    return fail(error != nullptr ? *error
                                 : "continuation prior hashes rejected");
  }

  const auto pilot_root = root / "ppo_pilot4x16_lr5e6_u15828_20260902";
  const auto registration_root = root /
      "registrations/ppo_pilot4x16_lr5e6_u15828_20260902";
  json::Object pilot;
  json::Object global_validation;
  json::Object update4;
  json::Object raw128;
  json::Object confirm;
  json::Object confirm_validation;
  if (!ReadJsonObject(pilot_root / "PILOT_RESULT.json", &pilot, error) ||
      !ReadJsonObject(registration_root / "validation.json",
                      &global_validation, error) ||
      !ReadJsonObject(pilot_root / "update_4/UPDATE_RESULT.json",
                      &update4, error) ||
      !ReadJsonObject(root /
          "raw_strength_screen_ppo_pilot_u4_u15828_128_20260902/SCREEN_RESULT.json",
          &raw128, error) ||
      !ReadJsonObject(root /
          "raw_strength_confirm_ppo_pilot_u4_u15828_256_20260902/CONFIRM_RESULT.json",
          &confirm, error) ||
      !ReadJsonObject(root /
          "raw_strength_confirm_ppo_pilot_u4_u15828_256_20260902/validation.json",
          &confirm_validation, error)) {
    return fail(error != nullptr ? *error
                                 : "continuation prior JSON read failed");
  }
  const auto pilot_updates = pilot.find("updates");
  if (!StringIs(pilot, "classification",
                "VALID_FOUR_UPDATE_PPO_PILOT_HEALTH_PASS") ||
      !StringIs(pilot, "status", "VALID") ||
      !IntIs(pilot, "completed_updates", 4) ||
      !IntIs(pilot, "actor_optimizer_steps_total", 64) ||
      !BoolIs(pilot, "raw128_authorized", true) ||
      !StringIs(pilot, "latest_actor_optimizer_state_sha256",
                kVrpoPpoContinuationPriorOptimizerStateSha256) ||
      !StringIs(pilot, "frozen_update1_corpus_sha256",
                kVrpoPpoContinuationFrozenU1PayloadSha256) ||
      pilot_updates == pilot.end() || !pilot_updates->second.IsArray() ||
      pilot_updates->second.GetArray().size() != 4 ||
      !StringIs(global_validation, "status", "VALID") ||
      !StringIs(global_validation, "classification",
                "VALID_FOUR_UPDATE_PPO_PILOT_HEALTH_PASS") ||
      !StringIs(update4, "classification", "VALID_UPDATE_HEALTH_PASS") ||
      !IntIs(update4, "update", 4) ||
      !IntIs(update4, "actor_optimizer_step_after", 64) ||
      !StringIs(update4, "actor_after_sha256",
                kVrpoPpoContinuationPriorActorValuesSha256) ||
      !StringIs(update4, "actor_optimizer_after_sha256",
                kVrpoPpoContinuationPriorOptimizerStateSha256) ||
      !StringIs(update4, "q_before_sha256",
                kVrpoPpoContinuationSourceQValuesSha256) ||
      !StringIs(update4, "q_after_sha256",
                kVrpoPpoContinuationSourceQValuesSha256) ||
      !StringIs(update4, "source_actor_model_sha256",
                "68febee771509f88446286cc50983afd22d64f7772f6866def77bafd4aae36d2") ||
      !StringIs(update4, "source_actor_manifest_sha256",
                "95df08e8c9ccdba0a402b610474022fa22374215842f7b0d6243ce09359b4512") ||
      !StringIs(update4, "source_experiment_uuid",
                "7b7b58a7-297c-4d98-8bb2-5f8f1b0a4c30") ||
      !BoolIs(update4, "strict_reload_passed", true) ||
      !BoolIs(update4, "canary_exact", true) ||
      !BoolIs(update4, "q_bit_inert", true) ||
      !StringIs(raw128, "status", "VALID") ||
      !StringIs(raw128, "classification",
                "VALID_RAW_POLICY_GROSS_DAMAGE_SCREEN_SURVIVOR") ||
      !StringIs(confirm, "status", "PASS") ||
      !StringIs(confirm, "classification",
                "CONFIRMED_NONINFERIOR_FOR_PPO_CONTINUATION_DESIGN") ||
      !StringIs(confirm_validation, "status", "PASS") ||
      !StringIs(confirm_validation, "classification",
                "CONFIRMED_NONINFERIOR_FOR_PPO_CONTINUATION_DESIGN")) {
    return fail("continuation prior pilot/confirmation semantics rejected");
  }
  const auto confirm_decision = confirm.find("decision");
  if (confirm_decision == confirm.end() ||
      !confirm_decision->second.IsObject() ||
      !BoolIs(confirm_decision->second.GetObject(),
              "next_bounded_PPO_continuation_design_authorized", true) ||
      !BoolIs(confirm_decision->second.GetObject(),
              "PPO_continuation_launch_authorized", false)) {
    return fail("continuation prior confirmation authorization is malformed");
  }
  for (int update = 1; update <= 4; ++update) {
    json::Object validation;
    if (!ReadJsonObject(registration_root /
            ("update_" + std::to_string(update) + ".validation.json"),
            &validation, error) ||
        !StringIs(validation, "status", "VALID") ||
        !IntIs(validation, "update", update) ||
        !IntIs(validation, "actor_optimizer_step_after", update * 16)) {
      return fail("continuation prior per-update validation rejected");
    }
  }
  const auto corpus_path = pilot_root / "update_1/ACTOR_CORPUS.bin";
  std::ifstream corpus_stream(corpus_path, std::ios::binary);
  const std::string corpus_bytes(
      (std::istreambuf_iterator<char>(corpus_stream)),
      std::istreambuf_iterator<char>());
  VrpoScheduleCorpusIdentity corpus_identity;
  if (!DecodeVrpoScheduleActorCorpus(
          corpus_bytes, &prior.frozen_update1_episodes, &corpus_identity,
          error) ||
      corpus_identity.payload_sha256 !=
          kVrpoPpoContinuationFrozenU1PayloadSha256) {
    return fail(error != nullptr && !error->empty()
                    ? *error
                    : "continuation frozen update-1 anchor rejected");
  }
  prior.frozen_update1_payload_sha256 = corpus_identity.payload_sha256;
  prior.prior_update_summaries = pilot_updates->second.GetArray();
  prior.actor_path = pilot_root / "update_4/actor_model.pt";
  prior.optimizer_path = pilot_root / "update_4/actor_optimizer.pt";
  *output = std::move(prior);
  return true;
}

inline bool EvidenceStillExact(
    const std::filesystem::path& root,
    const std::vector<VrpoPpoPilotEvidenceObservation>& expected,
    std::string* error) {
  const auto specs = VrpoPpoContinuationCanonicalEvidenceFiles();
  std::vector<std::string> compiled;
  for (const auto& spec : specs) compiled.push_back(spec.expected_sha256);
  std::vector<VrpoPpoPilotEvidenceObservation> observed;
  if (!ValidateVrpoPpoPilotEvidenceFiles(
          root, specs, compiled, &observed, error) ||
      observed.size() != expected.size()) {
    return false;
  }
  for (size_t i = 0; i < observed.size(); ++i) {
    if (!observed[i].matched || observed[i].identity != expected[i].identity ||
        observed[i].observed_sha256 != expected[i].observed_sha256 ||
        observed[i].observed_size != expected[i].observed_size) {
      if (error != nullptr) *error = "continuation prior evidence drifted";
      return false;
    }
  }
  return true;
}

inline bool ObserveBytes(const std::filesystem::path& root,
                         int64_t* measured_peak_bytes, std::string* error) {
  const int64_t observed =
      vrpo_ppo_pilot_internal::DirectoryRegularBytes(root);
  if (observed < 0 || measured_peak_bytes == nullptr) {
    if (error != nullptr) *error = "PPO-continuation byte accounting failed";
    return false;
  }
  *measured_peak_bytes = std::max(*measured_peak_bytes, observed);
  if (observed > kVrpoPpoContinuationPeakByteCeiling) {
    if (error != nullptr) *error = "PPO-continuation exceeded 10-GiB peak";
    return false;
  }
  return true;
}

}  // namespace vrpo_ppo_continuation_internal

struct VrpoPpoContinuationStartupConfig : VrpoPpoPilotStartupConfig {
  VrpoPpoContinuationPriorEvidence prior;
};

inline std::vector<std::string> VrpoPpoContinuationSourceRelativePaths() {
  std::vector<std::string> paths = VrpoPpoPilotSourceRelativePaths();
  paths.push_back("open_spiel/examples/dune_vrpo_ppo_continuation.h");
  return paths;
}

inline bool LoadVrpoPpoContinuationSourceIdentity(
    const std::filesystem::path& source_root,
    const std::string& registered_sha256, VrpoPhase4eSourceIdentity* output,
    std::string* error) {
  auto fail = [&](const std::string& message) {
    if (error != nullptr) *error = message;
    if (output != nullptr) *output = VrpoPhase4eSourceIdentity{};
    return false;
  };
  if (output == nullptr || !VrpoPhase4eLowerHex64(registered_sha256)) {
    return fail("PPO-continuation source identity is invalid");
  }
  std::error_code ec;
  const auto canonical_root = std::filesystem::canonical(source_root, ec);
  if (ec || !std::filesystem::is_directory(canonical_root)) {
    return fail("PPO-continuation source root is unreadable");
  }
  const auto paths = VrpoPpoContinuationSourceRelativePaths();
  std::set<std::string> seen;
  std::string payload;
  for (const auto& relative : paths) {
    if (relative.empty() || std::filesystem::path(relative).is_absolute() ||
        relative.find("..") != std::string::npos ||
        !seen.insert(relative).second) {
      return fail("PPO-continuation source list is noncanonical");
    }
    const auto path = canonical_root / relative;
    if (!std::filesystem::is_regular_file(path) ||
        std::filesystem::is_symlink(path)) {
      return fail("PPO-continuation source file missing: " + relative);
    }
    size_t size = 0;
    const std::string digest = ComputeFileSHA256(path.string(), &size);
    if (!VrpoPhase4eLowerHex64(digest) || size == 0) {
      return fail("PPO-continuation source file hash failed: " + relative);
    }
    payload.append(relative);
    payload.push_back('\0');
    payload.append(digest);
    payload.push_back('\n');
  }
  const std::string observed = ComputeStringSHA256(payload);
  if (observed != registered_sha256) {
    return fail("PPO-continuation registered source SHA-256 mismatch");
  }
  output->canonical_root = canonical_root;
  output->relative_paths = paths;
  output->combined_sha256 = observed;
  return true;
}

inline bool ValidateVrpoPpoContinuationStartupConfig(
    const VrpoPpoContinuationStartupConfig& config, std::string* error,
    bool test_fixture = false) {
  auto fail = [&](const std::string& message) {
    if (error != nullptr) *error = message;
    return false;
  };
  if (config.game != "dune_imperium" ||
      config.init_mode != "vrpo_ppo_continuation" ||
      config.profile != kVrpoPpoContinuationProfile ||
      (!test_fixture &&
       config.registration_id != kVrpoPpoContinuationRegistrationId) ||
      (test_fixture && config.registration_id.empty()) ||
      config.output_root.empty() || config.source_root.empty() ||
      config.evidence_root.empty() ||
      !VrpoPhase4eLowerHex64(config.source_code_sha256) ||
      !VrpoPhase4eLowerHex64(config.executed_binary_sha256) ||
      config.executed_binary_size <= 0 ||
      config.prior.observations.empty() ||
      config.prior.prior_update_summaries.size() != 4 ||
      config.prior.frozen_update1_episodes.size() != 16 ||
      config.prior.frozen_update1_payload_sha256.empty() ||
      config.prior.actor_path.empty() || config.prior.optimizer_path.empty()) {
    return fail("PPO-continuation identity/prior contract is invalid");
  }
  if (std::filesystem::exists(config.output_root)) {
    return fail("PPO-continuation output root must be fresh and absent");
  }
  if (config.rollout_games != kVrpoPpoContinuationGamesPerUpdate ||
      config.threads != 8 || config.eval_batch_size != 64 ||
      config.eval_timeout_ms != 2 ||
      !config.evaluator_device_synchronize ||
      config.deterministic_rollout_eval || config.seed_scheme_version != 2 ||
      !config.runtime_device_is_cuda || config.runtime_device_index < 0 ||
      !config.one_gpu_process || config.runtime_process_id <= 0 ||
      (!test_fixture && config.base_seed != kVrpoPpoContinuationBaseSeed) ||
      (!test_fixture &&
       config.start_episode_id != kVrpoPpoContinuationStartEpisodeId) ||
      (test_fixture && config.base_seed == 0) ||
      config.start_episode_id == 0 || config.diagnostics_only ||
      config.rollout_amp || config.train_amp || !config.allow_tf32 ||
      config.pipeline || config.online_search_collection ||
      config.search_pi_mode || config.train_value_only ||
      config.sample_counterfactual_states || config.has_search_label_dir ||
      config.ordinary_checkpoint_writes_enabled || config.anneal_lr) {
    return fail("PPO-continuation execution/side-path contract is invalid");
  }
  for (double shaping :
       {config.shaped_reward_weight, config.tleilaxu_breadcrumb_weight,
        config.tleilaxu_level7_breadcrumb_weight,
        config.specimen_exchange_penalty}) {
    if (!std::isfinite(shaping) || shaping != 0.0) {
      return fail("PPO-continuation shaping must be exactly zero");
    }
  }
  if (config.learning_rate != kVrpoPpoContinuationLearningRate ||
      config.ppo_update_epochs != kVrpoPpoContinuationActorEpochs ||
      config.reward_scale != 4.0 || config.gamma != 1.0 ||
      config.lambda != 1.0 || config.logit_cap != 10.0 ||
      config.ppo_minibatches != kVrpoTrainingMinibatchesPerEpoch ||
      config.ppo_minibatch_size != 2048 ||
      config.clip_epsilon != kVrpoTrainingClipEpsilon ||
      config.entropy_coefficient != kVrpoTrainingEntropyCoefficient ||
      config.value_coefficient != kVrpoTrainingValueCoefficient ||
      config.gradient_clip_norm != kVrpoTrainingGradientClipNorm ||
      !config.normalize_advantages || !config.clip_value_loss ||
      kVrpoPpoContinuationFinalActorSteps !=
          kVrpoPpoContinuationPriorActorSteps +
              kVrpoPpoContinuationNewUpdates *
                  kVrpoPpoContinuationStepsPerUpdate ||
      (!test_fixture &&
       config.start_episode_id +
                   kVrpoPpoContinuationNewUpdates *
                       kVrpoPpoContinuationGamesPerUpdate -
               1 !=
           kVrpoPpoContinuationEndEpisodeIdInclusive)) {
    return fail("PPO-continuation mechanics differ from registration");
  }
  return true;
}

struct VrpoPpoContinuationState {
  int attempted_new_updates = 0;
  int accepted_new_updates = 0;
  int64_t live_actor_optimizer_steps =
      kVrpoPpoContinuationPriorActorSteps;
  int64_t measured_peak_bytes = 0;
  std::string prior_actor_values_sha256;
  std::string source_q_values_sha256;
  std::string prior_actor_optimizer_state_sha256;
  std::string latest_actor_optimizer_state_sha256;
  std::string frozen_update1_corpus_sha256;
  std::vector<VrpoTrainingEpisode> frozen_update1_episodes;
  std::vector<VrpoPpoPilotEvidenceObservation> evidence_observations;
  json::Array prior_update_summaries;
  json::Array committed_new_updates;
  bool runtime_device_exact_match = false;
  bool stopped = false;
  bool completed = false;
  bool had_failure = false;
  std::string failure_reason;
};

inline bool InitializeVrpoPpoContinuationFromLoadedState(
    const VrpoPpoContinuationStartupConfig& startup,
    const std::shared_ptr<SharedDunePolicyValueNetImpl>& actor,
    const std::shared_ptr<torch::nn::Module>& q,
    torch::optim::AdamW& actor_optimizer,
    const torch::Device& selected_device,
    const VrpoPpoContinuationDeadline& deadline,
    VrpoPpoContinuationState* state, std::string* error,
    bool test_fixture = false) {
  auto fail = [&](const std::string& message) {
    if (error != nullptr) *error = message;
    if (state != nullptr) *state = VrpoPpoContinuationState{};
    return false;
  };
  if (state == nullptr || actor == nullptr || q == nullptr ||
      !ValidateVrpoPpoContinuationStartupConfig(
          startup, error, test_fixture) ||
      !deadline.Check("before continuation initialization", error) ||
      !vrpo_ppo_pilot_internal::CheckFreeSpaceBeforeStart(
          startup.output_root, error) ||
      !ValidateVrpoScheduleQProvenanceInstrumentation(
          kVrpoPpoContinuationQRole, error)) {
    return fail(error != nullptr && !error->empty()
                    ? *error
                    : "PPO-continuation initialization rejected");
  }
  vrpo_training_internal::ModuleRuntimePlacement actor_placement;
  vrpo_training_internal::ModuleRuntimePlacement q_placement;
  const int selected_cuda_index = selected_device.is_cuda()
      ? (selected_device.has_index() ? selected_device.index()
                                     : c10::cuda::current_device())
      : -1;
  if (!vrpo_training_internal::CaptureUniformModuleRuntimePlacement(
          *actor, &actor_placement, error) ||
      !vrpo_training_internal::CaptureUniformModuleRuntimePlacement(
          *q, &q_placement, error)) {
    return fail(error != nullptr && !error->empty()
                    ? *error
                    : "PPO-continuation actor/Q placement is nonuniform");
  }
  const bool actor_selected_device =
      actor_placement.device.type() == selected_device.type() &&
      (!selected_device.is_cuda() ||
       actor_placement.device.index() == selected_cuda_index);
  const bool q_selected_device =
      q_placement.device.type() == selected_device.type() &&
      (!selected_device.is_cuda() ||
       q_placement.device.index() == selected_cuda_index);
  if (actor_placement.dtype != torch::kFloat32 ||
      q_placement.dtype != torch::kFloat32 ||
      !actor_selected_device || !q_selected_device ||
      (!test_fixture &&
       (!selected_device.is_cuda() ||
        startup.runtime_device_index != selected_cuda_index))) {
    return fail(error != nullptr && !error->empty()
                    ? *error
                    : "PPO-continuation actor/Q placement must be uniform FP32 CUDA");
  }
  VrpoPpoContinuationState initialized;
  if (!vrpo_training_internal::ModuleValueSha256(
          *actor, "", &initialized.prior_actor_values_sha256, error) ||
      !vrpo_training_internal::ModuleValueSha256(
          *q, "", &initialized.source_q_values_sha256, error) ||
      !vrpo_ppo_pilot_internal::ActorOptimizerStateSha256(
          actor_optimizer, *actor, kVrpoPpoContinuationPriorActorSteps,
          &initialized.prior_actor_optimizer_state_sha256, error) ||
      (!test_fixture &&
       initialized.prior_actor_values_sha256 !=
           kVrpoPpoContinuationPriorActorValuesSha256) ||
      (!test_fixture &&
       initialized.prior_actor_optimizer_state_sha256 !=
           kVrpoPpoContinuationPriorOptimizerStateSha256) ||
      (!test_fixture &&
       initialized.source_q_values_sha256 !=
           kVrpoPpoContinuationSourceQValuesSha256)) {
    return fail(error != nullptr && !error->empty()
                    ? *error
                    : "PPO-continuation U4 actor/Adam/Q identity rejected");
  }
  initialized.latest_actor_optimizer_state_sha256 =
      initialized.prior_actor_optimizer_state_sha256;
  initialized.frozen_update1_corpus_sha256 =
      startup.prior.frozen_update1_payload_sha256;
  initialized.frozen_update1_episodes =
      startup.prior.frozen_update1_episodes;
  initialized.evidence_observations = startup.prior.observations;
  initialized.prior_update_summaries = startup.prior.prior_update_summaries;
  initialized.runtime_device_exact_match = true;
  std::error_code ec;
  bool owns_root = false;
  const bool claimed_root =
      std::filesystem::create_directory(startup.output_root, ec);
  if (claimed_root && !ec) owns_root = true;
  if (!claimed_root || ec ||
      !VrpoFsyncDirectory(startup.output_root.parent_path(), error) ||
      !vrpo_ppo_continuation_internal::ObserveBytes(
          startup.output_root, &initialized.measured_peak_bytes, error)) {
    if (owns_root) {
      vrpo_ppo_pilot_internal::CleanupPath(startup.output_root);
      owns_root = false;
    }
    return fail(error != nullptr && !error->empty()
                    ? *error
                    : "PPO-continuation output root creation failed");
  }
  *state = std::move(initialized);
  owns_root = false;
  return true;
}

inline bool LoadAndInitializeVrpoPpoContinuation(
    const VrpoPpoContinuationStartupConfig& startup,
    const std::shared_ptr<SharedDunePolicyValueNetImpl>& actor,
    const std::shared_ptr<torch::nn::Module>& q,
    const torch::Device& device,
    const VrpoPpoContinuationDeadline& deadline,
    std::unique_ptr<torch::optim::AdamW>* actor_optimizer,
    VrpoPpoContinuationState* state, std::string* error,
    bool test_fixture = false) {
  auto fail = [&](const std::string& message) {
    if (error != nullptr) *error = message;
    if (actor_optimizer != nullptr) actor_optimizer->reset();
    return false;
  };
  if (actor == nullptr || q == nullptr || actor_optimizer == nullptr ||
      state == nullptr || !deadline.Check("before strict U4 load", error)) {
    return fail(error != nullptr && !error->empty()
                    ? *error
                    : "PPO-continuation strict U4 load arguments invalid");
  }
  size_t actor_size = 0;
  size_t optimizer_size = 0;
  const std::string actor_file_sha = ComputeFileSHA256(
      startup.prior.actor_path.string(), &actor_size);
  const std::string optimizer_file_sha = ComputeFileSHA256(
      startup.prior.optimizer_path.string(), &optimizer_size);
  if (actor_size == 0 || optimizer_size == 0 ||
      (!test_fixture && actor_file_sha !=
           VrpoPpoContinuationCanonicalEvidenceFiles()[8].expected_sha256) ||
      (!test_fixture && optimizer_file_sha !=
           VrpoPpoContinuationCanonicalEvidenceFiles()[9].expected_sha256)) {
    return fail("PPO-continuation U4 actor/optimizer file identity mismatch");
  }
  if (!LoadVrpoModule(*actor, startup.prior.actor_path, error) ||
      !MakeVrpoScheduleFreshActorOptimizer(
          *actor, kVrpoPpoContinuationLearningRate, actor_optimizer, error)) {
    return fail(error != nullptr ? *error
                                 : "PPO-continuation U4 actor load failed");
  }
  try {
    torch::load(**actor_optimizer, startup.prior.optimizer_path.string());
  } catch (const std::exception& exception) {
    return fail(std::string("PPO-continuation U4 Adam load failed: ") +
                exception.what());
  }
  if (!vrpo_training_internal::MigrateAdamWOptimizerStateToParameterDevices(
          **actor_optimizer, error) ||
      !deadline.Check("after strict U4 load", error) ||
      !InitializeVrpoPpoContinuationFromLoadedState(
          startup, actor, q, **actor_optimizer, device, deadline, state,
          error, test_fixture)) {
    return fail(error != nullptr ? *error
                                 : "PPO-continuation U4 state rejected");
  }
  return true;
}

enum class VrpoPpoContinuationFailurePoint {
  kNone,
  kLateAfterTraining,
  kAfterActorTemp,
  kAfterOptimizerTemp,
  kAfterReload,
  kForceHealthStop,
  kForceHealthPassForTest,
};

enum class VrpoPpoContinuationDisposition {
  kContinue,
  kEarlyStop,
  kComplete,
};

inline bool ValidateVrpoPpoContinuationCommittedUpdateDirectory(
    const std::filesystem::path& output_root, int global_update,
    const json::Value& summary_value, std::string* error) {
  auto fail = [&](const std::string& message) {
    if (error != nullptr) *error = message;
    return false;
  };
  if (!summary_value.IsObject() ||
      global_update < kVrpoPpoContinuationFirstGlobalUpdate ||
      global_update > kVrpoPpoContinuationLastGlobalUpdate) {
    return fail("PPO-continuation committed summary is malformed");
  }
  const auto& summary = summary_value.GetObject();
  const auto directory =
      output_root / ("update_" + std::to_string(global_update));
  const auto checksums_path = directory / "CHECKSUMS.json";
  const auto result_path = directory / "UPDATE_RESULT.json";
  json::Object checksums;
  if (!vrpo_ppo_continuation_internal::ReadJsonObject(
          checksums_path, &checksums, error)) {
    return false;
  }
  const auto schema = checksums.find("schema");
  const auto files = checksums.find("files");
  if (schema == checksums.end() || !schema->second.IsString() ||
      schema->second.GetString() != kVrpoPpoContinuationChecksumsSchema ||
      files == checksums.end() || !files->second.IsArray() ||
      files->second.GetArray().size() != 5) {
    return fail("PPO-continuation checksum schema/file count is invalid");
  }
  const std::set<std::string> expected_files = {
      "ACTOR_CORPUS.bin", "EPISODE_EVIDENCE.json", "actor_model.pt",
      "actor_optimizer.pt", "UPDATE_RESULT.json"};
  std::set<std::string> seen;
  for (const auto& file_value : files->second.GetArray()) {
    if (!file_value.IsObject()) {
      return fail("PPO-continuation checksum row is invalid");
    }
    const auto& file = file_value.GetObject();
    const auto filename = file.find("filename");
    const auto sha = file.find("sha256");
    const auto size = file.find("size");
    if (filename == file.end() || !filename->second.IsString() ||
        sha == file.end() || !sha->second.IsString() ||
        !VrpoPhase4eLowerHex64(sha->second.GetString()) ||
        size == file.end() || !size->second.IsInt() ||
        size->second.GetInt() <= 0 ||
        !expected_files.count(filename->second.GetString()) ||
        !seen.insert(filename->second.GetString()).second) {
      return fail("PPO-continuation checksum row fields are invalid");
    }
    const auto path = directory / filename->second.GetString();
    if (!std::filesystem::is_regular_file(path) ||
        std::filesystem::is_symlink(path)) {
      return fail("PPO-continuation committed file is missing/nonregular");
    }
    size_t observed_size = 0;
    const std::string observed = ComputeFileSHA256(path.string(),
                                                   &observed_size);
    if (observed != sha->second.GetString() ||
        static_cast<int64_t>(observed_size) != size->second.GetInt()) {
      return fail("PPO-continuation committed file checksum drifted");
    }
  }
  if (seen != expected_files) {
    return fail("PPO-continuation committed file set is incomplete");
  }
  json::Object result;
  if (!vrpo_ppo_continuation_internal::ReadJsonObject(
          result_path, &result, error)) {
    return false;
  }
  if (!vrpo_ppo_continuation_internal::StringIs(
          result, "classification", "VALID_UPDATE_HEALTH_PASS") ||
      !vrpo_ppo_continuation_internal::StringIs(result, "status", "VALID") ||
      !vrpo_ppo_continuation_internal::IntIs(
          result, "global_update", global_update) ||
      !vrpo_ppo_continuation_internal::IntIs(
          result, "actor_optimizer_step_after", global_update * 16) ||
      !vrpo_ppo_continuation_internal::IntIs(
          result, "actor_optimizer_steps_this_update", 16) ||
      !vrpo_ppo_continuation_internal::BoolIs(
          result, "strict_reload_passed", true) ||
      !vrpo_ppo_continuation_internal::BoolIs(result, "canary_exact", true) ||
      !vrpo_ppo_continuation_internal::BoolIs(result, "q_bit_inert", true) ||
      !vrpo_ppo_continuation_internal::BoolIs(
          result, "current_rollout_health_pass", true) ||
      !vrpo_ppo_continuation_internal::BoolIs(
          result, "cumulative_fixed_update1_health_pass", true)) {
    return fail("PPO-continuation update result is not authorization-valid");
  }
  const auto summary_actor =
      summary.find("live_actor_after_disposition_sha256");
  const auto result_actor = result.find("actor_after_sha256");
  if (!vrpo_ppo_continuation_internal::IntIs(
          summary, "global_update", global_update) ||
      !vrpo_ppo_continuation_internal::BoolIs(
          summary, "current_update_rolled_back", false) ||
      summary_actor == summary.end() || !summary_actor->second.IsString() ||
      result_actor == result.end() || !result_actor->second.IsString() ||
      summary_actor->second.GetString() != result_actor->second.GetString()) {
    return fail("PPO-continuation committed summary is not valid");
  }
  size_t result_size = 0;
  const std::string result_sha =
      ComputeFileSHA256(result_path.string(), &result_size);
  const auto summary_sha = summary.find("update_result_sha256");
  const auto summary_size = summary.find("update_result_size");
  if (summary_sha == summary.end() || !summary_sha->second.IsString() ||
      summary_sha->second.GetString() != result_sha ||
      summary_size == summary.end() || !summary_size->second.IsInt() ||
      summary_size->second.GetInt() != static_cast<int64_t>(result_size)) {
    return fail("PPO-continuation summary/result identity mismatch");
  }
  return true;
}

inline bool WriteVrpoPpoContinuationUpdate(
    const VrpoPpoContinuationStartupConfig& startup,
    const VrpoPhase4ManifestBinding& binding,
    const VrpoExpandedExpectedLayout& layout,
    const std::vector<VrpoTrainingEpisode>& episodes,
    const VrpoPhase4ePairingStats& pairing,
    const std::string& collection_actor_values_sha256,
    const std::shared_ptr<SharedDunePolicyValueNetImpl>& actor,
    const std::shared_ptr<torch::nn::Module>& q,
    torch::optim::AdamW& actor_optimizer, const torch::Device& device,
    const VrpoPpoContinuationDeadline& deadline, int global_update,
    VrpoPpoContinuationFailurePoint failure_point,
    VrpoPpoContinuationState* state,
    VrpoPpoContinuationDisposition* disposition,
    json::Object* update_result, std::string* error) {
  const int local_update =
      global_update - kVrpoPpoContinuationFirstGlobalUpdate + 1;
  const auto temp_dir = startup.output_root /
      (".update_" + std::to_string(global_update) + ".tmp");
  const auto final_dir = startup.output_root /
      ("update_" + std::to_string(global_update));
  bool owns_temp_dir = false;
  bool owns_final_dir = false;
  auto fail = [&](const std::string& message) {
    if (owns_temp_dir) {
      vrpo_ppo_pilot_internal::CleanupPath(temp_dir);
      owns_temp_dir = false;
    }
    if (owns_final_dir) {
      vrpo_ppo_pilot_internal::CleanupPath(final_dir);
      owns_final_dir = false;
    }
    if (error != nullptr) *error = message;
    if (update_result != nullptr) update_result->clear();
    return false;
  };
  if (state == nullptr || disposition == nullptr || update_result == nullptr ||
      actor == nullptr || q == nullptr || state->stopped || state->completed ||
      local_update != state->attempted_new_updates + 1 || local_update < 1 ||
      local_update > kVrpoPpoContinuationNewUpdates ||
      episodes.size() != kVrpoPpoContinuationGamesPerUpdate ||
      pairing.episodes != kVrpoPpoContinuationGamesPerUpdate ||
      VrpoQConstructorCalls() != 1 || VrpoQForwardCheckedCalls() != 0 ||
      VrpoScheduleQTargetComputations() != 0 ||
      std::filesystem::exists(temp_dir) || std::filesystem::exists(final_dir) ||
      !deadline.Check("before continuation update", error)) {
    return fail(error != nullptr && !error->empty()
                    ? *error
                    : "PPO-continuation update sequence rejected");
  }
  const uint64_t expected_start = startup.start_episode_id +
      static_cast<uint64_t>(local_update - 1) *
          kVrpoPpoContinuationGamesPerUpdate;
  const uint64_t expected_next =
      expected_start + kVrpoPpoContinuationGamesPerUpdate;
  if (episodes.empty() || pairing.episode_evidence.empty() ||
      episodes.front().episode_id != expected_start ||
      episodes.back().episode_id + 1 != expected_next ||
      pairing.episode_evidence.front().episode_id != expected_start ||
      pairing.episode_evidence.back().episode_id + 1 != expected_next ||
      !ValidateVrpoPhase4ePairingStatsAgainstEpisodes(
          pairing, episodes, error)) {
    return fail(error != nullptr && !error->empty()
                    ? *error
                    : "PPO-continuation update IDs/pairing rejected");
  }

  const int64_t expected_before_step =
      kVrpoPpoContinuationPriorActorSteps +
      static_cast<int64_t>(local_update - 1) *
          kVrpoPpoContinuationStepsPerUpdate;
  const int64_t expected_after_step =
      expected_before_step + kVrpoPpoContinuationStepsPerUpdate;
  std::string actor_before;
  std::string q_before;
  std::string optimizer_before;
  if (!vrpo_training_internal::ModuleValueSha256(
          *actor, "", &actor_before, error) ||
      actor_before != collection_actor_values_sha256 ||
      !vrpo_training_internal::ModuleValueSha256(
          *q, "", &q_before, error) ||
      q_before != state->source_q_values_sha256 ||
      !vrpo_ppo_pilot_internal::ActorOptimizerStateSha256(
          actor_optimizer, *actor, expected_before_step, &optimizer_before,
          error) ||
      optimizer_before != state->latest_actor_optimizer_state_sha256) {
    return fail(error != nullptr && !error->empty()
                    ? *error
                    : "PPO-continuation pre-update state rejected");
  }

  std::string corpus_bytes;
  VrpoScheduleCorpusIdentity corpus_identity;
  if (!EncodeVrpoScheduleActorCorpus(
          episodes, &corpus_bytes, &corpus_identity, error)) {
    return fail(error != nullptr ? *error
                                 : "PPO-continuation corpus encode failed");
  }
  std::vector<VrpoTrainingEpisode> decoded;
  VrpoScheduleCorpusIdentity decoded_identity;
  if (!DecodeVrpoScheduleActorCorpus(
          corpus_bytes, &decoded, &decoded_identity, error) ||
      decoded_identity.payload_sha256 != corpus_identity.payload_sha256 ||
      state->frozen_update1_episodes.size() != 16 ||
      state->frozen_update1_corpus_sha256.empty()) {
    return fail(error != nullptr && !error->empty()
                    ? *error
                    : "PPO-continuation corpus/anchor rejected");
  }

  VrpoScheduleActorTransaction transaction;
  if (!transaction.Capture(*actor, actor_optimizer, error)) {
    return fail(error != nullptr ? *error
                                 : "PPO-continuation rollback capture failed");
  }
  const uint64_t update_seed = vrpo_training_internal::SplitMix64(
      startup.base_seed ^ expected_start ^
      (0x50504f434f4e5434ULL + static_cast<uint64_t>(global_update)));
  VrpoActorForward actor_forward = [actor](const torch::Tensor& input) {
    const auto out = actor->forward(input);
    return VrpoActorTrainingOutput{out.logits, out.values};
  };
  VrpoScheduleActorOnlyUpdateStats stats;
  const VrpoScheduleRunDeadline* no_schedule_deadline = nullptr;
  if (!RunVrpoScheduleActorOnlyPpoCell(
          decoded, update_seed, kVrpoPpoContinuationActorEpochs, *actor,
          actor_optimizer, actor_forward,
          /*four_epoch_equivalence_test=*/false, no_schedule_deadline, &stats,
          error, /*require_fresh_optimizer=*/false)) {
    return fail(error != nullptr ? *error
                                 : "PPO-continuation actor update failed");
  }
  if (!deadline.Check("after continuation actor update", error)) {
    return fail(*error);
  }
  if (failure_point == VrpoPpoContinuationFailurePoint::kLateAfterTraining) {
    return fail("injected PPO-continuation late failure after training");
  }
  std::string actor_after;
  std::string q_after;
  std::string optimizer_after;
  if (stats.ppo.actor_optimizer_steps !=
          kVrpoPpoContinuationStepsPerUpdate ||
      stats.ppo.q_optimizer_steps != 0 || stats.q_target_computations != 0 ||
      !vrpo_training_internal::ModuleValueSha256(
          *actor, "", &actor_after, error) || actor_after == actor_before ||
      !vrpo_training_internal::ModuleValueSha256(
          *q, "", &q_after, error) || q_after != q_before ||
      !vrpo_ppo_pilot_internal::ActorOptimizerStateSha256(
          actor_optimizer, *actor, expected_after_step, &optimizer_after,
          error)) {
    return fail(error != nullptr && !error->empty()
                    ? *error
                    : "PPO-continuation actor/Q/Adam postcondition failed");
  }

  VrpoScheduleHealthMetrics current_health;
  VrpoScheduleHealthMetrics cumulative_health;
  const bool inject_health = failure_point ==
          VrpoPpoContinuationFailurePoint::kForceHealthPassForTest ||
      failure_point == VrpoPpoContinuationFailurePoint::kForceHealthStop;
  if (inject_health) {
    current_health = VrpoScheduleHealthMetrics{
        64, 0.01, 0.02, 0.02, 0.03, 0.9, 1.0, 1.1,
        0.01, 0.02, 0.01, true};
    cumulative_health = current_health;
  } else if (!EvaluateVrpoSchedulePolicyHealth(
                 decoded, actor_forward, device, 256, &current_health,
                 error) ||
             !EvaluateVrpoSchedulePolicyHealth(
                 state->frozen_update1_episodes, actor_forward, device, 256,
                 &cumulative_health, error) ||
             !deadline.Check("after continuation health", error)) {
    return fail(error != nullptr ? *error
                                 : "PPO-continuation health failed");
  }
  bool current_pass =
      vrpo_ppo_pilot_internal::CurrentHealthPass(current_health);
  bool cumulative_pass =
      vrpo_ppo_pilot_internal::CumulativeHealthPass(cumulative_health);
  if (failure_point ==
      VrpoPpoContinuationFailurePoint::kForceHealthPassForTest) {
    current_pass = true;
    cumulative_pass = true;
  } else if (failure_point ==
             VrpoPpoContinuationFailurePoint::kForceHealthStop) {
    current_pass = false;
  }
  const bool health_pass = current_pass && cumulative_pass;

  std::error_code ec;
  if (!std::filesystem::create_directory(temp_dir, ec) || ec) {
    return fail("PPO-continuation temp directory creation failed");
  }
  owns_temp_dir = true;
  const auto corpus_path = temp_dir / "ACTOR_CORPUS.bin";
  if (!vrpo_ppo_pilot_internal::WriteAtomicText(
          corpus_path, corpus_bytes, error)) {
    return fail(*error);
  }
  json::Object evidence;
  evidence["schema"] = kVrpoPpoContinuationEvidenceSchema;
  evidence["global_update"] = static_cast<int64_t>(global_update);
  evidence["local_continuation_update"] = static_cast<int64_t>(local_update);
  evidence["start_episode_id"] = static_cast<int64_t>(expected_start);
  evidence["next_episode_id"] = static_cast<int64_t>(expected_next);
  evidence["episodes"] = pairing.episodes;
  evidence["capture_rows"] = pairing.capture_rows;
  evidence["rollout_rows"] = pairing.rollout_rows;
  evidence["paired_rows"] = pairing.paired_rows;
  evidence["pairing_sha256"] = pairing.canonical_sha256;
  evidence["outcome_convention"] = kVrpoPhase4eOutcomeConvention;
  json::Array episode_records;
  for (const auto& episode : pairing.episode_evidence) {
    json::Object item;
    item["episode_id"] = static_cast<int64_t>(episode.episode_id);
    item["capture_rows"] = episode.capture_rows;
    item["rollout_rows"] = episode.rollout_rows;
    item["paired_rows"] = episode.paired_rows;
    item["pairing_sha256"] = episode.pairing_sha256;
    json::Array returns;
    for (double value : episode.terminal_returns_absolute) {
      returns.emplace_back(VrpoPhase4eCanonicalDouble(value));
    }
    item["terminal_returns_exact"] = std::move(returns);
    episode_records.emplace_back(std::move(item));
  }
  evidence["episode_evidence"] = std::move(episode_records);
  const auto evidence_path = temp_dir / "EPISODE_EVIDENCE.json";
  if (!vrpo_ppo_pilot_internal::WriteAtomicText(
          evidence_path, json::ToString(evidence, true) + "\n", error)) {
    return fail(*error);
  }

  const auto actor_path = temp_dir / "actor_model.pt";
  const auto actor_temp = temp_dir / ".actor_model.pt.save";
  const auto optimizer_path = temp_dir / "actor_optimizer.pt";
  const auto optimizer_temp = temp_dir / ".actor_optimizer.pt.save";
  try {
    torch::save(actor, actor_temp.string());
  } catch (const std::exception& exception) {
    return fail(std::string("PPO-continuation actor save failed: ") +
                exception.what());
  }
  if (!VrpoFsyncFile(actor_temp, error)) return fail(*error);
  if (failure_point ==
      VrpoPpoContinuationFailurePoint::kAfterActorTemp) {
    return fail("injected PPO-continuation failure after actor temp");
  }
  try {
    torch::save(actor_optimizer, optimizer_temp.string());
  } catch (const std::exception& exception) {
    return fail(std::string("PPO-continuation optimizer save failed: ") +
                exception.what());
  }
  if (!VrpoFsyncFile(optimizer_temp, error)) return fail(*error);
  if (failure_point ==
      VrpoPpoContinuationFailurePoint::kAfterOptimizerTemp) {
    return fail("injected PPO-continuation failure after optimizer temp");
  }

  auto reloaded_actor = std::make_shared<SharedDunePolicyValueNetImpl>(
      layout.actor_observation_dim, layout.actor_hidden_dim,
      layout.actor_action_dim, layout.actor_residual_blocks, false);
  reloaded_actor->to(device, torch::kFloat32);
  std::unique_ptr<torch::optim::AdamW> reloaded_optimizer;
  if (!MakeVrpoScheduleFreshActorOptimizer(
          *reloaded_actor, kVrpoPpoContinuationLearningRate,
          &reloaded_optimizer, error)) {
    return fail(*error);
  }
  try {
    torch::load(reloaded_actor, actor_temp.string(), device);
    torch::load(*reloaded_optimizer, optimizer_temp.string());
  } catch (const std::exception& exception) {
    return fail(std::string("PPO-continuation strict reload failed: ") +
                exception.what());
  }
  if (!vrpo_training_internal::MigrateAdamWOptimizerStateToParameterDevices(
          *reloaded_optimizer, error)) {
    return fail(*error);
  }
  std::string reloaded_actor_hash;
  std::string reloaded_optimizer_hash;
  if (!vrpo_training_internal::ModuleValueSha256(
          *reloaded_actor, "", &reloaded_actor_hash, error) ||
      !vrpo_ppo_pilot_internal::ActorOptimizerStateSha256(
          *reloaded_optimizer, *reloaded_actor, expected_after_step,
          &reloaded_optimizer_hash, error) ||
      reloaded_actor_hash != actor_after ||
      reloaded_optimizer_hash != optimizer_after) {
    return fail("PPO-continuation strict reload hash mismatch");
  }
  torch::Tensor canary = decoded.front().rows.front().actor_input
                             .reshape({1, -1}).to(device, torch::kFloat32);
  torch::Tensor live_canary;
  torch::Tensor reload_canary;
  try {
    torch::NoGradGuard no_grad;
    live_canary = actor_forward(canary).logits.detach().cpu();
    reload_canary = reloaded_actor->forward(canary).logits.detach().cpu();
  } catch (const std::exception& exception) {
    return fail(std::string("PPO-continuation canary failed: ") +
                exception.what());
  }
  if (!torch::equal(live_canary, reload_canary)) {
    return fail("PPO-continuation strict reload canary mismatch");
  }
  if (failure_point == VrpoPpoContinuationFailurePoint::kAfterReload) {
    return fail("injected PPO-continuation failure after reload");
  }
  std::filesystem::rename(actor_temp, actor_path);
  std::filesystem::rename(optimizer_temp, optimizer_path);

  if ((!layout.test_fixture && !startup.prior.observations.empty() &&
       !vrpo_ppo_continuation_internal::EvidenceStillExact(
           startup.evidence_root, state->evidence_observations, error)) ||
      !ValidateVrpoScheduleQProvenanceInstrumentation(
          kVrpoPpoContinuationQRole, error)) {
    return fail(error != nullptr && !error->empty()
                    ? *error
                    : "PPO-continuation prior/Q provenance drifted");
  }
  json::Array checksum_files;
  for (const auto& path :
       {corpus_path, evidence_path, actor_path, optimizer_path}) {
    json::Object identity = vrpo_ppo_pilot_internal::FileIdentityJson(path);
    const auto sha = identity.find("sha256");
    const auto size = identity.find("size");
    if (sha == identity.end() || !sha->second.IsString() ||
        !VrpoPhase4eLowerHex64(sha->second.GetString()) ||
        size == identity.end() || !size->second.IsInt() ||
        size->second.GetInt() <= 0) {
      return fail("PPO-continuation retained file identity failed");
    }
    checksum_files.emplace_back(std::move(identity));
  }

  json::Object root;
  root["schema"] = kVrpoPpoContinuationUpdateSchema;
  root["classification"] = health_pass ? "VALID_UPDATE_HEALTH_PASS"
                                          : "VALID_EARLY_STOP_HEALTH";
  root["status"] = "VALID";
  root["global_update"] = static_cast<int64_t>(global_update);
  root["local_continuation_update"] = static_cast<int64_t>(local_update);
  root["registration_id"] = startup.registration_id;
  root["profile"] = startup.profile;
  root["learning_rate"] = startup.learning_rate;
  root["learning_rate_exact"] =
      VrpoPhase4eCanonicalDouble(startup.learning_rate);
  root["actor_epochs"] = int64_t{kVrpoPpoContinuationActorEpochs};
  root["update_seed"] = static_cast<int64_t>(update_seed);
  root["update_seed_uint64"] = std::to_string(update_seed);
  root["start_episode_id"] = static_cast<int64_t>(expected_start);
  root["end_episode_id_inclusive"] =
      static_cast<int64_t>(expected_next - 1);
  root["next_episode_id"] = static_cast<int64_t>(expected_next);
  root["rollout_games"] = int64_t{kVrpoPpoContinuationGamesPerUpdate};
  root["corpus_payload_sha256"] = corpus_identity.payload_sha256;
  root["corpus_rows"] = corpus_identity.rows;
  root["pairing_sha256"] = pairing.canonical_sha256;
  root["actor_epoch_partition_sha256"] =
      stats.ppo.actor_epoch_partition_sha256[0];
  root["collection_actor_values_sha256"] =
      collection_actor_values_sha256;
  root["actor_before_sha256"] = actor_before;
  root["actor_after_sha256"] = actor_after;
  root["actor_optimizer_before_sha256"] = optimizer_before;
  root["actor_optimizer_after_sha256"] = optimizer_after;
  root["actor_optimizer_step_before"] = expected_before_step;
  root["actor_optimizer_step_after"] = expected_after_step;
  root["actor_optimizer_steps_this_update"] =
      stats.ppo.actor_optimizer_steps;
  root["value_head_before_sha256"] = stats.ppo.value_head_before_sha256;
  root["value_head_after_sha256"] = stats.ppo.value_head_after_sha256;
  root["actor_and_value_moved"] =
      stats.ppo.actor_values_before_sha256 !=
          stats.ppo.actor_values_after_sha256 &&
      stats.ppo.value_head_before_sha256 !=
          stats.ppo.value_head_after_sha256;
  root["q_role"] = kVrpoPpoContinuationQRole;
  root["source_actor_model_sha256"] = binding.source_actor_model_sha256;
  root["source_actor_manifest_sha256"] =
      binding.source_actor_manifest_sha256;
  root["source_experiment_uuid"] = binding.experiment_uuid;
  root["q_before_sha256"] = q_before;
  root["q_after_sha256"] = q_after;
  root["q_bit_inert"] = q_before == q_after;
  root["q_constructor_calls"] = VrpoQConstructorCalls();
  root["q_forward_calls"] = VrpoQForwardCheckedCalls();
  root["q_target_computations"] = VrpoScheduleQTargetComputations();
  root["q_optimizer_constructed"] = false;
  root["q_optimizer_steps"] = int64_t{0};
  root["current_rollout_health"] =
      json::Value(VrpoScheduleHealthJson(current_health));
  root["current_rollout_health_pass"] = current_pass;
  root["cumulative_fixed_update1_health"] =
      json::Value(VrpoScheduleHealthJson(cumulative_health));
  root["cumulative_fixed_update1_health_pass"] = cumulative_pass;
  root["fixed_original_pilot_update1_corpus"] = true;
  root["strict_reload_passed"] = true;
  root["canary_exact"] = true;
  root["prior_evidence_untouched"] = true;
  root["rollback_state"] = health_pass
      ? "not_required_committed"
      : "health_gate_failed_current_actor_and_optimizer_rolled_back";
  root["post_update_candidate_retained_for_audit"] = !health_pass;
  const auto result_path = temp_dir / "UPDATE_RESULT.json";
  if (!vrpo_ppo_pilot_internal::WriteAtomicText(
          result_path, json::ToString(root, true) + "\n", error)) {
    return fail(*error);
  }
  checksum_files.emplace_back(
      vrpo_ppo_pilot_internal::FileIdentityJson(result_path));
  json::Object checksums;
  checksums["schema"] = kVrpoPpoContinuationChecksumsSchema;
  checksums["files"] = std::move(checksum_files);
  const auto checksums_path = temp_dir / "CHECKSUMS.json";
  if (!vrpo_ppo_pilot_internal::WriteAtomicText(
          checksums_path, json::ToString(checksums, true) + "\n", error) ||
      !vrpo_ppo_continuation_internal::ObserveBytes(
          startup.output_root, &state->measured_peak_bytes, error)) {
    return fail(*error);
  }
  std::filesystem::rename(temp_dir, final_dir);
  owns_temp_dir = false;
  owns_final_dir = true;
  if (!VrpoFsyncDirectory(startup.output_root, error) ||
      !deadline.Check("after continuation update commit", error) ||
      !vrpo_ppo_continuation_internal::ObserveBytes(
          startup.output_root, &state->measured_peak_bytes, error)) {
    return fail(error != nullptr ? *error
                                 : "PPO-continuation commit failed");
  }

  if (health_pass) transaction.Commit();
  state->attempted_new_updates = local_update;
  if (health_pass) {
    state->accepted_new_updates = local_update;
    state->live_actor_optimizer_steps = expected_after_step;
    state->latest_actor_optimizer_state_sha256 = optimizer_after;
  }
  json::Object summary;
  summary["global_update"] = static_cast<int64_t>(global_update);
  summary["local_continuation_update"] = static_cast<int64_t>(local_update);
  summary["classification"] = root["classification"];
  summary["start_episode_id"] = static_cast<int64_t>(expected_start);
  summary["next_episode_id"] = static_cast<int64_t>(expected_next);
  summary["update_seed"] = static_cast<int64_t>(update_seed);
  summary["update_seed_uint64"] = std::to_string(update_seed);
  summary["corpus_payload_sha256"] = corpus_identity.payload_sha256;
  summary["pairing_sha256"] = pairing.canonical_sha256;
  summary["actor_epoch_partition_sha256"] =
      stats.ppo.actor_epoch_partition_sha256[0];
  summary["actor_after_sha256"] = actor_after;
  summary["actor_optimizer_after_sha256"] = optimizer_after;
  summary["live_actor_after_disposition_sha256"] =
      health_pass ? actor_after : actor_before;
  summary["live_actor_optimizer_after_disposition_sha256"] =
      health_pass ? optimizer_after : optimizer_before;
  summary["current_update_rolled_back"] = !health_pass;
  summary["current_rollout_health_pass"] = current_pass;
  summary["cumulative_fixed_update1_health_pass"] = cumulative_pass;
  size_t result_size = 0;
  summary["update_result_sha256"] = ComputeFileSHA256(
      (final_dir / "UPDATE_RESULT.json").string(), &result_size);
  summary["update_result_size"] = static_cast<int64_t>(result_size);
  state->committed_new_updates.emplace_back(std::move(summary));
  if (!health_pass) {
    state->stopped = true;
    *disposition = VrpoPpoContinuationDisposition::kEarlyStop;
  } else if (global_update == kVrpoPpoContinuationLastGlobalUpdate) {
    state->completed = true;
    *disposition = VrpoPpoContinuationDisposition::kComplete;
  } else {
    *disposition = VrpoPpoContinuationDisposition::kContinue;
  }
  *update_result = std::move(root);
  owns_final_dir = false;
  return true;
}

inline bool VrpoPpoContinuationRaw128DesignAuthorization(
    const VrpoPpoContinuationStartupConfig& startup,
    const VrpoPpoContinuationState& state, torch::nn::Module& actor,
    torch::optim::AdamW& actor_optimizer, std::string* denial_reason,
    bool test_fixture = false) {
  auto deny = [&](const std::string& reason) {
    if (denial_reason != nullptr) *denial_reason = reason;
    return false;
  };
  if (!state.completed || state.stopped || state.had_failure ||
      state.attempted_new_updates != kVrpoPpoContinuationNewUpdates ||
      state.accepted_new_updates != kVrpoPpoContinuationNewUpdates ||
      state.prior_update_summaries.size() != 4 ||
      state.committed_new_updates.size() != 4 ||
      state.live_actor_optimizer_steps !=
          kVrpoPpoContinuationFinalActorSteps ||
      !state.runtime_device_exact_match ||
      (!test_fixture && startup.registration_id !=
                            kVrpoPpoContinuationRegistrationId) ||
      (!test_fixture && startup.profile != kVrpoPpoContinuationProfile) ||
      (!test_fixture && startup.base_seed !=
                            kVrpoPpoContinuationBaseSeed) ||
      (!test_fixture && startup.start_episode_id !=
                            kVrpoPpoContinuationStartEpisodeId) ||
      VrpoQConstructorCalls() != 1 || VrpoQForwardCheckedCalls() != 0 ||
      VrpoScheduleQTargetComputations() != 0) {
    return deny("PPO-continuation terminal state/counters are incomplete");
  }
  std::string actor_hash;
  std::string optimizer_hash;
  std::string hash_error;
  const auto& final_summary = state.committed_new_updates.back().GetObject();
  const auto final_actor =
      final_summary.find("live_actor_after_disposition_sha256");
  if (!vrpo_training_internal::ModuleValueSha256(
          actor, "", &actor_hash, &hash_error) ||
      final_actor == final_summary.end() || !final_actor->second.IsString() ||
      final_actor->second.GetString() != actor_hash ||
      !vrpo_ppo_pilot_internal::ActorOptimizerStateSha256(
          actor_optimizer, actor, kVrpoPpoContinuationFinalActorSteps,
          &optimizer_hash, &hash_error) ||
      optimizer_hash != state.latest_actor_optimizer_state_sha256) {
    return deny("PPO-continuation final actor/Adam state is invalid");
  }
  for (int update = kVrpoPpoContinuationFirstGlobalUpdate;
       update <= kVrpoPpoContinuationLastGlobalUpdate; ++update) {
    std::string update_error;
    if (!ValidateVrpoPpoContinuationCommittedUpdateDirectory(
            startup.output_root, update,
            state.committed_new_updates[
                update - kVrpoPpoContinuationFirstGlobalUpdate],
            &update_error)) {
      return deny(update_error);
    }
  }
  if (!test_fixture) {
    std::string evidence_error;
    if (!vrpo_ppo_continuation_internal::EvidenceStillExact(
            startup.evidence_root, state.evidence_observations,
            &evidence_error)) {
      return deny(evidence_error);
    }
  }
  if (denial_reason != nullptr) denial_reason->clear();
  return true;
}

inline bool WriteVrpoPpoContinuationGlobalResult(
    const VrpoPpoContinuationStartupConfig& startup,
    const VrpoPpoContinuationState& state, torch::nn::Module& actor,
    torch::optim::AdamW& actor_optimizer,
    const VrpoPpoContinuationDeadline& deadline, json::Object* result,
    std::string* error, bool test_fixture = false) {
  if (result == nullptr) {
    if (error != nullptr) *error = "null PPO-continuation global result";
    return false;
  }
  std::string authorization_denial;
  const bool raw128_design_authorized =
      VrpoPpoContinuationRaw128DesignAuthorization(
          startup, state, actor, actor_optimizer, &authorization_denial,
          test_fixture);
  const std::string classification = state.had_failure
      ? "INVALID"
      : raw128_design_authorized
          ? "VALID_EIGHT_TOTAL_UPDATE_PPO_HEALTH_PASS"
          : state.stopped ? "VALID_EARLY_STOP_HEALTH" : "INVALID";
  const std::string reason = state.had_failure
      ? state.failure_reason
      : raw128_design_authorized
          ? "exact prior-four plus new-four chain, health, reload, Q, and Adam gates passed"
          : state.stopped
              ? "predeclared health gate failed; candidate retained and live actor/Adam rolled back"
              : authorization_denial;
  if (!deadline.Check("before continuation global status", error)) {
    return false;
  }
  const int64_t retained =
      vrpo_ppo_pilot_internal::DirectoryRegularBytes(startup.output_root);
  if (retained < 0 || retained > kVrpoPpoContinuationRetainedByteCeiling) {
    if (error != nullptr) *error = "PPO-continuation retained bytes exceed 8 GiB";
    return false;
  }
  json::Object root;
  root["schema"] = kVrpoPpoContinuationResultSchema;
  root["classification"] = classification;
  root["status"] = classification == "INVALID" ? "INVALID" : "VALID";
  root["reason"] = reason;
  root["registration_id"] = startup.registration_id;
  root["compiled_registration_id"] =
      kVrpoPpoContinuationRegistrationId;
  root["profile"] = startup.profile;
  root["compiled_profile"] = kVrpoPpoContinuationProfile;
  root["purpose"] = "BOUNDED_ACTOR_ONLY_PPO_CONTINUATION_HEALTH";
  root["source_code_sha256"] = startup.source_code_sha256;
  root["executed_binary_sha256"] = startup.executed_binary_sha256;
  root["executed_binary_size"] = startup.executed_binary_size;
  root["immutable_prior_evidence"] = json::Value(
      VrpoPpoPilotEvidenceObservationsJson(state.evidence_observations));
  root["immutable_prior_evidence_count"] =
      static_cast<int64_t>(state.evidence_observations.size());
  root["learning_rate"] = startup.learning_rate;
  root["learning_rate_exact"] =
      VrpoPhase4eCanonicalDouble(startup.learning_rate);
  root["actor_epochs_per_update"] =
      int64_t{kVrpoPpoContinuationActorEpochs};
  root["games_per_update"] =
      int64_t{kVrpoPpoContinuationGamesPerUpdate};
  root["prior_completed_updates"] = int64_t{4};
  root["attempted_new_updates"] =
      static_cast<int64_t>(state.attempted_new_updates);
  root["accepted_new_updates"] =
      static_cast<int64_t>(state.accepted_new_updates);
  root["total_update_summaries"] = static_cast<int64_t>(
      state.prior_update_summaries.size() +
      state.committed_new_updates.size());
  root["actor_optimizer_steps_total"] = state.live_actor_optimizer_steps;
  root["actor_optimizer_step_ceiling"] =
      kVrpoPpoContinuationFinalActorSteps;
  root["q_role"] = kVrpoPpoContinuationQRole;
  root["q_constructor_calls"] = VrpoQConstructorCalls();
  root["q_forward_calls"] = VrpoQForwardCheckedCalls();
  root["q_target_computations"] = VrpoScheduleQTargetComputations();
  root["q_optimizer_constructed"] = false;
  root["q_optimizer_steps"] = int64_t{0};
  root["source_q_values_sha256"] = state.source_q_values_sha256;
  root["runtime_device_exact_match"] = state.runtime_device_exact_match;
  root["prior_actor_values_sha256"] = state.prior_actor_values_sha256;
  root["prior_actor_optimizer_state_sha256"] =
      state.prior_actor_optimizer_state_sha256;
  root["latest_actor_optimizer_state_sha256"] =
      state.latest_actor_optimizer_state_sha256;
  root["frozen_update1_corpus_sha256"] =
      state.frozen_update1_corpus_sha256;
  root["fresh_actor_optimizer_at_start"] = false;
  root["prior_pilot_actor_optimizer_loaded"] = true;
  root["source_archive_optimizer_moments_loaded"] = false;
  root["optimizer_persisted_across_updates_1_through_8"] =
      raw128_design_authorized;
  root["optimizer_persisted_through_last_accepted_update"] =
      !state.latest_actor_optimizer_state_sha256.empty() &&
      state.live_actor_optimizer_steps >=
          kVrpoPpoContinuationPriorActorSteps;
  root["start_episode_id"] =
      static_cast<int64_t>(startup.start_episode_id);
  root["compiled_base_seed"] =
      static_cast<int64_t>(kVrpoPpoContinuationBaseSeed);
  root["runtime_base_seed"] = static_cast<int64_t>(startup.base_seed);
  root["compiled_start_episode_id"] =
      static_cast<int64_t>(kVrpoPpoContinuationStartEpisodeId);
  root["compiled_end_episode_id_inclusive"] =
      static_cast<int64_t>(kVrpoPpoContinuationEndEpisodeIdInclusive);
  json::Array ranges;
  for (int update = kVrpoPpoContinuationFirstGlobalUpdate;
       update <= kVrpoPpoContinuationLastGlobalUpdate; ++update) {
    const uint64_t begin = kVrpoPpoContinuationStartEpisodeId +
        static_cast<uint64_t>(update - kVrpoPpoContinuationFirstGlobalUpdate) *
            kVrpoPpoContinuationGamesPerUpdate;
    json::Object range;
    range["global_update"] = static_cast<int64_t>(update);
    range["start_episode_id"] = static_cast<int64_t>(begin);
    range["end_episode_id_inclusive"] = static_cast<int64_t>(
        begin + kVrpoPpoContinuationGamesPerUpdate - 1);
    ranges.emplace_back(std::move(range));
  }
  root["compiled_update_episode_ranges"] = std::move(ranges);
  root["next_episode_id"] = static_cast<int64_t>(
      startup.start_episode_id +
      state.attempted_new_updates * kVrpoPpoContinuationGamesPerUpdate);
  json::Array all_updates = state.prior_update_summaries;
  for (const auto& item : state.committed_new_updates) {
    all_updates.emplace_back(item);
  }
  root["available_update_summaries"] = all_updates;
  if (raw128_design_authorized && all_updates.size() == 8) {
    root["all_eight_update_summaries"] = std::move(all_updates);
  }
  root["new_updates"] = state.committed_new_updates;
  root["retained_byte_ceiling"] = kVrpoPpoContinuationRetainedByteCeiling;
  root["peak_byte_ceiling"] = kVrpoPpoContinuationPeakByteCeiling;
  root["measured_peak_bytes"] = state.measured_peak_bytes;
  root["retained_bytes_before_status"] = retained;
  root["wall_clock_ceiling_seconds"] = int64_t{1800};
  root["elapsed_seconds"] = deadline.ElapsedSeconds();
  root["status_last"] = true;
  root["authorization_denial_reason"] = authorization_denial;
  root["raw128_design_authorized"] = raw128_design_authorized;
  root["authorizes_fresh_raw_128_design_only"] =
      raw128_design_authorized;
  root["raw_evaluation_launch_authorized"] = false;
  root["authorizes_longer_training"] = false;
  root["authorizes_vrpo"] = false;
  root["authorizes_promotion"] = false;
  const auto status_path = startup.output_root / "CONTINUATION_RESULT.json";
  if (!vrpo_ppo_pilot_internal::WriteAtomicText(
          status_path, json::ToString(root, true) + "\n", error)) {
    return false;
  }
  const int64_t retained_after =
      vrpo_ppo_pilot_internal::DirectoryRegularBytes(startup.output_root);
  if (retained_after < 0 ||
      retained_after > kVrpoPpoContinuationRetainedByteCeiling ||
      retained_after > kVrpoPpoContinuationPeakByteCeiling ||
      !deadline.Check("after continuation global status", error)) {
    std::error_code ec;
    std::filesystem::remove(status_path, ec);
    std::string ignored;
    VrpoFsyncDirectory(startup.output_root, &ignored);
    if (error != nullptr && error->empty()) {
      *error = "PPO-continuation final resource/deadline gate failed";
    }
    return false;
  }
  *result = std::move(root);
  return true;
}

}  // namespace open_spiel

#endif  // OPEN_SPIEL_BUILD_WITH_LIBTORCH
#endif  // OPEN_SPIEL_EXAMPLES_DUNE_VRPO_PPO_CONTINUATION_H_
