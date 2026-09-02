#ifndef OPEN_SPIEL_EXAMPLES_DUNE_VRPO_PPO_PILOT_H_
#define OPEN_SPIEL_EXAMPLES_DUNE_VRPO_PPO_PILOT_H_

// Finite actor-only PPO continuation pilot selected after the independent
// ultra-low learning-rate schedule screen.  This path is intentionally
// separate from phase4e, the schedule screen, and the ordinary PPO loop.

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "dune_vrpo_schedule_screen.h"

#ifdef OPEN_SPIEL_BUILD_WITH_LIBTORCH
#include <torch/torch.h>

namespace open_spiel {

inline constexpr char kVrpoPpoPilotProfile[] = "ppo_u3_4x16_v1";
inline constexpr char kVrpoPpoPilotRegistrationId[] =
    "VRPO_U15828_PPO_CAP10_LR5E6_PILOT4X16_20260902";
inline constexpr uint64_t kVrpoPpoPilotBaseSeed = 8304001;
inline constexpr uint64_t kVrpoPpoPilotStartEpisodeId = 1200200000;
inline constexpr uint64_t kVrpoPpoPilotEndEpisodeIdInclusive = 1200200063;
inline constexpr char kVrpoPpoPilotResultSchema[] =
    "dune_vrpo_ppo_pilot_result_v1";
inline constexpr char kVrpoPpoPilotUpdateSchema[] =
    "dune_vrpo_ppo_pilot_update_v1";
inline constexpr char kVrpoPpoPilotEvidenceSchema[] =
    "dune_vrpo_ppo_pilot_episode_evidence_v1";
inline constexpr char kVrpoPpoPilotChecksumsSchema[] =
    "dune_vrpo_ppo_pilot_checksums_v1";
inline constexpr char kVrpoPpoPilotQRole[] =
    "SOURCE_ARCHIVE_PROVENANCE_ONLY";
inline constexpr int kVrpoPpoPilotUpdates = 4;
inline constexpr int kVrpoPpoPilotGamesPerUpdate = 16;
inline constexpr int kVrpoPpoPilotActorEpochs = 1;
inline constexpr int64_t kVrpoPpoPilotStepsPerUpdate = 16;
inline constexpr int64_t kVrpoPpoPilotMaxActorSteps = 64;
inline constexpr double kVrpoPpoPilotLearningRate = 5.0e-6;
inline constexpr char kVrpoPpoPilotOriginArchiveSha256[] =
    "0be841acda5692165c4991f5352ebcfe79865aec985c54e8b82144a333bbbef7";
inline constexpr char kVrpoPpoPilotV3ScheduleRegistrationSha256[] =
    "3dc45c69ff22d2eb93de82be98fa6568abda65726e8d4f531e1eb708a91a18f7";
inline constexpr char kVrpoPpoPilotV3ScheduleResultSha256[] =
    "da3e251ad542b17a519aba8bce56426b509cfec58d5f0b7ef5183f2c8ae24369";
inline constexpr char kVrpoPpoPilotV3ScheduleValidationSha256[] =
    "ec6f4247cc709580da3cefd4e49aabae643b7acd68d38a27eea24436416b1078";
inline constexpr char kVrpoPpoPilotV3ScheduleCorpusManifestSha256[] =
    "f9386300910c8b4cb41cb206e5a1584789725fb9241efc9236b7cc3c35b39689";
inline constexpr char kVrpoPpoPilotV3ScheduleCorpusSha256[] =
    "64f5ddca0da6650922c33583071ebed3bd266896b32edb80969573f655e41a23";
inline constexpr char kVrpoPpoPilotSelectedScreenManifestSha256[] =
    "520257e587ae1fd88adc210eacd4d516461497e5ab61cc0af19da44de8438b78";
inline constexpr char kVrpoPpoPilotSelectedScreenResultSha256[] =
    "46b0ffbfe9f8f89ff9c28f670dc1f99c9ca77796536ac56afe8e667b44a3da96";
inline constexpr char kVrpoPpoPilotSelectedScreenValidationSha256[] =
    "98e57b2cc9d5f56d20b52fb7f8084d91fed67bbecb5155122db66dff005c9ba7";
inline constexpr char kVrpoPpoPilotConfirmManifestSha256[] =
    "ea16ee9c3f45e30dfe011360370363594a41a5479708ab3872426d84bbd26da5";
inline constexpr char kVrpoPpoPilotConfirmResultSha256[] =
    "229e45be50e76556e4b07549784ea5f346d1bfdc2d85d62e84f24356f05e5ff2";
inline constexpr char kVrpoPpoPilotConfirmValidationSha256[] =
    "0afb3704f88e9bd65866941ca2e3d17534321ded43b77f8ef586f46f946a0600";
inline constexpr int64_t kVrpoPpoPilotRetainedByteCeiling =
    8LL * 1024 * 1024 * 1024;
inline constexpr int64_t kVrpoPpoPilotPeakByteCeiling =
    10LL * 1024 * 1024 * 1024;

class VrpoPpoPilotDeadline {
 public:
  static VrpoPpoPilotDeadline Start(
      std::chrono::steady_clock::time_point start,
      std::chrono::seconds limit = std::chrono::seconds(1800)) {
    VrpoPpoPilotDeadline out;
    out.active_ = true;
    out.start_ = start;
    out.deadline_ = start + limit;
    return out;
  }

  static VrpoPpoPilotDeadline ExpireAfterChecksForTest(int checks) {
    VrpoPpoPilotDeadline out = Start(
        std::chrono::steady_clock::now(), std::chrono::hours(1));
    out.test_checks_remaining_ = checks;
    return out;
  }

  bool Check(const std::string& stage, std::string* error) const {
    if (test_checks_remaining_ >= 0) {
      if (test_checks_remaining_ == 0) {
        if (error != nullptr) {
          *error = "injected PPO-pilot deadline exceeded at " + stage;
        }
        return false;
      }
      --test_checks_remaining_;
    }
    if (!active_ || std::chrono::steady_clock::now() < deadline_) return true;
    if (error != nullptr) {
      *error = "PPO-pilot 1800-second deadline exceeded at " + stage;
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

struct VrpoPpoPilotEvidenceFileSpec {
  std::string identity;
  std::filesystem::path relative_path;
  std::string expected_sha256;
};

inline std::vector<VrpoPpoPilotEvidenceFileSpec>
VrpoPpoPilotCanonicalEvidenceFiles() {
  return {
      {"v3_schedule_registration",
       "registrations/schedule_screen_ultralow_lr_v3_u15828_20260902/registration.json",
       kVrpoPpoPilotV3ScheduleRegistrationSha256},
      {"v3_schedule_result",
       "schedule_screen_ultralow_lr_v3_u15828_20260902/SCREEN_RESULT.json",
       kVrpoPpoPilotV3ScheduleResultSha256},
      {"v3_schedule_validation",
       "registrations/schedule_screen_ultralow_lr_v3_u15828_20260902/validation.json",
       kVrpoPpoPilotV3ScheduleValidationSha256},
      {"v3_schedule_corpus_manifest",
       "schedule_screen_ultralow_lr_v3_u15828_20260902/CORPUS_MANIFEST.json",
       kVrpoPpoPilotV3ScheduleCorpusManifestSha256},
      {"v3_schedule_corpus",
       "schedule_screen_ultralow_lr_v3_u15828_20260902/ACTOR_CORPUS.bin",
       kVrpoPpoPilotV3ScheduleCorpusSha256},
      {"selected_screen_manifest",
       "raw_strength_screen_schedule_v3_selected_u15828_128_20260902/SCREEN_MANIFEST.json",
       kVrpoPpoPilotSelectedScreenManifestSha256},
      {"selected_screen_result",
       "raw_strength_screen_schedule_v3_selected_u15828_128_20260902/SCREEN_RESULT.json",
       kVrpoPpoPilotSelectedScreenResultSha256},
      {"selected_screen_validation",
       "raw_strength_screen_schedule_v3_selected_u15828_128_20260902/validation.json",
       kVrpoPpoPilotSelectedScreenValidationSha256},
      {"confirm_manifest",
       "raw_strength_confirm_schedule_v3_u3_u15828_256_20260902/CONFIRM_MANIFEST.json",
       kVrpoPpoPilotConfirmManifestSha256},
      {"confirm_result",
       "raw_strength_confirm_schedule_v3_u3_u15828_256_20260902/CONFIRM_RESULT.json",
       kVrpoPpoPilotConfirmResultSha256},
      {"confirm_validation",
       "raw_strength_confirm_schedule_v3_u3_u15828_256_20260902/validation.json",
       kVrpoPpoPilotConfirmValidationSha256},
  };
}

struct VrpoPpoPilotEvidenceObservation {
  std::string identity;
  std::string relative_path;
  std::string compiled_sha256;
  std::string cli_sha256;
  std::string observed_sha256;
  int64_t observed_size = 0;
  bool matched = false;
};

inline bool ValidateVrpoPpoPilotEvidenceFiles(
    const std::filesystem::path& evidence_root,
    const std::vector<VrpoPpoPilotEvidenceFileSpec>& specs,
    const std::vector<std::string>& cli_sha256,
    std::vector<VrpoPpoPilotEvidenceObservation>* observations,
    std::string* error) {
  auto fail = [&](const std::string& message) {
    if (error != nullptr) *error = message;
    if (observations != nullptr) observations->clear();
    return false;
  };
  if (observations == nullptr || specs.empty() ||
      specs.size() != cli_sha256.size()) {
    return fail("PPO-pilot immutable evidence arguments are invalid");
  }
  std::error_code ec;
  const auto canonical_root = std::filesystem::canonical(evidence_root, ec);
  if (ec || !std::filesystem::is_directory(canonical_root)) {
    return fail("PPO-pilot immutable evidence root is missing");
  }
  std::set<std::string> identities;
  std::set<std::string> paths;
  std::vector<VrpoPpoPilotEvidenceObservation> observed;
  for (size_t index = 0; index < specs.size(); ++index) {
    const auto& spec = specs[index];
    if (spec.identity.empty() || spec.relative_path.empty() ||
        spec.relative_path.is_absolute() ||
        spec.relative_path.string().find("..") != std::string::npos ||
        !identities.insert(spec.identity).second ||
        !paths.insert(spec.relative_path.string()).second ||
        !VrpoPhase4eLowerHex64(spec.expected_sha256) ||
        cli_sha256[index] != spec.expected_sha256) {
      return fail("PPO-pilot immutable evidence CLI/compiled identity mismatch at " +
                  spec.identity);
    }
    const auto path = canonical_root / spec.relative_path;
    if (!std::filesystem::is_regular_file(path) ||
        std::filesystem::is_symlink(path)) {
      return fail("PPO-pilot immutable evidence file missing/nonregular at " +
                  spec.identity);
    }
    size_t size = 0;
    const std::string digest = ComputeFileSHA256(path.string(), &size);
    if (size == 0 || digest != spec.expected_sha256) {
      return fail("PPO-pilot immutable evidence observed digest mismatch at " +
                  spec.identity);
    }
    observed.push_back({spec.identity, spec.relative_path.string(),
                        spec.expected_sha256, cli_sha256[index], digest,
                        static_cast<int64_t>(size), true});
  }
  *observations = std::move(observed);
  return true;
}

inline json::Array VrpoPpoPilotEvidenceObservationsJson(
    const std::vector<VrpoPpoPilotEvidenceObservation>& observations) {
  json::Array out;
  for (const auto& item : observations) {
    json::Object record;
    record["identity"] = item.identity;
    record["relative_path"] = item.relative_path;
    record["compiled_sha256"] = item.compiled_sha256;
    record["cli_sha256"] = item.cli_sha256;
    record["observed_sha256"] = item.observed_sha256;
    record["observed_size"] = item.observed_size;
    record["matched"] = item.matched;
    out.emplace_back(std::move(record));
  }
  return out;
}

struct VrpoPpoPilotStartupConfig {
  std::string game;
  std::string init_mode;
  std::string profile;
  std::string registration_id;
  std::filesystem::path input_archive;
  std::filesystem::path output_root;
  std::filesystem::path source_root;
  std::string source_code_sha256;
  std::string executed_binary_sha256;
  int64_t executed_binary_size = 0;
  std::string origin_archive_sha256;
  std::filesystem::path evidence_root;
  std::vector<std::string> evidence_cli_sha256;
  std::vector<VrpoPpoPilotEvidenceObservation> evidence_observations;
  int rollout_games = 0;
  int threads = 0;
  int eval_batch_size = 0;
  int eval_timeout_ms = 0;
  bool evaluator_device_synchronize = false;
  bool deterministic_rollout_eval = false;
  int seed_scheme_version = 0;
  bool runtime_device_is_cuda = false;
  int runtime_device_index = -1;
  bool one_gpu_process = false;
  int64_t runtime_process_id = 0;
  uint64_t base_seed = 0;
  uint64_t start_episode_id = 0;
  bool diagnostics_only = false;
  bool rollout_amp = false;
  bool train_amp = false;
  bool allow_tf32 = false;
  bool pipeline = false;
  bool online_search_collection = false;
  bool search_pi_mode = false;
  bool train_value_only = false;
  bool sample_counterfactual_states = false;
  bool has_search_label_dir = false;
  bool ordinary_checkpoint_writes_enabled = false;
  bool anneal_lr = false;
  double learning_rate = 0.0;
  int ppo_update_epochs = 0;
  double shaped_reward_weight = 0.0;
  double tleilaxu_breadcrumb_weight = 0.0;
  double tleilaxu_level7_breadcrumb_weight = 0.0;
  double specimen_exchange_penalty = 0.0;
  double reward_scale = 0.0;
  double gamma = 0.0;
  double lambda = 0.0;
  double logit_cap = 0.0;
  int ppo_minibatches = 0;
  int ppo_minibatch_size = 0;
  double clip_epsilon = 0.0;
  double entropy_coefficient = 0.0;
  double value_coefficient = 0.0;
  double gradient_clip_norm = 0.0;
  bool normalize_advantages = false;
  bool clip_value_loss = false;
};

inline std::vector<std::string> VrpoPpoPilotSourceRelativePaths() {
  std::vector<std::string> paths = VrpoScheduleSourceRelativePaths();
  paths.push_back("open_spiel/examples/dune_vrpo_ppo_pilot.h");
  return paths;
}

inline bool LoadVrpoPpoPilotSourceIdentity(
    const std::filesystem::path& source_root,
    const std::string& registered_sha256, VrpoPhase4eSourceIdentity* output,
    std::string* error) {
  auto fail = [&](const std::string& message) {
    if (error != nullptr) *error = message;
    if (output != nullptr) *output = VrpoPhase4eSourceIdentity{};
    return false;
  };
  if (output == nullptr || !VrpoPhase4eLowerHex64(registered_sha256)) {
    return fail("PPO-pilot source identity/registration is invalid");
  }
  std::error_code ec;
  const auto canonical_root = std::filesystem::canonical(source_root, ec);
  if (ec || !std::filesystem::is_directory(canonical_root)) {
    return fail("PPO-pilot source root is not a readable directory");
  }
  const auto paths = VrpoPpoPilotSourceRelativePaths();
  std::set<std::string> seen;
  std::string payload;
  for (const auto& relative : paths) {
    if (relative.empty() || std::filesystem::path(relative).is_absolute() ||
        relative.find("..") != std::string::npos ||
        !seen.insert(relative).second) {
      return fail("PPO-pilot source list is noncanonical or duplicated");
    }
    const auto path = canonical_root / relative;
    if (!std::filesystem::is_regular_file(path) ||
        std::filesystem::is_symlink(path)) {
      return fail("PPO-pilot source file is missing/nonregular: " + relative);
    }
    size_t size = 0;
    const std::string digest = ComputeFileSHA256(path.string(), &size);
    if (!VrpoPhase4eLowerHex64(digest) || size == 0) {
      return fail("PPO-pilot source file could not be hashed: " + relative);
    }
    payload.append(relative);
    payload.push_back('\0');
    payload.append(digest);
    payload.push_back('\n');
  }
  const std::string observed = ComputeStringSHA256(payload);
  if (observed != registered_sha256) {
    return fail("PPO-pilot registered source SHA-256 mismatch");
  }
  output->canonical_root = canonical_root;
  output->relative_paths = paths;
  output->combined_sha256 = observed;
  return true;
}

inline bool ValidateVrpoPpoPilotStartupConfig(
    const VrpoPpoPilotStartupConfig& config, std::string* error,
    bool test_fixture = false) {
  auto fail = [&](const std::string& message) {
    if (error != nullptr) *error = message;
    return false;
  };
  if (config.game != "dune_imperium" ||
      config.init_mode != "vrpo_ppo_pilot" ||
      config.profile != kVrpoPpoPilotProfile ||
      (!test_fixture &&
       config.registration_id != kVrpoPpoPilotRegistrationId) ||
      (test_fixture && config.registration_id.empty()) ||
      config.input_archive.empty() ||
      config.output_root.empty() || config.source_root.empty() ||
      !VrpoPhase4eLowerHex64(config.source_code_sha256) ||
      !VrpoPhase4eLowerHex64(config.executed_binary_sha256) ||
      config.executed_binary_size <= 0 ||
      !VrpoPhase4eLowerHex64(config.origin_archive_sha256) ||
      config.evidence_root.empty()) {
    return fail("PPO-pilot identity/prior-evidence contract is invalid");
  }
  const auto canonical_evidence = VrpoPpoPilotCanonicalEvidenceFiles();
  if (!test_fixture) {
    if (config.origin_archive_sha256 !=
            kVrpoPpoPilotOriginArchiveSha256 ||
        config.evidence_cli_sha256.size() != canonical_evidence.size() ||
        config.evidence_observations.size() != canonical_evidence.size()) {
      return fail("PPO-pilot compiled immutable evidence chain is incomplete");
    }
    for (size_t index = 0; index < canonical_evidence.size(); ++index) {
      const auto& expected = canonical_evidence[index];
      const auto& observed = config.evidence_observations[index];
      if (config.evidence_cli_sha256[index] != expected.expected_sha256 ||
          observed.identity != expected.identity ||
          observed.relative_path != expected.relative_path.string() ||
          observed.compiled_sha256 != expected.expected_sha256 ||
          observed.cli_sha256 != expected.expected_sha256 ||
          observed.observed_sha256 != expected.expected_sha256 ||
          observed.observed_size <= 0 || !observed.matched) {
        return fail("PPO-pilot compiled/CLI/observed evidence mismatch at " +
                    expected.identity);
      }
    }
  } else if (!VrpoPhase4eLowerHex64(config.origin_archive_sha256)) {
    return fail("PPO-pilot test-fixture origin identity is malformed");
  }
  VrpoPhase4eResolvedPaths paths;
  if (!ResolveVrpoPhase4ePaths(config.input_archive, config.output_root,
                               &paths, error)) {
    return false;
  }
  if (std::filesystem::exists(paths.output_root)) {
    return fail("PPO-pilot output root must be fresh and absent");
  }
  if (config.rollout_games != kVrpoPpoPilotGamesPerUpdate ||
      config.threads != 8 || config.eval_batch_size != 64 ||
      config.eval_timeout_ms != 2 ||
      !config.evaluator_device_synchronize ||
      config.deterministic_rollout_eval || config.seed_scheme_version != 2 ||
      !config.runtime_device_is_cuda || config.runtime_device_index < 0 ||
      !config.one_gpu_process || config.runtime_process_id <= 0 ||
      (!test_fixture && config.base_seed != kVrpoPpoPilotBaseSeed) ||
      (!test_fixture &&
       config.start_episode_id != kVrpoPpoPilotStartEpisodeId) ||
      (test_fixture && config.base_seed == 0) ||
      config.start_episode_id == 0 ||
      config.start_episode_id >
          std::numeric_limits<uint64_t>::max() -
              kVrpoPpoPilotUpdates * kVrpoPpoPilotGamesPerUpdate ||
      config.start_episode_id >
          static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) -
              kVrpoPpoPilotUpdates * kVrpoPpoPilotGamesPerUpdate ||
      config.diagnostics_only || config.rollout_amp || config.train_amp ||
      !config.allow_tf32 || config.pipeline ||
      config.online_search_collection || config.search_pi_mode ||
      config.train_value_only || config.sample_counterfactual_states ||
      config.has_search_label_dir ||
      config.ordinary_checkpoint_writes_enabled || config.anneal_lr) {
    return fail("PPO-pilot execution/side-path contract is invalid");
  }
  for (double shaping :
       {config.shaped_reward_weight, config.tleilaxu_breadcrumb_weight,
        config.tleilaxu_level7_breadcrumb_weight,
        config.specimen_exchange_penalty}) {
    if (!std::isfinite(shaping) || shaping != 0.0) {
      return fail("PPO-pilot shaping must be exactly zero");
    }
  }
  if (config.learning_rate != kVrpoPpoPilotLearningRate ||
      config.ppo_update_epochs != kVrpoPpoPilotActorEpochs ||
      config.reward_scale != 4.0 || config.gamma != 1.0 ||
      config.lambda != 1.0 || config.logit_cap != 10.0 ||
      config.ppo_minibatches != kVrpoTrainingMinibatchesPerEpoch ||
      config.ppo_minibatch_size != 2048 ||
      config.clip_epsilon != kVrpoTrainingClipEpsilon ||
      config.entropy_coefficient != kVrpoTrainingEntropyCoefficient ||
      config.value_coefficient != kVrpoTrainingValueCoefficient ||
      config.gradient_clip_norm != kVrpoTrainingGradientClipNorm ||
      !config.normalize_advantages || !config.clip_value_loss ||
      kVrpoPpoPilotStepsPerUpdate !=
          kVrpoPpoPilotActorEpochs * kVrpoTrainingMinibatchesPerEpoch ||
      kVrpoPpoPilotMaxActorSteps !=
          kVrpoPpoPilotUpdates * kVrpoPpoPilotStepsPerUpdate ||
      (!test_fixture &&
       config.start_episode_id +
               kVrpoPpoPilotUpdates * kVrpoPpoPilotGamesPerUpdate - 1 !=
           kVrpoPpoPilotEndEpisodeIdInclusive)) {
    return fail("PPO-pilot mechanics/step ceiling differs from registration");
  }
  return true;
}

namespace vrpo_ppo_pilot_internal {

inline int64_t DirectoryRegularBytes(const std::filesystem::path& root) {
  int64_t total = 0;
  std::error_code ec;
  for (std::filesystem::recursive_directory_iterator it(root, ec), end;
       !ec && it != end; it.increment(ec)) {
    if (it->is_regular_file(ec) && !it->is_symlink(ec)) {
      const auto size = it->file_size(ec);
      if (ec || size > static_cast<uintmax_t>(
                           std::numeric_limits<int64_t>::max() - total)) {
        return -1;
      }
      total += static_cast<int64_t>(size);
    }
  }
  return ec ? -1 : total;
}

inline bool CheckFreeSpaceBeforeStart(const std::filesystem::path& output_root,
                                      std::string* error) {
  std::error_code ec;
  std::filesystem::path probe = output_root.parent_path();
  while (!probe.empty() && !std::filesystem::exists(probe, ec)) {
    if (ec) break;
    probe = probe.parent_path();
  }
  if (ec || probe.empty()) {
    if (error != nullptr) *error = "cannot resolve PPO-pilot output filesystem";
    return false;
  }
  const auto space = std::filesystem::space(probe, ec);
  if (ec || space.available <
                static_cast<uintmax_t>(kVrpoPpoPilotPeakByteCeiling)) {
    if (error != nullptr) {
      *error = "insufficient free space for PPO-pilot 10-GiB peak ceiling";
    }
    return false;
  }
  return true;
}

inline void CleanupPath(const std::filesystem::path& path) {
  std::error_code ec;
  if (!path.empty()) std::filesystem::remove_all(path, ec);
}

inline bool ObserveBytes(const std::filesystem::path& root,
                         int64_t* measured_peak_bytes, std::string* error) {
  const int64_t observed = DirectoryRegularBytes(root);
  if (observed < 0 || measured_peak_bytes == nullptr) {
    if (error != nullptr) *error = "PPO-pilot byte accounting failed";
    return false;
  }
  *measured_peak_bytes = std::max(*measured_peak_bytes, observed);
  if (observed > kVrpoPpoPilotPeakByteCeiling) {
    if (error != nullptr) *error = "PPO-pilot peak bytes exceeded 10 GiB";
    return false;
  }
  return true;
}

inline bool WriteAtomicText(const std::filesystem::path& path,
                            const std::string& contents,
                            std::string* error) {
  const auto temp = std::filesystem::path(path.string() + ".tmp");
  if (std::filesystem::exists(path) || std::filesystem::exists(temp)) {
    if (error != nullptr) *error = "PPO-pilot atomic output already exists";
    return false;
  }
  {
    std::ofstream stream(temp, std::ios::binary | std::ios::trunc);
    stream.write(contents.data(), contents.size());
    stream.flush();
    if (!stream) {
      if (error != nullptr) *error = "PPO-pilot atomic output write failed";
      return false;
    }
  }
  if (!VrpoFsyncFile(temp, error)) return false;
  std::filesystem::rename(temp, path);
  return VrpoFsyncDirectory(path.parent_path(), error);
}

inline bool ActorOptimizerStateSha256(
    torch::optim::AdamW& optimizer, torch::nn::Module& actor,
    int64_t expected_step, std::string* output, std::string* error) {
  auto fail = [&](const std::string& message) {
    if (error != nullptr) *error = message;
    if (output != nullptr) output->clear();
    return false;
  };
  if (output == nullptr || expected_step < 0 ||
      !ValidateVrpoScheduleActorOptimizerCoverage(optimizer, actor, error) ||
      optimizer.state().size() != actor.named_parameters().size()) {
    return fail(error != nullptr && !error->empty()
                    ? *error
                    : "PPO-pilot actor optimizer coverage/state is invalid");
  }
  std::map<c10::TensorImpl*, std::string> names;
  for (const auto& item : actor.named_parameters()) {
    names[item.value().unsafeGetTensorImpl()] = item.key();
  }
  std::string payload = "dune_vrpo_ppo_pilot_adamw_state_v1";
  vrpo_capture_internal::AppendPod(
      &payload, static_cast<uint64_t>(names.size()));
  std::set<c10::TensorImpl*> seen;
  for (const auto& group : optimizer.param_groups()) {
    const auto& options = static_cast<const torch::optim::AdamWOptions&>(
        group.options());
    if (options.lr() != kVrpoPpoPilotLearningRate ||
        options.weight_decay() != 0.0) {
      return fail("PPO-pilot actor optimizer options changed");
    }
    for (const auto& parameter : group.params()) {
      c10::TensorImpl* key = parameter.unsafeGetTensorImpl();
      const auto name = names.find(key);
      const auto state_it = optimizer.state().find(key);
      const auto* state = state_it == optimizer.state().end()
          ? nullptr
          : dynamic_cast<const torch::optim::AdamWParamState*>(
                state_it->second.get());
      if (name == names.end() || !seen.insert(key).second || state == nullptr ||
          state->step() != expected_step || !state->exp_avg().defined() ||
          !state->exp_avg_sq().defined() ||
          state->exp_avg().sizes() != parameter.sizes() ||
          state->exp_avg_sq().sizes() != parameter.sizes() ||
          state->exp_avg().device() != parameter.device() ||
          state->exp_avg_sq().device() != parameter.device() ||
          state->exp_avg().scalar_type() != parameter.scalar_type() ||
          state->exp_avg_sq().scalar_type() != parameter.scalar_type() ||
          !torch::isfinite(state->exp_avg()).all().item<bool>() ||
          !torch::isfinite(state->exp_avg_sq()).all().item<bool>()) {
        return fail("PPO-pilot Adam state/device/dtype/step is invalid");
      }
      vrpo_schedule_internal::AppendString(&payload, name->second);
      vrpo_capture_internal::AppendPod(&payload, state->step());
      for (const torch::Tensor& moment :
           {state->exp_avg(), state->exp_avg_sq()}) {
        const torch::Tensor cpu =
            moment.detach().contiguous().cpu().to(torch::kFloat32);
        vrpo_capture_internal::AppendPod(&payload, cpu.numel());
        payload.append(reinterpret_cast<const char*>(cpu.data_ptr<float>()),
                       cpu.numel() * sizeof(float));
      }
    }
  }
  if (seen.size() != names.size()) {
    return fail("PPO-pilot actor optimizer omitted a parameter");
  }
  *output = ComputeStringSHA256(payload);
  return true;
}

inline bool CurrentHealthPass(const VrpoScheduleHealthMetrics& health) {
  return health.kl_mean <= 0.05 && health.kl_p99 <= 0.5 &&
      health.clip_fraction <= 0.20 && health.ratio_p01 >= 0.5 &&
      health.ratio_p99 <= 2.0 &&
      health.greedy_argmax_change_rate >= 0.005 &&
      health.greedy_argmax_change_rate <= 0.15;
}

inline bool CumulativeHealthPass(const VrpoScheduleHealthMetrics& health) {
  return health.kl_mean <= 0.05 && health.kl_p99 <= 0.5 &&
      health.ratio_p01 >= 0.5 && health.ratio_p99 <= 2.0 &&
      health.greedy_argmax_change_rate <= 0.15;
}

inline json::Object FileIdentityJson(const std::filesystem::path& path) {
  size_t size = 0;
  json::Object out;
  out["filename"] = path.filename().string();
  out["sha256"] = ComputeFileSHA256(path.string(), &size);
  out["size"] = static_cast<int64_t>(size);
  return out;
}

}  // namespace vrpo_ppo_pilot_internal

struct VrpoPpoPilotState {
  int completed_updates = 0;
  int64_t total_actor_optimizer_steps = 0;
  int64_t measured_peak_bytes = 0;
  std::string initial_actor_values_sha256;
  std::string source_q_values_sha256;
  std::string initial_actor_optimizer_state_sha256;
  std::string latest_actor_optimizer_state_sha256;
  std::string frozen_update1_corpus_sha256;
  std::vector<VrpoTrainingEpisode> frozen_update1_episodes;
  std::vector<VrpoPpoPilotEvidenceObservation> evidence_observations;
  json::Array committed_updates;
  bool stopped = false;
  bool completed = false;
  bool had_failure = false;
  std::string failure_reason;
};

inline bool ValidateVrpoPpoPilotCommittedUpdateDirectory(
    const std::filesystem::path& output_root, int update_number,
    const json::Value& summary_value, std::string* error) {
  auto fail = [&](const std::string& message) {
    if (error != nullptr) *error = message;
    return false;
  };
  if (!summary_value.IsObject() || update_number < 1 || update_number > 4) {
    return fail("PPO-pilot committed update summary is malformed");
  }
  const auto& summary = summary_value.GetObject();
  const auto directory = output_root /
      ("update_" + std::to_string(update_number));
  const auto checksums_path = directory / "CHECKSUMS.json";
  const auto result_path = directory / "UPDATE_RESULT.json";
  std::ifstream checksum_stream(checksums_path, std::ios::binary);
  std::string checksum_text((std::istreambuf_iterator<char>(checksum_stream)),
                            std::istreambuf_iterator<char>());
  const auto checksums = json::FromString(checksum_text);
  if (!checksums.has_value() || !checksums->IsObject()) {
    return fail("PPO-pilot committed CHECKSUMS.json is missing/malformed");
  }
  const auto& checksum_object = checksums->GetObject();
  const auto schema = checksum_object.find("schema");
  const auto files = checksum_object.find("files");
  if (schema == checksum_object.end() || !schema->second.IsString() ||
      schema->second.GetString() != kVrpoPpoPilotChecksumsSchema ||
      files == checksum_object.end() || !files->second.IsArray() ||
      files->second.GetArray().size() != 5) {
    return fail("PPO-pilot committed checksum schema/file count is invalid");
  }
  const std::set<std::string> expected_files = {
      "ACTOR_CORPUS.bin", "EPISODE_EVIDENCE.json", "actor_model.pt",
      "actor_optimizer.pt", "UPDATE_RESULT.json"};
  std::set<std::string> seen;
  for (const auto& file_value : files->second.GetArray()) {
    if (!file_value.IsObject()) return fail("PPO-pilot checksum row is invalid");
    const auto& file = file_value.GetObject();
    const auto filename = file.find("filename");
    const auto sha = file.find("sha256");
    const auto size = file.find("size");
    if (filename == file.end() || !filename->second.IsString() ||
        sha == file.end() || !sha->second.IsString() ||
        size == file.end() || !size->second.IsInt() ||
        size->second.GetInt() <= 0 ||
        !expected_files.count(filename->second.GetString()) ||
        !seen.insert(filename->second.GetString()).second) {
      return fail("PPO-pilot checksum row fields are invalid");
    }
    const auto path = directory / filename->second.GetString();
    if (!std::filesystem::is_regular_file(path) ||
        std::filesystem::is_symlink(path)) {
      return fail("PPO-pilot committed file is missing/nonregular");
    }
    size_t observed_size = 0;
    const std::string observed =
        ComputeFileSHA256(path.string(), &observed_size);
    if (observed != sha->second.GetString() ||
        static_cast<int64_t>(observed_size) != size->second.GetInt()) {
      return fail("PPO-pilot committed file checksum drifted");
    }
  }
  if (seen != expected_files) {
    return fail("PPO-pilot committed checksum file set is incomplete");
  }
  std::ifstream result_stream(result_path, std::ios::binary);
  std::string result_text((std::istreambuf_iterator<char>(result_stream)),
                          std::istreambuf_iterator<char>());
  const auto parsed = json::FromString(result_text);
  if (!parsed.has_value() || !parsed->IsObject()) {
    return fail("PPO-pilot committed update result is malformed");
  }
  const auto& result = parsed->GetObject();
  auto require_int = [&](const std::string& key, int64_t expected) {
    const auto it = result.find(key);
    return it != result.end() && it->second.IsInt() &&
        it->second.GetInt() == expected;
  };
  auto require_bool = [&](const std::string& key) {
    const auto it = result.find(key);
    return it != result.end() && it->second.IsBool() &&
        it->second.GetBool();
  };
  const auto classification = result.find("classification");
  if (!require_int("update", update_number) ||
      !require_int("actor_optimizer_step_after", update_number * 16) ||
      !require_int("actor_optimizer_steps_this_update", 16) ||
      !require_bool("strict_reload_passed") ||
      !require_bool("canary_exact") || !require_bool("q_bit_inert") ||
      !require_bool("current_rollout_health_pass") ||
      !require_bool("cumulative_fixed_update1_health_pass") ||
      classification == result.end() || !classification->second.IsString() ||
      classification->second.GetString() != "VALID_UPDATE_HEALTH_PASS") {
    return fail("PPO-pilot committed update result is not promotion-valid");
  }
  const auto summary_update = summary.find("update");
  const auto summary_current = summary.find("current_rollout_health_pass");
  const auto summary_cumulative =
      summary.find("cumulative_fixed_update1_health_pass");
  const auto summary_rollback = summary.find("current_update_rolled_back");
  const auto summary_actor =
      summary.find("live_actor_after_disposition_sha256");
  const auto result_actor = result.find("actor_after_sha256");
  if (summary_update == summary.end() || !summary_update->second.IsInt() ||
      summary_update->second.GetInt() != update_number ||
      summary_current == summary.end() || !summary_current->second.IsBool() ||
      !summary_current->second.GetBool() ||
      summary_cumulative == summary.end() ||
      !summary_cumulative->second.IsBool() ||
      !summary_cumulative->second.GetBool() ||
      summary_rollback == summary.end() ||
      !summary_rollback->second.IsBool() ||
      summary_rollback->second.GetBool() ||
      summary_actor == summary.end() || !summary_actor->second.IsString() ||
      result_actor == result.end() || !result_actor->second.IsString() ||
      summary_actor->second.GetString() != result_actor->second.GetString()) {
    return fail("PPO-pilot committed summary is not promotion-valid");
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
    return fail("PPO-pilot committed summary/result identity mismatch");
  }
  return true;
}

inline bool VrpoPpoPilotRaw128Authorization(
    const VrpoPpoPilotStartupConfig& startup,
    const VrpoExpandedArchiveIdentity& input_identity,
    const VrpoExpandedExpectedLayout& layout,
    const VrpoPpoPilotState& state, torch::nn::Module& actor,
    torch::optim::AdamW& actor_optimizer, std::string* denial_reason) {
  const bool test_fixture = layout.test_fixture;
  auto deny = [&](const std::string& reason) {
    if (denial_reason != nullptr) *denial_reason = reason;
    return false;
  };
  if (!state.completed || state.stopped || state.had_failure ||
      state.completed_updates != kVrpoPpoPilotUpdates ||
      state.committed_updates.size() != kVrpoPpoPilotUpdates ||
      state.total_actor_optimizer_steps != kVrpoPpoPilotMaxActorSteps ||
      (!test_fixture && input_identity.combined_sha256 !=
                            kVrpoPpoPilotOriginArchiveSha256) ||
      (!test_fixture && startup.origin_archive_sha256 !=
                            kVrpoPpoPilotOriginArchiveSha256) ||
      (!test_fixture && startup.registration_id !=
                            kVrpoPpoPilotRegistrationId) ||
      (!test_fixture && startup.profile != kVrpoPpoPilotProfile) ||
      (!test_fixture && startup.base_seed != kVrpoPpoPilotBaseSeed) ||
      (!test_fixture && startup.start_episode_id !=
                            kVrpoPpoPilotStartEpisodeId) ||
      (!test_fixture &&
       (layout.label != "production_dune_vrpo_layout_v1" ||
        layout.actor_hidden_dim != 2048 ||
        layout.actor_residual_blocks != 8)) ||
      (!test_fixture && state.evidence_observations.size() !=
                            VrpoPpoPilotCanonicalEvidenceFiles().size()) ||
      VrpoQConstructorCalls() != 1 || VrpoQForwardCheckedCalls() != 0 ||
      VrpoScheduleQTargetComputations() != 0) {
    return deny("PPO-pilot terminal state/counters/provenance are incomplete");
  }
  const auto canonical_evidence = VrpoPpoPilotCanonicalEvidenceFiles();
  for (size_t index = 0; index < state.evidence_observations.size(); ++index) {
    const auto& observation = state.evidence_observations[index];
    if (!observation.matched ||
        observation.compiled_sha256 != observation.cli_sha256 ||
        observation.compiled_sha256 != observation.observed_sha256 ||
        (!test_fixture &&
         (index >= canonical_evidence.size() ||
          observation.identity != canonical_evidence[index].identity ||
          observation.relative_path !=
              canonical_evidence[index].relative_path.string() ||
          observation.compiled_sha256 !=
              canonical_evidence[index].expected_sha256))) {
      return deny("PPO-pilot immutable evidence observation is not exact");
    }
  }
  std::string optimizer_hash;
  std::string actor_hash;
  std::string optimizer_error;
  const auto& final_summary = state.committed_updates.back().GetObject();
  const auto final_actor =
      final_summary.find("live_actor_after_disposition_sha256");
  if (!vrpo_training_internal::ModuleValueSha256(
          actor, "", &actor_hash, &optimizer_error) ||
      final_actor == final_summary.end() || !final_actor->second.IsString() ||
      final_actor->second.GetString() != actor_hash ||
      !vrpo_ppo_pilot_internal::ActorOptimizerStateSha256(
          actor_optimizer, actor, kVrpoPpoPilotMaxActorSteps,
          &optimizer_hash, &optimizer_error) ||
      optimizer_hash != state.latest_actor_optimizer_state_sha256) {
    return deny("PPO-pilot final actor optimizer step/device/dtype/hash is invalid");
  }
  for (int update = 1; update <= kVrpoPpoPilotUpdates; ++update) {
    std::string update_error;
    if (!ValidateVrpoPpoPilotCommittedUpdateDirectory(
            startup.output_root, update,
            state.committed_updates[update - 1], &update_error)) {
      return deny(update_error);
    }
  }
  VrpoExpandedArchiveIdentity input_after;
  std::string input_error;
  if (!ComputeVrpoExpandedArchiveIdentity(
          startup.input_archive, &input_after, &input_error) ||
      input_after.combined_sha256 != input_identity.combined_sha256) {
    return deny("PPO-pilot source origin changed before authorization");
  }
  if (denial_reason != nullptr) denial_reason->clear();
  return true;
}

enum class VrpoPpoPilotFailurePoint {
  kNone,
  kLateAfterTraining,
  kAfterActorTemp,
  kAfterOptimizerTemp,
  kAfterReload,
  kForceHealthStopAfterUpdate2,
  kForceHealthPassForTest,
};

enum class VrpoPpoPilotDisposition {
  kContinue,
  kEarlyStop,
  kComplete,
};

inline bool WriteVrpoPpoPilotGlobalResult(
    const VrpoPpoPilotStartupConfig& startup,
    const VrpoExpandedArchiveIdentity& input_identity,
    const VrpoExpandedExpectedLayout& layout,
    const VrpoPpoPilotState& state, torch::nn::Module& actor,
    torch::optim::AdamW& actor_optimizer,
    const VrpoPpoPilotDeadline& deadline,
    json::Object* result, std::string* error) {
  if (result == nullptr) {
    if (error != nullptr) *error = "PPO-pilot global status arguments invalid";
    return false;
  }
  std::string authorization_denial;
  const bool raw128_authorized = VrpoPpoPilotRaw128Authorization(
      startup, input_identity, layout, state, actor, actor_optimizer,
      &authorization_denial);
  const std::string classification = state.had_failure
      ? "INVALID"
      : raw128_authorized
          ? "VALID_FOUR_UPDATE_PPO_PILOT_HEALTH_PASS"
          : state.stopped
              ? "VALID_EARLY_STOP_HEALTH"
              : "INVALID";
  const std::string reason = state.had_failure
      ? state.failure_reason
      : raw128_authorized
          ? "all four immutable, update, reload, health, and optimizer gates passed"
          : state.stopped
              ? "predeclared health gate failed; current update rolled back"
              : authorization_denial;
  if (!deadline.Check("before global status", error)) return false;
  const int64_t retained_before =
      vrpo_ppo_pilot_internal::DirectoryRegularBytes(startup.output_root);
  if (retained_before < 0 ||
      retained_before > kVrpoPpoPilotRetainedByteCeiling) {
    if (error != nullptr) {
      *error = "PPO-pilot retained bytes exceeded 8 GiB before status";
    }
    return false;
  }
  json::Object root;
  root["schema"] = kVrpoPpoPilotResultSchema;
  root["classification"] = classification;
  root["status"] = classification == "INVALID" ? "INVALID" : "VALID";
  root["reason"] = reason;
  root["registration_id"] = startup.registration_id;
  root["compiled_registration_id"] = kVrpoPpoPilotRegistrationId;
  root["profile"] = startup.profile;
  root["compiled_profile"] = kVrpoPpoPilotProfile;
  root["purpose"] = "FINITE_ACTOR_ONLY_PPO_HEALTH_PILOT";
  root["source_code_sha256"] = startup.source_code_sha256;
  root["executed_binary_sha256"] = startup.executed_binary_sha256;
  root["executed_binary_size"] = startup.executed_binary_size;
  root["input_archive_sha256"] = input_identity.combined_sha256;
  root["input_actor_file_sha256"] = input_identity.files[0].sha256;
  root["input_q_file_sha256"] = input_identity.files[1].sha256;
  root["registered_origin_archive_sha256"] =
      startup.origin_archive_sha256;
  root["compiled_origin_archive_sha256"] =
      kVrpoPpoPilotOriginArchiveSha256;
  root["origin_archive_match"] =
      startup.origin_archive_sha256 == kVrpoPpoPilotOriginArchiveSha256 &&
      input_identity.combined_sha256 == kVrpoPpoPilotOriginArchiveSha256;
  root["immutable_evidence"] =
      json::Value(VrpoPpoPilotEvidenceObservationsJson(
          state.evidence_observations));
  root["learning_rate"] = startup.learning_rate;
  root["learning_rate_exact"] =
      VrpoPhase4eCanonicalDouble(startup.learning_rate);
  root["actor_epochs_per_update"] = int64_t{kVrpoPpoPilotActorEpochs};
  root["games_per_update"] = int64_t{kVrpoPpoPilotGamesPerUpdate};
  root["completed_updates"] = static_cast<int64_t>(state.completed_updates);
  root["actor_optimizer_steps_total"] =
      state.total_actor_optimizer_steps;
  root["actor_optimizer_step_ceiling"] = kVrpoPpoPilotMaxActorSteps;
  root["q_role"] = kVrpoPpoPilotQRole;
  root["q_constructor_calls"] = VrpoQConstructorCalls();
  root["q_forward_calls"] = VrpoQForwardCheckedCalls();
  root["q_target_computations"] = VrpoScheduleQTargetComputations();
  root["q_optimizer_constructed"] = false;
  root["q_optimizer_steps"] = int64_t{0};
  root["source_q_values_sha256"] = state.source_q_values_sha256;
  root["initial_actor_values_sha256"] =
      state.initial_actor_values_sha256;
  root["initial_actor_optimizer_state_sha256"] =
      state.initial_actor_optimizer_state_sha256;
  root["latest_actor_optimizer_state_sha256"] =
      state.latest_actor_optimizer_state_sha256;
  root["frozen_update1_corpus_sha256"] =
      state.frozen_update1_corpus_sha256;
  root["fresh_actor_optimizer_at_start"] = true;
  root["source_optimizer_moments_loaded"] = false;
  root["optimizer_persisted_across_updates"] = true;
  root["start_episode_id"] = static_cast<int64_t>(startup.start_episode_id);
  root["compiled_base_seed"] =
      static_cast<int64_t>(kVrpoPpoPilotBaseSeed);
  root["runtime_base_seed"] = static_cast<int64_t>(startup.base_seed);
  root["compiled_start_episode_id"] =
      static_cast<int64_t>(kVrpoPpoPilotStartEpisodeId);
  root["compiled_end_episode_id_inclusive"] =
      static_cast<int64_t>(kVrpoPpoPilotEndEpisodeIdInclusive);
  json::Array compiled_ranges;
  for (int update = 0; update < kVrpoPpoPilotUpdates; ++update) {
    json::Object range;
    const uint64_t begin = kVrpoPpoPilotStartEpisodeId +
        update * kVrpoPpoPilotGamesPerUpdate;
    range["update"] = static_cast<int64_t>(update + 1);
    range["start_episode_id"] = static_cast<int64_t>(begin);
    range["end_episode_id_inclusive"] = static_cast<int64_t>(
        begin + kVrpoPpoPilotGamesPerUpdate - 1);
    compiled_ranges.emplace_back(std::move(range));
  }
  root["compiled_update_episode_ranges"] = std::move(compiled_ranges);
  root["next_episode_id"] = static_cast<int64_t>(
      startup.start_episode_id +
      state.completed_updates * kVrpoPpoPilotGamesPerUpdate);
  root["updates"] = state.committed_updates;
  root["retained_byte_ceiling"] = kVrpoPpoPilotRetainedByteCeiling;
  root["peak_byte_ceiling"] = kVrpoPpoPilotPeakByteCeiling;
  root["measured_peak_bytes"] = state.measured_peak_bytes;
  root["retained_bytes_before_status"] = retained_before;
  root["wall_clock_ceiling_seconds"] = int64_t{1800};
  root["elapsed_seconds"] = deadline.ElapsedSeconds();
  root["status_last"] = true;
  root["authorization_denial_reason"] = authorization_denial;
  root["raw128_authorized"] = raw128_authorized;
  root["authorizes_fresh_raw_128_screen"] = raw128_authorized;
  root["authorizes_longer_training"] = false;
  root["authorizes_vrpo"] = false;
  root["authorizes_promotion"] = false;
  const auto status_path = startup.output_root / "PILOT_RESULT.json";
  if (!vrpo_ppo_pilot_internal::WriteAtomicText(
          status_path, json::ToString(root, true) + "\n", error)) {
    return false;
  }
  const int64_t retained_after =
      vrpo_ppo_pilot_internal::DirectoryRegularBytes(startup.output_root);
  if (retained_after < 0 ||
      retained_after > kVrpoPpoPilotRetainedByteCeiling ||
      retained_after > kVrpoPpoPilotPeakByteCeiling ||
      !deadline.Check("after global status fsync", error)) {
    std::error_code cleanup_ec;
    std::filesystem::remove(status_path, cleanup_ec);
    std::string ignored;
    VrpoFsyncDirectory(startup.output_root, &ignored);
    if (error != nullptr && error->empty()) {
      *error = "PPO-pilot final retained/deadline gate failed";
    }
    return false;
  }
  *result = std::move(root);
  return true;
}

inline bool InitializeVrpoPpoPilot(
    const VrpoPpoPilotStartupConfig& startup,
    const VrpoExpandedArchiveIdentity& input_identity,
    const VrpoExpandedExpectedLayout& layout,
    const std::shared_ptr<SharedDunePolicyValueNetImpl>& actor,
    const std::shared_ptr<torch::nn::Module>& q,
    torch::optim::AdamW& actor_optimizer,
    const VrpoPpoPilotDeadline& deadline, VrpoPpoPilotState* state,
    std::string* error) {
  auto fail = [&](const std::string& message) {
    if (error != nullptr) *error = message;
    if (state != nullptr) *state = VrpoPpoPilotState{};
    return false;
  };
  if (state == nullptr || actor == nullptr || q == nullptr ||
      !ValidateVrpoPpoPilotStartupConfig(startup, error,
                                         layout.test_fixture) ||
      startup.origin_archive_sha256 != input_identity.combined_sha256 ||
      (!layout.test_fixture && input_identity.combined_sha256 !=
                                   kVrpoPpoPilotOriginArchiveSha256) ||
      !deadline.Check("before pilot initialization", error) ||
      !vrpo_ppo_pilot_internal::CheckFreeSpaceBeforeStart(
          startup.output_root, error) ||
      !ValidateVrpoScheduleQProvenanceInstrumentation(
          kVrpoPpoPilotQRole, error) ||
      !vrpo_training_internal::OptimizerStateIsFresh(actor_optimizer,
                                                     error)) {
    return fail(error != nullptr && !error->empty()
                    ? *error
                    : "PPO-pilot initialization rejected");
  }
  VrpoPpoPilotState initialized;
  if (!vrpo_training_internal::ModuleValueSha256(
          *actor, "", &initialized.initial_actor_values_sha256, error) ||
      !vrpo_training_internal::ModuleValueSha256(
          *q, "", &initialized.source_q_values_sha256, error) ||
      !vrpo_ppo_pilot_internal::ActorOptimizerStateSha256(
          actor_optimizer, *actor, 0,
          &initialized.initial_actor_optimizer_state_sha256, error)) {
    return fail(error != nullptr ? *error
                                 : "PPO-pilot initial state hashing failed");
  }
  initialized.latest_actor_optimizer_state_sha256 =
      initialized.initial_actor_optimizer_state_sha256;
  initialized.evidence_observations = startup.evidence_observations;
  std::error_code ec;
  if (!std::filesystem::create_directories(startup.output_root, ec) || ec ||
      !VrpoFsyncDirectory(startup.output_root.parent_path(), error) ||
      !vrpo_ppo_pilot_internal::ObserveBytes(
          startup.output_root, &initialized.measured_peak_bytes, error)) {
    vrpo_ppo_pilot_internal::CleanupPath(startup.output_root);
    return fail(error != nullptr && !error->empty()
                    ? *error
                    : "PPO-pilot fresh output root creation failed");
  }
  *state = std::move(initialized);
  return true;
}

inline bool WriteVrpoPpoPilotUpdate(
    const VrpoPpoPilotStartupConfig& startup,
    const VrpoPhase4ManifestBinding& binding,
    const VrpoExpandedExpectedLayout& layout,
    const VrpoExpandedArchiveIdentity& input_identity,
    const std::vector<VrpoTrainingEpisode>& episodes,
    const VrpoPhase4ePairingStats& pairing,
    const std::string& collection_actor_values_sha256,
    const std::shared_ptr<SharedDunePolicyValueNetImpl>& actor,
    const std::shared_ptr<torch::nn::Module>& q,
    torch::optim::AdamW& actor_optimizer, const torch::Device& device,
    const VrpoPpoPilotDeadline& deadline, int update_number,
    VrpoPpoPilotFailurePoint failure_point, VrpoPpoPilotState* state,
    VrpoPpoPilotDisposition* disposition, json::Object* update_result,
    std::string* error) {
  const auto temp_dir = startup.output_root /
      (".update_" + std::to_string(update_number) + ".tmp");
  const auto final_dir = startup.output_root /
      ("update_" + std::to_string(update_number));
  auto fail = [&](const std::string& message) {
    vrpo_ppo_pilot_internal::CleanupPath(temp_dir);
    vrpo_ppo_pilot_internal::CleanupPath(final_dir);
    if (error != nullptr) *error = message;
    if (update_result != nullptr) update_result->clear();
    return false;
  };
  if (state == nullptr || disposition == nullptr || update_result == nullptr ||
      actor == nullptr || q == nullptr || state->stopped || state->completed ||
      update_number != state->completed_updates + 1 || update_number < 1 ||
      update_number > kVrpoPpoPilotUpdates ||
      episodes.size() != kVrpoPpoPilotGamesPerUpdate ||
      pairing.episodes != kVrpoPpoPilotGamesPerUpdate ||
      VrpoQConstructorCalls() != 1 || VrpoQForwardCheckedCalls() != 0 ||
      VrpoScheduleQTargetComputations() != 0 ||
      std::filesystem::exists(temp_dir) || std::filesystem::exists(final_dir) ||
      !deadline.Check("before update", error)) {
    return fail(error != nullptr && !error->empty()
                    ? *error
                    : "PPO-pilot update sequencing/input contract rejected");
  }
  const uint64_t expected_start = startup.start_episode_id +
      static_cast<uint64_t>(update_number - 1) *
          kVrpoPpoPilotGamesPerUpdate;
  const uint64_t expected_next =
      expected_start + kVrpoPpoPilotGamesPerUpdate;
  if (episodes.front().episode_id != expected_start ||
      episodes.back().episode_id + 1 != expected_next ||
      pairing.episode_evidence.front().episode_id != expected_start ||
      pairing.episode_evidence.back().episode_id + 1 != expected_next ||
      !ValidateVrpoPhase4ePairingStatsAgainstEpisodes(
          pairing, episodes, error)) {
    return fail(error != nullptr && !error->empty()
                    ? *error
                    : "PPO-pilot update IDs/pairing are invalid");
  }
  std::string actor_before;
  std::string q_before;
  const int64_t expected_before_step =
      static_cast<int64_t>(update_number - 1) * kVrpoPpoPilotStepsPerUpdate;
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
                    : "PPO-pilot pre-update actor/Q/optimizer binding failed");
  }

  std::string corpus_bytes;
  VrpoScheduleCorpusIdentity corpus_identity;
  if (!EncodeVrpoScheduleActorCorpus(
          episodes, &corpus_bytes, &corpus_identity, error)) {
    return fail(error != nullptr ? *error : "PPO-pilot corpus encode failed");
  }
  std::vector<VrpoTrainingEpisode> decoded;
  VrpoScheduleCorpusIdentity decoded_identity;
  if (!DecodeVrpoScheduleActorCorpus(
          corpus_bytes, &decoded, &decoded_identity, error) ||
      decoded_identity.payload_sha256 != corpus_identity.payload_sha256) {
    return fail(error != nullptr ? *error : "PPO-pilot corpus reload failed");
  }
  const auto& cumulative_corpus =
      update_number == 1 ? decoded : state->frozen_update1_episodes;
  if (cumulative_corpus.empty() && update_number != 1) {
    return fail("PPO-pilot fixed update-1 cumulative corpus is missing");
  }

  VrpoScheduleActorTransaction outer_transaction;
  if (!outer_transaction.Capture(*actor, actor_optimizer, error)) {
    return fail(error != nullptr ? *error
                                 : "PPO-pilot rollback capture failed");
  }
  const uint64_t update_seed = vrpo_training_internal::SplitMix64(
      startup.base_seed ^ expected_start ^
      (0x50504f50494c4f54ULL + static_cast<uint64_t>(update_number)));
  VrpoActorForward actor_forward = [actor](const torch::Tensor& input) {
    const auto out = actor->forward(input);
    return VrpoActorTrainingOutput{out.logits, out.values};
  };
  VrpoScheduleActorOnlyUpdateStats stats;
  const VrpoScheduleRunDeadline* no_schedule_deadline = nullptr;
  if (!RunVrpoScheduleActorOnlyPpoCell(
          decoded, update_seed, kVrpoPpoPilotActorEpochs, *actor,
          actor_optimizer, actor_forward,
          /*four_epoch_equivalence_test=*/false, no_schedule_deadline, &stats,
          error, /*require_fresh_optimizer=*/update_number == 1)) {
    return fail(error != nullptr ? *error
                                 : "PPO-pilot actor-only update failed");
  }
  if (!deadline.Check("after actor-only update", error)) {
    return fail(*error);
  }
  if (failure_point == VrpoPpoPilotFailurePoint::kLateAfterTraining) {
    return fail("injected PPO-pilot late failure after training");
  }
  const int64_t expected_after_step =
      static_cast<int64_t>(update_number) * kVrpoPpoPilotStepsPerUpdate;
  std::string actor_after;
  std::string q_after;
  std::string optimizer_after;
  if (stats.ppo.actor_optimizer_steps != kVrpoPpoPilotStepsPerUpdate ||
      stats.ppo.q_optimizer_steps != 0 ||
      stats.q_target_computations != 0 ||
      !vrpo_training_internal::ModuleValueSha256(
          *actor, "", &actor_after, error) || actor_after == actor_before ||
      !vrpo_training_internal::ModuleValueSha256(
          *q, "", &q_after, error) || q_after != q_before ||
      !vrpo_ppo_pilot_internal::ActorOptimizerStateSha256(
          actor_optimizer, *actor, expected_after_step, &optimizer_after,
          error)) {
    return fail(error != nullptr && !error->empty()
                    ? *error
                    : "PPO-pilot actor-step/Q-inert postcondition failed");
  }

  VrpoScheduleHealthMetrics current_health;
  VrpoScheduleHealthMetrics cumulative_health;
  const bool test_health_injection = layout.test_fixture &&
      (failure_point == VrpoPpoPilotFailurePoint::kForceHealthPassForTest ||
       failure_point ==
           VrpoPpoPilotFailurePoint::kForceHealthStopAfterUpdate2);
  if (test_health_injection) {
    current_health = VrpoScheduleHealthMetrics{
        64, 0.01, 0.02, 0.02, 0.03, 0.9, 1.0, 1.1,
        0.01, 0.02, 0.01, true};
    cumulative_health = current_health;
  } else {
    if (!EvaluateVrpoSchedulePolicyHealth(
            decoded, actor_forward, device, 256, &current_health, error) ||
        !EvaluateVrpoSchedulePolicyHealth(
            cumulative_corpus, actor_forward, device, 256,
            &cumulative_health, error) ||
        !deadline.Check("after health evaluation", error)) {
      return fail(error != nullptr ? *error
                                   : "PPO-pilot health evaluation failed");
    }
  }
  bool current_pass =
      vrpo_ppo_pilot_internal::CurrentHealthPass(current_health);
  bool cumulative_pass =
      vrpo_ppo_pilot_internal::CumulativeHealthPass(cumulative_health);
  if (failure_point == VrpoPpoPilotFailurePoint::kForceHealthPassForTest) {
    current_pass = true;
    cumulative_pass = true;
  }
  if (failure_point ==
          VrpoPpoPilotFailurePoint::kForceHealthStopAfterUpdate2 &&
      update_number == 2) {
    current_pass = false;
  }
  const bool health_pass = current_pass && cumulative_pass;

  std::error_code ec;
  if (!std::filesystem::create_directory(temp_dir, ec) || ec) {
    return fail("PPO-pilot update temp directory creation failed");
  }
  const auto corpus_path = temp_dir / "ACTOR_CORPUS.bin";
  if (!vrpo_ppo_pilot_internal::WriteAtomicText(
          corpus_path, corpus_bytes, error)) {
    return fail(*error);
  }
  json::Object evidence;
  evidence["schema"] = kVrpoPpoPilotEvidenceSchema;
  evidence["update"] = static_cast<int64_t>(update_number);
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
    return fail(std::string("PPO-pilot actor save failed: ") +
                exception.what());
  }
  if (!VrpoFsyncFile(actor_temp, error)) return fail(*error);
  if (failure_point == VrpoPpoPilotFailurePoint::kAfterActorTemp) {
    return fail("injected PPO-pilot failure after actor temp");
  }
  try {
    torch::save(actor_optimizer, optimizer_temp.string());
  } catch (const std::exception& exception) {
    return fail(std::string("PPO-pilot optimizer save failed: ") +
                exception.what());
  }
  if (!VrpoFsyncFile(optimizer_temp, error)) return fail(*error);
  if (failure_point == VrpoPpoPilotFailurePoint::kAfterOptimizerTemp) {
    return fail("injected PPO-pilot failure after optimizer temp");
  }

  auto reloaded_actor = std::make_shared<SharedDunePolicyValueNetImpl>(
      layout.actor_observation_dim, layout.actor_hidden_dim,
      layout.actor_action_dim, layout.actor_residual_blocks, false);
  reloaded_actor->to(device, torch::kFloat32);
  std::unique_ptr<torch::optim::AdamW> reloaded_optimizer;
  if (!MakeVrpoScheduleFreshActorOptimizer(
          *reloaded_actor, kVrpoPpoPilotLearningRate, &reloaded_optimizer,
          error)) {
    return fail(*error);
  }
  try {
    torch::load(reloaded_actor, actor_temp.string(), device);
    torch::load(*reloaded_optimizer, optimizer_temp.string());
  } catch (const std::exception& exception) {
    return fail(std::string("PPO-pilot strict reload failed: ") +
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
    return fail("PPO-pilot strict reload model/optimizer hash mismatch");
  }
  torch::Tensor canary = decoded.front().rows.front().actor_input
                             .reshape({1, -1}).to(device, torch::kFloat32);
  torch::Tensor live_canary;
  torch::Tensor reload_canary;
  try {
    torch::NoGradGuard no_grad;
    live_canary = actor_forward(canary).logits.detach().cpu();
    const auto out = reloaded_actor->forward(canary);
    reload_canary = out.logits.detach().cpu();
  } catch (const std::exception& exception) {
    return fail(std::string("PPO-pilot canary evaluation failed: ") +
                exception.what());
  }
  if (!torch::equal(live_canary, reload_canary)) {
    return fail("PPO-pilot strict reload canary mismatch");
  }
  if (failure_point == VrpoPpoPilotFailurePoint::kAfterReload) {
    return fail("injected PPO-pilot failure after reload");
  }
  std::filesystem::rename(actor_temp, actor_path);
  std::filesystem::rename(optimizer_temp, optimizer_path);

  VrpoExpandedArchiveIdentity input_after;
  if (!ComputeVrpoExpandedArchiveIdentity(
          startup.input_archive, &input_after, error) ||
      input_after.combined_sha256 != input_identity.combined_sha256 ||
      !ValidateVrpoScheduleQProvenanceInstrumentation(
          kVrpoPpoPilotQRole, error)) {
    return fail(error != nullptr && !error->empty()
                    ? *error
                    : "PPO-pilot input/Q provenance changed");
  }

  json::Array checksum_files;
  for (const auto& path :
       {corpus_path, evidence_path, actor_path, optimizer_path}) {
    json::Object identity =
        vrpo_ppo_pilot_internal::FileIdentityJson(path);
    const auto sha = identity.find("sha256");
    const auto size = identity.find("size");
    if (sha == identity.end() || !sha->second.IsString() ||
        !VrpoPhase4eLowerHex64(sha->second.GetString()) ||
        size == identity.end() || !size->second.IsInt() ||
        size->second.GetInt() <= 0) {
      return fail("PPO-pilot retained file identity failed");
    }
    checksum_files.emplace_back(std::move(identity));
  }

  json::Object root;
  root["schema"] = kVrpoPpoPilotUpdateSchema;
  root["classification"] = health_pass ? "VALID_UPDATE_HEALTH_PASS"
                                        : "VALID_EARLY_STOP_HEALTH";
  root["status"] = "VALID";
  root["update"] = static_cast<int64_t>(update_number);
  root["registration_id"] = startup.registration_id;
  root["profile"] = startup.profile;
  root["learning_rate"] = startup.learning_rate;
  root["learning_rate_exact"] =
      VrpoPhase4eCanonicalDouble(startup.learning_rate);
  root["actor_epochs"] = int64_t{kVrpoPpoPilotActorEpochs};
  root["update_seed"] = static_cast<int64_t>(update_seed);
  root["update_seed_uint64"] = std::to_string(update_seed);
  root["start_episode_id"] = static_cast<int64_t>(expected_start);
  root["end_episode_id_inclusive"] =
      static_cast<int64_t>(expected_next - 1);
  root["next_episode_id"] = static_cast<int64_t>(expected_next);
  root["rollout_games"] = int64_t{kVrpoPpoPilotGamesPerUpdate};
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
  root["value_head_before_sha256"] =
      stats.ppo.value_head_before_sha256;
  root["value_head_after_sha256"] = stats.ppo.value_head_after_sha256;
  root["actor_and_value_moved"] =
      stats.ppo.actor_values_before_sha256 !=
          stats.ppo.actor_values_after_sha256 &&
      stats.ppo.value_head_before_sha256 !=
          stats.ppo.value_head_after_sha256;
  root["q_role"] = kVrpoPpoPilotQRole;
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
  root["fixed_update1_corpus"] = true;
  root["strict_reload_passed"] = true;
  root["canary_exact"] = true;
  root["input_archive_untouched"] = true;
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
  checksums["schema"] = kVrpoPpoPilotChecksumsSchema;
  checksums["files"] = std::move(checksum_files);
  const auto checksums_path = temp_dir / "CHECKSUMS.json";
  if (!vrpo_ppo_pilot_internal::WriteAtomicText(
          checksums_path, json::ToString(checksums, true) + "\n", error) ||
      !vrpo_ppo_pilot_internal::ObserveBytes(
          startup.output_root, &state->measured_peak_bytes, error)) {
    return fail(*error);
  }
  std::filesystem::rename(temp_dir, final_dir);
  if (!VrpoFsyncDirectory(startup.output_root, error) ||
      !deadline.Check("after update commit", error) ||
      !vrpo_ppo_pilot_internal::ObserveBytes(
          startup.output_root, &state->measured_peak_bytes, error)) {
    return fail(error != nullptr ? *error
                                 : "PPO-pilot update commit failed");
  }
  if (health_pass) outer_transaction.Commit();
  state->completed_updates = update_number;
  state->total_actor_optimizer_steps += stats.ppo.actor_optimizer_steps;
  state->latest_actor_optimizer_state_sha256 =
      health_pass ? optimizer_after : optimizer_before;
  if (update_number == 1) {
    state->frozen_update1_episodes = decoded;
    state->frozen_update1_corpus_sha256 = corpus_identity.payload_sha256;
  }
  json::Object summary;
  summary["update"] = static_cast<int64_t>(update_number);
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
  state->committed_updates.emplace_back(std::move(summary));
  if (!health_pass) {
    state->stopped = true;
    *disposition = VrpoPpoPilotDisposition::kEarlyStop;
  } else if (update_number == kVrpoPpoPilotUpdates) {
    state->completed = true;
    *disposition = VrpoPpoPilotDisposition::kComplete;
  } else {
    *disposition = VrpoPpoPilotDisposition::kContinue;
  }
  *update_result = std::move(root);
  return true;
}

}  // namespace open_spiel

#endif  // OPEN_SPIEL_BUILD_WITH_LIBTORCH
#endif  // OPEN_SPIEL_EXAMPLES_DUNE_VRPO_PPO_PILOT_H_
