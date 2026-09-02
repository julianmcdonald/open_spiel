#ifndef OPEN_SPIEL_EXAMPLES_DUNE_VRPO_Q_WARMUP_H_
#define OPEN_SPIEL_EXAMPLES_DUNE_VRPO_Q_WARMUP_H_

// Frozen-actor Q-only warm-up selected after the bounded PPO continuation
// rolled back to the confirmed pilot-U4 actor.  This terminal mode is kept
// separate from phase4e, PPO continuation, and the ordinary PPO loop.

#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <set>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#if defined(__linux__)
#include <fcntl.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif

#include "dune_vrpo_ppo_continuation.h"

#ifdef OPEN_SPIEL_BUILD_WITH_LIBTORCH
#include <torch/torch.h>

namespace open_spiel {

inline constexpr char kVrpoQWarmupProfile[] =
    "VRPO_U15828_U4_QWARM4X16_HOLDOUT16_20260902";
inline constexpr char kVrpoQWarmupRegistrationId[] =
    "VRPO_U15828_U4_QWARM4X16_HOLDOUT16_20260902";
inline constexpr char kVrpoQWarmupResultSchema[] =
    "dune_vrpo_q_warmup_result_v1";
inline constexpr char kVrpoQWarmupUpdateSchema[] =
    "dune_vrpo_q_warmup_update_v1";
inline constexpr char kVrpoQWarmupMetricsSchema[] =
    "dune_vrpo_q_warmup_holdout_metrics_v1";
inline constexpr char kVrpoQWarmupChecksumsSchema[] =
    "dune_vrpo_q_warmup_checksums_v1";
inline constexpr char kVrpoQWarmupCentralCorpusSchema[] =
    "dune_vrpo_q_warmup_central_corpus_v1";
inline constexpr uint64_t kVrpoQWarmupBaseSeed = 8305001;
inline constexpr uint64_t kVrpoQWarmupTrainStartEpisodeId = 1200300000;
inline constexpr uint64_t kVrpoQWarmupTrainEndEpisodeIdInclusive = 1200300063;
inline constexpr uint64_t kVrpoQWarmupHoldoutStartEpisodeId = 1200300064;
inline constexpr uint64_t kVrpoQWarmupHoldoutEndEpisodeIdInclusive = 1200300079;
inline constexpr int kVrpoQWarmupUpdates = 4;
inline constexpr int kVrpoQWarmupGamesPerUpdate = 16;
inline constexpr int kVrpoQWarmupHoldoutGames = 16;
inline constexpr int kVrpoQWarmupQEpochs = 4;
inline constexpr int64_t kVrpoQWarmupStepsPerUpdate = 64;
inline constexpr int64_t kVrpoQWarmupFinalQSteps = 256;
inline constexpr uint64_t kVrpoQWarmupQSeed = 2026083001;
inline constexpr double kVrpoQWarmupImprovementRatio = 0.90;
inline constexpr double kVrpoQWarmupPearsonTolerance = 1e-6;
inline constexpr int64_t kVrpoQWarmupRetainedByteCeiling =
    8LL * 1024 * 1024 * 1024;
inline constexpr int64_t kVrpoQWarmupPeakByteCeiling =
    10LL * 1024 * 1024 * 1024;

inline constexpr char kVrpoQWarmupActorValuesSha256[] =
    "8604ee82e76e86591b123c69ebab0a85e6b4aabd3bc088b48f922a35590a988d";
inline constexpr char kVrpoQWarmupActorFileSha256[] =
    "8b3d4e5061ed8db611a266880d249b2064268d00f4c280d1e695c265f7ed46fd";
// These two identities intentionally hash the same Q tensors in different,
// independently versioned domains.  The canonical expanded-checkpoint
// identity is the bootstrap-manifest contract.  The runtime module identity is
// used for mutation/reload checks.  They must never be substituted.
inline constexpr char kVrpoQWarmupCanonicalQOriginSchema[] =
    "dune_vrpo_q_module_v1";
inline constexpr char kVrpoQWarmupRuntimeQStateSchema[] =
    "dune_vrpo_named_parameter_values_v1";
inline constexpr char kVrpoQWarmupQFileIdentitySchema[] =
    "sha256_file_bytes_v1";
inline constexpr char kVrpoQWarmupInitialQCanonicalValuesSha256[] =
    "b98937b5f924af181116c94726158c0be808c831d639038e69b80c5ff0666553";
inline constexpr char kVrpoQWarmupInitialQRuntimeValuesSha256[] =
    "ff77fe5fd6b5b557206405ca6c12aa157baa94095560fbbbadbe2b72ef1b2c26";
inline constexpr char kVrpoQWarmupInitialQFileSha256[] =
    "29a9e0b4bde117e00931be7cbc58f3f47d84cf9e2c492fd7141d5a1fce641646";
inline constexpr char kVrpoQWarmupInitialQOptimizerFileSha256[] =
    "598082ae85d76144343030ab0ce777f62126331a73de70dbc664d73deb043776";
inline constexpr char kVrpoQWarmupInitialQOptimizerStateSha256[] =
    "a844b0d10120ed761ae78247c5b96a9ed5420cac12785c0983bbd818aea715ec";

class VrpoQWarmupDeadline {
 public:
  static VrpoQWarmupDeadline Start(
      std::chrono::steady_clock::time_point start,
      std::chrono::seconds limit = std::chrono::seconds(1800)) {
    VrpoQWarmupDeadline out;
    out.active_ = true;
    out.start_ = start;
    out.deadline_ = start + limit;
    return out;
  }

  static VrpoQWarmupDeadline ExpireAfterChecksForTest(int checks) {
    VrpoQWarmupDeadline out = Start(std::chrono::steady_clock::now(),
                                    std::chrono::hours(1));
    out.test_checks_remaining_ = checks;
    return out;
  }

  bool Check(const std::string& stage, std::string* error) const {
    if (test_checks_remaining_ >= 0) {
      if (test_checks_remaining_ == 0) {
        if (error != nullptr) {
          *error = "injected Q-warmup deadline exceeded at " + stage;
        }
        return false;
      }
      --test_checks_remaining_;
    }
    if (!active_ || std::chrono::steady_clock::now() < deadline_) return true;
    if (error != nullptr) {
      *error = "Q-warmup 1800-second deadline exceeded at " + stage;
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
VrpoQWarmupCanonicalEvidenceFiles() {
  return {
      {"pilot_u4_actor",
       "ppo_pilot4x16_lr5e6_u15828_20260902/update_4/actor_model.pt",
       kVrpoQWarmupActorFileSha256},
      {"pilot_u4_result",
       "ppo_pilot4x16_lr5e6_u15828_20260902/update_4/UPDATE_RESULT.json",
       "bbbda80c821547e2a4149f1d2e31a5ab41b4bc4f66a70e7cf8663fb54b57ec3a"},
      {"pilot_result",
       "ppo_pilot4x16_lr5e6_u15828_20260902/PILOT_RESULT.json",
       "29d56508c5ab09e06245f68053352dca07f4f49bdbbaf605bba58d311402c8a6"},
      {"pilot_validation",
       "registrations/ppo_pilot4x16_lr5e6_u15828_20260902/validation.json",
       "c0f6b3880f55a64db6037785a2e7ede20bbaf78a5d1bbb09414fac0f8c1d56fb"},
      {"raw128_result",
       "raw_strength_screen_ppo_pilot_u4_u15828_128_20260902/SCREEN_RESULT.json",
       "37b7288b8fbf9c79ce7d3e01ca146149688b48ce86420e52ae0fa350f2318054"},
      {"raw128_validation",
       "raw_strength_screen_ppo_pilot_u4_u15828_128_20260902/validation.json",
       "c250feb8c86f42a3a9cae732f8cc91529ebc0f7a52d252cc0163cb4085e9b914"},
      {"confirm256_result",
       "raw_strength_confirm_ppo_pilot_u4_u15828_256_20260902/CONFIRM_RESULT.json",
       "19935d91ca60978ff483b4e9863130d70c10cea1512dc4b52f94ab24f84e92bf"},
      {"confirm256_validation",
       "raw_strength_confirm_ppo_pilot_u4_u15828_256_20260902/validation.json",
       "99c9ad6bbe968b129e638d632404b8ba6d7072c4d035763d03d5c83f1ebb556d"},
      {"continuation_registration",
       "registrations/ppo_continuation_u5_u8_lr5e6_u15828_20260902/registration.json",
       "247ecc4bef43b5e3ce1edb9223e2c8480a43031425e5074ebc671a8e58af15c8"},
      {"continuation_result",
       "ppo_continuation_u5_u8_lr5e6_u15828_20260902/CONTINUATION_RESULT.json",
       "7c2514ca0c556fa43625f321e0268cc11d58876d932d5d89f8410adcf1b4d86b"},
      {"continuation_validation",
       "registrations/ppo_continuation_u5_u8_lr5e6_u15828_20260902/validation.json",
       "a13b6d6182ce6caf74d8cd8326773b3698e809bc792282e79f20028577e97c5f"},
      {"continuation_u5_validation",
       "registrations/ppo_continuation_u5_u8_lr5e6_u15828_20260902/update_5.validation.json",
       "159b4e7f4ec406190491c8a283a67d161fae3f8fe3d3912d6c41f8cc406b3da1"},
      {"bootstrap_v2_registration",
       "registrations/bootstrap_u15828_fp32_v2_20260831/registration.json",
       "1ac3a7b7244e70a79a0ea749829726fc03d958d1e77d7609f8ba6774f86c5abf"},
      {"bootstrap_v2_result",
       "bootstrap_u15828_fp32_v2_20260831/BOOTSTRAP_RESULT.json",
       "48ac10a982e5c370e87fde3decea41af4a9f06d155f113a828989ccb92917ff5"},
      {"bootstrap_v2_validation",
       "registrations/bootstrap_u15828_fp32_v2_20260831/validation.json",
       "55e0ce2a69e92bb6beb515e7848a191168493a3a5d91ce780367f415f0347e52"},
      {"bootstrap_v2_vrpo_manifest",
       "bootstrap_u15828_fp32_v2_20260831/VRPO_CAP10/manifest.json",
       "4df2472e4e0e2529d3424f5c445b8a7644562e9a6f526fe685ad6f8d764b9196"},
      {"bootstrap_v2_q_model",
       "bootstrap_u15828_fp32_v2_20260831/VRPO_CAP10/q_model.pt",
       kVrpoQWarmupInitialQFileSha256},
      {"bootstrap_v2_q_optimizer",
       "bootstrap_u15828_fp32_v2_20260831/VRPO_CAP10/q_optimizer.pt",
       kVrpoQWarmupInitialQOptimizerFileSha256},
      {"invalid_q_warmup_registration",
       "registrations/vrpo_q_warmup4x16_holdout16_u4_20260902/registration.json",
       "9a7cd9954f85c68d96146b2580ff40eaa284c3b1f35a4a526d99bdc317f63635"},
      {"invalid_q_warmup_run_log",
       "registrations/vrpo_q_warmup4x16_holdout16_u4_20260902/run.log",
       "c18311e0fbd1445c361ecb16ccdef59df4ffdae29d9a11a4efc8cf1f475b9a0a"},
      {"invalid_q_warmup_validation",
       "registrations/vrpo_q_warmup4x16_holdout16_u4_20260902/validation.json",
       "d1262ddffae563f0d89d6ca82f85a016a7d73d404085bafd773d9d71f7008554"},
  };
}

struct VrpoQWarmupPriorEvidence {
  std::vector<VrpoPpoPilotEvidenceObservation> observations;
  std::filesystem::path actor_path;
  std::filesystem::path q_model_path;
  std::filesystem::path q_optimizer_path;
};

namespace vrpo_q_warmup_internal {

inline bool ClaimFreshDirectory(const std::filesystem::path& directory,
                                bool* owns_directory,
                                std::string* error) {
  if (owns_directory == nullptr || directory.empty()) {
    if (error != nullptr) *error = "Q-warmup directory claim is invalid";
    return false;
  }
  *owns_directory = false;
  std::error_code ec;
  const bool created = std::filesystem::create_directory(directory, ec);
  if (!created || ec) {
    if (error != nullptr) *error = "Q-warmup fresh directory already exists or cannot be claimed";
    return false;
  }
  *owns_directory = true;
  if (!VrpoFsyncDirectory(directory.parent_path(), error)) {
    vrpo_ppo_pilot_internal::CleanupPath(directory);
    *owns_directory = false;
    return false;
  }
  return true;
}

inline bool RenameDirectoryNoReplace(
    const std::filesystem::path& source,
    const std::filesystem::path& destination, bool* moved,
    std::string* error) {
  if (moved == nullptr || source.empty() || destination.empty()) {
    if (error != nullptr) *error = "Q-warmup no-replace rename is invalid";
    return false;
  }
  *moved = false;
#if defined(__linux__) && defined(SYS_renameat2)
  constexpr unsigned int kRenameNoReplace = 1U;
  errno = 0;
  const long result = ::syscall(
      SYS_renameat2, AT_FDCWD, source.c_str(), AT_FDCWD,
      destination.c_str(), kRenameNoReplace);
  if (result == 0) {
    *moved = true;
    return true;
  }
  if (error != nullptr) {
    *error = std::string("Q-warmup atomic no-replace rename failed: ") +
             std::strerror(errno);
  }
  return false;
#else
  if (error != nullptr) {
    *error = "Q-warmup atomic no-replace rename is unavailable";
  }
  return false;
#endif
}

inline bool WriteAtomicTextNoReplace(
    const std::filesystem::path& output, const std::string& contents,
    bool* owns_output, std::string* error) {
  if (owns_output == nullptr || output.empty()) {
    if (error != nullptr) *error = "Q-warmup atomic text arguments invalid";
    return false;
  }
  *owns_output = false;
  const auto temp = std::filesystem::path(output.string() + ".tmp");
#if defined(__linux__)
  const int fd = ::open(temp.c_str(), O_WRONLY | O_CREAT | O_EXCL, 0600);
  if (fd < 0) {
    if (error != nullptr) {
      *error = std::string("Q-warmup atomic text temp claim failed: ") +
               std::strerror(errno);
    }
    return false;
  }
  auto remove_temp = [&]() {
    std::error_code ignored;
    std::filesystem::remove(temp, ignored);
  };
  size_t written = 0;
  while (written < contents.size()) {
    const ssize_t count = ::write(fd, contents.data() + written,
                                  contents.size() - written);
    if (count <= 0) {
      if (error != nullptr) *error = "Q-warmup atomic text write failed";
      ::close(fd);
      remove_temp();
      return false;
    }
    written += static_cast<size_t>(count);
  }
  const bool fsync_ok = ::fsync(fd) == 0;
  const bool close_ok = ::close(fd) == 0;
  if (!fsync_ok || !close_ok) {
    if (error != nullptr) *error = "Q-warmup atomic text fsync/close failed";
    remove_temp();
    return false;
  }
  bool moved = false;
  if (!RenameDirectoryNoReplace(temp, output, &moved, error) || !moved) {
    remove_temp();
    return false;
  }
  *owns_output = true;
  if (!VrpoFsyncDirectory(output.parent_path(), error)) {
    std::error_code ignored;
    std::filesystem::remove(output, ignored);
    *owns_output = false;
    return false;
  }
  return true;
#else
  if (error != nullptr) {
    *error = "Q-warmup atomic no-replace text is unavailable";
  }
  return false;
#endif
}

inline bool ReadJsonObject(const std::filesystem::path& path,
                           json::Object* output, std::string* error) {
  if (output == nullptr || !std::filesystem::is_regular_file(path) ||
      std::filesystem::is_symlink(path)) {
    if (error != nullptr) *error = "Q-warmup JSON is unavailable";
    return false;
  }
  std::ifstream stream(path, std::ios::binary);
  const std::string text((std::istreambuf_iterator<char>(stream)),
                         std::istreambuf_iterator<char>());
  const auto parsed = json::FromString(text);
  if (!parsed.has_value() || !parsed->IsObject()) {
    if (error != nullptr) *error = "Q-warmup JSON is malformed";
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
    const std::filesystem::path& root, VrpoQWarmupPriorEvidence* output,
    std::string* error) {
  auto fail = [&](const std::string& message) {
    if (error != nullptr) *error = message;
    if (output != nullptr) *output = VrpoQWarmupPriorEvidence{};
    return false;
  };
  if (output == nullptr) return fail("null Q-warmup prior output");
  const auto specs = VrpoQWarmupCanonicalEvidenceFiles();
  std::vector<std::string> compiled;
  for (const auto& spec : specs) compiled.push_back(spec.expected_sha256);
  VrpoQWarmupPriorEvidence prior;
  if (!ValidateVrpoPpoPilotEvidenceFiles(
          root, specs, compiled, &prior.observations, error)) {
    return fail(error != nullptr ? *error : "Q-warmup evidence rejected");
  }
  json::Object u4;
  json::Object raw128;
  json::Object confirm;
  json::Object continuation;
  json::Object continuation_validation;
  json::Object continuation_u5;
  json::Object bootstrap;
  json::Object bootstrap_validation;
  json::Object bootstrap_manifest;
  json::Object invalid_predecessor_validation;
  if (!ReadJsonObject(root / specs[1].relative_path, &u4, error) ||
      !ReadJsonObject(root / specs[4].relative_path, &raw128, error) ||
      !ReadJsonObject(root / specs[6].relative_path, &confirm, error) ||
      !ReadJsonObject(root / specs[9].relative_path, &continuation, error) ||
      !ReadJsonObject(root / specs[10].relative_path,
                      &continuation_validation, error) ||
      !ReadJsonObject(root / specs[11].relative_path,
                      &continuation_u5, error) ||
      !ReadJsonObject(root / specs[13].relative_path, &bootstrap, error) ||
      !ReadJsonObject(root / specs[14].relative_path,
                      &bootstrap_validation, error) ||
      !ReadJsonObject(root / specs[15].relative_path,
                      &bootstrap_manifest, error) ||
      !ReadJsonObject(root / specs[20].relative_path,
                      &invalid_predecessor_validation, error)) {
    return fail(error != nullptr ? *error : "Q-warmup evidence parse failed");
  }
  if (!StringIs(u4, "actor_after_sha256", kVrpoQWarmupActorValuesSha256) ||
      !StringIs(u4, "classification", "VALID_UPDATE_HEALTH_PASS") ||
      !StringIs(raw128, "classification",
                "VALID_RAW_POLICY_GROSS_DAMAGE_SCREEN_SURVIVOR") ||
      !StringIs(confirm, "classification",
                "CONFIRMED_NONINFERIOR_FOR_PPO_CONTINUATION_DESIGN") ||
      !StringIs(continuation, "classification", "VALID_EARLY_STOP_HEALTH") ||
      !StringIs(continuation_validation, "classification",
                "VALID_EARLY_STOP_HEALTH") ||
      !BoolIs(continuation_validation,
              "candidate_retained_and_live_U4_rollback_passed", true) ||
      !IntIs(continuation_validation, "live_actor_optimizer_step", 64) ||
      !StringIs(continuation_u5, "live_actor_rolled_back_sha256",
                kVrpoQWarmupActorValuesSha256) ||
      !BoolIs(continuation_u5, "rollback_matches_prior_U4", true) ||
      !StringIs(bootstrap, "status", "VALID") ||
      !StringIs(bootstrap_validation, "status", "VALID") ||
      !StringIs(bootstrap_manifest, "q_values_sha256",
                kVrpoQWarmupInitialQCanonicalValuesSha256) ||
      !StringIs(bootstrap_manifest, "q_optimizer_state_sha256",
                kVrpoQWarmupInitialQOptimizerStateSha256) ||
      !IntIs(bootstrap_manifest, "global_update", 0) ||
      !StringIs(invalid_predecessor_validation, "status",
                "INVALID_FAIL_CLOSED") ||
      !StringIs(invalid_predecessor_validation, "classification",
                "Q_WARMUP_REJECTED_BEFORE_OUTPUT_ROOT_Q_RELOAD_IDENTITY_MISMATCH") ||
      !StringIs(invalid_predecessor_validation, "registration_sha256",
                "9a7cd9954f85c68d96146b2580ff40eaa284c3b1f35a4a526d99bdc317f63635")) {
    return fail("Q-warmup U4/confirmation/rollback/bootstrap semantics rejected");
  }
  const auto contract_it = bootstrap_manifest.find("phase4_contract");
  if (contract_it == bootstrap_manifest.end() ||
      !contract_it->second.IsObject() ||
      !StringIs(contract_it->second.GetObject(), "arm_id", "VRPO_CAP10") ||
      !IntIs(contract_it->second.GetObject(), "q_init_seed",
             static_cast<int64_t>(kVrpoQWarmupQSeed))) {
    return fail("Q-warmup bootstrap Q contract rejected");
  }
  prior.actor_path = root / specs[0].relative_path;
  prior.q_model_path = root / specs[16].relative_path;
  prior.q_optimizer_path = root / specs[17].relative_path;
  *output = std::move(prior);
  return true;
}

inline bool EvidenceStillExact(
    const std::filesystem::path& root,
    const std::vector<VrpoPpoPilotEvidenceObservation>& expected,
    std::string* error) {
  const auto specs = VrpoQWarmupCanonicalEvidenceFiles();
  std::vector<std::string> compiled;
  for (const auto& spec : specs) compiled.push_back(spec.expected_sha256);
  std::vector<VrpoPpoPilotEvidenceObservation> observed;
  if (!ValidateVrpoPpoPilotEvidenceFiles(
          root, specs, compiled, &observed, error) ||
      observed.size() != expected.size()) {
    return false;
  }
  for (size_t i = 0; i < observed.size(); ++i) {
    if (!observed[i].matched ||
        observed[i].identity != expected[i].identity ||
        observed[i].observed_sha256 != expected[i].observed_sha256 ||
        observed[i].observed_size != expected[i].observed_size) {
      if (error != nullptr) *error = "Q-warmup prior evidence drifted";
      return false;
    }
  }
  return true;
}

inline int64_t DirectoryRegularBytes(const std::filesystem::path& root) {
  return vrpo_ppo_pilot_internal::DirectoryRegularBytes(root);
}

inline bool ObserveBytes(const std::filesystem::path& root,
                         int64_t* measured_peak_bytes, std::string* error) {
  const int64_t observed = DirectoryRegularBytes(root);
  if (observed < 0 || measured_peak_bytes == nullptr) {
    if (error != nullptr) *error = "Q-warmup byte accounting failed";
    return false;
  }
  *measured_peak_bytes = std::max(*measured_peak_bytes, observed);
  if (observed > kVrpoQWarmupPeakByteCeiling) {
    if (error != nullptr) *error = "Q-warmup exceeded 10-GiB peak";
    return false;
  }
  return true;
}

inline bool CommitFinalStatusLast(
    const std::filesystem::path& output_root,
    const std::filesystem::path& status_path,
    const std::string& contents, const VrpoQWarmupDeadline& deadline,
    bool force_byte_overrun_for_test, bool force_deadline_overrun_for_test,
    bool* owns_output_root, int64_t* measured_peak_bytes,
    int64_t* retained_bytes_after_status, std::string* error) {
  auto fail_owned_root = [&](const std::string& message) {
    if (error != nullptr) *error = message;
    if (owns_output_root != nullptr && *owns_output_root) {
      vrpo_ppo_pilot_internal::CleanupPath(output_root);
      *owns_output_root = false;
    }
    return false;
  };
  if (owns_output_root == nullptr || !*owns_output_root ||
      measured_peak_bytes == nullptr || retained_bytes_after_status == nullptr ||
      status_path.parent_path() != output_root) {
    if (error != nullptr) *error = "Q-warmup final-status ownership is invalid";
    return false;
  }
  *retained_bytes_after_status = -1;
  bool owns_status = false;
  if (!WriteAtomicTextNoReplace(status_path, contents, &owns_status, error)) {
    // A preexisting status or temp belongs to somebody else. Preserve it and
    // the run root; this invocation did not acquire status ownership.
    return false;
  }
  if (!VrpoFsyncDirectory(output_root, error)) {
    return fail_owned_root(error != nullptr ? *error
                                             : "Q-warmup root fsync failed");
  }
  const int64_t observed = DirectoryRegularBytes(output_root);
  const int64_t effective_observed = force_byte_overrun_for_test
      ? kVrpoQWarmupPeakByteCeiling + 1
      : observed;
  if (observed < 0) {
    return fail_owned_root("Q-warmup final retained-byte accounting failed");
  }
  *retained_bytes_after_status = effective_observed;
  *measured_peak_bytes = std::max(*measured_peak_bytes, effective_observed);
  const bool byte_ok =
      effective_observed <= kVrpoQWarmupRetainedByteCeiling &&
      *measured_peak_bytes <= kVrpoQWarmupPeakByteCeiling;
  std::string deadline_error;
  const bool deadline_ok = !force_deadline_overrun_for_test &&
      deadline.Check("after final status write/fsync/rename/root-fsync",
                     &deadline_error);
  if (!byte_ok || !deadline_ok) {
    return fail_owned_root(
        !byte_ok ? "Q-warmup final status exceeded retained/peak byte ceiling"
                 : deadline_error.empty()
                       ? "Q-warmup final status exceeded deadline"
                       : deadline_error);
  }
  owns_status = false;
  *owns_output_root = false;
  return true;
}

inline std::string ModuleParametersAndBuffersSha256(
    torch::nn::Module& module, std::string* error) {
  std::string payload = "dune_vrpo_q_warmup_module_state_v1";
  auto append_named = [&](const auto& values, char kind) {
    for (const auto& item : values) {
      const torch::Tensor value =
          item.value().detach().contiguous().cpu().to(torch::kFloat32);
      if (!vrpo_training_internal::FiniteTensor(value)) return false;
      payload.push_back(kind);
      vrpo_schedule_internal::AppendString(&payload, item.key());
      vrpo_capture_internal::AppendPod(&payload, value.dim());
      for (int64_t size : value.sizes()) {
        vrpo_capture_internal::AppendPod(&payload, size);
      }
      payload.append(reinterpret_cast<const char*>(value.data_ptr<float>()),
                     value.numel() * sizeof(float));
    }
    return true;
  };
  if (!append_named(module.named_parameters(), 'P') ||
      !append_named(module.named_buffers(), 'B')) {
    if (error != nullptr) *error = "Q-warmup module state is nonfinite";
    return std::string();
  }
  return ComputeStringSHA256(payload);
}

struct QOriginIdentity {
  std::string q_model_file_identity_schema;
  std::string q_model_file_sha256;
  int64_t q_model_file_size = 0;
  std::string q_canonical_values_schema;
  std::string q_canonical_values_sha256;
  std::string q_runtime_module_values_schema;
  std::string q_runtime_module_values_sha256;
};

// Validates three separate claims before a Q optimizer or output directory may
// be constructed: exact serialized bytes, the canonical Dune-Q value identity
// used by expanded checkpoint manifests, and the generic runtime module-state
// identity used by training mutation checks.
inline bool ValidateQOriginIdentityBeforeOptimizer(
    DuneVrpoQNetImpl& q, const std::filesystem::path& q_model_path,
    const std::string& expected_file_sha256,
    const std::string& expected_canonical_values_sha256,
    const std::string& expected_runtime_module_values_sha256,
    QOriginIdentity* output, std::string* error) {
  auto fail = [&](const std::string& message) {
    if (error != nullptr) *error = message;
    if (output != nullptr) *output = QOriginIdentity{};
    return false;
  };
  if (output == nullptr ||
      !VrpoPhase4eLowerHex64(expected_file_sha256) ||
      !VrpoPhase4eLowerHex64(expected_canonical_values_sha256) ||
      !VrpoPhase4eLowerHex64(expected_runtime_module_values_sha256) ||
      expected_canonical_values_sha256 ==
          expected_runtime_module_values_sha256) {
    return fail("Q-warmup Q-origin expected identities are invalid");
  }
  size_t file_size = 0;
  const std::string file_sha256 =
      ComputeFileSHA256(q_model_path.string(), &file_size);
  if (file_size == 0 || file_sha256 != expected_file_sha256) {
    return fail("Q-warmup serialized Q file identity mismatch");
  }
  std::string canonical_values_sha256;
  std::string runtime_module_values_sha256;
  VrpoExpandedExpectedLayout canonical_layout;
  canonical_layout.test_fixture = false;
  if (!VrpoExpandedQValueIdentitySha256(
          q, canonical_layout, &canonical_values_sha256, error) ||
      !vrpo_training_internal::ModuleValueSha256(
          q, "", &runtime_module_values_sha256, error)) {
    return fail(error != nullptr && !error->empty()
                    ? *error
                    : "Q-warmup Q-origin value hashing failed");
  }
  if (canonical_values_sha256 != expected_canonical_values_sha256) {
    return fail("Q-warmup canonical expanded Dune-Q value identity mismatch");
  }
  if (runtime_module_values_sha256 !=
      expected_runtime_module_values_sha256) {
    return fail("Q-warmup generic runtime module-state identity mismatch");
  }
  if (canonical_values_sha256 == runtime_module_values_sha256) {
    return fail("Q-warmup independently versioned Q identity domains alias");
  }
  QOriginIdentity identity;
  identity.q_model_file_identity_schema = kVrpoQWarmupQFileIdentitySchema;
  identity.q_model_file_sha256 = file_sha256;
  identity.q_model_file_size = static_cast<int64_t>(file_size);
  identity.q_canonical_values_schema = kVrpoQWarmupCanonicalQOriginSchema;
  identity.q_canonical_values_sha256 = canonical_values_sha256;
  identity.q_runtime_module_values_schema = kVrpoQWarmupRuntimeQStateSchema;
  identity.q_runtime_module_values_sha256 = runtime_module_values_sha256;
  *output = std::move(identity);
  return true;
}

inline bool MakeFreshQOptimizer(
    torch::nn::Module& q,
    std::unique_ptr<torch::optim::AdamW>* optimizer,
    std::string* error) {
  if (optimizer == nullptr || q.named_parameters().is_empty()) {
    if (error != nullptr) *error = "Q-warmup Q optimizer arguments invalid";
    return false;
  }
  std::vector<torch::Tensor> parameters;
  std::set<void*> seen;
  for (const auto& item : q.named_parameters()) {
    void* identity = item.value().unsafeGetTensorImpl();
    if (!seen.insert(identity).second) {
      if (error != nullptr) *error = "Q-warmup Q parameter duplicated";
      return false;
    }
    parameters.push_back(item.value());
  }
  const auto spec = CanonicalVrpoPhase4OptimizerGroups()[2];
  *optimizer = std::make_unique<torch::optim::AdamW>(
      parameters,
      torch::optim::AdamWOptions(spec.learning_rate)
          .betas({spec.beta1, spec.beta2})
          .eps(spec.epsilon)
          .weight_decay(spec.weight_decay));
  MaterializeVrpoZeroAdamWState(**optimizer);
  return true;
}

inline bool QOptimizerStateSha256(
    torch::optim::AdamW& optimizer, torch::nn::Module& q,
    int64_t expected_step, std::string* output, std::string* error) {
  auto fail = [&](const std::string& message) {
    if (error != nullptr) *error = message;
    if (output != nullptr) output->clear();
    return false;
  };
  if (output == nullptr || expected_step < 0 ||
      optimizer.state().size() != q.named_parameters().size() ||
      optimizer.param_groups().size() != 1) {
    return fail(error != nullptr && !error->empty()
                    ? *error
                    : "Q-warmup Q optimizer coverage/state invalid");
  }
  const auto spec = CanonicalVrpoPhase4OptimizerGroups()[2];
  const auto& options = static_cast<const torch::optim::AdamWOptions&>(
      optimizer.param_groups()[0].options());
  if (options.lr() != spec.learning_rate ||
      std::get<0>(options.betas()) != spec.beta1 ||
      std::get<1>(options.betas()) != spec.beta2 ||
      options.eps() != spec.epsilon ||
      options.weight_decay() != spec.weight_decay) {
    return fail("Q-warmup Q optimizer options changed");
  }
  std::map<c10::TensorImpl*, std::string> names;
  for (const auto& item : q.named_parameters()) {
    names[item.value().unsafeGetTensorImpl()] = item.key();
  }
  std::string payload = "dune_vrpo_q_warmup_adamw_state_v1";
  vrpo_capture_internal::AppendPod(&payload,
                                   static_cast<uint64_t>(names.size()));
  std::set<c10::TensorImpl*> seen;
  for (const auto& group : optimizer.param_groups()) {
    for (const auto& parameter : group.params()) {
      auto* key = parameter.unsafeGetTensorImpl();
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
        return fail("Q-warmup Adam state/device/dtype/step invalid");
      }
      vrpo_schedule_internal::AppendString(&payload, name->second);
      vrpo_capture_internal::AppendPod(&payload, state->step());
      for (const auto& moment : {state->exp_avg(), state->exp_avg_sq()}) {
        const torch::Tensor value =
            moment.detach().contiguous().cpu().to(torch::kFloat32);
        vrpo_capture_internal::AppendPod(&payload, value.numel());
        payload.append(reinterpret_cast<const char*>(value.data_ptr<float>()),
                       value.numel() * sizeof(float));
      }
    }
  }
  if (seen.size() != names.size()) {
    return fail("Q-warmup Q optimizer omitted a parameter");
  }
  *output = ComputeStringSHA256(payload);
  return true;
}

class QTransaction {
 public:
  bool Capture(torch::nn::Module& q, torch::optim::AdamW& optimizer,
               std::string* error) {
    if (active_) {
      if (error != nullptr) *error = "Q-warmup transaction already active";
      return false;
    }
    q_ = &q;
    optimizer_ = &optimizer;
    training_ = q.is_training();
    try {
      if (!vrpo_training_internal::CaptureUniformModuleRuntimePlacement(
              q, &placement_, error)) {
        return false;
      }
      torch::serialize::OutputArchive module_archive;
      q.save(module_archive);
      std::ostringstream module_stream(std::ios::out | std::ios::binary);
      module_archive.save_to(module_stream);
      module_bytes_ = module_stream.str();
      torch::serialize::OutputArchive optimizer_archive;
      optimizer.save(optimizer_archive);
      std::ostringstream optimizer_stream(std::ios::out | std::ios::binary);
      optimizer_archive.save_to(optimizer_stream);
      optimizer_bytes_ = optimizer_stream.str();
    } catch (const std::exception& exception) {
      if (error != nullptr) {
        *error = std::string("Q-warmup transaction capture failed: ") +
                 exception.what();
      }
      Clear();
      return false;
    }
    active_ = true;
    return true;
  }

  bool Rollback(std::string* error) noexcept {
    if (!active_) return true;
    active_ = false;
    try {
      std::istringstream module_stream(module_bytes_,
                                       std::ios::in | std::ios::binary);
      torch::serialize::InputArchive module_archive;
      module_archive.load_from(module_stream, torch::kCPU);
      q_->load(module_archive);
      if (!vrpo_training_internal::RestoreModuleRuntimePlacement(
              *q_, placement_, error)) {
        Clear();
        return false;
      }
      optimizer_->state().clear();
      std::istringstream optimizer_stream(optimizer_bytes_,
                                          std::ios::in | std::ios::binary);
      torch::serialize::InputArchive optimizer_archive;
      optimizer_archive.load_from(optimizer_stream, torch::kCPU);
      optimizer_->load(optimizer_archive);
      if (!vrpo_training_internal::MigrateAdamWOptimizerStateToParameterDevices(
              *optimizer_, error)) {
        Clear();
        return false;
      }
      optimizer_->zero_grad();
      q_->train(training_);
    } catch (const std::exception& exception) {
      if (error != nullptr) {
        *error = std::string("Q-warmup transaction rollback failed: ") +
                 exception.what();
      }
      Clear();
      return false;
    }
    Clear();
    return true;
  }

  void Commit() {
    active_ = false;
    Clear();
  }

  ~QTransaction() {
    std::string ignored;
    Rollback(&ignored);
  }

 private:
  void Clear() {
    q_ = nullptr;
    optimizer_ = nullptr;
    module_bytes_.clear();
    optimizer_bytes_.clear();
    placement_ = vrpo_training_internal::ModuleRuntimePlacement{};
  }
  bool active_ = false;
  bool training_ = false;
  torch::nn::Module* q_ = nullptr;
  torch::optim::AdamW* optimizer_ = nullptr;
  std::string module_bytes_;
  std::string optimizer_bytes_;
  vrpo_training_internal::ModuleRuntimePlacement placement_;
};

}  // namespace vrpo_q_warmup_internal

struct VrpoQWarmupStartupConfig : VrpoPpoPilotStartupConfig {
  VrpoQWarmupPriorEvidence prior;
};

inline std::vector<std::string> VrpoQWarmupSourceRelativePaths() {
  std::vector<std::string> paths = VrpoPpoContinuationSourceRelativePaths();
  paths.push_back("open_spiel/examples/dune_vrpo_q_warmup.h");
  return paths;
}

inline bool LoadVrpoQWarmupSourceIdentity(
    const std::filesystem::path& source_root,
    const std::string& registered_sha256, VrpoPhase4eSourceIdentity* output,
    std::string* error) {
  auto fail = [&](const std::string& message) {
    if (error != nullptr) *error = message;
    if (output != nullptr) *output = VrpoPhase4eSourceIdentity{};
    return false;
  };
  if (output == nullptr || !VrpoPhase4eLowerHex64(registered_sha256)) {
    return fail("Q-warmup source identity is invalid");
  }
  std::error_code ec;
  const auto canonical_root = std::filesystem::canonical(source_root, ec);
  if (ec || !std::filesystem::is_directory(canonical_root)) {
    return fail("Q-warmup source root is unreadable");
  }
  std::set<std::string> seen;
  std::string payload;
  const auto paths = VrpoQWarmupSourceRelativePaths();
  for (const auto& relative : paths) {
    if (relative.empty() || std::filesystem::path(relative).is_absolute() ||
        relative.find("..") != std::string::npos ||
        !seen.insert(relative).second) {
      return fail("Q-warmup source list is noncanonical");
    }
    const auto path = canonical_root / relative;
    if (!std::filesystem::is_regular_file(path) ||
        std::filesystem::is_symlink(path)) {
      return fail("Q-warmup source file missing: " + relative);
    }
    size_t size = 0;
    const std::string digest = ComputeFileSHA256(path.string(), &size);
    if (!VrpoPhase4eLowerHex64(digest) || size == 0) {
      return fail("Q-warmup source file hash failed: " + relative);
    }
    payload.append(relative);
    payload.push_back('\0');
    payload.append(digest);
    payload.push_back('\n');
  }
  const std::string observed = ComputeStringSHA256(payload);
  if (observed != registered_sha256) {
    return fail("Q-warmup registered source SHA-256 mismatch");
  }
  output->canonical_root = canonical_root;
  output->relative_paths = paths;
  output->combined_sha256 = observed;
  return true;
}

inline bool EncodeVrpoQWarmupCentralCorpus(
    const std::vector<VrpoTrainingEpisode>& episodes, std::string* output,
    VrpoScheduleCorpusIdentity* identity, std::string* error) {
  auto fail = [&](const std::string& message) {
    if (error != nullptr) *error = message;
    if (output != nullptr) output->clear();
    if (identity != nullptr) *identity = VrpoScheduleCorpusIdentity{};
    return false;
  };
  if (output == nullptr || identity == nullptr ||
      episodes.size() != kVrpoQWarmupGamesPerUpdate ||
      !vrpo_training_internal::ValidateAllData(episodes, error)) {
    return fail(error != nullptr && !error->empty()
                    ? *error
                    : "Q-warmup central corpus input rejected");
  }
  *identity = VrpoScheduleCorpusIdentity{};
  std::string out;
  vrpo_schedule_internal::AppendString(
      &out, kVrpoQWarmupCentralCorpusSchema);
  vrpo_schedule_internal::Append(
      &out, static_cast<uint64_t>(episodes.size()));
  uint64_t previous_episode = 0;
  uint64_t previous_row = 0;
  for (const auto& episode : episodes) {
    if (episode.episode_id <= previous_episode || episode.rows.empty()) {
      return fail("Q-warmup central corpus episode order rejected");
    }
    previous_episode = episode.episode_id;
    vrpo_schedule_internal::Append(&out, episode.episode_id);
    vrpo_schedule_internal::Append(
        &out, static_cast<uint64_t>(episode.rows.size()));
    std::string episode_payload;
    for (const auto& row : episode.rows) {
      if (row.row_id <= previous_row || !row.q_input.defined() ||
          row.q_input.dim() != 1 ||
          row.q_input.scalar_type() != torch::kFloat32 ||
          !vrpo_training_internal::FiniteTensor(row.q_input)) {
        return fail("Q-warmup central corpus row/Q input rejected");
      }
      previous_row = row.row_id;
      vrpo_schedule_internal::AppendString(
          &episode_payload,
          vrpo_schedule_internal::ActorRowPayload(row));
      const torch::Tensor central =
          row.q_input.detach().contiguous().cpu().to(torch::kFloat32);
      vrpo_schedule_internal::Append(
          &episode_payload, static_cast<uint64_t>(central.numel()));
      episode_payload.append(
          reinterpret_cast<const char*>(central.data_ptr<float>()),
          central.numel() * sizeof(float));
      ++identity->rows;
    }
    const std::string digest = ComputeStringSHA256(episode_payload);
    identity->episode_sha256.push_back(digest);
    vrpo_schedule_internal::AppendString(&out, digest);
    out.append(episode_payload);
  }
  identity->episodes = episodes.size();
  identity->payload_sha256 = ComputeStringSHA256(out);
  vrpo_schedule_internal::AppendString(&out, identity->payload_sha256);
  identity->byte_size = static_cast<int64_t>(out.size());
  *output = std::move(out);
  return true;
}

inline bool DecodeVrpoQWarmupCentralCorpus(
    const std::string& input, std::vector<VrpoTrainingEpisode>* episodes,
    VrpoScheduleCorpusIdentity* identity, std::string* error) {
  auto fail = [&](const std::string& message) {
    if (error != nullptr) *error = message;
    if (episodes != nullptr) episodes->clear();
    if (identity != nullptr) *identity = VrpoScheduleCorpusIdentity{};
    return false;
  };
  if (episodes == nullptr || identity == nullptr || input.empty()) {
    return fail("Q-warmup central corpus decode arguments invalid");
  }
  episodes->clear();
  *identity = VrpoScheduleCorpusIdentity{};
  size_t offset = 0;
  std::string schema;
  uint64_t episode_count = 0;
  if (!vrpo_schedule_internal::ReadString(input, &offset, &schema) ||
      schema != kVrpoQWarmupCentralCorpusSchema ||
      !vrpo_schedule_internal::Read(input, &offset, &episode_count) ||
      episode_count != kVrpoQWarmupGamesPerUpdate) {
    return fail("Q-warmup central corpus header rejected");
  }
  uint64_t previous_episode = 0;
  uint64_t previous_row = 0;
  for (uint64_t e = 0; e < episode_count; ++e) {
    VrpoTrainingEpisode episode;
    uint64_t row_count = 0;
    std::string expected_episode_digest;
    if (!vrpo_schedule_internal::Read(
            input, &offset, &episode.episode_id) ||
        episode.episode_id <= previous_episode ||
        !vrpo_schedule_internal::Read(input, &offset, &row_count) ||
        row_count == 0 || row_count > 1000000 ||
        !vrpo_schedule_internal::ReadString(
            input, &offset, &expected_episode_digest) ||
        !VrpoPhase4eLowerHex64(expected_episode_digest)) {
      return fail("Q-warmup central corpus episode header rejected");
    }
    previous_episode = episode.episode_id;
    const size_t episode_payload_begin = offset;
    for (uint64_t r = 0; r < row_count; ++r) {
      std::string actor_payload;
      VrpoTrainingRow row;
      size_t actor_offset = 0;
      uint64_t central_size = 0;
      if (!vrpo_schedule_internal::ReadString(
              input, &offset, &actor_payload) || actor_payload.empty() ||
          !vrpo_schedule_internal::ParseActorRow(
              actor_payload, &actor_offset, &row, error) ||
          actor_offset != actor_payload.size() ||
          row.episode_id != episode.episode_id ||
          row.row_id <= previous_row ||
          !vrpo_schedule_internal::Read(
              input, &offset, &central_size) || central_size == 0 ||
          central_size > 100000 ||
          central_size > (input.size() - offset) / sizeof(float)) {
        return fail(error != nullptr && !error->empty()
                        ? *error
                        : "Q-warmup central corpus row prefix rejected");
      }
      previous_row = row.row_id;
      row.q_input = torch::from_blob(
          const_cast<char*>(input.data() + offset),
          {static_cast<int64_t>(central_size)}, torch::kFloat32).clone();
      offset += central_size * sizeof(float);
      episode.rows.push_back(std::move(row));
      ++identity->rows;
    }
    const std::string observed = ComputeStringSHA256(
        input.substr(episode_payload_begin, offset - episode_payload_begin));
    if (observed != expected_episode_digest) {
      return fail("Q-warmup central corpus episode digest mismatch");
    }
    identity->episode_sha256.push_back(observed);
    episodes->push_back(std::move(episode));
  }
  const size_t payload_end = offset;
  std::string expected_payload_digest;
  if (!vrpo_schedule_internal::ReadString(
          input, &offset, &expected_payload_digest) ||
      offset != input.size() ||
      ComputeStringSHA256(input.substr(0, payload_end)) !=
          expected_payload_digest ||
      !vrpo_training_internal::ValidateAllData(*episodes, error)) {
    return fail(error != nullptr && !error->empty()
                    ? *error
                    : "Q-warmup central corpus payload/exhaustive decode rejected");
  }
  identity->episodes = episodes->size();
  identity->payload_sha256 = expected_payload_digest;
  identity->byte_size = static_cast<int64_t>(input.size());
  return true;
}

inline bool ValidateVrpoQWarmupStartupConfig(
    const VrpoQWarmupStartupConfig& config, std::string* error,
    bool test_fixture = false) {
  auto fail = [&](const std::string& message) {
    if (error != nullptr) *error = message;
    return false;
  };
  if (config.game != "dune_imperium" ||
      config.init_mode != "vrpo_q_warmup" ||
      config.profile != kVrpoQWarmupProfile ||
      (!test_fixture &&
       config.registration_id != kVrpoQWarmupRegistrationId) ||
      (test_fixture && config.registration_id.empty()) ||
      config.output_root.empty() || config.source_root.empty() ||
      config.evidence_root.empty() ||
      !VrpoPhase4eLowerHex64(config.source_code_sha256) ||
      !VrpoPhase4eLowerHex64(config.executed_binary_sha256) ||
      config.executed_binary_size <= 0 || config.prior.observations.empty() ||
      config.prior.actor_path.empty() || config.prior.q_model_path.empty() ||
      config.prior.q_optimizer_path.empty()) {
    return fail("Q-warmup identity/prior contract is invalid");
  }
  if (std::filesystem::exists(config.output_root)) {
    return fail("Q-warmup output root must be fresh and absent");
  }
  if (config.rollout_games != kVrpoQWarmupGamesPerUpdate ||
      config.threads != 8 || config.eval_batch_size != 64 ||
      config.eval_timeout_ms != 2 ||
      !config.evaluator_device_synchronize ||
      config.deterministic_rollout_eval || config.seed_scheme_version != 2 ||
      !config.runtime_device_is_cuda || config.runtime_device_index < 0 ||
      !config.one_gpu_process || config.runtime_process_id <= 0 ||
      (!test_fixture && config.base_seed != kVrpoQWarmupBaseSeed) ||
      (!test_fixture &&
       config.start_episode_id != kVrpoQWarmupTrainStartEpisodeId) ||
      (test_fixture && config.base_seed == 0) ||
      config.start_episode_id == 0 || config.diagnostics_only ||
      config.rollout_amp || config.train_amp || !config.allow_tf32 ||
      config.pipeline || config.online_search_collection ||
      config.search_pi_mode || config.train_value_only ||
      config.sample_counterfactual_states || config.has_search_label_dir ||
      config.ordinary_checkpoint_writes_enabled || config.anneal_lr) {
    return fail("Q-warmup execution/side-path contract is invalid");
  }
  for (double shaping :
       {config.shaped_reward_weight, config.tleilaxu_breadcrumb_weight,
        config.tleilaxu_level7_breadcrumb_weight,
        config.specimen_exchange_penalty}) {
    if (!std::isfinite(shaping) || shaping != 0.0) {
      return fail("Q-warmup shaping must be exactly zero");
    }
  }
  const auto spec = CanonicalVrpoPhase4OptimizerGroups()[2];
  if (config.learning_rate != spec.learning_rate ||
      config.ppo_update_epochs != kVrpoQWarmupQEpochs ||
      config.reward_scale != 4.0 || config.gamma != 1.0 ||
      config.lambda != 1.0 || config.logit_cap != 10.0 ||
      config.ppo_minibatches != kVrpoTrainingMinibatchesPerEpoch ||
      config.ppo_minibatch_size != 2048 ||
      config.clip_epsilon != kVrpoTrainingClipEpsilon ||
      config.entropy_coefficient != kVrpoTrainingEntropyCoefficient ||
      config.value_coefficient != kVrpoTrainingValueCoefficient ||
      config.gradient_clip_norm != kVrpoTrainingGradientClipNorm ||
      !config.normalize_advantages || !config.clip_value_loss ||
      kVrpoQWarmupFinalQSteps !=
          kVrpoQWarmupUpdates * kVrpoQWarmupStepsPerUpdate ||
      (!test_fixture &&
       config.start_episode_id +
                   kVrpoQWarmupUpdates * kVrpoQWarmupGamesPerUpdate - 1 !=
               kVrpoQWarmupTrainEndEpisodeIdInclusive)) {
    return fail("Q-warmup mechanics differ from registration");
  }
  return true;
}

struct VrpoQWarmupUpdateStats {
  bool success = false;
  int64_t q_optimizer_steps = 0;
  int64_t q_backward_calls = 0;
  int64_t q_rows_seen = 0;
  int64_t target_refreshes = 0;
  double q_loss_mean = 0.0;
  double max_q_grad_norm = 0.0;
  double target_min = std::numeric_limits<double>::infinity();
  double target_max = -std::numeric_limits<double>::infinity();
  double prediction_min = std::numeric_limits<double>::infinity();
  double prediction_max = -std::numeric_limits<double>::infinity();
  std::string actor_state_before_sha256;
  std::string actor_state_after_sha256;
  std::string q_values_before_sha256;
  std::string q_values_after_sha256;
  std::string target_values_sha256;
  std::string target_bundle_sha256;
  std::array<std::string, 4> q_epoch_partition_sha256;
  std::string deterministic_summary_sha256;
};

namespace vrpo_q_warmup_internal {

inline std::string UpdateStatsSha256(const VrpoQWarmupUpdateStats& stats) {
  std::string payload = "dune_vrpo_q_warmup_update_stats_v1";
  vrpo_capture_internal::AppendPod(&payload, stats.q_optimizer_steps);
  vrpo_capture_internal::AppendPod(&payload, stats.q_backward_calls);
  vrpo_capture_internal::AppendPod(&payload, stats.q_rows_seen);
  vrpo_capture_internal::AppendPod(&payload, stats.target_refreshes);
  vrpo_capture_internal::AppendPod(&payload, stats.q_loss_mean);
  vrpo_capture_internal::AppendPod(&payload, stats.max_q_grad_norm);
  vrpo_capture_internal::AppendPod(&payload, stats.target_min);
  vrpo_capture_internal::AppendPod(&payload, stats.target_max);
  vrpo_capture_internal::AppendPod(&payload, stats.prediction_min);
  vrpo_capture_internal::AppendPod(&payload, stats.prediction_max);
  payload.append(stats.actor_state_before_sha256);
  payload.append(stats.actor_state_after_sha256);
  payload.append(stats.q_values_before_sha256);
  payload.append(stats.q_values_after_sha256);
  payload.append(stats.target_values_sha256);
  payload.append(stats.target_bundle_sha256);
  for (const auto& digest : stats.q_epoch_partition_sha256) {
    payload.append(digest);
  }
  return ComputeStringSHA256(payload);
}

inline bool RestoreActorRuntime(torch::nn::Module& actor,
                                const std::vector<bool>& requires_grad,
                                bool training, std::string* error) {
  try {
    size_t index = 0;
    for (auto& item : actor.named_parameters()) {
      if (index >= requires_grad.size()) {
        if (error != nullptr) *error = "Q-warmup actor parameter count changed";
        return false;
      }
      item.value().set_requires_grad(requires_grad[index++]);
      item.value().mutable_grad() = torch::Tensor();
    }
    if (index != requires_grad.size()) {
      if (error != nullptr) *error = "Q-warmup actor parameter count changed";
      return false;
    }
    actor.train(training);
    return true;
  } catch (const std::exception& exception) {
    if (error != nullptr) {
      *error = std::string("Q-warmup actor runtime restore failed: ") +
               exception.what();
    }
    return false;
  }
}

}  // namespace vrpo_q_warmup_internal

inline bool RunVrpoQWarmupQOnlyUpdate(
    const std::vector<VrpoTrainingEpisode>& episodes, uint64_t update_seed,
    torch::nn::Module& actor_model, torch::nn::Module& q_model,
    torch::optim::AdamW& q_optimizer,
    const VrpoActorForward& actor_forward, const VrpoQForward& q_forward,
    int64_t expected_q_step_before, VrpoQWarmupUpdateStats* output,
    std::string* error) {
  auto fail = [&](const std::string& message) {
    if (error != nullptr) *error = message;
    if (output != nullptr) *output = VrpoQWarmupUpdateStats{};
    return false;
  };
  if (output == nullptr || !actor_forward || !q_forward || update_seed == 0 ||
      expected_q_step_before < 0 ||
      expected_q_step_before % kVrpoQWarmupStepsPerUpdate != 0) {
    return fail("Q-warmup update arguments are incomplete");
  }
  *output = VrpoQWarmupUpdateStats{};
  const VrpoPhase4ArmConfig* arm =
      FindCanonicalVrpoPhase4Arm("VRPO_CAP10");
  if (arm == nullptr || arm->algorithm != VrpoPhase4Algorithm::kVrpo ||
      arm->q_epochs != kVrpoQWarmupQEpochs ||
      arm->critic_replay_rollouts != 1 || !arm->complete_game_batches ||
      !vrpo_training_internal::ValidateAllData(episodes, error) ||
      !vrpo_training_internal::ValidateOptimizerCoverage(
          q_optimizer, q_model, actor_model, error)) {
    return fail(error != nullptr && !error->empty()
                    ? *error
                    : "Q-warmup mechanics/data/coverage rejected");
  }
  std::string optimizer_before;
  if (!vrpo_q_warmup_internal::QOptimizerStateSha256(
          q_optimizer, q_model, expected_q_step_before,
          &optimizer_before, error)) {
    return false;
  }
  vrpo_training_internal::ModuleRuntimePlacement actor_placement;
  vrpo_training_internal::ModuleRuntimePlacement q_placement;
  if (!vrpo_training_internal::CapturePhase4dRuntimePlacements(
          actor_model, q_model, &actor_placement, &q_placement, error)) {
    return false;
  }

  std::array<VrpoEpisodePartitionPlan, 4> q_plans;
  for (int epoch = 0; epoch < kVrpoQWarmupQEpochs; ++epoch) {
    if (!BuildVrpoEpisodePartitionPlan(
            episodes,
            vrpo_training_internal::SplitMix64(
                update_seed + 0x10000 + epoch + 1),
            &q_plans[epoch], error)) {
      return false;
    }
  }
  VrpoQWarmupUpdateStats stats;
  stats.actor_state_before_sha256 =
      vrpo_q_warmup_internal::ModuleParametersAndBuffersSha256(
          actor_model, error);
  if (stats.actor_state_before_sha256.empty() ||
      !vrpo_training_internal::ModuleValueSha256(
          q_model, "", &stats.q_values_before_sha256, error)) {
    return false;
  }

  // Refresh the paper's Expected-SARSA(lambda) targets from the current Q and
  // the unchanged actor before every update.  ComputeVrpoTrainingTargets owns
  // the checked relative-to-absolute boundary and executes under NoGradGuard.
  VrpoTrainingTargetBundle targets;
  if (!ComputeVrpoTrainingTargets(*arm, episodes, actor_model, q_model,
                                  actor_forward, q_forward, &targets, error) ||
      !vrpo_training_internal::ValidateTargetBundleFloat32Domain(
          targets, error) ||
      !ValidateVrpoTrainingTargetsFresh(
          targets, actor_model, q_model, error)) {
    return false;
  }
  stats.target_refreshes = 1;
  stats.target_values_sha256 = targets.target_values_sha256;
  stats.target_bundle_sha256 = targets.canonical_sha256;
  std::map<uint64_t, VrpoTrainingTargetRow> target_by_row;
  for (const auto& target : targets.rows) {
    if (!target_by_row.emplace(target.row_id, target).second) {
      return fail("Q-warmup refreshed targets repeat a row");
    }
    for (double value : target.q_target_actor_relative) {
      stats.target_min = std::min(stats.target_min, value);
      stats.target_max = std::max(stats.target_max, value);
    }
  }

  std::vector<bool> actor_requires_grad;
  for (auto& item : actor_model.named_parameters()) {
    actor_requires_grad.push_back(item.value().requires_grad());
    item.value().set_requires_grad(false);
    item.value().mutable_grad() = torch::Tensor();
  }
  const bool actor_was_training = actor_model.is_training();
  actor_model.eval();
  vrpo_q_warmup_internal::QTransaction transaction;
  if (!transaction.Capture(q_model, q_optimizer, error)) {
    vrpo_q_warmup_internal::RestoreActorRuntime(
        actor_model, actor_requires_grad, actor_was_training, nullptr);
    return false;
  }
  auto fail_after_capture = [&](const std::string& message) {
    std::string rollback_error;
    const bool q_rolled_back = transaction.Rollback(&rollback_error);
    const bool actor_restored =
        vrpo_q_warmup_internal::RestoreActorRuntime(
            actor_model, actor_requires_grad, actor_was_training,
            &rollback_error);
    return fail((q_rolled_back && actor_restored) ? message
                                                   : message + "; " + rollback_error);
  };

  double loss_sum = 0.0;
  int64_t loss_count = 0;
  for (int epoch = 0; epoch < kVrpoQWarmupQEpochs; ++epoch) {
    stats.q_epoch_partition_sha256[epoch] =
        q_plans[epoch].canonical_sha256;
    for (const auto& minibatch : q_plans[epoch].minibatches) {
      const auto flat = vrpo_training_internal::FlattenEpisodes(
          episodes, minibatch.episode_indices);
      q_optimizer.zero_grad();
      torch::Tensor q_inputs;
      if (!vrpo_training_internal::StackInputs(
              flat.rows, false, q_placement, &q_inputs, error)) {
        return fail_after_capture(*error);
      }
      torch::Tensor q_output;
      try {
        q_output = q_forward(q_inputs);
      } catch (const std::exception& exception) {
        return fail_after_capture(
            std::string("Q-warmup forward failed: ") + exception.what());
      }
      if (!vrpo_training_internal::ValidateQOutput(
              q_output, flat.rows.size(), q_placement, error)) {
        return fail_after_capture(*error);
      }
      std::vector<float> packed(flat.rows.size() * 5);
      for (size_t index = 0; index < flat.rows.size(); ++index) {
        const auto& row = *flat.rows[index];
        const auto found = target_by_row.find(row.row_id);
        if (found == target_by_row.end() || row.chosen_action < 0 ||
            row.chosen_action >= q_output.size(1)) {
          return fail_after_capture("Q-warmup target/action lookup failed");
        }
        packed[5 * index] = static_cast<float>(row.chosen_action);
        for (int slot = 0; slot < kVrpoNumSeats; ++slot) {
          packed[5 * index + 1 + slot] = static_cast<float>(
              found->second.q_target_actor_relative[slot]);
        }
      }
      torch::Tensor packed_targets;
      try {
        packed_targets = torch::from_blob(
            packed.data(), {static_cast<int64_t>(flat.rows.size()), 5},
            torch::TensorOptions().dtype(torch::kFloat32)).clone()
                             .to(q_placement.device);
      } catch (const std::exception& exception) {
        return fail_after_capture(
            std::string("Q-warmup target placement failed: ") +
            exception.what());
      }
      const torch::Tensor chosen = packed_targets.select(1, 0)
          .to(torch::kInt64).reshape({-1, 1, 1})
          .expand({-1, 1, kVrpoNumSeats});
      const torch::Tensor predicted = q_output.gather(1, chosen).squeeze(1);
      const torch::Tensor target_tensor = packed_targets.slice(1, 1, 5);
      const torch::Tensor loss = 0.5 * torch::mse_loss(predicted,
                                                       target_tensor);
      if (!vrpo_training_internal::FiniteTensor(loss)) {
        return fail_after_capture("Q-warmup loss is nonfinite");
      }
      const auto range_cpu = torch::stack(
          {predicted.detach().min(), predicted.detach().max()})
          .to(torch::kFloat64).cpu();
      stats.prediction_min = std::min(
          stats.prediction_min, range_cpu[0].item<double>());
      stats.prediction_max = std::max(
          stats.prediction_max, range_cpu[1].item<double>());
      loss.backward();
      ++stats.q_backward_calls;
      double actor_grad_norm = 0.0;
      double q_grad_norm = 0.0;
      bool actor_grad = false;
      bool q_grad = false;
      if (!vrpo_training_internal::GradientNorm(
              actor_model, "", &actor_grad_norm, &actor_grad, error) ||
          !vrpo_training_internal::GradientNorm(
              q_model, "", &q_grad_norm, &q_grad, error)) {
        return fail_after_capture(*error);
      }
      if (actor_grad || actor_grad_norm != 0.0 || !q_grad ||
          q_grad_norm <= 0.0 || !std::isfinite(q_grad_norm)) {
        return fail_after_capture("Q-warmup actor/Q gradient gate failed");
      }
      stats.max_q_grad_norm = std::max(stats.max_q_grad_norm, q_grad_norm);
      torch::nn::utils::clip_grad_norm_(q_model.parameters(),
                                       kVrpoTrainingGradientClipNorm);
      q_optimizer.step();
      ++stats.q_optimizer_steps;
      stats.q_rows_seen += flat.rows.size();
      loss_sum += loss.detach().item<double>();
      ++loss_count;
    }
  }
  q_optimizer.zero_grad();
  if (!vrpo_q_warmup_internal::RestoreActorRuntime(
          actor_model, actor_requires_grad, actor_was_training, error)) {
    return fail_after_capture(*error);
  }
  stats.actor_state_after_sha256 =
      vrpo_q_warmup_internal::ModuleParametersAndBuffersSha256(
          actor_model, error);
  std::string optimizer_after;
  if (stats.actor_state_after_sha256.empty() ||
      stats.actor_state_after_sha256 != stats.actor_state_before_sha256 ||
      !vrpo_training_internal::ModuleValueSha256(
          q_model, "", &stats.q_values_after_sha256, error) ||
      stats.q_values_after_sha256 == stats.q_values_before_sha256 ||
      !vrpo_q_warmup_internal::QOptimizerStateSha256(
          q_optimizer, q_model,
          expected_q_step_before + kVrpoQWarmupStepsPerUpdate,
          &optimizer_after, error) ||
      stats.q_optimizer_steps != kVrpoQWarmupStepsPerUpdate ||
      stats.q_backward_calls != kVrpoQWarmupStepsPerUpdate ||
      loss_count != kVrpoQWarmupStepsPerUpdate || stats.q_rows_seen <= 0 ||
      !std::isfinite(stats.target_min) ||
      !std::isfinite(stats.target_max) ||
      !std::isfinite(stats.prediction_min) ||
      !std::isfinite(stats.prediction_max) ||
      !std::isfinite(stats.max_q_grad_norm)) {
    return fail_after_capture(error != nullptr && !error->empty()
                                  ? *error
                                  : "Q-warmup postconditions failed");
  }
  stats.q_loss_mean = loss_sum / loss_count;
  if (!std::isfinite(stats.q_loss_mean)) {
    return fail_after_capture("Q-warmup mean loss is nonfinite");
  }
  stats.success = true;
  stats.deterministic_summary_sha256 =
      vrpo_q_warmup_internal::UpdateStatsSha256(stats);
  transaction.Commit();
  *output = std::move(stats);
  return true;
}

struct VrpoQWarmupMetric {
  int64_t count = 0;
  double mse = 0.0;
  double mae = 0.0;
  double pearson = 0.0;
  double prediction_min = 0.0;
  double prediction_max = 0.0;
  double target_min = 0.0;
  double target_max = 0.0;
  bool zero_variance = false;
  bool finite = false;
};

inline bool ComputeVrpoQWarmupMetric(
    const std::vector<double>& predictions,
    const std::vector<double>& targets, VrpoQWarmupMetric* output,
    std::string* error) {
  if (output == nullptr || predictions.empty() ||
      predictions.size() != targets.size()) {
    if (error != nullptr) *error = "Q-warmup metric inputs are invalid";
    return false;
  }
  VrpoQWarmupMetric metric;
  metric.count = static_cast<int64_t>(predictions.size());
  metric.prediction_min = *std::min_element(predictions.begin(), predictions.end());
  metric.prediction_max = *std::max_element(predictions.begin(), predictions.end());
  metric.target_min = *std::min_element(targets.begin(), targets.end());
  metric.target_max = *std::max_element(targets.begin(), targets.end());
  const double prediction_mean =
      std::accumulate(predictions.begin(), predictions.end(), 0.0) /
      predictions.size();
  const double target_mean =
      std::accumulate(targets.begin(), targets.end(), 0.0) /
      targets.size();
  double prediction_ss = 0.0;
  double target_ss = 0.0;
  double cross = 0.0;
  for (size_t i = 0; i < predictions.size(); ++i) {
    if (!std::isfinite(predictions[i]) || !std::isfinite(targets[i])) {
      if (error != nullptr) *error = "Q-warmup metric input is nonfinite";
      return false;
    }
    const double residual = predictions[i] - targets[i];
    metric.mse += residual * residual;
    metric.mae += std::abs(residual);
    const double p = predictions[i] - prediction_mean;
    const double t = targets[i] - target_mean;
    prediction_ss += p * p;
    target_ss += t * t;
    cross += p * t;
  }
  metric.mse /= predictions.size();
  metric.mae /= predictions.size();
  metric.zero_variance = prediction_ss <= 1e-30 || target_ss <= 1e-30;
  metric.pearson = metric.zero_variance
      ? 0.0
      : cross / std::sqrt(prediction_ss * target_ss);
  metric.finite = std::isfinite(metric.mse) && std::isfinite(metric.mae) &&
      std::isfinite(metric.pearson) &&
      std::isfinite(metric.prediction_min) &&
      std::isfinite(metric.prediction_max) &&
      std::isfinite(metric.target_min) && std::isfinite(metric.target_max);
  if (!metric.finite) {
    if (error != nullptr) *error = "Q-warmup metric result is nonfinite";
    return false;
  }
  *output = metric;
  return true;
}

struct VrpoQWarmupMetricSet {
  VrpoQWarmupMetric chosen_q;
  VrpoQWarmupMetric policy_v;
  std::array<VrpoQWarmupMetric, kVrpoNumSeats> chosen_q_by_seat;
  std::array<VrpoQWarmupMetric, kVrpoNumSeats> policy_v_by_seat;
};

struct VrpoQWarmupHoldoutEvaluation {
  int64_t rows = 0;
  int64_t scalar_targets = 0;
  std::string corpus_payload_sha256;
  std::string row_identity_sha256;
  VrpoQWarmupMetricSet initial;
  VrpoQWarmupMetricSet final;
  bool identical_rows = false;
  bool adapter_exact = false;
  bool all_finite = false;
  bool improvement_pass = false;
  std::string classification;
};

inline bool VrpoQWarmupImprovementPass(
    const VrpoQWarmupMetricSet& initial,
    const VrpoQWarmupMetricSet& final, bool identical_rows,
    bool adapter_exact, bool all_finite) {
  return all_finite && identical_rows && adapter_exact &&
      final.chosen_q.mse <=
          kVrpoQWarmupImprovementRatio * initial.chosen_q.mse &&
      final.policy_v.mse <=
          kVrpoQWarmupImprovementRatio * initial.policy_v.mse &&
      final.chosen_q.pearson + kVrpoQWarmupPearsonTolerance >=
          initial.chosen_q.pearson &&
      final.policy_v.pearson + kVrpoQWarmupPearsonTolerance >=
          initial.policy_v.pearson;
}

namespace vrpo_q_warmup_internal {

struct MetricAccumulator {
  std::vector<double> chosen;
  std::vector<double> policy;
  std::vector<double> targets;
  std::array<std::vector<double>, kVrpoNumSeats> chosen_by_seat;
  std::array<std::vector<double>, kVrpoNumSeats> policy_by_seat;
  std::array<std::vector<double>, kVrpoNumSeats> targets_by_seat;
};

inline bool FinishMetricSet(const MetricAccumulator& accumulator,
                            VrpoQWarmupMetricSet* output,
                            std::string* error) {
  if (output == nullptr ||
      !ComputeVrpoQWarmupMetric(accumulator.chosen, accumulator.targets,
                                &output->chosen_q, error) ||
      !ComputeVrpoQWarmupMetric(accumulator.policy, accumulator.targets,
                                &output->policy_v, error)) {
    return false;
  }
  for (int seat = 0; seat < kVrpoNumSeats; ++seat) {
    if (!ComputeVrpoQWarmupMetric(
            accumulator.chosen_by_seat[seat],
            accumulator.targets_by_seat[seat],
            &output->chosen_q_by_seat[seat], error) ||
        !ComputeVrpoQWarmupMetric(
            accumulator.policy_by_seat[seat],
            accumulator.targets_by_seat[seat],
            &output->policy_v_by_seat[seat], error)) {
      return false;
    }
  }
  return true;
}

inline bool EvaluateOneQ(
    const std::vector<VrpoTrainingEpisode>& episodes,
    torch::nn::Module& actor, torch::nn::Module& q,
    const VrpoActorForward& actor_forward, const VrpoQForward& q_forward,
    MetricAccumulator* accumulator, std::string* row_identity,
    bool* adapter_exact, std::string* error) {
  if (accumulator == nullptr || row_identity == nullptr ||
      adapter_exact == nullptr) {
    if (error != nullptr) *error = "Q-warmup holdout output is incomplete";
    return false;
  }
  std::vector<size_t> all(episodes.size());
  std::iota(all.begin(), all.end(), 0);
  const auto flat = vrpo_training_internal::FlattenEpisodes(episodes, all);
  vrpo_training_internal::ModuleRuntimePlacement actor_placement;
  vrpo_training_internal::ModuleRuntimePlacement q_placement;
  if (!vrpo_training_internal::CapturePhase4dRuntimePlacements(
          actor, q, &actor_placement, &q_placement, error)) {
    return false;
  }
  std::string identity_payload = "dune_vrpo_q_warmup_holdout_rows_v1";
  torch::NoGradGuard no_grad;
  constexpr size_t kChunkRows = 128;
  for (size_t begin = 0; begin < flat.rows.size(); begin += kChunkRows) {
    const size_t end = std::min(flat.rows.size(), begin + kChunkRows);
    std::vector<const VrpoTrainingRow*> rows(flat.rows.begin() + begin,
                                             flat.rows.begin() + end);
    torch::Tensor actor_inputs;
    torch::Tensor q_inputs;
    if (!vrpo_training_internal::StackInputs(
            rows, true, actor_placement, &actor_inputs, error) ||
        !vrpo_training_internal::StackInputs(
            rows, false, q_placement, &q_inputs, error)) {
      return false;
    }
    VrpoActorTrainingOutput actor_output;
    torch::Tensor q_output;
    try {
      actor_output = actor_forward(actor_inputs);
      q_output = q_forward(q_inputs);
    } catch (const std::exception& exception) {
      if (error != nullptr) {
        *error = std::string("Q-warmup holdout forward failed: ") +
                 exception.what();
      }
      return false;
    }
    if (!vrpo_training_internal::ValidateActorOutput(
            actor_output, rows.size(), actor_placement, error) ||
        !vrpo_training_internal::ValidateQOutput(
            q_output, rows.size(), q_placement, error)) {
      return false;
    }
    std::vector<torch::Tensor> ignored_log;
    std::vector<torch::Tensor> ignored_prob;
    std::vector<std::vector<double>> probabilities;
    vrpo_training_internal::FlatBatch chunk_flat;
    chunk_flat.rows = rows;
    const VrpoPhase4ArmConfig* arm =
        FindCanonicalVrpoPhase4Arm("VRPO_CAP10");
    if (arm == nullptr ||
        !vrpo_training_internal::BuildLegalPolicies(
            actor_output, chunk_flat, arm->logit_cap, &ignored_log,
            &ignored_prob, &probabilities, error)) {
      return false;
    }
    const torch::Tensor q_cpu =
        q_output.detach().contiguous().cpu().to(torch::kFloat32);
    const auto values = q_cpu.accessor<float, 3>();
    for (size_t local = 0; local < rows.size(); ++local) {
      const VrpoTrainingRow& row = *rows[local];
      VrpoActorRelativeSeatValues chosen_relative;
      VrpoActorRelativeSeatValues policy_relative;
      for (int slot = 0; slot < kVrpoNumSeats; ++slot) {
        chosen_relative.slots[slot] =
            values[local][row.chosen_action][slot];
        double expected = 0.0;
        for (size_t legal = 0; legal < row.legal_actions.size(); ++legal) {
          expected += probabilities[local][legal] *
              values[local][row.legal_actions[legal]][slot];
        }
        policy_relative.slots[slot] = expected;
      }
      VrpoSeatValues chosen_absolute;
      VrpoSeatValues policy_absolute;
      if (!VrpoActorRelativeToAbsoluteSeatValues(
              row.actor, chosen_relative, &chosen_absolute, error) ||
          !VrpoActorRelativeToAbsoluteSeatValues(
              row.actor, policy_relative, &policy_absolute, error)) {
        return false;
      }
      const auto episode_it = std::find_if(
          episodes.begin(), episodes.end(), [&](const auto& episode) {
            return episode.episode_id == row.episode_id;
          });
      if (episode_it == episodes.end() || episode_it->rows.empty()) {
        if (error != nullptr) *error = "Q-warmup holdout episode lookup failed";
        return false;
      }
      const VrpoSeatValues actual = episode_it->rows.back().rewards;
      vrpo_capture_internal::AppendPod(&identity_payload, row.row_id);
      vrpo_capture_internal::AppendPod(&identity_payload, row.episode_id);
      vrpo_capture_internal::AppendPod(&identity_payload, row.step_index);
      vrpo_capture_internal::AppendPod(&identity_payload, row.actor);
      vrpo_capture_internal::AppendPod(&identity_payload, row.chosen_action);
      for (int seat = 0; seat < kVrpoNumSeats; ++seat) {
        accumulator->chosen.push_back(chosen_absolute[seat]);
        accumulator->policy.push_back(policy_absolute[seat]);
        accumulator->targets.push_back(actual[seat]);
        accumulator->chosen_by_seat[seat].push_back(chosen_absolute[seat]);
        accumulator->policy_by_seat[seat].push_back(policy_absolute[seat]);
        accumulator->targets_by_seat[seat].push_back(actual[seat]);
        // An independent inverse check prevents a same-bug comparison from
        // silently blessing the relative-to-absolute adapter.
        const int slot = (seat - row.actor + kVrpoNumSeats) % kVrpoNumSeats;
        if (chosen_absolute[seat] != chosen_relative.slots[slot] ||
            policy_absolute[seat] != policy_relative.slots[slot]) {
          if (error != nullptr) *error = "Q-warmup seat adapter mismatch";
          return false;
        }
      }
    }
  }
  *row_identity = ComputeStringSHA256(identity_payload);
  *adapter_exact = true;
  return true;
}

inline json::Object MetricJson(const VrpoQWarmupMetric& metric) {
  json::Object out;
  out["count"] = metric.count;
  out["mse"] = VrpoPhase4eCanonicalDouble(metric.mse);
  out["mae"] = VrpoPhase4eCanonicalDouble(metric.mae);
  out["pearson"] = VrpoPhase4eCanonicalDouble(metric.pearson);
  out["prediction_min"] = VrpoPhase4eCanonicalDouble(metric.prediction_min);
  out["prediction_max"] = VrpoPhase4eCanonicalDouble(metric.prediction_max);
  out["target_min"] = VrpoPhase4eCanonicalDouble(metric.target_min);
  out["target_max"] = VrpoPhase4eCanonicalDouble(metric.target_max);
  out["zero_variance"] = metric.zero_variance;
  out["finite"] = metric.finite;
  return out;
}

inline json::Object MetricSetJson(const VrpoQWarmupMetricSet& metrics) {
  json::Object out;
  out["chosen_q"] = json::Value(MetricJson(metrics.chosen_q));
  out["policy_v"] = json::Value(MetricJson(metrics.policy_v));
  json::Array chosen_by_seat;
  json::Array policy_by_seat;
  for (int seat = 0; seat < kVrpoNumSeats; ++seat) {
    chosen_by_seat.emplace_back(MetricJson(metrics.chosen_q_by_seat[seat]));
    policy_by_seat.emplace_back(MetricJson(metrics.policy_v_by_seat[seat]));
  }
  out["chosen_q_by_absolute_seat"] = std::move(chosen_by_seat);
  out["policy_v_by_absolute_seat"] = std::move(policy_by_seat);
  return out;
}

}  // namespace vrpo_q_warmup_internal

inline bool EvaluateVrpoQWarmupHoldout(
    const std::vector<VrpoTrainingEpisode>& episodes,
    const std::string& corpus_payload_sha256,
    torch::nn::Module& actor_model, torch::nn::Module& initial_q_model,
    torch::nn::Module& final_q_model,
    const VrpoActorForward& actor_forward,
    const VrpoQForward& initial_q_forward,
    const VrpoQForward& final_q_forward,
    VrpoQWarmupHoldoutEvaluation* output, std::string* error) {
  auto fail = [&](const std::string& message) {
    if (error != nullptr) *error = message;
    if (output != nullptr) *output = VrpoQWarmupHoldoutEvaluation{};
    return false;
  };
  if (output == nullptr || episodes.size() != kVrpoQWarmupHoldoutGames ||
      episodes.front().episode_id != kVrpoQWarmupHoldoutStartEpisodeId ||
      episodes.back().episode_id != kVrpoQWarmupHoldoutEndEpisodeIdInclusive ||
      !VrpoPhase4eLowerHex64(corpus_payload_sha256)) {
    return fail("Q-warmup holdout corpus/ID contract rejected");
  }
  VrpoQWarmupHoldoutEvaluation evaluation;
  evaluation.corpus_payload_sha256 = corpus_payload_sha256;
  vrpo_q_warmup_internal::MetricAccumulator initial_accumulator;
  vrpo_q_warmup_internal::MetricAccumulator final_accumulator;
  std::string initial_rows;
  std::string final_rows;
  bool initial_adapter = false;
  bool final_adapter = false;
  if (!vrpo_q_warmup_internal::EvaluateOneQ(
          episodes, actor_model, initial_q_model, actor_forward,
          initial_q_forward, &initial_accumulator, &initial_rows,
          &initial_adapter, error) ||
      !vrpo_q_warmup_internal::EvaluateOneQ(
          episodes, actor_model, final_q_model, actor_forward,
          final_q_forward, &final_accumulator, &final_rows,
          &final_adapter, error) ||
      initial_rows != final_rows ||
      initial_accumulator.targets != final_accumulator.targets ||
      !vrpo_q_warmup_internal::FinishMetricSet(
          initial_accumulator, &evaluation.initial, error) ||
      !vrpo_q_warmup_internal::FinishMetricSet(
          final_accumulator, &evaluation.final, error)) {
    return fail(error != nullptr && !error->empty()
                    ? *error
                    : "Q-warmup heldout initial/final pairing rejected");
  }
  evaluation.rows = static_cast<int64_t>(
      initial_accumulator.targets.size() / kVrpoNumSeats);
  evaluation.scalar_targets = initial_accumulator.targets.size();
  evaluation.row_identity_sha256 = initial_rows;
  evaluation.identical_rows = true;
  evaluation.adapter_exact = initial_adapter && final_adapter;
  auto finite_set = [](const VrpoQWarmupMetricSet& set) {
    if (!set.chosen_q.finite || !set.policy_v.finite) return false;
    for (int seat = 0; seat < kVrpoNumSeats; ++seat) {
      if (!set.chosen_q_by_seat[seat].finite ||
          !set.policy_v_by_seat[seat].finite) return false;
    }
    return true;
  };
  evaluation.all_finite = finite_set(evaluation.initial) &&
                          finite_set(evaluation.final);
  evaluation.improvement_pass = VrpoQWarmupImprovementPass(
      evaluation.initial, evaluation.final, evaluation.identical_rows,
      evaluation.adapter_exact, evaluation.all_finite);
  evaluation.classification = evaluation.improvement_pass
      ? "VALID_Q_WARMUP_HOLDOUT_IMPROVEMENT"
      : "VALID_Q_WARMUP_NO_IMPROVEMENT";
  *output = std::move(evaluation);
  return true;
}

struct VrpoQWarmupState {
  int completed_updates = 0;
  int64_t q_optimizer_steps = 0;
  int64_t measured_peak_bytes = 0;
  std::string actor_state_sha256;
  vrpo_q_warmup_internal::QOriginIdentity q_origin_identity;
  std::string initial_q_runtime_module_values_sha256;
  std::string latest_q_runtime_module_values_sha256;
  std::string latest_q_optimizer_state_sha256;
  std::vector<VrpoPpoPilotEvidenceObservation> evidence_observations;
  json::Array committed_updates;
  bool runtime_device_exact_match = false;
  bool owns_output_root = false;
  bool completed = false;
  bool had_failure = false;
  std::string failure_reason;
};

enum class VrpoQWarmupFailurePoint {
  kNone,
  kLateAfterTraining,
  kAfterQTemp,
  kAfterOptimizerTemp,
  kAfterReload,
  kBeforeHoldoutStatus,
  kForceFinalByteOverrunForTest,
  kForceFinalDeadlineOverrunForTest,
};

namespace vrpo_q_warmup_internal {

inline json::Object EpisodeEvidenceJson(
    const VrpoPhase4ePairingStats& pairing, const std::string& schema,
    int update, bool holdout) {
  json::Object evidence;
  evidence["schema"] = schema;
  evidence["update"] = static_cast<int64_t>(update);
  evidence["holdout"] = holdout;
  evidence["episodes"] = pairing.episodes;
  evidence["capture_rows"] = pairing.capture_rows;
  evidence["rollout_rows"] = pairing.rollout_rows;
  evidence["paired_rows"] = pairing.paired_rows;
  evidence["pairing_sha256"] = pairing.canonical_sha256;
  evidence["outcome_convention"] = kVrpoPhase4eOutcomeConvention;
  evidence["reward_scale_exact"] = VrpoPhase4eCanonicalDouble(4.0);
  json::Array records;
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
    records.emplace_back(std::move(item));
  }
  evidence["episode_evidence"] = std::move(records);
  return evidence;
}

inline bool SaveQAndOptimizer(
    torch::nn::Module& q, torch::optim::AdamW& optimizer,
    const std::filesystem::path& q_path,
    const std::filesystem::path& optimizer_path, std::string* error) {
  try {
    SaveVrpoModule(q, q_path);
    if (!VrpoFsyncFile(q_path, error)) return false;
    torch::save(optimizer, optimizer_path.string());
    if (!VrpoFsyncFile(optimizer_path, error)) return false;
    return true;
  } catch (const std::exception& exception) {
    if (error != nullptr) {
      *error = std::string("Q-warmup archive save failed: ") +
               exception.what();
    }
    return false;
  }
}

inline bool ReloadQAndOptimizerExact(
    const std::filesystem::path& q_path,
    const std::filesystem::path& optimizer_path,
    const torch::Device& device, int64_t expected_step,
    const std::string& expected_q_values_sha256,
    const std::string& expected_optimizer_sha256,
    const VrpoExpandedExpectedLayout& layout, std::string* error) {
  auto reloaded = std::make_shared<DuneVrpoQNetImpl>(kVrpoQWarmupQSeed);
  reloaded->to(device, torch::kFloat32);
  std::unique_ptr<torch::optim::AdamW> optimizer;
  if (!MakeFreshQOptimizer(*reloaded, &optimizer, error) ||
      !LoadVrpoModule(*reloaded, q_path, error)) {
    return false;
  }
  try {
    torch::load(*optimizer, optimizer_path.string());
  } catch (const std::exception& exception) {
    if (error != nullptr) {
      *error = std::string("Q-warmup optimizer reload failed: ") +
               exception.what();
    }
    return false;
  }
  if (!vrpo_training_internal::MigrateAdamWOptimizerStateToParameterDevices(
          *optimizer, error)) {
    return false;
  }
  std::string q_hash;
  std::string optimizer_hash;
  if (!vrpo_training_internal::ModuleValueSha256(
          *reloaded, "", &q_hash, error) ||
      !QOptimizerStateSha256(*optimizer, *reloaded, expected_step,
                             &optimizer_hash, error) ||
      q_hash != expected_q_values_sha256 ||
      optimizer_hash != expected_optimizer_sha256) {
    if (error != nullptr && error->empty()) {
      *error = "Q-warmup strict reload identity mismatch";
    }
    return false;
  }
  const VrpoPhase4eCanary canary = MakeVrpoPhase4eCanary(layout);
  torch::Tensor first;
  torch::Tensor second;
  try {
    if (!reloaded->ForwardChecked(canary.q_input.to(device), &first, error) ||
        !reloaded->ForwardChecked(canary.q_input.to(device), &second, error)) {
      return false;
    }
  } catch (const std::exception& exception) {
    if (error != nullptr) {
      *error = std::string("Q-warmup reload canary failed: ") +
               exception.what();
    }
    return false;
  }
  return torch::equal(first, second) &&
      VrpoPhase4eTensorSha256(first, "q_canary") ==
          VrpoPhase4eTensorSha256(second, "q_canary");
}

inline json::Object FileIdentityJson(const std::filesystem::path& path) {
  size_t size = 0;
  json::Object out;
  out["filename"] = path.filename().string();
  out["sha256"] = ComputeFileSHA256(path.string(), &size);
  out["size"] = static_cast<int64_t>(size);
  return out;
}

inline bool WriteChecksums(
    const std::filesystem::path& path,
    const std::vector<std::filesystem::path>& files,
    const std::string& registration_id, int update, bool holdout,
    std::string* error) {
  json::Object root;
  root["schema"] = kVrpoQWarmupChecksumsSchema;
  root["registration_id"] = registration_id;
  root["update"] = static_cast<int64_t>(update);
  root["holdout"] = holdout;
  json::Array records;
  for (const auto& file : files) {
    records.emplace_back(FileIdentityJson(file));
  }
  root["files"] = std::move(records);
  return vrpo_ppo_pilot_internal::WriteAtomicText(
      path, json::ToString(root, true) + "\n", error);
}

inline bool ValidateCommittedUpdateDirectory(
    const std::filesystem::path& directory, std::string* error) {
  const std::set<std::string> expected = {
      "Q_CORPUS.bin", "EPISODE_EVIDENCE.json", "q_model.pt",
      "q_optimizer.pt", "UPDATE_RESULT.json", "CHECKSUMS.json"};
  std::set<std::string> actual;
  if (!std::filesystem::is_directory(directory)) {
    if (error != nullptr) *error = "Q-warmup committed update is missing";
    return false;
  }
  for (const auto& entry : std::filesystem::directory_iterator(directory)) {
    if (!entry.is_regular_file() || entry.is_symlink()) {
      if (error != nullptr) *error = "Q-warmup update contains nonregular file";
      return false;
    }
    actual.insert(entry.path().filename().string());
  }
  if (actual != expected) {
    if (error != nullptr) *error = "Q-warmup update file set is partial/extra";
    return false;
  }
  json::Object checksums;
  if (!ReadJsonObject(directory / "CHECKSUMS.json", &checksums, error)) {
    return false;
  }
  const auto files = checksums.find("files");
  if (files == checksums.end() || !files->second.IsArray() ||
      files->second.GetArray().size() != 5) {
    if (error != nullptr) *error = "Q-warmup checksums file is malformed";
    return false;
  }
  const std::set<std::string> expected_recorded = {
      "Q_CORPUS.bin", "EPISODE_EVIDENCE.json", "q_model.pt",
      "q_optimizer.pt", "UPDATE_RESULT.json"};
  std::set<std::string> recorded;
  for (const auto& value : files->second.GetArray()) {
    if (!value.IsObject()) return false;
    const auto& item = value.GetObject();
    const auto name = item.find("filename");
    const auto sha = item.find("sha256");
    const auto size = item.find("size");
    if (name == item.end() || !name->second.IsString() ||
        sha == item.end() || !sha->second.IsString() ||
        size == item.end() || !size->second.IsInt()) {
      return false;
    }
    if (!recorded.insert(name->second.GetString()).second) {
      if (error != nullptr) *error = "Q-warmup checksums repeat a file";
      return false;
    }
    const auto file = directory / name->second.GetString();
    size_t observed_size = 0;
    const std::string observed = ComputeFileSHA256(file.string(),
                                                   &observed_size);
    if (observed != sha->second.GetString() ||
        static_cast<int64_t>(observed_size) != size->second.GetInt()) {
      if (error != nullptr) *error = "Q-warmup committed checksum mismatch";
      return false;
    }
  }
  if (recorded != expected_recorded) {
    if (error != nullptr) *error = "Q-warmup checksums omit/replace a file";
    return false;
  }
  return true;
}

inline bool ValidateCommittedHoldoutDirectory(
    const std::filesystem::path& directory, std::string* error) {
  const std::set<std::string> expected = {
      "HOLDOUT_CORPUS.bin", "HOLDOUT_EVIDENCE.json",
      "HOLDOUT_METRICS.json", "CHECKSUMS.json"};
  std::set<std::string> actual;
  if (!std::filesystem::is_directory(directory)) {
    if (error != nullptr) *error = "Q-warmup committed holdout is missing";
    return false;
  }
  for (const auto& entry : std::filesystem::directory_iterator(directory)) {
    if (!entry.is_regular_file() || entry.is_symlink()) return false;
    actual.insert(entry.path().filename().string());
  }
  if (actual != expected) {
    if (error != nullptr) *error = "Q-warmup holdout file set is partial/extra";
    return false;
  }
  json::Object checksums;
  if (!ReadJsonObject(directory / "CHECKSUMS.json", &checksums, error)) {
    return false;
  }
  const auto files = checksums.find("files");
  const std::set<std::string> expected_recorded = {
      "HOLDOUT_CORPUS.bin", "HOLDOUT_EVIDENCE.json",
      "HOLDOUT_METRICS.json"};
  if (files == checksums.end() || !files->second.IsArray() ||
      files->second.GetArray().size() != expected_recorded.size()) {
    return false;
  }
  std::set<std::string> recorded;
  for (const auto& value : files->second.GetArray()) {
    if (!value.IsObject()) return false;
    const auto& item = value.GetObject();
    const auto name = item.find("filename");
    const auto sha = item.find("sha256");
    const auto size = item.find("size");
    if (name == item.end() || !name->second.IsString() ||
        sha == item.end() || !sha->second.IsString() ||
        size == item.end() || !size->second.IsInt() ||
        !recorded.insert(name->second.GetString()).second) {
      return false;
    }
    size_t observed_size = 0;
    const auto path = directory / name->second.GetString();
    if (ComputeFileSHA256(path.string(), &observed_size) !=
            sha->second.GetString() ||
        static_cast<int64_t>(observed_size) != size->second.GetInt()) {
      return false;
    }
  }
  return recorded == expected_recorded;
}

}  // namespace vrpo_q_warmup_internal

inline bool InitializeVrpoQWarmupFromLoadedState(
    const VrpoQWarmupStartupConfig& startup,
    const std::shared_ptr<SharedDunePolicyValueNetImpl>& actor,
    const std::shared_ptr<DuneVrpoQNetImpl>& q,
    torch::optim::AdamW& q_optimizer, const torch::Device& selected_device,
    const VrpoQWarmupDeadline& deadline, VrpoQWarmupState* state,
    std::string* error, bool test_fixture = false,
    const vrpo_q_warmup_internal::QOriginIdentity* prevalidated_q_origin =
        nullptr) {
  auto fail = [&](const std::string& message) {
    if (error != nullptr) *error = message;
    if (state != nullptr) *state = VrpoQWarmupState{};
    return false;
  };
  if (state == nullptr || actor == nullptr || q == nullptr ||
      !ValidateVrpoQWarmupStartupConfig(startup, error, test_fixture) ||
      !deadline.Check("before Q-warmup initialization", error) ||
      !vrpo_ppo_pilot_internal::CheckFreeSpaceBeforeStart(
          startup.output_root, error)) {
    return fail(error != nullptr && !error->empty()
                    ? *error
                    : "Q-warmup initialization rejected");
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
    return fail(error != nullptr ? *error : "Q-warmup placement rejected");
  }
  const bool actor_device_ok =
      actor_placement.device.type() == selected_device.type() &&
      (!selected_device.is_cuda() ||
       actor_placement.device.index() == selected_cuda_index);
  const bool q_device_ok =
      q_placement.device.type() == selected_device.type() &&
      (!selected_device.is_cuda() ||
       q_placement.device.index() == selected_cuda_index);
  if (actor_placement.dtype != torch::kFloat32 ||
      q_placement.dtype != torch::kFloat32 || !actor_device_ok ||
      !q_device_ok ||
      (!test_fixture && (!selected_device.is_cuda() ||
                         startup.runtime_device_index != selected_cuda_index))) {
    return fail("Q-warmup actor/Q must be uniform FP32 on selected CUDA");
  }
  VrpoQWarmupState initialized;
  initialized.actor_state_sha256 =
      vrpo_q_warmup_internal::ModuleParametersAndBuffersSha256(*actor, error);
  std::string actor_values;
  std::string current_q_canonical_values;
  std::string current_q_runtime_values;
  VrpoExpandedExpectedLayout canonical_q_layout;
  canonical_q_layout.test_fixture = false;
  if (initialized.actor_state_sha256.empty() ||
      !vrpo_training_internal::ModuleValueSha256(
          *actor, "", &actor_values, error) ||
      !VrpoExpandedQValueIdentitySha256(
          *q, canonical_q_layout, &current_q_canonical_values, error) ||
      !vrpo_training_internal::ModuleValueSha256(
          *q, "", &current_q_runtime_values, error) ||
      !vrpo_q_warmup_internal::QOptimizerStateSha256(
          q_optimizer, *q, 0,
          &initialized.latest_q_optimizer_state_sha256, error) ||
      (!test_fixture && actor_values != kVrpoQWarmupActorValuesSha256) ||
      (!test_fixture &&
       (prevalidated_q_origin == nullptr ||
        prevalidated_q_origin->q_model_file_sha256 !=
            kVrpoQWarmupInitialQFileSha256 ||
        prevalidated_q_origin->q_canonical_values_sha256 !=
            kVrpoQWarmupInitialQCanonicalValuesSha256 ||
        prevalidated_q_origin->q_runtime_module_values_sha256 !=
            kVrpoQWarmupInitialQRuntimeValuesSha256 ||
        current_q_canonical_values !=
            prevalidated_q_origin->q_canonical_values_sha256 ||
        current_q_runtime_values !=
            prevalidated_q_origin->q_runtime_module_values_sha256))) {
    return fail(error != nullptr && !error->empty()
                    ? *error
                    : "Q-warmup exact actor/Q/optimizer origin rejected");
  }
  if (prevalidated_q_origin != nullptr) {
    initialized.q_origin_identity = *prevalidated_q_origin;
  } else {
    initialized.q_origin_identity.q_canonical_values_schema =
        kVrpoQWarmupCanonicalQOriginSchema;
    initialized.q_origin_identity.q_canonical_values_sha256 =
        current_q_canonical_values;
    initialized.q_origin_identity.q_runtime_module_values_schema =
        kVrpoQWarmupRuntimeQStateSchema;
    initialized.q_origin_identity.q_runtime_module_values_sha256 =
        current_q_runtime_values;
  }
  initialized.initial_q_runtime_module_values_sha256 =
      current_q_runtime_values;
  initialized.latest_q_runtime_module_values_sha256 =
      current_q_runtime_values;
  initialized.evidence_observations = startup.prior.observations;
  initialized.runtime_device_exact_match = true;
  bool owns_root = false;
  if (!vrpo_q_warmup_internal::ClaimFreshDirectory(
          startup.output_root, &owns_root, error) ||
      !vrpo_q_warmup_internal::ObserveBytes(
          startup.output_root, &initialized.measured_peak_bytes, error)) {
    if (owns_root) {
      vrpo_ppo_pilot_internal::CleanupPath(startup.output_root);
      owns_root = false;
    }
    return fail(error != nullptr && !error->empty()
                    ? *error
                    : "Q-warmup fresh output root creation failed");
  }
  // There is deliberately no actor optimizer construction on this path.
  for (auto& item : actor->named_parameters()) {
    item.value().set_requires_grad(false);
    item.value().mutable_grad() = torch::Tensor();
  }
  actor->eval();
  initialized.owns_output_root = true;
  *state = std::move(initialized);
  owns_root = false;
  return true;
}

inline bool LoadAndInitializeVrpoQWarmup(
    const VrpoQWarmupStartupConfig& startup,
    const std::shared_ptr<SharedDunePolicyValueNetImpl>& actor,
    const std::shared_ptr<DuneVrpoQNetImpl>& q,
    const torch::Device& device, const VrpoQWarmupDeadline& deadline,
    std::unique_ptr<torch::optim::AdamW>* q_optimizer,
    VrpoQWarmupState* state, std::string* error,
    bool test_fixture = false) {
  auto fail = [&](const std::string& message) {
    if (error != nullptr) *error = message;
    if (q_optimizer != nullptr) q_optimizer->reset();
    return false;
  };
  if (actor == nullptr || q == nullptr || q_optimizer == nullptr ||
      state == nullptr || !deadline.Check("before exact origins load", error)) {
    return fail(error != nullptr && !error->empty()
                    ? *error
                    : "Q-warmup origin-load arguments invalid");
  }
  size_t actor_size = 0;
  size_t q_size = 0;
  size_t q_optimizer_size = 0;
  const std::string actor_sha = ComputeFileSHA256(
      startup.prior.actor_path.string(), &actor_size);
  const std::string q_sha = ComputeFileSHA256(
      startup.prior.q_model_path.string(), &q_size);
  const std::string q_optimizer_sha = ComputeFileSHA256(
      startup.prior.q_optimizer_path.string(), &q_optimizer_size);
  if (actor_size == 0 || q_size == 0 || q_optimizer_size == 0 ||
      (!test_fixture && actor_sha != kVrpoQWarmupActorFileSha256) ||
      (!test_fixture && q_sha != kVrpoQWarmupInitialQFileSha256) ||
      (!test_fixture && q_optimizer_sha !=
                            kVrpoQWarmupInitialQOptimizerFileSha256) ||
      !LoadVrpoModule(*actor, startup.prior.actor_path, error) ||
      !LoadVrpoModule(*q, startup.prior.q_model_path, error)) {
    return fail(error != nullptr && !error->empty()
                    ? *error
                    : "Q-warmup exact origins rejected");
  }
  vrpo_q_warmup_internal::QOriginIdentity q_origin;
  if (!test_fixture &&
      !vrpo_q_warmup_internal::ValidateQOriginIdentityBeforeOptimizer(
          *q, startup.prior.q_model_path,
          kVrpoQWarmupInitialQFileSha256,
          kVrpoQWarmupInitialQCanonicalValuesSha256,
          kVrpoQWarmupInitialQRuntimeValuesSha256,
          &q_origin, error)) {
    return fail(error != nullptr && !error->empty()
                    ? *error
                    : "Q-warmup exact Q origin rejected before optimizer");
  }
  if (!vrpo_q_warmup_internal::MakeFreshQOptimizer(
          *q, q_optimizer, error) ||
      !deadline.Check("after exact origins load", error) ||
      !InitializeVrpoQWarmupFromLoadedState(
          startup, actor, q, **q_optimizer, device, deadline, state,
          error, test_fixture, test_fixture ? nullptr : &q_origin)) {
    return fail(error != nullptr && !error->empty()
                    ? *error
                    : "Q-warmup exact origins rejected");
  }
  return true;
}

inline bool WriteVrpoQWarmupUpdate(
    const VrpoQWarmupStartupConfig& startup,
    const VrpoExpandedExpectedLayout& layout,
    const std::vector<VrpoTrainingEpisode>& episodes,
    const VrpoPhase4ePairingStats& pairing,
    const std::shared_ptr<SharedDunePolicyValueNetImpl>& actor,
    const std::shared_ptr<DuneVrpoQNetImpl>& q,
    torch::optim::AdamW& q_optimizer, const torch::Device& device,
    const VrpoQWarmupDeadline& deadline, int update_number,
    VrpoQWarmupFailurePoint failure_point, VrpoQWarmupState* state,
    json::Object* update_result, std::string* error) {
  const auto temp_dir = startup.output_root /
      (".update_" + std::to_string(update_number) + ".tmp");
  const auto final_dir = startup.output_root /
      ("update_" + std::to_string(update_number));
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
  if (state == nullptr || update_result == nullptr || actor == nullptr ||
      q == nullptr || state->had_failure || state->completed ||
      update_number != state->completed_updates + 1 || update_number < 1 ||
      update_number > kVrpoQWarmupUpdates ||
      episodes.size() != kVrpoQWarmupGamesPerUpdate ||
      pairing.episodes != kVrpoQWarmupGamesPerUpdate ||
      std::filesystem::exists(temp_dir) ||
      std::filesystem::exists(final_dir) ||
      !deadline.Check("before Q-only update", error)) {
    return fail(error != nullptr && !error->empty()
                    ? *error
                    : "Q-warmup update sequencing rejected");
  }
  const uint64_t expected_start = startup.start_episode_id +
      static_cast<uint64_t>(update_number - 1) *
          kVrpoQWarmupGamesPerUpdate;
  const uint64_t expected_next = expected_start + kVrpoQWarmupGamesPerUpdate;
  if (episodes.front().episode_id != expected_start ||
      episodes.back().episode_id + 1 != expected_next ||
      pairing.episode_evidence.front().episode_id != expected_start ||
      pairing.episode_evidence.back().episode_id + 1 != expected_next ||
      !ValidateVrpoPhase4ePairingStatsAgainstEpisodes(
          pairing, episodes, error)) {
    return fail(error != nullptr && !error->empty()
                    ? *error
                    : "Q-warmup update IDs/pairing rejected");
  }
  const std::string actor_before =
      vrpo_q_warmup_internal::ModuleParametersAndBuffersSha256(
          *actor, error);
  std::string q_before;
  std::string optimizer_before;
  if (actor_before != state->actor_state_sha256 ||
      !vrpo_training_internal::ModuleValueSha256(
          *q, "", &q_before, error) ||
      q_before != state->latest_q_runtime_module_values_sha256 ||
      !vrpo_q_warmup_internal::QOptimizerStateSha256(
          q_optimizer, *q, state->q_optimizer_steps,
          &optimizer_before, error) ||
      optimizer_before != state->latest_q_optimizer_state_sha256) {
    return fail(error != nullptr && !error->empty()
                    ? *error
                    : "Q-warmup live actor/Q/optimizer binding rejected");
  }
  std::string corpus_bytes;
  VrpoScheduleCorpusIdentity corpus_identity;
  std::vector<VrpoTrainingEpisode> decoded;
  VrpoScheduleCorpusIdentity decoded_identity;
  if (!EncodeVrpoQWarmupCentralCorpus(
          episodes, &corpus_bytes, &corpus_identity, error) ||
      !DecodeVrpoQWarmupCentralCorpus(
          corpus_bytes, &decoded, &decoded_identity, error) ||
      decoded_identity.payload_sha256 != corpus_identity.payload_sha256 ||
      decoded.size() != episodes.size()) {
    return fail(error != nullptr && !error->empty()
                    ? *error
                    : "Q-warmup central corpus seal/reload rejected");
  }
  const uint64_t update_seed = vrpo_training_internal::SplitMix64(
      startup.base_seed ^ expected_start ^
      (0x515741524d555000ULL + static_cast<uint64_t>(update_number)));
  VrpoActorForward actor_forward = [actor](const torch::Tensor& input) {
    torch::NoGradGuard no_grad;
    const auto out = actor->forward(input);
    return VrpoActorTrainingOutput{out.logits.detach(), out.values.detach()};
  };
  VrpoQForward q_forward = [q](const torch::Tensor& input) {
    torch::Tensor output;
    std::string forward_error;
    if (!q->ForwardChecked(input, &output, &forward_error)) {
      throw std::runtime_error("Q-warmup Q forward rejected: " +
                               forward_error);
    }
    return output;
  };
  // The inner mechanics transaction protects failures during optimization.
  // This outer transaction additionally protects every late persistence,
  // reload, evidence, deadline, and resource gate below.
  vrpo_q_warmup_internal::QTransaction outer_transaction;
  if (!outer_transaction.Capture(*q, q_optimizer, error)) {
    return fail(error != nullptr ? *error
                                 : "Q-warmup outer rollback capture failed");
  }
  VrpoQWarmupUpdateStats stats;
  if (!RunVrpoQWarmupQOnlyUpdate(
          decoded, update_seed, *actor, *q, q_optimizer,
          actor_forward, q_forward, state->q_optimizer_steps,
          &stats, error) ||
      !deadline.Check("after Q-only update", error)) {
    return fail(error != nullptr && !error->empty()
                    ? *error
                    : "Q-warmup Q-only update failed");
  }
  if (failure_point == VrpoQWarmupFailurePoint::kLateAfterTraining) {
    return fail("injected Q-warmup late failure after training");
  }
  std::string optimizer_after;
  if (stats.actor_state_after_sha256 != state->actor_state_sha256 ||
      !vrpo_q_warmup_internal::QOptimizerStateSha256(
          q_optimizer, *q,
          state->q_optimizer_steps + kVrpoQWarmupStepsPerUpdate,
          &optimizer_after, error)) {
    return fail(error != nullptr && !error->empty()
                    ? *error
                    : "Q-warmup post-update state rejected");
  }
  if (!vrpo_q_warmup_internal::ClaimFreshDirectory(
          temp_dir, &owns_temp_dir, error)) {
    return fail("Q-warmup update temp directory creation failed");
  }
  const auto corpus_path = temp_dir / "Q_CORPUS.bin";
  const auto evidence_path = temp_dir / "EPISODE_EVIDENCE.json";
  const auto q_path = temp_dir / "q_model.pt";
  const auto optimizer_path = temp_dir / "q_optimizer.pt";
  const auto result_path = temp_dir / "UPDATE_RESULT.json";
  const auto checksums_path = temp_dir / "CHECKSUMS.json";
  if (!vrpo_ppo_pilot_internal::WriteAtomicText(
          corpus_path, corpus_bytes, error) ||
      !vrpo_ppo_pilot_internal::WriteAtomicText(
          evidence_path,
          json::ToString(vrpo_q_warmup_internal::EpisodeEvidenceJson(
                             pairing,
                             "dune_vrpo_q_warmup_episode_evidence_v1",
                             update_number, false),
                         true) + "\n",
          error) ||
      !vrpo_q_warmup_internal::SaveQAndOptimizer(
          *q, q_optimizer, q_path, optimizer_path, error)) {
    return fail(error != nullptr ? *error : "Q-warmup archive write failed");
  }
  if (failure_point == VrpoQWarmupFailurePoint::kAfterQTemp ||
      failure_point == VrpoQWarmupFailurePoint::kAfterOptimizerTemp) {
    return fail("injected Q-warmup archive failure");
  }
  if (!vrpo_q_warmup_internal::ReloadQAndOptimizerExact(
          q_path, optimizer_path, device,
          state->q_optimizer_steps + kVrpoQWarmupStepsPerUpdate,
          stats.q_values_after_sha256, optimizer_after, layout, error)) {
    return fail(error != nullptr ? *error : "Q-warmup reload rejected");
  }
  if (failure_point == VrpoQWarmupFailurePoint::kAfterReload) {
    return fail("injected Q-warmup failure after reload");
  }
  json::Object result;
  result["schema"] = kVrpoQWarmupUpdateSchema;
  result["registration_id"] = startup.registration_id;
  result["profile"] = startup.profile;
  result["update"] = static_cast<int64_t>(update_number);
  result["start_episode_id"] = static_cast<int64_t>(expected_start);
  result["next_episode_id"] = static_cast<int64_t>(expected_next);
  result["games"] = static_cast<int64_t>(episodes.size());
  result["rows"] = pairing.paired_rows;
  result["pairing_sha256"] = pairing.canonical_sha256;
  result["central_corpus_payload_sha256"] = corpus_identity.payload_sha256;
  result["actor_frozen"] = true;
  result["actor_optimizer_constructed"] = false;
  result["actor_optimizer_steps"] = int64_t{0};
  result["actor_backward_calls"] = int64_t{0};
  result["actor_state_before_sha256"] = stats.actor_state_before_sha256;
  result["actor_state_after_sha256"] = stats.actor_state_after_sha256;
  result["q_origin_file_identity_schema"] =
      state->q_origin_identity.q_model_file_identity_schema;
  result["q_origin_file_sha256"] =
      state->q_origin_identity.q_model_file_sha256;
  result["q_origin_canonical_values_schema"] =
      state->q_origin_identity.q_canonical_values_schema;
  result["q_origin_canonical_values_sha256"] =
      state->q_origin_identity.q_canonical_values_sha256;
  result["q_runtime_module_values_schema"] =
      kVrpoQWarmupRuntimeQStateSchema;
  result["q_runtime_module_values_before_sha256"] =
      stats.q_values_before_sha256;
  result["q_runtime_module_values_after_sha256"] =
      stats.q_values_after_sha256;
  result["q_optimizer_step_before"] = state->q_optimizer_steps;
  result["q_optimizer_steps_this_update"] = stats.q_optimizer_steps;
  result["q_optimizer_step_after"] =
      state->q_optimizer_steps + stats.q_optimizer_steps;
  result["q_optimizer_state_before_sha256"] = optimizer_before;
  result["q_optimizer_state_after_sha256"] = optimizer_after;
  result["target_refreshes"] = stats.target_refreshes;
  result["target_values_sha256"] = stats.target_values_sha256;
  result["target_bundle_sha256"] = stats.target_bundle_sha256;
  result["q_loss_mean"] = VrpoPhase4eCanonicalDouble(stats.q_loss_mean);
  result["max_q_grad_norm"] =
      VrpoPhase4eCanonicalDouble(stats.max_q_grad_norm);
  result["target_min"] = VrpoPhase4eCanonicalDouble(stats.target_min);
  result["target_max"] = VrpoPhase4eCanonicalDouble(stats.target_max);
  result["prediction_min"] =
      VrpoPhase4eCanonicalDouble(stats.prediction_min);
  result["prediction_max"] =
      VrpoPhase4eCanonicalDouble(stats.prediction_max);
  json::Array partitions;
  for (const auto& digest : stats.q_epoch_partition_sha256) {
    partitions.emplace_back(digest);
  }
  result["q_epoch_partition_sha256"] = std::move(partitions);
  result["deterministic_summary_sha256"] =
      stats.deterministic_summary_sha256;
  result["strict_reload_and_canary_passed"] = true;
  result["current_targets_recomputed_before_update"] = true;
  result["expected_sarsa_lambda_phase4d_reference"] = true;
  result["relative_to_absolute_adapter_checked"] = true;
  result["holdout_used_for_training"] = false;
  result["training_authorized_beyond_update"] =
      update_number < kVrpoQWarmupUpdates;
  if (!vrpo_ppo_pilot_internal::WriteAtomicText(
          result_path, json::ToString(result, true) + "\n", error) ||
      !vrpo_q_warmup_internal::WriteChecksums(
          checksums_path,
          {corpus_path, evidence_path, q_path, optimizer_path, result_path},
          startup.registration_id, update_number, false, error) ||
      !VrpoFsyncDirectory(temp_dir, error)) {
    return fail(error != nullptr ? *error : "Q-warmup evidence write failed");
  }
  bool renamed = false;
  if (!vrpo_q_warmup_internal::RenameDirectoryNoReplace(
          temp_dir, final_dir, &renamed, error) || !renamed) {
    return fail(error != nullptr && !error->empty()
                    ? *error
                    : "Q-warmup update no-replace commit failed");
  }
  owns_temp_dir = false;
  owns_final_dir = true;
  if (!VrpoFsyncDirectory(startup.output_root, error) ||
      !vrpo_q_warmup_internal::ValidateCommittedUpdateDirectory(
          final_dir, error) ||
      !vrpo_q_warmup_internal::ObserveBytes(
          startup.output_root, &state->measured_peak_bytes, error) ||
      !vrpo_q_warmup_internal::EvidenceStillExact(
          startup.evidence_root, state->evidence_observations, error) ||
      !deadline.Check("after update atomic commit", error)) {
    return fail(error != nullptr && !error->empty()
                    ? *error
                    : "Q-warmup atomic update commit rejected");
  }
  state->completed_updates = update_number;
  state->q_optimizer_steps += stats.q_optimizer_steps;
  state->latest_q_runtime_module_values_sha256 =
      stats.q_values_after_sha256;
  state->latest_q_optimizer_state_sha256 = optimizer_after;
  json::Object summary;
  summary["update"] = static_cast<int64_t>(update_number);
  summary["start_episode_id"] = static_cast<int64_t>(expected_start);
  summary["next_episode_id"] = static_cast<int64_t>(expected_next);
  summary["rows"] = pairing.paired_rows;
  summary["corpus_payload_sha256"] = corpus_identity.payload_sha256;
  summary["q_runtime_module_values_schema"] =
      kVrpoQWarmupRuntimeQStateSchema;
  summary["q_runtime_module_values_after_sha256"] =
      stats.q_values_after_sha256;
  summary["q_optimizer_step_after"] = state->q_optimizer_steps;
  summary["q_optimizer_state_after_sha256"] = optimizer_after;
  summary["update_result_sha256"] =
      ComputeFileSHA256((final_dir / "UPDATE_RESULT.json").string(), nullptr);
  summary["checksums_sha256"] =
      ComputeFileSHA256((final_dir / "CHECKSUMS.json").string(), nullptr);
  state->committed_updates.emplace_back(std::move(summary));
  state->completed = update_number == kVrpoQWarmupUpdates;
  outer_transaction.Commit();
  *update_result = std::move(result);
  owns_final_dir = false;
  return true;
}

inline bool WriteVrpoQWarmupHoldoutAndGlobalResult(
    const VrpoQWarmupStartupConfig& startup,
    const VrpoExpandedExpectedLayout& layout,
    const std::vector<VrpoTrainingEpisode>& holdout_episodes,
    const VrpoPhase4ePairingStats& pairing,
    const std::shared_ptr<SharedDunePolicyValueNetImpl>& actor,
    const std::shared_ptr<DuneVrpoQNetImpl>& initial_q,
    const std::shared_ptr<DuneVrpoQNetImpl>& final_q,
    torch::optim::AdamW& final_q_optimizer,
    const VrpoQWarmupDeadline& deadline,
    VrpoQWarmupFailurePoint failure_point, VrpoQWarmupState* state,
    json::Object* output, std::string* error) {
  const auto temp_dir = startup.output_root / ".holdout.tmp";
  const auto holdout_dir = startup.output_root / "holdout";
  bool owns_temp_dir = false;
  bool owns_holdout_dir = false;
  auto fail = [&](const std::string& message) {
    if (owns_temp_dir) {
      vrpo_ppo_pilot_internal::CleanupPath(temp_dir);
      owns_temp_dir = false;
    }
    if (owns_holdout_dir) {
      vrpo_ppo_pilot_internal::CleanupPath(holdout_dir);
      owns_holdout_dir = false;
    }
    if (error != nullptr) *error = message;
    if (output != nullptr) output->clear();
    return false;
  };
  if (state == nullptr || output == nullptr || actor == nullptr ||
      initial_q == nullptr || final_q == nullptr || !state->completed ||
      state->had_failure || !state->owns_output_root ||
      state->completed_updates != kVrpoQWarmupUpdates ||
      state->q_optimizer_steps != kVrpoQWarmupFinalQSteps ||
      holdout_episodes.size() != kVrpoQWarmupHoldoutGames ||
      pairing.episodes != kVrpoQWarmupHoldoutGames ||
      std::filesystem::exists(temp_dir) ||
      std::filesystem::exists(holdout_dir) ||
      !deadline.Check("before heldout evaluation", error)) {
    return fail(error != nullptr && !error->empty()
                    ? *error
                    : "Q-warmup heldout sequencing rejected");
  }
  if (holdout_episodes.front().episode_id !=
          kVrpoQWarmupHoldoutStartEpisodeId ||
      holdout_episodes.back().episode_id !=
          kVrpoQWarmupHoldoutEndEpisodeIdInclusive ||
      pairing.episode_evidence.front().episode_id !=
          kVrpoQWarmupHoldoutStartEpisodeId ||
      pairing.episode_evidence.back().episode_id !=
          kVrpoQWarmupHoldoutEndEpisodeIdInclusive ||
      !ValidateVrpoPhase4ePairingStatsAgainstEpisodes(
          pairing, holdout_episodes, error)) {
    return fail(error != nullptr && !error->empty()
                    ? *error
                    : "Q-warmup holdout IDs/pairing rejected");
  }
  const std::string actor_state =
      vrpo_q_warmup_internal::ModuleParametersAndBuffersSha256(*actor, error);
  std::string initial_q_hash;
  std::string final_q_hash;
  std::string optimizer_hash;
  if (actor_state != state->actor_state_sha256 ||
      !vrpo_training_internal::ModuleValueSha256(
          *initial_q, "", &initial_q_hash, error) ||
      initial_q_hash != state->initial_q_runtime_module_values_sha256 ||
      !vrpo_training_internal::ModuleValueSha256(
          *final_q, "", &final_q_hash, error) ||
      final_q_hash != state->latest_q_runtime_module_values_sha256 ||
      !vrpo_q_warmup_internal::QOptimizerStateSha256(
          final_q_optimizer, *final_q, kVrpoQWarmupFinalQSteps,
          &optimizer_hash, error) ||
      optimizer_hash != state->latest_q_optimizer_state_sha256) {
    return fail(error != nullptr && !error->empty()
                    ? *error
                    : "Q-warmup heldout actor/Q/Adam binding rejected");
  }
  std::string corpus_bytes;
  VrpoScheduleCorpusIdentity corpus_identity;
  std::vector<VrpoTrainingEpisode> decoded;
  VrpoScheduleCorpusIdentity decoded_identity;
  if (!EncodeVrpoQWarmupCentralCorpus(
          holdout_episodes, &corpus_bytes, &corpus_identity, error) ||
      !DecodeVrpoQWarmupCentralCorpus(
          corpus_bytes, &decoded, &decoded_identity, error) ||
      decoded_identity.payload_sha256 != corpus_identity.payload_sha256 ||
      decoded.size() != kVrpoQWarmupHoldoutGames) {
    return fail(error != nullptr && !error->empty()
                    ? *error
                    : "Q-warmup holdout corpus seal/reload rejected");
  }
  VrpoActorForward actor_forward = [actor](const torch::Tensor& input) {
    torch::NoGradGuard no_grad;
    const auto result = actor->forward(input);
    return VrpoActorTrainingOutput{result.logits.detach(),
                                   result.values.detach()};
  };
  VrpoQForward initial_forward = [initial_q](const torch::Tensor& input) {
    torch::Tensor out;
    std::string forward_error;
    if (!initial_q->ForwardChecked(input, &out, &forward_error)) {
      throw std::runtime_error("initial Q holdout forward rejected: " +
                               forward_error);
    }
    return out;
  };
  VrpoQForward final_forward = [final_q](const torch::Tensor& input) {
    torch::Tensor out;
    std::string forward_error;
    if (!final_q->ForwardChecked(input, &out, &forward_error)) {
      throw std::runtime_error("final Q holdout forward rejected: " +
                               forward_error);
    }
    return out;
  };
  VrpoQWarmupHoldoutEvaluation evaluation;
  if (!EvaluateVrpoQWarmupHoldout(
          decoded, corpus_identity.payload_sha256, *actor, *initial_q,
          *final_q, actor_forward, initial_forward, final_forward,
          &evaluation, error) ||
      !deadline.Check("after heldout evaluation", error)) {
    return fail(error != nullptr && !error->empty()
                    ? *error
                    : "Q-warmup heldout evaluation failed");
  }
  // No optimizer operation may occur on heldout rows.
  std::string optimizer_after_holdout;
  if (!vrpo_q_warmup_internal::QOptimizerStateSha256(
          final_q_optimizer, *final_q, kVrpoQWarmupFinalQSteps,
          &optimizer_after_holdout, error) ||
      optimizer_after_holdout != optimizer_hash ||
      vrpo_q_warmup_internal::ModuleParametersAndBuffersSha256(
          *actor, error) != actor_state) {
    return fail(error != nullptr && !error->empty()
                    ? *error
                    : "Q-warmup holdout mutated training state");
  }
  if (!vrpo_q_warmup_internal::ClaimFreshDirectory(
          temp_dir, &owns_temp_dir, error)) {
    return fail("Q-warmup holdout temp directory creation failed");
  }
  const auto corpus_path = temp_dir / "HOLDOUT_CORPUS.bin";
  const auto evidence_path = temp_dir / "HOLDOUT_EVIDENCE.json";
  const auto metrics_path = temp_dir / "HOLDOUT_METRICS.json";
  const auto checksums_path = temp_dir / "CHECKSUMS.json";
  json::Object metrics;
  metrics["schema"] = kVrpoQWarmupMetricsSchema;
  metrics["registration_id"] = startup.registration_id;
  metrics["classification"] = evaluation.classification;
  metrics["holdout_start_episode_id"] =
      static_cast<int64_t>(kVrpoQWarmupHoldoutStartEpisodeId);
  metrics["holdout_end_episode_id_inclusive"] =
      static_cast<int64_t>(kVrpoQWarmupHoldoutEndEpisodeIdInclusive);
  metrics["holdout_rows"] = evaluation.rows;
  metrics["holdout_scalar_targets"] = evaluation.scalar_targets;
  metrics["holdout_corpus_payload_sha256"] =
      evaluation.corpus_payload_sha256;
  metrics["holdout_row_identity_sha256"] = evaluation.row_identity_sha256;
  metrics["q_origin_canonical_values_schema"] =
      state->q_origin_identity.q_canonical_values_schema;
  metrics["q_origin_canonical_values_sha256"] =
      state->q_origin_identity.q_canonical_values_sha256;
  metrics["q_runtime_module_values_schema"] =
      kVrpoQWarmupRuntimeQStateSchema;
  metrics["initial_q_runtime_module_values_sha256"] = initial_q_hash;
  metrics["final_q_runtime_module_values_sha256"] = final_q_hash;
  metrics["initial"] = json::Value(
      vrpo_q_warmup_internal::MetricSetJson(evaluation.initial));
  metrics["final"] = json::Value(
      vrpo_q_warmup_internal::MetricSetJson(evaluation.final));
  metrics["chosen_q_mse_ratio"] = VrpoPhase4eCanonicalDouble(
      evaluation.initial.chosen_q.mse == 0.0
          ? (evaluation.final.chosen_q.mse == 0.0 ? 0.0
                                                  : std::numeric_limits<double>::max())
          : evaluation.final.chosen_q.mse /
                evaluation.initial.chosen_q.mse);
  metrics["policy_v_mse_ratio"] = VrpoPhase4eCanonicalDouble(
      evaluation.initial.policy_v.mse == 0.0
          ? (evaluation.final.policy_v.mse == 0.0 ? 0.0
                                                  : std::numeric_limits<double>::max())
          : evaluation.final.policy_v.mse /
                evaluation.initial.policy_v.mse);
  metrics["mse_ratio_max_inclusive"] =
      VrpoPhase4eCanonicalDouble(kVrpoQWarmupImprovementRatio);
  metrics["pearson_not_worse_tolerance"] =
      VrpoPhase4eCanonicalDouble(kVrpoQWarmupPearsonTolerance);
  metrics["initial_final_identical_rows"] = evaluation.identical_rows;
  metrics["same_central_states_actions_legal_policies"] =
      evaluation.identical_rows;
  metrics["relative_to_absolute_adapter_exact"] = evaluation.adapter_exact;
  metrics["all_finite"] = evaluation.all_finite;
  metrics["improvement_pass"] = evaluation.improvement_pass;
  metrics["holdout_used_for_training"] = false;
  metrics["q_optimizer_step_before_holdout"] = kVrpoQWarmupFinalQSteps;
  metrics["q_optimizer_step_after_holdout"] = kVrpoQWarmupFinalQSteps;
  if (!vrpo_ppo_pilot_internal::WriteAtomicText(
          corpus_path, corpus_bytes, error) ||
      !vrpo_ppo_pilot_internal::WriteAtomicText(
          evidence_path,
          json::ToString(vrpo_q_warmup_internal::EpisodeEvidenceJson(
                             pairing,
                             "dune_vrpo_q_warmup_holdout_evidence_v1",
                             0, true),
                         true) + "\n",
          error) ||
      !vrpo_ppo_pilot_internal::WriteAtomicText(
          metrics_path, json::ToString(metrics, true) + "\n", error) ||
      !vrpo_q_warmup_internal::WriteChecksums(
          checksums_path, {corpus_path, evidence_path, metrics_path},
          startup.registration_id, 0, true, error) ||
      !VrpoFsyncDirectory(temp_dir, error)) {
    return fail(error != nullptr ? *error : "Q-warmup holdout write failed");
  }
  bool renamed = false;
  if (!vrpo_q_warmup_internal::RenameDirectoryNoReplace(
          temp_dir, holdout_dir, &renamed, error) || !renamed) {
    return fail(error != nullptr && !error->empty()
                    ? *error
                    : "Q-warmup holdout no-replace commit failed");
  }
  owns_temp_dir = false;
  owns_holdout_dir = true;
  if (!VrpoFsyncDirectory(startup.output_root, error) ||
      !vrpo_q_warmup_internal::ValidateCommittedHoldoutDirectory(
          holdout_dir, error) ||
      failure_point == VrpoQWarmupFailurePoint::kBeforeHoldoutStatus ||
      !vrpo_q_warmup_internal::EvidenceStillExact(
          startup.evidence_root, state->evidence_observations, error) ||
      !vrpo_q_warmup_internal::ObserveBytes(
          startup.output_root, &state->measured_peak_bytes, error) ||
      !deadline.Check("before status-last commit", error)) {
    return fail(error != nullptr && !error->empty()
                    ? *error
                    : "Q-warmup status-last precommit rejected");
  }
  const int64_t retained_before_status =
      vrpo_q_warmup_internal::DirectoryRegularBytes(startup.output_root);
  if (retained_before_status < 0 ||
      retained_before_status > kVrpoQWarmupRetainedByteCeiling) {
    return fail("Q-warmup retained bytes exceed 8 GiB");
  }
  json::Object result;
  result["schema"] = kVrpoQWarmupResultSchema;
  result["status"] = "VALID";
  result["classification"] = evaluation.classification;
  result["registration_id"] = startup.registration_id;
  result["compiled_registration_id"] = kVrpoQWarmupRegistrationId;
  result["profile"] = startup.profile;
  result["compiled_profile"] = kVrpoQWarmupProfile;
  result["purpose"] = "FROZEN_ACTOR_Q_ONLY_WARMUP_DECISION_SCREEN";
  result["source_code_sha256"] = startup.source_code_sha256;
  result["executed_binary_sha256"] = startup.executed_binary_sha256;
  result["executed_binary_size"] = startup.executed_binary_size;
  result["immutable_evidence"] = json::Value(
      VrpoPpoPilotEvidenceObservationsJson(state->evidence_observations));
  result["confirmed_live_actor_values_sha256"] =
      kVrpoQWarmupActorValuesSha256;
  result["live_actor_state_sha256"] = state->actor_state_sha256;
  result["actor_frozen"] = true;
  result["actor_optimizer_constructed"] = false;
  result["actor_optimizer_steps"] = int64_t{0};
  result["actor_backward_calls"] = int64_t{0};
  result["q_model_file_identity_schema"] =
      state->q_origin_identity.q_model_file_identity_schema;
  result["q_model_file_sha256"] =
      state->q_origin_identity.q_model_file_sha256;
  result["q_model_file_size"] =
      state->q_origin_identity.q_model_file_size;
  result["q_canonical_values_schema"] =
      state->q_origin_identity.q_canonical_values_schema;
  result["initial_q_canonical_values_sha256"] =
      state->q_origin_identity.q_canonical_values_sha256;
  result["q_runtime_module_values_schema"] =
      state->q_origin_identity.q_runtime_module_values_schema;
  result["initial_q_runtime_module_values_sha256"] =
      state->initial_q_runtime_module_values_sha256;
  result["fresh_canonical_q_adam_step0"] = true;
  result["source_optimizer_moments_loaded"] = false;
  result["final_q_runtime_module_values_sha256"] =
      state->latest_q_runtime_module_values_sha256;
  result["q_optimizer_steps"] = state->q_optimizer_steps;
  result["q_optimizer_state_sha256"] =
      state->latest_q_optimizer_state_sha256;
  result["completed_train_updates"] =
      static_cast<int64_t>(state->completed_updates);
  result["train_start_episode_id"] =
      static_cast<int64_t>(kVrpoQWarmupTrainStartEpisodeId);
  result["train_end_episode_id_inclusive"] =
      static_cast<int64_t>(kVrpoQWarmupTrainEndEpisodeIdInclusive);
  result["holdout_start_episode_id"] =
      static_cast<int64_t>(kVrpoQWarmupHoldoutStartEpisodeId);
  result["holdout_end_episode_id_inclusive"] =
      static_cast<int64_t>(kVrpoQWarmupHoldoutEndEpisodeIdInclusive);
  result["next_episode_id"] = static_cast<int64_t>(
      kVrpoQWarmupHoldoutEndEpisodeIdInclusive + 1);
  result["updates"] = state->committed_updates;
  result["holdout_metrics"] = json::Value(metrics);
  result["holdout_metrics_sha256"] = ComputeFileSHA256(
      (holdout_dir / "HOLDOUT_METRICS.json").string(), nullptr);
  result["holdout_checksums_sha256"] = ComputeFileSHA256(
      (holdout_dir / "CHECKSUMS.json").string(), nullptr);
  result["retained_bytes_before_status"] = retained_before_status;
  result["retained_byte_ceiling"] = kVrpoQWarmupRetainedByteCeiling;
  result["peak_byte_ceiling"] = kVrpoQWarmupPeakByteCeiling;
  result["measured_peak_bytes"] = state->measured_peak_bytes;
  result["wall_clock_ceiling_seconds"] = int64_t{1800};
  const double precommit_elapsed_seconds = deadline.ElapsedSeconds();
  result["precommit_elapsed_seconds"] = precommit_elapsed_seconds;
  result["elapsed_seconds"] = precommit_elapsed_seconds;
  result["status_last"] = true;
  result["deadline_gate_passed"] = true;
  result["final_status_byte_and_deadline_gate_after_root_fsync"] = true;
  result["holdout_used_for_training"] = false;
  result["authorizes_design_of_one_safe_lr_vrpo_actor_screen"] =
      evaluation.improvement_pass;
  result["authorizes_vrpo_actor_training_launch"] = false;
  result["authorizes_longer_training"] = false;
  result["authorizes_playing_strength_evaluation"] = false;
  result["authorizes_promotion"] = false;
  result["stop_if_no_improvement"] = !evaluation.improvement_pass;
  const auto status_path = startup.output_root / "Q_WARMUP_RESULT.json";
  int64_t retained_after_status = -1;
  if (!vrpo_q_warmup_internal::CommitFinalStatusLast(
          startup.output_root, status_path,
          json::ToString(result, true) + "\n", deadline,
          failure_point ==
              VrpoQWarmupFailurePoint::kForceFinalByteOverrunForTest,
          failure_point ==
              VrpoQWarmupFailurePoint::kForceFinalDeadlineOverrunForTest,
          &state->owns_output_root, &state->measured_peak_bytes,
          &retained_after_status, error)) {
    return fail(error != nullptr && !error->empty()
                    ? *error
                    : "Q-warmup status-last write failed");
  }
  *output = std::move(result);
  owns_holdout_dir = false;
  return true;
}

}  // namespace open_spiel

#endif  // OPEN_SPIEL_BUILD_WITH_LIBTORCH

#endif  // OPEN_SPIEL_EXAMPLES_DUNE_VRPO_Q_WARMUP_H_
