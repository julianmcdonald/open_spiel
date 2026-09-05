#ifndef OPEN_SPIEL_EXAMPLES_DUNE_VRPO_Q_OFFLINE_SCREEN_H_
#define OPEN_SPIEL_EXAMPLES_DUNE_VRPO_Q_OFFLINE_SCREEN_H_

// Terminal, offline-only target/LR screen over the immutable Q-warmup retry
// corpora.  This path never constructs an evaluator, collects an environment
// transition, or constructs an actor optimizer.

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
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

#include "dune_vrpo_q_warmup.h"

#ifdef OPEN_SPIEL_BUILD_WITH_LIBTORCH
#include <torch/torch.h>

namespace open_spiel {

inline constexpr char kVrpoQOfflineProfile[] =
    "VRPO_U15828_U4_Q_OFFLINE_TARGET_LR_SCREEN_20260902";
inline constexpr char kVrpoQOfflineRegistrationId[] =
    "VRPO_U15828_U4_Q_OFFLINE_TARGET_LR_SCREEN_20260902";
inline constexpr char kVrpoQOfflineResultSchema[] =
    "dune_vrpo_q_offline_target_lr_screen_v1";
inline constexpr char kVrpoQOfflineCellSchema[] =
    "dune_vrpo_q_offline_target_lr_cell_v1";
inline constexpr char kVrpoQOfflineChecksumsSchema[] =
    "dune_vrpo_q_offline_checksums_v1";
inline constexpr int kVrpoQOfflineCells = 4;
inline constexpr int kVrpoQOfflineCorpora = 4;
inline constexpr int kVrpoQOfflineEpochsPerCorpus = 1;
inline constexpr int64_t kVrpoQOfflineStepsPerCorpus = 16;
inline constexpr int64_t kVrpoQOfflineFinalSteps = 64;
inline constexpr int64_t kVrpoQOfflineRetainedByteCeiling =
    2LL * 1024 * 1024 * 1024;
inline constexpr int64_t kVrpoQOfflinePeakByteCeiling =
    3LL * 1024 * 1024 * 1024;
inline constexpr uint64_t kVrpoQOfflinePartitionBaseSeed =
    kVrpoQWarmupBaseSeed;
inline constexpr char kVrpoQOfflineActorRuntimeSha256[] =
    "295251d8f69a36e14e25a3cb16934061481715270fd8b9fd1f524b44f505e681";
inline constexpr int64_t kVrpoQOfflineActorFileSize = 334205261;
inline constexpr int64_t kVrpoQOfflineInitialQFileSize = 42318498;
inline constexpr char kVrpoQOfflineFileIdentitySchema[] =
    "sha256_file_bytes_v1";
inline constexpr char kVrpoQOfflineActorValuesSchema[] =
    "dune_vrpo_named_parameter_values_v1";
inline constexpr char kVrpoQOfflineActorRuntimeStateSchema[] =
    "dune_vrpo_q_warmup_module_state_v1";

enum class VrpoQOfflineTargetKind { kTdLambda, kMonteCarlo };

struct VrpoQOfflineCellConfig {
  std::string id;
  VrpoQOfflineTargetKind target_kind = VrpoQOfflineTargetKind::kTdLambda;
  double learning_rate = 0.0;
  int fixed_rank = 0;
};

inline std::array<VrpoQOfflineCellConfig, kVrpoQOfflineCells>
CanonicalVrpoQOfflineCells() {
  return {{{"TD1", VrpoQOfflineTargetKind::kTdLambda, 1e-5, 0},
           {"TD2", VrpoQOfflineTargetKind::kTdLambda, 5e-6, 1},
           {"MC1", VrpoQOfflineTargetKind::kMonteCarlo, 1e-5, 2},
           {"MC2", VrpoQOfflineTargetKind::kMonteCarlo, 5e-6, 3}}};
}

inline std::array<std::string, kVrpoQOfflineCorpora>
VrpoQOfflineExpectedPartitionSha256() {
  return {{"9cb3fd5ddb4613d9a6674c550ce39c1952f1e94ac1bcbcd65a6d6af7896435d1",
           "e054bbe75a1f2a7b76c151cf0bd50924d615be85f34bba0e3d61b471a47f36fb",
           "aff9466239d8915e62b4e134fc9a604bf5037524b23ad0a83c7694b39d239f3e",
           "36a543a9cb4770ce10c11159348dc6b3c9ac4a447023c491d4d1fc6e9b71ddf7"}};
}

inline std::array<std::string, kVrpoQOfflineCorpora>
VrpoQOfflineTrainCorpusFileSha256() {
  return {{"b852ce907f09eb10051ba362fae494317ad3a05fd31e52a5df1a0a1493508eae",
           "7862b72951bef0aca2f7ddd52348bac0d1df2d4162632f51f78066dae4704ee3",
           "cbd9d887020d1af42b4fd6f998f2ae96a03757a89193855f0a24b2e24a76d952",
           "352823a8fbbf5a746bb668bab403343ad36c44ef5b1bea80c2c90f000cba8a0c"}};
}

inline std::array<std::string, kVrpoQOfflineCorpora>
VrpoQOfflineTrainCorpusPayloadSha256() {
  return {{"927e8d771c96cb3e797068677aa6c691aa864207b7ab9610563af7520d175f4d",
           "5f48aef751d0ba666607b11efd17de4913744671adc98a3ebeeba07684315cc5",
           "36190a741d8fd9d80f87069a27acaa61671b6b38937c95723783fa29e0400748",
           "187837bd1edbb4345285e92b1bb2830c92be7550256e0ca6d723accd270a4749"}};
}

inline std::array<int64_t, kVrpoQOfflineCorpora>
VrpoQOfflineTrainCorpusFileSizes() {
  return {{826846381, 883181983, 835272573, 807582428}};
}

inline constexpr char kVrpoQOfflineHoldoutFileSha256[] =
    "4a7e9d7603185cc99cc98d4dbdb360b77423e5929b64ba4bd9bca6dfd68f7628";
inline constexpr char kVrpoQOfflineHoldoutPayloadSha256[] =
    "f4bd450db2a11b009c83be2d50ffdaffa191b152e16cbb948c4afe1cb1af60b4";
inline constexpr int64_t kVrpoQOfflineHoldoutFileSize = 829419209;

inline uint64_t VrpoQOfflinePartitionSeed(int corpus_index) {
  if (corpus_index < 0 || corpus_index >= kVrpoQOfflineCorpora) return 0;
  const uint64_t update_number = static_cast<uint64_t>(corpus_index + 1);
  const uint64_t expected_start = kVrpoQWarmupTrainStartEpisodeId +
      static_cast<uint64_t>(corpus_index * kVrpoQWarmupGamesPerUpdate);
  const uint64_t update_seed = vrpo_training_internal::SplitMix64(
      kVrpoQOfflinePartitionBaseSeed ^ expected_start ^
      (0x515741524d555000ULL + update_number));
  return vrpo_training_internal::SplitMix64(update_seed + 0x10001ULL);
}

struct VrpoQOfflinePartitionBinding {
  uint64_t seed = 0;
  std::string expected_sha256;
};

inline bool VrpoQOfflinePartitionBindingForCell(
    const VrpoQOfflineCellConfig& cell, int corpus_index,
    VrpoQOfflinePartitionBinding* output, std::string* error) {
  if (output == nullptr || corpus_index < 0 ||
      corpus_index >= kVrpoQOfflineCorpora) {
    if (error != nullptr) *error = "offline-Q partition binding arguments invalid";
    return false;
  }
  const auto cells = CanonicalVrpoQOfflineCells();
  if (cell.fixed_rank < 0 || cell.fixed_rank >= kVrpoQOfflineCells ||
      cells[cell.fixed_rank].id != cell.id) {
    if (error != nullptr) *error = "offline-Q partition cell identity rejected";
    return false;
  }
  output->seed = VrpoQOfflinePartitionSeed(corpus_index);
  output->expected_sha256 =
      VrpoQOfflineExpectedPartitionSha256()[corpus_index];
  return output->seed != 0 &&
      VrpoPhase4eLowerHex64(output->expected_sha256);
}

inline bool ValidateVrpoQOfflinePartitionBinding(
    const VrpoQOfflineCellConfig& cell, int corpus_index,
    uint64_t observed_seed, const std::string& observed_sha256,
    std::string* error) {
  VrpoQOfflinePartitionBinding registered;
  if (!VrpoQOfflinePartitionBindingForCell(
          cell, corpus_index, &registered, error) ||
      observed_seed != registered.seed ||
      observed_sha256 != registered.expected_sha256) {
    if (error != nullptr && error->empty()) {
      *error = "offline-Q registered partition seed/hash mismatch";
    }
    return false;
  }
  return true;
}

class VrpoQOfflineDeadline {
 public:
  static VrpoQOfflineDeadline Start(
      std::chrono::steady_clock::time_point start,
      std::chrono::seconds limit = std::chrono::seconds(1200)) {
    VrpoQOfflineDeadline out;
    out.start_ = start;
    out.deadline_ = start + limit;
    out.active_ = true;
    return out;
  }
  static VrpoQOfflineDeadline ExpireAfterChecksForTest(int checks) {
    auto out = Start(std::chrono::steady_clock::now(), std::chrono::hours(1));
    out.test_checks_remaining_ = checks;
    return out;
  }
  bool Check(const std::string& stage, std::string* error) const {
    if (test_checks_remaining_ >= 0) {
      if (test_checks_remaining_-- == 0) {
        if (error != nullptr) *error = "injected offline-Q deadline at " + stage;
        return false;
      }
    }
    if (!active_ || std::chrono::steady_clock::now() < deadline_) return true;
    if (error != nullptr) *error = "offline-Q 1200-second deadline at " + stage;
    return false;
  }
  double ElapsedSeconds() const {
    return active_ ? std::chrono::duration<double>(
                         std::chrono::steady_clock::now() - start_).count()
                   : 0.0;
  }

 private:
  bool active_ = false;
  std::chrono::steady_clock::time_point start_{};
  std::chrono::steady_clock::time_point deadline_{};
  mutable int test_checks_remaining_ = -1;
};

struct VrpoQOfflineEvidence {
  VrpoQWarmupPriorEvidence origin;
  std::vector<VrpoPpoPilotEvidenceObservation> retry_observations;
  std::array<std::filesystem::path, kVrpoQOfflineCorpora> train_corpora;
  std::filesystem::path holdout_corpus;
};

inline std::vector<VrpoPpoPilotEvidenceFileSpec>
VrpoQOfflineRetryEvidenceFiles() {
  const std::string registration =
      "registrations/vrpo_q_warmup4x16_holdout16_u4_retry1_20260902/";
  const std::string run = "vrpo_q_warmup4x16_holdout16_u4_retry1_20260902/";
  return {
      {"retry_registration", registration + "registration.json", "fe02f0a8d219706989c16a56430c96a643e10a37ec01721959ff7fba0a5aa33d"},
      {"retry_registration_sha", registration + "registration.sha256", "db68621a98497a4ff3fe4c25c99cf4f10188c560d8aabbf39bdd64eb0def7b1b"},
      {"retry_run_log", registration + "run.log", "b0ea8572b908ae09f0ecc9d8aa40b6d5e27a77d235c3109f964e4f88f9815cd3"},
      {"retry_exit", registration + "exit.json", "34172d1d4c099f4701eea0ca741050b24092cd9522f7a863e5bd844ea050d28e"},
      {"retry_sidecars", registration + "sidecars.sha256", "00584adc4f0aa1c396110ed4d7b3713afb6587a46dd26d3983251af45fbeb424"},
      {"retry_validation", registration + "validation.json", "c04cb00c7d23bbd6ea93b8cd3f6ac27d769060611455a7a60c37045655e91b4a"},
      {"retry_validation_sha", registration + "validation.sha256", "92149f7fa7035e94ba4fef98f7e3af0beb551074fb4df097fa9a14ec1b89bb7a"},
      {"retry_result", run + "Q_WARMUP_RESULT.json", "21b89d8d3e74963e8f534ea1789d8d2dc892ac9226983bafe4888b169902e7c0"},
      {"u1_checksums", run + "update_1/CHECKSUMS.json", "0b6a5a3d05dd0c2cfda5d55e09fd814e2b804e8f2b252f19e4f3c4fcfa97b8b8"},
      {"u1_evidence", run + "update_1/EPISODE_EVIDENCE.json", "3da00e72140c6e0378a0f12205abb7ecc5bc381e29a5166367f34ca82d3b02a3"},
      {"u1_corpus", run + "update_1/Q_CORPUS.bin", "b852ce907f09eb10051ba362fae494317ad3a05fd31e52a5df1a0a1493508eae"},
      {"u1_result", run + "update_1/UPDATE_RESULT.json", "e727a5749b703c329346c65d99cdcce261b8642e6b18fed374f033c6e9e8c23c"},
      {"u2_checksums", run + "update_2/CHECKSUMS.json", "18516424239f361e1804232e8240d238a16848fcc43664d27cd35853755b28c4"},
      {"u2_evidence", run + "update_2/EPISODE_EVIDENCE.json", "227275f71a761df87df93a8506ed3a1d77f628c92e3724d5ca00d95ce64cbb0e"},
      {"u2_corpus", run + "update_2/Q_CORPUS.bin", "7862b72951bef0aca2f7ddd52348bac0d1df2d4162632f51f78066dae4704ee3"},
      {"u2_result", run + "update_2/UPDATE_RESULT.json", "c022b149181221e4b5ad4d82e931048cd878be1952c292e30af3e94a402e3c71"},
      {"u3_checksums", run + "update_3/CHECKSUMS.json", "df5bfccd3b53d26f7f79e37969dd4836f6fe30f95ea710dc6e8a6efbd66d1f04"},
      {"u3_evidence", run + "update_3/EPISODE_EVIDENCE.json", "0aaebf7624ce1bebb4ce517408a3a419542d19904ddb6baa2ae1b6e24eb4c0d5"},
      {"u3_corpus", run + "update_3/Q_CORPUS.bin", "cbd9d887020d1af42b4fd6f998f2ae96a03757a89193855f0a24b2e24a76d952"},
      {"u3_result", run + "update_3/UPDATE_RESULT.json", "bc157253f5212d4db745f87870d3d1aa4ee36be52af39bd741b4b2cf03d96a31"},
      {"u4_checksums", run + "update_4/CHECKSUMS.json", "ea4fe4b132196fccdce89dde6f65595fb70eef5c37ef5cd4912c86882e8e0f66"},
      {"u4_evidence", run + "update_4/EPISODE_EVIDENCE.json", "8f0c1feea035f06b448bb695c28afa81a9286fd0df24b25eec2167a8eeea2baf"},
      {"u4_corpus", run + "update_4/Q_CORPUS.bin", "352823a8fbbf5a746bb668bab403343ad36c44ef5b1bea80c2c90f000cba8a0c"},
      {"u4_result", run + "update_4/UPDATE_RESULT.json", "9da3c211def66f54aee71aa7e088e2fe9bad3d1fad378a5372d1baaa6a8a7c86"},
      {"holdout_checksums", run + "holdout/CHECKSUMS.json", "a02edad875409bbf03da2fd2d523831dda9988660da6fd43a6a518af253abded"},
      {"holdout_corpus", run + "holdout/HOLDOUT_CORPUS.bin", "4a7e9d7603185cc99cc98d4dbdb360b77423e5929b64ba4bd9bca6dfd68f7628"},
      {"holdout_evidence", run + "holdout/HOLDOUT_EVIDENCE.json", "47e935fc386e3a087ec757a148f24da19c77ea2d24f876fb492a948470ca1c7c"},
      {"holdout_metrics", run + "holdout/HOLDOUT_METRICS.json", "21ed29b53c9de16bcc81b3ba3512e6392d0160012c3fb039e127514c8d477475"},
  };
}

namespace vrpo_q_offline_internal {

inline bool ReadFile(const std::filesystem::path& path, std::string* output,
                     std::string* error) {
  if (output == nullptr || !std::filesystem::is_regular_file(path) ||
      std::filesystem::is_symlink(path)) {
    if (error != nullptr) *error = "offline-Q input is missing/nonregular";
    return false;
  }
  std::ifstream stream(path, std::ios::binary);
  if (!stream) {
    if (error != nullptr) *error = "offline-Q input open failed";
    return false;
  }
  output->assign(std::istreambuf_iterator<char>(stream),
                 std::istreambuf_iterator<char>());
  if (stream.bad() || output->empty()) {
    if (error != nullptr) *error = "offline-Q input read failed";
    output->clear();
    return false;
  }
  return true;
}

inline bool ValidateRetryEvidence(const std::filesystem::path& root,
                                  VrpoQOfflineEvidence* output,
                                  std::string* error) {
  auto fail = [&](const std::string& message) {
    if (error != nullptr) *error = message;
    if (output != nullptr) *output = VrpoQOfflineEvidence{};
    return false;
  };
  if (output == nullptr || !std::filesystem::is_directory(root)) {
    return fail("offline-Q evidence root is invalid");
  }
  VrpoQOfflineEvidence evidence;
  if (!vrpo_q_warmup_internal::ValidatePriorSemantics(
          root, &evidence.origin, error)) {
    return fail(error != nullptr ? *error : "offline-Q origin chain failed");
  }
  const auto specs = VrpoQOfflineRetryEvidenceFiles();
  std::vector<std::string> expected;
  for (const auto& spec : specs) expected.push_back(spec.expected_sha256);
  if (!ValidateVrpoPpoPilotEvidenceFiles(
          root, specs, expected, &evidence.retry_observations, error)) {
    return fail(error != nullptr ? *error : "offline-Q retry chain failed");
  }
  const auto run = root / "vrpo_q_warmup4x16_holdout16_u4_retry1_20260902";
  json::Object registration;
  json::Object validation;
  json::Object result;
  if (!vrpo_q_warmup_internal::ReadJsonObject(
          root / specs[0].relative_path, &registration, error) ||
      !vrpo_q_warmup_internal::ReadJsonObject(
          root / specs[5].relative_path, &validation, error) ||
      !vrpo_q_warmup_internal::ReadJsonObject(
          root / specs[7].relative_path, &result, error) ||
      !vrpo_q_warmup_internal::StringIs(
          registration, "schema", "dune_vrpo_q_warmup_registration_v1") ||
      !vrpo_q_warmup_internal::StringIs(
          registration, "registration_id", kVrpoQWarmupRegistrationId) ||
      !vrpo_q_warmup_internal::StringIs(validation, "status", "VALID") ||
      !vrpo_q_warmup_internal::StringIs(
          validation, "classification", "VALID_Q_WARMUP_NO_IMPROVEMENT") ||
      !vrpo_q_warmup_internal::StringIs(result, "status", "VALID") ||
      !vrpo_q_warmup_internal::StringIs(
          result, "classification", "VALID_Q_WARMUP_NO_IMPROVEMENT") ||
      !vrpo_q_warmup_internal::BoolIs(
          result, "holdout_used_for_training", false) ||
      !vrpo_q_warmup_internal::IntIs(result, "q_optimizer_steps", 256)) {
    return fail("offline-Q retry result semantics rejected");
  }
  for (int index = 0; index < kVrpoQOfflineCorpora; ++index) {
    const auto directory = run / ("update_" + std::to_string(index + 1));
    if (!vrpo_q_warmup_internal::ValidateCommittedUpdateDirectory(
            directory, error)) {
      return fail(error != nullptr ? *error : "offline-Q update archive invalid");
    }
    evidence.train_corpora[index] = directory / "Q_CORPUS.bin";
  }
  if (!vrpo_q_warmup_internal::ValidateCommittedHoldoutDirectory(
          run / "holdout", error)) {
    return fail(error != nullptr ? *error : "offline-Q holdout archive invalid");
  }
  evidence.holdout_corpus = run / "holdout/HOLDOUT_CORPUS.bin";
  *output = std::move(evidence);
  return true;
}

inline bool RetryEvidenceStillExact(const std::filesystem::path& root,
                                    const VrpoQOfflineEvidence& expected,
                                    std::string* error) {
  VrpoQOfflineEvidence observed;
  if (!ValidateRetryEvidence(root, &observed, error) ||
      observed.retry_observations.size() !=
          expected.retry_observations.size()) return false;
  for (size_t i = 0; i < observed.retry_observations.size(); ++i) {
    if (!observed.retry_observations[i].matched ||
        observed.retry_observations[i].observed_sha256 !=
            expected.retry_observations[i].observed_sha256 ||
        observed.retry_observations[i].observed_size !=
            expected.retry_observations[i].observed_size) return false;
  }
  return vrpo_q_warmup_internal::EvidenceStillExact(
      root, expected.origin.observations, error);
}

inline bool ValidateCorpusRange(const std::vector<VrpoTrainingEpisode>& episodes,
                                uint64_t first, uint64_t last,
                                std::set<uint64_t>* all_ids,
                                std::string* error) {
  if (episodes.size() != 16 || first > last || last - first + 1 != 16 ||
      episodes.front().episode_id != first ||
      episodes.back().episode_id != last) {
    if (error != nullptr) *error = "offline-Q corpus ID range rejected";
    return false;
  }
  for (size_t i = 0; i < episodes.size(); ++i) {
    if (episodes[i].episode_id != first + i || episodes[i].rows.empty() ||
        (all_ids != nullptr && !all_ids->insert(episodes[i].episode_id).second)) {
      if (error != nullptr) *error = "offline-Q corpus IDs duplicate/gap";
      return false;
    }
  }
  return true;
}

inline bool LoadCorpus(const std::filesystem::path& path,
                       const std::string& expected_file_sha256,
                       int64_t expected_file_size,
                       const std::string& expected_payload_sha256,
                       uint64_t first, uint64_t last,
                       std::vector<VrpoTrainingEpisode>* episodes,
                       VrpoScheduleCorpusIdentity* identity,
                       std::set<uint64_t>* all_ids, std::string* error) {
  size_t observed_size = 0;
  if (ComputeFileSHA256(path.string(), &observed_size) !=
          expected_file_sha256 || expected_file_size <= 0 ||
      observed_size != static_cast<size_t>(expected_file_size)) {
    if (error != nullptr) *error = "offline-Q corpus file digest mismatch";
    return false;
  }
  std::string bytes;
  if (!ReadFile(path, &bytes, error) ||
      !DecodeVrpoQWarmupCentralCorpus(bytes, episodes, identity, error) ||
      identity->payload_sha256 != expected_payload_sha256 ||
      identity->byte_size != static_cast<int64_t>(observed_size) ||
      !ValidateCorpusRange(*episodes, first, last, all_ids, error)) {
    return false;
  }
  bytes.clear();
  bytes.shrink_to_fit();
  return true;
}

inline bool MakeFreshOptimizer(
    torch::nn::Module& q, double learning_rate,
    std::unique_ptr<torch::optim::AdamW>* output, std::string* error) {
  if (output == nullptr || !(learning_rate == 1e-5 || learning_rate == 5e-6)) {
    if (error != nullptr) *error = "offline-Q optimizer LR rejected";
    return false;
  }
  const auto base = CanonicalVrpoPhase4OptimizerGroups()[2];
  *output = std::make_unique<torch::optim::AdamW>(
      q.parameters(), torch::optim::AdamWOptions(learning_rate)
                          .betas({base.beta1, base.beta2})
                          .eps(base.epsilon)
                          .weight_decay(base.weight_decay));
  MaterializeVrpoZeroAdamWState(**output);
  return true;
}

inline bool OptimizerStateSha256(torch::optim::AdamW& optimizer,
                                 torch::nn::Module& q, double learning_rate,
                                 int64_t expected_step, std::string* output,
                                 std::string* error) {
  auto fail = [&](const std::string& message) {
    if (error != nullptr) *error = message;
    if (output != nullptr) output->clear();
    return false;
  };
  if (output == nullptr || optimizer.param_groups().size() != 1 ||
      optimizer.state().size() != q.named_parameters().size()) {
    return fail("offline-Q optimizer coverage invalid");
  }
  const auto base = CanonicalVrpoPhase4OptimizerGroups()[2];
  const auto& options = static_cast<const torch::optim::AdamWOptions&>(
      optimizer.param_groups()[0].options());
  if (options.lr() != learning_rate ||
      std::get<0>(options.betas()) != base.beta1 ||
      std::get<1>(options.betas()) != base.beta2 ||
      options.eps() != base.epsilon ||
      options.weight_decay() != base.weight_decay) {
    return fail("offline-Q Adam options changed");
  }
  std::map<c10::TensorImpl*, std::string> names;
  for (const auto& item : q.named_parameters()) {
    if (!names.emplace(item.value().unsafeGetTensorImpl(), item.key()).second) {
      return fail("offline-Q duplicate Q parameter");
    }
  }
  std::string payload = "dune_vrpo_q_offline_adamw_state_v1";
  vrpo_capture_internal::AppendPod(&payload, learning_rate);
  vrpo_capture_internal::AppendPod(&payload, base.beta1);
  vrpo_capture_internal::AppendPod(&payload, base.beta2);
  vrpo_capture_internal::AppendPod(&payload, base.epsilon);
  vrpo_capture_internal::AppendPod(&payload, base.weight_decay);
  std::set<c10::TensorImpl*> seen;
  for (const auto& group : optimizer.param_groups()) {
    for (const auto& parameter : group.params()) {
      auto* key = parameter.unsafeGetTensorImpl();
      const auto name = names.find(key);
      const auto found = optimizer.state().find(key);
      const auto* state = found == optimizer.state().end()
          ? nullptr
          : dynamic_cast<const torch::optim::AdamWParamState*>(found->second.get());
      if (name == names.end() || !seen.insert(key).second || state == nullptr ||
          state->step() != expected_step || !state->exp_avg().defined() ||
          !state->exp_avg_sq().defined() ||
          state->exp_avg().sizes() != parameter.sizes() ||
          state->exp_avg_sq().sizes() != parameter.sizes() ||
          !vrpo_training_internal::FiniteTensor(state->exp_avg()) ||
          !vrpo_training_internal::FiniteTensor(state->exp_avg_sq())) {
        return fail("offline-Q Adam state invalid");
      }
      vrpo_schedule_internal::AppendString(&payload, name->second);
      vrpo_capture_internal::AppendPod(&payload, state->step());
      for (const auto& moment : {state->exp_avg(), state->exp_avg_sq()}) {
        auto cpu = moment.detach().contiguous().cpu().to(torch::kFloat32);
        vrpo_capture_internal::AppendPod(&payload, cpu.numel());
        payload.append(reinterpret_cast<const char*>(cpu.data_ptr<float>()),
                       cpu.numel() * sizeof(float));
      }
    }
  }
  if (seen.size() != names.size()) return fail("offline-Q Adam omitted parameter");
  *output = ComputeStringSHA256(payload);
  return true;
}

inline bool BuildMonteCarloTargets(
    const std::vector<VrpoTrainingEpisode>& episodes,
    std::map<uint64_t, VrpoTrainingTargetRow>* targets,
    std::string* target_sha256, std::string* error) {
  if (targets == nullptr || target_sha256 == nullptr) {
    if (error != nullptr) *error = "offline-Q MC outputs missing";
    return false;
  }
  targets->clear();
  std::vector<VrpoTrainingTargetRow> ordered;
  for (const auto& episode : episodes) {
    if (episode.rows.empty() || !episode.rows.back().terminal_after) {
      if (error != nullptr) *error = "offline-Q MC terminal row missing";
      return false;
    }
    const VrpoSeatValues actual = episode.rows.back().rewards;
    for (const auto& row : episode.rows) {
      VrpoTrainingTargetRow target;
      target.row_id = row.row_id;
      target.episode_id = row.episode_id;
      target.step_index = row.step_index;
      target.actor = row.actor;
      target.q_target_absolute = actual;
      VrpoActorRelativeSeatValues relative;
      for (int slot = 0; slot < kVrpoNumSeats; ++slot) {
        relative.slots[slot] = actual[(row.actor + slot) % kVrpoNumSeats];
        target.q_target_actor_relative[slot] = relative.slots[slot];
      }
      VrpoSeatValues roundtrip;
      if (!VrpoActorRelativeToAbsoluteSeatValues(
              row.actor, relative, &roundtrip, error) || roundtrip != actual ||
          !targets->emplace(row.row_id, target).second) {
        if (error != nullptr && error->empty()) {
          *error = "offline-Q MC absolute/relative mapping rejected";
        }
        return false;
      }
      ordered.push_back(target);
    }
  }
  *target_sha256 = vrpo_training_internal::TargetValuesSha256(ordered);
  return !target_sha256->empty();
}

inline bool BuildTargets(
    VrpoQOfflineTargetKind kind,
    const std::vector<VrpoTrainingEpisode>& episodes,
    torch::nn::Module& actor, torch::nn::Module& q,
    const VrpoActorForward& actor_forward, const VrpoQForward& q_forward,
    std::map<uint64_t, VrpoTrainingTargetRow>* targets,
    std::string* target_sha256, std::string* q_source_sha256,
    std::string* error) {
  if (q_source_sha256 == nullptr) {
    if (error != nullptr) *error = "offline-Q target Q-source output missing";
    return false;
  }
  if (kind == VrpoQOfflineTargetKind::kMonteCarlo) {
    *q_source_sha256 = "NOT_APPLICABLE_MONTE_CARLO_NO_BOOTSTRAP";
    return BuildMonteCarloTargets(episodes, targets, target_sha256, error);
  }
  const auto* arm = FindCanonicalVrpoPhase4Arm("VRPO_CAP10");
  VrpoTrainingTargetBundle bundle;
  if (arm == nullptr || arm->gamma != 1.0 || arm->lambda != 1.0 ||
      !ComputeVrpoTrainingTargets(*arm, episodes, actor, q, actor_forward,
                                  q_forward, &bundle, error) ||
      !ValidateVrpoTrainingTargetsFresh(bundle, actor, q, error)) return false;
  targets->clear();
  for (const auto& target : bundle.rows) {
    if (!targets->emplace(target.row_id, target).second) {
      if (error != nullptr) *error = "offline-Q TD targets duplicate row";
      return false;
    }
  }
  *target_sha256 = bundle.target_values_sha256;
  *q_source_sha256 = bundle.q_values_sha256;
  return true;
}

struct CellTrainingStats {
  int64_t optimizer_steps = 0;
  int64_t backward_calls = 0;
  int64_t rows_seen = 0;
  int target_refreshes = 0;
  double mean_loss = 0.0;
  double max_grad_norm = 0.0;
  std::string actor_before_sha256;
  std::string actor_after_sha256;
  std::string q_before_sha256;
  std::string q_after_sha256;
  std::string optimizer_before_sha256;
  std::string optimizer_after_sha256;
  std::array<std::string, 4> target_sha256;
  std::array<std::string, 4> pre_target_q_runtime_sha256;
  std::array<std::string, 4> target_q_runtime_source_sha256;
  std::array<std::string, 4> partition_sha256;
  std::array<std::string, 4> q_boundary_sha256;
};

inline bool TrainOneCorpus(
    const VrpoQOfflineCellConfig& cell, int corpus_index,
    const std::vector<VrpoTrainingEpisode>& episodes,
    torch::nn::Module& actor, torch::nn::Module& q,
    torch::optim::AdamW& optimizer,
    const VrpoActorForward& actor_forward, const VrpoQForward& q_forward,
    CellTrainingStats* stats, std::string* error,
    bool test_fixture = false) {
  if (stats == nullptr || corpus_index < 0 || corpus_index >= 4 ||
      stats->optimizer_steps != corpus_index * kVrpoQOfflineStepsPerCorpus) {
    if (error != nullptr) *error = "offline-Q corpus sequence rejected";
    return false;
  }
  VrpoEpisodePartitionPlan plan;
  VrpoQOfflinePartitionBinding partition_binding;
  if (!VrpoQOfflinePartitionBindingForCell(
          cell, corpus_index, &partition_binding, error) ||
      !BuildVrpoEpisodePartitionPlan(
          episodes, partition_binding.seed, &plan, error) ||
      (!test_fixture && !ValidateVrpoQOfflinePartitionBinding(
          cell, corpus_index, plan.epoch_seed, plan.canonical_sha256,
          error))) {
    return false;
  }
  std::set<size_t> scheduled;
  for (const auto& batch : plan.minibatches) {
    if (batch.episode_indices.size() != 1 ||
        !scheduled.insert(batch.episode_indices.front()).second) {
      if (error != nullptr) *error = "offline-Q M16 partition rejected";
      return false;
    }
  }
  if (scheduled.size() != 16) {
    if (error != nullptr) *error = "offline-Q partition omitted an episode";
    return false;
  }
  std::map<uint64_t, VrpoTrainingTargetRow> targets;
  if (!vrpo_training_internal::ModuleValueSha256(
          q, "", &stats->pre_target_q_runtime_sha256[corpus_index],
          error)) {
    return false;
  }
  if (!BuildTargets(cell.target_kind, episodes, actor, q, actor_forward,
                    q_forward, &targets, &stats->target_sha256[corpus_index],
                    &stats->target_q_runtime_source_sha256[corpus_index],
                    error)) return false;
  if ((cell.target_kind == VrpoQOfflineTargetKind::kTdLambda &&
       stats->target_q_runtime_source_sha256[corpus_index] !=
           stats->pre_target_q_runtime_sha256[corpus_index]) ||
      (cell.target_kind == VrpoQOfflineTargetKind::kMonteCarlo &&
       stats->target_q_runtime_source_sha256[corpus_index] !=
           "NOT_APPLICABLE_MONTE_CARLO_NO_BOOTSTRAP")) {
    if (error != nullptr) {
      *error = "offline-Q target/pre-Q hash relationship rejected";
    }
    return false;
  }
  ++stats->target_refreshes;
  stats->partition_sha256[corpus_index] = plan.canonical_sha256;
  vrpo_training_internal::ModuleRuntimePlacement q_placement;
  if (!vrpo_training_internal::CaptureUniformModuleRuntimePlacement(
          q, &q_placement, error) || q_placement.dtype != torch::kFloat32) {
    return false;
  }
  for (const auto& batch : plan.minibatches) {
    const auto flat = vrpo_training_internal::FlattenEpisodes(
        episodes, batch.episode_indices);
    torch::Tensor input;
    if (!vrpo_training_internal::StackInputs(
            flat.rows, false, q_placement, &input, error)) return false;
    optimizer.zero_grad();
    torch::Tensor output;
    try {
      output = q_forward(input);
    } catch (const std::exception& exception) {
      if (error != nullptr) *error = "offline-Q forward failed: " +
                                      std::string(exception.what());
      return false;
    }
    if (!vrpo_training_internal::ValidateQOutput(
            output, flat.rows.size(), q_placement, error)) return false;
    std::vector<int64_t> actions;
    std::vector<float> values;
    actions.reserve(flat.rows.size());
    values.reserve(flat.rows.size() * kVrpoNumSeats);
    for (const auto* row : flat.rows) {
      const auto target = targets.find(row->row_id);
      if (target == targets.end() || row->chosen_action < 0 ||
          row->chosen_action >= output.size(1)) {
        if (error != nullptr) *error = "offline-Q target/action lookup failed";
        return false;
      }
      actions.push_back(row->chosen_action);
      for (double value : target->second.q_target_actor_relative) {
        values.push_back(static_cast<float>(value));
      }
    }
    const auto chosen = torch::from_blob(
        actions.data(), {static_cast<int64_t>(actions.size()), 1, 1},
        torch::TensorOptions().dtype(torch::kInt64)).clone()
        .to(q_placement.device).expand({-1, 1, kVrpoNumSeats});
    const auto target = torch::from_blob(
        values.data(), {static_cast<int64_t>(actions.size()), kVrpoNumSeats},
        torch::TensorOptions().dtype(torch::kFloat32)).clone().to(q_placement.device);
    const auto predicted = output.gather(1, chosen).squeeze(1);
    const auto loss = 0.5 * torch::mse_loss(predicted, target);
    if (!vrpo_training_internal::FiniteTensor(loss)) {
      if (error != nullptr) *error = "offline-Q loss nonfinite";
      return false;
    }
    loss.backward();
    ++stats->backward_calls;
    double actor_norm = 0.0;
    double q_norm = 0.0;
    bool actor_grad = false;
    bool q_grad = false;
    if (!vrpo_training_internal::GradientNorm(
            actor, "", &actor_norm, &actor_grad, error) ||
        !vrpo_training_internal::GradientNorm(
            q, "", &q_norm, &q_grad, error) || actor_grad ||
        actor_norm != 0.0 || !q_grad || !(q_norm > 0.0) ||
        !std::isfinite(q_norm)) {
      if (error != nullptr && error->empty()) {
        *error = "offline-Q actor/Q gradient gate failed";
      }
      return false;
    }
    stats->max_grad_norm = std::max(stats->max_grad_norm, q_norm);
    torch::nn::utils::clip_grad_norm_(q.parameters(),
                                     kVrpoTrainingGradientClipNorm);
    optimizer.step();
    ++stats->optimizer_steps;
    stats->rows_seen += flat.rows.size();
    stats->mean_loss += loss.detach().item<double>();
  }
  if (!vrpo_training_internal::ModuleValueSha256(
          q, "", &stats->q_boundary_sha256[corpus_index], error)) return false;
  return true;
}

inline bool SaveCellArchive(
    const std::filesystem::path& directory, torch::nn::Module& q,
    torch::optim::AdamW& optimizer, const VrpoQOfflineCellConfig& cell,
    const json::Object& result, const torch::Device& device,
    const VrpoExpandedExpectedLayout& layout, std::string* error) {
  bool owns = false;
  if (!vrpo_q_warmup_internal::ClaimFreshDirectory(directory, &owns, error)) {
    return false;
  }
  auto fail = [&]() {
    if (owns) vrpo_ppo_pilot_internal::CleanupPath(directory);
    return false;
  };
  const auto q_path = directory / "q_model.pt";
  const auto optimizer_path = directory / "q_optimizer.pt";
  const auto result_path = directory / "TRAINING_EVIDENCE.json";
  if (!vrpo_q_warmup_internal::SaveQAndOptimizer(
          q, optimizer, q_path, optimizer_path, error) ||
      !vrpo_ppo_pilot_internal::WriteAtomicText(
          result_path, json::ToString(result, true) + "\n", error)) return fail();
  if (!VrpoFsyncDirectory(directory, error)) return fail();

  auto reloaded = std::make_shared<DuneVrpoQNetImpl>(kVrpoQWarmupQSeed);
  reloaded->to(device, torch::kFloat32);
  std::unique_ptr<torch::optim::AdamW> reloaded_optimizer;
  if (!MakeFreshOptimizer(*reloaded, cell.learning_rate, &reloaded_optimizer,
                          error) ||
      !LoadVrpoModule(*reloaded, q_path, error)) return fail();
  try {
    torch::load(*reloaded_optimizer, optimizer_path.string());
  } catch (const std::exception& exception) {
    if (error != nullptr) *error = "offline-Q optimizer reload failed: " +
                                    std::string(exception.what());
    return fail();
  }
  if (!vrpo_training_internal::MigrateAdamWOptimizerStateToParameterDevices(
          *reloaded_optimizer, error)) return fail();
  std::string live_q;
  std::string reload_q;
  std::string live_optimizer;
  std::string reload_optimizer;
  if (!vrpo_training_internal::ModuleValueSha256(q, "", &live_q, error) ||
      !vrpo_training_internal::ModuleValueSha256(
          *reloaded, "", &reload_q, error) || live_q != reload_q ||
      !OptimizerStateSha256(optimizer, q, cell.learning_rate,
                            kVrpoQOfflineFinalSteps, &live_optimizer, error) ||
      !OptimizerStateSha256(*reloaded_optimizer, *reloaded,
                            cell.learning_rate, kVrpoQOfflineFinalSteps,
                            &reload_optimizer, error) ||
      live_optimizer != reload_optimizer) return fail();
  const auto canary = MakeVrpoPhase4eCanary(layout);
  torch::Tensor first;
  torch::Tensor second;
  if (!reloaded->ForwardChecked(canary.q_input.to(device), &first, error) ||
      !reloaded->ForwardChecked(canary.q_input.to(device), &second, error) ||
      !torch::equal(first, second)) return fail();
  owns = false;
  return true;
}

inline bool FinalizeCellArchive(const std::filesystem::path& directory,
                                const VrpoQOfflineCellConfig& cell,
                                const json::Object& result,
                                std::string* error) {
  const auto result_path = directory / "CELL_RESULT.json";
  const auto checksums_path = directory / "CHECKSUMS.json";
  if (!std::filesystem::is_regular_file(directory / "q_model.pt") ||
      !std::filesystem::is_regular_file(directory / "q_optimizer.pt") ||
      !std::filesystem::is_regular_file(directory / "TRAINING_EVIDENCE.json") ||
      std::filesystem::exists(result_path) ||
      std::filesystem::exists(checksums_path) ||
      !vrpo_ppo_pilot_internal::WriteAtomicText(
          result_path, json::ToString(result, true) + "\n", error)) {
    if (error != nullptr && error->empty()) {
      *error = "offline-Q cell finalization file set rejected";
    }
    return false;
  }
  json::Object checksums;
  checksums["schema"] = kVrpoQOfflineChecksumsSchema;
  checksums["cell_id"] = cell.id;
  json::Array files;
  for (const std::string& name : {"q_model.pt", "q_optimizer.pt",
                                  "TRAINING_EVIDENCE.json",
                                  "CELL_RESULT.json"}) {
    const auto path = directory / name;
    size_t size = 0;
    json::Object file;
    file["filename"] = name;
    file["sha256"] = ComputeFileSHA256(path.string(), &size);
    file["size"] = static_cast<int64_t>(size);
    files.emplace_back(std::move(file));
  }
  checksums["files"] = std::move(files);
  return vrpo_ppo_pilot_internal::WriteAtomicText(
             checksums_path, json::ToString(checksums, true) + "\n", error) &&
      VrpoFsyncDirectory(directory, error);
}

inline json::Object MetricSetJson(const VrpoQWarmupMetricSet& metrics) {
  return vrpo_q_warmup_internal::MetricSetJson(metrics);
}

inline bool ValidateSelectionBiasMetadata(const json::Object& artifact,
                                          std::string* error) {
  const auto holdout_training = artifact.find("holdout_used_for_training");
  const auto holdout_selection = artifact.find("holdout_used_for_selection");
  const auto biased = artifact.find("selected_metrics_are_selection_biased");
  const auto multiplicity = artifact.find("selection_multiplicity");
  const auto classification = artifact.find("selection_evidence_classification");
  if (holdout_training == artifact.end() ||
      !holdout_training->second.IsBool() ||
      holdout_training->second.GetBool() ||
      holdout_selection == artifact.end() ||
      !holdout_selection->second.IsBool() ||
      !holdout_selection->second.GetBool() || biased == artifact.end() ||
      !biased->second.IsBool() || !biased->second.GetBool() ||
      multiplicity == artifact.end() || !multiplicity->second.IsInt() ||
      multiplicity->second.GetInt() != kVrpoQOfflineCells ||
      classification == artifact.end() ||
      !classification->second.IsString() ||
      classification->second.GetString() !=
          "SCREEN_ONLY_NOT_CONFIRMATORY_FRESH_DATA_REQUIRED") {
    if (error != nullptr) {
      *error = "offline-Q selection-bias metadata missing or drifted";
    }
    return false;
  }
  return true;
}

struct CellDecision {
  std::string id;
  int fixed_rank = 0;
  double chosen_ratio = std::numeric_limits<double>::infinity();
  double policy_ratio = std::numeric_limits<double>::infinity();
  double chosen_pearson_gain = -std::numeric_limits<double>::infinity();
  double policy_pearson_gain = -std::numeric_limits<double>::infinity();
  bool eligible = false;
};

inline int SelectCell(const std::vector<CellDecision>& cells) {
  int selected = -1;
  auto better = [&](const CellDecision& lhs, const CellDecision& rhs) {
    const double lhs_max = std::max(lhs.chosen_ratio, lhs.policy_ratio);
    const double rhs_max = std::max(rhs.chosen_ratio, rhs.policy_ratio);
    if (lhs_max != rhs_max) return lhs_max < rhs_max;
    const double lhs_sum = lhs.chosen_ratio + lhs.policy_ratio;
    const double rhs_sum = rhs.chosen_ratio + rhs.policy_ratio;
    if (lhs_sum != rhs_sum) return lhs_sum < rhs_sum;
    const double lhs_min = std::min(lhs.chosen_pearson_gain,
                                    lhs.policy_pearson_gain);
    const double rhs_min = std::min(rhs.chosen_pearson_gain,
                                    rhs.policy_pearson_gain);
    if (lhs_min != rhs_min) return lhs_min > rhs_min;
    return lhs.fixed_rank < rhs.fixed_rank;
  };
  for (size_t i = 0; i < cells.size(); ++i) {
    if (cells[i].eligible &&
        (selected < 0 || better(cells[i], cells[selected]))) {
      selected = static_cast<int>(i);
    }
  }
  return selected;
}

inline int64_t DirectoryBytes(const std::filesystem::path& root) {
  int64_t total = 0;
  std::error_code ec;
  for (std::filesystem::recursive_directory_iterator it(root, ec), end;
       !ec && it != end; it.increment(ec)) {
    if (it->is_regular_file() && !it->is_symlink()) {
      const auto size = it->file_size(ec);
      if (ec || size > static_cast<uint64_t>(
                           std::numeric_limits<int64_t>::max() - total)) return -1;
      total += static_cast<int64_t>(size);
    }
  }
  return ec ? -1 : total;
}

inline bool WriteGlobalStatusLast(const std::filesystem::path& output_root,
                                  const json::Object& result,
                                  bool* owns_root, std::string* error) {
  if (owns_root == nullptr || !*owns_root) {
    if (error != nullptr) *error = "offline-Q output ownership missing";
    return false;
  }
  const auto path = output_root / "SCREEN_RESULT.json";
  const std::string contents = json::ToString(result, true) + "\n";
  const int64_t before = DirectoryBytes(output_root);
  if (before < 0 || before > kVrpoQOfflineRetainedByteCeiling ||
      contents.size() > static_cast<size_t>(
          kVrpoQOfflineRetainedByteCeiling - before)) {
    if (error != nullptr) {
      *error = "offline-Q final status would exceed retained-byte ceiling";
    }
    return false;
  }
  bool owns_status = false;
  if (!vrpo_q_warmup_internal::WriteAtomicTextNoReplace(
          path, contents, &owns_status, error) || !owns_status ||
      !VrpoFsyncDirectory(output_root, error) ||
      DirectoryBytes(output_root) > kVrpoQOfflineRetainedByteCeiling) {
    return false;
  }
  *owns_root = false;
  return true;
}

}  // namespace vrpo_q_offline_internal

struct VrpoQOfflineStartupConfig {
  std::string game;
  std::string init_mode;
  std::string profile;
  std::string registration_id;
  std::filesystem::path output_root;
  std::filesystem::path evidence_root;
  std::filesystem::path source_root;
  std::string source_code_sha256;
  std::string executed_binary_sha256;
  int64_t executed_binary_size = 0;
  bool runtime_device_is_cuda = false;
  bool rollout_amp = false;
  bool train_amp = false;
  bool allow_tf32 = false;
  bool pipeline = false;
  bool diagnostics_only = false;
  bool any_collection_or_training_side_path = false;
  double logit_cap = 10.0;
  double reward_scale = 4.0;
  double gamma = 1.0;
  double lambda = 1.0;
  std::array<double, 4> shaping = {0.0, 0.0, 0.0, 0.0};
  VrpoQOfflineEvidence evidence;
};

namespace vrpo_q_offline_internal {

inline json::Object EvidenceObservationJson(
    const VrpoPpoPilotEvidenceObservation& observation) {
  json::Object item;
  item["identity"] = observation.identity;
  item["relative_path"] = observation.relative_path;
  item["compiled_sha256"] = observation.compiled_sha256;
  item["cli_sha256"] = observation.cli_sha256;
  item["observed_sha256"] = observation.observed_sha256;
  item["observed_size"] = observation.observed_size;
  item["matched"] = observation.matched;
  return item;
}

inline json::Array RetryObservationsJson(
    const std::vector<VrpoPpoPilotEvidenceObservation>& observations) {
  json::Array result;
  for (const auto& observation : observations) {
    result.emplace_back(EvidenceObservationJson(observation));
  }
  return result;
}

inline json::Array TrainCorpusBindingsJson() {
  const auto files = VrpoQOfflineTrainCorpusFileSha256();
  const auto payloads = VrpoQOfflineTrainCorpusPayloadSha256();
  const auto sizes = VrpoQOfflineTrainCorpusFileSizes();
  json::Array result;
  for (int corpus = 0; corpus < kVrpoQOfflineCorpora; ++corpus) {
    const uint64_t first = kVrpoQWarmupTrainStartEpisodeId +
        static_cast<uint64_t>(corpus * kVrpoQWarmupGamesPerUpdate);
    json::Object item;
    item["corpus_index"] = static_cast<int64_t>(corpus);
    item["start_episode_id"] = static_cast<int64_t>(first);
    item["end_episode_id_inclusive"] = static_cast<int64_t>(first + 15);
    item["file_identity_schema"] = kVrpoQOfflineFileIdentitySchema;
    item["file_sha256"] = files[corpus];
    item["file_size"] = sizes[corpus];
    item["payload_identity_schema"] = kVrpoQWarmupCentralCorpusSchema;
    item["payload_sha256"] = payloads[corpus];
    result.emplace_back(std::move(item));
  }
  return result;
}

inline json::Object HoldoutCorpusBindingJson() {
  json::Object item;
  item["start_episode_id"] =
      static_cast<int64_t>(kVrpoQWarmupHoldoutStartEpisodeId);
  item["end_episode_id_inclusive"] =
      static_cast<int64_t>(kVrpoQWarmupHoldoutEndEpisodeIdInclusive);
  item["file_identity_schema"] = kVrpoQOfflineFileIdentitySchema;
  item["file_sha256"] = kVrpoQOfflineHoldoutFileSha256;
  item["file_size"] = kVrpoQOfflineHoldoutFileSize;
  item["payload_identity_schema"] = kVrpoQWarmupCentralCorpusSchema;
  item["payload_sha256"] = kVrpoQOfflineHoldoutPayloadSha256;
  return item;
}

inline void ApplyGlobalArtifactBinding(
    const VrpoQOfflineStartupConfig& startup,
    const std::string& actor_runtime_sha256,
    const std::string& actor_values_sha256, json::Object* artifact) {
  if (artifact == nullptr) return;
  (*artifact)["compiled_registration_id"] = kVrpoQOfflineRegistrationId;
  (*artifact)["compiled_profile"] = kVrpoQOfflineProfile;
  (*artifact)["purpose"] =
      "FROZEN_U4_OFFLINE_Q_TARGET_LR_EXPLORATORY_SCREEN";
  (*artifact)["source_code_sha256"] = startup.source_code_sha256;
  (*artifact)["executed_binary_sha256"] =
      startup.executed_binary_sha256;
  (*artifact)["executed_binary_size"] = startup.executed_binary_size;
  (*artifact)["immutable_retry_observations"] =
      RetryObservationsJson(startup.evidence.retry_observations);
  (*artifact)["train_corpus_bindings"] = TrainCorpusBindingsJson();
  (*artifact)["holdout_corpus_binding"] = HoldoutCorpusBindingJson();
  (*artifact)["actor_file_identity_schema"] =
      kVrpoQOfflineFileIdentitySchema;
  (*artifact)["actor_file_sha256"] = kVrpoQWarmupActorFileSha256;
  (*artifact)["actor_file_size"] = kVrpoQOfflineActorFileSize;
  (*artifact)["actor_values_schema"] = kVrpoQOfflineActorValuesSchema;
  (*artifact)["actor_values_sha256"] = actor_values_sha256;
  (*artifact)["actor_parameters_and_buffers_schema"] =
      kVrpoQOfflineActorRuntimeStateSchema;
  (*artifact)["actor_parameters_and_buffers_sha256"] =
      actor_runtime_sha256;
  (*artifact)["initial_q_file_identity_schema"] =
      kVrpoQWarmupQFileIdentitySchema;
  (*artifact)["initial_q_file_sha256"] = kVrpoQWarmupInitialQFileSha256;
  (*artifact)["initial_q_file_size"] = kVrpoQOfflineInitialQFileSize;
  (*artifact)["initial_q_canonical_values_schema"] =
      kVrpoQWarmupCanonicalQOriginSchema;
  (*artifact)["initial_q_canonical_values_sha256"] =
      kVrpoQWarmupInitialQCanonicalValuesSha256;
  (*artifact)["initial_q_runtime_module_values_schema"] =
      kVrpoQWarmupRuntimeQStateSchema;
  (*artifact)["initial_q_runtime_module_values_sha256"] =
      kVrpoQWarmupInitialQRuntimeValuesSha256;
}

inline void ApplyCellArtifactBinding(
    const VrpoQOfflineStartupConfig& startup,
    const VrpoQOfflineCellConfig& cell, const CellTrainingStats& stats,
    json::Object* artifact) {
  if (artifact == nullptr) return;
  (*artifact)["compiled_registration_id"] = kVrpoQOfflineRegistrationId;
  (*artifact)["compiled_profile"] = kVrpoQOfflineProfile;
  (*artifact)["purpose"] = "OFFLINE_Q_TARGET_LR_SCREEN_CELL";
  (*artifact)["source_code_sha256"] = startup.source_code_sha256;
  (*artifact)["executed_binary_sha256"] =
      startup.executed_binary_sha256;
  (*artifact)["executed_binary_size"] = startup.executed_binary_size;
  (*artifact)["initial_q_file_identity_schema"] =
      kVrpoQWarmupQFileIdentitySchema;
  (*artifact)["initial_q_file_sha256"] = kVrpoQWarmupInitialQFileSha256;
  (*artifact)["initial_q_file_size"] = kVrpoQOfflineInitialQFileSize;
  (*artifact)["initial_q_canonical_values_schema"] =
      kVrpoQWarmupCanonicalQOriginSchema;
  (*artifact)["initial_q_canonical_values_sha256"] =
      kVrpoQWarmupInitialQCanonicalValuesSha256;
  (*artifact)["q_runtime_module_values_schema"] =
      kVrpoQWarmupRuntimeQStateSchema;
  (*artifact)["q_runtime_module_values_before_sha256"] =
      stats.q_before_sha256;
  (*artifact)["q_runtime_module_values_after_sha256"] =
      stats.q_after_sha256;
  (*artifact)["target_values_identity_schema"] =
      "dune_vrpo_phase4d_target_values_v1";
  (*artifact)["pre_target_q_runtime_module_values_schema"] =
      kVrpoQWarmupRuntimeQStateSchema;
  json::Array target_values;
  json::Array pre_q;
  json::Array target_q_source;
  json::Array target_q_source_schema;
  for (int corpus = 0; corpus < kVrpoQOfflineCorpora; ++corpus) {
    target_values.emplace_back(stats.target_sha256[corpus]);
    pre_q.emplace_back(stats.pre_target_q_runtime_sha256[corpus]);
    target_q_source.emplace_back(
        stats.target_q_runtime_source_sha256[corpus]);
    target_q_source_schema.emplace_back(
        cell.target_kind == VrpoQOfflineTargetKind::kTdLambda
            ? kVrpoQWarmupRuntimeQStateSchema
            : "NOT_APPLICABLE_MONTE_CARLO_NO_BOOTSTRAP");
  }
  (*artifact)["target_values_sha256_by_corpus"] = std::move(target_values);
  (*artifact)["pre_target_q_runtime_module_values_sha256_by_corpus"] =
      std::move(pre_q);
  (*artifact)["target_q_source_sha256_by_corpus"] =
      std::move(target_q_source);
  (*artifact)["target_q_source_schema_by_corpus"] =
      std::move(target_q_source_schema);
  (*artifact)["targets_refreshed_from_exact_pre_q_per_corpus"] =
      cell.target_kind == VrpoQOfflineTargetKind::kTdLambda;
  (*artifact)["monte_carlo_targets_have_no_q_bootstrap"] =
      cell.target_kind == VrpoQOfflineTargetKind::kMonteCarlo;
}

inline bool BoundFieldsMatch(const json::Object& artifact,
                             const json::Object& expected,
                             std::string* error) {
  for (const auto& field : expected) {
    const auto observed = artifact.find(field.first);
    if (observed == artifact.end() ||
        json::ToString(observed->second) != json::ToString(field.second)) {
      if (error != nullptr) {
        *error = "offline-Q artifact binding omitted/substituted field: " +
            field.first;
      }
      return false;
    }
  }
  return true;
}

inline bool ValidateGlobalArtifactBinding(
    const json::Object& artifact, const VrpoQOfflineStartupConfig& startup,
    const std::string& actor_runtime_sha256,
    const std::string& actor_values_sha256, std::string* error) {
  json::Object expected;
  ApplyGlobalArtifactBinding(startup, actor_runtime_sha256,
                             actor_values_sha256, &expected);
  return vrpo_q_warmup_internal::StringIs(
             artifact, "registration_id", startup.registration_id) &&
      vrpo_q_warmup_internal::StringIs(
          artifact, "profile", startup.profile) &&
      BoundFieldsMatch(artifact, expected, error);
}

inline bool ValidateCellArtifactBinding(
    const json::Object& artifact, const VrpoQOfflineStartupConfig& startup,
    const VrpoQOfflineCellConfig& cell, const CellTrainingStats& stats,
    std::string* error) {
  json::Object expected;
  ApplyCellArtifactBinding(startup, cell, stats, &expected);
  return vrpo_q_warmup_internal::StringIs(
             artifact, "registration_id", startup.registration_id) &&
      vrpo_q_warmup_internal::StringIs(artifact, "cell_id", cell.id) &&
      BoundFieldsMatch(artifact, expected, error);
}

}  // namespace vrpo_q_offline_internal

inline std::vector<std::string> VrpoQOfflineSourceRelativePaths() {
  auto paths = VrpoQWarmupSourceRelativePaths();
  paths.push_back("open_spiel/examples/dune_vrpo_q_offline_screen.h");
  return paths;
}

inline bool LoadVrpoQOfflineSourceIdentity(
    const std::filesystem::path& source_root,
    const std::string& registered_sha256, VrpoPhase4eSourceIdentity* output,
    std::string* error) {
  auto fail = [&](const std::string& message) {
    if (error != nullptr) *error = message;
    if (output != nullptr) *output = VrpoPhase4eSourceIdentity{};
    return false;
  };
  if (output == nullptr || !VrpoPhase4eLowerHex64(registered_sha256)) {
    return fail("offline-Q source identity invalid");
  }
  std::error_code ec;
  const auto root = std::filesystem::canonical(source_root, ec);
  if (ec || !std::filesystem::is_directory(root)) {
    return fail("offline-Q source root unreadable");
  }
  std::set<std::string> seen;
  std::string payload;
  for (const auto& relative : VrpoQOfflineSourceRelativePaths()) {
    if (relative.empty() || relative.find("..") != std::string::npos ||
        !seen.insert(relative).second) return fail("offline-Q source list invalid");
    const auto path = root / relative;
    size_t size = 0;
    const auto digest = ComputeFileSHA256(path.string(), &size);
    if (!std::filesystem::is_regular_file(path) ||
        std::filesystem::is_symlink(path) || size == 0 ||
        !VrpoPhase4eLowerHex64(digest)) return fail("offline-Q source file invalid");
    payload.append(relative);
    payload.push_back('\0');
    payload.append(digest);
    payload.push_back('\n');
  }
  const auto digest = ComputeStringSHA256(payload);
  if (digest != registered_sha256) return fail("offline-Q source digest mismatch");
  output->canonical_root = root;
  output->relative_paths = VrpoQOfflineSourceRelativePaths();
  output->combined_sha256 = digest;
  return true;
}

inline bool ValidateVrpoQOfflineStartup(const VrpoQOfflineStartupConfig& config,
                                        std::string* error,
                                        bool test_fixture = false) {
  auto fail = [&](const std::string& message) {
    if (error != nullptr) *error = message;
    return false;
  };
  if (config.game != "dune_imperium" ||
      config.init_mode != "vrpo_q_offline_screen" ||
      config.profile != kVrpoQOfflineProfile ||
      (!test_fixture && config.registration_id != kVrpoQOfflineRegistrationId) ||
      (test_fixture && config.registration_id.empty()) ||
      config.output_root.empty() || config.evidence_root.empty() ||
      config.source_root.empty() ||
      !VrpoPhase4eLowerHex64(config.source_code_sha256) ||
      !VrpoPhase4eLowerHex64(config.executed_binary_sha256) ||
      config.executed_binary_size <= 0 ||
      config.evidence.origin.actor_path.empty() ||
      config.evidence.origin.q_model_path.empty() ||
      config.evidence.retry_observations.empty()) {
    return fail("offline-Q identity/evidence config invalid");
  }
  if (std::filesystem::exists(config.output_root) ||
      !config.runtime_device_is_cuda || config.rollout_amp || config.train_amp ||
      !config.allow_tf32 || config.pipeline || config.diagnostics_only ||
      config.any_collection_or_training_side_path || config.logit_cap != 10.0 ||
      config.reward_scale != 4.0 || config.gamma != 1.0 ||
      config.lambda != 1.0) return fail("offline-Q execution contract invalid");
  for (double value : config.shaping) {
    if (!std::isfinite(value) || value != 0.0) {
      return fail("offline-Q shaping must be exactly zero");
    }
  }
  if (kVrpoQOfflineFinalSteps !=
          kVrpoQOfflineCorpora * kVrpoQOfflineStepsPerCorpus ||
      kVrpoQOfflineStepsPerCorpus != kVrpoTrainingMinibatchesPerEpoch) {
    return fail("offline-Q compiled schedule inconsistent");
  }
  return true;
}

inline bool RunVrpoQOfflineScreen(
    const VrpoQOfflineStartupConfig& startup,
    const VrpoPhase4eSourceIdentity& source_identity,
    const std::shared_ptr<SharedDunePolicyValueNetImpl>& actor,
    const torch::Device& device, const VrpoExpandedExpectedLayout& layout,
    const VrpoQOfflineDeadline& deadline, json::Object* output,
    std::string* error) {
  using namespace vrpo_q_offline_internal;
  auto fail = [&](const std::string& message, bool* owns_root) {
    if (owns_root != nullptr && *owns_root) {
      vrpo_ppo_pilot_internal::CleanupPath(startup.output_root);
      *owns_root = false;
    }
    if (error != nullptr) *error = message;
    if (output != nullptr) output->clear();
    return false;
  };
  bool owns_root = false;
  if (output == nullptr || actor == nullptr || !device.is_cuda() ||
      !ValidateVrpoQOfflineStartup(startup, error) ||
      source_identity.combined_sha256 != startup.source_code_sha256 ||
      !deadline.Check("before output claim", error)) {
    return fail(error != nullptr && !error->empty() ? *error
                                                     : "offline-Q startup failed",
                &owns_root);
  }
  VrpoPhase4eResolvedPaths resolved_paths;
  if (!ResolveVrpoPhase4ePaths(startup.evidence_root, startup.output_root,
                               &resolved_paths, error)) {
    return fail(error != nullptr ? *error
                                 : "offline-Q input/output containment failed",
                &owns_root);
  }
  std::error_code ec;
  const auto parent = startup.output_root.parent_path();
  const auto space = std::filesystem::space(parent, ec);
  if (ec || space.available < static_cast<uint64_t>(kVrpoQOfflinePeakByteCeiling) ||
      !vrpo_q_warmup_internal::ClaimFreshDirectory(
          startup.output_root, &owns_root, error)) {
    return fail(error != nullptr && !error->empty() ? *error
                                                     : "offline-Q resource/claim failed",
                &owns_root);
  }
  int64_t measured_peak_bytes = 0;
  auto observe = [&]() {
    const int64_t bytes = DirectoryBytes(startup.output_root);
    measured_peak_bytes = std::max(measured_peak_bytes, bytes);
    return bytes >= 0 && bytes <= kVrpoQOfflinePeakByteCeiling;
  };

  if (!LoadVrpoModule(*actor, startup.evidence.origin.actor_path, error)) {
    return fail(*error, &owns_root);
  }
  actor->to(device, torch::kFloat32);
  actor->eval();
  for (auto& item : actor->named_parameters()) {
    item.value().set_requires_grad(false);
    item.value().mutable_grad() = torch::Tensor();
  }
  const std::string actor_state =
      vrpo_q_warmup_internal::ModuleParametersAndBuffersSha256(*actor, error);
  std::string actor_values;
  if (actor_state != kVrpoQOfflineActorRuntimeSha256 ||
      !vrpo_training_internal::ModuleValueSha256(
          *actor, "", &actor_values, error) ||
      actor_values != kVrpoQWarmupActorValuesSha256) {
    return fail("offline-Q exact U4 actor identity rejected", &owns_root);
  }
  VrpoActorForward actor_forward = [actor](const torch::Tensor& input) {
    torch::NoGradGuard guard;
    const auto out = actor->forward(input);
    return VrpoActorTrainingOutput{out.logits.detach(), out.values.detach()};
  };

  const auto corpus_file_sha = VrpoQOfflineTrainCorpusFileSha256();
  const auto corpus_payload_sha = VrpoQOfflineTrainCorpusPayloadSha256();
  const auto corpus_file_sizes = VrpoQOfflineTrainCorpusFileSizes();
  std::vector<std::filesystem::path> cell_dirs;
  std::vector<CellTrainingStats> training_stats;
  std::set<uint64_t> training_ids;
  const auto cells = CanonicalVrpoQOfflineCells();

  for (const auto& cell : cells) {
    if (!deadline.Check("before cell " + cell.id, error)) {
      return fail(*error, &owns_root);
    }
    auto q = std::make_shared<DuneVrpoQNetImpl>(kVrpoQWarmupQSeed);
    q->to(device, torch::kFloat32);
    if (!LoadVrpoModule(*q, startup.evidence.origin.q_model_path, error)) {
      return fail(*error, &owns_root);
    }
    std::string q_canonical;
    std::string q_runtime;
    if (!VrpoExpandedQValueIdentitySha256(
            *q, layout, &q_canonical, error) ||
        !vrpo_training_internal::ModuleValueSha256(
            *q, "", &q_runtime, error) ||
        q_canonical != kVrpoQWarmupInitialQCanonicalValuesSha256 ||
        q_runtime != kVrpoQWarmupInitialQRuntimeValuesSha256) {
      return fail("offline-Q bootstrap Q identity rejected", &owns_root);
    }
    std::unique_ptr<torch::optim::AdamW> optimizer;
    if (!MakeFreshOptimizer(*q, cell.learning_rate, &optimizer, error)) {
      return fail(*error, &owns_root);
    }
    CellTrainingStats stats;
    stats.actor_before_sha256 = actor_state;
    stats.q_before_sha256 = q_runtime;
    if (!OptimizerStateSha256(*optimizer, *q, cell.learning_rate, 0,
                              &stats.optimizer_before_sha256, error)) {
      return fail(*error, &owns_root);
    }
    vrpo_q_warmup_internal::QTransaction transaction;
    if (!transaction.Capture(*q, *optimizer, error)) {
      return fail(*error, &owns_root);
    }
    for (int corpus = 0; corpus < 4; ++corpus) {
      std::vector<VrpoTrainingEpisode> episodes;
      VrpoScheduleCorpusIdentity identity;
      std::set<uint64_t> local_ids;
      const uint64_t first = kVrpoQWarmupTrainStartEpisodeId + corpus * 16;
      if (!LoadCorpus(startup.evidence.train_corpora[corpus],
                      corpus_file_sha[corpus], corpus_file_sizes[corpus],
                      corpus_payload_sha[corpus],
                      first, first + 15, &episodes, &identity, &local_ids,
                      error) ||
          (cell.fixed_rank == 0 && [&]() {
             for (uint64_t id : local_ids) {
               if (!training_ids.insert(id).second) return false;
             }
             return true;
           }() == false)) {
        return fail(error != nullptr && !error->empty()
                        ? *error : "offline-Q corpus leakage/identity failed",
                    &owns_root);
      }
      VrpoQForward q_forward = [q](const torch::Tensor& input) {
        torch::Tensor out;
        std::string forward_error;
        if (!q->ForwardChecked(input, &out, &forward_error)) {
          throw std::runtime_error(forward_error);
        }
        return out;
      };
      if (!TrainOneCorpus(cell, corpus, episodes, *actor, *q, *optimizer,
                          actor_forward, q_forward, &stats, error) ||
          !deadline.Check("after corpus", error)) {
        return fail(*error, &owns_root);
      }
    }
    optimizer->zero_grad();
    stats.actor_after_sha256 =
        vrpo_q_warmup_internal::ModuleParametersAndBuffersSha256(*actor, error);
    if (!vrpo_training_internal::ModuleValueSha256(
            *q, "", &stats.q_after_sha256, error) ||
        !OptimizerStateSha256(*optimizer, *q, cell.learning_rate,
                              kVrpoQOfflineFinalSteps,
                              &stats.optimizer_after_sha256, error) ||
        stats.actor_after_sha256 != actor_state ||
        stats.q_after_sha256 == stats.q_before_sha256 ||
        stats.optimizer_steps != kVrpoQOfflineFinalSteps ||
        stats.backward_calls != kVrpoQOfflineFinalSteps ||
        stats.target_refreshes != 4 || stats.rows_seen <= 0 ||
        !std::isfinite(stats.mean_loss) ||
        !std::isfinite(stats.max_grad_norm)) {
      return fail(error != nullptr && !error->empty()
                      ? *error : "offline-Q cell postconditions failed",
                  &owns_root);
    }
    stats.mean_loss /= kVrpoQOfflineFinalSteps;
    transaction.Commit();

    json::Object preliminary;
    preliminary["schema"] = kVrpoQOfflineCellSchema;
    preliminary["registration_id"] = startup.registration_id;
    preliminary["cell_id"] = cell.id;
    preliminary["target_kind"] = cell.target_kind == VrpoQOfflineTargetKind::kTdLambda
        ? "EXPECTED_SARSA_LAMBDA_CURRENT_Q" : "DIRECT_TERMINAL_MONTE_CARLO";
    preliminary["learning_rate"] = VrpoPhase4eCanonicalDouble(cell.learning_rate);
    preliminary["epochs_per_corpus"] = int64_t{1};
    preliminary["minibatches_per_corpus"] = int64_t{16};
    preliminary["q_optimizer_steps"] = stats.optimizer_steps;
    preliminary["actor_optimizer_constructed"] = false;
    preliminary["actor_optimizer_steps"] = int64_t{0};
    preliminary["actor_backward_calls"] = int64_t{0};
    preliminary["actor_state_before_sha256"] = stats.actor_before_sha256;
    preliminary["actor_state_after_sha256"] = stats.actor_after_sha256;
    preliminary["q_origin_sha256"] = stats.q_before_sha256;
    preliminary["q_final_sha256"] = stats.q_after_sha256;
    preliminary["q_optimizer_origin_sha256"] = stats.optimizer_before_sha256;
    preliminary["q_optimizer_final_sha256"] = stats.optimizer_after_sha256;
    preliminary["mean_loss"] = VrpoPhase4eCanonicalDouble(stats.mean_loss);
    preliminary["max_q_grad_norm"] = VrpoPhase4eCanonicalDouble(stats.max_grad_norm);
    json::Array targets;
    json::Array partitions;
    json::Array boundaries;
    for (int i = 0; i < 4; ++i) {
      targets.emplace_back(stats.target_sha256[i]);
      partitions.emplace_back(stats.partition_sha256[i]);
      boundaries.emplace_back(stats.q_boundary_sha256[i]);
    }
    preliminary["target_sha256_by_corpus"] = std::move(targets);
    preliminary["partition_sha256_by_corpus"] = std::move(partitions);
    preliminary["q_boundary_sha256_by_corpus"] = std::move(boundaries);
    json::Array partition_seeds;
    json::Array expected_partitions;
    for (int i = 0; i < kVrpoQOfflineCorpora; ++i) {
      partition_seeds.emplace_back(
          std::to_string(VrpoQOfflinePartitionSeed(i)));
      expected_partitions.emplace_back(
          VrpoQOfflineExpectedPartitionSha256()[i]);
    }
    preliminary["registered_partition_seed_by_corpus"] =
        std::move(partition_seeds);
    preliminary["registered_expected_partition_sha256_by_corpus"] =
        std::move(expected_partitions);
    preliminary["holdout_used_for_training"] = false;
    preliminary["strict_reload_and_canary_passed"] = true;
    ApplyCellArtifactBinding(startup, cell, stats, &preliminary);
    const auto temp = startup.output_root / ("." + cell.id + ".trained");
    if (!ValidateCellArtifactBinding(
            preliminary, startup, cell, stats, error) ||
        !SaveCellArchive(temp, *q, *optimizer, cell, preliminary, device,
                         layout, error) || !observe()) {
      return fail(error != nullptr && !error->empty()
                      ? *error : "offline-Q cell archive/resource failed",
                  &owns_root);
    }
    cell_dirs.push_back(temp);
    training_stats.push_back(std::move(stats));
  }

  bool cross_cell_partition_match = true;
  const auto expected_partitions = VrpoQOfflineExpectedPartitionSha256();
  for (int corpus = 0; corpus < kVrpoQOfflineCorpora; ++corpus) {
    for (int cell = 0; cell < kVrpoQOfflineCells; ++cell) {
      if (training_stats[cell].partition_sha256[corpus] !=
              expected_partitions[corpus] ||
          training_stats[cell].partition_sha256[corpus] !=
              training_stats[0].partition_sha256[corpus]) {
        cross_cell_partition_match = false;
      }
    }
  }
  if (!cross_cell_partition_match) {
    return fail("offline-Q cross-cell partition mismatch", &owns_root);
  }

  std::vector<VrpoTrainingEpisode> holdout;
  VrpoScheduleCorpusIdentity holdout_identity;
  std::set<uint64_t> all_ids = training_ids;
  if (!LoadCorpus(startup.evidence.holdout_corpus,
                  kVrpoQOfflineHoldoutFileSha256,
                  kVrpoQOfflineHoldoutFileSize,
                  kVrpoQOfflineHoldoutPayloadSha256,
                  kVrpoQWarmupHoldoutStartEpisodeId,
                  kVrpoQWarmupHoldoutEndEpisodeIdInclusive, &holdout,
                  &holdout_identity, &all_ids, error) ||
      all_ids.size() != 80 || training_ids.size() != 64) {
    return fail(error != nullptr && !error->empty()
                    ? *error : "offline-Q holdout leakage/identity failed",
                &owns_root);
  }
  auto initial_q = std::make_shared<DuneVrpoQNetImpl>(kVrpoQWarmupQSeed);
  initial_q->to(device, torch::kFloat32);
  if (!LoadVrpoModule(*initial_q, startup.evidence.origin.q_model_path, error)) {
    return fail(*error, &owns_root);
  }
  VrpoQForward initial_forward = [initial_q](const torch::Tensor& input) {
    torch::Tensor out;
    std::string forward_error;
    if (!initial_q->ForwardChecked(input, &out, &forward_error)) {
      throw std::runtime_error(forward_error);
    }
    return out;
  };
  vrpo_q_warmup_internal::MetricAccumulator initial_accumulator;
  std::string initial_rows;
  bool initial_adapter = false;
  VrpoQWarmupMetricSet initial_metrics;
  if (!vrpo_q_warmup_internal::EvaluateOneQ(
          holdout, *actor, *initial_q, actor_forward, initial_forward,
          &initial_accumulator, &initial_rows, &initial_adapter, error) ||
      !vrpo_q_warmup_internal::FinishMetricSet(
          initial_accumulator, &initial_metrics, error) ||
      !initial_adapter) return fail(*error, &owns_root);

  std::vector<CellDecision> decisions;
  json::Array result_cells;
  for (int index = 0; index < kVrpoQOfflineCells; ++index) {
    const auto& cell = cells[index];
    auto q = std::make_shared<DuneVrpoQNetImpl>(kVrpoQWarmupQSeed);
    q->to(device, torch::kFloat32);
    if (!LoadVrpoModule(*q, cell_dirs[index] / "q_model.pt", error)) {
      return fail(*error, &owns_root);
    }
    VrpoQForward forward = [q](const torch::Tensor& input) {
      torch::Tensor out;
      std::string forward_error;
      if (!q->ForwardChecked(input, &out, &forward_error)) {
        throw std::runtime_error(forward_error);
      }
      return out;
    };
    vrpo_q_warmup_internal::MetricAccumulator accumulator;
    std::string row_identity;
    bool adapter = false;
    VrpoQWarmupMetricSet final_metrics;
    if (!vrpo_q_warmup_internal::EvaluateOneQ(
            holdout, *actor, *q, actor_forward, forward, &accumulator,
            &row_identity, &adapter, error) ||
        row_identity != initial_rows ||
        accumulator.targets != initial_accumulator.targets ||
        !vrpo_q_warmup_internal::FinishMetricSet(
            accumulator, &final_metrics, error)) {
      return fail(error != nullptr && !error->empty()
                      ? *error : "offline-Q holdout pairing failed",
                  &owns_root);
    }
    CellDecision decision;
    decision.id = cell.id;
    decision.fixed_rank = cell.fixed_rank;
    decision.chosen_ratio = final_metrics.chosen_q.mse / initial_metrics.chosen_q.mse;
    decision.policy_ratio = final_metrics.policy_v.mse / initial_metrics.policy_v.mse;
    decision.chosen_pearson_gain = final_metrics.chosen_q.pearson -
                                   initial_metrics.chosen_q.pearson;
    decision.policy_pearson_gain = final_metrics.policy_v.pearson -
                                   initial_metrics.policy_v.pearson;
    decision.eligible = adapter &&
        VrpoQWarmupImprovementPass(initial_metrics, final_metrics, true,
                                   true, true);
    decisions.push_back(decision);
    json::Object summary;
    summary["cell_id"] = cell.id;
    summary["initial"] = MetricSetJson(initial_metrics);
    summary["final"] = MetricSetJson(final_metrics);
    summary["chosen_q_mse_ratio"] = VrpoPhase4eCanonicalDouble(decision.chosen_ratio);
    summary["policy_v_mse_ratio"] = VrpoPhase4eCanonicalDouble(decision.policy_ratio);
    summary["chosen_q_pearson_gain"] = VrpoPhase4eCanonicalDouble(decision.chosen_pearson_gain);
    summary["policy_v_pearson_gain"] = VrpoPhase4eCanonicalDouble(decision.policy_pearson_gain);
    summary["eligible"] = decision.eligible;
    summary["row_identity_sha256"] = row_identity;
    summary["holdout_payload_sha256"] = holdout_identity.payload_sha256;
    summary["schema"] = kVrpoQOfflineCellSchema;
    summary["registration_id"] = startup.registration_id;
    summary["target_kind"] =
        cell.target_kind == VrpoQOfflineTargetKind::kTdLambda
        ? "EXPECTED_SARSA_LAMBDA_CURRENT_Q"
        : "DIRECT_TERMINAL_MONTE_CARLO";
    summary["learning_rate"] =
        VrpoPhase4eCanonicalDouble(cell.learning_rate);
    summary["actor_frozen"] = true;
    summary["actor_optimizer_constructed"] = false;
    summary["actor_optimizer_steps"] = int64_t{0};
    summary["q_optimizer_steps"] = kVrpoQOfflineFinalSteps;
    summary["q_origin_sha256"] = training_stats[index].q_before_sha256;
    summary["q_final_sha256"] = training_stats[index].q_after_sha256;
    summary["q_optimizer_origin_sha256"] =
        training_stats[index].optimizer_before_sha256;
    summary["q_optimizer_final_sha256"] =
        training_stats[index].optimizer_after_sha256;
    summary["strict_reload_and_canary_passed"] = true;
    summary["cross_cell_partition_match"] = cross_cell_partition_match;
    summary["holdout_used_for_training"] = false;
    summary["holdout_used_for_selection"] = true;
    summary["selected_metrics_are_selection_biased"] = true;
    summary["selection_multiplicity"] = int64_t{kVrpoQOfflineCells};
    summary["selection_evidence_classification"] =
        "SCREEN_ONLY_NOT_CONFIRMATORY_FRESH_DATA_REQUIRED";
    summary["authorizes_fresh_data_confirmation_design"] =
        decision.eligible;
    summary["authorizes_actor_training"] = false;
    summary["authorizes_longer_training"] = false;
    summary["authorizes_playing_strength_evaluation"] = false;
    summary["authorizes_promotion"] = false;
    ApplyCellArtifactBinding(startup, cell, training_stats[index], &summary);
    if (!ValidateSelectionBiasMetadata(summary, error) ||
        !ValidateCellArtifactBinding(
            summary, startup, cell, training_stats[index], error) ||
        !FinalizeCellArchive(cell_dirs[index], cell, summary, error)) {
      return fail(error != nullptr && !error->empty()
                      ? *error : "offline-Q cell finalization failed",
                  &owns_root);
    }
    result_cells.emplace_back(std::move(summary));
    if (!deadline.Check("after holdout cell " + cell.id, error)) {
      return fail(*error, &owns_root);
    }
  }
  const int selected = SelectCell(decisions);
  for (int index = 0; index < kVrpoQOfflineCells; ++index) {
    const auto final_dir = startup.output_root / cells[index].id;
    bool moved = false;
    if (!vrpo_q_warmup_internal::RenameDirectoryNoReplace(
            cell_dirs[index], final_dir, &moved, error) || !moved ||
        !VrpoFsyncDirectory(startup.output_root, error)) {
      return fail(*error, &owns_root);
    }
  }
  if (!RetryEvidenceStillExact(startup.evidence_root, startup.evidence, error)) {
    return fail(*error, &owns_root);
  }
  VrpoPhase4eSourceIdentity source_after;
  if (!LoadVrpoQOfflineSourceIdentity(startup.source_root,
                                     startup.source_code_sha256,
                                     &source_after, error) ||
      source_after.combined_sha256 != source_identity.combined_sha256) {
    return fail(error != nullptr && !error->empty()
                    ? *error : "offline-Q source changed during run",
                &owns_root);
  }
  const int64_t retained_before_status = DirectoryBytes(startup.output_root);
  measured_peak_bytes = std::max(measured_peak_bytes, retained_before_status);
  if (retained_before_status < 0 ||
      retained_before_status > kVrpoQOfflineRetainedByteCeiling ||
      measured_peak_bytes > kVrpoQOfflinePeakByteCeiling ||
      !deadline.Check("before global status", error)) {
    return fail(error != nullptr && !error->empty()
                    ? *error : "offline-Q final resource/deadline gate failed",
                &owns_root);
  }
  json::Object result;
  result["schema"] = kVrpoQOfflineResultSchema;
  result["status"] = "VALID";
  result["classification"] = selected >= 0
      ? "VALID_EXPLORATORY_Q_CELL_SELECTED"
      : "VALID_NO_ELIGIBLE_Q_CELL_STOP";
  result["epistemic_label"] =
      "exploratory_offline_Q_target_LR_screen_not_playing_strength_evidence";
  result["registration_id"] = startup.registration_id;
  result["profile"] = startup.profile;
  result["cells"] = std::move(result_cells);
  result["selected_cell"] = selected >= 0 ? cells[selected].id : "NONE";
  result["selected_count"] = static_cast<int64_t>(selected >= 0 ? 1 : 0);
  result["train_episode_count"] = int64_t{64};
  result["holdout_episode_count"] = int64_t{16};
  result["holdout_used_for_training"] = false;
  result["holdout_used_for_selection"] = true;
  result["selected_metrics_are_selection_biased"] = true;
  result["selection_multiplicity"] = int64_t{kVrpoQOfflineCells};
  result["selection_evidence_classification"] =
      "SCREEN_ONLY_NOT_CONFIRMATORY_FRESH_DATA_REQUIRED";
  result["actor_frozen"] = true;
  result["actor_optimizer_constructed"] = false;
  result["actor_optimizer_steps"] = int64_t{0};
  result["environment_collection_calls"] = int64_t{0};
  result["q_steps_per_cell"] = kVrpoQOfflineFinalSteps;
  result["all_cells_independent_origin"] = true;
  result["all_cells_fresh_q_adam_step0"] = true;
  result["cross_cell_partition_match"] = cross_cell_partition_match;
  result["source_and_input_immutable"] = true;
  result["retained_bytes_before_status"] = retained_before_status;
  result["retained_byte_ceiling"] = kVrpoQOfflineRetainedByteCeiling;
  result["measured_peak_bytes"] = measured_peak_bytes;
  result["peak_byte_ceiling"] = kVrpoQOfflinePeakByteCeiling;
  result["elapsed_seconds"] = VrpoPhase4eCanonicalDouble(deadline.ElapsedSeconds());
  result["status_last"] = true;
  result["authorizes_fresh_data_q_warmstart_confirmation_design"] =
      selected >= 0;
  result["authorizes_vrpo_actor_training"] = false;
  result["authorizes_longer_training"] = false;
  result["authorizes_playing_strength_evaluation"] = false;
  result["authorizes_promotion"] = false;
  result["required_action"] = selected >= 0
      ? "design one fresh-data Q warmstart confirmation; do not train an actor"
      : "STOP current Q targets";
  ApplyGlobalArtifactBinding(startup, actor_state, actor_values, &result);
  if (!ValidateSelectionBiasMetadata(result, error) ||
      !ValidateGlobalArtifactBinding(
          result, startup, actor_state, actor_values, error) ||
      !WriteGlobalStatusLast(startup.output_root, result, &owns_root, error)) {
    return fail(*error, &owns_root);
  }
  *output = std::move(result);
  return true;
}

}  // namespace open_spiel
#endif  // OPEN_SPIEL_BUILD_WITH_LIBTORCH

#endif  // OPEN_SPIEL_EXAMPLES_DUNE_VRPO_Q_OFFLINE_SCREEN_H_
