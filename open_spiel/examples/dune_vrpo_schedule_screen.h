#ifndef OPEN_SPIEL_EXAMPLES_DUNE_VRPO_SCHEDULE_SCREEN_H_
#define OPEN_SPIEL_EXAMPLES_DUNE_VRPO_SCHEDULE_SCREEN_H_

// Isolated PPO schedule-health screen.  This is deliberately a terminal
// diagnostic path: it owns one sealed rollout, replays four independent PPO
// cells, and emits health evidence only.  It cannot be used by phase4e or by
// the ordinary PPO update loop.

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <limits>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "dune_vrpo_phase4e.h"

#ifdef OPEN_SPIEL_BUILD_WITH_LIBTORCH
#include <torch/torch.h>

namespace open_spiel {

inline constexpr char kVrpoScheduleProfile[] = "ultralow_lr_v3";
inline constexpr char kVrpoScheduleCorpusSchema[] =
    "dune_vrpo_schedule_actor_corpus_v3";
inline constexpr char kVrpoScheduleCellResultSchema[] =
    "dune_vrpo_schedule_cell_result_v3";
inline constexpr char kVrpoScheduleResultSchema[] =
    "dune_vrpo_schedule_health_screen_v3";
inline constexpr char kVrpoScheduleQRole[] =
    "SOURCE_ARCHIVE_PROVENANCE_ONLY";
inline std::atomic<int64_t> g_vrpo_schedule_q_target_computations{0};

inline void ResetVrpoScheduleQTargetInstrumentation() {
  g_vrpo_schedule_q_target_computations.store(0);
}

inline int64_t VrpoScheduleQTargetComputations() {
  return g_vrpo_schedule_q_target_computations.load();
}

inline void InjectVrpoScheduleQTargetComputationForTest() {
  g_vrpo_schedule_q_target_computations.fetch_add(1);
}
inline constexpr int kVrpoScheduleGames = 16;
inline constexpr int64_t kVrpoScheduleMaxActorSteps = 80;
inline constexpr int64_t kVrpoScheduleExpectedRowPresentations = 71475;
inline constexpr int64_t kVrpoScheduleRetainedByteCeiling =
    1879048192LL;  // 1.75 GiB.
inline constexpr int64_t kVrpoScheduleTemporaryByteCeiling =
    2415919104LL;  // 2.25 GiB.

struct VrpoScheduleCellSpec {
  std::string cell_id;
  double learning_rate = 0.0;
  int actor_epochs = 0;
};

class VrpoScheduleRunDeadline {
 public:
  VrpoScheduleRunDeadline() = default;

  static VrpoScheduleRunDeadline Start(
      std::chrono::steady_clock::time_point start,
      std::chrono::seconds limit = std::chrono::seconds(1200)) {
    VrpoScheduleRunDeadline out;
    out.active_ = true;
    out.start_ = start;
    out.deadline_ = start + limit;
    return out;
  }

  static VrpoScheduleRunDeadline ExpiredForTest() {
    return Start(std::chrono::steady_clock::now() -
                     std::chrono::seconds(1201),
                 std::chrono::seconds(1200));
  }

  static VrpoScheduleRunDeadline ExpireAfterChecksForTest(int checks) {
    VrpoScheduleRunDeadline out = Start(
        std::chrono::steady_clock::now(), std::chrono::hours(1));
    out.test_checks_remaining_ = checks;
    return out;
  }

  bool Check(const std::string& stage, std::string* error) const {
    if (test_checks_remaining_ >= 0) {
      if (test_checks_remaining_ == 0) {
        if (error != nullptr) {
          *error = "injected schedule deadline exceeded at " + stage;
        }
        return false;
      }
      --test_checks_remaining_;
    }
    if (!active_ || std::chrono::steady_clock::now() < deadline_) return true;
    if (error != nullptr) {
      *error = "schedule-screen 1200-second run deadline exceeded at " + stage;
    }
    return false;
  }

  double ElapsedSeconds() const {
    if (!active_) return 0.0;
    return std::chrono::duration<double>(
               std::chrono::steady_clock::now() - start_)
        .count();
  }

  bool active() const { return active_; }

 private:
  bool active_ = false;
  std::chrono::steady_clock::time_point start_{};
  std::chrono::steady_clock::time_point deadline_{};
  mutable int test_checks_remaining_ = -1;
};

inline const std::array<VrpoScheduleCellSpec, 4>*
FindVrpoScheduleCellsForProfile(const std::string& profile) {
  if (profile != kVrpoScheduleProfile) return nullptr;
  static const std::array<VrpoScheduleCellSpec, 4> cells = {{
      {"U1_LR1E5_E1", 1.0e-5, 1},
      {"U2_LR7P5E6_E1", 7.5e-6, 1},
      {"U3_LR5E6_E1", 5.0e-6, 1},
      {"U4_LR5E6_E2", 5.0e-6, 2},
  }};
  return &cells;
}

struct VrpoScheduleStartupConfig {
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

inline std::vector<std::string> VrpoScheduleSourceRelativePaths() {
  std::vector<std::string> paths = VrpoPhase4eSourceRelativePaths();
  paths.push_back("open_spiel/examples/dune_vrpo_schedule_screen.h");
  return paths;
}

inline bool LoadVrpoScheduleSourceIdentity(
    const std::filesystem::path& source_root,
    const std::string& registered_sha256, VrpoPhase4eSourceIdentity* output,
    std::string* error) {
  auto fail = [&](const std::string& message) {
    if (error != nullptr) *error = message;
    if (output != nullptr) *output = VrpoPhase4eSourceIdentity{};
    return false;
  };
  if (output == nullptr || !VrpoPhase4eLowerHex64(registered_sha256)) {
    return fail("schedule source identity/registration is invalid");
  }
  std::error_code ec;
  const auto canonical_root = std::filesystem::canonical(source_root, ec);
  if (ec || !std::filesystem::is_directory(canonical_root)) {
    return fail("schedule source root is not a readable directory");
  }
  const auto paths = VrpoScheduleSourceRelativePaths();
  std::set<std::string> seen;
  std::string payload;
  for (const auto& relative : paths) {
    if (relative.empty() || std::filesystem::path(relative).is_absolute() ||
        relative.find("..") != std::string::npos ||
        !seen.insert(relative).second) {
      return fail("schedule source list is noncanonical or duplicated");
    }
    const auto path = canonical_root / relative;
    if (!std::filesystem::is_regular_file(path) ||
        std::filesystem::is_symlink(path)) {
      return fail("schedule source file is missing/nonregular: " + relative);
    }
    size_t size = 0;
    const std::string digest = ComputeFileSHA256(path.string(), &size);
    if (!VrpoPhase4eLowerHex64(digest) || size == 0) {
      return fail("schedule source file could not be hashed: " + relative);
    }
    payload.append(relative);
    payload.push_back('\0');
    payload.append(digest);
    payload.push_back('\n');
  }
  const std::string observed = ComputeStringSHA256(payload);
  if (observed != registered_sha256) {
    return fail("schedule registered source SHA-256 mismatch");
  }
  output->canonical_root = canonical_root;
  output->relative_paths = paths;
  output->combined_sha256 = observed;
  return true;
}

inline bool ValidateVrpoScheduleQProvenanceInstrumentation(
    const std::string& q_role, std::string* error) {
  if (q_role != kVrpoScheduleQRole || VrpoQConstructorCalls() != 1 ||
      VrpoQForwardCheckedCalls() != 0 ||
      VrpoScheduleQTargetComputations() != 0) {
    if (error != nullptr) {
      *error = "schedule Q provenance requires role label, exactly one "
               "constructor, zero forwards, and zero Q-target computations";
    }
    return false;
  }
  return true;
}

inline bool ValidateVrpoScheduleStartupConfig(
    const VrpoScheduleStartupConfig& config, std::string* error) {
  auto fail = [&](const std::string& message) {
    if (error != nullptr) *error = message;
    return false;
  };
  if (config.game != "dune_imperium" ||
      config.init_mode != "vrpo_schedule_screen" ||
      config.profile != kVrpoScheduleProfile ||
      config.registration_id.empty() || config.input_archive.empty() ||
      config.output_root.empty() || config.source_root.empty() ||
      !VrpoPhase4eLowerHex64(config.source_code_sha256) ||
      !VrpoPhase4eLowerHex64(config.executed_binary_sha256) ||
      config.executed_binary_size <= 0) {
    return fail("schedule-screen identity/mode contract is invalid");
  }
  VrpoPhase4eResolvedPaths resolved;
  if (!ResolveVrpoPhase4ePaths(config.input_archive, config.output_root,
                               &resolved, error)) {
    return false;
  }
  if (std::filesystem::exists(resolved.output_root)) {
    return fail("schedule-screen output root must be fresh and absent");
  }
  if (config.rollout_games != kVrpoScheduleGames || config.threads != 8 ||
      config.eval_batch_size != 64 || config.eval_timeout_ms != 2 ||
      !config.evaluator_device_synchronize ||
      config.deterministic_rollout_eval || config.seed_scheme_version != 2 ||
      !config.runtime_device_is_cuda || config.runtime_device_index < 0 ||
      !config.one_gpu_process || config.runtime_process_id <= 0 ||
      config.base_seed != 8302001 ||
      config.start_episode_id != 1200000000 ||
      config.diagnostics_only || config.rollout_amp || config.train_amp ||
      !config.allow_tf32 || config.pipeline ||
      config.online_search_collection || config.search_pi_mode ||
      config.train_value_only || config.sample_counterfactual_states ||
      config.has_search_label_dir ||
      config.ordinary_checkpoint_writes_enabled) {
    return fail("schedule-screen execution/side-path contract is invalid");
  }
  for (double shaping :
       {config.shaped_reward_weight, config.tleilaxu_breadcrumb_weight,
        config.tleilaxu_level7_breadcrumb_weight,
        config.specimen_exchange_penalty}) {
    if (!std::isfinite(shaping) || shaping != 0.0) {
      return fail("schedule-screen shaping must be exactly zero");
    }
  }
  if (config.reward_scale != 4.0 || config.gamma != 1.0 ||
      config.lambda != 1.0 || config.logit_cap != 10.0 ||
      config.ppo_minibatches != kVrpoTrainingMinibatchesPerEpoch ||
      config.ppo_minibatch_size != 2048 ||
      config.clip_epsilon != kVrpoTrainingClipEpsilon ||
      config.entropy_coefficient != kVrpoTrainingEntropyCoefficient ||
      config.value_coefficient != kVrpoTrainingValueCoefficient ||
      config.gradient_clip_norm != kVrpoTrainingGradientClipNorm ||
      !config.normalize_advantages || !config.clip_value_loss) {
    return fail("schedule-screen PPO mechanics are not the registered values");
  }
  const auto* cells = FindVrpoScheduleCellsForProfile(config.profile);
  if (cells == nullptr) {
    return fail("schedule-screen profile has no canonical cell table");
  }
  int64_t steps = 0;
  std::set<std::string> ids;
  for (const auto& cell : *cells) {
    if (!ids.insert(cell.cell_id).second ||
        !std::isfinite(cell.learning_rate) || cell.learning_rate <= 0.0 ||
        (cell.actor_epochs != 1 && cell.actor_epochs != 2)) {
      return fail("schedule-screen cell table is malformed");
    }
    steps += cell.actor_epochs * kVrpoTrainingMinibatchesPerEpoch;
  }
  if (steps != kVrpoScheduleMaxActorSteps) {
    return fail("schedule-screen actor-step ceiling is not exact");
  }
  return true;
}

namespace vrpo_schedule_internal {

template <typename T>
inline void Append(std::string* output, const T& value) {
  output->append(reinterpret_cast<const char*>(&value), sizeof(T));
}

inline void AppendString(std::string* output, const std::string& value) {
  const uint64_t size = value.size();
  Append(output, size);
  output->append(value);
}

template <typename T>
inline bool Read(const std::string& input, size_t* offset, T* value) {
  if (offset == nullptr || value == nullptr ||
      *offset > input.size() || input.size() - *offset < sizeof(T)) {
    return false;
  }
  std::memcpy(value, input.data() + *offset, sizeof(T));
  *offset += sizeof(T);
  return true;
}

inline bool ReadString(const std::string& input, size_t* offset,
                       std::string* value) {
  uint64_t size = 0;
  if (!Read(input, offset, &size) || size > input.size() - *offset) {
    return false;
  }
  *value = input.substr(*offset, size);
  *offset += size;
  return true;
}

inline std::string ActorRowPayload(const VrpoTrainingRow& row) {
  std::string out;
  Append(&out, row.row_id);
  Append(&out, row.episode_id);
  Append(&out, row.step_index);
  const int32_t actor = row.actor;
  Append(&out, actor);
  torch::Tensor actor_input =
      row.actor_input.detach().contiguous().cpu().to(torch::kFloat32);
  const uint64_t observation_size = actor_input.numel();
  Append(&out, observation_size);
  out.append(reinterpret_cast<const char*>(actor_input.data_ptr<float>()),
             observation_size * sizeof(float));
  const uint64_t legal_size = row.legal_actions.size();
  Append(&out, legal_size);
  for (Action action : row.legal_actions) Append(&out, action);
  Append(&out, row.chosen_index);
  Append(&out, row.chosen_action);
  for (double probability : row.old_legal_probabilities) {
    Append(&out, probability);
  }
  Append(&out, row.old_chosen_log_probability);
  Append(&out, row.ppo_advantage);
  Append(&out, row.ppo_return);
  Append(&out, row.ppo_old_value);
  for (double reward : row.rewards) Append(&out, reward);
  const uint8_t terminal = row.terminal_after ? 1 : 0;
  Append(&out, terminal);
  return out;
}

inline bool ParseActorRow(const std::string& input, size_t* offset,
                          VrpoTrainingRow* row, std::string* error) {
  auto fail = [&](const std::string& message) {
    if (error != nullptr) *error = message;
    return false;
  };
  int32_t actor = -1;
  uint64_t observation_size = 0;
  if (!Read(input, offset, &row->row_id) ||
      !Read(input, offset, &row->episode_id) ||
      !Read(input, offset, &row->step_index) ||
      !Read(input, offset, &actor) ||
      !Read(input, offset, &observation_size) || observation_size == 0 ||
      observation_size > 100000 ||
      observation_size > (input.size() - *offset) / sizeof(float)) {
    return fail("actor corpus row prefix is malformed");
  }
  row->actor = actor;
  row->actor_input = torch::from_blob(
      const_cast<char*>(input.data() + *offset),
      {static_cast<int64_t>(observation_size)}, torch::kFloat32).clone();
  *offset += observation_size * sizeof(float);
  uint64_t legal_size = 0;
  if (!Read(input, offset, &legal_size) || legal_size == 0 ||
      legal_size > kVrpoDuneActionDim) {
    return fail("actor corpus legal width is malformed");
  }
  row->legal_actions.resize(legal_size);
  row->old_legal_probabilities.resize(legal_size);
  for (Action& action : row->legal_actions) {
    if (!Read(input, offset, &action)) return fail("truncated legal actions");
  }
  if (!Read(input, offset, &row->chosen_index) ||
      !Read(input, offset, &row->chosen_action)) {
    return fail("truncated chosen action");
  }
  for (double& probability : row->old_legal_probabilities) {
    if (!Read(input, offset, &probability)) {
      return fail("truncated behavior policy");
    }
  }
  if (!Read(input, offset, &row->old_chosen_log_probability) ||
      !Read(input, offset, &row->ppo_advantage) ||
      !Read(input, offset, &row->ppo_return) ||
      !Read(input, offset, &row->ppo_old_value)) {
    return fail("truncated PPO row fields");
  }
  for (double& reward : row->rewards) {
    if (!Read(input, offset, &reward)) return fail("truncated row rewards");
  }
  uint8_t terminal = 0;
  if (!Read(input, offset, &terminal) || terminal > 1) {
    return fail("invalid terminal marker");
  }
  row->terminal_after = terminal != 0;
  // The deliberately omitted central/Q tensor is represented by an explicit
  // undefined tensor after decode; a decoded corpus cannot accidentally be
  // passed to a VRPO/Q update.
  row->q_input = torch::Tensor();
  return true;
}

inline int64_t DirectoryRegularBytes(const std::filesystem::path& root) {
  int64_t total = 0;
  std::error_code ec;
  for (std::filesystem::recursive_directory_iterator it(root, ec), end;
       !ec && it != end; it.increment(ec)) {
    if (it->is_regular_file(ec) && !it->is_symlink(ec)) {
      const auto size = it->file_size(ec);
      if (!ec && size <= static_cast<uintmax_t>(
                            std::numeric_limits<int64_t>::max() - total)) {
        total += static_cast<int64_t>(size);
      }
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
    if (error != nullptr) *error = "cannot resolve output filesystem";
    return false;
  }
  const auto space = std::filesystem::space(probe, ec);
  if (ec || space.available <
                static_cast<uintmax_t>(kVrpoScheduleTemporaryByteCeiling)) {
    if (error != nullptr) {
      *error = "insufficient free space for schedule-screen temporary ceiling";
    }
    return false;
  }
  return true;
}

inline bool ObserveRootBytes(const std::filesystem::path& root,
                             int64_t observed_override,
                             int64_t* measured_peak_bytes,
                             std::string* error) {
  if (measured_peak_bytes == nullptr) {
    if (error != nullptr) *error = "null schedule byte-budget peak";
    return false;
  }
  const int64_t observed = observed_override >= 0
      ? observed_override
      : DirectoryRegularBytes(root);
  if (observed < 0) {
    if (error != nullptr) *error = "schedule root byte accounting failed";
    return false;
  }
  *measured_peak_bytes = std::max(*measured_peak_bytes, observed);
  if (observed > kVrpoScheduleTemporaryByteCeiling) {
    if (error != nullptr) {
      *error = "schedule-screen measured temporary bytes exceed 2.25 GiB";
    }
    return false;
  }
  return true;
}

inline void CleanupFreshRoot(const std::filesystem::path& root) {
  std::error_code ec;
  if (!root.empty()) std::filesystem::remove_all(root, ec);
}

}  // namespace vrpo_schedule_internal

struct VrpoScheduleCorpusIdentity {
  int64_t episodes = 0;
  int64_t rows = 0;
  std::vector<std::string> episode_sha256;
  std::string payload_sha256;
  int64_t byte_size = 0;
};

inline bool ValidateVrpoScheduleActorEpisodes(
    const std::vector<VrpoTrainingEpisode>& episodes, std::string* error);

inline bool EncodeVrpoScheduleActorCorpus(
    const std::vector<VrpoTrainingEpisode>& episodes, std::string* output,
    VrpoScheduleCorpusIdentity* identity, std::string* error) {
  auto fail = [&](const std::string& message) {
    if (error != nullptr) *error = message;
    if (output != nullptr) output->clear();
    if (identity != nullptr) *identity = VrpoScheduleCorpusIdentity{};
    return false;
  };
  if (output == nullptr || identity == nullptr ||
      episodes.size() != kVrpoScheduleGames) {
    return fail("actor corpus requires exactly 16 episodes");
  }
  std::string out;
  vrpo_schedule_internal::AppendString(&out, kVrpoScheduleCorpusSchema);
  const uint64_t episode_count = episodes.size();
  vrpo_schedule_internal::Append(&out, episode_count);
  uint64_t previous_episode_id = 0;
  uint64_t previous_row_id = 0;
  for (const auto& episode : episodes) {
    if (episode.episode_id <= previous_episode_id || episode.rows.empty()) {
      return fail("actor corpus episode order/content is invalid");
    }
    previous_episode_id = episode.episode_id;
    vrpo_schedule_internal::Append(&out, episode.episode_id);
    const uint64_t row_count = episode.rows.size();
    vrpo_schedule_internal::Append(&out, row_count);
    std::string episode_payload;
    for (const auto& row : episode.rows) {
      if (row.episode_id != episode.episode_id ||
          row.row_id <= previous_row_id || !row.actor_input.defined() ||
          !vrpo_training_internal::FiniteTensor(row.actor_input)) {
        return fail("actor corpus row identity/input is invalid");
      }
      previous_row_id = row.row_id;
      episode_payload.append(vrpo_schedule_internal::ActorRowPayload(row));
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
  identity->byte_size = out.size();
  *output = std::move(out);
  return true;
}

inline bool DecodeVrpoScheduleActorCorpus(
    const std::string& input, std::vector<VrpoTrainingEpisode>* episodes,
    VrpoScheduleCorpusIdentity* identity, std::string* error) {
  auto fail = [&](const std::string& message) {
    if (error != nullptr) *error = message;
    if (episodes != nullptr) episodes->clear();
    if (identity != nullptr) *identity = VrpoScheduleCorpusIdentity{};
    return false;
  };
  if (episodes == nullptr || identity == nullptr || input.empty()) {
    return fail("actor corpus decode output/input is invalid");
  }
  episodes->clear();
  *identity = VrpoScheduleCorpusIdentity{};
  size_t offset = 0;
  std::string schema;
  uint64_t episode_count = 0;
  if (!vrpo_schedule_internal::ReadString(input, &offset, &schema) ||
      schema != kVrpoScheduleCorpusSchema ||
      !vrpo_schedule_internal::Read(input, &offset, &episode_count) ||
      episode_count != kVrpoScheduleGames) {
    return fail("actor corpus header is invalid");
  }
  uint64_t previous_episode_id = 0;
  uint64_t previous_row_id = 0;
  for (uint64_t e = 0; e < episode_count; ++e) {
    VrpoTrainingEpisode episode;
    uint64_t row_count = 0;
    std::string expected_digest;
    if (!vrpo_schedule_internal::Read(input, &offset, &episode.episode_id) ||
        episode.episode_id <= previous_episode_id ||
        !vrpo_schedule_internal::Read(input, &offset, &row_count) ||
        row_count == 0 || row_count > 1000000 ||
        !vrpo_schedule_internal::ReadString(input, &offset,
                                             &expected_digest) ||
        !VrpoPhase4eLowerHex64(expected_digest)) {
      return fail("actor corpus episode header/order is invalid");
    }
    previous_episode_id = episode.episode_id;
    const size_t payload_start = offset;
    for (uint64_t r = 0; r < row_count; ++r) {
      VrpoTrainingRow row;
      if (!vrpo_schedule_internal::ParseActorRow(input, &offset, &row,
                                                  error) ||
          row.episode_id != episode.episode_id ||
          row.row_id <= previous_row_id) {
        return fail(error != nullptr && !error->empty()
                        ? *error
                        : "actor corpus row/order is invalid");
      }
      previous_row_id = row.row_id;
      episode.rows.push_back(std::move(row));
      ++identity->rows;
    }
    const std::string observed_digest = ComputeStringSHA256(
        input.substr(payload_start, offset - payload_start));
    if (observed_digest != expected_digest) {
      return fail("actor corpus episode digest mismatch");
    }
    identity->episode_sha256.push_back(observed_digest);
    episodes->push_back(std::move(episode));
  }
  const size_t payload_end = offset;
  std::string expected_payload_digest;
  if (!vrpo_schedule_internal::ReadString(input, &offset,
                                           &expected_payload_digest) ||
      offset != input.size() ||
      ComputeStringSHA256(input.substr(0, payload_end)) !=
          expected_payload_digest) {
    return fail("actor corpus exhaustive digest/trailing-byte check failed");
  }
  identity->episodes = episodes->size();
  identity->payload_sha256 = expected_payload_digest;
  identity->byte_size = input.size();
  if (!ValidateVrpoScheduleActorEpisodes(*episodes, error)) {
    return fail(error != nullptr ? *error
                                 : "decoded actor corpus is invalid");
  }
  return true;
}

inline bool ValidateVrpoScheduleActorEpisodes(
    const std::vector<VrpoTrainingEpisode>& episodes, std::string* error) {
  auto fail = [&](const std::string& message) {
    if (error != nullptr) *error = message;
    return false;
  };
  if (episodes.size() != kVrpoScheduleGames) {
    return fail("schedule corpus must contain exactly 16 complete episodes");
  }
  uint64_t previous_episode = 0;
  uint64_t previous_row = 0;
  int64_t observation_width = -1;
  for (const auto& episode : episodes) {
    if (episode.episode_id <= previous_episode || episode.rows.empty()) {
      return fail("schedule corpus episode order/content is invalid");
    }
    previous_episode = episode.episode_id;
    for (size_t row_index = 0; row_index < episode.rows.size(); ++row_index) {
      const auto& row = episode.rows[row_index];
      if (row.row_id <= previous_row || row.episode_id != episode.episode_id ||
          row.step_index != static_cast<int64_t>(row_index) ||
          row.actor < 0 || row.actor >= kVrpoNumSeats ||
          !row.actor_input.defined() || row.actor_input.dim() != 1 ||
          row.actor_input.scalar_type() != torch::kFloat32 ||
          !vrpo_training_internal::FiniteTensor(row.actor_input) ||
          row.q_input.defined()) {
        return fail("schedule actor row identity/input/Q-omission is invalid");
      }
      previous_row = row.row_id;
      if (observation_width < 0) observation_width = row.actor_input.size(0);
      if (row.actor_input.size(0) != observation_width ||
          row.legal_actions.empty() ||
          row.old_legal_probabilities.size() != row.legal_actions.size() ||
          row.chosen_index < 0 ||
          row.chosen_index >= static_cast<int>(row.legal_actions.size()) ||
          row.legal_actions[row.chosen_index] != row.chosen_action ||
          row.terminal_after != (row_index + 1 == episode.rows.size())) {
        return fail("schedule actor row shape/legal/terminal data is invalid");
      }
      std::set<Action> legal;
      double probability_sum = 0.0;
      for (size_t action = 0; action < row.legal_actions.size(); ++action) {
        const Action action_id = row.legal_actions[action];
        const double probability = row.old_legal_probabilities[action];
        if (action_id < 0 || action_id >= kVrpoDuneActionDim ||
            !legal.insert(action_id).second || !std::isfinite(probability) ||
            probability < 0.0 || probability > 1.0) {
          return fail("schedule actor legal behavior policy is invalid");
        }
        probability_sum += probability;
      }
      const double chosen_probability =
          row.old_legal_probabilities[row.chosen_index];
      if (std::abs(probability_sum - 1.0) > 1e-6 ||
          !(chosen_probability > 0.0) ||
          !vrpo_training_internal::Float32DomainFinite(
              row.old_chosen_log_probability) ||
          !vrpo_training_internal::Float32DomainFinite(row.ppo_advantage) ||
          !vrpo_training_internal::Float32DomainFinite(row.ppo_return) ||
          !vrpo_training_internal::Float32DomainFinite(row.ppo_old_value) ||
          std::abs(std::log(chosen_probability) -
                   row.old_chosen_log_probability) > 1e-6) {
        return fail("schedule PPO learner-consumed scalar is invalid");
      }
      for (double reward : row.rewards) {
        if (!std::isfinite(reward)) {
          return fail("schedule actor row reward is nonfinite");
        }
      }
    }
  }
  return observation_width > 0;
}

inline bool BuildVrpoScheduleEpisodePartitionPlan(
    const std::vector<VrpoTrainingEpisode>& episodes, uint64_t epoch_seed,
    VrpoEpisodePartitionPlan* output, std::string* error) {
  auto fail = [&](const std::string& message) {
    if (error != nullptr) *error = message;
    if (output != nullptr) *output = VrpoEpisodePartitionPlan{};
    return false;
  };
  if (output == nullptr || epoch_seed == 0 ||
      !ValidateVrpoScheduleActorEpisodes(episodes, error)) {
    return fail(error != nullptr && !error->empty()
                    ? *error
                    : "schedule partition input is invalid");
  }
  std::vector<size_t> order(episodes.size());
  for (size_t i = 0; i < order.size(); ++i) order[i] = i;
  std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) {
    if (episodes[a].rows.size() != episodes[b].rows.size()) {
      return episodes[a].rows.size() > episodes[b].rows.size();
    }
    const uint64_t ka = vrpo_training_internal::SplitMix64(
        epoch_seed ^ episodes[a].episode_id);
    const uint64_t kb = vrpo_training_internal::SplitMix64(
        epoch_seed ^ episodes[b].episode_id);
    return ka != kb ? ka < kb : episodes[a].episode_id < episodes[b].episode_id;
  });
  VrpoEpisodePartitionPlan plan;
  plan.epoch_seed = epoch_seed;
  for (size_t episode_index : order) {
    size_t destination = 0;
    for (size_t bin = 1; bin < plan.minibatches.size(); ++bin) {
      const auto& candidate = plan.minibatches[bin];
      const auto& current = plan.minibatches[destination];
      if (candidate.row_count < current.row_count ||
          (candidate.row_count == current.row_count &&
           vrpo_training_internal::SplitMix64(epoch_seed ^ bin) <
               vrpo_training_internal::SplitMix64(epoch_seed ^ destination))) {
        destination = bin;
      }
    }
    plan.minibatches[destination].episode_indices.push_back(episode_index);
    plan.minibatches[destination].row_count += episodes[episode_index].rows.size();
  }
  std::set<size_t> covered;
  std::string payload = "dune_vrpo_phase4d_whole_episode_partition_v1";
  vrpo_training_internal::AppendPod(&payload, epoch_seed);
  for (auto& minibatch : plan.minibatches) {
    if (minibatch.episode_indices.empty() || minibatch.row_count <= 0) {
      return fail("schedule partition produced an empty minibatch");
    }
    std::stable_sort(
        minibatch.episode_indices.begin(), minibatch.episode_indices.end(),
        [&](size_t a, size_t b) {
          const uint64_t ka = vrpo_training_internal::SplitMix64(
              epoch_seed + episodes[a].episode_id);
          const uint64_t kb = vrpo_training_internal::SplitMix64(
              epoch_seed + episodes[b].episode_id);
          return ka != kb ? ka < kb
                          : episodes[a].episode_id < episodes[b].episode_id;
        });
    vrpo_training_internal::AppendPod(&payload, minibatch.row_count);
    const uint64_t count = minibatch.episode_indices.size();
    vrpo_training_internal::AppendPod(&payload, count);
    for (size_t episode_index : minibatch.episode_indices) {
      if (!covered.insert(episode_index).second) {
        return fail("schedule partition repeats an episode");
      }
      vrpo_training_internal::AppendPod(
          &payload, episodes[episode_index].episode_id);
      const uint64_t rows = episodes[episode_index].rows.size();
      vrpo_training_internal::AppendPod(&payload, rows);
      for (const auto& row : episodes[episode_index].rows) {
        vrpo_training_internal::AppendPod(&payload, row.row_id);
      }
    }
  }
  if (covered.size() != episodes.size()) {
    return fail("schedule partition omitted an episode");
  }
  plan.canonical_sha256 = ComputeStringSHA256(payload);
  *output = std::move(plan);
  return true;
}

inline bool MakeVrpoScheduleFreshActorOptimizer(
    SharedDunePolicyValueNetImpl& actor_model, double learning_rate,
    std::unique_ptr<torch::optim::AdamW>* output, std::string* error) {
  auto fail = [&](const std::string& message) {
    if (error != nullptr) *error = message;
    if (output != nullptr) output->reset();
    return false;
  };
  if (output == nullptr || !std::isfinite(learning_rate) ||
      learning_rate <= 0.0) {
    return fail("schedule actor optimizer arguments are invalid");
  }
  std::vector<torch::Tensor> policy;
  std::vector<torch::Tensor> trunk_value;
  std::set<c10::TensorImpl*> covered;
  for (const auto& item : actor_model.named_parameters()) {
    if (!covered.insert(item.value().unsafeGetTensorImpl()).second) {
      return fail("schedule actor optimizer repeats a parameter");
    }
    if (item.key().rfind("policy_head", 0) == 0) {
      policy.push_back(item.value());
    } else {
      trunk_value.push_back(item.value());
    }
  }
  if (policy.empty() || trunk_value.empty() ||
      covered.size() != actor_model.named_parameters().size()) {
    return fail("schedule actor optimizer coverage is incomplete");
  }
  const auto specs = CanonicalVrpoPhase4OptimizerGroups();
  std::vector<torch::optim::OptimizerParamGroup> groups;
  groups.emplace_back(policy);
  groups.emplace_back(trunk_value);
  auto optimizer = std::make_unique<torch::optim::AdamW>(
      groups, torch::optim::AdamWOptions(learning_rate)
                  .betas({specs[0].beta1, specs[0].beta2})
                  .eps(specs[0].epsilon));
  for (size_t group = 0; group < groups.size(); ++group) {
    auto& options = static_cast<torch::optim::AdamWOptions&>(
        optimizer->param_groups()[group].options());
    options.lr(learning_rate);
    options.betas({specs[group].beta1, specs[group].beta2});
    options.eps(specs[group].epsilon);
    options.weight_decay(0.0);
  }
  MaterializeVrpoZeroAdamWState(*optimizer);
  if (!vrpo_training_internal::OptimizerStateIsFresh(*optimizer, error)) {
    return fail(error != nullptr ? *error
                                 : "schedule actor optimizer is not fresh");
  }
  *output = std::move(optimizer);
  return true;
}

inline bool ValidateVrpoScheduleActorOptimizerCoverage(
    torch::optim::Optimizer& optimizer, torch::nn::Module& actor,
    std::string* error) {
  std::set<c10::TensorImpl*> expected;
  for (const auto& item : actor.named_parameters()) {
    expected.insert(item.value().unsafeGetTensorImpl());
  }
  std::set<c10::TensorImpl*> actual;
  for (const auto& group : optimizer.param_groups()) {
    for (const auto& parameter : group.params()) {
      if (!actual.insert(parameter.unsafeGetTensorImpl()).second) {
        if (error != nullptr) *error = "schedule actor optimizer duplicates a parameter";
        return false;
      }
    }
  }
  if (actual != expected) {
    if (error != nullptr) *error = "schedule actor optimizer coverage is not exact";
    return false;
  }
  return true;
}

class VrpoScheduleActorTransaction {
 public:
  bool Capture(torch::nn::Module& actor, torch::optim::AdamW& optimizer,
               std::string* error) {
    actor_ = &actor;
    optimizer_ = &optimizer;
    training_ = actor.is_training();
    try {
      for (const auto& item : actor.named_parameters()) {
        parameters_.push_back(item.value().detach().clone());
        requires_grad_.push_back(item.value().requires_grad());
      }
      for (const auto& item : actor.named_buffers()) {
        buffers_.push_back(item.value().detach().clone());
      }
      for (const auto& group : optimizer.param_groups()) {
        for (const auto& parameter : group.params()) {
          const auto found = optimizer.state().find(
              parameter.unsafeGetTensorImpl());
          const auto* state = found == optimizer.state().end()
              ? nullptr
              : dynamic_cast<const torch::optim::AdamWParamState*>(
                    found->second.get());
          if (state == nullptr || !state->exp_avg().defined() ||
              !state->exp_avg_sq().defined()) {
            if (error != nullptr) *error = "schedule actor transaction state missing";
            return false;
          }
          optimizer_steps_.push_back(state->step());
          exp_avg_.push_back(state->exp_avg().detach().clone());
          exp_avg_sq_.push_back(state->exp_avg_sq().detach().clone());
          max_exp_avg_sq_.push_back(state->max_exp_avg_sq().defined()
                                        ? state->max_exp_avg_sq().detach().clone()
                                        : torch::Tensor());
        }
      }
    } catch (const std::exception& exception) {
      if (error != nullptr) {
        *error = std::string("schedule actor transaction capture failed: ") +
                 exception.what();
      }
      return false;
    }
    active_ = true;
    return true;
  }

  void Commit() { active_ = false; }

  ~VrpoScheduleActorTransaction() {
    if (!active_ || actor_ == nullptr || optimizer_ == nullptr) return;
    try {
      torch::NoGradGuard no_grad;
      size_t index = 0;
      for (auto& item : actor_->named_parameters()) {
        item.value().copy_(parameters_.at(index));
        item.value().set_requires_grad(requires_grad_.at(index));
        ++index;
      }
      index = 0;
      for (auto& item : actor_->named_buffers()) {
        item.value().copy_(buffers_.at(index++));
      }
      index = 0;
      for (const auto& group : optimizer_->param_groups()) {
        for (const auto& parameter : group.params()) {
          auto found = optimizer_->state().find(parameter.unsafeGetTensorImpl());
          auto* state = dynamic_cast<torch::optim::AdamWParamState*>(
              found->second.get());
          state->step(optimizer_steps_.at(index));
          state->exp_avg().copy_(exp_avg_.at(index));
          state->exp_avg_sq().copy_(exp_avg_sq_.at(index));
          if (max_exp_avg_sq_.at(index).defined()) {
            state->max_exp_avg_sq().copy_(max_exp_avg_sq_.at(index));
          }
          ++index;
        }
      }
      optimizer_->zero_grad();
      actor_->train(training_);
    } catch (...) {
      // Destructors cannot report. All capture shapes/order are validated, so
      // this path is defensive only.
    }
  }

 private:
  bool active_ = false;
  bool training_ = false;
  torch::nn::Module* actor_ = nullptr;
  torch::optim::AdamW* optimizer_ = nullptr;
  std::vector<torch::Tensor> parameters_;
  std::vector<bool> requires_grad_;
  std::vector<torch::Tensor> buffers_;
  std::vector<int64_t> optimizer_steps_;
  std::vector<torch::Tensor> exp_avg_;
  std::vector<torch::Tensor> exp_avg_sq_;
  std::vector<torch::Tensor> max_exp_avg_sq_;
};

struct VrpoScheduleActorOnlyUpdateStats {
  VrpoTrainingUpdateStats ppo;
  int64_t q_target_computations = -1;
};

inline bool RunVrpoScheduleActorOnlyPpoCell(
    const std::vector<VrpoTrainingEpisode>& episodes, uint64_t update_seed,
    int actor_epochs, torch::nn::Module& actor_model,
    torch::optim::AdamW& actor_optimizer,
    const VrpoActorForward& actor_forward,
    bool four_epoch_equivalence_test,
    const VrpoScheduleRunDeadline* deadline,
    VrpoScheduleActorOnlyUpdateStats* output, std::string* error,
    bool require_fresh_optimizer = true) {
  auto fail = [&](const std::string& message) {
    if (error != nullptr) *error = message;
    if (output != nullptr) *output = VrpoScheduleActorOnlyUpdateStats{};
    return false;
  };
  if (output == nullptr || !actor_forward || update_seed == 0 ||
      VrpoScheduleQTargetComputations() != 0 ||
      (actor_epochs != 1 && actor_epochs != 2 &&
       !(actor_epochs == 4 && four_epoch_equivalence_test)) ||
      !ValidateVrpoScheduleActorEpisodes(episodes, error) ||
      !ValidateVrpoScheduleActorOptimizerCoverage(
          actor_optimizer, actor_model, error) ||
      (require_fresh_optimizer &&
       !vrpo_training_internal::OptimizerStateIsFresh(actor_optimizer,
                                                      error))) {
    return fail(error != nullptr && !error->empty()
                    ? *error
                    : "schedule actor-only PPO input is invalid");
  }
  vrpo_training_internal::ModuleRuntimePlacement placement;
  if (!vrpo_training_internal::CaptureUniformModuleRuntimePlacement(
          actor_model, &placement, error) || placement.dtype != torch::kFloat32) {
    return fail(error != nullptr && !error->empty()
                    ? *error
                    : "schedule actor placement must be Float32");
  }
  std::array<VrpoEpisodePartitionPlan, 4> plans;
  for (int epoch = 0; epoch < 4; ++epoch) {
    if (!BuildVrpoScheduleEpisodePartitionPlan(
            episodes,
            vrpo_training_internal::SplitMix64(update_seed + epoch + 1),
            &plans[epoch], error)) {
      return false;
    }
  }
  if (!vrpo_training_internal::ValidatePlannedActorMinibatches(
          episodes, plans, error)) {
    return false;
  }
  if (deadline != nullptr && !deadline->Check("actor preflight", error)) {
    return false;
  }
  // Actor-only shape/finite preflight. No Q tensor, Q module, Q forward, or Q
  // optimizer exists on this path.
  {
    torch::NoGradGuard no_grad;
    for (const auto& minibatch : plans[0].minibatches) {
      const auto flat = vrpo_training_internal::FlattenEpisodes(
          episodes, minibatch.episode_indices);
      torch::Tensor inputs;
      if (!vrpo_training_internal::StackInputs(
              flat.rows, true, placement, &inputs, error)) {
        return false;
      }
      VrpoActorTrainingOutput evaluated;
      try {
        evaluated = actor_forward(inputs);
      } catch (const std::exception& exception) {
        return fail(std::string("schedule actor preflight failed: ") +
                    exception.what());
      }
      if (!vrpo_training_internal::ValidateActorOutput(
              evaluated, flat.rows.size(), placement, error)) {
        return false;
      }
      if (deadline != nullptr &&
          !deadline->Check("actor preflight minibatch", error)) {
        return false;
      }
    }
  }

  VrpoTrainingUpdateStats stats;
  stats.complete_episode_partitions = true;
  stats.current_rollout_only = true;
  stats.advantages_detached = true;
  stats.q_frozen_during_actor = true;
  if (!vrpo_training_internal::ModuleValueSha256(
          actor_model, "", &stats.actor_values_before_sha256, error) ||
      !vrpo_training_internal::ModuleValueSha256(
          actor_model, "value_head", &stats.value_head_before_sha256, error)) {
    return false;
  }
  VrpoScheduleActorTransaction transaction;
  if (!transaction.Capture(actor_model, actor_optimizer, error)) return false;
  double actor_loss_sum = 0.0;
  int64_t actor_loss_count = 0;
  for (int epoch = 0; epoch < actor_epochs; ++epoch) {
    stats.actor_epoch_partition_sha256[epoch] = plans[epoch].canonical_sha256;
    for (const auto& minibatch : plans[epoch].minibatches) {
      if (deadline != nullptr &&
          !deadline->Check("before actor minibatch", error)) {
        return false;
      }
      const auto flat = vrpo_training_internal::FlattenEpisodes(
          episodes, minibatch.episode_indices);
      actor_optimizer.zero_grad();
      torch::Tensor inputs;
      if (!vrpo_training_internal::StackInputs(
              flat.rows, true, placement, &inputs, error)) {
        return false;
      }
      VrpoActorTrainingOutput actor_output;
      try {
        actor_output = actor_forward(inputs);
      } catch (const std::exception& exception) {
        return fail(std::string("schedule actor forward failed: ") +
                    exception.what());
      }
      if (!vrpo_training_internal::ValidateActorOutput(
              actor_output, flat.rows.size(), placement, error)) {
        return false;
      }
      std::vector<torch::Tensor> legal_log_probabilities;
      std::vector<torch::Tensor> legal_probabilities;
      std::vector<std::vector<double>> detached_probabilities;
      if (!vrpo_training_internal::BuildLegalPolicies(
              actor_output, flat, 10.0, &legal_log_probabilities,
              &legal_probabilities, &detached_probabilities, error)) {
        return false;
      }
      std::vector<double> advantages(flat.rows.size());
      for (size_t index = 0; index < flat.rows.size(); ++index) {
        advantages[index] = flat.rows[index]->ppo_advantage;
      }
      std::vector<double> loss_advantages = advantages;
      std::vector<double> nontrivial_advantages;
      for (size_t index = 0; index < flat.rows.size(); ++index) {
        if (flat.rows[index]->legal_actions.size() > 1) {
          nontrivial_advantages.push_back(advantages[index]);
        }
      }
      if (nontrivial_advantages.size() > 1) {
        double mean = 0.0;
        for (double value : nontrivial_advantages) mean += value;
        mean /= nontrivial_advantages.size();
        double variance = 0.0;
        for (double value : nontrivial_advantages) {
          variance += (value - mean) * (value - mean);
        }
        variance /= nontrivial_advantages.size();
        const double scale = std::sqrt(variance) + 1e-8;
        for (double& value : loss_advantages) value = (value - mean) / scale;
      }
      std::vector<torch::Tensor> policy_terms;
      std::vector<torch::Tensor> entropy_terms;
      std::vector<torch::Tensor> value_losses;
      std::vector<torch::Tensor> ratio_diagnostics;
      std::vector<torch::Tensor> kl_diagnostics;
      std::vector<float> old_probability_values;
      for (const VrpoTrainingRow* row : flat.rows) {
        for (double probability : row->old_legal_probabilities) {
          old_probability_values.push_back(static_cast<float>(probability));
        }
      }
      torch::Tensor old_probability_batch;
      try {
        torch::Tensor old_probability_cpu = torch::from_blob(
            old_probability_values.data(),
            {static_cast<int64_t>(old_probability_values.size())},
            torch::TensorOptions().dtype(torch::kFloat32)).clone();
        old_probability_batch = old_probability_cpu.to(placement.device);
      } catch (const std::exception& exception) {
        return fail(std::string("schedule old-probability placement failed: ") +
                    exception.what());
      }
      int64_t old_probability_offset = 0;
      for (size_t index = 0; index < flat.rows.size(); ++index) {
        const VrpoTrainingRow& row = *flat.rows[index];
        const torch::Tensor& log_probs = legal_log_probabilities[index];
        const torch::Tensor& probs = legal_probabilities[index];
        torch::Tensor ratio = torch::exp(
            log_probs[row.chosen_index] - row.old_chosen_log_probability);
        ratio_diagnostics.push_back(ratio.detach());
        if (row.legal_actions.size() > 1) {
          torch::Tensor first = ratio * loss_advantages[index];
          torch::Tensor second = torch::clamp(
              ratio, 1.0 - kVrpoTrainingClipEpsilon,
              1.0 + kVrpoTrainingClipEpsilon) * loss_advantages[index];
          policy_terms.push_back(-torch::minimum(first, second));
          entropy_terms.push_back(-(probs * log_probs).sum());
        }
        torch::Tensor value = actor_output.values.dim() == 2
            ? actor_output.values[index][0]
            : actor_output.values[index];
        torch::Tensor unclipped = torch::pow(value - row.ppo_return, 2);
        torch::Tensor clipped_value = row.ppo_old_value + torch::clamp(
            value - row.ppo_old_value, -kVrpoTrainingClipEpsilon,
            kVrpoTrainingClipEpsilon);
        torch::Tensor clipped = torch::pow(clipped_value - row.ppo_return, 2);
        value_losses.push_back(0.5 * torch::maximum(unclipped, clipped));
        const int64_t legal_count = row.old_legal_probabilities.size();
        torch::Tensor old = old_probability_batch.narrow(
            0, old_probability_offset, legal_count);
        old_probability_offset += legal_count;
        torch::Tensor positive = old > 0.0;
        torch::Tensor safe_old = torch::clamp_min(old, 1e-30);
        kl_diagnostics.push_back(torch::where(
            positive, old * (torch::log(safe_old) - log_probs),
            torch::zeros_like(old)).sum().detach());
        stats.max_abs_advantage = std::max(
            stats.max_abs_advantage, std::abs(advantages[index]));
      }
      torch::Tensor diagnostics_cpu = torch::cat(
          {torch::stack(ratio_diagnostics), torch::stack(kl_diagnostics)})
          .to(torch::kFloat64).contiguous().cpu();
      const double* diagnostic_values = diagnostics_cpu.data_ptr<double>();
      for (size_t index = 0; index < flat.rows.size(); ++index) {
        const double ratio_value = diagnostic_values[index];
        const double kl_value = diagnostic_values[flat.rows.size() + index];
        if (!std::isfinite(ratio_value) || !std::isfinite(kl_value) ||
            kl_value < -1e-5) {
          return fail("schedule actor ratio/KL diagnostic is invalid");
        }
        stats.min_ratio = std::min(stats.min_ratio, ratio_value);
        stats.max_ratio = std::max(stats.max_ratio, ratio_value);
        stats.max_full_legal_kl = std::max(
            stats.max_full_legal_kl, std::max(0.0, kl_value));
      }
      if (policy_terms.empty()) {
        return fail("schedule actor minibatch has no nontrivial decision");
      }
      torch::Tensor total_loss =
          torch::stack(policy_terms).mean() -
          kVrpoTrainingEntropyCoefficient *
              torch::stack(entropy_terms).mean() +
          kVrpoTrainingValueCoefficient * torch::stack(value_losses).mean();
      if (!vrpo_training_internal::FiniteTensor(total_loss)) {
        return fail("schedule actor loss is nonfinite");
      }
      total_loss.backward();
      ++stats.actor_backward_calls;
      double actor_grad_norm = 0.0;
      double value_grad_norm = 0.0;
      bool actor_grad = false;
      bool value_grad = false;
      if (!vrpo_training_internal::GradientNorm(
              actor_model, "", &actor_grad_norm, &actor_grad, error) ||
          !vrpo_training_internal::GradientNorm(
              actor_model, "value_head", &value_grad_norm, &value_grad,
              error) ||
          !actor_grad || actor_grad_norm <= 0.0 || !value_grad) {
        return fail(error != nullptr && !error->empty()
                        ? *error
                        : "schedule actor gradient gate failed");
      }
      stats.max_actor_grad_norm = std::max(
          stats.max_actor_grad_norm, actor_grad_norm);
      stats.max_value_head_grad_norm = std::max(
          stats.max_value_head_grad_norm, value_grad_norm);
      torch::nn::utils::clip_grad_norm_(actor_model.parameters(),
                                       kVrpoTrainingGradientClipNorm);
      actor_optimizer.step();
      ++stats.actor_optimizer_steps;
      stats.actor_rows_seen += flat.rows.size();
      actor_loss_sum += total_loss.detach().item<double>();
      ++actor_loss_count;
      if (deadline != nullptr &&
          !deadline->Check("after actor minibatch", error)) {
        return false;
      }
    }
  }
  actor_optimizer.zero_grad();
  if (!vrpo_training_internal::ModuleValueSha256(
          actor_model, "", &stats.actor_values_after_sha256, error) ||
      !vrpo_training_internal::ModuleValueSha256(
          actor_model, "value_head", &stats.value_head_after_sha256, error)) {
    return false;
  }
  const int64_t expected_steps =
      actor_epochs * kVrpoTrainingMinibatchesPerEpoch;
  if (stats.actor_optimizer_steps != expected_steps ||
      actor_loss_count != expected_steps || stats.q_optimizer_steps != 0 ||
      stats.actor_values_before_sha256 == stats.actor_values_after_sha256 ||
      stats.value_head_before_sha256 == stats.value_head_after_sha256) {
    return fail("schedule actor-only PPO movement/step postcondition failed");
  }
  stats.actor_loss_mean = actor_loss_sum / actor_loss_count;
  stats.q_loss_mean = 0.0;
  if (!std::isfinite(stats.actor_loss_mean) ||
      !std::isfinite(stats.min_ratio) || !std::isfinite(stats.max_ratio) ||
      !std::isfinite(stats.max_full_legal_kl) ||
      !std::isfinite(stats.max_actor_grad_norm) ||
      !std::isfinite(stats.max_value_head_grad_norm)) {
    return fail("schedule actor-only diagnostics are nonfinite");
  }
  stats.success = true;
  stats.deterministic_summary_sha256 =
      vrpo_training_internal::StatsSha256(stats);
  transaction.Commit();
  if (VrpoScheduleQTargetComputations() != 0) {
    return fail("schedule actor-only Q-target counter changed");
  }
  output->ppo = std::move(stats);
  output->q_target_computations = VrpoScheduleQTargetComputations();
  return true;
}

struct VrpoScheduleHealthMetrics {
  int64_t nontrivial_rows = 0;
  double kl_mean = 0.0;
  double kl_p95 = 0.0;
  double kl_p99 = 0.0;
  double kl_max = 0.0;
  double ratio_p01 = 0.0;
  double ratio_p50 = 0.0;
  double ratio_p99 = 0.0;
  double clip_fraction = 0.0;
  double legal_tv_p95 = 0.0;
  double greedy_argmax_change_rate = 0.0;
  bool eligible = false;
};

inline double VrpoScheduleNearestRankQuantile(std::vector<double> values,
                                               double probability) {
  if (values.empty() || !std::isfinite(probability) || probability < 0.0 ||
      probability > 1.0) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  std::sort(values.begin(), values.end());
  if (probability == 0.0) return values.front();
  const size_t rank = std::min(
      values.size() - 1,
      static_cast<size_t>(std::ceil(probability * values.size()) - 1));
  return values[rank];
}

inline bool FinalizeVrpoScheduleHealthMetrics(
    const std::vector<double>& kls, const std::vector<double>& ratios,
    const std::vector<double>& tvs, int64_t argmax_changes,
    VrpoScheduleHealthMetrics* output, std::string* error) {
  if (output == nullptr || kls.empty() || kls.size() != ratios.size() ||
      kls.size() != tvs.size() || argmax_changes < 0 ||
      argmax_changes > static_cast<int64_t>(kls.size())) {
    if (error != nullptr) *error = "health metric inputs are incomplete";
    return false;
  }
  for (const auto* values : {&kls, &ratios, &tvs}) {
    if (!std::all_of(values->begin(), values->end(),
                     [](double value) { return std::isfinite(value); })) {
      if (error != nullptr) *error = "health metrics contain nonfinite values";
      return false;
    }
  }
  VrpoScheduleHealthMetrics metrics;
  metrics.nontrivial_rows = kls.size();
  for (double value : kls) metrics.kl_mean += value;
  metrics.kl_mean /= kls.size();
  metrics.kl_p95 = VrpoScheduleNearestRankQuantile(kls, 0.95);
  metrics.kl_p99 = VrpoScheduleNearestRankQuantile(kls, 0.99);
  metrics.kl_max = *std::max_element(kls.begin(), kls.end());
  metrics.ratio_p01 = VrpoScheduleNearestRankQuantile(ratios, 0.01);
  metrics.ratio_p50 = VrpoScheduleNearestRankQuantile(ratios, 0.50);
  metrics.ratio_p99 = VrpoScheduleNearestRankQuantile(ratios, 0.99);
  metrics.legal_tv_p95 = VrpoScheduleNearestRankQuantile(tvs, 0.95);
  metrics.clip_fraction = std::count_if(
      ratios.begin(), ratios.end(), [](double ratio) {
        return ratio < 1.0 - kVrpoTrainingClipEpsilon ||
               ratio > 1.0 + kVrpoTrainingClipEpsilon;
      }) / static_cast<double>(ratios.size());
  metrics.greedy_argmax_change_rate =
      argmax_changes / static_cast<double>(kls.size());
  metrics.eligible =
      metrics.kl_mean <= 0.05 && metrics.kl_p99 <= 0.5 &&
      metrics.clip_fraction <= 0.20 && metrics.ratio_p01 >= 0.5 &&
      metrics.ratio_p99 <= 2.0 &&
      metrics.greedy_argmax_change_rate >= 0.005 &&
      metrics.greedy_argmax_change_rate <= 0.15;
  *output = metrics;
  return true;
}

inline std::vector<size_t> SelectVrpoScheduleEligibleCells(
    const std::vector<VrpoScheduleHealthMetrics>& metrics) {
  std::vector<size_t> eligible;
  for (size_t i = 0; i < metrics.size(); ++i) {
    if (metrics[i].eligible) eligible.push_back(i);
  }
  if (eligible.size() <= 1) return eligible;
  const size_t lowest_kl = *std::min_element(
      eligible.begin(), eligible.end(), [&](size_t a, size_t b) {
        if (metrics[a].kl_mean != metrics[b].kl_mean) {
          return metrics[a].kl_mean < metrics[b].kl_mean;
        }
        return a < b;
      });
  const size_t highest_change = *std::max_element(
      eligible.begin(), eligible.end(), [&](size_t a, size_t b) {
        if (metrics[a].greedy_argmax_change_rate !=
            metrics[b].greedy_argmax_change_rate) {
          return metrics[a].greedy_argmax_change_rate <
                 metrics[b].greedy_argmax_change_rate;
        }
        return a > b;
      });
  if (lowest_kl != highest_change) return {lowest_kl, highest_change};
  size_t second = eligible.front();
  for (size_t candidate : eligible) {
    if (candidate != lowest_kl &&
        (second == lowest_kl ||
         metrics[candidate].greedy_argmax_change_rate >
             metrics[second].greedy_argmax_change_rate)) {
      second = candidate;
    }
  }
  return second == lowest_kl ? std::vector<size_t>{lowest_kl}
                             : std::vector<size_t>{lowest_kl, second};
}

inline bool EvaluateVrpoSchedulePolicyHealth(
    const std::vector<VrpoTrainingEpisode>& episodes,
    const VrpoActorForward& actor_forward, const torch::Device& device,
    int64_t chunk_rows, VrpoScheduleHealthMetrics* output,
    std::string* error) {
  auto fail = [&](const std::string& message) {
    if (error != nullptr) *error = message;
    return false;
  };
  if (!actor_forward || output == nullptr || chunk_rows <= 0 ||
      chunk_rows > 4096) {
    return fail("health evaluator arguments are invalid");
  }
  std::vector<const VrpoTrainingRow*> rows;
  for (const auto& episode : episodes) {
    for (const auto& row : episode.rows) {
      if (row.legal_actions.size() > 1) rows.push_back(&row);
    }
  }
  if (rows.empty()) return fail("health evaluator has no nontrivial rows");
  std::vector<double> kls, ratios, tvs;
  int64_t argmax_changes = 0;
  torch::NoGradGuard no_grad;
  for (size_t begin = 0; begin < rows.size(); begin += chunk_rows) {
    const size_t end = std::min(rows.size(), begin + chunk_rows);
    std::vector<torch::Tensor> observations;
    observations.reserve(end - begin);
    for (size_t i = begin; i < end; ++i) {
      observations.push_back(rows[i]->actor_input);
    }
    torch::Tensor inputs = torch::stack(observations).to(device, torch::kFloat32);
    VrpoActorTrainingOutput evaluated;
    try {
      evaluated = actor_forward(inputs);
    } catch (const std::exception& exception) {
      return fail(std::string("health evaluator forward failed: ") +
                  exception.what());
    }
    if (!evaluated.logits.defined() || evaluated.logits.dim() != 2 ||
        evaluated.logits.size(0) != static_cast<int64_t>(end - begin) ||
        evaluated.logits.size(1) != kVrpoDuneActionDim ||
        !vrpo_training_internal::FiniteTensor(evaluated.logits)) {
      return fail("health evaluator output is malformed");
    }
    torch::Tensor logits = evaluated.logits.detach().cpu();
    for (size_t local = 0; local < end - begin; ++local) {
      const VrpoTrainingRow& row = *rows[begin + local];
      std::vector<double> post;
      if (!VrpoTrainingLegalProbabilities(logits[local], row, 10.0, &post,
                                          error)) {
        return false;
      }
      double kl = 0.0;
      double tv = 0.0;
      size_t old_argmax = 0;
      size_t new_argmax = 0;
      for (size_t action = 0; action < post.size(); ++action) {
        const double old = row.old_legal_probabilities[action];
        if (!(post[action] > 0.0) || old < 0.0) {
          return fail("health evaluator probability is invalid");
        }
        if (old > 0.0) kl += old * (std::log(old) - std::log(post[action]));
        tv += std::abs(old - post[action]);
        if (row.old_legal_probabilities[action] >
            row.old_legal_probabilities[old_argmax]) old_argmax = action;
        if (post[action] > post[new_argmax]) new_argmax = action;
      }
      const double old_chosen =
          row.old_legal_probabilities[row.chosen_index];
      if (!(old_chosen > 0.0)) return fail("old chosen probability is zero");
      kls.push_back(std::max(0.0, kl));
      ratios.push_back(post[row.chosen_index] / old_chosen);
      tvs.push_back(0.5 * tv);
      if (old_argmax != new_argmax) ++argmax_changes;
    }
  }
  return FinalizeVrpoScheduleHealthMetrics(kls, ratios, tvs,
                                           argmax_changes, output, error);
}

inline json::Object VrpoScheduleHealthJson(
    const VrpoScheduleHealthMetrics& metrics) {
  json::Object out;
  out["nontrivial_rows"] = metrics.nontrivial_rows;
  out["full_legal_kl_mean"] = metrics.kl_mean;
  out["full_legal_kl_p95"] = metrics.kl_p95;
  out["full_legal_kl_p99"] = metrics.kl_p99;
  out["full_legal_kl_max"] = metrics.kl_max;
  out["chosen_ratio_p01"] = metrics.ratio_p01;
  out["chosen_ratio_p50"] = metrics.ratio_p50;
  out["chosen_ratio_p99"] = metrics.ratio_p99;
  out["clip_fraction"] = metrics.clip_fraction;
  out["legal_tv_p95"] = metrics.legal_tv_p95;
  out["greedy_argmax_change_rate"] = metrics.greedy_argmax_change_rate;
  out["eligible"] = metrics.eligible;
  return out;
}

enum class VrpoScheduleFailurePoint {
  kNone,
  kAfterCorpus,
  kAfterFirstCell,
  kBeforeGlobalStatus,
  kOversize,
  kDeadline,
  kAfterActorTemp,
  kFinalFsyncDeadline,
};

inline bool ExerciseVrpoScheduleResourceFailureForTest(
    const std::filesystem::path& fresh_root,
    VrpoScheduleFailurePoint failure_point, std::string* error) {
  if (std::filesystem::exists(fresh_root) ||
      (failure_point != VrpoScheduleFailurePoint::kOversize &&
       failure_point != VrpoScheduleFailurePoint::kDeadline &&
       failure_point != VrpoScheduleFailurePoint::kAfterActorTemp &&
       failure_point != VrpoScheduleFailurePoint::kFinalFsyncDeadline)) {
    if (error != nullptr) *error = "resource-failure probe input is invalid";
    return false;
  }
  std::error_code ec;
  if (!std::filesystem::create_directories(fresh_root, ec) || ec) {
    if (error != nullptr) *error = "resource-failure probe cannot create root";
    return false;
  }
  auto fail = [&](const std::string& message) {
    if (error != nullptr) *error = message;
    vrpo_schedule_internal::CleanupFreshRoot(fresh_root);
    return false;
  };
  int64_t peak = 0;
  if (failure_point == VrpoScheduleFailurePoint::kOversize) {
    std::string observed_error;
    if (!vrpo_schedule_internal::ObserveRootBytes(
            fresh_root, kVrpoScheduleTemporaryByteCeiling + 1, &peak,
            &observed_error)) {
      return fail(observed_error);
    }
  } else if (failure_point == VrpoScheduleFailurePoint::kDeadline) {
    const auto deadline = VrpoScheduleRunDeadline::ExpiredForTest();
    std::string deadline_error;
    if (!deadline.Check("resource probe", &deadline_error)) {
      return fail(deadline_error);
    }
  } else if (failure_point == VrpoScheduleFailurePoint::kAfterActorTemp) {
    const auto temp = fresh_root / ".actor_model.pt.tmp";
    std::ofstream stream(temp, std::ios::binary | std::ios::trunc);
    stream << "injected-temp";
    stream.flush();
    if (!stream) return fail("resource-failure temp injection failed");
    return fail("injected failure after actor temp");
  } else {
    const auto status_tmp = fresh_root / ".SCREEN_RESULT.json.tmp";
    const auto status = fresh_root / "SCREEN_RESULT.json";
    {
      std::ofstream stream(status_tmp, std::ios::trunc);
      stream << "{\"status\":\"PASS\"}\n";
      stream.flush();
      if (!stream) return fail("final-fsync injection temp write failed");
    }
    std::string fsync_error;
    if (!VrpoFsyncFile(status_tmp, &fsync_error)) return fail(fsync_error);
    std::filesystem::rename(status_tmp, status);
    if (!VrpoFsyncDirectory(fresh_root, &fsync_error)) {
      return fail(fsync_error);
    }
    const auto deadline = VrpoScheduleRunDeadline::ExpiredForTest();
    std::string deadline_error;
    if (!deadline.Check("after final status fsync", &deadline_error)) {
      return fail(deadline_error);
    }
  }
  return fail("resource-failure probe did not trigger");
}

inline bool LoadAndValidateVrpoScheduleSourceModules(
    const std::filesystem::path& directory,
    const VrpoPhase4ArmConfig& expected_arm,
    const VrpoPhase4ManifestBinding& binding,
    const VrpoExpandedExpectedLayout& layout,
    const VrpoExpandedArchiveIdentity& registered_identity,
    std::shared_ptr<SharedDunePolicyValueNetImpl> actor,
    std::shared_ptr<torch::nn::Module> q, const json::Object& manifest,
    std::string* error) {
  auto fail = [&](const std::string& message) {
    if (error != nullptr) *error = message;
    return false;
  };
  if (actor == nullptr || q == nullptr || expected_arm.arm_id != "PPO_CAP10" ||
      expected_arm.algorithm != VrpoPhase4Algorithm::kPpo ||
      expected_arm.logit_cap != 10.0) {
    return fail("schedule source module load arguments are invalid");
  }
  const auto paths = VrpoExpandedPaths(directory);
  const std::set<std::string> expected_names = {
      "actor_model.pt", "q_model.pt", "actor_optimizer.pt",
      "q_optimizer.pt", "manifest.json"};
  std::set<std::string> actual_names;
  std::error_code ec;
  for (const auto& entry : std::filesystem::directory_iterator(directory, ec)) {
    if (ec || !entry.is_regular_file() || entry.is_symlink()) {
      return fail("schedule source archive contains a nonregular entry");
    }
    actual_names.insert(entry.path().filename().string());
  }
  VrpoExpandedArchiveIdentity observed_identity;
  if (ec || actual_names != expected_names ||
      !ComputeVrpoExpandedArchiveIdentity(
          directory, &observed_identity, error) ||
      observed_identity.combined_sha256 !=
          registered_identity.combined_sha256) {
    return fail("schedule source archive file set/identity mismatch");
  }
  auto require_string = [&](const char* key, const std::string& expected) {
    const auto it = manifest.find(key);
    return it != manifest.end() && it->second.IsString() &&
           it->second.GetString() == expected;
  };
  auto require_int = [&](const char* key, int64_t expected) {
    const auto it = manifest.find(key);
    return it != manifest.end() && it->second.IsInt() &&
           it->second.GetInt() == expected;
  };
  const auto moments = manifest.find("source_optimizer_moments_loaded");
  if (!require_string("schema", kVrpoExpandedCheckpointSchema) ||
      !require_int("global_update", 0) ||
      !require_int("next_episode_id",
                   static_cast<int64_t>(binding.start_episode_id)) ||
      moments == manifest.end() || !moments->second.IsBool() ||
      moments->second.GetBool()) {
    return fail("schedule source archive is not a zero-update origin");
  }
  for (size_t index = 0; index < observed_identity.files.size(); ++index) {
    const auto& file = observed_identity.files[index];
    const char* stem = index == 0 ? "actor_model"
        : index == 1 ? "q_model"
        : index == 2 ? "actor_optimizer"
        : index == 3 ? "q_optimizer"
                     : nullptr;
    if (stem == nullptr) continue;
    if (!require_string((std::string(stem) + "_filename").c_str(),
                        file.filename) ||
        !require_int((std::string(stem) + "_size").c_str(), file.size) ||
        !require_string((std::string(stem) + "_sha256").c_str(),
                        file.sha256)) {
      return fail("schedule source manifest file identity mismatch");
    }
  }
  try {
    if (!LoadVrpoModule(*actor, paths.actor_model, error) ||
        !LoadVrpoModule(*q, paths.q_model, error)) {
      return fail(error != nullptr ? *error : "schedule module load failed");
    }
  } catch (const std::exception& exception) {
    return fail(std::string("schedule module deserialize failed: ") +
                exception.what());
  }
  std::string actor_hash;
  std::string q_hash;
  if (!vrpo_training_internal::ModuleValueSha256(
          *actor, "", &actor_hash, error) ||
      !VrpoExpandedQValueIdentitySha256(*q, layout, &q_hash, error) ||
      actor_hash != binding.actor_subset_sha256 ||
      q_hash != binding.q_init_sha256 ||
      !require_string("actor_values_sha256", actor_hash) ||
      !require_string("q_values_sha256", q_hash) ||
      !ValidateVrpoExpandedLiveLayout(*actor, *q, layout, binding, error)) {
    return fail(error != nullptr && !error->empty()
                    ? *error
                    : "schedule loaded module identity/layout mismatch");
  }
  const auto contract = manifest.find("phase4_contract");
  if (contract == manifest.end() || !contract->second.IsObject() ||
      !ValidateVrpoPhase4ManifestStrict(
          contract->second.GetObject(), expected_arm, binding, error)) {
    return fail(error != nullptr && !error->empty()
                    ? *error
                    : "schedule source phase4 contract is invalid");
  }
  return true;
}

inline bool WriteVrpoScheduleScreen(
    const VrpoScheduleStartupConfig& startup,
    const VrpoPhase4ManifestBinding& binding,
    const VrpoExpandedExpectedLayout& layout,
    const VrpoExpandedArchiveIdentity& input_identity,
    const std::vector<VrpoTrainingEpisode>& episodes,
    const VrpoPhase4ePairingStats& pairing,
    std::shared_ptr<SharedDunePolicyValueNetImpl> actor,
    std::shared_ptr<DuneVrpoQNetImpl> q, const torch::Device& device,
    const VrpoScheduleRunDeadline& deadline,
    VrpoScheduleFailurePoint failure_point, json::Object* result,
    std::string* error) {
  bool owns_root = false;
  int64_t measured_peak_bytes = 0;
  auto fail = [&](const std::string& message) {
    if (error != nullptr) *error = message;
    if (result != nullptr) result->clear();
    if (owns_root) vrpo_schedule_internal::CleanupFreshRoot(startup.output_root);
    return false;
  };
  if (result == nullptr || actor == nullptr || q == nullptr ||
      !ValidateVrpoScheduleStartupConfig(startup, error) ||
      !deadline.Check("before schedule writer", error) ||
      !vrpo_schedule_internal::CheckFreeSpaceBeforeStart(
          startup.output_root, error)) {
    return fail(error != nullptr && !error->empty()
                    ? *error
                    : "schedule-screen input is incomplete");
  }
  vrpo_training_internal::ModuleRuntimePlacement actor_placement;
  vrpo_training_internal::ModuleRuntimePlacement q_placement;
  if (!device.is_cuda() ||
      !vrpo_training_internal::CapturePhase4dRuntimePlacements(
          *actor, *q, &actor_placement, &q_placement, error) ||
      !actor_placement.device.is_cuda() ||
      actor_placement.device.index() != startup.runtime_device_index ||
      !ValidateVrpoScheduleQProvenanceInstrumentation(
          kVrpoScheduleQRole, error)) {
    return fail(error != nullptr && !error->empty()
                    ? *error
                    : "schedule-screen requires checked CUDA actor/Q provenance");
  }
  const VrpoPhase4ArmConfig* ppo_cap10 =
      FindCanonicalVrpoPhase4Arm("PPO_CAP10");
  if (ppo_cap10 == nullptr || binding.base_seed != startup.base_seed ||
      binding.start_episode_id != startup.start_episode_id ||
      layout.test_fixture || layout.label != "production_dune_vrpo_layout_v1" ||
      input_identity.combined_sha256.empty() ||
      pairing.episodes != kVrpoScheduleGames ||
      episodes.size() != kVrpoScheduleGames) {
    return fail("schedule-screen input binding/layout/pairing is invalid");
  }
  for (size_t index = 0; index < episodes.size(); ++index) {
    if (episodes[index].episode_id != startup.start_episode_id + index) {
      return fail("schedule-screen corpus episode IDs are not the exact range");
    }
  }
  if (!ValidateVrpoPhase4ePairingStatsAgainstEpisodes(
          pairing, episodes, error)) {
    return fail(error != nullptr ? *error
                                 : "schedule-screen pairing evidence failed");
  }
  std::error_code ec;
  if (!std::filesystem::create_directories(startup.output_root, ec) || ec) {
    return fail("cannot create schedule-screen fresh root");
  }
  owns_root = true;
  if (failure_point == VrpoScheduleFailurePoint::kOversize) {
    if (!vrpo_schedule_internal::ObserveRootBytes(
            startup.output_root, kVrpoScheduleTemporaryByteCeiling + 1,
            &measured_peak_bytes, error)) {
      return fail(*error);
    }
  }
  if (failure_point == VrpoScheduleFailurePoint::kDeadline) {
    return fail("injected schedule deadline failure");
  }
  if (!vrpo_schedule_internal::ObserveRootBytes(
          startup.output_root, -1, &measured_peak_bytes, error)) {
    return fail(*error);
  }

  std::string corpus;
  VrpoScheduleCorpusIdentity corpus_identity;
  if (!EncodeVrpoScheduleActorCorpus(episodes, &corpus, &corpus_identity,
                                     error)) {
    return fail(error != nullptr ? *error : "corpus encoding failed");
  }
  const auto corpus_path = startup.output_root / "ACTOR_CORPUS.bin";
  const auto corpus_tmp = startup.output_root / ".ACTOR_CORPUS.bin.tmp";
  {
    std::ofstream stream(corpus_tmp, std::ios::binary | std::ios::trunc);
    stream.write(corpus.data(), corpus.size());
    stream.flush();
    if (!stream) return fail("actor corpus temp write failed");
  }
  if (!vrpo_schedule_internal::ObserveRootBytes(
          startup.output_root, -1, &measured_peak_bytes, error)) {
    return fail(*error);
  }
  if (!VrpoFsyncFile(corpus_tmp, error)) return fail(*error);
  std::filesystem::rename(corpus_tmp, corpus_path);
  if (!VrpoFsyncDirectory(startup.output_root, error)) return fail(*error);
  if (!deadline.Check("after corpus file commit", error) ||
      !vrpo_schedule_internal::ObserveRootBytes(
          startup.output_root, -1, &measured_peak_bytes, error)) {
    return fail(*error);
  }
  size_t corpus_file_size = 0;
  const std::string corpus_file_sha256 =
      ComputeFileSHA256(corpus_path.string(), &corpus_file_size);
  if (!VrpoPhase4eLowerHex64(corpus_file_sha256) ||
      static_cast<int64_t>(corpus_file_size) != corpus_identity.byte_size) {
    return fail("actor corpus file identity mismatch");
  }
  std::ifstream corpus_stream(corpus_path, std::ios::binary);
  const std::string reloaded_corpus(
      (std::istreambuf_iterator<char>(corpus_stream)),
      std::istreambuf_iterator<char>());
  std::vector<VrpoTrainingEpisode> decoded;
  VrpoScheduleCorpusIdentity decoded_identity;
  if (!DecodeVrpoScheduleActorCorpus(reloaded_corpus, &decoded,
                                     &decoded_identity, error) ||
      decoded_identity.payload_sha256 != corpus_identity.payload_sha256 ||
      decoded_identity.rows != corpus_identity.rows ||
      decoded_identity.episode_sha256 != corpus_identity.episode_sha256) {
    return fail("actor corpus strict reload failed");
  }
  json::Object corpus_manifest;
  corpus_manifest["schema"] = kVrpoScheduleCorpusSchema;
  corpus_manifest["profile"] = startup.profile;
  corpus_manifest["registration_id"] = startup.registration_id;
  corpus_manifest["filename"] = "ACTOR_CORPUS.bin";
  corpus_manifest["file_sha256"] = corpus_file_sha256;
  corpus_manifest["payload_sha256"] = corpus_identity.payload_sha256;
  corpus_manifest["bytes"] = corpus_identity.byte_size;
  corpus_manifest["episodes"] = corpus_identity.episodes;
  corpus_manifest["rows"] = corpus_identity.rows;
  corpus_manifest["start_episode_id"] =
      static_cast<int64_t>(startup.start_episode_id);
  corpus_manifest["end_episode_id_inclusive"] =
      static_cast<int64_t>(startup.start_episode_id + 15);
  corpus_manifest["base_seed"] = static_cast<int64_t>(startup.base_seed);
  corpus_manifest["source_code_sha256"] = startup.source_code_sha256;
  corpus_manifest["executed_binary_sha256"] =
      startup.executed_binary_sha256;
  corpus_manifest["input_archive_sha256"] =
      input_identity.combined_sha256;
  corpus_manifest["input_actor_file_sha256"] =
      input_identity.files[0].sha256;
  corpus_manifest["input_q_file_sha256"] = input_identity.files[1].sha256;
  corpus_manifest["source_actor_model_sha256"] =
      binding.source_actor_model_sha256;
  corpus_manifest["source_actor_manifest_sha256"] =
      binding.source_actor_manifest_sha256;
  corpus_manifest["contains_q_tensors"] = false;
  corpus_manifest["strict_reload_passed"] = true;
  corpus_manifest["status"] = "SEALED";
  const auto corpus_manifest_tmp =
      startup.output_root / ".CORPUS_MANIFEST.json.tmp";
  const auto corpus_manifest_path =
      startup.output_root / "CORPUS_MANIFEST.json";
  {
    std::ofstream stream(corpus_manifest_tmp, std::ios::trunc);
    stream << json::ToString(corpus_manifest, true) << "\n";
    stream.flush();
    if (!stream) return fail("actor corpus manifest temp write failed");
  }
  if (!vrpo_schedule_internal::ObserveRootBytes(
          startup.output_root, -1, &measured_peak_bytes, error)) {
    return fail(*error);
  }
  if (!VrpoFsyncFile(corpus_manifest_tmp, error)) return fail(*error);
  std::filesystem::rename(corpus_manifest_tmp, corpus_manifest_path);
  if (!VrpoFsyncDirectory(startup.output_root, error)) return fail(*error);
  if (!deadline.Check("after sealed corpus manifest", error) ||
      !vrpo_schedule_internal::ObserveRootBytes(
          startup.output_root, -1, &measured_peak_bytes, error)) {
    return fail(*error);
  }
  if (failure_point == VrpoScheduleFailurePoint::kAfterCorpus) {
    return fail("injected schedule failure after corpus");
  }

  json::Array cell_results;
  std::vector<VrpoScheduleHealthMetrics> metrics;
  std::string common_actor_before;
  std::string common_q_before;
  std::array<std::string, 4> common_partitions;
  int64_t total_actor_steps = 0;
  int64_t total_row_presentations = 0;
  const auto* canonical_cells =
      FindVrpoScheduleCellsForProfile(startup.profile);
  if (canonical_cells == nullptr) {
    return fail("schedule-screen profile resolution failed");
  }
  const uint64_t update_seed = vrpo_training_internal::SplitMix64(
      startup.base_seed ^ 0x5343484544554c45ULL);
  for (int epoch = 0; epoch < 4; ++epoch) {
    VrpoEpisodePartitionPlan plan;
    if (!BuildVrpoScheduleEpisodePartitionPlan(
            decoded,
            vrpo_training_internal::SplitMix64(update_seed + epoch + 1),
            &plan, error)) {
      return fail(error != nullptr ? *error
                                   : "schedule partition construction failed");
    }
    common_partitions[epoch] = plan.canonical_sha256;
  }
  for (size_t cell_index = 0; cell_index < canonical_cells->size();
       ++cell_index) {
    if (!deadline.Check("before schedule cell", error)) return fail(*error);
    const VrpoScheduleCellSpec& cell = (*canonical_cells)[cell_index];
    std::string load_error;
    if (!LoadVrpoModule(*actor,
                        VrpoExpandedPaths(startup.input_archive).actor_model,
                        &load_error)) {
      return fail("cell source actor reload failed: " + load_error);
    }
    std::unique_ptr<torch::optim::AdamW> actor_optimizer;
    if (!MakeVrpoScheduleFreshActorOptimizer(
            *actor, cell.learning_rate, &actor_optimizer, &load_error)) {
      return fail("cell fresh actor optimizer construction failed: " +
                  load_error);
    }
    std::string actor_before, q_before;
    if (!vrpo_training_internal::ModuleValueSha256(
            *actor, "", &actor_before, &load_error) ||
        !vrpo_training_internal::ModuleValueSha256(
            *q, "", &q_before, &load_error)) {
      return fail("cell initial model hashing failed: " + load_error);
    }
    if (cell_index == 0) {
      common_actor_before = actor_before;
      common_q_before = q_before;
    } else if (actor_before != common_actor_before || q_before != common_q_before) {
      return fail("cells did not start from byte-identical actor/Q values");
    }
    if (actor_before != binding.actor_subset_sha256) {
      return fail("cell actor/Q provenance differs from bound origin");
    }
    VrpoActorForward actor_forward = [actor](const torch::Tensor& input) {
      const auto out = actor->forward(input);
      return VrpoActorTrainingOutput{out.logits, out.values};
    };
    VrpoScheduleActorOnlyUpdateStats actor_only_stats;
    if (!RunVrpoScheduleActorOnlyPpoCell(
            decoded, update_seed, cell.actor_epochs, *actor,
            *actor_optimizer, actor_forward,
            /*four_epoch_equivalence_test=*/false, &deadline,
            &actor_only_stats,
            &load_error)) {
      return fail("schedule cell update failed: " + load_error);
    }
    const VrpoTrainingUpdateStats& stats = actor_only_stats.ppo;
    std::string q_after;
    if (!vrpo_training_internal::ModuleValueSha256(
            *q, "", &q_after, &load_error) || q_after != q_before ||
        stats.q_optimizer_steps != 0 ||
        actor_only_stats.q_target_computations != 0 ||
        VrpoScheduleQTargetComputations() != 0 ||
        stats.actor_optimizer_steps !=
            cell.actor_epochs * kVrpoTrainingMinibatchesPerEpoch) {
      return fail("schedule cell step/Q-inert postcondition failed");
    }
    for (int epoch = 0; epoch < cell.actor_epochs; ++epoch) {
      if (stats.actor_epoch_partition_sha256[epoch] !=
          common_partitions[epoch]) {
        return fail("schedule cells used different epoch partitions");
      }
    }
    VrpoScheduleHealthMetrics health;
    if (!EvaluateVrpoSchedulePolicyHealth(decoded, actor_forward, device, 256,
                                          &health, &load_error)) {
      return fail("schedule cell health evaluation failed: " + load_error);
    }
    if (!deadline.Check("after schedule health evaluation", error)) {
      return fail(*error);
    }
    const auto cell_dir = startup.output_root / cell.cell_id;
    if (!std::filesystem::create_directory(cell_dir, ec) || ec) {
      return fail("cannot create cell output directory");
    }
    if (!vrpo_schedule_internal::ObserveRootBytes(
            startup.output_root, -1, &measured_peak_bytes, error)) {
      return fail(*error);
    }
    const auto actor_path = cell_dir / "actor_model.pt";
    const auto actor_tmp = cell_dir / ".actor_model.pt.tmp";
    try {
      torch::save(actor, actor_tmp.string());
    } catch (const std::exception& exception) {
      return fail(std::string("actor checkpoint save failed: ") +
                  exception.what());
    }
    if (!deadline.Check("after actor checkpoint temp", error) ||
        !vrpo_schedule_internal::ObserveRootBytes(
            startup.output_root, -1, &measured_peak_bytes, error)) {
      return fail(*error);
    }
    if (failure_point == VrpoScheduleFailurePoint::kAfterActorTemp) {
      return fail("injected schedule failure after actor temp");
    }
    if (!VrpoFsyncFile(actor_tmp, error)) return fail(*error);
    auto reloaded_actor = std::make_shared<SharedDunePolicyValueNetImpl>(
        layout.actor_observation_dim, layout.actor_hidden_dim,
        layout.actor_action_dim, layout.actor_residual_blocks, false);
    reloaded_actor->to(device);
    try {
      torch::load(reloaded_actor, actor_tmp.string(), device);
    } catch (const std::exception& exception) {
      return fail(std::string("actor checkpoint strict reload failed: ") +
                  exception.what());
    }
    std::string reloaded_hash;
    if (!vrpo_training_internal::ModuleValueSha256(
            *reloaded_actor, "", &reloaded_hash, &load_error) ||
        reloaded_hash != stats.actor_values_after_sha256) {
      return fail("actor checkpoint reload hash mismatch");
    }
    VrpoActorForward reloaded_forward =
        [reloaded_actor](const torch::Tensor& input) {
          const auto out = reloaded_actor->forward(input);
          return VrpoActorTrainingOutput{out.logits, out.values};
        };
    torch::Tensor canary = decoded.front().rows.front().actor_input
                               .reshape({1, -1}).to(device);
    torch::Tensor live_canary = actor_forward(canary).logits.detach().cpu();
    torch::Tensor reload_canary =
        reloaded_forward(canary).logits.detach().cpu();
    if (!torch::equal(live_canary, reload_canary)) {
      return fail("actor checkpoint reload canary mismatch");
    }
    std::filesystem::rename(actor_tmp, actor_path);
    if (!vrpo_schedule_internal::ObserveRootBytes(
            startup.output_root, -1, &measured_peak_bytes, error)) {
      return fail(*error);
    }
    size_t actor_file_size = 0;
    const std::string actor_file_sha =
        ComputeFileSHA256(actor_path.string(), &actor_file_size);
    json::Object cell_json;
    cell_json["schema"] = kVrpoScheduleCellResultSchema;
    cell_json["profile"] = startup.profile;
    cell_json["cell_id"] = cell.cell_id;
    cell_json["learning_rate"] = cell.learning_rate;
    cell_json["actor_epochs"] = static_cast<int64_t>(cell.actor_epochs);
    cell_json["actor_optimizer_steps"] = stats.actor_optimizer_steps;
    cell_json["actor_rows_seen"] = stats.actor_rows_seen;
    cell_json["update_seed"] = static_cast<int64_t>(update_seed);
    cell_json["q_optimizer_steps"] = stats.q_optimizer_steps;
    cell_json["q_optimizer_constructed"] = false;
    cell_json["q_optimizer_state_loaded"] = false;
    cell_json["q_role"] = kVrpoScheduleQRole;
    cell_json["q_constructor_calls"] = VrpoQConstructorCalls();
    cell_json["q_forward_calls"] = int64_t{0};
    cell_json["q_target_computations"] =
        actor_only_stats.q_target_computations;
    cell_json["training_rows_source"] =
        "STRICT_DECODED_ACTOR_CORPUS_ONLY";
    cell_json["actor_before_sha256"] = actor_before;
    cell_json["actor_after_sha256"] = stats.actor_values_after_sha256;
    cell_json["q_before_sha256"] = q_before;
    cell_json["q_after_sha256"] = q_after;
    cell_json["actor_loss_mean"] = stats.actor_loss_mean;
    cell_json["q_loss_mean"] = stats.q_loss_mean;
    cell_json["max_actor_grad_norm"] = stats.max_actor_grad_norm;
    cell_json["max_value_head_grad_norm"] =
        stats.max_value_head_grad_norm;
    cell_json["max_q_grad_norm"] = stats.max_q_grad_norm;
    cell_json["min_training_ratio"] = stats.min_ratio;
    cell_json["max_training_ratio"] = stats.max_ratio;
    cell_json["max_training_full_legal_kl"] =
        stats.max_full_legal_kl;
    json::Array partitions;
    for (int epoch = 0; epoch < cell.actor_epochs; ++epoch) {
      partitions.emplace_back(stats.actor_epoch_partition_sha256[epoch]);
    }
    cell_json["actor_epoch_partition_sha256"] = std::move(partitions);
    cell_json["health"] = json::Value(VrpoScheduleHealthJson(health));
    cell_json["actor_model_filename"] = "actor_model.pt";
    cell_json["actor_model_sha256"] = actor_file_sha;
    cell_json["actor_model_size"] = static_cast<int64_t>(actor_file_size);
    cell_json["strict_reload_passed"] = true;
    cell_json["canary_exact"] = true;
    cell_json["optimizer_continuation_saved"] = false;
    cell_json["q_checkpoint_saved"] = false;
    cell_json["purpose"] = "HEALTH_SCREEN_ONLY_NO_STRENGTH_NO_PROMOTION";
    cell_json["status"] = "PASS";
    const std::string serialized = json::ToString(cell_json, true) + "\n";
    const auto cell_status_tmp = cell_dir / ".CELL_RESULT.json.tmp";
    const auto cell_status = cell_dir / "CELL_RESULT.json";
    {
      std::ofstream stream(cell_status_tmp, std::ios::trunc);
      stream << serialized;
      stream.flush();
      if (!stream) return fail("cell result temp write failed");
    }
    if (!vrpo_schedule_internal::ObserveRootBytes(
            startup.output_root, -1, &measured_peak_bytes, error)) {
      return fail(*error);
    }
    if (!VrpoFsyncFile(cell_status_tmp, error)) return fail(*error);
    std::filesystem::rename(cell_status_tmp, cell_status);
    if (!VrpoFsyncDirectory(cell_dir, error)) return fail(*error);
    if (!deadline.Check("after schedule cell commit", error) ||
        !vrpo_schedule_internal::ObserveRootBytes(
            startup.output_root, -1, &measured_peak_bytes, error)) {
      return fail(*error);
    }
    total_actor_steps += stats.actor_optimizer_steps;
    total_row_presentations += stats.actor_rows_seen;
    metrics.push_back(health);
    cell_results.emplace_back(std::move(cell_json));
    if (failure_point == VrpoScheduleFailurePoint::kAfterFirstCell &&
        cell_index == 0) {
      return fail("injected schedule failure after first cell");
    }
  }
  if (total_actor_steps != kVrpoScheduleMaxActorSteps ||
      total_row_presentations != kVrpoScheduleExpectedRowPresentations) {
    return fail("schedule-screen hard step/row-presentation ceiling mismatch");
  }
  VrpoExpandedArchiveIdentity input_identity_after;
  if (!ComputeVrpoExpandedArchiveIdentity(
          startup.input_archive, &input_identity_after, error) ||
      input_identity_after.combined_sha256 != input_identity.combined_sha256) {
    return fail("schedule-screen input archive changed during transaction");
  }
  const int64_t retained_before_status =
      vrpo_schedule_internal::DirectoryRegularBytes(startup.output_root);
  if (retained_before_status < 0 ||
      retained_before_status > kVrpoScheduleRetainedByteCeiling) {
    return fail("schedule-screen retained-byte ceiling exceeded");
  }
  if (!ValidateVrpoScheduleQProvenanceInstrumentation(
          kVrpoScheduleQRole, error)) {
    return fail(*error);
  }
  const auto selected = SelectVrpoScheduleEligibleCells(metrics);
  json::Array selected_ids;
  for (size_t index : selected) {
    selected_ids.emplace_back((*canonical_cells)[index].cell_id);
  }
  json::Object root;
  root["schema"] = kVrpoScheduleResultSchema;
  root["profile"] = startup.profile;
  root["registration_id"] = startup.registration_id;
  root["purpose"] = "HEALTH_SCREEN_ONLY_NO_STRENGTH_NO_PROMOTION";
  root["training_authorized"] = false;
  root["promotion_authorized"] = false;
  root["one_collection"] = true;
  root["one_gpu_process"] = startup.one_gpu_process;
  root["runtime_device"] =
      "cuda:" + std::to_string(startup.runtime_device_index);
  root["runtime_device_is_cuda"] = startup.runtime_device_is_cuda;
  root["runtime_device_index"] =
      static_cast<int64_t>(startup.runtime_device_index);
  root["runtime_process_id"] = startup.runtime_process_id;
  root["rollout_games"] = static_cast<int64_t>(kVrpoScheduleGames);
  root["base_seed"] = static_cast<int64_t>(startup.base_seed);
  root["start_episode_id"] = static_cast<int64_t>(startup.start_episode_id);
  root["end_episode_id_inclusive"] =
      static_cast<int64_t>(startup.start_episode_id + 15);
  root["source_code_sha256"] = startup.source_code_sha256;
  root["executed_binary_sha256"] = startup.executed_binary_sha256;
  root["executed_binary_size"] = startup.executed_binary_size;
  root["input_archive_sha256"] = input_identity.combined_sha256;
  root["input_actor_file_sha256"] = input_identity.files[0].sha256;
  root["input_q_file_sha256"] = input_identity.files[1].sha256;
  root["input_manifest_file_sha256"] = input_identity.files[4].sha256;
  root["source_actor_model_sha256"] = binding.source_actor_model_sha256;
  root["source_actor_manifest_sha256"] = binding.source_actor_manifest_sha256;
  root["initial_actor_values_sha256"] = common_actor_before;
  root["initial_q_values_sha256"] = common_q_before;
  root["corpus_schema"] = kVrpoScheduleCorpusSchema;
  root["corpus_filename"] = "ACTOR_CORPUS.bin";
  root["corpus_sha256"] = corpus_file_sha256;
  root["corpus_payload_sha256"] = corpus_identity.payload_sha256;
  root["corpus_bytes"] = corpus_identity.byte_size;
  root["corpus_rows"] = corpus_identity.rows;
  root["corpus_contains_q_tensors"] = false;
  json::Array episode_digests;
  for (const auto& digest : corpus_identity.episode_sha256) {
    episode_digests.emplace_back(digest);
  }
  root["corpus_episode_sha256"] = std::move(episode_digests);
  root["cells"] = std::move(cell_results);
  root["selected_eligible_cells"] = std::move(selected_ids);
  root["selected_count"] = static_cast<int64_t>(selected.size());
  root["actor_optimizer_steps_total"] = total_actor_steps;
  root["row_presentations_total"] = total_row_presentations;
  root["q_optimizer_steps_total"] = int64_t{0};
  root["q_optimizer_constructed"] = false;
  root["q_optimizer_state_loaded"] = false;
  root["q_role"] = kVrpoScheduleQRole;
  root["q_constructor_calls"] = VrpoQConstructorCalls();
  root["q_forward_calls"] = VrpoQForwardCheckedCalls();
  root["q_target_computations"] = VrpoScheduleQTargetComputations();
  root["training_rows_source"] = "STRICT_DECODED_ACTOR_CORPUS_ONLY";
  root["original_rich_rows_used_after_seal"] = false;
  root["retained_byte_ceiling"] = kVrpoScheduleRetainedByteCeiling;
  root["temporary_byte_ceiling"] = kVrpoScheduleTemporaryByteCeiling;
  root["wall_clock_ceiling_seconds"] = int64_t{1200};
  root["elapsed_seconds_sampled_before_status_commit"] =
      deadline.ElapsedSeconds();
  root["final_deadline_check_after_status_fsync_required"] = true;
  root["input_archive_untouched"] = true;
  root["status_last"] = true;
  if (failure_point == VrpoScheduleFailurePoint::kBeforeGlobalStatus) {
    return fail("injected schedule failure before global status");
  }
  if (!deadline.Check("before global status", error)) return fail(*error);
  root["status"] = "PASS";
  const auto status_tmp = startup.output_root / ".SCREEN_RESULT.json.tmp";
  const auto status_path = startup.output_root / "SCREEN_RESULT.json";
  std::string serialized_status;
  int64_t prospective_retained = retained_before_status;
  for (int pass = 0; pass < 4; ++pass) {
    root["retained_bytes"] = prospective_retained;
    root["measured_peak_total_bytes"] =
        std::max(measured_peak_bytes, prospective_retained);
    serialized_status = json::ToString(root, true) + "\n";
    const int64_t next = retained_before_status + serialized_status.size();
    if (next == prospective_retained) break;
    prospective_retained = next;
  }
  if (prospective_retained > kVrpoScheduleRetainedByteCeiling) {
    return fail("schedule-screen final retained-byte ceiling exceeded");
  }
  {
    std::ofstream stream(status_tmp, std::ios::trunc);
    stream << serialized_status;
    stream.flush();
    if (!stream) return fail("global screen result temp write failed");
  }
  if (!vrpo_schedule_internal::ObserveRootBytes(
          startup.output_root, -1, &measured_peak_bytes, error)) {
    return fail(*error);
  }
  if (!VrpoFsyncFile(status_tmp, error)) return fail(*error);
  std::filesystem::rename(status_tmp, status_path);
  if (!VrpoFsyncDirectory(startup.output_root, error)) return fail(*error);
  if (failure_point == VrpoScheduleFailurePoint::kFinalFsyncDeadline) {
    return fail("injected deadline overrun after final status fsync");
  }
  if (!deadline.Check("after final status fsync", error)) {
    return fail(*error);
  }
  const int64_t final_retained =
      vrpo_schedule_internal::DirectoryRegularBytes(startup.output_root);
  const auto retained_it = root.find("retained_bytes");
  const auto peak_it = root.find("measured_peak_total_bytes");
  if (final_retained != prospective_retained ||
      retained_it == root.end() || !retained_it->second.IsInt() ||
      retained_it->second.GetInt() != final_retained ||
      peak_it == root.end() || !peak_it->second.IsInt() ||
      peak_it->second.GetInt() != measured_peak_bytes ||
      final_retained > kVrpoScheduleRetainedByteCeiling) {
    return fail("schedule-screen final measured byte accounting mismatch");
  }
  owns_root = false;
  *result = std::move(root);
  return true;
}

}  // namespace open_spiel

#endif  // OPEN_SPIEL_BUILD_WITH_LIBTORCH
#endif  // OPEN_SPIEL_EXAMPLES_DUNE_VRPO_SCHEDULE_SCREEN_H_
