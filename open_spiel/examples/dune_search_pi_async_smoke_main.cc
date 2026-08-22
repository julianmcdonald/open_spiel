#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <numeric>
#include <optional>
#include <queue>
#include <set>
#include <shared_mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <torch/torch.h>
#include <c10/cuda/CUDAFunctions.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

#include "open_spiel/abseil-cpp/absl/flags/flag.h"
#include "open_spiel/abseil-cpp/absl/flags/parse.h"
#include "open_spiel/spiel.h"
#include "open_spiel/utils/json.h"
#include "dune_batched_evaluator.h"
#include "dune_ppo_training_utils.h"
#include "dune_seed_utils.h"
#include "dune_search_pi.h"
#include "dune_search_pi_replay.h"
#include "dune_search_pi_scratch.h"
#include "dune_split_evaluator.h"

// Link-only PPO flag definitions required by dune_ppo_training_utils.cc.
ABSL_FLAG(int, ppo_minibatch_size, 2048, "");
ABSL_FLAG(int, ppo_update_epochs, 4, "");
ABSL_FLAG(double, ppo_clip_epsilon, 0.2, "");
ABSL_FLAG(bool, normalize_advantages, true, "");
ABSL_FLAG(bool, ppo_clip_value_loss, true, "");
ABSL_FLAG(double, entropy_coef, 0.01, "");
ABSL_FLAG(double, value_coef, 0.5, "");
ABSL_FLAG(double, target_kl, 0.0, "");
ABSL_FLAG(bool, train_amp, true, "");
ABSL_FLAG(bool, diagnostics_only, false, "");

ABSL_FLAG(std::string, output_dir, "", "Async smoke runtime.");
ABSL_FLAG(uint64_t, seed, 8291999, "Master seed.");
ABSL_FLAG(int64_t, first_episode_id, 3200000, "First episode ID.");
ABSL_FLAG(int, updates, 2, "Learner updates in the smoke.");
ABSL_FLAG(int, target_generation, -1,
          "Production target generation; -1 uses updates from current state.");
ABSL_FLAG(int, games_per_update, 4, "Completed game equivalents per update.");
ABSL_FLAG(int, actors, 4, "Actor workers.");
ABSL_FLAG(int, inference_lanes, 2,
          "Independent collector-model/batcher lanes per snapshot.");
ABSL_FLAG(int, batch_target, 32, "Inference batch target.");
ABSL_FLAG(int, batcher_timeout_ms, 2, "Inference batch timeout.");
ABSL_FLAG(int, full_primary_budget, 200, "Full primary simulations.");
ABSL_FLAG(int, full_other_budget, 64, "Full other simulations.");
ABSL_FLAG(int, cheap_primary_budget, 32, "Cheap primary simulations.");
ABSL_FLAG(int, cheap_other_budget, 16, "Cheap other simulations.");
ABSL_FLAG(double, full_root_probability, 0.25, "Full root probability.");
ABSL_FLAG(double, relative_time_budget_ms, 600000.0,
          "Per-root liveness watchdog.");
ABSL_FLAG(int, minibatch_size, 256, "Learner minibatch size.");
ABSL_FLAG(double, logit_cap, 10.0, "Learner logit cap.");
ABSL_FLAG(double, grad_clip_norm, 0.5, "Learner gradient clip.");
ABSL_FLAG(bool, crash_after_update, false,
          "Stop cleanly after update one to exercise resume.");
ABSL_FLAG(bool, verify_production_manifest_only, false,
          "Load and validate a production manifest, then exit without games.");
ABSL_FLAG(bool, checkpoint_every_update, false,
          "Test-only: publish a standard checkpoint after every update.");
ABSL_FLAG(bool, force_health_failure_for_test, false,
          "Test-only: make manifest verification fail closed.");
ABSL_FLAG(std::string, resume, "", "Async smoke state JSON.");
ABSL_FLAG(std::string, production_checkpoint_manifest, "",
          "Production checkpoint manifest used to seed a faithful comparison.");
ABSL_FLAG(std::string, production_run_dir, "",
          "Standard production run root for checkpoints and telemetry.");
ABSL_FLAG(size_t, replay_max_rows, 200000, "Maximum replay rows sampled per update.");
ABSL_FLAG(uint64_t, shaping_start_env_steps, 0,
          "Environment step at which shaping decay starts.");
ABSL_FLAG(uint64_t, shaping_decay_env_steps, 10000000,
          "Environment steps over which shaping decays.");

namespace open_spiel {
namespace {

struct CollectorSnapshot {
  struct CapBaseline {
    uint64_t decisions = 0;
    uint64_t legal_logits = 0;
    uint64_t saturated = 0;
  };

  int generation = 0;
  int inference_lanes = 0;
  uint64_t total_env_steps = 0;
  uint64_t shaping_start_env_steps = 0;
  uint64_t shaping_decay_env_steps = 0;
  float shaping_lambda = 1.0f;
  std::vector<std::shared_ptr<SharedDunePolicyValueNetImpl>> models;
  std::vector<std::shared_ptr<std::shared_mutex>> model_mutexes;
  std::vector<std::shared_ptr<BatchedEvaluator>> batchers;
  std::vector<std::shared_ptr<algorithms::Evaluator>> evaluators;
  std::vector<CapBaseline> cap_baselines;

  BatchedLogitCapAggregate ConsumeCapTelemetry() {
    BatchedLogitCapAggregate interval;
    if (cap_baselines.size() != batchers.size()) {
      cap_baselines.resize(batchers.size());
    }
    for (size_t lane = 0; lane < batchers.size(); ++lane) {
      const BatchedLogitCapAggregate current =
          batchers[lane]->GetLogitCapStats();
      CapBaseline& baseline = cap_baselines[lane];
      interval.decisions += current.decisions - baseline.decisions;
      interval.legal_logits += current.legal_logits - baseline.legal_logits;
      interval.saturated += current.saturated - baseline.saturated;
      interval.pre_max_abs = std::max(interval.pre_max_abs,
                                      current.pre_max_abs);
      interval.post_max_abs = std::max(interval.post_max_abs,
                                       current.post_max_abs);
      baseline.decisions = current.decisions;
      baseline.legal_logits = current.legal_logits;
      baseline.saturated = current.saturated;
    }
    return interval;
  }
};

struct Task {
  int64_t episode_id = -1;
  std::shared_ptr<CollectorSnapshot> snapshot;
};

struct Completion {
  int64_t episode_id = -1;
  int collector_generation = -1;
  int start_staleness = 0;
  double actor_completion_wall_seconds = 0.0;
  std::vector<SearchPiRow> rows;
  ScratchVisitGenerationStats visit_stats;
  SearchPiGameOutcome outcome;
};

struct LedgerEntry {
  int64_t episode_id = -1;
  int collector_generation = -1;
  std::string shard_path;
  int64_t rows = 0;
  int64_t requested_simulations = 0;
  int64_t completed_simulations = 0;
  int64_t full_policy_target_rows = 0;
  int64_t trajectory_transitions = 0;
  int start_staleness = 0;
  int64_t fallbacks = 0;
  int64_t watchdog_timeouts = 0;
  int64_t non_timeout_early_exits = 0;
};

struct UpdateCollectionTelemetry {
  int64_t requested_simulations = 0;
  int64_t completed_simulations = 0;
  int64_t full_policy_target_rows = 0;
  int64_t trajectory_transitions = 0;
  int64_t fallbacks = 0;
  int64_t watchdog_timeouts = 0;
  int64_t non_timeout_early_exits = 0;
  int64_t vp_breadcrumb_events = 0;
  int64_t specimen_anti_breadcrumb_events = 0;
  double vp_breadcrumb_raw_reward_sum = 0.0;
  double specimen_anti_breadcrumb_raw_penalty_sum = 0.0;
  double vp_breadcrumb_shaped_reward_sum = 0.0;
  double specimen_anti_breadcrumb_shaped_reward_sum = 0.0;
  double collector_generation_sum = 0.0;
  int collector_generation_min = std::numeric_limits<int>::max();
  int collector_generation_max = std::numeric_limits<int>::min();
  double collector_start_staleness_sum = 0.0;
  int collector_start_staleness_max = 0;
  double collector_staleness_sum = 0.0;
  int collector_staleness_max = 0;
  int64_t visit_full_rows = 0;
  int64_t visit_cheap_rows = 0;
  double visit_entropy_sum = 0.0;
  double visit_kl_sum = 0.0;
  int64_t visit_stat_games = 0;

  void Add(const Completion& completion, const ScratchVisitGenerationStats& visit,
           const LedgerEntry& entry, int learner_generation) {
    const SearchPiGameOutcome& outcome = completion.outcome;
    requested_simulations += entry.requested_simulations;
    completed_simulations += entry.completed_simulations;
    full_policy_target_rows += visit.full_rows;
    trajectory_transitions += outcome.trajectory_transitions;
    fallbacks += outcome.fallbacks;
    watchdog_timeouts += outcome.watchdog_timeouts;
    non_timeout_early_exits += outcome.non_timeout_early_exits;
    vp_breadcrumb_events += outcome.vp_breadcrumb_events;
    specimen_anti_breadcrumb_events += outcome.specimen_anti_breadcrumb_events;
    vp_breadcrumb_raw_reward_sum += outcome.vp_breadcrumb_raw_reward_sum;
    specimen_anti_breadcrumb_raw_penalty_sum +=
        outcome.specimen_anti_breadcrumb_raw_penalty_sum;
    vp_breadcrumb_shaped_reward_sum +=
        outcome.vp_breadcrumb_shaped_reward_sum;
    specimen_anti_breadcrumb_shaped_reward_sum +=
        outcome.specimen_anti_breadcrumb_shaped_reward_sum;
    collector_generation_sum += completion.collector_generation;
    collector_generation_min =
        std::min(collector_generation_min, completion.collector_generation);
    collector_generation_max =
        std::max(collector_generation_max, completion.collector_generation);
    collector_start_staleness_sum += completion.start_staleness;
    collector_start_staleness_max =
        std::max(collector_start_staleness_max, completion.start_staleness);
    const int staleness = std::max(
        0, learner_generation - completion.collector_generation);
    collector_staleness_sum += staleness;
    collector_staleness_max = std::max(collector_staleness_max, staleness);
    visit_entropy_sum += visit.target_entropy_mean * visit.full_rows;
    visit_kl_sum += visit.target_kl_mean * visit.full_rows;
    visit_stat_games += visit.full_rows > 0 ? 1 : 0;
  }
};

struct SmokeState {
  int next_update = 0;
  int initial_generation = 0;
  int64_t run_first_episode_id = 0;
  int64_t target_episode_end = 0;
  int64_t first_episode_id = 0;
  int64_t next_episode_id = 0;
  int published_generation = 0;
  uint64_t total_env_steps = 0;
  std::vector<std::string> replay_paths;
  std::vector<std::vector<std::string>> replay_cohort_groups;
  std::vector<LedgerEntry> completed;
};

struct ProductionResumeState {
  int generation = 0;
  int64_t first_episode_id = 0;
  int64_t next_episode_id = 0;
  uint64_t total_env_steps = 0;
  std::string model_checkpoint;
  std::string optimizer_checkpoint;
  std::vector<std::string> replay_paths;
  std::vector<std::vector<std::string>> replay_cohort_groups;
};

void AtomicJson(const std::string& path, const json::Object& object) {
  const std::string tmp = path + ".tmp";
  std::ofstream out(tmp, std::ios::trunc);
  if (!out) SpielFatalError("cannot write async smoke state: " + tmp);
  out << json::ToString(object, true) << "\n";
  out.flush();
  if (!out) SpielFatalError("failed writing async smoke state: " + tmp);
  std::error_code error;
  std::filesystem::rename(tmp, path, error);
  if (error) SpielFatalError("cannot publish async smoke state: " + error.message());
}

json::Object ToJson(const SmokeState& state) {
  json::Object object;
  object["schema_version"] = int64_t{1};
  object["next_update"] = static_cast<int64_t>(state.next_update);
  object["initial_generation"] =
      static_cast<int64_t>(state.initial_generation);
  object["run_first_episode_id"] = state.run_first_episode_id;
  object["target_episode_end"] = state.target_episode_end;
  object["first_episode_id"] = state.first_episode_id;
  object["next_episode_id"] = state.next_episode_id;
  object["published_generation"] =
      static_cast<int64_t>(state.published_generation);
  object["total_env_steps"] = static_cast<int64_t>(state.total_env_steps);
  json::Array replay_paths;
  for (const std::string& path : state.replay_paths) {
    replay_paths.push_back(path);
  }
  object["replay_paths"] = std::move(replay_paths);
  json::Array cohort_groups;
  for (const auto& group : state.replay_cohort_groups) {
    json::Array paths;
    for (const std::string& path : group) paths.push_back(path);
    cohort_groups.push_back(std::move(paths));
  }
  object["replay_cohort_groups"] = std::move(cohort_groups);
  json::Array ledger;
  for (const LedgerEntry& entry : state.completed) {
    json::Object item;
    item["episode_id"] = entry.episode_id;
    item["collector_generation"] =
        static_cast<int64_t>(entry.collector_generation);
    item["shard_path"] = entry.shard_path;
    item["rows"] = entry.rows;
    item["requested_simulations"] = entry.requested_simulations;
    item["completed_simulations"] = entry.completed_simulations;
    item["full_policy_target_rows"] = entry.full_policy_target_rows;
    item["trajectory_transitions"] = entry.trajectory_transitions;
    item["start_staleness"] = static_cast<int64_t>(entry.start_staleness);
    item["fallbacks"] = entry.fallbacks;
    item["watchdog_timeouts"] = entry.watchdog_timeouts;
    item["non_timeout_early_exits"] = entry.non_timeout_early_exits;
    ledger.push_back(std::move(item));
  }
  object["completed_ledger"] = std::move(ledger);
  return object;
}

SmokeState ReadState(const std::string& path) {
  std::ifstream input(path);
  if (!input) SpielFatalError("cannot read async smoke state: " + path);
  const std::string content((std::istreambuf_iterator<char>(input)),
                            std::istreambuf_iterator<char>());
  auto parsed = json::FromString(content);
  if (!parsed.has_value() || !parsed->IsObject()) {
    SpielFatalError("async smoke state is malformed: " + path);
  }
  const json::Object& object = parsed->GetObject();
  SmokeState state;
  state.next_update = static_cast<int>(object.at("next_update").GetInt());
  const auto first_episode = object.find("first_episode_id");
  state.first_episode_id = first_episode == object.end()
                               ? state.next_episode_id
                               : first_episode->second.GetInt();
  const auto initial_generation = object.find("initial_generation");
  state.initial_generation =
      initial_generation == object.end()
          ? state.published_generation
          : static_cast<int>(initial_generation->second.GetInt());
  const auto run_first_episode = object.find("run_first_episode_id");
  state.run_first_episode_id =
      run_first_episode == object.end()
          ? state.first_episode_id
          : run_first_episode->second.GetInt();
  const auto target_episode_end = object.find("target_episode_end");
  state.target_episode_end =
      target_episode_end == object.end() ? 0 : target_episode_end->second.GetInt();
  state.next_episode_id = object.at("next_episode_id").GetInt();
  state.published_generation =
      static_cast<int>(object.at("published_generation").GetInt());
  const auto total_env_steps = object.find("total_env_steps");
  state.total_env_steps = total_env_steps == object.end()
                              ? 0
                              : static_cast<uint64_t>(
                                    total_env_steps->second.GetInt());
  const auto replay_paths = object.find("replay_paths");
  if (replay_paths != object.end() && replay_paths->second.IsArray()) {
    for (const json::Value& value : replay_paths->second.GetArray()) {
      state.replay_paths.push_back(value.GetString());
    }
  }
  const auto cohort_groups = object.find("replay_cohort_groups");
  if (cohort_groups != object.end() && cohort_groups->second.IsArray()) {
    for (const json::Value& group_value : cohort_groups->second.GetArray()) {
      std::vector<std::string> group;
      for (const json::Value& value : group_value.GetArray()) {
        group.push_back(value.GetString());
      }
      state.replay_cohort_groups.push_back(std::move(group));
    }
  }
  for (const json::Value& value : object.at("completed_ledger").GetArray()) {
    const json::Object& item = value.GetObject();
    LedgerEntry entry;
    entry.episode_id = item.at("episode_id").GetInt();
    entry.collector_generation =
        static_cast<int>(item.at("collector_generation").GetInt());
    entry.shard_path = item.at("shard_path").GetString();
    entry.rows = item.at("rows").GetInt();
    const auto requested = item.find("requested_simulations");
    entry.requested_simulations =
        requested == item.end() ? item.at("completed_simulations").GetInt()
                                : requested->second.GetInt();
    entry.completed_simulations = item.at("completed_simulations").GetInt();
    const auto full_rows = item.find("full_policy_target_rows");
    entry.full_policy_target_rows =
        full_rows == item.end() ? 0 : full_rows->second.GetInt();
    const auto transitions = item.find("trajectory_transitions");
    entry.trajectory_transitions =
        transitions == item.end() ? 0 : transitions->second.GetInt();
    const auto start_staleness = item.find("start_staleness");
    entry.start_staleness = start_staleness == item.end()
                                ? 0
                                : static_cast<int>(
                                      start_staleness->second.GetInt());
    const auto fallbacks = item.find("fallbacks");
    entry.fallbacks = fallbacks == item.end() ? 0 : fallbacks->second.GetInt();
    const auto timeouts = item.find("watchdog_timeouts");
    entry.watchdog_timeouts =
        timeouts == item.end() ? 0 : timeouts->second.GetInt();
    const auto early_exits = item.find("non_timeout_early_exits");
    entry.non_timeout_early_exits =
        early_exits == item.end() ? 0 : early_exits->second.GetInt();
    state.completed.push_back(std::move(entry));
  }
  return state;
}

ProductionResumeState ReadProductionResumeState(const std::string& path) {
  std::ifstream input(path);
  if (!input) SpielFatalError("cannot read production checkpoint manifest: " + path);
  const std::string content((std::istreambuf_iterator<char>(input)),
                            std::istreambuf_iterator<char>());
  auto parsed = json::FromString(content);
  if (!parsed.has_value() || !parsed->IsObject()) {
    SpielFatalError("production checkpoint manifest is malformed: " + path);
  }
  const json::Object& object = parsed->GetObject();
  ProductionResumeState state;
  state.generation = static_cast<int>(object.at("generation").GetInt());
  state.next_episode_id = object.at("next_episode_id").GetInt();
  state.total_env_steps = static_cast<uint64_t>(
      object.at("total_env_steps").GetInt());
  state.model_checkpoint = object.at("model_checkpoint").GetString();
  state.optimizer_checkpoint = object.at("optimizer_checkpoint").GetString();
  // This is the continuation's first new episode, not the generation-zero
  // origin recorded in resolved_config.json.
  state.first_episode_id = state.next_episode_id;
  for (const json::Value& value : object.at("replay_paths").GetArray()) {
    state.replay_paths.push_back(value.GetString());
  }
  const auto cohort_groups = object.find("replay_cohort_groups");
  if (cohort_groups != object.end() && cohort_groups->second.IsArray()) {
    for (const json::Value& group_value : cohort_groups->second.GetArray()) {
      std::vector<std::string> group;
      for (const json::Value& value : group_value.GetArray()) {
        group.push_back(value.GetString());
      }
      state.replay_cohort_groups.push_back(std::move(group));
    }
  }
  if (state.replay_cohort_groups.empty()) {
    for (const std::string& replay_path : state.replay_paths) {
      state.replay_cohort_groups.push_back({replay_path});
    }
  }
  const size_t expected_replay_paths =
      static_cast<size_t>(std::min(state.generation, 8));
  const size_t logical_replay_cohorts = state.replay_cohort_groups.size();
  if (logical_replay_cohorts != expected_replay_paths) {
    SpielFatalError("production comparison logical replay horizon mismatch: " +
                    std::to_string(expected_replay_paths) + " expected, got " +
                    std::to_string(logical_replay_cohorts));
  }
  return state;
}

std::shared_ptr<CollectorSnapshot> MakeSnapshot(
    const std::shared_ptr<SharedDunePolicyValueNetImpl>& source,
    int generation, int64_t obs_size, int64_t action_dim, torch::Device device,
    int batch_target, int timeout_ms, int inference_lanes,
    uint64_t total_env_steps, uint64_t shaping_start_env_steps,
    uint64_t shaping_decay_env_steps) {
  auto snapshot = std::make_shared<CollectorSnapshot>();
  snapshot->generation = generation;
  snapshot->inference_lanes = inference_lanes;
  snapshot->total_env_steps = total_env_steps;
  snapshot->shaping_start_env_steps = shaping_start_env_steps;
  snapshot->shaping_decay_env_steps = shaping_decay_env_steps;
  snapshot->shaping_lambda = ComputeRewardLambda(
      total_env_steps, shaping_start_env_steps, shaping_decay_env_steps);
  SPIEL_CHECK_GT(inference_lanes, 0);
  snapshot->models.reserve(inference_lanes);
  snapshot->model_mutexes.reserve(inference_lanes);
  snapshot->batchers.reserve(inference_lanes);
  snapshot->evaluators.reserve(inference_lanes);
  for (int lane = 0; lane < inference_lanes; ++lane) {
    auto model = std::make_shared<SharedDunePolicyValueNetImpl>(
        obs_size, 2048, action_dim, 4, false);
    model->to(device);
    CopySearchPiModel(source, model);
    model->eval();
    auto model_mutex = std::make_shared<std::shared_mutex>();
    const float configured_cap =
        static_cast<float>(absl::GetFlag(FLAGS_logit_cap));
    auto batcher = std::make_shared<BatchedEvaluator>(
        model, batch_target, timeout_ms, device, model_mutex.get(),
        configured_cap, false, true);
    snapshot->models.push_back(std::move(model));
    snapshot->model_mutexes.push_back(std::move(model_mutex));
    snapshot->batchers.push_back(std::move(batcher));
    snapshot->evaluators.push_back(std::make_shared<BatchedNNEvaluator>(
        snapshot->batchers.back(), configured_cap));
  }
  snapshot->cap_baselines.resize(snapshot->batchers.size());
  return snapshot;
}

SearchPiConfig MakeSearchConfig(int64_t episode_id, int64_t obs_size,
                                uint64_t seed, float shaping_lambda,
                                uint64_t shaping_start_env_steps,
                                uint64_t shaping_decay_env_steps) {
  (void)obs_size;
  SearchPiConfig config;
  config.scratch_visit_v1 = true;
  config.scratch_q_v1 = false;
  config.games_per_generation = 1;
  config.next_episode_id = episode_id;
  config.seed_domain = seed;
  config.scratch_full_root_probability =
      ::absl::GetFlag(FLAGS_full_root_probability);
  config.scratch_full_primary_simulations =
      ::absl::GetFlag(FLAGS_full_primary_budget);
  config.scratch_full_other_simulations =
      ::absl::GetFlag(FLAGS_full_other_budget);
  config.scratch_cheap_primary_simulations =
      ::absl::GetFlag(FLAGS_cheap_primary_budget);
  config.scratch_cheap_other_simulations =
      ::absl::GetFlag(FLAGS_cheap_other_budget);
  config.behavior_temperature = 1.0;
  config.non_search_temperature = 1.0;
  config.searched_seat_unsearched_temperature = 1.0;
  config.dirichlet_epsilon = 0.0;
  config.forced_playouts_k = 0.0;
  config.root_noise_fpu_zero = false;
  config.relative_time_budget_ms =
      ::absl::GetFlag(FLAGS_relative_time_budget_ms);
  config.shaping_lambda = shaping_lambda;
  config.shaping_start_env_steps = shaping_start_env_steps;
  config.shaping_decay_env_steps = shaping_decay_env_steps;
  config.primary_simulations = ::absl::GetFlag(FLAGS_full_primary_budget);
  config.continuation_simulations = ::absl::GetFlag(FLAGS_full_other_budget);
  config.purchase_combat_budget = ::absl::GetFlag(FLAGS_full_other_budget);
  return config;
}

}  // namespace
}  // namespace open_spiel

namespace open_spiel {
namespace {

class TaskQueue {
 public:
  void Push(Task task) {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (closed_) return;
      queue_.push(std::move(task));
    }
    cv_.notify_one();
  }

  bool Pop(Task* task) {
    std::unique_lock<std::mutex> lock(mutex_);
    cv_.wait(lock, [&] { return closed_ || !queue_.empty(); });
    if (queue_.empty()) return false;
    *task = std::move(queue_.front());
    queue_.pop();
    return true;
  }

  void Close() {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      closed_ = true;
    }
    cv_.notify_all();
  }

  size_t Size() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return queue_.size();
  }

 private:
  mutable std::mutex mutex_;
  std::condition_variable cv_;
  std::queue<Task> queue_;
  bool closed_ = false;
};

void SaveStudentCheckpoint(const std::filesystem::path& output_dir,
                           int update,
                           const std::shared_ptr<SharedDunePolicyValueNetImpl>& model,
                           torch::optim::Optimizer& optimizer) {
  torch::save(model, (output_dir / ("student_update_" +
                                   std::to_string(update) + ".pt"))
                          .string());
  torch::save(optimizer, (output_dir / ("optimizer_update_" +
                                      std::to_string(update) + ".pt"))
                             .string());
}

void SaveProductionCheckpoint(
    const std::filesystem::path& run_dir, int generation,
    const SmokeState& state,
    const std::shared_ptr<SharedDunePolicyValueNetImpl>& model,
    torch::optim::Optimizer& optimizer) {
  const std::filesystem::path checkpoint_dir = run_dir / "checkpoints";
  std::filesystem::create_directories(checkpoint_dir);
  const std::string stem = "generation_" + std::to_string(generation);
  const std::filesystem::path model_path = checkpoint_dir / (stem + "_model.pt");
  const std::filesystem::path optimizer_path =
      checkpoint_dir / (stem + "_optimizer.pt");
  const std::filesystem::path model_tmp = model_path.string() + ".tmp";
  const std::filesystem::path optimizer_tmp = optimizer_path.string() + ".tmp";
  torch::save(model, model_tmp.string());
  torch::save(optimizer, optimizer_tmp.string());
  std::error_code error;
  std::filesystem::rename(model_tmp, model_path, error);
  if (error) SpielFatalError("cannot publish async model checkpoint: " +
                             error.message());
  std::filesystem::rename(optimizer_tmp, optimizer_path, error);
  if (error) SpielFatalError("cannot publish async optimizer checkpoint: " +
                             error.message());

  json::Object checkpoint;
  checkpoint["schema_version"] = int64_t{1};
  checkpoint["profile"] = "scratch_visit_v1";
  checkpoint["generation"] = static_cast<int64_t>(generation);
  checkpoint["next_episode_id"] = state.next_episode_id;
  checkpoint["total_env_steps"] =
      static_cast<int64_t>(state.total_env_steps);
  checkpoint["model_checkpoint"] = model_path.string();
  checkpoint["optimizer_checkpoint"] = optimizer_path.string();
  json::Array paths;
  json::Array cohort_groups;
  for (const auto& group : state.replay_cohort_groups) {
    json::Array group_paths;
    for (const std::string& path : group) {
      paths.push_back(path);
      group_paths.push_back(path);
    }
    cohort_groups.push_back(std::move(group_paths));
  }
  if (state.replay_cohort_groups.empty()) {
    for (const std::string& path : state.replay_paths) paths.push_back(path);
  }
  checkpoint["replay_paths"] = std::move(paths);
  checkpoint["replay_cohort_groups"] = std::move(cohort_groups);
  checkpoint["pending_prefetch_games"] = int64_t{0};
  checkpoint["pending_prefetch_collector_generation"] = int64_t{-1};
  checkpoint["pending_prefetch_env_steps"] = int64_t{0};
  AtomicJson((checkpoint_dir / (stem + ".json")).string(), checkpoint);
  AtomicJson((checkpoint_dir / "checkpoint_latest.json").string(), checkpoint);
  AtomicJson((run_dir / "checkpoint_latest.json").string(), checkpoint);
}

std::unique_ptr<torch::optim::AdamW> MakeSmokeOptimizer(
    const std::shared_ptr<SharedDunePolicyValueNetImpl>& model) {
  std::vector<torch::Tensor> policy;
  std::vector<torch::Tensor> other;
  const auto policy_parameters = model->policy_head->parameters();
  for (const torch::Tensor& parameter : model->parameters()) {
    bool is_policy = false;
    for (const torch::Tensor& candidate : policy_parameters) {
      if (parameter.is_same(candidate)) is_policy = true;
    }
    (is_policy ? policy : other).push_back(parameter);
  }
  std::vector<torch::optim::OptimizerParamGroup> groups;
  groups.emplace_back(policy);
  groups.emplace_back(other);
  return std::make_unique<torch::optim::AdamW>(
      groups, torch::optim::AdamWOptions(1e-4).eps(1e-5));
}

void CheckCompletedGame(const Completion& completion) {
  SPIEL_CHECK_EQ(completion.rows.size(),
                 static_cast<size_t>(completion.outcome.rows_emitted));
  SPIEL_CHECK_EQ(completion.outcome.fallbacks, 0);
  SPIEL_CHECK_EQ(completion.outcome.watchdog_timeouts, 0);
  SPIEL_CHECK_EQ(completion.outcome.non_timeout_early_exits, 0);
  for (const SearchPiRow& row : completion.rows) {
    SPIEL_CHECK_EQ(row.simulations_completed, row.simulations_requested);
  }
}

}  // namespace
}  // namespace open_spiel

int main(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);
  using namespace open_spiel;
  const std::string output = absl::GetFlag(FLAGS_output_dir);
  if (output.empty()) SpielFatalError("--output_dir is required");
  const std::filesystem::path output_dir = output;
  std::filesystem::create_directories(output_dir);
  const std::filesystem::path state_path = output_dir / "async_state.json";
  const int updates = absl::GetFlag(FLAGS_updates);
  const int requested_target_generation =
      absl::GetFlag(FLAGS_target_generation);
  const int games_per_update = absl::GetFlag(FLAGS_games_per_update);
  const int actor_count = absl::GetFlag(FLAGS_actors);
  const int inference_lanes = absl::GetFlag(FLAGS_inference_lanes);
  SPIEL_CHECK_GT(updates, 0);
  SPIEL_CHECK_GT(games_per_update, 0);
  SPIEL_CHECK_GT(actor_count, 0);
  SPIEL_CHECK_GT(inference_lanes, 0);
  SPIEL_CHECK_EQ(actor_count % inference_lanes, 0);

  auto game = LoadGame("dune_imperium");
  const int64_t obs_size = game->InformationStateTensorSize();
  const int64_t action_dim = game->NumDistinctActions();
  const torch::Device device(torch::cuda::is_available() ? torch::kCUDA
                                                          : torch::kCPU);
  auto student = std::make_shared<SharedDunePolicyValueNetImpl>(
      obs_size, 2048, action_dim, 4, false);
  student->to(device);
  SearchPiLearnerConfig learner_config;
  learner_config.learning_rate = 1e-4;
  learner_config.minibatch_size = ::absl::GetFlag(FLAGS_minibatch_size);
  learner_config.epochs = 1;
  learner_config.grad_clip_norm = 0.5;
  learner_config.logit_cap = absl::GetFlag(FLAGS_logit_cap);
  learner_config.policy_coef = 1.0;
  learner_config.value_coef = 1.0;
  auto optimizer = MakeSmokeOptimizer(student);

  SmokeState state;
  const std::string resume = ::absl::GetFlag(FLAGS_resume);
  const std::string production_manifest =
      ::absl::GetFlag(FLAGS_production_checkpoint_manifest);
  const std::filesystem::path production_run_dir =
      !::absl::GetFlag(FLAGS_production_run_dir).empty()
          ? std::filesystem::path(::absl::GetFlag(FLAGS_production_run_dir))
          : production_manifest.empty()
                ? std::filesystem::path()
                : std::filesystem::path(production_manifest)
                      .parent_path().parent_path();
  const uint64_t shaping_start_env_steps =
      ::absl::GetFlag(FLAGS_shaping_start_env_steps);
  const uint64_t shaping_decay_env_steps =
      ::absl::GetFlag(FLAGS_shaping_decay_env_steps);
  ProductionResumeState production_state;
  if (!production_manifest.empty()) {
    production_state = ReadProductionResumeState(production_manifest);
  }
  if (!resume.empty()) {
    state = ReadState(resume);
    if (state.next_update > 0) {
      torch::load(student,
                  (output_dir / ("student_update_" +
                                 std::to_string(state.next_update) + ".pt"))
                      .string(),
                  device);
      torch::load(*optimizer,
                  (output_dir / ("optimizer_update_" +
                                 std::to_string(state.next_update) + ".pt"))
                      .string(),
                  device);
    }
  } else if (!production_manifest.empty()) {
    torch::load(student, production_state.model_checkpoint, device);
    torch::load(*optimizer, production_state.optimizer_checkpoint, device);
    state.first_episode_id = production_state.first_episode_id;
    state.run_first_episode_id = production_state.first_episode_id;
    state.next_episode_id = production_state.next_episode_id;
    state.published_generation = production_state.generation;
    state.initial_generation = production_state.generation;
    state.total_env_steps = production_state.total_env_steps;
    state.replay_paths = production_state.replay_paths;
    state.replay_cohort_groups = production_state.replay_cohort_groups;
  } else {
    state.first_episode_id = ::absl::GetFlag(FLAGS_first_episode_id);
    state.run_first_episode_id = state.first_episode_id;
    state.next_episode_id = ::absl::GetFlag(FLAGS_first_episode_id);
    state.initial_generation = state.published_generation;
    AtomicJson(state_path.string(), ToJson(state));
  }

  std::vector<std::vector<SearchPiRow>> replay_window;
  std::map<int64_t, LedgerEntry> ledger;
  if (state.replay_cohort_groups.empty()) {
    for (const std::string& replay_path : state.replay_paths) {
      state.replay_cohort_groups.push_back({replay_path});
    }
  }
  for (const auto& group : state.replay_cohort_groups) {
    std::vector<SearchPiRow> cohort_rows;
    for (const std::string& replay_path : group) {
      std::vector<SearchPiRow> rows;
      std::string error;
      if (!ReadScratchSearchPiRowShardV2(replay_path, &rows, &error)) {
        SpielFatalError("async production replay load failed: " + error);
      }
      cohort_rows.insert(cohort_rows.end(), rows.begin(), rows.end());
    }
    replay_window.push_back(std::move(cohort_rows));
  }
  for (const LedgerEntry& entry : state.completed) {
    if (!ledger.emplace(entry.episode_id, entry).second) {
      SpielFatalError("duplicate completed episode in ledger");
    }
  }
  if (::absl::GetFlag(FLAGS_verify_production_manifest_only)) {
    if (production_manifest.empty()) {
      SpielFatalError("verify_production_manifest_only requires a manifest.");
    }
    json::Object verification;
    verification["status"] = "PASS";
    verification["generation"] =
        static_cast<int64_t>(state.published_generation);
    verification["first_episode_id"] = state.first_episode_id;
    verification["next_episode_id"] = state.next_episode_id;
    verification["total_env_steps"] =
        static_cast<int64_t>(state.total_env_steps);
    verification["replay_cohorts"] =
        static_cast<int64_t>(replay_window.size());
    if (::absl::GetFlag(FLAGS_force_health_failure_for_test)) {
      return 2;
    }
    std::cout << json::ToString(verification, true) << "\n";
    return 0;
  }

  auto latest_snapshot = MakeSnapshot(
      student, state.published_generation, obs_size, action_dim, device,
      ::absl::GetFlag(FLAGS_batch_target),
      ::absl::GetFlag(FLAGS_batcher_timeout_ms), inference_lanes,
      state.total_env_steps, shaping_start_env_steps,
      shaping_decay_env_steps);
  std::vector<std::shared_ptr<CollectorSnapshot>> collector_snapshots = {
      latest_snapshot};
  std::mutex snapshot_mutex;
  TaskQueue tasks;
  std::mutex completions_mutex;
  std::condition_variable completions_cv;
  std::queue<Completion> completions;
  std::atomic<bool> stop_workers{false};
  std::atomic<int> published_generation_atomic{state.published_generation};
  std::atomic<int> active_tasks{0};
  std::atomic<int64_t> last_actor_completion_ns{0};

  auto publish_state = [&]() { AtomicJson(state_path.string(), ToJson(state)); };
  const int target_generation = requested_target_generation >= 0
                                    ? requested_target_generation
                                    : state.published_generation + updates;
  SPIEL_CHECK_GT(target_generation, state.published_generation);
  if (state.initial_generation == 0 && state.published_generation != 0) {
    state.initial_generation = state.published_generation;
  }
  if (state.run_first_episode_id == 0) {
    state.run_first_episode_id = state.first_episode_id;
  }
  if (state.target_episode_end == 0) {
    state.target_episode_end =
        state.run_first_episode_id +
        static_cast<int64_t>(target_generation - state.initial_generation) *
            games_per_update;
  }
  const int64_t target_end = state.target_episode_end;
  const int total_games = static_cast<int>(
      target_end - state.run_first_episode_id);
  SPIEL_CHECK_GT(total_games, 0);
  const auto started = std::chrono::steady_clock::now();
  const std::set<int64_t> completed_at_resume = [&]() {
    std::set<int64_t> ids;
    for (const auto& item : ledger) ids.insert(item.first);
    return ids;
  }();
  std::atomic<int64_t> next_episode_to_issue{state.first_episode_id};

  auto make_task = [&](int64_t episode_id) {
    Task task;
    task.episode_id = episode_id;
    {
      std::lock_guard<std::mutex> lock(snapshot_mutex);
      task.snapshot = latest_snapshot;
    }
    return task;
  };

  auto reserve_next_episode = [&]() -> std::optional<int64_t> {
    int64_t candidate = next_episode_to_issue.load(std::memory_order_relaxed);
    while (candidate < target_end) {
      if (completed_at_resume.contains(candidate)) {
        next_episode_to_issue.compare_exchange_weak(
            candidate, candidate + 1, std::memory_order_relaxed,
            std::memory_order_relaxed);
        continue;
      }
      if (next_episode_to_issue.compare_exchange_weak(
              candidate, candidate + 1, std::memory_order_relaxed,
              std::memory_order_relaxed)) {
        return candidate;
      }
    }
    return std::nullopt;
  };
  auto issue_one = [&]() {
    const std::optional<int64_t> episode_id = reserve_next_episode();
    if (!episode_id.has_value()) return false;
    tasks.Push(make_task(*episode_id));
    return true;
  };

  auto worker = [&](int actor_id) {
    torch::InferenceMode inference_guard;
    Task task;
    while (!stop_workers.load(std::memory_order_relaxed) && tasks.Pop(&task)) {
      active_tasks.fetch_add(1, std::memory_order_relaxed);
      const int lane = actor_id % task.snapshot->inference_lanes;
      const int start_staleness = std::max(
          0, published_generation_atomic.load(std::memory_order_relaxed) -
                 task.snapshot->generation);
      SearchPiConfig config = MakeSearchConfig(
          task.episode_id, obs_size, ::absl::GetFlag(FLAGS_seed),
          task.snapshot->shaping_lambda,
          task.snapshot->shaping_start_env_steps,
          task.snapshot->shaping_decay_env_steps);
      std::vector<SearchPiRow> rows;
      SearchPiGenerationStats stats;
      SearchPiGenerator(config).GenerateGeneration(
          task.snapshot->generation, game, task.snapshot->evaluators[lane],
          task.snapshot->evaluators[lane], &rows, &stats,
          SearchPiArm::kSearched);
      ScratchVisitGenerationStats visit_stats;
      std::string target_error;
      if (!ApplyScratchVisitTargets(&rows, &visit_stats, &target_error)) {
        SpielFatalError("async smoke target construction failed: " +
                        target_error);
      }
      if (stats.games_played.size() != 1) {
        SpielFatalError("async smoke actor did not produce one game");
      }
      Completion completion;
      completion.episode_id = task.episode_id;
      completion.collector_generation = task.snapshot->generation;
      completion.start_staleness = start_staleness;
      completion.actor_completion_wall_seconds =
          std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                        started)
              .count();
      completion.rows = std::move(rows);
      completion.visit_stats = visit_stats;
      completion.outcome = stats.games_played.front();
      CheckCompletedGame(completion);
      const int64_t completion_ns =
          static_cast<int64_t>(completion.actor_completion_wall_seconds * 1e9);
      int64_t observed_ns =
          last_actor_completion_ns.load(std::memory_order_relaxed);
      while (completion_ns > observed_ns &&
             !last_actor_completion_ns.compare_exchange_weak(
                 observed_ns, completion_ns, std::memory_order_relaxed,
                 std::memory_order_relaxed)) {}
      {
        std::lock_guard<std::mutex> lock(completions_mutex);
        completions.push(std::move(completion));
      }
      active_tasks.fetch_sub(1, std::memory_order_relaxed);
      issue_one();
      completions_cv.notify_one();
    }
  };

  std::vector<std::thread> workers;
  workers.reserve(actor_count);
  for (int i = 0; i < actor_count; ++i) workers.emplace_back(worker, i);
  for (int i = 0; i < actor_count && issue_one(); ++i) {}
  state.next_episode_id = next_episode_to_issue.load(std::memory_order_relaxed);
  publish_state();

  std::unique_ptr<c10::cuda::CUDAStreamGuard> learner_guard;
  std::unique_ptr<c10::cuda::CUDAStream> learner_stream;
  if (device.is_cuda()) {
    const c10::DeviceIndex device_index =
        device.has_index() ? device.index() : c10::cuda::current_device();
    learner_stream = std::make_unique<c10::cuda::CUDAStream>(
        c10::cuda::getStreamFromPool(false, device_index));
  }
  struct LearnerInterval {
    int update = 0;
    double start_wall_seconds = 0.0;
    double end_wall_seconds = 0.0;
  };
  std::vector<LearnerInterval> learner_intervals;
  std::vector<json::Object> update_records;
  std::vector<int64_t> update_new_ids;
  std::vector<SearchPiRow> current_update_rows;
  std::vector<std::string> current_update_paths;
  UpdateCollectionTelemetry current_update_collection;
  int64_t current_update_env_steps = 0;
  int64_t total_learner_minibatches = 0;
  bool simulated_crash = false;
  while (state.published_generation < target_generation) {
    Completion completion;
    {
      std::unique_lock<std::mutex> lock(completions_mutex);
      completions_cv.wait(lock, [&] { return !completions.empty(); });
      completion = std::move(completions.front());
      completions.pop();
    }
    if (ledger.contains(completion.episode_id)) {
      SpielFatalError("duplicate completed episode observed");
    }
    const std::filesystem::path shard_path =
        output_dir / ("game_" + std::to_string(completion.episode_id) +
                      ".bin");
    std::string shard_error;
    if (!WriteScratchSearchPiRowShardV2(
            shard_path.string(), completion.rows, &shard_error)) {
      SpielFatalError("async smoke game shard write failed: " + shard_error);
    }
    LedgerEntry entry;
    entry.episode_id = completion.episode_id;
    entry.collector_generation = completion.collector_generation;
    entry.start_staleness = completion.start_staleness;
    entry.shard_path = shard_path.string();
    entry.rows = static_cast<int64_t>(completion.rows.size());
    entry.completed_simulations = 0;
    for (const SearchPiRow& row : completion.rows) {
      entry.requested_simulations += row.simulations_requested;
      entry.completed_simulations += row.simulations_completed;
      if (row.search_budget_class == "full" &&
          row.policy_target_weight > 0.0) {
        ++entry.full_policy_target_rows;
      }
    }
    entry.fallbacks = completion.outcome.fallbacks;
    entry.watchdog_timeouts = completion.outcome.watchdog_timeouts;
    entry.non_timeout_early_exits = completion.outcome.non_timeout_early_exits;
    current_update_collection.Add(
        completion, completion.visit_stats, entry,
        state.published_generation + 1);
    ledger.emplace(entry.episode_id, entry);
    state.completed.push_back(entry);
    current_update_rows.insert(current_update_rows.end(),
                               completion.rows.begin(), completion.rows.end());
    current_update_paths.push_back(shard_path.string());
    current_update_env_steps += completion.outcome.trajectory_transitions;
    update_new_ids.push_back(completion.episode_id);
    state.next_episode_id =
        next_episode_to_issue.load(std::memory_order_relaxed);
    publish_state();

    const int update_target = (state.next_update + 1) * games_per_update;
    if (static_cast<int>(state.completed.size()) < update_target) continue;
    const double boundary_wall_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                      started)
            .count();
    const int unfinished_or_queued = std::max(
        0, total_games - static_cast<int>(state.completed.size()));
    const auto learner_start = std::chrono::steady_clock::now();
    const double learner_start_wall_seconds =
        std::chrono::duration<double>(learner_start - started).count();
    if (device.is_cuda()) {
      learner_guard = std::make_unique<c10::cuda::CUDAStreamGuard>(
          *learner_stream);
    }
    replay_window.push_back(std::move(current_update_rows));
    while (replay_window.size() > 8) replay_window.erase(replay_window.begin());
    state.replay_cohort_groups.push_back(std::move(current_update_paths));
    while (state.replay_cohort_groups.size() > 8) {
      state.replay_cohort_groups.erase(state.replay_cohort_groups.begin());
    }
    state.replay_paths.clear();
    for (const auto& group : state.replay_cohort_groups) {
      state.replay_paths.insert(state.replay_paths.end(), group.begin(),
                                group.end());
    }
    std::string replay_error;
    const int learner_generation = state.published_generation + 1;
    std::vector<SearchPiRow> learner_rows = SampleScratchSearchPiReplayWindow(
        replay_window, learner_generation, ::absl::GetFlag(FLAGS_replay_max_rows),
        dune_seed::DeriveSeed(::absl::GetFlag(FLAGS_seed), learner_generation),
        &replay_error);
    if (!replay_error.empty()) {
      SpielFatalError("async production replay sampling failed: " + replay_error);
    }
    double target_entropy_sum = 0.0;
    int64_t target_entropy_count = 0;
    for (const SearchPiRow& row : learner_rows) {
      target_entropy_sum += row.target_entropy_norm;
      ++target_entropy_count;
    }
    state.total_env_steps += static_cast<uint64_t>(current_update_env_steps);
    ScratchSearchPiLearnerStats learner_stats = RunScratchSearchPiLearner(
        student, *optimizer, learner_rows, obs_size, action_dim, device,
        ::absl::GetFlag(FLAGS_seed), learner_generation, learner_config);
    learner_guard.reset();
    const double learner_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - learner_start).count();
    const double learner_end_wall_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                      started)
            .count();
    learner_intervals.push_back({state.next_update + 1,
                                 learner_start_wall_seconds,
                                 learner_end_wall_seconds});
    total_learner_minibatches += learner_stats.minibatches;

    ++state.published_generation;
    published_generation_atomic.store(state.published_generation,
                                      std::memory_order_relaxed);
    {
      std::lock_guard<std::mutex> lock(snapshot_mutex);
      latest_snapshot = MakeSnapshot(
          student, state.published_generation, obs_size, action_dim, device,
          ::absl::GetFlag(FLAGS_batch_target),
          ::absl::GetFlag(FLAGS_batcher_timeout_ms), inference_lanes,
          state.total_env_steps, shaping_start_env_steps,
          shaping_decay_env_steps);
      collector_snapshots.push_back(latest_snapshot);
    }
    BatchedLogitCapAggregate collector_cap_interval;
    for (const auto& snapshot : collector_snapshots) {
      const BatchedLogitCapAggregate interval =
          snapshot->ConsumeCapTelemetry();
      collector_cap_interval.decisions += interval.decisions;
      collector_cap_interval.legal_logits += interval.legal_logits;
      collector_cap_interval.saturated += interval.saturated;
      collector_cap_interval.pre_max_abs =
          std::max(collector_cap_interval.pre_max_abs, interval.pre_max_abs);
      collector_cap_interval.post_max_abs =
          std::max(collector_cap_interval.post_max_abs, interval.post_max_abs);
    }
    SaveStudentCheckpoint(output_dir, state.next_update + 1, student,
                          *optimizer);
    json::Object update;
    update["update"] = static_cast<int64_t>(state.next_update + 1);
    update["completed_games"] =
        static_cast<int64_t>(state.completed.size());
    update["new_games_this_update"] =
        static_cast<int64_t>(update_new_ids.size());
    update["replay_rows"] = static_cast<int64_t>(learner_rows.size());
    update["learner_seconds"] = learner_seconds;
    update["boundary_wall_seconds"] = boundary_wall_seconds;
    update["learner_start_wall_seconds"] = learner_start_wall_seconds;
    update["learner_end_wall_seconds"] = learner_end_wall_seconds;
    update["unfinished_or_queued_games_at_boundary"] =
        static_cast<int64_t>(unfinished_or_queued);
    update["active_actor_tasks_at_boundary"] =
        static_cast<int64_t>(active_tasks.load(std::memory_order_relaxed));
    update["queued_tasks_at_boundary"] =
        static_cast<int64_t>(tasks.Size());
    update["inference_lanes"] = static_cast<int64_t>(inference_lanes);
    update["batch_target_per_lane"] =
        static_cast<int64_t>(::absl::GetFlag(FLAGS_batch_target));
    update["learner_minibatches"] =
        static_cast<int64_t>(learner_stats.minibatches);
    update["learner_streamed"] = true;
    update["learner_value_prediction_mean_pre"] =
        learner_stats.critic_pred_mean_pre;
    update["learner_value_prediction_mean_post"] =
        learner_stats.critic_pred_mean_post;
    update["requested_simulations"] =
        current_update_collection.requested_simulations;
    update["completed_simulations"] =
        current_update_collection.completed_simulations;
    update["fallbacks"] = current_update_collection.fallbacks;
    update["watchdog_timeouts"] = current_update_collection.watchdog_timeouts;
    update["non_timeout_early_exits"] =
        current_update_collection.non_timeout_early_exits;
    update["generation_env_steps"] =
        current_update_collection.trajectory_transitions;
    update["total_env_steps"] = static_cast<int64_t>(state.total_env_steps);
    update["shaping_lambda"] =
        static_cast<double>(latest_snapshot->shaping_lambda);
    update["vp_breadcrumb_events"] =
        current_update_collection.vp_breadcrumb_events;
    update["specimen_anti_breadcrumb_events"] =
        current_update_collection.specimen_anti_breadcrumb_events;
    update["vp_breadcrumb_raw_reward_sum"] =
        current_update_collection.vp_breadcrumb_raw_reward_sum;
    update["specimen_anti_breadcrumb_raw_penalty_sum"] =
        current_update_collection.specimen_anti_breadcrumb_raw_penalty_sum;
    update["vp_breadcrumb_shaped_reward_sum"] =
        current_update_collection.vp_breadcrumb_shaped_reward_sum;
    update["specimen_anti_breadcrumb_shaped_reward_sum"] =
        current_update_collection.specimen_anti_breadcrumb_shaped_reward_sum;
    update["collector_generation_min"] =
        static_cast<int64_t>(current_update_collection.collector_generation_min);
    update["collector_generation_max"] =
        static_cast<int64_t>(current_update_collection.collector_generation_max);
    update["collector_generation_mean"] =
        current_update_collection.collector_generation_sum /
        std::max<size_t>(1, update_new_ids.size());
    update["collector_start_staleness_mean"] =
        current_update_collection.collector_start_staleness_sum /
        std::max<size_t>(1, update_new_ids.size());
    update["collector_start_staleness_max"] =
        static_cast<int64_t>(current_update_collection.collector_start_staleness_max);
    update["collector_staleness_mean"] =
        current_update_collection.collector_staleness_sum /
        std::max<size_t>(1, update_new_ids.size());
    update["collector_staleness_max"] =
        static_cast<int64_t>(current_update_collection.collector_staleness_max);
    update["collector_staleness_definition"] =
        "learner_generation_minus_collector_generation_for_this_update_games";
    update["collector_logit_cap_scope"] =
        "counter_delta_since_previous_generation_telemetry;maxima_lifetime_per_snapshot";
    update["collector_precap_max_legal_logit"] =
        collector_cap_interval.pre_max_abs;
    update["collector_postcap_max_legal_logit"] =
        collector_cap_interval.post_max_abs;
    update["collector_logit_cap_decisions"] =
        static_cast<int64_t>(collector_cap_interval.decisions);
    update["collector_logit_cap_legal_logits"] =
        static_cast<int64_t>(collector_cap_interval.legal_logits);
    update["collector_logit_cap_saturated"] =
        static_cast<int64_t>(collector_cap_interval.saturated);
    update["collector_logit_cap_saturation_rate"] =
        collector_cap_interval.legal_logits == 0
            ? 0.0
            : static_cast<double>(collector_cap_interval.saturated) /
                  static_cast<double>(collector_cap_interval.legal_logits);
    update["learner_precap_max_legal_logit"] =
        learner_stats.learner_precap_max_legal_logit;
    update["learner_postcap_max_legal_logit"] =
        learner_stats.learner_postcap_max_legal_logit;
    update["learner_logit_cap_decisions"] =
        learner_stats.learner_logit_cap_decisions;
    update["learner_logit_cap_legal_logits"] =
        learner_stats.learner_logit_cap_legal_logits;
    update["learner_logit_cap_saturated"] =
        learner_stats.learner_logit_cap_saturated;
    update["learner_logit_cap_saturation_rate"] =
        learner_stats.learner_logit_cap_legal_logits == 0
            ? 0.0
            : static_cast<double>(learner_stats.learner_logit_cap_saturated) /
                  static_cast<double>(learner_stats.learner_logit_cap_legal_logits);
    const std::set<int64_t> unique_update_ids(update_new_ids.begin(),
                                               update_new_ids.end());
    // Actors continuously issue the next available IDs while the learner is
    // processing a boundary. Consequently, the 512 completions in one
    // learner update are not required to form a contiguous episode interval:
    // a later-issued game may finish before an earlier-issued game. The global
    // summary checks the complete run range; this update-level contract counts
    // completion slots and separately records the observed ID span.
    const int64_t expected_update_games =
        static_cast<int64_t>(update_new_ids.size());
    const int64_t missing_update_ids = std::max<int64_t>(
        0, expected_update_games -
               static_cast<int64_t>(unique_update_ids.size()));
    const int64_t observed_first_episode_id =
        unique_update_ids.empty()
            ? -1
            : *unique_update_ids.begin();
    const int64_t observed_episode_end =
        unique_update_ids.empty() ? -1 : *unique_update_ids.rbegin() + 1;
    update["episode_accounting_scope"] =
        "completed_update_slots;continuous_issue_order_is_not_contiguous";
    update["observed_first_episode_id"] = observed_first_episode_id;
    update["observed_episode_end"] = observed_episode_end;
    update["expected_games_this_update"] = expected_update_games;
    update["nominal_games_per_update"] =
        static_cast<int64_t>(games_per_update);
    update["unique_episode_ids"] =
        static_cast<int64_t>(unique_update_ids.size());
    update["missing_episode_ids"] = missing_update_ids;
    update["duplicate_episode_ids"] =
        static_cast<int64_t>(update_new_ids.size() - unique_update_ids.size());
    update["episode_accounting_complete"] =
        unique_update_ids.size() == static_cast<size_t>(expected_update_games) &&
        missing_update_ids == 0 &&
        update_new_ids.size() == unique_update_ids.size();
    int64_t total_simulations = 0;
    int64_t cumulative_requested = 0;
    int64_t full_policy_target_rows = 0;
    int64_t cumulative_fallbacks = 0;
    int64_t cumulative_timeouts = 0;
    int64_t cumulative_early_exits = 0;
    double staleness_sum = 0.0;
    int max_staleness = 0;
    for (const LedgerEntry& item : state.completed) {
      cumulative_requested += item.requested_simulations;
      total_simulations += item.completed_simulations;
      cumulative_fallbacks += item.fallbacks;
      cumulative_timeouts += item.watchdog_timeouts;
      cumulative_early_exits += item.non_timeout_early_exits;
      const int staleness =
          state.published_generation - item.collector_generation;
      staleness_sum += staleness;
      max_staleness = std::max(max_staleness, staleness);
    }
    for (const LedgerEntry& entry : state.completed) {
      full_policy_target_rows += entry.full_policy_target_rows;
    }
    update["full_policy_target_rows"] =
        current_update_collection.full_policy_target_rows;
    update["cumulative_requested_simulations"] = cumulative_requested;
    update["cumulative_completed_simulations"] = total_simulations;
    update["cumulative_full_policy_target_rows"] = full_policy_target_rows;
    update["cumulative_fallbacks"] = cumulative_fallbacks;
    update["cumulative_watchdog_timeouts"] = cumulative_timeouts;
    update["cumulative_non_timeout_early_exits"] = cumulative_early_exits;
    update["cumulative_collector_staleness_mean"] = state.completed.empty()
                                                         ? 0.0
                                                         : staleness_sum /
                                                               state.completed.size();
    update["cumulative_collector_staleness_max"] =
        static_cast<int64_t>(max_staleness);
    update_records.push_back(update);
    AtomicJson((output_dir / ("update_" +
                              std::to_string(state.next_update + 1) +
                              ".json"))
                   .string(),
               update);
    if (!production_run_dir.empty()) {
      json::Object generation_telemetry = update;
      generation_telemetry["generation"] =
          static_cast<int64_t>(state.published_generation);
      generation_telemetry["games"] =
          static_cast<int64_t>(update_new_ids.size());
      generation_telemetry["policy_ce"] = learner_stats.policy_ce;
      generation_telemetry["value_mse"] = learner_stats.value_mse;
      generation_telemetry["policy_grad_norm"] =
          learner_stats.policy_grad_norm;
      generation_telemetry["value_grad_norm"] =
          learner_stats.value_grad_norm;
      generation_telemetry["value_prediction_mean"] =
          learner_stats.critic_pred_mean_post;
      generation_telemetry["value_prediction_sd"] =
          learner_stats.critic_pred_sd_post;
      generation_telemetry["value_saturation_rate"] =
          learner_stats.critic_saturation_post;
      generation_telemetry["visit_entropy_mean"] =
          target_entropy_count == 0
              ? 0.0
              : target_entropy_sum / target_entropy_count;
      generation_telemetry["learner_streamed_minibatches"] = true;
      generation_telemetry["inference_lanes"] =
          static_cast<int64_t>(inference_lanes);
      generation_telemetry["batch_target_per_lane"] =
          static_cast<int64_t>(::absl::GetFlag(FLAGS_batch_target));
      AtomicJson((production_run_dir /
                  ("generation_" +
                   std::to_string(state.published_generation) + ".json"))
                     .string(),
                 generation_telemetry);
      if (::absl::GetFlag(FLAGS_checkpoint_every_update) ||
          state.published_generation % 5 == 0) {
        SaveProductionCheckpoint(production_run_dir,
                                 state.published_generation, state, student,
                                 *optimizer);
      }
    }
    update_new_ids.clear();
    current_update_paths.clear();
    current_update_collection = UpdateCollectionTelemetry();
    current_update_env_steps = 0;
    ++state.next_update;
    state.next_episode_id =
        next_episode_to_issue.load(std::memory_order_relaxed);
    publish_state();
    if (::absl::GetFlag(FLAGS_crash_after_update) && state.next_update == 1) {
      simulated_crash = true;
      break;
    }
  }

  stop_workers.store(true, std::memory_order_relaxed);
  tasks.Close();
  completions_cv.notify_all();
  for (std::thread& worker_thread : workers) worker_thread.join();
  const double elapsed = std::chrono::duration<double>(
      std::chrono::steady_clock::now() - started).count();
  const double last_actor_completion_wall_seconds =
      last_actor_completion_ns.load(std::memory_order_relaxed) / 1.0e9;
  double total_overlap_seconds = 0.0;
  for (size_t i = 0; i < learner_intervals.size(); ++i) {
    const LearnerInterval& interval = learner_intervals[i];
    const double overlap = std::max(
        0.0, std::min(interval.end_wall_seconds,
                      last_actor_completion_wall_seconds) -
                  interval.start_wall_seconds);
    total_overlap_seconds += overlap;
    if (i < update_records.size()) {
      update_records[i]["search_learner_overlap_seconds"] = overlap;
      AtomicJson((output_dir / ("update_" +
                                std::to_string(interval.update) + ".json"))
                     .string(),
                 update_records[i]);
    }
  }
  int64_t total_requested_simulations = 0;
  int64_t total_completed_simulations = 0;
  int64_t total_full_policy_target_rows = 0;
  int64_t total_fallbacks = 0;
  int64_t total_watchdog_timeouts = 0;
  int64_t total_non_timeout_early_exits = 0;
  double final_staleness_sum = 0.0;
  int final_staleness_max = 0;
  double start_staleness_sum = 0.0;
  int start_staleness_max = 0;
  for (const LedgerEntry& entry : state.completed) {
    total_requested_simulations += entry.requested_simulations;
    total_completed_simulations += entry.completed_simulations;
    total_full_policy_target_rows += entry.full_policy_target_rows;
    total_fallbacks += entry.fallbacks;
    total_watchdog_timeouts += entry.watchdog_timeouts;
    total_non_timeout_early_exits += entry.non_timeout_early_exits;
    const int staleness =
        state.published_generation - entry.collector_generation;
    final_staleness_sum += staleness;
    final_staleness_max = std::max(final_staleness_max, staleness);
    start_staleness_sum += entry.start_staleness;
    start_staleness_max = std::max(start_staleness_max, entry.start_staleness);
  }
  int64_t missing_episode_ids = 0;
  const int64_t expected_first_episode_id = state.first_episode_id;
  for (int64_t episode_id = expected_first_episode_id;
       episode_id < expected_first_episode_id + total_games; ++episode_id) {
    if (!ledger.contains(episode_id)) ++missing_episode_ids;
  }
  const bool duplicate_free = ledger.size() == state.completed.size();
  const bool complete_health =
      total_requested_simulations == total_completed_simulations &&
      total_fallbacks == 0 && total_watchdog_timeouts == 0 &&
      total_non_timeout_early_exits == 0 && missing_episode_ids == 0 &&
      duplicate_free;
  constexpr double kSyncCollectionSeconds = 305.277009;
  constexpr double kSyncLearnerSeconds = 14.955119;
  constexpr int64_t kSyncCompletedSimulations = 2279896;
  constexpr int64_t kSyncFullPolicyTargetRows = 15770;
  constexpr int64_t kSyncLearnerMinibatches = 674;
  const double sync_wall_seconds =
      2.0 * (kSyncCollectionSeconds + kSyncLearnerSeconds);
  const double async_games_per_hour =
      elapsed > 0.0 ? state.completed.size() * 3600.0 / elapsed : 0.0;
  const double async_simulations_per_hour =
      elapsed > 0.0 ? total_completed_simulations * 3600.0 / elapsed : 0.0;
  const double async_full_rows_per_hour =
      elapsed > 0.0 ? total_full_policy_target_rows * 3600.0 / elapsed : 0.0;
  const double async_minibatches_per_hour =
      elapsed > 0.0 ? total_learner_minibatches * 3600.0 / elapsed : 0.0;
  const double sync_games_per_hour = 1024.0 * 3600.0 / sync_wall_seconds;
  const double sync_simulations_per_hour =
      2.0 * kSyncCompletedSimulations * 3600.0 / sync_wall_seconds;
  const double sync_full_rows_per_hour =
      2.0 * kSyncFullPolicyTargetRows * 3600.0 / sync_wall_seconds;
  const double sync_minibatches_per_hour =
      2.0 * kSyncLearnerMinibatches * 3600.0 / sync_wall_seconds;
  json::Object summary;
  summary["updates"] = static_cast<int64_t>(state.next_update);
  summary["completed_games"] = static_cast<int64_t>(state.completed.size());
  summary["requested_simulations"] = total_requested_simulations;
  summary["completed_simulations"] = total_completed_simulations;
  summary["full_policy_target_rows"] = total_full_policy_target_rows;
  summary["completed_ids_unique"] = duplicate_free;
  summary["duplicate_episode_ids"] = duplicate_free ? 0 : 1;
  summary["missing_episode_ids"] = missing_episode_ids;
  summary["expected_first_episode_id"] = expected_first_episode_id;
  summary["expected_episode_end"] = expected_first_episode_id + total_games;
  summary["fallbacks"] = total_fallbacks;
  summary["watchdog_timeouts"] = total_watchdog_timeouts;
  summary["non_timeout_early_exits"] = total_non_timeout_early_exits;
  summary["complete_health"] = complete_health;
  summary["next_episode_id"] = state.next_episode_id;
  summary["published_generation"] =
      static_cast<int64_t>(state.published_generation);
  summary["elapsed_seconds"] = elapsed;
  summary["last_actor_completion_wall_seconds"] =
      last_actor_completion_wall_seconds;
  summary["actor_learner_overlap_seconds"] = total_overlap_seconds;
  summary["actor_learner_overlap_fraction"] =
      elapsed > 0.0 ? total_overlap_seconds / elapsed : 0.0;
  summary["collector_staleness_mean"] = state.completed.empty()
                                             ? 0.0
                                             : final_staleness_sum /
                                                   state.completed.size();
  summary["collector_staleness_max"] =
      static_cast<int64_t>(final_staleness_max);
  summary["collector_start_staleness_mean"] = state.completed.empty()
                                                  ? 0.0
                                                  : start_staleness_sum /
                                                        state.completed.size();
  summary["collector_start_staleness_max"] =
      static_cast<int64_t>(start_staleness_max);
  summary["learner_minibatches"] = total_learner_minibatches;
  summary["completed_games_per_hour"] = async_games_per_hour;
  summary["completed_simulations_per_hour"] = async_simulations_per_hour;
  summary["full_policy_target_rows_per_hour"] = async_full_rows_per_hour;
  summary["learner_minibatches_per_hour"] = async_minibatches_per_hour;
  json::Object sync;
  sync["generations"] = int64_t{2};
  sync["games"] = int64_t{1024};
  sync["wall_seconds"] = sync_wall_seconds;
  sync["completed_simulations"] = 2 * kSyncCompletedSimulations;
  sync["full_policy_target_rows"] = 2 * kSyncFullPolicyTargetRows;
  sync["learner_minibatches"] = 2 * kSyncLearnerMinibatches;
  sync["completed_games_per_hour"] = sync_games_per_hour;
  sync["completed_simulations_per_hour"] = sync_simulations_per_hour;
  sync["full_policy_target_rows_per_hour"] = sync_full_rows_per_hour;
  sync["learner_minibatches_per_hour"] = sync_minibatches_per_hour;
  summary["synchronous_baseline_two_generations"] = std::move(sync);
  summary["useful_work_rate_improves"] =
      async_games_per_hour > sync_games_per_hour &&
      async_simulations_per_hour > sync_simulations_per_hour &&
      async_full_rows_per_hour > sync_full_rows_per_hour &&
      async_minibatches_per_hour > sync_minibatches_per_hour;
  summary["inference_lanes"] = static_cast<int64_t>(inference_lanes);
  summary["batch_target_per_lane"] =
      static_cast<int64_t>(::absl::GetFlag(FLAGS_batch_target));
  summary["batcher_timeout_ms"] =
      static_cast<int64_t>(::absl::GetFlag(FLAGS_batcher_timeout_ms));
  summary["relative_time_budget_ms"] =
      ::absl::GetFlag(FLAGS_relative_time_budget_ms);
  summary["simulated_crash"] = simulated_crash;
  AtomicJson((output_dir / "async_summary.json").string(), summary);
  std::cout << json::ToString(summary, true) << "\n";
  if (simulated_crash) return 77;
  return complete_health ? 0 : 2;
}
