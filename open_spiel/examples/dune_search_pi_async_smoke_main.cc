#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <mutex>
#include <numeric>
#include <queue>
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
ABSL_FLAG(int, games_per_update, 4, "Completed game equivalents per update.");
ABSL_FLAG(int, actors, 4, "Actor workers.");
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
ABSL_FLAG(std::string, resume, "", "Async smoke state JSON.");

namespace open_spiel {
namespace {

struct CollectorSnapshot {
  int generation = 0;
  std::shared_ptr<SharedDunePolicyValueNetImpl> model;
  std::shared_ptr<std::shared_mutex> model_mutex;
  std::shared_ptr<BatchedEvaluator> batcher;
  std::shared_ptr<algorithms::Evaluator> evaluator;
};

struct Task {
  int64_t episode_id = -1;
  std::shared_ptr<CollectorSnapshot> snapshot;
};

struct Completion {
  int64_t episode_id = -1;
  int collector_generation = -1;
  std::vector<SearchPiRow> rows;
  SearchPiGameOutcome outcome;
};

struct LedgerEntry {
  int64_t episode_id = -1;
  int collector_generation = -1;
  std::string shard_path;
  int64_t rows = 0;
  int64_t completed_simulations = 0;
};

struct SmokeState {
  int next_update = 0;
  int64_t next_episode_id = 0;
  int published_generation = 0;
  std::vector<LedgerEntry> completed;
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
  object["next_episode_id"] = state.next_episode_id;
  object["published_generation"] =
      static_cast<int64_t>(state.published_generation);
  json::Array ledger;
  for (const LedgerEntry& entry : state.completed) {
    json::Object item;
    item["episode_id"] = entry.episode_id;
    item["collector_generation"] =
        static_cast<int64_t>(entry.collector_generation);
    item["shard_path"] = entry.shard_path;
    item["rows"] = entry.rows;
    item["completed_simulations"] = entry.completed_simulations;
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
  state.next_episode_id = object.at("next_episode_id").GetInt();
  state.published_generation =
      static_cast<int>(object.at("published_generation").GetInt());
  for (const json::Value& value : object.at("completed_ledger").GetArray()) {
    const json::Object& item = value.GetObject();
    LedgerEntry entry;
    entry.episode_id = item.at("episode_id").GetInt();
    entry.collector_generation =
        static_cast<int>(item.at("collector_generation").GetInt());
    entry.shard_path = item.at("shard_path").GetString();
    entry.rows = item.at("rows").GetInt();
    entry.completed_simulations = item.at("completed_simulations").GetInt();
    state.completed.push_back(std::move(entry));
  }
  return state;
}

std::shared_ptr<CollectorSnapshot> MakeSnapshot(
    const std::shared_ptr<SharedDunePolicyValueNetImpl>& source,
    int generation, int64_t obs_size, int64_t action_dim, torch::Device device,
    int batch_target, int timeout_ms) {
  auto snapshot = std::make_shared<CollectorSnapshot>();
  snapshot->generation = generation;
  snapshot->model = std::make_shared<SharedDunePolicyValueNetImpl>(
      obs_size, 2048, action_dim, 4, false);
  snapshot->model->to(device);
  CopySearchPiModel(source, snapshot->model);
  snapshot->model->eval();
  snapshot->model_mutex = std::make_shared<std::shared_mutex>();
  snapshot->batcher = std::make_shared<BatchedEvaluator>(
      snapshot->model, batch_target, timeout_ms, device,
      snapshot->model_mutex.get(), 0.0f, false, true);
  snapshot->evaluator = std::make_shared<BatchedNNEvaluator>(
      snapshot->batcher, 10.0f);
  return snapshot;
}

SearchPiConfig MakeSearchConfig(int64_t episode_id, int64_t obs_size,
                                uint64_t seed) {
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

 private:
  std::mutex mutex_;
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
  const int games_per_update = absl::GetFlag(FLAGS_games_per_update);
  const int actor_count = absl::GetFlag(FLAGS_actors);
  SPIEL_CHECK_GT(updates, 0);
  SPIEL_CHECK_GT(games_per_update, 0);
  SPIEL_CHECK_GT(actor_count, 0);

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
  learner_config.logit_cap = 10.0;
  learner_config.policy_coef = 1.0;
  learner_config.value_coef = 1.0;
  auto optimizer = MakeSmokeOptimizer(student);

  SmokeState state;
  const std::string resume = ::absl::GetFlag(FLAGS_resume);
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
  } else {
    state.next_episode_id = ::absl::GetFlag(FLAGS_first_episode_id);
    AtomicJson(state_path.string(), ToJson(state));
  }

  std::vector<SearchPiRow> replay_rows;
  std::map<int64_t, LedgerEntry> ledger;
  for (const LedgerEntry& entry : state.completed) {
    if (!ledger.emplace(entry.episode_id, entry).second) {
      SpielFatalError("duplicate completed episode in ledger");
    }
    std::vector<SearchPiRow> rows;
    std::string error;
    if (!ReadScratchSearchPiRowShardV2(entry.shard_path, &rows, &error)) {
      SpielFatalError("async smoke replay load failed: " + error);
    }
    replay_rows.insert(replay_rows.end(), rows.begin(), rows.end());
  }

  auto latest_snapshot = MakeSnapshot(
      student, state.published_generation, obs_size, action_dim, device,
      ::absl::GetFlag(FLAGS_batch_target),
      ::absl::GetFlag(FLAGS_batcher_timeout_ms));
  std::mutex snapshot_mutex;
  TaskQueue tasks;
  std::mutex completions_mutex;
  std::condition_variable completions_cv;
  std::queue<Completion> completions;
  std::atomic<bool> stop_workers{false};

  auto publish_state = [&]() { AtomicJson(state_path.string(), ToJson(state)); };
  const int total_games = updates * games_per_update;
  int64_t target_end = ::absl::GetFlag(FLAGS_first_episode_id) + total_games;
  if (!resume.empty() && static_cast<int>(state.completed.size()) < total_games) {
    target_end = std::max(
        target_end,
        state.next_episode_id +
            (total_games - static_cast<int>(state.completed.size())));
  }

  auto make_task = [&](int64_t episode_id) {
    Task task;
    task.episode_id = episode_id;
    {
      std::lock_guard<std::mutex> lock(snapshot_mutex);
      task.snapshot = latest_snapshot;
    }
    return task;
  };

  auto worker = [&]() {
    torch::InferenceMode inference_guard;
    Task task;
    while (!stop_workers.load(std::memory_order_relaxed) && tasks.Pop(&task)) {
      SearchPiConfig config = MakeSearchConfig(
          task.episode_id, obs_size, ::absl::GetFlag(FLAGS_seed));
      std::vector<SearchPiRow> rows;
      SearchPiGenerationStats stats;
      SearchPiGenerator(config).GenerateGeneration(
          task.snapshot->generation, game, task.snapshot->evaluator,
          task.snapshot->evaluator, &rows, &stats, SearchPiArm::kSearched);
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
      completion.rows = std::move(rows);
      completion.outcome = stats.games_played.front();
      CheckCompletedGame(completion);
      {
        std::lock_guard<std::mutex> lock(completions_mutex);
        completions.push(std::move(completion));
      }
      completions_cv.notify_one();
    }
  };

  std::vector<std::thread> workers;
  workers.reserve(actor_count);
  for (int i = 0; i < actor_count; ++i) workers.emplace_back(worker);
  int64_t issued = state.next_episode_id;
  auto issue_one = [&]() {
    if (issued >= target_end) return false;
    tasks.Push(make_task(issued));
    ++issued;
    state.next_episode_id = issued;
    return true;
  };
  for (int i = 0; i < actor_count && issued < target_end; ++i) issue_one();
  state.next_episode_id = issued;
  publish_state();

  std::unique_ptr<c10::cuda::CUDAStreamGuard> learner_guard;
  std::unique_ptr<c10::cuda::CUDAStream> learner_stream;
  if (device.is_cuda()) {
    const c10::DeviceIndex device_index =
        device.has_index() ? device.index() : c10::cuda::current_device();
    learner_stream = std::make_unique<c10::cuda::CUDAStream>(
        c10::cuda::getStreamFromPool(false, device_index));
  }
  std::vector<int64_t> update_new_ids;
  std::vector<double> update_learner_seconds;
  const auto started = std::chrono::steady_clock::now();
  bool simulated_crash = false;
  while (state.next_update < updates) {
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
    entry.shard_path = shard_path.string();
    entry.rows = static_cast<int64_t>(completion.rows.size());
    entry.completed_simulations = 0;
    for (const SearchPiRow& row : completion.rows) {
      entry.completed_simulations += row.simulations_completed;
    }
    ledger.emplace(entry.episode_id, entry);
    state.completed.push_back(entry);
    replay_rows.insert(replay_rows.end(), completion.rows.begin(),
                       completion.rows.end());
    update_new_ids.push_back(completion.episode_id);
    issue_one();
    publish_state();

    const int update_target = (state.next_update + 1) * games_per_update;
    if (static_cast<int>(state.completed.size()) < update_target) continue;
    const auto learner_start = std::chrono::steady_clock::now();
    if (device.is_cuda()) {
      learner_guard = std::make_unique<c10::cuda::CUDAStreamGuard>(
          *learner_stream);
    }
    ScratchSearchPiLearnerStats learner_stats = RunScratchSearchPiLearner(
        student, *optimizer, replay_rows, obs_size, action_dim, device,
        ::absl::GetFlag(FLAGS_seed), state.next_update + 1, learner_config);
    learner_guard.reset();
    const double learner_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - learner_start).count();
    update_learner_seconds.push_back(learner_seconds);

    ++state.published_generation;
    {
      std::lock_guard<std::mutex> lock(snapshot_mutex);
      latest_snapshot = MakeSnapshot(
          student, state.published_generation, obs_size, action_dim, device,
          ::absl::GetFlag(FLAGS_batch_target),
          ::absl::GetFlag(FLAGS_batcher_timeout_ms));
    }
    SaveStudentCheckpoint(output_dir, state.next_update + 1, student,
                          *optimizer);
    json::Object update;
    update["update"] = static_cast<int64_t>(state.next_update + 1);
    update["completed_games"] =
        static_cast<int64_t>(state.completed.size());
    update["new_games_this_update"] =
        static_cast<int64_t>(update_new_ids.size());
    update["replay_rows"] = static_cast<int64_t>(replay_rows.size());
    update["learner_seconds"] = learner_seconds;
    update["learner_minibatches"] =
        static_cast<int64_t>(learner_stats.minibatches);
    update["learner_streamed"] = true;
    update["learner_value_prediction_mean_pre"] =
        learner_stats.critic_pred_mean_pre;
    update["learner_value_prediction_mean_post"] =
        learner_stats.critic_pred_mean_post;
    int64_t total_simulations = 0;
    int64_t full_policy_target_rows = 0;
    double staleness_sum = 0.0;
    int max_staleness = 0;
    for (const LedgerEntry& item : state.completed) {
      total_simulations += item.completed_simulations;
      const int staleness =
          state.published_generation - item.collector_generation;
      staleness_sum += staleness;
      max_staleness = std::max(max_staleness, staleness);
    }
    update["completed_simulations"] = total_simulations;
    for (const SearchPiRow& row : replay_rows) {
      if (row.search_budget_class == "full" &&
          row.policy_target_weight > 0.0) {
        ++full_policy_target_rows;
      }
    }
    update["full_policy_target_rows"] = full_policy_target_rows;
    update["staleness_mean"] = state.completed.empty()
                                    ? 0.0
                                    : staleness_sum / state.completed.size();
    update["staleness_max"] = static_cast<int64_t>(max_staleness);
    AtomicJson((output_dir / ("update_" +
                              std::to_string(state.next_update + 1) +
                              ".json"))
                   .string(),
               update);
    update_new_ids.clear();
    ++state.next_update;
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
  json::Object summary;
  summary["updates"] = static_cast<int64_t>(state.next_update);
  summary["completed_games"] = static_cast<int64_t>(state.completed.size());
  summary["completed_ids_unique"] = ledger.size() == state.completed.size();
  summary["next_episode_id"] = state.next_episode_id;
  summary["published_generation"] =
      static_cast<int64_t>(state.published_generation);
  summary["elapsed_seconds"] = elapsed;
  summary["simulated_crash"] = simulated_crash;
  AtomicJson((output_dir / "async_summary.json").string(), summary);
  std::cout << json::ToString(summary, true) << "\n";
  return simulated_crash ? 77 : 0;
}
