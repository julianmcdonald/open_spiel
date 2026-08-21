#include <algorithm>
#include <cmath>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <future>
#include <iostream>
#include <iterator>
#include <limits>
#include <memory>
#include <numeric>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <torch/torch.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAFunctions.h>
#include "open_spiel/abseil-cpp/absl/flags/flag.h"
#include "open_spiel/abseil-cpp/absl/flags/parse.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"
#include "open_spiel/utils/json.h"
#include "dune_search_pi_concurrent.h"
#include "dune_search_pi_replay.h"
#include "dune_search_pi_scratch.h"
#include "dune_ppo_training_utils.h"
#include "dune_evaluator.h"
#include "dune_seed_utils.h"
#include "dune_sha256.h"
#include "open_spiel/examples/dune_hotspot_profile.h"

// Link-only PPO flag definitions required by dune_ppo_training_utils.cc. The
// scratch learner never reads them; they keep this standalone executable from
// depending on dune_ppo_train.cc (and therefore from acquiring PPO behavior).
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

ABSL_FLAG(std::string, profile, open_spiel::kWarmSearchPiProfile,
          "ppo_warm_q_v1, ppo_warm_cmpo_norm_v1, scratch_q_v1 or scratch_visit_v1");
ABSL_FLAG(std::string, output_dir, "", "Fresh runtime output directory.");
ABSL_FLAG(std::string, init_model_checkpoint, "",
          "Warm-start model; forbidden for scratch profiles.");
ABSL_FLAG(std::string, init_model_sha256, "",
          "Required SHA-256 for the warm-start model.");
ABSL_FLAG(std::string, resume, "", "Scratch checkpoint JSON to resume.");
ABSL_FLAG(bool, verify_resume_only, false,
          "Load and validate checkpoint/cursor/replay, then exit.");
ABSL_FLAG(int, generations, 5, "Total generation boundary to reach.");
ABSL_FLAG(int, games_per_generation, 128, "Complete games per generation.");
ABSL_FLAG(int64_t, first_episode_id, 0, "Initial episode cursor.");
ABSL_FLAG(uint64_t, seed, 8271001, "Master seed and search stream domain.");
ABSL_FLAG(int, hidden_dim, 2048, "Network width.");
ABSL_FLAG(int, num_blocks, 8, "Residual blocks.");
ABSL_FLAG(bool, nonlinear_value_head, false, "Nonlinear value head.");
ABSL_FLAG(int, workers, 32, "Concurrent complete-game workers.");
ABSL_FLAG(int, batch_target, 64, "Shared inference batch target.");
ABSL_FLAG(int, inference_lanes, 1,
          "Independent immutable collector-model/batcher lanes.");
ABSL_FLAG(int, batcher_timeout_ms, 2, "Inference batch timeout.");
ABSL_FLAG(bool, batcher_telemetry, true,
          "Collect detailed per-batch/device telemetry.");
ABSL_FLAG(bool, evaluator_device_synchronize, true,
          "Use whole-device CUDA synchronize after evaluator D2H copies.");
ABSL_FLAG(int, prefetch_games, 0,
          "Bounded next-generation tail-prefetch games (0 disables).");
ABSL_FLAG(int, prefetch_trigger_running, 32,
          "Start tail prefetch below this many current games running.");
ABSL_FLAG(bool, hotspot_profile, false,
          "Collect opt-in CPU hotspot timings for one generation.");
ABSL_FLAG(int, hotspot_profile_games, 0,
          "Bounded complete games for hotspot profiling (0 keeps the run size).");
ABSL_FLAG(int, chunk_games, 4, "Seat-balanced games per work chunk.");
ABSL_FLAG(double, full_root_probability, 0.25, "Full-search root chance.");
ABSL_FLAG(int, full_primary_budget, 200, "Full primary simulations.");
ABSL_FLAG(int, full_other_budget, 64, "Full non-primary simulations.");
ABSL_FLAG(int, cheap_primary_budget, 32, "Cheap primary simulations.");
ABSL_FLAG(int, cheap_other_budget, 16, "Cheap non-primary simulations.");
ABSL_FLAG(double, relative_time_budget_ms, 600000.0,
          "Per-root technical watchdog.");
ABSL_FLAG(size_t, replay_max_rows, 100000, "Maximum learner rows.");
ABSL_FLAG(int, replay_generations, 4, "Recent shard window.");
ABSL_FLAG(double, learning_rate, 1e-4, "Search learner AdamW LR.");
ABSL_FLAG(int, minibatch_size, 256, "Search learner minibatch size.");
ABSL_FLAG(int, epochs, 1, "Search learner epochs.");
ABSL_FLAG(double, grad_clip_norm, 0.5, "Gradient clip norm.");
ABSL_FLAG(double, logit_cap, 10.0, "Legal-centered tanh logit cap.");
ABSL_FLAG(double, intermediate_vp_breadcrumb_weight, 0.2,
          "Training-only intermediate VP breadcrumb weight.");
ABSL_FLAG(double, specimen_exchange_penalty, 0.02,
          "Positive training-only anti-breadcrumb penalty.");
ABSL_FLAG(uint64_t, shaping_start_env_steps, 0,
          "Environment transition count at which shaping decay starts.");
ABSL_FLAG(uint64_t, shaping_decay_env_steps, 10000000,
          "Environment transition count over which shaping decays.");

namespace open_spiel {
namespace {

struct ResumeState {
  int generation = 0;
  int64_t next_episode_id = 0;
  uint64_t total_env_steps = 0;
  std::string model_path;
  std::string optimizer_path;
  std::string model_sha256;
  std::string optimizer_sha256;
  std::string student_digest;
  std::vector<std::string> replay_paths;
  std::vector<std::string> replay_sha256;
  std::vector<std::vector<std::string>> replay_cohort_groups;
  int pending_prefetch_games = 0;
  int pending_prefetch_collector_generation = -1;
  uint64_t pending_prefetch_env_steps = 0;
};

struct CudaMemorySnapshot {
  int64_t allocated_bytes = 0;
  int64_t reserved_bytes = 0;
  int64_t peak_allocated_bytes = 0;
  int64_t peak_reserved_bytes = 0;
};

void PrepareCudaCollection(torch::Device device) {
  if (!device.is_cuda()) return;
  const c10::DeviceIndex index =
      device.has_index() ? device.index() : c10::cuda::current_device();
  c10::cuda::CUDACachingAllocator::emptyCache();
  c10::cuda::CUDACachingAllocator::resetPeakStats(index);
}

void ClearUnusedCudaCache(torch::Device device) {
  if (!device.is_cuda()) return;
  c10::cuda::CUDACachingAllocator::emptyCache();
}

void ResetLearnerCudaPeak(torch::Device device) {
  if (!device.is_cuda()) return;
  c10::cuda::CUDACachingAllocator::emptyCache();
  const c10::DeviceIndex index =
      device.has_index() ? device.index() : c10::cuda::current_device();
  c10::cuda::CUDACachingAllocator::resetPeakStats(index);
}

CudaMemorySnapshot ReadCudaMemory(torch::Device device) {
  CudaMemorySnapshot snapshot;
  if (!device.is_cuda()) return snapshot;
  const c10::DeviceIndex index =
      device.has_index() ? device.index() : c10::cuda::current_device();
  const auto stats =
      c10::cuda::CUDACachingAllocator::getDeviceStats(index);
  constexpr size_t kAggregate = static_cast<size_t>(
      c10::CachingAllocator::StatType::AGGREGATE);
  snapshot.allocated_bytes = stats.allocated_bytes[kAggregate].current;
  snapshot.reserved_bytes = stats.reserved_bytes[kAggregate].current;
  snapshot.peak_allocated_bytes = stats.allocated_bytes[kAggregate].peak;
  snapshot.peak_reserved_bytes = stats.reserved_bytes[kAggregate].peak;
  return snapshot;
}

std::string ReadText(const std::string& path) {
  std::ifstream input(path);
  if (!input) SpielFatalError("cannot read file: " + path);
  return std::string(std::istreambuf_iterator<char>(input),
                     std::istreambuf_iterator<char>());
}

void AtomicJson(const std::string& path, const json::Object& value) {
  const std::string temporary = path + ".tmp";
  {
    std::ofstream output(temporary, std::ios::trunc);
    if (!output) SpielFatalError("cannot write JSON temp file: " + temporary);
    output << json::ToString(value, true) << "\n";
    output.flush();
    if (!output) SpielFatalError("failed writing JSON temp file: " + temporary);
  }
  std::error_code error;
  std::filesystem::rename(temporary, path, error);
  if (error) SpielFatalError("cannot atomically publish JSON: " + error.message());
}

std::unique_ptr<torch::optim::AdamW> MakeOptimizer(
    const std::shared_ptr<SharedDunePolicyValueNetImpl>& model,
    const SearchPiLearnerConfig& config) {
  std::vector<torch::Tensor> policy;
  std::vector<torch::Tensor> other;
  const auto policy_set = model->policy_head->parameters();
  for (torch::Tensor parameter : model->parameters()) {
    bool is_policy = false;
    for (const torch::Tensor& candidate : policy_set) {
      if (parameter.is_same(candidate)) is_policy = true;
    }
    (is_policy ? policy : other).push_back(parameter);
  }
  std::vector<torch::optim::OptimizerParamGroup> groups;
  groups.emplace_back(policy);
  groups.emplace_back(other);
  auto optimizer = std::make_unique<torch::optim::AdamW>(
      groups, torch::optim::AdamWOptions(config.learning_rate).eps(1e-5));
  static_cast<torch::optim::AdamWOptions&>(
      optimizer->param_groups()[0].options())
      .weight_decay(config.policy_weight_decay);
  static_cast<torch::optim::AdamWOptions&>(
      optimizer->param_groups()[1].options())
      .weight_decay(config.weight_decay);
  return optimizer;
}

std::string Fingerprint(const std::string& profile, const SearchPiConfig& search,
                        const SearchPiLearnerConfig& learner, int hidden_dim,
                        int blocks, bool nonlinear, int workers,
                        int batch_target, int timeout_ms, int chunk_games,
                        size_t replay_max, int replay_generations,
                        const std::string& init_sha, uint64_t seed,
                        int64_t first_episode_id) {
  std::ostringstream value;
  const bool normalized_profile = profile == kWarmNormalizedCmpoProfile;
  const bool visit_profile = profile == kScratchVisitSearchPiProfile;
  value << (normalized_profile ? "search_pi_cmpo_norm_v2|profile="
             : visit_profile ? "search_pi_visit_v1|profile="
                              : "search_pi_q_v1|profile=")
        << profile
        << "|hidden=" << hidden_dim
        << "|blocks=" << blocks << "|nonlinear=" << nonlinear
        << "|workers=" << workers << "|batch=" << batch_target
        << "|timeout=" << timeout_ms << "|chunk=" << chunk_games
        << "|games=" << search.games_per_generation
        << "|fullp=" << search.scratch_full_root_probability
        << "|fp=" << search.scratch_full_primary_simulations
        << "|fo=" << search.scratch_full_other_simulations
        << "|cp=" << search.scratch_cheap_primary_simulations
        << "|co=" << search.scratch_cheap_other_simulations
        << "|puct=" << search.puct_c << "|eps=" << search.dirichlet_epsilon
        << "|alpha=" << search.dirichlet_alpha_total
        << "|forced=" << search.forced_playouts_k
        << "|fpu0=" << search.root_noise_fpu_zero
        << "|watchdog=" << search.relative_time_budget_ms
        << "|behavior=" << search.behavior_temperature
        << "|vp_weight=" << search.intermediate_vp_breadcrumb_weight
        << "|specimen_penalty=" << search.specimen_exchange_penalty
        << "|shape_start=" << search.shaping_start_env_steps
        << "|shape_decay=" << search.shaping_decay_env_steps
        << "|replaymax=" << replay_max
        << "|replaygens=" << replay_generations
        << "|lr=" << learner.learning_rate << "|eps=1e-5"
        << "|mb=" << learner.minibatch_size << "|epochs=" << learner.epochs
        << "|clip=" << learner.grad_clip_norm
        << "|logit=" << learner.logit_cap << "|target="
        << (normalized_profile ? "normalized_cmpo"
             : visit_profile ? "visit_policy" : "regularized_q_kl")
        << "|initsha=" << init_sha;
  if (normalized_profile || visit_profile) {
    value << "|policy_coef=" << learner.policy_coef
          << "|value_coef=" << learner.value_coef << "|seed=" << seed
          << "|first_episode=" << first_episode_id;
  }
  return ComputeStringSHA256(value.str());
}

ResumeState ReadResume(const std::string& path, const std::string& profile,
                       int expected_hidden_dim, int expected_num_blocks,
                       uint64_t expected_seed, double expected_logit_cap,
                       int64_t expected_first_episode_id) {
  auto parsed = json::FromString(ReadText(path));
  if (!parsed.has_value() || !parsed->IsObject()) {
    SpielFatalError("scratch resume is not a JSON object.");
  }
  const json::Object& object = parsed->GetObject();
  if (object.at("profile").GetString() != profile) {
    SpielFatalError("historical/PPO checkpoint cannot resume this Q profile.");
  }
  ResumeState state;
  state.generation = object.at("generation").GetInt();
  state.next_episode_id = object.at("next_episode_id").GetInt();
  state.total_env_steps = static_cast<uint64_t>(
      object.at("total_env_steps").GetInt());
  state.model_path = object.at("model_checkpoint").GetString();
  state.optimizer_path = object.at("optimizer_checkpoint").GetString();
  if (state.next_episode_id < expected_first_episode_id) {
    SpielFatalError("resume episode cursor precedes the registered start.");
  }
  const std::filesystem::path resolved_path =
      std::filesystem::path(path).parent_path().parent_path() /
      "resolved_config.json";
  if (!std::filesystem::exists(resolved_path)) {
    SpielFatalError("resume semantic configuration is missing: " +
                    resolved_path.string());
  }
  const auto resolved = json::FromString(ReadText(resolved_path.string()));
  if (!resolved.has_value() || !resolved->IsObject()) {
    SpielFatalError("resume semantic configuration is malformed.");
  }
  const json::Object& resolved_object = resolved->GetObject();
  auto require_int = [&](const char* key, int64_t expected) {
    const auto it = resolved_object.find(key);
    if (it == resolved_object.end() || !it->second.IsInt() ||
        it->second.GetInt() != expected) {
      SpielFatalError(absl::StrCat("resume semantic mismatch for ", key));
    }
  };
  auto require_double = [&](const char* key, double expected) {
    const auto it = resolved_object.find(key);
    const double actual =
        it == resolved_object.end()
            ? std::numeric_limits<double>::quiet_NaN()
            : it->second.IsDouble()
                  ? it->second.GetDouble()
                  : it->second.IsInt()
                        ? static_cast<double>(it->second.GetInt())
                        : std::numeric_limits<double>::quiet_NaN();
    if (!std::isfinite(actual) || std::abs(actual - expected) > 1e-12) {
      SpielFatalError(absl::StrCat("resume semantic mismatch for ", key));
    }
  };
  const auto target_type = resolved_object.find("target_type");
  if (target_type == resolved_object.end() || !target_type->second.IsString() ||
      target_type->second.GetString() != "visit_policy") {
    SpielFatalError("resume semantic mismatch for target_type");
  }
  require_int("hidden_dim", expected_hidden_dim);
  require_int("num_blocks", expected_num_blocks);
  require_int("seed", static_cast<int64_t>(expected_seed));
  require_int("first_episode_id", expected_first_episode_id);
  require_double("logit_cap", expected_logit_cap);
  const auto model_sha = object.find("model_sha256");
  if (model_sha != object.end()) state.model_sha256 = model_sha->second.GetString();
  const auto optimizer_sha = object.find("optimizer_sha256");
  if (optimizer_sha != object.end()) {
    state.optimizer_sha256 = optimizer_sha->second.GetString();
  }
  const auto student_digest = object.find("student_digest");
  if (student_digest != object.end()) {
    state.student_digest = student_digest->second.GetString();
  }
  for (const json::Value& value : object.at("replay_paths").GetArray()) {
    state.replay_paths.push_back(value.GetString());
  }
  const auto replay_hashes = object.find("replay_sha256");
  if (replay_hashes != object.end()) {
    for (const json::Value& value : replay_hashes->second.GetArray()) {
      state.replay_sha256.push_back(value.GetString());
    }
  }
  const auto cohort_groups = object.find("replay_cohort_groups");
  if (cohort_groups != object.end() && cohort_groups->second.IsArray()) {
    for (const json::Value& group_value : cohort_groups->second.GetArray()) {
      if (!group_value.IsArray()) {
        SpielFatalError("scratch resume cohort group is not an array.");
      }
      std::vector<std::string> group;
      for (const json::Value& path_value : group_value.GetArray()) {
        group.push_back(path_value.GetString());
      }
      state.replay_cohort_groups.push_back(std::move(group));
    }
  }
  if (state.replay_cohort_groups.empty()) {
    for (const std::string& replay_path : state.replay_paths) {
      state.replay_cohort_groups.push_back({replay_path});
    }
  }
  const auto pending_games = object.find("pending_prefetch_games");
  if (pending_games != object.end()) {
    state.pending_prefetch_games =
        static_cast<int>(pending_games->second.GetInt());
  }
  const auto pending_generation =
      object.find("pending_prefetch_collector_generation");
  if (pending_generation != object.end()) {
    state.pending_prefetch_collector_generation =
        static_cast<int>(pending_generation->second.GetInt());
  }
  const auto pending_steps = object.find("pending_prefetch_env_steps");
  if (pending_steps != object.end()) {
    state.pending_prefetch_env_steps =
        static_cast<uint64_t>(pending_steps->second.GetInt());
  }
  return state;
}

json::Object ResolvedConfig(const std::string& profile,
                            const SearchPiConfig& search,
                            const SearchPiLearnerConfig& learner,
                            const std::string& fingerprint,
                            const std::string& init_model,
                            const std::string& init_sha, int hidden_dim,
                            int blocks, bool nonlinear, int workers,
                            int batch_target, int timeout_ms, int chunk_games,
                            int inference_lanes, size_t replay_max,
                            int replay_generations,
                            uint64_t seed, int64_t first_episode_id) {
  json::Object result;
  result["schema_version"] = int64_t{1};
  result["profile"] = profile;
  result["config_fingerprint"] = fingerprint;
  result["init_model_checkpoint"] = init_model;
  result["init_model_sha256"] = init_sha;
  result["hidden_dim"] = int64_t{hidden_dim};
  result["num_blocks"] = int64_t{blocks};
  result["nonlinear_value_head"] = nonlinear;
  result["games_per_generation"] = int64_t{search.games_per_generation};
  result["seed"] = int64_t{static_cast<int64_t>(seed)};
  result["first_episode_id"] = first_episode_id;
  result["total_env_steps_start"] = int64_t{0};
  result["workers"] = int64_t{workers};
  result["batch_target"] = int64_t{batch_target};
  result["inference_lanes"] = int64_t{inference_lanes};
  result["batcher_timeout_ms"] = int64_t{timeout_ms};
  result["chunk_games"] = int64_t{chunk_games};
  result["full_root_probability"] = search.scratch_full_root_probability;
  result["full_primary_budget"] =
      int64_t{search.scratch_full_primary_simulations};
  result["full_other_budget"] =
      int64_t{search.scratch_full_other_simulations};
  result["cheap_primary_budget"] =
      int64_t{search.scratch_cheap_primary_simulations};
  result["cheap_other_budget"] =
      int64_t{search.scratch_cheap_other_simulations};
  result["target_type"] =
      profile == kWarmNormalizedCmpoProfile
          ? "normalized_cmpo"
          : profile == kScratchVisitSearchPiProfile ? "visit_policy"
                                                     : "regularized_q_kl";
  result["kl_cap"] = profile == kScratchVisitSearchPiProfile ? 0.0 : 0.10;
  result["prior_floor"] =
      (profile == kWarmNormalizedCmpoProfile ||
       profile == kScratchVisitSearchPiProfile) ? 1e-12 : 1e-8;
  result["advantage_clip"] =
      profile == kWarmNormalizedCmpoProfile
          ? 1.0
          : profile == kScratchVisitSearchPiProfile ? 0.0 : 0.25;
  result["max_beta"] =
      profile == kWarmNormalizedCmpoProfile
          ? 1.0
          : profile == kScratchVisitSearchPiProfile ? 0.0 : 4.0;
  result["normalized_sigma_floor"] =
      profile == kWarmNormalizedCmpoProfile ? 1e-6 : 0.0;
  result["policy_coef"] = learner.policy_coef;
  result["value_coef"] = learner.value_coef;
  result["dirichlet_epsilon"] = search.dirichlet_epsilon;
  result["dirichlet_alpha_total"] = search.dirichlet_alpha_total;
  result["forced_playouts_k"] = search.forced_playouts_k;
  result["root_noise_fpu_zero"] = search.root_noise_fpu_zero;
  result["behavior_temperature"] = search.behavior_temperature;
  result["intermediate_vp_breadcrumb_weight"] =
      search.intermediate_vp_breadcrumb_weight;
  result["specimen_exchange_penalty"] = search.specimen_exchange_penalty;
  result["shaping_start_env_steps"] =
      static_cast<int64_t>(search.shaping_start_env_steps);
  result["shaping_decay_env_steps"] =
      static_cast<int64_t>(search.shaping_decay_env_steps);
  result["shaping_lambda"] = static_cast<double>(search.shaping_lambda);
  result["learning_rate"] = learner.learning_rate;
  result["adamw_epsilon"] = 1e-5;
  result["minibatch_size"] = int64_t{learner.minibatch_size};
  result["epochs"] = int64_t{learner.epochs};
  result["grad_clip_norm"] = learner.grad_clip_norm;
  result["logit_cap"] = learner.logit_cap;
  result["replay_max_rows"] = static_cast<int64_t>(replay_max);
  result["replay_generations"] = int64_t{replay_generations};
  return result;
}

json::Object SaveCheckpoint(
    const std::string& output_dir, const std::string& profile,
    const std::string& fingerprint, int generation, int64_t next_episode_id,
    uint64_t total_env_steps,
    const std::vector<std::string>& replay_paths,
    const std::vector<std::vector<std::string>>& replay_cohort_groups,
    int pending_prefetch_games,
    int pending_prefetch_collector_generation,
    uint64_t pending_prefetch_env_steps,
    const std::shared_ptr<SharedDunePolicyValueNetImpl>& model,
    torch::optim::Optimizer& optimizer) {
  const std::filesystem::path checkpoint_dir =
      std::filesystem::path(output_dir) / "checkpoints";
  std::filesystem::create_directories(checkpoint_dir);
  const std::string stem = "generation_" + std::to_string(generation);
  const std::string model_path = (checkpoint_dir / (stem + "_model.pt")).string();
  const std::string optimizer_path =
      (checkpoint_dir / (stem + "_optimizer.pt")).string();
  const std::string model_tmp = model_path + ".tmp";
  const std::string optimizer_tmp = optimizer_path + ".tmp";
  torch::save(model, model_tmp);
  torch::save(optimizer, optimizer_tmp);
  std::filesystem::rename(model_tmp, model_path);
  std::filesystem::rename(optimizer_tmp, optimizer_path);

  json::Object checkpoint;
  checkpoint["schema_version"] = int64_t{1};
  checkpoint["profile"] = profile;
  checkpoint["config_fingerprint"] = fingerprint;
  checkpoint["generation"] = int64_t{generation};
  checkpoint["next_episode_id"] = next_episode_id;
  checkpoint["total_env_steps"] =
      static_cast<int64_t>(total_env_steps);
  checkpoint["model_checkpoint"] = model_path;
  checkpoint["optimizer_checkpoint"] = optimizer_path;
  checkpoint["model_sha256"] = "";
  checkpoint["optimizer_sha256"] = "";
  checkpoint["student_digest"] = "";
  json::Array paths;
  json::Array replay_hashes;
  for (const std::string& path : replay_paths) {
    paths.push_back(path);
    replay_hashes.push_back("");
  }
  checkpoint["replay_paths"] = std::move(paths);
  checkpoint["replay_sha256"] = std::move(replay_hashes);
  json::Array cohort_groups;
  for (const auto& group : replay_cohort_groups) {
    json::Array paths_in_group;
    for (const std::string& path : group) paths_in_group.push_back(path);
    cohort_groups.push_back(std::move(paths_in_group));
  }
  checkpoint["replay_cohort_groups"] = std::move(cohort_groups);
  checkpoint["pending_prefetch_games"] =
      static_cast<int64_t>(pending_prefetch_games);
  checkpoint["pending_prefetch_collector_generation"] =
      static_cast<int64_t>(pending_prefetch_collector_generation);
  checkpoint["pending_prefetch_env_steps"] =
      static_cast<int64_t>(pending_prefetch_env_steps);
  AtomicJson((checkpoint_dir / (stem + ".json")).string(), checkpoint);
  AtomicJson((std::filesystem::path(output_dir) / "checkpoint_latest.json").string(),
             checkpoint);
  return checkpoint;
}

void SaveInitialRandomModelIfNeeded(
    const std::string& output_dir, const std::string& profile,
    const std::string& fingerprint, uint64_t seed, int64_t first_episode_id,
    bool random_init,
    const std::shared_ptr<SharedDunePolicyValueNetImpl>& model) {
  if (!random_init) return;
  const std::filesystem::path checkpoint_dir =
      std::filesystem::path(output_dir) / "checkpoints";
  std::filesystem::create_directories(checkpoint_dir);
  const std::filesystem::path model_path =
      checkpoint_dir / "generation_0_model.pt";
  const std::filesystem::path marker_path =
      checkpoint_dir / "generation_0.json";
  bool created_model = false;
  if (!std::filesystem::exists(model_path)) {
    const std::filesystem::path temporary = model_path.string() + ".tmp";
    torch::save(model, temporary.string());
    std::error_code error;
    std::filesystem::rename(temporary, model_path, error);
    if (error) {
      std::filesystem::remove(temporary);
      SpielFatalError("cannot publish generation-zero model: " +
                      error.message());
    }
    created_model = true;
  }
  if (!std::filesystem::exists(marker_path) && !created_model) {
    SpielFatalError("generation-zero model exists without its lineage marker.");
  }
  if (created_model) {
    json::Object marker;
    marker["schema_version"] = int64_t{1};
    marker["profile"] = profile;
    marker["config_fingerprint"] = fingerprint;
    marker["seed"] = static_cast<int64_t>(seed);
    marker["first_episode_id"] = first_episode_id;
    marker["generation"] = int64_t{0};
    marker["model_checkpoint"] = model_path.string();
    marker["model_sha256"] = ComputeFileSHA256(model_path.string());
    marker["student_digest"] = CanonicalSearchPiModuleDigest(*model);
    AtomicJson(marker_path.string(), marker);
  }
  const auto parsed = json::FromString(ReadText(marker_path.string()));
  if (!parsed.has_value() || !parsed->IsObject()) {
    SpielFatalError("generation-zero marker is malformed.");
  }
  const json::Object& marker = parsed->GetObject();
  if (marker.at("profile").GetString() != profile ||
      marker.at("config_fingerprint").GetString() != fingerprint ||
      marker.at("generation").GetInt() != 0 ||
      marker.at("seed").GetInt() != static_cast<int64_t>(seed) ||
      marker.at("first_episode_id").GetInt() != first_episode_id ||
      marker.at("model_checkpoint").GetString() != model_path.string() ||
      marker.at("model_sha256").GetString() !=
          ComputeFileSHA256(model_path.string()) ||
      marker.at("student_digest").GetString().empty()) {
    SpielFatalError("generation-zero lineage marker validation failed.");
  }
}

SearchPiRoleStats SumRole(const ConcurrentSearchPiCollectionResult& result,
                          DuneDecisionRole role) {
  if (role == DuneDecisionRole::kAgentPrimary) return result.primary;
  if (role == DuneDecisionRole::kAgentContinuation) return result.continuation;
  if (role == DuneDecisionRole::kPurchase) return result.purchase;
  if (role == DuneDecisionRole::kCombatIntrigue) return result.combat_intrigue;
  return result.other_optional;
}

void ValidateCollection(const ConcurrentSearchPiCollectionResult& result,
                        const SearchPiConfig& config) {
  SPIEL_CHECK_EQ(result.games.size(), config.games_per_generation);
  SPIEL_CHECK_EQ(result.leader_rows_emitted, 0);
  SPIEL_CHECK_EQ(result.retained_rows.size(), result.rows_total);
  for (const SearchPiRow& row : result.retained_rows) {
    SPIEL_CHECK_EQ(row.row_schema_version, 2);
    SPIEL_CHECK_TRUE(row.search_budget_class == "full" ||
                     row.search_budget_class == "cheap");
    if (row.target_type != "visit_policy") {
      SPIEL_CHECK_TRUE(row.regularized_q_valid);
      SPIEL_CHECK_EQ(row.regularized_q_error, "none");
    }
    SPIEL_CHECK_EQ(row.simulations_completed, row.simulations_requested);
    const ScratchSearchBudgetAssignment expected = AssignScratchSearchBudget(
        config, row.episode_id, row.decision_id, row.player, row.role);
    SPIEL_CHECK_EQ(row.search_budget_draw, expected.draw);
    SPIEL_CHECK_EQ(row.simulations_requested, expected.simulations);
  }
  const DuneDecisionRole roles[] = {
      DuneDecisionRole::kAgentPrimary, DuneDecisionRole::kAgentContinuation,
      DuneDecisionRole::kPurchase, DuneDecisionRole::kCombatIntrigue,
      DuneDecisionRole::kOtherOptional};
  for (DuneDecisionRole role : roles) {
    const SearchPiRoleStats stats = SumRole(result, role);
    SPIEL_CHECK_EQ(stats.fallbacks, 0);
    SPIEL_CHECK_EQ(stats.simulations_completed, stats.simulations_requested);
  }
}

}  // namespace
}  // namespace open_spiel

int main(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);
  using namespace open_spiel;
  const std::string profile = absl::GetFlag(FLAGS_profile);
  const std::string output_dir = absl::GetFlag(FLAGS_output_dir);
  if (profile != kWarmSearchPiProfile &&
      profile != kWarmNormalizedCmpoProfile &&
      profile != kScratchSearchPiProfile &&
      profile != kScratchVisitSearchPiProfile) {
    SpielFatalError(
        "profile must be ppo_warm_q_v1, ppo_warm_cmpo_norm_v1, scratch_q_v1 "
        "or scratch_visit_v1.");
  }
  if (output_dir.empty()) SpielFatalError("--output_dir is required.");
  const bool normalized = profile == kWarmNormalizedCmpoProfile;
  const bool visit = profile == kScratchVisitSearchPiProfile;
  const bool warm = profile == kWarmSearchPiProfile || normalized;
  const std::string init_model = absl::GetFlag(FLAGS_init_model_checkpoint);
  std::string init_sha = absl::GetFlag(FLAGS_init_model_sha256);
  if (warm && init_model.empty()) {
    SpielFatalError("warm Search-PI profiles require --init_model_checkpoint.");
  }
  if (!warm && !init_model.empty()) {
    SpielFatalError("scratch profiles generation zero rejects a model checkpoint.");
  }
  if (warm && absl::GetFlag(FLAGS_num_blocks) != 8) {
    SpielFatalError("warm Search-PI profiles retain the mature checkpoint's eight blocks.");
  }
  if (!warm && absl::GetFlag(FLAGS_num_blocks) != 4) {
    SpielFatalError("scratch profiles use exactly four residual blocks.");
  }
  if (absl::GetFlag(FLAGS_nonlinear_value_head)) {
    SpielFatalError("Q v1 requires nonlinear_value_head=false.");
  }
  if (visit && std::abs(absl::GetFlag(FLAGS_logit_cap) - 10.0) > 1e-12) {
    SpielFatalError("scratch_visit_v1 requires mandatory logit_cap=10.");
  }
  if (absl::GetFlag(FLAGS_intermediate_vp_breadcrumb_weight) < 0.0) {
    SpielFatalError("intermediate_vp_breadcrumb_weight must be nonnegative.");
  }
  if (absl::GetFlag(FLAGS_specimen_exchange_penalty) < 0.0) {
    SpielFatalError("specimen_exchange_penalty must be nonnegative.");
  }
  if (warm) {
    const std::string actual = ComputeFileSHA256(init_model);
    if (!init_sha.empty() && actual != init_sha) {
      SpielFatalError("warm-start model SHA-256 mismatch.");
    }
    init_sha = actual;
  }

  SearchPiConfig search;
  search.scratch_q_v1 = !normalized && !visit;
  search.scratch_visit_v1 = visit;
  search.normalized_cmpo = normalized;
  search.games_per_generation = absl::GetFlag(FLAGS_games_per_generation);
  const int hotspot_profile_games = absl::GetFlag(FLAGS_hotspot_profile_games);
  if (hotspot_profile_games > 0) {
    if (!absl::GetFlag(FLAGS_hotspot_profile) ||
        hotspot_profile_games % 4 != 0) {
      SpielFatalError(
          "hotspot_profile_games requires --hotspot_profile and a multiple of 4.");
    }
    search.games_per_generation = hotspot_profile_games;
  }
  search.next_episode_id = absl::GetFlag(FLAGS_first_episode_id);
  search.seed_domain = absl::GetFlag(FLAGS_seed);
  search.primary_simulations = absl::GetFlag(FLAGS_full_primary_budget);
  search.continuation_simulations = absl::GetFlag(FLAGS_full_other_budget);
  search.purchase_combat_budget = absl::GetFlag(FLAGS_full_other_budget);
  search.scratch_full_root_probability =
      absl::GetFlag(FLAGS_full_root_probability);
  search.scratch_full_primary_simulations =
      absl::GetFlag(FLAGS_full_primary_budget);
  search.scratch_full_other_simulations =
      absl::GetFlag(FLAGS_full_other_budget);
  search.scratch_cheap_primary_simulations =
      absl::GetFlag(FLAGS_cheap_primary_budget);
  search.scratch_cheap_other_simulations =
      absl::GetFlag(FLAGS_cheap_other_budget);
  search.relative_time_budget_ms =
      absl::GetFlag(FLAGS_relative_time_budget_ms);
  search.dirichlet_epsilon = 0.0;
  search.dirichlet_alpha_total = 10.83;
  search.forced_playouts_k = 0.0;
  search.root_noise_fpu_zero = false;
  search.target_sharpen_exponent = 1.0;
  search.behavior_temperature = normalized ? 0.0 : 1.0;
  search.non_search_temperature = 1.0;
  search.searched_seat_unsearched_temperature = 1.0;
  search.search_leader_draft = false;
  search.intermediate_vp_breadcrumb_weight =
      absl::GetFlag(FLAGS_intermediate_vp_breadcrumb_weight);
  search.specimen_exchange_penalty =
      absl::GetFlag(FLAGS_specimen_exchange_penalty);
  search.shaping_start_env_steps =
      absl::GetFlag(FLAGS_shaping_start_env_steps);
  search.shaping_decay_env_steps =
      absl::GetFlag(FLAGS_shaping_decay_env_steps);

  SearchPiLearnerConfig learner;
  learner.learning_rate = absl::GetFlag(FLAGS_learning_rate);
  learner.minibatch_size = absl::GetFlag(FLAGS_minibatch_size);
  learner.epochs = absl::GetFlag(FLAGS_epochs);
  learner.grad_clip_norm = absl::GetFlag(FLAGS_grad_clip_norm);
  learner.logit_cap = absl::GetFlag(FLAGS_logit_cap);
  learner.policy_coef = 1.0;
  learner.value_coef = normalized ? 0.0 : 1.0;
  learner.weight_decay = 0.0;
  learner.policy_weight_decay = 0.0;

  const int hidden = absl::GetFlag(FLAGS_hidden_dim);
  const int blocks = absl::GetFlag(FLAGS_num_blocks);
  const int workers = absl::GetFlag(FLAGS_workers);
  const int batch_target = absl::GetFlag(FLAGS_batch_target);
  const int inference_lanes = absl::GetFlag(FLAGS_inference_lanes);
  const int timeout_ms = absl::GetFlag(FLAGS_batcher_timeout_ms);
  const int chunk_games = absl::GetFlag(FLAGS_chunk_games);
  const size_t replay_max = absl::GetFlag(FLAGS_replay_max_rows);
  const int replay_generations = absl::GetFlag(FLAGS_replay_generations);
  const int required_replay_generations = visit ? 8 : 4;
  if (replay_generations != required_replay_generations) {
    SpielFatalError(visit ? "scratch_visit_v1 replay window is eight generations."
                          : "Q v1 replay window is four generations.");
  }
  if (inference_lanes < 1 || workers % inference_lanes != 0) {
    SpielFatalError("inference_lanes must evenly divide workers.");
  }
  const std::string fingerprint = Fingerprint(
      profile, search, learner, hidden, blocks, false, workers, batch_target,
      timeout_ms, chunk_games, replay_max, replay_generations, init_sha,
      absl::GetFlag(FLAGS_seed), absl::GetFlag(FLAGS_first_episode_id));

  std::filesystem::create_directories(output_dir);
  AtomicJson((std::filesystem::path(output_dir) / "resolved_config.json").string(),
             ResolvedConfig(profile, search, learner, fingerprint, init_model,
                            init_sha, hidden, blocks, false, workers,
                            batch_target, timeout_ms, chunk_games,
                            inference_lanes, replay_max,
                            replay_generations, absl::GetFlag(FLAGS_seed),
                            absl::GetFlag(FLAGS_first_episode_id)));

  at::set_num_threads(1);
  at::set_num_interop_threads(1);
  torch::manual_seed(dune_seed::DeriveSeed(absl::GetFlag(FLAGS_seed),
                                            dune_seed::kStreamModelInit));
  auto game = LoadGame("dune_imperium");
  const int64_t obs_size = game->InformationStateTensorSize();
  const int64_t action_dim = game->NumDistinctActions();
  torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);
  auto student = std::make_shared<SharedDunePolicyValueNetImpl>(
      obs_size, hidden, action_dim, blocks, false);
  student->to(device);
  auto optimizer = MakeOptimizer(student, learner);

  ResumeState state;
  state.next_episode_id = search.next_episode_id;
  state.total_env_steps = 0;
  const std::string resume = absl::GetFlag(FLAGS_resume);
  if (resume.empty()) {
    SaveInitialRandomModelIfNeeded(
        output_dir, profile, fingerprint, absl::GetFlag(FLAGS_seed),
        absl::GetFlag(FLAGS_first_episode_id), !warm, student);
  }
  if (!resume.empty()) {
    state = ReadResume(
        resume, profile, hidden, blocks, absl::GetFlag(FLAGS_seed),
        learner.logit_cap, absl::GetFlag(FLAGS_first_episode_id));
    torch::load(student, state.model_path, device);
    torch::load(*optimizer, state.optimizer_path, device);
    search.next_episode_id = state.next_episode_id;
  } else if (warm) {
    torch::load(student, init_model, device);
  }
  student->eval();
  std::vector<std::vector<SearchPiRow>> replay_window;
  if (!resume.empty()) {
    std::string replay_error;
    replay_window.reserve(state.replay_cohort_groups.size());
    for (const auto& group : state.replay_cohort_groups) {
      replay_window.emplace_back();
      for (const std::string& path : group) {
        std::vector<SearchPiRow> rows;
        if (!ReadScratchSearchPiRowShardV2(path, &rows, &replay_error)) {
          SpielFatalError("resume replay load failed: " + replay_error);
        }
        auto& cohort = replay_window.back();
        cohort.insert(cohort.end(),
                      std::make_move_iterator(rows.begin()),
                      std::make_move_iterator(rows.end()));
      }
    }
  }
  if (absl::GetFlag(FLAGS_verify_resume_only)) {
    if (resume.empty()) SpielFatalError("verify_resume_only requires --resume.");
    std::cout << "resume_verified generation=" << state.generation
              << " next_episode_id=" << state.next_episode_id
              << " replay_paths=" << state.replay_paths.size() << "\n";
    return 0;
  }

  const int target_generation = absl::GetFlag(FLAGS_generations);
  if (state.generation > target_generation) {
    SpielFatalError("resume generation exceeds requested target.");
  }
  for (int generation = state.generation + 1;
       generation <= target_generation; ++generation) {
    // Prefetched games from the preceding generation already occupy part of
    // this generation's fixed 512-game episode budget. They remain replay
    // data, but are never recollected under a duplicate episode ID.
    state.total_env_steps += state.pending_prefetch_env_steps;
    state.pending_prefetch_env_steps = 0;
    const int prefetched_for_current_generation = state.pending_prefetch_games;
    if (prefetched_for_current_generation < 0 ||
        prefetched_for_current_generation > search.games_per_generation) {
      SpielFatalError("invalid pending prefetch game count on resume");
    }
    const int current_collection_games =
        search.games_per_generation - prefetched_for_current_generation;
    if (current_collection_games <= 0) {
      SpielFatalError("prefetch consumed the entire generation game budget");
    }
    state.pending_prefetch_games = 0;
    state.pending_prefetch_collector_generation = -1;
    search.shaping_lambda = ComputeRewardLambda(
        state.total_env_steps, search.shaping_start_env_steps,
        search.shaping_decay_env_steps);
    const std::string student_before =
        visit ? std::string() : CanonicalSearchPiModuleDigest(*student);
    auto collector = std::make_shared<SharedDunePolicyValueNetImpl>(
        obs_size, hidden, action_dim, blocks, false);
    collector->to(device);
    CopySearchPiModel(student, collector);
    collector->eval();
    const std::string collector_digest =
        visit ? std::string() : CanonicalSearchPiModuleDigest(*collector);
    if (!visit) SPIEL_CHECK_EQ(collector_digest, student_before);
    SPIEL_CHECK_FALSE(ScratchCollectorOwnedByOptimizer(collector, *optimizer));

    ConcurrentSearchPiCollectionConfig collection;
    collection.search = search;
    collection.search.games_per_generation = current_collection_games;
    collection.search.next_episode_id = state.next_episode_id;
    collection.generation = generation;
    collection.collection_games = current_collection_games;
    collection.chunk_games = chunk_games;
    collection.requested_workers = workers;
    collection.batch_target = batch_target;
    collection.batcher_timeout_ms = timeout_ms;
    collection.enable_batcher_telemetry =
        absl::GetFlag(FLAGS_batcher_telemetry);
    collection.device_synchronize =
        absl::GetFlag(FLAGS_evaluator_device_synchronize);
    collection.inference_lanes = inference_lanes;
    collection.prefetch_games = absl::GetFlag(FLAGS_prefetch_games);
    collection.prefetch_trigger_running =
        absl::GetFlag(FLAGS_prefetch_trigger_running);
    collection.warmup_games = 0;
    collection.retain_rows = true;
    collection.logit_cap = learner.logit_cap;
    collection.output_dir =
        (std::filesystem::path(output_dir) / "collection" /
            ("generation_" + std::to_string(generation)))
            .string();
    DuneNNEvaluator::ResetLogitCapStats();
    // The collector model remains live, but unused blocks retained by the
    // learning phase are returned before the collection measurement starts.
    // Resetting allocator peaks here makes the recorded VRAM generation-local.
    if (absl::GetFlag(FLAGS_hotspot_profile)) {
      dune_hotspot::Reset();
    }
    PrepareCudaCollection(device);
    const auto collect_start = std::chrono::steady_clock::now();
    ConcurrentSearchPiCollectionResult collected = CollectSearchPiConcurrent(
        collection, game, collector, device);
    const double collection_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - collect_start).count();
    const DuneNNEvaluator::LogitCapAggregate collection_cap_stats =
        DuneNNEvaluator::GetLogitCapStats();
    const CudaMemorySnapshot collection_cuda_memory = ReadCudaMemory(device);
    const dune_hotspot::Snapshot hotspot_snapshot = dune_hotspot::Read();
    // The collector is immutable and no longer needed once collection has
    // returned. Release it before the learner, then return its freed blocks so
    // minibatch streaming starts with the student/optimizer footprint only.
    collector.reset();
    ClearUnusedCudaCache(device);
    ResetLearnerCudaPeak(device);
    NormalizedCmpoGenerationStats normalized_stats;
    ScratchVisitGenerationStats visit_stats;
    if (visit) {
      std::string target_error;
      if (!ApplyScratchVisitTargets(&collected.retained_rows, &visit_stats,
                                    &target_error)) {
        SpielFatalError("visit-policy target construction failed: " +
                        target_error);
      }
    } else if (normalized) {
      std::string target_error;
      if (!ApplyNormalizedCmpoTargets(&collected.retained_rows,
                                      &normalized_stats, &target_error)) {
        SpielFatalError("normalized CMPO target construction failed: " +
                        target_error);
      }
    }
    ValidateCollection(collected, collection.search);
    collector.reset();

    const std::filesystem::path shard_dir =
        std::filesystem::path(output_dir) / "shards";
    std::filesystem::create_directories(shard_dir);
    const std::string shard_path =
        ScratchSearchPiShardPathForGeneration(shard_dir.string(), generation);
    replay_window.push_back(std::move(collected.retained_rows));
    const size_t current_cohort_index = replay_window.size() - 1;
    state.replay_paths.push_back(shard_path);
    if (prefetched_for_current_generation > 0 &&
        !state.replay_cohort_groups.empty()) {
      // The prefetch shard and this current-generation shard are one logical
      // replay cohort. Keep both immutable files, but count them as one
      // generation for the eight-generation retention horizon.
      state.replay_cohort_groups.back().push_back(shard_path);
    } else {
      state.replay_cohort_groups.push_back({shard_path});
    }
    while (state.replay_cohort_groups.size() >
           static_cast<size_t>(replay_generations)) {
      const size_t files_in_oldest = state.replay_cohort_groups.front().size();
      state.replay_cohort_groups.erase(state.replay_cohort_groups.begin());
      for (size_t i = 0; i < files_in_oldest; ++i) {
        state.replay_paths.erase(state.replay_paths.begin());
        replay_window.erase(replay_window.begin());
      }
    }
    const std::vector<SearchPiRow>& current_rows = replay_window.back();
    const int prefetched_rows_count =
        static_cast<int>(collected.prefetched_rows.size());
    const int prefetched_games_count =
        static_cast<int>(collected.prefetched_games.size());
    std::vector<SearchPiRow> prefetched_rows =
        std::move(collected.prefetched_rows);
    const std::string prefetch_shard_path =
        (shard_dir / ("generation_" + std::to_string(generation + 1) +
                      "_prefetch_from_" + std::to_string(generation) +
                      ".bin"))
            .string();
    auto shard_write = std::async(
        std::launch::async, [&current_rows, shard_path]() {
          std::string error;
          const bool ok = WriteScratchSearchPiRowShardV2(
              shard_path, current_rows, &error);
          return std::make_pair(ok, error);
        });
    std::future<std::pair<bool, std::string>> prefetch_shard_write;
    if (!prefetched_rows.empty()) {
      prefetch_shard_write = std::async(
          std::launch::async,
          [&prefetched_rows, prefetch_shard_path]() {
            std::string error;
            const bool ok = WriteScratchSearchPiRowShardV2(
                prefetch_shard_path, prefetched_rows, &error);
            return std::make_pair(ok, error);
          });
    }
    std::string shard_error;
    std::vector<std::vector<SearchPiRow>> learner_window;
    const std::vector<std::vector<SearchPiRow>>* sampling_window =
        &replay_window;
    if (normalized) {
      learner_window = FilterNormalizedCmpoReplayWindow(replay_window);
      sampling_window = &learner_window;
    }
    std::vector<SearchPiRow> learner_rows = SampleScratchSearchPiReplayWindow(
        *sampling_window, generation, replay_max,
        dune_seed::DeriveSeed(absl::GetFlag(FLAGS_seed), generation),
        &shard_error);
    if (!shard_error.empty()) SpielFatalError("replay selection failed: " + shard_error);

    const auto learner_start = std::chrono::steady_clock::now();
    ScratchSearchPiLearnerStats learner_stats = RunScratchSearchPiLearner(
        student, *optimizer, learner_rows, obs_size, action_dim, device,
        absl::GetFlag(FLAGS_seed), generation, learner);
    const double learner_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - learner_start).count();
    const CudaMemorySnapshot learner_cuda_memory = ReadCudaMemory(device);
    SPIEL_CHECK_TRUE(learner_stats.policy_backward_executed);
    if (normalized) {
      SPIEL_CHECK_FALSE(learner_stats.value_backward_executed);
    } else {
      SPIEL_CHECK_TRUE(learner_stats.value_backward_executed);
    }
    const std::string student_after =
        visit ? std::string() : CanonicalSearchPiModuleDigest(*student);
    if (!visit) SPIEL_CHECK_NE(student_before, student_after);

    const auto shard_result = shard_write.get();
    if (!shard_result.first) {
      SpielFatalError("v2 shard write failed: " + shard_result.second);
    }
    if (!prefetched_rows.empty()) {
      const auto prefetch_shard_result = prefetch_shard_write.get();
      if (!prefetch_shard_result.first) {
        SpielFatalError("prefetch v2 shard write failed: " +
                        prefetch_shard_result.second);
      }
      replay_window.push_back(std::move(prefetched_rows));
      state.replay_paths.push_back(prefetch_shard_path);
      state.replay_cohort_groups.push_back({prefetch_shard_path});
      while (state.replay_cohort_groups.size() >
             static_cast<size_t>(replay_generations)) {
        const size_t files_in_oldest =
            state.replay_cohort_groups.front().size();
        state.replay_cohort_groups.erase(
            state.replay_cohort_groups.begin());
        for (size_t i = 0; i < files_in_oldest; ++i) {
          state.replay_paths.erase(state.replay_paths.begin());
          replay_window.erase(replay_window.begin());
        }
      }
    state.pending_prefetch_games =
          prefetched_games_count;
      state.pending_prefetch_collector_generation =
          collected.prefetch_collector_generation;
      for (const SearchPiGameOutcome& outcome : collected.prefetched_games) {
        state.pending_prefetch_env_steps +=
            static_cast<uint64_t>(outcome.trajectory_transitions);
      }
    }

    int64_t generation_env_steps = 0;
    int64_t generation_vp_breadcrumbs = 0;
    int64_t generation_specimen_breadcrumbs = 0;
    double generation_shaping_reward = 0.0;
    double generation_vp_raw_reward = 0.0;
    double generation_specimen_raw_penalty = 0.0;
    double generation_vp_shaped_reward = 0.0;
    double generation_specimen_shaped_reward = 0.0;
    for (const SearchPiGameOutcome& outcome : collected.games) {
      generation_env_steps += outcome.trajectory_transitions;
      generation_vp_breadcrumbs += outcome.vp_breadcrumb_events;
      generation_specimen_breadcrumbs +=
          outcome.specimen_anti_breadcrumb_events;
      generation_shaping_reward += outcome.shaping_reward_sum;
      generation_vp_raw_reward += outcome.vp_breadcrumb_raw_reward_sum;
      generation_specimen_raw_penalty +=
          outcome.specimen_anti_breadcrumb_raw_penalty_sum;
      generation_vp_shaped_reward +=
          outcome.vp_breadcrumb_shaped_reward_sum;
      generation_specimen_shaped_reward +=
          outcome.specimen_anti_breadcrumb_shaped_reward_sum;
    }
    state.total_env_steps += static_cast<uint64_t>(generation_env_steps);
    state.generation = generation;
    state.next_episode_id += current_collection_games +
                             state.pending_prefetch_games;
    const bool save_checkpoint = generation % 5 == 0 || target_generation < 5;
    json::Object checkpoint;
    if (save_checkpoint) {
      checkpoint = SaveCheckpoint(
          output_dir, profile, fingerprint, generation, state.next_episode_id,
          state.total_env_steps, state.replay_paths,
          state.replay_cohort_groups,
          state.pending_prefetch_games,
          state.pending_prefetch_collector_generation,
          state.pending_prefetch_env_steps, student, *optimizer);
    }

    int64_t full_rows = 0;
    int64_t cheap_rows = 0;
    double regularized_entropy = 0.0;
    double visit_entropy = 0.0;
    double raw_entropy = 0.0;
    double kl = 0.0;
    double beta = 0.0;
    double q_range = 0.0;
    double q_improvement = 0.0;
    int64_t direct_visits = 0;
    // A prefetch cohort may have been appended after the learner. Reacquire
    // the current vector by index because that append can reallocate the
    // replay-window outer vector and invalidate the earlier reference.
    const std::vector<SearchPiRow>& telemetry_current_rows =
        replay_window[current_cohort_index];
    for (const SearchPiRow& row : telemetry_current_rows) {
      (row.search_budget_class == "full" ? full_rows : cheap_rows)++;
      if (normalized && row.search_budget_class != "full") continue;
      regularized_entropy += row.regularized_q_entropy_norm;
      visit_entropy += row.target_entropy_norm;
      raw_entropy += row.raw_policy_entropy_norm;
      kl += row.regularized_q_kl;
      beta += row.regularized_q_beta;
      q_range += row.regularized_q_q_range;
      q_improvement += row.regularized_q_target_expected_q -
                       row.regularized_q_prior_expected_q;
      direct_visits += row.regularized_q_direct_visit_count;
    }
    const double row_den = std::max<int64_t>(
        1, normalized ? full_rows : collected.rows_total);
    json::Object telemetry;
    telemetry["schema_version"] = int64_t{1};
    telemetry["profile"] = profile;
    telemetry["generation"] = int64_t{generation};
    telemetry["games"] = static_cast<int64_t>(collected.games.size());
    telemetry["rows"] = collected.rows_total;
    telemetry["independent_terminal_outcomes"] =
        static_cast<int64_t>(collected.games.size());
    int64_t simulator_transitions = 0;
    for (const SearchPiGameOutcome& game_outcome : collected.games) {
      simulator_transitions += game_outcome.trajectory_transitions;
    }
    telemetry["simulator_transitions"] = simulator_transitions;
    telemetry["neural_network_calls"] = collected.inference_calls;
    telemetry["collection_wall_time_s"] = collected.wall_time_s;
    telemetry["main_collection_wall_time_s"] =
        collected.main_collection_wall_time_s;
    telemetry["tail_wall_time_s"] = collected.tail_wall_time_s;
    telemetry["main_games_completed_before_tail"] =
        static_cast<int64_t>(collected.main_games_completed_before_tail);
    telemetry["tail_prefetch_triggered"] = collected.prefetch_triggered;
    telemetry["current_collection_games"] =
        static_cast<int64_t>(collected.games.size());
    telemetry["prefetched_games_for_current_generation"] =
        static_cast<int64_t>(prefetched_for_current_generation);
    telemetry["generation_game_equivalents"] =
        static_cast<int64_t>(collected.games.size() +
                             prefetched_for_current_generation);
    telemetry["prefetched_games"] =
        static_cast<int64_t>(prefetched_games_count);
    telemetry["prefetched_rows"] =
        static_cast<int64_t>(prefetched_rows_count);
    telemetry["prefetch_collector_generation"] =
        static_cast<int64_t>(collected.prefetch_collector_generation);
    telemetry["prefetch_for_generation"] =
        static_cast<int64_t>(collected.prefetch_for_generation);
    if (absl::GetFlag(FLAGS_hotspot_profile)) {
      auto hotspot_ms = [&](dune_hotspot::Kind kind) {
        return static_cast<double>(hotspot_snapshot.nanoseconds[
            static_cast<size_t>(kind)]) /
               1.0e6;
      };
      auto hotspot_calls = [&](dune_hotspot::Kind kind) {
        return static_cast<int64_t>(hotspot_snapshot.calls[
            static_cast<size_t>(kind)]);
      };
      telemetry["hotspot_state_clone_ms"] =
          hotspot_ms(dune_hotspot::Kind::kStateClone);
      telemetry["hotspot_state_clone_calls"] =
          hotspot_calls(dune_hotspot::Kind::kStateClone);
      telemetry["hotspot_legal_actions_ms"] =
          hotspot_ms(dune_hotspot::Kind::kLegalActions);
      telemetry["hotspot_legal_actions_calls"] =
          hotspot_calls(dune_hotspot::Kind::kLegalActions);
      telemetry["hotspot_game_transitions_ms"] =
          hotspot_ms(dune_hotspot::Kind::kGameTransition);
      telemetry["hotspot_game_transitions_calls"] =
          hotspot_calls(dune_hotspot::Kind::kGameTransition);
      telemetry["hotspot_observation_construction_ms"] =
          hotspot_ms(dune_hotspot::Kind::kObservationConstruction);
      telemetry["hotspot_observation_construction_calls"] =
          hotspot_calls(dune_hotspot::Kind::kObservationConstruction);
      telemetry["hotspot_mcts_search_ms"] =
          hotspot_ms(dune_hotspot::Kind::kMctsSearch);
      telemetry["hotspot_mcts_search_calls"] =
          hotspot_calls(dune_hotspot::Kind::kMctsSearch);
      telemetry["hotspot_batcher_result_distribution_ms"] =
          hotspot_ms(dune_hotspot::Kind::kBatcherResultDistribution);
      telemetry["hotspot_batcher_result_distribution_calls"] =
          hotspot_calls(dune_hotspot::Kind::kBatcherResultDistribution);
    }
    telemetry["batcher_telemetry_valid"] = collected.telemetry_valid;
    telemetry["batcher_mean_batch_size"] =
        collected.telemetry.mean_batch_size;
    telemetry["batcher_target_occupancy"] =
        collected.telemetry.target_occupancy;
    telemetry["batcher_timeout_flush_batches"] =
        static_cast<int64_t>(collected.telemetry.timeout_flush_batches);
    telemetry["batcher_h2d_ms"] = collected.telemetry.h2d_ms;
    telemetry["batcher_forward_ms"] = collected.telemetry.forward_ms;
    telemetry["batcher_output_cast_ms"] =
        collected.telemetry.output_cast_ms;
    telemetry["batcher_d2h_ms"] = collected.telemetry.d2h_ms;
    telemetry["batcher_sync_ms"] = collected.telemetry.sync_ms;
    telemetry["batcher_device_timed_batches"] =
        static_cast<int64_t>(collected.telemetry.device_timed_batches);
    telemetry["inference_lanes"] =
        static_cast<int64_t>(collected.configured_inference_lanes);
    json::Array lane_telemetry;
    for (size_t lane = 0; lane < collected.lane_telemetry.size(); ++lane) {
      const BatcherTelemetry& lane_stats = collected.lane_telemetry[lane];
      json::Object lane_object;
      lane_object["lane"] = static_cast<int64_t>(lane);
      lane_object["workers"] = static_cast<int64_t>(
          lane < collected.lane_worker_counts.size()
              ? collected.lane_worker_counts[lane]
              : 0);
      lane_object["target_batch_size"] =
          static_cast<int64_t>(lane_stats.target_batch_size);
      lane_object["submitted_rows"] =
          static_cast<int64_t>(lane_stats.submitted_rows);
      lane_object["physical_batches"] =
          static_cast<int64_t>(lane_stats.physical_batches);
      lane_object["mean_batch_size"] = lane_stats.mean_batch_size;
      lane_object["target_occupancy"] = lane_stats.target_occupancy;
      lane_object["timeout_flush_batches"] =
          static_cast<int64_t>(lane_stats.timeout_flush_batches);
      lane_object["device_timed_batches"] =
          static_cast<int64_t>(lane_stats.device_timed_batches);
      lane_telemetry.push_back(std::move(lane_object));
    }
    telemetry["batcher_lane_telemetry"] = std::move(lane_telemetry);
    telemetry["batcher_telemetry_enabled"] =
        absl::GetFlag(FLAGS_batcher_telemetry);
    telemetry["evaluator_device_synchronize"] =
        absl::GetFlag(FLAGS_evaluator_device_synchronize);
    telemetry["cuda_cache_cleared_before_collection"] = device.is_cuda();
    telemetry["gpu_memory_allocated_after_collection_bytes"] =
        collection_cuda_memory.allocated_bytes;
    telemetry["gpu_memory_reserved_after_collection_bytes"] =
        collection_cuda_memory.reserved_bytes;
    telemetry["gpu_peak_memory_allocated_bytes"] =
        collection_cuda_memory.peak_allocated_bytes;
    telemetry["gpu_peak_memory_reserved_bytes"] =
        collection_cuda_memory.peak_reserved_bytes;
    telemetry["full_rows"] = full_rows;
    telemetry["cheap_rows"] = cheap_rows;
    telemetry["requested_simulations"] =
        collected.primary.simulations_requested +
        collected.continuation.simulations_requested +
        collected.purchase.simulations_requested +
        collected.combat_intrigue.simulations_requested +
        collected.other_optional.simulations_requested;
    telemetry["completed_simulations"] =
        collected.primary.simulations_completed +
        collected.continuation.simulations_completed +
        collected.purchase.simulations_completed +
        collected.combat_intrigue.simulations_completed +
        collected.other_optional.simulations_completed;
    int64_t fallback_count = collected.primary.fallbacks +
                             collected.continuation.fallbacks +
                             collected.purchase.fallbacks +
                             collected.combat_intrigue.fallbacks +
                             collected.other_optional.fallbacks;
    int64_t watchdog_timeouts = 0;
    int64_t non_timeout_early_exits = 0;
    for (const SearchPiGameOutcome& outcome : collected.games) {
      watchdog_timeouts += outcome.watchdog_timeouts;
      non_timeout_early_exits += outcome.non_timeout_early_exits;
    }
    telemetry["fallbacks"] = fallback_count;
    telemetry["watchdog_timeouts"] = watchdog_timeouts;
    telemetry["non_timeout_early_exits"] = non_timeout_early_exits;
    telemetry["completed_simulations_per_wall_second"] =
        collected.wall_time_s > 0.0
            ? static_cast<double>(
                  collected.primary.simulations_completed +
                  collected.continuation.simulations_completed +
                  collected.purchase.simulations_completed +
                  collected.combat_intrigue.simulations_completed +
                  collected.other_optional.simulations_completed) /
                  collected.wall_time_s
            : 0.0;
    const SearchPiRoleStats* latency_roles[] = {
        &collected.primary, &collected.continuation, &collected.purchase,
        &collected.combat_intrigue, &collected.other_optional};
    double max_root_time_ms = 0.0;
    double p95_root_time_ms = 0.0;
    for (const SearchPiRoleStats* role_stats : latency_roles) {
      max_root_time_ms = std::max(max_root_time_ms,
                                  role_stats->max_search_elapsed_ms);
      if (role_stats->searches_run > 0) {
        p95_root_time_ms = std::max(
            p95_root_time_ms,
            SearchPiLatencyQuantileUpperMs(*role_stats, 0.95));
      }
    }
    telemetry["max_root_time_ms"] = max_root_time_ms;
    telemetry["p95_root_time_ms"] = p95_root_time_ms;
    telemetry["raw_entropy_mean"] = raw_entropy / row_den;
    telemetry["visit_entropy_mean"] = visit_entropy / row_den;
    telemetry["regularized_q_entropy_mean"] = regularized_entropy / row_den;
    telemetry["regularized_q_kl_mean"] = kl / row_den;
    telemetry["selected_beta_mean"] = beta / row_den;
    telemetry["q_range_mean"] = q_range / row_den;
    telemetry["expected_q_improvement_mean"] = q_improvement / row_den;
    telemetry["direct_visit_count_mean"] = direct_visits / row_den;
    if (normalized) {
      std::vector<double> sorted_kls = normalized_stats.target_kls;
      std::sort(sorted_kls.begin(), sorted_kls.end());
      auto quantile = [&sorted_kls](double q) {
        if (sorted_kls.empty()) return 0.0;
        const double position = q * static_cast<double>(sorted_kls.size() - 1);
        const size_t lower = static_cast<size_t>(position);
        const size_t upper = std::min(sorted_kls.size() - 1, lower + 1);
        const double fraction = position - static_cast<double>(lower);
        return sorted_kls[lower] * (1.0 - fraction) +
               sorted_kls[upper] * fraction;
      };
      telemetry["normalized_cmpo_sigma"] = normalized_stats.sigma;
      telemetry["normalized_cmpo_target_rows"] =
          static_cast<int64_t>(normalized_stats.target_kls.size());
      telemetry["normalized_cmpo_target_kl_mean"] =
          normalized_stats.target_kls.empty()
              ? 0.0
              : std::accumulate(normalized_stats.target_kls.begin(),
                                normalized_stats.target_kls.end(), 0.0) /
                    static_cast<double>(normalized_stats.target_kls.size());
      telemetry["normalized_cmpo_target_kl_median"] = quantile(0.50);
      telemetry["normalized_cmpo_target_kl_p95"] = quantile(0.95);
      telemetry["normalized_cmpo_clip_low_fraction"] =
          normalized_stats.advantage_count == 0
              ? 0.0
              : static_cast<double>(normalized_stats.clipped_low) /
                    static_cast<double>(normalized_stats.advantage_count);
      telemetry["normalized_cmpo_clip_high_fraction"] =
          normalized_stats.advantage_count == 0
              ? 0.0
              : static_cast<double>(normalized_stats.clipped_high) /
                    static_cast<double>(normalized_stats.advantage_count);
      telemetry["normalized_cmpo_prior_expected_q_mean"] =
          normalized_stats.prior_expected_q_mean;
      telemetry["normalized_cmpo_target_expected_q_mean"] =
          normalized_stats.target_expected_q_mean;
      telemetry["normalized_cmpo_q_range_mean"] = normalized_stats.q_range_mean;
    }
    if (visit) {
      telemetry["visit_policy_target_rows"] = visit_stats.full_rows;
      telemetry["visit_policy_cheap_rows"] = visit_stats.cheap_rows;
      telemetry["visit_policy_target_entropy_mean"] =
          visit_stats.target_entropy_mean;
      telemetry["visit_policy_target_kl_mean"] = visit_stats.target_kl_mean;
      telemetry["visit_policy_prior_expected_q_mean"] =
          visit_stats.prior_expected_q_mean;
      telemetry["visit_policy_target_expected_q_mean"] =
          visit_stats.target_expected_q_mean;
      telemetry["visit_policy_q_range_mean"] = visit_stats.q_range_mean;
      telemetry["visit_policy_direct_visit_count_mean"] =
          visit_stats.direct_visit_count_mean;
    }
    telemetry["policy_ce"] = learner_stats.policy_ce;
    telemetry["value_mse"] = learner_stats.value_mse;
    telemetry["policy_grad_norm"] = learner_stats.policy_grad_norm;
    telemetry["value_grad_norm"] = learner_stats.value_grad_norm;
    telemetry["shared_trunk_cosine"] = learner_stats.shared_trunk_cosine;
    telemetry["value_prediction_mean"] = learner_stats.critic_pred_mean_post;
    telemetry["value_prediction_sd"] = learner_stats.critic_pred_sd_post;
    telemetry["value_saturation_rate"] = learner_stats.critic_saturation_post;
    telemetry["replay_rows"] = learner_stats.distinct_rows;
    telemetry["learner_steps"] = learner_stats.minibatches;
    telemetry["collection_seconds"] = collection_seconds;
    telemetry["learner_seconds"] = learner_seconds;
    telemetry["learner_streamed_minibatches"] = true;
    telemetry["learner_peak_memory_allocated_bytes"] =
        learner_cuda_memory.peak_allocated_bytes;
    telemetry["learner_peak_memory_reserved_bytes"] =
        learner_cuda_memory.peak_reserved_bytes;
    telemetry["learner_memory_allocated_after_update_bytes"] =
        learner_cuda_memory.allocated_bytes;
    telemetry["learner_memory_reserved_after_update_bytes"] =
        learner_cuda_memory.reserved_bytes;
    telemetry["student_before_digest"] = student_before;
    telemetry["collector_digest"] = collector_digest;
    telemetry["student_after_digest"] = student_after;
    telemetry["shaping_lambda"] = static_cast<double>(search.shaping_lambda);
    telemetry["total_env_steps"] =
        static_cast<int64_t>(state.total_env_steps);
    telemetry["generation_env_steps"] = generation_env_steps;
    telemetry["vp_breadcrumb_events"] = generation_vp_breadcrumbs;
    telemetry["specimen_anti_breadcrumb_events"] =
        generation_specimen_breadcrumbs;
    telemetry["shaping_reward_sum"] = generation_shaping_reward;
    telemetry["vp_breadcrumb_raw_reward_sum"] = generation_vp_raw_reward;
    telemetry["specimen_anti_breadcrumb_raw_penalty_sum"] =
        generation_specimen_raw_penalty;
    telemetry["vp_breadcrumb_shaped_reward_sum"] =
        generation_vp_shaped_reward;
    telemetry["specimen_anti_breadcrumb_shaped_reward_sum"] =
        generation_specimen_shaped_reward;
    telemetry["logit_cap"] = learner.logit_cap;
    telemetry["collector_precap_max_legal_logit"] =
        collection_cap_stats.pre_max_abs;
    telemetry["collector_postcap_max_legal_logit"] =
        collection_cap_stats.post_max_abs;
    telemetry["collector_logit_cap_decisions"] =
        static_cast<int64_t>(collection_cap_stats.decisions);
    telemetry["collector_logit_cap_saturated"] =
        static_cast<int64_t>(collection_cap_stats.saturated);
    telemetry["collector_logit_cap_legal_logits"] =
        static_cast<int64_t>(collection_cap_stats.legal_logits);
    telemetry["collector_logit_cap_saturation_rate"] =
        collection_cap_stats.legal_logits == 0
            ? 0.0
            : static_cast<double>(collection_cap_stats.saturated) /
                  static_cast<double>(collection_cap_stats.legal_logits);
    telemetry["learner_precap_max_legal_logit"] =
        learner_stats.learner_precap_max_legal_logit;
    telemetry["learner_postcap_max_legal_logit"] =
        learner_stats.learner_postcap_max_legal_logit;
    telemetry["learner_logit_cap_decisions"] =
        learner_stats.learner_logit_cap_decisions;
    telemetry["learner_logit_cap_saturated"] =
        learner_stats.learner_logit_cap_saturated;
    telemetry["learner_logit_cap_legal_logits"] =
        learner_stats.learner_logit_cap_legal_logits;
    telemetry["learner_logit_cap_saturation_rate"] =
        learner_stats.learner_logit_cap_legal_logits == 0
            ? 0.0
            : static_cast<double>(learner_stats.learner_logit_cap_saturated) /
                  static_cast<double>(learner_stats.learner_logit_cap_legal_logits);
    telemetry["checkpoint"] = checkpoint;
    telemetry["checkpoint_saved"] = save_checkpoint;
    AtomicJson((std::filesystem::path(output_dir) /
                ("generation_" + std::to_string(generation) + ".json"))
                   .string(),
               telemetry);
    std::cout << "completed_generation=" << generation
              << " games=" << collected.games.size()
              << " rows=" << collected.rows_total
              << " full=" << full_rows << " cheap=" << cheap_rows
              << " collection_s=" << collection_seconds
              << " learner_s=" << learner_seconds << "\n";
  }
  return 0;
}
