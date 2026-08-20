#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

#include <torch/torch.h>
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
ABSL_FLAG(int, batcher_timeout_ms, 2, "Inference batch timeout.");
ABSL_FLAG(int, chunk_games, 4, "Seat-balanced games per work chunk.");
ABSL_FLAG(double, full_root_probability, 0.25, "Full-search root chance.");
ABSL_FLAG(int, full_primary_budget, 200, "Full primary simulations.");
ABSL_FLAG(int, full_other_budget, 64, "Full non-primary simulations.");
ABSL_FLAG(int, cheap_primary_budget, 32, "Cheap primary simulations.");
ABSL_FLAG(int, cheap_other_budget, 16, "Cheap non-primary simulations.");
ABSL_FLAG(double, relative_time_budget_ms, 60000.0,
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
};

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
                       const std::string& fingerprint) {
  auto parsed = json::FromString(ReadText(path));
  if (!parsed.has_value() || !parsed->IsObject()) {
    SpielFatalError("scratch resume is not a JSON object.");
  }
  const json::Object& object = parsed->GetObject();
  if (object.at("profile").GetString() != profile) {
    SpielFatalError("historical/PPO checkpoint cannot resume this Q profile.");
  }
  if (object.at("config_fingerprint").GetString() != fingerprint) {
    SpielFatalError("Q-profile resume config fingerprint mismatch.");
  }
  ResumeState state;
  state.generation = object.at("generation").GetInt();
  state.next_episode_id = object.at("next_episode_id").GetInt();
  state.total_env_steps = static_cast<uint64_t>(
      object.at("total_env_steps").GetInt());
  state.model_path = object.at("model_checkpoint").GetString();
  state.optimizer_path = object.at("optimizer_checkpoint").GetString();
  state.model_sha256 = object.at("model_sha256").GetString();
  state.optimizer_sha256 = object.at("optimizer_sha256").GetString();
  state.student_digest = object.at("student_digest").GetString();
  for (const json::Value& value : object.at("replay_paths").GetArray()) {
    state.replay_paths.push_back(value.GetString());
  }
  for (const json::Value& value : object.at("replay_sha256").GetArray()) {
    state.replay_sha256.push_back(value.GetString());
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
                            size_t replay_max, int replay_generations,
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
  checkpoint["model_sha256"] = ComputeFileSHA256(model_path);
  checkpoint["optimizer_sha256"] = ComputeFileSHA256(optimizer_path);
  checkpoint["student_digest"] = CanonicalSearchPiModuleDigest(*model);
  json::Array paths;
  json::Array replay_hashes;
  for (const std::string& path : replay_paths) {
    paths.push_back(path);
    replay_hashes.push_back(ComputeFileSHA256(path));
  }
  checkpoint["replay_paths"] = std::move(paths);
  checkpoint["replay_sha256"] = std::move(replay_hashes);
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
  const int timeout_ms = absl::GetFlag(FLAGS_batcher_timeout_ms);
  const int chunk_games = absl::GetFlag(FLAGS_chunk_games);
  const size_t replay_max = absl::GetFlag(FLAGS_replay_max_rows);
  const int replay_generations = absl::GetFlag(FLAGS_replay_generations);
  const int required_replay_generations = visit ? 8 : 4;
  if (replay_generations != required_replay_generations) {
    SpielFatalError(visit ? "scratch_visit_v1 replay window is eight generations."
                          : "Q v1 replay window is four generations.");
  }
  const std::string fingerprint = Fingerprint(
      profile, search, learner, hidden, blocks, false, workers, batch_target,
      timeout_ms, chunk_games, replay_max, replay_generations, init_sha,
      absl::GetFlag(FLAGS_seed), absl::GetFlag(FLAGS_first_episode_id));

  std::filesystem::create_directories(output_dir);
  AtomicJson((std::filesystem::path(output_dir) / "resolved_config.json").string(),
             ResolvedConfig(profile, search, learner, fingerprint, init_model,
                            init_sha, hidden, blocks, false, workers,
                            batch_target, timeout_ms, chunk_games, replay_max,
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
  const std::filesystem::path generation_zero =
      std::filesystem::path(output_dir) / "checkpoints/generation_0_model.pt";
  if (!warm && !resume.empty() && !std::filesystem::exists(generation_zero)) {
    SpielFatalError("resume is missing the immutable generation-zero model.");
  }
  SaveInitialRandomModelIfNeeded(
      output_dir, profile, fingerprint, absl::GetFlag(FLAGS_seed),
      absl::GetFlag(FLAGS_first_episode_id), !warm, student);
  if (!resume.empty()) {
    state = ReadResume(resume, profile, fingerprint);
    torch::load(student, state.model_path, device);
    torch::load(*optimizer, state.optimizer_path, device);
    if (CanonicalSearchPiModuleDigest(*student) != state.student_digest) {
      SpielFatalError("resume model digest validation failed.");
    }
    search.next_episode_id = state.next_episode_id;
    for (const std::string& path : state.replay_paths) {
      std::vector<SearchPiRow> rows;
      std::string error;
      if (!ReadScratchSearchPiRowShardV2(path, &rows, &error)) {
        SpielFatalError("resume replay validation failed: " + error);
      }
    }
    if (state.model_path.empty() || state.optimizer_path.empty() ||
        ComputeFileSHA256(state.model_path) != state.model_sha256 ||
        ComputeFileSHA256(state.optimizer_path) != state.optimizer_sha256) {
      SpielFatalError("resume checkpoint SHA-256 validation failed.");
    }
    if (state.replay_sha256.size() != state.replay_paths.size()) {
      SpielFatalError("resume lacks replay SHA-256 entries.");
    }
    if (visit && state.replay_paths.size() !=
                     std::min<size_t>(state.generation,
                                      static_cast<size_t>(replay_generations))) {
      SpielFatalError("resume replay window is incomplete for its generation.");
    }
    if (!state.replay_sha256.empty()) {
      for (size_t i = 0; i < state.replay_paths.size(); ++i) {
        if (ComputeFileSHA256(state.replay_paths[i]) != state.replay_sha256[i]) {
          SpielFatalError("resume replay SHA-256 validation failed.");
        }
      }
    }
    const int64_t expected_episode =
        absl::GetFlag(FLAGS_first_episode_id) +
        static_cast<int64_t>(state.generation) *
            static_cast<int64_t>(absl::GetFlag(FLAGS_games_per_generation));
    if (state.next_episode_id != expected_episode) {
      SpielFatalError("resume episode cursor does not match generation geometry.");
    }
  } else if (warm) {
    torch::load(student, init_model, device);
  }
  student->eval();
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
    search.shaping_lambda = ComputeRewardLambda(
        state.total_env_steps, search.shaping_start_env_steps,
        search.shaping_decay_env_steps);
    const std::string student_before = CanonicalSearchPiModuleDigest(*student);
    auto collector = std::make_shared<SharedDunePolicyValueNetImpl>(
        obs_size, hidden, action_dim, blocks, false);
    collector->to(device);
    CopySearchPiModel(student, collector);
    collector->eval();
    const std::string collector_digest =
        CanonicalSearchPiModuleDigest(*collector);
    SPIEL_CHECK_EQ(collector_digest, student_before);
    SPIEL_CHECK_FALSE(ScratchCollectorOwnedByOptimizer(collector, *optimizer));

    ConcurrentSearchPiCollectionConfig collection;
    collection.search = search;
    collection.search.next_episode_id = state.next_episode_id;
    collection.generation = generation;
    collection.collection_games = search.games_per_generation;
    collection.chunk_games = chunk_games;
    collection.requested_workers = workers;
    collection.batch_target = batch_target;
    collection.batcher_timeout_ms = timeout_ms;
    collection.warmup_games = 0;
    collection.retain_rows = true;
    collection.logit_cap = learner.logit_cap;
    collection.output_dir =
        (std::filesystem::path(output_dir) / "collection" /
            ("generation_" + std::to_string(generation)))
            .string();
    DuneNNEvaluator::ResetLogitCapStats();
    const auto collect_start = std::chrono::steady_clock::now();
    ConcurrentSearchPiCollectionResult collected = CollectSearchPiConcurrent(
        collection, game, collector, device);
    const double collection_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - collect_start).count();
    const DuneNNEvaluator::LogitCapAggregate collection_cap_stats =
        DuneNNEvaluator::GetLogitCapStats();
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
    std::string shard_error;
    if (!WriteScratchSearchPiRowShardV2(shard_path, collected.retained_rows,
                                        &shard_error)) {
      SpielFatalError("v2 shard write failed: " + shard_error);
    }
    std::vector<SearchPiRow> round_trip;
    if (!ReadScratchSearchPiRowShardV2(shard_path, &round_trip, &shard_error) ||
        round_trip.size() != collected.retained_rows.size()) {
      SpielFatalError("v2 shard round-trip failed: " + shard_error);
    }
    state.replay_paths.push_back(shard_path);
    while (state.replay_paths.size() > static_cast<size_t>(replay_generations)) {
      state.replay_paths.erase(state.replay_paths.begin());
    }
    std::vector<std::vector<SearchPiRow>> replay_window;
    for (const std::string& path : state.replay_paths) {
      replay_window.emplace_back();
      if (!ReadScratchSearchPiRowShardV2(path, &replay_window.back(),
                                         &shard_error)) {
        SpielFatalError("v2 replay read failed: " + shard_error);
      }
    }
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
    SPIEL_CHECK_TRUE(learner_stats.policy_backward_executed);
    if (normalized) {
      SPIEL_CHECK_FALSE(learner_stats.value_backward_executed);
    } else {
      SPIEL_CHECK_TRUE(learner_stats.value_backward_executed);
    }
    const std::string student_after = CanonicalSearchPiModuleDigest(*student);
    SPIEL_CHECK_NE(student_before, student_after);

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
    state.next_episode_id += search.games_per_generation;
    json::Object checkpoint = SaveCheckpoint(
        output_dir, profile, fingerprint, generation, state.next_episode_id,
        state.total_env_steps,
        state.replay_paths, student, *optimizer);

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
    for (const SearchPiRow& row : collected.retained_rows) {
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
