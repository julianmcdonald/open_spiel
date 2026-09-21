#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <memory>
#include <chrono>
#include <filesystem>
#include <shared_mutex>
#include <iomanip>
#include <ctime>

#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"
#include "open_spiel/utils/json.h"
#include "open_spiel/games/dune_imperium/dune_imperium.h"
#include "open_spiel/games/dune_imperium/dune_imperium_common.h"
#include "absl/flags/flag.h"
#include "absl/flags/parse.h"

#include <torch/torch.h>

#include "dune_network.h"
#include "dune_evaluator.h"
#include "dune_batched_evaluator.h"
#include "dune_semantic_action_scorer.h"
#include "dune_ppo_training_utils.h"
#include "dune_eval_action_selection.h"
#include "dune_seed_utils.h"
#include "dune_sha256.h"

#include "dune_test_time_adaptation.h"

using namespace open_spiel;
using namespace open_spiel::dune_imperium;
using namespace open_spiel::dune_tt_adapt;

// Link satisfaction flags for dune_ppo_training_utils.cc
ABSL_FLAG(int, ppo_minibatch_size, 2048, "");
ABSL_FLAG(int, ppo_update_epochs, 4, "");
ABSL_FLAG(double, ppo_clip_epsilon, 0.2, "");
ABSL_FLAG(bool, normalize_advantages, true, "");
ABSL_FLAG(bool, ppo_clip_value_loss, true, "");
ABSL_FLAG(double, entropy_coef, 0.01, "");
ABSL_FLAG(double, value_coef, 0.5, "");
ABSL_FLAG(double, target_kl, 0.0, "");
ABSL_FLAG(bool, train_amp, true, "");
ABSL_FLAG(double, grad_clip_norm, 0.5, "");
ABSL_FLAG(double, logit_cap, 10.0, "");
ABSL_FLAG(uint64_t, shaping_start_env_steps, 206830543, "");
ABSL_FLAG(uint64_t, shaping_decay_env_steps, 0, "");
ABSL_FLAG(bool, diagnostics_only, false, "");

// Runner execution flags
ABSL_FLAG(std::string, manifest,
          "/home/warcr/projects/dune_drl/docs/experiment_records/tt_adapt_fresh_32_roots_manifest.json",
          "Path to decision roots manifest JSON.");
ABSL_FLAG(std::string, candidate_checkpoint,
          "/run/media/warcr/Storage/dune_drl_runtime/round7/u21328_continuation_20260921_004103/checkpoints/ppo_model_update_22263.pt",
          "Path to U22263 candidate checkpoint.");
ABSL_FLAG(std::string, opponent_checkpoint,
          "/run/media/warcr/Storage/dune_drl_runtime/round7/u21328_continuation_20260921_004103/checkpoints/ppo_model_update_22263.pt",
          "Path to opponent checkpoint (same as candidate for self-play evaluation).");
ABSL_FLAG(std::string, output_json, "",
          "Path to write output results JSON receipt.");
ABSL_FLAG(int, max_roots, -1,
          "Maximum roots to evaluate (-1 for all in manifest).");
ABSL_FLAG(double, timeout_seconds, 10.0,
          "Per-root adaptation timeout in seconds.");
ABSL_FLAG(int, k_train_worlds, 8,
          "Number of training rollout worlds.");
ABSL_FLAG(int, m_eval_worlds, 16,
          "Number of held-out evaluation worlds.");
ABSL_FLAG(int, comparator_worlds, 8,
          "Number of comparator selection rollout worlds (4 per candidate action).");
ABSL_FLAG(int, gradient_steps, 5,
          "Number of gradient adaptation steps.");
ABSL_FLAG(double, learning_rate, 1e-4,
          "Learning rate for AdamW adaptation.");
ABSL_FLAG(uint64_t, max_physical_batches, 80000,
          "Global ceiling for physical batched forward passes.");
ABSL_FLAG(uint64_t, max_logical_queries, 900000,
          "Global ceiling for logical evaluator queries.");
ABSL_FLAG(bool, preflight_dry_run, false,
          "If true, performs full model and manifest loading without running rollouts.");
ABSL_FLAG(bool, allow_development_fixtures, false,
          "If true, allows running on non-registered development fixture manifests.");

static const char* kExpectedU22263Sha256 =
    "2023230744db636b16558b14858ecb9f36ce32db9e35d149a7a0be84b48cbe7e";
static const char* kExpectedU21328Sha256 =
    "dbdc36ac64d776702f4de2cc421e63cd1b590afae7d1ff7d4f2a5f411d261777";
static const char* kExpectedU20328Sha256 =
    "41f09db77f6bdd751e7aea9fb0df6f427c81b4ee7009d9a4b992adf652866646";

inline bool IsKnownCheckpointSha(const std::string& sha) {
  return sha == kExpectedU22263Sha256 || sha == kExpectedU21328Sha256 || sha == kExpectedU20328Sha256;
}

static const std::vector<std::string> kRegisteredFreshBanks = {
    "/home/warcr/projects/dune_drl/docs/experiment_records/tt_adapt_fresh_32_roots_manifest.json"
};

inline std::string ComputeModelWeightsDigest(const std::shared_ptr<SharedDunePolicyValueNetImpl>& model) {
  torch::NoGradGuard guard;
  SHA256 hasher;
  for (const auto& p : model->parameters()) {
    auto flat = p.detach().to(torch::kCPU).contiguous().to(torch::kFloat32);
    const float* ptr = flat.data_ptr<float>();
    size_t bytes = flat.numel() * sizeof(float);
    hasher.Update(reinterpret_cast<const uint8_t*>(ptr), bytes);
  }
  return hasher.Final();
}

int main(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);

  std::cout << "================================================================================\n";
  std::cout << "Dune: Imperium - Native Test-Time Policy Adaptation Runner\n";
  std::cout << "================================================================================\n";

  const std::string manifest_path = absl::GetFlag(FLAGS_manifest);
  const std::string cand_ckpt_path = absl::GetFlag(FLAGS_candidate_checkpoint);
  const std::string opp_ckpt_path = absl::GetFlag(FLAGS_opponent_checkpoint);
  const std::string output_json_path = absl::GetFlag(FLAGS_output_json);
  const bool is_preflight = absl::GetFlag(FLAGS_preflight_dry_run);

  // 1. Verify Checkpoints exist and verify SHA-256 digests
  if (!std::filesystem::exists(cand_ckpt_path)) {
    std::cerr << "FATAL: Candidate checkpoint not found: " << cand_ckpt_path << "\n";
    return 1;
  }
  if (!std::filesystem::exists(opp_ckpt_path)) {
    std::cerr << "FATAL: Opponent checkpoint not found: " << opp_ckpt_path << "\n";
    return 1;
  }

  std::string actual_cand_sha256 = open_spiel::ComputeFileSHA256(cand_ckpt_path);
  std::string actual_opp_sha256 = open_spiel::ComputeFileSHA256(opp_ckpt_path);

  std::cout << "[CHECKPOINT CANDIDATE] " << cand_ckpt_path << "\n";
  std::cout << "  SHA-256: " << actual_cand_sha256 << "\n";
  if (!IsKnownCheckpointSha(actual_cand_sha256)) {
    std::cerr << "FATAL: Candidate SHA-256 is not a known approved checkpoint!\n  Actual: "
              << actual_cand_sha256 << "\n";
    return 1;
  }

  std::cout << "[CHECKPOINT OPPONENT]  " << opp_ckpt_path << "\n";
  std::cout << "  SHA-256: " << actual_opp_sha256 << "\n";
  if (!IsKnownCheckpointSha(actual_opp_sha256)) {
    std::cerr << "FATAL: Opponent SHA-256 is not a known approved checkpoint!\n  Actual: "
              << actual_opp_sha256 << "\n";
    return 1;
  }

  // 2. Select Device
  torch::Device device(torch::kCPU);
  if (torch::cuda::is_available()) {
    device = torch::Device(torch::kCUDA, 0);
    std::cout << "[DEVICE] CUDA available. Running on GPU (cuda:0).\n";
  } else {
    std::cout << "[DEVICE] CUDA unavailable. Running on CPU.\n";
  }

  // 3. Load Models
  auto cand_model = std::make_shared<SharedDunePolicyValueNetImpl>(
      kFullPublicInformationStateSize, 2048, 2391, 8,
      /*nonlinear_value_head=*/false, /*with_aux_heads=*/false,
      /*head_init_seed=*/0, /*with_semantic_scorer=*/true);
  LoadModelCheckpointRobust(cand_model, cand_ckpt_path, device);
  cand_model->to(device);
  cand_model->eval();
  for (auto& p : cand_model->parameters()) p.set_requires_grad(false);

  auto opp_model = std::make_shared<SharedDunePolicyValueNetImpl>(
      kFullPublicInformationStateSize, 2048, 2391, 8,
      /*nonlinear_value_head=*/false, /*with_aux_heads=*/false,
      /*head_init_seed=*/0, /*with_semantic_scorer=*/true);
  LoadModelCheckpointRobust(opp_model, opp_ckpt_path, device);
  opp_model->to(device);
  opp_model->eval();
  for (auto& p : opp_model->parameters()) p.set_requires_grad(false);

  // Evaluators and Operational Budget
  OperationalBudget budget;
  budget.max_physical_batches = absl::GetFlag(FLAGS_max_physical_batches);
  budget.max_logical_queries = absl::GetFlag(FLAGS_max_logical_queries);

  auto dispatch_hook = [&budget](size_t batch_size) -> bool {
    return budget.TryReservePhysicalBatches(1);
  };
  auto rejection_hook = [&budget](size_t batch_size) {
    budget.ReleaseLogicalQueries(batch_size);
  };

  std::shared_mutex cand_mutex;
  auto cand_coord = std::make_shared<BatchedEvaluator>(
      cand_model, 32, 1, device, &cand_mutex, 10.0f,
      /*device_synchronize=*/false, /*high_priority_stream=*/false,
      /*emit_batch_membership=*/false, /*rollout_amp=*/true, /*allow_tf32=*/true);
  cand_coord->SetDispatchBudgetHook(dispatch_hook, rejection_hook);
  auto cand_evaluator = std::make_shared<BatchedNNEvaluator>(cand_coord, 10.0f);

  std::shared_mutex opp_mutex;
  std::shared_ptr<BatchedEvaluator> opp_coord;
  std::shared_ptr<BatchedNNEvaluator> opp_evaluator;
  if (actual_cand_sha256 == actual_opp_sha256) {
    opp_coord = cand_coord;
    opp_evaluator = cand_evaluator;
  } else {
    opp_coord = std::make_shared<BatchedEvaluator>(
        opp_model, 16, 1, device, &opp_mutex, 10.0f,
        /*device_synchronize=*/false, /*high_priority_stream=*/false,
        /*emit_batch_membership=*/false, /*rollout_amp=*/true, /*allow_tf32=*/true);
    opp_coord->SetDispatchBudgetHook(dispatch_hook, rejection_hook);
    opp_evaluator = std::make_shared<BatchedNNEvaluator>(opp_coord, 10.0f);
  }

  // 4. Load Manifest & Validate Against Registered Banks
  if (!std::filesystem::exists(manifest_path)) {
    std::cerr << "FATAL: Manifest path not found: " << manifest_path << "\n";
    return 1;
  }

  bool is_registered_bank = false;
  try {
    std::string canon_mf = std::filesystem::canonical(manifest_path).string();
    for (const auto& reg : kRegisteredFreshBanks) {
      if (std::filesystem::exists(reg) && std::filesystem::canonical(reg).string() == canon_mf) {
        is_registered_bank = true;
        break;
      }
    }
  } catch (...) {
    is_registered_bank = false;
  }

  const bool allow_dev_fixtures = absl::GetFlag(FLAGS_allow_development_fixtures);
  if (!is_registered_bank && !allow_dev_fixtures) {
    std::cerr << "FATAL: Manifest is not a registered fresh evaluation bank: " << manifest_path << "\n";
    std::cerr << "Pass --allow_development_fixtures to run on development or synthetic fixtures.\n";
    return 1;
  }
  std::ifstream mf_file(manifest_path);
  std::string mf_str((std::istreambuf_iterator<char>(mf_file)),
                     std::istreambuf_iterator<char>());
  auto mf_json = json::FromString(mf_str);
  if (!mf_json.has_value() || !mf_json->IsArray()) {
    std::cerr << "FATAL: Failed to parse manifest JSON as array: " << manifest_path << "\n";
    return 1;
  }

  const auto& json_roots = mf_json->GetArray();
  std::vector<DecisionRoot> roots;
  int count_agent = 0, count_reveal = 0, count_combat = 0;

  for (size_t i = 0; i < json_roots.size(); ++i) {
    const auto& r_obj = json_roots[i].GetObject();
    DecisionRoot r;
    r.root_id = static_cast<int>(r_obj.at("root_id").GetInt());
    r.episode_id = static_cast<int>(r_obj.at("episode_id").GetInt());
    r.player = static_cast<Player>(r_obj.at("player").GetInt());
    r.round = static_cast<int>(r_obj.at("round").GetInt());
    r.stratum = r_obj.at("stratum").GetString();
    if (r_obj.find("phase") != r_obj.end()) {
      r.phase = r_obj.at("phase").GetString();
    }
    r.train_seed_base = static_cast<uint64_t>(r_obj.at("train_seed_base").GetInt());
    r.eval_seed_base = static_cast<uint64_t>(r_obj.at("eval_seed_base").GetInt());
    for (const auto& a_val : r_obj.at("history").GetArray()) {
      r.history.push_back(static_cast<Action>(a_val.GetInt()));
    }
    for (const auto& a_val : r_obj.at("legal_actions").GetArray()) {
      r.legal_actions.push_back(static_cast<Action>(a_val.GetInt()));
    }
    r.num_legal = r.legal_actions.size();

    if (r.stratum == "Agent" || r.stratum == "agent_turns") count_agent++;
    else if (r.stratum == "Reveal" || r.stratum == "reveal_turns") count_reveal++;
    else if (r.stratum == "Combat" || r.stratum == "combat") count_combat++;

    roots.push_back(r);
  }

  std::cout << "[MANIFEST] Loaded " << roots.size() << " decision roots from: " << manifest_path << "\n";
  std::cout << "  Strata Breakdown: " << count_agent << " Agent, " << count_reveal
            << " Reveal, " << count_combat << " Combat\n";

  // 5. Preflight Dry-Run Branch
  if (is_preflight) {
    std::cout << "\n--------------------------------------------------------------------------------\n";
    std::cout << "Preflight dry-run completed successfully.\n";
    std::cout << "Candidate & Opponent checkpoints verified, evaluators created, and manifest parsed.\n";
    std::cout << "STATUS: DEVELOPMENT_READINESS_PASS\n";
    std::cout << "--------------------------------------------------------------------------------\n";

    if (!output_json_path.empty()) {
      json::Object root_out;
      root_out["status"] = json::Value("DEVELOPMENT_READINESS_PASS");
      root_out["candidate_checkpoint"] = json::Value(cand_ckpt_path);
      root_out["candidate_sha256"] = json::Value(actual_cand_sha256);
      root_out["opponent_checkpoint"] = json::Value(opp_ckpt_path);
      root_out["opponent_sha256"] = json::Value(actual_opp_sha256);
      root_out["manifest_path"] = json::Value(manifest_path);
      root_out["roots_count"] = json::Value(static_cast<int64_t>(roots.size()));
      root_out["strata_agent"] = json::Value(static_cast<int64_t>(count_agent));
      root_out["strata_reveal"] = json::Value(static_cast<int64_t>(count_reveal));
      root_out["strata_combat"] = json::Value(static_cast<int64_t>(count_combat));
      root_out["pilot_started"] = json::Value(false);
      root_out["pilot_completed"] = json::Value(false);

      std::ofstream out_file(output_json_path);
      out_file << json::ToString(json::Value(root_out), true);
      std::cout << "Preflight receipt written to: " << output_json_path << "\n";
    }
    return 0;
  }

  // 6. Execution Setup
  std::shared_ptr<const Game> game = LoadGame("dune_imperium");

  AdaptationConfig config;
  config.k_train_worlds = absl::GetFlag(FLAGS_k_train_worlds);
  config.m_eval_worlds = absl::GetFlag(FLAGS_m_eval_worlds);
  config.comparator_worlds = absl::GetFlag(FLAGS_comparator_worlds);
  config.gradient_steps = absl::GetFlag(FLAGS_gradient_steps);
  config.learning_rate = static_cast<float>(absl::GetFlag(FLAGS_learning_rate));
  config.timeout_seconds = absl::GetFlag(FLAGS_timeout_seconds);

  TestTimeAdaptationController controller(cand_model, cand_evaluator, device, config);
  AdaptationClock real_clock;

  std::string initial_cand_digest = ComputeModelWeightsDigest(cand_model);

  int limit_roots = absl::GetFlag(FLAGS_max_roots);
  size_t num_to_run = roots.size();
  if (limit_roots > 0 && static_cast<size_t>(limit_roots) < num_to_run) {
    num_to_run = static_cast<size_t>(limit_roots);
  }

  std::cout << "\nStarting Native Pilot Execution across " << num_to_run << " decision roots...\n";
  std::cout << "Configuration: K=" << config.k_train_worlds << " train, M=" << config.m_eval_worlds
            << " eval, comp=" << config.comparator_worlds << " selection worlds, steps="
            << config.gradient_steps << ", lr=" << config.learning_rate << ".\n";
  std::cout << "Budget Ceilings: max_physical_batches=" << budget.max_physical_batches
            << ", max_logical_queries=" << budget.max_logical_queries << "\n";

  bool pilot_started = true;
  bool pilot_completed = false;
  std::vector<RootExecutionResult> completed_results;
  std::string abort_reason = "";

  auto total_start_time = std::chrono::steady_clock::now();

  for (size_t i = 0; i < num_to_run; ++i) {
    const auto& root = roots[i];

    // Pre-dispatch budget check
    if (!budget.CanDispatch(1, 1)) {
      abort_reason = "BUDGET_EXHAUSTED";
      std::cout << "\n[HALT] Operational budget ceiling reached before root " << root.root_id << ".\n";
      break;
    }

    RootExecutionResult root_res;
    bool ok = ExecuteDecisionRoot(
        game, root, cand_model, cand_evaluator, opp_evaluator,
        &controller, config, &budget, &real_clock, &root_res);

    if (!ok || !root_res.evaluation_completed) {
      abort_reason = "ROOT_EXECUTION_FAILURE";
      std::cout << "\n[HALT] Execution failed or timed out at root " << root.root_id
                << " (fallback=" << FallbackReasonToString(root_res.fallback_reason) << ").\n";
      break;
    }

    completed_results.push_back(root_res);
    std::cout << "  Root [" << std::setw(2) << (i + 1) << "/" << num_to_run << "] "
              << "ID=" << std::setw(2) << root.root_id << " (" << std::setw(6) << root.stratum << ") "
              << "Raw=" << root_res.raw_action << " Adapt=" << root_res.adapt_action
              << " Comp=" << root_res.comp_action << " | "
              << "MeanRaw=" << std::fixed << std::setprecision(3) << root_res.mean_raw
              << " MeanAdapt=" << root_res.mean_adapt
              << " Diff=" << (root_res.paired_diff >= 0 ? "+" : "") << root_res.paired_diff
              << " (" << std::setprecision(2) << root_res.elapsed_seconds << "s)\n";
  }

  auto total_end_time = std::chrono::steady_clock::now();
  double total_elapsed_sec = std::chrono::duration<double>(total_end_time - total_start_time).count();

  if (completed_results.size() == num_to_run) {
    pilot_completed = true;
  }

  std::string final_cand_digest = ComputeModelWeightsDigest(cand_model);
  bool blueprint_preserved = (initial_cand_digest == final_cand_digest);

  uint64_t total_batches = budget.physical_batches.load();
  uint64_t total_queries = budget.logical_queries.load();

  // 7. Aggregate Statistics
  std::vector<int> registered_root_ids;
  if (is_registered_bank && !allow_dev_fixtures) {
    for (const auto& r : roots) registered_root_ids.push_back(r.root_id);
  }
  PilotAggregateStats agg_stats = ComputePilotAggregateStats(
      completed_results, total_batches, total_queries, blueprint_preserved, registered_root_ids);

  std::cout << "\n================================================================================\n";
  std::cout << "Pilot Execution Summary:\n";
  std::cout << "================================================================================\n";
  std::cout << "Roots completed: " << completed_results.size() << " / " << num_to_run << "\n";
  std::cout << "Total physical batches: " << total_batches << " / " << budget.max_physical_batches << "\n";
  std::cout << "Total logical queries:  " << total_queries << " / " << budget.max_logical_queries << "\n";
  std::cout << "Blueprint Preserved:    " << (blueprint_preserved ? "TRUE" : "FALSE") << "\n";
  std::cout << "Total wall-clock time:  " << std::fixed << std::setprecision(1) << total_elapsed_sec << "s\n";
  std::cout << "Mean Raw Utility:       " << std::setprecision(4) << agg_stats.mean_raw_utility << "\n";
  std::cout << "Mean Adapt Utility:     " << agg_stats.mean_adapt_utility << "\n";
  std::cout << "Delta u_bar:            " << (agg_stats.delta_u_bar >= 0 ? "+" : "") << agg_stats.delta_u_bar << "\n";
  std::cout << "s_root (std dev):       " << agg_stats.s_root << "\n";
  std::cout << "SE(Delta u_bar):        " << agg_stats.se_delta_u_bar << "\n";
  std::cout << "95% CI:                 [" << agg_stats.ci_95_lower << ", " << agg_stats.ci_95_upper << "]\n";
  std::cout << "Comparator Delta u_bar: " << (agg_stats.comp_delta_u_bar >= 0 ? "+" : "") << agg_stats.comp_delta_u_bar << "\n";

  // Scientific Verdict Determination
  std::string scientific_verdict = "NULL";
  std::string overall_status = "SUCCESS";

  if (!pilot_completed) {
    overall_status = "FEASIBILITY_FAILURE_INCOMPLETE_EVALUATION";
    scientific_verdict = "NULL";
    std::cout << "VERDICT: Incomplete evaluation (" << abort_reason << "). Scientific verdict is null.\n";
  } else if (!is_registered_bank || allow_dev_fixtures || num_to_run < roots.size()) {
    overall_status = "SUBSET_RUN_COMPLETED";
    scientific_verdict = "NULL";
    std::cout << "STATUS: SUBSET_RUN_COMPLETED (Development / subset run; scientific verdict is null).\n";
  } else {
    // Registered fresh evaluation bank, full 32 roots completed
    if (agg_stats.all_gates_passed) {
      scientific_verdict = "PILOT_CONFIRMED_STRENGTH_IMPROVEMENT";
      std::cout << "VERDICT: PILOT_CONFIRMED_STRENGTH_IMPROVEMENT (All 4 gates passed)\n";
    } else if (agg_stats.delta_u_bar <= 0.0) {
      scientific_verdict = "NO_POSITIVE_DECISION_SIGNAL";
      std::cout << "VERDICT: NO_POSITIVE_DECISION_SIGNAL (Delta u_bar <= 0.0)\n";
    } else {
      scientific_verdict = "PILOT_NULL_RESULT";
      std::cout << "VERDICT: PILOT_NULL_RESULT (CI covers zero or failed secondary gates)\n";
    }
  }

  // 8. Output JSON Receipt
  if (!output_json_path.empty()) {
    json::Object out_obj;
    out_obj["pilot_started"] = json::Value(pilot_started);
    out_obj["pilot_completed"] = json::Value(pilot_completed);
    out_obj["status"] = json::Value(overall_status);
    if (pilot_completed) {
      out_obj["scientific_verdict"] = json::Value(scientific_verdict);
    } else {
      out_obj["scientific_verdict"] = json::Value(); // null
    }

    out_obj["candidate_checkpoint"] = json::Value(cand_ckpt_path);
    out_obj["candidate_sha256"] = json::Value(actual_cand_sha256);
    out_obj["opponent_checkpoint"] = json::Value(opp_ckpt_path);
    out_obj["opponent_sha256"] = json::Value(actual_opp_sha256);
    out_obj["manifest_path"] = json::Value(manifest_path);
    out_obj["blueprint_preserved"] = json::Value(blueprint_preserved);
    out_obj["initial_candidate_digest"] = json::Value(initial_cand_digest);
    out_obj["final_candidate_digest"] = json::Value(final_cand_digest);

    json::Object budget_obj;
    budget_obj["physical_batches_consumed"] = json::Value(static_cast<int64_t>(total_batches));
    budget_obj["max_physical_batches"] = json::Value(static_cast<int64_t>(budget.max_physical_batches));
    budget_obj["logical_queries_consumed"] = json::Value(static_cast<int64_t>(total_queries));
    budget_obj["max_logical_queries"] = json::Value(static_cast<int64_t>(budget.max_logical_queries));
    budget_obj["total_elapsed_seconds"] = json::Value(total_elapsed_sec);
    out_obj["budget_accounting"] = json::Value(budget_obj);

    json::Object stats_obj;
    stats_obj["roots_completed"] = json::Value(static_cast<int64_t>(completed_results.size()));
    stats_obj["mean_raw_utility"] = json::Value(agg_stats.mean_raw_utility);
    stats_obj["mean_adapt_utility"] = json::Value(agg_stats.mean_adapt_utility);
    stats_obj["mean_comp_utility"] = json::Value(agg_stats.mean_comp_utility);
    stats_obj["delta_u_bar"] = json::Value(agg_stats.delta_u_bar);
    stats_obj["s_root"] = json::Value(agg_stats.s_root);
    stats_obj["se_delta_u_bar"] = json::Value(agg_stats.se_delta_u_bar);
    stats_obj["ci_95_lower"] = json::Value(agg_stats.ci_95_lower);
    stats_obj["ci_95_upper"] = json::Value(agg_stats.ci_95_upper);
    stats_obj["comp_delta_u_bar"] = json::Value(agg_stats.comp_delta_u_bar);
    stats_obj["gate1_passed"] = json::Value(agg_stats.gate1_passed);
    stats_obj["gate2_passed"] = json::Value(agg_stats.gate2_passed);
    stats_obj["gate3_passed"] = json::Value(agg_stats.gate3_passed);
    stats_obj["gate4_passed"] = json::Value(agg_stats.gate4_passed);
    stats_obj["all_gates_passed"] = json::Value(agg_stats.all_gates_passed);
    out_obj["aggregate_statistics"] = json::Value(stats_obj);

    json::Array root_results_arr;
    for (const auto& r : completed_results) {
      json::Object r_obj;
      r_obj["root_id"] = json::Value(static_cast<int64_t>(r.root_id));
      r_obj["episode_id"] = json::Value(static_cast<int64_t>(r.episode_id));
      r_obj["player"] = json::Value(static_cast<int64_t>(r.player));
      r_obj["round"] = json::Value(static_cast<int64_t>(r.round));
      r_obj["stratum"] = json::Value(r.stratum);
      r_obj["raw_action"] = json::Value(static_cast<int64_t>(r.raw_action));
      r_obj["adapt_action"] = json::Value(static_cast<int64_t>(r.adapt_action));
      r_obj["comp_action"] = json::Value(static_cast<int64_t>(r.comp_action));
      r_obj["fallback_triggered"] = json::Value(r.fallback_triggered);
      r_obj["fallback_reason"] = json::Value(FallbackReasonToString(r.fallback_reason));
      r_obj["kl_divergence"] = json::Value(r.kl_divergence);
      r_obj["policy_grad_norm"] = json::Value(static_cast<double>(r.policy_grad_norm));
      r_obj["loss_value"] = json::Value(static_cast<double>(r.loss_value));
      r_obj["elapsed_seconds"] = json::Value(r.elapsed_seconds);
      r_obj["mean_raw"] = json::Value(r.mean_raw);
      r_obj["mean_adapt"] = json::Value(r.mean_adapt);
      r_obj["mean_comp"] = json::Value(r.mean_comp);
      r_obj["paired_diff"] = json::Value(r.paired_diff);
      r_obj["comp_diff"] = json::Value(r.comp_diff);

      json::Array raw_ret_arr, adapt_ret_arr, comp_ret_arr;
      for (double val : r.raw_returns) raw_ret_arr.push_back(json::Value(val));
      for (double val : r.adapt_returns) adapt_ret_arr.push_back(json::Value(val));
      for (double val : r.comp_returns) comp_ret_arr.push_back(json::Value(val));
      r_obj["raw_returns"] = json::Value(raw_ret_arr);
      r_obj["adapt_returns"] = json::Value(adapt_ret_arr);
      r_obj["comp_returns"] = json::Value(comp_ret_arr);

      root_results_arr.push_back(json::Value(r_obj));
    }
    out_obj["per_root_results"] = json::Value(root_results_arr);

    std::ofstream out_file(output_json_path);
    out_file << json::ToString(json::Value(out_obj), true);
    std::cout << "Receipt successfully saved to: " << output_json_path << "\n";
  }

  return pilot_completed ? 0 : 2;
}
