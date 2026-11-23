// Reproducible population evaluator for Dune Imperium.
//
// Replaces the ad-hoc dune_eval_1000.cc with deterministic, domain-separated
// seeding via dune_seed_utils.h. Produces per-game JSONL and aggregate JSON
// with Wilson confidence intervals.
//
// Key properties:
//   - Per-game seeds derived from (base_seed, domain, episode_id, stream)
//   - Deterministic seat rotation and opponent assignment
//   - Heterogeneous opponent architectures supported
//   - Greedy / stochastic policy selection
//   - Thread-count reproducibility (results identical regardless of --threads)

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include <torch/torch.h>

#include "open_spiel/abseil-cpp/absl/flags/flag.h"
#include "open_spiel/abseil-cpp/absl/flags/parse.h"

#include "dune_population_eval_internal.h"
#include "dune_seed_utils.h"

// ---------------------------------------------------------------------------
// Flags
// ---------------------------------------------------------------------------
ABSL_FLAG(bool, candidate_r758_teacher, false,
          "If true, candidate seat runs 1-round CompoundRolloutSearchDecision teacher at play time.");
ABSL_FLAG(int, candidate_eval_batch_size, -1,
          "Optional explicit target batch size for candidate BatchedEvaluator (-1 = auto).");
ABSL_FLAG(int, teacher_category_mask, 27,
          "R755 category bitmask for candidate_r758_teacher.");
ABSL_FLAG(int, teacher_top_k, 3,
          "Top-K candidates for candidate_r758_teacher.");
ABSL_FLAG(int, teacher_rollouts, 16,
          "Rollouts per action for candidate_r758_teacher.");
ABSL_FLAG(int, teacher_stage1_rollouts, 8,
          "Stage-1 rollouts for candidate_r758_teacher (0 = 1-stage).");
ABSL_FLAG(double, teacher_stage1_min_mean_diff, 0.03,
          "Stage-1 min mean diff vs raw for candidate_r758_teacher.");
ABSL_FLAG(double, teacher_min_t_stat, 1.50,
          "Min paired t-stat for candidate_r758_teacher.");
ABSL_FLAG(double, teacher_override_margin, 0.15,
          "Min override utility margin for candidate_r758_teacher.");
ABSL_FLAG(std::string, teacher_crn_mode, "deck_order",
          "CRN mode for candidate_r758_teacher: sequential, deck_order, event_keyed.");

ABSL_FLAG(std::string, model_checkpoint, "",
          "Path to the model checkpoint to evaluate.");
ABSL_FLAG(bool, enable_semantic_scorer, false,
          "Enable the neural Semantic Action Scorer for candidate evaluation.");
ABSL_FLAG(std::string, opponent_checkpoints, "",
          "Comma-separated paths to opponent model checkpoints. "
          "Empty or \"random\" for random opponents.");
ABSL_FLAG(int, num_games, 5000,
          "Total number of evaluation games to play.");
ABSL_FLAG(int, start_episode_id, 0,
          "First evaluation episode ID. Default 0 preserves historical runs.");
ABSL_FLAG(uint64_t, base_seed, 42,
          "Base seed for deterministic evaluation.");
ABSL_FLAG(std::string, domain, "EVAL_BASELINE",
          "Evaluation domain: EVAL_BASELINE, EVAL_DEV, or EVAL_FINAL.");
ABSL_FLAG(bool, greedy, true,
          "If true, use argmax policy; if false, sample from softmax.");
ABSL_FLAG(int, threads, 0,
          "Number of game worker threads. 0 = auto-detect.");
ABSL_FLAG(int, hidden_dim, 2048,
          "Hidden dimension of the evaluated model.");
ABSL_FLAG(int, num_blocks, 8,
          "Number of residual blocks in the evaluated model.");
ABSL_FLAG(int, opp_hidden_dim, -1,
          "Opponent hidden dim. -1 = same as --hidden_dim.");
ABSL_FLAG(int, opp_num_blocks, -1,
          "Opponent num blocks. -1 = same as --num_blocks.");
ABSL_FLAG(std::string, output_dir, "",
          "Directory for per-game JSONL and aggregate JSON. "
          "Empty = stdout only.");
ABSL_FLAG(float, temperature, 1.0f,
          "Softmax temperature for stochastic policy (--greedy=false).");
// PWO-5 gate 2 item (d). OPTIONAL flags: unset (the default) means INHERIT the
// global --greedy / --temperature, which is exactly the pre-gate-2 behaviour for
// every seat. A plain bool defaulting to true would NOT be backward compatible,
// because a run passing --greedy=false would then keep greedy opponents.
ABSL_FLAG(std::optional<bool>, opponent_greedy, std::nullopt,
          "Action selection for OPPONENT seats only. Unset (default) = inherit "
          "--greedy, i.e. identical to pre-existing behaviour. Set true to hold "
          "opponents at argmax while the candidate samples -- required to "
          "express the frozen strong population (raw prior, GREEDY) against a "
          "candidate sampled at temperature 1.0.");
ABSL_FLAG(std::optional<float>, opponent_temperature, std::nullopt,
          "Softmax temperature for OPPONENT seats only. Unset (default) = "
          "inherit --temperature. Inert when opponents are greedy.");
ABSL_FLAG(bool, deterministic_eval, true,
          "If true, use batch-1 mutex-serialized inference for strict bitwise "
          "thread-count reproducibility. Much slower than batched mode.");
ABSL_FLAG(bool, nonlinear_value_head, false,
          "Use the versioned nonlinear value head for the evaluated model.");
ABSL_FLAG(bool, opponent_nonlinear_value_head, false,
          "Use the versioned nonlinear value head for opponent checkpoints.");
ABSL_FLAG(double, candidate_logit_cap, 10.0,
          "Cap on legal-centered logits for the CANDIDATE (evaluated) model, "
          "applied by CenterAndCapLegalLogits. Default 10.0 preserves prior "
          "behavior (formerly the hard-coded kEvalLogitCap).");
ABSL_FLAG(double, opponent_logit_cap, 10.0,
          "Cap on legal-centered logits for OPPONENT models. Default 10.0 "
          "preserves prior behavior (formerly the hard-coded kEvalLogitCap).");
ABSL_FLAG(std::string, dump_logit_stats, "",
          "If non-empty, append one CSV row per CANDIDATE decision with pre-cap "
          "legal-centered logit statistics to this path (audit; off by default).");
ABSL_FLAG(std::string, candidate_market_appendix_mode, "auto",
          "Market appendix mode for candidate: 'auto' (infer from model/manifest), 'none', 'zeros', or 'ordered_market'.");
ABSL_FLAG(bool, record_market_diagnostics, false,
          "If true, collect per-game and aggregate market purchase diagnostics.");
ABSL_FLAG(std::string, market_diagnostics_json, "",
          "Optional explicit path for market diagnostics JSON. Defaults to "
          "<output_dir>/market_diagnostics.json if --record_market_diagnostics is true and --output_dir is set.");
ABSL_FLAG(std::string, market_diagnostics_csv, "",
          "Optional explicit path for market diagnostics CSV. Defaults to "
          "<output_dir>/market_diagnostics.csv if --record_market_diagnostics is true and --output_dir is set.");
ABSL_FLAG(std::string, market_diagnostics_games_jsonl, "",
          "Optional explicit path for compact per-game market diagnostics JSONL. Defaults to "
          "<output_dir>/market_diagnostics_games.jsonl if --record_market_diagnostics is true and --output_dir is set.");
ABSL_FLAG(int, trace_episode, -1,
          "If >= 0, print detailed action-by-action trace and VP breakdown for this episode ID (-2 for all).");
ABSL_FLAG(bool, record_history, false,
          "Record full action history per game in games.jsonl.");
ABSL_FLAG(bool, rollout_amp, true,
          "Use BF16 AMP autocast in evaluator forward pass.");
ABSL_FLAG(bool, allow_tf32, true,
          "Allow TF32 in CUDA matmuls/convolutions.");

namespace open_spiel {
namespace {

dune_pop_eval::EvalConfig BuildEvalConfigFromFlags() {
  dune_pop_eval::EvalConfig cfg;
  cfg.model_checkpoint = absl::GetFlag(FLAGS_model_checkpoint);
  cfg.opponent_str = absl::GetFlag(FLAGS_opponent_checkpoints);
  cfg.total_games = absl::GetFlag(FLAGS_num_games);
  cfg.start_episode_id = absl::GetFlag(FLAGS_start_episode_id);
  cfg.base_seed = absl::GetFlag(FLAGS_base_seed);
  cfg.domain_str = absl::GetFlag(FLAGS_domain);
  cfg.greedy = absl::GetFlag(FLAGS_greedy);
  cfg.requested_threads = absl::GetFlag(FLAGS_threads);
  cfg.hidden_dim = absl::GetFlag(FLAGS_hidden_dim);
  cfg.num_blocks = absl::GetFlag(FLAGS_num_blocks);
  cfg.opp_hidden_dim = absl::GetFlag(FLAGS_opp_hidden_dim);
  cfg.opp_num_blocks = absl::GetFlag(FLAGS_opp_num_blocks);
  cfg.output_dir = absl::GetFlag(FLAGS_output_dir);
  cfg.temperature = absl::GetFlag(FLAGS_temperature);

  // PWO-5 gate 2 item (d). Unset => inherit the global value; the resolution
  // itself lives in dune_eval::ResolveSelectionPolicy so it is unit-testable.
  const std::optional<bool> opponent_greedy_opt =
      absl::GetFlag(FLAGS_opponent_greedy);
  const std::optional<float> opponent_temperature_opt =
      absl::GetFlag(FLAGS_opponent_temperature);
  cfg.has_opponent_greedy = opponent_greedy_opt.has_value();
  cfg.opponent_greedy = opponent_greedy_opt.value_or(false);
  cfg.has_opponent_temperature = opponent_temperature_opt.has_value();
  cfg.opponent_temperature = opponent_temperature_opt.value_or(0.0f);

  cfg.candidate_logit_cap = absl::GetFlag(FLAGS_candidate_logit_cap);
  cfg.opponent_logit_cap = absl::GetFlag(FLAGS_opponent_logit_cap);
  cfg.dump_logit_stats_path = absl::GetFlag(FLAGS_dump_logit_stats);
  cfg.record_market_diagnostics = absl::GetFlag(FLAGS_record_market_diagnostics);
  cfg.market_diag_json_path = absl::GetFlag(FLAGS_market_diagnostics_json);
  cfg.market_diag_csv_path = absl::GetFlag(FLAGS_market_diagnostics_csv);
  cfg.market_diag_games_path = absl::GetFlag(FLAGS_market_diagnostics_games_jsonl);
  if (cfg.record_market_diagnostics && !cfg.output_dir.empty()) {
    if (cfg.market_diag_json_path.empty()) {
      cfg.market_diag_json_path = (std::filesystem::path(cfg.output_dir) / "market_diagnostics.json").string();
    }
    if (cfg.market_diag_csv_path.empty()) {
      cfg.market_diag_csv_path = (std::filesystem::path(cfg.output_dir) / "market_diagnostics.csv").string();
    }
    if (cfg.market_diag_games_path.empty()) {
      cfg.market_diag_games_path = (std::filesystem::path(cfg.output_dir) / "market_diagnostics_games.jsonl").string();
    }
  }
  if (!cfg.output_dir.empty()) {
    std::filesystem::create_directories(cfg.output_dir);
  }

  if (cfg.model_checkpoint.empty()) {
    std::cerr << "Error: --model_checkpoint is required.\n";
    std::exit(1);
  }

  cfg.domain = dune_pop_eval::ResolveDomain(cfg.domain_str);
  if (cfg.opp_hidden_dim < 0) cfg.opp_hidden_dim = cfg.hidden_dim;
  if (cfg.opp_num_blocks < 0) cfg.opp_num_blocks = cfg.num_blocks;

  if (!cfg.opponent_str.empty() && cfg.opponent_str != "random") {
    cfg.opponent_paths = dune_pop_eval::SplitCommaSeparated(cfg.opponent_str);
  }
  cfg.use_opponent_model = !cfg.opponent_paths.empty();

  cfg.deterministic = absl::GetFlag(FLAGS_deterministic_eval);
  cfg.rollout_amp = absl::GetFlag(FLAGS_rollout_amp);
  cfg.allow_tf32 = absl::GetFlag(FLAGS_allow_tf32);
  cfg.record_history = absl::GetFlag(FLAGS_record_history);
  cfg.nonlinear_value_head = absl::GetFlag(FLAGS_nonlinear_value_head);
  cfg.opponent_nonlinear_value_head = absl::GetFlag(FLAGS_opponent_nonlinear_value_head);
  return cfg;
}

// ---------------------------------------------------------------------------
// Main evaluation driver
// ---------------------------------------------------------------------------
void RunEvaluation() {
  const dune_pop_eval::EvalConfig cfg = BuildEvalConfigFromFlags();

  torch::InferenceMode inference_guard;
  std::unique_ptr<dune_pop_eval::EvalRuntime> runtime =
      dune_pop_eval::SetupEvaluation(cfg);
  dune_pop_eval::PrintEvaluationConfig(cfg, *runtime);

  // --- Optional per-candidate-decision logit-stat dump (audit) ---
  std::ofstream logit_dump;
  std::ofstream* logit_dump_ptr = nullptr;
  if (!cfg.dump_logit_stats_path.empty()) {
    logit_dump.open(cfg.dump_logit_stats_path);
    if (logit_dump.is_open()) {
      logit_dump << "episode_id,round,decision_role,n_legal,max_abs_z,p90_abs_z,"
                    "entropy_uncapped,entropy_cap10,entropy_cap30,"
                    "kl_cap10_uncapped,kl_cap10_cap30,"
                    "frac_legal_absz_ge5,frac_legal_absz_ge10,frac_legal_absz_ge20,"
                    // WO-1 Phase 3 (appended; columns above unchanged)
                    "norm_entropy,max_action_prob,eff_action_count,"
                    "configured_logit_cap,frac_legal_absz_ge_cap,"
                    "frac_legal_absz_ge_2cap\n";
      logit_dump_ptr = &logit_dump;
      std::cout << "Dumping candidate logit stats to " << cfg.dump_logit_stats_path
                << "\n";
    } else {
      std::cerr << "WARNING: could not open --dump_logit_stats path: "
                << cfg.dump_logit_stats_path << "\n";
    }
  }

  // --- Pre-allocate results (indexed by episode_id for determinism) ---
  std::vector<dune_pop_eval::GameResult> results(cfg.total_games);
  std::vector<dune_market_diag::CompactPerGameMarketDiagnostics> market_results;
  if (cfg.record_market_diagnostics) {
    market_results.resize(cfg.total_games);
  }

  auto start_time = std::chrono::steady_clock::now();

  dune_pop_eval::RunEvaluationGames(
      cfg, *runtime, logit_dump_ptr, results,
      cfg.record_market_diagnostics ? &market_results : nullptr);

  if (logit_dump.is_open()) logit_dump.close();

  dune_pop_eval::ReportMarketDiagnostics(cfg, market_results);

  auto end_time = std::chrono::steady_clock::now();
  double elapsed_secs =
      std::chrono::duration<double>(end_time - start_time).count();

  dune_pop_eval::ReportAggregateResults(cfg, *runtime, elapsed_secs, results);
}

}  // namespace
}  // namespace open_spiel

int main(int argc, char* argv[]) {
  setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8", 1);
  absl::ParseCommandLine(argc, argv);

  // PWO-5 section 10.2: the reserved final-gate base-seed range. No training
  // OR EVALUATION seed may enter [9000000, 9999999]; checked before any work.
  {
    const std::string stop = dune_seed::ReservedFinalGateSeedStop(
        "--base_seed", static_cast<long long>(absl::GetFlag(FLAGS_base_seed)));
    if (!stop.empty()) {
      std::cerr << stop << "\n";
      return 1;
    }
  }

  // Set PyTorch to use deterministic algorithms globally
  at::globalContext().setDeterministicAlgorithms(true, /*silent=*/true);

  // On GPU: game threads only do engine work. On CPU: Runner needs all cores.
  if (torch::cuda::is_available()) {
    at::set_num_threads(1);
  } else {
    at::set_num_threads(
        std::max(1, static_cast<int>(std::thread::hardware_concurrency())));
  }
  at::set_num_interop_threads(1);

  open_spiel::RunEvaluation();
  return 0;
}
