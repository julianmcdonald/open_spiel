// Copyright 2026
// CLI entry point for Dune Imperium Frozen Policy Diagnostics:
// 1. Timing preference: Does the frozen policy play resource intrigues before ending its turn?
// 2. Opponent placement response: Does early revelation change opponent placement distribution in unmodified games?
// 3. Strategic holding value: Does holding improve the holder's final placement utility in particular validated positions?

#include "dune_frozen_policy_diagnostics.h"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "open_spiel/abseil-cpp/absl/flags/flag.h"
#include "open_spiel/abseil-cpp/absl/flags/parse.h"
#include "open_spiel/abseil-cpp/absl/strings/str_format.h"
#include "open_spiel/abseil-cpp/absl/strings/str_split.h"

ABSL_FLAG(std::string, checkpoint, "",
          "Path to frozen model checkpoint (.pt). If empty, runs scripted controls.");
ABSL_FLAG(std::string, seat_checkpoints, "",
          "Optional comma-separated list of 4 checkpoints for seats 0..3.");
ABSL_FLAG(int, num_scenarios, 128,
          "Number of distinct base scenarios for ablation/diag2 (default 128).");
ABSL_FLAG(int, num_replicates, 64,
          "Number of independent policy replicates per branch per scenario (default 64).");
ABSL_FLAG(uint64_t, base_seed, 20260915,
          "Base RNG seed for deterministic evaluation.");
ABSL_FLAG(float, temperature, 1.0f,
          "Softmax temperature for stochastic sampling (default 1.0).");
ABSL_FLAG(bool, smoke_test, false,
          "If true, runs a bounded smoke test (5 ordinary games, 4 candidates, 8 confirmation, 8 continuation).");
ABSL_FLAG(bool, run_ablation, false,
          "If true, runs synthetic ablation suite (Diagnostic 1). Default false.");
ABSL_FLAG(bool, run_diagnostic2, true,
          "If true, runs Diagnostic 2 (timing preference / early-play vs repeated sampling).");
ABSL_FLAG(bool, run_mining, true,
          "If true, runs actual-play intrigue mining from ordinary games.");
ABSL_FLAG(int, ordinary_games, 100,
          "Number of ordinary self-play games to simulate from NewInitialState() (default 100).");
ABSL_FLAG(int, num_mining_games, 100,
          "Synonym for ordinary_games (default 100).");
ABSL_FLAG(int, discovery_pairs, 4,
          "Number of discovery paired rollouts per candidate root (default 4).");
ABSL_FLAG(int, max_candidates, 8,
          "Maximum number of discovered candidates to confirm (default 8).");
ABSL_FLAG(int, confirmation_samples, 64,
          "Number of fresh independent paired seeds for candidate confirmation (default 64).");
ABSL_FLAG(int, max_outcome_candidates, 4,
          "Maximum number of confirmed candidates for full-game continuation (default 4).");
ABSL_FLAG(int, continuation_samples, 64,
          "Number of full-game continuation paired replicates (default 64).");
ABSL_FLAG(bool, run_continuation, false,
          "If true, runs secondary synthetic full-game continuation diagnostic.");
ABSL_FLAG(std::string, continuation_mode, "unchanged",
          "Continuation mode: 'unchanged' or 'controlled'.");
ABSL_FLAG(std::string, control, "none",
          "Scripted control mode: 'none', 'positive', or 'nonreactive'.");
ABSL_FLAG(std::string, output_dir, "",
          "Directory for results, manifests, and telemetry records.");
ABSL_FLAG(bool, run_windfall_eval, false,
          "If true, runs Windfall-timing controller evaluation suite.");
ABSL_FLAG(std::string, previous_run_dir,
          "/run/media/warcr/Storage/dune_drl_runtime/diagnostics/run_20260915_120630",
          "Directory containing natural roots from previous run.");
ABSL_FLAG(std::string, opponent_checkpoint,
          "/run/media/warcr/Storage/dune_drl_runtime/round7/semantic_v3_ppo_b5300_to_b6400_20260914_233136/checkpoints/ppo_model_update_5300.pt",
          "Path to opponent model checkpoint (.pt) for other seats.");
ABSL_FLAG(int, max_bank_positions, 16,
          "Maximum distinct natural positions to select for evaluation (at most 16).");
ABSL_FLAG(int, windfall_continuation_samples, 64,
          "Number of paired continuation rollouts per position (default 64).");

namespace open_spiel {
namespace dune_diagnostics {

void RunDiagnosticsCli(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);

  std::string checkpoint = absl::GetFlag(FLAGS_checkpoint);
  std::string seat_checkpoints_str = absl::GetFlag(FLAGS_seat_checkpoints);
  int num_scenarios = absl::GetFlag(FLAGS_num_scenarios);
  int num_replicates = absl::GetFlag(FLAGS_num_replicates);
  uint64_t base_seed = absl::GetFlag(FLAGS_base_seed);
  float temperature = absl::GetFlag(FLAGS_temperature);
  bool smoke_test = absl::GetFlag(FLAGS_smoke_test);
  bool run_ablation = absl::GetFlag(FLAGS_run_ablation);
  bool run_diag2 = absl::GetFlag(FLAGS_run_diagnostic2);
  bool run_mining = absl::GetFlag(FLAGS_run_mining);
  int ordinary_games = absl::GetFlag(FLAGS_ordinary_games);
  int num_mining_games = absl::GetFlag(FLAGS_num_mining_games);
  if (ordinary_games > 0) num_mining_games = ordinary_games;

  int discovery_pairs = absl::GetFlag(FLAGS_discovery_pairs);
  int max_candidates = absl::GetFlag(FLAGS_max_candidates);
  int confirmation_samples = absl::GetFlag(FLAGS_confirmation_samples);
  int max_outcome_candidates = absl::GetFlag(FLAGS_max_outcome_candidates);
  int continuation_samples = absl::GetFlag(FLAGS_continuation_samples);

  bool run_continuation = absl::GetFlag(FLAGS_run_continuation);
  std::string continuation_mode = absl::GetFlag(FLAGS_continuation_mode);
  std::string control_mode = absl::GetFlag(FLAGS_control);
  std::string output_dir = absl::GetFlag(FLAGS_output_dir);
  bool run_windfall_eval = absl::GetFlag(FLAGS_run_windfall_eval);
  std::string previous_run_dir = absl::GetFlag(FLAGS_previous_run_dir);
  std::string opp_ckpt = absl::GetFlag(FLAGS_opponent_checkpoint);
  int max_bank_positions = absl::GetFlag(FLAGS_max_bank_positions);
  int windfall_continuation_samples = absl::GetFlag(FLAGS_windfall_continuation_samples);

  if (run_windfall_eval) {
    run_mining = false;
    run_diag2 = false;
    run_ablation = false;
    run_continuation = false;
  }

  if (smoke_test) {
    if (run_windfall_eval) {
      std::cout << "--- RUNNING BOUNDED WINDFALL SMOKE TEST (2 positions, 4 continuation replicates) ---\n";
      max_bank_positions = 2;
      windfall_continuation_samples = 4;
    } else {
      std::cout << "--- RUNNING BOUNDED SMOKE TEST (5 ordinary games, 4 candidates, 8 conf, 8 cont) ---\n";
      num_mining_games = 5;
      num_scenarios = 4;
      num_replicates = 4;
      discovery_pairs = 2;
      max_candidates = 4;
      confirmation_samples = 8;
      max_outcome_candidates = 2;
      continuation_samples = 8;
    }
  }

  // Explicit seat assignments (all 4 seats must be assigned)
  std::array<std::string, kNumPlayers> seat_ckpts;
  if (!seat_checkpoints_str.empty()) {
    std::vector<std::string> parts = absl::StrSplit(seat_checkpoints_str, ',');
    if (parts.size() == kNumPlayers) {
      for (size_t p = 0; p < kNumPlayers; ++p) seat_ckpts[p] = parts[p];
    } else {
      std::cerr << "Warning: --seat_checkpoints did not have 4 parts; falling back to --checkpoint\n";
      for (size_t p = 0; p < kNumPlayers; ++p) seat_ckpts[p] = checkpoint;
    }
  } else {
    for (size_t p = 0; p < kNumPlayers; ++p) seat_ckpts[p] = checkpoint;
  }

  std::cout << "=================================================================\n";
  std::cout << "=== DUNE: IMPERIUM FROZEN-POLICY DIAGNOSTICS SUITE            ===\n";
  std::cout << "=================================================================\n";
  std::cout << "Base Seed:            " << base_seed << "\n";
  std::cout << "Ordinary Games:       " << num_mining_games << " (unmodified self-play from NewInitialState)\n";
  std::cout << "Discovery Pairs:      " << discovery_pairs << "\n";
  std::cout << "Max Candidates:       " << max_candidates << "\n";
  std::cout << "Confirmation Samples: " << confirmation_samples << "\n";
  std::cout << "Continuation Samples: " << continuation_samples << "\n";
  std::cout << "Temperature:          " << temperature << "\n";
  std::cout << "Evaluated Model:      " << (checkpoint.empty() ? "<Scripted Controls>" : checkpoint) << "\n";
  if (run_windfall_eval) {
    std::cout << "Windfall Eval Mode:   ACTIVE\n";
    std::cout << "Previous Run Dir:     " << previous_run_dir << "\n";
    std::cout << "Opponent Model:       " << (opp_ckpt.empty() ? "<Same as candidate>" : opp_ckpt) << "\n";
    std::cout << "Max Bank Positions:   " << max_bank_positions << "\n";
    std::cout << "Continuation Samples: " << windfall_continuation_samples << "\n";
  }
  std::cout << "Seat Assignments:\n";
  for (int p = 0; p < kNumPlayers; ++p) {
    std::cout << "  - Seat " << p << ": " << (seat_ckpts[p].empty() ? "<Scripted Controls>" : seat_ckpts[p]) << "\n";
  }
  std::cout << "Output Dir:           " << (output_dir.empty() ? "<stdout>" : output_dir) << "\n";
  std::cout << "=================================================================\n\n";

  // Setup output directory
  if (!output_dir.empty()) {
    std::filesystem::create_directories(output_dir);
  }

  // 1. Select evaluator
  std::unique_ptr<IPolicyDiagnosticsEvaluator> evaluator;
  std::unique_ptr<IPolicyDiagnosticsEvaluator> opp_evaluator;
  if (!checkpoint.empty()) {
#ifdef OPEN_SPIEL_BUILD_WITH_LIBTORCH
    torch::Device device(torch::kCPU);
    if (torch::cuda::is_available()) {
      device = torch::Device(torch::kCUDA, 0);
      std::cout << "Using GPU Device: CUDA:0\n";
    } else {
      std::cout << "Using CPU Device\n";
    }
    evaluator = std::make_unique<NeuralPolicyDiagnosticsEvaluator>(checkpoint, device);
    if (!opp_ckpt.empty()) {
      opp_evaluator = std::make_unique<NeuralPolicyDiagnosticsEvaluator>(opp_ckpt, device);
    } else {
      opp_evaluator = std::make_unique<NeuralPolicyDiagnosticsEvaluator>(checkpoint, device);
    }
#else
    SpielFatalError("LibTorch is required to evaluate neural checkpoints.");
#endif
  } else if (control_mode == "positive") {
    std::cout << "Initializing Scripted Positive Control Evaluator...\n";
    evaluator = std::make_unique<ScriptedPositiveControlEvaluator>(0, 1);
    opp_evaluator = std::make_unique<ScriptedPositiveControlEvaluator>(0, 1);
  } else {
    std::cout << "Initializing Scripted Nonreactive Control Evaluator...\n";
    evaluator = std::make_unique<ScriptedNonreactiveControlEvaluator>(0, 1);
    opp_evaluator = std::make_unique<ScriptedNonreactiveControlEvaluator>(0, 1);
  }

  // Checkpoint SHA256 integrity checks
  const std::string kB7300ExpectedSha256 = "641c622015f4ee6cf13427cb1cca9ffced042274971e61856116ff3248aafde8";
  if (!checkpoint.empty() && checkpoint.find("ppo_model_update_7300.pt") != std::string::npos) {
    if (evaluator->ModelSha256() != kB7300ExpectedSha256) {
      SpielFatalError("B7300 SHA256 mismatch! Expected " + kB7300ExpectedSha256 +
                      " but got " + evaluator->ModelSha256());
    }
    std::cout << "Candidate B7300 SHA256 VERIFIED: " << evaluator->ModelSha256() << "\n";
  }
  const std::string kB5300ExpectedSha256 = "bed2103bd6f7acac516eeae0d13e13034f64d558318b5adba1a2db82332d754d";
  if (!opp_ckpt.empty() && opp_ckpt.find("ppo_model_update_5300.pt") != std::string::npos && opp_evaluator) {
    if (opp_evaluator->ModelSha256() != kB5300ExpectedSha256) {
      SpielFatalError("B5300 SHA256 mismatch! Expected " + kB5300ExpectedSha256 +
                      " but got " + opp_evaluator->ModelSha256());
    }
    std::cout << "Opponent B5300 SHA256 VERIFIED:  " << opp_evaluator->ModelSha256() << "\n";
  }

  // Write run manifest with explicit seat assignments and checksums
  if (!output_dir.empty()) {
    std::ofstream manifest_out(output_dir + "/run_manifest.json");
    manifest_out << "{\n"
                 << "  \"evaluated_checkpoint\": \"" << evaluator->ModelName() << "\",\n"
                 << "  \"evaluated_checkpoint_sha256\": \"" << evaluator->ModelSha256() << "\",\n"
                 << "  \"base_seed\": " << base_seed << ",\n"
                 << "  \"temperature\": " << temperature << ",\n"
                 << "  \"ordinary_games\": " << num_mining_games << ",\n"
                 << "  \"discovery_pairs\": " << discovery_pairs << ",\n"
                 << "  \"max_candidates\": " << max_candidates << ",\n"
                 << "  \"confirmation_samples\": " << confirmation_samples << ",\n"
                 << "  \"max_outcome_candidates\": " << max_outcome_candidates << ",\n"
                 << "  \"continuation_samples\": " << continuation_samples << ",\n"
                 << "  \"seats\": {\n";
    for (int p = 0; p < kNumPlayers; ++p) {
      std::string s_sha = seat_ckpts[p].empty() ? evaluator->ModelSha256() : ComputeFileSHA256(seat_ckpts[p]);
      manifest_out << "    \"seat_" << p << "\": {\"checkpoint\": \"" << seat_ckpts[p]
                   << "\", \"sha256\": \"" << s_sha << "\"}";
      if (p + 1 < kNumPlayers) manifest_out << ",\n";
      else manifest_out << "\n";
    }
    manifest_out << "  }\n}\n";
    std::cout << "Wrote run manifest to: " << output_dir << "/run_manifest.json\n";
  }

  // 2. Generate and validate scenario bank (if ablation or diag2 requested)
  std::vector<ScenarioConfig> bank;
  if (run_ablation || run_diag2 || run_continuation) {
    std::cout << "Generating scenario bank for timing/ablation...\n";
    int rejected_count = 0;
    bank = ScenarioGenerator::GenerateBank(num_scenarios, base_seed, &rejected_count);
    std::cout << "Generated " << bank.size() << " base scenarios (rejected candidates: "
              << rejected_count << ").\n";

    if (!output_dir.empty()) {
      std::ofstream scen_out(output_dir + "/scenario_manifest.json");
      scen_out << "[\n";
      for (size_t i = 0; i < bank.size(); ++i) {
        const auto& cfg = bank[i];
        scen_out << "  {\"scenario_id\": " << cfg.scenario_id
                 << ", \"family\": \"" << cfg.family_name
                 << "\", \"round\": " << cfg.round
                 << ", \"p_holder\": " << cfg.p_holder
                 << ", \"p_competitor\": " << cfg.p_competitor
                 << ", \"seed\": " << cfg.scenario_seed
                 << ", \"is_constructed\": true}";
        if (i + 1 < bank.size()) scen_out << ",\n";
        else scen_out << "\n";
      }
      scen_out << "]\n";
    }
  }

  // 3. Execute Actual-Play Intrigue Mining (Questions 2 & 3)
  IntrigueMiningSummary mining_summary;
  if (run_mining) {
    std::cout << "\n-----------------------------------------------------------------\n";
    std::cout << "Running Actual-Play Intrigue Mining Suite (" << num_mining_games
              << " ordinary games)...\n";
    std::cout << "-----------------------------------------------------------------\n";
    auto t_mine_start = std::chrono::steady_clock::now();

    mining_summary = RunOrdinaryGameIntrigueMining(
        evaluator.get(), num_mining_games, base_seed, discovery_pairs,
        max_candidates, confirmation_samples, max_outcome_candidates,
        continuation_samples, temperature);

    auto t_mine_end = std::chrono::steady_clock::now();
    double mine_seconds = std::chrono::duration<double>(t_mine_end - t_mine_start).count();
    std::cout << "Mining completed in " << std::fixed << std::setprecision(2)
              << mine_seconds << "s.\n\n";

    std::cout << "=================================================================\n";
    std::cout << "ACTUAL-PLAY INTRIGUE MINING RESULTS SUMMARY\n";
    std::cout << "=================================================================\n";
    std::cout << "Ordinary Games Simulated:          " << mining_summary.games_simulated << "\n";
    std::cout << "Total Resource Intrigues Played:    " << mining_summary.total_resource_intrigue_plays << "\n";
    std::cout << "  - Windfall (+2 Solari):           " << mining_summary.windfall_plays << "\n";
    std::cout << "  - Water Peddlers Union (+1 Water):" << mining_summary.water_peddlers_union_plays << "\n";
    std::cout << "Candidates Discovered:              " << mining_summary.candidates_discovered << "\n";
    std::cout << "Candidates Tested Confirmation:     " << mining_summary.candidates_tested_confirmation << "\n";
    std::cout << "Confirmed Divergent Candidates:     " << mining_summary.candidates_confirmed_divergent << "\n";
    std::cout << "Candidates Tested Continuation:     " << mining_summary.candidates_tested_continuation << "\n";
    std::cout << "Run Invalidation Failure:           " << (mining_summary.run_has_invalidation_failure ? "TRUE" : "FALSE") << "\n";
    if (mining_summary.run_has_invalidation_failure) {
      std::cout << "Invalidation Reason:                " << mining_summary.invalidation_reason << "\n";
    }
    std::cout << "\n";

    if (!mining_summary.candidates.empty()) {
      std::cout << "Discovered Candidate Details:\n";
      for (const auto& c : mining_summary.candidates) {
        std::cout << "  [Cand " << c.candidate_id << " | Rd " << c.round << " | "
                  << c.intrigue_name << "]: Holder=" << c.holder_leader << " (P" << c.holder_player
                  << ", " << c.holder_solari << "S/" << c.holder_water << "W) -> Opponent="
                  << c.opponent_leader << " (P" << c.opponent_player << ")\n";
        std::cout << "    Replay Verified: " << (c.replay_verified ? "PASSED" : ("FAILED (" + c.replay_failure_reason + ")")) << "\n";
        std::cout << "    Discovery Early: " << c.discovery_opp_action_early_name
                  << " vs Hold: " << c.discovery_opp_action_hold_name << "\n";
        std::cout << "    Target Action: " << c.target_action_name << " (Predicted dir: " << (c.predicted_direction > 0 ? "+1" : "-1") << ")\n";
        std::cout << "    Confirmation Divergence Rate: " << (c.confirmed_divergent_rate * 100.0)
                  << "% (" << c.confirmed_divergent_count << "/" << c.confirmation_samples << " fresh seeds)\n";
        std::cout << "    Target Prob Shift: " << (c.delta_prob * 100.0) << " pp (Raw p=" << c.p_value_raw
                  << ", Adj p=" << c.p_value_adjusted << ", Confirmed=" << (c.confirmed_response ? "YES" : "NO") << ")\n";
        if (c.continuation_samples > 0) {
          std::cout << "    Continuation Outcome: Delta Util=" << c.delta_holder_utility
                    << " (Early=" << c.mean_early_holder_utility << ", Hold=" << c.mean_hold_holder_utility
                    << "), Adj p=" << c.p_value_utility_adjusted << ", Delta VP=" << c.delta_holder_vp
                    << ", Verdict=" << c.verdict << "\n";
        }
        std::cout << "\n";
      }
    }

    if (!output_dir.empty()) {
      // 1. Write summary JSON
      std::ofstream mine_out(output_dir + "/mining_summary.json");
      mine_out << FormatIntrigueMiningReportJson(mining_summary, evaluator->ModelName(),
                                                evaluator->ModelSha256());

      // 2. Write roots JSONL
      std::ofstream roots_out(output_dir + "/mining_roots.jsonl");
      for (const auto& r : mining_summary.roots) {
        roots_out << "{\"candidate_id\": " << r.candidate_id
                  << ", \"game_id\": " << r.game_id
                  << ", \"game_seed\": " << r.game_seed
                  << ", \"round\": " << r.round
                  << ", \"step_index\": " << r.step_index
                  << ", \"holder_player\": " << r.holder_player
                  << ", \"holder_leader\": \"" << EscapeJson(r.holder_leader) << "\""
                  << ", \"holder_solari\": " << r.holder_solari
                  << ", \"holder_water\": " << r.holder_water
                  << ", \"holder_spice\": " << r.holder_spice
                  << ", \"target_opponent\": " << r.target_opponent
                  << ", \"opponent_leader\": \"" << EscapeJson(r.opponent_leader) << "\""
                  << ", \"opponent_solari\": " << r.opponent_solari
                  << ", \"opponent_water\": " << r.opponent_water
                  << ", \"opponent_spice\": " << r.opponent_spice
                  << ", \"opponent_agents\": " << r.opponent_agents
                  << ", \"intrigue_action\": " << r.intrigue_action
                  << ", \"intrigue_name\": \"" << EscapeJson(r.intrigue_name) << "\""
                  << ", \"history_len\": " << r.history_actions.size()
                  << ", \"replay_verified\": " << (r.replay_verified ? "true" : "false")
                  << ", \"replay_failure_reason\": \"" << EscapeJson(r.replay_failure_reason) << "\"}\n";
      }

      // 3. Write confirmation pairs JSONL
      std::ofstream conf_out(output_dir + "/mining_confirmation_pairs.jsonl");
      for (const auto& c : mining_summary.confirmation_records) {
        conf_out << "{\"candidate_id\": " << c.candidate_id
                 << ", \"pair_index\": " << c.pair_index
                 << ", \"chance_seed\": " << c.chance_seed
                 << ", \"holder_player\": " << c.holder_player
                 << ", \"target_opponent\": " << c.target_opponent
                 << ", \"act_early\": " << c.act_early
                 << ", \"act_early_name\": \"" << EscapeJson(c.act_early_name) << "\""
                 << ", \"act_hold\": " << c.act_hold
                 << ", \"act_hold_name\": \"" << EscapeJson(c.act_hold_name) << "\""
                 << ", \"reached_target_early\": " << (c.reached_target_early ? "true" : "false")
                 << ", \"reached_target_hold\": " << (c.reached_target_hold ? "true" : "false")
                 << ", \"indicator_early\": " << c.indicator_early
                 << ", \"indicator_hold\": " << c.indicator_hold
                 << ", \"indicator_diff\": " << c.indicator_diff
                 << ", \"is_valid\": " << (c.is_valid ? "true" : "false") << "}\n";
      }

      // 4. Write continuation pairs JSONL
      std::ofstream cont_pairs_out(output_dir + "/mining_continuation_pairs.jsonl");
      for (const auto& c : mining_summary.continuation_records) {
        cont_pairs_out << "{\"candidate_id\": " << c.candidate_id
                       << ", \"pair_index\": " << c.pair_index
                       << ", \"chance_seed\": " << c.chance_seed
                       << ", \"holder_player\": " << c.holder_player
                       << ", \"target_opponent\": " << c.target_opponent
                       << ", \"branch_a_terminal\": " << (c.branch_a_terminal ? "true" : "false")
                       << ", \"branch_a_status\": \"" << c.branch_a_status << "\""
                       << ", \"branch_b_terminal\": " << (c.branch_b_terminal ? "true" : "false")
                       << ", \"branch_b_status\": \"" << c.branch_b_status << "\""
                       << ", \"branch_b_intrigue_played_later\": " << (c.branch_b_intrigue_played_later ? "true" : "false")
                       << ", \"branch_b_intrigue_play_round\": " << c.branch_b_intrigue_play_round
                       << ", \"branch_b_intrigue_play_step\": " << c.branch_b_intrigue_play_step
                       << ", \"holder_utility_diff\": " << c.holder_utility_diff
                       << ", \"holder_vp_diff\": " << c.holder_vp_diff
                       << ", \"is_valid\": " << (c.is_valid ? "true" : "false")
                       << ", \"branch_a_returns\": ["
                       << c.branch_a_returns[0] << ", " << c.branch_a_returns[1] << ", "
                       << c.branch_a_returns[2] << ", " << c.branch_a_returns[3] << "]"
                       << ", \"branch_b_returns\": ["
                       << c.branch_b_returns[0] << ", " << c.branch_b_returns[1] << ", "
                       << c.branch_b_returns[2] << ", " << c.branch_b_returns[3] << "]"
                       << ", \"branch_a_vps\": ["
                       << c.branch_a_vps[0] << ", " << c.branch_a_vps[1] << ", "
                       << c.branch_a_vps[2] << ", " << c.branch_a_vps[3] << "]"
                       << ", \"branch_b_vps\": ["
                       << c.branch_b_vps[0] << ", " << c.branch_b_vps[1] << ", "
                       << c.branch_b_vps[2] << ", " << c.branch_b_vps[3] << "]"
                       << "}\n";
      }

      std::cout << "Wrote complete intrigue mining artifacts to: " << output_dir << "\n";
    }
  }

  // 4. Execute Diagnostic 2: Timing Preference / Early-Play vs Repeated Sampling (Question 1)
  if (run_diag2 && !bank.empty()) {
    std::cout << "\n-----------------------------------------------------------------\n";
    std::cout << "Running Diagnostic 2: Timing Preference / Early-Play vs Repeated Sampling...\n";
    std::cout << "-----------------------------------------------------------------\n";
    auto t_diag2_start = std::chrono::steady_clock::now();

    int diag2_reps = std::min(num_replicates, 16);
    auto [diag2_records, diag2_summary] = RunDiagnostic2Suite(
        bank, evaluator.get(), diag2_reps, base_seed, temperature);

    auto t_diag2_end = std::chrono::steady_clock::now();
    double diag2_seconds = std::chrono::duration<double>(t_diag2_end - t_diag2_start).count();
    std::cout << "Diagnostic 2 completed in " << std::fixed << std::setprecision(2)
              << diag2_seconds << "s.\n\n";

    std::cout << "=================================================================\n";
    std::cout << "DIAGNOSTIC 2 RESULTS SUMMARY (Timing Preference)\n";
    std::cout << "=================================================================\n";
    std::cout << "Total Traces Evaluated:         " << diag2_summary.total_traces_evaluated << "\n";
    std::cout << "Total Eligible P2 Decisions:    " << diag2_summary.total_decisions_evaluated << "\n";
    std::cout << "Windfall Rank 1 Frequency:      " << (diag2_summary.windfall_rank1_rate * 100.0)
              << "% (" << diag2_summary.windfall_rank1_count << " decisions)\n";
    std::cout << "Mean Windfall Probability:      " << (diag2_summary.mean_windfall_prob * 100.0) << "%\n";
    std::cout << "Mean Windfall Rank:             " << diag2_summary.mean_windfall_rank << "\n\n";

    std::cout << "Turn 1 vs Later Decisions:\n";
    std::cout << "  Sampled: Played on Turn 1 = " << diag2_summary.traces_played_initial_turn_sampled
              << ", Held Past Turn 1 = " << diag2_summary.traces_held_past_turn1_sampled << "\n";
    std::cout << "  Greedy:  Played on Turn 1 = " << diag2_summary.traces_played_initial_turn_greedy
              << ", Held Past Turn 1 = " << diag2_summary.traces_held_past_turn1_greedy << "\n\n";

    if (!output_dir.empty()) {
      std::ofstream d2_out(output_dir + "/diagnostic2_decisions.jsonl");
      for (const auto& r : diag2_records) {
        d2_out << "{\"scenario_id\": " << r.scenario_id
               << ", \"replicate\": " << r.replicate_idx
               << ", \"round\": " << r.round
               << ", \"turn_idx\": " << r.agent_turn_index
               << ", \"decision_in_turn\": " << r.decision_in_turn
               << ", \"decision_idx\": " << r.decision_index
               << ", \"opp_idx\": " << r.opportunity_index
               << ", \"is_initial_turn\": " << (r.is_initial_turn ? "true" : "false")
               << ", \"legal_actions\": " << r.legal_actions_count
               << ", \"windfall_prob\": " << r.windfall_prob
               << ", \"windfall_rank\": " << r.windfall_rank
               << ", \"argmax_action\": \"" << r.argmax_action_name << "\""
               << ", \"selected_action\": \"" << r.selected_action_name << "\""
               << ", \"selected_windfall\": " << (r.selected_windfall ? "true" : "false")
               << ", \"greedy\": " << (r.was_greedy ? "true" : "false") << "}\n";
      }

      std::ofstream d2_sum_out(output_dir + "/diagnostic2_summary.json");
      d2_sum_out << FormatDiagnostic2ReportJson(diag2_summary, evaluator->ModelName(),
                                              evaluator->ModelSha256());
    }
  }

  // 5. Execute Synthetic Ablation Suite (Diagnostic 1) only if explicitly requested
  if (run_ablation && !bank.empty()) {
    std::cout << "\n-----------------------------------------------------------------\n";
    std::cout << "Running Synthetic Ablation Suite (Diagnostic 1: EARLY vs HOLD)...\n";
    std::cout << "-----------------------------------------------------------------\n";
    std::vector<ReplicateResult> replicate_records;
    auto [summaries, agg] = RunDiagnostic1Suite(
        bank, evaluator.get(), num_replicates, base_seed, temperature, &replicate_records);

    if (!output_dir.empty()) {
      std::ofstream reps_out(output_dir + "/replicate_records.jsonl");
      for (const auto& r : replicate_records) {
        reps_out << "{\"scenario_id\": " << r.scenario_id
                 << ", \"replicate\": " << r.replicate_idx
                 << ", \"branch_a_action\": \"" << r.branch_a_action_name << "\""
                 << ", \"branch_a_sm\": " << (r.branch_a_chose_swordmaster ? "true" : "false")
                 << ", \"branch_b_action\": \"" << r.branch_b_action_name << "\""
                 << ", \"branch_b_sm\": " << (r.branch_b_chose_swordmaster ? "true" : "false")
                 << ", \"delta\": " << r.delta_swordmaster
                 << ", \"steps_a\": " << r.branch_a_steps
                 << ", \"steps_b\": " << r.branch_b_steps << "}\n";
      }

      std::ofstream report_out(output_dir + "/diagnostic1_summary.json");
      report_out << FormatDiagnostic1ReportJson(agg, summaries, evaluator->ModelName(),
                                              evaluator->ModelSha256());
    }
  }

  // 6. Execute Windfall-Timing Controller Evaluation Suite
  if (run_windfall_eval) {
    std::cout << "\n=================================================================\n";
    std::cout << "=== WINDFALL CONTROLLER EVALUATION SUITE                      ===\n";
    std::cout << "=================================================================\n";

    auto game = open_spiel::LoadGame("dune_imperium");

    std::cout << "\n--- Phase 1: Reconstructing & Verifying Natural Roots ---\n";
    std::vector<WindfallBankRecord> all_roots = ReconstructAndVerifyWindfallRoots(
        previous_run_dir, game, evaluator.get(), base_seed, temperature);

    std::cout << "\n--- Phase 2: Selecting Evaluation Bank (max " << max_bank_positions << " positions) ---\n";
    std::vector<WindfallBankRecord> selected_bank = SelectEvaluationBank(all_roots, max_bank_positions);
    std::cout << "Bank selection completed: " << selected_bank.size() << " positions chosen.\n";

    std::cout << "\n--- Phase 3: Executing Paired Continuations (" << windfall_continuation_samples
              << " pairs/position) ---\n";
    auto t_eval_start = std::chrono::steady_clock::now();

    WindfallEvaluationSummary win_summary = RunWindfallEvaluationSuite(
        game, evaluator.get(), opp_evaluator.get(), all_roots, selected_bank,
        output_dir, windfall_continuation_samples, base_seed, temperature);

    auto t_eval_end = std::chrono::steady_clock::now();
    double eval_seconds = std::chrono::duration<double>(t_eval_end - t_eval_start).count();

    std::cout << "\n=================================================================\n";
    std::cout << "WINDFALL CONTROLLER EVALUATION RESULTS SUMMARY\n";
    std::cout << "=================================================================\n";
    std::cout << "Positions Selected:                 " << win_summary.positions_selected << "\n";
    std::cout << "Continuations Per Position:         " << win_summary.continuations_per_position << "\n";
    std::cout << "Total Pairs Evaluated:              " << win_summary.total_continuation_pairs << "\n";
    std::cout << "Mean Utility Diff (Ctrl - Base):    " << win_summary.mean_placement_utility_diff << "\n";
    std::cout << "Utility Std Error:                  " << win_summary.placement_utility_std_err << "\n";
    std::cout << "Utility 95% Confidence Interval:    [" << win_summary.placement_utility_ci_lower << ", " << win_summary.placement_utility_ci_upper << "]\n";
    std::cout << "Mean VP Diff (Ctrl - Base):         " << win_summary.mean_holder_vp_diff << "\n";
    std::cout << "Early Plays Prevented:              " << win_summary.early_plays_prevented << "\n";
    std::cout << "Windfall Plays Executed:            " << win_summary.windfall_plays_executed << "\n";
    std::cout << "Necessary Spending Unlocked:        " << win_summary.necessary_spending_unlocked << "\n";
    std::cout << "Purchases Already Affordable:       " << win_summary.purchases_already_affordable << "\n";
    std::cout << "End-Turn Retentions:                " << win_summary.end_turn_retentions << "\n";
    std::cout << "Secrets Exceptions Triggered:       " << win_summary.secrets_exceptions_triggered << "\n";
    std::cout << "Terminal Value Plays:               " << win_summary.terminal_value_exceptions_triggered << "\n";
    std::cout << "Controller Fallbacks:               " << win_summary.fallbacks_triggered << "\n";
    std::cout << "Evaluation Verdict:                 " << win_summary.verdict << "\n";
    std::cout << "Verdict Explanation:                " << win_summary.verdict_explanation << "\n";
    std::cout << "Evaluation Completed in:            " << std::fixed << std::setprecision(2) << eval_seconds << "s\n";
    std::cout << "=================================================================\n\n";
  }

  std::cout << "=================================================================\n";
  std::cout << "DIAGNOSTICS SUITE EXECUTION COMPLETED.\n";
  std::cout << "=================================================================\n";
}

} // namespace dune_diagnostics
} // namespace open_spiel

int main(int argc, char** argv) {
  open_spiel::dune_diagnostics::RunDiagnosticsCli(argc, argv);
  return 0;
}
