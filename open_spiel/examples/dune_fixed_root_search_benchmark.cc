// Step 1: Fixed-Root Search Benchmark Harness
// Evaluates search variants against a 512-rollout paired reference across ~300 stratified census roots,
// followed by a 1,024-rollout re-check pass on disagreement roots.

#include <cstdint>
#include <iostream>
#include <string>

#include "open_spiel/abseil-cpp/absl/flags/flag.h"
#include "open_spiel/abseil-cpp/absl/flags/parse.h"
#include "open_spiel/examples/dune_fixed_root_search_benchmark_internal.h"

ABSL_FLAG(std::string, model_checkpoint, "", "Path to model checkpoint.");
ABSL_FLAG(std::string, sampled_roots_json, "", "Path to census_sampled_roots.json.");
ABSL_FLAG(std::string, obs_binary, "", "Path to census_obs.bin matching sampled roots.");
ABSL_FLAG(std::string, output_json, "", "Path to output JSON receipt (required).");
ABSL_FLAG(int, threads, 14, "Number of worker threads.");
ABSL_FLAG(int, num_evaluators, 6, "Number of batched evaluator coordinators.");
ABSL_FLAG(int, target_batch_size, 128, "Evaluator batch size.");
ABSL_FLAG(int, top_k_actions, 3, "Number of top candidates.");
ABSL_FLAG(double, min_override_margin, 0.15, "Override margin threshold in utility units.");
ABSL_FLAG(uint64_t, master_seed, 20261001ULL, "Master seed for domain-separated derivation.");
ABSL_FLAG(std::string, replay_trajectory_jsonl, "",
          "Path to recorded tournament trajectory_meta.jsonl for held-out policy cross-entropy evaluation.");
ABSL_FLAG(std::string, replay_obs_bin, "",
          "Path to recorded tournament trajectory_obs.bin matching replay_trajectory_jsonl.");
ABSL_FLAG(int, max_roots, -1, "Maximum number of roots to benchmark (-1 for all).");
ABSL_FLAG(bool, leaf_check_only, false, "Run only V2_16_Paired vs Ref512 for leaf check.");
ABSL_FLAG(bool, rollout_amp, false, "Use AMP in evaluator rollouts (default false).");
ABSL_FLAG(bool, allow_tf32, false, "Allow TF32 in evaluator (default false).");
ABSL_FLAG(double, temperature, 1.0, "Temperature applied to capped logits in replay mode.");
ABSL_FLAG(bool, dump_per_state, false, "Dump per-state candidate probabilities and entropy in output JSON.");
ABSL_FLAG(int, num_blocks, 8, "Number of residual blocks in model (8, 12, or 16).");
ABSL_FLAG(bool, sh_offline_check, false,
          "Run R7-39 Step 6 Successive Halving exploratory offline check (raw, sh_192_incumbent, sh_384_incumbent, sh_384_k5, Ref512).");
ABSL_FLAG(bool, sh_budget_check, false,
          "Run fixed-position SH budget check (raw, sh_192, sh_384, sh_768, sh_768_k5, sh_768_rep, Ref2048).");
ABSL_FLAG(bool, skip_game_string_check, false,
          "Allow mismatched game_string metadata; replay, seat and observation checks remain strict.");
ABSL_FLAG(bool, skip_recheck_1024, false,
          "Skip the 1,024-rollout disagreement re-check pass.");
ABSL_FLAG(std::string, base_receipt_json, "",
          "Optional path to existing sh_budget_check_sweep_receipt.json to reuse completed variants and incrementally extend Ref2048.");

// Mode 1: Raw Greedy Self-Play Root & Preservation Capture Mode
ABSL_FLAG(int, capture_self_play_games, 0,
          "Number of 4x greedy self-play games to capture placement roots and preservation pool from.");
ABSL_FLAG(int, capture_roots_per_seat_game, 2, "Max placement roots sampled per seat-game.");
ABSL_FLAG(int, capture_pres_per_seat_game, 4,
          "Max preservation decisions sampled per seat-game across all non-forced decision types.");
ABSL_FLAG(int, capture_max_placement_roots, 0,
          "Optional cap on total sampled placement roots across all games (0 = no cap).");
ABSL_FLAG(std::string, capture_out_dir, "",
          "Output directory for captured roots and preservation dataset.");
ABSL_FLAG(bool, auto_pass_empty_combat_intrigue, true,
          "Value for DuneImperiumState::SetAutoPassEmptyCombatIntrigueHands (false for legacy R7-41 roots).");

// Mode 2: Parallel-Roots sh_768_k5_allbuy Labelling Mode
ABSL_FLAG(bool, sh_distill_label, false, "Run parallel-roots sh_768_k5_allbuy labelling mode.");
ABSL_FLAG(int, parallel_roots, 1, "Number of roots to search concurrently across worker threads.");
ABSL_FLAG(bool, use_bounded_pool_when_unclocked, false,
          "Use bounded world pool in RunClockedPairedRolloutStage when max_time_ms <= 0.");
ABSL_FLAG(int, max_inflight_worlds, 32, "Max inflight worlds per root when using bounded pool.");
ABSL_FLAG(bool, placement_roots_only, false,
          "Filter loaded roots to AgentPrimary and AgentContinuation only.");
ABSL_FLAG(uint64_t, label_seed_domain, 0x0701ULL, "Seed domain for sh_distill_label.");

// Mode 3: True-State Tie-Breaker & History Replay Verification Mode
ABSL_FLAG(std::string, tie_breaker_games_jsonl, "",
          "Path to games.jsonl from dune_population_eval for history replay verification and Step 1.6 tie-breaker.");
ABSL_FLAG(bool, tie_breaker_run_rollouts, false,
          "If true (and r <= 1.5 or forced), run 512 paired true-state rollouts on up to 300 changed student decisions.");
ABSL_FLAG(int, tie_breaker_max_sample, 300,
          "Max changed student decisions to sample for true-state rollouts.");
ABSL_FLAG(int, tie_breaker_rollouts, 512, "Paired full-game rollouts per move in tie-breaker.");

int main(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);

  using namespace open_spiel::dune_imperium::fixed_root_benchmark;

  // Dispatch to R7-45 Step 1 Mode 1: Raw Greedy Self-Play Root & Preservation Capture Mode
  if (absl::GetFlag(FLAGS_capture_self_play_games) > 0) {
    return RunSelfPlayRootAndPreservationCapture();
  }

  // Dispatch to R7-45 Step 1 Mode 3: True-State Tie-Breaker & History Replay Verification Mode
  if (!absl::GetFlag(FLAGS_tie_breaker_games_jsonl).empty()) {
    return RunTrueStateTieBreakerAndHistoryReplay();
  }

  // Dispatch to R7-45 Step 1 Mode 2: Parallel-Roots sh_768_k5_allbuy Labelling Mode
  if (absl::GetFlag(FLAGS_sh_distill_label)) {
    return RunSuccessiveHalvingDistillLabel();
  }

  // Dispatch to held-out trajectory replay mode if --replay_trajectory_jsonl is provided
  if (!absl::GetFlag(FLAGS_replay_trajectory_jsonl).empty()) {
    const std::string model_path = absl::GetFlag(FLAGS_model_checkpoint);
    const std::string output_path = absl::GetFlag(FLAGS_output_json);
    std::string traj_meta = absl::GetFlag(FLAGS_replay_trajectory_jsonl);
    std::string traj_obs = absl::GetFlag(FLAGS_replay_obs_bin);
    if (traj_obs.empty()) {
      traj_obs = absl::GetFlag(FLAGS_obs_binary);
    }
    if (model_path.empty() || traj_meta.empty() || traj_obs.empty()) {
      std::cerr << "Usage for replay: dune_fixed_root_search_benchmark --model_checkpoint=... --replay_trajectory_jsonl=... --replay_obs_bin=... [--output_json=...]\n";
      return 1;
    }
    return RunHeldOutReplayEvaluation(model_path, traj_meta, traj_obs, output_path);
  }

  return RunFixedRootBenchmark();
}
