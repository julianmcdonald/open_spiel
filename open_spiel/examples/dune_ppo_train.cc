// Native LibTorch PPO trainer for Dune Imperium.
//
// This executable is intentionally separate from dune_benchmark.cc. It follows
// the OpenSpiel PyTorch PPO structure: collect an on-policy rollout batch,
// freeze it, run shuffled PPO epochs/minibatches, then sync the inference model.

#include <algorithm>
#include <atomic>
#include <chrono>
#include <csignal>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <deque>
#include <array>
#include <numeric>
#include <sstream>
#include <iomanip>

#include <random>
#include <set>
#include <shared_mutex>
#include <string>
#include <thread>
#include <unordered_set>
#include <vector>

#if defined(__GLIBC__)
#include <malloc.h>
#endif

#include "open_spiel/abseil-cpp/absl/flags/flag.h"
#include "dune_ppo_training_flags.h"
#include "open_spiel/abseil-cpp/absl/flags/parse.h"
#include "open_spiel/abseil-cpp/absl/flags/reflection.h"
#include "open_spiel/abseil-cpp/absl/strings/str_format.h"
#include "open_spiel/abseil-cpp/absl/strings/str_split.h"
#include "open_spiel/games/dune_imperium/dune_imperium.h"
#include "open_spiel/games/dune_imperium/dune_card_telemetry.h"

#include "open_spiel/games/dune_imperium/dune_imperium_util.h"
#include "open_spiel/spiel.h"

#ifdef OPEN_SPIEL_BUILD_WITH_LIBTORCH
#include <c10/cuda/CUDACachingAllocator.h>
#include <torch/torch.h>

#include "dune_network.h"
#include "dune_seed_utils.h"
#include "dune_sha256.h"
#include "dune_ppo_training_utils.h"
#include "dune_ppo_numerical_parity.h"
#include "dune_search_label_buffer.h"
#include "dune_search_routing.h"
#include "dune_evaluator.h"  // DuneNNEvaluator for online-collection snapshot inference
#include "dune_opponent_pool.h"
#include "dune_compound_turn_search.h"
#include "dune_wide_rollout_search.h"
#include "dune_batched_evaluator.h"
#include "dune_targeted_buy.h"
#include "open_spiel/utils/json.h"
#endif

#include "dune_ppo_train_run_context.h"

ABSL_FLAG(std::string, game, "dune_imperium", "The OpenSpiel game to train.");
ABSL_FLAG(bool, single_learner_training, false,
          "Enable single-seat rotating learner collection for opponent-pool study.");
ABSL_FLAG(bool, targeted_buy_exploration, false,
          "Enable targeted-buy exploration for Scientific Breakthrough and Stitched Horror.");
ABSL_FLAG(int, targeted_buy_quota_per_card, 8,
          "Targeted buy quota per card per update (default 8 distinct games).");
ABSL_FLAG(int, targeted_buy_max_round, 6,
          "Maximum round for targeted buy exploration (default 6, inclusive).");
ABSL_FLAG(std::string, targeted_buy_forced_csv, "",
          "Path to write targeted buy normalization records CSV.");
ABSL_FLAG(std::string, targeted_buy_policy_advantage, "gae",
          "Policy loss advantage for targeted forced buys: gae | full_return (default gae).");
ABSL_FLAG(std::string, targeted_buy_telemetry_csv, "",
          "Path to write targeted buy update-path telemetry CSV.");
ABSL_FLAG(std::string, training_opponent_pool, "",
          "Comma-separated list of frozen opponent checkpoint paths for training pool.");
ABSL_FLAG(int, threads, 64, "Rollout worker threads.");
ABSL_FLAG(int, total_updates, 1000, "Number of PPO collect/update cycles.");
ABSL_FLAG(int, rollout_transitions, 32768,
          "Minimum on-policy transitions collected before each PPO update.");
ABSL_FLAG(double, learning_rate, 2.5e-4, "AdamW learning rate.");
ABSL_FLAG(double, new_block_lr, 0.0,
          "AdamW learning rate for newly appended residual blocks (<=0 disables).");
ABSL_FLAG(int, new_block_first_index, 8,
          "0-indexed first residual block belonging to new_block_lr group (e.g. 8 for res9-res12).");
ABSL_FLAG(bool, anneal_lr, true, "Linearly anneal learning rate over updates.");
ABSL_FLAG(double, gamma, 1.0, "Discount factor.");
ABSL_FLAG(double, gae_lambda, 1.0, "GAE lambda.");
ABSL_FLAG(double, weight_decay, 0.0, "AdamW weight decay for torso/value params.");
ABSL_FLAG(double, policy_weight_decay, 0.0,
          "AdamW weight decay for policy-head params.");

ABSL_FLAG(int, eval_batch_size, 32, "Target inference batch size.");
ABSL_FLAG(int, eval_timeout_ms, 2, "Inference batch timeout in milliseconds.");
ABSL_FLAG(std::string, verification_sidecar_path, "",
          "Direct path for verification sidecar JSONL. Records full-precision KLs "
          "and complete input digests.");
ABSL_FLAG(bool, evaluator_device_synchronize, true,
          "Use whole-device CUDA synchronize after evaluator D2H copies.");
ABSL_FLAG(bool, deterministic, true, "Enable strict PyTorch/LibTorch deterministic algorithms.");
ABSL_FLAG(bool, deterministic_rollout_eval, false, "Use deterministic batch-1 rollout evaluation.");
ABSL_FLAG(std::string, numerical_parity_output, "",
          "Fresh output JSON for the read-only rollout-BF16 versus learner-FP32 "
          "parity gate. Requires --diagnostics_only and --init_mode=diagnostic; "
          "the path must not already exist.");
ABSL_FLAG(std::string, numerical_parity_registration_id, "",
          "Required immutable registration identifier recorded in the parity artifact.");
ABSL_FLAG(std::string, numerical_parity_source_root, "",
          "Required OpenSpiel source root containing the fixed parity source "
          "provenance list (normally third_party/open_spiel).");
ABSL_FLAG(std::string, numerical_parity_source_sha256, "",
          "Required registered SHA-256 of the canonical fixed parity source list.");
ABSL_FLAG(int, numerical_parity_min_rows, -1,
          "Required minimum total transition rows for a parity PASS.");
ABSL_FLAG(int, numerical_parity_min_nontrivial_rows, -1,
          "Required minimum rows with at least two legal actions for a parity PASS.");
ABSL_FLAG(double, numerical_parity_max_abs_logprob_delta, -1.0,
          "Required maximum absolute chosen-action log-probability drift.");
ABSL_FLAG(double, numerical_parity_max_full_kl, -1.0,
          "Required maximum for each direction of full legal categorical KL.");
ABSL_FLAG(double, numerical_parity_max_raw_mass_residual, -1.0,
          "Required maximum absolute raw legal-probability mass residual "
          "before v2 double-logsumexp KL normalization.");
ABSL_FLAG(double, numerical_parity_min_ratio, -1.0,
          "Required minimum unchanged-weight chosen-action probability ratio.");
ABSL_FLAG(double, numerical_parity_max_ratio, -1.0,
          "Required maximum unchanged-weight chosen-action probability ratio.");

ABSL_FLAG(double, shaped_reward_weight, 0.0,
          "Weight for intermediate VP shaped rewards.");
ABSL_FLAG(double, tleilaxu_breadcrumb_weight, 0.0,
          "Weight for Tleilaxu levels 5/6 breadcrumbs.");
ABSL_FLAG(double, tleilaxu_level7_breadcrumb_weight, 0.0,
          "Weight for Tleilaxu level 7 breadcrumb.");
ABSL_FLAG(bool, allow_shaping, false,
          "Allow experimental reward shaping flags to be non-zero.");
ABSL_FLAG(double, reward_scale, 4.0,
          "Divide shaped plus terminal rewards by this value.");
ABSL_FLAG(std::string, terminal_reward_mode, "placement",
          "Terminal reward scheme for training: 'placement' (+2.25, +0.25, -0.75, -1.75) "
          "or 'first_place' (+3.0 for 1st, -1.0 for 2nd/3rd/4th).");
ABSL_FLAG(double, round7_speed_bonus, 0.0,
          "Terminal reward bonus for candidate winning on or before round 7.");

ABSL_FLAG(std::string, model_checkpoint, "dune_ppo_model.pt",
          "Model checkpoint to load/save.");
ABSL_FLAG(std::string, optim_checkpoint, "dune_ppo_optimizer.pt",
          "Optimizer checkpoint to load/save.");
ABSL_FLAG(std::string, artifact_manifest, "",
          "Path to the Phase 1 baseline artifact manifest.json for validation.");
ABSL_FLAG(std::string, run_prefix, "dune_ppo",
          "Prefix for rotating checkpoints.");
ABSL_FLAG(int, checkpoint_interval, 10,
          "Save rotating checkpoints every N PPO updates. <=0 disables.");
ABSL_FLAG(bool, save_final_checkpoint, true,
          "Save final model and optimizer checkpoints.");
ABSL_FLAG(bool, save_pilot_checkpoints, true,
          "Save early pilot checkpoints at step 10, 25, 50.");
ABSL_FLAG(bool, allow_reward_transition, false,
          "Allow validated reward continuation transition from source checkpoint.");
ABSL_FLAG(std::string, reward_transition_source_fingerprint, "",
          "Expected source config fingerprint for reward continuation transition.");
ABSL_FLAG(int64_t, reward_transition_source_rollout_games, 0,
          "Expected source rollout_games when transitioning configuration.");
ABSL_FLAG(bool, allow_gae_transition, false,
          "Allow intentional transition of gae_lambda from a verified parent checkpoint.");
ABSL_FLAG(double, gae_transition_source_lambda, 1.0,
          "Expected source gae_lambda in parent manifest for transition.");
ABSL_FLAG(std::string, gae_transition_source_fingerprint, "",
          "Expected source config fingerprint for GAE lambda transition.");
ABSL_FLAG(bool, allow_learning_rate_transition, false,
          "Allow intentional transition of learning_rate from a verified parent checkpoint.");
ABSL_FLAG(double, learning_rate_transition_source_lr, 1e-5,
          "Expected source learning_rate in parent manifest for transition.");
ABSL_FLAG(std::string, learning_rate_transition_source_fingerprint, "",
          "Expected source config fingerprint for learning rate transition.");
ABSL_FLAG(bool, allow_target_kl_transition, false,
          "Allow intentional transition of target_kl from a verified parent checkpoint.");
ABSL_FLAG(double, target_kl_transition_source_target_kl, 0.01,
          "Expected source target_kl in parent manifest for transition.");
ABSL_FLAG(std::string, target_kl_transition_source_fingerprint, "",
          "Expected source config fingerprint for target_kl transition.");
ABSL_FLAG(bool, allow_single_learner_transition, false,
          "Allow intentional transition into single learner mode from a verified self-play parent checkpoint.");
ABSL_FLAG(std::string, single_learner_transition_source_fingerprint, "",
          "Expected source config fingerprint for single learner transition.");
ABSL_FLAG(int64_t, single_learner_transition_source_rollout_games, 256,
          "Expected source rollout_games when transitioning into single learner mode.");
ABSL_FLAG(int, seed, 1, "Base random seed.");
ABSL_FLAG(bool, pipeline, false,
          "Overlap next rollout collection with current PPO training. "
          "Beneficial with separate inference/training GPUs. On a single GPU, "
          "both workloads compete for compute and pipelining may be slower.");
ABSL_FLAG(bool, separate_actor_critic, false,
          "Arm D: Train fully separate actor and critic networks.");
ABSL_FLAG(bool, fresh_optimizer, false,
          "Start with a fresh AdamW optimizer state even when initializing from a checkpoint.");
ABSL_FLAG(std::string, critic_checkpoint, "",
          "Critic model checkpoint to load/save (defaults to model_checkpoint if empty).");
ABSL_FLAG(std::string, critic_optim_checkpoint, "",
          "Critic optimizer checkpoint to load/save.");
ABSL_FLAG(int, start_update_override, -1,
          "Explicit override for start_update counter (e.g. 201 for study updates).");
ABSL_FLAG(int, study_updates, -1,
          "Number of study updates to run from start_update (e.g. 400).");

ABSL_FLAG(std::string, cutoff_timestamp, "",
          "Timezone-aware ISO-8601 cutoff timestamp string (e.g. 2026-09-20T08:00:00+10:00).");
ABSL_FLAG(double, cutoff_epoch, 0.0,
          "Cutoff timestamp as unix epoch seconds.");
ABSL_FLAG(std::string, card_telemetry_path, "",
          "Path to JSON file where cumulative two-card tracking counters are saved.");


ABSL_FLAG(std::string, search_label_dir, "",
          "[LEGACY offline-distillation mode] Directory containing precomputed "
          "search teacher labels. Empty disables. Mutually exclusive with "
          "--online_search_collection (the Phase 18B online path).");
ABSL_FLAG(double, search_lambda, 0.5,
          "[LEGACY offline-distillation] Weight for the static search teacher KL "
          "distillation loss.");
ABSL_FLAG(int, search_minibatches_per_update, 2,
          "[LEGACY offline-distillation] Search distillation minibatches per PPO update.");
ABSL_FLAG(int, search_minibatch_size, 512,
          "[LEGACY offline-distillation] Size of each search distillation minibatch.");

// --- Phase 18B online auxiliary-search collection (combined optimization) ---
// Treatment-arm switch. When true, each update collects search examples online
// from the frozen pre-update snapshot and folds them into TrainPpoUpdate as an
// auxiliary CE+value loss. Mutually exclusive with --search_label_dir. Every
// other OnlineSearchConfig field keeps its library default (the frozen pilot
// values) and is NOT restated as a flag.
ABSL_FLAG(bool, online_search_collection, false,
          "Enable Phase 18B online auxiliary-search collection + combined optimization.");
ABSL_FLAG(int, auxiliary_games, 16,
          "Online collector auxiliary games per update (must be a multiple of 4).");
ABSL_FLAG(uint64_t, auxiliary_search_seed_domain, 0,
          "Isolated seed domain for online collection. REQUIRED nonzero when "
          "--online_search_collection is set (no silent default).");
ABSL_FLAG(double, collector_dirichlet_epsilon, 0.25,
          "Root Dirichlet noise weight for online collection (pilot 0.25; probe used 0).");
ABSL_FLAG(std::string, collector_acceptance_prior, "",
          "Prior the collector's covered-mass acceptance rule is measured "
          "against: 'raw_network_prior' (default when unset) or 'tree_prior' "
          "(pre-WO-20 post-noise behavior). Must be stated EXPLICITLY when "
          "resuming a checkpoint whose manifest predates the field, because its "
          "cumulative counters were accumulated under tree_prior.");
// --- Leader-selection search teacher (adopted 2026-08-16; fixed budget) ------
// Ordinary PPO already trains leader choices from game returns. This optional
// teacher adds search-generated labels; it is not required for leader learning.
ABSL_FLAG(bool, search_leader_draft, false,
          "Search the Leader choice for each auxiliary game's designated "
          "searched seat and emit it as a leader_selection training label. "
          "At most ONE Leader choice per auxiliary game; a forced "
          "one-legal-action pick is never searched and produces no label.");
ABSL_FLAG(int, leader_draft_simulations, 64,
          "Simulations at each non-forced Leader choice. FIXED at 64 -- the "
          "adopted permanent budget, not an experimental arm.");
ABSL_FLAG(bool, leader_mass_only_coverage, false,
          "Leader-only acceptance rule: keep the 0.50 covered-prior-mass "
          "threshold and the min-visit definition, but drop the generic "
          "three-covered-actions requirement AT LEADER NODES ONLY. Without it "
          "a concentrated 64-simulation Leader correction is discarded for "
          "covering too few distinct actions and the raw prior is taught "
          "instead. No other role is affected.");
ABSL_FLAG(double, search_loss_coef, 0.10,
          "Target auxiliary search-loss coefficient (warms 0 -> target).");
ABSL_FLAG(int, search_loss_warmup_update, 25,
          "Update at which the search-loss coefficient reaches its target.");
ABSL_FLAG(double, target_sharpen_exponent, 1.0,
          "CE-target sharpening exponent for online collection: the pruned root "
          "visit target is peaked as p_i = v_i^alpha / sum_j v_j^alpha before "
          "emission. 1.0 = inert (today's behavior).");
ABSL_FLAG(double, abort_grad_norm_ratio, 0.50,
          "Abort the run if the per-update aux/PPO grad-norm ratio exceeds this.");
ABSL_FLAG(double, swordmaster_grant_fraction, 0.0,
          "Endowment-curriculum: fraction of collector games granting the searched "
          "seat a free Swordmaster.");
ABSL_FLAG(int, swordmaster_grant_round, 2,
          "Round at which a selected game's Swordmaster grant fires.");

// --- Search-in-Training (TT-RL in training auxiliary supervision) ----------
ABSL_FLAG(bool, search_in_training, false,
          "Enable online compound search supervision during PPO training.");
ABSL_FLAG(std::string, search_teacher_checkpoint, "",
          "Path to frozen teacher model checkpoint for compound search supervision.");
ABSL_FLAG(int, search_budget_per_update, 32,
          "Number of learner decision roots to evaluate with search per PPO update.");
ABSL_FLAG(double, search_loss_lambda, 0.10,
          "Loss coefficient lambda for auxiliary search target cross-entropy.");
ABSL_FLAG(int, search_rollouts_per_action, 32,
          "Number of rollouts per candidate action in compound search.");
ABSL_FLAG(int, search_top_k, 3,
          "Number of top candidate actions evaluated by compound search.");
ABSL_FLAG(double, search_override_margin, 0.15,
          "Minimum utility margin required for search to override policy.");
ABSL_FLAG(double, search_blend_eta, 0.80,
          "Blend parameter eta for search target: (1-eta)*mu + eta*q_search.");
ABSL_FLAG(double, search_temperature, 0.50,
          "Softmax temperature for candidate utilities in search target.");
ABSL_FLAG(int, search_threads, 32,
          "Worker thread concurrency for search supervision.");
ABSL_FLAG(int, search_num_evaluators, 8,
          "Number of parallel BatchedEvaluator coordinators for frozen teacher.");
ABSL_FLAG(int, search_target_batch_size, 128,
          "GPU evaluator target batch size for search supervision.");
ABSL_FLAG(bool, search_full_turn_supervision, false,
          "When true, search takes over the entire compound turn from start to kActionEndTurn; "
          "when false, search supervises an isolated single decision.");
ABSL_FLAG(bool, search_high_stakes_only, true,
          "When true, filter learner decisions to high-stakes strategic states (Agent Placement, "
          "Combat Deployment, Reveal Purchase) matching tournament controller.");
ABSL_FLAG(bool, search_pairing, false,
          "Use paired common random number rollouts across search candidates.");
ABSL_FLAG(bool, search_truncation, false,
          "Truncate search rollouts at the searcher's first decision of the next round instead of rolling out to terminal.");
ABSL_FLAG(bool, search_teacher_amp, false,
          "When true, enables CUDA BF16 AMP and TF32 on search teacher evaluators only.");
ABSL_FLAG(std::string, search_target_mode, "mpo",
          "Search target mode: 'mpo' (default) or 'override_gated'.");
ABSL_FLAG(double, search_override_eta, 0.50,
          "Blend weight on onehot(best_candidate) in override_gated target mode.");

ABSL_FLAG(std::string, init_mode, "",
          "Initialization mode (required): random, checkpoint, bootstrap, validate_legacy.");
ABSL_FLAG(int, seed_scheme_version, 2, "Seed scheme version.");
ABSL_FLAG(int, target_end_update, 2450, "Absolute target update number to train until.");
// start_update is defined in dune_ppo_training_flags.cc
ABSL_FLAG(uint64_t, start_env_steps, 0, "Fallback start environment steps for bootstrap mode.");
ABSL_FLAG(uint64_t, start_episode_id, 0, "Fallback start episode ID for bootstrap mode.");
ABSL_FLAG(std::string, diagnostics_path, "",
          "Path to write structured diagnostics JSON/CSV. Extension determines format.");
ABSL_FLAG(std::string, phase_timing_path, "",
          "Direct path for the phase-timing JSONL sidecar. If empty, derived from "
          "--diagnostics_path or --run_prefix.");
ABSL_DECLARE_FLAG(bool, train_value_only);
// WO-PERF-TIMING: defined in dune_ppo_training_utils.cc beside the WO-PERF-1
// flags. Read here only to enforce the --pipeline incompatibility at startup.
ABSL_DECLARE_FLAG(std::string, phase_timing_mode);
ABSL_FLAG(bool, unfreeze_trunk, false, "Unfreeze the shared trunk (input + res blocks) during training.");
ABSL_FLAG(bool, sample_counterfactual_states, false, "Sample generic counterfactual successor states.");
ABSL_FLAG(int, counterfactual_samples_per_game, 4,
          "Maximum uniformly sampled alternate primary-action successors per game.");
ABSL_FLAG(int, counterfactual_replay_weight, 8,
          "Value-loss replay multiplicity for each sampled counterfactual successor.");
ABSL_DECLARE_FLAG(double, policy_kl_anchor_coeff);

ABSL_FLAG(bool, emit_canary_columns, false,
          "PWO-5 section 14.1a: emit the seventeen collapse-canary columns "
          "70-86 (diagnostics schema v4 -> v5, 69 -> 86 columns), measured at "
          "section 14.1b's FROZEN site -- the rollout behaviour-policy "
          "decision -- and nowhere else. Launching a PWO-5 arm with this false "
          "is section 17.5 item 14: without the columns the section 14.2 halt "
          "rail cannot fire, and an unmonitored 300-update run is exactly the "
          "failure mode the u175 collapse lineage represents.");
ABSL_FLAG(double, reverse_kl_coef, 0.0,
          "Arm B treatment: coefficient on reverse-KL penalty to rollout collection policy "
          "(0.05 in Arm B, 0.0 in Arm A). Detached, nonforced mean.");
ABSL_FLAG(std::string, save_initial_model_and_exit, "",
          "If non-empty, saves the newly constructed initial random model to this path and exits 0.");
ABSL_FLAG(bool, r755_shuffle_aux_examples, false,
          "If true under --r755_hybrid_mode, deterministically shuffle current_collect.aux_examples "
          "once per update on a dedicated seed stream before Gate-2 and TrainPpoUpdate.");
ABSL_FLAG(int, pause_for_gate_at_update, 0,
          "If > 0, synchronously block after flushing diagnostics and checkpoint at this update "
          "until --gate_decision_file contains CONTINUE or ABORT.");
ABSL_FLAG(std::string, gate_decision_file, "",
          "Path to gate decision file ('CONTINUE' or 'ABORT') read at --pause_for_gate_at_update.");
ABSL_FLAG(int, gate_timeout_seconds, 1800,
          "Maximum seconds to wait at --pause_for_gate_at_update before failing closed and exiting.");



namespace open_spiel::ppo_train_internal {
std::atomic<bool> g_stop_requested{false};
void SignalHandler(int signum) {
  g_stop_requested.store(true, std::memory_order_relaxed);
}
}  // namespace open_spiel::ppo_train_internal

int main(int argc, char** argv) {
  using namespace open_spiel::ppo_train_internal;
  PpoTrainRunContext ctx;
  if (auto rc = RunStartupPhase(ctx, argc, argv)) {
    return *rc;
  }
#ifdef OPEN_SPIEL_BUILD_WITH_LIBTORCH
  if (ctx.init_mode == "diagnostic" || ctx.init_mode == "checkpoint" ||
      ctx.init_mode == "study_resume") {
    RunResumePhase(ctx);
  } else {
    RunBootstrapPhase(ctx);
  }
  RunSetupPhase(ctx);
  if (auto rc = RunSetupPreloopPhase(ctx)) {
    return *rc;
  }
  for (int update = ctx.start_update; update <= ctx.target_end_update; ++update) {
    PpoTrainStepState step;
    step.update = update;
    RunLoopUpdatePhase(ctx, step);
    RunLoopAuxPhase(ctx, step);
    RunLoopPostPhase(ctx, step);
    if (step.should_break) {
      break;
    }
  }
  RunShutdownPhase(ctx);
#endif
  return 0;
}
