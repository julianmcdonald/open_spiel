// Native LibTorch PPO trainer for Dune Imperium.
//
// This executable is intentionally separate from dune_benchmark.cc. It follows
// the OpenSpiel PyTorch PPO structure: collect an on-policy rollout batch,
// freeze it, run shuffled PPO epochs/minibatches, then sync the inference model.

#include <algorithm>
#include <atomic>
#include <chrono>
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
#include <sstream>
#include <iomanip>

#include <random>
#include <set>
#include <shared_mutex>
#include <string>
#include <thread>
#include <unordered_set>
#include <vector>

#include "open_spiel/abseil-cpp/absl/flags/flag.h"
#include "open_spiel/abseil-cpp/absl/flags/parse.h"
#include "open_spiel/abseil-cpp/absl/strings/str_format.h"
#include "open_spiel/abseil-cpp/absl/strings/str_split.h"
#include "open_spiel/games/dune_imperium/dune_imperium.h"

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
#include "dune_vrpo.h"
#include "dune_vrpo_checkpoint.h"
#include "dune_vrpo_phase4e.h"
#include "dune_vrpo_ppo_pilot.h"
#include "dune_vrpo_ppo_continuation.h"
#include "dune_vrpo_schedule_screen.h"
#include "dune_pwo5_aux.h"
#include "dune_search_label_buffer.h"
#include "dune_search_pi.h"  // agent-turn search policy-iteration lane
#include "dune_search_pi_concurrent.h"
#include "dune_search_pi_replay.h"  // row shards + uniform replay-window sampler
#include "dune_search_routing.h"
#include "dune_evaluator.h"  // DuneNNEvaluator for online-collection snapshot inference
#include "open_spiel/utils/json.h"
#endif

ABSL_FLAG(std::string, game, "dune_imperium", "The OpenSpiel game to train.");
ABSL_FLAG(int, threads, 64, "Rollout worker threads.");
ABSL_FLAG(int, total_updates, 1000, "Number of PPO collect/update cycles.");
ABSL_FLAG(int, rollout_transitions, 32768,
          "Minimum on-policy transitions collected before each PPO update.");
ABSL_FLAG(int, rollout_games, 0,
          "Exact complete games collected per PPO update rollout batch. "
          "If > 0, overrides transition-threshold mode.");
ABSL_FLAG(int, ppo_minibatch_size, 2048, "PPO minibatch size.");
ABSL_FLAG(int, ppo_update_epochs, 4, "PPO epochs per rollout batch.");
ABSL_FLAG(double, learning_rate, 2.5e-4, "AdamW learning rate.");
ABSL_FLAG(bool, anneal_lr, true, "Linearly anneal learning rate over updates.");
ABSL_FLAG(double, gamma, 1.0, "Discount factor.");
ABSL_FLAG(double, gae_lambda, 1.0, "GAE lambda.");
ABSL_FLAG(bool, normalize_advantages, true,
          "Normalize advantages within each PPO minibatch.");
ABSL_FLAG(double, ppo_clip_epsilon, 0.2, "PPO surrogate/value clip epsilon.");
ABSL_FLAG(bool, ppo_clip_value_loss, true, "Use clipped value loss.");
ABSL_FLAG(double, entropy_coef, 0.01, "Entropy bonus coefficient.");
ABSL_FLAG(double, value_coef, 0.5, "Value loss coefficient.");
ABSL_FLAG(double, grad_clip_norm, 0.5, "Maximum gradient norm.");
ABSL_FLAG(double, target_kl, 0.0,
          "Stop PPO epochs early if approximate KL exceeds this value. <=0 disables.");
ABSL_FLAG(double, weight_decay, 0.0, "AdamW weight decay for torso/value params.");
ABSL_FLAG(double, policy_weight_decay, 0.0,
          "AdamW weight decay for policy-head params.");

ABSL_FLAG(int, eval_batch_size, 64, "Target inference batch size.");
ABSL_FLAG(int, eval_timeout_ms, 2, "Inference batch timeout in milliseconds.");
ABSL_FLAG(int, hidden_dim, 2048, "Network hidden dimension.");
ABSL_FLAG(int, num_blocks, 8, "Network residual block count.");
ABSL_FLAG(double, logit_cap, 10.0,
          "Smooth tanh cap on legal-centered policy logits. <=0 disables.");
ABSL_FLAG(bool, train_amp, true, "Use CUDA BF16 autocast for PPO updates.");
ABSL_FLAG(bool, rollout_amp, true,
          "Use CUDA BF16 autocast for BatchedEvaluator rollout inference. "
          "Default true preserves the historical rollout path.");
ABSL_FLAG(bool, allow_tf32, true,
          "Allow TF32 for CUDA CuBLAS/CuDNN. Default true preserves the "
          "historical runtime policy.");
ABSL_FLAG(bool, evaluator_device_synchronize, true,
          "Use whole-device CUDA synchronize after evaluator D2H copies.");
ABSL_FLAG(bool, deterministic, true, "Enable strict PyTorch/LibTorch deterministic algorithms.");
ABSL_FLAG(bool, deterministic_rollout_eval, false, "Use deterministic batch-1 rollout evaluation.");
ABSL_FLAG(bool, diagnostics_only, false, "Collect one rollout, write diagnostics, and exit without optimization.");
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
ABSL_FLAG(std::string, vrpo_capture_output, "",
          "Fresh atomic JSON output for the diagnostics-only VRPO episode capture.");
ABSL_FLAG(std::string, vrpo_capture_registration_id, "",
          "Required immutable registration identifier for VRPO capture.");
ABSL_FLAG(std::string, vrpo_capture_source_root, "",
          "Required source root for the fixed VRPO capture source list.");
ABSL_FLAG(std::string, vrpo_capture_source_sha256, "",
          "Required canonical VRPO capture source digest.");
ABSL_FLAG(double, vrpo_probability_tolerance,
          open_spiel::kVrpoRegisteredProbabilityTolerance,
          "Registered legal-probability validation tolerance for VRPO "
          "diagnostics; must remain exactly 1e-9.");
ABSL_FLAG(std::string, vrpo_q_preflight_output, "",
          "Fresh atomic JSON output for the no-training VRPO Q/reference preflight.");
ABSL_FLAG(std::string, vrpo_q_preflight_registration_id, "",
          "Required immutable registration identifier for the Q/reference preflight.");
ABSL_FLAG(uint64_t, vrpo_q_init_seed, 0,
          "Required deterministic VRPO Q initialization seed.");
ABSL_FLAG(int, vrpo_q_chunk_rows, 0,
          "Required Q forward chunk size in [1,256].");
ABSL_FLAG(double, vrpo_q_agreement_abs_tolerance, 1e-10,
          "Absolute scalar/tensor recurrence agreement tolerance.");
ABSL_FLAG(double, vrpo_q_agreement_rel_tolerance, 1e-10,
          "Relative scalar/tensor recurrence agreement tolerance.");
ABSL_FLAG(int64_t, vrpo_q_gpu_peak_increment_limit_bytes, 268435456,
          "Hard incremental GPU allocator peak ceiling for Q preflight.");
ABSL_FLAG(std::string, vrpo_bootstrap_root, "",
          "Fresh root for terminal VRPO four-arm expanded bootstrap.");
ABSL_FLAG(std::string, vrpo_bootstrap_registration_id, "",
          "Required immutable VRPO bootstrap registration identifier.");
ABSL_FLAG(std::string, vrpo_bootstrap_source_root, "",
          "Required source root for VRPO bootstrap code provenance.");
ABSL_FLAG(std::string, vrpo_bootstrap_source_sha256, "",
          "Required canonical VRPO bootstrap source digest.");
ABSL_FLAG(std::string, vrpo_bootstrap_source_manifest_sha256, "",
          "Required exact u15828 source manifest digest.");
ABSL_FLAG(int64_t, vrpo_bootstrap_source_manifest_size, 0,
          "Required exact u15828 source manifest size.");
ABSL_FLAG(std::string, vrpo_bootstrap_source_model_sha256, "",
          "Required exact u15828 actor model digest.");
ABSL_FLAG(int64_t, vrpo_bootstrap_source_model_size, 0,
          "Required exact u15828 actor model size.");
ABSL_FLAG(std::string, vrpo_bootstrap_binary_sha256, "",
          "Required exact executed trainer binary digest.");
ABSL_FLAG(int64_t, vrpo_bootstrap_binary_size, 0,
          "Required exact executed trainer binary size.");
ABSL_FLAG(std::string, vrpo_bootstrap_experiment_uuid, "",
          "Required registered four-arm experiment UUID.");
ABSL_FLAG(uint64_t, vrpo_bootstrap_q_seed, 0,
          "Required shared deterministic Q initialization seed.");
ABSL_FLAG(uint64_t, vrpo_bootstrap_base_seed, 0,
          "Required shared four-arm training base seed.");
ABSL_FLAG(uint64_t, vrpo_bootstrap_ppo_cap10_start, 0,
          "Shared paired episode start, repeated for PPO cap10.");
ABSL_FLAG(uint64_t, vrpo_bootstrap_ppo_cap10_end, 0,
          "Shared paired episode end inclusive, repeated for PPO cap10.");
ABSL_FLAG(uint64_t, vrpo_bootstrap_ppo_uncapped_start, 0,
          "Shared paired episode start, repeated for PPO uncapped.");
ABSL_FLAG(uint64_t, vrpo_bootstrap_ppo_uncapped_end, 0,
          "Shared paired episode end inclusive, repeated for PPO uncapped.");
ABSL_FLAG(uint64_t, vrpo_bootstrap_vrpo_cap10_start, 0,
          "Shared paired episode start, repeated for VRPO cap10.");
ABSL_FLAG(uint64_t, vrpo_bootstrap_vrpo_cap10_end, 0,
          "Shared paired episode end inclusive, repeated for VRPO cap10.");
ABSL_FLAG(uint64_t, vrpo_bootstrap_vrpo_uncapped_start, 0,
          "Shared paired episode start, repeated for VRPO uncapped.");
ABSL_FLAG(uint64_t, vrpo_bootstrap_vrpo_uncapped_end, 0,
          "Shared paired episode end inclusive, repeated for VRPO uncapped.");
ABSL_FLAG(std::string, vrpo_one_update_input_archive, "",
          "Strict phase-4 expanded arm archive consumed by exactly one VRPO update.");
ABSL_FLAG(std::string, vrpo_one_update_output_root, "",
          "Fresh root receiving the atomic phase-4e update archive and result.");
ABSL_FLAG(std::string, vrpo_one_update_registration_id, "",
          "Required immutable registration ID for the one-update run.");
ABSL_FLAG(std::string, vrpo_one_update_arm_id, "",
          "Required registered arm ID: PPO_CAP10, PPO_UNCAPPED, VRPO_CAP10, or VRPO_UNCAPPED.");
ABSL_FLAG(std::string, vrpo_one_update_source_root, "",
          "Required source root for the phase-4e fixed source list.");
ABSL_FLAG(std::string, vrpo_one_update_source_sha256, "",
          "Required SHA-256 for the phase-4e fixed source list.");
ABSL_FLAG(std::string, vrpo_one_update_binary_sha256, "",
          "Required SHA-256 of the executing one-update trainer binary.");
ABSL_FLAG(int64_t, vrpo_one_update_binary_size, 0,
          "Required byte size of the executing one-update trainer binary.");
ABSL_FLAG(std::string, vrpo_one_update_profile, "",
          "Required registered one-update profile: smoke16 or M16.");
ABSL_FLAG(std::string, vrpo_schedule_screen_input_archive, "",
          "Strict PPO_CAP10 phase-4c archive for the terminal schedule screen.");
ABSL_FLAG(std::string, vrpo_schedule_screen_output_root, "",
          "Fresh root for the terminal PPO schedule-health screen.");
ABSL_FLAG(std::string, vrpo_schedule_screen_registration_id, "",
          "Immutable schedule-health screen registration ID.");
ABSL_FLAG(std::string, vrpo_schedule_screen_profile, "",
          "Required schedule-health profile: ultralow_lr_v3.");
ABSL_FLAG(std::string, vrpo_schedule_screen_source_root, "",
          "Source root for the fixed schedule-screen source list.");
ABSL_FLAG(std::string, vrpo_schedule_screen_source_sha256, "",
          "Registered SHA-256 for the schedule-screen source list.");
ABSL_FLAG(std::string, vrpo_schedule_screen_binary_sha256, "",
          "Registered SHA-256 of the executing schedule-screen binary.");
ABSL_FLAG(int64_t, vrpo_schedule_screen_binary_size, 0,
          "Registered byte size of the executing schedule-screen binary.");
ABSL_FLAG(std::string, vrpo_ppo_pilot_input_archive, "",
          "Strict Sol-audited bootstrap-v2 PPO_CAP10 origin for the finite actor-only PPO pilot.");
ABSL_FLAG(std::string, vrpo_ppo_pilot_output_root, "",
          "Fresh root for exactly four sequential actor-only PPO updates.");
ABSL_FLAG(std::string, vrpo_ppo_pilot_registration_id, "",
          "Immutable finite PPO-pilot registration identifier.");
ABSL_FLAG(std::string, vrpo_ppo_pilot_profile, "",
          "Required finite pilot profile: ppo_u3_4x16_v1.");
ABSL_FLAG(std::string, vrpo_ppo_pilot_source_root, "",
          "Source root for the fixed PPO-pilot source list.");
ABSL_FLAG(std::string, vrpo_ppo_pilot_source_sha256, "",
          "Registered SHA-256 for the fixed PPO-pilot source list.");
ABSL_FLAG(std::string, vrpo_ppo_pilot_binary_sha256, "",
          "Registered SHA-256 of the executing PPO-pilot binary.");
ABSL_FLAG(int64_t, vrpo_ppo_pilot_binary_size, 0,
          "Registered byte size of the executing PPO-pilot binary.");
ABSL_FLAG(std::string, vrpo_ppo_pilot_origin_archive_sha256, "",
          "Registered combined SHA-256 of the exact bootstrap-v2 PPO_CAP10 origin.");
ABSL_FLAG(std::string, vrpo_ppo_pilot_evidence_root, "",
          "Root containing the exact immutable v3 schedule/screen/confirm evidence chain.");
ABSL_FLAG(std::string, vrpo_ppo_pilot_v3_schedule_registration_sha256, "", "Exact v3 schedule registration SHA-256.");
ABSL_FLAG(std::string, vrpo_ppo_pilot_v3_schedule_result_sha256, "", "Exact v3 schedule result SHA-256.");
ABSL_FLAG(std::string, vrpo_ppo_pilot_v3_schedule_validation_sha256, "", "Exact v3 schedule validation SHA-256.");
ABSL_FLAG(std::string, vrpo_ppo_pilot_v3_schedule_corpus_manifest_sha256, "", "Exact v3 corpus manifest SHA-256.");
ABSL_FLAG(std::string, vrpo_ppo_pilot_v3_schedule_corpus_sha256, "", "Exact v3 actor corpus SHA-256.");
ABSL_FLAG(std::string, vrpo_ppo_pilot_selected_screen_manifest_sha256, "", "Exact selected two-cell screen manifest SHA-256.");
ABSL_FLAG(std::string, vrpo_ppo_pilot_selected_screen_result_sha256, "", "Exact selected two-cell screen result SHA-256.");
ABSL_FLAG(std::string, vrpo_ppo_pilot_selected_screen_validation_sha256, "", "Exact selected two-cell screen validation SHA-256.");
ABSL_FLAG(std::string, vrpo_ppo_pilot_confirm_manifest_sha256, "", "Exact U3 256-game confirmation manifest SHA-256.");
ABSL_FLAG(std::string, vrpo_ppo_pilot_confirm_result_sha256, "", "Exact U3 256-game confirmation result SHA-256.");
ABSL_FLAG(std::string, vrpo_ppo_pilot_confirm_validation_sha256, "", "Exact U3 256-game confirmation validation SHA-256.");
ABSL_FLAG(std::string, vrpo_ppo_continuation_output_root, "",
          "Fresh root for the bounded actor-only PPO updates 5 through 8 continuation.");
ABSL_FLAG(std::string, vrpo_ppo_continuation_registration_id, "",
          "Immutable updates 5 through 8 continuation registration ID.");
ABSL_FLAG(std::string, vrpo_ppo_continuation_profile, "",
          "Required bounded continuation profile.");
ABSL_FLAG(std::string, vrpo_ppo_continuation_evidence_root, "",
          "Root containing the exact pilot/U4/raw128/confirm256 prior chain.");
ABSL_FLAG(std::string, vrpo_ppo_continuation_source_root, "",
          "Source root for the fixed PPO-continuation source list.");
ABSL_FLAG(std::string, vrpo_ppo_continuation_source_sha256, "",
          "Registered SHA-256 for the fixed PPO-continuation source list.");
ABSL_FLAG(std::string, vrpo_ppo_continuation_binary_sha256, "",
          "Registered SHA-256 of the executing continuation binary.");
ABSL_FLAG(int64_t, vrpo_ppo_continuation_binary_size, 0,
          "Registered byte size of the executing continuation binary.");

ABSL_FLAG(double, shaped_reward_weight, 0.0,
          "Weight for intermediate VP shaped rewards.");
ABSL_FLAG(double, tleilaxu_breadcrumb_weight, 0.0,
          "Weight for Tleilaxu levels 5/6 breadcrumbs.");
ABSL_FLAG(double, tleilaxu_level7_breadcrumb_weight, 0.0,
          "Weight for Tleilaxu level 7 breadcrumb.");
// Sign convention: the value is SUBTRACTED from reward — pass a POSITIVE value to penalize. (The phase-18 pilots passed -0.02, which was a +0.02 bonus.)
// A NEGATIVE value is now rejected fatally in main() (PWO-5 gate 2 item (a)).
ABSL_FLAG(double, specimen_exchange_penalty, 0.0,
          "Magnitude of the negative shaping SUBTRACTED from a transition that "
          "takes a ConvertSpecimenToTroop action (IDs 741-752; 740 is an unused "
          "base constant and is never legal). MUST BE >= 0: the value is "
          "subtracted, so a POSITIVE value penalizes and a NEGATIVE value would "
          "be a BONUS on the very behaviour this term exists to suppress -- a "
          "negative value is rejected fatally at startup. Training-only "
          "anti-breadcrumb (never eval). Apply the SAME value to BOTH pilot and "
          "control arms so the search-distillation contrast stays a pure "
          "experiment. Typical 0.02 (terminal win utility is 2.25). Requires "
          "--allow_shaping.");
ABSL_FLAG(bool, allow_shaping, false,
          "Allow experimental reward shaping flags to be non-zero.");
ABSL_FLAG(double, reward_scale, 4.0,
          "Divide shaped plus terminal rewards by this value.");

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
ABSL_FLAG(int, seed, 1, "Base random seed.");
ABSL_FLAG(bool, pipeline, false,
          "Overlap next rollout collection with current PPO training. "
          "Beneficial with separate inference/training GPUs. On a single GPU, "
          "both workloads compete for compute and pipelining may be slower.");

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
// Leader picks were never searched, so the policy head learned Leader selection
// from nothing but its own prior. These make the Leader decision a searched,
// LABELLED training target. All three default OFF/inert: a run that wants
// Leader teaching states so explicitly, and they are pinned into the config
// fingerprint so a resume cannot silently change the teacher.
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

// --- Agent-turn search policy iteration (search-PI) -----------------------
//
// A RUN-SCOPED mode: a run is either ordinary PPO or search-PI, never both and
// never interleaved per update. Search generates the behaviour trajectory, the
// search visit policy is the first-class policy target, terminal results from
// that trajectory train the value head, and NO PPO objective touches these
// rows. Mutually exclusive with --online_search_collection, --search_label_dir
// and --pipeline; enforced at startup, not at first use.
//
// This is NOT the closed PF-C mode. It shares no configuration with it: it
// builds its own DuneSearchConfig, and in particular it does not import the
// production Leader-teacher pins.
ABSL_FLAG(bool, search_pi_mode, false,
          "Enable the agent-turn search policy-iteration lane (run-scoped; "
          "mutually exclusive with PPO training, --online_search_collection, "
          "--search_label_dir and --pipeline).");
ABSL_FLAG(int, search_pi_generations, 1,
          "Number of frozen-collector -> learn -> checkpoint generations.");
ABSL_FLAG(int, search_pi_games_per_generation, 16,
          "Outcome-blind prefix of collection episode IDs exposed to the "
          "learner. Historical mode also uses this as collection size.");
ABSL_FLAG(int, search_pi_collection_games_per_generation, 0,
          "Complete games collected per generation. 0 preserves the historical "
          "value of --search_pi_games_per_generation; post-Gate-8 registers 512.");
ABSL_FLAG(int, search_pi_concurrent_workers, 1,
          "Exact concurrent game workers G; never silently capped.");
ABSL_FLAG(int, search_pi_batch_target, 0,
          "Shared collector batch target. 0 is historical serial/reference mode; "
          "post-Gate-8 registers 64.");
ABSL_FLAG(int, search_pi_batcher_timeout_ms, 2,
          "Shared collector batcher flush timeout in milliseconds.");
ABSL_FLAG(int, search_pi_chunk_games, 4,
          "Seat-balanced games per concurrent work unit.");
ABSL_FLAG(int, search_pi_warmup_games, 0,
          "Discarded throwaway collector warm-up games; registered value is 0.");
ABSL_FLAG(int, search_pi_purchase_combat_budget, 0,
          "0 selects the narrow teacher; 64 selects the wide teacher.");
ABSL_FLAG(double, search_pi_relative_time_budget_ms, 10000.0,
          "Per-root runaway watchdog. Post-Gate-8 requires 60000 ms.");
ABSL_FLAG(std::string, search_pi_collector_model_checkpoint, "",
          "Immutable model archive loaded into the collector, distinct from the "
          "mutable student model.");
ABSL_FLAG(std::string, search_pi_collector_model_sha256, "",
          "Required SHA-256 of the immutable collector archive.");
ABSL_FLAG(std::string, search_pi_generation_manifest_dir, "",
          "Fresh directory for immutable per-generation collection manifests.");
ABSL_FLAG(int, search_pi_max_filler_timeout_episodes, 0,
          "Named watchdog-timeout filler episodes allowed per generation. The "
          "registered 512-game lane fixes 2; every other failure still STOPs.");
ABSL_FLAG(int64_t, search_pi_first_episode_id, 0,
          "First collection episode ID for a fresh Search-PI lineage. Resume "
          "uses the committed search_pi cursor and requires this value to match.");
ABSL_FLAG(int, search_pi_primary_simulations, 200,
          "NEW simulations at each kAgentPrimary root.");
ABSL_FLAG(int, search_pi_continuation_simulations, 64,
          "NEW simulations at each kAgentContinuation root. Independent of the "
          "primary budget: the primary cannot consume it.");
ABSL_FLAG(double, search_pi_puct_c, 0.30, "PUCT exploration constant.");
ABSL_FLAG(double, search_pi_opponent_temperature, 1.0,
          "Policy opponent-model temperature inside the search.");
ABSL_FLAG(double, search_pi_root_prior_temperature, 1.0,
          "Root prior temperature.");
ABSL_FLAG(double, search_pi_dirichlet_epsilon, 0.0,
          "Root Dirichlet noise weight. 0 disables the exploration package.");
ABSL_FLAG(double, search_pi_dirichlet_alpha_total, 10.83,
          "Per-root alpha = total / N_legal. Inert while epsilon is 0.");
ABSL_FLAG(double, search_pi_forced_playouts_k, 0.0,
          "KataGo forced-playout coefficient. Must stay 0 while epsilon is 0: "
          "with no noise there are no forced visits to prune, and an armed "
          "pruner would subtract organic visits from the target.");
ABSL_FLAG(bool, search_pi_root_noise_fpu_zero, false,
          "FPU = 0 at the noised root. Inert while epsilon is 0.");
ABSL_FLAG(double, search_pi_target_sharpen_exponent, 1.0,
          "Target sharpening exponent. 1.0 = inert.");
ABSL_FLAG(std::string, search_pi_continuation_target, "total_visits",
          "How a continuation's target normalizes root visits after a re-root: "
          "total_visits (inherited + new; the tree-reuse convention) or "
          "new_visits_only.");
ABSL_FLAG(double, search_pi_behavior_temperature, 0.0,
          "Temperature for the EXECUTED action, drawn from the unpruned visit "
          "counts. 0 = argmax, so the trajectory reflects the stronger "
          "controller.");
ABSL_FLAG(double, search_pi_non_search_temperature, 1.0,
          "Raw-policy temperature for the non-searched opponent seats.");
ABSL_FLAG(double, search_pi_unsearched_role_temperature, 1.0,
          "Raw-policy temperature for the SEARCHED seat's own non-searched "
          "decisions (leader, purchase, combat-intrigue, other-optional).");
ABSL_FLAG(uint64_t, search_pi_seed_domain, 0,
          "Domain-separated seed stream for the lane (required nonzero).");
ABSL_FLAG(double, search_pi_learning_rate, 1.0e-4,
          "Learning rate for the DEDICATED search-PI optimizer. Independent of "
          "--learning_rate, which this mode never uses.");
ABSL_FLAG(int, search_pi_minibatch_size, 256, "Search-PI learner minibatch size.");
ABSL_FLAG(int, search_pi_epochs, 1, "Search-PI learner epochs per generation.");
ABSL_FLAG(double, search_pi_value_coef, 1.0,
          "Weight on the value MSE in CE + value_coef * MSE. Stated explicitly "
          "rather than inherited from --value_coef, which means something "
          "different inside a PPO total loss.");
ABSL_FLAG(double, search_pi_policy_coef, 1.0,
          "Weight on the search CE in policy_coef * CE + value_coef * MSE. At "
          "exactly 0 the policy backward is SKIPPED rather than scaled to zero, "
          "so the policy objective contributes no gradient at all -- which is "
          "what lets the value channel be ablated alone. At exactly 1 no "
          "multiply is inserted, so the objective is bit-identical to the "
          "runs that predate this flag.");

// --- Search-PI registration profiles --------------------------------------
//
// The lane below hard-pins ONE registered experiment ("post-Gate-8 3b"): the
// collector SHA-256, the 512/64/G=128 collection geometry, the episode base
// 600000, the 512 stride and the 1..4 resume window. Those pins are what stop
// a launcher typo from producing a substitute experiment that merely resembles
// the registration, and they must keep firing byte-for-byte for the arm that is
// live right now.
//
// A profile does not LOOSEN a pin; it selects which registered constants apply:
//   post_gate8_3b  today's behaviour, and the only profile that requires the
//                  fresh student to BE the frozen collector snapshot.
//   rung4a         hybrid two-model collection -- the pinned teacher still
//                  drives search, a second frozen archive supplies the policy
//                  prior, and the student starts from a trained checkpoint.
//   rung4_replay   row-shard capture, and the train-only replay arm that never
//                  collects at all.
// The teacher and the collection geometry are DELIBERATELY identical across all
// three: rung 4 varies the learner's diet, not the teacher, so the collector-SHA
// pin, the geometry pin and the one-generation-per-invocation rule stay armed in
// every profile.
ABSL_FLAG(std::string, search_pi_registration_profile, "post_gate8_3b",
          "Registered Search-PI experiment this invocation belongs to: "
          "post_gate8_3b (default; unchanged behaviour), rung4a or "
          "rung4_replay. Any other value is fatal at startup.");
ABSL_FLAG(int64_t, search_pi_registered_first_episode_id, 600000,
          "Episode-ID base of the registered lineage: generation N collects "
          "from base + (N-1) * collection games. Pinned to the default value "
          "under the post_gate8_3b profile.");
ABSL_FLAG(int, search_pi_max_generation, 5,
          "Exclusive upper bound on a resumable generation, so a resume must "
          "carry a completed boundary in 1..max-1. Pinned to the default value "
          "under the post_gate8_3b profile.");
ABSL_FLAG(std::string, search_pi_expected_initial_student_sha256, "",
          "Canonical module digest (CanonicalSearchPiModuleDigest, not a file "
          "hash) that a FRESH rung4 student must have. Empty disables the "
          "check. A rung4 arm starts from a TRAINED student, so 3b's rule -- "
          "fresh student == frozen collector snapshot -- cannot apply to it; "
          "that rule is unchanged for 3b itself.");
ABSL_FLAG(std::string, search_pi_policy_prior_model_checkpoint, "",
          "rung4a only: a SECOND immutable archive supplying the search's "
          "policy prior while the frozen collector keeps supplying value and "
          "opponent inference. Empty selects the single-model collection every "
          "other profile uses.");
ABSL_FLAG(std::string, search_pi_policy_prior_model_sha256, "",
          "Required SHA-256 of the policy-prior archive. Caller-supplied and "
          "checked against the file only -- unlike the collector hash, which "
          "must ALSO equal the one registered constant.");
ABSL_FLAG(std::string, search_pi_row_shard_out, "",
          "rung4_replay only: path the outcome-blind training prefix is "
          "written to once collection validates and BEFORE the learner runs, "
          "so a 512-game cohort survives a learner failure. Mutually exclusive "
          "with training from shards.");
ABSL_FLAG(std::string, search_pi_train_from_shards, "",
          "rung4_replay only: comma-separated ordered shard paths. Non-empty "
          "SKIPS collection entirely and trains on a uniform sample of the "
          "flattened window instead.");
ABSL_FLAG(int, search_pi_row_sample_count, 0,
          "Rows drawn from the replay window. Required positive when training "
          "from shards, and fatal if it exceeds the rows the window holds. The "
          "equal row count is what keeps the replay arm from also being a "
          "more-optimisation arm.");
ABSL_FLAG(uint64_t, search_pi_row_sample_seed, 0,
          "Seed for the uniform replay draw. Recorded in the generation marker "
          "so a sample is reproducible from artifacts alone.");

// --- Origin reset: inherit the student, re-base the lineage -----------------
//
// A rung-4 successor boots from the 3b generation-5 student and inherits its
// dedicated Adam moments, which is only possible from a checkpoint that carries
// a search_pi block (the fresh-optimizer refusal below this file's dedicated
// optimizer). That block also carries 3b's generation counter, episode cursor
// and cumulative totals, which the successor must NOT continue. These flags
// select ApplySearchPiOriginReset (dune_search_pi.h) to split the two.
//
// Default false, so every existing invocation -- including the live 3b run --
// reaches identical code. The reset is fatal under post_gate8_3b: re-basing 3b
// onto itself has no meaning, and a profile check is what stops the flag from
// being available to the arm it would damage.
ABSL_FLAG(bool, search_pi_origin_reset, false,
          "Re-base an inherited search-PI lineage onto a fresh experiment "
          "origin: keep the model and the optimizer moments, restart the "
          "generation counter, the episode cursor, the cumulative totals and "
          "the hash chains. Legal only under a rung4 profile.");
ABSL_FLAG(int, search_pi_origin_generation, 5,
          "Generation the inherited checkpoint must carry for the origin reset "
          "to fire. One third of the triple that identifies the origin.");
ABSL_FLAG(int64_t, search_pi_origin_next_episode_id, 602560,
          "Episode cursor the inherited checkpoint must carry for the origin "
          "reset to fire. Second third of the origin triple.");
ABSL_FLAG(std::string, search_pi_origin_config_fingerprint, "",
          "Search-PI config fingerprint the inherited checkpoint must carry "
          "for the origin reset to fire. Required non-empty when the reset is "
          "requested: it is the final third of the origin triple.");

ABSL_FLAG(std::string, init_mode, "",
          "Initialization mode (required): random, checkpoint, bootstrap, validate_legacy.");
ABSL_FLAG(uint64_t, shaping_start_env_steps, 206830543,
          "Environment steps at which shaped reward decay starts.");
ABSL_FLAG(uint64_t, shaping_decay_env_steps, 0,
          "Environment steps duration for shaped reward decay (0 means no decay).");
ABSL_FLAG(int, seed_scheme_version, 2, "Seed scheme version.");
ABSL_FLAG(int, target_end_update, 2450, "Absolute target update number to train until.");
ABSL_FLAG(int, start_update, 1, "Fallback start update for bootstrap mode.");
ABSL_FLAG(uint64_t, start_env_steps, 0, "Fallback start environment steps for bootstrap mode.");
ABSL_FLAG(uint64_t, start_episode_id, 0, "Fallback start episode ID for bootstrap mode.");
ABSL_FLAG(std::string, diagnostics_path, "",
          "Path to write structured diagnostics JSON/CSV. Extension determines format.");
ABSL_DECLARE_FLAG(bool, train_value_only);
// WO-PERF-TIMING: defined in dune_ppo_training_utils.cc beside the WO-PERF-1
// flags. Read here only to enforce the --pipeline incompatibility at startup.
ABSL_DECLARE_FLAG(std::string, phase_timing_mode);
ABSL_FLAG(bool, unfreeze_trunk, false, "Unfreeze the shared trunk (input + res blocks) during training.");
ABSL_FLAG(bool, nonlinear_value_head, false, "Use a nonlinear value head.");
ABSL_FLAG(bool, sample_counterfactual_states, false, "Sample generic counterfactual successor states.");
ABSL_FLAG(int, counterfactual_samples_per_game, 4,
          "Maximum uniformly sampled alternate primary-action successors per game.");
ABSL_FLAG(int, counterfactual_replay_weight, 8,
          "Value-loss replay multiplicity for each sampled counterfactual successor.");
ABSL_DECLARE_FLAG(double, policy_kl_anchor_coeff);

// ===========================================================================
// PWO-5 gate 3 — Appendix A.1 of docs/PWO5_PILOT_REGISTRATION.md
// ===========================================================================
//
// These THIRTEEN flags are FROZEN by that appendix, which registers them
// BEFORE they exist precisely so no post-hoc flag can enter the experiment
// claiming it was always intended. **A flag below whose name, type or compiled
// default differs from the appendix is a STOP, as is any ADDITIONAL flag gate 3
// introduces that the appendix does not list.** So: exactly these thirteen,
// exactly these names, exactly these types, exactly these defaults.
//
// EVERY COMPILED DEFAULT IS INERT (0.0 / "" / 0 / false). That is a registered
// property, not a convenience: a build that has gained these flags but is
// launched without them trains exactly as the pre-gate-3 binary did, which is
// what keeps the section 7.3 legacy-inference parity gate and the section 6.4
// no-aux-leakage check meaningful AFTER gate 3 lands.
//
// The three coefficients are the H-vs-T TREATMENT (section 9.1) and are
// therefore expected to differ across arms; everything else here is MATCHED and
// must be byte-identical on all six.

ABSL_FLAG(double, final_vp_head_coef, 0.0,
          "PWO-5 section 8.1: coefficient on the final_vp_head Huber loss. "
          "0.05 in H arms, 0.0 in T and P. Exactly 0.0 short-circuits the head "
          "entirely -- the loss term is NOT CONSTRUCTED, so no graph node "
          "exists and the head parameters receive no gradient at all (section "
          "7.4 rejects multiply-by-zero as the mechanism).");
ABSL_FLAG(double, terminal_round_head_coef, 0.0,
          "PWO-5 section 8.1: coefficient on the terminal_round_head "
          "cross-entropy over the three classes <=8 / 9 / 10. 0.05 in H arms, "
          "0.0 in T and P.");
ABSL_FLAG(double, next_own_action_head_coef, 0.0,
          "PWO-5 section 8.1: coefficient on the next_own_action_head "
          "cross-entropy over the full 2391-action vocabulary, unmasked. 0.15 "
          "in H arms, 0.0 in T and P.");
ABSL_FLAG(std::string, aux_target_path, "",
          "PWO-5 section 6.1 item 1: the parallel, separately hashed auxiliary "
          "target artifact keyed by (game_index, decision_index). MATCHED on "
          "all six arms and digest-verified on all six; LOADED AND USED only "
          "when at least one head coefficient is nonzero. The verification is "
          "unconditional, the use is not -- so a corrupt artifact is caught on "
          "every arm at launch, not only on the two that would have read it.");
ABSL_FLAG(std::string, aux_target_sha256, "",
          "PWO-5 section 5.2: sha256 of --aux_target_path, asserted on all six "
          "arms. A mismatch is fatal (section 17.5 item 3): a materialized "
          "artifact that is not hashed is an unpinned replay input.");
ABSL_FLAG(int, aux_games_per_update, 0,
          "PWO-5 section 8.6: G, the number of games drawn uniformly WITHOUT "
          "replacement ONCE per update, then partitioned. Registered 64.");
ABSL_FLAG(int, aux_rows_per_game, 0,
          "PWO-5 section 8.6: R, rows drawn uniformly without replacement from "
          "each drawn game. Registered 16.");
ABSL_FLAG(int, aux_batches_per_update, 0,
          "PWO-5 section 8.6: the partition count. Registered 2, giving "
          "2 x 32 games x 16 rows = 1,024 DISTINCT rows per update.");
ABSL_FLAG(std::string, aux_heldout_games_path, "",
          "PWO-5 section 8.5: the 60-game whole-trajectory held-out membership "
          "list. Held-out games are excluded from ALL offline losses, "
          "INCLUDING distillation, because the trunk is shared.");
ABSL_FLAG(std::string, aux_heldout_sha256, "",
          "PWO-5 section 8.5: sha256 of the canonical serialization of the 60 "
          "held-out game indices, asserted on resume and on the u600 extension "
          "so the split cannot silently differ across arms or replicates.");
ABSL_FLAG(double, huber_delta_final_vp, 0.10,
          "PWO-5 section 8.2: Huber delta for final_vp_head, on the /20 scale. "
          "The target spans [0.25, 0.80], so 0.10 puts the quadratic/linear "
          "knee at ~18% of the target range. The framework default 1.0 would "
          "be pure squared error over the whole reachable range and would make "
          "'Huber' a misnomer.");
ABSL_FLAG(uint64_t, head_init_constant, 20260800,
          "PWO-5 section 7.2: kHeadInitConstant. NOT a run seed -- it is a "
          "fixed registered constant so that the initial head parameters are "
          "byte-identical across ALL SIX arms. Deriving from the triplet seed "
          "would give T1/P1/H1 one set and T2/P2/H2 another. Outside the "
          "reserved final-gate base-seed range by section 10.2's arithmetic.");
ABSL_FLAG(bool, emit_canary_columns, false,
          "PWO-5 section 14.1a: emit the seventeen collapse-canary columns "
          "70-86 (diagnostics schema v4 -> v5, 69 -> 86 columns), measured at "
          "section 14.1b's FROZEN site -- the rollout behaviour-policy "
          "decision -- and nowhere else. Launching a PWO-5 arm with this false "
          "is section 17.5 item 14: without the columns the section 14.2 halt "
          "rail cannot fire, and an unmonitored 300-update run is exactly the "
          "failure mode the u175 collapse lineage represents.");

namespace open_spiel {
namespace {

std::set<std::vector<open_spiel::Action>> prohibited_histories;

#ifdef OPEN_SPIEL_BUILD_WITH_LIBTORCH

class PpoRolloutBuffer {
 public:
  size_t PushTrajectory(std::vector<PpoTransition>&& trajectory) {
    std::lock_guard<std::mutex> lock(mu_);
    size_t size = trajectory.size();
    trajectories_.push_back(std::move(trajectory));
    num_transitions_ += size;
    return num_transitions_;
  }

  size_t Size() const {
    std::lock_guard<std::mutex> lock(mu_);
    return num_transitions_;
  }

  std::vector<PpoTransition> TakeAll(bool* out_episode_ids_unique = nullptr) {
    std::lock_guard<std::mutex> lock(mu_);
    if (out_episode_ids_unique) {
      *out_episode_ids_unique = true;
      std::unordered_set<uint64_t> seen_ids;
      for (const auto& traj : trajectories_) {
        if (!traj.empty()) {
          uint64_t ep_id = traj[0].episode_id;
          if (seen_ids.count(ep_id)) {
            *out_episode_ids_unique = false;
          }
          seen_ids.insert(ep_id);
        }
      }
    }
    std::sort(trajectories_.begin(), trajectories_.end(),
              [](const std::vector<PpoTransition>& a,
                 const std::vector<PpoTransition>& b) {
                uint64_t id_a = a.empty() ? 0 : a[0].episode_id;
                uint64_t id_b = b.empty() ? 0 : b[0].episode_id;
                return id_a < id_b;
              });
    std::vector<PpoTransition> out;
    out.reserve(num_transitions_);
    for (auto& traj : trajectories_) {
      for (auto& transition : traj) {
        out.push_back(std::move(transition));
      }
    }
    trajectories_.clear();
    num_transitions_ = 0;
    return out;
  }

 private:
  mutable std::mutex mu_;
  std::vector<std::vector<PpoTransition>> trajectories_;
  size_t num_transitions_ = 0;
};

// SearchLabel, SearchLabelRole, SearchLabelFileEntry and SearchLabelBuffer now
// live in dune_search_label_buffer.h so that the trainer and the ruling-5 role
// tests link ONE definition rather than two that can drift.
using open_spiel::SearchLabel;
using open_spiel::SearchLabelBuffer;
using open_spiel::SearchLabelFileEntry;
using open_spiel::SearchLabelRole;

struct WorkerStats {
  uint64_t games = 0;
  uint64_t moves = 0;

  // Natural SM Acquisitions
  uint64_t sm_acquisitions_by_seat[4] = {0};
  uint64_t sm_acquisitions_by_leader[14] = {0};
  uint64_t sm_acquisitions_by_round[11] = {0};

  double conflict_vp_generated = 0.0;
  double conflict_vp_attributed = 0.0;
  double conflict_vp_unattributed = 0.0;

  // Raw VPs for conservation checks
  double raw_conflict_vp = 0.0;
  double raw_noncombat_vp = 0.0;
  double raw_total_vp = 0.0;

  // Phase A canary: per-decision PRE-cap legal-centered max|z| for every
  // nontrivial rollout decision this worker made, overall and split by decision
  // role. Purely observational -- collected before CenterAndCapLegalLogits
  // mutates the logits, and never read back into the acting path.
  std::vector<float> precap_absz_all;
  std::vector<float> precap_absz_primary;
  std::vector<float> precap_absz_continuation;
  std::vector<float> precap_absz_purchase;

  // --- PWO-5 section 14.1b: the three collapse canaries. ------------------
  //
  // Measured at the ROLLOUT BEHAVIOUR-POLICY DECISION and nowhere else. The
  // registration considered three candidate sites and rejected the other two:
  // the executed-PPO-minibatch site (existing col 36 `entropy`) is
  // presentation-weighted over an EVOLVING model and is truncated by
  // --target_kl, so an arm that early-stops more often would show a different
  // entropy for reasons unrelated to collapse -- "a safety rail must not be a
  // function of its own optimizer's stopping behaviour"; and a pre-optimizer
  // snapshot would cost a second forward over 32,768 transitions and still
  // lack the role and legal-action data.
  //
  // Population: unique rollout decisions of the acting seat with n_legal >= 2,
  // one measurement each. No re-presentation, no epoch weighting, no dependence
  // on KL early stop.
  //
  // These are NOT column 36 and may never be compared to or substituted for it.
  //
  // Role index is the integer value of DuneDecisionRole (7 members). The
  // kForcedOrBookkeeping slot is identically empty at this site, by the
  // section 14.1c proof: the classifier returns it iff the searched player is
  // the chance player OR n_legal <= 1, chance nodes `continue` before ever
  // reaching the classifier here, and the population already requires
  // n_legal >= 2. It is accumulated anyway so the partition identity is
  // checked rather than assumed.
  double canary_ne_sum = 0.0;             // sum of H_row / log(n_legal)
  int64_t canary_ne_n = 0;                // qualifying decisions
  double canary_ne_sum_role[7] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  int64_t canary_ne_n_role[7] = {0, 0, 0, 0, 0, 0, 0};
  double canary_maxprob_sum = 0.0;        // denominator is canary_ne_n
  int64_t canary_sat_ge_cap = 0;          // legal PRE-cap |z| >= logit_cap
  int64_t canary_sat_logits = 0;          // legal logits examined (LOGIT unit)

  // V3 numerical-parity capture contract. Populated only when the diagnostic
  // output flag is set; normal rollout leaves every counter/vector untouched.
  int64_t parity_decisions = 0;
  int64_t parity_full_width_ok = 0;
  int64_t parity_full_finite_ok = 0;
  int64_t parity_legal_ids_unique_in_range = 0;
  int64_t parity_chosen_once = 0;
  std::vector<std::string> parity_capture_errors;
};


// Copies BY NAME, not by position.
//
// PWO-5 makes the two models' module sets deliberately asymmetric: the training
// model carries the three auxiliary heads and the inference model does not
// (section 8.6 -- the heads must never be computed in a rollout/inference
// forward, and the cleanest way to guarantee that is for them not to exist
// there). The previous positional loop indexed `target_params[i]` for every `i`
// in the SOURCE's range, so a source with 6 extra tensors would have walked off
// the end of the target.
//
// Name-based copy is also strictly safer than positional in general: it cannot
// silently pair `policy_head.weight` with `value_head.weight` if a future
// registration order changes. When the module sets DO match it copies exactly
// the same tensors as before, so this is behaviour-preserving for every
// pre-PWO-5 caller.
void CopyModelWeights(std::shared_ptr<SharedDunePolicyValueNetImpl> source,
                      std::shared_ptr<SharedDunePolicyValueNetImpl> target) {
  torch::NoGradGuard no_grad;
  auto source_params = source->named_parameters();
  auto source_buffers = source->named_buffers();
  for (auto& item : target->named_parameters()) {
    auto* src = source_params.find(item.key());
    if (src == nullptr) {
      SpielFatalError(
          "CopyModelWeights: target parameter '" + item.key() +
          "' has no counterpart in the source model. The inference model may "
          "be a SUBSET of the training model (PWO-5 auxiliary heads), never a "
          "superset.");
    }
    item.value().copy_(src->to(item.value().device()));
  }
  for (auto& item : target->named_buffers()) {
    auto* src = source_buffers.find(item.key());
    if (src == nullptr) {
      SpielFatalError("CopyModelWeights: target buffer '" + item.key() +
                      "' has no counterpart in the source model.");
    }
    item.value().copy_(src->to(item.value().device()));
  }
}

void SyncModels(std::shared_ptr<SharedDunePolicyValueNetImpl> training_model,
                std::shared_ptr<SharedDunePolicyValueNetImpl> inference_model,
                std::shared_mutex* sync_mutex) {
  std::unique_lock<std::shared_mutex> lock(*sync_mutex);
  CopyModelWeights(training_model, inference_model);
}

std::string HashNonValueParameters(const std::shared_ptr<SharedDunePolicyValueNetImpl>& model) {
  std::stringstream ss;
  torch::NoGradGuard no_grad;
  bool unfreeze_trunk = absl::GetFlag(FLAGS_unfreeze_trunk);
  for (auto& name_param : model->named_parameters()) {
    std::string name = name_param.key();
    bool should_hash = false;
    if (unfreeze_trunk) {
      if (name.rfind("policy_head", 0) == 0) {
        should_hash = true;
      }
    } else {
      if (name.find("value_head") == std::string::npos) {
        should_hash = true;
      }
    }
    if (should_hash) {
      auto tensor = name_param.value().contiguous().cpu().to(torch::kFloat32);
      float* data = static_cast<float*>(tensor.data_ptr());
      size_t num_el = tensor.numel();
      ss.write(reinterpret_cast<const char*>(data), num_el * sizeof(float));
    }
  }
  return open_spiel::ComputeStringSHA256(ss.str());
}

// Canonical in-memory identity for the read-only parity gate. Parameter and
// buffer names, shapes and FP32 bytes are included so equality before/after the
// diagnostic proves that neither collection nor replay mutated the model.
std::string HashAllModelState(
    const std::shared_ptr<SharedDunePolicyValueNetImpl>& model) {
  std::stringstream ss;
  torch::NoGradGuard no_grad;
  auto append = [&](const std::string& name, const torch::Tensor& source) {
    const torch::Tensor tensor =
        source.detach().contiguous().cpu().to(torch::kFloat32);
    ss << name << ':';
    for (int64_t dim : tensor.sizes()) ss << dim << ',';
    ss << ';';
    ss.write(reinterpret_cast<const char*>(tensor.data_ptr<float>()),
             tensor.numel() * sizeof(float));
  };
  for (const auto& item : model->named_parameters()) {
    append("P/" + item.key(), item.value());
  }
  for (const auto& item : model->named_buffers()) {
    append("B/" + item.key(), item.value());
  }
  return ComputeStringSHA256(ss.str());
}

// PWO-5 section 7.2 — THE ARCHITECTURE-VERSIONED CHECKPOINT MIGRATION.
//
// Loads trunk, policy head, value head "WITHOUT NUMERICAL CHANGE" from a
// checkpoint that may PREDATE the three auxiliary heads, and leaves the heads at
// their deterministic init (section 7.2) when the checkpoint does not carry
// them.
//
// The mechanism is the temp-model copy already proven in this file for the
// nonlinear-value-head case below, plus one probe. It relies on a documented
// property of torch: `Module::load(InputArchive&)` iterates the MODULE's own
// parameters/buffers/children and reads each by key, so
//
//   * loading a PRE-migration archive into a HEAD-LESS temp model succeeds, and
//   * a POST-migration archive's extra head entries are simply never visited.
//
// So the temp model reads both generations, and the only question is whether to
// then also pull the heads across -- which `try_read` on the head submodule
// answers without ever partially mutating `model`.
//
// Numerical exactness: the trunk/policy/value tensors are `copy_`d from the
// loaded temp model, which is a bitwise copy of the same dtype and shape. This
// is what makes section 7.3's bitwise legacy-inference parity gate passable.
bool LoadModelCheckpointMigrating(
    std::shared_ptr<SharedDunePolicyValueNetImpl> model,
    const std::string& path, torch::Device device) {
  int64_t input_dim = model->input_layer->weight.size(1);
  int64_t hidden_dim = absl::GetFlag(FLAGS_hidden_dim);
  int64_t action_dim = model->policy_head->weight.size(0);
  int num_blocks = absl::GetFlag(FLAGS_num_blocks);
  const bool use_nonlinear = absl::GetFlag(FLAGS_nonlinear_value_head);

  auto temp_model = std::make_shared<SharedDunePolicyValueNetImpl>(
      input_dim, hidden_dim, action_dim, num_blocks, use_nonlinear,
      /*with_aux_heads=*/false);
  temp_model->to(device);
  torch::load(temp_model, path, device);

  torch::NoGradGuard no_grad;
  // Trunk.
  model->input_layer->weight.copy_(temp_model->input_layer->weight);
  if (model->input_layer->bias.defined() &&
      temp_model->input_layer->bias.defined()) {
    model->input_layer->bias.copy_(temp_model->input_layer->bias);
  }
  for (size_t i = 0; i < model->res_blocks.size(); ++i) {
    auto source_params = temp_model->res_blocks[i]->parameters();
    auto target_params = model->res_blocks[i]->parameters();
    for (size_t j = 0; j < source_params.size(); ++j) {
      target_params[j].copy_(source_params[j]);
    }
    auto source_buffers = temp_model->res_blocks[i]->buffers();
    auto target_buffers = model->res_blocks[i]->buffers();
    for (size_t j = 0; j < source_buffers.size(); ++j) {
      target_buffers[j].copy_(source_buffers[j]);
    }
  }
  // Policy head.
  model->policy_head->weight.copy_(temp_model->policy_head->weight);
  if (model->policy_head->bias.defined() &&
      temp_model->policy_head->bias.defined()) {
    model->policy_head->bias.copy_(temp_model->policy_head->bias);
  }
  // Value head(s). Unlike the nonlinear partial-copy path below, the migration
  // DOES carry the value head across: section 7.2 requires the value head load
  // "without numerical change", and a fresh critic here would silently restart
  // the value function at update 0 of every arm.
  model->value_head->weight.copy_(temp_model->value_head->weight);
  if (model->value_head->bias.defined() &&
      temp_model->value_head->bias.defined()) {
    model->value_head->bias.copy_(temp_model->value_head->bias);
  }
  if (use_nonlinear) {
    model->value_head2->weight.copy_(temp_model->value_head2->weight);
    if (model->value_head2->bias.defined() &&
        temp_model->value_head2->bias.defined()) {
      model->value_head2->bias.copy_(temp_model->value_head2->bias);
    }
  }

  // Now the heads, if and only if this checkpoint already has them.
  bool migrated = true;
  if (model->has_aux_heads_) {
    torch::serialize::InputArchive archive;
    archive.load_from(path, device);
    torch::serialize::InputArchive head_archive;
    if (archive.try_read("final_vp_head", head_archive)) {
      // Post-migration checkpoint: load all three normally.
      model->final_vp_head->load(head_archive);
      torch::serialize::InputArchive a2, a3;
      if (!archive.try_read("terminal_round_head", a2) ||
          !archive.try_read("next_own_action_head", a3)) {
        SpielFatalError(
            "PWO-5 migration: the checkpoint at " + path +
            " carries final_vp_head but not all three auxiliary heads. A "
            "partially migrated checkpoint is a STOP, not something to repair "
            "silently -- section 17.5 item 4.");
      }
      model->terminal_round_head->load(a2);
      model->next_own_action_head->load(a3);
      migrated = false;  // it was ALREADY migrated
    } else {
      std::cout << "[PWO-5] MIGRATION: " << path
                << " predates the auxiliary heads. Trunk, policy head and "
                   "value head loaded without numerical change; the three "
                   "heads keep their deterministic init from "
                   "kHeadInitConstant=" << absl::GetFlag(FLAGS_head_init_constant)
                << "." << std::endl;
    }
  }
  return migrated;
}

void LoadModelCheckpoint(std::shared_ptr<SharedDunePolicyValueNetImpl> model,
                         const std::string& path, torch::Device device) {
  // PWO-5: when the model carries the auxiliary heads, the migrating loader is
  // the only correct path -- a plain torch::load would demand head tensors the
  // base checkpoint does not have and would abort the run.
  if (model->has_aux_heads_) {
    LoadModelCheckpointMigrating(model, path, device);
    return;
  }
  if (!absl::GetFlag(FLAGS_nonlinear_value_head)) {
    torch::load(model, path, device);
    return;
  }
  int64_t input_dim = model->input_layer->weight.size(1);
  int64_t hidden_dim = absl::GetFlag(FLAGS_hidden_dim);
  int64_t action_dim = model->policy_head->weight.size(0);
  int num_blocks = absl::GetFlag(FLAGS_num_blocks);

  auto temp_model = std::make_shared<SharedDunePolicyValueNetImpl>(
      input_dim, hidden_dim, action_dim, num_blocks, /*use_nonlinear=*/false);
  temp_model->to(device);

  std::cout << "[INFO] Loading standard checkpoint for partial weight copy (nonlinear model)..." << std::endl;
  torch::load(temp_model, path, device);

  torch::NoGradGuard no_grad;
  model->input_layer->weight.copy_(temp_model->input_layer->weight);
  if (model->input_layer->bias.defined() && temp_model->input_layer->bias.defined()) {
    model->input_layer->bias.copy_(temp_model->input_layer->bias);
  }
  for (size_t i = 0; i < model->res_blocks.size(); ++i) {
    auto& target_block = model->res_blocks[i];
    auto& source_block = temp_model->res_blocks[i];
    auto source_params = source_block->parameters();
    auto target_params = target_block->parameters();
    for (size_t j = 0; j < source_params.size(); ++j) {
      target_params[j].copy_(source_params[j]);
    }
    auto source_buffers = source_block->buffers();
    auto target_buffers = target_block->buffers();
    for (size_t j = 0; j < source_buffers.size(); ++j) {
      target_buffers[j].copy_(source_buffers[j]);
    }
  }
  model->policy_head->weight.copy_(temp_model->policy_head->weight);
  if (model->policy_head->bias.defined() && temp_model->policy_head->bias.defined()) {
    model->policy_head->bias.copy_(temp_model->policy_head->bias);
  }
  std::cout << "[INFO] Partial weight copy completed successfully." << std::endl;
}

void SetOptimizerLearningRate(torch::optim::Optimizer& optimizer, double lr) {
  for (auto& group : optimizer.param_groups()) {
    auto& options = static_cast<torch::optim::AdamWOptions&>(group.options());
    options.lr(lr);
  }
}

std::unique_ptr<torch::optim::AdamW> MakeOptimizer(
    std::shared_ptr<SharedDunePolicyValueNetImpl> model) {
  if (absl::GetFlag(FLAGS_train_value_only)) {
    std::vector<torch::Tensor> trainable_params;
    for (auto& param : model->parameters()) {
      if (param.requires_grad()) {
        trainable_params.push_back(param);
      }
    }
    std::vector<torch::optim::OptimizerParamGroup> groups;
    groups.emplace_back(trainable_params);
    auto optimizer = std::make_unique<torch::optim::AdamW>(
        groups, torch::optim::AdamWOptions(absl::GetFlag(FLAGS_learning_rate)).eps(1e-5));
    static_cast<torch::optim::AdamWOptions&>(
        optimizer->param_groups()[0].options())
        .weight_decay(absl::GetFlag(FLAGS_weight_decay));
    return optimizer;
  }

  std::vector<torch::Tensor> policy_params;
  std::vector<torch::Tensor> other_params;
  // PWO-5 section 7.4: the three auxiliary heads sit in their OWN parameter
  // group with weight_decay = 0.0, IN ALL SIX ARMS.
  //
  // This is not tidiness. In a head-off arm (T, P) the coefficients are exactly
  // zero and the loss terms are not constructed, so the head parameters receive
  // no gradient -- but AdamW's DECOUPLED weight decay does not need a gradient
  // to move a parameter. Left in the default group at --weight_decay, a
  // head-off arm's head tensors would shrink on every step and would silently
  // diverge from a head-on arm's initial values, breaking both the section 9
  // "same tensors, same keys, same order" match and section 15.2 gate 7's
  // bitwise-equal-to-initial check.
  std::vector<torch::Tensor> aux_head_params;
  auto policy_params_set = model->policy_head->parameters();
  std::vector<torch::Tensor> aux_params_set;
  if (model->has_aux_heads_) {
    for (auto& p : model->final_vp_head->parameters()) aux_params_set.push_back(p);
    for (auto& p : model->terminal_round_head->parameters()) aux_params_set.push_back(p);
    for (auto& p : model->next_own_action_head->parameters()) aux_params_set.push_back(p);
  }
  for (auto& param : model->parameters()) {
    bool is_policy = false;
    for (auto& policy_param : policy_params_set) {
      if (param.is_same(policy_param)) {
        is_policy = true;
        break;
      }
    }
    bool is_aux = false;
    for (auto& aux_param : aux_params_set) {
      if (param.is_same(aux_param)) {
        is_aux = true;
        break;
      }
    }
    if (is_policy) {
      policy_params.push_back(param);
    } else if (is_aux) {
      aux_head_params.push_back(param);
    } else {
      other_params.push_back(param);
    }
  }

  // Group order is part of the checkpoint contract (section 9: "same
  // param_groups, same order"), and the post-load re-assertion below indexes by
  // it. Order: 0 = policy, 1 = everything else, 2 = the auxiliary heads.
  std::vector<torch::optim::OptimizerParamGroup> groups;
  groups.emplace_back(policy_params);
  groups.emplace_back(other_params);
  if (model->has_aux_heads_) {
    groups.emplace_back(aux_head_params);
  }
  auto optimizer = std::make_unique<torch::optim::AdamW>(
      groups, torch::optim::AdamWOptions(absl::GetFlag(FLAGS_learning_rate))
                  .eps(1e-5));
  static_cast<torch::optim::AdamWOptions&>(
      optimizer->param_groups()[0].options())
      .weight_decay(absl::GetFlag(FLAGS_policy_weight_decay));
  static_cast<torch::optim::AdamWOptions&>(
      optimizer->param_groups()[1].options())
      .weight_decay(absl::GetFlag(FLAGS_weight_decay));
  if (model->has_aux_heads_) {
    static_cast<torch::optim::AdamWOptions&>(
        optimizer->param_groups()[2].options())
        .weight_decay(0.0);  // section 7.4 -- hard zero, never a flag
  }
  return optimizer;
}

// PWO-5 section 7.2 — the OPTIMIZER half of the migration.
//
// The base optimizer checkpoint was written by an AdamW over the PRE-head
// parameter set in TWO groups. The migrated optimizer has THREE groups and 6
// more tensors, so `torch::load` straight into it cannot work: the archive and
// the optimizer disagree about how many parameters exist.
//
// The migration mirrors the model's: build a TEMPORARY AdamW whose group
// structure is exactly the pre-migration one -- and, critically, whose tensors
// are THE REAL MODEL'S OWN non-head parameters, not copies. AdamW keys its state
// map by `TensorImpl*`, so after loading, the temp optimizer's state is already
// keyed by the identities the real optimizer uses, and the entries can simply be
// moved across. Nothing is re-derived and no state tensor is reconstructed, so
// "optimizer state loaded without numerical change" is exact.
//
// Returns true if it migrated (the archive was pre-head), false if the caller
// should do an ordinary load.
bool LoadOptimizerCheckpointMigrating(
    std::shared_ptr<SharedDunePolicyValueNetImpl> model,
    torch::optim::AdamW& optimizer, const std::string& path,
    torch::Device device) {
  if (!model->has_aux_heads_) return false;

  // Reproduce the pre-migration grouping over the real model's tensors.
  std::vector<torch::Tensor> policy_params;
  std::vector<torch::Tensor> other_params;
  auto policy_params_set = model->policy_head->parameters();
  std::vector<torch::Tensor> aux_params_set;
  for (auto& p : model->final_vp_head->parameters()) aux_params_set.push_back(p);
  for (auto& p : model->terminal_round_head->parameters()) aux_params_set.push_back(p);
  for (auto& p : model->next_own_action_head->parameters()) aux_params_set.push_back(p);
  for (auto& param : model->parameters()) {
    bool is_policy = false;
    for (auto& q : policy_params_set) {
      if (param.is_same(q)) { is_policy = true; break; }
    }
    bool is_aux = false;
    for (auto& q : aux_params_set) {
      if (param.is_same(q)) { is_aux = true; break; }
    }
    if (is_policy) policy_params.push_back(param);
    else if (!is_aux) other_params.push_back(param);
  }

  std::vector<torch::optim::OptimizerParamGroup> legacy_groups;
  legacy_groups.emplace_back(policy_params);
  legacy_groups.emplace_back(other_params);
  torch::optim::AdamW legacy_optimizer(
      legacy_groups,
      torch::optim::AdamWOptions(absl::GetFlag(FLAGS_learning_rate)).eps(1e-5));

  try {
    torch::load(legacy_optimizer, path, device);
  } catch (const c10::Error& e) {
    // Not a pre-migration archive (or a genuinely broken one). Let the caller's
    // ordinary load run and report its own error, so a real corruption is not
    // misreported as a migration failure.
    return false;
  }

  int moved = 0;
  for (auto& entry : legacy_optimizer.state()) {
    optimizer.state()[entry.first] = std::move(entry.second);
    ++moved;
  }
  std::cout << "[PWO-5] MIGRATION: optimizer state moved for " << moved
            << " pre-head parameters from " << path
            << " without numerical change; the three heads' state is "
               "materialized separately at zero." << std::endl;
  return true;
}

// PWO-5 section 7.4 — MATERIALIZE the auxiliary heads' optimizer state.
//
// "This is a registered requirement, not a framework default." With the
// coefficient short-circuit, a head-off arm's head parameters never receive a
// gradient, and AdamW creates per-parameter state LAZILY on the first gradient.
// A head-off arm's optimizer state dict would therefore be MISSING those keys
// while a head-on arm's has them -- breaking section 9's "same tensors, same
// keys, same order" and making section 15.2 gate 7's "auxiliary optimizer state
// all-zero" unverifiable, because ABSENT IS NOT ZERO.
//
// Called in all six arms, whether or not a gradient ever flows.
void MaterializeAuxOptimizerState(
    std::shared_ptr<SharedDunePolicyValueNetImpl> model,
    torch::optim::AdamW& optimizer) {
  if (!model->has_aux_heads_) return;
  if (optimizer.param_groups().size() < 3) {
    SpielFatalError(
        "PWO-5 section 7.4: the model has auxiliary heads but the optimizer has "
        "fewer than three parameter groups. The head group is missing.");
  }
  torch::NoGradGuard no_grad;
  int materialized = 0;
  int already_present = 0;
  for (auto& param : optimizer.param_groups()[2].params()) {
    auto key = param.unsafeGetTensorImpl();
    if (optimizer.state().count(key) > 0) {
      ++already_present;
      continue;
    }
    auto state = std::make_unique<torch::optim::AdamWParamState>();
    state->step(0);
    state->exp_avg(torch::zeros_like(param, torch::MemoryFormat::Preserve));
    state->exp_avg_sq(torch::zeros_like(param, torch::MemoryFormat::Preserve));
    optimizer.state()[key] = std::move(state);
    ++materialized;
  }
  std::cout << "[PWO-5] auxiliary optimizer state: materialized "
            << materialized << " entries (step=0, exp_avg=0, exp_avg_sq=0), "
            << already_present << " already present." << std::endl;
}

std::string GenerateUUID() {
  std::random_device rd;
  std::mt19937_64 generator(rd());
  uint64_t r1 = generator();
  uint64_t r2 = generator();
  // Set version 4 (random) and variant (RFC 4122)
  uint32_t a = (r1 >> 32);
  uint16_t b = (r1 >> 16) & 0xFFFF;
  uint16_t c = (r1 & 0x0FFF) | 0x4000;
  uint16_t d = ((r2 >> 48) & 0x3FFF) | 0x8000;
  uint64_t e = r2 & 0xFFFFFFFFFFFFULL;
  return absl::StrFormat("%08x-%04x-%04x-%04x-%012llx", a, b, c, d, e);
}

// Acceptance prior source this run will actually use: the flag when stated,
// otherwise the collector's library default. Unrecognized names are fatal —
// never silently fall back, since the whole point of the field is that the
// coverage contract is chosen explicitly (WO-20).
AcceptancePriorSource EffectiveAcceptancePriorSource() {
  const std::string name = absl::GetFlag(FLAGS_collector_acceptance_prior);
  if (name.empty()) return OnlineSearchConfig().acceptance_prior_source;
  AcceptancePriorSource source;
  if (!ParseAcceptancePriorSource(name, &source)) {
    SpielFatalError(absl::StrCat(
        "--collector_acceptance_prior=", name, " is not a known prior source. ",
        "Use '", AcceptancePriorSourceName(AcceptancePriorSource::kRawNetworkPrior),
        "' or '", AcceptancePriorSourceName(AcceptancePriorSource::kTreePrior),
        "'."));
  }
  return source;
}

std::string ComputeLegacyConfigFingerprint() {
  open_spiel::json::Object config_obj;
  config_obj["game"] = absl::GetFlag(FLAGS_game);
  config_obj["ppo_minibatch_size"] = absl::GetFlag(FLAGS_ppo_minibatch_size);
  config_obj["ppo_update_epochs"] = absl::GetFlag(FLAGS_ppo_update_epochs);
  config_obj["learning_rate"] = absl::GetFlag(FLAGS_learning_rate);
  config_obj["anneal_lr"] = absl::GetFlag(FLAGS_anneal_lr);
  config_obj["gamma"] = absl::GetFlag(FLAGS_gamma);
  config_obj["gae_lambda"] = absl::GetFlag(FLAGS_gae_lambda);
  config_obj["normalize_advantages"] = absl::GetFlag(FLAGS_normalize_advantages);
  config_obj["ppo_clip_epsilon"] = absl::GetFlag(FLAGS_ppo_clip_epsilon);
  config_obj["ppo_clip_value_loss"] = absl::GetFlag(FLAGS_ppo_clip_value_loss);
  config_obj["entropy_coef"] = absl::GetFlag(FLAGS_entropy_coef);
  config_obj["value_coef"] = absl::GetFlag(FLAGS_value_coef);
  config_obj["grad_clip_norm"] = absl::GetFlag(FLAGS_grad_clip_norm);
  config_obj["target_kl"] = absl::GetFlag(FLAGS_target_kl);
  config_obj["weight_decay"] = absl::GetFlag(FLAGS_weight_decay);
  config_obj["policy_weight_decay"] = absl::GetFlag(FLAGS_policy_weight_decay);
  config_obj["hidden_dim"] = absl::GetFlag(FLAGS_hidden_dim);
  config_obj["num_blocks"] = absl::GetFlag(FLAGS_num_blocks);
  config_obj["logit_cap"] = absl::GetFlag(FLAGS_logit_cap);
  config_obj["shaped_reward_weight"] = absl::GetFlag(FLAGS_shaped_reward_weight);
  config_obj["tleilaxu_breadcrumb_weight"] = absl::GetFlag(FLAGS_tleilaxu_breadcrumb_weight);
  config_obj["tleilaxu_level7_breadcrumb_weight"] = absl::GetFlag(FLAGS_tleilaxu_level7_breadcrumb_weight);
  config_obj["specimen_exchange_penalty"] = absl::GetFlag(FLAGS_specimen_exchange_penalty);
  config_obj["shaping_start_env_steps"] = static_cast<int64_t>(absl::GetFlag(FLAGS_shaping_start_env_steps));
  config_obj["shaping_decay_env_steps"] = static_cast<int64_t>(absl::GetFlag(FLAGS_shaping_decay_env_steps));
  config_obj["reward_scale"] = absl::GetFlag(FLAGS_reward_scale);
  config_obj["train_amp"] = absl::GetFlag(FLAGS_train_amp);
  config_obj["pipeline"] = absl::GetFlag(FLAGS_pipeline);
  config_obj["rollout_games"] = absl::GetFlag(FLAGS_rollout_games);
  config_obj["rollout_transitions"] = absl::GetFlag(FLAGS_rollout_transitions);
  config_obj["search_lambda"] = absl::GetFlag(FLAGS_search_lambda);
  config_obj["search_minibatches_per_update"] = absl::GetFlag(FLAGS_search_minibatches_per_update);
  config_obj["search_minibatch_size"] = absl::GetFlag(FLAGS_search_minibatch_size);

  // Phase 18B online auxiliary-search collection (combined optimization).
  config_obj["online_search_collection"] = absl::GetFlag(FLAGS_online_search_collection);
  config_obj["auxiliary_games"] = absl::GetFlag(FLAGS_auxiliary_games);
  config_obj["auxiliary_search_seed_domain"] = static_cast<int64_t>(absl::GetFlag(FLAGS_auxiliary_search_seed_domain));
  config_obj["collector_dirichlet_epsilon"] = absl::GetFlag(FLAGS_collector_dirichlet_epsilon);
  config_obj["search_loss_coef"] = absl::GetFlag(FLAGS_search_loss_coef);
  config_obj["search_loss_warmup_update"] = absl::GetFlag(FLAGS_search_loss_warmup_update);
  config_obj["abort_grad_norm_ratio"] = absl::GetFlag(FLAGS_abort_grad_norm_ratio);
  config_obj["target_sharpen_exponent"] = absl::GetFlag(FLAGS_target_sharpen_exponent);
  config_obj["swordmaster_grant_fraction"] = absl::GetFlag(FLAGS_swordmaster_grant_fraction);
  config_obj["swordmaster_grant_round"] = absl::GetFlag(FLAGS_swordmaster_grant_round);
  // Leader teacher: pinned into the fingerprint so a resume cannot silently
  // change what the network is being taught at Leader nodes.
  //
  // Added CONDITIONALLY, and that is deliberate. An unconditional key would
  // change config_fingerprint for every existing run -- including runs whose
  // flags are entirely unchanged -- and every one of them would fail its resume
  // integrity check against a checkpoint written before this field existed.
  // With the flag off the object is byte-identical to before, so pre-Leader
  // runs resume untouched; with it on the fingerprint differs, which is exactly
  // the detection this pinning is for.
  if (absl::GetFlag(FLAGS_search_leader_draft)) {
    config_obj["search_leader_draft"] = true;
    config_obj["leader_draft_simulations"] =
        absl::GetFlag(FLAGS_leader_draft_simulations);
    config_obj["leader_mass_only_coverage"] =
        absl::GetFlag(FLAGS_leader_mass_only_coverage);
  }

  std::string json_str = open_spiel::json::ToString(config_obj);
  return open_spiel::ComputeStringSHA256(json_str);
}

json::Object BuildPrePrecisionConfigFingerprintObject() {
  open_spiel::json::Object config_obj;
  config_obj["game"] = absl::GetFlag(FLAGS_game);
  config_obj["ppo_minibatch_size"] = absl::GetFlag(FLAGS_ppo_minibatch_size);
  config_obj["ppo_update_epochs"] = absl::GetFlag(FLAGS_ppo_update_epochs);
  config_obj["learning_rate"] = absl::GetFlag(FLAGS_learning_rate);
  config_obj["anneal_lr"] = absl::GetFlag(FLAGS_anneal_lr);
  config_obj["gamma"] = absl::GetFlag(FLAGS_gamma);
  config_obj["gae_lambda"] = absl::GetFlag(FLAGS_gae_lambda);
  config_obj["normalize_advantages"] = absl::GetFlag(FLAGS_normalize_advantages);
  config_obj["ppo_clip_epsilon"] = absl::GetFlag(FLAGS_ppo_clip_epsilon);
  config_obj["ppo_clip_value_loss"] = absl::GetFlag(FLAGS_ppo_clip_value_loss);
  config_obj["entropy_coef"] = absl::GetFlag(FLAGS_entropy_coef);
  config_obj["value_coef"] = absl::GetFlag(FLAGS_value_coef);
  config_obj["grad_clip_norm"] = absl::GetFlag(FLAGS_grad_clip_norm);
  config_obj["target_kl"] = absl::GetFlag(FLAGS_target_kl);
  config_obj["weight_decay"] = absl::GetFlag(FLAGS_weight_decay);
  config_obj["policy_weight_decay"] = absl::GetFlag(FLAGS_policy_weight_decay);
  config_obj["hidden_dim"] = absl::GetFlag(FLAGS_hidden_dim);
  config_obj["num_blocks"] = absl::GetFlag(FLAGS_num_blocks);
  config_obj["logit_cap"] = absl::GetFlag(FLAGS_logit_cap);
  config_obj["shaped_reward_weight"] = absl::GetFlag(FLAGS_shaped_reward_weight);
  config_obj["tleilaxu_breadcrumb_weight"] = absl::GetFlag(FLAGS_tleilaxu_breadcrumb_weight);
  config_obj["tleilaxu_level7_breadcrumb_weight"] = absl::GetFlag(FLAGS_tleilaxu_level7_breadcrumb_weight);
  config_obj["specimen_exchange_penalty"] = absl::GetFlag(FLAGS_specimen_exchange_penalty);
  config_obj["shaping_start_env_steps"] = static_cast<int64_t>(absl::GetFlag(FLAGS_shaping_start_env_steps));
  config_obj["shaping_decay_env_steps"] = static_cast<int64_t>(absl::GetFlag(FLAGS_shaping_decay_env_steps));
  config_obj["reward_scale"] = absl::GetFlag(FLAGS_reward_scale);
  config_obj["train_amp"] = absl::GetFlag(FLAGS_train_amp);
  config_obj["pipeline"] = absl::GetFlag(FLAGS_pipeline);
  config_obj["rollout_games"] = absl::GetFlag(FLAGS_rollout_games);
  config_obj["rollout_transitions"] = absl::GetFlag(FLAGS_rollout_transitions);
  config_obj["search_lambda"] = absl::GetFlag(FLAGS_search_lambda);
  config_obj["search_minibatches_per_update"] = absl::GetFlag(FLAGS_search_minibatches_per_update);
  config_obj["search_minibatch_size"] = absl::GetFlag(FLAGS_search_minibatch_size);

  // Phase 18B online auxiliary-search collection (combined optimization).
  config_obj["online_search_collection"] = absl::GetFlag(FLAGS_online_search_collection);
  config_obj["auxiliary_games"] = absl::GetFlag(FLAGS_auxiliary_games);
  config_obj["auxiliary_search_seed_domain"] = static_cast<int64_t>(absl::GetFlag(FLAGS_auxiliary_search_seed_domain));
  config_obj["collector_dirichlet_epsilon"] = absl::GetFlag(FLAGS_collector_dirichlet_epsilon);
  config_obj["search_loss_coef"] = absl::GetFlag(FLAGS_search_loss_coef);
  config_obj["search_loss_warmup_update"] = absl::GetFlag(FLAGS_search_loss_warmup_update);
  config_obj["abort_grad_norm_ratio"] = absl::GetFlag(FLAGS_abort_grad_norm_ratio);
  config_obj["target_sharpen_exponent"] = absl::GetFlag(FLAGS_target_sharpen_exponent);
  config_obj["swordmaster_grant_fraction"] = absl::GetFlag(FLAGS_swordmaster_grant_fraction);
  config_obj["swordmaster_grant_round"] = absl::GetFlag(FLAGS_swordmaster_grant_round);
  // Leader teacher: pinned into the fingerprint so a resume cannot silently
  // change what the network is being taught at Leader nodes.
  //
  // Added CONDITIONALLY, and that is deliberate. An unconditional key would
  // change config_fingerprint for every existing run -- including runs whose
  // flags are entirely unchanged -- and every one of them would fail its resume
  // integrity check against a checkpoint written before this field existed.
  // With the flag off the object is byte-identical to before, so pre-Leader
  // runs resume untouched; with it on the fingerprint differs, which is exactly
  // the detection this pinning is for.
  if (absl::GetFlag(FLAGS_search_leader_draft)) {
    config_obj["search_leader_draft"] = true;
    config_obj["leader_draft_simulations"] =
        absl::GetFlag(FLAGS_leader_draft_simulations);
    config_obj["leader_mass_only_coverage"] =
        absl::GetFlag(FLAGS_leader_mass_only_coverage);
  }
  // Acceptance prior source is training semantics, not a launcher detail: it
  // decides which searches become CE targets. Fingerprinted so two runs that
  // measure coverage differently cannot stamp their examples with the same
  // provenance hash (WO-20). Deliberately NOT added to the legacy fingerprint
  // above — that one is a frozen field set whose job is to recognize older
  // manifests, so extending it would defeat its purpose.
  config_obj["collector_acceptance_prior"] =
      std::string(AcceptancePriorSourceName(EffectiveAcceptancePriorSource()));

  // New flags added for complete config fingerprint validation
  config_obj["deterministic"] = absl::GetFlag(FLAGS_deterministic);
  config_obj["deterministic_rollout_eval"] = absl::GetFlag(FLAGS_deterministic_rollout_eval);
  // Critic-remediation settings are training semantics, not launcher details.
  // Persist them in the fingerprint so a partially completed value-only run
  // cannot be resumed with a different set of trainable parameters or data.
  config_obj["train_value_only"] = absl::GetFlag(FLAGS_train_value_only);
  config_obj["unfreeze_trunk"] = absl::GetFlag(FLAGS_unfreeze_trunk);
  config_obj["nonlinear_value_head"] =
      absl::GetFlag(FLAGS_nonlinear_value_head);
  config_obj["sample_counterfactual_states"] =
      absl::GetFlag(FLAGS_sample_counterfactual_states);
  config_obj["counterfactual_samples_per_game"] =
      absl::GetFlag(FLAGS_counterfactual_samples_per_game);
  config_obj["counterfactual_replay_weight"] =
      absl::GetFlag(FLAGS_counterfactual_replay_weight);
  config_obj["policy_kl_anchor_coeff"] =
      absl::GetFlag(FLAGS_policy_kl_anchor_coeff);

  return config_obj;
}

std::string ComputeCurrentPrePrecisionConfigFingerprint() {
  return open_spiel::ComputePrePrecisionConfigFingerprint(
      BuildPrePrecisionConfigFingerprintObject());
}

std::string ComputeConfigFingerprint() {
  return open_spiel::ComputePrecisionConfigFingerprint(
      BuildPrePrecisionConfigFingerprintObject(),
      absl::GetFlag(FLAGS_rollout_amp),
      absl::GetFlag(FLAGS_allow_tf32));
}

// ---------------------------------------------------------------------------
// PWO-5 manifest / resume contract (registration Appendix A.1 note 3).
// ---------------------------------------------------------------------------
//
// Appendix A.1 registers that "the manifest gains matching fields, asserted on
// resume and on the u600 extension": both target flags, the held-out split
// digest, the three head coefficients, the three sampler-shape flags, the
// Huber delta, the head-init constant, --emit_canary_columns, and section
// 8.6's update-1 sampler digest.
//
// These are kept OUT of ComputeConfigFingerprint deliberately. That fingerprint
// is a frozen field set whose job is to recognize older manifests; extending it
// would make every pre-PWO-5 checkpoint fail to resume as an ANONYMOUS hash
// mismatch. A dedicated block instead fails with the field NAME that moved,
// and leaves every legacy fingerprint verifying exactly as before.
//
// Every double is serialized as a %.17g STRING, never as a JSON number:
// open_spiel's json.cc writer emits doubles as `%f` at six decimals, so a
// coefficient of 5e-8 would persist as "0.000000" and compare equal to zero.
std::string Pwo5Double(double v) {
  char buf[64];
  std::snprintf(buf, sizeof(buf), "%.17g", v);
  return std::string(buf);
}

// Populated once at startup by BuildPwo5ManifestFields(), then merged into
// every manifest this process writes. Empty when the run is not a PWO-5 arm
// (no --aux_target_path), in which case nothing is written and nothing is
// asserted -- a legacy run is untouched.
open_spiel::json::Object g_pwo5_manifest_fields;
bool g_pwo5_manifest_active = false;

// A canonical, order-stable digest over every field above, so a single
// comparison covers the whole block and a mismatch cannot be missed by a
// checker that forgot one key. json::Object is a std::map, so ToString emits
// keys in lexicographic order regardless of insertion order.
std::string ComputePwo5Fingerprint(const open_spiel::json::Object& fields) {
  return open_spiel::ComputeStringSHA256(open_spiel::json::ToString(fields));
}

// The head-initialization IDENTITY: a digest over the three auxiliary heads'
// parameter tensors as they stand immediately after construction/migration.
//
// Section 7.2 requires the initial head parameters be BYTE-IDENTICAL on all six
// arms (derived from kHeadInitConstant, not from any run seed). A digest makes
// that checkable rather than asserted, and it is what section 15.2 gate 7's
// "head parameters bitwise equal their initial values" is compared against.
std::string ComputeAuxHeadIdentity(
    const std::shared_ptr<SharedDunePolicyValueNetImpl>& model) {
  torch::NoGradGuard no_grad;
  std::string blob;
  auto append = [&blob](const torch::Tensor& t) {
    torch::Tensor c = t.detach().to(torch::kCPU).contiguous().to(torch::kFloat32);
    const auto n = c.numel();
    const char* p = reinterpret_cast<const char*>(c.data_ptr<float>());
    blob.append(p, static_cast<size_t>(n) * sizeof(float));
  };
  for (auto& p : model->final_vp_head->parameters()) append(p);
  for (auto& p : model->terminal_round_head->parameters()) append(p);
  for (auto& p : model->next_own_action_head->parameters()) append(p);
  return open_spiel::ComputeStringSHA256(blob);
}

std::string GetSearchLabelFingerprint(const std::string& search_label_dir) {
  if (search_label_dir.empty()) {
    return "";
  }
  std::filesystem::path manifest_path = std::filesystem::path(search_label_dir) / "manifest.json";
  if (!std::filesystem::exists(manifest_path)) {
    std::cout << "Warning: search_label_dir is specified but manifest.json not found at " << manifest_path << "\n";
    return "";
  }
  try {
    std::ifstream ifs(manifest_path);
    std::string content((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
    auto val_opt = open_spiel::json::FromString(content);
    if (val_opt.has_value() && val_opt->IsObject()) {
      const auto& obj = val_opt->GetObject();
      auto it = obj.find("search_label_fingerprint");
      if (it != obj.end() && it->second.IsString()) {
        return it->second.GetString();
      }
    }
  } catch (const std::exception& e) {
    std::cerr << "Error reading search label manifest at " << manifest_path << ": " << e.what() << "\n";
  }
  return "";
}

void SaveCheckpoint(std::shared_ptr<SharedDunePolicyValueNetImpl> model,
                    torch::optim::AdamW& optimizer,
                    const std::string& model_path,
                    const std::string& optim_path,
                    int global_update,
                    int target_end_update,
                    uint64_t total_env_steps,
                    uint64_t next_episode_id,
                    uint64_t base_seed,
                    int seed_scheme_version,
                    const std::string& config_fingerprint,
                    const std::string& search_label_fingerprint,
                    const std::string& run_uuid,
                    const open_spiel::OnlineCollectionState* aux_state = nullptr,
                    const open_spiel::SearchPiState* search_pi_state = nullptr) {
  std::string model_tmp = model_path + ".tmp";
  std::string optim_tmp = optim_path + ".tmp";

  std::filesystem::path manifest_path = model_path;
  manifest_path.replace_extension(".json");
  std::string manifest_path_str = manifest_path.string();
  std::string manifest_tmp = manifest_path_str + ".tmp";

  try {
    torch::save(model, model_tmp);
    torch::save(optimizer, optim_tmp);

    size_t model_size = 0;
    std::string model_hash = ComputeFileSHA256(model_tmp, &model_size);

    size_t optim_size = 0;
    std::string optim_hash = ComputeFileSHA256(optim_tmp, &optim_size);

    std::string checkpoint_uuid = GenerateUUID();
    json::Object manifest_obj;
    manifest_obj["schema_version"] = static_cast<int64_t>(2);
    manifest_obj["checkpoint_uuid"] = checkpoint_uuid;
    manifest_obj["global_update"] = static_cast<int64_t>(global_update);
    manifest_obj["target_end_update"] = static_cast<int64_t>(target_end_update);
    manifest_obj["total_env_steps"] = static_cast<int64_t>(total_env_steps);
    manifest_obj["next_episode_id"] = static_cast<int64_t>(next_episode_id);
    manifest_obj["base_seed"] = static_cast<int64_t>(base_seed);
    manifest_obj["seed_scheme_version"] = static_cast<int64_t>(seed_scheme_version);
    manifest_obj["config_fingerprint"] = config_fingerprint;
    WritePpoPrecisionManifestFields(
        manifest_obj, absl::GetFlag(FLAGS_rollout_amp),
        absl::GetFlag(FLAGS_allow_tf32));
    manifest_obj["search_label_fingerprint"] = search_label_fingerprint;
    manifest_obj["run_uuid"] = run_uuid;
    manifest_obj["model_filename"] = std::filesystem::path(model_path).filename().string();
    manifest_obj["model_file_size"] = static_cast<int64_t>(model_size);
    manifest_obj["model_sha256"] = model_hash;
    manifest_obj["optimizer_filename"] = std::filesystem::path(optim_path).filename().string();
    manifest_obj["optimizer_file_size"] = static_cast<int64_t>(optim_size);
    manifest_obj["optimizer_sha256"] = optim_hash;
    manifest_obj["hidden_dim"] = static_cast<int64_t>(absl::GetFlag(FLAGS_hidden_dim));
    manifest_obj["num_blocks"] = static_cast<int64_t>(absl::GetFlag(FLAGS_num_blocks));
    // PWO-5 Appendix A.1 note 3: the matching fields, asserted on resume.
    if (g_pwo5_manifest_active) {
      manifest_obj["pwo5"] = json::Value(g_pwo5_manifest_fields);
      manifest_obj["pwo5_fingerprint"] =
          json::Value(ComputePwo5Fingerprint(g_pwo5_manifest_fields));
    }
    // Phase 18B: persist online-collection state for exact resume (plan §18C).
    if (aux_state != nullptr) {
      open_spiel::WriteOnlineCollectionState(manifest_obj, *aux_state);
    }
    // Search-PI: the resume cursor and the full search configuration live HERE,
    // in the manifest, not in diagnostics.csv. Conditional, so a manifest
    // written by any other mode is byte-identical to before.
    if (search_pi_state != nullptr) {
      manifest_obj["search_pi"] =
          json::Value(open_spiel::WriteSearchPiState(*search_pi_state));
    }

    {
      std::ofstream ofs(manifest_tmp);
      if (!ofs) {
        throw std::runtime_error("Could not open manifest temp file for writing: " + manifest_tmp);
      }
      ofs << json::ToString(manifest_obj, true);
    }

    std::filesystem::rename(model_tmp, model_path);
    std::filesystem::rename(optim_tmp, optim_path);
    std::filesystem::rename(manifest_tmp, manifest_path);

    std::cout << "Saved checkpoint successfully:\n"
              << "  Model: " << model_path << " (" << model_size << " bytes, sha256=" << model_hash << ")\n"
              << "  Optimizer: " << optim_path << " (" << optim_size << " bytes, sha256=" << optim_hash << ")\n"
              << "  Manifest: " << manifest_path << "\n";
  } catch (const std::exception& e) {
    std::cerr << "CRITICAL ERROR: SaveCheckpoint failed: " << e.what() << "\n";
    if (std::filesystem::exists(model_tmp)) std::filesystem::remove(model_tmp);
    if (std::filesystem::exists(optim_tmp)) std::filesystem::remove(optim_tmp);
    if (std::filesystem::exists(manifest_tmp)) std::filesystem::remove(manifest_tmp);
    SpielFatalError("SaveCheckpoint failed, aborting training to prevent corrupted states.");
  }
}

struct CounterfactualPending {
  PpoTransition transition;
  int replay_weight;
};

int PpoSimulation(uint64_t master, uint64_t episode_id, const Game& game,
                  std::shared_ptr<IGameEvaluator> evaluator, int64_t obs_size,
                  std::vector<PpoTransition>* trajectory,
                  std::atomic<uint64_t>* total_env_steps,
                  float reward_lambda,
                  WorkerStats* local_stats,
                  VrpoCapturedEpisode* vrpo_episode) {

  if (vrpo_episode != nullptr) {
    *vrpo_episode = VrpoCapturedEpisode{};
    vrpo_episode->episode_id = episode_id;
  }

  auto chance_rng = dune_seed::MakeRng64(dune_seed::DeriveSeed(master, dune_seed::kDomainTrain, episode_id, dune_seed::kStreamChance));
  std::mt19937_64 policy_rng[4];
  for (int p = 0; p < 4; ++p) {
    uint64_t seed = dune_seed::DeriveSeed(master, dune_seed::kDomainTrain, episode_id, dune_seed::kStreamPolicyPlayer0 + p);
    policy_rng[p] = dune_seed::MakeRng64(seed);
  }

  while (true) {
    std::unique_ptr<State> state = game.NewInitialState();
    bool provides_info_state_tensor =
        game.GetType().provides_information_state_tensor;
    bool provides_observations_tensor =
        game.GetType().provides_observation_tensor;

    const auto* dune_state =
        dynamic_cast<const dune_imperium::DuneImperiumState*>(state.get());

    std::vector<int> current_tleilaxu(game.NumPlayers(), 0);
    std::vector<bool> had_swordmaster(game.NumPlayers(), false);
    std::vector<int> prev_conflict_vp(game.NumPlayers(), 0);
    std::vector<int> prev_total_vp(game.NumPlayers(), 0);
    if (dune_state != nullptr) {
      for (int p = 0; p < game.NumPlayers(); ++p) {
        current_tleilaxu[p] = dune_state->GetTleilaxuTrackForTesting(p);
        had_swordmaster[p] = dune_state->HasSwordmaster(p);
        prev_conflict_vp[p] = dune_state->ConflictVpDelta(p);
        prev_total_vp[p] = dune_state->GetPlayerVp(p);
      }
    }

    CombatCreditAccumulator combat_accumulator;
    std::vector<int> last_transition_index(game.NumPlayers(), -1);
    std::vector<int> pre_combat_strength(game.NumPlayers(), 0);
    dune_imperium::GamePhase pre_action_phase = dune_imperium::GamePhase::kDeal;

    float logit_cap = static_cast<float>(absl::GetFlag(FLAGS_logit_cap));
    double shaped_weight = absl::GetFlag(FLAGS_shaped_reward_weight);
    double tleilaxu_breadcrumb_weight =
        absl::GetFlag(FLAGS_tleilaxu_breadcrumb_weight);
    double tleilaxu_level7_breadcrumb_weight =
        absl::GetFlag(FLAGS_tleilaxu_level7_breadcrumb_weight);
    double specimen_exchange_penalty =
        absl::GetFlag(FLAGS_specimen_exchange_penalty);
    // PWO-5 section 14.1b. Hoisted per episode alongside the other flag reads,
    // so the canary measurement costs one bool test per decision when off.
    const bool emit_canary_columns =
        (local_stats != nullptr) && absl::GetFlag(FLAGS_emit_canary_columns);
    // Carried across the CenterAndCapLegalLogits call: the PRE-cap half (SAT,
    // and the role) is measured before it, the POST-cap half (NE,
    // max_action_prob) after it, and both halves must describe the SAME
    // decision.
    bool canary_measure_this_decision = false;
    int canary_role_this_decision = 0;

    torch::NoGradGuard no_grad;
    int game_length = 0;
    int counterfactual_samples = 0;
    std::vector<CounterfactualPending> pending_cf;
    while (!state->IsTerminal()) {
      ++game_length;
      if (game_length > 5000) {
        std::cerr << "Possible infinite loop detected. State:\n"
                  << state->ToString() << "\n";
        std::abort();
      }

      if (state->IsChanceNode()) {
        std::vector<std::pair<Action, double>> outcomes = state->ChanceOutcomes();
        Action action = game.GetType().chance_mode == GameType::ChanceMode::kSampledStochastic
                    ? outcomes.front().first
                    : SampleAction(outcomes, chance_rng).first;
        state->ApplyAction(action);
        continue;
      }

      if (state->CurrentPlayer() == kSimultaneousPlayerId) {
        std::vector<Action> joint_action;
        for (int p = 0; p < game.NumPlayers(); ++p) {
          std::vector<Action> actions = state->LegalActions(p);
          if (actions.empty()) {
            joint_action.push_back(0);
          } else {
            std::uniform_int_distribution<int> dis(0, actions.size() - 1);
            joint_action.push_back(actions[dis(policy_rng[p])]);
          }
        }
        state->ApplyActions(joint_action);
        continue;
      }

      if (absl::GetFlag(FLAGS_train_value_only)) {
        if (!prohibited_histories.empty() && prohibited_histories.count(state->History())) {
          SpielFatalError("CRITICAL PROTECTION BOUNDARY FAILURE: Attempted to process or train on a prohibited official-corpus state!");
        }
      }

      Player current_player = state->CurrentPlayer();
      std::vector<Action> actions = state->LegalActions();
      if (actions.empty()) {
        std::cerr << "Non-terminal state has no legal actions. State:\n"
                  << state->ToString() << "\n";
        std::abort();
      }

      std::vector<float> obs(obs_size, 0.0f);
      if (provides_info_state_tensor && current_player >= 0) {
        state->InformationStateTensor(current_player, absl::MakeSpan(obs));
      } else if (provides_observations_tensor && current_player >= 0) {
        state->ObservationTensor(current_player, absl::MakeSpan(obs));
      }

      EvalResult result = evaluator->Evaluate(obs);
      std::vector<float> logits = std::move(result.logits);
      const bool parity_capture =
          !absl::GetFlag(FLAGS_numerical_parity_output).empty();
      std::vector<float> parity_raw_legal_logits;
      if (parity_capture) {
        const int64_t decision_index =
            static_cast<int64_t>(trajectory->size());
        const std::string identity = absl::StrFormat(
            "episode=%llu decision=%lld",
            static_cast<unsigned long long>(episode_id),
            static_cast<long long>(decision_index));
        ++local_stats->parity_decisions;
        const std::vector<int64_t> legal_ids(actions.begin(), actions.end());
        std::string validation_error;
        const PpoParityPrecapCaptureValidation validation =
            ValidateAndCapturePpoParityPrecap(
                logits, game.NumDistinctActions(), legal_ids,
                &parity_raw_legal_logits, &validation_error);
        if (validation.full_width_ok) {
          ++local_stats->parity_full_width_ok;
        } else {
          local_stats->parity_capture_errors.push_back(absl::StrFormat(
              "%s: evaluator width %d != action_dim %d", identity,
              static_cast<int>(logits.size()), game.NumDistinctActions()));
        }
        if (validation.full_finite_ok) {
          ++local_stats->parity_full_finite_ok;
        } else {
          local_stats->parity_capture_errors.push_back(
              identity + ": evaluator emitted a nonfinite full logit");
        }
        if (validation.legal_ids_unique_in_range) {
          ++local_stats->parity_legal_ids_unique_in_range;
        } else {
          local_stats->parity_capture_errors.push_back(
              identity + ": legal action IDs are duplicated or out of range");
        }
      }

      // --- Phase A canary: pre-cap legal-centered |z| for this decision ---
      // Measured HERE because CenterAndCapLegalLogits below mutates `logits` in
      // place, destroying the raw values. The centering is a deliberate mirror
      // of that function's: double accumulator, mean cast to float, same
      // in-range guard -- so `z` is exactly what it would subtract, taken
      // BEFORE its tanh. Restricted to nontrivial decisions (>= 2 legal
      // actions) to match the trainer's own nontrivial definition; a forced
      // decision's centered logit is identically 0 and would only dilute the
      // percentiles. Observational only: no RNG draw, no tensor op, nothing
      // read back into the acting path, so the sampled action is unaffected.
      if (local_stats != nullptr && actions.size() >= 2) {
        double legal_sum = 0.0;
        int legal_count = 0;
        for (Action a : actions) {
          if (a >= 0 && static_cast<size_t>(a) < logits.size()) {
            legal_sum += logits[a];
            ++legal_count;
          }
        }
        if (legal_count > 0) {
          const float legal_mean = static_cast<float>(legal_sum / legal_count);
          float max_absz = 0.0f;
          for (Action a : actions) {
            if (a < 0 || static_cast<size_t>(a) >= logits.size()) continue;
            max_absz = std::max(max_absz, std::abs(logits[a] - legal_mean));
          }
          local_stats->precap_absz_all.push_back(max_absz);
          const DuneDecisionRole decision_role = ClassifyDuneDecisionRole(
              *state, current_player, /*has_active_session=*/false);
          switch (decision_role) {
            case DuneDecisionRole::kAgentPrimary:
              local_stats->precap_absz_primary.push_back(max_absz);
              break;
            case DuneDecisionRole::kAgentContinuation:
              local_stats->precap_absz_continuation.push_back(max_absz);
              break;
            case DuneDecisionRole::kPurchase:
              local_stats->precap_absz_purchase.push_back(max_absz);
              break;
            default:
              break;
          }

          // --- PWO-5 section 14.1b: SAT, the PRE-cap half. ----------------
          //
          // `SAT` is PER LOGIT, over the decision's legal actions, against the
          // CONFIGURED cap -- three properties that are each load-bearing and
          // none of which the existing `frac_decisions_absz_ge10` has:
          // that column is per DECISION (max|z| over the row), and hard-codes
          // 10.0 independently of --logit_cap. The two are reported side by
          // side and are never conflated.
          //
          // It must be PRE-cap. The cap is `c * tanh(z / c)`, not a clamp, so
          // in exact arithmetic |y| < c strictly for every finite z and a
          // post-cap definition would be IDENTICALLY ZERO -- a rail that can
          // never fire. In finite precision it is worse than zero: tanh rounds
          // to exactly 1.0 at |z| >= 90.1 in fp32 but at |z| >= 31.2 in bf16,
          // so the rail's sensitivity would become a function of the autocast
          // dtype. Captured here, before CenterAndCapLegalLogits mutates
          // `logits` in place.
          if (emit_canary_columns) {
            for (Action a : actions) {
              if (a < 0 || static_cast<size_t>(a) >= logits.size()) continue;
              ++local_stats->canary_sat_logits;
              if (std::abs(logits[a] - legal_mean) >= logit_cap) {
                ++local_stats->canary_sat_ge_cap;
              }
            }
            canary_role_this_decision = static_cast<int>(decision_role);
            canary_measure_this_decision = true;
          }
        }
      }

      CenterAndCapLegalLogits(logits, actions, logit_cap);

      // The ordinary PPO transition stores only the sampled action's scalar
      // log-probability. The numerical-parity gate needs the whole behavior
      // categorical to measure full legal KL, but retaining 2--50 floats per
      // decision is deliberately opt-in so normal training has no memory or
      // serialization change.
      std::vector<float> parity_behavior_log_probs;
      int parity_decision_role = -1;
      if (parity_capture) {
        parity_decision_role = static_cast<int>(ClassifyDuneDecisionRole(
            *state, current_player, /*has_active_session=*/false));
        float max_logit = -std::numeric_limits<float>::infinity();
        for (Action a : actions) {
          if (a >= 0 && static_cast<size_t>(a) < logits.size()) {
            max_logit = std::max(max_logit, logits[a]);
          }
        }
        std::vector<double> weights(actions.size(), 0.0);
        double total_weight = 0.0;
        for (size_t i = 0; i < actions.size(); ++i) {
          const Action a = actions[i];
          if (a >= 0 && static_cast<size_t>(a) < logits.size() &&
              std::isfinite(max_logit)) {
            weights[i] = std::exp(
                static_cast<double>(logits[a] - max_logit));
            if (!std::isfinite(weights[i]) || weights[i] < 0.0) {
              weights[i] = 0.0;
            }
          }
          total_weight += weights[i];
        }
        if (!(total_weight > 0.0) || !std::isfinite(total_weight)) {
          weights.assign(actions.size(), 1.0);
          total_weight = static_cast<double>(actions.size());
        }
        parity_behavior_log_probs.reserve(actions.size());
        for (double weight : weights) {
          parity_behavior_log_probs.push_back(
              weight > 0.0
                  ? static_cast<float>(std::log(weight / total_weight))
                  : -std::numeric_limits<float>::infinity());
        }
      }

      // --- PWO-5 section 14.1b: NE and max_action_prob, the POST-cap half. --
      //
      // Taken from the distribution the behaviour policy ACTUALLY SAMPLES FROM
      // (SamplePolicyAction, immediately below), which is the legal softmax of
      // the capped logits. Post-cap is what section 14.2's positive-entropy-
      // floor argument assumes: because the cap is applied BEFORE the softmax,
      // the softmax argument range is bounded by (-c, +c), so post-cap entropy
      // has a strictly positive floor for every n_legal >= 2 and can never
      // reach 0 -- which is why an absolute entropy threshold is unanchorable
      // and the rails are decline-based rather than absolute.
      //
      // The normalizer is log(n_legal) -- not log(vocabulary_size) and not
      // log(len(visits)). n_legal >= 2 here by construction, so log(n_legal)
      // > 0 always and the repository's two incompatible conventions for the
      // n_legal == 1 case are both moot.
      if (canary_measure_this_decision) {
        float max_logit = -std::numeric_limits<float>::infinity();
        for (Action a : actions) {
          if (a >= 0 && static_cast<size_t>(a) < logits.size()) {
            max_logit = std::max(max_logit, logits[a]);
          }
        }
        double total = 0.0;
        std::vector<double> probs;
        probs.reserve(actions.size());
        for (Action a : actions) {
          double w = 0.0;
          if (a >= 0 && static_cast<size_t>(a) < logits.size() &&
              std::isfinite(max_logit)) {
            w = std::exp(static_cast<double>(logits[a]) -
                         static_cast<double>(max_logit));
          }
          probs.push_back(w);
          total += w;
        }
        if (total > 0.0) {
          double h = 0.0;
          double max_p = 0.0;
          for (double& w : probs) {
            w /= total;
            if (w > 0.0) h -= w * std::log(w);
            max_p = std::max(max_p, w);
          }
          const double ne = h / std::log(static_cast<double>(actions.size()));
          local_stats->canary_ne_sum += ne;
          ++local_stats->canary_ne_n;
          local_stats->canary_ne_sum_role[canary_role_this_decision] += ne;
          ++local_stats->canary_ne_n_role[canary_role_this_decision];
          local_stats->canary_maxprob_sum += max_p;
        }
        canary_measure_this_decision = false;
      }

      std::optional<std::vector<double>> vrpo_legal_probabilities;
      if (vrpo_episode != nullptr) vrpo_legal_probabilities.emplace();
      const PolicyDistributionSample policy_sample =
          SamplePolicyDistribution(&policy_rng[current_player], logits,
                                   actions,
                                   vrpo_legal_probabilities.has_value()
                                       ? &*vrpo_legal_probabilities
                                       : nullptr);
      Action action = policy_sample.action;
      float old_log_prob = policy_sample.chosen_log_probability;
      if (parity_capture) {
        const std::vector<int64_t> legal_ids(actions.begin(), actions.end());
        std::string validation_error;
        if (ValidatePpoParityBehaviorCaptureWidthsAndChoice(
                legal_ids, action, parity_raw_legal_logits,
                parity_behavior_log_probs, &validation_error)) {
          ++local_stats->parity_chosen_once;
        } else {
          local_stats->parity_capture_errors.push_back(absl::StrFormat(
              "episode=%llu decision=%lld: %s",
              static_cast<unsigned long long>(episode_id),
              static_cast<long long>(trajectory->size()), validation_error));
        }
      }

      if (absl::GetFlag(FLAGS_sample_counterfactual_states) &&
          counterfactual_samples <
              absl::GetFlag(FLAGS_counterfactual_samples_per_game) &&
          ClassifyDuneDecisionRole(*state, current_player, false) == DuneDecisionRole::kAgentPrimary) {
        // Clone independent RNGs to prevent perturbing the live trajectory
        std::mt19937_64 cf_chance_rng = chance_rng;
        std::mt19937_64 cf_policy_rng[4];
        for (int p = 0; p < 4; ++p) {
          cf_policy_rng[p] = policy_rng[p];
        }

        std::vector<Action> alternate_actions;
        alternate_actions.reserve(actions.size());
        for (Action candidate : actions) {
          if (candidate != action) alternate_actions.push_back(candidate);
        }
        if (!alternate_actions.empty()) {
          // One uniformly sampled alternative per encountered primary state.
          // Episode/player RNG streams make this deterministic while the full
          // raw-policy game set provides coverage across all legal actions.
          std::uniform_int_distribution<size_t> alternate_distribution(
              0, alternate_actions.size() - 1);
          Action alt_act = alternate_actions[
              alternate_distribution(cf_policy_rng[current_player])];

          auto succ_state = state->Clone();
          succ_state->ApplyAction(alt_act);

          while (succ_state->IsChanceNode()) {
            auto outcomes = succ_state->ChanceOutcomes();
            if (outcomes.empty()) break;
            Action chance_act = game.GetType().chance_mode == GameType::ChanceMode::kSampledStochastic
                        ? outcomes.front().first
                        : SampleAction(outcomes, cf_chance_rng).first;
            succ_state->ApplyAction(chance_act);
          }

          std::vector<float> succ_obs(obs_size, 0.0f);
          bool valid = false;
          float succ_val = 0.0f;
          float target_return = 0.0f;
          if (!succ_state->IsTerminal()) {
            if (provides_info_state_tensor) {
              succ_state->InformationStateTensor(current_player, absl::MakeSpan(succ_obs));
              valid = true;
            } else if (provides_observations_tensor) {
              succ_state->ObservationTensor(current_player, absl::MakeSpan(succ_obs));
              valid = true;
            }
            if (valid) {
              EvalResult succ_result = evaluator->Evaluate(succ_obs);
              succ_val = succ_result.value;

              // Run a counterfactual rollout to get a terminal-return learning signal
              auto roll_state = succ_state->Clone();
              while (!roll_state->IsTerminal()) {
                if (roll_state->IsChanceNode()) {
                  auto outcomes = roll_state->ChanceOutcomes();
                  if (outcomes.empty()) break;
                  Action choice = game.GetType().chance_mode == GameType::ChanceMode::kSampledStochastic
                              ? outcomes.front().first
                              : SampleAction(outcomes, cf_chance_rng).first;
                  roll_state->ApplyAction(choice);
                  continue;
                }

                if (roll_state->CurrentPlayer() == kSimultaneousPlayerId) {
                  std::vector<Action> joint_action;
                  for (int p = 0; p < game.NumPlayers(); ++p) {
                    std::vector<Action> actions = roll_state->LegalActions(p);
                    if (actions.empty()) {
                      joint_action.push_back(0);
                    } else {
                      std::uniform_int_distribution<int> dis(0, actions.size() - 1);
                      joint_action.push_back(actions[dis(cf_policy_rng[p])]);
                    }
                  }
                  roll_state->ApplyActions(joint_action);
                  continue;
                }

                Player curr_p = roll_state->CurrentPlayer();
                std::vector<Action> curr_acts = roll_state->LegalActions();
                std::vector<float> curr_obs(obs_size, 0.0f);
                if (provides_info_state_tensor) {
                  roll_state->InformationStateTensor(curr_p, absl::MakeSpan(curr_obs));
                } else {
                  roll_state->ObservationTensor(curr_p, absl::MakeSpan(curr_obs));
                }
                EvalResult roll_res = evaluator->Evaluate(curr_obs);
                CenterAndCapLegalLogits(roll_res.logits, curr_acts, logit_cap);
                Action roll_act = SamplePolicyAction(&cf_policy_rng[curr_p], roll_res.logits, curr_acts).first;
                roll_state->ApplyAction(roll_act);
              }
              target_return = static_cast<float>(roll_state->Returns()[current_player]) / 4.0f;
            }
          } else {
            succ_val = static_cast<float>(succ_state->Returns()[current_player]) / 4.0f;
            target_return = succ_val;
            valid = true;
          }

          if (valid) {
            if (absl::GetFlag(FLAGS_train_value_only)) {
              if (!prohibited_histories.empty() && prohibited_histories.count(succ_state->History())) {
                SpielFatalError("CRITICAL PROTECTION BOUNDARY FAILURE: Attempted to process or train on a prohibited official-corpus state via counterfactual generation!");
              }
            }
            PpoTransition cf_trans;
            cf_trans.state = std::move(succ_obs);
            cf_trans.legal_actions = succ_state->IsTerminal() ? std::vector<Action>{} : succ_state->LegalActions();
            cf_trans.action = kInvalidAction;
            cf_trans.old_log_prob = 0.0f;
            cf_trans.reward = 0.0f;
            cf_trans.value = succ_val;
            cf_trans.advantage = target_return - succ_val;
            cf_trans.return_value = target_return;
            cf_trans.player_id = current_player;
            cf_trans.episode_id = episode_id;

            CounterfactualPending pending;
            pending.replay_weight =
                absl::GetFlag(FLAGS_counterfactual_replay_weight);
            pending.transition = std::move(cf_trans);
            pending_cf.push_back(std::move(pending));
            ++counterfactual_samples;
          }
        }
      }

      if (dune_state != nullptr) {
        pre_action_phase = dune_state->phase();
        if (current_player >= 0 && current_player < game.NumPlayers()) {
          pre_combat_strength[current_player] =
              dune_imperium::CombatStrength(*dune_state, current_player);
        }
      }

      if (vrpo_episode != nullptr) {
        if (dune_state == nullptr) {
          SpielFatalError("VRPO capture requires DuneImperiumState");
        }
        VrpoCapturedRow captured;
        captured.episode_id = episode_id;
        captured.global_row_index =
            static_cast<int64_t>(vrpo_episode->rows.size());
        captured.actor = current_player;
        captured.actor_observation = obs;
        captured.central_tensor =
            dune_state->VrpoCentralCriticTensor(current_player);
        captured.central_schema_sha256 =
            dune_imperium::kVrpoCentralCriticTensorSchemaSha256;
        captured.legal_actions = actions;
        captured.chosen_index =
            static_cast<int>(policy_sample.chosen_index);
        captured.chosen_action = action;
        captured.legal_behavior_probabilities =
            std::move(*vrpo_legal_probabilities);
        vrpo_episode->rows.push_back(std::move(captured));
      }

      state->ApplyAction(action);
      total_env_steps->fetch_add(1, std::memory_order_relaxed);

      if (dune_state != nullptr) {
        for (int p = 0; p < game.NumPlayers(); ++p) {
          if (!had_swordmaster[p] && dune_state->HasSwordmaster(p)) {
            had_swordmaster[p] = true;
            int purchase_round = dune_state->GetCurrentRound();
            int leader = dune_state->PlayerLeader(p);
            local_stats->sm_acquisitions_by_seat[p]++;
            if (leader >= 0 && leader < 14) {
              local_stats->sm_acquisitions_by_leader[leader]++;
            }
            if (purchase_round >= 1 && purchase_round <= 10) {
              local_stats->sm_acquisitions_by_round[purchase_round]++;
            }
          }
        }
      }

      PpoTransition transition;
      transition.state = std::move(obs);
      transition.legal_actions = std::move(actions);
      transition.action = action;
      transition.old_log_prob = old_log_prob;
      transition.reward = 0.0f;
      transition.value = result.value;
      transition.advantage = 0.0f;
      transition.return_value = 0.0f;
      transition.player_id = current_player;
      transition.episode_id = episode_id;
      transition.behavior_legal_log_probs =
          std::move(parity_behavior_log_probs);
      transition.behavior_raw_legal_logits =
          std::move(parity_raw_legal_logits);
      transition.decision_role = parity_decision_role;
      transition.behavior_physical_batch_id = result.physical_batch_id;
      transition.behavior_physical_batch_size = result.physical_batch_size;
      transition.behavior_physical_batch_row = result.physical_batch_row;
      trajectory->push_back(std::move(transition));

      if (current_player >= 0 && current_player < game.NumPlayers()) {
        last_transition_index[current_player] = static_cast<int>(trajectory->size() - 1);
      }

      if (dune_state != nullptr) {
        if (current_player >= 0 && current_player < game.NumPlayers() &&
            action != dune_imperium::kActionCombatPass) {
          int post_strength = dune_imperium::CombatStrength(*dune_state, current_player);
          int strength_delta = post_strength - pre_combat_strength[current_player];
          if (strength_delta > 0) {
            combat_accumulator.RecordDeployment(current_player, trajectory->size() - 1, strength_delta);
          }
        }

        // Specimen-exchange anti-breadcrumb (Item 3): a small negative shaping on
        // the transition that takes a ConvertSpecimenToTroop action (IDs 741-752
        // -- 740 is an unused base constant and is never legal; see
        // dune_specimen_conversion.h), to discourage the over-used
        // specimen->troop breadcrumb behavior. Training-only (never eval). Must
        // be set to the SAME value on both the pilot and control arms so the
        // search-distillation contrast stays pure.
        //
        // PWO-5 gate 2 items (b)+(c): the range predicate and the subtraction
        // both moved into shared, unit-tested helpers. The sign is the point --
        // a POSITIVE penalty must DECREASE this reward.
        if (specimen_exchange_penalty != 0.0 &&
            current_player >= 0 && current_player < game.NumPlayers() &&
            dune_shaping::IsSpecimenConversionAction(action)) {
          int idx = last_transition_index[current_player];
          if (idx >= 0 && idx < static_cast<int>(trajectory->size())) {
            (*trajectory)[idx].reward = ApplySpecimenExchangeShaping(
                (*trajectory)[idx].reward, action, specimen_exchange_penalty,
                reward_lambda);
          }
        }

        for (int p = 0; p < game.NumPlayers(); ++p) {
          int new_tleilaxu = dune_state->GetTleilaxuTrackForTesting(p);
          int tleilaxu_delta = new_tleilaxu - current_tleilaxu[p];
          current_tleilaxu[p] = new_tleilaxu;
          if (tleilaxu_delta > 0 &&
              (tleilaxu_breadcrumb_weight > 0.0 ||
               tleilaxu_level7_breadcrumb_weight > 0.0)) {
            float tleilaxu_reward = 0.0f;
            int start_l = new_tleilaxu - tleilaxu_delta;
            for (int l = start_l + 1; l <= new_tleilaxu; ++l) {
              if (l >= 1 && l <= 6) {
                tleilaxu_reward += static_cast<float>(tleilaxu_breadcrumb_weight) * reward_lambda;
              } else if (l == 7) {
                tleilaxu_reward += static_cast<float>(tleilaxu_level7_breadcrumb_weight) * reward_lambda;
              }
            }
            int idx = last_transition_index[p];
            if (idx >= 0 && idx < static_cast<int>(trajectory->size())) {
              (*trajectory)[idx].reward += tleilaxu_reward;
            }
          }
        }

        for (int p = 0; p < game.NumPlayers(); ++p) {
          int raw_conflict_vp_delta = dune_state->ConflictVpDelta(p) - prev_conflict_vp[p];
          prev_conflict_vp[p] = dune_state->ConflictVpDelta(p);

          int raw_total_vp_delta = dune_state->GetPlayerVp(p) - prev_total_vp[p];
          prev_total_vp[p] = dune_state->GetPlayerVp(p);

          int raw_noncombat = raw_total_vp_delta - raw_conflict_vp_delta;

          local_stats->raw_conflict_vp += raw_conflict_vp_delta;
          local_stats->raw_noncombat_vp += raw_noncombat;
          local_stats->raw_total_vp += raw_total_vp_delta;

          float combat_shape = std::max(raw_conflict_vp_delta, 0)
                                * static_cast<float>(shaped_weight) * reward_lambda;
          float noncombat_shape = raw_noncombat
                                  * static_cast<float>(shaped_weight) * reward_lambda;

          if (noncombat_shape != 0.0f) {
            int idx = last_transition_index[p];
            if (idx >= 0 && idx < static_cast<int>(trajectory->size())) {
              (*trajectory)[idx].reward += noncombat_shape;
            }
          }

          if (combat_shape > 0.0f) {
            local_stats->conflict_vp_generated += combat_shape;
            int total_investment = combat_accumulator.GetTotalInvestment(p);
            if (total_investment > 0) {
              for (const auto& ev : combat_accumulator.GetEvents(p)) {
                if (ev.transition_index >= 0 && ev.transition_index < static_cast<int>(trajectory->size())) {
                  float attributed_portion = combat_shape * static_cast<float>(ev.strength_delta) /
                                             static_cast<float>(total_investment);
                  (*trajectory)[ev.transition_index].reward += attributed_portion;
                  local_stats->conflict_vp_attributed += attributed_portion;
                }
              }
            } else {
              local_stats->conflict_vp_unattributed += combat_shape;
            }
          }
        }

        if (pre_action_phase == dune_imperium::GamePhase::kCombat &&
            dune_state->phase() != dune_imperium::GamePhase::kCombat) {
          combat_accumulator.ClearAll();
        }
      }
    }

    std::vector<double> terminal_returns = state->Returns();
    if (vrpo_episode != nullptr) {
      VrpoSeatValues returns = {terminal_returns[0], terminal_returns[1],
                                terminal_returns[2], terminal_returns[3]};
      VrpoZeroShapingRewardConfig config;
      config.reward_scale = absl::GetFlag(FLAGS_reward_scale);
      config.gamma = absl::GetFlag(FLAGS_gamma);
      config.lambda = absl::GetFlag(FLAGS_gae_lambda);
      config.probability_tolerance =
          absl::GetFlag(FLAGS_vrpo_probability_tolerance);
      config.shaped_reward_weight =
          absl::GetFlag(FLAGS_shaped_reward_weight);
      config.tleilaxu_breadcrumb_weight =
          absl::GetFlag(FLAGS_tleilaxu_breadcrumb_weight);
      config.tleilaxu_level7_breadcrumb_weight =
          absl::GetFlag(FLAGS_tleilaxu_level7_breadcrumb_weight);
      config.specimen_exchange_penalty =
          absl::GetFlag(FLAGS_specimen_exchange_penalty);
      std::string finalize_error;
      if (!FinalizeVrpoZeroShapingEpisode(
              vrpo_episode, returns, config, &finalize_error)) {
        SpielFatalError("VRPO capture finalization failed: " +
                        finalize_error);
      }
    }
    double gamma = absl::GetFlag(FLAGS_train_value_only) ? 1.0 : absl::GetFlag(FLAGS_gamma);
    double gae_lambda = absl::GetFlag(FLAGS_train_value_only) ? 1.0 : absl::GetFlag(FLAGS_gae_lambda);
    float reward_scale = static_cast<float>(
        std::max(1e-6, absl::GetFlag(FLAGS_reward_scale)));
    std::vector<float> last_value(game.NumPlayers(), 0.0f);
    std::vector<float> last_gae(game.NumPlayers(), 0.0f);
    std::vector<bool> seen_last_action(game.NumPlayers(), false);

    for (auto it = trajectory->rbegin(); it != trajectory->rend(); ++it) {
      if (it->player_id < 0 || it->player_id >= game.NumPlayers()) continue;
      int p = it->player_id;
      float reward = 0.0f;
      if (absl::GetFlag(FLAGS_train_value_only)) {
        if (!seen_last_action[p]) {
          reward = static_cast<float>(terminal_returns[p]);
          seen_last_action[p] = true;
        }
      } else {
        reward = it->reward;
        if (!seen_last_action[p]) {
          reward += static_cast<float>(terminal_returns[p]);
          seen_last_action[p] = true;
        }
      }
      reward = std::clamp(reward / reward_scale, -1.0f, 1.0f);

      float delta = reward + static_cast<float>(gamma) * last_value[p] -
                    it->value;
      float advantage =
          delta + static_cast<float>(gamma * gae_lambda) * last_gae[p];
      it->advantage = advantage;
      it->return_value = advantage + it->value;
      last_value[p] = it->value;
      last_gae[p] = advantage;
    }

    for (auto& cf : pending_cf) {
      for (int repeat = 0; repeat < cf.replay_weight; ++repeat) {
        if (repeat + 1 == cf.replay_weight) {
          trajectory->push_back(std::move(cf.transition));
        } else {
          trajectory->push_back(cf.transition);
        }
      }
    }

    return game_length;
  }
}

void RolloutWorker(int thread_id, const Game* game,
                    std::shared_ptr<IGameEvaluator> evaluator, int64_t obs_size,
                    PpoRolloutBuffer* rollout_buffer,
                    std::atomic<bool>* stop_collection,
                    std::atomic<uint64_t>* total_env_steps,
                    std::vector<WorkerStats>* worker_stats,
                    std::atomic<uint64_t>* local_episode_id,
                    uint64_t start_episode_id,
                    int rollout_games,
                    float reward_lambda,
                    VrpoCapturedEpisodeBuffer* vrpo_buffer) {
  uint64_t master = absl::GetFlag(FLAGS_seed);
  WorkerStats local_stats;
  while (true) {
    if (rollout_games == 0 && stop_collection->load(std::memory_order_relaxed)) {
      break;
    }
    uint64_t episode_id = local_episode_id->fetch_add(1, std::memory_order_relaxed);
    if (rollout_games > 0 && episode_id >= start_episode_id + rollout_games) {
      break;
    }
    std::vector<PpoTransition> trajectory;
    VrpoCapturedEpisode vrpo_episode;
    int moves = PpoSimulation(master, episode_id, *game, evaluator, obs_size, &trajectory,
                              total_env_steps, reward_lambda, &local_stats,
                              vrpo_buffer != nullptr ? &vrpo_episode : nullptr);
    if (vrpo_buffer != nullptr) {
      std::string publish_error;
      if (!vrpo_buffer->PublishValidated(std::move(vrpo_episode),
                                         &publish_error)) {
        SpielFatalError("VRPO capture publication failed: " + publish_error);
      }
    }
    local_stats.games += 1;
    local_stats.moves += moves;
    size_t size = rollout_buffer->PushTrajectory(std::move(trajectory));
    if (rollout_games == 0) {
      if (size >= static_cast<size_t>(absl::GetFlag(FLAGS_rollout_transitions))) {
        stop_collection->store(true, std::memory_order_relaxed);
      }
    }
  }
  (*worker_stats)[thread_id] = local_stats;
}

struct CollectResult {
  std::vector<PpoTransition> rollout;
  uint64_t games = 0;
  uint64_t moves = 0;
  double elapsed_seconds = 0.0;

  uint64_t sm_acquisitions_by_seat[4] = {0};
  uint64_t sm_acquisitions_by_leader[14] = {0};
  uint64_t sm_acquisitions_by_round[11] = {0};

  double conflict_vp_generated = 0.0;
  double conflict_vp_attributed = 0.0;
  double conflict_vp_unattributed = 0.0;

  bool episode_ids_unique = true;
  double raw_conflict_vp = 0.0;
  double raw_noncombat_vp = 0.0;
  double raw_total_vp = 0.0;

  int64_t parity_decisions = 0;
  int64_t parity_full_width_ok = 0;
  int64_t parity_full_finite_ok = 0;
  int64_t parity_legal_ids_unique_in_range = 0;
  int64_t parity_chosen_once = 0;
  std::vector<std::string> parity_capture_errors;

  // Phase A canary: concatenation of every worker's per-decision max|z|, merged
  // in thread-index order. Only the multiset matters downstream (percentiles
  // sort), so the merge order does not affect the reported statistics.
  std::vector<float> precap_absz_all;
  std::vector<float> precap_absz_primary;
  std::vector<float> precap_absz_continuation;
  std::vector<float> precap_absz_purchase;

  // PWO-5 section 14.1b canaries, summed over workers. Summation is exact for
  // the integer counts; the double sums are order-dependent at the last ulp,
  // and the merge is in thread-index order, which is deterministic given a
  // thread count. That is the same guarantee every other double accumulator in
  // this struct carries, and it is why every bitwise gate runs at --threads=1.
  double canary_ne_sum = 0.0;
  int64_t canary_ne_n = 0;
  double canary_ne_sum_role[7] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  int64_t canary_ne_n_role[7] = {0, 0, 0, 0, 0, 0, 0};
  double canary_maxprob_sum = 0.0;
  int64_t canary_sat_ge_cap = 0;
  int64_t canary_sat_logits = 0;

  // Phase 18B: online auxiliary-search examples collected from the SAME frozen
  // snapshot as `rollout`, so they travel together and are consumed by the same
  // TrainPpoUpdate. Empty unless --online_search_collection.
  std::vector<open_spiel::SearchTrainingExample> aux_examples;
  open_spiel::OnlineSearchCollectionStats aux_stats;
};

// Phase A: folds the rollout-side pre-cap |z| canaries into the update's stats
// so every WriteDiagnostics site persists them. Called immediately after
// episode_ids_unique is copied, which is the existing seam for carrying
// collection-side facts into the diagnostics row.
// PWO-5 section 14.1c: fold the rollout-side canaries into the update's stats,
// and CHECK the partition identity rather than assuming it.
//
// Two halves, both registered:
//   Sum_r n_r == norm_entropy_n            EXACTLY (integer equality) -- the
//     stronger half; it proves no qualifying decision was dropped or
//     double-counted.
//   |Sum_r (n_r * NE_r) / Sum_r n_r - NE_global| <= 1e-9  -- float slack only.
//
// A violation is a gate-3 failure, not a rounding artifact, so it is fatal here
// rather than reported.
void AttachCanaryStats(open_spiel::PpoUpdateStats* stats,
                       const CollectResult& collect) {
  stats->norm_entropy_n = collect.canary_ne_n;
  stats->norm_entropy =
      collect.canary_ne_n > 0 ? collect.canary_ne_sum / collect.canary_ne_n : 0.0;
  stats->max_action_prob =
      collect.canary_ne_n > 0 ? collect.canary_maxprob_sum / collect.canary_ne_n
                              : 0.0;
  stats->frac_legal_absz_ge_cap_n = collect.canary_sat_logits;
  stats->frac_legal_absz_ge_cap =
      collect.canary_sat_logits > 0
          ? static_cast<double>(collect.canary_sat_ge_cap) /
                static_cast<double>(collect.canary_sat_logits)
          : 0.0;

  int64_t role_n_total = 0;
  double weighted_sum = 0.0;
  for (int r = 0; r < 7; ++r) {
    stats->norm_entropy_n_role[r] = collect.canary_ne_n_role[r];
    // Zero-support semantics: mean is 0.0 when the count is 0, and 0.0 there
    // means "no support this update", never "entropy collapsed to zero".
    stats->norm_entropy_role[r] =
        collect.canary_ne_n_role[r] > 0
            ? collect.canary_ne_sum_role[r] / collect.canary_ne_n_role[r]
            : 0.0;
    role_n_total += collect.canary_ne_n_role[r];
    weighted_sum += collect.canary_ne_sum_role[r];
  }

  if (role_n_total != collect.canary_ne_n) {
    SpielFatalError(absl::StrFormat(
        "PWO-5 section 14.1c partition identity FAILED (count half): "
        "sum of role denominators = %lld but norm_entropy_n = %lld. A "
        "qualifying decision was dropped or double-counted.",
        static_cast<long long>(role_n_total),
        static_cast<long long>(collect.canary_ne_n)));
  }
  if (role_n_total > 0) {
    const double reconstructed = weighted_sum / role_n_total;
    if (std::abs(reconstructed - stats->norm_entropy) > 1e-9) {
      SpielFatalError(absl::StrFormat(
          "PWO-5 section 14.1c partition identity FAILED (mean half): "
          "row-count-weighted role mean %.17g vs norm_entropy %.17g, "
          "difference %.3g exceeds 1e-9.",
          reconstructed, stats->norm_entropy,
          std::abs(reconstructed - stats->norm_entropy)));
    }
  }
  // The seventh role must be identically empty at this site (section 14.1c's
  // proof). Checked, because the proof depends on the measurement site and a
  // future move of that site would silently invalidate it.
  if (collect.canary_ne_n_role[static_cast<int>(
          DuneDecisionRole::kForcedOrBookkeeping)] != 0) {
    SpielFatalError(
        "PWO-5 section 14.1c: kForcedOrBookkeeping carried a canary decision. "
        "That role is provably empty at the registered rollout measurement "
        "site (chance nodes continue before classification, and the population "
        "requires n_legal >= 2), so a nonzero count means the canaries are no "
        "longer being measured where section 14.1b froze them.");
  }
}

void AttachPrecapAbszStats(open_spiel::PpoUpdateStats* stats,
                           const CollectResult& collect) {
  stats->precap_absz_all =
      open_spiel::SummarizePrecapAbsz(collect.precap_absz_all);
  stats->precap_absz_primary =
      open_spiel::SummarizePrecapAbsz(collect.precap_absz_primary);
  stats->precap_absz_continuation =
      open_spiel::SummarizePrecapAbsz(collect.precap_absz_continuation);
  stats->precap_absz_purchase =
      open_spiel::SummarizePrecapAbsz(collect.precap_absz_purchase);
}



CollectResult CollectRollout(const Game* game,
                             std::shared_ptr<IGameEvaluator> evaluator,
                             int64_t obs_size,
                             std::atomic<uint64_t>* total_env_steps,
                             int num_threads,
                             std::atomic<uint64_t>* next_episode_id,
                             int rollout_games,
                             float reward_lambda,
                             VrpoCapturedEpisodeBuffer* vrpo_buffer = nullptr) {
  CollectResult result;
  PpoRolloutBuffer rollout_buffer;
  std::atomic<bool> stop_collection{false};
  std::vector<WorkerStats> worker_stats(num_threads);
  std::vector<std::thread> workers;
  workers.reserve(num_threads);

  uint64_t actual_start_ep = next_episode_id->load();
  std::atomic<uint64_t> local_episode_id{actual_start_ep};

  auto start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < num_threads; ++i) {
    workers.emplace_back(RolloutWorker, i, game, evaluator, obs_size,
                         &rollout_buffer, &stop_collection, total_env_steps,
                         &worker_stats, &local_episode_id, actual_start_ep,
                         rollout_games, reward_lambda, vrpo_buffer);
  }
  for (auto& worker : workers) worker.join();

  if (rollout_games > 0) {
    next_episode_id->store(actual_start_ep + rollout_games);
  } else {
    next_episode_id->store(local_episode_id.load());
  }
  auto end = std::chrono::high_resolution_clock::now();

  bool episode_ids_unique = true;
  result.rollout = rollout_buffer.TakeAll(&episode_ids_unique);
  result.episode_ids_unique = episode_ids_unique;

  for (const auto& stats : worker_stats) {
    result.games += stats.games;
    result.moves += stats.moves;

    for (int i = 0; i < 4; ++i) result.sm_acquisitions_by_seat[i] += stats.sm_acquisitions_by_seat[i];
    for (int i = 0; i < 14; ++i) result.sm_acquisitions_by_leader[i] += stats.sm_acquisitions_by_leader[i];
    for (int i = 0; i < 11; ++i) result.sm_acquisitions_by_round[i] += stats.sm_acquisitions_by_round[i];

    result.conflict_vp_generated += stats.conflict_vp_generated;
    result.conflict_vp_attributed += stats.conflict_vp_attributed;
    result.conflict_vp_unattributed += stats.conflict_vp_unattributed;

    result.raw_conflict_vp += stats.raw_conflict_vp;
    result.raw_noncombat_vp += stats.raw_noncombat_vp;
    result.raw_total_vp += stats.raw_total_vp;

    result.parity_decisions += stats.parity_decisions;
    result.parity_full_width_ok += stats.parity_full_width_ok;
    result.parity_full_finite_ok += stats.parity_full_finite_ok;
    result.parity_legal_ids_unique_in_range +=
        stats.parity_legal_ids_unique_in_range;
    result.parity_chosen_once += stats.parity_chosen_once;
    result.parity_capture_errors.insert(
        result.parity_capture_errors.end(),
        stats.parity_capture_errors.begin(),
        stats.parity_capture_errors.end());

    // Phase A canary: concatenate in thread-index order. Downstream only sorts,
    // so order is irrelevant to the reported percentiles.
    result.precap_absz_all.insert(result.precap_absz_all.end(),
                                  stats.precap_absz_all.begin(),
                                  stats.precap_absz_all.end());
    result.precap_absz_primary.insert(result.precap_absz_primary.end(),
                                      stats.precap_absz_primary.begin(),
                                      stats.precap_absz_primary.end());
    result.precap_absz_continuation.insert(result.precap_absz_continuation.end(),
                                           stats.precap_absz_continuation.begin(),
                                           stats.precap_absz_continuation.end());
    result.precap_absz_purchase.insert(result.precap_absz_purchase.end(),
                                       stats.precap_absz_purchase.begin(),
                                       stats.precap_absz_purchase.end());

    // PWO-5 section 14.1b canaries.
    result.canary_ne_sum += stats.canary_ne_sum;
    result.canary_ne_n += stats.canary_ne_n;
    for (int r = 0; r < 7; ++r) {
      result.canary_ne_sum_role[r] += stats.canary_ne_sum_role[r];
      result.canary_ne_n_role[r] += stats.canary_ne_n_role[r];
    }
    result.canary_maxprob_sum += stats.canary_maxprob_sum;
    result.canary_sat_ge_cap += stats.canary_sat_ge_cap;
    result.canary_sat_logits += stats.canary_sat_logits;
  }

  // Verify shaped reward conservation invariant
  if (std::abs(result.conflict_vp_generated - (result.conflict_vp_attributed + result.conflict_vp_unattributed)) > 1e-4) {
    open_spiel::SpielFatalError(absl::StrFormat("Shaped reward conservation violated: generated = %f, attributed = %f, unattributed = %f",
                                                result.conflict_vp_generated, result.conflict_vp_attributed, result.conflict_vp_unattributed));
  }

  // Verify signed raw VP conservation invariant
  if (std::abs(result.raw_conflict_vp + result.raw_noncombat_vp - result.raw_total_vp) > 1e-4) {
    open_spiel::SpielFatalError(absl::StrFormat("Signed raw VP conservation violated: raw_conflict = %f, raw_noncombat = %f, raw_total = %f",
                                                result.raw_conflict_vp, result.raw_noncombat_vp, result.raw_total_vp));
  }
  result.elapsed_seconds =
      std::chrono::duration<double>(end - start).count();
  return result;
}

std::string PpoParityRowIdentitySha256(
    int64_t row_index, const PpoTransition& transition) {
  std::stringstream bytes;
  bytes.write(reinterpret_cast<const char*>(&row_index), sizeof(row_index));
  bytes.write(reinterpret_cast<const char*>(&transition.episode_id),
              sizeof(transition.episode_id));
  bytes.write(reinterpret_cast<const char*>(&transition.player_id),
              sizeof(transition.player_id));
  bytes.write(reinterpret_cast<const char*>(&transition.action),
              sizeof(transition.action));
  bytes.write(reinterpret_cast<const char*>(&transition.decision_role),
              sizeof(transition.decision_role));
  bytes.write(reinterpret_cast<const char*>(&transition.advantage),
              sizeof(transition.advantage));
  bytes.write(reinterpret_cast<const char*>(&transition.old_log_prob),
              sizeof(transition.old_log_prob));
  const int64_t legal_count =
      static_cast<int64_t>(transition.legal_actions.size());
  bytes.write(reinterpret_cast<const char*>(&legal_count), sizeof(legal_count));
  for (size_t i = 0; i < transition.legal_actions.size(); ++i) {
    bytes.write(reinterpret_cast<const char*>(&transition.legal_actions[i]),
                sizeof(transition.legal_actions[i]));
    const float raw = i < transition.behavior_raw_legal_logits.size()
                          ? transition.behavior_raw_legal_logits[i]
                          : std::numeric_limits<float>::quiet_NaN();
    const float behavior = i < transition.behavior_legal_log_probs.size()
                               ? transition.behavior_legal_log_probs[i]
                               : std::numeric_limits<float>::quiet_NaN();
    bytes.write(reinterpret_cast<const char*>(&raw), sizeof(raw));
    bytes.write(reinterpret_cast<const char*>(&behavior), sizeof(behavior));
  }
  return ComputeStringSHA256(bytes.str());
}

std::string PpoParitySharedIdentitySha256(
    const std::vector<PpoTransition>& batch) {
  std::string payload;
  for (int64_t i = 0; i < static_cast<int64_t>(batch.size()); ++i) {
    payload.append(PpoParityRowIdentitySha256(i, batch[i]));
    payload.push_back('\n');
  }
  return ComputeStringSHA256(payload);
}

std::string PpoParityCapturedRawSha256(
    const std::vector<PpoTransition>& batch) {
  std::stringstream bytes;
  for (const auto& transition : batch) {
    for (float value : transition.behavior_raw_legal_logits) {
      bytes.write(reinterpret_cast<const char*>(&value), sizeof(value));
    }
  }
  return ComputeStringSHA256(bytes.str());
}

std::string PpoParityCapturedPolicySha256(
    const std::vector<PpoTransition>& batch) {
  std::stringstream bytes;
  for (const auto& transition : batch) {
    for (float value : transition.behavior_legal_log_probs) {
      bytes.write(reinterpret_cast<const char*>(&value), sizeof(value));
    }
  }
  return ComputeStringSHA256(bytes.str());
}

std::vector<PpoNumericalParityRow> ReplayPpoCpuIntegrityCell(
    const std::vector<PpoTransition>& batch, float logit_cap,
    bool require_bf16_grid,
    std::vector<std::string>* schema_errors,
    std::string* row_evidence_sha256, int64_t* raw_values_total,
    int64_t* raw_values_bf16_exact, int64_t* vector_width_rows_ok,
    int64_t* behavior_vector_exact_rows, int64_t* chosen_scalar_exact_rows) {
  if (raw_values_total != nullptr) *raw_values_total = 0;
  if (raw_values_bf16_exact != nullptr) *raw_values_bf16_exact = 0;
  if (vector_width_rows_ok != nullptr) *vector_width_rows_ok = 0;
  if (behavior_vector_exact_rows != nullptr) *behavior_vector_exact_rows = 0;
  if (chosen_scalar_exact_rows != nullptr) *chosen_scalar_exact_rows = 0;
  std::vector<PpoNumericalParityRow> rows;
  rows.reserve(batch.size());
  std::stringstream evidence;
  evidence << "dune_raw_ppo_numerical_parity_v3";
  evidence.put('\0');
  evidence << "rollout_cpu_recompute_integrity";
  evidence.put('\0');

  for (int64_t row_index = 0;
       row_index < static_cast<int64_t>(batch.size()); ++row_index) {
    const PpoTransition& transition = batch[row_index];
    const std::string identity =
        PpoParityRowIdentitySha256(row_index, transition);
    bool row_valid = true;
    auto error = [&](const std::string& message) {
      schema_errors->push_back(absl::StrFormat(
          "row %lld: %s", static_cast<long long>(row_index), message));
      row_valid = false;
    };
    const size_t legal_count = transition.legal_actions.size();
    if (legal_count == 0 ||
        transition.behavior_raw_legal_logits.size() != legal_count ||
        transition.behavior_legal_log_probs.size() != legal_count) {
      error("legal/raw/logprob vector widths differ or are empty");
    } else if (vector_width_rows_ok != nullptr) {
      ++*vector_width_rows_ok;
    }
    if (!PpoParityCapturedRawLogitsValid(
            transition.behavior_raw_legal_logits, require_bf16_grid,
            raw_values_total, raw_values_bf16_exact)) {
      error(require_bf16_grid
                ? "captured raw legal logits are not finite BF16-grid values"
                : "captured raw legal logits are not finite");
    }

    std::vector<float> recomputed;
    std::string recompute_error;
    if (!RecomputePpoBehaviorLegalLogProbs(
            transition.behavior_raw_legal_logits, logit_cap, &recomputed,
            &recompute_error)) {
      error("CPU behavior recompute failed: " + recompute_error);
    }
    bool behavior_exact = recomputed.size() ==
                          transition.behavior_legal_log_probs.size();
    if (behavior_exact) {
      for (size_t i = 0; i < recomputed.size(); ++i) {
        if (PpoParityFloatBits(recomputed[i]) != PpoParityFloatBits(
                transition.behavior_legal_log_probs[i])) {
          behavior_exact = false;
          break;
        }
      }
    }
    if (behavior_exact) {
      if (behavior_vector_exact_rows != nullptr) {
        ++*behavior_vector_exact_rows;
      }
    } else {
      error("CPU recompute does not bit-match stored behavior vector");
    }
    int chosen_index = -1;
    int chosen_count = 0;
    for (size_t i = 0; i < legal_count; ++i) {
      if (transition.legal_actions[i] == transition.action) {
        chosen_index = static_cast<int>(i);
        ++chosen_count;
      }
    }
    bool chosen_exact = chosen_count == 1 && chosen_index >= 0 &&
                        chosen_index < static_cast<int>(
                            transition.behavior_legal_log_probs.size()) &&
                        PpoParityFloatBits(transition.old_log_prob) ==
                            PpoParityFloatBits(
                                transition.behavior_legal_log_probs[
                                    chosen_index]);
    if (chosen_exact) {
      if (chosen_scalar_exact_rows != nullptr) ++*chosen_scalar_exact_rows;
    } else {
      error("chosen scalar does not bit-match stored behavior vector");
    }

    PpoNumericalParityInput input;
    input.legal_count = static_cast<int>(legal_count);
    input.decision_role = transition.decision_role;
    input.advantage = transition.advantage;
    input.chosen_index = chosen_index;
    input.stored_chosen_log_prob = transition.old_log_prob;
    for (float value : transition.behavior_legal_log_probs) {
      input.old_log_probs.push_back(value);
    }
    for (float value : recomputed) input.new_log_probs.push_back(value);
    PpoNumericalParityRow row;
    std::string parity_error;
    if (!ComputePpoNumericalParityRow(input, &row, &parity_error)) {
      error("parity row failed: " + parity_error);
    }
    if (!row_valid) {
      row.schema_errors = std::max<int64_t>(1, row.schema_errors);
      row.nonfinite_values = std::max<int64_t>(1, row.nonfinite_values);
    }
    row.row_identity_sha256 = identity;
    rows.push_back(row);
    evidence << identity;
    evidence.put('\0');
    for (float value : recomputed) {
      evidence.write(reinterpret_cast<const char*>(&value), sizeof(value));
    }
  }
  if (row_evidence_sha256 != nullptr) {
    *row_evidence_sha256 = ComputeStringSHA256(evidence.str());
  }
  return rows;
}

std::vector<PpoNumericalParityRow> ReplayPpoNumericalParityCell(
    const std::shared_ptr<SharedDunePolicyValueNetImpl>& model,
    const std::vector<PpoTransition>& batch, int64_t obs_size,
    int64_t action_dim, torch::Device device, bool model_train_mode,
    bool learner_autocast,
    const std::string& cell_label,
    std::vector<std::string>* schema_errors,
    std::string* row_evidence_sha256, int64_t* model_forward_calls,
    int64_t* dense_mask_rows_ok, std::string* raw_logit_sha256,
    std::string* policy_sha256) {
  std::vector<PpoNumericalParityRow> rows;
  rows.reserve(batch.size());
  std::stringstream evidence;
  std::stringstream raw_bytes, policy_bytes;
  evidence << "dune_raw_ppo_numerical_parity_v3";
  evidence.put('\0');
  evidence << cell_label;
  evidence.put('\0');
  if (model_forward_calls != nullptr) *model_forward_calls = 0;
  if (dense_mask_rows_ok != nullptr) *dense_mask_rows_ok = 0;
  const int64_t chunk_size = std::max<int64_t>(
      1, std::min<int64_t>(absl::GetFlag(FLAGS_ppo_minibatch_size),
                           static_cast<int64_t>(batch.size())));
  const float logit_cap = static_cast<float>(absl::GetFlag(FLAGS_logit_cap));
  const bool was_training = model->is_training();
  if (model_train_mode) {
    model->train();  // exact TrainPpoUpdate module mode
  } else {
    model->eval();   // v1 descriptive-control convention
  }
  torch::NoGradGuard no_grad;

  for (int64_t start = 0; start < static_cast<int64_t>(batch.size());
       start += chunk_size) {
    const int64_t count = std::min<int64_t>(
        chunk_size, static_cast<int64_t>(batch.size()) - start);
    torch::Tensor states_cpu = torch::zeros(
        {count, obs_size}, torch::TensorOptions().dtype(torch::kFloat32));
    torch::Tensor masks_cpu = torch::zeros(
        {count, action_dim}, torch::TensorOptions().dtype(torch::kBool));
    float* state_ptr = states_cpu.data_ptr<float>();
    bool* mask_ptr = masks_cpu.data_ptr<bool>();
    std::vector<bool> packed_row_valid(count, true);
    for (int64_t i = 0; i < count; ++i) {
      const PpoTransition& transition = batch[start + i];
      if (static_cast<int64_t>(transition.state.size()) != obs_size) {
        schema_errors->push_back(absl::StrFormat(
            "row %lld: observation width %lld != %lld",
            static_cast<long long>(start + i),
            static_cast<long long>(transition.state.size()),
            static_cast<long long>(obs_size)));
        packed_row_valid[i] = false;
        continue;
      }
      std::memcpy(state_ptr + i * obs_size, transition.state.data(),
                  obs_size * sizeof(float));
      for (Action action : transition.legal_actions) {
        if (action >= 0 && action < action_dim) {
          mask_ptr[i * action_dim + action] = true;
        }
      }
    }

    const torch::Tensor mask_popcounts =
        masks_cpu.sum(1).to(torch::kCPU).to(torch::kInt64).contiguous();
    const int64_t* mask_popcount_ptr = mask_popcounts.data_ptr<int64_t>();
    for (int64_t i = 0; i < count; ++i) {
      const PpoTransition& transition = batch[start + i];
      if (mask_popcount_ptr[i] ==
          static_cast<int64_t>(transition.legal_actions.size())) {
        if (dense_mask_rows_ok != nullptr) ++*dense_mask_rows_ok;
      } else {
        packed_row_valid[i] = false;
        schema_errors->push_back(absl::StrFormat(
            "row %lld: dense mask popcount %lld != legal count %lld",
            static_cast<long long>(start + i),
            static_cast<long long>(mask_popcount_ptr[i]),
            static_cast<long long>(transition.legal_actions.size())));
      }
    }

    const torch::Tensor states = states_cpu.to(device);
    const torch::Tensor masks = masks_cpu.to(device);
    if (states_cpu.scalar_type() != torch::kFloat32 ||
        states.scalar_type() != torch::kFloat32) {
      schema_errors->push_back("cell input/pre-forward dtype is not Float32");
    }
    torch::Tensor replay_raw_logits, replay_log_probs;
    auto replay_policy_path = [&]() {
      if (model_forward_calls != nullptr) ++*model_forward_calls;
      const auto outputs = model->forward(states);
      replay_raw_logits = outputs.logits;
      const torch::Tensor centered =
          CenterAndCapLogitsTensor(replay_raw_logits, masks, logit_cap);
      const torch::Tensor masked_logits =
          centered.masked_fill(masks.logical_not(), -1e9f);
      replay_log_probs = torch::log_softmax(masked_logits, -1);
    };
    if (learner_autocast) {
      // Exact production PPO policy path: TrainPpoUpdate wraps compute_loss --
      // including forward, CenterAndCapLogitsTensor, masking and log_softmax --
      // in this CUDA BF16 AutocastGuard when --train_amp=true.
      AutocastGuard autocast_guard(c10::DeviceType::CUDA, true);
      replay_policy_path();
    } else {
      // Descriptive cross-precision control retained from v1.
      replay_policy_path();
    }
    const torch::Tensor log_probs_cpu =
        replay_log_probs.to(torch::kCPU).to(torch::kFloat32).contiguous();
    const torch::Tensor raw_logits_cpu =
        replay_raw_logits.to(torch::kCPU).to(torch::kFloat32).contiguous();
    const float* new_log_probs = log_probs_cpu.data_ptr<float>();
    const float* raw_logits = raw_logits_cpu.data_ptr<float>();

    for (int64_t i = 0; i < count; ++i) {
      const int64_t row_index = start + i;
      const PpoTransition& transition = batch[row_index];
      const std::string row_identity =
          PpoParityRowIdentitySha256(row_index, transition);
      PpoNumericalParityInput input;
      input.legal_count = static_cast<int>(transition.legal_actions.size());
      input.decision_role = transition.decision_role;
      input.advantage = transition.advantage;
      input.stored_chosen_log_prob = transition.old_log_prob;
      input.old_log_probs.reserve(input.legal_count);
      input.new_log_probs.reserve(input.legal_count);
      for (int j = 0; j < input.legal_count; ++j) {
        const Action action = transition.legal_actions[j];
        input.old_log_probs.push_back(
            j < static_cast<int>(transition.behavior_legal_log_probs.size())
                ? transition.behavior_legal_log_probs[j]
                : std::numeric_limits<double>::quiet_NaN());
        input.new_log_probs.push_back(
            (action >= 0 && action < action_dim)
                ? new_log_probs[i * action_dim + action]
                : std::numeric_limits<double>::quiet_NaN());
        if (action == transition.action) input.chosen_index = j;
        const float replay_raw =
            (action >= 0 && action < action_dim)
                ? raw_logits[i * action_dim + action]
                : std::numeric_limits<float>::quiet_NaN();
        raw_bytes.write(reinterpret_cast<const char*>(&replay_raw),
                        sizeof(replay_raw));
        const float replay_policy = static_cast<float>(
            input.new_log_probs.back());
        policy_bytes.write(reinterpret_cast<const char*>(&replay_policy),
                           sizeof(replay_policy));
      }
      // Hash the exact row identity and BOTH aligned legal distributions. The
      // artifact's summaries can therefore be tied back to the measured row
      // evidence without embedding a many-megabyte raw dump in JSON.
      evidence.write(reinterpret_cast<const char*>(&row_index),
                     sizeof(row_index));
      evidence.write(reinterpret_cast<const char*>(&transition.episode_id),
                     sizeof(transition.episode_id));
      evidence.write(reinterpret_cast<const char*>(&transition.player_id),
                     sizeof(transition.player_id));
      evidence.write(reinterpret_cast<const char*>(&transition.action),
                     sizeof(transition.action));
      evidence.write(reinterpret_cast<const char*>(&transition.decision_role),
                     sizeof(transition.decision_role));
      evidence.write(reinterpret_cast<const char*>(&transition.advantage),
                     sizeof(transition.advantage));
      evidence.write(reinterpret_cast<const char*>(&transition.old_log_prob),
                     sizeof(transition.old_log_prob));
      for (int j = 0; j < input.legal_count; ++j) {
        evidence.write(
            reinterpret_cast<const char*>(&transition.legal_actions[j]),
            sizeof(transition.legal_actions[j]));
        evidence.write(reinterpret_cast<const char*>(&input.old_log_probs[j]),
                       sizeof(input.old_log_probs[j]));
        evidence.write(reinterpret_cast<const char*>(&input.new_log_probs[j]),
                       sizeof(input.new_log_probs[j]));
        const float raw =
            j < static_cast<int>(transition.behavior_raw_legal_logits.size())
                ? transition.behavior_raw_legal_logits[j]
                : std::numeric_limits<float>::quiet_NaN();
        evidence.write(reinterpret_cast<const char*>(&raw), sizeof(raw));
      }
      if (!packed_row_valid[i]) {
        PpoNumericalParityRow bad;
        bad.legal_count = static_cast<int>(transition.legal_actions.size());
        bad.decision_role = transition.decision_role;
        bad.advantage = transition.advantage;
        bad.schema_errors = 1;
        bad.nonfinite_values = 1;
        bad.row_identity_sha256 = row_identity;
        rows.push_back(bad);
        continue;
      }
      PpoNumericalParityRow row;
      std::string error;
      if (!ComputePpoNumericalParityRow(input, &row, &error)) {
        schema_errors->push_back(absl::StrFormat(
            "row %lld: %s", static_cast<long long>(row_index), error));
      }
      row.row_identity_sha256 = row_identity;
      // V2 never drops a bad row. Schema-error rows remain in every partition
      // and are excluded only from finite metric denominators.
      rows.push_back(row);
    }
  }
  if (was_training) {
    model->train();
  } else {
    model->eval();
  }
  if (row_evidence_sha256 != nullptr) {
    *row_evidence_sha256 = ComputeStringSHA256(evidence.str());
  }
  if (raw_logit_sha256 != nullptr) {
    *raw_logit_sha256 = ComputeStringSHA256(raw_bytes.str());
  }
  if (policy_sha256 != nullptr) {
    *policy_sha256 = ComputeStringSHA256(policy_bytes.str());
  }
  return rows;
}

std::vector<PpoNumericalParityRow> ReplayPpoCapturedLogitsBf16Cell(
    const std::vector<PpoTransition>& batch, int64_t action_dim,
    torch::Device device, float logit_cap,
    std::vector<std::string>* schema_errors,
    std::string* row_evidence_sha256, int64_t* model_forward_calls,
    int64_t* dense_mask_rows_ok) {
  if (model_forward_calls != nullptr) *model_forward_calls = 0;
  if (dense_mask_rows_ok != nullptr) *dense_mask_rows_ok = 0;
  std::vector<PpoNumericalParityRow> rows;
  rows.reserve(batch.size());
  std::stringstream evidence;
  evidence << "dune_raw_ppo_numerical_parity_v3";
  evidence.put('\0');
  evidence << "captured_logits_dense_cuda_bf16_postprocess";
  evidence.put('\0');
  const int64_t chunk_size = std::max<int64_t>(
      1, std::min<int64_t>(absl::GetFlag(FLAGS_ppo_minibatch_size),
                           static_cast<int64_t>(batch.size())));

  for (int64_t start = 0; start < static_cast<int64_t>(batch.size());
       start += chunk_size) {
    const int64_t count = std::min<int64_t>(
        chunk_size, static_cast<int64_t>(batch.size()) - start);
    torch::Tensor dense_cpu = torch::zeros(
        {count, action_dim}, torch::TensorOptions().dtype(torch::kFloat32));
    torch::Tensor masks_cpu = torch::zeros(
        {count, action_dim}, torch::TensorOptions().dtype(torch::kBool));
    float* dense = dense_cpu.data_ptr<float>();
    bool* mask = masks_cpu.data_ptr<bool>();
    std::vector<bool> packed_row_valid(count, true);
    for (int64_t i = 0; i < count; ++i) {
      const PpoTransition& transition = batch[start + i];
      if (transition.behavior_raw_legal_logits.size() !=
          transition.legal_actions.size()) {
        packed_row_valid[i] = false;
        schema_errors->push_back(absl::StrFormat(
            "row %lld: raw/legal vector width mismatch",
            static_cast<long long>(start + i)));
      }
      std::unordered_set<Action> unique;
      for (size_t j = 0; j < transition.legal_actions.size(); ++j) {
        const Action action = transition.legal_actions[j];
        const bool valid = action >= 0 && action < action_dim &&
                           unique.insert(action).second &&
                           j < transition.behavior_raw_legal_logits.size() &&
                           std::isfinite(
                               transition.behavior_raw_legal_logits[j]);
        if (!valid) {
          packed_row_valid[i] = false;
          continue;
        }
        dense[i * action_dim + action] =
            transition.behavior_raw_legal_logits[j];
        mask[i * action_dim + action] = true;
      }
    }
    const torch::Tensor mask_popcounts =
        masks_cpu.sum(1).to(torch::kCPU).to(torch::kInt64).contiguous();
    const int64_t* popcount = mask_popcounts.data_ptr<int64_t>();
    for (int64_t i = 0; i < count; ++i) {
      if (popcount[i] == static_cast<int64_t>(
                             batch[start + i].legal_actions.size())) {
        if (dense_mask_rows_ok != nullptr) ++*dense_mask_rows_ok;
      } else {
        packed_row_valid[i] = false;
        schema_errors->push_back(absl::StrFormat(
            "row %lld: dense mask popcount %lld != legal count %lld",
            static_cast<long long>(start + i),
            static_cast<long long>(popcount[i]),
            static_cast<long long>(
                batch[start + i].legal_actions.size())));
      }
    }

    const torch::Tensor dense_bf16 =
        dense_cpu.to(device).to(torch::kBFloat16);
    const torch::Tensor masks = masks_cpu.to(device);
    torch::Tensor replay_log_probs;
    {
      AutocastGuard autocast_guard(c10::DeviceType::CUDA, true);
      const torch::Tensor centered =
          CenterAndCapLogitsTensor(dense_bf16, masks, logit_cap);
      const torch::Tensor masked_logits =
          centered.masked_fill(masks.logical_not(), -1e9f);
      replay_log_probs = torch::log_softmax(masked_logits, -1);
    }
    const torch::Tensor log_probs_cpu =
        replay_log_probs.to(torch::kCPU).to(torch::kFloat32).contiguous();
    const float* new_log_probs = log_probs_cpu.data_ptr<float>();

    for (int64_t i = 0; i < count; ++i) {
      const int64_t row_index = start + i;
      const PpoTransition& transition = batch[row_index];
      const std::string identity =
          PpoParityRowIdentitySha256(row_index, transition);
      PpoNumericalParityInput input;
      input.legal_count = static_cast<int>(transition.legal_actions.size());
      input.decision_role = transition.decision_role;
      input.advantage = transition.advantage;
      input.stored_chosen_log_prob = transition.old_log_prob;
      for (int j = 0; j < input.legal_count; ++j) {
        const Action action = transition.legal_actions[j];
        input.old_log_probs.push_back(
            j < static_cast<int>(transition.behavior_legal_log_probs.size())
                ? transition.behavior_legal_log_probs[j]
                : std::numeric_limits<double>::quiet_NaN());
        input.new_log_probs.push_back(
            (action >= 0 && action < action_dim)
                ? new_log_probs[i * action_dim + action]
                : std::numeric_limits<double>::quiet_NaN());
        if (action == transition.action) input.chosen_index = j;
      }
      PpoNumericalParityRow row;
      std::string error;
      if (!ComputePpoNumericalParityRow(input, &row, &error)) {
        schema_errors->push_back(absl::StrFormat(
            "row %lld: %s", static_cast<long long>(row_index), error));
      }
      if (!packed_row_valid[i]) {
        row.schema_errors = std::max<int64_t>(1, row.schema_errors);
        row.nonfinite_values = std::max<int64_t>(1, row.nonfinite_values);
      }
      row.row_identity_sha256 = identity;
      rows.push_back(row);
      evidence << identity;
      evidence.put('\0');
      for (int j = 0; j < input.legal_count; ++j) {
        evidence.write(reinterpret_cast<const char*>(&input.new_log_probs[j]),
                       sizeof(input.new_log_probs[j]));
      }
    }
  }
  if (row_evidence_sha256 != nullptr) {
    *row_evidence_sha256 = ComputeStringSHA256(evidence.str());
  }
  return rows;
}

json::Object PpoNumericalParitySummaryJson(
    const PpoNumericalParitySummary& s) {
  json::Object out;
  out["rows"] = s.rows;
  out["finite_rows"] = s.finite_rows;
  out["legal_logits"] = s.legal_logits;
  out["old_probability_underflows"] = s.old_probability_underflows;
  out["new_probability_underflows"] = s.new_probability_underflows;
  out["nonfinite_values"] = s.nonfinite_values;
  out["schema_error_rows"] = s.schema_error_rows;
  out["mass_residual_rows"] = s.mass_residual_rows;
  out["mean_old_raw_mass_residual"] = s.mean_old_raw_mass_residual;
  out["max_old_raw_mass_residual"] = s.max_old_raw_mass_residual;
  out["mean_new_raw_mass_residual"] = s.mean_new_raw_mass_residual;
  out["max_new_raw_mass_residual"] = s.max_new_raw_mass_residual;
  out["mean_abs_chosen_log_prob_delta"] =
      s.mean_abs_chosen_log_prob_delta;
  out["max_abs_chosen_log_prob_delta"] =
      s.max_abs_chosen_log_prob_delta;
  out["mean_full_kl_old_new"] = s.mean_kl_old_new;
  out["max_full_kl_old_new"] = s.max_kl_old_new;
  out["mean_full_kl_new_old"] = s.mean_kl_new_old;
  out["max_full_kl_new_old"] = s.max_kl_new_old;
  out["ratio_min"] = s.ratio_min;
  out["ratio_p01"] = s.ratio_p01;
  out["ratio_p50"] = s.ratio_p50;
  out["ratio_p99"] = s.ratio_p99;
  out["ratio_max"] = s.ratio_max;
  out["ratio_lt_0_8"] = s.ratio_lt_0_8;
  out["ratio_gt_1_2"] = s.ratio_gt_1_2;
  return out;
}

template <typename Predicate>
json::Object PpoNumericalParitySubsetJson(
    const std::vector<PpoNumericalParityRow>& rows, Predicate predicate) {
  std::vector<PpoNumericalParityRow> subset;
  for (const auto& row : rows) {
    if (predicate(row)) subset.push_back(row);
  }
  return PpoNumericalParitySummaryJson(
      SummarizePpoNumericalParityRows(subset));
}

std::string PpoDecisionRoleName(int role) {
  switch (static_cast<DuneDecisionRole>(role)) {
    case DuneDecisionRole::kForcedOrBookkeeping: return "forced_or_bookkeeping";
    case DuneDecisionRole::kLeaderSelection: return "leader_selection";
    case DuneDecisionRole::kAgentPrimary: return "agent_primary";
    case DuneDecisionRole::kAgentContinuation: return "agent_continuation";
    case DuneDecisionRole::kPurchase: return "purchase";
    case DuneDecisionRole::kCombatIntrigue: return "combat_intrigue";
    case DuneDecisionRole::kOtherOptional: return "other_optional";
  }
  return "invalid_role_" + std::to_string(role);
}

json::Object PpoNumericalParitySplitsJson(
    const std::vector<PpoNumericalParityRow>& rows) {
  json::Object legal_splits;
  std::set<int> legal_counts;
  for (const auto& row : rows) legal_counts.insert(row.legal_count);
  for (int legal_count : legal_counts) {
    legal_splits[std::to_string(legal_count)] = json::Value(
        PpoNumericalParitySubsetJson(rows, [=](const auto& row) {
          return row.legal_count == legal_count;
        }));
  }
  json::Object probability_splits;
  for (const std::string& bucket : {
           "p_lt_1e-6", "p_1e-6_to_1e-4", "p_1e-4_to_1e-2",
           "p_1e-2_to_0_1", "p_0_1_to_0_5", "p_ge_0_5", "invalid"}) {
    probability_splits[bucket] = json::Value(PpoNumericalParitySubsetJson(
        rows, [&](const auto& row) {
          return PpoNumericalParityOldProbabilityBucket(row) == bucket;
        }));
  }
  json::Object advantage_splits;
  advantage_splits["negative"] = json::Value(PpoNumericalParitySubsetJson(
      rows, [](const auto& r) { return r.advantage < 0.0; }));
  advantage_splits["zero"] = json::Value(PpoNumericalParitySubsetJson(
      rows, [](const auto& r) { return r.advantage == 0.0; }));
  advantage_splits["positive"] = json::Value(PpoNumericalParitySubsetJson(
      rows, [](const auto& r) { return r.advantage > 0.0; }));
  advantage_splits["nonfinite"] = json::Value(PpoNumericalParitySubsetJson(
      rows, [](const auto& r) { return !std::isfinite(r.advantage); }));
  json::Object role_splits;
  for (int role = 0; role < 7; ++role) {
    role_splits[PpoDecisionRoleName(role)] = json::Value(
        PpoNumericalParitySubsetJson(rows, [=](const auto& row) {
          return row.decision_role == role;
        }));
  }
  role_splits["invalid"] = json::Value(PpoNumericalParitySubsetJson(
      rows, [](const auto& row) {
        return row.decision_role < 0 || row.decision_role >= 7;
      }));
  json::Object splits;
  splits["legal_count_exact"] = json::Value(legal_splits);
  splits["old_chosen_probability"] = json::Value(probability_splits);
  splits["advantage_sign"] = json::Value(advantage_splits);
  splits["decision_role"] = json::Value(role_splits);
  return splits;
}

using PpoParityViolationMap =
    std::map<std::string, std::vector<std::string>>;

PpoParityViolationMap BuildPpoParityViolationMap(
    const std::vector<PpoNumericalParityRow>& rows) {
  PpoParityViolationMap violations;
  for (const std::string& name : {
           "schema_errors", "probability_underflow", "nonfinite_values",
           "raw_mass_residual", "chosen_logprob_delta", "full_legal_kl",
           "ratio_min", "ratio_max"}) {
    violations[name] = {};
  }
  const double mass_bound =
      absl::GetFlag(FLAGS_numerical_parity_max_raw_mass_residual);
  const double delta_bound =
      absl::GetFlag(FLAGS_numerical_parity_max_abs_logprob_delta);
  const double kl_bound =
      absl::GetFlag(FLAGS_numerical_parity_max_full_kl);
  const double ratio_min = absl::GetFlag(FLAGS_numerical_parity_min_ratio);
  const double ratio_max = absl::GetFlag(FLAGS_numerical_parity_max_ratio);
  for (const auto& row : rows) {
    const std::string& identity = row.row_identity_sha256;
    if (row.schema_errors != 0) violations["schema_errors"].push_back(identity);
    if (row.old_probability_underflows != 0 ||
        row.new_probability_underflows != 0) {
      violations["probability_underflow"].push_back(identity);
    }
    if (row.nonfinite_values != 0) {
      violations["nonfinite_values"].push_back(identity);
    }
    if (!row.mass_residual_valid ||
        row.old_raw_mass_residual > mass_bound ||
        row.new_raw_mass_residual > mass_bound) {
      violations["raw_mass_residual"].push_back(identity);
    }
    if (std::abs(row.chosen_log_prob_delta) > delta_bound) {
      violations["chosen_logprob_delta"].push_back(identity);
    }
    if (row.kl_old_new > kl_bound || row.kl_new_old > kl_bound) {
      violations["full_legal_kl"].push_back(identity);
    }
    if (row.ratio < ratio_min) violations["ratio_min"].push_back(identity);
    if (row.ratio > ratio_max) violations["ratio_max"].push_back(identity);
  }
  return violations;
}

bool PpoParityViolationMapPasses(const PpoParityViolationMap& violations) {
  for (const auto& item : violations) {
    if (!item.second.empty()) return false;
  }
  return true;
}

json::Object PpoParityViolationSetJson(
    const std::vector<std::string>& identities) {
  std::string payload;
  std::string error;
  if (!CanonicalPpoParityViolationIdentityPayload(
          identities, &payload, &error)) {
    SpielFatalError("Violation identity set is malformed: " + error);
  }
  json::Object out;
  out["count"] = static_cast<int64_t>(identities.size());
  out["row_identity_set_sha256"] = ComputeStringSHA256(payload);
  return out;
}

json::Object PpoParityViolationMapJson(
    const PpoParityViolationMap& violations) {
  json::Object out;
  for (const auto& item : violations) {
    out[item.first] = json::Value(PpoParityViolationSetJson(item.second));
  }
  return out;
}

json::Object PpoParityViolationIntersectionsJson(
    const PpoParityViolationMap& first,
    const PpoParityViolationMap& second) {
  json::Object out;
  for (const auto& item : first) {
    const auto found = second.find(item.first);
    if (found == second.end()) {
      SpielFatalError("Violation maps have different threshold names");
    }
    const std::vector<std::string> intersection =
        IntersectPpoParityViolationIdentities(item.second, found->second);
    out[item.first] = json::Value(PpoParityViolationSetJson(intersection));
  }
  return out;
}

struct PpoNumericalParitySourceRecord {
  std::string relative_path;
  std::string absolute_path;
  int64_t size = 0;
  std::string sha256;
};

struct PpoNumericalParitySourceProvenance {
  std::string root;
  std::string combined_sha256;
  std::vector<PpoNumericalParitySourceRecord> files;
};

struct PpoParityCellResult {
  std::string name;
  std::vector<PpoNumericalParityRow> rows;
  std::vector<std::string> schema_errors;
  std::string row_evidence_sha256;
  std::string raw_logit_sha256;
  std::string policy_sha256;
  int64_t model_forward_calls = 0;
  int64_t dense_mask_rows_ok = 0;
  bool module_train_mode = false;
  bool autocast_enabled = false;
  std::string input_dtype = "Float32";
  std::string pre_forward_dtype = "Float32";
  double wall_time_s = 0.0;
  double device_time_s = -1.0;
};

struct PpoParityIntegrityStats {
  int64_t raw_values_total = 0;
  int64_t raw_values_bf16_exact = 0;
  int64_t vector_width_rows_ok = 0;
  int64_t behavior_vector_exact_rows = 0;
  int64_t chosen_scalar_exact_rows = 0;
};

struct PpoParityReplayGeometryStats {
  EvaluatorStats before;
  EvaluatorStats after;
  int64_t returned_metadata_exact_rows = 0;
  int64_t raw_legal_exact_rows = 0;
  int64_t cpu_policy_exact_rows = 0;
  int64_t model_forward_calls = 0;
  std::string captured_raw_sha256;
  std::string replay_raw_sha256;
};

struct PpoParityOriginalGeometryModelStats {
  int64_t raw_legal_exact_rows = 0;
  double raw_legal_max_abs_delta = 0.0;
};

PpoNumericalParitySourceProvenance LoadPpoNumericalParitySourceProvenance(
    const std::string& source_root, const std::string& registered_sha256) {
  std::error_code ec;
  const std::filesystem::path canonical_root =
      std::filesystem::canonical(source_root, ec);
  if (ec || !std::filesystem::is_directory(canonical_root)) {
    SpielFatalError(
        "Numerical parity source root is not a readable directory: " +
        source_root);
  }
  PpoNumericalParitySourceProvenance out;
  out.root = canonical_root.string();
  std::vector<std::pair<std::string, std::string>> digest_records;
  for (const std::string& relative :
       PpoNumericalParitySourceRelativePaths()) {
    const std::filesystem::path path = canonical_root / relative;
    if (!std::filesystem::is_regular_file(path)) {
      SpielFatalError(
          "Numerical parity required source is not a regular file: " +
          path.string());
    }
    size_t size = 0;
    const std::string digest = ComputeFileSHA256(path.string(), &size);
    if (digest.empty()) {
      SpielFatalError("Numerical parity could not hash required source: " +
                      path.string());
    }
    out.files.push_back({relative, path.string(),
                         static_cast<int64_t>(size), digest});
    digest_records.push_back({relative, digest});
  }
  std::string payload;
  std::string payload_error;
  if (!CanonicalPpoNumericalParitySourcePayload(
          digest_records, &payload, &payload_error)) {
    SpielFatalError("Numerical parity source canonicalization failed: " +
                    payload_error);
  }
  out.combined_sha256 = ComputeStringSHA256(payload);
  if (out.combined_sha256 != registered_sha256) {
    SpielFatalError(
        "Numerical parity registered source SHA-256 mismatch: expected " +
        registered_sha256 + " computed " + out.combined_sha256);
  }
  return out;
}

PpoNumericalParitySourceProvenance LoadVrpoCaptureSourceProvenance(
    const std::string& source_root, const std::string& registered_sha256,
    const std::vector<std::string>& relative_paths) {
  std::error_code ec;
  const std::filesystem::path canonical_root =
      std::filesystem::canonical(source_root, ec);
  if (ec || !std::filesystem::is_directory(canonical_root)) {
    SpielFatalError("VRPO capture source root is not a readable directory: " +
                    source_root);
  }
  PpoNumericalParitySourceProvenance out;
  out.root = canonical_root.string();
  std::string canonical_payload;
  for (const std::string& relative : relative_paths) {
    const std::filesystem::path path = canonical_root / relative;
    if (!std::filesystem::is_regular_file(path)) {
      SpielFatalError("VRPO capture required source is not regular: " +
                      path.string());
    }
    size_t size = 0;
    const std::string digest = ComputeFileSHA256(path.string(), &size);
    if (digest.size() != 64) {
      SpielFatalError("VRPO capture could not hash source: " + path.string());
    }
    out.files.push_back({relative, path.string(),
                         static_cast<int64_t>(size), digest});
    canonical_payload.append(relative);
    canonical_payload.push_back('\0');
    canonical_payload.append(digest);
    canonical_payload.push_back('\n');
  }
  out.combined_sha256 = ComputeStringSHA256(canonical_payload);
  if (out.combined_sha256 != registered_sha256) {
    SpielFatalError("VRPO capture source digest mismatch: expected " +
                    registered_sha256 + " computed " + out.combined_sha256);
  }
  return out;
}

PpoParityCellResult ReplayPpoInferenceExactGeometry(
    BatchedEvaluator* evaluator, const std::vector<PpoTransition>& batch,
    const PpoParityBatchGeometry& geometry, int64_t action_dim,
    float logit_cap, const EvaluatorStats& before,
    PpoParityReplayGeometryStats* stats) {
  PpoParityCellResult cell;
  cell.name = "live_inference_exact_original_batches";
  cell.module_train_mode = false;
  cell.autocast_enabled = false;
  cell.rows.resize(batch.size());
  stats->before = before;
  std::stringstream evidence, captured_raw, replay_raw, replay_policy_bytes;
  evidence << "dune_raw_ppo_numerical_parity_v4";
  evidence.put('\0');
  evidence << cell.name;
  evidence.put('\0');
  for (size_t group_index = 0;
       group_index < geometry.row_indices_by_group.size(); ++group_index) {
    const auto& indices = geometry.row_indices_by_group[group_index];
    std::vector<std::vector<float>> observations;
    observations.reserve(indices.size());
    for (size_t index : indices) observations.push_back(batch[index].state);
    const std::vector<EvalResult> replay = evaluator->EvaluateBatch(observations);
    ++stats->model_forward_calls;
    if (replay.size() != indices.size()) {
      cell.schema_errors.push_back("R returned row count differs from group");
      continue;
    }
    for (size_t position = 0; position < indices.size(); ++position) {
      const size_t index = indices[position];
      const PpoTransition& transition = batch[index];
      const EvalResult& result = replay[position];
      const std::string identity =
          PpoParityRowIdentitySha256(static_cast<int64_t>(index), transition);
      bool valid = true;
      auto error = [&](const std::string& message) {
        cell.schema_errors.push_back("row " + std::to_string(index) +
                                     ": " + message);
        valid = false;
      };
      const int64_t expected_replay_batch_id =
          static_cast<int64_t>(before.batches + group_index);
      if (result.physical_batch_id == expected_replay_batch_id &&
          result.physical_batch_size == static_cast<int32_t>(indices.size()) &&
          result.physical_batch_row == static_cast<int32_t>(position)) {
        ++stats->returned_metadata_exact_rows;
      } else {
        error("returned replay batch metadata differs from exact group");
      }
      if (static_cast<int64_t>(result.logits.size()) != action_dim ||
          !std::all_of(result.logits.begin(), result.logits.end(),
                       [](float value) { return std::isfinite(value); })) {
        error("returned full logits have wrong width or nonfinite value");
      }
      std::vector<float> replay_raw_legal;
      replay_raw_legal.reserve(transition.legal_actions.size());
      for (Action action : transition.legal_actions) {
        replay_raw_legal.push_back(
            action >= 0 && action < action_dim &&
                    action < static_cast<Action>(result.logits.size())
                ? result.logits[action]
                : std::numeric_limits<float>::quiet_NaN());
      }
      bool raw_exact = replay_raw_legal.size() ==
                       transition.behavior_raw_legal_logits.size();
      if (raw_exact) {
        for (size_t j = 0; j < replay_raw_legal.size(); ++j) {
          if (PpoParityFloatBits(replay_raw_legal[j]) != PpoParityFloatBits(
                  transition.behavior_raw_legal_logits[j])) {
            raw_exact = false;
            break;
          }
        }
      }
      if (raw_exact) ++stats->raw_legal_exact_rows;
      else error("returned raw legal logits are not bit-exact captured logits");
      std::vector<float> replay_policy;
      std::string recompute_error;
      if (!RecomputePpoBehaviorLegalLogProbs(
              replay_raw_legal, logit_cap, &replay_policy,
              &recompute_error)) {
        error("returned-logit CPU policy recompute failed: " + recompute_error);
      }
      bool policy_exact = replay_policy.size() ==
                          transition.behavior_legal_log_probs.size();
      if (policy_exact) {
        for (size_t j = 0; j < replay_policy.size(); ++j) {
          if (PpoParityFloatBits(replay_policy[j]) != PpoParityFloatBits(
                  transition.behavior_legal_log_probs[j])) {
            policy_exact = false;
            break;
          }
        }
      }
      if (policy_exact) ++stats->cpu_policy_exact_rows;
      else error("returned-logit CPU policy is not bit-exact captured policy");

      PpoNumericalParityInput input;
      input.legal_count = static_cast<int>(transition.legal_actions.size());
      input.decision_role = transition.decision_role;
      input.advantage = transition.advantage;
      input.stored_chosen_log_prob = transition.old_log_prob;
      for (size_t j = 0; j < transition.legal_actions.size(); ++j) {
        input.old_log_probs.push_back(
            j < transition.behavior_legal_log_probs.size()
                ? transition.behavior_legal_log_probs[j]
                : std::numeric_limits<double>::quiet_NaN());
        input.new_log_probs.push_back(
            j < replay_policy.size()
                ? replay_policy[j]
                : std::numeric_limits<double>::quiet_NaN());
        if (transition.legal_actions[j] == transition.action) {
          input.chosen_index = static_cast<int>(j);
        }
        const float captured =
            j < transition.behavior_raw_legal_logits.size()
                ? transition.behavior_raw_legal_logits[j]
                : std::numeric_limits<float>::quiet_NaN();
        const float returned =
            j < replay_raw_legal.size()
                ? replay_raw_legal[j]
                : std::numeric_limits<float>::quiet_NaN();
        captured_raw.write(reinterpret_cast<const char*>(&captured),
                           sizeof(captured));
        replay_raw.write(reinterpret_cast<const char*>(&returned),
                         sizeof(returned));
        const float replay_policy_value =
            j < replay_policy.size()
                ? replay_policy[j]
                : std::numeric_limits<float>::quiet_NaN();
        replay_policy_bytes.write(
            reinterpret_cast<const char*>(&replay_policy_value),
            sizeof(replay_policy_value));
      }
      PpoNumericalParityRow row;
      std::string row_error;
      if (!ComputePpoNumericalParityRow(input, &row, &row_error)) {
        error("R parity row failed: " + row_error);
      }
      if (!valid) {
        row.schema_errors = std::max<int64_t>(1, row.schema_errors);
        row.nonfinite_values = std::max<int64_t>(1, row.nonfinite_values);
      }
      row.row_identity_sha256 = identity;
      cell.rows[index] = row;
      evidence << identity;
      evidence.put('\0');
      for (float value : replay_raw_legal) {
        evidence.write(reinterpret_cast<const char*>(&value), sizeof(value));
      }
    }
  }
  stats->after = evaluator->GetStats();
  stats->captured_raw_sha256 = ComputeStringSHA256(captured_raw.str());
  stats->replay_raw_sha256 = ComputeStringSHA256(replay_raw.str());
  cell.model_forward_calls = stats->model_forward_calls;
  cell.row_evidence_sha256 = ComputeStringSHA256(evidence.str());
  cell.raw_logit_sha256 = stats->replay_raw_sha256;
  cell.policy_sha256 = ComputeStringSHA256(replay_policy_bytes.str());
  return cell;
}

PpoParityCellResult ReplayPpoTrainingModelOriginalGeometry(
    const std::shared_ptr<SharedDunePolicyValueNetImpl>& model,
    const std::vector<PpoTransition>& batch,
    const PpoParityBatchGeometry& geometry, int64_t obs_size,
    int64_t action_dim, torch::Device device, float logit_cap,
    bool learner_autocast,
    PpoParityOriginalGeometryModelStats* stats) {
  PpoParityCellResult cell;
  cell.name = "training_model_exact_original_batches";
  cell.module_train_mode = true;
  cell.autocast_enabled = learner_autocast;
  cell.rows.resize(batch.size());
  std::stringstream evidence, raw_bytes, policy_bytes;
  evidence << "dune_raw_ppo_numerical_parity_v4";
  evidence.put('\0');
  evidence << cell.name;
  evidence.put('\0');
  const bool was_training = model->is_training();
  model->train();
  torch::NoGradGuard no_grad;
  for (const auto& indices : geometry.row_indices_by_group) {
    const int64_t count = static_cast<int64_t>(indices.size());
    torch::Tensor states_cpu = torch::zeros(
        {count, obs_size}, torch::TensorOptions().dtype(torch::kFloat32));
    torch::Tensor masks_cpu = torch::zeros(
        {count, action_dim}, torch::TensorOptions().dtype(torch::kBool));
    float* state_data = states_cpu.data_ptr<float>();
    bool* mask_data = masks_cpu.data_ptr<bool>();
    std::vector<bool> valid(count, true);
    for (int64_t i = 0; i < count; ++i) {
      const PpoTransition& transition = batch[indices[i]];
      if (static_cast<int64_t>(transition.state.size()) != obs_size) {
        valid[i] = false;
        cell.schema_errors.push_back("D observation width mismatch");
      } else {
        std::memcpy(state_data + i * obs_size, transition.state.data(),
                    obs_size * sizeof(float));
      }
      for (Action action : transition.legal_actions) {
        if (action >= 0 && action < action_dim) {
          mask_data[i * action_dim + action] = true;
        }
      }
    }
    const torch::Tensor popcounts =
        masks_cpu.sum(1).to(torch::kCPU).to(torch::kInt64).contiguous();
    const int64_t* popcount = popcounts.data_ptr<int64_t>();
    for (int64_t i = 0; i < count; ++i) {
      if (popcount[i] == static_cast<int64_t>(
                             batch[indices[i]].legal_actions.size())) {
        ++cell.dense_mask_rows_ok;
      } else {
        valid[i] = false;
        cell.schema_errors.push_back("D dense-mask popcount mismatch");
      }
    }
    const torch::Tensor states = states_cpu.to(device);
    const torch::Tensor masks = masks_cpu.to(device);
    if (states_cpu.scalar_type() != torch::kFloat32 ||
        states.scalar_type() != torch::kFloat32) {
      cell.schema_errors.push_back(
          "D input/pre-forward dtype is not Float32");
    }
    torch::Tensor raw_logits, replay_log_probs;
    auto compute = [&]() {
      const auto outputs = model->forward(states);
      raw_logits = outputs.logits;
      const torch::Tensor centered =
          CenterAndCapLogitsTensor(raw_logits, masks, logit_cap);
      replay_log_probs = torch::log_softmax(
          centered.masked_fill(masks.logical_not(), -1e9f), -1);
    };
    if (learner_autocast) {
      AutocastGuard autocast_guard(c10::DeviceType::CUDA, true);
      compute();
    } else {
      compute();
    }
    ++cell.model_forward_calls;
    const torch::Tensor raw_cpu =
        raw_logits.to(torch::kCPU).to(torch::kFloat32).contiguous();
    const torch::Tensor log_probs_cpu =
        replay_log_probs.to(torch::kCPU).to(torch::kFloat32).contiguous();
    const float* raw_data = raw_cpu.data_ptr<float>();
    const float* logprob_data = log_probs_cpu.data_ptr<float>();
    for (int64_t i = 0; i < count; ++i) {
      const size_t index = indices[i];
      const PpoTransition& transition = batch[index];
      const std::string identity =
          PpoParityRowIdentitySha256(static_cast<int64_t>(index), transition);
      bool raw_exact = transition.behavior_raw_legal_logits.size() ==
                       transition.legal_actions.size();
      PpoNumericalParityInput input;
      input.legal_count = static_cast<int>(transition.legal_actions.size());
      input.decision_role = transition.decision_role;
      input.advantage = transition.advantage;
      input.stored_chosen_log_prob = transition.old_log_prob;
      for (size_t j = 0; j < transition.legal_actions.size(); ++j) {
        const Action action = transition.legal_actions[j];
        const float replay_raw =
            action >= 0 && action < action_dim
                ? raw_data[i * action_dim + action]
                : std::numeric_limits<float>::quiet_NaN();
        const float captured_raw =
            j < transition.behavior_raw_legal_logits.size()
                ? transition.behavior_raw_legal_logits[j]
                : std::numeric_limits<float>::quiet_NaN();
        if (PpoParityFloatBits(replay_raw) !=
            PpoParityFloatBits(captured_raw)) {
          raw_exact = false;
        }
        if (std::isfinite(replay_raw) && std::isfinite(captured_raw)) {
          stats->raw_legal_max_abs_delta = std::max(
              stats->raw_legal_max_abs_delta,
              std::abs(static_cast<double>(replay_raw) - captured_raw));
        }
        input.old_log_probs.push_back(
            j < transition.behavior_legal_log_probs.size()
                ? transition.behavior_legal_log_probs[j]
                : std::numeric_limits<double>::quiet_NaN());
        input.new_log_probs.push_back(
            action >= 0 && action < action_dim
                ? logprob_data[i * action_dim + action]
                : std::numeric_limits<double>::quiet_NaN());
        if (action == transition.action) input.chosen_index = j;
        evidence.write(reinterpret_cast<const char*>(&replay_raw),
                       sizeof(replay_raw));
        raw_bytes.write(reinterpret_cast<const char*>(&replay_raw),
                        sizeof(replay_raw));
        const float replay_policy = static_cast<float>(
            input.new_log_probs.back());
        policy_bytes.write(reinterpret_cast<const char*>(&replay_policy),
                           sizeof(replay_policy));
      }
      if (raw_exact) ++stats->raw_legal_exact_rows;
      PpoNumericalParityRow row;
      std::string error;
      if (!ComputePpoNumericalParityRow(input, &row, &error)) {
        cell.schema_errors.push_back("row " + std::to_string(index) +
                                     ": " + error);
        valid[i] = false;
      }
      if (!valid[i]) {
        row.schema_errors = std::max<int64_t>(1, row.schema_errors);
        row.nonfinite_values = std::max<int64_t>(1, row.nonfinite_values);
      }
      row.row_identity_sha256 = identity;
      cell.rows[index] = row;
      evidence << identity;
      evidence.put('\0');
    }
  }
  if (was_training) model->train(); else model->eval();
  cell.row_evidence_sha256 = ComputeStringSHA256(evidence.str());
  cell.raw_logit_sha256 = ComputeStringSHA256(raw_bytes.str());
  cell.policy_sha256 = ComputeStringSHA256(policy_bytes.str());
  return cell;
}

bool WritePpoNumericalParityArtifact(
    const std::string& output_path,
    const std::vector<PpoNumericalParityRow>& primary_rows,
    const std::vector<std::string>& primary_schema_errors,
    const std::string& primary_row_evidence_sha256,
    const std::vector<PpoNumericalParityRow>& control_rows,
    const std::vector<std::string>& control_schema_errors,
    const std::string& control_row_evidence_sha256,
    const CollectResult& collect,
    const std::string& model_hash_before,
    const std::string& model_hash_after,
    const std::string& inference_hash_before,
    const std::string& inference_hash_after,
    const std::string& source_manifest,
    const std::string& source_manifest_sha256, int64_t source_global_update,
    const std::string& source_checkpoint_uuid,
    const std::string& config_fingerprint, torch::Device device,
    int64_t obs_size, int64_t action_size, const std::string& command_line,
    const PpoNumericalParitySourceProvenance& source_provenance) {
  const PpoNumericalParitySummary primary =
      SummarizePpoNumericalParityRows(primary_rows);
  const PpoNumericalParitySummary control =
      SummarizePpoNumericalParityRows(control_rows);
  int64_t nontrivial_rows = 0;
  for (const auto& row : primary_rows) {
    if (row.legal_count > 1) ++nontrivial_rows;
  }

  json::Array failures;
  auto reject = [&](const std::string& reason) { failures.emplace_back(reason); };
  if (!primary_schema_errors.empty()) {
    reject(absl::StrFormat("%d malformed parity rows",
                           static_cast<int>(primary_schema_errors.size())));
  }
  if (primary.rows != static_cast<int64_t>(collect.rollout.size())) {
    reject("primary replay row count does not match collected transitions");
  }
  if (primary.mass_residual_rows != primary.rows) {
    reject("primary raw mass residual coverage is incomplete");
  }
  if (!collect.episode_ids_unique) reject("episode IDs are not unique");
  if (primary.rows < absl::GetFlag(FLAGS_numerical_parity_min_rows)) {
    reject("total row floor not met");
  }
  if (nontrivial_rows <
      absl::GetFlag(FLAGS_numerical_parity_min_nontrivial_rows)) {
    reject("nontrivial row floor not met");
  }
  if (primary.nonfinite_values != 0) reject("nonfinite parity values observed");
  if (primary.old_probability_underflows != 0) {
    reject("rollout behavior probability underflow observed");
  }
  if (primary.new_probability_underflows != 0) {
    reject("learner autocast probability underflow observed");
  }
  const double max_raw_mass_residual =
      absl::GetFlag(FLAGS_numerical_parity_max_raw_mass_residual);
  if (primary.mass_residual_rows == primary.rows &&
      !PpoNumericalParityRawMassWithinBound(primary,
                                             max_raw_mass_residual)) {
    reject("raw legal probability mass residual exceeds bound");
  }
  if (primary.max_abs_chosen_log_prob_delta >
      absl::GetFlag(FLAGS_numerical_parity_max_abs_logprob_delta)) {
    reject("chosen-action log-probability delta exceeds bound");
  }
  if (primary.max_kl_old_new >
          absl::GetFlag(FLAGS_numerical_parity_max_full_kl) ||
      primary.max_kl_new_old >
          absl::GetFlag(FLAGS_numerical_parity_max_full_kl)) {
    reject("full legal categorical KL exceeds bound");
  }
  if (primary.ratio_min <
      absl::GetFlag(FLAGS_numerical_parity_min_ratio)) {
    reject("chosen-action ratio minimum is below bound");
  }
  if (primary.ratio_max >
      absl::GetFlag(FLAGS_numerical_parity_max_ratio)) {
    reject("chosen-action ratio maximum is above bound");
  }
  if (model_hash_before != model_hash_after) {
    reject("model state changed during read-only diagnostic");
  }
  if (inference_hash_before != inference_hash_after) {
    reject("rollout inference model changed during read-only diagnostic");
  }
  if (model_hash_before != inference_hash_before) {
    reject("learner and rollout inference models differed before collection");
  }

  json::Object root;
  root["schema"] = "dune_raw_ppo_numerical_parity_v2";
  root["epistemic_label"] =
      "no_training_actual_learner_autocast_parity_gate_with_fp32_control";
  root["measurement_intent"] =
      "Gate unchanged-weight parity on the historical production learner "
      "CUDA BF16 autocast policy path; retain CUDA FP32 replay as a "
      "descriptive cross-precision control only.";
  root["prior_v1_disposition"] =
      "The prior v1 cross-precision REJECT is a separate valid artifact; v2 "
      "does not overwrite or reinterpret it.";
  root["gate_cell"] = "actual_learner_cuda_bf16_autocast";
  root["control_cell"] = "descriptive_cuda_fp32_no_autocast";
  root["control_can_affect_status"] = false;
  root["registration_id"] =
      absl::GetFlag(FLAGS_numerical_parity_registration_id);
  root["scope"] = "raw_ppo_only";
  root["optimizer_steps"] = int64_t{0};
  root["optimizer_checkpoint_loaded"] = false;
  root["command_line"] = command_line;
  root["config_fingerprint"] = config_fingerprint;
  root["device"] = device.str();
  root["rollout_precision"] =
      device.is_cuda() ? "cuda_bf16_autocast" : "cpu_fp32";
  root["primary_replay_precision"] = "cuda_bf16_autocast";
  root["descriptive_control_replay_precision"] =
      "cuda_fp32_no_autocast_tf32_allowed";
  root["train_amp"] = absl::GetFlag(FLAGS_train_amp);
  root["ppo_minibatch_size"] = static_cast<int64_t>(
      absl::GetFlag(FLAGS_ppo_minibatch_size));
  root["logit_cap"] = absl::GetFlag(FLAGS_logit_cap);
  root["seed"] = static_cast<int64_t>(absl::GetFlag(FLAGS_seed));
  root["threads"] = static_cast<int64_t>(absl::GetFlag(FLAGS_threads));
  root["eval_batch_size"] =
      static_cast<int64_t>(absl::GetFlag(FLAGS_eval_batch_size));
  root["eval_timeout_ms"] =
      static_cast<int64_t>(absl::GetFlag(FLAGS_eval_timeout_ms));
  root["rollout_games"] =
      static_cast<int64_t>(absl::GetFlag(FLAGS_rollout_games));
  root["hard_transition_ceiling"] = static_cast<int64_t>(
      absl::GetFlag(FLAGS_rollout_games)) * 5000;
  root["games_collected"] = static_cast<int64_t>(collect.games);
  root["transitions_collected"] =
      static_cast<int64_t>(collect.rollout.size());
  root["nontrivial_rows"] = nontrivial_rows;
  root["observation_size"] = obs_size;
  root["action_size"] = action_size;
  root["rollout_hash"] = ComputeRolloutHash(collect.rollout);
  root["model_state_sha256_before"] = model_hash_before;
  root["model_state_sha256_after"] = model_hash_after;
  root["inference_state_sha256_before"] = inference_hash_before;
  root["inference_state_sha256_after"] = inference_hash_after;

  size_t model_file_size = 0;
  const std::string model_path = absl::GetFlag(FLAGS_model_checkpoint);
  json::Object source;
  source["model_path"] = std::filesystem::absolute(model_path).string();
  source["model_size"] = static_cast<int64_t>(model_file_size);
  source["model_sha256"] =
      ComputeFileSHA256(model_path, &model_file_size);
  source["model_size"] = static_cast<int64_t>(model_file_size);
  source["manifest_path"] = std::filesystem::absolute(source_manifest).string();
  source["manifest_sha256"] = source_manifest_sha256;
  source["global_update"] = source_global_update;
  source["checkpoint_uuid"] = source_checkpoint_uuid;
  size_t binary_size = 0;
  std::error_code ec;
  const std::filesystem::path binary_path =
      std::filesystem::read_symlink("/proc/self/exe", ec);
  if (ec || binary_path.empty()) {
    reject("could not resolve executed binary path");
  } else {
    source["binary_path"] = binary_path.string();
    source["binary_sha256"] =
        ComputeFileSHA256(binary_path.string(), &binary_size);
    source["binary_size"] = static_cast<int64_t>(binary_size);
  }
  root["source"] = json::Value(source);

  json::Object source_code;
  source_code["root"] = source_provenance.root;
  source_code["combined_sha256"] = source_provenance.combined_sha256;
  source_code["canonical_contract"] =
      "fixed ordered relative_path + NUL + lowercase file_sha256 + newline";
  json::Array source_files;
  for (const auto& file : source_provenance.files) {
    json::Object record;
    record["relative_path"] = file.relative_path;
    record["absolute_path"] = file.absolute_path;
    record["size"] = file.size;
    record["sha256"] = file.sha256;
    source_files.emplace_back(json::Value(record));
  }
  source_code["files"] = json::Value(source_files);
  root["source_code"] = json::Value(source_code);

  json::Object thresholds;
  thresholds["min_rows"] = static_cast<int64_t>(
      absl::GetFlag(FLAGS_numerical_parity_min_rows));
  thresholds["min_nontrivial_rows"] = static_cast<int64_t>(
      absl::GetFlag(FLAGS_numerical_parity_min_nontrivial_rows));
  thresholds["max_abs_chosen_logprob_delta"] =
      absl::GetFlag(FLAGS_numerical_parity_max_abs_logprob_delta);
  thresholds["max_full_legal_kl_each_direction"] =
      absl::GetFlag(FLAGS_numerical_parity_max_full_kl);
  thresholds["max_raw_mass_residual"] =
      absl::GetFlag(FLAGS_numerical_parity_max_raw_mass_residual);
  thresholds["min_ratio"] =
      absl::GetFlag(FLAGS_numerical_parity_min_ratio);
  thresholds["max_ratio"] =
      absl::GetFlag(FLAGS_numerical_parity_max_ratio);
  thresholds["require_zero_underflows"] = true;
  thresholds["require_zero_nonfinite"] = true;
  thresholds["require_unchanged_model_hash"] = true;
  root["thresholds"] = json::Value(thresholds);

  auto cell_json = [&](const std::string& label, bool gate_eligible,
                       const std::string& precision,
                       const std::vector<PpoNumericalParityRow>& cell_rows,
                       const PpoNumericalParitySummary& summary,
                       const std::vector<std::string>& cell_errors,
                       const std::string& evidence_sha256) {
    json::Object cell;
    cell["label"] = label;
    cell["gate_eligible"] = gate_eligible;
    cell["precision"] = precision;
    cell["train_amp"] = gate_eligible;
    cell["ppo_minibatch_size"] = static_cast<int64_t>(
        absl::GetFlag(FLAGS_ppo_minibatch_size));
    cell["row_evidence_sha256"] = evidence_sha256;
    cell["row_evidence_contract"] =
        "v2 cell label then ordered row_index,episode_id,player_id,"
        "chosen_action,decision_role,advantage,stored_chosen_logprob,then "
        "each aligned (legal_action,rollout_raw_logprob,replay_raw_logprob),"
        "native fixed-width bytes";
    cell["overall"] = json::Value(PpoNumericalParitySummaryJson(summary));
    cell["splits"] = json::Value(PpoNumericalParitySplitsJson(cell_rows));
    json::Array errors;
    for (const std::string& error : cell_errors) errors.emplace_back(error);
    cell["schema_errors"] = json::Value(errors);
    return cell;
  };
  json::Object cells;
  cells["actual_learner_cuda_bf16_autocast"] = json::Value(cell_json(
      "actual_learner_cuda_bf16_autocast", true, "cuda_bf16_autocast",
      primary_rows, primary, primary_schema_errors,
      primary_row_evidence_sha256));
  cells["descriptive_cuda_fp32_no_autocast"] = json::Value(cell_json(
      "descriptive_cuda_fp32_no_autocast", false,
      "cuda_fp32_no_autocast_tf32_allowed", control_rows, control,
      control_schema_errors, control_row_evidence_sha256));
  root["cells"] = json::Value(cells);
  root["failure_reasons"] = json::Value(failures);
  // Status is assigned LAST: every provenance check above is allowed to reject.
  root["status"] = failures.empty() ? "PASS" : "REJECT";

  const std::filesystem::path output(output_path);
  const std::filesystem::path tmp = output_path + ".tmp";
  if (std::filesystem::exists(output) || std::filesystem::exists(tmp)) {
    SpielFatalError("Numerical parity output already exists; refusing overwrite: " +
                    output_path);
  }
  if (!output.parent_path().empty()) {
    std::filesystem::create_directories(output.parent_path());
  }
  {
    std::ofstream out(tmp, std::ios::trunc);
    if (!out) SpielFatalError("Cannot create numerical parity temp artifact");
    out << json::ToString(root, true) << "\n";
    out.flush();
    if (!out) SpielFatalError("Numerical parity artifact write failed");
  }
  std::filesystem::rename(tmp, output);
  return failures.empty();
}

bool WritePpoNumericalParityArtifactV3(
    const std::string& output_path, const PpoParityCellResult& integrity_cell,
    const PpoParityIntegrityStats& integrity_stats,
    const PpoParityCellResult& learner_cell,
    const PpoParityCellResult& postprocess_cell,
    const CollectResult& collect, const std::string& shared_identity_sha256,
    const std::string& model_hash_before,
    const std::string& model_hash_after,
    const std::string& inference_hash_before,
    const std::string& inference_hash_after,
    const std::string& source_manifest,
    const std::string& source_manifest_sha256, int64_t source_global_update,
    const std::string& source_checkpoint_uuid,
    const std::string& config_fingerprint, torch::Device device,
    int64_t obs_size, int64_t action_size, const std::string& command_line,
    const PpoNumericalParitySourceProvenance& source_provenance) {
  const int64_t transitions = static_cast<int64_t>(collect.rollout.size());
  const PpoNumericalParitySummary a =
      SummarizePpoNumericalParityRows(integrity_cell.rows);
  const PpoNumericalParitySummary b =
      SummarizePpoNumericalParityRows(learner_cell.rows);
  const PpoNumericalParitySummary c =
      SummarizePpoNumericalParityRows(postprocess_cell.rows);
  auto lower_hex64 = [](const std::string& value) {
    if (value.size() != 64) return false;
    for (char ch : value) {
      if (!((ch >= '0' && ch <= '9') || (ch >= 'a' && ch <= 'f'))) {
        return false;
      }
    }
    return true;
  };
  std::vector<std::string> validity_errors = collect.parity_capture_errors;
  auto require = [&](bool condition, const std::string& message) {
    if (!condition) validity_errors.push_back(message);
  };
  require(collect.episode_ids_unique, "episode IDs are not unique");
  require(static_cast<int64_t>(collect.games) ==
              absl::GetFlag(FLAGS_rollout_games),
          "collected game count differs from exact rollout_games");
  require(transitions >= absl::GetFlag(FLAGS_numerical_parity_min_rows),
          "collected transition row floor is not met");
  int64_t nontrivial_rows = 0;
  for (const auto& row : integrity_cell.rows) {
    if (row.legal_count > 1) ++nontrivial_rows;
  }
  require(nontrivial_rows >=
              absl::GetFlag(FLAGS_numerical_parity_min_nontrivial_rows),
          "nontrivial transition row floor is not met");
  require(collect.parity_decisions == transitions,
          "capture decision count differs from transitions");
  require(collect.parity_full_width_ok == transitions,
          "not every evaluator output has full action width");
  require(collect.parity_full_finite_ok == transitions,
          "not every full evaluator output is finite");
  require(collect.parity_legal_ids_unique_in_range == transitions,
          "not every legal-ID vector is unique and in range");
  require(collect.parity_chosen_once == transitions,
          "not every chosen action occurs exactly once");
  require(a.rows == transitions && b.rows == transitions && c.rows == transitions,
          "A/B/C row counts do not equal collected transitions");
  require(a.schema_error_rows == 0 && integrity_cell.schema_errors.empty(),
          "cell A has schema errors");
  require(b.schema_error_rows == 0 && learner_cell.schema_errors.empty(),
          "cell B has schema errors");
  require(c.schema_error_rows == 0 && postprocess_cell.schema_errors.empty(),
          "cell C has schema errors");
  require(a.finite_rows == a.rows, "cell A finite rows do not equal rows");
  require(a.max_abs_chosen_log_prob_delta == 0.0 &&
              a.max_kl_old_new == 0.0 && a.max_kl_new_old == 0.0 &&
              a.ratio_min == 1.0 && a.ratio_max == 1.0,
          "cell A recompute metrics are not bit-exact identity");
  require(integrity_cell.model_forward_calls == 0,
          "cell A executed a model forward");
  require(postprocess_cell.model_forward_calls == 0,
          "cell C executed a model forward");
  require(learner_cell.model_forward_calls > 0,
          "cell B executed no model forward");
  require(learner_cell.dense_mask_rows_ok == transitions &&
              postprocess_cell.dense_mask_rows_ok == transitions,
          "B/C dense-mask popcount coverage is incomplete");
  require(integrity_stats.vector_width_rows_ok == transitions,
          "A vector-width coverage is incomplete");
  require(integrity_stats.behavior_vector_exact_rows == transitions,
          "A behavior-vector bit-exact coverage is incomplete");
  require(integrity_stats.chosen_scalar_exact_rows == transitions,
          "A chosen-scalar bit-exact coverage is incomplete");
  require(integrity_stats.raw_values_total == a.legal_logits,
          "A raw-value count differs from legal-logit count");
  require(integrity_stats.raw_values_total > 0,
          "A captured no raw legal-logit values");
  require(integrity_stats.raw_values_bf16_exact ==
              integrity_stats.raw_values_total,
          "captured raw legal logits do not all round-trip FP32-BF16-FP32 exactly");
  require(lower_hex64(shared_identity_sha256),
          "shared row-identity hash is malformed");
  require(lower_hex64(integrity_cell.row_evidence_sha256) &&
              lower_hex64(learner_cell.row_evidence_sha256) &&
              lower_hex64(postprocess_cell.row_evidence_sha256),
          "one or more cell evidence hashes are malformed");
  require(integrity_cell.row_evidence_sha256 !=
              learner_cell.row_evidence_sha256 &&
              learner_cell.row_evidence_sha256 !=
                  postprocess_cell.row_evidence_sha256 &&
              integrity_cell.row_evidence_sha256 !=
                  postprocess_cell.row_evidence_sha256,
          "cell evidence hashes are not distinct");
  bool shared_rows = integrity_cell.rows.size() == learner_cell.rows.size() &&
                     learner_cell.rows.size() == postprocess_cell.rows.size();
  if (shared_rows) {
    for (size_t i = 0; i < integrity_cell.rows.size(); ++i) {
      if (integrity_cell.rows[i].row_identity_sha256 !=
              learner_cell.rows[i].row_identity_sha256 ||
          learner_cell.rows[i].row_identity_sha256 !=
              postprocess_cell.rows[i].row_identity_sha256 ||
          !lower_hex64(integrity_cell.rows[i].row_identity_sha256)) {
        shared_rows = false;
        break;
      }
    }
  }
  require(shared_rows, "A/B/C row identities differ");
  require(model_hash_before == model_hash_after &&
              inference_hash_before == inference_hash_after &&
              model_hash_before == inference_hash_before,
          "model immutability or initial learner/inference equality failed");

  const bool instrument_valid = validity_errors.empty();
  const PpoParityViolationMap b_violations =
      BuildPpoParityViolationMap(learner_cell.rows);
  const PpoParityViolationMap c_violations =
      BuildPpoParityViolationMap(postprocess_cell.rows);
  const bool b_pass = instrument_valid &&
                      PpoParityViolationMapPasses(b_violations);
  const bool c_pass = instrument_valid &&
                      PpoParityViolationMapPasses(c_violations);
  const PpoParityV3Classification classification =
      ClassifyPpoParityV3(instrument_valid, b_pass, c_pass);

  json::Object root;
  root["schema"] = "dune_raw_ppo_numerical_parity_v3";
  root["status"] = instrument_valid ? "VALID" : "INVALID";
  root["classification"] = PpoParityV3ClassificationName(classification);
  root["training_authorized"] = false;
  root["registration_id"] =
      absl::GetFlag(FLAGS_numerical_parity_registration_id);
  root["scope"] = "raw_ppo_only";
  root["epistemic_label"] =
      "no_training_three_cell_rollout_forward_postprocess_causal_diagnostic";
  root["optimizer_steps"] = int64_t{0};
  root["optimizer_checkpoint_loaded"] = false;
  root["backward_calls"] = int64_t{0};
  root["training_updates"] = int64_t{0};
  root["command_line"] = command_line;
  root["config_fingerprint"] = config_fingerprint;
  root["device"] = device.str();
  root["rollout_amp"] = absl::GetFlag(FLAGS_rollout_amp);
  root["train_amp"] = absl::GetFlag(FLAGS_train_amp);
  root["allow_tf32"] = absl::GetFlag(FLAGS_allow_tf32);
  root["ppo_minibatch_size"] = static_cast<int64_t>(
      absl::GetFlag(FLAGS_ppo_minibatch_size));
  root["logit_cap"] = absl::GetFlag(FLAGS_logit_cap);
  root["seed"] = static_cast<int64_t>(absl::GetFlag(FLAGS_seed));
  root["threads"] = static_cast<int64_t>(absl::GetFlag(FLAGS_threads));
  root["eval_batch_size"] =
      static_cast<int64_t>(absl::GetFlag(FLAGS_eval_batch_size));
  root["eval_timeout_ms"] =
      static_cast<int64_t>(absl::GetFlag(FLAGS_eval_timeout_ms));
  root["rollout_games"] =
      static_cast<int64_t>(absl::GetFlag(FLAGS_rollout_games));
  root["hard_transition_ceiling"] = static_cast<int64_t>(
      absl::GetFlag(FLAGS_rollout_games)) * 5000;
  root["games_collected"] = static_cast<int64_t>(collect.games);
  root["transitions_collected"] = transitions;
  root["nontrivial_rows"] = nontrivial_rows;
  root["observation_size"] = obs_size;
  root["action_size"] = action_size;
  root["rollout_hash"] = ComputeRolloutHash(collect.rollout);
  root["shared_row_identity_sha256"] = shared_identity_sha256;
  root["model_state_sha256_before"] = model_hash_before;
  root["model_state_sha256_after"] = model_hash_after;
  root["inference_state_sha256_before"] = inference_hash_before;
  root["inference_state_sha256_after"] = inference_hash_after;

  json::Object capture;
  capture["decisions"] = collect.parity_decisions;
  capture["full_width_ok"] = collect.parity_full_width_ok;
  capture["full_finite_ok"] = collect.parity_full_finite_ok;
  capture["legal_ids_unique_in_range"] =
      collect.parity_legal_ids_unique_in_range;
  capture["chosen_once"] = collect.parity_chosen_once;
  capture["raw_values_total"] = integrity_stats.raw_values_total;
  capture["raw_values_bf16_exact"] =
      integrity_stats.raw_values_bf16_exact;
  capture["raw_values_bf16_exact_fraction"] =
      integrity_stats.raw_values_total > 0
          ? static_cast<double>(integrity_stats.raw_values_bf16_exact) /
                integrity_stats.raw_values_total
          : 0.0;
  json::Array capture_errors;
  for (const std::string& error : validity_errors) {
    capture_errors.emplace_back(error);
  }
  capture["validity_errors"] = json::Value(capture_errors);
  root["hard_validity"] = json::Value(capture);

  auto cell_json = [&](const PpoParityCellResult& cell,
                       const std::string& role,
                       const std::string& precision,
                       const PpoNumericalParitySummary& summary,
                       const PpoParityViolationMap* violations) {
    json::Object out;
    out["name"] = cell.name;
    out["role"] = role;
    out["precision"] = precision;
    out["model_forward_calls"] = cell.model_forward_calls;
    out["dense_mask_rows_ok"] = cell.dense_mask_rows_ok;
    out["row_evidence_sha256"] = cell.row_evidence_sha256;
    out["overall"] = json::Value(PpoNumericalParitySummaryJson(summary));
    out["splits"] = json::Value(PpoNumericalParitySplitsJson(cell.rows));
    json::Array errors;
    for (const std::string& error : cell.schema_errors) {
      errors.emplace_back(error);
    }
    out["schema_errors"] = json::Value(errors);
    if (violations != nullptr) {
      out["threshold_pass"] = PpoParityViolationMapPasses(*violations);
      out["violations"] = json::Value(PpoParityViolationMapJson(*violations));
    }
    return out;
  };
  json::Object cells;
  cells["A"] = json::Value(cell_json(
      integrity_cell, "hard_validity", "rollout_cpu_float_arithmetic", a,
      nullptr));
  cells["B"] = json::Value(cell_json(
      learner_cell, "actual_learner_phenotype", "cuda_bf16_autocast", b,
      &b_violations));
  cells["C"] = json::Value(cell_json(
      postprocess_cell, "captured_logits_postprocess_isolate",
      "captured_logits_dense_cuda_bf16", c, &c_violations));
  root["cells"] = json::Value(cells);
  root["B_C_violation_intersections"] = json::Value(
      PpoParityViolationIntersectionsJson(b_violations, c_violations));

  json::Object thresholds;
  thresholds["min_rows"] = static_cast<int64_t>(
      absl::GetFlag(FLAGS_numerical_parity_min_rows));
  thresholds["min_nontrivial_rows"] = static_cast<int64_t>(
      absl::GetFlag(FLAGS_numerical_parity_min_nontrivial_rows));
  thresholds["max_abs_chosen_logprob_delta"] =
      absl::GetFlag(FLAGS_numerical_parity_max_abs_logprob_delta);
  thresholds["max_full_legal_kl_each_direction"] =
      absl::GetFlag(FLAGS_numerical_parity_max_full_kl);
  thresholds["max_raw_mass_residual"] =
      absl::GetFlag(FLAGS_numerical_parity_max_raw_mass_residual);
  thresholds["min_ratio"] =
      absl::GetFlag(FLAGS_numerical_parity_min_ratio);
  thresholds["max_ratio"] =
      absl::GetFlag(FLAGS_numerical_parity_max_ratio);
  root["thresholds"] = json::Value(thresholds);
  size_t model_file_size = 0;
  const std::string model_path = absl::GetFlag(FLAGS_model_checkpoint);
  json::Object source;
  source["model_path"] = std::filesystem::absolute(model_path).string();
  source["model_sha256"] = ComputeFileSHA256(model_path, &model_file_size);
  source["model_size"] = static_cast<int64_t>(model_file_size);
  source["manifest_path"] = std::filesystem::absolute(source_manifest).string();
  source["manifest_sha256"] = source_manifest_sha256;
  source["global_update"] = source_global_update;
  source["checkpoint_uuid"] = source_checkpoint_uuid;
  size_t binary_size = 0;
  std::error_code ec;
  const std::filesystem::path binary_path =
      std::filesystem::read_symlink("/proc/self/exe", ec);
  if (ec || binary_path.empty()) {
    validity_errors.push_back("could not resolve executed binary path");
    root["status"] = "INVALID";
    root["classification"] = "INVALID";
    json::Array updated_errors;
    for (const std::string& error : validity_errors) {
      updated_errors.emplace_back(error);
    }
    root["hard_validity"].GetObject()["validity_errors"] =
        json::Value(updated_errors);
  } else {
    source["binary_path"] = binary_path.string();
    source["binary_sha256"] =
        ComputeFileSHA256(binary_path.string(), &binary_size);
    source["binary_size"] = static_cast<int64_t>(binary_size);
  }
  root["source"] = json::Value(source);
  json::Object source_code;
  source_code["root"] = source_provenance.root;
  source_code["combined_sha256"] = source_provenance.combined_sha256;
  source_code["canonical_contract"] =
      "fixed ordered relative_path + NUL + lowercase file_sha256 + newline";
  json::Array source_files;
  for (const auto& file : source_provenance.files) {
    json::Object record;
    record["relative_path"] = file.relative_path;
    record["absolute_path"] = file.absolute_path;
    record["size"] = file.size;
    record["sha256"] = file.sha256;
    source_files.emplace_back(json::Value(record));
  }
  source_code["files"] = json::Value(source_files);
  root["source_code"] = json::Value(source_code);

  const std::filesystem::path output(output_path);
  const std::filesystem::path tmp = output_path + ".tmp";
  if (std::filesystem::exists(output) || std::filesystem::exists(tmp)) {
    SpielFatalError("Numerical parity output already exists; refusing overwrite: " +
                    output_path);
  }
  if (!output.parent_path().empty()) {
    std::filesystem::create_directories(output.parent_path());
  }
  {
    std::ofstream out(tmp, std::ios::trunc);
    if (!out) SpielFatalError("Cannot create numerical parity temp artifact");
    out << json::ToString(root, true) << "\n";
    out.flush();
    if (!out) SpielFatalError("Numerical parity artifact write failed");
  }
  std::filesystem::rename(tmp, output);
  return root["status"] == "VALID";
}

bool WritePpoNumericalParityArtifactV5(
    const std::string& output_path, const PpoParityCellResult& cell_a,
    const PpoParityIntegrityStats& a_stats,
    const PpoParityCellResult& cell_r,
    const PpoParityReplayGeometryStats& r_stats,
    const PpoParityCellResult& cell_b,
    const PpoParityCellResult& cell_d,
    const PpoParityOriginalGeometryModelStats& d_stats,
    const PpoParityBatchGeometry& geometry,
    const std::string& geometry_sha256, const CollectResult& collect,
    const std::string& shared_identity_sha256,
    const std::string& model_hash_before,
    const std::string& model_hash_after,
    const std::string& inference_hash_before,
    const std::string& inference_hash_after,
    const std::string& source_manifest,
    const std::string& source_manifest_sha256, int64_t source_global_update,
    const std::string& source_checkpoint_uuid,
    const std::string& config_fingerprint, torch::Device device,
    int64_t obs_size, int64_t action_size, const std::string& command_line,
    const PpoNumericalParitySourceProvenance& source_provenance,
    bool tf32_cublas_before, bool tf32_cudnn_before,
    bool tf32_cublas_after, bool tf32_cudnn_after) {
  const int64_t transitions = static_cast<int64_t>(collect.rollout.size());
  const auto sa = SummarizePpoNumericalParityRows(cell_a.rows);
  const auto sr = SummarizePpoNumericalParityRows(cell_r.rows);
  const auto sb = SummarizePpoNumericalParityRows(cell_b.rows);
  const auto sd = SummarizePpoNumericalParityRows(cell_d.rows);
  auto hex64 = [](const std::string& value) {
    if (value.size() != 64) return false;
    return std::all_of(value.begin(), value.end(), [](char c) {
      return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
    });
  };
  std::vector<std::string> validity_errors = collect.parity_capture_errors;
  validity_errors.insert(validity_errors.end(), geometry.errors.begin(),
                         geometry.errors.end());
  auto require = [&](bool ok, const std::string& error) {
    if (!ok) validity_errors.push_back(error);
  };
  require(collect.episode_ids_unique, "episode IDs are not unique");
  require(static_cast<int64_t>(collect.games) ==
              absl::GetFlag(FLAGS_rollout_games),
          "exact game count mismatch");
  require(transitions >= absl::GetFlag(FLAGS_numerical_parity_min_rows),
          "row floor not met");
  int64_t nontrivial_rows = 0;
  for (const auto& row : cell_a.rows) if (row.legal_count > 1) ++nontrivial_rows;
  require(nontrivial_rows >=
              absl::GetFlag(FLAGS_numerical_parity_min_nontrivial_rows),
          "nontrivial row floor not met");
  require(collect.parity_decisions == transitions &&
              collect.parity_full_width_ok == transitions &&
              collect.parity_full_finite_ok == transitions &&
              collect.parity_legal_ids_unique_in_range == transitions &&
              collect.parity_chosen_once == transitions,
          "capture counters do not cover every transition");
  require(geometry.valid && geometry.rows == transitions &&
              geometry.groups > 0 && geometry.max_batch_size <=
                  absl::GetFlag(FLAGS_eval_batch_size),
          "physical-batch geometry is invalid");
  require(r_stats.before.requests == static_cast<uint64_t>(transitions) &&
              r_stats.before.batches == static_cast<uint64_t>(geometry.groups) &&
              r_stats.before.max_batch_size ==
                  static_cast<uint64_t>(geometry.max_batch_size) &&
              std::abs(r_stats.before.avg_batch_size -
                       static_cast<double>(transitions) / geometry.groups) <
                  1e-12,
          "pre-R evaluator stats do not match captured geometry");
  require(r_stats.after.requests - r_stats.before.requests ==
                  static_cast<uint64_t>(transitions) &&
              r_stats.after.batches - r_stats.before.batches ==
                  static_cast<uint64_t>(geometry.groups) &&
              r_stats.after.max_batch_size == r_stats.before.max_batch_size,
          "post-R evaluator stats deltas do not match exact replay");
  require(r_stats.returned_metadata_exact_rows == transitions &&
              r_stats.raw_legal_exact_rows == transitions &&
              r_stats.cpu_policy_exact_rows == transitions &&
              r_stats.model_forward_calls == geometry.groups &&
              r_stats.captured_raw_sha256 == r_stats.replay_raw_sha256,
          "R exact metadata/raw/policy/hash replay failed");
  require(sa.rows == transitions && sr.rows == transitions &&
              sb.rows == transitions && sd.rows == transitions,
          "A/R/B/D row counts differ");
  require(cell_a.schema_errors.empty() && cell_r.schema_errors.empty() &&
              cell_b.schema_errors.empty() && cell_d.schema_errors.empty(),
          "one or more cells have schema errors");
  require(sa.finite_rows == transitions && sr.finite_rows == transitions &&
              sb.finite_rows == transitions && sd.finite_rows == transitions,
          "one or more cells have incomplete finite rows");
  require(sa.max_abs_chosen_log_prob_delta == 0.0 &&
              sa.max_kl_old_new == 0.0 && sa.max_kl_new_old == 0.0 &&
              sa.ratio_min == 1.0 && sa.ratio_max == 1.0,
          "A is not exact identity");
  require(cell_a.model_forward_calls == 0 && cell_b.model_forward_calls > 0 &&
              cell_d.model_forward_calls == geometry.groups,
          "model-forward call counts are invalid");
  require(cell_b.dense_mask_rows_ok == transitions &&
              cell_d.dense_mask_rows_ok == transitions,
          "B/D dense-mask coverage incomplete");
  require(a_stats.raw_values_total == sa.legal_logits &&
              a_stats.vector_width_rows_ok == transitions &&
              a_stats.behavior_vector_exact_rows == transitions &&
              a_stats.chosen_scalar_exact_rows == transitions,
          "A raw/vector proof incomplete");
  PpoParityV5PrecisionConfig precision_config;
  precision_config.rollout_amp = absl::GetFlag(FLAGS_rollout_amp);
  precision_config.train_amp = absl::GetFlag(FLAGS_train_amp);
  precision_config.allow_tf32 = absl::GetFlag(FLAGS_allow_tf32);
  precision_config.tf32_cublas_before = tf32_cublas_before;
  precision_config.tf32_cudnn_before = tf32_cudnn_before;
  precision_config.tf32_cublas_after = tf32_cublas_after;
  precision_config.tf32_cudnn_after = tf32_cudnn_after;
  std::string precision_error;
  require(ValidatePpoParityV5PrecisionConfig(
              precision_config, &precision_error),
          precision_error.empty() ? "v5 precision runtime contract failed"
                                  : precision_error);
  require(!cell_a.module_train_mode && !cell_a.autocast_enabled &&
              !cell_r.module_train_mode && !cell_r.autocast_enabled &&
              cell_b.module_train_mode && !cell_b.autocast_enabled &&
              cell_d.module_train_mode && !cell_d.autocast_enabled,
          "A/R/B/D module-mode or autocast configuration mismatch");
  for (const auto* cell : {&cell_a, &cell_r, &cell_b, &cell_d}) {
    require(cell->input_dtype == "Float32" &&
                cell->pre_forward_dtype == "Float32",
            "cell input/pre-forward dtype is not Float32");
  }
  require(hex64(geometry_sha256) && hex64(shared_identity_sha256),
          "geometry/shared hash malformed");
  const std::vector<const PpoParityCellResult*> cell_list =
      {&cell_a, &cell_r, &cell_b, &cell_d};
  std::set<std::string> evidence_hashes;
  bool cell_hashes_valid = true;
  for (const auto* cell : cell_list) {
    cell_hashes_valid = cell_hashes_valid &&
        hex64(cell->row_evidence_sha256) && hex64(cell->raw_logit_sha256) &&
        hex64(cell->policy_sha256);
    evidence_hashes.insert(cell->row_evidence_sha256);
  }
  require(cell_hashes_valid && evidence_hashes.size() == cell_list.size(),
          "cell raw/policy/evidence hashes malformed or evidence not distinct");
  const bool shared_rows = PpoParityV5RowsShareIdentities(
      cell_a.rows, cell_r.rows, cell_b.rows, cell_d.rows, transitions);
  require(shared_rows, "A/R/B/D row identities differ");
  require(model_hash_before == model_hash_after &&
              inference_hash_before == inference_hash_after &&
              model_hash_before == inference_hash_before,
          "model immutability/equality failure");

  const bool instrument_valid = validity_errors.empty();
  const auto b_violations = BuildPpoParityViolationMap(cell_b.rows);
  const auto d_violations = BuildPpoParityViolationMap(cell_d.rows);
  const bool b_pass = instrument_valid &&
                      PpoParityViolationMapPasses(b_violations);
  const bool d_pass = instrument_valid &&
                      PpoParityViolationMapPasses(d_violations);
  const auto classification = ClassifyPpoParityV5(
      instrument_valid, b_pass, d_pass);

  json::Object root;
  root["schema"] = "dune_raw_ppo_numerical_parity_v5a";
  root["status"] = instrument_valid ? "VALID" : "INVALID";
  root["classification"] = PpoParityV5ClassificationName(classification);
  root["training_authorized"] = false;
  root["matched_2x2_preflight_admitted"] =
      instrument_valid &&
      classification ==
          PpoParityV5Classification::kFp32Tf32AllowedCandidateAdmitted;
  root["registration_id"] =
      absl::GetFlag(FLAGS_numerical_parity_registration_id);
  root["scope"] = "raw_ppo_only";
  root["epistemic_label"] =
      "no_training_fp32_tf32_allowed_candidate_screen";
  root["optimizer_steps"] = int64_t{0};
  root["optimizer_checkpoint_loaded"] = false;
  root["backward_calls"] = int64_t{0};
  root["training_updates"] = int64_t{0};
  root["command_line"] = command_line;
  root["config_fingerprint"] = config_fingerprint;
  root["device"] = device.str();
  root["rollout_amp"] = absl::GetFlag(FLAGS_rollout_amp);
  root["train_amp"] = absl::GetFlag(FLAGS_train_amp);
  root["allow_tf32"] = absl::GetFlag(FLAGS_allow_tf32);
  root["ppo_minibatch_size"] = static_cast<int64_t>(
      absl::GetFlag(FLAGS_ppo_minibatch_size));
  root["logit_cap"] = absl::GetFlag(FLAGS_logit_cap);
  root["seed"] = static_cast<int64_t>(absl::GetFlag(FLAGS_seed));
  root["threads"] = static_cast<int64_t>(absl::GetFlag(FLAGS_threads));
  root["eval_batch_size"] = static_cast<int64_t>(
      absl::GetFlag(FLAGS_eval_batch_size));
  root["eval_timeout_ms"] = static_cast<int64_t>(
      absl::GetFlag(FLAGS_eval_timeout_ms));
  root["rollout_games"] = static_cast<int64_t>(
      absl::GetFlag(FLAGS_rollout_games));
  root["hard_transition_ceiling"] = static_cast<int64_t>(
      absl::GetFlag(FLAGS_rollout_games)) * 5000;
  root["games_collected"] = static_cast<int64_t>(collect.games);
  root["transitions_collected"] = transitions;
  root["nontrivial_rows"] = nontrivial_rows;
  root["observation_size"] = obs_size;
  root["action_size"] = action_size;
  root["rollout_hash"] = ComputeRolloutHash(collect.rollout);
  root["shared_row_identity_sha256"] = shared_identity_sha256;
  root["physical_batch_geometry_sha256"] = geometry_sha256;
  root["model_state_sha256_before"] = model_hash_before;
  root["model_state_sha256_after"] = model_hash_after;
  root["inference_state_sha256_before"] = inference_hash_before;
  root["inference_state_sha256_after"] = inference_hash_after;
  json::Object thresholds;
  thresholds["min_rows"] = static_cast<int64_t>(
      absl::GetFlag(FLAGS_numerical_parity_min_rows));
  thresholds["min_nontrivial_rows"] = static_cast<int64_t>(
      absl::GetFlag(FLAGS_numerical_parity_min_nontrivial_rows));
  thresholds["max_abs_chosen_logprob_delta"] =
      absl::GetFlag(FLAGS_numerical_parity_max_abs_logprob_delta);
  thresholds["max_full_legal_kl_each_direction"] =
      absl::GetFlag(FLAGS_numerical_parity_max_full_kl);
  thresholds["max_raw_mass_residual"] =
      absl::GetFlag(FLAGS_numerical_parity_max_raw_mass_residual);
  thresholds["min_ratio"] =
      absl::GetFlag(FLAGS_numerical_parity_min_ratio);
  thresholds["max_ratio"] =
      absl::GetFlag(FLAGS_numerical_parity_max_ratio);
  root["thresholds"] = json::Value(thresholds);
  json::Object precision;
  precision["input_dtype"] = "Float32";
  precision["pre_forward_dtype"] = "Float32";
  precision["rollout_autocast"] = absl::GetFlag(FLAGS_rollout_amp);
  precision["learner_autocast"] = absl::GetFlag(FLAGS_train_amp);
  precision["tf32_cublas_before"] = tf32_cublas_before;
  precision["tf32_cudnn_before"] = tf32_cudnn_before;
  precision["tf32_cublas_after"] = tf32_cublas_after;
  precision["tf32_cudnn_after"] = tf32_cudnn_after;
  precision["runtime_state_unchanged"] =
      tf32_cublas_before == tf32_cublas_after &&
      tf32_cudnn_before == tf32_cudnn_after;
  root["precision_runtime"] = json::Value(precision);
  json::Object capture;
  capture["decisions"] = collect.parity_decisions;
  capture["full_width_ok"] = collect.parity_full_width_ok;
  capture["full_finite_ok"] = collect.parity_full_finite_ok;
  capture["legal_ids_unique_in_range"] =
      collect.parity_legal_ids_unique_in_range;
  capture["chosen_once"] = collect.parity_chosen_once;
  capture["raw_values_total"] = a_stats.raw_values_total;
  capture["raw_values_bf16_exact_descriptive"] =
      a_stats.raw_values_bf16_exact;
  root["hard_capture"] = json::Value(capture);

  auto stats_json = [](const EvaluatorStats& stats) {
    json::Object out;
    out["requests"] = static_cast<int64_t>(stats.requests);
    out["batches"] = static_cast<int64_t>(stats.batches);
    out["max_batch_size"] = static_cast<int64_t>(stats.max_batch_size);
    out["mean_batch_size"] = stats.avg_batch_size;
    return out;
  };
  json::Object metadata;
  metadata["groups"] = geometry.groups;
  metadata["rows"] = geometry.rows;
  metadata["max_batch_size"] = geometry.max_batch_size;
  metadata["pre_R_stats"] = json::Value(stats_json(r_stats.before));
  metadata["post_R_stats"] = json::Value(stats_json(r_stats.after));
  metadata["returned_metadata_exact_rows"] =
      r_stats.returned_metadata_exact_rows;
  metadata["R_captured_raw_sha256"] = r_stats.captured_raw_sha256;
  metadata["R_replay_raw_sha256"] = r_stats.replay_raw_sha256;
  json::Array geometry_errors;
  for (const auto& error : validity_errors) geometry_errors.emplace_back(error);
  metadata["validity_errors"] = json::Value(geometry_errors);
  root["physical_batch_metadata"] = json::Value(metadata);

  auto cell_json = [&](const PpoParityCellResult& cell,
                       const std::string& role,
                       const PpoNumericalParitySummary& summary,
                       const PpoParityViolationMap* violations) {
    json::Object out;
    out["name"] = cell.name;
    out["role"] = role;
    out["model_forward_calls"] = cell.model_forward_calls;
    out["dense_mask_rows_ok"] = cell.dense_mask_rows_ok;
    out["module_mode"] = cell.module_train_mode ? "train" : "eval";
    out["autocast_enabled"] = cell.autocast_enabled;
    out["input_dtype"] = cell.input_dtype;
    out["pre_forward_dtype"] = cell.pre_forward_dtype;
    out["wall_time_s"] = cell.wall_time_s;
    out["device_time_s"] = cell.device_time_s;
    out["device_timing_available"] = cell.device_time_s >= 0.0;
    out["row_evidence_sha256"] = cell.row_evidence_sha256;
    out["raw_logit_sha256"] = cell.raw_logit_sha256;
    out["policy_sha256"] = cell.policy_sha256;
    out["overall"] = json::Value(PpoNumericalParitySummaryJson(summary));
    out["splits"] = json::Value(PpoNumericalParitySplitsJson(cell.rows));
    json::Array errors;
    for (const auto& error : cell.schema_errors) errors.emplace_back(error);
    out["schema_errors"] = json::Value(errors);
    if (violations != nullptr) {
      out["threshold_pass"] = PpoParityViolationMapPasses(*violations);
      out["violations"] = json::Value(PpoParityViolationMapJson(*violations));
    }
    return out;
  };
  json::Object cells;
  cells["A"] = json::Value(cell_json(cell_a, "hard_validity", sa, nullptr));
  cells["R"] = json::Value(cell_json(
      cell_r, "live_inference_exact_batch_replay", sr, nullptr));
  cells["B"] = json::Value(cell_json(
      cell_b, "full_fp32_tf32_2048_candidate", sb, &b_violations));
  json::Object d_json = cell_json(
      cell_d, "fp32_tf32_original_batch_geometry", sd, &d_violations);
  d_json["raw_legal_bit_exact_rows"] = d_stats.raw_legal_exact_rows;
  d_json["raw_legal_max_abs_delta"] = d_stats.raw_legal_max_abs_delta;
  cells["D"] = json::Value(d_json);
  root["cells"] = json::Value(cells);
  root["B_D_violation_intersections"] = json::Value(
      PpoParityViolationIntersectionsJson(b_violations, d_violations));

  size_t model_size = 0;
  const std::string model_path = absl::GetFlag(FLAGS_model_checkpoint);
  json::Object source;
  source["model_path"] = std::filesystem::absolute(model_path).string();
  source["model_sha256"] = ComputeFileSHA256(model_path, &model_size);
  source["model_size"] = static_cast<int64_t>(model_size);
  source["manifest_path"] = std::filesystem::absolute(source_manifest).string();
  source["manifest_sha256"] = source_manifest_sha256;
  source["global_update"] = source_global_update;
  source["checkpoint_uuid"] = source_checkpoint_uuid;
  size_t binary_size = 0;
  std::error_code ec;
  const auto binary_path = std::filesystem::read_symlink("/proc/self/exe", ec);
  if (ec || binary_path.empty()) {
    root["status"] = "INVALID";
    root["classification"] = "INVALID";
  } else {
    source["binary_path"] = binary_path.string();
    source["binary_sha256"] =
        ComputeFileSHA256(binary_path.string(), &binary_size);
    source["binary_size"] = static_cast<int64_t>(binary_size);
  }
  root["source"] = json::Value(source);
  json::Object source_code;
  source_code["root"] = source_provenance.root;
  source_code["combined_sha256"] = source_provenance.combined_sha256;
  json::Array source_files;
  for (const auto& file : source_provenance.files) {
    json::Object record;
    record["relative_path"] = file.relative_path;
    record["absolute_path"] = file.absolute_path;
    record["size"] = file.size;
    record["sha256"] = file.sha256;
    source_files.emplace_back(json::Value(record));
  }
  source_code["files"] = json::Value(source_files);
  root["source_code"] = json::Value(source_code);
  const std::filesystem::path output(output_path);
  const std::filesystem::path tmp = output_path + ".tmp";
  if (std::filesystem::exists(output) || std::filesystem::exists(tmp)) {
    SpielFatalError("Numerical parity output already exists; refusing overwrite: " +
                    output_path);
  }
  if (!output.parent_path().empty()) {
    std::filesystem::create_directories(output.parent_path());
  }
  {
    std::ofstream out(tmp, std::ios::trunc);
    if (!out) SpielFatalError("Cannot create numerical parity temp artifact");
    out << json::ToString(root, true) << "\n";
    out.flush();
    if (!out) SpielFatalError("Numerical parity artifact write failed");
  }
  std::filesystem::rename(tmp, output);
  return root["status"] == "VALID";
}

bool WriteVrpoCaptureArtifact(
    const std::string& output_path,
    const std::vector<VrpoCapturedEpisode>& episodes,
    const CollectResult& collect, const std::string& command_line,
    const PpoNumericalParitySourceProvenance& source_provenance,
    const std::string& source_manifest,
    const std::string& source_manifest_sha256, int64_t source_global_update,
    const std::string& source_checkpoint_uuid,
    const std::string& config_fingerprint,
    const std::string& model_hash_before,
    const std::string& model_hash_after,
    const std::string& inference_hash_before,
    const std::string& inference_hash_after,
    bool tf32_cublas_before, bool tf32_cudnn_before,
    bool tf32_cublas_after, bool tf32_cudnn_after,
    bool optimizer_constructed) {
  std::vector<std::string> errors;
  auto require = [&](bool ok, const std::string& message) {
    if (!ok) errors.push_back(message);
  };
  require(!episodes.empty(), "no complete captured episodes");
  require(episodes.size() == static_cast<size_t>(collect.games),
          "captured episode count differs from collected games");
  require(episodes.size() ==
              static_cast<size_t>(absl::GetFlag(FLAGS_rollout_games)),
          "captured episode count differs from registered games");
  std::string id_error;
  require(ValidateVrpoSortedEpisodeIds(
              episodes, absl::GetFlag(FLAGS_start_episode_id),
              absl::GetFlag(FLAGS_rollout_games), &id_error),
          id_error);
  require(!optimizer_constructed, "optimizer was constructed");
  require(model_hash_before == model_hash_after &&
              inference_hash_before == inference_hash_after &&
              model_hash_before == inference_hash_before,
          "model immutability/equality failed");
  require(!absl::GetFlag(FLAGS_rollout_amp) &&
              !absl::GetFlag(FLAGS_train_amp) &&
              absl::GetFlag(FLAGS_allow_tf32) && tf32_cublas_before &&
              tf32_cudnn_before && tf32_cublas_after && tf32_cudnn_after,
          "FP32/TF32 runtime contract failed");

  int64_t total_rows = 0;
  int64_t sampler_exact_rows = 0;
  size_t rollout_cursor = 0;
  std::set<uint64_t> episode_ids;
  json::Array episode_json;
  for (const auto& episode : episodes) {
    std::string validation_error;
    const bool episode_valid =
        ValidateVrpoCapturedEpisode(episode, &validation_error);
    require(episode_valid,
            "captured episode invalid: " + validation_error);
    std::string metadata_error;
    const bool metadata_valid = ValidateVrpoCaptureRewardMetadata(
        episode, absl::GetFlag(FLAGS_reward_scale),
        absl::GetFlag(FLAGS_gamma), absl::GetFlag(FLAGS_gae_lambda),
        absl::GetFlag(FLAGS_vrpo_probability_tolerance), &metadata_error);
    require(metadata_valid, metadata_error);
    require(episode_ids.insert(episode.episode_id).second,
            "duplicate captured episode ID");
    VrpoSeatValues reward_sums = {0.0, 0.0, 0.0, 0.0};
    int64_t terminal_rows = 0;
    for (const auto& row : episode.rows) {
      for (int seat = 0; seat < kVrpoNumSeats; ++seat) {
        reward_sums[seat] += row.rewards[seat];
      }
      if (row.terminal_after) ++terminal_rows;
      if (rollout_cursor < collect.rollout.size()) {
        const PpoTransition& transition = collect.rollout[rollout_cursor];
        VrpoRolloutPairingView pairing;
        pairing.episode_id = transition.episode_id;
        pairing.actor = transition.player_id;
        pairing.actor_observation = &transition.state;
        pairing.legal_actions = &transition.legal_actions;
        pairing.action = transition.action;
        pairing.chosen_log_probability = transition.old_log_prob;
        std::string pairing_error;
        if (ValidateVrpoCaptureRolloutPairing(
                row, pairing, &pairing_error)) {
          ++sampler_exact_rows;
        } else {
          errors.push_back("capture/rollout pairing failed: " + pairing_error);
        }
      }
      ++rollout_cursor;
    }
    total_rows += episode.rows.size();
    json::Object record;
    record["episode_id"] = static_cast<int64_t>(episode.episode_id);
    record["rows"] = static_cast<int64_t>(episode.rows.size());
    record["first_global_row_index"] = int64_t{0};
    record["last_global_row_index"] =
        static_cast<int64_t>(episode.rows.size()) - 1;
    record["capture_sha256"] = episode.capture_sha256;
    record["terminal_rows"] = terminal_rows;
    json::Array rewards;
    for (double value : reward_sums) rewards.emplace_back(value);
    record["absolute_reward_sums"] = std::move(rewards);
    json::Array terminal_returns;
    for (double value : episode.terminal_returns) {
      terminal_returns.emplace_back(value);
    }
    record["terminal_returns"] = std::move(terminal_returns);
    bool reward_conservation = episode_valid && metadata_valid;
    for (int seat = 0; seat < kVrpoNumSeats; ++seat) {
      reward_conservation = reward_conservation &&
          reward_sums[seat] ==
              std::clamp(episode.terminal_returns[seat] /
                             episode.reward_scale,
                         -1.0, 1.0);
    }
    record["reward_conservation"] = reward_conservation;
    require(reward_conservation, "episode reward conservation failed");
    episode_json.emplace_back(std::move(record));
  }
  require(rollout_cursor == collect.rollout.size(),
          "capture and rollout row counts differ");
  require(sampler_exact_rows == total_rows,
          "sampler chosen scalar/probability checks incomplete");
  require(collect.episode_ids_unique, "rollout episode IDs are not unique");

  json::Object root;
  root["schema"] = "dune_vrpo_diagnostics_capture_v1";
  root["registration_id"] =
      absl::GetFlag(FLAGS_vrpo_capture_registration_id);
  root["scope"] = "vrpo_capture_only";
  root["training_authorized"] = false;
  root["optimizer_constructed"] = optimizer_constructed;
  root["optimizer_steps"] = int64_t{0};
  root["backward_calls"] = int64_t{0};
  root["training_updates"] = int64_t{0};
  root["q_forward_calls"] = int64_t{0};
  root["command_line"] = command_line;
  root["config_fingerprint"] = config_fingerprint;
  root["games_collected"] = static_cast<int64_t>(collect.games);
  root["rollout_games"] =
      static_cast<int64_t>(absl::GetFlag(FLAGS_rollout_games));
  root["threads"] = static_cast<int64_t>(absl::GetFlag(FLAGS_threads));
  root["seed"] = static_cast<int64_t>(absl::GetFlag(FLAGS_seed));
  root["start_episode_id"] =
      static_cast<int64_t>(absl::GetFlag(FLAGS_start_episode_id));
  root["end_episode_id_inclusive"] =
      static_cast<int64_t>(absl::GetFlag(FLAGS_start_episode_id)) +
      static_cast<int64_t>(absl::GetFlag(FLAGS_rollout_games)) - 1;
  root["episode_ids_unique"] = collect.episode_ids_unique;
  root["captured_episodes"] = static_cast<int64_t>(episodes.size());
  root["captured_rows"] = total_rows;
  root["sampler_exact_rows"] = sampler_exact_rows;
  const VrpoExactNumericStrings exact_numeric =
      MakeVrpoRegisteredExactNumericStrings(
          absl::GetFlag(FLAGS_reward_scale), absl::GetFlag(FLAGS_gamma),
          absl::GetFlag(FLAGS_gae_lambda));
  std::string exact_numeric_error;
  require(PopulateVrpoExactNumericProvenance(
              &root, exact_numeric, absl::GetFlag(FLAGS_reward_scale),
              absl::GetFlag(FLAGS_gamma),
              absl::GetFlag(FLAGS_gae_lambda),
              absl::GetFlag(FLAGS_vrpo_probability_tolerance),
              /*include_q_tolerances=*/false,
              kVrpoRegisteredQAgreementAbsTolerance,
              kVrpoRegisteredQAgreementRelTolerance,
              &exact_numeric_error),
          exact_numeric_error);
  root["episodes"] = std::move(episode_json);
  root["capture_schema_label"] = kVrpoCaptureSchemaLabel;
  root["capture_schema_sha256"] = kVrpoCaptureSchemaSha256;
  root["central_schema_label"] =
      dune_imperium::kVrpoCentralCriticTensorSchemaLabel;
  root["central_schema_sha256"] =
      dune_imperium::kVrpoCentralCriticTensorSchemaSha256;
  root["reward_convention_label"] =
      kVrpoZeroShapingRewardConventionLabel;
  root["reward_convention_sha256"] =
      kVrpoZeroShapingRewardConventionSha256;
  root["rollout_amp"] = absl::GetFlag(FLAGS_rollout_amp);
  root["train_amp"] = absl::GetFlag(FLAGS_train_amp);
  root["allow_tf32"] = absl::GetFlag(FLAGS_allow_tf32);
  json::Object runtime;
  runtime["tf32_cublas_before"] = tf32_cublas_before;
  runtime["tf32_cublas_after"] = tf32_cublas_after;
  runtime["tf32_cudnn_before"] = tf32_cudnn_before;
  runtime["tf32_cudnn_after"] = tf32_cudnn_after;
  root["precision_runtime"] = std::move(runtime);
  root["model_state_sha256_before"] = model_hash_before;
  root["model_state_sha256_after"] = model_hash_after;
  root["inference_state_sha256_before"] = inference_hash_before;
  root["inference_state_sha256_after"] = inference_hash_after;

  size_t model_size = 0;
  const std::string model_path = absl::GetFlag(FLAGS_model_checkpoint);
  json::Object source;
  source["model_path"] = std::filesystem::absolute(model_path).string();
  source["model_sha256"] = ComputeFileSHA256(model_path, &model_size);
  source["model_size"] = static_cast<int64_t>(model_size);
  source["manifest_path"] = std::filesystem::absolute(source_manifest).string();
  source["manifest_sha256"] = source_manifest_sha256;
  source["global_update"] = source_global_update;
  source["checkpoint_uuid"] = source_checkpoint_uuid;
  size_t binary_size = 0;
  std::error_code ec;
  const auto binary_path = std::filesystem::read_symlink("/proc/self/exe", ec);
  require(!ec && !binary_path.empty(), "could not resolve binary path");
  if (!ec && !binary_path.empty()) {
    source["binary_path"] = binary_path.string();
    source["binary_sha256"] =
        ComputeFileSHA256(binary_path.string(), &binary_size);
    source["binary_size"] = static_cast<int64_t>(binary_size);
  }
  root["source"] = std::move(source);
  json::Object source_code;
  source_code["root"] = source_provenance.root;
  source_code["combined_sha256"] = source_provenance.combined_sha256;
  json::Array source_files;
  for (const auto& file : source_provenance.files) {
    json::Object record;
    record["relative_path"] = file.relative_path;
    record["absolute_path"] = file.absolute_path;
    record["size"] = file.size;
    record["sha256"] = file.sha256;
    source_files.emplace_back(std::move(record));
  }
  source_code["files"] = std::move(source_files);
  root["source_code"] = std::move(source_code);
  json::Array validation_errors;
  for (const auto& error : errors) validation_errors.emplace_back(error);
  root["validation_errors"] = std::move(validation_errors);
  // Status is deliberately assigned after every possible require/reject.
  const bool valid = errors.empty();
  root["classification"] = valid ? "VALID_CAPTURE" : "INVALID";
  root["status"] = valid ? "VALID" : "INVALID";

  const std::filesystem::path output(output_path);
  const std::filesystem::path tmp = output_path + ".tmp";
  if (std::filesystem::exists(output) || std::filesystem::exists(tmp)) {
    SpielFatalError("VRPO capture output already exists");
  }
  if (!output.parent_path().empty()) {
    std::filesystem::create_directories(output.parent_path());
  }
  {
    std::ofstream stream(tmp, std::ios::trunc);
    if (!stream) SpielFatalError("cannot create VRPO capture temp artifact");
    stream << json::ToString(root, true) << "\n";
    stream.flush();
    if (!stream) SpielFatalError("VRPO capture artifact write failed");
  }
  std::filesystem::rename(tmp, output);
  return valid;
}

struct VrpoValueStats {
  int64_t count = 0;
  double min = 0.0;
  double max = 0.0;
  double mean = 0.0;
  double stddev = 0.0;
};

VrpoValueStats SummarizeVrpoValues(const std::vector<double>& values) {
  VrpoValueStats out;
  if (values.empty()) return out;
  out.count = static_cast<int64_t>(values.size());
  out.min = *std::min_element(values.begin(), values.end());
  out.max = *std::max_element(values.begin(), values.end());
  double sum = 0.0;
  for (double value : values) sum += value;
  out.mean = sum / static_cast<double>(values.size());
  double squared = 0.0;
  for (double value : values) {
    const double delta = value - out.mean;
    squared += delta * delta;
  }
  out.stddev = std::sqrt(squared / static_cast<double>(values.size()));
  return out;
}

json::Object VrpoValueStatsJson(const VrpoValueStats& stats) {
  json::Object out;
  out["count"] = stats.count;
  out["min"] = stats.min;
  out["max"] = stats.max;
  out["mean"] = stats.mean;
  out["stddev"] = stats.stddev;
  return out;
}

struct VrpoQPreflightResult {
  bool valid = false;
  bool pre_q_gate_valid = false;
  bool q_constructed = false;
  std::vector<std::string> errors;
  int64_t pre_q_captured_rows = 0;
  int64_t pre_q_rollout_rows = 0;
  int64_t pre_q_paired_rows = 0;
  int64_t q_constructor_calls = 0;
  int64_t rows = 0;
  int64_t legal_values = 0;
  int64_t forced_rows = 0;
  int64_t nontrivial_rows = 0;
  std::array<int64_t, kVrpoNumSeats> actor_rows = {0, 0, 0, 0};
  int64_t q_forward_calls = 0;
  int64_t max_chunk_rows = 0;
  int64_t nonfinite_values = 0;
  int64_t gpu_peak_allocated_increment_bytes = 0;
  int64_t gpu_peak_reserved_increment_bytes = 0;
  std::string q_parameter_sha256_before;
  std::string q_parameter_sha256_after;
  std::string legal_q_evidence_sha256;
  std::string scalar_reference_sha256;
  std::string tensor_reference_sha256;
  std::string input_dtype = "Float32";
  std::string output_dtype = "Float32";
  std::string device;
  bool autocast_enabled = false;
  VrpoReferenceAgreement agreement;
  VrpoValueStats q_values;
  VrpoValueStats chosen_q_values;
  VrpoValueStats v_values;
  VrpoValueStats delta_values;
  VrpoValueStats g_values;
  VrpoValueStats q_target_values;
  VrpoValueStats actor_advantages;
  double pack_h2d_s = 0.0;
  double q_forward_s = 0.0;
  double legal_gather_d2h_s = 0.0;
  double scalar_reference_s = 0.0;
  double tensor_reference_s = 0.0;
  double total_s = 0.0;
};

struct VrpoCudaMemoryStats {
  int64_t allocated = 0;
  int64_t reserved = 0;
  int64_t peak_allocated = 0;
  int64_t peak_reserved = 0;
};

VrpoCudaMemoryStats ReadVrpoCudaMemory(torch::Device device) {
  VrpoCudaMemoryStats out;
  if (!device.is_cuda()) return out;
  const c10::DeviceIndex index =
      device.has_index() ? device.index() : c10::cuda::current_device();
  const auto stats = c10::cuda::CUDACachingAllocator::getDeviceStats(index);
  constexpr size_t aggregate = static_cast<size_t>(
      c10::CachingAllocator::StatType::AGGREGATE);
  out.allocated = stats.allocated_bytes[aggregate].current;
  out.reserved = stats.reserved_bytes[aggregate].current;
  out.peak_allocated = stats.allocated_bytes[aggregate].peak;
  out.peak_reserved = stats.reserved_bytes[aggregate].peak;
  return out;
}

VrpoQPreflightResult RunVrpoQReferencePreflight(
    const VrpoCapturedEpisode& episode, torch::Device device,
    uint64_t q_init_seed, int chunk_rows, double abs_tolerance,
    double rel_tolerance, int64_t gpu_peak_increment_limit_bytes) {
  VrpoQPreflightResult out;
  out.device = device.str();
  const auto total_start = std::chrono::steady_clock::now();
  auto reject = [&](const std::string& message) { out.errors.push_back(message); };
  std::string validation_error;
  if (!ValidateVrpoCapturedEpisode(episode, &validation_error)) {
    reject("captured episode invalid before Q: " + validation_error);
    return out;
  }
  std::vector<std::pair<size_t, size_t>> ranges;
  if (!BuildVrpoChunkRanges(
          episode.rows.size(), chunk_rows, &ranges, &validation_error)) {
    reject(validation_error);
    return out;
  }

  const VrpoCudaMemoryStats memory_before = ReadVrpoCudaMemory(device);
  if (device.is_cuda()) {
    const c10::DeviceIndex index =
        device.has_index() ? device.index() : c10::cuda::current_device();
    c10::cuda::CUDACachingAllocator::resetPeakStats(index);
  }

  auto q_model = std::make_shared<DuneVrpoQNetImpl>(q_init_seed);
  out.q_constructed = true;
  out.q_constructor_calls = VrpoQConstructorCalls();
  if (!VrpoQModuleParameterSha256(
          *q_model, &out.q_parameter_sha256_before, &validation_error)) {
    reject("Q initialization hash failed: " + validation_error);
    return out;
  }
  q_model->to(device);
  q_model->eval();

  std::vector<std::vector<VrpoActorRelativeSeatValues>> relative_q(
      episode.rows.size());
  std::vector<double> all_q;
  std::vector<double> chosen_q;
  out.rows = static_cast<int64_t>(episode.rows.size());
  for (const auto& row : episode.rows) {
    out.legal_values += static_cast<int64_t>(row.legal_actions.size()) *
                        kVrpoNumSeats;
    if (row.legal_actions.size() == 1) ++out.forced_rows;
    else ++out.nontrivial_rows;
    ++out.actor_rows[row.actor];
  }
  all_q.reserve(static_cast<size_t>(out.legal_values));
  chosen_q.reserve(episode.rows.size() * kVrpoNumSeats);

  {
    c10::InferenceMode inference_guard;
    torch::NoGradGuard no_grad;
    AutocastGuard autocast_guard(device.type(), false);
    out.autocast_enabled =
        device.is_cuda()
            ? at::autocast::is_autocast_enabled(at::kCUDA)
            : at::autocast::is_autocast_enabled(at::kCPU);
    for (const auto& range : ranges) {
      out.max_chunk_rows = std::max<int64_t>(
          out.max_chunk_rows, static_cast<int64_t>(range.second));
      const auto pack_start = std::chrono::steady_clock::now();
      torch::Tensor cpu = torch::empty(
          {static_cast<int64_t>(range.second),
           dune_imperium::kVrpoCentralCriticTensorSize},
          torch::TensorOptions().dtype(torch::kFloat32));
      float* destination = cpu.data_ptr<float>();
      for (size_t local = 0; local < range.second; ++local) {
        const auto& central = episode.rows[range.first + local].central_tensor;
        std::memcpy(
            destination + local * dune_imperium::kVrpoCentralCriticTensorSize,
            central.data(), central.size() * sizeof(float));
      }
      torch::Tensor input = cpu.to(device);
      out.pack_h2d_s += std::chrono::duration<double>(
          std::chrono::steady_clock::now() - pack_start).count();

      const auto forward_start = std::chrono::steady_clock::now();
      torch::Tensor dense_q;
      std::string forward_error;
      if (!q_model->ForwardChecked(input, &dense_q, &forward_error)) {
        reject("Q forward failed: " + forward_error);
        break;
      }
      ++out.q_forward_calls;
      if (dense_q.scalar_type() != torch::kFloat32) {
        reject("Q output dtype is not Float32");
        break;
      }
      out.q_forward_s += std::chrono::duration<double>(
          std::chrono::steady_clock::now() - forward_start).count();

      const auto gather_start = std::chrono::steady_clock::now();
      for (size_t local = 0; local < range.second; ++local) {
        const size_t global = range.first + local;
        const VrpoCapturedRow& row = episode.rows[global];
        torch::Tensor legal_cpu = torch::from_blob(
            const_cast<Action*>(row.legal_actions.data()),
            {static_cast<int64_t>(row.legal_actions.size())},
            torch::TensorOptions().dtype(torch::kInt64)).clone();
        torch::Tensor gathered = dense_q[static_cast<int64_t>(local)]
            .index_select(0, legal_cpu.to(device))
            .to(torch::kCPU)
            .to(torch::kFloat64)
            .contiguous();
        relative_q[global].resize(row.legal_actions.size());
        const double* values = gathered.data_ptr<double>();
        for (size_t a = 0; a < row.legal_actions.size(); ++a) {
          for (int slot = 0; slot < kVrpoNumSeats; ++slot) {
            const double value = values[a * kVrpoNumSeats + slot];
            relative_q[global][a].slots[slot] = value;
            all_q.push_back(value);
            if (!std::isfinite(value)) ++out.nonfinite_values;
            if (static_cast<int>(a) == row.chosen_index) {
              chosen_q.push_back(value);
            }
          }
        }
      }
      out.legal_gather_d2h_s += std::chrono::duration<double>(
          std::chrono::steady_clock::now() - gather_start).count();
    }
  }

  if (out.errors.empty() && out.nonfinite_values == 0) {
    std::string evidence_error;
    if (!VrpoLegalQEvidenceSha256(
            episode, relative_q, &out.legal_q_evidence_sha256,
            &evidence_error)) {
      reject(evidence_error);
    }
  }
  std::vector<VrpoTimelineRow> timeline;
  if (out.errors.empty() &&
      !VrpoCapturedEpisodeToTimeline(
          episode, relative_q, &timeline, &validation_error)) {
    reject(validation_error);
  }
  VrpoReferenceTrace scalar;
  if (out.errors.empty()) {
    const auto start = std::chrono::steady_clock::now();
    if (!ComputeVrpoExpectedSarsaLambdaReference(
            timeline, episode.gamma, episode.lambda, &scalar,
            &validation_error, episode.probability_tolerance)) {
      reject("scalar reference failed: " + validation_error);
    }
    out.scalar_reference_s = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - start).count();
    out.scalar_reference_sha256 = scalar.canonical_sha256;
  }
  VrpoTensorReferenceTrace tensor;
  if (out.errors.empty()) {
    const auto start = std::chrono::steady_clock::now();
    if (!ComputeVrpoExpectedSarsaLambdaTensorReference(
            timeline, episode.gamma, episode.lambda, &tensor,
            &validation_error, episode.probability_tolerance)) {
      reject("tensor reference failed: " + validation_error);
    }
    out.tensor_reference_s = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - start).count();
    out.tensor_reference_sha256 = tensor.canonical_sha256;
  }
  if (out.errors.empty() &&
      !CompareVrpoReferenceTraces(
          scalar, tensor, abs_tolerance, rel_tolerance,
          &out.agreement, &validation_error)) {
    reject(validation_error);
  }
  if (!VrpoQModuleParameterSha256(
          *q_model, &out.q_parameter_sha256_after, &validation_error)) {
    reject("post-forward Q hash failed: " + validation_error);
  }
  if (out.q_parameter_sha256_before != out.q_parameter_sha256_after) {
    reject("Q parameters changed during preflight");
  }

  const VrpoCudaMemoryStats memory_after = ReadVrpoCudaMemory(device);
  if (device.is_cuda()) {
    out.gpu_peak_allocated_increment_bytes = std::max<int64_t>(
        0, memory_after.peak_allocated - memory_before.allocated);
    out.gpu_peak_reserved_increment_bytes = std::max<int64_t>(
        0, memory_after.peak_reserved - memory_before.reserved);
    if (out.gpu_peak_allocated_increment_bytes >
            gpu_peak_increment_limit_bytes ||
        out.gpu_peak_reserved_increment_bytes >
            gpu_peak_increment_limit_bytes) {
      reject("Q preflight GPU memory ceiling exceeded");
    }
  }

  std::vector<double> v_values, delta_values, g_values, target_values,
      advantages;
  for (const auto& row : scalar.rows) {
    v_values.insert(v_values.end(), row.v.begin(), row.v.end());
    delta_values.insert(delta_values.end(), row.delta.begin(), row.delta.end());
    g_values.insert(g_values.end(), row.g.begin(), row.g.end());
    target_values.insert(target_values.end(), row.q_target.begin(),
                         row.q_target.end());
    advantages.push_back(row.actor_advantage);
  }
  out.q_values = SummarizeVrpoValues(all_q);
  out.chosen_q_values = SummarizeVrpoValues(chosen_q);
  out.v_values = SummarizeVrpoValues(v_values);
  out.delta_values = SummarizeVrpoValues(delta_values);
  out.g_values = SummarizeVrpoValues(g_values);
  out.q_target_values = SummarizeVrpoValues(target_values);
  out.actor_advantages = SummarizeVrpoValues(advantages);
  out.total_s = std::chrono::duration<double>(
      std::chrono::steady_clock::now() - total_start).count();
  const int64_t expected_calls = static_cast<int64_t>(ranges.size());
  if (out.q_forward_calls != expected_calls) {
    reject("Q forward call count differs from chunk partition");
  }
  out.valid = out.errors.empty();
  out.q_constructor_calls = VrpoQConstructorCalls();
  out.q_forward_calls = VrpoQForwardCheckedCalls();
  return out;
}

bool WriteVrpoQPreflightArtifact(
    const std::string& output_path,
    const std::vector<VrpoCapturedEpisode>& episodes,
    const CollectResult& collect, const VrpoQPreflightResult& result,
    const std::string& command_line,
    const PpoNumericalParitySourceProvenance& source_provenance,
    const std::string& source_manifest,
    const std::string& source_manifest_sha256, int64_t source_global_update,
    const std::string& source_checkpoint_uuid,
    const std::string& config_fingerprint,
    const std::string& model_hash_before,
    const std::string& model_hash_after,
    const std::string& inference_hash_before,
    const std::string& inference_hash_after,
    bool tf32_cublas_before, bool tf32_cudnn_before,
    bool tf32_cublas_after, bool tf32_cudnn_after,
    bool optimizer_constructed) {
  std::vector<std::string> errors = result.errors;
  auto require = [&](bool ok, const std::string& message) {
    if (!ok) errors.push_back(message);
  };
  require(episodes.size() == 1 && collect.games == 1,
          "Q preflight requires one complete game");
  std::string validation_error;
  require(ValidateVrpoSortedEpisodeIds(
              episodes, absl::GetFlag(FLAGS_start_episode_id), 1,
              &validation_error),
          validation_error);
  int64_t paired_rows = 0;
  size_t rollout_cursor = 0;
  for (const auto& episode : episodes) {
    std::string episode_error;
    require(ValidateVrpoCapturedEpisode(episode, &episode_error),
            "captured episode invalid: " + episode_error);
    for (const auto& row : episode.rows) {
      if (rollout_cursor < collect.rollout.size()) {
        const PpoTransition& transition = collect.rollout[rollout_cursor];
        VrpoRolloutPairingView pairing;
        pairing.episode_id = transition.episode_id;
        pairing.actor = transition.player_id;
        pairing.actor_observation = &transition.state;
        pairing.legal_actions = &transition.legal_actions;
        pairing.action = transition.action;
        pairing.chosen_log_probability = transition.old_log_prob;
        std::string pairing_error;
        if (ValidateVrpoCaptureRolloutPairing(row, pairing, &pairing_error)) {
          ++paired_rows;
        } else {
          errors.push_back("capture/rollout pairing failed: " + pairing_error);
        }
      }
      ++rollout_cursor;
    }
  }
  require(rollout_cursor == collect.rollout.size() &&
              paired_rows == result.rows,
          "Q preflight capture/rollout row coverage differs");
  require(result.pre_q_gate_valid &&
              result.pre_q_captured_rows == result.pre_q_rollout_rows &&
              result.pre_q_paired_rows == result.pre_q_captured_rows,
          "pre-Q exhaustive capture/rollout gate failed");
  require(result.q_constructed && result.q_constructor_calls == 1,
          "Q construction count is invalid");
  require(result.valid && result.rows > 0 && result.legal_values > 0 &&
              result.nonfinite_values == 0,
          "Q/reference computation is incomplete or invalid");
  require(result.q_forward_calls > 0 && result.max_chunk_rows > 0 &&
              result.max_chunk_rows <= kVrpoQPreflightMaxChunkRows,
          "Q forward/chunk counters are invalid");
  require(result.agreement.mismatch_count == 0 &&
              result.agreement.compared_values == result.rows * 17,
          "scalar/tensor recurrence agreement is incomplete");
  require(result.q_parameter_sha256_before.size() == 64 &&
              result.q_parameter_sha256_before ==
                  result.q_parameter_sha256_after &&
              result.legal_q_evidence_sha256.size() == 64 &&
              result.scalar_reference_sha256.size() == 64 &&
              result.tensor_reference_sha256.size() == 64,
          "Q/reference evidence hashes are invalid");
  require(!optimizer_constructed, "optimizer was constructed");
  require(model_hash_before == model_hash_after &&
              inference_hash_before == inference_hash_after &&
              model_hash_before == inference_hash_before,
          "actor model immutability/equality failed");
  require(!absl::GetFlag(FLAGS_rollout_amp) &&
              !absl::GetFlag(FLAGS_train_amp) &&
              absl::GetFlag(FLAGS_allow_tf32) && tf32_cublas_before &&
              tf32_cudnn_before && tf32_cublas_after && tf32_cudnn_after,
          "FP32/TF32 runtime contract failed");

  json::Object root;
  root["schema"] = "dune_vrpo_q_reference_preflight_v1";
  root["registration_id"] =
      absl::GetFlag(FLAGS_vrpo_q_preflight_registration_id);
  root["scope"] = "vrpo_q_reference_preflight_only";
  root["epistemic_label"] = "no_training_q_reference_preflight";
  root["training_authorized"] = false;
  root["optimizer_constructed"] = optimizer_constructed;
  root["optimizer_steps"] = int64_t{0};
  root["backward_calls"] = int64_t{0};
  root["training_updates"] = int64_t{0};
  root["pre_q_gate_valid"] = result.pre_q_gate_valid;
  root["pre_q_captured_rows"] = result.pre_q_captured_rows;
  root["pre_q_rollout_rows"] = result.pre_q_rollout_rows;
  root["pre_q_paired_rows"] = result.pre_q_paired_rows;
  root["q_constructed"] = result.q_constructed;
  root["q_constructor_calls"] = result.q_constructor_calls;
  root["command_line"] = command_line;
  root["config_fingerprint"] = config_fingerprint;
  root["games_collected"] = static_cast<int64_t>(collect.games);
  root["threads"] = static_cast<int64_t>(absl::GetFlag(FLAGS_threads));
  root["seed"] = static_cast<int64_t>(absl::GetFlag(FLAGS_seed));
  root["episode_id"] =
      static_cast<int64_t>(absl::GetFlag(FLAGS_start_episode_id));
  root["captured_rows"] = result.rows;
  root["paired_rows"] = paired_rows;
  root["legal_q_values"] = result.legal_values;
  root["forced_rows"] = result.forced_rows;
  root["nontrivial_rows"] = result.nontrivial_rows;
  json::Array actor_rows;
  for (int64_t count : result.actor_rows) actor_rows.emplace_back(count);
  root["actor_rows_absolute_seat"] = std::move(actor_rows);
  root["q_init_seed"] =
      static_cast<int64_t>(absl::GetFlag(FLAGS_vrpo_q_init_seed));
  root["q_chunk_rows"] =
      static_cast<int64_t>(absl::GetFlag(FLAGS_vrpo_q_chunk_rows));
  root["q_forward_calls"] = result.q_forward_calls;
  root["q_max_chunk_rows"] = result.max_chunk_rows;
  root["q_input_dtype"] = result.input_dtype;
  root["q_output_dtype"] = result.output_dtype;
  root["q_device"] = result.device;
  root["q_input_dim"] =
      static_cast<int64_t>(dune_imperium::kVrpoCentralCriticTensorSize);
  root["q_hidden_dim"] = int64_t{kVrpoQHiddenDim};
  root["q_residual_blocks"] = int64_t{kVrpoQNumResBlocks};
  root["q_action_dim"] = int64_t{kVrpoDuneActionDim};
  root["q_reward_perspectives"] = int64_t{kVrpoNumSeats};
  root["q_output_order"] = "actor_relative_slots_0_1_2_3";
  root["q_autocast_enabled"] = result.autocast_enabled;
  root["q_nonfinite_values"] = result.nonfinite_values;
  root["q_parameter_sha256_before"] = result.q_parameter_sha256_before;
  root["q_parameter_sha256_after"] = result.q_parameter_sha256_after;
  root["legal_q_evidence_sha256"] = result.legal_q_evidence_sha256;
  root["scalar_reference_sha256"] = result.scalar_reference_sha256;
  root["tensor_reference_sha256"] = result.tensor_reference_sha256;
  root["capture_sha256"] =
      episodes.empty() ? std::string() : episodes.front().capture_sha256;
  root["central_schema_sha256"] =
      dune_imperium::kVrpoCentralCriticTensorSchemaSha256;
  root["capture_schema_sha256"] = kVrpoCaptureSchemaSha256;
  root["reward_convention_sha256"] =
      kVrpoZeroShapingRewardConventionSha256;
  const VrpoExactNumericStrings exact_numeric =
      MakeVrpoRegisteredExactNumericStrings(
          absl::GetFlag(FLAGS_reward_scale), absl::GetFlag(FLAGS_gamma),
          absl::GetFlag(FLAGS_gae_lambda));
  std::string exact_numeric_error;
  require(PopulateVrpoExactNumericProvenance(
              &root, exact_numeric, absl::GetFlag(FLAGS_reward_scale),
              absl::GetFlag(FLAGS_gamma),
              absl::GetFlag(FLAGS_gae_lambda),
              absl::GetFlag(FLAGS_vrpo_probability_tolerance),
              /*include_q_tolerances=*/true,
              absl::GetFlag(FLAGS_vrpo_q_agreement_abs_tolerance),
              absl::GetFlag(FLAGS_vrpo_q_agreement_rel_tolerance),
              &exact_numeric_error),
          exact_numeric_error);
  json::Object agreement;
  agreement["compared_values"] = result.agreement.compared_values;
  agreement["mismatch_count"] = result.agreement.mismatch_count;
  agreement["max_abs_v"] = result.agreement.max_abs_v;
  agreement["max_abs_delta"] = result.agreement.max_abs_delta;
  agreement["max_abs_g"] = result.agreement.max_abs_g;
  agreement["max_abs_q_target"] = result.agreement.max_abs_q_target;
  agreement["max_abs_actor_advantage"] =
      result.agreement.max_abs_actor_advantage;
  root["scalar_tensor_agreement"] = std::move(agreement);
  json::Object summaries;
  summaries["q"] = VrpoValueStatsJson(result.q_values);
  summaries["chosen_q"] = VrpoValueStatsJson(result.chosen_q_values);
  summaries["v"] = VrpoValueStatsJson(result.v_values);
  summaries["delta"] = VrpoValueStatsJson(result.delta_values);
  summaries["g"] = VrpoValueStatsJson(result.g_values);
  summaries["q_target"] = VrpoValueStatsJson(result.q_target_values);
  summaries["actor_advantage"] =
      VrpoValueStatsJson(result.actor_advantages);
  root["numeric_summaries"] = std::move(summaries);
  json::Object timing;
  timing["pack_h2d_s"] = result.pack_h2d_s;
  timing["q_forward_s"] = result.q_forward_s;
  timing["legal_gather_d2h_s"] = result.legal_gather_d2h_s;
  timing["scalar_reference_s"] = result.scalar_reference_s;
  timing["tensor_reference_s"] = result.tensor_reference_s;
  timing["total_s"] = result.total_s;
  root["timing"] = std::move(timing);
  json::Object memory;
  memory["capture_host_tensor_bytes"] = result.rows *
      static_cast<int64_t>(kVrpoDuneInformationStateSize +
                           dune_imperium::kVrpoCentralCriticTensorSize) *
      static_cast<int64_t>(sizeof(float));
  memory["max_chunk_input_bytes"] = result.max_chunk_rows *
      static_cast<int64_t>(dune_imperium::kVrpoCentralCriticTensorSize) *
      static_cast<int64_t>(sizeof(float));
  memory["max_chunk_dense_q_bytes"] = result.max_chunk_rows *
      static_cast<int64_t>(kVrpoDuneActionDim) * kVrpoNumSeats *
      static_cast<int64_t>(sizeof(float));
  memory["gpu_peak_allocated_increment_bytes"] =
      result.gpu_peak_allocated_increment_bytes;
  memory["gpu_peak_reserved_increment_bytes"] =
      result.gpu_peak_reserved_increment_bytes;
  memory["gpu_peak_increment_limit_bytes"] =
      absl::GetFlag(FLAGS_vrpo_q_gpu_peak_increment_limit_bytes);
  root["memory"] = std::move(memory);
  root["rollout_amp"] = absl::GetFlag(FLAGS_rollout_amp);
  root["train_amp"] = absl::GetFlag(FLAGS_train_amp);
  root["allow_tf32"] = absl::GetFlag(FLAGS_allow_tf32);
  json::Object precision_runtime;
  precision_runtime["tf32_cublas_before"] = tf32_cublas_before;
  precision_runtime["tf32_cublas_after"] = tf32_cublas_after;
  precision_runtime["tf32_cudnn_before"] = tf32_cudnn_before;
  precision_runtime["tf32_cudnn_after"] = tf32_cudnn_after;
  root["precision_runtime"] = std::move(precision_runtime);
  root["model_state_sha256_before"] = model_hash_before;
  root["model_state_sha256_after"] = model_hash_after;
  root["inference_state_sha256_before"] = inference_hash_before;
  root["inference_state_sha256_after"] = inference_hash_after;

  size_t model_size = 0;
  const std::string model_path = absl::GetFlag(FLAGS_model_checkpoint);
  json::Object source;
  source["model_path"] = std::filesystem::absolute(model_path).string();
  source["model_sha256"] = ComputeFileSHA256(model_path, &model_size);
  source["model_size"] = static_cast<int64_t>(model_size);
  source["manifest_path"] = std::filesystem::absolute(source_manifest).string();
  source["manifest_sha256"] = source_manifest_sha256;
  source["global_update"] = source_global_update;
  source["checkpoint_uuid"] = source_checkpoint_uuid;
  size_t binary_size = 0;
  std::error_code ec;
  const auto binary_path = std::filesystem::read_symlink("/proc/self/exe", ec);
  require(!ec && !binary_path.empty(), "could not resolve binary path");
  if (!ec && !binary_path.empty()) {
    source["binary_path"] = binary_path.string();
    source["binary_sha256"] =
        ComputeFileSHA256(binary_path.string(), &binary_size);
    source["binary_size"] = static_cast<int64_t>(binary_size);
  }
  root["source"] = std::move(source);
  json::Object source_code;
  source_code["root"] = source_provenance.root;
  source_code["combined_sha256"] = source_provenance.combined_sha256;
  json::Array source_files;
  for (const auto& file : source_provenance.files) {
    json::Object record;
    record["relative_path"] = file.relative_path;
    record["absolute_path"] = file.absolute_path;
    record["size"] = file.size;
    record["sha256"] = file.sha256;
    source_files.emplace_back(std::move(record));
  }
  source_code["files"] = std::move(source_files);
  root["source_code"] = std::move(source_code);
  json::Array validation_errors;
  for (const auto& item : errors) validation_errors.emplace_back(item);
  root["validation_errors"] = std::move(validation_errors);
  const bool valid = errors.empty();
  root["classification"] =
      valid ? "VALID_Q_REFERENCE_PREFLIGHT" : "INVALID";
  root["status"] = valid ? "VALID" : "INVALID";

  const std::filesystem::path output(output_path);
  const std::filesystem::path tmp = output_path + ".tmp";
  if (std::filesystem::exists(output) || std::filesystem::exists(tmp)) {
    SpielFatalError("VRPO Q preflight output already exists");
  }
  if (!output.parent_path().empty()) {
    std::filesystem::create_directories(output.parent_path());
  }
  {
    std::ofstream stream(tmp, std::ios::trunc);
    if (!stream) SpielFatalError("cannot create VRPO Q preflight temp artifact");
    stream << json::ToString(root, true) << "\n";
    stream.flush();
    if (!stream) SpielFatalError("VRPO Q preflight artifact write failed");
  }
  std::filesystem::rename(tmp, output);
  return valid;
}


#endif  // OPEN_SPIEL_BUILD_WITH_LIBTORCH

}  // namespace
}  // namespace open_spiel

int main(int argc, char** argv) {
  setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8", 1);
  std::ostringstream parity_command_line_stream;
  for (int i = 0; i < argc; ++i) {
    if (i) parity_command_line_stream << ' ';
    parity_command_line_stream << std::quoted(std::string(argv[i]));
  }
  const std::string parity_command_line = parity_command_line_stream.str();
  absl::ParseCommandLine(argc, argv);
  const auto vrpo_schedule_process_start =
      std::chrono::steady_clock::now();
  const open_spiel::VrpoScheduleRunDeadline vrpo_schedule_deadline =
      open_spiel::VrpoScheduleRunDeadline::Start(
          vrpo_schedule_process_start, std::chrono::seconds(1200));
  const open_spiel::VrpoPpoPilotDeadline vrpo_ppo_pilot_deadline =
      open_spiel::VrpoPpoPilotDeadline::Start(
          vrpo_schedule_process_start, std::chrono::seconds(1800));
  const open_spiel::VrpoPpoContinuationDeadline
      vrpo_ppo_continuation_deadline =
          open_spiel::VrpoPpoContinuationDeadline::Start(
              vrpo_schedule_process_start, std::chrono::seconds(1800));
  const bool numerical_parity =
      !absl::GetFlag(FLAGS_numerical_parity_output).empty();
  const bool vrpo_capture =
      !absl::GetFlag(FLAGS_vrpo_capture_output).empty();
  const bool vrpo_q_preflight =
      !absl::GetFlag(FLAGS_vrpo_q_preflight_output).empty();
  const bool vrpo_bootstrap =
      !absl::GetFlag(FLAGS_vrpo_bootstrap_root).empty();
  const bool vrpo_one_update =
      !absl::GetFlag(FLAGS_vrpo_one_update_input_archive).empty() ||
      !absl::GetFlag(FLAGS_vrpo_one_update_output_root).empty();
  const bool vrpo_schedule_screen =
      !absl::GetFlag(FLAGS_vrpo_schedule_screen_input_archive).empty() ||
      !absl::GetFlag(FLAGS_vrpo_schedule_screen_output_root).empty();
  const bool vrpo_ppo_pilot =
      !absl::GetFlag(FLAGS_vrpo_ppo_pilot_input_archive).empty() ||
      !absl::GetFlag(FLAGS_vrpo_ppo_pilot_output_root).empty();
  const bool vrpo_ppo_continuation =
      !absl::GetFlag(FLAGS_vrpo_ppo_continuation_output_root).empty();
  const bool vrpo_diagnostics = vrpo_capture || vrpo_q_preflight;
  if (static_cast<int>(numerical_parity) + static_cast<int>(vrpo_capture) +
          static_cast<int>(vrpo_q_preflight) +
          static_cast<int>(vrpo_bootstrap) + static_cast<int>(vrpo_one_update) +
          static_cast<int>(vrpo_schedule_screen) +
          static_cast<int>(vrpo_ppo_pilot) +
          static_cast<int>(vrpo_ppo_continuation) >
      1) {
    open_spiel::SpielFatalError(
        "Numerical parity, VRPO capture, VRPO Q preflight, VRPO bootstrap, VRPO one-update, VRPO schedule-screen, PPO-pilot, and PPO-continuation modes are mutually exclusive");
  }
  if (vrpo_one_update &&
      (absl::GetFlag(FLAGS_vrpo_one_update_input_archive).empty() ||
       absl::GetFlag(FLAGS_vrpo_one_update_output_root).empty())) {
    open_spiel::SpielFatalError(
        "VRPO one-update requires both input archive and fresh output root");
  }
  if (vrpo_schedule_screen &&
      (absl::GetFlag(FLAGS_vrpo_schedule_screen_input_archive).empty() ||
       absl::GetFlag(FLAGS_vrpo_schedule_screen_output_root).empty())) {
    open_spiel::SpielFatalError(
        "VRPO schedule screen requires both input archive and fresh output root");
  }
  if (vrpo_ppo_pilot &&
      (absl::GetFlag(FLAGS_vrpo_ppo_pilot_input_archive).empty() ||
       absl::GetFlag(FLAGS_vrpo_ppo_pilot_output_root).empty())) {
    open_spiel::SpielFatalError(
        "PPO pilot requires both input archive and fresh output root");
  }
  if (vrpo_ppo_continuation &&
      (absl::GetFlag(FLAGS_vrpo_ppo_continuation_registration_id).empty() ||
       absl::GetFlag(FLAGS_vrpo_ppo_continuation_evidence_root).empty())) {
    open_spiel::SpielFatalError(
        "PPO continuation requires registration, evidence, and fresh output roots");
  }
  if (vrpo_schedule_screen) {
    std::string resource_error;
    if (!open_spiel::vrpo_schedule_internal::CheckFreeSpaceBeforeStart(
            absl::GetFlag(FLAGS_vrpo_schedule_screen_output_root),
            &resource_error) ||
        !vrpo_schedule_deadline.Check("before source/model load",
                                      &resource_error)) {
      open_spiel::SpielFatalError(
          "VRPO schedule screen resource/deadline gate failed: " +
          resource_error);
    }
  }
  if (vrpo_ppo_pilot) {
    std::string resource_error;
    if (!open_spiel::vrpo_ppo_pilot_internal::CheckFreeSpaceBeforeStart(
            absl::GetFlag(FLAGS_vrpo_ppo_pilot_output_root),
            &resource_error) ||
        !vrpo_ppo_pilot_deadline.Check("before source/model load",
                                       &resource_error)) {
      open_spiel::SpielFatalError(
          "PPO pilot resource/deadline gate failed: " + resource_error);
    }
  }
  if (vrpo_ppo_continuation) {
    std::string resource_error;
    if (!open_spiel::vrpo_ppo_pilot_internal::CheckFreeSpaceBeforeStart(
            absl::GetFlag(FLAGS_vrpo_ppo_continuation_output_root),
            &resource_error) ||
        !vrpo_ppo_continuation_deadline.Check(
            "before source/model load", &resource_error)) {
      open_spiel::SpielFatalError(
          "PPO continuation resource/deadline gate failed: " +
          resource_error);
    }
  }
  if (numerical_parity) {
    auto parity_stop = [](const std::string& message) {
      open_spiel::SpielFatalError(
          "Raw-PPO numerical parity configuration rejected: " + message);
    };
    if (!absl::GetFlag(FLAGS_diagnostics_only)) {
      parity_stop("--diagnostics_only must be true");
    }
    if (absl::GetFlag(FLAGS_init_mode) != "diagnostic") {
      parity_stop("--init_mode must be diagnostic");
    }
    if (absl::GetFlag(FLAGS_artifact_manifest).empty()) {
      parity_stop("--artifact_manifest is required");
    }
    if (absl::GetFlag(FLAGS_numerical_parity_registration_id).empty()) {
      parity_stop("--numerical_parity_registration_id is required");
    }
    if (absl::GetFlag(FLAGS_numerical_parity_source_root).empty()) {
      parity_stop("--numerical_parity_source_root is required");
    }
    const std::string registered_source_sha =
        absl::GetFlag(FLAGS_numerical_parity_source_sha256);
    if (registered_source_sha.size() != 64 ||
        !std::all_of(registered_source_sha.begin(), registered_source_sha.end(),
                     [](char c) {
                       return (c >= '0' && c <= '9') ||
                              (c >= 'a' && c <= 'f');
                     })) {
      parity_stop(
          "--numerical_parity_source_sha256 must be an explicit lowercase SHA-256");
    }
    const int games = absl::GetFlag(FLAGS_rollout_games);
    if (games <= 0 || games > 64) {
      parity_stop("--rollout_games must be in [1,64] (hard ceiling 320000 transitions)");
    }
    if (absl::GetFlag(FLAGS_rollout_amp) ||
        absl::GetFlag(FLAGS_train_amp) ||
        !absl::GetFlag(FLAGS_allow_tf32)) {
      parity_stop(
          "v5-A requires rollout_amp=false, train_amp=false, and "
          "allow_tf32=true");
    }
    if (absl::GetFlag(FLAGS_ppo_minibatch_size) != 2048) {
      parity_stop(
          "v5-A requires --ppo_minibatch_size=2048");
    }
    if (absl::GetFlag(FLAGS_deterministic_rollout_eval) ||
        !absl::GetFlag(FLAGS_evaluator_device_synchronize)) {
      parity_stop(
          "v4 requires BatchedEvaluator with deterministic_rollout_eval=false "
          "and evaluator_device_synchronize=true");
    }
    if (absl::GetFlag(FLAGS_pipeline) ||
        absl::GetFlag(FLAGS_online_search_collection) ||
        absl::GetFlag(FLAGS_search_pi_mode) ||
        !absl::GetFlag(FLAGS_search_label_dir).empty() ||
        absl::GetFlag(FLAGS_train_value_only) ||
        absl::GetFlag(FLAGS_sample_counterfactual_states)) {
      parity_stop("pipeline, search, distillation, value-only, and counterfactual paths must all be disabled");
    }
    const double max_logprob_delta =
        absl::GetFlag(FLAGS_numerical_parity_max_abs_logprob_delta);
    const double max_full_kl =
        absl::GetFlag(FLAGS_numerical_parity_max_full_kl);
    const double max_raw_mass_residual =
        absl::GetFlag(FLAGS_numerical_parity_max_raw_mass_residual);
    const double min_ratio =
        absl::GetFlag(FLAGS_numerical_parity_min_ratio);
    const double max_ratio =
        absl::GetFlag(FLAGS_numerical_parity_max_ratio);
    if (absl::GetFlag(FLAGS_numerical_parity_min_rows) <= 0 ||
        absl::GetFlag(FLAGS_numerical_parity_min_nontrivial_rows) <= 0 ||
        !std::isfinite(max_logprob_delta) || max_logprob_delta <= 0.0 ||
        !std::isfinite(max_full_kl) || max_full_kl <= 0.0 ||
        !std::isfinite(max_raw_mass_residual) ||
        max_raw_mass_residual != 1e-5 ||
        !std::isfinite(min_ratio) || !(min_ratio > 0.0 && min_ratio <= 1.0) ||
        !std::isfinite(max_ratio) || max_ratio < 1.0) {
      parity_stop("all sample floors and numerical bounds must be stated explicitly and be valid");
    }
    if (absl::GetFlag(FLAGS_numerical_parity_min_rows) > games * 5000 ||
        absl::GetFlag(FLAGS_numerical_parity_min_nontrivial_rows) >
            games * 5000) {
      parity_stop("a sample floor exceeds the hard transition ceiling");
    }
    if (std::filesystem::exists(
            absl::GetFlag(FLAGS_numerical_parity_output)) ||
        std::filesystem::exists(
            absl::GetFlag(FLAGS_numerical_parity_output) + ".tmp")) {
      parity_stop("output or temp path already exists (fresh output required)");
    }
  }
  if (vrpo_diagnostics) {
    open_spiel::VrpoCaptureStartupConfig config;
    config.game = absl::GetFlag(FLAGS_game);
    config.registration_id =
        vrpo_q_preflight
            ? absl::GetFlag(FLAGS_vrpo_q_preflight_registration_id)
            : absl::GetFlag(FLAGS_vrpo_capture_registration_id);
    config.source_root = absl::GetFlag(FLAGS_vrpo_capture_source_root);
    config.source_sha256 =
        absl::GetFlag(FLAGS_vrpo_capture_source_sha256);
    config.diagnostics_only = absl::GetFlag(FLAGS_diagnostics_only);
    config.init_mode = absl::GetFlag(FLAGS_init_mode);
    config.rollout_games = absl::GetFlag(FLAGS_rollout_games);
    config.threads = absl::GetFlag(FLAGS_threads);
    config.rollout_amp = absl::GetFlag(FLAGS_rollout_amp);
    config.train_amp = absl::GetFlag(FLAGS_train_amp);
    config.allow_tf32 = absl::GetFlag(FLAGS_allow_tf32);
    config.pipeline = absl::GetFlag(FLAGS_pipeline);
    config.online_search_collection =
        absl::GetFlag(FLAGS_online_search_collection);
    config.search_pi_mode = absl::GetFlag(FLAGS_search_pi_mode);
    config.train_value_only = absl::GetFlag(FLAGS_train_value_only);
    config.sample_counterfactual_states =
        absl::GetFlag(FLAGS_sample_counterfactual_states);
    config.has_search_label_dir =
        !absl::GetFlag(FLAGS_search_label_dir).empty();
    config.shaped_reward_weight =
        absl::GetFlag(FLAGS_shaped_reward_weight);
    config.tleilaxu_breadcrumb_weight =
        absl::GetFlag(FLAGS_tleilaxu_breadcrumb_weight);
    config.tleilaxu_level7_breadcrumb_weight =
        absl::GetFlag(FLAGS_tleilaxu_level7_breadcrumb_weight);
    config.specimen_exchange_penalty =
        absl::GetFlag(FLAGS_specimen_exchange_penalty);
    config.reward_scale = absl::GetFlag(FLAGS_reward_scale);
    config.gamma = absl::GetFlag(FLAGS_gamma);
    config.lambda = absl::GetFlag(FLAGS_gae_lambda);
    config.probability_tolerance =
        absl::GetFlag(FLAGS_vrpo_probability_tolerance);
    std::string gate_error;
    const bool gate_valid = vrpo_q_preflight
        ? open_spiel::ValidateVrpoQPreflightStartupConfig(
              open_spiel::VrpoQPreflightStartupConfig{
                  config,
                  absl::GetFlag(FLAGS_vrpo_q_init_seed),
                  absl::GetFlag(FLAGS_vrpo_q_chunk_rows),
                  absl::GetFlag(FLAGS_vrpo_q_agreement_abs_tolerance),
                  absl::GetFlag(FLAGS_vrpo_q_agreement_rel_tolerance),
                  absl::GetFlag(
                      FLAGS_vrpo_q_gpu_peak_increment_limit_bytes)},
              &gate_error)
        : open_spiel::ValidateVrpoCaptureStartupConfig(config, &gate_error);
    if (!gate_valid) {
      open_spiel::SpielFatalError(
          "VRPO diagnostics configuration rejected: " + gate_error);
    }
    if (absl::GetFlag(FLAGS_artifact_manifest).empty()) {
      open_spiel::SpielFatalError(
          "VRPO capture requires --artifact_manifest");
    }
    if (absl::GetFlag(FLAGS_checkpoint_interval) != 0 ||
        absl::GetFlag(FLAGS_save_final_checkpoint)) {
      open_spiel::SpielFatalError(
          "VRPO capture requires checkpoint writes disabled");
    }
    const std::string output = vrpo_q_preflight
        ? absl::GetFlag(FLAGS_vrpo_q_preflight_output)
        : absl::GetFlag(FLAGS_vrpo_capture_output);
    if (std::filesystem::exists(output) ||
        std::filesystem::exists(output + ".tmp")) {
      open_spiel::SpielFatalError(
          "VRPO capture output or temp already exists");
    }
  }
  open_spiel::VrpoBootstrapStartupConfig vrpo_bootstrap_config;
  if (vrpo_bootstrap) {
    vrpo_bootstrap_config.game = absl::GetFlag(FLAGS_game);
    vrpo_bootstrap_config.root = absl::GetFlag(FLAGS_vrpo_bootstrap_root);
    vrpo_bootstrap_config.registration_id =
        absl::GetFlag(FLAGS_vrpo_bootstrap_registration_id);
    vrpo_bootstrap_config.source_root =
        absl::GetFlag(FLAGS_vrpo_bootstrap_source_root);
    vrpo_bootstrap_config.source_code_sha256 =
        absl::GetFlag(FLAGS_vrpo_bootstrap_source_sha256);
    vrpo_bootstrap_config.source_actor_model_sha256 =
        absl::GetFlag(FLAGS_vrpo_bootstrap_source_model_sha256);
    vrpo_bootstrap_config.source_actor_model_size =
        absl::GetFlag(FLAGS_vrpo_bootstrap_source_model_size);
    vrpo_bootstrap_config.source_manifest_sha256 =
        absl::GetFlag(FLAGS_vrpo_bootstrap_source_manifest_sha256);
    vrpo_bootstrap_config.source_manifest_size =
        absl::GetFlag(FLAGS_vrpo_bootstrap_source_manifest_size);
    vrpo_bootstrap_config.executed_binary_sha256 =
        absl::GetFlag(FLAGS_vrpo_bootstrap_binary_sha256);
    vrpo_bootstrap_config.executed_binary_size =
        absl::GetFlag(FLAGS_vrpo_bootstrap_binary_size);
    vrpo_bootstrap_config.experiment_uuid =
        absl::GetFlag(FLAGS_vrpo_bootstrap_experiment_uuid);
    vrpo_bootstrap_config.diagnostics_only =
        absl::GetFlag(FLAGS_diagnostics_only);
    vrpo_bootstrap_config.init_mode = absl::GetFlag(FLAGS_init_mode);
    vrpo_bootstrap_config.q_init_seed =
        absl::GetFlag(FLAGS_vrpo_bootstrap_q_seed);
    vrpo_bootstrap_config.base_seed =
        absl::GetFlag(FLAGS_vrpo_bootstrap_base_seed);
    vrpo_bootstrap_config.rollout_amp = absl::GetFlag(FLAGS_rollout_amp);
    vrpo_bootstrap_config.train_amp = absl::GetFlag(FLAGS_train_amp);
    vrpo_bootstrap_config.allow_tf32 = absl::GetFlag(FLAGS_allow_tf32);
    vrpo_bootstrap_config.pipeline = absl::GetFlag(FLAGS_pipeline);
    vrpo_bootstrap_config.online_search_collection =
        absl::GetFlag(FLAGS_online_search_collection);
    vrpo_bootstrap_config.search_pi_mode =
        absl::GetFlag(FLAGS_search_pi_mode);
    vrpo_bootstrap_config.train_value_only =
        absl::GetFlag(FLAGS_train_value_only);
    vrpo_bootstrap_config.sample_counterfactual_states =
        absl::GetFlag(FLAGS_sample_counterfactual_states);
    vrpo_bootstrap_config.has_search_label_dir =
        !absl::GetFlag(FLAGS_search_label_dir).empty();
    vrpo_bootstrap_config.checkpoint_writes_enabled =
        absl::GetFlag(FLAGS_checkpoint_interval) != 0 ||
        absl::GetFlag(FLAGS_save_final_checkpoint);
    vrpo_bootstrap_config.shaped_reward_weight =
        absl::GetFlag(FLAGS_shaped_reward_weight);
    vrpo_bootstrap_config.tleilaxu_breadcrumb_weight =
        absl::GetFlag(FLAGS_tleilaxu_breadcrumb_weight);
    vrpo_bootstrap_config.tleilaxu_level7_breadcrumb_weight =
        absl::GetFlag(FLAGS_tleilaxu_level7_breadcrumb_weight);
    vrpo_bootstrap_config.specimen_exchange_penalty =
        absl::GetFlag(FLAGS_specimen_exchange_penalty);
    vrpo_bootstrap_config.reward_scale = absl::GetFlag(FLAGS_reward_scale);
    vrpo_bootstrap_config.gamma = absl::GetFlag(FLAGS_gamma);
    vrpo_bootstrap_config.lambda = absl::GetFlag(FLAGS_gae_lambda);
    const auto canonical_arms = open_spiel::CanonicalVrpoPhase4Arms();
    vrpo_bootstrap_config.ranges = {{
        {canonical_arms[0].arm_id,
         absl::GetFlag(FLAGS_vrpo_bootstrap_ppo_cap10_start),
         absl::GetFlag(FLAGS_vrpo_bootstrap_ppo_cap10_end)},
        {canonical_arms[1].arm_id,
         absl::GetFlag(FLAGS_vrpo_bootstrap_ppo_uncapped_start),
         absl::GetFlag(FLAGS_vrpo_bootstrap_ppo_uncapped_end)},
        {canonical_arms[2].arm_id,
         absl::GetFlag(FLAGS_vrpo_bootstrap_vrpo_cap10_start),
         absl::GetFlag(FLAGS_vrpo_bootstrap_vrpo_cap10_end)},
        {canonical_arms[3].arm_id,
         absl::GetFlag(FLAGS_vrpo_bootstrap_vrpo_uncapped_start),
         absl::GetFlag(FLAGS_vrpo_bootstrap_vrpo_uncapped_end)}}};
    std::string bootstrap_error;
    if (!open_spiel::ValidateVrpoBootstrapStartupConfig(
            vrpo_bootstrap_config, &bootstrap_error)) {
      open_spiel::SpielFatalError(
          "VRPO bootstrap configuration rejected: " + bootstrap_error);
    }
    if (absl::GetFlag(FLAGS_artifact_manifest).empty() ||
        absl::GetFlag(FLAGS_hidden_dim) != 2048 ||
        absl::GetFlag(FLAGS_num_blocks) != 8 ||
        absl::GetFlag(FLAGS_nonlinear_value_head) ||
        !absl::GetFlag(FLAGS_aux_target_path).empty() ||
        std::filesystem::exists(absl::GetFlag(FLAGS_vrpo_bootstrap_root))) {
      open_spiel::SpielFatalError(
          "VRPO bootstrap requires source manifest, production actor layout, no auxiliary layout, and fresh absent root");
    }
  }
  // PWO-5 section 10.2: the reserved final-gate base-seed range, enforced at
  // the launcher so the exclusion is mechanical rather than a matter of
  // operator care. Checked before ANY other work.
  {
    const std::string stop = dune_seed::ReservedFinalGateSeedStop(
        "--seed", static_cast<long long>(absl::GetFlag(FLAGS_seed)));
    if (!stop.empty()) {
      std::cerr << stop << "\n";
      return 1;
    }
  }
  // PWO-5 gate 2 item (a): reject a NEGATIVE --specimen_exchange_penalty
  // fatally, after argument parsing and before anything else runs.
  //
  // The flag is a magnitude that is SUBTRACTED from the reward of a conversion
  // transition, so a negative value is a BONUS on exactly the breadcrumb
  // behaviour the term exists to suppress. This is not a hypothetical foot-gun:
  // the u175 lineage was trained with `--specimen_exchange_penalty=-0.02` and
  // the committed calibration_results_v2/pilot300_search_seed12/launch.sh still
  // carries it. Until now the ONLY validation of this flag was the
  // --allow_shaping gate below, which tests `!= 0.0` and accepts negatives.
  if (absl::GetFlag(FLAGS_specimen_exchange_penalty) < 0.0) {
    std::cerr << "Fatal: --specimen_exchange_penalty="
              << absl::GetFlag(FLAGS_specimen_exchange_penalty)
              << " is NEGATIVE. This flag is a MAGNITUDE THAT IS SUBTRACTED "
                 "from the reward of a ConvertSpecimenToTroop transition, so a "
                 "negative value is a BONUS on the breadcrumb behaviour the "
                 "term exists to penalize (this is the u175-lineage defect). "
                 "Pass a value >= 0; 0.0 disables the term."
              << std::endl;
    return -1;
  }
  if (absl::GetFlag(FLAGS_shaped_reward_weight) != 0.0 ||
      absl::GetFlag(FLAGS_tleilaxu_breadcrumb_weight) != 0.0 ||
      absl::GetFlag(FLAGS_tleilaxu_level7_breadcrumb_weight) != 0.0 ||
      absl::GetFlag(FLAGS_specimen_exchange_penalty) != 0.0) {
    if (!absl::GetFlag(FLAGS_allow_shaping)) {
      std::cerr << "Fatal: Reward shaping weights are non-zero but --allow_shaping is not set. "
                << "To run with reward shaping, pass --allow_shaping=true." << std::endl;
      return -1;
    } else {
      std::cout << "Warning: Running with non-zero experimental reward shaping weights!" << std::endl;
    }
  }
  if (absl::GetFlag(FLAGS_deterministic)) {
    at::globalContext().setDeterministicAlgorithms(true, /*silent=*/true);
  }
  using namespace open_spiel;

#ifndef OPEN_SPIEL_BUILD_WITH_LIBTORCH
  std::cerr << "dune_ppo_train requires OPEN_SPIEL_BUILD_WITH_LIBTORCH.\n";
  return 1;
#else
  at::set_num_threads(1);
  at::set_num_interop_threads(1);

  uint64_t master = static_cast<uint64_t>(absl::GetFlag(FLAGS_seed));
  torch::manual_seed(dune_seed::DeriveSeed(master, dune_seed::kStreamModelInit));

  auto game = open_spiel::LoadGame(absl::GetFlag(FLAGS_game));
  int64_t obs_size = game->GetType().provides_information_state_tensor
                         ? game->InformationStateTensorSize()
                         : game->ObservationTensorSize();
  int64_t action_size = game->NumDistinctActions();

  if (vrpo_bootstrap) {
    std::error_code binary_ec;
    const auto executed_binary =
        std::filesystem::read_symlink("/proc/self/exe", binary_ec);
    if (binary_ec || executed_binary.empty()) {
      open_spiel::SpielFatalError(
          "VRPO bootstrap cannot resolve executed binary");
    }
    size_t executed_binary_size = 0;
    const std::string executed_binary_sha256 =
        open_spiel::ComputeFileSHA256(
            executed_binary.string(), &executed_binary_size);
    if (executed_binary_sha256 !=
            vrpo_bootstrap_config.executed_binary_sha256 ||
        static_cast<int64_t>(executed_binary_size) !=
            vrpo_bootstrap_config.executed_binary_size) {
      open_spiel::SpielFatalError(
          "VRPO bootstrap executed binary identity mismatch");
    }
    const auto source_provenance =
        open_spiel::LoadVrpoCaptureSourceProvenance(
            absl::GetFlag(FLAGS_vrpo_bootstrap_source_root),
            absl::GetFlag(FLAGS_vrpo_bootstrap_source_sha256),
            open_spiel::VrpoBootstrapSourceRelativePaths());
    const std::string source_manifest_path =
        absl::GetFlag(FLAGS_artifact_manifest);
    size_t source_manifest_size = 0;
    const std::string source_manifest_sha256 =
        open_spiel::ComputeFileSHA256(source_manifest_path,
                                     &source_manifest_size);
    if (source_manifest_sha256 !=
            vrpo_bootstrap_config.source_manifest_sha256 ||
        static_cast<int64_t>(source_manifest_size) !=
            vrpo_bootstrap_config.source_manifest_size) {
      open_spiel::SpielFatalError(
          "VRPO bootstrap source manifest digest mismatch");
    }
    std::ifstream manifest_stream(source_manifest_path);
    const std::string manifest_text(
        (std::istreambuf_iterator<char>(manifest_stream)),
        std::istreambuf_iterator<char>());
    const auto parsed_manifest = open_spiel::json::FromString(manifest_text);
    if (!parsed_manifest.has_value() || !parsed_manifest->IsObject()) {
      open_spiel::SpielFatalError("VRPO bootstrap source manifest is malformed");
    }
    const auto& manifest = parsed_manifest->GetObject();
    auto manifest_string = [&](const char* key) -> std::string {
      const auto it = manifest.find(key);
      if (it == manifest.end() || !it->second.IsString()) {
        open_spiel::SpielFatalError(
            std::string("VRPO bootstrap source manifest missing ") + key);
      }
      return it->second.GetString();
    };
    auto manifest_int = [&](const char* key) -> int64_t {
      const auto it = manifest.find(key);
      if (it == manifest.end() || !it->second.IsInt()) {
        open_spiel::SpielFatalError(
            std::string("VRPO bootstrap source manifest missing ") + key);
      }
      return it->second.GetInt();
    };
    const std::string source_model_path =
        absl::GetFlag(FLAGS_model_checkpoint);
    size_t source_model_size = 0;
    const std::string source_model_sha256 =
        open_spiel::ComputeFileSHA256(source_model_path, &source_model_size);
    std::string observed_identity_error;
    if (!open_spiel::ValidateVrpoBootstrapObservedFileIdentities(
            vrpo_bootstrap_config, source_model_sha256,
            static_cast<int64_t>(source_model_size),
            source_manifest_sha256,
            static_cast<int64_t>(source_manifest_size),
            executed_binary_sha256,
            static_cast<int64_t>(executed_binary_size),
            &observed_identity_error)) {
      open_spiel::SpielFatalError(observed_identity_error);
    }
    if (source_model_sha256 !=
            vrpo_bootstrap_config.source_actor_model_sha256 ||
        static_cast<int64_t>(source_model_size) !=
            vrpo_bootstrap_config.source_actor_model_size ||
        source_model_sha256 != manifest_string("model_sha256") ||
        static_cast<int64_t>(source_model_size) !=
            manifest_int("model_file_size") ||
        manifest_int("hidden_dim") != 2048 ||
        manifest_int("num_blocks") != 8 || obs_size != 5580 ||
        action_size != 2391) {
      open_spiel::SpielFatalError(
          "VRPO bootstrap source actor bytes/architecture mismatch");
    }
    auto source_actor =
        std::make_shared<open_spiel::SharedDunePolicyValueNetImpl>(
            obs_size, 2048, action_size, 8, false);
    open_spiel::LoadModelCheckpoint(
        source_actor, source_model_path, torch::Device(torch::kCPU));
    std::set<std::string> actor_names;
    for (const auto& item : source_actor->named_parameters()) {
      actor_names.insert(item.key());
    }
    std::vector<open_spiel::VrpoNamedParameterIdentity> source_actor_identity;
    std::string bootstrap_error;
    if (!open_spiel::VrpoNamedParameterIdentities(
            *source_actor, &actor_names, &source_actor_identity,
            &bootstrap_error)) {
      open_spiel::SpielFatalError(bootstrap_error);
    }
    const std::string source_actor_subset_sha =
        open_spiel::VrpoNamedParameterIdentitySha256(
            source_actor_identity, true);
    const auto arms = open_spiel::CanonicalVrpoPhase4Arms();
    std::array<open_spiel::VrpoBootstrapArmInput, 4> inputs;
    std::array<open_spiel::VrpoFreshOptimizers, 4> optimizer_storage;
    const std::string experiment_uuid =
        vrpo_bootstrap_config.experiment_uuid;
    std::string common_q_hash;
    std::string common_layout_hash;
    std::string common_zero_hash;
    const uint64_t common_start_episode_id =
        vrpo_bootstrap_config.ranges[0].start_episode_id;
    const uint64_t common_end_episode_id_inclusive =
        vrpo_bootstrap_config.ranges[0].end_episode_id_inclusive;
    for (size_t arm = 0; arm < inputs.size(); ++arm) {
      auto actor =
          std::make_shared<open_spiel::SharedDunePolicyValueNetImpl>(
              obs_size, 2048, action_size, 8, false);
      std::vector<open_spiel::VrpoNamedParameterIdentity> copied;
      if (!open_spiel::CopyVrpoActorSubsetByName(
              *source_actor, *actor, actor_names, &copied,
              &bootstrap_error) ||
          open_spiel::VrpoNamedParameterIdentitySha256(copied, true) !=
              source_actor_subset_sha) {
        open_spiel::SpielFatalError(
            "VRPO bootstrap actor subset copy identity mismatch");
      }
      auto q = std::make_shared<open_spiel::DuneVrpoQNetImpl>(
          absl::GetFlag(FLAGS_vrpo_bootstrap_q_seed));
      if (!open_spiel::MakeVrpoFreshOptimizers(
              *actor, *q, &optimizer_storage[arm], &bootstrap_error)) {
        open_spiel::SpielFatalError(bootstrap_error);
      }
      open_spiel::VrpoPhase4ManifestBinding binding_base;
      binding_base.source_actor_model_sha256 = source_model_sha256;
      binding_base.source_actor_manifest_sha256 = source_manifest_sha256;
      binding_base.source_code_sha256 = source_provenance.combined_sha256;
      binding_base.q_init_seed =
          absl::GetFlag(FLAGS_vrpo_bootstrap_q_seed);
      binding_base.experiment_uuid = experiment_uuid;
      binding_base.base_seed =
          absl::GetFlag(FLAGS_vrpo_bootstrap_base_seed);
      binding_base.start_episode_id = common_start_episode_id;
      binding_base.end_episode_id_inclusive =
          common_end_episode_id_inclusive;
      std::vector<open_spiel::VrpoNamedParameterIdentity> actor_layout_records;
      std::vector<open_spiel::VrpoNamedParameterIdentity> q_layout_records;
      open_spiel::VrpoNamedParameterIdentities(
          *actor, nullptr, &actor_layout_records, &bootstrap_error);
      open_spiel::VrpoNamedParameterIdentities(
          *q, nullptr, &q_layout_records, &bootstrap_error);
      open_spiel::VrpoExpandedExpectedLayout layout;
      layout.label = "production_dune_vrpo_layout_v1";
      layout.test_fixture = false;
      layout.actor_observation_dim = 5580;
      layout.actor_hidden_dim = 2048;
      layout.actor_action_dim = 2391;
      layout.actor_residual_blocks = 8;
      layout.actor_names_shapes_sha256 =
          open_spiel::VrpoNamedParameterIdentitySha256(
              actor_layout_records, false);
      layout.q_names_shapes_sha256 =
          open_spiel::VrpoNamedParameterIdentitySha256(
              q_layout_records, false);
      open_spiel::VrpoPhase4ManifestBinding binding;
      if (!open_spiel::DeriveVrpoPhase4ManifestBinding(
              *actor, *q, optimizer_storage[arm], binding_base,
              layout, &binding, &bootstrap_error)) {
        open_spiel::SpielFatalError(bootstrap_error);
      }
      if (binding.actor_subset_sha256 != source_actor_subset_sha ||
          (arm > 0 &&
           (binding.q_init_sha256 != common_q_hash ||
            binding.module_layout_sha256 != common_layout_hash ||
            binding.optimizer_zero_state_sha256 != common_zero_hash))) {
        open_spiel::SpielFatalError(
            "VRPO bootstrap four-arm actor/Q/layout/zero identity mismatch");
      }
      if (arm == 0) {
        common_q_hash = binding.q_init_sha256;
        common_layout_hash = binding.module_layout_sha256;
        common_zero_hash = binding.optimizer_zero_state_sha256;
      }
      inputs[arm].arm = arms[arm];
      inputs[arm].binding = std::move(binding);
      inputs[arm].layout = std::move(layout);
      inputs[arm].checkpoint_uuid = open_spiel::GenerateUUID();
      inputs[arm].actor = actor;
      inputs[arm].q = q;
      inputs[arm].optimizers = &optimizer_storage[arm];
    }
    open_spiel::json::Object bootstrap_result;
    if (!open_spiel::WriteVrpoBootstrapRootAtomic(
            absl::GetFlag(FLAGS_vrpo_bootstrap_root),
            vrpo_bootstrap_config, inputs,
            open_spiel::VrpoBootstrapFailurePoint::kNone,
            &bootstrap_result, &bootstrap_error)) {
      open_spiel::SpielFatalError(bootstrap_error);
    }
    std::cout << "VRPO four-arm expanded bootstrap VALID: "
              << absl::GetFlag(FLAGS_vrpo_bootstrap_root) << "\n";
    return 0;
  }

  torch::Device device(torch::kCPU);
  if (torch::cuda::is_available()) {
    device = torch::Device(torch::kCUDA);
    at::globalContext().setAllowTF32CuBLAS(
        absl::GetFlag(FLAGS_allow_tf32));
    at::globalContext().setAllowTF32CuDNN(
        absl::GetFlag(FLAGS_allow_tf32));
    at::autocast::set_autocast_dtype(at::kCUDA, at::ScalarType::BFloat16);
    std::cout << "CUDA available. PPO training on GPU.\n";
  }
  if (numerical_parity && !device.is_cuda()) {
    SpielFatalError(
        "Raw-PPO numerical parity requires CUDA: without it the rollout side "
        "does not execute BF16 autocast and cannot test the registered gate.");
  }
  if ((vrpo_schedule_screen || vrpo_ppo_pilot || vrpo_ppo_continuation) &&
      !device.is_cuda()) {
    SpielFatalError(
        "VRPO schedule screen, PPO pilot, and PPO continuation require CUDA");
  }
  open_spiel::PpoNumericalParitySourceProvenance
      parity_source_provenance;
  open_spiel::PpoNumericalParitySourceProvenance
      vrpo_source_provenance;
  open_spiel::VrpoPhase4eSourceIdentity vrpo_one_update_source_identity;
  open_spiel::VrpoPhase4eSourceIdentity vrpo_schedule_source_identity;
  open_spiel::VrpoPhase4eSourceIdentity vrpo_ppo_pilot_source_identity;
  open_spiel::VrpoPpoPilotStartupConfig vrpo_ppo_pilot_preflight;
  open_spiel::VrpoExpandedArchiveIdentity
      vrpo_ppo_pilot_preloaded_input_identity;
  open_spiel::VrpoPhase4eSourceIdentity
      vrpo_ppo_continuation_source_identity;
  open_spiel::VrpoPpoContinuationStartupConfig
      vrpo_ppo_continuation_preflight;
  if (numerical_parity) {
    // Source identity is verified before model load and, critically, before a
    // rollout worker can be created. A mismatch cannot produce a partial
    // collection that looks like evidence from the registered implementation.
    parity_source_provenance =
        open_spiel::LoadPpoNumericalParitySourceProvenance(
            absl::GetFlag(FLAGS_numerical_parity_source_root),
            absl::GetFlag(FLAGS_numerical_parity_source_sha256));
  }
  if (vrpo_diagnostics) {
    vrpo_source_provenance =
        open_spiel::LoadVrpoCaptureSourceProvenance(
            absl::GetFlag(FLAGS_vrpo_capture_source_root),
            absl::GetFlag(FLAGS_vrpo_capture_source_sha256),
            vrpo_q_preflight
                ? open_spiel::VrpoQPreflightSourceRelativePaths()
                : open_spiel::VrpoCaptureSourceRelativePaths());
  }
  if (vrpo_one_update) {
    std::string source_error;
    if (!open_spiel::LoadVrpoPhase4eSourceIdentity(
            absl::GetFlag(FLAGS_vrpo_one_update_source_root),
            absl::GetFlag(FLAGS_vrpo_one_update_source_sha256),
            &vrpo_one_update_source_identity, &source_error)) {
      SpielFatalError("VRPO one-update source identity rejected: " +
                      source_error);
    }
  }
  if (vrpo_schedule_screen) {
    std::string source_error;
    if (!open_spiel::LoadVrpoScheduleSourceIdentity(
            absl::GetFlag(FLAGS_vrpo_schedule_screen_source_root),
            absl::GetFlag(FLAGS_vrpo_schedule_screen_source_sha256),
            &vrpo_schedule_source_identity, &source_error)) {
      SpielFatalError("VRPO schedule-screen source identity rejected: " +
                      source_error);
    }
    if (!vrpo_schedule_deadline.Check("after source identity",
                                      &source_error)) {
      SpielFatalError(source_error);
    }
  }
  if (vrpo_ppo_pilot) {
    std::string source_error;
    if (absl::GetFlag(FLAGS_init_mode) != "vrpo_ppo_pilot" ||
        absl::GetFlag(FLAGS_vrpo_ppo_pilot_profile) !=
            open_spiel::kVrpoPpoPilotProfile ||
        absl::GetFlag(FLAGS_vrpo_ppo_pilot_registration_id) !=
            open_spiel::kVrpoPpoPilotRegistrationId ||
        absl::GetFlag(FLAGS_seed) !=
            static_cast<int>(open_spiel::kVrpoPpoPilotBaseSeed) ||
        absl::GetFlag(FLAGS_start_episode_id) !=
            open_spiel::kVrpoPpoPilotStartEpisodeId ||
        absl::GetFlag(FLAGS_total_updates) !=
            open_spiel::kVrpoPpoPilotUpdates) {
      SpielFatalError(
          "PPO pilot compiled profile/registration/seed/range contract rejected before model construction");
    }
    if (!open_spiel::LoadVrpoPpoPilotSourceIdentity(
            absl::GetFlag(FLAGS_vrpo_ppo_pilot_source_root),
            absl::GetFlag(FLAGS_vrpo_ppo_pilot_source_sha256),
            &vrpo_ppo_pilot_source_identity, &source_error) ||
        !vrpo_ppo_pilot_deadline.Check("after source identity",
                                       &source_error)) {
      SpielFatalError("PPO-pilot source identity rejected: " + source_error);
    }
    vrpo_ppo_pilot_preflight.input_archive =
        absl::GetFlag(FLAGS_vrpo_ppo_pilot_input_archive);
    vrpo_ppo_pilot_preflight.output_root =
        absl::GetFlag(FLAGS_vrpo_ppo_pilot_output_root);
    vrpo_ppo_pilot_preflight.origin_archive_sha256 =
        absl::GetFlag(FLAGS_vrpo_ppo_pilot_origin_archive_sha256);
    vrpo_ppo_pilot_preflight.evidence_root =
        absl::GetFlag(FLAGS_vrpo_ppo_pilot_evidence_root);
    vrpo_ppo_pilot_preflight.evidence_cli_sha256 = {
        absl::GetFlag(FLAGS_vrpo_ppo_pilot_v3_schedule_registration_sha256),
        absl::GetFlag(FLAGS_vrpo_ppo_pilot_v3_schedule_result_sha256),
        absl::GetFlag(FLAGS_vrpo_ppo_pilot_v3_schedule_validation_sha256),
        absl::GetFlag(FLAGS_vrpo_ppo_pilot_v3_schedule_corpus_manifest_sha256),
        absl::GetFlag(FLAGS_vrpo_ppo_pilot_v3_schedule_corpus_sha256),
        absl::GetFlag(FLAGS_vrpo_ppo_pilot_selected_screen_manifest_sha256),
        absl::GetFlag(FLAGS_vrpo_ppo_pilot_selected_screen_result_sha256),
        absl::GetFlag(FLAGS_vrpo_ppo_pilot_selected_screen_validation_sha256),
        absl::GetFlag(FLAGS_vrpo_ppo_pilot_confirm_manifest_sha256),
        absl::GetFlag(FLAGS_vrpo_ppo_pilot_confirm_result_sha256),
        absl::GetFlag(FLAGS_vrpo_ppo_pilot_confirm_validation_sha256)};
    std::error_code binary_ec;
    const auto binary_path =
        std::filesystem::read_symlink("/proc/self/exe", binary_ec);
    size_t binary_size = 0;
    const std::string binary_sha256 = binary_ec || binary_path.empty()
        ? std::string()
        : open_spiel::ComputeFileSHA256(binary_path.string(), &binary_size);
    if (binary_sha256 !=
            absl::GetFlag(FLAGS_vrpo_ppo_pilot_binary_sha256) ||
        static_cast<int64_t>(binary_size) !=
            absl::GetFlag(FLAGS_vrpo_ppo_pilot_binary_size) ||
        !open_spiel::ComputeVrpoExpandedArchiveIdentity(
            vrpo_ppo_pilot_preflight.input_archive,
            &vrpo_ppo_pilot_preloaded_input_identity, &source_error) ||
        vrpo_ppo_pilot_preloaded_input_identity.combined_sha256 !=
            open_spiel::kVrpoPpoPilotOriginArchiveSha256 ||
        vrpo_ppo_pilot_preflight.origin_archive_sha256 !=
            open_spiel::kVrpoPpoPilotOriginArchiveSha256 ||
        !open_spiel::ValidateVrpoPpoPilotEvidenceFiles(
            vrpo_ppo_pilot_preflight.evidence_root,
            open_spiel::VrpoPpoPilotCanonicalEvidenceFiles(),
            vrpo_ppo_pilot_preflight.evidence_cli_sha256,
            &vrpo_ppo_pilot_preflight.evidence_observations,
            &source_error)) {
      SpielFatalError(
          "PPO pilot immutable binary/origin/evidence chain rejected before model construction: " +
          source_error);
    }
    vrpo_ppo_pilot_preflight.executed_binary_sha256 = binary_sha256;
    vrpo_ppo_pilot_preflight.executed_binary_size = binary_size;
  }
  if (vrpo_ppo_continuation) {
    std::string source_error;
    if (absl::GetFlag(FLAGS_init_mode) != "vrpo_ppo_continuation" ||
        absl::GetFlag(FLAGS_vrpo_ppo_continuation_profile) !=
            open_spiel::kVrpoPpoContinuationProfile ||
        absl::GetFlag(FLAGS_vrpo_ppo_continuation_registration_id) !=
            open_spiel::kVrpoPpoContinuationRegistrationId ||
        absl::GetFlag(FLAGS_seed) != static_cast<int>(
            open_spiel::kVrpoPpoContinuationBaseSeed) ||
        absl::GetFlag(FLAGS_start_episode_id) !=
            open_spiel::kVrpoPpoContinuationStartEpisodeId ||
        absl::GetFlag(FLAGS_total_updates) !=
            open_spiel::kVrpoPpoContinuationNewUpdates) {
      SpielFatalError(
          "PPO continuation compiled profile/registration/seed/range rejected before model construction");
    }
    if (!open_spiel::LoadVrpoPpoContinuationSourceIdentity(
            absl::GetFlag(FLAGS_vrpo_ppo_continuation_source_root),
            absl::GetFlag(FLAGS_vrpo_ppo_continuation_source_sha256),
            &vrpo_ppo_continuation_source_identity, &source_error) ||
        !vrpo_ppo_continuation_deadline.Check(
            "after continuation source identity", &source_error)) {
      SpielFatalError("PPO continuation source identity rejected: " +
                      source_error);
    }
    vrpo_ppo_continuation_preflight.output_root =
        absl::GetFlag(FLAGS_vrpo_ppo_continuation_output_root);
    vrpo_ppo_continuation_preflight.evidence_root =
        absl::GetFlag(FLAGS_vrpo_ppo_continuation_evidence_root);
    if (!open_spiel::vrpo_ppo_continuation_internal::ValidatePriorSemantics(
            vrpo_ppo_continuation_preflight.evidence_root,
            &vrpo_ppo_continuation_preflight.prior, &source_error)) {
      SpielFatalError(
          "PPO continuation immutable pilot/confirmation chain rejected before model construction: " +
          source_error);
    }
    std::error_code binary_ec;
    const auto binary_path =
        std::filesystem::read_symlink("/proc/self/exe", binary_ec);
    size_t binary_size = 0;
    const std::string binary_sha256 = binary_ec || binary_path.empty()
        ? std::string()
        : open_spiel::ComputeFileSHA256(binary_path.string(), &binary_size);
    if (binary_sha256 !=
            absl::GetFlag(FLAGS_vrpo_ppo_continuation_binary_sha256) ||
        static_cast<int64_t>(binary_size) !=
            absl::GetFlag(FLAGS_vrpo_ppo_continuation_binary_size)) {
      SpielFatalError(
          "PPO continuation executed binary identity mismatch before model construction");
    }
    vrpo_ppo_continuation_preflight.executed_binary_sha256 = binary_sha256;
    vrpo_ppo_continuation_preflight.executed_binary_size = binary_size;
  }

  // PWO-5 section 7.2 / Appendix A.1.
  //
  // The migrated layout is used by ALL SIX ARMS, P arms included -- "the arms
  // differ only in coefficients, never in architecture". A P arm running the
  // pre-migration architecture would differ from T in TWO ways at once and
  // would not be a control for it. So the heads are constructed whenever the
  // auxiliary configuration is present, NOT when a coefficient is nonzero.
  //
  // The trigger is --aux_target_path, which Appendix A.1 files under MATCHED:
  // all six arms are launched with the same target artifact and all six verify
  // its digest, while whether the targets are ever READ is decided by the
  // coefficients alone. That keeps H vs T at exactly three differences.
  //
  // The init seed is the fixed constant, never the run seed, so all six arms'
  // initial head parameters are byte-identical (section 9's matching table).
  const bool pwo5_aux_layout = !absl::GetFlag(FLAGS_aux_target_path).empty();
  const uint64_t pwo5_head_init_seed =
      dune_seed::DeriveSeed(absl::GetFlag(FLAGS_head_init_constant),
                            dune_seed::kDomainTrain, 0,
                            dune_seed::kStreamModelInit);
  if (pwo5_aux_layout) {
    std::cout << "[PWO-5] auxiliary head layout ENABLED (--aux_target_path set). "
              << "head_init_constant=" << absl::GetFlag(FLAGS_head_init_constant)
              << " -> derived init seed " << pwo5_head_init_seed << std::endl;
  }

  std::shared_ptr<open_spiel::SharedDunePolicyValueNetImpl> training_model =
      std::make_shared<open_spiel::SharedDunePolicyValueNetImpl>(
          obs_size, absl::GetFlag(FLAGS_hidden_dim), action_size,
          absl::GetFlag(FLAGS_num_blocks), absl::GetFlag(FLAGS_nonlinear_value_head),
          pwo5_aux_layout, pwo5_head_init_seed);
  // The INFERENCE model deliberately does NOT get the heads. Section 8.6
  // requires the three heads be computed "never in the rollout/inference
  // forward"; constructing them here would make that a matter of care instead
  // of a structural impossibility.
  std::shared_ptr<open_spiel::SharedDunePolicyValueNetImpl> inference_model =
      std::make_shared<open_spiel::SharedDunePolicyValueNetImpl>(
          obs_size, absl::GetFlag(FLAGS_hidden_dim), action_size,
          absl::GetFlag(FLAGS_num_blocks), absl::GetFlag(FLAGS_nonlinear_value_head));
  training_model->to(device);
  inference_model->to(device);

  if (absl::GetFlag(FLAGS_sample_counterfactual_states) && !absl::GetFlag(FLAGS_train_value_only)) {
    SpielFatalError("Counterfactual states sampling can only be enabled during value-only training (train_value_only=true).");
  }
  if (absl::GetFlag(FLAGS_sample_counterfactual_states) &&
      (absl::GetFlag(FLAGS_counterfactual_samples_per_game) <= 0 ||
       absl::GetFlag(FLAGS_counterfactual_replay_weight) <= 0)) {
    SpielFatalError("Counterfactual sample and replay counts must both be positive.");
  }

  if (absl::GetFlag(FLAGS_train_value_only)) {
    if (absl::GetFlag(FLAGS_reward_scale) != 4.0) {
      SpielFatalError("reward_scale must be exactly 4.0 when train_value_only is active.");
    }
    {
      std::string corpus_path = "data/dune_diagnostic_corpus.json";
      std::ifstream f(corpus_path);
      if (!f.good()) {
        SpielFatalError(absl::StrCat("CRITICAL EXCEPTION: Prohibited histories corpus file ", corpus_path, " is missing! Fail-closed boundary triggered."));
      }
      std::string content((std::istreambuf_iterator<char>(f)),
                           std::istreambuf_iterator<char>());
      auto json_parsed = open_spiel::json::FromString(content);
      if (!json_parsed) {
        SpielFatalError(absl::StrCat("CRITICAL EXCEPTION: Failed to parse prohibited histories corpus from ", corpus_path, "!"));
      }
      auto json_arr = json_parsed.value().GetArray();
      for (const auto& item_val : json_arr) {
        auto item = item_val.GetObject();
        auto history_arr = item.at("history").GetArray();
        std::vector<open_spiel::Action> hist;
        for (const auto& act_val : history_arr) {
          hist.push_back(static_cast<open_spiel::Action>(act_val.GetInt()));
        }
        prohibited_histories.insert(hist);
      }
      std::cout << "[INFO] Loaded " << prohibited_histories.size()
                << " prohibited histories from official corpus to prevent training leakage.\n";
    }
    std::cout << "[INFO] train_value_only is true: freezing trunk & policy head parameters.\n";
    if (absl::GetFlag(FLAGS_unfreeze_trunk)) {
      if (absl::GetFlag(FLAGS_policy_kl_anchor_coeff) <= 0.0) {
        SpielFatalError("CRITICAL SAFETY ERROR: policy_kl_anchor_coeff must be positive when unfreezing the trunk layer in value-only training mode.");
      }
    } else {
      for (auto& p : training_model->input_layer->parameters()) {
        p.set_requires_grad(false);
      }
      for (auto& block : training_model->res_blocks) {
        for (auto& p : block->parameters()) {
          p.set_requires_grad(false);
        }
      }
    }
    for (auto& p : training_model->policy_head->parameters()) {
      p.set_requires_grad(false);
    }
  }

  training_model->train();
  inference_model->eval();

  std::unique_ptr<torch::optim::AdamW> optimizer;
  bool optimizer_constructed = false;
  if (!vrpo_one_update && !vrpo_schedule_screen && !vrpo_ppo_pilot &&
      !vrpo_ppo_continuation &&
      open_spiel::VrpoCaptureShouldConstructOptimizer(vrpo_diagnostics)) {
    optimizer = open_spiel::MakeOptimizer(training_model);
    optimizer_constructed = true;
    // Section 7.4: materialize BEFORE any load, so the entries exist in all six
    // arms; a subsequent load simply overwrites the ones the archive carries.
    open_spiel::MaterializeAuxOptimizerState(training_model, *optimizer);
  }

  // --- PWO-5 sections 5.2, 8.5, 8.6: the auxiliary data path. -------------
  //
  // The target artifact and the held-out membership list are MATCHED fields
  // (Appendix A.1): all six arms are launched with the same paths and the same
  // digests, and ALL SIX VERIFY THEM. Verification is UNCONDITIONAL; USE is not
  // -- whether the targets are ever read is decided by the coefficients alone.
  // That keeps H vs T at exactly three differences while still catching a
  // corrupt artifact on every arm at launch rather than only on the two that
  // would have read it.
  open_spiel::pwo5::AuxTargetStore pwo5_store;
  open_spiel::Pwo5AuxConfig pwo5_cfg;
  pwo5_cfg.final_vp_coef = absl::GetFlag(FLAGS_final_vp_head_coef);
  pwo5_cfg.terminal_round_coef = absl::GetFlag(FLAGS_terminal_round_head_coef);
  pwo5_cfg.next_own_action_coef = absl::GetFlag(FLAGS_next_own_action_head_coef);
  pwo5_cfg.huber_delta = absl::GetFlag(FLAGS_huber_delta_final_vp);
  std::vector<int32_t> pwo5_heldout_games;
  // Carried out of the block below so the Appendix A.1 manifest fields can be
  // assembled once `search_label_fingerprint` also exists.
  std::string pwo5_target_digest, pwo5_heldout_digest,
      pwo5_update1_sampler_digest, pwo5_head_identity;
  if (pwo5_aux_layout) {
    // (a) the held-out membership list, and its registered digest.
    const std::string heldout_path = absl::GetFlag(FLAGS_aux_heldout_games_path);
    if (heldout_path.empty()) {
      SpielFatalError(
          "PWO-5: --aux_target_path is set but --aux_heldout_games_path is "
          "empty. Section 8.5 requires the whole-trajectory split membership "
          "list on every arm; without it the loader's per-row % 11 rule would "
          "silently determine the heads' held-out set, which plan section 9.6 "
          "forbids.");
    }
    std::ifstream hin(heldout_path);
    if (!hin) SpielFatalError("PWO-5: cannot open " + heldout_path);
    std::string heldout_text((std::istreambuf_iterator<char>(hin)),
                             std::istreambuf_iterator<char>());
    // Parse the ascending integer list out of "heldout_game_indices": [...].
    {
      const std::size_t key = heldout_text.find("heldout_game_indices");
      if (key == std::string::npos) {
        SpielFatalError("PWO-5: " + heldout_path +
                        " has no heldout_game_indices");
      }
      const std::size_t lb = heldout_text.find('[', key);
      const std::size_t rb = heldout_text.find(']', lb);
      std::string body = heldout_text.substr(lb + 1, rb - lb - 1);
      for (char& c : body) if (c == ',') c = ' ';
      std::istringstream bs(body);
      int32_t v;
      while (bs >> v) pwo5_heldout_games.push_back(v);
    }
    // The canonical serialization is the ASCENDING indices joined by "," with
    // no spaces -- computed over the LIST, not the file, so a reformatted JSON
    // still verifies.
    std::sort(pwo5_heldout_games.begin(), pwo5_heldout_games.end());
    std::string canonical;
    for (std::size_t i = 0; i < pwo5_heldout_games.size(); ++i) {
      if (i) canonical += ",";
      canonical += std::to_string(pwo5_heldout_games[i]);
    }
    const std::string heldout_digest =
        open_spiel::ComputeStringSHA256(canonical);
    const std::string expect_heldout = absl::GetFlag(FLAGS_aux_heldout_sha256);
    if (!expect_heldout.empty() && expect_heldout != heldout_digest) {
      SpielFatalError(
          "PWO-5 section 8.5: held-out split digest mismatch. expected " +
          expect_heldout + " computed " + heldout_digest +
          ". The split cannot silently differ across arms, replicates or the "
          "u600 extension.");
    }

    // (b) the target artifact, and its registered digest. Section 5.2: a
    // materialized artifact that is not hashed is an unpinned replay input,
    // which section 7's exact-resume gate would not catch.
    const std::string target_path = absl::GetFlag(FLAGS_aux_target_path);
    size_t target_size = 0;
    const std::string target_digest =
        open_spiel::ComputeFileSHA256(target_path, &target_size);
    const std::string expect_target = absl::GetFlag(FLAGS_aux_target_sha256);
    if (!expect_target.empty() && expect_target != target_digest) {
      SpielFatalError("PWO-5 section 5.2: --aux_target_sha256 mismatch for " +
                      target_path + ". expected " + expect_target +
                      " computed " + target_digest);
    }
    std::string load_error;
    if (!pwo5_store.Load(target_path, pwo5_heldout_games, &load_error)) {
      SpielFatalError("PWO-5: auxiliary target artifact rejected: " +
                      load_error);
    }
    if (pwo5_store.obs_size() != obs_size) {
      SpielFatalError(absl::StrFormat(
          "PWO-5: auxiliary target obs_size %lld != the game's %lld. The "
          "targets were prepared against a different observation encoder.",
          static_cast<long long>(pwo5_store.obs_size()),
          static_cast<long long>(obs_size)));
    }
    std::cout << "[PWO-5] auxiliary targets: " << target_path << " sha256="
              << target_digest << "\n"
              << "[PWO-5]   training games " << pwo5_store.training_games().size()
              << ", training rows " << pwo5_store.training_row_count()
              << ", min rows/game " << pwo5_store.MinRowsPerTrainingGame()
              << ", held out " << pwo5_heldout_games.size()
              << " (digest " << heldout_digest << ")\n"
              << "[PWO-5]   head coefficients: final_vp="
              << pwo5_cfg.final_vp_coef << " terminal_round="
              << pwo5_cfg.terminal_round_coef << " next_own_action="
              << pwo5_cfg.next_own_action_coef
              << (pwo5_cfg.AnyActive() ? "  (HEAD-ON)" : "  (HEAD-OFF: no "
                  "auxiliary forward, no head output computed)")
              << std::endl;
    // Section 17.4 floors (a2)/(a3), checked here rather than trusted.
    if (pwo5_cfg.AnyActive()) {
      const int min_rows = pwo5_store.MinRowsPerTrainingGame();
      if (min_rows < absl::GetFlag(FLAGS_aux_rows_per_game)) {
        SpielFatalError(absl::StrFormat(
            "PWO-5 section 8.6: a training game has only %d rows but "
            "--aux_rows_per_game=%d. The without-replacement draw would be "
            "infeasible.",
            min_rows, absl::GetFlag(FLAGS_aux_rows_per_game)));
      }
      if (static_cast<int>(pwo5_store.training_games().size()) <
          absl::GetFlag(FLAGS_aux_games_per_update)) {
        SpielFatalError(
            "PWO-5 section 8.6: fewer training games than "
            "--aux_games_per_update.");
      }
    }

    // Section 8.6's update-1 sampler digest, computed EAGERLY on ALL SIX arms
    // rather than captured when update 1 happens to run. Two reasons, both
    // registered: T and P arms have all-zero coefficients so they never draw,
    // yet Appendix A.1 note 3 requires the digest in EVERY arm's manifest so a
    // triplet can be checked to have drawn identically; and the bootstrap
    // manifest is written before any update executes, so a lazily captured
    // digest could not appear in it at all.
    const uint64_t update1_aux_seed = dune_seed::DeriveSeed(
        master, dune_seed::kDomainTrain, /*update=*/1, /*aux=*/0,
        dune_seed::kStreamAuxSampling);
    const auto update1_draw = pwo5_store.Draw(
        update1_aux_seed, absl::GetFlag(FLAGS_aux_games_per_update),
        absl::GetFlag(FLAGS_aux_rows_per_game),
        absl::GetFlag(FLAGS_aux_batches_per_update));
    pwo5_update1_sampler_digest =
        open_spiel::ComputeStringSHA256(update1_draw.digest);
    pwo5_target_digest = target_digest;
    pwo5_heldout_digest = heldout_digest;
    pwo5_head_identity = ComputeAuxHeadIdentity(training_model);
    std::cout << "[PWO-5] update-1 sampler digest (precomputed) "
              << pwo5_update1_sampler_digest << "\n"
              << "[PWO-5] head-init identity " << pwo5_head_identity
              << std::endl;
  }

  std::shared_ptr<open_spiel::SharedDunePolicyValueNetImpl> anchor_model = nullptr;
  if (absl::GetFlag(FLAGS_train_value_only) && absl::GetFlag(FLAGS_unfreeze_trunk)) {
    anchor_model = std::make_shared<open_spiel::SharedDunePolicyValueNetImpl>(
        obs_size, absl::GetFlag(FLAGS_hidden_dim), action_size,
        absl::GetFlag(FLAGS_num_blocks), absl::GetFlag(FLAGS_nonlinear_value_head));
    anchor_model->to(device);
  }

  std::string init_mode = absl::GetFlag(FLAGS_init_mode);
  if (init_mode.empty()) {
    SpielFatalError("Required flag --init_mode is missing. Must be 'random', 'checkpoint', 'bootstrap', or 'validate_legacy'.");
  }

  // Phase 4e owns an intentionally separate optimizer/model path. It loads
  // only the strict expanded arm archive; neither legacy PPO checkpoint nor
  // optimizer state is read or writable on this path.
  std::shared_ptr<open_spiel::DuneVrpoQNetImpl> vrpo_one_update_q;
  open_spiel::VrpoFreshOptimizers vrpo_one_update_optimizers;
  open_spiel::VrpoPhase4ArmConfig vrpo_one_update_arm;
  open_spiel::VrpoPhase4ManifestBinding vrpo_one_update_binding;
  open_spiel::VrpoExpandedExpectedLayout vrpo_one_update_layout;
  open_spiel::VrpoExpandedArchiveIdentity vrpo_one_update_input_identity;
  open_spiel::VrpoPhase4eStartupConfig vrpo_one_update_startup;
  std::shared_ptr<open_spiel::DuneVrpoQNetImpl> vrpo_schedule_q;
  open_spiel::VrpoPhase4ManifestBinding vrpo_schedule_binding;
  open_spiel::VrpoExpandedExpectedLayout vrpo_schedule_layout;
  open_spiel::VrpoExpandedArchiveIdentity vrpo_schedule_input_identity;
  open_spiel::VrpoScheduleStartupConfig vrpo_schedule_startup;
  std::shared_ptr<open_spiel::DuneVrpoQNetImpl> vrpo_ppo_pilot_q;
  std::unique_ptr<torch::optim::AdamW> vrpo_ppo_pilot_actor_optimizer;
  open_spiel::VrpoPhase4ManifestBinding vrpo_ppo_pilot_binding;
  open_spiel::VrpoExpandedExpectedLayout vrpo_ppo_pilot_layout;
  open_spiel::VrpoExpandedArchiveIdentity vrpo_ppo_pilot_input_identity;
  open_spiel::VrpoPpoPilotStartupConfig vrpo_ppo_pilot_startup;
  open_spiel::VrpoPpoPilotState vrpo_ppo_pilot_state;
  std::shared_ptr<open_spiel::DuneVrpoQNetImpl>
      vrpo_ppo_continuation_q;
  std::unique_ptr<torch::optim::AdamW>
      vrpo_ppo_continuation_actor_optimizer;
  open_spiel::VrpoPhase4ManifestBinding vrpo_ppo_continuation_binding;
  open_spiel::VrpoExpandedExpectedLayout vrpo_ppo_continuation_layout;
  open_spiel::VrpoPpoContinuationStartupConfig
      vrpo_ppo_continuation_startup;
  open_spiel::VrpoPpoContinuationState vrpo_ppo_continuation_state;

  // WO-PERF-TIMING. Enforced HERE, at startup, because FLAGS_pipeline is
  // defined in this TU and validating at the first update would waste a whole
  // rollout before failing.
  //
  // --pipeline=true runs CollectRollout's GPU inference on a background thread
  // concurrently with TrainPpoUpdate, and nothing under examples/ isolates CUDA
  // streams -- there is no CUDAStreamGuard, setCurrentCUDAStream or
  // getStreamFromPool anywhere in the tree. Every phase's event pair would then
  // bracket collector kernels as well as PPO kernels, producing an entirely
  // plausible table that is wrong, with no post-hoc way to detect it.
  if (absl::GetFlag(FLAGS_phase_timing_mode) == "phases" &&
      absl::GetFlag(FLAGS_pipeline)) {
    SpielFatalError(
        "--phase_timing_mode=phases is incompatible with --pipeline=true: the "
        "background collection thread shares the compute stream, so every "
        "per-phase device time would silently include collector kernels. Run "
        "timed arms with --pipeline=false.");
  }

  std::string config_fingerprint = open_spiel::ComputeConfigFingerprint();
  const std::string pre_precision_config_fingerprint =
      open_spiel::ComputeCurrentPrePrecisionConfigFingerprint();
  std::string search_label_fingerprint = open_spiel::GetSearchLabelFingerprint(absl::GetFlag(FLAGS_search_label_dir));
  std::string run_uuid = open_spiel::GenerateUUID();

  // -------------------------------------------------------------------------
  // PWO-5 Appendix A.1 note 3: assemble the manifest block, ONCE, before any
  // manifest is written or read. Both init_mode branches below consume it.
  // -------------------------------------------------------------------------
  if (pwo5_aux_layout) {
    open_spiel::json::Object& f = open_spiel::g_pwo5_manifest_fields;
    // --- the five TREATMENT fields (section 9.1). Differing ACROSS arms is
    // the point; differing from the arm's OWN registered value across a resume
    // is the STOP, which is what the within-arm check below enforces.
    f["final_vp_head_coef"] =
        open_spiel::Pwo5Double(absl::GetFlag(FLAGS_final_vp_head_coef));
    f["terminal_round_head_coef"] =
        open_spiel::Pwo5Double(absl::GetFlag(FLAGS_terminal_round_head_coef));
    f["next_own_action_head_coef"] =
        open_spiel::Pwo5Double(absl::GetFlag(FLAGS_next_own_action_head_coef));
    f["search_lambda"] =
        open_spiel::Pwo5Double(absl::GetFlag(FLAGS_search_lambda));
    f["search_label_dir"] = absl::GetFlag(FLAGS_search_label_dir);
    // --- the MATCHED fields: byte-identical on all six arms.
    f["aux_target_path"] = absl::GetFlag(FLAGS_aux_target_path);
    f["aux_target_sha256"] = pwo5_target_digest;
    f["aux_heldout_games_path"] = absl::GetFlag(FLAGS_aux_heldout_games_path);
    f["aux_heldout_sha256"] = pwo5_heldout_digest;
    f["aux_games_per_update"] =
        static_cast<int64_t>(absl::GetFlag(FLAGS_aux_games_per_update));
    f["aux_rows_per_game"] =
        static_cast<int64_t>(absl::GetFlag(FLAGS_aux_rows_per_game));
    f["aux_batches_per_update"] =
        static_cast<int64_t>(absl::GetFlag(FLAGS_aux_batches_per_update));
    f["huber_delta_final_vp"] =
        open_spiel::Pwo5Double(absl::GetFlag(FLAGS_huber_delta_final_vp));
    f["head_init_constant"] =
        static_cast<int64_t>(absl::GetFlag(FLAGS_head_init_constant));
    f["emit_canary_columns"] = absl::GetFlag(FLAGS_emit_canary_columns);
    // --- identities the flags alone do not pin. The label pack's FILE ROLES
    // ride `search_label_fingerprint`: a role-aware manifest binds every
    // (filename, sha256, role) triple into that digest, so persisting it here
    // persists the role mapping.
    f["head_init_identity"] = pwo5_head_identity;
    f["search_label_fingerprint"] = search_label_fingerprint;
    // --- section 8.6's update-1 sampler digest. Matched WITHIN a triplet
    // (T/P/H share a base seed) and expected to differ BETWEEN triplets, so it
    // is neither a treatment nor a global matched field.
    f["update1_sampler_digest"] = pwo5_update1_sampler_digest;
    open_spiel::g_pwo5_manifest_active = true;
    std::cout << "[PWO-5] manifest fingerprint "
              << open_spiel::ComputePwo5Fingerprint(f) << std::endl;
  }

  std::atomic<uint64_t> total_env_steps{0};
  std::atomic<uint64_t> next_episode_id{0};
  // Phase 18B online-collection persistent state (restored from the manifest on
  // resume, persisted at every checkpoint). aux_next_episode_id_persist TRAILS
  // the collector's live cursor by one pipeline step: it is the next-episode-id
  // as of the LAST CONSUMED update, which is exactly what a resume must continue
  // from so episode ids neither repeat nor skip.
  uint64_t aux_next_episode_id_persist = 0;
  int64_t aux_cum_accepted = 0, aux_cum_rejected = 0;
  int64_t aux_cum_role_searches[open_spiel::OnlineCollectionState::kNumSearchRoles] =
      {0, 0, 0, 0};
  int64_t aux_cum_role_accepted[open_spiel::OnlineCollectionState::kNumSearchRoles] =
      {0, 0, 0, 0};
  int64_t aux_cum_granted = 0, aux_cum_organic = 0;
  std::string aux_hash_chain = "";
  // Search-PI resume state, restored from the manifest's own "search_pi" block.
  open_spiel::SearchPiState search_pi_resume;
  bool search_pi_resume_present = false;
  int start_update = 1;
  int target_end_update = absl::GetFlag(FLAGS_target_end_update);
  std::string parity_source_manifest;
  std::string parity_source_manifest_sha256;
  int64_t parity_source_global_update = -1;
  std::string parity_source_checkpoint_uuid;

  if (vrpo_one_update) {
    if (init_mode != "vrpo_one_update" ||
        absl::GetFlag(FLAGS_hidden_dim) != 2048 ||
        absl::GetFlag(FLAGS_num_blocks) != 8 ||
        absl::GetFlag(FLAGS_nonlinear_value_head) ||
        !absl::GetFlag(FLAGS_aux_target_path).empty() ||
        !absl::GetFlag(FLAGS_artifact_manifest).empty()) {
      SpielFatalError(
          "VRPO one-update requires its dedicated init mode, production actor layout, and no legacy source/auxiliary manifest path");
    }
    const open_spiel::VrpoPhase4ArmConfig* selected_arm =
        open_spiel::FindCanonicalVrpoPhase4Arm(
            absl::GetFlag(FLAGS_vrpo_one_update_arm_id));
    if (selected_arm == nullptr) {
      SpielFatalError("VRPO one-update selected arm is not canonical");
    }
    vrpo_one_update_arm = *selected_arm;
    std::error_code binary_ec;
    const auto binary_path =
        std::filesystem::read_symlink("/proc/self/exe", binary_ec);
    if (binary_ec || binary_path.empty()) {
      SpielFatalError("VRPO one-update cannot resolve the executed binary");
    }
    size_t binary_size = 0;
    const std::string binary_sha256 =
        open_spiel::ComputeFileSHA256(binary_path.string(), &binary_size);
    if (binary_sha256 != absl::GetFlag(FLAGS_vrpo_one_update_binary_sha256) ||
        static_cast<int64_t>(binary_size) !=
            absl::GetFlag(FLAGS_vrpo_one_update_binary_size)) {
      SpielFatalError("VRPO one-update executed binary identity mismatch");
    }
    vrpo_one_update_startup.game = absl::GetFlag(FLAGS_game);
    vrpo_one_update_startup.registration_id =
        absl::GetFlag(FLAGS_vrpo_one_update_registration_id);
    vrpo_one_update_startup.selected_arm_id = vrpo_one_update_arm.arm_id;
    vrpo_one_update_startup.input_archive =
        absl::GetFlag(FLAGS_vrpo_one_update_input_archive);
    vrpo_one_update_startup.output_root =
        absl::GetFlag(FLAGS_vrpo_one_update_output_root);
    vrpo_one_update_startup.source_root =
        absl::GetFlag(FLAGS_vrpo_one_update_source_root);
    vrpo_one_update_startup.source_code_sha256 =
        vrpo_one_update_source_identity.combined_sha256;
    vrpo_one_update_startup.executed_binary_sha256 = binary_sha256;
    vrpo_one_update_startup.executed_binary_size =
        static_cast<int64_t>(binary_size);
    vrpo_one_update_startup.profile =
        absl::GetFlag(FLAGS_vrpo_one_update_profile);
    vrpo_one_update_startup.rollout_games =
        absl::GetFlag(FLAGS_rollout_games);
    vrpo_one_update_startup.threads = absl::GetFlag(FLAGS_threads);
    vrpo_one_update_startup.diagnostics_only =
        absl::GetFlag(FLAGS_diagnostics_only);
    vrpo_one_update_startup.init_mode = init_mode;
    vrpo_one_update_startup.rollout_amp = absl::GetFlag(FLAGS_rollout_amp);
    vrpo_one_update_startup.train_amp = absl::GetFlag(FLAGS_train_amp);
    vrpo_one_update_startup.allow_tf32 = absl::GetFlag(FLAGS_allow_tf32);
    vrpo_one_update_startup.pipeline = absl::GetFlag(FLAGS_pipeline);
    vrpo_one_update_startup.online_search_collection =
        absl::GetFlag(FLAGS_online_search_collection);
    vrpo_one_update_startup.search_pi_mode =
        absl::GetFlag(FLAGS_search_pi_mode);
    vrpo_one_update_startup.train_value_only =
        absl::GetFlag(FLAGS_train_value_only);
    vrpo_one_update_startup.sample_counterfactual_states =
        absl::GetFlag(FLAGS_sample_counterfactual_states);
    vrpo_one_update_startup.has_search_label_dir =
        !absl::GetFlag(FLAGS_search_label_dir).empty();
    vrpo_one_update_startup.ordinary_checkpoint_writes_enabled =
        absl::GetFlag(FLAGS_checkpoint_interval) != 0 ||
        absl::GetFlag(FLAGS_save_final_checkpoint);
    vrpo_one_update_startup.shaped_reward_weight =
        absl::GetFlag(FLAGS_shaped_reward_weight);
    vrpo_one_update_startup.tleilaxu_breadcrumb_weight =
        absl::GetFlag(FLAGS_tleilaxu_breadcrumb_weight);
    vrpo_one_update_startup.tleilaxu_level7_breadcrumb_weight =
        absl::GetFlag(FLAGS_tleilaxu_level7_breadcrumb_weight);
    vrpo_one_update_startup.specimen_exchange_penalty =
        absl::GetFlag(FLAGS_specimen_exchange_penalty);
    vrpo_one_update_startup.reward_scale = absl::GetFlag(FLAGS_reward_scale);
    vrpo_one_update_startup.gamma = absl::GetFlag(FLAGS_gamma);
    vrpo_one_update_startup.lambda = absl::GetFlag(FLAGS_gae_lambda);
    // Bind the actual policy transform before strict archive loading. The
    // startup contract below rejects NaN and every non-exact arm/cap pairing.
    vrpo_one_update_startup.logit_cap = absl::GetFlag(FLAGS_logit_cap);
    std::string vrpo_one_update_error;
    if (!open_spiel::ValidateVrpoPhase4eSelectedArmLogitCap(
            vrpo_one_update_startup, &vrpo_one_update_error)) {
      SpielFatalError("VRPO one-update startup rejected: " +
                      vrpo_one_update_error);
    }
    json::Object input_manifest;
    if (!open_spiel::ReadVrpoPhase4eExternalBinding(
            vrpo_one_update_startup.input_archive, vrpo_one_update_arm,
            &vrpo_one_update_binding, &vrpo_one_update_layout, &input_manifest,
            &vrpo_one_update_input_identity, &vrpo_one_update_error)) {
      SpielFatalError("VRPO one-update input binding rejected: " +
                      vrpo_one_update_error);
    }
    vrpo_one_update_startup.input_archive_sha256 =
        vrpo_one_update_input_identity.combined_sha256;
    if (!open_spiel::ValidateVrpoPhase4eStartupConfig(
            vrpo_one_update_startup, &vrpo_one_update_error)) {
      SpielFatalError("VRPO one-update startup rejected: " +
                      vrpo_one_update_error);
    }
    // Never instantiate an archive-provided module shape: test fixtures and
    // alternate layouts are rejected before the canonical Q is constructed.
    if (vrpo_one_update_layout.test_fixture ||
        vrpo_one_update_layout.label != "production_dune_vrpo_layout_v1" ||
        vrpo_one_update_layout.actor_observation_dim != obs_size ||
        vrpo_one_update_layout.actor_hidden_dim != 2048 ||
        vrpo_one_update_layout.actor_action_dim != action_size ||
        vrpo_one_update_layout.actor_residual_blocks != 8) {
      SpielFatalError("VRPO one-update permits production expanded archives only");
    }
    vrpo_one_update_q = std::make_shared<open_spiel::DuneVrpoQNetImpl>(
        vrpo_one_update_binding.q_init_seed);
    vrpo_one_update_q->to(device);
    if (!open_spiel::MakeVrpoFreshOptimizers(
            *training_model, *vrpo_one_update_q, &vrpo_one_update_optimizers,
            &vrpo_one_update_error) ||
        !open_spiel::LoadAndValidateVrpoExpandedCheckpoint(
            vrpo_one_update_startup.input_archive, vrpo_one_update_arm,
            vrpo_one_update_binding, vrpo_one_update_layout, training_model,
            vrpo_one_update_q, vrpo_one_update_optimizers, &input_manifest,
            &vrpo_one_update_error, nullptr, 0,
            vrpo_one_update_binding.start_episode_id)) {
      SpielFatalError("VRPO one-update strict expanded archive load failed: " +
                      vrpo_one_update_error);
    }
    next_episode_id.store(vrpo_one_update_binding.start_episode_id);
    total_env_steps.store(0);
    start_update = 1;
  } else if (vrpo_schedule_screen) {
    if (init_mode != "vrpo_schedule_screen" ||
        absl::GetFlag(FLAGS_hidden_dim) != 2048 ||
        absl::GetFlag(FLAGS_num_blocks) != 8 ||
        absl::GetFlag(FLAGS_nonlinear_value_head) ||
        !absl::GetFlag(FLAGS_aux_target_path).empty() ||
        !absl::GetFlag(FLAGS_artifact_manifest).empty() ||
        absl::GetFlag(FLAGS_ppo_update_epochs) != 4 ||
        absl::GetFlag(FLAGS_learning_rate) != 2.5e-4) {
      SpielFatalError(
          "VRPO schedule screen requires its dedicated mode, production "
          "actor layout, canonical outer PPO flags, and no legacy source path");
    }
    const open_spiel::VrpoPhase4ArmConfig* ppo_cap10 =
        open_spiel::FindCanonicalVrpoPhase4Arm("PPO_CAP10");
    if (ppo_cap10 == nullptr) {
      SpielFatalError("VRPO schedule screen cannot resolve PPO_CAP10");
    }
    std::error_code binary_ec;
    const auto binary_path =
        std::filesystem::read_symlink("/proc/self/exe", binary_ec);
    if (binary_ec || binary_path.empty()) {
      SpielFatalError("VRPO schedule screen cannot resolve executed binary");
    }
    size_t binary_size = 0;
    const std::string binary_sha256 =
        open_spiel::ComputeFileSHA256(binary_path.string(), &binary_size);
    if (binary_sha256 !=
            absl::GetFlag(FLAGS_vrpo_schedule_screen_binary_sha256) ||
        static_cast<int64_t>(binary_size) !=
            absl::GetFlag(FLAGS_vrpo_schedule_screen_binary_size)) {
      SpielFatalError("VRPO schedule screen executed binary identity mismatch");
    }
    json::Object input_manifest;
    std::string schedule_error;
    if (!open_spiel::ReadVrpoPhase4eExternalBinding(
            absl::GetFlag(FLAGS_vrpo_schedule_screen_input_archive),
            *ppo_cap10, &vrpo_schedule_binding, &vrpo_schedule_layout,
            &input_manifest, &vrpo_schedule_input_identity,
            &schedule_error)) {
      SpielFatalError("VRPO schedule screen input binding rejected: " +
                      schedule_error);
    }
    vrpo_schedule_startup.game = absl::GetFlag(FLAGS_game);
    vrpo_schedule_startup.init_mode = init_mode;
    vrpo_schedule_startup.profile =
        absl::GetFlag(FLAGS_vrpo_schedule_screen_profile);
    vrpo_schedule_startup.registration_id =
        absl::GetFlag(FLAGS_vrpo_schedule_screen_registration_id);
    vrpo_schedule_startup.input_archive =
        absl::GetFlag(FLAGS_vrpo_schedule_screen_input_archive);
    vrpo_schedule_startup.output_root =
        absl::GetFlag(FLAGS_vrpo_schedule_screen_output_root);
    vrpo_schedule_startup.source_root =
        absl::GetFlag(FLAGS_vrpo_schedule_screen_source_root);
    vrpo_schedule_startup.source_code_sha256 =
        vrpo_schedule_source_identity.combined_sha256;
    vrpo_schedule_startup.executed_binary_sha256 = binary_sha256;
    vrpo_schedule_startup.executed_binary_size = binary_size;
    vrpo_schedule_startup.rollout_games = absl::GetFlag(FLAGS_rollout_games);
    vrpo_schedule_startup.threads = absl::GetFlag(FLAGS_threads);
    vrpo_schedule_startup.eval_batch_size =
        absl::GetFlag(FLAGS_eval_batch_size);
    vrpo_schedule_startup.eval_timeout_ms =
        absl::GetFlag(FLAGS_eval_timeout_ms);
    vrpo_schedule_startup.evaluator_device_synchronize =
        absl::GetFlag(FLAGS_evaluator_device_synchronize);
    vrpo_schedule_startup.deterministic_rollout_eval =
        absl::GetFlag(FLAGS_deterministic_rollout_eval);
    vrpo_schedule_startup.seed_scheme_version =
        absl::GetFlag(FLAGS_seed_scheme_version);
    vrpo_schedule_startup.runtime_device_is_cuda = device.is_cuda();
    vrpo_schedule_startup.runtime_device_index =
        device.has_index() ? device.index() : c10::cuda::current_device();
    vrpo_schedule_startup.one_gpu_process =
        vrpo_schedule_screen && device.is_cuda() &&
        vrpo_schedule_startup.runtime_device_index >= 0;
    vrpo_schedule_startup.runtime_process_id =
        static_cast<int64_t>(::getpid());
    vrpo_schedule_startup.base_seed = vrpo_schedule_binding.base_seed;
    vrpo_schedule_startup.start_episode_id =
        vrpo_schedule_binding.start_episode_id;
    vrpo_schedule_startup.diagnostics_only =
        absl::GetFlag(FLAGS_diagnostics_only);
    vrpo_schedule_startup.rollout_amp = absl::GetFlag(FLAGS_rollout_amp);
    vrpo_schedule_startup.train_amp = absl::GetFlag(FLAGS_train_amp);
    vrpo_schedule_startup.allow_tf32 = absl::GetFlag(FLAGS_allow_tf32);
    vrpo_schedule_startup.pipeline = absl::GetFlag(FLAGS_pipeline);
    vrpo_schedule_startup.online_search_collection =
        absl::GetFlag(FLAGS_online_search_collection);
    vrpo_schedule_startup.search_pi_mode =
        absl::GetFlag(FLAGS_search_pi_mode);
    vrpo_schedule_startup.train_value_only =
        absl::GetFlag(FLAGS_train_value_only);
    vrpo_schedule_startup.sample_counterfactual_states =
        absl::GetFlag(FLAGS_sample_counterfactual_states);
    vrpo_schedule_startup.has_search_label_dir =
        !absl::GetFlag(FLAGS_search_label_dir).empty();
    vrpo_schedule_startup.ordinary_checkpoint_writes_enabled =
        absl::GetFlag(FLAGS_checkpoint_interval) != 0 ||
        absl::GetFlag(FLAGS_save_final_checkpoint);
    vrpo_schedule_startup.shaped_reward_weight =
        absl::GetFlag(FLAGS_shaped_reward_weight);
    vrpo_schedule_startup.tleilaxu_breadcrumb_weight =
        absl::GetFlag(FLAGS_tleilaxu_breadcrumb_weight);
    vrpo_schedule_startup.tleilaxu_level7_breadcrumb_weight =
        absl::GetFlag(FLAGS_tleilaxu_level7_breadcrumb_weight);
    vrpo_schedule_startup.specimen_exchange_penalty =
        absl::GetFlag(FLAGS_specimen_exchange_penalty);
    vrpo_schedule_startup.reward_scale = absl::GetFlag(FLAGS_reward_scale);
    vrpo_schedule_startup.gamma = absl::GetFlag(FLAGS_gamma);
    vrpo_schedule_startup.lambda = absl::GetFlag(FLAGS_gae_lambda);
    vrpo_schedule_startup.logit_cap = absl::GetFlag(FLAGS_logit_cap);
    vrpo_schedule_startup.ppo_minibatches =
        open_spiel::kVrpoTrainingMinibatchesPerEpoch;
    vrpo_schedule_startup.ppo_minibatch_size =
        absl::GetFlag(FLAGS_ppo_minibatch_size);
    vrpo_schedule_startup.clip_epsilon =
        absl::GetFlag(FLAGS_ppo_clip_epsilon);
    vrpo_schedule_startup.entropy_coefficient =
        absl::GetFlag(FLAGS_entropy_coef);
    vrpo_schedule_startup.value_coefficient =
        absl::GetFlag(FLAGS_value_coef);
    vrpo_schedule_startup.gradient_clip_norm =
        absl::GetFlag(FLAGS_grad_clip_norm);
    vrpo_schedule_startup.normalize_advantages =
        absl::GetFlag(FLAGS_normalize_advantages);
    vrpo_schedule_startup.clip_value_loss = true;
    if (static_cast<uint64_t>(absl::GetFlag(FLAGS_seed)) !=
        vrpo_schedule_binding.base_seed) {
      SpielFatalError(
          "VRPO schedule screen --seed does not match source base seed");
    }
    if (!open_spiel::ValidateVrpoScheduleStartupConfig(
            vrpo_schedule_startup, &schedule_error)) {
      SpielFatalError("VRPO schedule screen startup rejected: " +
                      schedule_error);
    }
    if (vrpo_schedule_layout.test_fixture ||
        vrpo_schedule_layout.label != "production_dune_vrpo_layout_v1" ||
        vrpo_schedule_layout.actor_observation_dim != obs_size ||
        vrpo_schedule_layout.actor_hidden_dim != 2048 ||
        vrpo_schedule_layout.actor_action_dim != action_size ||
        vrpo_schedule_layout.actor_residual_blocks != 8) {
      SpielFatalError("VRPO schedule screen permits production archive only");
    }
    open_spiel::ResetVrpoQInstrumentation();
    open_spiel::ResetVrpoScheduleQTargetInstrumentation();
    vrpo_schedule_q = std::make_shared<open_spiel::DuneVrpoQNetImpl>(
        vrpo_schedule_binding.q_init_seed);
    vrpo_schedule_q->to(device);
    if (!vrpo_schedule_deadline.Check("before source actor/Q load",
                                      &schedule_error) ||
        !open_spiel::LoadAndValidateVrpoScheduleSourceModules(
            vrpo_schedule_startup.input_archive, *ppo_cap10,
            vrpo_schedule_binding, vrpo_schedule_layout,
            vrpo_schedule_input_identity, training_model, vrpo_schedule_q,
            input_manifest, &schedule_error) ||
        !open_spiel::ValidateVrpoScheduleQProvenanceInstrumentation(
            open_spiel::kVrpoScheduleQRole, &schedule_error) ||
        !vrpo_schedule_deadline.Check("after source actor/Q load",
                                      &schedule_error)) {
      SpielFatalError("VRPO schedule screen source load failed: " +
                      schedule_error);
    }
    next_episode_id.store(vrpo_schedule_binding.start_episode_id);
    total_env_steps.store(0);
    start_update = 1;
  } else if (vrpo_ppo_pilot) {
    if (init_mode != "vrpo_ppo_pilot" ||
        absl::GetFlag(FLAGS_hidden_dim) != 2048 ||
        absl::GetFlag(FLAGS_num_blocks) != 8 ||
        absl::GetFlag(FLAGS_nonlinear_value_head) ||
        !absl::GetFlag(FLAGS_aux_target_path).empty() ||
        !absl::GetFlag(FLAGS_artifact_manifest).empty() ||
        absl::GetFlag(FLAGS_total_updates) !=
            open_spiel::kVrpoPpoPilotUpdates ||
        absl::GetFlag(FLAGS_ppo_update_epochs) !=
            open_spiel::kVrpoPpoPilotActorEpochs ||
        absl::GetFlag(FLAGS_learning_rate) !=
            open_spiel::kVrpoPpoPilotLearningRate ||
        absl::GetFlag(FLAGS_anneal_lr)) {
      SpielFatalError(
          "PPO pilot requires its dedicated mode, production actor layout, "
          "fixed four-update LR5e-6/E1 profile, and no legacy paths/annealing");
    }
    const open_spiel::VrpoPhase4ArmConfig* ppo_cap10 =
        open_spiel::FindCanonicalVrpoPhase4Arm("PPO_CAP10");
    if (ppo_cap10 == nullptr) {
      SpielFatalError("PPO pilot cannot resolve PPO_CAP10 origin contract");
    }
    json::Object input_manifest;
    std::string pilot_error;
    vrpo_ppo_pilot_startup = vrpo_ppo_pilot_preflight;
    if (!open_spiel::ReadVrpoPhase4eExternalBinding(
            vrpo_ppo_pilot_startup.input_archive, *ppo_cap10,
            &vrpo_ppo_pilot_binding, &vrpo_ppo_pilot_layout,
            &input_manifest, &vrpo_ppo_pilot_input_identity, &pilot_error)) {
      SpielFatalError("PPO pilot input binding rejected: " + pilot_error);
    }
    vrpo_ppo_pilot_startup.game = absl::GetFlag(FLAGS_game);
    vrpo_ppo_pilot_startup.init_mode = init_mode;
    vrpo_ppo_pilot_startup.profile =
        absl::GetFlag(FLAGS_vrpo_ppo_pilot_profile);
    vrpo_ppo_pilot_startup.registration_id =
        absl::GetFlag(FLAGS_vrpo_ppo_pilot_registration_id);
    vrpo_ppo_pilot_startup.source_root =
        absl::GetFlag(FLAGS_vrpo_ppo_pilot_source_root);
    vrpo_ppo_pilot_startup.source_code_sha256 =
        vrpo_ppo_pilot_source_identity.combined_sha256;
    vrpo_ppo_pilot_startup.rollout_games =
        absl::GetFlag(FLAGS_rollout_games);
    vrpo_ppo_pilot_startup.threads = absl::GetFlag(FLAGS_threads);
    vrpo_ppo_pilot_startup.eval_batch_size =
        absl::GetFlag(FLAGS_eval_batch_size);
    vrpo_ppo_pilot_startup.eval_timeout_ms =
        absl::GetFlag(FLAGS_eval_timeout_ms);
    vrpo_ppo_pilot_startup.evaluator_device_synchronize =
        absl::GetFlag(FLAGS_evaluator_device_synchronize);
    vrpo_ppo_pilot_startup.deterministic_rollout_eval =
        absl::GetFlag(FLAGS_deterministic_rollout_eval);
    vrpo_ppo_pilot_startup.seed_scheme_version =
        absl::GetFlag(FLAGS_seed_scheme_version);
    vrpo_ppo_pilot_startup.runtime_device_is_cuda = device.is_cuda();
    vrpo_ppo_pilot_startup.runtime_device_index =
        device.has_index() ? device.index() : c10::cuda::current_device();
    vrpo_ppo_pilot_startup.one_gpu_process =
        device.is_cuda() && vrpo_ppo_pilot_startup.runtime_device_index >= 0;
    vrpo_ppo_pilot_startup.runtime_process_id =
        static_cast<int64_t>(::getpid());
    vrpo_ppo_pilot_startup.base_seed =
        static_cast<uint64_t>(absl::GetFlag(FLAGS_seed));
    vrpo_ppo_pilot_startup.start_episode_id =
        absl::GetFlag(FLAGS_start_episode_id);
    vrpo_ppo_pilot_startup.diagnostics_only =
        absl::GetFlag(FLAGS_diagnostics_only);
    vrpo_ppo_pilot_startup.rollout_amp = absl::GetFlag(FLAGS_rollout_amp);
    vrpo_ppo_pilot_startup.train_amp = absl::GetFlag(FLAGS_train_amp);
    vrpo_ppo_pilot_startup.allow_tf32 = absl::GetFlag(FLAGS_allow_tf32);
    vrpo_ppo_pilot_startup.pipeline = absl::GetFlag(FLAGS_pipeline);
    vrpo_ppo_pilot_startup.online_search_collection =
        absl::GetFlag(FLAGS_online_search_collection);
    vrpo_ppo_pilot_startup.search_pi_mode =
        absl::GetFlag(FLAGS_search_pi_mode);
    vrpo_ppo_pilot_startup.train_value_only =
        absl::GetFlag(FLAGS_train_value_only);
    vrpo_ppo_pilot_startup.sample_counterfactual_states =
        absl::GetFlag(FLAGS_sample_counterfactual_states);
    vrpo_ppo_pilot_startup.has_search_label_dir =
        !absl::GetFlag(FLAGS_search_label_dir).empty();
    vrpo_ppo_pilot_startup.ordinary_checkpoint_writes_enabled =
        absl::GetFlag(FLAGS_checkpoint_interval) != 0 ||
        absl::GetFlag(FLAGS_save_final_checkpoint);
    vrpo_ppo_pilot_startup.anneal_lr = absl::GetFlag(FLAGS_anneal_lr);
    vrpo_ppo_pilot_startup.learning_rate =
        absl::GetFlag(FLAGS_learning_rate);
    vrpo_ppo_pilot_startup.ppo_update_epochs =
        absl::GetFlag(FLAGS_ppo_update_epochs);
    vrpo_ppo_pilot_startup.shaped_reward_weight =
        absl::GetFlag(FLAGS_shaped_reward_weight);
    vrpo_ppo_pilot_startup.tleilaxu_breadcrumb_weight =
        absl::GetFlag(FLAGS_tleilaxu_breadcrumb_weight);
    vrpo_ppo_pilot_startup.tleilaxu_level7_breadcrumb_weight =
        absl::GetFlag(FLAGS_tleilaxu_level7_breadcrumb_weight);
    vrpo_ppo_pilot_startup.specimen_exchange_penalty =
        absl::GetFlag(FLAGS_specimen_exchange_penalty);
    vrpo_ppo_pilot_startup.reward_scale = absl::GetFlag(FLAGS_reward_scale);
    vrpo_ppo_pilot_startup.gamma = absl::GetFlag(FLAGS_gamma);
    vrpo_ppo_pilot_startup.lambda = absl::GetFlag(FLAGS_gae_lambda);
    vrpo_ppo_pilot_startup.logit_cap = absl::GetFlag(FLAGS_logit_cap);
    vrpo_ppo_pilot_startup.ppo_minibatches =
        open_spiel::kVrpoTrainingMinibatchesPerEpoch;
    vrpo_ppo_pilot_startup.ppo_minibatch_size =
        absl::GetFlag(FLAGS_ppo_minibatch_size);
    vrpo_ppo_pilot_startup.clip_epsilon =
        absl::GetFlag(FLAGS_ppo_clip_epsilon);
    vrpo_ppo_pilot_startup.entropy_coefficient =
        absl::GetFlag(FLAGS_entropy_coef);
    vrpo_ppo_pilot_startup.value_coefficient =
        absl::GetFlag(FLAGS_value_coef);
    vrpo_ppo_pilot_startup.gradient_clip_norm =
        absl::GetFlag(FLAGS_grad_clip_norm);
    vrpo_ppo_pilot_startup.normalize_advantages =
        absl::GetFlag(FLAGS_normalize_advantages);
    vrpo_ppo_pilot_startup.clip_value_loss =
        absl::GetFlag(FLAGS_ppo_clip_value_loss);
    if (!open_spiel::ValidateVrpoPpoPilotStartupConfig(
            vrpo_ppo_pilot_startup, &pilot_error)) {
      SpielFatalError("PPO pilot startup rejected: " + pilot_error);
    }
    if (vrpo_ppo_pilot_input_identity.combined_sha256 !=
            vrpo_ppo_pilot_startup.origin_archive_sha256 ||
        vrpo_ppo_pilot_input_identity.combined_sha256 !=
            vrpo_ppo_pilot_preloaded_input_identity.combined_sha256) {
      SpielFatalError("PPO pilot origin archive identity mismatch");
    }
    if (vrpo_ppo_pilot_layout.test_fixture ||
        vrpo_ppo_pilot_layout.label != "production_dune_vrpo_layout_v1" ||
        vrpo_ppo_pilot_layout.actor_observation_dim != obs_size ||
        vrpo_ppo_pilot_layout.actor_hidden_dim != 2048 ||
        vrpo_ppo_pilot_layout.actor_action_dim != action_size ||
        vrpo_ppo_pilot_layout.actor_residual_blocks != 8) {
      SpielFatalError("PPO pilot permits production archive only");
    }
    open_spiel::ResetVrpoQInstrumentation();
    open_spiel::ResetVrpoScheduleQTargetInstrumentation();
    vrpo_ppo_pilot_q = std::make_shared<open_spiel::DuneVrpoQNetImpl>(
        vrpo_ppo_pilot_binding.q_init_seed);
    vrpo_ppo_pilot_q->to(device);
    if (!vrpo_ppo_pilot_deadline.Check("before source actor/Q load",
                                       &pilot_error) ||
        !open_spiel::LoadAndValidateVrpoScheduleSourceModules(
            vrpo_ppo_pilot_startup.input_archive, *ppo_cap10,
            vrpo_ppo_pilot_binding, vrpo_ppo_pilot_layout,
            vrpo_ppo_pilot_input_identity, training_model,
            vrpo_ppo_pilot_q, input_manifest, &pilot_error) ||
        !open_spiel::MakeVrpoScheduleFreshActorOptimizer(
            *training_model, open_spiel::kVrpoPpoPilotLearningRate,
            &vrpo_ppo_pilot_actor_optimizer, &pilot_error) ||
        !open_spiel::InitializeVrpoPpoPilot(
            vrpo_ppo_pilot_startup, vrpo_ppo_pilot_input_identity,
            vrpo_ppo_pilot_layout, training_model, vrpo_ppo_pilot_q,
            *vrpo_ppo_pilot_actor_optimizer, vrpo_ppo_pilot_deadline,
            &vrpo_ppo_pilot_state, &pilot_error)) {
      SpielFatalError("PPO pilot source/init failed: " + pilot_error);
    }
    next_episode_id.store(vrpo_ppo_pilot_startup.start_episode_id);
    total_env_steps.store(0);
    start_update = 1;
  } else if (vrpo_ppo_continuation) {
    if (init_mode != "vrpo_ppo_continuation" ||
        absl::GetFlag(FLAGS_hidden_dim) != 2048 ||
        absl::GetFlag(FLAGS_num_blocks) != 8 ||
        absl::GetFlag(FLAGS_nonlinear_value_head) ||
        !absl::GetFlag(FLAGS_aux_target_path).empty() ||
        !absl::GetFlag(FLAGS_artifact_manifest).empty() ||
        absl::GetFlag(FLAGS_total_updates) !=
            open_spiel::kVrpoPpoContinuationNewUpdates ||
        absl::GetFlag(FLAGS_ppo_update_epochs) !=
            open_spiel::kVrpoPpoContinuationActorEpochs ||
        absl::GetFlag(FLAGS_learning_rate) !=
            open_spiel::kVrpoPpoContinuationLearningRate ||
        absl::GetFlag(FLAGS_anneal_lr)) {
      SpielFatalError(
          "PPO continuation requires its dedicated production layout and fixed U5-U8 LR5e-6/E1 profile");
    }
    std::string continuation_error;
    vrpo_ppo_continuation_startup = vrpo_ppo_continuation_preflight;
    vrpo_ppo_continuation_startup.game = absl::GetFlag(FLAGS_game);
    vrpo_ppo_continuation_startup.init_mode = init_mode;
    vrpo_ppo_continuation_startup.profile =
        absl::GetFlag(FLAGS_vrpo_ppo_continuation_profile);
    vrpo_ppo_continuation_startup.registration_id =
        absl::GetFlag(FLAGS_vrpo_ppo_continuation_registration_id);
    vrpo_ppo_continuation_startup.source_root =
        absl::GetFlag(FLAGS_vrpo_ppo_continuation_source_root);
    vrpo_ppo_continuation_startup.source_code_sha256 =
        vrpo_ppo_continuation_source_identity.combined_sha256;
    vrpo_ppo_continuation_startup.rollout_games =
        absl::GetFlag(FLAGS_rollout_games);
    vrpo_ppo_continuation_startup.threads = absl::GetFlag(FLAGS_threads);
    vrpo_ppo_continuation_startup.eval_batch_size =
        absl::GetFlag(FLAGS_eval_batch_size);
    vrpo_ppo_continuation_startup.eval_timeout_ms =
        absl::GetFlag(FLAGS_eval_timeout_ms);
    vrpo_ppo_continuation_startup.evaluator_device_synchronize =
        absl::GetFlag(FLAGS_evaluator_device_synchronize);
    vrpo_ppo_continuation_startup.deterministic_rollout_eval =
        absl::GetFlag(FLAGS_deterministic_rollout_eval);
    vrpo_ppo_continuation_startup.seed_scheme_version =
        absl::GetFlag(FLAGS_seed_scheme_version);
    vrpo_ppo_continuation_startup.runtime_device_is_cuda = device.is_cuda();
    vrpo_ppo_continuation_startup.runtime_device_index =
        device.has_index() ? device.index() : c10::cuda::current_device();
    vrpo_ppo_continuation_startup.one_gpu_process = device.is_cuda() &&
        vrpo_ppo_continuation_startup.runtime_device_index >= 0;
    vrpo_ppo_continuation_startup.runtime_process_id =
        static_cast<int64_t>(::getpid());
    vrpo_ppo_continuation_startup.base_seed =
        static_cast<uint64_t>(absl::GetFlag(FLAGS_seed));
    vrpo_ppo_continuation_startup.start_episode_id =
        absl::GetFlag(FLAGS_start_episode_id);
    vrpo_ppo_continuation_startup.diagnostics_only =
        absl::GetFlag(FLAGS_diagnostics_only);
    vrpo_ppo_continuation_startup.rollout_amp =
        absl::GetFlag(FLAGS_rollout_amp);
    vrpo_ppo_continuation_startup.train_amp =
        absl::GetFlag(FLAGS_train_amp);
    vrpo_ppo_continuation_startup.allow_tf32 =
        absl::GetFlag(FLAGS_allow_tf32);
    vrpo_ppo_continuation_startup.pipeline = absl::GetFlag(FLAGS_pipeline);
    vrpo_ppo_continuation_startup.online_search_collection =
        absl::GetFlag(FLAGS_online_search_collection);
    vrpo_ppo_continuation_startup.search_pi_mode =
        absl::GetFlag(FLAGS_search_pi_mode);
    vrpo_ppo_continuation_startup.train_value_only =
        absl::GetFlag(FLAGS_train_value_only);
    vrpo_ppo_continuation_startup.sample_counterfactual_states =
        absl::GetFlag(FLAGS_sample_counterfactual_states);
    vrpo_ppo_continuation_startup.has_search_label_dir =
        !absl::GetFlag(FLAGS_search_label_dir).empty();
    vrpo_ppo_continuation_startup.ordinary_checkpoint_writes_enabled =
        absl::GetFlag(FLAGS_checkpoint_interval) != 0 ||
        absl::GetFlag(FLAGS_save_final_checkpoint);
    vrpo_ppo_continuation_startup.anneal_lr =
        absl::GetFlag(FLAGS_anneal_lr);
    vrpo_ppo_continuation_startup.learning_rate =
        absl::GetFlag(FLAGS_learning_rate);
    vrpo_ppo_continuation_startup.ppo_update_epochs =
        absl::GetFlag(FLAGS_ppo_update_epochs);
    vrpo_ppo_continuation_startup.shaped_reward_weight =
        absl::GetFlag(FLAGS_shaped_reward_weight);
    vrpo_ppo_continuation_startup.tleilaxu_breadcrumb_weight =
        absl::GetFlag(FLAGS_tleilaxu_breadcrumb_weight);
    vrpo_ppo_continuation_startup.tleilaxu_level7_breadcrumb_weight =
        absl::GetFlag(FLAGS_tleilaxu_level7_breadcrumb_weight);
    vrpo_ppo_continuation_startup.specimen_exchange_penalty =
        absl::GetFlag(FLAGS_specimen_exchange_penalty);
    vrpo_ppo_continuation_startup.reward_scale =
        absl::GetFlag(FLAGS_reward_scale);
    vrpo_ppo_continuation_startup.gamma = absl::GetFlag(FLAGS_gamma);
    vrpo_ppo_continuation_startup.lambda =
        absl::GetFlag(FLAGS_gae_lambda);
    vrpo_ppo_continuation_startup.logit_cap =
        absl::GetFlag(FLAGS_logit_cap);
    vrpo_ppo_continuation_startup.ppo_minibatches =
        open_spiel::kVrpoTrainingMinibatchesPerEpoch;
    vrpo_ppo_continuation_startup.ppo_minibatch_size =
        absl::GetFlag(FLAGS_ppo_minibatch_size);
    vrpo_ppo_continuation_startup.clip_epsilon =
        absl::GetFlag(FLAGS_ppo_clip_epsilon);
    vrpo_ppo_continuation_startup.entropy_coefficient =
        absl::GetFlag(FLAGS_entropy_coef);
    vrpo_ppo_continuation_startup.value_coefficient =
        absl::GetFlag(FLAGS_value_coef);
    vrpo_ppo_continuation_startup.gradient_clip_norm =
        absl::GetFlag(FLAGS_grad_clip_norm);
    vrpo_ppo_continuation_startup.normalize_advantages =
        absl::GetFlag(FLAGS_normalize_advantages);
    vrpo_ppo_continuation_startup.clip_value_loss =
        absl::GetFlag(FLAGS_ppo_clip_value_loss);
    if (!open_spiel::ValidateVrpoPpoContinuationStartupConfig(
            vrpo_ppo_continuation_startup, &continuation_error)) {
      SpielFatalError("PPO continuation startup rejected: " +
                      continuation_error);
    }
    vrpo_ppo_continuation_layout.label =
        "production_dune_vrpo_layout_v1";
    vrpo_ppo_continuation_layout.test_fixture = false;
    vrpo_ppo_continuation_layout.actor_observation_dim = obs_size;
    vrpo_ppo_continuation_layout.actor_hidden_dim = 2048;
    vrpo_ppo_continuation_layout.actor_action_dim = action_size;
    vrpo_ppo_continuation_layout.actor_residual_blocks = 8;
    if (obs_size != 5580 || action_size != 2391) {
      SpielFatalError("PPO continuation permits the production Dune layout only");
    }
    vrpo_ppo_continuation_binding.source_actor_model_sha256 =
        "68febee771509f88446286cc50983afd22d64f7772f6866def77bafd4aae36d2";
    vrpo_ppo_continuation_binding.source_actor_manifest_sha256 =
        "95df08e8c9ccdba0a402b610474022fa22374215842f7b0d6243ce09359b4512";
    vrpo_ppo_continuation_binding.experiment_uuid =
        "7b7b58a7-297c-4d98-8bb2-5f8f1b0a4c30";
    vrpo_ppo_continuation_binding.q_init_seed =
        open_spiel::kVrpoPpoContinuationQSeed;
    vrpo_ppo_continuation_binding.base_seed =
        open_spiel::kVrpoPpoContinuationBaseSeed;
    vrpo_ppo_continuation_binding.start_episode_id =
        open_spiel::kVrpoPpoContinuationStartEpisodeId;
    vrpo_ppo_continuation_binding.end_episode_id_inclusive =
        open_spiel::kVrpoPpoContinuationEndEpisodeIdInclusive;
    open_spiel::ResetVrpoQInstrumentation();
    open_spiel::ResetVrpoScheduleQTargetInstrumentation();
    vrpo_ppo_continuation_q =
        std::make_shared<open_spiel::DuneVrpoQNetImpl>(
            open_spiel::kVrpoPpoContinuationQSeed);
    vrpo_ppo_continuation_q->to(device, torch::kFloat32);
    if (!open_spiel::LoadAndInitializeVrpoPpoContinuation(
            vrpo_ppo_continuation_startup, training_model,
            vrpo_ppo_continuation_q, device,
            vrpo_ppo_continuation_deadline,
            &vrpo_ppo_continuation_actor_optimizer,
            &vrpo_ppo_continuation_state, &continuation_error)) {
      SpielFatalError("PPO continuation strict U4 load/init failed: " +
                      continuation_error);
    }
    next_episode_id.store(vrpo_ppo_continuation_startup.start_episode_id);
    total_env_steps.store(0);
    start_update = 1;
  } else if (init_mode == "diagnostic") {
    // A model-only, read-only load path. It exists specifically so a completed
    // checkpoint can be audited without pretending to resume it, loading Adam
    // moments, synthesizing a bootstrap manifest, or making any checkpoint
    // path writable.
    if (!numerical_parity && !vrpo_diagnostics) {
      SpielFatalError(
          "init_mode=diagnostic is reserved for numerical parity or VRPO diagnostics");
    }
    const std::string model_path = absl::GetFlag(FLAGS_model_checkpoint);
    parity_source_manifest = absl::GetFlag(FLAGS_artifact_manifest);
    if (!std::filesystem::is_regular_file(model_path) ||
        !std::filesystem::is_regular_file(parity_source_manifest)) {
      SpielFatalError(
          "Diagnostic model checkpoint and provenance manifest must both be regular files");
    }
    size_t manifest_size = 0;
    parity_source_manifest_sha256 = ComputeFileSHA256(
        parity_source_manifest, &manifest_size);
    std::ifstream source_stream(parity_source_manifest);
    const std::string source_text((std::istreambuf_iterator<char>(source_stream)),
                                  std::istreambuf_iterator<char>());
    auto parsed = json::FromString(source_text);
    if (!parsed.has_value() || !parsed->IsObject()) {
      SpielFatalError("Numerical parity provenance manifest is malformed JSON");
    }
    const json::Object& source = parsed->GetObject();
    auto required_string = [&](const char* key) -> std::string {
      auto it = source.find(key);
      if (it == source.end() || !it->second.IsString()) {
        SpielFatalError(std::string("Numerical parity manifest missing string '") +
                        key + "'");
      }
      return it->second.GetString();
    };
    auto required_int = [&](const char* key) -> int64_t {
      auto it = source.find(key);
      if (it == source.end() || !it->second.IsInt()) {
        SpielFatalError(std::string("Numerical parity manifest missing integer '") +
                        key + "'");
      }
      return it->second.GetInt();
    };
    size_t model_size = 0;
    const std::string model_sha256 = ComputeFileSHA256(model_path, &model_size);
    if (model_sha256 != required_string("model_sha256") ||
        static_cast<int64_t>(model_size) != required_int("model_file_size")) {
      SpielFatalError(
          "Numerical parity model bytes do not match the provenance manifest");
    }
    if (absl::GetFlag(FLAGS_hidden_dim) != required_int("hidden_dim") ||
        absl::GetFlag(FLAGS_num_blocks) != required_int("num_blocks")) {
      SpielFatalError(
          "Numerical parity architecture flags do not match the provenance manifest");
    }
    parity_source_global_update = required_int("global_update");
    parity_source_checkpoint_uuid = required_string("checkpoint_uuid");
    LoadModelCheckpoint(training_model, model_path, device);
    start_update = static_cast<int>(parity_source_global_update + 1);
    next_episode_id.store(
        static_cast<uint64_t>(absl::GetFlag(FLAGS_start_episode_id)));
    total_env_steps.store(
        static_cast<uint64_t>(absl::GetFlag(FLAGS_start_env_steps)));
    std::cout << "Loaded read-only numerical-parity source model "
              << model_path << " sha256=" << model_sha256
              << "; optimizer checkpoint was not loaded.\n";
  } else if (init_mode == "random") {
    std::string model_path = absl::GetFlag(FLAGS_model_checkpoint);
    std::string optim_path = absl::GetFlag(FLAGS_optim_checkpoint);
    std::filesystem::path m_path = model_path;
    m_path.replace_extension(".json");
    std::string manifest_path = m_path.string();
    if (std::filesystem::exists(model_path) || std::filesystem::exists(optim_path) || std::filesystem::exists(manifest_path)) {
      SpielFatalError("init_mode=random but checkpoint or manifest file already exists. Refusing to overwrite.");
    }
    std::cout << "Starting fresh training run (init_mode=random).\n";
  } else if (init_mode == "checkpoint") {
    if (absl::GetFlag(FLAGS_pipeline)) {
      SpielFatalError("init_mode=checkpoint with pipeline=true is rejected because prefetched rollout is not persisted.");
    }
    std::string model_path = absl::GetFlag(FLAGS_model_checkpoint);
    std::string optim_path = absl::GetFlag(FLAGS_optim_checkpoint);
    std::filesystem::path m_path = model_path;
    m_path.replace_extension(".json");
    std::string manifest_path = m_path.string();

    // WO-20 acceptance-contract guard, deliberately BEFORE the fingerprint
    // check: adding the source to the fingerprint already makes a pre-WO-20
    // 18B checkpoint fail to resume, but it would fail as an anonymous hash
    // mismatch. Diagnose the specific cause first, while we can still name it.
    //
    // The cumulative accepted/rejected counters and the accepted-target hash
    // chain are only meaningful within ONE acceptance contract. A manifest
    // written before WO-20 has no source field and accumulated its counters
    // against the post-noise tree prior, so silently extending them under the
    // new raw-prior default would blend two contracts into one total.
    if (absl::GetFlag(FLAGS_online_search_collection)) {
      OnlineCollectionState prior_state;
      std::string oc_probe_err;
      if (ReadOnlineCollectionState(manifest_path, prior_state, oc_probe_err) &&
          prior_state.present) {
        const std::string effective =
            AcceptancePriorSourceName(EffectiveAcceptancePriorSource());
        const bool stated = !absl::GetFlag(FLAGS_collector_acceptance_prior).empty();
        if (prior_state.acceptance_prior_source.empty() && !stated) {
          SpielFatalError(absl::StrCat(
              "Refusing to resume: ", manifest_path, " predates WO-20 and has "
              "no online_collection.acceptance_prior_source, so its "
              "cum_accepted=", prior_state.cum_accepted, " / cum_rejected=",
              prior_state.cum_rejected, " were accumulated against the "
              "POST-NOISE tree prior. Continuing under the new default ('",
              effective, "') would mix two coverage contracts in one counter "
              "set and one accepted-target hash chain.\n"
              "  To continue that run's original contract: "
              "--collector_acceptance_prior=tree_prior\n"
              "  To adopt the new contract knowingly (counters and hash chain "
              "then span both): --collector_acceptance_prior=", effective, "\n"
              "  Note either choice changes the config fingerprint relative to "
              "the stored one, which is intended — the semantics did change."));
        }
        if (!prior_state.acceptance_prior_source.empty() &&
            prior_state.acceptance_prior_source != effective) {
          SpielFatalError(absl::StrCat(
              "Refusing to resume: ", manifest_path, " collected under "
              "acceptance_prior_source='", prior_state.acceptance_prior_source,
              "' but this run resolves to '", effective, "'. Its counters and "
              "hash chain cannot be extended across a change of coverage "
              "contract. Pass --collector_acceptance_prior=",
              prior_state.acceptance_prior_source, " to continue it, or start "
              "a fresh run."));
        }
      }
    }

    CheckpointManifest manifest;
    std::string err;
    // Legacy fingerprints predate critic-remediation flags and therefore
    // cannot prove which parameters or data source were active. Never use
    // that compatibility escape hatch for value-only resumes.
    std::string legacy_fingerprint = absl::GetFlag(FLAGS_train_value_only)
        ? ""
        : ComputeLegacyConfigFingerprint();
    if (!ParseAndValidateManifest(manifest_path, model_path, optim_path,
                                  master, target_end_update,
                                  absl::GetFlag(FLAGS_seed_scheme_version),
                                  config_fingerprint,
                                  pre_precision_config_fingerprint,
                                  absl::GetFlag(FLAGS_rollout_amp),
                                  absl::GetFlag(FLAGS_allow_tf32),
                                  search_label_fingerprint,
                                  absl::GetFlag(FLAGS_hidden_dim),
                                  absl::GetFlag(FLAGS_num_blocks),
                                  manifest, err, legacy_fingerprint)) {
      SpielFatalError(err);
    }

    // ---------------------------------------------------------------------
    // PWO-5 Appendix A.1 note 3: the within-arm invariant. Every field of the
    // block -- both target flags, the split digest, the three head
    // coefficients, the three sampler-shape flags, the Huber delta, the
    // head-init constant AND identity, --emit_canary_columns, the label pack's
    // role-bearing fingerprint, and the update-1 sampler digest -- MUST NOT
    // change across a resume or the u600 extension. An arm's own configuration
    // is fixed for its lifetime.
    //
    // The check runs HERE: before the model is loaded, before the optimizer is
    // restored, and before a single update executes.
    // ---------------------------------------------------------------------
    {
      std::string stored_fp, stored_block;
      open_spiel::json::Object stored_fields;
      bool stored_present = false;
      {
        std::ifstream mfs(manifest_path);
        std::string content((std::istreambuf_iterator<char>(mfs)),
                            std::istreambuf_iterator<char>());
        auto parsed = open_spiel::json::FromString(content);
        if (parsed && parsed->IsObject()) {
          const auto& obj = parsed->GetObject();
          auto it = obj.find("pwo5");
          auto fp_it = obj.find("pwo5_fingerprint");
          if (it != obj.end() && it->second.IsObject()) {
            stored_fields = it->second.GetObject();
            stored_present = true;
            stored_block = open_spiel::json::ToString(stored_fields);
          }
          if (fp_it != obj.end() && fp_it->second.IsString()) {
            stored_fp = fp_it->second.GetString();
          }
        }
      }
      if (stored_present != open_spiel::g_pwo5_manifest_active) {
        SpielFatalError(
            std::string("PWO-5 resume contract: the manifest ") +
            (stored_present ? "CARRIES" : "does NOT carry") +
            " a pwo5 block but this run " +
            (open_spiel::g_pwo5_manifest_active ? "IS" : "is NOT") +
            " a PWO-5 arm (--aux_target_path " +
            (open_spiel::g_pwo5_manifest_active ? "set" : "empty") +
            "). Resuming across that boundary would silently change the "
            "architecture, the auxiliary data path, or both.");
      }
      if (stored_present) {
        const std::string current_fp =
            open_spiel::ComputePwo5Fingerprint(open_spiel::g_pwo5_manifest_fields);
        if (stored_fp != current_fp) {
          // Name the fields that moved rather than emitting an anonymous hash
          // mismatch: a resume failure a human cannot diagnose is a resume
          // failure a human will work around.
          std::string detail;
          for (const auto& kv : open_spiel::g_pwo5_manifest_fields) {
            auto sit = stored_fields.find(kv.first);
            const std::string now = open_spiel::json::ToString(kv.second);
            const std::string was = (sit == stored_fields.end())
                                        ? std::string("<absent>")
                                        : open_spiel::json::ToString(sit->second);
            if (now != was) {
              detail += "\n    " + kv.first + ": manifest=" + was +
                        "  this run=" + now;
            }
          }
          for (const auto& kv : stored_fields) {
            if (open_spiel::g_pwo5_manifest_fields.find(kv.first) ==
                open_spiel::g_pwo5_manifest_fields.end()) {
              detail += "\n    " + kv.first + ": manifest=" +
                        open_spiel::json::ToString(kv.second) +
                        "  this run=<absent>";
            }
          }
          SpielFatalError(
              "PWO-5 resume contract VIOLATED (Appendix A.1 note 3).\n"
              "  manifest pwo5_fingerprint: " + stored_fp + "\n"
              "  this run's fingerprint:    " + current_fp + "\n"
              "  fields that differ:" + detail +
              "\n  An arm's own configuration is fixed for its lifetime. "
              "Training is refused before it continues.");
        }
        std::cout << "[PWO-5] resume contract verified: pwo5_fingerprint "
                  << current_fp << " matches the manifest ("
                  << open_spiel::g_pwo5_manifest_fields.size() << " fields)."
                  << std::endl;
        (void)stored_block;
      }
    }

    int global_update = manifest.global_update;
    if (global_update >= target_end_update) {
      SpielFatalError(absl::StrFormat(
          "global_update (%d) >= target_end_update (%d). "
          "Resuming is not possible as training is already finished.",
          global_update, target_end_update));
    }
    uint64_t manifest_env_steps = manifest.total_env_steps;
    uint64_t manifest_episode_id = manifest.next_episode_id;
    std::string manifest_run_uuid = manifest.run_uuid;

    // Phase 18B exact resume: restore online-collection cursor + cumulative
    // counters + accepted-target hash chain from the manifest (if present).
    {
      open_spiel::OnlineCollectionState restored;
      std::string oc_err;
      if (!open_spiel::ReadOnlineCollectionState(manifest_path, restored, oc_err)) {
        SpielFatalError("Failed to read online_collection manifest state: " + oc_err);
      }
      if (restored.present) {
        aux_next_episode_id_persist = restored.next_auxiliary_episode_id;
        aux_cum_accepted = restored.cum_accepted;
        aux_cum_rejected = restored.cum_rejected;
        aux_cum_granted = restored.cum_granted;
        aux_cum_organic = restored.cum_organic;
        for (int r = 0; r < open_spiel::OnlineCollectionState::kNumSearchRoles;
             ++r) {
          aux_cum_role_searches[r] = restored.cum_role_searches[r];
          aux_cum_role_accepted[r] = restored.cum_role_accepted[r];
        }
        aux_hash_chain = restored.accepted_hash_chain;
        std::cout << absl::StrFormat(
            "[18B] Restored online-collection state: next_aux_episode_id=%llu "
            "cum_accepted=%lld cum_granted=%lld cum_organic=%lld\n",
            (unsigned long long)aux_next_episode_id_persist,
            (long long)aux_cum_accepted, (long long)aux_cum_granted,
            (long long)aux_cum_organic);
      }
    }

    // Search-PI exact resume: the generation counter, the episode cursor, the
    // cumulative row/simulation counts and the target hash chain all come from
    // the manifest. Absent block = a checkpoint from another mode, which is not
    // an error here; the mode branch below decides whether that is acceptable.
    {
      std::ifstream mf(manifest_path);
      if (mf) {
        std::stringstream buf;
        buf << mf.rdbuf();
        auto parsed = json::FromString(buf.str());
        if (parsed.has_value() && parsed->IsObject()) {
          const json::Object& root = parsed->GetObject();
          auto it = root.find("search_pi");
          if (it != root.end() && it->second.IsObject()) {
            std::string sp_err;
            if (!open_spiel::ReadSearchPiState(it->second.GetObject(),
                                               &search_pi_resume, &sp_err)) {
              SpielFatalError("Failed to read search_pi manifest state: " +
                              sp_err);
            }
            search_pi_resume_present = true;
            std::cout << absl::StrFormat(
                "[search-PI] Restored state: generation=%d "
                "next_episode_id=%lld cum_rows=%lld chain=%s\n",
                search_pi_resume.generation,
                (long long)search_pi_resume.next_episode_id,
                (long long)search_pi_resume.cum_rows,
                search_pi_resume.target_hash_chain.substr(0, 16).c_str());
          }
        }
      }
    }

    try {
      LoadModelCheckpoint(training_model, model_path, device);
      if (anchor_model) {
        LoadModelCheckpoint(anchor_model, model_path, device);
        anchor_model->eval();
        for (auto& p : anchor_model->parameters()) {
          p.set_requires_grad(false);
        }
        std::cout << "[INFO] Loaded anchor model for policy KL penalty.\n";
      }
      if (absl::GetFlag(FLAGS_search_pi_mode)) {
        // See the bootstrap path: this mode owns a separate optimizer and
        // loads it itself, and only from its own lineage.
        std::cout << "[search-PI] Skipping the PPO optimizer load here; the "
                     "lane's own optimizer is handled in the search-PI branch.\n";
      } else if (!absl::GetFlag(FLAGS_train_value_only)) {
        // PWO-5 section 7.2: a pre-head archive migrates; a post-head archive
        // loads normally.
        if (!open_spiel::LoadOptimizerCheckpointMigrating(
                training_model, *optimizer, optim_path, device)) {
          torch::load(*optimizer, optim_path, device);
        }
      } else {
        std::cout << "[INFO] train_value_only is true: keeping fresh value-only optimizer. Skipping optimizer checkpoint load.\n";
      }
    } catch (const c10::Error& e) {
      SpielFatalError("LibTorch load failed: " + std::string(e.msg()));
    }

    open_spiel::SetOptimizerLearningRate(*optimizer, absl::GetFlag(FLAGS_learning_rate));
    for (size_t g = 0; g < optimizer->param_groups().size(); ++g) {
      auto& options =
          static_cast<torch::optim::AdamWOptions&>(
              optimizer->param_groups()[g].options());
      // PWO-5 section 7.4: group 2 is the auxiliary-head group and must stay
      // at weight_decay = 0.0. The pre-PWO-5 form of this loop was
      // `g == 0 ? policy_wd : wd`, which would have applied --weight_decay to
      // the heads on every resume and silently drifted a head-off arm's head
      // parameters away from their initial values.
      options.weight_decay(g == 0 ? absl::GetFlag(FLAGS_policy_weight_decay)
                          : g == 1 ? absl::GetFlag(FLAGS_weight_decay)
                                   : 0.0);
    }
    for (auto& pair : optimizer->state()) {
      if (auto* state =
              dynamic_cast<torch::optim::AdamWParamState*>(pair.second.get())) {
        if (state->exp_avg().defined()) state->exp_avg() = state->exp_avg().to(device);
        if (state->exp_avg_sq().defined()) state->exp_avg_sq() = state->exp_avg_sq().to(device);
        if (state->max_exp_avg_sq().defined()) {
          state->max_exp_avg_sq() = state->max_exp_avg_sq().to(device);
        }
      }
    }

    start_update = global_update + 1;
    total_env_steps.store(manifest_env_steps);
    next_episode_id.store(manifest_episode_id);
    run_uuid = manifest_run_uuid;

    std::cout << "Successfully verified and loaded checkpoint manifest. Resuming from update "
              << start_update << " to " << target_end_update << ".\n";
  } else if (init_mode == "bootstrap") {
    std::string model_path = absl::GetFlag(FLAGS_model_checkpoint);
    std::string optim_path = absl::GetFlag(FLAGS_optim_checkpoint);
    if (model_path.find("artifacts/baselines") != std::string::npos ||
        model_path.find("artifacts/branch_a_frozen") != std::string::npos) {
      SpielFatalError("Refusing to bootstrap directly inside the frozen baseline or branch_a_frozen directory: " + model_path +
                      "\nRun bootstrap beside a BRANCH COPY in a separate output directory.");
    }
    if (!std::filesystem::exists(model_path)) {
      SpielFatalError("Model file not found for bootstrap: " + model_path);
    }
    if (!std::filesystem::exists(optim_path)) {
      SpielFatalError("Optimizer file not found for bootstrap: " + optim_path);
    }

    try {
      LoadModelCheckpoint(training_model, model_path, device);
      if (anchor_model) {
        LoadModelCheckpoint(anchor_model, model_path, device);
        anchor_model->eval();
        for (auto& p : anchor_model->parameters()) {
          p.set_requires_grad(false);
        }
        std::cout << "[INFO] Loaded anchor model for policy KL penalty.\n";
      }
      if (absl::GetFlag(FLAGS_search_pi_mode)) {
        // The search-PI lane never steps this optimizer -- it builds its own,
        // and on a checkpoint with no search_pi block it deliberately starts
        // that one FRESH rather than inheriting another objective's Adam
        // moments. Loading the source optimizer here would therefore be reading
        // an artifact the mode has already refused to use, and it fails outright
        // when the source was written under a different parameter-group layout
        // (e.g. a PWO-5 lineage's three groups against this run's two).
        std::cout << "[search-PI] Skipping the PPO optimizer load: this mode "
                     "takes WEIGHTS from the checkpoint and builds its own "
                     "optimizer.\n";
      } else if (!absl::GetFlag(FLAGS_train_value_only)) {
        // PWO-5 section 7.2: the bootstrap path is where the Branch-A u2450
        // optimizer -- written before the three heads existed -- is migrated.
        if (!open_spiel::LoadOptimizerCheckpointMigrating(
                training_model, *optimizer, optim_path, device)) {
          torch::load(*optimizer, optim_path, device);
        }
      } else {
        std::cout << "[INFO] train_value_only is true: using fresh value-only optimizer (skipping baseline optimizer load).\n";
      }
    } catch (const c10::Error& e) {
      SpielFatalError("LibTorch load failed during bootstrap: " + std::string(e.msg()));
    }

    size_t model_size = 0;
    std::string model_hash = open_spiel::ComputeFileSHA256(model_path, &model_size);
    size_t optim_size = 0;
    std::string optim_hash = open_spiel::ComputeFileSHA256(optim_path, &optim_size);

    std::filesystem::path m_path = model_path;
    m_path.replace_extension(".json");
    std::string manifest_path = m_path.string();
    std::string manifest_tmp = manifest_path + ".tmp";

    json::Object manifest_obj;
    manifest_obj["schema_version"] = static_cast<int64_t>(2);
    manifest_obj["global_update"] = static_cast<int64_t>(absl::GetFlag(FLAGS_start_update));
    manifest_obj["target_end_update"] = static_cast<int64_t>(absl::GetFlag(FLAGS_target_end_update));
    manifest_obj["total_env_steps"] = static_cast<int64_t>(absl::GetFlag(FLAGS_start_env_steps));
    manifest_obj["next_episode_id"] = static_cast<int64_t>(absl::GetFlag(FLAGS_start_episode_id));
    manifest_obj["base_seed"] = static_cast<int64_t>(absl::GetFlag(FLAGS_seed));
    manifest_obj["seed_scheme_version"] = static_cast<int64_t>(absl::GetFlag(FLAGS_seed_scheme_version));
    manifest_obj["config_fingerprint"] = config_fingerprint;
    WritePpoPrecisionManifestFields(
        manifest_obj, absl::GetFlag(FLAGS_rollout_amp),
        absl::GetFlag(FLAGS_allow_tf32));
    manifest_obj["search_label_fingerprint"] = search_label_fingerprint;
    manifest_obj["run_uuid"] = open_spiel::GenerateUUID();
    manifest_obj["checkpoint_uuid"] = open_spiel::GenerateUUID();
    manifest_obj["model_filename"] = std::filesystem::path(model_path).filename().string();
    manifest_obj["model_file_size"] = static_cast<int64_t>(model_size);
    manifest_obj["model_sha256"] = model_hash;
    manifest_obj["optimizer_filename"] = std::filesystem::path(optim_path).filename().string();
    manifest_obj["optimizer_file_size"] = static_cast<int64_t>(optim_size);
    manifest_obj["optimizer_sha256"] = optim_hash;
    manifest_obj["hidden_dim"] = static_cast<int64_t>(absl::GetFlag(FLAGS_hidden_dim));
    manifest_obj["num_blocks"] = static_cast<int64_t>(absl::GetFlag(FLAGS_num_blocks));
    manifest_obj["legacy_migration_provenance"] = "Synthesized via init_mode=bootstrap";
    // PWO-5 Appendix A.1 note 3: the same block the checkpoint writer emits, so
    // the bootstrap manifest an arm resumes FROM already carries the contract.
    if (g_pwo5_manifest_active) {
      manifest_obj["pwo5"] = json::Value(g_pwo5_manifest_fields);
      manifest_obj["pwo5_fingerprint"] =
          json::Value(ComputePwo5Fingerprint(g_pwo5_manifest_fields));
    }

    {
      std::ofstream ofs(manifest_tmp);
      if (!ofs) {
        SpielFatalError("Could not open manifest temp file for writing: " + manifest_tmp);
      }
      ofs << json::ToString(manifest_obj, true);
    }
    std::filesystem::rename(manifest_tmp, manifest_path);
    std::cout << "Successfully bootstrapped manifest at " << manifest_path << "\n";
    exit(0);
  } else if (init_mode == "validate_legacy") {
    std::string model_path = absl::GetFlag(FLAGS_model_checkpoint);
    std::string optim_path = absl::GetFlag(FLAGS_optim_checkpoint);
    std::filesystem::path manifest_path;
    std::string artifact_manifest_flag = absl::GetFlag(FLAGS_artifact_manifest);
    if (!artifact_manifest_flag.empty()) {
      manifest_path = artifact_manifest_flag;
    } else {
      manifest_path = std::filesystem::path(model_path).parent_path() / "manifest.json";
    }

    if (!std::filesystem::exists(manifest_path)) {
      SpielFatalError("init_mode=validate_legacy but Phase 1 manifest file not found at: " + manifest_path.string());
    }

    std::ifstream ifs(manifest_path.string());
    if (!ifs) {
      SpielFatalError("Could not open manifest file at: " + manifest_path.string());
    }
    std::string content((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
    auto val_opt = open_spiel::json::FromString(content);
    if (!val_opt.has_value() || !val_opt->IsObject()) {
      SpielFatalError("Manifest file is malformed JSON at: " + manifest_path.string());
    }
    const auto& manifest_obj = val_opt->GetObject();

    auto get_object_field = [&](const std::string& key) -> const json::Object& {
      auto it = manifest_obj.find(key);
      if (it == manifest_obj.end() || !it->second.IsObject()) {
        SpielFatalError("Manifest missing required object field: " + key);
      }
      return it->second.GetObject();
    };

    auto get_nested_string = [](const json::Object& obj, const std::string& key) -> std::string {
      auto it = obj.find(key);
      if (it == obj.end() || !it->second.IsString()) {
        SpielFatalError("Nested object missing required string field: " + key);
      }
      return it->second.GetString();
    };

    auto get_nested_int = [](const json::Object& obj, const std::string& key) -> int64_t {
      auto it = obj.find(key);
      if (it == obj.end() || !it->second.IsInt()) {
        SpielFatalError("Nested object missing required int field: " + key);
      }
      return it->second.GetInt();
    };

    const auto& model_obj = get_object_field("model");
    const auto& optim_obj = get_object_field("optimizer");
    const auto& arch_obj = get_object_field("architecture");

    std::string expected_model_sha256 = get_nested_string(model_obj, "sha256");
    size_t expected_model_size = get_nested_int(model_obj, "size_bytes");

    std::string expected_optim_sha256 = get_nested_string(optim_obj, "sha256");
    size_t expected_optim_size = get_nested_int(optim_obj, "size_bytes");

    int64_t manifest_hidden_dim = get_nested_int(arch_obj, "hidden_dim");
    int64_t manifest_num_blocks = get_nested_int(arch_obj, "num_blocks");
    int64_t manifest_obs_size = get_nested_int(arch_obj, "observation_size");
    int64_t manifest_action_size = get_nested_int(arch_obj, "action_size");

    if (!std::filesystem::exists(model_path)) {
      SpielFatalError("Model file not found: " + model_path);
    }
    if (!std::filesystem::exists(optim_path)) {
      SpielFatalError("Optimizer file not found: " + optim_path);
    }

    size_t actual_model_size = 0;
    std::string actual_model_hash = open_spiel::ComputeFileSHA256(model_path, &actual_model_size);
    if (actual_model_size != expected_model_size) {
      SpielFatalError(absl::StrFormat("Model file size mismatch. Manifest: %d, Actual: %d", expected_model_size, actual_model_size));
    }
    if (actual_model_hash != expected_model_sha256) {
      SpielFatalError("Model SHA-256 hash mismatch. Manifest: " + expected_model_sha256 + ", Actual: " + actual_model_hash);
    }

    size_t actual_optim_size = 0;
    std::string actual_optim_hash = open_spiel::ComputeFileSHA256(optim_path, &actual_optim_size);
    if (actual_optim_size != expected_optim_size) {
      SpielFatalError(absl::StrFormat("Optimizer file size mismatch. Manifest: %d, Actual: %d", expected_optim_size, actual_optim_size));
    }
    if (actual_optim_hash != expected_optim_sha256) {
      SpielFatalError("Optimizer SHA-256 hash mismatch. Manifest: " + expected_optim_sha256 + ", Actual: " + actual_optim_hash);
    }

    if (absl::GetFlag(FLAGS_hidden_dim) != manifest_hidden_dim) {
      SpielFatalError(absl::StrFormat("hidden_dim mismatch. Flags: %d, Manifest: %d", absl::GetFlag(FLAGS_hidden_dim), manifest_hidden_dim));
    }
    if (absl::GetFlag(FLAGS_num_blocks) != manifest_num_blocks) {
      SpielFatalError(absl::StrFormat("num_blocks mismatch. Flags: %d, Manifest: %d", absl::GetFlag(FLAGS_num_blocks), manifest_num_blocks));
    }
    if (obs_size != manifest_obs_size) {
      SpielFatalError(absl::StrFormat("observation_size mismatch. Game: %d, Manifest: %d", obs_size, manifest_obs_size));
    }
    if (action_size != manifest_action_size) {
      SpielFatalError(absl::StrFormat("action_size mismatch. Game: %d, Manifest: %d", action_size, manifest_action_size));
    }

    try {
      LoadModelCheckpoint(training_model, model_path, device);
    } catch (const c10::Error& e) {
      SpielFatalError("LibTorch load failed: " + std::string(e.msg()));
    }

    int64_t actual_param_count = 0;
    for (const auto& param : training_model->parameters()) {
      actual_param_count += param.numel();
    }

    auto dummy_model = std::make_shared<open_spiel::SharedDunePolicyValueNetImpl>(
        manifest_obs_size, manifest_hidden_dim, manifest_action_size, manifest_num_blocks, absl::GetFlag(FLAGS_nonlinear_value_head));
    int64_t expected_param_count = 0;
    for (const auto& param : dummy_model->parameters()) {
      expected_param_count += param.numel();
    }

    if (actual_param_count != expected_param_count) {
      SpielFatalError(absl::StrFormat("Parameter count mismatch. Expected: %d (calculated from architecture), Actual: %d", expected_param_count, actual_param_count));
    }

    std::cout << absl::StrFormat("Legacy checkpoint validation successful!\n"
                                 "  Model parameter count: %d\n"
                                 "  Model size and hash match Phase 1 manifest.\n"
                                 "  Optimizer size and hash match Phase 1 manifest.\n", actual_param_count);
    exit(0);
  } else {
    SpielFatalError("Unsupported init_mode: " + init_mode +
                    " (expected random, checkpoint, bootstrap, "
                    "validate_legacy, diagnostic, vrpo_one_update, or "
                    "vrpo_schedule_screen, vrpo_ppo_pilot, or "
                    "vrpo_ppo_continuation)");
  }

  std::shared_mutex sync_mutex;
  std::mutex eval_mutex;
  open_spiel::SyncModels(training_model, inference_model, &sync_mutex);

  std::shared_ptr<open_spiel::IGameEvaluator> evaluator;
  if (absl::GetFlag(FLAGS_deterministic_rollout_eval)) {
    evaluator = std::make_shared<open_spiel::DeterministicEvaluator>(
        inference_model, device, &eval_mutex, &sync_mutex);
  } else {
    evaluator = std::make_shared<open_spiel::BatchedEvaluator>(
        inference_model, absl::GetFlag(FLAGS_eval_batch_size),
        absl::GetFlag(FLAGS_eval_timeout_ms), device, &sync_mutex, 0.0f,
        absl::GetFlag(FLAGS_evaluator_device_synchronize),
        /*high_priority_stream=*/false,
        /*emit_batch_membership=*/numerical_parity,
        absl::GetFlag(FLAGS_rollout_amp),
        absl::GetFlag(FLAGS_allow_tf32));
  }

  // =========================================================================
  // Agent-turn search policy iteration (search-PI)
  // =========================================================================
  //
  // RUN-SCOPED and terminal: this branch owns the whole run and returns. It
  // never reaches the PPO rollout below, never calls TrainPpoUpdate, and never
  // constructs a PPO surrogate, ratio, clip, entropy bonus, advantage term or
  // target-KL stop. That is the point -- PF-C's failure was that a competing
  // PPO policy objective was applied to the same weights at the same time.
  if (absl::GetFlag(FLAGS_search_pi_mode)) {
    // --- Mutual exclusion, enforced at startup rather than at first use -----
    if (absl::GetFlag(FLAGS_online_search_collection)) {
      SpielFatalError(
          "--search_pi_mode and --online_search_collection are mutually "
          "exclusive: the second is the closed PF-C combined-optimization mode "
          "(0.10 * (CE + 0.5*value_MSE) alongside PPO), and running both would "
          "reintroduce exactly the competing objective this lane removes.");
    }
    if (!absl::GetFlag(FLAGS_search_label_dir).empty()) {
      SpielFatalError(
          "--search_pi_mode and --search_label_dir are mutually exclusive "
          "(legacy offline distillation runs its own optimizer step).");
    }
    if (absl::GetFlag(FLAGS_pipeline)) {
      SpielFatalError(
          "--search_pi_mode and --pipeline are mutually exclusive. Pipelining "
          "collects generation N+1 against the snapshot while generation N is "
          "still learning, which breaks the frozen-collect -> learn -> sync "
          "boundary this mode is defined by. Re-enabling it requires PROVING "
          "the snapshot semantics, not assuming them.");
    }
    if (absl::GetFlag(FLAGS_search_leader_draft)) {
      SpielFatalError(
          "--search_leader_draft is not applicable in --search_pi_mode: this "
          "lane searches agent-turn decisions only, and its scope is not a "
          "tunable knob. The Leader teacher's production pins must not leak "
          "in; the manifest records search_leader_draft=false.");
    }
    if (absl::GetFlag(FLAGS_search_pi_seed_domain) == 0) {
      SpielFatalError(
          "--search_pi_seed_domain must be set nonzero (no silent default).");
    }
    if (absl::GetFlag(FLAGS_diagnostics_path).empty()) {
      // REQUIRED, not optional. Every scientific instrument this lane exists to
      // provide -- the CE/value split, the two gradient norms, the per-module
      // cosines, the per-role target and raw entropy, KL(target||raw), the
      // zero-simulation-by-role proof and the kept-despite-the-old-gate count --
      // lives only in the sidecar beside this path. The manifest carries the
      // cursor and the config, not the measurements. PF-C's per-update telemetry
      // is gone and its diagnosis tables are no longer reproducible from
      // artifacts; a run that silently produces none is not worth having.
      SpielFatalError(
          "--diagnostics_path is required with --search_pi_mode: the lane's "
          "learner and per-role telemetry are written to "
          "<diagnostics_path minus extension>_search_pi.jsonl, and without it "
          "the generation would leave no durable measurements.");
    }
    if (absl::GetFlag(FLAGS_train_value_only)) {
      // train_value_only rebuilds the optimizer with a single parameter group
      // and freezes the policy head -- incoherent with a lane whose primary
      // objective is a policy-target cross-entropy.
      SpielFatalError(
          "--train_value_only is incompatible with --search_pi_mode: this lane's "
          "first-class objective is the search-visit policy target.");
    }

    // --- Registration profile, resolved before anything reads a pin ---------
    //
    // Every pin below branches on this, so it is resolved first and exactly
    // once. The default reproduces the live post-Gate-8 3b path; the two rung-4
    // profiles change only what they are documented to change, and neither can
    // reach a check the other profile owns.
    const std::string registration_profile =
        absl::GetFlag(FLAGS_search_pi_registration_profile);
    const bool profile_3b = registration_profile == "post_gate8_3b";
    const bool profile_rung4a = registration_profile == "rung4a";
    const bool profile_rung4_replay = registration_profile == "rung4_replay";
    if (!profile_3b && !profile_rung4a && !profile_rung4_replay) {
      SpielFatalError("unrecognized --search_pi_registration_profile value '" +
                      registration_profile +
                      "' (expected post_gate8_3b, rung4a or rung4_replay).");
    }
    // Both rung-4 profiles are production-lane runs by construction. The pin
    // block below is only ENTERED on the concurrent path, so a rung-4 launch
    // that left the batch target at 0 would run with no geometry pin, no
    // collector pin and none of the learner manipulation checks -- a substitute
    // experiment wearing the profile's name. Refused rather than tolerated.
    if (!profile_3b && absl::GetFlag(FLAGS_search_pi_batch_target) <= 0) {
      SpielFatalError(
          "a rung4 profile requires the concurrent production lane: at a zero "
          "batch target the collector pin, the geometry pin and the learner "
          "manipulation checks are all skipped.");
    }

    const int64_t registered_first_episode_id =
        absl::GetFlag(FLAGS_search_pi_registered_first_episode_id);
    const int registered_max_generation =
        absl::GetFlag(FLAGS_search_pi_max_generation);
    const std::string expected_initial_student_digest =
        absl::GetFlag(FLAGS_search_pi_expected_initial_student_sha256);
    // Under 3b these three are not knobs. Honouring them there would let a NEW
    // flag disarm an OLD pin the live run depends on -- max_generation=99 would
    // erase the 1..4 resume window, and a restated episode base would let a
    // second lineage overwrite the registered one. They are inert by
    // construction rather than by launcher discipline.
    if (profile_3b && (registered_first_episode_id != 600000 ||
                       registered_max_generation != 5 ||
                       !expected_initial_student_digest.empty())) {
      SpielFatalError(
          "the post_gate8_3b profile pins the episode base 600000, the 1..4 "
          "resume window and the fresh-student identity; restating any of them "
          "requires a rung4 profile.");
    }

    // rung4a: the second frozen archive. Its SHA is verified against the file
    // like the collector's, but it is caller-supplied rather than one
    // registered constant, because rung 4a's whole question is which prior is
    // paired with the pinned teacher.
    const std::string policy_prior_path =
        absl::GetFlag(FLAGS_search_pi_policy_prior_model_checkpoint);
    const std::string policy_prior_expected_sha =
        absl::GetFlag(FLAGS_search_pi_policy_prior_model_sha256);
    if (!policy_prior_path.empty() && !profile_rung4a) {
      SpielFatalError(
          "a policy-prior archive is legal only under the rung4a profile; this "
          "invocation selected the profile " + registration_profile + ".");
    }
    if (policy_prior_path.empty() != policy_prior_expected_sha.empty()) {
      SpielFatalError(
          "the policy-prior archive and its SHA-256 are set together or not at "
          "all: an archive without a hash is an unverified second teacher, and "
          "a hash without an archive verifies nothing.");
    }

    // --- Origin reset: legality, checked HERE ------------------------------
    //
    // Beside the other profile-legality checks and BEFORE any checkpoint is
    // read, so an illegal request fails on the flag rather than deep inside the
    // resume path -- and so it fails even when the checkpoint carries no
    // search_pi block at all, where the reset call site below is never reached.
    // A request that silently did nothing is the worse outcome: the run would
    // proceed in the wrong episode domain wearing the profile's name.
    const bool origin_reset_requested =
        absl::GetFlag(FLAGS_search_pi_origin_reset);
    const std::string origin_reset_fingerprint =
        absl::GetFlag(FLAGS_search_pi_origin_config_fingerprint);
    if (origin_reset_requested && !profile_rung4a && !profile_rung4_replay) {
      SpielFatalError(
          "an origin reset is legal only under the rung4a and rung4_replay "
          "profiles; this invocation selected the profile " +
          registration_profile + ".");
    }
    if (origin_reset_requested && origin_reset_fingerprint.empty()) {
      SpielFatalError(
          "an origin reset requires the origin config fingerprint: without it "
          "the authorising predicate collapses to a generation and a cursor, "
          "which many checkpoints share.");
    }
    // A reset routes the lineage into the FRESH-lineage arm, and the launcher
    // derives the episode base and the registered base from one value, so that
    // arm's base check compares a constant against itself. The expected student
    // digest is therefore the ONLY check that ties the loaded weights to the
    // intended origin, and it is skipped when empty. Requiring it here turns a
    // fail-OPEN default into a fail-closed one: a hand-launched reset that
    // omitted it would otherwise train whatever checkpoint it was handed, under
    // the registered profile's name, with no identity check at all.
    if (origin_reset_requested &&
        absl::GetFlag(FLAGS_search_pi_expected_initial_student_sha256).empty()) {
      SpielFatalError(
          "an origin reset requires the expected initial student digest: it is "
          "the only remaining check that binds the loaded weights to the "
          "registered origin checkpoint.");
    }

    // rung4_replay: capture on one side, train-only on the other. They are two
    // halves of the same experiment and never the same invocation -- capture
    // produces the cohort a later training step consumes.
    const std::string row_shard_out =
        absl::GetFlag(FLAGS_search_pi_row_shard_out);
    const std::string train_from_shards =
        absl::GetFlag(FLAGS_search_pi_train_from_shards);
    const bool replay_train = !train_from_shards.empty();
    const int row_sample_count = absl::GetFlag(FLAGS_search_pi_row_sample_count);
    if ((!row_shard_out.empty() || replay_train) && !profile_rung4_replay) {
      SpielFatalError(
          "row-shard capture and shard training are legal only under the "
          "rung4_replay profile; this invocation selected the profile " +
          registration_profile + ".");
    }
    if (!row_shard_out.empty() && replay_train) {
      SpielFatalError(
          "shard capture and shard training are mutually exclusive: the first "
          "collects a cohort and writes it, the second never collects at all.");
    }
    if (replay_train && row_sample_count <= 0) {
      SpielFatalError(
          "training from shards requires a positive row sample count: an "
          "unstated count would let the replay arm take a different number of "
          "SGD steps than the fresh arm, which confounds replay with more "
          "optimisation.");
    }

    open_spiel::SearchPiConfig pi_cfg;
    pi_cfg.games_per_generation =
        absl::GetFlag(FLAGS_search_pi_games_per_generation);
    pi_cfg.training_games_per_generation = pi_cfg.games_per_generation;
    pi_cfg.collection_games_per_generation =
        absl::GetFlag(FLAGS_search_pi_collection_games_per_generation) > 0
            ? absl::GetFlag(FLAGS_search_pi_collection_games_per_generation)
            : pi_cfg.games_per_generation;
    pi_cfg.concurrent_workers =
        absl::GetFlag(FLAGS_search_pi_concurrent_workers);
    pi_cfg.batch_target = absl::GetFlag(FLAGS_search_pi_batch_target);
    pi_cfg.batcher_timeout_ms =
        absl::GetFlag(FLAGS_search_pi_batcher_timeout_ms);
    pi_cfg.chunk_games = absl::GetFlag(FLAGS_search_pi_chunk_games);
    pi_cfg.warmup_games = absl::GetFlag(FLAGS_search_pi_warmup_games);
    pi_cfg.max_filler_timeout_episodes =
        absl::GetFlag(FLAGS_search_pi_max_filler_timeout_episodes);
    pi_cfg.next_episode_id =
        absl::GetFlag(FLAGS_search_pi_first_episode_id);
    pi_cfg.primary_simulations =
        absl::GetFlag(FLAGS_search_pi_primary_simulations);
    pi_cfg.continuation_simulations =
        absl::GetFlag(FLAGS_search_pi_continuation_simulations);
    pi_cfg.purchase_combat_budget =
        absl::GetFlag(FLAGS_search_pi_purchase_combat_budget);
    pi_cfg.relative_time_budget_ms =
        absl::GetFlag(FLAGS_search_pi_relative_time_budget_ms);
    pi_cfg.puct_c = absl::GetFlag(FLAGS_search_pi_puct_c);
    pi_cfg.max_search_decision_depth = -1;  // uncapped, not a flag
    pi_cfg.use_opponent_model = true;
    pi_cfg.opponent_model_temperature =
        absl::GetFlag(FLAGS_search_pi_opponent_temperature);
    pi_cfg.root_prior_temperature =
        absl::GetFlag(FLAGS_search_pi_root_prior_temperature);
    pi_cfg.utility_divisor = 4.0;  // the model's own value scale
    pi_cfg.dirichlet_epsilon = absl::GetFlag(FLAGS_search_pi_dirichlet_epsilon);
    pi_cfg.dirichlet_alpha_total =
        absl::GetFlag(FLAGS_search_pi_dirichlet_alpha_total);
    pi_cfg.forced_playouts_k = absl::GetFlag(FLAGS_search_pi_forced_playouts_k);
    pi_cfg.root_noise_fpu_zero =
        absl::GetFlag(FLAGS_search_pi_root_noise_fpu_zero);
    pi_cfg.target_sharpen_exponent =
        absl::GetFlag(FLAGS_search_pi_target_sharpen_exponent);
    if (!open_spiel::ParseSearchPiContinuationTarget(
            absl::GetFlag(FLAGS_search_pi_continuation_target),
            &pi_cfg.continuation_target)) {
      SpielFatalError("Unrecognized --search_pi_continuation_target: '" +
                      absl::GetFlag(FLAGS_search_pi_continuation_target) +
                      "' (expected total_visits|new_visits_only).");
    }
    pi_cfg.behavior_temperature =
        absl::GetFlag(FLAGS_search_pi_behavior_temperature);
    pi_cfg.non_search_temperature =
        absl::GetFlag(FLAGS_search_pi_non_search_temperature);
    pi_cfg.searched_seat_unsearched_temperature =
        absl::GetFlag(FLAGS_search_pi_unsearched_role_temperature);
    pi_cfg.search_leader_draft = false;  // structural, and manifested as such
    pi_cfg.seed_domain = absl::GetFlag(FLAGS_search_pi_seed_domain);

    const std::string collector_path =
        absl::GetFlag(FLAGS_search_pi_collector_model_checkpoint);
    const std::string collector_expected_sha =
        absl::GetFlag(FLAGS_search_pi_collector_model_sha256);
    const bool production_concurrent = pi_cfg.batch_target > 0;
    std::shared_ptr<open_spiel::SharedDunePolicyValueNetImpl> collector_model;
    std::string collector_digest;
    if (production_concurrent || !collector_path.empty() ||
        !collector_expected_sha.empty()) {
      constexpr char kRegisteredCollectorSha256[] =
          "68febee771509f88446286cc50983afd22d64f7772f6866def77bafd4aae36d2";
      // The approved continuation has one exact production geometry. Enforcing
      // it in the consumer prevents a launcher typo from creating a substitute
      // experiment that merely resembles the registration.
      if (pi_cfg.collection_games_per_generation != 512 ||
          pi_cfg.training_games_per_generation != 64 ||
          pi_cfg.concurrent_workers != 128 || pi_cfg.batch_target != 64 ||
          pi_cfg.batcher_timeout_ms != 2 || pi_cfg.chunk_games != 4 ||
          pi_cfg.warmup_games != 0 ||
          pi_cfg.max_filler_timeout_episodes != 2 ||
          pi_cfg.relative_time_budget_ms != 60000.0 ||
          (pi_cfg.purchase_combat_budget != 0 &&
           pi_cfg.purchase_combat_budget != 64)) {
        SpielFatalError(
            "post-Gate-8 Search-PI geometry must be collection=512, "
            "training=64, G=128, target=64, timeout=2ms, chunks=4, "
            "warmup=0, watchdog=60000ms, filler cap=2 and scope in {0,64}");
      }
      if (absl::GetFlag(FLAGS_search_pi_generations) != 1) {
        SpielFatalError(
            "the post-Gate-8 trainer accepts exactly one generation per "
            "invocation so the durable supervisor must run evaluation and the "
            "decline rail before the next collection");
      }
      if (collector_path.empty() || collector_expected_sha.empty()) {
        SpielFatalError(
            "post-Gate-8 Search-PI requires the frozen collector archive and "
            "its registered SHA-256");
      }
      if (collector_expected_sha != kRegisteredCollectorSha256) {
        SpielFatalError(
            "post-Gate-8 collector SHA is not the registered u15828 archive: " +
            collector_expected_sha);
      }
      const std::string actual_sha =
          open_spiel::ComputeFileSHA256(collector_path);
      if (actual_sha != collector_expected_sha) {
        SpielFatalError("frozen collector archive SHA-256 mismatch: expected " +
                        collector_expected_sha + " actual " + actual_sha);
      }
      collector_model = std::make_shared<
          open_spiel::SharedDunePolicyValueNetImpl>(
          obs_size, absl::GetFlag(FLAGS_hidden_dim), action_size,
          absl::GetFlag(FLAGS_num_blocks),
          absl::GetFlag(FLAGS_nonlinear_value_head));
      collector_model->to(device);
      try {
        torch::load(collector_model, collector_path, device);
      } catch (const c10::Error& e) {
        SpielFatalError("frozen collector load failed: " +
                        std::string(e.msg()));
      }
      collector_model->to(device);
      collector_model->eval();
      collector_digest =
          open_spiel::CanonicalSearchPiModuleDigest(*collector_model);
      pi_cfg.frozen_collector_sha256 = actual_sha;
      pi_cfg.frozen_collector_digest = collector_digest;
      if (absl::GetFlag(FLAGS_search_pi_generation_manifest_dir).empty()) {
        SpielFatalError(
            "post-Gate-8 Search-PI requires --search_pi_generation_manifest_dir");
      }
    }

    // --- rung4a: the SECOND frozen model ------------------------------------
    //
    // Loaded exactly the way the collector above is loaded -- same constructor
    // arguments, same to(device) before and after, the same torch::load
    // try/catch and the same eval() -- so a hybrid run cannot differ from a
    // single-model run in how either archive reaches the device. Loaded OUTSIDE
    // the collector block so the failure is about this flag and not about which
    // branch happened to be taken.
    std::shared_ptr<open_spiel::SharedDunePolicyValueNetImpl> policy_prior_model;
    std::string policy_prior_digest;
    std::string policy_prior_sha;
    if (!policy_prior_path.empty()) {
      policy_prior_sha = open_spiel::ComputeFileSHA256(policy_prior_path);
      if (policy_prior_sha != policy_prior_expected_sha) {
        SpielFatalError("policy-prior archive SHA-256 mismatch: expected " +
                        policy_prior_expected_sha + " actual " +
                        policy_prior_sha);
      }
      policy_prior_model =
          std::make_shared<open_spiel::SharedDunePolicyValueNetImpl>(
              obs_size, absl::GetFlag(FLAGS_hidden_dim), action_size,
              absl::GetFlag(FLAGS_num_blocks),
              absl::GetFlag(FLAGS_nonlinear_value_head));
      policy_prior_model->to(device);
      try {
        torch::load(policy_prior_model, policy_prior_path, device);
      } catch (const c10::Error& e) {
        SpielFatalError("policy-prior load failed: " + std::string(e.msg()));
      }
      policy_prior_model->to(device);
      policy_prior_model->eval();
      policy_prior_digest =
          open_spiel::CanonicalSearchPiModuleDigest(*policy_prior_model);
    }

    open_spiel::SearchPiLearnerConfig pi_learn;
    pi_learn.learning_rate = absl::GetFlag(FLAGS_search_pi_learning_rate);
    pi_learn.minibatch_size = absl::GetFlag(FLAGS_search_pi_minibatch_size);
    pi_learn.epochs = absl::GetFlag(FLAGS_search_pi_epochs);
    pi_learn.value_coef = absl::GetFlag(FLAGS_search_pi_value_coef);
    pi_learn.policy_coef = absl::GetFlag(FLAGS_search_pi_policy_coef);
    // Both coefficients zero is not an ablation, it is a run that collects for
    // 40 minutes and then steps nothing. Caught here rather than discovered in
    // the telemetry afterwards.
    if (pi_learn.policy_coef == 0.0 && pi_learn.value_coef == 0.0) {
      SpielFatalError(
          "search-PI: --search_pi_policy_coef and --search_pi_value_coef are "
          "both 0, so neither objective would produce any gradient. Use "
          "--search_pi_learning_rate=0 for a deliberate zero-step control.");
    }
    pi_learn.grad_clip_norm = absl::GetFlag(FLAGS_grad_clip_norm);
    pi_learn.logit_cap = absl::GetFlag(FLAGS_logit_cap);
    // MakeOptimizer reads these two into the parameter groups, so they are part
    // of this lane's objective and must be fingerprinted with the rest of it.
    pi_learn.weight_decay = absl::GetFlag(FLAGS_weight_decay);
    pi_learn.policy_weight_decay = absl::GetFlag(FLAGS_policy_weight_decay);

    if (production_concurrent) {
      // Pin the complete shared teacher and rung-2 learner identity in the
      // consumer. A committed launcher is necessary provenance, but it is not
      // a substitute for refusing a disarming argument at runtime.
      if (pi_cfg.primary_simulations != 200 ||
          pi_cfg.continuation_simulations != 64 || pi_cfg.puct_c != 0.30 ||
          pi_cfg.continuation_target !=
              open_spiel::SearchPiContinuationTarget::kTotalVisits ||
          pi_cfg.behavior_temperature != 0.0 ||
          pi_cfg.root_prior_temperature != 1.0 ||
          pi_cfg.dirichlet_epsilon != 0.0 ||
          pi_cfg.forced_playouts_k != 0.0 ||
          pi_cfg.target_sharpen_exponent != 1.0 ||
          pi_cfg.seed_domain != 8160001 || pi_learn.logit_cap != 10.0 ||
          pi_learn.learning_rate != 1.0e-5 ||
          pi_learn.minibatch_size != 256 || pi_learn.epochs != 1 ||
          pi_learn.policy_coef != 1.0 || pi_learn.value_coef != 0.0 ||
          pi_learn.grad_clip_norm != 0.5 || pi_learn.weight_decay != 0.0 ||
          pi_learn.policy_weight_decay != 0.0) {
        SpielFatalError(
            "post-Gate-8 Search-PI teacher/learner arguments differ from the "
            "registered 200/64, puct=.30, total_visits, deterministic, "
            "seed-domain=8160001, CE-only lr=1e-5 one-epoch minibatch-256 "
            "package");
      }
      if (absl::GetFlag(FLAGS_checkpoint_interval) != 1 ||
          !absl::GetFlag(FLAGS_save_final_checkpoint)) {
        SpielFatalError(
            "post-Gate-8 Search-PI requires checkpoint_interval=1 and "
            "save_final_checkpoint=true");
      }
    }

    const std::string pi_fingerprint =
        open_spiel::SearchPiConfigFingerprint(pi_cfg, pi_learn);

    int pi_generation = 0;
    open_spiel::SearchPiState pi_state;
    // True once the inherited lineage has been re-based. Distinct from
    // `search_pi_resume_present`, which must STAY true: it is what lets the
    // dedicated optimizer inherit this lane's Adam moments below, and that
    // inheritance is the entire reason the successor reads a search_pi block at
    // all rather than starting from a block-less checkpoint.
    bool pi_origin_reset_applied = false;
    if (search_pi_resume_present) {
      // --- Origin reset, applied BEFORE both existing resume checks ---------
      //
      // Ordering is load-bearing. The fingerprint check immediately below and
      // the resume-generation bound further down must see the RE-BASED state,
      // not 3b's, so each keeps deciding on its own terms:
      //   - the fingerprint check still compares the checkpoint's recorded
      //     config against this run's, because the reset deliberately leaves
      //     `config` and `learner` alone (dune_search_pi.h). A successor that
      //     changed a fingerprinted knob is still refused here, which is
      //     correct: it would be inheriting Adam moments across a changed
      //     objective;
      //   - the generation bound sees generation 0 and therefore routes to the
      //     fresh-lineage branch, whose two checks -- the registered initial
      //     student digest and the registered episode base -- are the ones that
      //     actually bind a re-based lineage. A completed-boundary check in
      //     1..max-1 has nothing to say about a lineage that is starting over.
      if (origin_reset_requested) {
        open_spiel::SearchPiOriginResetRequest reset_request;
        reset_request.expected_origin_generation =
            absl::GetFlag(FLAGS_search_pi_origin_generation);
        reset_request.expected_origin_next_episode =
            absl::GetFlag(FLAGS_search_pi_origin_next_episode_id);
        reset_request.origin_config_fingerprint = origin_reset_fingerprint;
        // The base the LAUNCHER stated, read before the resume block below
        // overwrites pi_cfg.next_episode_id.
        //
        // This preserves the POSSIBILITY that the fresh-lineage base check
        // binds, for a caller that states the two bases independently. Note
        // that the supervisor does NOT: it derives --search_pi_first_episode_id
        // and --search_pi_registered_first_episode_id from one manifest key
        // precisely so they cannot disagree, so under the supervisor that check
        // compares a constant against itself. Do not read it as a second
        // independent guard on a supervised launch -- there, the expected
        // initial student digest (required above) is the one that binds.
        reset_request.new_first_episode_id = pi_cfg.next_episode_id;
        reset_request.new_config_fingerprint = pi_fingerprint;
        std::string reset_error;
        if (!open_spiel::ApplySearchPiOriginReset(&search_pi_resume,
                                                  reset_request, &reset_error)) {
          SpielFatalError("search-PI origin reset refused: " + reset_error);
        }
        pi_origin_reset_applied = true;
        std::cout << absl::StrFormat(
            "[search-PI] Origin reset: re-based the inherited lineage from "
            "generation=%d next_episode_id=%lld fp=%s onto profile=%s "
            "generation=0 next_episode_id=%lld fp=%s; model and optimizer "
            "moments inherited, cumulative counters and hash chains zeroed.\n",
            search_pi_resume.origin_reset_generation,
            (long long)search_pi_resume.origin_reset_next_episode_id,
            search_pi_resume.origin_reset_config_fingerprint.substr(0, 16),
            registration_profile,
            (long long)search_pi_resume.next_episode_id,
            pi_fingerprint.substr(0, 16));
      }
      const std::string prior_fp = open_spiel::SearchPiConfigFingerprint(
          search_pi_resume.config, search_pi_resume.learner);
      if (prior_fp != pi_fingerprint) {
        SpielFatalError(absl::StrFormat(
            "search_pi config fingerprint mismatch on resume: checkpoint=%s "
            "current=%s. A search-PI resume may not silently change the search "
            "budgets, puct_c, the exploration package, the target exponent or "
            "the learner's hyperparameters.",
            prior_fp, pi_fingerprint));
      }
      pi_state = search_pi_resume;
      pi_generation = search_pi_resume.generation;
      pi_cfg.next_episode_id = search_pi_resume.next_episode_id;
    }
    if (production_concurrent) {
      if (init_mode != "checkpoint") {
        SpielFatalError(
            "post-Gate-8 Search-PI must start/resume from a checkpoint");
      }
      // A re-based lineage is a FRESH lineage for every purpose this branch
      // decides. It has generation 0 and a cursor at its own registered base,
      // so the resume arm's completed-boundary rule (1..max-1) would reject it
      // for the one property that makes it correct. Routing it here does not
      // weaken anything: it substitutes two checks that bind -- the registered
      // initial student digest and the registered episode base -- for one that
      // is inapplicable, and it is reachable only via the rung4-gated flag.
      if (!search_pi_resume_present || pi_origin_reset_applied) {
        const std::string initial_student_digest =
            open_spiel::CanonicalSearchPiModuleDigest(*training_model);
        if (profile_3b) {
          if (initial_student_digest != collector_digest) {
            SpielFatalError(
                "fresh post-Gate-8 student is not the registered frozen "
                "u15828 collector snapshot");
          }
        } else if (!expected_initial_student_digest.empty() &&
                   initial_student_digest != expected_initial_student_digest) {
          // 3b's identity cannot apply to a rung-4 arm: those runs start from a
          // TRAINED student, so demanding it equal the teacher would refuse
          // every legitimate launch. The named digest replaces it -- the same
          // fail-closed shape against a different registered value.
          SpielFatalError("fresh " + registration_profile +
                          " student is not the registered initial student: "
                          "expected " + expected_initial_student_digest +
                          " actual " + initial_student_digest);
        }
        if (pi_cfg.next_episode_id != registered_first_episode_id) {
          SpielFatalError(
              profile_3b
                  ? std::string(
                        "fresh post-Gate-8 lineage must begin at episode 600000")
                  : absl::StrFormat("fresh %s lineage must begin at episode %d",
                                    registration_profile,
                                    registered_first_episode_id));
        }
      } else if (pi_generation < 1 ||
                 pi_generation >= registered_max_generation) {
        SpielFatalError(
            profile_3b
                ? std::string("post-Gate-8 resume generation must be a "
                              "completed boundary 1..4")
                : absl::StrFormat(
                      "%s resume generation must be a completed boundary 1..%d",
                      registration_profile, registered_max_generation - 1));
      }
    }

    // The DEDICATED optimizer: its own instance and its own learning rate. The
    // PPO optimizer built above is never stepped in this mode; reusing it would
    // couple two objectives' Adam moments.
    //
    // Built through MakeOptimizer rather than as a flat parameter list, so the
    // parameter-GROUP layout (policy / other / aux heads) matches everything
    // else that saves and loads an optimizer here. A one-group optimizer writes
    // a state dict the loader rejects with "different number of parameter
    // groups", which is how this was found.
    std::unique_ptr<torch::optim::AdamW> pi_optimizer_owner =
        MakeOptimizer(training_model);
    torch::optim::AdamW& pi_optimizer = *pi_optimizer_owner;
    SetOptimizerLearningRate(pi_optimizer, pi_learn.learning_rate);

    // Adam moments are inherited ONLY from a search-PI lineage. A checkpoint
    // with no search_pi block was written by some other mode, and seeding this
    // lane's optimizer from PPO's moments is precisely the cross-objective
    // contamination the dedicated-optimizer requirement exists to prevent -- so
    // that case starts fresh, loudly.
    if (init_mode == "checkpoint") {
      if (search_pi_resume_present) {
        const std::string pi_optim_path = absl::GetFlag(FLAGS_optim_checkpoint);
        try {
          // Same two-step as the PPO resume path: a pre-head archive migrates,
          // a post-head archive loads normally.
          if (!open_spiel::LoadOptimizerCheckpointMigrating(
                  training_model, pi_optimizer, pi_optim_path, device)) {
            torch::load(pi_optimizer, pi_optim_path, device);
          }
        } catch (const c10::Error& e) {
          SpielFatalError("search-PI optimizer resume failed: " +
                          std::string(e.msg()));
        }
        SetOptimizerLearningRate(pi_optimizer, pi_learn.learning_rate);
        std::cout << "[search-PI] Inherited optimizer moments from the "
                     "search-PI lineage.\n";
      } else {
        std::cout << "[search-PI] Checkpoint carries no search_pi block: "
                     "starting the dedicated optimizer FRESH rather than "
                     "inheriting another objective's Adam moments.\n";
      }
    }

    const int generations = absl::GetFlag(FLAGS_search_pi_generations);
    std::cout << absl::StrFormat(
        "[search-PI] ON | generations=%d games/gen=%d primary_sims=%d "
        "continuation_sims=%d puct_c=%.3f eps=%.3f forced_k=%.3f sharpen=%.3f "
        "behavior_temp=%.3f cont_target=%s lr=%.3g mb=%d epochs=%d "
        "value_coef=%.3f leader=false seed_domain=%llu fp=%s\n",
        generations, pi_cfg.games_per_generation, pi_cfg.primary_simulations,
        pi_cfg.continuation_simulations, pi_cfg.puct_c,
        pi_cfg.dirichlet_epsilon, pi_cfg.forced_playouts_k,
        pi_cfg.target_sharpen_exponent, pi_cfg.behavior_temperature,
        open_spiel::SearchPiContinuationTargetName(pi_cfg.continuation_target),
        pi_learn.learning_rate, pi_learn.minibatch_size, pi_learn.epochs,
        pi_learn.value_coef,
        (unsigned long long)pi_cfg.seed_domain, pi_fingerprint.substr(0, 16));
    // Printed only OFF the default profile, so the live 3b run's stdout stays
    // byte-identical to what its supervisor already parses.
    if (!profile_3b) {
      std::cout << absl::StrFormat(
          "[search-PI] profile=%s episode_base=%d max_gen=%d policy_prior=%s "
          "shard_out=%s train_from_shards=%s sample=%d seed=%llu\n",
          registration_profile, registered_first_episode_id,
          registered_max_generation,
          policy_prior_path.empty() ? std::string("none") : policy_prior_path,
          row_shard_out.empty() ? std::string("none") : row_shard_out,
          train_from_shards.empty() ? std::string("none") : train_from_shards,
          row_sample_count,
          (unsigned long long)absl::GetFlag(FLAGS_search_pi_row_sample_seed));
    }

    const std::string pi_diag_path = absl::GetFlag(FLAGS_diagnostics_path);

    for (int gi = 0; gi < generations; ++gi) {
      ++pi_generation;

      std::vector<open_spiel::SearchPiRow> pi_rows;
      open_spiel::SearchPiGenerationStats pi_stats;
      open_spiel::SearchPiTrainingCollectionValidation collection_validation;
      std::filesystem::path generation_manifest_dir;
      std::string student_digest_before_learn;
      std::string student_digest_after_learn;
      std::string policy_prior_digest_before;
      std::string policy_prior_digest_after;
      // Carried out of the collection branch so the marker, which is written
      // after `collected` has gone out of scope, can state which controller
      // produced the cohort without recomputing anything.
      bool collected_hybrid_policy_prior = false;
      std::string collected_frozen_model_digest;
      std::string collected_policy_prior_model_digest;
      int replay_shards_read = 0;
      int64_t replay_window_rows = 0;

      if (replay_train) {
        // --- rung4_replay: a training step with no teacher in the process ----
        //
        // The cohort was collected AND validated by an earlier invocation, so
        // everything collection owns is skipped rather than run with empty
        // arguments: no CollectSearchPiConcurrent, no 512->64 validator, no
        // episode-cursor advance and no collection directory. What is NOT
        // skipped is the learner and every manipulation check on it -- this arm
        // differs from the fresh arm only in where its rows came from, and a
        // weaker check here would make the two arms incomparable.
        const std::vector<std::string> shard_paths =
            absl::StrSplit(train_from_shards, ',');
        std::vector<std::vector<open_spiel::SearchPiRow>> window;
        window.reserve(shard_paths.size());
        for (const std::string& shard_path : shard_paths) {
          std::vector<open_spiel::SearchPiRow> cohort;
          std::string shard_error;
          if (!open_spiel::ReadSearchPiRowShard(shard_path, &cohort,
                                                &shard_error)) {
            // ReadSearchPiRowShard reports damage instead of aborting, and
            // leaves `cohort` EMPTY on failure. Continuing past it would train
            // a short window and call it a replay result.
            SpielFatalError("search-PI replay shard read failed for " +
                            shard_path + ": " + shard_error);
          }
          replay_window_rows += static_cast<int64_t>(cohort.size());
          window.push_back(std::move(cohort));
        }
        replay_shards_read = static_cast<int>(shard_paths.size());
        // The sampler's documented precondition, checked HERE so the failure
        // names the window: before the 8-cohort window has filled, the caller
        // -- this -- must clamp the request to what the window actually holds.
        if (static_cast<int64_t>(row_sample_count) > replay_window_rows) {
          SpielFatalError(absl::StrFormat(
              "replay sample of %d rows exceeds the %d rows held by %d shards",
              row_sample_count, replay_window_rows, replay_shards_read));
        }
        pi_rows = open_spiel::SampleUniformReplayWindow(
            window, static_cast<size_t>(row_sample_count),
            absl::GetFlag(FLAGS_search_pi_row_sample_seed));

        // Stand in for the collection validator's OUTPUTS -- the per-role
        // tallies and the trained hashes -- so the accounting, the generation
        // print and the state chains below need no replay branch of their own.
        // Every one of those fields means "what was trained on", which is
        // exactly what these values are; none of them claims a collection ran.
        for (const open_spiel::SearchPiRow& row : pi_rows) {
          const int role = static_cast<int>(row.role);
          if (role < 0 || role >= 7) {
            SpielFatalError(absl::StrFormat(
                "replay row carries an out-of-range decision role %d", role));
          }
          collection_validation.trained_rows_by_role[role]++;
          collection_validation.trained_target_hash =
              open_spiel::ChainSearchPiTargetHash(
                  collection_validation.trained_target_hash, row);
          collection_validation.trained_extended_hash =
              open_spiel::ChainSearchPiExtendedRowHash(
                  collection_validation.trained_extended_hash, row);
        }
        collection_validation.ok = true;
        pi_stats.generation = pi_generation;
        pi_stats.rows_total = static_cast<int64_t>(pi_rows.size());
        pi_stats.target_hash_chain = collection_validation.trained_target_hash;
        pi_stats.extended_hash_chain =
            collection_validation.trained_extended_hash;
        // The cursor does NOT advance: this invocation consumed no episodes,
        // and an advance here would silently skip a 512-game block for whatever
        // collecting invocation resumes this lineage next.
        pi_stats.first_episode_id = pi_cfg.next_episode_id;
        pi_stats.next_episode_id = pi_cfg.next_episode_id;

        // The marker still needs a home. Same fresh-directory rule as the
        // collecting path: a generation never writes into another's directory.
        generation_manifest_dir =
            std::filesystem::path(
                absl::GetFlag(FLAGS_search_pi_generation_manifest_dir)) /
            absl::StrCat("generation_", pi_generation);
        if (std::filesystem::exists(generation_manifest_dir)) {
          SpielFatalError("generation output already exists: " +
                          generation_manifest_dir.string());
        }
        std::filesystem::create_directories(generation_manifest_dir);
      } else if (production_concurrent) {
        // The base and the stride are the registered lineage's, not literals:
        // the stride IS the collection size, so a profile that collects a
        // different number of games per generation still lands on contiguous,
        // non-overlapping episode blocks. Under post_gate8_3b the geometry pin
        // above forces 512, so this is exactly 600000 + (N-1) * 512.
        const int64_t registered_first =
            registered_first_episode_id +
            static_cast<int64_t>(pi_generation - 1) *
                pi_cfg.collection_games_per_generation;
        if (pi_cfg.next_episode_id != registered_first) {
          SpielFatalError(absl::StrFormat(
              "generation %d cursor is %d, registered %d",
              pi_generation, pi_cfg.next_episode_id, registered_first));
        }
        const std::string digest_before_collection =
            open_spiel::CanonicalSearchPiModuleDigest(*collector_model);
        if (digest_before_collection != collector_digest) {
          SpielFatalError("frozen collector digest changed before collection");
        }
        // A hybrid run proves BOTH models at every boundary. Proving only the
        // collector would leave the prior free to drift, and a drifted prior is
        // a silently different teacher producing targets that are attributed to
        // the registered one.
        if (policy_prior_model != nullptr) {
          policy_prior_digest_before =
              open_spiel::CanonicalSearchPiModuleDigest(*policy_prior_model);
          if (policy_prior_digest_before != policy_prior_digest) {
            SpielFatalError("policy-prior digest changed before collection");
          }
        }
        generation_manifest_dir =
            std::filesystem::path(absl::GetFlag(
                FLAGS_search_pi_generation_manifest_dir)) /
            absl::StrCat("generation_", pi_generation);
        if (std::filesystem::exists(generation_manifest_dir)) {
          SpielFatalError("generation output already exists: " +
                          generation_manifest_dir.string());
        }
        const std::filesystem::path collection_dir =
            generation_manifest_dir / "collection";

        open_spiel::ConcurrentSearchPiCollectionConfig collect_cfg;
        collect_cfg.search = pi_cfg;
        collect_cfg.search.next_episode_id = registered_first;
        collect_cfg.generation = pi_generation;
        collect_cfg.collection_games =
            pi_cfg.collection_games_per_generation;
        collect_cfg.chunk_games = pi_cfg.chunk_games;
        collect_cfg.requested_workers = pi_cfg.concurrent_workers;
        collect_cfg.batch_target = pi_cfg.batch_target;
        collect_cfg.batcher_timeout_ms = pi_cfg.batcher_timeout_ms;
        collect_cfg.warmup_games = pi_cfg.warmup_games;
        collect_cfg.retain_rows = true;
        collect_cfg.logit_cap = static_cast<float>(pi_learn.logit_cap);
        collect_cfg.output_dir = collection_dir.string();
        collect_cfg.arm = open_spiel::SearchPiArm::kSearched;
        // The trailing argument is the rung4a hybrid: a null shared_ptr is the
        // single-model collection every other profile runs, so the call site is
        // one call rather than two branches that could drift apart.
        open_spiel::ConcurrentSearchPiCollectionResult collected =
            open_spiel::CollectSearchPiConcurrent(
                collect_cfg, game, collector_model, device,
                policy_prior_model);

        // The prefix split is the first operation performed on rows inside the
        // validator and reads only episode IDs. Outcomes cannot influence dose.
        open_spiel::SearchPiTrainingCollectionContract contract;
        contract.first_episode_id = registered_first;
        contract.collection_games = pi_cfg.collection_games_per_generation;
        contract.training_games = pi_cfg.training_games_per_generation;
        contract.requested_workers = pi_cfg.concurrent_workers;
        contract.batch_target = pi_cfg.batch_target;
        contract.batcher_timeout_ms = pi_cfg.batcher_timeout_ms;
        contract.primary_simulations = pi_cfg.primary_simulations;
        contract.continuation_simulations = pi_cfg.continuation_simulations;
        contract.purchase_combat_budget = pi_cfg.purchase_combat_budget;
        contract.max_filler_timeout_episodes =
            pi_cfg.max_filler_timeout_episodes;
        // Which collection geometry this generation registered. The validator
        // compares it against the RESULT in both directions, so a hybrid run
        // cannot be waved through by contract checks that only know one
        // batcher, and an incumbent result cannot satisfy a hybrid contract
        // (dune_search_pi_concurrent.h:125-130).
        contract.expect_hybrid_policy_prior = (policy_prior_model != nullptr);
        collection_validation =
            open_spiel::ValidateSearchPiTrainingCollection(collected, contract);
        pi_rows = collection_validation.training_rows;

        pi_stats.generation = pi_generation;
        pi_stats.games = pi_cfg.collection_games_per_generation;
        pi_stats.first_episode_id = registered_first;
        pi_stats.next_episode_id = registered_first +
                                   pi_cfg.collection_games_per_generation;
        pi_stats.collection_wall_time_s = collected.wall_time_s;
        pi_stats.inference_calls = collected.inference_calls;
        pi_stats.rows_total = collected.rows_total;
        pi_stats.primary = collected.primary;
        pi_stats.continuation = collected.continuation;
        pi_stats.purchase = collected.purchase;
        pi_stats.combat_intrigue = collected.combat_intrigue;
        pi_stats.other_optional = collected.other_optional;
        for (int role = 0; role < 7; ++role) {
          pi_stats.simulations_by_role[role] =
              collected.simulations_by_role[role];
          pi_stats.decisions_by_role[role] = collected.decisions_by_role[role];
        }
        pi_stats.leader_rows_emitted = collected.leader_rows_emitted;
        pi_stats.arm = collected.arm;
        pi_stats.games_played = collected.games;
        pi_stats.target_hash_chain = collected.target_hash_chain;
        pi_stats.extended_hash_chain = collected.extended_hash_chain;
        pi_cfg.next_episode_id = pi_stats.next_episode_id;

        const std::string digest_after_collection =
            open_spiel::CanonicalSearchPiModuleDigest(*collector_model);
        if (digest_after_collection != collector_digest) {
          collection_validation.errors.push_back(
              "frozen collector digest changed during collection");
          collection_validation.ok = false;
        }
        // Reported through the validation record rather than fataling on the
        // spot, exactly as the collector's is: the manifest below is the
        // durable evidence of WHY the generation refused to learn.
        if (policy_prior_model != nullptr) {
          policy_prior_digest_after =
              open_spiel::CanonicalSearchPiModuleDigest(*policy_prior_model);
          if (policy_prior_digest_after != policy_prior_digest) {
            collection_validation.errors.push_back(
                "policy-prior digest changed during collection");
            collection_validation.ok = false;
          }
          collected_hybrid_policy_prior = collected.hybrid_policy_prior;
          collected_frozen_model_digest = collected.frozen_model_digest;
          collected_policy_prior_model_digest =
              collected.policy_prior_model_digest;
          // The RESULT declares the mode, not the caller. A prior that was
          // handed over and then ignored would produce an INCUMBENT cohort
          // recorded as a hybrid one -- the exact misattribution rung 4a is
          // asking a question about.
          if (!collected_hybrid_policy_prior) {
            collection_validation.errors.push_back(
                "policy-prior model was supplied but the collection reports "
                "the incumbent single-model geometry");
            collection_validation.ok = false;
          }
          // Both models the collection actually ran, as the collector itself
          // hashed them, agreeing with the two identities this process holds.
          if (collected_frozen_model_digest != collector_digest ||
              collected_policy_prior_model_digest != policy_prior_digest) {
            collection_validation.errors.push_back(
                "collection-reported model digests disagree with the loaded "
                "frozen/policy-prior identities");
            collection_validation.ok = false;
          }
        }

        // Durable collection transaction, written before any learner call.
        open_spiel::json::Object manifest;
        manifest["schema_version"] = static_cast<int64_t>(1);
        manifest["generation"] = static_cast<int64_t>(pi_generation);
        manifest["validation_pass"] = collection_validation.ok;
        manifest["first_episode_id"] = registered_first;
        manifest["next_episode_id"] = pi_stats.next_episode_id;
        manifest["collection_games"] =
            static_cast<int64_t>(pi_cfg.collection_games_per_generation);
        manifest["training_games"] =
            static_cast<int64_t>(pi_cfg.training_games_per_generation);
        manifest["requested_workers"] =
            static_cast<int64_t>(collected.requested_workers);
        manifest["actual_workers"] =
            static_cast<int64_t>(collected.actual_workers);
        manifest["batch_target"] =
            static_cast<int64_t>(collected.configured_batch_target);
        manifest["batcher_timeout_ms"] =
            static_cast<int64_t>(collected.configured_batcher_timeout_ms);
        manifest["max_inflight_rows"] =
            static_cast<int64_t>(collected.max_inflight_rows);
        manifest["purchase_combat_budget"] =
            static_cast<int64_t>(pi_cfg.purchase_combat_budget);
        manifest["collector_archive_sha256"] =
            pi_cfg.frozen_collector_sha256;
        manifest["collector_digest_before"] = digest_before_collection;
        manifest["collector_digest_after"] = digest_after_collection;
        // Written only when a prior was loaded, so a post_gate8_3b manifest is
        // byte-identical to the ones its supervisor already reads. Every
        // existing key keeps its name and its meaning.
        if (policy_prior_model != nullptr) {
          manifest["policy_prior_archive_sha256"] = policy_prior_sha;
          manifest["policy_prior_digest_before"] = policy_prior_digest_before;
          manifest["policy_prior_digest_after"] = policy_prior_digest_after;
          // The collector's own account of what it ran, recorded beside this
          // process's boundary hashes rather than in place of them: the pair
          // asserts the two models were the same before, during and after.
          manifest["hybrid_policy_prior"] = collected_hybrid_policy_prior;
          manifest["frozen_model_digest"] = collected_frozen_model_digest;
          manifest["policy_prior_model_digest"] =
              collected_policy_prior_model_digest;
        }
        manifest["collected_target_hash"] =
            collection_validation.collected_target_hash;
        manifest["collected_extended_hash"] =
            collection_validation.collected_extended_hash;
        manifest["trained_target_hash"] =
            collection_validation.trained_target_hash;
        manifest["trained_extended_hash"] =
            collection_validation.trained_extended_hash;
        manifest["collected_rows"] =
            static_cast<int64_t>(collected.retained_rows.size());
        manifest["trained_rows"] = static_cast<int64_t>(pi_rows.size());
        open_spiel::json::Array filler;
        for (int64_t episode :
             collection_validation.filler_timeout_episode_ids) {
          filler.push_back(episode);
        }
        manifest["filler_timeout_episode_ids"] = std::move(filler);
        open_spiel::json::Array errors;
        for (const std::string& error : collection_validation.errors) {
          errors.push_back(error);
        }
        manifest["errors"] = std::move(errors);
        open_spiel::json::Object trained_by_role;
        for (int role = 0; role < 7; ++role) {
          trained_by_role[absl::StrCat(role)] =
              collection_validation.trained_rows_by_role[role];
        }
        manifest["trained_rows_by_role"] = std::move(trained_by_role);
        std::filesystem::create_directories(generation_manifest_dir);
        const std::filesystem::path manifest_path =
            generation_manifest_dir / "collection_manifest.json";
        const std::filesystem::path manifest_tmp =
            generation_manifest_dir / ".collection_manifest.json.tmp";
        {
          std::ofstream out(manifest_tmp, std::ios::trunc);
          out << open_spiel::json::ToString(manifest, true) << "\n";
          out.flush();
          if (!out) SpielFatalError("collection manifest write failed");
        }
        std::filesystem::rename(manifest_tmp, manifest_path);
        if (!collection_validation.ok) {
          std::string joined;
          for (const std::string& error : collection_validation.errors) {
            joined += "\n  - " + error;
          }
          SpielFatalError("collection validation failed before learner:" +
                          joined);
        }
        // rung4_replay capture. Written AFTER validation passes -- an invalid
        // cohort must never enter a replay window -- and BEFORE the learner
        // runs, so a 512-game collection survives a learner failure instead of
        // having to be re-collected. The rows are exactly the outcome-blind
        // 64-game training prefix the fresh arm trains on, so the two arms draw
        // from the same population by construction.
        if (!row_shard_out.empty()) {
          std::string shard_error;
          if (!open_spiel::WriteSearchPiRowShard(row_shard_out, pi_rows,
                                                 &shard_error)) {
            SpielFatalError("search-PI row shard write failed for " +
                            row_shard_out + ": " + shard_error);
          }
          std::cout << "[search-PI] wrote cohort shard " << row_shard_out
                    << " (" << pi_rows.size() << " rows)\n";
        }
      } else {
        // Historical reference path retained for old checkpoints and parity.
        auto pi_evaluator = std::make_shared<open_spiel::DuneNNEvaluator>(
            inference_model, device,
            static_cast<float>(absl::GetFlag(FLAGS_logit_cap)));
        open_spiel::SearchPiGenerator(pi_cfg).GenerateGeneration(
            pi_generation, game, pi_evaluator, &pi_rows, &pi_stats);
        pi_cfg.next_episode_id = pi_stats.next_episode_id;
      }

      // 3. LEARN: the dedicated AlphaZero-style objective, current-generation
      //    data only (no replay buffer in this slice).
      if (production_concurrent) {
        student_digest_before_learn =
            open_spiel::CanonicalSearchPiModuleDigest(*training_model);
      }
      open_spiel::SearchPiLearnerStats pi_lstats =
          open_spiel::RunSearchPiLearner(training_model, pi_optimizer, pi_rows,
                                         obs_size, action_size, device, master,
                                         pi_generation, pi_learn);
      if (!std::isfinite(pi_lstats.policy_grad_norm) ||
          pi_lstats.policy_grad_norm <= 0.0) {
        SpielFatalError("Search-PI policy gradient is zero or non-finite");
      }
      if (production_concurrent) {
        if (!pi_lstats.policy_backward_executed ||
            pi_lstats.value_backward_executed ||
            pi_lstats.value_grad_norm != 0.0) {
          SpielFatalError(
              "registered CE-only learner manipulation check failed");
        }
        student_digest_after_learn =
            open_spiel::CanonicalSearchPiModuleDigest(*training_model);
        if (student_digest_after_learn == student_digest_before_learn) {
          SpielFatalError(
              "learner reported a policy gradient but the student module "
              "digest did not advance");
        }
        const std::string digest_after_learn =
            open_spiel::CanonicalSearchPiModuleDigest(*collector_model);
        if (digest_after_learn != collector_digest) {
          SpielFatalError("frozen collector digest changed during learning");
        }
        if (policy_prior_model != nullptr &&
            open_spiel::CanonicalSearchPiModuleDigest(*policy_prior_model) !=
                policy_prior_digest) {
          SpielFatalError("policy-prior digest changed during learning");
        }
      } else {
        // Historical evolving-teacher behavior only. The approved production
        // lane never syncs the student into its collector.
        open_spiel::SyncModels(training_model, inference_model, &sync_mutex);
      }

      pi_state.generation = pi_generation;
      pi_state.next_episode_id = pi_stats.next_episode_id;
      const int64_t trained_rows = production_concurrent
          ? static_cast<int64_t>(pi_rows.size()) : pi_stats.rows_total;
      pi_state.cum_rows += trained_rows;
      pi_state.cum_primary_rows += production_concurrent
          ? collection_validation.trained_rows_by_role[2]
          : pi_stats.primary.rows_emitted;
      pi_state.cum_continuation_rows += production_concurrent
          ? collection_validation.trained_rows_by_role[3]
          : pi_stats.continuation.rows_emitted;
      pi_state.cum_purchase_rows += production_concurrent
          ? collection_validation.trained_rows_by_role[4]
          : pi_stats.purchase.rows_emitted;
      pi_state.cum_combat_intrigue_rows += production_concurrent
          ? collection_validation.trained_rows_by_role[5]
          : pi_stats.combat_intrigue.rows_emitted;
      pi_state.cum_other_optional_rows += production_concurrent
          ? collection_validation.trained_rows_by_role[6]
          : pi_stats.other_optional.rows_emitted;
      pi_state.cum_filler_timeout_episodes += production_concurrent
          ? collection_validation.filler_timeout_episode_ids.size() : 0;
      pi_state.cum_primary_simulations += pi_stats.primary.simulations_completed;
      pi_state.cum_continuation_simulations +=
          pi_stats.continuation.simulations_completed;
      const std::string generation_target_hash = production_concurrent
          ? collection_validation.trained_target_hash
          : pi_stats.target_hash_chain;
      const std::string generation_extended_hash = production_concurrent
          ? collection_validation.trained_extended_hash
          : pi_stats.extended_hash_chain;
      if (generation_target_hash.empty() || generation_extended_hash.empty()) {
        SpielFatalError("Search-PI generation has an empty target/row hash");
      }
      pi_state.target_hash_chain = open_spiel::ComputeStringSHA256(
          pi_state.target_hash_chain + generation_target_hash);
      pi_state.extended_hash_chain = open_spiel::ComputeStringSHA256(
          pi_state.extended_hash_chain + generation_extended_hash);
      if (production_concurrent) {
        // The COLLECTED chain advances only when something was collected. A
        // replay step contributes the empty string, and chaining that in would
        // move a collection-provenance chain forward on a generation that
        // collected nothing -- an unfalsifiable record. The TRAINED chain does
        // advance: it is the record of what this step learned from.
        if (!replay_train) {
          pi_state.collected_target_hash_chain =
              open_spiel::ComputeStringSHA256(
                  pi_state.collected_target_hash_chain +
                  collection_validation.collected_target_hash);
          pi_state.collected_extended_hash_chain =
              open_spiel::ComputeStringSHA256(
                  pi_state.collected_extended_hash_chain +
                  collection_validation.collected_extended_hash);
        }
        pi_state.trained_target_hash_chain =
            open_spiel::ComputeStringSHA256(
                pi_state.trained_target_hash_chain +
                collection_validation.trained_target_hash);
        pi_state.trained_extended_hash_chain =
            open_spiel::ComputeStringSHA256(
                pi_state.trained_extended_hash_chain +
                collection_validation.trained_extended_hash);
      }
      pi_state.config = pi_cfg;
      pi_state.learner = pi_learn;

      std::cout << absl::StrFormat(
          "[search-PI] gen %d | trained_rows=%lld collected_rows=%lld "
          "(primary %lld / cont %lld / purchase %lld / combat %lld / other %lld) "
          "sims=%lld/%lld fallbacks=%lld/%lld re_root hit/miss=%lld/%lld "
          "kept_vs_legacy_gate=%lld/%lld | CE=%.5f MSE=%.5f "
          "|g_pol|=%.4f |g_val|=%.4f cos=%.4f (pol %.4f trunk %.4f val %.4f) "
          "| collect=%.1fs\n",
          pi_generation, (long long)trained_rows,
          (long long)pi_stats.rows_total,
          (long long)(production_concurrent
              ? collection_validation.trained_rows_by_role[2]
              : pi_stats.primary.rows_emitted),
          (long long)(production_concurrent
              ? collection_validation.trained_rows_by_role[3]
              : pi_stats.continuation.rows_emitted),
          (long long)(production_concurrent
              ? collection_validation.trained_rows_by_role[4]
              : pi_stats.purchase.rows_emitted),
          (long long)(production_concurrent
              ? collection_validation.trained_rows_by_role[5]
              : pi_stats.combat_intrigue.rows_emitted),
          (long long)(production_concurrent
              ? collection_validation.trained_rows_by_role[6]
              : pi_stats.other_optional.rows_emitted),
          (long long)pi_stats.primary.simulations_completed,
          (long long)pi_stats.continuation.simulations_completed,
          (long long)pi_stats.primary.fallbacks,
          (long long)pi_stats.continuation.fallbacks,
          (long long)pi_stats.continuation.re_root_hits,
          (long long)pi_stats.continuation.re_root_misses,
          (long long)pi_stats.primary.kept_despite_legacy_gate,
          (long long)pi_stats.continuation.kept_despite_legacy_gate,
          pi_lstats.policy_ce, pi_lstats.value_mse, pi_lstats.policy_grad_norm,
          pi_lstats.value_grad_norm, pi_lstats.grad_cosine_overall,
          pi_lstats.grad_cosine_policy_head, pi_lstats.grad_cosine_trunk,
          pi_lstats.grad_cosine_value_head, pi_stats.collection_wall_time_s);

      // Zero-simulation invariant, checked every generation rather than only in
      // the unit tests: a routing regression must stop the run, not quietly
      // start teaching Leader or purchase rows.
      std::vector<int> off_scope = {0, 1};
      if (pi_cfg.purchase_combat_budget == 0) {
        off_scope.insert(off_scope.end(), {4, 5, 6});
      }
      for (int r : off_scope) {
        if (pi_stats.simulations_by_role[r] != 0) {
          SpielFatalError(absl::StrFormat(
              "search-PI scope violation: role %d ran %lld simulations; only "
              "this role is outside the selected teacher scope.",
              r, (long long)pi_stats.simulations_by_role[r]));
        }
        // A replay step runs no simulations at all, so the check above is
        // vacuous for it. The equivalent statement about a sampled cohort is
        // that no off-scope ROW reached the learner -- the validator's own rule
        // (dune_search_pi_concurrent.cc, "out-of-scope role N reached the
        // learner subset"), applied here because a shard list is caller-supplied
        // and could name a cohort collected under a different scope.
        if (replay_train && collection_validation.trained_rows_by_role[r] != 0) {
          SpielFatalError(absl::StrFormat(
              "search-PI replay scope violation: %lld sampled rows carry the "
              "off-scope role %d.",
              (long long)collection_validation.trained_rows_by_role[r], r));
        }
      }
      if (pi_stats.leader_rows_emitted != 0) {
        SpielFatalError("search-PI emitted a Leader row; the lane's scope is "
                        "agent-turn decisions only.");
      }

      open_spiel::WriteSearchPiTelemetry(pi_diag_path, pi_cfg, pi_learn,
                                         pi_stats, pi_lstats);

      const int ckpt_interval = absl::GetFlag(FLAGS_checkpoint_interval);
      const bool last = (gi == generations - 1);
      if ((ckpt_interval > 0 && pi_generation % ckpt_interval == 0) ||
          (last && absl::GetFlag(FLAGS_save_final_checkpoint))) {
        // global_update carries the generation counter (a monotone count of
        // optimizer-stepping passes, which is what it means here), but
        // target_end_update is passed through from the FLAG unchanged. Writing
        // `generations` into it instead would make the manifest's own PPO
        // resume validator reject the next generation, because that validator
        // compares the flag against the stored field. The lane's real
        // generation accounting lives in the "search_pi" block, not here.
        const std::string flat_model = absl::GetFlag(FLAGS_model_checkpoint);
        const std::string flat_optim = absl::GetFlag(FLAGS_optim_checkpoint);

        // The resume path, written exactly as before. The manifest's own
        // validator compares model_filename against this file, and the resume
        // reader derives the manifest from --model_checkpoint, so this write
        // must keep its original name for the lane to remain resumable.
        open_spiel::SaveCheckpoint(
            training_model, pi_optimizer, flat_model, flat_optim, pi_generation,
            absl::GetFlag(FLAGS_target_end_update), total_env_steps.load(),
            next_episode_id.load(), master,
            absl::GetFlag(FLAGS_seed_scheme_version), config_fingerprint,
            search_label_fingerprint, run_uuid,
            /*aux_state=*/nullptr, &pi_state);

        // The archival copy, at a path no other generation can claim. The pilot
        // checkpointed every generation to the SAME path, so each overwrote the
        // last and generations 1-4 no longer exist -- a ladder destroyed by its
        // own save (SEARCH_PI_LANE doc section 6, and the same failure as
        // `retention-ate-the-probe-ladder`). A second write is cheap; losing the
        // rung is not recoverable except by re-running the whole arm.
        //
        // Written second on purpose: if the process dies mid-save, the file
        // that resume depends on is already complete.
        std::filesystem::path gen_model = flat_model;
        std::filesystem::path gen_optim = flat_optim;
        const std::string suffix = absl::StrCat("_gen", pi_generation);
        gen_model.replace_filename(gen_model.stem().string() + suffix +
                                   gen_model.extension().string());
        gen_optim.replace_filename(gen_optim.stem().string() + suffix +
                                   gen_optim.extension().string());
        open_spiel::SaveCheckpoint(
            training_model, pi_optimizer, gen_model.string(),
            gen_optim.string(), pi_generation,
            absl::GetFlag(FLAGS_target_end_update), total_env_steps.load(),
            next_episode_id.load(), master,
            absl::GetFlag(FLAGS_seed_scheme_version), config_fingerprint,
            search_label_fingerprint, run_uuid,
            /*aux_state=*/nullptr, &pi_state);
        if (production_concurrent) {
          const std::string collector_digest_at_checkpoint =
              open_spiel::CanonicalSearchPiModuleDigest(*collector_model);
          if (collector_digest_at_checkpoint != collector_digest) {
            SpielFatalError(
                "frozen collector digest changed at checkpoint boundary");
          }
          // The fourth and last boundary for the hybrid's second model. With
          // the three above it, a rung4a generation proves both frozen
          // identities before collection, after collection, after learning and
          // at the checkpoint that makes the generation durable.
          if (policy_prior_model != nullptr &&
              open_spiel::CanonicalSearchPiModuleDigest(*policy_prior_model) !=
                  policy_prior_digest) {
            SpielFatalError(
                "policy-prior digest changed at checkpoint boundary");
          }

          auto file_record = [](const std::filesystem::path& path) {
            size_t size = 0;
            const std::string sha =
                open_spiel::ComputeFileSHA256(path.string(), &size);
            open_spiel::json::Object record;
            record["path"] = path.string();
            record["size"] = static_cast<int64_t>(size);
            record["sha256"] = sha;
            return record;
          };
          std::filesystem::path gen_checkpoint_manifest = gen_model;
          gen_checkpoint_manifest.replace_extension(".json");
          const std::filesystem::path collection_manifest =
              generation_manifest_dir / "collection_manifest.json";

          // This marker is written last. Its presence means collection was
          // validated, exactly one learner call completed, and the immutable
          // model/optimizer/state triple is hash-bound. The supervisor resumes
          // only from this boundary and never from a PID or a flat checkpoint.
          open_spiel::json::Object complete;
          complete["schema_version"] = static_cast<int64_t>(1);
          complete["phase"] = "generation_checkpoint_complete";
          complete["generation"] = static_cast<int64_t>(pi_generation);
          complete["learner_calls"] = static_cast<int64_t>(1);
          complete["config_fingerprint"] = pi_fingerprint;
          complete["first_episode_id"] = pi_stats.first_episode_id;
          complete["next_episode_id"] = pi_stats.next_episode_id;
          complete["collection_games"] = static_cast<int64_t>(
              pi_cfg.collection_games_per_generation);
          complete["training_games"] = static_cast<int64_t>(
              pi_cfg.training_games_per_generation);
          complete["purchase_combat_budget"] =
              static_cast<int64_t>(pi_cfg.purchase_combat_budget);
          complete["collector_archive_sha256"] =
              pi_cfg.frozen_collector_sha256;
          complete["collector_digest"] = collector_digest_at_checkpoint;
          // New keys only, and only when the hybrid is armed: the 3b marker the
          // supervisor already checks keeps exactly its current key set.
          if (policy_prior_model != nullptr) {
            complete["policy_prior_archive_sha256"] = policy_prior_sha;
            complete["policy_prior_digest"] = policy_prior_digest;
            complete["hybrid_policy_prior"] = collected_hybrid_policy_prior;
            complete["frozen_model_digest"] = collected_frozen_model_digest;
            complete["policy_prior_model_digest"] =
                collected_policy_prior_model_digest;
          }
          complete["student_digest_before_learn"] =
              student_digest_before_learn;
          complete["student_digest_after_learn"] = student_digest_after_learn;
          complete["policy_backward_executed"] =
              pi_lstats.policy_backward_executed;
          complete["value_backward_executed"] =
              pi_lstats.value_backward_executed;
          complete["policy_grad_norm"] = pi_lstats.policy_grad_norm;
          complete["value_grad_norm"] = pi_lstats.value_grad_norm;
          if (!replay_train) {
            complete["collected_target_hash"] =
                collection_validation.collected_target_hash;
            complete["collected_extended_hash"] =
                collection_validation.collected_extended_hash;
          }
          complete["trained_target_hash"] =
              collection_validation.trained_target_hash;
          complete["trained_extended_hash"] =
              collection_validation.trained_extended_hash;
          complete["cumulative_collected_target_hash_chain"] =
              pi_state.collected_target_hash_chain;
          complete["cumulative_collected_extended_hash_chain"] =
              pi_state.collected_extended_hash_chain;
          complete["cumulative_trained_target_hash_chain"] =
              pi_state.trained_target_hash_chain;
          complete["cumulative_trained_extended_hash_chain"] =
              pi_state.trained_extended_hash_chain;
          if (!replay_train) {
            complete["collection_manifest"] =
                open_spiel::json::Value(file_record(collection_manifest));
          } else {
            // What a replay step IS, stated in the marker rather than left to
            // be inferred from an absent collection manifest. The seed goes out
            // as a decimal STRING because json.cc has no unsigned 64-bit
            // number, and a wrapped-negative seed would not reproduce the draw.
            complete["replay_arm_training_step"] = true;
            complete["replay_shards_read"] =
                static_cast<int64_t>(replay_shards_read);
            complete["replay_window_rows"] = replay_window_rows;
            complete["replay_sample_count"] =
                static_cast<int64_t>(pi_rows.size());
            complete["replay_sample_seed"] = absl::StrCat(
                absl::GetFlag(FLAGS_search_pi_row_sample_seed));
          }
          complete["model_checkpoint"] =
              open_spiel::json::Value(file_record(gen_model));
          complete["optimizer_checkpoint"] =
              open_spiel::json::Value(file_record(gen_optim));
          complete["checkpoint_manifest"] = open_spiel::json::Value(
              file_record(gen_checkpoint_manifest));

          const std::filesystem::path complete_path =
              generation_manifest_dir / "generation_complete.json";
          const std::filesystem::path complete_tmp =
              generation_manifest_dir / ".generation_complete.json.tmp";
          {
            std::ofstream out(complete_tmp, std::ios::trunc);
            out << open_spiel::json::ToString(complete, true) << "\n";
            out.flush();
            if (!out) {
              SpielFatalError("generation-complete manifest write failed");
            }
          }
          std::filesystem::rename(complete_tmp, complete_path);
        }
        std::cout << "[search-PI] archived generation " << pi_generation
                  << " to " << gen_model.string() << "\n";
      }
    }

    std::cout << absl::StrFormat(
        "[search-PI] DONE | generations=%d cum_rows=%lld next_episode_id=%lld "
        "chain=%s ext_chain=%s\n",
        pi_generation, (long long)pi_state.cum_rows,
        (long long)pi_state.next_episode_id,
        pi_state.target_hash_chain.substr(0, 16),
        pi_state.extended_hash_chain.substr(0, 16));
    return 0;
  }

  // --- Phase 18B online auxiliary-search collection setup ---
  // The collector runs its 64-sim searches through a DuneNNEvaluator over the
  // FROZEN inference_model snapshot (never the training model, never post-update
  // weights). Collection is placed alongside each rollout (below) so aux and PPO
  // examples always come from the same snapshot. SyncModels only runs on the main
  // thread after the bg collection joins, so the snapshot is stable during a
  // collection even without the evaluator's sync_mutex.
  const bool online_search_collection = absl::GetFlag(FLAGS_online_search_collection);
  if (online_search_collection && !absl::GetFlag(FLAGS_search_label_dir).empty()) {
    SpielFatalError(
        "--online_search_collection and --search_label_dir are mutually "
        "exclusive (online collection vs legacy offline distillation).");
  }
  std::shared_ptr<open_spiel::OnlineSearchCollector> aux_collector;
  std::shared_ptr<open_spiel::DuneNNEvaluator> aux_evaluator;
  float aux_logit_cap = 0.0f;  // cap actually handed to aux_evaluator
  double aux_search_loss_coef_target = absl::GetFlag(FLAGS_search_loss_coef);
  int aux_search_loss_warmup = absl::GetFlag(FLAGS_search_loss_warmup_update);
  double aux_abort_ratio = absl::GetFlag(FLAGS_abort_grad_norm_ratio);
  open_spiel::OnlineSearchConfig aux_config;  // effective echo (persisted in manifest)
  if (online_search_collection) {
    if (absl::GetFlag(FLAGS_auxiliary_search_seed_domain) == 0) {
      SpielFatalError(
          "--auxiliary_search_seed_domain must be set nonzero when "
          "--online_search_collection is enabled (no silent default).");
    }
    aux_config.auxiliary_games = absl::GetFlag(FLAGS_auxiliary_games);
    aux_config.auxiliary_search_seed_domain =
        absl::GetFlag(FLAGS_auxiliary_search_seed_domain);
    aux_config.dirichlet_epsilon = absl::GetFlag(FLAGS_collector_dirichlet_epsilon);
    aux_config.acceptance_prior_source = EffectiveAcceptancePriorSource();
    aux_config.target_sharpen_exponent =
        absl::GetFlag(FLAGS_target_sharpen_exponent);
    aux_config.swordmaster_grant_fraction =
        absl::GetFlag(FLAGS_swordmaster_grant_fraction);
    aux_config.swordmaster_grant_round =
        absl::GetFlag(FLAGS_swordmaster_grant_round);
    // Leader teacher: explicit, never an implicit library default.
    aux_config.search_leader_draft = absl::GetFlag(FLAGS_search_leader_draft);
    aux_config.leader_draft_simulations =
        absl::GetFlag(FLAGS_leader_draft_simulations);
    aux_config.leader_mass_only_coverage =
        absl::GetFlag(FLAGS_leader_mass_only_coverage);
    if (aux_config.search_leader_draft &&
        aux_config.leader_draft_simulations <= 0) {
      SpielFatalError(
          "--leader_draft_simulations must be positive when "
          "--search_leader_draft is enabled.");
    }
    // Exact resume: continue the aux episode cursor from the manifest.
    aux_config.next_auxiliary_episode_id =
        static_cast<int64_t>(aux_next_episode_id_persist);
    // Same policy transform as the PPO rollout (which applies FLAGS_logit_cap
    // itself, below): the 18B teacher's priors and the student being trained
    // must not sit under different transforms. Echoed on the startup line so a
    // run log records the cap the aux evaluator actually received.
    aux_logit_cap = static_cast<float>(absl::GetFlag(FLAGS_logit_cap));
    aux_evaluator = std::make_shared<open_spiel::DuneNNEvaluator>(
        inference_model, device, aux_logit_cap);
    aux_collector = std::make_shared<open_spiel::OnlineSearchCollector>(
        aux_config, config_fingerprint);
    std::cout << absl::StrFormat(
        "[18B] Online collection ON | aux_games=%d seed_domain=%llu "
        "dirichlet_eps=%.3f grant_frac=%.3f grant_round=%d loss_coef=%.3f/warmup%d "
        "abort_ratio=%.3f sharpen=%.3f resume_ep=%llu logit_cap=%.3f "
        "accept_prior=%s\n",
        aux_config.auxiliary_games,
        (unsigned long long)aux_config.auxiliary_search_seed_domain,
        aux_config.dirichlet_epsilon, aux_config.swordmaster_grant_fraction,
        aux_config.swordmaster_grant_round, aux_search_loss_coef_target,
        aux_search_loss_warmup, aux_abort_ratio, aux_config.target_sharpen_exponent,
        (unsigned long long)aux_next_episode_id_persist, aux_logit_cap,
        // Which prior the acceptance rate in this run's diagnostics was measured
        // against — noise does not silently redefine it (WO-20).
        open_spiel::AcceptancePriorSourceName(aux_config.acceptance_prior_source));
  }

  // Collect one update's aux examples into `cr`, from the frozen snapshot. Runs
  // in whatever thread collects the paired PPO rollout (pre-loop main thread or
  // the bg pipeline thread), advancing the collector's private episode cursor.
  auto run_aux_collection = [&](int update_id, open_spiel::CollectResult& cr) {
    if (!online_search_collection) return;
    aux_collector->CollectUpdate(update_id, game, aux_evaluator, &cr.aux_examples,
                                 &cr.aux_stats);
  };
  // Fold a CONSUMED update's collection into the persisted cumulative counters,
  // the chained accepted-target hash, and the resume cursor (main thread only,
  // so no race with the bg collector).
  auto account_aux_consumed = [&](const open_spiel::CollectResult& cr) {
    if (!online_search_collection) return;
    const open_spiel::OnlineSearchCollectionStats& st = cr.aux_stats;
    aux_cum_accepted += st.accepted_targets;
    aux_cum_rejected += st.rejected_incomplete;
    aux_cum_granted += st.swordmaster_granted_games;
    aux_cum_organic += st.swordmaster_organic_games;
    for (int r = 0; r < open_spiel::OnlineCollectionState::kNumSearchRoles; ++r) {
      aux_cum_role_searches[r] += st.by_role[r].searches;
      aux_cum_role_accepted[r] += st.by_role[r].accepted;
    }
    aux_next_episode_id_persist = static_cast<uint64_t>(st.next_episode_id);
    std::string digest;
    for (const auto& ex : cr.aux_examples) {
      digest.append(reinterpret_cast<const char*>(&ex.episode_id), sizeof(ex.episode_id));
      int64_t did = ex.decision_id;
      digest.append(reinterpret_cast<const char*>(&did), sizeof(did));
      int32_t pl = ex.player;
      digest.append(reinterpret_cast<const char*>(&pl), sizeof(pl));
      for (Action a : ex.legal_actions)
        digest.append(reinterpret_cast<const char*>(&a), sizeof(a));
      for (double v : ex.normalized_visits)
        digest.append(reinterpret_cast<const char*>(&v), sizeof(v));
      digest.append(reinterpret_cast<const char*>(&ex.value_target), sizeof(ex.value_target));
    }
    aux_hash_chain = open_spiel::ComputeStringSHA256(aux_hash_chain + digest);
  };
  // Snapshot the persisted online-collection state for a checkpoint manifest.
  auto build_aux_state = [&]() {
    open_spiel::OnlineCollectionState s;
    s.present = true;
    s.auxiliary_games = aux_config.auxiliary_games;
    s.auxiliary_search_seed_domain = aux_config.auxiliary_search_seed_domain;
    s.collector_dirichlet_epsilon = aux_config.dirichlet_epsilon;
    s.target_sharpen_exponent = aux_config.target_sharpen_exponent;
    // Travels with cum_accepted/cum_rejected/accepted_hash_chain below: those
    // totals mean nothing without the contract they were measured under.
    s.acceptance_prior_source =
        AcceptancePriorSourceName(aux_config.acceptance_prior_source);
    s.swordmaster_grant_fraction = aux_config.swordmaster_grant_fraction;
    s.swordmaster_grant_round = aux_config.swordmaster_grant_round;
    s.search_loss_coef_target = aux_search_loss_coef_target;
    s.search_loss_warmup_update = aux_search_loss_warmup;
    s.abort_grad_norm_ratio = aux_abort_ratio;
    s.next_auxiliary_episode_id = aux_next_episode_id_persist;
    s.cum_accepted = aux_cum_accepted;
    s.cum_rejected = aux_cum_rejected;
    for (int r = 0; r < open_spiel::OnlineCollectionState::kNumSearchRoles;
         ++r) {
      s.cum_role_searches[r] = aux_cum_role_searches[r];
      s.cum_role_accepted[r] = aux_cum_role_accepted[r];
    }
    s.cum_granted = aux_cum_granted;
    s.cum_organic = aux_cum_organic;
    s.accepted_hash_chain = aux_hash_chain;
    return s;
  };

  std::cout << absl::StrFormat(
      "Initialized dune_ppo_train | obs=%d actions=%d hidden=%d blocks=%d "
      "rollout_transitions=%d rollout_games=%d minibatch=%d epochs=%d clip=%.3f vf_coef=%.3f "
      "ent_coef=%.4f gamma=%.3f gae_lambda=%.3f\n",
      obs_size, action_size, absl::GetFlag(FLAGS_hidden_dim),
      absl::GetFlag(FLAGS_num_blocks),
      absl::GetFlag(FLAGS_rollout_transitions),
      absl::GetFlag(FLAGS_rollout_games),
      absl::GetFlag(FLAGS_ppo_minibatch_size),
      absl::GetFlag(FLAGS_ppo_update_epochs),
      absl::GetFlag(FLAGS_ppo_clip_epsilon),
      absl::GetFlag(FLAGS_value_coef), absl::GetFlag(FLAGS_entropy_coef),
      absl::GetFlag(FLAGS_gamma), absl::GetFlag(FLAGS_gae_lambda));

  uint64_t total_games = 0;
  uint64_t total_moves = 0;
  auto training_start = std::chrono::high_resolution_clock::now();

  double base_lr = absl::GetFlag(FLAGS_learning_rate);
  bool pipeline = absl::GetFlag(FLAGS_pipeline);
  int num_threads = absl::GetFlag(FLAGS_threads);

  open_spiel::SearchLabelBuffer search_buffer;
  search_buffer.SetExpectedDimensions(obs_size, action_size);
  std::string search_label_dir = absl::GetFlag(FLAGS_search_label_dir);
  if (!search_label_dir.empty()) {
    search_buffer.LoadFromDirectory(search_label_dir);
  }

  int rollout_games = absl::GetFlag(FLAGS_rollout_games);

  std::string initial_non_value_hash = "";
  if (absl::GetFlag(FLAGS_train_value_only)) {
    initial_non_value_hash = HashNonValueParameters(training_model);
    std::cout << "[INFO] Initial non-value parameters SHA256: " << initial_non_value_hash << "\n";
  }

  // Collect first rollout synchronously.
  const std::string parity_model_hash_before =
      (numerical_parity || vrpo_diagnostics || vrpo_schedule_screen ||
       vrpo_ppo_pilot || vrpo_ppo_continuation)
          ? open_spiel::HashAllModelState(training_model) : "";
  const std::string parity_inference_hash_before =
      (numerical_parity || vrpo_diagnostics || vrpo_schedule_screen ||
       vrpo_ppo_pilot || vrpo_ppo_continuation)
          ? open_spiel::HashAllModelState(inference_model) : "";
  const bool parity_tf32_cublas_before =
      (numerical_parity || vrpo_diagnostics) &&
      at::globalContext().allowTF32CuBLAS();
  const bool parity_tf32_cudnn_before =
      (numerical_parity || vrpo_diagnostics) &&
      at::globalContext().allowTF32CuDNN();
  float reward_lambda = ComputeRewardLambda(total_env_steps.load(),
                                            absl::GetFlag(FLAGS_shaping_start_env_steps),
                                            absl::GetFlag(FLAGS_shaping_decay_env_steps));
  open_spiel::VrpoCapturedEpisodeBuffer vrpo_capture_buffer;
  if (vrpo_schedule_screen || vrpo_ppo_pilot || vrpo_ppo_continuation) {
    std::string deadline_error;
    const bool deadline_ok = vrpo_schedule_screen
        ? vrpo_schedule_deadline.Check("before rollout collection",
                                       &deadline_error)
        : vrpo_ppo_pilot
            ? vrpo_ppo_pilot_deadline.Check("before rollout collection",
                                            &deadline_error)
            : vrpo_ppo_continuation_deadline.Check(
                  "before rollout collection", &deadline_error);
    if (!deadline_ok) {
      SpielFatalError(deadline_error);
    }
  }
  open_spiel::CollectResult current_collect = open_spiel::CollectRollout(
      game.get(), evaluator, obs_size, &total_env_steps, num_threads, &next_episode_id,
      rollout_games, reward_lambda,
      (vrpo_diagnostics || vrpo_one_update || vrpo_schedule_screen ||
       vrpo_ppo_pilot || vrpo_ppo_continuation)
          ? &vrpo_capture_buffer
          : nullptr);
  if (vrpo_schedule_screen || vrpo_ppo_pilot || vrpo_ppo_continuation) {
    std::string deadline_error;
    const bool deadline_ok = vrpo_schedule_screen
        ? vrpo_schedule_deadline.Check("after rollout collection",
                                       &deadline_error)
        : vrpo_ppo_pilot
            ? vrpo_ppo_pilot_deadline.Check("after rollout collection",
                                            &deadline_error)
            : vrpo_ppo_continuation_deadline.Check(
                  "after rollout collection", &deadline_error);
    if (!deadline_ok) {
      SpielFatalError(deadline_error);
    }
  }
  if (vrpo_capture) {
    std::vector<open_spiel::VrpoCapturedEpisode> episodes =
        vrpo_capture_buffer.TakeSorted();
    const std::string model_hash_after =
        open_spiel::HashAllModelState(training_model);
    const std::string inference_hash_after =
        open_spiel::HashAllModelState(inference_model);
    const bool tf32_cublas_after =
        at::globalContext().allowTF32CuBLAS();
    const bool tf32_cudnn_after =
        at::globalContext().allowTF32CuDNN();
    const bool valid = open_spiel::WriteVrpoCaptureArtifact(
        absl::GetFlag(FLAGS_vrpo_capture_output), episodes, current_collect,
        parity_command_line, vrpo_source_provenance,
        parity_source_manifest, parity_source_manifest_sha256,
        parity_source_global_update, parity_source_checkpoint_uuid,
        config_fingerprint, parity_model_hash_before, model_hash_after,
        parity_inference_hash_before, inference_hash_after,
        parity_tf32_cublas_before, parity_tf32_cudnn_before,
        tf32_cublas_after, tf32_cudnn_after, optimizer_constructed);
    std::cout << "VRPO diagnostics capture "
              << (valid ? "VALID" : "INVALID") << ": "
              << absl::GetFlag(FLAGS_vrpo_capture_output) << "\n";
    return valid ? 0 : 3;
  }
  if (vrpo_q_preflight) {
    std::vector<open_spiel::VrpoCapturedEpisode> episodes =
        vrpo_capture_buffer.TakeSorted();
    open_spiel::ResetVrpoQInstrumentation();
    std::vector<open_spiel::VrpoRolloutPairingView> rollout_views;
    rollout_views.reserve(current_collect.rollout.size());
    for (const auto& transition : current_collect.rollout) {
      open_spiel::VrpoRolloutPairingView view;
      view.episode_id = transition.episode_id;
      view.actor = transition.player_id;
      view.actor_observation = &transition.state;
      view.legal_actions = &transition.legal_actions;
      view.action = transition.action;
      view.chosen_log_probability = transition.old_log_prob;
      rollout_views.push_back(view);
    }
    const open_spiel::VrpoPreQGateResult pre_q_gate =
        open_spiel::ValidateVrpoPreQCaptureRolloutGate(
            episodes, rollout_views,
            absl::GetFlag(FLAGS_start_episode_id),
            absl::GetFlag(FLAGS_rollout_games));
    open_spiel::VrpoQPreflightResult q_result;
    q_result.pre_q_gate_valid = pre_q_gate.valid;
    q_result.pre_q_captured_rows = pre_q_gate.captured_rows;
    q_result.pre_q_rollout_rows = pre_q_gate.rollout_rows;
    q_result.pre_q_paired_rows = pre_q_gate.paired_rows;
    if (pre_q_gate.valid) {
      q_result = open_spiel::RunVrpoQReferencePreflight(
          episodes.front(), device,
          absl::GetFlag(FLAGS_vrpo_q_init_seed),
          absl::GetFlag(FLAGS_vrpo_q_chunk_rows),
          absl::GetFlag(FLAGS_vrpo_q_agreement_abs_tolerance),
          absl::GetFlag(FLAGS_vrpo_q_agreement_rel_tolerance),
          absl::GetFlag(FLAGS_vrpo_q_gpu_peak_increment_limit_bytes));
      q_result.pre_q_gate_valid = true;
      q_result.pre_q_captured_rows = pre_q_gate.captured_rows;
      q_result.pre_q_rollout_rows = pre_q_gate.rollout_rows;
      q_result.pre_q_paired_rows = pre_q_gate.paired_rows;
    } else {
      q_result.errors = pre_q_gate.errors;
      q_result.q_constructed = false;
      q_result.q_constructor_calls =
          open_spiel::VrpoQConstructorCalls();
      q_result.q_forward_calls =
          open_spiel::VrpoQForwardCheckedCalls();
    }
    const std::string model_hash_after =
        open_spiel::HashAllModelState(training_model);
    const std::string inference_hash_after =
        open_spiel::HashAllModelState(inference_model);
    const bool tf32_cublas_after =
        at::globalContext().allowTF32CuBLAS();
    const bool tf32_cudnn_after =
        at::globalContext().allowTF32CuDNN();
    const bool valid = open_spiel::WriteVrpoQPreflightArtifact(
        absl::GetFlag(FLAGS_vrpo_q_preflight_output), episodes,
        current_collect, q_result, parity_command_line,
        vrpo_source_provenance, parity_source_manifest,
        parity_source_manifest_sha256, parity_source_global_update,
        parity_source_checkpoint_uuid, config_fingerprint,
        parity_model_hash_before, model_hash_after,
        parity_inference_hash_before, inference_hash_after,
        parity_tf32_cublas_before, parity_tf32_cudnn_before,
        tf32_cublas_after, tf32_cudnn_after, optimizer_constructed);
    std::cout << "VRPO Q/reference preflight "
              << (valid ? "VALID" : "INVALID") << ": "
              << absl::GetFlag(FLAGS_vrpo_q_preflight_output) << "\n";
    return valid ? 0 : 3;
  }
  if (vrpo_one_update) {
    std::vector<open_spiel::VrpoCapturedEpisode> episodes =
        vrpo_capture_buffer.TakeSorted();
    std::vector<open_spiel::VrpoTrainingEpisode> training_episodes;
    open_spiel::VrpoPhase4ePairingStats pairing;
    std::string phase4e_error;
    if (!current_collect.episode_ids_unique ||
        current_collect.games !=
            static_cast<uint64_t>(open_spiel::kVrpoPhase4eGames) ||
        !open_spiel::BuildVrpoPhase4eTrainingEpisodes(
            episodes, current_collect.rollout,
            vrpo_one_update_binding.start_episode_id,
            open_spiel::kVrpoPhase4eGames, &training_episodes, &pairing,
            &phase4e_error)) {
      SpielFatalError("VRPO one-update complete-game capture/pairing failed: " +
                      phase4e_error);
    }
    open_spiel::VrpoActorForward actor_forward =
        [model = training_model](const torch::Tensor& input) {
          const auto output = model->forward(input);
          return open_spiel::VrpoActorTrainingOutput{
              output.logits, output.values};
        };
    open_spiel::VrpoQForward q_forward =
        [q = vrpo_one_update_q](const torch::Tensor& input) {
          torch::Tensor output;
          std::string error;
          if (!q->ForwardChecked(input, &output, &error)) {
            throw std::runtime_error("VRPO Q forward rejected: " + error);
          }
          return output;
        };
    open_spiel::json::Object result;
    if (!open_spiel::WriteVrpoPhase4eOneUpdate(
            vrpo_one_update_startup, vrpo_one_update_arm,
            vrpo_one_update_binding, vrpo_one_update_layout,
            vrpo_one_update_input_identity, training_episodes, pairing,
            training_model, vrpo_one_update_q, &vrpo_one_update_optimizers,
            actor_forward, q_forward, open_spiel::VrpoPhase4eFailurePoint::kNone,
            &result, &phase4e_error)) {
      SpielFatalError("VRPO one-update transaction failed: " + phase4e_error);
    }
    std::cout << "VRPO one-update VALID: "
              << vrpo_one_update_startup.output_root << "\n";
    return 0;
  }
  if (vrpo_schedule_screen) {
    std::vector<open_spiel::VrpoCapturedEpisode> episodes =
        vrpo_capture_buffer.TakeSorted();
    std::vector<open_spiel::VrpoTrainingEpisode> training_episodes;
    open_spiel::VrpoPhase4ePairingStats pairing;
    std::string schedule_error;
    if (!current_collect.episode_ids_unique ||
        current_collect.games !=
            static_cast<uint64_t>(open_spiel::kVrpoScheduleGames) ||
        !open_spiel::BuildVrpoPhase4eTrainingEpisodes(
            episodes, current_collect.rollout,
            vrpo_schedule_binding.start_episode_id,
            open_spiel::kVrpoScheduleGames, &training_episodes, &pairing,
            &schedule_error)) {
      SpielFatalError(
          "VRPO schedule screen complete-game capture/pairing failed: " +
          schedule_error);
    }
    const std::string model_hash_after_collection =
        open_spiel::HashAllModelState(training_model);
    const std::string inference_hash_after_collection =
        open_spiel::HashAllModelState(inference_model);
    if (model_hash_after_collection != parity_model_hash_before ||
        inference_hash_after_collection != parity_inference_hash_before) {
      SpielFatalError(
          "VRPO schedule screen collection mutated actor model state");
    }
    open_spiel::json::Object screen_result;
    if (!open_spiel::WriteVrpoScheduleScreen(
            vrpo_schedule_startup, vrpo_schedule_binding,
            vrpo_schedule_layout, vrpo_schedule_input_identity,
            training_episodes, pairing, training_model, vrpo_schedule_q,
            device, vrpo_schedule_deadline,
            open_spiel::VrpoScheduleFailurePoint::kNone,
            &screen_result, &schedule_error)) {
      SpielFatalError("VRPO schedule screen transaction failed: " +
                      schedule_error);
    }
    std::cout << "VRPO schedule health screen VALID: "
              << vrpo_schedule_startup.output_root << "\n";
    return 0;
  }
  if (vrpo_ppo_pilot) {
    std::vector<open_spiel::VrpoCapturedEpisode> captured =
        vrpo_capture_buffer.TakeSorted();
    std::string collection_actor_hash =
        vrpo_ppo_pilot_state.initial_actor_values_sha256;
    std::string first_actor_hash;
    std::string first_inference_hash;
    std::string first_collection_error;
    if (!open_spiel::vrpo_training_internal::ModuleValueSha256(
            *training_model, "", &first_actor_hash,
            &first_collection_error) ||
        !open_spiel::vrpo_training_internal::ModuleValueSha256(
            *inference_model, "", &first_inference_hash,
            &first_collection_error) ||
        first_actor_hash != collection_actor_hash ||
        first_inference_hash != collection_actor_hash) {
      SpielFatalError(
          "PPO pilot first collection actor/inference binding failed: " +
          first_collection_error);
    }
    for (int update = 1; update <= open_spiel::kVrpoPpoPilotUpdates;
         ++update) {
      std::vector<open_spiel::VrpoTrainingEpisode> training_episodes;
      open_spiel::VrpoPhase4ePairingStats pairing;
      std::string pilot_error;
      const uint64_t expected_start =
          vrpo_ppo_pilot_startup.start_episode_id +
          static_cast<uint64_t>(update - 1) *
              open_spiel::kVrpoPpoPilotGamesPerUpdate;
      if (!current_collect.episode_ids_unique ||
          current_collect.games != static_cast<uint64_t>(
              open_spiel::kVrpoPpoPilotGamesPerUpdate) ||
          !open_spiel::BuildVrpoPhase4eTrainingEpisodes(
              captured, current_collect.rollout, expected_start,
              open_spiel::kVrpoPpoPilotGamesPerUpdate, &training_episodes,
              &pairing, &pilot_error)) {
        open_spiel::json::Object invalid_result;
        std::string status_error;
        vrpo_ppo_pilot_state.had_failure = true;
        vrpo_ppo_pilot_state.failure_reason =
            "complete-game capture/pairing failed: " + pilot_error;
        open_spiel::WriteVrpoPpoPilotGlobalResult(
            vrpo_ppo_pilot_startup, vrpo_ppo_pilot_input_identity,
            vrpo_ppo_pilot_layout, vrpo_ppo_pilot_state, *training_model,
            *vrpo_ppo_pilot_actor_optimizer,
            vrpo_ppo_pilot_deadline, &invalid_result, &status_error);
        SpielFatalError("PPO pilot complete-game capture/pairing failed: " +
                        pilot_error);
      }
      open_spiel::VrpoPpoPilotDisposition disposition;
      open_spiel::json::Object update_result;
      if (!open_spiel::WriteVrpoPpoPilotUpdate(
              vrpo_ppo_pilot_startup, vrpo_ppo_pilot_binding,
              vrpo_ppo_pilot_layout, vrpo_ppo_pilot_input_identity,
              training_episodes, pairing, collection_actor_hash,
              training_model, vrpo_ppo_pilot_q,
              *vrpo_ppo_pilot_actor_optimizer, device,
              vrpo_ppo_pilot_deadline, update,
              open_spiel::VrpoPpoPilotFailurePoint::kNone,
              &vrpo_ppo_pilot_state, &disposition, &update_result,
              &pilot_error)) {
        open_spiel::json::Object invalid_result;
        std::string status_error;
        vrpo_ppo_pilot_state.had_failure = true;
        vrpo_ppo_pilot_state.failure_reason =
            "update transaction failed: " + pilot_error;
        open_spiel::WriteVrpoPpoPilotGlobalResult(
            vrpo_ppo_pilot_startup, vrpo_ppo_pilot_input_identity,
            vrpo_ppo_pilot_layout, vrpo_ppo_pilot_state, *training_model,
            *vrpo_ppo_pilot_actor_optimizer,
            vrpo_ppo_pilot_deadline, &invalid_result, &status_error);
        SpielFatalError("PPO pilot update transaction failed: " +
                        pilot_error);
      }
      if (disposition == open_spiel::VrpoPpoPilotDisposition::kEarlyStop ||
          disposition == open_spiel::VrpoPpoPilotDisposition::kComplete) {
        open_spiel::json::Object pilot_result;
        if (!open_spiel::WriteVrpoPpoPilotGlobalResult(
                vrpo_ppo_pilot_startup, vrpo_ppo_pilot_input_identity,
                vrpo_ppo_pilot_layout, vrpo_ppo_pilot_state, *training_model,
                *vrpo_ppo_pilot_actor_optimizer,
                vrpo_ppo_pilot_deadline, &pilot_result, &pilot_error)) {
          SpielFatalError("PPO pilot terminal status failed: " + pilot_error);
        }
        std::cout << "PPO pilot "
                  << pilot_result.at("classification").GetString() << ": "
                  << vrpo_ppo_pilot_startup.output_root << "\n";
        return 0;
      }

      open_spiel::SyncModels(training_model, inference_model, &sync_mutex);
      std::string inference_hash;
      if (!open_spiel::vrpo_training_internal::ModuleValueSha256(
              *training_model, "", &collection_actor_hash, &pilot_error) ||
          !open_spiel::vrpo_training_internal::ModuleValueSha256(
              *inference_model, "", &inference_hash, &pilot_error) ||
          inference_hash != collection_actor_hash ||
          !vrpo_ppo_pilot_deadline.Check("before next rollout collection",
                                         &pilot_error)) {
        vrpo_ppo_pilot_state.had_failure = true;
        vrpo_ppo_pilot_state.failure_reason =
            "inference synchronization failed: " + pilot_error;
        open_spiel::json::Object invalid_result;
        std::string status_error;
        open_spiel::WriteVrpoPpoPilotGlobalResult(
            vrpo_ppo_pilot_startup, vrpo_ppo_pilot_input_identity,
            vrpo_ppo_pilot_layout, vrpo_ppo_pilot_state, *training_model,
            *vrpo_ppo_pilot_actor_optimizer, vrpo_ppo_pilot_deadline,
            &invalid_result, &status_error);
        SpielFatalError("PPO pilot inference synchronization failed: " +
                        pilot_error);
      }
      open_spiel::VrpoCapturedEpisodeBuffer next_capture;
      current_collect = open_spiel::CollectRollout(
          game.get(), evaluator, obs_size, &total_env_steps, num_threads,
          &next_episode_id, rollout_games, reward_lambda, &next_capture);
      captured = next_capture.TakeSorted();
      std::string actor_after_collection;
      std::string inference_after_collection;
      if (!open_spiel::vrpo_training_internal::ModuleValueSha256(
              *training_model, "", &actor_after_collection, &pilot_error) ||
          !open_spiel::vrpo_training_internal::ModuleValueSha256(
              *inference_model, "", &inference_after_collection,
              &pilot_error) ||
          actor_after_collection != collection_actor_hash ||
          inference_after_collection != collection_actor_hash ||
          !vrpo_ppo_pilot_deadline.Check("after next rollout collection",
                                         &pilot_error)) {
        vrpo_ppo_pilot_state.had_failure = true;
        vrpo_ppo_pilot_state.failure_reason =
            "collection mutated actor or exceeded deadline: " + pilot_error;
        open_spiel::json::Object invalid_result;
        std::string status_error;
        open_spiel::WriteVrpoPpoPilotGlobalResult(
            vrpo_ppo_pilot_startup, vrpo_ppo_pilot_input_identity,
            vrpo_ppo_pilot_layout, vrpo_ppo_pilot_state, *training_model,
            *vrpo_ppo_pilot_actor_optimizer, vrpo_ppo_pilot_deadline,
            &invalid_result, &status_error);
        SpielFatalError("PPO pilot collection mutated the actor: " +
                        pilot_error);
      }
    }
    SpielFatalError("PPO pilot loop exited without a terminal disposition");
  }
  if (vrpo_ppo_continuation) {
    std::vector<open_spiel::VrpoCapturedEpisode> captured =
        vrpo_capture_buffer.TakeSorted();
    std::string collection_actor_hash =
        vrpo_ppo_continuation_state.prior_actor_values_sha256;
    std::string first_actor_hash;
    std::string first_inference_hash;
    std::string continuation_error;
    if (!open_spiel::vrpo_training_internal::ModuleValueSha256(
            *training_model, "", &first_actor_hash, &continuation_error) ||
        !open_spiel::vrpo_training_internal::ModuleValueSha256(
            *inference_model, "", &first_inference_hash,
            &continuation_error) ||
        first_actor_hash != collection_actor_hash ||
        first_inference_hash != collection_actor_hash) {
      SpielFatalError(
          "PPO continuation first collection actor binding failed: " +
          continuation_error);
    }
    for (int global_update =
             open_spiel::kVrpoPpoContinuationFirstGlobalUpdate;
         global_update <= open_spiel::kVrpoPpoContinuationLastGlobalUpdate;
         ++global_update) {
      const int local_update = global_update -
          open_spiel::kVrpoPpoContinuationFirstGlobalUpdate + 1;
      const uint64_t expected_start =
          vrpo_ppo_continuation_startup.start_episode_id +
          static_cast<uint64_t>(local_update - 1) *
              open_spiel::kVrpoPpoContinuationGamesPerUpdate;
      std::vector<open_spiel::VrpoTrainingEpisode> training_episodes;
      open_spiel::VrpoPhase4ePairingStats pairing;
      if (!current_collect.episode_ids_unique ||
          current_collect.games != static_cast<uint64_t>(
              open_spiel::kVrpoPpoContinuationGamesPerUpdate) ||
          !open_spiel::BuildVrpoPhase4eTrainingEpisodes(
              captured, current_collect.rollout, expected_start,
              open_spiel::kVrpoPpoContinuationGamesPerUpdate,
              &training_episodes, &pairing, &continuation_error)) {
        vrpo_ppo_continuation_state.had_failure = true;
        vrpo_ppo_continuation_state.failure_reason =
            "complete-game capture/pairing failed: " + continuation_error;
        open_spiel::json::Object invalid_result;
        std::string status_error;
        open_spiel::WriteVrpoPpoContinuationGlobalResult(
            vrpo_ppo_continuation_startup,
            vrpo_ppo_continuation_state, *training_model,
            *vrpo_ppo_continuation_actor_optimizer,
            vrpo_ppo_continuation_deadline, &invalid_result, &status_error);
        SpielFatalError(
            "PPO continuation capture/pairing failed: " +
            continuation_error);
      }
      open_spiel::VrpoPpoContinuationDisposition disposition;
      open_spiel::json::Object update_result;
      if (!open_spiel::WriteVrpoPpoContinuationUpdate(
              vrpo_ppo_continuation_startup,
              vrpo_ppo_continuation_binding,
              vrpo_ppo_continuation_layout, training_episodes, pairing,
              collection_actor_hash, training_model,
              vrpo_ppo_continuation_q,
              *vrpo_ppo_continuation_actor_optimizer, device,
              vrpo_ppo_continuation_deadline, global_update,
              open_spiel::VrpoPpoContinuationFailurePoint::kNone,
              &vrpo_ppo_continuation_state, &disposition, &update_result,
              &continuation_error)) {
        vrpo_ppo_continuation_state.had_failure = true;
        vrpo_ppo_continuation_state.failure_reason =
            "update transaction failed: " + continuation_error;
        open_spiel::json::Object invalid_result;
        std::string status_error;
        open_spiel::WriteVrpoPpoContinuationGlobalResult(
            vrpo_ppo_continuation_startup,
            vrpo_ppo_continuation_state, *training_model,
            *vrpo_ppo_continuation_actor_optimizer,
            vrpo_ppo_continuation_deadline, &invalid_result, &status_error);
        SpielFatalError("PPO continuation transaction failed: " +
                        continuation_error);
      }
      if (disposition ==
              open_spiel::VrpoPpoContinuationDisposition::kEarlyStop ||
          disposition ==
              open_spiel::VrpoPpoContinuationDisposition::kComplete) {
        open_spiel::json::Object continuation_result;
        if (!open_spiel::WriteVrpoPpoContinuationGlobalResult(
                vrpo_ppo_continuation_startup,
                vrpo_ppo_continuation_state, *training_model,
                *vrpo_ppo_continuation_actor_optimizer,
                vrpo_ppo_continuation_deadline, &continuation_result,
                &continuation_error)) {
          SpielFatalError("PPO continuation terminal status failed: " +
                          continuation_error);
        }
        std::cout << "PPO continuation "
                  << continuation_result.at("classification").GetString()
                  << ": " << vrpo_ppo_continuation_startup.output_root
                  << "\n";
        return 0;
      }

      open_spiel::SyncModels(training_model, inference_model, &sync_mutex);
      std::string inference_hash;
      if (!open_spiel::vrpo_training_internal::ModuleValueSha256(
              *training_model, "", &collection_actor_hash,
              &continuation_error) ||
          !open_spiel::vrpo_training_internal::ModuleValueSha256(
              *inference_model, "", &inference_hash,
              &continuation_error) ||
          inference_hash != collection_actor_hash ||
          !vrpo_ppo_continuation_deadline.Check(
              "before next rollout collection", &continuation_error)) {
        vrpo_ppo_continuation_state.had_failure = true;
        vrpo_ppo_continuation_state.failure_reason =
            "inference synchronization failed: " + continuation_error;
        open_spiel::json::Object invalid_result;
        std::string status_error;
        open_spiel::WriteVrpoPpoContinuationGlobalResult(
            vrpo_ppo_continuation_startup,
            vrpo_ppo_continuation_state, *training_model,
            *vrpo_ppo_continuation_actor_optimizer,
            vrpo_ppo_continuation_deadline, &invalid_result, &status_error);
        SpielFatalError("PPO continuation inference sync failed: " +
                        continuation_error);
      }
      open_spiel::VrpoCapturedEpisodeBuffer next_capture;
      current_collect = open_spiel::CollectRollout(
          game.get(), evaluator, obs_size, &total_env_steps, num_threads,
          &next_episode_id, rollout_games, reward_lambda, &next_capture);
      captured = next_capture.TakeSorted();
      std::string actor_after_collection;
      std::string inference_after_collection;
      if (!open_spiel::vrpo_training_internal::ModuleValueSha256(
              *training_model, "", &actor_after_collection,
              &continuation_error) ||
          !open_spiel::vrpo_training_internal::ModuleValueSha256(
              *inference_model, "", &inference_after_collection,
              &continuation_error) ||
          actor_after_collection != collection_actor_hash ||
          inference_after_collection != collection_actor_hash ||
          !vrpo_ppo_continuation_deadline.Check(
              "after next rollout collection", &continuation_error)) {
        vrpo_ppo_continuation_state.had_failure = true;
        vrpo_ppo_continuation_state.failure_reason =
            "collection mutated actor or exceeded deadline: " +
            continuation_error;
        open_spiel::json::Object invalid_result;
        std::string status_error;
        open_spiel::WriteVrpoPpoContinuationGlobalResult(
            vrpo_ppo_continuation_startup,
            vrpo_ppo_continuation_state, *training_model,
            *vrpo_ppo_continuation_actor_optimizer,
            vrpo_ppo_continuation_deadline, &invalid_result, &status_error);
        SpielFatalError("PPO continuation collection failed: " +
                        continuation_error);
      }
    }
    SpielFatalError(
        "PPO continuation loop exited without terminal disposition");
  }
  if (numerical_parity) {
    const auto batch_evaluator =
        std::dynamic_pointer_cast<open_spiel::BatchedEvaluator>(evaluator);
    if (batch_evaluator == nullptr) {
      SpielFatalError("v4 numerical parity requires BatchedEvaluator");
    }
    const open_spiel::EvaluatorStats evaluator_before_r =
        batch_evaluator->GetStats();
    std::vector<open_spiel::PpoParityBatchMembership> membership;
    membership.reserve(current_collect.rollout.size());
    for (const auto& transition : current_collect.rollout) {
      membership.push_back({transition.behavior_physical_batch_id,
                            transition.behavior_physical_batch_size,
                            transition.behavior_physical_batch_row});
    }
    const open_spiel::PpoParityBatchGeometry geometry =
        open_spiel::ReconstructPpoParityBatchGeometry(
            membership, absl::GetFlag(FLAGS_eval_batch_size));
    const std::string geometry_sha256 =
        open_spiel::ComputeStringSHA256(geometry.canonical_payload);

    open_spiel::PpoParityCellResult integrity_cell;
    integrity_cell.name = "rollout_cpu_recompute_integrity";
    integrity_cell.module_train_mode = false;
    integrity_cell.autocast_enabled = false;
    open_spiel::PpoParityIntegrityStats integrity_stats;
    const auto a_start = std::chrono::steady_clock::now();
    integrity_cell.rows = open_spiel::ReplayPpoCpuIntegrityCell(
        current_collect.rollout,
        static_cast<float>(absl::GetFlag(FLAGS_logit_cap)),
        absl::GetFlag(FLAGS_rollout_amp),
        &integrity_cell.schema_errors, &integrity_cell.row_evidence_sha256,
        &integrity_stats.raw_values_total,
        &integrity_stats.raw_values_bf16_exact,
        &integrity_stats.vector_width_rows_ok,
        &integrity_stats.behavior_vector_exact_rows,
        &integrity_stats.chosen_scalar_exact_rows);
    integrity_cell.wall_time_s = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - a_start).count();
    integrity_cell.raw_logit_sha256 =
        open_spiel::PpoParityCapturedRawSha256(current_collect.rollout);
    integrity_cell.policy_sha256 =
        open_spiel::PpoParityCapturedPolicySha256(current_collect.rollout);
    const int64_t transitions =
        static_cast<int64_t>(current_collect.rollout.size());
    const bool hard_capture_valid =
        current_collect.parity_capture_errors.empty() &&
        current_collect.parity_decisions == transitions &&
        current_collect.parity_full_width_ok == transitions &&
        current_collect.parity_full_finite_ok == transitions &&
        current_collect.parity_legal_ids_unique_in_range == transitions &&
        current_collect.parity_chosen_once == transitions &&
        integrity_cell.schema_errors.empty() &&
        static_cast<int64_t>(integrity_cell.rows.size()) == transitions &&
        integrity_stats.vector_width_rows_ok == transitions &&
        integrity_stats.behavior_vector_exact_rows == transitions &&
        integrity_stats.chosen_scalar_exact_rows == transitions &&
        geometry.valid && parity_tf32_cublas_before &&
        parity_tf32_cudnn_before &&
        evaluator_before_r.requests == static_cast<uint64_t>(transitions) &&
        evaluator_before_r.batches ==
            static_cast<uint64_t>(geometry.groups) &&
        evaluator_before_r.max_batch_size ==
            static_cast<uint64_t>(geometry.max_batch_size);

    open_spiel::PpoParityCellResult replay_cell;
    open_spiel::PpoParityReplayGeometryStats replay_stats;
    open_spiel::PpoParityCellResult learner_cell;
    learner_cell.name = "actual_learner_full_fp32_tf32_2048";
    learner_cell.module_train_mode = true;
    learner_cell.autocast_enabled = false;
    open_spiel::PpoParityCellResult geometry_model_cell;
    open_spiel::PpoParityOriginalGeometryModelStats geometry_model_stats;
    if (hard_capture_valid) {
      const auto r_start = std::chrono::steady_clock::now();
      replay_cell = open_spiel::ReplayPpoInferenceExactGeometry(
          batch_evaluator.get(), current_collect.rollout, geometry,
          action_size, static_cast<float>(absl::GetFlag(FLAGS_logit_cap)),
          evaluator_before_r, &replay_stats);
      replay_cell.wall_time_s = std::chrono::duration<double>(
          std::chrono::steady_clock::now() - r_start).count();
      const bool r_valid = replay_cell.schema_errors.empty() &&
          replay_stats.returned_metadata_exact_rows == transitions &&
          replay_stats.raw_legal_exact_rows == transitions &&
          replay_stats.cpu_policy_exact_rows == transitions &&
          replay_stats.captured_raw_sha256 == replay_stats.replay_raw_sha256 &&
          replay_stats.after.requests - replay_stats.before.requests ==
              static_cast<uint64_t>(transitions) &&
          replay_stats.after.batches - replay_stats.before.batches ==
              static_cast<uint64_t>(geometry.groups) &&
          replay_stats.after.max_batch_size ==
              replay_stats.before.max_batch_size;
      if (r_valid) {
        const auto b_start = std::chrono::steady_clock::now();
        learner_cell.rows = open_spiel::ReplayPpoNumericalParityCell(
            training_model, current_collect.rollout, obs_size, action_size,
            device, /*model_train_mode=*/true,
            /*learner_autocast=*/false, learner_cell.name,
            &learner_cell.schema_errors, &learner_cell.row_evidence_sha256,
            &learner_cell.model_forward_calls,
            &learner_cell.dense_mask_rows_ok,
            &learner_cell.raw_logit_sha256,
            &learner_cell.policy_sha256);
        learner_cell.wall_time_s = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - b_start).count();
        const auto d_start = std::chrono::steady_clock::now();
        geometry_model_cell =
            open_spiel::ReplayPpoTrainingModelOriginalGeometry(
                training_model, current_collect.rollout, geometry, obs_size,
                action_size, device,
                static_cast<float>(absl::GetFlag(FLAGS_logit_cap)),
                /*learner_autocast=*/false,
                &geometry_model_stats);
        geometry_model_cell.wall_time_s = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - d_start).count();
      } else {
        learner_cell.schema_errors.push_back(
            "skipped because R exact replay validity failed");
        geometry_model_cell.schema_errors.push_back(
            "skipped because R exact replay validity failed");
      }
    } else {
      replay_cell.name = "live_inference_exact_original_batches";
      replay_cell.schema_errors.push_back(
          "skipped because A/capture/geometry validity failed");
      learner_cell.schema_errors.push_back(
          "skipped because A/capture/geometry validity failed");
      geometry_model_cell.name = "training_model_exact_original_batches";
      geometry_model_cell.schema_errors.push_back(
          "skipped because A/capture/geometry validity failed");
    }
    const std::string parity_model_hash_after =
        open_spiel::HashAllModelState(training_model);
    const std::string parity_inference_hash_after =
        open_spiel::HashAllModelState(inference_model);
    const bool parity_tf32_cublas_after =
        at::globalContext().allowTF32CuBLAS();
    const bool parity_tf32_cudnn_after =
        at::globalContext().allowTF32CuDNN();
    const bool valid = open_spiel::WritePpoNumericalParityArtifactV5(
        absl::GetFlag(FLAGS_numerical_parity_output), integrity_cell,
        integrity_stats, replay_cell, replay_stats, learner_cell,
        geometry_model_cell, geometry_model_stats, geometry, geometry_sha256,
        current_collect,
        open_spiel::PpoParitySharedIdentitySha256(current_collect.rollout),
        parity_model_hash_before,
        parity_model_hash_after, parity_inference_hash_before,
        parity_inference_hash_after, parity_source_manifest,
        parity_source_manifest_sha256, parity_source_global_update,
        parity_source_checkpoint_uuid, config_fingerprint, device, obs_size,
        action_size, parity_command_line, parity_source_provenance,
        parity_tf32_cublas_before, parity_tf32_cudnn_before,
        parity_tf32_cublas_after, parity_tf32_cudnn_after);
    std::cout << "Raw-PPO numerical parity v5-A "
              << (valid ? "VALID classification" : "INVALID instrument")
              << ": "
              << absl::GetFlag(FLAGS_numerical_parity_output) << "\n"
              << "No optimizer checkpoint was loaded and optimizer.step was "
                 "not called; no backward pass or training update ran.\n";
    return valid ? 0 : 3;
  }
  if (absl::GetFlag(FLAGS_diagnostics_only)) {
    open_spiel::PpoUpdateStats stats =
        open_spiel::TrainPpoUpdate(training_model, *optimizer, current_collect.rollout,
                                   obs_size, action_size, device, master, start_update, anchor_model);
    stats.episode_ids_unique = current_collect.episode_ids_unique;
    AttachPrecapAbszStats(&stats, current_collect);
    AttachCanaryStats(&stats, current_collect);
    std::string diagnostics_path = absl::GetFlag(FLAGS_diagnostics_path);
    if (!diagnostics_path.empty()) {
      double val_kl = search_buffer.ComputeValidationKL(training_model, device, static_cast<float>(absl::GetFlag(FLAGS_logit_cap)));
      open_spiel::WriteDiagnostics(diagnostics_path, start_update, stats,
                                   current_collect.conflict_vp_generated,
                                   current_collect.conflict_vp_attributed,
                                   current_collect.conflict_vp_unattributed,
                                   master, run_uuid, absl::GetFlag(FLAGS_run_prefix), config_fingerprint,
                                   current_collect.raw_conflict_vp,
                                   current_collect.raw_noncombat_vp,
                                   current_collect.raw_total_vp,
                                   val_kl,
                                   absl::GetFlag(FLAGS_emit_canary_columns));
    }
    std::cout << "Diagnostics-only run complete. Exiting.\n";
    exit(0);
  }

  total_games += current_collect.games;
  total_moves += current_collect.moves;

  // Phase 18B: aux examples for the first update, from the same frozen snapshot
  // the pre-loop rollout used (sequential, main thread).
  run_aux_collection(start_update, current_collect);

  for (int update = start_update; update <= target_end_update; ++update) {
    if (absl::GetFlag(FLAGS_anneal_lr)) {
      double frac = 1.0 - static_cast<double>(update - 1) /
                              std::max(1, target_end_update);
      open_spiel::SetOptimizerLearningRate(*optimizer,
                                           std::max(1e-8, frac * base_lr));
    }

    auto wall_start = std::chrono::high_resolution_clock::now();

    // Launch background collection for the next update (pipelined).
    // The inference model is frozen during PPO training, so the background
    // collection safely uses the current policy snapshot.
    open_spiel::CollectResult next_collect;
    std::thread bg_collect_thread;
    bool have_bg = pipeline && update < target_end_update;
    if (have_bg) {
      float next_reward_lambda = ComputeRewardLambda(total_env_steps.load(),
                                                     absl::GetFlag(FLAGS_shaping_start_env_steps),
                                                     absl::GetFlag(FLAGS_shaping_decay_env_steps));
      bg_collect_thread = std::thread([&, next_reward_lambda]() {
        next_collect = open_spiel::CollectRollout(
            game.get(), evaluator, obs_size, &total_env_steps, num_threads, &next_episode_id,
            rollout_games, next_reward_lambda);
        // Phase 18B: aux for the NEXT update, from the same frozen snapshot, run
        // sequentially AFTER the rollout (design: collect PPO first, then aux).
        run_aux_collection(update + 1, next_collect);
      });
    }

    auto ppo_start = std::chrono::high_resolution_clock::now();
    const double this_search_coef =
        online_search_collection
            ? open_spiel::SearchLossCoefForUpdate(update, aux_search_loss_coef_target,
                                                  aux_search_loss_warmup)
            : 0.0;
    // --- PWO-5 section 8.6: draw this update's auxiliary batch. ----------
    //
    // ONE draw per update (aux = 0), on its own RNG stream. Reusing
    // kStreamSearchSampling is FORBIDDEN: the two samplers draw different
    // populations (41,132 rows vs 20,582) in different units (games vs rows),
    // so a shared stream would couple head and distillation row choice.
    open_spiel::Pwo5AuxBatch pwo5_batch;
    std::string pwo5_draw_digest;
    if (pwo5_aux_layout && pwo5_cfg.AnyActive()) {
      const uint64_t aux_seed = dune_seed::DeriveSeed(
          master, dune_seed::kDomainTrain, update, /*aux=*/0,
          dune_seed::kStreamAuxSampling);
      const auto draw = pwo5_store.Draw(
          aux_seed, absl::GetFlag(FLAGS_aux_games_per_update),
          absl::GetFlag(FLAGS_aux_rows_per_game),
          absl::GetFlag(FLAGS_aux_batches_per_update));
      pwo5_draw_digest = open_spiel::ComputeStringSHA256(draw.digest);
      if (update == 1) {
        // The manifest's update-1 digest was precomputed at startup. Assert
        // the REALIZED draw reproduces it, so the recorded value is a
        // measurement of this run rather than a prediction about it.
        if (pwo5_draw_digest != pwo5_update1_sampler_digest) {
          SpielFatalError(
              "PWO-5 section 8.6: the realized update-1 sampler digest " +
              pwo5_draw_digest + " does not match the precomputed digest " +
              pwo5_update1_sampler_digest +
              " recorded in the manifest. The sampler is not reproducible.");
        }
        std::cout << "[PWO-5] update-1 sampler digest " << pwo5_draw_digest
                  << " (matches the manifest)" << std::endl;
      }
      // Flatten batch -> game -> rows into one tensor set, with game_id a
      // 0-based index over the update's sampled games so the trajectory
      // weighting inside the loss is exact.
      std::vector<float> obs_flat;
      std::vector<float> fvp;
      std::vector<int64_t> tround, nact, gid;
      int64_t g_counter = 0;
      for (const auto& batch_games : draw.batches) {
        for (const auto& game_rows : batch_games) {
          for (int64_t r : game_rows) {
            const auto& row = pwo5_store.rows()[r];
            const float* o = pwo5_store.observation(r);
            obs_flat.insert(obs_flat.end(), o, o + pwo5_store.obs_size());
            fvp.push_back(row.final_vp_target);
            tround.push_back(row.terminal_round_class);
            nact.push_back(row.next_own_action);
            gid.push_back(g_counter);
          }
          ++g_counter;
        }
      }
      const int64_t nrows = static_cast<int64_t>(fvp.size());
      auto fopt = torch::TensorOptions().dtype(torch::kFloat32);
      auto iopt = torch::TensorOptions().dtype(torch::kInt64);
      pwo5_batch.obs = torch::from_blob(obs_flat.data(),
                                        {nrows, pwo5_store.obs_size()}, fopt)
                           .clone().to(device);
      pwo5_batch.final_vp =
          torch::from_blob(fvp.data(), {nrows}, fopt).clone().to(device);
      pwo5_batch.terminal_round =
          torch::from_blob(tround.data(), {nrows}, iopt).clone().to(device);
      pwo5_batch.next_action =
          torch::from_blob(nact.data(), {nrows}, iopt).clone().to(device);
      pwo5_batch.game_id =
          torch::from_blob(gid.data(), {nrows}, iopt).clone().to(device);
      pwo5_batch.num_games = g_counter;
      // Section 8.2's registered trajectory denominators, computed HERE from
      // the draw itself. Every drawn game contributes `aux_rows_per_game` rows,
      // and final_vp / terminal_round have a target on every row, so both are
      // the full drawn-game count. next_own_action can legitimately be lower:
      // a row whose decision is the game's LAST label row has no later action
      // (target -1), and a game all of whose sampled rows were terminal would
      // contribute nothing and drop out of the denominator entirely. That is
      // measured rather than assumed.
      {
        std::vector<char> na_present(static_cast<size_t>(g_counter), 0);
        for (std::size_t i = 0; i < nact.size(); ++i) {
          if (nact[i] >= 0) na_present[static_cast<size_t>(gid[i])] = 1;
        }
        int64_t na_games = 0;
        for (char c : na_present) na_games += c ? 1 : 0;
        pwo5_batch.sampled_games = g_counter;
        pwo5_batch.final_vp_games = g_counter;
        pwo5_batch.terminal_round_games = g_counter;
        pwo5_batch.next_own_action_games = na_games;
      }
      pwo5_batch.valid = true;
    }

    open_spiel::PpoUpdateStats stats =
        open_spiel::TrainPpoUpdate(training_model, *optimizer, current_collect.rollout,
                                   obs_size, action_size, device, master, update, anchor_model,
                                   current_collect.aux_examples, this_search_coef,
                                   online_search_collection ? aux_abort_ratio : 0.0,
                                   pwo5_batch, pwo5_cfg);
    stats.episode_ids_unique = current_collect.episode_ids_unique;
    AttachPrecapAbszStats(&stats, current_collect);
    AttachCanaryStats(&stats, current_collect);

    // Phase 18B clean abort at the latest valid checkpoint (plan §18C): a
    // per-update aux/PPO grad-norm ratio over threshold stops the run; this
    // update's mutated weights are discarded (never checkpointed) and we exit
    // nonzero with a distinct message.
    if (stats.aux_ratio_abort) {
      if (have_bg && bg_collect_thread.joinable()) bg_collect_thread.join();
      // Record the aborting update before exiting: its ratio is the one row a
      // post-mortem needs, and stderr is not a durable diagnostics channel.
      std::string abort_diagnostics_path = absl::GetFlag(FLAGS_diagnostics_path);
      if (!abort_diagnostics_path.empty()) {
        open_spiel::WriteDiagnostics(
            abort_diagnostics_path, update, stats,
            current_collect.conflict_vp_generated,
            current_collect.conflict_vp_attributed,
            current_collect.conflict_vp_unattributed, master, run_uuid,
            absl::GetFlag(FLAGS_run_prefix), config_fingerprint,
            current_collect.raw_conflict_vp, current_collect.raw_noncombat_vp,
            current_collect.raw_total_vp,
            search_buffer.ComputeValidationKL(
                training_model, device,
                static_cast<float>(absl::GetFlag(FLAGS_logit_cap))),
            absl::GetFlag(FLAGS_emit_canary_columns));
      }
      std::cerr << absl::StrFormat(
          "ABORT[18B grad-ratio]: update %d aux/PPO norm ratio %.4f > %.4f "
          "(aux_norm=%.4f ppo_norm=%.4f). Keeping last valid checkpoint; this "
          "update is NOT saved.\n",
          update, stats.aux_ppo_norm_ratio, aux_abort_ratio,
          stats.aux_grad_norm_mean, stats.ppo_grad_norm_mean);
      std::exit(3);
    }
    // Fold this consumed update's collection into the persisted cumulative
    // counters / hash chain / resume cursor (main thread only).
    account_aux_consumed(current_collect);

    if (absl::GetFlag(FLAGS_train_value_only)) {
      std::string current_non_value_hash = HashNonValueParameters(training_model);
      if (current_non_value_hash != initial_non_value_hash) {
        SpielFatalError(absl::StrFormat("Policy/trunk parameters changed during value-only training! Expected: %s, Got: %s",
                                        initial_non_value_hash, current_non_value_hash));
      }
    }

    double ppo_elapsed = std::chrono::duration<double>(
        std::chrono::high_resolution_clock::now() - ppo_start).count();

    // Log PPO updates to diagnostics CSV
    std::string diagnostics_path = absl::GetFlag(FLAGS_diagnostics_path);
    if (!diagnostics_path.empty()) {
      double val_kl = search_buffer.ComputeValidationKL(training_model, device, static_cast<float>(absl::GetFlag(FLAGS_logit_cap)));
      open_spiel::WriteDiagnostics(diagnostics_path, update, stats,
                                   current_collect.conflict_vp_generated,
                                   current_collect.conflict_vp_attributed,
                                   current_collect.conflict_vp_unattributed,
                                   master, run_uuid, absl::GetFlag(FLAGS_run_prefix), config_fingerprint,
                                   current_collect.raw_conflict_vp,
                                   current_collect.raw_noncombat_vp,
                                   current_collect.raw_total_vp,
                                   val_kl,
                                   absl::GetFlag(FLAGS_emit_canary_columns));
      // PWO-5 amendment 1 ruling 6: the head telemetry sidecar. Path DERIVED
      // from --diagnostics_path (no new flag), gated by the already-registered
      // --emit_canary_columns.
      if (absl::GetFlag(FLAGS_emit_canary_columns)) {
        open_spiel::WritePwo5HeadTelemetry(
            diagnostics_path, update, stats, run_uuid, config_fingerprint,
            static_cast<uint64_t>(absl::GetFlag(FLAGS_seed)));
      }
      // WO-PERF-TIMING. Path DERIVED from --diagnostics_path (no new path
      // flag), and self-gated: WritePhaseTiming returns immediately unless
      // --phase_timing_mode=phases armed the timer. `ppo_elapsed` is carried in
      // so the sidecar records the figure its attribution is compared against.
      open_spiel::WritePhaseTiming(diagnostics_path, update, stats, ppo_elapsed,
                                   run_uuid,
                                   absl::GetFlag(FLAGS_run_prefix));
    }

    // --- Search auxiliary distillation steps ---
    double search_kl_sum = 0.0;
    double search_grad_sum = 0.0;
    double search_lambda = absl::GetFlag(FLAGS_search_lambda);
    int search_minibatches_per_update = absl::GetFlag(FLAGS_search_minibatches_per_update);
    int search_minibatch_size = absl::GetFlag(FLAGS_search_minibatch_size);
    float logit_cap = static_cast<float>(absl::GetFlag(FLAGS_logit_cap));

    // Per-update refresh, for the streaming case where a producer drops new
    // .bin packs into the directory while training runs.
    //
    // A ROLE-AWARE dataset is CLOSED at load time -- its membership is exactly
    // the manifest's list, and a rescan would reload those files without their
    // roles, which is the precise defect ruling 5 removes. So the refresh is
    // SKIPPED rather than attempted; LoadNewFiles itself still refuses a
    // role-aware directory, which is what makes this a guard and not a
    // convention.
    if (!search_label_dir.empty() && !search_buffer.role_aware()) {
      search_buffer.LoadNewFiles(search_label_dir);
    }

    if (search_lambda > 0.0 && search_buffer.Size() > 0) {
      auto cpu_float = torch::TensorOptions().dtype(torch::kFloat32);
      auto cpu_bool = torch::TensorOptions().dtype(torch::kBool);

      for (int aux = 0; aux < search_minibatches_per_update; ++aux) {
        uint64_t aux_seed = dune_seed::DeriveSeed(master, dune_seed::kDomainTrain, update, aux, dune_seed::kStreamSearchSampling);
        std::mt19937 aux_rng = dune_seed::MakeRng32(aux_seed);
        std::vector<SearchLabel> search_batch = search_buffer.Sample(search_minibatch_size, &aux_rng);
        if (search_batch.empty()) continue;

        int64_t sb_size = search_batch.size();
        torch::Tensor search_states_cpu = torch::empty({sb_size, obs_size}, cpu_float);
        torch::Tensor search_masks_cpu = torch::zeros({sb_size, action_size}, cpu_bool);
        torch::Tensor search_teacher_probs_cpu = torch::zeros({sb_size, action_size}, cpu_float);

        float* states_ptr = search_states_cpu.data_ptr<float>();
        bool* masks_ptr = search_masks_cpu.data_ptr<bool>();
        float* teacher_ptr = search_teacher_probs_cpu.data_ptr<float>();

        for (int64_t i = 0; i < sb_size; ++i) {
          const auto& label = search_batch[i];
          std::memcpy(states_ptr + i * obs_size, label.state.data(), obs_size * sizeof(float));
          for (const auto& ap : label.teacher_probs) {
            int64_t action_id = ap.first;
            float prob = ap.second;
            if (action_id >= 0 && action_id < action_size) {
              masks_ptr[i * action_size + action_id] = true;
              teacher_ptr[i * action_size + action_id] = prob;
            }
          }
        }

        torch::Tensor search_states = search_states_cpu.to(device);
        torch::Tensor search_masks = search_masks_cpu.to(device);
        torch::Tensor search_teacher_probs = search_teacher_probs_cpu.to(device);

        optimizer->zero_grad();

        torch::Tensor mean_kl;
        auto compute_distill_loss = [&]() {
          auto outputs = training_model->forward(search_states);
          torch::Tensor logits = CenterAndCapLogitsTensor(outputs.logits, search_masks, logit_cap);
          torch::Tensor masked_logits = logits.masked_fill(search_masks.logical_not(), -1e9f);
          torch::Tensor log_probs = torch::log_softmax(masked_logits, -1);

          torch::Tensor log_teacher = torch::log(search_teacher_probs.clamp_min(1e-12f));
          torch::Tensor kl_loss = search_teacher_probs * (log_teacher - log_probs);
          mean_kl = kl_loss.sum(-1).mean();
        };

        if (device.is_cuda() && absl::GetFlag(FLAGS_train_amp)) {
          AutocastGuard autocast_guard(c10::DeviceType::CUDA, true);
          compute_distill_loss();
        } else {
          compute_distill_loss();
        }

        torch::Tensor search_loss = search_lambda * mean_kl;
        search_loss.backward();

        double search_grad_norm = torch::nn::utils::clip_grad_norm_(
            training_model->parameters(), std::numeric_limits<double>::infinity());
        if (!std::isfinite(search_grad_norm) || !std::isfinite(mean_kl.item<double>())) {
          std::cerr << "Warning: Non-finite search distillation gradient (" << search_grad_norm
                    << ") or loss. Skipping optimizer step.\n";
          optimizer->zero_grad();
          continue;
        }
        torch::nn::utils::clip_grad_norm_(
            training_model->parameters(), absl::GetFlag(FLAGS_grad_clip_norm));
        optimizer->step();

        search_kl_sum += mean_kl.item<double>();
        search_grad_sum += search_grad_norm;
      }
    }

    // Join background collection before syncing models.
    if (have_bg) {
      bg_collect_thread.join();
      total_games += next_collect.games;
      total_moves += next_collect.moves;
    }

    open_spiel::SyncModels(training_model, inference_model, &sync_mutex);

    auto wall_end = std::chrono::high_resolution_clock::now();

    double collect_elapsed = current_collect.elapsed_seconds;
    double wall_elapsed =
        std::chrono::duration<double>(wall_end - wall_start).count();
    double sps = collect_elapsed > 0.0
                     ? current_collect.rollout.size() / collect_elapsed
                     : 0.0;

    std::cout << absl::StrFormat(
        "Update %5d/%d | Transitions: %6zu | Games: %llu | "
        "Collect: %.2fs (%.0f t/s) | PPO: %.2fs | Wall: %.2fs | "
        "PolicyLoss: %.6f | ValueLoss: %.6f | Entropy: %.4f | "
        "ApproxKL: %.6f | ClipFrac: %.3f | ExplVar: %.3f%s\n",
        update, target_end_update, current_collect.rollout.size(),
        static_cast<unsigned long long>(total_games),
        collect_elapsed, sps, ppo_elapsed, wall_elapsed,
        stats.policy_loss, stats.value_loss, stats.entropy, stats.approx_kl,
        stats.clip_fraction, stats.explained_variance,
        stats.early_stopped ? " | early-stop" : "");

    // Phase 18B combined-optimization + collector diagnostics.
    if (online_search_collection) {
      const open_spiel::OnlineSearchCollectionStats& cst = current_collect.aux_stats;
      auto acc = [](const open_spiel::PerRoleSearchStats& r) {
        return r.searches > 0 ? static_cast<double>(r.accepted) / r.searches : 0.0;
      };
      auto mkl = [](const open_spiel::PerRoleSearchStats& r) {
        return r.searches > 0 ? r.sum_kl / r.searches : 0.0;
      };
      std::cout << absl::StrFormat(
          "  [18B Aux] used=%d coef=%.4f ce=%.4f vmse=%.4f | aux_norm=%.4f "
          "ppo_norm=%.4f ratio=%.3f\n",
          stats.aux_examples_used, this_search_coef, stats.aux_ce,
          stats.aux_value_mse, stats.aux_grad_norm_mean, stats.ppo_grad_norm_mean,
          stats.aux_ppo_norm_ratio);
      std::cout << absl::StrFormat(
          "  [18B Collector] accepted=%d rejected=%d wall=%.1fs | acc/KL "
          "primary %.2f/%.2f cont %.2f/%.2f purch %.2f/%.2f | SM granted=%d organic=%d\n",
          cst.accepted_targets, cst.rejected_incomplete, cst.collection_wall_time_s,
          acc(cst.by_role[0]), mkl(cst.by_role[0]), acc(cst.by_role[1]),
          mkl(cst.by_role[1]), acc(cst.by_role[2]), mkl(cst.by_role[2]),
          static_cast<int>(cst.swordmaster_granted_games),
          static_cast<int>(cst.swordmaster_organic_games));
    }

    std::cout << absl::StrFormat(
        "  [Conflict VP Stats] Generated: %.2f | Attributed: %.2f | Unattributed: %.2f\n",
        current_collect.conflict_vp_generated,
        current_collect.conflict_vp_attributed,
        current_collect.conflict_vp_unattributed);

    {
      uint64_t total_acquisitions = 0;
      for (int i = 0; i < 4; ++i) total_acquisitions += current_collect.sm_acquisitions_by_seat[i];
      if (total_acquisitions > 0) {
        std::cout << absl::StrFormat(
            "  [Natural SM Acquisitions] Total: %d | Seat P0: %d | P1: %d | P2: %d | P3: %d\n",
            total_acquisitions, current_collect.sm_acquisitions_by_seat[0],
            current_collect.sm_acquisitions_by_seat[1], current_collect.sm_acquisitions_by_seat[2],
            current_collect.sm_acquisitions_by_seat[3]);
        std::cout << absl::StrFormat(
            "  [SM Acquisitions by Round] R1: %d | R2: %d | R3: %d | R4: %d | R5: %d | R6-10: %d\n",
            current_collect.sm_acquisitions_by_round[1], current_collect.sm_acquisitions_by_round[2],
            current_collect.sm_acquisitions_by_round[3], current_collect.sm_acquisitions_by_round[4],
            current_collect.sm_acquisitions_by_round[5],
            current_collect.sm_acquisitions_by_round[6] + current_collect.sm_acquisitions_by_round[7] +
            current_collect.sm_acquisitions_by_round[8] + current_collect.sm_acquisitions_by_round[9] +
            current_collect.sm_acquisitions_by_round[10]);
        std::cout << absl::StrFormat(
            "  [SM Acquisitions by Leader] Leto (4): %d | Rabban (6): %d | Other Leaders: %d\n",
            current_collect.sm_acquisitions_by_leader[4], current_collect.sm_acquisitions_by_leader[6],
            total_acquisitions - current_collect.sm_acquisitions_by_leader[4] - current_collect.sm_acquisitions_by_leader[6]);
      }
    }

    if (search_lambda > 0.0 && search_buffer.Size() > 0) {
      double avg_search_kl = (search_minibatches_per_update > 0) ? (search_kl_sum / search_minibatches_per_update) : 0.0;
      double avg_search_grad = (search_minibatches_per_update > 0) ? (search_grad_sum / search_minibatches_per_update) : 0.0;
      double avg_ppo_grad = (stats.grad_norm_count > 0) ? (stats.grad_norm_sum / stats.grad_norm_count) : 0.0;
      double ratio = (avg_ppo_grad > 0.0) ? (avg_search_grad / avg_ppo_grad) : 0.0;
      std::cout << absl::StrFormat(
          "SearchKL: %.3f | SearchGrad: %.3f | PPOGrad: %.3f | Ratio: %.2f | EV: %.3f\n",
          avg_search_kl, avg_search_grad, avg_ppo_grad, ratio, stats.explained_variance);
    }

    int checkpoint_interval = absl::GetFlag(FLAGS_checkpoint_interval);
    int training_step = update - start_update + 1;
    bool is_pilot_update = (training_step == 10 || training_step == 25 || training_step == 50);
    if (is_pilot_update || (checkpoint_interval > 0 && update % checkpoint_interval == 0)) {
      std::string prefix = absl::GetFlag(FLAGS_run_prefix);
      std::string model_path =
          absl::StrCat(prefix, "_model_update_", update, ".pt");
      std::string optim_path =
          absl::StrCat(prefix, "_optimizer_update_", update, ".pt");
      open_spiel::OnlineCollectionState aux_ckpt = build_aux_state();
      open_spiel::SaveCheckpoint(training_model, *optimizer, model_path,
                                 optim_path, update, target_end_update,
                                 total_env_steps.load(), next_episode_id.load(),
                                 master, absl::GetFlag(FLAGS_seed_scheme_version),
                                 config_fingerprint, search_label_fingerprint,
                                 run_uuid,
                                 online_search_collection ? &aux_ckpt : nullptr);
    }

    // Advance to next rollout.
    if (have_bg) {
      current_collect = std::move(next_collect);
    } else if (update < target_end_update) {
      float next_reward_lambda = ComputeRewardLambda(total_env_steps.load(),
                                                     absl::GetFlag(FLAGS_shaping_start_env_steps),
                                                     absl::GetFlag(FLAGS_shaping_decay_env_steps));
      current_collect = open_spiel::CollectRollout(
          game.get(), evaluator, obs_size, &total_env_steps, num_threads, &next_episode_id,
          rollout_games, next_reward_lambda);
      total_games += current_collect.games;
      total_moves += current_collect.moves;
      // Phase 18B (non-pipelined path): aux for the next update, same snapshot.
      run_aux_collection(update + 1, current_collect);
    }
  }

  auto training_end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = training_end - training_start;
  auto eval_stats = evaluator->GetStats();
  std::cout << absl::StrFormat(
      "\n=== PPO Training Complete ===\nElapsed: %.2fs\nEnv Steps: %llu\n"
      "Games: %llu\nMoves: %llu\nEvaluator Requests: %llu | Batches: %llu | "
      "Avg Batch: %.2f | Max Batch: %llu\n",
      elapsed.count(),
      static_cast<unsigned long long>(total_env_steps.load()),
      static_cast<unsigned long long>(total_games),
      static_cast<unsigned long long>(total_moves),
      static_cast<unsigned long long>(eval_stats.requests),
      static_cast<unsigned long long>(eval_stats.batches),
      eval_stats.avg_batch_size,
      static_cast<unsigned long long>(eval_stats.max_batch_size));

  if (absl::GetFlag(FLAGS_save_final_checkpoint)) {
    open_spiel::OnlineCollectionState aux_final = build_aux_state();
    open_spiel::SaveCheckpoint(training_model, *optimizer,
                               absl::GetFlag(FLAGS_model_checkpoint),
                               absl::GetFlag(FLAGS_optim_checkpoint),
                               target_end_update, target_end_update,
                               total_env_steps.load(), next_episode_id.load(),
                               master, absl::GetFlag(FLAGS_seed_scheme_version),
                               config_fingerprint, search_label_fingerprint,
                               run_uuid,
                               online_search_collection ? &aux_final : nullptr);
  }
  return 0;
#endif
}
