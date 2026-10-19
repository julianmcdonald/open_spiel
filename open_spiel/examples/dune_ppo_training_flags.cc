// Trainer flags that dune_ppo_training_utils.cc reads.
//
// Every binary that links dune_ppo_training_utils.cc needs these defined.
// They live here, with the trainer's defaults, so dune_ppo_train and the
// tools that reuse its training utilities share one definition. The few
// tests that need different defaults define all twelve themselves and do
// not link this file. Declarations: dune_ppo_training_flags.h.

#include <cstdint>
#include <string>

#include "open_spiel/abseil-cpp/absl/flags/flag.h"
#include "dune_ppo_training_flags.h"

// Rollout batch size.
ABSL_FLAG(int, rollout_games, 0,
          "Exact complete games collected per PPO update rollout batch. "
          "If > 0, overrides transition-threshold mode.");

// Network architecture and input layout.
ABSL_FLAG(int, hidden_dim, 2048, "Network hidden dimension.");
ABSL_FLAG(int, num_blocks, 8, "Network residual block count.");
ABSL_FLAG(bool, nonlinear_value_head, false, "Use a nonlinear value head.");
ABSL_FLAG(std::string, market_appendix_mode, "none",
          "Card-slot input mode: 'none' (5580), 'zeros' or 'ordered_market' (6215), "
          "or 'ordered_card_slots_v2' (7271: Imperium, Tleilaxu, and own Helena reserve).");

// Runtime numerics and action scoring.
ABSL_FLAG(bool, allow_tf32, true,
          "Allow TF32 for CUDA CuBLAS/CuDNN. Default true preserves the "
          "historical runtime policy.");
ABSL_FLAG(bool, enable_semantic_scorer, false, "Enable the neural Semantic Action Scorer on legal candidate actions.");
ABSL_FLAG(bool, enable_zone_encoder, false, "Enable the zone encoder adapter feeding trunk.");
ABSL_FLAG(std::string, semantic_descriptor_schema, "semantic_action_v3",
          "Semantic descriptor schema version (e.g. semantic_action_v3, semantic_action_v4).");

// Training-only reward shaping.
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
ABSL_FLAG(double, family_atomics_penalty, 0.0,
          "Magnitude of the negative shaping SUBTRACTED from a transition that "
          "uses Family Atomics (ID 1591) when NOT in that player's own reveal turn "
          "with remaining persuasion > 0. MUST BE >= 0. Typical 0.10. Requires "
          "--allow_shaping.");
ABSL_FLAG(double, plot_intrigue_penalty, 0.0,
          "Magnitude of the negative shaping SUBTRACTED from a transition that "
          "plays a plot intrigue card (IDs 1600-1699) when the player holds fewer "
          "than plot_intrigue_exemption_threshold intrigue cards before playing. "
          "MUST BE >= 0. Typical 0.01. Requires --allow_shaping.");
ABSL_FLAG(int, plot_intrigue_exemption_threshold, 3,
          "Intrigue hand size threshold at or above which plot intrigue plays are "
          "exempt from penalty. Default 3.");

// PWO-5 gate 3 (registered in docs/PWO5_PILOT_REGISTRATION.md Appendix A.1;
// name, type and default are frozen).
ABSL_FLAG(uint64_t, head_init_constant, 20260800,
          "PWO-5 section 7.2: kHeadInitConstant. NOT a run seed -- it is a "
          "fixed registered constant so that the initial head parameters are "
          "byte-identical across ALL SIX arms. Deriving from the triplet seed "
          "would give T1/P1/H1 one set and T2/P2/H2 another. Outside the "
          "reserved final-gate base-seed range by section 10.2's arithmetic.");

// PPO optimization and loss hyperparameters.
ABSL_FLAG(int, ppo_minibatch_size, 2048, "PPO minibatch size.");
ABSL_FLAG(int, ppo_update_epochs, 4, "PPO epochs per rollout batch.");
ABSL_FLAG(bool, normalize_advantages, true,
          "Normalize advantages within each PPO minibatch.");
ABSL_FLAG(double, ppo_clip_epsilon, 0.2, "PPO surrogate/value clip epsilon.");
ABSL_FLAG(bool, ppo_clip_value_loss, true, "Use clipped value loss.");
ABSL_FLAG(double, entropy_coef, 0.01, "Entropy bonus coefficient.");
ABSL_FLAG(double, value_coef, 0.5, "Value loss coefficient.");
ABSL_FLAG(double, grad_clip_norm, 0.5, "Maximum gradient norm.");
ABSL_FLAG(double, target_kl, 0.0,
          "Stop PPO epochs early if approximate KL exceeds this value. <=0 disables.");
ABSL_FLAG(double, logit_cap, 10.0,
          "Smooth tanh cap on legal-centered policy logits. <=0 disables.");
ABSL_FLAG(bool, train_amp, true, "Use CUDA BF16 autocast for PPO updates.");
ABSL_FLAG(bool, diagnostics_only, false,
          "Collect one rollout, write diagnostics, and exit without optimization.");
ABSL_FLAG(uint64_t, shaping_start_env_steps, 206830543,
          "Environment steps at which shaped reward decay starts.");
ABSL_FLAG(uint64_t, shaping_decay_env_steps, 0,
          "Environment steps duration for shaped reward decay (0 means no decay).");

// 12-block H/J study flags.
ABSL_FLAG(bool, enable_ema_tracking, false,
          "Enable persistent EMA tracking for evaluation and play.");
ABSL_FLAG(double, ema_half_life, 100.0,
          "EMA half-life in completed updates (default 100).");
ABSL_FLAG(double, ema_decay, 0.0,
          "Explicit EMA decay factor per update. If <= 0, computed from ema_half_life: pow(0.5, 1/half_life).");
ABSL_FLAG(std::string, ema_model_checkpoint, "",
          "Optional path to existing EMA model checkpoint to resume.");

// EMAgnet forward KL regularization flags.
ABSL_FLAG(double, ema_magnet_kl_coef, 0.0,
          "Forward KL regularization coefficient toward moving parameter-EMA policy: "
          "adds ema_magnet_kl_coef * KL(pi_EMA || pi_theta) to actor loss.");
ABSL_FLAG(bool, allow_ema_magnet_transition, false,
          "Allow transition from a parent manifest with different ema_magnet_kl_coef.");
ABSL_FLAG(double, ema_magnet_transition_source_coef, 0.0,
          "Expected parent ema_magnet_kl_coef for transition.");
ABSL_FLAG(std::string, ema_magnet_transition_source_fingerprint, "",
          "Expected parent config fingerprint for ema_magnet transition.");

// Mixed-episode multi-learner opponent pool flags.
ABSL_FLAG(double, pool_game_fraction, 0.0,
          "Fraction of training games with opponent pool seats (default 0.0).");
ABSL_FLAG(int, pool_opponents_per_game, 0,
          "Number of opponent pool seats in a selected pool game (1..3, default 0).");
ABSL_FLAG(bool, allow_pool_transition, false,
          "Allow transition from a parent manifest with different pool settings.");
ABSL_FLAG(std::string, pool_transition_source_fingerprint, "",
          "Expected parent config fingerprint for pool transition.");

