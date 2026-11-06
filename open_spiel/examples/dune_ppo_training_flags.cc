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
          "'ordered_card_slots_v2' (6255), 'full_public_information_v3' (9182), "
          "'full_public_information_v4' (9941), or 'full_public_information_v5' (10445).");
ABSL_FLAG(std::string, deck_pool_semantics, "fixed",
          "Observation semantics for deck pool features: 'fixed' (default) or 'legacy'.");
ABSL_FLAG(bool, allow_deck_pool_semantics_transition, false,
          "Permit a one-time deck_pool_semantics transition on resume at the source update.");
ABSL_FLAG(std::string, deck_pool_semantics_transition_source_fingerprint, "",
          "Expected source config fingerprint for deck_pool_semantics transition validation.");
ABSL_FLAG(int64_t, deck_pool_semantics_transition_source_update, -1,
          "Expected source global_update for one-time deck_pool_semantics transition.");
ABSL_FLAG(int, start_update, 1, "Fallback start update for bootstrap mode.");
ABSL_FLAG(bool, allow_market_appendix_transition, false,
          "Permit initializing a run from a checkpoint with a different market_appendix_mode.");
ABSL_FLAG(std::string, market_appendix_transition_source_mode, "",
          "Expected source market_appendix_mode for transition validation.");
ABSL_FLAG(std::string, market_appendix_transition_source_fingerprint, "",
          "Expected source config fingerprint for transition validation.");

// Runtime numerics and action scoring.
ABSL_FLAG(bool, allow_tf32, true,
          "Allow TF32 for CUDA CuBLAS/CuDNN. Default true preserves the "
          "historical runtime policy.");
ABSL_FLAG(bool, allow_tf32_transition, false,
          "Allow intentional allow_tf32 transition when resuming from checkpoint.");
ABSL_FLAG(std::string, tf32_transition_source_fingerprint, "",
          "Expected source config fingerprint for allow_tf32 transition validation.");
ABSL_FLAG(bool, enable_semantic_scorer, false, "Enable the neural Semantic Action Scorer on legal candidate actions.");
ABSL_FLAG(bool, enable_zone_encoder, false, "Enable the zone encoder adapter feeding trunk.");
ABSL_FLAG(std::string, semantic_descriptor_schema, "semantic_action_v4",
          "Semantic descriptor schema version (e.g. semantic_action_v3, semantic_action_v4).");
ABSL_FLAG(bool, allow_semantic_schema_transition, false,
          "Allow intentional transition of semantic_descriptor_schema on resume.");
ABSL_FLAG(std::string, semantic_schema_transition_source_fingerprint, "",
          "Expected source config fingerprint for semantic schema transition validation.");
ABSL_FLAG(bool, allow_legacy_v3_on_v4_obs, false,
          "Allow legacy semantic_action_v3 schema on full_public_information_v4 observation.");
ABSL_FLAG(bool, slot_indexed_out_layer_f, false,
          "Enable slot-indexed out_layer_f (2 output units for Tleilaxu slots 0/1) for semantic action scorer.");

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
ABSL_FLAG(double, v4_appendix_lr_mult, 1.0,
          "Multiplier applied to the Adam parameter update on the appendix input columns (>= 9182). Set to 10.0 to match new_block_lr=5e-5 when base_lr=5e-6.");
ABSL_FLAG(int, v5_anneal_legacy_slot_residual_updates, 0,
          "When > 0 on semantic_action_v5, linearly anneals within-family asymmetric residuals (w_s - bar_w_F, b_s - bar_b_F) on dynamic purchase/choice slot families in policy_head to zero over the first N updates of the run (preserving Step-0 parity at start and exact zero slot bias after N updates).");
ABSL_FLAG(bool, actor_only_advantage_filtering, false,
          "EasyPPO (arXiv:2609.36802v1): when true with --advantage_filtering, keep 100% of transitions for critic value_loss regression and apply the top-quantile |A| filter strictly to the actor policy loss.");
ABSL_FLAG(double, acpo_surrogate_clip, 0.0,
          "ACPO (arXiv:2609.36816v2): when > 0.0, clamp standardized minibatch advantages in the PPO surrogate loss to [-acpo_surrogate_clip, +acpo_surrogate_clip] to bound outlier policy gradient spikes.");
ABSL_FLAG(double, acpo_surrogate_clip_neg, 0.0,
          "DAZ-PPO (arXiv:2610.32194): when > 0, clips negative standardized advantages at -acpo_surrogate_clip_neg (e.g. 1.5 -> lower bound -1.5) inside the PPO policy surrogate loss.");
ABSL_FLAG(double, acpo_surrogate_clip_pos, 0.0,
          "DAZ-PPO (arXiv:2610.32194): when > 0, clips positive standardized advantages at +acpo_surrogate_clip_pos (e.g. 2.5 -> upper bound +2.5) inside the PPO policy surrogate loss.");
ABSL_FLAG(bool, phase_stratified_advantage_norm, false,
          "When true, standardizes GAE advantages within each decision-role stratum across multi-choice rollout rows before top-quantile filtering or minibatch updates.");

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

// R7-55 / R7-57 hybrid search-in-training flags.
ABSL_FLAG(bool, r755_hybrid_mode, false,
          "Enable R7-55/R7-57 hybrid 1-round search-in-training mode.");
ABSL_FLAG(double, r755_search_prob, 0.25,
          "Per-turn / per-root search probability in R7-55 hybrid mode.");
ABSL_FLAG(int, r755_category_mask, 31,
          "Bitmask of enabled R7-55 search categories (bits 0..4 for Categories 1..5).");
ABSL_FLAG(double, r755_min_t_stat, 1.0,
          "Minimum paired t-statistic across rollouts to accept an override in R7-55 mode.");
ABSL_FLAG(bool, r755_execute_overrides, true,
          "When true, execute accepted overrides a* in the training trajectory.");
ABSL_FLAG(std::string, r755_gate1_output, "",
          "If non-empty, run Gate 1 40-game search & Ref512 audit and write JSON to this path.");
ABSL_FLAG(std::string, r755_gate2_output, "",
          "If non-empty, emit Gate 2 post-update verification diagnostics JSON to this path.");
ABSL_FLAG(double, r755_search_loss_coef, -1.0,
          "If >= 0.0 when r755_hybrid_mode=true, overrides FLAGS_search_loss_coef without changing config_fingerprint.");
ABSL_FLAG(std::string, r755_crn_mode, "sequential",
          "CRN mode for r755_hybrid_mode rollouts: 'sequential', 'deck_order', or 'event_keyed'.");
ABSL_FLAG(bool, r755_reuse_root_prior, false,
          "If true, pass collector's already-evaluated root legal probabilities into CompoundRolloutSearchDecision to skip duplicate root Prior() GPU call.");
ABSL_FLAG(bool, r755_dedup_paired_worlds, false,
          "If true in kPaired mode, determinize each world seed once via ResampleFromInfostate and Clone() across candidates.");
ABSL_FLAG(double, r755_max_raw_prior, 1.0,
          "If < 1.0, skip r755 search when the root top-1 legal probability >= r755_max_raw_prior.");
ABSL_FLAG(int, r755_stage1_rollouts, 0,
          "If > 0 and < rollouts_per_action in kPaired mode, run 2-stage sequential halving with Stage 1 of this many paired worlds.");
ABSL_FLAG(double, r755_stage1_min_mean_diff, -1e9,
          "In 2-stage paired search, stop after Stage 1 (keep raw_action) if the best challenger's Stage-1 mean paired utility diff (u_cand - u_raw) < this threshold.");
ABSL_FLAG(std::string, r755_root_dump_jsonl, "",
          "If non-empty, append one JSON line per searched r755 root with category, priors, per-world candidate utilities, nn_evals, and override status.");

// R7-55 Phase 3 / Cat 3 & Stage 1 rollout flags.
ABSL_FLAG(bool, r755_cat3_strict_card_tech_only, true,
          "Strictly restrict Cat 3 (kRevealPurchase) candidate enumeration to card and tech acquisition actions.");
ABSL_FLAG(double, r755_cat3_search_prob, -1.0,
          "Per-turn search probability for Cat 3 (if < 0, falls back to r755_search_prob).");
ABSL_FLAG(double, r755_cat3_low_prior_boost_prob, -1.0,
          "If >= 0, boost Cat 3 search probability when any legal purchase has prior < r755_cat3_low_prior_threshold.");
ABSL_FLAG(double, r755_cat3_low_prior_threshold, 0.02,
          "Prior threshold below which Cat 3 search probability is boosted.");
ABSL_FLAG(std::string, r755_cat3_truncation_mode, "next_round",
          "Truncation mode for Cat 3 rollouts (e.g. 'next_round', 'none').");
ABSL_FLAG(int, r755_cat3_stage1_rollouts, 8,
          "Stage 1 rollout count for Cat 3 2-stage search.");
ABSL_FLAG(int, r755_cat3_stage2_indep_rollouts, 24,
          "Stage 2 independent rollout count for Cat 3 confirmed candidates.");
ABSL_FLAG(double, r755_cat3_mpo_temperature, 0.02,
          "Base temperature tau for Cat 3 MPO relative target computation.");
ABSL_FLAG(double, r755_cat3_mpo_max_kl, 0.05,
          "Max KL epsilon for Cat 3 MPO relative target computation.");
ABSL_FLAG(std::string, r755_cat3_crn_mode, "",
          "CRN mode override for Cat 3 rollouts (if non-empty, overrides r755_crn_mode).");
ABSL_FLAG(bool, r755_cat3_symmetric_confirmation, false,
          "Require symmetric confirmation across candidates in Cat 3 Stage 2.");
ABSL_FLAG(int, r755_stage1_subwave_rollouts, 0,
          "Number of rollouts in Stage 1 subwave before early exit check.");
ABSL_FLAG(double, r755_stage1_subwave_min_diff, -0.05,
          "Minimum difference for Stage 1 subwave early exit.");
ABSL_FLAG(bool, r755_leaf_value_centering, false,
          "Center 4-seat leaf values across players in rollout search.");
ABSL_FLAG(bool, r755_finish_unvisited_aux_on_kl_stop, false,
          "When true, if PPO KL early stop triggers before all auxiliary search "
          "examples are visited, run a finish pass over unvisited auxiliary "
          "examples paired with EMA magnet anchor transitions.");

// R7-65 joint agent search & absorption flags.
ABSL_FLAG(std::string, r765_aux_norm_mode, "sum_over_mb",
          "Auxiliary hybrid loss scaling mode: 'sum_over_mb' (default) or 'per_row_capped'.");
ABSL_FLAG(double, r765_aux_max_slice_coef, 0.24,
          "Maximum total slice weight cap when r765_aux_norm_mode='per_row_capped'.");
ABSL_FLAG(bool, r765_joint_agent_search, false,
          "When true in r755_hybrid_mode, use SearchV2JointPlacementDecision with Stage-2-confirmed follow-up targets on Cat 1 agent root turns.");
ABSL_FLAG(double, r765_agent_search_prob, -1.0,
          "Per-turn search probability for Cat 1 agent turns when >= 0.0 (overrides r755_search_prob on Cat 1).");
ABSL_FLAG(double, r765_agent_max_raw_prior, -1.0,
          "When >= 0.0, max raw prior gate for Cat 1 and Cat 2 agent-turn search (e.g. 0.95), keeping Cat 3 on r755_max_raw_prior.");
ABSL_FLAG(bool, r765_seq_v1_agent_search, false,
          "When true in r755_hybrid_mode, run unclocked Search v1 SuccessiveHalvingRolloutSearchDecision (top_k=5) sequentially on Cat 1 and Cat 2 agent-turn decisions.");
ABSL_FLAG(int, r765_agent_ref_rollouts, 192,
          "Unclocked reference_rollouts_per_action (total_budget = 3 * ref_rollouts) for r765 joint agent search.");
ABSL_FLAG(double, r765_mpo_temperature, 0.15,
          "Base temperature tau for pairwise MPO targets on tactical overrides.");
ABSL_FLAG(double, r765_mpo_max_kl, 0.05,
          "Max KL cap for pairwise MPO targets on tactical overrides.");
ABSL_FLAG(std::string, r765_absorption_diag_jsonl, "",
          "If non-empty, write per-update T0/T1/T2/T3 passive preference & MPO constraint diagnostics JSONL.");



