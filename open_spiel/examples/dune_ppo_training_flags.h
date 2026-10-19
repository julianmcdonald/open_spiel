#ifndef OPEN_SPIEL_EXAMPLES_DUNE_PPO_TRAINING_FLAGS_H_
#define OPEN_SPIEL_EXAMPLES_DUNE_PPO_TRAINING_FLAGS_H_

// Trainer flags read by dune_ppo_training_utils.cc. Defined in
// dune_ppo_training_flags.cc.

#include <cstdint>
#include <string>

#include "open_spiel/abseil-cpp/absl/flags/declare.h"

ABSL_DECLARE_FLAG(int, rollout_games);
ABSL_DECLARE_FLAG(int, hidden_dim);
ABSL_DECLARE_FLAG(int, num_blocks);
ABSL_DECLARE_FLAG(bool, nonlinear_value_head);
ABSL_DECLARE_FLAG(std::string, market_appendix_mode);
ABSL_DECLARE_FLAG(bool, allow_tf32);
ABSL_DECLARE_FLAG(bool, enable_semantic_scorer);
ABSL_DECLARE_FLAG(bool, enable_zone_encoder);
ABSL_DECLARE_FLAG(std::string, semantic_descriptor_schema);
ABSL_DECLARE_FLAG(double, specimen_exchange_penalty);
ABSL_DECLARE_FLAG(double, family_atomics_penalty);
ABSL_DECLARE_FLAG(double, plot_intrigue_penalty);
ABSL_DECLARE_FLAG(int, plot_intrigue_exemption_threshold);
ABSL_DECLARE_FLAG(uint64_t, head_init_constant);

// PPO optimization and loss hyperparameters.
ABSL_DECLARE_FLAG(int, ppo_minibatch_size);
ABSL_DECLARE_FLAG(int, ppo_update_epochs);
ABSL_DECLARE_FLAG(bool, normalize_advantages);
ABSL_DECLARE_FLAG(double, ppo_clip_epsilon);
ABSL_DECLARE_FLAG(bool, ppo_clip_value_loss);
ABSL_DECLARE_FLAG(double, entropy_coef);
ABSL_DECLARE_FLAG(double, value_coef);
ABSL_DECLARE_FLAG(double, grad_clip_norm);
ABSL_DECLARE_FLAG(double, target_kl);
ABSL_DECLARE_FLAG(double, logit_cap);
ABSL_DECLARE_FLAG(bool, train_amp);
ABSL_DECLARE_FLAG(bool, diagnostics_only);
ABSL_DECLARE_FLAG(uint64_t, shaping_start_env_steps);
ABSL_DECLARE_FLAG(uint64_t, shaping_decay_env_steps);

// 12-block H/J study flags.
ABSL_DECLARE_FLAG(bool, enable_ema_tracking);
ABSL_DECLARE_FLAG(double, ema_half_life);
ABSL_DECLARE_FLAG(double, ema_decay);
ABSL_DECLARE_FLAG(std::string, ema_model_checkpoint);
ABSL_DECLARE_FLAG(bool, advantage_filtering);
ABSL_DECLARE_FLAG(double, advantage_filtering_quantile);
ABSL_DECLARE_FLAG(bool, enable_study_readouts);

// EMAgnet forward KL regularization flags.
ABSL_DECLARE_FLAG(double, ema_magnet_kl_coef);
ABSL_DECLARE_FLAG(bool, allow_ema_magnet_transition);
ABSL_DECLARE_FLAG(double, ema_magnet_transition_source_coef);
ABSL_DECLARE_FLAG(std::string, ema_magnet_transition_source_fingerprint);

// Mixed-episode multi-learner opponent pool flags.
ABSL_DECLARE_FLAG(double, pool_game_fraction);
ABSL_DECLARE_FLAG(int, pool_opponents_per_game);
ABSL_DECLARE_FLAG(bool, allow_pool_transition);
ABSL_DECLARE_FLAG(std::string, pool_transition_source_fingerprint);

#endif  // OPEN_SPIEL_EXAMPLES_DUNE_PPO_TRAINING_FLAGS_H_

