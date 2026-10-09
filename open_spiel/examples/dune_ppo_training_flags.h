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

#endif  // OPEN_SPIEL_EXAMPLES_DUNE_PPO_TRAINING_FLAGS_H_
