#ifndef OPEN_SPIEL_EXAMPLES_DUNE_PPO_TRAINING_UTILS_H_
#define OPEN_SPIEL_EXAMPLES_DUNE_PPO_TRAINING_UTILS_H_

#include <algorithm>
#include <vector>
#include <string>
#include <memory>
#include <cstring>
#include <cstdint>
#include <deque>
#include <functional>
#include <iosfwd>
#include <map>
#include <random>

#include "open_spiel/abseil-cpp/absl/strings/string_view.h"
#include "open_spiel/spiel.h"       // Action, ActionsAndProbs
#include "open_spiel/utils/json.h"  // json::Object for manifest online-collection state
#include "dune_specimen_conversion.h"  // the single 741-752 conversion predicate
#include "dune_search_routing.h"



#ifdef OPEN_SPIEL_BUILD_WITH_LIBTORCH
#include <torch/torch.h>
#include <c10/util/Half.h>
#include <c10/core/Event.h>
#include <c10/core/impl/VirtualGuardImpl.h>
#include <chrono>
#include <optional>
#include "dune_network.h"
#include "dune_online_search_collector.h"  // SearchTrainingExample (18B combined opt)
#include "open_spiel/abseil-cpp/absl/flags/declare.h"
#endif

ABSL_DECLARE_FLAG(bool, rollout_amp);
ABSL_DECLARE_FLAG(bool, aux_full_batch_presentation);
ABSL_DECLARE_FLAG(double, aux_value_coef);
ABSL_DECLARE_FLAG(bool, search_aux_decoupled);
ABSL_DECLARE_FLAG(int, search_aux_steps);
ABSL_DECLARE_FLAG(int, search_aux_epochs);
ABSL_DECLARE_FLAG(int, search_aux_batch_size);
ABSL_DECLARE_FLAG(double, search_aux_target_kl);
ABSL_DECLARE_FLAG(double, search_target_max_kl);
ABSL_DECLARE_FLAG(bool, critic_keeps_training);
ABSL_DECLARE_FLAG(bool, privileged_critic);
ABSL_DECLARE_FLAG(bool, purchase_exploration);
ABSL_DECLARE_FLAG(bool, random_leader_draft);
ABSL_DECLARE_FLAG(double, logit_penalty_coef);
ABSL_DECLARE_FLAG(bool, search_aux_anchoring);
ABSL_DECLARE_FLAG(double, search_aux_beta);
ABSL_DECLARE_FLAG(double, search_aux_alpha);
ABSL_DECLARE_FLAG(double, search_aux_rail_kl_target);
ABSL_DECLARE_FLAG(bool, search_research_diagnostic);
ABSL_DECLARE_FLAG(bool, use_legacy_unoptimized_reads);
ABSL_DECLARE_FLAG(bool, use_wave1_ppo_update);
ABSL_DECLARE_FLAG(bool, use_grad_norm_reuse);
ABSL_DECLARE_FLAG(bool, cache_ema_log_probs);
ABSL_DECLARE_FLAG(int, ppo_microbatch_size);

extern std::string g_reward_transition_source_fingerprint;
extern int64_t g_reward_transition_source_update;
extern std::string g_gae_transition_source_fingerprint;
extern int64_t g_gae_transition_source_update;
extern std::string g_learning_rate_transition_source_fingerprint;
extern int64_t g_learning_rate_transition_source_update;
extern std::string g_target_kl_transition_source_fingerprint;
extern int64_t g_target_kl_transition_source_update;
extern std::string g_single_learner_transition_source_fingerprint;
extern int64_t g_single_learner_transition_source_update;
extern std::string g_ema_magnet_transition_source_fingerprint;
extern int64_t g_ema_magnet_transition_source_update;
extern std::string g_pool_transition_source_fingerprint;
extern int64_t g_pool_transition_source_update;
extern std::string g_market_appendix_transition_source_fingerprint;
extern int64_t g_market_appendix_transition_source_update;
extern std::string g_deck_pool_semantics_transition_source_fingerprint;
extern int64_t g_deck_pool_semantics_transition_source_update;
extern std::string g_tf32_transition_source_fingerprint;
extern int64_t g_tf32_transition_source_update;
extern std::string g_semantic_schema_transition_source_fingerprint;
extern int64_t g_semantic_schema_transition_source_update;

#include "dune_ppo_training_types.h"
#include "dune_ppo_training_replay.h"
#include "dune_ppo_training_checkpoint.h"

#endif // OPEN_SPIEL_EXAMPLES_DUNE_PPO_TRAINING_UTILS_H_
