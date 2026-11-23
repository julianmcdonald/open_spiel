#include "dune_ppo_training_utils.h"
#include "dune_sha256.h"
#include "dune_search_routing.h"  // DuneDecisionRole, for the PWO-5 canary role order
#include "dune_targeted_buy.h"
#include "open_spiel/abseil-cpp/absl/strings/str_format.h"
#include "open_spiel/utils/json.h"
#include "open_spiel/games/dune_imperium/dune_imperium.h"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>

#ifdef OPEN_SPIEL_BUILD_WITH_LIBTORCH
#include <ATen/Parallel.h>
#include <ATen/ops/_fused_adamw.h>
#include <ATen/ops/_foreach_norm.h>
#include <ATen/ops/_foreach_mul.h>
#include "open_spiel/abseil-cpp/absl/flags/declare.h"
#include "open_spiel/abseil-cpp/absl/flags/flag.h"
#include "dune_seed_utils.h"
#include <c10/core/Event.h>
#include <c10/core/impl/VirtualGuardImpl.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <atomic>
#include <chrono>
#include <mutex>
#include <optional>
#include <unordered_set>
#include <cmath>


#include "dune_ppo_training_flags.h"
ABSL_FLAG(double, policy_kl_anchor_coeff, 0.0, "Coefficient for policy KL anchor penalty to prevent policy drift when unfreezing trunk in value-only training");

std::string g_reward_transition_source_fingerprint = "";
int64_t g_reward_transition_source_update = -1;
std::string g_gae_transition_source_fingerprint = "";
int64_t g_gae_transition_source_update = -1;
std::string g_learning_rate_transition_source_fingerprint = "";
int64_t g_learning_rate_transition_source_update = -1;
std::string g_target_kl_transition_source_fingerprint = "";
int64_t g_target_kl_transition_source_update = -1;
std::string g_single_learner_transition_source_fingerprint = "";
int64_t g_single_learner_transition_source_update = -1;
std::string g_ema_magnet_transition_source_fingerprint = "";
int64_t g_ema_magnet_transition_source_update = -1;
std::string g_pool_transition_source_fingerprint = "";
int64_t g_pool_transition_source_update = -1;
std::string g_market_appendix_transition_source_fingerprint = "";
int64_t g_market_appendix_transition_source_update = -1;
std::string g_deck_pool_semantics_transition_source_fingerprint = "";
int64_t g_deck_pool_semantics_transition_source_update = -1;
std::string g_tf32_transition_source_fingerprint = "";
int64_t g_tf32_transition_source_update = -1;
std::string g_semantic_schema_transition_source_fingerprint = "";
int64_t g_semantic_schema_transition_source_update = -1;

ABSL_FLAG(bool, train_value_only, false, "Train only the value head parameters.");
ABSL_FLAG(bool, rollout_amp, true, "Use CUDA BF16 autocast during rollout generation.");
ABSL_FLAG(bool, verify_packing_alignment, false, "Verify packing alignment at the minibatch staging boundary.");
ABSL_FLAG(double, aux_value_coef, -1.0, "Coefficient for auxiliary value regression loss. Default -1.0 uses PPO value_coef; set 0.0 to disable auxiliary search-value regression.");
ABSL_FLAG(bool, aux_full_batch_presentation, false, "When true, presents all search auxiliary examples at every PPO minibatch step rather than slicing across minibatches.");
ABSL_FLAG(bool, search_aux_decoupled, false, "When true, runs search auxiliary supervision as a standalone, bounded phase after PPO rather than per-minibatch.");
ABSL_FLAG(int, search_aux_steps, 4, "Number of auxiliary optimizer steps in the decoupled phase (0 = full pass over labels, default 4).");
ABSL_FLAG(int, search_aux_epochs, 1, "Number of full epochs over auxiliary search examples in the decoupled phase.");
ABSL_FLAG(int, search_aux_batch_size, 64, "Minibatch size for decoupled auxiliary optimizer steps.");
ABSL_FLAG(double, search_aux_target_kl, 0.01, "Cumulative KL ceiling from post-PPO reference policy; steps exceeding this are rejected.");
ABSL_FLAG(double, search_target_max_kl, 0.05, "Maximum KL ceiling for MPO target distribution.");
ABSL_FLAG(bool, critic_keeps_training, false,
          "In separate actor-critic PPO, continue training critic after actor KL early stop.");
ABSL_FLAG(bool, privileged_critic, false,
          "Feed privileged opponent features to central critic in separate actor-critic.");
ABSL_FLAG(bool, purchase_exploration, false,
          "Explore reveal purchases in selected training games.");
ABSL_FLAG(bool, random_leader_draft, true,
          "Uniformly sample legal leaders during PPO training, retaining the "
          "policy log probability for outcome-based draft learning. "
          "This is biased exploration; evaluation and play are unaffected.");
ABSL_FLAG(double, logit_penalty_coef, 0.0,
          "Coefficient for precap logit penalty on nontrivial decisions.");
ABSL_FLAG(bool, search_aux_anchoring, false,
          "When true, enables anchored auxiliary search supervision with anchor pool and rail rejection.");
ABSL_FLAG(double, search_aux_beta, 1.0,
          "Coefficient for policy KL anchor penalty in auxiliary loss.");
ABSL_FLAG(double, search_aux_alpha, 1.0,
          "Coefficient for value anchor MSE penalty in auxiliary loss.");
ABSL_FLAG(double, search_aux_rail_kl_target, 0.001,
          "Maximum allowable mean KL on rail set before rejecting step and ending aux phase.");
ABSL_FLAG(bool, search_research_diagnostic, false,
          "When true, runs diagnostic re-search on up to 16 override labels.");
// --- WO-PERF-1 (2026-08-14): diagnostics-only performance flags. Both
// defaults reproduce today's behavior bit-for-bit; neither statistic has any
// consumer outside diagnostics (PPO Phase-0 findings, 2026-07-29).
ABSL_FLAG(std::string, diag_prepass_mode, "full",
          "Diagnostics overhead mode. 'full' (default): explicit full-verification "
          "mode, running full-rollout forward passes and SHA-256 rollout hashing "
          "every update. 'cadenced': lightweight per-update monitoring; reads critic "
          "saturation from rollout values, marks skipped rollout_hash as 'unmeasured' "
          "and policy_kl_before as 0.0 with measured_transitions=-1 sentinel, "
          "and runs expensive hashes and forward passes only on startup/resume and "
          "every --diag_prepass_interval updates.");
ABSL_FLAG(int, diag_prepass_interval, 25,
          "With --diag_prepass_mode=cadenced, measure policy_kl_before and "
          "rollout_hash on updates where global_update %% this interval == 0 "
          "(absolute, like --checkpoint_interval), plus always on the first update "
          "of the process. <= 0 means startup/resume only.");
ABSL_FLAG(std::string, grad_telemetry_mode, "per_param",
          "Per-module gradient-norm telemetry. 'per_param' (default): today's "
          "behavior, one device-to-host .item() read per parameter tensor per "
          "minibatch (70 reads/minibatch at production size). 'accumulated': "
          "the six group sums stay on-device in float64 and are read back "
          "once per update; values are identical on CPU by construction.");
// --- WO-PERF-TIMING (2026-08-16): per-phase PPO wall attribution. -----------
// Declared here, beside the WO-PERF-1 flags, so it registers in every binary
// linking this TU -- the same pattern for the same reason.
ABSL_FLAG(std::string, phase_timing_mode, "off",
          "off|phases -- per-phase PPO wall attribution. Default 'off' is "
          "bit-for-bit today's behavior: no timers armed, no CUDA events "
          "created, no clock read, no sidecar written. 'phases' brackets the "
          "eight PPO phases with CUDA events on the compute stream (read once "
          "after a single end-of-update synchronise) and writes a "
          "phase_timing.jsonl sidecar beside --diagnostics_path. The mode "
          "perturbs what it measures; the sidecar reports its own sync cost, "
          "its event count and the trainer's PPO figure so the discrepancy is "
          "visible. NOT part of ComputeConfigFingerprint(), so a checkpoint "
          "manifest cannot detect a mid-run flip: every timed arm must be a "
          "FRESH run, never a resume. Rejected with --pipeline=true, whose "
          "background collection thread shares the compute stream.");

ABSL_FLAG(bool, advantage_filtering, false,
          "When true, standardizes advantages across multi-choice rollout rows, "
          "retains top quantile by |std_adv|, and skips minibatch renormalization.");
ABSL_FLAG(double, advantage_filtering_quantile, 0.25,
          "Quantile cutoff for advantage filtering (default 0.25).");
ABSL_FLAG(bool, enable_study_readouts, false,
          "When true, logs unfiltered KL diagnostic and single-action value error readouts.");
ABSL_FLAG(bool, use_legacy_unoptimized_reads, false,
          "Internal test flag to exercise legacy sync/upload path in TrainPpoUpdate.");
ABSL_FLAG(bool, use_wave1_ppo_update, false,
          "Internal test flag to exercise Wave 1 PPO update (pinned double-buffering, packed_scalars, ema_logits_cache, but legacy clip/AdamW/softmax/zeros_like).");
ABSL_FLAG(bool, use_grad_norm_reuse, true,
          "When true, reuses pre-clip _foreach_norm across grad_telemetry and grad_clip and keeps fused AdamW step buffers on device.");
ABSL_FLAG(bool, cache_ema_log_probs, true,
          "When true, caches EMA log-probs across PPO epochs on device; set false on 24-block runs to save ~1.5 GiB VRAM.");
ABSL_FLAG(int, ppo_microbatch_size, 0,
          "When > 0, splits PPO forward+backward inside each minibatch (and "
          "KL-stop EMA anchor pass) into microbatches of at most this many rows "
          "with exact full-minibatch denominator weighting, scoping EMA forward "
          "autocast before learner forward to save ~2.3 GiB VRAM on 24-block runs "
          "without changing ppo_minibatch_size or config_fingerprint.");
#endif

namespace open_spiel {

#ifdef OPEN_SPIEL_BUILD_WITH_LIBTORCH
PolicyDistributionSample SampleTrainingPolicyDistribution(
    const State& state, std::mt19937_64* rng,
    const std::vector<float>& logits, const std::vector<Action>& legal_actions,
    std::vector<double>* legal_probabilities) {
  auto sample = SamplePolicyDistribution(rng, logits, legal_actions,
                                        legal_probabilities);
  const auto* dune_state =
      dynamic_cast<const dune_imperium::DuneImperiumState*>(&state);
  if (absl::GetFlag(FLAGS_random_leader_draft) && dune_state != nullptr &&
      dune_state->phase() == dune_imperium::GamePhase::kLeaderDraft) {
    std::uniform_int_distribution<size_t> pick(0, legal_actions.size() - 1);
    sample.chosen_index = pick(*rng);
    sample.action = legal_actions[sample.chosen_index];
    sample.leader_draft_overridden = true;
    // Match forced-buy exploration: learn preferences from outcomes without
    // training the draft policy toward the uniform behavior distribution.
    sample.chosen_log_probability = static_cast<float>(
        std::log((*legal_probabilities)[sample.chosen_index]));
  }
  return sample;
}
#endif

// PWO-5 gate 2 item (b). The term is SUBTRACTED, so a positive penalty lowers
// the reward. The range predicate is the shared 741-752 one, never an inline
// `base .. base + 12`. See dune_specimen_conversion.h for why 740 is excluded.
// `reward_lambda` is a FLOAT at the one call site, and the arithmetic below is
// written in float for that reason: the original in-line statement was
// `reward -= static_cast<float>(penalty) * reward_lambda`, all-float, and a
// double intermediate here would be a last-ulp behaviour change smuggled in
// under a refactor.
float ApplySpecimenExchangeShaping(float reward, Action action,
                                   double specimen_exchange_penalty,
                                   float reward_lambda) {
  if (specimen_exchange_penalty == 0.0) return reward;
  if (!dune_shaping::IsSpecimenConversionAction(action)) return reward;
  return reward - static_cast<float>(specimen_exchange_penalty) * reward_lambda;
}

float ApplyFamilyAtomicsShaping(float reward, Action action,
                                bool is_own_reveal_turn, int remaining_persuasion,
                                double family_atomics_penalty,
                                float reward_lambda) {
  if (family_atomics_penalty == 0.0) return reward;
  if (action != dune_imperium::kActionFamilyAtomics) return reward;
  if (is_own_reveal_turn && remaining_persuasion > 0) return reward;
  return reward - static_cast<float>(family_atomics_penalty) * reward_lambda;
}

float ApplyPlotIntrigueShaping(float reward, Action action,
                               int intrigue_hand_size_before_play,
                               int exemption_threshold,
                               double plot_intrigue_penalty,
                               float reward_lambda) {
  if (plot_intrigue_penalty == 0.0) return reward;
  if (!IsPlotIntrigueAction(action)) return reward;
  if (intrigue_hand_size_before_play >= exemption_threshold) return reward;
  return reward - static_cast<float>(plot_intrigue_penalty) * reward_lambda;
}

DuneDecisionRole ClassifyTransitionDecisionRole(const PpoTransition& t) {
  if (t.legal_actions.size() <= 1) {
    return DuneDecisionRole::kForcedOrBookkeeping;
  }
  if (t.decision_role >= 1 && t.decision_role <= 6) {
    return static_cast<DuneDecisionRole>(t.decision_role);
  }

  // 1. Leader selection
  for (Action a : t.legal_actions) {
    if (a >= dune_imperium::kActionLeaderPick0 &&
        a < dune_imperium::kActionBuyHelenaReserve0) {
      return DuneDecisionRole::kLeaderSelection;
    }
  }

  // 2. Purchase
  bool has_purchase = false;
  bool all_purchase_or_end_turn = true;
  for (Action a : t.legal_actions) {
    bool is_purchase_action =
        (a >= dune_imperium::kActionBuyImperiumRow0 &&
         a < dune_imperium::kActionBuyImperiumRow0 + 8) ||
        (a >= dune_imperium::kActionBuyHelenaReserve0 &&
         a < dune_imperium::kActionBuyHelenaReserve0 + 8) ||
        (a == dune_imperium::kActionBuyReserveArrakisLiaison) ||
        (a == dune_imperium::kActionBuyReserveTheSpiceMustFlow) ||
        (a >= dune_imperium::kActionTechAcquire0 &&
         a <= dune_imperium::kActionTechAcquireSkip) ||
        (a >= dune_imperium::kActionTleilaxuAcquire0 &&
         a <= dune_imperium::kActionTleilaxuAcquireSkip) ||
        (a >= dune_imperium::kActionTechAcquireWithSolari0 &&
         a <= dune_imperium::kActionTechAcquireWithSolariSkip);
    if (is_purchase_action) {
      has_purchase = true;
    }
    if (!is_purchase_action && a != dune_imperium::kActionEndTurn) {
      all_purchase_or_end_turn = false;
      break;
    }
  }
  if (has_purchase && all_purchase_or_end_turn) {
    return DuneDecisionRole::kPurchase;
  }

  // 3. CombatIntrigue
  for (Action a : t.legal_actions) {
    if (a == dune_imperium::kActionCombatPass ||
        (a >= dune_imperium::kActionPlayIntrigueCombatCard0 &&
         a < dune_imperium::kActionPlayIntrigueCombatCard0 + dune_imperium::kMaxIntrigueCards) ||
        (a >= dune_imperium::kActionCombatCommit0 &&
         a < dune_imperium::kActionCombatCommit0 + 48)) {
      return DuneDecisionRole::kCombatIntrigue;
    }
  }

  // 4. AgentPrimary vs AgentContinuation
  bool has_agent_card_play = false;
  bool has_space_place = false;
  for (Action a : t.legal_actions) {
    if ((a >= dune_imperium::kActionSelectAgentCard0 &&
         a < dune_imperium::kActionSelectAgentCard0 + 256) ||
        a == dune_imperium::kActionPlayKwisatzHaderach) {
      has_agent_card_play = true;
    }
    if (a >= dune_imperium::kActionAgentSpaceConspire &&
        a <= dune_imperium::kActionAgentSpaceSietchTabr) {
      has_space_place = true;
    }
  }
  if (has_agent_card_play && !has_space_place) {
    return DuneDecisionRole::kAgentPrimary;
  }
  if (has_space_place) {
    return DuneDecisionRole::kAgentContinuation;
  }

  // 5. OtherOptional
  return DuneDecisionRole::kOtherOptional;
}

void StandardizeAdvantagesPhaseStratified(
    std::vector<PpoTransition>* batch,
    const std::vector<int64_t>& mc_indices,
    int64_t min_stratum_count) {
  if (batch == nullptr || mc_indices.empty()) {
    return;
  }

  // Compute global adv_mean and adv_std as fallback.
  double global_sum = 0.0;
  for (int64_t idx : mc_indices) {
    global_sum += (*batch)[idx].advantage;
  }
  const double global_mean = global_sum / mc_indices.size();
  double global_sq_sum = 0.0;
  for (int64_t idx : mc_indices) {
    double diff = (*batch)[idx].advantage - global_mean;
    global_sq_sum += diff * diff;
  }
  const double global_std = std::sqrt(global_sq_sum / mc_indices.size());
  const double global_denom = global_std + 1e-8;

  // Group mc_indices into strata 0..6.
  std::array<std::vector<int64_t>, 7> strata;
  for (int64_t idx : mc_indices) {
    DuneDecisionRole role = ClassifyTransitionDecisionRole((*batch)[idx]);
    int s = static_cast<int>(role);
    if (s >= 0 && s < 7) {
      strata[s].push_back(idx);
    } else {
      strata[static_cast<int>(DuneDecisionRole::kOtherOptional)].push_back(idx);
    }
  }

  // For each stratum s with count >= min_stratum_count and std_s > 1e-8,
  // standardize: (*batch)[idx].advantage = (adv - mean_s) / (std_s + 1e-8).
  // Otherwise use: (adv - global_mean) / (global_std + 1e-8).
  for (int s = 0; s < 7; ++s) {
    const auto& s_indices = strata[s];
    if (s_indices.empty()) continue;
    const int64_t count = static_cast<int64_t>(s_indices.size());
    if (count >= min_stratum_count) {
      double s_sum = 0.0;
      for (int64_t idx : s_indices) {
        s_sum += (*batch)[idx].advantage;
      }
      const double s_mean = s_sum / count;
      double s_sq_sum = 0.0;
      for (int64_t idx : s_indices) {
        double diff = (*batch)[idx].advantage - s_mean;
        s_sq_sum += diff * diff;
      }
      const double s_std = std::sqrt(s_sq_sum / count);
      if (s_std > 1e-8) {
        const double s_denom = s_std + 1e-8;
        for (int64_t idx : s_indices) {
          (*batch)[idx].advantage = static_cast<float>(((*batch)[idx].advantage - s_mean) / s_denom);
        }
        continue;
      }
    }
    // Fallback to global standardization.
    for (int64_t idx : s_indices) {
      (*batch)[idx].advantage = static_cast<float>(((*batch)[idx].advantage - global_mean) / global_denom);
    }
  }
}

std::string ComputeRolloutHash(const std::vector<PpoTransition>& batch) {
  open_spiel::SHA256 hasher;
  for (const auto& t : batch) {
    hasher.Update(&t.episode_id, sizeof(t.episode_id));
    hasher.Update(&t.player_id, sizeof(t.player_id));
    hasher.Update(&t.action, sizeof(t.action));
    hasher.Update(&t.old_log_prob, sizeof(t.old_log_prob));
    hasher.Update(&t.advantage, sizeof(t.advantage));
    hasher.Update(&t.return_value, sizeof(t.return_value));
    hasher.Update(&t.value, sizeof(t.value));
    hasher.Update(t.state.data(), t.state.size() * sizeof(float));
    for (Action a : t.legal_actions) {
      hasher.Update(&a, sizeof(a));
    }
  }
  return hasher.Final();
}

std::string ComputeRolloutSemanticDigest(const std::vector<PpoTransition>& batch) {
  open_spiel::SHA256 hasher;
  for (const auto& t : batch) {
    hasher.Update(&t.episode_id, sizeof(t.episode_id));
    hasher.Update(&t.player_id, sizeof(t.player_id));
    hasher.Update(&t.action, sizeof(t.action));
    hasher.Update(&t.old_log_prob, sizeof(t.old_log_prob));
    hasher.Update(&t.advantage, sizeof(t.advantage));
    hasher.Update(&t.return_value, sizeof(t.return_value));
    hasher.Update(&t.value, sizeof(t.value));
    hasher.Update(t.state.data(), t.state.size() * sizeof(float));
    for (Action a : t.legal_actions) {
      hasher.Update(&a, sizeof(a));
    }
    const auto& c = t.candidate_data;
    const uint64_t num_cands = c.actions.size();
    hasher.Update(&num_cands, sizeof(num_cands));
    if (num_cands > 0) {
      hasher.Update(c.actions.data(), c.actions.size() * sizeof(Action));
      hasher.Update(c.features.data(), c.features.size() * sizeof(float));
      hasher.Update(c.card_ids.data(), c.card_ids.size() * sizeof(int64_t));
      hasher.Update(c.space_ids.data(), c.space_ids.size() * sizeof(int64_t));
      hasher.Update(c.supported.data(), c.supported.size() * sizeof(uint8_t));
      hasher.Update(c.roles.data(), c.roles.size() * sizeof(dune_semantic::ActionRole));
    }
  }
  return hasher.Final();
}

bool VerifyRolloutCandidateAlignment(
    const std::vector<PpoTransition>& batch,
    std::string* error_msg) {
  for (size_t i = 0; i < batch.size(); ++i) {
    const auto& t = batch[i];
    const auto& c = t.candidate_data;
    if (c.actions.size() != t.legal_actions.size()) {
      if (error_msg) {
        *error_msg = absl::StrFormat(
            "Row %zu: candidate action count %zu != legal action count %zu",
            i, c.actions.size(), t.legal_actions.size());
      }
      return false;
    }
    for (size_t j = 0; j < c.actions.size(); ++j) {
      if (c.actions[j] != t.legal_actions[j]) {
        if (error_msg) {
          *error_msg = absl::StrFormat(
              "Row %zu candidate %zu: action mismatch (%d != %d)",
              i, j, static_cast<int>(c.actions[j]), static_cast<int>(t.legal_actions[j]));
        }
        return false;
      }
    }
    const size_t expected_feats = c.actions.size() * dune_semantic::kSemanticFeatDim;
    if (c.features.size() != expected_feats ||
        c.card_ids.size() != c.actions.size() ||
        c.space_ids.size() != c.actions.size() ||
        c.supported.size() != c.actions.size() ||
        c.roles.size() != c.actions.size()) {
      if (error_msg) {
        *error_msg = absl::StrFormat(
            "Row %zu: candidate descriptor vector dimension mismatch", i);
      }
      return false;
    }
  }
  return true;
}

bool VerifyPackedMinibatchAlignment(
    const std::vector<PpoTransition>& batch,
    const int64_t* mb_indices,
    int64_t start_seq,
    int64_t mb_len,
    int64_t obs_size,
    int64_t action_dim,
    const float* staging_states_ptr,
    const bool* staging_masks_ptr,
    const int64_t* staging_actions_ptr,
    const std::vector<const dune_semantic::CandidateActionData*>& mb_cands,
    std::string* error_msg) {
  if (static_cast<int64_t>(mb_cands.size()) != mb_len) {
    if (error_msg) {
      *error_msg = absl::StrFormat(
          "Minibatch candidate vector size %zu != mb_len %ld",
          mb_cands.size(), mb_len);
    }
    return false;
  }
  for (int64_t i = 0; i < mb_len; ++i) {
    int64_t idx = mb_indices ? mb_indices[i] : (start_seq + i);
    const PpoTransition& t = batch[idx];
    if (std::memcmp(staging_states_ptr + i * obs_size, t.state.data(),
                    obs_size * sizeof(float)) != 0) {
      if (error_msg) {
        *error_msg = absl::StrFormat("Minibatch row %ld state mismatch with batch[%ld]", i, idx);
      }
      return false;
    }
    if (staging_actions_ptr[i] != t.action) {
      if (error_msg) {
        *error_msg = absl::StrFormat("Minibatch row %ld action mismatch (%ld != %ld)",
                                     i, staging_actions_ptr[i], t.action);
      }
      return false;
    }
    for (Action a = 0; a < action_dim; ++a) {
      bool expected_legal = std::find(t.legal_actions.begin(), t.legal_actions.end(), a) != t.legal_actions.end();
      if (staging_masks_ptr[i * action_dim + a] != expected_legal) {
        if (error_msg) {
          *error_msg = absl::StrFormat("Minibatch row %ld mask mismatch for action %d", i, static_cast<int>(a));
        }
        return false;
      }
    }
    if (mb_cands[i] != &t.candidate_data) {
      if (error_msg) {
        *error_msg = absl::StrFormat(
            "Minibatch row %ld candidate pointer mismatch (%p != %p)",
            i, static_cast<const void*>(mb_cands[i]), static_cast<const void*>(&t.candidate_data));
      }
      return false;
    }
    if (mb_cands[i]->actions != t.legal_actions) {
      if (error_msg) {
        *error_msg = absl::StrFormat("Minibatch row %ld candidate actions mismatch with legal_actions", i);
      }
      return false;
    }
  }
  return true;
}

#ifdef OPEN_SPIEL_BUILD_WITH_LIBTORCH
float ComputeRewardLambda(uint64_t env_steps, uint64_t start_steps, uint64_t decay_steps) {
  if (decay_steps == 0) {
    return 1.0f;
  }
  if (env_steps < start_steps) {
    return 1.0f;
  }
  double elapsed = static_cast<double>(env_steps - start_steps);
  double lambda = std::clamp(1.0 - elapsed / static_cast<double>(decay_steps), 0.0, 1.0);
  return static_cast<float>(lambda);
}

torch::Tensor LegalLogitMean(const torch::Tensor& logits,
                             const torch::Tensor& legal_mask) {
  torch::Tensor logits_f32 = logits.to(torch::kFloat32);
  torch::Tensor legal_counts =
      legal_mask.to(torch::kFloat32).sum(-1, /*keepdim=*/true).clamp_min(1.0f);
  torch::Tensor legal_sum =
      torch::where(legal_mask, logits_f32, torch::zeros_like(logits_f32))
          .sum(-1, /*keepdim=*/true);
  return legal_sum / legal_counts;
}

torch::Tensor ApplyLogitCapTensor(const torch::Tensor& logits,
                                  float logit_cap) {
  torch::Tensor logits_f32 = logits.to(torch::kFloat32);
  if (logit_cap <= 0.0f) return logits_f32;
  return logit_cap * torch::tanh(logits_f32 / logit_cap);
}

torch::Tensor CenterAndCapLogitsTensor(const torch::Tensor& logits,
                                       const torch::Tensor& legal_mask,
                                       float logit_cap) {
  torch::Tensor logits_f32 = logits.to(torch::kFloat32);
  return ApplyLogitCapTensor(logits_f32 - LegalLogitMean(logits_f32, legal_mask),
                             logit_cap);
}

torch::Tensor ApproxKlK3Tensor(const torch::Tensor& log_ratio) {
  torch::Tensor x = log_ratio.to(torch::kFloat32);
  // Precompute exact factorials 2!..12! in double before casting 1/k! to FP32.
  double fact[13];
  fact[0] = 1.0;
  fact[1] = 1.0;
  for (int k = 2; k <= 12; ++k) {
    fact[k] = fact[k - 1] * static_cast<double>(k);
  }
  torch::Tensor poly = torch::full_like(x, static_cast<float>(1.0 / fact[12]));
  for (int k = 11; k >= 2; --k) {
    poly = static_cast<float>(1.0 / fact[k]) + x * poly;
  }
  return torch::where(x.abs() < 0.5f, x * x * poly, torch::expm1(x) - x);
}
#endif  // OPEN_SPIEL_BUILD_WITH_LIBTORCH

std::vector<double> ComputeTerminalReturns(
    const State& state,
    const std::string& reward_mode,
    double round7_speed_bonus,
    double win_speed_bonus_per_round) {
  const std::vector<double> raw_returns = state.Returns();
  const int num_players = state.NumPlayers();
  std::vector<double> terminal_returns = raw_returns;
  if (num_players <= 0 || raw_returns.empty()) {
    return terminal_returns;
  }

  const double max_raw = *std::max_element(raw_returns.begin(), raw_returns.end());
  int num_tied_first = 0;
  int sole_winner = -1;
  for (int p = 0; p < num_players; ++p) {
    if (std::abs(raw_returns[p] - max_raw) < 1e-6) {
      ++num_tied_first;
      sole_winner = p;
    }
  }
  if (num_tied_first != 1) {
    sole_winner = -1;
  }

  if (reward_mode == "first_place") {
    const double tied_first_reward =
        static_cast<double>(num_players - num_tied_first) /
        static_cast<double>(std::max(1, num_tied_first));
    for (int p = 0; p < num_players; ++p) {
      if (std::abs(raw_returns[p] - max_raw) < 1e-6) {
        terminal_returns[p] = tied_first_reward;
      } else {
        terminal_returns[p] = -1.0;
      }
    }
  } else if (reward_mode != "placement") {
    SpielFatalError("Unknown terminal_reward_mode: " + reward_mode);
  }

  if (round7_speed_bonus != 0.0 || win_speed_bonus_per_round != 0.0) {
    const auto* dune_state = dynamic_cast<const dune_imperium::DuneImperiumState*>(&state);
    // Since commit 66d73717, DuneImperiumState returns at the end of a round
    // when end_game is true BEFORE ++round_, so GetCurrentRound() at IsTerminal()
    // is already the exact ending round (e.g., 7 for Round 7, 8 for Round 8).
    const int rounds_played = (dune_state != nullptr) ? dune_state->GetCurrentRound() : 0;
    if (dune_state != nullptr && rounds_played > 0 &&
        sole_winner >= 0 && std::abs(max_raw - 2.25) < 1e-6) {
      if (round7_speed_bonus != 0.0 && rounds_played <= 7) {
        terminal_returns[sole_winner] += round7_speed_bonus;
      }
      if (win_speed_bonus_per_round != 0.0) {
        const int rounds_saved = std::clamp(9 - rounds_played, 0, 4);
        terminal_returns[sole_winner] +=
            win_speed_bonus_per_round * static_cast<double>(rounds_saved);
      }
    }
  }
  return terminal_returns;
}

}  // namespace open_spiel
