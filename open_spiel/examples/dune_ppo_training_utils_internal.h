#pragma once

#include <cstdint>
#include <memory>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

#include "dune_ppo_training_flags.h"
#include "dune_ppo_training_utils.h"

#ifdef OPEN_SPIEL_BUILD_WITH_LIBTORCH

ABSL_DECLARE_FLAG(double, policy_kl_anchor_coeff);
ABSL_DECLARE_FLAG(bool, train_value_only);
ABSL_DECLARE_FLAG(bool, rollout_amp);
ABSL_DECLARE_FLAG(bool, verify_packing_alignment);
ABSL_DECLARE_FLAG(double, aux_value_coef);
ABSL_DECLARE_FLAG(bool, aux_full_batch_presentation);
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
ABSL_DECLARE_FLAG(std::string, diag_prepass_mode);
ABSL_DECLARE_FLAG(int, diag_prepass_interval);
ABSL_DECLARE_FLAG(std::string, grad_telemetry_mode);
ABSL_DECLARE_FLAG(std::string, phase_timing_mode);
ABSL_DECLARE_FLAG(bool, advantage_filtering);
ABSL_DECLARE_FLAG(double, advantage_filtering_quantile);
ABSL_DECLARE_FLAG(bool, enable_study_readouts);
ABSL_DECLARE_FLAG(bool, use_legacy_unoptimized_reads);
ABSL_DECLARE_FLAG(bool, use_wave1_ppo_update);
ABSL_DECLARE_FLAG(bool, use_grad_norm_reuse);
ABSL_DECLARE_FLAG(bool, cache_ema_log_probs);
ABSL_DECLARE_FLAG(int, ppo_microbatch_size);

namespace open_spiel::ppo_utils_internal {

bool ConsumeFirstPrepassForModel(const void* model_identity);

// Staging & prepass helpers for TrainPpoUpdateSeparate.
struct SeparateStagingBuffers {
  torch::Tensor states_cpu;
  torch::Tensor appendix_cpu;
  torch::Tensor masks_cpu;
  torch::Tensor actions_cpu;
  torch::Tensor old_log_probs_cpu;
  torch::Tensor advantages_cpu;
  torch::Tensor returns_cpu;
  torch::Tensor old_values_cpu;
  torch::Tensor nontrivial_cpu;
  torch::Tensor policy_mask_cpu;
  torch::Tensor off_policy_cpu;

  float* states_ptr = nullptr;
  c10::Half* appendix_ptr = nullptr;
  bool* masks_ptr = nullptr;
  int64_t* actions_ptr = nullptr;
  float* old_log_probs_ptr = nullptr;
  float* advantages_ptr = nullptr;
  float* returns_ptr = nullptr;
  float* old_values_ptr = nullptr;
  bool* nontrivial_ptr = nullptr;
  bool* policy_mask_ptr = nullptr;
  bool* off_policy_ptr = nullptr;
};

SeparateStagingBuffers AllocateSeparateStagingBuffers(
    int64_t max_mb, int64_t obs_size, int64_t action_dim, bool is_privileged);

void RecomputeSeparateValuesAndGae(
    const std::shared_ptr<SharedDunePolicyValueNetImpl>& critic_model,
    SeparateStagingBuffers& staging,
    std::vector<PpoTransition>& batch,
    int64_t obs_size,
    torch::Device device,
    double gamma,
    double gae_lambda,
    int64_t max_mb,
    bool is_privileged,
    torch::Tensor* out_recomputed_values_cpu,
    torch::Tensor* out_returns_cpu);

void StageAndUploadSeparateMinibatch(
    const std::vector<PpoTransition>& batch,
    SeparateStagingBuffers& staging,
    const int64_t* mb_indices,
    int64_t start_seq,
    int64_t mb_len,
    int64_t obs_size,
    int64_t action_dim,
    bool is_privileged,
    bool full_ppo_fields,
    torch::Device device,
    torch::Tensor& out_states,
    torch::Tensor& out_critic_states,
    torch::Tensor& out_masks,
    torch::Tensor& out_actions,
    torch::Tensor& out_old_log_probs,
    torch::Tensor& out_nontrivial,
    torch::Tensor& out_policy_mask,
    std::vector<const dune_semantic::CandidateActionData*>* out_cands,
    torch::Tensor* out_advantages = nullptr,
    torch::Tensor* out_returns = nullptr,
    torch::Tensor* out_old_values = nullptr,
    torch::Tensor* out_off_policy = nullptr);

void ComputeSeparatePrepassDiagnostics(
    const std::shared_ptr<SharedDunePolicyValueNetImpl>& actor_model,
    const std::shared_ptr<SharedDunePolicyValueNetImpl>& critic_model,
    const std::vector<PpoTransition>& batch,
    SeparateStagingBuffers& staging,
    const torch::Tensor& recomputed_values_cpu,
    const torch::Tensor& returns_cpu,
    int64_t obs_size,
    int64_t action_dim,
    int64_t minibatch_size,
    bool is_privileged,
    float logit_cap,
    bool compute_diagnostics,
    bool prepass_cadenced,
    bool expensive_checks_now,
    torch::Device device,
    PhaseTimer* phase_timer,
    PpoUpdateStats* stats);

// Aux evaluation, snapshot/restore, and anchored batch setup helpers.
struct PackedEvaluationBatch {
  torch::Tensor states;
  torch::Tensor masks;
  std::vector<const dune_semantic::CandidateActionData*> cands;
  torch::Tensor ref_alogp;
  torch::Tensor ref_probs;
  torch::Tensor ref_values;
  double ref_mean_entropy = 0.0;
  int64_t count = 0;
};

PackedEvaluationBatch PackEvaluationBatch(
    const std::vector<const std::vector<float>*>& obs_ptrs,
    const std::vector<const std::vector<open_spiel::Action>*>& legals_ptrs,
    const std::vector<const dune_semantic::CandidateActionData*>& cand_ptrs,
    int64_t obs_size, int64_t action_dim,
    const std::shared_ptr<SharedDunePolicyValueNetImpl>& model,
    torch::Device device,
    bool compute_entropy = false,
    bool compute_values = false,
    double logit_cap = 10.0);

double EvaluateBatchKL(
    const std::shared_ptr<SharedDunePolicyValueNetImpl>& model,
    const PackedEvaluationBatch& batch,
    torch::Device device,
    double logit_cap = 10.0);

double EvaluateBatchEntropy(
    const std::shared_ptr<SharedDunePolicyValueNetImpl>& model,
    const PackedEvaluationBatch& batch,
    torch::Device device,
    double logit_cap = 10.0);

struct OptParamState {
  int64_t step = 0;
  torch::Tensor exp_avg;
  torch::Tensor exp_avg_sq;
  torch::Tensor max_exp_avg_sq;
};

std::unordered_map<void*, OptParamState> SnapshotOptState(
    torch::optim::AdamW& opt);

void RestoreOptState(
    torch::optim::AdamW& opt,
    const std::unordered_map<void*, OptParamState>& backup);

std::vector<torch::Tensor> SnapshotModel(
    const std::shared_ptr<SharedDunePolicyValueNetImpl>& m);

void RestoreModel(
    const std::shared_ptr<SharedDunePolicyValueNetImpl>& m,
    const std::vector<torch::Tensor>& backup);

struct AnchoredAuxBatches {
  PackedEvaluationBatch anchor_batch;
  PackedEvaluationBatch rail_agr_batch;
  PackedEvaluationBatch rail_roll_batch;
  PackedEvaluationBatch hs_batch;
};

AnchoredAuxBatches PrepareAnchoredAuxBatches(
    const std::shared_ptr<SharedDunePolicyValueNetImpl>& model,
    const std::vector<SearchTrainingExample>& agreement_examples,
    const std::vector<PpoTransition>& all_rollout_transitions,
    int64_t obs_size, int64_t action_dim,
    torch::Device device,
    double aux_logit_cap,
    std::mt19937_64* rng);

}  // namespace open_spiel::ppo_utils_internal

#endif  // OPEN_SPIEL_BUILD_WITH_LIBTORCH
