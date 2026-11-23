#pragma once

#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "dune_ppo_training_utils_internal.h"

#ifdef OPEN_SPIEL_BUILD_WITH_LIBTORCH
#include <c10/core/Event.h>

namespace open_spiel::ppo_utils_internal {

struct FusedGroupStepCache {
  std::vector<const c10::TensorImpl*> active_keys;
  torch::Tensor step_buf_dev;
  std::vector<torch::Tensor> state_steps_views;
};

struct MinibatchAuxState {
  std::vector<torch::Tensor> aux_grad_snapshot;
  bool this_mb_has_aux = false;
  torch::Tensor t1_diag_alogp_before;
  torch::Tensor t1_diag_a_states;
  torch::Tensor t1_diag_a_masks;
  torch::Tensor t1_diag_a_masks_cpu;
  torch::Tensor t1_diag_a_targets;
  std::vector<const dune_semantic::CandidateActionData*> t1_diag_a_cands;
  std::vector<std::pair<Action, Action>> t1_diag_a_pairs;
  std::vector<int> t1_diag_a_ages;
  double this_aux_norm = 0.0;
};

struct MinibatchPpoBuffers {
  int epoch = 0;
  int64_t start = 0;
  int64_t mb_index = 0;
  int64_t mb_len = 0;
  const int64_t* mb_indices = nullptr;
  torch::Tensor permutation_device;

  torch::Tensor mb_states;
  torch::Tensor mb_masks;
  torch::Tensor mb_actions;
  torch::Tensor mb_old_log_probs;
  torch::Tensor mb_nontrivial;
  torch::Tensor mb_off_policy;
  torch::Tensor mb_advantages;
  torch::Tensor mb_returns;
  torch::Tensor mb_old_values;
  std::vector<const dune_semantic::CandidateActionData*> mb_candidate_actions;
  int64_t mb_num_nontrivial = 0;
  int64_t mb_num_onpolicy_nontrivial = 0;
  torch::Tensor mb_coll_lp;

  torch::Tensor policy_loss;
  torch::Tensor value_loss;
  torch::Tensor entropy;
  torch::Tensor approx_kl;
  torch::Tensor clip_fraction;
  torch::Tensor value_clip_frac_t;
  torch::Tensor reverse_kl_penalty_t;
  torch::Tensor ema_magnet_kl_t;
};

struct TrainPpoUpdateContext {
  // Inputs
  std::shared_ptr<SharedDunePolicyValueNetImpl> model;
  torch::optim::AdamW* optimizer = nullptr;
  std::vector<PpoTransition>* batch = nullptr;
  int64_t obs_size = 0;
  int64_t action_dim = 0;
  torch::Device device = torch::kCPU;
  uint64_t master = 0;
  int global_update = 0;
  std::shared_ptr<SharedDunePolicyValueNetImpl> anchor_model;
  const std::vector<SearchTrainingExample>* search_examples = nullptr;
  double search_loss_coef = 0.0;
  double abort_grad_norm_ratio = 0.0;
  std::shared_ptr<SharedDunePolicyValueNetImpl> collection_model;
  double reverse_kl_coef = 0.0;
  bool compute_diagnostics = true;
  std::shared_ptr<SharedDunePolicyValueNetImpl> ema_model;
  double ema_magnet_kl_coef = 0.0;

  PhaseTimer* phase_timer = nullptr;
  PpoUpdateStats stats;

  // Mode flags & hyperparameters
  bool r755_hybrid = false;
  bool phase_timing_on = false;
  bool prepass_cadenced = false;
  bool expensive_checks_now = false;
  int64_t n = 0;
  int64_t minibatch_size = 0;
  int64_t max_mb = 0;
  bool use_pinned = false;
  int update_epochs = 1;
  float clip_epsilon = 0.0f;
  bool normalize_advantages = false;
  bool clip_value_loss = false;
  float entropy_coef = 0.0f;
  float value_coef = 0.0f;
  float logit_cap = 0.0f;
  double target_kl = 0.0;

  // Prepass return/value tensors
  torch::Tensor returns_cpu;
  torch::Tensor old_values_cpu;
  float* returns_ptr = nullptr;
  float* old_values_ptr = nullptr;

  // Double-buffered CPU staging tensors & CUDA events
  torch::Tensor staging_states_cpu[2];
  torch::Tensor staging_masks_cpu[2];
  torch::Tensor staging_actions_cpu[2];
  torch::Tensor staging_old_log_probs_cpu[2];
  torch::Tensor staging_advantages_cpu[2];
  torch::Tensor staging_returns_cpu[2];
  torch::Tensor staging_old_values_cpu[2];
  torch::Tensor staging_nontrivial_cpu[2];
  torch::Tensor staging_off_policy_cpu[2];
  int staging_slot = 0;
  std::unique_ptr<c10::Event> staging_upload_done[2];
  bool staging_slot_recorded[2] = {false, false};

  // Active training batch
  std::vector<PpoTransition>* train_batch_ptr = nullptr;
  std::vector<PpoTransition> filtered_batch;
  int64_t n_train = 0;
  int64_t nontrivial_count = 0;

  // Phase 18B aux state
  bool aux_active = false;
  std::vector<size_t> purchase_param_indices;
  std::vector<size_t> action_head_param_indices;
  std::vector<uint8_t> need_snapshot_param;
  int64_t aux_n = 0;
  torch::Tensor aux_states;
  torch::Tensor aux_masks;
  torch::Tensor aux_targets;
  torch::Tensor aux_values;
  std::vector<const dune_semantic::CandidateActionData*> all_aux_cands;
  std::vector<std::pair<Action, Action>> all_aux_mpo_pairs;
  std::vector<double> all_aux_row_weights;
  std::vector<int> all_aux_ages;
  std::vector<int64_t> aux_slice_start;
  std::vector<int64_t> aux_slice_len;
  double aux_ce_sum = 0.0;
  double aux_vmse_sum = 0.0;
  int64_t aux_slice_uses = 0;
  double aux_norm_sum = 0.0;
  double ppo_only_norm_sum = 0.0;
  int64_t aux_norm_count = 0;
  std::vector<uint8_t> aux_visited_mask;

  // PPO loss loop accumulators & telemetry
  double weighted_policy_loss_sum = 0.0;
  double weighted_entropy_sum = 0.0;
  double weighted_kl_sum = 0.0;
  double weighted_clip_fraction_sum = 0.0;
  int64_t total_nontrivial_count = 0;
  int64_t total_onpolicy_nontrivial_count = 0;
  double value_loss_sum = 0.0;
  double value_clip_frac_sum = 0.0;
  bool grad_telemetry_accumulated = false;
  torch::Tensor head_norm_accum;

  // Reverse KL & EMAgnet state
  bool reverse_kl_active = false;
  torch::Tensor collection_log_probs;
  double weighted_reverse_kl_sum = 0.0;
  bool ema_magnet_active = false;
  double weighted_ema_magnet_kl_sum = 0.0;
  int64_t ema_magnet_nontrivial_count = 0;
  bool use_ema_cache = false;
  torch::Tensor ema_log_probs_cache;

  // Forced buy & H/J study state
  std::vector<size_t> forced_indices;
  std::vector<float> log_pi_before;
  bool do_filtering = false;
  double filter_quantile = 0.0;
  bool study_readouts = false;
  bool actor_only_filtering = false;
  double actor_adv_threshold = 0.0;
  bool phase_stratified_norm = false;
  std::vector<int64_t> sa_indices;

  // Optimizer & grad-norm reuse state
  std::vector<FusedGroupStepCache> fused_step_caches;
  torch::Tensor cached_group_indices_dev;
  std::vector<const c10::TensorImpl*> cached_telemetry_keys;
  bool reuse_foreach_norms = false;
};

void StageAndUploadMinibatch(
    TrainPpoUpdateContext& ctx,
    const int64_t* mb_indices,
    int64_t start_seq,
    int64_t mb_len,
    bool full_ppo_fields,
    torch::Tensor& out_states,
    torch::Tensor& out_masks,
    torch::Tensor& out_actions,
    torch::Tensor& out_old_log_probs,
    torch::Tensor& out_nontrivial,
    std::vector<const dune_semantic::CandidateActionData*>* out_cands,
    torch::Tensor* out_advantages = nullptr,
    torch::Tensor* out_returns = nullptr,
    torch::Tensor* out_old_values = nullptr,
    int64_t* out_num_nontrivial = nullptr,
    torch::Tensor* out_off_policy = nullptr,
    int64_t* out_num_onpolicy_nontrivial = nullptr);

void ExecuteOptimizerStep(TrainPpoUpdateContext& ctx);

void RecordT1SliceDeltaLogOdds(
    TrainPpoUpdateContext& ctx,
    const torch::Tensor& slice_states,
    const torch::Tensor& slice_masks,
    const torch::Tensor& slice_masks_cpu,
    const torch::Tensor& slice_targets_cpu,
    const torch::Tensor& slice_logp_before_cpu,
    const std::vector<const dune_semantic::CandidateActionData*>& slice_cands,
    const std::vector<std::pair<Action, Action>>& slice_pairs,
    const std::vector<int>& slice_ages);

std::vector<float> ComputeForcedLogProbs(
    const TrainPpoUpdateContext& ctx,
    const std::vector<size_t>& indices);

double ComputeValueMse(
    const TrainPpoUpdateContext& ctx,
    const std::vector<int64_t>& indices);

// Returns true if --diagnostics_only triggered an early return.
bool RunTrainPpoUpdatePrepass(TrainPpoUpdateContext& ctx);

void RunMinibatchAuxStep(
    TrainPpoUpdateContext& ctx,
    int64_t mb_index,
    int64_t mb_len,
    MinibatchAuxState& mb_aux);

void AccumulateAuxPpoGradStats(
    TrainPpoUpdateContext& ctx,
    const MinibatchAuxState& mb_aux);

void RunMicrobatchPpoForwardBackward(
    TrainPpoUpdateContext& ctx,
    int64_t cfg_micro_mb,
    MinibatchPpoBuffers& mb);

void RunTrainPpoUpdateEpochLoop(TrainPpoUpdateContext& ctx);

void RunTrainPpoUpdateFinishPass(TrainPpoUpdateContext& ctx);

}  // namespace open_spiel::ppo_utils_internal

#endif  // OPEN_SPIEL_BUILD_WITH_LIBTORCH
