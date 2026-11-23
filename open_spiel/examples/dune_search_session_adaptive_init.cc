#include "dune_search_session_internal.h"

#include <algorithm>
#include <cstdlib>

namespace open_spiel {
namespace {

void InitEvaluatorPool(
    const std::shared_ptr<SharedDunePolicyValueNetImpl>& model,
    std::unique_ptr<std::shared_mutex>& mutex,
    std::vector<std::shared_ptr<BatchedEvaluator>>& coordinators,
    std::vector<std::shared_ptr<BatchedNNEvaluator>>& evaluators,
    int num_evaluators,
    int target_batch_size,
    int max_inflight_worlds,
    const torch::Device& device,
    bool use_cuda_amp,
    dune_imperium::MarketAppendixMode detected_mode) {
  mutex = std::make_unique<std::shared_mutex>();
  coordinators.resize(num_evaluators);
  evaluators.resize(num_evaluators);
  const std::vector<int> warmup_sizes =
      dune_search_session_internal::BuildSessionWarmupSizes(target_batch_size,
                                                            max_inflight_worlds);
  for (int e = 0; e < num_evaluators; ++e) {
    coordinators[e] = std::make_shared<BatchedEvaluator>(
        model, target_batch_size, 1, device,
        mutex.get(), 10.0f, false, true, false,
        /*rollout_amp=*/use_cuda_amp, /*allow_tf32=*/use_cuda_amp);
    evaluators[e] = std::make_shared<BatchedNNEvaluator>(
        coordinators[e], 10.0f, detected_mode);
    if (use_cuda_amp && std::getenv("DUNE_DISABLE_SESSION_GRAPH_WARMUP") == nullptr) {
      coordinators[e]->WarmupCudaGraphs(warmup_sizes);
    }
  }
}

}  // namespace

DuneAdaptiveRolloutSession::DuneAdaptiveRolloutSession(
    const std::string& checkpoint_path,
    const std::string& device_str,
    int hidden_dim,
    int num_blocks,
    int reference_rollouts_per_action,
    int leader_rollouts,
    int target_batch_size,
    uint64_t seed,
    bool use_successive_halving,
    bool fixed_top5_candidates,
    int max_inflight_worlds,
    int num_evaluators,
    int top5_min_budget,
    bool all_buy_candidates,
    bool enable_fp8_trunk,
    bool search_v2)
    : checkpoint_path_(checkpoint_path),
      device_str_(device_str),
      hidden_dim_(hidden_dim),
      num_blocks_(num_blocks),
      reference_rollouts_per_action_(reference_rollouts_per_action),
      leader_rollouts_(leader_rollouts),
      target_batch_size_(target_batch_size),
      seed_(seed),
      use_successive_halving_(use_successive_halving),
      fixed_top5_candidates_(fixed_top5_candidates),
      max_inflight_worlds_(std::max(1, max_inflight_worlds)),
      num_evaluators_(std::max(1, num_evaluators)),
      top5_min_budget_(top5_min_budget),
      all_buy_candidates_(all_buy_candidates),
      enable_fp8_trunk_(enable_fp8_trunk),
      search_v2_(search_v2) {
  const char* env_fp8 = std::getenv("DUNE_ENABLE_FP8_TRUNK");
  if (env_fp8 != nullptr && (std::string(env_fp8) == "1" || std::string(env_fp8) == "true")) {
    enable_fp8_trunk_ = true;
  }
  torch::Device device(torch::kCPU);
  if (device_str == "cuda" && torch::cuda::is_available()) {
    device = torch::Device(torch::kCUDA);
  } else if (device_str == "cpu") {
    device = torch::Device(torch::kCPU);
  } else {
    device = torch::Device(device_str);
  }

  impl_ = std::make_unique<Impl>(device);

  const bool use_cuda_amp = impl_->device.is_cuda();
  auto loaded = dune_search_session_internal::LoadDuneCheckpointNet(
      checkpoint_path_, impl_->device,
      dune_imperium::kFullPublicInformationStateSize,
      dune_imperium::kNumDistinctPlayerActions,
      hidden_dim_, num_blocks_,
      /*freeze_params=*/true, use_cuda_amp, enable_fp8_trunk_);
  hidden_dim_ = loaded.hidden_dim;
  num_blocks_ = loaded.num_blocks;
  if (loaded.has_json_num_blocks) {
    rollout_num_blocks_ = num_blocks_;
  }
  impl_->model = std::move(loaded.model);

  InitEvaluatorPool(impl_->model, impl_->mutex, impl_->coordinators,
                    impl_->evaluators, num_evaluators_, target_batch_size_,
                    max_inflight_worlds_, impl_->device, use_cuda_amp,
                    loaded.detected_mode);

  ResetSession("initialization");
}

DuneAdaptiveRolloutSession::~DuneAdaptiveRolloutSession() = default;

void DuneAdaptiveRolloutSession::set_superhuman_mode(bool val) {
  superhuman_mode_ = val;
  if (val) {
    full_search_ = true;
    search_v2_ = false;
    win_speed_tiebreak_eps_ = 0.05;
    ceiling_aware_margin_ = true;
    unbiased_stage2_gate_ = true;
    all_buy_candidates_ = true;
    fixed_top5_candidates_ = true;
    use_event_keyed_crn_ = true;
    rescue_optional_abilities_ = false;
    if (impl_ && impl_->device.is_cuda() && !cuda_superhuman_priors_initialized_) {
      dune_search_session_internal::InitCudaSuperhumanPriors(
          ms_per_rollout_by_round_, ms_per_rollout_ema_, rollout_num_blocks_,
          cuda_superhuman_priors_initialized_);
    }
  } else if (!search_v2_) {
    ms_per_rollout_by_round_ = {{20.2, 18.5, 17.3, 15.2, 13.4, 10.6, 8.5, 6.1, 4.3, 3.2, 2.5, 2.0}};
    cuda_superhuman_priors_initialized_ = false;
  }
  if (!val) {
    rescue_optional_abilities_ = false;
  }
}

void DuneAdaptiveRolloutSession::ApplySessionConfig(
    const DuneSearchSessionConfig& cfg) {
  superhuman_mode_ = cfg.superhuman_mode;
  full_search_ = cfg.superhuman_mode ? true : cfg.full_search;
  set_search_v2(cfg.superhuman_mode ? false : cfg.search_v2);
  if (cfg.superhuman_mode) {
    superhuman_mode_ = true;
    if (impl_ && impl_->device.is_cuda() && !cuda_superhuman_priors_initialized_) {
      dune_search_session_internal::InitCudaSuperhumanPriors(
          ms_per_rollout_by_round_, ms_per_rollout_ema_, rollout_num_blocks_,
          cuda_superhuman_priors_initialized_);
    }
  }
  unbiased_stage2_gate_ = cfg.unbiased_stage2_gate;
  fixed_top5_candidates_ = cfg.fixed_top5_candidates;
  all_buy_candidates_ = cfg.all_buy_candidates;
  use_event_keyed_crn_ = cfg.use_event_keyed_crn;
  ceiling_aware_margin_ = cfg.ceiling_aware_margin;
  win_speed_tiebreak_eps_ = std::max(0.0, cfg.win_speed_tiebreak_eps);
  kl_policy_tau_ = std::max(0.0, cfg.kl_policy_tau);
  reserve_ms_ = std::max(0.0, cfg.reserve_ms);
  prior_log_ratio_weight_ = std::max(0.0, cfg.prior_log_ratio_weight);
  stage1_high_conf_prior_skip_ = std::max(0.0, cfg.stage1_high_conf_prior_skip);

  enable_macro_space_candidates_ = cfg.enable_macro_space_candidates;
  macro_space_min_round_ = cfg.macro_space_min_round;
  enable_endgame_ply2_ = cfg.enable_endgame_ply2;
  endgame_ply2_stage2_only_ = cfg.endgame_ply2_stage2_only;
  endgame_ply2_worlds_ = cfg.endgame_ply2_worlds;
  endgame_ply2_margin_ = cfg.endgame_ply2_margin;
  enable_intrigue_belief_ = cfg.enable_intrigue_belief;
  intrigue_belief_age_combat_slope_ = cfg.intrigue_belief_age_combat_slope;
  intrigue_belief_age_endgame_slope_ = cfg.intrigue_belief_age_endgame_slope;
  intrigue_belief_close_pass_combat_penalty_ = cfg.intrigue_belief_close_pass_combat_penalty;
  rescue_optional_abilities_ = cfg.rescue_optional_abilities;
  if (!cfg.rollout_checkpoint.empty()) {
    set_rollout_checkpoint(cfg.rollout_checkpoint);
  }
}

DuneSearchSessionConfig DuneAdaptiveRolloutSession::GetSessionConfig() const {
  DuneSearchSessionConfig cfg;
  cfg.superhuman_mode = superhuman_mode_;
  cfg.full_search = full_search_;
  cfg.search_v2 = search_v2_;
  cfg.unbiased_stage2_gate = unbiased_stage2_gate_;
  cfg.fixed_top5_candidates = fixed_top5_candidates_;
  cfg.all_buy_candidates = all_buy_candidates_;
  cfg.use_event_keyed_crn = use_event_keyed_crn_;
  cfg.ceiling_aware_margin = ceiling_aware_margin_;
  cfg.win_speed_tiebreak_eps = win_speed_tiebreak_eps_;
  cfg.kl_policy_tau = kl_policy_tau_;
  cfg.reserve_ms = reserve_ms_;
  cfg.prior_log_ratio_weight = prior_log_ratio_weight_;
  cfg.stage1_high_conf_prior_skip = stage1_high_conf_prior_skip_;

  cfg.enable_macro_space_candidates = enable_macro_space_candidates_;
  cfg.macro_space_min_round = macro_space_min_round_;
  cfg.enable_endgame_ply2 = enable_endgame_ply2_;
  cfg.endgame_ply2_stage2_only = endgame_ply2_stage2_only_;
  cfg.endgame_ply2_worlds = endgame_ply2_worlds_;
  cfg.endgame_ply2_margin = endgame_ply2_margin_;
  cfg.enable_intrigue_belief = enable_intrigue_belief_;
  cfg.intrigue_belief_age_combat_slope = intrigue_belief_age_combat_slope_;
  cfg.intrigue_belief_age_endgame_slope = intrigue_belief_age_endgame_slope_;
  cfg.intrigue_belief_close_pass_combat_penalty = intrigue_belief_close_pass_combat_penalty_;
  cfg.rescue_optional_abilities = rescue_optional_abilities_;
  cfg.rollout_checkpoint = rollout_checkpoint_path_;
  return cfg;
}

void DuneAdaptiveRolloutSession::set_enable_fp8_trunk(bool val) {
  if (enable_fp8_trunk_ == val) return;
  enable_fp8_trunk_ = val;
  cuda_superhuman_priors_initialized_ = false;
  if (!impl_ || !impl_->device.is_cuda()) return;
  const std::vector<int> warmup_sizes =
      dune_search_session_internal::BuildSessionWarmupSizes(target_batch_size_,
                                                            max_inflight_worlds_);
  for (auto& coord : impl_->coordinators) {
    if (coord) {
      coord->SetEnableFp8Trunk(val);
      if (std::getenv("DUNE_DISABLE_SESSION_GRAPH_WARMUP") == nullptr) {
        coord->WarmupCudaGraphs(warmup_sizes);
      }
    }
  }
  for (auto& coord : impl_->rollout_coordinators) {
    if (coord) {
      coord->SetEnableFp8Trunk(val);
      if (std::getenv("DUNE_DISABLE_SESSION_GRAPH_WARMUP") == nullptr) {
        coord->WarmupCudaGraphs(warmup_sizes);
      }
    }
  }
  if (superhuman_mode_) {
    dune_search_session_internal::InitCudaSuperhumanPriors(
        ms_per_rollout_by_round_, ms_per_rollout_ema_, rollout_num_blocks_,
        cuda_superhuman_priors_initialized_);
  }
}

const std::vector<std::shared_ptr<BatchedNNEvaluator>>&
DuneAdaptiveRolloutSession::ActiveRolloutEvaluators() const {
  return (impl_->rollout_evaluators.empty()) ? impl_->evaluators
                                             : impl_->rollout_evaluators;
}

void DuneAdaptiveRolloutSession::set_rollout_checkpoint(const std::string& path) {
  if (path.empty() || path == checkpoint_path_) {
    if (impl_) {
      impl_->rollout_evaluators.clear();
      impl_->rollout_coordinators.clear();
      impl_->rollout_model.reset();
      impl_->rollout_mutex.reset();
    }
    rollout_checkpoint_path_ = "";
    rollout_num_blocks_ = num_blocks_;
    cuda_superhuman_priors_initialized_ = false;
    if (superhuman_mode_ && impl_ && impl_->device.is_cuda()) {
      dune_search_session_internal::InitCudaSuperhumanPriors(
          ms_per_rollout_by_round_, ms_per_rollout_ema_, rollout_num_blocks_,
          cuda_superhuman_priors_initialized_);
    }
    return;
  }

  const bool use_cuda_amp = impl_->device.is_cuda();
  auto loaded = dune_search_session_internal::LoadDuneCheckpointNet(
      path, impl_->device,
      dune_imperium::kFullPublicInformationStateSize,
      dune_imperium::kNumDistinctPlayerActions,
      hidden_dim_, num_blocks_,
      /*freeze_params=*/true, use_cuda_amp, enable_fp8_trunk_);
  impl_->rollout_model = std::move(loaded.model);

  InitEvaluatorPool(impl_->rollout_model, impl_->rollout_mutex,
                    impl_->rollout_coordinators, impl_->rollout_evaluators,
                    num_evaluators_, target_batch_size_, max_inflight_worlds_,
                    impl_->device, use_cuda_amp, loaded.detected_mode);

  rollout_checkpoint_path_ = path;
  rollout_num_blocks_ = loaded.num_blocks;
  cuda_superhuman_priors_initialized_ = false;
  if (superhuman_mode_ && impl_ && impl_->device.is_cuda()) {
    dune_search_session_internal::InitCudaSuperhumanPriors(
        ms_per_rollout_by_round_, ms_per_rollout_ema_, rollout_num_blocks_,
        cuda_superhuman_priors_initialized_);
  }
}

PolicyPriorDetails DuneAdaptiveRolloutSession::PriorWithDetails(const State& state) const {
  if (!impl_ || impl_->evaluators.empty() || !impl_->evaluators.front() || state.IsTerminal()) {
    return {false, {}, {}, 0.0f};
  }
  if (state.IsChanceNode()) {
    PolicyPriorDetails details;
    auto outcomes = state.ChanceOutcomes();
    details.actions.reserve(outcomes.size());
    details.probabilities.reserve(outcomes.size());
    for (const auto& [action, prob] : outcomes) {
      details.actions.push_back(action);
      details.probabilities.push_back(static_cast<float>(prob));
    }
    details.ok = !details.actions.empty();
    details.value = 0.0f;
    return details;
  }
  open_spiel::CompactEvalResult res = impl_->evaluators.front()->PriorWithDetails(state);
  return {res.ok, std::move(res.actions), std::move(res.probabilities), res.value};
}

void DuneAdaptiveRolloutSession::DiscardPendingAction() {
  last_search_state_.reset();
  has_pending_commit_ = false;
}

void DuneAdaptiveRolloutSession::ResetSession(const std::string& reason) {
  if (impl_) {
    impl_->tree.Reset();
  }
  has_active_session_ = false;
  has_pending_commit_ = false;
  last_search_state_.reset();
  last_player_ = kInvalidPlayer;
  last_round_ = -1;
  last_hand_ = 0;
  last_intrigues_ = 0;
  last_history_.clear();
  session_new_simulations_completed_ = 0;
  session_elapsed_time_ms_ = 0.0;
  turn_elapsed_search_time_ms_ = 0.0;
  pending_macro_space_hint_ = kInvalidAction;
  if (!search_v2_) {
    if (!superhuman_mode_ || !impl_ || !impl_->device.is_cuda()) {
      ms_per_rollout_by_round_ = {{20.2, 18.5, 17.3, 15.2, 13.4, 10.6, 8.5, 6.1, 4.3, 3.2, 2.5, 2.0}};
      cuda_superhuman_priors_initialized_ = false;
    } else if (!cuda_superhuman_priors_initialized_) {
      dune_search_session_internal::InitCudaSuperhumanPriors(
          ms_per_rollout_by_round_, ms_per_rollout_ema_, rollout_num_blocks_,
          cuda_superhuman_priors_initialized_);
    }
  } else if (!search_v2_round_priors_initialized_) {
    ms_per_rollout_by_round_ = {{15.0, 11.8, 11.1, 8.9, 8.0, 6.5, 5.4, 3.5, 2.5, 2.5, 2.5, 2.5}};
    ms_per_rollout_ema_ = 11.8;
    search_v2_round_priors_initialized_ = true;
  }
  cached_joint_followups_.clear();
  cached_joint_followup_round_ = -1;
  cached_joint_followup_player_ = kInvalidPlayer;
}

}  // namespace open_spiel
