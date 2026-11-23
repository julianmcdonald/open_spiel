#ifndef OPEN_SPIEL_EXAMPLES_DUNE_SEARCH_SESSION_INTERNAL_H_
#define OPEN_SPIEL_EXAMPLES_DUNE_SEARCH_SESSION_INTERNAL_H_

#include <array>
#include <cstdint>
#include <memory>
#include <shared_mutex>
#include <string>
#include <vector>

#include <torch/torch.h>

#include "dune_batched_evaluator.h"
#include "dune_compound_turn_search.h"
#include "dune_evaluator.h"
#include "dune_search_session.h"
#include "dune_wide_rollout_search.h"
#include "open_spiel/games/dune_imperium/dune_imperium.h"

namespace open_spiel {

struct DuneAdaptiveRolloutSession::Impl {
  torch::Device device;
  std::shared_ptr<SharedDunePolicyValueNetImpl> model;
  std::unique_ptr<std::shared_mutex> mutex;
  std::vector<std::shared_ptr<BatchedEvaluator>> coordinators;
  std::vector<std::shared_ptr<BatchedNNEvaluator>> evaluators;
  dune_imperium::TurnSearchTree tree;

  std::shared_ptr<SharedDunePolicyValueNetImpl> rollout_model;
  std::unique_ptr<std::shared_mutex> rollout_mutex;
  std::vector<std::shared_ptr<BatchedEvaluator>> rollout_coordinators;
  std::vector<std::shared_ptr<BatchedNNEvaluator>> rollout_evaluators;

  explicit Impl(torch::Device dev) : device(dev) {}
};

namespace dune_search_session_internal {

std::string LocalPhaseToString(dune_imperium::GamePhase phase);

Action PickActionRespectingTemperature(const ActionsAndProbs& policy,
                                       double temperature,
                                       double r_val);

Action ArgmaxVisitAction(const SearchDiagnostics& diagnostics);
int GetActionVisits(const SearchDiagnostics& diagnostics, Action action);
double GetActionQValue(const SearchDiagnostics& diagnostics, Action action);

struct LoadedCheckpointNet {
  std::shared_ptr<SharedDunePolicyValueNetImpl> model;
  dune_imperium::MarketAppendixMode detected_mode =
      dune_imperium::MarketAppendixMode::kNone;
  int hidden_dim = 0;
  int num_blocks = 0;
  bool has_json_num_blocks = false;
};

LoadedCheckpointNet LoadDuneCheckpointNet(
    const std::string& checkpoint_path,
    const torch::Device& device,
    int64_t default_input_dim,
    int64_t action_size,
    int hidden_dim,
    int num_blocks,
    bool freeze_params,
    bool use_cuda_amp,
    bool enable_fp8_trunk);

std::vector<int> BuildSessionWarmupSizes(int target_batch_size,
                                         int max_inflight_worlds);

void InitCudaSuperhumanPriors(std::array<double, 12>& ms_per_round,
                              double& ms_ema,
                              int rollout_num_blocks,
                              bool& initialized_flag);

struct BranchRemResult {
  double weighted_rem = 0.0;
  int raw_rem_count = 0;
};

BranchRemResult CountBranchSearchedWeighted(
    const State& state,
    Player p,
    BatchedNNEvaluator* prior_eval,
    int hand0,
    int int0,
    Action first_action,
    Action graft_action,
    Action second_action,
    int max_count,
    double stage1_high_conf_prior_skip,
    bool allow_non_hs_subsearch);

int ComputeEffectiveRoundBucket(const State& state, Player p);

void SyncAdaptiveTurnTree(
    const State& state,
    const dune_imperium::DuneImperiumState& dune_state,
    Player p,
    dune_imperium::TurnSearchTree& tree,
    bool has_active_session,
    Player last_player,
    int last_round,
    int last_hand,
    int last_intrigues,
    const std::vector<Action>& last_history,
    double& turn_elapsed_search_time_ms,
    Action& pending_macro_space_hint);

void UpdateAdaptiveRolloutRateEma(
    const State& state,
    Player p,
    bool is_leader,
    double remaining_time_ms,
    double elapsed_ms,
    const DuneSearchResult& result,
    bool search_v2,
    bool superhuman_mode,
    double& ms_per_rollout_ema,
    std::array<double, 12>& ms_per_rollout_by_round);

void PopulateRolloutStageDiagnostics(
    DuneSearchResult& result,
    const dune_imperium::CompoundSearchDecisionResult& c_res);

void PopulateCandidateDiagnosticsByAction(
    DuneSearchResult& result,
    const std::vector<Action>& legal,
    const dune_imperium::CompoundSearchDecisionResult& c_res);

void PopulateJointCandidateDiagnostics(
    DuneSearchResult& result,
    const std::vector<Action>& legal,
    const std::vector<dune_imperium::JointAgentCandidate>& joint_cands,
    const dune_imperium::CompoundSearchDecisionResult& c_res);

void CheckLiveDeadlineFallback(
    DuneSearchResult& result,
    const dune_imperium::CompoundSearchDecisionResult& c_res,
    double remaining_time_ms,
    bool is_leader,
    bool search_v2,
    int num_cands);

struct AdaptiveRolloutDispatchParams {
  Player player = kInvalidPlayer;
  double remaining_time_ms = -1.0;
  double target_ms = -1.0;
  int ref_rollouts = 0;
  double round_ms_est = 24.0;
  uint64_t decision_seed = 0;
  int max_inflight_worlds = 96;
  int num_evaluators = 1;
  int top5_min_budget = 576;
  bool full_search = false;
  bool superhuman_mode = false;
  bool fixed_top5_candidates = true;
  bool all_buy_candidates = true;
  bool use_event_keyed_crn = true;
  bool use_successive_halving = true;
  bool unbiased_stage2_gate = false;
  bool rescue_optional_abilities = false;
  bool ceiling_aware_margin = true;
  double win_speed_tiebreak_eps = 0.0;
  double prior_log_ratio_weight = 0.02;
  double stage1_high_conf_prior_skip = 0.95;
  bool enable_macro_space_candidates = false;
  int macro_space_min_round = 1;
  Action forced_include_action = kInvalidAction;
  bool enable_endgame_ply2 = false;
  bool endgame_ply2_stage2_only = true;
  int endgame_ply2_worlds = 4;
  double endgame_ply2_margin = 0.12;
  bool enable_intrigue_belief = false;
  double intrigue_belief_age_combat_slope = 0.25;
  double intrigue_belief_age_endgame_slope = 0.30;
  double intrigue_belief_close_pass_combat_penalty = -1.20;
  dune_imperium::SearchV2TuningConfig search_v2_tuning{};
};

void RunAdaptiveLeaderRolloutSearch(
    const State& state,
    Player p,
    const std::vector<Action>& legal,
    const ActionsAndProbs& raw_prior,
    const std::vector<std::shared_ptr<BatchedNNEvaluator>>& evaluators,
    int effective_leader_rollouts,
    double leader_deadline_ms,
    int leader_inflight,
    uint64_t leader_seed,
    bool use_event_keyed_crn,
    double win_speed_tiebreak_eps,
    bool ceiling_aware_margin,
    double remaining_time_ms,
    bool search_v2,
    DuneSearchResult& result);

void RunAdaptiveV2JointRolloutSearch(
    const State& state,
    const AdaptiveRolloutDispatchParams& params,
    const std::vector<Action>& legal,
    Action raw_action,
    const ActionsAndProbs& raw_prior,
    dune_imperium::TurnSearchTree* tree,
    const std::vector<std::shared_ptr<BatchedNNEvaluator>>& evaluators,
    const std::vector<dune_imperium::JointAgentCandidate>& joint_cands,
    DuneSearchResult& result,
    bool& out_cached_followups_valid,
    int& out_chosen_c_idx,
    Action& out_chosen_first,
    Action& out_chosen_graft,
    Action& out_chosen_second,
    double& out_chosen_q,
    int& out_chosen_visits);

void RunAdaptiveV2SingleRolloutSearch(
    const State& state,
    const AdaptiveRolloutDispatchParams& params,
    const std::vector<Action>& legal,
    const ActionsAndProbs& raw_prior,
    dune_imperium::TurnSearchTree* tree,
    const std::vector<std::shared_ptr<BatchedNNEvaluator>>& evaluators,
    DuneSearchResult& result);

void RunAdaptiveV1RolloutSearch(
    const State& state,
    const AdaptiveRolloutDispatchParams& params,
    const std::vector<Action>& legal,
    const ActionsAndProbs& raw_prior,
    dune_imperium::TurnSearchTree* tree,
    const std::vector<std::shared_ptr<BatchedNNEvaluator>>& evaluators,
    BatchedNNEvaluator* prior_eval,
    DuneSearchResult& result,
    Action& out_chosen_second_action);

void FinalizeTreeReuseOrFallback(
    dune_imperium::TurnSearchTree* tree,
    const std::vector<Action>& legal,
    Action raw_action,
    const ActionsAndProbs& raw_prior,
    bool search_v2,
    bool live_deadline_starved,
    double remaining_time_ms,
    double available_after_reserve_ms,
    DuneSearchResult& result);

void PlanV1SearchBudget(
    const State& state,
    const dune_imperium::DuneImperiumState& dune_state,
    Player p,
    const std::vector<Action>& legal,
    const ActionsAndProbs& raw_prior,
    bool is_agent_turn_phase,
    bool full_search,
    bool superhuman_mode,
    bool is_cuda,
    double reserve_ms,
    int reference_rollouts_per_action,
    double available_after_reserve_ms,
    double remaining_time_ms,
    double round_ms_est,
    BatchedNNEvaluator* prior_eval,
    DuneSearchResult& result,
    double& target_ms,
    int& ref_rollouts);

}  // namespace dune_search_session_internal

}  // namespace open_spiel

#endif  // OPEN_SPIEL_EXAMPLES_DUNE_SEARCH_SESSION_INTERNAL_H_
