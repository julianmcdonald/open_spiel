#pragma once

#include <algorithm>
#include <atomic>
#include <chrono>
#include <csignal>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <deque>
#include <array>
#include <numeric>
#include <sstream>
#include <iomanip>

#include <random>
#include <set>
#include <shared_mutex>
#include <string>
#include <thread>
#include <unordered_set>
#include <vector>

#include "open_spiel/abseil-cpp/absl/flags/flag.h"
#include "open_spiel/abseil-cpp/absl/flags/reflection.h"
#include "dune_ppo_training_flags.h"
#include "open_spiel/abseil-cpp/absl/strings/str_format.h"
#include "open_spiel/abseil-cpp/absl/strings/str_split.h"
#include "open_spiel/abseil-cpp/absl/strings/str_join.h"
#include "open_spiel/games/dune_imperium/dune_imperium.h"
#include "open_spiel/games/dune_imperium/dune_card_telemetry.h"

#include "open_spiel/games/dune_imperium/dune_imperium_util.h"
#include "open_spiel/spiel.h"

#ifdef OPEN_SPIEL_BUILD_WITH_LIBTORCH
#include <c10/cuda/CUDACachingAllocator.h>
#include <torch/torch.h>

#include "dune_network.h"
#include "dune_seed_utils.h"
#include "dune_sha256.h"
#include "dune_ppo_training_utils.h"
#include "dune_ppo_numerical_parity.h"
#include "dune_search_label_buffer.h"
#include "dune_search_routing.h"
#include "dune_evaluator.h"  // DuneNNEvaluator for online-collection snapshot inference
#include "dune_opponent_pool.h"
#include "dune_compound_turn_search.h"
#include "dune_wide_rollout_search.h"
#include "dune_batched_evaluator.h"
#include "dune_targeted_buy.h"
#include "open_spiel/utils/json.h"
#endif

#include "dune_ppo_train_flags_internal.h"

namespace open_spiel::ppo_train_internal {

extern std::set<std::vector<open_spiel::Action>> prohibited_histories;
extern std::atomic<bool> g_stop_requested;
void SignalHandler(int signum);

#ifdef OPEN_SPIEL_BUILD_WITH_LIBTORCH

class PpoRolloutBuffer {
 public:
  size_t PushTrajectory(std::vector<PpoTransition>&& trajectory);
  size_t Size() const;
  std::vector<PpoTransition> TakeAll(bool* out_episode_ids_unique = nullptr);

 private:
  mutable std::mutex mu_;
  std::vector<std::vector<PpoTransition>> trajectories_;
  size_t num_transitions_ = 0;
};

// SearchLabel, SearchLabelRole, SearchLabelFileEntry and SearchLabelBuffer now
// live in dune_search_label_buffer.h so that the trainer and the ruling-5 role
// tests link ONE definition rather than two that can drift.
using open_spiel::SearchLabel;
using open_spiel::SearchLabelBuffer;
using open_spiel::SearchLabelFileEntry;
using open_spiel::SearchLabelRole;

using dune_imperium::CardMetrics;
using dune_imperium::SaveCardTelemetry;
using dune_imperium::LoadCardTelemetry;

struct R765OverrideProbeRecord {
  uint64_t root_uid = 0;
  std::string category;  // "card", "same_card_space", "cross_card_space", "graft", "cat3_purchase"
  std::vector<float> observation;
  std::vector<Action> legal_actions;
  dune_semantic::CandidateActionData candidate_data;
  Action a_star = open_spiel::kInvalidAction;
  Action a_ref = open_spiel::kInvalidAction;
  double delta_q = 0.0;
  double realized_kl = 0.0;
  double realized_temp = 0.0;
  bool kl_cap_bound = false;
  double t0_log_odds = 0.0;
  double target_log_odds = 0.0;
  bool target_flips_argmax = false;
  double t2_live_log_odds = 0.0;
  double t2_ema_log_odds = 0.0;
};

struct R765AgreedProbeRecord {
  std::vector<float> observation;
  std::vector<Action> legal_actions;
  dune_semantic::CandidateActionData candidate_data;
  Action greedy_action = open_spiel::kInvalidAction;
};

struct WorkerStats {
  uint64_t games = 0;
  uint64_t moves = 0;
  uint64_t search_eligible_decisions = 0;

  CardMetrics card11_metrics;
  CardMetrics card13_metrics;
  open_spiel::dune_targeted_buy::TargetedBuyTelemetry targeted_buy_telemetry;

  // Reward penalties telemetry
  uint64_t specimen_conversions = 0;
  double specimen_penalty_deducted = 0.0;
  uint64_t atomics_penalized = 0;
  uint64_t atomics_exempt = 0;
  double atomics_penalty_deducted = 0.0;
  uint64_t plot_penalized = 0;
  uint64_t plot_exempt = 0;
  double plot_penalty_deducted = 0.0;

  // D2 purchase exploration
  uint64_t forced_buys = 0;

  // High-stakes sanity check counters
  uint64_t nontrivial_transitions = 0;
  uint64_t high_stakes_pre_transitions = 0;
  uint64_t high_stakes_post_transitions = 0;

  // Natural SM Acquisitions
  uint64_t sm_acquisitions_by_seat[4] = {0};
  uint64_t sm_acquisitions_by_leader[14] = {0};
  uint64_t sm_acquisitions_by_round[11] = {0};

  double conflict_vp_generated = 0.0;
  double conflict_vp_attributed = 0.0;
  double conflict_vp_unattributed = 0.0;

  // Raw VPs for conservation checks
  double raw_conflict_vp = 0.0;
  double raw_noncombat_vp = 0.0;
  double raw_total_vp = 0.0;

  // Phase A canary: per-decision PRE-cap legal-centered max|z| for every
  // nontrivial rollout decision this worker made, overall and split by decision
  // role. Purely observational -- collected before CenterAndCapLegalLogits
  // mutates the logits, and never read back into the acting path.
  std::vector<float> precap_absz_all;
  std::vector<float> precap_absz_primary;
  std::vector<float> precap_absz_continuation;
  std::vector<float> precap_absz_purchase;

  // --- PWO-5 section 14.1b: the three collapse canaries. ------------------
  //
  // Measured at the ROLLOUT BEHAVIOUR-POLICY DECISION and nowhere else. The
  // registration considered three candidate sites and rejected the other two:
  // the executed-PPO-minibatch site (existing col 36 `entropy`) is
  // presentation-weighted over an EVOLVING model and is truncated by
  // --target_kl, so an arm that early-stops more often would show a different
  // entropy for reasons unrelated to collapse -- "a safety rail must not be a
  // function of its own optimizer's stopping behaviour"; and a pre-optimizer
  // snapshot would cost a second forward over 32,768 transitions and still
  // lack the role and legal-action data.
  //
  // Population: unique rollout decisions of the acting seat with n_legal >= 2,
  // one measurement each. No re-presentation, no epoch weighting, no dependence
  // on KL early stop.
  //
  // These are NOT column 36 and may never be compared to or substituted for it.
  //
  // Role index is the integer value of DuneDecisionRole (7 members). The
  // kForcedOrBookkeeping slot is identically empty at this site, by the
  // section 14.1c proof: the classifier returns it iff the searched player is
  // the chance player OR n_legal <= 1, chance nodes `continue` before ever
  // reaching the classifier here, and the population already requires
  // n_legal >= 2. It is accumulated anyway so the partition identity is
  // checked rather than assumed.
  double canary_ne_sum = 0.0;             // sum of H_row / log(n_legal)
  int64_t canary_ne_n = 0;                // qualifying decisions
  double canary_ne_sum_role[7] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  int64_t canary_ne_n_role[7] = {0, 0, 0, 0, 0, 0, 0};
  double canary_maxprob_sum = 0.0;        // denominator is canary_ne_n
  int64_t canary_sat_ge_cap = 0;          // legal PRE-cap |z| >= logit_cap
  int64_t canary_sat_logits = 0;          // legal logits examined (LOGIT unit)

  // V3 numerical-parity capture contract. Populated only when the diagnostic
  // output flag is set; normal rollout leaves every counter/vector untouched.
  int64_t parity_decisions = 0;
  int64_t parity_full_width_ok = 0;
  int64_t parity_full_finite_ok = 0;
  int64_t parity_legal_ids_unique_in_range = 0;
  int64_t parity_chosen_once = 0;
  std::vector<std::string> parity_capture_errors;

  int64_t r755_searched_by_cat[6] = {0, 0, 0, 0, 0, 0};
  int64_t r755_executed_overrides_by_cat[6] = {0, 0, 0, 0, 0, 0};

  // R755 Cat 3 Gate 2 Critic / GAE diagnostics: 9 buckets (3 rounds x 3 priors)
  // for (i) 4*(G_buy - V_buy), (ii) 4*(G_after - V_after), (iii) 4*(G_buy - (V_buy + A_buy))
  double r755_cat3_i_sum[9] = {0.0};
  double r755_cat3_ii_sum[9] = {0.0};
  double r755_cat3_iii_sum[9] = {0.0};
  int64_t r755_cat3_bucket_count[9] = {0};

  // Buy tracking: Imperium, Tleilaxu, Tech buys, unspent persuasion
  int64_t r755_imperium_buys = 0;
  int64_t r755_tleilaxu_buys = 0;
  int64_t r755_tech_buys = 0;
  int64_t r755_reveal_turns = 0;
  double r755_unspent_persuasion_sum = 0.0;

  std::vector<R765OverrideProbeRecord> r765_override_probes;
  std::vector<R765AgreedProbeRecord> r765_agreed_probes;
};

struct CounterfactualPending {
  PpoTransition transition;
  int replay_weight;
};

struct CollectResult {
  std::vector<PpoTransition> rollout;
  uint64_t games = 0;
  uint64_t moves = 0;
  double elapsed_seconds = 0.0;

  CardMetrics card11_metrics;
  CardMetrics card13_metrics;
  open_spiel::dune_targeted_buy::TargetedBuyTelemetry targeted_buy_telemetry;
  int targeted_buy_achieved_card11 = 0;
  int targeted_buy_achieved_card13 = 0;
  int targeted_buy_designated_opps_card11 = 0;
  int targeted_buy_designated_opps_card13 = 0;
  int targeted_buy_total_opps_card11 = 0;
  int targeted_buy_total_opps_card13 = 0;

  // Reward penalties telemetry
  uint64_t specimen_conversions = 0;
  double specimen_penalty_deducted = 0.0;
  uint64_t atomics_penalized = 0;
  uint64_t atomics_exempt = 0;
  double atomics_penalty_deducted = 0.0;
  uint64_t plot_penalized = 0;
  uint64_t plot_exempt = 0;
  double plot_penalty_deducted = 0.0;

  // D2 purchase exploration
  uint64_t forced_buys = 0;

  // High-stakes sanity check counters
  uint64_t nontrivial_transitions = 0;
  uint64_t high_stakes_pre_transitions = 0;
  uint64_t high_stakes_post_transitions = 0;

  uint64_t sm_acquisitions_by_seat[4] = {0};
  uint64_t sm_acquisitions_by_leader[14] = {0};
  uint64_t sm_acquisitions_by_round[11] = {0};

  double conflict_vp_generated = 0.0;
  double conflict_vp_attributed = 0.0;
  double conflict_vp_unattributed = 0.0;

  bool episode_ids_unique = true;
  double raw_conflict_vp = 0.0;
  double raw_noncombat_vp = 0.0;
  double raw_total_vp = 0.0;

  int64_t parity_decisions = 0;
  int64_t parity_full_width_ok = 0;
  int64_t parity_full_finite_ok = 0;
  int64_t parity_legal_ids_unique_in_range = 0;
  int64_t parity_chosen_once = 0;
  std::vector<std::string> parity_capture_errors;

  // Phase A canary: concatenation of every worker's per-decision max|z|, merged
  // in thread-index order. Only the multiset matters downstream (percentiles
  // sort), so the merge order does not affect the reported statistics.
  std::vector<float> precap_absz_all;
  std::vector<float> precap_absz_primary;
  std::vector<float> precap_absz_continuation;
  std::vector<float> precap_absz_purchase;

  // PWO-5 section 14.1b canaries, summed over workers. Summation is exact for
  // the integer counts; the double sums are order-dependent at the last ulp,
  // and the merge is in thread-index order, which is deterministic given a
  // thread count. That is the same guarantee every other double accumulator in
  // this struct carries, and it is why every bitwise gate runs at --threads=1.
  double canary_ne_sum = 0.0;
  int64_t canary_ne_n = 0;
  double canary_ne_sum_role[7] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  int64_t canary_ne_n_role[7] = {0, 0, 0, 0, 0, 0, 0};
  double canary_maxprob_sum = 0.0;
  int64_t canary_sat_ge_cap = 0;
  int64_t canary_sat_logits = 0;

  // Phase 18B: online auxiliary-search examples collected from the SAME frozen
  // snapshot as `rollout`, so they travel together and are consumed by the same
  // TrainPpoUpdate. Empty unless --online_search_collection.
  std::vector<open_spiel::SearchTrainingExample> aux_examples;
  open_spiel::OnlineSearchCollectionStats aux_stats;

  // Search-in-Training: reservoir-sampled decision snapshots visited by the learner
  std::vector<open_spiel::dune_imperium::SearchSnapshot> search_snapshots;
  uint64_t search_eligible_decisions = 0;

  int64_t r755_searched_by_cat[6] = {0, 0, 0, 0, 0, 0};
  int64_t r755_executed_overrides_by_cat[6] = {0, 0, 0, 0, 0, 0};

  // R755 Cat 3 Gate 2 Critic / GAE diagnostics: 9 buckets (3 rounds x 3 priors)
  // for (i) 4*(G_buy - V_buy), (ii) 4*(G_after - V_after), (iii) 4*(G_buy - (V_buy + A_buy))
  double r755_cat3_i_sum[9] = {0.0};
  double r755_cat3_ii_sum[9] = {0.0};
  double r755_cat3_iii_sum[9] = {0.0};
  int64_t r755_cat3_bucket_count[9] = {0};

  // Buy tracking: Imperium, Tleilaxu, Tech buys, unspent persuasion
  int64_t r755_imperium_buys = 0;
  int64_t r755_tleilaxu_buys = 0;
  int64_t r755_tech_buys = 0;
  int64_t r755_reveal_turns = 0;
  double r755_unspent_persuasion_sum = 0.0;

  std::vector<R765OverrideProbeRecord> r765_override_probes;
  std::vector<R765AgreedProbeRecord> r765_agreed_probes;
};

struct R755Gate2Diagnostics {
  int update = 0;
  int64_t total_transitions = 0;
  int64_t single_action_transitions = 0;
  int64_t multi_action_transitions = 0;
  int64_t override_masked_transitions = 0;
  int64_t ppo_multi_action_transitions = 0;
  bool mask_identity_holds = true;
  int64_t nan_inf_count = 0;
  double logprob_recompute_max_abs_diff = 0.0;
  double collector_vs_learner_mean_diff = 0.0;
  double collector_vs_learner_p99_diff = 0.0;
  double collector_vs_learner_max_diff = 0.0;
  double collector_vs_learner_kl = 0.0;
  double mean_student_kl = 0.0;
  double mean_target_kl = 0.0;
  double mean_delta_p_target = 0.0;
  double mean_delta_p_student = 0.0;
  double student_to_target_ratio = 0.0;
  double mean_non_override_kl = 0.0;
  double pre_step_logprob_mean_abs_diff = 0.0;
  double pre_step_logprob_p99_abs_diff = 0.0;
  double pre_step_logprob_max_abs_diff = 0.0;
  double pre_step_ratio_min = 1.0;
  double pre_step_ratio_p01 = 1.0;
  double pre_step_ratio_p05 = 1.0;
  double pre_step_ratio_p50 = 1.0;
  double pre_step_ratio_p95 = 1.0;
  double pre_step_ratio_p99 = 1.0;
  double pre_step_ratio_max = 1.0;
  double pre_step_clip_frac = 0.0;
  int64_t aux_distinct_visited = 0;
  int64_t aux_total_presentations = 0;
  int64_t visited_aux_count = 0;
  int64_t unvisited_aux_count = 0;
  double sum_delta_p_target = 0.0;
  double sum_delta_p_student = 0.0;
  double visited_sum_delta_p_target = 0.0;
  double visited_sum_delta_p_student = 0.0;
  double unvisited_sum_delta_p_target = 0.0;
  double unvisited_sum_delta_p_student = 0.0;
  double visited_student_to_target_ratio = 0.0;
  double unvisited_student_to_target_ratio = 0.0;
  int64_t searched_by_cat[6] = {0, 0, 0, 0, 0, 0};
  int64_t executed_overrides_by_cat[6] = {0, 0, 0, 0, 0, 0};
  double cat3_aux_distinct_visited_fraction = 0.0;
  double imperium_buy_rate = 0.0;
  double tleilaxu_buy_rate = 0.0;
  double tech_buy_rate = 0.0;
  double mean_unspent_persuasion = 0.0;
  double purchase_grad_cosine = 0.0;
  std::vector<double> cat3_i_avg;
  std::vector<double> cat3_ii_avg;
  std::vector<double> cat3_iii_avg;
  std::vector<int64_t> cat3_bucket_counts;
};

extern std::vector<R755Gate2Diagnostics> g_gate2_diagnostics;

struct Gate1AuditedRoot {
  int game_id = 0;
  int category = 0;
  int round = 0;
  int round_bucket = 0;
  std::unique_ptr<State> state_clone;
  Action raw_action = kInvalidAction;
  Action best_action = kInvalidAction;
  double delta_p = 0.0;
  std::vector<std::pair<Action, double>> raw_policy;
  int rollout_steps = 0;
  int rollout_nn_evals = 0;
  double inclusion_prob = 1.0;
  double delta_q = 0.0;
  double delta_v_exec = 0.0;
};

// --- Helpers (dune_ppo_train_helpers.cc) ---
void SyncModels(std::shared_ptr<SharedDunePolicyValueNetImpl> training_model,
                std::shared_ptr<SharedDunePolicyValueNetImpl> inference_model,
                std::shared_mutex* sync_mutex);
void SyncDualModels(
    std::shared_ptr<SharedDunePolicyValueNetImpl> training_actor,
    std::shared_ptr<SharedDunePolicyValueNetImpl> inference_actor,
    std::shared_ptr<SharedDunePolicyValueNetImpl> training_critic,
    std::shared_ptr<SharedDunePolicyValueNetImpl> inference_critic,
    std::shared_mutex* sync_mutex);
std::string HashNonValueParameters(const std::shared_ptr<SharedDunePolicyValueNetImpl>& model);
std::string HashAllModelState(const std::shared_ptr<SharedDunePolicyValueNetImpl>& model);
double GetExpectedGroupLearningRate(size_t group_idx, size_t total_groups, double base_lr, double new_block_lr);
void SetOptimizerLearningRate(torch::optim::Optimizer& optimizer, double lr, double new_block_lr = 0.0);
std::string SummarizeParamNames(const std::vector<std::string>& names);
void PrintOptimizerGroupSummary(
    const torch::optim::Optimizer& optimizer,
    const std::shared_ptr<SharedDunePolicyValueNetImpl>& model);
std::unique_ptr<torch::optim::AdamW> MakeOptimizer(
    std::shared_ptr<SharedDunePolicyValueNetImpl> model);
std::string GenerateUUID();
AcceptancePriorSource EffectiveAcceptancePriorSource();
std::string ComputeLegacyConfigFingerprint();
json::Object BuildPrePrecisionConfigFingerprintObject();
std::string ComputeCurrentPrePrecisionConfigFingerprint();
std::string ComputeConfigFingerprint();
std::string GetSearchLabelFingerprint(const std::string& search_label_dir);

// --- Transitions (dune_ppo_train_transitions.cc) ---
std::string ValidateAndComputeRewardTransitionFingerprint(
    const std::string& manifest_path, const std::string& current_config_fingerprint);
std::string ValidateAndComputeGaeTransitionFingerprint(
    const std::string& manifest_path, const std::string& current_config_fingerprint);
std::string ValidateAndComputeLearningRateTransitionFingerprint(
    const std::string& manifest_path, const std::string& current_config_fingerprint);
std::string ValidateAndComputeTargetKlTransitionFingerprint(
    const std::string& manifest_path, const std::string& current_config_fingerprint);
std::string ValidateAndComputeSingleLearnerTransitionFingerprint(
    const std::string& manifest_path, const std::string& current_config_fingerprint);
std::string ValidateAndComputeEmaMagnetTransitionFingerprint(
    const std::string& manifest_path, const std::string& current_config_fingerprint);
std::string ValidateAndComputePoolTransitionFingerprint(
    const std::string& manifest_path, const std::string& current_config_fingerprint);
std::string ValidateAndComputeMarketAppendixTransitionFingerprint(
    const std::string& manifest_path, const std::string& current_config_fingerprint);
std::string ValidateAndComputeSemanticSchemaTransitionFingerprint(
    const std::string& manifest_path, const std::string& current_config_fingerprint);
std::string ValidateAndComputeDeckPoolSemanticsTransitionFingerprint(
    const std::string& manifest_path, const std::string& current_config_fingerprint);
std::string ValidateAndComputeTf32TransitionFingerprint(
    const std::string& manifest_path, const std::string& current_config_fingerprint);

// --- Checkpoint (dune_ppo_train_checkpoint.cc) ---
void UpdateEmaWeights(
    const std::shared_ptr<SharedDunePolicyValueNetImpl>& model,
    std::shared_ptr<SharedDunePolicyValueNetImpl>& ema_model,
    double decay);
void SaveCheckpoint(
    std::shared_ptr<SharedDunePolicyValueNetImpl> model, torch::optim::AdamW& optimizer,
    const std::string& model_path, const std::string& optim_path, int global_update,
    int target_end_update, uint64_t total_env_steps, uint64_t next_episode_id,
    uint64_t base_seed, int seed_scheme_version, const std::string& config_fingerprint,
    const std::string& search_label_fingerprint, const std::string& run_uuid,
    const open_spiel::OnlineCollectionState* aux_state = nullptr,
    std::shared_ptr<SharedDunePolicyValueNetImpl> ema_model = nullptr,
    int64_t ema_updates_absorbed = 0, double ema_half_life = 0.0, double ema_decay = 0.0);
bool PublishFinalCheckpointFromExisting(
    const std::string& prefix, int update, bool separate_actor_critic);

// --- Simulation & Rollout (dune_ppo_train_sim.cc / _rollout.cc) ---
int PpoSimulation(uint64_t master, uint64_t episode_id, const Game& game,
                  std::shared_ptr<IGameEvaluator> evaluator, int64_t obs_size,
                  std::vector<PpoTransition>* trajectory,
                  std::atomic<uint64_t>* total_env_steps,
                  float reward_lambda,
                  WorkerStats* local_stats,
                  const std::vector<std::shared_ptr<IGameEvaluator>>& opponent_evaluators,
                  std::vector<open_spiel::dune_imperium::SearchSnapshot>* local_snapshots = nullptr,
                  uint64_t* decision_count = nullptr,
                  std::mt19937_64* reservoir_rng = nullptr,
                  std::vector<open_spiel::SearchTrainingExample>* r755_aux_examples = nullptr);
void RolloutWorker(int thread_id, const Game* game,
                   std::shared_ptr<IGameEvaluator> evaluator, int64_t obs_size,
                   PpoRolloutBuffer* rollout_buffer,
                   std::atomic<bool>* stop_collection,
                   std::atomic<uint64_t>* total_env_steps,
                   std::vector<WorkerStats>* worker_stats,
                   std::atomic<uint64_t>* local_episode_id,
                   uint64_t start_episode_id,
                   int rollout_games,
                   float reward_lambda,
                   const std::vector<std::shared_ptr<IGameEvaluator>>& opponent_evaluators,
                   std::vector<open_spiel::dune_imperium::SearchSnapshot>* local_snapshots = nullptr,
                   int global_update = 0,
                   std::vector<open_spiel::SearchTrainingExample>* worker_aux_examples = nullptr);
void AttachCanaryStats(open_spiel::PpoUpdateStats* stats, const CollectResult& collect);
void AttachPrecapAbszStats(open_spiel::PpoUpdateStats* stats, const CollectResult& collect);
CollectResult CollectRollout(const Game* game,
                             std::shared_ptr<IGameEvaluator> evaluator,
                             int64_t obs_size,
                             std::atomic<uint64_t>* total_env_steps,
                             int num_threads,
                             std::atomic<uint64_t>* next_episode_id,
                             int rollout_games,
                             float reward_lambda,
                             const std::vector<std::shared_ptr<IGameEvaluator>>& opponent_evaluators = {},
                             int global_update = 0);


// --- Gates (dune_ppo_train_gates.cc) ---
void WriteR755Gate2Json(const std::string& path, const std::vector<R755Gate2Diagnostics>& diag);
void RunR755Gate1Audit(
    const Game* game,
    std::shared_ptr<IGameEvaluator> evaluator,
    int64_t obs_size,
    uint64_t master_seed,
    int rollout_games,
    const std::string& output_path);

#endif  // OPEN_SPIEL_BUILD_WITH_LIBTORCH

}  // namespace open_spiel::ppo_train_internal

#include "dune_ppo_train_parity_internal.h"
