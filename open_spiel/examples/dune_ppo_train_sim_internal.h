#pragma once

#include "dune_ppo_train_internal.h"

namespace open_spiel::ppo_train_internal {

#ifdef OPEN_SPIEL_BUILD_WITH_LIBTORCH

struct SimStepDecision {
  Player current_player = kInvalidPlayer;
  bool is_learner = false;
  std::vector<Action> actions;
  std::vector<float> obs;
  dune_semantic::CandidateActionData cand_data;
  const dune_semantic::CandidateActionData* p_cand_data = nullptr;
  EvalResult eval_result;
  std::vector<double> legal_probabilities;
  Action action = kInvalidAction;
  float old_log_prob = 0.0f;
  bool leader_draft_overridden = false;
  bool is_forced_buy = false;
  bool is_targeted_forced_buy = false;
  int targeted_buy_card_id = 0;
  bool r755_override_accepted = false;

  bool parity_capture = false;
  int parity_decision_role = 0;
  std::vector<float> parity_raw_legal_logits;
  std::vector<float> parity_behavior_log_probs;

  bool is_atomics_action = false;
  bool atomics_exempt = false;
  bool is_plot_intrigue_action = false;
  bool plot_intrigue_exempt = false;
  std::vector<uint16_t> pre_action_privileged_appendix;
  bool pre_action_is_high_stakes = false;
  bool post_action_is_high_stakes = false;
};

struct PpoSimContext {
  uint64_t master = 0;
  uint64_t episode_id = 0;
  const Game* game = nullptr;
  std::shared_ptr<IGameEvaluator> evaluator;
  int64_t obs_size = 0;
  std::vector<PpoTransition>* trajectory = nullptr;
  std::atomic<uint64_t>* total_env_steps = nullptr;
  float reward_lambda = 1.0f;
  WorkerStats* local_stats = nullptr;
  const std::vector<std::shared_ptr<IGameEvaluator>>* opponent_evaluators = nullptr;
  std::vector<open_spiel::dune_imperium::SearchSnapshot>* local_snapshots = nullptr;
  uint64_t* decision_count = nullptr;
  std::mt19937_64* reservoir_rng = nullptr;
  std::vector<open_spiel::SearchTrainingExample>* r755_aux_examples = nullptr;

  bool single_learner = false;
  dune_opponent_pool::SeatAssignment seat_assignment;
  int learner_seat = -1;

  std::mt19937_64 chance_rng;
  std::mt19937_64 policy_rng[4];
  std::mt19937_64 explore_rng;

  bool do_purchase_exploration = false;
  int purchase_exploration_seat = -1;
  int forced_round_1 = -1;
  int forced_round_2 = -1;
  bool forced_round_1_done = false;
  bool forced_round_2_done = false;
  int designated_explore_seat = 0;
  bool episode_forced_card11 = false;
  bool episode_forced_card13 = false;

  bool r755_hybrid = false;
  open_spiel::dune_imperium::TurnSearchTree r755_turn_trees[4];
  bool r755_space_placed_this_turn[4] = {false, false, false, false};
  bool r755_turn_coin_active[4] = {false, false, false, false};
  Player r755_last_turn_player = open_spiel::kInvalidPlayer;
  Action r755_last_action = open_spiel::kInvalidAction;
  int64_t r755_decision_index = 0;
  std::array<open_spiel::dune_imperium::PendingJointAgentOverride, 4> r765_pending_joint;
  std::array<bool, 4> r765_saw_chance_since_root = {false, false, false, false};
  std::vector<std::shared_ptr<open_spiel::BatchedNNEvaluator>> r755_search_evals;

  std::unique_ptr<State> state;
  const dune_imperium::DuneImperiumState* dune_state = nullptr;
  bool provides_info_state_tensor = false;
  bool provides_observations_tensor = false;

  std::vector<int> current_tleilaxu;
  std::vector<bool> had_swordmaster;
  std::vector<int> prev_conflict_vp;
  std::vector<int> prev_total_vp;
  std::vector<int> prev_hand_sb;
  std::vector<int> prev_hand_sh;
  CombatCreditAccumulator combat_accumulator;
  std::vector<int> last_transition_index;
  std::vector<int> pre_combat_strength;
  dune_imperium::GamePhase pre_action_phase = dune_imperium::GamePhase::kDeal;
  bool pending_horror_trash[4] = {false, false, false, false};

  float logit_cap = 0.0f;
  double shaped_weight = 0.0;
  double tleilaxu_breadcrumb_weight = 0.0;
  double tleilaxu_level7_breadcrumb_weight = 0.0;
  double specimen_exchange_penalty = 0.0;
  double family_atomics_penalty = 0.0;
  double plot_intrigue_penalty = 0.0;
  int plot_intrigue_exemption_threshold = 0;

  // PWO-5 section 14.1b: PRE-cap (SAT, role) and POST-cap (NE, max_action_prob)
  // state carried across CenterAndCapLegalLogits on the same decision.
  bool emit_canary_columns = false;
  bool canary_measure_this_decision = false;
  int canary_role_this_decision = 0;

  int game_length = 0;
  int counterfactual_samples = 0;
  std::vector<CounterfactualPending> pending_cf;
  Player last_player = kInvalidPlayer;
  Action last_action = kInvalidAction;
};

// --- Step evaluation, search override, sampling & transition recording (dune_ppo_train_sim_step.cc) ---
void EvaluateAndSelectStepAction(PpoSimContext* ctx, SimStepDecision* step);
void RecordLearnerTransition(PpoSimContext* ctx, SimStepDecision* step);

// --- Tracking, diagnostics & reward shaping (dune_ppo_train_sim_tracking.cc) ---
void CheckHandDraws(PpoSimContext* ctx);
void TrackTwoCardDecisions(PpoSimContext* ctx, const SimStepDecision& step);
void ValidateParityCapturePostStep(PpoSimContext* ctx, const SimStepDecision& step);
void CapturePreActionTrackingAndUpdateTrees(PpoSimContext* ctx, SimStepDecision* step);
void RecordPostActionHighStakesAndSwordmaster(PpoSimContext* ctx, SimStepDecision* step);
void ApplyStepTrackingAndShaping(PpoSimContext* ctx, const SimStepDecision& step);
void MaybeRecordR765AgreedProbe(PpoSimContext* ctx, const SimStepDecision& step,
                                Action greedy_action);
void EmitR765PairwiseMpoAuxAndProbe(PpoSimContext* ctx, const SimStepDecision& step,
                                    uint64_t root_uid, const char* probe_category,
                                    const open_spiel::ActionsAndProbs& raw_policy_pairs,
                                    Action a_ref, double q_ref,
                                    Action a_star, double q_star);
void EmitR755CompoundAuxAndProbe(
    PpoSimContext* ctx, const SimStepDecision& step, uint64_t root_uid,
    open_spiel::dune_imperium::R755SearchCategory cat,
    const open_spiel::ActionsAndProbs& raw_policy_pairs,
    const open_spiel::dune_imperium::CompoundSearchDecisionResult& search_res);
void DumpR755RootJsonl(
    uint64_t episode_id, int64_t decision_index, int cat_idx,
    const open_spiel::dune_imperium::CompoundSearchDecisionResult& search_res,
    double max_raw_p, double p_2);

// --- Counterfactual successor sampling (dune_ppo_train_sim.cc) ---
void SampleCounterfactualSuccessor(PpoSimContext* ctx, const SimStepDecision& step);

#endif  // OPEN_SPIEL_BUILD_WITH_LIBTORCH

}  // namespace open_spiel::ppo_train_internal
