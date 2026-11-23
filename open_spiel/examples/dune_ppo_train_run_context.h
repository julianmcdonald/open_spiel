#pragma once

#include <optional>

#include "dune_ppo_train_internal.h"

namespace open_spiel::ppo_train_internal {

#ifdef OPEN_SPIEL_BUILD_WITH_LIBTORCH

struct KeptLabelsRecord {
  int update_id = 0;
  std::vector<open_spiel::SearchTrainingExample> examples;
  double ce_before = 0.0;
  double ce_after = 0.0;
};

struct R765AbsorptionUpdateRecord {
  int update = 0;
  std::vector<R765OverrideProbeRecord> probes;
  double mean_t2_live_delta = 0.0;
};

struct R765CollateralPanelItem {
  std::vector<float> obs;
  std::vector<Action> legal_actions;
  dune_semantic::CandidateActionData candidate_data;
  int baseline_greedy_action = -1;
  bool is_hp65 = false;
};

struct PpoTrainStepState {
  int update = 0;
  std::chrono::high_resolution_clock::time_point wall_start{};
  CollectResult next_collect;
  std::thread bg_collect_thread;
  bool have_bg = false;

  open_spiel::dune_imperium::SearchPairingMode pairing_mode =
      open_spiel::dune_imperium::SearchPairingMode::kUnpaired;
  open_spiel::dune_imperium::SearchTruncationMode truncation_mode =
      open_spiel::dune_imperium::SearchTruncationMode::kTerminal;

  std::chrono::high_resolution_clock::time_point ppo_start{};
  open_spiel::dune_imperium::SearchSupervisionResult search_sup_res;
  double this_search_coef = 0.0;
  open_spiel::PpoUpdateStats stats;
  double ppo_elapsed = 0.0;

  double search_kl_sum = 0.0;
  double search_grad_sum = 0.0;
  double search_lambda = 0.0;
  int search_minibatches_per_update = 0;
  std::vector<open_spiel::SearchTrainingExample> train_aux_examples;

  bool is_checkpoint = false;
  bool should_break = false;

  ~PpoTrainStepState() {
    if (bg_collect_thread.joinable()) {
      bg_collect_thread.join();
    }
  }
};

#endif  // OPEN_SPIEL_BUILD_WITH_LIBTORCH

struct PpoTrainRunContext {
  std::string parity_command_line;
  bool numerical_parity = false;

#ifdef OPEN_SPIEL_BUILD_WITH_LIBTORCH
  uint64_t master = 0;
  std::shared_ptr<const Game> game;
  int64_t obs_size = 0;
  int64_t action_size = 0;
  torch::Device device{torch::kCPU};
  PpoNumericalParitySourceProvenance parity_source_provenance;

  bool enable_scorer = false;
  bool enable_zone_enc = false;
  std::string sem_schema;
  dune_imperium::DeckPoolSemantics resolved_target_deck_pool_semantics =
      dune_imperium::DeckPoolSemantics::kLegacy;

  std::shared_ptr<open_spiel::SharedDunePolicyValueNetImpl> training_model;
  std::shared_ptr<open_spiel::SharedDunePolicyValueNetImpl> inference_model;
  std::shared_ptr<open_spiel::SharedDunePolicyValueNetImpl> ema_model;
  int64_t ema_updates_absorbed = 0;
  double ema_half_life = 0.0;
  double ema_decay = 0.0;

  std::shared_ptr<open_spiel::SharedDunePolicyValueNetImpl> training_critic;
  std::shared_ptr<open_spiel::SharedDunePolicyValueNetImpl> inference_critic;
  std::unique_ptr<torch::optim::AdamW> optimizer;
  std::unique_ptr<torch::optim::AdamW> search_aux_optimizer;
  std::unique_ptr<torch::optim::AdamW> critic_optimizer;

  std::shared_ptr<open_spiel::SharedDunePolicyValueNetImpl> collection_model;
  std::shared_ptr<open_spiel::SharedDunePolicyValueNetImpl> anchor_model;

  std::string init_mode;
  std::string config_fingerprint;
  std::string pre_precision_config_fingerprint;
  std::string search_label_fingerprint;
  std::string run_uuid;
  std::string gate_session_nonce;

  std::atomic<uint64_t> total_env_steps{0};
  std::atomic<uint64_t> next_episode_id{0};

  // Phase 18B online-collection persistent state.
  bool online_search_collection = false;
  uint64_t aux_next_episode_id_persist = 0;
  int64_t aux_cum_accepted = 0;
  int64_t aux_cum_rejected = 0;
  int64_t aux_cum_role_searches[open_spiel::OnlineCollectionState::kNumSearchRoles] =
      {0, 0, 0, 0};
  int64_t aux_cum_role_accepted[open_spiel::OnlineCollectionState::kNumSearchRoles] =
      {0, 0, 0, 0};
  int64_t aux_cum_granted = 0;
  int64_t aux_cum_organic = 0;
  std::string aux_hash_chain;
  std::shared_ptr<open_spiel::OnlineSearchCollector> aux_collector;
  std::shared_ptr<open_spiel::DuneNNEvaluator> aux_evaluator;
  float aux_logit_cap = 0.0f;
  double aux_search_loss_coef_target = 0.0;
  int aux_search_loss_warmup = 0;
  double aux_abort_ratio = 0.0;
  open_spiel::OnlineSearchConfig aux_config;

  int start_update = 1;
  int target_end_update = 0;
  std::string parity_source_manifest;
  std::string parity_source_manifest_sha256;
  int64_t parity_source_global_update = -1;
  std::string parity_source_checkpoint_uuid;

  std::shared_mutex sync_mutex;
  std::mutex eval_mutex;
  std::shared_ptr<open_spiel::IGameEvaluator> evaluator;
  std::vector<std::shared_ptr<open_spiel::SharedDunePolicyValueNetImpl>>
      opponent_models;
  std::vector<std::shared_ptr<open_spiel::IGameEvaluator>> opponent_evaluators;
  std::vector<std::string> opponent_hashes_before;

  uint64_t total_games = 0;
  uint64_t total_moves = 0;
  std::chrono::high_resolution_clock::time_point training_start{};
  double base_lr = 0.0;
  bool pipeline = false;
  int num_threads = 64;

  open_spiel::SearchLabelBuffer search_buffer;
  std::string search_label_dir;

  std::vector<std::shared_ptr<SharedDunePolicyValueNetImpl>> search_teacher_models;
  std::vector<std::unique_ptr<std::shared_mutex>> search_teacher_mutexes;
  std::vector<std::shared_ptr<open_spiel::BatchedEvaluator>> search_teacher_coords;
  std::vector<std::shared_ptr<open_spiel::BatchedNNEvaluator>>
      search_teacher_evaluators;

  int rollout_games = 0;
  std::string initial_non_value_hash;
  float reward_lambda = 0.0f;
  CollectResult current_collect;

  dune_imperium::CardMetrics cum_card11;
  dune_imperium::CardMetrics cum_card13;
  std::string card_telemetry_path;
  std::string cutoff_str;
  double cutoff_epoch = 0.0;
  int last_completed_update = 0;
  bool final_checkpoint_saved = false;

  std::deque<KeptLabelsRecord> kept_labels_history_25u;
  std::deque<R765AbsorptionUpdateRecord> r765_absorption_history;
  open_spiel::R765ReplayCohortManager r765_replay_mgr;
  open_spiel::R765PreserveWindowGuard r765_preserve_guard;
  std::vector<R765CollateralPanelItem> r765_collateral_panel;
  open_spiel::V5LegacySlotResidualState v5_slot_residual_state;

  // Cross-TU helpers implemented in phase files.
  void RunAuxCollection(int update_id, CollectResult& cr);
  void AccountAuxConsumed(const CollectResult& cr);
  open_spiel::OnlineCollectionState BuildAuxState() const;
  std::string ResolveInputSidecarPath(const std::string& filename) const;
  void SaveAuxiliarySidecars(int ckpt_update);
  void WriteVerificationSidecarIfEnabled(int update_id,
                                         const CollectResult& collect,
                                         const open_spiel::PpoUpdateStats& stats) const;
#endif  // OPEN_SPIEL_BUILD_WITH_LIBTORCH
};

std::optional<int> RunStartupPhase(PpoTrainRunContext& ctx, int argc, char** argv);

#ifdef OPEN_SPIEL_BUILD_WITH_LIBTORCH
void RunResumePhase(PpoTrainRunContext& ctx);
void RunBootstrapPhase(PpoTrainRunContext& ctx);
void RunSetupPhase(PpoTrainRunContext& ctx);
std::optional<int> RunSetupPreloopPhase(PpoTrainRunContext& ctx);
void RunLoopUpdatePhase(PpoTrainRunContext& ctx, PpoTrainStepState& step);
void RunLoopAuxPhase(PpoTrainRunContext& ctx, PpoTrainStepState& step);
void RunLoopPostPhase(PpoTrainRunContext& ctx, PpoTrainStepState& step);
void RunShutdownPhase(PpoTrainRunContext& ctx);

std::vector<std::vector<double>> EvalModelLegalProbs(
    const PpoTrainRunContext& ctx,
    const std::shared_ptr<SharedDunePolicyValueNetImpl>& eval_m,
    const std::vector<const std::vector<float>*>& obs_ptrs,
    const std::vector<const dune_semantic::CandidateActionData*>& cads,
    const std::vector<const std::vector<Action>*>& legals_list,
    size_t max_chunk_size = 2048);
#endif  // OPEN_SPIEL_BUILD_WITH_LIBTORCH

}  // namespace open_spiel::ppo_train_internal
