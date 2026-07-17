#ifndef OPEN_SPIEL_EXAMPLES_DUNE_SEARCH_SESSION_H_
#define OPEN_SPIEL_EXAMPLES_DUNE_SEARCH_SESSION_H_

#include <memory>
#include <string>
#include <vector>
#include <chrono>
#include <random>
#include "open_spiel/spiel.h"
#include "dune_puct_is_mcts.h"
#include "dune_search_routing.h"

namespace open_spiel {

enum class DuneSearchBudgetMode {
  kPolicyOnly = 0,
  kFixedSessionSimulations = 1,
  kTrainingFullFast = 2,
  kLiveDeadline = 3
};

struct ControllerDecision {
  Action selected_action = kInvalidAction;
  Action raw_reference_action = kInvalidAction;
  Action mcts_proposed_action = kInvalidAction;
  bool confidence_fallback = false;
  bool mcts_overrode_raw = false;
  bool stability_checkpoint_reached = false;
  bool stability_agreement = false;

  // Per-criterion pass/fail fields
  bool pass_complete_search = false;
  bool pass_min_actions = false;
  bool pass_prior_mass = false;
  bool pass_meaningful_visits = false;
  bool pass_q_margin = false;
  bool pass_stability = false;
};

// Helper function to sample an action from a policy prior.
Action SampleActionFromPrior(const ActionsAndProbs& prior, double r_val);

class DuneSearchSession {
 public:
  DuneSearchSession(
      const DuneSearchConfig& config,
      std::shared_ptr<algorithms::Evaluator> evaluator,
      DuneSearchBudgetMode budget_mode);

  DuneSearchSession(
      const DuneSearchConfig& config,
      const std::vector<std::shared_ptr<algorithms::Evaluator>>& evaluators,
      DuneSearchBudgetMode budget_mode);

  DuneSearchResult Search(const State& state, double remaining_time_ms = -1.0);

  ControllerDecision SelectControllerAction(
      const State& state,
      const DuneSearchResult& search_result,
      double r_val);

  DuneSearchResult CommitAction(const ControllerDecision& decision);

  void DiscardPendingAction();

  DuneSearchResult SearchAndSelect(const State& state);
  DuneSearchResult SearchAndSelect(const State& state, double r_val);

  // Setters/Getters for session metadata/diagnostics
  void SetEpisodeId(int episode_id) { episode_id_ = episode_id; }
  void SetUpdateId(int update_id) { update_id_ = update_id; }
  void ResetSession(const std::string& reason);
  void HandleReRootMismatch(const std::string& reason);
  bool HasActiveSession() const { return has_active_session_; }
  std::shared_ptr<DunePUCTISMCTSBot> GetBot() const { return placement_bot_; }
  std::shared_ptr<DunePUCTISMCTSBot> GetShortBot() const { return short_bot_; }

  int session_new_simulations_completed() const { return session_new_simulations_completed_; }
  int short_sims_completed() const { return short_sims_completed_; }
  double session_elapsed_time_ms() const { return session_elapsed_time_ms_; }
  bool is_full_session() const { return is_full_session_; }
  std::string last_re_root_status() const { return last_re_root_status_; }
  std::string intermediate_re_root_status() const { return intermediate_re_root_status_; }
  void SetLastRequestedMaxSimsForTesting(int sims) { last_requested_max_sims_ = sims; }

 private:
  DuneSearchConfig config_;
  std::vector<std::shared_ptr<algorithms::Evaluator>> evaluators_;
  DuneSearchBudgetMode budget_mode_;

  std::shared_ptr<DunePUCTISMCTSBot> placement_bot_;
  std::shared_ptr<DunePUCTISMCTSBot> short_bot_;

  // Session state tracking
  bool has_active_session_ = false;
  std::string session_id_ = "";
  Player active_player_ = kInvalidPlayer;
  int round_ = -1;
  int episode_id_ = 0;
  int update_id_ = 0;
  int decision_id_ = 0;

  std::mt19937 rng_;

  // Cumulative budget tracking
  int session_new_simulations_completed_ = 0;
  double session_elapsed_time_ms_ = 0.0;
  double long_agent_elapsed_time_ms_ = 0.0;
  int cumulative_simulations_ = 0;

  // Short window tracking
  DuneDecisionRole last_role_ = DuneDecisionRole::kForcedOrBookkeeping;
  std::chrono::steady_clock::time_point short_window_start_time_;
  int short_sims_completed_ = 0;

  // Training Full/Fast choice (rolled once per placement activation)
  bool is_full_session_ = false;
  bool training_full_fast_rolled_ = false;

  // Re-routing hit/miss status
  std::string last_re_root_status_ = "none";
  std::string intermediate_re_root_status_ = "none";
  bool post_chance_branch_miss_ = false;

  // Clock tracking for live deadline mode
  std::chrono::steady_clock::time_point absolute_live_deadline_;
  bool live_deadline_initialized_ = false;

  std::vector<Action> session_history_;
  std::vector<Action> last_input_history_;

  std::string last_reset_reason_ = "none";

  // Commit lifecycle state
  bool has_pending_commit_ = false;
  std::unique_ptr<State> last_search_state_;
  DuneSearchResult last_search_result_;
  int last_requested_max_sims_ = 0;
};

// Factory helper to construct neural network evaluator from checkpoint
std::shared_ptr<algorithms::Evaluator> MakeDuneNNEvaluator(
    const std::string& checkpoint_path,
    const std::string& device_str,
    int hidden_dim = 1024,
    int num_blocks = 4);

void LoadCalibratedParameters(DuneSearchConfig& config);

} // namespace open_spiel

#endif // OPEN_SPIEL_EXAMPLES_DUNE_SEARCH_SESSION_H_
