#ifndef OPEN_SPIEL_EXAMPLES_DUNE_PUCT_IS_MCTS_H_
#define OPEN_SPIEL_EXAMPLES_DUNE_PUCT_IS_MCTS_H_

#include <memory>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "open_spiel/abseil-cpp/absl/container/flat_hash_map.h"
#include "open_spiel/algorithms/mcts.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_bots.h"

namespace open_spiel {

enum class DuneISMCTSFinalPolicyType {
  kNormalizedVisitCount,
  kMaxVisitCount,
  kMaxValue,
};

// ---------------------------------------------------------------------------
// Reusable Types outside example binaries
// ---------------------------------------------------------------------------

enum class SearchOpponentMode {
  kMaxN = 0,
  kPolicy = 1
};

struct DuneSearchConfig {
  int max_simulations = 64;
  double relative_time_budget_ms = 10000.0; // relative search budget per move in ms
  int max_nodes = 50000;
  double puct_c = 0.15;
  SearchOpponentMode opponent_mode = SearchOpponentMode::kMaxN;
  double temperature = 1.0;
  double opponent_temperature = 1.0;
  int max_world_samples = -1;
  double utility_divisor = 4.0;
  int min_visit_threshold = 2;
  double covered_prior_threshold = 0.50;
  uint64_t seed = 42;
  DuneISMCTSFinalPolicyType final_policy_type = DuneISMCTSFinalPolicyType::kNormalizedVisitCount;

  // Dirichlet noise parameters (legacy/optional)
  double dirichlet_epsilon = 0.0;
  double dirichlet_alpha = 0.3;
  bool use_observation_string = true;
  bool verbose_diagnostics = false;
  bool check_strategic_state = false;
  double root_prior_temperature = 1.0;
  double training_root_prior_temperature = 1.0;

  // Session calibration parameters
  int fixed_continuation_reserve = 0;
  int purchase_combat_budget = 16;
  double live_continuation_reserve_seconds = 0.0;
  int fixed_session_limit = 200;
  std::string model_checkpoint_path = "";

  // Conservative override fields
  bool conservative_override_enabled = false;
  double conservative_covered_prior_threshold = 0.95;
  int conservative_meaningful_visit_threshold = 10;
  double conservative_q_margin_threshold = 0.03;
  double conservative_stability_checkpoint_fraction = 0.5;
  bool conservative_continuation_overrides_disabled = true;

  // Maximum multi-action player decisions traversed from the root. Negative
  // preserves uncapped behavior; chance and forced transitions do not count.
  int max_search_decision_depth = -1;
};

struct SearchDiagnostics {
  std::vector<Action> actions;       // All legal actions at root
  std::vector<int> visit_counts;     // N(a) per action
  std::vector<double> q_values;      // Empirical Q for covered, root_value for unsupported
  std::vector<double> priors;        // Neural prior π(a)
  double root_value = 0.0;           // Cached V(s) at root
  int total_root_visits = 0;
  int num_covered_actions = 0;       // Actions with visits >= min_visit_threshold
  double covered_prior_mass = 0.0;   // Sum of π(a) for covered actions

  // 18A metrics
  double max_depth = 0.0;
  double mean_depth = 0.0;
  double p95_depth = 0.0;
  int deepest_simulated_round = 0;
  double terminal_leaf_fraction = 0.0;
  int max_decision_depth = 0;
  double mean_decision_depth = 0.0;
  int unique_nodes = 0;
  int inference_count = 0;
  double raw_to_search_policy_kl = 0.0;
  double chosen_action_raw_prior_probability = 0.0;
  int chosen_action_raw_prior_rank = 0;
  std::vector<int> forced_visit_counts;
  std::vector<int> pruned_visit_counts;
  bool action_changed_vs_raw_argmax = false;
  std::vector<std::vector<Action>> leaf_histories;
  // In-memory sampled determinized leaves for fidelity diagnostics. Histories
  // alone cannot always be replayed through a different hidden-world sample.
  std::vector<std::shared_ptr<State>> sampled_leaf_states;

  // Telemetry fields
  std::string protocol_version = "v2";
  std::string session_id = "";
  int searched_seat = -1;
  int round = -1;
  std::string phase = "";
  std::string decision_role = "";
  std::string budget_mode = "";
  int hard_sim_limit = 0;
  int soft_sim_limit = 0;
  double hard_time_limit_ms = 0.0;
  double soft_time_limit_ms = 0.0;
  double elapsed_search_time_ms = 0.0;
  double observation_wait_time_ms = 0.0;
  int inherited_root_visits = 0;
  int newly_completed_simulations = 0;
  int session_cumulative_simulations = 0;
  int short_window_cumulative_simulations = 0;
  double session_cumulative_search_time_ms = 0.0;
  double long_agent_session_cumulative_time_ms = 0.0;
  std::string re_root_status = "none"; // "hit", "miss", "none"
  bool post_chance_branch_miss = false;
  double root_coverage = 0.0;
  std::string reset_reason = "none";
  int tree_node_count = 0;
  int inference_count_telemetry = 0;
  Action selected_action = -1;
  bool legality_result = true;
  std::string fallback_reason = "none";

  // Centralized controller selection and conservative override telemetry
  Action raw_reference_action = -1;
  Action mcts_proposed_action = -1;
  bool confidence_fallback = false;
  bool mcts_overrode_raw = false;
  Action stability_checkpoint_action = -1;
  bool stability_checkpoint_reached = false;
  bool stability_agreement = false;
  bool pass_complete_search = false;
  bool pass_min_actions = false;
  bool pass_prior_mass = false;
  bool pass_meaningful_visits = false;
  bool pass_q_margin = false;
  bool pass_stability = false;
};

struct DuneSearchResult {
  ActionsAndProbs policy;
  SearchDiagnostics diagnostics;
  int simulations_completed = 0;
  double elapsed_time_ms = 0.0;
  bool timeout_status = false;
  bool used_fallback = false;
  std::string fallback_reason = "none"; // "none", "timeout", "low_coverage", "max_nodes"
  int inference_count = 0; // Track NN evaluator inferences
};

struct SearchTrainingExample {
  std::vector<float> observation;
  Player player = kInvalidPlayer;
  std::vector<Action> legal_actions;
  std::vector<double> normalized_visit_target;
  double normalized_terminal_target = 0.0;
  std::string checkpoint_hash;
  int64_t update_id = 0;
  int64_t episode_id = 0;
  int64_t decision_id = 0;
  SearchDiagnostics diagnostics;
};

// Strategic-state classification function declarations
bool IsStrategicAction(const std::string& action_str);
bool IsStrategicState(const State& state, Player searched_player);

// ---------------------------------------------------------------------------
// Bot structures
// ---------------------------------------------------------------------------

struct DuneChildInfo {
  int visits = 0;
  double return_sum = 0.0;
  double prior = 0.0;
  double value() const { return visits > 0 ? return_sum / visits : 0.0; }
};

struct DuneISMCTSNode {
  absl::flat_hash_map<Action, DuneChildInfo> child_info;
  int total_visits = -1;
  bool priors_initialized = false;
  double cached_value = 0.0;  // Neural V(s) for this node's current player
  std::vector<double> cached_values; // Neural V(s) for all players
  bool dirichlet_noise_applied = false;
};

struct TestBotAccessor;

class DunePUCTISMCTSBot : public Bot {
  friend struct TestBotAccessor;
  friend class DuneSearchSession;
 public:
  DunePUCTISMCTSBot(
      const DuneSearchConfig& config,
      const std::vector<std::shared_ptr<algorithms::Evaluator>>& evaluators);

  DunePUCTISMCTSBot(
      const DuneSearchConfig& config,
      std::shared_ptr<algorithms::Evaluator> evaluator)
      : DunePUCTISMCTSBot(config, std::vector<std::shared_ptr<algorithms::Evaluator>>(4, evaluator)) {}

  DunePUCTISMCTSBot(uint64_t seed, std::shared_ptr<algorithms::Evaluator> evaluator,
                    double puct_c, int max_simulations,
                    int max_world_samples = -1,
                    double temperature = 1.0,
                    double dirichlet_epsilon = 0.0,
                    double dirichlet_alpha = 0.3,
                    double utility_divisor = 4.0,
                    bool use_observation_string = true,
                    DuneISMCTSFinalPolicyType final_policy_type = DuneISMCTSFinalPolicyType::kNormalizedVisitCount,
                    bool use_opponent_model = false,
                    double opponent_temperature = 0.0,
                    bool verbose_diagnostics = false);

  Action Step(const State& state) override;
  bool ProvidesPolicy() override { return true; }
  ActionsAndProbs GetPolicy(const State& state) override;
  std::pair<ActionsAndProbs, Action> StepWithPolicy(const State& state) override;

  DuneSearchResult RunSearch(const State& state, int max_sims = -1, double max_time_ms = -1.0, int start_sim_index = 0);
  SearchDiagnostics GetRootDiagnostics(const State& state, int min_visit_threshold, Action chosen_action = kInvalidAction) const;
  const DuneSearchResult& GetLastSearchResult() const;

  void Restart() override { Reset(); }
  void RestartAt(const State& state) override { Reset(); }

  DuneSearchConfig& GetConfig() { return config_; }
  const DuneSearchConfig& GetConfig() const { return config_; }

  std::pair<Player, std::string> GetStateKey(const State& state) const;
  const absl::flat_hash_map<std::pair<Player, std::string>, DuneISMCTSNode*>& nodes() const { return nodes_; }

 private:
  void Reset();
  double RandomNumber();
  std::unique_ptr<State> SampleRootState(const State& state, int sim_index);
  std::unique_ptr<State> ResampleFromInfostate(const State& state, int sim_index);
  DuneISMCTSNode* LookupOrCreateNode(const State& state);
  DuneISMCTSNode* CreateNewNode(const State& state);
  DuneISMCTSNode* LookupNode(const State& state);

  // Core search procedures
  std::vector<double> RunSimulation(State* state, int depth,
                                    int decision_depth, int sim_index);
  std::vector<double> EvaluateCappedLeaf(const State& state, int depth,
                                         int decision_depth);
  void RecordSampledLeaf(const State& state);
  Action SelectActionTreePolicy(DuneISMCTSNode* node, const std::vector<Action>& legal_actions);
  void InitializePriorsAndValue(DuneISMCTSNode* node, const State& state);
  ActionsAndProbs FilterAndNormalizePriors(DuneISMCTSNode* node, const std::vector<Action>& legal_actions) const;
  ActionsAndProbs GetFinalPolicy(const State& state, DuneISMCTSNode* node) const;

  std::mt19937 rng_;
  DuneSearchConfig config_;
  std::vector<std::shared_ptr<algorithms::Evaluator>> evaluators_;

  Player searching_player_ = kInvalidPlayer;
  int max_depth_this_search_ = 0;
  double sum_depth_this_search_ = 0.0;
  int num_sims_this_search_ = 0;
  int max_decision_depth_this_search_ = 0;
  double sum_decision_depth_this_search_ = 0.0;
  int total_lookups_ = 0;
  int reused_lookups_ = 0;
  int search_count_ = 0;
  bool is_continuation_ = false;

  // 18A diagnostics tracking fields
  std::vector<int> simulation_depths_this_search_;
  int terminal_leaf_simulations_count_ = 0;
  int max_round_this_search_ = 0;

  absl::flat_hash_map<std::pair<Player, std::string>, DuneISMCTSNode*> nodes_;
  std::vector<std::unique_ptr<DuneISMCTSNode>> node_pool_;
  std::vector<std::unique_ptr<State>> root_samples_;
  DuneISMCTSNode* root_node_;

  // Reusable Tree Session fields
  Player last_searching_player_ = kInvalidPlayer;
  std::vector<std::shared_ptr<algorithms::Evaluator>> last_evaluators_;

  DuneSearchResult last_search_result_;
  int inference_count_this_search_ = 0;
  absl::flat_hash_map<std::pair<Player, std::string>, ActionsAndProbs> opponent_prior_cache_;
  absl::flat_hash_set<std::pair<Player, std::string>> visited_nodes_this_search_;
  bool in_session_ = false;
  bool has_deadline_ = false;
  std::chrono::steady_clock::time_point deadline_;
  std::vector<std::vector<Action>> current_search_leaf_histories_;
  std::vector<std::shared_ptr<State>> current_search_sampled_leaf_states_;
  uint64_t diagnostic_leaf_states_seen_ = 0;
};

}  // namespace open_spiel

#endif  // OPEN_SPIEL_EXAMPLES_DUNE_PUCT_IS_MCTS_H_
