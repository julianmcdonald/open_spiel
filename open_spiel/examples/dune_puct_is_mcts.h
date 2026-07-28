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
  // Visits an action needs before it counts as "covered" by the search (and
  // before its empirical Q is trusted over the root value). See
  // EffectiveMinVisitThreshold: values below 1 are meaningless — a zero-visit
  // action has no empirical Q — and are clamped to 1 rather than honored.
  int min_visit_threshold = 2;
  double covered_prior_threshold = 0.50;
  uint64_t seed = 42;
  DuneISMCTSFinalPolicyType final_policy_type = DuneISMCTSFinalPolicyType::kNormalizedVisitCount;

  // Dirichlet noise parameters (legacy/optional)
  double dirichlet_epsilon = 0.0;
  double dirichlet_alpha = 0.3;

  // --- KataGo root-exploration package (Phase 18B; §3.1-3.2 of 1902.10565). ---
  // Activated only when dirichlet_epsilon > 0 (i.e. at "noised roots"). Implement
  // as a UNIT: noise without target pruning pollutes CE targets at low budgets.
  // Per-root Dirichlet concentration: when > 0, the effective alpha at a root is
  // dirichlet_alpha_total / N_legal (KataGo's inverse-legal-count scaling),
  // overriding the fixed dirichlet_alpha. 0 keeps the legacy fixed alpha.
  double dirichlet_alpha_total = 0.0;
  // Forced-playout coefficient k: when > 0, each root child c is forced to at
  // least sqrt(k * P_noised(c) * N_root) visits by treating its PUCT urgency as
  // infinite while under quota. 0 disables forced playouts. KataGo uses k = 2.
  double forced_playouts_k = 0.0;
  // First-play-urgency at the noised root: when true, unvisited root children use
  // FPU = 0 instead of the node's cached value (KataGo footnote 3). Root only.
  bool root_noise_fpu_zero = false;
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
  std::vector<double> priors;        // Tree prior π(a) at root — POST-noise when noised
  // The UNTRANSFORMED network prior, aligned to `actions` and normalized over
  // the same legal-action set as `priors`. This is the baseline the item-4
  // per-role KL telemetry is defined against, so it strips BOTH root
  // transformations the tree prior may carry: `root_prior_temperature` scaling
  // and the Dirichlet mixture. Populated whenever the root is present in the
  // search tree; empty only when it is not (in which case `priors` was built
  // straight from the evaluator and is already the raw prior).
  //
  // WO-15: this used to be populated only when Dirichlet noise ran, and it
  // snapshotted the POST-temperature prior, so under a non-unit
  // `root_prior_temperature` the "raw" baseline was not raw. Consumers keep the
  // `raw_priors.empty() ? priors : raw_priors` idiom; it is now exact.
  std::vector<double> raw_priors;
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
  // Session-level budget/deadline limit that was hit (if any), kept SEPARATE
  // from fallback_reason so the session no longer destroys RunSearch's own
  // reason (e.g. "low_coverage") when clamping simulations to zero.
  std::string budget_limit_reason = "none";

  // Centralized controller selection and conservative override telemetry
  Action raw_reference_action = -1;
  Action mcts_proposed_action = -1;
  bool confidence_fallback = false;
  bool mcts_overrode_raw = false;
  Action stability_checkpoint_action = -1;
  bool stability_checkpoint_reached = false;
  bool stability_agreement = false;

  // PWO-3 (docs/PWO3_REGISTRATION.md section 4.2). TELEMETRY ONLY: recorded at
  // checkpoints already visited by the loop, drawing no RNG and mutating no tree
  // state.
  //
  // The coverage-gate INPUTS at the half-budget checkpoint. PWO-2 recorded only
  // stability_checkpoint_action, and never emitted even that. On a FIXED tier
  // this is behaviour-neutral in the strong sense -- the search stops on a COUNT,
  // so the snapshot's cost cannot change the result -- and section 6.1.1 proves
  // that on all 384 Branch-A fixed_800 rows.
  int stability_checkpoint_num_covered_actions = 0;
  double stability_checkpoint_covered_prior_mass = 0.0;
  // The half-TIME checkpoint. A live tier's budget is TIME, not simulations: its
  // max_simulations is 100000, while 0 of PWO-2's 384 ISOLATED live_50 rows
  // reached 50,000 simulations (max 40,951), so the sim-count checkpoint is
  // unreached on EVERY live row and "half budget" there means half the deadline.
  //
  // NOT bitwise-neutral, and registration 14.3a says so: a live search stops on
  // the CLOCK, so this snapshot's cost is time not spent simulating and can shift
  // simulations_completed slightly. It fires at most once per search and costs one
  // prior pass plus an argmax -- order 1e-6 of a 50 s deadline -- and it cannot
  // change the searched TREE. Live simulations_completed was never reproducible
  // anyway (PWO-2 final report 3.4: agreement on 0/76 rows across its own rerun,
  // with chosen_action identical throughout).
  Action half_time_checkpoint_action = -1;
  bool half_time_checkpoint_reached = false;
  int half_time_checkpoint_sim = -1;
  int half_time_checkpoint_num_covered_actions = 0;
  double half_time_checkpoint_covered_prior_mass = 0.0;
  bool pass_complete_search = false;
  bool pass_min_actions = false;
  bool pass_prior_mass = false;
  bool pass_meaningful_visits = false;
  bool pass_q_margin = false;
  bool pass_stability = false;

  // PWO-4 Amendment 2 section 5.1 (RATIFIED 2026-07-29,
  // docs/PWO4_AMENDMENT_2_RETAINED_TREE.md). RETENTION TELEMETRY, EMISSION ONLY.
  //
  // WHY THIS EXISTS. On the !in_session_ path this bot RETAINS its whole node
  // map across a seat's decisions within a game (RunSearch's reset-or-retain
  // block), so cumulative root visits legitimately exceed the simulation budget
  // -- 593 of 969 rows on the CP1 gate block, up to 6.39x. inherited_root_visits
  // CANNOT detect that: its only writers are in dune_search_session.cc, so on
  // this path it holds its initializer of 0 forever, and three registered
  // "freshness proofs" built on it were vacuous. These fields are the live
  // replacement, and the accounting identity over them
  //
  //     sum(final) - sum(pre_search) == simulations_completed
  //
  // is what now proves a search contributed exactly its budget of NEW
  // simulations.
  //
  // BEHAVIOUR NEUTRALITY. Snapshots only: they copy child visit counts at points
  // the search already reaches, draw no RNG, mutate no tree state, and are read
  // by nothing in selection, expansion, backup or seeding. On a FIXED-simulation
  // tier -- which PWO-4 pins, with disable_time_limit=true so no clock can
  // truncate -- cost cannot change the result at all, because the search stops
  // on a COUNT. The claim is not left as an assertion: the section 7.1 A<->B2
  // fidelity gate compares every pre-existing field bitwise against a reference
  // binary that does not contain this code, and one mismatch is a registered
  // STOP.
  //
  // Valid ONLY when a search actually ran. RunSearch's two early returns
  // (single legal action; non-strategic state) leave snapshots_valid false and
  // the vectors empty -- ABSENT, not zero. A zero vector would be
  // indistinguishable from a real cold-root reading.
  bool retention_snapshots_valid = false;
  // The controller state key was present in nodes_ AFTER the reset-or-retain
  // block resolved and BEFORE LookupOrCreateNode ran -- i.e. the retain branch
  // was taken. False whenever Reset() fired, since the map is then empty.
  bool root_reused_pre_search = false;
  // Root-child visits in legal-action order, snapshotted immediately before the
  // simulation loop -- after LookupOrCreateNode, InitializePriorsAndValue and
  // the WO-15 cold-root expansion bookkeeping. A fresh root's child_info is
  // populated with ZERO visits at that point, so a cold root's vector is all
  // zeros: populated, not absent, which is why retention_snapshots_valid is a
  // separate flag rather than an emptiness test.
  std::vector<int> visit_counts_pre_search;
  // The same vector at the stability checkpoint (sim floor(max_sims * fraction),
  // = 100 of 200 under the PWO-4 pin).
  std::vector<int> visit_counts_checkpoint;
  // The same vector at search end. Snapshotted INDEPENDENTLY of visit_counts
  // rather than copied from it, so that requiring the two to be equal is a real
  // cross-check of GetRootDiagnostics against this path instead of a tautology.
  std::vector<int> visit_counts_final;
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

// NOTE: A vestigial `SearchTrainingExample` struct used to live here but was
// never constructed or read anywhere. It was removed to avoid a name collision
// with the live, documented `open_spiel::SearchTrainingExample` emitted by the
// Phase 18B online collector (see dune_online_search_collector.h), which any
// translation unit wiring CollectUpdate must include alongside this header.

// Strategic-state classification function declarations
bool IsStrategicAction(const std::string& action_str);
bool IsStrategicState(const State& state, Player searched_player);

// Per-root Dirichlet-noise stream seed (Phase 18B bug fix). Mixes the search
// seed with the root's identity (player + state key) and the per-bot search
// counter so noise does not repeat across roots handled by one bot instance (the
// old code passed a constant position, seeding every root identically). Exposed
// for unit testing. Distinct roots => distinct seeds; identical inputs => equal.
uint64_t DeriveRootNoiseSeed(uint64_t config_seed, uint64_t search_count,
                             int root_player, const std::string& root_key_str);

// Stable 64-bit identity for a node's transposition key. Exposed so tests can
// build nodes whose tie-break streams differ exactly as the search's do.
uint64_t DeriveNodeKeyHash(int player, const std::string& key_str);

// PUCT tie-break stream seed (WO-15 / search finding 9). MUST include the
// node's identity: keying only on (seed, stream, total_visits) gave every node
// at the same local visit count the identical draw, so ties among equal-PUCT
// children resolved in one correlated repeating pattern across the whole tree
// (worst at fresh nodes, where uniform priors make ties the common case).
// Deterministic: identical inputs => identical seed.
uint64_t DeriveTieBreakSeed(uint64_t config_seed, uint64_t node_key_hash,
                            int total_visits);

// The coverage/Q gate actually applied, given a configured min_visit_threshold.
//
// WO-15 (search findings 3 and 5) defines the <= 0 edge instead of honoring it.
// A threshold of 0 made a zero-visit root pass the coverage gate trivially,
// which (a) resurrected the historical uniform-policy starvation bug in
// GetFinalPolicy and (b) evaluated `return_sum / visits` as 0/0, feeding NaN Q
// values into telemetry and into PruneForcedPlayouts' PUCT comparisons (where
// NaN compares false and silently disables pruning). "Covered" is only
// meaningful for an action the search actually visited, so the floor is 1.
constexpr int EffectiveMinVisitThreshold(int configured_threshold) {
  return configured_threshold < 1 ? 1 : configured_threshold;
}

// ---------------------------------------------------------------------------
// Bot structures
// ---------------------------------------------------------------------------

struct DuneChildInfo {
  int visits = 0;
  double return_sum = 0.0;
  // The tree prior the search actually uses. At the root this may carry
  // `root_prior_temperature` scaling and, when enabled, the Dirichlet mixture.
  double prior = 0.0;
  // The UNTRANSFORMED network prior for this action, snapshotted at expansion
  // before either root transformation is applied. Populated for every expanded
  // node, and it is what every starvation fallback degrades to — the invariant
  // is "starved paths play the RAW network prior", and playing a noised or
  // temperature-flattened prior would execute exploration noise as if it were
  // the policy. Also the item-4 KL telemetry baseline.
  double raw_prior = 0.0;
  double value() const { return visits > 0 ? return_sum / visits : 0.0; }
};

struct DuneISMCTSNode {
  absl::flat_hash_map<Action, DuneChildInfo> child_info;
  int total_visits = -1;
  bool priors_initialized = false;
  double cached_value = 0.0;  // Neural V(s) for this node's current player
  std::vector<double> cached_values; // Neural V(s) for all players
  bool dirichlet_noise_applied = false;
  // Stable identity of this node's transposition key, mixed into the PUCT
  // tie-break stream so two different nodes at the same local visit count do
  // not draw the same tie-break (search finding 9). 0 for nodes built outside
  // the tree (tests); harmless, since determinism is what matters there.
  uint64_t key_hash = 0;
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
  // Same shape as FilterAndNormalizePriors but over DuneChildInfo::raw_prior:
  // the untransformed network prior, free of root temperature scaling and
  // Dirichlet noise. This is the policy every starvation path degrades to.
  ActionsAndProbs FilterAndNormalizeRawPriors(DuneISMCTSNode* node, const std::vector<Action>& legal_actions) const;
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
