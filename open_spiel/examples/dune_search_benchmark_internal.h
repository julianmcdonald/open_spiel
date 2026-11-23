#ifndef OPEN_SPIEL_EXAMPLES_DUNE_SEARCH_BENCHMARK_INTERNAL_H_
#define OPEN_SPIEL_EXAMPLES_DUNE_SEARCH_BENCHMARK_INTERNAL_H_

#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <random>
#include <set>
#include <shared_mutex>
#include <string>
#include <vector>

#include <torch/torch.h>

#include "dune_batched_evaluator.h"
#include "dune_evaluator.h"
#include "dune_market_diagnostics.h"
#include "dune_network.h"
#include "dune_puct_is_mcts.h"
#include "dune_search_session.h"
#include "open_spiel/abseil-cpp/absl/flags/declare.h"
#include "open_spiel/games/dune_imperium/dune_imperium.h"
#include "open_spiel/spiel.h"

ABSL_DECLARE_FLAG(std::string, model_checkpoint);
ABSL_DECLARE_FLAG(std::string, search_value_checkpoint);
ABSL_DECLARE_FLAG(bool, force_split_evaluator);
ABSL_DECLARE_FLAG(std::string, opponent_checkpoint);
ABSL_DECLARE_FLAG(int, seed);
ABSL_DECLARE_FLAG(int, games);
ABSL_DECLARE_FLAG(int, start_episode_id);
ABSL_DECLARE_FLAG(std::string, game_jsonl_path);
ABSL_DECLARE_FLAG(std::string, search_jsonl_path);
ABSL_DECLARE_FLAG(std::string, aggregate_json_path);
ABSL_DECLARE_FLAG(bool, record_market_diagnostics);
ABSL_DECLARE_FLAG(std::string, market_diag_json_path);
ABSL_DECLARE_FLAG(std::string, market_diag_csv_path);
ABSL_DECLARE_FLAG(std::string, market_diag_games_path);
ABSL_DECLARE_FLAG(std::string, candidate_decision_traces_path);
ABSL_DECLARE_FLAG(bool, record_candidate_decision_traces);
ABSL_DECLARE_FLAG(std::string, output_dir);
ABSL_DECLARE_FLAG(int, threads);
ABSL_DECLARE_FLAG(int, hidden_dim);
ABSL_DECLARE_FLAG(int, num_blocks);
ABSL_DECLARE_FLAG(int, opp_hidden_dim);
ABSL_DECLARE_FLAG(int, opp_num_blocks);
ABSL_DECLARE_FLAG(int, max_simulations);
ABSL_DECLARE_FLAG(double, puct_c);
ABSL_DECLARE_FLAG(int, max_world_samples);
ABSL_DECLARE_FLAG(double, utility_divisor);
ABSL_DECLARE_FLAG(double, temperature);
ABSL_DECLARE_FLAG(double, simulated_opponent_temperature);
ABSL_DECLARE_FLAG(double, external_opponent_temperature);
ABSL_DECLARE_FLAG(double, dirichlet_epsilon);
ABSL_DECLARE_FLAG(double, dirichlet_alpha);
ABSL_DECLARE_FLAG(bool, rotate_seat);
ABSL_DECLARE_FLAG(bool, use_opponent_model);
ABSL_DECLARE_FLAG(bool, search_leader_draft);
ABSL_DECLARE_FLAG(bool, leader_mass_only_coverage);
ABSL_DECLARE_FLAG(int, leader_draft_simulations);
ABSL_DECLARE_FLAG(std::string, evaluator_mode);
ABSL_DECLARE_FLAG(bool, corpus_roots);
ABSL_DECLARE_FLAG(std::string, corpus_root_path);
ABSL_DECLARE_FLAG(int, corpus_root_count);
ABSL_DECLARE_FLAG(std::string, corpus_root_role);
ABSL_DECLARE_FLAG(int, batch_target_rows);
ABSL_DECLARE_FLAG(int, batch_max_queue_wait_ms);
ABSL_DECLARE_FLAG(int, corpus_workers);
ABSL_DECLARE_FLAG(std::string, corpus_root_jsonl_path);
ABSL_DECLARE_FLAG(std::string, corpus_controller);
ABSL_DECLARE_FLAG(int, corpus_warmup_searches);
ABSL_DECLARE_FLAG(std::string, batcher_telemetry_json_path);
ABSL_DECLARE_FLAG(bool, nonlinear_value_head);
ABSL_DECLARE_FLAG(bool, opponent_nonlinear_value_head);
ABSL_DECLARE_FLAG(bool, verbose_diagnostics);
ABSL_DECLARE_FLAG(bool, check_strategic_state);
ABSL_DECLARE_FLAG(bool, matched_fallback_sampling);
ABSL_DECLARE_FLAG(bool, disable_time_limit);
ABSL_DECLARE_FLAG(double, root_prior_temperature);
ABSL_DECLARE_FLAG(int, fixed_continuation_reserve);
ABSL_DECLARE_FLAG(int, purchase_combat_budget);
ABSL_DECLARE_FLAG(double, live_continuation_reserve_seconds);
ABSL_DECLARE_FLAG(bool, live_deadline);
ABSL_DECLARE_FLAG(bool, use_session);
ABSL_DECLARE_FLAG(bool, policy_only);
ABSL_DECLARE_FLAG(bool, audit_training_policy_iteration);
ABSL_DECLARE_FLAG(int, pi_primary_simulations);
ABSL_DECLARE_FLAG(int, pi_continuation_simulations);
ABSL_DECLARE_FLAG(int, pi_short_window_simulations);
ABSL_DECLARE_FLAG(std::string, fresh_search_roles);
ABSL_DECLARE_FLAG(int, force_swordmaster_rounds);
ABSL_DECLARE_FLAG(int, grant_swordmaster_round);
ABSL_DECLARE_FLAG(double, relative_time_budget_ms);
ABSL_DECLARE_FLAG(int, max_nodes);
ABSL_DECLARE_FLAG(bool, conservative_override_enabled);
ABSL_DECLARE_FLAG(double, conservative_covered_prior_threshold);
ABSL_DECLARE_FLAG(int, conservative_meaningful_visit_threshold);
ABSL_DECLARE_FLAG(double, conservative_q_margin_threshold);
ABSL_DECLARE_FLAG(double, conservative_stability_checkpoint_fraction);
ABSL_DECLARE_FLAG(bool, conservative_continuation_overrides_disabled);
ABSL_DECLARE_FLAG(std::string, action_selection_mode);
ABSL_DECLARE_FLAG(std::string, controller_mode);
ABSL_DECLARE_FLAG(std::string, mixed_opponent_controller);
ABSL_DECLARE_FLAG(double, candidate_logit_cap);
ABSL_DECLARE_FLAG(double, opponent_logit_cap);

namespace open_spiel {

constexpr int kPaceMaxRounds = 10;

struct GameStats {
  int search_wins = 0;
  int opponent_wins[3] = {0, 0, 0};
  int search_wins_by_seat[4] = {0, 0, 0, 0};
  int games_by_seat[4] = {0, 0, 0, 0};
  int search_placements[4] = {0, 0, 0, 0};
  double search_return_sum = 0.0;
  double search_return_by_seat[4] = {0.0, 0.0, 0.0, 0.0};
  int search_swordmasters = 0;
  int opponent_swordmasters = 0;
  std::vector<int> rounds_played;
  double search_step_time_sum = 0.0;
  int search_steps_count = 0;
  double abs_terminal_return_sum = 0.0;
  int64_t terminal_returns_count = 0;
  double vp_margin_sum = 0.0;
  int64_t vp_margin_count = 0;
  std::vector<double> vp_margins;
  int search_mcts_steps_count = 0;
  double search_mcts_step_time_sum = 0.0;
  int64_t total_simulations_completed = 0;
  int64_t total_timeouts = 0;
  int64_t total_fallbacks = 0;
  int64_t total_inferences = 0;
  std::map<std::string, int64_t> fallback_reason_counts;
  int64_t total_incomplete_searches = 0;
  std::vector<double> search_mcts_step_times;
  double covered_prior_mass_sum = 0.0;
  int64_t role_counts[7] = {0, 0, 0, 0, 0, 0, 0};
};

// --- WO-1 Phase 2: seat controller topology -------------------------------

enum class SeatControllerMode { kSingle, kHomogeneous, kMixed };
enum class MixedOpponentController { kRandomLegal, kRawPolicy };

// Exactly what drove one seat for one game. Emitted per game so a result can
// never be re-read without knowing which controller produced it -- the Phase 0
// registration requires slice opponents and controllers to be pinned in the
// run manifest, not described in prose.
struct SeatControllerProvenance {
  std::string controller = "unset";   // search|raw_policy|random_legal
  std::string checkpoint;             // "" when the seat uses no model
  std::string checkpoint_sha256;      // "" when the seat uses no model
  int max_simulations = -1;           // -1 when not a searched seat
  std::string budget_mode = "none";
  double logit_cap = -1.0;            // -1 means "no evaluator on this seat"
};

struct GameSeatSetup {
  std::array<bool, 4> seat_is_searched{};
  std::array<SeatControllerProvenance, 4> seat_provenance;
  std::vector<std::unique_ptr<DuneSearchSession>> seat_sessions;
  std::vector<std::shared_ptr<open_spiel::algorithms::Evaluator>> seat_evaluators;
  std::unique_ptr<DunePUCTISMCTSBot> search_bot;
  std::vector<std::unique_ptr<Bot>> bots;
  std::array<uint64_t, 4> seat_config_seed{};
};

struct TurnAuditState {
  bool in_search_turn = false;
  int turn_decisions = 0;
  int turn_primary_sims = 0;
  int turn_continuation_sims = 0;
  double turn_wall_time_ms = 0.0;
  bool turn_had_timeout = false;
  bool turn_had_fallback = false;
  std::string turn_fallback_reason = "none";
};

struct GameOutcomeWriteContext {
  int episode_id = 0;
  uint64_t game_seed = 0;
  int search_seat = 0;
  int winner = -1;
  const std::vector<double>* returns = nullptr;
  double search_vp = 0.0;
  double vp_margin = 0.0;
  int corrected_round = 0;
  int search_steps_count = 0;
  double game_duration_s = 0.0;
  const std::array<int, 4>* sm_acquire_round_by_seat = nullptr;
  const std::map<int, int>* search_seat_solari_by_round = nullptr;
  const std::map<int, int>* search_seat_placements_by_round = nullptr;
  const std::map<int, int>* search_seat_agents_avail_by_round = nullptr;
  const std::array<SeatControllerProvenance, 4>* seat_provenance = nullptr;
  const std::array<std::array<int, 4>, kPaceMaxRounds + 2>* vp_at_round_end = nullptr;
  const std::array<bool, kPaceMaxRounds + 2>* round_end_seen = nullptr;
  const std::array<int, 4>* specimen_conversions = nullptr;
};

struct CorpusRootRecord {
  int corpus_idx = 0;
  Player player = 0;
  int round = 0;
  uint64_t seed = 0;
  std::string category;
  std::vector<Action> history;
  std::vector<Action> recorded_legals;
};

struct CorpusRootOutcome {
  int root_idx = 0;
  int corpus_idx = 0;
  Player player = 0;
  int round = 0;
  Action selected_action = kInvalidAction;
  Action raw_argmax_action = kInvalidAction;
  std::vector<Action> actions;
  std::vector<int> visit_counts;
  std::vector<double> q_values;
  std::vector<double> priors;
  std::vector<double> raw_priors;
  int num_covered_actions = 0;
  double covered_prior_mass = 0.0;
  bool used_fallback = false;
  std::string fallback_reason = "none";
  double root_value = 0.0;
  int total_root_visits = 0;
  int simulations_completed = 0;
  double elapsed_time_ms = 0.0;   // the SEARCH's own internal elapsed
  double root_wall_ms = 0.0;
  double time_nn_inference_ms = 0.0;
  double time_tree_engine_ms = 0.0;
  double time_sample_root_ms = 0.0;
  double time_other_overhead_ms = 0.0;
  bool populated = false;
};

// Checkpoint digests, computed ONCE in main() BEFORE any worker thread is
// spawned and only read afterwards. That write-before-spawn ordering is what
// makes these safe to touch from every worker without a lock; do not assign to
// them from inside WorkerThread.
extern std::string g_candidate_ckpt_sha256;
extern std::string g_candidate_value_ckpt_sha256;
extern std::string g_opponent_ckpt_sha256;
extern bool g_using_split_evaluator;

// Defined in dune_search_benchmark.cc (delegating to dune_terminal_vp_report.h).
int GetTrueFinalVp(const dune_imperium::DuneImperiumState* dune_state, int p);

// --- Setup helpers (dune_search_benchmark_setup.cc) -------------------------

std::set<DuneDecisionRole> ParseFreshSearchRoles(const std::string& csv);
SeatControllerMode ParseControllerMode(const std::string& s);
MixedOpponentController ParseMixedOpponentController(const std::string& s);
std::string BudgetModeName(DuneSearchBudgetMode m);
std::string DuneDecisionRoleToString(open_spiel::DuneDecisionRole role);

std::shared_ptr<SharedDunePolicyValueNetImpl> LoadDuneBenchmarkModel(
    const std::string& ckpt_path,
    int64_t obs_size,
    int64_t action_size,
    int h_dim,
    int n_blocks,
    bool nonlinear,
    torch::Device device);

void ConfigureOutputPathsFromOutputDir(bool record_market_diagnostics);

std::shared_ptr<open_spiel::algorithms::Evaluator> BuildSearchEvaluator(
    const std::shared_ptr<SharedDunePolicyValueNetImpl>& search_model,
    const std::shared_ptr<SharedDunePolicyValueNetImpl>& search_value_model,
    const std::shared_ptr<open_spiel::BatchedEvaluator>& search_coord,
    torch::Device device,
    float candidate_logit_cap);

GameSeatSetup SetupGameSeats(
    std::mt19937& game_rng,
    SeatControllerMode controller_mode,
    MixedOpponentController mixed_opponent,
    int search_seat,
    float candidate_logit_cap,
    float opponent_logit_cap,
    torch::Device device,
    const std::shared_ptr<open_spiel::algorithms::Evaluator>& search_evaluator,
    const std::shared_ptr<SharedDunePolicyValueNetImpl>& opponent_model,
    const std::shared_ptr<open_spiel::BatchedEvaluator>& opp_coord);

void InitSharedBatchedCoordinators(
    const std::shared_ptr<SharedDunePolicyValueNetImpl>& search_model,
    const std::shared_ptr<SharedDunePolicyValueNetImpl>& opponent_model,
    torch::Device device,
    std::shared_mutex& search_coord_mutex,
    std::shared_mutex& opp_coord_mutex,
    std::shared_ptr<open_spiel::BatchedEvaluator>* search_coord,
    std::shared_ptr<open_spiel::BatchedEvaluator>* opp_coord);

// --- Runner helpers (dune_search_benchmark_runner.cc) -----------------------

Action SelectRawPriorAction(
    open_spiel::algorithms::Evaluator& evaluator,
    const State& state,
    double temperature,
    std::mt19937& rng);

void WorkerThread(
    int thread_id,
    std::shared_ptr<const Game> game,
    std::shared_ptr<SharedDunePolicyValueNetImpl> search_model,
    std::shared_ptr<SharedDunePolicyValueNetImpl> search_value_model,
    std::shared_ptr<SharedDunePolicyValueNetImpl> opponent_model,
    std::atomic<int>& next_game_id,
    int total_games,
    std::atomic<int>& completed_games,
    std::mutex& log_mutex,
    GameStats& global_stats,
    std::mutex& stats_mutex,
    bool record_market_diagnostics = false,
    std::vector<dune_market_diag::CompactPerGameMarketDiagnostics>* market_results = nullptr,
    std::shared_ptr<open_spiel::BatchedEvaluator> search_coord = nullptr,
    std::shared_ptr<open_spiel::BatchedEvaluator> opp_coord = nullptr);

// --- Stats & Report helpers (dune_search_benchmark_report.cc) ----------------

void ComputeWilsonWinrateCI(int wins, int total, double* lcb, double* ucb);
void ComputeVpMarginCI(const std::vector<double>& margins, double* mean, double* lcb, double* ucb);

void UpdateStepSearchStats(
    DuneDecisionRole role,
    bool is_strategic,
    double step_duration,
    const DuneSearchResult& last_res,
    const DuneSearchSession* cur_session,
    TurnAuditState& audit_state,
    GameStats& thread_stats);

void RecordTerminalGameStats(
    const State& state,
    const dune_imperium::DuneImperiumState* dune_state,
    int search_seat,
    GameStats& thread_stats,
    int* winner_out,
    double* search_vp_out,
    double* margin_out,
    int* corrected_round_out);

void MergeGameStats(
    GameStats& dst,
    const GameStats& src,
    const std::vector<int>* game_role_counts = nullptr);

// --- JSONL helpers (dune_search_benchmark_jsonl.cc) -------------------------

void WriteSearchDecisionJsonl(
    const std::string& search_jsonl,
    int episode_id,
    uint64_t matched_step_count,
    uint64_t game_seed,
    int search_seat,
    Player current_player,
    const DuneSearchResult& last_res,
    Action chosen_action,
    const State& state,
    const std::array<SeatControllerProvenance, 4>& seat_provenance,
    bool matched_sampled_this_decision,
    std::mutex& log_mutex);

void WriteCandidateDecisionTraceJsonl(
    const std::string& cand_traces_path,
    int episode_id,
    uint64_t matched_step_count,
    Player current_player,
    DuneDecisionRole role,
    const DunePUCTISMCTSBot* search_bot,
    bool search_this_role,
    const DuneSearchSession* cur_session,
    const DuneSearchResult& last_res,
    Action chosen_action,
    bool forced_swordmaster,
    const State& state,
    const std::array<SeatControllerProvenance, 4>& seat_provenance,
    double step_duration,
    std::mutex& log_mutex);

void WriteGameOutcomeJsonl(
    const std::string& game_jsonl,
    const dune_imperium::DuneImperiumState* dune_state,
    const GameOutcomeWriteContext& ctx,
    std::mutex& log_mutex);

// --- Report helpers (dune_search_benchmark_report.cc) -----------------------

void PrintEpisode67Breakdown(const dune_imperium::DuneImperiumState* dune_state);

void PrintBatcherCoordinatorStats(
    const std::shared_ptr<open_spiel::BatchedEvaluator>& search_coord,
    const std::shared_ptr<open_spiel::BatchedEvaluator>& opp_coord);

void WriteMarketDiagnosticsReports(
    const std::vector<dune_market_diag::CompactPerGameMarketDiagnostics>& market_results,
    int total_games);

void PrintAndWriteFinalBenchmarkReport(
    const GameStats& global_stats,
    int total_games,
    int num_threads,
    const std::string& model_ckpt,
    const std::string& opp_ckpt,
    bool has_opponent_model,
    const std::string& full_game_eval_mode,
    bool batched_mode,
    const std::shared_ptr<open_spiel::BatchedEvaluator>& search_coord,
    const std::shared_ptr<open_spiel::BatchedEvaluator>& opp_coord,
    std::chrono::steady_clock::time_point run_start_time);

// --- Corpus-root driver (dune_search_benchmark_corpus.cc) -------------------

std::unique_ptr<State> ReconstructCorpusRootState(
    const std::shared_ptr<const Game>& game,
    const std::vector<Action>& history);

int RunCorpusRootBenchmark(
    std::shared_ptr<const Game> game,
    std::shared_ptr<SharedDunePolicyValueNetImpl> search_model,
    torch::Device device);

}  // namespace open_spiel

#endif  // OPEN_SPIEL_EXAMPLES_DUNE_SEARCH_BENCHMARK_INTERNAL_H_
