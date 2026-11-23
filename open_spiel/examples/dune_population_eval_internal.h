#ifndef OPEN_SPIEL_EXAMPLES_DUNE_POPULATION_EVAL_INTERNAL_H_
#define OPEN_SPIEL_EXAMPLES_DUNE_POPULATION_EVAL_INTERNAL_H_

// Internal shared declarations and data structures for dune_population_eval.

#include <array>
#include <atomic>
#include <cstdint>
#include <fstream>
#include <memory>
#include <mutex>
#include <optional>
#include <shared_mutex>
#include <string>
#include <utility>
#include <vector>

#include <torch/torch.h>

#include "open_spiel/abseil-cpp/absl/flags/declare.h"
#include "open_spiel/games/dune_imperium/dune_imperium.h"
#include "open_spiel/spiel.h"

#include "dune_batched_evaluator.h"
#include "dune_market_diagnostics.h"
#include "dune_network.h"

// Flags accessed across translation units (defined in dune_population_eval.cc).
ABSL_DECLARE_FLAG(bool, candidate_r758_teacher);
ABSL_DECLARE_FLAG(int, candidate_eval_batch_size);
ABSL_DECLARE_FLAG(int, teacher_category_mask);
ABSL_DECLARE_FLAG(int, teacher_top_k);
ABSL_DECLARE_FLAG(int, teacher_rollouts);
ABSL_DECLARE_FLAG(int, teacher_stage1_rollouts);
ABSL_DECLARE_FLAG(double, teacher_stage1_min_mean_diff);
ABSL_DECLARE_FLAG(double, teacher_min_t_stat);
ABSL_DECLARE_FLAG(double, teacher_override_margin);
ABSL_DECLARE_FLAG(std::string, teacher_crn_mode);
ABSL_DECLARE_FLAG(bool, enable_semantic_scorer);
ABSL_DECLARE_FLAG(std::string, candidate_market_appendix_mode);
ABSL_DECLARE_FLAG(int, trace_episode);

namespace open_spiel {
namespace dune_pop_eval {

using dune_imperium::DuneImperiumState;
using dune_imperium::kMaxRounds;
using dune_imperium::kNumPlayers;
using dune_imperium::kTargetVp;

extern std::atomic<int64_t> g_teacher_searched_roots;
extern std::atomic<int64_t> g_teacher_overrides;

// ---------------------------------------------------------------------------
// Wilson score interval
// ---------------------------------------------------------------------------
struct WilsonCI {
  double point;
  double lower;
  double upper;
};

// ---------------------------------------------------------------------------
// Per-game result
// ---------------------------------------------------------------------------
struct GameResult {
  int episode_id;
  int seat;
  std::vector<std::string> opponents;  // Exact size 3
  int placement;                       // 1-based (mapped directly from returns)
  double game_return;
  int ending_round;
  int current_vp;
  int final_scored_vp;
  uint64_t chance_seed = 0;
  std::vector<open_spiel::Action> history;

  // PWO-5 section 13.5. The registered VP-margin estimand is
  //
  //     vp_margin = candidate's TRUE final VP - MEAN of the three opponents'
  //
  // -- true final VP, not track VP; the mean of three, not the best. Before
  // this, popeval emitted only the CANDIDATE's `final_scored_vp`, so the
  // quantity section 15.2 gate 2 asks for could not be computed from its
  // output at all; a reconstruction from `vp_end_by_round` computes a
  // different thing.
  //
  // `FinalScoredVp` is the ENGINE's function. It is NOT `GetTrueFinalVp`, the
  // REPORTING helper that drops the tech-tile-8 guard and disagrees with the
  // engine on 212 of 1,600 seat values -- the defect that became the PWO-5
  // gate-3 STOP. Every seat here goes through the engine.
  std::array<int, kNumPlayers> final_scored_vp_all{};
  std::array<int, kNumPlayers> player_leaders{};
  // Exactly representable only as a rational with denominator 3, so it is
  // carried as a double and SERIALIZED AT ROUND-TRIP PRECISION (%.17g). It is
  // never routed through open_spiel's JSON writer, which emits doubles as %f
  // at six decimal places and would silently floor anything below 5e-7 to 0.0.
  double vp_margin = 0.0;
  bool vp_margin_valid = false;
  // --- Phase-3 pace/threshold telemetry (new keys only) ---
  // vp_end_rN[p] = player p's running (track) VP at the end of round N.
  // has_rN=false => the game ended before round N completed (emitted as null).
  std::array<int, kNumPlayers> vp_end_r5{};
  std::array<int, kNumPlayers> vp_end_r6{};
  std::array<int, kNumPlayers> vp_end_r7{};
  bool has_r5 = false;
  bool has_r6 = false;
  bool has_r7 = false;
  std::array<int, kNumPlayers> r7_track_vp{};
  bool has_r7_track_vp = false;
  int end_trigger_player = -1;  // -1 => null
  int end_trigger_round = -1;   // -1 => null
  int winner_triggered = -1;    // -1 => null, 0 => false, 1 => true

  // --- WO-1 Phase 3 pace/tempo telemetry (new keys only) ---
  // Track VP at the end of EVERY round, 1..terminal. Supersedes the r5/r6/r7
  // triple above, which stays for backward compatibility. vp_end_by_round[N-1]
  // is round N; only the first `rounds_captured` entries are meaningful.
  std::array<std::array<int, kNumPlayers>, kMaxRounds> vp_end_by_round{};
  int rounds_captured = 0;
  // First round at whose END this seat's track VP was >= kTargetVp; -1 => null.
  std::array<int, kNumPlayers> first_round_vp_ge_11{};
  // Every seat at track VP >= kTargetVp in the terminal round. May hold more
  // than one seat: crossing is not exclusive and no causal trigger is assigned.
  std::vector<int> terminal_threshold_set;
  bool candidate_in_terminal_threshold_set = false;
  // "threshold" | "round_limit" | "other". Derived from the same accessors the
  // engine's own end test uses (dune_imperium.cc:8891-8904): end_game is set
  // when round_ > kMaxRounds OR any vp_[p] >= kTargetVp. "threshold" wins when
  // both hold, matching end_trigger_player's existing precedence. "other" is
  // unreachable by construction and is emitted as a tripwire.
  std::string terminal_reason = "other";
  // Applied ConvertSpecimenToTroop actions per seat over the whole game.
  std::array<int, kNumPlayers> specimen_conversions{};
  // Round 7 penalty telemetry per seat
  std::array<bool, kNumPlayers> atomics_used{};
  std::array<int, kNumPlayers> atomics_in_reveal{-1, -1, -1, -1};
  std::array<int, kNumPlayers> atomics_persuasion{-1, -1, -1, -1};
  std::array<int, kNumPlayers> atomics_round{-1, -1, -1, -1};
  std::array<int, kNumPlayers> plot_plays{};
  std::array<std::vector<int>, kNumPlayers> plot_pre_hand_sizes{};
  std::array<int, kNumPlayers> research_col{};
  std::array<int, kNumPlayers> research_row{};
  std::array<std::vector<std::pair<int, int>>, kNumPlayers> research_paths{};
  // Step 3e: reveals taken while player still had an agent to place, and solari at each round end
  std::array<int, kNumPlayers> reveals_with_agents_remaining{};
  std::array<std::array<int, kNumPlayers>, kMaxRounds> solari_end_by_round{};
};

// ---------------------------------------------------------------------------
// Opponent checkpoint metadata & evaluation configuration/runtime structs
// ---------------------------------------------------------------------------
struct OpponentMetadata {
  std::string path;
  int hidden_dim;
  int num_blocks;
  int input_dim;
  dune_imperium::MarketAppendixMode market_mode;
  std::string market_mode_str;
  dune_imperium::DeckPoolSemantics deck_pool_semantics =
      dune_imperium::DeckPoolSemantics::kLegacy;
};

struct EvalConfig {
  std::string model_checkpoint;
  std::string opponent_str;
  int total_games = 0;
  int start_episode_id = 0;
  uint64_t base_seed = 42;
  std::string domain_str;
  uint64_t domain = 0;
  bool greedy = true;
  int requested_threads = 0;
  int hidden_dim = 2048;
  int num_blocks = 8;
  int opp_hidden_dim = -1;
  int opp_num_blocks = -1;
  std::string output_dir;
  float temperature = 1.0f;
  bool has_opponent_greedy = false;
  bool opponent_greedy = false;
  bool has_opponent_temperature = false;
  float opponent_temperature = 0.0f;
  double candidate_logit_cap = 10.0;
  double opponent_logit_cap = 10.0;
  std::string dump_logit_stats_path;
  bool record_market_diagnostics = false;
  std::string market_diag_json_path;
  std::string market_diag_csv_path;
  std::string market_diag_games_path;
  bool deterministic = true;
  bool rollout_amp = true;
  bool allow_tf32 = true;
  bool record_history = false;
  bool nonlinear_value_head = false;
  bool opponent_nonlinear_value_head = false;
  bool use_opponent_model = false;
  std::vector<std::string> opponent_paths;
};

struct EvalRuntime {
  std::shared_ptr<const Game> game;
  int64_t obs_size = 0;
  int64_t action_size = 0;
  bool provides_info_state_tensor = false;
  bool provides_observations_tensor = false;
  torch::Device device{torch::kCPU};
  unsigned int num_threads = 1;
  int eval_batch_size = 1;
  int opp_eval_batch_size = 1;
  int eval_timeout_ms = 1;
  std::string device_name;
  // Synchronization primitives (address-stable inside heap-allocated EvalRuntime)
  std::shared_mutex sync_mutex;
  std::mutex eval_mutex;

  int main_hidden_dim = 0;
  int main_num_blocks = 0;
  int main_input_dim = 0;
  dune_imperium::MarketAppendixMode cand_market_mode =
      dune_imperium::MarketAppendixMode::kNone;
  std::shared_ptr<IGameEvaluator> model_evaluator;

  std::vector<std::string> opponent_names;
  std::vector<OpponentMetadata> opp_metadata;
  std::vector<std::shared_ptr<IGameEvaluator>> opponent_evaluators;
};

// ---------------------------------------------------------------------------
// Setup & model loading helpers (dune_population_eval_setup.cc)
// ---------------------------------------------------------------------------
uint64_t ResolveDomain(const std::string& domain_str);
std::vector<std::string> SplitCommaSeparated(const std::string& value);
dune_imperium::MarketAppendixMode ParseMarketMode(const std::string& str);
std::string MarketModeToString(dune_imperium::MarketAppendixMode mode);

bool DetectModelDimensions(
    const std::string& model_path, int* hidden_dim, int* num_blocks,
    int* input_dim = nullptr, std::string* market_mode = nullptr,
    dune_imperium::DeckPoolSemantics* deck_pool_semantics = nullptr,
    std::string* semantic_descriptor_schema = nullptr,
    bool* slot_indexed_out_layer_f = nullptr);

std::unique_ptr<EvalRuntime> SetupEvaluation(const EvalConfig& cfg);
void PrintEvaluationConfig(const EvalConfig& cfg, const EvalRuntime& runtime);

// ---------------------------------------------------------------------------
// Game simulation & worker pool (dune_population_eval_games.cc)
// ---------------------------------------------------------------------------
void RunEvaluationGames(
    const EvalConfig& cfg,
    const EvalRuntime& runtime,
    std::ofstream* dump_out,
    std::vector<GameResult>& results,
    std::vector<dune_market_diag::CompactPerGameMarketDiagnostics>* market_results);

// ---------------------------------------------------------------------------
// Reporting & diagnostics (dune_population_eval_report.cc)
// ---------------------------------------------------------------------------
WilsonCI WilsonScore(int successes, int trials, double z = 1.96);
const char* GamePhaseRole(dune_imperium::GamePhase ph);

void AppendLogitStatsRow(std::ofstream& out, int episode_id, int round,
                         const std::string& role,
                         const std::vector<Action>& legal,
                         const std::vector<float>& raw_logits,
                         double configured_cap);

void ReportMarketDiagnostics(
    const EvalConfig& cfg,
    const std::vector<dune_market_diag::CompactPerGameMarketDiagnostics>& market_results);

void ReportAggregateResults(const EvalConfig& cfg,
                            const EvalRuntime& runtime,
                            double elapsed_secs,
                            const std::vector<GameResult>& results);

}  // namespace dune_pop_eval
}  // namespace open_spiel

#endif  // OPEN_SPIEL_EXAMPLES_DUNE_POPULATION_EVAL_INTERNAL_H_
