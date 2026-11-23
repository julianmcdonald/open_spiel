#ifndef OPEN_SPIEL_EXAMPLES_DUNE_PLAY_SEARCH_TOURNAMENT_INTERNAL_H_
#define OPEN_SPIEL_EXAMPLES_DUNE_PLAY_SEARCH_TOURNAMENT_INTERNAL_H_

#include <atomic>
#include <chrono>
#include <csignal>
#include <fstream>
#include <memory>
#include <optional>
#include <set>
#include <shared_mutex>
#include <string>
#include <vector>

#include "open_spiel/examples/dune_batched_evaluator.h"
#include "open_spiel/examples/dune_compound_turn_search.h"
#include "open_spiel/examples/dune_network.h"
#include "open_spiel/examples/dune_wide_rollout_search.h"
#include "open_spiel/games/dune_imperium/dune_imperium.h"
#include "open_spiel/spiel.h"
#include "open_spiel/utils/json.h"

namespace open_spiel::dune_imperium::play_search_tournament {

using Clock = std::chrono::steady_clock;
inline constexpr double kOverrideMargin = 0.15;
inline constexpr int kControllerVersion = 3;
inline constexpr const char* kSimulationPolicy = "candidate_model_all_seats";

extern volatile std::sig_atomic_t interrupted;

enum class Mode {
  kRaw,
  kLegacy,
  kWide,
  kLeader,
  kAdaptive,
  kAdaptiveLeader,
  kShAdaptive,
  kShAdaptiveLeader,
  kFullShAdaptiveLeader,
  kShSuperhuman
};

struct Options {
  Mode mode;
  std::string mode_name;
  std::string cand_profile;
  std::string model_path;
  std::string opponent_path;
  std::string roots_path;
  std::string output_path;
  std::string device_name;
  int games;
  int start_game_id;
  int threads;
  int batch_size;
  int rollouts;
  int max_roots;
  uint64_t master_seed;
  bool rollout_amp;
  bool allow_tf32;
  double max_seconds;
  std::vector<int> explicit_game_ids;
  std::string config_key;
  SearchRolloutCrnMode crn_mode = SearchRolloutCrnMode::kSequential;
  bool unbiased_stage2_gate = false;
  bool fixed_top5_candidates = false;
  bool all_buy_candidates = false;
  int leader_rollouts = 32;
  bool sample_opponent_leader_draft = false;
  bool leader_rollouts_explicit = false;
  bool crn_mode_explicit = false;
  bool unbiased_stage2_gate_explicit = false;
  bool fixed_top5_explicit = false;
  bool all_buy_explicit = false;
  bool rollouts_explicit = false;
  double prior_log_ratio_weight = 0.02;
  double stage1_high_conf_prior_skip = 0.95;
  double win_speed_tiebreak_eps = 0.0;
  bool ceiling_aware_margin = true;
  double kl_policy_tau = 0.0;
  double game_clock_ms = -1.0;
};

inline std::string CrnModeName(SearchRolloutCrnMode mode) {
  switch (mode) {
    case SearchRolloutCrnMode::kSequential: return "sequential";
    case SearchRolloutCrnMode::kDeckOrderCrn: return "deck_order";
    case SearchRolloutCrnMode::kEventKeyedCrn: return "event_keyed";
  }
  return "sequential";
}

inline double SecondsSince(Clock::time_point start) {
  return std::chrono::duration<double>(Clock::now() - start).count();
}

struct ModelBundle {
  std::shared_ptr<SharedDunePolicyValueNetImpl> model;
  std::unique_ptr<std::shared_mutex> mutex;
  std::shared_ptr<BatchedEvaluator> coordinator;
  std::shared_ptr<BatchedNNEvaluator> evaluator;
  int num_blocks;
  std::string model_sha256;
};

struct RunControl {
  Clock::time_point start = Clock::now();
  double max_seconds;
  std::atomic<bool> failed{false};
  bool Stop() const {
    return interrupted || failed.load() ||
           (max_seconds > 0 && SecondsSince(start) >= max_seconds);
  }
};

struct GameOutcome {
  int game_id = 0;
  uint64_t seed = 0;
  Player seat = 0;
  std::vector<double> returns;
  std::vector<int> vps;
  int placement = 0;
  int ending_round = 0;
  int cand_r7_track_vp = 0;
  int max4_r7_track_vp = 0;
  int searches = 0;
  int overrides = 0;
  int new_rollouts = 0;
  int reused_rollouts = 0;
  int new_info_resets = 0;
  int decisions = 0;
  int wide_roots = 0;
  int prior_rank4plus_selections = 0;
  double search_seconds = 0;
  double game_seconds = 0;
  double sum_paired_corr = 0.0;
  double sum_paired_diff_var = 0.0;
  int paired_corr_pairs = 0;
  int paired_diff_var_pairs = 0;
};
using GameResult = GameOutcome;

struct Summary {
  int completed_games = 0;
  int wins = 0;
  double sum_utility = 0.0;
  double sum_ending_round = 0.0;
  int rounds_le_7_count = 0;
  double sum_cand_r7_track_vp = 0.0;
  double sum_max4_r7_track_vp = 0.0;
  double sum_paired_corr = 0.0;
  double sum_paired_diff_var = 0.0;
  int paired_corr_pairs = 0;
  int paired_diff_var_pairs = 0;
  double mean_paired_rollout_corr() const {
    return paired_corr_pairs > 0 ? (sum_paired_corr / paired_corr_pairs) : 0.0;
  }
  double mean_paired_diff_var() const {
    return paired_diff_var_pairs > 0 ? (sum_paired_diff_var / paired_diff_var_pairs) : 0.0;
  }
  double avg_ending_round() const {
    return completed_games > 0 ? (sum_ending_round / completed_games) : 0.0;
  }
  double frac_rounds_le_7() const {
    return completed_games > 0
               ? (static_cast<double>(rounds_le_7_count) / completed_games)
               : 0.0;
  }
  double avg_cand_r7_track_vp() const {
    return completed_games > 0 ? (sum_cand_r7_track_vp / completed_games) : 0.0;
  }
  double avg_max4_r7_track_vp() const {
    return completed_games > 0 ? (sum_max4_r7_track_vp / completed_games) : 0.0;
  }
  double avg_utility() const {
    return completed_games > 0 ? (sum_utility / completed_games) : 0.0;
  }
  double win_rate() const {
    return completed_games > 0
               ? (static_cast<double>(wins) / completed_games)
               : 0.0;
  }
};

struct RootCase {
  int root_id;
  int game_id;
  int seat;
  int round;
  std::string decision_type;
  std::vector<Action> history;
};

struct TournamentRunStats {
  int newly_completed = 0;
  std::string error;
};

json::Value ReadJson(const std::string& path);
int IntField(const json::Object& object, const std::string& field);
std::string StringField(const json::Object& object, const std::string& field);
bool BoolField(const json::Object& object, const std::string& field,
               bool absent_value);
double NumberField(const json::Object& object, const std::string& field,
                   double absent_value);

std::optional<GameOutcome> PlayGame(
    int game_id, const Game& game, const ModelBundle& candidate,
    const ModelBundle& opponent, const Options& o, const RunControl& control);

json::Object BaseRecord(const Options& o, const std::string& kind);
json::Object GameRecord(const GameOutcome& out, const Options& o);

int RunRoots(const Options& o, const Game& game, const ModelBundle& candidate,
             const RunControl& control, const std::set<int>& completed,
             std::ofstream& output);

TournamentRunStats RunTournamentGames(
    const Options& o, const Game& game, const ModelBundle& candidate,
    const ModelBundle& opponent, RunControl& control,
    const std::vector<int>& pending, int skipped, Summary& summary,
    std::ofstream& output);

}  // namespace open_spiel::dune_imperium::play_search_tournament

#endif  // OPEN_SPIEL_EXAMPLES_DUNE_PLAY_SEARCH_TOURNAMENT_INTERNAL_H_
