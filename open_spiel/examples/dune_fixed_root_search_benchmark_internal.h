#ifndef OPEN_SPIEL_EXAMPLES_DUNE_FIXED_ROOT_SEARCH_BENCHMARK_INTERNAL_H_
#define OPEN_SPIEL_EXAMPLES_DUNE_FIXED_ROOT_SEARCH_BENCHMARK_INTERNAL_H_

// Internal shared types, flag declarations, and cross-TU declarations for
// dune_fixed_root_search_benchmark.

#include <cstdint>
#include <fstream>
#include <memory>
#include <shared_mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "open_spiel/abseil-cpp/absl/flags/declare.h"
#include "open_spiel/examples/dune_batched_evaluator.h"
#include "open_spiel/examples/dune_compound_turn_search.h"
#include "open_spiel/examples/dune_network.h"
#include "open_spiel/examples/dune_semantic_action_scorer.h"
#include "open_spiel/examples/dune_wide_rollout_search.h"
#include "open_spiel/games/dune_imperium/dune_imperium.h"
#include "open_spiel/spiel.h"
#include "open_spiel/utils/json.h"

ABSL_DECLARE_FLAG(std::string, model_checkpoint);
ABSL_DECLARE_FLAG(std::string, sampled_roots_json);
ABSL_DECLARE_FLAG(std::string, obs_binary);
ABSL_DECLARE_FLAG(std::string, output_json);
ABSL_DECLARE_FLAG(int, threads);
ABSL_DECLARE_FLAG(int, num_evaluators);
ABSL_DECLARE_FLAG(int, target_batch_size);
ABSL_DECLARE_FLAG(int, top_k_actions);
ABSL_DECLARE_FLAG(double, min_override_margin);
ABSL_DECLARE_FLAG(uint64_t, master_seed);
ABSL_DECLARE_FLAG(std::string, replay_trajectory_jsonl);
ABSL_DECLARE_FLAG(std::string, replay_obs_bin);
ABSL_DECLARE_FLAG(int, max_roots);
ABSL_DECLARE_FLAG(bool, leaf_check_only);
ABSL_DECLARE_FLAG(bool, rollout_amp);
ABSL_DECLARE_FLAG(bool, allow_tf32);
ABSL_DECLARE_FLAG(double, temperature);
ABSL_DECLARE_FLAG(bool, dump_per_state);
ABSL_DECLARE_FLAG(int, num_blocks);
ABSL_DECLARE_FLAG(bool, sh_offline_check);
ABSL_DECLARE_FLAG(bool, sh_budget_check);
ABSL_DECLARE_FLAG(bool, skip_game_string_check);
ABSL_DECLARE_FLAG(bool, skip_recheck_1024);
ABSL_DECLARE_FLAG(std::string, base_receipt_json);

ABSL_DECLARE_FLAG(int, capture_self_play_games);
ABSL_DECLARE_FLAG(int, capture_roots_per_seat_game);
ABSL_DECLARE_FLAG(int, capture_pres_per_seat_game);
ABSL_DECLARE_FLAG(int, capture_max_placement_roots);
ABSL_DECLARE_FLAG(std::string, capture_out_dir);
ABSL_DECLARE_FLAG(bool, auto_pass_empty_combat_intrigue);

ABSL_DECLARE_FLAG(bool, sh_distill_label);
ABSL_DECLARE_FLAG(int, parallel_roots);
ABSL_DECLARE_FLAG(bool, use_bounded_pool_when_unclocked);
ABSL_DECLARE_FLAG(int, max_inflight_worlds);
ABSL_DECLARE_FLAG(bool, placement_roots_only);
ABSL_DECLARE_FLAG(uint64_t, label_seed_domain);

ABSL_DECLARE_FLAG(std::string, tie_breaker_games_jsonl);
ABSL_DECLARE_FLAG(bool, tie_breaker_run_rollouts);
ABSL_DECLARE_FLAG(int, tie_breaker_max_sample);
ABSL_DECLARE_FLAG(int, tie_breaker_rollouts);

namespace open_spiel::dune_imperium::fixed_root_benchmark {

namespace benchmark_domains {
constexpr uint64_t kDomainV0_Run1        = 0x0501;
constexpr uint64_t kDomainV0_Run2        = 0x0502;
constexpr uint64_t kDomainV1             = 0x0503;
constexpr uint64_t kDomainV2_16_Paired   = 0x0504;
constexpr uint64_t kDomainV2_16_Unpaired = 0x0505;
constexpr uint64_t kDomainV2_32_Paired   = 0x0506;
constexpr uint64_t kDomainV2_32_Unpaired = 0x0507;
constexpr uint64_t kDomainV2_64_Paired   = 0x0508;
constexpr uint64_t kDomainV2_64_Unpaired = 0x0509;
constexpr uint64_t kDomainRef512         = 0x050A;
constexpr uint64_t kDomainRecheck1024    = 0x050B;

constexpr uint64_t kDomainRaw            = 0x0601;
constexpr uint64_t kDomainSH192Incumbent = 0x0602;
constexpr uint64_t kDomainSH384Incumbent = 0x0603;
constexpr uint64_t kDomainSH384K5        = 0x0604;
constexpr uint64_t kDomainSHRef512       = 0x0605;

constexpr uint64_t kDomainSHBudgetShared  = 0x0701;
constexpr uint64_t kDomainSHBudgetRep     = 0x0702;
constexpr uint64_t kDomainSHBudgetRef2048 = 0x0703;
}  // namespace benchmark_domains

enum class BenchmarkVariantType {
  kCompound,
  kRaw,
  kSuccessiveHalvingDynamic,
  kSuccessiveHalvingFixedTop5,
  kPairedReferenceOnChosenActions,
};

struct VariantConfig {
  std::string name;
  int rollouts;
  SearchPairingMode pairing;
  SearchTruncationMode truncation;
  uint64_t seed_domain;
  BenchmarkVariantType type = BenchmarkVariantType::kCompound;
  int reference_rollouts_per_action = 0;
  int fixed_top_k = 0;
  bool all_buy = false;
};

struct SampledRoot {
  int root_id = 0;
  int game_id = 0;
  int decision_ordinal = 0;
  int obs_row = 0;
  int search_seat = 0;
  int round = 0;
  std::string round_band;
  std::string decision_type;
  double stratum_weight = 1.0;
  std::vector<int> legal_actions;
  int raw_action = 0;
  int census_chosen_action = 0;
  bool census_overridden = false;
  std::vector<int64_t> action_history;
  std::string split = "train";
  std::vector<double> pi_parent;
};

struct VariantResult {
  int chosen_action = 0;
  bool overridden = false;
  double raw_mean_utility = 0.0;
  double chosen_mean_utility = 0.0;
  std::vector<std::pair<int, double>> candidate_utilities;
  std::vector<std::pair<int, std::vector<double>>> candidate_rollout_utilities;
  uint64_t decision_seed = 0;
  uint64_t evaluations = 0;
  double duration_ms = 0.0;
  int stage1_planned_worlds = 0;
  int stage1_kept_worlds = 0;
  int stage2_planned_worlds = 0;
  int stage2_kept_worlds = 0;
  bool deadline_bound = false;
  std::vector<std::pair<int, double>> candidate_pair_sd_vs_raw;
};

struct EvaluatorBundle {
  torch::Device device{torch::kCPU};
  // Declare mutexes before coords/evaluators so destruction happens in safe reverse order.
  std::vector<std::unique_ptr<std::shared_mutex>> model_mutexes;
  std::vector<std::shared_ptr<open_spiel::SharedDunePolicyValueNetImpl>> models;
  std::vector<std::shared_ptr<open_spiel::BatchedEvaluator>> coords;
  std::vector<std::shared_ptr<open_spiel::BatchedNNEvaluator>> evaluators;
};

struct RootLoadVerifyOptions {
  bool verify_game_string = false;
  bool filter_placement_only = false;
  bool check_legal_during_replay = false;
  bool check_obs_stream_good = false;
  bool fill_missing_pi_parent = false;
  bool require_non_empty = false;
  bool log_loaded_count = false;
  bool verbose_parity_messages = false;
};

struct BenchmarkVariantSummary {
  std::string name;
  int rollouts = 0;
  std::string pairing_str;
  std::string mode_str;
  double agree_pct = 0.0;
  double mean_gain = 0.0;
  double avg_evals = 0.0;
  double avg_time_ms = 0.0;
  int override_count = 0;
};

struct BenchmarkSummaryReportData {
  std::vector<BenchmarkVariantSummary> variant_summaries;
  std::string ref_name;
  int ref_rollouts = 0;
  double ref_avg_evals = 0.0;
  double ref_avg_time_ms = 0.0;
};

struct LeafCheckStats {
  double correlation = 0.0;
  int64_t candidate_pairs = 0;
};

struct ReplaySubgroupMetrics {
  int64_t count = 0;
  int64_t greedy_changed_count = 0;
  int64_t greedy_agree_search_count = 0;
  double sum_prob_search_move = 0.0;
  double sum_max_p_diff_mu = 0.0;
  double sum_policy_entropy = 0.0;
  int64_t value_covered_count = 0;
  double sum_value_diff = 0.0;
  double sum_ce = 0.0;
};

struct HeldOutReplayStats {
  ReplaySubgroupMetrics ovr_stats;
  ReplaySubgroupMetrics non_ovr_stats;
  int64_t total_rows_read = 0;
  int64_t fresh_search_decisions = 0;
  double total_ce_all = 0.0;
  int64_t top1_matches = 0;
  double total_max_prob_diff = 0.0;
  std::unordered_map<int, double> round_ce_sum;
  std::unordered_map<int, int64_t> round_dec_count;
  open_spiel::json::Array dumped_decisions;
};

struct HeldOutReplayComputedMetrics {
  double mean_ce_all = 0.0;
  double top1_acc = 0.0;
  double mean_max_diff = 0.0;
  double override_pct = 0.0;
  double ovr_greedy_change_rate = 0.0;
  double ovr_greedy_agree_search = 0.0;
  double ovr_mean_prob_search = 0.0;
  double ovr_mean_max_p_diff = 0.0;
  double ovr_mean_entropy = 0.0;
  double ovr_val_changed_diff = 0.0;
  double mean_ce_override = 0.0;
  double non_ovr_greedy_change_rate = 0.0;
  double non_ovr_mean_max_p_diff = 0.0;
  double non_ovr_mean_entropy = 0.0;
  double non_ovr_val_changed_diff = 0.0;
  double mean_ce_non_override = 0.0;
};

struct CapturedPlacementRoot {
  int game_id = 0;
  int decision_ordinal = 0;
  int search_seat = 0;
  int round = 0;
  std::string round_band;
  std::string decision_type;
  std::vector<int> legal_actions;
  int raw_action = 0;
  std::vector<int64_t> action_history;
  std::string split;
  std::vector<double> pi_parent;
  std::vector<float> obs;
  dune_semantic::CandidateActionData cand_data;
};

struct CapturedPreservationItem {
  int game_id = 0;
  int split_id = 0;
  int player = 0;
  std::vector<float> obs;
  dune_semantic::CandidateActionData cand_data;
  std::vector<int> legal_actions;
  std::vector<double> pi_parent;
};

struct LabelledResult {
  int root_id = 0;
  int game_id = 0;
  int obs_row = 0;
  int search_seat = 0;
  int round = 0;
  std::string decision_type;
  std::string split;
  int raw_action = 0;
  int chosen_action = 0;
  bool overridden = false;
  double d2 = 0.0;
  double se2 = 0.0;
  double w = 0.0;
  std::vector<int> candidates;
  std::vector<int> legal_actions;
  std::vector<double> pi_parent;
  double duration_ms = 0.0;
  int stage1_kept_worlds = 0;
  int stage2_kept_worlds = 0;
};

struct DistillLabelStats {
  int total_overrides = 0;
  double override_rate = 0.0;
  double mean_w = 0.0;
  double total_wall_s = 0.0;
  double roots_per_s = 0.0;
};

struct ParsedGame {
  int episode_id = 0;
  int seat = 0;
  int placement = 0;
  double recorded_return = 0.0;
  int round = 0;
  uint64_t seed = 0;
  std::vector<int64_t> history;
};

struct ChangedDecision {
  int game_idx = 0;
  int seat = 0;
  Action a_parent = 0;
  Action a_student = 0;
  std::vector<int64_t> history_prefix;
};

struct TieBreakerStats {
  double mean_delta = 0.0;
  double dU = 0.0;
  double boot_se = 0.0;
  double ci_low = 0.0;
  double ci_high = 0.0;
};

template <typename T>
inline void WritePod(std::ofstream& ofs, const T& val) {
  ofs.write(reinterpret_cast<const char*>(&val), sizeof(T));
}

template <typename T>
inline void ReadPod(std::ifstream& ifs, T& val) {
  ifs.read(reinterpret_cast<char*>(&val), sizeof(T));
}

template <typename T>
inline void WriteVector(std::ofstream& ofs, const std::vector<T>& vec) {
  uint64_t sz = vec.size();
  WritePod(ofs, sz);
  if (sz > 0) {
    ofs.write(reinterpret_cast<const char*>(vec.data()), sz * sizeof(T));
  }
}

template <typename T>
inline void ReadVector(std::ifstream& ifs, std::vector<T>& vec) {
  uint64_t sz = 0;
  ReadPod(ifs, sz);
  vec.resize(sz);
  if (sz > 0) {
    ifs.read(reinterpret_cast<char*>(vec.data()), sz * sizeof(T));
  }
}

// Setup functions (dune_fixed_root_search_benchmark_setup.cc)
EvaluatorBundle InitEvaluatorBundle(
    int num_evals,
    int target_batch_size,
    int num_blocks,
    const std::string& model_path,
    bool rollout_amp,
    bool allow_tf32,
    bool disable_global_tf32,
    bool log_device);

VariantResult ParseVariantResultFromJson(const open_spiel::json::Object& vr_obj);

bool LoadBenchmarkCheckpoint(
    const std::string& ckpt_path,
    const std::vector<VariantConfig>& variants,
    std::vector<std::vector<VariantResult>>& all_results,
    size_t& completed_roots);

bool LoadBaseReceiptVariants(
    const std::string& base_receipt_path,
    const std::vector<SampledRoot>& sampled_roots,
    const std::vector<VariantConfig>& variants,
    std::vector<std::vector<VariantResult>>& all_results,
    std::vector<std::vector<bool>>& has_base_variant);

std::vector<SampledRoot> LoadSampledRootsJson(
    const std::string& roots_path,
    const std::string& current_game_str,
    const RootLoadVerifyOptions& opts);

void VerifySampledRootsObsParity(
    const open_spiel::Game& game,
    const std::string& obs_path,
    const std::shared_ptr<open_spiel::BatchedNNEvaluator>& evaluator,
    const RootLoadVerifyOptions& opts,
    std::vector<SampledRoot>& sampled_roots);

std::vector<VariantConfig> BuildBenchmarkVariants();

void ReadPreservationItem(std::ifstream& ifs, CapturedPreservationItem& item);

int LoadPreservationCheckpoint(
    const std::string& ckpt_path,
    int num_games,
    std::vector<std::vector<CapturedPreservationItem>>& per_game_pres,
    std::atomic<uint8_t>* game_completed);

int LoadLabelCheckpoint(
    const std::string& ckpt_path,
    const std::vector<SampledRoot>& sampled_roots,
    std::vector<LabelledResult>& results,
    std::atomic<uint8_t>* completed);

std::vector<ParsedGame> LoadTieBreakerGames(const std::string& games_jsonl_path);

// Stats functions (dune_fixed_root_search_benchmark_stats.cc)
double ComputePairedRolloutSdVsRaw(
    const std::vector<double>& raw_rollouts,
    const std::vector<double>& cand_rollouts);

void ReevaluateExtendedRef2048Choice(
    double min_override_margin,
    VariantResult& cached_ref);

LeafCheckStats ComputeLeafCheckStats(
    const std::vector<SampledRoot>& sampled_roots,
    const std::vector<std::vector<VariantResult>>& all_results);

BenchmarkSummaryReportData ComputeBenchmarkSummaryReportData(
    const std::vector<SampledRoot>& sampled_roots,
    const std::vector<VariantConfig>& variants,
    const std::vector<std::vector<VariantResult>>& all_results,
    size_t ref_idx);

void AccumulateHeldOutReplayDecision(
    int round,
    int64_t obs_row,
    bool is_override,
    Action raw_a,
    Action best_cand_action,
    Action argmax_mu,
    Action model_argmax,
    double policy_entropy,
    const std::vector<Action>& legal_actions,
    const std::vector<float>& capped_logits,
    const std::unordered_map<Action, double>& model_probs,
    const std::unordered_map<Action, double>& mu_map,
    const std::vector<std::pair<Action, double>>& candidate_utilities,
    const std::unordered_map<Action, double>& cand_util_map,
    const std::vector<double>& target_dist,
    HeldOutReplayStats& stats);

HeldOutReplayComputedMetrics ComputeHeldOutReplayMetrics(const HeldOutReplayStats& stats);

DistillLabelStats ComputeDistillLabelStats(
    const std::vector<LabelledResult>& results,
    double total_wall_s);

TieBreakerStats ComputeTieBreakerBootstrapStats(
    size_t num_games,
    uint64_t master_seed,
    const std::vector<std::vector<ChangedDecision>>& per_game_changed,
    const std::vector<ChangedDecision>& sampled_decisions,
    const std::vector<double>& deltas);

// Report functions (dune_fixed_root_search_benchmark_report.cc)
void SaveBenchmarkReceipt(
    const std::string& output_path,
    const open_spiel::json::Object& root_out_json,
    const std::vector<SampledRoot>& sampled_roots,
    const std::vector<VariantConfig>& variants,
    const std::vector<std::vector<VariantResult>>& all_results,
    const std::vector<VariantResult>& recheck_1024_results,
    size_t num_roots_completed);

void ReportLeafCheckAndSave(
    const std::string& output_path,
    const std::string& checkpoint_path,
    const LeafCheckStats& leaf_stats,
    const std::vector<SampledRoot>& sampled_roots,
    const std::vector<VariantConfig>& variants,
    const std::vector<std::vector<VariantResult>>& all_results);

open_spiel::json::Object PrintAndBuildMainBenchmarkReport(
    const std::string& output_path,
    const BenchmarkSummaryReportData& summary_data,
    const std::vector<SampledRoot>& sampled_roots,
    const std::vector<VariantConfig>& variants,
    const std::vector<std::vector<VariantResult>>& all_results);

void PrintAndSaveHeldOutReplayReport(
    const std::string& model_path,
    const std::string& trajectory_jsonl_path,
    const std::string& obs_bin_path,
    const std::string& output_json_path,
    const HeldOutReplayStats& stats,
    const HeldOutReplayComputedMetrics& metrics);

void WritePreservationItem(std::ofstream& ofs, const CapturedPreservationItem& item);

void SavePreservationCheckpoint(
    const std::string& out_dir,
    int num_games,
    const std::atomic<uint8_t>* game_completed,
    const std::vector<std::vector<CapturedPreservationItem>>& per_game_pres);

void SaveCaptureOutputs(
    const open_spiel::Game& game,
    const std::string& out_dir,
    int num_games,
    uint64_t base_seed,
    const std::vector<CapturedPlacementRoot>& all_sampled_roots,
    const std::vector<CapturedPreservationItem>& all_pres,
    double elapsed_s);

open_spiel::json::Object LabelledResultToJson(const LabelledResult& lr);

void SaveDistillLabelCheckpoint(
    const std::string& path,
    size_t total_roots,
    int completed_count,
    const std::vector<LabelledResult>& results,
    const std::atomic<uint8_t>* completed);

void SaveDistillLabelFinalOutput(
    const std::string& output_path,
    const std::string& ckpt_path,
    const std::vector<LabelledResult>& results,
    const DistillLabelStats& stats);

void PrintAndSaveTieBreakerSummary(
    const std::string& output_path,
    size_t num_games,
    size_t num_changed,
    double r_changed,
    bool has_rollout_stats,
    size_t sampled_count,
    int tie_breaker_rollouts,
    const TieBreakerStats& tb_stats);

// Runner functions (dune_fixed_root_search_benchmark_runner.cc)
int RunFixedRootBenchmark();

// Auxiliary mode functions (dune_fixed_root_search_benchmark_modes.cc)
int RunHeldOutReplayEvaluation(
    const std::string& model_path,
    const std::string& trajectory_jsonl_path,
    const std::string& obs_bin_path,
    const std::string& output_json_path);

int RunSelfPlayRootAndPreservationCapture();
int RunSuccessiveHalvingDistillLabel();
int RunTrueStateTieBreakerAndHistoryReplay();

}  // namespace open_spiel::dune_imperium::fixed_root_benchmark

#endif  // OPEN_SPIEL_EXAMPLES_DUNE_FIXED_ROOT_SEARCH_BENCHMARK_INTERNAL_H_
