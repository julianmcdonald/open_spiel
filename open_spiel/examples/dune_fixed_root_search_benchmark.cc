// Step 1: Fixed-Root Search Benchmark Harness
// Evaluates search variants against a 512-rollout paired reference across ~300 stratified census roots,
// followed by a 1,024-rollout re-check pass on disagreement roots.

#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <mutex>
#include <numeric>
#include <random>
#include <string>
#include <vector>
#include <atomic>
#include <thread>
#include <sstream>

#include "open_spiel/abseil-cpp/absl/flags/flag.h"
#include "open_spiel/abseil-cpp/absl/flags/parse.h"
#include "open_spiel/abseil-cpp/absl/strings/str_format.h"
#include "open_spiel/games/dune_imperium/dune_imperium.h"
#include "open_spiel/examples/dune_batched_evaluator.h"
#include "open_spiel/examples/dune_compound_turn_search.h"
#include "open_spiel/examples/dune_network.h"
#include "open_spiel/examples/dune_seed_utils.h"
#include "open_spiel/examples/dune_semantic_action_scorer.h"
#include "open_spiel/examples/dune_wide_rollout_search.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"
#include "open_spiel/utils/json.h"

using namespace open_spiel;
using namespace open_spiel::dune_imperium;

ABSL_FLAG(std::string, model_checkpoint, "", "Path to model checkpoint.");
ABSL_FLAG(std::string, sampled_roots_json, "", "Path to census_sampled_roots.json.");
ABSL_FLAG(std::string, obs_binary, "", "Path to census_obs.bin matching sampled roots.");
ABSL_FLAG(std::string, output_json, "", "Path to output JSON receipt (required).");
ABSL_FLAG(int, threads, 14, "Number of worker threads.");
ABSL_FLAG(int, num_evaluators, 6, "Number of batched evaluator coordinators.");
ABSL_FLAG(int, target_batch_size, 128, "Evaluator batch size.");
ABSL_FLAG(int, top_k_actions, 3, "Number of top candidates.");
ABSL_FLAG(double, min_override_margin, 0.15, "Override margin threshold in utility units.");
ABSL_FLAG(uint64_t, master_seed, 20261001ULL, "Master seed for domain-separated derivation.");
ABSL_FLAG(std::string, replay_trajectory_jsonl, "",
          "Path to recorded tournament trajectory_meta.jsonl for held-out policy cross-entropy evaluation.");
ABSL_FLAG(std::string, replay_obs_bin, "",
          "Path to recorded tournament trajectory_obs.bin matching replay_trajectory_jsonl.");
ABSL_FLAG(int, max_roots, -1, "Maximum number of roots to benchmark (-1 for all).");
ABSL_FLAG(bool, leaf_check_only, false, "Run only V2_16_Paired vs Ref512 for leaf check.");
ABSL_FLAG(bool, rollout_amp, false, "Use AMP in evaluator rollouts (default false).");
ABSL_FLAG(bool, allow_tf32, false, "Allow TF32 in evaluator (default false).");
ABSL_FLAG(double, temperature, 1.0, "Temperature applied to capped logits in replay mode.");
ABSL_FLAG(bool, dump_per_state, false, "Dump per-state candidate probabilities and entropy in output JSON.");
ABSL_FLAG(int, num_blocks, 8, "Number of residual blocks in model (8, 12, or 16).");
ABSL_FLAG(bool, sh_offline_check, false,
          "Run R7-39 Step 6 Successive Halving exploratory offline check (raw, sh_192_incumbent, sh_384_incumbent, sh_384_k5, Ref512).");
ABSL_FLAG(bool, sh_budget_check, false,
          "Run fixed-position SH budget check (raw, sh_192, sh_384, sh_768, sh_768_k5, sh_768_rep, Ref2048).");
ABSL_FLAG(bool, skip_game_string_check, false,
          "Allow mismatched game_string metadata; replay, seat and observation checks remain strict.");
ABSL_FLAG(bool, skip_recheck_1024, false,
          "Skip the 1,024-rollout disagreement re-check pass.");
ABSL_FLAG(std::string, base_receipt_json, "",
          "Optional path to existing sh_budget_check_sweep_receipt.json to reuse completed variants and incrementally extend Ref2048.");

// Mode 1: Raw Greedy Self-Play Root & Preservation Capture Mode
ABSL_FLAG(int, capture_self_play_games, 0,
          "Number of 4x greedy self-play games to capture placement roots and preservation pool from.");
ABSL_FLAG(int, capture_roots_per_seat_game, 2, "Max placement roots sampled per seat-game.");
ABSL_FLAG(int, capture_pres_per_seat_game, 4,
          "Max preservation decisions sampled per seat-game across all non-forced decision types.");
ABSL_FLAG(int, capture_max_placement_roots, 0,
          "Optional cap on total sampled placement roots across all games (0 = no cap).");
ABSL_FLAG(std::string, capture_out_dir, "",
          "Output directory for captured roots and preservation dataset.");
ABSL_FLAG(bool, auto_pass_empty_combat_intrigue, true,
          "Value for DuneImperiumState::SetAutoPassEmptyCombatIntrigueHands (false for legacy R7-41 roots).");

// Mode 2: Parallel-Roots sh_768_k5_allbuy Labelling Mode
ABSL_FLAG(bool, sh_distill_label, false, "Run parallel-roots sh_768_k5_allbuy labelling mode.");
ABSL_FLAG(int, parallel_roots, 1, "Number of roots to search concurrently across worker threads.");
ABSL_FLAG(bool, use_bounded_pool_when_unclocked, false,
          "Use bounded world pool in RunClockedPairedRolloutStage when max_time_ms <= 0.");
ABSL_FLAG(int, max_inflight_worlds, 32, "Max inflight worlds per root when using bounded pool.");
ABSL_FLAG(bool, placement_roots_only, false,
          "Filter loaded roots to AgentPrimary and AgentContinuation only.");
ABSL_FLAG(uint64_t, label_seed_domain, 0x0701ULL, "Seed domain for sh_distill_label.");

// Mode 3: True-State Tie-Breaker & History Replay Verification Mode
ABSL_FLAG(std::string, tie_breaker_games_jsonl, "",
          "Path to games.jsonl from dune_population_eval for history replay verification and Step 1.6 tie-breaker.");
ABSL_FLAG(bool, tie_breaker_run_rollouts, false,
          "If true (and r <= 1.5 or forced), run 512 paired true-state rollouts on up to 300 changed student decisions.");
ABSL_FLAG(int, tie_breaker_max_sample, 300,
          "Max changed student decisions to sample for true-state rollouts.");
ABSL_FLAG(int, tie_breaker_rollouts, 512, "Paired full-game rollouts per move in tie-breaker.");

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

void SaveBenchmarkReceipt(
    const std::string& output_path,
    const open_spiel::json::Object& root_out_json,
    const std::vector<SampledRoot>& sampled_roots,
    const std::vector<VariantConfig>& variants,
    const std::vector<std::vector<VariantResult>>& all_results,
    const std::vector<VariantResult>& recheck_1024_results,
    size_t num_roots_completed) {
  open_spiel::json::Object current_root_json = root_out_json;
  open_spiel::json::Array roots_results_arr;
  roots_results_arr.reserve(num_roots_completed);

  for (size_t r_idx = 0; r_idx < num_roots_completed; ++r_idx) {
    const auto& root = sampled_roots[r_idx];
    open_spiel::json::Object r_obj;
    r_obj["root_id"] = open_spiel::json::Value(static_cast<int64_t>(root.root_id));
    r_obj["game_id"] = open_spiel::json::Value(static_cast<int64_t>(root.game_id));
    r_obj["round"] = open_spiel::json::Value(static_cast<int64_t>(root.round));
    r_obj["round_band"] = open_spiel::json::Value(root.round_band);
    r_obj["decision_type"] = open_spiel::json::Value(root.decision_type);
    r_obj["stratum_weight"] = open_spiel::json::Value(root.stratum_weight);
    r_obj["raw_action"] = open_spiel::json::Value(static_cast<int64_t>(root.raw_action));

    open_spiel::json::Object var_map;
    for (size_t v_idx = 0; v_idx < variants.size(); ++v_idx) {
      const auto& vr = all_results[r_idx][v_idx];
      open_spiel::json::Object vr_obj;
      vr_obj["chosen_action"] = open_spiel::json::Value(static_cast<int64_t>(vr.chosen_action));
      vr_obj["overridden"] = open_spiel::json::Value(vr.overridden);
      vr_obj["raw_mean_utility"] = open_spiel::json::Value(vr.raw_mean_utility);
      vr_obj["chosen_mean_utility"] = open_spiel::json::Value(vr.chosen_mean_utility);
      vr_obj["decision_seed"] = open_spiel::json::Value(static_cast<int64_t>(vr.decision_seed));
      vr_obj["evaluations"] = open_spiel::json::Value(static_cast<int64_t>(vr.evaluations));
      vr_obj["duration_ms"] = open_spiel::json::Value(vr.duration_ms);
      vr_obj["stage1_planned_worlds"] = open_spiel::json::Value(static_cast<int64_t>(vr.stage1_planned_worlds));
      vr_obj["stage1_kept_worlds"] = open_spiel::json::Value(static_cast<int64_t>(vr.stage1_kept_worlds));
      vr_obj["stage2_planned_worlds"] = open_spiel::json::Value(static_cast<int64_t>(vr.stage2_planned_worlds));
      vr_obj["stage2_kept_worlds"] = open_spiel::json::Value(static_cast<int64_t>(vr.stage2_kept_worlds));
      vr_obj["deadline_bound"] = open_spiel::json::Value(vr.deadline_bound);

      open_spiel::json::Array sd_arr;
      for (const auto& sd_pair : vr.candidate_pair_sd_vs_raw) {
        open_spiel::json::Array pair_arr;
        pair_arr.push_back(open_spiel::json::Value(static_cast<int64_t>(sd_pair.first)));
        pair_arr.push_back(open_spiel::json::Value(sd_pair.second));
        sd_arr.push_back(open_spiel::json::Value(pair_arr));
      }
      vr_obj["candidate_pair_sd_vs_raw"] = open_spiel::json::Value(sd_arr);

      open_spiel::json::Array cu_arr;
      for (const auto& cu : vr.candidate_utilities) {
        open_spiel::json::Array pair_arr;
        pair_arr.push_back(open_spiel::json::Value(static_cast<int64_t>(cu.first)));
        pair_arr.push_back(open_spiel::json::Value(cu.second));
        cu_arr.push_back(open_spiel::json::Value(pair_arr));
      }
      vr_obj["candidate_utilities"] = open_spiel::json::Value(cu_arr);

      open_spiel::json::Array cru_arr;
      for (const auto& cru : vr.candidate_rollout_utilities) {
        open_spiel::json::Array pair_arr;
        pair_arr.push_back(open_spiel::json::Value(static_cast<int64_t>(cru.first)));
        open_spiel::json::Array vals_arr;
        vals_arr.reserve(cru.second.size());
        for (double u : cru.second) {
          vals_arr.push_back(open_spiel::json::Value(u));
        }
        pair_arr.push_back(open_spiel::json::Value(vals_arr));
        cru_arr.push_back(open_spiel::json::Value(pair_arr));
      }
      vr_obj["candidate_rollout_utilities"] = open_spiel::json::Value(cru_arr);

      var_map[variants[v_idx].name] = open_spiel::json::Value(vr_obj);
    }

    if (r_idx < recheck_1024_results.size() && recheck_1024_results[r_idx].chosen_action != 0) {
      const auto& rk = recheck_1024_results[r_idx];
      open_spiel::json::Object rk_obj;
      rk_obj["chosen_action"] = open_spiel::json::Value(static_cast<int64_t>(rk.chosen_action));
      rk_obj["overridden"] = open_spiel::json::Value(rk.overridden);
      rk_obj["raw_mean_utility"] = open_spiel::json::Value(rk.raw_mean_utility);
      rk_obj["chosen_mean_utility"] = open_spiel::json::Value(rk.chosen_mean_utility);
      rk_obj["decision_seed"] = open_spiel::json::Value(static_cast<int64_t>(rk.decision_seed));
      rk_obj["evaluations"] = open_spiel::json::Value(static_cast<int64_t>(rk.evaluations));
      rk_obj["duration_ms"] = open_spiel::json::Value(rk.duration_ms);

      open_spiel::json::Array cu_arr;
      for (const auto& cu : rk.candidate_utilities) {
        open_spiel::json::Array pair_arr;
        pair_arr.push_back(open_spiel::json::Value(static_cast<int64_t>(cu.first)));
        pair_arr.push_back(open_spiel::json::Value(cu.second));
        cu_arr.push_back(open_spiel::json::Value(pair_arr));
      }
      rk_obj["candidate_utilities"] = open_spiel::json::Value(cu_arr);

      open_spiel::json::Array cru_arr;
      for (const auto& cru : rk.candidate_rollout_utilities) {
        open_spiel::json::Array pair_arr;
        pair_arr.push_back(open_spiel::json::Value(static_cast<int64_t>(cru.first)));
        open_spiel::json::Array vals_arr;
        vals_arr.reserve(cru.second.size());
        for (double u : cru.second) {
          vals_arr.push_back(open_spiel::json::Value(u));
        }
        pair_arr.push_back(open_spiel::json::Value(vals_arr));
        cru_arr.push_back(open_spiel::json::Value(pair_arr));
      }
      rk_obj["candidate_rollout_utilities"] = open_spiel::json::Value(cru_arr);

      var_map["Ref1024_Recheck"] = open_spiel::json::Value(rk_obj);
    }

    r_obj["variants"] = open_spiel::json::Value(var_map);
    roots_results_arr.push_back(open_spiel::json::Value(r_obj));
  }
  current_root_json["roots_results"] = open_spiel::json::Value(roots_results_arr);

  std::string tmp_path = output_path + ".tmp";
  std::ofstream out_file(tmp_path);
  if (out_file.is_open()) {
    out_file << open_spiel::json::ToString(current_root_json);
    out_file.close();
    std::rename(tmp_path.c_str(), output_path.c_str());
  }
}

VariantResult ParseVariantResultFromJson(const open_spiel::json::Object& vr_obj) {
  VariantResult vr;
  if (vr_obj.find("chosen_action") != vr_obj.end()) {
    vr.chosen_action = static_cast<int>(vr_obj.at("chosen_action").GetInt());
  }
  if (vr_obj.find("overridden") != vr_obj.end()) {
    vr.overridden = vr_obj.at("overridden").GetBool();
  }
  if (vr_obj.find("raw_mean_utility") != vr_obj.end()) {
    vr.raw_mean_utility = vr_obj.at("raw_mean_utility").IsDouble()
        ? vr_obj.at("raw_mean_utility").GetDouble()
        : static_cast<double>(vr_obj.at("raw_mean_utility").GetInt());
  }
  if (vr_obj.find("chosen_mean_utility") != vr_obj.end()) {
    vr.chosen_mean_utility = vr_obj.at("chosen_mean_utility").IsDouble()
        ? vr_obj.at("chosen_mean_utility").GetDouble()
        : static_cast<double>(vr_obj.at("chosen_mean_utility").GetInt());
  }
  if (vr_obj.find("decision_seed") != vr_obj.end()) {
    vr.decision_seed = static_cast<uint64_t>(vr_obj.at("decision_seed").GetInt());
  }
  if (vr_obj.find("evaluations") != vr_obj.end()) {
    vr.evaluations = static_cast<uint64_t>(vr_obj.at("evaluations").GetInt());
  }
  if (vr_obj.find("duration_ms") != vr_obj.end()) {
    vr.duration_ms = vr_obj.at("duration_ms").IsDouble()
        ? vr_obj.at("duration_ms").GetDouble()
        : static_cast<double>(vr_obj.at("duration_ms").GetInt());
  }
  if (vr_obj.find("stage1_planned_worlds") != vr_obj.end()) {
    vr.stage1_planned_worlds = static_cast<int>(vr_obj.at("stage1_planned_worlds").GetInt());
  }
  if (vr_obj.find("stage1_kept_worlds") != vr_obj.end()) {
    vr.stage1_kept_worlds = static_cast<int>(vr_obj.at("stage1_kept_worlds").GetInt());
  }
  if (vr_obj.find("stage2_planned_worlds") != vr_obj.end()) {
    vr.stage2_planned_worlds = static_cast<int>(vr_obj.at("stage2_planned_worlds").GetInt());
  }
  if (vr_obj.find("stage2_kept_worlds") != vr_obj.end()) {
    vr.stage2_kept_worlds = static_cast<int>(vr_obj.at("stage2_kept_worlds").GetInt());
  }
  if (vr_obj.find("deadline_bound") != vr_obj.end()) {
    vr.deadline_bound = vr_obj.at("deadline_bound").GetBool();
  }
  if (vr_obj.find("candidate_pair_sd_vs_raw") != vr_obj.end()) {
    for (const auto& sd_val : vr_obj.at("candidate_pair_sd_vs_raw").GetArray()) {
      const auto& pair_arr = sd_val.GetArray();
      int a = static_cast<int>(pair_arr[0].GetInt());
      double sd = pair_arr[1].IsDouble() ? pair_arr[1].GetDouble() : static_cast<double>(pair_arr[1].GetInt());
      vr.candidate_pair_sd_vs_raw.push_back({a, sd});
    }
  }
  if (vr_obj.find("candidate_utilities") != vr_obj.end()) {
    for (const auto& cu_val : vr_obj.at("candidate_utilities").GetArray()) {
      const auto& pair_arr = cu_val.GetArray();
      int a = static_cast<int>(pair_arr[0].GetInt());
      double u = pair_arr[1].IsDouble() ? pair_arr[1].GetDouble() : static_cast<double>(pair_arr[1].GetInt());
      vr.candidate_utilities.push_back({a, u});
    }
  }
  if (vr_obj.find("candidate_rollout_utilities") != vr_obj.end()) {
    for (const auto& cru_val : vr_obj.at("candidate_rollout_utilities").GetArray()) {
      const auto& pair_arr = cru_val.GetArray();
      int a = static_cast<int>(pair_arr[0].GetInt());
      std::vector<double> vals;
      for (const auto& u_val : pair_arr[1].GetArray()) {
        vals.push_back(u_val.IsDouble() ? u_val.GetDouble() : static_cast<double>(u_val.GetInt()));
      }
      vr.candidate_rollout_utilities.push_back({a, vals});
    }
  }
  return vr;
}

bool LoadBenchmarkCheckpoint(
    const std::string& ckpt_path,
    const std::vector<VariantConfig>& variants,
    std::vector<std::vector<VariantResult>>& all_results,
    size_t& completed_roots) {
  std::ifstream f(ckpt_path);
  if (!f.is_open()) return false;
  std::string content((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
  f.close();
  auto json_opt = open_spiel::json::FromString(content);
  if (!json_opt || !json_opt->IsObject()) return false;
  const auto& obj = json_opt->GetObject();
  if (obj.find("roots_results") == obj.end()) return false;
  const auto& roots_arr = obj.at("roots_results").GetArray();
  if (roots_arr.empty()) return false;

  for (size_t r_idx = 0; r_idx < roots_arr.size(); ++r_idx) {
    if (r_idx >= all_results.size()) break;
    const auto& r_obj = roots_arr[r_idx].GetObject();
    if (r_obj.find("variants") == r_obj.end()) break;
    const auto& var_map = r_obj.at("variants").GetObject();

    bool root_complete = true;
    for (size_t v_idx = 0; v_idx < variants.size(); ++v_idx) {
      const std::string& v_name = variants[v_idx].name;
      if (var_map.find(v_name) == var_map.end()) {
        root_complete = false;
        break;
      }
      const auto& vr_obj = var_map.at(v_name).GetObject();
      all_results[r_idx][v_idx] = ParseVariantResultFromJson(vr_obj);
    }
    if (root_complete) {
      completed_roots = r_idx + 1;
    } else {
      break;
    }
  }
  return completed_roots > 0;
}

bool LoadBaseReceiptVariants(
    const std::string& base_receipt_path,
    const std::vector<SampledRoot>& sampled_roots,
    const std::vector<VariantConfig>& variants,
    std::vector<std::vector<VariantResult>>& all_results,
    std::vector<std::vector<bool>>& has_base_variant) {
  std::ifstream f(base_receipt_path);
  if (!f.is_open()) {
    std::cerr << "[BASE-RECEIPT] Error: Could not open " << base_receipt_path << "\n";
    return false;
  }
  std::string content((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
  f.close();
  auto json_opt = open_spiel::json::FromString(content);
  if (!json_opt || !json_opt->IsObject()) {
    std::cerr << "[BASE-RECEIPT] Error: Failed to parse JSON in " << base_receipt_path << "\n";
    return false;
  }
  const auto& obj = json_opt->GetObject();
  if (obj.find("roots_results") == obj.end()) {
    std::cerr << "[BASE-RECEIPT] Error: 'roots_results' key missing in " << base_receipt_path << "\n";
    return false;
  }
  const auto& roots_arr = obj.at("roots_results").GetArray();
  if (roots_arr.empty()) {
    std::cerr << "[BASE-RECEIPT] Warning: 'roots_results' is empty in " << base_receipt_path << "\n";
    return false;
  }

  std::unordered_map<int, size_t> root_id_to_idx;
  for (size_t r = 0; r < sampled_roots.size(); ++r) {
    root_id_to_idx[sampled_roots[r].root_id] = r;
  }
  std::unordered_map<std::string, size_t> variant_name_to_idx;
  for (size_t v = 0; v < variants.size(); ++v) {
    variant_name_to_idx[variants[v].name] = v;
  }

  size_t loaded_count = 0;
  size_t root_match_count = 0;
  for (size_t i = 0; i < roots_arr.size(); ++i) {
    const auto& r_obj = roots_arr[i].GetObject();
    if (r_obj.find("root_id") == r_obj.end() || r_obj.find("variants") == r_obj.end()) {
      continue;
    }
    int root_id = static_cast<int>(r_obj.at("root_id").GetInt());
    auto it_r = root_id_to_idx.find(root_id);
    if (it_r == root_id_to_idx.end()) {
      continue;
    }
    size_t r_idx = it_r->second;
    root_match_count++;
    const auto& var_map = r_obj.at("variants").GetObject();
    for (const auto& [v_name, vr_val] : var_map) {
      auto it_v = variant_name_to_idx.find(v_name);
      if (it_v == variant_name_to_idx.end()) {
        continue;
      }
      size_t v_idx = it_v->second;
      all_results[r_idx][v_idx] = ParseVariantResultFromJson(vr_val.GetObject());
      has_base_variant[r_idx][v_idx] = true;
      loaded_count++;
    }
  }

  std::cout << absl::StrFormat(
      "[BASE-RECEIPT] Loaded %zu variant results across %zu matched roots from %s\n",
      loaded_count, root_match_count, base_receipt_path);
  return true;
}

int RunHeldOutReplayEvaluation(
    const std::string& model_path,
    const std::string& trajectory_jsonl_path,
    const std::string& obs_bin_path,
    const std::string& output_json_path) {
  std::cout << "================================================================================\n";
  std::cout << "HELD-OUT TRAJECTORY REPLAY POLICY EVALUATION\n";
  std::cout << "================================================================================\n";
  std::cout << "Model checkpoint:       " << model_path << "\n";
  std::cout << "Replay trajectory meta: " << trajectory_jsonl_path << "\n";
  std::cout << "Replay observation bin: " << obs_bin_path << "\n";
  if (!output_json_path.empty()) {
    std::cout << "Output JSON receipt:    " << output_json_path << "\n";
  }

  // Load Game
  std::shared_ptr<const open_spiel::Game> game = open_spiel::LoadGame("dune_imperium");

  // Open observation binary file
  std::ifstream obs_file(obs_bin_path, std::ios::binary);
  if (!obs_file.is_open()) {
    open_spiel::SpielFatalError("Failed to open observation binary: " + obs_bin_path);
  }
  const size_t obs_elem_count = open_spiel::dune_imperium::kFullPublicInformationStateSize;
  const size_t obs_byte_stride = obs_elem_count * sizeof(float);

  // Open trajectory JSONL
  std::ifstream meta_file(trajectory_jsonl_path);
  if (!meta_file.is_open()) {
    open_spiel::SpielFatalError("Failed to open trajectory meta jsonl: " + trajectory_jsonl_path);
  }

  // Setup Device & Model Evaluator
  torch::Device device(torch::kCUDA, 0);
  if (!torch::cuda::is_available()) {
    device = torch::Device(torch::kCPU);
  }
  std::cout << "[DEVICE] Using device: " << (device.is_cuda() ? "CUDA" : "CPU") << "\n";

  auto model = std::make_shared<open_spiel::SharedDunePolicyValueNetImpl>(
      open_spiel::dune_imperium::kFullPublicInformationStateSize, 2048, 2391,
      absl::GetFlag(FLAGS_num_blocks),
      /*nonlinear_value_head=*/false, /*with_aux_heads=*/false,
      /*head_init_seed=*/0, /*with_semantic_scorer=*/true);
  model->to(device);
  open_spiel::LoadModelCheckpointRobust(model, model_path, device);
  model->eval();

  std::shared_mutex model_mutex;
  auto coord = std::make_shared<open_spiel::BatchedEvaluator>(
      model, 64, /*timeout_ms=*/1, device, &model_mutex, 10.0f,
      /*device_synchronize=*/false, /*high_priority_stream=*/true,
      /*emit_batch_membership=*/false, /*rollout_amp=*/true, /*allow_tf32=*/true);
  auto evaluator = std::make_shared<open_spiel::BatchedNNEvaluator>(coord, 10.0f);
  std::cout << "[MODEL] Successfully initialized policy evaluator with semantic scoring.\n";

  struct SubgroupMetrics {
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

  SubgroupMetrics ovr_stats;
  SubgroupMetrics non_ovr_stats;

  std::string line;
  int64_t total_rows_read = 0;
  int64_t fresh_search_decisions = 0;
  double total_ce_all = 0.0;
  int64_t top1_matches = 0;
  double total_max_prob_diff = 0.0;

  // Track breakdown by round
  std::unordered_map<int, double> round_ce_sum;
  std::unordered_map<int, int64_t> round_dec_count;

  std::vector<float> obs_buffer(obs_elem_count);
  open_spiel::json::Array dumped_decisions;

  while (std::getline(meta_file, line)) {
    if (line.empty()) continue;
    ++total_rows_read;
    auto json_opt = open_spiel::json::FromString(line);
    if (!json_opt || !json_opt->IsObject()) continue;
    const auto& obj = json_opt->GetObject();

    // Filter strictly to decision_role == "fresh_search"
    std::string decision_role = "";
    auto it_role = obj.find("decision_role");
    if (it_role != obj.end() && it_role->second.IsString()) {
      decision_role = it_role->second.GetString();
    }
    if (decision_role != "fresh_search") continue;

    int search_seat = static_cast<int>(obj.at("search_seat").GetInt());
    int round = static_cast<int>(obj.at("round").GetInt());
    int64_t obs_row = obj.at("obs_row").GetInt();

    // Parse legal_actions
    std::vector<Action> legal_actions;
    if (obj.find("legal_actions") != obj.end() && obj.at("legal_actions").IsArray()) {
      for (const auto& a_val : obj.at("legal_actions").GetArray()) {
        legal_actions.push_back(static_cast<Action>(a_val.GetInt()));
      }
    }
    if (legal_actions.empty()) continue;

    // Parse recorded mu (U28700 raw prior)
    std::vector<std::pair<Action, double>> raw_policy;
    std::unordered_map<Action, double> mu_map;
    Action argmax_mu = -1;
    double max_mu = -1.0;
    if (obj.find("mu") != obj.end() && obj.at("mu").IsObject()) {
      for (const auto& kv : obj.at("mu").GetObject()) {
        Action a = static_cast<Action>(std::stoi(kv.first));
        double prob = kv.second.GetDouble();
        raw_policy.push_back({a, prob});
        mu_map[a] = prob;
        if (prob > max_mu) {
          max_mu = prob;
          argmax_mu = a;
        }
      }
    }
    if (raw_policy.empty() || argmax_mu == -1) continue;

    // Parse candidate_utilities
    std::vector<std::pair<Action, double>> candidate_utilities;
    std::unordered_map<Action, double> cand_util_map;
    if (obj.find("candidate_utilities") != obj.end() && obj.at("candidate_utilities").IsArray()) {
      for (const auto& item : obj.at("candidate_utilities").GetArray()) {
        if (item.IsArray() && item.GetArray().size() >= 2) {
          Action a = static_cast<Action>(item.GetArray()[0].GetInt());
          double u = item.GetArray()[1].GetDouble();
          candidate_utilities.push_back({a, u});
          cand_util_map[a] = u;
        }
      }
    }
    if (candidate_utilities.empty()) continue;

    // Build target with ComputeMpoRelativeTarget(mu, legal_actions, candidate_utilities, 0.50, 0.05)
    std::vector<double> target_dist = ComputeMpoRelativeTarget(
        raw_policy, legal_actions, candidate_utilities, /*base_temperature=*/0.50, /*max_target_kl=*/0.05);
    if (target_dist.size() != legal_actions.size()) continue;

    // Recompute override states from candidate_utilities with the 0.15 rule (best candidate beats raw by >= 0.15)
    double raw_u = obj.at("raw_mean_utility").GetDouble();
    Action raw_a = static_cast<Action>(obj.at("raw_action").GetInt());
    Action best_cand_action = raw_a;
    double best_cand_utility = -1e9;
    for (const auto& cu : candidate_utilities) {
      if (cu.second > best_cand_utility) {
        best_cand_utility = cu.second;
        best_cand_action = cu.first;
      }
    }
    bool is_override = (best_cand_action != raw_a) && (best_cand_utility >= raw_u + 0.15);

    // Read observation from disk
    obs_file.seekg(obs_row * obs_byte_stride, std::ios::beg);
    obs_file.read(reinterpret_cast<char*>(obs_buffer.data()), obs_byte_stride);
    if (!obs_file) {
      open_spiel::SpielFatalError(absl::StrFormat("Failed to read obs row %d from %s", obs_row, obs_bin_path));
    }

    // Replay state from action history
    std::unique_ptr<open_spiel::State> state = game->NewInitialState();
    if (obj.find("action_history") != obj.end() && obj.at("action_history").IsArray()) {
      for (const auto& a_val : obj.at("action_history").GetArray()) {
        state->ApplyAction(static_cast<Action>(a_val.GetInt()));
      }
    }
    if (state->CurrentPlayer() != search_seat) {
      open_spiel::SpielFatalError(absl::StrFormat(
          "Trajectory replay seat mismatch at row %lld: state->CurrentPlayer()=%d, search_seat=%d",
          static_cast<long long>(obs_row), state->CurrentPlayer(), search_seat));
    }

    // Verify observation consistency between disk and replayed state
    std::vector<float> recomputed_obs = evaluator->GetConsumedObservation(*state, search_seat);
    float obs_diff = 0.0f;
    for (size_t k = 0; k < obs_elem_count; ++k) {
      obs_diff = std::max(obs_diff, std::abs(recomputed_obs[k] - obs_buffer[k]));
    }
    if (obs_diff > 1e-4f) {
      open_spiel::SpielFatalError(absl::StrFormat(
          "Observation mismatch at row %lld: max diff = %f",
          static_cast<long long>(obs_row), obs_diff));
    }

    // Query policy via model and apply temperature to capped logits
    dune_semantic::CandidateActionData cand_data;
    const auto* dune = dynamic_cast<const dune_imperium::DuneImperiumState*>(state.get());
    if (dune != nullptr) {
      dune_semantic::ExtractCandidateDescriptors(*dune, legal_actions, &cand_data, dune_semantic::kDescriptorSchemaVersionV3);
    }
    open_spiel::EvalResult full = coord->EvaluateWithActions(recomputed_obs, &cand_data);
    std::vector<float> capped_logits = full.logits;
    open_spiel::CenterAndCapLegalLogitsWithStats(capped_logits, legal_actions, 10.0f);

    double temp = absl::GetFlag(FLAGS_temperature);
    if (temp <= 0.0) temp = 1.0;

    double max_l = -1e9;
    for (Action a : legal_actions) {
      double l = static_cast<double>(capped_logits[a]) / temp;
      if (l > max_l) max_l = l;
    }
    double sum_exp = 0.0;
    for (Action a : legal_actions) {
      sum_exp += std::exp(static_cast<double>(capped_logits[a]) / temp - max_l);
    }
    std::unordered_map<Action, double> model_probs;
    Action model_argmax = -1;
    double max_p = -1.0;
    double policy_entropy = 0.0;
    for (Action a : legal_actions) {
      double p = std::exp(static_cast<double>(capped_logits[a]) / temp - max_l) / sum_exp;
      model_probs[a] = p;
      if (p > max_p) {
        max_p = p;
        model_argmax = a;
      }
      if (p > 1e-15) {
        policy_entropy -= p * std::log(p);
      }
    }

    if (absl::GetFlag(FLAGS_dump_per_state)) {
      open_spiel::json::Object dec_obj;
      dec_obj["obs_row"] = open_spiel::json::Value(static_cast<int64_t>(obs_row));
      dec_obj["is_override"] = open_spiel::json::Value(is_override);
      dec_obj["entropy"] = open_spiel::json::Value(policy_entropy);
      dec_obj["model_argmax"] = open_spiel::json::Value(static_cast<int64_t>(model_argmax));
      dec_obj["argmax_mu"] = open_spiel::json::Value(static_cast<int64_t>(argmax_mu));
      dec_obj["raw_action"] = open_spiel::json::Value(static_cast<int64_t>(raw_a));
      dec_obj["best_cand_action"] = open_spiel::json::Value(static_cast<int64_t>(best_cand_action));

      open_spiel::json::Array cand_arr;
      open_spiel::json::Array cand_probs_arr;
      open_spiel::json::Array cand_logits_arr;
      for (const auto& cu : candidate_utilities) {
        Action ca = cu.first;
        cand_arr.push_back(open_spiel::json::Value(static_cast<int64_t>(ca)));
        cand_probs_arr.push_back(open_spiel::json::Value(model_probs[ca]));
        cand_logits_arr.push_back(open_spiel::json::Value(static_cast<double>(capped_logits[ca])));
      }
      dec_obj["candidate_actions"] = open_spiel::json::Value(cand_arr);
      dec_obj["candidate_probs"] = open_spiel::json::Value(cand_probs_arr);
      dec_obj["candidate_capped_logits"] = open_spiel::json::Value(cand_logits_arr);
      dumped_decisions.push_back(open_spiel::json::Value(dec_obj));
    }

    // max |p - mu| over legal actions
    double max_p_diff_mu = 0.0;
    for (Action a : legal_actions) {
      double p_val = model_probs.count(a) ? model_probs[a] : 0.0;
      double mu_val = mu_map.count(a) ? mu_map[a] : 0.0;
      max_p_diff_mu = std::max(max_p_diff_mu, std::abs(p_val - mu_val));
    }

    // Value of changed choices:
    // where the student's greedy move differs from argmax(mu) and both are among recorded candidates,
    // utility difference = cand_util_map[model_argmax] - cand_util_map[argmax_mu].
    bool val_covered = false;
    double val_diff = 0.0;
    if (model_argmax != argmax_mu &&
        cand_util_map.count(model_argmax) &&
        cand_util_map.count(argmax_mu)) {
      val_covered = true;
      val_diff = cand_util_map[model_argmax] - cand_util_map[argmax_mu];
    }

    // Compute cross-entropy against MPO relative target
    double ce = 0.0;
    Action target_argmax = -1;
    double max_q = -1.0;
    double max_movement = 0.0;
    for (size_t i = 0; i < legal_actions.size(); ++i) {
      Action a = legal_actions[i];
      double q = target_dist[i];
      if (q > max_q) {
        max_q = q;
        target_argmax = a;
      }
      double p = model_probs.count(a) ? model_probs[a] : 0.0;
      ce -= q * std::log(std::max(p, 1e-12));
      max_movement = std::max(max_movement, std::abs(p - q));
    }

    total_ce_all += ce;
    if (model_argmax == target_argmax) {
      ++top1_matches;
    }
    total_max_prob_diff += max_movement;
    round_ce_sum[round] += ce;
    round_dec_count[round]++;
    ++fresh_search_decisions;

    SubgroupMetrics& g = is_override ? ovr_stats : non_ovr_stats;
    g.count++;
    if (model_argmax != argmax_mu) {
      g.greedy_changed_count++;
    }
    g.sum_max_p_diff_mu += max_p_diff_mu;
    g.sum_policy_entropy += policy_entropy;
    if (val_covered) {
      g.value_covered_count++;
      g.sum_value_diff += val_diff;
    }
    g.sum_ce += ce;

    if (is_override) {
      if (model_argmax == best_cand_action) {
        g.greedy_agree_search_count++;
      }
      double teacher_prob = model_probs.count(best_cand_action) ? model_probs[best_cand_action] : 0.0;
      g.sum_prob_search_move += teacher_prob;
    }
  }

  if (fresh_search_decisions == 0) {
    std::cerr << "WARNING: No fresh_search decisions found in " << trajectory_jsonl_path << "\n";
    return 1;
  }

  double mean_ce_all = total_ce_all / fresh_search_decisions;
  double top1_acc = 100.0 * static_cast<double>(top1_matches) / fresh_search_decisions;
  double mean_max_diff = total_max_prob_diff / fresh_search_decisions;

  // Override metrics
  double ovr_greedy_change_rate = (ovr_stats.count > 0) ? (100.0 * ovr_stats.greedy_changed_count / ovr_stats.count) : 0.0;
  double ovr_greedy_agree_search = (ovr_stats.count > 0) ? (100.0 * ovr_stats.greedy_agree_search_count / ovr_stats.count) : 0.0;
  double ovr_mean_prob_search = (ovr_stats.count > 0) ? (ovr_stats.sum_prob_search_move / ovr_stats.count) : 0.0;
  double ovr_mean_max_p_diff = (ovr_stats.count > 0) ? (ovr_stats.sum_max_p_diff_mu / ovr_stats.count) : 0.0;
  double ovr_mean_entropy = (ovr_stats.count > 0) ? (ovr_stats.sum_policy_entropy / ovr_stats.count) : 0.0;
  double ovr_val_changed_diff = (ovr_stats.value_covered_count > 0) ? (ovr_stats.sum_value_diff / ovr_stats.value_covered_count) : 0.0;
  double mean_ce_override = (ovr_stats.count > 0) ? (ovr_stats.sum_ce / ovr_stats.count) : 0.0;
  double override_pct = 100.0 * static_cast<double>(ovr_stats.count) / fresh_search_decisions;

  // Non-override metrics
  double non_ovr_greedy_change_rate = (non_ovr_stats.count > 0) ? (100.0 * non_ovr_stats.greedy_changed_count / non_ovr_stats.count) : 0.0;
  double non_ovr_mean_max_p_diff = (non_ovr_stats.count > 0) ? (non_ovr_stats.sum_max_p_diff_mu / non_ovr_stats.count) : 0.0;
  double non_ovr_mean_entropy = (non_ovr_stats.count > 0) ? (non_ovr_stats.sum_policy_entropy / non_ovr_stats.count) : 0.0;
  double non_ovr_val_changed_diff = (non_ovr_stats.value_covered_count > 0) ? (non_ovr_stats.sum_value_diff / non_ovr_stats.value_covered_count) : 0.0;
  double mean_ce_non_override = (non_ovr_stats.count > 0) ? (non_ovr_stats.sum_ce / non_ovr_stats.count) : 0.0;

  std::cout << "================================================================================\n";
  std::cout << "HELD-OUT EVALUATION AGAINST RECORDED U28700 PRIOR (mu)\n";
  std::cout << "================================================================================\n";
  std::cout << absl::StrFormat("Total fresh_search decisions evaluated: %lld\n\n",
                               static_cast<long long>(fresh_search_decisions));

  std::cout << absl::StrFormat("--- OVERRIDE STATES (N = %lld, %.2f%%) ---\n",
                               static_cast<long long>(ovr_stats.count), override_pct);
  std::cout << absl::StrFormat("  Greedy change rate [argmax(p) != argmax(mu)]:       %6.2f%% (%lld / %lld)\n",
                               ovr_greedy_change_rate, static_cast<long long>(ovr_stats.greedy_changed_count), static_cast<long long>(ovr_stats.count));
  std::cout << absl::StrFormat("  Greedy agreement with search move:                  %6.2f%% (%lld / %lld)\n",
                               ovr_greedy_agree_search, static_cast<long long>(ovr_stats.greedy_agree_search_count), static_cast<long long>(ovr_stats.count));
  std::cout << absl::StrFormat("  Mean probability on search move:                    %6.4f (U28700 baseline: 0.0887)\n",
                               ovr_mean_prob_search);
  std::cout << absl::StrFormat("  Mean max |p - mu|:                                  %6.4f\n",
                               ovr_mean_max_p_diff);
  std::cout << absl::StrFormat("  Mean policy entropy:                                %6.4f\n",
                               ovr_mean_entropy);
  std::cout << absl::StrFormat("  Value of changed choices (mean utility diff):       %+6.4f ladder units (N=%lld covered)\n",
                               ovr_val_changed_diff, static_cast<long long>(ovr_stats.value_covered_count));
  std::cout << absl::StrFormat("  Mean CE (MPO target):                               %6.4f\n\n",
                               mean_ce_override);

  std::cout << absl::StrFormat("--- NON-OVERRIDE STATES (N = %lld, %.2f%%) ---\n",
                               static_cast<long long>(non_ovr_stats.count), 100.0 - override_pct);
  std::cout << absl::StrFormat("  Greedy change rate [argmax(p) != argmax(mu)]:       %6.2f%% (%lld / %lld)\n",
                               non_ovr_greedy_change_rate, static_cast<long long>(non_ovr_stats.greedy_changed_count), static_cast<long long>(non_ovr_stats.count));
  std::cout << absl::StrFormat("  Mean max |p - mu|:                                  %6.4f\n",
                               non_ovr_mean_max_p_diff);
  std::cout << absl::StrFormat("  Mean policy entropy:                                %6.4f\n",
                               non_ovr_mean_entropy);
  std::cout << absl::StrFormat("  Value of changed choices (mean utility diff):       %+6.4f ladder units (N=%lld covered)\n",
                               non_ovr_val_changed_diff, static_cast<long long>(non_ovr_stats.value_covered_count));
  std::cout << absl::StrFormat("  Mean CE (MPO target):                               %6.4f\n",
                               mean_ce_non_override);
  std::cout << "--------------------------------------------------------------------------------\n";
  std::cout << absl::StrFormat("  All States Mean CE:                                 %6.4f\n", mean_ce_all);
  std::cout << absl::StrFormat("  All States MPO Target Top-1 Match:                  %6.2f%%\n", top1_acc);
  std::cout << absl::StrFormat("  All States Mean Max Prob Movement (vs MPO target):  %6.4f\n", mean_max_diff);
  std::cout << "--------------------------------------------------------------------------------\n";
  std::cout << "Per-Round Cross-Entropy (All States):\n";
  for (int r = 1; r <= 10; ++r) {
    if (round_dec_count.count(r) && round_dec_count[r] > 0) {
      std::cout << absl::StrFormat("  Round %2d: CE = %.4f (%lld decisions)\n",
                                   r, round_ce_sum[r] / round_dec_count[r],
                                   static_cast<long long>(round_dec_count[r]));
    }
  }
  std::cout << "================================================================================\n";

  if (!output_json_path.empty()) {
    open_spiel::json::Object out_obj;
    out_obj["model_checkpoint"] = open_spiel::json::Value(model_path);
    out_obj["trajectory_meta_jsonl"] = open_spiel::json::Value(trajectory_jsonl_path);
    out_obj["trajectory_obs_bin"] = open_spiel::json::Value(obs_bin_path);
    out_obj["fresh_search_decisions"] = open_spiel::json::Value(fresh_search_decisions);

    // Override subgroup
    open_spiel::json::Object ovr_json;
    ovr_json["count"] = open_spiel::json::Value(ovr_stats.count);
    ovr_json["override_pct"] = open_spiel::json::Value(override_pct);
    ovr_json["greedy_change_rate_pct"] = open_spiel::json::Value(ovr_greedy_change_rate);
    ovr_json["greedy_agree_search_pct"] = open_spiel::json::Value(ovr_greedy_agree_search);
    ovr_json["mean_prob_search_move"] = open_spiel::json::Value(ovr_mean_prob_search);
    ovr_json["mean_max_p_diff_mu"] = open_spiel::json::Value(ovr_mean_max_p_diff);
    ovr_json["mean_policy_entropy"] = open_spiel::json::Value(ovr_mean_entropy);
    ovr_json["value_changed_utility_diff"] = open_spiel::json::Value(ovr_val_changed_diff);
    ovr_json["value_changed_count_covered"] = open_spiel::json::Value(ovr_stats.value_covered_count);
    ovr_json["mean_cross_entropy"] = open_spiel::json::Value(mean_ce_override);
    out_obj["override_states"] = open_spiel::json::Value(ovr_json);

    // Non-override subgroup
    open_spiel::json::Object non_ovr_json;
    non_ovr_json["count"] = open_spiel::json::Value(non_ovr_stats.count);
    non_ovr_json["pct"] = open_spiel::json::Value(100.0 - override_pct);
    non_ovr_json["greedy_change_rate_pct"] = open_spiel::json::Value(non_ovr_greedy_change_rate);
    non_ovr_json["mean_max_p_diff_mu"] = open_spiel::json::Value(non_ovr_mean_max_p_diff);
    non_ovr_json["mean_policy_entropy"] = open_spiel::json::Value(non_ovr_mean_entropy);
    non_ovr_json["value_changed_utility_diff"] = open_spiel::json::Value(non_ovr_val_changed_diff);
    non_ovr_json["value_changed_count_covered"] = open_spiel::json::Value(non_ovr_stats.value_covered_count);
    non_ovr_json["mean_cross_entropy"] = open_spiel::json::Value(mean_ce_non_override);
    out_obj["non_override_states"] = open_spiel::json::Value(non_ovr_json);

    out_obj["temperature"] = open_spiel::json::Value(absl::GetFlag(FLAGS_temperature));
    out_obj["mean_policy_entropy_non_override"] = open_spiel::json::Value(non_ovr_mean_entropy);
    out_obj["mean_policy_entropy_override"] = open_spiel::json::Value(ovr_mean_entropy);
    if (absl::GetFlag(FLAGS_dump_per_state)) {
      out_obj["decisions"] = open_spiel::json::Value(dumped_decisions);
    }

    // Legacy fields for backward compatibility
    out_obj["override_decisions"] = open_spiel::json::Value(ovr_stats.count);
    out_obj["override_pct"] = open_spiel::json::Value(override_pct);
    out_obj["mean_teacher_prob_override"] = open_spiel::json::Value(ovr_mean_prob_search);
    out_obj["mean_cross_entropy_all"] = open_spiel::json::Value(mean_ce_all);
    out_obj["mean_cross_entropy_override"] = open_spiel::json::Value(mean_ce_override);
    out_obj["top1_accuracy_pct"] = open_spiel::json::Value(top1_acc);
    out_obj["mean_max_prob_diff"] = open_spiel::json::Value(mean_max_diff);

    open_spiel::json::Object rounds_obj;
    for (const auto& kv : round_dec_count) {
      open_spiel::json::Object r_info;
      r_info["count"] = open_spiel::json::Value(kv.second);
      r_info["mean_ce"] = open_spiel::json::Value(round_ce_sum[kv.first] / kv.second);
      rounds_obj[std::to_string(kv.first)] = open_spiel::json::Value(r_info);
    }
    out_obj["round_breakdown"] = open_spiel::json::Value(rounds_obj);

    std::ofstream out_file(output_json_path);
    if (out_file.is_open()) {
      out_file << open_spiel::json::ToString(open_spiel::json::Value(out_obj)) << "\n";
      out_file.close();
      std::cout << "[OUTPUT] Saved receipt to: " << output_json_path << "\n";
    }
  }

  return 0;
}

namespace {
template <typename T>
void WritePod(std::ofstream& ofs, const T& val) {
  ofs.write(reinterpret_cast<const char*>(&val), sizeof(T));
}

template <typename T>
void ReadPod(std::ifstream& ifs, T& val) {
  ifs.read(reinterpret_cast<char*>(&val), sizeof(T));
}

template <typename T>
void WriteVector(std::ofstream& ofs, const std::vector<T>& vec) {
  uint64_t sz = vec.size();
  WritePod(ofs, sz);
  if (sz > 0) {
    ofs.write(reinterpret_cast<const char*>(vec.data()), sz * sizeof(T));
  }
}

template <typename T>
void ReadVector(std::ifstream& ifs, std::vector<T>& vec) {
  uint64_t sz = 0;
  ReadPod(ifs, sz);
  vec.resize(sz);
  if (sz > 0) {
    ifs.read(reinterpret_cast<char*>(vec.data()), sz * sizeof(T));
  }
}
}  // namespace

int RunSelfPlayRootAndPreservationCapture() {
  const int num_games = absl::GetFlag(FLAGS_capture_self_play_games);
  const int roots_per_seat = absl::GetFlag(FLAGS_capture_roots_per_seat_game);
  const int pres_per_seat = absl::GetFlag(FLAGS_capture_pres_per_seat_game);
  const int max_placement_roots = absl::GetFlag(FLAGS_capture_max_placement_roots);
  const std::string out_dir = absl::GetFlag(FLAGS_capture_out_dir);
  const std::string model_path = absl::GetFlag(FLAGS_model_checkpoint);
  const int num_threads = std::max(1, absl::GetFlag(FLAGS_threads));
  const int num_evals = absl::GetFlag(FLAGS_num_evaluators);
  const int target_batch_size = absl::GetFlag(FLAGS_target_batch_size);
  const uint64_t base_seed = absl::GetFlag(FLAGS_master_seed);
  const int num_blocks = absl::GetFlag(FLAGS_num_blocks);

  if (out_dir.empty()) {
    open_spiel::SpielFatalError("--capture_out_dir must be specified when --capture_self_play_games > 0");
  }
  if (model_path.empty()) {
    open_spiel::SpielFatalError("--model_checkpoint must be specified when --capture_self_play_games > 0");
  }

  std::cout << "================================================================================\n";
  std::cout << "MODE 1: RAW GREEDY SELF-PLAY ROOT & PRESERVATION CAPTURE\n";
  std::cout << "================================================================================\n";
  std::cout << "Model checkpoint:   " << model_path << "\n";
  std::cout << "Output dir:         " << out_dir << "\n";
  std::cout << "Games:              " << num_games << "\n";
  std::cout << "Threads:            " << num_threads << "\n";
  std::cout << "Roots / seat-game:  " << roots_per_seat << "\n";
  std::cout << "Pres / seat-game:   " << pres_per_seat << "\n";
  std::cout << "Max placement roots:" << max_placement_roots << "\n";
  std::cout << "Master seed:        " << base_seed << "\n";
  std::cout << "Num blocks:         " << num_blocks << "\n";

  open_spiel::dune_imperium::DuneImperiumState::SetAutoPassEmptyCombatIntrigueHands(
      absl::GetFlag(FLAGS_auto_pass_empty_combat_intrigue));

  at::globalContext().setAllowTF32CuBLAS(false);
  at::globalContext().setAllowTF32CuDNN(false);

  torch::Device device(torch::kCUDA, 0);
  if (!torch::cuda::is_available()) {
    device = torch::Device(torch::kCPU);
  }
  std::cout << "[DEVICE] Using device: " << (device.is_cuda() ? "CUDA" : "CPU") << "\n";

  std::vector<std::shared_ptr<open_spiel::SharedDunePolicyValueNetImpl>> models(num_evals);
  std::vector<std::unique_ptr<std::shared_mutex>> model_mutexes(num_evals);
  std::vector<std::shared_ptr<open_spiel::BatchedEvaluator>> coords(num_evals);
  std::vector<std::shared_ptr<open_spiel::BatchedNNEvaluator>> evaluators(num_evals);

  for (int e = 0; e < num_evals; ++e) {
    models[e] = std::make_shared<open_spiel::SharedDunePolicyValueNetImpl>(
        open_spiel::dune_imperium::kFullPublicInformationStateSize, 2048, 2391,
        num_blocks,
        /*nonlinear_value_head=*/false, /*with_aux_heads=*/false,
        /*head_init_seed=*/0, /*with_semantic_scorer=*/true);
    models[e]->to(device);
    open_spiel::LoadModelCheckpointRobust(models[e], model_path, device);
    models[e]->eval();

    model_mutexes[e] = std::make_unique<std::shared_mutex>();
    coords[e] = std::make_shared<open_spiel::BatchedEvaluator>(
        models[e], target_batch_size, /*timeout_ms=*/1, device, model_mutexes[e].get(), 10.0f,
        /*device_synchronize=*/false, /*high_priority_stream=*/true,
        /*emit_batch_membership=*/false,
        /*rollout_amp=*/false,
        /*allow_tf32=*/false);
    evaluators[e] = std::make_shared<open_spiel::BatchedNNEvaluator>(coords[e], 10.0f);
  }

  std::shared_ptr<const open_spiel::Game> game = open_spiel::LoadGame("dune_imperium");

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

  std::vector<std::vector<CapturedPlacementRoot>> per_game_roots(num_games);
  std::vector<std::vector<CapturedPreservationItem>> per_game_pres(num_games);

  std::filesystem::create_directories(out_dir);
  std::string ckpt_path = out_dir + "/preservation_pool.bin.ckpt";

  std::unique_ptr<std::atomic<uint8_t>[]> game_completed(new std::atomic<uint8_t>[num_games]());
  for (int i = 0; i < num_games; ++i) {
    game_completed[i].store(0, std::memory_order_relaxed);
  }
  std::atomic<int> completed_games_count{0};
  std::mutex ckpt_mutex;

  if (roots_per_seat == 0 && std::filesystem::exists(ckpt_path)) {
    std::ifstream ifs(ckpt_path, std::ios::binary);
    if (ifs.is_open()) {
      uint32_t magic = 0;
      uint32_t version = 0;
      ReadPod(ifs, magic);
      ReadPod(ifs, version);
      if (magic == 0x50434B50U && version == 1U) {
        uint64_t saved_completed_games = 0;
        ReadPod(ifs, saved_completed_games);
        for (uint64_t i = 0; i < saved_completed_games; ++i) {
          int g = -1;
          ReadPod(ifs, g);
          uint64_t num_items = 0;
          ReadPod(ifs, num_items);
          if (g >= 0 && g < num_games) {
            per_game_pres[g].resize(num_items);
            for (uint64_t item_idx = 0; item_idx < num_items; ++item_idx) {
              auto& item = per_game_pres[g][item_idx];
              ReadPod(ifs, item.game_id);
              ReadPod(ifs, item.split_id);
              ReadPod(ifs, item.player);
              ReadVector(ifs, item.obs);
              ReadVector(ifs, item.cand_data.actions);
              ReadVector(ifs, item.cand_data.features);
              ReadVector(ifs, item.cand_data.card_ids);
              ReadVector(ifs, item.cand_data.space_ids);
              ReadVector(ifs, item.cand_data.supported);
              ReadVector(ifs, item.cand_data.roles);
              ReadVector(ifs, item.legal_actions);
              ReadVector(ifs, item.pi_parent);
            }
            if (game_completed[g].load(std::memory_order_relaxed) == 0) {
              game_completed[g].store(1, std::memory_order_release);
              completed_games_count.fetch_add(1, std::memory_order_relaxed);
            }
          } else {
            for (uint64_t item_idx = 0; item_idx < num_items; ++item_idx) {
              CapturedPreservationItem dummy;
              ReadPod(ifs, dummy.game_id);
              ReadPod(ifs, dummy.split_id);
              ReadPod(ifs, dummy.player);
              ReadVector(ifs, dummy.obs);
              ReadVector(ifs, dummy.cand_data.actions);
              ReadVector(ifs, dummy.cand_data.features);
              ReadVector(ifs, dummy.cand_data.card_ids);
              ReadVector(ifs, dummy.cand_data.space_ids);
              ReadVector(ifs, dummy.cand_data.supported);
              ReadVector(ifs, dummy.cand_data.roles);
              ReadVector(ifs, dummy.legal_actions);
              ReadVector(ifs, dummy.pi_parent);
            }
          }
        }
        std::cout << absl::StrFormat("[RESUME] Loaded %d completed games from %s\n",
                                     completed_games_count.load(), ckpt_path);
      } else {
        std::cerr << "[RESUME] Warning: Checkpoint magic/version mismatch in " << ckpt_path << ", ignoring.\n";
      }
    }
  }

  if (completed_games_count.load() == num_games) {
    std::cout << absl::StrFormat("[CAPTURE] All %d games already completed from checkpoint\n", num_games);
  }

  std::atomic<int> next_game{0};
  std::vector<std::thread> workers;
  workers.reserve(num_threads);

  auto start_time = std::chrono::steady_clock::now();

  for (int t = 0; t < num_threads; ++t) {
    workers.emplace_back([&, t]() {
      int eval_idx = t % num_evals;
      while (true) {
        int g = next_game.fetch_add(1);
        if (g >= num_games) break;
        if (game_completed[g].load(std::memory_order_acquire) != 0) continue;

        int mod20 = g % 20;
        std::string split = (mod20 < 14) ? "train" : (mod20 < 17 ? "val" : "test");
        int split_id = (mod20 < 14) ? 0 : (mod20 < 17 ? 1 : 2);

        uint64_t chance_seed = dune_seed::DeriveSeed(base_seed, 0x4501ULL, static_cast<uint64_t>(g));
        std::mt19937 rng(chance_seed);
        std::unique_ptr<State> state = game->NewInitialState();
        std::vector<int64_t> history;

        std::array<std::vector<CapturedPlacementRoot>, 4> seat_placements;
        std::array<std::vector<CapturedPreservationItem>, 4> seat_all_decisions;
        int decision_ordinal = 0;

        while (!state->IsTerminal()) {
          if (state->IsChanceNode()) {
            auto outcomes = state->ChanceOutcomes();
            double u = std::generate_canonical<double, 53>(rng);
            Action ca = open_spiel::SampleAction(outcomes, u).first;
            state->ApplyAction(ca);
            history.push_back(ca);
            continue;
          }

          Player cur = state->CurrentPlayer();
          std::vector<Action> legals = state->LegalActions();
          if (legals.empty()) break;

          if (legals.size() == 1) {
            Action a = legals.front();
            state->ApplyAction(a);
            history.push_back(a);
            continue;
          }

          const auto* dune = dynamic_cast<const dune_imperium::DuneImperiumState*>(state.get());
          GamePhase phase = dune ? dune->phase() : GamePhase::kAgentTurns;

          std::string dec_type;
          bool has_tech = false, has_tleilaxu = false, has_imperium = false;
          for (Action a : legals) {
            if ((70 <= a && a <= 75) || (770 <= a && a <= 775)) has_tech = true;
            if (90 <= a && a <= 95) has_tleilaxu = true;
            if ((10 <= a && a < 18) || (572 <= a && a < 580) || a == 598 || a == 599) has_imperium = true;
          }
          if (has_tech || has_tleilaxu || has_imperium) {
            if (has_tech) dec_type = "Purchase:Tech";
            else if (has_tleilaxu) dec_type = "Purchase:Tleilaxu";
            else if (has_imperium) dec_type = "Purchase:Imperium";
            else dec_type = "Purchase:Other";
          } else {
            bool has_leader = false;
            for (Action a : legals) {
              if (558 <= a && a <= 571) { has_leader = true; break; }
            }
            if (has_leader) {
              dec_type = "LeaderSelect";
            } else {
              bool has_combat_act = false;
              for (Action a : legals) {
                if (a == 40 || (1700 <= a && a < 1800) || (650 <= a && a < 698)) {
                  has_combat_act = true;
                  break;
                }
              }
              if (phase == GamePhase::kCombat || has_combat_act) {
                dec_type = "CombatIntrigue";
              } else if (phase == GamePhase::kAgentTurns) {
                bool has_card = false;
                bool has_space = false;
                for (Action a : legals) {
                  if ((1800 <= a && a < 2056) || a == 3) has_card = true;
                  if (600 <= a && a <= 621) has_space = true;
                }
                if (has_card && !has_space) dec_type = "AgentPrimary";
                else if (has_space) dec_type = "AgentContinuation";
                else dec_type = "Other";
              } else {
                dec_type = "Other";
              }
            }
          }

          auto prior = evaluators[eval_idx]->Prior(*state);
          Action raw_action = legals.front();
          double max_p = -1.0;
          std::vector<double> pi_p(legals.size(), 0.0);
          for (size_t i = 0; i < legals.size(); ++i) {
            Action a = legals[i];
            double p = 0.0;
            for (const auto& ap : prior) {
              if (ap.first == a) { p = ap.second; break; }
            }
            pi_p[i] = p;
            if (p > max_p) {
              max_p = p;
              raw_action = a;
            }
          }

          std::vector<float> obs = evaluators[eval_idx]->GetConsumedObservation(*state, cur);
          dune_semantic::CandidateActionData cand_data;
          if (dune) {
            dune_semantic::ExtractCandidateDescriptors(*dune, legals, &cand_data, dune_semantic::kDescriptorSchemaVersionV3);
          }
          int cur_round = dune ? dune->GetCurrentRound() : 1;
          std::string round_band = (cur_round <= 3) ? "R1-3" : (cur_round <= 6 ? "R4-6" : "R7+");

          CapturedPreservationItem pres_item;
          pres_item.game_id = g;
          pres_item.split_id = split_id;
          pres_item.player = cur;
          pres_item.obs = obs;
          pres_item.cand_data = cand_data;
          pres_item.legal_actions.assign(legals.begin(), legals.end());
          pres_item.pi_parent = pi_p;
          seat_all_decisions[cur].push_back(std::move(pres_item));

          if (roots_per_seat > 0 && (dec_type == "AgentPrimary" || dec_type == "AgentContinuation")) {
            CapturedPlacementRoot pl_root;
            pl_root.game_id = g;
            pl_root.decision_ordinal = decision_ordinal;
            pl_root.search_seat = cur;
            pl_root.round = cur_round;
            pl_root.round_band = round_band;
            pl_root.decision_type = dec_type;
            pl_root.legal_actions.assign(legals.begin(), legals.end());
            pl_root.raw_action = raw_action;
            pl_root.action_history = history;
            pl_root.split = split;
            pl_root.pi_parent = std::move(pi_p);
            pl_root.obs = std::move(obs);
            pl_root.cand_data = std::move(cand_data);
            seat_placements[cur].push_back(std::move(pl_root));
          }

          decision_ordinal++;
          state->ApplyAction(raw_action);
          history.push_back(raw_action);
        }

        for (int s = 0; s < 4; ++s) {
          auto& pl_list = seat_placements[s];
          if (!pl_list.empty()) {
            std::mt19937 seat_rng(dune_seed::DeriveSeed(base_seed, 0x4502ULL, static_cast<uint64_t>(g * 4 + s)));
            std::vector<size_t> p_idx(pl_list.size());
            std::iota(p_idx.begin(), p_idx.end(), 0);
            std::shuffle(p_idx.begin(), p_idx.end(), seat_rng);
            size_t k_take = std::min<size_t>(roots_per_seat, p_idx.size());
            p_idx.resize(k_take);
            std::sort(p_idx.begin(), p_idx.end());
            for (size_t idx : p_idx) {
              per_game_roots[g].push_back(std::move(pl_list[idx]));
            }
          }

          auto& all_list = seat_all_decisions[s];
          if (!all_list.empty()) {
            std::mt19937 pres_rng(dune_seed::DeriveSeed(base_seed, 0x4503ULL, static_cast<uint64_t>(g * 4 + s)));
            std::vector<size_t> a_idx(all_list.size());
            std::iota(a_idx.begin(), a_idx.end(), 0);
            std::shuffle(a_idx.begin(), a_idx.end(), pres_rng);
            size_t k_take = std::min<size_t>(pres_per_seat, a_idx.size());
            a_idx.resize(k_take);
            std::sort(a_idx.begin(), a_idx.end());
            for (size_t idx : a_idx) {
              per_game_pres[g].push_back(std::move(all_list[idx]));
            }
          }
        }

        game_completed[g].store(1, std::memory_order_release);
        int done = ++completed_games_count;

        if (done % 10 == 0 || done == num_games) {
          std::cout << absl::StrFormat("[CAPTURE] Completed %d / %d self-play games\n", done, num_games);
        }

        if (roots_per_seat == 0 && done % 200 == 0) {
          std::lock_guard<std::mutex> lock(ckpt_mutex);
          std::string tmp_ckpt = out_dir + "/preservation_pool.bin.ckpt.tmp";
          std::string ckpt_file = out_dir + "/preservation_pool.bin.ckpt";
          std::ofstream ofs(tmp_ckpt, std::ios::binary);
          if (ofs.is_open()) {
            WritePod(ofs, 0x50434B50U);
            WritePod(ofs, 1U);
            uint64_t completed_count = 0;
            for (int i = 0; i < num_games; ++i) {
              if (game_completed[i].load(std::memory_order_acquire) != 0) {
                completed_count++;
              }
            }
            WritePod(ofs, completed_count);
            for (int i = 0; i < num_games; ++i) {
              if (game_completed[i].load(std::memory_order_acquire) != 0) {
                WritePod(ofs, i);
                uint64_t n_items = per_game_pres[i].size();
                WritePod(ofs, n_items);
                for (const auto& item : per_game_pres[i]) {
                  WritePod(ofs, item.game_id);
                  WritePod(ofs, item.split_id);
                  WritePod(ofs, item.player);
                  WriteVector(ofs, item.obs);
                  WriteVector(ofs, item.cand_data.actions);
                  WriteVector(ofs, item.cand_data.features);
                  WriteVector(ofs, item.cand_data.card_ids);
                  WriteVector(ofs, item.cand_data.space_ids);
                  WriteVector(ofs, item.cand_data.supported);
                  WriteVector(ofs, item.cand_data.roles);
                  WriteVector(ofs, item.legal_actions);
                  WriteVector(ofs, item.pi_parent);
                }
              }
            }
            ofs.close();
            std::error_code ec;
            std::filesystem::rename(tmp_ckpt, ckpt_file, ec);
            if (!ec) {
              std::cout << absl::StrFormat("[CAPTURE] Checkpointed %zu completed games to %s\n",
                                           completed_count, ckpt_file);
            }
          }
        }
      }
    });
  }

  for (auto& w : workers) {
    w.join();
  }

  std::vector<CapturedPlacementRoot> all_sampled_roots;
  for (int g = 0; g < num_games; ++g) {
    for (auto& r : per_game_roots[g]) {
      all_sampled_roots.push_back(std::move(r));
    }
  }

  if (max_placement_roots > 0 && all_sampled_roots.size() > static_cast<size_t>(max_placement_roots)) {
    std::mt19937 sub_rng(dune_seed::DeriveSeed(base_seed, 0x4504ULL, 0ULL));
    std::vector<size_t> indices(all_sampled_roots.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::shuffle(indices.begin(), indices.end(), sub_rng);
    indices.resize(max_placement_roots);
    std::sort(indices.begin(), indices.end());
    std::vector<CapturedPlacementRoot> trimmed;
    trimmed.reserve(max_placement_roots);
    for (size_t idx : indices) {
      trimmed.push_back(std::move(all_sampled_roots[idx]));
    }
    all_sampled_roots = std::move(trimmed);
  }

  std::vector<CapturedPreservationItem> all_pres;
  for (int g = 0; g < num_games; ++g) {
    for (auto& p : per_game_pres[g]) {
      all_pres.push_back(std::move(p));
    }
  }

  std::filesystem::create_directories(out_dir);

  // 1. Write sampled_roots.json
  open_spiel::json::Object meta;
  meta["game_string"] = open_spiel::json::Value(game->ToString());
  meta["total_games"] = open_spiel::json::Value(static_cast<int64_t>(num_games));
  meta["total_roots"] = open_spiel::json::Value(static_cast<int64_t>(all_sampled_roots.size()));
  meta["master_seed"] = open_spiel::json::Value(static_cast<int64_t>(base_seed));

  open_spiel::json::Array roots_arr;
  roots_arr.reserve(all_sampled_roots.size());
  for (size_t i = 0; i < all_sampled_roots.size(); ++i) {
    const auto& r = all_sampled_roots[i];
    open_spiel::json::Object r_obj;
    r_obj["root_id"] = open_spiel::json::Value(static_cast<int64_t>(i));
    r_obj["game_id"] = open_spiel::json::Value(static_cast<int64_t>(r.game_id));
    r_obj["decision_ordinal"] = open_spiel::json::Value(static_cast<int64_t>(r.decision_ordinal));
    r_obj["obs_row"] = open_spiel::json::Value(static_cast<int64_t>(i));
    r_obj["search_seat"] = open_spiel::json::Value(static_cast<int64_t>(r.search_seat));
    r_obj["round"] = open_spiel::json::Value(static_cast<int64_t>(r.round));
    r_obj["round_band"] = open_spiel::json::Value(r.round_band);
    r_obj["decision_type"] = open_spiel::json::Value(r.decision_type);
    r_obj["stratum_weight"] = open_spiel::json::Value(1.0);
    r_obj["raw_action"] = open_spiel::json::Value(static_cast<int64_t>(r.raw_action));
    r_obj["census_chosen_action"] = open_spiel::json::Value(static_cast<int64_t>(r.raw_action));
    r_obj["census_overridden"] = open_spiel::json::Value(false);

    open_spiel::json::Array la_arr;
    for (int a : r.legal_actions) la_arr.push_back(open_spiel::json::Value(static_cast<int64_t>(a)));
    r_obj["legal_actions"] = open_spiel::json::Value(la_arr);

    open_spiel::json::Array ah_arr;
    for (int64_t a : r.action_history) ah_arr.push_back(open_spiel::json::Value(a));
    r_obj["action_history"] = open_spiel::json::Value(ah_arr);

    r_obj["split"] = open_spiel::json::Value(r.split);

    open_spiel::json::Array pi_arr;
    for (double p : r.pi_parent) pi_arr.push_back(open_spiel::json::Value(p));
    r_obj["pi_parent"] = open_spiel::json::Value(pi_arr);

    roots_arr.push_back(open_spiel::json::Value(r_obj));
  }

  open_spiel::json::Object root_out_json;
  root_out_json["metadata"] = open_spiel::json::Value(meta);
  root_out_json["roots"] = open_spiel::json::Value(roots_arr);

  std::string roots_json_path = out_dir + "/sampled_roots.json";
  {
    std::ofstream ofs(roots_json_path);
    if (!ofs.is_open()) open_spiel::SpielFatalError("Failed to write to " + roots_json_path);
    ofs << open_spiel::json::ToString(open_spiel::json::Value(root_out_json), true) << "\n";
  }
  std::cout << "[SAVED] " << roots_json_path << " (" << all_sampled_roots.size() << " roots)\n";

  // 2. Write sampled_obs.bin
  std::string obs_bin_path = out_dir + "/sampled_obs.bin";
  {
    std::ofstream ofs(obs_bin_path, std::ios::binary);
    if (!ofs.is_open()) open_spiel::SpielFatalError("Failed to write to " + obs_bin_path);
    for (size_t i = 0; i < all_sampled_roots.size(); ++i) {
      ofs.write(reinterpret_cast<const char*>(all_sampled_roots[i].obs.data()),
                all_sampled_roots[i].obs.size() * sizeof(float));
    }
  }
  std::cout << "[SAVED] " << obs_bin_path << "\n";

  // 3. Write sampled_cand.bin
  std::string cand_bin_path = out_dir + "/sampled_cand.bin";
  {
    std::ofstream ofs(cand_bin_path, std::ios::binary);
    if (!ofs.is_open()) open_spiel::SpielFatalError("Failed to write to " + cand_bin_path);
    WritePod(ofs, 0x43414E44U);
    WritePod(ofs, 1U);
    uint64_t nr = all_sampled_roots.size();
    WritePod(ofs, nr);
    for (size_t i = 0; i < all_sampled_roots.size(); ++i) {
      const auto& cd = all_sampled_roots[i].cand_data;
      WriteVector(ofs, cd.actions);
      WriteVector(ofs, cd.features);
      WriteVector(ofs, cd.card_ids);
      WriteVector(ofs, cd.space_ids);
      WriteVector(ofs, cd.supported);
      WriteVector(ofs, cd.roles);
    }
  }
  std::cout << "[SAVED] " << cand_bin_path << "\n";

  // 4. Write preservation_pool.bin
  std::string pres_bin_path = out_dir + "/preservation_pool.bin";
  {
    std::ofstream ofs(pres_bin_path, std::ios::binary);
    if (!ofs.is_open()) open_spiel::SpielFatalError("Failed to write to " + pres_bin_path);
    WritePod(ofs, 0x50524553U);
    WritePod(ofs, 1U);
    uint64_t np = all_pres.size();
    WritePod(ofs, np);
    for (size_t i = 0; i < all_pres.size(); ++i) {
      const auto& item = all_pres[i];
      WritePod(ofs, item.game_id);
      WritePod(ofs, item.split_id);
      WritePod(ofs, item.player);
      WriteVector(ofs, item.obs);
      WriteVector(ofs, item.cand_data.actions);
      WriteVector(ofs, item.cand_data.features);
      WriteVector(ofs, item.cand_data.card_ids);
      WriteVector(ofs, item.cand_data.space_ids);
      WriteVector(ofs, item.cand_data.supported);
      WriteVector(ofs, item.cand_data.roles);
      WriteVector(ofs, item.legal_actions);
      WriteVector(ofs, item.pi_parent);
    }
  }
  std::cout << "[SAVED] " << pres_bin_path << " (" << all_pres.size() << " preservation decisions)\n";

  std::remove((out_dir + "/preservation_pool.bin.ckpt").c_str());
  std::remove((out_dir + "/preservation_pool.bin.ckpt.tmp").c_str());

  auto end_time = std::chrono::steady_clock::now();
  double elapsed_s = std::chrono::duration<double>(end_time - start_time).count();
  std::cout << absl::StrFormat("[CAPTURE] Complete in %.1f s. Sampled %zu placement roots, %zu preservation decisions.\n",
                               elapsed_s, all_sampled_roots.size(), all_pres.size());
  return 0;
}

int RunSuccessiveHalvingDistillLabel() {
  const std::string model_path = absl::GetFlag(FLAGS_model_checkpoint);
  const std::string roots_path = absl::GetFlag(FLAGS_sampled_roots_json);
  const std::string obs_path = absl::GetFlag(FLAGS_obs_binary);
  const std::string output_path = absl::GetFlag(FLAGS_output_json);
  const int parallel_roots = std::max(1, absl::GetFlag(FLAGS_parallel_roots));
  const int num_evals = absl::GetFlag(FLAGS_num_evaluators);
  const int target_batch_size = absl::GetFlag(FLAGS_target_batch_size);
  const uint64_t master_seed = absl::GetFlag(FLAGS_master_seed);
  const uint64_t label_seed_domain = absl::GetFlag(FLAGS_label_seed_domain);
  const bool use_bounded_pool = absl::GetFlag(FLAGS_use_bounded_pool_when_unclocked);
  const int max_inflight = absl::GetFlag(FLAGS_max_inflight_worlds);
  const bool placement_only = absl::GetFlag(FLAGS_placement_roots_only);
  const int num_blocks = absl::GetFlag(FLAGS_num_blocks);

  if (model_path.empty() || roots_path.empty() || obs_path.empty() || output_path.empty()) {
    open_spiel::SpielFatalError("--model_checkpoint, --sampled_roots_json, --obs_binary, and --output_json must be specified.");
  }

  std::cout << "================================================================================\n";
  std::cout << "MODE 2: PARALLEL-ROOTS SH_768_K5_ALLBUY LABELLING MODE\n";
  std::cout << "================================================================================\n";
  std::cout << "Model checkpoint:   " << model_path << "\n";
  std::cout << "Sampled roots JSON: " << roots_path << "\n";
  std::cout << "Observation bin:    " << obs_path << "\n";
  std::cout << "Output JSON:        " << output_path << "\n";
  std::cout << "Parallel roots:     " << parallel_roots << "\n";
  std::cout << "Master seed:        " << master_seed << "\n";
  std::cout << "Label seed domain:  " << label_seed_domain << "\n";
  std::cout << "Placement only:     " << (placement_only ? "true" : "false") << "\n";

  if (absl::GetFlag(FLAGS_sh_budget_check)) {
    open_spiel::dune_imperium::DuneImperiumState::SetAutoPassEmptyCombatIntrigueHands(false);
  } else {
    open_spiel::dune_imperium::DuneImperiumState::SetAutoPassEmptyCombatIntrigueHands(
        absl::GetFlag(FLAGS_auto_pass_empty_combat_intrigue));
  }

  at::globalContext().setAllowTF32CuBLAS(false);
  at::globalContext().setAllowTF32CuDNN(false);

  torch::Device device(torch::kCUDA, 0);
  if (!torch::cuda::is_available()) {
    device = torch::Device(torch::kCPU);
  }
  std::cout << "[DEVICE] Using device: " << (device.is_cuda() ? "CUDA" : "CPU") << "\n";

  std::vector<std::shared_ptr<open_spiel::SharedDunePolicyValueNetImpl>> models(num_evals);
  std::vector<std::unique_ptr<std::shared_mutex>> model_mutexes(num_evals);
  std::vector<std::shared_ptr<open_spiel::BatchedEvaluator>> coords(num_evals);
  std::vector<std::shared_ptr<open_spiel::BatchedNNEvaluator>> evaluators(num_evals);

  for (int e = 0; e < num_evals; ++e) {
    models[e] = std::make_shared<open_spiel::SharedDunePolicyValueNetImpl>(
        open_spiel::dune_imperium::kFullPublicInformationStateSize, 2048, 2391,
        num_blocks,
        /*nonlinear_value_head=*/false, /*with_aux_heads=*/false,
        /*head_init_seed=*/0, /*with_semantic_scorer=*/true);
    models[e]->to(device);
    open_spiel::LoadModelCheckpointRobust(models[e], model_path, device);
    models[e]->eval();

    model_mutexes[e] = std::make_unique<std::shared_mutex>();
    coords[e] = std::make_shared<open_spiel::BatchedEvaluator>(
        models[e], target_batch_size, /*timeout_ms=*/1, device, model_mutexes[e].get(), 10.0f,
        /*device_synchronize=*/false, /*high_priority_stream=*/true,
        /*emit_batch_membership=*/false,
        /*rollout_amp=*/false,
        /*allow_tf32=*/false);
    evaluators[e] = std::make_shared<open_spiel::BatchedNNEvaluator>(coords[e], 10.0f);
  }

  std::shared_ptr<const open_spiel::Game> game = open_spiel::LoadGame("dune_imperium");

  // Read roots
  std::ifstream roots_file(roots_path);
  if (!roots_file.is_open()) open_spiel::SpielFatalError("Failed to open " + roots_path);
  std::string roots_content((std::istreambuf_iterator<char>(roots_file)), std::istreambuf_iterator<char>());
  roots_file.close();
  auto roots_json_opt = open_spiel::json::FromString(roots_content);
  if (!roots_json_opt) open_spiel::SpielFatalError("Failed to parse " + roots_path);
  const auto& roots_obj = roots_json_opt->GetObject();
  const auto& roots_arr = roots_obj.at("roots").GetArray();

  std::vector<SampledRoot> sampled_roots;
  sampled_roots.reserve(roots_arr.size());
  for (const auto& r_val : roots_arr) {
    const auto& r = r_val.GetObject();
    SampledRoot sr;
    sr.root_id = static_cast<int>(r.at("root_id").GetInt());
    sr.game_id = static_cast<int>(r.at("game_id").GetInt());
    sr.decision_ordinal = static_cast<int>(r.at("decision_ordinal").GetInt());
    sr.obs_row = static_cast<int>(r.at("obs_row").GetInt());
    sr.search_seat = static_cast<int>(r.at("search_seat").GetInt());
    sr.round = static_cast<int>(r.at("round").GetInt());
    sr.round_band = r.at("round_band").GetString();
    sr.decision_type = r.at("decision_type").GetString();
    sr.stratum_weight = r.at("stratum_weight").GetDouble();
    sr.raw_action = static_cast<int>(r.at("raw_action").GetInt());
    sr.census_chosen_action = static_cast<int>(r.at("census_chosen_action").GetInt());
    sr.census_overridden = r.at("census_overridden").GetBool();
    for (const auto& a_val : r.at("legal_actions").GetArray()) {
      sr.legal_actions.push_back(static_cast<int>(a_val.GetInt()));
    }
    for (const auto& a_val : r.at("action_history").GetArray()) {
      sr.action_history.push_back(static_cast<int64_t>(a_val.GetInt()));
    }
    if (r.find("split") != r.end() && r.at("split").IsString()) {
      sr.split = r.at("split").GetString();
    } else {
      sr.split = "train";
    }
    if (r.find("pi_parent") != r.end() && r.at("pi_parent").IsArray()) {
      for (const auto& p_val : r.at("pi_parent").GetArray()) {
        sr.pi_parent.push_back(p_val.GetDouble());
      }
    }
    sampled_roots.push_back(std::move(sr));
  }

  if (placement_only) {
    std::vector<SampledRoot> filtered;
    for (auto& r : sampled_roots) {
      if (r.decision_type == "AgentPrimary" || r.decision_type == "AgentContinuation") {
        filtered.push_back(std::move(r));
      }
    }
    sampled_roots = std::move(filtered);
    std::cout << "[FILTER] Kept " << sampled_roots.size() << " placement roots (--placement_roots_only=true)\n";
  }

  if (absl::GetFlag(FLAGS_max_roots) > 0 &&
      static_cast<size_t>(absl::GetFlag(FLAGS_max_roots)) < sampled_roots.size()) {
    sampled_roots.resize(absl::GetFlag(FLAGS_max_roots));
    std::cout << "[LOAD] Trimmed to " << sampled_roots.size() << " roots (--max_roots flag).\n";
  }

  // Replay and verify observation binary
  std::ifstream obs_file(obs_path, std::ios::binary);
  if (!obs_file.is_open()) open_spiel::SpielFatalError("Failed to open " + obs_path);
  const size_t obs_elem_count = open_spiel::dune_imperium::kFullPublicInformationStateSize;
  const size_t obs_byte_stride = obs_elem_count * sizeof(float);
  std::vector<float> rec_obs_buffer(obs_elem_count);

  std::cout << "[VERIFY] Replaying and verifying observation parity...\n";
  for (size_t i = 0; i < sampled_roots.size(); ++i) {
    auto& root = sampled_roots[i];
    std::unique_ptr<State> state = game->NewInitialState();
    for (int64_t a : root.action_history) {
      state->ApplyAction(a);
    }
    if (state->CurrentPlayer() != root.search_seat) {
      open_spiel::SpielFatalError(absl::StrFormat("Root %d seat mismatch: state=%d, root=%d",
                                                 root.root_id, state->CurrentPlayer(), root.search_seat));
    }
    std::vector<float> recomputed_obs = evaluators[0]->GetConsumedObservation(*state, root.search_seat);
    obs_file.seekg(root.obs_row * obs_byte_stride);
    obs_file.read(reinterpret_cast<char*>(rec_obs_buffer.data()), obs_byte_stride);
    float max_diff = 0.0f;
    for (size_t j = 0; j < obs_elem_count; ++j) {
      float d = std::abs(recomputed_obs[j] - rec_obs_buffer[j]);
      if (d > max_diff) max_diff = d;
    }
    if (max_diff > 1e-4f) {
      open_spiel::SpielFatalError(absl::StrFormat("Observation mismatch on root %d: max_diff=%e", root.root_id, max_diff));
    }
    if (root.pi_parent.empty()) {
      auto prior = evaluators[0]->Prior(*state);
      root.pi_parent.assign(root.legal_actions.size(), 0.0);
      for (size_t idx = 0; idx < root.legal_actions.size(); ++idx) {
        int a = root.legal_actions[idx];
        for (const auto& ap : prior) {
          if (ap.first == a) { root.pi_parent[idx] = ap.second; break; }
        }
      }
    }
  }
  obs_file.close();
  std::cout << "[VERIFY] Observation parity verified for all " << sampled_roots.size() << " roots.\n";

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

  std::vector<LabelledResult> results(sampled_roots.size());
  std::unique_ptr<std::atomic<uint8_t>[]> completed(new std::atomic<uint8_t>[sampled_roots.size()]());
  std::atomic<int> completed_count{0};

  std::string ckpt_path = output_path + ".checkpoint.json";
  std::ifstream ckpt_file(ckpt_path);
  if (ckpt_file.is_open()) {
    std::string content((std::istreambuf_iterator<char>(ckpt_file)), std::istreambuf_iterator<char>());
    auto json_opt = open_spiel::json::FromString(content);
    if (json_opt && json_opt->IsObject()) {
      const auto& c_obj = json_opt->GetObject();
      if (c_obj.find("roots") != c_obj.end() && c_obj.at("roots").IsArray()) {
        for (const auto& r_val : c_obj.at("roots").GetArray()) {
          const auto& r = r_val.GetObject();
          int rid = static_cast<int>(r.at("root_id").GetInt());
          for (size_t i = 0; i < sampled_roots.size(); ++i) {
            if (sampled_roots[i].root_id == rid) {
              LabelledResult lr;
              lr.root_id = rid;
              lr.game_id = static_cast<int>(r.at("game_id").GetInt());
              lr.obs_row = static_cast<int>(r.at("obs_row").GetInt());
              lr.search_seat = static_cast<int>(r.at("search_seat").GetInt());
              lr.round = static_cast<int>(r.at("round").GetInt());
              lr.decision_type = r.at("decision_type").GetString();
              lr.split = r.at("split").GetString();
              lr.raw_action = static_cast<int>(r.at("raw_action").GetInt());
              lr.chosen_action = static_cast<int>(r.at("chosen_action").GetInt());
              lr.overridden = r.at("overridden").GetBool();
              lr.d2 = r.at("d2").GetDouble();
              lr.se2 = r.at("se2").GetDouble();
              lr.w = r.at("w").GetDouble();
              for (const auto& c_val : r.at("candidates").GetArray()) lr.candidates.push_back(static_cast<int>(c_val.GetInt()));
              for (const auto& a_val : r.at("legal_actions").GetArray()) lr.legal_actions.push_back(static_cast<int>(a_val.GetInt()));
              for (const auto& p_val : r.at("pi_parent").GetArray()) lr.pi_parent.push_back(p_val.GetDouble());
              lr.duration_ms = r.at("duration_ms").GetDouble();
              lr.stage1_kept_worlds = static_cast<int>(r.at("stage1_kept_worlds").GetInt());
              lr.stage2_kept_worlds = static_cast<int>(r.at("stage2_kept_worlds").GetInt());
              results[i] = std::move(lr);
              completed[i].store(1, std::memory_order_release);
              completed_count++;
              break;
            }
          }
        }
        std::cout << absl::StrFormat("[RESUME] Loaded %d completed roots from checkpoint %s\n", completed_count.load(), ckpt_path);
      }
    }
  }

  std::mutex save_mutex;
  auto save_checkpoint_fn = [&](const std::string& path) {
    open_spiel::json::Object c_obj;
    c_obj["total_roots"] = open_spiel::json::Value(static_cast<int64_t>(sampled_roots.size()));
    c_obj["completed_roots"] = open_spiel::json::Value(static_cast<int64_t>(completed_count.load()));
    open_spiel::json::Array arr;
    for (size_t i = 0; i < sampled_roots.size(); ++i) {
      if (completed[i].load(std::memory_order_acquire) == 0) continue;
      const auto& lr = results[i];
      open_spiel::json::Object r_obj;
      r_obj["root_id"] = open_spiel::json::Value(static_cast<int64_t>(lr.root_id));
      r_obj["game_id"] = open_spiel::json::Value(static_cast<int64_t>(lr.game_id));
      r_obj["obs_row"] = open_spiel::json::Value(static_cast<int64_t>(lr.obs_row));
      r_obj["search_seat"] = open_spiel::json::Value(static_cast<int64_t>(lr.search_seat));
      r_obj["round"] = open_spiel::json::Value(static_cast<int64_t>(lr.round));
      r_obj["decision_type"] = open_spiel::json::Value(lr.decision_type);
      r_obj["split"] = open_spiel::json::Value(lr.split);
      r_obj["raw_action"] = open_spiel::json::Value(static_cast<int64_t>(lr.raw_action));
      r_obj["chosen_action"] = open_spiel::json::Value(static_cast<int64_t>(lr.chosen_action));
      r_obj["overridden"] = open_spiel::json::Value(lr.overridden);
      r_obj["d2"] = open_spiel::json::Value(lr.d2);
      r_obj["se2"] = open_spiel::json::Value(lr.se2);
      r_obj["w"] = open_spiel::json::Value(lr.w);
      open_spiel::json::Array c_arr;
      for (int a : lr.candidates) c_arr.push_back(open_spiel::json::Value(static_cast<int64_t>(a)));
      r_obj["candidates"] = open_spiel::json::Value(c_arr);
      open_spiel::json::Array la_arr;
      for (int a : lr.legal_actions) la_arr.push_back(open_spiel::json::Value(static_cast<int64_t>(a)));
      r_obj["legal_actions"] = open_spiel::json::Value(la_arr);
      open_spiel::json::Array pi_arr;
      for (double p : lr.pi_parent) pi_arr.push_back(open_spiel::json::Value(p));
      r_obj["pi_parent"] = open_spiel::json::Value(pi_arr);
      r_obj["duration_ms"] = open_spiel::json::Value(lr.duration_ms);
      r_obj["stage1_kept_worlds"] = open_spiel::json::Value(static_cast<int64_t>(lr.stage1_kept_worlds));
      r_obj["stage2_kept_worlds"] = open_spiel::json::Value(static_cast<int64_t>(lr.stage2_kept_worlds));
      arr.push_back(open_spiel::json::Value(r_obj));
    }
    c_obj["roots"] = open_spiel::json::Value(arr);
    std::string tmp_path = path + ".tmp";
    std::ofstream ofs(tmp_path);
    if (ofs.is_open()) {
      ofs << open_spiel::json::ToString(open_spiel::json::Value(c_obj), true);
      ofs.close();
      std::filesystem::rename(tmp_path, path);
    }
  };

  std::atomic<size_t> next_root_idx{0};
  std::vector<std::thread> workers;
  workers.reserve(parallel_roots);

  auto search_start = std::chrono::steady_clock::now();

  for (int w = 0; w < parallel_roots; ++w) {
    workers.emplace_back([&, w]() {
      while (true) {
        size_t r_idx = next_root_idx.fetch_add(1);
        if (r_idx >= sampled_roots.size()) break;
        if (completed[r_idx].load(std::memory_order_acquire) != 0) continue;

        const auto& root = sampled_roots[r_idx];
        std::unique_ptr<State> state = game->NewInitialState();
        for (int64_t a : root.action_history) {
          state->ApplyAction(a);
        }

        uint64_t dec_seed = dune_seed::DeriveSeed(master_seed, label_seed_domain, static_cast<uint64_t>(root.root_id));
        int eff_top_k = (root.decision_type.rfind("Purchase", 0) == 0)
            ? static_cast<int>(root.legal_actions.size()) : 5;

        auto t0 = std::chrono::steady_clock::now();
        auto s_res = SuccessiveHalvingRolloutSearchDecision(
            *state, root.search_seat, nullptr, evaluators,
            /*reference_rollouts_per_action=*/256, /*min_override_margin=*/0.15,
            dec_seed, SearchRolloutCrnMode::kSequential, /*unbiased_stage2_gate=*/false,
            /*max_time_ms=*/-1.0, eff_top_k, max_inflight, use_bounded_pool);
        auto t1 = std::chrono::steady_clock::now();
        double dur = std::chrono::duration<double, std::milli>(t1 - t0).count();

        LabelledResult lr;
        lr.root_id = root.root_id;
        lr.game_id = root.game_id;
        lr.obs_row = root.obs_row;
        lr.search_seat = root.search_seat;
        lr.round = root.round;
        lr.decision_type = root.decision_type;
        lr.split = root.split;
        lr.raw_action = root.raw_action;
        lr.chosen_action = s_res.action;
        lr.overridden = s_res.overridden;
        lr.d2 = s_res.mean_d_stage2;
        lr.se2 = s_res.se_d_stage2;
        lr.w = s_res.overridden ? std::clamp((lr.d2 - lr.se2) / 0.15, 0.0, 1.0) : 0.0;
        for (const auto& cu : s_res.candidate_utilities) {
          lr.candidates.push_back(cu.first);
        }
        lr.legal_actions = root.legal_actions;
        lr.pi_parent = root.pi_parent;
        lr.duration_ms = dur;
        lr.stage1_kept_worlds = s_res.stage1_kept_worlds;
        lr.stage2_kept_worlds = s_res.stage2_kept_worlds;

        results[r_idx] = std::move(lr);
        completed[r_idx].store(1, std::memory_order_release);
        int done = completed_count.fetch_add(1) + 1;

        if (done % 10 == 0 || done == static_cast<int>(sampled_roots.size())) {
          std::cout << absl::StrFormat("[LABEL] Completed %d / %zu roots (Root %d, over=%s, w=%.3f, dur=%.1f ms)\n",
                                       done, sampled_roots.size(), root.root_id,
                                       results[r_idx].overridden ? "true" : "false", results[r_idx].w, dur);
        }
        if (done % 20 == 0 || done == static_cast<int>(sampled_roots.size())) {
          std::lock_guard<std::mutex> lock(save_mutex);
          save_checkpoint_fn(ckpt_path);
        }
      }
    });
  }

  for (auto& w : workers) {
    w.join();
  }

  auto search_end = std::chrono::steady_clock::now();
  double total_wall_s = std::chrono::duration<double>(search_end - search_start).count();
  double roots_per_s = sampled_roots.size() / std::max(1e-4, total_wall_s);

  int total_overrides = 0;
  double sum_w = 0.0;
  for (const auto& r : results) {
    if (r.overridden) total_overrides++;
    sum_w += r.w;
  }
  double override_rate = 100.0 * total_overrides / std::max<size_t>(1, results.size());
  double mean_w = sum_w / std::max<size_t>(1, results.size());

  open_spiel::json::Object meta_out;
  meta_out["total_roots"] = open_spiel::json::Value(static_cast<int64_t>(results.size()));
  meta_out["wall_time_sec"] = open_spiel::json::Value(total_wall_s);
  meta_out["roots_per_sec"] = open_spiel::json::Value(roots_per_s);
  meta_out["override_rate_pct"] = open_spiel::json::Value(override_rate);
  meta_out["total_overrides"] = open_spiel::json::Value(static_cast<int64_t>(total_overrides));
  meta_out["mean_weight_w"] = open_spiel::json::Value(mean_w);

  open_spiel::json::Array arr_out;
  for (const auto& lr : results) {
    open_spiel::json::Object r_obj;
    r_obj["root_id"] = open_spiel::json::Value(static_cast<int64_t>(lr.root_id));
    r_obj["game_id"] = open_spiel::json::Value(static_cast<int64_t>(lr.game_id));
    r_obj["obs_row"] = open_spiel::json::Value(static_cast<int64_t>(lr.obs_row));
    r_obj["search_seat"] = open_spiel::json::Value(static_cast<int64_t>(lr.search_seat));
    r_obj["round"] = open_spiel::json::Value(static_cast<int64_t>(lr.round));
    r_obj["decision_type"] = open_spiel::json::Value(lr.decision_type);
    r_obj["split"] = open_spiel::json::Value(lr.split);
    r_obj["raw_action"] = open_spiel::json::Value(static_cast<int64_t>(lr.raw_action));
    r_obj["chosen_action"] = open_spiel::json::Value(static_cast<int64_t>(lr.chosen_action));
    r_obj["overridden"] = open_spiel::json::Value(lr.overridden);
    r_obj["d2"] = open_spiel::json::Value(lr.d2);
    r_obj["se2"] = open_spiel::json::Value(lr.se2);
    r_obj["w"] = open_spiel::json::Value(lr.w);
    open_spiel::json::Array c_arr;
    for (int a : lr.candidates) c_arr.push_back(open_spiel::json::Value(static_cast<int64_t>(a)));
    r_obj["candidates"] = open_spiel::json::Value(c_arr);
    open_spiel::json::Array la_arr;
    for (int a : lr.legal_actions) la_arr.push_back(open_spiel::json::Value(static_cast<int64_t>(a)));
    r_obj["legal_actions"] = open_spiel::json::Value(la_arr);
    open_spiel::json::Array pi_arr;
    for (double p : lr.pi_parent) pi_arr.push_back(open_spiel::json::Value(p));
    r_obj["pi_parent"] = open_spiel::json::Value(pi_arr);
    r_obj["duration_ms"] = open_spiel::json::Value(lr.duration_ms);
    r_obj["stage1_kept_worlds"] = open_spiel::json::Value(static_cast<int64_t>(lr.stage1_kept_worlds));
    r_obj["stage2_kept_worlds"] = open_spiel::json::Value(static_cast<int64_t>(lr.stage2_kept_worlds));
    arr_out.push_back(open_spiel::json::Value(r_obj));
  }

  open_spiel::json::Object final_out;
  final_out["metadata"] = open_spiel::json::Value(meta_out);
  final_out["roots"] = open_spiel::json::Value(arr_out);

  {
    std::ofstream ofs(output_path);
    if (!ofs.is_open()) open_spiel::SpielFatalError("Failed to open " + output_path);
    ofs << open_spiel::json::ToString(open_spiel::json::Value(final_out), true) << "\n";
  }
  std::remove(ckpt_path.c_str());
  std::cout << absl::StrFormat("[LABEL] Finished labelling %zu roots in %.1f s (%.2f roots/s). Overrides: %d (%.1f%%), mean w = %.4f. Saved to %s\n",
                               results.size(), total_wall_s, roots_per_s, total_overrides, override_rate, mean_w, output_path);
  return 0;
}

int RunTrueStateTieBreakerAndHistoryReplay() {
  const std::string games_jsonl_path = absl::GetFlag(FLAGS_tie_breaker_games_jsonl);
  const std::string model_path = absl::GetFlag(FLAGS_model_checkpoint);
  const std::string output_path = absl::GetFlag(FLAGS_output_json);
  const bool run_rollouts = absl::GetFlag(FLAGS_tie_breaker_run_rollouts);
  const int max_sample = absl::GetFlag(FLAGS_tie_breaker_max_sample);
  const int tie_breaker_rollouts = absl::GetFlag(FLAGS_tie_breaker_rollouts);
  const int parallel_roots = std::max(1, absl::GetFlag(FLAGS_parallel_roots));
  const int num_threads = std::max(1, absl::GetFlag(FLAGS_threads));
  const uint64_t master_seed = absl::GetFlag(FLAGS_master_seed);
  const int num_blocks = absl::GetFlag(FLAGS_num_blocks);
  const int num_evals = absl::GetFlag(FLAGS_num_evaluators);
  const int target_batch_size = absl::GetFlag(FLAGS_target_batch_size);

  if (games_jsonl_path.empty()) {
    open_spiel::SpielFatalError("--tie_breaker_games_jsonl must be specified.");
  }

  std::cout << "================================================================================\n";
  std::cout << "MODE 3: TRUE-STATE TIE-BREAKER & HISTORY REPLAY VERIFICATION\n";
  std::cout << "================================================================================\n";
  std::cout << "Games JSONL:        " << games_jsonl_path << "\n";
  std::cout << "Model checkpoint:   " << model_path << "\n";
  std::cout << "Output JSON:        " << output_path << "\n";
  std::cout << "Run rollouts:       " << (run_rollouts ? "true" : "false") << "\n";
  std::cout << "Max sample:         " << max_sample << "\n";
  std::cout << "Rollouts/action:    " << tie_breaker_rollouts << "\n";
  std::cout << "Threads:            " << num_threads << "\n";
  std::cout << "Parallel roots:     " << parallel_roots << "\n";

  open_spiel::dune_imperium::DuneImperiumState::SetAutoPassEmptyCombatIntrigueHands(
      absl::GetFlag(FLAGS_auto_pass_empty_combat_intrigue));

  at::globalContext().setAllowTF32CuBLAS(false);
  at::globalContext().setAllowTF32CuDNN(false);

  torch::Device device(torch::kCUDA, 0);
  if (!torch::cuda::is_available()) {
    device = torch::Device(torch::kCPU);
  }

  std::vector<std::shared_ptr<open_spiel::SharedDunePolicyValueNetImpl>> models;
  std::vector<std::unique_ptr<std::shared_mutex>> model_mutexes;
  std::vector<std::shared_ptr<open_spiel::BatchedEvaluator>> coords;
  std::vector<std::shared_ptr<open_spiel::BatchedNNEvaluator>> evaluators;

  if (!model_path.empty()) {
    models.resize(num_evals);
    model_mutexes.resize(num_evals);
    coords.resize(num_evals);
    evaluators.resize(num_evals);

    for (int e = 0; e < num_evals; ++e) {
      models[e] = std::make_shared<open_spiel::SharedDunePolicyValueNetImpl>(
          open_spiel::dune_imperium::kFullPublicInformationStateSize, 2048, 2391,
          num_blocks,
          /*nonlinear_value_head=*/false, /*with_aux_heads=*/false,
          /*head_init_seed=*/0, /*with_semantic_scorer=*/true);
      models[e]->to(device);
      open_spiel::LoadModelCheckpointRobust(models[e], model_path, device);
      models[e]->eval();

      model_mutexes[e] = std::make_unique<std::shared_mutex>();
      coords[e] = std::make_shared<open_spiel::BatchedEvaluator>(
          models[e], target_batch_size, /*timeout_ms=*/1, device, model_mutexes[e].get(), 10.0f,
          /*device_synchronize=*/false, /*high_priority_stream=*/true,
          /*emit_batch_membership=*/false,
          /*rollout_amp=*/false,
          /*allow_tf32=*/false);
      evaluators[e] = std::make_shared<open_spiel::BatchedNNEvaluator>(coords[e], 10.0f);
    }
  }

  std::shared_ptr<const open_spiel::Game> game = open_spiel::LoadGame("dune_imperium");

  struct ParsedGame {
    int episode_id = 0;
    int seat = 0;
    int placement = 0;
    double recorded_return = 0.0;
    int round = 0;
    uint64_t seed = 0;
    std::vector<int64_t> history;
  };

  std::vector<ParsedGame> parsed_games;
  {
    std::ifstream ifs(games_jsonl_path);
    if (!ifs.is_open()) open_spiel::SpielFatalError("Failed to open " + games_jsonl_path);
    std::string line;
    while (std::getline(ifs, line)) {
      if (line.empty()) continue;
      uint64_t line_seed = 0;
      size_t seed_pos = line.find("\"seed\":");
      if (seed_pos != std::string::npos) {
        size_t val_pos = seed_pos + 7;
        while (val_pos < line.size() && (line[val_pos] == ' ' || line[val_pos] == '\t')) {
          val_pos++;
        }
        if (val_pos < line.size() && (std::isdigit(line[val_pos]) || line[val_pos] == '-')) {
          char* end_ptr = nullptr;
          line_seed = std::strtoull(line.c_str() + val_pos, &end_ptr, 10);
          size_t end_idx = (end_ptr != nullptr) ? (end_ptr - line.c_str()) : val_pos;
          if (end_idx > val_pos) {
            line.replace(val_pos, end_idx - val_pos, "0");
          }
        }
      }
      auto json_opt = open_spiel::json::FromString(line);
      if (!json_opt || !json_opt->IsObject()) continue;
      const auto& obj = json_opt->GetObject();
      ParsedGame pg;
      pg.seed = line_seed;
      pg.episode_id = static_cast<int>(obj.at("episode_id").GetInt());
      pg.seat = static_cast<int>(obj.at("seat").GetInt());
      pg.placement = static_cast<int>(obj.at("placement").GetInt());
      pg.recorded_return = obj.at("return").GetDouble();
      if (obj.find("ending_round") != obj.end()) pg.round = static_cast<int>(obj.at("ending_round").GetInt());
      else if (obj.find("round") != obj.end()) pg.round = static_cast<int>(obj.at("round").GetInt());
      if (obj.find("history") != obj.end() && obj.at("history").IsArray()) {
        for (const auto& h_val : obj.at("history").GetArray()) {
          pg.history.push_back(static_cast<int64_t>(h_val.GetInt()));
        }
      }
      parsed_games.push_back(std::move(pg));
    }
  }
  std::cout << absl::StrFormat("[TIE-BREAKER] Loaded %zu games from %s\n", parsed_games.size(), games_jsonl_path);

  struct ChangedDecision {
    int game_idx = 0;
    int seat = 0;
    Action a_parent = 0;
    Action a_student = 0;
    std::vector<int64_t> history_prefix;
  };

  std::vector<std::vector<ChangedDecision>> per_game_changed(parsed_games.size());
  std::atomic<size_t> next_g_idx{0};
  std::vector<std::thread> workers;
  workers.reserve(num_threads);

  for (int t = 0; t < num_threads; ++t) {
    workers.emplace_back([&, t]() {
      int eval_idx = evaluators.empty() ? 0 : (t % evaluators.size());
      while (true) {
        size_t g = next_g_idx.fetch_add(1);
        if (g >= parsed_games.size()) break;
        const auto& pg = parsed_games[g];

        std::unique_ptr<State> state = game->NewInitialState();
        std::vector<int64_t> prefix;
        prefix.reserve(pg.history.size());

        for (size_t step = 0; step < pg.history.size(); ++step) {
          int64_t a = pg.history[step];
          if (!model_path.empty() && !state->IsChanceNode() && state->CurrentPlayer() == pg.seat) {
            auto legals = state->LegalActions();
            if (legals.size() > 1) {
              auto prior = evaluators[eval_idx]->Prior(*state);
              Action a_parent = legals.front();
              double max_p = -1.0;
              for (const auto& ap : prior) {
                if (ap.second > max_p) {
                  max_p = ap.second;
                  a_parent = ap.first;
                }
              }
              Action a_student = static_cast<Action>(a);
              if (a_student != a_parent) {
                ChangedDecision cd;
                cd.game_idx = static_cast<int>(g);
                cd.seat = pg.seat;
                cd.a_parent = a_parent;
                cd.a_student = a_student;
                cd.history_prefix = prefix;
                per_game_changed[g].push_back(std::move(cd));
              }
            }
          }
          state->ApplyAction(a);
          prefix.push_back(a);
        }

        if (!state->IsTerminal()) {
          open_spiel::SpielFatalError(absl::StrFormat("Game %d (episode %d) did not terminate at end of history!", g, pg.episode_id));
        }
        double r_ret = state->Returns()[pg.seat];
        if (std::abs(r_ret - pg.recorded_return) >= 1e-4) {
          open_spiel::SpielFatalError(absl::StrFormat("Game %d (episode %d) return mismatch: actual=%f, recorded=%f",
                                                     g, pg.episode_id, r_ret, pg.recorded_return));
        }
        int comp_placement = 4;
        if (std::abs(r_ret - 2.25) < 1e-4) comp_placement = 1;
        else if (std::abs(r_ret - 0.25) < 1e-4) comp_placement = 2;
        else if (std::abs(r_ret - (-0.75)) < 1e-4) comp_placement = 3;
        else if (std::abs(r_ret - (-1.75)) < 1e-4) comp_placement = 4;
        else {
          int rank = 1;
          for (int p = 0; p < 4; ++p) {
            if (p != pg.seat && state->Returns()[p] > r_ret + 1e-4) rank++;
          }
          comp_placement = rank;
        }
        if (comp_placement != pg.placement) {
          open_spiel::SpielFatalError(absl::StrFormat("Game %d (episode %d) placement mismatch: comp=%d, recorded=%d",
                                                     g, pg.episode_id, comp_placement, pg.placement));
        }
      }
    });
  }

  for (auto& w : workers) {
    w.join();
  }

  std::cout << "[VERIFY] 100% replay match verified (terminal, returns, and placements matched across all games).\n";

  std::vector<ChangedDecision> all_changed;
  for (size_t g = 0; g < parsed_games.size(); ++g) {
    for (auto& cd : per_game_changed[g]) {
      all_changed.push_back(std::move(cd));
    }
  }

  const size_t N_games = parsed_games.size();
  const size_t N_changed = all_changed.size();
  const double r = static_cast<double>(N_changed) / static_cast<double>(N_games);
  std::cout << absl::StrFormat("[TIE-BREAKER] Found %zu changed decisions across %zu games (r = %.4f changed/game)\n",
                               N_changed, N_games, r);

  open_spiel::json::Object summary;
  summary["total_games"] = open_spiel::json::Value(static_cast<int64_t>(N_games));
  summary["total_changed_decisions"] = open_spiel::json::Value(static_cast<int64_t>(N_changed));
  summary["r_changed_per_game"] = open_spiel::json::Value(r);
  summary["replay_verified"] = open_spiel::json::Value(true);

  if (run_rollouts && N_changed > 0 && !evaluators.empty()) {
    size_t M = std::min<size_t>(static_cast<size_t>(max_sample), N_changed);
    std::mt19937 rng(master_seed);
    std::vector<size_t> perm(N_changed);
    std::iota(perm.begin(), perm.end(), 0);
    std::shuffle(perm.begin(), perm.end(), rng);
    perm.resize(M);
    std::sort(perm.begin(), perm.end());

    std::vector<ChangedDecision> sampled_decisions;
    sampled_decisions.reserve(M);
    for (size_t idx : perm) {
      sampled_decisions.push_back(all_changed[idx]);
    }

    std::cout << absl::StrFormat("[TIE-BREAKER] Running %d paired true-state rollouts on %zu sampled changed decisions...\n",
                                 tie_breaker_rollouts, M);

    std::vector<double> deltas(M, 0.0);
    std::atomic<size_t> next_sample_idx{0};
    std::vector<std::thread> roll_workers;
    roll_workers.reserve(parallel_roots);

    for (int pw = 0; pw < parallel_roots; ++pw) {
      roll_workers.emplace_back([&, pw]() {
        while (true) {
          size_t i = next_sample_idx.fetch_add(1);
          if (i >= M) break;
          const auto& cd = sampled_decisions[i];
          std::unique_ptr<State> state = game->NewInitialState();
          for (int64_t a : cd.history_prefix) {
            state->ApplyAction(a);
          }
          uint64_t dec_seed = dune_seed::DeriveSeed(master_seed, 0x4509ULL, static_cast<uint64_t>(i));
          std::vector<Action> cands = {cd.a_parent, cd.a_student};
          auto s_res = PairedPriorSortedRolloutSearchDecision(
              *state, cd.seat, nullptr, evaluators, cands,
              tie_breaker_rollouts, /*min_override_margin=*/0.0, /*min_t_stat=*/0.0,
              dec_seed, SearchRolloutCrnMode::kSequential, /*sample_opponent_leader_draft=*/false,
              /*max_time_ms=*/-1.0, absl::GetFlag(FLAGS_max_inflight_worlds),
              /*resample_from_infostate=*/false, /*use_bounded_pool_when_unclocked=*/true);

          double u_student = 0.0, u_parent = 0.0;
          for (const auto& cu : s_res.candidate_utilities) {
            if (cu.first == cd.a_student) u_student = cu.second;
            if (cu.first == cd.a_parent) u_parent = cu.second;
          }
          deltas[i] = u_student - u_parent;

          if ((i + 1) % 20 == 0 || i + 1 == M) {
            std::cout << absl::StrFormat("  [TIE-BREAKER] Evaluated %zu / %zu changed moves (delta=%.4f)\n",
                                         i + 1, M, deltas[i]);
          }
        }
      });
    }

    for (auto& w : roll_workers) {
      w.join();
    }

    double sum_delta = 0.0;
    for (double d : deltas) sum_delta += d;
    double mean_delta = sum_delta / static_cast<double>(M);
    double dU = r * mean_delta;

    // Game-clustered bootstrap
    std::vector<double> sampled_delta_sum(N_games, 0.0);
    std::vector<size_t> sampled_delta_cnt(N_games, 0);
    for (size_t i = 0; i < M; ++i) {
      int g = sampled_decisions[i].game_idx;
      sampled_delta_sum[g] += deltas[i];
      sampled_delta_cnt[g]++;
    }

    std::mt19937 boot_rng(master_seed + 12345);
    std::uniform_int_distribution<size_t> g_dist(0, N_games - 1);
    std::vector<double> boot_dU(10000);
    for (int b = 0; b < 10000; ++b) {
      double boot_c_sum = 0.0;
      double boot_delta_sum = 0.0;
      size_t boot_delta_cnt = 0;
      for (size_t k = 0; k < N_games; ++k) {
        size_t g = g_dist(boot_rng);
        boot_c_sum += static_cast<double>(per_game_changed[g].size());
        boot_delta_sum += sampled_delta_sum[g];
        boot_delta_cnt += sampled_delta_cnt[g];
      }
      double r_boot = boot_c_sum / static_cast<double>(N_games);
      double mean_delta_boot = (boot_delta_cnt > 0) ? (boot_delta_sum / static_cast<double>(boot_delta_cnt)) : mean_delta;
      boot_dU[b] = r_boot * mean_delta_boot;
    }
    std::sort(boot_dU.begin(), boot_dU.end());
    double ci_low = boot_dU[250];
    double ci_high = boot_dU[9750];
    double boot_mean = std::accumulate(boot_dU.begin(), boot_dU.end(), 0.0) / 10000.0;
    double boot_var = 0.0;
    for (double d : boot_dU) {
      double diff = d - boot_mean;
      boot_var += diff * diff;
    }
    double boot_se = std::sqrt(boot_var / 9999.0);

    std::cout << "================================================================================\n";
    std::cout << absl::StrFormat("  Tie-Breaker dU: %.4f (SE: %.4f, 95%% CI: [%.4f, %.4f])\n",
                                 dU, boot_se, ci_low, ci_high);
    std::cout << absl::StrFormat("  Mean Delta:     %.4f over %zu moves\n", mean_delta, M);
    std::cout << "================================================================================\n";

    summary["sampled_changed_decisions"] = open_spiel::json::Value(static_cast<int64_t>(M));
    summary["mean_delta"] = open_spiel::json::Value(mean_delta);
    summary["du_per_seat_game"] = open_spiel::json::Value(dU);
    summary["du_bootstrap_se"] = open_spiel::json::Value(boot_se);
    summary["du_bootstrap_ci_low"] = open_spiel::json::Value(ci_low);
    summary["du_bootstrap_ci_high"] = open_spiel::json::Value(ci_high);
    summary["rollouts_per_action"] = open_spiel::json::Value(static_cast<int64_t>(tie_breaker_rollouts));
  }

  if (!output_path.empty()) {
    std::ofstream ofs(output_path);
    if (ofs.is_open()) {
      ofs << open_spiel::json::ToString(open_spiel::json::Value(summary), true) << "\n";
      std::cout << "[SAVED] Summary written to " << output_path << "\n";
    }
  }

  return 0;
}


int main(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);

  // Dispatch to R7-45 Step 1 Mode 1: Raw Greedy Self-Play Root & Preservation Capture Mode
  if (absl::GetFlag(FLAGS_capture_self_play_games) > 0) {
    return RunSelfPlayRootAndPreservationCapture();
  }

  // Dispatch to R7-45 Step 1 Mode 3: True-State Tie-Breaker & History Replay Verification Mode
  if (!absl::GetFlag(FLAGS_tie_breaker_games_jsonl).empty()) {
    return RunTrueStateTieBreakerAndHistoryReplay();
  }

  // Dispatch to R7-45 Step 1 Mode 2: Parallel-Roots sh_768_k5_allbuy Labelling Mode
  if (absl::GetFlag(FLAGS_sh_distill_label)) {
    return RunSuccessiveHalvingDistillLabel();
  }

  const std::string model_path = absl::GetFlag(FLAGS_model_checkpoint);
  const std::string roots_path = absl::GetFlag(FLAGS_sampled_roots_json);
  const std::string obs_path = absl::GetFlag(FLAGS_obs_binary);
  const std::string output_path = absl::GetFlag(FLAGS_output_json);
  const int num_evals = absl::GetFlag(FLAGS_num_evaluators);
  const int target_batch_size = absl::GetFlag(FLAGS_target_batch_size);
  const int top_k = absl::GetFlag(FLAGS_top_k_actions);
  const int reference_top_k = (absl::GetFlag(FLAGS_sh_offline_check) || absl::GetFlag(FLAGS_sh_budget_check))
      ? std::max(top_k, 5) : top_k;
  const double min_override_margin = absl::GetFlag(FLAGS_min_override_margin);
  const uint64_t master_seed = absl::GetFlag(FLAGS_master_seed);

  // Dispatch to held-out trajectory replay mode if --replay_trajectory_jsonl is provided
  if (!absl::GetFlag(FLAGS_replay_trajectory_jsonl).empty()) {
    std::string traj_meta = absl::GetFlag(FLAGS_replay_trajectory_jsonl);
    std::string traj_obs = absl::GetFlag(FLAGS_replay_obs_bin);
    if (traj_obs.empty()) {
      traj_obs = absl::GetFlag(FLAGS_obs_binary);
    }
    if (model_path.empty() || traj_meta.empty() || traj_obs.empty()) {
      std::cerr << "Usage for replay: dune_fixed_root_search_benchmark --model_checkpoint=... --replay_trajectory_jsonl=... --replay_obs_bin=... [--output_json=...]\n";
      return 1;
    }
    return RunHeldOutReplayEvaluation(model_path, traj_meta, traj_obs, output_path);
  }

  if (model_path.empty() || roots_path.empty() || obs_path.empty() || output_path.empty()) {
    std::cerr << "Usage: dune_fixed_root_search_benchmark --model_checkpoint=... --sampled_roots_json=... --obs_binary=... --output_json=...\n";
    return 1;
  }

  std::cout << "================================================================================\n";
  std::cout << "STEP 1: FIXED-ROOT SEARCH BENCHMARK HARNESS\n";
  std::cout << "================================================================================\n";
  std::cout << "Model checkpoint:   " << model_path << "\n";
  std::cout << "Sampled roots JSON: " << roots_path << "\n";
  std::cout << "Observation bin:    " << obs_path << "\n";
  std::cout << "Master seed:        " << master_seed << "\n";

  // Load Game
  std::shared_ptr<const open_spiel::Game> game = open_spiel::LoadGame("dune_imperium");
  std::string current_game_str = game->ToString();

  // Read Sampled Roots JSON
  std::ifstream roots_file(roots_path);
  if (!roots_file.is_open()) {
    open_spiel::SpielFatalError("Failed to open sampled roots file: " + roots_path);
  }
  std::string roots_content((std::istreambuf_iterator<char>(roots_file)),
                            std::istreambuf_iterator<char>());
  roots_file.close();

  auto roots_json_opt = open_spiel::json::FromString(roots_content);
  if (!roots_json_opt) {
    open_spiel::SpielFatalError("Failed to parse sampled roots JSON from: " + roots_path);
  }
  const auto& roots_obj = roots_json_opt->GetObject();

  // MUST FIX 4(a): Fatal check if metadata.game_string is missing or mismatched
  if (roots_obj.find("metadata") == roots_obj.end()) {
    open_spiel::SpielFatalError("Fatal: 'metadata' object is missing from sampled roots JSON: " + roots_path);
  }
  const auto& meta_obj = roots_obj.at("metadata").GetObject();
  if (meta_obj.find("game_string") == meta_obj.end()) {
    open_spiel::SpielFatalError("Fatal: 'metadata.game_string' is missing from sampled roots JSON: " + roots_path);
  }
  std::string expected_game_str = meta_obj.at("game_string").GetString();
  if (expected_game_str.empty()) {
    open_spiel::SpielFatalError("Fatal: 'metadata.game_string' in sampled roots JSON is empty!");
  }
  if (expected_game_str != current_game_str) {
    if (absl::GetFlag(FLAGS_skip_game_string_check)) {
      std::cout << absl::StrFormat(
          "[WARNING] Game string mismatch between census and benchmark (ignored due to --skip_game_string_check)!\nExpected: %s\nActual:   %s\n",
          expected_game_str, current_game_str);
    } else {
      open_spiel::SpielFatalError(absl::StrFormat(
          "Fatal: Game string mismatch between census and benchmark!\nExpected: %s\nActual:   %s",
          expected_game_str, current_game_str));
    }
  } else {
    std::cout << "[PARITY] Verified game->ToString() matches census summary string.\n";
  }

  std::vector<SampledRoot> sampled_roots;
  const auto& roots_arr = roots_obj.at("roots").GetArray();
  sampled_roots.reserve(roots_arr.size());
  for (const auto& r_val : roots_arr) {
    const auto& r = r_val.GetObject();
    SampledRoot sr;
    sr.root_id = static_cast<int>(r.at("root_id").GetInt());
    sr.game_id = static_cast<int>(r.at("game_id").GetInt());
    sr.decision_ordinal = static_cast<int>(r.at("decision_ordinal").GetInt());
    sr.obs_row = static_cast<int>(r.at("obs_row").GetInt());
    sr.search_seat = static_cast<int>(r.at("search_seat").GetInt());
    sr.round = static_cast<int>(r.at("round").GetInt());
    sr.round_band = r.at("round_band").GetString();
    sr.decision_type = r.at("decision_type").GetString();
    sr.stratum_weight = r.at("stratum_weight").GetDouble();
    sr.raw_action = static_cast<int>(r.at("raw_action").GetInt());
    sr.census_chosen_action = static_cast<int>(r.at("census_chosen_action").GetInt());
    sr.census_overridden = r.at("census_overridden").GetBool();

    for (const auto& a_val : r.at("legal_actions").GetArray()) {
      sr.legal_actions.push_back(static_cast<int>(a_val.GetInt()));
    }
    for (const auto& a_val : r.at("action_history").GetArray()) {
      sr.action_history.push_back(static_cast<int64_t>(a_val.GetInt()));
    }
    if (r.find("split") != r.end() && r.at("split").IsString()) {
      sr.split = r.at("split").GetString();
    } else {
      sr.split = "train";
    }
    if (r.find("pi_parent") != r.end() && r.at("pi_parent").IsArray()) {
      for (const auto& p_val : r.at("pi_parent").GetArray()) {
        sr.pi_parent.push_back(p_val.GetDouble());
      }
    }
    sampled_roots.push_back(std::move(sr));
  }
  if (absl::GetFlag(FLAGS_max_roots) > 0 &&
      static_cast<size_t>(absl::GetFlag(FLAGS_max_roots)) < sampled_roots.size()) {
    sampled_roots.resize(absl::GetFlag(FLAGS_max_roots));
    std::cout << "[LOAD] Trimmed to " << sampled_roots.size() << " roots (--max_roots flag).\n";
  } else {
    std::cout << "[LOAD] Loaded " << sampled_roots.size() << " sampled roots for benchmark.\n";
  }

  // Initialize Torch Device & Evaluators
  torch::Device device(torch::kCUDA, 0);
  if (!torch::cuda::is_available()) {
    device = torch::Device(torch::kCPU);
  }
  std::cout << "[DEVICE] Using device: " << (device.is_cuda() ? "CUDA" : "CPU") << "\n";

  std::vector<std::shared_ptr<open_spiel::SharedDunePolicyValueNetImpl>> models(num_evals);
  std::vector<std::unique_ptr<std::shared_mutex>> model_mutexes(num_evals);
  std::vector<std::shared_ptr<open_spiel::BatchedEvaluator>> coords(num_evals);
  std::vector<std::shared_ptr<open_spiel::BatchedNNEvaluator>> evaluators(num_evals);

  for (int e = 0; e < num_evals; ++e) {
    models[e] = std::make_shared<open_spiel::SharedDunePolicyValueNetImpl>(
        open_spiel::dune_imperium::kFullPublicInformationStateSize, 2048, 2391,
        absl::GetFlag(FLAGS_num_blocks),
        /*nonlinear_value_head=*/false, /*with_aux_heads=*/false,
        /*head_init_seed=*/0, /*with_semantic_scorer=*/true);
    models[e]->to(device);
    open_spiel::LoadModelCheckpointRobust(models[e], model_path, device);
    models[e]->eval();

    model_mutexes[e] = std::make_unique<std::shared_mutex>();
    coords[e] = std::make_shared<open_spiel::BatchedEvaluator>(
        models[e], target_batch_size, /*timeout_ms=*/1, device, model_mutexes[e].get(), 10.0f,
        /*device_synchronize=*/false, /*high_priority_stream=*/true,
        /*emit_batch_membership=*/false,
        /*rollout_amp=*/absl::GetFlag(FLAGS_rollout_amp),
        /*allow_tf32=*/absl::GetFlag(FLAGS_allow_tf32));
    evaluators[e] = std::make_shared<open_spiel::BatchedNNEvaluator>(coords[e], 10.0f);
  }
  std::cout << "[MODEL] Successfully initialized " << num_evals << " evaluator coordinators.\n";

  // Replay check & observation verification for all roots
  std::ifstream obs_file(obs_path, std::ios::binary);
  if (!obs_file.is_open()) {
    open_spiel::SpielFatalError("Failed to open observation binary: " + obs_path);
  }

  const size_t obs_elem_count = open_spiel::dune_imperium::kFullPublicInformationStateSize;
  const size_t obs_byte_stride = obs_elem_count * sizeof(float);
  std::vector<float> rec_obs_buffer(obs_elem_count);

  if (absl::GetFlag(FLAGS_sh_budget_check)) {
    open_spiel::dune_imperium::DuneImperiumState::SetAutoPassEmptyCombatIntrigueHands(false);
  }

  std::cout << "[VERIFY] Replaying action histories and verifying observations against binary...\n";
  for (size_t i = 0; i < sampled_roots.size(); ++i) {
    const auto& root = sampled_roots[i];
    std::unique_ptr<open_spiel::State> state = game->NewInitialState();
    bool replay_failed = false;
    for (int64_t a : root.action_history) {
      std::vector<open_spiel::Action> legals = state->LegalActions();
      bool is_legal = false;
      for (open_spiel::Action la : legals) {
        if (la == a) {
          is_legal = true;
          break;
        }
      }
      if (!is_legal) {
        replay_failed = true;
        break;
      }
      state->ApplyAction(a);
    }
    if (replay_failed || state->CurrentPlayer() != root.search_seat) {
      open_spiel::SpielFatalError(absl::StrFormat(
          "Root %d seat mismatch or replay failed: state->CurrentPlayer()=%d, root.search_seat=%d",
          root.root_id, state->CurrentPlayer(), root.search_seat));
    }

    // Rebuild observation with exact same evaluator call
    std::vector<float> recomputed_obs = evaluators[0]->GetConsumedObservation(*state, root.search_seat);
    if (recomputed_obs.size() != obs_elem_count) {
      open_spiel::SpielFatalError(absl::StrFormat(
          "Observation size mismatch: recomputed=%zu, expected=%zu",
          recomputed_obs.size(), obs_elem_count));
    }

    // Read recorded observation from binary
    obs_file.seekg(root.obs_row * obs_byte_stride);
    obs_file.read(reinterpret_cast<char*>(rec_obs_buffer.data()), obs_byte_stride);
    if (!obs_file.good()) {
      open_spiel::SpielFatalError(absl::StrFormat(
          "Failed to read observation binary at row %d for root %d", root.obs_row, root.root_id));
    }

    // Check tolerance
    float max_diff = 0.0f;
    for (size_t j = 0; j < obs_elem_count; ++j) {
      float diff = std::abs(recomputed_obs[j] - rec_obs_buffer[j]);
      if (diff > max_diff) max_diff = diff;
    }
    if (max_diff > 1e-4f) {
      open_spiel::SpielFatalError(absl::StrFormat(
          "Observation mismatch on root %d (game %d, dec %d): max_diff=%e > 1e-4",
          root.root_id, root.game_id, root.decision_ordinal, max_diff));
    }
  }
  obs_file.close();
  std::cout << absl::StrFormat("[VERIFY] Verified replay, seat and observation parity for %zu roots.\n", sampled_roots.size());
  if (sampled_roots.empty()) {
    open_spiel::SpielFatalError("No roots provided for benchmark!");
  }

  // Configure Variants
  std::vector<VariantConfig> variants;
  if (absl::GetFlag(FLAGS_sh_budget_check)) {
    variants = {
        {"raw", 0, SearchPairingMode::kUnpaired, SearchTruncationMode::kTerminal,
         benchmark_domains::kDomainRaw, BenchmarkVariantType::kRaw, 0, 0, false},
        {"sh_192", 192, SearchPairingMode::kPaired, SearchTruncationMode::kTerminal,
         benchmark_domains::kDomainSHBudgetShared, BenchmarkVariantType::kSuccessiveHalvingDynamic, 64, 0, false},
        {"sh_384", 384, SearchPairingMode::kPaired, SearchTruncationMode::kTerminal,
         benchmark_domains::kDomainSHBudgetShared, BenchmarkVariantType::kSuccessiveHalvingDynamic, 128, 0, false},
        {"sh_384_k5", 384, SearchPairingMode::kPaired, SearchTruncationMode::kTerminal,
         benchmark_domains::kDomainSHBudgetShared, BenchmarkVariantType::kSuccessiveHalvingFixedTop5, 128, 5, false},
        {"sh_768", 768, SearchPairingMode::kPaired, SearchTruncationMode::kTerminal,
         benchmark_domains::kDomainSHBudgetShared, BenchmarkVariantType::kSuccessiveHalvingDynamic, 256, 0, false},
        {"sh_768_k5", 768, SearchPairingMode::kPaired, SearchTruncationMode::kTerminal,
         benchmark_domains::kDomainSHBudgetShared, BenchmarkVariantType::kSuccessiveHalvingFixedTop5, 256, 5, false},
        {"sh_768_k8", 768, SearchPairingMode::kPaired, SearchTruncationMode::kTerminal,
         benchmark_domains::kDomainSHBudgetShared, BenchmarkVariantType::kSuccessiveHalvingFixedTop5, 256, 8, false},
        {"sh_768_k5_allbuy", 768, SearchPairingMode::kPaired, SearchTruncationMode::kTerminal,
         benchmark_domains::kDomainSHBudgetShared, BenchmarkVariantType::kSuccessiveHalvingFixedTop5, 256, 5, true},
        {"sh_768_rep", 768, SearchPairingMode::kPaired, SearchTruncationMode::kTerminal,
         benchmark_domains::kDomainSHBudgetRep, BenchmarkVariantType::kSuccessiveHalvingDynamic, 256, 0, false},
        {"Ref2048", 2048, SearchPairingMode::kPaired, SearchTruncationMode::kTerminal,
         benchmark_domains::kDomainSHBudgetRef2048, BenchmarkVariantType::kPairedReferenceOnChosenActions, 0, 0, false}
    };
  } else if (absl::GetFlag(FLAGS_sh_offline_check)) {
    variants = {
        {"raw", 0, SearchPairingMode::kUnpaired, SearchTruncationMode::kTerminal,
         benchmark_domains::kDomainRaw, BenchmarkVariantType::kRaw, 0, 0, false},
        {"sh_192_incumbent", 192, SearchPairingMode::kPaired, SearchTruncationMode::kTerminal,
         benchmark_domains::kDomainSH192Incumbent, BenchmarkVariantType::kSuccessiveHalvingDynamic, 64, 0, false},
        {"sh_384_incumbent", 384, SearchPairingMode::kPaired, SearchTruncationMode::kTerminal,
         benchmark_domains::kDomainSH384Incumbent, BenchmarkVariantType::kSuccessiveHalvingDynamic, 128, 0, false},
        {"sh_384_k5", 384, SearchPairingMode::kPaired, SearchTruncationMode::kTerminal,
         benchmark_domains::kDomainSH384K5, BenchmarkVariantType::kSuccessiveHalvingFixedTop5, 128, 5, false},
        {"Ref512", 512, SearchPairingMode::kPaired, SearchTruncationMode::kTerminal,
         benchmark_domains::kDomainSHRef512, BenchmarkVariantType::kCompound, 0, 0, false}
    };
  } else if (absl::GetFlag(FLAGS_leaf_check_only)) {
    variants = {
        {"V2_16_Paired", 16, SearchPairingMode::kPaired, SearchTruncationMode::kNextRoundSearchPlayer,
         benchmark_domains::kDomainV2_16_Paired, BenchmarkVariantType::kCompound, 0, 0, false},
        {"Ref512", 512, SearchPairingMode::kPaired, SearchTruncationMode::kTerminal,
         benchmark_domains::kDomainRef512, BenchmarkVariantType::kCompound, 0, 0, false}
    };
  } else {
    variants = {
        {"V0_Run1", 64, SearchPairingMode::kUnpaired, SearchTruncationMode::kTerminal,
         benchmark_domains::kDomainV0_Run1, BenchmarkVariantType::kCompound, 0, 0, false},
        {"V0_Run2", 64, SearchPairingMode::kUnpaired, SearchTruncationMode::kTerminal,
         benchmark_domains::kDomainV0_Run2, BenchmarkVariantType::kCompound, 0, 0, false},
        {"V1", 64, SearchPairingMode::kPaired, SearchTruncationMode::kTerminal,
         benchmark_domains::kDomainV1, BenchmarkVariantType::kCompound, 0, 0, false},
        {"V2_16_Paired", 16, SearchPairingMode::kPaired, SearchTruncationMode::kNextRoundSearchPlayer,
         benchmark_domains::kDomainV2_16_Paired, BenchmarkVariantType::kCompound, 0, 0, false},
        {"V2_16_Unpaired", 16, SearchPairingMode::kUnpaired, SearchTruncationMode::kNextRoundSearchPlayer,
         benchmark_domains::kDomainV2_16_Unpaired, BenchmarkVariantType::kCompound, 0, 0, false},
        {"V2_32_Paired", 32, SearchPairingMode::kPaired, SearchTruncationMode::kNextRoundSearchPlayer,
         benchmark_domains::kDomainV2_32_Paired, BenchmarkVariantType::kCompound, 0, 0, false},
        {"V2_32_Unpaired", 32, SearchPairingMode::kUnpaired, SearchTruncationMode::kNextRoundSearchPlayer,
         benchmark_domains::kDomainV2_32_Unpaired, BenchmarkVariantType::kCompound, 0, 0, false},
        {"V2_64_Paired", 64, SearchPairingMode::kPaired, SearchTruncationMode::kNextRoundSearchPlayer,
         benchmark_domains::kDomainV2_64_Paired, BenchmarkVariantType::kCompound, 0, 0, false},
        {"V2_64_Unpaired", 64, SearchPairingMode::kUnpaired, SearchTruncationMode::kNextRoundSearchPlayer,
         benchmark_domains::kDomainV2_64_Unpaired, BenchmarkVariantType::kCompound, 0, 0, false},
        {"Ref512", 512, SearchPairingMode::kPaired, SearchTruncationMode::kTerminal,
         benchmark_domains::kDomainRef512, BenchmarkVariantType::kCompound, 0, 0, false}
    };
  }

  std::cout << "\n[BENCHMARK] Executing " << variants.size() << " variants across " << sampled_roots.size() << " roots...\n";

  // Data storage: results[root_idx][variant_idx]
  std::vector<std::vector<VariantResult>> all_results(sampled_roots.size(),
                                                      std::vector<VariantResult>(variants.size()));

  std::vector<std::vector<bool>> has_base_variant(sampled_roots.size(),
                                                  std::vector<bool>(variants.size(), false));
  const std::string base_receipt_json = absl::GetFlag(FLAGS_base_receipt_json);
  if (!base_receipt_json.empty()) {
    if (!LoadBaseReceiptVariants(base_receipt_json, sampled_roots, variants, all_results, has_base_variant)) {
      open_spiel::SpielFatalError(absl::StrFormat("Failed to load base receipt JSON: %s", base_receipt_json));
    }
  }

  const std::string checkpoint_path = output_path + ".checkpoint.json";
  size_t start_root_idx = 0;
  if (LoadBenchmarkCheckpoint(checkpoint_path, variants, all_results, start_root_idx)) {
    std::cout << absl::StrFormat("[RESUME] Resuming from checkpoint: %zu / %zu roots already completed!\n",
                                 start_root_idx, sampled_roots.size());
  }

  if (!base_receipt_json.empty()) {
    size_t ref_v_idx = variants.size();
    for (size_t v = 0; v < variants.size(); ++v) {
      if (variants[v].type == BenchmarkVariantType::kPairedReferenceOnChosenActions) {
        ref_v_idx = v;
        break;
      }
    }
    if (ref_v_idx < variants.size()) {
      int sanity_root_idx = -1;
      for (size_t r = 0; r < sampled_roots.size(); ++r) {
        if (has_base_variant[r][ref_v_idx] &&
            all_results[r][ref_v_idx].evaluations > 0 &&
            all_results[r][ref_v_idx].stage1_kept_worlds > 0) {
          sanity_root_idx = static_cast<int>(r);
          break;
        }
      }
      if (sanity_root_idx >= 0) {
        const auto& root = sampled_roots[sanity_root_idx];
        const auto& cached_ref_vr = all_results[sanity_root_idx][ref_v_idx];
        std::unique_ptr<open_spiel::State> sanity_state = game->NewInitialState();
        for (int64_t a : root.action_history) {
          sanity_state->ApplyAction(a);
        }
        std::vector<Action> distinct_actions;
        for (const auto& cu : cached_ref_vr.candidate_utilities) {
          distinct_actions.push_back(cu.first);
        }
        auto s_res = PairedPriorSortedRolloutSearchDecision(
            *sanity_state, root.search_seat, /*turn_tree=*/nullptr, evaluators,
            distinct_actions, /*rollouts_per_action=*/2048, min_override_margin,
            /*min_t_stat=*/1.0, cached_ref_vr.decision_seed, SearchRolloutCrnMode::kSequential,
            /*sample_opponent_leader_draft=*/false, /*max_time_ms=*/900000.0,
            /*max_inflight_worlds=*/128);
        double max_u_diff = std::abs(s_res.raw_mean_utility - cached_ref_vr.raw_mean_utility);
        SPIEL_CHECK_EQ(s_res.candidate_utilities.size(), cached_ref_vr.candidate_utilities.size());
        for (size_t i = 0; i < s_res.candidate_utilities.size(); ++i) {
          SPIEL_CHECK_EQ(s_res.candidate_utilities[i].first, cached_ref_vr.candidate_utilities[i].first);
          double diff = std::abs(s_res.candidate_utilities[i].second - cached_ref_vr.candidate_utilities[i].second);
          if (diff > max_u_diff) max_u_diff = diff;
        }
        if (max_u_diff > 1e-5) {
          open_spiel::SpielFatalError(absl::StrFormat(
              "[SANITY-CHECK] Ref2048 reproduction failed on root %d: max_u_diff=%e > 1e-5",
              root.root_id, max_u_diff));
        }
        std::cout << absl::StrFormat(
            "[SANITY-CHECK] Verified Ref2048 reproduction on root %d (seed=%llu): max_u_diff=%e\n",
            root.root_id, static_cast<unsigned long long>(cached_ref_vr.decision_seed), max_u_diff);
      }
    }
  }

  auto get_effective_top_k = [](const VariantConfig& vc, const SampledRoot& rt) -> int {
    if (vc.type != BenchmarkVariantType::kSuccessiveHalvingFixedTop5) return 0;
    if (vc.all_buy && rt.decision_type.rfind("Purchase", 0) == 0) {
      return static_cast<int>(rt.legal_actions.size());
    }
    return vc.fixed_top_k;
  };
  bool verified_k_equivalence = false;

  for (size_t r_idx = start_root_idx; r_idx < sampled_roots.size(); ++r_idx) {
    const auto& root = sampled_roots[r_idx];

    // Replay state
    std::unique_ptr<open_spiel::State> state = game->NewInitialState();
    for (int64_t a : root.action_history) {
      state->ApplyAction(a);
    }

    if (r_idx % 25 == 0 || r_idx + 1 == sampled_roots.size()) {
      std::cout << absl::StrFormat("  Processing root %zu / %zu (Game %d, Rnd %d, %s)...\n",
                                   r_idx + 1, sampled_roots.size(), root.game_id, root.round, root.decision_type);
    }

    for (size_t v_idx = 0; v_idx < variants.size(); ++v_idx) {
      const auto& v_cfg = variants[v_idx];

      if (v_cfg.type != BenchmarkVariantType::kPairedReferenceOnChosenActions) {
        if (has_base_variant[r_idx][v_idx]) {
          continue;
        }
      }

      uint64_t dec_seed = (v_cfg.type == BenchmarkVariantType::kPairedReferenceOnChosenActions &&
                           has_base_variant[r_idx][v_idx] &&
                           all_results[r_idx][v_idx].decision_seed != 0)
          ? all_results[r_idx][v_idx].decision_seed
          : dune_seed::DeriveSeed(master_seed, v_cfg.seed_domain, root.root_id);

      if (v_cfg.type == BenchmarkVariantType::kPairedReferenceOnChosenActions &&
          has_base_variant[r_idx][v_idx]) {
        const Action raw_a = all_results[r_idx][0].chosen_action;
        std::vector<Action> distinct_actions;
        distinct_actions.push_back(raw_a);
        for (size_t prev_v = 0; prev_v < v_idx; ++prev_v) {
          Action a = all_results[r_idx][prev_v].chosen_action;
          if (std::find(distinct_actions.begin(), distinct_actions.end(), a) == distinct_actions.end()) {
            distinct_actions.push_back(a);
          }
        }

        auto cached_ref = all_results[r_idx][v_idx];
        // Case 1: distinct_actions.size() <= 1
        if (distinct_actions.size() <= 1) {
          continue;
        }

        // Case 2: distinct_actions.size() > 1 AND (cached_ref.evaluations == 0 || cached_ref.stage1_kept_worlds == 0)
        if (cached_ref.evaluations == 0 || cached_ref.stage1_kept_worlds == 0) {
          uint64_t evals_before = open_spiel::GetTotalNetworkEvaluations();
          auto t_start = std::chrono::steady_clock::now();

          auto s_res = PairedPriorSortedRolloutSearchDecision(
              *state, root.search_seat, /*turn_tree=*/nullptr, evaluators,
              distinct_actions, /*rollouts_per_action=*/v_cfg.rollouts, min_override_margin,
              /*min_t_stat=*/1.0, dec_seed, SearchRolloutCrnMode::kSequential,
              /*sample_opponent_leader_draft=*/false, /*max_time_ms=*/900000.0,
              /*max_inflight_worlds=*/128);

          auto t_end = std::chrono::steady_clock::now();
          uint64_t evals_after = open_spiel::GetTotalNetworkEvaluations();
          double dur_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();

          VariantResult vr;
          vr.decision_seed = dec_seed;
          vr.chosen_action = s_res.action;
          vr.overridden = s_res.overridden;
          vr.raw_mean_utility = s_res.raw_mean_utility;
          vr.chosen_mean_utility = s_res.chosen_mean_utility;
          for (const auto& cu : s_res.candidate_utilities) {
            vr.candidate_utilities.push_back({cu.first, cu.second});
          }
          for (const auto& cru : s_res.candidate_rollout_utilities) {
            vr.candidate_rollout_utilities.push_back({cru.first, cru.second});
          }
          vr.stage1_planned_worlds = s_res.stage1_planned_worlds;
          vr.stage1_kept_worlds = s_res.stage1_kept_worlds;
          vr.stage2_planned_worlds = s_res.stage2_planned_worlds;
          vr.stage2_kept_worlds = s_res.stage2_kept_worlds;
          vr.deadline_bound = (s_res.stage1_kept_worlds < s_res.stage1_planned_worlds);

          const std::vector<double>& raw_rollouts = s_res.candidate_rollout_utilities.empty()
              ? std::vector<double>{}
              : s_res.candidate_rollout_utilities.front().second;
          for (const auto& cru : s_res.candidate_rollout_utilities) {
            Action cand_a = cru.first;
            if (cand_a == raw_a) {
              vr.candidate_pair_sd_vs_raw.push_back({static_cast<int>(cand_a), 0.0});
            } else {
              const auto& cand_rollouts = cru.second;
              const size_t n = std::min(raw_rollouts.size(), cand_rollouts.size());
              if (n < 2) {
                vr.candidate_pair_sd_vs_raw.push_back({static_cast<int>(cand_a), 0.0});
              } else {
                double sum_diff = 0.0;
                for (size_t i = 0; i < n; ++i) {
                  sum_diff += (cand_rollouts[i] - raw_rollouts[i]);
                }
                const double mean_diff = sum_diff / static_cast<double>(n);
                double sum_sq = 0.0;
                for (size_t i = 0; i < n; ++i) {
                  double d = (cand_rollouts[i] - raw_rollouts[i]) - mean_diff;
                  sum_sq += d * d;
                }
                const double var = sum_sq / static_cast<double>(n - 1);
                const double sd = std::sqrt(std::max(0.0, var));
                vr.candidate_pair_sd_vs_raw.push_back({static_cast<int>(cand_a), sd});
              }
            }
          }
          vr.evaluations = evals_after - evals_before;
          vr.duration_ms = dur_ms;
          all_results[r_idx][v_idx] = std::move(vr);
          std::cout << absl::StrFormat(
              "[REF2048-EXTEND] Root %d: populated previously-empty Ref2048 for override actions\n",
              root.root_id);
          continue;
        }

        // Case 3: distinct_actions.size() > 1 AND cached_ref.evaluations > 0
        std::vector<Action> missing_actions;
        for (Action a : distinct_actions) {
          bool found = false;
          for (const auto& cu : cached_ref.candidate_utilities) {
            if (cu.first == static_cast<int>(a)) {
              found = true;
              break;
            }
          }
          if (!found) {
            missing_actions.push_back(a);
          }
        }

        if (missing_actions.empty()) {
          continue;
        }

        std::vector<Action> eval_actions = {raw_a};
        for (Action ma : missing_actions) {
          eval_actions.push_back(ma);
        }

        uint64_t evals_before = open_spiel::GetTotalNetworkEvaluations();
        auto t_start = std::chrono::steady_clock::now();

        auto s_res = PairedPriorSortedRolloutSearchDecision(
            *state, root.search_seat, /*turn_tree=*/nullptr, evaluators,
            eval_actions, /*rollouts_per_action=*/v_cfg.rollouts, min_override_margin,
            /*min_t_stat=*/1.0, dec_seed, SearchRolloutCrnMode::kSequential,
            /*sample_opponent_leader_draft=*/false, /*max_time_ms=*/900000.0,
            /*max_inflight_worlds=*/128);

        auto t_end = std::chrono::steady_clock::now();
        uint64_t evals_after = open_spiel::GetTotalNetworkEvaluations();
        double dur_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();

        double raw_diff = std::abs(s_res.raw_mean_utility - cached_ref.raw_mean_utility);
        if (raw_diff > 1e-5) {
          open_spiel::SpielFatalError(absl::StrFormat(
              "[REF2048-EXTEND] Raw mean utility diff %e > 1e-5 on root %d during extension",
              raw_diff, root.root_id));
        }

        const std::vector<double>& raw_rollouts = s_res.candidate_rollout_utilities.empty()
            ? std::vector<double>{}
            : s_res.candidate_rollout_utilities.front().second;

        for (const auto& cru : s_res.candidate_rollout_utilities) {
          Action cand_a = cru.first;
          if (cand_a == raw_a) continue;

          for (const auto& cu : s_res.candidate_utilities) {
            if (cu.first == cand_a) {
              cached_ref.candidate_utilities.push_back({static_cast<int>(cand_a), cu.second});
              break;
            }
          }

          cached_ref.candidate_rollout_utilities.push_back({static_cast<int>(cand_a), cru.second});

          const auto& cand_rollouts = cru.second;
          const size_t n = std::min(raw_rollouts.size(), cand_rollouts.size());
          if (n < 2) {
            cached_ref.candidate_pair_sd_vs_raw.push_back({static_cast<int>(cand_a), 0.0});
          } else {
            double sum_diff = 0.0;
            for (size_t i = 0; i < n; ++i) {
              sum_diff += (cand_rollouts[i] - raw_rollouts[i]);
            }
            const double mean_diff = sum_diff / static_cast<double>(n);
            double sum_sq = 0.0;
            for (size_t i = 0; i < n; ++i) {
              double d = (cand_rollouts[i] - raw_rollouts[i]) - mean_diff;
              sum_sq += d * d;
            }
            const double var = sum_sq / static_cast<double>(n - 1);
            const double sd = std::sqrt(std::max(0.0, var));
            cached_ref.candidate_pair_sd_vs_raw.push_back({static_cast<int>(cand_a), sd});
          }
        }

        const size_t K = cached_ref.candidate_rollout_utilities.size();
        const auto& base_raw_rollouts = cached_ref.candidate_rollout_utilities.front().second;
        Action best_action = raw_a;
        double best_mean_u = cached_ref.raw_mean_utility;
        bool overridden = false;

        for (size_t c_idx = 1; c_idx < K; ++c_idx) {
          Action cand_a = static_cast<Action>(cached_ref.candidate_rollout_utilities[c_idx].first);
          const auto& cand_rollouts = cached_ref.candidate_rollout_utilities[c_idx].second;
          double sum_u = 0.0;
          const int n_c = static_cast<int>(cand_rollouts.size());
          for (int r = 0; r < n_c; ++r) {
            sum_u += cand_rollouts[r];
          }
          double mean_u = (n_c > 0) ? (sum_u / n_c) : 0.0;
          double mean_d = mean_u - cached_ref.raw_mean_utility;

          const size_t n_pairs = std::min(base_raw_rollouts.size(), cand_rollouts.size());
          double sum_sq_diff = 0.0;
          for (size_t r = 0; r < n_pairs; ++r) {
            double d_r = cand_rollouts[r] - base_raw_rollouts[r];
            double diff = d_r - mean_d;
            sum_sq_diff += diff * diff;
          }
          double var_d = (n_pairs > 1) ? (sum_sq_diff / (n_pairs - 1)) : 0.0;
          double se_d = (n_pairs > 1) ? std::sqrt(var_d / n_pairs) : 0.0;
          const double t_stat = (n_pairs >= 2) ? ((se_d > 1e-9) ? (mean_d / se_d) : (mean_d > 0.0 ? 100.0 : 0.0)) : 0.0;

          if (mean_d > min_override_margin && t_stat >= 1.0 && mean_u > best_mean_u) {
            best_mean_u = mean_u;
            best_action = cand_a;
            overridden = true;
          }
        }

        cached_ref.chosen_action = best_action;
        cached_ref.chosen_mean_utility = best_mean_u;
        cached_ref.overridden = overridden;
        cached_ref.evaluations += (evals_after - evals_before);
        cached_ref.duration_ms += dur_ms;

        all_results[r_idx][v_idx] = cached_ref;
        std::cout << absl::StrFormat(
            "[REF2048-EXTEND] Root %d: appended %zu new action(s) to populated Ref2048 (raw_u diff=%e)\n",
            root.root_id, missing_actions.size(), raw_diff);
        continue;
      }

      uint64_t evals_before = open_spiel::GetTotalNetworkEvaluations();
      auto t_start = std::chrono::steady_clock::now();

      VariantResult vr;
      vr.decision_seed = dec_seed;

      if (v_cfg.type == BenchmarkVariantType::kRaw) {
        const auto legal = state->LegalActions();
        const ActionsAndProbs raw_prior = evaluators.front()->Prior(*state);
        const auto sorted_by_prior = GetSortedCandidatesByPrior(raw_prior, legal);
        Action raw_a = sorted_by_prior.empty() ? (legal.empty() ? kInvalidAction : legal.front()) : sorted_by_prior.front().first;
        vr.chosen_action = raw_a;
        vr.overridden = false;
        vr.raw_mean_utility = 0.0;
        vr.chosen_mean_utility = 0.0;
      } else if (v_cfg.type == BenchmarkVariantType::kSuccessiveHalvingDynamic ||
                 v_cfg.type == BenchmarkVariantType::kSuccessiveHalvingFixedTop5) {
        int eff_top_k = 0;
        if (v_cfg.type == BenchmarkVariantType::kSuccessiveHalvingFixedTop5) {
          eff_top_k = get_effective_top_k(v_cfg, root);
          const int eff_K = std::min<int>(eff_top_k, static_cast<int>(root.legal_actions.size()));

          int matched_prev_v = -1;
          for (size_t prev_v = 0; prev_v < v_idx; ++prev_v) {
            if (variants[prev_v].type == BenchmarkVariantType::kSuccessiveHalvingFixedTop5 &&
                variants[prev_v].reference_rollouts_per_action == v_cfg.reference_rollouts_per_action &&
                variants[prev_v].seed_domain == v_cfg.seed_domain &&
                std::min<int>(get_effective_top_k(variants[prev_v], root), static_cast<int>(root.legal_actions.size())) == eff_K &&
                all_results[r_idx][prev_v].stage1_planned_worlds > 0) {
              matched_prev_v = static_cast<int>(prev_v);
              break;
            }
          }

          if (matched_prev_v >= 0) {
            if (!verified_k_equivalence) {
              auto s_res = SuccessiveHalvingRolloutSearchDecision(
                  *state, root.search_seat, /*turn_tree=*/nullptr, evaluators,
                  v_cfg.reference_rollouts_per_action, min_override_margin,
                  dec_seed, SearchRolloutCrnMode::kSequential,
                  /*unbiased_stage2_gate=*/false, /*max_time_ms=*/-1.0,
                  eff_top_k);
              if (s_res.action != all_results[r_idx][matched_prev_v].chosen_action ||
                  std::abs(s_res.chosen_mean_utility - all_results[r_idx][matched_prev_v].chosen_mean_utility) > 1e-6) {
                open_spiel::SpielFatalError(absl::StrFormat(
                    "[SANITY-CHECK] K-equivalence mismatch between %s and %s on root %d: action %d vs %d, utility %f vs %f",
                    v_cfg.name, variants[matched_prev_v].name, root.root_id,
                    s_res.action, all_results[r_idx][matched_prev_v].chosen_action,
                    s_res.chosen_mean_utility, all_results[r_idx][matched_prev_v].chosen_mean_utility));
              }
              std::cout << absl::StrFormat(
                  "[SANITY-CHECK] Verified exact top-k equivalence between %s and %s on root %d (eff_K=%d)\n",
                  v_cfg.name, variants[matched_prev_v].name, root.root_id, eff_K);
              verified_k_equivalence = true;
            }
            all_results[r_idx][v_idx] = all_results[r_idx][matched_prev_v];
            continue;
          }
        }

        auto s_res = SuccessiveHalvingRolloutSearchDecision(
            *state, root.search_seat, /*turn_tree=*/nullptr, evaluators,
            v_cfg.reference_rollouts_per_action, min_override_margin,
            dec_seed, SearchRolloutCrnMode::kSequential,
            /*unbiased_stage2_gate=*/false, /*max_time_ms=*/-1.0,
            eff_top_k);
        vr.chosen_action = s_res.action;
        vr.overridden = s_res.overridden;
        vr.raw_mean_utility = s_res.raw_mean_utility;
        vr.chosen_mean_utility = s_res.chosen_mean_utility;
        for (const auto& cu : s_res.candidate_utilities) {
          vr.candidate_utilities.push_back({cu.first, cu.second});
        }
        for (const auto& cru : s_res.candidate_rollout_utilities) {
          vr.candidate_rollout_utilities.push_back({cru.first, cru.second});
        }
        vr.stage1_planned_worlds = s_res.stage1_planned_worlds;
        vr.stage1_kept_worlds = s_res.stage1_kept_worlds;
        vr.stage2_planned_worlds = s_res.stage2_planned_worlds;
        vr.stage2_kept_worlds = s_res.stage2_kept_worlds;
        vr.deadline_bound = false;
      } else if (v_cfg.type == BenchmarkVariantType::kPairedReferenceOnChosenActions) {
        const Action raw_a = all_results[r_idx][0].chosen_action;
        std::vector<Action> distinct_actions;
        distinct_actions.push_back(raw_a);
        for (size_t prev_v = 0; prev_v < v_idx; ++prev_v) {
          Action a = all_results[r_idx][prev_v].chosen_action;
          if (std::find(distinct_actions.begin(), distinct_actions.end(), a) == distinct_actions.end()) {
            distinct_actions.push_back(a);
          }
        }

        if (distinct_actions.size() <= 1) {
          vr.chosen_action = raw_a;
          vr.overridden = false;
          vr.raw_mean_utility = 0.0;
          vr.chosen_mean_utility = 0.0;
          vr.candidate_utilities.push_back({static_cast<int>(raw_a), 0.0});
          vr.candidate_rollout_utilities.push_back({static_cast<int>(raw_a), {}});
          vr.candidate_pair_sd_vs_raw.push_back({static_cast<int>(raw_a), 0.0});
          vr.stage1_planned_worlds = 0;
          vr.stage1_kept_worlds = 0;
          vr.stage2_planned_worlds = 0;
          vr.stage2_kept_worlds = 0;
          vr.deadline_bound = false;
        } else {
          auto s_res = PairedPriorSortedRolloutSearchDecision(
              *state, root.search_seat, /*turn_tree=*/nullptr, evaluators,
              distinct_actions, /*rollouts_per_action=*/v_cfg.rollouts, min_override_margin,
              /*min_t_stat=*/1.0, dec_seed, SearchRolloutCrnMode::kSequential,
              /*sample_opponent_leader_draft=*/false, /*max_time_ms=*/900000.0,
              /*max_inflight_worlds=*/128);
          vr.chosen_action = s_res.action;
          vr.overridden = s_res.overridden;
          vr.raw_mean_utility = s_res.raw_mean_utility;
          vr.chosen_mean_utility = s_res.chosen_mean_utility;
          for (const auto& cu : s_res.candidate_utilities) {
            vr.candidate_utilities.push_back({cu.first, cu.second});
          }
          for (const auto& cru : s_res.candidate_rollout_utilities) {
            vr.candidate_rollout_utilities.push_back({cru.first, cru.second});
          }
          vr.stage1_planned_worlds = s_res.stage1_planned_worlds;
          vr.stage1_kept_worlds = s_res.stage1_kept_worlds;
          vr.stage2_planned_worlds = s_res.stage2_planned_worlds;
          vr.stage2_kept_worlds = s_res.stage2_kept_worlds;
          vr.deadline_bound = (s_res.stage1_kept_worlds < s_res.stage1_planned_worlds);

          const std::vector<double>& raw_rollouts = s_res.candidate_rollout_utilities.empty()
              ? std::vector<double>{}
              : s_res.candidate_rollout_utilities.front().second;
          for (const auto& cru : s_res.candidate_rollout_utilities) {
            Action cand_a = cru.first;
            if (cand_a == raw_a) {
              vr.candidate_pair_sd_vs_raw.push_back({static_cast<int>(cand_a), 0.0});
            } else {
              const auto& cand_rollouts = cru.second;
              const size_t n = std::min(raw_rollouts.size(), cand_rollouts.size());
              if (n < 2) {
                vr.candidate_pair_sd_vs_raw.push_back({static_cast<int>(cand_a), 0.0});
              } else {
                double sum_diff = 0.0;
                for (size_t i = 0; i < n; ++i) {
                  sum_diff += (cand_rollouts[i] - raw_rollouts[i]);
                }
                const double mean_diff = sum_diff / static_cast<double>(n);
                double sum_sq = 0.0;
                for (size_t i = 0; i < n; ++i) {
                  double d = (cand_rollouts[i] - raw_rollouts[i]) - mean_diff;
                  sum_sq += d * d;
                }
                const double var = sum_sq / static_cast<double>(n - 1);
                const double sd = std::sqrt(std::max(0.0, var));
                vr.candidate_pair_sd_vs_raw.push_back({static_cast<int>(cand_a), sd});
              }
            }
          }
        }
      } else {
        const int eff_top_k = v_cfg.name == "Ref512" ? reference_top_k : top_k;
        auto s_res = CompoundRolloutSearchDecision(
            *state, root.search_seat, /*turn_tree=*/nullptr, evaluators,
            eff_top_k, v_cfg.rollouts, min_override_margin,
            TreeReuseMode::kTargetFill, dec_seed,
            v_cfg.pairing, v_cfg.truncation);
        vr.chosen_action = s_res.action;
        vr.overridden = s_res.overridden;
        vr.raw_mean_utility = s_res.raw_mean_utility;
        vr.chosen_mean_utility = s_res.chosen_mean_utility;
        vr.stage1_planned_worlds = v_cfg.rollouts;
        vr.stage1_kept_worlds = v_cfg.rollouts;
        vr.stage2_planned_worlds = 0;
        vr.stage2_kept_worlds = 0;
        vr.deadline_bound = false;
        for (const auto& cu : s_res.candidate_utilities) {
          vr.candidate_utilities.push_back({cu.first, cu.second});
        }
        for (const auto& cru : s_res.candidate_rollout_utilities) {
          vr.candidate_rollout_utilities.push_back({cru.first, cru.second});
        }
      }

      auto t_end = std::chrono::steady_clock::now();
      uint64_t evals_after = open_spiel::GetTotalNetworkEvaluations();
      double dur_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();

      vr.evaluations = evals_after - evals_before;
      vr.duration_ms = dur_ms;
      all_results[r_idx][v_idx] = std::move(vr);
    }

    if ((r_idx + 1) % 10 == 0 || r_idx + 1 == sampled_roots.size()) {
      open_spiel::json::Object temp_summary;
      SaveBenchmarkReceipt(checkpoint_path, temp_summary, sampled_roots, variants, all_results, {}, r_idx + 1);
    }
  }

  // If leaf_check_only, compute and report V2_16_Paired vs Ref512 correlation and exit
  if (absl::GetFlag(FLAGS_leaf_check_only)) {
    std::vector<double> critic_x;
    std::vector<double> critic_y;
    for (size_t r_idx = 0; r_idx < sampled_roots.size(); ++r_idx) {
      const auto& root = sampled_roots[r_idx];
      Action raw_a = static_cast<Action>(root.raw_action);
      const auto& v2_res = all_results[r_idx][0];
      const auto& ref_res = all_results[r_idx][1];

      std::unordered_map<Action, double> v2_cand_utils;
      for (const auto& cu : v2_res.candidate_utilities) v2_cand_utils[cu.first] = cu.second;
      double q_v2_raw = v2_cand_utils.count(raw_a) ? v2_cand_utils[raw_a] : v2_res.raw_mean_utility;

      std::unordered_map<Action, double> ref_cand_utils;
      for (const auto& cu : ref_res.candidate_utilities) ref_cand_utils[cu.first] = cu.second;
      double q_ref_raw = ref_cand_utils.count(raw_a) ? ref_cand_utils[raw_a] : ref_res.raw_mean_utility;

      for (const auto& cu : v2_cand_utils) {
        Action c = cu.first;
        if (c == raw_a || !ref_cand_utils.count(c)) continue;
        critic_x.push_back(cu.second - q_v2_raw);
        critic_y.push_back(ref_cand_utils[c] - q_ref_raw);
      }
    }
    double v2_ref_corr = 0.0;
    if (critic_x.size() >= 2) {
      double mx = std::accumulate(critic_x.begin(), critic_x.end(), 0.0) / critic_x.size();
      double my = std::accumulate(critic_y.begin(), critic_y.end(), 0.0) / critic_y.size();
      double cov = 0.0, vx = 0.0, vy = 0.0;
      for (size_t i = 0; i < critic_x.size(); ++i) {
        cov += (critic_x[i] - mx) * (critic_y[i] - my);
        vx += (critic_x[i] - mx) * (critic_x[i] - mx);
        vy += (critic_y[i] - my) * (critic_y[i] - my);
      }
      double denom = std::sqrt(vx * vy);
      v2_ref_corr = (denom > 1e-12) ? (cov / denom) : 0.0;
    }
    std::cout << "\n====================================================================================================\n";
    std::cout << absl::StrFormat(
        "LEAF CHECK (N=%zu roots, N=%zu candidate pairs): V2_16_Paired vs Ref512 correlation = %.4f (U28700 baseline: ~0.8)\n",
        sampled_roots.size(), critic_x.size(), v2_ref_corr);
    std::cout << "====================================================================================================\n";
    open_spiel::json::Object leaf_out_json;
    leaf_out_json["leaf_check_v2_ref_correlation"] = open_spiel::json::Value(v2_ref_corr);
    leaf_out_json["leaf_check_candidate_pairs"] = open_spiel::json::Value(static_cast<int64_t>(critic_x.size()));
    SaveBenchmarkReceipt(output_path, leaf_out_json, sampled_roots, variants, all_results, {}, sampled_roots.size());
    std::remove(checkpoint_path.c_str());
    return 0;
  }

  const size_t ref_idx = variants.size() - 1;  // Reference is the last variant
  SPIEL_CHECK_TRUE(variants[ref_idx].name == "Ref512" || variants[ref_idx].name == "Ref2048");

  std::cout << absl::StrFormat("[BENCHMARK] All roots evaluated! Analyzing true gain against %s...\n\n",
                               variants[ref_idx].name);

  // Summarize metrics
  double total_weight_sum = 0.0;
  for (const auto& r : sampled_roots) total_weight_sum += r.stratum_weight;

  // MUST FIX 4(b): Summary table drops / labels reference baseline and reports Evals/Decision
  std::cout << "====================================================================================================\n";
  std::cout << "| Candidate Variant | Rollouts | Pairing  | Mode      | Agreement (%) | Mean True Gain* | Evals/Dec |\n";
  std::cout << "| :---              | :---     | :---     | :---      | :---          | :---            | :---      |\n";

  open_spiel::json::Object root_out_json;
  open_spiel::json::Array variant_summaries;

  for (size_t v_idx = 0; v_idx < ref_idx; ++v_idx) {
    const auto& v_cfg = variants[v_idx];
    double weighted_agree = 0.0;
    double weighted_true_gain = 0.0;
    uint64_t total_evals = 0;
    double total_time_ms = 0.0;
    int override_count = 0;

    for (size_t r_idx = 0; r_idx < sampled_roots.size(); ++r_idx) {
      const auto& root = sampled_roots[r_idx];
      const auto& vr = all_results[r_idx][v_idx];
      const auto& ref_vr = all_results[r_idx][ref_idx];
      double w = root.stratum_weight;

      total_evals += vr.evaluations;
      total_time_ms += vr.duration_ms;
      if (vr.overridden) override_count++;

      // Agreement with reference action
      if (vr.chosen_action == ref_vr.chosen_action) {
        weighted_agree += w;
      }

      // MUST FIX 4(b): Fatal check if variant's chosen action is not in reference candidate_utilities
      bool found_chosen = false;
      double ref_chosen_u = 0.0;
      double ref_raw_u = ref_vr.raw_mean_utility;
      for (const auto& r_cu : ref_vr.candidate_utilities) {
        if (r_cu.first == vr.chosen_action) {
          ref_chosen_u = r_cu.second;
          found_chosen = true;
          break;
        }
      }
      if (!found_chosen) {
        open_spiel::SpielFatalError(absl::StrFormat(
            "Fatal: Variant %s chosen action %d on root %d not found in %s candidate_utilities! "
            "%s must evaluate all top candidate actions.",
            v_cfg.name, vr.chosen_action, root.root_id,
            variants[ref_idx].name, variants[ref_idx].name));
      }
      double true_gain = ref_chosen_u - ref_raw_u;
      weighted_true_gain += w * true_gain;
    }

    double agree_pct = 100.0 * weighted_agree / total_weight_sum;
    double mean_gain = weighted_true_gain / total_weight_sum;
    double avg_evals = static_cast<double>(total_evals) / sampled_roots.size();
    double avg_time_ms = total_time_ms / sampled_roots.size();

    std::string pairing_str = (v_cfg.pairing == SearchPairingMode::kPaired) ? "Paired" : "Unpaired";
    std::string mode_str = (v_cfg.truncation == SearchTruncationMode::kTerminal) ? "Terminal" : "Truncated";

    std::cout << absl::StrFormat(
        "| %-17s | %-8d | %-8s | %-9s | %12.1f%% | %14.3f* | %9.1f |\n",
        v_cfg.name, v_cfg.rollouts, pairing_str, mode_str, agree_pct, mean_gain, avg_evals);

    open_spiel::json::Object v_sum_obj;
    v_sum_obj["name"] = open_spiel::json::Value(v_cfg.name);
    v_sum_obj["rollouts"] = open_spiel::json::Value(static_cast<int64_t>(v_cfg.rollouts));
    v_sum_obj["pairing"] = open_spiel::json::Value(pairing_str);
    v_sum_obj["truncation"] = open_spiel::json::Value(mode_str);
    v_sum_obj["agreement_pct"] = open_spiel::json::Value(agree_pct);
    v_sum_obj["mean_true_gain"] = open_spiel::json::Value(mean_gain);
    v_sum_obj["avg_evaluations"] = open_spiel::json::Value(avg_evals);
    v_sum_obj["avg_time_ms"] = open_spiel::json::Value(avg_time_ms);
    v_sum_obj["overrides"] = open_spiel::json::Value(static_cast<int64_t>(override_count));
    variant_summaries.push_back(open_spiel::json::Value(v_sum_obj));
  }

  // Reference statistics reported separately
  uint64_t ref_total_evals = 0;
  double ref_total_time_ms = 0.0;
  for (size_t r_idx = 0; r_idx < sampled_roots.size(); ++r_idx) {
    ref_total_evals += all_results[r_idx][ref_idx].evaluations;
    ref_total_time_ms += all_results[r_idx][ref_idx].duration_ms;
  }
  double ref_avg_evals = static_cast<double>(ref_total_evals) / sampled_roots.size();
  double ref_avg_time = ref_total_time_ms / sampled_roots.size();

  std::cout << "====================================================================================================\n";
  std::cout << absl::StrFormat(
      "[Reference Baseline] %s: %d rollouts | Paired | Terminal | Avg Evals/Dec: %.1f | Avg Time: %.1f ms\n",
      variants[ref_idx].name, variants[ref_idx].rollouts, ref_avg_evals, ref_avg_time);
  std::cout << "====================================================================================================\n";
  std::cout << absl::StrFormat("* True Gain is measured against independent %d-rollout paired reference utilities.\n\n",
                               variants[ref_idx].rollouts);

  root_out_json["variant_summaries"] = open_spiel::json::Value(variant_summaries);

  // Save complete main sweep immediately so results are preserved before recheck begins
  SaveBenchmarkReceipt(output_path, root_out_json, sampled_roots, variants, all_results, {}, sampled_roots.size());
  std::cout << "[RECEIPT] Saved complete main sweep (all variants across all roots) to " << output_path << "\n";


  std::vector<VariantResult> recheck_1024_results;
  if (!absl::GetFlag(FLAGS_skip_recheck_1024) && !absl::GetFlag(FLAGS_sh_budget_check)) {
    // 1,024-Rollout Re-check Pass on Disagreement Roots
    std::vector<int> disagreement_root_indices;
    for (size_t r_idx = 0; r_idx < sampled_roots.size(); ++r_idx) {
      const auto& ref_vr = all_results[r_idx][ref_idx];
      bool has_disagreement = false;
      for (size_t v_idx = 0; v_idx < ref_idx; ++v_idx) {
        if (all_results[r_idx][v_idx].chosen_action != ref_vr.chosen_action) {
          has_disagreement = true;
          break;
        }
      }
      if (has_disagreement) {
        disagreement_root_indices.push_back(static_cast<int>(r_idx));
      }
    }

    std::cout << absl::StrFormat(
        "[RE-CHECK] Identified %zu disagreement roots. Running 1,024-rollout paired re-check (domain 0x050B)...\n",
        disagreement_root_indices.size());

    recheck_1024_results.resize(sampled_roots.size());
    for (size_t d_idx = 0; d_idx < disagreement_root_indices.size(); ++d_idx) {
      int r_idx = disagreement_root_indices[d_idx];
      const auto& root = sampled_roots[r_idx];
      std::unique_ptr<open_spiel::State> state = game->NewInitialState();
      for (int64_t a : root.action_history) {
        state->ApplyAction(a);
      }

      if (d_idx % 10 == 0 || d_idx + 1 == disagreement_root_indices.size()) {
        std::cout << absl::StrFormat("  Re-checking disagreement root %zu / %zu (Root %d, Game %d)...\n",
                                     d_idx + 1, disagreement_root_indices.size(), root.root_id, root.game_id);
      }

      uint64_t dec_seed = dune_seed::DeriveSeed(master_seed, benchmark_domains::kDomainRecheck1024, root.root_id);
      uint64_t evals_before = open_spiel::GetTotalNetworkEvaluations();
      auto t_start = std::chrono::steady_clock::now();
      auto s_res = CompoundRolloutSearchDecision(
          *state, root.search_seat, /*turn_tree=*/nullptr, evaluators,
          reference_top_k, 1024, min_override_margin,
          TreeReuseMode::kTargetFill, dec_seed,
          SearchPairingMode::kPaired, SearchTruncationMode::kTerminal);
      auto t_end = std::chrono::steady_clock::now();
      uint64_t evals_after = open_spiel::GetTotalNetworkEvaluations();

      VariantResult vr;
      vr.chosen_action = s_res.action;
      vr.overridden = s_res.overridden;
      vr.raw_mean_utility = s_res.raw_mean_utility;
      vr.chosen_mean_utility = s_res.chosen_mean_utility;
      vr.decision_seed = dec_seed;
      vr.evaluations = evals_after - evals_before;
      vr.duration_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
      for (const auto& cu : s_res.candidate_utilities) {
        vr.candidate_utilities.push_back({cu.first, cu.second});
      }
      for (const auto& cru : s_res.candidate_rollout_utilities) {
        vr.candidate_rollout_utilities.push_back({cru.first, cru.second});
      }
      recheck_1024_results[r_idx] = std::move(vr);

      if ((d_idx + 1) % 5 == 0 || d_idx + 1 == disagreement_root_indices.size()) {
        SaveBenchmarkReceipt(output_path, root_out_json, sampled_roots, variants, all_results, recheck_1024_results, sampled_roots.size());
      }
    }
    std::cout << "[RE-CHECK] 1,024-rollout re-check complete!\n\n";
  } else {
    std::cout << "[RE-CHECK] Skipping 1,024-rollout re-check pass (--skip_recheck_1024=true).\n\n";
  }

  SaveBenchmarkReceipt(output_path, root_out_json, sampled_roots, variants, all_results, recheck_1024_results, sampled_roots.size());
  std::cout << "[RECEIPT] Saved finalized benchmark receipt JSON to " << output_path << "\n";
  std::remove(checkpoint_path.c_str());

  return 0;
}
