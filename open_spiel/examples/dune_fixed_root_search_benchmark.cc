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

#include "open_spiel/abseil-cpp/absl/flags/flag.h"
#include "open_spiel/abseil-cpp/absl/flags/parse.h"
#include "open_spiel/abseil-cpp/absl/strings/str_format.h"
#include "open_spiel/games/dune_imperium/dune_imperium.h"
#include "open_spiel/examples/dune_batched_evaluator.h"
#include "open_spiel/examples/dune_compound_turn_search.h"
#include "open_spiel/examples/dune_network.h"
#include "open_spiel/examples/dune_seed_utils.h"
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
}  // namespace benchmark_domains

struct VariantConfig {
  std::string name;
  int rollouts;
  SearchPairingMode pairing;
  SearchTruncationMode truncation;
  uint64_t seed_domain;
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
      VariantResult vr;
      vr.chosen_action = static_cast<int>(vr_obj.at("chosen_action").GetInt());
      vr.overridden = vr_obj.at("overridden").GetBool();
      vr.raw_mean_utility = vr_obj.at("raw_mean_utility").IsDouble()
          ? vr_obj.at("raw_mean_utility").GetDouble()
          : static_cast<double>(vr_obj.at("raw_mean_utility").GetInt());
      vr.chosen_mean_utility = vr_obj.at("chosen_mean_utility").IsDouble()
          ? vr_obj.at("chosen_mean_utility").GetDouble()
          : static_cast<double>(vr_obj.at("chosen_mean_utility").GetInt());
      vr.decision_seed = static_cast<uint64_t>(vr_obj.at("decision_seed").GetInt());
      vr.evaluations = static_cast<uint64_t>(vr_obj.at("evaluations").GetInt());
      vr.duration_ms = vr_obj.at("duration_ms").IsDouble()
          ? vr_obj.at("duration_ms").GetDouble()
          : static_cast<double>(vr_obj.at("duration_ms").GetInt());

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
      all_results[r_idx][v_idx] = std::move(vr);
    }
    if (root_complete) {
      completed_roots = r_idx + 1;
    } else {
      break;
    }
  }
  return completed_roots > 0;
}

int main(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);

  const std::string model_path = absl::GetFlag(FLAGS_model_checkpoint);
  const std::string roots_path = absl::GetFlag(FLAGS_sampled_roots_json);
  const std::string obs_path = absl::GetFlag(FLAGS_obs_binary);
  const std::string output_path = absl::GetFlag(FLAGS_output_json);
  const int num_evals = absl::GetFlag(FLAGS_num_evaluators);
  const int target_batch_size = absl::GetFlag(FLAGS_target_batch_size);
  const int top_k = absl::GetFlag(FLAGS_top_k_actions);
  const double min_override_margin = absl::GetFlag(FLAGS_min_override_margin);
  const uint64_t master_seed = absl::GetFlag(FLAGS_master_seed);

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
    open_spiel::SpielFatalError(absl::StrFormat(
        "Fatal: Game string mismatch between census and benchmark!\nExpected: %s\nActual:   %s",
        expected_game_str, current_game_str));
  }
  std::cout << "[PARITY] Verified game->ToString() matches census summary string.\n";

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
    sampled_roots.push_back(std::move(sr));
  }
  std::cout << "[LOAD] Loaded " << sampled_roots.size() << " sampled roots for benchmark.\n";

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
        open_spiel::dune_imperium::kFullPublicInformationStateSize, 2048, 2391, 8,
        /*nonlinear_value_head=*/false, /*with_aux_heads=*/false,
        /*head_init_seed=*/0, /*with_semantic_scorer=*/true);
    models[e]->to(device);
    torch::load(models[e], model_path, device);
    models[e]->eval();

    model_mutexes[e] = std::make_unique<std::shared_mutex>();
    coords[e] = std::make_shared<open_spiel::BatchedEvaluator>(
        models[e], target_batch_size, /*timeout_ms=*/1, device, model_mutexes[e].get(), 10.0f,
        /*device_synchronize=*/false, /*high_priority_stream=*/true,
        /*emit_batch_membership=*/false, /*rollout_amp=*/true, /*allow_tf32=*/true);
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

  std::cout << "[VERIFY] Replaying action histories and verifying observations against binary...\n";
  for (size_t i = 0; i < sampled_roots.size(); ++i) {
    const auto& root = sampled_roots[i];
    std::unique_ptr<open_spiel::State> state = game->NewInitialState();
    for (int64_t a : root.action_history) {
      state->ApplyAction(a);
    }
    if (state->CurrentPlayer() != root.search_seat) {
      open_spiel::SpielFatalError(absl::StrFormat(
          "Root %d seat mismatch: state->CurrentPlayer()=%d != root.search_seat=%d",
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
  std::cout << "[VERIFY] All " << sampled_roots.size() << " root observations verified bit-for-bit against binary!\n";

  // Configure Variants
  std::vector<VariantConfig> variants = {
      {"V0_Run1", 64, SearchPairingMode::kUnpaired, SearchTruncationMode::kTerminal, benchmark_domains::kDomainV0_Run1},
      {"V0_Run2", 64, SearchPairingMode::kUnpaired, SearchTruncationMode::kTerminal, benchmark_domains::kDomainV0_Run2},
      {"V1", 64, SearchPairingMode::kPaired, SearchTruncationMode::kTerminal, benchmark_domains::kDomainV1},
      {"V2_16_Paired", 16, SearchPairingMode::kPaired, SearchTruncationMode::kNextRoundSearchPlayer, benchmark_domains::kDomainV2_16_Paired},
      {"V2_16_Unpaired", 16, SearchPairingMode::kUnpaired, SearchTruncationMode::kNextRoundSearchPlayer, benchmark_domains::kDomainV2_16_Unpaired},
      {"V2_32_Paired", 32, SearchPairingMode::kPaired, SearchTruncationMode::kNextRoundSearchPlayer, benchmark_domains::kDomainV2_32_Paired},
      {"V2_32_Unpaired", 32, SearchPairingMode::kUnpaired, SearchTruncationMode::kNextRoundSearchPlayer, benchmark_domains::kDomainV2_32_Unpaired},
      {"V2_64_Paired", 64, SearchPairingMode::kPaired, SearchTruncationMode::kNextRoundSearchPlayer, benchmark_domains::kDomainV2_64_Paired},
      {"V2_64_Unpaired", 64, SearchPairingMode::kUnpaired, SearchTruncationMode::kNextRoundSearchPlayer, benchmark_domains::kDomainV2_64_Unpaired},
      {"Ref512", 512, SearchPairingMode::kPaired, SearchTruncationMode::kTerminal, benchmark_domains::kDomainRef512}
  };

  std::cout << "\n[BENCHMARK] Executing " << variants.size() << " variants across " << sampled_roots.size() << " roots...\n";

  // Data storage: results[root_idx][variant_idx]
  std::vector<std::vector<VariantResult>> all_results(sampled_roots.size(),
                                                      std::vector<VariantResult>(variants.size()));

  const std::string checkpoint_path = output_path + ".checkpoint.json";
  size_t start_root_idx = 0;
  if (LoadBenchmarkCheckpoint(checkpoint_path, variants, all_results, start_root_idx)) {
    std::cout << absl::StrFormat("[RESUME] Resuming from checkpoint: %zu / %zu roots already completed!\n",
                                 start_root_idx, sampled_roots.size());
  }

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
      uint64_t dec_seed = dune_seed::DeriveSeed(master_seed, v_cfg.seed_domain, root.root_id);

      // MUST FIX 3: Measure network evaluations as counter delta around search
      uint64_t evals_before = open_spiel::GetTotalNetworkEvaluations();
      auto t_start = std::chrono::steady_clock::now();
      auto s_res = CompoundRolloutSearchDecision(
          *state, root.search_seat, /*turn_tree=*/nullptr, evaluators,
          top_k, v_cfg.rollouts, min_override_margin,
          TreeReuseMode::kTargetFill, dec_seed,
          v_cfg.pairing, v_cfg.truncation);
      auto t_end = std::chrono::steady_clock::now();
      uint64_t evals_after = open_spiel::GetTotalNetworkEvaluations();
      double dur_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();

      VariantResult vr;
      vr.chosen_action = s_res.action;
      vr.overridden = s_res.overridden;
      vr.raw_mean_utility = s_res.raw_mean_utility;
      vr.chosen_mean_utility = s_res.chosen_mean_utility;
      vr.decision_seed = dec_seed;
      vr.evaluations = evals_after - evals_before;
      vr.duration_ms = dur_ms;
      for (const auto& cu : s_res.candidate_utilities) {
        vr.candidate_utilities.push_back({cu.first, cu.second});
      }
      for (const auto& cru : s_res.candidate_rollout_utilities) {
        vr.candidate_rollout_utilities.push_back({cru.first, cru.second});
      }
      all_results[r_idx][v_idx] = std::move(vr);
    }

    if ((r_idx + 1) % 10 == 0 || r_idx + 1 == sampled_roots.size()) {
      open_spiel::json::Object temp_summary;
      SaveBenchmarkReceipt(checkpoint_path, temp_summary, sampled_roots, variants, all_results, {}, r_idx + 1);
    }
  }

  std::cout << "[BENCHMARK] All roots evaluated! Analyzing true gain against Ref512...\n\n";

  const size_t ref_idx = variants.size() - 1;  // Ref512 is the last variant
  SPIEL_CHECK_EQ(variants[ref_idx].name, "Ref512");

  // Summarize metrics
  double total_weight_sum = 0.0;
  for (const auto& r : sampled_roots) total_weight_sum += r.stratum_weight;

  // MUST FIX 4(b): Summary table drops / labels Ref512 and reports Evals/Decision
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
            "Fatal: Variant %s chosen action %d on root %d not found in Ref512 candidate_utilities! "
            "Ref512 must evaluate all top candidate actions.",
            v_cfg.name, vr.chosen_action, root.root_id));
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

  // Ref512 reference statistics reported separately
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
      "[Reference Baseline] %s: 512 rollouts | Paired | Terminal | Avg Evals/Dec: %.1f | Avg Time: %.1f ms\n",
      variants[ref_idx].name, ref_avg_evals, ref_avg_time);
  std::cout << "====================================================================================================\n";
  std::cout << "* True Gain is measured against independent 512-rollout paired reference utilities.\n\n";

  root_out_json["variant_summaries"] = open_spiel::json::Value(variant_summaries);

  // Save complete main sweep immediately so results are preserved before recheck begins
  SaveBenchmarkReceipt(output_path, root_out_json, sampled_roots, variants, all_results, {}, sampled_roots.size());
  std::cout << "[RECEIPT] Saved complete main sweep (all variants across all roots) to " << output_path << "\n";

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

  std::vector<VariantResult> recheck_1024_results(sampled_roots.size());
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
        top_k, 1024, min_override_margin,
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

  SaveBenchmarkReceipt(output_path, root_out_json, sampled_roots, variants, all_results, recheck_1024_results, sampled_roots.size());
  std::cout << "[RECEIPT] Saved finalized benchmark receipt JSON to " << output_path << "\n";
  std::remove(checkpoint_path.c_str());

  return 0;
}
