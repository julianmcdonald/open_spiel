#include "open_spiel/examples/dune_fixed_root_search_benchmark_internal.h"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <unordered_map>

#include "open_spiel/abseil-cpp/absl/flags/flag.h"
#include "open_spiel/abseil-cpp/absl/strings/str_format.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel::dune_imperium::fixed_root_benchmark {

EvaluatorBundle InitEvaluatorBundle(
    int num_evals,
    int target_batch_size,
    int num_blocks,
    const std::string& model_path,
    bool rollout_amp,
    bool allow_tf32,
    bool disable_global_tf32,
    bool log_device) {
  if (disable_global_tf32) {
    at::globalContext().setAllowTF32CuBLAS(false);
    at::globalContext().setAllowTF32CuDNN(false);
  }

  EvaluatorBundle bundle;
  bundle.device = torch::Device(torch::kCUDA, 0);
  if (!torch::cuda::is_available()) {
    bundle.device = torch::Device(torch::kCPU);
  }
  if (log_device) {
    std::cout << "[DEVICE] Using device: " << (bundle.device.is_cuda() ? "CUDA" : "CPU") << "\n";
  }

  bundle.model_mutexes.resize(num_evals);
  bundle.models.resize(num_evals);
  bundle.coords.resize(num_evals);
  bundle.evaluators.resize(num_evals);

  for (int e = 0; e < num_evals; ++e) {
    bundle.models[e] = std::make_shared<open_spiel::SharedDunePolicyValueNetImpl>(
        open_spiel::dune_imperium::kFullPublicInformationStateSize, 2048, 2391,
        num_blocks,
        /*nonlinear_value_head=*/false, /*with_aux_heads=*/false,
        /*head_init_seed=*/0, /*with_semantic_scorer=*/true);
    bundle.models[e]->to(bundle.device);
    open_spiel::LoadModelCheckpointRobust(bundle.models[e], model_path, bundle.device);
    bundle.models[e]->eval();

    bundle.model_mutexes[e] = std::make_unique<std::shared_mutex>();
    bundle.coords[e] = std::make_shared<open_spiel::BatchedEvaluator>(
        bundle.models[e], target_batch_size, /*timeout_ms=*/1, bundle.device,
        bundle.model_mutexes[e].get(), 10.0f,
        /*device_synchronize=*/false, /*high_priority_stream=*/true,
        /*emit_batch_membership=*/false,
        /*rollout_amp=*/rollout_amp,
        /*allow_tf32=*/allow_tf32);
    bundle.evaluators[e] = std::make_shared<open_spiel::BatchedNNEvaluator>(bundle.coords[e], 10.0f);
  }
  return bundle;
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

std::vector<SampledRoot> LoadSampledRootsJson(
    const std::string& roots_path,
    const std::string& current_game_str,
    const RootLoadVerifyOptions& opts) {
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

  if (opts.verify_game_string) {
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

  if (opts.filter_placement_only) {
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
  } else if (opts.log_loaded_count) {
    std::cout << "[LOAD] Loaded " << sampled_roots.size() << " sampled roots for benchmark.\n";
  }
  return sampled_roots;
}

void VerifySampledRootsObsParity(
    const open_spiel::Game& game,
    const std::string& obs_path,
    const std::shared_ptr<open_spiel::BatchedNNEvaluator>& evaluator,
    const RootLoadVerifyOptions& opts,
    std::vector<SampledRoot>& sampled_roots) {
  std::ifstream obs_file(obs_path, std::ios::binary);
  if (!obs_file.is_open()) {
    open_spiel::SpielFatalError("Failed to open observation binary: " + obs_path);
  }

  const size_t obs_elem_count = open_spiel::dune_imperium::kFullPublicInformationStateSize;
  const size_t obs_byte_stride = obs_elem_count * sizeof(float);
  std::vector<float> rec_obs_buffer(obs_elem_count);

  if (opts.verbose_parity_messages) {
    std::cout << "[VERIFY] Replaying action histories and verifying observations against binary...\n";
  } else {
    std::cout << "[VERIFY] Replaying and verifying observation parity...\n";
  }

  for (size_t i = 0; i < sampled_roots.size(); ++i) {
    auto& root = sampled_roots[i];
    std::unique_ptr<open_spiel::State> state = game.NewInitialState();
    bool replay_failed = false;
    for (int64_t a : root.action_history) {
      if (opts.check_legal_during_replay) {
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
      }
      state->ApplyAction(a);
    }
    if (replay_failed || state->CurrentPlayer() != root.search_seat) {
      if (opts.verbose_parity_messages) {
        open_spiel::SpielFatalError(absl::StrFormat(
            "Root %d seat mismatch or replay failed: state->CurrentPlayer()=%d, root.search_seat=%d",
            root.root_id, state->CurrentPlayer(), root.search_seat));
      } else {
        open_spiel::SpielFatalError(absl::StrFormat(
            "Root %d seat mismatch: state=%d, root=%d",
            root.root_id, state->CurrentPlayer(), root.search_seat));
      }
    }

    std::vector<float> recomputed_obs = evaluator->GetConsumedObservation(*state, root.search_seat);
    if (opts.check_legal_during_replay && recomputed_obs.size() != obs_elem_count) {
      open_spiel::SpielFatalError(absl::StrFormat(
          "Observation size mismatch: recomputed=%zu, expected=%zu",
          recomputed_obs.size(), obs_elem_count));
    }

    obs_file.seekg(root.obs_row * obs_byte_stride);
    obs_file.read(reinterpret_cast<char*>(rec_obs_buffer.data()), obs_byte_stride);
    if (opts.check_obs_stream_good && !obs_file.good()) {
      open_spiel::SpielFatalError(absl::StrFormat(
          "Failed to read observation binary at row %d for root %d", root.obs_row, root.root_id));
    }

    float max_diff = 0.0f;
    for (size_t j = 0; j < obs_elem_count; ++j) {
      float diff = std::abs(recomputed_obs[j] - rec_obs_buffer[j]);
      if (diff > max_diff) max_diff = diff;
    }
    if (max_diff > 1e-4f) {
      if (opts.verbose_parity_messages) {
        open_spiel::SpielFatalError(absl::StrFormat(
            "Observation mismatch on root %d (game %d, dec %d): max_diff=%e > 1e-4",
            root.root_id, root.game_id, root.decision_ordinal, max_diff));
      } else {
        open_spiel::SpielFatalError(absl::StrFormat(
            "Observation mismatch on root %d: max_diff=%e", root.root_id, max_diff));
      }
    }

    if (opts.fill_missing_pi_parent && root.pi_parent.empty()) {
      auto prior = evaluator->Prior(*state);
      root.pi_parent.assign(root.legal_actions.size(), 0.0);
      for (size_t idx = 0; idx < root.legal_actions.size(); ++idx) {
        int a = root.legal_actions[idx];
        for (const auto& ap : prior) {
          if (ap.first == a) {
            root.pi_parent[idx] = ap.second;
            break;
          }
        }
      }
    }
  }
  obs_file.close();

  if (opts.verbose_parity_messages) {
    std::cout << absl::StrFormat("[VERIFY] Verified replay, seat and observation parity for %zu roots.\n",
                                 sampled_roots.size());
  } else {
    std::cout << "[VERIFY] Observation parity verified for all " << sampled_roots.size() << " roots.\n";
  }

  if (opts.require_non_empty && sampled_roots.empty()) {
    open_spiel::SpielFatalError("No roots provided for benchmark!");
  }
}

std::vector<VariantConfig> BuildBenchmarkVariants() {
  if (absl::GetFlag(FLAGS_sh_budget_check)) {
    return {
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
  }
  if (absl::GetFlag(FLAGS_sh_offline_check)) {
    return {
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
  }
  if (absl::GetFlag(FLAGS_leaf_check_only)) {
    return {
        {"V2_16_Paired", 16, SearchPairingMode::kPaired, SearchTruncationMode::kNextRoundSearchPlayer,
         benchmark_domains::kDomainV2_16_Paired, BenchmarkVariantType::kCompound, 0, 0, false},
        {"Ref512", 512, SearchPairingMode::kPaired, SearchTruncationMode::kTerminal,
         benchmark_domains::kDomainRef512, BenchmarkVariantType::kCompound, 0, 0, false}
    };
  }
  return {
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

void ReadPreservationItem(std::ifstream& ifs, CapturedPreservationItem& item) {
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

int LoadPreservationCheckpoint(
    const std::string& ckpt_path,
    int num_games,
    std::vector<std::vector<CapturedPreservationItem>>& per_game_pres,
    std::atomic<uint8_t>* game_completed) {
  int completed_games_count = 0;
  if (!std::filesystem::exists(ckpt_path)) return 0;

  std::ifstream ifs(ckpt_path, std::ios::binary);
  if (!ifs.is_open()) return 0;

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
          ReadPreservationItem(ifs, per_game_pres[g][item_idx]);
        }
        if (game_completed[g].load(std::memory_order_relaxed) == 0) {
          game_completed[g].store(1, std::memory_order_release);
          completed_games_count++;
        }
      } else {
        for (uint64_t item_idx = 0; item_idx < num_items; ++item_idx) {
          CapturedPreservationItem dummy;
          ReadPreservationItem(ifs, dummy);
        }
      }
    }
    std::cout << absl::StrFormat("[RESUME] Loaded %d completed games from %s\n",
                                 completed_games_count, ckpt_path);
  } else {
    std::cerr << "[RESUME] Warning: Checkpoint magic/version mismatch in " << ckpt_path << ", ignoring.\n";
  }
  return completed_games_count;
}

int LoadLabelCheckpoint(
    const std::string& ckpt_path,
    const std::vector<SampledRoot>& sampled_roots,
    std::vector<LabelledResult>& results,
    std::atomic<uint8_t>* completed) {
  int completed_count = 0;
  std::ifstream ckpt_file(ckpt_path);
  if (!ckpt_file.is_open()) return 0;

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
      std::cout << absl::StrFormat("[RESUME] Loaded %d completed roots from checkpoint %s\n",
                                   completed_count, ckpt_path);
    }
  }
  return completed_count;
}

std::vector<ParsedGame> LoadTieBreakerGames(const std::string& games_jsonl_path) {
  std::vector<ParsedGame> parsed_games;
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

  std::cout << absl::StrFormat("[TIE-BREAKER] Loaded %zu games from %s\n",
                               parsed_games.size(), games_jsonl_path);
  return parsed_games;
}

}  // namespace open_spiel::dune_imperium::fixed_root_benchmark
