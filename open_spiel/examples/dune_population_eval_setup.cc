// Setup, dimension auto-detection, and model/evaluator loading for
// dune_population_eval.

#include "dune_population_eval_internal.h"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "open_spiel/abseil-cpp/absl/flags/flag.h"
#include "open_spiel/abseil-cpp/absl/strings/str_format.h"
#include "open_spiel/utils/json.h"

#include "dune_seed_utils.h"

namespace open_spiel {
namespace dune_pop_eval {

// ---------------------------------------------------------------------------
// Domain resolution & string helpers
// ---------------------------------------------------------------------------
uint64_t ResolveDomain(const std::string& domain_str) {
  if (domain_str == "EVAL_BASELINE") return dune_seed::kDomainEvalBaseline;
  if (domain_str == "EVAL_DEV")      return dune_seed::kDomainEvalDev;
  if (domain_str == "EVAL_FINAL")    return dune_seed::kDomainEvalFinal;
  std::cerr << "Unknown domain: " << domain_str
            << ". Must be EVAL_BASELINE, EVAL_DEV, or EVAL_FINAL.\n";
  std::exit(1);
}

std::vector<std::string> SplitCommaSeparated(const std::string& value) {
  std::vector<std::string> items;
  std::stringstream ss(value);
  std::string item;
  while (std::getline(ss, item, ',')) {
    // Trim whitespace
    item.erase(item.begin(), std::find_if(item.begin(), item.end(),
               [](unsigned char ch) { return !std::isspace(ch); }));
    item.erase(std::find_if(item.rbegin(), item.rend(),
               [](unsigned char ch) { return !std::isspace(ch); }).base(),
               item.end());
    if (!item.empty()) items.push_back(item);
  }
  return items;
}

dune_imperium::MarketAppendixMode ParseMarketMode(const std::string& str) {
  return dune_imperium::ParseMarketAppendixMode(str);
}

std::string MarketModeToString(dune_imperium::MarketAppendixMode mode) {
  return dune_imperium::MarketAppendixModeToString(mode);
}

// ---------------------------------------------------------------------------
// Auto-detect model hidden_dim and num_blocks from JSON manifest or weight keys
// ---------------------------------------------------------------------------
bool DetectModelDimensions(
    const std::string& model_path, int* hidden_dim, int* num_blocks,
    int* input_dim, std::string* market_mode,
    dune_imperium::DeckPoolSemantics* deck_pool_semantics,
    std::string* semantic_descriptor_schema,
    bool* slot_indexed_out_layer_f) {
  open_spiel::DetectedModelDimensions detected;
  if (!open_spiel::DetectModelDimensions(
          model_path, &detected, /*require_sidecar_json=*/true)) {
    return false;
  }
  if (hidden_dim != nullptr) {
    *hidden_dim = detected.hidden_dim;
  }
  if (num_blocks != nullptr) {
    *num_blocks = detected.num_blocks;
  }
  if (input_dim != nullptr && detected.has_input_dim) {
    *input_dim = static_cast<int>(detected.input_dim);
  }
  if (market_mode != nullptr && !detected.market_mode.empty()) {
    *market_mode = detected.market_mode;
  }
  if (deck_pool_semantics != nullptr) {
    *deck_pool_semantics = detected.deck_pool_semantics;
  }
  if (semantic_descriptor_schema != nullptr &&
      !detected.semantic_descriptor_schema.empty()) {
    *semantic_descriptor_schema = detected.semantic_descriptor_schema;
  }
  if (slot_indexed_out_layer_f != nullptr) {
    *slot_indexed_out_layer_f = detected.slot_indexed_out_layer_f;
  }
  return true;
}

// ---------------------------------------------------------------------------
// Runtime initialization: game, device/threads, candidate & opponent models
// ---------------------------------------------------------------------------
std::unique_ptr<EvalRuntime> SetupEvaluation(const EvalConfig& cfg) {
  auto rt = std::make_unique<EvalRuntime>();

  if (cfg.use_opponent_model) {
    for (const std::string& opp_path : cfg.opponent_paths) {
      rt->opponent_names.push_back(
          std::filesystem::path(opp_path).stem().string());
    }
  }

  // Initialize game
  rt->game = LoadGame("dune_imperium");

  rt->provides_info_state_tensor =
      rt->game->GetType().provides_information_state_tensor;
  rt->provides_observations_tensor =
      rt->game->GetType().provides_observation_tensor;
  if (rt->provides_info_state_tensor) {
    rt->obs_size = rt->game->InformationStateTensorSize();
  } else if (rt->provides_observations_tensor) {
    rt->obs_size = rt->game->ObservationTensorSize();
  }
  if (rt->obs_size == 0) {
    std::cerr << "Error: observation size is 0.\n";
    std::exit(1);
  }
  rt->action_size = rt->game->NumDistinctActions();

  // Device setup
  rt->device = torch::cuda::is_available()
                   ? torch::Device(torch::kCUDA)
                   : torch::Device(torch::kCPU);

  if (cfg.deterministic) {
    std::cout << "INFO: Running in deterministic mode (default). Use --deterministic_eval=false for faster batched evaluation.\n";
  }

  // Determine thread count
  unsigned int hw_threads = std::thread::hardware_concurrency();
  if (hw_threads == 0) hw_threads = 4;
  if (cfg.requested_threads > 0) {
    rt->num_threads = static_cast<unsigned int>(cfg.requested_threads);
  } else {
    rt->num_threads = hw_threads * 4;
    if (rt->device.is_cuda() && !cfg.deterministic && cfg.total_games > 0) {
      rt->num_threads = std::max(
          rt->num_threads,
          std::min(256u, static_cast<unsigned int>(cfg.total_games)));
    }
  }

  if (rt->device.is_cuda()) {
    rt->eval_batch_size = std::min(64u, std::max(1u, rt->num_threads / 5));
    rt->opp_eval_batch_size = std::min(128u, std::max(1u, rt->num_threads / 2));
    rt->eval_timeout_ms = 1;
  } else {
    rt->eval_batch_size = std::min(32u, rt->num_threads);
    rt->opp_eval_batch_size = std::min(32u, rt->num_threads);
    rt->eval_timeout_ms = 2;
  }

  if (absl::GetFlag(FLAGS_candidate_eval_batch_size) > 0) {
    rt->eval_batch_size = absl::GetFlag(FLAGS_candidate_eval_batch_size);
  } else if (absl::GetFlag(FLAGS_candidate_r758_teacher) && rt->device.is_cuda()) {
    rt->eval_batch_size = 256;
  }

  if (absl::GetFlag(FLAGS_candidate_r758_teacher) && cfg.deterministic) {
    SpielFatalError("--candidate_r758_teacher=true requires batched evaluation (--deterministic_eval=false)");
  }

  rt->device_name = rt->device.is_cuda() ? "CUDA (GPU)" : "CPU";

  // Load eval model with auto-detected dimensions
  rt->main_hidden_dim = cfg.hidden_dim;
  rt->main_num_blocks = cfg.num_blocks;
  rt->main_input_dim = static_cast<int>(rt->obs_size);
  std::string main_detected_market_mode = "none";
  dune_imperium::DeckPoolSemantics main_detected_semantics = dune_imperium::DeckPoolSemantics::kLegacy;
  std::string main_detected_schema = "semantic_action_v3";
  bool main_slot_indexed_f = false;
  if (!DetectModelDimensions(cfg.model_checkpoint, &rt->main_hidden_dim, &rt->main_num_blocks,
                             &rt->main_input_dim, &main_detected_market_mode,
                             &main_detected_semantics, &main_detected_schema, &main_slot_indexed_f)) {
    SpielFatalError("Failed to detect model dimensions for main checkpoint: " + cfg.model_checkpoint);
  }

  std::string cand_market_flag = absl::GetFlag(FLAGS_candidate_market_appendix_mode);
  if (cand_market_flag == "auto") {
    if (dune_imperium::IsExtendedInformationStateSize(rt->main_input_dim)) {
      rt->cand_market_mode = ParseMarketMode(main_detected_market_mode);
      if (rt->cand_market_mode == dune_imperium::MarketAppendixMode::kNone) {
        SpielFatalError(absl::StrFormat(
            "Expanded candidate checkpoint %s has absent or invalid market_appendix_mode: '%s'",
            cfg.model_checkpoint, main_detected_market_mode));
      }
    } else {
      rt->cand_market_mode = dune_imperium::MarketAppendixMode::kNone;
    }
  } else {
    rt->cand_market_mode = ParseMarketMode(cand_market_flag);
    if (rt->cand_market_mode == dune_imperium::MarketAppendixMode::kNone &&
        dune_imperium::IsExtendedInformationStateSize(rt->main_input_dim)) {
      SpielFatalError(absl::StrFormat(
          "Expanded candidate checkpoint %s cannot use market_appendix_mode=none",
          cfg.model_checkpoint));
    }
  }

  if (rt->cand_market_mode != dune_imperium::MarketAppendixMode::kNone) {
    SPIEL_CHECK_EQ(rt->main_input_dim,
                   dune_imperium::GetInformationStateSchema(rt->cand_market_mode, main_detected_semantics).size);
  }

  bool cand_has_scorer = absl::GetFlag(FLAGS_enable_semantic_scorer) ||
                         CheckpointHasSemanticScorer(cfg.model_checkpoint, rt->device);
  if (dune_imperium::IsFullPublicInformationStateSize(rt->main_input_dim) && !cand_has_scorer) {
    SpielFatalError("Candidate full public information actor requires an active semantic scorer with schema v3, v4, or v5");
  }
  auto model = std::make_shared<SharedDunePolicyValueNetImpl>(
      rt->main_input_dim, rt->main_hidden_dim, rt->action_size, rt->main_num_blocks,
      cfg.nonlinear_value_head, false, 0, cand_has_scorer, false, main_detected_schema);
  model->SetSemanticDescriptorSchema(main_detected_schema);
  if (main_slot_indexed_f && model->semantic_scorer_ != nullptr) {
    model->semantic_scorer_->EnableF(true);
  }
  model->eval();
  LoadModelCheckpointRobust(model, cfg.model_checkpoint, rt->device);
  model->market_appendix_mode_ = rt->cand_market_mode;
  model->deck_pool_semantics_ = main_detected_semantics;
  model->to(rt->device);

  if (rt->device.is_cuda()) {
    at::globalContext().setAllowTF32CuBLAS(cfg.allow_tf32);
    at::globalContext().setAllowTF32CuDNN(cfg.allow_tf32);
  }

  if (cfg.deterministic) {
    rt->model_evaluator = std::make_shared<DeterministicEvaluator>(
        model, rt->device, &rt->eval_mutex, nullptr, nullptr, cfg.rollout_amp);
  } else {
    const float cand_logit_cap_val = absl::GetFlag(FLAGS_candidate_r758_teacher)
                                         ? static_cast<float>(cfg.candidate_logit_cap)
                                         : 0.0f;
    rt->model_evaluator = std::make_shared<BatchedEvaluator>(
        model, rt->eval_batch_size, rt->eval_timeout_ms, rt->device,
        &rt->sync_mutex, cand_logit_cap_val, /*device_synchronize=*/false, false, false,
        cfg.rollout_amp, cfg.allow_tf32, /*critic_model=*/nullptr,
        /*clear_autocast_cache_per_batch=*/false);
  }

  // Load opponent models with auto-detected dimensions
  std::unordered_map<std::string, size_t> seen_opp_path_to_idx;
  for (const std::string& opp_path : cfg.opponent_paths) {
    auto it = seen_opp_path_to_idx.find(opp_path);
    if (it != seen_opp_path_to_idx.end()) {
      size_t prev_idx = it->second;
      rt->opp_metadata.push_back(rt->opp_metadata[prev_idx]);
      rt->opponent_evaluators.push_back(rt->opponent_evaluators[prev_idx]);
      continue;
    }
    int opp_detected_hidden = cfg.opp_hidden_dim;
    int opp_detected_blocks = cfg.opp_num_blocks;
    int opp_detected_input_dim = static_cast<int>(rt->obs_size);
    std::string opp_detected_market_mode_str = "none";
    dune_imperium::DeckPoolSemantics opp_detected_semantics = dune_imperium::DeckPoolSemantics::kLegacy;
    std::string opp_detected_schema = "semantic_action_v3";
    bool opp_slot_indexed_f = false;
    if (!DetectModelDimensions(opp_path, &opp_detected_hidden, &opp_detected_blocks,
                               &opp_detected_input_dim, &opp_detected_market_mode_str,
                               &opp_detected_semantics, &opp_detected_schema, &opp_slot_indexed_f)) {
      SpielFatalError("Failed to detect model dimensions for opponent checkpoint: " + opp_path);
    }
    auto opp_market_mode = ParseMarketMode(opp_detected_market_mode_str);
    if (dune_imperium::IsExtendedInformationStateSize(opp_detected_input_dim)) {
      if (opp_market_mode == dune_imperium::MarketAppendixMode::kNone) {
        SpielFatalError(absl::StrFormat(
            "Expanded opponent checkpoint %s has absent or invalid market_appendix_mode: '%s'",
            opp_path, opp_detected_market_mode_str));
      }
      SPIEL_CHECK_EQ(opp_detected_input_dim,
                     dune_imperium::GetInformationStateSchema(opp_market_mode, opp_detected_semantics).size);
    }
    size_t new_idx = rt->opp_metadata.size();
    rt->opp_metadata.push_back({opp_path, opp_detected_hidden, opp_detected_blocks,
                                opp_detected_input_dim, opp_market_mode,
                                MarketModeToString(opp_market_mode), opp_detected_semantics});

    bool opp_has_scorer = CheckpointHasSemanticScorer(opp_path, rt->device);
    if (dune_imperium::IsFullPublicInformationStateSize(opp_detected_input_dim) && !opp_has_scorer) {
      SpielFatalError("Opponent full public information actor requires an active semantic scorer with schema v3, v4, or v5: " + opp_path);
    }
    auto opp_model = std::make_shared<SharedDunePolicyValueNetImpl>(
        opp_detected_input_dim, opp_detected_hidden, rt->action_size, opp_detected_blocks,
        cfg.opponent_nonlinear_value_head, false, 0, opp_has_scorer, false, opp_detected_schema);
    opp_model->SetSemanticDescriptorSchema(opp_detected_schema);
    if (opp_slot_indexed_f && opp_model->semantic_scorer_ != nullptr) {
      opp_model->semantic_scorer_->EnableF(true);
    }
    opp_model->eval();
    LoadModelCheckpointRobust(opp_model, opp_path, rt->device);
    opp_model->to(rt->device);

    if (cfg.deterministic) {
      rt->opponent_evaluators.push_back(std::make_shared<DeterministicEvaluator>(
          opp_model, rt->device, &rt->eval_mutex, nullptr, nullptr, cfg.rollout_amp));
    } else {
      rt->opponent_evaluators.push_back(std::make_shared<BatchedEvaluator>(
          opp_model, rt->opp_eval_batch_size, rt->eval_timeout_ms, rt->device,
          &rt->sync_mutex, 0.0f, /*device_synchronize=*/false, false, false,
          cfg.rollout_amp, cfg.allow_tf32, /*critic_model=*/nullptr,
          /*clear_autocast_cache_per_batch=*/false));
    }
    seen_opp_path_to_idx[opp_path] = new_idx;
  }

  return rt;
}

void PrintEvaluationConfig(const EvalConfig& cfg, const EvalRuntime& rt) {
  std::cout << "=== Dune Population Evaluator ===\n"
            << "Model:      " << cfg.model_checkpoint << "\n"
            << "Opponents:  "
            << (cfg.use_opponent_model ? cfg.opponent_str : "Random Agents") << "\n"
            << "Domain:     " << cfg.domain_str << "\n"
            << "Base seed:  " << cfg.base_seed << "\n"
            << "Games:      " << cfg.total_games << "\n"
            << "Greedy:     " << (cfg.greedy ? "true" : "false") << "\n"
            << "Temperature:" << cfg.temperature << "\n"
            << "OppGreedy:  "
            << (cfg.has_opponent_greedy ? (cfg.opponent_greedy ? "true" : "false")
                                        : "(inherit --greedy)") << "\n"
            << "OppTemp:    "
            << (cfg.has_opponent_temperature ? std::to_string(cfg.opponent_temperature)
                                             : std::string("(inherit --temperature)"))
            << "\n"
            << "CandCap:    " << cfg.candidate_logit_cap << "\n"
            << "OppCap:     " << cfg.opponent_logit_cap << "\n"
            << "Threads:    " << rt.num_threads << "\n"
            << "Eval mode:  " << (cfg.deterministic ? "Deterministic (batch-1)" : "Batched") << "\n"
            << "Rollout AMP:" << (cfg.rollout_amp ? "true" : "false") << "\n"
            << "Allow TF32: " << (cfg.allow_tf32 ? "true" : "false") << "\n"
            << "Record Hist:" << (cfg.record_history ? "true" : "false") << "\n"
            << "Nonlinear: "
            << (cfg.nonlinear_value_head ? "true" : "false")
            << "\n"
            << "Batch size: " << rt.eval_batch_size << "\n"
            << "Device:     " << rt.device_name << "\n"
            << "Hidden dim: " << rt.main_hidden_dim << " / Blocks: " << rt.main_num_blocks
            << " / Obs dim: " << rt.main_input_dim
            << " (Market: " << MarketModeToString(rt.cand_market_mode) << ")\n";
  if (cfg.use_opponent_model && (cfg.opp_hidden_dim != cfg.hidden_dim ||
                                 cfg.opp_num_blocks != cfg.num_blocks)) {
    std::cout << "Opp dim:    " << cfg.opp_hidden_dim
              << " / Blocks: " << cfg.opp_num_blocks << "\n";
  }
  std::cout << std::endl;
}

}  // namespace dune_pop_eval
}  // namespace open_spiel
