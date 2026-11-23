#include <algorithm>
#include <cmath>
#include <csignal>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <set>
#include <shared_mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include <torch/torch.h>

#include "open_spiel/examples/dune_play_search_tournament_internal.h"
#include "open_spiel/examples/dune_seed_utils.h"
#include "open_spiel/examples/dune_sha256.h"


ABSL_FLAG(std::string, mode, "raw",
          "Controller: raw, legacy, wide, leader, adaptive, adaptive_leader, "
          "sh_adaptive, sh_adaptive_leader, full_sh_adaptive_leader, or sh_superhuman.");
ABSL_FLAG(std::string, cand_profile, "",
          "Optional candidate profile override (e.g. sh_superhuman).");
ABSL_FLAG(std::string, model_checkpoint, "", "Frozen candidate checkpoint.");
ABSL_FLAG(std::string, opponent_checkpoint, "", "Different frozen actual-opponent checkpoint.");
ABSL_FLAG(int, games, 4, "Number of requested games.");
ABSL_FLAG(std::string, game_ids, "", "Optional comma-separated list of explicit game IDs to run.");
ABSL_FLAG(int, start_game_id, 0, "First game ID, independent of scheduling.");
ABSL_FLAG(uint64_t, master_seed, 20261014ULL, "Shared seed across controller arms.");
ABSL_FLAG(int, threads, 4, "Concurrent full-game workers.");
ABSL_FLAG(int, target_batch_size, 256, "Evaluator target batch size.");
ABSL_FLAG(int, rollouts_per_action, 64, "Legacy reference rollouts per root action.");
ABSL_FLAG(std::string, output_jsonl, "", "Append completed games or roots to this file.");
ABSL_FLAG(double, max_seconds, 0.0, "Wall limit including model loading; zero disables it.");
ABSL_FLAG(std::string, device, "cpu", "Explicit device: cpu or cuda.");
ABSL_FLAG(bool, rollout_amp, false, "Use AMP for every candidate and opponent evaluation.");
ABSL_FLAG(bool, allow_tf32, false, "Allow TF32 for every candidate and opponent evaluation.");
ABSL_FLAG(std::string, roots_json, "", "Optional census_sampled_roots.json for serial root timing.");
ABSL_FLAG(int, max_roots, 12, "Maximum balanced roots; -1 selects all.");
ABSL_FLAG(std::string, crn_mode, "sequential",
          "CRN mode: sequential, deck_order, or event_keyed.");
ABSL_FLAG(bool, unbiased_stage2_gate, false,
          "Use unbiased Stage-2 Successive Halving gate.");
ABSL_FLAG(bool, fixed_top5_candidates, false,
          "Use fixed top-5 candidates (up to min(5, legal)) in Successive Halving.");
ABSL_FLAG(bool, all_buy_candidates, false,
          "Include all legal card and tech acquisitions at buy decisions.");
ABSL_FLAG(int, leader_rollouts, 32, "Leader pick rollout search rollouts.");
ABSL_FLAG(bool, sample_opponent_leader_draft, false,
          "Sample opponent leader draft mixture in rollout search.");
ABSL_FLAG(double, prior_log_ratio_weight, 0.02,
          "Prior log-ratio weight for override margin adjustment.");
ABSL_FLAG(double, stage1_high_conf_prior_skip, 0.95,
          "Skip Stage 2 when raw prior >= threshold and raw leads Stage 1.");
ABSL_FLAG(double, win_speed_tiebreak_eps, -1.0,
          "Terminal win-speed tie-breaker epsilon (-1 uses profile default: "
          "0.05 for sh_superhuman, 0.0 otherwise).");
ABSL_FLAG(bool, ceiling_aware_margin, true,
          "Scale override margin smoothly near +2.25 utility ceiling.");
ABSL_FLAG(double, kl_policy_tau, 0.0,
          "Optional KL-regularized search policy temperature (>0 enables smooth distribution).");
ABSL_FLAG(double, game_clock_ms, -1.0,
          "Optional per-game candidate search clock budget in ms (-1 disables clock).");

namespace open_spiel::dune_imperium::play_search_tournament {

volatile std::sig_atomic_t interrupted = 0;

namespace {

void HandleSignal(int) { interrupted = 1; }

Options ReadOptions(int argc = 0, char** argv = nullptr) {
  Options o;
  for (int i = 1; i < argc; ++i) {
    std::string_view arg = argv[i];
    if (arg.rfind("--leader_rollouts=", 0) == 0 ||
        arg.rfind("-leader_rollouts=", 0) == 0 ||
        arg == "--leader_rollouts" || arg == "-leader_rollouts") {
      o.leader_rollouts_explicit = true;
    } else if (arg.rfind("--crn_mode=", 0) == 0 ||
               arg.rfind("-crn_mode=", 0) == 0 ||
               arg == "--crn_mode" || arg == "-crn_mode") {
      o.crn_mode_explicit = true;
    } else if (arg.rfind("--unbiased_stage2_gate", 0) == 0 ||
               arg.rfind("-unbiased_stage2_gate", 0) == 0 ||
               arg.rfind("--nounbiased_stage2_gate", 0) == 0) {
      o.unbiased_stage2_gate_explicit = true;
    } else if (arg.rfind("--fixed_top5_candidates", 0) == 0 ||
               arg.rfind("-fixed_top5_candidates", 0) == 0 ||
               arg.rfind("--nofixed_top5_candidates", 0) == 0) {
      o.fixed_top5_explicit = true;
    } else if (arg.rfind("--all_buy_candidates", 0) == 0 ||
               arg.rfind("-all_buy_candidates", 0) == 0 ||
               arg.rfind("--noall_buy_candidates", 0) == 0) {
      o.all_buy_explicit = true;
    } else if (arg.rfind("--rollouts_per_action=", 0) == 0 ||
               arg.rfind("-rollouts_per_action=", 0) == 0 ||
               arg == "--rollouts_per_action" || arg == "-rollouts_per_action") {
      o.rollouts_explicit = true;
    }
  }
  o.cand_profile = absl::GetFlag(FLAGS_cand_profile);
  o.mode_name = absl::GetFlag(FLAGS_mode);
  if (!o.cand_profile.empty()) {
    if (o.cand_profile == "sh_superhuman") {
      o.mode_name = "sh_superhuman";
    } else {
      throw std::runtime_error("--cand_profile must be empty or sh_superhuman");
    }
  }
  if (o.mode_name == "raw") o.mode = Mode::kRaw;
  else if (o.mode_name == "legacy") o.mode = Mode::kLegacy;
  else if (o.mode_name == "wide") o.mode = Mode::kWide;
  else if (o.mode_name == "leader") o.mode = Mode::kLeader;
  else if (o.mode_name == "adaptive") o.mode = Mode::kAdaptive;
  else if (o.mode_name == "adaptive_leader") o.mode = Mode::kAdaptiveLeader;
  else if (o.mode_name == "sh_adaptive") o.mode = Mode::kShAdaptive;
  else if (o.mode_name == "sh_adaptive_leader") o.mode = Mode::kShAdaptiveLeader;
  else if (o.mode_name == "full_sh_adaptive_leader") o.mode = Mode::kFullShAdaptiveLeader;
  else if (o.mode_name == "sh_superhuman") o.mode = Mode::kShSuperhuman;
  else throw std::runtime_error("--mode must be raw, legacy, wide, leader, adaptive, adaptive_leader, sh_adaptive, sh_adaptive_leader, full_sh_adaptive_leader, or sh_superhuman");
  o.model_path = std::filesystem::canonical(absl::GetFlag(FLAGS_model_checkpoint)).string();
  o.opponent_path = std::filesystem::canonical(absl::GetFlag(FLAGS_opponent_checkpoint)).string();
  if (o.model_path == o.opponent_path) {
    throw std::runtime_error("Candidate and actual opponent checkpoints must differ");
  }
  o.roots_path = absl::GetFlag(FLAGS_roots_json);
  if (!o.roots_path.empty()) o.roots_path = std::filesystem::canonical(o.roots_path).string();
  o.output_path = absl::GetFlag(FLAGS_output_jsonl);
  o.device_name = absl::GetFlag(FLAGS_device);
  if (o.device_name == "CPU") o.device_name = "cpu";
  if (o.device_name != "cpu" && o.device_name != "cuda") {
    throw std::runtime_error("--device must be cpu or cuda");
  }
  o.games = absl::GetFlag(FLAGS_games);
  o.start_game_id = absl::GetFlag(FLAGS_start_game_id);
  std::string game_ids_str = absl::GetFlag(FLAGS_game_ids);
  if (!game_ids_str.empty()) {
    std::stringstream ss(game_ids_str);
    std::string token;
    while (std::getline(ss, token, ',')) {
      if (!token.empty()) {
        int gid = std::stoi(token);
        if (gid < 0) throw std::runtime_error("game_id must be non-negative");
        o.explicit_game_ids.push_back(gid);
      }
    }
    if (o.explicit_game_ids.empty()) {
      throw std::runtime_error("--game_ids cannot be empty if specified");
    }
    o.games = static_cast<int>(o.explicit_game_ids.size());
  }
  o.threads = absl::GetFlag(FLAGS_threads);
  o.batch_size = absl::GetFlag(FLAGS_target_batch_size);
  o.rollouts = absl::GetFlag(FLAGS_rollouts_per_action);
  o.max_roots = absl::GetFlag(FLAGS_max_roots);
  o.master_seed = absl::GetFlag(FLAGS_master_seed);
  o.rollout_amp = absl::GetFlag(FLAGS_rollout_amp);
  o.allow_tf32 = absl::GetFlag(FLAGS_allow_tf32);
  o.max_seconds = absl::GetFlag(FLAGS_max_seconds);
  std::string crn_mode_str = absl::GetFlag(FLAGS_crn_mode);
  if (crn_mode_str == "sequential") {
    o.crn_mode = SearchRolloutCrnMode::kSequential;
  } else if (crn_mode_str == "deck_order") {
    o.crn_mode = SearchRolloutCrnMode::kDeckOrderCrn;
  } else if (crn_mode_str == "event_keyed") {
    o.crn_mode = SearchRolloutCrnMode::kEventKeyedCrn;
  } else {
    throw std::runtime_error("--crn_mode must be sequential, deck_order, or event_keyed");
  }
  o.unbiased_stage2_gate = absl::GetFlag(FLAGS_unbiased_stage2_gate);
  o.fixed_top5_candidates = absl::GetFlag(FLAGS_fixed_top5_candidates);
  o.all_buy_candidates = absl::GetFlag(FLAGS_all_buy_candidates);
  o.leader_rollouts = absl::GetFlag(FLAGS_leader_rollouts);
  o.sample_opponent_leader_draft = absl::GetFlag(FLAGS_sample_opponent_leader_draft);
  o.prior_log_ratio_weight = absl::GetFlag(FLAGS_prior_log_ratio_weight);
  o.stage1_high_conf_prior_skip = absl::GetFlag(FLAGS_stage1_high_conf_prior_skip);
  o.ceiling_aware_margin = absl::GetFlag(FLAGS_ceiling_aware_margin);
  o.kl_policy_tau = std::max(0.0, absl::GetFlag(FLAGS_kl_policy_tau));
  o.game_clock_ms = absl::GetFlag(FLAGS_game_clock_ms);
  const double raw_eps = absl::GetFlag(FLAGS_win_speed_tiebreak_eps);
  if (o.mode == Mode::kShSuperhuman) {
    if (!o.crn_mode_explicit) o.crn_mode = SearchRolloutCrnMode::kEventKeyedCrn;
    if (!o.unbiased_stage2_gate_explicit) o.unbiased_stage2_gate = true;
    if (!o.fixed_top5_explicit) o.fixed_top5_candidates = true;
    if (!o.all_buy_explicit) o.all_buy_candidates = true;
    if (!o.rollouts_explicit) o.rollouts = 192;
    o.win_speed_tiebreak_eps = (raw_eps >= 0.0) ? raw_eps : 0.05;
  } else {
    o.win_speed_tiebreak_eps = (raw_eps >= 0.0) ? raw_eps : 0.0;
  }
  if (o.mode == Mode::kLeader && !o.leader_rollouts_explicit) {
    o.leader_rollouts = o.rollouts;
  }
  if (o.output_path.empty() || o.games < 0 || o.start_game_id < 0 ||
      o.start_game_id > std::numeric_limits<int>::max() - o.games ||
      o.threads < 1 || o.batch_size < 1 || o.rollouts < 1 ||
      o.leader_rollouts < 1 ||
      o.max_roots == 0 || o.max_roots < -1 ||
      !std::isfinite(o.max_seconds) || o.max_seconds < 0) {
    throw std::runtime_error("Invalid output, count, or time-limit option");
  }
  if (o.mode == Mode::kWide && o.rollouts < 2) {
    throw std::runtime_error("Wide requires --rollouts_per_action >= 2");
  }
  json::Object identity{
      {"controller_version", kControllerVersion}, {"mode", o.mode_name},
      {"simulation_policy", kSimulationPolicy},
      {"model_checkpoint", o.model_path}, {"opponent_checkpoint", o.opponent_path},
      {"master_seed", std::to_string(o.master_seed)}, {"roots_json", o.roots_path},
      {"rollouts_per_action", o.rollouts}, {"margin", kOverrideMargin},
      {"rollout_amp", o.rollout_amp}, {"allow_tf32", o.allow_tf32},
      {"device", o.device_name}, {"threads", o.threads},
      {"target_batch_size", o.batch_size},
      {"crn_mode", CrnModeName(o.crn_mode)},
      {"unbiased_stage2_gate", o.unbiased_stage2_gate},
      {"fixed_top5_candidates", o.fixed_top5_candidates},
      {"sample_opponent_leader_draft", o.sample_opponent_leader_draft},
      {"leader_rollouts", o.leader_rollouts}};
  if (o.win_speed_tiebreak_eps > 0.0) {
    identity["win_speed_tiebreak_eps"] = o.win_speed_tiebreak_eps;
  }
  if (!o.ceiling_aware_margin) {
    identity["ceiling_aware_margin"] = o.ceiling_aware_margin;
  }
  if (o.all_buy_candidates) {
    identity["all_buy_candidates"] = o.all_buy_candidates;
  }
  if (o.kl_policy_tau > 0.0) {
    identity["kl_policy_tau"] = o.kl_policy_tau;
  }
  if (o.game_clock_ms > 0.0) {
    identity["game_clock_ms"] = o.game_clock_ms;
  }
  o.config_key = ComputeStringSHA256(json::ToString(identity));
  return o;
}

ModelBundle LoadFrozenModel(const std::string& path, const Options& o,
                            torch::Device device) {
  std::filesystem::path p_ckpt(path);
  std::filesystem::path sidecar = p_ckpt;
  sidecar.replace_extension(".json");
  if (!std::filesystem::exists(sidecar)) {
    std::filesystem::path parent_manifest = p_ckpt.parent_path() / "manifest.json";
    if (std::filesystem::exists(parent_manifest)) {
      sidecar = parent_manifest;
    }
  }
  const auto value = ReadJson(sidecar.string());
  if (!value.IsObject()) throw std::runtime_error("Manifest must be an object: " + path);
  const auto& meta = value.GetObject();

  int observation_dim = -1;
  if (auto it = meta.find("observation_dim"); it != meta.end() && it->second.IsInt()) {
    observation_dim = static_cast<int>(it->second.GetInt());
  }
  int hidden_dim = -1;
  int num_blocks = -1;
  if (auto it = meta.find("hidden_dim"); it != meta.end() && it->second.IsInt()) {
    hidden_dim = static_cast<int>(it->second.GetInt());
  }
  if (auto it = meta.find("num_blocks"); it != meta.end() && it->second.IsInt()) {
    num_blocks = static_cast<int>(it->second.GetInt());
  }
  if ((hidden_dim <= 0 || num_blocks <= 0) && meta.find("architecture") != meta.end() &&
      meta.at("architecture").IsObject()) {
    const auto& arch = meta.at("architecture").GetObject();
    if (hidden_dim <= 0 && arch.find("hidden_dim") != arch.end() && arch.at("hidden_dim").IsInt()) {
      hidden_dim = static_cast<int>(arch.at("hidden_dim").GetInt());
    }
    if (num_blocks <= 0 && arch.find("num_blocks") != arch.end() && arch.at("num_blocks").IsInt()) {
      num_blocks = static_cast<int>(arch.at("num_blocks").GetInt());
    }
  }
  if (observation_dim <= 0 || hidden_dim <= 0 || num_blocks <= 0) {
    torch::serialize::InputArchive archive;
    archive.load_from(path, torch::kCPU);
    torch::serialize::InputArchive in_archive;
    if (archive.try_read("input_layer", in_archive)) {
      torch::Tensor w;
      if (in_archive.try_read("weight", w) && w.dim() == 2) {
        if (hidden_dim <= 0) hidden_dim = static_cast<int>(w.size(0));
        if (observation_dim <= 0) observation_dim = static_cast<int>(w.size(1));
      }
    }
    if (num_blocks <= 0) {
      int detected_blocks = 0;
      for (int b = 0; b < 64; ++b) {
        torch::serialize::InputArchive blk;
        if (!archive.try_read("block_" + std::to_string(b), blk)) break;
        ++detected_blocks;
      }
      if (detected_blocks > 0) num_blocks = detected_blocks;
    }
  }
  if (observation_dim <= 0 || hidden_dim <= 0 || num_blocks <= 0) {
    throw std::runtime_error("Cannot resolve checkpoint dimensions: " + path);
  }

  std::string market_mode_str = "none";
  if (auto it = meta.find("market_appendix_mode"); it != meta.end() && it->second.IsString()) {
    market_mode_str = it->second.GetString();
  }
  const MarketAppendixMode market_mode = ParseMarketAppendixMode(market_mode_str);
  DeckPoolSemantics deck_pool_semantics = DeckPoolSemantics::kLegacy;
  if (auto it = meta.find("deck_pool_semantics"); it != meta.end() && it->second.IsString()) {
    deck_pool_semantics = ParseDeckPoolSemanticsOrDie(it->second.GetString());
  }
  const auto schema = GetInformationStateSchema(market_mode, deck_pool_semantics);
  if (observation_dim != schema.size) {
    throw std::runtime_error("Checkpoint observation_dim (" +
                             std::to_string(observation_dim) +
                             ") mismatches schema size (" +
                             std::to_string(schema.size) + ") for " + path);
  }

  std::string descriptor_schema = "";
  if (auto it = meta.find("semantic_descriptor_schema"); it != meta.end() && it->second.IsString()) {
    descriptor_schema = it->second.GetString();
  }
  const bool enable_semantic_scorer_meta = BoolField(meta, "enable_semantic_scorer", false);
  const bool has_scorer = enable_semantic_scorer_meta ||
                          CheckpointHasSemanticScorer(path, torch::kCPU);
  const bool enable_zone_encoder = BoolField(meta, "enable_zone_encoder", false);
  const bool nonlinear_value_head = BoolField(meta, "nonlinear_value_head", false);
  const bool with_aux_heads = BoolField(meta, "with_aux_heads", false);
  bool slot_indexed_out_layer_f = false;
  if (auto it_sif = meta.find("slot_indexed_out_layer_f"); it_sif != meta.end()) {
    slot_indexed_out_layer_f =
        (it_sif->second.IsBool() && it_sif->second.GetBool()) ||
        (it_sif->second.IsString() && it_sif->second.GetString() == "true");
  }
  if (IsFullPublicInformationStateSize(observation_dim)) {
    if (!has_scorer ||
        (descriptor_schema != dune_semantic::kDescriptorSchemaVersionV3 &&
         descriptor_schema != dune_semantic::kDescriptorSchemaVersionV4 &&
         descriptor_schema != dune_semantic::kDescriptorSchemaVersionV5)) {
      throw std::runtime_error(
          "Full public information checkpoint requires semantic_action_v3/v4/v5 scorer: " + path);
    }
  }

  ModelBundle bundle;
  bundle.num_blocks = num_blocks;
  if (auto it_sha = meta.find("model_sha256"); it_sha != meta.end() && it_sha->second.IsString() &&
      !it_sha->second.GetString().empty()) {
    bundle.model_sha256 = it_sha->second.GetString();
  } else {
    bundle.model_sha256 = ComputeFileSHA256(path);
  }
  bundle.model = std::make_shared<SharedDunePolicyValueNetImpl>(
      observation_dim, hidden_dim, kNumDistinctPlayerActions, num_blocks,
      nonlinear_value_head, with_aux_heads, 0, has_scorer, enable_zone_encoder,
      descriptor_schema);
  if (!descriptor_schema.empty()) {
    bundle.model->SetSemanticDescriptorSchema(descriptor_schema);
  }
  if (slot_indexed_out_layer_f && bundle.model->semantic_scorer_ != nullptr) {
    bundle.model->semantic_scorer_->EnableF(true);
  }
  LoadModelCheckpointRobust(bundle.model, path, torch::Device(torch::kCPU));
  bundle.model->to(device);
  bundle.model->eval();
  for (auto& parameter : bundle.model->parameters()) parameter.set_requires_grad(false);
  if (device.is_cuda() && o.rollout_amp) {
    bundle.model->RefreshBf16InferenceCache();
  }
  bundle.mutex = std::make_unique<std::shared_mutex>();
  bundle.coordinator = std::make_shared<BatchedEvaluator>(
      bundle.model, o.batch_size, 1, device, bundle.mutex.get(), 10.0f,
      false, true, false, o.rollout_amp, o.allow_tf32);
  bundle.evaluator = std::make_shared<BatchedNNEvaluator>(
      bundle.coordinator, 10.0f, market_mode);
  std::cout << "[MODEL] " << path << " blocks=" << num_blocks
            << " hidden=" << hidden_dim
            << " obs=" << observation_dim
            << " market=" << market_mode_str
            << " deck_pool=" << DeckPoolSemanticsToString(deck_pool_semantics)
            << " schema=" << (descriptor_schema.empty() ? "none" : descriptor_schema)
            << " precision=" << (o.rollout_amp ? "AMP" : "FP32")
            << " tf32=" << o.allow_tf32
            << " device=" << device << '\n' << std::flush;
  return bundle;
}

std::set<int> ReadCompleted(const Options& o, Summary* summary_out = nullptr) {
  std::set<int> completed;
  std::ifstream input(o.output_path);
  if (!input && std::filesystem::exists(o.output_path)) {
    throw std::runtime_error("Cannot read existing output: " + o.output_path);
  }
  std::string line;
  const std::string kind = o.roots_path.empty() ? "game" : "root";
  while (std::getline(input, line)) {
    if (line.empty()) continue;
    const auto value = json::FromString(line);
    if (!value || !value->IsObject()) throw std::runtime_error("Malformed existing JSONL record");
    const auto& row = value->GetObject();
    if (StringField(row, "config_key") != o.config_key || StringField(row, "record_type") != kind) {
      throw std::runtime_error("Existing output has a different run identity");
    }
    int id = IntField(row, kind == "game" ? "game_id" : "root_id");
    if (!completed.insert(id).second) throw std::runtime_error("Duplicate completed ID in existing output");
    if (kind == "game") {
      auto returns = row.find("returns");
      if (returns == row.end() || !returns->second.IsArray() ||
          returns->second.GetArray().size() != 4 ||
          IntField(row, "seat") != id % 4 ||
          IntField(row, "placement") < 1 || IntField(row, "placement") > 4) {
        throw std::runtime_error("Incomplete completed game in existing output");
      }
      for (const auto& value : returns->second.GetArray()) {
        if (!value.IsNumber() ||
            !std::isfinite(value.IsDouble() ? value.GetDouble() : value.GetInt())) {
          throw std::runtime_error("Invalid terminal return in existing output");
        }
      }
      const uint64_t seed = dune_seed::DeriveSeed(o.master_seed, dune_seed::kStreamSearchSampling, id);
      if (StringField(row, "seed") != std::to_string(seed)) {
        throw std::runtime_error("Existing game has a mismatched seed");
      }
      if (summary_out != nullptr) {
        summary_out->completed_games++;
        const int placement = IntField(row, "placement");
        if (placement == 1) summary_out->wins++;
        summary_out->sum_utility += NumberField(row, "utility", 0.0);
        const int end_r = static_cast<int>(NumberField(row, "ending_round", 0.0));
        summary_out->sum_ending_round += end_r;
        if (end_r > 0 && end_r <= 7) summary_out->rounds_le_7_count++;
        summary_out->sum_cand_r7_track_vp += NumberField(row, "cand_r7_track_vp", 0.0);
        summary_out->sum_max4_r7_track_vp += NumberField(row, "max4_r7_track_vp", 0.0);
      }
    }
  }
  return completed;
}

}  // namespace

json::Value ReadJson(const std::string& path) {
  std::ifstream input(path);
  if (!input) throw std::runtime_error("Cannot read " + path);
  std::string content((std::istreambuf_iterator<char>(input)), {});
  auto parsed = json::FromString(content);
  if (!parsed) throw std::runtime_error("Invalid JSON in " + path);
  return *parsed;
}

int IntField(const json::Object& object, const std::string& field) {
  auto it = object.find(field);
  if (it == object.end() || !it->second.IsInt()) {
    throw std::runtime_error("Missing integer field: " + field);
  }
  return static_cast<int>(it->second.GetInt());
}

std::string StringField(const json::Object& object, const std::string& field) {
  auto it = object.find(field);
  if (it == object.end() || !it->second.IsString()) {
    throw std::runtime_error("Missing string field: " + field);
  }
  return it->second.GetString();
}

bool BoolField(const json::Object& object, const std::string& field,
               bool absent_value) {
  auto it = object.find(field);
  if (it == object.end()) return absent_value;
  if (!it->second.IsBool()) {
    throw std::runtime_error("Invalid boolean field: " + field);
  }
  return it->second.GetBool();
}

double NumberField(const json::Object& object, const std::string& field,
                   double absent_value) {
  auto it = object.find(field);
  if (it == object.end()) return absent_value;
  if (it->second.IsDouble()) return it->second.GetDouble();
  if (it->second.IsInt()) return static_cast<double>(it->second.GetInt());
  return absent_value;
}

}  // namespace open_spiel::dune_imperium::play_search_tournament

int main(int argc, char** argv) {
  using namespace open_spiel;
  using namespace open_spiel::dune_imperium::play_search_tournament;

  absl::ParseCommandLine(argc, argv);
  std::signal(SIGINT, HandleSignal);
  std::signal(SIGTERM, HandleSignal);
  try {
    const Options o = ReadOptions(argc, argv);
    RunControl control;
    control.max_seconds = o.max_seconds;
    Summary summary;
    const auto completed = ReadCompleted(o, &summary);
    std::vector<int> pending;
    int skipped = 0;
    if (o.roots_path.empty()) {
      if (!o.explicit_game_ids.empty()) {
        for (int id : o.explicit_game_ids) {
          if (completed.count(id)) ++skipped;
          else pending.push_back(id);
        }
      } else {
        for (int i = 0; i < o.games; ++i) {
          const int id = o.start_game_id + i;
          if (completed.count(id)) ++skipped;
          else pending.push_back(id);
        }
      }
      if (pending.empty()) {
        std::cout << "[DONE] requested games already complete: " << skipped
                  << " win_rate=" << summary.win_rate()
                  << " avg_utility=" << summary.avg_utility()
                  << " ending_round=" << summary.avg_ending_round()
                  << " cand_r7_track_vp=" << summary.avg_cand_r7_track_vp()
                  << " max4_r7_track_vp=" << summary.avg_max4_r7_track_vp()
                  << " frac_rounds_le_7=" << summary.frac_rounds_le_7() << '\n';
        return 0;
      }
    }
    std::ofstream output(o.output_path, std::ios::app);
    if (!output) throw std::runtime_error("Cannot append to " + o.output_path);
    torch::set_num_threads(1);
    torch::set_num_interop_threads(1);
    torch::Device device(torch::kCPU);
    if (o.device_name == "cuda") {
      if (!torch::cuda::is_available()) throw std::runtime_error("CUDA requested but unavailable");
      device = torch::Device(torch::kCUDA, 0);
    }
    auto candidate = LoadFrozenModel(o.model_path, o, device);
    if (control.Stop()) return 2;
    auto opponent = LoadFrozenModel(o.opponent_path, o, device);
    if (control.Stop()) return 2;
    if (candidate.model_sha256 == opponent.model_sha256) {
      throw std::runtime_error("Candidate and opponent manifests identify the same checkpoint");
    }
    auto game = LoadGame("dune_imperium");
    if (!o.roots_path.empty()) return RunRoots(o, *game, candidate, control, completed, output);

    const auto stats = RunTournamentGames(
        o, *game, candidate, opponent, control, pending, skipped, summary, output);
    const int done = stats.newly_completed;
    std::cout << "[DONE] completed=" << done + skipped << '/' << o.games
              << " newly_completed=" << done << " skipped=" << skipped
              << " elapsed=" << SecondsSince(control.start)
              << "s network_evaluations=" << g_total_network_evaluations.load()
              << " win_rate=" << summary.win_rate()
              << " avg_utility=" << summary.avg_utility()
              << " ending_round=" << summary.avg_ending_round()
              << " cand_r7_track_vp=" << summary.avg_cand_r7_track_vp()
              << " max4_r7_track_vp=" << summary.avg_max4_r7_track_vp()
              << " frac_rounds_le_7=" << summary.frac_rounds_le_7()
              << '\n' << std::flush;
    if (!stats.error.empty()) throw std::runtime_error(stats.error);
    return done + skipped == o.games ? 0 : 2;
  } catch (const std::exception& e) {
    std::cerr << "[ERROR] " << e.what() << '\n';
    return 1;
  }
}
