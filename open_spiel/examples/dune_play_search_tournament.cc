#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <csignal>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <random>
#include <set>
#include <shared_mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include <torch/torch.h>

#include "open_spiel/examples/dune_batched_evaluator.h"
#include "open_spiel/examples/dune_compound_turn_search.h"
#include "open_spiel/examples/dune_network.h"
#include "open_spiel/examples/dune_wide_rollout_search.h"
#include "open_spiel/examples/dune_seed_utils.h"
#include "open_spiel/examples/dune_sha256.h"
#include "open_spiel/games/dune_imperium/dune_imperium.h"
#include "open_spiel/spiel.h"
#include "open_spiel/utils/json.h"

ABSL_FLAG(int, ppo_minibatch_size, 2048, "");
ABSL_FLAG(int, ppo_update_epochs, 4, "");
ABSL_FLAG(double, ppo_clip_epsilon, 0.2, "");
ABSL_FLAG(bool, normalize_advantages, true, "");
ABSL_FLAG(bool, ppo_clip_value_loss, true, "");
ABSL_FLAG(double, entropy_coef, 0.01, "");
ABSL_FLAG(double, value_coef, 0.5, "");
ABSL_FLAG(double, target_kl, 0.0, "");
ABSL_FLAG(bool, train_amp, true, "");
ABSL_FLAG(double, grad_clip_norm, 0.5, "");
ABSL_FLAG(double, logit_cap, 10.0, "");
ABSL_FLAG(uint64_t, shaping_start_env_steps, 0, "");
ABSL_FLAG(uint64_t, shaping_decay_env_steps, 0, "");
ABSL_FLAG(bool, diagnostics_only, false, "");

ABSL_FLAG(std::string, mode, "raw",
          "Controller: raw, legacy, wide, leader, adaptive, or adaptive_leader.");
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

namespace {
using namespace open_spiel;
using namespace open_spiel::dune_imperium;
using Clock = std::chrono::steady_clock;
constexpr double kOverrideMargin = 0.15;
constexpr int kControllerVersion = 3;
constexpr const char* kSimulationPolicy = "candidate_model_all_seats";
volatile std::sig_atomic_t interrupted = 0;
void HandleSignal(int) { interrupted = 1; }

enum class Mode {
  kRaw,
  kLegacy,
  kWide,
  kLeader,
  kAdaptive,
  kAdaptiveLeader,
  kShAdaptive,
  kShAdaptiveLeader,
  kFullShAdaptiveLeader
};

struct Options {
  Mode mode;
  std::string mode_name;
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
};

double SecondsSince(Clock::time_point start) {
  return std::chrono::duration<double>(Clock::now() - start).count();
}

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

Options ReadOptions() {
  Options o;
  o.mode_name = absl::GetFlag(FLAGS_mode);
  if (o.mode_name == "raw") o.mode = Mode::kRaw;
  else if (o.mode_name == "legacy") o.mode = Mode::kLegacy;
  else if (o.mode_name == "wide") o.mode = Mode::kWide;
  else if (o.mode_name == "leader") o.mode = Mode::kLeader;
  else if (o.mode_name == "adaptive") o.mode = Mode::kAdaptive;
  else if (o.mode_name == "adaptive_leader") o.mode = Mode::kAdaptiveLeader;
  else if (o.mode_name == "sh_adaptive") o.mode = Mode::kShAdaptive;
  else if (o.mode_name == "sh_adaptive_leader") o.mode = Mode::kShAdaptiveLeader;
  else if (o.mode_name == "full_sh_adaptive_leader") o.mode = Mode::kFullShAdaptiveLeader;
  else throw std::runtime_error("--mode must be raw, legacy, wide, leader, adaptive, adaptive_leader, sh_adaptive, sh_adaptive_leader, or full_sh_adaptive_leader");
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
  if (o.output_path.empty() || o.games < 0 || o.start_game_id < 0 ||
      o.start_game_id > std::numeric_limits<int>::max() - o.games ||
      o.threads < 1 || o.batch_size < 1 || o.rollouts < 1 ||
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
      {"target_batch_size", o.batch_size}};
  o.config_key = ComputeStringSHA256(json::ToString(identity));
  return o;
}

struct ModelBundle {
  std::shared_ptr<SharedDunePolicyValueNetImpl> model;
  std::unique_ptr<std::shared_mutex> mutex;
  std::shared_ptr<BatchedEvaluator> coordinator;
  std::shared_ptr<BatchedNNEvaluator> evaluator;
  int num_blocks;
  std::string model_sha256;
};

ModelBundle LoadFrozenModel(const std::string& path, const Options& o,
                            torch::Device device) {
  auto sidecar = std::filesystem::path(path).replace_extension(".json");
  const auto value = ReadJson(sidecar.string());
  if (!value.IsObject()) throw std::runtime_error("Manifest must be an object: " + path);
  const auto& meta = value.GetObject();
  const int observation_dim = IntField(meta, "observation_dim");
  const int hidden_dim = IntField(meta, "hidden_dim");
  const int num_blocks = IntField(meta, "num_blocks");
  if (observation_dim != kFullPublicInformationStateSize || hidden_dim != 2048 ||
      (num_blocks != 8 && num_blocks != 12) ||
      StringField(meta, "market_appendix_mode") != "full_public_information_v3" ||
      StringField(meta, "semantic_descriptor_schema") != "semantic_action_v3" ||
      !BoolField(meta, "enable_semantic_scorer", false) ||
      BoolField(meta, "enable_zone_encoder", false) ||
      BoolField(meta, "nonlinear_value_head", false) ||
      BoolField(meta, "with_aux_heads", false)) {
    throw std::runtime_error("Unsupported checkpoint architecture: " + path);
  }
  ModelBundle bundle;
  bundle.num_blocks = num_blocks;
  bundle.model_sha256 = StringField(meta, "model_sha256");
  bundle.model = std::make_shared<SharedDunePolicyValueNetImpl>(
      observation_dim, hidden_dim, kNumDistinctPlayerActions, num_blocks,
      false, false, 0, true, false);
  LoadModelCheckpointRobust(bundle.model, path, torch::Device(torch::kCPU));
  bundle.model->to(device);
  bundle.model->eval();
  for (auto& parameter : bundle.model->parameters()) parameter.set_requires_grad(false);
  bundle.mutex = std::make_unique<std::shared_mutex>();
  bundle.coordinator = std::make_shared<BatchedEvaluator>(
      bundle.model, o.batch_size, 1, device, bundle.mutex.get(), 10.0f,
      false, true, false, o.rollout_amp, o.allow_tf32);
  bundle.evaluator = std::make_shared<BatchedNNEvaluator>(bundle.coordinator, 10.0f);
  std::cout << "[MODEL] " << path << " blocks=" << num_blocks
            << " obs=" << observation_dim << " precision="
            << (o.rollout_amp ? "AMP" : "FP32") << " tf32=" << o.allow_tf32
            << " device=" << device << '\n' << std::flush;
  return bundle;
}

struct RunControl {
  Clock::time_point start = Clock::now();
  double max_seconds;
  std::atomic<bool> failed{false};
  bool Stop() const {
    return interrupted || failed.load() ||
           (max_seconds > 0 && SecondsSince(start) >= max_seconds);
  }
};

bool IsLeader(const State& state) {
  const auto& dune = static_cast<const DuneImperiumState&>(state);
  return dune.phase() == GamePhase::kLeaderDraft ||
         dune.phase() == GamePhase::kLeaderOfferChance;
}

int PriorRank(const ActionsAndProbs& prior, const std::vector<Action>& legal,
              Action chosen) {
  const Action raw = PickGreedyAction(prior, legal);
  if (chosen == raw) return 1;
  std::vector<std::pair<double, Action>> scored;
  for (Action a : legal) {
    if (a == raw) continue;
    double probability = 0;
    for (const auto& item : prior) if (item.first == a) probability = item.second;
    scored.emplace_back(probability, a);
  }
  std::sort(scored.rbegin(), scored.rend());
  for (size_t i = 0; i < scored.size(); ++i) if (scored[i].second == chosen) return i + 2;
  throw std::runtime_error("Controller selected an illegal action");
}

CompoundSearchDecisionResult SearchDecision(
    const State& state, Player seat, TurnSearchTree* tree,
    const std::vector<std::shared_ptr<BatchedNNEvaluator>>& evaluators,
    const Options& o, uint64_t seed) {
  if (o.mode == Mode::kShAdaptive || o.mode == Mode::kShAdaptiveLeader ||
      o.mode == Mode::kFullShAdaptiveLeader) {
    return SuccessiveHalvingRolloutSearchDecision(state, seat, tree, evaluators,
                                                  o.rollouts, kOverrideMargin, seed);
  }
  if (o.mode == Mode::kAdaptive || o.mode == Mode::kAdaptiveLeader) {
    return AdaptiveRolloutSearchDecision(state, seat, tree, evaluators,
                                        o.rollouts, kOverrideMargin, seed);
  }
  if (o.mode == Mode::kWide) {
    return WideRolloutSearchDecision(state, seat, tree, evaluators,
                                    o.rollouts, kOverrideMargin, seed);
  }
  return CompoundRolloutSearchDecision(
      state, seat, tree, evaluators, 3, o.rollouts, kOverrideMargin,
      TreeReuseMode::kAdditive, seed, SearchPairingMode::kUnpaired,
      SearchTruncationMode::kTerminal);
}

struct GameOutcome {
  int game_id = 0;
  uint64_t seed = 0;
  Player seat = 0;
  std::vector<double> returns;
  std::vector<int> vps;
  int placement = 0;
  int ending_round = 0;
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
};

std::optional<GameOutcome> PlayGame(
    int game_id, const Game& game, const ModelBundle& candidate,
    const ModelBundle& opponent, const Options& o, const RunControl& control) {
  GameOutcome out;
  out.game_id = game_id;
  out.seat = game_id % 4;
  out.seed = dune_seed::DeriveSeed(o.master_seed, dune_seed::kStreamSearchSampling, game_id);
  const auto start = Clock::now();
  auto state = game.NewInitialState();
  std::mt19937 chance_rng(out.seed);
  TurnSearchTree tree;
  Player last_player = kInvalidPlayer;
  uint64_t search_seed_counter = 0;
  const std::vector<std::shared_ptr<BatchedNNEvaluator>> simulations{candidate.evaluator};

  while (!state->IsTerminal()) {
    if (control.Stop()) return std::nullopt;
    if (state->IsChanceNode()) {
      const auto& before = static_cast<const DuneImperiumState&>(*state);
      const int hand = before.GetPlayerCardsInHand(out.seat);
      const int intrigues = before.GetPlayerIntrigues(out.seat).size();
      Action action = SampleAction(state->ChanceOutcomes(),
          std::generate_canonical<double, 53>(chance_rng)).first;
      state->ApplyAction(action);
      const auto& after = static_cast<const DuneImperiumState&>(*state);
      if (after.GetPlayerCardsInHand(out.seat) > hand ||
          static_cast<int>(after.GetPlayerIntrigues(out.seat).size()) > intrigues) {
        tree.Reset();
        ++out.new_info_resets;
      }
      continue;
    }
    const Player current = state->CurrentPlayer();
    if (current != last_player) {
      tree.Reset();
      last_player = current;
    }
    const auto legal = state->LegalActions();
    if (legal.empty()) throw std::runtime_error("Nonterminal state has no legal actions");
    auto& evaluator = current == out.seat ? candidate.evaluator : opponent.evaluator;
    const auto prior = evaluator->Prior(*state);
    if (prior.empty()) throw std::runtime_error("Empty neural prior");
    const Action raw = PickGreedyAction(prior, legal);
    Action chosen = raw;
    if (current == out.seat) {
      ++out.decisions;
      if ((o.mode == Mode::kLeader || o.mode == Mode::kAdaptiveLeader ||
           o.mode == Mode::kShAdaptiveLeader ||
           o.mode == Mode::kFullShAdaptiveLeader) &&
          IsLeader(*state) && legal.size() > 1) {
        const auto search_start = Clock::now();
        const int rollouts = (o.mode == Mode::kLeader ? o.rollouts : std::min(32, o.rollouts));
        const uint64_t leader_seed =
            dune_seed::DeriveSeed(out.seed, 0x4C454144ULL, current);
        const auto result = LeaderPickRolloutSearchDecision(
            *state, current, simulations, rollouts, 0.10, leader_seed);
        out.search_seconds += SecondsSince(search_start);
        if (control.Stop()) return std::nullopt;
        chosen = result.action;
        ++out.searches;
        out.overrides += result.overridden;
        out.new_rollouts += result.new_rollouts_executed;
      } else {
        const bool search_allowed =
            o.mode != Mode::kRaw && o.mode != Mode::kLeader && !IsLeader(*state);
        if (o.mode == Mode::kFullShAdaptiveLeader) {
          const bool should_search =
              search_allowed && IsFullStrategicState(*state, current, prior, 0.020);
          if (should_search) {
            const auto search_start = Clock::now();
            const uint64_t seed = dune_seed::DeriveSeed(out.seed, current, search_seed_counter++);
            const auto result = SearchDecision(*state, current, &tree, simulations, o, seed);
            out.search_seconds += SecondsSince(search_start);
            if (control.Stop()) return std::nullopt;
            chosen = result.action;
            ++out.searches;
            out.overrides += result.overridden;
            out.new_rollouts += result.new_rollouts_executed;
            out.reused_rollouts += result.prior_rollouts_reused;
            out.prior_rank4plus_selections += result.overridden && PriorRank(prior, legal, chosen) >= 4;
          }
        } else if (search_allowed && IsHighStakesStrategicState(*state, current) && legal.size() > 1) {
          const auto search_start = Clock::now();
          const uint64_t seed = dune_seed::DeriveSeed(out.seed, current, search_seed_counter++);
          const auto result = SearchDecision(*state, current, &tree, simulations, o, seed);
          out.search_seconds += SecondsSince(search_start);
          if (control.Stop()) return std::nullopt;
          chosen = result.action;
          ++out.searches;
          out.overrides += result.overridden;
          out.new_rollouts += result.new_rollouts_executed;
          out.reused_rollouts += result.prior_rollouts_reused;
          if (o.mode == Mode::kWide && legal.size() > 3) {
            ++out.wide_roots;
          }
          out.prior_rank4plus_selections += result.overridden && PriorRank(prior, legal, chosen) >= 4;
        } else if (search_allowed && tree.current_node() && !tree.current_node()->children.empty()) {
          double best = -1e9;
          for (Action a : legal) {
            if (tree.HasPriorData(a) && tree.GetMeanUtility(a) > best) {
              best = tree.GetMeanUtility(a);
              chosen = a;
            }
          }
        }
      }
      if (o.mode != Mode::kRaw) {
        tree.Advance(chosen);
        if (chosen == kActionEndTurn) tree.Reset();
      }
    }
    if (std::find(legal.begin(), legal.end(), chosen) == legal.end()) {
      throw std::runtime_error("Selected action is not legal");
    }
    const auto& before = static_cast<const DuneImperiumState&>(*state);
    const int hand = current == out.seat ? before.GetPlayerCardsInHand(out.seat) : 0;
    const int intrigues = current == out.seat ? before.GetPlayerIntrigues(out.seat).size() : 0;
    state->ApplyAction(chosen);
    const auto& after = static_cast<const DuneImperiumState&>(*state);
    if (current == out.seat && chosen != kActionEndTurn &&
        (after.GetPlayerCardsInHand(out.seat) > hand ||
         static_cast<int>(after.GetPlayerIntrigues(out.seat).size()) > intrigues)) {
      tree.Reset();
      ++out.new_info_resets;
    }
  }
  out.returns = state->Returns();
  if (out.returns.size() != 4 ||
      !std::all_of(out.returns.begin(), out.returns.end(), [](double r) { return std::isfinite(r); })) {
    throw std::runtime_error("Invalid terminal returns");
  }
  out.placement = 1;
  for (double r : out.returns) out.placement += r > out.returns[out.seat];
  const auto& dune = static_cast<const DuneImperiumState&>(*state);
  for (int p = 0; p < 4; ++p) out.vps.push_back(dune.GetPlayerVp(p));
  out.ending_round = dune.GetCurrentRound() - 1;
  out.game_seconds = SecondsSince(start);
  return out;
}

json::Object BaseRecord(const Options& o, const std::string& kind) {
  return {{"record_type", kind}, {"config_key", o.config_key}, {"mode", o.mode_name},
          {"controller_version", kControllerVersion}, {"simulation_policy", kSimulationPolicy},
          {"model_checkpoint", o.model_path}, {"opponent_checkpoint", o.opponent_path},
          {"master_seed", std::to_string(o.master_seed)}, {"rollouts_per_action", o.rollouts},
          {"margin", kOverrideMargin}, {"rollout_amp", o.rollout_amp},
          {"allow_tf32", o.allow_tf32}, {"device", o.device_name},
          {"threads", o.threads}, {"target_batch_size", o.batch_size}};
}

json::Object GameRecord(const GameOutcome& out, const Options& o) {
  auto row = BaseRecord(o, "game");
  row["game_id"] = out.game_id;
  row["seed"] = std::to_string(out.seed);
  row["seat"] = out.seat;
  json::Array returns, vps;
  for (double r : out.returns) returns.emplace_back(r);
  for (int vp : out.vps) vps.emplace_back(vp);
  row["returns"] = returns;
  row["final_vps"] = vps;
  row["utility"] = out.returns[out.seat];
  row["placement"] = out.placement;
  row["ending_round"] = out.ending_round;
  row["searches"] = out.searches;
  row["overrides"] = out.overrides;
  row["new_rollouts"] = out.new_rollouts;
  row["reused_rollouts"] = out.reused_rollouts;
  row["new_info_resets"] = out.new_info_resets;
  row["decisions"] = out.decisions;
  row["wide_roots"] = out.wide_roots;
  row["prior_rank4plus_selections"] = out.prior_rank4plus_selections;
  row["search_wall_seconds"] = out.search_seconds;
  row["game_wall_seconds"] = out.game_seconds;
  return row;
}

std::set<int> ReadCompleted(const Options& o) {
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
    }
  }
  return completed;
}

struct RootCase {
  int root_id;
  int game_id;
  int seat;
  int round;
  std::string decision_type;
  std::vector<Action> history;
};

std::vector<RootCase> ReadRoots(const Options& o) {
  const auto value = ReadJson(o.roots_path);
  if (!value.IsObject()) throw std::runtime_error("Root corpus must be an object");
  const auto& object = value.GetObject();
  auto roots = object.find("roots");
  if (roots == object.end() || !roots->second.IsArray()) throw std::runtime_error("Missing roots array");
  std::map<std::pair<int, std::string>, std::vector<RootCase>> strata;
  std::set<int> ids;
  for (const auto& item : roots->second.GetArray()) {
    if (!item.IsObject()) throw std::runtime_error("Root must be an object");
    const auto& row = item.GetObject();
    RootCase root{IntField(row, "root_id"), IntField(row, "game_id"),
                  IntField(row, "search_seat"), IntField(row, "round"),
                  StringField(row, "decision_type"), {}};
    if (!ids.insert(root.root_id).second || root.seat < 0 || root.seat >= 4) {
      throw std::runtime_error("Duplicate root ID or invalid seat");
    }
    auto history = row.find("action_history");
    if (history == row.end() || !history->second.IsArray()) throw std::runtime_error("Missing root history");
    for (const auto& action : history->second.GetArray()) {
      if (!action.IsInt()) throw std::runtime_error("Invalid action in root history");
      root.history.push_back(action.GetInt());
    }
    const int band = root.round <= 3 ? 0 : root.round <= 6 ? 1 : 2;
    strata[{band, root.decision_type}].push_back(std::move(root));
  }
  for (auto& [key, bucket] : strata) {
    std::sort(bucket.begin(), bucket.end(), [](const auto& a, const auto& b) { return a.root_id < b.root_id; });
  }
  std::vector<RootCase> result;
  for (size_t offset = 0;; ++offset) {
    bool added = false;
    for (const auto& [key, bucket] : strata) {
      if (offset < bucket.size()) {
        result.push_back(bucket[offset]);
        added = true;
        if (o.max_roots > 0 && result.size() == static_cast<size_t>(o.max_roots)) return result;
      }
    }
    if (!added) return result;
  }
}

int RunRoots(const Options& o, const Game& game, const ModelBundle& candidate,
             const RunControl& control, const std::set<int>& completed,
             std::ofstream& output) {
  const auto roots = ReadRoots(o);
  const std::vector<std::shared_ptr<BatchedNNEvaluator>> evaluators{candidate.evaluator};
  int done = 0;
  for (const auto& root : roots) {
    if (completed.count(root.root_id)) { ++done; continue; }
    if (control.Stop()) break;
    auto state = game.NewInitialState();
    for (size_t index = 0; index < root.history.size(); ++index) {
      const Action action = root.history[index];
      const auto legal = state->LegalActions();
      if (std::find(legal.begin(), legal.end(), action) == legal.end()) {
        throw std::runtime_error("Illegal root history: root_id=" + std::to_string(root.root_id) +
            " index=" + std::to_string(index) + " action=" + std::to_string(action) +
            " current_player=" + std::to_string(state->CurrentPlayer()));
      }
      state->ApplyAction(action);
    }
    if (state->IsTerminal() || state->IsChanceNode() || state->CurrentPlayer() != root.seat) {
      throw std::runtime_error("Replayed root has the wrong actor or is not a player decision");
    }
    const auto start = Clock::now();
    const auto nn_start = g_total_network_evaluations.load();
    const auto legal = state->LegalActions();
    const auto prior = candidate.evaluator->Prior(*state);
    const Action raw = PickGreedyAction(prior, legal);
    const uint64_t seed = dune_seed::DeriveSeed(o.master_seed, dune_seed::kStreamSearchSampling, root.root_id);
    CompoundSearchDecisionResult result;
    result.action = raw;
    TurnSearchTree tree;
    const bool should_search =
        o.mode != Mode::kRaw && o.mode != Mode::kLeader && !IsLeader(*state) &&
        (o.mode == Mode::kFullShAdaptiveLeader
             ? IsFullStrategicState(*state, root.seat, prior, 0.020)
             : (IsHighStakesStrategicState(*state, root.seat) && legal.size() > 1));
    if (should_search) {
      result = SearchDecision(*state, root.seat, &tree, evaluators, o, seed);
    }
    if (control.Stop()) break;
    auto row = BaseRecord(o, "root");
    row["roots_json"] = o.roots_path;
    row["root_id"] = root.root_id;
    row["game_id"] = root.game_id;
    row["seed"] = std::to_string(seed);
    row["seat"] = root.seat;
    row["round"] = root.round;
    row["decision_type"] = root.decision_type;
    row["chosen_action"] = result.action;
    row["raw_action"] = raw;
    row["selected_prior_rank"] = PriorRank(prior, legal, result.action);
    const bool wide_applicable = should_search && o.mode == Mode::kWide && legal.size() > 3;
    row["wide_applicable"] = wide_applicable;
    row["overridden"] = result.overridden;
    row["new_rollouts"] = result.new_rollouts_executed;
    row["reused_rollouts"] = result.prior_rollouts_reused;
    row["search_wall_seconds"] = SecondsSince(start);
    row["network_evaluations"] = static_cast<int64_t>(g_total_network_evaluations.load() - nn_start);
    json::Array utilities;
    for (const auto& [action, utility] : result.candidate_utilities) {
      utilities.emplace_back(json::Array{action, utility});
    }
    row["candidate_utilities"] = utilities;
    output << json::ToString(row) << '\n' << std::flush;
    if (!output) throw std::runtime_error("Failed to append root output");
    std::cout << "[ROOT] " << ++done << '/' << roots.size() << " id=" << root.root_id
              << " elapsed=" << SecondsSince(control.start) << "s\n" << std::flush;
  }
  return done == static_cast<int>(roots.size()) ? 0 : 2;
}
}  // namespace

int main(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);
  std::signal(SIGINT, HandleSignal);
  std::signal(SIGTERM, HandleSignal);
  try {
    const Options o = ReadOptions();
    RunControl control;
    control.max_seconds = o.max_seconds;
    const auto completed = ReadCompleted(o);
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
        std::cout << "[DONE] requested games already complete: " << skipped << '\n';
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

    std::atomic<size_t> next{0};
    std::mutex writer;
    int done = 0;
    std::string error;
    const auto play_start = Clock::now();
    auto worker = [&]() {
      try {
        while (!control.Stop()) {
          const size_t index = next.fetch_add(1);
          if (index >= pending.size()) break;
          auto result = PlayGame(pending[index], *game, candidate, opponent, o, control);
          if (!result) break;
          const auto row = GameRecord(*result, o);
          std::lock_guard<std::mutex> lock(writer);
          output << json::ToString(row) << '\n' << std::flush;
          if (!output) throw std::runtime_error("Failed to append completed game");
          ++done;
          const double elapsed = SecondsSince(play_start);
          const double eta = elapsed / done * (pending.size() - done);
          std::cout << "[GAME] " << done + skipped << '/' << o.games
                    << " id=" << result->game_id << " seat=" << result->seat
                    << " placement=" << result->placement << " utility="
                    << result->returns[result->seat] << " elapsed=" << elapsed
                    << "s eta=" << eta << "s\n" << std::flush;
        }
      } catch (const std::exception& e) {
        control.failed.store(true);
        std::lock_guard<std::mutex> lock(writer);
        if (error.empty()) error = e.what();
      }
    };
    std::vector<std::thread> workers;
    for (int i = 0; i < o.threads; ++i) workers.emplace_back(worker);
    for (auto& thread : workers) thread.join();
    std::cout << "[DONE] completed=" << done + skipped << '/' << o.games
              << " newly_completed=" << done << " skipped=" << skipped
              << " elapsed=" << SecondsSince(control.start)
              << "s network_evaluations=" << g_total_network_evaluations.load()
              << '\n' << std::flush;
    if (!error.empty()) throw std::runtime_error(error);
    return done + skipped == o.games ? 0 : 2;
  } catch (const std::exception& e) {
    std::cerr << "[ERROR] " << e.what() << '\n';
    return 1;
  }
}
