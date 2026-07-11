// Reproducible population evaluator for Dune Imperium.
//
// Replaces the ad-hoc dune_eval_1000.cc with deterministic, domain-separated
// seeding via dune_seed_utils.h. Produces per-game JSONL and aggregate JSON
// with Wilson confidence intervals.
//
// Key properties:
//   - Per-game seeds derived from (base_seed, domain, episode_id, stream)
//   - Deterministic seat rotation and opponent assignment
//   - Heterogeneous opponent architectures supported
//   - Greedy / stochastic policy selection
//   - Thread-count reproducibility (results identical regardless of --threads)

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <memory>
#include <mutex>
#include <random>
#include <shared_mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>
#include <unordered_map>

#include "open_spiel/spiel.h"
#include "open_spiel/games/dune_imperium/dune_imperium.h"
#include "open_spiel/abseil-cpp/absl/flags/flag.h"
#include "open_spiel/abseil-cpp/absl/flags/parse.h"
#include "open_spiel/abseil-cpp/absl/strings/str_format.h"
#include <torch/torch.h>

#include "dune_network.h"
#include "dune_seed_utils.h"
#include "open_spiel/utils/json.h"

// ---------------------------------------------------------------------------
// Flags
// ---------------------------------------------------------------------------
ABSL_FLAG(std::string, model_checkpoint, "",
          "Path to the model checkpoint to evaluate.");
ABSL_FLAG(std::string, opponent_checkpoints, "",
          "Comma-separated paths to opponent model checkpoints. "
          "Empty or \"random\" for random opponents.");
ABSL_FLAG(int, num_games, 5000,
          "Total number of evaluation games to play.");
ABSL_FLAG(uint64_t, base_seed, 42,
          "Base seed for deterministic evaluation.");
ABSL_FLAG(std::string, domain, "EVAL_BASELINE",
          "Evaluation domain: EVAL_BASELINE, EVAL_DEV, or EVAL_FINAL.");
ABSL_FLAG(bool, greedy, true,
          "If true, use argmax policy; if false, sample from softmax.");
ABSL_FLAG(int, threads, 0,
          "Number of game worker threads. 0 = auto-detect.");
ABSL_FLAG(int, hidden_dim, 2048,
          "Hidden dimension of the evaluated model.");
ABSL_FLAG(int, num_blocks, 8,
          "Number of residual blocks in the evaluated model.");
ABSL_FLAG(int, opp_hidden_dim, -1,
          "Opponent hidden dim. -1 = same as --hidden_dim.");
ABSL_FLAG(int, opp_num_blocks, -1,
          "Opponent num blocks. -1 = same as --num_blocks.");
ABSL_FLAG(std::string, output_dir, "",
          "Directory for per-game JSONL and aggregate JSON. "
          "Empty = stdout only.");
ABSL_FLAG(float, temperature, 1.0f,
          "Softmax temperature for stochastic policy (--greedy=false).");
ABSL_FLAG(bool, deterministic_eval, true,
          "If true, use batch-1 mutex-serialized inference for strict bitwise "
          "thread-count reproducibility. Much slower than batched mode.");

namespace open_spiel {
namespace {

using dune_imperium::DuneImperiumState;
using dune_imperium::kNumPlayers;

constexpr float kEvalLogitCap = 10.0f;

// ---------------------------------------------------------------------------
// Domain resolution
// ---------------------------------------------------------------------------
uint64_t ResolveDomain(const std::string& domain_str) {
  if (domain_str == "EVAL_BASELINE") return dune_seed::kDomainEvalBaseline;
  if (domain_str == "EVAL_DEV")      return dune_seed::kDomainEvalDev;
  if (domain_str == "EVAL_FINAL")    return dune_seed::kDomainEvalFinal;
  std::cerr << "Unknown domain: " << domain_str
            << ". Must be EVAL_BASELINE, EVAL_DEV, or EVAL_FINAL.\n";
  std::exit(1);
}

// ---------------------------------------------------------------------------
// Wilson score interval
// ---------------------------------------------------------------------------
struct WilsonCI {
  double point;
  double lower;
  double upper;
};

WilsonCI WilsonScore(int successes, int trials, double z = 1.96) {
  if (trials == 0) return {0.0, 0.0, 0.0};
  double n = static_cast<double>(trials);
  double p_hat = static_cast<double>(successes) / n;
  double z2 = z * z;
  double denom = 1.0 + z2 / n;
  double center = (p_hat + z2 / (2.0 * n)) / denom;
  double margin = z * std::sqrt((p_hat * (1.0 - p_hat) + z2 / (4.0 * n)) / n)
                  / denom;
  return {p_hat, std::max(0.0, center - margin),
                 std::min(1.0, center + margin)};
}

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
};

// ---------------------------------------------------------------------------
// Comma-separated path splitting
// ---------------------------------------------------------------------------
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



// ---------------------------------------------------------------------------
// Auto-detect model hidden_dim and num_blocks from JSON manifest or weight keys
// ---------------------------------------------------------------------------
bool DetectModelDimensions(const std::string& model_path, int* hidden_dim, int* num_blocks) {
  // 1. Try to find a JSON file
  std::string json_path = "";
  std::filesystem::path p(model_path);
  std::filesystem::path p_json = p;
  p_json.replace_extension(".json");
  if (std::filesystem::exists(p_json)) {
    json_path = p_json.string();
  } else {
    std::filesystem::path p_parent_manifest = p.parent_path() / "manifest.json";
    if (std::filesystem::exists(p_parent_manifest)) {
      json_path = p_parent_manifest.string();
    }
  }

  if (!json_path.empty()) {
    try {
      std::ifstream f(json_path);
      if (f) {
        std::string content((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
        auto val_opt = json::FromString(content);
        if (val_opt.has_value() && val_opt->IsObject()) {
          const auto& obj = val_opt->GetObject();
          auto it_hd = obj.find("hidden_dim");
          auto it_nb = obj.find("num_blocks");
          if (it_hd != obj.end() && it_hd->second.IsInt() &&
              it_nb != obj.end() && it_nb->second.IsInt()) {
            *hidden_dim = it_hd->second.GetInt();
            *num_blocks = it_nb->second.GetInt();
            return true;
          }
          auto it_arch = obj.find("architecture");
          if (it_arch != obj.end() && it_arch->second.IsObject()) {
            const auto& arch_obj = it_arch->second.GetObject();
            auto it_arch_hd = arch_obj.find("hidden_dim");
            auto it_arch_nb = arch_obj.find("num_blocks");
            if (it_arch_hd != arch_obj.end() && it_arch_hd->second.IsInt() &&
                it_arch_nb != arch_obj.end() && it_arch_nb->second.IsInt()) {
              *hidden_dim = it_arch_hd->second.GetInt();
              *num_blocks = it_arch_nb->second.GetInt();
              return true;
            }
          }
        }
      }
    } catch (...) {
      // Fallback
    }
  }

  // 2. Fall back to weights key-based inspection
  try {
    torch::serialize::InputArchive archive;
    archive.load_from(model_path, torch::kCPU);
    
    torch::serialize::InputArchive input_layer_archive;
    archive.read("input_layer", input_layer_archive);
    torch::Tensor weight;
    input_layer_archive.read("weight", weight);
    *hidden_dim = weight.size(0);

    int blocks = 0;
    while (true) {
      torch::serialize::InputArchive res_archive;
      std::string block_name = "res" + std::to_string(blocks + 1);
      try {
        archive.read(block_name, res_archive);
        blocks++;
      } catch (...) {
        break;
      }
    }
    *num_blocks = blocks;
    
    std::cerr << "WARNING: Manifest JSON not found for model checkpoint " << model_path 
              << ". Auto-detected architecture (hidden_dim=" << *hidden_dim 
              << ", num_blocks=" << *num_blocks << ") from weight keys. "
              << "Note: this inspection logic is coupled to the SharedDunePolicyValueNetImpl class architecture.\n";
    return true;
  } catch (const std::exception& e) {
    std::cerr << "ERROR: Failed to detect model dimensions or load archive from " << model_path 
              << ": " << e.what() << "\n";
    return false;
  }
}

// ---------------------------------------------------------------------------
// Worker thread: plays games with deterministic per-game seeds.
// Results are stored in a pre-allocated vector indexed by episode_id,
// guaranteeing thread-count-independent output.
// ---------------------------------------------------------------------------
void WorkerThread(
    int /*thread_id*/,
    std::shared_ptr<const Game> game,
    std::shared_ptr<IGameEvaluator> model_evaluator,
    const std::vector<std::shared_ptr<IGameEvaluator>>& opponent_evaluators,
    const std::vector<std::string>& opponent_names,
    int64_t obs_size,
    bool provides_info_state_tensor,
    bool provides_observations_tensor,
    std::atomic<int>& next_game_id,
    int total_games,
    uint64_t base_seed,
    uint64_t domain,
    bool greedy,
    float temperature,
    std::vector<GameResult>& results) {

  // Pre-allocate observation buffer reused across all games
  std::vector<float> obs(obs_size, 0.0f);

  while (true) {
    int episode_id = next_game_id++;
    if (episode_id >= total_games) break;

    // --- Round-robin seat assignment for exact balance ---
    int model_player = episode_id % kNumPlayers;

    // --- Per-game chance RNG ---
    uint64_t chance_seed = dune_seed::DeriveSeed(
        base_seed, domain, static_cast<uint64_t>(episode_id),
        dune_seed::kStreamChance);
    std::mt19937 chance_rng = dune_seed::MakeRng32(chance_seed);

    // --- Per-player policy RNGs (for stochastic mode) ---
    std::array<std::mt19937_64, 4> policy_rngs;
    for (int p = 0; p < kNumPlayers; ++p) {
      uint64_t policy_stream = dune_seed::kStreamPolicyPlayer0 +
                               static_cast<uint64_t>(p);
      uint64_t pseed = dune_seed::DeriveSeed(
          base_seed, domain, static_cast<uint64_t>(episode_id),
          policy_stream);
      policy_rngs[p] = dune_seed::MakeRng64(pseed);
    }

    // --- Per-game opponent assignment RNG ---
    std::vector<size_t> player_opp_idx(kNumPlayers, 0);
    if (!opponent_evaluators.empty()) {
      uint64_t opp_assign_seed = dune_seed::DeriveSeed(
          base_seed, domain, static_cast<uint64_t>(episode_id),
          dune_seed::kStreamOpponentAssign);
      std::mt19937_64 opp_rng = dune_seed::MakeRng64(opp_assign_seed);
      std::uniform_int_distribution<size_t> opp_dist(0, opponent_evaluators.size() - 1);

      for (int p = 0; p < kNumPlayers; ++p) {
        if (p == model_player) continue;
        player_opp_idx[p] = opp_dist(opp_rng);
      }
    }

    std::unique_ptr<State> state = game->NewInitialState();
    int game_length = 0;

    while (!state->IsTerminal()) {
      ++game_length;
      if (game_length > 5000) {
        std::cerr << "Possible infinite loop in episode " << episode_id
                  << "! Length: " << game_length
                  << " Player: " << state->CurrentPlayer() << std::endl;
        std::abort();
      }

      if (state->IsChanceNode()) {
        auto outcomes = state->ChanceOutcomes();
        Action action;
        if (game->GetType().chance_mode ==
            GameType::ChanceMode::kSampledStochastic) {
          action = outcomes.front().first;
        } else {
          action = SampleAction(outcomes, chance_rng).first;
        }
        state->ApplyAction(action);
        continue;
      }

      Player current_player = state->CurrentPlayer();
      std::vector<Action> legal_actions = state->LegalActions();
      if (legal_actions.empty()) {
        std::cerr << "Empty legal actions in episode " << episode_id << "!\n";
        break;
      }

      Action chosen_action = -1;
      bool use_model = (current_player == model_player);
      bool use_opponent_model = (!use_model && !opponent_evaluators.empty());

      if (use_model || use_opponent_model) {
        // Fill observation buffer
        std::fill(obs.begin(), obs.end(), 0.0f);
        if (provides_info_state_tensor) {
          state->InformationStateTensor(current_player, absl::MakeSpan(obs));
        } else if (provides_observations_tensor) {
          state->ObservationTensor(current_player, absl::MakeSpan(obs));
        }

        // Select evaluator
        std::shared_ptr<IGameEvaluator> evaluator = model_evaluator;
        if (!use_model) {
          size_t idx = player_opp_idx[current_player];
          evaluator = opponent_evaluators[idx];
        }

        EvalResult result = evaluator->Evaluate(obs);
        CenterAndCapLegalLogits(result.logits, legal_actions, kEvalLogitCap);

        if (greedy || legal_actions.size() == 1) {
          // Argmax
          chosen_action = legal_actions.front();
          float max_logit = -1e9f;
          for (Action a : legal_actions) {
            if (result.logits[a] > max_logit) {
              max_logit = result.logits[a];
              chosen_action = a;
            }
          }
        } else {
          // Stochastic: softmax with temperature, then sample
          double max_logit = -1e30;
          for (Action a : legal_actions) {
            if (result.logits[a] > max_logit) {
              max_logit = result.logits[a];
            }
          }
          std::vector<double> probs(legal_actions.size());
          double sum = 0.0;
          for (size_t i = 0; i < legal_actions.size(); ++i) {
            probs[i] = std::exp(
                (static_cast<double>(result.logits[legal_actions[i]]) -
                 max_logit) / static_cast<double>(temperature));
            sum += probs[i];
          }
          for (auto& p : probs) p /= sum;

          std::uniform_real_distribution<double> dist(0.0, 1.0);
          double r = dist(policy_rngs[current_player]);
          double cumulative = 0.0;
          chosen_action = legal_actions.back();
          for (size_t i = 0; i < legal_actions.size(); ++i) {
            cumulative += probs[i];
            if (r < cumulative) {
              chosen_action = legal_actions[i];
              break;
            }
          }
        }
      } else {
        // Random opponent (no model loaded)
        std::uniform_int_distribution<size_t> dist(
            0, legal_actions.size() - 1);
        chosen_action = legal_actions[dist(policy_rngs[current_player])];
      }

      state->ApplyAction(chosen_action);
    }

    // --- Collect results ---
    std::vector<double> returns = state->Returns();
    auto* dune_state = dynamic_cast<const DuneImperiumState*>(state.get());

    // Map returns cleanly directly to placement:
    // 2.25 -> 1st, 0.25 -> 2nd, -0.75 -> 3rd, -1.75 -> 4th
    int placement = 4;
    double r = returns[model_player];
    if (std::abs(r - 2.25) < 1e-4) {
      placement = 1;
    } else if (std::abs(r - 0.25) < 1e-4) {
      placement = 2;
    } else if (std::abs(r - (-0.75)) < 1e-4) {
      placement = 3;
    } else if (std::abs(r - (-1.75)) < 1e-4) {
      placement = 4;
    }

    GameResult& gr = results[episode_id];
    gr.episode_id = episode_id;
    gr.seat = model_player;

    // Track opponent names for seats other than model_player
    gr.opponents.clear();
    for (int p = 0; p < kNumPlayers; ++p) {
      if (p == model_player) continue;
      if (opponent_names.empty()) {
        gr.opponents.push_back("random");
      } else {
        size_t opp_idx = player_opp_idx[p];
        gr.opponents.push_back(opponent_names[opp_idx]);
      }
    }

    gr.placement = placement;
    gr.game_return = returns[model_player];
    if (dune_state) {
      gr.ending_round = dune_state->IsTerminal() ? (dune_state->GetCurrentRound() - 1)
                                                 : dune_state->GetCurrentRound();
    } else {
      gr.ending_round = -1;
    }
    gr.current_vp = dune_state
                    ? dune_state->GetPlayerVpForTesting(model_player)
                    : -1;
    gr.final_scored_vp = dune_state
                         ? dune_state->FinalScoredVp(model_player)
                         : -1;
  }
}

// ---------------------------------------------------------------------------
// Main evaluation driver
// ---------------------------------------------------------------------------
void RunEvaluation() {
  const std::string model_checkpoint = absl::GetFlag(FLAGS_model_checkpoint);
  const std::string opponent_str = absl::GetFlag(FLAGS_opponent_checkpoints);
  const int total_games = absl::GetFlag(FLAGS_num_games);
  const uint64_t base_seed = absl::GetFlag(FLAGS_base_seed);
  const std::string domain_str = absl::GetFlag(FLAGS_domain);
  const bool greedy = absl::GetFlag(FLAGS_greedy);
  const int requested_threads = absl::GetFlag(FLAGS_threads);
  const int hidden_dim = absl::GetFlag(FLAGS_hidden_dim);
  const int num_blocks = absl::GetFlag(FLAGS_num_blocks);
  int opp_hidden_dim = absl::GetFlag(FLAGS_opp_hidden_dim);
  int opp_num_blocks = absl::GetFlag(FLAGS_opp_num_blocks);
  const std::string output_dir = absl::GetFlag(FLAGS_output_dir);
  const float temperature = absl::GetFlag(FLAGS_temperature);

  if (model_checkpoint.empty()) {
    std::cerr << "Error: --model_checkpoint is required.\n";
    std::exit(1);
  }

  uint64_t domain = ResolveDomain(domain_str);
  if (opp_hidden_dim < 0) opp_hidden_dim = hidden_dim;
  if (opp_num_blocks < 0) opp_num_blocks = num_blocks;

  // Parse opponent paths
  std::vector<std::string> opponent_paths;
  if (!opponent_str.empty() && opponent_str != "random") {
    opponent_paths = SplitCommaSeparated(opponent_str);
  }
  bool use_opponent_model = !opponent_paths.empty();

  std::vector<std::string> opponent_names;
  if (use_opponent_model) {
    for (const std::string& opp_path : opponent_paths) {
      opponent_names.push_back(
          std::filesystem::path(opp_path).stem().string());
    }
  }

  // Initialize game
  std::shared_ptr<const Game> game = LoadGame("dune_imperium");

  int64_t obs_size = 0;
  bool provides_info_state_tensor =
      game->GetType().provides_information_state_tensor;
  bool provides_observations_tensor =
      game->GetType().provides_observation_tensor;
  if (provides_info_state_tensor) {
    obs_size = game->InformationStateTensorSize();
  } else if (provides_observations_tensor) {
    obs_size = game->ObservationTensorSize();
  }
  if (obs_size == 0) {
    std::cerr << "Error: observation size is 0.\n";
    std::exit(1);
  }
  int64_t action_size = game->NumDistinctActions();

  // Determine thread count
  unsigned int hw_threads = std::thread::hardware_concurrency();
  if (hw_threads == 0) hw_threads = 4;
  unsigned int num_threads;
  if (requested_threads > 0) {
    num_threads = static_cast<unsigned int>(requested_threads);
  } else {
    num_threads = hw_threads * 4;
  }

  // Device setup
  torch::InferenceMode inference_guard;
  torch::Device device = torch::cuda::is_available()
                         ? torch::Device(torch::kCUDA)
                         : torch::Device(torch::kCPU);

  int eval_batch_size;
  int eval_timeout_ms;
  if (device.is_cuda()) {
    eval_batch_size = std::min(64u, num_threads);
    eval_timeout_ms = 1;
  } else {
    eval_batch_size = std::min(32u, num_threads);
    eval_timeout_ms = 2;
  }

  std::string device_name = device.is_cuda() ? "CUDA (GPU)" : "CPU";
  bool deterministic = absl::GetFlag(FLAGS_deterministic_eval);
  if (deterministic) {
    std::cout << "INFO: Running in deterministic mode (default). Use --deterministic_eval=false for faster batched evaluation.\n";
  }

  // Synchronization primitives (only used for the active evaluator mode)
  std::shared_mutex sync_mutex;  // For BatchedEvaluator (shared read lock)
  std::mutex eval_mutex;         // For DeterministicEvaluator (exclusive lock)

  // Load eval model with auto-detected dimensions
  int main_hidden_dim = hidden_dim;
  int main_num_blocks = num_blocks;
  if (!DetectModelDimensions(model_checkpoint, &main_hidden_dim, &main_num_blocks)) {
    SpielFatalError("Failed to detect model dimensions for main checkpoint: " + model_checkpoint);
  }

  auto model = std::make_shared<SharedDunePolicyValueNetImpl>(
      obs_size, main_hidden_dim, action_size, main_num_blocks);
  model->eval();
  {
    torch::serialize::InputArchive archive;
    archive.load_from(model_checkpoint, device);
    model->load(archive);
  }
  model->to(device);

  std::shared_ptr<IGameEvaluator> model_evaluator;
  if (deterministic) {
    model_evaluator = std::make_shared<DeterministicEvaluator>(
        model, device, &eval_mutex);
  } else {
    model_evaluator = std::make_shared<BatchedEvaluator>(
        model, eval_batch_size, eval_timeout_ms, device,
        &sync_mutex, 0.0f);
  }

  // Load opponent models with auto-detected dimensions
  struct OpponentMetadata {
    std::string path;
    int hidden_dim;
    int num_blocks;
  };
  std::vector<OpponentMetadata> opp_metadata;
  std::vector<std::shared_ptr<IGameEvaluator>> opponent_evaluators;
  for (const std::string& opp_path : opponent_paths) {
    int opp_detected_hidden = opp_hidden_dim;
    int opp_detected_blocks = opp_num_blocks;
    if (!DetectModelDimensions(opp_path, &opp_detected_hidden, &opp_detected_blocks)) {
      SpielFatalError("Failed to detect model dimensions for opponent checkpoint: " + opp_path);
    }
    opp_metadata.push_back({opp_path, opp_detected_hidden, opp_detected_blocks});

    auto opp_model = std::make_shared<SharedDunePolicyValueNetImpl>(
        obs_size, opp_detected_hidden, action_size, opp_detected_blocks);
    opp_model->eval();
    {
      torch::serialize::InputArchive archive;
      archive.load_from(opp_path, device);
      opp_model->load(archive);
    }
    opp_model->to(device);

    if (deterministic) {
      opponent_evaluators.push_back(std::make_shared<DeterministicEvaluator>(
          opp_model, device, &eval_mutex));
    } else {
      opponent_evaluators.push_back(std::make_shared<BatchedEvaluator>(
          opp_model, eval_batch_size, eval_timeout_ms, device,
          &sync_mutex, 0.0f));
    }
  }

  // --- Print configuration ---
  std::cout << "=== Dune Population Evaluator ===\n"
            << "Model:      " << model_checkpoint << "\n"
            << "Opponents:  "
            << (use_opponent_model ? opponent_str : "Random Agents") << "\n"
            << "Domain:     " << domain_str << "\n"
            << "Base seed:  " << base_seed << "\n"
            << "Games:      " << total_games << "\n"
            << "Greedy:     " << (greedy ? "true" : "false") << "\n"
            << "Temperature:" << temperature << "\n"
            << "Threads:    " << num_threads << "\n"
            << "Eval mode:  " << (deterministic ? "Deterministic (batch-1)" : "Batched") << "\n"
            << "Batch size: " << eval_batch_size << "\n"
            << "Device:     " << device_name << "\n"
            << "Hidden dim: " << hidden_dim << " / Blocks: " << num_blocks
            << "\n";
  if (use_opponent_model && (opp_hidden_dim != hidden_dim ||
                              opp_num_blocks != num_blocks)) {
    std::cout << "Opp dim:    " << opp_hidden_dim
              << " / Blocks: " << opp_num_blocks << "\n";
  }
  std::cout << std::endl;

  // --- Pre-allocate results (indexed by episode_id for determinism) ---
  std::vector<GameResult> results(total_games);

  auto start_time = std::chrono::steady_clock::now();

  // Launch worker threads
  std::atomic<int> next_game_id{0};
  std::vector<std::thread> threads;
  for (unsigned int t = 0; t < num_threads; ++t) {
    threads.emplace_back(
        WorkerThread, static_cast<int>(t), game, model_evaluator,
        std::cref(opponent_evaluators), std::cref(opponent_names), obs_size,
        provides_info_state_tensor, provides_observations_tensor,
        std::ref(next_game_id), total_games,
        base_seed, domain, greedy, temperature,
        std::ref(results));
  }
  for (auto& th : threads) {
    if (th.joinable()) th.join();
  }

  auto end_time = std::chrono::steady_clock::now();
  double elapsed_secs =
      std::chrono::duration<double>(end_time - start_time).count();

  // --- Write per-game JSONL ---
  std::ofstream jsonl_out;
  if (!output_dir.empty()) {
    std::filesystem::create_directories(output_dir);
    std::string jsonl_path =
        (std::filesystem::path(output_dir) / "games.jsonl").string();
    jsonl_out.open(jsonl_path);
    if (!jsonl_out.is_open()) {
      std::cerr << "Error: cannot open " << jsonl_path << " for writing.\n";
      std::exit(1);
    }
  }

  // Compute aggregates while writing JSONL
  int first_place_count = 0;
  double return_sum = 0.0;
  double vp_sum = 0.0;
  int ending_round_le7 = 0;
  int ending_round_le8 = 0;
  int placement_counts[4] = {0, 0, 0, 0};
  int wins_by_seat[4] = {0, 0, 0, 0};
  int games_by_seat[4] = {0, 0, 0, 0};

  for (int i = 0; i < total_games; ++i) {
    const GameResult& gr = results[i];

    // Aggregation
    if (gr.placement == 1) {
      first_place_count++;
      wins_by_seat[gr.seat]++;
    }
    return_sum += gr.game_return;
    vp_sum += gr.final_scored_vp;
    if (gr.ending_round <= 7) ending_round_le7++;
    if (gr.ending_round <= 8) ending_round_le8++;
    if (gr.placement >= 1 && gr.placement <= 4) {
      placement_counts[gr.placement - 1]++;
    }
    games_by_seat[gr.seat]++;

    // JSONL output with opponents serialized as a JSON array of strings
    if (jsonl_out.is_open()) {
      jsonl_out << "{\"episode_id\":" << gr.episode_id
                << ",\"seat\":" << gr.seat
                << ",\"opponents\":[";
      for (size_t idx = 0; idx < gr.opponents.size(); ++idx) {
        if (idx > 0) jsonl_out << ",";
        jsonl_out << "\"" << gr.opponents[idx] << "\"";
      }
      jsonl_out << "]"
                << ",\"placement\":" << gr.placement
                << ",\"return\":" << absl::StrFormat("%.4f", gr.game_return)
                << ",\"ending_round\":" << gr.ending_round
                << ",\"track_vp\":" << gr.current_vp
                << ",\"final_scored_vp\":" << gr.final_scored_vp
                << "}\n";
    }
  }

  if (jsonl_out.is_open()) {
    jsonl_out.close();
    std::cout << "Per-game results written to "
              << (std::filesystem::path(output_dir) / "games.jsonl").string()
              << "\n\n";
  }

  // --- Compute aggregate statistics ---
  double mean_return = total_games > 0 ? return_sum / total_games : 0.0;
  double mean_vp = total_games > 0 ? vp_sum / total_games : 0.0;

  // Confidence intervals
  WilsonCI first_place_ci = WilsonScore(first_place_count, total_games);
  WilsonCI round_le7_ci = WilsonScore(ending_round_le7, total_games);
  WilsonCI round_le8_ci = WilsonScore(ending_round_le8, total_games);

  // Return CI (normal approximation)
  double return_var = 0.0;
  double vp_var = 0.0;
  for (int i = 0; i < total_games; ++i) {
    double d = results[i].game_return - mean_return;
    return_var += d * d;
    double dv = results[i].final_scored_vp - mean_vp;
    vp_var += dv * dv;
  }
  double return_se = total_games > 1
      ? std::sqrt(return_var / (total_games - 1)) / std::sqrt(total_games)
      : 0.0;
  double vp_se = total_games > 1
      ? std::sqrt(vp_var / (total_games - 1)) / std::sqrt(total_games)
      : 0.0;

  // --- Print aggregate results ---
  std::cout << absl::StrFormat("Completed %d games in %.1f seconds (%.1f g/s)\n\n",
                               total_games, elapsed_secs,
                               total_games / elapsed_secs);

  std::cout << "=== Aggregate Results ===\n\n";

  std::cout << absl::StrFormat(
      "First-place rate:     %.2f%%   Wilson 95%% CI [%.2f%%, %.2f%%]\n",
      first_place_ci.point * 100, first_place_ci.lower * 100,
      first_place_ci.upper * 100);

  std::cout << absl::StrFormat(
      "Mean final_scored_vp: %.3f   95%% CI [%.3f, %.3f]\n",
      mean_vp, mean_vp - 1.96 * vp_se, mean_vp + 1.96 * vp_se);

  std::cout << absl::StrFormat(
      "Mean return:          %.4f   95%% CI [%.4f, %.4f]\n",
      mean_return, mean_return - 1.96 * return_se,
      mean_return + 1.96 * return_se);

  std::cout << absl::StrFormat(
      "P(round <= 7):        %.2f%%   Wilson 95%% CI [%.2f%%, %.2f%%]\n",
      round_le7_ci.point * 100, round_le7_ci.lower * 100,
      round_le7_ci.upper * 100);

  std::cout << absl::StrFormat(
      "P(round <= 8):        %.2f%%   Wilson 95%% CI [%.2f%%, %.2f%%]\n",
      round_le8_ci.point * 100, round_le8_ci.lower * 100,
      round_le8_ci.upper * 100);

  std::cout << "\nPlacement distribution:\n";
  for (int r = 0; r < 4; ++r) {
    WilsonCI pci = WilsonScore(placement_counts[r], total_games);
    std::cout << absl::StrFormat(
        "  %dst: %5d / %d  (%.2f%%)  Wilson [%.2f%%, %.2f%%]\n",
        r + 1, placement_counts[r], total_games,
        pci.point * 100, pci.lower * 100, pci.upper * 100);
  }

  std::cout << "\nWin rate by seat:\n";
  for (int s = 0; s < 4; ++s) {
    if (games_by_seat[s] > 0) {
      WilsonCI sci = WilsonScore(wins_by_seat[s], games_by_seat[s]);
      std::cout << absl::StrFormat(
          "  Seat P%d: %d/%d  (%.2f%%)  Wilson [%.2f%%, %.2f%%]\n",
          s, wins_by_seat[s], games_by_seat[s],
          sci.point * 100, sci.lower * 100, sci.upper * 100);
    }
  }

  // --- Write aggregate JSON with seat-wise and placement counts ---
  if (!output_dir.empty()) {
    std::string agg_path =
        (std::filesystem::path(output_dir) / "aggregate.json").string();
    std::ofstream agg_out(agg_path);
    if (agg_out.is_open()) {
      agg_out << "{\n"
              << "  \"model_checkpoint\": \""
              << model_checkpoint << "\",\n"
              << "  \"opponent_checkpoints\": \""
              << (use_opponent_model ? opponent_str : "random") << "\",\n"
              << "  \"domain\": \"" << domain_str << "\",\n"
              << "  \"base_seed\": " << base_seed << ",\n"
              << "  \"num_games\": " << total_games << ",\n"
              << "  \"greedy\": " << (greedy ? "true" : "false") << ",\n"
              << "  \"temperature\": "
              << absl::StrFormat("%.2f", temperature) << ",\n"
              << "  \"hidden_dim\": " << main_hidden_dim << ",\n"
              << "  \"num_blocks\": " << main_num_blocks << ",\n"
              << "  \"opp_hidden_dim\": " << (opp_metadata.empty() ? -1 : opp_metadata[0].hidden_dim) << ",\n"
              << "  \"opp_num_blocks\": " << (opp_metadata.empty() ? -1 : opp_metadata[0].num_blocks) << ",\n"
              << "  \"execution_mode\": \"" << (deterministic ? "deterministic" : "batched") << "\",\n"
              << "  \"threads\": " << num_threads << ",\n"
              << "  \"detected_opponent_architectures\": [\n";
      for (size_t i = 0; i < opp_metadata.size(); ++i) {
        agg_out << "    {\n"
                << "      \"checkpoint\": \"" << opp_metadata[i].path << "\",\n"
                << "      \"hidden_dim\": " << opp_metadata[i].hidden_dim << ",\n"
                << "      \"num_blocks\": " << opp_metadata[i].num_blocks << "\n"
                << "    }" << (i + 1 < opp_metadata.size() ? "," : "") << "\n";
      }
      agg_out << "  ],\n"
              << "  \"elapsed_seconds\": "
              << absl::StrFormat("%.1f", elapsed_secs) << ",\n"
              << "  \"first_place_rate\": "
              << absl::StrFormat("%.6f", first_place_ci.point) << ",\n"
              << "  \"first_place_wilson_lower\": "
              << absl::StrFormat("%.6f", first_place_ci.lower) << ",\n"
              << "  \"first_place_wilson_upper\": "
              << absl::StrFormat("%.6f", first_place_ci.upper) << ",\n"
              << "  \"mean_final_scored_vp\": "
              << absl::StrFormat("%.4f", mean_vp) << ",\n"
              << "  \"mean_final_scored_vp_ci_lower\": "
              << absl::StrFormat("%.4f", mean_vp - 1.96 * vp_se) << ",\n"
              << "  \"mean_final_scored_vp_ci_upper\": "
              << absl::StrFormat("%.4f", mean_vp + 1.96 * vp_se) << ",\n"
              << "  \"mean_return\": "
              << absl::StrFormat("%.6f", mean_return) << ",\n"
              << "  \"mean_return_ci_lower\": "
              << absl::StrFormat("%.6f", mean_return - 1.96 * return_se)
              << ",\n"
              << "  \"mean_return_ci_upper\": "
              << absl::StrFormat("%.6f", mean_return + 1.96 * return_se)
              << ",\n"
              << "  \"p_round_le7\": "
              << absl::StrFormat("%.6f", round_le7_ci.point) << ",\n"
              << "  \"p_round_le7_wilson_lower\": "
              << absl::StrFormat("%.6f", round_le7_ci.lower) << ",\n"
              << "  \"p_round_le7_wilson_upper\": "
              << absl::StrFormat("%.6f", round_le7_ci.upper) << ",\n"
              << "  \"p_round_le8\": "
              << absl::StrFormat("%.6f", round_le8_ci.point) << ",\n"
              << "  \"p_round_le8_wilson_lower\": "
              << absl::StrFormat("%.6f", round_le8_ci.lower) << ",\n"
              << "  \"p_round_le8_wilson_upper\": "
              << absl::StrFormat("%.6f", round_le8_ci.upper) << ",\n"
              << "  \"placement_1st_count\": " << placement_counts[0] << ",\n"
              << "  \"placement_2nd_count\": " << placement_counts[1] << ",\n"
              << "  \"placement_3rd_count\": " << placement_counts[2] << ",\n"
              << "  \"placement_4th_count\": " << placement_counts[3] << ",\n"
              << "  \"seat_0_wins\": " << wins_by_seat[0] << ",\n"
              << "  \"seat_0_games\": " << games_by_seat[0] << ",\n"
              << "  \"seat_1_wins\": " << wins_by_seat[1] << ",\n"
              << "  \"seat_1_games\": " << games_by_seat[1] << ",\n"
              << "  \"seat_2_wins\": " << wins_by_seat[2] << ",\n"
              << "  \"seat_2_games\": " << games_by_seat[2] << ",\n"
              << "  \"seat_3_wins\": " << wins_by_seat[3] << ",\n"
              << "  \"seat_3_games\": " << games_by_seat[3] << "\n"
              << "}\n";
      agg_out.close();
      std::cout << "\nAggregate results written to " << agg_path << "\n";
    }
  }
}

}  // namespace
}  // namespace open_spiel

int main(int argc, char* argv[]) {
  setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8", 1);
  absl::ParseCommandLine(argc, argv);

  // Set PyTorch to use deterministic algorithms globally
  at::globalContext().setDeterministicAlgorithms(true, /*silent=*/true);

  // On GPU: game threads only do engine work. On CPU: Runner needs all cores.
  if (torch::cuda::is_available()) {
    at::set_num_threads(1);
  } else {
    at::set_num_threads(
        std::max(1, static_cast<int>(std::thread::hardware_concurrency())));
  }
  at::set_num_interop_threads(1);

  open_spiel::RunEvaluation();
  return 0;
}
