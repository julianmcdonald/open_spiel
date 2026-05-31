#include <iostream>
#include <memory>
#include <vector>
#include <string>
#include <algorithm>
#include <filesystem>
#include <random>
#include <thread>
#include <atomic>

#include "open_spiel/spiel.h"
#include "open_spiel/games/dune_imperium/dune_imperium.h"
#include "open_spiel/abseil-cpp/absl/strings/str_format.h"
#include <torch/torch.h>

#include "dune_network.h"

namespace open_spiel {
namespace {

Action GetModelAction(
    const std::shared_ptr<SharedDunePolicyValueNetImpl>& model,
    const State& state,
    const std::vector<Action>& legal_actions,
    Player player,
    torch::Tensor& state_tensor,
    std::vector<float>& obs,
    int64_t obs_size,
    int64_t action_size,
    bool provides_info_state_tensor,
    bool provides_observations_tensor) {
  // Zero and fill the pre-allocated observation buffer (no heap alloc)
  std::fill(obs.begin(), obs.end(), 0.0f);
  if (provides_info_state_tensor) {
    state.InformationStateTensor(player, absl::MakeSpan(obs));
  } else if (provides_observations_tensor) {
    state.ObservationTensor(player, absl::MakeSpan(obs));
  }

  // Copy into pre-allocated tensor instead of creating a new one each call
  std::memcpy(state_tensor.data_ptr<float>(), obs.data(), obs_size * sizeof(float));

  // Forward pass - keep output alive so logits_ptr is valid
  auto output = model->forward(state_tensor);
  float* logits_ptr = output.logits.data_ptr<float>();

  Action chosen_action = legal_actions.front();
  float max_logit = -1e9f;
  for (Action a : legal_actions) {
    // Logits shape [1, action_size], row-major so index directly
    if (logits_ptr[a] > max_logit) {
      max_logit = logits_ptr[a];
      chosen_action = a;
    }
  }
  return chosen_action;
}

struct ThreadStats {
  int model_wins = 0;
  int opponent_wins[3] = {0, 0, 0};
  int model_wins_by_seat[4] = {0, 0, 0, 0};
  int games_by_seat[4] = {0, 0, 0, 0};
};

void WorkerThread(
    int thread_id,
    std::shared_ptr<const Game> game,
    const std::string& model_checkpoint,
    const std::string& opponent_checkpoint,
    bool use_opponent_model,
    int64_t obs_size,
    int64_t action_size,
    bool provides_info_state_tensor,
    bool provides_observations_tensor,
    std::atomic<int>& next_game_id,
    int total_games,
    int hidden_dim,
    int num_blocks,
    ThreadStats& stats) {
  // InferenceMode is faster than NoGradGuard - disables autograd, view tracking, version counting
  torch::InferenceMode inference_guard;

  // Force CPU device to eliminate CUDA context switching overhead across threads
  torch::Device device(torch::kCPU);

  // Initialize and load local models for thread safety
  auto model_a = std::make_shared<SharedDunePolicyValueNetImpl>(obs_size, hidden_dim, action_size, num_blocks);
  model_a->eval();
  try {
    torch::serialize::InputArchive archive;
    archive.load_from(model_checkpoint, device);
    model_a->load(archive);
  } catch (const c10::Error& e) {
    std::cerr << "Thread " << thread_id << " failed to load Model A checkpoint: " << model_checkpoint << "\n" << e.msg() << "\n";
    std::exit(1);
  }
  model_a->to(device);

  std::shared_ptr<SharedDunePolicyValueNetImpl> model_b = nullptr;
  if (use_opponent_model) {
    model_b = std::make_shared<SharedDunePolicyValueNetImpl>(obs_size, hidden_dim, action_size, num_blocks);
    model_b->eval();
    try {
      torch::serialize::InputArchive archive;
      archive.load_from(opponent_checkpoint, device);
      model_b->load(archive);
    } catch (const c10::Error& e) {
      std::cerr << "Thread " << thread_id << " failed to load Model B checkpoint: " << opponent_checkpoint << "\n" << e.msg() << "\n";
      std::exit(1);
    }
    model_b->to(device);
  }

  // Seed unique thread-local RNG using random_device combined with thread ID
  std::random_device rd;
  std::mt19937 rng(rd() ^ thread_id);

  // Pre-allocate buffers reused across ALL games and ALL action evaluations.
  // Without this, every single action decision heap-allocates these, and with
  // 16 threads doing hundreds of decisions per game, the global allocator lock
  // becomes the bottleneck (100% CPU but mostly spinning on malloc contention).
  std::vector<float> obs(obs_size, 0.0f);
  torch::Tensor state_tensor = torch::zeros({1, obs_size}, torch::kFloat);

  while (true) {
    int g = next_game_id++;
    if (g >= total_games) {
      break;
    }

    int model_player = g % 4;
    stats.games_by_seat[model_player]++;

    std::unique_ptr<State> state = game->NewInitialState();
    int game_length = 0;

    while (!state->IsTerminal()) {
      ++game_length;
      if (game_length > 5000) {
        std::cerr << "Possible infinite loop detected in thread " << thread_id << "! Game: " << g
                  << " Length: " << game_length << " Player: " << state->CurrentPlayer() << std::endl;
        std::abort();
      }

      if (state->IsChanceNode()) {
        std::vector<std::pair<Action, double>> outcomes = state->ChanceOutcomes();
        Action action;
        if (game->GetType().chance_mode == GameType::ChanceMode::kSampledStochastic) {
          action = outcomes.front().first;
        } else {
          action = SampleAction(outcomes, rng).first;
        }
        state->ApplyAction(action);
        continue;
      }

      Player current_player = state->CurrentPlayer();
      std::vector<Action> legal_actions = state->LegalActions();
      if (legal_actions.empty()) {
        std::cerr << "Error in thread " << thread_id << ": empty legal actions at non-terminal state in game " << g << "!\n";
        break;
      }

      Action chosen_action = -1;

      if (current_player == model_player) {
        chosen_action = GetModelAction(
            model_a, *state, legal_actions, current_player, state_tensor, obs,
            obs_size, action_size,
            provides_info_state_tensor, provides_observations_tensor);
      } else {
        if (use_opponent_model) {
          chosen_action = GetModelAction(
              model_b, *state, legal_actions, current_player, state_tensor, obs,
              obs_size, action_size,
              provides_info_state_tensor, provides_observations_tensor);
        } else {
          std::uniform_int_distribution<std::size_t> dist(0, legal_actions.size() - 1);
          chosen_action = legal_actions[dist(rng)];
        }
      }

      state->ApplyAction(chosen_action);
    }

    std::vector<double> returns = state->Returns();
    int winner = -1;
    double max_return = -999.0;
    for (int p = 0; p < 4; ++p) {
      if (returns[p] > max_return) {
        max_return = returns[p];
        winner = p;
      }
    }

    if (winner == model_player) {
      stats.model_wins++;
      stats.model_wins_by_seat[model_player]++;
    } else {
      int relative_idx = (winner - model_player + 4) % 4 - 1;
      if (relative_idx >= 0 && relative_idx < 3) {
        stats.opponent_wins[relative_idx]++;
      }
    }
  }
}

void RunEvaluation(const std::string& model_checkpoint,
                   const std::string& opponent_checkpoint,
                   int total_games = 1000,
                   int requested_threads = 0,
                   int hidden_dim = 1024,
                   int num_blocks = 4) {
  // Get clean names using std::filesystem::path
  std::string model_name = std::filesystem::path(model_checkpoint).filename().string();
  bool use_opponent_model = (!opponent_checkpoint.empty() && opponent_checkpoint != "random");
  std::string opponent_name = use_opponent_model ? std::filesystem::path(opponent_checkpoint).filename().string() : "Random Agent";

  // 1. Initialize Game (single immutable instance shared read-only across threads)
  std::shared_ptr<const Game> game = LoadGame("dune_imperium");

  // General observation shape retrieval
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
    std::cerr << "Error: dynamic observation size is 0.\n";
    std::exit(1);
  }

  int64_t action_size = game->NumDistinctActions();

  // Determine thread count. LibTorch's internal allocator uses spinlocks, so
  // too many threads doing forward passes causes contention (100% CPU but no
  // real work). For model-vs-model (4 forward passes per step), cap at 4.
  unsigned int hw_threads = std::thread::hardware_concurrency();
  if (hw_threads == 0) hw_threads = 4;
  unsigned int num_threads;
  if (requested_threads > 0) {
    num_threads = static_cast<unsigned int>(requested_threads);
  } else if (use_opponent_model) {
    // Model-vs-model: 4x more forward passes, cap threads to avoid spinlock hell
    num_threads = std::min(hw_threads, 4u);
  } else {
    // Model-vs-random: only 1/4 actions need inference, can use more threads
    num_threads = hw_threads;
  }
  std::cout << "Starting evaluation of " << total_games << " games using " << num_threads << " threads on CPU...\n";

  std::atomic<int> next_game_id{0};
  std::vector<ThreadStats> thread_stats_vec(num_threads);
  std::vector<std::thread> threads;

  for (unsigned int t = 0; t < num_threads; ++t) {
    threads.emplace_back(
        WorkerThread, t, game, model_checkpoint, opponent_checkpoint, use_opponent_model,
        obs_size, action_size, provides_info_state_tensor, provides_observations_tensor,
        std::ref(next_game_id), total_games, hidden_dim, num_blocks, std::ref(thread_stats_vec[t]));
  }

  // Join all threads
  for (auto& th : threads) {
    if (th.joinable()) {
      th.join();
    }
  }

  // Global Statistics aggregation
  int model_wins = 0;
  int opponent_wins[3] = {0, 0, 0};
  int model_wins_by_seat[4] = {0, 0, 0, 0};
  int games_by_seat[4] = {0, 0, 0, 0};

  for (unsigned int t = 0; t < num_threads; ++t) {
    model_wins += thread_stats_vec[t].model_wins;
    for (int p = 0; p < 3; ++p) {
      opponent_wins[p] += thread_stats_vec[t].opponent_wins[p];
    }
    for (int s = 0; s < 4; ++s) {
      model_wins_by_seat[s] += thread_stats_vec[t].model_wins_by_seat[s];
      games_by_seat[s] += thread_stats_vec[t].games_by_seat[s];
    }
  }

  // Print final results exactly as requested (no additional logging, only final result)
  std::cout << "Evaluation of " << total_games << " games completed.\n";
  std::cout << "Model checkpoint A (evaluated): " << model_checkpoint << "\n";
  if (use_opponent_model) {
    std::cout << "Model checkpoint B (opponent):  " << opponent_checkpoint << "\n\n";
  } else {
    std::cout << "Opponents used:                 Random Agents\n\n";
  }

  std::cout << "Overall Winrates:\n";
  std::cout << "  " << absl::StrFormat("%-28s", model_name + " (Eval):")
            << absl::StrFormat("%.2f%% (%d wins)", (model_wins * 100.0 / total_games), model_wins) << "\n";
  std::cout << "  " << absl::StrFormat("%-28s", opponent_name + " 1 (Eval + 1):")
            << absl::StrFormat("%.2f%% (%d wins)", (opponent_wins[0] * 100.0 / total_games), opponent_wins[0]) << "\n";
  std::cout << "  " << absl::StrFormat("%-28s", opponent_name + " 2 (Eval + 2):")
            << absl::StrFormat("%.2f%% (%d wins)", (opponent_wins[1] * 100.0 / total_games), opponent_wins[1]) << "\n";
  std::cout << "  " << absl::StrFormat("%-28s", opponent_name + " 3 (Eval + 3):")
            << absl::StrFormat("%.2f%% (%d wins)", (opponent_wins[2] * 100.0 / total_games), opponent_wins[2]) << "\n\n";

  std::cout << "Eval Model Winrate by Seat:\n";
  for (int p = 0; p < 4; ++p) {
    double seat_winrate = games_by_seat[p] > 0 ? (model_wins_by_seat[p] * 100.0 / games_by_seat[p]) : 0.0;
    std::cout << "  Seat P" << p << ": " << absl::StrFormat("%.2f%% (%d/%d wins)", seat_winrate, model_wins_by_seat[p], games_by_seat[p]) << "\n";
  }
}

} // namespace
} // namespace open_spiel

int main(int argc, char* argv[]) {
  // Clamp internal LibTorch CPU threads globally to 1 to prevent thread pool thrashing
  at::set_num_threads(1);
  at::set_num_interop_threads(1);

  // Usage: dune_eval_1000 <model_a> <num_games> [opponent_model|"random"] [num_threads] [hidden_dim] [num_blocks]
  std::string model_checkpoint = "/home/warcr/projects/dune_drl/dune_stage_a_run1_model.pt";
  int num_games = 1000;
  std::string opponent_checkpoint = "";
  int num_threads = 0;  // 0 = auto-detect
  int hidden_dim = 1024;
  int num_blocks = 4;

  if (argc > 1) {
    model_checkpoint = argv[1];
  }
  if (argc > 2) {
    try {
      num_games = std::stoi(argv[2]);
    } catch (...) {
      std::cerr << "Warning: invalid number of games specified. Defaulting to 1000.\n";
    }
  }
  if (argc > 3) {
    opponent_checkpoint = argv[3];
  }
  if (argc > 4) {
    try {
      num_threads = std::stoi(argv[4]);
    } catch (...) {
      std::cerr << "Warning: invalid thread count specified. Using auto-detect.\n";
    }
  }
  if (argc > 5) {
    try {
      hidden_dim = std::stoi(argv[5]);
    } catch (...) {
      std::cerr << "Warning: invalid hidden dimension specified. Defaulting to 1024.\n";
    }
  }
  if (argc > 6) {
    try {
      num_blocks = std::stoi(argv[6]);
    } catch (...) {
      std::cerr << "Warning: invalid block count specified. Defaulting to 4.\n";
    }
  }

  open_spiel::RunEvaluation(model_checkpoint, opponent_checkpoint, num_games, num_threads, hidden_dim, num_blocks);
  return 0;
}
