#include <iostream>
#include <memory>
#include <vector>
#include <string>
#include <algorithm>
#include <filesystem>
#include <random>
#include <thread>
#include <atomic>
#include <chrono>

#include "open_spiel/spiel.h"
#include "open_spiel/games/dune_imperium/dune_imperium.h"
#include "open_spiel/abseil-cpp/absl/strings/str_format.h"
#include <torch/torch.h>

#include "dune_network.h"

namespace open_spiel {
namespace {

struct ThreadStats {
  int model_wins = 0;
  int opponent_wins[3] = {0, 0, 0};
  int model_wins_by_seat[4] = {0, 0, 0, 0};
  int games_by_seat[4] = {0, 0, 0, 0};
};

// Batched worker: game threads submit observations to shared BatchedEvaluator(s).
// The Runner thread inside BatchedEvaluator accumulates requests into batches,
// runs one gemm (matrix-matrix multiply) using ALL CPU cores, then distributes
// results. This reads the 334MB weight matrix ONCE per batch instead of once
// per observation, giving ~Nx throughput improvement where N = batch size.
void BatchedWorkerThread(
    int thread_id,
    std::shared_ptr<const Game> game,
    std::shared_ptr<BatchedEvaluator> model_evaluator,
    std::shared_ptr<BatchedEvaluator> opponent_evaluator,  // nullptr if random
    int64_t obs_size,
    bool provides_info_state_tensor,
    bool provides_observations_tensor,
    std::atomic<int>& next_game_id,
    int total_games,
    ThreadStats& stats) {
  std::random_device rd;
  std::mt19937 rng(rd() ^ thread_id);

  // Pre-allocate observation buffer reused across all games
  std::vector<float> obs(obs_size, 0.0f);

  while (true) {
    int g = next_game_id++;
    if (g >= total_games) break;

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
        std::cerr << "Error in thread " << thread_id << ": empty legal actions in game " << g << "!\n";
        break;
      }

      Action chosen_action = -1;

      bool use_model = (current_player == model_player);
      bool use_opponent_model = (!use_model && opponent_evaluator != nullptr);

      if (use_model || use_opponent_model) {
        // Fill observation buffer
        std::fill(obs.begin(), obs.end(), 0.0f);
        if (provides_info_state_tensor) {
          state->InformationStateTensor(current_player, absl::MakeSpan(obs));
        } else if (provides_observations_tensor) {
          state->ObservationTensor(current_player, absl::MakeSpan(obs));
        }

        // Submit to BatchedEvaluator — this thread sleeps until the batch fires
        auto& evaluator = use_model ? model_evaluator : opponent_evaluator;
        EvalResult result = evaluator->Evaluate(obs);

        // Argmax over legal actions
        chosen_action = legal_actions.front();
        float max_logit = -1e9f;
        for (Action a : legal_actions) {
          if (result.logits[a] > max_logit) {
            max_logit = result.logits[a];
            chosen_action = a;
          }
        }
      } else {
        // Random opponent
        std::uniform_int_distribution<std::size_t> dist(0, legal_actions.size() - 1);
        chosen_action = legal_actions[dist(rng)];
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
                   int num_blocks = 4,
                   int opp_hidden_dim = -1,
                   int opp_num_blocks = -1) {
  // Default opponent architecture to same as eval model
  if (opp_hidden_dim < 0) opp_hidden_dim = hidden_dim;
  if (opp_num_blocks < 0) opp_num_blocks = num_blocks;
  std::string model_name = std::filesystem::path(model_checkpoint).filename().string();
  bool use_opponent_model = (!opponent_checkpoint.empty() && opponent_checkpoint != "random");
  std::string opponent_name = use_opponent_model ? std::filesystem::path(opponent_checkpoint).filename().string() : "Random Agent";

  // 1. Initialize Game
  std::shared_ptr<const Game> game = LoadGame("dune_imperium");

  int64_t obs_size = 0;
  bool provides_info_state_tensor = game->GetType().provides_information_state_tensor;
  bool provides_observations_tensor = game->GetType().provides_observation_tensor;
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

  // 2. Determine game thread count.
  // Game threads are LIGHTWEIGHT — they sleep while waiting for batched inference.
  // Use many threads (4x hardware) to ensure the batch fills quickly.
  unsigned int hw_threads = std::thread::hardware_concurrency();
  if (hw_threads == 0) hw_threads = 4;
  unsigned int num_threads;
  if (requested_threads > 0) {
    num_threads = static_cast<unsigned int>(requested_threads);
  } else {
    // More game threads = more concurrent inference requests = fuller batches.
    // Model-vs-model: every action from every player needs inference, so threads
    // submit requests constantly. 4x hw_threads keeps the batch saturated.
    // Model-vs-random: only 1/4 of actions need inference, so we need even more
    // threads to compensate for the sparse request rate.
    num_threads = hw_threads * 4;
  }

  // Batch size tuned per device. GPU handles large batches efficiently;
  // CPU sees diminishing returns above ~32.
  int eval_timeout_ms = 2;

  // 3. Auto-detect GPU. The BatchedEvaluator already supports CUDA:
  //    pinned memory, async H2D/D2H transfers, FP16 autocast, TF32.
  torch::InferenceMode inference_guard;
  torch::Device device = torch::cuda::is_available() ? torch::Device(torch::kCUDA) : torch::Device(torch::kCPU);
  
  int eval_batch_size;
  if (device.is_cuda()) {
    // GPU: large batches are free (massively parallel). Use 64 to saturate.
    eval_batch_size = std::min(64u, num_threads);
    eval_timeout_ms = 1;  // GPU is fast, don't wait
  } else {
    eval_batch_size = std::min(32u, num_threads);
  }

  std::string device_name = device.is_cuda() ? "CUDA (GPU)" : "CPU";

  auto sync_mutex = std::make_shared<std::shared_mutex>();

  auto model_a = std::make_shared<SharedDunePolicyValueNetImpl>(obs_size, hidden_dim, action_size, num_blocks);
  model_a->eval();
  try {
    torch::serialize::InputArchive archive;
    archive.load_from(model_checkpoint, device);
    model_a->load(archive);
  } catch (const c10::Error& e) {
    std::cerr << "Failed to load Model A checkpoint: " << model_checkpoint << "\n" << e.msg() << "\n";
    std::exit(1);
  }
  model_a->to(device);

  // logit_cap=10.0f must match training to ensure consistent action rankings
  auto model_a_evaluator = std::make_shared<BatchedEvaluator>(
      model_a, eval_batch_size, eval_timeout_ms, device, sync_mutex.get(), 10.0f);

  std::shared_ptr<BatchedEvaluator> model_b_evaluator = nullptr;
  if (use_opponent_model) {
    auto model_b = std::make_shared<SharedDunePolicyValueNetImpl>(obs_size, opp_hidden_dim, action_size, opp_num_blocks);
    model_b->eval();
    try {
      torch::serialize::InputArchive archive;
      archive.load_from(opponent_checkpoint, device);
      model_b->load(archive);
    } catch (const c10::Error& e) {
      std::cerr << "Failed to load Model B checkpoint: " << opponent_checkpoint << "\n" << e.msg() << "\n";
      std::exit(1);
    }
    model_b->to(device);
    model_b_evaluator = std::make_shared<BatchedEvaluator>(
        model_b, eval_batch_size, eval_timeout_ms, device, sync_mutex.get(), 10.0f);
  }

  std::cout << "Starting evaluation of " << total_games << " games using " << num_threads
            << " game threads, batch_size=" << eval_batch_size 
            << ", timeout=" << eval_timeout_ms << "ms on " << device_name << "...\n";
  auto start_time = std::chrono::steady_clock::now();

  // 4. Launch game worker threads
  std::atomic<int> next_game_id{0};
  std::vector<ThreadStats> thread_stats_vec(num_threads);
  std::vector<std::thread> threads;

  for (unsigned int t = 0; t < num_threads; ++t) {
    threads.emplace_back(
        BatchedWorkerThread, t, game, model_a_evaluator, model_b_evaluator,
        obs_size, provides_info_state_tensor, provides_observations_tensor,
        std::ref(next_game_id), total_games, std::ref(thread_stats_vec[t]));
  }

  for (auto& th : threads) {
    if (th.joinable()) th.join();
  }

  auto end_time = std::chrono::steady_clock::now();
  double elapsed_secs = std::chrono::duration<double>(end_time - start_time).count();

  // 5. Aggregate statistics
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

  // 6. Print results
  std::cout << "\nEvaluation of " << total_games << " games completed in "
            << absl::StrFormat("%.1f", elapsed_secs) << " seconds ("
            << absl::StrFormat("%.1f", total_games / elapsed_secs) << " games/sec).\n";
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
  // On GPU: game threads only do engine work, no torch math. Keep internal threads at 1.
  // On CPU: the Runner thread needs all cores for batched gemm.
  if (torch::cuda::is_available()) {
    at::set_num_threads(1);
  } else {
    at::set_num_threads(std::max(1, (int)std::thread::hardware_concurrency()));
  }
  at::set_num_interop_threads(1);

  // Usage: dune_eval_1000 <model_a> <num_games> [opponent|"random"] [threads] [hidden_dim] [num_blocks] [opp_hidden_dim] [opp_num_blocks]
  std::string model_checkpoint = "/home/warcr/projects/dune_drl/dune_stage_a_run1_model.pt";
  int num_games = 1000;
  std::string opponent_checkpoint = "";
  int num_threads = 0;
  int hidden_dim = 2048;
  int num_blocks = 8;
  int opp_hidden_dim = -1;  // -1 = same as eval model
  int opp_num_blocks = -1;

  if (argc > 1) model_checkpoint = argv[1];
  if (argc > 2) {
    try { num_games = std::stoi(argv[2]); } catch (...) {
      std::cerr << "Warning: invalid number of games specified. Defaulting to 1000.\n";
    }
  }
  if (argc > 3) opponent_checkpoint = argv[3];
  if (argc > 4) {
    try { num_threads = std::stoi(argv[4]); } catch (...) {
      std::cerr << "Warning: invalid thread count specified. Using auto-detect.\n";
    }
  }
  if (argc > 5) {
    try { hidden_dim = std::stoi(argv[5]); } catch (...) {
      std::cerr << "Warning: invalid hidden dimension specified. Defaulting to 2048.\n";
    }
  }
  if (argc > 6) {
    try { num_blocks = std::stoi(argv[6]); } catch (...) {
      std::cerr << "Warning: invalid block count specified. Defaulting to 8.\n";
    }
  }
  if (argc > 7) {
    try { opp_hidden_dim = std::stoi(argv[7]); } catch (...) {
      std::cerr << "Warning: invalid opponent hidden dim. Using eval model's dim.\n";
    }
  }
  if (argc > 8) {
    try { opp_num_blocks = std::stoi(argv[8]); } catch (...) {
      std::cerr << "Warning: invalid opponent block count. Using eval model's count.\n";
    }
  }

  open_spiel::RunEvaluation(model_checkpoint, opponent_checkpoint, num_games, num_threads, hidden_dim, num_blocks, opp_hidden_dim, opp_num_blocks);
  return 0;
}
