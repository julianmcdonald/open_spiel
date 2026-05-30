#include <iostream>
#include <memory>
#include <vector>
#include <string>
#include <algorithm>
#include <filesystem>
#include <random>

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
    torch::Device device,
    int64_t obs_size,
    int64_t action_size,
    bool provides_info_state_tensor,
    bool provides_observations_tensor) {
  std::vector<float> obs(obs_size, 0.0f);
  if (provides_info_state_tensor) {
    state.InformationStateTensor(player, absl::MakeSpan(obs));
  } else if (provides_observations_tensor) {
    state.ObservationTensor(player, absl::MakeSpan(obs));
  }

  torch::Tensor state_tensor = torch::from_blob(obs.data(), {1, obs_size}, torch::kFloat).to(device);
  torch::Tensor logits = model->forward(state_tensor).logits.squeeze(0);
  torch::Tensor cpu_logits = logits.contiguous().to(torch::kCPU);

  std::vector<float> prev_logits_vec(action_size, 0.0f);
  std::memcpy(prev_logits_vec.data(), cpu_logits.data_ptr<float>(), action_size * sizeof(float));

  Action chosen_action = legal_actions.front();
  float max_logit = -1e9f;
  for (Action a : legal_actions) {
    if (prev_logits_vec[a] > max_logit) {
      max_logit = prev_logits_vec[a];
      chosen_action = a;
    }
  }
  return chosen_action;
}

void RunEvaluation(const std::string& model_checkpoint,
                   const std::string& opponent_checkpoint,
                   int total_games = 1000) {
  // Clamp internal LibTorch CPU threads to 1 to prevent thread pool thrashing
  at::set_num_threads(1);
  at::set_num_interop_threads(1);

  // Get clean names using std::filesystem::path
  std::string model_name = std::filesystem::path(model_checkpoint).filename().string();
  bool use_opponent_model = (!opponent_checkpoint.empty() && opponent_checkpoint != "random");
  std::string opponent_name = use_opponent_model ? std::filesystem::path(opponent_checkpoint).filename().string() : "Random Agent";

  // 1. Initialize Game
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

  // Device Management
  torch::Device device(torch::kCPU);
  try {
    if (torch::cuda::is_available()) {
      device = torch::Device(torch::kCUDA);
    }
  } catch (...) {
    // If checking for CUDA throws due to driver OOM, default to CPU
    device = torch::Device(torch::kCPU);
  }

  // 2. Initialize Models
  auto model_a = std::make_shared<SharedDunePolicyValueNetImpl>(obs_size, 1024, action_size);
  model_a->eval(); // eval mode

  // Load Model A weights with explicit device mapping
  try {
    torch::serialize::InputArchive archive;
    archive.load_from(model_checkpoint, device);
    model_a->load(archive);
  } catch (const c10::Error& e) {
    if (device.is_cuda()) {
      std::cerr << "Warning: Failed to load Model A checkpoint onto CUDA device. Falling back to CPU.\n";
      std::cerr << "Error details: " << e.msg() << "\n";
      device = torch::Device(torch::kCPU);
      try {
        torch::serialize::InputArchive archive;
        archive.load_from(model_checkpoint, device);
        model_a->load(archive);
      } catch (const c10::Error& fallback_err) {
        std::cerr << "Failed to load Model A checkpoint on CPU: " << model_checkpoint << "\n";
        std::cerr << fallback_err.msg() << "\n";
        std::exit(1);
      }
    } else {
      std::cerr << "Failed to load Model A checkpoint: " << model_checkpoint << "\n";
      std::cerr << e.msg() << "\n";
      std::exit(1);
    }
  }
  model_a->to(device);

  std::shared_ptr<SharedDunePolicyValueNetImpl> model_b = nullptr;
  if (use_opponent_model) {
    model_b = std::make_shared<SharedDunePolicyValueNetImpl>(obs_size, 1024, action_size);
    model_b->eval();
    try {
      torch::serialize::InputArchive archive;
      archive.load_from(opponent_checkpoint, device);
      model_b->load(archive);
    } catch (const c10::Error& e) {
      if (device.is_cuda()) {
        std::cerr << "Warning: Failed to load Model B checkpoint onto CUDA device. Falling back to CPU.\n";
        std::cerr << "Error details: " << e.msg() << "\n";
        device = torch::Device(torch::kCPU);
        model_a->to(device); // Keep Model A synced on CPU
        try {
          torch::serialize::InputArchive archive;
          archive.load_from(opponent_checkpoint, device);
          model_b->load(archive);
        } catch (const c10::Error& fallback_err) {
          std::cerr << "Failed to load Model B checkpoint on CPU: " << opponent_checkpoint << "\n";
          std::cerr << fallback_err.msg() << "\n";
          std::exit(1);
        }
      } else {
        std::cerr << "Failed to load Model B checkpoint: " << opponent_checkpoint << "\n";
        std::cerr << e.msg() << "\n";
        std::exit(1);
      }
    }
    model_b->to(device);
  }

  torch::NoGradGuard no_grad;

  // Random Number Generator for Random Players and Chance nodes
  std::mt19937 rng(std::random_device{}());

  // Statistics
  int model_wins = 0;
  int opponent_wins[3] = {0, 0, 0}; // [0]: Opponent 1, [1]: Opponent 2, [2]: Opponent 3 (relative to model)
  int model_wins_by_seat[4] = {0, 0, 0, 0}; // [0]: Seat 0, [1]: Seat 1, etc.
  int games_by_seat[4] = {0, 0, 0, 0};

  for (int g = 0; g < total_games; ++g) {
    int model_player = g % 4; // Ensure perfectly balanced seat rotation (250 games in each seat)
    games_by_seat[model_player]++;

    std::unique_ptr<State> state = game->NewInitialState();
    
    int game_length = 0;

    while (!state->IsTerminal()) {
      ++game_length;
      if (game_length > 5000) {
        std::cerr << "Possible infinite loop detected! Game length: " << game_length
                  << " Player: " << state->CurrentPlayer()
                  << "\nState string:\n" << state->ToString() << std::endl;
        std::abort();
      }

      if (state->IsChanceNode()) {
        std::vector<std::pair<Action, double>> outcomes = state->ChanceOutcomes();
        Action action;
        if (game->GetType().chance_mode ==
            GameType::ChanceMode::kSampledStochastic) {
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
        std::cerr << "Error: empty legal actions at non-terminal state!\n";
        break;
      }

      Action chosen_action = -1;

      if (current_player == model_player) {
        chosen_action = GetModelAction(
            model_a, *state, legal_actions, current_player, device, obs_size, action_size,
            provides_info_state_tensor, provides_observations_tensor);
      } else {
        if (use_opponent_model) {
          chosen_action = GetModelAction(
              model_b, *state, legal_actions, current_player, device, obs_size, action_size,
              provides_info_state_tensor, provides_observations_tensor);
        } else {
          // Random agent
          std::uniform_int_distribution<std::size_t> dist(0, legal_actions.size() - 1);
          chosen_action = legal_actions[dist(rng)];
        }
      }

      state->ApplyAction(chosen_action);
    }

    // Determine winner based on returns (which handle all official tie-breakers stochastically and return placement values)
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
      model_wins++;
      model_wins_by_seat[model_player]++;
    } else {
      // Find which opponent player it was relative to the model
      int relative_idx = (winner - model_player + 4) % 4 - 1; // 0 for Opponent 1, 1 for Opponent 2, 2 for Opponent 3
      if (relative_idx >= 0 && relative_idx < 3) {
        opponent_wins[relative_idx]++;
      }
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
  // Parse command-line arguments
  std::string model_checkpoint = "/home/warcr/projects/dune_drl/dune_stage_a_run1_model.pt";
  int num_games = 1000;
  std::string opponent_checkpoint = "";

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

  open_spiel::RunEvaluation(model_checkpoint, opponent_checkpoint, num_games);
  return 0;
}
