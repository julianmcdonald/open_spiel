// Timed, multi-threaded Dune Imperium self-play benchmark driver
// Bypasses LibTorch to establish a pure C++ engine performance baseline.

#include <iostream>
#include <random>
#include <vector>
#include <thread>
#include <atomic>
#include <chrono>

#include "open_spiel/abseil-cpp/absl/flags/flag.h"
#include "open_spiel/abseil-cpp/absl/flags/parse.h"
#include "open_spiel/abseil-cpp/absl/strings/str_format.h"
#include "open_spiel/spiel.h"

ABSL_FLAG(std::string, game, "dune_imperium", "The name of the game to play.");
ABSL_FLAG(int, games, 1000, "How many games to play in total.");
ABSL_FLAG(int, threads, 8, "How many threads to run.");

namespace open_spiel {
namespace {

int RandomSimulation(std::mt19937* rng, const Game& game) {
  std::unique_ptr<State> state = game.NewInitialState();

  bool provides_info_state_tensor =
      game.GetType().provides_information_state_tensor;
  bool provides_observations_tensor =
      game.GetType().provides_observation_tensor;
  std::vector<float> obs;
  if (provides_info_state_tensor) {
    obs = std::vector<float>(game.InformationStateTensorSize());
  } else if (provides_observations_tensor) {
    obs = std::vector<float>(game.ObservationTensorSize());
  }

  int game_length = 0;
  while (!state->IsTerminal()) {
    if (provides_info_state_tensor && state->CurrentPlayer() >= 0) {
      state->InformationStateTensor(state->CurrentPlayer(),
                                    absl::MakeSpan(obs));
    } else if (provides_observations_tensor && state->CurrentPlayer() >= 0) {
      state->ObservationTensor(state->CurrentPlayer(), absl::MakeSpan(obs));
    }
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
      if (game.GetType().chance_mode ==
          GameType::ChanceMode::kSampledStochastic) {
        action = outcomes.front().first;
      } else {
        action = SampleAction(outcomes, *rng).first;
      }
      state->ApplyAction(action);
    } else if (state->CurrentPlayer() == kSimultaneousPlayerId) {
      std::vector<Action> joint_action;
      for (int p = 0; p < game.NumPlayers(); p++) {
        std::vector<Action> actions = state->LegalActions(p);
        Action action = 0;
        if (!actions.empty()) {
          std::uniform_int_distribution<int> dis(0, actions.size() - 1);
          action = actions[dis(*rng)];
        }
        joint_action.push_back(action);
      }
      state->ApplyActions(joint_action);
    } else {
      std::vector<Action> actions = state->LegalActions();
      if (actions.empty()) {
        std::cerr << "Spiel Fatal Error: Non-terminal state has empty LegalActions(). Player: " 
                  << state->CurrentPlayer() << "\nState string:\n" << state->ToString() << std::endl;
        std::abort();
      }
      std::uniform_int_distribution<int> dis(0, actions.size() - 1);
      Action action = actions[dis(*rng)];
      state->ApplyAction(action);
    }
  }
  return game_length;
}

void ThreadWorker(int thread_id, const Game* game, std::atomic<int>& games_completed,
                  int total_games, std::atomic<int>& total_moves) {
  std::mt19937 rng(std::random_device{}() + thread_id);
  int moves = 0;
  while (true) {
    int prev_completed = games_completed.fetch_add(1);
    if (prev_completed >= total_games) {
      games_completed.fetch_sub(1);
      break;
    }
    moves += RandomSimulation(&rng, *game);
  }
  total_moves.fetch_add(moves);
}

} // namespace
} // namespace open_spiel

int main(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);

  std::string game_name = absl::GetFlag(FLAGS_game);
  int total_games = absl::GetFlag(FLAGS_games);
  int num_threads = absl::GetFlag(FLAGS_threads);

  std::cout << absl::StrFormat("Initializing Multi-threaded Benchmark...\n");
  std::cout << absl::StrFormat("Game: %s\n", game_name);
  std::cout << absl::StrFormat("Total games to simulate: %d\n", total_games);
  std::cout << absl::StrFormat("Number of threads: %d\n", num_threads);

  auto game = open_spiel::LoadGame(game_name);

  std::atomic<int> games_completed(0);
  std::atomic<int> total_moves(0);

  auto start_time = std::chrono::high_resolution_clock::now();

  std::vector<std::thread> threads;
  threads.reserve(num_threads);
  for (int i = 0; i < num_threads; ++i) {
    threads.emplace_back(open_spiel::ThreadWorker, i, game.get(),
                         std::ref(games_completed), total_games, std::ref(total_moves));
  }

  for (auto& t : threads) {
    t.join();
  }

  auto end_time = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = end_time - start_time;

  double seconds = elapsed.count();
  int moves = total_moves.load();
  int completed = games_completed.load();

  std::cout << absl::StrFormat("\n=== Benchmark Completed ===\n");
  std::cout << absl::StrFormat("Elapsed Time: %.3f seconds\n", seconds);
  std::cout << absl::StrFormat("Games Completed: %d\n", completed);
  std::cout << absl::StrFormat("Total Moves Executed: %d\n", moves);
  std::cout << absl::StrFormat("Games Per Second (GPS): %.2f\n", completed / seconds);
  std::cout << absl::StrFormat("Moves Per Second (MPS): %.2f\n", moves / seconds);

  return 0;
}
