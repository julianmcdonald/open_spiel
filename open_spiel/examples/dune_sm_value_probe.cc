#include <iostream>
#include <string>
#include <vector>
#include <memory>
#include <random>
#include <fstream>
#include <filesystem>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <atomic>
#include <thread>
#include <mutex>

#include "open_spiel/spiel.h"
#include "open_spiel/spiel_bots.h"
#include "open_spiel/abseil-cpp/absl/flags/flag.h"
#include "open_spiel/abseil-cpp/absl/flags/parse.h"
#include "open_spiel/abseil-cpp/absl/random/distributions.h"
#include "open_spiel/abseil-cpp/absl/strings/str_format.h"
#include "open_spiel/games/dune_imperium/dune_imperium.h"
#include "open_spiel/games/dune_imperium/dune_imperium_common.h"
#include "dune_puct_is_mcts.h"
#include "dune_search_session.h"
#include "dune_evaluator.h"
#include "dune_warmstart_helpers.h"
#include "dune_network.h"
#include <torch/torch.h>

ABSL_FLAG(std::string, model_checkpoint, "dune_ppo_stage_9_search_model.pt", "Path to the model checkpoint");
ABSL_FLAG(int, games, 40, "Total number of games to simulate");
ABSL_FLAG(int, threads, 20, "Number of parallel worker threads");
ABSL_FLAG(int, search_simulations, 400, "MCTS simulation budget for search players");
ABSL_FLAG(double, puct_c, 1.0, "Exploration constant");
ABSL_FLAG(double, utility_divisor, 4.0, "Terminal utility divisor");
ABSL_FLAG(double, search_opponent_temperature, 0.0, "Opponent temperature");
ABSL_FLAG(int, hidden_dim, 2048, "Network hidden dimension");
ABSL_FLAG(int, num_blocks, 8, "Network residual block count");
ABSL_FLAG(int, seed, 42, "RNG seed");

uint64_t ComputeFileHash(const std::string& filepath) {
  std::ifstream file(filepath, std::ios::binary);
  if (!file) {
    return 0;
  }
  uint64_t hash = 14695981039346656037ULL;
  char buffer[4096];
  while (file.read(buffer, sizeof(buffer))) {
    std::streamsize bytes_read = file.gcount();
    for (std::streamsize i = 0; i < bytes_read; ++i) {
      hash ^= static_cast<uint8_t>(buffer[i]);
      hash *= 1099511628211ULL;
    }
  }
  std::streamsize bytes_read = file.gcount();
  for (std::streamsize i = 0; i < bytes_read; ++i) {
    hash ^= static_cast<uint8_t>(buffer[i]);
    hash *= 1099511628211ULL;
  }
  return hash;
}

double CalculateMedian(std::vector<int> rounds) {
  if (rounds.empty()) return 0.0;
  std::sort(rounds.begin(), rounds.end());
  size_t n = rounds.size();
  if (n % 2 == 1) {
    return rounds[n / 2];
  } else {
    return (rounds[n / 2 - 1] + rounds[n / 2]) / 2.0;
  }
}

int main(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);

  std::string model_checkpoint = absl::GetFlag(FLAGS_model_checkpoint);
  if (model_checkpoint.empty()) {
    std::cerr << "Error: --model_checkpoint must be specified.\n";
    return 1;
  }

  int num_games = absl::GetFlag(FLAGS_games);
  int num_threads = absl::GetFlag(FLAGS_threads);
  int probe_simulations = absl::GetFlag(FLAGS_search_simulations);
  double puct_c = absl::GetFlag(FLAGS_puct_c);
  double utility_divisor = absl::GetFlag(FLAGS_utility_divisor);
  double search_opponent_temperature = absl::GetFlag(FLAGS_search_opponent_temperature);
  int seed = absl::GetFlag(FLAGS_seed);

  uint64_t fingerprint = ComputeFileHash(model_checkpoint);
  std::cout << absl::StrFormat("Loading checkpoint: %s (fingerprint: 0x%016llx)\n", model_checkpoint, fingerprint);

  auto game = open_spiel::LoadGame("dune_imperium");
  int64_t obs_size = game->GetType().provides_information_state_tensor
                         ? game->InformationStateTensorSize()
                         : game->ObservationTensorSize();
  int64_t action_size = game->NumDistinctActions();

  torch::Device device(torch::kCPU);
  if (torch::cuda::is_available()) {
    device = torch::Device(torch::kCUDA);
  }
  std::cout << "Running on device: " << (device.is_cuda() ? "CUDA GPU" : "CPU") << "\n";

  auto model = std::make_shared<open_spiel::SharedDunePolicyValueNetImpl>(
      obs_size, absl::GetFlag(FLAGS_hidden_dim), action_size, absl::GetFlag(FLAGS_num_blocks));
  try {
    torch::load(model, model_checkpoint, device);
    std::cout << "Loaded checkpoint model successfully.\n";
  } catch (const c10::Error& e) {
    std::cerr << "Failed to load model checkpoint: " << e.msg() << "\n";
    return 1;
  }
  model->to(device);
  model->eval();

  std::atomic<int> games_completed{0};
  std::atomic<int> games_finished{0};
  std::atomic<int> search_win_shares_scaled{0}; // Win shares multiplied by 1000
  std::atomic<int> policy_win_shares_scaled{0};

  std::mutex stats_mutex;
  std::vector<int> search_finish_rounds;
  std::vector<int> policy_finish_rounds;

  auto worker_fn = [&](int thread_id) {
    std::mt19937 rng(seed + 100 + thread_id);

    // Create evaluator and bots for this thread
    auto thread_evaluator = std::make_shared<open_spiel::DuneNNEvaluator>(
        model, device);

    std::vector<std::unique_ptr<open_spiel::DuneSearchSession>> thread_sessions;
    for (int p = 0; p < 4; ++p) {
      open_spiel::DuneSearchConfig bot_cfg;
      bot_cfg.max_simulations = probe_simulations;
      bot_cfg.puct_c = puct_c;
      bot_cfg.opponent_mode = open_spiel::SearchOpponentMode::kPolicy;
      bot_cfg.opponent_temperature = search_opponent_temperature;
      bot_cfg.temperature = 0.0;
      bot_cfg.relative_time_budget_ms = std::numeric_limits<double>::infinity();
      bot_cfg.seed = rng();
      bot_cfg.fixed_session_limit = probe_simulations;
      bot_cfg.purchase_combat_budget = 16;
      open_spiel::LoadCalibratedParameters(bot_cfg);
      bot_cfg.utility_divisor = utility_divisor;

      thread_sessions.push_back(std::make_unique<open_spiel::DuneSearchSession>(
          bot_cfg, thread_evaluator, open_spiel::DuneSearchBudgetMode::kFixedSessionSimulations));
    }

    while (true) {
      int game_idx = games_completed.fetch_add(1);
      if (game_idx >= num_games) {
        games_completed.fetch_sub(1);
        break;
      }

      // Determine seat assignment configuration
      // We choose 2 search players and 2 policy players out of 4 seats.
      // There are 6 combinations of choosing 2 seats. We rotate them systematically.
      int config_idx = game_idx % 6;
      bool is_search[4] = {false, false, false, false};
      if (config_idx == 0) { is_search[0] = is_search[1] = true; }
      else if (config_idx == 1) { is_search[0] = is_search[2] = true; }
      else if (config_idx == 2) { is_search[0] = is_search[3] = true; }
      else if (config_idx == 3) { is_search[1] = is_search[2] = true; }
      else if (config_idx == 4) { is_search[1] = is_search[3] = true; }
      else if (config_idx == 5) { is_search[2] = is_search[3] = true; }

      for (int p = 0; p < 4; ++p) {
        thread_sessions[p]->ResetSession("new_game");
      }

      auto state = game->NewInitialState();

      int last_logged_round = -1;
      while (!state->IsTerminal()) {
        const auto* dune_state = dynamic_cast<const open_spiel::dune_imperium::DuneImperiumState*>(state.get());
        if (dune_state) {
          int current_round = dune_state->GetCurrentRound();
          if (current_round > last_logged_round) {
            last_logged_round = current_round;
            std::cout << absl::StrFormat("  Game %d: Starting Round %d...\n", game_idx + 1, current_round) << std::flush;
          }
        }
        if (state->IsChanceNode()) {
          auto outcomes = state->ChanceOutcomes();
          open_spiel::Action choice = open_spiel::SampleAction(outcomes, absl::Uniform(rng, 0.0, 1.0)).first;
          state->ApplyAction(choice);
        } else if (state->CurrentPlayer() == open_spiel::kSimultaneousPlayerId) {
          std::vector<open_spiel::Action> joint_action;
          for (int p = 0; p < game->NumPlayers(); ++p) {
            std::vector<open_spiel::Action> actions = state->LegalActions(p);
            if (actions.empty()) {
              joint_action.push_back(0);
            } else {
              std::uniform_int_distribution<int> dis(0, actions.size() - 1);
              joint_action.push_back(actions[dis(rng)]);
            }
          }
          state->ApplyActions(joint_action);
        } else {
          open_spiel::Player cur_player = state->CurrentPlayer();
          open_spiel::Action action = open_spiel::kInvalidAction;
          if (is_search[cur_player]) {
            open_spiel::DuneSearchResult res = thread_sessions[cur_player]->SearchAndSelect(*state);
            action = res.diagnostics.selected_action;
          } else {
            open_spiel::ActionsAndProbs prior = thread_evaluator->Prior(*state);
            double max_p = -1.0;
            for (const auto& ap : prior) {
              if (ap.second > max_p) {
                max_p = ap.second;
                action = ap.first;
              }
            }
          }
          state->ApplyAction(action);
        }
      }

      // Determine winner(s)
      std::vector<double> returns = state->Returns();
      double max_return = -9999.0;
      std::vector<int> winners;
      for (int p = 0; p < 4; ++p) {
        if (returns[p] > max_return) {
          max_return = returns[p];
          winners = {p};
        } else if (returns[p] == max_return) {
          winners.push_back(p);
        }
      }

      const auto* dune_state = dynamic_cast<const open_spiel::dune_imperium::DuneImperiumState*>(state.get());
      int round = dune_state ? dune_state->GetCurrentRound() : -1;

      // Compute win increment in scaled integer units (scaled by 1000 to avoid float atomics)
      int win_increment = 1000 / winners.size();
      for (int w : winners) {
        if (is_search[w]) {
          search_win_shares_scaled.fetch_add(win_increment);
        } else {
          policy_win_shares_scaled.fetch_add(win_increment);
        }
      }

      {
        std::lock_guard<std::mutex> lock(stats_mutex);
        for (int w : winners) {
          if (is_search[w]) {
            search_finish_rounds.push_back(round);
          } else {
            policy_finish_rounds.push_back(round);
          }
        }
      }

      int current_finished = games_finished.fetch_add(1) + 1;
      std::cout << absl::StrFormat("Finished game %d / %d on thread %d. Winner(s):", current_finished, num_games, thread_id);
      for (int w : winners) {
        std::cout << absl::StrFormat(" P%d (%s)", w, is_search[w] ? "MCTS" : "Policy");
      }
      std::cout << absl::StrFormat(" in round %d.\n", round) << std::flush;
    }
  };

  std::cout << "Starting " << num_games << " match games on " << num_threads << " threads...\n";
  std::vector<std::thread> workers;
  for (int i = 0; i < num_threads; ++i) {
    workers.emplace_back(worker_fn, i);
  }

  for (auto& t : workers) {
    t.join();
  }

  double search_win_shares = search_win_shares_scaled.load() / 1000.0;
  double policy_win_shares = policy_win_shares_scaled.load() / 1000.0;
  double total_win_shares = search_win_shares + policy_win_shares;

  double search_winrate = total_win_shares > 0.0 ? search_win_shares / total_win_shares : 0.0;
  double policy_winrate = total_win_shares > 0.0 ? policy_win_shares / total_win_shares : 0.0;

  double median_search_round = CalculateMedian(search_finish_rounds);
  double median_policy_round = CalculateMedian(policy_finish_rounds);

  std::cout << "\n========================================\n";
  std::cout << "      MATCHUP RESULTS: MCTS VS POLICY   \n";
  std::cout << "========================================\n";
  std::cout << absl::StrFormat("Total games played:  %d\n", num_games);
  std::cout << absl::StrFormat("Search Win Shares:   %.2f (Winrate: %.2f%%)\n", search_win_shares, search_winrate * 100.0);
  std::cout << absl::StrFormat("Policy Win Shares:   %.2f (Winrate: %.2f%%)\n", policy_win_shares, policy_winrate * 100.0);
  std::cout << "\n";
  std::cout << absl::StrFormat("MCTS Winner Median Round:   %.1f\n", median_search_round);
  std::cout << absl::StrFormat("Policy Winner Median Round: %.1f\n", median_policy_round);
  std::cout << "========================================\n";

  return 0;
}
