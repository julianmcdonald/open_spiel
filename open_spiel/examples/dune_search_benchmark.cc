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
#include <sstream>
#include <cctype>
#include <functional>
#include <mutex>
#include <map>

#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"
#include "open_spiel/games/dune_imperium/dune_imperium.h"
#include "open_spiel/abseil-cpp/absl/strings/str_format.h"
#include "open_spiel/abseil-cpp/absl/flags/flag.h"
#include "open_spiel/abseil-cpp/absl/flags/parse.h"
#include "open_spiel/abseil-cpp/absl/random/distributions.h"
#include <torch/torch.h>

#include "dune_network.h"
#include "dune_evaluator.h"
#include "dune_puct_is_mcts.h"

ABSL_FLAG(std::string, model_checkpoint, "", "Path to load search agent model checkpoint.");
ABSL_FLAG(std::string, opponent_checkpoint, "", "Path to load opponent model checkpoint. If empty, the search agent model is reused. If 'random', random opponents are used.");
ABSL_FLAG(int, games, 1000, "How many games to play in total.");
ABSL_FLAG(int, threads, 8, "How many threads to run.");
ABSL_FLAG(int, hidden_dim, 2048, "Search agent model hidden dimension.");
ABSL_FLAG(int, num_blocks, 8, "Search agent model residual blocks count.");
ABSL_FLAG(int, opp_hidden_dim, -1, "Opponent model hidden dimension. If -1, inherits hidden_dim.");
ABSL_FLAG(int, opp_num_blocks, -1, "Opponent model block count. If -1, inherits num_blocks.");
ABSL_FLAG(int, max_simulations, 50, "MCTS simulation budget per move.");
ABSL_FLAG(double, puct_c, 1.0, "PUCT exploration constant.");
ABSL_FLAG(int, max_world_samples, -1, "Number of cached world samples for determinization (-1 = fresh resampling).");
ABSL_FLAG(double, value_scale, 4.0, "Scaling factor for leaf neural values (couples with PUCT bot's internal return scaling).");
ABSL_FLAG(double, temperature, 0.0, "Softmax temperature for final move choice (0.0 = greedy).");
ABSL_FLAG(double, opponent_temperature, 0.0, "Softmax temperature for opponent action selection (0.0 = greedy).");
ABSL_FLAG(double, dirichlet_epsilon, 0.0, "Dirichlet noise weight at root.");
ABSL_FLAG(double, dirichlet_alpha, 0.3, "Dirichlet noise alpha.");
ABSL_FLAG(bool, rotate_seat, true, "Rotate the seat of the search agent across games.");
ABSL_FLAG(bool, use_opponent_model, false, "Whether non-search players in simulation follow the PPO prior policy instead of PUCT.");
ABSL_FLAG(bool, verbose_diagnostics, true, "Print IS-MCTS node-reuse and depth diagnostics periodically.");

namespace open_spiel {
namespace {

class DuneGreedyBot : public Bot {
 public:
  DuneGreedyBot(std::unique_ptr<DuneNNEvaluator> evaluator, int seed, double temperature)
      : evaluator_(std::move(evaluator)), rng_(seed), temperature_(temperature) {}
  
  Action Step(const State& state) override {
    if (state.IsTerminal()) return kInvalidAction;
    ActionsAndProbs prior = evaluator_->Prior(state);
    return SelectBestAction(prior, state);
  }
  
  bool ProvidesPolicy() override { return true; }
  
  ActionsAndProbs GetPolicy(const State& state) override { 
    return evaluator_->Prior(state); 
  }
  
  std::pair<ActionsAndProbs, Action> StepWithPolicy(const State& state) override {
    ActionsAndProbs policy = GetPolicy(state);
    Action chosen = SelectBestAction(policy, state);
    return {policy, chosen};
  }
  
  void Restart() override {}
  void RestartAt(const State& state) override {}
  
 private:
  double RandomNumber() {
    return absl::Uniform(rng_, 0.0, 1.0);
  }

  Action SelectBestAction(const ActionsAndProbs& policy, const State& state) {
    if (policy.empty()) {
      auto legals = state.LegalActions();
      return legals.empty() ? kInvalidAction : legals.front();
    }
    double temp = temperature_;
    if (temp <= 0.0) {
      Action best_action = policy[0].first;
      double best_prob = policy[0].second;
      for (const auto& ap : policy) {
        if (ap.second > best_prob) {
          best_prob = ap.second;
          best_action = ap.first;
        }
      }
      return best_action;
    } else {
      ActionsAndProbs scaled_policy;
      scaled_policy.reserve(policy.size());
      double sum = 0.0;
      double inv_temp = 1.0 / temp;
      for (const auto& ap : policy) {
        double p = std::pow(std::max(ap.second, 1e-12), inv_temp);
        scaled_policy.push_back({ap.first, p});
        sum += p;
      }
      for (auto& ap : scaled_policy) {
        ap.second /= sum;
      }
      return SampleAction(scaled_policy, RandomNumber()).first;
    }
  }

  std::unique_ptr<DuneNNEvaluator> evaluator_;
  std::mt19937 rng_;
  double temperature_;
};

class DuneRandomBot : public Bot {
 public:
  DuneRandomBot(int seed) : rng_(seed) {}
  Action Step(const State& state) override {
    if (state.IsTerminal()) return kInvalidAction;
    auto legals = state.LegalActions();
    if (legals.empty()) return kInvalidAction;
    return legals[absl::Uniform(rng_, 0u, legals.size())];
  }
  bool ProvidesPolicy() override { return false; }
  ActionsAndProbs GetPolicy(const State& state) override { return {}; }
  std::pair<ActionsAndProbs, Action> StepWithPolicy(const State& state) override {
    return {{}, Step(state)};
  }
  void Restart() override {}
  void RestartAt(const State& state) override {}
 private:
  std::mt19937 rng_;
};

struct GameStats {
  int search_wins = 0;
  int opponent_wins[3] = {0, 0, 0};
  int search_wins_by_seat[4] = {0, 0, 0, 0};
  int games_by_seat[4] = {0, 0, 0, 0};
  int search_placements[4] = {0, 0, 0, 0};
  double search_return_sum = 0.0;
  double search_return_by_seat[4] = {0.0, 0.0, 0.0, 0.0};
  int search_swordmasters = 0;
  int opponent_swordmasters = 0;
  std::vector<int> rounds_played;
  double search_step_time_sum = 0.0;
  int search_steps_count = 0;
  double abs_terminal_return_sum = 0.0;
  int64_t terminal_returns_count = 0;
};

void WorkerThread(
    int thread_id,
    std::shared_ptr<const Game> game,
    std::shared_ptr<SharedDunePolicyValueNetImpl> search_model,
    std::shared_ptr<SharedDunePolicyValueNetImpl> opponent_model,
    std::atomic<int>& next_game_id,
    int total_games,
    std::atomic<int>& completed_games,
    std::mutex& log_mutex,
    GameStats& global_stats,
    std::mutex& stats_mutex) {
  
  std::random_device rd;
  std::mt19937 rng(rd() ^ thread_id);
  torch::InferenceMode inference_guard;

  // Local thread stats accumulation to reduce lock contention
  GameStats thread_stats;

  while (true) {
    int g = next_game_id++;
    if (g >= total_games) break;

    bool rotate_seat = absl::GetFlag(FLAGS_rotate_seat);
    int search_seat = rotate_seat ? (g % 4) : 0;
    thread_stats.games_by_seat[search_seat]++;

    torch::Device device = torch::cuda::is_available() ? torch::Device(torch::kCUDA) : torch::Device(torch::kCPU);
    
    // Create thread-local, bot-specific evaluator wrapping the shared search model
    auto search_evaluator = std::make_shared<DuneNNEvaluator>(
        search_model, device, absl::GetFlag(FLAGS_value_scale), 10.0f);

    std::vector<std::unique_ptr<Bot>> bots(4);
    for (int p = 0; p < 4; ++p) {
      if (p == search_seat) {
        bots[p] = std::make_unique<DunePUCTISMCTSBot>(
            /*seed=*/rng(),
            search_evaluator,
            absl::GetFlag(FLAGS_puct_c),
            absl::GetFlag(FLAGS_max_simulations),
            absl::GetFlag(FLAGS_max_world_samples),
            absl::GetFlag(FLAGS_temperature),
            absl::GetFlag(FLAGS_dirichlet_epsilon),
            absl::GetFlag(FLAGS_dirichlet_alpha),
            1.0,
            /*use_observation_string=*/true,
            DuneISMCTSFinalPolicyType::kNormalizedVisitCount,
            absl::GetFlag(FLAGS_use_opponent_model),
            absl::GetFlag(FLAGS_opponent_temperature),
            absl::GetFlag(FLAGS_verbose_diagnostics)
        );
      } else {
        if (opponent_model != nullptr) {
          auto local_opp_eval = std::make_unique<DuneNNEvaluator>(
              opponent_model, device, 1.0, 10.0f);
          bots[p] = std::make_unique<DuneGreedyBot>(
              std::move(local_opp_eval), rng(),
              absl::GetFlag(FLAGS_opponent_temperature));
        } else {
          bots[p] = std::make_unique<DuneRandomBot>(rng());
        }
      }
    }

    std::unique_ptr<State> state = game->NewInitialState();
    int game_length = 0;

    while (!state->IsTerminal()) {
      ++game_length;
      if (game_length > 5000) {
        std::cerr << "Infinite loop guard hit in thread " << thread_id << "!\n";
        std::abort();
      }

      if (state->IsChanceNode()) {
        auto outcomes = state->ChanceOutcomes();
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
      Action chosen_action = -1;

      if (current_player == search_seat) {
        auto step_start = std::chrono::steady_clock::now();
        chosen_action = bots[current_player]->Step(*state);
        auto step_end = std::chrono::steady_clock::now();
        double step_duration = std::chrono::duration<double>(step_end - step_start).count();
        thread_stats.search_step_time_sum += step_duration;
        thread_stats.search_steps_count++;
      } else {
        chosen_action = bots[current_player]->Step(*state);
      }

      state->ApplyAction(chosen_action);
    }

    std::vector<double> returns = state->Returns();
    for (double r : returns) {
      thread_stats.abs_terminal_return_sum += std::abs(r);
      thread_stats.terminal_returns_count++;
    }
    int winner = -1;
    double max_return = -999.0;
    for (int p = 0; p < 4; ++p) {
      if (returns[p] > max_return) {
        max_return = returns[p];
        winner = p;
      }
    }

    if (winner == search_seat) {
      thread_stats.search_wins++;
      thread_stats.search_wins_by_seat[search_seat]++;
    } else {
      int relative_idx = (winner - search_seat + 4) % 4 - 1;
      if (relative_idx >= 0 && relative_idx < 3) {
        thread_stats.opponent_wins[relative_idx]++;
      }
    }

    thread_stats.search_return_sum += returns[search_seat];
    thread_stats.search_return_by_seat[search_seat] += returns[search_seat];

    int rank = 0;
    for (int p = 0; p < 4; ++p) {
      if (returns[p] > returns[search_seat]) {
        ++rank;
      }
    }
    if (rank >= 0 && rank < 4) {
      thread_stats.search_placements[rank]++;
    }

    const auto* dune_state = dynamic_cast<const dune_imperium::DuneImperiumState*>(state.get());
    if (dune_state != nullptr) {
      for (int p = 0; p < 4; ++p) {
        if (dune_state->HasSwordmaster(p)) {
          if (p == search_seat) {
            thread_stats.search_swordmasters++;
          } else {
            thread_stats.opponent_swordmasters++;
          }
        }
      }
      thread_stats.rounds_played.push_back(dune_state->GetCurrentRound());
    }

    // Flush stats and perform intermediate logging
    int comp = ++completed_games;
    if (comp % 25 == 0 || comp == total_games) {
      std::lock_guard<std::mutex> stats_lock(stats_mutex);
      global_stats.search_wins += thread_stats.search_wins;
      for (int p = 0; p < 3; ++p) global_stats.opponent_wins[p] += thread_stats.opponent_wins[p];
      for (int s = 0; s < 4; ++s) {
        global_stats.search_wins_by_seat[s] += thread_stats.search_wins_by_seat[s];
        global_stats.games_by_seat[s] += thread_stats.games_by_seat[s];
        global_stats.search_return_by_seat[s] += thread_stats.search_return_by_seat[s];
      }
      for (int r = 0; r < 4; ++r) global_stats.search_placements[r] += thread_stats.search_placements[r];
      global_stats.search_return_sum += thread_stats.search_return_sum;
      global_stats.search_swordmasters += thread_stats.search_swordmasters;
      global_stats.opponent_swordmasters += thread_stats.opponent_swordmasters;
      global_stats.rounds_played.insert(global_stats.rounds_played.end(), thread_stats.rounds_played.begin(), thread_stats.rounds_played.end());
      global_stats.search_step_time_sum += thread_stats.search_step_time_sum;
      global_stats.search_steps_count += thread_stats.search_steps_count;
      global_stats.abs_terminal_return_sum += thread_stats.abs_terminal_return_sum;
      global_stats.terminal_returns_count += thread_stats.terminal_returns_count;

      thread_stats = GameStats();

      std::lock_guard<std::mutex> log_lock(log_mutex);
      double avg_step_time = global_stats.search_steps_count > 0 ? (global_stats.search_step_time_sum / global_stats.search_steps_count) : 0.0;
      double winrate = (global_stats.search_wins * 100.0) / comp;
      double search_sm_rate = (global_stats.search_swordmasters * 100.0) / comp;
      double opp_sm_rate = (global_stats.opponent_swordmasters * 100.0) / (3.0 * comp);

      std::cout << absl::StrFormat("[Progress] Games: %d/%d | Search Winrate: %.2f%% | Search SM Rate: %.2f%% | Opp SM Rate: %.2f%% | Avg Search Step Time: %.4fs\n",
                                   comp, total_games, winrate, search_sm_rate, opp_sm_rate, avg_step_time) << std::flush;
    }
  }

  // Merge remaining stats on completion
  {
    std::lock_guard<std::mutex> stats_lock(stats_mutex);
    global_stats.search_wins += thread_stats.search_wins;
    for (int p = 0; p < 3; ++p) global_stats.opponent_wins[p] += thread_stats.opponent_wins[p];
    for (int s = 0; s < 4; ++s) {
      global_stats.search_wins_by_seat[s] += thread_stats.search_wins_by_seat[s];
      global_stats.games_by_seat[s] += thread_stats.games_by_seat[s];
      global_stats.search_return_by_seat[s] += thread_stats.search_return_by_seat[s];
    }
    for (int r = 0; r < 4; ++r) global_stats.search_placements[r] += thread_stats.search_placements[r];
    global_stats.search_return_sum += thread_stats.search_return_sum;
    global_stats.search_swordmasters += thread_stats.search_swordmasters;
    global_stats.opponent_swordmasters += thread_stats.opponent_swordmasters;
    global_stats.rounds_played.insert(global_stats.rounds_played.end(), thread_stats.rounds_played.begin(), thread_stats.rounds_played.end());
    global_stats.search_step_time_sum += thread_stats.search_step_time_sum;
    global_stats.search_steps_count += thread_stats.search_steps_count;
    global_stats.abs_terminal_return_sum += thread_stats.abs_terminal_return_sum;
    global_stats.terminal_returns_count += thread_stats.terminal_returns_count;
  }
}

} // namespace
} // namespace open_spiel

int main(int argc, char* argv[]) {
  absl::ParseCommandLine(argc, argv);

  // Set PyTorch thread limit to 1 to avoid thread contention across CPU-bound forward runs
  at::set_num_threads(1);
  at::set_num_interop_threads(1);

  std::string model_ckpt = absl::GetFlag(FLAGS_model_checkpoint);
  if (model_ckpt.empty()) {
    std::cerr << "Error: --model_checkpoint is required!\n";
    return 1;
  }

  int hidden_dim = absl::GetFlag(FLAGS_hidden_dim);
  int num_blocks = absl::GetFlag(FLAGS_num_blocks);
  int opp_hidden_dim = absl::GetFlag(FLAGS_opp_hidden_dim);
  if (opp_hidden_dim < 0) opp_hidden_dim = hidden_dim;
  int opp_num_blocks = absl::GetFlag(FLAGS_opp_num_blocks);
  if (opp_num_blocks < 0) opp_num_blocks = num_blocks;

  std::shared_ptr<const open_spiel::Game> game = open_spiel::LoadGame("dune_imperium");
  int64_t obs_size = game->InformationStateTensorSize();
  int64_t action_size = game->NumDistinctActions();

  torch::Device device = torch::cuda::is_available() ? torch::Device(torch::kCUDA) : torch::Device(torch::kCPU);

  std::cout << "Loading search model weights from: " << model_ckpt << " on device " << device << "\n";
  auto search_model = std::make_shared<open_spiel::SharedDunePolicyValueNetImpl>(obs_size, hidden_dim, action_size, num_blocks);
  search_model->eval();
  try {
    torch::serialize::InputArchive archive;
    archive.load_from(model_ckpt, device);
    search_model->load(archive);
  } catch (const c10::Error& e) {
    std::cerr << "Failed to load search model weights:\n" << e.msg() << "\n";
    return 1;
  }
  search_model->to(device);

  std::shared_ptr<open_spiel::SharedDunePolicyValueNetImpl> opponent_model = nullptr;
  std::string opp_ckpt = absl::GetFlag(FLAGS_opponent_checkpoint);
  if (opp_ckpt != "random") {
    if (opp_ckpt.empty()) {
      opp_ckpt = model_ckpt;
    }
    std::cout << "Loading opponent model weights from: " << opp_ckpt << " on device " << device << "\n";
    opponent_model = std::make_shared<open_spiel::SharedDunePolicyValueNetImpl>(obs_size, opp_hidden_dim, action_size, opp_num_blocks);
    opponent_model->eval();
    try {
      torch::serialize::InputArchive archive;
      archive.load_from(opp_ckpt, device);
      opponent_model->load(archive);
    } catch (const c10::Error& e) {
      std::cerr << "Failed to load opponent model weights:\n" << e.msg() << "\n";
      return 1;
    }
    opponent_model->to(device);
  } else {
    std::cout << "Running against uniform random opponents.\n";
  }

  int total_games = absl::GetFlag(FLAGS_games);
  int num_threads = absl::GetFlag(FLAGS_threads);
  if (num_threads <= 0) {
    num_threads = std::max(1, (int)std::thread::hardware_concurrency());
  }

  std::cout << "\nStarting evaluation of " << total_games << " games using " << num_threads << " threads...\n\n";

  std::atomic<int> next_game_id{0};
  std::atomic<int> completed_games{0};
  std::mutex log_mutex;
  std::mutex stats_mutex;
  open_spiel::GameStats global_stats;

  std::vector<std::thread> worker_threads;
  worker_threads.reserve(num_threads);
  for (int t = 0; t < num_threads; ++t) {
    worker_threads.emplace_back(
        open_spiel::WorkerThread, t, game, search_model, opponent_model,
        std::ref(next_game_id), total_games, std::ref(completed_games),
        std::ref(log_mutex), std::ref(global_stats), std::ref(stats_mutex));
  }

  for (auto& th : worker_threads) {
    if (th.joinable()) th.join();
  }

  // Print summary results
  std::cout << "\n======================================================================\n";
  std::cout << "Evaluation Completed!\n";
  std::cout << "Total Games:         " << total_games << "\n";
  std::cout << "Search Checkpoint:   " << model_ckpt << "\n";
  std::cout << "Opponent Checkpoint: " << (opponent_model == nullptr ? "Uniform Random Agents" : (opp_ckpt == model_ckpt ? "Same as search (Self-play evaluation)" : opp_ckpt)) << "\n";
  std::cout << "======================================================================\n\n";

  std::cout << "Overall Winrates:\n";
  std::cout << absl::StrFormat("  Search Agent:  %.2f%% (%d/%d wins)\n",
                               (global_stats.search_wins * 100.0 / total_games),
                               global_stats.search_wins, total_games);
  std::cout << absl::StrFormat("  Opponent 1:    %.2f%% (%d/%d wins)\n",
                               (global_stats.opponent_wins[0] * 100.0 / total_games),
                               global_stats.opponent_wins[0], total_games);
  std::cout << absl::StrFormat("  Opponent 2:    %.2f%% (%d/%d wins)\n",
                               (global_stats.opponent_wins[1] * 100.0 / total_games),
                               global_stats.opponent_wins[1], total_games);
  std::cout << absl::StrFormat("  Opponent 3:    %.2f%% (%d/%d wins)\n",
                               (global_stats.opponent_wins[2] * 100.0 / total_games),
                               global_stats.opponent_wins[2], total_games);

  std::cout << "\nSearch Agent Winrate by Seat:\n";
  for (int p = 0; p < 4; ++p) {
    double seat_winrate = global_stats.games_by_seat[p] > 0
                              ? (global_stats.search_wins_by_seat[p] * 100.0 / global_stats.games_by_seat[p])
                              : 0.0;
    double seat_equity = global_stats.games_by_seat[p] > 0
                             ? (global_stats.search_return_by_seat[p] / global_stats.games_by_seat[p])
                             : 0.0;
    std::cout << absl::StrFormat("  Seat P%d: %.2f%% (%d/%d wins), equity %.4f\n",
                                 p, seat_winrate, global_stats.search_wins_by_seat[p],
                                 global_stats.games_by_seat[p], seat_equity);
  }

  std::cout << "\nSearch Agent Placement:\n";
  std::cout << absl::StrFormat("  Mean Return: %.4f\n", (global_stats.search_return_sum / total_games));
  for (int r = 0; r < 4; ++r) {
    std::cout << absl::StrFormat("  Place %d: %.2f%% (%d/%d)\n",
                                 r + 1, (global_stats.search_placements[r] * 100.0 / total_games),
                                 global_stats.search_placements[r], total_games);
  }

  std::cout << "\nSwordmaster Acquisition Rates:\n";
  std::cout << absl::StrFormat("  Search Agent:  %.2f%% (%d/%d opportunities)\n",
                               (global_stats.search_swordmasters * 100.0 / total_games),
                               global_stats.search_swordmasters, total_games);
  std::cout << absl::StrFormat("  Opponent Bots: %.2f%% (%d/%d opportunities)\n",
                               (global_stats.opponent_swordmasters * 100.0 / (3 * total_games)),
                               global_stats.opponent_swordmasters, 3 * total_games);

  // Round distribution histogram
  std::cout << "\nGame End Round Distribution:\n";
  std::map<int, int> round_counts;
  for (int r : global_stats.rounds_played) {
    round_counts[r]++;
  }
  for (const auto& pair : round_counts) {
    std::cout << absl::StrFormat("  Round %2d: %5d games (%.2f%%)\n",
                                 pair.first, pair.second, (pair.second * 100.0 / total_games));
  }

  double avg_step_time = global_stats.search_steps_count > 0
                             ? (global_stats.search_step_time_sum / global_stats.search_steps_count)
                             : 0.0;
  std::cout << absl::StrFormat("\nAverage Search Step Time: %.4fs (across %d steps)\n",
                               avg_step_time, global_stats.search_steps_count);

  double mean_abs_leaf_value = open_spiel::DuneNNEvaluator::global_num_leaf_evaluations.load() > 0
      ? (open_spiel::DuneNNEvaluator::global_abs_leaf_value_sum.load() / open_spiel::DuneNNEvaluator::global_num_leaf_evaluations.load())
      : 0.0;
  double mean_abs_terminal_return = global_stats.terminal_returns_count > 0
      ? (global_stats.abs_terminal_return_sum / global_stats.terminal_returns_count)
      : 0.0;
  double value_scale_used = absl::GetFlag(FLAGS_value_scale);
  double raw_mean_abs_leaf_value = value_scale_used > 0.0 ? (mean_abs_leaf_value / value_scale_used) : mean_abs_leaf_value;
  std::cout << absl::StrFormat("Mean |leaf value|: %.4f (scaled) / %.4f (raw) vs Mean |terminal return|: %.4f (value_scale = %.1f)\n",
                               mean_abs_leaf_value, raw_mean_abs_leaf_value, mean_abs_terminal_return, value_scale_used);

  return 0;
}
