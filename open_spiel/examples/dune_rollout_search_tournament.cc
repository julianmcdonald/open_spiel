#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <memory>
#include <chrono>
#include <mutex>
#include <shared_mutex>
#include <atomic>
#include <thread>
#include <future>
#include <numeric>
#include <cmath>
#include <iomanip>
#include <algorithm>

#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"
#include "open_spiel/utils/json.h"
#include "open_spiel/games/dune_imperium/dune_imperium.h"
#include "open_spiel/games/dune_imperium/dune_imperium_common.h"
#include "absl/flags/flag.h"
#include "absl/flags/parse.h"

#include <torch/torch.h>

#include "dune_network.h"
#include "dune_evaluator.h"
#include "dune_batched_evaluator.h"
#include "dune_semantic_action_scorer.h"
#include "dune_eval_action_selection.h"
#include "dune_seed_utils.h"
#include "dune_sha256.h"
#include "dune_puct_is_mcts.h"

using namespace open_spiel;
using namespace open_spiel::dune_imperium;

// Link satisfaction flags
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
ABSL_FLAG(uint64_t, shaping_start_env_steps, 206830543, "");
ABSL_FLAG(uint64_t, shaping_decay_env_steps, 0, "");
ABSL_FLAG(bool, diagnostics_only, false, "");

// Tournament flags
ABSL_FLAG(std::string, model_checkpoint,
          "/run/media/warcr/Storage/dune_drl_runtime/round7/u21328_continuation_20260921_004103/checkpoints/ppo_model_update_22263.pt",
          "Path to Champion model checkpoint (U22263).");
ABSL_FLAG(int, games, 100, "Total number of games to play.");
ABSL_FLAG(int, threads, 8, "Number of concurrent game worker threads.");
ABSL_FLAG(int, num_evaluators, 4, "Number of parallel batched evaluator coordinators.");
ABSL_FLAG(int, target_batch_size, 128, "GPU evaluator target batch size.");
ABSL_FLAG(int, top_k_actions, 3, "Number of top candidate actions to evaluate with rollouts.");
ABSL_FLAG(int, rollouts_per_action, 64, "Number of Monte Carlo rollouts per candidate action.");
ABSL_FLAG(double, min_override_margin, 0.15,
          "Minimum mean utility advantage needed for candidate to override raw policy.");
ABSL_FLAG(bool, high_stakes_only, true,
          "Only search on high-stakes strategic turns (Agent placement, combat deploy, card buy).");
ABSL_FLAG(bool, rotate_seat, true, "Whether to rotate the searching agent's seat across games.");
ABSL_FLAG(int, search_seat, 0, "Fixed search seat if rotate_seat is false.");
ABSL_FLAG(uint64_t, master_seed, 20260921, "Master random seed.");
ABSL_FLAG(std::string, output_json,
          "/home/warcr/projects/dune_drl/docs/experiment_records/rollout_search_100_games_receipt.json",
          "Path to write structured tournament results JSON.");

struct GameOutcome {
  int game_id = 0;
  int search_seat = 0;
  int winner_seat = 0;
  bool search_won = false;
  double search_utility = 0.0;
  int search_vp = 0;
  std::vector<double> final_returns;
  std::vector<int> final_vps;
  int total_searches = 0;
  int total_overrides = 0;
  int total_decisions = 0;
  int ending_round = 0;
  double duration_seconds = 0.0;
};

// Pick greedy action from prior over legal actions
inline Action PickGreedyAction(const ActionsAndProbs& prior, const std::vector<Action>& legal_actions) {
  if (legal_actions.empty()) return kInvalidAction;
  if (prior.empty()) return legal_actions.front();
  double best_p = -1.0;
  Action best_a = legal_actions.front();
  for (Action a : legal_actions) {
    for (const auto& ap : prior) {
      if (ap.first == a) {
        if (ap.second > best_p) {
          best_p = ap.second;
          best_a = a;
        }
        break;
      }
    }
  }
  return best_a;
}

// Get top-K candidate actions sorted by prior probability
inline std::vector<Action> GetTopKActions(
    const ActionsAndProbs& prior,
    const std::vector<Action>& legal_actions,
    int k) {
  if (legal_actions.size() <= static_cast<size_t>(k)) {
    return legal_actions;
  }
  std::vector<std::pair<double, Action>> scored;
  scored.reserve(legal_actions.size());
  for (Action a : legal_actions) {
    double p = 0.0;
    for (const auto& ap : prior) {
      if (ap.first == a) {
        p = ap.second;
        break;
      }
    }
    scored.push_back({p, a});
  }
  std::sort(scored.rbegin(), scored.rend());
  std::vector<Action> top_actions;
  top_actions.reserve(k);
  for (int i = 0; i < k; ++i) {
    top_actions.push_back(scored[i].second);
  }
  return top_actions;
}

// Check for high-stakes decision points: Agent Placement, Combat Deployment, Reveal Purchase
inline bool IsHighStakesStrategicState(const State& state, Player searched_player) {
  if (state.CurrentPlayer() != searched_player) return false;
  std::vector<Action> legal_actions = state.LegalActions();
  if (legal_actions.size() < 2) return false;

  const auto* dune_state = dynamic_cast<const dune_imperium::DuneImperiumState*>(&state);
  if (!dune_state) return false;

  for (Action a : legal_actions) {
    std::string s = dune_state->ActionToString(searched_player, a);
    if (s.rfind("PlaceAgent", 0) == 0 ||
        s.rfind("SelectAgentCard", 0) == 0 ||
        s.rfind("Deploy ", 0) == 0 ||
        s.rfind("CombatCommit", 0) == 0 ||
        s.rfind("Buy", 0) == 0) {
      return true;
    }
  }
  return false;
}

// Run a single rollout to terminal from a cloned state
inline double SimulateRollout(
    State* sim_state,
    Player search_player,
    BatchedNNEvaluator* evaluator,
    uint64_t rollout_seed) {
  std::mt19937 rng(rollout_seed);
  int steps = 0;
  const int max_steps = 2500;

  while (!sim_state->IsTerminal() && steps < max_steps) {
    steps++;
    if (sim_state->IsChanceNode()) {
      auto chance_outcomes = sim_state->ChanceOutcomes();
      double u = std::generate_canonical<double, 53>(rng);
      Action ca = open_spiel::SampleAction(chance_outcomes, u).first;
      sim_state->ApplyAction(ca);
      continue;
    }
    std::vector<Action> legals = sim_state->LegalActions();
    if (legals.empty()) break;
    if (legals.size() == 1) {
      sim_state->ApplyAction(legals.front());
      continue;
    }
    ActionsAndProbs prior = evaluator->Prior(*sim_state);
    Action a = PickGreedyAction(prior, legals);
    sim_state->ApplyAction(a);
  }

  std::vector<double> ret = sim_state->Returns();
  if (static_cast<size_t>(search_player) < ret.size()) {
    return ret[search_player];
  }
  return 0.0;
}

struct SearchDecisionResult {
  Action action;
  bool overridden = false;
  double raw_mean_utility = 0.0;
  double chosen_mean_utility = 0.0;
};

// Policy-Guided Rollout Search decision with conservative override margin
inline SearchDecisionResult RolloutSearchDecision(
    const State& state,
    Player search_player,
    const std::vector<std::shared_ptr<BatchedNNEvaluator>>& evaluators,
    int top_k,
    int rollouts_per_action,
    double min_override_margin,
    uint64_t decision_seed) {
  std::vector<Action> legal_actions = state.LegalActions();
  if (legal_actions.empty()) return {kInvalidAction, false, 0.0, 0.0};
  if (legal_actions.size() == 1) return {legal_actions.front(), false, 0.0, 0.0};

  ActionsAndProbs raw_prior = evaluators.front()->Prior(state);
  Action raw_action = PickGreedyAction(raw_prior, legal_actions);

  std::vector<Action> candidates = GetTopKActions(raw_prior, legal_actions, top_k);
  if (candidates.size() <= 1) {
    return {raw_action, false, 0.0, 0.0};
  }

  // Ensure candidate 0 is always the baseline raw_action
  if (candidates.front() != raw_action) {
    auto it = std::find(candidates.begin(), candidates.end(), raw_action);
    if (it != candidates.end()) {
      std::swap(candidates.front(), *it);
    } else {
      candidates.insert(candidates.begin(), raw_action);
      if (candidates.size() > static_cast<size_t>(top_k)) {
        candidates.pop_back();
      }
    }
  }

  const auto* dune = dynamic_cast<const dune_imperium::DuneImperiumState*>(&state);

  struct RolloutTask {
    size_t c_idx;
    Action action;
    std::unique_ptr<State> sim_state;
    uint64_t wseed;
  };

  std::vector<RolloutTask> tasks;
  tasks.reserve(candidates.size() * rollouts_per_action);

  for (size_t c_idx = 0; c_idx < candidates.size(); ++c_idx) {
    Action cand_a = candidates[c_idx];
    for (int r = 0; r < rollouts_per_action; ++r) {
      uint64_t wseed = dune_seed::DeriveSeed(decision_seed, c_idx, r);
      std::unique_ptr<State> sim_state = nullptr;

      if (dune) {
        std::mt19937 wrng(wseed);
        auto rng_func = [&wrng]() {
          return std::generate_canonical<double, 53>(wrng);
        };
        sim_state = dune->ResampleFromInfostate(search_player, rng_func);
      }
      if (!sim_state) {
        sim_state = state.Clone();
      }

      sim_state->ApplyAction(cand_a);
      tasks.push_back({c_idx, cand_a, std::move(sim_state), wseed + 999});
    }
  }

  // Concurrently step all rollouts, striping across the pool of BatchedEvaluators
  std::vector<std::future<double>> futures;
  futures.reserve(tasks.size());

  for (size_t i = 0; i < tasks.size(); ++i) {
    auto* s_ptr = tasks[i].sim_state.get();
    uint64_t s_seed = tasks[i].wseed;
    auto* active_evaluator = evaluators[i % evaluators.size()].get();
    futures.push_back(std::async(
        std::launch::async,
        [s_ptr, search_player, active_evaluator, s_seed]() {
          return SimulateRollout(s_ptr, search_player, active_evaluator, s_seed);
        }));
  }

  std::vector<double> candidate_utility_sum(candidates.size(), 0.0);
  for (size_t i = 0; i < futures.size(); ++i) {
    double u = futures[i].get();
    candidate_utility_sum[tasks[i].c_idx] += u;
  }

  double raw_mean_u = candidate_utility_sum[0] / static_cast<double>(rollouts_per_action);
  Action best_action = raw_action;
  double best_mean_u = raw_mean_u;
  bool overridden = false;

  for (size_t c_idx = 1; c_idx < candidates.size(); ++c_idx) {
    double mean_u = candidate_utility_sum[c_idx] / static_cast<double>(rollouts_per_action);
    if (mean_u > raw_mean_u + min_override_margin && mean_u > best_mean_u) {
      best_mean_u = mean_u;
      best_action = candidates[c_idx];
      overridden = true;
    }
  }

  return {best_action, overridden, raw_mean_u, best_mean_u};
}

// Play a single full game
GameOutcome PlayTournamentGame(
    int game_id,
    int search_seat,
    const std::shared_ptr<const Game>& game,
    const std::vector<std::shared_ptr<BatchedNNEvaluator>>& evaluators,
    int thread_id,
    int top_k,
    int rollouts_per_action,
    double min_override_margin,
    bool high_stakes_only,
    uint64_t game_seed) {
  auto start_time = std::chrono::steady_clock::now();
  std::unique_ptr<State> state = game->NewInitialState();
  std::mt19937 chance_rng(game_seed);

  int searches = 0;
  int overrides = 0;
  int decisions = 0;
  uint64_t search_seed_counter = 0;
  auto* greedy_evaluator = evaluators[thread_id % evaluators.size()].get();

  while (!state->IsTerminal()) {
    if (state->IsChanceNode()) {
      auto chance_outcomes = state->ChanceOutcomes();
      double u = std::generate_canonical<double, 53>(chance_rng);
      Action ca = open_spiel::SampleAction(chance_outcomes, u).first;
      state->ApplyAction(ca);
      continue;
    }

    Player cur = state->CurrentPlayer();
    std::vector<Action> legals = state->LegalActions();
    if (legals.empty()) break;

    Action chosen_action = kInvalidAction;
    if (cur == search_seat) {
      decisions++;
      bool should_search = high_stakes_only
          ? IsHighStakesStrategicState(*state, cur)
          : (IsStrategicState(*state, cur) && legals.size() > 1);

      if (should_search && legals.size() > 1) {
        searches++;
        uint64_t dseed = dune_seed::DeriveSeed(game_seed, cur, search_seed_counter++);
        auto s_res = RolloutSearchDecision(
            *state, cur, evaluators, top_k, rollouts_per_action,
            min_override_margin, dseed);
        chosen_action = s_res.action;
        if (s_res.overridden) overrides++;
      } else {
        ActionsAndProbs prior = greedy_evaluator->Prior(*state);
        chosen_action = PickGreedyAction(prior, legals);
      }
    } else {
      ActionsAndProbs prior = greedy_evaluator->Prior(*state);
      chosen_action = PickGreedyAction(prior, legals);
    }

    state->ApplyAction(chosen_action);
  }

  auto end_time = std::chrono::steady_clock::now();
  double elapsed = std::chrono::duration<double>(end_time - start_time).count();

  GameOutcome outcome;
  outcome.game_id = game_id;
  outcome.search_seat = search_seat;
  outcome.final_returns = state->Returns();
  outcome.total_searches = searches;
  outcome.total_overrides = overrides;
  outcome.total_decisions = decisions;
  outcome.duration_seconds = elapsed;

  const auto* dune_state = dynamic_cast<const dune_imperium::DuneImperiumState*>(state.get());
  outcome.final_vps.resize(4, 0);
  for (int p = 0; p < 4; ++p) {
    if (dune_state) {
      outcome.final_vps[p] = dune_state->GetPlayerVp(p);
    }
  }
  if (dune_state && dune_state->IsTerminal()) {
    outcome.ending_round = dune_state->GetCurrentRound() - 1;
  } else if (dune_state) {
    outcome.ending_round = dune_state->GetCurrentRound();
  }

  int winner = 0;
  double max_ret = -999.0;
  for (int p = 0; p < 4; ++p) {
    if (outcome.final_returns[p] > max_ret) {
      max_ret = outcome.final_returns[p];
      winner = p;
    }
  }
  outcome.winner_seat = winner;
  outcome.search_won = (winner == search_seat);
  outcome.search_utility = outcome.final_returns[search_seat];
  outcome.search_vp = outcome.final_vps[search_seat];

  return outcome;
}

int main(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);

  const std::string model_path = absl::GetFlag(FLAGS_model_checkpoint);
  const int total_games = absl::GetFlag(FLAGS_games);
  const int num_threads = absl::GetFlag(FLAGS_threads);
  const int num_evaluators = std::max(1, absl::GetFlag(FLAGS_num_evaluators));
  const int target_batch_size = absl::GetFlag(FLAGS_target_batch_size);
  const int top_k = absl::GetFlag(FLAGS_top_k_actions);
  const int rollouts_per_action = absl::GetFlag(FLAGS_rollouts_per_action);
  const double min_override_margin = absl::GetFlag(FLAGS_min_override_margin);
  const bool high_stakes_only = absl::GetFlag(FLAGS_high_stakes_only);
  const bool rotate_seat = absl::GetFlag(FLAGS_rotate_seat);
  const int default_search_seat = absl::GetFlag(FLAGS_search_seat);
  const uint64_t master_seed = absl::GetFlag(FLAGS_master_seed);
  const std::string output_json_path = absl::GetFlag(FLAGS_output_json);

  std::cout << "================================================================================\n";
  std::cout << "DUNE: IMPERIUM -- 100-GAME TOURNAMENT (POLICY-GUIDED ROLLOUT SEARCH)\n";
  std::cout << "================================================================================\n";
  std::cout << "Model Checkpoint: " << model_path << "\n";
  std::cout << "Games: " << total_games << ", Threads: " << num_threads
            << ", Evaluators: " << num_evaluators << ", Target Batch Size: " << target_batch_size << "\n";
  std::cout << "Search Topology: 1 Searching Agent (Top-" << top_k << ", " << rollouts_per_action
            << " rollouts/act = " << (top_k * rollouts_per_action) << " rollouts/search) vs 3 Raw Greedy Opponents\n";
  std::cout << "Override Margin: >=" << std::fixed << std::setprecision(2) << min_override_margin
            << " utility advantage | High Stakes Only: " << (high_stakes_only ? "TRUE" : "FALSE") << "\n";
  std::cout << "Seat Rotation: " << (rotate_seat ? "TRUE (balanced 25 games/seat)" : "FALSE") << "\n";
  std::cout << "================================================================================\n";

  // Check CUDA
  torch::Device device(torch::kCPU);
  if (torch::cuda::is_available()) {
    device = torch::Device(torch::kCUDA, 0);
    std::cout << "[CUDA] Initialized on device: " << device << "\n";
  } else {
    std::cout << "[CPU] CUDA not available, falling back to CPU.\n";
  }

  // Load Model Replicas & Multi-Coordinator Pool
  std::vector<std::shared_ptr<SharedDunePolicyValueNetImpl>> models(num_evaluators);
  std::vector<std::unique_ptr<std::shared_mutex>> model_mutexes(num_evaluators);
  std::vector<std::shared_ptr<BatchedEvaluator>> coords(num_evaluators);
  std::vector<std::shared_ptr<BatchedNNEvaluator>> evaluators(num_evaluators);

  for (int e = 0; e < num_evaluators; ++e) {
    models[e] = std::make_shared<SharedDunePolicyValueNetImpl>(
        kFullPublicInformationStateSize, 2048, 2391, 8,
        /*nonlinear_value_head=*/false, /*with_aux_heads=*/false,
        /*head_init_seed=*/0, /*with_semantic_scorer=*/true);
    models[e]->to(device);
    torch::load(models[e], model_path, device);
    models[e]->eval();

    model_mutexes[e] = std::make_unique<std::shared_mutex>();
    coords[e] = std::make_shared<BatchedEvaluator>(
        models[e], target_batch_size, /*timeout_ms=*/1, device, model_mutexes[e].get(), 10.0f,
        /*device_synchronize=*/false, /*high_priority_stream=*/true,
        /*emit_batch_membership=*/false, /*rollout_amp=*/true, /*allow_tf32=*/true);
    evaluators[e] = std::make_shared<BatchedNNEvaluator>(coords[e], 10.0f);
  }
  std::cout << "[MODEL] Successfully initialized " << num_evaluators << " parallel evaluator coordinator(s).\n";

  std::shared_ptr<const Game> game = LoadGame("dune_imperium");

  // Thread pool execution
  std::vector<GameOutcome> outcomes(total_games);
  std::atomic<int> next_game_idx{0};
  std::atomic<int> completed_games(0);
  std::atomic<int> search_wins(0);
  std::atomic<double> search_utility_sum(0.0);
  std::atomic<int> ending_round_sum(0);
  std::mutex cout_mutex;

  auto tournament_start = std::chrono::steady_clock::now();

  auto worker = [&](int thread_id) {
    while (true) {
      int g = next_game_idx.fetch_add(1);
      if (g >= total_games) break;

      int s_seat = rotate_seat ? (g % 4) : default_search_seat;
      uint64_t gseed = dune_seed::DeriveSeed(master_seed, dune_seed::kStreamSearchSampling, g);

      GameOutcome res = PlayTournamentGame(
          g, s_seat, game, evaluators, thread_id, top_k, rollouts_per_action,
          min_override_margin, high_stakes_only, gseed);

      outcomes[g] = res;

      int done = completed_games.fetch_add(1) + 1;
      if (res.search_won) search_wins.fetch_add(1);
      ending_round_sum.fetch_add(res.ending_round);
      
      // Update running utility sum safely
      double cur_sum = search_utility_sum.load();
      while (!search_utility_sum.compare_exchange_weak(cur_sum, cur_sum + res.search_utility)) {}

      double current_win_rate = 100.0 * search_wins.load() / done;
      double current_mean_u = search_utility_sum.load() / done;
      double current_mean_round = static_cast<double>(ending_round_sum.load()) / done;

      std::lock_guard<std::mutex> lock(cout_mutex);
      std::cout << "Game [" << std::setw(3) << done << "/" << total_games << "] "
                << "Seat=" << res.search_seat << " "
                << (res.search_won ? "WIN " : "LOSS") << " "
                << "R=" << res.ending_round << " "
                << "Util=" << std::fixed << std::setprecision(2) << std::showpos << res.search_utility << " "
                << "VP=" << std::noshowpos << res.search_vp << " "
                << "(Searches: " << res.total_searches << ", Overrides: " << res.total_overrides << ", "
                << std::setprecision(1) << res.duration_seconds << "s) | "
                << "Running WR: " << std::setprecision(1) << current_win_rate << "% "
                << "MeanUtil: " << std::setprecision(3) << std::showpos << current_mean_u << " "
                << "MeanR: " << std::setprecision(2) << current_mean_round << "\n"
                << std::flush << std::noshowpos;
    }
  };

  std::vector<std::thread> workers;
  workers.reserve(num_threads);
  for (int t = 0; t < num_threads; ++t) {
    workers.emplace_back(worker, t);
  }
  for (auto& w : workers) {
    w.join();
  }

  auto tournament_end = std::chrono::steady_clock::now();
  double total_wall_time = std::chrono::duration<double>(tournament_end - tournament_start).count();

  // Aggregate analysis
  int total_wins = 0;
  int total_searches_all = 0;
  int total_overrides_all = 0;
  int total_ending_round = 0;
  int ending_round_le7 = 0;
  int ending_round_le8 = 0;
  double total_search_util = 0.0;
  std::vector<int> seat_wins(4, 0);
  std::vector<int> seat_games(4, 0);
  std::vector<double> seat_util(4, 0.0);
  std::vector<double> utilities;
  utilities.reserve(total_games);

  for (const auto& out : outcomes) {
    if (out.search_won) {
      total_wins++;
      seat_wins[out.search_seat]++;
    }
    seat_games[out.search_seat]++;
    seat_util[out.search_seat] += out.search_utility;
    total_search_util += out.search_utility;
    total_searches_all += out.total_searches;
    total_overrides_all += out.total_overrides;
    total_ending_round += out.ending_round;
    if (out.ending_round <= 7) ending_round_le7++;
    if (out.ending_round <= 8) ending_round_le8++;
    utilities.push_back(out.search_utility);
  }

  double win_rate = 100.0 * total_wins / total_games;
  double mean_util = total_search_util / total_games;
  double mean_round = static_cast<double>(total_ending_round) / total_games;
  double p_round_le7 = 100.0 * ending_round_le7 / total_games;
  double p_round_le8 = 100.0 * ending_round_le8 / total_games;
  double override_pct = (total_searches_all > 0)
      ? (100.0 * total_overrides_all / total_searches_all)
      : 0.0;

  // Compute Standard Error and 95% CI
  double sq_sum = 0.0;
  for (double u : utilities) {
    sq_sum += (u - mean_util) * (u - mean_util);
  }
  double std_dev = (total_games > 1) ? std::sqrt(sq_sum / (total_games - 1)) : 0.0;
  double se_util = std_dev / std::sqrt(total_games);
  double ci_lower = mean_util - 1.96 * se_util;
  double ci_upper = mean_util + 1.96 * se_util;

  std::cout << "\n================================================================================\n";
  std::cout << "TOURNAMENT SUMMARY REPORT (" << total_games << " GAMES)\n";
  std::cout << "================================================================================\n";
  std::cout << "Total Wall-Clock Time: " << std::fixed << std::setprecision(1) << total_wall_time << "s ("
            << (total_wall_time / 60.0) << " minutes)\n";
  std::cout << "Search Agent Win Rate: " << std::setprecision(2) << win_rate << "% ("
            << total_wins << "/" << total_games << ") [Expected random: 25.00%]\n";
  std::cout << "Mean Placement Utility: " << std::setprecision(4) << std::showpos << mean_util
            << " +/- " << std::noshowpos << (1.96 * se_util) << "\n";
  std::cout << "95% Confidence Interval: [" << std::setprecision(4) << std::showpos << ci_lower
            << ", " << ci_upper << "]\n" << std::noshowpos;
  std::cout << "Mean Ending Round: " << std::fixed << std::setprecision(2) << mean_round
            << " (<=7: " << std::setprecision(1) << p_round_le7 << "%, <=8: " << p_round_le8 << "%)\n";
  std::cout << "Searches Total: " << total_searches_all << " (mean "
            << std::fixed << std::setprecision(1) << (static_cast<double>(total_searches_all) / total_games)
            << " / game)\n";
  std::cout << "Overrides Total: " << total_overrides_all << " ("
            << std::setprecision(1) << override_pct << "% override rate)\n";

  std::cout << "\nSeat Breakdown:\n";
  for (int p = 0; p < 4; ++p) {
    double s_wr = seat_games[p] > 0 ? (100.0 * seat_wins[p] / seat_games[p]) : 0.0;
    double s_mu = seat_games[p] > 0 ? (seat_util[p] / seat_games[p]) : 0.0;
    std::cout << "  Seat " << p << ": " << seat_wins[p] << "/" << seat_games[p]
              << " (" << std::fixed << std::setprecision(1) << s_wr << "%) | Mean Util: "
              << std::setprecision(3) << std::showpos << s_mu << "\n" << std::noshowpos;
  }
  std::cout << "================================================================================\n";

  // Write JSON receipt
  if (!output_json_path.empty()) {
    json::Object root_obj;
    root_obj["tournament_type"] = json::Value("policy_guided_rollout_search_vs_greedy");
    root_obj["model_checkpoint"] = json::Value(model_path);
    root_obj["total_games"] = json::Value(static_cast<int64_t>(total_games));
    root_obj["num_threads"] = json::Value(static_cast<int64_t>(num_threads));
    root_obj["top_k_actions"] = json::Value(static_cast<int64_t>(top_k));
    root_obj["rollouts_per_action"] = json::Value(static_cast<int64_t>(rollouts_per_action));
    root_obj["min_override_margin"] = json::Value(min_override_margin);
    root_obj["high_stakes_only"] = json::Value(high_stakes_only);
    root_obj["total_wall_time_seconds"] = json::Value(total_wall_time);
    root_obj["win_rate_percent"] = json::Value(win_rate);
    root_obj["total_wins"] = json::Value(static_cast<int64_t>(total_wins));
    root_obj["mean_placement_utility"] = json::Value(mean_util);
    root_obj["std_dev"] = json::Value(std_dev);
    root_obj["se_utility"] = json::Value(se_util);
    root_obj["ci_95_lower"] = json::Value(ci_lower);
    root_obj["ci_95_upper"] = json::Value(ci_upper);
    root_obj["mean_ending_round"] = json::Value(mean_round);
    root_obj["p_round_le7_percent"] = json::Value(p_round_le7);
    root_obj["p_round_le8_percent"] = json::Value(p_round_le8);
    root_obj["total_searches"] = json::Value(static_cast<int64_t>(total_searches_all));
    root_obj["total_overrides"] = json::Value(static_cast<int64_t>(total_overrides_all));
    root_obj["override_rate_percent"] = json::Value(override_pct);

    json::Array games_arr;
    for (const auto& out : outcomes) {
      json::Object g_obj;
      g_obj["game_id"] = json::Value(static_cast<int64_t>(out.game_id));
      g_obj["search_seat"] = json::Value(static_cast<int64_t>(out.search_seat));
      g_obj["winner_seat"] = json::Value(static_cast<int64_t>(out.winner_seat));
      g_obj["search_won"] = json::Value(out.search_won);
      g_obj["ending_round"] = json::Value(static_cast<int64_t>(out.ending_round));
      g_obj["search_utility"] = json::Value(out.search_utility);
      g_obj["search_vp"] = json::Value(static_cast<int64_t>(out.search_vp));
      g_obj["total_searches"] = json::Value(static_cast<int64_t>(out.total_searches));
      g_obj["total_overrides"] = json::Value(static_cast<int64_t>(out.total_overrides));
      g_obj["duration_seconds"] = json::Value(out.duration_seconds);
      games_arr.push_back(json::Value(g_obj));
    }
    root_obj["games"] = json::Value(games_arr);

    std::ofstream ofs(output_json_path);
    ofs << json::ToString(json::Value(root_obj), true);
    std::cout << "Receipt written to: " << output_json_path << "\n";
  }

  return 0;
}
