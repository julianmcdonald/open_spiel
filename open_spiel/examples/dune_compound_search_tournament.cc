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
#include <filesystem>

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
#include "dune_compound_turn_search.h"

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
ABSL_FLAG(int, threads, 14, "Number of concurrent game worker threads.");
ABSL_FLAG(int, num_evaluators, 6, "Number of parallel batched evaluator coordinators.");
ABSL_FLAG(int, target_batch_size, 128, "GPU evaluator target batch size.");
ABSL_FLAG(int, top_k_actions, 3, "Number of top candidate actions to evaluate with rollouts.");
ABSL_FLAG(int, rollouts_per_action, 64, "Number of Monte Carlo rollouts per candidate action.");
ABSL_FLAG(double, min_override_margin, 0.15,
          "Minimum mean utility advantage needed for candidate to override raw policy.");
ABSL_FLAG(std::string, tree_reuse_mode, "target_fill",
          "Tree reuse mode: 'target_fill' (top up visits) or 'additive' (accumulate visits).");
ABSL_FLAG(bool, high_stakes_only, true,
          "Only search on high-stakes strategic turns (Agent placement, combat deploy, card buy).");
ABSL_FLAG(bool, rotate_seat, true, "Whether to rotate the searching agent's seat across games.");
ABSL_FLAG(int, search_seat, 0, "Fixed search seat if rotate_seat is false.");
ABSL_FLAG(uint64_t, master_seed, 20260921, "Master random seed.");
ABSL_FLAG(std::string, output_json,
          "/home/warcr/projects/dune_drl/docs/experiment_records/compound_search_100_games_receipt.json",
          "Path to write structured tournament results JSON.");
ABSL_FLAG(std::string, trajectory_meta_jsonl, "",
          "Path to write trajectory metadata JSONL.");
ABSL_FLAG(std::string, trajectory_obs_bin, "",
          "Path to write raw float32 observation binary matching trajectory decisions.");
ABSL_FLAG(double, softmax_temperature, 0.5,
          "Temperature for search target distribution over candidate utilities.");
ABSL_FLAG(double, eta_blend, 0.80,
          "Blend parameter for search target: (1-eta)*mu + eta*q_search.");

struct RecordedDecision {
  int game_id = 0;
  int decision_ordinal = 0;
  int search_seat = 0;
  int round = 0;
  std::string phase;
  bool is_searched = false;
  std::string decision_role;  // "fresh_search", "continuation", "preservation"
  bool is_forced = false;
  int chosen_action = 0;
  std::vector<int> legal_actions;
  std::vector<float> observation;
  std::vector<std::pair<int, float>> mu;
  std::vector<std::pair<int, float>> q_search;
  std::vector<std::pair<int, float>> q_blend;
};

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
  int total_reused_rollouts = 0;
  int total_new_rollouts = 0;
  int total_new_info_resets = 0;
  int ending_round = 0;
  double duration_seconds = 0.0;
  std::vector<RecordedDecision> decisions_trace;
};

using open_spiel::dune_imperium::IsHighStakesStrategicState;

inline const char* GetGamePhaseString(GamePhase phase) {
  switch (phase) {
    case GamePhase::kLeaderOfferChance: return "leader_offer_chance";
    case GamePhase::kLeaderDraft: return "leader_draft";
    case GamePhase::kDeal: return "deal";
    case GamePhase::kRoundStart: return "round_start";
    case GamePhase::kAgentTurns: return "agent_turns";
    case GamePhase::kRevealTurns: return "reveal_turns";
    case GamePhase::kCombat: return "combat";
    case GamePhase::kMakers: return "makers";
    case GamePhase::kRecall: return "recall";
    case GamePhase::kTerminal: return "terminal";
    default: return "unknown";
  }
}

// Play a single full game using compound turn search
GameOutcome PlayCompoundTournamentGame(
    int game_id,
    int search_seat,
    const std::shared_ptr<const Game>& game,
    const std::vector<std::shared_ptr<BatchedNNEvaluator>>& evaluators,
    int thread_id,
    int top_k,
    int rollouts_per_action,
    double min_override_margin,
    TreeReuseMode reuse_mode,
    bool high_stakes_only,
    uint64_t game_seed,
    double softmax_temperature = 0.5,
    double eta_blend = 0.80,
    bool record_trajectories = false) {
  auto start_time = std::chrono::steady_clock::now();
  std::unique_ptr<State> state = game->NewInitialState();
  std::mt19937 chance_rng(game_seed);

  int searches = 0;
  int overrides = 0;
  int decisions = 0;
  int reused_rollouts = 0;
  int new_rollouts = 0;
  int new_info_resets = 0;
  uint64_t search_seed_counter = 0;
  auto* greedy_evaluator = evaluators[thread_id % evaluators.size()].get();

  TurnSearchTree turn_tree;
  Player last_player = kInvalidPlayer;
  GameOutcome outcome;

  while (!state->IsTerminal()) {
    if (state->IsChanceNode()) {
      const auto* dune_before = dynamic_cast<const dune_imperium::DuneImperiumState*>(state.get());
      int hand_before = dune_before ? dune_before->GetPlayerCardsInHand(search_seat) : 0;
      int int_before = dune_before ? static_cast<int>(dune_before->GetPlayerIntrigues(search_seat).size()) : 0;

      auto chance_outcomes = state->ChanceOutcomes();
      double u = std::generate_canonical<double, 53>(chance_rng);
      Action ca = open_spiel::SampleAction(chance_outcomes, u).first;
      state->ApplyAction(ca);

      const auto* dune_after = dynamic_cast<const dune_imperium::DuneImperiumState*>(state.get());
      if (dune_after) {
        int hand_after = dune_after->GetPlayerCardsInHand(search_seat);
        int int_after = static_cast<int>(dune_after->GetPlayerIntrigues(search_seat).size());
        if (hand_after > hand_before || int_after > int_before) {
          turn_tree.Reset();
          new_info_resets++;
        }
      }
      continue;
    }

    Player cur = state->CurrentPlayer();
    if (cur != last_player) {
      turn_tree.Reset();
      last_player = cur;
    }

    std::vector<Action> legals = state->LegalActions();
    if (legals.empty()) break;

    Action chosen_action = kInvalidAction;
    if (cur == search_seat) {
      decisions++;
      bool should_search = high_stakes_only
          ? IsHighStakesStrategicState(*state, cur)
          : (IsStrategicState(*state, cur) && legals.size() > 1);

      std::vector<float> obs_tensor;
      if (record_trajectories) {
        obs_tensor = greedy_evaluator->GetConsumedObservation(*state, cur);
      }

      ActionsAndProbs raw_prior = greedy_evaluator->Prior(*state);
      std::vector<std::pair<Action, double>> mu_legal;
      mu_legal.reserve(legals.size());
      for (Action a : legals) {
        double p = 0.0;
        for (const auto& ap : raw_prior) {
          if (ap.first == a) { p = ap.second; break; }
        }
        mu_legal.push_back({a, p});
      }

      bool is_forced = (legals.size() <= 1);
      bool is_searched = false;
      std::string decision_role = "preservation";
      std::vector<std::pair<Action, double>> q_search_target;
      std::vector<std::pair<Action, double>> q_blend_target;

      if (should_search && legals.size() > 1) {
        searches++;
        uint64_t dseed = dune_seed::DeriveSeed(game_seed, cur, search_seed_counter++);
        auto s_res = CompoundRolloutSearchDecision(
            *state, cur, &turn_tree, evaluators, top_k, rollouts_per_action,
            min_override_margin, reuse_mode, dseed);
        chosen_action = s_res.action;
        if (s_res.overridden) overrides++;
        reused_rollouts += s_res.prior_rollouts_reused;
        new_rollouts += s_res.new_rollouts_executed;

        is_searched = true;
        decision_role = "fresh_search";
        q_search_target = ComputeSoftmaxTarget(s_res.candidate_utilities, legals, softmax_temperature);
        q_blend_target = BlendTargets(mu_legal, q_search_target, eta_blend);
      } else if (turn_tree.current_node() != nullptr && !turn_tree.current_node()->children.empty()) {
        std::vector<std::pair<Action, double>> continuation_cands;
        for (Action a : legals) {
          if (turn_tree.HasPriorData(a)) {
            continuation_cands.push_back({a, turn_tree.GetMeanUtility(a)});
          }
        }
        if (!continuation_cands.empty()) {
          is_searched = true;
          decision_role = "continuation";
          q_search_target = ComputeSoftmaxTarget(continuation_cands, legals, softmax_temperature);
          q_blend_target = BlendTargets(mu_legal, q_search_target, eta_blend);

          double best_u = -1e9;
          Action best_a = PickGreedyAction(raw_prior, legals);
          for (const auto& cu : continuation_cands) {
            if (cu.second > best_u) {
              best_u = cu.second;
              best_a = cu.first;
            }
          }
          chosen_action = best_a;
        } else {
          chosen_action = PickGreedyAction(raw_prior, legals);
          q_search_target = mu_legal;
          q_blend_target = mu_legal;
        }
      } else {
        chosen_action = PickGreedyAction(raw_prior, legals);
        q_search_target = mu_legal;
        q_blend_target = mu_legal;
      }

      if (record_trajectories) {
        RecordedDecision dec;
        dec.game_id = game_id;
        dec.decision_ordinal = decisions - 1;
        dec.search_seat = cur;
        const auto* dune_s = dynamic_cast<const dune_imperium::DuneImperiumState*>(state.get());
        dec.round = dune_s ? dune_s->GetCurrentRound() : 1;
        dec.phase = dune_s ? GetGamePhaseString(dune_s->phase()) : "UNKNOWN";
        dec.is_searched = is_searched;
        dec.decision_role = decision_role;
        dec.is_forced = is_forced;
        dec.chosen_action = chosen_action;
        for (Action a : legals) dec.legal_actions.push_back(a);
        dec.observation = std::move(obs_tensor);
        for (const auto& p : mu_legal) dec.mu.push_back({p.first, static_cast<float>(p.second)});
        for (const auto& p : q_search_target) dec.q_search.push_back({p.first, static_cast<float>(p.second)});
        for (const auto& p : q_blend_target) dec.q_blend.push_back({p.first, static_cast<float>(p.second)});
        outcome.decisions_trace.push_back(std::move(dec));
      }

      // Advance search tree along chosen real action
      turn_tree.Advance(chosen_action);
      if (chosen_action == dune_imperium::kActionEndTurn) {
        turn_tree.Reset();
      }
    } else {
      ActionsAndProbs prior = greedy_evaluator->Prior(*state);
      chosen_action = PickGreedyAction(prior, legals);
    }

    const auto* dune_before = dynamic_cast<const dune_imperium::DuneImperiumState*>(state.get());
    int hand_before = (cur == search_seat && dune_before) ? dune_before->GetPlayerCardsInHand(search_seat) : 0;
    int int_before = (cur == search_seat && dune_before) ? static_cast<int>(dune_before->GetPlayerIntrigues(search_seat).size()) : 0;

    state->ApplyAction(chosen_action);

    if (cur == search_seat && chosen_action != dune_imperium::kActionEndTurn) {
      const auto* dune_after = dynamic_cast<const dune_imperium::DuneImperiumState*>(state.get());
      if (dune_after) {
        int hand_after = dune_after->GetPlayerCardsInHand(search_seat);
        int int_after = static_cast<int>(dune_after->GetPlayerIntrigues(search_seat).size());
        if (hand_after > hand_before || int_after > int_before) {
          turn_tree.Reset();
          new_info_resets++;
        }
      }
    }
  }

  auto end_time = std::chrono::steady_clock::now();
  double elapsed = std::chrono::duration<double>(end_time - start_time).count();

  outcome.game_id = game_id;
  outcome.search_seat = search_seat;
  outcome.final_returns = state->Returns();
  outcome.total_searches = searches;
  outcome.total_overrides = overrides;
  outcome.total_decisions = decisions;
  outcome.total_reused_rollouts = reused_rollouts;
  outcome.total_new_rollouts = new_rollouts;
  outcome.total_new_info_resets = new_info_resets;
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
  const std::string reuse_mode_str = absl::GetFlag(FLAGS_tree_reuse_mode);
  const bool high_stakes_only = absl::GetFlag(FLAGS_high_stakes_only);
  const bool rotate_seat = absl::GetFlag(FLAGS_rotate_seat);
  const int default_search_seat = absl::GetFlag(FLAGS_search_seat);
  const uint64_t master_seed = absl::GetFlag(FLAGS_master_seed);
  const std::string output_json_path = absl::GetFlag(FLAGS_output_json);
  const std::string trajectory_meta_path = absl::GetFlag(FLAGS_trajectory_meta_jsonl);
  const std::string trajectory_obs_path = absl::GetFlag(FLAGS_trajectory_obs_bin);
  const double softmax_temp = absl::GetFlag(FLAGS_softmax_temperature);
  const double eta_blend = absl::GetFlag(FLAGS_eta_blend);
  const bool record_trajectories = !trajectory_meta_path.empty() && !trajectory_obs_path.empty();

  TreeReuseMode reuse_mode = (reuse_mode_str == "additive")
      ? TreeReuseMode::kAdditive
      : TreeReuseMode::kTargetFill;

  std::cout << "================================================================================\n";
  std::cout << "DUNE: IMPERIUM -- TOURNAMENT (COMPOUND TURN SEARCH & TREE REUSE)\n";
  std::cout << "================================================================================\n";
  std::cout << "Model Checkpoint: " << model_path << "\n";
  std::cout << "Games: " << total_games << ", Threads: " << num_threads
            << ", Evaluators: " << num_evaluators << ", Target Batch Size: " << target_batch_size << "\n";
  std::cout << "Search Topology: 1 Compound Searching Agent (Top-" << top_k << ", " << rollouts_per_action
            << " rollouts/act) vs 3 Raw Greedy Opponents\n";
  std::cout << "Tree Reuse Mode: " << (reuse_mode == TreeReuseMode::kTargetFill ? "TARGET_FILL" : "ADDITIVE") << "\n";
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
  std::atomic<int> total_reused_sum(0);
  std::atomic<int> total_new_sum(0);
  std::atomic<int> total_resets_sum(0);
  std::mutex cout_mutex;

  auto tournament_start = std::chrono::steady_clock::now();

  auto worker = [&](int thread_id) {
    while (true) {
      int g = next_game_idx.fetch_add(1);
      if (g >= total_games) break;

      int s_seat = rotate_seat ? (g % 4) : default_search_seat;
      uint64_t gseed = dune_seed::DeriveSeed(master_seed, dune_seed::kStreamSearchSampling, g);

      GameOutcome res = PlayCompoundTournamentGame(
          g, s_seat, game, evaluators, thread_id, top_k, rollouts_per_action,
          min_override_margin, reuse_mode, high_stakes_only, gseed,
          softmax_temp, eta_blend, record_trajectories);

      outcomes[g] = res;

      int done = completed_games.fetch_add(1) + 1;
      if (res.search_won) search_wins.fetch_add(1);
      ending_round_sum.fetch_add(res.ending_round);
      total_reused_sum.fetch_add(res.total_reused_rollouts);
      total_new_sum.fetch_add(res.total_new_rollouts);
      total_resets_sum.fetch_add(res.total_new_info_resets);
      
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
                << "(Searches: " << res.total_searches << ", Overrides: " << res.total_overrides
                << ", Reused: " << res.total_reused_rollouts << ", New: " << res.total_new_rollouts
                << ", Resets: " << res.total_new_info_resets << ", "
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
  std::vector<double> utilities;
  std::vector<int> search_vps;
  std::vector<int> seat_games(4, 0);
  std::vector<int> seat_wins(4, 0);
  std::vector<double> seat_util_sum(4, 0.0);

  for (const auto& out : outcomes) {
    if (out.search_won) total_wins++;
    total_searches_all += out.total_searches;
    total_overrides_all += out.total_overrides;
    total_ending_round += out.ending_round;
    utilities.push_back(out.search_utility);
    search_vps.push_back(out.search_vp);

    seat_games[out.search_seat]++;
    if (out.search_won) seat_wins[out.search_seat]++;
    seat_util_sum[out.search_seat] += out.search_utility;
  }

  double win_rate = 100.0 * total_wins / total_games;
  double mean_util = std::accumulate(utilities.begin(), utilities.end(), 0.0) / total_games;
  double mean_vp = std::accumulate(search_vps.begin(), search_vps.end(), 0.0) / total_games;
  double mean_round = static_cast<double>(total_ending_round) / total_games;

  double sq_sum = 0.0;
  for (double u : utilities) sq_sum += (u - mean_util) * (u - mean_util);
  double std_dev = std::sqrt(sq_sum / (total_games - 1));
  double std_err = std_dev / std::sqrt(total_games);
  double ci95 = 1.96 * std_err;

  std::cout << "\n================================================================================\n";
  std::cout << "COMPOUND TURN SEARCH TOURNAMENT FINAL SUMMARY (" << total_games << " GAMES)\n";
  std::cout << "================================================================================\n";
  std::cout << "Searching Agent Win Rate:  " << std::fixed << std::setprecision(1) << win_rate << "% ("
            << total_wins << "/" << total_games << ")  [Baseline random seat win rate: 25.0%]\n";
  std::cout << "Searching Agent Utility:   " << std::showpos << std::setprecision(3) << mean_util
            << " +/- " << std::noshowpos << std::setprecision(3) << ci95 << " (95% CI)\n";
  std::cout << "Average Ending Round:      " << std::fixed << std::setprecision(2) << mean_round << "\n";
  std::cout << "Searching Agent Mean VP:   " << std::fixed << std::setprecision(2) << mean_vp << " VP\n";
  std::cout << "Rollouts Reused in Tree:   " << total_reused_sum.load()
            << " | New Rollouts: " << total_new_sum.load()
            << " | Info Resets: " << total_resets_sum.load() << "\n";
  std::cout << "Total Wall Time:           " << std::setprecision(1) << total_wall_time << "s ("
            << std::setprecision(2) << (total_wall_time / 60.0) << " min)\n";
  std::cout << "Throughput:                " << std::setprecision(2) << (total_games / (total_wall_time / 60.0))
            << " games/min (" << std::setprecision(1) << (total_wall_time / total_games) << "s/game)\n";
  std::cout << "Total Searches Triggered:  " << total_searches_all << " (avg "
            << std::setprecision(1) << (static_cast<double>(total_searches_all) / total_games) << " / game)\n";
  std::cout << "Total Policy Overrides:    " << total_overrides_all << " ("
            << std::setprecision(1) << (100.0 * total_overrides_all / total_searches_all) << "% override rate)\n";
  std::cout << "================================================================================\n";
  std::cout << "Per-Seat Performance:\n";
  for (int s = 0; s < 4; ++s) {
    double swr = seat_games[s] > 0 ? (100.0 * seat_wins[s] / seat_games[s]) : 0.0;
    double smu = seat_games[s] > 0 ? (seat_util_sum[s] / seat_games[s]) : 0.0;
    std::cout << "  Seat " << s << ": " << seat_wins[s] << "/" << seat_games[s]
              << " (" << std::setprecision(1) << swr << "% WR, Mean Util: "
              << std::setprecision(3) << std::showpos << smu << std::noshowpos << ")\n";
  }
  std::cout << "================================================================================\n";

  // Write receipt JSON
  json::Object root;
  root["tournament_type"] = json::Value("compound_turn_search");
  root["model_checkpoint"] = json::Value(model_path);
  root["total_games"] = json::Value(static_cast<int64_t>(total_games));
  root["win_rate"] = json::Value(win_rate);
  root["total_wins"] = json::Value(static_cast<int64_t>(total_wins));
  root["mean_utility"] = json::Value(mean_util);
  root["utility_ci95"] = json::Value(ci95);
  root["mean_ending_round"] = json::Value(mean_round);
  root["mean_vp"] = json::Value(mean_vp);
  root["total_reused_rollouts"] = json::Value(static_cast<int64_t>(total_reused_sum.load()));
  root["total_new_rollouts"] = json::Value(static_cast<int64_t>(total_new_sum.load()));
  root["total_new_info_resets"] = json::Value(static_cast<int64_t>(total_resets_sum.load()));
  root["total_searches"] = json::Value(static_cast<int64_t>(total_searches_all));
  root["total_overrides"] = json::Value(static_cast<int64_t>(total_overrides_all));
  root["override_rate_pct"] = json::Value(100.0 * total_overrides_all / total_searches_all);
  root["total_wall_time_seconds"] = json::Value(total_wall_time);

  json::Array seat_stats;
  for (int s = 0; s < 4; ++s) {
    json::Object s_obj;
    s_obj["seat"] = json::Value(static_cast<int64_t>(s));
    s_obj["games"] = json::Value(static_cast<int64_t>(seat_games[s]));
    s_obj["wins"] = json::Value(static_cast<int64_t>(seat_wins[s]));
    s_obj["win_rate"] = json::Value(seat_games[s] > 0 ? (100.0 * seat_wins[s] / seat_games[s]) : 0.0);
    s_obj["mean_utility"] = json::Value(seat_games[s] > 0 ? (seat_util_sum[s] / seat_games[s]) : 0.0);
    seat_stats.push_back(json::Value(s_obj));
  }
  root["seat_statistics"] = json::Value(seat_stats);

  json::Array games_arr;
  for (const auto& out : outcomes) {
    json::Object g_obj;
    g_obj["game_id"] = json::Value(static_cast<int64_t>(out.game_id));
    g_obj["search_seat"] = json::Value(static_cast<int64_t>(out.search_seat));
    g_obj["winner_seat"] = json::Value(static_cast<int64_t>(out.winner_seat));
    g_obj["search_won"] = json::Value(out.search_won);
    g_obj["search_utility"] = json::Value(out.search_utility);
    g_obj["search_vp"] = json::Value(static_cast<int64_t>(out.search_vp));
    g_obj["ending_round"] = json::Value(static_cast<int64_t>(out.ending_round));
    g_obj["total_searches"] = json::Value(static_cast<int64_t>(out.total_searches));
    g_obj["total_overrides"] = json::Value(static_cast<int64_t>(out.total_overrides));
    g_obj["reused_rollouts"] = json::Value(static_cast<int64_t>(out.total_reused_rollouts));
    g_obj["new_rollouts"] = json::Value(static_cast<int64_t>(out.total_new_rollouts));
    g_obj["new_info_resets"] = json::Value(static_cast<int64_t>(out.total_new_info_resets));
    g_obj["duration_seconds"] = json::Value(out.duration_seconds);
    games_arr.push_back(json::Value(g_obj));
  }
  root["games"] = json::Value(games_arr);

  if (record_trajectories) {
    std::filesystem::create_directories(std::filesystem::path(trajectory_meta_path).parent_path());
    std::filesystem::create_directories(std::filesystem::path(trajectory_obs_path).parent_path());
    std::ofstream obs_file(trajectory_obs_path, std::ios::binary);
    std::ofstream meta_file(trajectory_meta_path);
    int total_decisions_written = 0;

    for (const auto& out : outcomes) {
      for (const auto& dec : out.decisions_trace) {
        if (!dec.observation.empty()) {
          obs_file.write(reinterpret_cast<const char*>(dec.observation.data()),
                         dec.observation.size() * sizeof(float));
        }

        json::Object dec_obj;
        dec_obj["game_id"] = json::Value(static_cast<int64_t>(dec.game_id));
        dec_obj["decision_ordinal"] = json::Value(static_cast<int64_t>(dec.decision_ordinal));
        dec_obj["search_seat"] = json::Value(static_cast<int64_t>(dec.search_seat));
        dec_obj["round"] = json::Value(static_cast<int64_t>(dec.round));
        dec_obj["phase"] = json::Value(dec.phase);
        dec_obj["is_searched"] = json::Value(dec.is_searched);
        dec_obj["decision_role"] = json::Value(dec.decision_role);
        dec_obj["is_forced"] = json::Value(dec.is_forced);
        dec_obj["chosen_action"] = json::Value(static_cast<int64_t>(dec.chosen_action));

        json::Array legals_arr;
        for (int a : dec.legal_actions) legals_arr.push_back(json::Value(static_cast<int64_t>(a)));
        dec_obj["legal_actions"] = json::Value(legals_arr);

        json::Object mu_obj;
        for (const auto& p : dec.mu) mu_obj[std::to_string(p.first)] = json::Value(static_cast<double>(p.second));
        dec_obj["mu"] = json::Value(mu_obj);

        json::Object qs_obj;
        for (const auto& p : dec.q_search) qs_obj[std::to_string(p.first)] = json::Value(static_cast<double>(p.second));
        dec_obj["q_search"] = json::Value(qs_obj);

        json::Object qb_obj;
        for (const auto& p : dec.q_blend) qb_obj[std::to_string(p.first)] = json::Value(static_cast<double>(p.second));
        dec_obj["q_blend"] = json::Value(qb_obj);

        dec_obj["search_won"] = json::Value(out.search_won);
        dec_obj["search_utility"] = json::Value(out.search_utility);
        dec_obj["search_vp"] = json::Value(static_cast<int64_t>(out.search_vp));

        json::Array rets_arr;
        for (double r : out.final_returns) rets_arr.push_back(json::Value(r));
        dec_obj["final_returns"] = json::Value(rets_arr);

        meta_file << json::ToString(dec_obj) << "\n";
        total_decisions_written++;
      }
    }
    obs_file.close();
    meta_file.close();
    std::cout << "[TRAJECTORY] Successfully wrote " << total_decisions_written
              << " decisions to " << trajectory_meta_path
              << " and observations to " << trajectory_obs_path << "\n";
  }

  std::ofstream out_file(output_json_path);
  if (out_file.is_open()) {
    out_file << json::ToString(root);
    out_file.close();
    std::cout << "[RECEIPT] Saved results to " << output_json_path << "\n";
  } else {
    std::cerr << "[WARNING] Failed to open " << output_json_path << " for writing.\n";
  }

  return 0;
}
