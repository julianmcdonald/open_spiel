#include "dune_search_benchmark_internal.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>

#include "dune_sha256.h"
#include "open_spiel/abseil-cpp/absl/flags/flag.h"
#include "open_spiel/abseil-cpp/absl/strings/str_format.h"
#include "open_spiel/spiel_utils.h"
#include "open_spiel/utils/json.h"

namespace open_spiel {


void ComputeWilsonWinrateCI(int wins, int total, double* lcb, double* ucb) {
  if (total == 0) {
    *lcb = 0.0;
    *ucb = 0.0;
    return;
  }
  double p = static_cast<double>(wins) / total;
  double z = 1.96;
  double denom = 1.0 + z * z / total;
  double center = p + z * z / (2.0 * total);
  double spread = z * std::sqrt((p * (1.0 - p) / total) + (z * z / (4.0 * total * total)));
  *lcb = (center - spread) / denom * 100.0;
  *ucb = (center + spread) / denom * 100.0;
}

void ComputeVpMarginCI(const std::vector<double>& margins, double* mean, double* lcb, double* ucb) {
  int n = margins.size();
  if (n <= 1) {
    *mean = 0.0;
    *lcb = 0.0;
    *ucb = 0.0;
    return;
  }
  double sum = 0.0;
  for (double x : margins) {
    sum += x;
  }
  *mean = sum / n;
  double sum_sq_diff = 0.0;
  for (double x : margins) {
    sum_sq_diff += (x - *mean) * (x - *mean);
  }
  double variance = sum_sq_diff / (n - 1);
  double std_dev = std::sqrt(variance);
  double margin_of_error = 1.96 * std_dev / std::sqrt(n);
  *lcb = *mean - margin_of_error;
  *ucb = *mean + margin_of_error;
}

void UpdateStepSearchStats(
    DuneDecisionRole role,
    bool is_strategic,
    double step_duration,
    const DuneSearchResult& last_res,
    const DuneSearchSession* cur_session,
    TurnAuditState& audit_state,
    GameStats& thread_stats) {
  // Update turn statistics
  if (role == open_spiel::DuneDecisionRole::kAgentPrimary) {
    audit_state.in_search_turn = true;
    audit_state.turn_decisions = 1;
    audit_state.turn_primary_sims = last_res.simulations_completed;
    audit_state.turn_continuation_sims = 0;
    audit_state.turn_wall_time_ms = step_duration * 1000.0;
    audit_state.turn_had_timeout = last_res.timeout_status;
    audit_state.turn_had_fallback = last_res.used_fallback;
    audit_state.turn_fallback_reason = last_res.fallback_reason;
  } else if (audit_state.in_search_turn && role == open_spiel::DuneDecisionRole::kAgentContinuation) {
    audit_state.turn_decisions++;
    audit_state.turn_continuation_sims += last_res.simulations_completed;
    audit_state.turn_wall_time_ms += step_duration * 1000.0;
    if (last_res.timeout_status) audit_state.turn_had_timeout = true;
    if (last_res.used_fallback) {
      audit_state.turn_had_fallback = true;
      audit_state.turn_fallback_reason = last_res.fallback_reason;
    }
  }
  thread_stats.search_step_time_sum += step_duration;
  thread_stats.search_steps_count++;
  if (is_strategic) {
    thread_stats.search_mcts_step_time_sum += step_duration;
    thread_stats.search_mcts_steps_count++;
    thread_stats.search_mcts_step_times.push_back(step_duration);

    thread_stats.total_simulations_completed += last_res.simulations_completed;
    thread_stats.total_inferences += last_res.inference_count;
    thread_stats.covered_prior_mass_sum += last_res.diagnostics.covered_prior_mass;
    if (last_res.timeout_status) {
      thread_stats.total_timeouts++;
    }
    if (last_res.used_fallback) {
      thread_stats.total_fallbacks++;
      thread_stats.fallback_reason_counts[last_res.fallback_reason]++;
    }
    int reserve = absl::GetFlag(FLAGS_fixed_continuation_reserve);
    int target_sims = (role == open_spiel::DuneDecisionRole::kAgentPrimary)
        ? (absl::GetFlag(FLAGS_max_simulations) - reserve)
        : absl::GetFlag(FLAGS_max_simulations);
    int completed_sims = (role == open_spiel::DuneDecisionRole::kAgentPrimary || !cur_session)
        ? last_res.simulations_completed
        : cur_session->session_new_simulations_completed();

    bool is_incomplete = false;
    if (absl::GetFlag(FLAGS_live_deadline)) {
      int step_sims = last_res.simulations_completed;
      if (role == open_spiel::DuneDecisionRole::kAgentContinuation) {
        if (step_sims < absl::GetFlag(FLAGS_fixed_continuation_reserve)) {
          is_incomplete = true;
        }
      } else if (role == open_spiel::DuneDecisionRole::kPurchase ||
                 role == open_spiel::DuneDecisionRole::kCombatIntrigue ||
                 role == open_spiel::DuneDecisionRole::kOtherOptional) {
        if (step_sims < absl::GetFlag(FLAGS_purchase_combat_budget)) {
          is_incomplete = true;
        }
      }
    } else {
      if (completed_sims < target_sims) {
        is_incomplete = true;
      }
    }

    if (is_incomplete) {
      thread_stats.total_incomplete_searches++;
      if (!absl::GetFlag(FLAGS_live_deadline)) {
        if (last_res.fallback_reason == "timeout" ||
            last_res.fallback_reason == "max_nodes" ||
            last_res.timeout_status) {
          std::cerr << "\nCRITICAL FAILURE: Strategic search stopped before completing "
                    << target_sims
                    << " simulations due to " << last_res.fallback_reason
                    << " (completed " << completed_sims << " simulations).\n";
          std::exit(1);
        }
      }
    }
  }
}

void RecordTerminalGameStats(
    const State& state,
    const dune_imperium::DuneImperiumState* dune_state,
    int search_seat,
    GameStats& thread_stats,
    int* winner_out,
    double* search_vp_out,
    double* margin_out,
    int* corrected_round_out) {
  std::vector<double> returns = state.Returns();
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

  *winner_out = winner;

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
    int corrected_round = dune_state->GetCurrentRound();
    thread_stats.rounds_played.push_back(corrected_round);

    double search_vp = GetTrueFinalVp(dune_state, search_seat);
    double opp_vp_sum = 0.0;
    for (int p = 0; p < 4; ++p) {
      if (p != search_seat) {
        opp_vp_sum += GetTrueFinalVp(dune_state, p);
      }
    }
    double margin = search_vp - (opp_vp_sum / 3.0);
    thread_stats.vp_margin_sum += margin;
    thread_stats.vp_margin_count++;
    thread_stats.vp_margins.push_back(margin);

    *search_vp_out = search_vp;
    *margin_out = margin;
    *corrected_round_out = corrected_round;
  }
}

void MergeGameStats(
    GameStats& dst,
    const GameStats& src,
    const std::vector<int>* game_role_counts) {
  dst.search_wins += src.search_wins;
  for (int p = 0; p < 3; ++p) dst.opponent_wins[p] += src.opponent_wins[p];
  for (int s = 0; s < 4; ++s) {
    dst.search_wins_by_seat[s] += src.search_wins_by_seat[s];
    dst.games_by_seat[s] += src.games_by_seat[s];
    dst.search_return_by_seat[s] += src.search_return_by_seat[s];
  }
  for (int r = 0; r < 4; ++r) dst.search_placements[r] += src.search_placements[r];
  dst.search_return_sum += src.search_return_sum;
  dst.search_swordmasters += src.search_swordmasters;
  dst.opponent_swordmasters += src.opponent_swordmasters;
  dst.rounds_played.insert(dst.rounds_played.end(), src.rounds_played.begin(), src.rounds_played.end());
  dst.search_step_time_sum += src.search_step_time_sum;
  dst.search_steps_count += src.search_steps_count;
  dst.abs_terminal_return_sum += src.abs_terminal_return_sum;
  dst.terminal_returns_count += src.terminal_returns_count;
  dst.vp_margin_sum += src.vp_margin_sum;
  dst.vp_margin_count += src.vp_margin_count;
  dst.vp_margins.insert(dst.vp_margins.end(), src.vp_margins.begin(), src.vp_margins.end());
  dst.search_mcts_steps_count += src.search_mcts_steps_count;
  dst.search_mcts_step_time_sum += src.search_mcts_step_time_sum;
  dst.total_simulations_completed += src.total_simulations_completed;
  dst.total_timeouts += src.total_timeouts;
  dst.total_fallbacks += src.total_fallbacks;
  dst.total_inferences += src.total_inferences;
  dst.total_incomplete_searches += src.total_incomplete_searches;
  dst.search_mcts_step_times.insert(dst.search_mcts_step_times.end(),
                                    src.search_mcts_step_times.begin(),
                                    src.search_mcts_step_times.end());
  dst.covered_prior_mass_sum += src.covered_prior_mass_sum;
  for (const auto& pair : src.fallback_reason_counts) {
    dst.fallback_reason_counts[pair.first] += pair.second;
  }
  if (game_role_counts != nullptr) {
    for (int i = 0; i < 7; ++i) {
      dst.role_counts[i] += (*game_role_counts)[i];
    }
  }
}


void PrintEpisode67Breakdown(const dune_imperium::DuneImperiumState* dune_state) {
  std::cout << "\n=== EPISODE 67 VP BREAKDOWN FOR PLAYER 3 ===" << std::endl;
  std::cout << "Base VP: " << dune_state->GetPlayerVp(3) << std::endl;
  std::cout << "Conflict VPs: " << dune_state->ConflictVpDelta(3) << std::endl;
  std::cout << "Tleilaxu Track Level: " << dune_state->GetTleilaxuTrackForTesting(3) << std::endl;
  std::cout << "The Spice Must Flow Cards Owned: " << dune_state->CountImperiumCardsOwnedForTesting(3, 10) << std::endl;
  std::cout << "Emperor Influence: " << dune_state->GetPlayerInfluenceForTesting(3, dune_imperium::Faction::kEmperor) << std::endl;
  std::cout << "Spacing Guild Influence: " << dune_state->GetPlayerInfluenceForTesting(3, dune_imperium::Faction::kSpacingGuild) << std::endl;
  std::cout << "Bene Gesserit Influence: " << dune_state->GetPlayerInfluenceForTesting(3, dune_imperium::Faction::kBeneGesserit) << std::endl;
  std::cout << "Fremen Influence: " << dune_state->GetPlayerInfluenceForTesting(3, dune_imperium::Faction::kFremen) << std::endl;

  std::cout << "Alliance Owners: Emperor=" << dune_state->GetAllianceOwnerForTesting(0)
            << ", SG=" << dune_state->GetAllianceOwnerForTesting(1)
            << ", BG=" << dune_state->GetAllianceOwnerForTesting(2)
            << ", Fremen=" << dune_state->GetAllianceOwnerForTesting(3) << std::endl;

  std::cout << "Intrigue Cards in Hand: ";
  for (int intrigue_id : dune_state->GetIntrigueHandForTesting(3)) {
    std::cout << intrigue_id << " ";
  }
  std::cout << std::endl;

  std::cout << "All Owned Cards (ID xCount): ";
  for (int card_id = 0; card_id < 200; ++card_id) {
    int count = dune_state->CountImperiumCardsOwnedForTesting(3, card_id);
    if (count > 0) {
      std::cout << card_id << "x" << count << " ";
    }
  }
  std::cout << "\n============================================\n" << std::endl;
}

void PrintBatcherCoordinatorStats(
    const std::shared_ptr<open_spiel::BatchedEvaluator>& search_coord,
    const std::shared_ptr<open_spiel::BatchedEvaluator>& opp_coord) {
  if (search_coord != nullptr) {
    auto s = search_coord->GetStats();
    std::cout << "\nCandidate Shared Batcher Telemetry:\n"
              << "  Total Requests:   " << s.requests << "\n"
              << "  Physical Batches: " << s.batches << "\n"
              << "  Avg Batch Size:   " << std::fixed << std::setprecision(2) << s.avg_batch_size << "\n"
              << "  Max Batch Size:   " << s.max_batch_size << "\n";
  }
  if (opp_coord != nullptr) {
    auto s = opp_coord->GetStats();
    std::cout << "\nOpponent Shared Batcher Telemetry:\n"
              << "  Total Requests:   " << s.requests << "\n"
              << "  Physical Batches: " << s.batches << "\n"
              << "  Avg Batch Size:   " << std::fixed << std::setprecision(2) << s.avg_batch_size << "\n"
              << "  Max Batch Size:   " << s.max_batch_size << "\n";
  }
}

void WriteMarketDiagnosticsReports(
    const std::vector<dune_market_diag::CompactPerGameMarketDiagnostics>& market_results,
    int total_games) {
  auto report = open_spiel::dune_market_diag::MarketDiagnosticsReport::AggregateReport(
      market_results, total_games);
  std::string diag_json = absl::GetFlag(FLAGS_market_diag_json_path);
  if (!diag_json.empty()) {
    if (!report.WriteJson(diag_json)) {
      open_spiel::SpielFatalError("Failed writing market diagnostics JSON to " + diag_json);
    }
    std::cout << "Market diagnostics JSON written to " << diag_json << "\n";
  }
  std::string diag_csv = absl::GetFlag(FLAGS_market_diag_csv_path);
  if (!diag_csv.empty()) {
    if (!report.WriteCsv(diag_csv)) {
      open_spiel::SpielFatalError("Failed writing market diagnostics CSV to " + diag_csv);
    }
    std::cout << "Market diagnostics CSV written to " << diag_csv << "\n";
  }
  std::string diag_games = absl::GetFlag(FLAGS_market_diag_games_path);
  if (!diag_games.empty()) {
    if (!open_spiel::dune_market_diag::WritePerGameJsonl(diag_games, market_results)) {
      open_spiel::SpielFatalError("Failed writing market diagnostics games JSONL to " + diag_games);
    }
    std::cout << "Market diagnostics games JSONL written to " << diag_games << "\n";
  }
}

void PrintAndWriteFinalBenchmarkReport(
    const GameStats& global_stats,
    int total_games,
    int num_threads,
    const std::string& model_ckpt,
    const std::string& opp_ckpt,
    bool has_opponent_model,
    const std::string& full_game_eval_mode,
    bool batched_mode,
    const std::shared_ptr<open_spiel::BatchedEvaluator>& search_coord,
    const std::shared_ptr<open_spiel::BatchedEvaluator>& opp_coord,
    std::chrono::steady_clock::time_point run_start_time) {
  // Print summary results
  std::cout << "\n======================================================================\n";
  std::cout << "Evaluation Completed!\n";
  std::cout << "Total Games:         " << total_games << "\n";
  std::cout << "Search Checkpoint:   " << model_ckpt << "\n";
  std::cout << "Opponent Checkpoint: " << (!has_opponent_model ? "Uniform Random Agents" : (opp_ckpt == model_ckpt ? "Same as search (Self-play evaluation)" : opp_ckpt)) << "\n";
  std::cout << "======================================================================\n\n";

  std::cout << "Overall Winrates:\n";
  double lcb_winrate = 0.0, ucb_winrate = 0.0;
  ComputeWilsonWinrateCI(global_stats.search_wins, total_games, &lcb_winrate, &ucb_winrate);
  std::cout << absl::StrFormat("  Search Agent:  %.2f%% (%d/%d wins), 95%% Wilson CI: [%.2f%%, %.2f%%]\n",
                               (global_stats.search_wins * 100.0 / total_games),
                               global_stats.search_wins, total_games, lcb_winrate, ucb_winrate);
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
  double mean_vp_margin = 0.0, lcb_vp_margin = 0.0, ucb_vp_margin = 0.0;
  ComputeVpMarginCI(global_stats.vp_margins, &mean_vp_margin, &lcb_vp_margin, &ucb_vp_margin);
  std::cout << absl::StrFormat("  Mean VP Margin: %+.4f (95%% CI: [%.4f, %.4f])\n", mean_vp_margin, lcb_vp_margin, ucb_vp_margin);
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
  double avg_mcts_step_time = global_stats.search_mcts_steps_count > 0
                                  ? (global_stats.search_mcts_step_time_sum / global_stats.search_mcts_steps_count)
                                  : 0.0;
  std::cout << absl::StrFormat("Average MCTS Step Time: %.4fs (across %d strategic steps)\n",
                               avg_mcts_step_time, global_stats.search_mcts_steps_count);

  double mean_abs_leaf_value = open_spiel::DuneNNEvaluator::global_num_leaf_evaluations.load() > 0
      ? (open_spiel::DuneNNEvaluator::global_abs_leaf_value_sum.load() / open_spiel::DuneNNEvaluator::global_num_leaf_evaluations.load())
      : 0.0;
  double mean_abs_terminal_return = global_stats.terminal_returns_count > 0
      ? (global_stats.abs_terminal_return_sum / global_stats.terminal_returns_count)
      : 0.0;
  double utility_divisor_used = absl::GetFlag(FLAGS_utility_divisor);
  double raw_mean_abs_leaf_value = mean_abs_leaf_value * utility_divisor_used;
  std::cout << absl::StrFormat("Mean |leaf value|: %.4f (normalized/NN) / %.4f (raw equivalent) vs Mean |terminal return|: %.4f (raw) (utility_divisor = %.1f)\n",
                               mean_abs_leaf_value, raw_mean_abs_leaf_value, mean_abs_terminal_return, utility_divisor_used);

  double p95_latency = 0.0;
  if (!global_stats.search_mcts_step_times.empty()) {
    std::vector<double> sorted_times = global_stats.search_mcts_step_times;
    std::sort(sorted_times.begin(), sorted_times.end());
    size_t idx = static_cast<size_t>(std::round(0.95 * (sorted_times.size() - 1)));
    p95_latency = sorted_times[idx];
  }
  double total_wall_time = std::chrono::duration<double>(std::chrono::steady_clock::now() - run_start_time).count();

  std::cout << "\nIS-MCTS MCTS Search Diagnostics:\n";
  std::cout << absl::StrFormat("  Total Strategic MCTS Steps: %d\n", global_stats.search_mcts_steps_count);
  if (global_stats.search_mcts_steps_count > 0) {
    double avg_sims = static_cast<double>(global_stats.total_simulations_completed) / global_stats.search_mcts_steps_count;
    double avg_inf = static_cast<double>(global_stats.total_inferences) / global_stats.search_mcts_steps_count;
    double timeout_rate = (global_stats.total_timeouts * 100.0) / global_stats.search_mcts_steps_count;
    double fallback_rate = (global_stats.total_fallbacks * 100.0) / global_stats.search_mcts_steps_count;
    std::cout << absl::StrFormat("  Average simulations completed:  %.2f\n", avg_sims);
    std::cout << absl::StrFormat("  Average NN inferences per step: %.2f\n", avg_inf);
    std::cout << absl::StrFormat("  Timeout rate:                   %.2f%% (%d/%d steps)\n", timeout_rate, global_stats.total_timeouts, global_stats.search_mcts_steps_count);
    std::cout << absl::StrFormat("  Fallback rate:                  %.2f%% (%d/%d steps)\n", fallback_rate, global_stats.total_fallbacks, global_stats.search_mcts_steps_count);
    std::cout << absl::StrFormat("  Mean MCTS Latency:              %.4fs\n", avg_mcts_step_time);
    std::cout << absl::StrFormat("  P95 MCTS Latency:               %.4fs\n", p95_latency);
    std::cout << absl::StrFormat("  Wall Time:                      %.2fs (%.2f games/hour)\n", total_wall_time, (total_games * 3600.0 / total_wall_time));
    if (!global_stats.fallback_reason_counts.empty()) {
      std::cout << "  Fallback reasons break down:\n";
      for (const auto& pair : global_stats.fallback_reason_counts) {
        std::cout << absl::StrFormat("    - %s: %d\n", pair.first, pair.second);
      }
    }
  } else {
    std::cout << "  No strategic steps were evaluated by MCTS.\n";
  }

  std::string aggregate_json_path = absl::GetFlag(FLAGS_aggregate_json_path);
  if (!aggregate_json_path.empty()) {
    std::string checkpoint_hash = "unknown";
    try {
      checkpoint_hash = open_spiel::ComputeFileSHA256(model_ckpt);
    } catch (const std::exception& e) {
      std::cerr << "Warning: Could not compute SHA-256 for checkpoint: " << e.what() << "\n";
    }

    open_spiel::json::Object agg_obj;
    agg_obj["checkpoint_hash"] = checkpoint_hash;
    agg_obj["opponent_checkpoint"] = opp_ckpt;
    agg_obj["start_episode_id"] = static_cast<int64_t>(absl::GetFlag(FLAGS_start_episode_id));
    agg_obj["games_played"] = static_cast<int64_t>(total_games);
    agg_obj["threads"] = static_cast<int64_t>(num_threads);
    agg_obj["use_session"] = absl::GetFlag(FLAGS_use_session);
    agg_obj["fresh_search_roles"] = absl::GetFlag(FLAGS_fresh_search_roles);
    agg_obj["force_swordmaster_rounds"] = static_cast<int64_t>(absl::GetFlag(FLAGS_force_swordmaster_rounds));
    agg_obj["grant_swordmaster_round"] = static_cast<int64_t>(absl::GetFlag(FLAGS_grant_swordmaster_round));
    agg_obj["max_simulations"] = static_cast<int64_t>(absl::GetFlag(FLAGS_max_simulations));
    agg_obj["puct_c"] = absl::GetFlag(FLAGS_puct_c);
    agg_obj["root_prior_temperature"] = absl::GetFlag(FLAGS_root_prior_temperature);
    agg_obj["use_opponent_model"] = absl::GetFlag(FLAGS_use_opponent_model);
    // WO-1 Phase 2: run-level controller topology + evaluator caps, so an
    // aggregate can never be read without knowing which controller produced it.
    agg_obj["controller_mode"] = absl::GetFlag(FLAGS_controller_mode);
    agg_obj["mixed_opponent_controller"] =
        absl::GetFlag(FLAGS_mixed_opponent_controller);
    agg_obj["candidate_logit_cap"] = absl::GetFlag(FLAGS_candidate_logit_cap);
    agg_obj["opponent_logit_cap"] = absl::GetFlag(FLAGS_opponent_logit_cap);
    agg_obj["candidate_checkpoint_sha256"] = open_spiel::g_candidate_ckpt_sha256;
    agg_obj["opponent_checkpoint_sha256"] = open_spiel::g_opponent_ckpt_sha256;
    agg_obj["nonlinear_value_head"] =
        absl::GetFlag(FLAGS_nonlinear_value_head);
    agg_obj["opponent_nonlinear_value_head"] =
        absl::GetFlag(FLAGS_opponent_nonlinear_value_head);
    agg_obj["simulated_opponent_temperature"] = absl::GetFlag(FLAGS_simulated_opponent_temperature);
    agg_obj["external_opponent_temperature"] = absl::GetFlag(FLAGS_external_opponent_temperature);
    agg_obj["disable_time_limit"] = absl::GetFlag(FLAGS_disable_time_limit);
    agg_obj["audit_training_policy_iteration"] =
        absl::GetFlag(FLAGS_audit_training_policy_iteration);
    agg_obj["pi_primary_simulations"] =
        static_cast<int64_t>(absl::GetFlag(FLAGS_pi_primary_simulations));
    agg_obj["pi_continuation_simulations"] = static_cast<int64_t>(
        absl::GetFlag(FLAGS_pi_continuation_simulations));
    agg_obj["pi_short_window_simulations"] = static_cast<int64_t>(
        absl::GetFlag(FLAGS_pi_short_window_simulations));
    agg_obj["evaluator_mode"] = full_game_eval_mode;
    if (batched_mode) {
      agg_obj["batch_target_rows"] = static_cast<int64_t>(absl::GetFlag(FLAGS_batch_target_rows));
      agg_obj["batch_max_queue_wait_ms"] = static_cast<int64_t>(absl::GetFlag(FLAGS_batch_max_queue_wait_ms));
      if (search_coord != nullptr) {
        auto s = search_coord->GetStats();
        open_spiel::json::Object cs;
        cs["requests"] = static_cast<int64_t>(s.requests);
        cs["batches"] = static_cast<int64_t>(s.batches);
        cs["avg_batch_size"] = s.avg_batch_size;
        cs["max_batch_size"] = static_cast<int64_t>(s.max_batch_size);
        agg_obj["candidate_batcher_stats"] = cs;
      }
      if (opp_coord != nullptr) {
        auto s = opp_coord->GetStats();
        open_spiel::json::Object os;
        os["requests"] = static_cast<int64_t>(s.requests);
        os["batches"] = static_cast<int64_t>(s.batches);
        os["avg_batch_size"] = s.avg_batch_size;
        os["max_batch_size"] = static_cast<int64_t>(s.max_batch_size);
        agg_obj["opponent_batcher_stats"] = os;
      }
    }
    agg_obj["utility_divisor"] = absl::GetFlag(FLAGS_utility_divisor);
    agg_obj["elapsed_wall_time_s"] = total_wall_time;
    agg_obj["games_per_hour"] = total_games * 3600.0 / total_wall_time;
    agg_obj["search_wins"] = static_cast<int64_t>(global_stats.search_wins);
    agg_obj["search_winrate"] = (global_stats.search_wins * 100.0 / total_games);

    agg_obj["search_winrate_lcb"] = lcb_winrate;
    agg_obj["search_winrate_ucb"] = ucb_winrate;

    agg_obj["mean_vp_margin"] = mean_vp_margin;
    agg_obj["mean_vp_margin_lcb"] = lcb_vp_margin;
    agg_obj["mean_vp_margin_ucb"] = ucb_vp_margin;

    agg_obj["strategic_root_count"] = static_cast<int64_t>(global_stats.search_mcts_steps_count);
    agg_obj["total_simulations_completed"] = static_cast<int64_t>(global_stats.total_simulations_completed);
    agg_obj["incomplete_searches"] = static_cast<int64_t>(global_stats.total_incomplete_searches);

    open_spiel::json::Object roles_obj;
    roles_obj["kForcedOrBookkeeping"] = static_cast<int64_t>(global_stats.role_counts[0]);
    roles_obj["kLeaderSelection"] = static_cast<int64_t>(global_stats.role_counts[1]);
    roles_obj["kAgentPrimary"] = static_cast<int64_t>(global_stats.role_counts[2]);
    roles_obj["kAgentContinuation"] = static_cast<int64_t>(global_stats.role_counts[3]);
    roles_obj["kPurchase"] = static_cast<int64_t>(global_stats.role_counts[4]);
    roles_obj["kCombatIntrigue"] = static_cast<int64_t>(global_stats.role_counts[5]);
    roles_obj["kOtherOptional"] = static_cast<int64_t>(global_stats.role_counts[6]);
    agg_obj["decision_role_counts"] = roles_obj;

    double timeout_rate = global_stats.search_mcts_steps_count > 0 ? (global_stats.total_timeouts * 100.0) / global_stats.search_mcts_steps_count : 0.0;
    double fallback_rate = global_stats.search_mcts_steps_count > 0 ? (global_stats.total_fallbacks * 100.0) / global_stats.search_mcts_steps_count : 0.0;
    agg_obj["timeout_rate"] = timeout_rate;
    agg_obj["fallback_rate"] = fallback_rate;
    agg_obj["total_inferences"] = static_cast<int64_t>(global_stats.total_inferences);

    double avg_sims = global_stats.search_mcts_steps_count > 0 ? static_cast<double>(global_stats.total_simulations_completed) / global_stats.search_mcts_steps_count : 0.0;
    double avg_inf = global_stats.search_mcts_steps_count > 0 ? static_cast<double>(global_stats.total_inferences) / global_stats.search_mcts_steps_count : 0.0;
    agg_obj["average_simulations"] = avg_sims;
    agg_obj["average_inferences_per_step"] = avg_inf;
    agg_obj["mean_latency_s"] = avg_mcts_step_time;
    agg_obj["p95_latency_s"] = p95_latency;

    double avg_covered = global_stats.search_mcts_steps_count > 0 ? (global_stats.covered_prior_mass_sum / global_stats.search_mcts_steps_count) : 0.0;
    agg_obj["average_coverage"] = avg_covered;

    agg_obj["search_swordmaster_rate"] = (global_stats.search_swordmasters * 100.0) / total_games;
    agg_obj["opponent_swordmaster_rate"] = (global_stats.opponent_swordmasters * 100.0) / (3.0 * total_games);

    open_spiel::json::Object reasons_obj;
    for (const auto& pair : global_stats.fallback_reason_counts) {
      reasons_obj[pair.first] = static_cast<int64_t>(pair.second);
    }
    agg_obj["fallback_reasons"] = reasons_obj;

    open_spiel::json::Object rounds_obj;
    double sum_rounds = 0.0;
    for (int r : global_stats.rounds_played) {
      sum_rounds += r;
    }
    for (const auto& pair : round_counts) {
      rounds_obj[std::to_string(pair.first)] = static_cast<int64_t>(pair.second);
    }
    agg_obj["rounds_played_distribution"] = rounds_obj;

    double mean_rounds = global_stats.rounds_played.empty() ? 0.0 : sum_rounds / global_stats.rounds_played.size();
    agg_obj["mean_rounds"] = mean_rounds;

    double search_completeness = global_stats.search_mcts_steps_count > 0
        ? 1.0 - (static_cast<double>(global_stats.total_incomplete_searches) / global_stats.search_mcts_steps_count)
        : 1.0;
    agg_obj["search_completeness"] = search_completeness;

    std::ofstream agg_file(aggregate_json_path);
    if (agg_file) {
      agg_file << open_spiel::json::ToString(agg_obj, true);
    }
  }
}

}  // namespace open_spiel
