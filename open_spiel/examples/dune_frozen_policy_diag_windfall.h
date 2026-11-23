#ifndef OPEN_SPIEL_EXAMPLES_DUNE_FROZEN_POLICY_DIAG_WINDFALL_H_
#define OPEN_SPIEL_EXAMPLES_DUNE_FROZEN_POLICY_DIAG_WINDFALL_H_

#include "dune_frozen_policy_diag_evaluators.h"
#include "dune_frozen_policy_diag_mining.h"
#include "dune_frozen_policy_diag_report.h"

namespace open_spiel {
namespace dune_diagnostics {

// ===========================================================================
// Windfall Controller Evaluation Suite & Bank Selection
// ===========================================================================

inline bool ParseMiningRootJsonLine(const std::string& line, WindfallBankRecord* rec) {
  if (line.find("\"intrigue_name\": \"Windfall\"") == std::string::npos &&
      line.find("\"intrigue_name\":\"Windfall\"") == std::string::npos) {
    return false;
  }
  auto extract_int = [&](const std::string& key) -> int {
    std::string needle = "\"" + key + "\": ";
    size_t pos = line.find(needle);
    if (pos == std::string::npos) {
      needle = "\"" + key + "\":";
      pos = line.find(needle);
      if (pos == std::string::npos) return 0;
    }
    pos += needle.size();
    return std::stoi(line.substr(pos));
  };
  auto extract_uint64 = [&](const std::string& key) -> uint64_t {
    std::string needle = "\"" + key + "\": ";
    size_t pos = line.find(needle);
    if (pos == std::string::npos) {
      needle = "\"" + key + "\":";
      pos = line.find(needle);
      if (pos == std::string::npos) return 0ULL;
    }
    pos += needle.size();
    return std::stoull(line.substr(pos));
  };
  auto extract_str = [&](const std::string& key) -> std::string {
    std::string needle = "\"" + key + "\": \"";
    size_t pos = line.find(needle);
    if (pos == std::string::npos) {
      needle = "\"" + key + "\":\"";
      pos = line.find(needle);
      if (pos == std::string::npos) return "";
    }
    pos += needle.size();
    size_t end_pos = line.find("\"", pos);
    if (end_pos == std::string::npos) return "";
    return line.substr(pos, end_pos - pos);
  };

  rec->candidate_id = extract_int("candidate_id");
  rec->game_id = extract_int("game_id");
  rec->game_seed = extract_uint64("game_seed");
  rec->round = extract_int("round");
  rec->step_index = extract_int("step_index");
  rec->holder_player = extract_int("holder_player");
  rec->holder_leader = extract_str("holder_leader");
  rec->holder_solari = extract_int("holder_solari");
  rec->holder_water = extract_int("holder_water");
  rec->holder_spice = extract_int("holder_spice");
  rec->target_opponent = extract_int("target_opponent");
  rec->opponent_leader = extract_str("opponent_leader");
  rec->opponent_solari = extract_int("opponent_solari");
  rec->intrigue_action = extract_int("intrigue_action");
  rec->intrigue_name = extract_str("intrigue_name");
  return true;
}

inline std::vector<WindfallBankRecord> ReconstructAndVerifyWindfallRoots(
    const std::string& previous_run_dir,
    const std::shared_ptr<const Game>& game,
    IPolicyDiagnosticsEvaluator* evaluator,
    uint64_t base_seed = 20260915,
    float temperature = 1.0f) {

  std::string roots_file = previous_run_dir + "/mining_roots.jsonl";
  std::ifstream infile(roots_file);
  if (!infile) {
    SpielFatalError("Cannot open roots file: " + roots_file);
  }

  std::vector<WindfallBankRecord> bank;
  std::string line;
  while (std::getline(infile, line)) {
    if (line.empty()) continue;
    WindfallBankRecord rec;
    if (ParseMiningRootJsonLine(line, &rec)) {
      bank.push_back(rec);
    }
  }

  std::cout << "Loaded " << bank.size() << " Windfall root candidates from " << roots_file << "\n";
  std::cout << "Reconstructing full history actions and verifying bitwise replay...\n";

  for (size_t i = 0; i < bank.size(); ++i) {
    auto& rec = bank[i];
    int g = rec.game_id;
    uint64_t game_chance_seed = dune_seed::DeriveSeed(base_seed, 0x00E0, g, 0, dune_seed::kStreamChance);
    std::array<uint64_t, kNumPlayers> game_policy_seeds;
    std::array<std::mt19937_64, kNumPlayers> game_policy_rngs;
    for (int p = 0; p < kNumPlayers; ++p) {
      game_policy_seeds[p] = dune_seed::DeriveSeed(base_seed, 0x00E0, g, 0, dune_seed::kStreamPolicyPlayer0 + p);
      game_policy_rngs[p].seed(game_policy_seeds[p]);
    }
    std::mt19937_64 game_chance_rng(game_chance_seed);

    std::unique_ptr<State> state = game->NewInitialState();
    auto* s = dynamic_cast<DuneImperiumState*>(state.get());
    std::vector<Action> history;
    int steps = 0;
    bool found_step = false;

    while (!s->IsTerminal() && steps < 3000) {
      steps++;
      AdvanceStatus st = AdvanceThroughChanceAndAcks(s, game_chance_rng, &history);
      if (st != AdvanceStatus::kSuccessDecision) break;

      Player cp = s->CurrentPlayer();
      auto legal = s->LegalActions();
      if (legal.empty()) break;

      if (steps == rec.step_index) {
        found_step = true;
        rec.history_actions = history;
        rec.base_snapshot = evaluator->CaptureSnapshot(*s, rec.holder_player);

        if (cp != rec.holder_player) {
          rec.replay_verified = false;
          rec.replay_failure_reason = "Player mismatch at step: expected " + std::to_string(rec.holder_player) +
                                      ", got " + std::to_string(cp);
          break;
        }
        if (s->GetCurrentRound() != rec.round) {
          rec.replay_verified = false;
          rec.replay_failure_reason = "Round mismatch: expected " + std::to_string(rec.round) +
                                      ", got " + std::to_string(s->GetCurrentRound());
          break;
        }
        if (s->GetPlayerSolari(rec.holder_player) != rec.holder_solari) {
          rec.replay_verified = false;
          rec.replay_failure_reason = "Solari mismatch: expected " + std::to_string(rec.holder_solari) +
                                      ", got " + std::to_string(s->GetPlayerSolari(rec.holder_player));
          break;
        }
        if (!ContainsAction(legal, kActionPlayWindfall)) {
          rec.replay_verified = false;
          rec.replay_failure_reason = "kActionPlayWindfall not legal at step";
          break;
        }

        auto rep_res = VerifyReplayHistory(game, rec.history_actions, rec.holder_player, rec.base_snapshot, evaluator);
        rec.replay_verified = rep_res.matches;
        rec.replay_failure_reason = rep_res.failure_reason;
        break;
      }

      Action chosen = evaluator->SelectAction(*s, cp, false, temperature, game_policy_rngs[cp]);
      s->ApplyAction(chosen);
      history.push_back(chosen);
    }

    if (!found_step && rec.replay_failure_reason.empty()) {
      rec.replay_verified = false;
      rec.replay_failure_reason = "Step index " + std::to_string(rec.step_index) + " not reached in game " + std::to_string(g);
    }

    std::cout << "  Root [" << (i + 1) << "/" << bank.size() << "] Cand " << rec.candidate_id
              << " Game " << rec.game_id << " R" << rec.round << " Step " << rec.step_index
              << " P" << rec.holder_player << " (" << rec.holder_leader << "): "
              << (rec.replay_verified ? "REPLAY VERIFIED" : ("FAILED: " + rec.replay_failure_reason))
              << "\n";
  }

  return bank;
}

inline std::vector<WindfallBankRecord> SelectEvaluationBank(
    std::vector<WindfallBankRecord>& all_roots,
    int max_positions = 16) {

  std::map<int, std::vector<size_t>> round_to_indices;
  for (size_t i = 0; i < all_roots.size(); ++i) {
    if (all_roots[i].replay_verified) {
      round_to_indices[all_roots[i].round].push_back(i);
    } else {
      all_roots[i].is_selected = false;
      all_roots[i].selection_reason = "Excluded: Replay verification failed (" + all_roots[i].replay_failure_reason + ")";
    }
  }

  for (auto& [rnd, idxs] : round_to_indices) {
    std::sort(idxs.begin(), idxs.end(), [&](size_t a, size_t b) {
      return all_roots[a].game_id < all_roots[b].game_id;
    });
  }

  std::vector<size_t> selected_indices;
  std::set<int> selected_game_ids;

  // Pass 1: pick 1 candidate from every available round (rounds 2 through 10)
  for (auto& [rnd, idxs] : round_to_indices) {
    if (static_cast<int>(selected_indices.size()) >= max_positions) break;
    for (size_t idx : idxs) {
      int gid = all_roots[idx].game_id;
      if (selected_game_ids.find(gid) == selected_game_ids.end()) {
        selected_indices.push_back(idx);
        selected_game_ids.insert(gid);
        all_roots[idx].is_selected = true;
        all_roots[idx].selection_reason = "Selected in round-robin pass 1 for Round " + std::to_string(rnd);
        break;
      }
    }
  }

  // Pass 2: fill remaining slots up to max_positions from rounds with additional candidates
  while (static_cast<int>(selected_indices.size()) < max_positions) {
    bool added_any = false;
    for (auto& [rnd, idxs] : round_to_indices) {
      if (static_cast<int>(selected_indices.size()) >= max_positions) break;
      for (size_t idx : idxs) {
        if (!all_roots[idx].is_selected) {
          int gid = all_roots[idx].game_id;
          if (selected_game_ids.find(gid) == selected_game_ids.end()) {
            selected_indices.push_back(idx);
            selected_game_ids.insert(gid);
            all_roots[idx].is_selected = true;
            all_roots[idx].selection_reason = "Selected in round-robin pass 2 for Round " + std::to_string(rnd);
            added_any = true;
            break;
          }
        }
      }
    }
    if (!added_any) break;
  }

  // Mark exclusions for any remaining eligible candidates
  for (size_t i = 0; i < all_roots.size(); ++i) {
    if (all_roots[i].replay_verified && !all_roots[i].is_selected) {
      all_roots[i].selection_reason = "Excluded: Exceeded 16-position quota to balance round representation (Round " +
                                      std::to_string(all_roots[i].round) + ")";
    }
  }

  std::vector<WindfallBankRecord> selected_bank;
  for (size_t idx : selected_indices) {
    selected_bank.push_back(all_roots[idx]);
  }

  std::sort(selected_bank.begin(), selected_bank.end(), [](const WindfallBankRecord& a, const WindfallBankRecord& b) {
    if (a.round != b.round) return a.round < b.round;
    return a.game_id < b.game_id;
  });

  return selected_bank;
}

inline WindfallEvaluationSummary RunWindfallEvaluationSuite(
    const std::shared_ptr<const Game>& game,
    IPolicyDiagnosticsEvaluator* candidate_evaluator,
    IPolicyDiagnosticsEvaluator* opponent_evaluator,
    const std::vector<WindfallBankRecord>& all_roots,
    const std::vector<WindfallBankRecord>& selected_bank,
    const std::string& output_dir,
    int num_replicates = 64,
    uint64_t base_seed = 20260915,
    float temperature = 1.0f) {

  WindfallEvaluationSummary summary;
  summary.candidate_model = candidate_evaluator->ModelName();
  summary.candidate_sha256 = candidate_evaluator->ModelSha256();
  summary.opponent_model = opponent_evaluator->ModelName();
  summary.opponent_sha256 = opponent_evaluator->ModelSha256();
  summary.base_seed = base_seed;
  summary.temperature = temperature;
  summary.total_natural_roots_reconstructed = static_cast<int>(all_roots.size());
  int verified_cnt = 0;
  for (const auto& r : all_roots) {
    if (r.replay_verified) verified_cnt++;
  }
  summary.total_natural_roots_verified = verified_cnt;
  summary.positions_selected = static_cast<int>(selected_bank.size());
  summary.continuations_per_position = num_replicates;
  summary.total_continuation_pairs = summary.positions_selected * num_replicates;
  summary.bank = all_roots;

  WindfallTimingControlledEvaluator controlled_eval(candidate_evaluator, 0);

  std::vector<double> position_mean_util_diffs;
  std::vector<double> position_mean_vp_diffs;
  std::vector<double> all_util_diffs;
  std::vector<int> all_vp_diffs;

  std::cout << "\n=================================================================\n";
  std::cout << "=== RUNNING FROZEN-POLICY WINDFALL TIMING EVALUATION          ===\n";
  std::cout << "=================================================================\n";
  std::cout << "Positions:      " << selected_bank.size() << "\n";
  std::cout << "Replicates:     " << num_replicates << " per position ("
            << (selected_bank.size() * num_replicates) << " paired, "
            << (selected_bank.size() * num_replicates * 2) << " total games)\n";
  std::cout << "Arm A:          Unchanged " << candidate_evaluator->ModelName() << "\n";
  std::cout << "Arm B:          " << candidate_evaluator->ModelName() << " + WindfallTimingController\n";
  std::cout << "Opponents:      3x " << opponent_evaluator->ModelName() << "\n";
  std::cout << "Seed Domain:    0x00E8\n";
  std::cout << "=================================================================\n\n";

  for (size_t pos_idx = 0; pos_idx < selected_bank.size(); ++pos_idx) {
    const auto& root = selected_bank[pos_idx];
    int holder = root.holder_player;
    int opp_target = root.target_opponent;
    controlled_eval.SetHolder(holder);

    std::unique_ptr<State> base_state = game->NewInitialState();
    for (Action a : root.history_actions) {
      base_state->ApplyAction(a);
    }

    double pos_util_sum = 0.0;
    double pos_vp_sum = 0.0;
    int valid_replicates = 0;

    std::cout << "Evaluating Position [" << (pos_idx + 1) << "/" << selected_bank.size() << "] "
              << "Cand " << root.candidate_id << " (Game " << root.game_id << ", R" << root.round
              << ", Step " << root.step_index << ", P" << holder << " " << root.holder_leader
              << ", Solari=" << root.holder_solari << ")..." << std::flush;

    for (int m = 0; m < num_replicates; ++m) {
      uint64_t roll_chance_seed = dune_seed::DeriveSeed(base_seed, 0x00E8, root.candidate_id, m, dune_seed::kStreamChance);
      std::array<uint64_t, kNumPlayers> p_seeds;
      std::array<std::mt19937_64, kNumPlayers> p_rngs_a, p_rngs_b;
      for (int p = 0; p < kNumPlayers; ++p) {
        p_seeds[p] = dune_seed::DeriveSeed(base_seed, 0x00E8, root.candidate_id, m, dune_seed::kStreamPolicyPlayer0 + p);
        p_rngs_a[p].seed(p_seeds[p]);
        p_rngs_b[p].seed(p_seeds[p]);
      }
      std::mt19937_64 c_rng_a(roll_chance_seed);
      std::mt19937_64 c_rng_b(roll_chance_seed);

      PairedContinuationRecord cont_rec;
      cont_rec.candidate_id = root.candidate_id;
      cont_rec.pair_index = m;
      cont_rec.chance_seed = roll_chance_seed;
      cont_rec.policy_seeds = p_seeds;
      cont_rec.holder_player = holder;
      cont_rec.target_opponent = opp_target;

      // --- Rollout Arm A (Baseline) ---
      auto sa = base_state->Clone();
      int steps_a = 0;
      bool windfall_played_a = false;
      int play_round_a = -1, play_step_a = -1;

      while (!sa->IsTerminal() && steps_a < 3000) {
        steps_a++;
        AdvanceStatus st = AdvanceThroughChanceAndAcks(sa.get(), c_rng_a);
        if (st == AdvanceStatus::kSuccessTerminal) break;
        if (st != AdvanceStatus::kSuccessDecision) break;
        Player cp = sa->CurrentPlayer();
        Action a = open_spiel::kInvalidAction;
        if (cp == holder) {
          a = candidate_evaluator->SelectAction(*sa, cp, false, temperature, p_rngs_a[cp]);
        } else {
          a = opponent_evaluator->SelectAction(*sa, cp, false, temperature, p_rngs_a[cp]);
        }
        if (cp == holder && a == kActionPlayWindfall && !windfall_played_a) {
          windfall_played_a = true;
          play_round_a = dynamic_cast<DuneImperiumState*>(sa.get())->GetCurrentRound();
          play_step_a = steps_a;
        }
        sa->ApplyAction(a);
      }

      // --- Rollout Arm B (Controlled) ---
      auto sb = base_state->Clone();
      int steps_b = 0;
      bool windfall_played_b = false;
      int play_round_b = -1, play_step_b = -1;
      controlled_eval.SetContext(root.candidate_id, m);
      controlled_eval.ResetTurnStep();

      while (!sb->IsTerminal() && steps_b < 3000) {
        steps_b++;
        AdvanceStatus st = AdvanceThroughChanceAndAcks(sb.get(), c_rng_b);
        if (st == AdvanceStatus::kSuccessTerminal) break;
        if (st != AdvanceStatus::kSuccessDecision) break;
        Player cp = sb->CurrentPlayer();
        Action a = open_spiel::kInvalidAction;
        if (cp == holder) {
          a = controlled_eval.SelectAction(*sb, cp, false, temperature, p_rngs_b[cp]);
        } else {
          a = opponent_evaluator->SelectAction(*sb, cp, false, temperature, p_rngs_b[cp]);
        }
        if (cp == holder && a == kActionPlayWindfall && !windfall_played_b) {
          windfall_played_b = true;
          play_round_b = dynamic_cast<DuneImperiumState*>(sb.get())->GetCurrentRound();
          play_step_b = steps_b;
        }
        sb->ApplyAction(a);
      }

      cont_rec.branch_a_terminal = sa->IsTerminal();
      cont_rec.branch_a_status = sa->IsTerminal() ? "TERMINAL" : "TIMEOUT";
      cont_rec.branch_a_windfall_played = windfall_played_a;
      cont_rec.branch_a_windfall_play_round = play_round_a;
      cont_rec.branch_a_windfall_play_step = play_step_a;

      cont_rec.branch_b_terminal = sb->IsTerminal();
      cont_rec.branch_b_status = sb->IsTerminal() ? "TERMINAL" : "TIMEOUT";
      cont_rec.branch_b_windfall_played = windfall_played_b;
      cont_rec.branch_b_windfall_play_round = play_round_b;
      cont_rec.branch_b_windfall_play_step = play_step_b;
      cont_rec.branch_b_intrigue_played_later = windfall_played_b;
      cont_rec.branch_b_intrigue_play_round = play_round_b;
      cont_rec.branch_b_intrigue_play_step = play_step_b;

      // EXPLICIT NATURAL TERMINAL CHECK BEFORE READING RETURNS
      if (sa->IsTerminal() && sb->IsTerminal()) {
        cont_rec.is_valid = true;
        for (int p = 0; p < kNumPlayers; ++p) {
          cont_rec.branch_a_returns[p] = sa->Returns()[p];
          cont_rec.branch_b_returns[p] = sb->Returns()[p];
          cont_rec.branch_a_vps[p] = dynamic_cast<DuneImperiumState*>(sa.get())->GetPlayerVp(p);
          cont_rec.branch_b_vps[p] = dynamic_cast<DuneImperiumState*>(sb.get())->GetPlayerVp(p);
        }
        double u_a = cont_rec.branch_a_returns[holder];
        double u_b = cont_rec.branch_b_returns[holder];
        int vp_a = cont_rec.branch_a_vps[holder];
        int vp_b = cont_rec.branch_b_vps[holder];

        double u_diff = u_b - u_a;
        int vp_diff = vp_b - vp_a;
        cont_rec.holder_utility_diff = u_diff;
        cont_rec.holder_vp_diff = vp_diff;

        pos_util_sum += u_diff;
        pos_vp_sum += vp_diff;
        all_util_diffs.push_back(u_diff);
        all_vp_diffs.push_back(vp_diff);
        valid_replicates++;
      } else {
        cont_rec.is_valid = false;
        summary.run_has_invalidation_failure = true;
        summary.invalidation_reason = "Continuation timed out before terminal state (A=" +
            cont_rec.branch_a_status + ", B=" + cont_rec.branch_b_status + ")";
      }

      summary.continuation_records.push_back(cont_rec);
    }

    if (valid_replicates > 0) {
      double mean_u = pos_util_sum / valid_replicates;
      double mean_v = pos_vp_sum / valid_replicates;
      position_mean_util_diffs.push_back(mean_u);
      position_mean_vp_diffs.push_back(mean_v);
      std::cout << " Delta Util=" << std::fixed << std::setprecision(4) << mean_u
                << ", Delta VP=" << std::setprecision(2) << mean_v << "\n";
    } else {
      std::cout << " FAILED (0 valid replicates)\n";
    }
  }

  summary.decisions = controlled_eval.DecisionRecords();
  summary.total_controller_decisions = static_cast<int>(summary.decisions.size());

  for (const auto& d : summary.decisions) {
    if (d.windfall_played) {
      summary.windfall_plays_executed++;
      if (d.plan_requires_windfall) summary.necessary_spending_unlocked++;
    } else {
      summary.early_plays_prevented++;
      if (d.action_taken == kActionEndTurn) summary.end_turn_retentions++;
      else summary.purchases_already_affordable++;
    }
    if (d.secrets_exception) summary.secrets_exceptions_triggered++;
    if (d.terminal_value_exception) summary.terminal_value_exceptions_triggered++;
    if (d.fallback_active) summary.fallbacks_triggered++;
  }

  size_t K = position_mean_util_diffs.size();
  if (K > 0) {
    double sum_u = 0.0, sum_vp = 0.0;
    for (size_t k = 0; k < K; ++k) {
      sum_u += position_mean_util_diffs[k];
      sum_vp += position_mean_vp_diffs[k];
    }
    summary.mean_placement_utility_diff = sum_u / K;
    summary.mean_holder_vp_diff = sum_vp / K;

    if (K > 1) {
      double var_u = 0.0;
      for (double d : position_mean_util_diffs) {
        var_u += (d - summary.mean_placement_utility_diff) * (d - summary.mean_placement_utility_diff);
      }
      var_u /= (K - 1);
      summary.placement_utility_std_err = std::sqrt(var_u / K);

      double t_crit = (K >= 16) ? 2.131 : 2.20;
      summary.placement_utility_ci_lower = summary.mean_placement_utility_diff - t_crit * summary.placement_utility_std_err;
      summary.placement_utility_ci_upper = summary.mean_placement_utility_diff + t_crit * summary.placement_utility_std_err;
    }
  }

  if (summary.run_has_invalidation_failure) {
    summary.verdict = "Invalid";
    summary.verdict_explanation = "Correctness or record-integrity failure occurred: " + summary.invalidation_reason;
  } else if (summary.placement_utility_ci_lower > 0.0) {
    summary.verdict = "Beneficial on the tested bank";
    summary.verdict_explanation = "Positive placement-utility effect with 95% uncertainty interval excluding zero, without correctness failures.";
  } else if (summary.placement_utility_ci_upper < 0.0) {
    summary.verdict = "Harmful on the tested bank";
    summary.verdict_explanation = "Supported negative placement-utility effect with 95% uncertainty interval strictly below zero.";
  } else {
    summary.verdict = "Inconclusive";
    summary.verdict_explanation = "Neither direction established: 95% confidence interval spans zero (uncertainty includes both positive and negative outcomes).";
  }

  if (!output_dir.empty()) {
    std::filesystem::create_directories(output_dir);

    // 1. Controller decisions JSONL
    std::ofstream dec_out(output_dir + "/controller_decisions.jsonl");
    for (const auto& d : summary.decisions) {
      dec_out << FormatWindfallDecisionRecordJson(d) << "\n";
    }

    // 2. Continuation pairs JSONL
    std::ofstream cont_out(output_dir + "/continuation_pairs.jsonl");
    for (const auto& c : summary.continuation_records) {
      cont_out << FormatPairedContinuationRecordJson(c) << "\n";
    }

    // 3. Frozen evaluation bank JSONL
    std::ofstream bank_out(output_dir + "/frozen_evaluation_bank.jsonl");
    for (const auto& b : summary.bank) {
      bank_out << FormatWindfallBankRecordJson(b) << "\n";
    }

    // 4. Summary JSON
    std::ofstream sum_out(output_dir + "/evaluation_summary.json");
    sum_out << FormatWindfallEvaluationReportJson(summary);

    // 5. Run manifest JSON
    std::ofstream man_out(output_dir + "/run_manifest.json");
    man_out << "{\n"
            << "  \"candidate_model\": \"" << EscapeJson(summary.candidate_model) << "\",\n"
            << "  \"candidate_sha256\": \"" << EscapeJson(summary.candidate_sha256) << "\",\n"
            << "  \"opponent_model\": \"" << EscapeJson(summary.opponent_model) << "\",\n"
            << "  \"opponent_sha256\": \"" << EscapeJson(summary.opponent_sha256) << "\",\n"
            << "  \"base_seed\": " << summary.base_seed << ",\n"
            << "  \"temperature\": " << summary.temperature << ",\n"
            << "  \"num_positions\": " << summary.positions_selected << ",\n"
            << "  \"num_replicates\": " << summary.continuations_per_position << ",\n"
            << "  \"seed_domain\": \"0x00E8\",\n"
            << "  \"verdict\": \"" << EscapeJson(summary.verdict) << "\"\n"
            << "}\n";

    // 6. Bank manifest JSON
    std::ofstream bman_out(output_dir + "/bank_manifest.json");
    bman_out << "{\n"
             << "  \"total_reconstructed\": " << summary.bank.size() << ",\n"
             << "  \"total_selected\": " << summary.positions_selected << ",\n"
             << "  \"selection_rule\": \"Outcome-independent round-robin across rounds 2-10, tie-broken by game_id ascending, max 1 position per source game\",\n"
             << "  \"positions\": [\n";
    for (size_t i = 0; i < summary.bank.size(); ++i) {
      const auto& b = summary.bank[i];
      bman_out << "    {\"candidate_id\": " << b.candidate_id
               << ", \"game_id\": " << b.game_id
               << ", \"round\": " << b.round
               << ", \"step_index\": " << b.step_index
               << ", \"holder_player\": " << b.holder_player
               << ", \"holder_leader\": \"" << EscapeJson(b.holder_leader) << "\""
               << ", \"holder_solari\": " << b.holder_solari
               << ", \"replay_verified\": " << (b.replay_verified ? "true" : "false")
               << ", \"is_selected\": " << (b.is_selected ? "true" : "false")
               << ", \"selection_reason\": \"" << EscapeJson(b.selection_reason) << "\"}";
      if (i + 1 < summary.bank.size()) bman_out << ",\n";
      else bman_out << "\n";
    }
    bman_out << "  ]\n}\n";

    std::cout << "\nWrote all evaluation artifacts to: " << output_dir << "\n";
  }

  return summary;
}

} // namespace dune_diagnostics
} // namespace open_spiel

#endif // OPEN_SPIEL_EXAMPLES_DUNE_FROZEN_POLICY_DIAG_WINDFALL_H_
