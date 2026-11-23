#ifndef OPEN_SPIEL_EXAMPLES_DUNE_FROZEN_POLICY_DIAG_MINING_H_
#define OPEN_SPIEL_EXAMPLES_DUNE_FROZEN_POLICY_DIAG_MINING_H_

#include "dune_frozen_policy_diag_evaluators.h"

namespace open_spiel {
namespace dune_diagnostics {

inline ReplayVerificationResult VerifyReplayHistory(
    const std::shared_ptr<const Game>& game,
    const std::vector<Action>& history,
    Player target_player,
    const ModelInputSnapshot& expected_snapshot,
    IPolicyDiagnosticsEvaluator* evaluator) {
  ReplayVerificationResult res;
  res.matches = false;
  if (!game) {
    res.failure_reason = "Game pointer is null";
    return res;
  }
  std::unique_ptr<State> s = game->NewInitialState();
  for (size_t i = 0; i < history.size(); ++i) {
    if (s->IsTerminal()) {
      res.failure_reason = "Game terminated unexpectedly during replay at step " + std::to_string(i);
      return res;
    }
    s->ApplyAction(history[i]);
  }
  res.steps_replayed = history.size();

  // 1. Verify state string
  std::string replayed_state_str = s->ToString();
  if (replayed_state_str != expected_snapshot.state_string) {
    res.failure_reason = "Replayed state string mismatch.";
    return res;
  }

  // 2. Capture snapshot at replayed state
  ModelInputSnapshot replayed_snap = evaluator->CaptureSnapshot(*s, target_player);

  // 3. Verify legal actions
  if (replayed_snap.legal_actions != expected_snapshot.legal_actions) {
    res.failure_reason = "Replayed legal actions mismatch.";
    return res;
  }

  // 4. Verify observation tensor (including appendix)
  if (replayed_snap.observation_tensor.size() != expected_snapshot.observation_tensor.size()) {
    res.failure_reason = "Observation tensor size mismatch: expected " +
        std::to_string(expected_snapshot.observation_tensor.size()) +
        ", got " + std::to_string(replayed_snap.observation_tensor.size());
    return res;
  }
  for (size_t i = 0; i < expected_snapshot.observation_tensor.size(); ++i) {
    if (std::abs(replayed_snap.observation_tensor[i] - expected_snapshot.observation_tensor[i]) > 1e-5f) {
      res.failure_reason = "Observation tensor value mismatch at index " + std::to_string(i);
      return res;
    }
  }

  // 5. Verify semantic action features and categorical IDs
  if (replayed_snap.action_features != expected_snapshot.action_features) {
    res.failure_reason = "Semantic action features mismatch.";
    return res;
  }
  if (replayed_snap.action_card_ids != expected_snapshot.action_card_ids) {
    res.failure_reason = "Semantic action card IDs mismatch.";
    return res;
  }
  if (replayed_snap.action_space_ids != expected_snapshot.action_space_ids) {
    res.failure_reason = "Semantic action space IDs mismatch.";
    return res;
  }
  if (replayed_snap.action_supported != expected_snapshot.action_supported) {
    res.failure_reason = "Semantic action supported flags mismatch.";
    return res;
  }

  // 6. Verify policy probabilities
  if (replayed_snap.policy.size() != expected_snapshot.policy.size()) {
    res.failure_reason = "Policy size mismatch.";
    return res;
  }
  for (size_t i = 0; i < expected_snapshot.policy.size(); ++i) {
    if (replayed_snap.policy[i].first != expected_snapshot.policy[i].first ||
        std::abs(replayed_snap.policy[i].second - expected_snapshot.policy[i].second) > 1e-5) {
      res.failure_reason = "Policy probability mismatch at action " +
          std::to_string(expected_snapshot.policy[i].first);
      return res;
    }
  }

  res.matches = true;
  return res;
}

// ===========================================================================
// Actual-Play Intrigue Mining Engine
// ===========================================================================

inline std::string LeaderName(LeaderId id) {
  for (const auto& entry : dune_imperium::kLeaders) {
    if (entry.id == static_cast<int>(id)) return std::string(entry.name);
  }
  return absl::StrCat("Leader_", static_cast<int>(id));
}

// Advances both branches to the designated target opponent's main placement decision (board space 600..621,
// Reveal, or AgentPass). Any intervening player actions before the target opponent are simulated using
// that player's dedicated policy RNG stream.
inline bool AdvanceToOpponentMainPlacement(
    DuneImperiumState* s,
    Player target_opponent,
    IPolicyDiagnosticsEvaluator* evaluator,
    std::mt19937_64& chance_rng,
    std::array<std::mt19937_64, kNumPlayers>& policy_rngs,
    bool greedy,
    float temperature,
    Action* out_opp_action,
    ModelInputSnapshot* out_snapshot = nullptr,
    int max_steps = 300) {

  int steps = 0;
  while (!s->IsTerminal() && steps < max_steps) {
    steps++;
    AdvanceStatus st = AdvanceThroughChanceAndAcks(s, chance_rng);
    if (st != AdvanceStatus::kSuccessDecision) {
      return false;
    }

    Player cp = s->CurrentPlayer();
    auto legal = s->LegalActions();
    if (legal.empty()) return false;

    if (cp == target_opponent) {
      Action act = evaluator->SelectAction(*s, cp, greedy, temperature, policy_rngs[cp]);
      if (IsAgentSpaceAction(act) || act == kActionReveal || act == kActionAgentPass) {
        if (out_opp_action != nullptr) *out_opp_action = act;
        if (out_snapshot != nullptr) {
          *out_snapshot = evaluator->CaptureSnapshot(*s, cp);
        }
        return true;
      }
      // Card selection, graft partner, or free intrigue
      s->ApplyAction(act);
    } else {
      // Intervening action by another player
      Action act = evaluator->SelectAction(*s, cp, greedy, temperature, policy_rngs[cp]);
      s->ApplyAction(act);
    }
  }
  return false;
}

// Step-down Holm-Bonferroni adjusted p-values for multiple testing control.
inline std::vector<double> ComputeHolmBonferroniAdjustedPValues(const std::vector<double>& raw_p) {
  size_t m = raw_p.size();
  if (m == 0) return {};
  if (m == 1) return {raw_p[0]};

  std::vector<size_t> order(m);
  std::iota(order.begin(), order.end(), 0);
  std::sort(order.begin(), order.end(), [&raw_p](size_t i, size_t j) {
    return raw_p[i] < raw_p[j];
  });

  std::vector<double> adj_p(m, 0.0);
  double running_max = 0.0;
  for (size_t k = 0; k < m; ++k) {
    size_t idx = order[k];
    double multiplier = static_cast<double>(m - k);
    double cur_adj = std::min(1.0, raw_p[idx] * multiplier);
    running_max = std::max(running_max, cur_adj);
    adj_p[idx] = running_max;
  }
  return adj_p;
}

// Simulates ordinary self-play games from NewInitialState() without card grants or artificial modifications.
// Collects candidate roots, performs complete model input replay verification, runs paired confirmations
// with Holm-Bonferroni correction, and runs full-game continuations with explicit terminal state checks.
inline IntrigueMiningSummary RunOrdinaryGameIntrigueMining(
    IPolicyDiagnosticsEvaluator* evaluator,
    int num_games = 100,
    uint64_t base_seed = 20260915,
    int discovery_pairs = 4,
    int max_candidates = 8,
    int confirmation_samples = 64,
    int max_outcome_candidates = 4,
    int continuation_samples = 64,
    float temperature = 1.0f) {

  IntrigueMiningSummary summary;
  summary.games_simulated = num_games;
  int candidate_counter = 0;

  auto game = open_spiel::LoadGame("dune_imperium");

  for (int g = 0; g < num_games; ++g) {
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
    while (!s->IsTerminal() && steps < 3000) {
      steps++;
      AdvanceStatus st = AdvanceThroughChanceAndAcks(s, game_chance_rng, &history);
      if (st != AdvanceStatus::kSuccessDecision) break;

      Player cp = s->CurrentPlayer();
      auto legal = s->LegalActions();
      if (legal.empty()) break;

      Action chosen = evaluator->SelectAction(*s, cp, /*greedy=*/false, temperature, game_policy_rngs[cp]);

      // Check if chosen action is an early resource intrigue play while EndTurn is legal
      bool is_resource_intrigue = (chosen == kActionPlayWindfall || chosen == kActionPlayWaterPeddlersUnion);
      if (is_resource_intrigue && ContainsAction(legal, kActionEndTurn)) {
        summary.total_resource_intrigue_plays++;
        if (chosen == kActionPlayWindfall) summary.windfall_plays++;
        if (chosen == kActionPlayWaterPeddlersUnion) summary.water_peddlers_union_plays++;

        // Determine designated target opponent before branching:
        // Probe Branch B to see who the next active player will be
        auto temp_b = s->Clone();
        temp_b->ApplyAction(kActionEndTurn);
        std::mt19937_64 temp_c_rng(game_chance_seed + steps);
        AdvanceStatus temp_st = AdvanceThroughChanceAndAcks(temp_b.get(), temp_c_rng);
        Player designated_opponent = (temp_st == AdvanceStatus::kSuccessDecision) ? temp_b->CurrentPlayer() : -1;

        if (designated_opponent != -1 && designated_opponent != cp) {
          IntrigueMiningCandidate cand;
          cand.candidate_id = candidate_counter++;
          cand.game_id = g;
          cand.game_seed = game_chance_seed;
          cand.step_index = steps;
          cand.round = s->GetCurrentRound();
          cand.holder_player = cp;
          cand.holder_leader = LeaderName(static_cast<LeaderId>(s->PlayerLeader(cp)));
          cand.intrigue_action = chosen;
          cand.intrigue_name = (chosen == kActionPlayWindfall ? "Windfall" : "Water Peddlers Union");
          cand.holder_solari = s->GetPlayerSolari(cp);
          cand.holder_water = s->GetPlayerWaterForTesting(cp);
          cand.holder_spice = s->GetPlayerSpiceForTesting(cp);

          cand.opponent_player = designated_opponent;
          cand.opponent_leader = LeaderName(static_cast<LeaderId>(s->PlayerLeader(designated_opponent)));
          cand.opponent_solari = s->GetPlayerSolari(designated_opponent);
          cand.opponent_water = s->GetPlayerWaterForTesting(designated_opponent);
          cand.opponent_spice = s->GetPlayerSpiceForTesting(designated_opponent);
          cand.opponent_agents = s->GetPlayerAgentsRemainingForTesting(designated_opponent);

          NaturalRootRecord root_rec;
          root_rec.candidate_id = cand.candidate_id;
          root_rec.game_id = g;
          root_rec.game_seed = game_chance_seed;
          root_rec.round = cand.round;
          root_rec.step_index = steps;
          root_rec.holder_player = cp;
          root_rec.holder_leader = cand.holder_leader;
          root_rec.holder_solari = cand.holder_solari;
          root_rec.holder_water = cand.holder_water;
          root_rec.holder_spice = cand.holder_spice;
          root_rec.target_opponent = designated_opponent;
          root_rec.opponent_leader = cand.opponent_leader;
          root_rec.opponent_solari = cand.opponent_solari;
          root_rec.opponent_water = cand.opponent_water;
          root_rec.opponent_spice = cand.opponent_spice;
          root_rec.opponent_agents = cand.opponent_agents;
          root_rec.intrigue_action = chosen;
          root_rec.intrigue_name = cand.intrigue_name;
          root_rec.history_actions = history;
          root_rec.base_snapshot = evaluator->CaptureSnapshot(*s, cp);

          // 4 Discovery paired rollouts
          int divergent_trials = 0;
          std::map<Action, int> count_a, count_b;
          Action first_act_a = open_spiel::kInvalidAction, first_act_b = open_spiel::kInvalidAction;

          for (int disc_trial = 0; disc_trial < discovery_pairs; ++disc_trial) {
            uint64_t disc_chance_seed = dune_seed::DeriveSeed(game_chance_seed, 0x00E1, cand.candidate_id, disc_trial, dune_seed::kStreamChance);
            std::array<std::mt19937_64, kNumPlayers> disc_p_rngs_a, disc_p_rngs_b;
            for (int p = 0; p < kNumPlayers; ++p) {
              uint64_t s_p = dune_seed::DeriveSeed(game_chance_seed, 0x00E1, cand.candidate_id, disc_trial, dune_seed::kStreamPolicyPlayer0 + p);
              disc_p_rngs_a[p].seed(s_p);
              disc_p_rngs_b[p].seed(s_p);
            }
            std::mt19937_64 c_rng_a(disc_chance_seed);
            std::mt19937_64 c_rng_b(disc_chance_seed);

            // Branch A
            auto da = s->Clone();
            da->ApplyAction(chosen);
            AdvanceThroughChanceAndAcks(da.get(), c_rng_a);
            if (ContainsAction(da->LegalActions(), kActionEndTurn)) {
              da->ApplyAction(kActionEndTurn);
            }
            Action cur_act_a = open_spiel::kInvalidAction;
            bool ok_a = AdvanceToOpponentMainPlacement(
                dynamic_cast<DuneImperiumState*>(da.get()), designated_opponent, evaluator,
                c_rng_a, disc_p_rngs_a, /*greedy=*/false, temperature, &cur_act_a);

            // Branch B
            auto db = s->Clone();
            db->ApplyAction(kActionEndTurn);
            Action cur_act_b = open_spiel::kInvalidAction;
            bool ok_b = AdvanceToOpponentMainPlacement(
                dynamic_cast<DuneImperiumState*>(db.get()), designated_opponent, evaluator,
                c_rng_b, disc_p_rngs_b, /*greedy=*/false, temperature, &cur_act_b);

            if (ok_a && ok_b) {
              count_a[cur_act_a]++;
              count_b[cur_act_b]++;
              if (disc_trial == 0) {
                first_act_a = cur_act_a;
                first_act_b = cur_act_b;
              }
              if (cur_act_a != cur_act_b) {
                divergent_trials++;
              }
            }
          }

          cand.discovery_divergent_count = divergent_trials;
          cand.discovery_opp_action_early = first_act_a;
          cand.discovery_opp_action_early_name = DiagnosticActionToString(first_act_a);
          cand.discovery_opp_action_hold = first_act_b;
          cand.discovery_opp_action_hold_name = DiagnosticActionToString(first_act_b);

          std::set<Action> all_actions;
          for (const auto& [a, c] : count_a) all_actions.insert(a);
          for (const auto& [a, c] : count_b) all_actions.insert(a);

          Action best_target = first_act_a;
          double max_shift = -1.0;
          int best_dir = 1;
          for (Action a : all_actions) {
            double freq_a = static_cast<double>(count_a[a]) / discovery_pairs;
            double freq_b = static_cast<double>(count_b[a]) / discovery_pairs;
            double diff = freq_a - freq_b;
            if (std::abs(diff) > max_shift) {
              max_shift = std::abs(diff);
              best_target = a;
              best_dir = (diff >= 0.0) ? 1 : -1;
            }
          }
          cand.target_action = best_target;
          cand.target_action_name = DiagnosticActionToString(best_target);
          cand.predicted_direction = best_dir;
          cand.discovery_delta = (discovery_pairs > 0) ?
              (static_cast<double>(count_a[best_target] - count_b[best_target]) / discovery_pairs) : 0.0;

          summary.candidates.push_back(cand);
          summary.roots.push_back(root_rec);
          summary.candidates_discovered++;
        }
      }

      s->ApplyAction(chosen);
      history.push_back(chosen);
    }
  }

  // Deterministic candidate ranking rule:
  // 1) Disagreement count descending
  // 2) Effect magnitude descending
  // 3) Game ID ascending
  // 4) Step index ascending
  std::vector<size_t> candidate_indices(summary.candidates.size());
  std::iota(candidate_indices.begin(), candidate_indices.end(), 0);
  std::sort(candidate_indices.begin(), candidate_indices.end(), [&](size_t i, size_t j) {
    const auto& a = summary.candidates[i];
    const auto& b = summary.candidates[j];
    if (a.discovery_divergent_count != b.discovery_divergent_count) {
      return a.discovery_divergent_count > b.discovery_divergent_count;
    }
    if (std::abs(a.discovery_delta) != std::abs(b.discovery_delta)) {
      return std::abs(a.discovery_delta) > std::abs(b.discovery_delta);
    }
    if (a.game_id != b.game_id) {
      return a.game_id < b.game_id;
    }
    return a.step_index < b.step_index;
  });

  std::vector<size_t> selected_confirmation_indices;
  for (size_t idx : candidate_indices) {
    if (static_cast<int>(selected_confirmation_indices.size()) >= max_candidates) break;
    selected_confirmation_indices.push_back(idx);
  }
  summary.candidates_tested_confirmation = static_cast<int>(selected_confirmation_indices.size());

  // Run Confirmation Pass on selected candidates
  for (size_t idx : selected_confirmation_indices) {
    auto& cand = summary.candidates[idx];
    auto& root = summary.roots[idx];

    // 1. Complete Replay Verification
    auto replay_res = VerifyReplayHistory(game, root.history_actions, cand.holder_player, root.base_snapshot, evaluator);
    cand.replay_verified = replay_res.matches;
    cand.replay_failure_reason = replay_res.failure_reason;
    root.replay_verified = replay_res.matches;
    root.replay_failure_reason = replay_res.failure_reason;

    // 2. Reconstruct base state
    std::unique_ptr<State> base_state = game->NewInitialState();
    for (Action a : root.history_actions) {
      base_state->ApplyAction(a);
    }

    // Direct placement argmax without sampling
    auto da_direct = base_state->Clone();
    da_direct->ApplyAction(cand.intrigue_action);
    std::mt19937_64 direct_c_rng(12345);
    AdvanceThroughChanceAndAcks(da_direct.get(), direct_c_rng);
    if (ContainsAction(da_direct->LegalActions(), kActionEndTurn)) {
      da_direct->ApplyAction(kActionEndTurn);
    }
    AdvanceThroughChanceAndAcks(da_direct.get(), direct_c_rng);

    auto db_direct = base_state->Clone();
    db_direct->ApplyAction(kActionEndTurn);
    AdvanceThroughChanceAndAcks(db_direct.get(), direct_c_rng);

    if (da_direct->CurrentPlayer() == cand.opponent_player && db_direct->CurrentPlayer() == cand.opponent_player) {
      auto policy_a = evaluator->GetPolicy(*da_direct, cand.opponent_player);
      auto policy_b = evaluator->GetPolicy(*db_direct, cand.opponent_player);
      double max_a = -1.0, max_b = -1.0;
      Action arg_a = open_spiel::kInvalidAction, arg_b = open_spiel::kInvalidAction;
      for (const auto& [a, p] : policy_a) {
        if (p > max_a) { max_a = p; arg_a = a; }
        if (a == cand.target_action) cand.opponent_target_prob_early = p;
      }
      for (const auto& [a, p] : policy_b) {
        if (p > max_b) { max_b = p; arg_b = a; }
        if (a == cand.target_action) cand.opponent_target_prob_hold = p;
      }
      cand.opponent_argmax_early = arg_a;
      cand.opponent_argmax_early_name = DiagnosticActionToString(arg_a);
      cand.opponent_argmax_hold = arg_b;
      cand.opponent_argmax_hold_name = DiagnosticActionToString(arg_b);
    }

    // 3. Fresh sample rollouts (independent seeds from discovery)
    cand.confirmation_samples = confirmation_samples;
    std::vector<int> diffs;
    int div_count = 0;
    int early_target_cnt = 0;
    int hold_target_cnt = 0;

    for (int m = 0; m < confirmation_samples; ++m) {
      uint64_t fresh_chance_seed = dune_seed::DeriveSeed(cand.game_seed, 0x00E2, cand.candidate_id, m, dune_seed::kStreamChance);
      std::array<std::mt19937_64, kNumPlayers> p_rngs_a, p_rngs_b;
      std::array<uint64_t, kNumPlayers> p_seeds;
      for (int p = 0; p < kNumPlayers; ++p) {
        p_seeds[p] = dune_seed::DeriveSeed(cand.game_seed, 0x00E2, cand.candidate_id, m, dune_seed::kStreamPolicyPlayer0 + p);
        p_rngs_a[p].seed(p_seeds[p]);
        p_rngs_b[p].seed(p_seeds[p]);
      }
      std::mt19937_64 c_rng_a(fresh_chance_seed);
      std::mt19937_64 c_rng_b(fresh_chance_seed);

      // Branch A
      auto sa = base_state->Clone();
      sa->ApplyAction(cand.intrigue_action);
      AdvanceThroughChanceAndAcks(sa.get(), c_rng_a);
      if (ContainsAction(sa->LegalActions(), kActionEndTurn)) {
        sa->ApplyAction(kActionEndTurn);
      }
      Action act_a = open_spiel::kInvalidAction;
      bool ok_a = AdvanceToOpponentMainPlacement(
          dynamic_cast<DuneImperiumState*>(sa.get()), cand.opponent_player, evaluator,
          c_rng_a, p_rngs_a, /*greedy=*/false, temperature, &act_a);

      // Branch B
      auto sb = base_state->Clone();
      sb->ApplyAction(kActionEndTurn);
      Action act_b = open_spiel::kInvalidAction;
      bool ok_b = AdvanceToOpponentMainPlacement(
          dynamic_cast<DuneImperiumState*>(sb.get()), cand.opponent_player, evaluator,
          c_rng_b, p_rngs_b, /*greedy=*/false, temperature, &act_b);

      PairedConfirmationRecord conf_rec;
      conf_rec.candidate_id = cand.candidate_id;
      conf_rec.pair_index = m;
      conf_rec.chance_seed = fresh_chance_seed;
      conf_rec.policy_seeds = p_seeds;
      conf_rec.holder_player = cand.holder_player;
      conf_rec.target_opponent = cand.opponent_player;
      conf_rec.reached_target_early = ok_a;
      conf_rec.reached_target_hold = ok_b;
      conf_rec.act_early = act_a;
      conf_rec.act_early_name = DiagnosticActionToString(act_a);
      conf_rec.act_hold = act_b;
      conf_rec.act_hold_name = DiagnosticActionToString(act_b);

      if (ok_a && ok_b) {
        if (act_a != act_b) div_count++;
        int ind_a = (act_a == cand.target_action) ? 1 : 0;
        int ind_b = (act_b == cand.target_action) ? 1 : 0;
        if (ind_a) early_target_cnt++;
        if (ind_b) hold_target_cnt++;
        int diff = ind_a - ind_b;
        diffs.push_back(diff);
        conf_rec.indicator_early = ind_a;
        conf_rec.indicator_hold = ind_b;
        conf_rec.indicator_diff = diff;
        conf_rec.is_valid = true;
      } else {
        conf_rec.is_valid = false;
        summary.run_has_invalidation_failure = true;
        summary.invalidation_reason = "Confirmation pair failed to reach target opponent in both branches";
      }
      summary.confirmation_records.push_back(conf_rec);
    }

    cand.confirmed_divergent_count = div_count;
    cand.confirmed_divergent_rate = (confirmation_samples > 0) ?
        (static_cast<double>(div_count) / confirmation_samples) : 0.0;
    cand.confirmation_early_target_count = early_target_cnt;
    cand.confirmation_hold_target_count = hold_target_cnt;

    if (!diffs.empty()) {
      double n = static_cast<double>(diffs.size());
      double sum_d = std::accumulate(diffs.begin(), diffs.end(), 0.0);
      cand.delta_prob = sum_d / n;
      double var_d = 0.0;
      for (int d : diffs) {
        var_d += (d - cand.delta_prob) * (d - cand.delta_prob);
      }
      var_d = (n > 1.0) ? (var_d / (n - 1.0)) : 0.0;
      cand.std_err_prob = std::sqrt(var_d / n);
      double z = (cand.std_err_prob > 0.0) ? (std::abs(cand.delta_prob) / cand.std_err_prob) : 0.0;
      cand.p_value_raw = std::erfc(z / std::sqrt(2.0));
      cand.ci_prob_lower = cand.delta_prob - 1.95996 * cand.std_err_prob;
      cand.ci_prob_upper = cand.delta_prob + 1.95996 * cand.std_err_prob;
    }
  }

  // Multiple testing correction across tested confirmation candidates
  std::vector<double> conf_raw_p;
  for (size_t idx : selected_confirmation_indices) {
    conf_raw_p.push_back(summary.candidates[idx].p_value_raw);
  }
  std::vector<double> conf_adj_p = ComputeHolmBonferroniAdjustedPValues(conf_raw_p);
  for (size_t i = 0; i < selected_confirmation_indices.size(); ++i) {
    size_t idx = selected_confirmation_indices[i];
    auto& cand = summary.candidates[idx];
    cand.p_value_adjusted = conf_adj_p[i];

    bool dir_match = (cand.predicted_direction * cand.delta_prob > 0.0);
    bool practical = (std::abs(cand.delta_prob) >= 0.05);
    bool sig = (cand.p_value_adjusted < 0.05);
    if (dir_match && practical && sig) {
      cand.confirmed_response = true;
      summary.candidates_confirmed_divergent++;
    } else {
      cand.confirmed_response = false;
    }
  }

  // Continuation selection: up to max_outcome_candidates
  std::vector<size_t> continuation_indices;
  for (size_t idx : selected_confirmation_indices) {
    if (summary.candidates[idx].confirmed_response) {
      continuation_indices.push_back(idx);
      if (static_cast<int>(continuation_indices.size()) >= max_outcome_candidates) break;
    }
  }
  if (static_cast<int>(continuation_indices.size()) < max_outcome_candidates) {
    for (size_t idx : selected_confirmation_indices) {
      if (std::find(continuation_indices.begin(), continuation_indices.end(), idx) == continuation_indices.end()) {
        continuation_indices.push_back(idx);
        if (static_cast<int>(continuation_indices.size()) >= max_outcome_candidates) break;
      }
    }
  }
  summary.candidates_tested_continuation = static_cast<int>(continuation_indices.size());

  // Run Full-Game Continuation on selected outcome candidates
  for (size_t idx : continuation_indices) {
    auto& cand = summary.candidates[idx];
    auto& root = summary.roots[idx];

    std::unique_ptr<State> base_state = game->NewInitialState();
    for (Action a : root.history_actions) {
      base_state->ApplyAction(a);
    }

    cand.continuation_samples = continuation_samples;
    std::vector<double> util_diffs;
    std::vector<int> vp_diffs;
    double util_sum_a = 0.0, util_sum_b = 0.0;
    double vp_sum_a = 0.0, vp_sum_b = 0.0;

    for (int m = 0; m < continuation_samples; ++m) {
      uint64_t roll_chance_seed = dune_seed::DeriveSeed(cand.game_seed, 0x00E3, cand.candidate_id, m, dune_seed::kStreamChance);
      std::array<std::mt19937_64, kNumPlayers> p_rngs_a, p_rngs_b;
      std::array<uint64_t, kNumPlayers> p_seeds;
      for (int p = 0; p < kNumPlayers; ++p) {
        p_seeds[p] = dune_seed::DeriveSeed(cand.game_seed, 0x00E3, cand.candidate_id, m, dune_seed::kStreamPolicyPlayer0 + p);
        p_rngs_a[p].seed(p_seeds[p]);
        p_rngs_b[p].seed(p_seeds[p]);
      }
      std::mt19937_64 c_rng_a(roll_chance_seed);
      std::mt19937_64 c_rng_b(roll_chance_seed);

      PairedContinuationRecord cont_rec;
      cont_rec.candidate_id = cand.candidate_id;
      cont_rec.pair_index = m;
      cont_rec.chance_seed = roll_chance_seed;
      cont_rec.policy_seeds = p_seeds;
      cont_rec.holder_player = cand.holder_player;
      cont_rec.target_opponent = cand.opponent_player;

      // Rollout Branch A
      auto sa = base_state->Clone();
      sa->ApplyAction(cand.intrigue_action);
      AdvanceThroughChanceAndAcks(sa.get(), c_rng_a);
      if (ContainsAction(sa->LegalActions(), kActionEndTurn)) {
        sa->ApplyAction(kActionEndTurn);
      }
      int steps_a = 0;
      while (!sa->IsTerminal() && steps_a < 3000) {
        steps_a++;
        AdvanceStatus st = AdvanceThroughChanceAndAcks(sa.get(), c_rng_a);
        if (st == AdvanceStatus::kSuccessTerminal) break;
        if (st != AdvanceStatus::kSuccessDecision) break;
        Player cp = sa->CurrentPlayer();
        Action a = evaluator->SelectAction(*sa, cp, /*greedy=*/false, temperature, p_rngs_a[cp]);
        sa->ApplyAction(a);
      }

      // Rollout Branch B
      auto sb = base_state->Clone();
      sb->ApplyAction(kActionEndTurn);
      int steps_b = 0;
      bool played_later = false;
      int play_round = -1, play_step = -1;
      while (!sb->IsTerminal() && steps_b < 3000) {
        steps_b++;
        AdvanceStatus st = AdvanceThroughChanceAndAcks(sb.get(), c_rng_b);
        if (st == AdvanceStatus::kSuccessTerminal) break;
        if (st != AdvanceStatus::kSuccessDecision) break;
        Player cp = sb->CurrentPlayer();
        Action a = evaluator->SelectAction(*sb, cp, /*greedy=*/false, temperature, p_rngs_b[cp]);
        if (cp == cand.holder_player && a == cand.intrigue_action) {
          played_later = true;
          play_round = dynamic_cast<DuneImperiumState*>(sb.get())->GetCurrentRound();
          play_step = steps_b;
        }
        sb->ApplyAction(a);
      }

      cont_rec.branch_a_terminal = sa->IsTerminal();
      cont_rec.branch_a_status = sa->IsTerminal() ? "TERMINAL" : "TIMEOUT";
      cont_rec.branch_b_terminal = sb->IsTerminal();
      cont_rec.branch_b_status = sb->IsTerminal() ? "TERMINAL" : "TIMEOUT";
      cont_rec.branch_b_intrigue_played_later = played_later;
      cont_rec.branch_b_intrigue_play_round = play_round;
      cont_rec.branch_b_intrigue_play_step = play_step;

      // EXPLICIT TERMINAL CHECK BEFORE READING RETURNS
      if (sa->IsTerminal() && sb->IsTerminal()) {
        cont_rec.is_valid = true;
        for (int p = 0; p < kNumPlayers; ++p) {
          cont_rec.branch_a_returns[p] = sa->Returns()[p];
          cont_rec.branch_b_returns[p] = sb->Returns()[p];
          cont_rec.branch_a_vps[p] = dynamic_cast<DuneImperiumState*>(sa.get())->GetPlayerVp(p);
          cont_rec.branch_b_vps[p] = dynamic_cast<DuneImperiumState*>(sb.get())->GetPlayerVp(p);
        }
        double u_a = cont_rec.branch_a_returns[cand.holder_player];
        double u_b = cont_rec.branch_b_returns[cand.holder_player];
        int vp_a = cont_rec.branch_a_vps[cand.holder_player];
        int vp_b = cont_rec.branch_b_vps[cand.holder_player];

        util_sum_a += u_a;
        util_sum_b += u_b;
        vp_sum_a += vp_a;
        vp_sum_b += vp_b;

        double u_diff = u_b - u_a;
        int vp_diff = vp_b - vp_a;
        cont_rec.holder_utility_diff = u_diff;
        cont_rec.holder_vp_diff = vp_diff;
        util_diffs.push_back(u_diff);
        vp_diffs.push_back(vp_diff);
      } else {
        cont_rec.is_valid = false;
        cand.has_invalidation_failure = true;
        cand.invalidation_reason = "Continuation timed out before terminal state (A=" +
            cont_rec.branch_a_status + ", B=" + cont_rec.branch_b_status + ")";
        summary.run_has_invalidation_failure = true;
        summary.invalidation_reason = cand.invalidation_reason;
      }
      summary.continuation_records.push_back(cont_rec);
    }

    if (!util_diffs.empty()) {
      double n = static_cast<double>(util_diffs.size());
      cand.mean_early_holder_utility = util_sum_a / n;
      cand.mean_hold_holder_utility = util_sum_b / n;
      cand.delta_holder_utility = cand.mean_hold_holder_utility - cand.mean_early_holder_utility;

      cand.mean_early_holder_vp = vp_sum_a / n;
      cand.mean_hold_holder_vp = vp_sum_b / n;
      cand.delta_holder_vp = cand.mean_hold_holder_vp - cand.mean_early_holder_vp;

      double var_u = 0.0;
      for (double d : util_diffs) {
        var_u += (d - cand.delta_holder_utility) * (d - cand.delta_holder_utility);
      }
      var_u = (n > 1.0) ? (var_u / (n - 1.0)) : 0.0;
      cand.std_err_utility = std::sqrt(var_u / n);
      double t = (cand.std_err_utility > 0.0) ? (std::abs(cand.delta_holder_utility) / cand.std_err_utility) : 0.0;
      cand.p_value_utility_raw = std::erfc(t / std::sqrt(2.0));
    }
  }

  // Multiple testing correction across tested outcome candidates
  std::vector<double> out_raw_p;
  for (size_t idx : continuation_indices) {
    out_raw_p.push_back(summary.candidates[idx].p_value_utility_raw);
  }
  std::vector<double> out_adj_p = ComputeHolmBonferroniAdjustedPValues(out_raw_p);
  for (size_t i = 0; i < continuation_indices.size(); ++i) {
    size_t idx = continuation_indices[i];
    auto& cand = summary.candidates[idx];
    cand.p_value_utility_adjusted = out_adj_p[i];

    // Conservative critical value for Holm-Bonferroni adjusted CI (M <= 4 -> alpha = 0.05 / 4 -> z ~ 2.50)
    double z_crit = 2.4977;
    cand.utility_ci_lower_adjusted = cand.delta_holder_utility - z_crit * cand.std_err_utility;
    cand.utility_ci_upper_adjusted = cand.delta_holder_utility + z_crit * cand.std_err_utility;

    // Decision rule: A positive average alone is strictly reported as Inconclusive
    if (cand.has_invalidation_failure) {
      cand.verdict = "Invalid";
    } else if (cand.delta_holder_utility > 0.0 && cand.p_value_utility_adjusted < 0.05 && cand.utility_ci_lower_adjusted > 0.0) {
      cand.verdict = "Supported";
    } else if (cand.delta_holder_utility < 0.0 && cand.p_value_utility_adjusted < 0.05 && cand.utility_ci_upper_adjusted < 0.0) {
      cand.verdict = "Contradicted";
    } else {
      cand.verdict = "Inconclusive";
    }
  }

  return summary;
}

inline IntrigueMiningSummary RunIntrigueMiningSuite(
    IPolicyDiagnosticsEvaluator* evaluator,
    int num_games = 100,
    uint64_t base_seed = 20260915,
    int confirmation_samples = 64,
    float temperature = 1.0f) {
  return RunOrdinaryGameIntrigueMining(
      evaluator, num_games, base_seed, /*discovery_pairs=*/4,
      /*max_candidates=*/8, confirmation_samples,
      /*max_outcome_candidates=*/4, /*continuation_samples=*/64,
      temperature);
}

} // namespace dune_diagnostics
} // namespace open_spiel

#endif // OPEN_SPIEL_EXAMPLES_DUNE_FROZEN_POLICY_DIAG_MINING_H_
