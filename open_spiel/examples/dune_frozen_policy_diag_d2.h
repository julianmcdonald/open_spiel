#ifndef OPEN_SPIEL_EXAMPLES_DUNE_FROZEN_POLICY_DIAG_D2_H_
#define OPEN_SPIEL_EXAMPLES_DUNE_FROZEN_POLICY_DIAG_D2_H_

#include "dune_frozen_policy_diag_d1.h"

namespace open_spiel {
namespace dune_diagnostics {

// ===========================================================================
// Diagnostic 2: Early-Play Preference vs Repeated Sampling
// ===========================================================================

inline std::pair<std::vector<Diagnostic2OpportunityRecord>, Diagnostic2Summary>
RunDiagnostic2Suite(
    const std::vector<ScenarioConfig>& bank,
    IPolicyDiagnosticsEvaluator* evaluator,
    int replicates_per_scenario = 16,
    uint64_t base_seed = 20260915,
    float temperature = 1.0f) {

  std::vector<Diagnostic2OpportunityRecord> records;
  Diagnostic2Summary summary;

  int rank1_count = 0;
  double total_prob = 0.0;
  double total_rank = 0.0;
  int total_decisions = 0;
  int total_traces_sampled = 0;
  int total_traces_greedy = 0;

  // Track opportunities per trace
  std::map<int, int> opp_seen_sampled;
  std::map<int, int> opp_played_sampled;
  std::map<int, int> opp_seen_greedy;
  std::map<int, int> opp_played_greedy;

  for (const auto& cfg : bank) {
    for (int r = 0; r < replicates_per_scenario; ++r) {
      uint64_t chance_seed = dune_seed::DeriveSeed(
          base_seed, dune_seed::kDomainEvalBaseline, cfg.scenario_id, r,
          dune_seed::kStreamChance);
      uint64_t policy_seed = dune_seed::DeriveSeed(
          base_seed, dune_seed::kDomainEvalBaseline, cfg.scenario_id, r,
          dune_seed::kStreamPolicyPlayer0 + cfg.p_holder);

      auto run_trace = [&](bool is_greedy) {
        auto base_state = ScenarioGenerator::BuildScenarioState(cfg, nullptr);
        auto* s = base_state.get();
        std::mt19937_64 chance_rng(chance_seed);
        std::mt19937_64 policy_rng(policy_seed);

        int opp_idx = 0;
        int agent_turn_index = 1;
        int decision_in_turn = 0;
        bool played_initial_turn = false;
        bool initial_turn_ended = false;
        std::vector<std::string> initial_turn_actions;
        bool windfall_played = false;
        int steps = 0;

        while (!s->IsTerminal() && steps < 100 && !windfall_played) {
          steps++;
          AdvanceStatus st = AdvanceThroughChanceAndAcks(s, chance_rng);
          if (st != AdvanceStatus::kSuccessDecision) break;

          Player cp = s->CurrentPlayer();
          auto legal = s->LegalActions();
          if (legal.empty()) break;

          if (cp == cfg.p_holder && ContainsAction(legal, kActionPlayWindfall)) {
            // Eligible P2 decision where Windfall is legal
            Diagnostic2OpportunityRecord rec;
            rec.scenario_id = cfg.scenario_id;
            rec.replicate_idx = r;
            rec.player = cp;
            rec.round = s->GetCurrentRound();
            rec.agent_turn_index = agent_turn_index;
            rec.decision_in_turn = decision_in_turn;
            rec.decision_index = opp_idx;
            rec.opportunity_index = opp_idx;
            rec.is_initial_turn = (agent_turn_index == 1);
            rec.legal_actions_count = static_cast<int>(legal.size());
            rec.was_greedy = is_greedy;

            // Compute exact policy probabilities
            auto policy = evaluator->GetPolicy(*s, cp);
            double w_prob = 0.0;
            double max_p = -1.0;
            Action argmax_a = open_spiel::kInvalidAction;
            for (const auto& [a, p] : policy) {
              if (a == kActionPlayWindfall) w_prob = p;
              if (p > max_p) {
                max_p = p;
                argmax_a = a;
              }
            }
            rec.windfall_prob = w_prob;
            rec.argmax_action = argmax_a;
            rec.argmax_action_name = DiagnosticActionToString(argmax_a);
            rec.argmax_prob = max_p;

            // Rank
            int rank = 1;
            for (const auto& [a, p] : policy) {
              if (p > w_prob) rank++;
            }
            rec.windfall_rank = rank;

            // Action selection
            Action chosen = evaluator->SelectAction(*s, cp, is_greedy, temperature, policy_rng);
            rec.selected_action = chosen;
            rec.selected_action_name = DiagnosticActionToString(chosen);
            rec.selected_windfall = (chosen == kActionPlayWindfall);

            records.push_back(rec);

            if (!is_greedy) {
              total_decisions++;
              total_prob += w_prob;
              total_rank += rank;
              if (rank == 1) rank1_count++;
              summary.argmax_action_distribution[rec.argmax_action_name]++;
              opp_seen_sampled[opp_idx]++;
              if (rec.selected_windfall) opp_played_sampled[opp_idx]++;
            } else {
              opp_seen_greedy[opp_idx]++;
              if (rec.selected_windfall) opp_played_greedy[opp_idx]++;
            }

            if (agent_turn_index == 1) {
              initial_turn_actions.push_back(rec.selected_action_name);
            }

            if (chosen == kActionPlayWindfall) {
              windfall_played = true;
              if (agent_turn_index == 1) {
                played_initial_turn = true;
              }
            }

            if (chosen == kActionEndTurn) {
              if (agent_turn_index == 1) initial_turn_ended = true;
              agent_turn_index++;
              decision_in_turn = 0;
            } else {
              decision_in_turn++;
            }

            opp_idx++;
            s->ApplyAction(chosen);
            AdvanceThroughChanceAndAcks(s, chance_rng);
          } else {
            // Other decisions
            Action a = evaluator->SelectAction(*s, cp, is_greedy, temperature, policy_rng);
            if (cp == cfg.p_holder) {
              if (agent_turn_index == 1) {
                initial_turn_actions.push_back(DiagnosticActionToString(a));
              }
              if (a == kActionEndTurn) {
                if (agent_turn_index == 1) initial_turn_ended = true;
                agent_turn_index++;
                decision_in_turn = 0;
              } else {
                decision_in_turn++;
              }
            }
            s->ApplyAction(a);
            AdvanceThroughChanceAndAcks(s, chance_rng);
          }
        }

        // Trace level recording
        bool held_past_turn1 = initial_turn_ended && !played_initial_turn;
        std::string seq_str;
        for (size_t i = 0; i < initial_turn_actions.size(); ++i) {
          if (i > 0) seq_str += " -> ";
          seq_str += initial_turn_actions[i];
        }
        if (seq_str.empty()) seq_str = "<none>";

        if (is_greedy) {
          total_traces_greedy++;
          if (played_initial_turn) summary.traces_played_initial_turn_greedy++;
          if (held_past_turn1) summary.traces_held_past_turn1_greedy++;
          summary.initial_turn_action_sequences_greedy[seq_str]++;
        } else {
          total_traces_sampled++;
          if (played_initial_turn) summary.traces_played_initial_turn_sampled++;
          if (held_past_turn1) summary.traces_held_past_turn1_sampled++;
          summary.initial_turn_action_sequences_sampled[seq_str]++;
        }
      };

      // Run sampled trace
      run_trace(/*is_greedy=*/false);
      // Run greedy trace (only on replicate 0 to avoid duplicates)
      if (r == 0) {
        run_trace(/*is_greedy=*/true);
      }
    }
  }

  summary.total_traces_evaluated = total_traces_sampled;
  summary.total_decisions_evaluated = total_decisions;
  summary.windfall_rank1_count = rank1_count;
  summary.windfall_rank1_rate = (total_decisions > 0) ? (static_cast<double>(rank1_count) / total_decisions) : 0.0;
  summary.mean_windfall_prob = (total_decisions > 0) ? (total_prob / total_decisions) : 0.0;
  summary.mean_windfall_rank = (total_decisions > 0) ? (total_rank / total_decisions) : 0.0;

  // Rate calculations: separating conditional rate from cumulative rate
  int cum_sampled = 0;
  for (const auto& [opp, seen] : opp_seen_sampled) {
    int played = opp_played_sampled[opp];
    cum_sampled += played;
    summary.conditional_play_rate_sampled[opp] = (seen > 0) ? (static_cast<double>(played) / seen) : 0.0;
    summary.cumulative_play_rate_sampled[opp] = (total_traces_sampled > 0) ? (static_cast<double>(cum_sampled) / total_traces_sampled) : 0.0;
  }
  int cum_greedy = 0;
  for (const auto& [opp, seen] : opp_seen_greedy) {
    int played = opp_played_greedy[opp];
    cum_greedy += played;
    summary.conditional_play_rate_greedy[opp] = (seen > 0) ? (static_cast<double>(played) / seen) : 0.0;
    summary.cumulative_play_rate_greedy[opp] = (total_traces_greedy > 0) ? (static_cast<double>(cum_greedy) / total_traces_greedy) : 0.0;
  }

  return {records, summary};
}

} // namespace dune_diagnostics
} // namespace open_spiel

#endif // OPEN_SPIEL_EXAMPLES_DUNE_FROZEN_POLICY_DIAG_D2_H_
