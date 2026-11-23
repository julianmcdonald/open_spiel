#ifndef OPEN_SPIEL_EXAMPLES_DUNE_FROZEN_POLICY_DIAG_REPORT_H_
#define OPEN_SPIEL_EXAMPLES_DUNE_FROZEN_POLICY_DIAG_REPORT_H_

#include "dune_frozen_policy_diag_types.h"

namespace open_spiel {
namespace dune_diagnostics {

// ===========================================================================
// JSON Serialization Helpers
// ===========================================================================

inline std::string FormatWindfallDecisionRecordJson(const WindfallDecisionRecord& r) {
  std::ostringstream ss;
  ss << "{\"candidate_id\": " << r.candidate_id
     << ", \"replicate_id\": " << r.replicate_id
     << ", \"round\": " << r.round
     << ", \"turn_step\": " << r.turn_step
     << ", \"player\": " << r.player
     << ", \"actual_solari\": " << r.actual_solari
     << ", \"windfall_available\": " << (r.windfall_available ? "true" : "false")
     << ", \"windfall_played\": " << (r.windfall_played ? "true" : "false")
     << ", \"plan_requires_windfall\": " << (r.plan_requires_windfall ? "true" : "false")
     << ", \"windfall_reason\": \"" << EscapeJson(r.windfall_reason) << "\""
     << ", \"action_taken\": " << r.action_taken
     << ", \"action_taken_name\": \"" << EscapeJson(r.action_taken_name) << "\""
     << ", \"secrets_exception\": " << (r.secrets_exception ? "true" : "false")
     << ", \"terminal_value_exception\": " << (r.terminal_value_exception ? "true" : "false")
     << ", \"fallback_active\": " << (r.fallback_active ? "true" : "false")
     << ", \"fallback_reason\": \"" << EscapeJson(r.fallback_reason) << "\""
     << ", \"solari_after\": " << r.solari_after
     << ", \"resource_use_followed\": \"" << EscapeJson(r.resource_use_followed) << "\""
     << ", \"proposed_chain\": [";
  for (size_t i = 0; i < r.proposed_action_chain.size(); ++i) {
    if (i > 0) ss << ", ";
    ss << "{\"action\": " << r.proposed_action_chain[i]
       << ", \"name\": \"" << EscapeJson(r.proposed_action_names[i]) << "\"}";
  }
  ss << "]}";
  return ss.str();
}

inline std::string FormatPairedContinuationRecordJson(const PairedContinuationRecord& c) {
  std::ostringstream ss;
  ss << "{"
     << "\"candidate_id\":" << c.candidate_id << ","
     << "\"pair_index\":" << c.pair_index << ","
     << "\"chance_seed\":" << c.chance_seed << ","
     << "\"policy_seeds\":[" << c.policy_seeds[0] << "," << c.policy_seeds[1] << "," << c.policy_seeds[2] << "," << c.policy_seeds[3] << "],"
     << "\"holder_player\":" << c.holder_player << ","
     << "\"target_opponent\":" << c.target_opponent << ","
     << "\"branch_a_terminal\":" << (c.branch_a_terminal ? "true" : "false") << ","
     << "\"branch_a_status\":\"" << EscapeJson(c.branch_a_status) << "\","
     << "\"branch_a_returns\":[" << c.branch_a_returns[0] << "," << c.branch_a_returns[1] << "," << c.branch_a_returns[2] << "," << c.branch_a_returns[3] << "],"
     << "\"branch_a_vps\":[" << c.branch_a_vps[0] << "," << c.branch_a_vps[1] << "," << c.branch_a_vps[2] << "," << c.branch_a_vps[3] << "],"
     << "\"branch_a_windfall_played\":" << (c.branch_a_windfall_played ? "true" : "false") << ","
     << "\"branch_a_windfall_play_round\":" << c.branch_a_windfall_play_round << ","
     << "\"branch_a_windfall_play_step\":" << c.branch_a_windfall_play_step << ","
     << "\"branch_b_terminal\":" << (c.branch_b_terminal ? "true" : "false") << ","
     << "\"branch_b_status\":\"" << EscapeJson(c.branch_b_status) << "\","
     << "\"branch_b_returns\":[" << c.branch_b_returns[0] << "," << c.branch_b_returns[1] << "," << c.branch_b_returns[2] << "," << c.branch_b_returns[3] << "],"
     << "\"branch_b_vps\":[" << c.branch_b_vps[0] << "," << c.branch_b_vps[1] << "," << c.branch_b_vps[2] << "," << c.branch_b_vps[3] << "],"
     << "\"branch_b_intrigue_played_later\":" << (c.branch_b_intrigue_played_later ? "true" : "false") << ","
     << "\"branch_b_intrigue_play_round\":" << c.branch_b_intrigue_play_round << ","
     << "\"branch_b_intrigue_play_step\":" << c.branch_b_intrigue_play_step << ","
     << "\"branch_b_windfall_played\":" << (c.branch_b_windfall_played ? "true" : "false") << ","
     << "\"branch_b_windfall_play_round\":" << c.branch_b_windfall_play_round << ","
     << "\"branch_b_windfall_play_step\":" << c.branch_b_windfall_play_step << ","
     << "\"holder_utility_diff\":" << c.holder_utility_diff << ","
     << "\"holder_vp_diff\":" << c.holder_vp_diff << ","
     << "\"is_valid\":" << (c.is_valid ? "true" : "false")
     << "}";
  return ss.str();
}

inline std::string FormatWindfallBankRecordJson(const WindfallBankRecord& r) {
  std::ostringstream ss;
  ss << "{\"candidate_id\": " << r.candidate_id
     << ", \"game_id\": " << r.game_id
     << ", \"game_seed\": " << r.game_seed
     << ", \"round\": " << r.round
     << ", \"step_index\": " << r.step_index
     << ", \"holder_player\": " << r.holder_player
     << ", \"holder_leader\": \"" << EscapeJson(r.holder_leader) << "\""
     << ", \"holder_solari\": " << r.holder_solari
     << ", \"holder_water\": " << r.holder_water
     << ", \"holder_spice\": " << r.holder_spice
     << ", \"target_opponent\": " << r.target_opponent
     << ", \"opponent_leader\": \"" << EscapeJson(r.opponent_leader) << "\""
     << ", \"opponent_solari\": " << r.opponent_solari
     << ", \"intrigue_action\": " << r.intrigue_action
     << ", \"intrigue_name\": \"" << EscapeJson(r.intrigue_name) << "\""
     << ", \"history_len\": " << r.history_actions.size()
     << ", \"replay_verified\": " << (r.replay_verified ? "true" : "false")
     << ", \"replay_failure_reason\": \"" << EscapeJson(r.replay_failure_reason) << "\""
     << ", \"is_selected\": " << (r.is_selected ? "true" : "false")
     << ", \"selection_reason\": \"" << EscapeJson(r.selection_reason) << "\""
     << ", \"history_actions\": [";
  for (size_t i = 0; i < r.history_actions.size(); ++i) {
    if (i > 0) ss << ", ";
    ss << r.history_actions[i];
  }
  ss << "]}";
  return ss.str();
}

inline std::string FormatWindfallEvaluationReportJson(const WindfallEvaluationSummary& s) {
  std::ostringstream ss;
  ss << "{\n"
     << "  \"candidate_model\": \"" << EscapeJson(s.candidate_model) << "\",\n"
     << "  \"candidate_sha256\": \"" << EscapeJson(s.candidate_sha256) << "\",\n"
     << "  \"opponent_model\": \"" << EscapeJson(s.opponent_model) << "\",\n"
     << "  \"opponent_sha256\": \"" << EscapeJson(s.opponent_sha256) << "\",\n"
     << "  \"base_seed\": " << s.base_seed << ",\n"
     << "  \"temperature\": " << s.temperature << ",\n"
     << "  \"total_natural_roots_reconstructed\": " << s.total_natural_roots_reconstructed << ",\n"
     << "  \"total_natural_roots_verified\": " << s.total_natural_roots_verified << ",\n"
     << "  \"positions_selected\": " << s.positions_selected << ",\n"
     << "  \"continuations_per_position\": " << s.continuations_per_position << ",\n"
     << "  \"total_continuation_pairs\": " << s.total_continuation_pairs << ",\n"
     << "  \"run_has_invalidation_failure\": " << (s.run_has_invalidation_failure ? "true" : "false") << ",\n"
     << "  \"invalidation_reason\": \"" << EscapeJson(s.invalidation_reason) << "\",\n"
     << "  \"primary_metric\": {\n"
     << "    \"mean_placement_utility_diff\": " << s.mean_placement_utility_diff << ",\n"
     << "    \"placement_utility_std_err\": " << s.placement_utility_std_err << ",\n"
     << "    \"placement_utility_ci_lower\": " << s.placement_utility_ci_lower << ",\n"
     << "    \"placement_utility_ci_upper\": " << s.placement_utility_ci_upper << ",\n"
     << "    \"mean_holder_vp_diff\": " << s.mean_holder_vp_diff << "\n"
     << "  },\n"
     << "  \"secondary_metrics\": {\n"
     << "    \"total_controller_decisions\": " << s.total_controller_decisions << ",\n"
     << "    \"early_plays_prevented\": " << s.early_plays_prevented << ",\n"
     << "    \"windfall_plays_executed\": " << s.windfall_plays_executed << ",\n"
     << "    \"necessary_spending_unlocked\": " << s.necessary_spending_unlocked << ",\n"
     << "    \"purchases_already_affordable\": " << s.purchases_already_affordable << ",\n"
     << "    \"end_turn_retentions\": " << s.end_turn_retentions << ",\n"
     << "    \"secrets_exceptions_triggered\": " << s.secrets_exceptions_triggered << ",\n"
     << "    \"terminal_value_exceptions_triggered\": " << s.terminal_value_exceptions_triggered << ",\n"
     << "    \"fallbacks_triggered\": " << s.fallbacks_triggered << "\n"
     << "  },\n"
     << "  \"verdict\": \"" << EscapeJson(s.verdict) << "\",\n"
     << "  \"verdict_explanation\": \"" << EscapeJson(s.verdict_explanation) << "\"\n"
     << "}\n";
  return ss.str();
}

inline std::string FormatDiagnostic2ReportJson(
    const Diagnostic2Summary& sum,
    const std::string& model_name,
    const std::string& model_sha256) {

  std::ostringstream ss;
  ss << "{\n";
  ss << "  \"model_name\": \"" << EscapeJson(model_name) << "\",\n";
  ss << "  \"model_sha256\": \"" << EscapeJson(model_sha256) << "\",\n";
  ss << "  \"total_traces_evaluated\": " << sum.total_traces_evaluated << ",\n";
  ss << "  \"total_decisions_evaluated\": " << sum.total_decisions_evaluated << ",\n";
  ss << "  \"windfall_rank1_count\": " << sum.windfall_rank1_count << ",\n";
  ss << "  \"windfall_rank1_rate\": " << std::fixed << std::setprecision(4) << sum.windfall_rank1_rate << ",\n";
  ss << "  \"mean_windfall_prob\": " << sum.mean_windfall_prob << ",\n";
  ss << "  \"mean_windfall_rank\": " << sum.mean_windfall_rank << ",\n";
  ss << "  \"traces_played_initial_turn_sampled\": " << sum.traces_played_initial_turn_sampled << ",\n";
  ss << "  \"traces_held_past_turn1_sampled\": " << sum.traces_held_past_turn1_sampled << ",\n";
  ss << "  \"traces_played_initial_turn_greedy\": " << sum.traces_played_initial_turn_greedy << ",\n";
  ss << "  \"traces_held_past_turn1_greedy\": " << sum.traces_held_past_turn1_greedy << ",\n";

  ss << "  \"conditional_play_rate_sampled\": {";
  bool first = true;
  for (const auto& [opp, r] : sum.conditional_play_rate_sampled) {
    if (!first) ss << ", ";
    first = false;
    ss << "\"" << opp << "\": " << r;
  }
  ss << "},\n";

  ss << "  \"cumulative_play_rate_sampled\": {";
  first = true;
  for (const auto& [opp, r] : sum.cumulative_play_rate_sampled) {
    if (!first) ss << ", ";
    first = false;
    ss << "\"" << opp << "\": " << r;
  }
  ss << "},\n";

  ss << "  \"conditional_play_rate_greedy\": {";
  first = true;
  for (const auto& [opp, r] : sum.conditional_play_rate_greedy) {
    if (!first) ss << ", ";
    first = false;
    ss << "\"" << opp << "\": " << r;
  }
  ss << "},\n";

  ss << "  \"cumulative_play_rate_greedy\": {";
  first = true;
  for (const auto& [opp, r] : sum.cumulative_play_rate_greedy) {
    if (!first) ss << ", ";
    first = false;
    ss << "\"" << opp << "\": " << r;
  }
  ss << "},\n";

  ss << "  \"initial_turn_action_sequences_greedy\": {\n";
  first = true;
  for (const auto& [seq, cnt] : sum.initial_turn_action_sequences_greedy) {
    if (!first) ss << ",\n";
    first = false;
    ss << "    \"" << EscapeJson(seq) << "\": " << cnt;
  }
  ss << "\n  }\n";
  ss << "}\n";
  return ss.str();
}

inline std::string FormatIntrigueMiningReportJson(
    const IntrigueMiningSummary& sum,
    const std::string& model_name,
    const std::string& model_sha256) {

  std::ostringstream ss;
  ss << "{\n";
  ss << "  \"model_name\": \"" << EscapeJson(model_name) << "\",\n";
  ss << "  \"model_sha256\": \"" << EscapeJson(model_sha256) << "\",\n";
  ss << "  \"games_simulated\": " << sum.games_simulated << ",\n";
  ss << "  \"total_resource_intrigue_plays\": " << sum.total_resource_intrigue_plays << ",\n";
  ss << "  \"windfall_plays\": " << sum.windfall_plays << ",\n";
  ss << "  \"water_peddlers_union_plays\": " << sum.water_peddlers_union_plays << ",\n";
  ss << "  \"candidates_discovered\": " << sum.candidates_discovered << ",\n";
  ss << "  \"candidates_tested_confirmation\": " << sum.candidates_tested_confirmation << ",\n";
  ss << "  \"candidates_confirmed_divergent\": " << sum.candidates_confirmed_divergent << ",\n";
  ss << "  \"candidates_tested_continuation\": " << sum.candidates_tested_continuation << ",\n";
  ss << "  \"run_has_invalidation_failure\": " << (sum.run_has_invalidation_failure ? "true" : "false") << ",\n";
  ss << "  \"invalidation_reason\": \"" << EscapeJson(sum.invalidation_reason) << "\",\n";
  ss << "  \"candidates\": [\n";

  for (size_t i = 0; i < sum.candidates.size(); ++i) {
    const auto& c = sum.candidates[i];
    ss << "    {\n"
       << "      \"candidate_id\": " << c.candidate_id << ",\n"
       << "      \"game_id\": " << c.game_id << ",\n"
       << "      \"round\": " << c.round << ",\n"
       << "      \"step_index\": " << c.step_index << ",\n"
       << "      \"intrigue\": \"" << EscapeJson(c.intrigue_name) << "\",\n"
       << "      \"holder\": " << c.holder_player << ",\n"
       << "      \"holder_leader\": \"" << EscapeJson(c.holder_leader) << "\",\n"
       << "      \"holder_solari\": " << c.holder_solari << ",\n"
       << "      \"holder_water\": " << c.holder_water << ",\n"
       << "      \"opponent\": " << c.opponent_player << ",\n"
       << "      \"opponent_leader\": \"" << EscapeJson(c.opponent_leader) << "\",\n"
       << "      \"opponent_solari\": " << c.opponent_solari << ",\n"
       << "      \"opponent_water\": " << c.opponent_water << ",\n"
       << "      \"replay_verified\": " << (c.replay_verified ? "true" : "false") << ",\n"
       << "      \"replay_failure_reason\": \"" << EscapeJson(c.replay_failure_reason) << "\",\n"
       << "      \"discovery_divergent_count\": " << c.discovery_divergent_count << ",\n"
       << "      \"target_action\": \"" << EscapeJson(c.target_action_name) << "\",\n"
       << "      \"predicted_direction\": " << c.predicted_direction << ",\n"
       << "      \"discovery_delta\": " << c.discovery_delta << ",\n"
       << "      \"discovery_early_action\": \"" << EscapeJson(c.discovery_opp_action_early_name) << "\",\n"
       << "      \"discovery_hold_action\": \"" << EscapeJson(c.discovery_opp_action_hold_name) << "\",\n"
       << "      \"confirmation_samples\": " << c.confirmation_samples << ",\n"
       << "      \"confirmed_divergent_count\": " << c.confirmed_divergent_count << ",\n"
       << "      \"confirmed_divergent_rate\": " << std::fixed << std::setprecision(4) << c.confirmed_divergent_rate << ",\n"
       << "      \"delta_prob\": " << c.delta_prob << ",\n"
       << "      \"std_err_prob\": " << c.std_err_prob << ",\n"
       << "      \"p_value_raw\": " << c.p_value_raw << ",\n"
       << "      \"p_value_adjusted\": " << c.p_value_adjusted << ",\n"
       << "      \"ci_prob_95\": [" << c.ci_prob_lower << ", " << c.ci_prob_upper << "],\n"
       << "      \"confirmed_response\": " << (c.confirmed_response ? "true" : "false") << ",\n"
       << "      \"opponent_argmax_early\": \"" << EscapeJson(c.opponent_argmax_early_name) << "\",\n"
       << "      \"opponent_argmax_hold\": \"" << EscapeJson(c.opponent_argmax_hold_name) << "\",\n"
       << "      \"opponent_target_prob_early\": " << c.opponent_target_prob_early << ",\n"
       << "      \"opponent_target_prob_hold\": " << c.opponent_target_prob_hold << ",\n"
       << "      \"continuation_samples\": " << c.continuation_samples << ",\n"
       << "      \"has_invalidation_failure\": " << (c.has_invalidation_failure ? "true" : "false") << ",\n"
       << "      \"invalidation_reason\": \"" << EscapeJson(c.invalidation_reason) << "\",\n"
       << "      \"mean_early_holder_utility\": " << c.mean_early_holder_utility << ",\n"
       << "      \"mean_hold_holder_utility\": " << c.mean_hold_holder_utility << ",\n"
       << "      \"delta_holder_utility\": " << c.delta_holder_utility << ",\n"
       << "      \"std_err_utility\": " << c.std_err_utility << ",\n"
       << "      \"p_value_utility_raw\": " << c.p_value_utility_raw << ",\n"
       << "      \"p_value_utility_adjusted\": " << c.p_value_utility_adjusted << ",\n"
       << "      \"utility_ci_adjusted\": [" << c.utility_ci_lower_adjusted << ", " << c.utility_ci_upper_adjusted << "],\n"
       << "      \"mean_early_holder_vp\": " << c.mean_early_holder_vp << ",\n"
       << "      \"mean_hold_holder_vp\": " << c.mean_hold_holder_vp << ",\n"
       << "      \"delta_holder_vp\": " << c.delta_holder_vp << ",\n"
       << "      \"verdict\": \"" << EscapeJson(c.verdict) << "\"\n"
       << "    }";
    if (i + 1 < sum.candidates.size()) ss << ",\n";
    else ss << "\n";
  }
  ss << "  ]\n";
  ss << "}\n";
  return ss.str();
}

inline std::string FormatDiagnostic1ReportJson(
    const Diagnostic1Aggregate& agg,
    const std::vector<ScenarioSummary>& summaries,
    const std::string& model_name,
    const std::string& model_sha256) {

  std::ostringstream ss;
  ss << "{\n";
  ss << "  \"model_name\": \"" << model_name << "\",\n";
  ss << "  \"model_sha256\": \"" << model_sha256 << "\",\n";
  ss << "  \"total_scenarios\": " << agg.total_scenarios << ",\n";
  ss << "  \"replicates_per_scenario\": " << agg.replicates_per_scenario << ",\n";
  ss << "  \"total_paired_trials\": " << agg.total_paired_trials << ",\n";
  ss << "  \"branch_a_swordmaster_freq\": " << std::fixed << std::setprecision(4) << agg.mean_branch_a_freq << ",\n";
  ss << "  \"branch_b_swordmaster_freq\": " << agg.mean_branch_b_freq << ",\n";
  ss << "  \"mean_effect\": " << agg.effect_ci.mean << ",\n";
  ss << "  \"effect_ci_95\": [" << agg.effect_ci.lower << ", " << agg.effect_ci.upper << "],\n";
  ss << "  \"saturated_always_count\": " << agg.saturated_always_count << ",\n";
  ss << "  \"saturated_never_count\": " << agg.saturated_never_count << ",\n";
  ss << "  \"sensitive_count\": " << agg.sensitive_count << ",\n";
  ss << "  \"greedy_branch_a_freq\": " << agg.greedy_branch_a_freq << ",\n";
  ss << "  \"greedy_branch_b_freq\": " << agg.greedy_branch_b_freq << ",\n";
  ss << "  \"greedy_mean_effect\": " << agg.greedy_mean_effect << ",\n";

  ss << "  \"family_breakdown\": {\n";
  bool first_fam = true;
  for (const auto& [fam, ci] : agg.family_effects) {
    if (!first_fam) ss << ",\n";
    first_fam = false;
    ss << "    \"" << fam << "\": {\"mean\": " << ci.mean << ", \"ci_lower\": "
       << ci.lower << ", \"ci_upper\": " << ci.upper << "}";
  }
  ss << "\n  },\n";

  ss << "  \"alternative_spaces_branch_b\": {\n";
  bool first_sp = true;
  for (const auto& [sp, cnt] : agg.alternative_spaces_branch_b) {
    if (!first_sp) ss << ",\n";
    first_sp = false;
    ss << "    \"" << sp << "\": " << cnt;
  }
  ss << "\n  }\n";
  ss << "}\n";
  return ss.str();
}

} // namespace dune_diagnostics
} // namespace open_spiel

#endif // OPEN_SPIEL_EXAMPLES_DUNE_FROZEN_POLICY_DIAG_REPORT_H_
