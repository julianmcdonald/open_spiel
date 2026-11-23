// Copyright 2026
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "open_spiel/python/pybind11/games_dune_imperium_internal.h"

#ifdef OPEN_SPIEL_BUILD_WITH_LIBTORCH

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "dune_evaluator.h"
#include "dune_puct_is_mcts.h"
#include "dune_search_routing.h"
#include "dune_search_session.h"
#include "open_spiel/spiel.h"

namespace py = ::pybind11;

namespace open_spiel {
namespace dune_imperium_pybind {

#pragma GCC optimize("no-var-tracking-assignments")
void RegisterDuneImperiumSearch(py::module_ &di) {
  py::enum_<open_spiel::SearchOpponentMode>(di, "SearchOpponentMode")
      .value("MAX_N", open_spiel::SearchOpponentMode::kMaxN)
      .value("POLICY", open_spiel::SearchOpponentMode::kPolicy)
      .export_values();

  py::enum_<open_spiel::DuneISMCTSFinalPolicyType>(di, "DuneISMCTSFinalPolicyType")
      .value("NORMALIZED_VISIT_COUNT", open_spiel::DuneISMCTSFinalPolicyType::kNormalizedVisitCount)
      .value("MAX_VISIT_COUNT", open_spiel::DuneISMCTSFinalPolicyType::kMaxVisitCount)
      .value("MAX_VALUE", open_spiel::DuneISMCTSFinalPolicyType::kMaxValue)
      .export_values();

  py::class_<open_spiel::DuneSearchConfig>(di, "DuneSearchConfig")
      .def(py::init<>())
      .def_readwrite("max_simulations", &open_spiel::DuneSearchConfig::max_simulations)
      .def_readwrite("relative_time_budget_ms", &open_spiel::DuneSearchConfig::relative_time_budget_ms)
      .def_readwrite("max_nodes", &open_spiel::DuneSearchConfig::max_nodes)
      .def_readwrite("puct_c", &open_spiel::DuneSearchConfig::puct_c)
      .def_readwrite("opponent_mode", &open_spiel::DuneSearchConfig::opponent_mode)
      .def_readwrite("temperature", &open_spiel::DuneSearchConfig::temperature)
      .def_readwrite("opponent_temperature", &open_spiel::DuneSearchConfig::opponent_temperature)
      .def_readwrite("max_world_samples", &open_spiel::DuneSearchConfig::max_world_samples)
      .def_readwrite("utility_divisor", &open_spiel::DuneSearchConfig::utility_divisor)
      .def_readwrite("min_visit_threshold", &open_spiel::DuneSearchConfig::min_visit_threshold)
      .def_readwrite("covered_prior_threshold", &open_spiel::DuneSearchConfig::covered_prior_threshold)
      .def_readwrite("seed", &open_spiel::DuneSearchConfig::seed)
      .def_readwrite("final_policy_type", &open_spiel::DuneSearchConfig::final_policy_type)
      .def_readwrite("dirichlet_epsilon", &open_spiel::DuneSearchConfig::dirichlet_epsilon)
      .def_readwrite("dirichlet_alpha", &open_spiel::DuneSearchConfig::dirichlet_alpha)
      .def_readwrite("use_observation_string", &open_spiel::DuneSearchConfig::use_observation_string)
      .def_readwrite("verbose_diagnostics", &open_spiel::DuneSearchConfig::verbose_diagnostics)
      .def_readwrite("check_strategic_state", &open_spiel::DuneSearchConfig::check_strategic_state)
      // Leader-selection search (adopted 2026-08-16 at a fixed 64-simulation
      // budget). All three are required together: enabling the search without
      // mass-only coverage makes the search-result path reject the concentrated
      // Leader policy as low_coverage and hand back the raw prior.
      .def_readwrite("search_leader_draft", &open_spiel::DuneSearchConfig::search_leader_draft)
      .def_readwrite("leader_draft_simulations", &open_spiel::DuneSearchConfig::leader_draft_simulations)
      .def_readwrite("leader_mass_only_coverage", &open_spiel::DuneSearchConfig::leader_mass_only_coverage)
      .def_readwrite("fixed_continuation_reserve", &open_spiel::DuneSearchConfig::fixed_continuation_reserve)
      .def_readwrite("purchase_combat_budget", &open_spiel::DuneSearchConfig::purchase_combat_budget)
      .def_readwrite("live_continuation_reserve_seconds", &open_spiel::DuneSearchConfig::live_continuation_reserve_seconds)
      .def_readwrite("fixed_session_limit", &open_spiel::DuneSearchConfig::fixed_session_limit)
      .def_readwrite("model_checkpoint_path", &open_spiel::DuneSearchConfig::model_checkpoint_path)
      .def_readwrite("root_prior_temperature", &open_spiel::DuneSearchConfig::root_prior_temperature)
      .def_readwrite("training_root_prior_temperature", &open_spiel::DuneSearchConfig::training_root_prior_temperature)
      .def_readwrite("conservative_override_enabled", &open_spiel::DuneSearchConfig::conservative_override_enabled)
      .def_readwrite("conservative_covered_prior_threshold", &open_spiel::DuneSearchConfig::conservative_covered_prior_threshold)
      .def_readwrite("conservative_meaningful_visit_threshold", &open_spiel::DuneSearchConfig::conservative_meaningful_visit_threshold)
      .def_readwrite("conservative_q_margin_threshold", &open_spiel::DuneSearchConfig::conservative_q_margin_threshold)
      .def_readwrite("conservative_stability_checkpoint_fraction", &open_spiel::DuneSearchConfig::conservative_stability_checkpoint_fraction)
      .def_readwrite("conservative_continuation_overrides_disabled", &open_spiel::DuneSearchConfig::conservative_continuation_overrides_disabled)
      .def_readwrite("max_search_decision_depth", &open_spiel::DuneSearchConfig::max_search_decision_depth);

  py::class_<open_spiel::SearchDiagnostics>(di, "SearchDiagnostics")
      .def_readonly("actions", &open_spiel::SearchDiagnostics::actions)
      .def_readonly("visit_counts", &open_spiel::SearchDiagnostics::visit_counts)
      .def_readonly("q_values", &open_spiel::SearchDiagnostics::q_values)
      .def_readonly("priors", &open_spiel::SearchDiagnostics::priors)
      .def_readonly("root_value", &open_spiel::SearchDiagnostics::root_value)
      .def_readonly("total_root_visits", &open_spiel::SearchDiagnostics::total_root_visits)
      .def_readonly("num_covered_actions", &open_spiel::SearchDiagnostics::num_covered_actions)
      .def_readonly("covered_prior_mass", &open_spiel::SearchDiagnostics::covered_prior_mass)
      .def_readonly("max_decision_depth", &open_spiel::SearchDiagnostics::max_decision_depth)
      .def_readonly("mean_decision_depth", &open_spiel::SearchDiagnostics::mean_decision_depth)
      .def_readonly("protocol_version", &open_spiel::SearchDiagnostics::protocol_version)
      .def_readonly("session_id", &open_spiel::SearchDiagnostics::session_id)
      .def_readonly("searched_seat", &open_spiel::SearchDiagnostics::searched_seat)
      .def_readonly("round", &open_spiel::SearchDiagnostics::round)
      .def_readonly("phase", &open_spiel::SearchDiagnostics::phase)
      .def_readonly("decision_role", &open_spiel::SearchDiagnostics::decision_role)
      .def_readonly("budget_mode", &open_spiel::SearchDiagnostics::budget_mode)
      .def_readonly("hard_sim_limit", &open_spiel::SearchDiagnostics::hard_sim_limit)
      .def_readonly("soft_sim_limit", &open_spiel::SearchDiagnostics::soft_sim_limit)
      .def_readonly("hard_time_limit_ms", &open_spiel::SearchDiagnostics::hard_time_limit_ms)
      .def_readonly("soft_time_limit_ms", &open_spiel::SearchDiagnostics::soft_time_limit_ms)
      .def_readonly("elapsed_search_time_ms", &open_spiel::SearchDiagnostics::elapsed_search_time_ms)
      .def_readonly("observation_wait_time_ms", &open_spiel::SearchDiagnostics::observation_wait_time_ms)
      .def_readonly("inherited_root_visits", &open_spiel::SearchDiagnostics::inherited_root_visits)
      .def_readonly("newly_completed_simulations", &open_spiel::SearchDiagnostics::newly_completed_simulations)
      .def_readonly("session_cumulative_simulations", &open_spiel::SearchDiagnostics::session_cumulative_simulations)
      .def_readonly("short_window_cumulative_simulations", &open_spiel::SearchDiagnostics::short_window_cumulative_simulations)
      .def_readonly("session_cumulative_search_time_ms", &open_spiel::SearchDiagnostics::session_cumulative_search_time_ms)
      .def_readonly("long_agent_session_cumulative_time_ms", &open_spiel::SearchDiagnostics::long_agent_session_cumulative_time_ms)
      .def_readonly("re_root_status", &open_spiel::SearchDiagnostics::re_root_status)
      .def_readonly("post_chance_branch_miss", &open_spiel::SearchDiagnostics::post_chance_branch_miss)
      .def_readonly("root_coverage", &open_spiel::SearchDiagnostics::root_coverage)
      .def_readonly("reset_reason", &open_spiel::SearchDiagnostics::reset_reason)
      .def_readonly("tree_node_count", &open_spiel::SearchDiagnostics::tree_node_count)
      .def_readonly("selected_action", &open_spiel::SearchDiagnostics::selected_action)
      .def_readonly("legality_result", &open_spiel::SearchDiagnostics::legality_result)
      .def_readonly("fallback_reason", &open_spiel::SearchDiagnostics::fallback_reason)
      .def_readonly("raw_reference_action", &open_spiel::SearchDiagnostics::raw_reference_action)
      .def_readonly("mcts_proposed_action", &open_spiel::SearchDiagnostics::mcts_proposed_action)
      .def_readonly("confidence_fallback", &open_spiel::SearchDiagnostics::confidence_fallback)
      .def_readonly("mcts_overrode_raw", &open_spiel::SearchDiagnostics::mcts_overrode_raw)
      .def_readonly("stability_checkpoint_action", &open_spiel::SearchDiagnostics::stability_checkpoint_action)
      .def_readonly("stability_checkpoint_reached", &open_spiel::SearchDiagnostics::stability_checkpoint_reached)
      .def_readonly("stability_agreement", &open_spiel::SearchDiagnostics::stability_agreement)
      .def_readonly("pass_complete_search", &open_spiel::SearchDiagnostics::pass_complete_search)
      .def_readonly("pass_min_actions", &open_spiel::SearchDiagnostics::pass_min_actions)
      .def_readonly("pass_prior_mass", &open_spiel::SearchDiagnostics::pass_prior_mass)
      .def_readonly("pass_meaningful_visits", &open_spiel::SearchDiagnostics::pass_meaningful_visits)
      .def_readonly("pass_q_margin", &open_spiel::SearchDiagnostics::pass_q_margin)
      .def_readonly("pass_stability", &open_spiel::SearchDiagnostics::pass_stability)
      .def_readonly("stage1_planned_worlds", &open_spiel::SearchDiagnostics::stage1_planned_worlds)
      .def_readonly("stage1_kept_worlds", &open_spiel::SearchDiagnostics::stage1_kept_worlds)
      .def_readonly("stage2_planned_worlds", &open_spiel::SearchDiagnostics::stage2_planned_worlds)
      .def_readonly("stage2_kept_worlds", &open_spiel::SearchDiagnostics::stage2_kept_worlds)
      .def_readonly("stage1_rollouts_executed", &open_spiel::SearchDiagnostics::stage1_rollouts_executed)
      .def_readonly("stage2_rollouts_executed", &open_spiel::SearchDiagnostics::stage2_rollouts_executed)
      .def_readonly("stage1_elapsed_ms", &open_spiel::SearchDiagnostics::stage1_elapsed_ms)
      .def_readonly("stage2_elapsed_ms", &open_spiel::SearchDiagnostics::stage2_elapsed_ms)
      .def_readonly("termination_reason", &open_spiel::SearchDiagnostics::termination_reason)
      .def_readonly("stage_outcome", &open_spiel::SearchDiagnostics::termination_reason)
      .def_readonly("pred_rem_0", &open_spiel::SearchDiagnostics::pred_rem_0)
      .def_readonly("pred_max_rem_sub", &open_spiel::SearchDiagnostics::pred_max_rem_sub)
      .def_readonly("pred_eff_rem", &open_spiel::SearchDiagnostics::pred_eff_rem)
      .def_readonly("pred_max_raw_rem", &open_spiel::SearchDiagnostics::pred_max_raw_rem);

  py::class_<open_spiel::DuneSearchResult>(di, "DuneSearchResult")
      .def_readonly("policy", &open_spiel::DuneSearchResult::policy)
      .def_readonly("diagnostics", &open_spiel::DuneSearchResult::diagnostics)
      .def_readonly("simulations_completed", &open_spiel::DuneSearchResult::simulations_completed)
      .def_readonly("elapsed_time_ms", &open_spiel::DuneSearchResult::elapsed_time_ms)
      .def_readonly("timeout_status", &open_spiel::DuneSearchResult::timeout_status)
      .def_readonly("used_fallback", &open_spiel::DuneSearchResult::used_fallback)
      .def_readonly("fallback_reason", &open_spiel::DuneSearchResult::fallback_reason)
      .def_readonly("inference_count", &open_spiel::DuneSearchResult::inference_count);

  py::class_<open_spiel::PolicyPriorDetails>(di, "PolicyPriorDetails")
      .def_readonly("ok", &open_spiel::PolicyPriorDetails::ok)
      .def_readonly("actions", &open_spiel::PolicyPriorDetails::actions)
      .def_readonly("probabilities", &open_spiel::PolicyPriorDetails::probabilities)
      .def_readonly("value", &open_spiel::PolicyPriorDetails::value);

  py::class_<open_spiel::PairedActionEvaluationResult>(di, "PairedActionEvaluationResult")
      .def_readonly("u_first", &open_spiel::PairedActionEvaluationResult::u_first)
      .def_readonly("u_second", &open_spiel::PairedActionEvaluationResult::u_second)
      .def_readonly("diff_mean", &open_spiel::PairedActionEvaluationResult::diff_mean)
      .def_readonly("diff_se", &open_spiel::PairedActionEvaluationResult::diff_se)
      .def_readonly("num_worlds", &open_spiel::PairedActionEvaluationResult::num_worlds);

  py::class_<open_spiel::ControllerDecision>(di, "ControllerDecision")
      .def(py::init<>())
      .def_readwrite("selected_action", &open_spiel::ControllerDecision::selected_action)
      .def_readwrite("raw_reference_action", &open_spiel::ControllerDecision::raw_reference_action)
      .def_readwrite("mcts_proposed_action", &open_spiel::ControllerDecision::mcts_proposed_action)
      .def_readwrite("confidence_fallback", &open_spiel::ControllerDecision::confidence_fallback)
      .def_readwrite("mcts_overrode_raw", &open_spiel::ControllerDecision::mcts_overrode_raw)
      .def_readwrite("stability_checkpoint_reached", &open_spiel::ControllerDecision::stability_checkpoint_reached)
      .def_readwrite("stability_agreement", &open_spiel::ControllerDecision::stability_agreement)
      .def_readwrite("pass_complete_search", &open_spiel::ControllerDecision::pass_complete_search)
      .def_readwrite("pass_min_actions", &open_spiel::ControllerDecision::pass_min_actions)
      .def_readwrite("pass_prior_mass", &open_spiel::ControllerDecision::pass_prior_mass)
      .def_readwrite("pass_meaningful_visits", &open_spiel::ControllerDecision::pass_meaningful_visits)
      .def_readwrite("pass_q_margin", &open_spiel::ControllerDecision::pass_q_margin)
      .def_readwrite("pass_stability", &open_spiel::ControllerDecision::pass_stability);

  py::classh<open_spiel::DunePUCTISMCTSBot, open_spiel::Bot>(di, "DunePUCTISMCTSBot")
      .def(py::init<const open_spiel::DuneSearchConfig&, std::shared_ptr<open_spiel::algorithms::Evaluator>>(),
           py::arg("config"), py::arg("evaluator"))
      .def(py::init<const open_spiel::DuneSearchConfig&, const std::vector<std::shared_ptr<open_spiel::algorithms::Evaluator>>&>(),
           py::arg("config"), py::arg("evaluators"))
      .def("run_search", &open_spiel::DunePUCTISMCTSBot::RunSearch, py::arg("state"), py::arg("max_sims") = -1, py::arg("max_time_ms") = -1.0, py::arg("start_sim_index") = 0)
      .def("get_root_diagnostics", &open_spiel::DunePUCTISMCTSBot::GetRootDiagnostics,
           py::arg("state"), py::arg("min_visit_threshold") = 2, py::arg("chosen_action") = open_spiel::kInvalidAction)
      .def("get_last_search_result", &open_spiel::DunePUCTISMCTSBot::GetLastSearchResult);

  py::enum_<open_spiel::DuneSearchBudgetMode>(di, "DuneSearchBudgetMode")
      .value("POLICY_ONLY", open_spiel::DuneSearchBudgetMode::kPolicyOnly)
      .value("FIXED_SESSION_SIMULATIONS", open_spiel::DuneSearchBudgetMode::kFixedSessionSimulations)
      .value("TRAINING_FULL_FAST", open_spiel::DuneSearchBudgetMode::kTrainingFullFast)
      .value("LIVE_DEADLINE", open_spiel::DuneSearchBudgetMode::kLiveDeadline);

  py::class_<open_spiel::DuneSearchSession>(di, "DuneSearchSession")
      .def(py::init<const open_spiel::DuneSearchConfig&, std::shared_ptr<open_spiel::algorithms::Evaluator>, open_spiel::DuneSearchBudgetMode>(),
           py::arg("config"), py::arg("evaluator"), py::arg("budget_mode"))
      .def(py::init<const open_spiel::DuneSearchConfig&, const std::vector<std::shared_ptr<open_spiel::algorithms::Evaluator>>&, open_spiel::DuneSearchBudgetMode>(),
           py::arg("config"), py::arg("evaluators"), py::arg("budget_mode"))
      .def("search", &open_spiel::DuneSearchSession::Search, py::arg("state"), py::arg("remaining_time_ms") = -1.0)
      .def("select_controller_action", &open_spiel::DuneSearchSession::SelectControllerAction, py::arg("state"), py::arg("search_result"), py::arg("r_val"))
      .def("commit_action", &open_spiel::DuneSearchSession::CommitAction, py::arg("decision"))
      .def("discard_pending_action", &open_spiel::DuneSearchSession::DiscardPendingAction)
      .def("search_and_select", py::overload_cast<const State&>(&open_spiel::DuneSearchSession::SearchAndSelect), py::arg("state"))
      .def("search_and_select", py::overload_cast<const State&, double>(&open_spiel::DuneSearchSession::SearchAndSelect), py::arg("state"), py::arg("r_val"))
      .def("set_episode_id", &open_spiel::DuneSearchSession::SetEpisodeId, py::arg("episode_id"))
      .def("set_update_id", &open_spiel::DuneSearchSession::SetUpdateId, py::arg("update_id"))
      .def("reset_session", &open_spiel::DuneSearchSession::ResetSession, py::arg("reason"))
      .def("has_active_session", &open_spiel::DuneSearchSession::HasActiveSession)
      .def("get_bot", &open_spiel::DuneSearchSession::GetBot)
      .def("get_short_bot", &open_spiel::DuneSearchSession::GetShortBot)
      .def_property_readonly("session_new_simulations_completed", &open_spiel::DuneSearchSession::session_new_simulations_completed)
      .def_property_readonly("session_elapsed_time_ms", &open_spiel::DuneSearchSession::session_elapsed_time_ms);

  py::class_<open_spiel::DuneAdaptiveRolloutSession>(di, "DuneAdaptiveRolloutSession")
      .def(py::init<const std::string&, const std::string&, int, int, int, int, int, uint64_t, bool, bool, int, int, int, bool, bool, bool>(),
           py::arg("checkpoint_path"),
           py::arg("device_str") = "cuda",
           py::arg("hidden_dim") = 2048,
           py::arg("num_blocks") = 16,
           py::arg("reference_rollouts_per_action") = 64,
           py::arg("leader_rollouts") = 32,
           py::arg("target_batch_size") = 256,
           py::arg("seed") = 20261014ULL,
           py::arg("use_successive_halving") = true,
           py::arg("fixed_top5_candidates") = true,
           py::arg("max_inflight_worlds") = 96,
           py::arg("num_evaluators") = 1,
           py::arg("top5_min_budget") = 576,
           py::arg("all_buy_candidates") = true,
           py::arg("enable_fp8_trunk") = false,
           py::arg("search_v2") = false)
      .def("search", &open_spiel::DuneAdaptiveRolloutSession::Search,
           py::arg("state"), py::arg("remaining_time_ms") = -1.0)
      .def("evaluate_action_pair_terminal",
           &open_spiel::DuneAdaptiveRolloutSession::EvaluateActionPairTerminal,
           py::arg("state"), py::arg("a_first"), py::arg("a_second"),
           py::arg("num_worlds") = 64, py::arg("seed") = 20261104ULL)
      .def("prior_with_details", &open_spiel::DuneAdaptiveRolloutSession::PriorWithDetails,
           py::arg("state"))
      .def("discard_pending_action", &open_spiel::DuneAdaptiveRolloutSession::DiscardPendingAction)
      .def("reset_session", &open_spiel::DuneAdaptiveRolloutSession::ResetSession, py::arg("reason"))
      .def("set_seed", &open_spiel::DuneAdaptiveRolloutSession::set_seed, py::arg("seed"))
      .def("has_active_session", &open_spiel::DuneAdaptiveRolloutSession::HasActiveSession)
      .def_property_readonly("session_new_simulations_completed",
                             &open_spiel::DuneAdaptiveRolloutSession::session_new_simulations_completed)
      .def_property_readonly("session_elapsed_time_ms",
                             &open_spiel::DuneAdaptiveRolloutSession::session_elapsed_time_ms)
      .def_property("fixed_top5_candidates",
                    &open_spiel::DuneAdaptiveRolloutSession::fixed_top5_candidates,
                    &open_spiel::DuneAdaptiveRolloutSession::set_fixed_top5_candidates)
      .def("set_fixed_top5_candidates",
           &open_spiel::DuneAdaptiveRolloutSession::set_fixed_top5_candidates,
           py::arg("val"))
      .def_property("max_inflight_worlds",
                    &open_spiel::DuneAdaptiveRolloutSession::max_inflight_worlds,
                    &open_spiel::DuneAdaptiveRolloutSession::set_max_inflight_worlds)
      .def("set_max_inflight_worlds",
           &open_spiel::DuneAdaptiveRolloutSession::set_max_inflight_worlds,
           py::arg("val"))
      .def_property_readonly("num_evaluators",
                             &open_spiel::DuneAdaptiveRolloutSession::num_evaluators)
      .def_property("top5_min_budget",
                    &open_spiel::DuneAdaptiveRolloutSession::top5_min_budget,
                    &open_spiel::DuneAdaptiveRolloutSession::set_top5_min_budget)
      .def("set_top5_min_budget",
           &open_spiel::DuneAdaptiveRolloutSession::set_top5_min_budget,
           py::arg("val"))
      .def_property("all_buy_candidates",
                    &open_spiel::DuneAdaptiveRolloutSession::all_buy_candidates,
                    &open_spiel::DuneAdaptiveRolloutSession::set_all_buy_candidates)
      .def("set_all_buy_candidates",
           &open_spiel::DuneAdaptiveRolloutSession::set_all_buy_candidates,
           py::arg("val"))
      .def_property("reserve_ms",
                    &open_spiel::DuneAdaptiveRolloutSession::reserve_ms,
                    &open_spiel::DuneAdaptiveRolloutSession::set_reserve_ms)
      .def("set_reserve_ms",
           &open_spiel::DuneAdaptiveRolloutSession::set_reserve_ms,
           py::arg("val"))
      .def_property("full_search",
                    &open_spiel::DuneAdaptiveRolloutSession::full_search,
                    &open_spiel::DuneAdaptiveRolloutSession::set_full_search)
      .def("set_full_search",
           &open_spiel::DuneAdaptiveRolloutSession::set_full_search,
           py::arg("val"))
      .def_property("superhuman_mode",
                    &open_spiel::DuneAdaptiveRolloutSession::superhuman_mode,
                    &open_spiel::DuneAdaptiveRolloutSession::set_superhuman_mode)
      .def("set_superhuman_mode",
           &open_spiel::DuneAdaptiveRolloutSession::set_superhuman_mode,
           py::arg("val"))
      .def_property("win_speed_tiebreak_eps",
                    &open_spiel::DuneAdaptiveRolloutSession::win_speed_tiebreak_eps,
                    &open_spiel::DuneAdaptiveRolloutSession::set_win_speed_tiebreak_eps)
      .def("set_win_speed_tiebreak_eps",
           &open_spiel::DuneAdaptiveRolloutSession::set_win_speed_tiebreak_eps,
           py::arg("val"))
      .def_property("ceiling_aware_margin",
                    &open_spiel::DuneAdaptiveRolloutSession::ceiling_aware_margin,
                    &open_spiel::DuneAdaptiveRolloutSession::set_ceiling_aware_margin)
      .def("set_ceiling_aware_margin",
           &open_spiel::DuneAdaptiveRolloutSession::set_ceiling_aware_margin,
           py::arg("val"))
      .def_property("unbiased_stage2_gate",
                    &open_spiel::DuneAdaptiveRolloutSession::unbiased_stage2_gate,
                    &open_spiel::DuneAdaptiveRolloutSession::set_unbiased_stage2_gate)
      .def("set_unbiased_stage2_gate",
           &open_spiel::DuneAdaptiveRolloutSession::set_unbiased_stage2_gate,
           py::arg("val"))
      .def_property("kl_policy_tau",
                    &open_spiel::DuneAdaptiveRolloutSession::kl_policy_tau,
                    &open_spiel::DuneAdaptiveRolloutSession::set_kl_policy_tau)
      .def("set_kl_policy_tau",
           &open_spiel::DuneAdaptiveRolloutSession::set_kl_policy_tau,
           py::arg("val"))
      .def_property("search_v2",
                    &open_spiel::DuneAdaptiveRolloutSession::search_v2,
                    &open_spiel::DuneAdaptiveRolloutSession::set_search_v2)
      .def("set_search_v2",
           &open_spiel::DuneAdaptiveRolloutSession::set_search_v2,
           py::arg("val"))
      .def_property("prior_log_ratio_weight",
                    &open_spiel::DuneAdaptiveRolloutSession::prior_log_ratio_weight,
                    &open_spiel::DuneAdaptiveRolloutSession::set_prior_log_ratio_weight)
      .def("set_prior_log_ratio_weight",
           &open_spiel::DuneAdaptiveRolloutSession::set_prior_log_ratio_weight,
           py::arg("val"))
      .def_property("stage1_high_conf_prior_skip",
                    &open_spiel::DuneAdaptiveRolloutSession::stage1_high_conf_prior_skip,
                    &open_spiel::DuneAdaptiveRolloutSession::set_stage1_high_conf_prior_skip)
      .def("set_stage1_high_conf_prior_skip",
           &open_spiel::DuneAdaptiveRolloutSession::set_stage1_high_conf_prior_skip,
           py::arg("val"))
      .def_property("use_event_keyed_crn",
                    &open_spiel::DuneAdaptiveRolloutSession::use_event_keyed_crn,
                    &open_spiel::DuneAdaptiveRolloutSession::set_use_event_keyed_crn)
      .def("set_use_event_keyed_crn",
           &open_spiel::DuneAdaptiveRolloutSession::set_use_event_keyed_crn,
           py::arg("val"))
      .def_property_readonly("ms_per_rollout_ema",
                             &open_spiel::DuneAdaptiveRolloutSession::ms_per_rollout_ema)
      .def_property_readonly("turn_elapsed_search_time_ms",
                             &open_spiel::DuneAdaptiveRolloutSession::turn_elapsed_search_time_ms)
      .def_property("enable_fp8_trunk",
                    &open_spiel::DuneAdaptiveRolloutSession::enable_fp8_trunk,
                    &open_spiel::DuneAdaptiveRolloutSession::set_enable_fp8_trunk)
      .def("set_enable_fp8_trunk",
           &open_spiel::DuneAdaptiveRolloutSession::set_enable_fp8_trunk,
           py::arg("val"))
      .def_property("rollout_checkpoint",
                    &open_spiel::DuneAdaptiveRolloutSession::rollout_checkpoint,
                    &open_spiel::DuneAdaptiveRolloutSession::set_rollout_checkpoint)
      .def("set_rollout_checkpoint",
           &open_spiel::DuneAdaptiveRolloutSession::set_rollout_checkpoint,
           py::arg("path"))
      .def_property_readonly("rollout_num_blocks",
                             &open_spiel::DuneAdaptiveRolloutSession::rollout_num_blocks)
      .def_property("enable_macro_space_candidates",
                    &open_spiel::DuneAdaptiveRolloutSession::enable_macro_space_candidates,
                    &open_spiel::DuneAdaptiveRolloutSession::set_enable_macro_space_candidates)
      .def("set_enable_macro_space_candidates",
           &open_spiel::DuneAdaptiveRolloutSession::set_enable_macro_space_candidates,
           py::arg("val"))
      .def_property("macro_space_min_round",
                    &open_spiel::DuneAdaptiveRolloutSession::macro_space_min_round,
                    &open_spiel::DuneAdaptiveRolloutSession::set_macro_space_min_round)
      .def("set_macro_space_min_round",
           &open_spiel::DuneAdaptiveRolloutSession::set_macro_space_min_round,
           py::arg("val"))
      .def_property("enable_endgame_ply2",
                    &open_spiel::DuneAdaptiveRolloutSession::enable_endgame_ply2,
                    &open_spiel::DuneAdaptiveRolloutSession::set_enable_endgame_ply2)
      .def("set_enable_endgame_ply2",
           &open_spiel::DuneAdaptiveRolloutSession::set_enable_endgame_ply2,
           py::arg("val"))
      .def_property("endgame_ply2_stage2_only",
                    &open_spiel::DuneAdaptiveRolloutSession::endgame_ply2_stage2_only,
                    &open_spiel::DuneAdaptiveRolloutSession::set_endgame_ply2_stage2_only)
      .def("set_endgame_ply2_stage2_only",
           &open_spiel::DuneAdaptiveRolloutSession::set_endgame_ply2_stage2_only,
           py::arg("val"))
      .def_property("endgame_ply2_worlds",
                    &open_spiel::DuneAdaptiveRolloutSession::endgame_ply2_worlds,
                    &open_spiel::DuneAdaptiveRolloutSession::set_endgame_ply2_worlds)
      .def("set_endgame_ply2_worlds",
           &open_spiel::DuneAdaptiveRolloutSession::set_endgame_ply2_worlds,
           py::arg("val"))
      .def_property("endgame_ply2_margin",
                    &open_spiel::DuneAdaptiveRolloutSession::endgame_ply2_margin,
                    &open_spiel::DuneAdaptiveRolloutSession::set_endgame_ply2_margin)
      .def("set_endgame_ply2_margin",
           &open_spiel::DuneAdaptiveRolloutSession::set_endgame_ply2_margin,
           py::arg("val"))
      .def_property("enable_intrigue_belief",
                    &open_spiel::DuneAdaptiveRolloutSession::enable_intrigue_belief,
                    &open_spiel::DuneAdaptiveRolloutSession::set_enable_intrigue_belief)
      .def("set_enable_intrigue_belief",
           &open_spiel::DuneAdaptiveRolloutSession::set_enable_intrigue_belief,
           py::arg("val"))
      .def("set_intrigue_belief_params",
           &open_spiel::DuneAdaptiveRolloutSession::set_intrigue_belief_params,
           py::arg("age_combat_slope"), py::arg("age_endgame_slope"),
           py::arg("close_pass_penalty"))
      .def_property("rescue_optional_abilities",
                    &open_spiel::DuneAdaptiveRolloutSession::rescue_optional_abilities,
                    &open_spiel::DuneAdaptiveRolloutSession::set_rescue_optional_abilities)
      .def("set_rescue_optional_abilities",
           &open_spiel::DuneAdaptiveRolloutSession::set_rescue_optional_abilities,
           py::arg("val"))
      .def("get_search_v2_tuning", [](const open_spiel::DuneAdaptiveRolloutSession& self) {
        py::dict d;
        const auto& t = self.search_v2_tuning();
        d["subwave_1a_max_budget_ms"] = t.subwave_1a_max_budget_ms;
        d["legacy_s1_inflight_cap"] = t.legacy_s1_inflight_cap;
        d["seed_s2_ms_with_min_prior"] = t.seed_s2_ms_with_min_prior;
        d["s2_max_w0"] = t.s2_max_w0;
        d["s2_max_wave_worlds"] = t.s2_max_wave_worlds;
        d["s2_inflight_cap"] = t.s2_inflight_cap;
        d["legacy_clamp_session_inflight"] = t.legacy_clamp_session_inflight;
        d["r_stage1_cap"] = t.r_stage1_cap;
        d["s1_budget_frac"] = t.s1_budget_frac;
        d["enable_conditional_3arm_stage2"] = t.enable_conditional_3arm_stage2;
        d["three_arm_min_rem_ms"] = t.three_arm_min_rem_ms;
        d["three_arm_max_s1_gap"] = t.three_arm_max_s1_gap;
        d["three_arm_prune_se"] = t.three_arm_prune_se;
        d["unclocked_c96_3arm_mode"] = t.unclocked_c96_3arm_mode;
        d["unclocked_3arm_min_s2_worlds"] = t.unclocked_3arm_min_s2_worlds;
        d["enable_rem0_turn_reservation"] = t.enable_rem0_turn_reservation;
        return d;
      })
      .def("configure_search_v2_tuning", [](open_spiel::DuneAdaptiveRolloutSession& self,
                                            py::object subwave_1a_max_budget_ms,
                                            py::object legacy_s1_inflight_cap,
                                            py::object seed_s2_ms_with_min_prior,
                                            py::object s2_max_w0,
                                            py::object s2_max_wave_worlds,
                                            py::object s2_inflight_cap,
                                            py::object legacy_clamp_session_inflight,
                                            py::object r_stage1_cap,
                                            py::object s1_budget_frac,
                                            py::object enable_conditional_3arm_stage2,
                                            py::object three_arm_min_rem_ms,
                                            py::object three_arm_max_s1_gap,
                                            py::object three_arm_prune_se,
                                            py::object enable_rem0_turn_reservation,
                                            py::object unclocked_c96_3arm_mode,
                                            py::object unclocked_3arm_min_s2_worlds) {
        auto cfg = self.search_v2_tuning();
        if (!subwave_1a_max_budget_ms.is_none()) cfg.subwave_1a_max_budget_ms = subwave_1a_max_budget_ms.cast<double>();
        if (!legacy_s1_inflight_cap.is_none()) cfg.legacy_s1_inflight_cap = legacy_s1_inflight_cap.cast<bool>();
        if (!seed_s2_ms_with_min_prior.is_none()) cfg.seed_s2_ms_with_min_prior = seed_s2_ms_with_min_prior.cast<bool>();
        if (!s2_max_w0.is_none()) cfg.s2_max_w0 = s2_max_w0.cast<int>();
        if (!s2_max_wave_worlds.is_none()) cfg.s2_max_wave_worlds = s2_max_wave_worlds.cast<int>();
        if (!s2_inflight_cap.is_none()) cfg.s2_inflight_cap = s2_inflight_cap.cast<int>();
        if (!legacy_clamp_session_inflight.is_none()) cfg.legacy_clamp_session_inflight = legacy_clamp_session_inflight.cast<bool>();
        if (!r_stage1_cap.is_none()) cfg.r_stage1_cap = r_stage1_cap.cast<int>();
        if (!s1_budget_frac.is_none()) cfg.s1_budget_frac = s1_budget_frac.cast<double>();
        if (!enable_conditional_3arm_stage2.is_none()) cfg.enable_conditional_3arm_stage2 = enable_conditional_3arm_stage2.cast<bool>();
        if (!three_arm_min_rem_ms.is_none()) cfg.three_arm_min_rem_ms = three_arm_min_rem_ms.cast<double>();
        if (!three_arm_max_s1_gap.is_none()) cfg.three_arm_max_s1_gap = three_arm_max_s1_gap.cast<double>();
        if (!three_arm_prune_se.is_none()) cfg.three_arm_prune_se = three_arm_prune_se.cast<double>();
        if (!enable_rem0_turn_reservation.is_none()) cfg.enable_rem0_turn_reservation = enable_rem0_turn_reservation.cast<bool>();
        if (!unclocked_c96_3arm_mode.is_none()) cfg.unclocked_c96_3arm_mode = unclocked_c96_3arm_mode.cast<bool>();
        if (!unclocked_3arm_min_s2_worlds.is_none()) cfg.unclocked_3arm_min_s2_worlds = unclocked_3arm_min_s2_worlds.cast<int>();
        self.set_search_v2_tuning(cfg);
      },
      py::arg("subwave_1a_max_budget_ms") = py::none(),
      py::arg("legacy_s1_inflight_cap") = py::none(),
      py::arg("seed_s2_ms_with_min_prior") = py::none(),
      py::arg("s2_max_w0") = py::none(),
      py::arg("s2_max_wave_worlds") = py::none(),
      py::arg("s2_inflight_cap") = py::none(),
      py::arg("legacy_clamp_session_inflight") = py::none(),
      py::arg("r_stage1_cap") = py::none(),
      py::arg("s1_budget_frac") = py::none(),
      py::arg("enable_conditional_3arm_stage2") = py::none(),
      py::arg("three_arm_min_rem_ms") = py::none(),
      py::arg("three_arm_max_s1_gap") = py::none(),
      py::arg("three_arm_prune_se") = py::none(),
      py::arg("enable_rem0_turn_reservation") = py::none(),
      py::arg("unclocked_c96_3arm_mode") = py::none(),
      py::arg("unclocked_3arm_min_s2_worlds") = py::none())
      .def("get_ms_per_rollout_by_round", &open_spiel::DuneAdaptiveRolloutSession::GetMsPerRolloutByRound)
      .def("set_ms_per_rollout_by_round", &open_spiel::DuneAdaptiveRolloutSession::SetMsPerRolloutByRound, py::arg("vals"))
      .def("run_direct_search_v2_root", [](open_spiel::DuneAdaptiveRolloutSession& self,
                                           const open_spiel::State& state,
                                           double target_budget_ms,
                                           uint64_t decision_seed,
                                           bool is_purchase,
                                           int override_ref_rollouts) {
        auto res = self.RunDirectSearchV2Root(state, target_budget_ms, decision_seed, is_purchase, override_ref_rollouts);
        py::dict d;
        d["elapsed_ms"] = res.elapsed_ms;
        d["stage1_elapsed_ms"] = res.stage1_elapsed_ms;
        d["stage2_elapsed_ms"] = res.stage2_elapsed_ms;
        d["new_rollouts_executed"] = res.new_rollouts_executed;
        d["stage1_rollouts_executed"] = res.stage1_rollouts_executed;
        d["stage2_rollouts_executed"] = res.stage2_rollouts_executed;
        d["stage1_planned_worlds"] = res.stage1_planned_worlds;
        d["stage1_kept_worlds"] = res.stage1_kept_worlds;
        d["stage2_planned_worlds"] = res.stage2_planned_worlds;
        d["stage2_kept_worlds"] = res.stage2_kept_worlds;
        d["termination_reason"] = res.termination_reason;
        d["overridden"] = res.overridden;
        d["usable_override"] = res.usable_override;
        d["chosen_first_action"] = res.chosen_first_action;
        d["chosen_graft_action"] = res.chosen_graft_action;
        d["chosen_second_action"] = res.chosen_second_action;
        d["raw_first_action"] = res.raw_first_action;
        d["raw_graft_action"] = res.raw_graft_action;
        d["raw_second_action"] = res.raw_second_action;
        d["chosen_candidate_index"] = res.chosen_candidate_index;
        d["num_candidates"] = res.num_candidates;
        d["est_ms_per_sim_used"] = res.est_ms_per_sim_used;
        return d;
      }, py::arg("state"), py::arg("target_budget_ms"), py::arg("decision_seed"), py::arg("is_purchase") = false, py::arg("override_ref_rollouts") = -1)
      .def("run_direct_search_v1_sequential_root", [](open_spiel::DuneAdaptiveRolloutSession& self,
                                                      const open_spiel::State& state,
                                                      double target_budget_ms,
                                                      uint64_t decision_seed,
                                                      int override_ref_rollouts) {
        auto res = self.RunDirectSearchV1SequentialRoot(state, target_budget_ms, decision_seed, override_ref_rollouts);
        py::dict d;
        d["elapsed_ms"] = res.elapsed_ms;
        d["new_rollouts_executed"] = res.new_rollouts_executed;
        d["overridden"] = res.overridden;
        d["usable_override"] = res.usable_override;
        d["chosen_first_action"] = res.chosen_first_action;
        d["chosen_graft_action"] = res.chosen_graft_action;
        d["chosen_second_action"] = res.chosen_second_action;
        d["raw_first_action"] = res.raw_first_action;
        d["raw_graft_action"] = res.raw_graft_action;
        d["raw_second_action"] = res.raw_second_action;
        return d;
      }, py::arg("state"), py::arg("target_budget_ms"), py::arg("decision_seed"), py::arg("override_ref_rollouts") = -1)
      .def("evaluate_joint_candidates_paired", [](open_spiel::DuneAdaptiveRolloutSession& self,
                                                 const open_spiel::State& state,
                                                 const std::vector<open_spiel::Action>& first_actions,
                                                 const std::vector<open_spiel::Action>& graft_actions,
                                                 const std::vector<open_spiel::Action>& second_actions,
                                                 int num_worlds,
                                                 uint64_t decision_seed) {
        auto res = self.EvaluateJointCandidatesPaired(state, first_actions, graft_actions, second_actions, num_worlds, decision_seed);
        py::dict d;
        d["mean_utilities"] = res.mean_utilities;
        d["per_candidate_rollouts"] = res.per_candidate_rollouts;
        d["elapsed_ms"] = res.elapsed_ms;
        d["rollouts_executed"] = res.rollouts_executed;
        return d;
      }, py::arg("state"), py::arg("first_actions"), py::arg("graft_actions"), py::arg("second_actions"), py::arg("num_worlds"), py::arg("decision_seed"));

  di.def("make_dune_nn_evaluator", &open_spiel::MakeDuneNNEvaluator,
         py::arg("checkpoint_path"), py::arg("device_str"),
         py::arg("hidden_dim") = 1024, py::arg("num_blocks") = 4);

  py::enum_<open_spiel::DuneDecisionRole>(di, "DuneDecisionRole")
      .value("FORCED_OR_BOOKKEEPING", open_spiel::DuneDecisionRole::kForcedOrBookkeeping)
      .value("LEADER_SELECTION", open_spiel::DuneDecisionRole::kLeaderSelection)
      .value("AGENT_PRIMARY", open_spiel::DuneDecisionRole::kAgentPrimary)
      .value("AGENT_CONTINUATION", open_spiel::DuneDecisionRole::kAgentContinuation)
      .value("PURCHASE", open_spiel::DuneDecisionRole::kPurchase)
      .value("COMBAT_INTRIGUE", open_spiel::DuneDecisionRole::kCombatIntrigue)
      .value("OTHER_OPTIONAL", open_spiel::DuneDecisionRole::kOtherOptional);

  di.def("classify_dune_decision_role", &open_spiel::ClassifyDuneDecisionRole,
         py::arg("state"), py::arg("searched_player"), py::arg("has_active_session"));
}

}  // namespace dune_imperium_pybind
}  // namespace open_spiel

#endif
