#include "dune_search_benchmark_internal.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <map>

#include "open_spiel/abseil-cpp/absl/flags/flag.h"
#include "open_spiel/abseil-cpp/absl/strings/str_format.h"
#include "open_spiel/utils/json.h"

namespace open_spiel {

void WriteSearchDecisionJsonl(
    const std::string& search_jsonl,
    int episode_id,
    uint64_t matched_step_count,
    uint64_t game_seed,
    int search_seat,
    Player current_player,
    const DuneSearchResult& last_res,
    Action chosen_action,
    const State& state,
    const std::array<SeatControllerProvenance, 4>& seat_provenance,
    bool matched_sampled_this_decision,
    std::mutex& log_mutex) {
  SearchDiagnostics diag = last_res.diagnostics;
  open_spiel::json::Object search_obj;
  search_obj["episode_id"] = static_cast<int64_t>(episode_id);
  search_obj["decision_ordinal"] = static_cast<int64_t>(matched_step_count);
  search_obj["game_seed"] = static_cast<int64_t>(game_seed);
  search_obj["search_seat"] = static_cast<int64_t>(search_seat);
  search_obj["player"] = static_cast<int64_t>(current_player);
  search_obj["decision_role"] = diag.decision_role;
  search_obj["is_strategic"] = (diag.decision_role == "2" || diag.decision_role == "3");
  search_obj["re_root_status"] = diag.re_root_status;
  search_obj["reset_reason"] = diag.reset_reason;
  search_obj["hard_sim_limit"] = static_cast<int64_t>(diag.hard_sim_limit);
  search_obj["simulations_completed"] = static_cast<int64_t>(last_res.simulations_completed);
  search_obj["inference_count"] = static_cast<int64_t>(last_res.inference_count);
  search_obj["timeout_status"] = last_res.timeout_status;
  search_obj["used_fallback"] = last_res.used_fallback;
  search_obj["fallback_reason"] = last_res.fallback_reason;
  search_obj["selection_mode"] = diag.selection_mode;
  search_obj["selection_reason"] = diag.selection_reason;
  search_obj["search_accepted"] = diag.search_accepted;
  search_obj["incumbent_action"] = static_cast<int64_t>(diag.incumbent_action);
  search_obj["raw_prior_max_prob"] = diag.raw_prior_max_prob;
  search_obj["raw_prior_selected_prob"] = diag.raw_prior_selected_prob;
  search_obj["raw_prior_max_prob_hex"] = diag.raw_prior_max_prob_hex;
  search_obj["raw_prior_selected_prob_hex"] = diag.raw_prior_selected_prob_hex;
  search_obj["elapsed_time_ms"] = last_res.elapsed_time_ms;
  search_obj["action_chosen"] = static_cast<int64_t>(chosen_action);
  const auto* dune_state = dynamic_cast<const dune_imperium::DuneImperiumState*>(&state);
  if (dune_state != nullptr) {
    search_obj["action_chosen_string"] = dune_state->ActionToString(current_player, chosen_action);
  }
  open_spiel::json::Array legals_arr;
  for (Action a : state.LegalActions()) {
    legals_arr.push_back(static_cast<int64_t>(a));
  }
  search_obj["legal_actions"] = legals_arr;
  open_spiel::json::Array visits_arr;
  for (int v : diag.visit_counts) {
    visits_arr.push_back(static_cast<int64_t>(v));
  }
  search_obj["visit_counts"] = visits_arr;
  open_spiel::json::Array q_arr;
  for (double q : diag.q_values) {
    q_arr.push_back(q);
  }
  search_obj["q_values"] = q_arr;
  open_spiel::json::Array priors_arr;
  for (double p_val : diag.priors) {
    priors_arr.push_back(p_val);
  }
  search_obj["priors"] = priors_arr;
  search_obj["root_value"] = diag.root_value;
  search_obj["max_depth"] = diag.max_depth;
  search_obj["mean_depth"] = diag.mean_depth;
  search_obj["p95_depth"] = diag.p95_depth;
  search_obj["deepest_simulated_round"] = static_cast<int64_t>(diag.deepest_simulated_round);
  search_obj["terminal_leaf_fraction"] = diag.terminal_leaf_fraction;
  search_obj["unique_nodes"] = static_cast<int64_t>(diag.unique_nodes);
  search_obj["raw_to_search_policy_kl"] = diag.raw_to_search_policy_kl;
  search_obj["chosen_action_raw_prior_probability"] = diag.chosen_action_raw_prior_probability;
  search_obj["chosen_action_raw_prior_rank"] = static_cast<int64_t>(diag.chosen_action_raw_prior_rank);
  search_obj["action_changed_vs_raw_argmax"] = diag.action_changed_vs_raw_argmax;

  open_spiel::json::Array forced_arr;
  for (int v : diag.forced_visit_counts) {
    forced_arr.push_back(static_cast<int64_t>(v));
  }
  search_obj["forced_visit_counts"] = forced_arr;

  open_spiel::json::Array pruned_arr;
  for (int v : diag.pruned_visit_counts) {
    pruned_arr.push_back(static_cast<int64_t>(v));
  }
  search_obj["pruned_visit_counts"] = pruned_arr;

  // Write 18A-v2 specific diagnostics/telemetry
  search_obj["protocol_version"] = diag.protocol_version;
  search_obj["session_id"] = diag.session_id;
  search_obj["searched_seat"] = static_cast<int64_t>(diag.searched_seat);
  search_obj["round"] = static_cast<int64_t>(diag.round);
  search_obj["phase"] = diag.phase;
  search_obj["decision_role"] = diag.decision_role;
  search_obj["budget_mode"] = diag.budget_mode;
  search_obj["budget_mode_name"] =
      seat_provenance[current_player].budget_mode;
  search_obj["audit_training_policy_iteration"] =
      absl::GetFlag(FLAGS_audit_training_policy_iteration);
  search_obj["hard_sim_limit"] = static_cast<int64_t>(diag.hard_sim_limit);
  search_obj["soft_sim_limit"] = static_cast<int64_t>(diag.soft_sim_limit);
  search_obj["hard_time_limit_ms"] = diag.hard_time_limit_ms;
  search_obj["soft_time_limit_ms"] = diag.soft_time_limit_ms;
  search_obj["elapsed_search_time_ms"] = diag.elapsed_search_time_ms;
  search_obj["observation_wait_time_ms"] = diag.observation_wait_time_ms;
  // PF-2 Part B corroboration marker. Emitted ONLY when the flag is
  // ON: with it off the field is ABSENT, so the flag-OFF neutrality
  // projection stays exactly the five Part-A fields and a correct
  // implementation cannot fail that proof. In ON mode it is present
  // on EVERY candidate decision row, true iff Part B sampled that
  // decision — so the corroboration count is
  // sum(row["matched_fallback_sampled"] == true), never a count of
  // field occurrences. It corroborates that the branch was
  // exercised; the classification-parity UNIT TEST, not this
  // marker, governs whether the classification is correct.
  if (absl::GetFlag(FLAGS_matched_fallback_sampling)) {
    search_obj["matched_fallback_sampled"] =
        matched_sampled_this_decision;
  }
  // PF-2 Part A, fifth field. Computed in search
  // (dune_puct_is_mcts.cc:772/:785/:793, stored to diagnostics at
  // :1086) but serialized ONLY by dune_pwo4_trajectory.cc:1116 —
  // this binary emitted zero occurrences, so B1's registered
  // retention-rate deliverable could not be produced by the binary
  // that runs B1. Denominator for that rate is positive-simulation,
  // valid-snapshot rows, not all rows.
  search_obj["root_reused_pre_search"] = diag.root_reused_pre_search;
  search_obj["inherited_root_visits"] = static_cast<int64_t>(diag.inherited_root_visits);
  search_obj["newly_completed_simulations"] = static_cast<int64_t>(diag.newly_completed_simulations);
  search_obj["session_cumulative_simulations"] = static_cast<int64_t>(diag.session_cumulative_simulations);
  search_obj["short_window_cumulative_simulations"] = static_cast<int64_t>(diag.short_window_cumulative_simulations);
  search_obj["session_cumulative_search_time_ms"] = diag.session_cumulative_search_time_ms;
  search_obj["long_agent_session_cumulative_time_ms"] = diag.long_agent_session_cumulative_time_ms;
  search_obj["re_root_status"] = diag.re_root_status;
  search_obj["post_chance_branch_miss"] = diag.post_chance_branch_miss;
  search_obj["root_coverage"] = diag.root_coverage;
  search_obj["reset_reason"] = diag.reset_reason;
  search_obj["tree_node_count"] = static_cast<int64_t>(diag.tree_node_count);
  search_obj["legality_result"] = diag.legality_result;
  search_obj["num_covered_actions"] = static_cast<int64_t>(diag.num_covered_actions);
  search_obj["covered_prior_mass"] = diag.covered_prior_mass;
  search_obj["mean_decision_depth"] = diag.mean_decision_depth;
  search_obj["max_decision_depth"] = static_cast<int64_t>(diag.max_decision_depth);
  search_obj["time_sample_root_ms"] = diag.time_sample_root_ms;
  search_obj["time_nn_inference_ms"] = diag.time_nn_inference_ms;
  search_obj["time_tree_engine_ms"] = diag.time_tree_engine_ms;
  search_obj["time_other_overhead_ms"] = diag.time_other_overhead_ms;

  open_spiel::json::Array hist_arr;
  for (Action a : state.History()) {
    hist_arr.push_back(static_cast<int64_t>(a));
  }
  search_obj["state_history"] = hist_arr;

  // --- WO-1 Phase 3: per-decision policy shape + budget health ---
  // Policy-shape stats are reported for TWO distributions, because
  // they answer different questions: the raw network prior (how
  // decisive is the policy head) and the search's visit
  // distribution (how decisive is the controller after search).
  // NOTE: pre-cap |z| saturation is deliberately NOT emitted here.
  // DuneNNEvaluator applies cap*tanh(z/cap) inside Prior() and
  // discards the pre-cap logits, and the tanh squash means post-cap
  // values never reach the cap -- any saturation fraction derived
  // from them would be identically zero. That statistic is a
  // property of the policy network, and it is measured exactly, at
  // the configured cap and 2x cap, by dune_population_eval's
  // --dump_logit_stats path whose dump runs BEFORE the cap.
  auto shape_stats = [](const std::vector<double>& w,
                        double* norm_h, double* max_p,
                        double* eff_n) {
    double total = 0.0;
    for (double v : w) total += (v > 0.0 ? v : 0.0);
    const int n = static_cast<int>(w.size());
    if (n <= 0 || total <= 0.0) {
      *norm_h = 0.0; *max_p = 0.0; *eff_n = 0.0;
      return;
    }
    double h = 0.0, mx = 0.0;
    for (double v : w) {
      const double p = (v > 0.0 ? v : 0.0) / total;
      if (p > 0.0) h -= p * std::log(p);
      mx = std::max(mx, p);
    }
    // n == 1 has log(n) == 0: a forced decision carries no choice
    // entropy, so define it as 0.0 rather than emitting NaN.
    *norm_h = (n > 1) ? (h / std::log(static_cast<double>(n))) : 0.0;
    *max_p = mx;
    *eff_n = std::exp(h);
  };
  const std::vector<double>& raw_w =
      diag.raw_priors.empty() ? diag.priors : diag.raw_priors;
  double rp_h = 0.0, rp_max = 0.0, rp_eff = 0.0;
  shape_stats(raw_w, &rp_h, &rp_max, &rp_eff);
  search_obj["raw_prior_norm_entropy"] = rp_h;
  search_obj["raw_prior_max_action_prob"] = rp_max;
  search_obj["raw_prior_eff_action_count"] = rp_eff;
  std::vector<double> visit_w;
  visit_w.reserve(diag.visit_counts.size());
  for (int v : diag.visit_counts) visit_w.push_back(static_cast<double>(v));
  double sp_h = 0.0, sp_max = 0.0, sp_eff = 0.0;
  shape_stats(visit_w, &sp_h, &sp_max, &sp_eff);
  search_obj["search_policy_norm_entropy"] = sp_h;
  search_obj["search_policy_max_action_prob"] = sp_max;
  search_obj["search_policy_eff_action_count"] = sp_eff;

  search_obj["sims_per_sec"] =
      (last_res.elapsed_time_ms > 0.0)
          ? (last_res.simulations_completed * 1000.0 /
             last_res.elapsed_time_ms)
          : 0.0;
  // Which SESSION-level budget ran out, if any. Set by the session
  // alongside (never overwriting) RunSearch's own fallback_reason.
  search_obj["budget_limit_reason"] = diag.budget_limit_reason;
  // "none" is the SENTINEL for "no session limit was hit", not an
  // empty string -- testing emptiness marks every row starved.
  // Also excluded: soft_sim_limit <= 0, which is true for every
  // forced/bookkeeping decision. Those are non-search decisions,
  // not starved searches.
  const bool hit_session_limit =
      !diag.budget_limit_reason.empty() &&
      diag.budget_limit_reason != "none";
  search_obj["budget_starved"] =
      hit_session_limit ||
      last_res.fallback_reason == "short_window_budget_exceeded" ||
      last_res.fallback_reason == "live_deadline_reached";

  std::lock_guard<std::mutex> lock(log_mutex);
  std::ofstream search_file(search_jsonl, std::ios::app);
  if (search_file) {
    search_file << open_spiel::json::ToString(search_obj, false) << "\n";
  }
}

void WriteCandidateDecisionTraceJsonl(
    const std::string& cand_traces_path,
    int episode_id,
    uint64_t matched_step_count,
    Player current_player,
    DuneDecisionRole role,
    const DunePUCTISMCTSBot* search_bot,
    bool search_this_role,
    const DuneSearchSession* cur_session,
    const DuneSearchResult& last_res,
    Action chosen_action,
    bool forced_swordmaster,
    const State& state,
    const std::array<SeatControllerProvenance, 4>& seat_provenance,
    double step_duration,
    std::mutex& log_mutex) {
  Action raw_prior_argmax = kInvalidAction;
  if (search_bot != nullptr && search_this_role) {
    raw_prior_argmax = search_bot->GetRootRawPriorArgmax();
  }
  if (raw_prior_argmax == kInvalidAction && cur_session != nullptr) {
    raw_prior_argmax = last_res.diagnostics.raw_reference_action;
  }
  if (raw_prior_argmax == kInvalidAction && last_res.diagnostics.raw_reference_action != kInvalidAction) {
    raw_prior_argmax = last_res.diagnostics.raw_reference_action;
  }
  if (raw_prior_argmax == kInvalidAction && !last_res.policy.empty()) {
    double max_p = -1.0;
    for (const auto& ap : last_res.policy) {
      if (ap.second > max_p) {
        max_p = ap.second;
        raw_prior_argmax = ap.first;
      }
    }
  }
  if (raw_prior_argmax == kInvalidAction) {
    auto legals = state.LegalActions();
    if (legals.size() == 1) {
      raw_prior_argmax = legals[0];
    }
  }

  bool is_greedy_match = (raw_prior_argmax != kInvalidAction) && (chosen_action == raw_prior_argmax);
  bool is_bypass = forced_swordmaster ||
                   (last_res.simulations_completed == 0 &&
                    (role == open_spiel::DuneDecisionRole::kForcedOrBookkeeping ||
                     role == open_spiel::DuneDecisionRole::kLeaderSelection ||
                     last_res.fallback_reason == "non_strategic_state" ||
                     state.LegalActions().size() <= 1 ||
                     !search_this_role));

  open_spiel::json::Object trace_obj;
  trace_obj["episode_id"] = static_cast<int64_t>(episode_id);
  trace_obj["decision_index"] = static_cast<int64_t>(matched_step_count);
  trace_obj["seat"] = static_cast<int64_t>(current_player);
  trace_obj["role"] = DuneDecisionRoleToString(role);
  trace_obj["evaluator"] = seat_provenance[current_player].controller;
  trace_obj["selected_action"] = static_cast<int64_t>(chosen_action);
  trace_obj["raw_prior_argmax"] = static_cast<int64_t>(raw_prior_argmax);
  trace_obj["is_greedy_match"] = is_greedy_match;
  trace_obj["simulations_completed"] = static_cast<int64_t>(last_res.simulations_completed);
  trace_obj["is_bypass"] = is_bypass;
  trace_obj["fallback_occurred"] = last_res.used_fallback;
  trace_obj["fallback_reason"] = last_res.fallback_reason;
  trace_obj["timed_out"] = last_res.timeout_status;
  trace_obj["selection_mode"] = last_res.diagnostics.selection_mode;
  trace_obj["selection_reason"] = last_res.diagnostics.selection_reason;
  trace_obj["search_accepted"] = last_res.diagnostics.search_accepted;
  trace_obj["incumbent_action"] = static_cast<int64_t>(last_res.diagnostics.incumbent_action);
  trace_obj["raw_prior_max_prob"] = last_res.diagnostics.raw_prior_max_prob;
  trace_obj["raw_prior_selected_prob"] = last_res.diagnostics.raw_prior_selected_prob;
  trace_obj["raw_prior_max_prob_hex"] = last_res.diagnostics.raw_prior_max_prob_hex;
  trace_obj["raw_prior_selected_prob_hex"] = last_res.diagnostics.raw_prior_selected_prob_hex;
  trace_obj["wall_time_ms"] = step_duration * 1000.0;

  std::lock_guard<std::mutex> lock(log_mutex);
  std::ofstream trace_file(cand_traces_path, std::ios::app);
  if (trace_file) {
    trace_file << open_spiel::json::ToString(trace_obj, false) << "\n";
  }
}

void WriteGameOutcomeJsonl(
    const std::string& game_jsonl,
    const dune_imperium::DuneImperiumState* dune_state,
    const GameOutcomeWriteContext& ctx,
    std::mutex& log_mutex) {
  open_spiel::json::Object game_obj;
  game_obj["episode_id"] = static_cast<int64_t>(ctx.episode_id);
  game_obj["seed"] = static_cast<int64_t>(ctx.game_seed);
  game_obj["search_seat"] = static_cast<int64_t>(ctx.search_seat);
  game_obj["winner"] = static_cast<int64_t>(ctx.winner);

  open_spiel::json::Array returns_arr;
  for (double r : *ctx.returns) {
    returns_arr.push_back(r);
  }
  game_obj["returns"] = returns_arr;
  game_obj["search_return"] = (*ctx.returns)[ctx.search_seat];
  game_obj["search_vp"] = ctx.search_vp;
  game_obj["policy_checkpoint"] = absl::GetFlag(FLAGS_model_checkpoint);
  game_obj["policy_checkpoint_sha256"] = open_spiel::g_candidate_ckpt_sha256;
  game_obj["value_checkpoint"] = absl::GetFlag(FLAGS_search_value_checkpoint).empty()
                                     ? absl::GetFlag(FLAGS_model_checkpoint)
                                     : absl::GetFlag(FLAGS_search_value_checkpoint);
  game_obj["value_checkpoint_sha256"] = open_spiel::g_candidate_value_ckpt_sha256;
  game_obj["opponent_checkpoint"] = absl::GetFlag(FLAGS_opponent_checkpoint);
  game_obj["opponent_checkpoint_sha256"] = open_spiel::g_opponent_ckpt_sha256;
  game_obj["is_split_evaluator"] = open_spiel::g_using_split_evaluator;
  game_obj["effective_search_mode"] =
      absl::GetFlag(FLAGS_audit_training_policy_iteration)
          ? "training_policy_iteration"
          : (absl::GetFlag(FLAGS_policy_only) ? "policy_only" : "other");

  open_spiel::json::Array opp_vps_arr;
  for (int p = 0; p < 4; ++p) {
    if (p != ctx.search_seat) {
      opp_vps_arr.push_back(static_cast<double>(GetTrueFinalVp(dune_state, p)));
    }
  }
  game_obj["opponent_vps"] = opp_vps_arr;

  game_obj["vp_margin"] = ctx.vp_margin;
  game_obj["rounds_played"] = static_cast<int64_t>(ctx.corrected_round);
  game_obj["search_steps"] = static_cast<int64_t>(ctx.search_steps_count);
  game_obj["search_swordmasters"] = dune_state->HasSwordmaster(ctx.search_seat);
  game_obj["search_seat_leader"] = static_cast<int64_t>(dune_state->PlayerLeader(ctx.search_seat));
  game_obj["candidate_leader"] = static_cast<int64_t>(dune_state->PlayerLeader(ctx.search_seat));
  open_spiel::json::Array leaders_arr;
  for (int p = 0; p < 4; ++p) {
    leaders_arr.push_back(static_cast<int64_t>(dune_state->PlayerLeader(p)));
  }
  game_obj["leaders"] = leaders_arr;
  game_obj["search_seat_swordmaster_round"] =
      static_cast<int64_t>((*ctx.sm_acquire_round_by_seat)[ctx.search_seat]);
  // Same quantity for all four seats, seat-indexed (-1 = never acquired).
  open_spiel::json::Array sm_rounds_arr;
  for (int p = 0; p < 4; ++p) {
    sm_rounds_arr.push_back(
        static_cast<int64_t>((*ctx.sm_acquire_round_by_seat)[p]));
  }
  game_obj["swordmaster_rounds"] = sm_rounds_arr;
  open_spiel::json::Array sbr_rounds, sbr_solari;
  for (const auto& kv : *ctx.search_seat_solari_by_round) {
    sbr_rounds.push_back(static_cast<int64_t>(kv.first));
    sbr_solari.push_back(static_cast<int64_t>(kv.second));
  }
  game_obj["search_seat_solari_rounds"] = sbr_rounds;
  game_obj["search_seat_solari_values"] = sbr_solari;
  // Third-agent utilization: per round, agents available (peak) vs placed
  // (deployed) by the searched seat. Iterate the availability map so every
  // observed round appears; placements default to 0 when none were used.
  open_spiel::json::Array spr_rounds, spr_placed, spr_avail;
  for (const auto& kv : *ctx.search_seat_agents_avail_by_round) {
    spr_rounds.push_back(static_cast<int64_t>(kv.first));
    spr_avail.push_back(static_cast<int64_t>(kv.second));
    auto it = ctx.search_seat_placements_by_round->find(kv.first);
    spr_placed.push_back(static_cast<int64_t>(
        it == ctx.search_seat_placements_by_round->end() ? 0 : it->second));
  }
  game_obj["search_seat_placement_rounds"] = spr_rounds;
  game_obj["search_seat_placements_used"] = spr_placed;
  game_obj["search_seat_agents_available"] = spr_avail;

  open_spiel::json::Array opp_sm_arr;
  for (int p = 0; p < 4; ++p) {
    if (p != ctx.search_seat) {
      opp_sm_arr.push_back(dune_state->HasSwordmaster(p));
    }
  }
  game_obj["opponent_swordmasters"] = opp_sm_arr;

  // --- Per-seat terminal VP decomposition -----------------------------
  // WHAT IS EXACT AND WHAT IS AN ATTRIBUTION. The engine keeps no
  // per-event VP ledger, so only the first split below is an identity:
  //
  //   final_scored_vp = vp_track + endgame_vp
  //
  // holds by construction -- FinalScoredVp is `vp_[p] + ComputeEndgameVp(p)`
  // (dune_imperium.cc:2403), so `endgame_vp` is taken as the DIFFERENCE
  // rather than recomputed. This file must never reimplement endgame
  // scoring; see dune_terminal_vp_report.h for why that rule exists.
  //
  // Everything else is an attribution of components *within* vp_track,
  // reconstructed from terminal state, and it is NOT forced to sum:
  //
  //  - conflict_vp is the engine's own cumulative counter, measured as the
  //    vp_ delta across conflict resolution (dune_imperium.cc:4744). Exact
  //    for conflict, but it absorbs any other vp_ change inside that same
  //    window.
  //  - alliance_vp counts alliances held AT THE END, +1 each. It is a
  //    snapshot, not a history: alliances change hands, and the loss path
  //    floors at zero (`vp_[old] = max(vp_[old] - 1, 0)`,
  //    dune_imperium_state_util.cc:361), so a seat that lost an alliance
  //    while at 0 VP never paid it back. This term can therefore both
  //    over- and under-state the true contribution.
  //  - tleilaxu_vp derives from the terminal track position: the scarab
  //    track grants gain_vp at positions 4 and 7 (kScarabTrackRewards) and
  //    only ever advances, so cumulative VP is a function of where it
  //    stopped.
  //
  // unattributed_track_vp is the REMAINDER, not a component -- card,
  // space, and leader effects that terminal state cannot separate. It may
  // legitimately be negative (see the alliance floor above). Do not
  // "correct" it.
  static const char* kFactionNames[4] = {"emperor", "spacing_guild",
                                         "bene_gesserit", "fremen"};
  open_spiel::json::Array vp_breakdown_arr;
  for (int p = 0; p < 4; ++p) {
    open_spiel::json::Object vb;
    const int final_vp = static_cast<int>(GetTrueFinalVp(dune_state, p));
    const int vp_track = dune_state->GetPlayerVp(p);
    const int conflict_vp = dune_state->ConflictVpDelta(p);

    open_spiel::json::Object alliances_obj;
    int alliance_vp = 0;
    for (int f = 0; f < 4; ++f) {
      // `HasAlliance(p, f)` is private; its public equivalent is this
      // comparison -- the method body is exactly
      // `alliance_owner_[faction] == player`
      // (dune_imperium_state_util.cc:889). The two differ only in the
      // out-of-range guard HasAlliance adds, and p and f are both loop
      // indices over [0, 4) here, so the guard can never fire.
      const bool held = dune_state->GetAllianceOwnerForTesting(f) == p;
      alliances_obj[kFactionNames[f]] = held;
      if (held) ++alliance_vp;
    }

    const int tl_track = dune_state->GetTleilaxuTrackForTesting(p);
    const int tleilaxu_vp = (tl_track >= 4 ? 1 : 0) + (tl_track >= 7 ? 1 : 0);
    (void)tleilaxu_vp;

    open_spiel::json::Array tech_arr;
    for (int tile : dune_state->GetPlayerTechTilesForTesting(p)) {
      tech_arr.push_back(static_cast<int64_t>(tile));
    }

    // --- the ledger: exact, complete, non-overlapping attribution -----
    // Every vp_ mutation is recorded at its source by
    // DuneImperiumState::AddVp, so these sum to vp_track EXACTLY, with no
    // residual. Emitted both as the raw event stream (round-by-round) and
    // as per-source totals.
    //
    // This supersedes the terminal-state reconstruction that used to live
    // here. That reconstruction could not work: it double-counted
    // (cumulative_conflict_vp_delta_ absorbs the friendship and alliance
    // grants that fire inside its own window) and it could not see the 12
    // sources that leave no terminal trace at all.
    open_spiel::json::Array ledger_arr;
    std::map<std::string, int> by_source;
    int ledger_sum = 0;
    for (const auto& ev : dune_state->GetVpEvents(p)) {
      open_spiel::json::Object e;
      e["round"] = static_cast<int64_t>(ev.round);
      e["source"] = std::string(dune_imperium::VpSourceName(ev.source));
      e["delta"] = static_cast<int64_t>(ev.delta);
      ledger_arr.push_back(e);
      by_source[dune_imperium::VpSourceName(ev.source)] += ev.delta;
      ledger_sum += ev.delta;
    }
    open_spiel::json::Object by_source_obj;
    for (const auto& kv : by_source) {
      by_source_obj[kv.first] = static_cast<int64_t>(kv.second);
    }

    vb["seat"] = static_cast<int64_t>(p);
    vb["final_scored_vp"] = static_cast<int64_t>(final_vp);
    vb["vp_track"] = static_cast<int64_t>(vp_track);
    vb["endgame_vp"] = static_cast<int64_t>(final_vp - vp_track);
    vb["vp_ledger"] = ledger_arr;
    vb["vp_by_source"] = by_source_obj;
    vb["vp_ledger_sum"] = static_cast<int64_t>(ledger_sum);
    // Must be 0. Non-zero means some site mutates vp_ without going
    // through AddVp -- a defect, not a measurement.
    vb["vp_ledger_residual"] = static_cast<int64_t>(vp_track - ledger_sum);
    // Terminal-state descriptors, kept for context. These are POSITIONS,
    // not attribution -- the ledger above is the attribution.
    vb["conflict_vp_counter"] = static_cast<int64_t>(conflict_vp);
    vb["alliances"] = alliances_obj;
    vb["alliances_held"] = static_cast<int64_t>(alliance_vp);
    vb["tleilaxu_track"] = static_cast<int64_t>(tl_track);
    vb["tech_tiles"] = tech_arr;
    vb["has_swordmaster"] = dune_state->HasSwordmaster(p);
    vb["leader"] = static_cast<int64_t>(dune_state->PlayerLeader(p));
    vp_breakdown_arr.push_back(vb);
  }
  game_obj["vp_breakdown"] = vp_breakdown_arr;

  game_obj["wall_time_s"] = ctx.game_duration_s;

  // --- WO-1 Phase 2: per-seat controller provenance -------------------
  // The candidate seat under analysis. Equal to search_seat in every
  // mode; emitted under its own name because in homogeneous mode
  // "the searched seat" is no longer a unique thing.
  game_obj["candidate_seat"] = static_cast<int64_t>(ctx.search_seat);
  game_obj["controller_mode"] = absl::GetFlag(FLAGS_controller_mode);
  open_spiel::json::Array seat_ctrl_arr;
  for (int p = 0; p < 4; ++p) {
    open_spiel::json::Object sc;
    sc["seat"] = static_cast<int64_t>(p);
    sc["controller"] = (*ctx.seat_provenance)[p].controller;
    sc["checkpoint"] = (*ctx.seat_provenance)[p].checkpoint;
    sc["checkpoint_sha256"] = (*ctx.seat_provenance)[p].checkpoint_sha256;
    sc["max_simulations"] =
        static_cast<int64_t>((*ctx.seat_provenance)[p].max_simulations);
    sc["budget_mode"] = (*ctx.seat_provenance)[p].budget_mode;
    sc["logit_cap"] = (*ctx.seat_provenance)[p].logit_cap;
    sc["is_candidate"] = (p == ctx.search_seat);
    seat_ctrl_arr.push_back(sc);
  }
  game_obj["seat_controllers"] = seat_ctrl_arr;

  // --- WO-1 Phase 3: pace/tempo ---------------------------------------
  int rounds_captured = 0;
  open_spiel::json::Array vp_by_round_arr;
  for (int rd = 1; rd <= kPaceMaxRounds; ++rd) {
    if (!(*ctx.round_end_seen)[rd]) break;
    open_spiel::json::Array one_round;
    for (int p = 0; p < 4; ++p) {
      one_round.push_back(static_cast<int64_t>((*ctx.vp_at_round_end)[rd][p]));
    }
    vp_by_round_arr.push_back(one_round);
    rounds_captured = rd;
  }
  game_obj["vp_end_by_round"] = vp_by_round_arr;

  open_spiel::json::Array first_ge11_arr;
  for (int p = 0; p < 4; ++p) {
    int first = -1;
    for (int rd = 1; rd <= rounds_captured; ++rd) {
      if ((*ctx.vp_at_round_end)[rd][p] >= dune_imperium::kTargetVp) { first = rd; break; }
    }
    // "never crossed" emits JSON null, matching dune_population_eval's
    // encoding of the same key exactly -- the two binaries' Phase 3
    // fields must be interchangeable for downstream analysis.
    if (first < 0) {
      first_ge11_arr.push_back(open_spiel::json::Null());
    } else {
      first_ge11_arr.push_back(static_cast<int64_t>(first));
    }
  }
  game_obj["first_round_vp_ge_11"] = first_ge11_arr;

  open_spiel::json::Array thr_set_arr;
  bool cand_in_thr = false;
  for (int p = 0; p < 4; ++p) {
    if (dune_state->GetPlayerVp(p) >= dune_imperium::kTargetVp) {
      thr_set_arr.push_back(static_cast<int64_t>(p));
      if (p == ctx.search_seat) cand_in_thr = true;
    }
  }
  game_obj["terminal_threshold_set"] = thr_set_arr;
  game_obj["candidate_in_terminal_threshold_set"] = cand_in_thr;
  // Mirrors the engine's end test (dune_imperium.cc:8891-8904):
  // end_game <- (round_ > 10) OR (any vp_[p] >= kTargetVp); threshold
  // wins when both hold. "other" is unreachable by construction.
  game_obj["terminal_reason"] =
      !thr_set_arr.empty()
          ? std::string("threshold")
          : (dune_state->GetCurrentRound() > kPaceMaxRounds
                 ? std::string("round_limit")
                 : std::string("other"));
  open_spiel::json::Array spec_arr;
  for (int p = 0; p < 4; ++p) {
    spec_arr.push_back(static_cast<int64_t>((*ctx.specimen_conversions)[p]));
  }
  game_obj["specimen_conversions"] = spec_arr;

  std::lock_guard<std::mutex> lock(log_mutex);
  std::ofstream game_file(game_jsonl, std::ios::app);
  if (game_file) {
    game_file << open_spiel::json::ToString(game_obj, false) << "\n";
  }
}

}  // namespace open_spiel
