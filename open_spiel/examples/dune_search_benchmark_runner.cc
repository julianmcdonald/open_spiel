#include "dune_search_benchmark_internal.h"

#include <cstdlib>
#include <iostream>
#include <limits>

#include "dune_pf2_matched_fallback.h"
#include "dune_seed_utils.h"
#include "dune_specimen_conversion.h"
#include "dune_warmstart_helpers.h"
#include "open_spiel/abseil-cpp/absl/flags/flag.h"
#include "open_spiel/abseil-cpp/absl/random/distributions.h"
#include "open_spiel/abseil-cpp/absl/strings/str_format.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {

// Raw-prior action selection for roles filtered OUT of fresh search in arm 4.
// Matches the policy-only arm's PickActionRespectingTemperature exactly: at
// temperature 0 the first strictly-max-prior action (argmax, first-max
// tie-break); at temperature > 0, an inverse-CDF sample from the prior. Empty
// prior degrades to the first legal action.
Action SelectRawPriorAction(open_spiel::algorithms::Evaluator& evaluator, const State& state,
                            double temperature, std::mt19937& rng) {
  ActionsAndProbs prior = evaluator.Prior(state);
  if (prior.empty()) {
    std::vector<Action> legals = state.LegalActions();
    return legals.empty() ? kInvalidAction : legals.front();
  }
  if (temperature == 0.0) {
    Action best = prior.front().first;
    double best_p = prior.front().second;
    for (const auto& ap : prior) {
      if (ap.second > best_p) {
        best_p = ap.second;
        best = ap.first;
      }
    }
    return best;
  }
  double r = absl::Uniform(rng, 0.0, 1.0);
  return SampleActionFromPrior(prior, r);
}

void WorkerThread(
    int thread_id,
    std::shared_ptr<const Game> game,
    std::shared_ptr<SharedDunePolicyValueNetImpl> search_model,
    std::shared_ptr<SharedDunePolicyValueNetImpl> search_value_model,
    std::shared_ptr<SharedDunePolicyValueNetImpl> opponent_model,
    std::atomic<int>& next_game_id,
    int total_games,
    std::atomic<int>& completed_games,
    std::mutex& log_mutex,
    GameStats& global_stats,
    std::mutex& stats_mutex,
    bool record_market_diagnostics,
    std::vector<dune_market_diag::CompactPerGameMarketDiagnostics>* market_results,
    std::shared_ptr<open_spiel::BatchedEvaluator> search_coord,
    std::shared_ptr<open_spiel::BatchedEvaluator> opp_coord) {
  torch::InferenceMode inference_guard;

  // Local thread stats accumulation to reduce lock contention
  GameStats thread_stats;

  // Arm-4 decomposition role filter (Path B only). Empty set = search every role.
  const std::set<DuneDecisionRole> fresh_search_roles =
      ParseFreshSearchRoles(absl::GetFlag(FLAGS_fresh_search_roles));
  // Swordmaster probe (arm B): rounds through which to force acquisition (0 = off).
  const int force_swordmaster_rounds = absl::GetFlag(FLAGS_force_swordmaster_rounds);
  // Swordmaster endowment probe (arm B-endow): round at which to grant the
  // searched seat a free Swordmaster (0 = off). Persistence is the engine's job.
  const int grant_swordmaster_round = absl::GetFlag(FLAGS_grant_swordmaster_round);

  // The per-move wall-clock budget this run is configured for. It is handed to
  // the session on every decision so that kLiveDeadline builds its clock from
  // what the operator actually asked for. Before this, the benchmark passed
  // nothing and the session substituted a hardcoded 52000.0, so
  // --relative_time_budget_ms was inert in live mode and the emitted
  // hard_time_limit_ms echoed 52000 regardless. Ignored outside live mode (the
  // session only reads it in the kLiveDeadline branch), so the same call site
  // serves every budget mode.
  const double configured_move_budget_ms =
      absl::GetFlag(FLAGS_disable_time_limit)
          ? std::numeric_limits<double>::infinity()
          : absl::GetFlag(FLAGS_relative_time_budget_ms);

  const SeatControllerMode controller_mode =
      ParseControllerMode(absl::GetFlag(FLAGS_controller_mode));
  const MixedOpponentController mixed_opponent =
      ParseMixedOpponentController(absl::GetFlag(FLAGS_mixed_opponent_controller));
  const float candidate_logit_cap =
      static_cast<float>(absl::GetFlag(FLAGS_candidate_logit_cap));
  const float opponent_logit_cap =
      static_cast<float>(absl::GetFlag(FLAGS_opponent_logit_cap));

  while (true) {
    int offset = next_game_id++;
    if (offset >= total_games) break;
    int g = absl::GetFlag(FLAGS_start_episode_id) + offset;

    uint64_t game_seed = dune_seed::DeriveSeed(absl::GetFlag(FLAGS_seed), dune_seed::kStreamBlueprint, g);
    std::mt19937 game_rng(game_seed);
    // Separate stream for the swordmaster-acquisition heuristic (arm B), so
    // forcing does not consume game_rng and the chance/opponent realizations stay
    // aligned with arm A until the game paths themselves diverge.
    std::mt19937 force_rng(dune_seed::DeriveSeed(absl::GetFlag(FLAGS_seed), dune_seed::kStreamSearchGate, g));
    auto game_start_time = std::chrono::steady_clock::now();

    // PF-2 Part B: the matched arm's decision ordinal. Mirrors Path-B's
    // DunePUCTISMCTSBot::search_count_ EXACTLY (dune_puct_is_mcts.cc:1179-1180):
    // zero per game/bot, +1 on EVERY candidate decision — single-legal,
    // strategic and fallback alike — and incremented BEFORE the seed is
    // derived, so the first decision derives with 1, not 0. It is a plain local,
    // so nothing resets it mid-game (Path-B's is likewise untouched by tree
    // Reset()). An implementation that only counted eligible fallbacks would
    // pass every per-decision tuple test and still desynchronise the stream —
    // which is exactly what the ordinal sequence test exists to catch.
    uint64_t matched_step_count = 0;

    bool rotate_seat = absl::GetFlag(FLAGS_rotate_seat);
    int search_seat = rotate_seat ? (g % 4) : 0;
    thread_stats.games_by_seat[search_seat]++;

    torch::Device device = torch::cuda::is_available() ? torch::Device(torch::kCUDA) : torch::Device(torch::kCPU);
    // Create thread-local, bot-specific evaluator wrapping search models
    std::shared_ptr<open_spiel::algorithms::Evaluator> search_evaluator =
        BuildSearchEvaluator(search_model, search_value_model, search_coord,
                             device, candidate_logit_cap);

    GameSeatSetup seat_setup = SetupGameSeats(
        game_rng, controller_mode, mixed_opponent, search_seat,
        candidate_logit_cap, opponent_logit_cap, device, search_evaluator,
        opponent_model, opp_coord);

    std::unique_ptr<State> state = game->NewInitialState();
    int game_length = 0;
    const auto* dune_state_for_diag =
        dynamic_cast<const dune_imperium::DuneImperiumState*>(state.get());

    dune_market_diag::GameMarketTracker market_tracker;
    if (record_market_diagnostics) {
      market_tracker.InitGame(g, search_seat);
    }

    std::vector<int> game_role_counts(7, 0);
    TurnAuditState audit_state;

    // Probe telemetry (3b): the Swordmaster-acquisition round and the searched
    // seat's solari trajectory by round. Forced funding/acquisition decisions
    // bypass the search log, so this in-loop sampling is the only place they
    // surface. Acquisition is now sampled for ALL FOUR seats (live-eval
    // telemetry): opponents acquire Swordmaster too, and the round they do it
    // is not recoverable from terminal state -- HasSwordmaster is a terminal
    // bool. -1 means never acquired. The searched seat's entry is what the
    // legacy scalar `search_seat_swordmaster_round` reports, so the two can
    // never disagree.
    std::array<int, 4> sm_acquire_round_by_seat;
    sm_acquire_round_by_seat.fill(-1);
    std::map<int, int> search_seat_solari_by_round;
    // Third-agent utilization (arm B-endow diagnostic): per round, the agents the
    // searched seat DEPLOYED (counted as agents_remaining decrements) and its peak
    // available agents that round. Utilization = placements / available; a
    // stranded third agent shows up as placements < available.
    std::map<int, int> search_seat_placements_by_round;
    std::map<int, int> search_seat_agents_avail_by_round;
    int prev_search_agents = -1;

    // --- WO-1 Phase 3: pace/tempo round-end VP tracking --------------------
    // Same edge-detected round-boundary snapshot dune_population_eval uses:
    // GetCurrentRound() is bumped before the terminal check, so observing
    // cur == N+1 means round N just ended; read every seat's track VP then.
    std::array<std::array<int, 4>, kPaceMaxRounds + 2> vp_at_round_end{};
    std::array<bool, kPaceMaxRounds + 2> round_end_seen{};
    std::array<int, 4> specimen_conversions{};
    int pace_last_round = -1;
    {
      const auto* s0 = dynamic_cast<const dune_imperium::DuneImperiumState*>(state.get());
      pace_last_round = s0 ? s0->GetCurrentRound() : -1;
    }
    auto snapshot_round_ends = [&]() {
      const auto* ds = dynamic_cast<const dune_imperium::DuneImperiumState*>(state.get());
      if (ds == nullptr) return;
      int cur = ds->GetCurrentRound();
      while (pace_last_round >= 0 && pace_last_round < cur) {
        if (pace_last_round >= 1 && pace_last_round <= kPaceMaxRounds) {
          for (int p = 0; p < 4; ++p) {
            vp_at_round_end[pace_last_round][p] = ds->GetPlayerVp(p);
          }
          round_end_seen[pace_last_round] = true;
        }
        ++pace_last_round;
      }
    };

    while (!state->IsTerminal()) {
      snapshot_round_ends();
      Player current_player = state->CurrentPlayer();
      if (const auto* trk = dynamic_cast<const dune_imperium::DuneImperiumState*>(state.get())) {
        const int trk_round = trk->GetCurrentRound();
        search_seat_solari_by_round[trk_round] = trk->GetPlayerSolari(search_seat);
        for (int p = 0; p < 4; ++p) {
          if (sm_acquire_round_by_seat[p] < 0 && trk->HasSwordmaster(p)) {
            sm_acquire_round_by_seat[p] = trk_round;
          }
        }
        // Per-round third-agent utilization: track peak available agents and
        // count deployments as decrements in agents_remaining. A grant/round
        // reset raises the count (not a deployment) and is ignored here.
        const int cur_agents =
            trk->GetPlayerAgentsRemainingForTesting(search_seat);
        if (cur_agents > search_seat_agents_avail_by_round[trk_round]) {
          search_seat_agents_avail_by_round[trk_round] = cur_agents;
        }
        if (prev_search_agents >= 0 && cur_agents < prev_search_agents) {
          search_seat_placements_by_round[trk_round] +=
              (prev_search_agents - cur_agents);
        }
        prev_search_agents = cur_agents;
      }
      // Swordmaster endowment (arm B-endow): grant the searched seat a free
      // Swordmaster at its FIRST decision of the grant round, before it places
      // any agent that round. Fires once (HasSwordmaster gates re-entry);
      // persistence across later rounds is the engine's permanent third agent.
      // No solari deducted, no action forced.
      if (grant_swordmaster_round > 0 && current_player == search_seat) {
        if (auto* gr_state =
                dynamic_cast<dune_imperium::DuneImperiumState*>(state.get())) {
          if (gr_state->GetCurrentRound() == grant_swordmaster_round &&
              !gr_state->HasSwordmaster(search_seat)) {
            gr_state->SetSwordmasterForTesting(search_seat, true);
          }
        }
      }
      if (audit_state.in_search_turn && current_player != search_seat && current_player != kChancePlayerId) {
        std::cout << "--- End of Agent Turn Audit Summary ---\n"
                  << "Decisions: " << audit_state.turn_decisions
                  << " (Primary: 1, Continuations: " << (audit_state.turn_decisions - 1) << ")\n"
                  << "Total simulations completed: " << (audit_state.turn_primary_sims + audit_state.turn_continuation_sims)
                  << " (Primary: " << audit_state.turn_primary_sims << ", Continuation: " << audit_state.turn_continuation_sims << ")\n"
                  << "Total wall-clock time: " << audit_state.turn_wall_time_ms << " ms\n"
                  << "Timeout status: " << (audit_state.turn_had_timeout ? "Yes" : "No") << "\n"
                  << "Fallback status: " << (audit_state.turn_had_fallback ? "Yes" : "No")
                  << " (Reason: " << audit_state.turn_fallback_reason << ")\n"
                  << "---------------------------------------\n";
        audit_state.in_search_turn = false;
      }
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
          action = SampleAction(outcomes, game_rng).first;
        }
        state->ApplyAction(action);
        continue;
      }

      if (record_market_diagnostics && dune_state_for_diag != nullptr && current_player == search_seat) {
        market_tracker.OnCandidateDecision(*dune_state_for_diag, search_seat, state->LegalActions());
      }

      Action chosen_action = -1;
      open_spiel::DuneDecisionRole role = open_spiel::DuneDecisionRole::kForcedOrBookkeeping;
      bool search_this_role = false;
      double step_duration = 0.0;
      DuneSearchResult last_res;

      // Swordmaster probe (arm B): before the searched seat searches, force it
      // toward legal Swordmaster acquisition in the early rounds. Bypasses the
      // search machinery entirely (no simulations, no search-stats pollution).
      bool forced_swordmaster = false;
      if (force_swordmaster_rounds > 0 && current_player == search_seat) {
        const auto* sm_state =
            dynamic_cast<const dune_imperium::DuneImperiumState*>(state.get());
        if (sm_state != nullptr &&
            sm_state->GetCurrentRound() <= force_swordmaster_rounds &&
            !sm_state->HasSwordmaster(search_seat)) {
          // Probe validity: force acquisition ONLY at decisions where a legal
          // swordmaster/smuggling/shipping move exists. When none does the helper
          // returns kInvalidAction and we fall through to the normal search path
          // below (forced_swordmaster stays false). A uniform-random fallback here
          // would randomize purchases/combat/reveals in rounds 1-3 (the 18A
          // uniform-policy poison) and misattribute the loss to swordmaster.
          chosen_action = ChooseHeuristicAcquisitionAction(
              *state, state->LegalActions(), search_seat, &force_rng,
              /*return_invalid_on_no_acquisition=*/true);
          forced_swordmaster = (chosen_action != kInvalidAction);
        }
      }

      if (forced_swordmaster) {
        // chosen_action already set by the acquisition heuristic; skip search.
      } else if (seat_setup.seat_is_searched[current_player]) {
        // In homogeneous mode every seat lands here; `cur_session` is that seat's
        // own session, never a shared one.
        auto& cur_session = seat_setup.seat_sessions[current_player];
        bool has_active = cur_session ? cur_session->HasActiveSession() : false;
        role = open_spiel::ClassifyDuneDecisionRole(*state, current_player, has_active);
        game_role_counts[static_cast<int>(role)]++;
        bool is_strategic = (role == open_spiel::DuneDecisionRole::kAgentPrimary ||
                             role == open_spiel::DuneDecisionRole::kAgentContinuation);
        auto step_start = std::chrono::steady_clock::now();
        // PF-2 Part B ordinal. Incremented here — once per candidate decision,
        // before either branch and before any seed derivation — so it counts
        // exactly what Path-B's search_count_ counts. Unused when the flag is
        // off; an unread increment has no behavioural effect, which is what
        // keeps the flag-OFF projection identical.
        ++matched_step_count;
        bool matched_sampled_this_decision = false;
        if (cur_session && absl::GetFlag(FLAGS_matched_fallback_sampling)) {
          // PF-2 Part B, TWO-PHASE lifecycle. SearchAndSelectWithDeadline is
          // Search -> SelectControllerAction -> CommitAction
          // (dune_search_session.cc:849-856), and CommitAction PERSISTS the
          // decision into the session history. Overriding the returned action
          // afterwards would make the session's history record the argmax while
          // the game plays the sampled action — the two desynchronise on
          // exactly the decisions this arm exists to change, and every
          // subsequent re-root would key off a history that never happened.
          // The override therefore has to land on the ControllerDecision BEFORE
          // it is committed.
          //
          // r_val = 0.0 is provably inert here rather than merely unused:
          // SelectControllerAction feeds r_val only to
          // PickActionRespectingTemperature, which returns argmax without
          // reading it when config_.temperature == 0.0, and the session's rng_
          // is consumed nowhere except the two SearchAndSelect* wrappers this
          // branch bypasses. The fatal guard in main() enforces
          // --temperature=0, so the precondition is checked, not assumed.
          DuneSearchResult searched =
              cur_session->Search(*state, configured_move_budget_ms);
          ControllerDecision dec =
              cur_session->SelectControllerAction(*state, searched, 0.0);
          // PF-2 Part B. Classification calls IsStrategicState DIRECTLY on this
          // arm's OWN state. It deliberately does NOT reuse the session's role
          // routing (dune_search_session.cc:331 forced/leader, :347
          // purchase/combat — the latter live here via
          // --purchase_combat_budget=0): those exits sit exactly at the classes
          // where the role notion diverges from IsStrategicState, so making
          // them sample is the natural and WRONG implementation. Purchase and
          // combat-intrigue decisions are STRATEGIC under the gate that
          // actually produces Path-B's fallbacks, and non-strategic under the
          // roles; sampling there would put the matched arm at decisions the
          // searched arm searches.
          //
          // Two clauses, not one: RunSearch() returns early on single-legal
          // BEFORE the strategic branch, so single-legal decisions never
          // produce a fallback. !IsStrategicState alone returns true for them
          // (the predicate is false below 2 legals) and would sweep in the
          // forced actions — inflating the class roughly eightfold and letting
          // a count check pass on decisions that were never choices.
          //
          // No searched-arm output is read anywhere here: the class comes from
          // (state, player) alone, so the arms share a RULE, never a decision.
          // Every rule below is the SHARED one from
          // dune_pf2_matched_fallback.h. It is deliberately not written out
          // here: when these were inline, the conformance tests could only
          // re-express them, and a test of a re-expression says nothing about
          // this binary (Sol's review, 2026-08-01). The tests now call these
          // exact functions.
          const bool eligible =
              open_spiel::pf2::MatchedFallbackEligible(*state, current_player);
          if (eligible && !searched.policy.empty()) {
            const uint64_t step_seed = open_spiel::pf2::MatchedFallbackStepSeed(
                seat_setup.seat_config_seed[current_player], matched_step_count);
            const double r_val = open_spiel::pf2::MatchedFallbackRVal(step_seed);
            dec.selected_action =
                open_spiel::pf2::MatchedFallbackSelect(searched.policy, r_val);
            matched_sampled_this_decision = true;
          }
          // Commit AFTER the override: the session's persisted history and the
          // action the game plays are now the same action by construction.
          last_res = cur_session->CommitAction(dec);
          chosen_action = last_res.diagnostics.selected_action;
        } else if (cur_session) {
          // Flag OFF: byte-identical to the pre-change binary. The two-phase
          // path above is never entered, so the OFF-flag neutrality projection
          // is unaffected by Part B entirely.
          last_res = cur_session->SearchAndSelectWithDeadline(
              *state, configured_move_budget_ms);
          chosen_action = last_res.diagnostics.selected_action;
        } else {
          // Path B: fresh full search per decision, sample-on-fallback (verbatim
          // July-14 reference selection). GetLastSearchResult exposes diagnostics.
          // Arm-4 role filter: with a non-empty --fresh_search_roles set, only the
          // listed roles are searched; every other searched-seat decision plays the
          // raw-prior argmax, identical to the policy-only arm. This decomposes the
          // full-search arm into "policy-only + search at the listed roles".
          search_this_role =
              fresh_search_roles.empty() || fresh_search_roles.count(role) > 0;
          if (search_this_role) {
            chosen_action = seat_setup.search_bot->Step(*state);
            last_res = seat_setup.search_bot->GetLastSearchResult();
            last_res.diagnostics.selected_action = chosen_action;
          } else {
            DuneActionSelectionMode sel_mode =
                ParseDuneActionSelectionMode(absl::GetFlag(FLAGS_action_selection_mode));
            double temp = absl::GetFlag(FLAGS_temperature);
            if (sel_mode == DuneActionSelectionMode::kPreserveRawGreedy && temp == 0.0) {
              chosen_action = SelectRawPriorAction(
                  *seat_setup.seat_evaluators[current_player], *state, 0.0, game_rng);
            } else {
              chosen_action = SelectRawPriorAction(
                  *seat_setup.seat_evaluators[current_player], *state, temp, game_rng);
            }
            last_res = DuneSearchResult();
            last_res.simulations_completed = 0;
            last_res.used_fallback = true;
            last_res.fallback_reason = "role_filtered_raw_prior";
            last_res.diagnostics.selected_action = chosen_action;
            ActionsAndProbs prior = seat_setup.seat_evaluators[current_player]->Prior(*state);
            Action prior_argmax = kInvalidAction;
            double max_p = -1.0;
            double chosen_p = 0.0;
            for (const auto& ap : prior) {
              if (ap.second > max_p) {
                max_p = ap.second;
                prior_argmax = ap.first;
              }
              if (ap.first == chosen_action) {
                chosen_p = ap.second;
              }
            }
            last_res.diagnostics.raw_reference_action = prior_argmax;
            last_res.diagnostics.decision_role = std::to_string(static_cast<int>(role));
            last_res.diagnostics.incumbent_action = prior_argmax;
            last_res.diagnostics.raw_prior_max_prob = (max_p >= 0.0) ? max_p : 0.0;
            last_res.diagnostics.raw_prior_selected_prob = chosen_p;
            last_res.diagnostics.raw_prior_max_prob_hex =
                absl::StrFormat("%a", last_res.diagnostics.raw_prior_max_prob);
            last_res.diagnostics.raw_prior_selected_prob_hex =
                absl::StrFormat("%a", chosen_p);
            last_res.diagnostics.selection_mode = DuneActionSelectionModeToString(sel_mode);
            last_res.diagnostics.selection_reason = "role_filter_bypass";
            last_res.diagnostics.search_accepted = false;
          }
          // PF-2 Part A. Path B emits the session-populated diagnostic fields,
          // but only dune_search_session.cc:501 ever writes them, so on the
          // fresh path they stay blank/-1 (13,395 of 26,147 historical
          // `fallback_reason=="none"` rows carry round -1 and an empty role).
          // The values are already computed locally at :743; this copies them
          // onto the record that gets serialized. Telemetry only: no
          // control-flow, RNG, or action-selection effect.
          //
          // Scoped to the Path-B branch DELIBERATELY. Hoisting it past the
          // enclosing if/else would overwrite the session path's own values
          // and break the policy-only half of the flag-OFF neutrality proof.
          // Shared with the field regression test (dune_pf2_matched_fallback.h)
          // so the test exercises this population, not a copy of it.
          open_spiel::pf2::PopulatePathBDiagnostics(
              last_res.diagnostics, *state, static_cast<int>(role));
        }
        auto step_end = std::chrono::steady_clock::now();
        step_duration = std::chrono::duration<double>(step_end - step_start).count();

        UpdateStepSearchStats(role, is_strategic, step_duration, last_res,
                              cur_session.get(), audit_state, thread_stats);

        std::string search_jsonl = absl::GetFlag(FLAGS_search_jsonl_path);
        if (!search_jsonl.empty()) {
          WriteSearchDecisionJsonl(
              search_jsonl, g, matched_step_count, game_seed, search_seat,
              current_player, last_res, chosen_action, *state,
              seat_setup.seat_provenance, matched_sampled_this_decision,
              log_mutex);
        }
      } else {
        chosen_action = seat_setup.bots[current_player]->Step(*state);
      }

      std::string cand_traces_path = absl::GetFlag(FLAGS_candidate_decision_traces_path);
      if (!cand_traces_path.empty() && current_player == search_seat) {
        WriteCandidateDecisionTraceJsonl(
            cand_traces_path, g, matched_step_count, current_player, role,
            seat_setup.search_bot.get(), search_this_role,
            seat_setup.seat_sessions[current_player].get(), last_res,
            chosen_action, forced_swordmaster, *state,
            seat_setup.seat_provenance, step_duration, log_mutex);
      }

      // WO-1 Phase 3 addendum: count applied specimen->troop conversions per
      // seat, at application time, by engine action-ID range. Measurement only
      // -- no shaping is enabled anywhere by this counter.
      //
      // PWO-5 gate 2 item (c): the range is 741-752, not 740-752. Numerically a
      // no-op on every committed run -- 740 is an unused base constant, absent
      // from all 10,903 conversion-legal rows of the frozen PWO-4 stream -- so
      // no committed conversion count moves. See dune_specimen_conversion.h.
      if (dune_shaping::IsSpecimenConversionAction(chosen_action) &&
          current_player >= 0 && current_player < 4) {
        ++specimen_conversions[current_player];
      }
      if (record_market_diagnostics && dune_state_for_diag != nullptr) {
        market_tracker.BeforeApplyAction(*dune_state_for_diag, current_player, search_seat, chosen_action);
      }
      state->ApplyAction(chosen_action);
      if (record_market_diagnostics && dune_state_for_diag != nullptr) {
        market_tracker.AfterApplyAction(*dune_state_for_diag, current_player, search_seat, chosen_action);
      }
    }

    if (record_market_diagnostics && market_results != nullptr &&
        offset < static_cast<int>(market_results->size())) {
      (*market_results)[offset] = market_tracker.GetData();
    }

    // Capture the terminal round's VP snapshot (the loop exits before it).
    snapshot_round_ends();

    if (audit_state.in_search_turn) {
      std::cout << "--- End of Agent Turn Audit Summary ---\n"
                << "Decisions: " << audit_state.turn_decisions
                << " (Primary: 1, Continuations: " << (audit_state.turn_decisions - 1) << ")\n"
                << "Total simulations completed: " << (audit_state.turn_primary_sims + audit_state.turn_continuation_sims)
                << " (Primary: " << audit_state.turn_primary_sims << ", Continuation: " << audit_state.turn_continuation_sims << ")\n"
                << "Total wall-clock time: " << audit_state.turn_wall_time_ms << " ms\n"
                << "Timeout status: " << (audit_state.turn_had_timeout ? "Yes" : "No") << "\n"
                << "Fallback status: " << (audit_state.turn_had_fallback ? "Yes" : "No")
                << " (Reason: " << audit_state.turn_fallback_reason << ")\n"
                << "---------------------------------------\n";
    }

    std::cout << "=== Game " << g << " Role Counts ===\n"
              << "kForcedOrBookkeeping: " << game_role_counts[0] << "\n"
              << "kLeaderSelection: " << game_role_counts[1] << "\n"
              << "kAgentPrimary: " << game_role_counts[2] << "\n"
              << "kAgentContinuation: " << game_role_counts[3] << "\n"
              << "kPurchase: " << game_role_counts[4] << "\n"
              << "kCombatIntrigue: " << game_role_counts[5] << "\n"
              << "kOtherOptional: " << game_role_counts[6] << "\n"
              << "========================\n";

    const auto* dune_state = dynamic_cast<const dune_imperium::DuneImperiumState*>(state.get());
    int winner = -1;
    double search_vp = 0.0;
    double margin = 0.0;
    int corrected_round = 0;
    RecordTerminalGameStats(*state, dune_state, search_seat, thread_stats,
                            &winner, &search_vp, &margin, &corrected_round);

    if (dune_state != nullptr) {
      if (g == 67) {
        PrintEpisode67Breakdown(dune_state);
      }

      double game_duration_s = std::chrono::duration<double>(
                                   std::chrono::steady_clock::now() - game_start_time).count();
      std::string game_jsonl = absl::GetFlag(FLAGS_game_jsonl_path);
      if (!game_jsonl.empty()) {
        std::vector<double> returns = state->Returns();
        GameOutcomeWriteContext ctx;
        ctx.episode_id = g;
        ctx.game_seed = game_seed;
        ctx.search_seat = search_seat;
        ctx.winner = winner;
        ctx.returns = &returns;
        ctx.search_vp = search_vp;
        ctx.vp_margin = margin;
        ctx.corrected_round = corrected_round;
        ctx.search_steps_count = thread_stats.search_steps_count;
        ctx.game_duration_s = game_duration_s;
        ctx.sm_acquire_round_by_seat = &sm_acquire_round_by_seat;
        ctx.search_seat_solari_by_round = &search_seat_solari_by_round;
        ctx.search_seat_placements_by_round = &search_seat_placements_by_round;
        ctx.search_seat_agents_avail_by_round = &search_seat_agents_avail_by_round;
        ctx.seat_provenance = &seat_setup.seat_provenance;
        ctx.vp_at_round_end = &vp_at_round_end;
        ctx.round_end_seen = &round_end_seen;
        ctx.specimen_conversions = &specimen_conversions;
        WriteGameOutcomeJsonl(game_jsonl, dune_state, ctx, log_mutex);
      }
    }

    // Flush stats and perform intermediate logging under a single lock
    {
      std::lock_guard<std::mutex> stats_lock(stats_mutex);
      MergeGameStats(global_stats, thread_stats, &game_role_counts);

      int comp = ++completed_games;
      if (comp % 25 == 0 || comp == total_games) {
        std::lock_guard<std::mutex> log_lock(log_mutex);
        double avg_step_time = global_stats.search_steps_count > 0 ? (global_stats.search_step_time_sum / global_stats.search_steps_count) : 0.0;
        double winrate = (global_stats.search_wins * 100.0) / comp;
        double search_sm_rate = (global_stats.search_swordmasters * 100.0) / comp;
        double opp_sm_rate = (global_stats.opponent_swordmasters * 100.0) / (3.0 * comp);

        std::cout << absl::StrFormat("[Progress] Games: %d/%d | Search Winrate: %.2f%% | Search SM Rate: %.2f%% | Opp SM Rate: %.2f%% | Avg Search Step Time: %.4fs\n",
                                     comp, total_games, winrate, search_sm_rate, opp_sm_rate, avg_step_time) << std::flush;
      }
    }
    thread_stats = GameStats();
  }

  // Merge remaining stats on completion
  {
    std::lock_guard<std::mutex> stats_lock(stats_mutex);
    MergeGameStats(global_stats, thread_stats);
  }
}

}  // namespace open_spiel
