// Agent-turn search policy iteration (search-PI): a genuinely separate
// policy-iteration lane.
//
// WHAT THIS IS NOT. It is not the Phase 18B online auxiliary collector
// (dune_online_search_collector.{h,cc}) and it does not touch it. That mode --
// PF-C -- kept an ordinary search-free PPO rollout and added
// 0.10 * (CE + 0.5 * value_MSE) from ~168-198 examples generated in SEPARATE
// auxiliary games, backpropagated alongside the PPO objective. The combined
// auxiliary gradient reached ~0.69 of the PPO gradient and the policy flattened
// (docs/PF_C_SEARCH_TEACHER_DIAGNOSIS_2026_08_15.md sections 1-3). That result is
// closed-negative and nothing here revises it.
//
// WHAT THIS IS. An AlphaZero/KataGo-shaped loop in which
//   1. search generates the behaviour trajectory,
//   2. the search visit policy is the first-class policy target,
//   3. terminal results FROM THAT SEARCH-GUIDED TRAJECTORY train the value head,
//   4. no PPO policy objective is applied to these rows at all.
// One generation is a clean frozen-collect -> learn -> sync boundary.
//
// SCOPE. Search runs at DuneDecisionRole::kAgentPrimary and
// kAgentContinuation only. Leader draft, purchase, combat-intrigue and
// other-optional decisions run ZERO new training-search simulations, by
// construction (see SearchPiSearchConfigFor()). Inference-time and play-time
// search behaviour is untouched.

#ifndef OPEN_SPIEL_EXAMPLES_DUNE_SEARCH_PI_H_
#define OPEN_SPIEL_EXAMPLES_DUNE_SEARCH_PI_H_

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "open_spiel/spiel.h"
#include "open_spiel/utils/json.h"
#include "dune_network.h"
#include "dune_puct_is_mcts.h"      // DuneSearchConfig, DuneSearchResult
#include "dune_search_routing.h"    // DuneDecisionRole
#include "dune_search_session.h"    // DuneSearchSession, DuneSearchBudgetMode

namespace open_spiel {

namespace algorithms {
class Evaluator;
}  // namespace algorithms

// ---------------------------------------------------------------------------
// Fallback taxonomy
// ---------------------------------------------------------------------------
//
// The PF-C collector treated its coverage/acceptance heuristic as a BEHAVIOUR
// SWITCH: a search covering fewer than min(3, legal) actions at >= 2 visits was
// discarded and the raw policy was played and taught instead
// (dune_online_search_collector.cc:673-749). A concentrated search -- exactly
// the case where the teacher is most confident -- was therefore thrown away.
//
// In this lane coverage is TELEMETRY. The only reasons to abandon a search
// result are genuine technical failures, enumerated here. Anything not on this
// list is executed and labelled.
enum class SearchPiFallback {
  kNone = 0,               // usable search; target emitted
  kNoSearchRun,            // role is not a search target (not a failure)
  kZeroVisits,             // search produced no root visits at all
  kInvalidAlignment,       // actions/visits/priors vectors disagree in length
  kIllegalAction,          // selected action is not legal at this state
  kTimeoutBeforeUsefulSearch,  // wall/deadline expiry with nothing usable
  kEmptyPolicy,            // search returned an empty action set
};

// Stable, log/JSON-safe names. PERSISTED in row telemetry -- do not rename.
const char* SearchPiFallbackName(SearchPiFallback reason);

// ---------------------------------------------------------------------------
// Early-exit taxonomy -- ORTHOGONAL to the fallback taxonomy above
// ---------------------------------------------------------------------------
//
// The simulation loop has three early exits (dune_puct_is_mcts.cc:982-1005):
// the deadline break, the max_nodes break, and a simulation that returned
// nothing. All three leave a USABLE tree, and ClassifySearchPiResult returns
// kNone for any search with at least one root visit -- so a truncated search
// emits a perfectly normal row and reports zero fallbacks. Only budget
// arithmetic could detect it.
//
// That is exactly the instrument gap rung 2's Amendment 1 recorded: arm D
// generation 4 completed 86,971 of 87,000 configured primary simulations,
// which the gate caught as a count, but the artifacts could not say which exit
// fired or at which root. This enum is the attribution the artifacts lacked.
//
// It classifies the STOP, never the row's fate. Nothing here suppresses a
// target, changes an executed action, or turns a usable search into a fallback:
// a short search still teaches, exactly as it did before, and now it announces
// itself.
enum class SearchPiEarlyExit {
  kNone = 0,           // spent its full configured budget
  kNoSearchRun,        // no budget was requested (an off-scope role)
  kTimeout,            // deadline break, or a simulation that returned nothing
  kMaxNodes,           // node-pool cap (DuneSearchConfig::max_nodes)
  kSingleLegalAction,  // forced root: RunSearch returns before the loop
  kEmptyOrZeroVisits,  // ran, produced no usable root visits
  kSessionBudget,      // a session-level budget clamp reported the truncation
  kUnclassified,       // SHORT with no recognised cause -- an instrument defect
};
inline constexpr int kSearchPiEarlyExitCount = 8;

// Stable, log/JSON-safe names. PERSISTED in telemetry -- do not rename.
const char* SearchPiEarlyExitName(SearchPiEarlyExit reason);

// Classify one completed search against the budget it was ASKED to spend.
// `requested_simulations` is the session's own `soft_sim_limit`, i.e. the count
// it handed RunSearch -- not the lane's configured budget. Reading the
// session's number is what lets a disagreement between configuration and
// session show up as a classified early exit instead of being papered over by
// re-deriving the expectation from the same config that produced it.
SearchPiEarlyExit ClassifySearchPiEarlyExit(
    const DuneSearchResult& result, const std::vector<Action>& legal_actions,
    int requested_simulations);

// ---------------------------------------------------------------------------
// Which controller drives the searched seat
// ---------------------------------------------------------------------------
//
// The teacher audit (rung-3 §7a) needs the strength of the SEARCHED controller
// measured against the raw policy on the same protocol, and the repository had
// no evaluation path that measured it: dune_search_benchmark spends a
// cumulative session budget and starves continuations, and dune_population_eval
// is the raw controller with no session at all. Both measure a different
// controller than collection measures.
//
// So the audit runs the collection generator itself, and the only thing this
// selects is which action the searched seat EXECUTES at kAgentPrimary and
// kAgentContinuation. Everything else -- session activation and re-rooting,
// role classification, the searched seat's off-scope decisions, the opponents,
// the chance stream and every seed derivation -- is shared between the two arms
// by construction rather than by care.
//
// Deliberately NOT a SearchPiConfig field: the config is fingerprinted into
// every checkpoint manifest and a resume compares that fingerprint, so adding a
// field here would invalidate every artifact rung 1 and rung 2 recorded. It is
// a per-call argument whose default is the collection controller.
enum class SearchPiArm {
  kSearched = 0,  // the collection controller: the search picks the action
  kRawArgmax,     // the matched control: raw-prior argmax, zero simulations
};
const char* SearchPiArmName(SearchPiArm arm);

// Which root visit vector a continuation's target normalizes.
//
// After a re-root the root's child visit counts include visits INHERITED from
// the previous search in the same placement activation. Two defensible
// conventions exist and the choice must be recorded, not left implicit:
//   kTotalVisits  -- normalize inherited + new (the tree-reuse convention, and
//                    what AlphaZero/KataGo do with a reused subtree). DEFAULT.
//   kNewVisitsOnly-- normalize (visit_counts_final - visit_counts_pre_search),
//                    i.e. only what THIS search added.
// Both are computable because the PWO-4 Amendment 2 retention snapshots are
// live on the session path (dune_puct_is_mcts.h:282-300). Rows carry the
// inherited counts either way, so a pilot can measure the difference later.
enum class SearchPiContinuationTarget {
  kTotalVisits = 0,
  kNewVisitsOnly = 1,
};

const char* SearchPiContinuationTargetName(SearchPiContinuationTarget mode);
bool ParseSearchPiContinuationTarget(const std::string& name,
                                     SearchPiContinuationTarget* out);

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------
//
// Deliberately NOT derived from OnlineSearchConfig and deliberately NOT read
// from calibration_results_v2/protocol_manifest.json: LoadCalibratedParameters
// (dune_search_session.cc:582) mutates puct_c and the root prior temperature
// from a file that changes underneath a run. Every field here is explicit,
// flag-settable and fingerprinted into the checkpoint manifest.
struct SearchPiConfig {
  // --- Generation shape ---
  int games_per_generation = 16;   // multiple of 4 (seat balance)
  int64_t next_episode_id = 0;     // persisted cursor, advanced per generation

  // --- Training-only search budget. Independent, NOT a shared session pool. ---
  // PF-C ran a uniform cold 64 sims at every root and built a FRESH session per
  // root, so nothing was ever reused. Here one persistent session spans a
  // placement activation and the two roles have separate, exact budgets: the
  // primary cannot consume the continuation's, because neither is subtracted
  // from a common pool (see DuneSearchBudgetMode::kTrainingPolicyIteration in
  // dune_search_session.cc).
  int primary_simulations = 200;       // NEW simulations at kAgentPrimary
  int continuation_simulations = 64;   // NEW simulations per kAgentContinuation

  // --- Scope: the wide teacher's one axis. ---
  // NEW simulations at kPurchase, kCombatIntrigue and kOtherOptional -- one
  // budget covers all three, because <= 0 makes DuneSearchSession::Search
  // return policy_only_purchase_combat before any budget is resolved
  // (dune_search_session.cc:375, the `policy_only_purchase_combat` return --
  // :354-355 was a misread and is the comment above a DIFFERENT early return).
  // The default 0 is the NARROW teacher and
  // is exactly what this lane has always run, so leaving it alone reproduces
  // rung 1/2/3a behaviour bit-for-bit; the batched lane's wide arm registers 64
  // (SEARCH_PI_BATCHED_TEACHER_REGISTRATION_2026_08_18.md §3.3). It is a
  // parameter of the wide teacher's IDENTITY, not a tuning knob: scope and
  // batching must never vary in the same cell.
  int purchase_combat_budget = 0;

  // --- Runaway watchdog, NOT a budget. ---
  // The lane historically never set DuneSearchConfig::relative_time_budget_ms,
  // so it inherited the 10,000 ms struct default (dune_puct_is_mcts.h:34) -- a
  // default that was never this lane's decision and that bit twice (D4's
  // shortfall, 3a's three clips). The default here is that same 10,000 ms so
  // that not setting it is behaviourally identical to the historical lane and
  // the batch-1 parity references still reproduce; the batched teacher sets
  // 60,000 ms explicitly as a FATAL watchdog with no shortfall tolerance.
  // Whatever the value, the gate that protects the science is the budget
  // invariant (simulations_completed == simulations_requested), not this clock.
  double relative_time_budget_ms = 10000.0;

  // --- Search core ---
  double puct_c = 0.30;
  int max_search_decision_depth = -1;  // uncapped
  bool use_opponent_model = true;      // policy opponent model
  double opponent_model_temperature = 1.0;
  double root_prior_temperature = 1.0;
  double utility_divisor = 4.0;        // value scale; matches the model's

  // --- KataGo root-exploration package. OFF. ---
  // A unit: forced playouts and FPU=0 only mean anything at a noised root. With
  // epsilon 0 there are no forced playouts to prune, so forced_playouts_k is 0
  // rather than the collector's 2.0 -- an armed pruner with no noise to correct
  // would silently subtract organic visits from the target.
  double dirichlet_epsilon = 0.0;
  double dirichlet_alpha_total = 10.83;  // inert while epsilon == 0
  double forced_playouts_k = 0.0;        // inert while epsilon == 0
  bool root_noise_fpu_zero = false;

  // --- Target construction ---
  double target_sharpen_exponent = 1.0;  // inert at 1.0
  SearchPiContinuationTarget continuation_target =
      SearchPiContinuationTarget::kTotalVisits;

  // --- Behaviour ---
  // 0.0 = argmax. The trajectory must reflect the STRONGER controller, which is
  // the whole premise of policy iteration; PF-C sampled its executed action at
  // temperature 1.0 from the visit distribution.
  double behavior_temperature = 0.0;
  // Opponent seats, and the searched seat's own non-searched decisions
  // (leader/purchase/combat/other-optional). Raw policy at T=1.
  double non_search_temperature = 1.0;
  double searched_seat_unsearched_temperature = 1.0;

  // --- Leader teacher: OFF, and recorded as off. ---
  // Commit 8227054 pins the Leader teacher into production runs via
  // LEADER_TEACHER_PINS. This lane constructs its own configuration and never
  // imports production_argv(); whether a later pilot launcher counts as a
  // "production run" that must carry those pins is a registration-time
  // decision, not one this file resolves. The value is manifested so the answer
  // is legible from a run's own artifacts.
  bool search_leader_draft = false;

  // --- Seeds ---
  uint64_t seed_domain = 0;  // REQUIRED nonzero; distinct from PPO's domains

  // --- Coverage heuristic: TELEMETRY ONLY. ---
  // Retained so a row records what the old gate WOULD have decided. Nothing
  // here can suppress a target or change an executed action.
  int telemetry_min_coverage = 3;
  int telemetry_min_visits_per_action = 2;
  double telemetry_min_prior_mass = 0.50;
};

// The dedicated learner's configuration. Independent of every PPO knob.
struct SearchPiLearnerConfig {
  double learning_rate = 1.0e-4;
  int minibatch_size = 256;
  int epochs = 1;
  // Stated explicitly rather than inherited from --value_coef: this lane's
  // objective is CE + value_coef * MSE with no PPO term, so the number means
  // something different here than it does inside a PPO total loss.
  double value_coef = 1.0;
  // The CE's own coefficient, so the ablation matrix can run the value channel
  // ALONE (arm B) the same way value_coef runs the CE channel alone (arms A/D).
  //
  // At exactly 0.0 the policy backward is SKIPPED, not scaled to zero. Scaling
  // would still traverse the graph and write an all-zeros gradient, which is a
  // different thing from "the policy objective contributed nothing": a zero
  // times a non-finite gradient is NaN, and an optimizer that sees a defined
  // zero gradient is not in the same state as one that sees none. The skip is
  // what makes "eliminated" a structural claim rather than a numeric hope.
  // value_coef == 0.0 skips the value backward symmetrically.
  //
  // At exactly 1.0 no multiply is inserted at all, so the anchor arm's numerics
  // are bit-identical to the pilot's, which predates this field.
  double policy_coef = 1.0;
  double grad_clip_norm = 0.5;
  double logit_cap = 10.0;
  // Decoupled AdamW weight decay, carried here so it enters the resume
  // fingerprint. It is applied by the optimizer at every step regardless of
  // gradient, so a resume that silently changed it would be changing the
  // objective; leaving it out of the fingerprint made that change undetectable.
  double weight_decay = 0.0;
  double policy_weight_decay = 0.0;
};

// ---------------------------------------------------------------------------
// One search row
// ---------------------------------------------------------------------------
//
// A NEW structure, not a reinterpretation of SearchTrainingExample. The PF-C
// row is the closed mode's data contract; widening it would make the two lanes
// share a schema they do not share a meaning with.
struct SearchPiRow {
  // --- Model input: the exact tensor the model's forward() consumes. ---
  // information_state (5580) vs observation (5576) is a known seam; the flag
  // records which one this row actually holds.
  std::vector<float> observation;
  bool observation_is_information_state = false;

  Player player = kInvalidPlayer;
  DuneDecisionRole role = DuneDecisionRole::kOtherOptional;
  std::vector<Action> legal_actions;   // root legal actions, target alignment

  // --- Policies, all aligned to legal_actions ---
  std::vector<double> raw_policy;      // untransformed network prior
  std::vector<int> raw_visits;         // UNPRUNED root visits (drive behaviour)
  std::vector<int> target_visits;      // pruned when the package is on
  std::vector<double> target_probs;    // normalized target; sums to 1
  Action chosen_action = kInvalidAction;

  // --- Value target: engine Returns() only. ---
  // Two functions in this repo compute a "final VP" and the reporting helper
  // drops a tech-tile-8 guard; that divergence was a prior gate STOP. This is
  // state->Returns()[player] / utility_divisor and nothing else.
  double value_target = 0.0;
  bool value_target_attached = false;

  // --- Identity ---
  int generation = -1;
  int64_t episode_id = -1;
  int decision_id = -1;

  // --- Search provenance ---
  int simulations_completed = 0;       // NEW simulations this search
  // What the session asked RunSearch for, so a row records its own budget
  // rather than leaving a reader to look it up in the configuration. Neither
  // this nor `early_exit` enters either hash chain: they describe how a row was
  // produced in a way the chains already cover through simulations_completed,
  // and widening a chain would void every recorded one.
  int simulations_requested = 0;
  SearchPiEarlyExit early_exit = SearchPiEarlyExit::kNone;
  int inherited_root_visits = 0;
  std::vector<int> visits_pre_search;  // retention snapshot, when valid
  bool retention_snapshots_valid = false;
  std::string re_root_status = "none"; // "hit" | "miss" | "none"
  SearchPiFallback fallback = SearchPiFallback::kNone;
  std::string search_fallback_reason = "none";  // the session's own string
  double root_value = 0.0;
  std::vector<double> q_values;

  // --- Distribution scalars. ---
  // Computed HERE, at full double precision, before anything is serialized.
  // OpenSpiel's json.cc floors floats under 5e-7 to exactly 0.0, and a floored
  // raw prior makes KL(target || raw) infinite; recomputing these from a
  // serialized row would manufacture that infinity.
  double raw_policy_entropy_norm = 0.0;   // H(raw) / log(n_legal)
  double target_entropy_norm = 0.0;       // H(target) / log(n_legal)
  double raw_policy_max_prob = 0.0;
  double target_max_prob = 0.0;
  double kl_target_given_raw = 0.0;       // KL(target || raw_policy)
  // True when some target mass sat on an action whose raw prior underflowed to
  // <= 0, so the KL above used a floor instead of diverging. Reported rather
  // than hidden: an unflagged finite KL there would be a fabricated number.
  bool kl_raw_prior_floored = false;
  bool target_argmax_differs_from_raw = false;

  // --- Coverage heuristic, telemetry only (never a behaviour switch) ---
  int num_covered_actions = 0;
  double covered_prior_mass = 0.0;
  bool would_pass_legacy_coverage_gate = false;
};

// ---------------------------------------------------------------------------
// Per-generation collection telemetry, split primary vs continuation
// ---------------------------------------------------------------------------
struct SearchPiRoleStats {
  int64_t roots_seen = 0;
  int64_t searches_run = 0;
  int64_t rows_emitted = 0;
  int64_t fallbacks = 0;
  int64_t simulations_completed = 0;

  // --- Early-exit accounting (closes Amendment 1's instrument gap) ---
  //
  // `simulations_requested` is the sum of what the session asked for, so the
  // shortfall is a subtraction rather than a multiplication against an assumed
  // per-root budget -- the arithmetic that was the ONLY detector of D4's 29
  // missing simulations. The counters then say which exit produced it, and the
  // `first_short_*` fields name the root, which no artifact could do before.
  int64_t simulations_requested = 0;
  int64_t searches_short_of_budget = 0;
  int64_t simulation_shortfall = 0;
  int64_t early_exit_counts[kSearchPiEarlyExitCount] = {0, 0, 0, 0, 0, 0, 0, 0};
  int64_t first_short_episode_id = -1;
  int64_t first_short_decision_id = -1;
  int first_short_simulations_completed = -1;
  int first_short_simulations_requested = -1;
  // BOTH reason strings at that root, carried verbatim. They come from
  // different places and only one of them can name a kSessionBudget exit:
  // `first_short_bot_reason` is RunSearch's own ("timeout", "max_nodes",
  // "low_coverage", "zero_visits"), while `first_short_session_reason` is
  // DuneSearchSession's budget_limit_reason, the SOLE discriminator for
  // kSessionBudget. Carrying only one would reintroduce Amendment 1's gap one
  // level down: the class named, the cause dropped.
  std::string first_short_bot_reason;
  std::string first_short_session_reason;

  // --- Deadline headroom ---
  //
  // Every searched root runs under a live wall-clock deadline: the lane never
  // sets DuneSearchConfig::relative_time_budget_ms, so it stays at its 10,000 ms
  // default and dune_puct_is_mcts.cc:982 breaks the simulation loop on it. That
  // is a runaway guard at one game per process, but it is NOT inert under
  // concurrency -- parallel games share one CUDA stream, per-simulation latency
  // rises, and a truncated search weakens ONLY the arm that searches.
  //
  // A zero shortfall says the deadline did not fire. These say how close it
  // came, which is the difference between "it held this time" and "it holds".
  double max_search_elapsed_ms = 0.0;
  double sum_search_elapsed_ms = 0.0;

  // PER-ROOT LATENCY DISTRIBUTION, as a fixed-bin histogram.
  //
  // The registration makes the latency TAIL an acceptance criterion per
  // searched arm -- P50, P95 and max against the 60,000 ms watchdog -- and the
  // instrument reported only the max and the mean. A mean cannot see a tail,
  // and a max is one root. The criterion existed with nothing able to report
  // it, which is the same defect class as a STOP the instrument cannot raise.
  //
  // A histogram rather than a reservoir or a sorted vector: exact quantiles
  // would mean retaining ~200k doubles per role per arm, and these are read
  // once at the end. 10 ms bins to 60 s, plus an overflow bin, gives P95 to
  // within 10 ms against a 60,000 ms limit -- three orders of magnitude finer
  // than any decision it feeds.
  static constexpr int kLatencyBins = 6001;   // [0,10) .. [59990,60000), +over
  static constexpr double kLatencyBinMs = 10.0;
  int64_t search_elapsed_hist[kLatencyBins + 1] = {0};
  double configured_time_limit_ms = 0.0;  // what the deadline actually was

  int64_t inherited_visits = 0;
  int64_t re_root_hits = 0;
  int64_t re_root_misses = 0;
  // Rows a concentrated-search gate would have discarded but this lane kept.
  int64_t kept_despite_legacy_gate = 0;
  double sum_target_entropy_norm = 0.0;
  double sum_raw_entropy_norm = 0.0;
  double sum_kl_target_given_raw = 0.0;
  int64_t target_argmax_overrides = 0;
};

// ---------------------------------------------------------------------------
// One game's outcome
// ---------------------------------------------------------------------------
//
// The generator always played complete games and always read Returns() at the
// end -- to attach value targets -- but it kept none of it, because a training
// generation only ever needed the aggregate. A strength measurement needs the
// per-game record: the audit's contrast is a PAIRED test on shared episode ids,
// and pairing is a per-episode join.
struct SearchPiGameOutcome {
  int64_t episode_id = -1;
  Player searched_seat = kInvalidPlayer;
  std::vector<double> returns;       // engine Returns(), every seat
  double searched_seat_return = 0.0;
  // 1..4 off the terminal ladder {+2.25, +0.25, -0.75, -1.75}; 0 means the
  // return matched no rung, which is reported rather than snapped to the
  // nearest one -- an off-ladder return means the value pipeline changed.
  int searched_seat_placement = 0;
  int64_t searched_seat_decisions = 0;
  int64_t rows_emitted = 0;
  int64_t primary_roots = 0;
  int64_t continuation_roots = 0;
  int64_t primary_simulations = 0;
  int64_t continuation_simulations = 0;
  int64_t fallbacks = 0;
  int64_t searches_short_of_budget = 0;
  int64_t off_scope_simulations = 0;  // structurally 0; carried so it is checked

  // --- Additive game-shape telemetry (Fable, 2026-08-18) -------------------
  // Requested pre-registration so a game-length criterion is a STANDING column
  // in every audit and readout rather than a bespoke re-analysis. Costs one
  // integer and four VPs per game and changes no contrast.
  //
  // BOTH of these have a wrong derivation that looks right, so both are taken
  // from game truth:
  //
  // `final_round` is GetCurrentRound() - 1. The engine increments round_
  // (dune_imperium.cc:8960) BEFORE the end-game test immediately below it, so
  // at kTerminal the counter has already moved past the round actually played.
  // A game decided on round 7 reports GetCurrentRound() == 8. Emitting the raw
  // counter would put every game-length criterion off by exactly one, silently
  // and in the same direction, which is indistinguishable from a real effect.
  //
  // `final_vp` is DuneImperiumState::FinalScoredVp(p) -- vp_[p] plus
  // ComputeEndgameVp(p) -- which is the expression Returns() itself ranks
  // placements on. It is NOT the GetTrueFinalVp reporting helper copied around
  // several example binaries: that helper awards the "all four factions >= 3
  // influence" bonus unconditionally while the engine gates it behind owning
  // tech tile 8, so the helper overstates by exactly 1 for any seat with all
  // factions >= 3 and no tile 8. They disagree on 212 of 1,600 (game, seat)
  // pairs, and that divergence is a registered STOP elsewhere in this program
  // (docs/PWO5_GATE3_STOP_FINAL_VP_TARGET.md). Nothing scores a game with the
  // helper.
  int final_round = 0;
  std::vector<int> final_vp;  // FinalScoredVp per seat, engine order

  // --- Per-episode budget accounting (2026-08-18) ---------------------------
  //
  // The registration's exact invariant is `completed == requested` at every
  // searched root. It was assertable only over SUMMARY TOTALS, which include
  // the very episodes the watchdog discard removes -- so a single tolerated
  // fire made the totals unequal and the cell INVALID, and the whole discard
  // rule was unreachable. Requested is carried per episode so the invariant can
  // be asserted over the RETAINED set, which is what §9 actually registers.
  //
  // `*_requested` comes from the session's own soft_sim_limit, never from
  // roots x budget, so `requested == roots x budget` stays a real check rather
  // than an identity. Structurally zero in the raw arms: they take the branch
  // that never resolves a budget, exactly as their role stats do.
  int64_t primary_simulations_requested = 0;
  int64_t continuation_simulations_requested = 0;

  // The WIDE teacher's three roles, kept SEPARATE from continuation. They were
  // previously absorbed by the `else` branch that splits primary from
  // continuation, so in a wide arm `continuation_roots` silently counted
  // ~55 extra roots per game and the per-episode split disagreed with the
  // summary role buckets it is supposed to reconcile against.
  int64_t wide_roots = 0;
  int64_t wide_simulations = 0;
  int64_t wide_simulations_requested = 0;

  // --- The played-action manipulation check (Fable, 2026-08-18) -------------
  //
  // The 2026-08-18 review found a wide arm that ran its simulations and
  // DISCARDED them: it played the narrow arm's moves at a 1.392x compute bill
  // and passed every gate that existed. What makes that failure invisible is
  // that the searched arm's telemetry is all about the search, not about what
  // was PLAYED -- mcts_proposed_action is precisely the field that can diverge
  // from the action applied to the state.
  //
  // So this counts, per searched role class, the roots where the action
  // ACTUALLY PLAYED differs from the raw-prior argmax that the matched control
  // would have played at the same root. It is a floor, not a target: a wide arm
  // whose count is zero played the control's moves. The raw arms are
  // structurally zero here by construction, and that is asserted rather than
  // assumed.
  //
  // The argmax is recomputed from the search's own diagnostics -- no extra
  // Prior() call, so no extra inference and no perturbation of throughput --
  // with the same first-wins tie-break as RawPriorArgmaxAction.
  // PER ROLE, indexed by DuneDecisionRole. Deliberately not the two lumped
  // counters this started as. Those claimed to assert "per class" while
  // asserting over a union, and the union hides the exact failure they exist
  // to catch: with the observed role frequencies a wide arm whose kPurchase
  // and kCombatIntrigue searches are entirely inert still shows a healthy
  // count from kOtherOptional alone, passes, and bills for three roles while
  // testing one. The same hole applied across kAgentPrimary and
  // kAgentContinuation.
  //
  // A per-role floor must be conditioned on the role having been REACHED --
  // kCombatIntrigue is observed at zero roots in whole generations, which is a
  // property of the game and not of the measurement.
  int64_t roots_played_differs_from_raw[7] = {0, 0, 0, 0, 0, 0, 0};

  // FNV-1a 64 over the searched seat's (decision_id, played action) sequence.
  //
  // A DIVERGENCE DETECTOR, not a parity reference: the lane's bitwise claims
  // are the sha256 target/extended row chains and nothing else. Its one job is
  // to make "wide played the same trajectory as narrow on episode X" a fact
  // that can be read off two artifacts instead of inferred from coinciding
  // returns. A silently null wide arm reproduces narrow's digest on EVERY
  // episode; a partially null one reproduces it on some.
  uint64_t played_action_digest = 0xcbf29ce484222325ull;  // FNV offset basis
};

// Placement 1..4 from an engine return, or 0 when it sits on no rung.
// `utility_divisor` scales the ladder the same way SearchPiRow::value_target
// does, so one helper serves a raw Returns() value (divisor 1.0) and a value
// target (divisor as configured).
int SearchPiPlacementFromReturn(double engine_return, double utility_divisor);

struct SearchPiGenerationStats {
  int generation = -1;
  int games = 0;
  int64_t first_episode_id = -1;
  int64_t next_episode_id = -1;
  double collection_wall_time_s = 0.0;
  int64_t inference_calls = 0;
  int64_t rows_total = 0;

  SearchPiRoleStats primary;
  SearchPiRoleStats continuation;

  // The WIDE teacher's three additional roles, each with its own full role
  // stats rather than a null bucket.
  //
  // These exist because the alternative was silence. Before them, a search at
  // kPurchase / kCombatIntrigue / kOtherOptional had `rs == nullptr`, so its
  // simulations_requested, searches_short_of_budget, early-exit classes and
  // fallbacks were all skipped -- meaning a wide-role search that completed 1
  // of 64 simulations incremented nothing, failed no invariant, and could not
  // fire --fail_on_short_search. The exact-budget invariant is the gate that
  // protects this audit, and an invariant that cannot see two thirds of the
  // wide teacher's roots is not a gate. Structurally zero for the narrow
  // teacher, which is asserted.
  SearchPiRoleStats purchase;
  SearchPiRoleStats combat_intrigue;
  SearchPiRoleStats other_optional;

  // Zero-simulation proof for every role this lane must not search. Indexed by
  // DuneDecisionRole. Which entries are legitimately nonzero is now
  // SCOPE-DEPENDENT: narrow searches {2,3} only, wide searches {2,3,4,5,6},
  // and {0,1} are structurally zero in both.
  int64_t simulations_by_role[7] = {0, 0, 0, 0, 0, 0, 0};
  int64_t decisions_by_role[7] = {0, 0, 0, 0, 0, 0, 0};

  // Leader rows emitted. Structurally always 0 in this lane; asserted by tests.
  int64_t leader_rows_emitted = 0;

  // Which controller drove the searched seat, and one entry per game in
  // episode order. Empty for no generation the generator ran.
  SearchPiArm arm = SearchPiArm::kSearched;
  std::vector<SearchPiGameOutcome> games_played;

  // Rolling SHA-256 over the emitted targets, for resume verification.
  // LEGACY FORMAT, retained byte-for-byte: it is the only chain any prior run
  // recorded, so it is the sole provenance tie back to the pilot's generations.
  // Do not widen it -- widen the one below.
  std::string target_hash_chain;

  // Rolling SHA-256 over EVERY learner and analysis input on the row: the
  // observation tensor, the player/role/provenance, the legal actions, the raw
  // snapshot prior, the target, the chosen action and the value target.
  //
  // The legacy chain covers targets only. It therefore proves target-stream
  // identity, NOT row identity -- two collections could agree on every target
  // while disagreeing on the observations those targets were attached to, and
  // the legacy chain would call them equal. An ablation matrix whose whole
  // design is "only the learner differs" needs the stronger statement, so this
  // is the chain the cross-arm equality assertion tests.
  std::string extended_hash_chain;
};

// ---------------------------------------------------------------------------
// Learner telemetry
// ---------------------------------------------------------------------------
struct SearchPiLearnerStats {
  double policy_ce = 0.0;    // mean over executed minibatches
  double value_mse = 0.0;    // mean over executed minibatches

  // The two objectives' gradients, measured separately by backpropagating each
  // alone. PF-C could not do this: its aux CE and value MSE went through ONE
  // combined backward, so no per-component norm existed and "the CE term did
  // it" was inference, not measurement (diagnosis section 2).
  double policy_grad_norm = 0.0;
  double value_grad_norm = 0.0;

  // The same two norms, split across the three parameter groups. The totals
  // above establish which objective carries more MASS; they cannot say where it
  // lands, and "the value objective dominated the shared trunk" was a claim the
  // lane had no instrument for -- it was read off total norms and had to be
  // withdrawn (SEARCH_PI_LANE doc section 6). These six numbers are that
  // instrument.
  //
  // Two of the six are STRUCTURAL ZEROS, not measurements: the CE never reaches
  // the value head and the value MSE never reaches the policy head. They are
  // emitted anyway, because an observed 0.0 there is the decomposition proving
  // itself -- a nonzero would mean ClassifyParam had mis-grouped a parameter.
  double policy_grad_norm_trunk = 0.0;
  double policy_grad_norm_policy_head = 0.0;
  double policy_grad_norm_value_head = 0.0;  // structural 0
  double value_grad_norm_trunk = 0.0;
  double value_grad_norm_policy_head = 0.0;  // structural 0
  double value_grad_norm_value_head = 0.0;

  // Whether each objective's backward actually ran. At coefficient 0 the lane
  // SKIPS the backward rather than scaling it, so these record the structural
  // fact directly instead of leaving a reader to infer it from a zero norm --
  // which is also what a genuinely zero gradient would look like.
  bool policy_backward_executed = false;
  bool value_backward_executed = false;

  double grad_cosine_overall = 0.0;
  double grad_cosine_policy_head = 0.0;
  double grad_cosine_trunk = 0.0;
  double grad_cosine_value_head = 0.0;

  // Whether each cosine above is a MEASUREMENT or a structural non-answer.
  //
  // The CE gradient cannot reach the value head and the value gradient cannot
  // reach the policy head, so those two groups always have a zero denominator
  // and their "cosine" is a 0.0 sentinel, not an observed orthogonality. Without
  // these flags a reader sees `grad_cosine_policy_head: 0` and concludes the two
  // objectives were measured orthogonal there, which is a stronger claim than
  // anything that was computed. Only the trunk cosine is informative;
  // grad_cosine_overall is the trunk dot product over whole-model norms, i.e. a
  // diluted trunk figure rather than an independent quantity.
  bool grad_cosine_overall_defined = false;
  bool grad_cosine_policy_head_defined = false;
  bool grad_cosine_trunk_defined = false;
  bool grad_cosine_value_head_defined = false;

  // --- What the critic was asked to fit, and what it already predicted. ---
  //
  // The lane logged neither, so "value MSE 0.33-0.54" could only be read against
  // distribution-free bounds on the terminal ladder. Those bounds exclude a
  // constant target but cannot separate a critic making confident wrong
  // state-dependent reads from one whose outputs left the target range: both
  // clear the same floor. Prediction mean/sd beside target mean/sd separates
  // them in one line.
  //
  // Measured over ALL rows under no_grad, immediately before the first optimizer
  // step and immediately after the last. Pre and post are what make the churn
  // visible as movement rather than as a level.
  double value_target_mean = 0.0;
  double value_target_sd = 0.0;
  double critic_pred_mean_pre = 0.0;
  double critic_pred_sd_pre = 0.0;
  double critic_pred_mean_post = 0.0;
  double critic_pred_sd_post = 0.0;
  bool critic_pred_measured = false;

  // The searched seat's realized outcome mix, deduplicated to one entry per
  // EPISODE -- the ~16 outcome samples a generation actually carries, not the
  // ~1,450 rows that repeat them. Index 0..3 is placement 1st..4th, recovered
  // from the value target against the terminal ladder {+2.25, +0.25, -0.75,
  // -1.75}/4. A target that matches no rung increments `outcome_unmapped`
  // rather than being forced onto the nearest one.
  int64_t outcome_placements[4] = {0, 0, 0, 0};
  int64_t outcome_episodes = 0;
  int64_t outcome_unmapped = 0;

  int64_t distinct_rows = 0;
  int64_t presentations = 0;
  int64_t minibatches = 0;

  // Structural, always false. The dedicated learner constructs no PPO surrogate,
  // no ratio, no clip, no entropy bonus, no advantage term and no target-KL
  // early stop. Exposed so a test can assert the negative rather than infer it.
  bool ppo_policy_loss_constructed = false;
};

// ---------------------------------------------------------------------------
// Pure helpers (exposed for unit testing)
// ---------------------------------------------------------------------------

// The DuneSearchConfig this lane runs a session with, for a given SearchPiConfig.
// Sets purchase_combat_budget = 0 and search_leader_draft = false, which is what
// makes every non-agent-turn role a guaranteed zero-simulation policy-only
// return inside DuneSearchSession::Search.
DuneSearchConfig SearchPiSearchConfigFor(const SearchPiConfig& config,
                                         uint64_t search_seed);

// Is this a role the lane searches?
// SCOPE-CONDITIONAL. Agent turns are always searched; the three short-window
// roles are searched exactly when the wide teacher's budget is armed.
//
// This predicate does far more than name a role, which is why it is the one
// place the scope axis is decided. It gates, in SearchPiGenerator: whether the
// searched action is PLAYED at all, whether the raw control plays argmax or a
// T=1 sample, whether a row is emitted, and whether role stats are recorded.
//
// It was previously hard-coded to agent turns, and `purchase_combat_budget`
// only reached DuneSearchSession. The result was a wide teacher that ran its
// simulations and then DISCARDED them: `chosen` stayed invalid, so the generic
// tail played SampleRawPolicyAction and the search survived only as
// `mcts_proposed_action`, recorded and never played. A wide arm would have
// played the narrow arm's actions at a 1.392x compute bill, and no test,
// invariant or analyzer check could have detected it.
//
// The default of 0 is the narrow teacher, so every existing caller keeps
// agent-turn-only semantics bit-for-bit and the recorded parity chains stay
// valid references.
inline bool IsSearchPiSearchRole(DuneDecisionRole role,
                                 int purchase_combat_budget = 0) {
  if (role == DuneDecisionRole::kAgentPrimary ||
      role == DuneDecisionRole::kAgentContinuation) {
    return true;
  }
  if (purchase_combat_budget <= 0) return false;
  return role == DuneDecisionRole::kPurchase ||
         role == DuneDecisionRole::kCombatIntrigue ||
         role == DuneDecisionRole::kOtherOptional;
}

// Classify a completed search as usable or a specific technical failure.
// `legal_actions` is the truth from the state, checked against the diagnostics.
SearchPiFallback ClassifySearchPiResult(const DuneSearchResult& result,
                                        const std::vector<Action>& legal_actions);

// Normalized Shannon entropy over `probs`, divided by log(n). Returns 0 for
// n <= 1 (a forced choice carries no entropy, and log(1) == 0 would divide by
// zero).
double NormalizedEntropy(const std::vector<double>& probs);

// KL(target || raw). `floored` is set when a positive target mass met a
// non-positive raw prior and the tiny floor was used instead of diverging.
double KlTargetGivenRaw(const std::vector<double>& target,
                        const std::vector<double>& raw, bool* floored);

// Build the policy target from root visits: select the visit vector per the
// continuation convention, prune when the exploration package is armed, sharpen,
// then normalize. Returns false and leaves `out_*` untouched when the inputs
// cannot produce a valid target.
bool BuildSearchPiTarget(const SearchPiConfig& config, DuneDecisionRole role,
                         const SearchDiagnostics& diag,
                         std::vector<int>* out_target_visits,
                         std::vector<double>* out_target_probs);

// Pick the executed action from the UNPRUNED visit counts at
// `behavior_temperature`. Temperature 0 is argmax with a deterministic
// lowest-index tie-break.
int SelectSearchPiActionIndex(const std::vector<int>& raw_visits,
                              double behavior_temperature, double r_val);

// The raw-prior argmax. Every technical-failure fallback plays this, never a
// uniform draw -- the standing search/eval parity invariant is that starved
// paths degrade to the raw network prior.
Action RawPriorArgmaxAction(const ActionsAndProbs& prior,
                            const std::vector<Action>& legal_actions);

// Fill the distribution scalars on a row from its own vectors. Call once, at
// construction, before serialization.
void FillSearchPiRowScalars(SearchPiRow* row);

// ---------------------------------------------------------------------------
// Generator
// ---------------------------------------------------------------------------
class SearchPiGenerator {
 public:
  explicit SearchPiGenerator(const SearchPiConfig& config);

  // Plays `config.games_per_generation` complete games from the FROZEN
  // pre-generation snapshot `evaluator`, appends rows to `out`, fills `stats`,
  // and advances the episode cursor.
  //
  // Seat rotation is a pure function of episode_id, as the existing collector
  // does. The designated seat drives ONE persistent DuneSearchSession across
  // each placement activation; all other seats play the frozen raw policy at
  // `non_search_temperature`.
  //
  // `arm` selects the searched seat's controller and nothing else. It defaults
  // to the collection controller, so the trainer's call site says what it has
  // always said and the training lane cannot acquire a new behaviour by
  // omission.
  void GenerateGeneration(int generation,
                          const std::shared_ptr<const Game>& game,
                          const std::shared_ptr<algorithms::Evaluator>& evaluator,
                          std::vector<SearchPiRow>* out,
                          SearchPiGenerationStats* stats,
                          SearchPiArm arm = SearchPiArm::kSearched);

  static Player SearchedSeatForEpisode(int64_t episode_id, int num_players);

  const SearchPiConfig& config() const { return config_; }

 private:
  SearchPiConfig config_;
};

// ---------------------------------------------------------------------------
// Dedicated learner
// ---------------------------------------------------------------------------
//
// mean legal-action CE(search_target, policy) + value_coef * mean MSE(return, value)
//
// and nothing else. No PPO surrogate, no importance ratio, no clipping, no
// entropy bonus, no GAE/advantage policy term, no target-KL early stop, no
// 0.10 search_loss_coef, no simultaneous PPO backward. `optimizer` must be this
// lane's OWN optimizer instance, not the PPO one.
SearchPiLearnerStats RunSearchPiLearner(
    std::shared_ptr<SharedDunePolicyValueNetImpl> model,
    torch::optim::Optimizer& optimizer, const std::vector<SearchPiRow>& rows,
    int64_t obs_size, int64_t action_dim, torch::Device device,
    uint64_t master_seed, int generation, const SearchPiLearnerConfig& cfg);

// Appends one generation's learner + collection telemetry to a JSONL sidecar
// beside `diagnostics_path`. A SIDECAR, deliberately: diagnostics.csv compares
// its header byte-for-byte on resume, so widening it is fatal against every
// existing run's file.
std::string SearchPiTelemetryPath(const std::string& diagnostics_path);
void WriteSearchPiTelemetry(const std::string& diagnostics_path,
                            const SearchPiConfig& config,
                            const SearchPiLearnerConfig& learner_cfg,
                            const SearchPiGenerationStats& gen_stats,
                            const SearchPiLearnerStats& learner_stats);

// ---------------------------------------------------------------------------
// Checkpoint-manifest state
// ---------------------------------------------------------------------------
//
// The resume cursor and the configuration fingerprint live in the MANIFEST, not
// in diagnostics.csv.
struct SearchPiState {
  int generation = 0;
  int64_t next_episode_id = 0;
  int64_t cum_rows = 0;
  int64_t cum_primary_rows = 0;
  int64_t cum_continuation_rows = 0;
  int64_t cum_primary_simulations = 0;
  int64_t cum_continuation_simulations = 0;
  std::string target_hash_chain;
  std::string extended_hash_chain;
  SearchPiConfig config;
  SearchPiLearnerConfig learner;
};

// Every search-configuration field, including the budgets, puct_c,
// forced_playouts_k, epsilon and the target exponent, plus the learner's own
// hyperparameters and the leader flag.
json::Object WriteSearchPiState(const SearchPiState& state);
bool ReadSearchPiState(const json::Object& obj, SearchPiState* out,
                       std::string* error);

// Stable fingerprint over the fields a resume must not silently change.
std::string SearchPiConfigFingerprint(const SearchPiConfig& config,
                                      const SearchPiLearnerConfig& learner);

// Rolling target hash: chain(prev, row) -- covers episode/decision ids, the
// player, the legal actions, the target probabilities and the value target.
// LEGACY. Frozen: prior runs' recorded chains are only comparable against this
// exact byte layout, and reproducing the pilot's generation-1 chain is the
// provenance tie for the matrix's anchor arm.
std::string ChainSearchPiTargetHash(const std::string& prev,
                                    const SearchPiRow& row);

// Rolling EXTENDED row hash: everything the legacy chain covers, plus the
// observation tensor and its information-state flag, the decision role, the raw
// snapshot prior, and the search provenance (new simulations, inherited visits,
// re-root status, fallback class). This is the chain that can assert two
// collections produced the same ROWS rather than merely the same targets.
std::string ChainSearchPiExtendedRowHash(const std::string& prev,
                                         const SearchPiRow& row);

}  // namespace open_spiel

#endif  // OPEN_SPIEL_EXAMPLES_DUNE_SEARCH_PI_H_
