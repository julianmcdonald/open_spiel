// PWO-4 trajectory-distillation generator.
//
// Registration of record: docs/PWO4_TRAJECTORY_REGISTRATION.md revision 3
// (commit 1c60e36, sha256 57ec6310...0733), frozen by docs/PWO4_FREEZE_ADDENDUM.md.
// Execution plan: docs/PWO4_IMPLEMENTATION_PROMPT_2026_07_28.md revision 2.
//
// ---------------------------------------------------------------------------
// WHY THIS IS A NEW BINARY AND NOT A MODE ON dune_search_benchmark
// ---------------------------------------------------------------------------
// dune_search_benchmark is the REFERENCE INSTRUMENT of the section 7.1
// cross-build fidelity gate (sha256 60ea430a...). Adding a mode would relink it
// and destroy the only fixed point the gate has. Precedent: dune_pwo3_prior_recovery
// is a separate target from dune_pwo3_teacher_audit for exactly this reason
// (examples/CMakeLists.txt:88-91). The shared behaviour is reached through the
// SAME code -- DunePUCTISMCTSBot::Step() on the dune_search library -- never
// through a reimplementation.
//
// ---------------------------------------------------------------------------
// WHAT THIS TOOL REPRODUCES, AND HOW THE REPRODUCTION IS KEPT EXACT
// ---------------------------------------------------------------------------
// The Path-B (--use_session=false) branch of dune_search_benchmark's WorkerThread:
// a fresh full search of exactly 200 simulations at every decision the reference
// controller searches, one searched seat per game, rotated as g % 4.
//
// The RNG consumption order is the whole ballgame (registration section 4.1a,
// review surface item 1). Reproduced verbatim from dune_search_benchmark.cc:
//
//   g            = start_episode_id + offset                            [:480]
//   game_seed    = DeriveSeed(seed, kStreamBlueprint, g)                 [:481]
//   game_rng     = mt19937(game_seed)                                    [:482]
//   force_rng    = mt19937(DeriveSeed(seed, kStreamSearchGate, g))       [:486]
//                  -- constructed, never consumed (no swordmaster forcing)
//   search_seat  = g % 4                                                 [:492]
//   for p in 0..3 IN SEAT ORDER:  seat_seed = game_rng()                 [:519-526]
//       config.seed = seat_seed  (searched seat)                         [:555]
//       DuneGreedyBot(seat_seed) (every other seat)                      [:613-615]
//   chance nodes: SampleAction(outcomes, game_rng)                       [:747]
//
// The seat draw fires for ALL FOUR seats, searched or not: :526 sits OUTSIDE the
// seat_is_searched[p] branch at :527, and the source comment at :520-525 warns
// that moving or conditionalising it "silently reshuffles every downstream chance
// realization". SelectRawPriorAction (:810), the third game_rng consumer, never
// executes here because --fresh_search_roles is empty.
//
// ---------------------------------------------------------------------------
// TWO RECORD CLASSES (work order section 1) -- NOT the same population
// ---------------------------------------------------------------------------
//   audit.jsonl   every searched-seat decision, including the zero-simulation
//                 single-action and non-strategic ones. ~197.8 rows/game.
//   labels.jsonl  ONLY search_expected && simulations_completed == 200 &&
//                 inherited_root_visits == 0. ~121.4 rows/game.
//
// A non-eligible decision is NOT a weight-zero label: it has no target, no
// coverage reading and no stability reading, and never enters labels.jsonl.
// Weight-zero labels live INSIDE the label stream -- a row that completed 200
// simulations but failed coverage or stability (registration section 6.1).
//
// ---------------------------------------------------------------------------
// SERIALIZATION -- never json.cc's default double formatting
// ---------------------------------------------------------------------------
// open_spiel/utils/json.cc:302 returns std::to_string(v): a six-decimal %f that
// floors anything below 5e-7 to exactly 0.0. That defect cost PWO-3 an entire
// amendment. Every registered double here is written as STRINGS:
//   <name>_exact    %.17g  (max_digits10 -- parsing reproduces the bit pattern)
//   <name>_legacy6  json::ToString(json::Value(v)) -- the ACTUAL legacy formatter,
//                   called rather than reimplemented, so the section 7.1 six-decimal
//                   projection compares strings produced by one formatter. Python's
//                   f"{x:.6f}" is a SECOND instrument for the quantity under audit
//                   and is never used for this.
//   raw_prior_vector_hex   %a, as JSON strings (0x1.91eb851eb851fp-4 is not a
//                   JSON number and must never be written bare).
//
// ---------------------------------------------------------------------------
// DEAD REFERENCE FIELDS -- measured, not assumed (registration section 3.3)
// ---------------------------------------------------------------------------
// Registration section 3.3 names three fields dead on the fresh path. Inspection of
// the committed reference run (calibration_results_v2/step2_pathB_freshsearch_seed13/
// search.jsonl) shows the dead set is BROADER: every field the DuneSearchSession
// populates is a struct default, because bot->Step() never touches that struct.
// Measured dead on a 200-simulation reference row:
//
//   decision_role ""      is_strategic false   root_coverage 0.0   (section 3.3's three)
//   round -1              phase ""             searched_seat -1
//   session_id ""         budget_mode ""       tree_node_count 0
//   hard_sim_limit 0      soft_sim_limit 0     hard_time_limit_ms 0.0
//   soft_time_limit_ms 0.0                     newly_completed_simulations 0
//   session_cumulative_simulations 0           short_window_cumulative_simulations 0
//   session_cumulative_search_time_ms 0.0      long_agent_session_cumulative_time_ms 0.0
//
// This generator therefore emits BOTH, under distinct names:
//   <field>                     the LIVE value, recomputed from engine state
//   <field>_diag_reference      the reference's own dead value, verbatim
//
// so section 7.1 equality can be checked on EVERY reference field rather than
// excluding a class of them. Recording the dead value is what makes "this field is
// dead" evidence instead of an assertion. root_coverage is emitted ONLY as
// _diag_reference: registration section 3.3.3 forbids consuming it.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "open_spiel/abseil-cpp/absl/flags/flag.h"
#include "open_spiel/abseil-cpp/absl/flags/parse.h"
#include "open_spiel/abseil-cpp/absl/random/distributions.h"
#include "open_spiel/abseil-cpp/absl/strings/str_format.h"
#include "open_spiel/games/dune_imperium/dune_imperium.h"
#include "open_spiel/games/dune_imperium/dune_imperium_cards.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"
#include "open_spiel/utils/json.h"
#include <torch/torch.h>
#include <ATen/detail/CUDAHooksInterface.h>
#include <ATen/cuda/CUDAContext.h>  // getDeviceProperties, as dune_pwo3_prior_recovery.cc:82

#include "dune_evaluator.h"
#include "dune_network.h"
#include "dune_puct_is_mcts.h"
#include "dune_pwo2_common.h"
#include "dune_pwo3_common.h"
#include "dune_search_routing.h"
#include "dune_seed_utils.h"
#include "dune_sha256.h"

// --- Output paths ----------------------------------------------------------
ABSL_FLAG(std::string, audit_jsonl_path, "", "Per-decision audit telemetry (every searched-seat decision).");
ABSL_FLAG(std::string, labels_jsonl_path, "", "The PWO-5 label stream (successful searches only).");
ABSL_FLAG(std::string, games_jsonl_path, "", "Per-game summary + the full action history used by the role verifier.");
ABSL_FLAG(std::string, manifest_json_path, "", "Run manifest: every pin, every digest, the provenance contract.");
ABSL_FLAG(std::string, engine_tree_hash, "", "REQUIRED. git rev-parse HEAD:games/dune_imperium.");
ABSL_FLAG(std::string, registration_sha256, "", "REQUIRED. sha256 of docs/PWO4_TRAJECTORY_REGISTRATION.md.");
ABSL_FLAG(std::string, amendment_sha256, "", "REQUIRED. sha256 of docs/PWO4_AMENDMENT_1_GATE_SEMANTICS.md. The amendment governs the gate semantics this stream is judged by, so a stream that cannot name it is unjudgeable.");

// --- Frozen controller MODES. These are not tunables. They exist as flags so a
// --- launcher must state them and the binary can FATALLY reject any other value:
// --- a mode silently defaulted is a mode nobody declared. Amendment 1 section 5.
ABSL_FLAG(bool, use_session, false, "FROZEN false. The July-14 reference protocol: a fresh full search per decision, no session.");
ABSL_FLAG(std::string, controller_mode, "single", "FROZEN 'single'. Homogeneous + use_session=false is a fatal error in the reference itself (dune_search_benchmark.cc:1516-1524).");
ABSL_FLAG(std::string, fresh_search_roles, "", "FROZEN empty. A non-empty value is the arm-4 decomposition filter and would change the eligible set.");
ABSL_FLAG(bool, policy_only, false, "FROZEN false.");
ABSL_FLAG(bool, live_deadline, false, "FROZEN false. With policy_only=false this yields budget_mode = kFixedSessionSimulations.");
ABSL_FLAG(int, force_swordmaster_rounds, 0, "FROZEN 0. No curriculum, no forcing (registration section 2.6).");
ABSL_FLAG(int, grant_swordmaster_round, 0, "FROZEN 0. No endowment.");

// --- Run range (registration section 4.2) ----------------------------------
ABSL_FLAG(int, games, 8, "How many games to play.");
ABSL_FLAG(int, start_episode_id, 0, "First game index g.");

// --- The section 2.6 controller pin. EVERY flag is passed explicitly by the
// --- launcher, including those equal to the current default: defaults drift,
// --- registrations do not. Defaults here mirror dune_search_benchmark.cc.
ABSL_FLAG(std::string, model_checkpoint, "", "Teacher checkpoint (Branch-A u2450).");
ABSL_FLAG(std::string, opponent_checkpoint, "", "Empty => the candidate model driven as DuneGreedyBot.");
ABSL_FLAG(int, seed, 20260729, "Base seed.");
ABSL_FLAG(int, threads, 1, "Pinned to 1 (registration section 2.4).");
ABSL_FLAG(int, hidden_dim, 2048, "Candidate model hidden dimension.");
ABSL_FLAG(int, num_blocks, 8, "Candidate model residual blocks.");
ABSL_FLAG(int, opp_hidden_dim, 2048, "Opponent model hidden dimension (pinned, not inherited).");
ABSL_FLAG(int, opp_num_blocks, 8, "Opponent model residual blocks (pinned, not inherited).");
ABSL_FLAG(int, max_simulations, 200, "Exactly 200 (registration section 2.2).");
ABSL_FLAG(double, puct_c, 0.3, "PUCT exploration constant.");
ABSL_FLAG(int, max_world_samples, -1, "Determinization world-sample cache.");
ABSL_FLAG(double, utility_divisor, 4.0, "Terminal utility divisor.");
ABSL_FLAG(double, temperature, 0.0, "Final move choice temperature.");
ABSL_FLAG(double, simulated_opponent_temperature, 1.0, "In-search opponent temperature.");
ABSL_FLAG(double, external_opponent_temperature, 0.0, "External raw-policy opponent temperature.");
ABSL_FLAG(double, dirichlet_epsilon, 0.0, "Root noise OFF.");
ABSL_FLAG(double, dirichlet_alpha, 0.3, "Inert at epsilon=0; pinned anyway.");
ABSL_FLAG(bool, rotate_seat, true, "search_seat = g % 4.");
ABSL_FLAG(bool, use_opponent_model, true, "=> opponent_mode = kPolicy.");
ABSL_FLAG(bool, nonlinear_value_head, false, "Candidate value head.");
ABSL_FLAG(bool, opponent_nonlinear_value_head, false, "Opponent value head.");
ABSL_FLAG(bool, verbose_diagnostics, true, "Required for the label telemetry.");
ABSL_FLAG(bool, check_strategic_state, true, "Bypass search at non-strategic states.");
ABSL_FLAG(bool, disable_time_limit, true, "=> relative_time_budget_ms resolves to infinity.");
ABSL_FLAG(double, root_prior_temperature, 1.0, "Root prior temperature.");
ABSL_FLAG(int, fixed_continuation_reserve, 0, "Continuation reserve simulations.");
ABSL_FLAG(int, purchase_combat_budget, 16, "Session knob; inert with no session.");
ABSL_FLAG(double, live_continuation_reserve_seconds, 10.0, "Inert with no session.");
ABSL_FLAG(double, relative_time_budget_ms, 52000.0, "Inert under disable_time_limit.");
ABSL_FLAG(int, max_nodes, 50000, "The reference's effective node ceiling.");
ABSL_FLAG(double, candidate_logit_cap, 10.0, "Candidate evaluator logit cap.");
ABSL_FLAG(double, opponent_logit_cap, 10.0, "Opponent evaluator logit cap.");
ABSL_FLAG(bool, conservative_override_enabled, false, "Overrides OFF.");
ABSL_FLAG(double, conservative_covered_prior_threshold, 0.95, "Inert while overrides are off.");
ABSL_FLAG(int, conservative_meaningful_visit_threshold, 10, "Inert.");
ABSL_FLAG(double, conservative_q_margin_threshold, 0.03, "Inert.");
ABSL_FLAG(double, conservative_stability_checkpoint_fraction, 0.5, "NOT inert: the section 5.3 prefix checkpoint => sim 100 of 200.");
ABSL_FLAG(bool, conservative_continuation_overrides_disabled, true, "Inert while overrides are off.");

// --- Four constants hardcoded in the reference's config block, made explicit
// --- here at exactly the inherited value (dune_search_benchmark.cc:553-556).
ABSL_FLAG(int, min_visit_threshold, 2, "Hardcoded at dune_search_benchmark.cc:553.");
ABSL_FLAG(double, covered_prior_threshold, 0.50, "Hardcoded at dune_search_benchmark.cc:554. A coverage-rule operand.");

// --- Five DuneSearchConfig fields with no flag and no assignment in the
// --- reference's config block: it runs them at struct defaults. Pinned here.
ABSL_FLAG(double, dirichlet_alpha_total, 0.0, "dune_puct_is_mcts.h:61. KataGo alpha scaling OFF.");
ABSL_FLAG(double, forced_playouts_k, 0.0, "dune_puct_is_mcts.h:65. Forced playouts OFF.");
ABSL_FLAG(bool, root_noise_fpu_zero, false, "dune_puct_is_mcts.h:68.");
ABSL_FLAG(double, training_root_prior_temperature, 1.0, "dune_puct_is_mcts.h:73.");
ABSL_FLAG(int, max_search_decision_depth, -1, "dune_puct_is_mcts.h:92. Unbounded.");

namespace open_spiel {
namespace {

// ---------------------------------------------------------------------------
// Serialization helpers. See the header comment: json.cc's default double
// formatting must never touch a registered measurement.
// ---------------------------------------------------------------------------

// %.17g is max_digits10 for IEEE double: parsing the string reproduces the bits.
// Identical to dune_pwo3_teacher_audit.cc:306 / dune_pwo3_prior_recovery.cc:145.
std::string FmtExact(double v) { return absl::StrFormat("%.17g", v); }

// The hex-float form: the one representation whose round-trip does not depend on
// a decimal parser agreeing with a decimal printer.
std::string FmtHex(double v) { return absl::StrFormat("%a", v); }

// The LEGACY six-decimal projection, produced by CALLING open_spiel's own json.cc
// (json.cc:299-302, std::to_string) rather than by re-implementing its formatting.
// This is the string the section 7.1 gate compares against the reference's stored
// literal. Re-implementing "%.6f" in C++ or Python would be a second instrument
// for the very quantity under audit.
std::string LegacyProjection(double v) { return json::ToString(json::Value(v)); }

// One double -> two string fields. Never a bare JSON number.
void PutDouble(json::Object* o, const std::string& name, double v) {
  (*o)[name + "_exact"] = FmtExact(v);
  (*o)[name + "_legacy6"] = LegacyProjection(v);
}

// One double vector -> two string arrays.
void PutDoubleVec(json::Object* o, const std::string& name,
                  const std::vector<double>& v) {
  json::Array ex, lg;
  for (double x : v) {
    ex.push_back(FmtExact(x));
    lg.push_back(LegacyProjection(x));
  }
  (*o)[name + "_exact"] = ex;
  (*o)[name + "_legacy6"] = lg;
}

json::Array ToJsonArray(const std::vector<Action>& v) {
  json::Array a;
  for (Action x : v) a.push_back(static_cast<int64_t>(x));
  return a;
}

json::Array ToJsonArray(const std::vector<int>& v) {
  json::Array a;
  for (int x : v) a.push_back(static_cast<int64_t>(x));
  return a;
}

const char* PhaseName(dune_imperium::GamePhase p) {
  switch (p) {
    case dune_imperium::GamePhase::kLeaderOfferChance: return "LeaderOfferChance";
    case dune_imperium::GamePhase::kLeaderDraft:       return "LeaderDraft";
    case dune_imperium::GamePhase::kDeal:              return "Deal";
    case dune_imperium::GamePhase::kRoundStart:        return "RoundStart";
    case dune_imperium::GamePhase::kAgentTurns:        return "AgentTurns";
    case dune_imperium::GamePhase::kRevealTurns:       return "RevealTurns";
    case dune_imperium::GamePhase::kCombat:            return "Combat";
    case dune_imperium::GamePhase::kMakers:            return "Makers";
    case dune_imperium::GamePhase::kRecall:            return "Recall";
    case dune_imperium::GamePhase::kTerminal:          return "Terminal";
  }
  return "Unknown";
}

// The registered role-name set (work order CP0.6 guard 2). pwo2::RoleName's
// SCREAMING_CASE spellings are the PWO-2/PWO-3 wire format; the registration and
// the section 9.3 floors name roles as AgentPrimary / Purchase / ... so the
// stream carries the registration's spelling and the analyzer's floor table can
// be read against the registration without a translation step.
const char* RegisteredRoleName(DuneDecisionRole role) {
  switch (role) {
    case DuneDecisionRole::kForcedOrBookkeeping: return "ForcedOrBookkeeping";
    case DuneDecisionRole::kLeaderSelection:     return "LeaderSelection";
    case DuneDecisionRole::kAgentPrimary:        return "AgentPrimary";
    case DuneDecisionRole::kAgentContinuation:   return "AgentContinuation";
    case DuneDecisionRole::kPurchase:            return "Purchase";
    case DuneDecisionRole::kCombatIntrigue:      return "CombatIntrigue";
    case DuneDecisionRole::kOtherOptional:       return "OtherOptional";
  }
  return "UNKNOWN";
}

// Argmax with LOWEST-ID tie-break over a weight vector aligned to `actions`.
// Mirrors GetRootArgmaxAction (dune_puct_is_mcts.cc:715-729): iterate in legal
// order, strict `>`, so the first maximum wins. legal_actions is ascending.
Action ArgmaxLowestId(const std::vector<Action>& actions,
                      const std::vector<double>& weights) {
  if (actions.empty()) return kInvalidAction;
  Action best = actions[0];
  double best_w = weights[0];
  for (size_t i = 1; i < actions.size(); ++i) {
    if (weights[i] > best_w) {
      best_w = weights[i];
      best = actions[i];
    }
  }
  return best;
}

// The NVIDIA kernel-module version, read from procfs (the driver reporting
// itself) rather than by transcribing an nvidia-smi subprocess.
std::string DriverVersionLine() {
  std::ifstream f("/proc/driver/nvidia/version");
  if (!f) return "unavailable";
  std::string line;
  std::getline(f, line);
  return line;
}

// dune_search_benchmark.cc:204-271, verbatim in behaviour: the frozen raw-policy
// opponents of the reference configuration. Reproduced rather than reused because
// it lives in that binary's anonymous namespace; the evaluator it drives is the
// same DuneNNEvaluator on the same shared model.
class DuneGreedyBot : public Bot {
 public:
  DuneGreedyBot(std::unique_ptr<DuneNNEvaluator> evaluator, int seed, double temperature)
      : evaluator_(std::move(evaluator)), rng_(seed), temperature_(temperature) {}

  Action Step(const State& state) override {
    if (state.IsTerminal()) return kInvalidAction;
    ActionsAndProbs prior = evaluator_->Prior(state);
    return SelectBestAction(prior, state);
  }
  bool ProvidesPolicy() override { return true; }
  ActionsAndProbs GetPolicy(const State& state) override { return evaluator_->Prior(state); }
  std::pair<ActionsAndProbs, Action> StepWithPolicy(const State& state) override {
    ActionsAndProbs policy = GetPolicy(state);
    return {policy, SelectBestAction(policy, state)};
  }
  void Restart() override {}
  void RestartAt(const State& state) override {}

 private:
  double RandomNumber() { return absl::Uniform(rng_, 0.0, 1.0); }

  Action SelectBestAction(const ActionsAndProbs& policy, const State& state) {
    if (policy.empty()) {
      auto legals = state.LegalActions();
      return legals.empty() ? kInvalidAction : legals.front();
    }
    double temp = temperature_;
    if (temp <= 0.0) {
      Action best_action = policy[0].first;
      double best_prob = policy[0].second;
      for (const auto& ap : policy) {
        if (ap.second > best_prob) {
          best_prob = ap.second;
          best_action = ap.first;
        }
      }
      return best_action;
    }
    ActionsAndProbs scaled_policy;
    scaled_policy.reserve(policy.size());
    double sum = 0.0;
    double inv_temp = 1.0 / temp;
    for (const auto& ap : policy) {
      double p = std::pow(std::max(ap.second, 1e-12), inv_temp);
      scaled_policy.push_back({ap.first, p});
      sum += p;
    }
    for (auto& ap : scaled_policy) ap.second /= sum;
    return SampleAction(scaled_policy, RandomNumber()).first;
  }

  std::unique_ptr<DuneNNEvaluator> evaluator_;
  std::mt19937 rng_;
  double temperature_;
};

// dune_search_benchmark.cc:146-202, verbatim: base VP plus the four endgame
// bonuses. The terminal outcome is persisted per trajectory for TRAJECTORY-LEVEL
// REPORTING ONLY and is never a causal label for an individual move
// (registration section 3.1, plan section 8.3.2, ruling O3).
int GetTrueFinalVp(const dune_imperium::DuneImperiumState* dune_state, int p) {
  int base_vp = dune_state->GetPlayerVpForTesting(p);
  int endgame_vp = 0;

  const auto& hand = dune_state->GetIntrigueHandForTesting(p);
  for (int intrigue_id : hand) {
    const auto* intrigue = dune_imperium::FindIntrigueCardById(intrigue_id);
    if (intrigue && (intrigue->phase_mask & dune_imperium::kIntriguePhaseEndgameMask) != 0) {
      endgame_vp += dune_state->EndgameIntrigueVpBonusForTesting(p, intrigue_id);
    }
  }

  const auto& tech_tiles = dune_state->GetPlayerTechTilesForTesting(p);
  if (std::find(tech_tiles.begin(), tech_tiles.end(), 6) != tech_tiles.end()) {
    int tsmf_count = 0;
    auto count_tsmf = [&](const std::vector<int>& cards) {
      for (int id : cards) if (id == dune_imperium::kCardTheSpiceMustFlow) tsmf_count++;
    };
    auto* mutable_state = const_cast<dune_imperium::DuneImperiumState*>(dune_state);
    count_tsmf(mutable_state->GetPlayerDrawDeckForTesting(p));
    count_tsmf(mutable_state->GetPlayerDiscardForTesting(p));
    count_tsmf(mutable_state->GetPlayerHandForTesting(p));
    count_tsmf(dune_state->GetPlayedAgentCardsForTesting(p));
    count_tsmf(dune_state->GetRevealedCardsForTesting(p));
    if (tsmf_count >= 2) endgame_vp += 1;
  }

  bool all_3 = true;
  for (int f = 0; f < 4; ++f) {
    if (dune_state->GetPlayerInfluenceForTesting(p, static_cast<dune_imperium::Faction>(f)) < 3) {
      all_3 = false;
    }
  }
  if (all_3) endgame_vp += 1;

  if (std::find(tech_tiles.begin(), tech_tiles.end(), 14) != tech_tiles.end()) {
    int low_influence_factions = 0;
    for (int f = 0; f < 4; ++f) {
      if (dune_state->GetPlayerInfluenceForTesting(p, static_cast<dune_imperium::Faction>(f)) <= 1) {
        low_influence_factions++;
      }
    }
    endgame_vp += low_influence_factions;
  }
  return base_vp + endgame_vp;
}

// The section 2.6 pin, assembled once and reused for every seat's config so the
// only field that varies across seats is `.seed`.
DuneSearchConfig MakePinnedConfig(uint64_t seat_seed) {
  DuneSearchConfig config{
      .max_simulations = absl::GetFlag(FLAGS_max_simulations),
      .relative_time_budget_ms =
          absl::GetFlag(FLAGS_disable_time_limit)
              ? std::numeric_limits<double>::infinity()
              : absl::GetFlag(FLAGS_relative_time_budget_ms),
      .max_nodes = absl::GetFlag(FLAGS_max_nodes),
      .puct_c = absl::GetFlag(FLAGS_puct_c),
      .opponent_mode = absl::GetFlag(FLAGS_use_opponent_model)
                           ? SearchOpponentMode::kPolicy
                           : SearchOpponentMode::kMaxN,
      .temperature = absl::GetFlag(FLAGS_temperature),
      .opponent_temperature = absl::GetFlag(FLAGS_simulated_opponent_temperature),
      .max_world_samples = absl::GetFlag(FLAGS_max_world_samples),
      .utility_divisor = absl::GetFlag(FLAGS_utility_divisor),
      .min_visit_threshold = absl::GetFlag(FLAGS_min_visit_threshold),
      .covered_prior_threshold = absl::GetFlag(FLAGS_covered_prior_threshold),
      .seed = seat_seed,
      .final_policy_type = DuneISMCTSFinalPolicyType::kNormalizedVisitCount,
      .dirichlet_epsilon = absl::GetFlag(FLAGS_dirichlet_epsilon),
      .dirichlet_alpha = absl::GetFlag(FLAGS_dirichlet_alpha),
      .use_observation_string = true,
      .verbose_diagnostics = absl::GetFlag(FLAGS_verbose_diagnostics),
      .check_strategic_state = absl::GetFlag(FLAGS_check_strategic_state),
      .root_prior_temperature = absl::GetFlag(FLAGS_root_prior_temperature),
  };
  // The five struct-default pins (registration section 2.6). Declared rather than
  // inherited: a struct default is one edit away from silently redefining the
  // controller, and the designated-initializer comment at dune_puct_is_mcts.h:140-147
  // records a positional-initializer bug that once slid use_observation_string=true
  // into dirichlet_alpha_total as 1.0, switching KataGo alpha scaling ON.
  config.dirichlet_alpha_total = absl::GetFlag(FLAGS_dirichlet_alpha_total);
  config.forced_playouts_k = absl::GetFlag(FLAGS_forced_playouts_k);
  config.root_noise_fpu_zero = absl::GetFlag(FLAGS_root_noise_fpu_zero);
  config.training_root_prior_temperature = absl::GetFlag(FLAGS_training_root_prior_temperature);
  config.max_search_decision_depth = absl::GetFlag(FLAGS_max_search_decision_depth);

  config.fixed_session_limit = absl::GetFlag(FLAGS_max_simulations);
  config.fixed_continuation_reserve = absl::GetFlag(FLAGS_fixed_continuation_reserve);
  config.purchase_combat_budget = absl::GetFlag(FLAGS_purchase_combat_budget);
  config.live_continuation_reserve_seconds = absl::GetFlag(FLAGS_live_continuation_reserve_seconds);
  config.model_checkpoint_path = absl::GetFlag(FLAGS_model_checkpoint);
  config.conservative_override_enabled = absl::GetFlag(FLAGS_conservative_override_enabled);
  config.conservative_covered_prior_threshold = absl::GetFlag(FLAGS_conservative_covered_prior_threshold);
  config.conservative_meaningful_visit_threshold = absl::GetFlag(FLAGS_conservative_meaningful_visit_threshold);
  config.conservative_q_margin_threshold = absl::GetFlag(FLAGS_conservative_q_margin_threshold);
  config.conservative_stability_checkpoint_fraction = absl::GetFlag(FLAGS_conservative_stability_checkpoint_fraction);
  config.conservative_continuation_overrides_disabled = absl::GetFlag(FLAGS_conservative_continuation_overrides_disabled);
  return config;
}

std::shared_ptr<SharedDunePolicyValueNetImpl> LoadModel(
    const std::string& path, int64_t obs, int64_t act, int hidden, int blocks,
    bool nonlinear_head, torch::Device dev) {
  auto m = std::make_shared<SharedDunePolicyValueNetImpl>(obs, hidden, act, blocks, nonlinear_head);
  m->eval();
  try {
    torch::serialize::InputArchive archive;
    archive.load_from(path, dev);
    m->load(archive);
  } catch (const c10::Error& e) {
    std::cerr << "STOP: failed to load " << path << ": " << e.msg() << "\n";
    std::exit(1);
  }
  m->to(dev);
  return m;
}

}  // namespace
}  // namespace open_spiel

using namespace open_spiel;

int main(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);

  // --- Preconditions. Every one of these is a registered STOP; failing loudly
  // --- before any GPU work is what keeps a misconfigured run from producing
  // --- plausible data. ------------------------------------------------------
  const int num_threads = absl::GetFlag(FLAGS_threads);
  if (num_threads != 1) {
    std::cerr << "STOP: --threads must be 1 (registration section 2.4).\n";
    return 1;
  }
  const int base_seed = absl::GetFlag(FLAGS_seed);
  // Registration section 4.3: the final-gate base-seed reservation, made
  // mechanical rather than a matter of operator care.
  if (base_seed >= 9000000 && base_seed <= 9999999) {
    std::cerr << "STOP: --seed " << base_seed << " lies in the RESERVED final-gate "
                 "base-seed range [9000000, 9999999] (registration section 4.3).\n";
    return 1;
  }
  if (absl::GetFlag(FLAGS_max_simulations) != 200) {
    std::cerr << "STOP: --max_simulations must be exactly 200 (registration section 2.2). "
                 "200 is the budget the section 8.3.1 evidence was measured at, not a "
                 "compute knob.\n";
    return 1;
  }
  if (absl::GetFlag(FLAGS_model_checkpoint).empty()) {
    std::cerr << "STOP: --model_checkpoint is required.\n";
    return 1;
  }
  if (absl::GetFlag(FLAGS_engine_tree_hash).empty()) {
    std::cerr << "STOP: --engine_tree_hash is REQUIRED. The stream is only "
                 "meaningful against the frozen engine that produced it.\n";
    return 1;
  }
  if (absl::GetFlag(FLAGS_registration_sha256).empty()) {
    std::cerr << "STOP: --registration_sha256 is REQUIRED (the frozen registration's "
                 "content hash, 57ec6310...0733).\n";
    return 1;
  }
  if (absl::GetFlag(FLAGS_amendment_sha256).empty()) {
    std::cerr << "STOP: --amendment_sha256 is REQUIRED (docs/PWO4_AMENDMENT_1_GATE_SEMANTICS.md). "
                 "Amendment 1 defines the gate semantics this stream is judged by; a stream that "
                 "cannot name its amendment cannot be adjudicated.\n";
    return 1;
  }

  // --- The frozen controller MODES, asserted fatally. -----------------------
  // Each of these changes WHAT IS MEASURED, not how fast it is measured, so a
  // wrong value must not produce data at all. They are flags rather than
  // constants precisely so a launcher has to state them: registration section 2.6's
  // rule is that no flag default defines a measurement-of-record configuration.
  {
    struct ModePin { const char* name; bool ok; std::string got; const char* want; };
    const std::vector<ModePin> pins = {
      {"--use_session", absl::GetFlag(FLAGS_use_session) == false,
       absl::GetFlag(FLAGS_use_session) ? "true" : "false", "false"},
      {"--controller_mode", absl::GetFlag(FLAGS_controller_mode) == "single",
       absl::GetFlag(FLAGS_controller_mode), "single"},
      {"--fresh_search_roles", absl::GetFlag(FLAGS_fresh_search_roles).empty(),
       absl::GetFlag(FLAGS_fresh_search_roles), "(empty)"},
      {"--policy_only", absl::GetFlag(FLAGS_policy_only) == false,
       absl::GetFlag(FLAGS_policy_only) ? "true" : "false", "false"},
      {"--live_deadline", absl::GetFlag(FLAGS_live_deadline) == false,
       absl::GetFlag(FLAGS_live_deadline) ? "true" : "false", "false"},
      {"--force_swordmaster_rounds", absl::GetFlag(FLAGS_force_swordmaster_rounds) == 0,
       std::to_string(absl::GetFlag(FLAGS_force_swordmaster_rounds)), "0"},
      {"--grant_swordmaster_round", absl::GetFlag(FLAGS_grant_swordmaster_round) == 0,
       std::to_string(absl::GetFlag(FLAGS_grant_swordmaster_round)), "0"},
    };
    bool all_ok = true;
    for (const ModePin& p : pins) {
      if (!p.ok) {
        std::cerr << "STOP: " << p.name << " is FROZEN at " << p.want << " but was given '"
                  << p.got << "' (registration section 2.6).\n";
        all_ok = false;
      }
    }
    if (!all_ok) return 1;
  }
  if (absl::GetFlag(FLAGS_disable_time_limit) != true) {
    std::cerr << "STOP: --disable_time_limit is FROZEN true, so no clock can truncate a "
                 "count-terminated search (registration section 2.2).\n";
    return 1;
  }
  if (absl::GetFlag(FLAGS_temperature) != 0.0) {
    std::cerr << "STOP: --temperature is FROZEN at 0.0 (registration section 2.6).\n";
    return 1;
  }
  if (absl::GetFlag(FLAGS_dirichlet_epsilon) != 0.0) {
    std::cerr << "STOP: --dirichlet_epsilon is FROZEN at 0.0: root noise OFF. A noised root "
                 "would separate `priors` from `raw_prior_vector` and break the Amendment-1 "
                 "section 4 identity assert.\n";
    return 1;
  }
  if (absl::GetFlag(FLAGS_root_prior_temperature) != 1.0) {
    std::cerr << "STOP: --root_prior_temperature is FROZEN at 1.0, for the same reason as "
                 "--dirichlet_epsilon.\n";
    return 1;
  }
  if (absl::GetFlag(FLAGS_covered_prior_threshold) != 0.50 ||
      absl::GetFlag(FLAGS_min_visit_threshold) != 2) {
    std::cerr << "STOP: the coverage-rule operands are FROZEN at min_visit_threshold=2 and "
                 "covered_prior_threshold=0.50 (registration section 6.2). A divergence would "
                 "silently change the weight rule.\n";
    return 1;
  }
  if (absl::GetFlag(FLAGS_conservative_stability_checkpoint_fraction) != 0.5) {
    std::cerr << "STOP: --conservative_stability_checkpoint_fraction is FROZEN at 0.5 -- it is "
                 "the section 5.3 prefix checkpoint, NOT inert.\n";
    return 1;
  }
  const std::string audit_path = absl::GetFlag(FLAGS_audit_jsonl_path);
  const std::string labels_path = absl::GetFlag(FLAGS_labels_jsonl_path);
  const std::string games_path = absl::GetFlag(FLAGS_games_jsonl_path);
  const std::string manifest_path = absl::GetFlag(FLAGS_manifest_json_path);
  if (audit_path.empty() || labels_path.empty() || games_path.empty() || manifest_path.empty()) {
    std::cerr << "STOP: --audit_jsonl_path, --labels_jsonl_path, --games_jsonl_path "
                 "and --manifest_json_path are all required.\n";
    return 1;
  }

  at::set_num_threads(1);
  at::set_num_interop_threads(1);
  torch::InferenceMode inference_guard;

  const std::string model_ckpt = absl::GetFlag(FLAGS_model_checkpoint);
  const std::string model_sha = ComputeFileSHA256(model_ckpt);
  const std::string self_sha = ComputeFileSHA256(argv[0]);

  std::shared_ptr<const Game> game = LoadGame("dune_imperium");
  const int64_t obs_size = game->InformationStateTensorSize();
  const int64_t action_size = game->NumDistinctActions();
  torch::Device device = torch::cuda::is_available() ? torch::Device(torch::kCUDA)
                                                     : torch::Device(torch::kCPU);

  auto search_model = LoadModel(model_ckpt, obs_size, action_size,
                                absl::GetFlag(FLAGS_hidden_dim),
                                absl::GetFlag(FLAGS_num_blocks),
                                absl::GetFlag(FLAGS_nonlinear_value_head), device);

  // --opponent_checkpoint empty resolves to the candidate model driven as a
  // greedy raw-policy bot (registration section 2.6) -- the frozen raw-policy
  // opponents of the reference configuration.
  std::string opp_ckpt = absl::GetFlag(FLAGS_opponent_checkpoint);
  if (opp_ckpt.empty()) opp_ckpt = model_ckpt;
  const std::string opp_sha = ComputeFileSHA256(opp_ckpt);
  auto opponent_model = LoadModel(opp_ckpt, obs_size, action_size,
                                  absl::GetFlag(FLAGS_opp_hidden_dim),
                                  absl::GetFlag(FLAGS_opp_num_blocks),
                                  absl::GetFlag(FLAGS_opponent_nonlinear_value_head), device);

  const int total_games = absl::GetFlag(FLAGS_games);
  const int start_episode_id = absl::GetFlag(FLAGS_start_episode_id);
  const float candidate_logit_cap = static_cast<float>(absl::GetFlag(FLAGS_candidate_logit_cap));
  const float opponent_logit_cap = static_cast<float>(absl::GetFlag(FLAGS_opponent_logit_cap));
  const bool check_strategic = absl::GetFlag(FLAGS_check_strategic_state);
  const int max_sims = absl::GetFlag(FLAGS_max_simulations);
  const int stability_checkpoint_sim = static_cast<int>(
      std::floor(max_sims * absl::GetFlag(FLAGS_conservative_stability_checkpoint_fraction)));

  std::ofstream audit_out(audit_path);
  std::ofstream labels_out(labels_path);
  std::ofstream games_out(games_path);
  if (!audit_out || !labels_out || !games_out) {
    std::cerr << "STOP: cannot open an output path.\n";
    return 1;
  }

  int64_t n_audit_rows = 0, n_label_rows = 0;
  int64_t n_search_expected_true = 0, n_search_expected_false = 0;
  int64_t n_single_action = 0, n_non_strategic = 0;
  const auto run_start = std::chrono::steady_clock::now();

  // =========================================================================
  // Game loop. Sequential by construction: --threads=1 with one game in flight
  // is the pinned configuration (registration section 2.4) and the reference's
  // per-game state is thread-local anyway.
  // =========================================================================
  for (int offset = 0; offset < total_games; ++offset) {
    const int g = start_episode_id + offset;

    // --- The inherited seed derivation, verbatim (registration section 4.1) ---
    const uint64_t game_seed = dune_seed::DeriveSeed(base_seed, dune_seed::kStreamBlueprint, g);
    std::mt19937 game_rng(game_seed);
    // A SEPARATE stream. Constructed so it can never consume game_rng; inert here
    // because there is no swordmaster forcing (registration section 2.6).
    std::mt19937 force_rng(dune_seed::DeriveSeed(base_seed, dune_seed::kStreamSearchGate, g));
    (void)force_rng;
    const int search_seat = absl::GetFlag(FLAGS_rotate_seat) ? (g % 4) : 0;

    auto search_evaluator = std::make_shared<DuneNNEvaluator>(search_model, device, candidate_logit_cap);

    std::unique_ptr<DunePUCTISMCTSBot> search_bot;
    std::vector<std::unique_ptr<Bot>> bots(4);
    std::vector<uint64_t> seat_seeds(4, 0);
    for (int p = 0; p < 4; ++p) {
      // EXACTLY ONE game_rng() draw per seat, in seat order, for EVERY seat --
      // searched or not. dune_search_benchmark.cc:526 sits outside the
      // seat_is_searched branch at :527, and :520-525 warns that moving or
      // conditionalising this draw "silently reshuffles every downstream chance
      // realization". A generator that drew only for the searched seat would
      // desynchronize game_rng from the reference on the very first game.
      const uint64_t seat_seed = game_rng();
      seat_seeds[p] = seat_seed;
      if (p == search_seat) {
        search_bot = std::make_unique<DunePUCTISMCTSBot>(MakePinnedConfig(seat_seed), search_evaluator);
      } else {
        auto local_opp_eval = std::make_unique<DuneNNEvaluator>(opponent_model, device, opponent_logit_cap);
        bots[p] = std::make_unique<DuneGreedyBot>(
            std::move(local_opp_eval), static_cast<int>(seat_seed),
            absl::GetFlag(FLAGS_external_opponent_temperature));
      }
    }
    const uint64_t search_config_seed = seat_seeds[search_seat];

    // A MIRROR of DunePUCTISMCTSBot::search_count_, which is private. Step()
    // increments it exactly once per call, unconditionally and AFTER RunSearch
    // returns (dune_puct_is_mcts.cc:1126), and this loop calls Step() exactly once
    // per searched-seat decision -- so the mirror is exact by construction.
    //
    // It is NOT an unobservable bookkeeping detail: `final_action_sampling_seed`
    // is derived from it, and on a coverage-fallback row RunSearch returns the raw
    // prior rather than a one-hot policy, so Step()'s SampleAction draw genuinely
    // selects the action. A mirror that drifted would change executed actions on
    // those rows and the section 7.1 fidelity gate would surface it.
    int search_count = 0;

    std::unique_ptr<State> state = game->NewInitialState();
    std::vector<Action> full_history;
    int decision_index = 0;
    int game_length = 0;
    std::map<std::string, int64_t> game_role_counts;
    int64_t game_label_rows = 0;
    std::array<int, 4> specimen_conversions{};
    std::vector<json::Object> pending_audit, pending_labels;

    while (!state->IsTerminal()) {
      if (++game_length > 5000) {
        std::cerr << "STOP: infinite-loop guard hit in game " << g << ".\n";
        return 1;
      }

      if (state->IsChanceNode()) {
        auto outcomes = state->ChanceOutcomes();
        // The SAME call on the SAME code path as the reference (:741-750). Never
        // a reimplemented draw: the consumption of game_rng is what keeps the
        // chance realizations aligned.
        Action action = (game->GetType().chance_mode == GameType::ChanceMode::kSampledStochastic)
                            ? outcomes.front().first
                            : SampleAction(outcomes, game_rng).first;
        state->ApplyAction(action);
        full_history.push_back(action);
        continue;
      }

      const Player current_player = state->CurrentPlayer();
      Action chosen_action = kInvalidAction;

      if (current_player != search_seat) {
        chosen_action = bots[current_player]->Step(*state);
        if (chosen_action >= dune_imperium::kActionConvertSpecimenToTroop0 &&
            chosen_action <= dune_imperium::kActionConvertSpecimenToTroop0 + 12 &&
            current_player >= 0 && current_player < 4) {
          ++specimen_conversions[current_player];
        }
        state->ApplyAction(chosen_action);
        full_history.push_back(chosen_action);
        continue;
      }

      // ===================================================================
      // A searched-seat decision. Everything below produces exactly one audit
      // record, and a label row iff the search actually ran to completion.
      // ===================================================================
      const std::vector<Action> legal_actions = state->LegalActions();
      const int n_legal = static_cast<int>(legal_actions.size());
      const size_t history_length = full_history.size();

      // --- CP0.4: search_expected, evaluated BEFORE Step(), from the reference
      // --- controller's own pre-loop predicate. This REPLACES any
      // --- fallback-derived notion of eligibility. The n_legal >= 2 term is the
      // --- single-legal-action early return at dune_puct_is_mcts.cc:779; the
      // --- IsStrategicState term is the check at :788, the same predicate
      // --- RunSearch applies.
      const bool state_is_strategic = IsStrategicState(*state, current_player);
      const bool search_expected = (n_legal >= 2) && (!check_strategic || state_is_strategic);

      // --- CP0.6: decision_role emitted LIVE. On the fresh path the reference's
      // --- decision_role is dead (""), which makes the section 9.3 per-role floor
      // --- unmeasurable. has_active_session is FALSE: there is no session on this
      // --- path, which is exactly what dune_search_benchmark.cc:784-785 passes.
      const DuneDecisionRole role = ClassifyDuneDecisionRole(*state, current_player, false);
      const std::string role_name = RegisteredRoleName(role);
      // Guard 1 and guard 2 (work order CP0.6): non-empty, and in the registered
      // role set. Guard 3 -- equality against an INDEPENDENTLY reconstructed
      // classification -- cannot be discharged here: calling the classifier twice
      // on this same live state would prove only that the function is
      // deterministic. It is discharged by dune_pwo4_role_verify, which replays
      // `history_length` actions from a fresh initial state and reclassifies.
      // That is why history_length and the full per-game action history below are
      // part of the schema rather than a convenience.
      if (role_name == "UNKNOWN" || role_name.empty()) {
        std::cerr << "STOP: decision_role is empty or outside the registered role set "
                     "at game " << g << " decision " << decision_index << ".\n";
        return 1;
      }

      const auto* dune_state = dynamic_cast<const dune_imperium::DuneImperiumState*>(state.get());
      const int live_round = dune_state ? dune_state->GetCurrentRound() : -1;
      const std::string live_phase = dune_state ? PhaseName(dune_state->phase()) : "";

      const int search_count_before = search_count;
      const auto step_start = std::chrono::steady_clock::now();
      // The SAME entry point the reference drives (:804): fresh full search per
      // decision, sample-on-fallback selection included.
      chosen_action = search_bot->Step(*state);
      const auto step_end = std::chrono::steady_clock::now();
      const double step_duration_ms = std::chrono::duration<double, std::milli>(step_end - step_start).count();
      const int search_count_after = ++search_count;

      DuneSearchResult last_res = search_bot->GetLastSearchResult();
      last_res.diagnostics.selected_action = chosen_action;
      const SearchDiagnostics& diag = last_res.diagnostics;

      // --- The post-search final-action sampling draw, reconstructed.
      // Step() computes these internally and exposes none of them
      // (dune_puct_is_mcts.cc:1126-1130). They are recomputed here from the same
      // inputs by the same functions, and the reconstruction is SELF-VALIDATING:
      // Amendment 1 section 3 requires the analyzer to recompute the inverse-CDF
      // sample from (deployed_policy, r_val) and match deployed_action on every
      // coverage-fallback row. If any step of this reconstruction were wrong,
      // that check would fail on those rows.
      //
      // Amendment 1 section 4: this seed is INERT only on non-fallback rows,
      // where the policy is one-hot and SampleAction returns the argmax whatever
      // r_val is. On a coverage fallback the policy is the raw-prior
      // DISTRIBUTION and the draw genuinely selects the action.
      const uint64_t final_action_sampling_seed = dune_seed::Combine(
          search_config_seed, dune_seed::kStreamBlueprint, search_count_after);
      std::mt19937 step_rng_replica(final_action_sampling_seed);
      const double final_action_sampling_r_val =
          absl::Uniform(step_rng_replica, 0.0, 1.0);

      // --- Registration section 2.3 / work order CP0.4: a game containing a
      // --- short-searched decision is not a coherent fully searched trajectory,
      // --- which is the section 8.3 premise. VOID THE GAME AND STOP.
      if (search_expected && (last_res.simulations_completed != max_sims ||
                              diag.inherited_root_visits != 0)) {
        std::cerr << "STOP (registration section 2.3): game " << g << " decision "
                  << decision_index << " was search_expected but completed "
                  << last_res.simulations_completed << " simulations with "
                  << diag.inherited_root_visits << " inherited root visits. "
                  << "The GAME is inadmissible; generation halts for diagnosis.\n";
        return 1;
      }
      // The independent cross-check of CP0.4. This expression is a CROSS-CHECK,
      // never the definition: a disagreement means the controller's predicate and
      // the observed fallback taxonomy have diverged, and the definition must be
      // re-derived from source before proceeding.
      const bool diagnostic_expr = (n_legal >= 2) && (last_res.fallback_reason != "non_strategic_state");
      if (diagnostic_expr != search_expected) {
        std::cerr << "STOP: search_expected (" << search_expected << ") disagrees with the "
                     "diagnostic expression (" << diagnostic_expr << ") at game " << g
                  << " decision " << decision_index << ", fallback_reason='"
                  << last_res.fallback_reason << "'.\n";
        return 1;
      }

      if (search_expected) {
        ++n_search_expected_true;
      } else {
        ++n_search_expected_false;
        if (n_legal < 2) ++n_single_action; else ++n_non_strategic;
      }

      // ------------------------------------------------------------------
      // The common block: every field the reference emits, plus the live
      // replacements for the ones that are dead on this path.
      // ------------------------------------------------------------------
      json::Object row;
      row["game_index"] = static_cast<int64_t>(g);
      row["episode_id"] = static_cast<int64_t>(g);          // reference's name
      row["decision_index"] = static_cast<int64_t>(decision_index);
      row["history_length"] = static_cast<int64_t>(history_length);
      row["acting_player"] = static_cast<int64_t>(current_player);
      row["player"] = static_cast<int64_t>(current_player);  // reference's name
      row["searched_seat"] = static_cast<int64_t>(search_seat);
      row["search_seat"] = static_cast<int64_t>(search_seat);  // reference's name
      row["round"] = static_cast<int64_t>(live_round);
      row["phase"] = live_phase;
      row["decision_role"] = role_name;
      row["is_strategic"] = state_is_strategic;
      row["n_legal"] = static_cast<int64_t>(n_legal);
      row["legal_actions"] = ToJsonArray(legal_actions);
      row["executed_action"] = static_cast<int64_t>(chosen_action);
      row["action_chosen"] = static_cast<int64_t>(chosen_action);  // reference's name
      if (dune_state != nullptr) {
        row["action_chosen_string"] = dune_state->ActionToString(current_player, chosen_action);
      }
      row["simulations_completed"] = static_cast<int64_t>(last_res.simulations_completed);
      row["inherited_root_visits"] = static_cast<int64_t>(diag.inherited_root_visits);
      row["fallback_reason"] = last_res.fallback_reason;
      row["used_fallback"] = last_res.used_fallback;
      row["timeout_status"] = last_res.timeout_status;
      row["search_expected"] = search_expected;
      row["search_expected_diagnostic_expr"] = diagnostic_expr;
      row["inference_count"] = static_cast<int64_t>(last_res.inference_count);

      // Seed provenance (registration section 4.1a). No single scalar reproduces a
      // search; recording all four makes the counter's trajectory auditable
      // instead of inferred.
      row["seed"] = static_cast<int64_t>(base_seed);
      row["game_seed"] = static_cast<int64_t>(game_seed);
      row["search_config_seed"] = static_cast<int64_t>(search_config_seed);
      row["search_count_before"] = static_cast<int64_t>(search_count_before);
      row["search_count_after"] = static_cast<int64_t>(search_count_after);
      // POST-search. Seeds ONLY the draw that samples the final action from
      // res.policy. NEVER label this the search seed (registration section 4.1a
      // error 4). It is inert wherever the policy is one-hot -- i.e. wherever the
      // coverage gate did NOT fire -- and LIVE where it did (Amendment 1 section 4).
      row["final_action_sampling_seed"] = static_cast<int64_t>(final_action_sampling_seed);
      PutDouble(&row, "final_action_sampling_r_val", final_action_sampling_r_val);
      // uint64 decimal strings. json::Value takes int64_t, so a seed above 2^63
      // serializes NEGATIVE; the int64 fields above keep reference parity, and
      // these carry the unambiguous value the analyzer recomputes against.
      row["game_seed_u64"] = std::to_string(game_seed);
      row["search_config_seed_u64"] = std::to_string(search_config_seed);
      row["final_action_sampling_seed_u64"] = std::to_string(final_action_sampling_seed);

      // The reference's remaining live telemetry, under the reference's names.
      row["protocol_version"] = diag.protocol_version;
      row["re_root_status"] = diag.re_root_status;
      row["reset_reason"] = diag.reset_reason;
      row["post_chance_branch_miss"] = diag.post_chance_branch_miss;
      row["legality_result"] = diag.legality_result;
      row["unique_nodes"] = static_cast<int64_t>(diag.unique_nodes);
      row["deepest_simulated_round"] = static_cast<int64_t>(diag.deepest_simulated_round);
      row["visit_counts"] = ToJsonArray(diag.visit_counts);
      row["forced_visit_counts"] = ToJsonArray(diag.forced_visit_counts);
      row["pruned_visit_counts"] = ToJsonArray(diag.pruned_visit_counts);
      row["chosen_action_raw_prior_rank"] = static_cast<int64_t>(diag.chosen_action_raw_prior_rank);
      row["action_changed_vs_raw_argmax"] = diag.action_changed_vs_raw_argmax;
      PutDouble(&row, "root_value", diag.root_value);
      PutDouble(&row, "max_depth", diag.max_depth);
      PutDouble(&row, "mean_depth", diag.mean_depth);
      PutDouble(&row, "p95_depth", diag.p95_depth);
      PutDouble(&row, "terminal_leaf_fraction", diag.terminal_leaf_fraction);
      PutDouble(&row, "raw_to_search_policy_kl", diag.raw_to_search_policy_kl);
      PutDouble(&row, "chosen_action_raw_prior_probability", diag.chosen_action_raw_prior_probability);
      PutDoubleVec(&row, "q_values", diag.q_values);
      PutDoubleVec(&row, "priors", diag.priors);
      // Timing. `elapsed_time_ms` is named in section 7.2's CLOSED exclusion list
      // and is what the section 7.4 pilot reads: the reference's own accounting
      // found per-row elapsed_time_ms sums covering 99.7% of both reference runs'
      // wall clock (registration section 9.1), so no second wall-clock field is
      // added here. A generator-only timing field would need an exclusion the
      // closed list does not grant.
      PutDouble(&row, "elapsed_time_ms", last_res.elapsed_time_ms);
      (void)step_duration_ms;

      // The reference's Phase-3 policy-shape block, recomputed on the same inputs
      // by the same formulas (dune_search_benchmark.cc:1008-1044).
      auto shape_stats = [](const std::vector<double>& w, double* norm_h, double* max_p, double* eff_n) {
        double total = 0.0;
        for (double v : w) total += (v > 0.0 ? v : 0.0);
        const int n = static_cast<int>(w.size());
        if (n <= 0 || total <= 0.0) { *norm_h = 0.0; *max_p = 0.0; *eff_n = 0.0; return; }
        double h = 0.0, mx = 0.0;
        for (double v : w) {
          const double p = (v > 0.0 ? v : 0.0) / total;
          if (p > 0.0) h -= p * std::log(p);
          mx = std::max(mx, p);
        }
        *norm_h = (n > 1) ? (h / std::log(static_cast<double>(n))) : 0.0;
        *max_p = mx;
        *eff_n = std::exp(h);
      };
      const std::vector<double>& raw_w = diag.raw_priors.empty() ? diag.priors : diag.raw_priors;
      double rp_h = 0.0, rp_max = 0.0, rp_eff = 0.0;
      shape_stats(raw_w, &rp_h, &rp_max, &rp_eff);
      PutDouble(&row, "raw_prior_norm_entropy", rp_h);
      PutDouble(&row, "raw_prior_max_action_prob", rp_max);
      PutDouble(&row, "raw_prior_eff_action_count", rp_eff);
      std::vector<double> visit_w;
      visit_w.reserve(diag.visit_counts.size());
      for (int v : diag.visit_counts) visit_w.push_back(static_cast<double>(v));
      double sp_h = 0.0, sp_max = 0.0, sp_eff = 0.0;
      shape_stats(visit_w, &sp_h, &sp_max, &sp_eff);
      PutDouble(&row, "search_policy_norm_entropy", sp_h);
      PutDouble(&row, "search_policy_max_action_prob", sp_max);
      PutDouble(&row, "search_policy_eff_action_count", sp_eff);
      PutDouble(&row, "sims_per_sec",
                last_res.elapsed_time_ms > 0.0
                    ? (last_res.simulations_completed * 1000.0 / last_res.elapsed_time_ms)
                    : 0.0);
      row["budget_limit_reason"] = diag.budget_limit_reason;
      const bool hit_session_limit =
          !diag.budget_limit_reason.empty() && diag.budget_limit_reason != "none";
      row["budget_starved"] = hit_session_limit ||
                              last_res.fallback_reason == "short_window_budget_exceeded" ||
                              last_res.fallback_reason == "live_deadline_reached";

      // --- The reference's DEAD fields, recorded verbatim so their deadness is
      // --- evidence rather than an assertion, and so section 7.1 equality can be
      // --- checked on every reference field instead of excluding a class.
      row["decision_role_diag_reference"] = diag.decision_role;
      row["is_strategic_diag_reference"] = (diag.decision_role == "2" || diag.decision_role == "3");
      row["round_diag_reference"] = static_cast<int64_t>(diag.round);
      row["phase_diag_reference"] = diag.phase;
      row["searched_seat_diag_reference"] = static_cast<int64_t>(diag.searched_seat);
      row["session_id_diag_reference"] = diag.session_id;
      row["budget_mode_diag_reference"] = diag.budget_mode;
      row["hard_sim_limit_diag_reference"] = static_cast<int64_t>(diag.hard_sim_limit);
      row["soft_sim_limit_diag_reference"] = static_cast<int64_t>(diag.soft_sim_limit);
      row["tree_node_count_diag_reference"] = static_cast<int64_t>(diag.tree_node_count);
      row["newly_completed_simulations_diag_reference"] = static_cast<int64_t>(diag.newly_completed_simulations);
      row["session_cumulative_simulations_diag_reference"] = static_cast<int64_t>(diag.session_cumulative_simulations);
      row["short_window_cumulative_simulations_diag_reference"] = static_cast<int64_t>(diag.short_window_cumulative_simulations);
      // root_coverage is emitted ONLY here. Registration section 3.3.3: it must
      // not be consumed. Coverage is recomputed from num_covered_actions /
      // covered_prior_mass / n_legal on label rows.
      PutDouble(&row, "root_coverage_diag_reference", diag.root_coverage);
      PutDouble(&row, "hard_time_limit_ms_diag_reference", diag.hard_time_limit_ms);
      PutDouble(&row, "soft_time_limit_ms_diag_reference", diag.soft_time_limit_ms);
      PutDouble(&row, "elapsed_search_time_ms_diag_reference", diag.elapsed_search_time_ms);
      PutDouble(&row, "observation_wait_time_ms_diag_reference", diag.observation_wait_time_ms);
      PutDouble(&row, "session_cumulative_search_time_ms_diag_reference", diag.session_cumulative_search_time_ms);
      PutDouble(&row, "long_agent_session_cumulative_time_ms_diag_reference", diag.long_agent_session_cumulative_time_ms);

      // Per-row provenance.
      row["binary_sha256"] = self_sha;
      row["model_sha256"] = model_sha;
      row["opponent_model_sha256"] = opp_sha;
      row["engine_tree_hash"] = absl::GetFlag(FLAGS_engine_tree_hash);
      row["registration_sha256"] = absl::GetFlag(FLAGS_registration_sha256);
      row["amendment_sha256"] = absl::GetFlag(FLAGS_amendment_sha256);
      row["configured_max_simulations"] = static_cast<int64_t>(max_sims);
      row["configured_max_nodes"] = static_cast<int64_t>(absl::GetFlag(FLAGS_max_nodes));
      PutDouble(&row, "configured_covered_prior_threshold", absl::GetFlag(FLAGS_covered_prior_threshold));
      row["configured_min_visit_threshold"] = static_cast<int64_t>(absl::GetFlag(FLAGS_min_visit_threshold));

      game_role_counts[role_name]++;
      pending_audit.push_back(row);
      ++n_audit_rows;

      // ------------------------------------------------------------------
      // The LABEL row. Only a successfully searched decision reaches here.
      // A non-eligible decision is NOT a weight-zero label: it has no target,
      // no coverage reading and no stability reading (work order section 1).
      // ------------------------------------------------------------------
      if (search_expected && last_res.simulations_completed == max_sims &&
          diag.inherited_root_visits == 0) {
        json::Object lab = row;

        // --- Raw prior. The `raw_priors.empty() ? priors : raw_priors` idiom is
        // --- the one every consumer in this codebase uses
        // --- (dune_puct_is_mcts.cc:1295-1297): raw_priors is empty only when the
        // --- root was absent from the tree, and on that path `priors` was built
        // --- straight from the evaluator and IS raw.
        const bool from_raw_priors = (diag.raw_priors.size() == legal_actions.size());
        const std::vector<double>& raw_prior_vector = from_raw_priors ? diag.raw_priors : diag.priors;
        lab["raw_prior_source"] = std::string(from_raw_priors ? "raw_priors" : "priors");
        PutDoubleVec(&lab, "raw_prior_vector", raw_prior_vector);
        json::Array hex_arr;
        for (double x : raw_prior_vector) hex_arr.push_back(FmtHex(x));  // JSON STRINGS
        lab["raw_prior_vector_hex"] = hex_arr;
        lab["raw_argmax_action"] = static_cast<int64_t>(ArgmaxLowestId(legal_actions, raw_prior_vector));

        // --- The policy target: the normalized UNSHARPENED visit distribution,
        // --- EXPONENT 1.0 (plan section 8.3.3 item 3, the authority for the target
        // --- definition). NEVER derived from deployed_policy, which is one-hot at
        // --- --temperature=0.0.
        double total_visits = 0.0;
        for (int v : diag.visit_counts) total_visits += static_cast<double>(v);
        std::vector<double> policy_target(diag.visit_counts.size(), 0.0);
        if (total_visits > 0.0) {
          for (size_t i = 0; i < diag.visit_counts.size(); ++i) {
            policy_target[i] = static_cast<double>(diag.visit_counts[i]) / total_visits;
          }
        }
        PutDoubleVec(&lab, "policy_target", policy_target);
        lab["total_root_visits"] = static_cast<int64_t>(diag.total_root_visits);
        PutDouble(&lab, "visit_counts_sum", total_visits);
        lab["policy_target_exponent"] = static_cast<int64_t>(1);

        // --- Actions. pre_gate_visit_argmax_action mirrors
        // --- GetRootArgmaxAction (:715-729, :1021) -- the FINAL visit argmax,
        // --- which is what stability_agreement compares against (ruling R4), NOT
        // --- deployed_action.
        std::vector<double> visit_w2;
        visit_w2.reserve(diag.visit_counts.size());
        for (int v : diag.visit_counts) visit_w2.push_back(static_cast<double>(v));
        const Action pre_gate_visit_argmax = ArgmaxLowestId(legal_actions, visit_w2);
        lab["pre_gate_visit_argmax_action"] = static_cast<int64_t>(pre_gate_visit_argmax);
        lab["deployed_action"] = static_cast<int64_t>(chosen_action);
        json::Array dep_actions;
        std::vector<double> dep_probs;
        for (const auto& ap : last_res.policy) {
          dep_actions.push_back(static_cast<int64_t>(ap.first));
          dep_probs.push_back(ap.second);
        }
        lab["deployed_policy_actions"] = dep_actions;
        PutDoubleVec(&lab, "deployed_policy", dep_probs);

        // --- Coverage. Both instruments recorded on the row: the C++ gate's own
        // --- decision, and the generator-side recomputation from the emitted
        // --- inputs. The analyzer recomputes a THIRD time from the same inputs;
        // --- registration section 6.2 requires the C++ flag and the analyzer to
        // --- agree on every label row.
        const int num_covered_actions = diag.num_covered_actions;
        const double covered_prior_mass = diag.covered_prior_mass;
        lab["num_covered_actions"] = static_cast<int64_t>(num_covered_actions);
        PutDouble(&lab, "covered_prior_mass", covered_prior_mass);
        const bool coverage_gate_fired = (last_res.fallback_reason == "low_coverage");
        lab["coverage_gate_fired"] = coverage_gate_fired;
        lab["coverage_fallback"] = coverage_gate_fired;
        lab["coverage_gate_recomputed_cpp"] =
            pwo3::CoverageGateFires(num_covered_actions, covered_prior_mass, n_legal);
        lab["coverage_gate_changed_action"] = (chosen_action != pre_gate_visit_argmax);
        lab["zero_visits_fallback"] = (last_res.fallback_reason == "zero_visits");
        // On a completed 200-simulation search the only reachable reasons are
        // these three. Anything else means a branch this schema does not model.
        if (last_res.fallback_reason != "none" && !coverage_gate_fired &&
            last_res.fallback_reason != "zero_visits") {
          std::cerr << "STOP: unexpected fallback_reason '" << last_res.fallback_reason
                    << "' on a completed 200-simulation row at game " << g
                    << " decision " << decision_index << ".\n";
          return 1;
        }

        // --- Half/full stability: the IN-SEARCH prefix checkpoint at sim 100 of
        // --- 200, same seed, same tree, zero extra compute (registration section 5.3).
        lab["half_budget_checkpoint_reached"] = diag.stability_checkpoint_reached;
        lab["half_budget_checkpoint_sim"] = static_cast<int64_t>(stability_checkpoint_sim);
        lab["half_budget_visit_argmax_action"] = static_cast<int64_t>(diag.stability_checkpoint_action);
        lab["half_budget_num_covered_actions"] = static_cast<int64_t>(diag.stability_checkpoint_num_covered_actions);
        PutDouble(&lab, "half_budget_covered_prior_mass", diag.stability_checkpoint_covered_prior_mass);
        lab["stability_agreement"] = diag.stability_agreement;

        // --- The section 6.1 weight rule. Deterministic and binary; no fitted
        // --- lambda of any kind and no post-generation threshold tuning. The
        // --- analyzer recomputes it independently.
        const bool coverage_passes = !pwo3::CoverageGateFires(num_covered_actions, covered_prior_mass, n_legal);
        const int weight = (last_res.simulations_completed == max_sims &&
                            diag.inherited_root_visits == 0 && coverage_passes &&
                            diag.stability_agreement)
                               ? 1 : 0;
        lab["weight"] = static_cast<int64_t>(weight);

        // --- Conversion (Amendment section 7.3 + registration section 8.3).
        // --- Amount is action_id - 740 over the REAL range 741-752; 740 is an
        // --- unused sentinel, never legal, never counted, never shaped.
        const std::vector<Action> legal_conversions = pwo3::LegalConversionActions(legal_actions);
        const bool conversion_legal = !legal_conversions.empty();
        lab["conversion_legal"] = conversion_legal;
        lab["legal_conversion_actions"] = ToJsonArray(legal_conversions);
        lab["n_legal_conversion_actions"] = static_cast<int64_t>(legal_conversions.size());
        json::Array conv_amounts;
        for (Action a : legal_conversions) conv_amounts.push_back(static_cast<int64_t>(pwo3::ConversionAmount(a)));
        lab["legal_conversion_amounts"] = conv_amounts;
        lab["chosen_conversion_amount"] = static_cast<int64_t>(pwo3::ConversionAmount(chosen_action));
        lab["pre_gate_conversion_amount"] = static_cast<int64_t>(pwo3::ConversionAmount(pre_gate_visit_argmax));
        lab["raw_argmax_conversion_amount"] = static_cast<int64_t>(
            pwo3::ConversionAmount(ArgmaxLowestId(legal_actions, raw_prior_vector)));
        // The section 8 RISK quantities. PWO-3 measured DEPLOYED fixed_800 actions
        // and never measured conversion VISIT MASS; a soft target can carry
        // conversion probability where the argmax never converts, so this is a risk
        // to be MEASURED, not an expected fact.
        std::vector<double> conv_visit_probs;
        double total_conversion_visit_mass = 0.0;
        for (Action a : legal_conversions) {
          double p = 0.0;
          for (size_t i = 0; i < legal_actions.size(); ++i) {
            if (legal_actions[i] == a) { p = policy_target[i]; break; }
          }
          conv_visit_probs.push_back(p);
          total_conversion_visit_mass += p;
        }
        PutDoubleVec(&lab, "conversion_visit_prob_by_action", conv_visit_probs);
        PutDouble(&lab, "total_conversion_visit_mass", total_conversion_visit_mass);
        lab["deployed_conversion_action"] =
            static_cast<int64_t>(pwo3::IsConversion(chosen_action) ? chosen_action : kInvalidAction);

        pending_labels.push_back(lab);
        ++n_label_rows;
        ++game_label_rows;
      }

      if (chosen_action >= dune_imperium::kActionConvertSpecimenToTroop0 &&
          chosen_action <= dune_imperium::kActionConvertSpecimenToTroop0 + 12 &&
          current_player >= 0 && current_player < 4) {
        ++specimen_conversions[current_player];
      }
      state->ApplyAction(chosen_action);
      full_history.push_back(chosen_action);
      ++decision_index;
    }

    // --- Terminal outcome. TRAJECTORY-LEVEL REPORTING ONLY -- never a causal
    // --- label for an individual move (registration section 3.1, ruling O3).
    const std::vector<double> returns = state->Returns();
    const auto* final_state = dynamic_cast<const dune_imperium::DuneImperiumState*>(state.get());
    int winner = -1;
    double max_return = -std::numeric_limits<double>::infinity();
    for (int p = 0; p < 4; ++p) {
      if (returns[p] > max_return) { max_return = returns[p]; winner = p; }
    }
    int searched_seat_rank = 0;
    for (int p = 0; p < 4; ++p) if (returns[p] > returns[search_seat]) ++searched_seat_rank;

    json::Object gobj;
    gobj["game_index"] = static_cast<int64_t>(g);
    gobj["game_seed"] = static_cast<int64_t>(game_seed);
    gobj["searched_seat"] = static_cast<int64_t>(search_seat);
    gobj["search_config_seed"] = static_cast<int64_t>(search_config_seed);
    json::Array seat_seed_arr;
    for (uint64_t s : seat_seeds) seat_seed_arr.push_back(static_cast<int64_t>(s));
    gobj["seat_seeds"] = seat_seed_arr;
    gobj["winner"] = static_cast<int64_t>(winner);
    gobj["searched_seat_rank"] = static_cast<int64_t>(searched_seat_rank);
    PutDoubleVec(&gobj, "returns", returns);
    json::Array vp_arr;
    for (int p = 0; p < 4; ++p) {
      vp_arr.push_back(static_cast<int64_t>(final_state ? GetTrueFinalVp(final_state, p) : -1));
    }
    gobj["final_vp"] = vp_arr;
    gobj["rounds_played"] = static_cast<int64_t>(final_state ? final_state->GetCurrentRound() - 1 : -1);
    json::Array spec_arr;
    for (int p = 0; p < 4; ++p) spec_arr.push_back(static_cast<int64_t>(specimen_conversions[p]));
    gobj["specimen_conversions"] = spec_arr;
    gobj["n_decisions"] = static_cast<int64_t>(decision_index);
    gobj["n_audit_rows"] = static_cast<int64_t>(pending_audit.size());
    gobj["n_label_rows"] = static_cast<int64_t>(game_label_rows);
    gobj["search_count_final"] = static_cast<int64_t>(search_count);
    json::Object rc;
    for (const auto& kv : game_role_counts) rc[kv.first] = static_cast<int64_t>(kv.second);
    gobj["role_counts"] = rc;
    // The full action history, chance outcomes included, in application order.
    // This is what dune_pwo4_role_verify replays to reconstruct each decision
    // state INDEPENDENTLY of the emitter (work order CP0.6 guard 3). Without it
    // the guard degrades to calling the classifier twice on the same live state,
    // which proves only that the function is deterministic.
    gobj["history"] = ToJsonArray(full_history);
    gobj["history_length"] = static_cast<int64_t>(full_history.size());

    // The mirror invariant: Step() was called exactly once per searched-seat
    // decision, so the counter advanced exactly that many times.
    if (search_count != static_cast<int>(pending_audit.size())) {
      std::cerr << "STOP: search_count mirror (" << search_count << ") != audit rows ("
                << pending_audit.size() << ") in game " << g << ".\n";
      return 1;
    }

    // The game completed without voiding, so its rows are admissible. Writing is
    // deferred to here on purpose: a game voided mid-play (registration section 2.3)
    // must contribute NO rows, and buffering is what makes that structural rather
    // than a cleanup step that could be forgotten.
    for (const auto& r : pending_audit) audit_out << json::ToString(r, false) << "\n";
    for (const auto& r : pending_labels) labels_out << json::ToString(r, false) << "\n";
    games_out << json::ToString(gobj, false) << "\n";
    audit_out.flush();
    labels_out.flush();
    games_out.flush();

    std::cerr << "  game " << g << " seat " << search_seat << ": "
              << pending_audit.size() << " audit, " << game_label_rows << " labels\n";
  }

  const double elapsed_s = std::chrono::duration<double>(
      std::chrono::steady_clock::now() - run_start).count();

  // =========================================================================
  // The manifest. Section 7.3 requires the provenance contract to be MANDATORY,
  // not optional: it declares the expected row counts for BOTH record classes and
  // every input digest, and the analyzer hard-fails on its absence. An optional
  // contract is worse than none.
  // =========================================================================
  {
    json::Object m;
    m["tool"] = std::string("dune_pwo4_trajectory");
    m["registration"] = std::string("docs/PWO4_TRAJECTORY_REGISTRATION.md revision 3");
    m["registration_sha256"] = absl::GetFlag(FLAGS_registration_sha256);
    m["amendment"] = std::string("docs/PWO4_AMENDMENT_1_GATE_SEMANTICS.md");
    m["amendment_sha256"] = absl::GetFlag(FLAGS_amendment_sha256);
    m["generator_binary_sha256"] = self_sha;
    m["model_path"] = model_ckpt;
    m["model_sha256"] = model_sha;
    m["opponent_model_path"] = opp_ckpt;
    m["opponent_model_sha256"] = opp_sha;
    m["engine_tree_hash"] = absl::GetFlag(FLAGS_engine_tree_hash);

    // --- The provenance contract (section 7.3). MANDATORY.
    json::Object contract;
    contract["expected_audit_rows"] = static_cast<int64_t>(n_audit_rows);
    contract["expected_label_rows"] = static_cast<int64_t>(n_label_rows);
    contract["expected_games"] = static_cast<int64_t>(total_games);
    contract["audit_path"] = audit_path;
    contract["labels_path"] = labels_path;
    contract["games_path"] = games_path;
    m["provenance_contract"] = contract;

    // --- Inference configuration of record (registration section 2.5). The device
    // --- is IMPLICIT in the source (there is no device flag), so it is RECORDED
    // --- here -- the exact omission Amendment 2 section 6 had to disclose after
    // --- the fact.
    json::Object ac;
    ac["enabled"] = device.is_cuda();
    ac["device"] = std::string(device.is_cuda() ? "cuda" : "cpu");
    // BFloat16, NOT FP16. dune_network.h:158 sets
    // at::autocast::set_autocast_dtype(at::kCUDA, at::ScalarType::BFloat16).
    // A committed PWO-3 artifact carries a descriptive note reading "FP16 autocast
    // on CUDA"; that wording is wrong and is deliberately not copied here
    // (work order section 12.1). BF16 has 8 mantissa bits, which is why section 7.2's
    // determinism claim is bounded to this box and these library versions.
    ac["dtype"] = std::string(device.is_cuda() ? "bfloat16" : "n/a");
    ac["source"] = std::string(
        "dune_evaluator.h:48,:88,:158 open AutocastGuard(device_.type(), "
        "device_.is_cuda()); dune_network.h:158-159 sets the CUDA autocast dtype "
        "to at::ScalarType::BFloat16 and enables it.");
    m["autocast"] = ac;

    std::string gpu_name = "n/a";
    std::string compute_capability = "n/a";
    if (device.is_cuda()) {
      const auto* props = at::cuda::getDeviceProperties(0);
      gpu_name = props->name;
      compute_capability = absl::StrFormat("%d.%d", props->major, props->minor);
    }
    json::Object dev;
    dev["device"] = std::string(device.is_cuda() ? "CUDA" : "CPU");
    dev["gpu_name"] = gpu_name;
    dev["gpu_compute_capability"] = compute_capability;
    dev["driver_version"] = DriverVersionLine();
    dev["libtorch_version"] = std::string(TORCH_VERSION);
    dev["cuda_runtime_version"] = static_cast<int64_t>(at::detail::getCUDAHooks().versionCUDART());
    dev["cudnn_version"] = static_cast<int64_t>(at::detail::getCUDAHooks().versionCuDNN());
    dev["torch_cuda_config"] = at::detail::getCUDAHooks().showConfig();
    dev["at_num_threads"] = static_cast<int64_t>(at::get_num_threads());
    m["inference_config"] = dev;

    // --- The full controller pin, echoed. Every flag, including those equal to
    // --- the current default: defaults drift, registrations do not.
    json::Object pin;
    pin["seed"] = static_cast<int64_t>(base_seed);
    pin["start_episode_id"] = static_cast<int64_t>(start_episode_id);
    pin["games"] = static_cast<int64_t>(total_games);
    pin["threads"] = static_cast<int64_t>(num_threads);
    pin["max_simulations"] = static_cast<int64_t>(max_sims);
    pin["max_nodes"] = static_cast<int64_t>(absl::GetFlag(FLAGS_max_nodes));
    pin["disable_time_limit"] = absl::GetFlag(FLAGS_disable_time_limit);
    // The frozen MODES, echoed from the flags that were fatally asserted above --
    // never as literals. A manifest that hardcodes what it claims to observe
    // records the author's intention, not the run's configuration.
    pin["use_session"] = absl::GetFlag(FLAGS_use_session);
    pin["controller_mode"] = absl::GetFlag(FLAGS_controller_mode);
    pin["rotate_seat"] = absl::GetFlag(FLAGS_rotate_seat);
    pin["fresh_search_roles"] = absl::GetFlag(FLAGS_fresh_search_roles);
    pin["policy_only"] = absl::GetFlag(FLAGS_policy_only);
    pin["live_deadline"] = absl::GetFlag(FLAGS_live_deadline);
    pin["force_swordmaster_rounds"] = static_cast<int64_t>(absl::GetFlag(FLAGS_force_swordmaster_rounds));
    pin["grant_swordmaster_round"] = static_cast<int64_t>(absl::GetFlag(FLAGS_grant_swordmaster_round));
    pin["check_strategic_state"] = absl::GetFlag(FLAGS_check_strategic_state);
    pin["use_opponent_model"] = absl::GetFlag(FLAGS_use_opponent_model);
    pin["nonlinear_value_head"] = absl::GetFlag(FLAGS_nonlinear_value_head);
    pin["opponent_nonlinear_value_head"] = absl::GetFlag(FLAGS_opponent_nonlinear_value_head);
    pin["verbose_diagnostics"] = absl::GetFlag(FLAGS_verbose_diagnostics);
    pin["root_noise_fpu_zero"] = absl::GetFlag(FLAGS_root_noise_fpu_zero);
    pin["conservative_override_enabled"] = absl::GetFlag(FLAGS_conservative_override_enabled);
    pin["conservative_continuation_overrides_disabled"] = absl::GetFlag(FLAGS_conservative_continuation_overrides_disabled);
    pin["hidden_dim"] = static_cast<int64_t>(absl::GetFlag(FLAGS_hidden_dim));
    pin["num_blocks"] = static_cast<int64_t>(absl::GetFlag(FLAGS_num_blocks));
    pin["opp_hidden_dim"] = static_cast<int64_t>(absl::GetFlag(FLAGS_opp_hidden_dim));
    pin["opp_num_blocks"] = static_cast<int64_t>(absl::GetFlag(FLAGS_opp_num_blocks));
    pin["max_world_samples"] = static_cast<int64_t>(absl::GetFlag(FLAGS_max_world_samples));
    pin["fixed_continuation_reserve"] = static_cast<int64_t>(absl::GetFlag(FLAGS_fixed_continuation_reserve));
    pin["purchase_combat_budget"] = static_cast<int64_t>(absl::GetFlag(FLAGS_purchase_combat_budget));
    pin["min_visit_threshold"] = static_cast<int64_t>(absl::GetFlag(FLAGS_min_visit_threshold));
    pin["conservative_meaningful_visit_threshold"] = static_cast<int64_t>(absl::GetFlag(FLAGS_conservative_meaningful_visit_threshold));
    pin["max_search_decision_depth"] = static_cast<int64_t>(absl::GetFlag(FLAGS_max_search_decision_depth));
    pin["final_policy_type"] = std::string("kNormalizedVisitCount");
    pin["use_observation_string"] = true;
    pin["stability_checkpoint_sim"] = static_cast<int64_t>(stability_checkpoint_sim);
    // Doubles in the pin go through the same round-trip discipline as measurements.
    PutDouble(&pin, "puct_c", absl::GetFlag(FLAGS_puct_c));
    PutDouble(&pin, "temperature", absl::GetFlag(FLAGS_temperature));
    PutDouble(&pin, "simulated_opponent_temperature", absl::GetFlag(FLAGS_simulated_opponent_temperature));
    PutDouble(&pin, "external_opponent_temperature", absl::GetFlag(FLAGS_external_opponent_temperature));
    PutDouble(&pin, "root_prior_temperature", absl::GetFlag(FLAGS_root_prior_temperature));
    PutDouble(&pin, "training_root_prior_temperature", absl::GetFlag(FLAGS_training_root_prior_temperature));
    PutDouble(&pin, "utility_divisor", absl::GetFlag(FLAGS_utility_divisor));
    PutDouble(&pin, "dirichlet_epsilon", absl::GetFlag(FLAGS_dirichlet_epsilon));
    PutDouble(&pin, "dirichlet_alpha", absl::GetFlag(FLAGS_dirichlet_alpha));
    PutDouble(&pin, "dirichlet_alpha_total", absl::GetFlag(FLAGS_dirichlet_alpha_total));
    PutDouble(&pin, "forced_playouts_k", absl::GetFlag(FLAGS_forced_playouts_k));
    PutDouble(&pin, "covered_prior_threshold", absl::GetFlag(FLAGS_covered_prior_threshold));
    PutDouble(&pin, "candidate_logit_cap", absl::GetFlag(FLAGS_candidate_logit_cap));
    PutDouble(&pin, "opponent_logit_cap", absl::GetFlag(FLAGS_opponent_logit_cap));
    PutDouble(&pin, "relative_time_budget_ms", absl::GetFlag(FLAGS_relative_time_budget_ms));
    PutDouble(&pin, "live_continuation_reserve_seconds", absl::GetFlag(FLAGS_live_continuation_reserve_seconds));
    PutDouble(&pin, "conservative_stability_checkpoint_fraction", absl::GetFlag(FLAGS_conservative_stability_checkpoint_fraction));
    PutDouble(&pin, "conservative_covered_prior_threshold", absl::GetFlag(FLAGS_conservative_covered_prior_threshold));
    PutDouble(&pin, "conservative_q_margin_threshold", absl::GetFlag(FLAGS_conservative_q_margin_threshold));
    m["controller_pin"] = pin;

    json::Object acct;
    acct["audit_rows"] = static_cast<int64_t>(n_audit_rows);
    acct["label_rows"] = static_cast<int64_t>(n_label_rows);
    acct["search_expected_true"] = static_cast<int64_t>(n_search_expected_true);
    acct["search_expected_false"] = static_cast<int64_t>(n_search_expected_false);
    acct["search_expected_false_single_action"] = static_cast<int64_t>(n_single_action);
    acct["search_expected_false_non_strategic"] = static_cast<int64_t>(n_non_strategic);
    PutDouble(&acct, "wall_clock_seconds", elapsed_s);
    m["accounting"] = acct;

    m["serialization"] = std::string(
        "Every registered double is written as STRINGS: <name>_exact (%.17g, "
        "max_digits10) and <name>_legacy6 (json::ToString(json::Value(v)), the "
        "actual json.cc formatter, CALLED not reimplemented). raw_prior_vector "
        "additionally carries _hex (%a) as JSON strings. No registered measurement "
        "is routed through json.cc's default double formatting, whose six-decimal "
        "%f floors anything below 5e-7 to exactly 0.0.");
    m["search_count_note"] = std::string(
        "search_count_before/after MIRROR DunePUCTISMCTSBot::search_count_, which is "
        "private. Step() increments it once per call, unconditionally, AFTER "
        "RunSearch returns (dune_puct_is_mcts.cc:1126); this generator calls Step() "
        "exactly once per searched-seat decision, so the mirror is exact by "
        "construction and is asserted per game against the audit row count. It is "
        "observable: final_action_sampling_seed derives from it and genuinely "
        "selects the action on coverage-fallback rows, where RunSearch returns the "
        "raw prior rather than a one-hot policy.");

    std::ofstream mf(manifest_path);
    if (!mf) {
      std::cerr << "STOP: cannot open --manifest_json_path.\n";
      return 1;
    }
    mf << json::ToString(m) << "\n";
  }

  std::cerr << "\ndone: " << total_games << " games, " << n_audit_rows
            << " audit rows, " << n_label_rows << " label rows, "
            << absl::StrFormat("%.1f", elapsed_s) << " s\n";
  return 0;
}
