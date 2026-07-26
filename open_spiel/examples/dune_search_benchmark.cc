#include <iostream>
#include <memory>
#include <vector>
#include <string>
#include <algorithm>
#include <filesystem>
#include <random>
#include <thread>
#include <atomic>
#include <chrono>
#include <sstream>
#include <cctype>
#include <functional>
#include <mutex>
#include <map>
#include <set>

#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"
#include "open_spiel/games/dune_imperium/dune_imperium.h"
#include "open_spiel/abseil-cpp/absl/strings/str_format.h"
#include "open_spiel/abseil-cpp/absl/flags/flag.h"
#include "open_spiel/abseil-cpp/absl/flags/parse.h"
#include "open_spiel/abseil-cpp/absl/random/distributions.h"
#include "open_spiel/utils/json.h"
#include <torch/torch.h>

#include "dune_network.h"
#include "dune_evaluator.h"
#include "dune_puct_is_mcts.h"
#include <array>
#include "dune_search_session.h"
#include "dune_seed_utils.h"
#include "dune_sha256.h"
#include "dune_warmstart_helpers.h"  // ChooseHeuristicAcquisitionAction (swordmaster probe)
#include "open_spiel/games/dune_imperium/dune_imperium_cards.h"
#include <fstream>

ABSL_FLAG(std::string, model_checkpoint, "", "Path to load search agent model checkpoint.");
ABSL_FLAG(std::string, opponent_checkpoint, "", "Path to load opponent model checkpoint. If empty, the search agent model is reused. If 'random', random opponents are used.");
ABSL_FLAG(int, seed, 42, "Seed for deterministic RNG.");
ABSL_FLAG(int, games, 1000, "How many games to play in total.");
ABSL_FLAG(int, start_episode_id, 0, "Episode ID to start the range of games (for resumability).");
ABSL_FLAG(std::string, game_jsonl_path, "", "Path to write structured game-level JSONL outcomes.");
ABSL_FLAG(std::string, search_jsonl_path, "", "Path to write structured search-level JSONL diagnostics.");
ABSL_FLAG(std::string, aggregate_json_path, "", "Path to write structured aggregate JSON outcome.");
ABSL_FLAG(int, threads, 8, "How many threads to run.");
ABSL_FLAG(int, hidden_dim, 2048, "Search agent model hidden dimension.");
ABSL_FLAG(int, num_blocks, 8, "Search agent model residual blocks count.");
ABSL_FLAG(int, opp_hidden_dim, -1, "Opponent model hidden dimension. If -1, inherits hidden_dim.");
ABSL_FLAG(int, opp_num_blocks, -1, "Opponent model block count. If -1, inherits num_blocks.");
ABSL_FLAG(int, max_simulations, 50, "MCTS simulation budget per move.");
ABSL_FLAG(double, puct_c, 0.3, "PUCT exploration constant.");
ABSL_FLAG(int, max_world_samples, -1, "Number of cached world samples for determinization (-1 = fresh resampling).");
ABSL_FLAG(double, utility_divisor, 4.0, "Terminal utility divisor (couples with search value normalization).");
ABSL_FLAG(double, temperature, 0.0, "Softmax temperature for final move choice (0.0 = greedy).");
ABSL_FLAG(double, simulated_opponent_temperature, 1.0, "Softmax temperature for simulated opponent action selection inside search (default 1.0).");
ABSL_FLAG(double, external_opponent_temperature, 0.0, "Softmax temperature for external benchmark opponent action selection (default 0.0).");
ABSL_FLAG(double, dirichlet_epsilon, 0.0, "Dirichlet noise weight at root.");
ABSL_FLAG(double, dirichlet_alpha, 0.3, "Dirichlet noise alpha.");
ABSL_FLAG(bool, rotate_seat, true, "Rotate the seat of the search agent across games.");
ABSL_FLAG(bool, use_opponent_model, false, "Whether non-search players in simulation follow the PPO prior policy instead of PUCT.");
ABSL_FLAG(bool, nonlinear_value_head, false,
          "Use the versioned nonlinear value head for the search model.");
ABSL_FLAG(bool, opponent_nonlinear_value_head, false,
          "Use the versioned nonlinear value head for the external opponent model.");
ABSL_FLAG(bool, verbose_diagnostics, true, "Print IS-MCTS node-reuse and depth diagnostics periodically.");
ABSL_FLAG(bool, check_strategic_state, false, "Whether to bypass MCTS search at non-strategic states.");
ABSL_FLAG(bool, disable_time_limit, false, "Disable the time limit per move for fixed-simulation evaluation.");
ABSL_FLAG(double, root_prior_temperature, 1.0, "Root prior temperature.");
ABSL_FLAG(int, fixed_continuation_reserve, 0, "Continuation reserve simulations.");
ABSL_FLAG(int, purchase_combat_budget, 16, "Purchase/combat short-window simulation budget.");
ABSL_FLAG(double, live_continuation_reserve_seconds, 10.0, "Live continuation reserve seconds.");
ABSL_FLAG(bool, live_deadline, false, "Use live deadline budget mode instead of fixed simulations.");
ABSL_FLAG(bool, use_session, true, "If true (default), drive the search seat through the persistent DuneSearchSession. If false, reproduce the July-14 reference protocol: a fresh full search per decision via DunePUCTISMCTSBot::Step() (bot->Step(), including its sample-on-fallback selection), with no shared session budget.");
ABSL_FLAG(bool, policy_only, false, "Policy-only control arm: run the session in kPolicyOnly budget mode so the candidate plays the raw network policy on every decision (no search). Pair with --purchase_combat_budget=0 and --temperature=0 for a pure greedy-policy baseline vs the greedy opponents.");
ABSL_FLAG(std::string, fresh_search_roles, "",
          "Decomposition arm 4 (Path B only, requires --use_session=false): "
          "comma-separated decision roles that receive a fresh full search; every "
          "OTHER searched-seat decision plays the raw-prior argmax (identical to the "
          "policy-only arm's selection). Empty (default) = search every decision "
          "(unchanged full-search behavior). Recognized tokens: primary, "
          "continuation, purchase, combat, other, leader, forced.");
ABSL_FLAG(int, force_swordmaster_rounds, 0,
          "Swordmaster probe (arm B): if > 0, the searched seat is forced toward "
          "legal Swordmaster acquisition (Task-10 acquisition heuristic) on every "
          "decision while GetCurrentRound() <= this value AND it does not yet own "
          "Swordmaster; afterwards it plays normally. 0 (default) = arm A (no "
          "forcing). Uses a separate rng so paired seeds stay aligned with arm A.");
ABSL_FLAG(int, grant_swordmaster_round, 0,
          "Swordmaster endowment probe (arm B-endow): if > 0, at the searched "
          "seat's FIRST decision of this round (before it places any agent) the "
          "seat is GRANTED Swordmaster free via SetSwordmasterForTesting -- no "
          "solari cost, no forcing of any action. Persistence across later rounds "
          "is handled by the engine (permanent third agent, commit c8b3bf6); the "
          "probe adds NO persistence logic. 0 (default) = off (= arm A).");
ABSL_FLAG(double, relative_time_budget_ms, 52000.0, "Time budget per move (ms).");
ABSL_FLAG(bool, conservative_override_enabled, false, "Enforce conservative override selection protocol");
ABSL_FLAG(double, conservative_covered_prior_threshold, 0.95, "Minimum covered prior mass to avoid fallback");
ABSL_FLAG(int, conservative_meaningful_visit_threshold, 10, "Minimum visits required for raw and argmax actions");
ABSL_FLAG(double, conservative_q_margin_threshold, 0.03, "Q margin threshold for MCTS to override raw");
ABSL_FLAG(double, conservative_stability_checkpoint_fraction, 0.5, "Fraction of budget at which to check stability");
ABSL_FLAG(bool, conservative_continuation_overrides_disabled, true, "Disable overrides during continuation decisions");

// --- WO-1 Phase 2: seat controller topology -------------------------------
// Historically this binary supported exactly ONE searched seat (search_seat =
// rotate_seat ? g%4 : 0); every other seat was driven by --opponent_checkpoint.
// The pace program needs the homogeneous four-copy configuration (the PRIMARY
// endpoint's controller) and an explicitly-labelled mixed slice, so the seat
// topology is now a first-class flag. 'single' reproduces the legacy behavior
// byte-for-byte and is the default.
ABSL_FLAG(std::string, controller_mode, "single",
          "Seat controller topology. 'single' (default, legacy): exactly one "
          "searched seat, the other three driven by --opponent_checkpoint. "
          "'homogeneous': all four seats run the search controller (the "
          "four-copy production-controller configuration). 'mixed': the "
          "rotating candidate seat is searched, the other three run "
          "--mixed_opponent_controller.");
ABSL_FLAG(std::string, mixed_opponent_controller, "random_legal",
          "Controller for the three non-candidate seats when "
          "--controller_mode=mixed. 'random_legal': uniform-random legal "
          "action (slice S1). 'raw_policy': raw network prior from "
          "--opponent_checkpoint (or the candidate checkpoint when unset), "
          "selected at --external_opponent_temperature.");
// Formerly the hard-coded 10.0f passed to every DuneNNEvaluator here. Split
// per-model and named to match dune_population_eval's flags of the same name.
// Default 10.0 preserves prior behavior exactly.
ABSL_FLAG(double, candidate_logit_cap, 10.0,
          "Logit cap for the CANDIDATE / searched-seat evaluator. Default 10.0 "
          "preserves prior behavior (formerly hard-coded).");
ABSL_FLAG(double, opponent_logit_cap, 10.0,
          "Logit cap for OPPONENT-model evaluators. Default 10.0 preserves "
          "prior behavior (formerly hard-coded).");

namespace open_spiel {
namespace {

int GetTrueFinalVp(const dune_imperium::DuneImperiumState* dune_state, int p) {
  int base_vp = dune_state->GetPlayerVpForTesting(p);
  int endgame_vp = 0;

  // 1. Endgame Intrigues
  const auto& hand = dune_state->GetIntrigueHandForTesting(p);
  for (int intrigue_id : hand) {
    const auto* intrigue = dune_imperium::FindIntrigueCardById(intrigue_id);
    if (intrigue && (intrigue->phase_mask & dune_imperium::kIntriguePhaseEndgameMask) != 0) {
      endgame_vp += dune_state->EndgameIntrigueVpBonusForTesting(p, intrigue_id);
    }
  }

  // 2. Tech tile 6: Holtzman Engine
  const auto& tech_tiles = dune_state->GetPlayerTechTilesForTesting(p);
  if (std::find(tech_tiles.begin(), tech_tiles.end(), 6) != tech_tiles.end()) {
    int tsmf_count = 0;
    auto count_tsmf = [&](const std::vector<int>& cards) {
      for (int id : cards) {
        if (id == dune_imperium::kCardTheSpiceMustFlow) tsmf_count++;
      }
    };
    auto* mutable_state = const_cast<dune_imperium::DuneImperiumState*>(dune_state);
    count_tsmf(mutable_state->GetPlayerDrawDeckForTesting(p));
    count_tsmf(mutable_state->GetPlayerDiscardForTesting(p));
    count_tsmf(mutable_state->GetPlayerHandForTesting(p));
    count_tsmf(dune_state->GetPlayedAgentCardsForTesting(p));
    count_tsmf(dune_state->GetRevealedCardsForTesting(p));
    if (tsmf_count >= 2) {
      endgame_vp += 1;
    }
  }

  // 3. Faction Influence milestone (>=3 in all 4 factions)
  bool all_3 = true;
  for (int f = 0; f < 4; ++f) {
    if (dune_state->GetPlayerInfluenceForTesting(p, static_cast<dune_imperium::Faction>(f)) < 3) {
      all_3 = false;
    }
  }
  if (all_3) {
    endgame_vp += 1;
  }

  // 4. Tech tile 14: Spy Satellites
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

  ActionsAndProbs GetPolicy(const State& state) override {
    return evaluator_->Prior(state);
  }

  std::pair<ActionsAndProbs, Action> StepWithPolicy(const State& state) override {
    ActionsAndProbs policy = GetPolicy(state);
    Action chosen = SelectBestAction(policy, state);
    return {policy, chosen};
  }

  void Restart() override {}
  void RestartAt(const State& state) override {}

 private:
  double RandomNumber() {
    return absl::Uniform(rng_, 0.0, 1.0);
  }

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
    } else {
      ActionsAndProbs scaled_policy;
      scaled_policy.reserve(policy.size());
      double sum = 0.0;
      double inv_temp = 1.0 / temp;
      for (const auto& ap : policy) {
        double p = std::pow(std::max(ap.second, 1e-12), inv_temp);
        scaled_policy.push_back({ap.first, p});
        sum += p;
      }
      for (auto& ap : scaled_policy) {
        ap.second /= sum;
      }
      return SampleAction(scaled_policy, RandomNumber()).first;
    }
  }

  std::unique_ptr<DuneNNEvaluator> evaluator_;
  std::mt19937 rng_;
  double temperature_;
};

class DuneRandomBot : public Bot {
 public:
  DuneRandomBot(int seed) : rng_(seed) {}
  Action Step(const State& state) override {
    if (state.IsTerminal()) return kInvalidAction;
    auto legals = state.LegalActions();
    if (legals.empty()) return kInvalidAction;
    return legals[absl::Uniform(rng_, 0u, legals.size())];
  }
  bool ProvidesPolicy() override { return false; }
  ActionsAndProbs GetPolicy(const State& state) override { return {}; }
  std::pair<ActionsAndProbs, Action> StepWithPolicy(const State& state) override {
    return {{}, Step(state)};
  }
  void Restart() override {}
  void RestartAt(const State& state) override {}
 private:
  std::mt19937 rng_;
};

struct GameStats {
  int search_wins = 0;
  int opponent_wins[3] = {0, 0, 0};
  int search_wins_by_seat[4] = {0, 0, 0, 0};
  int games_by_seat[4] = {0, 0, 0, 0};
  int search_placements[4] = {0, 0, 0, 0};
  double search_return_sum = 0.0;
  double search_return_by_seat[4] = {0.0, 0.0, 0.0, 0.0};
  int search_swordmasters = 0;
  int opponent_swordmasters = 0;
  std::vector<int> rounds_played;
  double search_step_time_sum = 0.0;
  int search_steps_count = 0;
  double abs_terminal_return_sum = 0.0;
  int64_t terminal_returns_count = 0;
  double vp_margin_sum = 0.0;
  int64_t vp_margin_count = 0;
  std::vector<double> vp_margins;
  int search_mcts_steps_count = 0;
  double search_mcts_step_time_sum = 0.0;
  int64_t total_simulations_completed = 0;
  int64_t total_timeouts = 0;
  int64_t total_fallbacks = 0;
  int64_t total_inferences = 0;
  std::map<std::string, int64_t> fallback_reason_counts;
  int64_t total_incomplete_searches = 0;
  std::vector<double> search_mcts_step_times;
  double covered_prior_mass_sum = 0.0;
  int64_t role_counts[7] = {0, 0, 0, 0, 0, 0, 0};
};


// Parses --fresh_search_roles (comma-separated role tokens) into a role set.
// Empty input -> empty set, which the driver treats as "search every role"
// (unchanged full-search behavior). Unknown tokens are a fatal misconfiguration.
std::set<DuneDecisionRole> ParseFreshSearchRoles(const std::string& csv) {
  std::set<DuneDecisionRole> roles;
  std::stringstream ss(csv);
  std::string tok;
  while (std::getline(ss, tok, ',')) {
    size_t b = tok.find_first_not_of(" \t");
    size_t e = tok.find_last_not_of(" \t");
    if (b == std::string::npos) continue;  // skip empty/whitespace token
    tok = tok.substr(b, e - b + 1);
    if (tok == "primary") roles.insert(DuneDecisionRole::kAgentPrimary);
    else if (tok == "continuation") roles.insert(DuneDecisionRole::kAgentContinuation);
    else if (tok == "purchase") roles.insert(DuneDecisionRole::kPurchase);
    else if (tok == "combat") roles.insert(DuneDecisionRole::kCombatIntrigue);
    else if (tok == "other") roles.insert(DuneDecisionRole::kOtherOptional);
    else if (tok == "leader") roles.insert(DuneDecisionRole::kLeaderSelection);
    else if (tok == "forced") roles.insert(DuneDecisionRole::kForcedOrBookkeeping);
    else SpielFatalError("Unrecognized --fresh_search_roles token: '" + tok + "'");
  }
  return roles;
}

// Raw-prior action selection for roles filtered OUT of fresh search in arm 4.
// Matches the policy-only arm's PickActionRespectingTemperature exactly: at
// temperature 0 the first strictly-max-prior action (argmax, first-max
// tie-break); at temperature > 0, an inverse-CDF sample from the prior. Empty
// prior degrades to the first legal action.
Action SelectRawPriorAction(DuneNNEvaluator& evaluator, const State& state,
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


// --- WO-1 Phase 2: seat controller topology -------------------------------

enum class SeatControllerMode { kSingle, kHomogeneous, kMixed };
enum class MixedOpponentController { kRandomLegal, kRawPolicy };

SeatControllerMode ParseControllerMode(const std::string& s) {
  if (s == "single") return SeatControllerMode::kSingle;
  if (s == "homogeneous") return SeatControllerMode::kHomogeneous;
  if (s == "mixed") return SeatControllerMode::kMixed;
  SpielFatalError("Unrecognized --controller_mode: '" + s +
                  "' (expected single|homogeneous|mixed)");
}

MixedOpponentController ParseMixedOpponentController(const std::string& s) {
  if (s == "random_legal") return MixedOpponentController::kRandomLegal;
  if (s == "raw_policy") return MixedOpponentController::kRawPolicy;
  SpielFatalError("Unrecognized --mixed_opponent_controller: '" + s +
                  "' (expected random_legal|raw_policy)");
}

// Exactly what drove one seat for one game. Emitted per game so a result can
// never be re-read without knowing which controller produced it -- the Phase 0
// registration requires slice opponents and controllers to be pinned in the
// run manifest, not described in prose.
struct SeatControllerProvenance {
  std::string controller = "unset";   // search|raw_policy|random_legal
  std::string checkpoint;             // "" when the seat uses no model
  std::string checkpoint_sha256;      // "" when the seat uses no model
  int max_simulations = -1;           // -1 when not a searched seat
  std::string budget_mode = "none";
  double logit_cap = -1.0;            // -1 means "no evaluator on this seat"
};

// Checkpoint digests, computed ONCE in main() BEFORE any worker thread is
// spawned and only read afterwards. That write-before-spawn ordering is what
// makes these safe to touch from every worker without a lock; do not assign to
// them from inside WorkerThread.
std::string g_candidate_ckpt_sha256;
std::string g_opponent_ckpt_sha256;

std::string BudgetModeName(DuneSearchBudgetMode m) {
  switch (m) {
    case DuneSearchBudgetMode::kPolicyOnly: return "policy_only";
    case DuneSearchBudgetMode::kFixedSessionSimulations:
      return "fixed_session_simulations";
    case DuneSearchBudgetMode::kTrainingFullFast: return "training_full_fast";
    case DuneSearchBudgetMode::kLiveDeadline: return "live_deadline";
  }
  return "unknown";
}

void WorkerThread(
    int thread_id,
    std::shared_ptr<const Game> game,
    std::shared_ptr<SharedDunePolicyValueNetImpl> search_model,
    std::shared_ptr<SharedDunePolicyValueNetImpl> opponent_model,
    std::atomic<int>& next_game_id,
    int total_games,
    std::atomic<int>& completed_games,
    std::mutex& log_mutex,
    GameStats& global_stats,
    std::mutex& stats_mutex) {

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

    bool rotate_seat = absl::GetFlag(FLAGS_rotate_seat);
    int search_seat = rotate_seat ? (g % 4) : 0;
    thread_stats.games_by_seat[search_seat]++;

    torch::Device device = torch::cuda::is_available() ? torch::Device(torch::kCUDA) : torch::Device(torch::kCPU);
    // Create thread-local, bot-specific evaluator wrapping the shared search model
    auto search_evaluator = std::make_shared<DuneNNEvaluator>(
        search_model, device, candidate_logit_cap);

    // Which seats run the search controller this game. In 'single' (legacy) and
    // 'mixed' that is exactly the candidate seat; in 'homogeneous' it is all
    // four. `search_seat` remains the CANDIDATE seat in every mode, so all the
    // candidate-relative statistics below keep their meaning.
    std::array<bool, 4> seat_is_searched{};
    for (int p = 0; p < 4; ++p) {
      seat_is_searched[p] = (controller_mode == SeatControllerMode::kHomogeneous)
                                ? true
                                : (p == search_seat);
    }
    std::array<SeatControllerProvenance, 4> seat_provenance;

    // One session per searched seat. Index `search_seat` is the candidate's.
    std::vector<std::unique_ptr<DuneSearchSession>> seat_sessions(4);
    // Per-seat evaluators: searched seats share the candidate evaluator (all
    // four copies are the SAME controller in homogeneous mode, by definition).
    std::vector<std::shared_ptr<DuneNNEvaluator>> seat_evaluators(4);
    // Path B (--use_session=false): a plain search bot driven by Step() per
    // decision, reproducing the July-14 reference protocol verbatim.
    std::unique_ptr<DunePUCTISMCTSBot> search_bot;
    std::vector<std::unique_ptr<Bot>> bots(4);
    for (int p = 0; p < 4; ++p) {
      // Exactly ONE game_rng() draw per seat, in seat order, in EVERY mode.
      // The legacy code drew once for the searched seat's config.seed and once
      // per opponent bot, which is the same count and order; preserving that is
      // what keeps 'single' mode bitwise-identical to the pre-WO-1 binary.
      // Moving, adding or conditionalising this draw silently reshuffles every
      // downstream chance realization.
      const uint64_t seat_seed = game_rng();
      if (seat_is_searched[p]) {
        // Designated initializers bind by NAME. This block used to be
        // positional, and Phase 18B's three inserted KataGo fields shifted
        // everything after dirichlet_alpha by three slots, so this binary
        // silently ran with dirichlet_alpha_total=1.0 and
        // root_noise_fpu_zero=true while DROPPING --check_strategic_state,
        // --verbose_diagnostics and --root_prior_temperature entirely
        // (bool<->double in a braced-init list is only a -Wnarrowing warning).
        // See WO-15 / search finding 2 — latent in the legacy constructor, live
        // here.
        DuneSearchConfig config{
            .max_simulations = absl::GetFlag(FLAGS_max_simulations),
            .relative_time_budget_ms =
                absl::GetFlag(FLAGS_disable_time_limit)
                    ? std::numeric_limits<double>::infinity()
                    : absl::GetFlag(FLAGS_relative_time_budget_ms),
            .max_nodes = 50000,
            .puct_c = absl::GetFlag(FLAGS_puct_c),
            .opponent_mode = absl::GetFlag(FLAGS_use_opponent_model)
                                 ? SearchOpponentMode::kPolicy
                                 : SearchOpponentMode::kMaxN,
            .temperature = absl::GetFlag(FLAGS_temperature),
            .opponent_temperature =
                absl::GetFlag(FLAGS_simulated_opponent_temperature),
            .max_world_samples = absl::GetFlag(FLAGS_max_world_samples),
            .utility_divisor = absl::GetFlag(FLAGS_utility_divisor),
            .min_visit_threshold = 2,
            .covered_prior_threshold = 0.50,
            .seed = seat_seed,
            .final_policy_type = DuneISMCTSFinalPolicyType::kNormalizedVisitCount,
            .dirichlet_epsilon = absl::GetFlag(FLAGS_dirichlet_epsilon),
            .dirichlet_alpha = absl::GetFlag(FLAGS_dirichlet_alpha),
            .use_observation_string = true,
            .verbose_diagnostics = absl::GetFlag(FLAGS_verbose_diagnostics),
            .check_strategic_state = absl::GetFlag(FLAGS_check_strategic_state),
            .root_prior_temperature =
                absl::GetFlag(FLAGS_root_prior_temperature),
        };
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
        DuneSearchBudgetMode budget_mode = absl::GetFlag(FLAGS_policy_only)
            ? DuneSearchBudgetMode::kPolicyOnly
            : (absl::GetFlag(FLAGS_live_deadline)
                ? DuneSearchBudgetMode::kLiveDeadline
                : DuneSearchBudgetMode::kFixedSessionSimulations);
        seat_evaluators[p] = search_evaluator;
        if (absl::GetFlag(FLAGS_use_session)) {
          seat_sessions[p] = std::make_unique<DuneSearchSession>(config, search_evaluator, budget_mode);
        } else {
          // Reference protocol: fresh full search per decision, no session.
          // Path B is single-seat only (guarded in main()), so this is the
          // candidate's bot.
          search_bot = std::make_unique<DunePUCTISMCTSBot>(config, search_evaluator);
        }
        seat_provenance[p] = SeatControllerProvenance{
            /*controller=*/"search",
            /*checkpoint=*/absl::GetFlag(FLAGS_model_checkpoint),
            /*checkpoint_sha256=*/g_candidate_ckpt_sha256,
            /*max_simulations=*/absl::GetFlag(FLAGS_max_simulations),
            /*budget_mode=*/BudgetModeName(budget_mode),
            /*logit_cap=*/static_cast<double>(candidate_logit_cap)};
      } else if (controller_mode == SeatControllerMode::kMixed &&
                 mixed_opponent == MixedOpponentController::kRandomLegal) {
        // Slice S1: candidate + 3x random-legal, stated explicitly rather than
        // inferred from --opponent_checkpoint=random.
        bots[p] = std::make_unique<DuneRandomBot>(static_cast<int>(seat_seed));
        seat_provenance[p] = SeatControllerProvenance{
            /*controller=*/"random_legal", /*checkpoint=*/"",
            /*checkpoint_sha256=*/"", /*max_simulations=*/-1,
            /*budget_mode=*/"none", /*logit_cap=*/-1.0};
      } else {
        // 'single' legacy behavior, and 'mixed' with raw_policy opponents: both
        // resolve to the --opponent_checkpoint-driven bot. opponent_model is
        // null exactly when --opponent_checkpoint=random.
        if (opponent_model != nullptr) {
          auto local_opp_eval = std::make_unique<DuneNNEvaluator>(
              opponent_model, device, opponent_logit_cap);
          bots[p] = std::make_unique<DuneGreedyBot>(
              std::move(local_opp_eval), static_cast<int>(seat_seed),
              absl::GetFlag(FLAGS_external_opponent_temperature));
          seat_provenance[p] = SeatControllerProvenance{
              /*controller=*/"raw_policy",
              /*checkpoint=*/absl::GetFlag(FLAGS_opponent_checkpoint).empty()
                  ? absl::GetFlag(FLAGS_model_checkpoint)
                  : absl::GetFlag(FLAGS_opponent_checkpoint),
              /*checkpoint_sha256=*/g_opponent_ckpt_sha256,
              /*max_simulations=*/-1, /*budget_mode=*/"none",
              /*logit_cap=*/static_cast<double>(opponent_logit_cap)};
        } else {
          bots[p] = std::make_unique<DuneRandomBot>(static_cast<int>(seat_seed));
          seat_provenance[p] = SeatControllerProvenance{
              /*controller=*/"random_legal", /*checkpoint=*/"",
              /*checkpoint_sha256=*/"", /*max_simulations=*/-1,
              /*budget_mode=*/"none", /*logit_cap=*/-1.0};
        }
      }
    }

    std::unique_ptr<State> state = game->NewInitialState();
    int game_length = 0;

    std::vector<int> game_role_counts(7, 0);
    bool in_search_turn = false;
    int turn_decisions = 0;
    int turn_primary_sims = 0;
    int turn_continuation_sims = 0;
    double turn_wall_time_ms = 0.0;
    bool turn_had_timeout = false;
    bool turn_had_fallback = false;
    std::string turn_fallback_reason = "none";

    // Probe telemetry (3b): the searched seat's Swordmaster-acquisition round and
    // its solari trajectory by round. Forced funding/acquisition decisions bypass
    // the search log, so this in-loop sampling is the only place they surface.
    int sm_acquire_round = -1;
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
    constexpr int kPaceMaxRounds = 10;
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
        if (sm_acquire_round < 0 && trk->HasSwordmaster(search_seat)) {
          sm_acquire_round = trk_round;
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
      if (in_search_turn && current_player != search_seat && current_player != kChancePlayerId) {
        std::cout << "--- End of Agent Turn Audit Summary ---\n"
                  << "Decisions: " << turn_decisions << " (Primary: 1, Continuations: " << (turn_decisions - 1) << ")\n"
                  << "Total simulations completed: " << (turn_primary_sims + turn_continuation_sims) << " (Primary: " << turn_primary_sims << ", Continuation: " << turn_continuation_sims << ")\n"
                  << "Total wall-clock time: " << turn_wall_time_ms << " ms\n"
                  << "Timeout status: " << (turn_had_timeout ? "Yes" : "No") << "\n"
                  << "Fallback status: " << (turn_had_fallback ? "Yes" : "No") << " (Reason: " << turn_fallback_reason << ")\n"
                  << "---------------------------------------\n";
        in_search_turn = false;
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

      Action chosen_action = -1;

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
      } else if (seat_is_searched[current_player]) {
        // In homogeneous mode every seat lands here; `cur_session` is that seat's
        // own session, never a shared one.
        auto& cur_session = seat_sessions[current_player];
        bool has_active = cur_session ? cur_session->HasActiveSession() : false;
        open_spiel::DuneDecisionRole role = open_spiel::ClassifyDuneDecisionRole(*state, current_player, has_active);
        game_role_counts[static_cast<int>(role)]++;
        bool is_strategic = (role == open_spiel::DuneDecisionRole::kAgentPrimary || role == open_spiel::DuneDecisionRole::kAgentContinuation);
        auto step_start = std::chrono::steady_clock::now();
        DuneSearchResult last_res;
        if (cur_session) {
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
          const bool search_this_role =
              fresh_search_roles.empty() || fresh_search_roles.count(role) > 0;
          if (search_this_role) {
            chosen_action = search_bot->Step(*state);
            last_res = search_bot->GetLastSearchResult();
            last_res.diagnostics.selected_action = chosen_action;
          } else {
            chosen_action = SelectRawPriorAction(
                *seat_evaluators[current_player], *state,
                absl::GetFlag(FLAGS_temperature), game_rng);
            last_res = DuneSearchResult();
            last_res.simulations_completed = 0;
            last_res.used_fallback = true;
            last_res.fallback_reason = "role_filtered_raw_prior";
            last_res.diagnostics.selected_action = chosen_action;
            last_res.diagnostics.decision_role = std::to_string(static_cast<int>(role));
          }
        }
        auto step_end = std::chrono::steady_clock::now();
        double step_duration = std::chrono::duration<double>(step_end - step_start).count();

        // Update turn statistics
        if (role == open_spiel::DuneDecisionRole::kAgentPrimary) {
          in_search_turn = true;
          turn_decisions = 1;
          turn_primary_sims = last_res.simulations_completed;
          turn_continuation_sims = 0;
          turn_wall_time_ms = step_duration * 1000.0;
          turn_had_timeout = last_res.timeout_status;
          turn_had_fallback = last_res.used_fallback;
          turn_fallback_reason = last_res.fallback_reason;
        } else if (in_search_turn && role == open_spiel::DuneDecisionRole::kAgentContinuation) {
          turn_decisions++;
          turn_continuation_sims += last_res.simulations_completed;
          turn_wall_time_ms += step_duration * 1000.0;
          if (last_res.timeout_status) turn_had_timeout = true;
          if (last_res.used_fallback) {
            turn_had_fallback = true;
            turn_fallback_reason = last_res.fallback_reason;
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
              if (last_res.fallback_reason == "timeout" || last_res.fallback_reason == "max_nodes" || last_res.timeout_status) {
                std::cerr << "\nCRITICAL FAILURE: Strategic search stopped before completing "
                          << target_sims
                          << " simulations due to " << last_res.fallback_reason
                          << " (completed " << completed_sims << " simulations).\n";
                std::exit(1);
              }
            }
          }
        }
            std::string search_jsonl = absl::GetFlag(FLAGS_search_jsonl_path);
            if (!search_jsonl.empty()) {
              SearchDiagnostics diag = last_res.diagnostics;
              open_spiel::json::Object search_obj;
              search_obj["episode_id"] = static_cast<int64_t>(g);
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
              search_obj["elapsed_time_ms"] = last_res.elapsed_time_ms;
              search_obj["action_chosen"] = static_cast<int64_t>(chosen_action);
              const auto* dune_state = dynamic_cast<const dune_imperium::DuneImperiumState*>(state.get());
              if (dune_state != nullptr) {
                search_obj["action_chosen_string"] = dune_state->ActionToString(current_player, chosen_action);
              }
              open_spiel::json::Array legals_arr;
              for (Action a : state->LegalActions()) {
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
              search_obj["hard_sim_limit"] = static_cast<int64_t>(diag.hard_sim_limit);
              search_obj["soft_sim_limit"] = static_cast<int64_t>(diag.soft_sim_limit);
              search_obj["hard_time_limit_ms"] = diag.hard_time_limit_ms;
              search_obj["soft_time_limit_ms"] = diag.soft_time_limit_ms;
              search_obj["elapsed_search_time_ms"] = diag.elapsed_search_time_ms;
              search_obj["observation_wait_time_ms"] = diag.observation_wait_time_ms;
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
          } else {
        chosen_action = bots[current_player]->Step(*state);
      }

      // WO-1 Phase 3 addendum: count applied specimen->troop conversions per
      // seat, at application time, by engine action-ID range. Measurement only
      // -- no shaping is enabled anywhere by this counter.
      if (chosen_action >= dune_imperium::kActionConvertSpecimenToTroop0 &&
          chosen_action <= dune_imperium::kActionConvertSpecimenToTroop0 + 12 &&
          current_player >= 0 && current_player < 4) {
        ++specimen_conversions[current_player];
      }
      state->ApplyAction(chosen_action);
    }

    // Capture the terminal round's VP snapshot (the loop exits before it).
    snapshot_round_ends();

    if (in_search_turn) {
      std::cout << "--- End of Agent Turn Audit Summary ---\n"
                << "Decisions: " << turn_decisions << " (Primary: 1, Continuations: " << (turn_decisions - 1) << ")\n"
                << "Total simulations completed: " << (turn_primary_sims + turn_continuation_sims) << " (Primary: " << turn_primary_sims << ", Continuation: " << turn_continuation_sims << ")\n"
                << "Total wall-clock time: " << turn_wall_time_ms << " ms\n"
                << "Timeout status: " << (turn_had_timeout ? "Yes" : "No") << "\n"
                << "Fallback status: " << (turn_had_fallback ? "Yes" : "No") << " (Reason: " << turn_fallback_reason << ")\n"
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

    std::vector<double> returns = state->Returns();
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

    const auto* dune_state = dynamic_cast<const dune_imperium::DuneImperiumState*>(state.get());
    if (dune_state != nullptr) {
      if (g == 67) {
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
      for (int p = 0; p < 4; ++p) {
        if (dune_state->HasSwordmaster(p)) {
          if (p == search_seat) {
            thread_stats.search_swordmasters++;
          } else {
            thread_stats.opponent_swordmasters++;
          }
        }
      }
      int corrected_round = dune_state->GetCurrentRound() - 1;
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

      double game_duration_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - game_start_time).count();

      std::string game_jsonl = absl::GetFlag(FLAGS_game_jsonl_path);
      if (!game_jsonl.empty()) {
        open_spiel::json::Object game_obj;
        game_obj["episode_id"] = static_cast<int64_t>(g);
        game_obj["seed"] = static_cast<int64_t>(game_seed);
        game_obj["search_seat"] = static_cast<int64_t>(search_seat);
        game_obj["winner"] = static_cast<int64_t>(winner);

        open_spiel::json::Array returns_arr;
        for (double r : returns) {
          returns_arr.push_back(r);
        }
        game_obj["returns"] = returns_arr;
        game_obj["search_return"] = returns[search_seat];
        game_obj["search_vp"] = search_vp;

        open_spiel::json::Array opp_vps_arr;
        for (int p = 0; p < 4; ++p) {
          if (p != search_seat) {
            opp_vps_arr.push_back(static_cast<double>(GetTrueFinalVp(dune_state, p)));
          }
        }
        game_obj["opponent_vps"] = opp_vps_arr;

        game_obj["vp_margin"] = margin;
        game_obj["rounds_played"] = static_cast<int64_t>(corrected_round);
        game_obj["search_steps"] = static_cast<int64_t>(thread_stats.search_steps_count);
        game_obj["search_swordmasters"] = dune_state->HasSwordmaster(search_seat);
        game_obj["search_seat_leader"] = static_cast<int64_t>(dune_state->PlayerLeader(search_seat));
        game_obj["search_seat_swordmaster_round"] = static_cast<int64_t>(sm_acquire_round);
        open_spiel::json::Array sbr_rounds, sbr_solari;
        for (const auto& kv : search_seat_solari_by_round) {
          sbr_rounds.push_back(static_cast<int64_t>(kv.first));
          sbr_solari.push_back(static_cast<int64_t>(kv.second));
        }
        game_obj["search_seat_solari_rounds"] = sbr_rounds;
        game_obj["search_seat_solari_values"] = sbr_solari;
        // Third-agent utilization: per round, agents available (peak) vs placed
        // (deployed) by the searched seat. Iterate the availability map so every
        // observed round appears; placements default to 0 when none were used.
        open_spiel::json::Array spr_rounds, spr_placed, spr_avail;
        for (const auto& kv : search_seat_agents_avail_by_round) {
          spr_rounds.push_back(static_cast<int64_t>(kv.first));
          spr_avail.push_back(static_cast<int64_t>(kv.second));
          auto it = search_seat_placements_by_round.find(kv.first);
          spr_placed.push_back(static_cast<int64_t>(
              it == search_seat_placements_by_round.end() ? 0 : it->second));
        }
        game_obj["search_seat_placement_rounds"] = spr_rounds;
        game_obj["search_seat_placements_used"] = spr_placed;
        game_obj["search_seat_agents_available"] = spr_avail;

        open_spiel::json::Array opp_sm_arr;
        for (int p = 0; p < 4; ++p) {
          if (p != search_seat) {
            opp_sm_arr.push_back(dune_state->HasSwordmaster(p));
          }
        }
        game_obj["opponent_swordmasters"] = opp_sm_arr;
        game_obj["wall_time_s"] = game_duration_s;

        // --- WO-1 Phase 2: per-seat controller provenance -------------------
        // The candidate seat under analysis. Equal to search_seat in every
        // mode; emitted under its own name because in homogeneous mode
        // "the searched seat" is no longer a unique thing.
        game_obj["candidate_seat"] = static_cast<int64_t>(search_seat);
        game_obj["controller_mode"] = absl::GetFlag(FLAGS_controller_mode);
        open_spiel::json::Array seat_ctrl_arr;
        for (int p = 0; p < 4; ++p) {
          open_spiel::json::Object sc;
          sc["seat"] = static_cast<int64_t>(p);
          sc["controller"] = seat_provenance[p].controller;
          sc["checkpoint"] = seat_provenance[p].checkpoint;
          sc["checkpoint_sha256"] = seat_provenance[p].checkpoint_sha256;
          sc["max_simulations"] =
              static_cast<int64_t>(seat_provenance[p].max_simulations);
          sc["budget_mode"] = seat_provenance[p].budget_mode;
          sc["logit_cap"] = seat_provenance[p].logit_cap;
          sc["is_candidate"] = (p == search_seat);
          seat_ctrl_arr.push_back(sc);
        }
        game_obj["seat_controllers"] = seat_ctrl_arr;

        // --- WO-1 Phase 3: pace/tempo ---------------------------------------
        int rounds_captured = 0;
        open_spiel::json::Array vp_by_round_arr;
        for (int rd = 1; rd <= kPaceMaxRounds; ++rd) {
          if (!round_end_seen[rd]) break;
          open_spiel::json::Array one_round;
          for (int p = 0; p < 4; ++p) {
            one_round.push_back(static_cast<int64_t>(vp_at_round_end[rd][p]));
          }
          vp_by_round_arr.push_back(one_round);
          rounds_captured = rd;
        }
        game_obj["vp_end_by_round"] = vp_by_round_arr;

        open_spiel::json::Array first_ge11_arr;
        for (int p = 0; p < 4; ++p) {
          int first = -1;
          for (int rd = 1; rd <= rounds_captured; ++rd) {
            if (vp_at_round_end[rd][p] >= dune_imperium::kTargetVp) { first = rd; break; }
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
            if (p == search_seat) cand_in_thr = true;
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
          spec_arr.push_back(static_cast<int64_t>(specimen_conversions[p]));
        }
        game_obj["specimen_conversions"] = spec_arr;

        std::lock_guard<std::mutex> lock(log_mutex);
        std::ofstream game_file(game_jsonl, std::ios::app);
        if (game_file) {
          game_file << open_spiel::json::ToString(game_obj, false) << "\n";
        }
      }
    }

    // Flush stats and perform intermediate logging under a single lock
    {
      std::lock_guard<std::mutex> stats_lock(stats_mutex);
      global_stats.search_wins += thread_stats.search_wins;
      for (int p = 0; p < 3; ++p) global_stats.opponent_wins[p] += thread_stats.opponent_wins[p];
      for (int s = 0; s < 4; ++s) {
        global_stats.search_wins_by_seat[s] += thread_stats.search_wins_by_seat[s];
        global_stats.games_by_seat[s] += thread_stats.games_by_seat[s];
        global_stats.search_return_by_seat[s] += thread_stats.search_return_by_seat[s];
      }
      for (int r = 0; r < 4; ++r) global_stats.search_placements[r] += thread_stats.search_placements[r];
      global_stats.search_return_sum += thread_stats.search_return_sum;
      global_stats.search_swordmasters += thread_stats.search_swordmasters;
      global_stats.opponent_swordmasters += thread_stats.opponent_swordmasters;
      global_stats.rounds_played.insert(global_stats.rounds_played.end(), thread_stats.rounds_played.begin(), thread_stats.rounds_played.end());
      global_stats.search_step_time_sum += thread_stats.search_step_time_sum;
      global_stats.search_steps_count += thread_stats.search_steps_count;
      global_stats.abs_terminal_return_sum += thread_stats.abs_terminal_return_sum;
      global_stats.terminal_returns_count += thread_stats.terminal_returns_count;
      global_stats.vp_margin_sum += thread_stats.vp_margin_sum;
      global_stats.vp_margin_count += thread_stats.vp_margin_count;
      global_stats.vp_margins.insert(global_stats.vp_margins.end(), thread_stats.vp_margins.begin(), thread_stats.vp_margins.end());
      global_stats.search_mcts_steps_count += thread_stats.search_mcts_steps_count;
      global_stats.search_mcts_step_time_sum += thread_stats.search_mcts_step_time_sum;
      global_stats.total_simulations_completed += thread_stats.total_simulations_completed;
      global_stats.total_timeouts += thread_stats.total_timeouts;
      global_stats.total_fallbacks += thread_stats.total_fallbacks;
      global_stats.total_inferences += thread_stats.total_inferences;
      global_stats.total_incomplete_searches += thread_stats.total_incomplete_searches;
      global_stats.search_mcts_step_times.insert(global_stats.search_mcts_step_times.end(), thread_stats.search_mcts_step_times.begin(), thread_stats.search_mcts_step_times.end());
      global_stats.covered_prior_mass_sum += thread_stats.covered_prior_mass_sum;
      for (const auto& pair : thread_stats.fallback_reason_counts) {
        global_stats.fallback_reason_counts[pair.first] += pair.second;
      }
      for (int i = 0; i < 7; ++i) {
        global_stats.role_counts[i] += game_role_counts[i];
      }

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
    global_stats.search_wins += thread_stats.search_wins;
    for (int p = 0; p < 3; ++p) global_stats.opponent_wins[p] += thread_stats.opponent_wins[p];
    for (int s = 0; s < 4; ++s) {
      global_stats.search_wins_by_seat[s] += thread_stats.search_wins_by_seat[s];
      global_stats.games_by_seat[s] += thread_stats.games_by_seat[s];
      global_stats.search_return_by_seat[s] += thread_stats.search_return_by_seat[s];
    }
    for (int r = 0; r < 4; ++r) global_stats.search_placements[r] += thread_stats.search_placements[r];
    global_stats.search_return_sum += thread_stats.search_return_sum;
    global_stats.search_swordmasters += thread_stats.search_swordmasters;
    global_stats.opponent_swordmasters += thread_stats.opponent_swordmasters;
    global_stats.rounds_played.insert(global_stats.rounds_played.end(), thread_stats.rounds_played.begin(), thread_stats.rounds_played.end());
    global_stats.search_step_time_sum += thread_stats.search_step_time_sum;
    global_stats.search_steps_count += thread_stats.search_steps_count;
    global_stats.abs_terminal_return_sum += thread_stats.abs_terminal_return_sum;
    global_stats.terminal_returns_count += thread_stats.terminal_returns_count;
    global_stats.vp_margin_sum += thread_stats.vp_margin_sum;
    global_stats.vp_margin_count += thread_stats.vp_margin_count;
    global_stats.vp_margins.insert(global_stats.vp_margins.end(), thread_stats.vp_margins.begin(), thread_stats.vp_margins.end());
    global_stats.search_mcts_steps_count += thread_stats.search_mcts_steps_count;
    global_stats.search_mcts_step_time_sum += thread_stats.search_mcts_step_time_sum;
    global_stats.total_simulations_completed += thread_stats.total_simulations_completed;
    global_stats.total_timeouts += thread_stats.total_timeouts;
    global_stats.total_fallbacks += thread_stats.total_fallbacks;
    global_stats.total_inferences += thread_stats.total_inferences;
    global_stats.total_incomplete_searches += thread_stats.total_incomplete_searches;
    global_stats.search_mcts_step_times.insert(global_stats.search_mcts_step_times.end(), thread_stats.search_mcts_step_times.begin(), thread_stats.search_mcts_step_times.end());
    global_stats.covered_prior_mass_sum += thread_stats.covered_prior_mass_sum;
    for (const auto& pair : thread_stats.fallback_reason_counts) {
      global_stats.fallback_reason_counts[pair.first] += pair.second;
    }
  }
}

} // namespace
} // namespace open_spiel

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

int main(int argc, char* argv[]) {
  auto run_start_time = std::chrono::steady_clock::now();
  absl::ParseCommandLine(argc, argv);

  // Arm-4 (agent-phase decomposition): the --fresh_search_roles filter is only
  // defined for the Path B fresh-search driver. Fail fast on misuse or typos,
  // before any threads spawn.
  if (!absl::GetFlag(FLAGS_fresh_search_roles).empty()) {
    if (absl::GetFlag(FLAGS_use_session)) {
      open_spiel::SpielFatalError(
          "--fresh_search_roles requires --use_session=false (Path B fresh search).");
    }
    (void)open_spiel::ParseFreshSearchRoles(absl::GetFlag(FLAGS_fresh_search_roles));
  }

  // Set PyTorch thread limit to 1 to avoid thread contention across CPU-bound forward runs
  at::set_num_threads(1);
  at::set_num_interop_threads(1);

  std::string model_ckpt = absl::GetFlag(FLAGS_model_checkpoint);
  if (model_ckpt.empty()) {
    std::cerr << "Error: --model_checkpoint is required!\n";
    return 1;
  }

  // --- WO-1 Phase 2: validate the seat controller topology ------------------
  const open_spiel::SeatControllerMode cli_controller_mode =
      open_spiel::ParseControllerMode(absl::GetFlag(FLAGS_controller_mode));
  open_spiel::ParseMixedOpponentController(
      absl::GetFlag(FLAGS_mixed_opponent_controller));  // validate early
  if (cli_controller_mode != open_spiel::SeatControllerMode::kSingle &&
      !absl::GetFlag(FLAGS_use_session)) {
    std::cerr << "Error: --controller_mode=" << absl::GetFlag(FLAGS_controller_mode)
              << " requires --use_session=true. Path B (--use_session=false) is "
                 "the single-seat July-14 reference protocol and has no "
                 "multi-seat form; running it here would silently search only "
                 "one seat.\n";
    return 1;
  }
  if (cli_controller_mode == open_spiel::SeatControllerMode::kHomogeneous &&
      absl::GetFlag(FLAGS_opponent_checkpoint) != "random" &&
      !absl::GetFlag(FLAGS_opponent_checkpoint).empty()) {
    std::cerr << "Warning: --controller_mode=homogeneous searches all four "
                 "seats with the candidate model; --opponent_checkpoint="
              << absl::GetFlag(FLAGS_opponent_checkpoint)
              << " is unused and will not appear in seat provenance.\n";
  }

  // Checkpoint digests for per-seat provenance. Computed ONCE here, before any
  // worker thread exists, and only read afterwards -- that ordering is what
  // makes the globals safe without a lock.
  open_spiel::g_candidate_ckpt_sha256 =
      open_spiel::ComputeFileSHA256(model_ckpt);
  {
    const std::string opp = absl::GetFlag(FLAGS_opponent_checkpoint);
    if (opp == "random") {
      open_spiel::g_opponent_ckpt_sha256 = "";
    } else if (opp.empty()) {
      open_spiel::g_opponent_ckpt_sha256 = open_spiel::g_candidate_ckpt_sha256;
    } else {
      open_spiel::g_opponent_ckpt_sha256 = open_spiel::ComputeFileSHA256(opp);
    }
  }

  int hidden_dim = absl::GetFlag(FLAGS_hidden_dim);
  int num_blocks = absl::GetFlag(FLAGS_num_blocks);
  int opp_hidden_dim = absl::GetFlag(FLAGS_opp_hidden_dim);
  if (opp_hidden_dim < 0) opp_hidden_dim = hidden_dim;
  int opp_num_blocks = absl::GetFlag(FLAGS_opp_num_blocks);
  if (opp_num_blocks < 0) opp_num_blocks = num_blocks;

  std::shared_ptr<const open_spiel::Game> game = open_spiel::LoadGame("dune_imperium");
  int64_t obs_size = game->InformationStateTensorSize();
  int64_t action_size = game->NumDistinctActions();

  torch::Device device = torch::cuda::is_available() ? torch::Device(torch::kCUDA) : torch::Device(torch::kCPU);

  std::cout << "Loading search model weights from: " << model_ckpt << " on device " << device << "\n";
  auto search_model = std::make_shared<open_spiel::SharedDunePolicyValueNetImpl>(
      obs_size, hidden_dim, action_size, num_blocks,
      absl::GetFlag(FLAGS_nonlinear_value_head));
  search_model->eval();
  try {
    torch::serialize::InputArchive archive;
    archive.load_from(model_ckpt, device);
    search_model->load(archive);
  } catch (const c10::Error& e) {
    std::cerr << "Failed to load search model weights:\n" << e.msg() << "\n";
    return 1;
  }
  search_model->to(device);

  std::shared_ptr<open_spiel::SharedDunePolicyValueNetImpl> opponent_model = nullptr;
  std::string opp_ckpt = absl::GetFlag(FLAGS_opponent_checkpoint);
  if (opp_ckpt != "random") {
    if (opp_ckpt.empty()) {
      opp_ckpt = model_ckpt;
    }
    std::cout << "Loading opponent model weights from: " << opp_ckpt << " on device " << device << "\n";
    opponent_model = std::make_shared<open_spiel::SharedDunePolicyValueNetImpl>(
        obs_size, opp_hidden_dim, action_size, opp_num_blocks,
        absl::GetFlag(FLAGS_opponent_nonlinear_value_head));
    opponent_model->eval();
    try {
      torch::serialize::InputArchive archive;
      archive.load_from(opp_ckpt, device);
      opponent_model->load(archive);
    } catch (const c10::Error& e) {
      std::cerr << "Failed to load opponent model weights:\n" << e.msg() << "\n";
      return 1;
    }
    opponent_model->to(device);
  } else {
    std::cout << "Running against uniform random opponents.\n";
  }

  int total_games = absl::GetFlag(FLAGS_games);
  int num_threads = absl::GetFlag(FLAGS_threads);
  if (num_threads <= 0) {
    num_threads = std::max(1, (int)std::thread::hardware_concurrency());
  }

  std::cout << "\nStarting evaluation of " << total_games << " games using " << num_threads << " threads...\n\n";

  std::atomic<int> next_game_id{0};
  std::atomic<int> completed_games{0};
  std::mutex log_mutex;
  std::mutex stats_mutex;
  open_spiel::GameStats global_stats;

  std::vector<std::thread> worker_threads;
  worker_threads.reserve(num_threads);
  for (int t = 0; t < num_threads; ++t) {
    worker_threads.emplace_back(
        open_spiel::WorkerThread, t, game, search_model, opponent_model,
        std::ref(next_game_id), total_games, std::ref(completed_games),
        std::ref(log_mutex), std::ref(global_stats), std::ref(stats_mutex));
  }

  for (auto& th : worker_threads) {
    if (th.joinable()) th.join();
  }

  // Print summary results
  std::cout << "\n======================================================================\n";
  std::cout << "Evaluation Completed!\n";
  std::cout << "Total Games:         " << total_games << "\n";
  std::cout << "Search Checkpoint:   " << model_ckpt << "\n";
  std::cout << "Opponent Checkpoint: " << (opponent_model == nullptr ? "Uniform Random Agents" : (opp_ckpt == model_ckpt ? "Same as search (Self-play evaluation)" : opp_ckpt)) << "\n";
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
    agg_obj["utility_divisor"] = absl::GetFlag(FLAGS_utility_divisor);
    agg_obj["elapsed_wall_time_s"] = total_wall_time;
    agg_obj["games_per_hour"] = total_games * 3600.0 / total_wall_time;
    agg_obj["search_wins"] = static_cast<int64_t>(global_stats.search_wins);
    agg_obj["search_winrate"] = (global_stats.search_wins * 100.0 / total_games);

    double lcb_winrate = 0.0, ucb_winrate = 0.0;
    ComputeWilsonWinrateCI(global_stats.search_wins, total_games, &lcb_winrate, &ucb_winrate);
    agg_obj["search_winrate_lcb"] = lcb_winrate;
    agg_obj["search_winrate_ucb"] = ucb_winrate;

    double mean_vp_margin = 0.0, lcb_vp_margin = 0.0, ucb_vp_margin = 0.0;
    ComputeVpMarginCI(global_stats.vp_margins, &mean_vp_margin, &lcb_vp_margin, &ucb_vp_margin);
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
    std::map<int, int> round_counts;
    double sum_rounds = 0.0;
    for (int r : global_stats.rounds_played) {
      round_counts[r]++;
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

  return 0;
}
