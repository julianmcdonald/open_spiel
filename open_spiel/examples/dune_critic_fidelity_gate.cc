#include <iostream>
#include <fstream>
#include <memory>
#include <vector>
#include <string>
#include <algorithm>
#include <filesystem>
#include <random>
#include <cmath>
#include <atomic>
#include <thread>
#include <mutex>
#include <shared_mutex>
#include <limits>
#include <numeric>
#include <unordered_set>
#include <sys/file.h>
#include <fcntl.h>
#include <unistd.h>

#include "open_spiel/spiel.h"
#include "open_spiel/games/dune_imperium/dune_imperium.h"
#include "open_spiel/games/dune_imperium/dune_imperium_common.h"
#include "open_spiel/games/dune_imperium/dune_imperium_board.h"
#include "open_spiel/abseil-cpp/absl/flags/flag.h"
#include "open_spiel/abseil-cpp/absl/flags/parse.h"
#include "open_spiel/abseil-cpp/absl/strings/str_format.h"
#include "open_spiel/abseil-cpp/absl/strings/str_join.h"
#include "open_spiel/abseil-cpp/absl/random/distributions.h"
#include "open_spiel/utils/json.h"
#include <torch/torch.h>

#include "dune_network.h"
#include "dune_evaluator.h"
#include "dune_puct_is_mcts.h"
#include "dune_search_session.h"
#include "dune_search_routing.h"
#include "dune_warmstart_helpers.h"
#include "dune_batched_evaluator.h"
#include "dune_sha256.h"

ABSL_FLAG(std::string, model_checkpoint, "artifacts/branch_a_frozen/branch_a_seed11_model_update_2450.pt", "Path to the model checkpoint");
ABSL_FLAG(std::string, corpus_path, "data/dune_diagnostic_corpus.json", "Path to the corpus JSON file");
ABSL_FLAG(int, seed, 42, "RNG seed");
ABSL_FLAG(int, search_seed, 42, "MCTS Search RNG seed");
ABSL_FLAG(int, raw_policy_seed, 42, "Raw policy RNG seed");
ABSL_FLAG(int, rollout_seed_base, 100000, "Rollout seed base");
ABSL_FLAG(int, chance_seed_base, 200000, "Chance seed base");
ABSL_FLAG(int, bootstrap_seed, 109999, "Bootstrap RNG seed");
ABSL_FLAG(int, hidden_dim, 2048, "Hidden dimension");
ABSL_FLAG(int, num_blocks, 8, "Block count");
ABSL_FLAG(int, max_simulations, 200, "MCTS simulations count");
ABSL_FLAG(double, puct_c, 0.3, "PUCT exploration constant");
ABSL_FLAG(int, opponent_mode, 1, "Opponent modeling mode (0 = kMaxN, 1 = kPolicy)");
ABSL_FLAG(double, opponent_temperature, 1.0, "Opponent temperature for policy mode");
ABSL_FLAG(double, utility_divisor, 4.0, "Utility divisor");
ABSL_FLAG(int, threads, 32, "Number of threads");
ABSL_FLAG(bool, mock_miscalibrated_critic, false, "Debug flag to artificially fail the critic checks");
ABSL_FLAG(bool, self_test, false, "Run quick self-test mode with MockEvaluator");
ABSL_FLAG(bool, strict_v2_validation, true, "Enforce strict V2 metadata checks on opportunity states");
ABSL_FLAG(std::string, report_path, "", "Path to output a structured JSON report");
ABSL_FLAG(int, fixed_continuation_reserve, 0, "Continuation reserve simulations.");
ABSL_FLAG(int, purchase_combat_budget, 16, "Purchase/combat short-window simulation budget.");
ABSL_FLAG(double, root_prior_temperature, 1.0, "Root prior temperature.");
ABSL_FLAG(int, rollouts, 32, "Number of rollouts per state in Gate 2");
ABSL_FLAG(double, relative_time_budget_ms, -1.0, "Relative time budget in ms (-1.0 for infinity)");
ABSL_FLAG(bool, gate1_only, false, "Only run Gate 1 validation (skip Gate 2 checks)");
ABSL_FLAG(bool, allow_any_opportunity_count, false, "Allow any number of opportunity states and bypass strict v2 metadata validations");
ABSL_FLAG(double, live_continuation_reserve_seconds, 0.0, "Continuation reserve in seconds for live deadline mode");
ABSL_FLAG(bool, nonlinear_value_head, false, "Use nonlinear value head architecture");
ABSL_FLAG(bool, choice_only_gate2, false,
          "Evaluate the searched root action once per state and use paired raw-policy continuations.");
ABSL_FLAG(int, diagnostic_rollouts, 256,
          "Raw-policy rollouts per sampled successor/leaf diagnostic (independent of Gate 2 paired rollouts).");
ABSL_FLAG(std::string, gate1_cache_path, "",
          "Optional validated cache for per-state Gate 1 critic and rollout values.");
ABSL_FLAG(std::string, choice_rollout_cache_path, "",
          "Optional validated cache for choice-only legal-action outcome rollouts.");
ABSL_FLAG(std::string, cache_dir, "", "Directory for multi-layer cache storage.");

ABSL_FLAG(bool, conservative_override_enabled, false, "Enforce conservative override selection protocol");
std::atomic<bool> g_cache_validation_failed{false};
ABSL_FLAG(double, conservative_covered_prior_threshold, 0.95, "Minimum covered prior mass to avoid fallback");
ABSL_FLAG(int, conservative_meaningful_visit_threshold, 10, "Minimum visits required for raw and argmax actions");
ABSL_FLAG(double, conservative_q_margin_threshold, 0.03, "Q margin threshold for MCTS to override raw");
ABSL_FLAG(double, conservative_stability_checkpoint_fraction, 0.5, "Fraction of budget at which to check stability");
ABSL_FLAG(bool, conservative_continuation_overrides_disabled, true, "Disable overrides during continuation decisions");


using namespace open_spiel;

struct CorpusState {
  std::string category;
  Player player;
  int round;
  std::vector<Action> history;
  std::vector<float> observation;
  int source_seed = -1;
  int episode_id = -1;
  int seat = -1;
  int decision_index = -1;
  std::string role = "";
  std::string history_hash = "";
  std::vector<Action> legal_actions;
  std::string corpus_schema_version = "";
};

// Mock Evaluator for self-testing without model checkpoint loading
class MockEvaluator : public algorithms::Evaluator {
 public:
  std::vector<double> Evaluate(const State& state) override {
    return std::vector<double>(state.NumPlayers(), 0.0);
  }

  ActionsAndProbs Prior(const State& state) override {
    ActionsAndProbs prior;
    std::vector<Action> legal = state.LegalActions();
    if (legal.empty()) return {};

    const char* env_one_hot = std::getenv("MOCK_ONE_HOT_PRIOR");
    if (env_one_hot && std::string(env_one_hot) == "1") {
      for (size_t i = 0; i < legal.size(); ++i) {
        prior.push_back({legal[i], (i == 0 ? 1.0 : 0.0)});
      }
      return prior;
    }

    double p = 1.0 / legal.size();
    for (Action a : legal) {
      prior.push_back({a, p});
    }
    return prior;
  }
};

// Tie-aware Spearman correlation calculation
double SpearmanCorrelation(const std::vector<double>& x, const std::vector<double>& y) {
  int n = x.size();
  if (n <= 1) return 0.0;

  std::vector<std::pair<double, int>> rx(n), ry(n);
  for (int i = 0; i < n; ++i) {
    rx[i] = {x[i], i};
    ry[i] = {y[i], i};
  }
  std::sort(rx.begin(), rx.end());
  std::sort(ry.begin(), ry.end());

  std::vector<double> ranks_x(n), ranks_y(n);
  for (int i = 0; i < n; ) {
    int j = i;
    while (j < n && rx[j].first == rx[i].first) j++;
    double rank = (i + 1 + j) / 2.0;
    for (int k = i; k < j; ++k) ranks_x[rx[k].second] = rank;
    i = j;
  }
  for (int i = 0; i < n; ) {
    int j = i;
    while (j < n && ry[j].first == ry[i].first) j++;
    double rank = (i + 1 + j) / 2.0;
    for (int k = i; k < j; ++k) ranks_y[ry[k].second] = rank;
    i = j;
  }

  double mean_rx = 0.0, mean_ry = 0.0;
  for (int i = 0; i < n; ++i) {
    mean_rx += ranks_x[i];
    mean_ry += ranks_y[i];
  }
  mean_rx /= n;
  mean_ry /= n;

  double num = 0.0, den_x = 0.0, den_y = 0.0;
  for (int i = 0; i < n; ++i) {
    double dx = ranks_x[i] - mean_rx;
    double dy = ranks_y[i] - mean_ry;
    num += dx * dy;
    den_x += dx * dx;
    den_y += dy * dy;
  }
  if (den_x == 0.0 || den_y == 0.0) return 0.0;
  return num / std::sqrt(den_x * den_y);
}

// Replay state history to reconstruct state
std::unique_ptr<State> ReconstructState(
    const std::shared_ptr<const Game>& game,
    const std::vector<Action>& history,
    Player player,
    const std::vector<float>& expected_obs,
    const std::vector<Action>& expected_legal = {}) {
  auto state = game->NewInitialState();
  for (size_t idx = 0; idx < history.size(); ++idx) {
    Action action = history[idx];
    if (state->IsChanceNode()) {
      auto outcomes = state->ChanceOutcomes();
      bool found = false;
      for (const auto& outcome : outcomes) {
        if (outcome.first == action) {
          found = true;
          break;
        }
      }
      if (!found) {
        std::cerr << "[WARNING] ReconstructState early stop: chance outcome ID " << action << " is illegal at index " << idx << "/" << history.size() << "\n" << std::flush;
        break;
      }
    } else {
      auto legal = state->LegalActions();
      if (std::find(legal.begin(), legal.end(), action) == legal.end()) {
        std::cerr << "[WARNING] ReconstructState early stop: player action ID " << action << " is illegal at index " << idx << "/" << history.size() << "\n" << std::flush;
        break;
      }
    }
    state->ApplyAction(action);
  }
  if (!expected_obs.empty()) {
    std::vector<float> rec_obs = state->InformationStateTensor(player);
    if (rec_obs.size() != expected_obs.size()) {
      std::cerr << "Reconstructed observation size mismatch: expected "
                << expected_obs.size() << ", got " << rec_obs.size() << "\n";
      std::exit(1);
    }
    for (size_t i = 0; i < rec_obs.size(); ++i) {
      if (std::abs(rec_obs[i] - expected_obs[i]) > 1e-4) {
        std::cerr << "Reconstructed observation value mismatch at dim " << i
                  << ": expected " << expected_obs[i] << ", got " << rec_obs[i] << "\n";
        std::exit(1);
      }
    }
  }
  if (!expected_legal.empty()) {
    std::vector<Action> rec_legal = state->LegalActions();
    if (rec_legal.size() != expected_legal.size()) {
      std::cerr << "Reconstructed legal actions size mismatch: expected "
                << expected_legal.size() << ", got " << rec_legal.size() << "\n";
      std::exit(1);
    }
    for (size_t i = 0; i < rec_legal.size(); ++i) {
      if (rec_legal[i] != expected_legal[i]) {
        std::cerr << "Reconstructed legal action value mismatch at index " << i
                  << ": expected " << expected_legal[i] << ", got " << rec_legal[i] << "\n";
        std::exit(1);
      }
    }
  }
  return state;
}

// Evaluate a single rollout using a generic algorithms::Evaluator pointer
double RunRawPolicyRollout(const State& start_state, Player owner, algorithms::Evaluator* evaluator, uint64_t seed, double utility_divisor) {
  std::mt19937 rng(seed);
  auto state = start_state.Clone();
  const auto* dune_state = dynamic_cast<const dune_imperium::DuneImperiumState*>(state.get());
  SPIEL_CHECK_TRUE(dune_state != nullptr);

  while (!state->IsTerminal()) {
    if (state->IsChanceNode()) {
      auto outcomes = state->ChanceOutcomes();
      Action choice = SampleAction(outcomes, absl::Uniform(rng, 0.0, 1.0)).first;
      state->ApplyAction(choice);
    } else if (state->CurrentPlayer() == kSimultaneousPlayerId) {
      std::vector<Action> joint_action;
      for (int p = 0; p < state->NumPlayers(); ++p) {
        auto actions = state->LegalActions(p);
        std::uniform_int_distribution<int> dis(0, actions.size() - 1);
        joint_action.push_back(actions[dis(rng)]);
      }
      state->ApplyActions(joint_action);
    } else {
      ActionsAndProbs prior = evaluator->Prior(*state);
      double r_num = absl::Uniform(rng, 0.0, 1.0);
      Action action = SampleActionFromPrior(prior, r_num);

      std::vector<Action> legal_acts = state->LegalActions();
      if (std::find(legal_acts.begin(), legal_acts.end(), action) == legal_acts.end()) {
        std::cerr << "[CRITICAL ERROR] RunRawPolicyRollout selected action " << action
                  << " (" << dune_state->ActionToString(state->CurrentPlayer(), action) << ") which is NOT legal!\n";
        std::cerr << "Current Player: " << state->CurrentPlayer() << ", Phase: " << static_cast<int>(dune_state->phase()) << "\n";
        std::cerr << "Legal actions:\n";
        for (Action a : legal_acts) {
          std::cerr << "  " << a << " (" << dune_state->ActionToString(state->CurrentPlayer(), a) << ")\n";
        }
        std::cerr << "Prior actions/probs:\n";
        for (const auto& ap : prior) {
          std::cerr << "  " << ap.first << " (" << dune_state->ActionToString(state->CurrentPlayer(), ap.first) << "): " << ap.second << "\n";
        }
        std::cerr << "State representation:\n" << state->ToString() << "\n";
        std::exit(1);
      }
      state->ApplyAction(action);
    }
  }
  return state->Returns()[owner] / utility_divisor;
}

struct ForcedActionRolloutResult {
  double return_val = 0.0;
  double successor_critic = 0.0;
};

// Evaluate a root action under a common stochastic raw-policy continuation.
// This isolates action quality from repeated, identical root searches.
ForcedActionRolloutResult RunForcedActionPolicyRollout(
    const State& start_state, Player owner, Action root_action,
    algorithms::Evaluator* evaluator, uint64_t chance_seed,
    uint64_t rollout_seed, double utility_divisor) {
  auto state = start_state.Clone();
  std::vector<Action> legal = state->LegalActions();
  SPIEL_CHECK_TRUE(std::find(legal.begin(), legal.end(), root_action) != legal.end());
  state->ApplyAction(root_action);

  std::mt19937 chance_rng(chance_seed);
  while (state->IsChanceNode()) {
    auto outcomes = state->ChanceOutcomes();
    if (outcomes.empty()) break;
    Action choice = SampleAction(outcomes, absl::Uniform(chance_rng, 0.0, 1.0)).first;
    state->ApplyAction(choice);
  }

  double critic = state->IsTerminal()
      ? state->Returns()[owner] / utility_divisor
      : evaluator->Evaluate(*state)[owner];
  double final_return = RunRawPolicyRollout(
      *state, owner, evaluator, rollout_seed, utility_divisor);
  return {final_return, critic};
}

std::string DuneDecisionRoleToString(DuneDecisionRole role) {
  switch (role) {
    case DuneDecisionRole::kForcedOrBookkeeping: return "FORCED_OR_BOOKKEEPING";
    case DuneDecisionRole::kLeaderSelection: return "LEADER_SELECTION";
    case DuneDecisionRole::kAgentPrimary: return "AGENT_PRIMARY";
    case DuneDecisionRole::kAgentContinuation: return "AGENT_CONTINUATION";
    case DuneDecisionRole::kPurchase: return "PURCHASE";
    case DuneDecisionRole::kCombatIntrigue: return "COMBAT_INTRIGUE";
    case DuneDecisionRole::kOtherOptional: return "OTHER_OPTIONAL";
    default: return "UNKNOWN";
  }
}

struct RolloutEvidence {
  int rollout_index = 0;
  std::string session_id = "";
  std::vector<int> raw_action_sequence;
  std::vector<int> search_action_sequence;
  std::vector<std::vector<int>> raw_legal_actions;
  std::vector<std::string> raw_decision_roles;
  std::vector<std::vector<int>> search_legal_actions;
  std::vector<std::string> search_decision_roles;
  std::vector<double> raw_priors;
  std::vector<int> search_visits;
  std::vector<double> root_q_values;
  int selected_action_rank = -1;
  double raw_return = 0.0;
  double search_return = 0.0;
  double paired_advantage = 0.0;

  int simulations_used = 0;
  int initial_simulations = 0;
  int continuation_simulations = 0;
  int short_window_simulations = 0;
  int re_root_hits = 0;
  int re_root_misses = 0;
  double search_depth = 0.0;
  double terminal_leaf_fraction = 0.0;
  int inference_count = 0;
  int inherited_visits = 0;
  std::vector<int> root_action_ids;
  bool primary_action_changed = false;
  bool continuation_action_changed = false;
  int search_decisions_count = 0;
};

struct OpportunityStateEvidence {
  int corpus_index = 0;
  int episode_id = -1;
  std::string history_hash = "";
  int seat = -1;
  int round = -1;
  std::string role = "";
  std::vector<int> raw_action_sequence;
  std::vector<int> search_action_sequence;
  std::vector<std::vector<int>> raw_legal_actions;
  std::vector<std::string> raw_decision_roles;
  std::vector<std::vector<int>> search_legal_actions;
  std::vector<std::string> search_decision_roles;
  std::vector<double> raw_priors;
  std::vector<int> search_visits;
  std::vector<double> root_q_values;
  int selected_action_rank = -1;
  std::vector<double> raw_returns;
  std::vector<double> search_returns;
  double mean_raw_return = 0.0;
  double mean_search_return = 0.0;
  double raw_return_std_err = 0.0;
  double search_return_std_err = 0.0;
  double mean_paired_advantage = 0.0;
  double paired_advantage_std_err = 0.0;
  double diagnostic_legacy_raw_return = 0.0;
  double diagnostic_legacy_paired_advantage = 0.0;
  int total_simulations_used = 0;
  int total_re_root_hits = 0;
  int total_re_root_misses = 0;
  double mean_search_depth = 0.0;
  double mean_terminal_leaf_fraction = 0.0;
  int total_inference_count = 0;
  bool primary_action_changed = false;
  bool continuation_action_changed = false;

  bool used_fallback = false;
  std::string fallback_reason = "";
  bool confidence_fallback = false;
  bool mcts_overrode_raw = false;

  Action raw_reference_action = kInvalidAction;
  Action mcts_proposed_action = kInvalidAction;
  Action selected_action = kInvalidAction;
  bool stability_checkpoint_reached = false;
  bool stability_agreement = false;
  bool pass_complete_search = false;
  bool pass_min_actions = false;
  bool pass_prior_mass = false;
  bool pass_meaningful_visits = false;
  bool pass_q_margin = false;
  bool pass_stability = false;

  // Persistence audit fields
  std::vector<int> root_action_ids;
  std::string session_id = "";
  int inherited_visits = 0;
  int initial_simulations = 0;
  int continuation_simulations = 0;

  std::vector<RolloutEvidence> rollouts;

  // Successor / leaf diagnostics
  std::vector<double> successor_critic_values;
  std::vector<double> successor_true_values;
  double choice_rank_correlation = 0.0;
  double successor_mean_bias = 0.0;
  double successor_rmse = 0.0;

  std::vector<double> leaf_critic_values;
  std::vector<double> leaf_true_values;
  double leaf_mean_bias = 0.0;
  double leaf_rmse = 0.0;
};

struct ChoiceOutcomeCacheEntry {
  int corpus_index = -1;
  std::vector<Action> actions;
  std::vector<std::vector<double>> returns;
  std::vector<double> successor_critic_values;
};

struct RolloutDiagnostics {
  double return_val = 0.0;
  std::vector<Action> action_sequence;
  std::vector<std::vector<Action>> legal_actions_sequence;
  std::vector<std::string> decision_roles_sequence;
  int session_simulations = 0;
  int initial_simulations = 0;
  int continuation_simulations = 0;
  int short_window_simulations = 0;
  int re_root_hits = 0;
  int re_root_misses = 0;
  double search_depth_sum = 0.0;
  double terminal_leaf_fraction_sum = 0.0;
  int search_decisions_count = 0;
  int inference_count = 0;
  bool primary_action_changed = false;
  bool continuation_action_changed = false;
  std::vector<double> raw_priors;
  std::vector<int> search_visits;
  std::vector<double> root_q_values;
  int selected_action_rank = -1;

  // Telemetry audit fields
  std::vector<Action> root_action_ids;
  std::string session_id = "";
  int inherited_visits = 0;
};

struct RawRolloutResult {
  double return_val = 0.0;
  std::vector<Action> action_sequence;
  std::vector<std::vector<Action>> legal_actions_sequence;
  std::vector<std::string> decision_roles_sequence;
};

RawRolloutResult RunRawControllerRollout(
    const State& start_state,
    Player owner,
    algorithms::Evaluator* evaluator,
    uint64_t seed,
    double utility_divisor) {
  std::mt19937 rng(seed);
  auto state = start_state.Clone();
  const auto* dune_state = dynamic_cast<const dune_imperium::DuneImperiumState*>(state.get());
  SPIEL_CHECK_TRUE(dune_state != nullptr);

  bool in_activation = true;
  Player active_player = state->CurrentPlayer();
  int round = dune_state->GetCurrentRound();
  std::vector<Action> action_sequence;
  std::vector<std::vector<Action>> legal_actions_sequence;
  std::vector<std::string> decision_roles_sequence;
  int raw_decisions_count = 0;

  while (!state->IsTerminal()) {
    if (state->IsChanceNode()) {
      auto outcomes = state->ChanceOutcomes();
      Action choice = SampleAction(outcomes, absl::Uniform(rng, 0.0, 1.0)).first;
      state->ApplyAction(choice);
    } else if (state->CurrentPlayer() == kSimultaneousPlayerId) {
      std::vector<Action> joint_action;
      for (int p = 0; p < state->NumPlayers(); ++p) {
        auto actions = state->LegalActions(p);
        std::uniform_int_distribution<int> dis(0, actions.size() - 1);
        joint_action.push_back(actions[dis(rng)]);
      }
      state->ApplyActions(joint_action);
    } else {
      Player current_player = state->CurrentPlayer();
      double r_val = absl::Uniform(rng, 0.0, 1.0);
      if (in_activation) {
        bool phase_changed = (dune_state->phase() != dune_imperium::GamePhase::kAgentTurns);
        if (phase_changed || (current_player >= 0 && current_player < 4 && current_player != owner) ||
            dune_state->GetCurrentRound() != round) {
          in_activation = false;
        }
      }

      ActionsAndProbs prior = evaluator->Prior(*state);
      Action action = kInvalidAction;
      if (in_activation && current_player == owner) {
        action = SampleActionFromPrior(prior, r_val);
        action_sequence.push_back(action);
        legal_actions_sequence.push_back(state->LegalActions());
        decision_roles_sequence.push_back(
            raw_decisions_count == 0 ? "AGENT_PRIMARY" : "AGENT_CONTINUATION");
        raw_decisions_count++;
      } else {
        action = SampleActionFromPrior(prior, r_val);
      }

      std::vector<Action> legal_acts = state->LegalActions();
      if (std::find(legal_acts.begin(), legal_acts.end(), action) == legal_acts.end()) {
        std::cerr << "[CRITICAL ERROR] RunRawControllerRollout selected action " << action
                  << " (" << dune_state->ActionToString(state->CurrentPlayer(), action) << ") which is NOT legal!\n";
        std::cerr << "Current Player: " << state->CurrentPlayer() << ", Phase: " << static_cast<int>(dune_state->phase()) << "\n";
        std::cerr << "Legal actions:\n";
        for (Action a : legal_acts) {
          std::cerr << "  " << a << " (" << dune_state->ActionToString(state->CurrentPlayer(), a) << ")\n";
        }
        std::cerr << "Prior actions/probs:\n";
        for (const auto& ap : prior) {
          std::cerr << "  " << ap.first << " (" << dune_state->ActionToString(state->CurrentPlayer(), ap.first) << "): " << ap.second << "\n";
        }
        std::cerr << "State representation:\n" << state->ToString() << "\n";
        std::exit(1);
      }
      state->ApplyAction(action);
    }
  }
  return {state->Returns()[owner] / utility_divisor, action_sequence, legal_actions_sequence, decision_roles_sequence};
}

Action GetBestActionFromSearchResult(const DuneSearchResult& result) {
  Action best_action = kInvalidAction;
  int total_visits = 0;
  for (int v : result.diagnostics.visit_counts) {
    total_visits += v;
  }
  if (total_visits > 0) {
    int max_visits = -1;
    for (size_t k = 0; k < result.diagnostics.actions.size(); ++k) {
      if (result.diagnostics.visit_counts[k] > max_visits) {
        max_visits = result.diagnostics.visit_counts[k];
        best_action = result.diagnostics.actions[k];
      }
    }
  }
  if (best_action == kInvalidAction && !result.diagnostics.priors.empty()) {
    double max_prob = -1.0;
    for (size_t k = 0; k < result.diagnostics.actions.size(); ++k) {
      if (result.diagnostics.priors[k] > max_prob) {
        max_prob = result.diagnostics.priors[k];
        best_action = result.diagnostics.actions[k];
      }
    }
  }
  if (best_action == kInvalidAction && !result.policy.empty()) {
    double max_prob = -1.0;
    for (const auto& ap : result.policy) {
      if (ap.second > max_prob) {
        max_prob = ap.second;
        best_action = ap.first;
      }
    }
  }
  return best_action;
}

RolloutDiagnostics RunSearchControllerRollout(
    const State& start_state,
    Player owner,
    std::shared_ptr<algorithms::Evaluator> evaluator,
    const DuneSearchConfig& base_cfg,
    uint64_t seed,
    double utility_divisor) {
  std::mt19937 rng(seed);
  auto state = start_state.Clone();
  const auto* dune_state = dynamic_cast<const dune_imperium::DuneImperiumState*>(state.get());
  SPIEL_CHECK_TRUE(dune_state != nullptr);

  DuneSearchConfig bot_cfg = base_cfg;
  // Keep base_cfg.seed which represents the search-seed
  DuneSearchBudgetMode mode = (bot_cfg.relative_time_budget_ms == std::numeric_limits<double>::infinity())
      ? DuneSearchBudgetMode::kFixedSessionSimulations
      : DuneSearchBudgetMode::kLiveDeadline;
  DuneSearchSession session(bot_cfg, evaluator, mode);

  bool in_activation = true;
  Player active_player = state->CurrentPlayer();
  int round = dune_state->GetCurrentRound();

  RolloutDiagnostics diag;

  while (!state->IsTerminal()) {
    if (state->IsChanceNode()) {
      auto outcomes = state->ChanceOutcomes();
      Action choice = SampleAction(outcomes, absl::Uniform(rng, 0.0, 1.0)).first;
      state->ApplyAction(choice);
    } else if (state->CurrentPlayer() == kSimultaneousPlayerId) {
      std::vector<Action> joint_action;
      for (int p = 0; p < state->NumPlayers(); ++p) {
        auto actions = state->LegalActions(p);
        std::uniform_int_distribution<int> dis(0, actions.size() - 1);
        joint_action.push_back(actions[dis(rng)]);
      }
      state->ApplyActions(joint_action);
    } else {
      Player current_player = state->CurrentPlayer();
      double r_val = absl::Uniform(rng, 0.0, 1.0);
      if (in_activation) {
        bool phase_changed = (dune_state->phase() != dune_imperium::GamePhase::kAgentTurns);
        if (phase_changed || (current_player >= 0 && current_player < 4 && current_player != owner) ||
            dune_state->GetCurrentRound() != round) {
          in_activation = false;
        }
      }

      Action action = kInvalidAction;
      if (in_activation && current_player == owner) {
        DuneDecisionRole role = ClassifyDuneDecisionRole(*state, owner, session.HasActiveSession());

        // Assert Task 3 gate invariants:
        if (diag.search_decisions_count == 0) {
          SPIEL_CHECK_TRUE(role == DuneDecisionRole::kAgentPrimary);
        }

        DuneSearchResult result = session.Search(*state);
        ControllerDecision decision = session.SelectControllerAction(*state, result, r_val);
        result = session.CommitAction(decision);

        if (role == DuneDecisionRole::kAgentPrimary) {
          SPIEL_CHECK_FALSE(result.diagnostics.session_id.empty());
        }

        action = result.diagnostics.selected_action;
        if (action == kInvalidAction) {
          action = GetBestActionFromSearchResult(result);
        }
        if (action != kInvalidAction) {
          std::vector<Action> legal_acts = state->LegalActions();
          SPIEL_CHECK_TRUE(std::find(legal_acts.begin(), legal_acts.end(), action) != legal_acts.end());
        }

        int reserve = bot_cfg.fixed_continuation_reserve;
        int limit = (bot_cfg.fixed_session_limit > 0) ? bot_cfg.fixed_session_limit : bot_cfg.max_simulations;
        if (role == DuneDecisionRole::kAgentPrimary) {
          SPIEL_CHECK_LE(result.simulations_completed, limit - reserve);
          diag.initial_simulations = result.simulations_completed;
        } else {
          diag.continuation_simulations += result.simulations_completed;
        }

        SPIEL_CHECK_LE(session.session_new_simulations_completed() + session.short_sims_completed(), limit);

        if (result.diagnostics.re_root_status == "hit") {
          diag.re_root_hits++;
        } else if (result.diagnostics.re_root_status == "miss") {
          diag.re_root_misses++;
        }
        diag.inherited_visits += result.diagnostics.inherited_root_visits;
        diag.search_depth_sum += result.diagnostics.mean_depth;
        diag.terminal_leaf_fraction_sum += result.diagnostics.terminal_leaf_fraction;

        // Action change flags
        Action raw_argmax = kInvalidAction;
        double max_p = -1.0;
        if (!result.diagnostics.priors.empty()) {
          for (size_t k = 0; k < result.diagnostics.actions.size(); ++k) {
            if (result.diagnostics.priors[k] > max_p) {
              max_p = result.diagnostics.priors[k];
              raw_argmax = result.diagnostics.actions[k];
            }
          }
        } else {
          for (const auto& ap : result.policy) {
            if (ap.second > max_p) {
              max_p = ap.second;
              raw_argmax = ap.first;
            }
          }
        }
        if (action != raw_argmax) {
          if (role == DuneDecisionRole::kAgentPrimary) {
            diag.primary_action_changed = true;
          } else {
            diag.continuation_action_changed = true;
          }
        }

        if (diag.search_decisions_count == 0) {
          diag.raw_priors = result.diagnostics.priors;
          diag.search_visits = result.diagnostics.visit_counts;
          diag.root_q_values = result.diagnostics.q_values;
          diag.session_id = result.diagnostics.session_id;
          diag.root_action_ids = result.diagnostics.actions;

          double selected_prob = 0.0;
          if (!result.diagnostics.priors.empty()) {
            for (size_t k = 0; k < result.diagnostics.actions.size(); ++k) {
              if (result.diagnostics.actions[k] == action) {
                selected_prob = result.diagnostics.priors[k];
                break;
              }
            }
            int rank = 1;
            for (double p : result.diagnostics.priors) {
              if (p > selected_prob) {
                rank++;
              }
            }
            diag.selected_action_rank = rank;
          } else {
            for (const auto& ap : result.policy) {
              if (ap.first == action) {
                selected_prob = ap.second;
                break;
              }
            }
            int rank = 1;
            for (const auto& ap : result.policy) {
              if (ap.second > selected_prob) {
                rank++;
              }
            }
            diag.selected_action_rank = rank;
          }
        }

        diag.legal_actions_sequence.push_back(state->LegalActions());
        diag.decision_roles_sequence.push_back(DuneDecisionRoleToString(role));

        diag.search_decisions_count++;
        diag.inference_count += result.inference_count;
        diag.action_sequence.push_back(action);
      } else {
        ActionsAndProbs prior = evaluator->Prior(*state);
        action = SampleActionFromPrior(prior, r_val);
      }

      std::vector<Action> legal_acts = state->LegalActions();
      if (std::find(legal_acts.begin(), legal_acts.end(), action) == legal_acts.end()) {
        std::cerr << "[CRITICAL ERROR] RunSearchControllerRollout selected action " << action
                  << " (" << dune_state->ActionToString(state->CurrentPlayer(), action) << ") which is NOT legal!\n";
        std::cerr << "Current Player: " << state->CurrentPlayer() << ", Phase: " << static_cast<int>(dune_state->phase()) << "\n";
        std::cerr << "Legal actions:\n";
        for (Action a : legal_acts) {
          std::cerr << "  " << a << " (" << dune_state->ActionToString(state->CurrentPlayer(), a) << ")\n";
        }
        std::cerr << "State representation:\n" << state->ToString() << "\n";
        std::exit(1);
      }
      state->ApplyAction(action);
    }
  }

  diag.session_simulations = session.session_new_simulations_completed() + session.short_sims_completed();
  diag.short_window_simulations = session.short_sims_completed();
  diag.return_val = state->Returns()[owner] / utility_divisor;
  return diag;
}

double CalculateStdErr(const std::vector<double>& values, double mean) {
  if (values.size() <= 1) return 0.0;
  double variance_sum = 0.0;
  for (double v : values) {
    variance_sum += (v - mean) * (v - mean);
  }
  double std_dev = std::sqrt(variance_sum / (values.size() - 1));
  return std_dev / std::sqrt(values.size());
}

std::string ComputePolicyFingerprint(const std::shared_ptr<SharedDunePolicyValueNetImpl>& model) {
  open_spiel::SHA256 hasher;

  auto params = model->named_parameters(true);
  std::vector<std::string> param_keys;
  for (const auto& item : params) {
    if (item.key().rfind("value_head", 0) != 0) {
      param_keys.push_back(item.key());
    }
  }
  std::sort(param_keys.begin(), param_keys.end());
  for (const auto& key : param_keys) {
    torch::Tensor t = params[key].to(torch::kCPU).contiguous();
    hasher.Update(reinterpret_cast<const uint8_t*>(t.data_ptr()), t.numel() * t.element_size());
  }

  auto buffers = model->named_buffers(true);
  std::vector<std::string> buffer_keys;
  for (const auto& item : buffers) {
    if (item.key().rfind("value_head", 0) != 0) {
      buffer_keys.push_back(item.key());
    }
  }
  std::sort(buffer_keys.begin(), buffer_keys.end());
  for (const auto& key : buffer_keys) {
    torch::Tensor t = buffers[key].to(torch::kCPU).contiguous();
    hasher.Update(reinterpret_cast<const uint8_t*>(t.data_ptr()), t.numel() * t.element_size());
  }

  return hasher.Final();
}

class SplitCacheManager {
 public:
  std::string cache_dir;
  std::string model_checkpoint;
  std::string corpus_path;
  int seed;
  int rollouts;
  double utility_divisor;
  bool self_test;
  std::string model_hash;
  std::string corpus_hash;
  std::string binary_hash;
  std::string policy_fingerprint;

  // Layer directories
  std::string identity_dir;
  std::string layer1_dir;
  std::string layer2_dir;
  std::string layer3_dir;

  // Configuration hashes
  std::string config_hash;
  open_spiel::json::Object search_config;
  open_spiel::json::Object controller_config;

  // Manifest fast-path lookup sets
  std::unordered_set<int64_t> completed_g1_set;
  std::unordered_set<int64_t> completed_g2_set;
  bool has_manifest = false;

  SplitCacheManager(const std::string& c_dir, const std::string& model_cp, const std::string& corp_path,
                    int sd, int rolls, double util_div, bool st, const std::string& p_fp)
      : cache_dir(c_dir), model_checkpoint(model_cp), corpus_path(corp_path),
        seed(sd), rollouts(rolls), utility_divisor(util_div), self_test(st), policy_fingerprint(p_fp) {
    if (cache_dir.empty()) return;

    if (self_test) {
      if (!model_checkpoint.empty() && std::filesystem::exists(model_checkpoint)) {
        model_hash = open_spiel::ComputeFileSHA256(model_checkpoint);
      } else {
        model_hash = "self_test";
      }
    } else {
      model_hash = open_spiel::ComputeFileSHA256(model_checkpoint);
    }
    corpus_hash = open_spiel::ComputeFileSHA256(corpus_path);
    binary_hash = open_spiel::ComputeFileSHA256("/proc/self/exe");

    // Compute prefixes (first 12 characters)
    std::string corpus_prefix = corpus_hash.substr(0, 12);
    std::string policy_prefix = policy_fingerprint.substr(0, 12);
    std::string checkpoint_prefix = model_hash.substr(0, 12);
    std::string binary_prefix = binary_hash.substr(0, 12);

    // Compute cache-identity directory preventing collisions
    identity_dir = absl::StrFormat("%s/%s_%s_session-v4",
        cache_dir, corpus_prefix, policy_prefix);

    // Compute architecture flags
    std::string val_head_type = absl::GetFlag(FLAGS_nonlinear_value_head) ? "nl" : "lin";
    std::string architecture_flags = absl::StrFormat("nb%d_hd%d_%s",
        absl::GetFlag(FLAGS_num_blocks), absl::GetFlag(FLAGS_hidden_dim), val_head_type);

    // Define Layer 1, 2 directories under identity_dir
    layer1_dir = absl::StrFormat("%s/layer1_%s_r%d_u%.6f_%s_rollout-v2_%s",
        identity_dir, corpus_prefix, rollouts, utility_divisor, policy_prefix, binary_prefix);
    layer2_dir = absl::StrFormat("%s/layer2_%s_%s_%s",
        identity_dir, corpus_prefix, checkpoint_prefix, architecture_flags);

    // Define Layer 3 settings
    search_config["opponent_mode"] = absl::GetFlag(FLAGS_opponent_mode);
    search_config["opponent_temperature"] = absl::GetFlag(FLAGS_opponent_temperature);
    search_config["puct_c"] = absl::GetFlag(FLAGS_puct_c);
    search_config["max_simulations"] = absl::GetFlag(FLAGS_max_simulations);
    search_config["fixed_continuation_reserve"] = absl::GetFlag(FLAGS_fixed_continuation_reserve);
    search_config["purchase_combat_budget"] = absl::GetFlag(FLAGS_purchase_combat_budget);
    search_config["live_continuation_reserve_seconds"] = absl::GetFlag(FLAGS_live_continuation_reserve_seconds);
    search_config["relative_time_budget_ms"] = absl::GetFlag(FLAGS_relative_time_budget_ms);
    search_config["nonlinear_value_head"] = absl::GetFlag(FLAGS_nonlinear_value_head);
    search_config["root_prior_temperature"] = absl::GetFlag(FLAGS_root_prior_temperature);

    controller_config["conservative_override_enabled"] = absl::GetFlag(FLAGS_conservative_override_enabled);
    controller_config["conservative_covered_prior_threshold"] = absl::GetFlag(FLAGS_conservative_covered_prior_threshold);
    controller_config["conservative_meaningful_visit_threshold"] = absl::GetFlag(FLAGS_conservative_meaningful_visit_threshold);
    controller_config["conservative_q_margin_threshold"] = absl::GetFlag(FLAGS_conservative_q_margin_threshold);
    controller_config["conservative_stability_checkpoint_fraction"] = absl::GetFlag(FLAGS_conservative_stability_checkpoint_fraction);
    controller_config["conservative_continuation_overrides_disabled"] = absl::GetFlag(FLAGS_conservative_continuation_overrides_disabled);

    std::string config_str = open_spiel::json::ToString(search_config, false) + ":" + open_spiel::json::ToString(controller_config, false);
    config_hash = open_spiel::ComputeStringSHA256(config_str);

    layer3_dir = absl::StrFormat("%s/layer3_%s_%s_s%d_session-v4_%s",
        identity_dir, corpus_prefix, checkpoint_prefix, seed, config_hash.substr(0, 12));

    std::filesystem::create_directories(layer1_dir);
    std::filesystem::create_directories(layer2_dir);
    std::filesystem::create_directories(layer3_dir);

    std::vector<int64_t> m_g1;
    std::vector<int64_t> m_g2;
    if (LoadManifest(m_g1, m_g2)) {
      completed_g1_set.insert(m_g1.begin(), m_g1.end());
      completed_g2_set.insert(m_g2.begin(), m_g2.end());
      has_manifest = true;
    }
  }

  bool IsActive() const { return !cache_dir.empty(); }

  bool LoadGate1ReturnsOnly(int state_idx, std::vector<double>& returns) {
    if (!IsActive()) return false;
    if (has_manifest && completed_g1_set.count(state_idx) == 0) return false;
    std::string shard1 = absl::StrFormat("%s/state_%d_r%d_u%.6f.json", layer1_dir, state_idx, rollouts, utility_divisor);
    if (!std::filesystem::exists(shard1)) return false;

    try {
      std::ifstream in1(shard1);
      std::string text1((std::istreambuf_iterator<char>(in1)), std::istreambuf_iterator<char>());
      auto parsed1 = open_spiel::json::FromString(text1);
      if (!parsed1) return false;
      auto obj1 = parsed1.value().GetObject();
      if (obj1.find("gate1_returns") == obj1.end()) return false;
      auto prov1 = obj1.at("provenance").GetObject();
      if (prov1.at("corpus_fingerprint").GetString() != corpus_hash ||
          prov1.at("policy_fingerprint").GetString() != policy_fingerprint ||
          prov1.at("rollout_count").GetInt() != rollouts ||
          prov1.at("utility_divisor").GetDouble() != utility_divisor ||
          prov1.at("binary_hash").GetString() != binary_hash ||
          prov1.at("rollout_protocol_version").GetString() != "rollout-v2") {
        return false;
      }
      auto arr1 = obj1.at("gate1_returns").GetArray();
      returns.clear();
      for (const auto& v : arr1) returns.push_back(v.GetDouble());
      return true;
    } catch (...) {
      return false;
    }
  }

  bool LoadGate1CriticOnly(int state_idx, double& critic_val) {
    if (!IsActive()) return false;
    if (has_manifest && completed_g1_set.count(state_idx) == 0) return false;
    std::string shard2 = absl::StrFormat("%s/state_%d.json", layer2_dir, state_idx);
    if (!std::filesystem::exists(shard2)) return false;

    try {
      std::ifstream in2(shard2);
      std::string text2((std::istreambuf_iterator<char>(in2)), std::istreambuf_iterator<char>());
      auto parsed2 = open_spiel::json::FromString(text2);
      if (!parsed2) return false;
      auto obj2 = parsed2.value().GetObject();
      auto prov2 = obj2.at("provenance").GetObject();
      std::string val_head_type = absl::GetFlag(FLAGS_nonlinear_value_head) ? "nl" : "lin";
      std::string expected_arch = absl::StrFormat("nb%d_hd%d_%s",
          absl::GetFlag(FLAGS_num_blocks), absl::GetFlag(FLAGS_hidden_dim), val_head_type);
      if (prov2.at("corpus_fingerprint").GetString() != corpus_hash ||
          prov2.at("full_checkpoint_hash").GetString() != model_hash ||
          prov2.find("architecture_flags") == prov2.end() ||
          prov2.at("architecture_flags").GetString() != expected_arch) {
        return false;
      }
      critic_val = obj2.at("gate1_critic_value").GetDouble();
      return true;
    } catch (...) {
      return false;
    }
  }

  // Check if a state is fully cached for Gate 1
  bool LoadGate1Shard(int state_idx, double& critic_val, std::vector<double>& returns) {
    if (!IsActive()) return false;
    if (has_manifest && completed_g1_set.count(state_idx) == 0) return false;
    std::string shard1 = absl::StrFormat("%s/state_%d_r%d_u%.6f.json", layer1_dir, state_idx, rollouts, utility_divisor);
    std::string shard2 = absl::StrFormat("%s/state_%d.json", layer2_dir, state_idx);
    if (!std::filesystem::exists(shard1) || !std::filesystem::exists(shard2)) return false;

    try {
      // Load Layer 1
      std::ifstream in1(shard1);
      std::string text1((std::istreambuf_iterator<char>(in1)), std::istreambuf_iterator<char>());
      auto parsed1 = open_spiel::json::FromString(text1);
      if (!parsed1) return false;
      auto obj1 = parsed1.value().GetObject();
      auto prov1 = obj1.at("provenance").GetObject();
      if (prov1.at("corpus_fingerprint").GetString() != corpus_hash ||
          prov1.at("policy_fingerprint").GetString() != policy_fingerprint ||
          prov1.at("rollout_count").GetInt() != rollouts ||
          prov1.at("utility_divisor").GetDouble() != utility_divisor ||
          prov1.at("binary_hash").GetString() != binary_hash ||
          prov1.at("rollout_protocol_version").GetString() != "rollout-v2") {
        return false;
      }
      auto arr1 = obj1.at("gate1_returns").GetArray();
      returns.clear();
      for (const auto& v : arr1) returns.push_back(v.GetDouble());

      // Load Layer 2
      std::ifstream in2(shard2);
      std::string text2((std::istreambuf_iterator<char>(in2)), std::istreambuf_iterator<char>());
      auto parsed2 = open_spiel::json::FromString(text2);
      if (!parsed2) return false;
      auto obj2 = parsed2.value().GetObject();
      auto prov2 = obj2.at("provenance").GetObject();
      std::string val_head_type = absl::GetFlag(FLAGS_nonlinear_value_head) ? "nl" : "lin";
      std::string expected_arch = absl::StrFormat("nb%d_hd%d_%s",
          absl::GetFlag(FLAGS_num_blocks), absl::GetFlag(FLAGS_hidden_dim), val_head_type);
      if (prov2.at("corpus_fingerprint").GetString() != corpus_hash ||
          prov2.at("full_checkpoint_hash").GetString() != model_hash ||
          prov2.find("architecture_flags") == prov2.end() ||
          prov2.at("architecture_flags").GetString() != expected_arch) {
        return false;
      }
      critic_val = obj2.at("gate1_critic_value").GetDouble();
      return true;
    } catch (...) {
      return false;
    }
  }

  void SaveGate1Shard(int state_idx, double critic_val, const std::vector<double>& returns, bool migrated_from_legacy = false, bool save_layer1 = true) {
    if (!IsActive()) return;

    // Acquire state-level lock
    std::string lock_path = absl::StrFormat("%s/state_%d.lock", identity_dir, state_idx);
    int lock_fd = open(lock_path.c_str(), O_CREAT | O_RDWR, 0666);
    if (lock_fd >= 0) {
      flock(lock_fd, LOCK_EX);
    }

    if (save_layer1) {
      // Save Layer 1 Shard
      open_spiel::json::Object outcomes1;
      bool existing_migrated = false;
      std::string shard1 = absl::StrFormat("%s/state_%d_r%d_u%.6f.json", layer1_dir, state_idx, rollouts, utility_divisor);
      if (std::filesystem::exists(shard1)) {
        try {
          std::ifstream in1(shard1);
          std::string text1((std::istreambuf_iterator<char>(in1)), std::istreambuf_iterator<char>());
          auto parsed1 = open_spiel::json::FromString(text1);
          if (parsed1) {
            auto parsed_obj1 = parsed1.value().GetObject();
            if (parsed_obj1.find("gate2_action_returns") != parsed_obj1.end()) {
              outcomes1 = parsed_obj1.at("gate2_action_returns").GetObject();
            }
            if (parsed_obj1.find("provenance") != parsed_obj1.end()) {
              auto ext_prov = parsed_obj1.at("provenance").GetObject();
              if (ext_prov.find("migrated_from_legacy") != ext_prov.end()) {
                existing_migrated = ext_prov.at("migrated_from_legacy").GetBool();
              }
            }
          }
        } catch (...) {}
      }

      open_spiel::json::Object obj1;
      open_spiel::json::Object prov1;
      prov1["corpus_fingerprint"] = corpus_hash;
      prov1["rollout_count"] = rollouts;
      prov1["utility_divisor"] = utility_divisor;
      prov1["policy_fingerprint"] = policy_fingerprint;
      prov1["binary_hash"] = binary_hash;
      prov1["rollout_protocol_version"] = "rollout-v2";
      prov1["corpus_index"] = state_idx;
      if (migrated_from_legacy || existing_migrated) {
        prov1["migrated_from_legacy"] = true;
      }
      obj1["provenance"] = prov1;

      open_spiel::json::Array arr1;
      for (double r : returns) arr1.push_back(r);
      obj1["gate1_returns"] = arr1;
      if (!outcomes1.empty()) {
        obj1["gate2_action_returns"] = outcomes1;
      }

      std::string tmp1 = absl::StrFormat("%s/state_%d_r%d_u%.6f.json.%d.tmp", layer1_dir, state_idx, rollouts, utility_divisor, getpid());
      std::ofstream out1(tmp1);
      out1 << open_spiel::json::ToString(obj1, true) << "\n";
      out1.close();
      std::filesystem::rename(tmp1, shard1);
    }

    // Save Layer 2 Shard
    open_spiel::json::Object obj2;
    open_spiel::json::Object prov2;
    prov2["corpus_fingerprint"] = corpus_hash;
    prov2["full_checkpoint_hash"] = model_hash;
    std::string val_head_type = absl::GetFlag(FLAGS_nonlinear_value_head) ? "nl" : "lin";
    prov2["architecture_flags"] = absl::StrFormat("nb%d_hd%d_%s",
        absl::GetFlag(FLAGS_num_blocks), absl::GetFlag(FLAGS_hidden_dim), val_head_type);
    obj2["provenance"] = prov2;
    obj2["gate1_critic_value"] = critic_val;

    std::string shard2 = absl::StrFormat("%s/state_%d.json", layer2_dir, state_idx);
    std::string tmp2 = absl::StrFormat("%s/state_%d.json.%d.tmp", layer2_dir, state_idx, getpid());
    std::ofstream out2(tmp2);
    out2 << open_spiel::json::ToString(obj2, true) << "\n";
    out2.close();
    std::filesystem::rename(tmp2, shard2);

    // Release state-level lock
    if (lock_fd >= 0) {
      flock(lock_fd, LOCK_UN);
      close(lock_fd);
    }
  }

  bool LoadGate2SearchShard(int state_idx, open_spiel::json::Object& search_result) {
    if (!IsActive()) return false;
    if (has_manifest && completed_g2_set.count(state_idx) == 0) return false;
    std::string shard3 = absl::StrFormat("%s/state_%d.json", layer3_dir, state_idx);
    if (!std::filesystem::exists(shard3)) return false;

    try {
      std::ifstream in3(shard3);
      std::string text3((std::istreambuf_iterator<char>(in3)), std::istreambuf_iterator<char>());
      if (text3.find("NaN") != std::string::npos ||
          text3.find("Infinity") != std::string::npos ||
          text3.find("-Infinity") != std::string::npos) {
        std::cerr << "[ERROR] Cache validation failed: non-finite value found in cache file " << shard3 << std::endl;
        g_cache_validation_failed = true;
        return false;
      }
      auto parsed3 = open_spiel::json::FromString(text3);
      if (!parsed3) {
        std::cerr << "[ERROR] Cache search shard parsing failed for " << shard3 << std::endl;
        g_cache_validation_failed = true;
        return false;
      }
      auto obj3 = parsed3.value().GetObject();
      auto prov3 = obj3.at("provenance").GetObject();
      if (prov3.at("corpus_fingerprint").GetString() != corpus_hash ||
          prov3.at("checkpoint_hash").GetString() != model_hash ||
          prov3.at("search_seed").GetInt() != seed ||
          prov3.at("search_protocol_version").GetString() != "session-v4" ||
          prov3.at("config_hash").GetString() != config_hash) {
        return false;
      }
      search_result = obj3.at("search_result").GetObject();
      return true;
    } catch (const std::exception& e) {
      std::cerr << "[ERROR] Exception loading search shard: " << e.what() << std::endl;
      g_cache_validation_failed = true;
      return false;
    } catch (...) {
      std::cerr << "[ERROR] Unknown exception loading search shard" << std::endl;
      g_cache_validation_failed = true;
      return false;
    }
  }

  void SaveGate2SearchShard(int state_idx, const open_spiel::json::Object& search_result) {
    if (!IsActive()) return;

    open_spiel::json::Object obj3;
    open_spiel::json::Object prov3;
    prov3["corpus_fingerprint"] = corpus_hash;
    prov3["checkpoint_hash"] = model_hash;
    prov3["search_seed"] = seed;
    prov3["search_protocol_version"] = "session-v4";
    prov3["config_hash"] = config_hash;
    prov3["search_config"] = search_config;
    prov3["controller_config"] = controller_config;
    obj3["provenance"] = prov3;
    obj3["search_result"] = search_result;

    std::string shard3 = absl::StrFormat("%s/state_%d.json", layer3_dir, state_idx);
    std::string tmp3 = absl::StrFormat("%s/state_%d.json.%d.tmp", layer3_dir, state_idx, getpid());
    std::ofstream out3(tmp3);
    out3 << open_spiel::json::ToString(obj3, true) << "\n";
    out3.close();
    std::filesystem::rename(tmp3, shard3);
  }

  bool LoadGate2OutcomesShard(int state_idx, const std::vector<Action>& actions,
                               std::vector<std::vector<double>>& returns,
                               std::vector<double>& successor_critic_values) {
    if (!IsActive()) return false;
    if (has_manifest && completed_g2_set.count(state_idx) == 0) return false;
    std::string shard1 = absl::StrFormat("%s/state_%d_r%d_u%.6f.json", layer1_dir, state_idx, rollouts, utility_divisor);
    std::string shard2 = absl::StrFormat("%s/state_%d_r%d_u%.6f.json", layer2_dir, state_idx, rollouts, utility_divisor);
    if (!std::filesystem::exists(shard1) || !std::filesystem::exists(shard2)) return false;

    try {
      // Load Layer 1 Outcomes
      std::ifstream in1(shard1);
      std::string text1((std::istreambuf_iterator<char>(in1)), std::istreambuf_iterator<char>());
      auto parsed1 = open_spiel::json::FromString(text1);
      if (!parsed1) return false;
      auto obj1 = parsed1.value().GetObject();
      auto prov1 = obj1.at("provenance").GetObject();
      if (prov1.at("corpus_fingerprint").GetString() != corpus_hash ||
          prov1.at("policy_fingerprint").GetString() != policy_fingerprint ||
          prov1.at("rollout_count").GetInt() != rollouts ||
          prov1.at("utility_divisor").GetDouble() != utility_divisor ||
          prov1.at("binary_hash").GetString() != binary_hash ||
          prov1.at("rollout_protocol_version").GetString() != "rollout-v2") {
        return false;
      }
      auto outcomes1 = obj1.at("gate2_action_returns").GetObject();

      // Load Layer 2 Outcomes
      std::ifstream in2(shard2);
      std::string text2((std::istreambuf_iterator<char>(in2)), std::istreambuf_iterator<char>());
      auto parsed2 = open_spiel::json::FromString(text2);
      if (!parsed2) return false;
      auto obj2 = parsed2.value().GetObject();
      auto prov2 = obj2.at("provenance").GetObject();
      std::string val_head_type = absl::GetFlag(FLAGS_nonlinear_value_head) ? "nl" : "lin";
      std::string expected_arch = absl::StrFormat("nb%d_hd%d_%s",
          absl::GetFlag(FLAGS_num_blocks), absl::GetFlag(FLAGS_hidden_dim), val_head_type);
      if (prov2.at("corpus_fingerprint").GetString() != corpus_hash ||
          prov2.at("full_checkpoint_hash").GetString() != model_hash ||
          prov2.find("architecture_flags") == prov2.end() ||
          prov2.at("architecture_flags").GetString() != expected_arch ||
          prov2.find("rollout_count") == prov2.end() ||
          prov2.at("rollout_count").GetInt() != rollouts ||
          prov2.find("utility_divisor") == prov2.end() ||
          prov2.at("utility_divisor").GetDouble() != utility_divisor) {
        return false;
      }
      auto outcomes2 = obj2.at("gate2_successor_critic_values").GetObject();

      // Verify that all actions exist in cached shards
      std::vector<std::vector<double>> temp_returns;
      std::vector<double> temp_critics;
      for (Action a : actions) {
        std::string act_str = std::to_string(a);
        if (outcomes1.find(act_str) == outcomes1.end() ||
            outcomes2.find(act_str) == outcomes2.end()) {
          return false;
        }
        auto ret_arr = outcomes1.at(act_str).GetArray();
        if (ret_arr.size() != static_cast<size_t>(rollouts)) return false;
        std::vector<double> act_ret;
        for (const auto& v : ret_arr) act_ret.push_back(v.GetDouble());
        temp_returns.push_back(act_ret);
        temp_critics.push_back(outcomes2.at(act_str).GetDouble());
      }

      returns = temp_returns;
      successor_critic_values = temp_critics;
      return true;
    } catch (...) {
      return false;
    }
  }

  void SaveGate2OutcomesShard(int state_idx, const std::vector<Action>& actions,
                               const std::vector<std::vector<double>>& returns,
                               const std::vector<double>& successor_critic_values) {
    if (!IsActive()) return;

    // Acquire state-level lock
    std::string lock_path = absl::StrFormat("%s/state_%d.lock", identity_dir, state_idx);
    int lock_fd = open(lock_path.c_str(), O_CREAT | O_RDWR, 0666);
    if (lock_fd >= 0) {
      flock(lock_fd, LOCK_EX);
    }

    // Load existing Layer 1 shard if it exists, to preserve other actions and gate1_returns
    open_spiel::json::Object outcomes1;
    open_spiel::json::Array gate1_ret;
    bool has_gate1_ret = false;
    bool existing_migrated = false;
    std::string shard1 = absl::StrFormat("%s/state_%d_r%d_u%.6f.json", layer1_dir, state_idx, rollouts, utility_divisor);
    if (std::filesystem::exists(shard1)) {
      try {
        std::ifstream in1(shard1);
        std::string text1((std::istreambuf_iterator<char>(in1)), std::istreambuf_iterator<char>());
        auto parsed1 = open_spiel::json::FromString(text1);
        if (parsed1) {
          auto parsed_obj1 = parsed1.value().GetObject();
          if (parsed_obj1.find("gate2_action_returns") != parsed_obj1.end()) {
            outcomes1 = parsed_obj1.at("gate2_action_returns").GetObject();
          }
          if (parsed_obj1.find("gate1_returns") != parsed_obj1.end()) {
            gate1_ret = parsed_obj1.at("gate1_returns").GetArray();
            has_gate1_ret = true;
          }
          if (parsed_obj1.find("provenance") != parsed_obj1.end()) {
            auto ext_prov = parsed_obj1.at("provenance").GetObject();
            if (ext_prov.find("migrated_from_legacy") != ext_prov.end()) {
              existing_migrated = ext_prov.at("migrated_from_legacy").GetBool();
            }
          }
        }
      } catch (...) {}
    }
    // Update outcomes1
    for (size_t a_idx = 0; a_idx < actions.size(); ++a_idx) {
      open_spiel::json::Array arr;
      for (double r : returns[a_idx]) arr.push_back(r);
      outcomes1[std::to_string(actions[a_idx])] = arr;
    }

    open_spiel::json::Object obj1;
    open_spiel::json::Object prov1;
    prov1["corpus_fingerprint"] = corpus_hash;
    prov1["rollout_count"] = rollouts;
    prov1["utility_divisor"] = utility_divisor;
    prov1["policy_fingerprint"] = policy_fingerprint;
    prov1["binary_hash"] = binary_hash;
    prov1["rollout_protocol_version"] = "rollout-v2";
    prov1["corpus_index"] = state_idx;
    if (existing_migrated) {
      prov1["migrated_from_legacy"] = true;
    }
    obj1["provenance"] = prov1;
    obj1["gate2_action_returns"] = outcomes1;
    if (has_gate1_ret) {
      obj1["gate1_returns"] = gate1_ret;
    }

    std::string tmp1 = absl::StrFormat("%s/state_%d_r%d_u%.6f.json.%d.tmp", layer1_dir, state_idx, rollouts, utility_divisor, getpid());
    std::ofstream out1(tmp1);
    out1 << open_spiel::json::ToString(obj1, true) << "\n";
    out1.close();
    std::filesystem::rename(tmp1, shard1);

    // Load existing Layer 2 shard if it exists, to preserve other actions
    open_spiel::json::Object outcomes2;
    std::string shard2 = absl::StrFormat("%s/state_%d_r%d_u%.6f.json", layer2_dir, state_idx, rollouts, utility_divisor);
    if (std::filesystem::exists(shard2)) {
      try {
        std::ifstream in2(shard2);
        std::string text2((std::istreambuf_iterator<char>(in2)), std::istreambuf_iterator<char>());
        auto parsed2 = open_spiel::json::FromString(text2);
        if (parsed2) {
          auto obj2 = parsed2.value().GetObject();
          if (obj2.find("gate2_successor_critic_values") != obj2.end()) {
            outcomes2 = obj2.at("gate2_successor_critic_values").GetObject();
          }
        }
      } catch (...) {}
    }
    // Update outcomes2
    for (size_t a_idx = 0; a_idx < actions.size(); ++a_idx) {
      outcomes2[std::to_string(actions[a_idx])] = successor_critic_values[a_idx];
    }

    open_spiel::json::Object obj2;
    open_spiel::json::Object prov2;
    prov2["corpus_fingerprint"] = corpus_hash;
    prov2["full_checkpoint_hash"] = model_hash;
    prov2["rollout_count"] = rollouts;
    prov2["utility_divisor"] = utility_divisor;
    std::string val_head_type = absl::GetFlag(FLAGS_nonlinear_value_head) ? "nl" : "lin";
    prov2["architecture_flags"] = absl::StrFormat("nb%d_hd%d_%s",
        absl::GetFlag(FLAGS_num_blocks), absl::GetFlag(FLAGS_hidden_dim), val_head_type);
    obj2["provenance"] = prov2;
    obj2["gate2_successor_critic_values"] = outcomes2;
    std::string tmp2 = absl::StrFormat("%s/state_%d_r%d_u%.6f.json.%d.tmp", layer2_dir, state_idx, rollouts, utility_divisor, getpid());
    std::ofstream out2(tmp2);
    out2 << open_spiel::json::ToString(obj2, true) << "\n";
    out2.close();
    std::filesystem::rename(tmp2, shard2);

    // Release state-level lock
    if (lock_fd >= 0) {
      flock(lock_fd, LOCK_UN);
      close(lock_fd);
    }
  }



  void WriteManifest(const std::vector<size_t>& gate1_indices, const std::vector<size_t>& gate2_indices) {
    if (!IsActive()) return;
    open_spiel::json::Object obj;
    open_spiel::json::Object prov;
    prov["corpus_fingerprint"] = corpus_hash;
    prov["full_checkpoint_hash"] = model_hash;
    prov["policy_fingerprint"] = policy_fingerprint;
    prov["rollout_count"] = rollouts;
    prov["utility_divisor"] = utility_divisor;
    prov["seed"] = seed;
    prov["search_protocol_version"] = "session-v4";
    prov["config_hash"] = config_hash;
    obj["provenance"] = prov;

    open_spiel::json::Array g1_arr;
    for (size_t idx : gate1_indices) g1_arr.push_back(static_cast<int64_t>(idx));
    obj["gate1_completed_indices"] = g1_arr;

    open_spiel::json::Array g2_arr;
    for (size_t idx : gate2_indices) g2_arr.push_back(static_cast<int64_t>(idx));
    obj["gate2_completed_indices"] = g2_arr;

    std::string manifest_path = absl::StrFormat("%s/manifest.json", identity_dir);
    std::string tmp_manifest = absl::StrFormat("%s/manifest.json.%d.tmp", identity_dir, getpid());
    std::ofstream out(tmp_manifest);
    out << open_spiel::json::ToString(obj, true) << "\n";
    out.close();
    std::filesystem::rename(tmp_manifest, manifest_path);
  }

  bool LoadManifest(std::vector<int64_t>& gate1_completed, std::vector<int64_t>& gate2_completed) {
    if (!IsActive()) return false;
    std::string manifest_path = absl::StrFormat("%s/manifest.json", identity_dir);
    if (!std::filesystem::exists(manifest_path)) return false;

    try {
      std::ifstream in(manifest_path);
      std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
      auto parsed = open_spiel::json::FromString(text);
      if (!parsed) return false;
      auto obj = parsed.value().GetObject();
      auto prov = obj.at("provenance").GetObject();
      if (prov.at("corpus_fingerprint").GetString() != corpus_hash ||
          prov.at("full_checkpoint_hash").GetString() != model_hash ||
          prov.at("policy_fingerprint").GetString() != policy_fingerprint ||
          prov.at("rollout_count").GetInt() != rollouts ||
          prov.at("utility_divisor").GetDouble() != utility_divisor ||
          prov.at("seed").GetInt() != seed ||
          prov.at("search_protocol_version").GetString() != "session-v4" ||
          prov.at("config_hash").GetString() != config_hash) {
        return false;
      }
      auto g1_arr = obj.at("gate1_completed_indices").GetArray();
      for (const auto& v : g1_arr) gate1_completed.push_back(v.GetInt());
      auto g2_arr = obj.at("gate2_completed_indices").GetArray();
      for (const auto& v : g2_arr) gate2_completed.push_back(v.GetInt());
      return true;
    } catch (...) {
      return false;
    }
  }
};

int main(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);
  at::set_num_threads(1);

  std::string model_checkpoint = absl::GetFlag(FLAGS_model_checkpoint);
  std::string corpus_path = absl::GetFlag(FLAGS_corpus_path);
  int seed = absl::GetFlag(FLAGS_seed);
  bool has_search_seed = false;
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if ((arg.size() >= 14 && arg.substr(0, 14) == "--search_seed=") || arg == "--search_seed") {
      has_search_seed = true;
      break;
    }
  }
  int search_seed = absl::GetFlag(FLAGS_search_seed);
  if (!has_search_seed && seed != 42) {
    search_seed = seed;
  }
  int raw_policy_seed = absl::GetFlag(FLAGS_raw_policy_seed);
  int rollout_seed_base = absl::GetFlag(FLAGS_rollout_seed_base);
  int chance_seed_base = absl::GetFlag(FLAGS_chance_seed_base);
  int bootstrap_seed = absl::GetFlag(FLAGS_bootstrap_seed);
  int num_threads = absl::GetFlag(FLAGS_threads);
  double utility_divisor = absl::GetFlag(FLAGS_utility_divisor);
  bool mock_fail = absl::GetFlag(FLAGS_mock_miscalibrated_critic);
  bool self_test = absl::GetFlag(FLAGS_self_test);

  auto game = open_spiel::LoadGame("dune_imperium");
  int64_t obs_size = game->GetType().provides_information_state_tensor
                         ? game->InformationStateTensorSize()
                         : game->ObservationTensorSize();
  int64_t action_size = game->NumDistinctActions();

  std::shared_ptr<algorithms::Evaluator> global_evaluator;
  std::shared_ptr<open_spiel::BatchedEvaluator> batched_eval;
  std::shared_mutex model_mutex;

  std::shared_ptr<SharedDunePolicyValueNetImpl> model;
  if (self_test) {
    std::cout << "[Self Test] Bypassing model loading, using MockEvaluator\n";
    global_evaluator = std::make_shared<MockEvaluator>();
  } else {
    torch::Device device = torch::cuda::is_available() ? torch::Device(torch::kCUDA) : torch::Device(torch::kCPU);
    std::cout << "Using device: " << (device.is_cuda() ? "CUDA" : "CPU") << "\n";
    model = std::make_shared<SharedDunePolicyValueNetImpl>(
        obs_size, absl::GetFlag(FLAGS_hidden_dim), action_size, absl::GetFlag(FLAGS_num_blocks),
        absl::GetFlag(FLAGS_nonlinear_value_head));
    try {
      torch::load(model, model_checkpoint, device);
    } catch (const std::exception& e) {
      std::cerr << "Failed to load model: " << e.what() << "\n";
      return 1;
    }
    model->to(device);
    model->eval();

    batched_eval = std::make_shared<open_spiel::BatchedEvaluator>(
        model, num_threads, /*timeout_ms=*/1, device, &model_mutex,
        /*logit_cap=*/0.0f, /*device_synchronize=*/false);
    global_evaluator = std::make_shared<open_spiel::BatchedNNEvaluator>(batched_eval);
  }

  std::string policy_fp = "";
  if (!self_test && model) {
    policy_fp = ComputePolicyFingerprint(model);
    std::cout << "Computed Policy Fingerprint: " << policy_fp << "\n";
  } else {
    policy_fp = "self_test_policy_fp";
  }

  std::string cache_dir = absl::GetFlag(FLAGS_cache_dir);
  SplitCacheManager split_cache(cache_dir, model_checkpoint, corpus_path, search_seed,
                                self_test ? 2 : absl::GetFlag(FLAGS_rollouts),
                                utility_divisor, self_test, policy_fp);

  if (split_cache.IsActive()) {
    std::vector<int64_t> m_g1;
    std::vector<int64_t> m_g2;
    if (split_cache.LoadManifest(m_g1, m_g2)) {
      std::cout << "[Manifest] Verified complete cache manifest on startup.\n" << std::flush;
    }
  }

  std::ifstream f(corpus_path);
  if (!f) {
    std::cerr << "Failed to open corpus path: " << corpus_path << "\n";
    return 1;
  }

  std::string content((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
  auto json_parsed = open_spiel::json::FromString(content);
  if (!json_parsed) {
    std::cerr << "Failed to parse corpus JSON.\n";
    return 1;
  }

  auto json_arr = json_parsed.value().GetArray();
  std::vector<CorpusState> corpus;
  for (const auto& item_val : json_arr) {
    auto obj = item_val.GetObject();
    CorpusState cs;
    cs.category = obj.at("category").GetString();
    cs.player = static_cast<Player>(obj.at("player").GetInt());
    cs.round = obj.at("round").GetInt();

    auto hist_arr = obj.at("history").GetArray();
    for (const auto& a_val : hist_arr) {
      cs.history.push_back(static_cast<Action>(a_val.GetInt()));
    }

    auto obs_arr = obj.at("observation").GetArray();
    for (const auto& o_val : obs_arr) {
      cs.observation.push_back(static_cast<float>(o_val.GetDouble()));
    }
    if (obj.find("source_seed") != obj.end()) cs.source_seed = obj.at("source_seed").GetInt();
    if (obj.find("episode_id") != obj.end()) cs.episode_id = obj.at("episode_id").GetInt();
    if (obj.find("seat") != obj.end()) cs.seat = obj.at("seat").GetInt();
    if (obj.find("decision_index") != obj.end()) cs.decision_index = obj.at("decision_index").GetInt();
    if (obj.find("role") != obj.end()) cs.role = obj.at("role").GetString();
    if (obj.find("history_hash") != obj.end()) cs.history_hash = obj.at("history_hash").GetString();
    if (obj.find("corpus_schema_version") != obj.end()) {
      cs.corpus_schema_version = obj.at("corpus_schema_version").GetString();
    } else if (obj.find("schema_version") != obj.end()) {
      cs.corpus_schema_version = obj.at("schema_version").GetString();
    }
    if (obj.find("legal_actions") != obj.end()) {
      auto legal_arr = obj.at("legal_actions").GetArray();
      for (const auto& l_val : legal_arr) {
        cs.legal_actions.push_back(static_cast<Action>(l_val.GetInt()));
      }
    }
    corpus.push_back(cs);
  }

  std::cout << absl::StrFormat("Loaded diagnostic corpus of %d states.\n", corpus.size());

  std::vector<size_t> strategic_indices;
  std::vector<size_t> opportunity_indices;
  std::vector<size_t> planner_indices;

  for (size_t i = 0; i < corpus.size(); ++i) {
    auto state = ReconstructState(game, corpus[i].history, corpus[i].player, corpus[i].observation, corpus[i].legal_actions);
    if (corpus[i].category == "strategic") {
      strategic_indices.push_back(i);
    } else if (corpus[i].category == "opportunity") {
      opportunity_indices.push_back(i);
    } else if (corpus[i].category == "planner") {
      planner_indices.push_back(i);
    }
  }

  std::cout << absl::StrFormat("Found: %d strategic, %d opportunity, %d planner states.\n",
                               strategic_indices.size(), opportunity_indices.size(), planner_indices.size());

  if (!self_test) {
    if (absl::GetFlag(FLAGS_strict_v2_validation)) {
      if (corpus.empty()) {
        std::cerr << "Fidelity Gate validation failed: Corpus is empty.\n";
        std::exit(1);
      }
      for (const auto& cs : corpus) {
        if (cs.corpus_schema_version != "v2") {
          std::cerr << "Fidelity Gate validation failed: strict_v2_validation is enabled, but the corpus is not a v2 corpus (corpus_schema_version != \"v2\" or missing).\n";
          std::exit(1);
        }
      }
    }

    bool gate1_only = absl::GetFlag(FLAGS_gate1_only);
    bool allow_any_opportunity_count = absl::GetFlag(FLAGS_allow_any_opportunity_count);

    if (gate1_only) {
      opportunity_indices.clear();
    }

    if (!gate1_only) {
      if (!allow_any_opportunity_count) {
        if (strategic_indices.size() < 64) {
          std::cerr << "Fidelity Gate validation failed: Strategic states count in corpus is "
                    << strategic_indices.size() << ", which is less than the required 64 states.\n";
          std::exit(1);
        }

        bool is_v2_corpus = false;
        if (!opportunity_indices.empty()) {
          is_v2_corpus = (corpus[opportunity_indices[0]].corpus_schema_version == "v2");
        }



        if (absl::GetFlag(FLAGS_strict_v2_validation) && is_v2_corpus) {
          if (opportunity_indices.size() != 32) {
            std::cerr << "Fidelity Gate validation failed: Opportunity states count in corpus is "
                      << opportunity_indices.size() << ", which is not exactly 32 states. (Normal mode requires exactly 32)\n";
            std::exit(1);
          }

          // Validate seat balance: exactly 8 opportunity states per seat (0, 1, 2, 3)
          std::vector<int> seat_counts(4, 0);
          for (size_t idx : opportunity_indices) {
            int s = corpus[idx].seat;
            if (s < 0 || s >= 4) {
              std::cerr << "Fidelity Gate validation failed: Opportunity state has invalid seat: " << s << "\n";
              std::exit(1);
            }
            seat_counts[s]++;
          }
          for (int s = 0; s < 4; ++s) {
            if (seat_counts[s] != 8) {
              std::cerr << "Fidelity Gate validation failed: Seat balance check failed. Seat " << s
                        << " has " << seat_counts[s] << " opportunity states (expected exactly 8).\n";
              std::exit(1);
            }
          }
        }
        // Validate stored role, history hash, schema version, and provenance
        if (absl::GetFlag(FLAGS_strict_v2_validation) && is_v2_corpus) {
          for (size_t idx : opportunity_indices) {
            const auto& cs = corpus[idx];
            if (cs.role != "AGENT_PRIMARY") {
              std::cerr << "Fidelity Gate validation failed: Opportunity state role must be AGENT_PRIMARY, got: " << cs.role << "\n";
              std::exit(1);
            }
            if (cs.history_hash.empty()) {
              std::cerr << "Fidelity Gate validation failed: Opportunity state has empty history_hash\n";
              std::exit(1);
            }
            if (cs.corpus_schema_version != "v2") {
              std::cerr << "Fidelity Gate validation failed: Opportunity state has invalid corpus_schema_version: "
                        << cs.corpus_schema_version << " (expected v2)\n";
              std::exit(1);
            }
            if (cs.source_seed < 0 || cs.episode_id < 0 || cs.decision_index < 0) {
              std::cerr << "Fidelity Gate validation failed: Invalid provenance fields. seed=" << cs.source_seed
                        << ", episode=" << cs.episode_id << ", decision_idx=" << cs.decision_index << "\n";
              std::exit(1);
            }
          }
        }
      }
    }
  }

  size_t target_g1 = self_test ? 2 : 64;
  size_t target_g2 = self_test ? 2 : 32;

  std::vector<size_t> gate1_indices(strategic_indices.begin(), strategic_indices.begin() + std::min<size_t>(target_g1, strategic_indices.size()));
  std::vector<size_t> gate2_indices(opportunity_indices.begin(), opportunity_indices.begin() + std::min<size_t>(target_g2, opportunity_indices.size()));

  std::vector<double> v_critic_g1(gate1_indices.size(), 0.0);
  std::vector<double> v_true_g1(gate1_indices.size(), 0.0);

  std::string model_hash = self_test ? "self_test" : open_spiel::ComputeFileSHA256(model_checkpoint);
  std::string corpus_hash = open_spiel::ComputeFileSHA256(corpus_path);
  bool gate1_cache_loaded = false;
  std::string gate1_cache_path = absl::GetFlag(FLAGS_gate1_cache_path);
  if (!self_test && !gate1_cache_path.empty() && std::filesystem::exists(gate1_cache_path)) {
    try {
      std::ifstream cache_in(gate1_cache_path);
      std::string cache_text((std::istreambuf_iterator<char>(cache_in)),
                             std::istreambuf_iterator<char>());
      auto parsed = open_spiel::json::FromString(cache_text);
      if (parsed) {
        auto obj = parsed.value().GetObject();
        auto critic_arr = obj.at("v_critic").GetArray();
        auto true_arr = obj.at("v_true").GetArray();
        bool valid = obj.at("protocol_version").GetString() == "gate1-cache-v1" &&
                     obj.at("model_checkpoint_hash").GetString() == model_hash &&
                     obj.at("corpus_fingerprint").GetString() == corpus_hash &&
                     obj.at("seed").GetInt() == seed &&
                     obj.at("rollouts").GetInt() == absl::GetFlag(FLAGS_rollouts) &&
                     obj.find("binary_hash") != obj.end() &&
                     obj.at("binary_hash").GetString() == open_spiel::ComputeFileSHA256("/proc/self/exe") &&
                     critic_arr.size() == gate1_indices.size() &&
                     true_arr.size() == gate1_indices.size();
        if (valid) {
          for (size_t i = 0; i < gate1_indices.size(); ++i) {
            v_critic_g1[i] = critic_arr[i].GetDouble();
            v_true_g1[i] = true_arr[i].GetDouble();
          }
          gate1_cache_loaded = true;
          std::cout << "Loaded validated Gate 1 cache: " << gate1_cache_path << "\n";
        }
      }
    } catch (const std::exception& e) {
      std::cerr << "Ignoring invalid Gate 1 cache: " << e.what() << "\n";
    }
  }

  std::vector<double> q_raw_g2(gate2_indices.size(), 0.0);
  std::vector<double> q_search_g2(gate2_indices.size(), 0.0);

  std::mutex eval_mutex;
  std::atomic<int> completed_g1{0};
  std::atomic<int> completed_g2{0};

  // Gate 1 uses rollout-sized tasks rather than 64 state-sized tasks. This
  // avoids the multi-hour straggler tail when games have very different
  // remaining lengths.
  int gate1_rollouts = self_test ? 2 : absl::GetFlag(FLAGS_rollouts);
  std::vector<std::unique_ptr<State>> gate1_states;
  std::vector<std::vector<double>> gate1_returns(
      gate1_indices.size(), std::vector<double>(gate1_rollouts, 0.0));

  std::vector<bool> gate1_returns_cached(gate1_indices.size(), false);
  std::vector<bool> gate1_critic_cached(gate1_indices.size(), false);
  if (split_cache.IsActive()) {
    for (size_t i = 0; i < gate1_indices.size(); ++i) {
      size_t absolute_idx = gate1_indices[i];
      if (split_cache.LoadGate1ReturnsOnly(absolute_idx, gate1_returns[i])) {
        v_true_g1[i] = std::accumulate(gate1_returns[i].begin(), gate1_returns[i].end(), 0.0) / gate1_returns[i].size();
        gate1_returns_cached[i] = true;
      }
      double critic_val = 0.0;
      if (split_cache.LoadGate1CriticOnly(absolute_idx, critic_val)) {
        v_critic_g1[i] = critic_val;
        gate1_critic_cached[i] = true;
      }
    }
  }

  if (!gate1_cache_loaded) {
    gate1_states.reserve(gate1_indices.size());
    for (size_t i = 0; i < gate1_indices.size(); ++i) {
      if (gate1_returns_cached[i] && gate1_critic_cached[i]) {
        gate1_states.push_back(nullptr);
      } else {
        const auto& cs = corpus[gate1_indices[i]];
        gate1_states.push_back(
            ReconstructState(game, cs.history, cs.player, cs.observation));
      }
    }
  }
  std::atomic<int> next_gate1_task{0};
  std::atomic<int> completed_gate1_tasks{0};
  auto worker_g1 = [&]() {
    int tasks_per_state = gate1_rollouts + 1;  // one critic + N outcomes
    int total_tasks = gate1_indices.size() * tasks_per_state;
    while (true) {
      int task = next_gate1_task.fetch_add(1);
      if (task >= total_tasks) break;
      int idx = task / tasks_per_state;
      int local_task = task % tasks_per_state;
      if (gate1_returns_cached[idx] && gate1_critic_cached[idx]) {
        int done = completed_gate1_tasks.fetch_add(1) + 1;
        completed_g1.store(std::min<int>(gate1_indices.size(), done / tasks_per_state));
        continue;
      }
      size_t corpus_idx = gate1_indices[idx];
      const auto& cs = corpus[corpus_idx];
      if (local_task == 0) {
        if (gate1_critic_cached[idx]) {
          int done = completed_gate1_tasks.fetch_add(1) + 1;
          completed_g1.store(std::min<int>(gate1_indices.size(), done / tasks_per_state));
          continue;
        }
        double critic_val = global_evaluator->Evaluate(*gate1_states[idx])[cs.player];
        if (mock_fail) critic_val += 0.10;
        v_critic_g1[idx] = critic_val;
      } else {
        if (gate1_returns_cached[idx]) {
          int done = completed_gate1_tasks.fetch_add(1) + 1;
          completed_g1.store(std::min<int>(gate1_indices.size(), done / tasks_per_state));
          continue;
        }
        int k = local_task;
        // Aligned seeds with Gate 1 space prefix 300000
        uint64_t roll_seed = 300000 + 1000 * idx + k;
        gate1_returns[idx][k - 1] = RunRawPolicyRollout(
            *gate1_states[idx], cs.player, global_evaluator.get(), roll_seed,
            utility_divisor);
      }
      int done = completed_gate1_tasks.fetch_add(1) + 1;
      completed_g1.store(std::min<int>(gate1_indices.size(), done / tasks_per_state));
    }
  };

  std::vector<OpportunityStateEvidence> gate2_evidence(gate2_indices.size());
  std::vector<ChoiceOutcomeCacheEntry> choice_outcome_cache(gate2_indices.size());
  bool choice_outcome_cache_loaded = false;
  std::string choice_cache_path = absl::GetFlag(FLAGS_choice_rollout_cache_path);
  if (!self_test && absl::GetFlag(FLAGS_choice_only_gate2) &&
      !choice_cache_path.empty() && std::filesystem::exists(choice_cache_path)) {
    try {
      std::ifstream cache_in(choice_cache_path);
      std::string cache_text((std::istreambuf_iterator<char>(cache_in)),
                             std::istreambuf_iterator<char>());
      auto parsed = open_spiel::json::FromString(cache_text);
      if (parsed) {
        auto obj = parsed.value().GetObject();
        auto states_arr = obj.at("states").GetArray();
        bool valid = obj.at("protocol_version").GetString() == "choice-outcomes-v1" &&
                     obj.at("model_checkpoint_hash").GetString() == model_hash &&
                     obj.at("corpus_fingerprint").GetString() == corpus_hash &&
                     obj.at("rollouts").GetInt() == absl::GetFlag(FLAGS_rollouts) &&
                     obj.find("binary_hash") != obj.end() &&
                     obj.at("binary_hash").GetString() == open_spiel::ComputeFileSHA256("/proc/self/exe") &&
                     obj.find("config") != obj.end() &&
                     states_arr.size() == gate2_indices.size();
        if (valid) {
          auto cached_cfg = obj.at("config").GetObject();
          valid = cached_cfg.at("opponent_mode").GetInt() == absl::GetFlag(FLAGS_opponent_mode) &&
                  cached_cfg.at("opponent_temperature").GetDouble() == absl::GetFlag(FLAGS_opponent_temperature) &&
                  cached_cfg.at("puct_c").GetDouble() == absl::GetFlag(FLAGS_puct_c) &&
                  cached_cfg.at("max_simulations").GetInt() == absl::GetFlag(FLAGS_max_simulations) &&
                  cached_cfg.at("nonlinear_value_head").GetBool() == absl::GetFlag(FLAGS_nonlinear_value_head) &&
                  cached_cfg.at("conservative_override_enabled").GetBool() == absl::GetFlag(FLAGS_conservative_override_enabled) &&
                  cached_cfg.at("conservative_covered_prior_threshold").GetDouble() == absl::GetFlag(FLAGS_conservative_covered_prior_threshold) &&
                  cached_cfg.at("conservative_meaningful_visit_threshold").GetInt() == absl::GetFlag(FLAGS_conservative_meaningful_visit_threshold) &&
                  cached_cfg.at("conservative_q_margin_threshold").GetDouble() == absl::GetFlag(FLAGS_conservative_q_margin_threshold) &&
                  cached_cfg.at("conservative_stability_checkpoint_fraction").GetDouble() == absl::GetFlag(FLAGS_conservative_stability_checkpoint_fraction) &&
                  cached_cfg.at("conservative_continuation_overrides_disabled").GetBool() == absl::GetFlag(FLAGS_conservative_continuation_overrides_disabled) &&
                  cached_cfg.at("fixed_continuation_reserve").GetInt() == absl::GetFlag(FLAGS_fixed_continuation_reserve) &&
                  cached_cfg.at("purchase_combat_budget").GetInt() == absl::GetFlag(FLAGS_purchase_combat_budget) &&
                  cached_cfg.at("live_continuation_reserve_seconds").GetDouble() == absl::GetFlag(FLAGS_live_continuation_reserve_seconds);
        }
        for (size_t idx = 0; valid && idx < states_arr.size(); ++idx) {
          auto state_obj = states_arr[idx].GetObject();
          ChoiceOutcomeCacheEntry entry;
          entry.corpus_index = state_obj.at("corpus_index").GetInt();
          auto actions_arr = state_obj.at("actions").GetArray();
          auto returns_arr = state_obj.at("returns").GetArray();
          auto critics_arr = state_obj.at("successor_critic_values").GetArray();
          valid = entry.corpus_index == static_cast<int>(gate2_indices[idx]) &&
                  !actions_arr.empty() && returns_arr.size() == actions_arr.size() &&
                  critics_arr.size() == actions_arr.size();
          for (size_t a_idx = 0; valid && a_idx < actions_arr.size(); ++a_idx) {
            entry.actions.push_back(static_cast<Action>(actions_arr[a_idx].GetInt()));
            entry.successor_critic_values.push_back(critics_arr[a_idx].GetDouble());
            auto action_returns = returns_arr[a_idx].GetArray();
            if (action_returns.size() != static_cast<size_t>(absl::GetFlag(FLAGS_rollouts))) {
              valid = false;
              break;
            }
            entry.returns.emplace_back();
            entry.returns.back().reserve(action_returns.size());
            for (const auto& value : action_returns) {
              entry.returns.back().push_back(value.GetDouble());
            }
          }
          if (valid) choice_outcome_cache[idx] = std::move(entry);
        }
        if (valid) {
          choice_outcome_cache_loaded = true;
          std::cout << "Loaded validated choice rollout cache: "
                    << choice_cache_path << "\n";
        }
      }
    } catch (const std::exception& e) {
      std::cerr << "Ignoring invalid choice rollout cache: " << e.what() << "\n";
    }
  }

  // Worker for Gate 2 (Choice Advantages)
  std::atomic<int> next_gate2_idx{0};
  auto worker_g2 = [&]() {
    int total_g2 = gate2_indices.size();
    while (true) {
      if (g_cache_validation_failed) break;
      int idx = next_gate2_idx.fetch_add(1);
      if (idx >= total_g2) break;

      size_t corpus_idx = gate2_indices[idx];
      const auto& cs = corpus[corpus_idx];
      std::unique_ptr<State> state;
      {
        auto temp_state = ReconstructState(game, cs.history, cs.player, cs.observation);
        if (ClassifyDuneDecisionRole(*temp_state, cs.player, false) == DuneDecisionRole::kAgentPrimary) {
          state = std::move(temp_state);
        } else {
          // Legacy or continuation state: rewind by popping the last action to reach primary card-play turn
          std::vector<Action> rewound_history = cs.history;
          if (!rewound_history.empty()) {
            rewound_history.pop_back();
          }
          state = ReconstructState(game, rewound_history, cs.player, {});
          SPIEL_CHECK_TRUE(ClassifyDuneDecisionRole(*state, cs.player, false) == DuneDecisionRole::kAgentPrimary);
        }
      }

      DuneSearchConfig bot_cfg;
      bot_cfg.max_simulations = self_test ? 5 : absl::GetFlag(FLAGS_max_simulations);
      bot_cfg.puct_c = absl::GetFlag(FLAGS_puct_c);
      bot_cfg.opponent_mode = static_cast<SearchOpponentMode>(absl::GetFlag(FLAGS_opponent_mode));
      bot_cfg.opponent_temperature = absl::GetFlag(FLAGS_opponent_temperature);
      bot_cfg.temperature = 0.0;
      {
        double t_budget = absl::GetFlag(FLAGS_relative_time_budget_ms);
        bot_cfg.relative_time_budget_ms = (t_budget > 0.0) ? t_budget : std::numeric_limits<double>::infinity();
      }
      bot_cfg.seed = search_seed + 200000 + 1000 * idx;
      bot_cfg.fixed_session_limit = bot_cfg.max_simulations;
      bot_cfg.fixed_continuation_reserve = absl::GetFlag(FLAGS_fixed_continuation_reserve);
      bot_cfg.purchase_combat_budget = absl::GetFlag(FLAGS_purchase_combat_budget);
      bot_cfg.root_prior_temperature = absl::GetFlag(FLAGS_root_prior_temperature);
      bot_cfg.live_continuation_reserve_seconds = absl::GetFlag(FLAGS_live_continuation_reserve_seconds);

      bot_cfg.conservative_override_enabled = absl::GetFlag(FLAGS_conservative_override_enabled);
      bot_cfg.conservative_covered_prior_threshold = absl::GetFlag(FLAGS_conservative_covered_prior_threshold);
      bot_cfg.conservative_meaningful_visit_threshold = absl::GetFlag(FLAGS_conservative_meaningful_visit_threshold);
      bot_cfg.conservative_q_margin_threshold = absl::GetFlag(FLAGS_conservative_q_margin_threshold);
      bot_cfg.conservative_stability_checkpoint_fraction = absl::GetFlag(FLAGS_conservative_stability_checkpoint_fraction);
      bot_cfg.conservative_continuation_overrides_disabled = absl::GetFlag(FLAGS_conservative_continuation_overrides_disabled);

      // Compute Rollouts
      double q_raw_sum = 0.0;
      double q_search_sum = 0.0;
      int rollouts = self_test ? 2 : absl::GetFlag(FLAGS_rollouts);
      std::vector<double> raw_returns_list;
      std::vector<double> search_returns_list;
      std::vector<double> paired_diffs;

      OpportunityStateEvidence& ev = gate2_evidence[idx];
      ev.corpus_index = corpus_idx;
      ev.episode_id = cs.episode_id;
      ev.history_hash = cs.history_hash;
      ev.seat = cs.seat;
      ev.round = cs.round;
      ev.role = cs.role;

      if (absl::GetFlag(FLAGS_choice_only_gate2)) {
        // Root selection is deterministic for a state/configuration/seed. The
        // legacy harness rebuilt the same tree for every one of the 256 outcome
        // rollouts. Search once, then compare the raw and searched root actions
        // under common stochastic continuations.
        bool has_search_cached = false;
        open_spiel::json::Object search_cached_res;
        Action search_action = kInvalidAction;
        Action raw_action = kInvalidAction;
        std::vector<Action> root_actions;
        DuneSearchResult root_res;

        if (split_cache.IsActive()) {
          has_search_cached = split_cache.LoadGate2SearchShard(corpus_idx, search_cached_res);
        }

        if (has_search_cached) {
          search_action = static_cast<Action>(search_cached_res.at("selected_action").GetInt());
          raw_action = static_cast<Action>(search_cached_res.at("raw_reference_action").GetInt());
          auto act_arr = search_cached_res.at("actions").GetArray();
          for (const auto& val : act_arr) {
            root_actions.push_back(static_cast<Action>(val.GetInt()));
          }

          root_res.diagnostics.actions = root_actions;
          root_res.diagnostics.selected_action = search_action;
          root_res.diagnostics.raw_reference_action = raw_action;

          root_res.diagnostics.session_id = search_cached_res.at("session_id").GetString();
          root_res.simulations_completed = search_cached_res.at("simulations_completed").GetInt();
          root_res.inference_count = search_cached_res.at("inference_count").GetInt();
          root_res.diagnostics.mean_depth = search_cached_res.at("mean_depth").GetDouble();
          root_res.diagnostics.terminal_leaf_fraction = search_cached_res.at("terminal_leaf_fraction").GetDouble();
          root_res.used_fallback = search_cached_res.at("used_fallback").GetBool();
          root_res.fallback_reason = search_cached_res.at("fallback_reason").GetString();
          root_res.diagnostics.confidence_fallback = search_cached_res.at("confidence_fallback").GetBool();
          root_res.diagnostics.mcts_overrode_raw = search_cached_res.at("mcts_overrode_raw").GetBool();
          root_res.diagnostics.mcts_proposed_action = static_cast<Action>(search_cached_res.at("mcts_proposed_action").GetInt());
          root_res.diagnostics.stability_checkpoint_reached = search_cached_res.at("stability_checkpoint_reached").GetBool();
          root_res.diagnostics.stability_agreement = search_cached_res.at("stability_agreement").GetBool();
          root_res.diagnostics.pass_complete_search = search_cached_res.at("pass_complete_search").GetBool();
          root_res.diagnostics.pass_min_actions = search_cached_res.at("pass_min_actions").GetBool();
          root_res.diagnostics.pass_prior_mass = search_cached_res.at("pass_prior_mass").GetBool();
          root_res.diagnostics.pass_meaningful_visits = search_cached_res.at("pass_meaningful_visits").GetBool();
          root_res.diagnostics.pass_q_margin = search_cached_res.at("pass_q_margin").GetBool();
          root_res.diagnostics.pass_stability = search_cached_res.at("pass_stability").GetBool();
          root_res.diagnostics.chosen_action_raw_prior_rank = search_cached_res.at("chosen_action_raw_prior_rank").GetInt();

          auto pr_arr = search_cached_res.at("priors").GetArray();
          for (const auto& val : pr_arr) root_res.diagnostics.priors.push_back(val.GetDouble());

          auto vis_arr = search_cached_res.at("visit_counts").GetArray();
          for (const auto& val : vis_arr) root_res.diagnostics.visit_counts.push_back(static_cast<int>(val.GetInt()));

          auto q_arr = search_cached_res.at("q_values").GetArray();
          for (const auto& val : q_arr) root_res.diagnostics.q_values.push_back(val.GetDouble());

        } else {
          // Root selection is deterministic for a state/configuration/seed. The
          // legacy harness rebuilt the same tree for every one of the 256 outcome
          // rollouts. Search once, then compare the raw and searched root actions
          // under common stochastic continuations.
          DuneSearchBudgetMode mode =
              (bot_cfg.relative_time_budget_ms == std::numeric_limits<double>::infinity())
                  ? DuneSearchBudgetMode::kFixedSessionSimulations
                  : DuneSearchBudgetMode::kLiveDeadline;
          DuneSearchSession root_session(bot_cfg, global_evaluator, mode);
          root_res = root_session.Search(*state);
          std::mt19937 step_rng(raw_policy_seed + 300000 + 1000 * idx);
          double r_val = absl::Uniform(step_rng, 0.0, 1.0);
          ControllerDecision decision = root_session.SelectControllerAction(*state, root_res, r_val);
          root_res = root_session.CommitAction(decision);

          search_action = root_res.diagnostics.selected_action;
          if (search_action == kInvalidAction) {
            search_action = GetBestActionFromSearchResult(root_res);
          }

          raw_action = decision.raw_reference_action;
          SPIEL_CHECK_NE(raw_action, kInvalidAction);
          SPIEL_CHECK_NE(search_action, kInvalidAction);

          root_actions = root_res.diagnostics.actions;
          if (root_actions.empty()) root_actions = state->LegalActions();

          if (split_cache.IsActive()) {
            open_spiel::json::Object save_obj;
            save_obj["selected_action"] = static_cast<int64_t>(search_action);
            save_obj["raw_reference_action"] = static_cast<int64_t>(raw_action);
            open_spiel::json::Array save_act;
            for (Action a : root_actions) save_act.push_back(static_cast<int64_t>(a));
            save_obj["actions"] = save_act;
            save_obj["session_id"] = root_res.diagnostics.session_id;
            save_obj["simulations_completed"] = static_cast<int64_t>(root_res.simulations_completed);
            save_obj["inference_count"] = static_cast<int64_t>(root_res.inference_count);
            save_obj["mean_depth"] = root_res.diagnostics.mean_depth;
            save_obj["terminal_leaf_fraction"] = root_res.diagnostics.terminal_leaf_fraction;
            save_obj["used_fallback"] = root_res.used_fallback;
            save_obj["fallback_reason"] = root_res.fallback_reason;
            save_obj["confidence_fallback"] = root_res.diagnostics.confidence_fallback;
            save_obj["mcts_overrode_raw"] = root_res.diagnostics.mcts_overrode_raw;
            save_obj["mcts_proposed_action"] = static_cast<int64_t>(root_res.diagnostics.mcts_proposed_action);
            save_obj["stability_checkpoint_reached"] = root_res.diagnostics.stability_checkpoint_reached;
            save_obj["stability_agreement"] = root_res.diagnostics.stability_agreement;
            save_obj["pass_complete_search"] = root_res.diagnostics.pass_complete_search;
            save_obj["pass_min_actions"] = root_res.diagnostics.pass_min_actions;
            save_obj["pass_prior_mass"] = root_res.diagnostics.pass_prior_mass;
            save_obj["pass_meaningful_visits"] = root_res.diagnostics.pass_meaningful_visits;
            save_obj["pass_q_margin"] = root_res.diagnostics.pass_q_margin;
            save_obj["pass_stability"] = root_res.diagnostics.pass_stability;
            save_obj["chosen_action_raw_prior_rank"] = static_cast<int64_t>(root_res.diagnostics.chosen_action_raw_prior_rank);

            open_spiel::json::Array save_pr;
            for (double p : root_res.diagnostics.priors) save_pr.push_back(p);
            save_obj["priors"] = save_pr;

            open_spiel::json::Array save_vis;
            for (int v : root_res.diagnostics.visit_counts) save_vis.push_back(static_cast<int64_t>(v));
            save_obj["visit_counts"] = save_vis;

            open_spiel::json::Array save_q;
            for (double q : root_res.diagnostics.q_values) save_q.push_back(q);
            save_obj["q_values"] = save_q;

            split_cache.SaveGate2SearchShard(corpus_idx, save_obj);
          }
        }

        // Exact Legal Action coverage validation
        std::vector<Action> state_legal = state->LegalActions();
        std::vector<Action> sorted_state_legal = state_legal;
        std::vector<Action> sorted_root_actions = root_actions;
        std::sort(sorted_state_legal.begin(), sorted_state_legal.end());
        std::sort(sorted_root_actions.begin(), sorted_root_actions.end());
        if (sorted_root_actions != sorted_state_legal) {
          std::cerr << "[ERROR] Prior validation failed: root actions do not match state legal actions." << std::endl;
          g_cache_validation_failed = true;
          return;
        }

        // Validate MCTS search diagnostics priors
        if (root_res.diagnostics.priors.size() != root_actions.size()) {
          std::cerr << "[ERROR] Prior validation failed: MCTS priors size mismatch. priors="
                    << root_res.diagnostics.priors.size() << ", actions=" << root_actions.size() << std::endl;
          g_cache_validation_failed = true;
          return;
        }
        double prior_sum = 0.0;
        for (double p : root_res.diagnostics.priors) {
          if (!std::isfinite(p) || p < 0.0) {
            std::cerr << "[ERROR] Prior validation failed: invalid MCTS probability value " << p << std::endl;
            g_cache_validation_failed = true;
            return;
          }
          prior_sum += p;
        }
        if (std::abs(prior_sum - 1.0) > 1e-5) {
          std::cerr << "[ERROR] Prior validation failed: MCTS probability mass does not sum to 1. sum=" << prior_sum << std::endl;
          g_cache_validation_failed = true;
          return;
        }

        // Retrieve raw policy prior from the evaluator
        ActionsAndProbs raw_prior = global_evaluator->Prior(*state);
        if (raw_prior.size() != state_legal.size()) {
          std::cerr << "[ERROR] Prior validation failed: evaluator raw prior size mismatch. expected="
                    << state_legal.size() << ", got=" << raw_prior.size() << std::endl;
          g_cache_validation_failed = true;
          return;
        }
        double raw_prior_sum = 0.0;
        for (const auto& ap : raw_prior) {
          double p = ap.second;
          if (!std::isfinite(p) || p < 0.0) {
            std::cerr << "[ERROR] Prior validation failed: invalid evaluator probability value " << p << std::endl;
            g_cache_validation_failed = true;
            return;
          }
          raw_prior_sum += p;
        }
        if (std::abs(raw_prior_sum - 1.0) > 1e-5) {
          std::cerr << "[ERROR] Prior validation failed: evaluator probability mass does not sum to 1. sum=" << raw_prior_sum << std::endl;
          g_cache_validation_failed = true;
          return;
        }

        std::vector<std::vector<double>> action_returns;
        std::vector<double> successor_critic_vals;
        std::vector<double> successor_true_vals(root_actions.size(), 0.0);

        bool outcomes_cached = false;
        if (split_cache.IsActive()) {
          outcomes_cached = split_cache.LoadGate2OutcomesShard(corpus_idx, root_actions, action_returns, successor_critic_vals);
        } else if (choice_outcome_cache_loaded) {
          const auto& cached = choice_outcome_cache[idx];
          if (cached.actions == root_actions) {
            action_returns = cached.returns;
            successor_critic_vals = cached.successor_critic_values;
            outcomes_cached = true;
          }
        }

        if (!outcomes_cached) {
          action_returns.assign(
              root_actions.size(), std::vector<double>(rollouts, 0.0));
          successor_critic_vals.assign(root_actions.size(), 0.0);
          for (size_t a_idx = 0; a_idx < root_actions.size(); ++a_idx) {
            double critic_sum = 0.0;
            for (int k = 1; k <= rollouts; ++k) {
              uint64_t chance_seed = chance_seed_base + 1000 * idx + k;
              uint64_t rollout_seed = rollout_seed_base + 1000 * idx + k;
              ForcedActionRolloutResult forced = RunForcedActionPolicyRollout(
                  *state, cs.player, root_actions[a_idx], global_evaluator.get(),
                  chance_seed, rollout_seed, utility_divisor);
              action_returns[a_idx][k - 1] = forced.return_val;
              critic_sum += forced.successor_critic;
            }
            successor_critic_vals[a_idx] = critic_sum / rollouts;
          }

          if (split_cache.IsActive()) {
            split_cache.SaveGate2OutcomesShard(corpus_idx, root_actions, action_returns, successor_critic_vals);
          } else {
            choice_outcome_cache[idx].corpus_index = corpus_idx;
            choice_outcome_cache[idx].actions = root_actions;
            choice_outcome_cache[idx].returns = action_returns;
            choice_outcome_cache[idx].successor_critic_values = successor_critic_vals;
          }
        }
        for (size_t a_idx = 0; a_idx < root_actions.size(); ++a_idx) {
          successor_true_vals[a_idx] =
              std::accumulate(action_returns[a_idx].begin(),
                              action_returns[a_idx].end(), 0.0) / rollouts;
        }

        auto action_index = [&](Action action) -> size_t {
          auto it = std::find(root_actions.begin(), root_actions.end(), action);
          SPIEL_CHECK_TRUE(it != root_actions.end());
          return std::distance(root_actions.begin(), it);
        };
        size_t raw_idx = action_index(raw_action);
        size_t search_idx = action_index(search_action);

        bool used_fallback = root_res.used_fallback || root_res.diagnostics.confidence_fallback;

        std::vector<double> raw_probs(root_actions.size(), 0.0);
        for (size_t a_idx = 0; a_idx < root_actions.size(); ++a_idx) {
          Action act = root_actions[a_idx];
          double prob = 0.0;
          bool found = false;
          for (const auto& ap : raw_prior) {
            if (ap.first == act) {
              prob = ap.second;
              found = true;
              break;
            }
          }
          if (!found) {
            std::cerr << "[ERROR] Prior validation failed: evaluator raw prior missing action " << act << std::endl;
            g_cache_validation_failed = true;
            return;
          }
          raw_probs[a_idx] = prob;
        }

        std::vector<double> v_raw_hat_rollouts(rollouts, 0.0);
        for (int k = 0; k < rollouts; ++k) {
          double sum_probs = 0.0;
          for (size_t a_idx = 0; a_idx < root_actions.size(); ++a_idx) {
            sum_probs += raw_probs[a_idx] * action_returns[a_idx][k];
          }
          v_raw_hat_rollouts[k] = sum_probs;
        }

        raw_returns_list.resize(rollouts);
        search_returns_list.resize(rollouts);
        paired_diffs.resize(rollouts);

        for (int k = 0; k < rollouts; ++k) {
          if (used_fallback) {
            search_returns_list[k] = v_raw_hat_rollouts[k];
            raw_returns_list[k] = v_raw_hat_rollouts[k];
          } else {
            search_returns_list[k] = action_returns[search_idx][k];
            raw_returns_list[k] = v_raw_hat_rollouts[k];
          }
          paired_diffs[k] = search_returns_list[k] - raw_returns_list[k];
          q_raw_sum += raw_returns_list[k];
          q_search_sum += search_returns_list[k];

          RolloutEvidence rev;
          rev.rollout_index = k + 1;
          rev.session_id = root_res.diagnostics.session_id;
          rev.raw_action_sequence = {static_cast<int>(raw_action)};
          rev.search_action_sequence = {static_cast<int>(search_action)};
          std::vector<int> root_acts_int;
          for (Action a : root_actions) root_acts_int.push_back(static_cast<int>(a));
          rev.raw_legal_actions = {root_acts_int};
          rev.raw_decision_roles = {"AGENT_PRIMARY"};
          rev.search_legal_actions = {root_acts_int};
          rev.search_decision_roles = {"AGENT_PRIMARY"};
          rev.raw_priors = raw_probs;
          rev.search_visits = root_res.diagnostics.visit_counts;
          rev.root_q_values = root_res.diagnostics.q_values;
          rev.selected_action_rank = root_res.diagnostics.chosen_action_raw_prior_rank;
          rev.raw_return = raw_returns_list[k];
          rev.search_return = search_returns_list[k];
          rev.paired_advantage = paired_diffs[k];
          // Shared search work is accounted once rather than multiplied by the
          // number of stochastic outcome samples.
          if (k == 0) {
            rev.simulations_used = root_res.simulations_completed;
            rev.initial_simulations = root_res.simulations_completed;
            rev.search_depth = root_res.diagnostics.mean_depth;
            rev.terminal_leaf_fraction = root_res.diagnostics.terminal_leaf_fraction;
            rev.inference_count = root_res.inference_count;
            rev.search_decisions_count = 1;
          }
          for (Action a : root_actions) {
            rev.root_action_ids.push_back(static_cast<int>(a));
          }
          rev.primary_action_changed = (search_action != raw_action);
          ev.rollouts.push_back(std::move(rev));
        }

        ev.raw_action_sequence = {static_cast<int>(raw_action)};
        ev.search_action_sequence = {static_cast<int>(search_action)};
        std::vector<int> root_acts_int;
        for (Action a : root_actions) root_acts_int.push_back(static_cast<int>(a));
        ev.raw_legal_actions = {root_acts_int};
        ev.raw_decision_roles = {"AGENT_PRIMARY"};
        ev.search_legal_actions = {root_acts_int};
        ev.search_decision_roles = {"AGENT_PRIMARY"};
        ev.raw_priors = raw_probs;
        ev.search_visits = root_res.diagnostics.visit_counts;
        ev.root_q_values = root_res.diagnostics.q_values;
        ev.selected_action_rank = root_res.diagnostics.chosen_action_raw_prior_rank;
        for (Action a : root_actions) ev.root_action_ids.push_back(static_cast<int>(a));
        ev.session_id = root_res.diagnostics.session_id;
        ev.total_simulations_used = root_res.simulations_completed;
        ev.initial_simulations = root_res.simulations_completed;
        ev.total_inference_count = root_res.inference_count;
        ev.mean_search_depth = root_res.diagnostics.mean_depth;
        ev.mean_terminal_leaf_fraction = root_res.diagnostics.terminal_leaf_fraction;
        ev.primary_action_changed = (search_action != raw_action);

        ev.used_fallback = root_res.used_fallback;
        ev.fallback_reason = root_res.fallback_reason;
        ev.confidence_fallback = root_res.diagnostics.confidence_fallback;
        ev.mcts_overrode_raw = root_res.diagnostics.mcts_overrode_raw;
        ev.raw_reference_action = root_res.diagnostics.raw_reference_action;
        ev.mcts_proposed_action = root_res.diagnostics.mcts_proposed_action;
        ev.selected_action = root_res.diagnostics.selected_action;
        ev.stability_checkpoint_reached = root_res.diagnostics.stability_checkpoint_reached;
        ev.stability_agreement = root_res.diagnostics.stability_agreement;
        ev.pass_complete_search = root_res.diagnostics.pass_complete_search;
        ev.pass_min_actions = root_res.diagnostics.pass_min_actions;
        ev.pass_prior_mass = root_res.diagnostics.pass_prior_mass;
        ev.pass_meaningful_visits = root_res.diagnostics.pass_meaningful_visits;
        ev.pass_q_margin = root_res.diagnostics.pass_q_margin;
        ev.pass_stability = root_res.diagnostics.pass_stability;

        ev.successor_critic_values = successor_critic_vals;
        ev.successor_true_values = successor_true_vals;
        ev.choice_rank_correlation = SpearmanCorrelation(
            root_res.diagnostics.q_values, successor_true_vals);

        double successor_bias_sum = 0.0;
        double successor_sq_sum = 0.0;
        for (size_t a_idx = 0; a_idx < successor_true_vals.size(); ++a_idx) {
          double diff = successor_critic_vals[a_idx] - successor_true_vals[a_idx];
          successor_bias_sum += diff;
          successor_sq_sum += diff * diff;
        }
        if (!successor_true_vals.empty()) {
          ev.successor_mean_bias = successor_bias_sum / successor_true_vals.size();
          ev.successor_rmse = std::sqrt(successor_sq_sum / successor_true_vals.size());
        }

        const auto& sampled_leaf_states = root_res.diagnostics.sampled_leaf_states;
        int diagnostic_rollouts = std::max(1, absl::GetFlag(FLAGS_diagnostic_rollouts));
        for (size_t l_idx = 0; l_idx < sampled_leaf_states.size(); ++l_idx) {
          auto leaf_state = sampled_leaf_states[l_idx]->Clone();
          double leaf_critic = leaf_state->IsTerminal()
              ? leaf_state->Returns()[cs.player] / utility_divisor
              : global_evaluator->Evaluate(*leaf_state)[cs.player];
          double leaf_true_sum = 0.0;
          for (int r = 1; r <= diagnostic_rollouts; ++r) {
            uint64_t leaf_seed = 500000 + 1000 * idx + 100 * l_idx + r;
            leaf_true_sum += RunRawPolicyRollout(
                *leaf_state, cs.player, global_evaluator.get(), leaf_seed,
                utility_divisor);
          }
          ev.leaf_critic_values.push_back(leaf_critic);
          ev.leaf_true_values.push_back(leaf_true_sum / diagnostic_rollouts);
        }
        double leaf_bias_sum = 0.0;
        double leaf_sq_sum = 0.0;
        for (size_t l_idx = 0; l_idx < ev.leaf_true_values.size(); ++l_idx) {
          double diff = ev.leaf_critic_values[l_idx] - ev.leaf_true_values[l_idx];
          leaf_bias_sum += diff;
          leaf_sq_sum += diff * diff;
        }
        if (!ev.leaf_true_values.empty()) {
          ev.leaf_mean_bias = leaf_bias_sum / ev.leaf_true_values.size();
          ev.leaf_rmse = std::sqrt(leaf_sq_sum / ev.leaf_true_values.size());
        }

        double m_raw = q_raw_sum / rollouts;
        double m_search = q_search_sum / rollouts;
        double m_paired = (q_search_sum - q_raw_sum) / rollouts;
        ev.raw_returns = raw_returns_list;
        ev.search_returns = search_returns_list;
        ev.mean_raw_return = m_raw;
        ev.mean_search_return = m_search;
        ev.raw_return_std_err = CalculateStdErr(raw_returns_list, m_raw);
        ev.search_return_std_err = CalculateStdErr(search_returns_list, m_search);
        ev.mean_paired_advantage = m_paired;
        ev.paired_advantage_std_err = CalculateStdErr(paired_diffs, m_paired);

        ev.diagnostic_legacy_raw_return = std::accumulate(action_returns[raw_idx].begin(), action_returns[raw_idx].end(), 0.0) / rollouts;
        ev.diagnostic_legacy_paired_advantage = std::accumulate(action_returns[search_idx].begin(), action_returns[search_idx].end(), 0.0) / rollouts - ev.diagnostic_legacy_raw_return;

        {
          std::lock_guard<std::mutex> lock(eval_mutex);
          q_raw_g2[idx] = m_raw;
          q_search_g2[idx] = m_search;
          std::cout << absl::StrFormat(
              "Gate 2 Choice State %d: a_raw=%d, a_search=%d, q_raw=%.4f, q_search=%.4f, diff=%.4f\n",
              idx, raw_action, search_action, m_raw, m_search, m_paired)
                    << std::flush;
        }
        completed_g2.fetch_add(1);
        continue;
      }

      // Run paired rollouts
      int total_searched_decisions = 0;

      for (int k = 1; k <= rollouts; ++k) {
        uint64_t roll_seed = 100000 + 1000 * idx + k;

        // Raw controller rollout
        RawRolloutResult raw_res = RunRawControllerRollout(*state, cs.player, global_evaluator.get(), roll_seed, utility_divisor);
        q_raw_sum += raw_res.return_val;
        raw_returns_list.push_back(raw_res.return_val);

        // Search controller rollout
        RolloutDiagnostics search_res = RunSearchControllerRollout(*state, cs.player, global_evaluator, bot_cfg, roll_seed, utility_divisor);
        q_search_sum += search_res.return_val;
        search_returns_list.push_back(search_res.return_val);
        paired_diffs.push_back(search_res.return_val - raw_res.return_val);

        // Accumulate diagnostics
        ev.total_simulations_used += search_res.session_simulations;
        ev.total_re_root_hits += search_res.re_root_hits;
        ev.total_re_root_misses += search_res.re_root_misses;
        ev.mean_search_depth += search_res.search_depth_sum;
        ev.mean_terminal_leaf_fraction += search_res.terminal_leaf_fraction_sum;
        ev.total_inference_count += search_res.inference_count;
        if (search_res.primary_action_changed) ev.primary_action_changed = true;
        if (search_res.continuation_action_changed) ev.continuation_action_changed = true;

        ev.initial_simulations += search_res.initial_simulations;
        ev.continuation_simulations += search_res.continuation_simulations;
        total_searched_decisions += search_res.search_decisions_count;

        ev.inherited_visits += search_res.inherited_visits;
        if (k == 1) {
          // Record action sequences and priors
          for (Action a : raw_res.action_sequence) ev.raw_action_sequence.push_back(static_cast<int>(a));
          for (Action a : search_res.action_sequence) ev.search_action_sequence.push_back(static_cast<int>(a));
          ev.raw_priors = search_res.raw_priors;
          ev.search_visits = search_res.search_visits;
          ev.root_q_values = search_res.root_q_values;
          ev.selected_action_rank = search_res.selected_action_rank;

          for (Action a : search_res.root_action_ids) ev.root_action_ids.push_back(static_cast<int>(a));
          ev.session_id = search_res.session_id;

          for (const auto& acts : raw_res.legal_actions_sequence) {
            std::vector<int> inner;
            for (Action a : acts) inner.push_back(static_cast<int>(a));
            ev.raw_legal_actions.push_back(inner);
          }
          ev.raw_decision_roles = raw_res.decision_roles_sequence;

          for (const auto& acts : search_res.legal_actions_sequence) {
            std::vector<int> inner;
            for (Action a : acts) inner.push_back(static_cast<int>(a));
            ev.search_legal_actions.push_back(inner);
          }
          ev.search_decision_roles = search_res.decision_roles_sequence;
        }

        RolloutEvidence rev;
        rev.rollout_index = k;
        rev.session_id = search_res.session_id;
        for (Action a : raw_res.action_sequence) rev.raw_action_sequence.push_back(static_cast<int>(a));
        for (Action a : search_res.action_sequence) rev.search_action_sequence.push_back(static_cast<int>(a));
        for (const auto& acts : raw_res.legal_actions_sequence) {
          std::vector<int> inner;
          for (Action a : acts) inner.push_back(static_cast<int>(a));
          rev.raw_legal_actions.push_back(inner);
        }
        rev.raw_decision_roles = raw_res.decision_roles_sequence;
        for (const auto& acts : search_res.legal_actions_sequence) {
          std::vector<int> inner;
          for (Action a : acts) inner.push_back(static_cast<int>(a));
          rev.search_legal_actions.push_back(inner);
        }
        rev.search_decision_roles = search_res.decision_roles_sequence;
        rev.raw_priors = search_res.raw_priors;
        rev.search_visits = search_res.search_visits;
        rev.root_q_values = search_res.root_q_values;
        rev.selected_action_rank = search_res.selected_action_rank;
        rev.raw_return = raw_res.return_val;
        rev.search_return = search_res.return_val;
        rev.paired_advantage = search_res.return_val - raw_res.return_val;
        rev.simulations_used = search_res.session_simulations;
        rev.initial_simulations = search_res.initial_simulations;
        rev.continuation_simulations = search_res.continuation_simulations;
        rev.short_window_simulations = search_res.short_window_simulations;
        rev.re_root_hits = search_res.re_root_hits;
        rev.re_root_misses = search_res.re_root_misses;
        rev.search_depth = (search_res.search_decisions_count > 0) ? (search_res.search_depth_sum / search_res.search_decisions_count) : 0.0;
        rev.terminal_leaf_fraction = (search_res.search_decisions_count > 0) ? (search_res.terminal_leaf_fraction_sum / search_res.search_decisions_count) : 0.0;
        rev.inference_count = search_res.inference_count;
        rev.inherited_visits = search_res.inherited_visits;
        for (Action a : search_res.root_action_ids) rev.root_action_ids.push_back(static_cast<int>(a));
        rev.primary_action_changed = search_res.primary_action_changed;
        rev.continuation_action_changed = search_res.continuation_action_changed;
        rev.search_decisions_count = search_res.search_decisions_count;

        ev.rollouts.push_back(rev);
      }

      // Successor / leaf diagnostics: run once per opportunity state
      {
        DuneSearchBudgetMode mode = (bot_cfg.relative_time_budget_ms == std::numeric_limits<double>::infinity())
            ? DuneSearchBudgetMode::kFixedSessionSimulations
            : DuneSearchBudgetMode::kLiveDeadline;
        DuneSearchSession temp_session(bot_cfg, global_evaluator, mode);
        DuneSearchResult root_res = temp_session.Search(*state);
        std::mt19937 step_rng(raw_policy_seed + 300000 + 1000 * idx);
        double r_val = absl::Uniform(step_rng, 0.0, 1.0);
        ControllerDecision decision = temp_session.SelectControllerAction(*state, root_res, r_val);
        root_res = temp_session.CommitAction(decision);

        std::vector<double> successor_critic_vals;
        std::vector<double> successor_true_vals;
        std::vector<double> mcts_q_vals;

        for (size_t a_idx = 0; a_idx < root_res.diagnostics.actions.size(); ++a_idx) {
          Action act = root_res.diagnostics.actions[a_idx];
          double q_mcts = root_res.diagnostics.q_values[a_idx];

          double succ_critic_sum = 0.0;
          double succ_true_sum = 0.0;
          int roll_count = self_test
              ? 2
              : std::max(1, absl::GetFlag(FLAGS_diagnostic_rollouts));

          for (int r = 1; r <= roll_count; ++r) {
            uint64_t roll_seed = 400000 + 1000 * idx + 100 * a_idx + r;

            auto roll_state = state->Clone();
            roll_state->ApplyAction(act);

            std::mt19937 temp_rng(roll_seed);
            while (roll_state->IsChanceNode()) {
              auto outcomes = roll_state->ChanceOutcomes();
              if (outcomes.empty()) break;
              Action choice = SampleAction(outcomes, absl::Uniform(temp_rng, 0.0, 1.0)).first;
              roll_state->ApplyAction(choice);
            }

            double c_val = 0.0;
            if (!roll_state->IsTerminal()) {
              c_val = global_evaluator->Evaluate(*roll_state)[cs.player];
              if (mock_fail) {
                c_val += 0.10;
              }
            } else {
              c_val = roll_state->Returns()[cs.player] / utility_divisor;
            }
            succ_critic_sum += c_val;

            succ_true_sum += RunRawPolicyRollout(*roll_state, cs.player, global_evaluator.get(), roll_seed, utility_divisor);
          }

          successor_critic_vals.push_back(succ_critic_sum / roll_count);
          successor_true_vals.push_back(succ_true_sum / roll_count);
          mcts_q_vals.push_back(q_mcts);
        }

        // Calculate choice rank correlation, successor bias, and successor RMSE
        double corr = 0.0;
        if (mcts_q_vals.size() > 1 && successor_true_vals.size() > 1) {
          corr = SpearmanCorrelation(mcts_q_vals, successor_true_vals);
        }
        double bias_sum = 0.0;
        double rmse_sum = 0.0;
        for (size_t a_idx = 0; a_idx < successor_critic_vals.size(); ++a_idx) {
          double diff = successor_critic_vals[a_idx] - successor_true_vals[a_idx];
          bias_sum += diff;
          rmse_sum += diff * diff;
        }
        double succ_bias = successor_critic_vals.empty() ? 0.0 : (bias_sum / successor_critic_vals.size());
        double succ_rmse = successor_critic_vals.empty() ? 0.0 : std::sqrt(rmse_sum / successor_critic_vals.size());

        ev.successor_critic_values = successor_critic_vals;
        ev.successor_true_values = successor_true_vals;
        ev.choice_rank_correlation = corr;
        ev.successor_mean_bias = succ_bias;
        ev.successor_rmse = succ_rmse;

        ev.used_fallback = root_res.used_fallback;
        ev.fallback_reason = root_res.fallback_reason;
        ev.confidence_fallback = root_res.diagnostics.confidence_fallback;
        ev.mcts_overrode_raw = root_res.diagnostics.mcts_overrode_raw;
        ev.raw_reference_action = root_res.diagnostics.raw_reference_action;
        ev.mcts_proposed_action = root_res.diagnostics.mcts_proposed_action;
        ev.selected_action = root_res.diagnostics.selected_action;
        ev.stability_checkpoint_reached = root_res.diagnostics.stability_checkpoint_reached;
        ev.stability_agreement = root_res.diagnostics.stability_agreement;
        ev.pass_complete_search = root_res.diagnostics.pass_complete_search;
        ev.pass_min_actions = root_res.diagnostics.pass_min_actions;
        ev.pass_prior_mass = root_res.diagnostics.pass_prior_mass;
        ev.pass_meaningful_visits = root_res.diagnostics.pass_meaningful_visits;
        ev.pass_q_margin = root_res.diagnostics.pass_q_margin;
        ev.pass_stability = root_res.diagnostics.pass_stability;

        // Leaf-state diagnostics (evaluating actual leaf nodes visited during search)
        std::vector<double> leaf_critic_vals;
        std::vector<double> leaf_true_vals;

        const auto& sampled_leaf_states = root_res.diagnostics.sampled_leaf_states;
        for (size_t l_idx = 0; l_idx < sampled_leaf_states.size(); ++l_idx) {
          auto leaf_state = sampled_leaf_states[l_idx]->Clone();

          double leaf_critic_val = 0.0;
          if (!leaf_state->IsTerminal()) {
            leaf_critic_val = global_evaluator->Evaluate(*leaf_state)[cs.player];
            if (mock_fail) {
              leaf_critic_val += 0.10;
            }
          } else {
            leaf_critic_val = leaf_state->Returns()[cs.player] / utility_divisor;
          }

          double leaf_true_sum = 0.0;
          int roll_count = self_test
              ? 2
              : std::max(1, absl::GetFlag(FLAGS_diagnostic_rollouts));
          for (int r = 1; r <= roll_count; ++r) {
            uint64_t roll_seed = 500000 + 1000 * idx + 100 * l_idx + r;
            leaf_true_sum += RunRawPolicyRollout(*leaf_state, cs.player, global_evaluator.get(), roll_seed, utility_divisor);
          }
          double leaf_true_val = leaf_true_sum / roll_count;

          leaf_critic_vals.push_back(leaf_critic_val);
          leaf_true_vals.push_back(leaf_true_val);
        }

        double leaf_bias_sum = 0.0;
        double leaf_rmse_sum = 0.0;
        for (size_t l_idx = 0; l_idx < leaf_critic_vals.size(); ++l_idx) {
          double diff = leaf_critic_vals[l_idx] - leaf_true_vals[l_idx];
          leaf_bias_sum += diff;
          leaf_rmse_sum += diff * diff;
        }

        ev.leaf_critic_values = leaf_critic_vals;
        ev.leaf_true_values = leaf_true_vals;
        ev.leaf_mean_bias = leaf_critic_vals.empty() ? 0.0 : (leaf_bias_sum / leaf_critic_vals.size());
        ev.leaf_rmse = leaf_critic_vals.empty() ? 0.0 : std::sqrt(leaf_rmse_sum / leaf_critic_vals.size());
      }

      // Compute averages and standard errors
      double m_raw = q_raw_sum / rollouts;
      double m_search = q_search_sum / rollouts;
      double m_paired = (q_search_sum - q_raw_sum) / rollouts;

      ev.raw_returns = raw_returns_list;
      ev.search_returns = search_returns_list;
      ev.mean_raw_return = m_raw;
      ev.mean_search_return = m_search;
      ev.raw_return_std_err = CalculateStdErr(raw_returns_list, m_raw);
      ev.search_return_std_err = CalculateStdErr(search_returns_list, m_search);
      ev.mean_paired_advantage = m_paired;
      ev.paired_advantage_std_err = CalculateStdErr(paired_diffs, m_paired);

      if (total_searched_decisions > 0) {
        ev.mean_search_depth /= total_searched_decisions;
        ev.mean_terminal_leaf_fraction /= total_searched_decisions;
      }

      {
        std::lock_guard<std::mutex> lock(eval_mutex);
        q_raw_g2[idx] = m_raw;
        q_search_g2[idx] = m_search;

        int a_raw = ev.raw_action_sequence.empty() ? -1 : ev.raw_action_sequence.front();
        int a_search = ev.search_action_sequence.empty() ? -1 : ev.search_action_sequence.front();

        std::cout << absl::StrFormat("Gate 2 State %d: a_raw=%d, a_search=%d, q_raw=%.4f, q_search=%.4f, diff=%.4f\n",
                                     idx, a_raw, a_search, m_raw, m_search, m_search - m_raw) << std::flush;
      }
      completed_g2.fetch_add(1);
    }
  };

  std::atomic<bool> heartbeat_stop{false};
  std::thread heartbeat_thread([&]() {
    while (!heartbeat_stop.load()) {
      std::this_thread::sleep_for(std::chrono::seconds(5));
      if (heartbeat_stop.load()) break;
      int g1 = completed_g1.load();
      int g2 = completed_g2.load();
      std::cout << absl::StrFormat("[Heartbeat] Progress: Gate 1 %d/%d states, Gate 2 %d/%d states\n",
                                   g1, gate1_indices.size(), g2, gate2_indices.size()) << std::flush;
    }
  });

  // Launch Gate 1
  std::cout << "Starting thread workers for Gate 1...\n" << std::flush;
  std::vector<std::thread> workers_g1;
  if (gate1_cache_loaded) {
    completed_g1.store(gate1_indices.size());
  } else {
    for (int i = 0; i < num_threads; ++i) {
      workers_g1.emplace_back(worker_g1);
    }
    for (auto& w : workers_g1) {
      w.join();
    }
    for (size_t idx = 0; idx < gate1_indices.size(); ++idx) {
      bool need_layer1 = !gate1_returns_cached[idx];
      bool need_layer2 = !gate1_critic_cached[idx];
      if (need_layer1 || need_layer2) {
        v_true_g1[idx] = std::accumulate(
            gate1_returns[idx].begin(), gate1_returns[idx].end(), 0.0) /
            gate1_rollouts;
        if (split_cache.IsActive()) {
          split_cache.SaveGate1Shard(gate1_indices[idx], v_critic_g1[idx], gate1_returns[idx], false, need_layer1);
        }
      }
    }
    completed_g1.store(gate1_indices.size());
    if (!self_test && !gate1_cache_path.empty()) {
      open_spiel::json::Object cache;
      cache["protocol_version"] = "gate1-cache-v1";
      cache["model_checkpoint_hash"] = model_hash;
      cache["corpus_fingerprint"] = corpus_hash;
      cache["seed"] = seed;
      cache["rollouts"] = absl::GetFlag(FLAGS_rollouts);
      cache["binary_hash"] = open_spiel::ComputeFileSHA256("/proc/self/exe");
      open_spiel::json::Array critic_arr;
      open_spiel::json::Array true_arr;
      for (double v : v_critic_g1) critic_arr.push_back(v);
      for (double v : v_true_g1) true_arr.push_back(v);
      cache["v_critic"] = critic_arr;
      cache["v_true"] = true_arr;
      std::filesystem::create_directories(
          std::filesystem::path(gate1_cache_path).parent_path());
      std::string tmp_path = gate1_cache_path + ".tmp";
      std::ofstream cache_out(tmp_path);
      cache_out << open_spiel::json::ToString(cache, true) << "\n";
      cache_out.close();
      std::error_code ec;
      std::filesystem::rename(tmp_path, gate1_cache_path, ec);
      if (ec) {
        std::cerr << "Failed to publish Gate 1 cache: " << ec.message() << "\n";
      } else {
        std::cout << "Saved Gate 1 cache: " << gate1_cache_path << "\n";
      }
    }
  }

  // Launch Gate 2
  std::cout << "Gate 1 completed. Starting Gate 2...\n" << std::flush;
  std::vector<std::thread> workers_g2;
  for (int i = 0; i < num_threads; ++i) {
    workers_g2.emplace_back(worker_g2);
  }
  for (auto& w : workers_g2) {
    w.join();
  }

  if (g_cache_validation_failed) {
    std::cerr << "[ERROR] Cache validation failed cleanly (e.g. malformed JSON or NaN values in cache)." << std::endl;
    std::exit(2);
  }

  if (!self_test && absl::GetFlag(FLAGS_choice_only_gate2) &&
      !choice_outcome_cache_loaded && !choice_cache_path.empty()) {
    open_spiel::json::Object cache;
    cache["protocol_version"] = "choice-outcomes-v1";
    cache["model_checkpoint_hash"] = model_hash;
    cache["corpus_fingerprint"] = corpus_hash;
    cache["rollouts"] = absl::GetFlag(FLAGS_rollouts);
    open_spiel::json::Object config;
    config["opponent_mode"] = absl::GetFlag(FLAGS_opponent_mode);
    config["opponent_temperature"] = absl::GetFlag(FLAGS_opponent_temperature);
    config["puct_c"] = absl::GetFlag(FLAGS_puct_c);
    config["max_simulations"] = absl::GetFlag(FLAGS_max_simulations);
    config["nonlinear_value_head"] = absl::GetFlag(FLAGS_nonlinear_value_head);
    config["conservative_override_enabled"] = absl::GetFlag(FLAGS_conservative_override_enabled);
    config["conservative_covered_prior_threshold"] = absl::GetFlag(FLAGS_conservative_covered_prior_threshold);
    config["conservative_meaningful_visit_threshold"] = absl::GetFlag(FLAGS_conservative_meaningful_visit_threshold);
    config["conservative_q_margin_threshold"] = absl::GetFlag(FLAGS_conservative_q_margin_threshold);
    config["conservative_stability_checkpoint_fraction"] = absl::GetFlag(FLAGS_conservative_stability_checkpoint_fraction);
    config["conservative_continuation_overrides_disabled"] = absl::GetFlag(FLAGS_conservative_continuation_overrides_disabled);
    config["fixed_continuation_reserve"] = absl::GetFlag(FLAGS_fixed_continuation_reserve);
    config["purchase_combat_budget"] = absl::GetFlag(FLAGS_purchase_combat_budget);
    config["live_continuation_reserve_seconds"] = absl::GetFlag(FLAGS_live_continuation_reserve_seconds);
    cache["config"] = config;
    cache["binary_hash"] = open_spiel::ComputeFileSHA256("/proc/self/exe");
    open_spiel::json::Array states_arr;
    for (const auto& entry : choice_outcome_cache) {
      open_spiel::json::Object state_obj;
      state_obj["corpus_index"] = entry.corpus_index;
      open_spiel::json::Array actions_arr;
      open_spiel::json::Array returns_arr;
      open_spiel::json::Array critics_arr;
      for (size_t a_idx = 0; a_idx < entry.actions.size(); ++a_idx) {
        actions_arr.push_back(static_cast<int64_t>(entry.actions[a_idx]));
        critics_arr.push_back(entry.successor_critic_values[a_idx]);
        open_spiel::json::Array action_returns;
        for (double value : entry.returns[a_idx]) action_returns.push_back(value);
        returns_arr.push_back(action_returns);
      }
      state_obj["actions"] = actions_arr;
      state_obj["returns"] = returns_arr;
      state_obj["successor_critic_values"] = critics_arr;
      states_arr.push_back(state_obj);
    }
    cache["states"] = states_arr;
    std::filesystem::create_directories(
        std::filesystem::path(choice_cache_path).parent_path());
    std::string tmp_path = choice_cache_path + ".tmp";
    std::ofstream cache_out(tmp_path);
    cache_out << open_spiel::json::ToString(cache, true) << "\n";
    cache_out.close();
    std::error_code ec;
    std::filesystem::rename(tmp_path, choice_cache_path, ec);
    if (ec) {
      std::cerr << "Failed to publish choice rollout cache: "
                << ec.message() << "\n";
    } else {
      std::cout << "Saved choice rollout cache: " << choice_cache_path << "\n";
    }
  }

  // Stop heartbeat thread
  heartbeat_stop = true;
  if (heartbeat_thread.joinable()) {
    heartbeat_thread.join();
  }

  std::cout << "Fidelity simulation rollouts complete.\n";

  // ==========================================
  // Report Gate 1 (Critic Fidelity)
  // ==========================================
  double total_bias = 0.0;
  double total_sq_err = 0.0;
  for (size_t i = 0; i < gate1_indices.size(); ++i) {
    double diff = v_critic_g1[i] - v_true_g1[i];
    total_bias += diff;
    total_sq_err += diff * diff;
  }
  double mean_bias = total_bias / gate1_indices.size();
  double rmse = std::sqrt(total_sq_err / gate1_indices.size());
  double rank_corr = SpearmanCorrelation(v_critic_g1, v_true_g1);

  std::cout << "\n============================================\n";
  std::cout << "      GATE 1: CRITIC FIDELITY REPORT        \n";
  std::cout << "============================================\n";
  std::cout << absl::StrFormat("States Evaluated:     %d\n", gate1_indices.size());
  std::cout << absl::StrFormat("Normalized Mean Bias: %.4f (Limit <= 0.05)\n", mean_bias);
  std::cout << absl::StrFormat("RMSE:                 %.4f\n", rmse);
  std::cout << absl::StrFormat("Rank Correlation:     %.4f\n", rank_corr);

  // Group by round
  std::vector<int> round_counts(6, 0);
  std::vector<double> round_bias(6, 0.0);
  std::vector<double> round_rmse(6, 0.0);

  for (size_t i = 0; i < gate1_indices.size(); ++i) {
    int r = corpus[gate1_indices[i]].round;
    if (r >= 1 && r <= 5) {
      round_counts[r]++;
      double diff = v_critic_g1[i] - v_true_g1[i];
      round_bias[r] += diff;
      round_rmse[r] += diff * diff;
    }
  }

  std::cout << "\nCalibration by Round:\n";
  for (int r = 1; r <= 5; ++r) {
    if (round_counts[r] > 0) {
      double rb = round_bias[r] / round_counts[r];
      double rrmse = std::sqrt(round_rmse[r] / round_counts[r]);
      std::cout << absl::StrFormat("  Round %d: Count=%2d, Bias=% .4f, RMSE=%.4f\n", r, round_counts[r], rb, rrmse);
    }
  }

  // Reliability tables (5 predicted bins)
  std::vector<int> bin_counts(5, 0);
  std::vector<double> bin_pred_sum(5, 0.0);
  std::vector<double> bin_act_sum(5, 0.0);

  for (size_t i = 0; i < gate1_indices.size(); ++i) {
    double pred = v_critic_g1[i];
    int bin_idx = 0;
    if (pred < -0.6) bin_idx = 0;
    else if (pred < -0.2) bin_idx = 1;
    else if (pred < 0.2) bin_idx = 2;
    else if (pred < 0.6) bin_idx = 3;
    else bin_idx = 4;

    bin_counts[bin_idx]++;
    bin_pred_sum[bin_idx] += pred;
    bin_act_sum[bin_idx] += v_true_g1[i];
  }

  std::cout << "\nBinned Predicted vs. Actual Calibration Table:\n";
  std::cout << "  Bin boundaries   | Count | Mean Predicted | Mean Actual | Mean Bias \n";
  std::cout << "  -----------------+-------+----------------+-------------+-----------\n";
  std::vector<std::string> bin_labels = {"[-1.0, -0.6)", "[-0.6, -0.2)", "[-0.2,  0.2)", "[ 0.2,  0.6)", "[ 0.6,  1.0]"};
  for (int b = 0; b < 5; ++b) {
    if (bin_counts[b] > 0) {
      double mp = bin_pred_sum[b] / bin_counts[b];
      double ma = bin_act_sum[b] / bin_counts[b];
      std::cout << absl::StrFormat("  %-16s | %5d | % 14.4f | %11.4f | % .4f\n", bin_labels[b], bin_counts[b], mp, ma, mp - ma);
    } else {
      std::cout << absl::StrFormat("  %-16s |     0 |            N/A |         N/A |       N/A\n", bin_labels[b]);
    }
  }

  // ==========================================
  // Report Gate 2 (Choice Advantages)
  // ==========================================
  double mean_adv = 0.0;
  double legacy_mean_adv = 0.0;
  double bootstrap_lcb = 0.0;
  double hierarchical_lcb = 0.0;
  bool bootstrap_passed = true;
  std::vector<double> boot_means(10000, 0.0);
  std::vector<double> hierarchical_means(10000, 0.0);
  std::vector<int> unique_episodes;

  if (!gate2_indices.empty()) {
    std::vector<double> paired_advantages(gate2_indices.size(), 0.0);
    double total_adv = 0.0;
    double total_legacy_adv = 0.0;

    for (size_t i = 0; i < gate2_indices.size(); ++i) {
      paired_advantages[i] = gate2_evidence[i].mean_paired_advantage;
      total_adv += paired_advantages[i];
      total_legacy_adv += gate2_evidence[i].diagnostic_legacy_paired_advantage;
    }
    mean_adv = total_adv / gate2_indices.size();
    legacy_mean_adv = total_legacy_adv / gate2_indices.size();

    // Group opportunity states by cs.episode_id with deterministic sorting
    std::unordered_map<int, std::vector<size_t>> episode_to_indices;
    for (size_t i = 0; i < gate2_indices.size(); ++i) {
      int ep_id = gate2_evidence[i].episode_id;
      if (episode_to_indices.find(ep_id) == episode_to_indices.end()) {
        unique_episodes.push_back(ep_id);
      }
      episode_to_indices[ep_id].push_back(i);
    }
    std::sort(unique_episodes.begin(), unique_episodes.end());

    std::cout << "\n============================================\n";
    std::cout << "      GATE 2: CHOICE ADVANTAGES REPORT      \n";
    std::cout << "============================================\n";
    std::cout << absl::StrFormat("Opportunity States:      %d\n", gate2_indices.size());
    std::cout << absl::StrFormat("Effective Episodes:      %d\n", unique_episodes.size());
    std::cout << absl::StrFormat("Search vs. Raw mean Q:   % .4f\n", mean_adv);
    std::cout << absl::StrFormat("Legacy (sampled) mean Q: % .4f\n", legacy_mean_adv);

    int num_boots = 10000;
    std::mt19937 boot_rng(bootstrap_seed);
    std::uniform_int_distribution<int> ep_dist(0, unique_episodes.size() - 1);

    for (int b = 0; b < num_boots; ++b) {
      // 1. Episode cluster bootstrap
      double cluster_sum = 0.0;
      size_t cluster_count = 0;

      // 2. Hierarchical bootstrap
      double hierarchical_sum = 0.0;
      size_t hierarchical_count = 0;

      for (size_t k = 0; k < unique_episodes.size(); ++k) {
        int ep_idx = ep_dist(boot_rng);
        int ep_id = unique_episodes[ep_idx];
        const auto& state_idxs = episode_to_indices[ep_id];

        // Cluster bootstrap adds all states of the sampled episode
        for (size_t idx : state_idxs) {
          cluster_sum += paired_advantages[idx];
          cluster_count++;
        }

        // Hierarchical bootstrap samples states within the sampled episode with replacement
        std::uniform_int_distribution<int> state_dist(0, state_idxs.size() - 1);
        for (size_t s_idx = 0; s_idx < state_idxs.size(); ++s_idx) {
          int s_pos = state_dist(boot_rng);
          hierarchical_sum += paired_advantages[state_idxs[s_pos]];
          hierarchical_count++;
        }
      }

      boot_means[b] = cluster_count > 0 ? (cluster_sum / cluster_count) : 0.0;
      hierarchical_means[b] = hierarchical_count > 0 ? (hierarchical_sum / hierarchical_count) : 0.0;
    }

    // Find cluster LCB
    std::vector<double> sorted_boot_means = boot_means;
    std::sort(sorted_boot_means.begin(), sorted_boot_means.end());
    bootstrap_lcb = sorted_boot_means[static_cast<int>(0.05 * num_boots)];

    // Find hierarchical LCB
    std::vector<double> sorted_hierarchical_means = hierarchical_means;
    std::sort(sorted_hierarchical_means.begin(), sorted_hierarchical_means.end());
    hierarchical_lcb = sorted_hierarchical_means[static_cast<int>(0.05 * num_boots)];

    if (mock_fail) {
      bootstrap_lcb = -0.01;
      hierarchical_lcb = -0.01;
    }

    std::cout << absl::StrFormat("95%% Bootstrap LCB:        % .4f (Limit > 0.0)\n", bootstrap_lcb);
    std::cout << absl::StrFormat("95%% Hierarchical LCB:     % .4f\n", hierarchical_lcb);
    std::cout << "============================================\n";
    bootstrap_passed = (bootstrap_lcb > 0.0);
  } else {
    std::cout << "\n============================================\n";
    std::cout << "      GATE 2: SKIPPED (NO OPP STATES)       \n";
    std::cout << "============================================\n";
  }

  bool bias_passed = (std::abs(mean_bias) <= 0.05);

  if (self_test) {
    if (!mock_fail) {
      bias_passed = true;
      bootstrap_passed = true;
      bootstrap_lcb = 0.01;
    }
  }

  if (!bias_passed) {
    std::cerr << absl::StrFormat("\nCRITICAL FAILURE: Critic absolute mean bias %.4f exceeds 0.05 limit!\n", std::abs(mean_bias));
  }
  if (!gate2_indices.empty() && !bootstrap_passed) {
    std::cerr << absl::StrFormat("\nCRITICAL FAILURE: 95%% Bootstrap LCB %.4f is not positive!\n", bootstrap_lcb);
  }

  if (split_cache.IsActive()) {
    split_cache.WriteManifest(gate1_indices, gate2_indices);
  }

  // Save structured JSON report if requested
  std::string report_path = absl::GetFlag(FLAGS_report_path);
  if (!report_path.empty()) {
    std::filesystem::create_directories(std::filesystem::path(report_path).parent_path());
    std::string report_tmp_path = report_path + ".tmp";
    std::ofstream out(report_tmp_path);
    if (out) {
      open_spiel::json::Object root;
      root["command_line"] = absl::StrJoin(std::vector<std::string>(argv, argv + argc), " ");
      root["self_test"] = self_test;

      open_spiel::json::Object config;
      config["model_checkpoint"] = model_checkpoint;
      config["corpus_path"] = corpus_path;
      config["seed"] = absl::GetFlag(FLAGS_seed);
      config["search_seed"] = static_cast<int64_t>(search_seed);
      config["raw_policy_seed"] = static_cast<int64_t>(raw_policy_seed);
      config["rollout_seed_base"] = static_cast<int64_t>(rollout_seed_base);
      config["chance_seed_base"] = static_cast<int64_t>(chance_seed_base);
      config["bootstrap_seed"] = static_cast<int64_t>(bootstrap_seed);
      config["hidden_dim"] = absl::GetFlag(FLAGS_hidden_dim);
      config["num_blocks"] = absl::GetFlag(FLAGS_num_blocks);
      config["max_simulations"] = absl::GetFlag(FLAGS_max_simulations);
      config["puct_c"] = absl::GetFlag(FLAGS_puct_c);
      config["opponent_mode"] = absl::GetFlag(FLAGS_opponent_mode);
      config["opponent_temperature"] = absl::GetFlag(FLAGS_opponent_temperature);
      config["utility_divisor"] = absl::GetFlag(FLAGS_utility_divisor);
      config["threads"] = absl::GetFlag(FLAGS_threads);
      config["mock_miscalibrated_critic"] = absl::GetFlag(FLAGS_mock_miscalibrated_critic);
      config["self_test"] = absl::GetFlag(FLAGS_self_test);
      config["strict_v2_validation"] = absl::GetFlag(FLAGS_strict_v2_validation);
      config["fixed_continuation_reserve"] = absl::GetFlag(FLAGS_fixed_continuation_reserve);
      config["purchase_combat_budget"] = absl::GetFlag(FLAGS_purchase_combat_budget);
      config["root_prior_temperature"] = absl::GetFlag(FLAGS_root_prior_temperature);
      config["rollouts"] = absl::GetFlag(FLAGS_rollouts);
      config["relative_time_budget_ms"] = absl::GetFlag(FLAGS_relative_time_budget_ms);
      config["gate1_only"] = absl::GetFlag(FLAGS_gate1_only);
      config["allow_any_opportunity_count"] = absl::GetFlag(FLAGS_allow_any_opportunity_count);
      config["live_continuation_reserve_seconds"] = absl::GetFlag(FLAGS_live_continuation_reserve_seconds);
      config["nonlinear_value_head"] = absl::GetFlag(FLAGS_nonlinear_value_head);
      config["choice_only_gate2"] = absl::GetFlag(FLAGS_choice_only_gate2);
      config["diagnostic_rollouts"] = absl::GetFlag(FLAGS_diagnostic_rollouts);
      config["gate1_cache_path"] = absl::GetFlag(FLAGS_gate1_cache_path);
      config["choice_rollout_cache_path"] = absl::GetFlag(FLAGS_choice_rollout_cache_path);
      config["conservative_override_enabled"] = absl::GetFlag(FLAGS_conservative_override_enabled);
      config["conservative_covered_prior_threshold"] = absl::GetFlag(FLAGS_conservative_covered_prior_threshold);
      config["conservative_meaningful_visit_threshold"] = absl::GetFlag(FLAGS_conservative_meaningful_visit_threshold);
      config["conservative_q_margin_threshold"] = absl::GetFlag(FLAGS_conservative_q_margin_threshold);
      config["conservative_stability_checkpoint_fraction"] = absl::GetFlag(FLAGS_conservative_stability_checkpoint_fraction);
      config["conservative_continuation_overrides_disabled"] = absl::GetFlag(FLAGS_conservative_continuation_overrides_disabled);
      root["config"] = config;

      if (batched_eval) {
        EvaluatorStats stats = batched_eval->GetStats();
        open_spiel::json::Object evaluator_stats;
        evaluator_stats["requests"] = static_cast<int64_t>(stats.requests);
        evaluator_stats["batches"] = static_cast<int64_t>(stats.batches);
        evaluator_stats["max_batch_size"] = static_cast<int64_t>(stats.max_batch_size);
        evaluator_stats["avg_batch_size"] = stats.avg_batch_size;
        root["evaluator_stats"] = evaluator_stats;
      }

      try {
        root["corpus_fingerprint"] = open_spiel::ComputeFileSHA256(corpus_path);
      } catch (...) {
        root["corpus_fingerprint"] = "unknown";
      }

      if (!self_test) {
        root["model_checkpoint_path"] = model_checkpoint;
        try {
          root["model_checkpoint_hash"] = open_spiel::ComputeFileSHA256(model_checkpoint);
        } catch (...) {
          root["model_checkpoint_hash"] = "unknown";
        }
      } else {
        root["model_checkpoint_path"] = "self_test";
        root["model_checkpoint_hash"] = "self_test";
      }

      try {
        root["binary_hash"] = open_spiel::ComputeFileSHA256(argv[0]);
      } catch (...) {
        root["binary_hash"] = "unknown";
      }
      root["search_protocol_version"] = "session-v4";

      root["passed"] = bias_passed && (gate2_indices.empty() || bootstrap_passed);

      open_spiel::json::Object g1;
      g1["states_evaluated"] = static_cast<int64_t>(gate1_indices.size());
      g1["mean_bias"] = mean_bias;
      g1["rmse"] = rmse;
      g1["rank_correlation"] = rank_corr;
      root["gate1"] = g1;

      open_spiel::json::Object g2;
      g2["opportunity_states"] = static_cast<int64_t>(gate2_indices.size());
      g2["search_vs_raw_mean_q"] = mean_adv;
      g2["point_estimate"] = mean_adv;
      g2["bootstrap_lcb"] = bootstrap_lcb;
      g2["legacy_search_vs_raw_mean_q"] = legacy_mean_adv;
      g2["effective_episode_count"] = static_cast<int64_t>(unique_episodes.size());
      g2["hierarchical_lcb"] = hierarchical_lcb;

      open_spiel::json::Array boot_reps_arr;
      for (double val : boot_means) boot_reps_arr.push_back(val);
      g2["bootstrap_replicates"] = boot_reps_arr;

      open_spiel::json::Array hier_reps_arr;
      for (double val : hierarchical_means) hier_reps_arr.push_back(val);
      g2["hierarchical_replicates"] = hier_reps_arr;

      open_spiel::json::Array evidence_arr;
      for (const auto& ev : gate2_evidence) {
        open_spiel::json::Object ev_obj;
        ev_obj["corpus_index"] = static_cast<int64_t>(ev.corpus_index);
        ev_obj["episode_id"] = static_cast<int64_t>(ev.episode_id);
        ev_obj["history_hash"] = ev.history_hash;
        ev_obj["seat"] = static_cast<int64_t>(ev.seat);
        ev_obj["round"] = static_cast<int64_t>(ev.round);
        ev_obj["role"] = ev.role;

        open_spiel::json::Array raw_act_arr;
        for (int a : ev.raw_action_sequence) raw_act_arr.push_back(static_cast<int64_t>(a));
        ev_obj["raw_action_sequence"] = raw_act_arr;

        open_spiel::json::Array search_act_arr;
        for (int a : ev.search_action_sequence) search_act_arr.push_back(static_cast<int64_t>(a));
        ev_obj["search_action_sequence"] = search_act_arr;

        open_spiel::json::Array raw_legal_acts_arr;
        for (const auto& acts : ev.raw_legal_actions) {
          open_spiel::json::Array inner;
          for (int a : acts) inner.push_back(static_cast<int64_t>(a));
          raw_legal_acts_arr.push_back(inner);
        }
        ev_obj["raw_legal_actions"] = raw_legal_acts_arr;

        open_spiel::json::Array raw_roles_arr;
        for (const auto& r : ev.raw_decision_roles) raw_roles_arr.push_back(r);
        ev_obj["raw_decision_roles"] = raw_roles_arr;

        open_spiel::json::Array search_legal_acts_arr;
        for (const auto& acts : ev.search_legal_actions) {
          open_spiel::json::Array inner;
          for (int a : acts) inner.push_back(static_cast<int64_t>(a));
          search_legal_acts_arr.push_back(inner);
        }
        ev_obj["search_legal_actions"] = search_legal_acts_arr;

        open_spiel::json::Array search_roles_arr;
        for (const auto& r : ev.search_decision_roles) search_roles_arr.push_back(r);
        ev_obj["search_decision_roles"] = search_roles_arr;

        open_spiel::json::Array raw_priors_arr;
        for (double p : ev.raw_priors) raw_priors_arr.push_back(p);
        ev_obj["raw_priors"] = raw_priors_arr;

        open_spiel::json::Array search_visits_arr;
        for (int v : ev.search_visits) search_visits_arr.push_back(static_cast<int64_t>(v));
        ev_obj["search_visits"] = search_visits_arr;

        open_spiel::json::Array root_q_arr;
        for (double q : ev.root_q_values) root_q_arr.push_back(q);
        ev_obj["root_q_values"] = root_q_arr;

        ev_obj["selected_action_rank"] = static_cast<int64_t>(ev.selected_action_rank);

        open_spiel::json::Array raw_returns_arr;
        for (double r : ev.raw_returns) raw_returns_arr.push_back(r);
        ev_obj["raw_returns"] = raw_returns_arr;

        open_spiel::json::Array search_returns_arr;
        for (double r : ev.search_returns) search_returns_arr.push_back(r);
        ev_obj["search_returns"] = search_returns_arr;

        ev_obj["mean_raw_return"] = ev.mean_raw_return;
        ev_obj["mean_search_return"] = ev.mean_search_return;
        ev_obj["raw_return_std_err"] = ev.raw_return_std_err;
        ev_obj["search_return_std_err"] = ev.search_return_std_err;
        ev_obj["mean_paired_advantage"] = ev.mean_paired_advantage;
        ev_obj["paired_advantage_std_err"] = ev.paired_advantage_std_err;
        ev_obj["diagnostic_legacy_raw_return"] = ev.diagnostic_legacy_raw_return;
        ev_obj["diagnostic_legacy_paired_advantage"] = ev.diagnostic_legacy_paired_advantage;

        ev_obj["total_simulations_used"] = static_cast<int64_t>(ev.total_simulations_used);
        ev_obj["total_re_root_hits"] = static_cast<int64_t>(ev.total_re_root_hits);
        ev_obj["total_re_root_misses"] = static_cast<int64_t>(ev.total_re_root_misses);
        ev_obj["mean_search_depth"] = ev.mean_search_depth;
        ev_obj["mean_terminal_leaf_fraction"] = ev.mean_terminal_leaf_fraction;
        ev_obj["total_inference_count"] = static_cast<int64_t>(ev.total_inference_count);
        ev_obj["primary_action_changed"] = ev.primary_action_changed;
        ev_obj["continuation_action_changed"] = ev.continuation_action_changed;

        ev_obj["used_fallback"] = ev.used_fallback;
        ev_obj["fallback_reason"] = ev.fallback_reason;
        ev_obj["confidence_fallback"] = ev.confidence_fallback;
        ev_obj["mcts_overrode_raw"] = ev.mcts_overrode_raw;
        ev_obj["raw_reference_action"] = static_cast<int64_t>(ev.raw_reference_action);
        ev_obj["mcts_proposed_action"] = static_cast<int64_t>(ev.mcts_proposed_action);
        ev_obj["selected_action"] = static_cast<int64_t>(ev.selected_action);
        ev_obj["stability_checkpoint_reached"] = ev.stability_checkpoint_reached;
        ev_obj["stability_agreement"] = ev.stability_agreement;
        ev_obj["pass_complete_search"] = ev.pass_complete_search;
        ev_obj["pass_min_actions"] = ev.pass_min_actions;
        ev_obj["pass_prior_mass"] = ev.pass_prior_mass;
        ev_obj["pass_meaningful_visits"] = ev.pass_meaningful_visits;
        ev_obj["pass_q_margin"] = ev.pass_q_margin;
        ev_obj["pass_stability"] = ev.pass_stability;

        open_spiel::json::Array root_actions_arr;
        for (int a : ev.root_action_ids) root_actions_arr.push_back(static_cast<int64_t>(a));
        ev_obj["root_action_ids"] = root_actions_arr;

        ev_obj["session_id"] = ev.session_id;
        ev_obj["inherited_visits"] = static_cast<int64_t>(ev.inherited_visits);
        ev_obj["initial_simulations"] = static_cast<int64_t>(ev.initial_simulations);
        ev_obj["continuation_simulations"] = static_cast<int64_t>(ev.continuation_simulations);

        open_spiel::json::Array successor_critic_arr;
        for (double v : ev.successor_critic_values) successor_critic_arr.push_back(v);
        ev_obj["successor_critic_values"] = successor_critic_arr;

        open_spiel::json::Array successor_true_arr;
        for (double v : ev.successor_true_values) successor_true_arr.push_back(v);
        ev_obj["successor_true_values"] = successor_true_arr;

        ev_obj["choice_rank_correlation"] = ev.choice_rank_correlation;
        ev_obj["successor_mean_bias"] = ev.successor_mean_bias;
        ev_obj["successor_rmse"] = ev.successor_rmse;

        open_spiel::json::Array leaf_critic_arr;
        for (double v : ev.leaf_critic_values) leaf_critic_arr.push_back(v);
        ev_obj["leaf_critic_values"] = leaf_critic_arr;

        open_spiel::json::Array leaf_true_arr;
        for (double v : ev.leaf_true_values) leaf_true_arr.push_back(v);
        ev_obj["leaf_true_values"] = leaf_true_arr;

        ev_obj["leaf_mean_bias"] = ev.leaf_mean_bias;
        ev_obj["leaf_rmse"] = ev.leaf_rmse;

        open_spiel::json::Array rollouts_arr;
        for (const auto& r : ev.rollouts) {
          open_spiel::json::Object r_obj;
          r_obj["rollout_index"] = static_cast<int64_t>(r.rollout_index);
          r_obj["session_id"] = r.session_id;

          open_spiel::json::Array raw_act_arr2;
          for (int a : r.raw_action_sequence) raw_act_arr2.push_back(static_cast<int64_t>(a));
          r_obj["raw_action_sequence"] = raw_act_arr2;

          open_spiel::json::Array search_act_arr2;
          for (int a : r.search_action_sequence) search_act_arr2.push_back(static_cast<int64_t>(a));
          r_obj["search_action_sequence"] = search_act_arr2;

          open_spiel::json::Array raw_legal_acts_arr2;
          for (const auto& acts : r.raw_legal_actions) {
            open_spiel::json::Array inner;
            for (int a : acts) inner.push_back(static_cast<int64_t>(a));
            raw_legal_acts_arr2.push_back(inner);
          }
          r_obj["raw_legal_actions"] = raw_legal_acts_arr2;

          open_spiel::json::Array raw_roles_arr2;
          for (const auto& role : r.raw_decision_roles) raw_roles_arr2.push_back(role);
          r_obj["raw_decision_roles"] = raw_roles_arr2;

          open_spiel::json::Array search_legal_acts_arr2;
          for (const auto& acts : r.search_legal_actions) {
            open_spiel::json::Array inner;
            for (int a : acts) inner.push_back(static_cast<int64_t>(a));
            search_legal_acts_arr2.push_back(inner);
          }
          r_obj["search_legal_actions"] = search_legal_acts_arr2;

          open_spiel::json::Array search_roles_arr2;
          for (const auto& role : r.search_decision_roles) search_roles_arr2.push_back(role);
          r_obj["search_decision_roles"] = search_roles_arr2;

          open_spiel::json::Array raw_priors_arr2;
          for (double p : r.raw_priors) raw_priors_arr2.push_back(p);
          r_obj["raw_priors"] = raw_priors_arr2;

          open_spiel::json::Array search_visits_arr2;
          for (int v : r.search_visits) search_visits_arr2.push_back(static_cast<int64_t>(v));
          r_obj["search_visits"] = search_visits_arr2;

          open_spiel::json::Array root_q_arr2;
          for (double q : r.root_q_values) root_q_arr2.push_back(q);
          r_obj["root_q_values"] = root_q_arr2;

          r_obj["selected_action_rank"] = static_cast<int64_t>(r.selected_action_rank);
          r_obj["raw_return"] = r.raw_return;
          r_obj["search_return"] = r.search_return;
          r_obj["paired_advantage"] = r.paired_advantage;
          r_obj["simulations_used"] = static_cast<int64_t>(r.simulations_used);
          r_obj["initial_simulations"] = static_cast<int64_t>(r.initial_simulations);
          r_obj["continuation_simulations"] = static_cast<int64_t>(r.continuation_simulations);
          r_obj["short_window_simulations"] = static_cast<int64_t>(r.short_window_simulations);
          r_obj["re_root_hits"] = static_cast<int64_t>(r.re_root_hits);
          r_obj["re_root_misses"] = static_cast<int64_t>(r.re_root_misses);
          r_obj["search_depth"] = r.search_depth;
          r_obj["terminal_leaf_fraction"] = r.terminal_leaf_fraction;
          r_obj["inference_count"] = static_cast<int64_t>(r.inference_count);
          r_obj["inherited_visits"] = static_cast<int64_t>(r.inherited_visits);

          open_spiel::json::Array root_actions_arr2;
          for (int a : r.root_action_ids) root_actions_arr2.push_back(static_cast<int64_t>(a));
          r_obj["root_action_ids"] = root_actions_arr2;

          r_obj["primary_action_changed"] = r.primary_action_changed;
          r_obj["continuation_action_changed"] = r.continuation_action_changed;
          r_obj["search_decisions_count"] = static_cast<int64_t>(r.search_decisions_count);

          rollouts_arr.push_back(r_obj);
        }
        ev_obj["rollouts"] = rollouts_arr;

        evidence_arr.push_back(ev_obj);
      }
      g2["evidence"] = evidence_arr;
      root["gate2"] = g2;

      out << open_spiel::json::ToString(root, true) << "\n";
      out.close();
      std::error_code rename_error;
      std::filesystem::rename(report_tmp_path, report_path, rename_error);
      if (rename_error) {
        std::cerr << "Failed to atomically publish report: "
                  << rename_error.message() << "\n";
        return 1;
      }
      std::cout << "Saved structured JSON report to: " << report_path << "\n";
    }
  }

  if (self_test && !mock_fail) {
    std::cout << "\nSUCCESS: Self-test completed successfully.\n";
    return 0;
  }

  if (bias_passed && (gate2_indices.empty() || bootstrap_passed)) {
    std::cout << "\nSUCCESS: Critic fidelity gate PASSED.\n";
    return 0;
  } else {
    std::cerr << "\nFAILURE: Gating checks FAILED. Do not begin search training.\n";
    return 2;
  }
}
