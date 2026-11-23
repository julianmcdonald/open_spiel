#ifndef OPEN_SPIEL_EXAMPLES_DUNE_FROZEN_POLICY_DIAG_TYPES_H_
#define OPEN_SPIEL_EXAMPLES_DUNE_FROZEN_POLICY_DIAG_TYPES_H_

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <numeric>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "open_spiel/spiel.h"
#include "open_spiel/games/dune_imperium/dune_imperium.h"
#include "open_spiel/games/dune_imperium/dune_imperium_common.h"
#include "open_spiel/games/dune_imperium/dune_imperium_content.h"
#include "open_spiel/games/dune_imperium/dune_imperium_content_generated.h"
#include "open_spiel/games/dune_imperium/dune_imperium_test_utils.h"
#include "open_spiel/utils/json.h"

#include "dune_seed_utils.h"
#include "dune_eval_action_selection.h"

namespace open_spiel {
namespace dune_diagnostics {

using dune_imperium::AgentSpaceNameForAction;
using dune_imperium::BuildStateAtFirstAgentTurn;
using dune_imperium::ContainsAction;
using dune_imperium::DuneImperiumState;
using dune_imperium::Faction;
using dune_imperium::FindImperiumCardById;
using dune_imperium::FindImperiumCardByName;
using dune_imperium::GamePhase;
using dune_imperium::IntrigueType;
using dune_imperium::IsAgentSpaceAction;
using dune_imperium::kActionAgentPass;
using dune_imperium::kActionAgentSpaceArrakeen;
using dune_imperium::kActionAgentSpaceCarthag;
using dune_imperium::kActionAgentSpaceConspire;
using dune_imperium::kActionAgentSpaceFoldspace;
using dune_imperium::kActionAgentSpaceHardyWarriors;
using dune_imperium::kActionAgentSpaceHeighliner;
using dune_imperium::kActionAgentSpaceHighCouncil;
using dune_imperium::kActionAgentSpaceImperialBasin;
using dune_imperium::kActionAgentSpaceInterstellarShipping;
using dune_imperium::kActionAgentSpaceMentat;
using dune_imperium::kActionAgentSpaceSecrets;
using dune_imperium::kActionAgentSpaceSelectiveBreeding;
using dune_imperium::kActionAgentSpaceSietchTabr;
using dune_imperium::kActionAgentSpaceSmuggling;
using dune_imperium::kActionAgentSpaceStillsuits;
using dune_imperium::kActionAgentSpaceSwordmaster;
using dune_imperium::kActionAgentSpaceWealth;
using dune_imperium::kActionEndTurn;
using dune_imperium::kActionPlayAgentSolo;
using dune_imperium::kActionPlayIntriguePlotCard0;
using dune_imperium::kActionReveal;
using dune_imperium::kActionSelectAgentCard0;
using dune_imperium::kActionSelectGraftPartner0;
using dune_imperium::kCardArrakisLiaison;
using dune_imperium::kCardConvincingArgument;
using dune_imperium::kCardDagger;
using dune_imperium::kCardDiplomacy;
using dune_imperium::kCardDuneTheDesertPlanet;
using dune_imperium::kCardReconnaissance;
using dune_imperium::kCardSeekAllies;
using dune_imperium::kCardSignetRing;
using dune_imperium::kIntrigueSecretsOfTheSisterhood;
using dune_imperium::kIntrigueWaterPeddlersUnion;
using dune_imperium::kIntrigueWindfall;
using dune_imperium::kNumPlayers;
using dune_imperium::LeaderId;

enum class AdvanceStatus {
  kSuccessDecision,
  kSuccessTerminal,
  kErrorStepLimit,
  kErrorEmptyLegal
};

inline const char* AdvanceStatusToString(AdvanceStatus status) {
  switch (status) {
    case AdvanceStatus::kSuccessDecision: return "SuccessDecision";
    case AdvanceStatus::kSuccessTerminal: return "SuccessTerminal";
    case AdvanceStatus::kErrorStepLimit: return "ErrorStepLimit";
    case AdvanceStatus::kErrorEmptyLegal: return "ErrorEmptyLegal";
  }
  return "UnknownStatus";
}

inline std::ostream& operator<<(std::ostream& os, AdvanceStatus status) {
  return os << AdvanceStatusToString(status);
}

// Production chance helper that samples chance outcomes according to true probabilities
// using chance_rng and handles deterministic acknowledgments without consuming chance draws.
inline AdvanceStatus AdvanceThroughChanceAndAcks(
    State* state,
    std::mt19937_64& chance_rng,
    std::vector<Action>* history = nullptr,
    int max_steps = 500) {
  int steps = 0;
  while (!state->IsTerminal() && steps < max_steps) {
    if (state->IsChanceNode()) {
      auto outcomes = state->ChanceOutcomes();
      if (outcomes.empty()) {
        return AdvanceStatus::kErrorEmptyLegal;
      }
      Action action = SampleAction(outcomes, chance_rng).first;
      state->ApplyAction(action);
      if (history != nullptr) {
        history->push_back(action);
      }
      steps++;
      continue;
    }
    auto legal = state->LegalActions();
    if (legal.empty()) {
      return AdvanceStatus::kErrorEmptyLegal;
    }
    if (legal.size() == 1 && legal[0] == dune_imperium::kActionAcknowledgeChance) {
      state->ApplyAction(dune_imperium::kActionAcknowledgeChance);
      if (history != nullptr) {
        history->push_back(dune_imperium::kActionAcknowledgeChance);
      }
      steps++;
      continue;
    }
    return AdvanceStatus::kSuccessDecision;
  }
  if (state->IsTerminal()) {
    return AdvanceStatus::kSuccessTerminal;
  }
  return AdvanceStatus::kErrorStepLimit;
}

inline constexpr Action kActionPlayWaterPeddlersUnion =
    kActionPlayIntriguePlotCard0 + kIntrigueWaterPeddlersUnion; // 1658
inline constexpr Action kActionPlayWindfall =
    kActionPlayIntriguePlotCard0 + kIntrigueWindfall; // 1659

inline std::string DiagnosticActionToString(Action a) {
  if (a == kActionPlayWindfall) return "PlayPlotIntrigue(Windfall)";
  if (a == kActionPlayWaterPeddlersUnion) return "PlayPlotIntrigue(WaterPeddlersUnion)";
  if (a == 580) return "PaulPeek";
  if (a == 1591) return "FamilyAtomics";
  if (a == kActionEndTurn) return "EndTurn";
  if (a == kActionReveal) return "Reveal";
  if (a == kActionAgentPass) return "AgentPass";
  if (a >= 600 && a <= 621) {
    return "PlaceAgent[" + dune_imperium::SpaceToName(static_cast<dune_imperium::BoardSpaceID>(a)) + "]";
  }
  return "Action[" + std::to_string(a) + "]";
}

inline constexpr int kIntrigueEconomicPositioning = 23;

inline std::string EscapeJson(const std::string& s) {
  std::string out;
  out.reserve(s.size());
  for (char c : s) {
    if (c == '"') out += "\\\"";
    else if (c == '\\') out += "\\\\";
    else if (c == '\n') out += "\\n";
    else if (c == '\r') out += "\\r";
    else if (c == '\t') out += "\\t";
    else out += c;
  }
  return out;
}

inline std::vector<Action> GetPlayerLegalActions(const State& state, Player player) {
  if (state.CurrentPlayer() == player) {
    return state.LegalActions();
  }
  const auto* dune = dynamic_cast<const DuneImperiumState*>(&state);
  if (dune != nullptr) {
    auto clone = std::unique_ptr<DuneImperiumState>(
        dynamic_cast<DuneImperiumState*>(dune->Clone().release()));
    clone->SetCurrentPlayerForTesting(player);
    return clone->LegalActions();
  }
  return {};
}

// Complete model input snapshot for verification (state string, legal actions, observation tensor
// with appendix, semantic action features/IDs/supported flags, and policy output).
struct ModelInputSnapshot {
  std::string state_string;
  std::vector<Action> legal_actions;
  std::vector<float> observation_tensor;
  std::vector<float> action_features;
  std::vector<int64_t> action_card_ids;
  std::vector<int64_t> action_space_ids;
  std::vector<uint8_t> action_supported;
  ActionsAndProbs policy;
};

// ===========================================================================
// Statistical and Confidence Interval Utilities
// ===========================================================================

struct WilsonInterval {
  double point = 0.0;
  double lower = 0.0;
  double upper = 0.0;
};

inline WilsonInterval ComputeWilsonInterval(int successes, int trials,
                                            double z = 1.959963984540054) {
  if (trials <= 0) return {0.0, 0.0, 0.0};
  double p = static_cast<double>(successes) / static_cast<double>(trials);
  double z2 = z * z;
  double n = static_cast<double>(trials);
  double denom = 1.0 + z2 / n;
  double center = (p + z2 / (2.0 * n)) / denom;
  double half = (z * std::sqrt((p * (1.0 - p) / n) + (z2 / (4.0 * n * n)))) / denom;
  return {p, std::max(0.0, center - half), std::min(1.0, center + half)};
}

struct MeanAndCI {
  double mean = 0.0;
  double se = 0.0;
  double lower = 0.0;
  double upper = 0.0;
};

inline MeanAndCI ComputeMeanAndCI(const std::vector<double>& values,
                                  double z = 1.959963984540054) {
  if (values.empty()) return {0.0, 0.0, 0.0, 0.0};
  double n = static_cast<double>(values.size());
  double sum = std::accumulate(values.begin(), values.end(), 0.0);
  double mean = sum / n;
  if (values.size() == 1) return {mean, 0.0, mean, mean};
  double var = 0.0;
  for (double v : values) {
    var += (v - mean) * (v - mean);
  }
  var /= (n - 1.0);
  double se = std::sqrt(std::max(0.0, var) / n);
  return {mean, se, mean - z * se, mean + z * se};
}

// ===========================================================================
// Scenario Bank Definition & Families
// ===========================================================================

enum class ScenarioFamily {
  kCombatTension = 0,     // High-stakes conflict & open combat space vs Swordmaster
  kShippingEconomy = 1,   // Shipping track freighter advancement & conversion vs Swordmaster
  kFactionAlliance = 2,   // Close race for faction alliance (+1 VP) vs Swordmaster
  kMentatHighCouncil = 3  // High Council (5 solari) / Mentat tempo alternative vs Swordmaster
};

inline const char* ScenarioFamilyToString(ScenarioFamily f) {
  switch (f) {
    case ScenarioFamily::kCombatTension:     return "CombatTension";
    case ScenarioFamily::kShippingEconomy:   return "ShippingEconomy";
    case ScenarioFamily::kFactionAlliance:   return "FactionAllianceRace";
    case ScenarioFamily::kMentatHighCouncil: return "MentatHighCouncil";
  }
  return "UnknownFamily";
}

inline ScenarioFamily StringToScenarioFamily(const std::string& s) {
  if (s == "CombatTension")     return ScenarioFamily::kCombatTension;
  if (s == "ShippingEconomy")   return ScenarioFamily::kShippingEconomy;
  if (s == "FactionAllianceRace") return ScenarioFamily::kFactionAlliance;
  if (s == "MentatHighCouncil") return ScenarioFamily::kMentatHighCouncil;
  return ScenarioFamily::kCombatTension;
}

struct ScenarioConfig {
  int scenario_id = 0;
  ScenarioFamily family = ScenarioFamily::kCombatTension;
  std::string family_name;
  int round = 2;
  int p_holder = 0;     // P2 (holds Windfall, 6 solari)
  int p_competitor = 1; // P1 (8 solari, acts before P2's next agent)
  int p_opponent_a = 2; // P3 (owns Swordmaster)
  int p_opponent_b = 3; // P4 (owns Swordmaster)
  uint64_t scenario_seed = 0;
  std::vector<std::string> fixture_edits;
};

// ===========================================================================
// Replicate & Diagnostic Data Structures
// ===========================================================================

struct ReplicateResult {
  int scenario_id = 0;
  int replicate_idx = 0;
  uint64_t chance_seed = 0;
  uint64_t policy_seed = 0;
  bool is_greedy = false;

  Action branch_a_chosen_action = open_spiel::kInvalidAction;
  std::string branch_a_action_name;
  bool branch_a_chose_swordmaster = false;

  Action branch_b_chosen_action = open_spiel::kInvalidAction;
  std::string branch_b_action_name;
  bool branch_b_chose_swordmaster = false;

  double delta_swordmaster = 0.0; // branch_a - branch_b
  int branch_a_steps = 0;
  int branch_b_steps = 0;
};

struct ScenarioSummary {
  int scenario_id = 0;
  std::string family;
  int round = 2;
  int num_replicates = 0;
  double branch_a_swordmaster_freq = 0.0;
  double branch_b_swordmaster_freq = 0.0;
  double effect = 0.0; // freq_a - freq_b
  bool is_saturated_always = false; // both >= 0.95
  bool is_saturated_never = false;  // both <= 0.05

  // Greedy evaluation
  Action greedy_a_action = open_spiel::kInvalidAction;
  std::string greedy_a_action_name;
  bool greedy_a_swordmaster = false;
  Action greedy_b_action = open_spiel::kInvalidAction;
  std::string greedy_b_action_name;
  bool greedy_b_swordmaster = false;
  double greedy_effect = 0.0;

  // Distribution of spaces chosen by P1
  std::map<std::string, int> branch_a_spaces;
  std::map<std::string, int> branch_b_spaces;
};

struct Diagnostic1Aggregate {
  int total_scenarios = 0;
  int replicates_per_scenario = 0;
  int total_paired_trials = 0;

  double mean_branch_a_freq = 0.0;
  WilsonInterval branch_a_wilson;
  double mean_branch_b_freq = 0.0;
  WilsonInterval branch_b_wilson;
  MeanAndCI effect_ci;

  int saturated_always_count = 0;
  int saturated_never_count = 0;
  int sensitive_count = 0;

  std::map<std::string, MeanAndCI> family_effects;
  std::map<std::string, int> alternative_spaces_branch_a;
  std::map<std::string, int> alternative_spaces_branch_b;

  // Greedy aggregate
  double greedy_branch_a_freq = 0.0;
  double greedy_branch_b_freq = 0.0;
  double greedy_mean_effect = 0.0;
};

struct Diagnostic2OpportunityRecord {
  int scenario_id = 0;
  int replicate_idx = 0;
  int player = 0;
  int round = 2;
  int agent_turn_index = 1;      // 1 = first agent turn of round, 2 = second agent turn, etc.
  int decision_in_turn = 0;      // 0, 1, 2... index of action within current agent turn
  int decision_index = 0;        // overall sequential decision index where windfall is legal
  int opportunity_index = 0;     // backward-compatible alias for decision_index
  int legal_actions_count = 0;
  double windfall_prob = 0.0;
  int windfall_rank = 0;         // 1 = argmax
  Action argmax_action = open_spiel::kInvalidAction;
  std::string argmax_action_name;
  double argmax_prob = 0.0;
  Action selected_action = open_spiel::kInvalidAction;
  std::string selected_action_name;
  bool selected_windfall = false;
  bool was_greedy = false;
  bool is_initial_turn = true;   // true if occurring before P2 ends its first turn
};

struct Diagnostic2Summary {
  int total_decisions_evaluated = 0;
  int total_traces_evaluated = 0;
  int windfall_rank1_count = 0;
  double windfall_rank1_rate = 0.0;
  double mean_windfall_prob = 0.0;
  double mean_windfall_rank = 0.0;

  // Turn-level outcomes across evaluated traces
  int traces_played_initial_turn_sampled = 0;
  int traces_held_past_turn1_sampled = 0;
  int traces_played_initial_turn_greedy = 0;
  int traces_held_past_turn1_greedy = 0;

  std::map<std::string, int> argmax_action_distribution;

  // Step-level rates (distinguishing conditional vs cumulative)
  std::map<int, double> conditional_play_rate_sampled;
  std::map<int, double> cumulative_play_rate_sampled;
  std::map<int, double> conditional_play_rate_greedy;
  std::map<int, double> cumulative_play_rate_greedy;

  // Common initial-turn action sequences (e.g. "PaulPeek -> Windfall -> EndTurn")
  std::map<std::string, int> initial_turn_action_sequences_sampled;
  std::map<std::string, int> initial_turn_action_sequences_greedy;
};

struct NaturalRootRecord {
  int candidate_id = 0;
  int game_id = 0;
  uint64_t game_seed = 0;
  int round = 1;
  int step_index = 0;
  int holder_player = 0;
  std::string holder_leader;
  int holder_solari = 0;
  int holder_water = 0;
  int holder_spice = 0;

  int target_opponent = 0;
  std::string opponent_leader;
  int opponent_solari = 0;
  int opponent_water = 0;
  int opponent_spice = 0;
  int opponent_agents = 0;

  Action intrigue_action = open_spiel::kInvalidAction;
  std::string intrigue_name;

  std::vector<Action> history_actions;
  ModelInputSnapshot base_snapshot;

  bool replay_verified = false;
  std::string replay_failure_reason;
};

struct PairedConfirmationRecord {
  int candidate_id = 0;
  int pair_index = 0;
  uint64_t chance_seed = 0;
  std::array<uint64_t, kNumPlayers> policy_seeds = {0};
  int holder_player = 0;
  int target_opponent = 0;
  Action act_early = open_spiel::kInvalidAction;
  std::string act_early_name;
  Action act_hold = open_spiel::kInvalidAction;
  std::string act_hold_name;
  bool reached_target_early = false;
  bool reached_target_hold = false;
  int indicator_early = 0;
  int indicator_hold = 0;
  int indicator_diff = 0; // early - hold
  bool is_valid = true;
};

struct PairedContinuationRecord {
  int candidate_id = 0;
  int pair_index = 0;
  uint64_t chance_seed = 0;
  std::array<uint64_t, kNumPlayers> policy_seeds = {0};
  int holder_player = 0;
  int target_opponent = 0;

  bool branch_a_terminal = false;
  std::string branch_a_status; // "TERMINAL" or "TIMEOUT"
  std::array<double, kNumPlayers> branch_a_returns = {0.0, 0.0, 0.0, 0.0};
  std::array<int, kNumPlayers> branch_a_vps = {0, 0, 0, 0};
  bool branch_a_windfall_played = false;
  int branch_a_windfall_play_round = -1;
  int branch_a_windfall_play_step = -1;

  bool branch_b_terminal = false;
  std::string branch_b_status; // "TERMINAL" or "TIMEOUT"
  std::array<double, kNumPlayers> branch_b_returns = {0.0, 0.0, 0.0, 0.0};
  std::array<int, kNumPlayers> branch_b_vps = {0, 0, 0, 0};
  bool branch_b_intrigue_played_later = false;
  int branch_b_intrigue_play_round = -1;
  int branch_b_intrigue_play_step = -1;
  bool branch_b_windfall_played = false;
  int branch_b_windfall_play_round = -1;
  int branch_b_windfall_play_step = -1;

  double holder_utility_diff = 0.0; // B - A (Holding - Early)
  int holder_vp_diff = 0;          // B - A
  bool is_valid = true;
};

struct IntrigueMiningCandidate {
  int candidate_id = 0;
  int game_id = 0;
  uint64_t game_seed = 0;
  int round = 1;
  int step_index = 0;
  int holder_player = 0;
  std::string holder_leader;
  Action intrigue_action = open_spiel::kInvalidAction;
  std::string intrigue_name; // "Windfall" or "Water Peddlers Union"

  // Holder resources
  int holder_solari = 0;
  int holder_water = 0;
  int holder_spice = 0;

  // Designated opponent
  int opponent_player = 0;
  std::string opponent_leader;
  int opponent_solari = 0;
  int opponent_water = 0;
  int opponent_spice = 0;
  int opponent_agents = 0;

  // Replay verification
  bool replay_verified = false;
  std::string replay_failure_reason;

  // Discovery metrics (4 pairs)
  int discovery_divergent_count = 0;
  Action target_action = open_spiel::kInvalidAction;
  std::string target_action_name;
  int predicted_direction = 0; // +1 if early > hold, -1 if hold > early
  double discovery_delta = 0.0;
  Action discovery_opp_action_early = open_spiel::kInvalidAction;
  std::string discovery_opp_action_early_name;
  Action discovery_opp_action_hold = open_spiel::kInvalidAction;
  std::string discovery_opp_action_hold_name;

  // Confirmation metrics (64 fresh pairs)
  int confirmation_samples = 0;
  int confirmed_divergent_count = 0;
  double confirmed_divergent_rate = 0.0;
  int confirmation_early_target_count = 0;
  int confirmation_hold_target_count = 0;
  double delta_prob = 0.0; // Early - Hold
  double std_err_prob = 0.0;
  double p_value_raw = 1.0;
  double p_value_adjusted = 1.0;
  double ci_prob_lower = 0.0;
  double ci_prob_upper = 0.0;
  bool confirmed_response = false; // direction match && |delta| >= 0.05 && p_adj < 0.05

  // Placement decision argmax & target prob
  Action opponent_argmax_early = open_spiel::kInvalidAction;
  std::string opponent_argmax_early_name;
  double opponent_target_prob_early = 0.0;
  Action opponent_argmax_hold = open_spiel::kInvalidAction;
  std::string opponent_argmax_hold_name;
  double opponent_target_prob_hold = 0.0;

  // Final game continuation metrics (64 pairs)
  int continuation_samples = 0;
  bool has_invalidation_failure = false;
  std::string invalidation_reason;
  double mean_early_holder_utility = 0.0;
  double mean_hold_holder_utility = 0.0;
  double delta_holder_utility = 0.0; // Hold - Early
  double std_err_utility = 0.0;
  double p_value_utility_raw = 1.0;
  double p_value_utility_adjusted = 1.0;
  double utility_ci_lower_adjusted = 0.0;
  double utility_ci_upper_adjusted = 0.0;
  double mean_early_holder_vp = 0.0;
  double mean_hold_holder_vp = 0.0;
  double delta_holder_vp = 0.0;
  std::string verdict = "Inconclusive"; // "Supported", "Inconclusive", or "Contradicted"
};

struct IntrigueMiningSummary {
  int games_simulated = 0;
  int total_resource_intrigue_plays = 0;
  int windfall_plays = 0;
  int water_peddlers_union_plays = 0;
  int candidates_discovered = 0;
  int candidates_confirmed_divergent = 0;
  int candidates_tested_confirmation = 0;
  int candidates_tested_continuation = 0;
  bool run_has_invalidation_failure = false;
  std::string invalidation_reason;
  std::vector<IntrigueMiningCandidate> candidates;
  std::vector<NaturalRootRecord> roots;
  std::vector<PairedConfirmationRecord> confirmation_records;
  std::vector<PairedContinuationRecord> continuation_records;
};

struct FullGameContinuationResult {
  int scenario_id = 0;
  int replicate_idx = 0;
  std::string mode; // "unchanged" or "controlled"
  bool delayed_opportunity_available = false;
  bool intended_opportunity_disappeared = false;
  int branch_a_p2_placement = 0;
  int branch_b_p2_placement = 0;
  double branch_a_p2_utility = 0.0;
  double branch_b_p2_utility = 0.0;
  int branch_a_ending_round = 0;
  int branch_b_ending_round = 0;
};

struct ReplayVerificationResult {
  bool matches = false;
  std::string failure_reason;
  size_t steps_replayed = 0;
};

struct WindfallDecisionRecord {
  int candidate_id = -1;
  int replicate_id = -1;
  int game_id = -1;
  int round = 0;
  int turn_step = 0;
  int player = 0;
  int actual_solari = 0;
  bool windfall_available = false;
  bool windfall_played = false;

  // Proposal
  std::vector<Action> proposed_action_chain;
  std::vector<std::string> proposed_action_names;
  bool plan_requires_windfall = false;
  std::string windfall_reason;

  // Real execution
  Action action_taken = open_spiel::kInvalidAction;
  std::string action_taken_name;

  // Exceptions and fallbacks
  bool secrets_exception = false;
  bool terminal_value_exception = false;
  bool fallback_active = false;
  std::string fallback_reason;

  // After-action state
  int solari_after = 0;
  std::string resource_use_followed;
};

struct WindfallBankRecord {
  int candidate_id = 0;
  int game_id = 0;
  uint64_t game_seed = 0;
  int round = 1;
  int step_index = 0;
  int holder_player = 0;
  std::string holder_leader;
  int holder_solari = 0;
  int holder_water = 0;
  int holder_spice = 0;
  int target_opponent = 0;
  std::string opponent_leader;
  int opponent_solari = 0;
  Action intrigue_action = open_spiel::kInvalidAction;
  std::string intrigue_name;
  std::vector<Action> history_actions;
  ModelInputSnapshot base_snapshot;
  bool replay_verified = false;
  std::string replay_failure_reason;
  bool is_selected = false;
  std::string selection_reason;
};

struct WindfallEvaluationSummary {
  std::string candidate_model;
  std::string candidate_sha256;
  std::string opponent_model;
  std::string opponent_sha256;
  std::array<std::string, kNumPlayers> seat_checkpoints;
  std::array<std::string, kNumPlayers> seat_sha256s;
  uint64_t base_seed = 0;
  float temperature = 1.0f;
  int total_natural_roots_reconstructed = 0;
  int total_natural_roots_verified = 0;
  int positions_selected = 0;
  int continuations_per_position = 0;
  int total_continuation_pairs = 0;
  bool run_has_invalidation_failure = false;
  std::string invalidation_reason;

  // Primary metric: equally weighted mean placement-utility difference across positions
  double mean_placement_utility_diff = 0.0;
  double placement_utility_std_err = 0.0;
  double placement_utility_ci_lower = 0.0;
  double placement_utility_ci_upper = 0.0;
  double mean_holder_vp_diff = 0.0;

  // Secondary metrics
  int total_controller_decisions = 0;
  int early_plays_prevented = 0;
  int windfall_plays_executed = 0;
  int necessary_spending_unlocked = 0;
  int purchases_already_affordable = 0;
  int end_turn_retentions = 0;
  int secrets_exceptions_triggered = 0;
  int terminal_value_exceptions_triggered = 0;
  int fallbacks_triggered = 0;

  std::string verdict; // "Beneficial", "Harmful", "Inconclusive", "Invalid"
  std::string verdict_explanation;

  std::vector<WindfallBankRecord> bank;
  std::vector<WindfallDecisionRecord> decisions;
  std::vector<PairedContinuationRecord> continuation_records;
};

} // namespace dune_diagnostics
} // namespace open_spiel

#endif // OPEN_SPIEL_EXAMPLES_DUNE_FROZEN_POLICY_DIAG_TYPES_H_
