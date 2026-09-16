#ifndef OPEN_SPIEL_EXAMPLES_DUNE_FROZEN_POLICY_DIAGNOSTICS_H_
#define OPEN_SPIEL_EXAMPLES_DUNE_FROZEN_POLICY_DIAGNOSTICS_H_

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

#ifdef OPEN_SPIEL_BUILD_WITH_LIBTORCH
#include <torch/torch.h>
#include "dune_network.h"
#include "dune_semantic_action_scorer.h"
#include "dune_sha256.h"
#endif

namespace open_spiel {
namespace dune_diagnostics {

using dune_imperium::ActionToString;
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
  return ActionToString(a);
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

// ===========================================================================
// Evaluator Interface & Scripted Controls
// ===========================================================================

class IPolicyDiagnosticsEvaluator {
 public:
  virtual ~IPolicyDiagnosticsEvaluator() = default;
  virtual ActionsAndProbs GetPolicy(const State& state, Player player) = 0;
  virtual Action SelectAction(const State& state, Player player, bool greedy,
                              float temperature, std::mt19937_64& rng) = 0;
  virtual std::vector<float> GetRawLogits(const State& state, Player player) = 0;
  virtual bool IsNeural() const = 0;
  virtual std::string ModelName() const = 0;
  virtual std::string ModelSha256() const = 0;
  virtual ModelInputSnapshot CaptureSnapshot(const State& state, Player player) {
    ModelInputSnapshot snap;
    snap.state_string = state.ToString();
    snap.legal_actions = GetPlayerLegalActions(state, player);
    snap.observation_tensor = state.InformationStateTensor(player);
    snap.policy = GetPolicy(state, player);
    return snap;
  }
};

// Positive control: Observes P2's visible solari.
// If P2 has >= 8 solari, P1 aggressively takes Swordmaster.
// If P2 has < 8 solari, P1 takes an alternative space.
class ScriptedPositiveControlEvaluator : public IPolicyDiagnosticsEvaluator {
 public:
  explicit ScriptedPositiveControlEvaluator(int p_holder, int p_competitor)
      : p_holder_(p_holder), p_competitor_(p_competitor) {}

  bool IsNeural() const override { return false; }
  std::string ModelName() const override { return "ScriptedPositiveControl"; }
  std::string ModelSha256() const override { return "positive_control_sha256_synthetic"; }

  std::vector<float> GetRawLogits(const State& state, Player player) override {
    auto legal = GetPlayerLegalActions(state, player);
    std::vector<float> logits(3000, -100.0f);
    for (Action a : legal) logits[a] = 0.0f;
    return logits;
  }

  ActionsAndProbs GetPolicy(const State& state, Player player) override {
    auto legal = GetPlayerLegalActions(state, player);
    ActionsAndProbs policy;
    if (legal.empty()) return policy;
    double p = 1.0 / legal.size();
    for (Action a : legal) policy.push_back({a, p});
    return policy;
  }

  Action SelectAction(const State& state, Player player, bool /*greedy*/,
                      float /*temperature*/, std::mt19937_64& /*rng*/) override {
    const auto* dune = dynamic_cast<const DuneImperiumState*>(&state);
    SPIEL_CHECK_TRUE(dune != nullptr);
    auto legal = state.LegalActions();
    SPIEL_CHECK_FALSE(legal.empty());

    if (player == p_competitor_) {
      int holder_solari = dune->GetPlayerSolari(p_holder_);
      bool threat = (holder_solari >= 8);

      // Card selection phase
      if (dune_imperium::HasCardSelectionAction(legal)) {
        if (threat) {
          // Select card with Landsraad access to enable Swordmaster
          for (Action a : legal) {
            if (a >= kActionSelectAgentCard0 && a < kActionSelectAgentCard0 + 256) {
              int cid = a - kActionSelectAgentCard0;
              const auto* card = FindImperiumCardById(cid);
              if (card && (card->access_mask & dune_imperium::kAccessLandsraad)) {
                return a;
              }
            }
          }
        } else {
          // Non-threat: pick a card that does NOT access Landsraad if possible
          for (Action a : legal) {
            if (a >= kActionSelectAgentCard0 && a < kActionSelectAgentCard0 + 256) {
              int cid = a - kActionSelectAgentCard0;
              const auto* card = FindImperiumCardById(cid);
              if (card && !(card->access_mask & dune_imperium::kAccessLandsraad)) {
                return a;
              }
            }
          }
        }
        return legal[0];
      }

      // Solo play
      if (ContainsAction(legal, kActionPlayAgentSolo)) {
        return kActionPlayAgentSolo;
      }

      // Graft partner selection
      for (Action a : legal) {
        if (a >= kActionSelectGraftPartner0 && a < kActionSelectGraftPartner0 + 256) {
          return a;
        }
      }

      // Board space placement
      if (threat && ContainsAction(legal, kActionAgentSpaceSwordmaster)) {
        return kActionAgentSpaceSwordmaster;
      }

      // Otherwise pick an alternative space or legal action
      for (Action a : legal) {
        if (IsAgentSpaceAction(a) && a != kActionAgentSpaceSwordmaster) {
          return a;
        }
      }
      return legal[0];
    }

    // Default player behavior
    return dune_imperium::DefaultProgressionAction(legal);
  }

 private:
  int p_holder_;
  int p_competitor_;
};

// ===========================================================================
// Scripted Nonreactive Control Evaluator
// ===========================================================================

class ScriptedNonreactiveControlEvaluator : public IPolicyDiagnosticsEvaluator {
 private:
  int p_holder_ = 0;
  int p_competitor_ = 1;

 public:
  ScriptedNonreactiveControlEvaluator(int p_holder, int p_competitor)
      : p_holder_(p_holder), p_competitor_(p_competitor) {}

  bool IsNeural() const override { return false; }
  std::string ModelName() const override { return "ScriptedNonreactiveControl"; }
  std::string ModelSha256() const override { return "nonreactive_control_sha256_synthetic"; }

  std::vector<float> GetRawLogits(const State& state, Player player) override {
    auto legal = GetPlayerLegalActions(state, player);
    std::vector<float> logits(3000, -100.0f);
    for (Action a : legal) logits[a] = 0.0f;
    return logits;
  }

  ActionsAndProbs GetPolicy(const State& state, Player player) override {
    auto legal = GetPlayerLegalActions(state, player);
    ActionsAndProbs policy;
    if (legal.empty()) return policy;
    double p = 1.0 / legal.size();
    for (Action a : legal) policy.push_back({a, p});
    return policy;
  }

  Action SelectAction(const State& state, Player player, bool /*greedy*/,
                      float /*temperature*/, std::mt19937_64& /*rng*/) override {
    const auto* dune = dynamic_cast<const DuneImperiumState*>(&state);
    SPIEL_CHECK_TRUE(dune != nullptr);
    auto legal = state.LegalActions();
    SPIEL_CHECK_FALSE(legal.empty());

    if (player == p_competitor_) {
      // Fixed priority regardless of P2's solari
      if (dune_imperium::HasCardSelectionAction(legal)) {
        for (Action a : legal) {
          if (a >= kActionSelectAgentCard0 && a < kActionSelectAgentCard0 + 256) return a;
        }
      }
      if (ContainsAction(legal, kActionPlayAgentSolo)) return kActionPlayAgentSolo;
      for (Action a : legal) {
        if (a >= kActionSelectGraftPartner0 && a < kActionSelectGraftPartner0 + 256) return a;
      }
      // Fixed board space preference: High Council > Mentat > Swordmaster > Arrakeen
      if (ContainsAction(legal, kActionAgentSpaceHighCouncil)) return kActionAgentSpaceHighCouncil;
      if (ContainsAction(legal, kActionAgentSpaceMentat)) return kActionAgentSpaceMentat;
      if (ContainsAction(legal, kActionAgentSpaceSwordmaster)) return kActionAgentSpaceSwordmaster;
      if (ContainsAction(legal, kActionAgentSpaceArrakeen)) return kActionAgentSpaceArrakeen;
      for (Action a : legal) {
        if (IsAgentSpaceAction(a)) return a;
      }
      if (ContainsAction(legal, kActionReveal)) return kActionReveal;
      if (ContainsAction(legal, kActionEndTurn)) return kActionEndTurn;
      return legal[0];
    }

    if (ContainsAction(legal, kActionEndTurn)) return kActionEndTurn;
    return dune_imperium::TraceSimulationAction(legal);
  }
};

#ifdef OPEN_SPIEL_BUILD_WITH_LIBTORCH
class NeuralPolicyDiagnosticsEvaluator : public IPolicyDiagnosticsEvaluator {
 public:
  NeuralPolicyDiagnosticsEvaluator(const std::string& checkpoint_path,
                                   torch::Device device,
                                   float logit_cap = 10.0f)
      : path_(checkpoint_path), device_(device), logit_cap_(logit_cap) {
    std::cout << "Loading neural checkpoint for diagnostics: " << checkpoint_path << std::endl;
    sha256_ = ComputeFileSHA256(checkpoint_path);
    std::cout << "Checkpoint SHA256: " << sha256_ << std::endl;

    // Detect architecture and schema
    ReadMetadata();

    model_ = std::make_shared<SharedDunePolicyValueNetImpl>(
        obs_dim_, hidden_dim_, kActionDim, num_blocks_,
        /*use_nonlinear=*/false, /*with_aux_heads=*/false,
        /*head_init_seed=*/0, with_semantic_scorer_);

    torch::serialize::InputArchive archive;
    archive.load_from(checkpoint_path, device_);
    model_->load(archive);
    model_->to(device_);
    model_->eval();

    evaluator_ = std::make_shared<DeterministicEvaluator>(
        model_, device_, &eval_mutex_, nullptr, nullptr, /*rollout_amp=*/false);
  }

  bool IsNeural() const override { return true; }
  std::string ModelName() const override { return path_; }
  std::string ModelSha256() const override { return sha256_; }

  int ObservationDim() const { return obs_dim_; }
  int HiddenDim() const { return hidden_dim_; }
  int NumBlocks() const { return num_blocks_; }
  bool WithSemanticScorer() const { return with_semantic_scorer_; }
  dune_imperium::MarketAppendixMode MarketMode() const { return market_mode_; }
  bool HasSemanticScorer() const { return evaluator_ != nullptr && evaluator_->HasSemanticScorer(); }
  std::string SemanticDescriptorSchema() const {
    return evaluator_ != nullptr ? evaluator_->SemanticDescriptorSchema() : dune_semantic::kDescriptorSchemaVersionV3;
  }

  ModelInputSnapshot CaptureSnapshot(const State& state, Player player) override {
    ModelInputSnapshot snap;
    snap.state_string = state.ToString();
    snap.legal_actions = GetPlayerLegalActions(state, player);
    const auto* dune = dynamic_cast<const DuneImperiumState*>(&state);
    if (dune != nullptr) {
      if (market_mode_ != dune_imperium::MarketAppendixMode::kNone) {
        snap.observation_tensor = dune->InformationStateTensorWithAppendix(player, market_mode_);
      } else {
        snap.observation_tensor = state.InformationStateTensor(player);
      }
      if (evaluator_ != nullptr && evaluator_->HasSemanticScorer() && !snap.legal_actions.empty()) {
        dune_semantic::CandidateActionData cand_data;
        dune_semantic::ExtractCandidateDescriptors(
            *dune, snap.legal_actions, &cand_data, evaluator_->SemanticDescriptorSchema());
        snap.action_features = cand_data.features;
        snap.action_card_ids = cand_data.card_ids;
        snap.action_space_ids = cand_data.space_ids;
        snap.action_supported = cand_data.supported;
      }
    } else {
      snap.observation_tensor = state.InformationStateTensor(player);
    }
    snap.policy = GetPolicy(state, player);
    return snap;
  }

  std::vector<float> GetRawLogits(const State& state, Player player) override {
    const auto* dune = dynamic_cast<const DuneImperiumState*>(&state);
    std::vector<float> obs;
    if (market_mode_ != dune_imperium::MarketAppendixMode::kNone) {
      obs = dune->InformationStateTensorWithAppendix(player, market_mode_);
    } else {
      obs = state.InformationStateTensor(player);
    }
    std::vector<Action> legal = GetPlayerLegalActions(state, player);

    std::unique_ptr<dune_semantic::CandidateActionData> cand_data;
    if (evaluator_->HasSemanticScorer() && !legal.empty()) {
      cand_data = std::make_unique<dune_semantic::CandidateActionData>();
      dune_semantic::ExtractCandidateDescriptors(
          *dune, legal, cand_data.get(), evaluator_->SemanticDescriptorSchema());
    }

    auto res = evaluator_->EvaluateWithActions(obs, cand_data.get());
    return res.logits;
  }

  ActionsAndProbs GetPolicy(const State& state, Player player) override {
    auto legal = GetPlayerLegalActions(state, player);
    if (legal.empty()) return {};

    std::vector<float> logits = GetRawLogits(state, player);
    CenterAndCapLegalLogitsWithStats(logits, legal, logit_cap_);

    double max_logit = -1e30;
    for (Action a : legal) {
      if (a >= 0 && static_cast<size_t>(a) < logits.size()) {
        max_logit = std::max(max_logit, static_cast<double>(logits[a]));
      }
    }
    double sum = 0.0;
    std::vector<double> probs(legal.size());
    for (size_t i = 0; i < legal.size(); ++i) {
      Action a = legal[i];
      if (a >= 0 && static_cast<size_t>(a) < logits.size()) {
        probs[i] = std::exp(static_cast<double>(logits[a]) - max_logit);
        sum += probs[i];
      }
    }
    ActionsAndProbs policy;
    policy.reserve(legal.size());
    for (size_t i = 0; i < legal.size(); ++i) {
      double p = (sum > 0.0) ? (probs[i] / sum) : (1.0 / legal.size());
      policy.push_back({legal[i], p});
    }
    return policy;
  }

  Action SelectAction(const State& state, Player player, bool greedy,
                      float temperature, std::mt19937_64& rng) override {
    auto legal = state.LegalActions();
    if (legal.empty()) return open_spiel::kInvalidAction;
    if (legal.size() == 1) return legal[0];

    std::vector<float> logits = GetRawLogits(state, player);
    CenterAndCapLegalLogitsWithStats(logits, legal, logit_cap_);

    dune_eval::SelectionPolicy sp{greedy, temperature};
    return dune_eval::SelectActionFromLogits(logits, legal, sp, rng);
  }

 private:
  void ReadMetadata() {
    std::filesystem::path p(path_);
    std::filesystem::path p_json = p;
    p_json.replace_extension(".json");
    if (std::filesystem::exists(p_json)) {
      std::ifstream f(p_json);
      if (f) {
        std::string content((std::istreambuf_iterator<char>(f)),
                            std::istreambuf_iterator<char>());
        auto val_opt = json::FromString(content);
        if (val_opt.has_value() && val_opt->IsObject()) {
          const auto& obj = val_opt->GetObject();
          auto it_od = obj.find("observation_dim");
          if (it_od != obj.end() && it_od->second.IsInt()) {
            obs_dim_ = it_od->second.GetInt();
          }
          auto it_hd = obj.find("hidden_dim");
          if (it_hd != obj.end() && it_hd->second.IsInt()) {
            hidden_dim_ = it_hd->second.GetInt();
          }
          auto it_nb = obj.find("num_blocks");
          if (it_nb != obj.end() && it_nb->second.IsInt()) {
            num_blocks_ = it_nb->second.GetInt();
          }
          auto it_ess = obj.find("enable_semantic_scorer");
          if (it_ess != obj.end()) {
            with_semantic_scorer_ = (it_ess->second.IsBool() && it_ess->second.GetBool()) ||
                                   (it_ess->second.IsString() && it_ess->second.GetString() == "true");
          }
          auto it_mm = obj.find("market_appendix_mode");
          if (it_mm != obj.end() && it_mm->second.IsString()) {
            market_mode_str_ = it_mm->second.GetString();
          }
        }
      }
    }

    if (obs_dim_ == dune_imperium::kFullPublicInformationStateSize) {
      market_mode_ = dune_imperium::MarketAppendixMode::kFullPublicInformationV3;
      with_semantic_scorer_ = true;
    } else if (obs_dim_ == dune_imperium::kOrderedCardSlotsInformationStateSize) {
      market_mode_ = dune_imperium::MarketAppendixMode::kOrderedCardSlotsV2;
    } else {
      market_mode_ = dune_imperium::MarketAppendixMode::kNone;
    }
  }

  std::string path_;
  std::string sha256_;
  torch::Device device_;
  float logit_cap_ = 10.0f;
  int obs_dim_ = 9182;
  int hidden_dim_ = 2048;
  int num_blocks_ = 8;
  static constexpr int kActionDim = 2391;
  bool with_semantic_scorer_ = true;
  std::string market_mode_str_ = "full_public_information_v3";
  dune_imperium::MarketAppendixMode market_mode_ =
      dune_imperium::MarketAppendixMode::kFullPublicInformationV3;

  std::mutex eval_mutex_;
  std::shared_ptr<SharedDunePolicyValueNetImpl> model_;
  std::shared_ptr<DeterministicEvaluator> evaluator_;
};
#endif

struct ReplayVerificationResult {
  bool matches = false;
  std::string failure_reason;
  size_t steps_replayed = 0;
};

inline ReplayVerificationResult VerifyReplayHistory(
    const std::shared_ptr<const Game>& game,
    const std::vector<Action>& history,
    Player target_player,
    const ModelInputSnapshot& expected_snapshot,
    IPolicyDiagnosticsEvaluator* evaluator) {
  ReplayVerificationResult res;
  res.matches = false;
  if (!game) {
    res.failure_reason = "Game pointer is null";
    return res;
  }
  std::unique_ptr<State> s = game->NewInitialState();
  for (size_t i = 0; i < history.size(); ++i) {
    if (s->IsTerminal()) {
      res.failure_reason = "Game terminated unexpectedly during replay at step " + std::to_string(i);
      return res;
    }
    s->ApplyAction(history[i]);
  }
  res.steps_replayed = history.size();

  // 1. Verify state string
  std::string replayed_state_str = s->ToString();
  if (replayed_state_str != expected_snapshot.state_string) {
    res.failure_reason = "Replayed state string mismatch.";
    return res;
  }

  // 2. Capture snapshot at replayed state
  ModelInputSnapshot replayed_snap = evaluator->CaptureSnapshot(*s, target_player);

  // 3. Verify legal actions
  if (replayed_snap.legal_actions != expected_snapshot.legal_actions) {
    res.failure_reason = "Replayed legal actions mismatch.";
    return res;
  }

  // 4. Verify observation tensor (including appendix)
  if (replayed_snap.observation_tensor.size() != expected_snapshot.observation_tensor.size()) {
    res.failure_reason = "Observation tensor size mismatch: expected " +
        std::to_string(expected_snapshot.observation_tensor.size()) +
        ", got " + std::to_string(replayed_snap.observation_tensor.size());
    return res;
  }
  for (size_t i = 0; i < expected_snapshot.observation_tensor.size(); ++i) {
    if (std::abs(replayed_snap.observation_tensor[i] - expected_snapshot.observation_tensor[i]) > 1e-5f) {
      res.failure_reason = "Observation tensor value mismatch at index " + std::to_string(i);
      return res;
    }
  }

  // 5. Verify semantic action features and categorical IDs
  if (replayed_snap.action_features != expected_snapshot.action_features) {
    res.failure_reason = "Semantic action features mismatch.";
    return res;
  }
  if (replayed_snap.action_card_ids != expected_snapshot.action_card_ids) {
    res.failure_reason = "Semantic action card IDs mismatch.";
    return res;
  }
  if (replayed_snap.action_space_ids != expected_snapshot.action_space_ids) {
    res.failure_reason = "Semantic action space IDs mismatch.";
    return res;
  }
  if (replayed_snap.action_supported != expected_snapshot.action_supported) {
    res.failure_reason = "Semantic action supported flags mismatch.";
    return res;
  }

  // 6. Verify policy probabilities
  if (replayed_snap.policy.size() != expected_snapshot.policy.size()) {
    res.failure_reason = "Policy size mismatch.";
    return res;
  }
  for (size_t i = 0; i < expected_snapshot.policy.size(); ++i) {
    if (replayed_snap.policy[i].first != expected_snapshot.policy[i].first ||
        std::abs(replayed_snap.policy[i].second - expected_snapshot.policy[i].second) > 1e-5) {
      res.failure_reason = "Policy probability mismatch at action " +
          std::to_string(expected_snapshot.policy[i].first);
      return res;
    }
  }

  res.matches = true;
  return res;
}

// ===========================================================================
// Windfall-Timing Controller & Decision Tracking
// ===========================================================================

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

class WindfallTimingControlledEvaluator : public IPolicyDiagnosticsEvaluator {
 public:
  WindfallTimingControlledEvaluator(IPolicyDiagnosticsEvaluator* base_evaluator,
                                   int designated_holder = 0)
      : base_evaluator_(base_evaluator), designated_holder_(designated_holder) {
    SPIEL_CHECK_TRUE(base_evaluator_ != nullptr);
  }

  void SetHolder(int holder) { designated_holder_ = holder; }
  void SetContext(int candidate_id, int replicate_id) {
    candidate_id_ = candidate_id;
    replicate_id_ = replicate_id;
    windfall_played_this_run_ = false;
    resource_followup_recorded_ = false;
    last_resource_action_ = open_spiel::kInvalidAction;
    last_resource_action_name_.clear();
  }

  void ResetTurnStep() { turn_step_ = 0; }

  const std::vector<WindfallDecisionRecord>& DecisionRecords() const {
    return decision_records_;
  }

  void ClearDecisionRecords() { decision_records_.clear(); }

  bool IsNeural() const override { return base_evaluator_->IsNeural(); }
  std::string ModelName() const override {
    return base_evaluator_->ModelName() + " [WindfallControlled]";
  }
  std::string ModelSha256() const override { return base_evaluator_->ModelSha256(); }

  std::vector<float> GetRawLogits(const State& state, Player player) override {
    return base_evaluator_->GetRawLogits(state, player);
  }

  ActionsAndProbs GetPolicy(const State& state, Player player) override {
    return base_evaluator_->GetPolicy(state, player);
  }

  ModelInputSnapshot CaptureSnapshot(const State& state, Player player) override {
    return base_evaluator_->CaptureSnapshot(state, player);
  }

  void RecordResourceFollowUp(Action action) {
    if (windfall_played_this_run_ && !resource_followup_recorded_) {
      if (dune_imperium::IsAgentSpaceAction(action) ||
          (action >= dune_imperium::kActionBuyImperiumRow0 && action <= dune_imperium::kActionBuyImperiumRow0 + 10) ||
          action == dune_imperium::kActionBuyReserveArrakisLiaison ||
          action == dune_imperium::kActionBuyReserveTheSpiceMustFlow ||
          (action >= dune_imperium::kActionTechAcquire0 && action <= dune_imperium::kActionTechAcquire0 + 6)) {
        resource_followup_recorded_ = true;
        last_resource_action_ = action;
        last_resource_action_name_ = DiagnosticActionToString(action);
        if (!decision_records_.empty()) {
          decision_records_.back().resource_use_followed = last_resource_action_name_;
        }
      }
    }
  }

  Action SelectAction(const State& state, Player player, bool greedy,
                      float temperature, std::mt19937_64& rng) override {
    turn_step_++;

    // 1. Controller applies ONLY to designated holder
    if (player != designated_holder_) {
      return base_evaluator_->SelectAction(state, player, greedy, temperature, rng);
    }

    const auto* dune = dynamic_cast<const DuneImperiumState*>(&state);
    SPIEL_CHECK_TRUE(dune != nullptr);

    auto legal = state.LegalActions();
    if (legal.empty()) return open_spiel::kInvalidAction;
    if (legal.size() == 1) {
      RecordResourceFollowUp(legal[0]);
      return legal[0];
    }

    // 2. Controller applies ONLY while player possesses Windfall
    const auto& intrigues = dune->GetPlayerIntrigues(player);
    bool possesses_windfall =
        (std::find(intrigues.begin(), intrigues.end(), kIntrigueWindfall) != intrigues.end());

    // Check if Windfall is currently legal to play
    bool windfall_legal = ContainsAction(legal, kActionPlayWindfall);
    if (!possesses_windfall || !windfall_legal) {
      Action act = base_evaluator_->SelectAction(state, player, greedy, temperature, rng);
      RecordResourceFollowUp(act);
      return act;
    }

    // At this point: player possesses Windfall AND kActionPlayWindfall is currently legal!
    WindfallDecisionRecord rec;
    rec.candidate_id = candidate_id_;
    rec.replicate_id = replicate_id_;
    rec.round = dune->GetCurrentRound();
    rec.turn_step = turn_step_;
    rec.player = player;
    rec.actual_solari = dune->GetPlayerSolari(player);
    rec.windfall_available = true;

    // 3. Secrets exception:
    // "For this prototype, conservatively treat holding four or more intrigue cards as exposure.
    //  In that condition, preserve the baseline policy’s Windfall decision rather than forcing retention."
    if (intrigues.size() >= 4) {
      rec.secrets_exception = true;
      Action act = base_evaluator_->SelectAction(state, player, greedy, temperature, rng);
      rec.action_taken = act;
      rec.action_taken_name = DiagnosticActionToString(act);
      rec.windfall_played = (act == kActionPlayWindfall);
      rec.windfall_reason = "Secrets exception: holder possesses >= 4 intrigue cards (exposure risk); baseline decision preserved";
      if (rec.windfall_played) {
        rec.solari_after = rec.actual_solari + 2;
        windfall_played_this_run_ = true;
      } else {
        rec.solari_after = rec.actual_solari;
      }
      decision_records_.push_back(rec);
      RecordResourceFollowUp(act);
      return act;
    }

    // 4. Terminal Value Exception:
    // "Also preserve any terminal value of Windfall’s resources under the engine’s rules.
    //  The controller must not accidentally forfeit usable resources at game end because no purchase remains."
    bool potential_endgame = (dune->GetCurrentRound() >= 10);
    for (int p = 0; p < kNumPlayers; ++p) {
      if (dune->GetPlayerVp(p) >= 10) {
        potential_endgame = true;
        break;
      }
    }

    bool has_economic_positioning =
        (std::find(intrigues.begin(), intrigues.end(), kIntrigueEconomicPositioning) != intrigues.end());
    bool economic_positioning_threshold =
        (has_economic_positioning && (rec.actual_solari == 8 || rec.actual_solari == 9));

    if (economic_positioning_threshold ||
        (potential_endgame && (dune->phase() == GamePhase::kRevealTurns ||
                               dune->GetPlayerAgentsRemainingForTesting(player) == 0))) {
      rec.terminal_value_exception = true;
      rec.plan_requires_windfall = true;
      rec.windfall_reason = economic_positioning_threshold
          ? "Terminal value: Economic Positioning threshold (solari reaches >= 10 for +1 VP)"
          : "Terminal value: Endgame round potential; playing Windfall preserves solari tiebreaker";
      rec.action_taken = kActionPlayWindfall;
      rec.action_taken_name = "PlayPlotIntrigue(Windfall)";
      rec.windfall_played = true;
      rec.solari_after = rec.actual_solari + 2;
      windfall_played_this_run_ = true;
      decision_records_.push_back(rec);
      return kActionPlayWindfall;
    }

    // 5. Speculative Proposal in temporary clone with Windfall resolved:
    // "Use the native policy to propose an immediate action plan from a temporary clone in which Windfall has been legally resolved."
    try {
      auto temp_clone = state.Clone();
      temp_clone->ApplyAction(kActionPlayWindfall);

      // Dedicated proposal RNG stream derived from player's context to avoid polluting real stream
      uint64_t prop_seed = dune_seed::DeriveSeed(20260915, 0x00B0, turn_step_, player, dune_seed::kStreamChance);
      std::mt19937_64 prop_chance_rng(prop_seed);
      std::mt19937_64 prop_policy_rng(prop_seed ^ 0xFEEDFACEULL);

      AdvanceStatus st = AdvanceThroughChanceAndAcks(temp_clone.get(), prop_chance_rng);
      if (st != AdvanceStatus::kSuccessDecision && st != AdvanceStatus::kSuccessTerminal) {
        throw std::runtime_error("Proposal clone advance failed: " + std::string(AdvanceStatusToString(st)));
      }

      if (temp_clone->IsTerminal() || temp_clone->CurrentPlayer() != player) {
        throw std::runtime_error("Proposal clone unexpected player transition");
      }

      auto prop_legal = temp_clone->LegalActions();
      if (prop_legal.empty()) {
        throw std::runtime_error("Proposal clone legal actions empty");
      }

      // Query policy in temp_clone for proposal
      Action a1 = base_evaluator_->SelectAction(*temp_clone, player, /*greedy=*/false, temperature, prop_policy_rng);
      rec.proposed_action_chain.push_back(a1);
      rec.proposed_action_names.push_back(DiagnosticActionToString(a1));

      // Case A: a1 is agent card selection
      if (a1 >= kActionSelectAgentCard0 && a1 < kActionSelectAgentCard0 + 256) {
        temp_clone->ApplyAction(a1);
        AdvanceThroughChanceAndAcks(temp_clone.get(), prop_chance_rng);

        auto next_legal = temp_clone->LegalActions();
        if (temp_clone->CurrentPlayer() == player && !next_legal.empty()) {
          if (ContainsAction(next_legal, kActionPlayAgentSolo)) {
            temp_clone->ApplyAction(kActionPlayAgentSolo);
            rec.proposed_action_chain.push_back(kActionPlayAgentSolo);
            rec.proposed_action_names.push_back("PlayAgentSolo");
            AdvanceThroughChanceAndAcks(temp_clone.get(), prop_chance_rng);
            next_legal = temp_clone->LegalActions();
          } else {
            bool has_graft = false;
            for (Action ga : next_legal) {
              if (ga >= kActionSelectGraftPartner0 && ga < kActionSelectGraftPartner0 + 256) {
                has_graft = true;
                break;
              }
            }
            if (has_graft) {
              Action graft_act = base_evaluator_->SelectAction(*temp_clone, player, /*greedy=*/false, temperature, prop_policy_rng);
              temp_clone->ApplyAction(graft_act);
              rec.proposed_action_chain.push_back(graft_act);
              rec.proposed_action_names.push_back(DiagnosticActionToString(graft_act));
              AdvanceThroughChanceAndAcks(temp_clone.get(), prop_chance_rng);
              next_legal = temp_clone->LegalActions();
            }
          }
        }

        // Space selection
        if (temp_clone->CurrentPlayer() == player && !next_legal.empty()) {
          Action a_space = base_evaluator_->SelectAction(*temp_clone, player, /*greedy=*/false, temperature, prop_policy_rng);
          rec.proposed_action_chain.push_back(a_space);
          rec.proposed_action_names.push_back(DiagnosticActionToString(a_space));

          // Test whether this chain is legal in the REAL state (without Windfall)
          auto test_real = state.Clone();
          auto* d_real = dynamic_cast<DuneImperiumState*>(test_real.get());
          bool chain_legal_in_real = false;

          if (ContainsAction(d_real->LegalActions(), a1)) {
            d_real->ApplyAction(a1);
            AdvanceThroughChanceAndAcks(d_real, prop_chance_rng);
            auto real_legal_2 = d_real->LegalActions();

            for (size_t k = 1; k + 1 < rec.proposed_action_chain.size(); ++k) {
              Action step_act = rec.proposed_action_chain[k];
              if (ContainsAction(real_legal_2, step_act)) {
                d_real->ApplyAction(step_act);
                AdvanceThroughChanceAndAcks(d_real, prop_chance_rng);
                real_legal_2 = d_real->LegalActions();
              }
            }

            if (ContainsAction(real_legal_2, a_space)) {
              chain_legal_in_real = true;
            }
          }

          if (!chain_legal_in_real) {
            // Plan REQUIRES Windfall!
            rec.plan_requires_windfall = true;
            rec.windfall_reason = "Board space " + DiagnosticActionToString(a_space) + " requires solari from Windfall";
            rec.action_taken = kActionPlayWindfall;
            rec.action_taken_name = "PlayPlotIntrigue(Windfall)";
            rec.windfall_played = true;
            rec.solari_after = rec.actual_solari + 2;
            windfall_played_this_run_ = true;
            decision_records_.push_back(rec);
            return kActionPlayWindfall;
          } else {
            // Space is already affordable without Windfall! Retain Windfall!
            rec.plan_requires_windfall = false;
            rec.windfall_reason = "Board space " + DiagnosticActionToString(a_space) + " is already affordable without Windfall";
            rec.action_taken = a1;
            rec.action_taken_name = DiagnosticActionToString(a1);
            rec.windfall_played = false;
            rec.solari_after = rec.actual_solari;
            decision_records_.push_back(rec);
            RecordResourceFollowUp(a1);
            return a1;
          }
        }
        if (ContainsAction(legal, a1)) {
          rec.plan_requires_windfall = false;
          rec.windfall_reason = "Card selection without identified solari requirement; retaining Windfall";
          rec.action_taken = a1;
          rec.action_taken_name = DiagnosticActionToString(a1);
          rec.windfall_played = false;
          rec.solari_after = rec.actual_solari;
          decision_records_.push_back(rec);
          RecordResourceFollowUp(a1);
          return a1;
        }
      }

      // Case B: a1 is not an agent card selection (e.g. EndTurn, Reveal, or purchase/tech)
      if (a1 == kActionEndTurn) {
        rec.plan_requires_windfall = false;
        rec.windfall_reason = "Plan ends turn without requiring solari";
        rec.action_taken = kActionEndTurn;
        rec.action_taken_name = "EndTurn";
        rec.windfall_played = false;
        rec.solari_after = rec.actual_solari;
        decision_records_.push_back(rec);
        return kActionEndTurn;
      }

      if (a1 == kActionReveal) {
        rec.plan_requires_windfall = false;
        rec.windfall_reason = "Plan reveals without requiring solari";
        rec.action_taken = kActionReveal;
        rec.action_taken_name = "Reveal";
        rec.windfall_played = false;
        rec.solari_after = rec.actual_solari;
        decision_records_.push_back(rec);
        return kActionReveal;
      }

      // Purchase or tech action: test legality in real state
      if (ContainsAction(legal, a1)) {
        rec.plan_requires_windfall = false;
        rec.windfall_reason = "Action " + DiagnosticActionToString(a1) + " is already affordable without Windfall";
        rec.action_taken = a1;
        rec.action_taken_name = DiagnosticActionToString(a1);
        rec.windfall_played = false;
        rec.solari_after = rec.actual_solari;
        decision_records_.push_back(rec);
        RecordResourceFollowUp(a1);
        return a1;
      } else {
        rec.plan_requires_windfall = true;
        rec.windfall_reason = "Action " + DiagnosticActionToString(a1) + " requires solari from Windfall";
        rec.action_taken = kActionPlayWindfall;
        rec.action_taken_name = "PlayPlotIntrigue(Windfall)";
        rec.windfall_played = true;
        rec.solari_after = rec.actual_solari + 2;
        windfall_played_this_run_ = true;
        decision_records_.push_back(rec);
        return kActionPlayWindfall;
      }

    } catch (const std::exception& e) {
      rec.fallback_active = true;
      rec.fallback_reason = std::string("Exception during proposal simulation: ") + e.what();
      Action act = base_evaluator_->SelectAction(state, player, greedy, temperature, rng);
      rec.action_taken = act;
      rec.action_taken_name = DiagnosticActionToString(act);
      rec.windfall_played = (act == kActionPlayWindfall);
      rec.solari_after = rec.windfall_played ? (rec.actual_solari + 2) : rec.actual_solari;
      if (rec.windfall_played) windfall_played_this_run_ = true;
      decision_records_.push_back(rec);
      RecordResourceFollowUp(act);
      return act;
    }
  }

 private:
  IPolicyDiagnosticsEvaluator* base_evaluator_ = nullptr;
  int designated_holder_ = 0;
  int candidate_id_ = -1;
  int replicate_id_ = -1;
  int turn_step_ = 0;
  bool windfall_played_this_run_ = false;
  bool resource_followup_recorded_ = false;
  Action last_resource_action_ = open_spiel::kInvalidAction;
  std::string last_resource_action_name_;
  std::vector<WindfallDecisionRecord> decision_records_;
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

// ===========================================================================
// Scenario Generator Implementation
// ===========================================================================

class ScenarioGenerator {
 public:
  // Constructs the base state for a given scenario config.
  // Records all fixture edits for transparency and auditability.
  static std::unique_ptr<DuneImperiumState> BuildScenarioState(
      const ScenarioConfig& config, std::vector<std::string>* edit_log) {
    auto base_state = BuildStateAtFirstAgentTurn();
    auto* state = dynamic_cast<DuneImperiumState*>(base_state.release());
    SPIEL_CHECK_TRUE(state != nullptr);

    int p_holder = config.p_holder;
    int p_competitor = config.p_competitor;
    int p_opp_a = config.p_opponent_a;
    int p_opp_b = config.p_opponent_b;

    // 1. Set round
    state->SetRoundNumberForTesting(config.round);
    if (edit_log) edit_log->push_back(absl::StrCat("SetRoundNumberForTesting(", config.round, ")"));

    // 2. Set leaders (avoiding Leto to preserve exact 8 solari cost)
    state->SetLeaderForTesting(p_holder, static_cast<int>(LeaderId::kPaulAtreides));
    state->SetLeaderForTesting(p_competitor, static_cast<int>(LeaderId::kGlossuRabban));
    state->SetLeaderForTesting(p_opp_a, static_cast<int>(LeaderId::kIlbanRichese));
    state->SetLeaderForTesting(p_opp_b, static_cast<int>(LeaderId::kHelenaRichese));
    if (edit_log) {
      edit_log->push_back("SetLeaderForTesting(P0=Paul, P1=Rabban, P2=Ilban, P3=Helena)");
    }

    // 3. Set Swordmaster ownership: P3 and P4 already own; P1 and P2 do not.
    state->SetSwordmasterForTesting(p_holder, false);
    state->SetSwordmasterForTesting(p_competitor, false);
    state->SetSwordmasterForTesting(p_opp_a, true);
    state->SetSwordmasterForTesting(p_opp_b, true);
    if (edit_log) {
      edit_log->push_back("SetSwordmasterForTesting(P0=false, P1=false, P2=true, P3=true)");
    }

    // 4. Ensure Swordmaster board space is unoccupied (space index 10 = kSwordmaster)
    state->SetAgentSpaceOwnerForTesting(10, -1);
    if (edit_log) edit_log->push_back("SetAgentSpaceOwnerForTesting(Space10_Swordmaster, -1)");

    // 5. Set Solari: P1 has 8, P2 has 6
    state->SetPlayerSolariForTesting(p_holder, 6);
    state->SetPlayerSolariForTesting(p_competitor, 8);
    if (edit_log) {
      edit_log->push_back("SetPlayerSolariForTesting(P0=6, P1=8)");
    }

    // 6. Set Intrigues: P2 holds Windfall (id 59); competitor has none or non-conflicting
    state->SetPlayerIntriguesForTesting(p_holder, {kIntrigueWindfall});
    state->SetPlayerIntriguesForTesting(p_competitor, {});
    if (edit_log) {
      edit_log->push_back("SetPlayerIntriguesForTesting(P0={kIntrigueWindfall})");
    }

    // 7. Set Agents remaining: both have at least 1 agent available
    state->SetPlayerAgentsRemainingForTesting(p_holder, 1);
    state->SetPlayerAgentsRemainingForTesting(p_competitor, 2);
    state->SetPlayerAgentsRemainingForTesting(p_opp_a, 1);
    state->SetPlayerAgentsRemainingForTesting(p_opp_b, 1);
    if (edit_log) {
      edit_log->push_back("SetPlayerAgentsRemainingForTesting(P0=1, P1=2, P2=1, P3=1)");
    }

    // Remove Family Atomics from all players so free-action chains are not cluttered
    for (int p = 0; p < kNumPlayers; ++p) {
      state->SetAtomicsUsedForTesting(p, true);
    }
    if (edit_log) {
      edit_log->push_back("SetAtomicsUsedForTesting(all=true)");
    }

    // 8. Hands with card conservation (using legal starter cards and standard row cards)
    // P2 hand: must have Landsraad access (Dagger) for intended Swordmaster opportunity
    state->SetPlayerHandForTesting(p_holder, {kCardDagger, kCardConvincingArgument, kCardDuneTheDesertPlanet});
    state->SetPlayerDeckForTesting(p_holder, {kCardDagger, kCardDagger, kCardDiplomacy, kCardReconnaissance});

    // P1 hand: Landsraad access (Signet Ring / Dagger) plus family-tailored cards
    switch (config.family) {
      case ScenarioFamily::kCombatTension: {
        // High-stakes conflict: Siege of Arrakeen / Carthag / Imperial Basin
        state->SetCurrentConflictForTesting(13 + (config.scenario_id % 3)); // 13, 14, or 15
        state->SetTroopsForTesting(p_competitor, /*garrison=*/4, /*combat=*/2);
        state->SetTroopsForTesting(p_opp_a, /*garrison=*/1, /*combat=*/3);
        state->SetTroopsForTesting(p_holder, /*garrison=*/1, /*combat=*/0);
        state->SetPlayerHandForTesting(p_competitor, {kCardDagger, kCardSignetRing, kCardDuneTheDesertPlanet, kCardConvincingArgument});
        if (edit_log) {
          edit_log->push_back("Family_CombatTension: SetConflict(Siege), P1 garrison=4, combat=2, opp combat=3");
        }
        break;
      }
      case ScenarioFamily::kShippingEconomy: {
        // Shipping alternative: P1 has freighter on shipping track, spice available
        state->SetPlayerShippingLevelForTesting(p_competitor, 1 + (config.scenario_id % 2));
        state->SetPlayerSpiceForTesting(p_competitor, 3);
        state->SetPlayerHandForTesting(p_competitor, {kCardDagger, kCardSignetRing, kCardDiplomacy, kCardConvincingArgument});
        if (edit_log) {
          edit_log->push_back("Family_ShippingEconomy: P1 shipping_level=1, spice=3, shipping space open");
        }
        break;
      }
      case ScenarioFamily::kFactionAlliance: {
        // Alliance race on Fremen track: P1 at 3 influence, P3 at 3 influence
        state->SetPlayerInfluenceForTesting(p_competitor, Faction::kFremen, 3);
        state->SetPlayerInfluenceForTesting(p_opp_a, Faction::kFremen, 3);
        state->SetPlayerWaterForTesting(p_competitor, 2);
        state->SetPlayerHandForTesting(p_competitor, {kCardDagger, kCardDiplomacy, kCardSignetRing, kCardDuneTheDesertPlanet});
        if (edit_log) {
          edit_log->push_back("Family_FactionAlliance: P1 Fremen=3, P2 Fremen=3, water=2 (Sietch Tabr race)");
        }
        break;
      }
      case ScenarioFamily::kMentatHighCouncil: {
        // High Council (5 solari) / Mentat (2 solari) alternative
        state->SetPlayerPersuasionForTesting(p_competitor, 2);
        state->SetPlayerHandForTesting(p_competitor, {kCardDagger, kCardSignetRing, kCardDiplomacy, kCardConvincingArgument});
        if (edit_log) {
          edit_log->push_back("Family_MentatHighCouncil: P1 solari=8 (High Council/Mentat reachable)");
        }
        break;
      }
    }

    // 9. P2 turn status: P2 is at the end of its preceding agent turn
    // main action done, EndTurn and Windfall are available
    state->SetCurrentPlayerForTesting(p_holder);
    state->SetCurrentTurnMainActionDoneForTesting(p_holder, true);
    if (edit_log) {
      edit_log->push_back("SetCurrentPlayerForTesting(P0), SetCurrentTurnMainActionDoneForTesting(P0, true)");
    }

    return std::unique_ptr<DuneImperiumState>(state);
  }

  // Verifies all engine legality and scenario requirements on a state.
  static bool VerifyScenarioRequirements(const DuneImperiumState& state,
                                         const ScenarioConfig& config,
                                         std::string* failure_reason = nullptr) {
    int p_holder = config.p_holder;
    int p_competitor = config.p_competitor;
    int p_opp_a = config.p_opponent_a;
    int p_opp_b = config.p_opponent_b;

    // Requirement: Neither P1 nor P2 owns Swordmaster
    if (state.HasSwordmaster(p_competitor)) {
      if (failure_reason) *failure_reason = "Competitor P1 already owns Swordmaster";
      return false;
    }
    if (state.HasSwordmaster(p_holder)) {
      if (failure_reason) *failure_reason = "Holder P2 already owns Swordmaster";
      return false;
    }

    // Requirement: P3 and P4 own Swordmaster
    if (!state.HasSwordmaster(p_opp_a)) {
      if (failure_reason) *failure_reason = "Opponent P3 does not own Swordmaster";
      return false;
    }
    if (!state.HasSwordmaster(p_opp_b)) {
      if (failure_reason) *failure_reason = "Opponent P4 does not own Swordmaster";
      return false;
    }

    // Requirement: Engine legality confirms P3 and P4 cannot acquire Swordmaster
    {
      std::unique_ptr<DuneImperiumState> test_c = 
          std::unique_ptr<DuneImperiumState>(dynamic_cast<DuneImperiumState*>(state.Clone().release()));
      test_c->SetCurrentPlayerForTesting(p_opp_a);
      test_c->SetPhaseForTesting(GamePhase::kAgentTurns);
      test_c->SetCurrentTurnMainActionDoneForTesting(p_opp_a, false);
      if (dune_imperium::SpaceIsReachable(test_c.get(), kActionAgentSpaceSwordmaster)) {
        if (failure_reason) *failure_reason = "Opponent P3 can illegally acquire Swordmaster";
        return false;
      }
      test_c->SetCurrentPlayerForTesting(p_opp_b);
      test_c->SetCurrentTurnMainActionDoneForTesting(p_opp_b, false);
      if (dune_imperium::SpaceIsReachable(test_c.get(), kActionAgentSpaceSwordmaster)) {
        if (failure_reason) *failure_reason = "Opponent P4 can illegally acquire Swordmaster";
        return false;
      }
    }

    // Requirement: Swordmaster board space is unoccupied
    if ((state.GetAgentSpaceOwnerForTesting(10) & 0x0F) != 0) {
      if (failure_reason) *failure_reason = "Swordmaster board space is occupied";
      return false;
    }

    // Requirement: Solari amounts
    if (state.GetPlayerSolari(p_competitor) != 8) {
      if (failure_reason) *failure_reason = "Competitor P1 does not have 8 solari";
      return false;
    }
    if (state.GetPlayerSolari(p_holder) != 6) {
      if (failure_reason) *failure_reason = "Holder P2 does not have 6 solari";
      return false;
    }

    // Requirement: P2 holds Windfall
    auto intrigues = state.GetPlayerIntrigues(p_holder);
    if (std::find(intrigues.begin(), intrigues.end(), kIntrigueWindfall) == intrigues.end()) {
      if (failure_reason) *failure_reason = "Holder P2 does not hold Windfall in intrigue hand";
      return false;
    }

    // Requirement: Both have remaining agents
    if (state.GetPlayerAgentsRemainingForTesting(p_competitor) < 1) {
      if (failure_reason) *failure_reason = "Competitor P1 has no remaining agents";
      return false;
    }
    if (state.GetPlayerAgentsRemainingForTesting(p_holder) < 1) {
      if (failure_reason) *failure_reason = "Holder P2 has no remaining agents";
      return false;
    }

    // Requirement: Start comparison at legal point where Windfall and EndTurn are available
    if (state.CurrentPlayer() != p_holder) {
      if (failure_reason) *failure_reason = "Current player is not Holder P2";
      return false;
    }
    auto holder_legal = state.LegalActions();
    if (!ContainsAction(holder_legal, kActionPlayWindfall)) {
      if (failure_reason) *failure_reason = "Play Windfall action is not legal for P2";
      return false;
    }
    if (!ContainsAction(holder_legal, kActionEndTurn)) {
      if (failure_reason) *failure_reason = "EndTurn action is not legal for P2";
      return false;
    }

    // Test Branch A and Branch B divergence on clones
    std::mt19937_64 check_c_rng(12345);
    auto clone_a = state.Clone();
    auto* da = dynamic_cast<DuneImperiumState*>(clone_a.get());
    da->ApplyAction(kActionPlayWindfall);
    AdvanceThroughChanceAndAcks(da, check_c_rng);
    if (da->GetPlayerSolari(p_holder) != 8) {
      if (failure_reason) *failure_reason = "Branch A did not produce 8 visible solari";
      return false;
    }
    da->ApplyAction(kActionEndTurn);
    AdvanceThroughChanceAndAcks(da, check_c_rng);

    auto clone_b = state.Clone();
    auto* db = dynamic_cast<DuneImperiumState*>(clone_b.get());
    db->ApplyAction(kActionEndTurn);
    AdvanceThroughChanceAndAcks(db, check_c_rng);
    if (db->GetPlayerSolari(p_holder) != 6) {
      if (failure_reason) *failure_reason = "Branch B did not preserve 6 visible solari";
      return false;
    }

    // Confirm both branches advance to competitor P1 before P2 acts again
    if (da->CurrentPlayer() != p_competitor) {
      if (failure_reason) *failure_reason = "Branch A did not advance to P1";
      return false;
    }
    if (db->CurrentPlayer() != p_competitor) {
      if (failure_reason) *failure_reason = "Branch B did not advance to P1";
      return false;
    }

    // Confirm Swordmaster is reachable for P1
    if (!dune_imperium::SpaceIsReachable(da, kActionAgentSpaceSwordmaster)) {
      if (failure_reason) *failure_reason = "Swordmaster is not reachable for P1 in Branch A";
      return false;
    }
    if (!dune_imperium::SpaceIsReachable(db, kActionAgentSpaceSwordmaster)) {
      if (failure_reason) *failure_reason = "Swordmaster is not reachable for P1 in Branch B";
      return false;
    }

    return true;
  }

  // Generates a complete, balanced scenario bank across the 4 families.
  static std::vector<ScenarioConfig> GenerateBank(int count = 128,
                                                  uint64_t base_seed = 20260915,
                                                  int* rejected_count = nullptr) {
    std::vector<ScenarioConfig> bank;
    bank.reserve(count);
    int rejected = 0;

    int per_family = count / 4;
    int remainder = count % 4;

    int id = 0;
    for (int f = 0; f < 4; ++f) {
      ScenarioFamily fam = static_cast<ScenarioFamily>(f);
      int fam_count = per_family + (f < remainder ? 1 : 0);
      for (int i = 0; i < fam_count; ++i) {
        ScenarioConfig cfg;
        cfg.scenario_id = id;
        cfg.family = fam;
        cfg.family_name = ScenarioFamilyToString(fam);
        cfg.round = 2 + (i % 4); // rounds 2, 3, 4, 5
        cfg.p_holder = 0;
        cfg.p_competitor = 1;
        cfg.p_opponent_a = 2;
        cfg.p_opponent_b = 3;
        cfg.scenario_seed = dune_seed::DeriveSeed(base_seed, 0x00A0, id, f, i);

        auto state = BuildScenarioState(cfg, &cfg.fixture_edits);
        std::string err;
        if (VerifyScenarioRequirements(*state, cfg, &err)) {
          bank.push_back(cfg);
          id++;
        } else {
          rejected++;
          std::cerr << "Warning: Candidate scenario " << id << " failed: " << err << "\n";
        }
      }
    }

    if (rejected_count) *rejected_count = rejected;
    return bank;
  }
};

// ===========================================================================
// Diagnostic 1: Opponent Response Execution Engine
// ===========================================================================

inline ReplicateResult RunSingleReplicate(
    const ScenarioConfig& cfg,
    int replicate_idx,
    IPolicyDiagnosticsEvaluator* evaluator,
    uint64_t base_seed,
    bool is_greedy,
    float temperature) {
  ReplicateResult res;
  res.scenario_id = cfg.scenario_id;
  res.replicate_idx = replicate_idx;
  res.is_greedy = is_greedy;

  res.chance_seed = dune_seed::DeriveSeed(
      base_seed, dune_seed::kDomainEvalBaseline, cfg.scenario_id, replicate_idx,
      dune_seed::kStreamChance);
  res.policy_seed = dune_seed::DeriveSeed(
      base_seed, dune_seed::kDomainEvalBaseline, cfg.scenario_id, replicate_idx,
      dune_seed::kStreamPolicyPlayer0 + cfg.p_competitor);

  // Helper lambda to run one branch
  auto run_branch = [&](bool is_early) -> std::pair<Action, int> {
    auto base_state = ScenarioGenerator::BuildScenarioState(cfg, nullptr);
    auto* s = base_state.get();

    std::mt19937_64 chance_rng(res.chance_seed);
    std::mt19937_64 policy_rng(res.policy_seed);

    if (is_early) {
      // Branch A (EARLY): P2 plays Windfall, then ends turn
      s->ApplyAction(kActionPlayWindfall);
      AdvanceThroughChanceAndAcks(s, chance_rng);
      if (ContainsAction(s->LegalActions(), kActionEndTurn)) {
        s->ApplyAction(kActionEndTurn);
      }
      AdvanceThroughChanceAndAcks(s, chance_rng);
    } else {
      // Branch B (HOLD): P2 ends turn holding Windfall
      s->ApplyAction(kActionEndTurn);
      AdvanceThroughChanceAndAcks(s, chance_rng);
    }

    // Advance until P1 commits to a board space, reveals, or game ends
    int steps = 0;
    Action committed_action = open_spiel::kInvalidAction;

    while (!s->IsTerminal() && steps < 100) {
      steps++;
      AdvanceStatus st = AdvanceThroughChanceAndAcks(s, chance_rng);
      if (st != AdvanceStatus::kSuccessDecision) break;

      Player cp = s->CurrentPlayer();
      if (cp != cfg.p_competitor) {
        // Intervening player action
        Action a = evaluator->SelectAction(*s, cp, is_greedy, temperature, policy_rng);
        s->ApplyAction(a);
        continue;
      }

      // Current player is P1 (competitor)
      Action action = evaluator->SelectAction(*s, cp, is_greedy, temperature, policy_rng);

      // Check if this action is a committed placement or reveal
      if (IsAgentSpaceAction(action)) {
        committed_action = action;
        break;
      }
      if (action == kActionReveal || action == kActionAgentPass) {
        committed_action = action;
        break;
      }

      // Intermediate action (card selection, graft partner, solo, or free intrigue)
      s->ApplyAction(action);
    }

    return {committed_action, steps};
  };

  // Run Branch A
  auto [action_a, steps_a] = run_branch(/*is_early=*/true);
  res.branch_a_chosen_action = action_a;
  res.branch_a_action_name = (action_a == kActionReveal) ? "Reveal" :
                             (action_a == kActionAgentPass) ? "Pass" :
                             AgentSpaceNameForAction(action_a);
  res.branch_a_chose_swordmaster = (action_a == kActionAgentSpaceSwordmaster);
  res.branch_a_steps = steps_a;

  // Run Branch B
  auto [action_b, steps_b] = run_branch(/*is_early=*/false);
  res.branch_b_chosen_action = action_b;
  res.branch_b_action_name = (action_b == kActionReveal) ? "Reveal" :
                             (action_b == kActionAgentPass) ? "Pass" :
                             AgentSpaceNameForAction(action_b);
  res.branch_b_chose_swordmaster = (action_b == kActionAgentSpaceSwordmaster);
  res.branch_b_steps = steps_b;

  res.delta_swordmaster = (res.branch_a_chose_swordmaster ? 1.0 : 0.0) -
                          (res.branch_b_chose_swordmaster ? 1.0 : 0.0);
  return res;
}

// Runs Diagnostic 1 for all scenarios and replicates
inline std::pair<std::vector<ScenarioSummary>, Diagnostic1Aggregate>
RunDiagnostic1Suite(
    const std::vector<ScenarioConfig>& bank,
    IPolicyDiagnosticsEvaluator* evaluator,
    int replicates_per_scenario = 64,
    uint64_t base_seed = 20260915,
    float temperature = 1.0f,
    std::vector<ReplicateResult>* raw_records = nullptr) {

  std::vector<ScenarioSummary> summaries;
  summaries.reserve(bank.size());

  Diagnostic1Aggregate agg;
  agg.total_scenarios = static_cast<int>(bank.size());
  agg.replicates_per_scenario = replicates_per_scenario;

  std::vector<double> scenario_effects;
  scenario_effects.reserve(bank.size());

  std::map<std::string, std::vector<double>> family_effects_raw;

  int total_a_sm = 0;
  int total_b_sm = 0;
  int total_reps = 0;

  int greedy_a_sm = 0;
  int greedy_b_sm = 0;

  for (const auto& cfg : bank) {
    ScenarioSummary ss;
    ss.scenario_id = cfg.scenario_id;
    ss.family = cfg.family_name;
    ss.round = cfg.round;
    ss.num_replicates = replicates_per_scenario;

    int sm_a_count = 0;
    int sm_b_count = 0;

    for (int r = 0; r < replicates_per_scenario; ++r) {
      auto rep = RunSingleReplicate(cfg, r, evaluator, base_seed,
                                    /*is_greedy=*/false, temperature);
      if (raw_records) raw_records->push_back(rep);

      if (rep.branch_a_chose_swordmaster) sm_a_count++;
      if (rep.branch_b_chose_swordmaster) sm_b_count++;

      ss.branch_a_spaces[rep.branch_a_action_name]++;
      ss.branch_b_spaces[rep.branch_b_action_name]++;
      agg.alternative_spaces_branch_a[rep.branch_a_action_name]++;
      agg.alternative_spaces_branch_b[rep.branch_b_action_name]++;
    }

    ss.branch_a_swordmaster_freq = static_cast<double>(sm_a_count) / replicates_per_scenario;
    ss.branch_b_swordmaster_freq = static_cast<double>(sm_b_count) / replicates_per_scenario;
    ss.effect = ss.branch_a_swordmaster_freq - ss.branch_b_swordmaster_freq;

    ss.is_saturated_always = (ss.branch_a_swordmaster_freq >= 0.95 && ss.branch_b_swordmaster_freq >= 0.95);
    ss.is_saturated_never = (ss.branch_a_swordmaster_freq <= 0.05 && ss.branch_b_swordmaster_freq <= 0.05);

    if (ss.is_saturated_always) agg.saturated_always_count++;
    else if (ss.is_saturated_never) agg.saturated_never_count++;
    else agg.sensitive_count++;

    // Evaluate greedy policy for this scenario
    auto greedy_rep = RunSingleReplicate(cfg, 0, evaluator, base_seed,
                                         /*is_greedy=*/true, temperature);
    ss.greedy_a_action = greedy_rep.branch_a_chosen_action;
    ss.greedy_a_action_name = greedy_rep.branch_a_action_name;
    ss.greedy_a_swordmaster = greedy_rep.branch_a_chose_swordmaster;
    ss.greedy_b_action = greedy_rep.branch_b_chosen_action;
    ss.greedy_b_action_name = greedy_rep.branch_b_action_name;
    ss.greedy_b_swordmaster = greedy_rep.branch_b_chose_swordmaster;
    ss.greedy_effect = (ss.greedy_a_swordmaster ? 1.0 : 0.0) - (ss.greedy_b_swordmaster ? 1.0 : 0.0);

    if (ss.greedy_a_swordmaster) greedy_a_sm++;
    if (ss.greedy_b_swordmaster) greedy_b_sm++;

    total_a_sm += sm_a_count;
    total_b_sm += sm_b_count;
    total_reps += replicates_per_scenario;

    scenario_effects.push_back(ss.effect);
    family_effects_raw[ss.family].push_back(ss.effect);

    summaries.push_back(ss);
  }

  agg.total_paired_trials = total_reps;
  agg.mean_branch_a_freq = static_cast<double>(total_a_sm) / total_reps;
  agg.branch_a_wilson = ComputeWilsonInterval(total_a_sm, total_reps);
  agg.mean_branch_b_freq = static_cast<double>(total_b_sm) / total_reps;
  agg.branch_b_wilson = ComputeWilsonInterval(total_b_sm, total_reps);
  agg.effect_ci = ComputeMeanAndCI(scenario_effects);

  agg.greedy_branch_a_freq = static_cast<double>(greedy_a_sm) / bank.size();
  agg.greedy_branch_b_freq = static_cast<double>(greedy_b_sm) / bank.size();
  agg.greedy_mean_effect = agg.greedy_branch_a_freq - agg.greedy_branch_b_freq;

  for (const auto& [fam, effs] : family_effects_raw) {
    agg.family_effects[fam] = ComputeMeanAndCI(effs);
  }

  return {summaries, agg};
}

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

// ===========================================================================
// Secondary Diagnostic: Full-Game Continuation
// ===========================================================================

inline FullGameContinuationResult RunContinuationReplicate(
    const ScenarioConfig& cfg,
    int replicate_idx,
    IPolicyDiagnosticsEvaluator* evaluator,
    uint64_t base_seed,
    const std::string& mode, // "unchanged" or "controlled"
    float temperature) {

  FullGameContinuationResult res;
  res.scenario_id = cfg.scenario_id;
  res.replicate_idx = replicate_idx;
  res.mode = mode;

  uint64_t chance_seed = dune_seed::DeriveSeed(
      base_seed, dune_seed::kDomainEvalBaseline, cfg.scenario_id, replicate_idx,
      dune_seed::kStreamChance);
  uint64_t policy_seed = dune_seed::DeriveSeed(
      base_seed, dune_seed::kDomainEvalBaseline, cfg.scenario_id, replicate_idx,
      dune_seed::kStreamPolicyPlayer0);

  auto run_full_game = [&](bool is_early) -> std::pair<double, int> {
    auto base_state = ScenarioGenerator::BuildScenarioState(cfg, nullptr);
    auto* s = base_state.get();
    std::mt19937_64 chance_rng(chance_seed);
    std::mt19937_64 policy_rng(policy_seed);

    if (is_early) {
      s->ApplyAction(kActionPlayWindfall);
      AdvanceThroughChanceAndAcks(s, chance_rng);
      if (ContainsAction(s->LegalActions(), kActionEndTurn)) {
        s->ApplyAction(kActionEndTurn);
      }
      AdvanceThroughChanceAndAcks(s, chance_rng);
    } else {
      s->ApplyAction(kActionEndTurn);
      AdvanceThroughChanceAndAcks(s, chance_rng);
    }

    int steps = 0;
    while (!s->IsTerminal() && steps < 5000) {
      steps++;
      AdvanceStatus st = AdvanceThroughChanceAndAcks(s, chance_rng);
      if (st == AdvanceStatus::kSuccessTerminal) break;
      if (st != AdvanceStatus::kSuccessDecision) break;

      auto legal = s->LegalActions();
      if (legal.empty()) break;

      Player cp = s->CurrentPlayer();

      if (!is_early && mode == "controlled" && cp == cfg.p_holder &&
          ContainsAction(legal, kActionPlayWindfall)) {
        // Controlled delayed timing plan:
        // When P2's turn arrives and Windfall is in hand:
        // Check if Swordmaster space is unoccupied and reachable
        if (dune_imperium::SpaceIsReachable(s, kActionAgentSpaceSwordmaster)) {
          res.delayed_opportunity_available = true;
          // Play Windfall now (delayed)
          s->ApplyAction(kActionPlayWindfall);
          AdvanceThroughChanceAndAcks(s, chance_rng);
          // Now apply Swordmaster
          dune_imperium::SelectCardAndApplySpace(s, kActionAgentSpaceSwordmaster);
          continue;
        } else {
          res.intended_opportunity_disappeared = true;
          // Legal fallback: play unchanged policy
        }
      }

      Action a = evaluator->SelectAction(*s, cp, /*greedy=*/false, temperature, policy_rng);
      s->ApplyAction(a);
    }

    // Explicit terminal check before reading returns
    double util = 0.0;
    if (s->IsTerminal()) {
      util = s->Returns()[cfg.p_holder];
    } else {
      std::cerr << "Warning: Continuation replicate reached step limit without terminating.\n";
    }
    int r = s->GetCurrentRound();
    return {util, r};
  };

  auto [util_a, r_a] = run_full_game(/*is_early=*/true);
  res.branch_a_p2_utility = util_a;
  res.branch_a_ending_round = r_a;

  auto [util_b, r_b] = run_full_game(/*is_early=*/false);
  res.branch_b_p2_utility = util_b;
  res.branch_b_ending_round = r_b;

  return res;
}

// ===========================================================================
// Actual-Play Intrigue Mining Engine
// ===========================================================================

inline std::string LeaderName(LeaderId id) {
  for (const auto& entry : dune_imperium::kLeaders) {
    if (entry.id == static_cast<int>(id)) return std::string(entry.name);
  }
  return absl::StrCat("Leader_", static_cast<int>(id));
}

// Advances both branches to the designated target opponent's main placement decision (board space 600..621,
// Reveal, or AgentPass). Any intervening player actions before the target opponent are simulated using
// that player's dedicated policy RNG stream.
inline bool AdvanceToOpponentMainPlacement(
    DuneImperiumState* s,
    Player target_opponent,
    IPolicyDiagnosticsEvaluator* evaluator,
    std::mt19937_64& chance_rng,
    std::array<std::mt19937_64, kNumPlayers>& policy_rngs,
    bool greedy,
    float temperature,
    Action* out_opp_action,
    ModelInputSnapshot* out_snapshot = nullptr,
    int max_steps = 300) {

  int steps = 0;
  while (!s->IsTerminal() && steps < max_steps) {
    steps++;
    AdvanceStatus st = AdvanceThroughChanceAndAcks(s, chance_rng);
    if (st != AdvanceStatus::kSuccessDecision) {
      return false;
    }

    Player cp = s->CurrentPlayer();
    auto legal = s->LegalActions();
    if (legal.empty()) return false;

    if (cp == target_opponent) {
      Action act = evaluator->SelectAction(*s, cp, greedy, temperature, policy_rngs[cp]);
      if (IsAgentSpaceAction(act) || act == kActionReveal || act == kActionAgentPass) {
        if (out_opp_action != nullptr) *out_opp_action = act;
        if (out_snapshot != nullptr) {
          *out_snapshot = evaluator->CaptureSnapshot(*s, cp);
        }
        return true;
      }
      // Card selection, graft partner, or free intrigue
      s->ApplyAction(act);
    } else {
      // Intervening action by another player
      Action act = evaluator->SelectAction(*s, cp, greedy, temperature, policy_rngs[cp]);
      s->ApplyAction(act);
    }
  }
  return false;
}

// Step-down Holm-Bonferroni adjusted p-values for multiple testing control.
inline std::vector<double> ComputeHolmBonferroniAdjustedPValues(const std::vector<double>& raw_p) {
  size_t m = raw_p.size();
  if (m == 0) return {};
  if (m == 1) return {raw_p[0]};

  std::vector<size_t> order(m);
  std::iota(order.begin(), order.end(), 0);
  std::sort(order.begin(), order.end(), [&raw_p](size_t i, size_t j) {
    return raw_p[i] < raw_p[j];
  });

  std::vector<double> adj_p(m, 0.0);
  double running_max = 0.0;
  for (size_t k = 0; k < m; ++k) {
    size_t idx = order[k];
    double multiplier = static_cast<double>(m - k);
    double cur_adj = std::min(1.0, raw_p[idx] * multiplier);
    running_max = std::max(running_max, cur_adj);
    adj_p[idx] = running_max;
  }
  return adj_p;
}

// Simulates ordinary self-play games from NewInitialState() without card grants or artificial modifications.
// Collects candidate roots, performs complete model input replay verification, runs paired confirmations
// with Holm-Bonferroni correction, and runs full-game continuations with explicit terminal state checks.
inline IntrigueMiningSummary RunOrdinaryGameIntrigueMining(
    IPolicyDiagnosticsEvaluator* evaluator,
    int num_games = 100,
    uint64_t base_seed = 20260915,
    int discovery_pairs = 4,
    int max_candidates = 8,
    int confirmation_samples = 64,
    int max_outcome_candidates = 4,
    int continuation_samples = 64,
    float temperature = 1.0f) {

  IntrigueMiningSummary summary;
  summary.games_simulated = num_games;
  int candidate_counter = 0;

  auto game = open_spiel::LoadGame("dune_imperium");

  for (int g = 0; g < num_games; ++g) {
    uint64_t game_chance_seed = dune_seed::DeriveSeed(base_seed, 0x00E0, g, 0, dune_seed::kStreamChance);
    std::array<uint64_t, kNumPlayers> game_policy_seeds;
    std::array<std::mt19937_64, kNumPlayers> game_policy_rngs;
    for (int p = 0; p < kNumPlayers; ++p) {
      game_policy_seeds[p] = dune_seed::DeriveSeed(base_seed, 0x00E0, g, 0, dune_seed::kStreamPolicyPlayer0 + p);
      game_policy_rngs[p].seed(game_policy_seeds[p]);
    }
    std::mt19937_64 game_chance_rng(game_chance_seed);

    std::unique_ptr<State> state = game->NewInitialState();
    auto* s = dynamic_cast<DuneImperiumState*>(state.get());

    std::vector<Action> history;
    int steps = 0;
    while (!s->IsTerminal() && steps < 3000) {
      steps++;
      AdvanceStatus st = AdvanceThroughChanceAndAcks(s, game_chance_rng, &history);
      if (st != AdvanceStatus::kSuccessDecision) break;

      Player cp = s->CurrentPlayer();
      auto legal = s->LegalActions();
      if (legal.empty()) break;

      Action chosen = evaluator->SelectAction(*s, cp, /*greedy=*/false, temperature, game_policy_rngs[cp]);

      // Check if chosen action is an early resource intrigue play while EndTurn is legal
      bool is_resource_intrigue = (chosen == kActionPlayWindfall || chosen == kActionPlayWaterPeddlersUnion);
      if (is_resource_intrigue && ContainsAction(legal, kActionEndTurn)) {
        summary.total_resource_intrigue_plays++;
        if (chosen == kActionPlayWindfall) summary.windfall_plays++;
        if (chosen == kActionPlayWaterPeddlersUnion) summary.water_peddlers_union_plays++;

        // Determine designated target opponent before branching:
        // Probe Branch B to see who the next active player will be
        auto temp_b = s->Clone();
        temp_b->ApplyAction(kActionEndTurn);
        std::mt19937_64 temp_c_rng(game_chance_seed + steps);
        AdvanceStatus temp_st = AdvanceThroughChanceAndAcks(temp_b.get(), temp_c_rng);
        Player designated_opponent = (temp_st == AdvanceStatus::kSuccessDecision) ? temp_b->CurrentPlayer() : -1;

        if (designated_opponent != -1 && designated_opponent != cp) {
          IntrigueMiningCandidate cand;
          cand.candidate_id = candidate_counter++;
          cand.game_id = g;
          cand.game_seed = game_chance_seed;
          cand.step_index = steps;
          cand.round = s->GetCurrentRound();
          cand.holder_player = cp;
          cand.holder_leader = LeaderName(static_cast<LeaderId>(s->PlayerLeader(cp)));
          cand.intrigue_action = chosen;
          cand.intrigue_name = (chosen == kActionPlayWindfall ? "Windfall" : "Water Peddlers Union");
          cand.holder_solari = s->GetPlayerSolari(cp);
          cand.holder_water = s->GetPlayerWaterForTesting(cp);
          cand.holder_spice = s->GetPlayerSpiceForTesting(cp);

          cand.opponent_player = designated_opponent;
          cand.opponent_leader = LeaderName(static_cast<LeaderId>(s->PlayerLeader(designated_opponent)));
          cand.opponent_solari = s->GetPlayerSolari(designated_opponent);
          cand.opponent_water = s->GetPlayerWaterForTesting(designated_opponent);
          cand.opponent_spice = s->GetPlayerSpiceForTesting(designated_opponent);
          cand.opponent_agents = s->GetPlayerAgentsRemainingForTesting(designated_opponent);

          NaturalRootRecord root_rec;
          root_rec.candidate_id = cand.candidate_id;
          root_rec.game_id = g;
          root_rec.game_seed = game_chance_seed;
          root_rec.round = cand.round;
          root_rec.step_index = steps;
          root_rec.holder_player = cp;
          root_rec.holder_leader = cand.holder_leader;
          root_rec.holder_solari = cand.holder_solari;
          root_rec.holder_water = cand.holder_water;
          root_rec.holder_spice = cand.holder_spice;
          root_rec.target_opponent = designated_opponent;
          root_rec.opponent_leader = cand.opponent_leader;
          root_rec.opponent_solari = cand.opponent_solari;
          root_rec.opponent_water = cand.opponent_water;
          root_rec.opponent_spice = cand.opponent_spice;
          root_rec.opponent_agents = cand.opponent_agents;
          root_rec.intrigue_action = chosen;
          root_rec.intrigue_name = cand.intrigue_name;
          root_rec.history_actions = history;
          root_rec.base_snapshot = evaluator->CaptureSnapshot(*s, cp);

          // 4 Discovery paired rollouts
          int divergent_trials = 0;
          std::map<Action, int> count_a, count_b;
          Action first_act_a = open_spiel::kInvalidAction, first_act_b = open_spiel::kInvalidAction;

          for (int disc_trial = 0; disc_trial < discovery_pairs; ++disc_trial) {
            uint64_t disc_chance_seed = dune_seed::DeriveSeed(game_chance_seed, 0x00E1, cand.candidate_id, disc_trial, dune_seed::kStreamChance);
            std::array<std::mt19937_64, kNumPlayers> disc_p_rngs_a, disc_p_rngs_b;
            for (int p = 0; p < kNumPlayers; ++p) {
              uint64_t s_p = dune_seed::DeriveSeed(game_chance_seed, 0x00E1, cand.candidate_id, disc_trial, dune_seed::kStreamPolicyPlayer0 + p);
              disc_p_rngs_a[p].seed(s_p);
              disc_p_rngs_b[p].seed(s_p);
            }
            std::mt19937_64 c_rng_a(disc_chance_seed);
            std::mt19937_64 c_rng_b(disc_chance_seed);

            // Branch A
            auto da = s->Clone();
            da->ApplyAction(chosen);
            AdvanceThroughChanceAndAcks(da.get(), c_rng_a);
            if (ContainsAction(da->LegalActions(), kActionEndTurn)) {
              da->ApplyAction(kActionEndTurn);
            }
            Action cur_act_a = open_spiel::kInvalidAction;
            bool ok_a = AdvanceToOpponentMainPlacement(
                dynamic_cast<DuneImperiumState*>(da.get()), designated_opponent, evaluator,
                c_rng_a, disc_p_rngs_a, /*greedy=*/false, temperature, &cur_act_a);

            // Branch B
            auto db = s->Clone();
            db->ApplyAction(kActionEndTurn);
            Action cur_act_b = open_spiel::kInvalidAction;
            bool ok_b = AdvanceToOpponentMainPlacement(
                dynamic_cast<DuneImperiumState*>(db.get()), designated_opponent, evaluator,
                c_rng_b, disc_p_rngs_b, /*greedy=*/false, temperature, &cur_act_b);

            if (ok_a && ok_b) {
              count_a[cur_act_a]++;
              count_b[cur_act_b]++;
              if (disc_trial == 0) {
                first_act_a = cur_act_a;
                first_act_b = cur_act_b;
              }
              if (cur_act_a != cur_act_b) {
                divergent_trials++;
              }
            }
          }

          cand.discovery_divergent_count = divergent_trials;
          cand.discovery_opp_action_early = first_act_a;
          cand.discovery_opp_action_early_name = DiagnosticActionToString(first_act_a);
          cand.discovery_opp_action_hold = first_act_b;
          cand.discovery_opp_action_hold_name = DiagnosticActionToString(first_act_b);

          std::set<Action> all_actions;
          for (const auto& [a, c] : count_a) all_actions.insert(a);
          for (const auto& [a, c] : count_b) all_actions.insert(a);

          Action best_target = first_act_a;
          double max_shift = -1.0;
          int best_dir = 1;
          for (Action a : all_actions) {
            double freq_a = static_cast<double>(count_a[a]) / discovery_pairs;
            double freq_b = static_cast<double>(count_b[a]) / discovery_pairs;
            double diff = freq_a - freq_b;
            if (std::abs(diff) > max_shift) {
              max_shift = std::abs(diff);
              best_target = a;
              best_dir = (diff >= 0.0) ? 1 : -1;
            }
          }
          cand.target_action = best_target;
          cand.target_action_name = DiagnosticActionToString(best_target);
          cand.predicted_direction = best_dir;
          cand.discovery_delta = (discovery_pairs > 0) ?
              (static_cast<double>(count_a[best_target] - count_b[best_target]) / discovery_pairs) : 0.0;

          summary.candidates.push_back(cand);
          summary.roots.push_back(root_rec);
          summary.candidates_discovered++;
        }
      }

      s->ApplyAction(chosen);
      history.push_back(chosen);
    }
  }

  // Deterministic candidate ranking rule:
  // 1) Disagreement count descending
  // 2) Effect magnitude descending
  // 3) Game ID ascending
  // 4) Step index ascending
  std::vector<size_t> candidate_indices(summary.candidates.size());
  std::iota(candidate_indices.begin(), candidate_indices.end(), 0);
  std::sort(candidate_indices.begin(), candidate_indices.end(), [&](size_t i, size_t j) {
    const auto& a = summary.candidates[i];
    const auto& b = summary.candidates[j];
    if (a.discovery_divergent_count != b.discovery_divergent_count) {
      return a.discovery_divergent_count > b.discovery_divergent_count;
    }
    if (std::abs(a.discovery_delta) != std::abs(b.discovery_delta)) {
      return std::abs(a.discovery_delta) > std::abs(b.discovery_delta);
    }
    if (a.game_id != b.game_id) {
      return a.game_id < b.game_id;
    }
    return a.step_index < b.step_index;
  });

  std::vector<size_t> selected_confirmation_indices;
  for (size_t idx : candidate_indices) {
    if (static_cast<int>(selected_confirmation_indices.size()) >= max_candidates) break;
    selected_confirmation_indices.push_back(idx);
  }
  summary.candidates_tested_confirmation = static_cast<int>(selected_confirmation_indices.size());

  // Run Confirmation Pass on selected candidates
  for (size_t idx : selected_confirmation_indices) {
    auto& cand = summary.candidates[idx];
    auto& root = summary.roots[idx];

    // 1. Complete Replay Verification
    auto replay_res = VerifyReplayHistory(game, root.history_actions, cand.holder_player, root.base_snapshot, evaluator);
    cand.replay_verified = replay_res.matches;
    cand.replay_failure_reason = replay_res.failure_reason;
    root.replay_verified = replay_res.matches;
    root.replay_failure_reason = replay_res.failure_reason;

    // 2. Reconstruct base state
    std::unique_ptr<State> base_state = game->NewInitialState();
    for (Action a : root.history_actions) {
      base_state->ApplyAction(a);
    }

    // Direct placement argmax without sampling
    auto da_direct = base_state->Clone();
    da_direct->ApplyAction(cand.intrigue_action);
    std::mt19937_64 direct_c_rng(12345);
    AdvanceThroughChanceAndAcks(da_direct.get(), direct_c_rng);
    if (ContainsAction(da_direct->LegalActions(), kActionEndTurn)) {
      da_direct->ApplyAction(kActionEndTurn);
    }
    AdvanceThroughChanceAndAcks(da_direct.get(), direct_c_rng);

    auto db_direct = base_state->Clone();
    db_direct->ApplyAction(kActionEndTurn);
    AdvanceThroughChanceAndAcks(db_direct.get(), direct_c_rng);

    if (da_direct->CurrentPlayer() == cand.opponent_player && db_direct->CurrentPlayer() == cand.opponent_player) {
      auto policy_a = evaluator->GetPolicy(*da_direct, cand.opponent_player);
      auto policy_b = evaluator->GetPolicy(*db_direct, cand.opponent_player);
      double max_a = -1.0, max_b = -1.0;
      Action arg_a = open_spiel::kInvalidAction, arg_b = open_spiel::kInvalidAction;
      for (const auto& [a, p] : policy_a) {
        if (p > max_a) { max_a = p; arg_a = a; }
        if (a == cand.target_action) cand.opponent_target_prob_early = p;
      }
      for (const auto& [a, p] : policy_b) {
        if (p > max_b) { max_b = p; arg_b = a; }
        if (a == cand.target_action) cand.opponent_target_prob_hold = p;
      }
      cand.opponent_argmax_early = arg_a;
      cand.opponent_argmax_early_name = DiagnosticActionToString(arg_a);
      cand.opponent_argmax_hold = arg_b;
      cand.opponent_argmax_hold_name = DiagnosticActionToString(arg_b);
    }

    // 3. Fresh sample rollouts (independent seeds from discovery)
    cand.confirmation_samples = confirmation_samples;
    std::vector<int> diffs;
    int div_count = 0;
    int early_target_cnt = 0;
    int hold_target_cnt = 0;

    for (int m = 0; m < confirmation_samples; ++m) {
      uint64_t fresh_chance_seed = dune_seed::DeriveSeed(cand.game_seed, 0x00E2, cand.candidate_id, m, dune_seed::kStreamChance);
      std::array<std::mt19937_64, kNumPlayers> p_rngs_a, p_rngs_b;
      std::array<uint64_t, kNumPlayers> p_seeds;
      for (int p = 0; p < kNumPlayers; ++p) {
        p_seeds[p] = dune_seed::DeriveSeed(cand.game_seed, 0x00E2, cand.candidate_id, m, dune_seed::kStreamPolicyPlayer0 + p);
        p_rngs_a[p].seed(p_seeds[p]);
        p_rngs_b[p].seed(p_seeds[p]);
      }
      std::mt19937_64 c_rng_a(fresh_chance_seed);
      std::mt19937_64 c_rng_b(fresh_chance_seed);

      // Branch A
      auto sa = base_state->Clone();
      sa->ApplyAction(cand.intrigue_action);
      AdvanceThroughChanceAndAcks(sa.get(), c_rng_a);
      if (ContainsAction(sa->LegalActions(), kActionEndTurn)) {
        sa->ApplyAction(kActionEndTurn);
      }
      Action act_a = open_spiel::kInvalidAction;
      bool ok_a = AdvanceToOpponentMainPlacement(
          dynamic_cast<DuneImperiumState*>(sa.get()), cand.opponent_player, evaluator,
          c_rng_a, p_rngs_a, /*greedy=*/false, temperature, &act_a);

      // Branch B
      auto sb = base_state->Clone();
      sb->ApplyAction(kActionEndTurn);
      Action act_b = open_spiel::kInvalidAction;
      bool ok_b = AdvanceToOpponentMainPlacement(
          dynamic_cast<DuneImperiumState*>(sb.get()), cand.opponent_player, evaluator,
          c_rng_b, p_rngs_b, /*greedy=*/false, temperature, &act_b);

      PairedConfirmationRecord conf_rec;
      conf_rec.candidate_id = cand.candidate_id;
      conf_rec.pair_index = m;
      conf_rec.chance_seed = fresh_chance_seed;
      conf_rec.policy_seeds = p_seeds;
      conf_rec.holder_player = cand.holder_player;
      conf_rec.target_opponent = cand.opponent_player;
      conf_rec.reached_target_early = ok_a;
      conf_rec.reached_target_hold = ok_b;
      conf_rec.act_early = act_a;
      conf_rec.act_early_name = DiagnosticActionToString(act_a);
      conf_rec.act_hold = act_b;
      conf_rec.act_hold_name = DiagnosticActionToString(act_b);

      if (ok_a && ok_b) {
        if (act_a != act_b) div_count++;
        int ind_a = (act_a == cand.target_action) ? 1 : 0;
        int ind_b = (act_b == cand.target_action) ? 1 : 0;
        if (ind_a) early_target_cnt++;
        if (ind_b) hold_target_cnt++;
        int diff = ind_a - ind_b;
        diffs.push_back(diff);
        conf_rec.indicator_early = ind_a;
        conf_rec.indicator_hold = ind_b;
        conf_rec.indicator_diff = diff;
        conf_rec.is_valid = true;
      } else {
        conf_rec.is_valid = false;
        summary.run_has_invalidation_failure = true;
        summary.invalidation_reason = "Confirmation pair failed to reach target opponent in both branches";
      }
      summary.confirmation_records.push_back(conf_rec);
    }

    cand.confirmed_divergent_count = div_count;
    cand.confirmed_divergent_rate = (confirmation_samples > 0) ?
        (static_cast<double>(div_count) / confirmation_samples) : 0.0;
    cand.confirmation_early_target_count = early_target_cnt;
    cand.confirmation_hold_target_count = hold_target_cnt;

    if (!diffs.empty()) {
      double n = static_cast<double>(diffs.size());
      double sum_d = std::accumulate(diffs.begin(), diffs.end(), 0.0);
      cand.delta_prob = sum_d / n;
      double var_d = 0.0;
      for (int d : diffs) {
        var_d += (d - cand.delta_prob) * (d - cand.delta_prob);
      }
      var_d = (n > 1.0) ? (var_d / (n - 1.0)) : 0.0;
      cand.std_err_prob = std::sqrt(var_d / n);
      double z = (cand.std_err_prob > 0.0) ? (std::abs(cand.delta_prob) / cand.std_err_prob) : 0.0;
      cand.p_value_raw = std::erfc(z / std::sqrt(2.0));
      cand.ci_prob_lower = cand.delta_prob - 1.95996 * cand.std_err_prob;
      cand.ci_prob_upper = cand.delta_prob + 1.95996 * cand.std_err_prob;
    }
  }

  // Multiple testing correction across tested confirmation candidates
  std::vector<double> conf_raw_p;
  for (size_t idx : selected_confirmation_indices) {
    conf_raw_p.push_back(summary.candidates[idx].p_value_raw);
  }
  std::vector<double> conf_adj_p = ComputeHolmBonferroniAdjustedPValues(conf_raw_p);
  for (size_t i = 0; i < selected_confirmation_indices.size(); ++i) {
    size_t idx = selected_confirmation_indices[i];
    auto& cand = summary.candidates[idx];
    cand.p_value_adjusted = conf_adj_p[i];

    bool dir_match = (cand.predicted_direction * cand.delta_prob > 0.0);
    bool practical = (std::abs(cand.delta_prob) >= 0.05);
    bool sig = (cand.p_value_adjusted < 0.05);
    if (dir_match && practical && sig) {
      cand.confirmed_response = true;
      summary.candidates_confirmed_divergent++;
    } else {
      cand.confirmed_response = false;
    }
  }

  // Continuation selection: up to max_outcome_candidates
  std::vector<size_t> continuation_indices;
  for (size_t idx : selected_confirmation_indices) {
    if (summary.candidates[idx].confirmed_response) {
      continuation_indices.push_back(idx);
      if (static_cast<int>(continuation_indices.size()) >= max_outcome_candidates) break;
    }
  }
  if (static_cast<int>(continuation_indices.size()) < max_outcome_candidates) {
    for (size_t idx : selected_confirmation_indices) {
      if (std::find(continuation_indices.begin(), continuation_indices.end(), idx) == continuation_indices.end()) {
        continuation_indices.push_back(idx);
        if (static_cast<int>(continuation_indices.size()) >= max_outcome_candidates) break;
      }
    }
  }
  summary.candidates_tested_continuation = static_cast<int>(continuation_indices.size());

  // Run Full-Game Continuation on selected outcome candidates
  for (size_t idx : continuation_indices) {
    auto& cand = summary.candidates[idx];
    auto& root = summary.roots[idx];

    std::unique_ptr<State> base_state = game->NewInitialState();
    for (Action a : root.history_actions) {
      base_state->ApplyAction(a);
    }

    cand.continuation_samples = continuation_samples;
    std::vector<double> util_diffs;
    std::vector<int> vp_diffs;
    double util_sum_a = 0.0, util_sum_b = 0.0;
    double vp_sum_a = 0.0, vp_sum_b = 0.0;

    for (int m = 0; m < continuation_samples; ++m) {
      uint64_t roll_chance_seed = dune_seed::DeriveSeed(cand.game_seed, 0x00E3, cand.candidate_id, m, dune_seed::kStreamChance);
      std::array<std::mt19937_64, kNumPlayers> p_rngs_a, p_rngs_b;
      std::array<uint64_t, kNumPlayers> p_seeds;
      for (int p = 0; p < kNumPlayers; ++p) {
        p_seeds[p] = dune_seed::DeriveSeed(cand.game_seed, 0x00E3, cand.candidate_id, m, dune_seed::kStreamPolicyPlayer0 + p);
        p_rngs_a[p].seed(p_seeds[p]);
        p_rngs_b[p].seed(p_seeds[p]);
      }
      std::mt19937_64 c_rng_a(roll_chance_seed);
      std::mt19937_64 c_rng_b(roll_chance_seed);

      PairedContinuationRecord cont_rec;
      cont_rec.candidate_id = cand.candidate_id;
      cont_rec.pair_index = m;
      cont_rec.chance_seed = roll_chance_seed;
      cont_rec.policy_seeds = p_seeds;
      cont_rec.holder_player = cand.holder_player;
      cont_rec.target_opponent = cand.opponent_player;

      // Rollout Branch A
      auto sa = base_state->Clone();
      sa->ApplyAction(cand.intrigue_action);
      AdvanceThroughChanceAndAcks(sa.get(), c_rng_a);
      if (ContainsAction(sa->LegalActions(), kActionEndTurn)) {
        sa->ApplyAction(kActionEndTurn);
      }
      int steps_a = 0;
      while (!sa->IsTerminal() && steps_a < 3000) {
        steps_a++;
        AdvanceStatus st = AdvanceThroughChanceAndAcks(sa.get(), c_rng_a);
        if (st == AdvanceStatus::kSuccessTerminal) break;
        if (st != AdvanceStatus::kSuccessDecision) break;
        Player cp = sa->CurrentPlayer();
        Action a = evaluator->SelectAction(*sa, cp, /*greedy=*/false, temperature, p_rngs_a[cp]);
        sa->ApplyAction(a);
      }

      // Rollout Branch B
      auto sb = base_state->Clone();
      sb->ApplyAction(kActionEndTurn);
      int steps_b = 0;
      bool played_later = false;
      int play_round = -1, play_step = -1;
      while (!sb->IsTerminal() && steps_b < 3000) {
        steps_b++;
        AdvanceStatus st = AdvanceThroughChanceAndAcks(sb.get(), c_rng_b);
        if (st == AdvanceStatus::kSuccessTerminal) break;
        if (st != AdvanceStatus::kSuccessDecision) break;
        Player cp = sb->CurrentPlayer();
        Action a = evaluator->SelectAction(*sb, cp, /*greedy=*/false, temperature, p_rngs_b[cp]);
        if (cp == cand.holder_player && a == cand.intrigue_action) {
          played_later = true;
          play_round = dynamic_cast<DuneImperiumState*>(sb.get())->GetCurrentRound();
          play_step = steps_b;
        }
        sb->ApplyAction(a);
      }

      cont_rec.branch_a_terminal = sa->IsTerminal();
      cont_rec.branch_a_status = sa->IsTerminal() ? "TERMINAL" : "TIMEOUT";
      cont_rec.branch_b_terminal = sb->IsTerminal();
      cont_rec.branch_b_status = sb->IsTerminal() ? "TERMINAL" : "TIMEOUT";
      cont_rec.branch_b_intrigue_played_later = played_later;
      cont_rec.branch_b_intrigue_play_round = play_round;
      cont_rec.branch_b_intrigue_play_step = play_step;

      // EXPLICIT TERMINAL CHECK BEFORE READING RETURNS
      if (sa->IsTerminal() && sb->IsTerminal()) {
        cont_rec.is_valid = true;
        for (int p = 0; p < kNumPlayers; ++p) {
          cont_rec.branch_a_returns[p] = sa->Returns()[p];
          cont_rec.branch_b_returns[p] = sb->Returns()[p];
          cont_rec.branch_a_vps[p] = dynamic_cast<DuneImperiumState*>(sa.get())->GetPlayerVp(p);
          cont_rec.branch_b_vps[p] = dynamic_cast<DuneImperiumState*>(sb.get())->GetPlayerVp(p);
        }
        double u_a = cont_rec.branch_a_returns[cand.holder_player];
        double u_b = cont_rec.branch_b_returns[cand.holder_player];
        int vp_a = cont_rec.branch_a_vps[cand.holder_player];
        int vp_b = cont_rec.branch_b_vps[cand.holder_player];

        util_sum_a += u_a;
        util_sum_b += u_b;
        vp_sum_a += vp_a;
        vp_sum_b += vp_b;

        double u_diff = u_b - u_a;
        int vp_diff = vp_b - vp_a;
        cont_rec.holder_utility_diff = u_diff;
        cont_rec.holder_vp_diff = vp_diff;
        util_diffs.push_back(u_diff);
        vp_diffs.push_back(vp_diff);
      } else {
        cont_rec.is_valid = false;
        cand.has_invalidation_failure = true;
        cand.invalidation_reason = "Continuation timed out before terminal state (A=" +
            cont_rec.branch_a_status + ", B=" + cont_rec.branch_b_status + ")";
        summary.run_has_invalidation_failure = true;
        summary.invalidation_reason = cand.invalidation_reason;
      }
      summary.continuation_records.push_back(cont_rec);
    }

    if (!util_diffs.empty()) {
      double n = static_cast<double>(util_diffs.size());
      cand.mean_early_holder_utility = util_sum_a / n;
      cand.mean_hold_holder_utility = util_sum_b / n;
      cand.delta_holder_utility = cand.mean_hold_holder_utility - cand.mean_early_holder_utility;

      cand.mean_early_holder_vp = vp_sum_a / n;
      cand.mean_hold_holder_vp = vp_sum_b / n;
      cand.delta_holder_vp = cand.mean_hold_holder_vp - cand.mean_early_holder_vp;

      double var_u = 0.0;
      for (double d : util_diffs) {
        var_u += (d - cand.delta_holder_utility) * (d - cand.delta_holder_utility);
      }
      var_u = (n > 1.0) ? (var_u / (n - 1.0)) : 0.0;
      cand.std_err_utility = std::sqrt(var_u / n);
      double t = (cand.std_err_utility > 0.0) ? (std::abs(cand.delta_holder_utility) / cand.std_err_utility) : 0.0;
      cand.p_value_utility_raw = std::erfc(t / std::sqrt(2.0));
    }
  }

  // Multiple testing correction across tested outcome candidates
  std::vector<double> out_raw_p;
  for (size_t idx : continuation_indices) {
    out_raw_p.push_back(summary.candidates[idx].p_value_utility_raw);
  }
  std::vector<double> out_adj_p = ComputeHolmBonferroniAdjustedPValues(out_raw_p);
  for (size_t i = 0; i < continuation_indices.size(); ++i) {
    size_t idx = continuation_indices[i];
    auto& cand = summary.candidates[idx];
    cand.p_value_utility_adjusted = out_adj_p[i];

    // Conservative critical value for Holm-Bonferroni adjusted CI (M <= 4 -> alpha = 0.05 / 4 -> z ~ 2.50)
    double z_crit = 2.4977;
    cand.utility_ci_lower_adjusted = cand.delta_holder_utility - z_crit * cand.std_err_utility;
    cand.utility_ci_upper_adjusted = cand.delta_holder_utility + z_crit * cand.std_err_utility;

    // Decision rule: A positive average alone is strictly reported as Inconclusive
    if (cand.has_invalidation_failure) {
      cand.verdict = "Invalid";
    } else if (cand.delta_holder_utility > 0.0 && cand.p_value_utility_adjusted < 0.05 && cand.utility_ci_lower_adjusted > 0.0) {
      cand.verdict = "Supported";
    } else if (cand.delta_holder_utility < 0.0 && cand.p_value_utility_adjusted < 0.05 && cand.utility_ci_upper_adjusted < 0.0) {
      cand.verdict = "Contradicted";
    } else {
      cand.verdict = "Inconclusive";
    }
  }

  return summary;
}

inline IntrigueMiningSummary RunIntrigueMiningSuite(
    IPolicyDiagnosticsEvaluator* evaluator,
    int num_games = 100,
    uint64_t base_seed = 20260915,
    int confirmation_samples = 64,
    float temperature = 1.0f) {
  return RunOrdinaryGameIntrigueMining(
      evaluator, num_games, base_seed, /*discovery_pairs=*/4,
      /*max_candidates=*/8, confirmation_samples,
      /*max_outcome_candidates=*/4, /*continuation_samples=*/64,
      temperature);
}

// ===========================================================================
// Windfall Controller Evaluation Suite & Bank Selection
// ===========================================================================

inline bool ParseMiningRootJsonLine(const std::string& line, WindfallBankRecord* rec) {
  if (line.find("\"intrigue_name\": \"Windfall\"") == std::string::npos &&
      line.find("\"intrigue_name\":\"Windfall\"") == std::string::npos) {
    return false;
  }
  auto extract_int = [&](const std::string& key) -> int {
    std::string needle = "\"" + key + "\": ";
    size_t pos = line.find(needle);
    if (pos == std::string::npos) {
      needle = "\"" + key + "\":";
      pos = line.find(needle);
      if (pos == std::string::npos) return 0;
    }
    pos += needle.size();
    return std::stoi(line.substr(pos));
  };
  auto extract_uint64 = [&](const std::string& key) -> uint64_t {
    std::string needle = "\"" + key + "\": ";
    size_t pos = line.find(needle);
    if (pos == std::string::npos) {
      needle = "\"" + key + "\":";
      pos = line.find(needle);
      if (pos == std::string::npos) return 0ULL;
    }
    pos += needle.size();
    return std::stoull(line.substr(pos));
  };
  auto extract_str = [&](const std::string& key) -> std::string {
    std::string needle = "\"" + key + "\": \"";
    size_t pos = line.find(needle);
    if (pos == std::string::npos) {
      needle = "\"" + key + "\":\"";
      pos = line.find(needle);
      if (pos == std::string::npos) return "";
    }
    pos += needle.size();
    size_t end_pos = line.find("\"", pos);
    if (end_pos == std::string::npos) return "";
    return line.substr(pos, end_pos - pos);
  };

  rec->candidate_id = extract_int("candidate_id");
  rec->game_id = extract_int("game_id");
  rec->game_seed = extract_uint64("game_seed");
  rec->round = extract_int("round");
  rec->step_index = extract_int("step_index");
  rec->holder_player = extract_int("holder_player");
  rec->holder_leader = extract_str("holder_leader");
  rec->holder_solari = extract_int("holder_solari");
  rec->holder_water = extract_int("holder_water");
  rec->holder_spice = extract_int("holder_spice");
  rec->target_opponent = extract_int("target_opponent");
  rec->opponent_leader = extract_str("opponent_leader");
  rec->opponent_solari = extract_int("opponent_solari");
  rec->intrigue_action = extract_int("intrigue_action");
  rec->intrigue_name = extract_str("intrigue_name");
  return true;
}

inline std::vector<WindfallBankRecord> ReconstructAndVerifyWindfallRoots(
    const std::string& previous_run_dir,
    const std::shared_ptr<const Game>& game,
    IPolicyDiagnosticsEvaluator* evaluator,
    uint64_t base_seed = 20260915,
    float temperature = 1.0f) {

  std::string roots_file = previous_run_dir + "/mining_roots.jsonl";
  std::ifstream infile(roots_file);
  if (!infile) {
    SpielFatalError("Cannot open roots file: " + roots_file);
  }

  std::vector<WindfallBankRecord> bank;
  std::string line;
  while (std::getline(infile, line)) {
    if (line.empty()) continue;
    WindfallBankRecord rec;
    if (ParseMiningRootJsonLine(line, &rec)) {
      bank.push_back(rec);
    }
  }

  std::cout << "Loaded " << bank.size() << " Windfall root candidates from " << roots_file << "\n";
  std::cout << "Reconstructing full history actions and verifying bitwise replay...\n";

  for (size_t i = 0; i < bank.size(); ++i) {
    auto& rec = bank[i];
    int g = rec.game_id;
    uint64_t game_chance_seed = dune_seed::DeriveSeed(base_seed, 0x00E0, g, 0, dune_seed::kStreamChance);
    std::array<uint64_t, kNumPlayers> game_policy_seeds;
    std::array<std::mt19937_64, kNumPlayers> game_policy_rngs;
    for (int p = 0; p < kNumPlayers; ++p) {
      game_policy_seeds[p] = dune_seed::DeriveSeed(base_seed, 0x00E0, g, 0, dune_seed::kStreamPolicyPlayer0 + p);
      game_policy_rngs[p].seed(game_policy_seeds[p]);
    }
    std::mt19937_64 game_chance_rng(game_chance_seed);

    std::unique_ptr<State> state = game->NewInitialState();
    auto* s = dynamic_cast<DuneImperiumState*>(state.get());
    std::vector<Action> history;
    int steps = 0;
    bool found_step = false;

    while (!s->IsTerminal() && steps < 3000) {
      steps++;
      AdvanceStatus st = AdvanceThroughChanceAndAcks(s, game_chance_rng, &history);
      if (st != AdvanceStatus::kSuccessDecision) break;

      Player cp = s->CurrentPlayer();
      auto legal = s->LegalActions();
      if (legal.empty()) break;

      if (steps == rec.step_index) {
        found_step = true;
        rec.history_actions = history;
        rec.base_snapshot = evaluator->CaptureSnapshot(*s, rec.holder_player);

        if (cp != rec.holder_player) {
          rec.replay_verified = false;
          rec.replay_failure_reason = "Player mismatch at step: expected " + std::to_string(rec.holder_player) +
                                      ", got " + std::to_string(cp);
          break;
        }
        if (s->GetCurrentRound() != rec.round) {
          rec.replay_verified = false;
          rec.replay_failure_reason = "Round mismatch: expected " + std::to_string(rec.round) +
                                      ", got " + std::to_string(s->GetCurrentRound());
          break;
        }
        if (s->GetPlayerSolari(rec.holder_player) != rec.holder_solari) {
          rec.replay_verified = false;
          rec.replay_failure_reason = "Solari mismatch: expected " + std::to_string(rec.holder_solari) +
                                      ", got " + std::to_string(s->GetPlayerSolari(rec.holder_player));
          break;
        }
        if (!ContainsAction(legal, kActionPlayWindfall)) {
          rec.replay_verified = false;
          rec.replay_failure_reason = "kActionPlayWindfall not legal at step";
          break;
        }

        auto rep_res = VerifyReplayHistory(game, rec.history_actions, rec.holder_player, rec.base_snapshot, evaluator);
        rec.replay_verified = rep_res.matches;
        rec.replay_failure_reason = rep_res.failure_reason;
        break;
      }

      Action chosen = evaluator->SelectAction(*s, cp, false, temperature, game_policy_rngs[cp]);
      s->ApplyAction(chosen);
      history.push_back(chosen);
    }

    if (!found_step && rec.replay_failure_reason.empty()) {
      rec.replay_verified = false;
      rec.replay_failure_reason = "Step index " + std::to_string(rec.step_index) + " not reached in game " + std::to_string(g);
    }

    std::cout << "  Root [" << (i + 1) << "/" << bank.size() << "] Cand " << rec.candidate_id
              << " Game " << rec.game_id << " R" << rec.round << " Step " << rec.step_index
              << " P" << rec.holder_player << " (" << rec.holder_leader << "): "
              << (rec.replay_verified ? "REPLAY VERIFIED" : ("FAILED: " + rec.replay_failure_reason))
              << "\n";
  }

  return bank;
}

inline std::vector<WindfallBankRecord> SelectEvaluationBank(
    std::vector<WindfallBankRecord>& all_roots,
    int max_positions = 16) {

  std::map<int, std::vector<size_t>> round_to_indices;
  for (size_t i = 0; i < all_roots.size(); ++i) {
    if (all_roots[i].replay_verified) {
      round_to_indices[all_roots[i].round].push_back(i);
    } else {
      all_roots[i].is_selected = false;
      all_roots[i].selection_reason = "Excluded: Replay verification failed (" + all_roots[i].replay_failure_reason + ")";
    }
  }

  for (auto& [rnd, idxs] : round_to_indices) {
    std::sort(idxs.begin(), idxs.end(), [&](size_t a, size_t b) {
      return all_roots[a].game_id < all_roots[b].game_id;
    });
  }

  std::vector<size_t> selected_indices;
  std::set<int> selected_game_ids;

  // Pass 1: pick 1 candidate from every available round (rounds 2 through 10)
  for (auto& [rnd, idxs] : round_to_indices) {
    if (static_cast<int>(selected_indices.size()) >= max_positions) break;
    for (size_t idx : idxs) {
      int gid = all_roots[idx].game_id;
      if (selected_game_ids.find(gid) == selected_game_ids.end()) {
        selected_indices.push_back(idx);
        selected_game_ids.insert(gid);
        all_roots[idx].is_selected = true;
        all_roots[idx].selection_reason = "Selected in round-robin pass 1 for Round " + std::to_string(rnd);
        break;
      }
    }
  }

  // Pass 2: fill remaining slots up to max_positions from rounds with additional candidates
  while (static_cast<int>(selected_indices.size()) < max_positions) {
    bool added_any = false;
    for (auto& [rnd, idxs] : round_to_indices) {
      if (static_cast<int>(selected_indices.size()) >= max_positions) break;
      for (size_t idx : idxs) {
        if (!all_roots[idx].is_selected) {
          int gid = all_roots[idx].game_id;
          if (selected_game_ids.find(gid) == selected_game_ids.end()) {
            selected_indices.push_back(idx);
            selected_game_ids.insert(gid);
            all_roots[idx].is_selected = true;
            all_roots[idx].selection_reason = "Selected in round-robin pass 2 for Round " + std::to_string(rnd);
            added_any = true;
            break;
          }
        }
      }
    }
    if (!added_any) break;
  }

  // Mark exclusions for any remaining eligible candidates
  for (size_t i = 0; i < all_roots.size(); ++i) {
    if (all_roots[i].replay_verified && !all_roots[i].is_selected) {
      all_roots[i].selection_reason = "Excluded: Exceeded 16-position quota to balance round representation (Round " +
                                      std::to_string(all_roots[i].round) + ")";
    }
  }

  std::vector<WindfallBankRecord> selected_bank;
  for (size_t idx : selected_indices) {
    selected_bank.push_back(all_roots[idx]);
  }

  std::sort(selected_bank.begin(), selected_bank.end(), [](const WindfallBankRecord& a, const WindfallBankRecord& b) {
    if (a.round != b.round) return a.round < b.round;
    return a.game_id < b.game_id;
  });

  return selected_bank;
}

inline WindfallEvaluationSummary RunWindfallEvaluationSuite(
    const std::shared_ptr<const Game>& game,
    IPolicyDiagnosticsEvaluator* candidate_evaluator,
    IPolicyDiagnosticsEvaluator* opponent_evaluator,
    const std::vector<WindfallBankRecord>& all_roots,
    const std::vector<WindfallBankRecord>& selected_bank,
    const std::string& output_dir,
    int num_replicates = 64,
    uint64_t base_seed = 20260915,
    float temperature = 1.0f) {

  WindfallEvaluationSummary summary;
  summary.candidate_model = candidate_evaluator->ModelName();
  summary.candidate_sha256 = candidate_evaluator->ModelSha256();
  summary.opponent_model = opponent_evaluator->ModelName();
  summary.opponent_sha256 = opponent_evaluator->ModelSha256();
  summary.base_seed = base_seed;
  summary.temperature = temperature;
  summary.total_natural_roots_reconstructed = static_cast<int>(all_roots.size());
  int verified_cnt = 0;
  for (const auto& r : all_roots) {
    if (r.replay_verified) verified_cnt++;
  }
  summary.total_natural_roots_verified = verified_cnt;
  summary.positions_selected = static_cast<int>(selected_bank.size());
  summary.continuations_per_position = num_replicates;
  summary.total_continuation_pairs = summary.positions_selected * num_replicates;
  summary.bank = all_roots;

  WindfallTimingControlledEvaluator controlled_eval(candidate_evaluator, 0);

  std::vector<double> position_mean_util_diffs;
  std::vector<double> position_mean_vp_diffs;
  std::vector<double> all_util_diffs;
  std::vector<int> all_vp_diffs;

  std::cout << "\n=================================================================\n";
  std::cout << "=== RUNNING FROZEN-POLICY WINDFALL TIMING EVALUATION          ===\n";
  std::cout << "=================================================================\n";
  std::cout << "Positions:      " << selected_bank.size() << "\n";
  std::cout << "Replicates:     " << num_replicates << " per position ("
            << (selected_bank.size() * num_replicates) << " paired, "
            << (selected_bank.size() * num_replicates * 2) << " total games)\n";
  std::cout << "Arm A:          Unchanged " << candidate_evaluator->ModelName() << "\n";
  std::cout << "Arm B:          " << candidate_evaluator->ModelName() << " + WindfallTimingController\n";
  std::cout << "Opponents:      3x " << opponent_evaluator->ModelName() << "\n";
  std::cout << "Seed Domain:    0x00E8\n";
  std::cout << "=================================================================\n\n";

  for (size_t pos_idx = 0; pos_idx < selected_bank.size(); ++pos_idx) {
    const auto& root = selected_bank[pos_idx];
    int holder = root.holder_player;
    int opp_target = root.target_opponent;
    controlled_eval.SetHolder(holder);

    std::unique_ptr<State> base_state = game->NewInitialState();
    for (Action a : root.history_actions) {
      base_state->ApplyAction(a);
    }

    double pos_util_sum = 0.0;
    double pos_vp_sum = 0.0;
    int valid_replicates = 0;

    std::cout << "Evaluating Position [" << (pos_idx + 1) << "/" << selected_bank.size() << "] "
              << "Cand " << root.candidate_id << " (Game " << root.game_id << ", R" << root.round
              << ", Step " << root.step_index << ", P" << holder << " " << root.holder_leader
              << ", Solari=" << root.holder_solari << ")..." << std::flush;

    for (int m = 0; m < num_replicates; ++m) {
      uint64_t roll_chance_seed = dune_seed::DeriveSeed(base_seed, 0x00E8, root.candidate_id, m, dune_seed::kStreamChance);
      std::array<uint64_t, kNumPlayers> p_seeds;
      std::array<std::mt19937_64, kNumPlayers> p_rngs_a, p_rngs_b;
      for (int p = 0; p < kNumPlayers; ++p) {
        p_seeds[p] = dune_seed::DeriveSeed(base_seed, 0x00E8, root.candidate_id, m, dune_seed::kStreamPolicyPlayer0 + p);
        p_rngs_a[p].seed(p_seeds[p]);
        p_rngs_b[p].seed(p_seeds[p]);
      }
      std::mt19937_64 c_rng_a(roll_chance_seed);
      std::mt19937_64 c_rng_b(roll_chance_seed);

      PairedContinuationRecord cont_rec;
      cont_rec.candidate_id = root.candidate_id;
      cont_rec.pair_index = m;
      cont_rec.chance_seed = roll_chance_seed;
      cont_rec.policy_seeds = p_seeds;
      cont_rec.holder_player = holder;
      cont_rec.target_opponent = opp_target;

      // --- Rollout Arm A (Baseline) ---
      auto sa = base_state->Clone();
      int steps_a = 0;
      bool windfall_played_a = false;
      int play_round_a = -1, play_step_a = -1;

      while (!sa->IsTerminal() && steps_a < 3000) {
        steps_a++;
        AdvanceStatus st = AdvanceThroughChanceAndAcks(sa.get(), c_rng_a);
        if (st == AdvanceStatus::kSuccessTerminal) break;
        if (st != AdvanceStatus::kSuccessDecision) break;
        Player cp = sa->CurrentPlayer();
        Action a = open_spiel::kInvalidAction;
        if (cp == holder) {
          a = candidate_evaluator->SelectAction(*sa, cp, false, temperature, p_rngs_a[cp]);
        } else {
          a = opponent_evaluator->SelectAction(*sa, cp, false, temperature, p_rngs_a[cp]);
        }
        if (cp == holder && a == kActionPlayWindfall && !windfall_played_a) {
          windfall_played_a = true;
          play_round_a = dynamic_cast<DuneImperiumState*>(sa.get())->GetCurrentRound();
          play_step_a = steps_a;
        }
        sa->ApplyAction(a);
      }

      // --- Rollout Arm B (Controlled) ---
      auto sb = base_state->Clone();
      int steps_b = 0;
      bool windfall_played_b = false;
      int play_round_b = -1, play_step_b = -1;
      controlled_eval.SetContext(root.candidate_id, m);
      controlled_eval.ResetTurnStep();

      while (!sb->IsTerminal() && steps_b < 3000) {
        steps_b++;
        AdvanceStatus st = AdvanceThroughChanceAndAcks(sb.get(), c_rng_b);
        if (st == AdvanceStatus::kSuccessTerminal) break;
        if (st != AdvanceStatus::kSuccessDecision) break;
        Player cp = sb->CurrentPlayer();
        Action a = open_spiel::kInvalidAction;
        if (cp == holder) {
          a = controlled_eval.SelectAction(*sb, cp, false, temperature, p_rngs_b[cp]);
        } else {
          a = opponent_evaluator->SelectAction(*sb, cp, false, temperature, p_rngs_b[cp]);
        }
        if (cp == holder && a == kActionPlayWindfall && !windfall_played_b) {
          windfall_played_b = true;
          play_round_b = dynamic_cast<DuneImperiumState*>(sb.get())->GetCurrentRound();
          play_step_b = steps_b;
        }
        sb->ApplyAction(a);
      }

      cont_rec.branch_a_terminal = sa->IsTerminal();
      cont_rec.branch_a_status = sa->IsTerminal() ? "TERMINAL" : "TIMEOUT";
      cont_rec.branch_a_windfall_played = windfall_played_a;
      cont_rec.branch_a_windfall_play_round = play_round_a;
      cont_rec.branch_a_windfall_play_step = play_step_a;

      cont_rec.branch_b_terminal = sb->IsTerminal();
      cont_rec.branch_b_status = sb->IsTerminal() ? "TERMINAL" : "TIMEOUT";
      cont_rec.branch_b_windfall_played = windfall_played_b;
      cont_rec.branch_b_windfall_play_round = play_round_b;
      cont_rec.branch_b_windfall_play_step = play_step_b;
      cont_rec.branch_b_intrigue_played_later = windfall_played_b;
      cont_rec.branch_b_intrigue_play_round = play_round_b;
      cont_rec.branch_b_intrigue_play_step = play_step_b;

      // EXPLICIT NATURAL TERMINAL CHECK BEFORE READING RETURNS
      if (sa->IsTerminal() && sb->IsTerminal()) {
        cont_rec.is_valid = true;
        for (int p = 0; p < kNumPlayers; ++p) {
          cont_rec.branch_a_returns[p] = sa->Returns()[p];
          cont_rec.branch_b_returns[p] = sb->Returns()[p];
          cont_rec.branch_a_vps[p] = dynamic_cast<DuneImperiumState*>(sa.get())->GetPlayerVp(p);
          cont_rec.branch_b_vps[p] = dynamic_cast<DuneImperiumState*>(sb.get())->GetPlayerVp(p);
        }
        double u_a = cont_rec.branch_a_returns[holder];
        double u_b = cont_rec.branch_b_returns[holder];
        int vp_a = cont_rec.branch_a_vps[holder];
        int vp_b = cont_rec.branch_b_vps[holder];

        double u_diff = u_b - u_a;
        int vp_diff = vp_b - vp_a;
        cont_rec.holder_utility_diff = u_diff;
        cont_rec.holder_vp_diff = vp_diff;

        pos_util_sum += u_diff;
        pos_vp_sum += vp_diff;
        all_util_diffs.push_back(u_diff);
        all_vp_diffs.push_back(vp_diff);
        valid_replicates++;
      } else {
        cont_rec.is_valid = false;
        summary.run_has_invalidation_failure = true;
        summary.invalidation_reason = "Continuation timed out before terminal state (A=" +
            cont_rec.branch_a_status + ", B=" + cont_rec.branch_b_status + ")";
      }

      summary.continuation_records.push_back(cont_rec);
    }

    if (valid_replicates > 0) {
      double mean_u = pos_util_sum / valid_replicates;
      double mean_v = pos_vp_sum / valid_replicates;
      position_mean_util_diffs.push_back(mean_u);
      position_mean_vp_diffs.push_back(mean_v);
      std::cout << " Delta Util=" << std::fixed << std::setprecision(4) << mean_u
                << ", Delta VP=" << std::setprecision(2) << mean_v << "\n";
    } else {
      std::cout << " FAILED (0 valid replicates)\n";
    }
  }

  summary.decisions = controlled_eval.DecisionRecords();
  summary.total_controller_decisions = static_cast<int>(summary.decisions.size());

  for (const auto& d : summary.decisions) {
    if (d.windfall_played) {
      summary.windfall_plays_executed++;
      if (d.plan_requires_windfall) summary.necessary_spending_unlocked++;
    } else {
      summary.early_plays_prevented++;
      if (d.action_taken == kActionEndTurn) summary.end_turn_retentions++;
      else summary.purchases_already_affordable++;
    }
    if (d.secrets_exception) summary.secrets_exceptions_triggered++;
    if (d.terminal_value_exception) summary.terminal_value_exceptions_triggered++;
    if (d.fallback_active) summary.fallbacks_triggered++;
  }

  size_t K = position_mean_util_diffs.size();
  if (K > 0) {
    double sum_u = 0.0, sum_vp = 0.0;
    for (size_t k = 0; k < K; ++k) {
      sum_u += position_mean_util_diffs[k];
      sum_vp += position_mean_vp_diffs[k];
    }
    summary.mean_placement_utility_diff = sum_u / K;
    summary.mean_holder_vp_diff = sum_vp / K;

    if (K > 1) {
      double var_u = 0.0;
      for (double d : position_mean_util_diffs) {
        var_u += (d - summary.mean_placement_utility_diff) * (d - summary.mean_placement_utility_diff);
      }
      var_u /= (K - 1);
      summary.placement_utility_std_err = std::sqrt(var_u / K);

      double t_crit = (K >= 16) ? 2.131 : 2.20;
      summary.placement_utility_ci_lower = summary.mean_placement_utility_diff - t_crit * summary.placement_utility_std_err;
      summary.placement_utility_ci_upper = summary.mean_placement_utility_diff + t_crit * summary.placement_utility_std_err;
    }
  }

  if (summary.run_has_invalidation_failure) {
    summary.verdict = "Invalid";
    summary.verdict_explanation = "Correctness or record-integrity failure occurred: " + summary.invalidation_reason;
  } else if (summary.placement_utility_ci_lower > 0.0) {
    summary.verdict = "Beneficial on the tested bank";
    summary.verdict_explanation = "Positive placement-utility effect with 95% uncertainty interval excluding zero, without correctness failures.";
  } else if (summary.placement_utility_ci_upper < 0.0) {
    summary.verdict = "Harmful on the tested bank";
    summary.verdict_explanation = "Supported negative placement-utility effect with 95% uncertainty interval strictly below zero.";
  } else {
    summary.verdict = "Inconclusive";
    summary.verdict_explanation = "Neither direction established: 95% confidence interval spans zero (uncertainty includes both positive and negative outcomes).";
  }

  if (!output_dir.empty()) {
    std::filesystem::create_directories(output_dir);

    // 1. Controller decisions JSONL
    std::ofstream dec_out(output_dir + "/controller_decisions.jsonl");
    for (const auto& d : summary.decisions) {
      dec_out << FormatWindfallDecisionRecordJson(d) << "\n";
    }

    // 2. Continuation pairs JSONL
    std::ofstream cont_out(output_dir + "/continuation_pairs.jsonl");
    for (const auto& c : summary.continuation_records) {
      cont_out << FormatPairedContinuationRecordJson(c) << "\n";
    }

    // 3. Frozen evaluation bank JSONL
    std::ofstream bank_out(output_dir + "/frozen_evaluation_bank.jsonl");
    for (const auto& b : summary.bank) {
      bank_out << FormatWindfallBankRecordJson(b) << "\n";
    }

    // 4. Summary JSON
    std::ofstream sum_out(output_dir + "/evaluation_summary.json");
    sum_out << FormatWindfallEvaluationReportJson(summary);

    // 5. Run manifest JSON
    std::ofstream man_out(output_dir + "/run_manifest.json");
    man_out << "{\n"
            << "  \"candidate_model\": \"" << EscapeJson(summary.candidate_model) << "\",\n"
            << "  \"candidate_sha256\": \"" << EscapeJson(summary.candidate_sha256) << "\",\n"
            << "  \"opponent_model\": \"" << EscapeJson(summary.opponent_model) << "\",\n"
            << "  \"opponent_sha256\": \"" << EscapeJson(summary.opponent_sha256) << "\",\n"
            << "  \"base_seed\": " << summary.base_seed << ",\n"
            << "  \"temperature\": " << summary.temperature << ",\n"
            << "  \"num_positions\": " << summary.positions_selected << ",\n"
            << "  \"num_replicates\": " << summary.continuations_per_position << ",\n"
            << "  \"seed_domain\": \"0x00E8\",\n"
            << "  \"verdict\": \"" << EscapeJson(summary.verdict) << "\"\n"
            << "}\n";

    // 6. Bank manifest JSON
    std::ofstream bman_out(output_dir + "/bank_manifest.json");
    bman_out << "{\n"
             << "  \"total_reconstructed\": " << summary.bank.size() << ",\n"
             << "  \"total_selected\": " << summary.positions_selected << ",\n"
             << "  \"selection_rule\": \"Outcome-independent round-robin across rounds 2-10, tie-broken by game_id ascending, max 1 position per source game\",\n"
             << "  \"positions\": [\n";
    for (size_t i = 0; i < summary.bank.size(); ++i) {
      const auto& b = summary.bank[i];
      bman_out << "    {\"candidate_id\": " << b.candidate_id
               << ", \"game_id\": " << b.game_id
               << ", \"round\": " << b.round
               << ", \"step_index\": " << b.step_index
               << ", \"holder_player\": " << b.holder_player
               << ", \"holder_leader\": \"" << EscapeJson(b.holder_leader) << "\""
               << ", \"holder_solari\": " << b.holder_solari
               << ", \"replay_verified\": " << (b.replay_verified ? "true" : "false")
               << ", \"is_selected\": " << (b.is_selected ? "true" : "false")
               << ", \"selection_reason\": \"" << EscapeJson(b.selection_reason) << "\"}";
      if (i + 1 < summary.bank.size()) bman_out << ",\n";
      else bman_out << "\n";
    }
    bman_out << "  ]\n}\n";

    std::cout << "\nWrote all evaluation artifacts to: " << output_dir << "\n";
  }

  return summary;
}

// ===========================================================================
// JSON Serialization Helpers
// ===========================================================================



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

#endif // OPEN_SPIEL_EXAMPLES_DUNE_FROZEN_POLICY_DIAGNOSTICS_H_
