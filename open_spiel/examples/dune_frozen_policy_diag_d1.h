#ifndef OPEN_SPIEL_EXAMPLES_DUNE_FROZEN_POLICY_DIAG_D1_H_
#define OPEN_SPIEL_EXAMPLES_DUNE_FROZEN_POLICY_DIAG_D1_H_

#include "dune_frozen_policy_diag_evaluators.h"

namespace open_spiel {
namespace dune_diagnostics {

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

} // namespace dune_diagnostics
} // namespace open_spiel

#endif // OPEN_SPIEL_EXAMPLES_DUNE_FROZEN_POLICY_DIAG_D1_H_
