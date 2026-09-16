// Copyright 2026
// Unit tests for Dune Imperium Frozen-Policy Diagnostics:
// Opponent response (Windfall reveal -> Swordmaster block), early-play preference,
// ordinary-game intrigue mining, and rigorous verification controls.

#include "dune_frozen_policy_diagnostics.h"

#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "open_spiel/spiel_utils.h"
#include "open_spiel/games/dune_imperium/dune_imperium.h"
#include "open_spiel/games/dune_imperium/dune_imperium_common.h"
#include "open_spiel/games/dune_imperium/dune_imperium_test_utils.h"

namespace open_spiel {
namespace dune_diagnostics {
namespace {

using dune_imperium::ActionToString;
using dune_imperium::ContainsAction;
using dune_imperium::DuneImperiumState;
using dune_imperium::kActionAgentSpaceSwordmaster;
using dune_imperium::kActionEndTurn;
using dune_imperium::kIntrigueSecretsOfTheSisterhood;
using dune_imperium::kIntrigueWindfall;
using dune_imperium::kIntrigueCalculatedHire;
using dune_imperium::kIntrigueAmbush;
using dune_imperium::FindImperiumCardById;

// 1. Validate that EARLY produces 8 visible solari and consumes Windfall,
//    while HOLD preserves 6 visible solari and Windfall.
void TestEarlyVsHoldStateDivergence() {
  std::cout << "Running TestEarlyVsHoldStateDivergence...\n";
  ScenarioConfig cfg;
  cfg.scenario_id = 0;
  cfg.family = ScenarioFamily::kCombatTension;
  cfg.round = 2;
  cfg.p_holder = 0;
  cfg.p_competitor = 1;
  cfg.p_opponent_a = 2;
  cfg.p_opponent_b = 3;

  auto base_state = ScenarioGenerator::BuildScenarioState(cfg, nullptr);
  SPIEL_CHECK_EQ(base_state->GetPlayerSolari(0), 6);
  SPIEL_CHECK_EQ(base_state->CurrentPlayer(), 0);

  std::mt19937_64 chance_rng_a(42);
  std::mt19937_64 chance_rng_b(42);

  // Branch A (EARLY)
  auto branch_a = base_state->Clone();
  auto* da = dynamic_cast<DuneImperiumState*>(branch_a.get());
  da->ApplyAction(kActionPlayWindfall);
  AdvanceThroughChanceAndAcks(da, chance_rng_a);
  SPIEL_CHECK_EQ(da->GetPlayerSolari(0), 8);
  auto intrigues_a = da->GetPlayerIntrigues(0);
  SPIEL_CHECK_TRUE(std::find(intrigues_a.begin(), intrigues_a.end(), kIntrigueWindfall) == intrigues_a.end());

  da->ApplyAction(kActionEndTurn);
  AdvanceThroughChanceAndAcks(da, chance_rng_a);
  SPIEL_CHECK_EQ(da->CurrentPlayer(), 1);
  SPIEL_CHECK_EQ(da->GetPlayerSolari(0), 8);

  // Branch B (HOLD)
  auto branch_b = base_state->Clone();
  auto* db = dynamic_cast<DuneImperiumState*>(branch_b.get());
  db->ApplyAction(kActionEndTurn);
  AdvanceThroughChanceAndAcks(db, chance_rng_b);
  SPIEL_CHECK_EQ(db->CurrentPlayer(), 1);
  SPIEL_CHECK_EQ(db->GetPlayerSolari(0), 6);
  auto intrigues_b = db->GetPlayerIntrigues(0);
  SPIEL_CHECK_TRUE(std::find(intrigues_b.begin(), intrigues_b.end(), kIntrigueWindfall) != intrigues_b.end());

  std::cout << "PASS: TestEarlyVsHoldStateDivergence\n";
}

// 2. Validate that cloning leaves the original state completely unchanged.
void TestCloningImmutability() {
  std::cout << "Running TestCloningImmutability...\n";
  ScenarioConfig cfg;
  cfg.scenario_id = 1;
  cfg.family = ScenarioFamily::kShippingEconomy;
  cfg.round = 2;

  auto base_state = ScenarioGenerator::BuildScenarioState(cfg, nullptr);
  std::string original_to_string = base_state->ToString();
  auto original_legal = base_state->LegalActions();
  std::vector<float> original_obs = base_state->ObservationTensor(0);

  std::mt19937_64 chance_rng_a(101);
  std::mt19937_64 chance_rng_b(102);

  // Clone and execute destructive actions on branches
  auto branch_a = base_state->Clone();
  branch_a->ApplyAction(kActionPlayWindfall);
  AdvanceThroughChanceAndAcks(branch_a.get(), chance_rng_a);
  branch_a->ApplyAction(kActionEndTurn);
  AdvanceThroughChanceAndAcks(branch_a.get(), chance_rng_a);

  auto branch_b = base_state->Clone();
  branch_b->ApplyAction(kActionEndTurn);
  AdvanceThroughChanceAndAcks(branch_b.get(), chance_rng_b);

  // Verify original state is byte-for-byte and functionally identical
  SPIEL_CHECK_EQ(base_state->ToString(), original_to_string);
  SPIEL_CHECK_EQ(base_state->LegalActions(), original_legal);
  SPIEL_CHECK_EQ(base_state->ObservationTensor(0), original_obs);
  SPIEL_CHECK_EQ(base_state->GetPlayerSolari(0), 6);
  SPIEL_CHECK_EQ(base_state->CurrentPlayer(), 0);

  std::cout << "PASS: TestCloningImmutability\n";
}

// 3. Validate that opponent inputs change after public reveal,
//    and unrevealed intrigue swapping leaves opponent inputs strictly unchanged.
void TestObservationTensorsAndInformationHiding() {
  std::cout << "Running TestObservationTensorsAndInformationHiding...\n";
  ScenarioConfig cfg;
  cfg.scenario_id = 2;
  cfg.family = ScenarioFamily::kFactionAlliance;
  cfg.round = 2;
  cfg.p_holder = 0;
  cfg.p_competitor = 1;

  auto base_state = ScenarioGenerator::BuildScenarioState(cfg, nullptr);
  std::mt19937_64 chance_rng_a(201);
  std::mt19937_64 chance_rng_b(202);

  // Advance Branch A (EARLY) and Branch B (HOLD) to P1's decision point
  auto branch_a = base_state->Clone();
  branch_a->ApplyAction(kActionPlayWindfall);
  AdvanceThroughChanceAndAcks(branch_a.get(), chance_rng_a);
  branch_a->ApplyAction(kActionEndTurn);
  AdvanceThroughChanceAndAcks(branch_a.get(), chance_rng_a);

  auto branch_b = base_state->Clone();
  branch_b->ApplyAction(kActionEndTurn);
  AdvanceThroughChanceAndAcks(branch_b.get(), chance_rng_b);

  // Requirement: Opponent P1's observation tensor changes after public reveal (8 vs 6 solari)
  std::vector<float> obs_p1_early = branch_a->ObservationTensor(1);
  std::vector<float> obs_p1_hold = branch_b->ObservationTensor(1);
  SPIEL_CHECK_FALSE(obs_p1_early == obs_p1_hold);

  // Requirement: Swapping one unrevealed intrigue for another leaves opponent P1's complete observation unchanged
  auto state1 = ScenarioGenerator::BuildScenarioState(cfg, nullptr);
  auto* s1 = dynamic_cast<DuneImperiumState*>(state1.get());
  s1->SetPlayerIntriguesForTesting(0, {kIntrigueWindfall}); // Windfall (59)

  auto state2 = ScenarioGenerator::BuildScenarioState(cfg, nullptr);
  auto* s2 = dynamic_cast<DuneImperiumState*>(state2.get());
  s2->SetPlayerIntriguesForTesting(0, {kIntrigueSecretsOfTheSisterhood}); // Secrets of the Sisterhood (44)

  std::vector<float> obs_p1_s1 = s1->ObservationTensor(1);
  std::vector<float> obs_p1_s2 = s2->ObservationTensor(1);
  SPIEL_CHECK_TRUE(obs_p1_s1 == obs_p1_s2); // Strict byte identity for opponent

  // Requirement: The holder's own observation identifies its own intrigue card
  std::vector<float> obs_p0_s1 = s1->ObservationTensor(0);
  std::vector<float> obs_p0_s2 = s2->ObservationTensor(0);
  SPIEL_CHECK_FALSE(obs_p0_s1 == obs_p0_s2); // Holder identifies its own card

  std::cout << "PASS: TestObservationTensorsAndInformationHiding\n";
}

// 4. Validate that P3 and P4 cannot take Swordmaster,
//    and Swordmaster placement by P1 really blocks P2.
void TestSwordmasterLegalityAndBlocking() {
  std::cout << "Running TestSwordmasterLegalityAndBlocking...\n";
  ScenarioConfig cfg;
  cfg.scenario_id = 3;
  cfg.family = ScenarioFamily::kMentatHighCouncil;
  cfg.round = 2;
  cfg.p_holder = 0;
  cfg.p_competitor = 1;
  cfg.p_opponent_a = 2;
  cfg.p_opponent_b = 3;

  auto base_state = ScenarioGenerator::BuildScenarioState(cfg, nullptr);
  auto* s = dynamic_cast<DuneImperiumState*>(base_state.get());
  std::mt19937_64 chance_rng(301);

  // Confirm P3 and P4 cannot take Swordmaster
  {
    std::unique_ptr<DuneImperiumState> test_c = 
        std::unique_ptr<DuneImperiumState>(dynamic_cast<DuneImperiumState*>(s->Clone().release()));
    test_c->SetCurrentPlayerForTesting(2);
    test_c->SetPhaseForTesting(GamePhase::kAgentTurns);
    test_c->SetCurrentTurnMainActionDoneForTesting(2, false);
    SPIEL_CHECK_FALSE(dune_imperium::SpaceIsReachable(test_c.get(), kActionAgentSpaceSwordmaster));
    test_c->SetCurrentPlayerForTesting(3);
    test_c->SetCurrentTurnMainActionDoneForTesting(3, false);
    SPIEL_CHECK_FALSE(dune_imperium::SpaceIsReachable(test_c.get(), kActionAgentSpaceSwordmaster));
  }

  // P2 holds Windfall and ends turn
  s->ApplyAction(kActionEndTurn);
  AdvanceThroughChanceAndAcks(s, chance_rng);
  SPIEL_CHECK_EQ(s->CurrentPlayer(), 1);

  // P1 takes Swordmaster
  SPIEL_CHECK_TRUE(dune_imperium::SpaceIsReachable(s, kActionAgentSpaceSwordmaster));
  dune_imperium::SelectCardAndApplySpace(s, kActionAgentSpaceSwordmaster);
  SPIEL_CHECK_TRUE(s->HasSwordmaster(1));

  // End P1 turn
  if (ContainsAction(s->LegalActions(), kActionEndTurn)) {
    s->ApplyAction(kActionEndTurn);
    AdvanceThroughChanceAndAcks(s, chance_rng);
  }

  // Advance until P2 gets its next agent placement turn
  while (s->CurrentPlayer() != 0 && !s->IsTerminal()) {
    auto legal = s->LegalActions();
    if (legal.empty()) break;
    s->ApplyAction(dune_imperium::DefaultProgressionAction(legal));
    AdvanceThroughChanceAndAcks(s, chance_rng);
  }

  // Verify that Swordmaster is now blocked for P2
  SPIEL_CHECK_EQ(s->CurrentPlayer(), 0);
  SPIEL_CHECK_FALSE(dune_imperium::SpaceIsReachable(s, kActionAgentSpaceSwordmaster));

  std::cout << "PASS: TestSwordmasterLegalityAndBlocking\n";
}

// 5. Control Test 1: Production Chance Helper
void TestProductionChanceHelper() {
  std::cout << "Running TestProductionChanceHelper...\n";
  auto game = open_spiel::LoadGame("dune_imperium");
  std::unique_ptr<State> state = game->NewInitialState();
  SPIEL_CHECK_TRUE(state->IsChanceNode());

  std::mt19937_64 rng1(20260915);
  std::vector<Action> history;
  AdvanceStatus st = AdvanceThroughChanceAndAcks(state.get(), rng1, &history);
  SPIEL_CHECK_EQ(st, AdvanceStatus::kSuccessDecision);
  SPIEL_CHECK_FALSE(state->IsChanceNode());
  SPIEL_CHECK_GE(state->CurrentPlayer(), 0);
  SPIEL_CHECK_LT(state->CurrentPlayer(), 4);
  SPIEL_CHECK_GT(history.size(), 0);

  std::cout << "PASS: TestProductionChanceHelper\n";
}

// 6. Control Test 2: Identical Branch Replay
void TestIdenticalBranchReplay() {
  std::cout << "Running TestIdenticalBranchReplay...\n";
  ScenarioConfig cfg;
  cfg.scenario_id = 0;
  cfg.family = ScenarioFamily::kCombatTension;
  cfg.round = 2;
  auto base_state = ScenarioGenerator::BuildScenarioState(cfg, nullptr);

  ScriptedNonreactiveControlEvaluator eval(0, 1);
  std::mt19937_64 rng_a(9999);
  std::mt19937_64 rng_b(9999);
  std::array<std::mt19937_64, 4> p_rngs_a, p_rngs_b;
  for (int p = 0; p < 4; ++p) {
    p_rngs_a[p].seed(1000 + p);
    p_rngs_b[p].seed(1000 + p);
  }

  auto branch_a = base_state->Clone();
  auto* da = dynamic_cast<DuneImperiumState*>(branch_a.get());
  da->ApplyAction(kActionEndTurn);
  Action act_a = open_spiel::kInvalidAction;
  bool ok_a = AdvanceToOpponentMainPlacement(
      da, 1, &eval, rng_a, p_rngs_a, false, 1.0f, &act_a);

  auto branch_b = base_state->Clone();
  auto* db = dynamic_cast<DuneImperiumState*>(branch_b.get());
  db->ApplyAction(kActionEndTurn);
  Action act_b = open_spiel::kInvalidAction;
  bool ok_b = AdvanceToOpponentMainPlacement(
      db, 1, &eval, rng_b, p_rngs_b, false, 1.0f, &act_b);

  SPIEL_CHECK_TRUE(ok_a);
  SPIEL_CHECK_TRUE(ok_b);
  SPIEL_CHECK_EQ(act_a, act_b);
  SPIEL_CHECK_EQ(da->ToString(), db->ToString());
  std::cout << "PASS: TestIdenticalBranchReplay\n";
}

// 7. Control Test 3: Scripted Positive Control
void TestScriptedPositiveControl() {
  std::cout << "Running TestScriptedPositiveControl...\n";
  ScenarioConfig cfg;
  cfg.scenario_id = 0;
  cfg.family = ScenarioFamily::kCombatTension;
  cfg.round = 2;
  auto base_state = ScenarioGenerator::BuildScenarioState(cfg, nullptr);

  ScriptedPositiveControlEvaluator eval(/*p_holder=*/0, /*p_competitor=*/1);
  std::mt19937_64 c_rng_a(42), c_rng_b(42);
  std::array<std::mt19937_64, 4> p_rngs_a, p_rngs_b;
  for (int p = 0; p < 4; ++p) {
    p_rngs_a[p].seed(100 + p);
    p_rngs_b[p].seed(100 + p);
  }

  // Branch A: EARLY reveal -> P0 has 8 solari -> P1 takes Swordmaster
  auto branch_a = base_state->Clone();
  auto* da = dynamic_cast<DuneImperiumState*>(branch_a.get());
  da->ApplyAction(kActionPlayWindfall);
  AdvanceThroughChanceAndAcks(da, c_rng_a);
  da->ApplyAction(kActionEndTurn);
  Action act_a = open_spiel::kInvalidAction;
  bool ok_a = AdvanceToOpponentMainPlacement(
      da, 1, &eval, c_rng_a, p_rngs_a, false, 1.0f, &act_a);
  SPIEL_CHECK_TRUE(ok_a);
  SPIEL_CHECK_EQ(act_a, kActionAgentSpaceSwordmaster);

  // Branch B: HOLD -> P0 has 6 solari -> P1 does NOT take Swordmaster
  auto branch_b = base_state->Clone();
  auto* db = dynamic_cast<DuneImperiumState*>(branch_b.get());
  db->ApplyAction(kActionEndTurn);
  Action act_b = open_spiel::kInvalidAction;
  bool ok_b = AdvanceToOpponentMainPlacement(
      db, 1, &eval, c_rng_b, p_rngs_b, false, 1.0f, &act_b);
  SPIEL_CHECK_TRUE(ok_b);
  SPIEL_CHECK_NE(act_b, kActionAgentSpaceSwordmaster);

  std::cout << "PASS: TestScriptedPositiveControl\n";
}

// 8. Control Test 4: Scripted Nonreactive Control
void TestScriptedNonreactiveControl() {
  std::cout << "Running TestScriptedNonreactiveControl...\n";
  ScenarioConfig cfg;
  cfg.scenario_id = 0;
  cfg.family = ScenarioFamily::kCombatTension;
  cfg.round = 2;
  auto base_state = ScenarioGenerator::BuildScenarioState(cfg, nullptr);

  ScriptedNonreactiveControlEvaluator eval(/*p_holder=*/0, /*p_competitor=*/1);
  std::mt19937_64 c_rng_a(123), c_rng_b(123);
  std::array<std::mt19937_64, 4> p_rngs_a, p_rngs_b;
  for (int p = 0; p < 4; ++p) {
    p_rngs_a[p].seed(200 + p);
    p_rngs_b[p].seed(200 + p);
  }

  // Branch A: EARLY
  auto branch_a = base_state->Clone();
  auto* da = dynamic_cast<DuneImperiumState*>(branch_a.get());
  da->ApplyAction(kActionPlayWindfall);
  AdvanceThroughChanceAndAcks(da, c_rng_a);
  da->ApplyAction(kActionEndTurn);
  Action act_a = open_spiel::kInvalidAction;
  bool ok_a = AdvanceToOpponentMainPlacement(da, 1, &eval, c_rng_a, p_rngs_a, false, 1.0f, &act_a);
  SPIEL_CHECK_TRUE(ok_a);

  // Branch B: HOLD
  auto branch_b = base_state->Clone();
  auto* db = dynamic_cast<DuneImperiumState*>(branch_b.get());
  db->ApplyAction(kActionEndTurn);
  Action act_b = open_spiel::kInvalidAction;
  bool ok_b = AdvanceToOpponentMainPlacement(db, 1, &eval, c_rng_b, p_rngs_b, false, 1.0f, &act_b);
  SPIEL_CHECK_TRUE(ok_b);

  // Nonreactive evaluator is indifferent to holder's solari
  SPIEL_CHECK_EQ(act_a, act_b);
  std::cout << "PASS: TestScriptedNonreactiveControl\n";
}

// 9. Control Test 5: Private Intrigue Hidden From Opponent
void TestPrivateIntrigueHiddenFromOpponent() {
  std::cout << "Running TestPrivateIntrigueHiddenFromOpponent...\n";
  ScenarioConfig cfg;
  cfg.scenario_id = 0;
  cfg.family = ScenarioFamily::kCombatTension;
  cfg.round = 2;

  auto state1 = ScenarioGenerator::BuildScenarioState(cfg, nullptr);
  auto* s1 = dynamic_cast<DuneImperiumState*>(state1.get());
  s1->SetPlayerIntriguesForTesting(0, {kIntrigueWindfall});

  auto state2 = ScenarioGenerator::BuildScenarioState(cfg, nullptr);
  auto* s2 = dynamic_cast<DuneImperiumState*>(state2.get());
  s2->SetPlayerIntriguesForTesting(0, {kIntrigueSecretsOfTheSisterhood});

  // Strict byte equality for opponent observation tensor
  std::vector<float> obs_p1_s1 = s1->ObservationTensor(1);
  std::vector<float> obs_p1_s2 = s2->ObservationTensor(1);
  SPIEL_CHECK_TRUE(obs_p1_s1 == obs_p1_s2);

  // Distinct for holder observation tensor
  std::vector<float> obs_p0_s1 = s1->ObservationTensor(0);
  std::vector<float> obs_p0_s2 = s2->ObservationTensor(0);
  SPIEL_CHECK_FALSE(obs_p0_s1 == obs_p0_s2);

  std::cout << "PASS: TestPrivateIntrigueHiddenFromOpponent\n";
}

// 10. Control Test 6: Replay Complete Input Verification
void TestReplayCompleteInputVerification() {
  std::cout << "Running TestReplayCompleteInputVerification...\n";
  auto game = open_spiel::LoadGame("dune_imperium");
  std::unique_ptr<State> state = game->NewInitialState();
  auto* s = dynamic_cast<DuneImperiumState*>(state.get());

  std::mt19937_64 chance_rng(20260915);
  std::vector<Action> history;
  AdvanceStatus st = AdvanceThroughChanceAndAcks(s, chance_rng, &history);
  SPIEL_CHECK_EQ(st, AdvanceStatus::kSuccessDecision);

  ScriptedNonreactiveControlEvaluator eval(0, 1);
  Player cp = s->CurrentPlayer();
  ModelInputSnapshot snap = eval.CaptureSnapshot(*s, cp);

  // Verification against exact history should succeed
  auto res = VerifyReplayHistory(game, history, cp, snap, &eval);
  SPIEL_CHECK_TRUE(res.matches);
  SPIEL_CHECK_EQ(res.steps_replayed, history.size());

  // Mutated history or snapshot should fail with descriptive error
  ModelInputSnapshot corrupted_snap = snap;
  corrupted_snap.state_string += "_corrupted";
  auto res_fail = VerifyReplayHistory(game, history, cp, corrupted_snap, &eval);
  SPIEL_CHECK_FALSE(res_fail.matches);
  SPIEL_CHECK_FALSE(res_fail.failure_reason.empty());

  std::cout << "PASS: TestReplayCompleteInputVerification\n";
}

// 11. Validate Full Scenario Bank Generation
void TestFullScenarioBankGeneration() {
  std::cout << "Running TestFullScenarioBankGeneration (128 positions)...\n";
  int rejected = 0;
  auto bank = ScenarioGenerator::GenerateBank(/*count=*/128, /*base_seed=*/20260915, &rejected);
  SPIEL_CHECK_EQ(bank.size(), 128);
  SPIEL_CHECK_EQ(rejected, 0);

  std::map<std::string, int> family_counts;
  for (const auto& cfg : bank) {
    family_counts[cfg.family_name]++;
    std::string err;
    auto state = ScenarioGenerator::BuildScenarioState(cfg, nullptr);
    SPIEL_CHECK_TRUE(ScenarioGenerator::VerifyScenarioRequirements(*state, cfg, &err));
  }

  std::cout << "Generated 128 scenarios successfully across families:\n";
  for (const auto& [fam, cnt] : family_counts) {
    std::cout << "  - " << fam << ": " << cnt << " positions\n";
    SPIEL_CHECK_EQ(cnt, 32);
  }

  std::cout << "PASS: TestFullScenarioBankGeneration\n";
}

// 12. Validate Diagnostic 2 Telemetry on Scripted Evaluator
void TestDiagnostic2Telemetry() {
  std::cout << "Running TestDiagnostic2Telemetry...\n";
  auto bank = ScenarioGenerator::GenerateBank(/*count=*/4, /*base_seed=*/123);
  ScriptedNonreactiveControlEvaluator eval(/*p_holder=*/0, /*p_competitor=*/1);

  auto [records, summary] = RunDiagnostic2Suite(bank, &eval, /*replicates_per_scenario=*/4, /*base_seed=*/123);
  SPIEL_CHECK_GT(summary.total_decisions_evaluated, 0);
  std::cout << "Diagnostic 2 evaluated " << summary.total_decisions_evaluated << " decisions.\n";

  for (const auto& r : records) {
    SPIEL_CHECK_GE(r.windfall_prob, 0.0);
    SPIEL_CHECK_LE(r.windfall_prob, 1.0);
    SPIEL_CHECK_GE(r.windfall_rank, 1);
    SPIEL_CHECK_LE(r.windfall_rank, r.legal_actions_count);
    SPIEL_CHECK_GE(r.opportunity_index, 0);
    SPIEL_CHECK_GE(r.agent_turn_index, 1);
    SPIEL_CHECK_GE(r.decision_in_turn, 0);
  }

  SPIEL_CHECK_GT(summary.total_traces_evaluated, 0);
  std::cout << "PASS: TestDiagnostic2Telemetry\n";
}

// 13. Validate Ordinary Game Intrigue Mining on Scripted Evaluator
void TestIntrigueMiningSuiteScripted() {
  std::cout << "Running TestIntrigueMiningSuiteScripted...\n";
  ScriptedPositiveControlEvaluator pos_eval(/*p_holder=*/0, /*p_competitor=*/1);
  auto summary = RunOrdinaryGameIntrigueMining(
      &pos_eval, /*num_games=*/4, /*base_seed=*/42, /*discovery_pairs=*/2,
      /*max_candidates=*/4, /*confirmation_samples=*/4,
      /*max_outcome_candidates=*/2, /*continuation_samples=*/4);

  SPIEL_CHECK_EQ(summary.games_simulated, 4);
  std::cout << "Mining simulated " << summary.games_simulated << " natural games, detected "
            << summary.total_resource_intrigue_plays << " resource intrigue plays, discovered "
            << summary.candidates_discovered << " candidates, tested "
            << summary.candidates_tested_confirmation << " confirmation candidates.\n";

  std::cout << "PASS: TestIntrigueMiningSuiteScripted\n";
}

// ===========================================================================
// Focused Unit Tests for Narrow Windfall Timing Controller
// ===========================================================================

class MockSpendingPlanEvaluator : public IPolicyDiagnosticsEvaluator {
 public:
  MockSpendingPlanEvaluator(bool target_swordmaster = true, bool throw_in_proposal = false)
      : target_swordmaster_(target_swordmaster), throw_in_proposal_(throw_in_proposal) {}

  bool IsNeural() const override { return false; }
  std::string ModelName() const override { return "MockSpendingPlanEvaluator"; }
  std::string ModelSha256() const override { return "mock_spending_sha256"; }

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
    if (throw_in_proposal_ && throw_count_++ == 0) {
      throw std::runtime_error("Simulated proposal error for fallback test");
    }

    auto legal = state.LegalActions();
    SPIEL_CHECK_FALSE(legal.empty());

    // 1. If card selection action is available:
    if (dune_imperium::HasCardSelectionAction(legal)) {
      if (target_swordmaster_) {
        // Pick card with Landsraad access
        for (Action a : legal) {
          if (a >= kActionSelectAgentCard0 && a < kActionSelectAgentCard0 + 256) {
            int cid = a - kActionSelectAgentCard0;
            const auto* card = FindImperiumCardById(cid);
            if (card && (card->access_mask & dune_imperium::kAccessLandsraad)) {
              return a;
            }
          }
        }
      }
      return legal[0];
    }

    if (ContainsAction(legal, kActionPlayAgentSolo)) {
      return kActionPlayAgentSolo;
    }

    // 2. If board space: pick Swordmaster if available and targeted
    if (target_swordmaster_ && ContainsAction(legal, kActionAgentSpaceSwordmaster)) {
      return kActionAgentSpaceSwordmaster;
    }

    // 3. If baseline prefers playing Windfall when legal
    if (ContainsAction(legal, kActionPlayWindfall)) {
      return kActionPlayWindfall;
    }

    // 4. Default progression
    return dune_imperium::DefaultProgressionAction(legal);
  }

 private:
  bool target_swordmaster_ = true;
  bool throw_in_proposal_ = false;
  int throw_count_ = 0;
};

// 14. Focused Test 1: Windfall unlocks 8-solari Swordmaster from 6 solari
void TestWindfallControllerUnlocksSwordmasterAtSixSolari() {
  std::cout << "Running TestWindfallControllerUnlocksSwordmasterAtSixSolari...\n";
  ScenarioConfig cfg;
  cfg.scenario_id = 0;
  cfg.round = 2;
  cfg.p_holder = 0;
  auto state = ScenarioGenerator::BuildScenarioState(cfg, nullptr);
  state->SetCurrentTurnMainActionDoneForTesting(0, false);
  SPIEL_CHECK_EQ(state->GetPlayerSolari(0), 6);
  SPIEL_CHECK_EQ(state->CurrentPlayer(), 0);

  MockSpendingPlanEvaluator base_eval(/*target_swordmaster=*/true);
  WindfallTimingControlledEvaluator controller(&base_eval, /*designated_holder=*/0);
  controller.SetContext(0, 0);

  std::mt19937_64 rng(42);
  Action a1 = controller.SelectAction(*state, 0, false, 1.0f, rng);
  // Controller should recognize that Swordmaster requires Windfall, so it plays Windfall!
  SPIEL_CHECK_EQ(a1, kActionPlayWindfall);

  const auto& recs = controller.DecisionRecords();
  SPIEL_CHECK_EQ(recs.size(), 1);
  SPIEL_CHECK_TRUE(recs[0].plan_requires_windfall);
  SPIEL_CHECK_TRUE(recs[0].windfall_played);
  SPIEL_CHECK_EQ(recs[0].solari_after, 8);

  // Apply Windfall on real state
  state->ApplyAction(kActionPlayWindfall);
  AdvanceThroughChanceAndAcks(state.get(), rng);
  SPIEL_CHECK_EQ(state->GetPlayerSolari(0), 8);

  // Next action: card selection with Landsraad access
  Action a2 = controller.SelectAction(*state, 0, false, 1.0f, rng);
  SPIEL_CHECK_TRUE(a2 >= kActionSelectAgentCard0 && a2 < kActionSelectAgentCard0 + 256);
  state->ApplyAction(a2);
  AdvanceThroughChanceAndAcks(state.get(), rng);

  // Solo play
  if (ContainsAction(state->LegalActions(), kActionPlayAgentSolo)) {
    state->ApplyAction(kActionPlayAgentSolo);
    AdvanceThroughChanceAndAcks(state.get(), rng);
  }

  // Board space: Swordmaster
  Action a3 = controller.SelectAction(*state, 0, false, 1.0f, rng);
  SPIEL_CHECK_EQ(a3, kActionAgentSpaceSwordmaster);
  state->ApplyAction(a3);
  AdvanceThroughChanceAndAcks(state.get(), rng);

  SPIEL_CHECK_TRUE(state->HasSwordmaster(0));
  std::cout << "PASS: TestWindfallControllerUnlocksSwordmasterAtSixSolari\n";
}

// 15. Focused Test 2: Controller retains Windfall when purchase is already affordable
void TestWindfallControllerRetainsWhenAlreadyAffordable() {
  std::cout << "Running TestWindfallControllerRetainsWhenAlreadyAffordable...\n";
  ScenarioConfig cfg;
  cfg.scenario_id = 0;
  cfg.round = 2;
  cfg.p_holder = 0;
  auto state = ScenarioGenerator::BuildScenarioState(cfg, nullptr);
  state->SetCurrentTurnMainActionDoneForTesting(0, false);
  // Set solari to 8 (already affordable)
  state->SetPlayerSolariForTesting(0, 8);

  MockSpendingPlanEvaluator base_eval(/*target_swordmaster=*/true);
  WindfallTimingControlledEvaluator controller(&base_eval, /*designated_holder=*/0);
  controller.SetContext(0, 0);

  std::mt19937_64 rng(42);
  Action a1 = controller.SelectAction(*state, 0, false, 1.0f, rng);
  // Controller sees Swordmaster is already affordable, so it RETAINS Windfall and selects the card!
  SPIEL_CHECK_NE(a1, kActionPlayWindfall);
  SPIEL_CHECK_TRUE(a1 >= kActionSelectAgentCard0 && a1 < kActionSelectAgentCard0 + 256);

  const auto& recs = controller.DecisionRecords();
  SPIEL_CHECK_EQ(recs.size(), 1);
  SPIEL_CHECK_FALSE(recs[0].plan_requires_windfall);
  SPIEL_CHECK_FALSE(recs[0].windfall_played);
  SPIEL_CHECK_EQ(recs[0].solari_after, 8);

  // Verify Windfall is still possessed in real state
  auto intrigues = state->GetPlayerIntrigues(0);
  SPIEL_CHECK_TRUE(std::find(intrigues.begin(), intrigues.end(), kIntrigueWindfall) != intrigues.end());

  std::cout << "PASS: TestWindfallControllerRetainsWhenAlreadyAffordable\n";
}

// 16. Focused Test 3: Controller retains Windfall through EndTurn when spending is finished
void TestWindfallControllerRetainsThroughEndTurn() {
  std::cout << "Running TestWindfallControllerRetainsThroughEndTurn...\n";
  ScenarioConfig cfg;
  cfg.scenario_id = 0;
  cfg.round = 2;
  cfg.p_holder = 0;
  auto state = ScenarioGenerator::BuildScenarioState(cfg, nullptr);

  // Set holder agents remaining to 0 and solari to 6, simulating reveal/end-turn condition
  state->SetPlayerAgentsRemainingForTesting(0, 0);

  MockSpendingPlanEvaluator base_eval(/*target_swordmaster=*/false);
  WindfallTimingControlledEvaluator controller(&base_eval, /*designated_holder=*/0);
  controller.SetContext(0, 0);

  std::mt19937_64 rng(42);
  Action a = controller.SelectAction(*state, 0, false, 1.0f, rng);

  // Should NOT play Windfall
  SPIEL_CHECK_NE(a, kActionPlayWindfall);
  const auto& recs = controller.DecisionRecords();
  if (!recs.empty()) {
    SPIEL_CHECK_FALSE(recs.back().windfall_played);
  }

  std::cout << "PASS: TestWindfallControllerRetainsThroughEndTurn\n";
}

// 17. Focused Test 4: Intrigue timing window around card selection and placement
void TestIntrigueTimingWindowBeforeCardSelection() {
  std::cout << "Running TestIntrigueTimingWindowBeforeCardSelection...\n";
  ScenarioConfig cfg;
  cfg.scenario_id = 0;
  cfg.round = 2;
  cfg.p_holder = 0;
  auto state = ScenarioGenerator::BuildScenarioState(cfg, nullptr);
  state->SetCurrentTurnMainActionDoneForTesting(0, false);

  // Window 1: Before card selection, kActionPlayWindfall is legal
  auto legal_before = state->LegalActions();
  SPIEL_CHECK_TRUE(ContainsAction(legal_before, kActionPlayWindfall));

  // Select agent card
  Action select_act = open_spiel::kInvalidAction;
  for (Action a : legal_before) {
    if (a >= kActionSelectAgentCard0 && a < kActionSelectAgentCard0 + 256) {
      select_act = a;
      break;
    }
  }
  SPIEL_CHECK_NE(select_act, open_spiel::kInvalidAction);
  state->ApplyAction(select_act);
  std::mt19937_64 rng(42);
  AdvanceThroughChanceAndAcks(state.get(), rng);

  // Window 2: After card selection, kActionPlayWindfall is NOT legal!
  auto legal_after = state->LegalActions();
  SPIEL_CHECK_FALSE(ContainsAction(legal_after, kActionPlayWindfall));

  std::cout << "PASS: TestIntrigueTimingWindowBeforeCardSelection\n";
}

// 18. Focused Test 5: Secrets exception preserves baseline behavior
void TestSecretsExceptionPreservesBaseline() {
  std::cout << "Running TestSecretsExceptionPreservesBaseline...\n";
  ScenarioConfig cfg;
  cfg.scenario_id = 0;
  cfg.round = 2;
  cfg.p_holder = 0;
  auto state = ScenarioGenerator::BuildScenarioState(cfg, nullptr);

  // Set holder to have 4 intrigue cards (exposure risk)
  state->SetPlayerIntriguesForTesting(
      0, {kIntrigueWindfall, kIntrigueSecretsOfTheSisterhood, kIntrigueCalculatedHire, kIntrigueAmbush});
  SPIEL_CHECK_GE(state->GetPlayerIntrigues(0).size(), 4);

  // Evaluator preferring Windfall
  MockSpendingPlanEvaluator base_eval(/*target_swordmaster=*/false);
  WindfallTimingControlledEvaluator controller(&base_eval, /*designated_holder=*/0);
  controller.SetContext(0, 0);

  std::mt19937_64 rng(42);
  Action a = controller.SelectAction(*state, 0, false, 1.0f, rng);

  // With 4 intrigues, secrets exception applies and baseline decision (Windfall) is preserved!
  SPIEL_CHECK_EQ(a, kActionPlayWindfall);
  const auto& recs = controller.DecisionRecords();
  SPIEL_CHECK_EQ(recs.size(), 1);
  SPIEL_CHECK_TRUE(recs[0].secrets_exception);
  SPIEL_CHECK_TRUE(recs[0].windfall_played);

  std::cout << "PASS: TestSecretsExceptionPreservesBaseline\n";
}

// 19. Focused Test 6: Speculative proposals do not mutate real state
void TestSpeculativeProposalsDoNotMutateRealState() {
  std::cout << "Running TestSpeculativeProposalsDoNotMutateRealState...\n";
  ScenarioConfig cfg;
  cfg.scenario_id = 0;
  cfg.round = 2;
  cfg.p_holder = 0;
  auto state = ScenarioGenerator::BuildScenarioState(cfg, nullptr);
  state->SetCurrentTurnMainActionDoneForTesting(0, false);

  std::string before_str = state->ToString();
  int before_solari = state->GetPlayerSolari(0);
  auto before_legal = state->LegalActions();
  auto before_intrigues = state->GetPlayerIntrigues(0);

  MockSpendingPlanEvaluator base_eval(/*target_swordmaster=*/true);
  WindfallTimingControlledEvaluator controller(&base_eval, /*designated_holder=*/0);
  controller.SetContext(0, 0);

  std::mt19937_64 rng(42);
  Action chosen = controller.SelectAction(*state, 0, false, 1.0f, rng);
  SPIEL_CHECK_NE(chosen, open_spiel::kInvalidAction);

  // Verify real state was not mutated by proposal clone simulation
  SPIEL_CHECK_EQ(state->ToString(), before_str);
  SPIEL_CHECK_EQ(state->GetPlayerSolari(0), before_solari);
  SPIEL_CHECK_EQ(state->LegalActions(), before_legal);
  SPIEL_CHECK_EQ(state->GetPlayerIntrigues(0), before_intrigues);

  std::cout << "PASS: TestSpeculativeProposalsDoNotMutateRealState\n";
}

// 20. Focused Test 7: Controller chance and opponent isolation
void TestControllerChanceAndOpponentIsolation() {
  std::cout << "Running TestControllerChanceAndOpponentIsolation...\n";
  ScenarioConfig cfg;
  cfg.scenario_id = 0;
  cfg.round = 2;
  cfg.p_holder = 0;
  cfg.p_competitor = 1;
  auto state = ScenarioGenerator::BuildScenarioState(cfg, nullptr);
  state->SetCurrentTurnMainActionDoneForTesting(0, false);

  // Give competitor private secret cards
  state->SetPlayerIntriguesForTesting(1, {kIntrigueSecretsOfTheSisterhood});
  std::string opp_intrigues_before = state->ToString();

  MockSpendingPlanEvaluator base_eval(/*target_swordmaster=*/true);
  WindfallTimingControlledEvaluator controller(&base_eval, /*designated_holder=*/0);
  controller.SetContext(0, 0);

  std::mt19937_64 rng(42);
  controller.SelectAction(*state, 0, false, 1.0f, rng);

  // Competitor private hand remains untouched
  auto opp_intrigues_after = state->GetPlayerIntrigues(1);
  SPIEL_CHECK_EQ(opp_intrigues_after.size(), 1);
  SPIEL_CHECK_EQ(opp_intrigues_after[0], kIntrigueSecretsOfTheSisterhood);

  std::cout << "PASS: TestControllerChanceAndOpponentIsolation\n";
}

// 21. Focused Test 8: Normal chance sampling during real play
void TestNormalChanceSamplingDuringRealPlay() {
  std::cout << "Running TestNormalChanceSamplingDuringRealPlay...\n";
  auto game = open_spiel::LoadGame("dune_imperium");
  std::unique_ptr<State> state = game->NewInitialState();
  SPIEL_CHECK_TRUE(state->IsChanceNode());

  std::mt19937_64 rng(12345);
  std::vector<Action> history;
  AdvanceStatus st = AdvanceThroughChanceAndAcks(state.get(), rng, &history);
  SPIEL_CHECK_EQ(st, AdvanceStatus::kSuccessDecision);
  SPIEL_CHECK_GT(history.size(), 5);

  std::cout << "PASS: TestNormalChanceSamplingDuringRealPlay\n";
}

// 22. Focused Test 9: Necessary spending and terminal resource value preserved
void TestTerminalValuePreservation() {
  std::cout << "Running TestTerminalValuePreservation...\n";
  ScenarioConfig cfg;
  cfg.scenario_id = 0;
  cfg.round = 10; // Round 10 endgame
  cfg.p_holder = 0;
  auto state = ScenarioGenerator::BuildScenarioState(cfg, nullptr);

  // Holder has 0 agents remaining and solari is 6; round is 10
  state->SetPlayerAgentsRemainingForTesting(0, 0);

  MockSpendingPlanEvaluator base_eval(/*target_swordmaster=*/false);
  WindfallTimingControlledEvaluator controller(&base_eval, /*designated_holder=*/0);
  controller.SetContext(0, 0);

  std::mt19937_64 rng(42);
  Action a = controller.SelectAction(*state, 0, false, 1.0f, rng);

  // Terminal value exception triggers Windfall play to preserve solari tiebreaker!
  SPIEL_CHECK_EQ(a, kActionPlayWindfall);
  const auto& recs = controller.DecisionRecords();
  SPIEL_CHECK_EQ(recs.size(), 1);
  SPIEL_CHECK_TRUE(recs[0].terminal_value_exception);
  SPIEL_CHECK_TRUE(recs[0].windfall_played);

  // Case B: Economic Positioning threshold
  state->SetRoundNumberForTesting(5); // Non-endgame round
  state->SetPlayerSolariForTesting(0, 8);
  state->SetPlayerIntriguesForTesting(0, {kIntrigueWindfall, kIntrigueEconomicPositioning});
  state->SetPlayerAgentsRemainingForTesting(0, 1);
  controller.ClearDecisionRecords();

  Action a_ep = controller.SelectAction(*state, 0, false, 1.0f, rng);
  SPIEL_CHECK_EQ(a_ep, kActionPlayWindfall);
  const auto& recs_ep = controller.DecisionRecords();
  SPIEL_CHECK_EQ(recs_ep.size(), 1);
  SPIEL_CHECK_TRUE(recs_ep[0].terminal_value_exception);
  SPIEL_CHECK_TRUE(recs_ep[0].windfall_played);

  std::cout << "PASS: TestTerminalValuePreservation\n";
}

// 23. Focused Test 10: Fallback logging on unsupported or exception
void TestFallbackLoggingOnUnsupportedOrException() {
  std::cout << "Running TestFallbackLoggingOnUnsupportedOrException...\n";
  ScenarioConfig cfg;
  cfg.scenario_id = 0;
  cfg.round = 2;
  cfg.p_holder = 0;
  auto state = ScenarioGenerator::BuildScenarioState(cfg, nullptr);

  // Throwing evaluator triggers catch block
  MockSpendingPlanEvaluator base_eval(/*target_swordmaster=*/true, /*throw_in_proposal=*/true);
  WindfallTimingControlledEvaluator controller(&base_eval, /*designated_holder=*/0);
  controller.SetContext(0, 0);

  std::mt19937_64 rng(42);
  // Controller should catch exception, log fallback, and return safe baseline action
  try {
    controller.SelectAction(*state, 0, false, 1.0f, rng);
  } catch (const std::exception& e) {
    // Should NOT escape
    SPIEL_CHECK_TRUE(false);
  }

  const auto& recs = controller.DecisionRecords();
  SPIEL_CHECK_EQ(recs.size(), 1);
  SPIEL_CHECK_TRUE(recs[0].fallback_active);
  SPIEL_CHECK_FALSE(recs[0].fallback_reason.empty());

  std::cout << "PASS: TestFallbackLoggingOnUnsupportedOrException\n";
}

// 24. Focused Test 11: Identical branch replay produces identical results with matched streams
void TestIdenticalBranchReplayDeterministic() {
  std::cout << "Running TestIdenticalBranchReplayDeterministic...\n";
  ScenarioConfig cfg;
  cfg.scenario_id = 0;
  cfg.round = 2;
  cfg.p_holder = 0;
  auto state_a = ScenarioGenerator::BuildScenarioState(cfg, nullptr);
  auto state_b = ScenarioGenerator::BuildScenarioState(cfg, nullptr);
  state_a->SetCurrentTurnMainActionDoneForTesting(0, false);
  state_b->SetCurrentTurnMainActionDoneForTesting(0, false);

  MockSpendingPlanEvaluator base_eval(/*target_swordmaster=*/true);
  WindfallTimingControlledEvaluator controller_a(&base_eval, /*designated_holder=*/0);
  WindfallTimingControlledEvaluator controller_b(&base_eval, /*designated_holder=*/0);

  std::mt19937_64 rng_a(777);
  std::mt19937_64 rng_b(777);

  Action act_a = controller_a.SelectAction(*state_a, 0, false, 1.0f, rng_a);
  Action act_b = controller_b.SelectAction(*state_b, 0, false, 1.0f, rng_b);

  SPIEL_CHECK_EQ(act_a, act_b);
  SPIEL_CHECK_EQ(controller_a.DecisionRecords().size(), controller_b.DecisionRecords().size());
  SPIEL_CHECK_EQ(controller_a.DecisionRecords()[0].windfall_played,
                 controller_b.DecisionRecords()[0].windfall_played);

  std::cout << "PASS: TestIdenticalBranchReplayDeterministic\n";
}

} // namespace
} // namespace dune_diagnostics
} // namespace open_spiel

int main() {
  open_spiel::dune_diagnostics::TestEarlyVsHoldStateDivergence();
  open_spiel::dune_diagnostics::TestCloningImmutability();
  open_spiel::dune_diagnostics::TestObservationTensorsAndInformationHiding();
  open_spiel::dune_diagnostics::TestSwordmasterLegalityAndBlocking();
  open_spiel::dune_diagnostics::TestProductionChanceHelper();
  open_spiel::dune_diagnostics::TestIdenticalBranchReplay();
  open_spiel::dune_diagnostics::TestScriptedPositiveControl();
  open_spiel::dune_diagnostics::TestScriptedNonreactiveControl();
  open_spiel::dune_diagnostics::TestPrivateIntrigueHiddenFromOpponent();
  open_spiel::dune_diagnostics::TestReplayCompleteInputVerification();
  open_spiel::dune_diagnostics::TestFullScenarioBankGeneration();
  open_spiel::dune_diagnostics::TestDiagnostic2Telemetry();
  open_spiel::dune_diagnostics::TestIntrigueMiningSuiteScripted();

  // Focused unit tests for Windfall-timing controller
  open_spiel::dune_diagnostics::TestWindfallControllerUnlocksSwordmasterAtSixSolari();
  open_spiel::dune_diagnostics::TestWindfallControllerRetainsWhenAlreadyAffordable();
  open_spiel::dune_diagnostics::TestWindfallControllerRetainsThroughEndTurn();
  open_spiel::dune_diagnostics::TestIntrigueTimingWindowBeforeCardSelection();
  open_spiel::dune_diagnostics::TestSecretsExceptionPreservesBaseline();
  open_spiel::dune_diagnostics::TestSpeculativeProposalsDoNotMutateRealState();
  open_spiel::dune_diagnostics::TestControllerChanceAndOpponentIsolation();
  open_spiel::dune_diagnostics::TestNormalChanceSamplingDuringRealPlay();
  open_spiel::dune_diagnostics::TestTerminalValuePreservation();
  open_spiel::dune_diagnostics::TestFallbackLoggingOnUnsupportedOrException();
  open_spiel::dune_diagnostics::TestIdenticalBranchReplayDeterministic();

  std::cout << "\n============================================\n";
  std::cout << "ALL FROZEN POLICY DIAGNOSTICS TESTS PASSED!\n";
  std::cout << "============================================\n";
  return 0;
}
