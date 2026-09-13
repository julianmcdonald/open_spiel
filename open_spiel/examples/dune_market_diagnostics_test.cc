// Copyright 2026
// Unit tests for Dune Imperium Market Purchase Diagnostics.

#include "dune_market_diagnostics.h"

#include <filesystem>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "open_spiel/games/dune_imperium/dune_imperium.h"
#include "open_spiel/games/dune_imperium/dune_imperium_cards.h"
#include "open_spiel/games/dune_imperium/dune_imperium_common.h"
#include "open_spiel/games/dune_imperium/dune_imperium_content_generated.h"
#include "open_spiel/games/dune_imperium/dune_imperium_test_utils.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace dune_market_diag {
namespace {

using dune_imperium::ActionToString;
using dune_imperium::BuildStateAtFirstRevealTurn;
using dune_imperium::DuneImperiumState;
using dune_imperium::IntrigueChoiceKind;
using dune_imperium::kActionBuyImperiumRow0;
using dune_imperium::kActionBuyReserveArrakisLiaison;
using dune_imperium::kActionBuyReserveTheSpiceMustFlow;
using dune_imperium::kActionIntrigueChoiceCount0;
using dune_imperium::kActionIntrigueChoiceImperiumRow0;
using dune_imperium::kActionIntrigueChoiceImperiumRowTop0;
using dune_imperium::kActionIntrigueChoiceModeA;
using dune_imperium::kActionIntrigueChoiceModeB;
using dune_imperium::kActionIntrigueChoiceSkip;
using dune_imperium::kActionIntrigueChoiceTech0;
using dune_imperium::kActionTechAcquire0;
using dune_imperium::kActionTechAcquireWithSolari0;
using dune_imperium::kActionTleilaxuAcquire0;
using dune_imperium::kCardArrakisLiaison;
using dune_imperium::kCardDagger;
using dune_imperium::kCardReconnaissance;
using dune_imperium::kCardTheSpiceMustFlow;
using dune_imperium::kExpansionAll;
using dune_imperium::kExpansionImmortality;
using dune_imperium::kExpansionIx;
using dune_imperium::kImperiumCards;
using dune_imperium::kTechSpaceport;
using dune_imperium::kTechTiles;
using dune_imperium::kTleilaxuCards;

// ---------------------------------------------------------------------------
// 1. Tech Purchase Routes: Solari (70..72), Spice (770..772), Choices (554..556)
// ---------------------------------------------------------------------------
void TestTechPurchaseRoutes() {
  std::cout << "Running TestTechPurchaseRoutes...\n";
  auto state_ptr = BuildStateAtFirstRevealTurn();
  auto* state = dynamic_cast<DuneImperiumState*>(state_ptr.get());
  SPIEL_CHECK_TRUE(state != nullptr);

  const int p = state->CurrentPlayer();
  state->SetPlayerSolariForTesting(p, 20);
  state->SetPlayerSpiceForTesting(p, 20);

  // Setup tech market tile in slot 0
  const int tech_tile = state->GetTechMarketTile(0);
  SPIEL_CHECK_GE(tech_tile, 0);
  SPIEL_CHECK_LT(tech_tile, kNumTechTiles);

  GameMarketTracker tracker;
  tracker.InitGame(/*episode_id=*/1, /*candidate_seat=*/p);

  // Set pending space tech acquire so tech actions are legal
  state->SetPendingSpaceTechAcquirePlayerForTesting(p);

  // Test Route A: Solari/Spice acquire (action 70)
  auto legal = state->LegalActions();
  tracker.OnCandidateDecision(*state, p, legal);
  const auto& data1 = tracker.GetData();
  SPIEL_CHECK_TRUE(data1.tech_visible.test(tech_tile));
  SPIEL_CHECK_TRUE(data1.tech_had_legal.test(tech_tile));
  SPIEL_CHECK_GE(data1.tech_legal_opps[tech_tile], 1);

  tracker.BeforeApplyAction(*state, p, p, kActionTechAcquire0);
  state->ApplyAction(kActionTechAcquire0);
  tracker.AfterApplyAction(*state, p, p, kActionTechAcquire0);
  ResolvePendingChanceNodes(state);

  const auto& data2 = tracker.GetData();
  SPIEL_CHECK_TRUE(data2.tech_purchased.test(tech_tile));
  SPIEL_CHECK_EQ(data2.tech_purchases[tech_tile], 1);

  // Test Route B: Machine Culture / Rhombur signet choice (actions 554..556)
  state->SetPendingIntrigueChoiceForTesting(
      p, IntrigueChoiceKind::kMachineCultureTech, /*ends_turn=*/false);
  state->SetPendingIntrigueChoiceCardIdForTesting(-1);
  const int slot1_tile = state->GetTechMarketTile(1);
  legal = state->LegalActions();
  tracker.OnCandidateDecision(*state, p, legal);
  const auto& data3 = tracker.GetData();
  SPIEL_CHECK_TRUE(data3.tech_had_legal.test(slot1_tile));

  tracker.BeforeApplyAction(*state, p, p, kActionIntrigueChoiceTech0 + 1);
  state->ApplyAction(kActionIntrigueChoiceTech0 + 1);
  tracker.AfterApplyAction(*state, p, p, kActionIntrigueChoiceTech0 + 1);
  ResolvePendingChanceNodes(state);

  const auto& data4 = tracker.GetData();
  SPIEL_CHECK_TRUE(data4.tech_purchased.test(slot1_tile));
  SPIEL_CHECK_EQ(data4.tech_purchases[slot1_tile], 1);
  std::cout << "TestTechPurchaseRoutes PASSED.\n";
}

// ---------------------------------------------------------------------------
// 2. Appropriate Two-Phase Resolution
// ---------------------------------------------------------------------------
void TestAppropriateTwoPhaseResolution() {
  std::cout << "Running TestAppropriateTwoPhaseResolution...\n";
  auto state_ptr = BuildStateAtFirstRevealTurn();
  auto* state = dynamic_cast<DuneImperiumState*>(state_ptr.get());
  SPIEL_CHECK_TRUE(state != nullptr);

  const int p = state->CurrentPlayer();
  state->SetPlayerSolariForTesting(p, 10);
  state->SetPlayerSpiceForTesting(p, 10);

  const int target_slot = 1;
  const int target_tile = state->GetTechMarketTile(target_slot);

  GameMarketTracker tracker;
  tracker.InitGame(/*episode_id=*/2, /*candidate_seat=*/p);

  // Set pending appropriate tech acquire for player p
  state->SetPendingAppropriateTechAcquirePlayerForTesting(p);

  // Phase 1: Player selects the tile to acquire
  auto legal1 = state->LegalActions();
  tracker.OnCandidateDecision(*state, p, legal1);
  int opps_phase1 = tracker.GetData().tech_legal_opps[target_tile];
  SPIEL_CHECK_GE(opps_phase1, 1);

  // Apply Phase 1 action (Appropriate triggers currency choice if applicable)
  tracker.BeforeApplyAction(*state, p, p, kActionTechAcquire0 + target_slot);
  state->ApplyAction(kActionTechAcquire0 + target_slot);
  tracker.AfterApplyAction(*state, p, p, kActionTechAcquire0 + target_slot);

  // If in Phase 2, verify OnCandidateDecision does NOT double-count the decision opportunity
  if (state->GetPendingAppropriateSelectedTechSlot() >= 0) {
    auto legal2 = state->LegalActions();
    tracker.OnCandidateDecision(*state, p, legal2);
    int opps_phase2 = tracker.GetData().tech_legal_opps[target_tile];
    // Opps must NOT increment during Phase 2!
    SPIEL_CHECK_EQ(opps_phase2, opps_phase1);

    // Now resolve payment choice
    Action pay_action = kActionTechAcquireWithSolari0 + target_slot;
    tracker.BeforeApplyAction(*state, p, p, pay_action);
    state->ApplyAction(pay_action);
    tracker.AfterApplyAction(*state, p, p, pay_action);
    ResolvePendingChanceNodes(state);
  }

  // Confirm exactly 1 purchase recorded
  const auto& final_data = tracker.GetData();
  SPIEL_CHECK_EQ(final_data.tech_purchases[target_tile], 1);
  std::cout << "TestAppropriateTwoPhaseResolution PASSED.\n";
}

// ---------------------------------------------------------------------------
// 3. Delayed Acquisitions and Topdeck Placement (Spaceport / Research Marker 1)
// ---------------------------------------------------------------------------
void TestDelayedAcquisitionsAndTopdeckPlacement() {
  std::cout << "Running TestDelayedAcquisitionsAndTopdeckPlacement...\n";
  auto state_ptr = BuildStateAtFirstRevealTurn();
  auto* state = dynamic_cast<DuneImperiumState*>(state_ptr.get());
  SPIEL_CHECK_TRUE(state != nullptr);

  const int p = state->CurrentPlayer();
  state->SetPlayerPersuasionForTesting(p, 10);
  state->GainTechTileForTesting(p, kTechSpaceport);

  const int card_to_buy = state->GetImperiumRowCardForTesting(0);
  SPIEL_CHECK_GE(card_to_buy, 0);
  const int pre_owned = state->GetDeckPoolTotalCards(p, card_to_buy);

  GameMarketTracker tracker;
  tracker.InitGame(/*episode_id=*/3, /*candidate_seat=*/p);

  // Step 1: Buy card from row 0
  auto legal = state->LegalActions();
  tracker.OnCandidateDecision(*state, p, legal);

  tracker.BeforeApplyAction(*state, p, p, kActionBuyImperiumRow0);
  state->ApplyAction(kActionBuyImperiumRow0);
  tracker.AfterApplyAction(*state, p, p, kActionBuyImperiumRow0);

  // In pending placement:
  // 1) Spaceport choice is pending
  SPIEL_CHECK_EQ(state->GetPendingIntrigueChoiceKind(p),
                 static_cast<int>(IntrigueChoiceKind::kTechSpaceportTopdeckChoice));
  // 2) deck_pool_total_cards_ already incremented
  SPIEL_CHECK_EQ(state->GetDeckPoolTotalCards(p, card_to_buy), pre_owned + 1);
  // 3) Market tracker recorded exactly ONE purchase
  const auto& pending_data = tracker.GetData();
  SPIEL_CHECK_TRUE(pending_data.imperium_purchased.test(card_to_buy));
  SPIEL_CHECK_EQ(pending_data.imperium_purchases[card_to_buy], 1);

  // Drain chance refill before candidate decision
  ResolvePendingChanceNodes(state);
  SPIEL_CHECK_EQ(state->CurrentPlayer(), p);
  SPIEL_CHECK_EQ(state->GetPendingIntrigueChoiceKind(p),
                 static_cast<int>(IntrigueChoiceKind::kTechSpaceportTopdeckChoice));

  // Step 2: Resolve placement choice (Mode A: put on top of deck)
  auto choice_legal = state->LegalActions();
  tracker.OnCandidateDecision(*state, p, choice_legal);
  tracker.BeforeApplyAction(*state, p, p, kActionIntrigueChoiceModeA);
  state->ApplyAction(kActionIntrigueChoiceModeA);
  tracker.AfterApplyAction(*state, p, p, kActionIntrigueChoiceModeA);
  ResolvePendingChanceNodes(state);

  // In completed placement:
  // Verify purchase count is STILL exactly ONE (not double counted to 2!)
  const auto& completed_data = tracker.GetData();
  SPIEL_CHECK_EQ(completed_data.imperium_purchases[card_to_buy], 1);
  SPIEL_CHECK_EQ(state->GetDeckPoolTotalCards(p, card_to_buy), pre_owned + 1);

  std::cout << "TestDelayedAcquisitionsAndTopdeckPlacement PASSED.\n";
}

// ---------------------------------------------------------------------------
// 4. Tleilaxu Master Deferred Hand-Placement Choice (Marker 2 active vs inactive)
// ---------------------------------------------------------------------------
void TestTleilaxuMasterDeferredHandPlacement() {
  std::cout << "Running TestTleilaxuMasterDeferredHandPlacement...\n";
  auto state_ptr = BuildStateAtFirstRevealTurn();
  auto* state = dynamic_cast<DuneImperiumState*>(state_ptr.get());
  SPIEL_CHECK_TRUE(state != nullptr);

  const int p = state->CurrentPlayer();

  // Activate Genetic Marker 2 (column >= kResearchSecondMarkerColumn)
  state->SetPlayerResearchBottomTrackPosForTesting(p, dune_imperium::kResearchSecondMarkerColumn, 0);
  SPIEL_CHECK_TRUE(state->HasSecondGeneticMarker(p));

  const int card_id = state->GetImperiumRowCardForTesting(0);
  SPIEL_CHECK_GE(card_id, 0);
  const int pre_owned = state->GetDeckPoolTotalCards(p, card_id);

  GameMarketTracker tracker;
  tracker.InitGame(/*episode_id=*/4, /*candidate_seat=*/p);

  // Setup Tleilaxu Master acquire choice
  state->SetPendingIntrigueChoiceForTesting(
      p, IntrigueChoiceKind::kTleilaxuMasterAcquire, /*ends_turn=*/false);
  state->SetPendingIntrigueChoiceCardIdForTesting(-1);

  auto legal1 = state->LegalActions();
  tracker.OnCandidateDecision(*state, p, legal1);
  SPIEL_CHECK_GE(tracker.GetData().tleilaxu_master_opps[card_id], 1);

  // Action 1: Select row slot 0
  tracker.BeforeApplyAction(*state, p, p, kActionIntrigueChoiceImperiumRow0);
  state->ApplyAction(kActionIntrigueChoiceImperiumRow0);
  tracker.AfterApplyAction(*state, p, p, kActionIntrigueChoiceImperiumRow0);

  // While pending hand placement:
  // 1) kTleilaxuMasterToHandChoice is pending
  SPIEL_CHECK_EQ(state->GetPendingIntrigueChoiceKind(p),
                 static_cast<int>(IntrigueChoiceKind::kTleilaxuMasterToHandChoice));
  // 2) deck_pool_total_cards_ has NOT yet incremented (ownership increments on resolution)
  SPIEL_CHECK_EQ(state->GetDeckPoolTotalCards(p, card_id), pre_owned);
  // 3) Transaction is preserved, acquisitions not counted yet
  SPIEL_CHECK_EQ(tracker.GetData().tleilaxu_master_acquisitions[card_id], 0);

  // Action 2: Choose Mode A (put into hand)
  auto legal2 = state->LegalActions();
  tracker.OnCandidateDecision(*state, p, legal2);
  tracker.BeforeApplyAction(*state, p, p, kActionIntrigueChoiceModeA);
  state->ApplyAction(kActionIntrigueChoiceModeA);
  tracker.AfterApplyAction(*state, p, p, kActionIntrigueChoiceModeA);
  ResolvePendingChanceNodes(state);

  // After resolution:
  // 1) deck_pool_total_cards_ is now pre_owned + 1
  SPIEL_CHECK_EQ(state->GetDeckPoolTotalCards(p, card_id), pre_owned + 1);
  // 2) Transaction completed: exactly ONE acquisition recorded!
  SPIEL_CHECK_EQ(tracker.GetData().tleilaxu_master_acquisitions[card_id], 1);

  std::cout << "TestTleilaxuMasterDeferredHandPlacement PASSED.\n";
}

// ---------------------------------------------------------------------------
// 5. Reclaimed Forces Paid Option Resolution
// ---------------------------------------------------------------------------
void TestReclaimedForcesPaidOption() {
  std::cout << "Running TestReclaimedForcesPaidOption...\n";
  auto state_ptr = BuildStateAtFirstRevealTurn();
  auto* state = dynamic_cast<DuneImperiumState*>(state_ptr.get());
  SPIEL_CHECK_TRUE(state != nullptr);

  const int p = state->CurrentPlayer();
  state->SetSpecimensForTesting(p, 5);

  GameMarketTracker tracker;
  tracker.InitGame(/*episode_id=*/5, /*candidate_seat=*/p);

  auto legal = state->LegalActions();
  tracker.OnCandidateDecision(*state, p, legal);
  const auto& data1 = tracker.GetData();
  SPIEL_CHECK_TRUE(data1.reclaimed_forces.visible);
  SPIEL_CHECK_TRUE(data1.reclaimed_forces.had_legal);
  SPIEL_CHECK_GE(data1.reclaimed_forces.legal_opps, 1);

  // Apply Reclaimed Forces (action kActionTleilaxuAcquire0)
  tracker.BeforeApplyAction(*state, p, p, kActionTleilaxuAcquire0);
  state->ApplyAction(kActionTleilaxuAcquire0);
  tracker.AfterApplyAction(*state, p, p, kActionTleilaxuAcquire0);

  // Verify Reclaimed Forces purchased recorded
  const auto& data2 = tracker.GetData();
  SPIEL_CHECK_TRUE(data2.reclaimed_forces.purchased);
  SPIEL_CHECK_EQ(data2.reclaimed_forces.purchases, 1);

  // Reclaimed Forces produces a subchoice (troops vs scarab); resolve it
  SPIEL_CHECK_EQ(state->GetPendingIntrigueChoiceKind(p),
                 static_cast<int>(IntrigueChoiceKind::kReclaimedForcesChoice));
  state->ApplyAction(kActionIntrigueChoiceModeA);

  std::cout << "TestReclaimedForcesPaidOption PASSED.\n";
}

// ---------------------------------------------------------------------------
// 6. Bypass Protocol Free vs Paid Branches
// ---------------------------------------------------------------------------
void TestBypassProtocolBranches() {
  std::cout << "Running TestBypassProtocolBranches...\n";
  auto state_ptr = BuildStateAtFirstRevealTurn();
  auto* state = dynamic_cast<DuneImperiumState*>(state_ptr.get());
  SPIEL_CHECK_TRUE(state != nullptr);

  const int p = state->CurrentPlayer();
  state->SetPlayerSpiceForTesting(p, 5);

  // Put cheap card in slot 0 (cost <= 3)
  state->SetImperiumRowCardForTesting(0, kCardReconnaissance);  // cost 1
  // Put medium card in slot 1 (cost <= 5)
  state->SetImperiumRowCardForTesting(1, kCardArrakisLiaison);   // cost 2

  state->SetPendingIntrigueChoiceForTesting(
      p, IntrigueChoiceKind::kBypassProtocol, /*ends_turn=*/false);
  state->SetPendingIntrigueChoiceCardIdForTesting(-1);

  GameMarketTracker tracker;
  tracker.InitGame(/*episode_id=*/6, /*candidate_seat=*/p);

  auto legal = state->LegalActions();
  tracker.OnCandidateDecision(*state, p, legal);

  const auto& data1 = tracker.GetData();
  // Both cards should have opportunities in free and/or paid branches
  SPIEL_CHECK_GE(data1.bypass_free_opps[kCardReconnaissance], 1);
  SPIEL_CHECK_GE(data1.bypass_paid_opps[kCardReconnaissance], 1);

  // 1. Test Free branch acquisition (slot 0)
  tracker.BeforeApplyAction(*state, p, p, kActionIntrigueChoiceImperiumRow0);
  state->ApplyAction(kActionIntrigueChoiceImperiumRow0);
  tracker.AfterApplyAction(*state, p, p, kActionIntrigueChoiceImperiumRow0);
  ResolvePendingChanceNodes(state);

  const auto& data2 = tracker.GetData();
  SPIEL_CHECK_EQ(data2.bypass_free_acquisitions[kCardReconnaissance], 1);
  SPIEL_CHECK_EQ(data2.bypass_paid_acquisitions[kCardReconnaissance], 0);

  // 2. Test Paid branch acquisition (slot 1)
  state->SetPlayerSpiceForTesting(p, 5);
  state->SetImperiumRowCardForTesting(1, kCardArrakisLiaison);
  state->SetPendingIntrigueChoiceForTesting(
      p, IntrigueChoiceKind::kBypassProtocol, /*ends_turn=*/false);
  state->SetPendingIntrigueChoiceCardIdForTesting(-1);

  auto legal_paid = state->LegalActions();
  tracker.OnCandidateDecision(*state, p, legal_paid);
  const Action paid_action = kActionIntrigueChoiceImperiumRowTop0 + 1;
  SPIEL_CHECK_TRUE(std::find(legal_paid.begin(), legal_paid.end(), paid_action) != legal_paid.end());

  tracker.BeforeApplyAction(*state, p, p, paid_action);
  state->ApplyAction(paid_action);
  tracker.AfterApplyAction(*state, p, p, paid_action);
  ResolvePendingChanceNodes(state);

  const auto& data3 = tracker.GetData();
  SPIEL_CHECK_EQ(data3.bypass_paid_acquisitions[kCardArrakisLiaison], 1);
  SPIEL_CHECK_EQ(state->GetPlayerSpiceForTesting(p), 3);  // 5 - 2 = 3 spice

  std::cout << "TestBypassProtocolBranches PASSED.\n";
}

// ---------------------------------------------------------------------------
// 7. Harvest Cells Acquisitions: Rotating Tleilaxu Card & Reclaimed Forces
// ---------------------------------------------------------------------------
void TestHarvestCellsAcquisitions() {
  std::cout << "Running TestHarvestCellsAcquisitions...\n";

  // --- Route A: Rotating Tleilaxu card purchase ---
  {
    auto state_ptr = BuildStateAtFirstRevealTurn();
    auto* state = dynamic_cast<DuneImperiumState*>(state_ptr.get());
    SPIEL_CHECK_TRUE(state != nullptr);

    const int p = state->CurrentPlayer();
    state->SetSpecimensForTesting(p, 10);

    // Slot 0 = Corrino Genes (Tleilaxu id 4, cost 1, imperium card 111).
    // Slot 1 = Guild Banker (Tleilaxu id 8, cost 2, imperium card 115).
    state->SetTleilaxuRowForTesting({4, 8});

    const int tlx_id = 4;
    const int imperium_id = kTleilaxuCards[tlx_id].imperium_card_id;
    const int pre_owned = state->GetDeckPoolTotalCards(p, imperium_id);
    const int pre_specimens = state->GetSpecimensForTesting(p);
    const int cost = kTleilaxuCards[tlx_id].specimen_cost;

    state->SetPendingIntrigueChoiceForTesting(
        p, IntrigueChoiceKind::kHarvestCellsBuy, /*ends_turn=*/false);
    state->SetPendingIntrigueChoiceCardIdForTesting(-1);

    GameMarketTracker tracker;
    tracker.InitGame(/*episode_id=*/10, /*candidate_seat=*/p);

    auto legal = state->LegalActions();
    const Action buy_slot0 = kActionIntrigueChoiceCount0 + 0;
    SPIEL_CHECK_TRUE(std::find(legal.begin(), legal.end(), buy_slot0) != legal.end());

    tracker.OnCandidateDecision(*state, p, legal);
    const auto& data1 = tracker.GetData();
    SPIEL_CHECK_TRUE(data1.tleilaxu_visible.test(tlx_id));
    SPIEL_CHECK_TRUE(data1.tleilaxu_had_legal.test(tlx_id));
    SPIEL_CHECK_GE(data1.tleilaxu_legal_opps[tlx_id], 1);
    SPIEL_CHECK_TRUE(data1.reclaimed_forces.visible);
    SPIEL_CHECK_TRUE(data1.reclaimed_forces.had_legal);
    SPIEL_CHECK_GE(data1.reclaimed_forces.legal_opps, 1);

    tracker.BeforeApplyAction(*state, p, p, buy_slot0);
    state->ApplyAction(buy_slot0);
    tracker.AfterApplyAction(*state, p, p, buy_slot0);

    // Verify specimens spent and ownership increased
    SPIEL_CHECK_EQ(state->GetSpecimensForTesting(p), pre_specimens - cost);
    SPIEL_CHECK_EQ(state->GetDeckPoolTotalCards(p, imperium_id), pre_owned + 1);

    const auto& data2 = tracker.GetData();
    SPIEL_CHECK_TRUE(data2.tleilaxu_purchased.test(tlx_id));
    SPIEL_CHECK_EQ(data2.tleilaxu_purchases[tlx_id], 1);

    // Drain chance refill before candidate decision
    ResolvePendingChanceNodes(state);

    // Next chained choice is kPostCombatChoice
    SPIEL_CHECK_EQ(state->GetPendingIntrigueChoiceKind(p),
                   static_cast<int>(IntrigueChoiceKind::kPostCombatChoice));
    auto chained_legal = state->LegalActions();
    tracker.OnCandidateDecision(*state, p, chained_legal);
    tracker.BeforeApplyAction(*state, p, p, kActionIntrigueChoiceSkip);
    state->ApplyAction(kActionIntrigueChoiceSkip);
    tracker.AfterApplyAction(*state, p, p, kActionIntrigueChoiceSkip);

    // Verify purchase count remains exactly 1 (no double counting)
    const auto& data3 = tracker.GetData();
    SPIEL_CHECK_EQ(data3.tleilaxu_purchases[tlx_id], 1);
  }

  // --- Route B: Reclaimed Forces purchase ---
  {
    auto state_ptr = BuildStateAtFirstRevealTurn();
    auto* state = dynamic_cast<DuneImperiumState*>(state_ptr.get());
    SPIEL_CHECK_TRUE(state != nullptr);

    const int p = state->CurrentPlayer();
    state->SetSpecimensForTesting(p, 5);
    state->SetTleilaxuRowForTesting({4, 8});

    state->SetPendingIntrigueChoiceForTesting(
        p, IntrigueChoiceKind::kHarvestCellsBuy, /*ends_turn=*/false);
    state->SetPendingIntrigueChoiceCardIdForTesting(-1);

    GameMarketTracker tracker;
    tracker.InitGame(/*episode_id=*/11, /*candidate_seat=*/p);

    auto legal = state->LegalActions();
    const Action buy_reclaimed = kActionIntrigueChoiceCount0 + 2;  // slot 2 is Reclaimed Forces
    SPIEL_CHECK_TRUE(std::find(legal.begin(), legal.end(), buy_reclaimed) != legal.end());

    tracker.OnCandidateDecision(*state, p, legal);
    tracker.BeforeApplyAction(*state, p, p, buy_reclaimed);
    state->ApplyAction(buy_reclaimed);
    tracker.AfterApplyAction(*state, p, p, buy_reclaimed);

    // Verify specimens spent (cost 3)
    SPIEL_CHECK_EQ(state->GetSpecimensForTesting(p), 2);

    const auto& data = tracker.GetData();
    SPIEL_CHECK_TRUE(data.reclaimed_forces.purchased);
    SPIEL_CHECK_EQ(data.reclaimed_forces.purchases, 1);

    // Verify chained choice is kReclaimedForcesChoice
    SPIEL_CHECK_EQ(state->GetPendingIntrigueChoiceKind(p),
                   static_cast<int>(IntrigueChoiceKind::kReclaimedForcesChoice));
    auto chained_legal = state->LegalActions();
    tracker.OnCandidateDecision(*state, p, chained_legal);
    tracker.BeforeApplyAction(*state, p, p, kActionIntrigueChoiceModeA);
    state->ApplyAction(kActionIntrigueChoiceModeA);
    tracker.AfterApplyAction(*state, p, p, kActionIntrigueChoiceModeA);

    // Verify purchase count remains exactly 1
    SPIEL_CHECK_EQ(tracker.GetData().reclaimed_forces.purchases, 1);
  }

  std::cout << "TestHarvestCellsAcquisitions PASSED.\n";
}

// ---------------------------------------------------------------------------
// 8. Fail-Closed Validation Tests (Rejecting Invalid Records)
// ---------------------------------------------------------------------------
void ThrowingErrorHandler(const std::string& msg) {
  throw std::runtime_error(msg);
}

void ExitingErrorHandler(const std::string& msg) {
  std::cerr << "Spiel Fatal Error: " << msg << std::endl;
  std::exit(1);
}

void TestFailClosedValidation() {
  std::cout << "Running TestFailClosedValidation...\n";

  open_spiel::SetErrorHandler(ThrowingErrorHandler);

  // Test 1: Surgeon reproduction case: purchases > legal opps (or purchases > 0 with 0 visibility)
  {
    CompactPerGameMarketDiagnostics bad_game;
    bad_game.episode_id = 0;
    bad_game.candidate_seat = 0;
    bad_game.tleilaxu_purchased.set(1);  // Tleilaxu card 1 (Tleilaxu Surgeon)
    bad_game.tleilaxu_purchases[1] = 1;
    bad_game.tleilaxu_purchase_rounds[1][1] = 1;
    // visible and had_legal are false, legal_opps is 0!

    bool caught = false;
    try {
      MarketDiagnosticsReport::AggregateReport({bad_game}, 1);
    } catch (const std::exception& e) {
      caught = true;
    }
    SPIEL_CHECK_TRUE(caught);
  }

  // Test 2: purchases > legal opps in a game where visible is true
  {
    CompactPerGameMarketDiagnostics bad_game;
    bad_game.episode_id = 0;
    bad_game.candidate_seat = 0;
    bad_game.imperium_visible.set(10);
    bad_game.imperium_had_legal.set(10);
    bad_game.imperium_legal_opps[10] = 1;
    bad_game.imperium_purchased.set(10);
    bad_game.imperium_purchases[10] = 2;  // purchases (2) > opps (1)
    bad_game.imperium_purchase_rounds[10][1] = 2;

    bool caught = false;
    try {
      MarketDiagnosticsReport::AggregateReport({bad_game}, 1);
    } catch (const std::exception& e) {
      caught = true;
    }
    SPIEL_CHECK_TRUE(caught);
  }

  // Test 3: Purchase round sum mismatch
  {
    CompactPerGameMarketDiagnostics bad_game;
    bad_game.episode_id = 0;
    bad_game.candidate_seat = 0;
    bad_game.imperium_visible.set(10);
    bad_game.imperium_had_legal.set(10);
    bad_game.imperium_legal_opps[10] = 2;
    bad_game.imperium_purchased.set(10);
    bad_game.imperium_purchases[10] = 2;
    bad_game.imperium_purchase_rounds[10][1] = 1;  // sum = 1 != 2

    bool caught = false;
    try {
      MarketDiagnosticsReport::AggregateReport({bad_game}, 1);
    } catch (const std::exception& e) {
      caught = true;
    }
    SPIEL_CHECK_TRUE(caught);
  }

  // Test 4: Duplicate episode ID
  {
    CompactPerGameMarketDiagnostics g1, g2;
    g1.episode_id = 42;
    g1.candidate_seat = 0;
    g2.episode_id = 42;  // Duplicate!
    g2.candidate_seat = 1;

    bool caught = false;
    try {
      MarketDiagnosticsReport::AggregateReport({g1, g2}, 2);
    } catch (const std::exception& e) {
      caught = true;
    }
    SPIEL_CHECK_TRUE(caught);
  }

  // Test 5: Negative episode ID
  {
    CompactPerGameMarketDiagnostics g;
    g.episode_id = -1;
    g.candidate_seat = 0;

    bool caught = false;
    try {
      MarketDiagnosticsReport::AggregateReport({g}, 1);
    } catch (const std::exception& e) {
      caught = true;
    }
    SPIEL_CHECK_TRUE(caught);
  }

  open_spiel::SetErrorHandler(ExitingErrorHandler);
  std::cout << "TestFailClosedValidation PASSED.\n";
}

// ---------------------------------------------------------------------------
// 9. Per-Game Serialization and Exact Reconciliation Test
// ---------------------------------------------------------------------------
void TestPerGameSerializationAndReconciliation() {
  std::cout << "Running TestPerGameSerializationAndReconciliation...\n";

  std::vector<CompactPerGameMarketDiagnostics> games;
  for (int ep = 0; ep < 20; ++ep) {
    CompactPerGameMarketDiagnostics g;
    g.episode_id = ep;
    g.candidate_seat = ep % 4;

    // Set some imperium cards
    for (int i = 0; i < 5; ++i) {
      g.imperium_visible.set(i);
      g.imperium_had_legal.set(i);
      g.imperium_legal_opps[i] = ep + 1;
      if (ep % 2 == 0) {
        g.imperium_purchased.set(i);
        g.imperium_purchases[i] = 1;
        g.imperium_purchase_rounds[i][(ep % 5) + 1] = 1;
      }
    }

    // Set some Tleilaxu cards
    g.tleilaxu_visible.set(3);
    g.tleilaxu_had_legal.set(3);
    g.tleilaxu_legal_opps[3] = 2;
    if (ep % 3 == 0) {
      g.tleilaxu_purchased.set(3);
      g.tleilaxu_purchases[3] = 1;
      g.tleilaxu_purchase_rounds[3][2] = 1;
    }

    // Set some Tech tiles
    g.tech_visible.set(1);
    g.tech_had_legal.set(1);
    g.tech_legal_opps[1] = 1;

    // Fixed supplies
    g.arrakis_liaison.visible = true;
    g.arrakis_liaison.had_legal = true;
    g.arrakis_liaison.legal_opps = 3;
    g.arrakis_liaison.purchased = true;
    g.arrakis_liaison.purchases = 1;
    g.arrakis_liaison.purchase_rounds[1] = 1;

    g.reclaimed_forces.visible = true;
    g.reclaimed_forces.had_legal = true;
    g.reclaimed_forces.legal_opps = 1;

    // Special routes
    g.bypass_free_opps[2] = 1;
    g.family_atomics_uses = (ep == 0) ? 1 : 0;

    games.push_back(g);
  }

  const std::string tmp_path = "/tmp/test_market_diagnostics_games.jsonl";
  SPIEL_CHECK_TRUE(WritePerGameJsonl(tmp_path, games));

  auto read_games = ReadPerGameJsonl(tmp_path);
  SPIEL_CHECK_EQ(read_games.size(), games.size());

  auto orig_report = MarketDiagnosticsReport::AggregateReport(games, games.size());
  auto read_report = MarketDiagnosticsReport::AggregateReport(read_games, read_games.size());

  // Must verify reconciliation without error
  VerifyReconciliation(orig_report, read_report);

  // Tampering test: verify that reconciliation fails if data is altered
  open_spiel::SetErrorHandler(ThrowingErrorHandler);
  read_games[0].imperium_legal_opps[0] += 5;
  auto tampered_report = MarketDiagnosticsReport::AggregateReport(read_games, read_games.size());
  bool caught_reconciliation_failure = false;
  try {
    VerifyReconciliation(orig_report, tampered_report);
  } catch (const std::exception& e) {
    caught_reconciliation_failure = true;
  }
  open_spiel::SetErrorHandler(ExitingErrorHandler);
  SPIEL_CHECK_TRUE(caught_reconciliation_failure);

  std::filesystem::remove(tmp_path);
  std::cout << "TestPerGameSerializationAndReconciliation PASSED.\n";
}

// ---------------------------------------------------------------------------
// 10. Non-Interference: Bitwise Identical States / Observations With Telem OFF vs ON
// ---------------------------------------------------------------------------
void TestNonInterferenceBitwiseIdentical() {
  std::cout << "Running TestNonInterferenceBitwiseIdentical...\n";
  std::shared_ptr<const Game> game = LoadGame("dune_imperium");

  // Run 1: Telem OFF
  std::unique_ptr<State> state1 = game->NewInitialState();
  std::vector<Action> actions_taken;
  std::mt19937 rng1(12345);

  for (int step = 0; step < 80 && !state1->IsTerminal(); ++step) {
    if (state1->IsChanceNode()) {
      auto outcomes = state1->ChanceOutcomes();
      state1->ApplyAction(outcomes.front().first);
    } else {
      auto legal = state1->LegalActions();
      std::uniform_int_distribution<size_t> dist(0, legal.size() - 1);
      Action a = legal[dist(rng1)];
      actions_taken.push_back(a);
      state1->ApplyAction(a);
    }
  }

  // Run 2: Telem ON with exact same actions
  std::unique_ptr<State> state2 = game->NewInitialState();
  const auto* dune_state2 = dynamic_cast<const DuneImperiumState*>(state2.get());
  GameMarketTracker tracker;
  tracker.InitGame(/*episode_id=*/7, /*candidate_seat=*/0);

  size_t act_idx = 0;
  for (int step = 0; step < 80 && !state2->IsTerminal(); ++step) {
    if (state2->IsChanceNode()) {
      auto outcomes = state2->ChanceOutcomes();
      state2->ApplyAction(outcomes.front().first);
    } else {
      Action a = actions_taken[act_idx++];
      Player current_player = state2->CurrentPlayer();
      auto legal = state2->LegalActions();
      if (current_player == 0) {
        tracker.OnCandidateDecision(*dune_state2, 0, legal);
      }
      tracker.BeforeApplyAction(*dune_state2, current_player, 0, a);
      state2->ApplyAction(a);
      tracker.AfterApplyAction(*dune_state2, current_player, 0, a);
    }
  }

  // Verify bitwise state equality
  SPIEL_CHECK_EQ(state1->ToString(), state2->ToString());
  SPIEL_CHECK_EQ(state1->Returns(), state2->Returns());

  std::vector<float> obs1(game->ObservationTensorSize());
  std::vector<float> obs2(game->ObservationTensorSize());
  for (int p = 0; p < 4; ++p) {
    state1->ObservationTensor(p, absl::MakeSpan(obs1));
    state2->ObservationTensor(p, absl::MakeSpan(obs2));
    SPIEL_CHECK_EQ(obs1, obs2);
  }

  std::cout << "TestNonInterferenceBitwiseIdentical PASSED.\n";
}

// ---------------------------------------------------------------------------
// 11. Invariants Check: games_purchased <= games_legal_opportunity <= games_visible
// ---------------------------------------------------------------------------
void TestInvariantValidation() {
  std::cout << "Running TestInvariantValidation...\n";
  std::vector<CompactPerGameMarketDiagnostics> all_games;

  for (int ep = 0; ep < 100; ++ep) {
    CompactPerGameMarketDiagnostics g;
    g.episode_id = ep;
    g.candidate_seat = ep % 4;

    // Simulate visible, legal, and purchased
    for (int i = 0; i < kNumImperiumCards; ++i) {
      if (ep % 2 == 0) g.imperium_visible.set(i);
      if (ep % 4 == 0) {
        g.imperium_had_legal.set(i);
        g.imperium_legal_opps[i] = 2;
      }
      if (ep % 8 == 0) {
        g.imperium_purchased.set(i);
        g.imperium_purchases[i] = 1;
        g.imperium_purchase_rounds[i][1] = 1;
      }
    }
    all_games.push_back(g);
  }

  auto report = MarketDiagnosticsReport::AggregateReport(all_games, 100);
  for (const auto& item : report.GetImperiumItems()) {
    SPIEL_CHECK_LE(item.games_purchased, item.games_legal_opportunity);
    SPIEL_CHECK_LE(item.games_legal_opportunity, item.games_visible);
    SPIEL_CHECK_LE(item.games_visible, 100);
    if (item.games_legal_opportunity > 0) {
      SPIEL_CHECK_GE(item.conversion_rate, 0.0);
      SPIEL_CHECK_LE(item.conversion_rate, 1.0);
    } else {
      SPIEL_CHECK_EQ(item.conversion_rate, -1.0);
    }
  }

  std::cout << "TestInvariantValidation PASSED.\n";
}

}  // namespace
}  // namespace dune_market_diag
}  // namespace open_spiel

int main(int argc, char** argv) {
  open_spiel::dune_market_diag::TestTechPurchaseRoutes();
  open_spiel::dune_market_diag::TestAppropriateTwoPhaseResolution();
  open_spiel::dune_market_diag::TestDelayedAcquisitionsAndTopdeckPlacement();
  open_spiel::dune_market_diag::TestTleilaxuMasterDeferredHandPlacement();
  open_spiel::dune_market_diag::TestReclaimedForcesPaidOption();
  open_spiel::dune_market_diag::TestBypassProtocolBranches();
  open_spiel::dune_market_diag::TestHarvestCellsAcquisitions();
  open_spiel::dune_market_diag::TestFailClosedValidation();
  open_spiel::dune_market_diag::TestPerGameSerializationAndReconciliation();
  open_spiel::dune_market_diag::TestNonInterferenceBitwiseIdentical();
  open_spiel::dune_market_diag::TestInvariantValidation();

  std::cout << "ALL 11 MARKET DIAGNOSTICS UNIT TESTS PASSED!\n";
  return 0;
}
