#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <random>

#include "open_spiel/spiel_utils.h"
#include "open_spiel/games/dune_imperium/dune_imperium.h"
#include "open_spiel/games/dune_imperium/dune_imperium_cards.h"
#include "open_spiel/games/dune_imperium/dune_imperium_content.h"
#include "open_spiel/games/dune_imperium/dune_imperium_test_utils.h"
#include "dune_targeted_buy.h"

namespace open_spiel {
namespace dune_targeted_buy {
namespace {

int failures = 0;

void Check(bool condition, const std::string& description) {
  if (condition) {
    std::cout << "  PASS: " << description << std::endl;
  } else {
    std::cout << "  FAIL: " << description << std::endl;
    failures++;
  }
}

// 1. Card Identity Tests
void TestCardIdentity() {
  std::cout << "\n=== Test 1: Card Identity ===" << std::endl;
  using open_spiel::dune_imperium::kTleilaxuCards;
  Check(kTleilaxuCards[kScientificBreakthroughTleilaxuId].imperium_card_id == kScientificBreakthroughCardId,
        "Scientific Breakthrough Tleilaxu ID 11 maps to Imperium Card ID 118");
  Check(kTleilaxuCards[kStitchedHorrorTleilaxuId].imperium_card_id == kStitchedHorrorCardId,
        "Stitched Horror Tleilaxu ID 13 maps to Imperium Card ID 120");
  Check(kTleilaxuCards[kScientificBreakthroughTleilaxuId].specimen_cost == 3,
        "Scientific Breakthrough costs 3 specimens");
  Check(kTleilaxuCards[kStitchedHorrorTleilaxuId].specimen_cost == 3,
        "Stitched Horror costs 3 specimens");

  Check(dune_imperium::kActionTleilaxuAcquire0 + 1 == 91, "Slot 0 acquire action ID is 91");
  Check(dune_imperium::kActionTleilaxuAcquire0 + 2 == 92, "Slot 1 acquire action ID is 92");
}

// 2. Legality Tests
void TestLegality() {
  std::cout << "\n=== Test 2: Direct Action Legality ===" << std::endl;
  auto state = dune_imperium::BuildStateAtFirstRevealTurn("dune_imperium(enable_immortality=true)");
  auto* impl = static_cast<dune_imperium::DuneImperiumState*>(state.get());
  const Player p = impl->CurrentPlayer();

  // Put SB in slot 0, SH in slot 1
  impl->SetTleilaxuRowForTesting({kScientificBreakthroughTleilaxuId, kStitchedHorrorTleilaxuId});

  // Insufficient specimens: 2 specimens (costs 3)
  impl->SetSpecimensForTesting(p, 2);
  auto legal = impl->LegalActions();

  int slot = -1;
  Action act = kInvalidAction;
  Check(!FindLegalTleilaxuBuyAction(*impl, legal, kScientificBreakthroughTleilaxuId, &slot, &act),
        "Scientific Breakthrough cannot be acquired with 2 specimens (illegal)");
  Check(!FindLegalTleilaxuBuyAction(*impl, legal, kStitchedHorrorTleilaxuId, &slot, &act),
        "Stitched Horror cannot be acquired with 2 specimens (illegal)");

  // Sufficient specimens: 3 specimens
  impl->SetSpecimensForTesting(p, 3);
  legal = impl->LegalActions();
  Check(FindLegalTleilaxuBuyAction(*impl, legal, kScientificBreakthroughTleilaxuId, &slot, &act) &&
        slot == 0 && act == 91,
        "Scientific Breakthrough in slot 0 is legal with 3 specimens (action 91)");
  Check(FindLegalTleilaxuBuyAction(*impl, legal, kStitchedHorrorTleilaxuId, &slot, &act) &&
        slot == 1 && act == 92,
        "Stitched Horror in slot 1 is legal with 3 specimens (action 92)");
}

// 3. Forcing Enforcement: Rejected Purchases on Actual Game States
void TestForcingRejections() {
  std::cout << "\n=== Test 3: Forcing Enforcement - Rejected Purchases ===" << std::endl;
  auto state = dune_imperium::BuildStateAtFirstRevealTurn("dune_imperium(enable_immortality=true)");
  auto* impl = static_cast<dune_imperium::DuneImperiumState*>(state.get());
  const Player p = impl->CurrentPlayer();
  std::mt19937 rng(42);

  // Set SB in slot 0, SH in slot 1, 3 specimens
  impl->SetTleilaxuRowForTesting({kScientificBreakthroughTleilaxuId, kStitchedHorrorTleilaxuId});
  impl->SetSpecimensForTesting(p, 3);
  auto legal = impl->LegalActions();
  std::vector<double> probs(legal.size(), 1.0 / legal.size());

  // 3a: Exploration disabled
  {
    TargetedBuyTracker tracker;
    tracker.Reset(false, 8, 8, 6);
    bool ep_forced_11 = false, ep_forced_13 = false;
    auto dec = DecideTargetedForcedBuy(
        tracker, *impl, p, /*designated_seat=*/p, /*round=*/3,
        legal, probs, ep_forced_11, ep_forced_13, rng);
    Check(!dec.forced, "Disabled tracker rejects forcing");
    Check(dec.rejection_reason == "disabled", "Rejection reason is 'disabled'");
  }

  // 3b: Non-designated seat
  {
    TargetedBuyTracker tracker;
    tracker.Reset(true, 8, 8, 6);
    bool ep_forced_11 = false, ep_forced_13 = false;
    int other_seat = (p + 1) % 4;
    auto dec = DecideTargetedForcedBuy(
        tracker, *impl, p, /*designated_seat=*/other_seat, /*round=*/3,
        legal, probs, ep_forced_11, ep_forced_13, rng);
    Check(!dec.forced, "Non-designated seat rejects forcing");
    Check(dec.rejection_reason == "seat_not_designated", "Rejection reason is 'seat_not_designated'");
  }

  // 3c: Ineligible round (round 7 > max_round 6)
  {
    TargetedBuyTracker tracker;
    tracker.Reset(true, 8, 8, 6);
    bool ep_forced_11 = false, ep_forced_13 = false;
    auto dec = DecideTargetedForcedBuy(
        tracker, *impl, p, /*designated_seat=*/p, /*round=*/7,
        legal, probs, ep_forced_11, ep_forced_13, rng);
    Check(!dec.forced, "Round 7 (> max_round 6) rejects forcing");
    Check(dec.rejection_reason == "round_exceeds_max", "Rejection reason is 'round_exceeds_max'");
  }

  // 3d: Insufficient specimens on game state
  {
    impl->SetSpecimensForTesting(p, 2);
    auto legal_2spec = impl->LegalActions();
    std::vector<double> probs_2spec(legal_2spec.size(), 1.0 / legal_2spec.size());
    TargetedBuyTracker tracker;
    tracker.Reset(true, 8, 8, 6);
    bool ep_forced_11 = false, ep_forced_13 = false;
    auto dec = DecideTargetedForcedBuy(
        tracker, *impl, p, /*designated_seat=*/p, /*round=*/3,
        legal_2spec, probs_2spec, ep_forced_11, ep_forced_13, rng);
    Check(!dec.forced, "Insufficient specimens (2) rejects forcing");
    Check(dec.rejection_reason == "neither_card_legal", "Rejection reason is 'neither_card_legal'");
    // Restore specimens
    impl->SetSpecimensForTesting(p, 3);
  }

  // 3e: Target cards missing from Tleilaxu row
  {
    impl->SetTleilaxuRowForTesting({1, 2});  // Neither 11 nor 13
    auto legal_other = impl->LegalActions();
    std::vector<double> probs_other(legal_other.size(), 1.0 / legal_other.size());
    TargetedBuyTracker tracker;
    tracker.Reset(true, 8, 8, 6);
    bool ep_forced_11 = false, ep_forced_13 = false;
    auto dec = DecideTargetedForcedBuy(
        tracker, *impl, p, /*designated_seat=*/p, /*round=*/3,
        legal_other, probs_other, ep_forced_11, ep_forced_13, rng);
    Check(!dec.forced, "Absence of target cards from row rejects forcing");
    Check(dec.rejection_reason == "neither_card_legal", "Rejection reason is 'neither_card_legal'");
    // Restore row
    impl->SetTleilaxuRowForTesting({kScientificBreakthroughTleilaxuId, kStitchedHorrorTleilaxuId});
  }
}

// 4. Forcing Enforcement: Successful Decisions & Biased Log-Prob Computation
void TestForcingDecisionsAndBiasedLogProb() {
  std::cout << "\n=== Test 4: Forcing Enforcement - Successful Decisions & Policy Log-Prob ===" << std::endl;
  auto state = dune_imperium::BuildStateAtFirstRevealTurn("dune_imperium(enable_immortality=true)");
  auto* impl = static_cast<dune_imperium::DuneImperiumState*>(state.get());
  const Player p = impl->CurrentPlayer();
  std::mt19937 rng(12345);

  impl->SetTleilaxuRowForTesting({kScientificBreakthroughTleilaxuId, kStitchedHorrorTleilaxuId});
  impl->SetSpecimensForTesting(p, 3);
  auto legal = impl->LegalActions();

  // Create known policy probability distribution over legal actions
  std::vector<double> probs(legal.size(), 0.0);
  double expected_prob_11 = 0.05;
  double expected_prob_13 = 0.02;
  for (size_t i = 0; i < legal.size(); ++i) {
    if (legal[i] == 91) probs[i] = expected_prob_11;
    else if (legal[i] == 92) probs[i] = expected_prob_13;
    else probs[i] = (1.0 - expected_prob_11 - expected_prob_13) / (legal.size() - 2);
  }

  TargetedBuyTracker tracker;
  tracker.Reset(true, 8, 8, 6);
  bool ep_forced_11 = false, ep_forced_13 = false;

  auto dec = DecideTargetedForcedBuy(
      tracker, *impl, p, /*designated_seat=*/p, /*round=*/4,
      legal, probs, ep_forced_11, ep_forced_13, rng);

  Check(dec.forced, "Targeted buy was successfully forced");
  Check(dec.forced_action == 91 || dec.forced_action == 92, "Forced action is legal Tleilaxu acquire action (91 or 92)");
  if (dec.forced_action == 91) {
    Check(dec.forced_card_tleilaxu_id == kScientificBreakthroughTleilaxuId, "Action 91 forced Scientific Breakthrough (ID 11)");
    Check(ep_forced_11 && !ep_forced_13, "Episode marked card 11 as forced");
    Check(std::abs(dec.old_log_prob - static_cast<float>(std::log(expected_prob_11))) < 1e-5f,
          "old_log_prob strictly matches policy log-prob log(0.05) for biased exploration");
    Check(tracker.achieved_games_card11.load() == 1, "Card 11 achieved quota incremented to 1");
  } else {
    Check(dec.forced_card_tleilaxu_id == kStitchedHorrorTleilaxuId, "Action 92 forced Stitched Horror (ID 13)");
    Check(ep_forced_13 && !ep_forced_11, "Episode marked card 13 as forced");
    Check(std::abs(dec.old_log_prob - static_cast<float>(std::log(expected_prob_13))) < 1e-5f,
          "old_log_prob strictly matches policy log-prob log(0.02) for biased exploration");
    Check(tracker.achieved_games_card13.load() == 1, "Card 13 achieved quota incremented to 1");
  }

  Check(tracker.designated_opps_card11.load() == 1, "Designated opportunities for card 11 incremented");
  Check(tracker.designated_opps_card13.load() == 1, "Designated opportunities for card 13 incremented");
}

// 5. Forcing Enforcement: Once-Per-Game Restriction
void TestOncePerGameEnforcement() {
  std::cout << "\n=== Test 5: Forcing Enforcement - Once-Per-Game Limit ===" << std::endl;
  auto state = dune_imperium::BuildStateAtFirstRevealTurn("dune_imperium(enable_immortality=true)");
  auto* impl = static_cast<dune_imperium::DuneImperiumState*>(state.get());
  const Player p = impl->CurrentPlayer();
  std::mt19937 rng(999);

  // Setup: only Card 11 in row, card 13 absent
  impl->SetTleilaxuRowForTesting({kScientificBreakthroughTleilaxuId, 1});
  impl->SetSpecimensForTesting(p, 3);
  auto legal = impl->LegalActions();
  std::vector<double> probs(legal.size(), 1.0 / legal.size());

  TargetedBuyTracker tracker;
  tracker.Reset(true, 8, 8, 6);
  bool ep_forced_11 = false, ep_forced_13 = false;

  // First forcing in episode: succeeds
  auto dec1 = DecideTargetedForcedBuy(
      tracker, *impl, p, p, /*round=*/2, legal, probs, ep_forced_11, ep_forced_13, rng);
  Check(dec1.forced && dec1.forced_card_tleilaxu_id == kScientificBreakthroughTleilaxuId,
        "First turn in episode forces Card 11");
  Check(ep_forced_11, "ep_forced_card11 is now true");

  // Second turn in same episode with only Card 11 available: MUST BE REJECTED
  auto dec2 = DecideTargetedForcedBuy(
      tracker, *impl, p, p, /*round=*/3, legal, probs, ep_forced_11, ep_forced_13, rng);
  Check(!dec2.forced, "Second attempt in same episode for Card 11 is rejected");
  Check(dec2.rejection_reason == "card11_already_forced_in_episode",
        "Rejection reason is 'card11_already_forced_in_episode'");

  // Third turn in same episode where Card 13 becomes available: Card 13 CAN be forced
  impl->SetTleilaxuRowForTesting({kScientificBreakthroughTleilaxuId, kStitchedHorrorTleilaxuId});
  auto legal_both = impl->LegalActions();
  std::vector<double> probs_both(legal_both.size(), 1.0 / legal_both.size());

  auto dec3 = DecideTargetedForcedBuy(
      tracker, *impl, p, p, /*round=*/4, legal_both, probs_both, ep_forced_11, ep_forced_13, rng);
  Check(dec3.forced && dec3.forced_card_tleilaxu_id == kStitchedHorrorTleilaxuId,
        "Card 13 can still be forced once even if Card 11 was already forced");
  Check(ep_forced_13, "ep_forced_card13 is now true");

  // Fourth turn in same episode where both are available: BOTH MUST BE REJECTED
  auto dec4 = DecideTargetedForcedBuy(
      tracker, *impl, p, p, /*round=*/5, legal_both, probs_both, ep_forced_11, ep_forced_13, rng);
  Check(!dec4.forced, "Fourth turn rejected because both cards have reached single-game limit");
  Check(dec4.rejection_reason == "both_cards_already_forced_in_episode",
        "Rejection reason is 'both_cards_already_forced_in_episode'");
}

// 6. Forcing Enforcement: Quota Exhaustion Across Distinct Games
void TestQuotaExhaustionEnforcement() {
  std::cout << "\n=== Test 6: Forcing Enforcement - Quota Exhaustion Across Games ===" << std::endl;
  auto state = dune_imperium::BuildStateAtFirstRevealTurn("dune_imperium(enable_immortality=true)");
  auto* impl = static_cast<dune_imperium::DuneImperiumState*>(state.get());
  const Player p = impl->CurrentPlayer();
  std::mt19937 rng(777);

  // Setup only Card 11 in row
  impl->SetTleilaxuRowForTesting({kScientificBreakthroughTleilaxuId, 1});
  impl->SetSpecimensForTesting(p, 3);
  auto legal = impl->LegalActions();
  std::vector<double> probs(legal.size(), 1.0 / legal.size());

  TargetedBuyTracker tracker;
  tracker.Reset(true, /*q11=*/8, /*q13=*/8, /*max_round=*/6);

  // Simulate 8 distinct games, each forcing Card 11 once
  int forced_count = 0;
  for (int game = 0; game < 8; ++game) {
    bool ep_forced_11 = false, ep_forced_13 = false;
    auto dec = DecideTargetedForcedBuy(
        tracker, *impl, p, p, /*round=*/2, legal, probs, ep_forced_11, ep_forced_13, rng);
    if (dec.forced && dec.forced_card_tleilaxu_id == kScientificBreakthroughTleilaxuId) {
      forced_count++;
    }
  }
  Check(forced_count == 8, "Successfully forced Card 11 across exactly 8 distinct games");
  Check(tracker.achieved_games_card11.load() == 8, "achieved_games_card11 counter is 8");

  // Game 9: designated seat, round 2, 3 specimens, fresh episode flags: MUST BE REJECTED (quota reached)
  bool ep_forced_11_game9 = false, ep_forced_13_game9 = false;
  auto dec9 = DecideTargetedForcedBuy(
      tracker, *impl, p, p, /*round=*/2, legal, probs, ep_forced_11_game9, ep_forced_13_game9, rng);
  Check(!dec9.forced, "Game 9 rejects forcing because Card 11 quota of 8 is exhausted");
  Check(dec9.rejection_reason == "quota_exhausted", "Rejection reason is 'quota_exhausted'");
  Check(tracker.achieved_games_card11.load() == 8, "Quota counter stays clamped at 8");
}

}  // namespace
}  // namespace dune_targeted_buy
}  // namespace open_spiel

int main() {
  open_spiel::dune_targeted_buy::TestCardIdentity();
  open_spiel::dune_targeted_buy::TestLegality();
  open_spiel::dune_targeted_buy::TestForcingRejections();
  open_spiel::dune_targeted_buy::TestForcingDecisionsAndBiasedLogProb();
  open_spiel::dune_targeted_buy::TestOncePerGameEnforcement();
  open_spiel::dune_targeted_buy::TestQuotaExhaustionEnforcement();

  if (open_spiel::dune_targeted_buy::failures == 0) {
    std::cout << "\nALL TARGETED BUY UNIT TESTS PASSED." << std::endl;
    return 0;
  } else {
    std::cerr << "\nUNIT TESTS FAILED: " << open_spiel::dune_targeted_buy::failures << " failure(s)." << std::endl;
    return 1;
  }
}
