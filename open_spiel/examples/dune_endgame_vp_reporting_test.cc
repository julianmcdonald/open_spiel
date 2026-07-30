// Regression tests for the Memocorders endgame-VP reporting defect.
//
// THE DEFECT. `dune_pwo4_trajectory.cc` and `dune_search_benchmark.cc` each
// carried a hand-written copy of the engine's endgame-VP arithmetic, named
// GetTrueFinalVp. Both copies awarded the "all four factions at influence >= 3"
// +1 UNCONDITIONALLY. The engine awards it only to a seat that owns TECH TILE 8
// (Memocorders) -- `ComputeEndgameVp`, dune_imperium.cc:2303-2310. So the copies
// OVERSTATED by exactly 1 for any seat at >= 3 influence in all four factions
// without Memocorders.
//
// SCOPE. This was a REPORTING defect, never a reward defect. `Returns()` ranks
// placements on `vp_[p] + ComputeEndgameVp(p)` (dune_imperium.cc:2357) and
// `FinalScoredVp` returns that same expression (:2403), so every placement,
// every reward and every committed `returns` array was always correct. What was
// wrong was the reported `final_vp` / `vp_margin` / `search_vp` /
// `opponent_vps` telemetry.
//
// THE FIX. Both reporters now route through DuneImperiumState::FinalScoredVp.
// See docs/PWO5_AMENDMENT_1_TARGET_EXPOSURE_TELEMETRY_2026_07_31.md section 10.
//
// WHY THIS FILE LIVES IN examples/ AND NOT IN games/dune_imperium/.
// `git rev-parse HEAD:games/dune_imperium` is a PINNED REGISTERED QUANTITY --
// registration sections 0.1, 5.2 and gate 4's standing preflight all assert the
// engine tree hash `a67f1925305d35d9a2f30fd9658b9f01fe21bdd6`, and every PWO-5
// arm asserts it at launch. Adding a file to that directory would change the
// tree hash and trip STOP condition 9 (engine-freeze check fails). These tests
// therefore use only the engine's PUBLIC api plus its existing ForTesting
// accessors, and add nothing to the frozen tree.
//
// The four cases are the ones the principal's instruction registers:
//   1. all four factions >= 3 WITHOUT Memocorders -> no bonus;
//   2. all four factions >= 3 WITH Memocorders    -> exactly +1;
//   3. reporter output == FinalScoredVp for every seat;
//   4. Returns()/placement scoring unchanged and consistent.

#include <algorithm>
#include <iostream>
#include <memory>
#include <random>
#include <vector>

#include "open_spiel/games/dune_imperium/dune_imperium.h"
#include "open_spiel/games/dune_imperium/dune_imperium_cards.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"

// THE PRODUCTION DEFINITION ITSELF -- not a copy of it. Case 3 below would be
// worthless if this test reimplemented the reporter: a reverted binary would
// still pass. It calls the same inline function dune_pwo4_trajectory.cc and
// dune_search_benchmark.cc call.
#include "dune_terminal_vp_report.h"

namespace open_spiel {
namespace {

using dune_imperium::DuneImperiumState;
using dune_imperium::Faction;

constexpr int kMemocordersTechTile = 8;
constexpr int kNumSeats = 4;

// ---------------------------------------------------------------------------
// THE ARITHMETIC THAT WAS WRONG, reproduced verbatim.
// ---------------------------------------------------------------------------
// This is the removed GetTrueFinalVp, kept HERE and ONLY here so the tests can
// demonstrate the defect rather than merely assert its absence. A regression
// test that only checks the fixed path is a test that cannot fail if the fix is
// silently reverted to something else that happens to agree on the sampled
// states; this one reproduces the bug and pins the exact disagreement.
//
// It is not a scoring function. Nothing outside this file may call it.
int LegacyReportedVp(const DuneImperiumState* s, int p) {
  int endgame_vp = 0;

  for (int intrigue_id : s->GetIntrigueHandForTesting(p)) {
    const auto* intrigue = dune_imperium::FindIntrigueCardById(intrigue_id);
    if (intrigue &&
        (intrigue->phase_mask & dune_imperium::kIntriguePhaseEndgameMask) != 0) {
      endgame_vp += s->EndgameIntrigueVpBonusForTesting(p, intrigue_id);
    }
  }

  const std::vector<int> tech_tiles = s->GetPlayerTechTilesForTesting(p);
  if (std::find(tech_tiles.begin(), tech_tiles.end(), 6) != tech_tiles.end()) {
    int tsmf_count = 0;
    auto count_tsmf = [&](const std::vector<int>& cards) {
      for (int id : cards) {
        if (id == dune_imperium::kCardTheSpiceMustFlow) tsmf_count++;
      }
    };
    auto* mutable_state = const_cast<DuneImperiumState*>(s);
    count_tsmf(mutable_state->GetPlayerDrawDeckForTesting(p));
    count_tsmf(mutable_state->GetPlayerDiscardForTesting(p));
    count_tsmf(mutable_state->GetPlayerHandForTesting(p));
    count_tsmf(s->GetPlayedAgentCardsForTesting(p));
    count_tsmf(s->GetRevealedCardsForTesting(p));
    if (tsmf_count >= 2) endgame_vp += 1;
  }

  // <<< THE DEFECT: no tech-tile-8 guard. The engine has one here. >>>
  bool all_3 = true;
  for (int f = 0; f < dune_imperium::kNumFactions; ++f) {
    if (s->GetPlayerInfluenceForTesting(p, static_cast<Faction>(f)) < 3) {
      all_3 = false;
    }
  }
  if (all_3) endgame_vp += 1;

  if (std::find(tech_tiles.begin(), tech_tiles.end(), 14) != tech_tiles.end()) {
    int low = 0;
    for (int f = 0; f < dune_imperium::kNumFactions; ++f) {
      if (s->GetPlayerInfluenceForTesting(p, static_cast<Faction>(f)) <= 1) {
        low++;
      }
    }
    endgame_vp += low;
  }
  return s->GetPlayerVpForTesting(p) + endgame_vp;
}

// The reporter THE BINARIES ACTUALLY CALL. Both dune_pwo4_trajectory.cc's and
// dune_search_benchmark.cc's GetTrueFinalVp now forward to this exact inline
// function, so case 3 exercises the shipped path rather than a restatement of
// it.
int ReporterFinalVp(const DuneImperiumState* s, int p) {
  return dune_report::TerminalVpForReporting(s, p);
}

// ---------------------------------------------------------------------------
// A terminal state.
// ---------------------------------------------------------------------------
// FinalScoredVp asserts IsTerminal(), so every case needs a real terminal
// state. Random legal play reaches one quickly: the engine hard-caps the game
// at kMaxRounds = 10.
std::unique_ptr<State> PlayToTerminal(int seed) {
  std::shared_ptr<const Game> game = LoadGame("dune_imperium");
  std::unique_ptr<State> state = game->NewInitialState();
  std::mt19937 rng(seed);
  int guard = 0;
  while (!state->IsTerminal()) {
    SPIEL_CHECK_LT(++guard, 200000);  // the engine caps rounds; this catches a hang
    if (state->IsChanceNode()) {
      std::vector<std::pair<Action, double>> outcomes = state->ChanceOutcomes();
      SPIEL_CHECK_FALSE(outcomes.empty());
      double roll = std::uniform_real_distribution<double>(0.0, 1.0)(rng);
      double acc = 0.0;
      Action chosen = outcomes.back().first;
      for (const auto& [action, prob] : outcomes) {
        acc += prob;
        if (roll <= acc) { chosen = action; break; }
      }
      state->ApplyAction(chosen);
    } else {
      std::vector<Action> legal = state->LegalActions();
      SPIEL_CHECK_FALSE(legal.empty());
      state->ApplyAction(
          legal[std::uniform_int_distribution<size_t>(0, legal.size() - 1)(rng)]);
    }
  }
  return state;
}

// Force a seat into the "all four factions at >= 3" configuration and set its
// Memocorders ownership, leaving every other endgame input untouched. Returns
// the seat's FinalScoredVp under that configuration.
int ScoreWithAllFactionsAtThree(DuneImperiumState* s, int seat,
                                bool owns_memocorders) {
  for (int f = 0; f < dune_imperium::kNumFactions; ++f) {
    s->SetPlayerInfluenceForTesting(seat, static_cast<Faction>(f), 3);
  }
  std::vector<int> tiles = s->GetPlayerTechTilesForTesting(seat);
  tiles.erase(std::remove(tiles.begin(), tiles.end(), kMemocordersTechTile),
              tiles.end());
  if (owns_memocorders) tiles.push_back(kMemocordersTechTile);
  s->SetTechTilesOwnedForTesting(seat, tiles);
  return s->FinalScoredVp(seat);
}

// ---------------------------------------------------------------------------
// 1 + 2. The Memocorders semantics.
// ---------------------------------------------------------------------------
void MemocordersGatesTheAllFactionsBonusTest() {
  std::unique_ptr<State> state = PlayToTerminal(20260800);
  auto* s = static_cast<DuneImperiumState*>(state.get());
  SPIEL_CHECK_TRUE(s->IsTerminal());

  for (int seat = 0; seat < kNumSeats; ++seat) {
    // Case 1: all four factions >= 3, NO Memocorders -> no bonus.
    const int without = ScoreWithAllFactionsAtThree(s, seat, false);
    // Case 2: identical state, Memocorders added -> exactly +1.
    const int with = ScoreWithAllFactionsAtThree(s, seat, true);

    SPIEL_CHECK_EQ(with - without, 1);

    // And the bonus really is gated on the influence condition, not merely on
    // owning the tile: drop one faction to 2 while KEEPING Memocorders and the
    // +1 must disappear.
    // Dropping Fremen 3 -> 2 crosses no other endgame clause (tech tile 14
    // counts factions at <= 1), so the ONLY change is the lost bonus.
    s->SetPlayerInfluenceForTesting(seat, Faction::kFremen, 2);
    SPIEL_CHECK_EQ(s->FinalScoredVp(seat), without);

    // Restore the all-three configuration for the legacy comparison below.
    const int restored = ScoreWithAllFactionsAtThree(s, seat, false);
    SPIEL_CHECK_EQ(restored, without);

    // THE DEFECT, DEMONSTRATED. Without Memocorders the legacy arithmetic
    // awards the bonus anyway, so it overstates by exactly 1 -- this is the
    // 212/1,600 divergence in the frozen PWO-4 corpus, reproduced on demand.
    SPIEL_CHECK_EQ(LegacyReportedVp(s, seat), without + 1);
    SPIEL_CHECK_NE(LegacyReportedVp(s, seat), s->FinalScoredVp(seat));

    // With Memocorders the two agree, which is why the defect was invisible on
    // most seats.
    ScoreWithAllFactionsAtThree(s, seat, true);
    SPIEL_CHECK_EQ(LegacyReportedVp(s, seat), s->FinalScoredVp(seat));
  }
  std::cout << "  PASS MemocordersGatesTheAllFactionsBonus\n";
}

// ---------------------------------------------------------------------------
// 3. Reporter output == FinalScoredVp for every seat.
// ---------------------------------------------------------------------------
void ReporterMatchesFinalScoredVpTest() {
  // Several independent terminal states, so the check is not a property of one
  // lucky game.
  for (int seed : {20260801, 20260802, 20260810, 20260811, 20260812}) {
    std::unique_ptr<State> state = PlayToTerminal(seed);
    auto* s = static_cast<DuneImperiumState*>(state.get());
    SPIEL_CHECK_TRUE(s->IsTerminal());
    for (int seat = 0; seat < kNumSeats; ++seat) {
      SPIEL_CHECK_EQ(ReporterFinalVp(s, seat), s->FinalScoredVp(seat));
    }

    // NATURAL STATES ARE NOT ENOUGH, and saying so is the point. The defect
    // only manifests on a seat at >= 3 influence in all four factions WITHOUT
    // Memocorders; if no sampled game happens to produce one, the loop above
    // passes against a reverted reporter and proves nothing. So the condition
    // is CONSTRUCTED on every seat, and the reporter is checked there.
    for (int seat = 0; seat < kNumSeats; ++seat) {
      const int engine_score = ScoreWithAllFactionsAtThree(s, seat, false);
      SPIEL_CHECK_EQ(ReporterFinalVp(s, seat), engine_score);
      // And this is precisely where the old arithmetic disagreed, so the check
      // above has teeth rather than being an identity.
      SPIEL_CHECK_EQ(LegacyReportedVp(s, seat), engine_score + 1);
      SPIEL_CHECK_NE(ReporterFinalVp(s, seat), LegacyReportedVp(s, seat));
    }
  }
  std::cout << "  PASS ReporterMatchesFinalScoredVp\n";
}

// ---------------------------------------------------------------------------
// 4. Returns()/placement scoring unchanged and consistent.
// ---------------------------------------------------------------------------
void ReturnsRemainConsistentWithFinalScoredVpTest() {
  for (int seed : {20260801, 20260802, 20260810, 20260811, 20260812}) {
    std::unique_ptr<State> state = PlayToTerminal(seed);
    auto* s = static_cast<DuneImperiumState*>(state.get());
    SPIEL_CHECK_TRUE(s->IsTerminal());

    const std::vector<double> returns = s->Returns();
    SPIEL_CHECK_EQ(static_cast<int>(returns.size()), kNumSeats);

    // Returns() sorts on `vp_[p] + ComputeEndgameVp(p)` FIRST
    // (dune_imperium.cc:2352-2372), which is exactly FinalScoredVp. So a strict
    // VP advantage must produce a strictly better placement payout; ties are
    // broken by other keys and carry no constraint here.
    for (int a = 0; a < kNumSeats; ++a) {
      for (int b = 0; b < kNumSeats; ++b) {
        if (a == b) continue;
        if (s->FinalScoredVp(a) > s->FinalScoredVp(b)) {
          SPIEL_CHECK_GT(returns[a], returns[b]);
        }
      }
    }

    // Zero-sum placement payout is unchanged by the reporting fix.
    double sum = 0.0;
    for (double r : returns) sum += r;
    SPIEL_CHECK_FLOAT_NEAR(sum, 0.0, 1e-9);
  }
  std::cout << "  PASS ReturnsRemainConsistentWithFinalScoredVp\n";
}

// ---------------------------------------------------------------------------
// The bonus is worth exactly 1 VP and nothing else moves.
// ---------------------------------------------------------------------------
// Guards the other direction: that routing through the engine did not silently
// change any OTHER endgame clause. On a seat whose influence is NOT all-at-3,
// the two arithmetics must agree exactly, tile 8 or no tile 8 -- so any
// disagreement outside the Memocorders clause fails here.
void OnlyTheMemocordersClauseEverDiffersTest() {
  for (int seed : {20260801, 20260802, 20260810, 20260811, 20260812}) {
    std::unique_ptr<State> state = PlayToTerminal(seed);
    auto* s = static_cast<DuneImperiumState*>(state.get());
    for (int seat = 0; seat < kNumSeats; ++seat) {
      const bool all_three = [&] {
        for (int f = 0; f < dune_imperium::kNumFactions; ++f) {
          if (s->GetPlayerInfluenceForTesting(seat, static_cast<Faction>(f)) < 3)
            return false;
        }
        return true;
      }();
      const std::vector<int> tiles = s->GetPlayerTechTilesForTesting(seat);
      const bool has_memocorders =
          std::find(tiles.begin(), tiles.end(), kMemocordersTechTile) !=
          tiles.end();

      const int engine = s->FinalScoredVp(seat);
      const int legacy = LegacyReportedVp(s, seat);
      const int expected_gap = (all_three && !has_memocorders) ? 1 : 0;
      SPIEL_CHECK_EQ(legacy - engine, expected_gap);
    }
  }
  std::cout << "  PASS OnlyTheMemocordersClauseEverDiffers\n";
}

}  // namespace
}  // namespace open_spiel

int main(int argc, char** argv) {
  std::cout << "dune_endgame_vp_reporting_test\n";
  open_spiel::MemocordersGatesTheAllFactionsBonusTest();
  open_spiel::ReporterMatchesFinalScoredVpTest();
  open_spiel::ReturnsRemainConsistentWithFinalScoredVpTest();
  open_spiel::OnlyTheMemocordersClauseEverDiffersTest();
  std::cout << "ALL PASS\n";
  return 0;
}
