// Invariant tests for the per-event VP attribution ledger.
//
// WHAT THE LEDGER IS FOR. The question "where did this seat's points come from"
// used to be answerable only by reconstructing it from terminal state, and that
// reconstruction is not sound. Of the ~22 live sites that grant VP, 12 leave no
// terminal trace whatsoever -- a plot intrigue is played and discarded to a
// SHARED pile, Spy Satellites ERASES the very tech tile that granted the point,
// and the pay-resources-for-VP cards stay in the deck and can fire on every
// shuffle cycle. Worse, the one existing accumulator overlaps the others:
// `cumulative_conflict_vp_delta_` snapshots vp_ at the top of
// ApplyConflictChoice and diffs at the bottom, AFTER its ExecuteInfluenceGain
// calls, so friendship and alliance VP earned during conflict resolution land
// inside the "conflict" number. Summing conflict + alliance + friendship
// therefore DOUBLE-COUNTS.
//
// So the ledger is not a nicety. It is the only construction that can answer
// the question at all.
//
// THE INVARIANT. Every mutation of vp_ goes through DuneImperiumState::AddVp,
// which records the delta it ACTUALLY applied. Therefore, for every seat, at
// every point in the game:
//
//     sum(ledger deltas) == vp_[seat]
//
// A non-zero residual does not mean the arithmetic is off. It means some site
// mutates vp_ without going through AddVp, and that site's points are invisible
// to attribution. That is the defect this file exists to catch.
//
// THE CLAMP. Two removal paths floor at zero (alliance loss, friendship loss).
// A ledger recording the NOMINAL -1 rather than the clamped 0 would drift from
// vp_ permanently the first time a seat at 0 VP lost an alliance. AddVp records
// the applied amount, and ClampedRemovalRecordsWhatWasApplied pins that.
//
// WHY THIS FILE LIVES IN examples/ AND NOT IN games/dune_imperium/.
// `git rev-parse HEAD:games/dune_imperium` is a pinned registered quantity
// asserted at launch by many registrations. Adding a file to that directory
// would change the tree hash. These tests use only the engine's public API and
// its existing ForTesting accessors.

#include <algorithm>
#include <iostream>
#include <map>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "open_spiel/games/dune_imperium/dune_imperium.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace {

using dune_imperium::DuneImperiumState;
using dune_imperium::VpSource;

int LedgerSum(const DuneImperiumState* state, Player p) {
  int sum = 0;
  for (const auto& ev : state->GetVpEvents(p)) sum += ev.delta;
  return sum;
}

std::unique_ptr<State> NewDuneState() {
  return LoadGame("dune_imperium")->NewInitialState();
}

// Plays `games` random playouts to terminal, asserting the invariant after
// EVERY action, not merely at the end -- a mid-game drift that happens to
// cancel out by terminal would otherwise pass.
void LedgerSumsToVpAtEveryStep() {
  std::cout << "Running LedgerSumsToVpAtEveryStep\n" << std::flush;
  constexpr int kGames = 12;
  int total_events = 0;
  std::map<std::string, int> sources_seen;

  for (int g = 0; g < kGames; ++g) {
    std::mt19937 rng(20260816 + g);
    auto state = NewDuneState();
    int steps = 0;
    while (!state->IsTerminal() && steps < 20000) {
      const auto* ds = static_cast<const DuneImperiumState*>(state.get());
      for (Player p = 0; p < 4; ++p) {
        SPIEL_CHECK_EQ(LedgerSum(ds, p), ds->GetPlayerVp(p));
      }
      std::vector<Action> legal = state->LegalActions();
      SPIEL_CHECK_FALSE(legal.empty());
      state->ApplyAction(legal[std::uniform_int_distribution<size_t>(
          0, legal.size() - 1)(rng)]);
      ++steps;
    }

    const auto* ds = static_cast<const DuneImperiumState*>(state.get());
    for (Player p = 0; p < 4; ++p) {
      SPIEL_CHECK_EQ(LedgerSum(ds, p), ds->GetPlayerVp(p));
      // The ledger is never empty: the starting endowment is always recorded.
      SPIEL_CHECK_GE(ds->GetVpEvents(p).size(), 1u);
      // Compared as ints: VpSource is a scoped enum with no operator<<, which
      // the check macros need for their failure message.
      SPIEL_CHECK_EQ(static_cast<int>(ds->GetVpEvents(p)[0].source),
                     static_cast<int>(VpSource::kStartingEndowment));
      for (const auto& ev : ds->GetVpEvents(p)) {
        ++total_events;
        sources_seen[dune_imperium::VpSourceName(ev.source)] += 1;
        SPIEL_CHECK_NE(ev.delta, 0);  // zero-effect events are never recorded
        SPIEL_CHECK_GE(ev.round, 0);
      }
    }
  }
  std::cout << "  " << kGames << " playouts, " << total_events
            << " ledger events; sources observed:\n";
  for (const auto& kv : sources_seen) {
    std::cout << "    " << kv.first << " x" << kv.second << "\n";
  }
}

// The endowment must be in the ledger, or every downstream consumer has to
// know to add a magic 1 -- which is exactly the bug this replaced.
void StartingEndowmentIsRecorded() {
  std::cout << "Running StartingEndowmentIsRecorded\n" << std::flush;
  auto state = NewDuneState();
  const auto* ds = static_cast<const DuneImperiumState*>(state.get());
  for (Player p = 0; p < 4; ++p) {
    SPIEL_CHECK_EQ(ds->GetPlayerVp(p), dune_imperium::kStartingVp);
    SPIEL_CHECK_EQ(ds->GetVpEvents(p).size(), 1u);
    SPIEL_CHECK_EQ(ds->GetVpEvents(p)[0].delta, dune_imperium::kStartingVp);
    SPIEL_CHECK_EQ(static_cast<int>(ds->GetVpEvents(p)[0].source),
                   static_cast<int>(VpSource::kStartingEndowment));
    SPIEL_CHECK_EQ(LedgerSum(ds, p), ds->GetPlayerVp(p));
  }
}

// A removal against 0 VP is floored. The ledger must record what the floor
// allowed (nothing), not the nominal -1, or it drifts from vp_ forever.
void ClampedRemovalRecordsWhatWasApplied() {
  std::cout << "Running ClampedRemovalRecordsWhatWasApplied\n" << std::flush;
  auto state = NewDuneState();
  auto* ds = static_cast<DuneImperiumState*>(state.get());

  constexpr auto kEmp = dune_imperium::Faction::kEmperor;

  // A normal up/down cycle first: crossing 2 grants +1, dropping back removes
  // it, and the ledger tracks vp_ through both.
  ds->SetPlayerInfluenceForTesting(0, kEmp, 0);
  const int base_vp = ds->GetPlayerVp(0);
  ds->ExecuteInfluenceGainForTesting(0, kEmp, 2);
  SPIEL_CHECK_EQ(ds->GetPlayerVp(0), base_vp + 1);
  SPIEL_CHECK_EQ(LedgerSum(ds, 0), ds->GetPlayerVp(0));
  ds->ExecuteInfluenceLossForTesting(0, kEmp, 2);
  SPIEL_CHECK_EQ(ds->GetPlayerVp(0), base_vp);
  SPIEL_CHECK_EQ(LedgerSum(ds, 0), ds->GetPlayerVp(0));

  // Now the clamp itself. Put the seat back above the threshold, then force it
  // to 0 VP while it still HOLDS the friendship. Dropping the influence now
  // fires a -1 against a zero balance: the engine floors it, so nothing is
  // owed and nothing may be recorded. A ledger that wrote the nominal -1 here
  // would read -1 while vp_ read 0, and would stay wrong for the whole game.
  ds->ExecuteInfluenceGainForTesting(0, kEmp, 2);
  ds->SetPlayerVpForTesting(0, 0);
  SPIEL_CHECK_EQ(ds->GetPlayerVp(0), 0);
  SPIEL_CHECK_EQ(LedgerSum(ds, 0), 0);
  const size_t before_clamped_removal = ds->GetVpEvents(0).size();

  ds->ExecuteInfluenceLossForTesting(0, kEmp, 2);
  SPIEL_CHECK_EQ(ds->GetPlayerVp(0), 0);
  SPIEL_CHECK_EQ(LedgerSum(ds, 0), 0);
  SPIEL_CHECK_EQ(ds->GetVpEvents(0).size(), before_clamped_removal);
}

// The ledger describes the PATH, so it must never reach the transposition key:
// two identical positions reached by different routes must still share search
// statistics. It is classified derived-transient for this reason.
void LedgerNeverReachesObservationOrInfostate() {
  std::cout << "Running LedgerNeverReachesObservationOrInfostate\n" << std::flush;
  auto a = NewDuneState();
  auto b = NewDuneState();
  auto* da = static_cast<DuneImperiumState*>(a.get());
  auto* db = static_cast<DuneImperiumState*>(b.get());

  // Drive both to the SAME encoded position by different routes. `a` crosses
  // the 2-influence threshold once; `b` crosses it up, down and up again. Both
  // finish at influence 2 with the same vp_, but b's ledger carries two extra
  // events. The ledger is the only thing that differs -- which is precisely the
  // path-vs-position distinction the transposition key must not see.
  constexpr auto kEmp = dune_imperium::Faction::kEmperor;
  da->SetPlayerInfluenceForTesting(0, kEmp, 0);
  db->SetPlayerInfluenceForTesting(0, kEmp, 0);

  da->ExecuteInfluenceGainForTesting(0, kEmp, 2);

  db->ExecuteInfluenceGainForTesting(0, kEmp, 2);
  db->ExecuteInfluenceLossForTesting(0, kEmp, 2);
  db->ExecuteInfluenceGainForTesting(0, kEmp, 2);

  SPIEL_CHECK_EQ(da->GetPlayerVp(0), db->GetPlayerVp(0));
  SPIEL_CHECK_EQ(da->GetPlayerInfluenceForTesting(0, kEmp),
                 db->GetPlayerInfluenceForTesting(0, kEmp));
  SPIEL_CHECK_LT(da->GetVpEvents(0).size(), db->GetVpEvents(0).size());

  for (Player p = 0; p < 4; ++p) {
    SPIEL_CHECK_EQ(da->ObservationString(p), db->ObservationString(p));
    SPIEL_CHECK_EQ(da->InformationStateString(p), db->InformationStateString(p));
    // Through the base State*, whose vector-returning overloads the derived
    // span-taking ones would otherwise hide.
    SPIEL_CHECK_TRUE(a->ObservationTensor(p) == b->ObservationTensor(p));
    SPIEL_CHECK_TRUE(a->InformationStateTensor(p) ==
                     b->InformationStateTensor(p));
  }
}

// Clone must deep-copy the ledger; a shared or dropped ledger would make
// search-internal states corrupt the trunk's attribution.
void CloneCarriesAnIndependentLedger() {
  std::cout << "Running CloneCarriesAnIndependentLedger\n" << std::flush;
  auto state = NewDuneState();
  auto* ds = static_cast<DuneImperiumState*>(state.get());
  ds->SetPlayerVpForTesting(0, 4);

  auto clone = state->Clone();
  auto* dc = static_cast<DuneImperiumState*>(clone.get());
  SPIEL_CHECK_EQ(LedgerSum(dc, 0), 4);

  // Mutating the clone must not touch the original.
  dc->SetPlayerVpForTesting(0, 9);
  SPIEL_CHECK_EQ(LedgerSum(dc, 0), 9);
  SPIEL_CHECK_EQ(LedgerSum(ds, 0), 4);
  SPIEL_CHECK_EQ(ds->GetPlayerVp(0), 4);
}

}  // namespace
}  // namespace open_spiel

int main(int argc, char** argv) {
  std::cout << "dune_vp_ledger_test\n";
  open_spiel::StartingEndowmentIsRecorded();
  std::cout << "  PASS StartingEndowmentIsRecorded\n";
  open_spiel::ClampedRemovalRecordsWhatWasApplied();
  std::cout << "  PASS ClampedRemovalRecordsWhatWasApplied\n";
  open_spiel::LedgerNeverReachesObservationOrInfostate();
  std::cout << "  PASS LedgerNeverReachesObservationOrInfostate\n";
  open_spiel::CloneCarriesAnIndependentLedger();
  std::cout << "  PASS CloneCarriesAnIndependentLedger\n";
  open_spiel::LedgerSumsToVpAtEveryStep();
  std::cout << "  PASS LedgerSumsToVpAtEveryStep\n";
  std::cout << "ALL PASS\n";
  return 0;
}
