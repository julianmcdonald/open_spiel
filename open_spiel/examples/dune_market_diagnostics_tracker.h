#ifndef OPEN_SPIEL_EXAMPLES_DUNE_MARKET_DIAGNOSTICS_TRACKER_H_
#define OPEN_SPIEL_EXAMPLES_DUNE_MARKET_DIAGNOSTICS_TRACKER_H_

#include <algorithm>
#include <bitset>
#include <cstddef>
#include <vector>

#include "open_spiel/examples/dune_market_diagnostics_types.h"
#include "open_spiel/games/dune_imperium/dune_imperium.h"
#include "open_spiel/spiel.h"

namespace open_spiel {
namespace dune_market_diag {

// ---------------------------------------------------------------------------
// In-game transaction tracker
// ---------------------------------------------------------------------------
class GameMarketTracker {
 public:
  enum class TransType {
    kNone,
    kImperiumRow,
    kReserveArrakis,
    kReserveTsmf,
    kTleilaxuRow,
    kReclaimedForces,
    kTechTile,
    kHelenaReserve,
    kBypassFree,
    kBypassPaid,
    kArmandEcaz,
    kTleilaxuMasterImmediate,
    kTleilaxuMasterDeferredStart,
    kTleilaxuMasterToHandResolve,
    kMarketRemovalAtomics,
    kMarketRemovalHelena
  };

  struct PendingTransaction {
    TransType type = TransType::kNone;
    int item_id = -1;
    int pre_owned = 0;
    int player = -1;
    int round = 0;
    int specimens_before = 0;
    int tech_count_before = 0;
  };

  GameMarketTracker() = default;

  void InitGame(int episode_id, int candidate_seat) {
    data_ = CompactPerGameMarketDiagnostics{};
    data_.episode_id = episode_id;
    data_.candidate_seat = candidate_seat;
    pending_ = PendingTransaction{};
    deferred_tleilaxu_master_card_ = -1;
    deferred_tleilaxu_master_pre_owned_ = 0;
  }

  // Called when a decision point is reached. Marks visibility and legal opportunities for candidate decisions.
  void OnCandidateDecision(const DuneImperiumState& state, int model_player,
                           const std::vector<Action>& legal_actions) {
    if (state.CurrentPlayer() != model_player) return;

    // 1. Mark Market Visibility during candidate decisions
    const auto& imp_row = state.GetImperiumRowForTesting();
    for (size_t s = 0; s < imp_row.size() && s < 5; ++s) {
      const int card_id = imp_row[s];
      if (card_id >= 0 && card_id < kNumImperiumCards) {
        data_.imperium_visible.set(card_id);
      }
    }

    const auto& tlx_row = state.GetTleilaxuRowForTesting();
    for (size_t s = 0; s < tlx_row.size() && s < 2; ++s) {
      const int tlx_id = tlx_row[s];
      if (tlx_id >= 0 && tlx_id < kNumTleilaxuCards) {
        data_.tleilaxu_visible.set(tlx_id);
      }
    }

    const auto& tech_market = state.GetTechMarketForTesting();
    for (size_t s = 0; s < tech_market.size() && s < 3; ++s) {
      const int tech_id = tech_market[s];
      if (tech_id >= 0 && tech_id < kNumTechTiles) {
        data_.tech_visible.set(tech_id);
      }
    }

    data_.arrakis_liaison.visible = true;
    data_.spice_must_flow.visible = true;
    if ((state.enabled_expansions_mask() & dune_imperium::kExpansionImmortality) != 0) {
      data_.reclaimed_forces.visible = true;
    }

    // 2. Step-local bitsets to deduplicate multiple actions for the same item in this single decision step
    std::bitset<128> step_imperium;
    std::bitset<32> step_tleilaxu;
    std::bitset<32> step_tech;
    bool step_arrakis = false;
    bool step_tsmf = false;
    bool step_reclaimed = false;
    std::bitset<128> step_bypass_free;
    std::bitset<128> step_bypass_paid;
    std::bitset<128> step_armand;
    std::bitset<128> step_tlx_master;

    const int choice_kind = state.GetPendingIntrigueChoiceKind(model_player);

    if (choice_kind == static_cast<int>(IntrigueChoiceKind::kBypassProtocol)) {
      for (Action action : legal_actions) {
        if (action >= dune_imperium::kActionIntrigueChoiceImperiumRow0 &&
            action <= dune_imperium::kActionIntrigueChoiceImperiumRow0 + 5) {
          int slot = action - dune_imperium::kActionIntrigueChoiceImperiumRow0;
          int card = (slot == 5) ? kCardArrakisLiaison : state.GetImperiumRowCardForTesting(slot);
          if (card >= 0 && card < kNumImperiumCards) step_bypass_free.set(card);
        } else if (action >= dune_imperium::kActionIntrigueChoiceImperiumRowTop0 &&
                   action <= dune_imperium::kActionIntrigueChoiceImperiumRowTop0 + 5) {
          int slot = action - dune_imperium::kActionIntrigueChoiceImperiumRowTop0;
          int card = (slot == 5) ? kCardArrakisLiaison : state.GetImperiumRowCardForTesting(slot);
          if (card >= 0 && card < kNumImperiumCards) step_bypass_paid.set(card);
        }
      }
    } else if (choice_kind == static_cast<int>(IntrigueChoiceKind::kArmandEcazAcquire)) {
      for (Action action : legal_actions) {
        if (action >= dune_imperium::kActionIntrigueChoiceImperiumRow0 &&
            action <= dune_imperium::kActionIntrigueChoiceImperiumRow0 + 5) {
          int slot = action - dune_imperium::kActionIntrigueChoiceImperiumRow0;
          int card = (slot == 5) ? kCardArrakisLiaison : state.GetImperiumRowCardForTesting(slot);
          if (card >= 0 && card < kNumImperiumCards) step_armand.set(card);
        }
      }
    } else if (choice_kind == static_cast<int>(IntrigueChoiceKind::kTleilaxuMasterAcquire)) {
      for (Action action : legal_actions) {
        if (action >= dune_imperium::kActionIntrigueChoiceImperiumRow0 &&
            action <= dune_imperium::kActionIntrigueChoiceImperiumRow0 + 5) {
          int slot = action - dune_imperium::kActionIntrigueChoiceImperiumRow0;
          int card = (slot == 5) ? kCardArrakisLiaison : state.GetImperiumRowCardForTesting(slot);
          if (card >= 0 && card < kNumImperiumCards) step_tlx_master.set(card);
        }
      }
    } else if (choice_kind == static_cast<int>(IntrigueChoiceKind::kMachineCultureTech) ||
               choice_kind == static_cast<int>(IntrigueChoiceKind::kRhomburTechPick)) {
      for (Action action : legal_actions) {
        if (action >= dune_imperium::kActionIntrigueChoiceTech0 &&
            action <= dune_imperium::kActionIntrigueChoiceTech0 + 2) {
          int slot = action - dune_imperium::kActionIntrigueChoiceTech0;
          int tile = state.GetTechMarketTile(slot);
          if (tile >= 0 && tile < kNumTechTiles) step_tech.set(tile);
        }
      }
    } else if (choice_kind == static_cast<int>(IntrigueChoiceKind::kHarvestCellsBuy)) {
      for (Action action : legal_actions) {
        if (action >= dune_imperium::kActionIntrigueChoiceCount0 &&
            action <= dune_imperium::kActionIntrigueChoiceCount0 + static_cast<int>(tlx_row.size())) {
          int slot = action - dune_imperium::kActionIntrigueChoiceCount0;
          if (slot == static_cast<int>(tlx_row.size())) {
            step_reclaimed = true;
          } else if (slot >= 0 && slot < static_cast<int>(tlx_row.size())) {
            int tlx_id = tlx_row[slot];
            if (tlx_id >= 1 && tlx_id < kNumTleilaxuCards) {
              step_tleilaxu.set(tlx_id);
            }
          }
        }
      }
    } else {
      for (Action action : legal_actions) {
        // Ordinary Imperium Row Buy
        if (action >= dune_imperium::kActionBuyImperiumRow0 &&
            action < dune_imperium::kActionBuyImperiumRow0 + 5) {
          int slot = action - dune_imperium::kActionBuyImperiumRow0;
          int card = state.GetImperiumRowCardForTesting(slot);
          if (card >= 0 && card < kNumImperiumCards) step_imperium.set(card);
        } else if (action == dune_imperium::kActionBuyReserveArrakisLiaison) {
          step_arrakis = true;
        } else if (action == dune_imperium::kActionBuyReserveTheSpiceMustFlow) {
          step_tsmf = true;
        } else if (action == dune_imperium::kActionTleilaxuAcquire0) {
          step_reclaimed = true;
        } else if (action >= dune_imperium::kActionTleilaxuAcquire0 + 1 &&
                   action <= dune_imperium::kActionTleilaxuAcquire0 + 2) {
          int slot = action - (dune_imperium::kActionTleilaxuAcquire0 + 1);
          if (slot < static_cast<int>(tlx_row.size())) {
            int tlx_id = tlx_row[slot];
            if (tlx_id >= 0 && tlx_id < kNumTleilaxuCards) step_tleilaxu.set(tlx_id);
          }
        } else {
          // Tech acquisitions: standard actions 70..72 and 770..772
          // If in Appropriate Phase 2 (pending_appropriate_selected_tech_slot_ >= 0),
          // the player is picking currency for an ALREADY selected tile; do not count a new opportunity!
          if (state.GetPendingAppropriateSelectedTechSlot() < 0) {
            if (action >= dune_imperium::kActionTechAcquire0 &&
                action < dune_imperium::kActionTechAcquire0 + 3) {
              int slot = action - dune_imperium::kActionTechAcquire0;
              int tile = state.GetTechMarketTile(slot);
              if (tile >= 0 && tile < kNumTechTiles) step_tech.set(tile);
            } else if (action >= dune_imperium::kActionTechAcquireWithSolari0 &&
                       action < dune_imperium::kActionTechAcquireWithSolari0 + 3) {
              int slot = action - dune_imperium::kActionTechAcquireWithSolari0;
              int tile = state.GetTechMarketTile(slot);
              if (tile >= 0 && tile < kNumTechTiles) step_tech.set(tile);
            }
          }
        }
      }
    }

    // Accumulate opportunities into per-game record
    for (size_t c = 0; c < kNumImperiumCards; ++c) {
      if (step_imperium.test(c)) {
        data_.imperium_had_legal.set(c);
        data_.imperium_legal_opps[c]++;
      }
      if (step_bypass_free.test(c)) data_.bypass_free_opps[c]++;
      if (step_bypass_paid.test(c)) data_.bypass_paid_opps[c]++;
      if (step_armand.test(c)) data_.armand_opps[c]++;
      if (step_tlx_master.test(c)) data_.tleilaxu_master_opps[c]++;
    }

    for (size_t t = 0; t < kNumTleilaxuCards; ++t) {
      if (step_tleilaxu.test(t)) {
        data_.tleilaxu_had_legal.set(t);
        data_.tleilaxu_legal_opps[t]++;
      }
    }

    for (size_t k = 0; k < kNumTechTiles; ++k) {
      if (step_tech.test(k)) {
        data_.tech_had_legal.set(k);
        data_.tech_legal_opps[k]++;
      }
    }

    if (step_arrakis) {
      data_.arrakis_liaison.had_legal = true;
      data_.arrakis_liaison.legal_opps++;
    }
    if (step_tsmf) {
      data_.spice_must_flow.had_legal = true;
      data_.spice_must_flow.legal_opps++;
    }
    if (step_reclaimed) {
      data_.reclaimed_forces.had_legal = true;
      data_.reclaimed_forces.legal_opps++;
    }
  }

  // Pre-action snapshot for authoritative acquisition accounting
  void BeforeApplyAction(const DuneImperiumState& state, Player current_player,
                         int model_player, Action action) {
    if (current_player != model_player) return;
    pending_ = PendingTransaction{};
    pending_.player = current_player;
    pending_.round = state.GetCurrentRound();

    const int choice_kind = state.GetPendingIntrigueChoiceKind(model_player);

    if (choice_kind == static_cast<int>(IntrigueChoiceKind::kBypassProtocol)) {
      if (action >= dune_imperium::kActionIntrigueChoiceImperiumRow0 &&
          action <= dune_imperium::kActionIntrigueChoiceImperiumRow0 + 5) {
        int slot = action - dune_imperium::kActionIntrigueChoiceImperiumRow0;
        int card = (slot == 5) ? kCardArrakisLiaison : state.GetImperiumRowCardForTesting(slot);
        pending_.type = TransType::kBypassFree;
        pending_.item_id = card;
        pending_.pre_owned = state.GetDeckPoolTotalCards(current_player, card);
      } else if (action >= dune_imperium::kActionIntrigueChoiceImperiumRowTop0 &&
                 action <= dune_imperium::kActionIntrigueChoiceImperiumRowTop0 + 5) {
        int slot = action - dune_imperium::kActionIntrigueChoiceImperiumRowTop0;
        int card = (slot == 5) ? kCardArrakisLiaison : state.GetImperiumRowCardForTesting(slot);
        pending_.type = TransType::kBypassPaid;
        pending_.item_id = card;
        pending_.pre_owned = state.GetDeckPoolTotalCards(current_player, card);
      }
    } else if (choice_kind == static_cast<int>(IntrigueChoiceKind::kArmandEcazAcquire)) {
      if (action >= dune_imperium::kActionIntrigueChoiceImperiumRow0 &&
          action <= dune_imperium::kActionIntrigueChoiceImperiumRow0 + 5) {
        int slot = action - dune_imperium::kActionIntrigueChoiceImperiumRow0;
        int card = (slot == 5) ? kCardArrakisLiaison : state.GetImperiumRowCardForTesting(slot);
        pending_.type = TransType::kArmandEcaz;
        pending_.item_id = card;
        pending_.pre_owned = state.GetDeckPoolTotalCards(current_player, card);
      }
    } else if (choice_kind == static_cast<int>(IntrigueChoiceKind::kTleilaxuMasterAcquire)) {
      if (action >= dune_imperium::kActionIntrigueChoiceImperiumRow0 &&
          action <= dune_imperium::kActionIntrigueChoiceImperiumRow0 + 5) {
        int slot = action - dune_imperium::kActionIntrigueChoiceImperiumRow0;
        int card = (slot == 5) ? kCardArrakisLiaison : state.GetImperiumRowCardForTesting(slot);
        if (state.HasSecondGeneticMarker(current_player)) {
          // Defer until kTleilaxuMasterToHandChoice completes on the next action
          pending_.type = TransType::kTleilaxuMasterDeferredStart;
          pending_.item_id = card;
          deferred_tleilaxu_master_card_ = card;
          deferred_tleilaxu_master_pre_owned_ = state.GetDeckPoolTotalCards(current_player, card);
        } else {
          pending_.type = TransType::kTleilaxuMasterImmediate;
          pending_.item_id = card;
          pending_.pre_owned = state.GetDeckPoolTotalCards(current_player, card);
        }
      }
    } else if (choice_kind == static_cast<int>(IntrigueChoiceKind::kTleilaxuMasterToHandChoice)) {
      pending_.type = TransType::kTleilaxuMasterToHandResolve;
      pending_.item_id = (deferred_tleilaxu_master_card_ >= 0) ? deferred_tleilaxu_master_card_
                                                                : state.GetPendingIntrigueChoiceCardId();
      pending_.pre_owned = deferred_tleilaxu_master_pre_owned_;
    } else if (choice_kind == static_cast<int>(IntrigueChoiceKind::kMachineCultureTech) ||
               choice_kind == static_cast<int>(IntrigueChoiceKind::kRhomburTechPick)) {
      if (action >= dune_imperium::kActionIntrigueChoiceTech0 &&
          action <= dune_imperium::kActionIntrigueChoiceTech0 + 2) {
        int slot = action - dune_imperium::kActionIntrigueChoiceTech0;
        int tile = state.GetTechMarketTile(slot);
        pending_.type = TransType::kTechTile;
        pending_.item_id = tile;
        pending_.tech_count_before = static_cast<int>(state.GetPlayerTechTilesForTesting(current_player).size());
      }
    } else if (choice_kind == static_cast<int>(IntrigueChoiceKind::kHarvestCellsBuy)) {
      const auto& tlx_row = state.GetTleilaxuRowForTesting();
      if (action >= dune_imperium::kActionIntrigueChoiceCount0 &&
          action <= dune_imperium::kActionIntrigueChoiceCount0 + static_cast<int>(tlx_row.size())) {
        int slot = action - dune_imperium::kActionIntrigueChoiceCount0;
        if (slot == static_cast<int>(tlx_row.size())) {
          pending_.type = TransType::kReclaimedForces;
          pending_.item_id = 0;
          pending_.specimens_before = state.GetSpecimensForTesting(current_player);
        } else if (slot >= 0 && slot < static_cast<int>(tlx_row.size())) {
          int tlx_id = tlx_row[slot];
          if (tlx_id >= 1 && tlx_id <= 18) {
            int imperium_id = kTleilaxuCards[tlx_id].imperium_card_id;
            pending_.type = TransType::kTleilaxuRow;
            pending_.item_id = tlx_id;
            pending_.pre_owned = state.GetDeckPoolTotalCards(current_player, imperium_id);
          }
        }
      }
    } else if (choice_kind == static_cast<int>(IntrigueChoiceKind::kHelenaReserve)) {
      if (action >= dune_imperium::kActionIntrigueChoiceImperiumRow0 &&
          action < dune_imperium::kActionIntrigueChoiceImperiumRow0 + 5) {
        int slot = action - dune_imperium::kActionIntrigueChoiceImperiumRow0;
        int card = state.GetImperiumRowCardForTesting(slot);
        pending_.type = TransType::kMarketRemovalHelena;
        pending_.item_id = card;
      }
    } else if (action == dune_imperium::kActionFamilyAtomics) {
      pending_.type = TransType::kMarketRemovalAtomics;
    } else if (action >= dune_imperium::kActionBuyHelenaReserve0 &&
               action < dune_imperium::kActionBuyHelenaReserve0 + 1) {
      const auto& res = state.GetHelenaReservedCards(current_player);
      int card = (!res.empty()) ? res[0] : -1;
      pending_.type = TransType::kHelenaReserve;
      pending_.item_id = card;
      pending_.pre_owned = state.GetDeckPoolTotalCards(current_player, card);
    } else if (action >= dune_imperium::kActionBuyImperiumRow0 &&
               action < dune_imperium::kActionBuyImperiumRow0 + 5) {
      int slot = action - dune_imperium::kActionBuyImperiumRow0;
      int card = state.GetImperiumRowCardForTesting(slot);
      pending_.type = TransType::kImperiumRow;
      pending_.item_id = card;
      pending_.pre_owned = state.GetDeckPoolTotalCards(current_player, card);
    } else if (action == dune_imperium::kActionBuyReserveArrakisLiaison) {
      pending_.type = TransType::kReserveArrakis;
      pending_.item_id = kCardArrakisLiaison;
      pending_.pre_owned = state.GetDeckPoolTotalCards(current_player, kCardArrakisLiaison);
    } else if (action == dune_imperium::kActionBuyReserveTheSpiceMustFlow) {
      pending_.type = TransType::kReserveTsmf;
      pending_.item_id = kCardTheSpiceMustFlow;
      pending_.pre_owned = state.GetDeckPoolTotalCards(current_player, kCardTheSpiceMustFlow);
    } else if (action == dune_imperium::kActionTleilaxuAcquire0) {
      pending_.type = TransType::kReclaimedForces;
      pending_.item_id = 0;
      pending_.specimens_before = state.GetSpecimensForTesting(current_player);
    } else if (action >= dune_imperium::kActionTleilaxuAcquire0 + 1 &&
               action <= dune_imperium::kActionTleilaxuAcquire0 + 2) {
      int slot = action - (dune_imperium::kActionTleilaxuAcquire0 + 1);
      const auto& tlx_row = state.GetTleilaxuRowForTesting();
      int tlx_id = (slot < static_cast<int>(tlx_row.size())) ? tlx_row[slot] : -1;
      int imperium_id = (tlx_id >= 1 && tlx_id <= 18) ? kTleilaxuCards[tlx_id].imperium_card_id : -1;
      pending_.type = TransType::kTleilaxuRow;
      pending_.item_id = tlx_id;
      pending_.pre_owned = state.GetDeckPoolTotalCards(current_player, imperium_id);
    } else {
      // Tech tile acquisitions: 70..72 or 770..772
      int slot = -1;
      if (action >= dune_imperium::kActionTechAcquire0 &&
          action < dune_imperium::kActionTechAcquire0 + 3) {
        slot = action - dune_imperium::kActionTechAcquire0;
      } else if (action >= dune_imperium::kActionTechAcquireWithSolari0 &&
                 action < dune_imperium::kActionTechAcquireWithSolari0 + 3) {
        slot = action - dune_imperium::kActionTechAcquireWithSolari0;
      }
      if (slot >= 0) {
        int tile = state.GetTechMarketTile(slot);
        pending_.type = TransType::kTechTile;
        pending_.item_id = tile;
        pending_.tech_count_before = static_cast<int>(state.GetPlayerTechTilesForTesting(current_player).size());
      }
    }
  }

  // Post-action verification and recording
  void AfterApplyAction(const DuneImperiumState& state, Player current_player,
                        int model_player, Action /*action*/) {
    if (current_player != model_player || pending_.type == TransType::kNone) return;

    const int round_idx = std::min(std::max(1, pending_.round), kMaxTrackedRounds - 1);

    switch (pending_.type) {
      case TransType::kImperiumRow: {
        int card = pending_.item_id;
        if (card >= 0 && card < kNumImperiumCards) {
          int post_owned = state.GetDeckPoolTotalCards(current_player, card);
          if (post_owned == pending_.pre_owned + 1) {
            data_.imperium_purchased.set(card);
            data_.imperium_purchases[card]++;
            data_.imperium_purchase_rounds[card][round_idx]++;
          }
        }
        break;
      }
      case TransType::kReserveArrakis: {
        int post_owned = state.GetDeckPoolTotalCards(current_player, kCardArrakisLiaison);
        if (post_owned == pending_.pre_owned + 1) {
          data_.arrakis_liaison.purchased = true;
          data_.arrakis_liaison.purchases++;
          data_.arrakis_liaison.purchase_rounds[round_idx]++;
        }
        break;
      }
      case TransType::kReserveTsmf: {
        int post_owned = state.GetDeckPoolTotalCards(current_player, kCardTheSpiceMustFlow);
        if (post_owned == pending_.pre_owned + 1) {
          data_.spice_must_flow.purchased = true;
          data_.spice_must_flow.purchases++;
          data_.spice_must_flow.purchase_rounds[round_idx]++;
        }
        break;
      }
      case TransType::kTleilaxuRow: {
        int tlx_id = pending_.item_id;
        if (tlx_id >= 1 && tlx_id <= 18) {
          int imperium_id = kTleilaxuCards[tlx_id].imperium_card_id;
          int post_owned = state.GetDeckPoolTotalCards(current_player, imperium_id);
          if (post_owned == pending_.pre_owned + 1) {
            data_.tleilaxu_purchased.set(tlx_id);
            data_.tleilaxu_purchases[tlx_id]++;
            data_.tleilaxu_purchase_rounds[tlx_id][round_idx]++;
          }
        }
        break;
      }
      case TransType::kReclaimedForces: {
        int post_specimens = state.GetSpecimensForTesting(current_player);
        int choice_kind = state.GetPendingIntrigueChoiceKind(current_player);
        if (post_specimens == pending_.specimens_before - 3 ||
            choice_kind == static_cast<int>(IntrigueChoiceKind::kReclaimedForcesChoice)) {
          data_.reclaimed_forces.purchased = true;
          data_.reclaimed_forces.purchases++;
          data_.reclaimed_forces.purchase_rounds[round_idx]++;
        }
        break;
      }
      case TransType::kTechTile: {
        int tile = pending_.item_id;
        if (tile >= 0 && tile < kNumTechTiles) {
          int post_count = static_cast<int>(state.GetPlayerTechTilesForTesting(current_player).size());
          if (post_count == pending_.tech_count_before + 1) {
            data_.tech_purchased.set(tile);
            data_.tech_purchases[tile]++;
            data_.tech_purchase_rounds[tile][round_idx]++;
          }
        }
        break;
      }
      case TransType::kHelenaReserve: {
        int card = pending_.item_id;
        if (card >= 0 && card < kNumImperiumCards) {
          int post_owned = state.GetDeckPoolTotalCards(current_player, card);
          if (post_owned == pending_.pre_owned + 1) {
            data_.helena_reserve_purchases[card]++;
          }
        }
        break;
      }
      case TransType::kBypassFree: {
        int card = pending_.item_id;
        if (card >= 0 && card < kNumImperiumCards) {
          int post_owned = state.GetDeckPoolTotalCards(current_player, card);
          if (post_owned == pending_.pre_owned + 1) {
            data_.bypass_free_acquisitions[card]++;
          }
        }
        break;
      }
      case TransType::kBypassPaid: {
        int card = pending_.item_id;
        if (card >= 0 && card < kNumImperiumCards) {
          int post_owned = state.GetDeckPoolTotalCards(current_player, card);
          if (post_owned == pending_.pre_owned + 1) {
            data_.bypass_paid_acquisitions[card]++;
          }
        }
        break;
      }
      case TransType::kArmandEcaz: {
        int card = pending_.item_id;
        if (card >= 0 && card < kNumImperiumCards) {
          int post_owned = state.GetDeckPoolTotalCards(current_player, card);
          if (post_owned == pending_.pre_owned + 1) {
            data_.armand_acquisitions[card]++;
          }
        }
        break;
      }
      case TransType::kTleilaxuMasterImmediate: {
        int card = pending_.item_id;
        if (card >= 0 && card < kNumImperiumCards) {
          int post_owned = state.GetDeckPoolTotalCards(current_player, card);
          if (post_owned == pending_.pre_owned + 1) {
            data_.tleilaxu_master_acquisitions[card]++;
          }
        }
        break;
      }
      case TransType::kTleilaxuMasterDeferredStart:
        // Do not count yet; wait for resolution in kTleilaxuMasterToHandResolve
        break;
      case TransType::kTleilaxuMasterToHandResolve: {
        int card = pending_.item_id;
        if (card >= 0 && card < kNumImperiumCards) {
          int post_owned = state.GetDeckPoolTotalCards(current_player, card);
          if (post_owned == pending_.pre_owned + 1) {
            data_.tleilaxu_master_acquisitions[card]++;
          }
        }
        deferred_tleilaxu_master_card_ = -1;
        deferred_tleilaxu_master_pre_owned_ = 0;
        break;
      }
      case TransType::kMarketRemovalAtomics: {
        data_.family_atomics_uses++;
        break;
      }
      case TransType::kMarketRemovalHelena: {
        int card = pending_.item_id;
        if (card >= 0 && card < kNumImperiumCards) {
          data_.helena_removals[card]++;
        }
        break;
      }
      default:
        break;
    }

    pending_ = PendingTransaction{};
  }

  const CompactPerGameMarketDiagnostics& GetData() const { return data_; }

 private:
  CompactPerGameMarketDiagnostics data_;
  PendingTransaction pending_;
  int deferred_tleilaxu_master_card_ = -1;
  int deferred_tleilaxu_master_pre_owned_ = 0;
};

}  // namespace dune_market_diag
}  // namespace open_spiel

#endif  // OPEN_SPIEL_EXAMPLES_DUNE_MARKET_DIAGNOSTICS_TRACKER_H_
