#ifndef OPEN_SPIEL_EXAMPLES_DUNE_MARKET_DIAGNOSTICS_H_
#define OPEN_SPIEL_EXAMPLES_DUNE_MARKET_DIAGNOSTICS_H_

#include <algorithm>
#include <array>
#include <bitset>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

#include "open_spiel/abseil-cpp/absl/strings/str_format.h"
#include "open_spiel/games/dune_imperium/dune_imperium.h"
#include "open_spiel/games/dune_imperium/dune_imperium_cards.h"
#include "open_spiel/games/dune_imperium/dune_imperium_common.h"
#include "open_spiel/games/dune_imperium/dune_imperium_content.h"
#include "open_spiel/games/dune_imperium/dune_imperium_content_generated.h"
#include "open_spiel/json/include/nlohmann/json.hpp"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace dune_market_diag {

using dune_imperium::DuneImperiumState;
using dune_imperium::IntrigueChoiceKind;
using dune_imperium::kCardArrakisLiaison;
using dune_imperium::kCardTheSpiceMustFlow;
using dune_imperium::kImperiumCards;
using open_spiel::kInvalidAction;
using dune_imperium::kInvalidCard;
using dune_imperium::kMaxRounds;
using dune_imperium::kNumImperiumCardIds;
using dune_imperium::kNumPlayers;
using dune_imperium::kTechTiles;
using dune_imperium::kTleilaxuCards;

inline constexpr int kNumImperiumCards = 126;
inline constexpr int kNumTleilaxuCards = 19;
inline constexpr int kNumTechTiles = 18;
inline constexpr int kMaxTrackedRounds = 12;

// Check if an Imperium card is eligible to appear in the rotating draw deck / row.
inline bool IsEligibleRotatingImperiumCard(int card_id, uint32_t expansion_mask = dune_imperium::kExpansionAll) {
  if (card_id < 0 || card_id >= kNumImperiumCards) return false;
  const auto& card = kImperiumCards[card_id];
  return (card.deck_count > 0) && ((card.expansion_mask & expansion_mask) != 0);
}

inline bool IsStarterCard(int card_id) {
  return card_id >= 0 && card_id <= 6;
}

inline bool IsFixedImperiumReserve(int card_id) {
  return card_id == kCardArrakisLiaison || card_id == kCardTheSpiceMustFlow;
}

// ---------------------------------------------------------------------------
// Compact per-game data structure
// ---------------------------------------------------------------------------
struct FixedSupplyPerGame {
  bool visible = false;
  bool had_legal = false;
  bool purchased = false;
  uint16_t legal_opps = 0;
  uint16_t purchases = 0;
  std::array<uint8_t, kMaxTrackedRounds> purchase_rounds{};
};

struct CompactPerGameMarketDiagnostics {
  int episode_id = -1;
  int candidate_seat = -1;

  // Rotating Imperium Row (126 entries in namespace; only eligible cards appear in rotating row)
  std::bitset<128> imperium_visible;
  std::bitset<128> imperium_had_legal;
  std::bitset<128> imperium_purchased;
  std::array<uint16_t, kNumImperiumCards> imperium_legal_opps{};
  std::array<uint16_t, kNumImperiumCards> imperium_purchases{};
  std::array<std::array<uint8_t, kMaxTrackedRounds>, kNumImperiumCards> imperium_purchase_rounds{};

  // Rotating Tleilaxu Row (IDs 1..18 are rotating; 0 is Reclaimed Forces fixed reserve)
  std::bitset<32> tleilaxu_visible;
  std::bitset<32> tleilaxu_had_legal;
  std::bitset<32> tleilaxu_purchased;
  std::array<uint16_t, kNumTleilaxuCards> tleilaxu_legal_opps{};
  std::array<uint16_t, kNumTleilaxuCards> tleilaxu_purchases{};
  std::array<std::array<uint8_t, kMaxTrackedRounds>, kNumTleilaxuCards> tleilaxu_purchase_rounds{};

  // Face-Up Tech Tiles (0..17)
  std::bitset<32> tech_visible;
  std::bitset<32> tech_had_legal;
  std::bitset<32> tech_purchased;
  std::array<uint16_t, kNumTechTiles> tech_legal_opps{};
  std::array<uint16_t, kNumTechTiles> tech_purchases{};
  std::array<std::array<uint8_t, kMaxTrackedRounds>, kNumTechTiles> tech_purchase_rounds{};

  // Fixed supplies
  FixedSupplyPerGame arrakis_liaison;
  FixedSupplyPerGame spice_must_flow;
  FixedSupplyPerGame reclaimed_forces;

  // Special acquisition / removal events
  std::array<uint16_t, kNumImperiumCards> bypass_free_opps{};
  std::array<uint16_t, kNumImperiumCards> bypass_free_acquisitions{};
  std::array<uint16_t, kNumImperiumCards> bypass_paid_opps{};
  std::array<uint16_t, kNumImperiumCards> bypass_paid_acquisitions{};
  std::array<uint16_t, kNumImperiumCards> armand_opps{};
  std::array<uint16_t, kNumImperiumCards> armand_acquisitions{};
  std::array<uint16_t, kNumImperiumCards> tleilaxu_master_opps{};
  std::array<uint16_t, kNumImperiumCards> tleilaxu_master_acquisitions{};
  std::array<uint16_t, kNumImperiumCards> helena_removals{};
  std::array<uint16_t, kNumImperiumCards> helena_reserve_purchases{};
  uint16_t family_atomics_uses = 0;

  nlohmann::json ToJson() const {
    nlohmann::json j;
    j["episode_id"] = episode_id;
    j["candidate_seat"] = candidate_seat;

    auto fixed_to_json = [](const FixedSupplyPerGame& f) {
      nlohmann::json fj;
      if (f.visible) fj["vis"] = 1;
      if (f.had_legal) fj["had_legal"] = 1;
      if (f.purchased) fj["purchased"] = 1;
      if (f.legal_opps > 0) fj["opps"] = f.legal_opps;
      if (f.purchases > 0) fj["buys"] = f.purchases;
      nlohmann::json rnd_j = nlohmann::json::object();
      bool has_rnd = false;
      for (int r = 0; r < kMaxTrackedRounds; ++r) {
        if (f.purchase_rounds[r] > 0) {
          rnd_j[std::to_string(r)] = f.purchase_rounds[r];
          has_rnd = true;
        }
      }
      if (has_rnd) fj["rounds"] = rnd_j;
      return fj;
    };

    if (arrakis_liaison.visible || arrakis_liaison.had_legal || arrakis_liaison.purchased) {
      j["arrakis"] = fixed_to_json(arrakis_liaison);
    }
    if (spice_must_flow.visible || spice_must_flow.had_legal || spice_must_flow.purchased) {
      j["tsmf"] = fixed_to_json(spice_must_flow);
    }
    if (reclaimed_forces.visible || reclaimed_forces.had_legal || reclaimed_forces.purchased) {
      j["reclaimed"] = fixed_to_json(reclaimed_forces);
    }

    nlohmann::json imp_arr = nlohmann::json::array();
    for (int i = 0; i < kNumImperiumCards; ++i) {
      if (!imperium_visible.test(i) && !imperium_had_legal.test(i) && !imperium_purchased.test(i)) continue;
      nlohmann::json cj;
      cj["id"] = i;
      if (imperium_visible.test(i)) cj["vis"] = 1;
      if (imperium_had_legal.test(i)) cj["had_legal"] = 1;
      if (imperium_purchased.test(i)) cj["purchased"] = 1;
      if (imperium_legal_opps[i] > 0) cj["opps"] = imperium_legal_opps[i];
      if (imperium_purchases[i] > 0) cj["buys"] = imperium_purchases[i];
      nlohmann::json rnd_j = nlohmann::json::object();
      bool has_rnd = false;
      for (int r = 0; r < kMaxTrackedRounds; ++r) {
        if (imperium_purchase_rounds[i][r] > 0) {
          rnd_j[std::to_string(r)] = imperium_purchase_rounds[i][r];
          has_rnd = true;
        }
      }
      if (has_rnd) cj["rounds"] = rnd_j;
      imp_arr.push_back(cj);
    }
    if (!imp_arr.empty()) j["imperium"] = imp_arr;

    nlohmann::json tlx_arr = nlohmann::json::array();
    for (int i = 1; i < kNumTleilaxuCards; ++i) {
      if (!tleilaxu_visible.test(i) && !tleilaxu_had_legal.test(i) && !tleilaxu_purchased.test(i)) continue;
      nlohmann::json tj;
      tj["id"] = i;
      if (tleilaxu_visible.test(i)) tj["vis"] = 1;
      if (tleilaxu_had_legal.test(i)) tj["had_legal"] = 1;
      if (tleilaxu_purchased.test(i)) tj["purchased"] = 1;
      if (tleilaxu_legal_opps[i] > 0) tj["opps"] = tleilaxu_legal_opps[i];
      if (tleilaxu_purchases[i] > 0) tj["buys"] = tleilaxu_purchases[i];
      nlohmann::json rnd_j = nlohmann::json::object();
      bool has_rnd = false;
      for (int r = 0; r < kMaxTrackedRounds; ++r) {
        if (tleilaxu_purchase_rounds[i][r] > 0) {
          rnd_j[std::to_string(r)] = tleilaxu_purchase_rounds[i][r];
          has_rnd = true;
        }
      }
      if (has_rnd) tj["rounds"] = rnd_j;
      tlx_arr.push_back(tj);
    }
    if (!tlx_arr.empty()) j["tleilaxu"] = tlx_arr;

    nlohmann::json tech_arr = nlohmann::json::array();
    for (int i = 0; i < kNumTechTiles; ++i) {
      if (!tech_visible.test(i) && !tech_had_legal.test(i) && !tech_purchased.test(i)) continue;
      nlohmann::json tech_j;
      tech_j["id"] = i;
      if (tech_visible.test(i)) tech_j["vis"] = 1;
      if (tech_had_legal.test(i)) tech_j["had_legal"] = 1;
      if (tech_purchased.test(i)) tech_j["purchased"] = 1;
      if (tech_legal_opps[i] > 0) tech_j["opps"] = tech_legal_opps[i];
      if (tech_purchases[i] > 0) tech_j["buys"] = tech_purchases[i];
      nlohmann::json rnd_j = nlohmann::json::object();
      bool has_rnd = false;
      for (int r = 0; r < kMaxTrackedRounds; ++r) {
        if (tech_purchase_rounds[i][r] > 0) {
          rnd_j[std::to_string(r)] = tech_purchase_rounds[i][r];
          has_rnd = true;
        }
      }
      if (has_rnd) tech_j["rounds"] = rnd_j;
      tech_arr.push_back(tech_j);
    }
    if (!tech_arr.empty()) j["tech"] = tech_arr;

    // Special routes
    nlohmann::json sp_j = nlohmann::json::object();
    bool has_sp = false;
    if (family_atomics_uses > 0) {
      sp_j["family_atomics_uses"] = family_atomics_uses;
      has_sp = true;
    }
    auto add_sp_map = [&](const std::string& key, const std::array<uint16_t, kNumImperiumCards>& arr) {
      nlohmann::json m = nlohmann::json::object();
      bool any = false;
      for (int i = 0; i < kNumImperiumCards; ++i) {
        if (arr[i] > 0) {
          m[std::to_string(i)] = arr[i];
          any = true;
        }
      }
      if (any) {
        sp_j[key] = m;
        has_sp = true;
      }
    };
    add_sp_map("bypass_free_opps", bypass_free_opps);
    add_sp_map("bypass_free_acquisitions", bypass_free_acquisitions);
    add_sp_map("bypass_paid_opps", bypass_paid_opps);
    add_sp_map("bypass_paid_acquisitions", bypass_paid_acquisitions);
    add_sp_map("armand_opps", armand_opps);
    add_sp_map("armand_acquisitions", armand_acquisitions);
    add_sp_map("tleilaxu_master_opps", tleilaxu_master_opps);
    add_sp_map("tleilaxu_master_acquisitions", tleilaxu_master_acquisitions);
    add_sp_map("helena_removals", helena_removals);
    add_sp_map("helena_reserve_purchases", helena_reserve_purchases);
    if (has_sp) j["special"] = sp_j;

    return j;
  }

  static CompactPerGameMarketDiagnostics FromJson(const nlohmann::json& j) {
    CompactPerGameMarketDiagnostics g;
    g.episode_id = j.value("episode_id", -1);
    g.candidate_seat = j.value("candidate_seat", -1);

    auto parse_fixed = [](const nlohmann::json& fj, FixedSupplyPerGame& f) {
      f.visible = (fj.value("vis", 0) != 0);
      f.had_legal = (fj.value("had_legal", 0) != 0);
      f.purchased = (fj.value("purchased", 0) != 0);
      f.legal_opps = static_cast<uint16_t>(fj.value("opps", 0));
      f.purchases = static_cast<uint16_t>(fj.value("buys", 0));
      if (fj.contains("rounds") && fj["rounds"].is_object()) {
        for (auto it = fj["rounds"].begin(); it != fj["rounds"].end(); ++it) {
          int r = std::stoi(it.key());
          if (r >= 0 && r < kMaxTrackedRounds) {
            f.purchase_rounds[r] = static_cast<uint8_t>(it.value().get<int>());
          }
        }
      }
    };

    if (j.contains("arrakis")) parse_fixed(j["arrakis"], g.arrakis_liaison);
    if (j.contains("tsmf")) parse_fixed(j["tsmf"], g.spice_must_flow);
    if (j.contains("reclaimed")) parse_fixed(j["reclaimed"], g.reclaimed_forces);

    if (j.contains("imperium") && j["imperium"].is_array()) {
      for (const auto& cj : j["imperium"]) {
        int id = cj.value("id", -1);
        if (id >= 0 && id < kNumImperiumCards) {
          if (cj.value("vis", 0) != 0) g.imperium_visible.set(id);
          if (cj.value("had_legal", 0) != 0) g.imperium_had_legal.set(id);
          if (cj.value("purchased", 0) != 0) g.imperium_purchased.set(id);
          g.imperium_legal_opps[id] = static_cast<uint16_t>(cj.value("opps", 0));
          g.imperium_purchases[id] = static_cast<uint16_t>(cj.value("buys", 0));
          if (cj.contains("rounds") && cj["rounds"].is_object()) {
            for (auto it = cj["rounds"].begin(); it != cj["rounds"].end(); ++it) {
              int r = std::stoi(it.key());
              if (r >= 0 && r < kMaxTrackedRounds) {
                g.imperium_purchase_rounds[id][r] = static_cast<uint8_t>(it.value().get<int>());
              }
            }
          }
        }
      }
    }

    if (j.contains("tleilaxu") && j["tleilaxu"].is_array()) {
      for (const auto& tj : j["tleilaxu"]) {
        int id = tj.value("id", -1);
        if (id >= 1 && id < kNumTleilaxuCards) {
          if (tj.value("vis", 0) != 0) g.tleilaxu_visible.set(id);
          if (tj.value("had_legal", 0) != 0) g.tleilaxu_had_legal.set(id);
          if (tj.value("purchased", 0) != 0) g.tleilaxu_purchased.set(id);
          g.tleilaxu_legal_opps[id] = static_cast<uint16_t>(tj.value("opps", 0));
          g.tleilaxu_purchases[id] = static_cast<uint16_t>(tj.value("buys", 0));
          if (tj.contains("rounds") && tj["rounds"].is_object()) {
            for (auto it = tj["rounds"].begin(); it != tj["rounds"].end(); ++it) {
              int r = std::stoi(it.key());
              if (r >= 0 && r < kMaxTrackedRounds) {
                g.tleilaxu_purchase_rounds[id][r] = static_cast<uint8_t>(it.value().get<int>());
              }
            }
          }
        }
      }
    }

    if (j.contains("tech") && j["tech"].is_array()) {
      for (const auto& tech_j : j["tech"]) {
        int id = tech_j.value("id", -1);
        if (id >= 0 && id < kNumTechTiles) {
          if (tech_j.value("vis", 0) != 0) g.tech_visible.set(id);
          if (tech_j.value("had_legal", 0) != 0) g.tech_had_legal.set(id);
          if (tech_j.value("purchased", 0) != 0) g.tech_purchased.set(id);
          g.tech_legal_opps[id] = static_cast<uint16_t>(tech_j.value("opps", 0));
          g.tech_purchases[id] = static_cast<uint16_t>(tech_j.value("buys", 0));
          if (tech_j.contains("rounds") && tech_j["rounds"].is_object()) {
            for (auto it = tech_j["rounds"].begin(); it != tech_j["rounds"].end(); ++it) {
              int r = std::stoi(it.key());
              if (r >= 0 && r < kMaxTrackedRounds) {
                g.tech_purchase_rounds[id][r] = static_cast<uint8_t>(it.value().get<int>());
              }
            }
          }
        }
      }
    }

    if (j.contains("special") && j["special"].is_object()) {
      const auto& sp_j = j["special"];
      g.family_atomics_uses = static_cast<uint16_t>(sp_j.value("family_atomics_uses", 0));
      auto parse_sp_map = [&](const std::string& key, std::array<uint16_t, kNumImperiumCards>& arr) {
        if (sp_j.contains(key) && sp_j[key].is_object()) {
          for (auto it = sp_j[key].begin(); it != sp_j[key].end(); ++it) {
            int i = std::stoi(it.key());
            if (i >= 0 && i < kNumImperiumCards) {
              arr[i] = static_cast<uint16_t>(it.value().get<int>());
            }
          }
        }
      };
      parse_sp_map("bypass_free_opps", g.bypass_free_opps);
      parse_sp_map("bypass_free_acquisitions", g.bypass_free_acquisitions);
      parse_sp_map("bypass_paid_opps", g.bypass_paid_opps);
      parse_sp_map("bypass_paid_acquisitions", g.bypass_paid_acquisitions);
      parse_sp_map("armand_opps", g.armand_opps);
      parse_sp_map("armand_acquisitions", g.armand_acquisitions);
      parse_sp_map("tleilaxu_master_opps", g.tleilaxu_master_opps);
      parse_sp_map("tleilaxu_master_acquisitions", g.tleilaxu_master_acquisitions);
      parse_sp_map("helena_removals", g.helena_removals);
      parse_sp_map("helena_reserve_purchases", g.helena_reserve_purchases);
    }

    return g;
  }
};

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

// ---------------------------------------------------------------------------
// Aggregated report statistics across all games
// ---------------------------------------------------------------------------
struct AggregatedItemStats {
  int id = -1;
  std::string name;
  int cost = 0;
  bool eligible_rotating = true;
  int games_visible = 0;
  int games_legal_opportunity = 0;
  int legal_decision_opportunities = 0;
  int purchases = 0;
  int games_purchased = 0;
  double conversion_rate = -1.0;  // -1.0 indicates N/A
  std::array<int, kMaxTrackedRounds> purchase_rounds{};
};

class MarketDiagnosticsReport {
 public:
  MarketDiagnosticsReport() = default;

  void Aggregate(const std::vector<CompactPerGameMarketDiagnostics>& all_games, int total_games) {
    total_games_ = total_games;
    imperium_items_.clear();
    tleilaxu_items_.clear();
    tech_items_.clear();

    // 1. Initialize Imperium Row items
    for (int i = 0; i < kNumImperiumCards; ++i) {
      AggregatedItemStats item;
      item.id = i;
      item.name = std::string(kImperiumCards[i].name);
      item.cost = kImperiumCards[i].persuasion_cost;
      item.eligible_rotating = IsEligibleRotatingImperiumCard(i);
      imperium_items_.push_back(item);
    }

    // 2. Initialize Tleilaxu Row items (1..18)
    for (int i = 1; i < kNumTleilaxuCards; ++i) {
      AggregatedItemStats item;
      item.id = i;
      item.name = std::string(kTleilaxuCards[i].name);
      item.cost = kTleilaxuCards[i].specimen_cost;
      item.eligible_rotating = true;
      tleilaxu_items_.push_back(item);
    }

    // 3. Initialize Tech items (0..17)
    for (int i = 0; i < kNumTechTiles; ++i) {
      AggregatedItemStats item;
      item.id = i;
      item.name = std::string(kTechTiles[i].name);
      item.cost = kTechTiles[i].spice_cost;
      item.eligible_rotating = true;
      tech_items_.push_back(item);
    }

    // Fixed supplies
    arrakis_ = AggregatedItemStats{kCardArrakisLiaison, "Arrakis Liaison", 2, false};
    tsmf_ = AggregatedItemStats{kCardTheSpiceMustFlow, "The Spice Must Flow", 9, false};
    reclaimed_ = AggregatedItemStats{0, "Reclaimed Forces", 3, false};

    if (total_games <= 0) {
      open_spiel::SpielFatalError(absl::StrFormat("Invalid total_games in Aggregate: %d", total_games));
    }

    // Validate episode IDs: non-negative and strictly unique
    std::vector<int> seen_episodes;
    seen_episodes.reserve(all_games.size());
    for (const auto& game : all_games) {
      if (game.episode_id < 0) {
        open_spiel::SpielFatalError("Negative or uninitialized episode_id encountered in Aggregate");
      }
      seen_episodes.push_back(game.episode_id);
    }
    std::sort(seen_episodes.begin(), seen_episodes.end());
    for (size_t i = 1; i < seen_episodes.size(); ++i) {
      if (seen_episodes[i] == seen_episodes[i - 1]) {
        open_spiel::SpielFatalError(absl::StrFormat(
            "Duplicate episode_id %d encountered in market diagnostics Aggregate", seen_episodes[i]));
      }
    }

    // 4. Accumulate per-game records with strict per-game invariant validation
    for (const auto& game : all_games) {
      // Per-game Imperium Row validation
      for (int i = 0; i < kNumImperiumCards; ++i) {
        if (game.imperium_purchased.test(i)) {
          if (!game.imperium_had_legal.test(i) || !game.imperium_visible.test(i) ||
              game.imperium_purchases[i] == 0 || game.imperium_legal_opps[i] == 0) {
            open_spiel::SpielFatalError(absl::StrFormat(
                "Per-game invariant violation: episode %d purchased imperium card %d without legal opp or visibility",
                game.episode_id, i));
          }
        }
        if (game.imperium_purchases[i] > game.imperium_legal_opps[i]) {
          open_spiel::SpielFatalError(absl::StrFormat(
              "Per-game invariant violation: episode %d imperium card %d purchases (%d) > legal opps (%d)",
              game.episode_id, i, game.imperium_purchases[i], game.imperium_legal_opps[i]));
        }
        int rsum = 0;
        for (int r = 1; r < kMaxTrackedRounds; ++r) {
          rsum += game.imperium_purchase_rounds[i][r];
        }
        if (rsum != game.imperium_purchases[i]) {
          open_spiel::SpielFatalError(absl::StrFormat(
              "Per-game invariant violation: episode %d imperium card %d purchase round sum (%d) != purchases (%d)",
              game.episode_id, i, rsum, game.imperium_purchases[i]));
        }
      }

      // Per-game Tleilaxu Row validation
      for (int i = 1; i < kNumTleilaxuCards; ++i) {
        if (game.tleilaxu_purchased.test(i)) {
          if (!game.tleilaxu_had_legal.test(i) || !game.tleilaxu_visible.test(i) ||
              game.tleilaxu_purchases[i] == 0 || game.tleilaxu_legal_opps[i] == 0) {
            open_spiel::SpielFatalError(absl::StrFormat(
                "Per-game invariant violation: episode %d purchased tleilaxu card %d without legal opp or visibility",
                game.episode_id, i));
          }
        }
        if (game.tleilaxu_purchases[i] > game.tleilaxu_legal_opps[i]) {
          open_spiel::SpielFatalError(absl::StrFormat(
              "Per-game invariant violation: episode %d tleilaxu card %d purchases (%d) > legal opps (%d)",
              game.episode_id, i, game.tleilaxu_purchases[i], game.tleilaxu_legal_opps[i]));
        }
        int rsum = 0;
        for (int r = 1; r < kMaxTrackedRounds; ++r) {
          rsum += game.tleilaxu_purchase_rounds[i][r];
        }
        if (rsum != game.tleilaxu_purchases[i]) {
          open_spiel::SpielFatalError(absl::StrFormat(
              "Per-game invariant violation: episode %d tleilaxu card %d purchase round sum (%d) != purchases (%d)",
              game.episode_id, i, rsum, game.tleilaxu_purchases[i]));
        }
      }

      // Per-game Tech Tiles validation
      for (int i = 0; i < kNumTechTiles; ++i) {
        if (game.tech_purchased.test(i)) {
          if (!game.tech_had_legal.test(i) || !game.tech_visible.test(i) ||
              game.tech_purchases[i] == 0 || game.tech_legal_opps[i] == 0) {
            open_spiel::SpielFatalError(absl::StrFormat(
                "Per-game invariant violation: episode %d purchased tech tile %d without legal opp or visibility",
                game.episode_id, i));
          }
        }
        if (game.tech_purchases[i] > game.tech_legal_opps[i]) {
          open_spiel::SpielFatalError(absl::StrFormat(
              "Per-game invariant violation: episode %d tech tile %d purchases (%d) > legal opps (%d)",
              game.episode_id, i, game.tech_purchases[i], game.tech_legal_opps[i]));
        }
        int rsum = 0;
        for (int r = 1; r < kMaxTrackedRounds; ++r) {
          rsum += game.tech_purchase_rounds[i][r];
        }
        if (rsum != game.tech_purchases[i]) {
          open_spiel::SpielFatalError(absl::StrFormat(
              "Per-game invariant violation: episode %d tech tile %d purchase round sum (%d) != purchases (%d)",
              game.episode_id, i, rsum, game.tech_purchases[i]));
        }
      }

      // Per-game fixed supplies validation
      auto check_fixed_game = [&](const FixedSupplyPerGame& f, const std::string& name) {
        if (f.purchased) {
          if (!f.had_legal || !f.visible || f.purchases == 0 || f.legal_opps == 0) {
            open_spiel::SpielFatalError(absl::StrFormat(
                "Per-game invariant violation: episode %d purchased fixed %s without legal opp or visibility",
                game.episode_id, name));
          }
        }
        if (f.purchases > f.legal_opps) {
          open_spiel::SpielFatalError(absl::StrFormat(
              "Per-game invariant violation: episode %d fixed %s purchases (%d) > legal opps (%d)",
              game.episode_id, name, f.purchases, f.legal_opps));
        }
        int rsum = 0;
        for (int r = 1; r < kMaxTrackedRounds; ++r) {
          rsum += f.purchase_rounds[r];
        }
        if (rsum != f.purchases) {
          open_spiel::SpielFatalError(absl::StrFormat(
              "Per-game invariant violation: episode %d fixed %s purchase round sum (%d) != purchases (%d)",
              game.episode_id, name, rsum, f.purchases));
        }
      };
      check_fixed_game(game.arrakis_liaison, "arrakis_liaison");
      check_fixed_game(game.spice_must_flow, "the_spice_must_flow");
      check_fixed_game(game.reclaimed_forces, "reclaimed_forces");

      // Imperium Row accumulation
      for (int i = 0; i < kNumImperiumCards; ++i) {
        if (game.imperium_visible.test(i)) imperium_items_[i].games_visible++;
        if (game.imperium_had_legal.test(i)) imperium_items_[i].games_legal_opportunity++;
        if (game.imperium_purchased.test(i)) imperium_items_[i].games_purchased++;
        imperium_items_[i].legal_decision_opportunities += game.imperium_legal_opps[i];
        imperium_items_[i].purchases += game.imperium_purchases[i];
        for (int r = 0; r < kMaxTrackedRounds; ++r) {
          imperium_items_[i].purchase_rounds[r] += game.imperium_purchase_rounds[i][r];
        }
      }

      // Tleilaxu Row accumulation
      for (int i = 1; i < kNumTleilaxuCards; ++i) {
        int idx = i - 1;
        if (game.tleilaxu_visible.test(i)) tleilaxu_items_[idx].games_visible++;
        if (game.tleilaxu_had_legal.test(i)) tleilaxu_items_[idx].games_legal_opportunity++;
        if (game.tleilaxu_purchased.test(i)) tleilaxu_items_[idx].games_purchased++;
        tleilaxu_items_[idx].legal_decision_opportunities += game.tleilaxu_legal_opps[i];
        tleilaxu_items_[idx].purchases += game.tleilaxu_purchases[i];
        for (int r = 0; r < kMaxTrackedRounds; ++r) {
          tleilaxu_items_[idx].purchase_rounds[r] += game.tleilaxu_purchase_rounds[i][r];
        }
      }

      // Tech Tiles accumulation
      for (int i = 0; i < kNumTechTiles; ++i) {
        if (game.tech_visible.test(i)) tech_items_[i].games_visible++;
        if (game.tech_had_legal.test(i)) tech_items_[i].games_legal_opportunity++;
        if (game.tech_purchased.test(i)) tech_items_[i].games_purchased++;
        tech_items_[i].legal_decision_opportunities += game.tech_legal_opps[i];
        tech_items_[i].purchases += game.tech_purchases[i];
        for (int r = 0; r < kMaxTrackedRounds; ++r) {
          tech_items_[i].purchase_rounds[r] += game.tech_purchase_rounds[i][r];
        }
      }

      // Fixed supplies accumulation
      if (game.arrakis_liaison.visible) arrakis_.games_visible++;
      if (game.arrakis_liaison.had_legal) arrakis_.games_legal_opportunity++;
      if (game.arrakis_liaison.purchased) arrakis_.games_purchased++;
      arrakis_.legal_decision_opportunities += game.arrakis_liaison.legal_opps;
      arrakis_.purchases += game.arrakis_liaison.purchases;
      for (int r = 0; r < kMaxTrackedRounds; ++r) {
        arrakis_.purchase_rounds[r] += game.arrakis_liaison.purchase_rounds[r];
      }

      if (game.spice_must_flow.visible) tsmf_.games_visible++;
      if (game.spice_must_flow.had_legal) tsmf_.games_legal_opportunity++;
      if (game.spice_must_flow.purchased) tsmf_.games_purchased++;
      tsmf_.legal_decision_opportunities += game.spice_must_flow.legal_opps;
      tsmf_.purchases += game.spice_must_flow.purchases;
      for (int r = 0; r < kMaxTrackedRounds; ++r) {
        tsmf_.purchase_rounds[r] += game.spice_must_flow.purchase_rounds[r];
      }

      if (game.reclaimed_forces.visible) reclaimed_.games_visible++;
      if (game.reclaimed_forces.had_legal) reclaimed_.games_legal_opportunity++;
      if (game.reclaimed_forces.purchased) reclaimed_.games_purchased++;
      reclaimed_.legal_decision_opportunities += game.reclaimed_forces.legal_opps;
      reclaimed_.purchases += game.reclaimed_forces.purchases;
      for (int r = 0; r < kMaxTrackedRounds; ++r) {
        reclaimed_.purchase_rounds[r] += game.reclaimed_forces.purchase_rounds[r];
      }

      // Special acquisition routes accumulation
      for (int i = 0; i < kNumImperiumCards; ++i) {
        special_routes_.bypass_free_opps[i] += game.bypass_free_opps[i];
        special_routes_.bypass_free_acquisitions[i] += game.bypass_free_acquisitions[i];
        special_routes_.bypass_paid_opps[i] += game.bypass_paid_opps[i];
        special_routes_.bypass_paid_acquisitions[i] += game.bypass_paid_acquisitions[i];
        special_routes_.armand_opps[i] += game.armand_opps[i];
        special_routes_.armand_acquisitions[i] += game.armand_acquisitions[i];
        special_routes_.tleilaxu_master_opps[i] += game.tleilaxu_master_opps[i];
        special_routes_.tleilaxu_master_acquisitions[i] += game.tleilaxu_master_acquisitions[i];
        special_routes_.helena_removals[i] += game.helena_removals[i];
        special_routes_.helena_reserve_purchases[i] += game.helena_reserve_purchases[i];
      }
      special_routes_.family_atomics_uses += game.family_atomics_uses;
    }

    // 5. Compute conversion rates and run fail-closed validation checks
    auto finalize_item = [total_games](AggregatedItemStats& item) {
      if (item.games_legal_opportunity > 0) {
        item.conversion_rate = static_cast<double>(item.games_purchased) /
                               static_cast<double>(item.games_legal_opportunity);
      } else {
        item.conversion_rate = -1.0;
      }
      int round_sum = 0;
      for (int r = 1; r < kMaxTrackedRounds; ++r) {
        round_sum += item.purchase_rounds[r];
      }
      // Strict fail-closed invariant check:
      // 1. games_purchased <= games_legal_opportunity <= games_visible <= total_games
      // 2. purchases <= legal_decision_opportunities
      // 3. round_sum == purchases
      if (item.games_purchased > item.games_legal_opportunity ||
          item.games_legal_opportunity > item.games_visible ||
          item.games_visible > total_games ||
          item.purchases > item.legal_decision_opportunities ||
          round_sum != item.purchases) {
        open_spiel::SpielFatalError(absl::StrFormat(
            "Aggregate invariant violation for item %s (ID %d): "
            "purchased_games=%d, legal_opp_games=%d, visible_games=%d, total_games=%d, "
            "purchases=%d, legal_opps=%d, round_sum=%d",
            item.name, item.id, item.games_purchased, item.games_legal_opportunity,
            item.games_visible, total_games, item.purchases, item.legal_decision_opportunities,
            round_sum));
      }
    };

    for (auto& item : imperium_items_) finalize_item(item);
    for (auto& item : tleilaxu_items_) finalize_item(item);
    for (auto& item : tech_items_) finalize_item(item);
    finalize_item(arrakis_);
    finalize_item(tsmf_);
    finalize_item(reclaimed_);
  }

  static MarketDiagnosticsReport AggregateReport(
      const std::vector<CompactPerGameMarketDiagnostics>& all_games, int total_games) {
    MarketDiagnosticsReport report;
    report.Aggregate(all_games, total_games);
    return report;
  }

  // Export full JSON diagnostics
  void WriteJson(std::ostream& os) const {
    auto format_item_json = [](const AggregatedItemStats& item, bool eligible) -> std::string {
      std::ostringstream ss;
      ss << "      {\n"
         << "        \"id\": " << item.id << ",\n"
         << "        \"name\": \"" << item.name << "\",\n"
         << "        \"cost\": " << item.cost << ",\n"
         << "        \"eligible_rotating\": " << (eligible ? "true" : "false") << ",\n"
         << "        \"games_visible\": " << item.games_visible << ",\n"
         << "        \"games_legal_opportunity\": " << item.games_legal_opportunity << ",\n"
         << "        \"legal_decision_opportunities\": " << item.legal_decision_opportunities << ",\n"
         << "        \"purchases\": " << item.purchases << ",\n"
         << "        \"games_purchased\": " << item.games_purchased << ",\n"
         << "        \"conversion_rate\": ";
      if (item.conversion_rate >= 0.0) {
        ss << absl::StrFormat("%.6f", item.conversion_rate);
      } else {
        ss << "null";
      }
      ss << ",\n        \"purchase_rounds\": [";
      for (int r = 1; r <= 10; ++r) {
        if (r > 1) ss << ", ";
        ss << item.purchase_rounds[r];
      }
      ss << "]\n      }";
      return ss.str();
    };

    os << "{\n"
       << "  \"total_games\": " << total_games_ << ",\n"
       << "  \"imperium_row\": [\n";
    bool first = true;
    for (const auto& item : imperium_items_) {
      if (!first) os << ",\n";
      first = false;
      os << format_item_json(item, item.eligible_rotating);
    }
    os << "\n  ],\n"
       << "  \"tleilaxu_row\": [\n";
    first = true;
    for (const auto& item : tleilaxu_items_) {
      if (!first) os << ",\n";
      first = false;
      os << format_item_json(item, true);
    }
    os << "\n  ],\n"
       << "  \"tech_tiles\": [\n";
    first = true;
    for (const auto& item : tech_items_) {
      if (!first) os << ",\n";
      first = false;
      os << format_item_json(item, true);
    }
    os << "\n  ],\n"
       << "  \"fixed_supplies\": [\n"
       << format_item_json(arrakis_, false) << ",\n"
       << format_item_json(tsmf_, false) << ",\n"
       << format_item_json(reclaimed_, false) << "\n"
       << "  ],\n"
       << "  \"special_routes\": {\n"
       << "    \"family_atomics_uses\": " << special_routes_.family_atomics_uses << ",\n"
       << "    \"events\": [\n";
    bool first_sp = true;
    for (int i = 0; i < kNumImperiumCards; ++i) {
      if (special_routes_.bypass_free_opps[i] == 0 &&
          special_routes_.bypass_free_acquisitions[i] == 0 &&
          special_routes_.bypass_paid_opps[i] == 0 &&
          special_routes_.bypass_paid_acquisitions[i] == 0 &&
          special_routes_.armand_opps[i] == 0 &&
          special_routes_.armand_acquisitions[i] == 0 &&
          special_routes_.tleilaxu_master_opps[i] == 0 &&
          special_routes_.tleilaxu_master_acquisitions[i] == 0 &&
          special_routes_.helena_removals[i] == 0 &&
          special_routes_.helena_reserve_purchases[i] == 0) {
        continue;
      }
      if (!first_sp) os << ",\n";
      first_sp = false;
      os << "      {\"id\": " << i << ", \"name\": \"" << kImperiumCards[i].name << "\""
         << ", \"bypass_free_opps\": " << special_routes_.bypass_free_opps[i]
         << ", \"bypass_free_acquisitions\": " << special_routes_.bypass_free_acquisitions[i]
         << ", \"bypass_paid_opps\": " << special_routes_.bypass_paid_opps[i]
         << ", \"bypass_paid_acquisitions\": " << special_routes_.bypass_paid_acquisitions[i]
         << ", \"armand_opps\": " << special_routes_.armand_opps[i]
         << ", \"armand_acquisitions\": " << special_routes_.armand_acquisitions[i]
         << ", \"tleilaxu_master_opps\": " << special_routes_.tleilaxu_master_opps[i]
         << ", \"tleilaxu_master_acquisitions\": " << special_routes_.tleilaxu_master_acquisitions[i]
         << ", \"helena_removals\": " << special_routes_.helena_removals[i]
         << ", \"helena_reserve_purchases\": " << special_routes_.helena_reserve_purchases[i]
         << "}";
    }
    os << "\n    ]\n  }\n"
       << "}\n";
  }

  bool WriteJson(const std::string& path) const {
    std::ofstream os(path);
    if (!os.is_open()) return false;
    WriteJson(os);
    return true;
  }

  // Export full CSV diagnostics
  void WriteCsv(std::ostream& os) const {
    os << "market,id,name,cost,eligible_rotating,games_visible,games_legal_opportunity,legal_decision_opportunities,purchases,games_purchased,purchase_conversion,r1,r2,r3,r4,r5,r6,r7,r8,r9,r10\n";
    auto write_item_csv = [&os](const std::string& market, const AggregatedItemStats& item, bool eligible) {
      os << market << ","
         << item.id << ",\""
         << item.name << "\","
         << item.cost << ","
         << (eligible ? "true" : "false") << ","
         << item.games_visible << ","
         << item.games_legal_opportunity << ","
         << item.legal_decision_opportunities << ","
         << item.purchases << ","
         << item.games_purchased << ",";
      if (item.conversion_rate >= 0.0) {
        os << absl::StrFormat("%.4f", item.conversion_rate);
      } else {
        os << "N/A";
      }
      for (int r = 1; r <= 10; ++r) {
        os << "," << item.purchase_rounds[r];
      }
      os << "\n";
    };

    for (const auto& item : imperium_items_) write_item_csv("imperium_row", item, item.eligible_rotating);
    for (const auto& item : tleilaxu_items_) write_item_csv("tleilaxu_row", item, true);
    for (const auto& item : tech_items_) write_item_csv("tech_tiles", item, true);
    write_item_csv("fixed_supply", arrakis_, false);
    write_item_csv("fixed_supply", tsmf_, false);
    write_item_csv("fixed_supply", reclaimed_, false);
  }

  bool WriteCsv(const std::string& path) const {
    std::ofstream os(path);
    if (!os.is_open()) return false;
    WriteCsv(os);
    return true;
  }

  int GetTotalGames() const { return total_games_; }
  const std::vector<AggregatedItemStats>& GetImperiumItems() const { return imperium_items_; }
  const std::vector<AggregatedItemStats>& GetTleilaxuItems() const { return tleilaxu_items_; }
  const std::vector<AggregatedItemStats>& GetTechItems() const { return tech_items_; }
  const AggregatedItemStats& GetArrakis() const { return arrakis_; }
  const AggregatedItemStats& GetTsmf() const { return tsmf_; }
  const AggregatedItemStats& GetReclaimed() const { return reclaimed_; }

  struct SpecialRouteAggregates {
    std::array<int, kNumImperiumCards> bypass_free_opps{};
    std::array<int, kNumImperiumCards> bypass_free_acquisitions{};
    std::array<int, kNumImperiumCards> bypass_paid_opps{};
    std::array<int, kNumImperiumCards> bypass_paid_acquisitions{};
    std::array<int, kNumImperiumCards> armand_opps{};
    std::array<int, kNumImperiumCards> armand_acquisitions{};
    std::array<int, kNumImperiumCards> tleilaxu_master_opps{};
    std::array<int, kNumImperiumCards> tleilaxu_master_acquisitions{};
    std::array<int, kNumImperiumCards> helena_removals{};
    std::array<int, kNumImperiumCards> helena_reserve_purchases{};
    int family_atomics_uses = 0;
  };

  const SpecialRouteAggregates& GetSpecialRoutes() const { return special_routes_; }

 private:
  int total_games_ = 0;
  std::vector<AggregatedItemStats> imperium_items_;
  std::vector<AggregatedItemStats> tleilaxu_items_;
  std::vector<AggregatedItemStats> tech_items_;
  AggregatedItemStats arrakis_;
  AggregatedItemStats tsmf_;
  AggregatedItemStats reclaimed_;
  SpecialRouteAggregates special_routes_;
};

inline bool WritePerGameJsonl(
    const std::string& path,
    const std::vector<CompactPerGameMarketDiagnostics>& all_games) {
  std::ofstream os(path);
  if (!os.is_open()) return false;
  for (const auto& game : all_games) {
    if (game.episode_id < 0) continue;
    os << game.ToJson().dump() << "\n";
  }
  return os.good();
}

inline std::vector<CompactPerGameMarketDiagnostics> ReadPerGameJsonl(const std::string& path) {
  std::ifstream is(path);
  if (!is.is_open()) {
    open_spiel::SpielFatalError("Cannot open per-game JSONL file for reading: " + path);
  }
  std::vector<CompactPerGameMarketDiagnostics> all_games;
  std::string line;
  while (std::getline(is, line)) {
    if (line.empty()) continue;
    nlohmann::json j = nlohmann::json::parse(line);
    all_games.push_back(CompactPerGameMarketDiagnostics::FromJson(j));
  }
  return all_games;
}

inline void VerifyReconciliation(const MarketDiagnosticsReport& a, const MarketDiagnosticsReport& b) {
  if (a.GetTotalGames() != b.GetTotalGames()) {
    open_spiel::SpielFatalError(absl::StrFormat(
        "Reconciliation failure: total_games mismatch (%d vs %d)",
        a.GetTotalGames(), b.GetTotalGames()));
  }

  auto check_items = [](const std::vector<AggregatedItemStats>& list_a,
                        const std::vector<AggregatedItemStats>& list_b,
                        const std::string& category) {
    if (list_a.size() != list_b.size()) {
      open_spiel::SpielFatalError(absl::StrFormat(
          "Reconciliation failure: %s item count mismatch (%zu vs %zu)",
          category, list_a.size(), list_b.size()));
    }
    for (size_t i = 0; i < list_a.size(); ++i) {
      const auto& ia = list_a[i];
      const auto& ib = list_b[i];
      if (ia.id != ib.id ||
          ia.games_visible != ib.games_visible ||
          ia.games_legal_opportunity != ib.games_legal_opportunity ||
          ia.legal_decision_opportunities != ib.legal_decision_opportunities ||
          ia.purchases != ib.purchases ||
          ia.games_purchased != ib.games_purchased ||
          ia.purchase_rounds != ib.purchase_rounds) {
        open_spiel::SpielFatalError(absl::StrFormat(
            "Reconciliation failure in %s item %s (ID %d): "
            "orig: vis=%d, had_leg=%d, opps=%d, buys=%d, games_bought=%d; "
            "recomputed: vis=%d, had_leg=%d, opps=%d, buys=%d, games_bought=%d",
            category, ia.name, ia.id,
            ia.games_visible, ia.games_legal_opportunity, ia.legal_decision_opportunities,
            ia.purchases, ia.games_purchased,
            ib.games_visible, ib.games_legal_opportunity, ib.legal_decision_opportunities,
            ib.purchases, ib.games_purchased));
      }
    }
  };

  check_items(a.GetImperiumItems(), b.GetImperiumItems(), "ImperiumRow");
  check_items(a.GetTleilaxuItems(), b.GetTleilaxuItems(), "TleilaxuRow");
  check_items(a.GetTechItems(), b.GetTechItems(), "TechTiles");

  auto check_fixed = [](const AggregatedItemStats& ia, const AggregatedItemStats& ib, const std::string& name) {
    if (ia.games_visible != ib.games_visible ||
        ia.games_legal_opportunity != ib.games_legal_opportunity ||
        ia.legal_decision_opportunities != ib.legal_decision_opportunities ||
        ia.purchases != ib.purchases ||
        ia.games_purchased != ib.games_purchased ||
        ia.purchase_rounds != ib.purchase_rounds) {
      open_spiel::SpielFatalError(absl::StrFormat("Reconciliation failure in fixed item %s", name));
    }
  };
  check_fixed(a.GetArrakis(), b.GetArrakis(), "Arrakis");
  check_fixed(a.GetTsmf(), b.GetTsmf(), "TheSpiceMustFlow");
  check_fixed(a.GetReclaimed(), b.GetReclaimed(), "ReclaimedForces");

  const auto& sa = a.GetSpecialRoutes();
  const auto& sb = b.GetSpecialRoutes();
  if (sa.family_atomics_uses != sb.family_atomics_uses ||
      sa.bypass_free_opps != sb.bypass_free_opps ||
      sa.bypass_free_acquisitions != sb.bypass_free_acquisitions ||
      sa.bypass_paid_opps != sb.bypass_paid_opps ||
      sa.bypass_paid_acquisitions != sb.bypass_paid_acquisitions ||
      sa.armand_opps != sb.armand_opps ||
      sa.armand_acquisitions != sb.armand_acquisitions ||
      sa.tleilaxu_master_opps != sb.tleilaxu_master_opps ||
      sa.tleilaxu_master_acquisitions != sb.tleilaxu_master_acquisitions ||
      sa.helena_removals != sb.helena_removals ||
      sa.helena_reserve_purchases != sb.helena_reserve_purchases) {
    open_spiel::SpielFatalError("Reconciliation failure in special routes");
  }
}

}  // namespace dune_market_diag
}  // namespace open_spiel

#endif  // OPEN_SPIEL_EXAMPLES_DUNE_MARKET_DIAGNOSTICS_H_
