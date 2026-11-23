#ifndef OPEN_SPIEL_EXAMPLES_DUNE_MARKET_DIAGNOSTICS_TYPES_H_
#define OPEN_SPIEL_EXAMPLES_DUNE_MARKET_DIAGNOSTICS_TYPES_H_

#include <array>
#include <bitset>
#include <cstdint>
#include <string>

#include "open_spiel/games/dune_imperium/dune_imperium.h"
#include "open_spiel/games/dune_imperium/dune_imperium_cards.h"
#include "open_spiel/games/dune_imperium/dune_imperium_common.h"
#include "open_spiel/games/dune_imperium/dune_imperium_content.h"
#include "open_spiel/games/dune_imperium/dune_imperium_content_generated.h"
#include "open_spiel/json/include/nlohmann/json.hpp"
#include "open_spiel/spiel.h"

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

}  // namespace dune_market_diag
}  // namespace open_spiel

#endif  // OPEN_SPIEL_EXAMPLES_DUNE_MARKET_DIAGNOSTICS_TYPES_H_
