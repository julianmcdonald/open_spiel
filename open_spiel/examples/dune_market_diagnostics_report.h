#ifndef OPEN_SPIEL_EXAMPLES_DUNE_MARKET_DIAGNOSTICS_REPORT_H_
#define OPEN_SPIEL_EXAMPLES_DUNE_MARKET_DIAGNOSTICS_REPORT_H_

#include <algorithm>
#include <array>
#include <cstddef>
#include <fstream>
#include <ostream>
#include <sstream>
#include <string>
#include <vector>

#include "open_spiel/abseil-cpp/absl/strings/str_format.h"
#include "open_spiel/examples/dune_market_diagnostics_types.h"
#include "open_spiel/games/dune_imperium/dune_imperium_content.h"
#include "open_spiel/json/include/nlohmann/json.hpp"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace dune_market_diag {

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

#endif  // OPEN_SPIEL_EXAMPLES_DUNE_MARKET_DIAGNOSTICS_REPORT_H_
