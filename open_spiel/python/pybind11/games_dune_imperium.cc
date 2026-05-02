// Copyright 2026
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "open_spiel/python/pybind11/games_dune_imperium.h"

#include <vector>

#include "open_spiel/games/dune_imperium/dune_imperium.h"
#include "open_spiel/games/dune_imperium/dune_imperium_cards.h"
#include "open_spiel/games/dune_imperium/dune_imperium_common.h"
#include "open_spiel/games/dune_imperium/dune_imperium_content.h"
#include "open_spiel/games/dune_imperium/dune_imperium_util.h"
#include "open_spiel/spiel.h"


namespace py = ::pybind11;

using open_spiel::State;

using open_spiel::dune_imperium::DuneImperiumState;
using open_spiel::dune_imperium::Faction;
using open_spiel::dune_imperium::GamePhase;
using open_spiel::dune_imperium::ImperiumCard;
using open_spiel::dune_imperium::LeaderId;

void open_spiel::init_pyspiel_games_dune_imperium(py::module &m) {
  py::module_ di = m.def_submodule("dune_imperium");

  // ---- Enums ----

  // --- Structs & Helpers ---
  py::class_<open_spiel::dune_imperium::IntrigueCard>(di, "IntrigueCard")
      .def_readonly("id", &open_spiel::dune_imperium::IntrigueCard::id)
      .def_readonly("name", &open_spiel::dune_imperium::IntrigueCard::name)
      .def_readonly(
          "combat_strength_bonus",
          &open_spiel::dune_imperium::IntrigueCard::combat_strength_bonus);

  di.def(
      "get_intrigue_card",
      [](int id) -> const open_spiel::dune_imperium::IntrigueCard * {
        return open_spiel::dune_imperium::FindIntrigueCardById(id);
      },
      py::arg("id"), py::return_value_policy::reference);

  py::class_<ImperiumCard>(di, "ImperiumCard")
      .def_readonly("id", &ImperiumCard::id)
      .def_readonly("name", &ImperiumCard::name)
      .def_readonly("reveal_swords", &ImperiumCard::reveal_swords)
      .def_readonly("reveal_troops", &ImperiumCard::reveal_troops)
      .def_readonly("agent_troops", &ImperiumCard::agent_troops);

  di.def(
      "get_imperium_card",
      [](int id) -> const ImperiumCard * {
        return open_spiel::dune_imperium::FindImperiumCardById(id);
      },
      py::arg("id"), py::return_value_policy::reference);
  py::enum_<GamePhase>(di, "GamePhase")
      .value("LEADER_OFFER_CHANCE", GamePhase::kLeaderOfferChance)
      .value("LEADER_DRAFT", GamePhase::kLeaderDraft)
      .value("DEAL", GamePhase::kDeal)
      .value("ROUND_START", GamePhase::kRoundStart)
      .value("AGENT_TURNS", GamePhase::kAgentTurns)
      .value("REVEAL_TURNS", GamePhase::kRevealTurns)
      .value("COMBAT", GamePhase::kCombat)
      .value("MAKERS", GamePhase::kMakers)
      .value("RECALL", GamePhase::kRecall)
      .value("TERMINAL", GamePhase::kTerminal)
      .export_values();

  py::enum_<Faction>(di, "Faction")
      .value("EMPEROR", Faction::kEmperor)
      .value("SPACING_GUILD", Faction::kSpacingGuild)
      .value("BENE_GESSERIT", Faction::kBeneGesserit)
      .value("FREMEN", Faction::kFremen)
      .export_values();

  py::enum_<LeaderId>(di, "LeaderId")
      .value("ARMAND_ECAZ", LeaderId::kArmandEcaz)
      .value("VLADIMIR_HARKONNEN", LeaderId::kVladimirHarkonnen)
      .value("ILBAN_RICHESE", LeaderId::kIlbanRichese)
      .value("ARIANA_THORVALD", LeaderId::kArianaThorvald)
      .value("LETO_ATREIDES", LeaderId::kLetoAtreides)
      .value("MEMNON_THORVALD", LeaderId::kMemnonThorvald)
      .value("GLOSSU_RABBAN", LeaderId::kGlossuRabban)
      .value("HELENA_RICHESE", LeaderId::kHelenaRichese)
      .value("ILESA_ECAZ", LeaderId::kIlesaEcaz)
      .value("PAUL_ATREIDES", LeaderId::kPaulAtreides)
      .value("RHOMBUR_VERNIUS", LeaderId::kRhomburVernius)
      .value("YUNA_MORITANI", LeaderId::kYunaMoritani)
      .value("TESSIA_VERNIUS", LeaderId::kTessiaVernius)
      .value("HUNDRO_MORITANI", LeaderId::kHundroMoritani)
      .export_values();

  // ---- DuneImperiumState ----

  py::classh<DuneImperiumState, State>(m, "DuneImperiumState")

      // --- Setters (state injection) ---
      .def("set_player_hand", &DuneImperiumState::SetPlayerHandForTesting,
           py::arg("player"), py::arg("hand"))
      .def("set_player_deck", &DuneImperiumState::SetPlayerDeckForTesting,
           py::arg("player"), py::arg("deck"))
      .def("set_player_draw_deck",
           &DuneImperiumState::SetPlayerDrawDeckForTesting, py::arg("player"),
           py::arg("deck"))
      .def("set_imperium_row", &DuneImperiumState::SetImperiumRowForTesting,
           py::arg("row"))
      .def("set_tleilaxu_row", &DuneImperiumState::SetTleilaxuRowForTesting,
           py::arg("row"))
      .def("set_imperium_draw_deck",
           &DuneImperiumState::SetImperiumDrawDeckForTesting, py::arg("deck"))
      .def("set_phase", &DuneImperiumState::SetPhaseForTesting,
           py::arg("phase"))
      .def("set_current_player", &DuneImperiumState::SetCurrentPlayerForTesting,
           py::arg("player"))
      .def("set_round_number", &DuneImperiumState::SetRoundNumberForTesting,
           py::arg("round"))
      .def("set_leader", &DuneImperiumState::SetLeaderForTesting,
           py::arg("player"), py::arg("leader_id"))
      .def("set_player_intrigues",
           &DuneImperiumState::SetPlayerIntriguesForTesting, py::arg("player"),
           py::arg("intrigues"))
      .def("set_player_intrigue_hand",
           &DuneImperiumState::SetPlayerIntrigueHandForTesting,
           py::arg("player"), py::arg("cards"))
      .def("set_intrigue_draw_deck",
           &DuneImperiumState::SetIntrigueDrawDeckForTesting, py::arg("deck"))
      .def("set_player_spice", &DuneImperiumState::SetPlayerSpiceForTesting,
           py::arg("player"), py::arg("spice"))
      .def("set_player_solari", &DuneImperiumState::SetPlayerSolariForTesting,
           py::arg("player"), py::arg("solari"))
      .def("set_player_water", &DuneImperiumState::SetPlayerWaterForTesting,
           py::arg("player"), py::arg("water"))
      .def("set_player_vp", &DuneImperiumState::SetPlayerVpForTesting,
           py::arg("player"), py::arg("vp"))
      .def("set_player_persuasion",
           &DuneImperiumState::SetPlayerPersuasionForTesting, py::arg("player"),
           py::arg("persuasion"))
      .def("set_player_influence",
           &DuneImperiumState::SetPlayerInfluenceForTesting, py::arg("player"),
           py::arg("faction"), py::arg("influence"))
      .def("set_vladimir_secret_factions",
           &DuneImperiumState::SetVladimirSecretFactionsForTesting,
           py::arg("player"), py::arg("first"), py::arg("second"))
      .def("set_player_troops_in_garrison",
           &DuneImperiumState::SetPlayerTroopsInGarrisonForTesting,
           py::arg("player"), py::arg("amount"))
      .def("set_troops", &DuneImperiumState::SetTroopsForTesting,
           py::arg("player"), py::arg("garrison"), py::arg("combat"))
      .def("set_player_dreadnoughts_in_garrison",
           &DuneImperiumState::SetPlayerDreadnoughtsInGarrisonForTesting,
           py::arg("player"), py::arg("amount"))
      .def("set_agent_space_owner",
           &DuneImperiumState::SetAgentSpaceOwnerForTesting,
           py::arg("board_index"), py::arg("player"))
      .def("set_high_council_owned",
           &DuneImperiumState::SetHighCouncilOwnedForTesting, py::arg("player"),
           py::arg("owned"))
      .def("set_swordmaster", &DuneImperiumState::SetSwordmasterForTesting,
           py::arg("player"), py::arg("owned"))
      .def("gain_tech_tile", &DuneImperiumState::GainTechTileForTesting,
           py::arg("player"), py::arg("tech_id"))
      .def("set_tech_tile_owned",
           &DuneImperiumState::SetTechTileOwnedForTesting, py::arg("player"),
           py::arg("tech_id"))
      .def("set_tech_tiles_owned",
           &DuneImperiumState::SetTechTilesOwnedForTesting, py::arg("player"),
           py::arg("tiles"))
      .def("set_tech_market", &DuneImperiumState::SetTechMarketForTesting,
           py::arg("slot"), py::arg("tech_id"))
      .def("set_tech_stack", &DuneImperiumState::SetTechStackForTesting,
           py::arg("stack"), py::arg("tiles"))
      .def("set_bonus_spice", &DuneImperiumState::SetBonusSpiceForTesting,
           py::arg("space_action"), py::arg("amount"))
      .def("set_player_agents_remaining",
           &DuneImperiumState::SetPlayerAgentsRemainingForTesting,
           py::arg("player"), py::arg("agents"))
      .def("set_pending_reward_card_draws",
           &DuneImperiumState::SetPendingRewardCardDrawsForTesting,
           py::arg("player"), py::arg("amount"))
      .def("set_player_shipping_level",
           &DuneImperiumState::SetPlayerShippingLevelForTesting,
           py::arg("player"), py::arg("level"))
      .def("set_alliance_owner", &DuneImperiumState::SetAllianceOwnerForTesting,
           py::arg("faction"), py::arg("player"))
      .def("set_tech_negotiators",
           &DuneImperiumState::SetTechNegotiatorsForTesting, py::arg("player"),
           py::arg("amount"))
      .def("set_specimens", &DuneImperiumState::SetSpecimensForTesting,
           py::arg("player"), py::arg("amount"))
      .def("set_current_conflict",
           &DuneImperiumState::SetCurrentConflictForTesting,
           py::arg("conflict_id"))
      .def("set_player_troops_in_combat",
           &DuneImperiumState::SetPlayerTroopsInCombatForTesting,
           py::arg("player"), py::arg("troops"))
      .def("set_player_dreadnoughts_in_combat",
           &DuneImperiumState::SetPlayerDreadnoughtsInCombatForTesting,
           py::arg("player"), py::arg("dreads"))
      .def("set_player_revealed",
           &DuneImperiumState::SetPlayerRevealedForTesting, py::arg("player"),
           py::arg("revealed"))
      .def("set_mentat_available",
           &DuneImperiumState::SetMentatAvailableForTesting, py::arg("val"))
      .def("set_atomics_used", &DuneImperiumState::SetAtomicsUsedForTesting,
           py::arg("player"), py::arg("used"))
      .def("set_player_research_bottom_track_pos",
           &DuneImperiumState::SetPlayerResearchBottomTrackPosForTesting,
           py::arg("player"), py::arg("col"), py::arg("row"))
      .def("set_reserve_supply", &DuneImperiumState::SetReserveSupplyForTesting,
           py::arg("tsmf"), py::arg("al"))

      // --- Getters (sync validation) ---
      .def("get_player_hand", &DuneImperiumState::GetPlayerHandForTesting,
           py::arg("player"), py::return_value_policy::reference_internal)
      .def("get_player_draw_deck",
           &DuneImperiumState::GetPlayerDrawDeckForTesting, py::arg("player"),
           py::return_value_policy::reference_internal)
      .def("get_player_discard", &DuneImperiumState::GetPlayerDiscardForTesting,
           py::arg("player"), py::return_value_policy::reference_internal)
      .def("set_player_discard", &DuneImperiumState::SetPlayerDiscardForTesting,
           py::arg("player"), py::arg("discard"))

      // --- Revealed zone injection (OCR bridge) ---
      .def("get_revealed_cards", &DuneImperiumState::GetRevealedCardsForTesting,
           py::arg("player"), py::return_value_policy::reference_internal)
      .def("set_revealed_cards", &DuneImperiumState::SetRevealedCardsForTesting,
           py::arg("player"), py::arg("cards"))

      // --- Deck Pool (opponent card probability tracking) ---
      .def("set_deck_pool_total_cards",
           &DuneImperiumState::SetDeckPoolTotalCardsForTesting,
           py::arg("player"), py::arg("cards"))
      .def("get_deck_pool_total_cards",
           &DuneImperiumState::GetDeckPoolTotalCardsForTesting,
           py::arg("player"))
      .def("set_deck_pool_discard",
           &DuneImperiumState::SetDeckPoolDiscardForTesting, py::arg("player"),
           py::arg("cards"))
      .def("get_deck_pool_discard",
           &DuneImperiumState::GetDeckPoolDiscardForTesting, py::arg("player"))
      .def("set_deck_pool_draw_deck",
           &DuneImperiumState::SetDeckPoolDrawDeckForTesting, py::arg("player"),
           py::arg("cards"))
      .def("get_deck_pool_draw_deck",
           &DuneImperiumState::GetDeckPoolDrawDeckForTesting, py::arg("player"))
      .def("set_deck_pool_cards_drawn",
           &DuneImperiumState::SetDeckPoolCardsDrawnForTesting,
           py::arg("player"), py::arg("count"))
      .def("get_deck_pool_cards_drawn",
           &DuneImperiumState::GetDeckPoolCardsDrawnForTesting,
           py::arg("player"))
      .def("get_imperium_row", &DuneImperiumState::GetImperiumRowForTesting,
           py::return_value_policy::reference_internal)
      .def("get_tleilaxu_row", &DuneImperiumState::GetTleilaxuRowForTesting,
           py::return_value_policy::reference_internal)
      .def("get_imperium_draw_deck",
           &DuneImperiumState::GetImperiumDrawDeckForTesting,
           py::return_value_policy::reference_internal)
      .def("get_imperium_discard",
           &DuneImperiumState::GetImperiumDiscardForTesting,
           py::return_value_policy::reference_internal)
      .def("get_intrigue_discard",
           &DuneImperiumState::GetIntrigueDiscardForTesting,
           py::return_value_policy::reference_internal)
      .def("set_intrigue_discard",
           &DuneImperiumState::SetIntrigueDiscardForTesting, py::arg("discard"))
      .def("get_intrigue_hand", &DuneImperiumState::GetIntrigueHandForTesting,
           py::arg("player"), py::return_value_policy::reference_internal)
      .def("get_intrigue_draw_deck",
           &DuneImperiumState::GetIntrigueDrawDeckForTesting,
           py::return_value_policy::reference_internal)
      .def("get_player_spice", &DuneImperiumState::GetPlayerSpiceForTesting,
           py::arg("player"))
      .def("get_player_solari", &DuneImperiumState::GetPlayerSolariForTesting,
           py::arg("player"))
      .def("get_player_water", &DuneImperiumState::GetPlayerWaterForTesting,
           py::arg("player"))
      .def("get_player_vp", &DuneImperiumState::GetPlayerVpForTesting,
           py::arg("player"))
      .def("get_player_persuasion",
           &DuneImperiumState::GetPlayerPersuasionForTesting, py::arg("player"))
      .def("get_player_influence",
           &DuneImperiumState::GetPlayerInfluenceForTesting, py::arg("player"),
           py::arg("faction"))
      .def("get_player_troops", &DuneImperiumState::GetPlayerTroopsForTesting,
           py::arg("player"))
      .def("get_player_troops_in_garrison",
           &DuneImperiumState::GetPlayerTroopsInGarrisonForTesting,
           py::arg("player"))
      .def("get_player_dreadnoughts_in_garrison",
           &DuneImperiumState::GetPlayerDreadnoughtsInGarrisonForTesting,
           py::arg("player"))
      .def("get_agent_space_owner",
           &DuneImperiumState::GetAgentSpaceOwnerForTesting,
           py::arg("board_index"))
      .def("player_leader", &DuneImperiumState::PlayerLeader, py::arg("player"))
      .def("get_player_tech_tiles",
           &DuneImperiumState::GetPlayerTechTilesForTesting, py::arg("player"))
      .def("is_tech_flipped", &DuneImperiumState::IsTechFlippedForTesting,
           py::arg("player"), py::arg("tech_id"))
      .def("get_bonus_spice", &DuneImperiumState::GetBonusSpiceForTesting,
           py::arg("space_action"))
      .def("get_tech_market", &DuneImperiumState::GetTechMarketForTesting,
           py::return_value_policy::reference_internal)
      .def("get_tech_negotiators",
           &DuneImperiumState::GetTechNegotiatorsForTesting, py::arg("player"))
      .def("get_player_agents_remaining",
           &DuneImperiumState::GetPlayerAgentsRemainingForTesting,
           py::arg("player"))
      .def("get_specimens", &DuneImperiumState::GetSpecimensForTesting,
           py::arg("player"))
      .def("get_player_cards_in_hand", &DuneImperiumState::GetPlayerCardsInHand,
           py::arg("player"))
      .def("get_pending_intrigue_player",
           &DuneImperiumState::pending_intrigue_draws_player)
      .def("get_pending_draw_player",
           &DuneImperiumState::pending_player_draw_player)
      .def("troops_in_combat", &DuneImperiumState::TroopsInCombat,
           py::arg("player"))
      .def("dreadnoughts_in_combat", &DuneImperiumState::DreadnoughtsInCombat,
           py::arg("player"))
      .def("combat_card_bonus", &DuneImperiumState::CombatCardBonus,
           py::arg("player"))
      .def("set_combat_card_bonus",
           &DuneImperiumState::SetCombatCardBonusForTesting, py::arg("player"),
           py::arg("bonus"))
      .def("combat_intrigue_bonus", &DuneImperiumState::CombatIntrigueBonus,
           py::arg("player"))
      .def("set_combat_intrigue_bonus",
           &DuneImperiumState::SetCombatIntrigueBonusForTesting,
           py::arg("player"), py::arg("bonus"))
      .def("get_reveal_persuasion_bonus",
           &DuneImperiumState::GetRevealPersuasionBonusForTesting,
           py::arg("player"))
      .def("get_topdeck_acquire",
           &DuneImperiumState::GetTopdeckAcquireForTesting, py::arg("player"))
      .def("get_reveal_done", &DuneImperiumState::GetRevealDoneForTesting,
           py::arg("player"))
      .def("get_played_agent_cards",
           &DuneImperiumState::GetPlayedAgentCardsForTesting, py::arg("player"),
           py::return_value_policy::reference_internal)
      .def("has_swordmaster", &DuneImperiumState::HasSwordmaster,
           py::arg("player"))
      .def("has_high_council", &DuneImperiumState::HasHighCouncil,
           py::arg("player"))
      .def("get_player_solari_direct", &DuneImperiumState::GetPlayerSolari,
           py::arg("player"))
      .def("player_shipping_level",
           &DuneImperiumState::PlayerShippingLevelForTesting, py::arg("player"))
      .def("get_tech_stack_size",
           &DuneImperiumState::GetTechStackSizeForTesting, py::arg("stack"))
      .def("get_mentat_available", &DuneImperiumState::MentatAvailableThisRound)
      .def("get_atomics_used", &DuneImperiumState::GetAtomicsUsedForTesting,
           py::arg("player"))
      .def("get_reserve_supply_tsmf",
           &DuneImperiumState::GetReserveSupplyTsmfForTesting)
      .def("get_reserve_supply_arrakis_liaison",
           &DuneImperiumState::GetReserveSupplyArrakisLiaisonForTesting)
      .def("count_imperium_cards_owned",
           &DuneImperiumState::CountImperiumCardsOwnedForTesting,
           py::arg("player"), py::arg("card_id"))
      .def("is_space_reachable", &DuneImperiumState::IsSpaceReachableForTesting,
           py::arg("space_action"))
      .def("get_tleilaxu_track", &DuneImperiumState::GetTleilaxuTrackForTesting,
           py::arg("player"))
      .def("get_research_bottom_col",
           &DuneImperiumState::GetResearchBottomColForTesting,
           py::arg("player"))
      .def("get_research_bottom_row",
           &DuneImperiumState::GetResearchBottomRowForTesting,
           py::arg("player"))
      .def("get_win_reward_solari",
           &DuneImperiumState::GetWinRewardSolariForTesting, py::arg("player"))
      .def("get_vladimir_secret_factions",
           &DuneImperiumState::GetVladimirSecretFactionsForTesting,
           py::arg("player"))
      .def("get_vladimir_revealed",
           &DuneImperiumState::GetVladimirRevealedForTesting, py::arg("player"))
      .def("get_hundro_player", &DuneImperiumState::GetHundroPlayerForTesting)
      .def("get_hundro_known_drawn_intrigue",
           &DuneImperiumState::GetHundroKnownDrawnIntrigueForTesting,
           py::arg("player"))
      .def("get_intrigue_deck_top_cards",
           &DuneImperiumState::GetIntrigueDeckTopCardsForTesting,
           py::return_value_policy::reference_internal)
      .def("set_intrigue_deck_top_cards",
           &DuneImperiumState::SetIntrigueDeckTopCardsForTesting,
           py::arg("cards"))
      .def("set_hundro_player", &DuneImperiumState::SetHundroPlayerForTesting,
           py::arg("player"))
      .def(
          "get_tessia_snooper_tokens",
          [](const DuneImperiumState &state, int player) {
            auto tokens = state.GetTessiaSnooperTokens(player);
            return std::vector<bool>(tokens.begin(), tokens.end());
          },
          py::arg("player"))
      .def("get_tessia_rewards_claimed",
           &DuneImperiumState::GetTessiaSnooperRewardsClaimed,
           py::arg("player"))
      .def("get_paul_known_top_card",
           &DuneImperiumState::GetPaulKnownTopCardForTesting, py::arg("player"))
      .def("deck_reshuffle_count", &DuneImperiumState::deck_reshuffle_count,
           py::arg("player"))
      .def("set_paul_known_top_card",
           &DuneImperiumState::SetPaulKnownTopCardForTesting, py::arg("player"),
           py::arg("card_id"))
      .def("ilesa_set_aside_card", &DuneImperiumState::SetAsideCardForIlesa,
           py::arg("player"))
      .def("set_ilesa_set_aside_card",
           &DuneImperiumState::SetIlesaSetAsideCardForTesting,
           py::arg("player"), py::arg("card_id"))
      .def("get_player_trashed_cards",
           &DuneImperiumState::GetPlayerTrashedCards, py::arg("player"))
      .def("get_pending_intrigue_choice_kind",
           &DuneImperiumState::GetPendingIntrigueChoiceKindForTesting)
      .def("get_pending_intrigue_choice_player",
           &DuneImperiumState::GetPendingIntrigueChoicePlayerForTesting)

      // --- Action helpers ---
      .def("gain_troops", &DuneImperiumState::GainTroopsForTesting,
           py::arg("player"), py::arg("count"))
      .def("grant_troops_to_combat",
           &DuneImperiumState::GrantTroopsToCombatForTesting, py::arg("player"),
           py::arg("count"))
      .def("gain_specimens", &DuneImperiumState::GainSpecimensForTesting,
           py::arg("player"), py::arg("count"))
      .def("gain_tech_negotiators",
           &DuneImperiumState::GainTechNegotiatorsForTesting, py::arg("player"),
           py::arg("count"))
      .def("gain_scarabs", &DuneImperiumState::GainScarabsForTesting,
           py::arg("player"), py::arg("amount"))
      .def("trigger_research", &DuneImperiumState::TriggerResearchForTesting,
           py::arg("player"), py::arg("steps"))
      .def("acquire_imperium_card",
           &DuneImperiumState::AcquireImperiumCardForTesting, py::arg("player"),
           py::arg("card_id"), py::arg("to_topdeck"))
      .def("begin_pending_combat_commit",
           &DuneImperiumState::BeginPendingCombatCommitForTesting,
           py::arg("player"))
      .def("resolve_combat", &DuneImperiumState::ResolveCombatForTesting)
      .def("total_troops", &DuneImperiumState::TotalTroopsForTesting,
           py::arg("player"))
      .def("lose_troops", &DuneImperiumState::LoseTroopsForTesting,
           py::arg("player"), py::arg("count"))
      .def(
          "combat_strength",
          [](const DuneImperiumState &s, int player) {
            return open_spiel::dune_imperium::CombatStrength(s, player);
          },
          py::arg("player"));
}
