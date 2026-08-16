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
#ifdef OPEN_SPIEL_BUILD_WITH_LIBTORCH
#include "dune_puct_is_mcts.h"
#include "dune_search_session.h"
#include "dune_evaluator.h"
#include "dune_search_routing.h"
#endif
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

#pragma GCC optimize("no-var-tracking-assignments")
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
      .def("get_phase", &DuneImperiumState::phase)
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
      .def("pending_shipping_choice", &DuneImperiumState::pending_shipping_choice)
      .def("pending_shipping_player", &DuneImperiumState::pending_shipping_player)
      .def("pending_shipping_needs_level1_choice", &DuneImperiumState::pending_shipping_needs_level1_choice)
      .def("pending_shipping_needs_influence_choice", &DuneImperiumState::pending_shipping_needs_influence_choice)
      .def("pending_shipping_needs_tech_choice", &DuneImperiumState::pending_shipping_needs_tech_choice)
      .def("pending_shipping_needs_troops_choice", &DuneImperiumState::pending_shipping_needs_troops_choice)
      .def("pending_reward_card_draw", &DuneImperiumState::PendingRewardCardDraw)
      .def("pending_reward_intrigue_draw", &DuneImperiumState::pending_reward_intrigue_draw)
      .def("pending_reward_shipping", &DuneImperiumState::pending_reward_shipping)
      .def("pending_reward_research", &DuneImperiumState::pending_reward_research)
      .def("pending_reward_influence_amount", &DuneImperiumState::PendingRewardInfluenceAmount)
      .def("pending_reward_influence_faction", &DuneImperiumState::pending_reward_influence_faction)
      .def("pending_reward_scarab", &DuneImperiumState::pending_reward_scarab)
      .def("pending_reward_control_bonus", &DuneImperiumState::pending_reward_control_bonus)
      .def("pending_reward_dissecting_kit_trash", &DuneImperiumState::pending_reward_dissecting_kit_trash)
      .def("pending_reward_scientific_breakthrough_trash", &DuneImperiumState::pending_reward_scientific_breakthrough_trash)
      .def("pending_smuggler_thopter_draw", &DuneImperiumState::pending_smuggler_thopter_draw)
      .def("pending_spice_trader_discard", &DuneImperiumState::pending_spice_trader_discard)
      .def("pending_tleilaxu_master_acquire", &DuneImperiumState::pending_tleilaxu_master_acquire)
      .def("pending_reward_tech_negotiation", &DuneImperiumState::pending_reward_tech_negotiation)
      .def("pending_reward_baron_ring", &DuneImperiumState::pending_reward_baron_ring)
      .def("pending_reward_rabban_ring", &DuneImperiumState::pending_reward_rabban_ring)
      .def("pending_reward_leto_ring", &DuneImperiumState::pending_reward_leto_ring)
      .def("pending_reward_yuna_ring", &DuneImperiumState::pending_reward_yuna_ring)
      .def("pending_reward_hundro_ring", &DuneImperiumState::pending_reward_hundro_ring)
      .def("pending_reward_ilesa_ring", &DuneImperiumState::pending_reward_ilesa_ring)
      .def("pending_reward_slig_farmer", &DuneImperiumState::pending_reward_slig_farmer)
      .def("pending_reward_firm_grip", &DuneImperiumState::pending_reward_firm_grip)
      .def("pending_reward_shifting_allegiances", &DuneImperiumState::pending_reward_shifting_allegiances)
      .def("pending_reward_in_the_shadows", &DuneImperiumState::pending_reward_in_the_shadows)
      .def("pending_reward_web_of_power_1", &DuneImperiumState::pending_reward_web_of_power_1)
      .def("pending_reward_web_of_power_2", &DuneImperiumState::pending_reward_web_of_power_2)
      .def("pending_reward_web_of_power_3", &DuneImperiumState::pending_reward_web_of_power_3)
      .def("pending_reward_spice_smugglers", &DuneImperiumState::pending_reward_spice_smugglers)
      .def("pending_reward_esmar_tuek", &DuneImperiumState::pending_reward_esmar_tuek)
      .def("pending_reward_duncan_idaho", &DuneImperiumState::pending_reward_duncan_idaho)
      .def("pending_reward_fremen_camp", &DuneImperiumState::pending_reward_fremen_camp)
      .def("pending_reward_guild_ambassador", &DuneImperiumState::pending_reward_guild_ambassador)
      .def("pending_reward_local_fence", &DuneImperiumState::pending_reward_local_fence)
      .def("pending_reward_organ_merchants", &DuneImperiumState::pending_reward_organ_merchants)
      .def("pending_reward_satellite_ban", &DuneImperiumState::pending_reward_satellite_ban)
      .def("pending_reward_sayyadina", &DuneImperiumState::pending_reward_sayyadina)
      .def("pending_reward_stillsuits_manufacturer", &DuneImperiumState::pending_reward_stillsuits_manufacturer)
      .def("pending_reward_tleilaxu_surgeon", &DuneImperiumState::pending_reward_tleilaxu_surgeon)
      .def("played_tleilaxu_infiltrator", &DuneImperiumState::played_tleilaxu_infiltrator)
      .def("pending_reward_opulence", &DuneImperiumState::pending_reward_opulence)
      .def("pending_reward_weirding_way", &DuneImperiumState::pending_reward_weirding_way)
      .def("pending_reward_truthsayer", &DuneImperiumState::pending_reward_truthsayer)
      .def("pending_reward_other_memory", &DuneImperiumState::pending_reward_other_memory)
      .def("pending_reward_imperium_ceremony", &DuneImperiumState::pending_reward_imperium_ceremony)
      .def("pending_reward_imperial_bashar", &DuneImperiumState::pending_reward_imperial_bashar)
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

#ifdef OPEN_SPIEL_BUILD_WITH_LIBTORCH
  py::enum_<open_spiel::SearchOpponentMode>(di, "SearchOpponentMode")
      .value("MAX_N", open_spiel::SearchOpponentMode::kMaxN)
      .value("POLICY", open_spiel::SearchOpponentMode::kPolicy)
      .export_values();

  py::enum_<open_spiel::DuneISMCTSFinalPolicyType>(di, "DuneISMCTSFinalPolicyType")
      .value("NORMALIZED_VISIT_COUNT", open_spiel::DuneISMCTSFinalPolicyType::kNormalizedVisitCount)
      .value("MAX_VISIT_COUNT", open_spiel::DuneISMCTSFinalPolicyType::kMaxVisitCount)
      .value("MAX_VALUE", open_spiel::DuneISMCTSFinalPolicyType::kMaxValue)
      .export_values();

  py::class_<open_spiel::DuneSearchConfig>(di, "DuneSearchConfig")
      .def(py::init<>())
      .def_readwrite("max_simulations", &open_spiel::DuneSearchConfig::max_simulations)
      .def_readwrite("relative_time_budget_ms", &open_spiel::DuneSearchConfig::relative_time_budget_ms)
      .def_readwrite("max_nodes", &open_spiel::DuneSearchConfig::max_nodes)
      .def_readwrite("puct_c", &open_spiel::DuneSearchConfig::puct_c)
      .def_readwrite("opponent_mode", &open_spiel::DuneSearchConfig::opponent_mode)
      .def_readwrite("temperature", &open_spiel::DuneSearchConfig::temperature)
      .def_readwrite("opponent_temperature", &open_spiel::DuneSearchConfig::opponent_temperature)
      .def_readwrite("max_world_samples", &open_spiel::DuneSearchConfig::max_world_samples)
      .def_readwrite("utility_divisor", &open_spiel::DuneSearchConfig::utility_divisor)
      .def_readwrite("min_visit_threshold", &open_spiel::DuneSearchConfig::min_visit_threshold)
      .def_readwrite("covered_prior_threshold", &open_spiel::DuneSearchConfig::covered_prior_threshold)
      .def_readwrite("seed", &open_spiel::DuneSearchConfig::seed)
      .def_readwrite("final_policy_type", &open_spiel::DuneSearchConfig::final_policy_type)
      .def_readwrite("dirichlet_epsilon", &open_spiel::DuneSearchConfig::dirichlet_epsilon)
      .def_readwrite("dirichlet_alpha", &open_spiel::DuneSearchConfig::dirichlet_alpha)
      .def_readwrite("use_observation_string", &open_spiel::DuneSearchConfig::use_observation_string)
      .def_readwrite("verbose_diagnostics", &open_spiel::DuneSearchConfig::verbose_diagnostics)
      .def_readwrite("check_strategic_state", &open_spiel::DuneSearchConfig::check_strategic_state)
      // Leader-selection search (adopted 2026-08-16 at a fixed 64-simulation
      // budget). All three are required together: enabling the search without
      // mass-only coverage makes the search-result path reject the concentrated
      // Leader policy as low_coverage and hand back the raw prior.
      .def_readwrite("search_leader_draft", &open_spiel::DuneSearchConfig::search_leader_draft)
      .def_readwrite("leader_draft_simulations", &open_spiel::DuneSearchConfig::leader_draft_simulations)
      .def_readwrite("leader_mass_only_coverage", &open_spiel::DuneSearchConfig::leader_mass_only_coverage)
      .def_readwrite("fixed_continuation_reserve", &open_spiel::DuneSearchConfig::fixed_continuation_reserve)
      .def_readwrite("purchase_combat_budget", &open_spiel::DuneSearchConfig::purchase_combat_budget)
      .def_readwrite("live_continuation_reserve_seconds", &open_spiel::DuneSearchConfig::live_continuation_reserve_seconds)
      .def_readwrite("fixed_session_limit", &open_spiel::DuneSearchConfig::fixed_session_limit)
      .def_readwrite("model_checkpoint_path", &open_spiel::DuneSearchConfig::model_checkpoint_path)
      .def_readwrite("root_prior_temperature", &open_spiel::DuneSearchConfig::root_prior_temperature)
      .def_readwrite("training_root_prior_temperature", &open_spiel::DuneSearchConfig::training_root_prior_temperature)
      .def_readwrite("conservative_override_enabled", &open_spiel::DuneSearchConfig::conservative_override_enabled)
      .def_readwrite("conservative_covered_prior_threshold", &open_spiel::DuneSearchConfig::conservative_covered_prior_threshold)
      .def_readwrite("conservative_meaningful_visit_threshold", &open_spiel::DuneSearchConfig::conservative_meaningful_visit_threshold)
      .def_readwrite("conservative_q_margin_threshold", &open_spiel::DuneSearchConfig::conservative_q_margin_threshold)
      .def_readwrite("conservative_stability_checkpoint_fraction", &open_spiel::DuneSearchConfig::conservative_stability_checkpoint_fraction)
      .def_readwrite("conservative_continuation_overrides_disabled", &open_spiel::DuneSearchConfig::conservative_continuation_overrides_disabled)
      .def_readwrite("max_search_decision_depth", &open_spiel::DuneSearchConfig::max_search_decision_depth);

  py::class_<open_spiel::SearchDiagnostics>(di, "SearchDiagnostics")
      .def_readonly("actions", &open_spiel::SearchDiagnostics::actions)
      .def_readonly("visit_counts", &open_spiel::SearchDiagnostics::visit_counts)
      .def_readonly("q_values", &open_spiel::SearchDiagnostics::q_values)
      .def_readonly("priors", &open_spiel::SearchDiagnostics::priors)
      .def_readonly("root_value", &open_spiel::SearchDiagnostics::root_value)
      .def_readonly("total_root_visits", &open_spiel::SearchDiagnostics::total_root_visits)
      .def_readonly("num_covered_actions", &open_spiel::SearchDiagnostics::num_covered_actions)
      .def_readonly("covered_prior_mass", &open_spiel::SearchDiagnostics::covered_prior_mass)
      .def_readonly("max_decision_depth", &open_spiel::SearchDiagnostics::max_decision_depth)
      .def_readonly("mean_decision_depth", &open_spiel::SearchDiagnostics::mean_decision_depth)
      .def_readonly("protocol_version", &open_spiel::SearchDiagnostics::protocol_version)
      .def_readonly("session_id", &open_spiel::SearchDiagnostics::session_id)
      .def_readonly("searched_seat", &open_spiel::SearchDiagnostics::searched_seat)
      .def_readonly("round", &open_spiel::SearchDiagnostics::round)
      .def_readonly("phase", &open_spiel::SearchDiagnostics::phase)
      .def_readonly("decision_role", &open_spiel::SearchDiagnostics::decision_role)
      .def_readonly("budget_mode", &open_spiel::SearchDiagnostics::budget_mode)
      .def_readonly("hard_sim_limit", &open_spiel::SearchDiagnostics::hard_sim_limit)
      .def_readonly("soft_sim_limit", &open_spiel::SearchDiagnostics::soft_sim_limit)
      .def_readonly("hard_time_limit_ms", &open_spiel::SearchDiagnostics::hard_time_limit_ms)
      .def_readonly("soft_time_limit_ms", &open_spiel::SearchDiagnostics::soft_time_limit_ms)
      .def_readonly("elapsed_search_time_ms", &open_spiel::SearchDiagnostics::elapsed_search_time_ms)
      .def_readonly("observation_wait_time_ms", &open_spiel::SearchDiagnostics::observation_wait_time_ms)
      .def_readonly("inherited_root_visits", &open_spiel::SearchDiagnostics::inherited_root_visits)
      .def_readonly("newly_completed_simulations", &open_spiel::SearchDiagnostics::newly_completed_simulations)
      .def_readonly("session_cumulative_simulations", &open_spiel::SearchDiagnostics::session_cumulative_simulations)
      .def_readonly("short_window_cumulative_simulations", &open_spiel::SearchDiagnostics::short_window_cumulative_simulations)
      .def_readonly("session_cumulative_search_time_ms", &open_spiel::SearchDiagnostics::session_cumulative_search_time_ms)
      .def_readonly("long_agent_session_cumulative_time_ms", &open_spiel::SearchDiagnostics::long_agent_session_cumulative_time_ms)
      .def_readonly("re_root_status", &open_spiel::SearchDiagnostics::re_root_status)
      .def_readonly("post_chance_branch_miss", &open_spiel::SearchDiagnostics::post_chance_branch_miss)
      .def_readonly("root_coverage", &open_spiel::SearchDiagnostics::root_coverage)
      .def_readonly("reset_reason", &open_spiel::SearchDiagnostics::reset_reason)
      .def_readonly("tree_node_count", &open_spiel::SearchDiagnostics::tree_node_count)
      .def_readonly("selected_action", &open_spiel::SearchDiagnostics::selected_action)
      .def_readonly("legality_result", &open_spiel::SearchDiagnostics::legality_result)
      .def_readonly("fallback_reason", &open_spiel::SearchDiagnostics::fallback_reason)
      .def_readonly("raw_reference_action", &open_spiel::SearchDiagnostics::raw_reference_action)
      .def_readonly("mcts_proposed_action", &open_spiel::SearchDiagnostics::mcts_proposed_action)
      .def_readonly("confidence_fallback", &open_spiel::SearchDiagnostics::confidence_fallback)
      .def_readonly("mcts_overrode_raw", &open_spiel::SearchDiagnostics::mcts_overrode_raw)
      .def_readonly("stability_checkpoint_action", &open_spiel::SearchDiagnostics::stability_checkpoint_action)
      .def_readonly("stability_checkpoint_reached", &open_spiel::SearchDiagnostics::stability_checkpoint_reached)
      .def_readonly("stability_agreement", &open_spiel::SearchDiagnostics::stability_agreement)
      .def_readonly("pass_complete_search", &open_spiel::SearchDiagnostics::pass_complete_search)
      .def_readonly("pass_min_actions", &open_spiel::SearchDiagnostics::pass_min_actions)
      .def_readonly("pass_prior_mass", &open_spiel::SearchDiagnostics::pass_prior_mass)
      .def_readonly("pass_meaningful_visits", &open_spiel::SearchDiagnostics::pass_meaningful_visits)
      .def_readonly("pass_q_margin", &open_spiel::SearchDiagnostics::pass_q_margin)
      .def_readonly("pass_stability", &open_spiel::SearchDiagnostics::pass_stability);

  py::class_<open_spiel::DuneSearchResult>(di, "DuneSearchResult")
      .def_readonly("policy", &open_spiel::DuneSearchResult::policy)
      .def_readonly("diagnostics", &open_spiel::DuneSearchResult::diagnostics)
      .def_readonly("simulations_completed", &open_spiel::DuneSearchResult::simulations_completed)
      .def_readonly("elapsed_time_ms", &open_spiel::DuneSearchResult::elapsed_time_ms)
      .def_readonly("timeout_status", &open_spiel::DuneSearchResult::timeout_status)
      .def_readonly("used_fallback", &open_spiel::DuneSearchResult::used_fallback)
      .def_readonly("fallback_reason", &open_spiel::DuneSearchResult::fallback_reason)
      .def_readonly("inference_count", &open_spiel::DuneSearchResult::inference_count);

  py::class_<open_spiel::ControllerDecision>(di, "ControllerDecision")
      .def(py::init<>())
      .def_readwrite("selected_action", &open_spiel::ControllerDecision::selected_action)
      .def_readwrite("raw_reference_action", &open_spiel::ControllerDecision::raw_reference_action)
      .def_readwrite("mcts_proposed_action", &open_spiel::ControllerDecision::mcts_proposed_action)
      .def_readwrite("confidence_fallback", &open_spiel::ControllerDecision::confidence_fallback)
      .def_readwrite("mcts_overrode_raw", &open_spiel::ControllerDecision::mcts_overrode_raw)
      .def_readwrite("stability_checkpoint_reached", &open_spiel::ControllerDecision::stability_checkpoint_reached)
      .def_readwrite("stability_agreement", &open_spiel::ControllerDecision::stability_agreement)
      .def_readwrite("pass_complete_search", &open_spiel::ControllerDecision::pass_complete_search)
      .def_readwrite("pass_min_actions", &open_spiel::ControllerDecision::pass_min_actions)
      .def_readwrite("pass_prior_mass", &open_spiel::ControllerDecision::pass_prior_mass)
      .def_readwrite("pass_meaningful_visits", &open_spiel::ControllerDecision::pass_meaningful_visits)
      .def_readwrite("pass_q_margin", &open_spiel::ControllerDecision::pass_q_margin)
      .def_readwrite("pass_stability", &open_spiel::ControllerDecision::pass_stability);

  py::classh<open_spiel::DunePUCTISMCTSBot, open_spiel::Bot>(di, "DunePUCTISMCTSBot")
      .def(py::init<const open_spiel::DuneSearchConfig&, std::shared_ptr<open_spiel::algorithms::Evaluator>>(),
           py::arg("config"), py::arg("evaluator"))
      .def(py::init<const open_spiel::DuneSearchConfig&, const std::vector<std::shared_ptr<open_spiel::algorithms::Evaluator>>&>(),
           py::arg("config"), py::arg("evaluators"))
      .def("run_search", &open_spiel::DunePUCTISMCTSBot::RunSearch, py::arg("state"), py::arg("max_sims") = -1, py::arg("max_time_ms") = -1.0, py::arg("start_sim_index") = 0)
      .def("get_root_diagnostics", &open_spiel::DunePUCTISMCTSBot::GetRootDiagnostics,
           py::arg("state"), py::arg("min_visit_threshold") = 2, py::arg("chosen_action") = open_spiel::kInvalidAction)
      .def("get_last_search_result", &open_spiel::DunePUCTISMCTSBot::GetLastSearchResult);

  py::enum_<open_spiel::DuneSearchBudgetMode>(di, "DuneSearchBudgetMode")
      .value("POLICY_ONLY", open_spiel::DuneSearchBudgetMode::kPolicyOnly)
      .value("FIXED_SESSION_SIMULATIONS", open_spiel::DuneSearchBudgetMode::kFixedSessionSimulations)
      .value("TRAINING_FULL_FAST", open_spiel::DuneSearchBudgetMode::kTrainingFullFast)
      .value("LIVE_DEADLINE", open_spiel::DuneSearchBudgetMode::kLiveDeadline);

  py::class_<open_spiel::DuneSearchSession>(di, "DuneSearchSession")
      .def(py::init<const open_spiel::DuneSearchConfig&, std::shared_ptr<open_spiel::algorithms::Evaluator>, open_spiel::DuneSearchBudgetMode>(),
           py::arg("config"), py::arg("evaluator"), py::arg("budget_mode"))
      .def(py::init<const open_spiel::DuneSearchConfig&, const std::vector<std::shared_ptr<open_spiel::algorithms::Evaluator>>&, open_spiel::DuneSearchBudgetMode>(),
           py::arg("config"), py::arg("evaluators"), py::arg("budget_mode"))
      .def("search", &open_spiel::DuneSearchSession::Search, py::arg("state"), py::arg("remaining_time_ms") = -1.0)
      .def("select_controller_action", &open_spiel::DuneSearchSession::SelectControllerAction, py::arg("state"), py::arg("search_result"), py::arg("r_val"))
      .def("commit_action", &open_spiel::DuneSearchSession::CommitAction, py::arg("decision"))
      .def("discard_pending_action", &open_spiel::DuneSearchSession::DiscardPendingAction)
      .def("search_and_select", py::overload_cast<const State&>(&open_spiel::DuneSearchSession::SearchAndSelect), py::arg("state"))
      .def("search_and_select", py::overload_cast<const State&, double>(&open_spiel::DuneSearchSession::SearchAndSelect), py::arg("state"), py::arg("r_val"))
      .def("set_episode_id", &open_spiel::DuneSearchSession::SetEpisodeId, py::arg("episode_id"))
      .def("set_update_id", &open_spiel::DuneSearchSession::SetUpdateId, py::arg("update_id"))
      .def("reset_session", &open_spiel::DuneSearchSession::ResetSession, py::arg("reason"))
      .def("has_active_session", &open_spiel::DuneSearchSession::HasActiveSession)
      .def("get_bot", &open_spiel::DuneSearchSession::GetBot)
      .def("get_short_bot", &open_spiel::DuneSearchSession::GetShortBot)
      .def_property_readonly("session_new_simulations_completed", &open_spiel::DuneSearchSession::session_new_simulations_completed)
      .def_property_readonly("session_elapsed_time_ms", &open_spiel::DuneSearchSession::session_elapsed_time_ms);

  di.def("make_dune_nn_evaluator", &open_spiel::MakeDuneNNEvaluator,
         py::arg("checkpoint_path"), py::arg("device_str"),
         py::arg("hidden_dim") = 1024, py::arg("num_blocks") = 4);

  py::enum_<open_spiel::DuneDecisionRole>(di, "DuneDecisionRole")
      .value("FORCED_OR_BOOKKEEPING", open_spiel::DuneDecisionRole::kForcedOrBookkeeping)
      .value("LEADER_SELECTION", open_spiel::DuneDecisionRole::kLeaderSelection)
      .value("AGENT_PRIMARY", open_spiel::DuneDecisionRole::kAgentPrimary)
      .value("AGENT_CONTINUATION", open_spiel::DuneDecisionRole::kAgentContinuation)
      .value("PURCHASE", open_spiel::DuneDecisionRole::kPurchase)
      .value("COMBAT_INTRIGUE", open_spiel::DuneDecisionRole::kCombatIntrigue)
      .value("OTHER_OPTIONAL", open_spiel::DuneDecisionRole::kOtherOptional);

  di.def("classify_dune_decision_role", &open_spiel::ClassifyDuneDecisionRole,
         py::arg("state"), py::arg("searched_player"), py::arg("has_active_session"));
#endif
}
