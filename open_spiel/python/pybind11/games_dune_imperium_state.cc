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

#include "open_spiel/python/pybind11/games_dune_imperium_internal.h"

#include <array>
#include <string>
#include <vector>

#include "open_spiel/games/dune_imperium/dune_imperium.h"
#include "open_spiel/games/dune_imperium/dune_imperium_cards.h"
#include "open_spiel/games/dune_imperium/dune_imperium_common.h"
#include "open_spiel/games/dune_imperium/dune_imperium_content.h"
#include "open_spiel/games/dune_imperium/dune_imperium_util.h"
#include "open_spiel/spiel.h"

namespace py = ::pybind11;

using open_spiel::Player;
using open_spiel::State;
using open_spiel::dune_imperium::DuneImperiumState;
using open_spiel::dune_imperium::ImperiumCard;

namespace open_spiel {
namespace dune_imperium_pybind {

#pragma GCC optimize("no-var-tracking-assignments")
void RegisterDuneImperiumState(py::module &m, py::module_ &di) {
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
      .def_readonly("access_mask", &ImperiumCard::access_mask)
      .def_readonly("reveal_swords", &ImperiumCard::reveal_swords)
      .def_readonly("reveal_troops", &ImperiumCard::reveal_troops)
      .def_readonly("agent_troops", &ImperiumCard::agent_troops);

  di.def(
      "get_imperium_card",
      [](int id) -> const ImperiumCard * {
        return open_spiel::dune_imperium::FindImperiumCardById(id);
      },
      py::arg("id"), py::return_value_policy::reference);

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
      .def("set_player_deck_top_cards",
           &DuneImperiumState::SetPlayerDeckTopCardsForTesting,
           py::arg("player"), py::arg("cards"))
      .def("get_player_deck_top_cards",
           &DuneImperiumState::GetPlayerDeckTopCardsForTesting,
           py::arg("player"))
      .def("public_hand_card_arrived", &DuneImperiumState::PublicHandCardArrived,
           py::arg("player"), py::arg("card_id"))
      .def("public_hand_card_departed", &DuneImperiumState::PublicHandCardDeparted,
           py::arg("player"), py::arg("card_id"))
      .def("public_hand_cards_revealed", &DuneImperiumState::PublicHandCardsRevealed,
           py::arg("player"))
      .def("refresh_public_card_knowledge", &DuneImperiumState::RefreshPublicCardKnowledge,
           py::arg("player"))
      .def("begin_ilesa_public_knowledge", &DuneImperiumState::BeginIlesaPublicKnowledge,
           py::arg("player"))
      .def("return_ilesa_public_knowledge", &DuneImperiumState::ReturnIlesaPublicKnowledge,
           py::arg("player"), py::arg("bonus_eligible"))
      .def("synchronize_public_hand_knowledge",
           &DuneImperiumState::SynchronizePublicHandKnowledge,
           py::arg("player"), py::arg("cards"), py::arg("complete") = false,
           py::arg("owned_pool_complete") = true)
      .def("get_publicly_known_hand_cards", &DuneImperiumState::GetPubliclyKnownHandCards,
           py::arg("player"))
      .def("get_ilesa_public_candidates", &DuneImperiumState::GetIlesaPublicCandidates,
           py::arg("player"))
      .def("get_owned_imperium_card_pool", &DuneImperiumState::GetOwnedImperiumCardPool,
           py::arg("player"))
      .def("get_publicly_known_snooper_card", &DuneImperiumState::GetPubliclyKnownSnooperCard)
      .def("public_card_knowledge_digest", &DuneImperiumState::PublicCardKnowledgeDigest)
      .def("set_public_bonus_evidence_complete", &DuneImperiumState::SetPublicBonusEvidenceComplete,
           py::arg("player"), py::arg("complete"))
      .def("observe_ilesa_public_bonus", &DuneImperiumState::ObserveIlesaPublicBonus,
           py::arg("player"), py::arg("played"), py::arg("reward"),
           py::arg("complete"))
      .def("reconcile_hidden_hand_copies", &DuneImperiumState::ReconcileHiddenHandCopies,
           py::arg("player"), py::arg("required"),
           py::arg("protected_cards") = std::vector<int>{})
      .def("reconcile_hidden_hand_for_public_knowledge",
           &DuneImperiumState::ReconcileHiddenHandForPublicKnowledge,
           py::arg("player"))
      .def("reconcile_revealed_card_copies", &DuneImperiumState::ReconcileRevealedCardCopies,
           py::arg("player"), py::arg("cards"), py::arg("complete"))
      .def("reconcile_discard_card_copies", &DuneImperiumState::ReconcileDiscardCardCopies,
           py::arg("player"), py::arg("cards"), py::arg("complete"))
      .def("reconcile_hand_and_ilesa_for_reveal", &DuneImperiumState::ReconcileHandAndIlesaForReveal,
           py::arg("player"), py::arg("observed_cards"), py::arg("complete") = false)
      .def("get_pending_graft_primary_id", &DuneImperiumState::GetPendingGraftPrimaryId)
      .def("get_pending_graft_player", &DuneImperiumState::GetPendingGraftPlayer)
      .def("get_ilesa_bonus_card", &DuneImperiumState::GetIlesaBonusCardForTesting,
           py::arg("player"))
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
      .def("execute_influence_loss",
           &DuneImperiumState::ExecuteInfluenceLossForTesting, py::arg("player"),
           py::arg("faction"), py::arg("amount"))
      .def("set_vladimir_secret_factions",
           &DuneImperiumState::SetVladimirSecretFactionsForTesting,
           py::arg("player"), py::arg("first"), py::arg("second"))
      .def("set_player_troops_in_garrison",
           &DuneImperiumState::SetPlayerTroopsInGarrisonForTesting,
           py::arg("player"), py::arg("amount"))
      .def("set_troops", &DuneImperiumState::SetTroopsForTesting,
           py::arg("player"), py::arg("garrison"), py::arg("combat"))
      .def("set_player_troops_gained_this_turn",
           &DuneImperiumState::SetPlayerTroopsGainedThisTurnForTesting,
           py::arg("player"), py::arg("amount"))
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
      .def("set_helena_reserved_cards",
           &DuneImperiumState::SetHelenaReservedCardsForTesting,
           py::arg("player"), py::arg("cards"))
      .def("get_helena_reserved_cards",
           &DuneImperiumState::GetHelenaReservedCards, py::arg("player"),
           py::return_value_policy::reference_internal)

      // --- Getters (sync validation) ---
      .def("get_player_hand",
           static_cast<const std::vector<int>& (DuneImperiumState::*)(
               open_spiel::Player) const>(
               &DuneImperiumState::GetPlayerHandForTesting),
           py::arg("player"), py::return_value_policy::reference_internal)
      .def("get_player_draw_deck",
           static_cast<const std::vector<int>& (DuneImperiumState::*)(
               open_spiel::Player) const>(
               &DuneImperiumState::GetPlayerDrawDeckForTesting),
           py::arg("player"),
           py::return_value_policy::reference_internal)
      .def("get_pending_poison_snooper_card",
           &DuneImperiumState::GetPendingPoisonSnooperCardForTesting)
      .def("get_player_discard", &DuneImperiumState::GetPlayerDiscardForTesting,
           py::arg("player"), py::return_value_policy::reference_internal)
      .def("set_player_discard", &DuneImperiumState::SetPlayerDiscardForTesting,
           py::arg("player"), py::arg("discard"))

      // --- Revealed zone injection (OCR bridge) ---
      .def("get_revealed_cards", &DuneImperiumState::GetRevealedCardsForTesting,
           py::arg("player"), py::return_value_policy::reference_internal)
      .def("set_revealed_cards", &DuneImperiumState::SetRevealedCardsForTesting,
           py::arg("player"), py::arg("cards"))
      .def("set_steam_prefer_played_in_play_trash",
           &DuneImperiumState::SetSteamPreferPlayedInPlayTrash,
           py::arg("prefer"))
      .def("get_pending_trash_choice_descriptors",
           &DuneImperiumState::GetPendingTrashChoiceDescriptors)
      .def("steam_select_queued_trash_choice",
           &DuneImperiumState::SteamSelectQueuedTrashChoice,
           py::arg("player"), py::arg("descriptor"))
      .def("copy_state_from",
           &DuneImperiumState::CopyStateFrom,
           py::arg("other"))

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
      .def("set_fixed_deck_pool_discard",
           &DuneImperiumState::SetFixedDeckPoolDiscardForTesting, py::arg("player"),
           py::arg("cards"))
      .def("get_fixed_deck_pool_discard",
           &DuneImperiumState::GetFixedDeckPoolDiscardForTesting, py::arg("player"))
      .def("set_fixed_deck_pool_draw_deck",
           &DuneImperiumState::SetFixedDeckPoolDrawDeckForTesting, py::arg("player"),
           py::arg("cards"))
      .def("get_fixed_deck_pool_draw_deck",
           &DuneImperiumState::GetFixedDeckPoolDrawDeckForTesting, py::arg("player"))
      .def("set_fixed_deck_pool_cards_drawn",
           &DuneImperiumState::SetFixedDeckPoolCardsDrawnForTesting,
           py::arg("player"), py::arg("count"))
      .def("get_fixed_deck_pool_cards_drawn",
           &DuneImperiumState::GetFixedDeckPoolCardsDrawnForTesting,
           py::arg("player"))
      .def("get_imperium_row", &DuneImperiumState::GetImperiumRowForTesting,
           py::return_value_policy::reference_internal)
      .def("get_imperium_market_appendix",
           &DuneImperiumState::GetImperiumMarketAppendix)
      .def("information_state_tensor_with_appendix",
           [](const DuneImperiumState &s, Player player,
              open_spiel::dune_imperium::MarketAppendixMode mode,
              const std::string &deck_pool_semantics) -> std::vector<float> {
             return s.InformationStateTensorWithAppendix(
                 player, mode,
                 dune_imperium::ParseDeckPoolSemanticsOrDie(deck_pool_semantics));
           },
           py::arg("player"), py::arg("mode"),
           py::arg("deck_pool_semantics") = "legacy")
      .def("information_state_tensor_with_appendix",
           [](const DuneImperiumState &s, Player player,
              const std::string &mode_str,
              const std::string &deck_pool_semantics) -> std::vector<float> {
             return s.InformationStateTensorWithAppendix(
                 player, dune_imperium::ParseMarketAppendixMode(mode_str),
                 dune_imperium::ParseDeckPoolSemanticsOrDie(deck_pool_semantics));
           },
           py::arg("player"), py::arg("mode") = "ordered_market",
           py::arg("deck_pool_semantics") = "legacy")
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
      .def("begin_steam_intrigue_knowledge",
           &DuneImperiumState::BeginSteamIntrigueKnowledge,
           py::arg("observer"), py::arg("observations"))
      .def("has_steam_intrigue_knowledge",
           &DuneImperiumState::HasSteamIntrigueKnowledge, py::arg("observer"))
      .def("set_steam_intrigue_simulation_for_search",
           &DuneImperiumState::SetSteamIntrigueSimulationForSearch, py::arg("observer"))
      .def("steam_intrigue_simulation_for_search",
           &DuneImperiumState::SteamIntrigueSimulationForSearch)
      .def("get_pending_intrigue_operation_for_steam",
           &DuneImperiumState::GetPendingIntrigueOperationForSteam)
      .def("get_steam_intrigue_knowledge_view",
           &DuneImperiumState::GetSteamIntrigueKnowledgeView, py::arg("observer"))
      .def("set_steam_observed_intrigue_chance",
           &DuneImperiumState::SetSteamObservedIntrigueChance,
           py::arg("observer"), py::arg("card"), py::arg("action") = -1)
      .def("clear_steam_observed_intrigue_chance",
           &DuneImperiumState::ClearSteamObservedIntrigueChance)
      .def("reconcile_intrigue_assignments_for_steam",
           &DuneImperiumState::ReconcileIntrigueAssignmentsForSteam,
           py::arg("observer"), py::arg("observations"), py::arg("required_action") = -1)
      .def("resync_intrigue_hand_for_steam", &DuneImperiumState::ResyncIntrigueHandForSteam,
           py::arg("observer"), py::arg("player"), py::arg("count"), py::arg("observations"))
      .def("replace_steam_intrigue_hand_for_steam",
           &DuneImperiumState::ReplaceSteamIntrigueHandForSteam,
           py::arg("observer"), py::arg("player"), py::arg("count"), py::arg("observations"))
      .def("get_intrigue_inventory_for_steam", [](const DuneImperiumState& s) {
        py::dict result;
        std::vector<std::vector<int>> hands;
        for (int p = 0; p < open_spiel::dune_imperium::kNumPlayers; ++p)
          hands.push_back(s.GetPlayerIntrigues(p));
        result["hands"] = py::cast(hands);
        result["draw_bag"] = py::cast(s.GetIntrigueDrawDeckForTesting());
        result["topdeck"] = py::cast(s.GetIntrigueDeckTopCardsForTesting());
        result["bottomdeck"] = py::cast(s.GetIntrigueDeckBottomCardsForTesting());
        result["bottomdeck_owners"] = py::cast(s.GetIntrigueDeckBottomCardOwnersForTesting());
        const int ceremony_player = s.GetPendingCeremonyPlayer();
        result["ceremony"] = ceremony_player >= 0
            ? py::cast(s.GetPendingCeremonyPeekCards())
            : py::cast(std::array<int, 2>{-1, -1});
        result["ceremony_player"] = ceremony_player;
        result["discard"] = py::cast(s.GetIntrigueDiscardForTesting());
        const int hundro_player = s.GetPendingHundroSetupPlayer();
        // This offer aliases the last two hand entries; it is not another zone.
        result["hundro_offer"] = hundro_player >= 0
            ? py::make_tuple(s.GetPendingHundroSetupCard(0), s.GetPendingHundroSetupCard(1))
            : py::make_tuple(-1, -1);
        result["hundro_player"] = hundro_player;
        return result;
      })
      .def("get_pending_steal_intrigue_for_steam", &DuneImperiumState::GetPendingStealIntrigueForSteam)
      .def("set_steam_intrigue_search_constraints", &DuneImperiumState::SetSteamIntrigueSearchConstraints,
           py::arg("observer"), py::arg("observations"))
      .def("set_steam_endgame_intrigue_evidence", &DuneImperiumState::SetSteamEndgameIntrigueEvidence,
           py::arg("hands"), py::arg("optional_selections") = std::vector<std::vector<int>>{})
      .def("observe_steam_endgame_scarab", &DuneImperiumState::ObserveSteamEndgameScarab,
           py::arg("player"))
      .def("get_steam_endgame_intrigue_evidence", &DuneImperiumState::GetSteamEndgameIntrigueEvidence)
      .def("get_steam_endgame_intrigue_selections", &DuneImperiumState::GetSteamEndgameIntrigueSelections)
      .def("has_steam_endgame_intrigue_evidence", &DuneImperiumState::HasSteamEndgameIntrigueEvidence)
      .def("clear_steam_endgame_intrigue_evidence", &DuneImperiumState::ClearSteamEndgameIntrigueEvidence)
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
      .def("get_final_scored_vp", &DuneImperiumState::FinalScoredVp,
           py::arg("player"),
           "Return the player's scored VP, including final bonuses. Terminal states only.")
      .def("get_endgame_intrigue_vp_bonus",
           &DuneImperiumState::EndgameIntrigueVpBonusForTesting,
           py::arg("player"), py::arg("card_id"),
           "Return this card's final VP bonus for the player, regardless of ownership.")
      .def("get_endgame_intrigue_spice_bonus",
           &DuneImperiumState::EndgameIntrigueSpiceBonusForTesting,
           py::arg("player"), py::arg("card_id"),
           "Return this card's final spice tiebreaker bonus, regardless of ownership.")
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
      .def("get_player_troops_gained_this_turn",
           &DuneImperiumState::GetPlayerTroopsGainedThisTurnForTesting,
           py::arg("player"))
      .def("get_player_dreadnoughts_in_garrison",
           &DuneImperiumState::GetPlayerDreadnoughtsInGarrisonForTesting,
           py::arg("player"))
      .def("get_agent_space_owner",
           &DuneImperiumState::GetAgentSpaceOwnerForTesting,
           py::arg("board_index"))
      .def("player_leader", &DuneImperiumState::PlayerLeader, py::arg("player"))
      .def("get_phase", &DuneImperiumState::phase)
      .def("get_current_round", &DuneImperiumState::GetCurrentRound)
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
      .def("get_player_revealed", &DuneImperiumState::GetPlayerRevealedForTesting,
           py::arg("player"))
      .def("is_player_in_reveal_turn", &DuneImperiumState::IsPlayerInRevealTurn,
           py::arg("player"))
      .def("get_pending_armand_reveal_trash_active",
           &DuneImperiumState::GetPendingArmandRevealTrashActive)
      .def("get_pending_armand_reveal_trash_player",
           &DuneImperiumState::GetPendingArmandRevealTrashPlayer)
      .def("get_pending_disposal_reveal_trash_player",
           &DuneImperiumState::GetPendingDisposalRevealTrashPlayer)
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
      .def("pending_acquire_influence_amount", &DuneImperiumState::PendingAcquireInfluenceAmount)
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
      .def("pending_reward_in_the_shadows",
           static_cast<std::vector<int> (DuneImperiumState::*)() const>(
               &DuneImperiumState::pending_reward_in_the_shadows))
      .def("pending_reward_in_the_shadows",
           &DuneImperiumState::GetPendingRewardInTheShadows, py::arg("player"))
      .def("pending_armand_acquire",
           static_cast<std::vector<int> (DuneImperiumState::*)() const>(
               &DuneImperiumState::pending_armand_acquire))
      .def("pending_armand_acquire",
           &DuneImperiumState::GetPendingArmandAcquire, py::arg("player"))
      .def("set_pending_armand_acquire",
           &DuneImperiumState::SetPendingArmandAcquireForTesting,
           py::arg("player"), py::arg("val"))
      .def("pending_helena_reserve",
           static_cast<std::vector<int> (DuneImperiumState::*)() const>(
               &DuneImperiumState::pending_helena_reserve))
      .def("pending_helena_reserve",
           &DuneImperiumState::GetPendingHelenaReserve, py::arg("player"))
      .def("pending_rhombur_signet",
           static_cast<std::vector<int> (DuneImperiumState::*)() const>(
               &DuneImperiumState::pending_rhombur_signet))
      .def("pending_rhombur_signet",
           &DuneImperiumState::GetPendingRhomburSignet, py::arg("player"))
      .def("pending_tessia_signet",
           static_cast<std::vector<int> (DuneImperiumState::*)() const>(
               &DuneImperiumState::pending_tessia_signet))
      .def("pending_tessia_signet",
           &DuneImperiumState::GetPendingTessiaSignet, py::arg("player"))
      .def("pending_tessia_first_reward_choice",
           static_cast<std::vector<bool> (DuneImperiumState::*)() const>(
               &DuneImperiumState::pending_tessia_first_reward_choice))
      .def("pending_tessia_first_reward_choice",
           &DuneImperiumState::GetPendingTessiaFirstRewardChoice, py::arg("player"))
      .def("pending_ixian_engineer_tech",
           static_cast<std::vector<int> (DuneImperiumState::*)() const>(
               &DuneImperiumState::pending_ixian_engineer_tech))
      .def("pending_ixian_engineer_tech",
           &DuneImperiumState::GetPendingIxianEngineerTech, py::arg("player"))
      .def("pending_memnon_high_council",
           static_cast<std::vector<int> (DuneImperiumState::*)() const>(
               &DuneImperiumState::pending_memnon_high_council))
      .def("pending_memnon_high_council",
           &DuneImperiumState::GetPendingMemnonHighCouncil, py::arg("player"))
      .def("pending_court_intrigue_return",
           static_cast<std::vector<int> (DuneImperiumState::*)() const>(
               &DuneImperiumState::pending_court_intrigue_return))
      .def("pending_court_intrigue_return",
           &DuneImperiumState::GetPendingCourtIntrigueReturn, py::arg("player"))
      .def("pending_high_priority_travel",
           static_cast<std::vector<int> (DuneImperiumState::*)() const>(
               &DuneImperiumState::pending_high_priority_travel))
      .def("pending_high_priority_travel",
           &DuneImperiumState::GetPendingHighPriorityTravel, py::arg("player"))
      .def("pending_guild_chief_admin_discard",
           static_cast<std::vector<int> (DuneImperiumState::*)() const>(
               &DuneImperiumState::pending_guild_chief_admin_discard))
      .def("pending_guild_chief_admin_discard",
           &DuneImperiumState::GetPendingGuildChiefAdminDiscard, py::arg("player"))
      .def("pending_ix_guild_compact_discard",
           static_cast<std::vector<int> (DuneImperiumState::*)() const>(
               &DuneImperiumState::pending_ix_guild_compact_discard))
      .def("pending_ix_guild_compact_discard",
           &DuneImperiumState::GetPendingIxGuildCompactDiscard, py::arg("player"))
      .def("set_pending_reward_in_the_shadows",
           &DuneImperiumState::SetPendingRewardInTheShadowsForTesting,
           py::arg("player"), py::arg("val"))
      .def("set_pending_guild_chief_admin_discard",
           &DuneImperiumState::SetPendingGuildChiefAdminDiscardForTesting,
           py::arg("player"), py::arg("val"))
      .def("set_pending_ix_guild_compact_discard",
           &DuneImperiumState::SetPendingIxGuildCompactDiscardForTesting,
           py::arg("player"), py::arg("val"))
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
      .def("set_intrigue_deck_bottom_cards_for_testing",
           &DuneImperiumState::SetIntrigueDeckBottomCardsForTesting, py::arg("cards"))
      .def("set_intrigue_deck_bottom_card_owners_for_testing",
           &DuneImperiumState::SetIntrigueDeckBottomCardOwnersForTesting, py::arg("owners"))
      .def("draw_intrigue_cards_for_testing",
           &DuneImperiumState::DrawIntrigueCardsForTesting, py::arg("player"), py::arg("count"))
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
      .def("get_pending_tessia_reward_kind",
           &DuneImperiumState::GetPendingTessiaRewardKind,
           py::arg("player"), py::arg("faction"))
      .def("get_pending_memocorders_influence",
           &DuneImperiumState::GetPendingMemocordersInfluence,
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
      .def("get_active_intrigue_choice_kind",
           &DuneImperiumState::GetActiveIntrigueChoiceKind)
      .def("prioritize_research_scarab_trade_for_steam",
           &DuneImperiumState::PrioritizeResearchScarabTradeForSteam,
           py::arg("player"))
      .def("get_pending_intrigue_choice_card_id",
           &DuneImperiumState::GetPendingIntrigueChoiceCardId)
      .def("reveal_retreat_available",
           &DuneImperiumState::RevealRetreatAvailable, py::arg("player"))
      .def("reveal_retreat_actions",
           &DuneImperiumState::RevealRetreatActions, py::arg("player"))
      .def("bene_gesserit_sister_choice_available",
           &DuneImperiumState::BeneGesseritSisterChoiceAvailable,
           py::arg("player"))
      .def("bene_gesserit_sister_choice_pending",
           &DuneImperiumState::BeneGesseritSisterChoicePending,
           py::arg("player"))
      .def("prioritize_winning_dreadnought_choice_for_steam",
           &DuneImperiumState::PrioritizeWinningDreadnoughtChoiceForSteam,
           py::arg("player"))
      .def("prioritize_combat_shipping_for_steam",
           &DuneImperiumState::PrioritizeCombatShippingForSteam,
           py::arg("player"))
      .def("prioritize_combat_influence_for_steam",
           &DuneImperiumState::PrioritizeCombatInfluenceForSteam,
           py::arg("player"))
      .def("get_pending_intrigue_choice_player",
           &DuneImperiumState::GetPendingIntrigueChoicePlayerForTesting)
      .def("get_pending_ceremony_peek_cards",
           &DuneImperiumState::GetPendingCeremonyPeekCardsForTesting)
      .def("set_pending_ceremony_peek_cards",
           &DuneImperiumState::SetPendingCeremonyPeekForTesting,
           py::arg("player"), py::arg("cards"))
      .def("begin_imperium_ceremony_peek_for_testing",
           &DuneImperiumState::BeginImperiumCeremonyPeekForTesting, py::arg("player"))

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
      .def("resolve_observed_bypass_protocol",
           &DuneImperiumState::ResolveObservedBypassProtocol, py::arg("player"),
           py::arg("card_id"), py::arg("to_topdeck"))
      .def("acquire_tleilaxu_imperium_card",
           &DuneImperiumState::AcquireTleilaxuImperiumCardForTesting,
           py::arg("player"), py::arg("card_id"))
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

}  // namespace dune_imperium_pybind
}  // namespace open_spiel
