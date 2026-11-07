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
#include <array>
#include <chrono>
#include <cmath>

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
  py::enum_<open_spiel::dune_imperium::MarketAppendixMode>(di, "MarketAppendixMode")
      .value("kNone", open_spiel::dune_imperium::MarketAppendixMode::kNone)
      .value("kZeros", open_spiel::dune_imperium::MarketAppendixMode::kZeros)
      .value("kOrderedMarket", open_spiel::dune_imperium::MarketAppendixMode::kOrderedMarket)
      .value("kOrderedCardSlotsV2", open_spiel::dune_imperium::MarketAppendixMode::kOrderedCardSlotsV2)
      .value("kFullPublicInformationV3", open_spiel::dune_imperium::MarketAppendixMode::kFullPublicInformationV3)
      .value("kFullPublicInformationV4", open_spiel::dune_imperium::MarketAppendixMode::kFullPublicInformationV4)
      .value("kFullPublicInformationV5", open_spiel::dune_imperium::MarketAppendixMode::kFullPublicInformationV5)
      .export_values();

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
      .def_readonly("pass_stability", &open_spiel::SearchDiagnostics::pass_stability)
      .def_readonly("stage1_planned_worlds", &open_spiel::SearchDiagnostics::stage1_planned_worlds)
      .def_readonly("stage1_kept_worlds", &open_spiel::SearchDiagnostics::stage1_kept_worlds)
      .def_readonly("stage2_planned_worlds", &open_spiel::SearchDiagnostics::stage2_planned_worlds)
      .def_readonly("stage2_kept_worlds", &open_spiel::SearchDiagnostics::stage2_kept_worlds)
      .def_readonly("stage1_rollouts_executed", &open_spiel::SearchDiagnostics::stage1_rollouts_executed)
      .def_readonly("stage2_rollouts_executed", &open_spiel::SearchDiagnostics::stage2_rollouts_executed)
      .def_readonly("stage1_elapsed_ms", &open_spiel::SearchDiagnostics::stage1_elapsed_ms)
      .def_readonly("stage2_elapsed_ms", &open_spiel::SearchDiagnostics::stage2_elapsed_ms)
      .def_readonly("termination_reason", &open_spiel::SearchDiagnostics::termination_reason)
      .def_readonly("stage_outcome", &open_spiel::SearchDiagnostics::termination_reason)
      .def_readonly("pred_rem_0", &open_spiel::SearchDiagnostics::pred_rem_0)
      .def_readonly("pred_max_rem_sub", &open_spiel::SearchDiagnostics::pred_max_rem_sub)
      .def_readonly("pred_eff_rem", &open_spiel::SearchDiagnostics::pred_eff_rem)
      .def_readonly("pred_max_raw_rem", &open_spiel::SearchDiagnostics::pred_max_raw_rem);

  py::class_<open_spiel::DuneSearchResult>(di, "DuneSearchResult")
      .def_readonly("policy", &open_spiel::DuneSearchResult::policy)
      .def_readonly("diagnostics", &open_spiel::DuneSearchResult::diagnostics)
      .def_readonly("simulations_completed", &open_spiel::DuneSearchResult::simulations_completed)
      .def_readonly("elapsed_time_ms", &open_spiel::DuneSearchResult::elapsed_time_ms)
      .def_readonly("timeout_status", &open_spiel::DuneSearchResult::timeout_status)
      .def_readonly("used_fallback", &open_spiel::DuneSearchResult::used_fallback)
      .def_readonly("fallback_reason", &open_spiel::DuneSearchResult::fallback_reason)
      .def_readonly("inference_count", &open_spiel::DuneSearchResult::inference_count);

  py::class_<open_spiel::PolicyPriorDetails>(di, "PolicyPriorDetails")
      .def_readonly("ok", &open_spiel::PolicyPriorDetails::ok)
      .def_readonly("actions", &open_spiel::PolicyPriorDetails::actions)
      .def_readonly("probabilities", &open_spiel::PolicyPriorDetails::probabilities)
      .def_readonly("value", &open_spiel::PolicyPriorDetails::value);

  py::class_<open_spiel::PairedActionEvaluationResult>(di, "PairedActionEvaluationResult")
      .def_readonly("u_first", &open_spiel::PairedActionEvaluationResult::u_first)
      .def_readonly("u_second", &open_spiel::PairedActionEvaluationResult::u_second)
      .def_readonly("diff_mean", &open_spiel::PairedActionEvaluationResult::diff_mean)
      .def_readonly("diff_se", &open_spiel::PairedActionEvaluationResult::diff_se)
      .def_readonly("num_worlds", &open_spiel::PairedActionEvaluationResult::num_worlds);

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

  py::class_<open_spiel::DuneAdaptiveRolloutSession>(di, "DuneAdaptiveRolloutSession")
      .def(py::init<const std::string&, const std::string&, int, int, int, int, int, uint64_t, bool, bool, int, int, int, bool, bool, bool>(),
           py::arg("checkpoint_path"),
           py::arg("device_str") = "cuda",
           py::arg("hidden_dim") = 2048,
           py::arg("num_blocks") = 16,
           py::arg("reference_rollouts_per_action") = 64,
           py::arg("leader_rollouts") = 32,
           py::arg("target_batch_size") = 256,
           py::arg("seed") = 20261014ULL,
           py::arg("use_successive_halving") = true,
           py::arg("fixed_top5_candidates") = true,
           py::arg("max_inflight_worlds") = 96,
           py::arg("num_evaluators") = 1,
           py::arg("top5_min_budget") = 576,
           py::arg("all_buy_candidates") = true,
           py::arg("enable_fp8_trunk") = false,
           py::arg("search_v2") = false)
      .def("search", &open_spiel::DuneAdaptiveRolloutSession::Search,
           py::arg("state"), py::arg("remaining_time_ms") = -1.0)
      .def("evaluate_action_pair_terminal",
           &open_spiel::DuneAdaptiveRolloutSession::EvaluateActionPairTerminal,
           py::arg("state"), py::arg("a_first"), py::arg("a_second"),
           py::arg("num_worlds") = 64, py::arg("seed") = 20261104ULL)
      .def("prior_with_details", &open_spiel::DuneAdaptiveRolloutSession::PriorWithDetails,
           py::arg("state"))
      .def("discard_pending_action", &open_spiel::DuneAdaptiveRolloutSession::DiscardPendingAction)
      .def("reset_session", &open_spiel::DuneAdaptiveRolloutSession::ResetSession, py::arg("reason"))
      .def("set_seed", &open_spiel::DuneAdaptiveRolloutSession::set_seed, py::arg("seed"))
      .def("has_active_session", &open_spiel::DuneAdaptiveRolloutSession::HasActiveSession)
      .def_property_readonly("session_new_simulations_completed",
                             &open_spiel::DuneAdaptiveRolloutSession::session_new_simulations_completed)
      .def_property_readonly("session_elapsed_time_ms",
                             &open_spiel::DuneAdaptiveRolloutSession::session_elapsed_time_ms)
      .def_property("fixed_top5_candidates",
                    &open_spiel::DuneAdaptiveRolloutSession::fixed_top5_candidates,
                    &open_spiel::DuneAdaptiveRolloutSession::set_fixed_top5_candidates)
      .def("set_fixed_top5_candidates",
           &open_spiel::DuneAdaptiveRolloutSession::set_fixed_top5_candidates,
           py::arg("val"))
      .def_property("max_inflight_worlds",
                    &open_spiel::DuneAdaptiveRolloutSession::max_inflight_worlds,
                    &open_spiel::DuneAdaptiveRolloutSession::set_max_inflight_worlds)
      .def("set_max_inflight_worlds",
           &open_spiel::DuneAdaptiveRolloutSession::set_max_inflight_worlds,
           py::arg("val"))
      .def_property_readonly("num_evaluators",
                             &open_spiel::DuneAdaptiveRolloutSession::num_evaluators)
      .def_property("top5_min_budget",
                    &open_spiel::DuneAdaptiveRolloutSession::top5_min_budget,
                    &open_spiel::DuneAdaptiveRolloutSession::set_top5_min_budget)
      .def("set_top5_min_budget",
           &open_spiel::DuneAdaptiveRolloutSession::set_top5_min_budget,
           py::arg("val"))
      .def_property("all_buy_candidates",
                    &open_spiel::DuneAdaptiveRolloutSession::all_buy_candidates,
                    &open_spiel::DuneAdaptiveRolloutSession::set_all_buy_candidates)
      .def("set_all_buy_candidates",
           &open_spiel::DuneAdaptiveRolloutSession::set_all_buy_candidates,
           py::arg("val"))
      .def_property("reserve_ms",
                    &open_spiel::DuneAdaptiveRolloutSession::reserve_ms,
                    &open_spiel::DuneAdaptiveRolloutSession::set_reserve_ms)
      .def("set_reserve_ms",
           &open_spiel::DuneAdaptiveRolloutSession::set_reserve_ms,
           py::arg("val"))
      .def_property("full_search",
                    &open_spiel::DuneAdaptiveRolloutSession::full_search,
                    &open_spiel::DuneAdaptiveRolloutSession::set_full_search)
      .def("set_full_search",
           &open_spiel::DuneAdaptiveRolloutSession::set_full_search,
           py::arg("val"))
      .def_property("search_v2",
                    &open_spiel::DuneAdaptiveRolloutSession::search_v2,
                    &open_spiel::DuneAdaptiveRolloutSession::set_search_v2)
      .def("set_search_v2",
           &open_spiel::DuneAdaptiveRolloutSession::set_search_v2,
           py::arg("val"))
      .def_property("prior_log_ratio_weight",
                    &open_spiel::DuneAdaptiveRolloutSession::prior_log_ratio_weight,
                    &open_spiel::DuneAdaptiveRolloutSession::set_prior_log_ratio_weight)
      .def("set_prior_log_ratio_weight",
           &open_spiel::DuneAdaptiveRolloutSession::set_prior_log_ratio_weight,
           py::arg("val"))
      .def_property("stage1_high_conf_prior_skip",
                    &open_spiel::DuneAdaptiveRolloutSession::stage1_high_conf_prior_skip,
                    &open_spiel::DuneAdaptiveRolloutSession::set_stage1_high_conf_prior_skip)
      .def("set_stage1_high_conf_prior_skip",
           &open_spiel::DuneAdaptiveRolloutSession::set_stage1_high_conf_prior_skip,
           py::arg("val"))
      .def_property("use_event_keyed_crn",
                    &open_spiel::DuneAdaptiveRolloutSession::use_event_keyed_crn,
                    &open_spiel::DuneAdaptiveRolloutSession::set_use_event_keyed_crn)
      .def("set_use_event_keyed_crn",
           &open_spiel::DuneAdaptiveRolloutSession::set_use_event_keyed_crn,
           py::arg("val"))
      .def_property_readonly("ms_per_rollout_ema",
                             &open_spiel::DuneAdaptiveRolloutSession::ms_per_rollout_ema)
      .def_property_readonly("turn_elapsed_search_time_ms",
                             &open_spiel::DuneAdaptiveRolloutSession::turn_elapsed_search_time_ms)
      .def_property("enable_fp8_trunk",
                    &open_spiel::DuneAdaptiveRolloutSession::enable_fp8_trunk,
                    &open_spiel::DuneAdaptiveRolloutSession::set_enable_fp8_trunk)
      .def("set_enable_fp8_trunk",
           &open_spiel::DuneAdaptiveRolloutSession::set_enable_fp8_trunk,
           py::arg("val"))
      .def("get_search_v2_tuning", [](const open_spiel::DuneAdaptiveRolloutSession& self) {
        py::dict d;
        const auto& t = self.search_v2_tuning();
        d["subwave_1a_max_budget_ms"] = t.subwave_1a_max_budget_ms;
        d["legacy_s1_inflight_cap"] = t.legacy_s1_inflight_cap;
        d["seed_s2_ms_with_min_prior"] = t.seed_s2_ms_with_min_prior;
        d["s2_max_w0"] = t.s2_max_w0;
        d["s2_max_wave_worlds"] = t.s2_max_wave_worlds;
        d["s2_inflight_cap"] = t.s2_inflight_cap;
        d["legacy_clamp_session_inflight"] = t.legacy_clamp_session_inflight;
        d["r_stage1_cap"] = t.r_stage1_cap;
        d["s1_budget_frac"] = t.s1_budget_frac;
        d["enable_conditional_3arm_stage2"] = t.enable_conditional_3arm_stage2;
        d["three_arm_min_rem_ms"] = t.three_arm_min_rem_ms;
        d["three_arm_max_s1_gap"] = t.three_arm_max_s1_gap;
        d["three_arm_prune_se"] = t.three_arm_prune_se;
        d["unclocked_c96_3arm_mode"] = t.unclocked_c96_3arm_mode;
        d["unclocked_3arm_min_s2_worlds"] = t.unclocked_3arm_min_s2_worlds;
        d["enable_rem0_turn_reservation"] = t.enable_rem0_turn_reservation;
        return d;
      })
      .def("configure_search_v2_tuning", [](open_spiel::DuneAdaptiveRolloutSession& self,
                                            py::object subwave_1a_max_budget_ms,
                                            py::object legacy_s1_inflight_cap,
                                            py::object seed_s2_ms_with_min_prior,
                                            py::object s2_max_w0,
                                            py::object s2_max_wave_worlds,
                                            py::object s2_inflight_cap,
                                            py::object legacy_clamp_session_inflight,
                                            py::object r_stage1_cap,
                                            py::object s1_budget_frac,
                                            py::object enable_conditional_3arm_stage2,
                                            py::object three_arm_min_rem_ms,
                                            py::object three_arm_max_s1_gap,
                                            py::object three_arm_prune_se,
                                            py::object enable_rem0_turn_reservation,
                                            py::object unclocked_c96_3arm_mode,
                                            py::object unclocked_3arm_min_s2_worlds) {
        auto cfg = self.search_v2_tuning();
        if (!subwave_1a_max_budget_ms.is_none()) cfg.subwave_1a_max_budget_ms = subwave_1a_max_budget_ms.cast<double>();
        if (!legacy_s1_inflight_cap.is_none()) cfg.legacy_s1_inflight_cap = legacy_s1_inflight_cap.cast<bool>();
        if (!seed_s2_ms_with_min_prior.is_none()) cfg.seed_s2_ms_with_min_prior = seed_s2_ms_with_min_prior.cast<bool>();
        if (!s2_max_w0.is_none()) cfg.s2_max_w0 = s2_max_w0.cast<int>();
        if (!s2_max_wave_worlds.is_none()) cfg.s2_max_wave_worlds = s2_max_wave_worlds.cast<int>();
        if (!s2_inflight_cap.is_none()) cfg.s2_inflight_cap = s2_inflight_cap.cast<int>();
        if (!legacy_clamp_session_inflight.is_none()) cfg.legacy_clamp_session_inflight = legacy_clamp_session_inflight.cast<bool>();
        if (!r_stage1_cap.is_none()) cfg.r_stage1_cap = r_stage1_cap.cast<int>();
        if (!s1_budget_frac.is_none()) cfg.s1_budget_frac = s1_budget_frac.cast<double>();
        if (!enable_conditional_3arm_stage2.is_none()) cfg.enable_conditional_3arm_stage2 = enable_conditional_3arm_stage2.cast<bool>();
        if (!three_arm_min_rem_ms.is_none()) cfg.three_arm_min_rem_ms = three_arm_min_rem_ms.cast<double>();
        if (!three_arm_max_s1_gap.is_none()) cfg.three_arm_max_s1_gap = three_arm_max_s1_gap.cast<double>();
        if (!three_arm_prune_se.is_none()) cfg.three_arm_prune_se = three_arm_prune_se.cast<double>();
        if (!enable_rem0_turn_reservation.is_none()) cfg.enable_rem0_turn_reservation = enable_rem0_turn_reservation.cast<bool>();
        if (!unclocked_c96_3arm_mode.is_none()) cfg.unclocked_c96_3arm_mode = unclocked_c96_3arm_mode.cast<bool>();
        if (!unclocked_3arm_min_s2_worlds.is_none()) cfg.unclocked_3arm_min_s2_worlds = unclocked_3arm_min_s2_worlds.cast<int>();
        self.set_search_v2_tuning(cfg);
      },
      py::arg("subwave_1a_max_budget_ms") = py::none(),
      py::arg("legacy_s1_inflight_cap") = py::none(),
      py::arg("seed_s2_ms_with_min_prior") = py::none(),
      py::arg("s2_max_w0") = py::none(),
      py::arg("s2_max_wave_worlds") = py::none(),
      py::arg("s2_inflight_cap") = py::none(),
      py::arg("legacy_clamp_session_inflight") = py::none(),
      py::arg("r_stage1_cap") = py::none(),
      py::arg("s1_budget_frac") = py::none(),
      py::arg("enable_conditional_3arm_stage2") = py::none(),
      py::arg("three_arm_min_rem_ms") = py::none(),
      py::arg("three_arm_max_s1_gap") = py::none(),
      py::arg("three_arm_prune_se") = py::none(),
      py::arg("enable_rem0_turn_reservation") = py::none(),
      py::arg("unclocked_c96_3arm_mode") = py::none(),
      py::arg("unclocked_3arm_min_s2_worlds") = py::none())
      .def("get_ms_per_rollout_by_round", &open_spiel::DuneAdaptiveRolloutSession::GetMsPerRolloutByRound)
      .def("set_ms_per_rollout_by_round", &open_spiel::DuneAdaptiveRolloutSession::SetMsPerRolloutByRound, py::arg("vals"))
      .def("run_direct_search_v2_root", [](open_spiel::DuneAdaptiveRolloutSession& self,
                                           const open_spiel::State& state,
                                           double target_budget_ms,
                                           uint64_t decision_seed,
                                           bool is_purchase,
                                           int override_ref_rollouts) {
        auto res = self.RunDirectSearchV2Root(state, target_budget_ms, decision_seed, is_purchase, override_ref_rollouts);
        py::dict d;
        d["elapsed_ms"] = res.elapsed_ms;
        d["stage1_elapsed_ms"] = res.stage1_elapsed_ms;
        d["stage2_elapsed_ms"] = res.stage2_elapsed_ms;
        d["new_rollouts_executed"] = res.new_rollouts_executed;
        d["stage1_rollouts_executed"] = res.stage1_rollouts_executed;
        d["stage2_rollouts_executed"] = res.stage2_rollouts_executed;
        d["stage1_planned_worlds"] = res.stage1_planned_worlds;
        d["stage1_kept_worlds"] = res.stage1_kept_worlds;
        d["stage2_planned_worlds"] = res.stage2_planned_worlds;
        d["stage2_kept_worlds"] = res.stage2_kept_worlds;
        d["termination_reason"] = res.termination_reason;
        d["overridden"] = res.overridden;
        d["usable_override"] = res.usable_override;
        d["chosen_first_action"] = res.chosen_first_action;
        d["chosen_graft_action"] = res.chosen_graft_action;
        d["chosen_second_action"] = res.chosen_second_action;
        d["raw_first_action"] = res.raw_first_action;
        d["raw_graft_action"] = res.raw_graft_action;
        d["raw_second_action"] = res.raw_second_action;
        d["chosen_candidate_index"] = res.chosen_candidate_index;
        d["num_candidates"] = res.num_candidates;
        d["est_ms_per_sim_used"] = res.est_ms_per_sim_used;
        return d;
      }, py::arg("state"), py::arg("target_budget_ms"), py::arg("decision_seed"), py::arg("is_purchase") = false, py::arg("override_ref_rollouts") = -1)
      .def("run_direct_search_v1_sequential_root", [](open_spiel::DuneAdaptiveRolloutSession& self,
                                                      const open_spiel::State& state,
                                                      double target_budget_ms,
                                                      uint64_t decision_seed,
                                                      int override_ref_rollouts) {
        auto res = self.RunDirectSearchV1SequentialRoot(state, target_budget_ms, decision_seed, override_ref_rollouts);
        py::dict d;
        d["elapsed_ms"] = res.elapsed_ms;
        d["new_rollouts_executed"] = res.new_rollouts_executed;
        d["overridden"] = res.overridden;
        d["usable_override"] = res.usable_override;
        d["chosen_first_action"] = res.chosen_first_action;
        d["chosen_graft_action"] = res.chosen_graft_action;
        d["chosen_second_action"] = res.chosen_second_action;
        d["raw_first_action"] = res.raw_first_action;
        d["raw_graft_action"] = res.raw_graft_action;
        d["raw_second_action"] = res.raw_second_action;
        return d;
      }, py::arg("state"), py::arg("target_budget_ms"), py::arg("decision_seed"), py::arg("override_ref_rollouts") = -1)
      .def("evaluate_joint_candidates_paired", [](open_spiel::DuneAdaptiveRolloutSession& self,
                                                 const open_spiel::State& state,
                                                 const std::vector<open_spiel::Action>& first_actions,
                                                 const std::vector<open_spiel::Action>& graft_actions,
                                                 const std::vector<open_spiel::Action>& second_actions,
                                                 int num_worlds,
                                                 uint64_t decision_seed) {
        auto res = self.EvaluateJointCandidatesPaired(state, first_actions, graft_actions, second_actions, num_worlds, decision_seed);
        py::dict d;
        d["mean_utilities"] = res.mean_utilities;
        d["per_candidate_rollouts"] = res.per_candidate_rollouts;
        d["elapsed_ms"] = res.elapsed_ms;
        d["rollouts_executed"] = res.rollouts_executed;
        return d;
      }, py::arg("state"), py::arg("first_actions"), py::arg("graft_actions"), py::arg("second_actions"), py::arg("num_worlds"), py::arg("decision_seed"));

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

  di.def("set_force_legacy_v5_scorer", [](bool val) {
    open_spiel::dune_semantic::SetForceLegacyV5Scorer(val);
  }, py::arg("val"));

  di.def("get_force_legacy_v5_scorer", []() {
    return open_spiel::dune_semantic::GetForceLegacyV5Scorer();
  });

  di.def("run_v5_scorer_zero_sync_probe", [](int batch_size, int u_states, bool force_legacy) {
    if (!torch::cuda::is_available()) {
      open_spiel::SpielFatalError("CUDA is required for run_v5_scorer_zero_sync_probe");
    }
    torch::Device cuda_dev(torch::kCUDA);

    static std::shared_ptr<open_spiel::dune_semantic::SemanticActionScorerImpl> probe_scorer;
    static open_spiel::dune_semantic::SemanticScorerBatchWorkspace probe_ws;
    if (!probe_scorer) {
      probe_scorer = std::make_shared<open_spiel::dune_semantic::SemanticActionScorerImpl>(
          /*enable_f=*/true, /*slot_indexed_f=*/true, /*enable_v5=*/true);
      probe_scorer->to(cuda_dev);
      probe_scorer->eval();
      probe_scorer->RefreshBf16Cache();
    }
    probe_ws.EnsureCapacity(batch_size * 16, cuda_dev);

    std::vector<open_spiel::dune_semantic::CandidateActionData> cad_storage(batch_size);
    std::vector<const open_spiel::dune_semantic::CandidateActionData*> batch_actions(batch_size);

    for (int b = 0; b < batch_size; ++b) {
      auto& cad = cad_storage[b];
      const int num_cands = 8;
      cad.actions.resize(num_cands);
      cad.card_ids.resize(num_cands, 0);
      cad.space_ids.resize(num_cands, 0);
      cad.supported.resize(num_cands, 1);
      cad.roles.resize(num_cands, open_spiel::dune_semantic::ActionRole::kBoardSpace);
      cad.features.resize(num_cands * open_spiel::dune_semantic::kSemanticFeatDim, 0.0f);
      cad.v5_families.resize(num_cands, -1);
      cad.v5_item_ids.resize(num_cands, 279);
      cad.v5_ctx_features.resize(num_cands * open_spiel::dune_semantic::kV5CtxDim, 0.0f);

      const bool is_v5_row = (b < u_states);
      for (int j = 0; j < num_cands; ++j) {
        cad.actions[j] = static_cast<open_spiel::Action>(100 + j);
        cad.card_ids[j] = (b * 7 + j) % open_spiel::dune_semantic::kCardVocabSize;
        cad.space_ids[j] = (b * 13 + j) % open_spiel::dune_semantic::kSpaceVocabSize;
        if (is_v5_row) {
          cad.v5_families[j] = j % open_spiel::dune_semantic::kV5NumFamilies;
          cad.v5_item_ids[j] = (b * 17 + j * 3) % open_spiel::dune_semantic::kV5ItemVocabSize;
        }
      }
      batch_actions[b] = &cad;
    }

    open_spiel::dune_semantic::SetForceLegacyV5Scorer(force_legacy);

    torch::InferenceMode no_grad(true);
    at::autocast::set_autocast_enabled(at::kCUDA, true);
    at::autocast::set_autocast_dtype(at::kCUDA, torch::kBFloat16);

    static torch::Tensor static_trunk;
    static torch::Tensor static_logits;
    if (!static_trunk.defined() || static_trunk.size(0) != batch_size) {
      static_trunk = torch::randn({batch_size, 2048}, torch::TensorOptions().device(cuda_dev).dtype(torch::kFloat32));
      static_logits = torch::zeros({batch_size, 2391}, torch::TensorOptions().device(cuda_dev).dtype(torch::kFloat32));
    }

    auto h_p = probe_scorer->ProjectTrunk(static_trunk);
    open_spiel::dune_semantic::ApplySemanticScorerBatch(
        probe_scorer, static_trunk, batch_actions, static_logits, cuda_dev, &probe_ws,
        /*allow_inplace=*/true, h_p);

    at::autocast::set_autocast_enabled(at::kCUDA, false);
  }, py::arg("batch_size"), py::arg("u_states"), py::arg("force_legacy"));

  di.def("run_v5_scorer_parity_suite", [](const std::string& checkpoint_path,
                                          const std::vector<std::string>& serialized_states) {
    if (!torch::cuda::is_available()) {
      open_spiel::SpielFatalError("CUDA is required for run_v5_scorer_parity_suite");
    }
    torch::Device cuda_dev(torch::kCUDA);

    double fp64_fwd = 0.0, fp64_bwd = 0.0;
    double fp32_fwd = 0.0, fp32_bwd = 0.0;
    double fp32_tf32_fwd = 0.0, fp32_tf32_bwd = 0.0;
    double bf16_max_logit_diff = 0.0;
    int64_t bf16_argmax_flips = 0;
    const int64_t total_states = static_cast<int64_t>(serialized_states.size());

    {
      py::gil_scoped_release release;

      const int B = 256;
      const std::vector<int> test_u = {0, 4, 64, 256};

      auto run_dtype_parity = [&](at::ScalarType dtype, bool allow_tf32, double& max_fwd, double& max_bwd) {
        max_fwd = 0.0;
        max_bwd = 0.0;
        bool orig_tf32 = at::globalContext().allowTF32CuBLAS();
        at::globalContext().setAllowTF32CuBLAS(allow_tf32);

        for (int U : test_u) {
          std::vector<open_spiel::dune_semantic::CandidateActionData> cad_storage(B);
          std::vector<const open_spiel::dune_semantic::CandidateActionData*> batch_actions(B);
          for (int b = 0; b < B; ++b) {
            auto& cad = cad_storage[b];
            const int num_cands = 8;
            cad.actions.resize(num_cands);
            cad.card_ids.resize(num_cands);
            cad.space_ids.resize(num_cands);
            cad.supported.resize(num_cands, 1);
            cad.roles.resize(num_cands, open_spiel::dune_semantic::ActionRole::kBoardSpace);
            cad.features.resize(num_cands * open_spiel::dune_semantic::kSemanticFeatDim, 0.01f);
            cad.v5_families.resize(num_cands, -1);
            cad.v5_item_ids.resize(num_cands, 279);
            cad.v5_ctx_features.resize(num_cands * open_spiel::dune_semantic::kV5CtxDim, 0.0f);

            const bool is_v5_row = (b < U);
            for (int j = 0; j < num_cands; ++j) {
              cad.actions[j] = static_cast<open_spiel::Action>(100 + j);
              cad.card_ids[j] = (b * 7 + j) % open_spiel::dune_semantic::kCardVocabSize;
              cad.space_ids[j] = (b * 13 + j) % open_spiel::dune_semantic::kSpaceVocabSize;
              if (is_v5_row) {
                cad.v5_families[j] = j % open_spiel::dune_semantic::kV5NumFamilies;
                cad.v5_item_ids[j] = (b * 17 + j * 3) % open_spiel::dune_semantic::kV5ItemVocabSize;
                for (int f = 0; f < open_spiel::dune_semantic::kV5CtxDim; ++f) {
                  cad.v5_ctx_features[j * open_spiel::dune_semantic::kV5CtxDim + f] = 0.02f * ((f + j) % 10);
                }
              }
            }
            batch_actions[b] = &cad;
          }

          torch::manual_seed(2026 + U);
          auto scorer_leg = std::make_shared<open_spiel::dune_semantic::SemanticActionScorerImpl>(
              /*enable_f=*/true, /*slot_indexed_f=*/true, /*enable_v5=*/true);
          scorer_leg->to(cuda_dev);
          scorer_leg->to(dtype);
          scorer_leg->train();

          auto scorer_fast = std::make_shared<open_spiel::dune_semantic::SemanticActionScorerImpl>(
              /*enable_f=*/true, /*slot_indexed_f=*/true, /*enable_v5=*/true);
          scorer_fast->to(cuda_dev);
          scorer_fast->to(dtype);
          scorer_fast->train();

          {
            torch::NoGradGuard no_grad;
            auto leg_params = scorer_leg->named_parameters();
            auto fast_params = scorer_fast->named_parameters();
            for (const auto& item : leg_params) {
              fast_params[item.key()].copy_(item.value());
            }
          }

          auto opts = torch::TensorOptions().device(cuda_dev).dtype(dtype);
          torch::Tensor trunk_base = torch::randn({B, 2048}, opts);
          torch::Tensor logits_base = torch::randn({B, 2391}, opts);
          torch::Tensor target = torch::randn({B, 2391}, opts);

          // 1) Legacy pass
          open_spiel::dune_semantic::SetForceLegacyV5Scorer(true);
          torch::Tensor trunk_leg = trunk_base.clone().detach().requires_grad_(true);
          torch::Tensor logits_leg = logits_base.clone();
          open_spiel::dune_semantic::ApplySemanticScorerBatch(scorer_leg, trunk_leg, batch_actions, logits_leg, cuda_dev);
          torch::Tensor loss_leg = (logits_leg * target).sum();
          loss_leg.backward();

          // 2) Fast pass
          open_spiel::dune_semantic::SetForceLegacyV5Scorer(false);
          torch::Tensor trunk_fast = trunk_base.clone().detach().requires_grad_(true);
          torch::Tensor logits_fast = logits_base.clone();
          open_spiel::dune_semantic::ApplySemanticScorerBatch(scorer_fast, trunk_fast, batch_actions, logits_fast, cuda_dev);
          torch::Tensor loss_fast = (logits_fast * target).sum();
          loss_fast.backward();

          double f_diff = (logits_leg - logits_fast).abs().max().item<double>();
          max_fwd = std::max(max_fwd, f_diff);

          double b_diff = (trunk_leg.grad() - trunk_fast.grad()).abs().max().item<double>();
          auto leg_params = scorer_leg->named_parameters();
          auto fast_params = scorer_fast->named_parameters();
          for (const auto& item : leg_params) {
            if (item.value().grad().defined()) {
              double p_diff = (item.value().grad() - fast_params[item.key()].grad()).abs().max().item<double>();
              b_diff = std::max(b_diff, p_diff);
            }
          }
          max_bwd = std::max(max_bwd, b_diff);
        }
        at::globalContext().setAllowTF32CuBLAS(orig_tf32);
      };

      run_dtype_parity(torch::kFloat64, false, fp64_fwd, fp64_bwd);
      run_dtype_parity(torch::kFloat32, false, fp32_fwd, fp32_bwd);
      run_dtype_parity(torch::kFloat32, true, fp32_tf32_fwd, fp32_tf32_bwd);

      // 2) Real 1024-state checkpoint evaluation in BF16
      int64_t ckpt_input_dim = 10445;
      {
        std::string json_path = std::filesystem::path(checkpoint_path).replace_extension(".json").string();
        std::ifstream jf(json_path);
        if (jf.good()) {
          std::string content((std::istreambuf_iterator<char>(jf)), std::istreambuf_iterator<char>());
          auto pos = content.find("\"observation_dim\":");
          if (pos != std::string::npos) {
            ckpt_input_dim = std::stoll(content.substr(pos + 18));
          }
        }
      }
      auto model = std::make_shared<open_spiel::SharedDunePolicyValueNetImpl>(
          ckpt_input_dim, 2048, 2391, 16,
          /*use_nonlinear=*/false, /*with_aux_heads=*/false, /*head_init_seed=*/0,
          /*with_semantic_scorer=*/true);
      open_spiel::LoadModelCheckpointRobust(model, checkpoint_path, cuda_dev);
      model->to(cuda_dev);
      model->eval();
      model->RefreshBf16InferenceCache();

      auto game = open_spiel::LoadGame("dune_imperium");
      const size_t N = serialized_states.size();
      std::vector<std::vector<float>> all_obs(N);
      std::vector<std::vector<open_spiel::Action>> all_legals(N);
      std::vector<open_spiel::dune_semantic::CandidateActionData> all_cands(N);

      for (size_t i = 0; i < N; ++i) {
        auto state = game->DeserializeState(serialized_states[i]);
        const auto* dune = dynamic_cast<const open_spiel::dune_imperium::DuneImperiumState*>(state.get());
        all_obs[i] = dune->InformationStateTensorWithAppendix(
            dune->CurrentPlayer(),
            model->market_appendix_mode_,
            model->deck_pool_semantics_);
        all_legals[i] = dune->LegalActions();
        open_spiel::dune_semantic::ExtractCandidateDescriptors(
            *dune, all_legals[i], &all_cands[i], model->semantic_descriptor_schema_);
      }

      open_spiel::dune_semantic::SemanticScorerBatchWorkspace eval_ws;
      eval_ws.EnsureCapacity(256 * 16, cuda_dev);

      for (size_t start = 0; start < N; start += B) {
        size_t cur_B = std::min<size_t>(B, N - start);
        std::vector<float> batch_obs_flat(cur_B * ckpt_input_dim);
        for (size_t i = 0; i < cur_B; ++i) {
          size_t copy_len = std::min<size_t>(all_obs[start + i].size(), static_cast<size_t>(ckpt_input_dim));
          std::memcpy(batch_obs_flat.data() + i * ckpt_input_dim,
                      all_obs[start + i].data(),
                      copy_len * sizeof(float));
          if (copy_len < static_cast<size_t>(ckpt_input_dim)) {
            std::memset(batch_obs_flat.data() + i * ckpt_input_dim + copy_len, 0,
                        (ckpt_input_dim - copy_len) * sizeof(float));
          }
        }
        std::vector<const open_spiel::dune_semantic::CandidateActionData*> batch_cands(cur_B);
        for (size_t i = 0; i < cur_B; ++i) {
          batch_cands[i] = &all_cands[start + i];
        }

        torch::InferenceMode no_grad(true);
        at::autocast::set_autocast_enabled(at::kCUDA, true);
        at::autocast::set_autocast_dtype(at::kCUDA, torch::kBFloat16);

        auto x = torch::from_blob(batch_obs_flat.data(), {static_cast<int64_t>(cur_B), ckpt_input_dim},
                                  torch::TensorOptions().dtype(torch::kFloat32)).to(cuda_dev);
        auto outputs = model->forward(x);
        auto h_p = model->semantic_scorer_->ProjectTrunk(outputs.trunk);

        // Legacy pass
        open_spiel::dune_semantic::SetForceLegacyV5Scorer(true);
        auto logits_leg = outputs.logits.clone();
        open_spiel::dune_semantic::ApplySemanticScorerBatch(
            model->semantic_scorer_, outputs.trunk, batch_cands, logits_leg, cuda_dev, &eval_ws,
            /*allow_inplace=*/true, h_p);

        // Fast pass
        open_spiel::dune_semantic::SetForceLegacyV5Scorer(false);
        auto logits_fast = outputs.logits.clone();
        open_spiel::dune_semantic::ApplySemanticScorerBatch(
            model->semantic_scorer_, outputs.trunk, batch_cands, logits_fast, cuda_dev, &eval_ws,
            /*allow_inplace=*/true, h_p);

        at::autocast::set_autocast_enabled(at::kCUDA, false);

        auto leg_cpu = logits_leg.to(torch::kFloat32).to(torch::kCPU);
        auto fast_cpu = logits_fast.to(torch::kFloat32).to(torch::kCPU);
        const float* leg_ptr = leg_cpu.data_ptr<float>();
        const float* fast_ptr = fast_cpu.data_ptr<float>();
        const int64_t action_dim = logits_leg.size(1);

        for (size_t i = 0; i < cur_B; ++i) {
          const auto& legals = all_legals[start + i];
          if (legals.empty()) continue;

          double max_leg = -1e9, max_fast = -1e9;
          open_spiel::Action best_leg = legals[0], best_fast = legals[0];

          for (open_spiel::Action a : legals) {
            float l_val = leg_ptr[i * action_dim + a];
            float f_val = fast_ptr[i * action_dim + a];
            double diff = std::abs(static_cast<double>(l_val) - static_cast<double>(f_val));
            bf16_max_logit_diff = std::max(bf16_max_logit_diff, diff);

            if (l_val > max_leg) { max_leg = l_val; best_leg = a; }
            if (f_val > max_fast) { max_fast = f_val; best_fast = a; }
          }
          if (best_leg != best_fast) {
            bf16_argmax_flips++;
          }
        }
      }

      open_spiel::dune_semantic::SetForceLegacyV5Scorer(false);
    }

    py::dict result;
    result["fp64_fwd_max_diff"] = fp64_fwd;
    result["fp64_bwd_max_diff"] = fp64_bwd;
    result["fp32_fwd_max_diff"] = fp32_fwd;
    result["fp32_bwd_max_diff"] = fp32_bwd;
    result["fp32_tf32_fwd_max_diff"] = fp32_tf32_fwd;
    result["fp32_tf32_bwd_max_diff"] = fp32_tf32_bwd;
    result["bf16_max_logit_diff"] = bf16_max_logit_diff;
    result["bf16_argmax_flips"] = bf16_argmax_flips;
    result["bf16_total_states"] = total_states;

    return result;
  }, py::arg("checkpoint_path"), py::arg("serialized_states"));

  di.def("benchmark_v5_scorer_batch", [](const std::string& checkpoint_path,
                                         const std::vector<std::string>& v5_serialized,
                                         const std::vector<std::string>& non_v5_serialized,
                                         int u_states, int batch_size,
                                         int warmup_iters, int timed_iters,
                                         bool force_legacy) {
    if (!torch::cuda::is_available()) {
      open_spiel::SpielFatalError("CUDA is required for benchmark_v5_scorer_batch");
    }
    torch::Device cuda_dev(torch::kCUDA);

    double elapsed_ms = 0.0;
    double us_per_batch = 0.0;
    double checksum = 0.0;
    int64_t total_cands = 0;
    int64_t m_v5 = 0;

    {
      py::gil_scoped_release release;

      struct CachedBench {
        std::string path;
        std::shared_ptr<open_spiel::SharedDunePolicyValueNetImpl> model;
      };
      static CachedBench cached_bench;
      if (cached_bench.path != checkpoint_path || !cached_bench.model) {
        int64_t input_dim = 10445;
        std::string json_path = std::filesystem::path(checkpoint_path).replace_extension(".json").string();
        std::ifstream jf(json_path);
        if (jf.good()) {
          std::string content((std::istreambuf_iterator<char>(jf)), std::istreambuf_iterator<char>());
          auto pos = content.find("\"observation_dim\":");
          if (pos != std::string::npos) {
            input_dim = std::stoll(content.substr(pos + 18));
          }
        }
        auto m = std::make_shared<open_spiel::SharedDunePolicyValueNetImpl>(
            input_dim, 2048, 2391, 16,
            /*use_nonlinear=*/false, /*with_aux_heads=*/false, /*head_init_seed=*/0,
            /*with_semantic_scorer=*/true);
        open_spiel::LoadModelCheckpointRobust(m, checkpoint_path, cuda_dev);
        m->to(cuda_dev);
        m->eval();
        m->RefreshBf16InferenceCache();
        cached_bench.path = checkpoint_path;
        cached_bench.model = m;
      }
      auto model = cached_bench.model;
      const int64_t input_dim = model->input_layer->options.in_features();

      auto game = open_spiel::LoadGame("dune_imperium");

      std::vector<bool> is_v5_slot(batch_size, false);
      if (u_states >= batch_size) {
        std::fill(is_v5_slot.begin(), is_v5_slot.end(), true);
      } else if (u_states > 0) {
        for (int i = 0; i < u_states; ++i) {
          int slot = (i * batch_size) / u_states;
          is_v5_slot[slot] = true;
        }
      }

      std::vector<open_spiel::dune_semantic::CandidateActionData> cad_storage(batch_size);
      std::vector<const open_spiel::dune_semantic::CandidateActionData*> batch_actions(batch_size);
      std::vector<float> obs_flat(batch_size * input_dim);

      size_t v5_idx_counter = 0;
      size_t non_v5_idx_counter = 0;

      for (int b = 0; b < batch_size; ++b) {
        std::string s_str;
        if (is_v5_slot[b]) {
          s_str = v5_serialized[v5_idx_counter++ % v5_serialized.size()];
        } else {
          s_str = non_v5_serialized[non_v5_idx_counter++ % non_v5_serialized.size()];
        }
        auto state = game->DeserializeState(s_str);
        const auto* dune = dynamic_cast<const open_spiel::dune_imperium::DuneImperiumState*>(state.get());
        auto obs = dune->InformationStateTensorWithAppendix(
            dune->CurrentPlayer(),
            model->market_appendix_mode_,
            model->deck_pool_semantics_);
        size_t copy_len = std::min<size_t>(obs.size(), static_cast<size_t>(input_dim));
        std::memcpy(obs_flat.data() + b * input_dim, obs.data(), copy_len * sizeof(float));
        if (copy_len < static_cast<size_t>(input_dim)) {
          std::memset(obs_flat.data() + b * input_dim + copy_len, 0, (input_dim - copy_len) * sizeof(float));
        }

        auto legals = dune->LegalActions();
        open_spiel::dune_semantic::ExtractCandidateDescriptors(
            *dune, legals, &cad_storage[b], model->semantic_descriptor_schema_);
        batch_actions[b] = &cad_storage[b];
        total_cands += cad_storage[b].size();
        for (int64_t fam : cad_storage[b].v5_families) {
          if (fam >= 0) ++m_v5;
        }
      }

      torch::InferenceMode no_grad(true);
      at::autocast::set_autocast_enabled(at::kCUDA, true);
      at::autocast::set_autocast_dtype(at::kCUDA, torch::kBFloat16);

      auto x = torch::from_blob(obs_flat.data(), {batch_size, input_dim},
                                torch::TensorOptions().dtype(torch::kFloat32)).to(cuda_dev);
      auto outputs = model->forward(x);
      torch::Tensor trunk = outputs.trunk;
      torch::Tensor h_p = model->semantic_scorer_->ProjectTrunk(trunk);
      torch::Tensor inout_logits = outputs.logits.clone();

      std::vector<open_spiel::dune_semantic::SemanticScorerBatchWorkspace> ws_ring(4);
      for (auto& ws : ws_ring) {
        ws.EnsureCapacity(total_cands * 2, cuda_dev);
      }

      open_spiel::dune_semantic::SetForceLegacyV5Scorer(force_legacy);

      // Warmup
      for (int it = 0; it < warmup_iters; ++it) {
        open_spiel::dune_semantic::ApplySemanticScorerBatch(
            model->semantic_scorer_, trunk, batch_actions, inout_logits, cuda_dev,
            &ws_ring[it & 3], /*allow_inplace=*/true, h_p);
      }
      c10::cuda::getCurrentCUDAStream().synchronize();

      // Timed iterations
      auto t0 = std::chrono::high_resolution_clock::now();
      for (int it = 0; it < timed_iters; ++it) {
        open_spiel::dune_semantic::ApplySemanticScorerBatch(
            model->semantic_scorer_, trunk, batch_actions, inout_logits, cuda_dev,
            &ws_ring[it & 3], /*allow_inplace=*/true, h_p);
      }
      c10::cuda::getCurrentCUDAStream().synchronize();
      auto t1 = std::chrono::high_resolution_clock::now();

      at::autocast::set_autocast_enabled(at::kCUDA, false);

      elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
      us_per_batch = (elapsed_ms * 1000.0) / timed_iters;
      checksum = inout_logits.sum().item<double>();
      if (std::isnan(checksum) || std::isinf(checksum)) {
        open_spiel::SpielFatalError("Non-finite checksum in benchmark_v5_scorer_batch!");
      }

      open_spiel::dune_semantic::SetForceLegacyV5Scorer(false);
    }

    py::dict res;
    res["elapsed_ms"] = elapsed_ms;
    res["us_per_batch"] = us_per_batch;
    res["total_cands"] = total_cands;
    res["m_v5"] = m_v5;
    res["checksum"] = checksum;
    return res;
  }, py::arg("checkpoint_path"), py::arg("v5_serialized"),
     py::arg("non_v5_serialized"), py::arg("u_states"),
     py::arg("batch_size"), py::arg("warmup_iters"),
     py::arg("timed_iters"), py::arg("force_legacy"));
#endif
}
