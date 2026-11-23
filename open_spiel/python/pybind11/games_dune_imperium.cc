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

#include "open_spiel/games/dune_imperium/dune_imperium.h"
#include "open_spiel/games/dune_imperium/dune_imperium_common.h"
#include "open_spiel/python/pybind11/games_dune_imperium_internal.h"

namespace py = ::pybind11;

using open_spiel::dune_imperium::Faction;
using open_spiel::dune_imperium::GamePhase;
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

  open_spiel::dune_imperium_pybind::RegisterDuneImperiumState(m, di);

#ifdef OPEN_SPIEL_BUILD_WITH_LIBTORCH
  open_spiel::dune_imperium_pybind::RegisterDuneImperiumSearch(di);
  open_spiel::dune_imperium_pybind::RegisterDuneImperiumV5Scorer(di);
#endif
}
