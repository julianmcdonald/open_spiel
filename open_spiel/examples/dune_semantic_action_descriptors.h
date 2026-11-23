#ifndef OPEN_SPIEL_EXAMPLES_DUNE_SEMANTIC_ACTION_DESCRIPTORS_H_
#define OPEN_SPIEL_EXAMPLES_DUNE_SEMANTIC_ACTION_DESCRIPTORS_H_

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <optional>
#include <string>
#include <type_traits>
#include <vector>

#include "open_spiel/spiel.h"
#include "open_spiel/games/dune_imperium/dune_imperium.h"
#include "open_spiel/games/dune_imperium/dune_imperium_cards.h"
#include "open_spiel/games/dune_imperium/dune_imperium_board.h"
#include "open_spiel/games/dune_imperium/dune_imperium_common.h"
#include "open_spiel/games/dune_imperium/dune_imperium_content.h"

namespace open_spiel {
namespace dune_semantic {

inline constexpr const char* kDescriptorSchemaVersionV2 = "semantic_action_v2";
inline constexpr const char* kDescriptorSchemaVersionV3 = "semantic_action_v3";
inline constexpr const char* kDescriptorSchemaVersionV4 = "semantic_action_v4";
inline constexpr const char* kDescriptorSchemaVersionV5 = "semantic_action_v5";
inline constexpr const char* kDescriptorSchemaVersion = kDescriptorSchemaVersionV4;

inline constexpr int kV5ItemVocabSize = 280;
inline constexpr int kV5EmbedDim = 288;
inline constexpr int kV5CtxDim = 8;
inline constexpr int kV5NumFamilies = 5;

inline constexpr int kIntrigueCardVocabOffset = 128;

enum class ActionRole : uint8_t {
  kUnsupported = 0,
  kPrimaryCard = 1,
  kBoardSpace = 2,
  kMarketPurchase = 3,
  kIndexedIntrigue = 4,
  kGraftSolo = 5,
  kHundroOffer = 6,
  kTleilaxuPurchase = 7,
};

inline constexpr int kCardVocabSize = 256;
inline constexpr int kCardEmbedDim = 32;
inline constexpr int kSpaceVocabSize = 32;
inline constexpr int kSpaceEmbedDim = 16;
inline constexpr int kSemanticFeatDim = 56;
inline constexpr int kTotalDescDim = kSemanticFeatDim + kCardEmbedDim + kSpaceEmbedDim; // 104
inline constexpr int kScorerHiddenDim = 256;

// Structure holding packed candidate action data for a single game decision state.
struct CandidateActionData {
  std::vector<Action> actions;
  std::vector<float> features;       // Flattened [num_actions * kSemanticFeatDim]
  std::vector<int64_t> card_ids;     // [num_actions] in [0, kCardVocabSize - 1]
  std::vector<int64_t> space_ids;    // [num_actions] in [0, kSpaceVocabSize - 1]
  std::vector<uint8_t> supported;    // [num_actions] 1 if supported role, 0 otherwise
  std::vector<ActionRole> roles;     // [num_actions] ActionRole
  std::vector<int64_t> v5_item_ids;  // [num_actions] default 279
  std::vector<int64_t> v5_families;  // [num_actions] default -1
  std::vector<float> v5_ctx_features;// [num_actions * kV5CtxDim] default 0.0f

  bool empty() const { return actions.empty(); }
  size_t size() const { return actions.size(); }
  void clear() {
    actions.clear();
    features.clear();
    card_ids.clear();
    space_ids.clear();
    supported.clear();
    roles.clear();
    v5_item_ids.clear();
    v5_families.clear();
    v5_ctx_features.clear();
  }
};

// Extracts candidate action descriptors and categorical IDs for all legal actions at the given state.
inline void ExtractCandidateDescriptors(
    const dune_imperium::DuneImperiumState& state,
    const std::vector<Action>& legal_actions,
    CandidateActionData* out_data,
    const std::string& schema_version = kDescriptorSchemaVersionV3) {
  if (out_data == nullptr) return;
  out_data->clear();
  const size_t num_acts = legal_actions.size();
  out_data->actions = legal_actions;
  out_data->features.assign(num_acts * kSemanticFeatDim, 0.0f);
  out_data->card_ids.assign(num_acts, 0);
  out_data->space_ids.assign(num_acts, 0);
  out_data->supported.assign(num_acts, 0);
  out_data->roles.assign(num_acts, ActionRole::kUnsupported);
  out_data->v5_item_ids.assign(num_acts, 279);
  out_data->v5_families.assign(num_acts, -1);
  out_data->v5_ctx_features.assign(num_acts * kV5CtxDim, 0.0f);

  const Player cur_player = state.CurrentPlayer();
  const int leader_id = (cur_player >= 0 && cur_player < state.NumPlayers())
                            ? state.PlayerLeader(cur_player)
                            : 0;

  for (size_t i = 0; i < num_acts; ++i) {
    const Action action = legal_actions[i];
    float* f = &out_data->features[i * kSemanticFeatDim];

    // 1. Primary-Card Selection (SelectAgentCard)
    if (action >= dune_imperium::kActionSelectAgentCard0 &&
        action < dune_imperium::kActionSelectAgentCard0 + 256) {
      const int card_id = action - dune_imperium::kActionSelectAgentCard0;
      out_data->card_ids[i] = std::clamp(card_id, 0, kCardVocabSize - 1);
      out_data->space_ids[i] = 0;
      out_data->supported[i] = 1;
      out_data->roles[i] = ActionRole::kPrimaryCard;

      // Role one-hot
      f[1] = 1.0f; // is_primary_card

      const auto* card = dune_imperium::FindImperiumCardById(card_id);
      if (card != nullptr) {
        f[9] = static_cast<float>(card->persuasion_cost) / 10.0f;
        // Access mask (7 bits)
        f[10] = (card->access_mask & dune_imperium::kAccessEmperor) ? 1.0f : 0.0f;
        f[11] = (card->access_mask & dune_imperium::kAccessSpacingGuild) ? 1.0f : 0.0f;
        f[12] = (card->access_mask & dune_imperium::kAccessBeneGesserit) ? 1.0f : 0.0f;
        f[13] = (card->access_mask & dune_imperium::kAccessFremen) ? 1.0f : 0.0f;
        f[14] = (card->access_mask & dune_imperium::kAccessSpiceTrade) ? 1.0f : 0.0f;
        f[15] = (card->access_mask & dune_imperium::kAccessLandsraad) ? 1.0f : 0.0f;
        f[16] = (card->access_mask & dune_imperium::kAccessCity) ? 1.0f : 0.0f;
        // Faction mask (4 bits)
        f[17] = (card->faction_mask & (1u << 0)) ? 1.0f : 0.0f;
        f[18] = (card->faction_mask & (1u << 1)) ? 1.0f : 0.0f;
        f[19] = (card->faction_mask & (1u << 2)) ? 1.0f : 0.0f;
        f[20] = (card->faction_mask & (1u << 3)) ? 1.0f : 0.0f;
        // Card stats
        f[21] = static_cast<float>(card->reveal_persuasion) / 10.0f;
        f[22] = static_cast<float>(card->reveal_swords) / 5.0f;
        f[23] = static_cast<float>(card->agent_card_draw) / 3.0f;
        f[24] = static_cast<float>(card->reveal_card_draw) / 3.0f;
        f[25] = static_cast<float>(card->agent_troops) / 5.0f;
        f[26] = static_cast<float>(card->reveal_troops) / 5.0f;
      }
      continue;
    }

    // 2. Board-Space Placement
    const auto* space = dune_imperium::FindAgentSpace(action);
    if (space != nullptr) {
      out_data->card_ids[i] = 0;
      out_data->space_ids[i] = std::clamp(space->board_index, 0, kSpaceVocabSize - 1);
      out_data->supported[i] = 1;
      out_data->roles[i] = ActionRole::kBoardSpace;

      // Role one-hot
      f[2] = 1.0f; // is_board_space

      // Space semantics
      f[27] = (space->required_access_mask & dune_imperium::kAccessEmperor) ? 1.0f : 0.0f;
      f[28] = (space->required_access_mask & dune_imperium::kAccessSpacingGuild) ? 1.0f : 0.0f;
      f[29] = (space->required_access_mask & dune_imperium::kAccessBeneGesserit) ? 1.0f : 0.0f;
      f[30] = (space->required_access_mask & dune_imperium::kAccessFremen) ? 1.0f : 0.0f;
      f[31] = (space->required_access_mask & dune_imperium::kAccessSpiceTrade) ? 1.0f : 0.0f;
      f[32] = (space->required_access_mask & dune_imperium::kAccessLandsraad) ? 1.0f : 0.0f;
      f[33] = (space->required_access_mask & dune_imperium::kAccessCity) ? 1.0f : 0.0f;

      // Space faction affiliation (0: Emperor, 1: SpacingGuild, 2: BG, 3: Fremen)
      if (space->gain_influence_faction >= 0 && space->gain_influence_faction < 4) {
        f[34 + space->gain_influence_faction] = 1.0f;
      }
      f[38] = space->is_combat ? 1.0f : 0.0f;
      f[39] = space->single_occupancy ? 1.0f : 0.0f;
      f[40] = space->swordmaster ? 1.0f : 0.0f;
      f[41] = space->mentat ? 1.0f : 0.0f;

      // Base vs Effective Costs
      const int solari_base = space->cost_solari;
      const int solari_eff = dune_imperium::AgentSpaceSolariCost(*space, leader_id);
      const bool guild_accord_played = state.GuildAccordPlayedForPendingPlacement(cur_player);
      const int spice_base = space->cost_spice;
      const int spice_eff = state.AgentSpaceSpiceCost(*space, guild_accord_played);
      f[42] = static_cast<float>(solari_base) / 10.0f;
      f[43] = static_cast<float>(solari_eff) / 10.0f;
      f[44] = static_cast<float>(spice_base) / 10.0f;
      f[45] = static_cast<float>(spice_eff) / 10.0f;
      f[46] = static_cast<float>(space->cost_water) / 5.0f;
      f[47] = static_cast<float>(space->cost_water) / 5.0f; // Water cost effective = base
      f[48] = (solari_eff < solari_base) ? 1.0f : 0.0f;      // has_solari_discount
      f[49] = (spice_eff < spice_base) ? 1.0f : 0.0f;        // has_spice_discount

      // Resource Gains
      f[50] = static_cast<float>(space->gain_solari) / 10.0f;
      f[51] = static_cast<float>(space->gain_spice) / 10.0f;
      f[52] = static_cast<float>(space->gain_water) / 5.0f;
      f[53] = static_cast<float>(space->gain_troops) / 5.0f;
      f[54] = static_cast<float>(space->gain_shipping) / 3.0f;
      f[55] = static_cast<float>(space->gain_research) / 3.0f;
      continue;
    }

    // 3. Imperium Market Purchases (slots 0..4)
    if (action >= dune_imperium::kActionBuyImperiumRow0 &&
        action <= dune_imperium::kActionBuyImperiumRow0 + 4) {
      const int slot = action - dune_imperium::kActionBuyImperiumRow0;
      out_data->space_ids[i] = 0;
      out_data->supported[i] = 1;
      out_data->roles[i] = ActionRole::kMarketPurchase;

      // Role one-hot
      f[3] = 1.0f; // is_market_purchase

      // Market slot one-hot (stays attached to slot action)
      if (slot >= 0 && slot < 5) {
        f[4 + slot] = 1.0f;
      }

      // Card-dependent fields move with the card currently in this slot
      const int card_id = state.GetImperiumRowCardForTesting(slot);
      if (card_id != dune_imperium::kInvalidCard && card_id >= 0 &&
          card_id < kCardVocabSize) {
        out_data->card_ids[i] = card_id;
        const auto* card = dune_imperium::FindImperiumCardById(card_id);
        if (card != nullptr) {
          f[9] = static_cast<float>(card->persuasion_cost) / 10.0f;
          f[10] = (card->access_mask & dune_imperium::kAccessEmperor) ? 1.0f : 0.0f;
          f[11] = (card->access_mask & dune_imperium::kAccessSpacingGuild) ? 1.0f : 0.0f;
          f[12] = (card->access_mask & dune_imperium::kAccessBeneGesserit) ? 1.0f : 0.0f;
          f[13] = (card->access_mask & dune_imperium::kAccessFremen) ? 1.0f : 0.0f;
          f[14] = (card->access_mask & dune_imperium::kAccessSpiceTrade) ? 1.0f : 0.0f;
          f[15] = (card->access_mask & dune_imperium::kAccessLandsraad) ? 1.0f : 0.0f;
          f[16] = (card->access_mask & dune_imperium::kAccessCity) ? 1.0f : 0.0f;
          f[17] = (card->faction_mask & (1u << 0)) ? 1.0f : 0.0f;
          f[18] = (card->faction_mask & (1u << 1)) ? 1.0f : 0.0f;
          f[19] = (card->faction_mask & (1u << 2)) ? 1.0f : 0.0f;
          f[20] = (card->faction_mask & (1u << 3)) ? 1.0f : 0.0f;
          f[21] = static_cast<float>(card->reveal_persuasion) / 10.0f;
          f[22] = static_cast<float>(card->reveal_swords) / 5.0f;
          f[23] = static_cast<float>(card->agent_card_draw) / 3.0f;
          f[24] = static_cast<float>(card->reveal_card_draw) / 3.0f;
          f[25] = static_cast<float>(card->agent_troops) / 5.0f;
          f[26] = static_cast<float>(card->reveal_troops) / 5.0f;
        }
        if (schema_version == kDescriptorSchemaVersionV5) {
          out_data->v5_item_ids[i] = card_id;
          out_data->v5_families[i] = 0;
        }
      } else {
        out_data->card_ids[i] = 0;
      }
      continue;
    }

    // 4. Indexed Intrigue Choices (actions 700..719)
    if (action >= dune_imperium::kActionIntrigueChoiceTrashIntrigue0 &&
        action < dune_imperium::kActionIntrigueChoiceTrashIntrigue0 + 20) {
      if (schema_version != kDescriptorSchemaVersionV2) {
        const int idx = action - dune_imperium::kActionIntrigueChoiceTrashIntrigue0;
        out_data->space_ids[i] = 0;
        if (cur_player >= 0 && cur_player < state.NumPlayers()) {
          const auto& hand = state.GetIntrigueHandForTesting(cur_player);
          if (idx >= 0 && idx < static_cast<int>(hand.size())) {
            const int c = hand[idx];
            out_data->card_ids[i] = (c >= 0 && c < 64) ? (kIntrigueCardVocabOffset + c) : 0;
            out_data->supported[i] = 1;
            out_data->roles[i] = ActionRole::kIndexedIntrigue;
            f[0] = 0.0f;
          } else {
            out_data->card_ids[i] = 0;
            out_data->supported[i] = 0;
            out_data->roles[i] = ActionRole::kUnsupported;
            f[0] = 1.0f;
          }
        } else {
          out_data->card_ids[i] = 0;
          out_data->supported[i] = 0;
          out_data->roles[i] = ActionRole::kUnsupported;
          f[0] = 1.0f;
        }
      } else {
        out_data->card_ids[i] = 0;
        out_data->space_ids[i] = 0;
        out_data->supported[i] = 0;
        out_data->roles[i] = ActionRole::kUnsupported;
        f[0] = 1.0f;
      }
      continue;
    }

    // 5. Graft Primary Identity on Solo Agent Play
    if (action == dune_imperium::kActionPlayAgentSolo) {
      if (schema_version != kDescriptorSchemaVersionV2) {
        const int primary = state.GetPendingGraftPrimaryId();
        out_data->space_ids[i] = 0;
        if (primary != dune_imperium::kInvalidCard && primary >= 0 && primary < kCardVocabSize) {
          out_data->card_ids[i] = primary;
          out_data->supported[i] = 1;
          out_data->roles[i] = ActionRole::kGraftSolo;
          f[0] = 0.0f;
        } else {
          out_data->card_ids[i] = 0;
          out_data->supported[i] = 0;
          out_data->roles[i] = ActionRole::kUnsupported;
          f[0] = 1.0f;
        }
      } else {
        out_data->card_ids[i] = 0;
        out_data->space_ids[i] = 0;
        out_data->supported[i] = 0;
        out_data->roles[i] = ActionRole::kUnsupported;
        f[0] = 1.0f;
      }
      continue;
    }

    // 6. Hundro Setup Mode A/B Offers
    if (state.GetPendingHundroSetupPlayer() == cur_player) {
      if (action == dune_imperium::kActionIntrigueChoiceModeA) {
        if (schema_version != kDescriptorSchemaVersionV2) {
          const int c = state.GetPendingHundroSetupCard(0);
          out_data->space_ids[i] = 0;
          if (c != dune_imperium::kInvalidCard && c >= 0 && c < 64) {
            out_data->card_ids[i] = kIntrigueCardVocabOffset + c;
            out_data->supported[i] = 1;
            out_data->roles[i] = ActionRole::kHundroOffer;
            f[0] = 0.0f;
          } else {
            out_data->card_ids[i] = 0;
            out_data->supported[i] = 0;
            out_data->roles[i] = ActionRole::kUnsupported;
            f[0] = 1.0f;
          }
        } else {
          out_data->card_ids[i] = 0;
          out_data->space_ids[i] = 0;
          out_data->supported[i] = 0;
          out_data->roles[i] = ActionRole::kUnsupported;
          f[0] = 1.0f;
        }
        continue;
      } else if (action == dune_imperium::kActionIntrigueChoiceModeB) {
        if (schema_version != kDescriptorSchemaVersionV2) {
          const int c = state.GetPendingHundroSetupCard(1);
          out_data->space_ids[i] = 0;
          if (c != dune_imperium::kInvalidCard && c >= 0 && c < 64) {
            out_data->card_ids[i] = kIntrigueCardVocabOffset + c;
            out_data->supported[i] = 1;
            out_data->roles[i] = ActionRole::kHundroOffer;
            f[0] = 0.0f;
          } else {
            out_data->card_ids[i] = 0;
            out_data->supported[i] = 0;
            out_data->roles[i] = ActionRole::kUnsupported;
            f[0] = 1.0f;
          }
        } else {
          out_data->card_ids[i] = 0;
          out_data->space_ids[i] = 0;
          out_data->supported[i] = 0;
          out_data->roles[i] = ActionRole::kUnsupported;
          f[0] = 1.0f;
        }
        continue;
      }
    }

    // 7. Tleilaxu Market Row Purchases (actions 91 and 92)
    if (action == dune_imperium::kActionTleilaxuAcquire0 + 1 ||
        action == dune_imperium::kActionTleilaxuAcquire0 + 2) {
      if (schema_version == kDescriptorSchemaVersionV5) {
        out_data->card_ids[i] = 0;
        out_data->space_ids[i] = 0;
        out_data->supported[i] = 0;
        out_data->roles[i] = ActionRole::kUnsupported;
        f[0] = 1.0f;
        const int slot = action - (dune_imperium::kActionTleilaxuAcquire0 + 1);
        const int local_id = state.GetTleilaxuRowCardForTesting(slot);
        if (local_id != dune_imperium::kInvalidCard && local_id >= 1 &&
            local_id < static_cast<int>(dune_imperium::kTleilaxuCards.size())) {
          const int shared_id = dune_imperium::kTleilaxuCards[local_id].imperium_card_id;
          if (shared_id >= 0 && shared_id < kCardVocabSize) {
            out_data->v5_item_ids[i] = shared_id;
            out_data->v5_families[i] = 2;
            out_data->card_ids[i] = shared_id;
          }
        }
        continue;
      } else if (schema_version != kDescriptorSchemaVersionV2 &&
                 schema_version != kDescriptorSchemaVersionV3) {
        const int slot = action - (dune_imperium::kActionTleilaxuAcquire0 + 1);
        const int local_id = state.GetTleilaxuRowCardForTesting(slot);
        out_data->space_ids[i] = 0;
        if (local_id != dune_imperium::kInvalidCard && local_id >= 1 &&
            local_id < static_cast<int>(dune_imperium::kTleilaxuCards.size())) {
          const int shared_id = dune_imperium::kTleilaxuCards[local_id].imperium_card_id;
          if (shared_id >= 0 && shared_id < kCardVocabSize) {
            out_data->card_ids[i] = shared_id;
            out_data->supported[i] = 1;
            out_data->roles[i] = ActionRole::kTleilaxuPurchase;
            f[0] = 0.0f;

            const auto* card = dune_imperium::FindImperiumCardById(shared_id);
            if (card != nullptr) {
              f[9] = static_cast<float>(card->persuasion_cost) / 10.0f;
              f[10] = (card->access_mask & dune_imperium::kAccessEmperor) ? 1.0f : 0.0f;
              f[11] = (card->access_mask & dune_imperium::kAccessSpacingGuild) ? 1.0f : 0.0f;
              f[12] = (card->access_mask & dune_imperium::kAccessBeneGesserit) ? 1.0f : 0.0f;
              f[13] = (card->access_mask & dune_imperium::kAccessFremen) ? 1.0f : 0.0f;
              f[14] = (card->access_mask & dune_imperium::kAccessSpiceTrade) ? 1.0f : 0.0f;
              f[15] = (card->access_mask & dune_imperium::kAccessLandsraad) ? 1.0f : 0.0f;
              f[16] = (card->access_mask & dune_imperium::kAccessCity) ? 1.0f : 0.0f;
              f[17] = (card->faction_mask & (1u << 0)) ? 1.0f : 0.0f;
              f[18] = (card->faction_mask & (1u << 1)) ? 1.0f : 0.0f;
              f[19] = (card->faction_mask & (1u << 2)) ? 1.0f : 0.0f;
              f[20] = (card->faction_mask & (1u << 3)) ? 1.0f : 0.0f;
              f[21] = static_cast<float>(card->reveal_persuasion) / 10.0f;
              f[22] = static_cast<float>(card->reveal_swords) / 5.0f;
              f[23] = static_cast<float>(card->agent_card_draw) / 3.0f;
              f[24] = static_cast<float>(card->reveal_card_draw) / 3.0f;
              f[25] = static_cast<float>(card->agent_troops) / 5.0f;
              f[26] = static_cast<float>(card->reveal_troops) / 5.0f;
            }
          } else {
            out_data->card_ids[i] = 0;
            out_data->supported[i] = 0;
            out_data->roles[i] = ActionRole::kUnsupported;
            f[0] = 1.0f;
          }
        } else {
          out_data->card_ids[i] = 0;
          out_data->supported[i] = 0;
          out_data->roles[i] = ActionRole::kUnsupported;
          f[0] = 1.0f;
        }
      } else {
        out_data->card_ids[i] = 0;
        out_data->space_ids[i] = 0;
        out_data->supported[i] = 0;
        out_data->roles[i] = ActionRole::kUnsupported;
        f[0] = 1.0f;
      }
      continue;
    }

    // 8. Helena Reserve Purchase (action 572)
    if (schema_version == kDescriptorSchemaVersionV5 &&
        action == dune_imperium::kActionBuyHelenaReserve0) {
      out_data->card_ids[i] = 0;
      out_data->space_ids[i] = 0;
      out_data->supported[i] = 0;
      out_data->roles[i] = ActionRole::kUnsupported;
      f[0] = 1.0f;
      const int cur_p = state.CurrentPlayer();
      const auto& h_res = (cur_p >= 0 && cur_p < 4) ? state.GetHelenaReservedCards(cur_p) : std::vector<int>{};
      if (!h_res.empty() && h_res[0] >= 0 && h_res[0] < kCardVocabSize &&
          dune_imperium::FindImperiumCardById(h_res[0]) != nullptr) {
        const int cid = h_res[0];
        out_data->v5_item_ids[i] = cid;
        out_data->v5_families[i] = 1;
        out_data->card_ids[i] = cid;
        const auto* card = dune_imperium::FindImperiumCardById(cid);
        float* v5_ctx = &out_data->v5_ctx_features[i * kV5CtxDim];
        v5_ctx[0] = (card && card->persuasion_cost > 0) ? 0.2f : 0.0f;
      }
      continue;
    }

    // 9. Imperium Row Choice (actions 538..543 and 546..551)
    if (schema_version == kDescriptorSchemaVersionV5 &&
        ((action >= dune_imperium::kActionIntrigueChoiceImperiumRow0 &&
          action < dune_imperium::kActionIntrigueChoiceImperiumRow0 + 6) ||
         (action >= dune_imperium::kActionIntrigueChoiceImperiumRowTop0 &&
          action < dune_imperium::kActionIntrigueChoiceImperiumRowTop0 + 6))) {
      out_data->card_ids[i] = 0;
      out_data->space_ids[i] = 0;
      out_data->supported[i] = 0;
      out_data->roles[i] = ActionRole::kUnsupported;
      f[0] = 1.0f;
      const bool is_top = (action >= dune_imperium::kActionIntrigueChoiceImperiumRowTop0 &&
                           action < dune_imperium::kActionIntrigueChoiceImperiumRowTop0 + 6);
      const int slot = is_top ? (action - dune_imperium::kActionIntrigueChoiceImperiumRowTop0)
                              : (action - dune_imperium::kActionIntrigueChoiceImperiumRow0);
      const int cid = (slot == 5) ? dune_imperium::kCardArrakisLiaison : state.GetImperiumRowCardForTesting(slot);
      if (cid >= 0 && cid < kCardVocabSize && dune_imperium::FindImperiumCardById(cid) != nullptr) {
        out_data->v5_item_ids[i] = cid;
        out_data->v5_families[i] = 3;
        out_data->card_ids[i] = cid;
        const int cur_p = state.CurrentPlayer();
        const bool is_helena_pick = (!is_top && slot < 5 && cur_p >= 0 && cur_p < 4 && state.GetPendingHelenaReserve(cur_p) > 0);
        float* v5_ctx = &out_data->v5_ctx_features[i * kV5CtxDim];
        v5_ctx[4] = is_top ? 1.0f : 0.0f;
        v5_ctx[5] = is_helena_pick ? 1.0f : 0.0f;
        v5_ctx[6] = (!is_helena_pick) ? 1.0f : 0.0f;
      }
      continue;
    }

    // 10. Tech Choice or Buy (actions 70..72, 770..772, 554..556)
    if (schema_version == kDescriptorSchemaVersionV5 &&
        ((action >= dune_imperium::kActionTechAcquire0 &&
          action < dune_imperium::kActionTechAcquire0 + 3) ||
         (action >= dune_imperium::kActionTechAcquireWithSolari0 &&
          action < dune_imperium::kActionTechAcquireWithSolari0 + 3) ||
         (action >= dune_imperium::kActionIntrigueChoiceTech0 &&
          action < dune_imperium::kActionIntrigueChoiceTech0 + 3))) {
      out_data->card_ids[i] = 0;
      out_data->space_ids[i] = 0;
      out_data->supported[i] = 0;
      out_data->roles[i] = ActionRole::kUnsupported;
      f[0] = 1.0f;
      const bool is_solari = (action >= dune_imperium::kActionTechAcquireWithSolari0 &&
                              action < dune_imperium::kActionTechAcquireWithSolari0 + 3);
      const bool is_intrigue_tech = (action >= dune_imperium::kActionIntrigueChoiceTech0 &&
                                     action < dune_imperium::kActionIntrigueChoiceTech0 + 3);
      const int raw_slot = is_solari ? (action - dune_imperium::kActionTechAcquireWithSolari0)
                                     : (is_intrigue_tech ? (action - dune_imperium::kActionIntrigueChoiceTech0)
                                                         : (action - dune_imperium::kActionTechAcquire0));
      const int app_slot = (!is_intrigue_tech) ? state.GetPendingAppropriateSelectedTechSlot() : -1;
      const int slot = (app_slot >= 0 && app_slot < 3) ? app_slot : raw_slot;
      const int tech_id = state.GetTechMarketTile(slot);
      if (tech_id >= 0 && tech_id < static_cast<int>(dune_imperium::kTechTiles.size())) {
        out_data->v5_item_ids[i] = 256 + tech_id;
        out_data->v5_families[i] = 4;
        out_data->card_ids[i] = 0;
        const int cur_p = state.CurrentPlayer();
        const int base_cost = dune_imperium::kTechTiles[tech_id].spice_cost;
        int eff_cost = base_cost;
        if (cur_p >= 0 && cur_p < 4) {
          if (is_intrigue_tech) {
            eff_cost = state.TechAcquireCostForPlayerHelper(cur_p, tech_id, 0);
          } else if (is_solari) {
            eff_cost = std::max(0, base_cost - state.GetTechNegotiatorsForTesting(cur_p));
          } else {
            const int disc = (state.GetPendingSpaceTechAcquirePlayer() == cur_p) ? state.GetPendingSpaceTechAcquireDiscount() : 0;
            eff_cost = state.TechAcquireCostForPlayerHelper(cur_p, tech_id, disc);
          }
        }
        float* v5_ctx = &out_data->v5_ctx_features[i * kV5CtxDim];
        v5_ctx[0] = std::max(0, base_cost - eff_cost) / 5.0f;
        v5_ctx[1] = (eff_cost == 0) ? 1.0f : 0.0f;
        v5_ctx[2] = is_solari ? 1.0f : 0.0f;
        v5_ctx[3] = std::min(1.0f, static_cast<float>(state.GetTechStackSizeForTesting(slot)) / 6.0f);
        v5_ctx[7] = (app_slot >= 0 && app_slot < 3) ? 1.0f : 0.0f;
      }
      continue;
    }

    // 11. Unsupported Roles
    f[0] = 1.0f; // is_unsupported
    out_data->card_ids[i] = 0;
    out_data->space_ids[i] = 0;
    out_data->supported[i] = 0;
    out_data->roles[i] = ActionRole::kUnsupported;
  }
}

} // namespace dune_semantic
} // namespace open_spiel

#endif // OPEN_SPIEL_EXAMPLES_DUNE_SEMANTIC_ACTION_DESCRIPTORS_H_
