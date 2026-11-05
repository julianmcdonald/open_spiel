#ifndef OPEN_SPIEL_EXAMPLES_DUNE_SEMANTIC_ACTION_SCORER_H_
#define OPEN_SPIEL_EXAMPLES_DUNE_SEMANTIC_ACTION_SCORER_H_

#include <vector>
#include <string>
#include <cstdint>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <optional>
#include <type_traits>

#include "open_spiel/spiel.h"
#include "open_spiel/games/dune_imperium/dune_imperium.h"
#include "open_spiel/games/dune_imperium/dune_imperium_cards.h"
#include "open_spiel/games/dune_imperium/dune_imperium_board.h"
#include "open_spiel/games/dune_imperium/dune_imperium_common.h"
#include "open_spiel/games/dune_imperium/dune_imperium_content.h"

#ifdef OPEN_SPIEL_BUILD_WITH_LIBTORCH
#include <torch/torch.h>
#include <torch/nn/functional/linear.h>
#include <ATen/autocast_mode.h>
#include <ATen/CPUGeneratorImpl.h>
#include <ATen/ops/linalg_qr.h>
#include <c10/core/Event.h>
#include <c10/cuda/CUDAStream.h>
#endif

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

#ifdef OPEN_SPIEL_BUILD_WITH_LIBTORCH
#include <ATen/ops/_scaled_mm.h>

// Helper struct for persistent BF16 inference weights.
// Holds un-registered BF16 weight and bias tensors updated in place.
struct Bf16LinearCache {
  torch::Tensor weight;
  torch::Tensor bias;

  void Refresh(const torch::nn::Linear& fc) {
    if (!fc || !fc->weight.defined() || !fc->weight.is_cuda()) return;
    torch::NoGradGuard no_grad;
    if (!weight.defined() || weight.sizes() != fc->weight.sizes() ||
        weight.device() != fc->weight.device() ||
        weight.scalar_type() != torch::kBFloat16) {
      weight = torch::empty(fc->weight.sizes(),
                            fc->weight.options().dtype(torch::kBFloat16));
    }
    weight.copy_(fc->weight);
    if (fc->bias.defined()) {
      if (!bias.defined() || bias.sizes() != fc->bias.sizes() ||
          bias.device() != fc->bias.device() ||
          bias.scalar_type() != torch::kBFloat16) {
        bias = torch::empty(fc->bias.sizes(),
                            fc->bias.options().dtype(torch::kBFloat16));
      }
      bias.copy_(fc->bias);
    } else {
      bias = torch::Tensor();
    }
  }

  torch::Tensor Forward(const torch::nn::Linear& fc, const torch::Tensor& x,
                        bool use_bf16_cache) const {
    if (use_bf16_cache && !torch::GradMode::is_enabled() && weight.defined() &&
        at::autocast::is_autocast_enabled(at::kCUDA) && x.is_cuda()) {
      return torch::nn::functional::linear(x, weight, bias);
    }
    return const_cast<torch::nn::Linear&>(fc)->forward(x);
  }
};

// Helper struct for persistent FP8 (E4M3) inference weights and dynamic activations.
// Uses at::_scaled_mm for fast Ada Lovelace (RTX 4080 SUPER SM 8.9) Tensor Cores.
struct Fp8LinearCache {
  torch::Tensor w_fp8_nk_;    // [out_features, in_features] contiguous in kFloat8_e4m3fn
  torch::Tensor w_fp8_t_;     // [in_features, out_features] column-major (stride(0) == 1)
  torch::Tensor w_scale_;     // [1] or scalar kFloat32 tensor on CUDA
  torch::Tensor bias_bf16_;   // [out_features] kBFloat16 tensor on CUDA (or undefined)
  Bf16LinearCache bf16_cache_;
  bool ready_ = false;

  void Refresh(const torch::nn::Linear& fc) {
    if (!fc || !fc->weight.defined() || !fc->weight.is_cuda()) return;
    bf16_cache_.Refresh(fc);

    torch::NoGradGuard no_grad;
    auto w_f32 = fc->weight.to(torch::kFloat32);
    auto w_amax = w_f32.abs().amax();
    auto w_scale = (w_amax / 448.0f).clamp_min(1e-12f);

    if (!w_scale_.defined() || w_scale_.device() != fc->weight.device()) {
      w_scale_ = torch::empty({}, fc->weight.options().dtype(torch::kFloat32));
    }
    w_scale_.copy_(w_scale);

    auto w_quant = (w_f32 / w_scale_).clamp(-448.0f, 448.0f).to(torch::kFloat8_e4m3fn);
    if (!w_fp8_nk_.defined() || w_fp8_nk_.sizes() != fc->weight.sizes() ||
        w_fp8_nk_.device() != fc->weight.device() ||
        w_fp8_nk_.scalar_type() != torch::kFloat8_e4m3fn) {
      w_fp8_nk_ = torch::empty(fc->weight.sizes(),
                               fc->weight.options().dtype(torch::kFloat8_e4m3fn));
      w_fp8_t_ = w_fp8_nk_.t();
    }
    w_fp8_nk_.copy_(w_quant);

    if (fc->bias.defined()) {
      if (!bias_bf16_.defined() || bias_bf16_.sizes() != fc->bias.sizes() ||
          bias_bf16_.device() != fc->bias.device() ||
          bias_bf16_.scalar_type() != torch::kBFloat16) {
        bias_bf16_ = torch::empty(fc->bias.sizes(),
                                  fc->bias.options().dtype(torch::kBFloat16));
      }
      bias_bf16_.copy_(fc->bias);
    } else {
      bias_bf16_ = torch::Tensor();
    }
    ready_ = true;
  }

  torch::Tensor Forward(const torch::nn::Linear& fc, const torch::Tensor& x,
                        bool use_bf16_cache, bool use_fp8,
                        bool known_non_negative = false) const {
    if (!use_fp8 || torch::GradMode::is_enabled() || !x.is_cuda() ||
        !at::autocast::is_autocast_enabled(at::kCUDA) || !ready_ ||
        !w_fp8_t_.defined() || !w_scale_.defined()) {
      return bf16_cache_.Forward(fc, x, use_bf16_cache);
    }

    const auto orig_sizes = x.sizes();
    torch::Tensor x_2d = (x.dim() > 2) ? x.reshape({-1, x.size(-1)}) : x;
    const int64_t M = x_2d.size(0);
    const int64_t K = x_2d.size(1);
    const int64_t N = w_fp8_t_.size(1);

    // Keep small batches (M < 32, including 1-row root priors and tiny tail
    // flushes) and non-16-aligned dimensions in exact BF16.
    if (M < 32 || K < 16 || (K % 16 != 0) || N < 16 || (N % 16 != 0)) {
      return bf16_cache_.Forward(fc, x, use_bf16_cache);
    }

    const bool needs_pad = (M % 16 != 0);
    const int64_t M_pad = needs_pad ? ((M + 15) / 16) * 16 : M;

    torch::Tensor x_input = x_2d;
    if (needs_pad) {
      x_input = torch::zeros({M_pad, K}, x_2d.options());
      x_input.narrow(0, 0, M).copy_(x_2d);
    }

    // Dynamic per-tensor activation scale purely on GPU without host sync or
    // [M, K] FP32 expansion/clamp: 440.0f leaves safe headroom below E4M3 464 NaN
    // bin even under worst-case BF16 ULP rounding.
    constexpr float kFp8ActMax = 440.0f;
    auto x_amax = (known_non_negative ? x_input.amax() : x_input.abs().amax())
                      .to(torch::kFloat32)
                      .clamp_min(1e-12f);
    auto x_scale = x_amax * (1.0f / kFp8ActMax);
    auto inv_scale = (kFp8ActMax / x_amax).to(x_input.scalar_type());
    auto x_fp8 = (x_input * inv_scale).to(torch::kFloat8_e4m3fn).contiguous();

    std::optional<at::Tensor> bias_opt = std::nullopt;
    if (bias_bf16_.defined()) {
      bias_opt = bias_bf16_;
    }

    auto out = at::_scaled_mm(x_fp8, w_fp8_t_, x_scale, w_scale_, bias_opt,
                              /*scale_result=*/std::nullopt,
                              /*out_dtype=*/torch::kBFloat16,
                              /*use_fast_accum=*/true);

    if (needs_pad) {
      out = out.narrow(0, 0, M);
    }
    if (x.dim() > 2) {
      out = out.reshape(orig_sizes);
    }
    return out;
  }
};

// Neural Semantic Action Scorer Module.
// Computes a scalar correction from the actor representation and candidate action descriptors.
struct SemanticActionScorerImpl : torch::nn::Module {
  torch::nn::Embedding card_embedding{nullptr};
  torch::nn::Embedding space_embedding{nullptr};
  torch::nn::Linear desc_proj{nullptr};
  torch::nn::LayerNorm desc_ln{nullptr};
  torch::nn::Linear trunk_proj{nullptr};
  torch::nn::LayerNorm trunk_ln{nullptr};
  torch::nn::Linear mlp1{nullptr};
  torch::nn::LayerNorm mlp1_ln{nullptr};
  torch::nn::Linear out_layer{nullptr};
  torch::nn::Linear out_layer_ext{nullptr};
  torch::nn::Linear out_layer_f{nullptr};
  bool enable_f_ = false;
  bool slot_indexed_f_ = false;

  bool enable_v5_ = false;
  torch::nn::Embedding v5_item_emb{nullptr};
  torch::nn::Linear v5_card_adapter{nullptr};
  torch::nn::Linear v5_ctx_proj{nullptr};
  torch::nn::Linear v5_query_proj{nullptr};

  Bf16LinearCache desc_proj_bf16_;
  Bf16LinearCache trunk_proj_bf16_;
  Bf16LinearCache mlp1_bf16_;
  Bf16LinearCache out_layer_bf16_;
  Bf16LinearCache out_layer_ext_bf16_;
  Bf16LinearCache out_layer_f_bf16_;
  Bf16LinearCache v5_card_adapter_bf16_;
  Bf16LinearCache v5_ctx_proj_bf16_;
  Bf16LinearCache v5_query_proj_bf16_;
  torch::Tensor v5_item_emb_bf16_;
  bool use_bf16_cache_ = false;

  SemanticActionScorerImpl(bool enable_f = false, bool slot_indexed_f = false, bool enable_v5 = false)
      : enable_f_(enable_f), slot_indexed_f_(slot_indexed_f) {
    card_embedding = register_module(
        "card_embedding", torch::nn::Embedding(kCardVocabSize, kCardEmbedDim));
    space_embedding = register_module(
        "space_embedding", torch::nn::Embedding(kSpaceVocabSize, kSpaceEmbedDim));
    desc_proj = register_module(
        "desc_proj", torch::nn::Linear(kTotalDescDim, kScorerHiddenDim));
    desc_ln = register_module(
        "desc_ln", torch::nn::LayerNorm(torch::nn::LayerNormOptions({kScorerHiddenDim})));
    trunk_proj = register_module(
        "trunk_proj", torch::nn::Linear(2048, kScorerHiddenDim));
    trunk_ln = register_module(
        "trunk_ln", torch::nn::LayerNorm(torch::nn::LayerNormOptions({kScorerHiddenDim})));
    mlp1 = register_module(
        "mlp1", torch::nn::Linear(kScorerHiddenDim * 2, kScorerHiddenDim));
    mlp1_ln = register_module(
        "mlp1_ln", torch::nn::LayerNorm(torch::nn::LayerNormOptions({kScorerHiddenDim})));
    out_layer = register_module(
        "out_layer", torch::nn::Linear(kScorerHiddenDim, 1));
    out_layer_ext = register_module(
        "out_layer_ext", torch::nn::Linear(kScorerHiddenDim, 1));

    // Initialize the final output layers to zero so initial corrections are exactly 0.0.
    {
      torch::NoGradGuard no_grad;
      out_layer->weight.zero_();
      if (out_layer->bias.defined()) {
        out_layer->bias.zero_();
      }
      out_layer_ext->weight.zero_();
      if (out_layer_ext->bias.defined()) {
        out_layer_ext->bias.zero_();
      }
    }

    if (enable_f_) {
      EnableF(slot_indexed_f_);
    }
    if (enable_v5) {
      EnableV5();
    }
  }

  void EnableF(bool slot_indexed = false) {
    enable_f_ = true;
    slot_indexed_f_ = slot_indexed;
    const int64_t out_dim = slot_indexed_f_ ? 2 : 1;
    if (!out_layer_f || out_layer_f->options.out_features() != out_dim) {
      out_layer_f = register_module(
          "out_layer_f", torch::nn::Linear(kScorerHiddenDim, out_dim));
      torch::NoGradGuard no_grad;
      out_layer_f->weight.zero_();
      if (out_layer_f->bias.defined()) {
        out_layer_f->bias.zero_();
      }
      if (use_bf16_cache_) {
        out_layer_f_bf16_.Refresh(out_layer_f);
      }
    }
  }

  void EnableV5() {
    if (enable_v5_ && v5_query_proj) return;
    enable_v5_ = true;

    v5_item_emb = register_module(
        "v5_item_emb",
        torch::nn::Embedding(torch::nn::EmbeddingOptions(kV5ItemVocabSize, kV5EmbedDim)));
    v5_card_adapter = register_module(
        "v5_card_adapter",
        torch::nn::Linear(torch::nn::LinearOptions(kCardEmbedDim, kV5EmbedDim).bias(false)));
    v5_ctx_proj = register_module(
        "v5_ctx_proj",
        torch::nn::Linear(torch::nn::LinearOptions(kV5CtxDim, kV5EmbedDim).bias(false)));
    v5_query_proj = register_module(
        "v5_query_proj",
        torch::nn::Linear(kScorerHiddenDim, kV5NumFamilies * kV5EmbedDim));

    torch::NoGradGuard no_grad;
    auto dev = (card_embedding && card_embedding->weight.defined())
                   ? card_embedding->weight.device()
                   : torch::kCPU;

    torch::Generator gen = at::detail::createCPUGenerator(20261104ULL);
    auto mat = torch::randn({kV5EmbedDim, kV5EmbedDim}, gen, torch::kFloat32);
    auto qr = at::linalg_qr(mat);
    auto q = std::get<0>(qr);
    v5_item_emb->to(dev);
    v5_item_emb->weight.copy_(q.narrow(0, 0, kV5ItemVocabSize).to(dev));
    v5_item_emb->weight[279].zero_();

    v5_card_adapter->to(dev);
    v5_card_adapter->weight.zero_();

    v5_ctx_proj->to(dev);
    v5_ctx_proj->weight.zero_();

    v5_query_proj->to(dev);
    v5_query_proj->weight.zero_();
    if (v5_query_proj->bias.defined()) {
      v5_query_proj->bias.zero_();
    }

    if (use_bf16_cache_) {
      RefreshBf16Cache();
    }
  }

  void RefreshBf16Cache() {
    desc_proj_bf16_.Refresh(desc_proj);
    trunk_proj_bf16_.Refresh(trunk_proj);
    mlp1_bf16_.Refresh(mlp1);
    out_layer_bf16_.Refresh(out_layer);
    if (out_layer_ext) {
      out_layer_ext_bf16_.Refresh(out_layer_ext);
    }
    if (out_layer_f) {
      out_layer_f_bf16_.Refresh(out_layer_f);
    }
    if (enable_v5_ && v5_query_proj) {
      v5_card_adapter_bf16_.Refresh(v5_card_adapter);
      v5_ctx_proj_bf16_.Refresh(v5_ctx_proj);
      v5_query_proj_bf16_.Refresh(v5_query_proj);
      if (v5_item_emb && v5_item_emb->weight.defined() && v5_item_emb->weight.is_cuda()) {
        v5_item_emb_bf16_ = v5_item_emb->weight.detach().to(torch::kBFloat16);
      }
    }
    use_bf16_cache_ = true;
  }

  void DisableBf16Cache() {
    use_bf16_cache_ = false;
  }

  bool bf16_cache_ready() const {
    return use_bf16_cache_;
  }

  bool is_f_enabled() const {
    return enable_f_;
  }

  bool is_v5_enabled() const {
    return enable_v5_;
  }

  void Load(torch::serialize::InputArchive& archive,
            const std::string& schema_version = kDescriptorSchemaVersionV3) {
    if (schema_version == kDescriptorSchemaVersionV4) {
      EnableF();
    }
    auto read_child = [&](const std::string& name, auto& module) {
      torch::serialize::InputArchive child;
      if (!archive.try_read(name, child)) {
        SpielFatalError("SemanticActionScorer: missing required submodule '" + name + "'");
      }
      module->load(child);
    };

    read_child("card_embedding", card_embedding);
    read_child("space_embedding", space_embedding);
    read_child("desc_proj", desc_proj);
    read_child("desc_ln", desc_ln);
    read_child("trunk_proj", trunk_proj);
    read_child("trunk_ln", trunk_ln);
    read_child("mlp1", mlp1);
    read_child("mlp1_ln", mlp1_ln);
    read_child("out_layer", out_layer);

    torch::serialize::InputArchive ext_arch;
    if (archive.try_read("out_layer_ext", ext_arch)) {
      out_layer_ext->load(ext_arch);
    } else {
      if (schema_version == kDescriptorSchemaVersionV2) {
        torch::NoGradGuard no_grad;
        out_layer_ext->weight.zero_();
        if (out_layer_ext->bias.defined()) {
          out_layer_ext->bias.zero_();
        }
      } else {
        SpielFatalError("SemanticActionScorer: missing required submodule 'out_layer_ext' in v3 checkpoint");
      }
    }

    if (enable_f_ && out_layer_f) {
      torch::serialize::InputArchive f_arch;
      if (archive.try_read("out_layer_f", f_arch)) {
        torch::Tensor w;
        if (f_arch.try_read("weight", w)) {
          const int64_t ckpt_out_dim = w.size(0);
          if (out_layer_f->options.out_features() != ckpt_out_dim) {
            slot_indexed_f_ = (ckpt_out_dim == 2);
            out_layer_f = register_module(
                "out_layer_f", torch::nn::Linear(kScorerHiddenDim, ckpt_out_dim));
          }
        }
        out_layer_f->load(f_arch);
        if (out_layer_f->options.out_features() == 2) {
          slot_indexed_f_ = true;
        }
      } else {
        torch::NoGradGuard no_grad;
        out_layer_f->weight.zero_();
        if (out_layer_f->bias.defined()) {
          out_layer_f->bias.zero_();
        }
      }
      if (use_bf16_cache_) {
        out_layer_f_bf16_.Refresh(out_layer_f);
      }
    }

    torch::serialize::InputArchive probe_v5;
    bool ckpt_has_v5 = archive.try_read("v5_query_proj", probe_v5);
    if (schema_version == kDescriptorSchemaVersionV5 || enable_v5_ || ckpt_has_v5) {
      EnableV5();
      if (ckpt_has_v5) {
        read_child("v5_item_emb", v5_item_emb);
        read_child("v5_card_adapter", v5_card_adapter);
        read_child("v5_ctx_proj", v5_ctx_proj);
        read_child("v5_query_proj", v5_query_proj);
      } else {
        torch::NoGradGuard no_grad;
        auto dev = (card_embedding && card_embedding->weight.defined())
                       ? card_embedding->weight.device()
                       : torch::kCPU;
        torch::Generator gen = at::detail::createCPUGenerator(20261104ULL);
        auto mat = torch::randn({kV5EmbedDim, kV5EmbedDim}, gen, torch::kFloat32);
        auto qr = at::linalg_qr(mat);
        auto q = std::get<0>(qr);
        v5_item_emb->to(dev);
        v5_item_emb->weight.copy_(q.narrow(0, 0, kV5ItemVocabSize).to(dev));
        v5_item_emb->weight[279].zero_();

        v5_card_adapter->to(dev);
        v5_card_adapter->weight.zero_();

        v5_ctx_proj->to(dev);
        v5_ctx_proj->weight.zero_();

        v5_query_proj->to(dev);
        v5_query_proj->weight.zero_();
        if (v5_query_proj->bias.defined()) {
          v5_query_proj->bias.zero_();
        }
      }
      if (use_bf16_cache_) {
        RefreshBf16Cache();
      }
    }
  }

  void load(torch::serialize::InputArchive& archive) {
    if (enable_v5_) {
      Load(archive, kDescriptorSchemaVersionV5);
    } else if (enable_f_) {
      Load(archive, kDescriptorSchemaVersionV4);
    } else {
      Load(archive, kDescriptorSchemaVersionV3);
    }
  }

  torch::Tensor ProjectTrunk(const torch::Tensor& trunk) {
    return torch::relu(trunk_ln->forward(
        trunk_proj_bf16_.Forward(trunk_proj, trunk, use_bf16_cache_)));
  }

  // Forward computation: adds corrections directly to inout_logits at the specified positions.
  void ComputeAndAddCorrections(
      const torch::Tensor& trunk,            // [B, 2048]
      const torch::Tensor& batch_indices,    // [M]
      const torch::Tensor& action_indices,   // [M]
      const torch::Tensor& feat_tensor,      // [M, 56]
      const torch::Tensor& card_ids,         // [M]
      const torch::Tensor& space_ids,        // [M]
      const torch::Tensor& supported_mask,   // [M]
      torch::Tensor& inout_logits,           // [B, 2391]
      const torch::Tensor& is_ext_mask = torch::Tensor(),
      const torch::Tensor& is_f_mask = torch::Tensor(),
      bool allow_inplace = false,
      std::optional<bool> any_ext_hint = std::nullopt,
      std::optional<bool> any_f_hint = std::nullopt,
      const torch::Tensor& precomputed_trunk_proj = torch::Tensor(),
      const torch::Tensor& v5_item_ids = torch::Tensor(),
      const torch::Tensor& v5_families = torch::Tensor(),
      const torch::Tensor& v5_ctx_feat = torch::Tensor(),
      std::optional<bool> any_v5_hint = std::nullopt) {
    if (batch_indices.numel() == 0 || feat_tensor.numel() == 0) {
      return;
    }

    const bool has_f_mask = is_f_mask.defined() && is_f_mask.numel() > 0;
    const bool any_f = any_f_hint.has_value()
        ? (*any_f_hint && has_f_mask)
        : (has_f_mask && (!enable_f_ ? is_f_mask.any().item<bool>() : true));
    if (any_f && (!enable_f_ || !out_layer_f)) {
      SpielFatalError(
          "SemanticActionScorer::ComputeAndAddCorrections: kTleilaxuPurchase "
          "(is_f_mask=true) encountered while enable_f_ is false; "
          "semantic_action_v4 scorer required.");
    }
    const bool has_ext_mask = is_ext_mask.defined() && is_ext_mask.numel() > 0;
    const bool any_ext = any_ext_hint.has_value()
        ? (*any_ext_hint && has_ext_mask)
        : has_ext_mask;

    const bool has_v5_tensor = v5_families.defined() && v5_families.numel() > 0;
    const bool any_v5 = any_v5_hint.has_value()
        ? (*any_v5_hint && has_v5_tensor)
        : (has_v5_tensor && (v5_families >= 0).any().item<bool>());
    if (any_v5 && (!enable_v5_ || !v5_query_proj)) {
      SpielFatalError(
          "SemanticActionScorer::ComputeAndAddCorrections: v5 action encountered "
          "while enable_v5_ is false; semantic_action_v5 scorer required.");
    }

    auto c_emb = card_embedding->forward(card_ids);     // [M, 32]
    auto s_emb = space_embedding->forward(space_ids);   // [M, 16]
    auto d_full = torch::cat({feat_tensor, c_emb, s_emb}, -1); // [M, 104]

    auto d_p = torch::relu(desc_ln->forward(desc_proj_bf16_.Forward(desc_proj, d_full, use_bf16_cache_)));  // [M, 256]
    auto h_p = precomputed_trunk_proj.defined() ? precomputed_trunk_proj : ProjectTrunk(trunk); // [B, 256]
    auto h_gathered = h_p.index_select(0, batch_indices);                  // [M, 256]

    auto z = torch::cat({h_gathered, d_p}, -1);                            // [M, 512]
    auto u = torch::relu(mlp1_ln->forward(mlp1_bf16_.Forward(mlp1, z, use_bf16_cache_)));              // [M, 256]
    
    auto legacy_corr = out_layer_bf16_.Forward(out_layer, u, use_bf16_cache_).squeeze(-1);                  // [M]
    torch::Tensor raw_corr = legacy_corr;
    if (any_ext) {
      auto ext_corr = out_layer_ext_bf16_.Forward(out_layer_ext, u, use_bf16_cache_).squeeze(-1);                 // [M]
      raw_corr = torch::where(is_ext_mask, ext_corr, raw_corr);
    }
    if (any_f) {
      torch::Tensor f_corr;
      if (slot_indexed_f_ && out_layer_f->options.out_features() == 2) {
        auto f_out = out_layer_f_bf16_.Forward(out_layer_f, u, use_bf16_cache_); // [M, 2]
        auto col_indices = (action_indices == (dune_imperium::kActionTleilaxuAcquire0 + 2)).to(torch::kInt64).unsqueeze(1); // [M, 1]
        f_corr = f_out.gather(1, col_indices).squeeze(1); // [M]
      } else {
        f_corr = out_layer_f_bf16_.Forward(out_layer_f, u, use_bf16_cache_).squeeze(-1); // [M]
      }
      raw_corr = torch::where(is_f_mask, f_corr, raw_corr);
    }

    torch::Tensor is_v5_mask;
    if (any_v5) {
      is_v5_mask = (v5_families >= 0);
      if (torch::GradMode::is_enabled()) {
        raw_corr = torch::where(is_v5_mask, raw_corr.detach(), raw_corr);
      }
    }

    auto corr = (raw_corr * supported_mask).to(inout_logits.dtype());      // [M]

    if (any_v5) {
      auto v5_idx = torch::nonzero(is_v5_mask).squeeze(-1);
      if (v5_idx.numel() > 0) {
        auto b_v5 = batch_indices.index_select(0, v5_idx);
        auto item_v5 = v5_item_ids.index_select(0, v5_idx);
        auto fam_v5 = v5_families.index_select(0, v5_idx);
        auto ctx_v5 = v5_ctx_feat.index_select(0, v5_idx);
        auto h_v5 = h_p.index_select(0, b_v5);

        auto q_all = v5_query_proj_bf16_.Forward(v5_query_proj, h_v5, use_bf16_cache_).view({-1, kV5NumFamilies, kV5EmbedDim});
        auto fam_expanded = fam_v5.unsqueeze(1).unsqueeze(2).expand({-1, 1, kV5EmbedDim});
        auto q_sel = q_all.gather(1, fam_expanded).squeeze(1);

        torch::Tensor e_id;
        if (use_bf16_cache_ && !torch::GradMode::is_enabled() &&
            v5_item_emb_bf16_.defined() && item_v5.is_cuda() &&
            at::autocast::is_autocast_enabled(at::kCUDA)) {
          e_id = at::embedding(v5_item_emb_bf16_, item_v5);
        } else {
          e_id = v5_item_emb->forward(item_v5);
        }

        auto c_v5 = c_emb.index_select(0, v5_idx);
        if (torch::GradMode::is_enabled()) {
          c_v5 = c_v5.detach();
        }
        auto c_v5_masked = c_v5 * (item_v5 < 256).unsqueeze(-1).to(c_v5.dtype());

        auto k_item = e_id +
                      v5_card_adapter_bf16_.Forward(v5_card_adapter, c_v5_masked, use_bf16_cache_) +
                      v5_ctx_proj_bf16_.Forward(v5_ctx_proj, ctx_v5, use_bf16_cache_);

        auto v5_delta = (q_sel.to(torch::kFloat32) * k_item.to(torch::kFloat32)).sum(-1);

        auto base_corr = (raw_corr * supported_mask).to(torch::kFloat32);
        base_corr = base_corr.index_put({v5_idx}, base_corr.index_select(0, v5_idx) + v5_delta);
        corr = base_corr.to(inout_logits.dtype());
      }
    }

    if (allow_inplace && !torch::GradMode::is_enabled()) {
      inout_logits.index_put_(
          {batch_indices, action_indices},
          inout_logits.index({batch_indices, action_indices}) + corr,
          /*accumulate=*/false);
    } else {
      inout_logits = inout_logits.index_put(
          {batch_indices, action_indices},
          inout_logits.index({batch_indices, action_indices}) + corr);
    }
  }

  torch::Tensor ComputeV5CorrectionOnly(
      const torch::Tensor& trunk,
      const torch::Tensor& item_ids,
      const torch::Tensor& families,
      const torch::Tensor& ctx_features,
      const torch::Tensor& card_ids = torch::Tensor()) {
    if (!enable_v5_ || !v5_query_proj) {
      return torch::zeros({item_ids.size(0)}, trunk.options());
    }
    auto h_p = ProjectTrunk(trunk);
    auto q_all = v5_query_proj_bf16_.Forward(v5_query_proj, h_p, use_bf16_cache_).view({-1, kV5NumFamilies, kV5EmbedDim});
    auto fam_expanded = families.unsqueeze(1).unsqueeze(2).expand({-1, 1, kV5EmbedDim});
    auto q_sel = q_all.gather(1, fam_expanded).squeeze(1);

    torch::Tensor e_id;
    if (use_bf16_cache_ && !torch::GradMode::is_enabled() &&
        v5_item_emb_bf16_.defined() && item_ids.is_cuda() &&
        at::autocast::is_autocast_enabled(at::kCUDA)) {
      e_id = at::embedding(v5_item_emb_bf16_, item_ids);
    } else {
      e_id = v5_item_emb->forward(item_ids);
    }

    torch::Tensor c_emb = (card_ids.defined() && card_embedding)
        ? card_embedding->forward(card_ids)
        : torch::zeros({item_ids.size(0), kCardEmbedDim}, trunk.options());
    if (torch::GradMode::is_enabled()) {
      c_emb = c_emb.detach();
    }
    auto c_masked = c_emb * (item_ids < 256).unsqueeze(-1).to(c_emb.dtype());

    auto k_item = e_id +
                  v5_card_adapter_bf16_.Forward(v5_card_adapter, c_masked, use_bf16_cache_) +
                  v5_ctx_proj_bf16_.Forward(v5_ctx_proj, ctx_features, use_bf16_cache_);

    return (q_sel.to(torch::kFloat32) * k_item.to(torch::kFloat32)).sum(-1);
  }
};
TORCH_MODULE(SemanticActionScorer);

struct SemanticScorerBatchWorkspace {
  int64_t capacity_m = 0;
  torch::Device dev = torch::kCPU;
  torch::Tensor ints_host_buf;
  torch::Tensor bools_host_buf;
  torch::Tensor supp_host_buf;
  torch::Tensor feat_host_buf;
  torch::Tensor v5_ctx_host_buf;
  torch::Tensor ints_dev_buf;
  torch::Tensor bools_dev_buf;
  torch::Tensor supp_dev_buf;
  torch::Tensor feat_dev_buf;
  torch::Tensor v5_ctx_dev_buf;

  void EnsureCapacity(int64_t m, torch::Device device) {
    const bool on_cuda = device.is_cuda();
    if (m <= capacity_m &&
        (!on_cuda || (ints_dev_buf.defined() && dev == device)) &&
        ints_host_buf.defined() && ints_host_buf.is_pinned() == on_cuda) {
      return;
    }
    const int64_t alloc_m =
        std::max<int64_t>(m, std::max<int64_t>(capacity_m * 2, 4096));
    auto host_opts =
        torch::TensorOptions().device(torch::kCPU).pinned_memory(on_cuda);
    ints_host_buf = torch::empty({6 * alloc_m}, host_opts.dtype(torch::kInt64));
    bools_host_buf = torch::empty({2 * alloc_m}, host_opts.dtype(torch::kBool));
    supp_host_buf = torch::empty({alloc_m}, host_opts.dtype(torch::kFloat32));
    feat_host_buf = torch::empty(
        {alloc_m, static_cast<int64_t>(kSemanticFeatDim)},
        host_opts.dtype(torch::kFloat32));
    v5_ctx_host_buf = torch::empty(
        {alloc_m, static_cast<int64_t>(kV5CtxDim)},
        host_opts.dtype(torch::kFloat32));
    if (on_cuda) {
      auto dev_opts = torch::TensorOptions().device(device);
      ints_dev_buf = torch::empty({6 * alloc_m}, dev_opts.dtype(torch::kInt64));
      bools_dev_buf = torch::empty({2 * alloc_m}, dev_opts.dtype(torch::kBool));
      supp_dev_buf = torch::empty({alloc_m}, dev_opts.dtype(torch::kFloat32));
      feat_dev_buf = torch::empty(
          {alloc_m, static_cast<int64_t>(kSemanticFeatDim)},
          dev_opts.dtype(torch::kFloat32));
      v5_ctx_dev_buf = torch::empty(
          {alloc_m, static_cast<int64_t>(kV5CtxDim)},
          dev_opts.dtype(torch::kFloat32));
    }
    capacity_m = alloc_m;
    dev = device;
  }
};

// Helper function to gather candidate descriptors from a batch and apply corrections.
inline void ApplySemanticScorerBatch(
    const std::shared_ptr<SemanticActionScorerImpl>& scorer,
    const torch::Tensor& trunk,
    const std::vector<const CandidateActionData*>& batch_actions,
    torch::Tensor& inout_logits,
    torch::Device device,
    SemanticScorerBatchWorkspace* workspace = nullptr,
    bool allow_inplace = false,
    const torch::Tensor& precomputed_trunk_proj = torch::Tensor()) {
  if (!scorer) return;
  size_t total_cands = 0;
  for (const auto* data : batch_actions) {
    if (data != nullptr) total_cands += data->size();
  }
  if (total_cands == 0) return;

  const bool on_cuda = device.is_cuda();
  const int64_t m = static_cast<int64_t>(total_cands);

  torch::Tensor ints_host, bools_host, supp_host, feat_host, v5_ctx_host;
  SemanticScorerBatchWorkspace* active_ws = workspace;
  if (active_ws == nullptr && !torch::GradMode::is_enabled()) {
    thread_local SemanticScorerBatchWorkspace tls_ws;
    active_ws = &tls_ws;
  }
  if (active_ws != nullptr) {
    active_ws->EnsureCapacity(m, device);
    ints_host = active_ws->ints_host_buf.narrow(0, 0, 6 * m).view({6, m});
    bools_host = active_ws->bools_host_buf.narrow(0, 0, 2 * m).view({2, m});
    supp_host = active_ws->supp_host_buf.narrow(0, 0, m);
    feat_host = active_ws->feat_host_buf.narrow(0, 0, m);
    v5_ctx_host = active_ws->v5_ctx_host_buf.narrow(0, 0, m);
  } else {
    auto host_opts = torch::TensorOptions().pinned_memory(on_cuda);
    ints_host = torch::empty({6, m}, host_opts.dtype(torch::kInt64));
    bools_host = torch::empty({2, m}, host_opts.dtype(torch::kBool));
    supp_host = torch::empty({m}, host_opts.dtype(torch::kFloat32));
    feat_host = torch::empty(
        {m, static_cast<int64_t>(kSemanticFeatDim)},
        host_opts.dtype(torch::kFloat32));
    v5_ctx_host = torch::empty(
        {m, static_cast<int64_t>(kV5CtxDim)},
        host_opts.dtype(torch::kFloat32));
  }

  int64_t* b_ptr = ints_host.data_ptr<int64_t>();
  int64_t* a_ptr = b_ptr + m;
  int64_t* c_ptr = b_ptr + 2 * m;
  int64_t* s_ptr = b_ptr + 3 * m;
  int64_t* v5_item_ptr = b_ptr + 4 * m;
  int64_t* v5_fam_ptr = b_ptr + 5 * m;

  bool* is_ext_ptr = bools_host.data_ptr<bool>();
  bool* is_f_ptr = is_ext_ptr + m;

  float* supp_ptr = supp_host.data_ptr<float>();
  float* feat_ptr = feat_host.data_ptr<float>();
  float* v5_ctx_ptr = v5_ctx_host.data_ptr<float>();

  bool any_ext = false;
  bool any_f = false;
  bool any_v5 = false;
  size_t cursor = 0;
  for (size_t b = 0; b < batch_actions.size(); ++b) {
    const auto* data = batch_actions[b];
    if (data == nullptr || data->empty()) continue;
    const size_t sz = data->size();
    const bool has_v5 = !data->v5_families.empty();
    for (size_t j = 0; j < sz; ++j) {
      const size_t idx = cursor + j;
      b_ptr[idx] = static_cast<int64_t>(b);
      a_ptr[idx] = static_cast<int64_t>(data->actions[j]);
      c_ptr[idx] = data->card_ids[j];
      s_ptr[idx] = data->space_ids[j];
      supp_ptr[idx] = data->supported[j] ? 1.0f : 0.0f;
      bool ext = false;
      bool f_role = false;
      if (j < data->roles.size()) {
        ext = (data->roles[j] == ActionRole::kIndexedIntrigue ||
               data->roles[j] == ActionRole::kGraftSolo ||
               data->roles[j] == ActionRole::kHundroOffer);
        f_role = (data->roles[j] == ActionRole::kTleilaxuPurchase);
      }
      is_ext_ptr[idx] = ext;
      is_f_ptr[idx] = f_role;
      if (ext) any_ext = true;
      if (f_role) any_f = true;

      int64_t v5_fam = (has_v5 && j < data->v5_families.size()) ? data->v5_families[j] : -1;
      int64_t v5_item = (has_v5 && j < data->v5_item_ids.size()) ? data->v5_item_ids[j] : 279;
      v5_fam_ptr[idx] = v5_fam;
      v5_item_ptr[idx] = v5_item;
      if (v5_fam >= 0) any_v5 = true;
    }
    if (!data->features.empty()) {
      std::memcpy(feat_ptr + cursor * kSemanticFeatDim,
                  data->features.data(),
                  data->features.size() * sizeof(float));
    }
    if (has_v5 && !data->v5_ctx_features.empty()) {
      std::memcpy(v5_ctx_ptr + cursor * kV5CtxDim,
                  data->v5_ctx_features.data(),
                  data->v5_ctx_features.size() * sizeof(float));
    } else {
      std::memset(v5_ctx_ptr + cursor * kV5CtxDim, 0, sz * kV5CtxDim * sizeof(float));
    }
    cursor += sz;
  }

  if (any_f && (!scorer->enable_f_ || !scorer->out_layer_f)) {
    SpielFatalError(
        "SemanticActionScorer::ComputeAndAddCorrections: kTleilaxuPurchase "
        "(is_f_mask=true) encountered while enable_f_ is false; "
        "semantic_action_v4 scorer required.");
  }

  torch::Tensor ints_dev, supp_t, feat_t, v5_ctx_t;
  torch::Tensor is_ext_t, is_f_t;
  if (on_cuda && active_ws != nullptr) {
    ints_dev = active_ws->ints_dev_buf.narrow(0, 0, 6 * m).view({6, m});
    supp_t = active_ws->supp_dev_buf.narrow(0, 0, m);
    feat_t = active_ws->feat_dev_buf.narrow(0, 0, m);
    ints_dev.copy_(ints_host, /*non_blocking=*/true);
    supp_t.copy_(supp_host, /*non_blocking=*/true);
    feat_t.copy_(feat_host, /*non_blocking=*/true);
    if (any_ext || any_f) {
      auto bools_dev = active_ws->bools_dev_buf.narrow(0, 0, 2 * m).view({2, m});
      bools_dev.copy_(bools_host, /*non_blocking=*/true);
      is_ext_t = bools_dev[0];
      is_f_t = bools_dev[1];
    }
    if (any_v5) {
      v5_ctx_t = active_ws->v5_ctx_dev_buf.narrow(0, 0, m);
      v5_ctx_t.copy_(v5_ctx_host, /*non_blocking=*/true);
    }
  } else {
    ints_dev = on_cuda ? ints_host.to(device, /*non_blocking=*/true) : ints_host;
    supp_t = on_cuda ? supp_host.to(device, /*non_blocking=*/true) : supp_host;
    feat_t = on_cuda ? feat_host.to(device, /*non_blocking=*/true) : feat_host;
    if (any_ext || any_f) {
      torch::Tensor bools_dev = on_cuda ? bools_host.to(device, /*non_blocking=*/true) : bools_host;
      is_ext_t = bools_dev[0];
      is_f_t = bools_dev[1];
    }
    if (any_v5) {
      v5_ctx_t = on_cuda ? v5_ctx_host.to(device, /*non_blocking=*/true) : v5_ctx_host;
    }
  }

  torch::Tensor b_t = ints_dev[0];
  torch::Tensor a_t = ints_dev[1];
  torch::Tensor c_t = ints_dev[2];
  torch::Tensor s_t = ints_dev[3];
  torch::Tensor v5_item_t = ints_dev[4];
  torch::Tensor v5_fam_t = ints_dev[5];

  scorer->ComputeAndAddCorrections(
      trunk, b_t, a_t, feat_t, c_t, s_t, supp_t, inout_logits,
      is_ext_t, is_f_t, allow_inplace, any_ext, any_f, precomputed_trunk_proj,
      v5_item_t, v5_fam_t, v5_ctx_t, any_v5);
}

#endif // OPEN_SPIEL_BUILD_WITH_LIBTORCH

} // namespace dune_semantic
} // namespace open_spiel

#endif // OPEN_SPIEL_EXAMPLES_DUNE_SEMANTIC_ACTION_SCORER_H_
