#ifndef OPEN_SPIEL_EXAMPLES_DUNE_SEMANTIC_ACTION_SCORER_H_
#define OPEN_SPIEL_EXAMPLES_DUNE_SEMANTIC_ACTION_SCORER_H_

#include <vector>
#include <string>
#include <cstdint>
#include <algorithm>
#include <cmath>

#include "open_spiel/spiel.h"
#include "open_spiel/games/dune_imperium/dune_imperium.h"
#include "open_spiel/games/dune_imperium/dune_imperium_cards.h"
#include "open_spiel/games/dune_imperium/dune_imperium_board.h"
#include "open_spiel/games/dune_imperium/dune_imperium_common.h"

#ifdef OPEN_SPIEL_BUILD_WITH_LIBTORCH
#include <torch/torch.h>
#endif

namespace open_spiel {
namespace dune_semantic {

inline constexpr const char* kDescriptorSchemaVersion = "semantic_action_v2";

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

  bool empty() const { return actions.empty(); }
  size_t size() const { return actions.size(); }
  void clear() {
    actions.clear();
    features.clear();
    card_ids.clear();
    space_ids.clear();
    supported.clear();
  }
};

// Extracts candidate action descriptors and categorical IDs for all legal actions at the given state.
inline void ExtractCandidateDescriptors(
    const dune_imperium::DuneImperiumState& state,
    const std::vector<Action>& legal_actions,
    CandidateActionData* out_data) {
  if (out_data == nullptr) return;
  out_data->clear();
  const size_t num_acts = legal_actions.size();
  out_data->actions = legal_actions;
  out_data->features.assign(num_acts * kSemanticFeatDim, 0.0f);
  out_data->card_ids.assign(num_acts, 0);
  out_data->space_ids.assign(num_acts, 0);
  out_data->supported.assign(num_acts, 0);

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
      } else {
        out_data->card_ids[i] = 0;
      }
      continue;
    }

    // 4. Indexed Intrigue Choices (actions 700..719)
    if (action >= dune_imperium::kActionIntrigueChoiceTrashIntrigue0 &&
        action < dune_imperium::kActionIntrigueChoiceTrashIntrigue0 + 20) {
      const int idx = action - dune_imperium::kActionIntrigueChoiceTrashIntrigue0;
      f[0] = 1.0f;
      out_data->space_ids[i] = 0;
      out_data->supported[i] = 0;
      if (cur_player >= 0 && cur_player < state.NumPlayers()) {
        const auto& hand = state.GetIntrigueHandForTesting(cur_player);
        if (idx >= 0 && idx < static_cast<int>(hand.size())) {
          out_data->card_ids[i] = hand[idx] + 1;
        } else {
          out_data->card_ids[i] = 0;
        }
      } else {
        out_data->card_ids[i] = 0;
      }
      continue;
    }

    // 5. Graft Primary Identity on Solo Agent Play
    if (action == dune_imperium::kActionPlayAgentSolo) {
      f[0] = 1.0f;
      out_data->card_ids[i] = state.GetPendingGraftPrimaryId();
      out_data->space_ids[i] = 0;
      out_data->supported[i] = 0;
      continue;
    }

    // 6. Hundro Setup Mode A/B Offers
    if (state.GetPendingHundroSetupPlayer() == cur_player) {
      if (action == dune_imperium::kActionIntrigueChoiceModeA) {
        f[0] = 1.0f;
        out_data->card_ids[i] = state.GetPendingHundroSetupCard(0) + 1;
        out_data->space_ids[i] = 0;
        out_data->supported[i] = 0;
        continue;
      } else if (action == dune_imperium::kActionIntrigueChoiceModeB) {
        f[0] = 1.0f;
        out_data->card_ids[i] = state.GetPendingHundroSetupCard(1) + 1;
        out_data->space_ids[i] = 0;
        out_data->supported[i] = 0;
        continue;
      }
    }

    // 7. Unsupported Roles
    f[0] = 1.0f; // is_unsupported
    out_data->card_ids[i] = 0;
    out_data->space_ids[i] = 0;
    out_data->supported[i] = 0;
  }
}

#ifdef OPEN_SPIEL_BUILD_WITH_LIBTORCH

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

  SemanticActionScorerImpl() {
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

    // Initialize the final output layer to zero so initial corrections are exactly 0.0.
    {
      torch::NoGradGuard no_grad;
      out_layer->weight.zero_();
      if (out_layer->bias.defined()) {
        out_layer->bias.zero_();
      }
    }
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
      torch::Tensor& inout_logits) {         // [B, 2391]
    if (batch_indices.numel() == 0 || feat_tensor.numel() == 0) {
      return;
    }

    auto c_emb = card_embedding->forward(card_ids);     // [M, 32]
    auto s_emb = space_embedding->forward(space_ids);   // [M, 16]
    auto d_full = torch::cat({feat_tensor, c_emb, s_emb}, -1); // [M, 104]

    auto d_p = torch::relu(desc_ln->forward(desc_proj->forward(d_full)));  // [M, 256]
    auto h_p = torch::relu(trunk_ln->forward(trunk_proj->forward(trunk))); // [B, 256]
    auto h_gathered = h_p.index_select(0, batch_indices);                  // [M, 256]

    auto z = torch::cat({h_gathered, d_p}, -1);                            // [M, 512]
    auto u = torch::relu(mlp1_ln->forward(mlp1->forward(z)));              // [M, 256]
    auto raw_corr = out_layer->forward(u).squeeze(-1);                     // [M]
    auto corr = (raw_corr * supported_mask).to(inout_logits.dtype());                                 // [M]

    // Out-of-place index_put preserves autograd graph for backward passes.
    inout_logits = inout_logits.index_put({batch_indices, action_indices},
                                          inout_logits.index({batch_indices, action_indices}) + corr);
  }
};
TORCH_MODULE(SemanticActionScorer);

// Helper function to gather candidate descriptors from a batch and apply corrections.
inline void ApplySemanticScorerBatch(
    const std::shared_ptr<SemanticActionScorerImpl>& scorer,
    const torch::Tensor& trunk,
    const std::vector<const CandidateActionData*>& batch_actions,
    torch::Tensor& inout_logits,
    torch::Device device) {
  if (!scorer) return;
  size_t total_cands = 0;
  for (const auto* data : batch_actions) {
    if (data != nullptr) total_cands += data->size();
  }
  if (total_cands == 0) return;

  std::vector<int64_t> batch_indices;
  std::vector<int64_t> action_indices;
  std::vector<float> features;
  std::vector<int64_t> card_ids;
  std::vector<int64_t> space_ids;
  std::vector<float> supported;

  batch_indices.reserve(total_cands);
  action_indices.reserve(total_cands);
  features.reserve(total_cands * kSemanticFeatDim);
  card_ids.reserve(total_cands);
  space_ids.reserve(total_cands);
  supported.reserve(total_cands);

  for (size_t b = 0; b < batch_actions.size(); ++b) {
    const auto* data = batch_actions[b];
    if (data == nullptr || data->empty()) continue;
    const size_t sz = data->size();
    for (size_t j = 0; j < sz; ++j) {
      batch_indices.push_back(static_cast<int64_t>(b));
      action_indices.push_back(static_cast<int64_t>(data->actions[j]));
      card_ids.push_back(data->card_ids[j]);
      space_ids.push_back(data->space_ids[j]);
      supported.push_back(data->supported[j] ? 1.0f : 0.0f);
    }
    features.insert(features.end(), data->features.begin(), data->features.end());
  }

  auto opts_i64 = torch::TensorOptions().dtype(torch::kInt64).device(device);
  auto opts_f32 = torch::TensorOptions().dtype(torch::kFloat32).device(device);

  torch::Tensor b_t = torch::tensor(batch_indices, opts_i64);
  torch::Tensor a_t = torch::tensor(action_indices, opts_i64);
  torch::Tensor c_t = torch::tensor(card_ids, opts_i64);
  torch::Tensor s_t = torch::tensor(space_ids, opts_i64);
  torch::Tensor supp_t = torch::tensor(supported, opts_f32);
  torch::Tensor feat_t = torch::tensor(features, torch::TensorOptions().dtype(torch::kFloat32))
                             .reshape({static_cast<int64_t>(total_cands), static_cast<int64_t>(kSemanticFeatDim)})
                             .to(device);

  scorer->ComputeAndAddCorrections(trunk, b_t, a_t, feat_t, c_t, s_t, supp_t, inout_logits);
}

#endif // OPEN_SPIEL_BUILD_WITH_LIBTORCH

} // namespace dune_semantic
} // namespace open_spiel

#endif // OPEN_SPIEL_EXAMPLES_DUNE_SEMANTIC_ACTION_SCORER_H_
