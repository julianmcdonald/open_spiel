#ifndef OPEN_SPIEL_EXAMPLES_DUNE_NETWORK_MODEL_H_
#define OPEN_SPIEL_EXAMPLES_DUNE_NETWORK_MODEL_H_

#include <ATen/autocast_mode.h>
#include <torch/torch.h>

#include <cmath>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "dune_seed_utils.h"  // MakeTorchCPUGenerator, for the PWO-5 head init
#include "dune_semantic_action_scorer.h"

namespace open_spiel {

using dune_semantic::Bf16LinearCache;
using dune_semantic::Fp8LinearCache;

// 1. Residual Block Definition
struct ResBlockImpl : torch::nn::Module {
  torch::nn::Linear fc1{nullptr};
  torch::nn::Linear fc2{nullptr};
  torch::nn::LayerNorm ln1{nullptr};
  torch::nn::LayerNorm ln2{nullptr};

  Bf16LinearCache fc1_bf16_;
  Bf16LinearCache fc2_bf16_;
  Fp8LinearCache fc1_fp8_;
  Fp8LinearCache fc2_fp8_;
  bool use_fp8_ = false;

  ResBlockImpl(int64_t dim) {
    fc1 = register_module("fc1", torch::nn::Linear(dim, dim));
    fc2 = register_module("fc2", torch::nn::Linear(dim, dim));
    ln1 = register_module("ln1", torch::nn::LayerNorm(torch::nn::LayerNormOptions({dim})));
    ln2 = register_module("ln2", torch::nn::LayerNorm(torch::nn::LayerNormOptions({dim})));
  }

  void RefreshBf16Cache() {
    fc1_bf16_.Refresh(fc1);
    fc2_bf16_.Refresh(fc2);
    if (use_fp8_) {
      fc1_fp8_.Refresh(fc1);
      fc2_fp8_.Refresh(fc2);
    }
  }

  void PrepareFp8Cache() {
    fc1_bf16_.Refresh(fc1);
    fc2_bf16_.Refresh(fc2);
    fc1_fp8_.Refresh(fc1);
    fc2_fp8_.Refresh(fc2);
    use_fp8_ = true;
  }

  void DisableFp8Cache() {
    use_fp8_ = false;
  }

  torch::Tensor forward(torch::Tensor x, bool use_bf16_cache = false, bool use_fp8 = false, bool input_non_negative = false) {
    torch::Tensor residual = x;
    const bool opt_fp8 = use_fp8 || use_fp8_;
    if (opt_fp8) {
      x = torch::relu(ln1->forward(fc1_fp8_.Forward(fc1, x, use_bf16_cache, /*use_fp8=*/true, input_non_negative)));
      x = ln2->forward(fc2_fp8_.Forward(fc2, x, use_bf16_cache, /*use_fp8=*/true, /*known_non_negative=*/true));
    } else {
      x = torch::relu(ln1->forward(fc1_bf16_.Forward(fc1, x, use_bf16_cache)));
      x = ln2->forward(fc2_bf16_.Forward(fc2, x, use_bf16_cache));
    }
    return torch::relu(x + residual);
  }
};
TORCH_MODULE(ResBlock);
using ResidualBlockImpl = ResBlockImpl;

// Multiplier applied to the value head's scalar output layer at construction.
// See the rationale block in the constructor below.
inline constexpr double kValueHeadOutputInitScale = 0.01;

// PWO-5 section 7.2: the same x0.01 applied to the three auxiliary heads'
// output weights, for the same reason and with the same precedent.
inline constexpr double kAuxHeadOutputInitScale = 0.01;

// 1b. Zone Encoder for Prototype F: extracts card zone features from 9,182 obs,
// pools them via shared card_embedding weights, and projects into trunk.
struct ZoneEncoderImpl : torch::nn::Module {
  static constexpr int64_t kZoneInputDim = 1370;
  static constexpr int64_t kZoneHiddenDim = 512;
  static constexpr int64_t kTrunkDim = 2048;

  torch::nn::Linear fc1{nullptr};
  torch::nn::LayerNorm ln1{nullptr};
  torch::nn::Linear out_proj{nullptr};

  torch::Tensor tleilaxu_card_map_;

  Bf16LinearCache fc1_bf16_;
  Bf16LinearCache out_proj_bf16_;
  torch::Tensor card_embed_bf16_;

  ZoneEncoderImpl() {
    fc1 = register_module("fc1", torch::nn::Linear(kZoneInputDim, kZoneHiddenDim));
    ln1 = register_module("ln1", torch::nn::LayerNorm(torch::nn::LayerNormOptions({kZoneHiddenDim})));
    out_proj = register_module("out_proj", torch::nn::Linear(kZoneHiddenDim, kTrunkDim));

    // Zero-initialize out_proj (weight and bias) so initial output delta is exactly 0.0.
    {
      torch::NoGradGuard no_grad;
      out_proj->weight.zero_();
      if (out_proj->bias.defined()) {
        out_proj->bias.zero_();
      }
    }
  }

  void load(torch::serialize::InputArchive& archive) {
    torch::serialize::InputArchive fc1_arch, ln1_arch, out_arch;
    if (archive.try_read("fc1", fc1_arch)) fc1->load(fc1_arch);
    if (archive.try_read("ln1", ln1_arch)) ln1->load(ln1_arch);
    if (archive.try_read("out_proj", out_arch)) out_proj->load(out_arch);
  }

  void RefreshBf16Cache(const torch::Tensor& card_embed) {
    fc1_bf16_.Refresh(fc1);
    out_proj_bf16_.Refresh(out_proj);
    if (card_embed.defined() && card_embed.is_cuda()) {
      torch::NoGradGuard no_grad;
      if (!card_embed_bf16_.defined() || card_embed_bf16_.sizes() != card_embed.sizes() ||
          card_embed_bf16_.device() != card_embed.device() ||
          card_embed_bf16_.scalar_type() != torch::kBFloat16) {
        card_embed_bf16_ = torch::empty(card_embed.sizes(),
                                        card_embed.options().dtype(torch::kBFloat16));
      }
      card_embed_bf16_.copy_(card_embed);
    } else {
      card_embed_bf16_ = torch::Tensor();
    }
  }

  torch::Tensor forward(const torch::Tensor& obs, const torch::Tensor& card_embed_table, bool use_bf16_cache = false) {
    if (obs.size(1) < dune_imperium::kFullPublicInformationStateSize) {
      return torch::zeros({obs.size(0), kTrunkDim}, obs.options());
    }

    if (!tleilaxu_card_map_.defined() ||
        tleilaxu_card_map_.device() != obs.device() ||
        tleilaxu_card_map_.dtype() != obs.dtype()) {
      torch::NoGradGuard no_grad;
      tleilaxu_card_map_ = torch::zeros(
          {20, dune_imperium::kNumImperiumCardIds},
          obs.options());
      for (size_t local_id = 1; local_id < dune_imperium::kTleilaxuCards.size(); ++local_id) {
        int shared_id = dune_imperium::kTleilaxuCards[local_id].imperium_card_id;
        if (shared_id >= 0 && shared_id < dune_imperium::kNumImperiumCardIds) {
          tleilaxu_card_map_[local_id][shared_id] = 1.0f;
        }
      }
    }

    const int64_t B = obs.size(0);
    // card_embed_table is [256, 32]; slice to [126, 32]
    auto card_embed = (use_bf16_cache && !torch::GradMode::is_enabled() &&
                       card_embed_bf16_.defined() &&
                       at::autocast::is_autocast_enabled(at::kCUDA) && obs.is_cuda())
                          ? card_embed_bf16_.slice(0, 0, dune_imperium::kNumImperiumCardIds)
                          : card_embed_table.slice(0, 0, dune_imperium::kNumImperiumCardIds);
    const int64_t C = card_embed.size(1); // 32

    // 1. Self piles: 5 piles x 126 dims, starting at 475 (hand, draw, discard, played, revealed)
    auto self_piles = obs.slice(1, 475, 475 + 5 * 126).reshape({B, 5, 126});
    auto self_embed = torch::matmul(self_piles, card_embed).reshape({B, 5 * C}); // [B, 160]

    // 2. Opponent piles:
    // Opponent deck pools: 3 opponents x 3 piles (total, discard, draw)
    auto opp1_piles = obs.slice(1, 1146, 1146 + 3 * 126);
    auto opp2_piles = obs.slice(1, 1525, 1525 + 3 * 126);
    auto opp3_piles = obs.slice(1, 1904, 1904 + 3 * 126);
    // Opponent revealed: 3 opponents x 126 dims at 6255
    auto opp_revealed = obs.slice(1, 6255, 6255 + 3 * 126);
    auto all_opp = torch::cat({opp1_piles, opp2_piles, opp3_piles, opp_revealed}, 1).reshape({B, 12, 126});
    auto opp_embed = torch::matmul(all_opp, card_embed).reshape({B, 12 * C}); // [B, 384]

    // 3. Market slots: 5 slots x 127 dims at 5580 (126 one-hot + 1 empty)
    auto mkt_raw = obs.slice(1, 5580, 5580 + 5 * 127).reshape({B, 5, 127});
    auto mkt_cards = mkt_raw.slice(2, 0, 126);
    auto mkt_empty = mkt_raw.slice(2, 126, 127);
    auto mkt_card_embed = torch::matmul(mkt_cards, card_embed); // [B, 5, C]
    auto mkt_combined = torch::cat({mkt_card_embed, mkt_empty}, 2).reshape({B, 5 * (C + 1)}); // [B, 165]

    // 4. Tleilaxu market: 2 slots x 20 dims at 6215 (local 0 = Reclaimed, 1..18 = cards, 19 = empty)
    auto tx_raw = obs.slice(1, 6215, 6215 + 2 * 20).reshape({B, 2, 20});
    auto tx_reclaimed = tx_raw.slice(2, 0, 1);
    auto tx_empty = tx_raw.slice(2, 19, 20);
    auto tx_cards_126 = torch::matmul(tx_raw, tleilaxu_card_map_);
    auto tx_card_embed = torch::matmul(tx_cards_126, card_embed); // [B, 2, C]
    auto tx_combined = torch::cat({tx_card_embed, tx_reclaimed, tx_empty}, 2).reshape({B, 2 * (C + 2)}); // [B, 68]

    // 5. Ordered topdeck: 4 players x 4 depths x 127 dims at 6655
    auto td_raw = obs.slice(1, 6655, 6655 + 16 * 127).reshape({B, 16, 127});
    auto td_cards = td_raw.slice(2, 0, 126);
    auto td_empty = td_raw.slice(2, 126, 127);
    auto td_card_embed = torch::matmul(td_cards, card_embed); // [B, 16, C]
    auto td_combined = torch::cat({td_card_embed, td_empty}, 2).reshape({B, 16 * (C + 1)}); // [B, 528]

    // 6. Pending graft: 1 slot x 127 dims at 8691
    auto graft_raw = obs.slice(1, 8691, 8691 + 127);
    auto graft_card = graft_raw.slice(1, 0, 126);
    auto graft_empty = graft_raw.slice(1, 126, 127);
    auto graft_card_embed = torch::matmul(graft_card, card_embed); // [B, C]
    auto graft_combined = torch::cat({graft_card_embed, graft_empty}, 1); // [B, C + 1 = 33]

    // 7. Imperium discard: 126 dims at 2632
    auto imp_discard = obs.slice(1, 2632, 2632 + 126);
    auto imp_discard_embed = torch::matmul(imp_discard, card_embed); // [B, C = 32]

    auto zone_repr = torch::cat({
        self_embed,         // 160
        opp_embed,          // 384
        mkt_combined,       // 165
        tx_combined,        // 68
        td_combined,        // 528
        graft_combined,     // 33
        imp_discard_embed   // 32
    }, 1); // Total: 1370 dims

    auto h = torch::relu(ln1->forward(fc1_bf16_.Forward(fc1, zone_repr, use_bf16_cache)));
    return out_proj_bf16_.Forward(out_proj, h, use_bf16_cache);
  }
};
TORCH_MODULE(ZoneEncoder);

inline int SemanticSchemaRank(const std::string& schema) {
  if (schema == dune_semantic::kDescriptorSchemaVersionV5) return 5;
  if (schema == dune_semantic::kDescriptorSchemaVersionV4) return 4;
  if (schema == dune_semantic::kDescriptorSchemaVersionV3) return 3;
  if (schema == dune_semantic::kDescriptorSchemaVersionV2) return 2;
  return 0;
}

// 2. Main Dual-Headed Policy-Value Network Definition
struct SharedDunePolicyValueNetImpl : torch::nn::Module {
  torch::nn::Linear input_layer{nullptr};
  std::vector<std::shared_ptr<ResBlockImpl>> res_blocks;
  torch::nn::Linear policy_head{nullptr};
  torch::nn::Linear value_head{nullptr};
  torch::nn::Linear value_head2{nullptr};
  bool use_nonlinear_value_head_ = false;

  Bf16LinearCache input_layer_bf16_;
  Bf16LinearCache policy_head_bf16_;
  Bf16LinearCache value_head_bf16_;
  Bf16LinearCache value_head2_bf16_;
  bool use_bf16_inference_cache_ = false;
  bool use_fp8_trunk_ = false;

  // --- PWO-5 section 7.2: the three auxiliary heads. -----------------------
  //
  // Constructed ONLY when `with_aux_heads` is true, which ONLY dune_ppo_train
  // passes. That default is load-bearing three times over:
  //
  //   * every evaluation binary keeps the pre-migration module set, so it can
  //     still load the frozen Branch-A u2450 opponent checkpoint, which has no
  //     head tensors;
  //   * a candidate checkpoint that DOES have them still loads into a
  //     head-less module, because torch's Module::load walks the MODULE's own
  //     members and ignores extra archive entries;
  //   * it makes section 9.5's "auxiliary outputs unused at runtime" a
  //     STRUCTURAL fact rather than a convention -- the evaluator cannot
  //     compute a head output because the head does not exist in its process.
  //
  // The heads are NEVER computed by forward(). Section 8.6 registers a
  // "dedicated auxiliary forward path": ForwardAux() below is the only entry
  // point that touches them, and it is called only from the auxiliary slice of
  // the PPO minibatch loop, only when a head coefficient is nonzero. So a
  // head-off arm (T, P) never computes a head output at all, which is what
  // makes the head-off claim exact rather than a multiply-by-zero.
  torch::nn::Linear final_vp_head{nullptr};        // 1 output
  torch::nn::Linear terminal_round_head{nullptr};  // 3 classes: <=8 / 9 / 10
  torch::nn::Linear next_own_action_head{nullptr}; // action_dim classes
  bool has_aux_heads_ = false;
  bool with_semantic_scorer_ = false;
  std::shared_ptr<dune_semantic::SemanticActionScorerImpl> semantic_scorer_{nullptr};
  bool enable_zone_encoder_ = false;
  std::shared_ptr<ZoneEncoderImpl> zone_encoder_{nullptr};
  dune_imperium::MarketAppendixMode market_appendix_mode_ = dune_imperium::MarketAppendixMode::kNone;
  dune_imperium::DeckPoolSemantics deck_pool_semantics_ = dune_imperium::DeckPoolSemantics::kLegacy;
  std::string semantic_descriptor_schema_ = dune_semantic::kDescriptorSchemaVersionV3;
  torch::Tensor v5_dynamic_slot_mask_;

  SharedDunePolicyValueNetImpl(int64_t input_dim, int64_t hidden_dim = 2048, int64_t action_dim = 2391, int num_blocks = 8, bool use_nonlinear = false, bool with_aux_heads = false, uint64_t head_init_seed = 0, bool with_semantic_scorer = false, bool enable_zone_encoder = false, std::string semantic_descriptor_schema = "") {
    use_nonlinear_value_head_ = use_nonlinear;
    with_semantic_scorer_ = with_semantic_scorer;
    if (!semantic_descriptor_schema.empty()) {
      semantic_descriptor_schema_ = semantic_descriptor_schema;
    }
    if (input_dim == dune_imperium::kFullPublicInformationV5StateSize) {
      market_appendix_mode_ = dune_imperium::MarketAppendixMode::kFullPublicInformationV5;
      if (with_semantic_scorer_ && semantic_descriptor_schema.empty()) {
        semantic_descriptor_schema_ = dune_semantic::kDescriptorSchemaVersionV5;
      }
    } else if (input_dim == dune_imperium::kFullPublicInformationV4StateSize) {
      market_appendix_mode_ = dune_imperium::MarketAppendixMode::kFullPublicInformationV4;
      if (with_semantic_scorer_ && semantic_descriptor_schema.empty()) {
        semantic_descriptor_schema_ = dune_semantic::kDescriptorSchemaVersionV3;
      }
    } else if (input_dim == dune_imperium::kFullPublicInformationStateSize) {
      market_appendix_mode_ = dune_imperium::MarketAppendixMode::kFullPublicInformationV3;
      if (with_semantic_scorer_ && semantic_descriptor_schema.empty()) {
        semantic_descriptor_schema_ = dune_semantic::kDescriptorSchemaVersionV3;
      }
    } else if (input_dim == dune_imperium::kOrderedCardSlotsInformationStateSize) {
      market_appendix_mode_ = dune_imperium::MarketAppendixMode::kOrderedCardSlotsV2;
    } else if (input_dim == dune_imperium::kLegacyInformationStateSize) {
      market_appendix_mode_ = dune_imperium::MarketAppendixMode::kNone;
    }
    input_layer = register_module("input_layer", torch::nn::Linear(input_dim, hidden_dim));

    // Register custom submodules dynamically (res1, res2, etc.)
    for (int i = 0; i < num_blocks; ++i) {
      auto block = std::make_shared<ResBlockImpl>(hidden_dim);
      res_blocks.push_back(block);
      register_module("res" + std::to_string(i + 1), block);
    }

    policy_head = register_module("policy_head", torch::nn::Linear(hidden_dim, action_dim));
    if (use_nonlinear) {
      value_head = register_module("value_head", torch::nn::Linear(hidden_dim, hidden_dim / 2));
      value_head2 = register_module("value_head2", torch::nn::Linear(hidden_dim / 2, 1));
    } else {
      value_head = register_module("value_head", torch::nn::Linear(hidden_dim, 1));
    }

    // Small-init on the scalar-producing value layer (canonical PPO practice).
    //
    // Without it a from-scratch run drives the tanh at :forward() into hard
    // saturation on the FIRST optimizer step and stays pinned there for
    // hundreds of updates. The trap is not a step-size problem, so neither
    // grad_clip_norm nor a lower learning rate escapes it:
    //
    //   - the trunk output x is post-ReLU, so every component is >= 0 (~80%
    //     strictly positive, mean ~1.8, sum_i mean(x_i) ~ 3.6e3 at hidden=2048);
    //   - AdamW's first step is lr * m_hat/sqrt(v_hat) = lr * sign(g) for every
    //     coordinate, independent of |g|. Clipping rescales g and is therefore
    //     a no-op on the step actually taken;
    //   - dL/dx = delta * w, so with default init (|w|_2 ~ 0.57) the whole
    //     shared trunk is sign-stepped coherently in the one direction that
    //     moves the value output, and ~6.7e7 trunk coordinates each contribute
    //     lr to the same scalar. The resulting first-step displacement of the
    //     pre-tanh z is ~45 — two orders of magnitude past the [-0.4375,
    //     +0.5625] target range — so |v| lands at ~1.0 and 1 - v^2 then
    //     throttles the value gradient by ~50x while AdamW's beta2 memory still
    //     carries the pre-saturation second moment.
    //
    // Scaling this layer by 0.01 shrinks both z and dL/dx by the same factor
    // and keeps the critic inside the linear region of the tanh, where the
    // gradient it needs to fit the placement targets survives.
    //
    // This runs at CONSTRUCTION only. Every checkpoint path constructs and then
    // overwrites via torch::load, so transfer runs are bitwise unaffected. The
    // single exception is the nonlinear-head partial-copy path
    // (dune_ppo_train.cc LoadModelCheckpoint), which deliberately does not copy
    // the value head — there the fresh head is exactly the scratch-critic case
    // this guards, so the smaller init applies by design.
    {
      torch::NoGradGuard no_grad;
      torch::nn::Linear& value_out = use_nonlinear ? value_head2 : value_head;
      value_out->weight.mul_(kValueHeadOutputInitScale);
      if (value_out->bias.defined()) {
        value_out->bias.mul_(kValueHeadOutputInitScale);
      }
    }

    // --- PWO-5 section 7.2: deterministic auxiliary-head initialization. ----
    //
    // Registered recipe: linear from the trunk, "weights x 0.01 of the standard
    // init, bias 0", derived from a FIXED REGISTERED CONSTANT and NOT from the
    // run seed, so that the initial head parameters are BYTE-IDENTICAL ACROSS
    // ALL SIX ARMS (section 9's matching table checks this rather than assuming
    // it). Deriving from the triplet seed would give T1/P1/H1 one set and
    // T2/P2/H2 another, which satisfies the ablation's within-triplet
    // requirement but NOT the stronger all-six property the work order states.
    //
    // The x0.01 mirrors this repo's verified critic-init fix (submodule
    // b48a35ee): a standard-init tanh value head saturated at update 2 and
    // dwelt there ~837 updates. A near-zero-output init also keeps the
    // auxiliary gradients small at the start, so H's early trajectory is not
    // dominated by head transients.
    //
    // Determinism: torch::nn::Linear's constructor above already drew from the
    // global generator, so we OVERWRITE both tensors from a private,
    // seed-derived CPU generator instead of relying on that draw. Whatever the
    // ambient global RNG state is, these three heads come out identical.
    if (with_aux_heads) {
      has_aux_heads_ = true;
      final_vp_head =
          register_module("final_vp_head", torch::nn::Linear(hidden_dim, 1));
      terminal_round_head =
          register_module("terminal_round_head", torch::nn::Linear(hidden_dim, 3));
      next_own_action_head = register_module(
          "next_own_action_head", torch::nn::Linear(hidden_dim, action_dim));

      torch::NoGradGuard no_grad;
      at::Generator head_gen = dune_seed::MakeTorchCPUGenerator(head_init_seed);
      // kaiming_uniform_ with a=sqrt(5) is what torch::nn::Linear's own
      // reset_parameters() uses, so "the standard init" is reproduced exactly
      // and then scaled -- rather than a different distribution scaled to look
      // similar. bias is set to EXACTLY zero, per the registered recipe.
      auto init_head = [&](torch::nn::Linear& head) {
        const double bound =
            std::sqrt(6.0 / ((1.0 + 5.0) * head->weight.size(1)));
        head->weight.uniform_(-bound, bound, head_gen);
        head->weight.mul_(kAuxHeadOutputInitScale);
        if (head->bias.defined()) head->bias.zero_();
      };
      init_head(final_vp_head);
      init_head(terminal_round_head);
      init_head(next_own_action_head);
    }

    if (with_semantic_scorer_) {
      bool enable_f = (semantic_descriptor_schema_ == dune_semantic::kDescriptorSchemaVersionV4);
      bool enable_v5 = (semantic_descriptor_schema_ == dune_semantic::kDescriptorSchemaVersionV5);
      semantic_scorer_ = std::make_shared<dune_semantic::SemanticActionScorerImpl>(enable_f, false, enable_v5);
      register_module("semantic_scorer", semantic_scorer_);
    }
    if (enable_zone_encoder) {
      EnableZoneEncoder();
    }
  }

  void SetSemanticDescriptorSchema(const std::string& schema) {
    semantic_descriptor_schema_ = schema;
    if (with_semantic_scorer_ && semantic_scorer_) {
      if (schema == dune_semantic::kDescriptorSchemaVersionV4) {
        semantic_scorer_->EnableF();
      } else if (schema == dune_semantic::kDescriptorSchemaVersionV5) {
        semantic_scorer_->EnableV5();
      }
    }
  }

  void EnableSemanticScorer(bool enable_f = false, bool enable_v5 = false) {
    if (!with_semantic_scorer_) {
      with_semantic_scorer_ = true;
      enable_f = enable_f || (semantic_descriptor_schema_ == dune_semantic::kDescriptorSchemaVersionV4);
      enable_v5 = enable_v5 || (semantic_descriptor_schema_ == dune_semantic::kDescriptorSchemaVersionV5);
      semantic_scorer_ = std::make_shared<dune_semantic::SemanticActionScorerImpl>(enable_f, false, enable_v5);
      register_module("semantic_scorer", semantic_scorer_);
    }
  }

  void EnableZoneEncoder() {
    enable_zone_encoder_ = true;
    if (!zone_encoder_) {
      zone_encoder_ = std::make_shared<ZoneEncoderImpl>();
      register_module("zone_encoder", zone_encoder_);
    }
  }

  void RefreshBf16InferenceCache() {
    input_layer_bf16_.Refresh(input_layer);
    for (auto& block : res_blocks) {
      if (block) {
        block->RefreshBf16Cache();
      }
    }
    policy_head_bf16_.Refresh(policy_head);
    if (value_head) {
      value_head_bf16_.Refresh(value_head);
    }
    if (use_nonlinear_value_head_ && value_head2) {
      value_head2_bf16_.Refresh(value_head2);
    }
    if (with_semantic_scorer_ && semantic_scorer_) {
      semantic_scorer_->RefreshBf16Cache();
      if (enable_zone_encoder_ && zone_encoder_) {
        zone_encoder_->RefreshBf16Cache(semantic_scorer_->card_embedding->weight);
      }
    }
    use_bf16_inference_cache_ = true;
  }

  void DisableBf16InferenceCache() {
    DisableFp8Cache();
    use_bf16_inference_cache_ = false;
    if (with_semantic_scorer_ && semantic_scorer_) {
      semantic_scorer_->DisableBf16Cache();
    }
  }

  bool bf16_inference_cache_ready() const {
    return use_bf16_inference_cache_;
  }

  void PrepareFp8Cache() {
    RefreshBf16InferenceCache();
    for (auto& block : res_blocks) {
      if (block) {
        block->PrepareFp8Cache();
      }
    }
    use_fp8_trunk_ = true;
  }

  void DisableFp8Cache() {
    use_fp8_trunk_ = false;
    for (auto& block : res_blocks) {
      if (block) {
        block->DisableFp8Cache();
      }
    }
  }

  bool fp8_cache_ready() const {
    return use_fp8_trunk_;
  }

  void set_use_fp8_trunk(bool val) {
    if (val) {
      PrepareFp8Cache();
    } else {
      DisableFp8Cache();
    }
  }

  bool use_fp8_trunk() const {
    return use_fp8_trunk_;
  }

  struct ModelOutputs {
    torch::Tensor logits;
    torch::Tensor values;
    torch::Tensor trunk;
  };

  // The three auxiliary head outputs. Produced ONLY by ForwardAux().
  struct AuxOutputs {
    torch::Tensor final_vp;        // [B, 1]
    torch::Tensor terminal_round;  // [B, 3]  logits
    torch::Tensor next_own_action; // [B, action_dim]  logits
  };

  // Shared trunk. Split out of forward() so ForwardAux() reuses exactly the
  // same computation rather than a parallel copy of it.
  torch::Tensor Trunk(torch::Tensor x) {
    torch::Tensor trunk_in = input_layer_bf16_.Forward(input_layer, x, use_bf16_inference_cache_);
    if (enable_zone_encoder_ && zone_encoder_ && with_semantic_scorer_ && semantic_scorer_) {
      trunk_in = trunk_in + zone_encoder_->forward(x, semantic_scorer_->card_embedding->weight, use_bf16_inference_cache_);
    }
    x = torch::relu(trunk_in);
    for (auto& block : res_blocks) {
      x = block->forward(x, use_bf16_inference_cache_, use_fp8_trunk_, /*input_non_negative=*/true);
    }
    return x;
  }

  ModelOutputs forward(torch::Tensor x) {
    x = Trunk(x);

    torch::Tensor logits = policy_head_bf16_.Forward(policy_head, x, use_bf16_inference_cache_);
    if (torch::GradMode::is_enabled() &&
        semantic_descriptor_schema_ == dune_semantic::kDescriptorSchemaVersionV5 &&
        logits.size(-1) == 2391) {
      if (!v5_dynamic_slot_mask_.defined() || v5_dynamic_slot_mask_.device() != logits.device()) {
        static const std::vector<int64_t> kDynamicSlots = {
            10, 11, 12, 13, 14, 70, 71, 72, 91, 92, 538, 539, 540, 541, 542,
            546, 547, 548, 549, 550, 554, 555, 556, 572, 770, 771, 772};
        auto mask = torch::zeros({1, 2391}, torch::TensorOptions().dtype(torch::kBool).device(logits.device()));
        for (int64_t a : kDynamicSlots) {
          mask[0][a] = true;
        }
        v5_dynamic_slot_mask_ = mask;
      }
      logits = torch::where(v5_dynamic_slot_mask_, logits.detach(), logits);
    }
    torch::Tensor values;
    if (use_nonlinear_value_head_) {
      auto h = torch::relu(value_head_bf16_.Forward(value_head, x, use_bf16_inference_cache_));
      values = torch::tanh(value_head2_bf16_.Forward(value_head2, h, use_bf16_inference_cache_));
    } else {
      values = torch::tanh(value_head_bf16_.Forward(value_head, x, use_bf16_inference_cache_));
    }
    return {logits, values, x};
  }

  // PWO-5 section 8.6's DEDICATED AUXILIARY FORWARD PATH.
  //
  // The three heads are computed here and NOWHERE ELSE -- never in the PPO
  // forward, never in the rollout/inference forward, never in an evaluation
  // forward. Callers must check the coefficients first; this asserts rather
  // than silently returning empty tensors, because a head-off arm reaching this
  // function at all is a bug in the caller, not a no-op.
  AuxOutputs ForwardAux(torch::Tensor x) {
    if (!has_aux_heads_) {
      throw std::runtime_error(
          "ForwardAux called on a model constructed without auxiliary heads. "
          "The heads are a dedicated forward path (PWO-5 section 8.6); a "
          "head-off arm must never reach this call.");
    }
    torch::Tensor h = Trunk(x);
    return {final_vp_head->forward(h), terminal_round_head->forward(h),
            next_own_action_head->forward(h)};
  }
};
TORCH_MODULE(SharedDunePolicyValueNet);

}  // namespace open_spiel

#endif  // OPEN_SPIEL_EXAMPLES_DUNE_NETWORK_MODEL_H_
