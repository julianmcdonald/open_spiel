#ifndef OPEN_SPIEL_EXAMPLES_DUNE_SEMANTIC_ACTION_SCORER_NET_H_
#define OPEN_SPIEL_EXAMPLES_DUNE_SEMANTIC_ACTION_SCORER_NET_H_

#include "dune_semantic_action_descriptors.h"

#ifdef OPEN_SPIEL_BUILD_WITH_LIBTORCH
#include <atomic>
#include <cstdlib>
#include <optional>
#include <string>
#include <tuple>

#include <torch/torch.h>
#include <torch/nn/functional/linear.h>
#include <ATen/autocast_mode.h>
#include <ATen/CPUGeneratorImpl.h>
#include <ATen/ops/linalg_qr.h>
#include <ATen/ops/_scaled_mm.h>
#include <c10/core/Event.h>
#include <c10/cuda/CUDAStream.h>

namespace open_spiel {
namespace dune_semantic {

inline std::atomic<bool>& ForceLegacyV5ScorerFlag() {
  static std::atomic<bool> flag([]() {
    const char* env = std::getenv("DUNE_LEGACY_V5_SCORER");
    return env != nullptr && std::string(env) == "1";
  }());
  return flag;
}

inline void SetForceLegacyV5Scorer(bool v) {
  ForceLegacyV5ScorerFlag().store(v, std::memory_order_relaxed);
}

inline bool GetForceLegacyV5Scorer() {
  return ForceLegacyV5ScorerFlag().load(std::memory_order_relaxed);
}

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
    if (out_layer_f.is_empty() || out_layer_f->options.out_features() != out_dim) {
      auto new_f = torch::nn::Linear(kScorerHiddenDim, out_dim);
      if (card_embedding && card_embedding->weight.defined()) {
        new_f->to(card_embedding->weight.device());
      }
      if (out_layer_f.is_empty()) {
        out_layer_f = register_module("out_layer_f", new_f);
      } else {
        out_layer_f = replace_module("out_layer_f", new_f);
      }
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
            auto new_f = torch::nn::Linear(kScorerHiddenDim, ckpt_out_dim);
            if (card_embedding && card_embedding->weight.defined()) {
              new_f->to(card_embedding->weight.device());
            }
            out_layer_f = replace_module("out_layer_f", new_f);
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
      std::optional<bool> any_v5_hint = std::nullopt,
      const torch::Tensor& v5_idx_pre = torch::Tensor(),
      const torch::Tensor& v5_unique_states_pre = torch::Tensor(),
      const torch::Tensor& v5_state_slot_pre = torch::Tensor()) {
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

    const bool use_fast_v5 = v5_idx_pre.defined() && !GetForceLegacyV5Scorer();

    if (use_fast_v5) {
      const bool fast_any_v5 = (v5_idx_pre.numel() > 0);
      if (fast_any_v5 && (!enable_v5_ || !v5_query_proj)) {
        SpielFatalError(
            "SemanticActionScorer::ComputeAndAddCorrections: v5 action encountered "
            "while enable_v5_ is false; semantic_action_v5 scorer required.");
      }

      auto c_emb = card_embedding->forward(card_ids);     // [M, 32]
      auto s_emb = space_embedding->forward(space_ids);   // [M, 16]
      auto d_full = torch::cat({feat_tensor.to(c_emb.dtype()), c_emb, s_emb}, -1); // [M, 104]

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

      if (fast_any_v5 && torch::GradMode::is_enabled()) {
        auto is_v5_mask = (v5_families >= 0);
        raw_corr = torch::where(is_v5_mask, raw_corr.detach(), raw_corr);
      }

      auto corr = (raw_corr * supported_mask.to(raw_corr.dtype())).to(inout_logits.dtype());      // [M]

      if (fast_any_v5) {
        const int64_t m_v5 = v5_idx_pre.numel();
        auto item_v5 = v5_item_ids.index_select(0, v5_idx_pre);
        auto fam_v5 = v5_families.index_select(0, v5_idx_pre);
        auto ctx_v5 = v5_ctx_feat.to(h_p.dtype()).index_select(0, v5_idx_pre);
        auto h_u = h_p.index_select(0, v5_unique_states_pre);  // [U, 256]
        auto q_u = v5_query_proj_bf16_.Forward(v5_query_proj, h_u, use_bf16_cache_);  // [U, 1440]
        auto q_all = q_u.index_select(0, v5_state_slot_pre).view({m_v5, kV5NumFamilies, kV5EmbedDim});

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

        auto c_v5 = c_emb.index_select(0, v5_idx_pre);
        if (torch::GradMode::is_enabled()) {
          c_v5 = c_v5.detach();
        }
        auto c_v5_masked = c_v5 * (item_v5 < 256).unsqueeze(-1).to(c_v5.dtype());

        auto k_item = e_id +
                      v5_card_adapter_bf16_.Forward(v5_card_adapter, c_v5_masked, use_bf16_cache_) +
                      v5_ctx_proj_bf16_.Forward(v5_ctx_proj, ctx_v5, use_bf16_cache_);

        auto comp_dtype = (raw_corr.dtype() == torch::kFloat64) ? torch::kFloat64 : torch::kFloat32;
        auto v5_delta = (q_sel.to(comp_dtype) * k_item.to(comp_dtype)).sum(-1);

        auto base_corr = (raw_corr * supported_mask.to(raw_corr.dtype())).to(comp_dtype);
        base_corr = base_corr.index_put({v5_idx_pre}, base_corr.index_select(0, v5_idx_pre) + v5_delta);
        corr = base_corr.to(inout_logits.dtype());
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
      return;
    }

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
    auto d_full = torch::cat({feat_tensor.to(c_emb.dtype()), c_emb, s_emb}, -1); // [M, 104]

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

    auto corr = (raw_corr * supported_mask.to(raw_corr.dtype())).to(inout_logits.dtype());      // [M]

    if (any_v5) {
      auto v5_idx = torch::nonzero(is_v5_mask).squeeze(-1);
      if (v5_idx.numel() > 0) {
        auto b_v5 = batch_indices.index_select(0, v5_idx);
        auto item_v5 = v5_item_ids.index_select(0, v5_idx);
        auto fam_v5 = v5_families.index_select(0, v5_idx);
        auto ctx_v5 = v5_ctx_feat.to(h_p.dtype()).index_select(0, v5_idx);
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

        auto comp_dtype = (raw_corr.dtype() == torch::kFloat64) ? torch::kFloat64 : torch::kFloat32;
        auto v5_delta = (q_sel.to(comp_dtype) * k_item.to(comp_dtype)).sum(-1);

        auto base_corr = (raw_corr * supported_mask.to(raw_corr.dtype())).to(comp_dtype);
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

} // namespace dune_semantic
} // namespace open_spiel

#endif // OPEN_SPIEL_BUILD_WITH_LIBTORCH

#endif // OPEN_SPIEL_EXAMPLES_DUNE_SEMANTIC_ACTION_SCORER_NET_H_
