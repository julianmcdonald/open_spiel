#ifndef OPEN_SPIEL_EXAMPLES_DUNE_SEMANTIC_ACTION_SCORER_BATCH_H_
#define OPEN_SPIEL_EXAMPLES_DUNE_SEMANTIC_ACTION_SCORER_BATCH_H_

#include "dune_semantic_action_scorer_net.h"

#ifdef OPEN_SPIEL_BUILD_WITH_LIBTORCH
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <memory>
#include <vector>

#include <c10/core/Event.h>
#include <c10/cuda/CUDAStream.h>

namespace open_spiel {
namespace dune_semantic {

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

  std::vector<int32_t> v5_state_to_slot;
  torch::Tensor v5_idx_pinned;
  torch::Tensor v5_unique_states_pinned;
  torch::Tensor v5_state_slot_pinned;
  torch::Tensor v5_idx_dev_buf;
  torch::Tensor v5_unique_states_dev_buf;
  torch::Tensor v5_state_slot_dev_buf;

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

    v5_idx_pinned = torch::empty({alloc_m}, host_opts.dtype(torch::kInt64));
    v5_unique_states_pinned = torch::empty({alloc_m}, host_opts.dtype(torch::kInt64));
    v5_state_slot_pinned = torch::empty({alloc_m}, host_opts.dtype(torch::kInt64));

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

      v5_idx_dev_buf = torch::empty({alloc_m}, dev_opts.dtype(torch::kInt64));
      v5_unique_states_dev_buf = torch::empty({alloc_m}, dev_opts.dtype(torch::kInt64));
      v5_state_slot_dev_buf = torch::empty({alloc_m}, dev_opts.dtype(torch::kInt64));
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
    if (on_cuda) {
      struct StreamWsEntry {
        c10::DeviceIndex dev_idx;
        c10::StreamId stream_id;
        std::unique_ptr<SemanticScorerBatchWorkspace> ws;
      };
      thread_local std::vector<StreamWsEntry> tls_cuda_workspaces;
      c10::DeviceIndex dev_index = device.has_index() ? device.index() : c10::cuda::current_device();
      c10::cuda::CUDAStream cur_stream = c10::cuda::getCurrentCUDAStream(dev_index);
      c10::DeviceIndex d_idx = cur_stream.device_index();
      c10::StreamId s_id = cur_stream.id();
      for (auto& entry : tls_cuda_workspaces) {
        if (entry.dev_idx == d_idx && entry.stream_id == s_id) {
          active_ws = entry.ws.get();
          break;
        }
      }
      if (active_ws == nullptr) {
        tls_cuda_workspaces.push_back({d_idx, s_id, std::make_unique<SemanticScorerBatchWorkspace>()});
        active_ws = tls_cuda_workspaces.back().ws.get();
      }
    } else {
      thread_local SemanticScorerBatchWorkspace tls_cpu_ws;
      active_ws = &tls_cpu_ws;
    }
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

  const bool use_fast_v5 = !GetForceLegacyV5Scorer();
  torch::Tensor v5_idx_pinned, v5_unique_states_pinned, v5_state_slot_pinned;
  int64_t* v5_idx_ptr = nullptr;
  int64_t* v5_unique_ptr = nullptr;
  int64_t* v5_slot_ptr = nullptr;
  std::vector<int32_t>* state_to_slot_ptr = nullptr;
  thread_local std::vector<int32_t> tls_state_to_slot;

  if (use_fast_v5) {
    if (active_ws != nullptr) {
      v5_idx_pinned = active_ws->v5_idx_pinned.narrow(0, 0, m);
      v5_unique_states_pinned = active_ws->v5_unique_states_pinned.narrow(0, 0, m);
      v5_state_slot_pinned = active_ws->v5_state_slot_pinned.narrow(0, 0, m);
      state_to_slot_ptr = &active_ws->v5_state_to_slot;
    } else {
      auto host_opts = torch::TensorOptions().device(torch::kCPU).pinned_memory(on_cuda);
      v5_idx_pinned = torch::empty({m}, host_opts.dtype(torch::kInt64));
      v5_unique_states_pinned = torch::empty({m}, host_opts.dtype(torch::kInt64));
      v5_state_slot_pinned = torch::empty({m}, host_opts.dtype(torch::kInt64));
      state_to_slot_ptr = &tls_state_to_slot;
    }
    v5_idx_ptr = v5_idx_pinned.data_ptr<int64_t>();
    v5_unique_ptr = v5_unique_states_pinned.data_ptr<int64_t>();
    v5_slot_ptr = v5_state_slot_pinned.data_ptr<int64_t>();

    if (state_to_slot_ptr->size() < batch_actions.size()) {
      state_to_slot_ptr->resize(batch_actions.size(), -1);
    }
  }

  int64_t m_v5 = 0;
  int64_t u_v5 = 0;
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
      if (v5_fam >= 0) {
        any_v5 = true;
        if (use_fast_v5) {
          int32_t slot_u = (*state_to_slot_ptr)[b];
          if (slot_u < 0) {
            slot_u = static_cast<int32_t>(u_v5);
            (*state_to_slot_ptr)[b] = slot_u;
            v5_unique_ptr[u_v5++] = static_cast<int64_t>(b);
          }
          v5_idx_ptr[m_v5] = static_cast<int64_t>(idx);
          v5_slot_ptr[m_v5] = static_cast<int64_t>(slot_u);
          ++m_v5;
        }
      }
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

  if (use_fast_v5) {
    for (int64_t ui = 0; ui < u_v5; ++ui) {
      (*state_to_slot_ptr)[static_cast<size_t>(v5_unique_ptr[ui])] = -1;
    }
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

  torch::Tensor v5_idx_t, v5_unique_t, v5_slot_t;
  if (use_fast_v5) {
    if (m_v5 > 0) {
      auto v5_idx_sub = v5_idx_pinned.narrow(0, 0, m_v5);
      auto v5_unique_sub = v5_unique_states_pinned.narrow(0, 0, u_v5);
      auto v5_slot_sub = v5_state_slot_pinned.narrow(0, 0, m_v5);
      if (on_cuda && active_ws != nullptr) {
        v5_idx_t = active_ws->v5_idx_dev_buf.narrow(0, 0, m_v5);
        v5_unique_t = active_ws->v5_unique_states_dev_buf.narrow(0, 0, u_v5);
        v5_slot_t = active_ws->v5_state_slot_dev_buf.narrow(0, 0, m_v5);
        v5_idx_t.copy_(v5_idx_sub, /*non_blocking=*/true);
        v5_unique_t.copy_(v5_unique_sub, /*non_blocking=*/true);
        v5_slot_t.copy_(v5_slot_sub, /*non_blocking=*/true);
      } else {
        v5_idx_t = on_cuda ? v5_idx_sub.to(device, /*non_blocking=*/true) : v5_idx_sub;
        v5_unique_t = on_cuda ? v5_unique_sub.to(device, /*non_blocking=*/true) : v5_unique_sub;
        v5_slot_t = on_cuda ? v5_slot_sub.to(device, /*non_blocking=*/true) : v5_slot_sub;
      }
    } else {
      v5_idx_t = b_t.narrow(0, 0, 0);
      v5_unique_t = b_t.narrow(0, 0, 0);
      v5_slot_t = b_t.narrow(0, 0, 0);
    }
  }

  scorer->ComputeAndAddCorrections(
      trunk, b_t, a_t, feat_t, c_t, s_t, supp_t, inout_logits,
      is_ext_t, is_f_t, allow_inplace, any_ext, any_f, precomputed_trunk_proj,
      v5_item_t, v5_fam_t, v5_ctx_t, any_v5,
      v5_idx_t, v5_unique_t, v5_slot_t);
}

} // namespace dune_semantic
} // namespace open_spiel

#endif // OPEN_SPIEL_BUILD_WITH_LIBTORCH

#endif // OPEN_SPIEL_EXAMPLES_DUNE_SEMANTIC_ACTION_SCORER_BATCH_H_
