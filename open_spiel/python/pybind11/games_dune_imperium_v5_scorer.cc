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

#ifdef OPEN_SPIEL_BUILD_WITH_LIBTORCH

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <memory>
#include <string>
#include <vector>

#include <c10/cuda/CUDAStream.h>
#include <torch/torch.h>

#include "dune_evaluator.h"
#include "open_spiel/games/dune_imperium/dune_imperium.h"
#include "open_spiel/spiel.h"

namespace py = ::pybind11;

namespace open_spiel {
namespace dune_imperium_pybind {

#pragma GCC optimize("no-var-tracking-assignments")
void RegisterDuneImperiumV5Scorer(py::module_ &di) {
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
}

}  // namespace dune_imperium_pybind
}  // namespace open_spiel

#endif
