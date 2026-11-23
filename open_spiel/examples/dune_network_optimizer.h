#ifndef OPEN_SPIEL_EXAMPLES_DUNE_NETWORK_OPTIMIZER_H_
#define OPEN_SPIEL_EXAMPLES_DUNE_NETWORK_OPTIMIZER_H_

#include <torch/torch.h>

#include <algorithm>
#include <cstdint>
#include <exception>
#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "dune_network_model.h"
#include "open_spiel/spiel.h"

namespace open_spiel {

inline std::unique_ptr<torch::optim::AdamW> MakeDuneOptimizer(
    std::shared_ptr<SharedDunePolicyValueNetImpl> model,
    double lr = 2.5e-4, double weight_decay = 0.0,
    double policy_weight_decay = 0.0, double new_block_lr = 0.0,
    int new_block_first_idx = 8) {
  std::vector<torch::Tensor> policy_params;
  std::vector<torch::Tensor> other_params;
  std::vector<torch::Tensor> aux_head_params;
  std::vector<torch::Tensor> new_block_params;

  std::vector<torch::Tensor> new_block_params_set;
  if (new_block_lr > 0.0) {
    for (int b = new_block_first_idx; b < static_cast<int>(model->res_blocks.size()); ++b) {
      for (auto& p : model->res_blocks[b]->parameters()) {
        new_block_params_set.push_back(p);
      }
    }
  }

  std::vector<torch::Tensor> f_params_set;
  if (model->semantic_scorer_ && model->semantic_scorer_->out_layer_f) {
    for (auto& p : model->semantic_scorer_->out_layer_f->parameters()) {
      f_params_set.push_back(p);
    }
  }
  if (model->semantic_scorer_ && model->semantic_scorer_->enable_v5_) {
    if (model->semantic_scorer_->v5_item_emb) {
      for (auto& p : model->semantic_scorer_->v5_item_emb->parameters()) {
        f_params_set.push_back(p);
      }
    }
    if (model->semantic_scorer_->v5_card_adapter) {
      for (auto& p : model->semantic_scorer_->v5_card_adapter->parameters()) {
        f_params_set.push_back(p);
      }
    }
    if (model->semantic_scorer_->v5_ctx_proj) {
      for (auto& p : model->semantic_scorer_->v5_ctx_proj->parameters()) {
        f_params_set.push_back(p);
      }
    }
    if (model->semantic_scorer_->v5_query_proj) {
      for (auto& p : model->semantic_scorer_->v5_query_proj->parameters()) {
        f_params_set.push_back(p);
      }
    }
  }
  if (model->enable_zone_encoder_ && model->zone_encoder_) {
    for (auto& p : model->zone_encoder_->parameters()) {
      f_params_set.push_back(p);
    }
  }

  auto policy_params_set = model->policy_head->parameters();
  std::vector<torch::Tensor> aux_params_set;
  if (model->has_aux_heads_) {
    for (auto& p : model->final_vp_head->parameters()) aux_params_set.push_back(p);
    for (auto& p : model->terminal_round_head->parameters()) aux_params_set.push_back(p);
    for (auto& p : model->next_own_action_head->parameters()) aux_params_set.push_back(p);
  }
  for (auto& param : model->parameters()) {
    bool is_policy = false;
    for (auto& policy_param : policy_params_set) {
      if (param.is_same(policy_param)) { is_policy = true; break; }
    }
    bool is_aux = false;
    for (auto& aux_param : aux_params_set) {
      if (param.is_same(aux_param)) { is_aux = true; break; }
    }
    bool is_new_block = false;
    for (auto& nb_param : new_block_params_set) {
      if (param.is_same(nb_param)) { is_new_block = true; break; }
    }
    bool is_f = false;
    for (auto& f_param : f_params_set) {
      if (param.is_same(f_param)) { is_f = true; break; }
    }
    if (is_policy) policy_params.push_back(param);
    else if (is_aux) aux_head_params.push_back(param);
    else if (is_new_block) new_block_params.push_back(param);
    else if (is_f) { /* Handled below: appended to higher-rate group */ }
    else other_params.push_back(param);
  }

  if (new_block_lr > 0.0) {
    for (auto& p : f_params_set) {
      new_block_params.push_back(p);
    }
  } else {
    for (auto& p : f_params_set) {
      other_params.push_back(p);
    }
  }

  std::vector<torch::optim::OptimizerParamGroup> groups;
  groups.emplace_back(policy_params);
  groups.emplace_back(other_params);
  if (model->has_aux_heads_) {
    groups.emplace_back(aux_head_params);
  }
  if (new_block_lr > 0.0) {
    groups.emplace_back(new_block_params);
  }
  auto optimizer = std::make_unique<torch::optim::AdamW>(
      groups, torch::optim::AdamWOptions(lr).eps(1e-5));
  static_cast<torch::optim::AdamWOptions&>(
      optimizer->param_groups()[0].options()).weight_decay(policy_weight_decay);
  static_cast<torch::optim::AdamWOptions&>(
      optimizer->param_groups()[1].options()).weight_decay(weight_decay);
  if (model->has_aux_heads_) {
    static_cast<torch::optim::AdamWOptions&>(
        optimizer->param_groups()[2].options()).weight_decay(0.0);
  }
  if (new_block_lr > 0.0) {
    const size_t nb_idx = optimizer->param_groups().size() - 1;
    auto& nb_options = static_cast<torch::optim::AdamWOptions&>(
        optimizer->param_groups()[nb_idx].options());
    nb_options.lr(new_block_lr);
    nb_options.weight_decay(0.0);
  }
  return optimizer;
}

inline std::unique_ptr<torch::optim::AdamW> BuildAdamWOptimizer(
    SharedDunePolicyValueNetImpl& model,
    double lr = 2.5e-4, double weight_decay = 0.0,
    double policy_weight_decay = 0.0, double new_block_lr = 0.0,
    int new_block_first_idx = 8) {
  auto shared_ptr = std::shared_ptr<SharedDunePolicyValueNetImpl>(&model, [](SharedDunePolicyValueNetImpl*){});
  return MakeDuneOptimizer(shared_ptr, lr, weight_decay, policy_weight_decay, new_block_lr, new_block_first_idx);
}

inline std::unique_ptr<torch::optim::AdamW> BuildAdamWOptimizer(
    std::shared_ptr<SharedDunePolicyValueNetImpl> model,
    double lr = 2.5e-4, double weight_decay = 0.0,
    double policy_weight_decay = 0.0, double new_block_lr = 0.0,
    int new_block_first_idx = 8) {
  return MakeDuneOptimizer(model, lr, weight_decay, policy_weight_decay, new_block_lr, new_block_first_idx);
}

inline void ZeroDynamicSlotPolicyHeadMoments(
    torch::optim::AdamW& optimizer,
    const std::shared_ptr<SharedDunePolicyValueNetImpl>& model) {
  if (!model || !model->policy_head) return;
  const bool is_v5 =
      (model->semantic_descriptor_schema_ == dune_semantic::kDescriptorSchemaVersionV5) ||
      (model->semantic_scorer_ && model->semantic_scorer_->enable_v5_);
  if (!is_v5) return;

  torch::NoGradGuard no_grad;
  static const std::vector<int64_t> kDynamicSlots = {
      10, 11, 12, 13, 14, 70, 71, 72, 91, 92, 538, 539, 540, 541, 542,
      546, 547, 548, 549, 550, 554, 555, 556, 572, 770, 771, 772};
  auto zero_dynamic_moments = [&](const torch::Tensor& param) {
    auto it = optimizer.state().find(param.unsafeGetTensorImpl());
    if (it != optimizer.state().end()) {
      auto* adam_state = dynamic_cast<torch::optim::AdamWParamState*>(it->second.get());
      if (adam_state != nullptr) {
        for (int64_t row : kDynamicSlots) {
          if (adam_state->exp_avg().defined() && row < adam_state->exp_avg().size(0)) {
            adam_state->exp_avg()[row].zero_();
          }
          if (adam_state->exp_avg_sq().defined() && row < adam_state->exp_avg_sq().size(0)) {
            adam_state->exp_avg_sq()[row].zero_();
          }
          if (adam_state->max_exp_avg_sq().defined() && row < adam_state->max_exp_avg_sq().size(0)) {
            adam_state->max_exp_avg_sq()[row].zero_();
          }
        }
      }
    }
  };
  zero_dynamic_moments(model->policy_head->weight);
  if (model->policy_head->bias.defined()) {
    zero_dynamic_moments(model->policy_head->bias);
  }
}

namespace optimizer_migration_internal {

inline bool TryMigrateSameGroupsAcrossInputDim(
    const std::shared_ptr<SharedDunePolicyValueNetImpl>& model,
    torch::optim::AdamW& optimizer,
    const std::string& path,
    torch::Device device,
    int64_t ckpt_input_dim,
    int64_t target_input_dim,
    bool requires_width_or_block_migration) {
  const int64_t target_hidden_dim = model->input_layer->weight.size(0);
  std::vector<torch::optim::OptimizerParamGroup> src_groups;
  for (size_t g = 0; g < optimizer.param_groups().size(); ++g) {
    std::vector<torch::Tensor> group_params;
    for (size_t i = 0; i < optimizer.param_groups()[g].params().size(); ++i) {
      const auto& p = optimizer.param_groups()[g].params()[i];
      if (p.unsafeGetTensorImpl() == model->input_layer->weight.unsafeGetTensorImpl()) {
        group_params.push_back(torch::zeros(
            {target_hidden_dim, ckpt_input_dim},
            torch::TensorOptions().dtype(p.dtype()).device(device).requires_grad(true)));
      } else {
        group_params.push_back(torch::zeros_like(
            p, torch::TensorOptions().dtype(p.dtype()).device(device).requires_grad(true)));
      }
    }
    src_groups.emplace_back(group_params, std::make_unique<torch::optim::AdamWOptions>(1e-4));
  }

  torch::optim::AdamW src_optimizer(src_groups, torch::optim::AdamWOptions(1e-4));
  try {
    torch::load(src_optimizer, path, device);
  } catch (const std::exception& e) {
    if (requires_width_or_block_migration) {
      SpielFatalError(absl::StrCat(
          "LoadOptimizerCheckpointMigrating: Failed to load source optimizer for input_dim migration (",
          ckpt_input_dim, " -> ", target_input_dim, "): ", e.what()));
    }
    return false;
  }

  optimizer.state().clear();
  for (size_t g = 0; g < optimizer.param_groups().size(); ++g) {
    for (size_t i = 0; i < optimizer.param_groups()[g].params().size(); ++i) {
      const auto& src_p = src_optimizer.param_groups()[g].params()[i];
      const auto& dst_p = optimizer.param_groups()[g].params()[i];
      auto src_key = src_p.unsafeGetTensorImpl();
      auto state_it = src_optimizer.state().find(src_key);
      if (state_it == src_optimizer.state().end()) {
        continue;
      }
      if (dst_p.unsafeGetTensorImpl() == model->input_layer->weight.unsafeGetTensorImpl()) {
        auto* src_adam_state = dynamic_cast<torch::optim::AdamWParamState*>(state_it->second.get());
        if (src_adam_state) {
          auto new_state = std::make_unique<torch::optim::AdamWParamState>();
          new_state->step(src_adam_state->step());
          if (src_adam_state->exp_avg().defined()) {
            torch::Tensor new_exp_avg = torch::zeros(
                {target_hidden_dim, target_input_dim}, src_adam_state->exp_avg().options());
            const int64_t copy_cols = std::min(ckpt_input_dim, target_input_dim);
            new_exp_avg.slice(1, 0, copy_cols).copy_(src_adam_state->exp_avg().slice(1, 0, copy_cols));
            new_state->exp_avg(new_exp_avg);
          }
          if (src_adam_state->exp_avg_sq().defined()) {
            torch::Tensor new_exp_avg_sq = torch::zeros(
                {target_hidden_dim, target_input_dim}, src_adam_state->exp_avg_sq().options());
            const int64_t copy_cols = std::min(ckpt_input_dim, target_input_dim);
            new_exp_avg_sq.slice(1, 0, copy_cols).copy_(src_adam_state->exp_avg_sq().slice(1, 0, copy_cols));
            new_state->exp_avg_sq(new_exp_avg_sq);
          }
          if (src_adam_state->max_exp_avg_sq().defined()) {
            torch::Tensor new_max = torch::zeros(
                {target_hidden_dim, target_input_dim}, src_adam_state->max_exp_avg_sq().options());
            const int64_t copy_cols = std::min(ckpt_input_dim, target_input_dim);
            new_max.slice(1, 0, copy_cols).copy_(src_adam_state->max_exp_avg_sq().slice(1, 0, copy_cols));
            new_state->max_exp_avg_sq(new_max);
          }
          optimizer.state()[dst_p.unsafeGetTensorImpl()] = std::move(new_state);
        }
      } else {
        optimizer.state()[dst_p.unsafeGetTensorImpl()] = std::move(state_it->second);
      }
    }
  }
  ZeroDynamicSlotPolicyHeadMoments(optimizer, model);
  std::cout << "[INFO] Migrated optimizer checkpoint across input_dim "
            << ckpt_input_dim << " -> " << target_input_dim
            << " (preserving all " << optimizer.param_groups().size()
            << " parameter groups).\n";
  return true;
}

inline bool TryMigrateV5Group1ToGroup2(
    const std::shared_ptr<SharedDunePolicyValueNetImpl>& model,
    torch::optim::AdamW& optimizer,
    const std::string& path,
    torch::Device device,
    double lr,
    int64_t ckpt_input_dim,
    int64_t target_input_dim,
    size_t total_ckpt_params,
    bool requires_width_or_block_migration,
    const std::vector<int64_t>& group_param_counts,
    const std::vector<torch::Tensor>& v5_params,
    size_t dst_g0_n,
    size_t dst_g1_n,
    size_t dst_g2_n) {
  for (size_t k = 0; k < v5_params.size(); ++k) {
    const auto& dst_p = optimizer.param_groups()[2].params()[dst_g2_n - v5_params.size() + k];
    if (!dst_p.is_same(v5_params[k])) {
      SpielFatalError(absl::StrCat(
          "LoadOptimizerCheckpointMigrating: destination group 2 param at index ",
          dst_g2_n - v5_params.size() + k, " does not match v5_params[", k, "]"));
    }
  }

  const int64_t target_hidden_dim = model->input_layer->weight.size(0);
  auto make_src_tensor = [&](const torch::Tensor& p) -> torch::Tensor {
    if (p.unsafeGetTensorImpl() == model->input_layer->weight.unsafeGetTensorImpl() &&
        ckpt_input_dim != target_input_dim) {
      return torch::zeros(
          {target_hidden_dim, ckpt_input_dim},
          torch::TensorOptions().dtype(p.dtype()).device(device).requires_grad(true));
    }
    return torch::zeros_like(
        p, torch::TensorOptions().dtype(p.dtype()).device(device).requires_grad(true));
  };

  std::vector<torch::Tensor> src_g0;
  src_g0.reserve(dst_g0_n);
  for (size_t i = 0; i < dst_g0_n; ++i) {
    src_g0.push_back(make_src_tensor(optimizer.param_groups()[0].params()[i]));
  }

  std::vector<torch::Tensor> src_g1;
  src_g1.reserve(dst_g1_n + v5_params.size());
  for (size_t i = 0; i < dst_g1_n; ++i) {
    src_g1.push_back(make_src_tensor(optimizer.param_groups()[1].params()[i]));
  }
  for (size_t k = 0; k < v5_params.size(); ++k) {
    src_g1.push_back(make_src_tensor(v5_params[k]));
  }

  std::vector<torch::Tensor> src_g2;
  const size_t src_g2_n = dst_g2_n - v5_params.size();
  src_g2.reserve(src_g2_n);
  for (size_t i = 0; i < src_g2_n; ++i) {
    src_g2.push_back(make_src_tensor(optimizer.param_groups()[2].params()[i]));
  }

  std::vector<torch::optim::OptimizerParamGroup> src_groups;
  src_groups.emplace_back(src_g0);
  src_groups.emplace_back(src_g1);
  src_groups.emplace_back(src_g2);

  torch::optim::AdamW src_optimizer(src_groups, torch::optim::AdamWOptions(lr).eps(1e-5));
  try {
    torch::load(src_optimizer, path, device);
  } catch (const std::exception& e) {
    if (requires_width_or_block_migration) {
      SpielFatalError(absl::StrCat(
          "LoadOptimizerCheckpointMigrating: Failed to load source optimizer for v5 Group 1 -> Group 2 migration: ",
          e.what()));
    }
    return false;
  }

  int moved = 0;
  optimizer.state().clear();

  auto move_param_state = [&](const torch::Tensor& src_t, const torch::Tensor& dst_t) {
    auto src_key = src_t.unsafeGetTensorImpl();
    auto state_it = src_optimizer.state().find(src_key);
    if (state_it == src_optimizer.state().end()) {
      return;
    }
    auto* src_adam_state = dynamic_cast<torch::optim::AdamWParamState*>(state_it->second.get());
    if (!src_adam_state) {
      optimizer.state()[dst_t.unsafeGetTensorImpl()] = std::move(state_it->second);
      ++moved;
      return;
    }
    if (dst_t.unsafeGetTensorImpl() == model->input_layer->weight.unsafeGetTensorImpl() &&
        ckpt_input_dim != target_input_dim) {
      auto new_state = std::make_unique<torch::optim::AdamWParamState>();
      new_state->step(src_adam_state->step());
      const int64_t copy_cols = std::min(ckpt_input_dim, target_input_dim);
      if (src_adam_state->exp_avg().defined()) {
        torch::Tensor new_exp_avg = torch::zeros(
            {target_hidden_dim, target_input_dim}, src_adam_state->exp_avg().options().device(device));
        new_exp_avg.slice(1, 0, copy_cols).copy_(src_adam_state->exp_avg().slice(1, 0, copy_cols));
        new_state->exp_avg(new_exp_avg);
      }
      if (src_adam_state->exp_avg_sq().defined()) {
        torch::Tensor new_exp_avg_sq = torch::zeros(
            {target_hidden_dim, target_input_dim}, src_adam_state->exp_avg_sq().options().device(device));
        new_exp_avg_sq.slice(1, 0, copy_cols).copy_(src_adam_state->exp_avg_sq().slice(1, 0, copy_cols));
        new_state->exp_avg_sq(new_exp_avg_sq);
      }
      if (src_adam_state->max_exp_avg_sq().defined()) {
        torch::Tensor new_max = torch::zeros(
            {target_hidden_dim, target_input_dim}, src_adam_state->max_exp_avg_sq().options().device(device));
        new_max.slice(1, 0, copy_cols).copy_(src_adam_state->max_exp_avg_sq().slice(1, 0, copy_cols));
        new_state->max_exp_avg_sq(new_max);
      }
      optimizer.state()[dst_t.unsafeGetTensorImpl()] = std::move(new_state);
      ++moved;
    } else {
      if (src_adam_state->exp_avg().defined() && src_adam_state->exp_avg().device() != device) {
        src_adam_state->exp_avg(src_adam_state->exp_avg().to(device));
      }
      if (src_adam_state->exp_avg_sq().defined() && src_adam_state->exp_avg_sq().device() != device) {
        src_adam_state->exp_avg_sq(src_adam_state->exp_avg_sq().to(device));
      }
      if (src_adam_state->max_exp_avg_sq().defined() && src_adam_state->max_exp_avg_sq().device() != device) {
        src_adam_state->max_exp_avg_sq(src_adam_state->max_exp_avg_sq().to(device));
      }
      optimizer.state()[dst_t.unsafeGetTensorImpl()] = std::move(state_it->second);
      ++moved;
    }
  };

  // Group 0:
  for (size_t i = 0; i < dst_g0_n; ++i) {
    move_param_state(src_g0[i], optimizer.param_groups()[0].params()[i]);
  }
  // Group 1:
  for (size_t i = 0; i < dst_g1_n; ++i) {
    move_param_state(src_g1[i], optimizer.param_groups()[1].params()[i]);
  }
  for (size_t k = 0; k < v5_params.size(); ++k) {
    move_param_state(src_g1[dst_g1_n + k], v5_params[k]);
  }
  // Group 2:
  for (size_t i = 0; i < src_g2_n; ++i) {
    move_param_state(src_g2[i], optimizer.param_groups()[2].params()[i]);
  }

  if (moved != static_cast<int>(total_ckpt_params)) {
    SpielFatalError(absl::StrCat(
        "LoadOptimizerCheckpointMigrating: expected to move ", total_ckpt_params,
        " parameter states in v5 Group 1 -> Group 2 migration, but moved ", moved));
  }

  ZeroDynamicSlotPolicyHeadMoments(optimizer, model);
  std::cout << "[MIGRATION] Migrated v5 scorer parameters from optimizer Group 1 to Group 2 ("
            << group_param_counts[0] << ", " << group_param_counts[1] << ", " << group_param_counts[2]
            << ") -> (" << dst_g0_n << ", " << dst_g1_n << ", " << dst_g2_n
            << ") preserving all " << moved << " AdamW states." << std::endl;
  return true;
}

inline bool TryMigrateU31400ThreeGroupLayout(
    const std::shared_ptr<SharedDunePolicyValueNetImpl>& model,
    torch::optim::AdamW& optimizer,
    const std::string& path,
    torch::Device device,
    double lr,
    int64_t ckpt_input_dim,
    int64_t target_input_dim,
    int dst_num_blocks,
    int64_t ckpt_g2_count,
    int64_t ckpt_f_params,
    bool requires_width_or_block_migration,
    torch::serialize::InputArchive& state_archive,
    const std::vector<std::string>& state_keys) {
  const bool keep_f_in_dst =
      (ckpt_f_params == 2 && !model->semantic_scorer_->enable_v5_ &&
       !model->semantic_scorer_->out_layer_f.is_empty());
  const int expected_moved = 2 + 86 + 8 * (dst_num_blocks - 8) + (keep_f_in_dst ? 2 : 0);
  std::cout << "[MIGRATION] Detected " << dst_num_blocks << "-block 3-group source layout (2, 86, "
            << ckpt_g2_count << " params, ckpt_f_params=" << ckpt_f_params
            << "). Building matching 3-group source optimizer.\n";
  std::vector<torch::Tensor> src_g0;
  src_g0.push_back(torch::zeros_like(model->policy_head->weight, device));
  if (model->policy_head->bias.defined()) {
    src_g0.push_back(torch::zeros_like(model->policy_head->bias, device));
  }

  std::vector<torch::Tensor> src_g1;
  // input_layer (2)
  src_g1.push_back(torch::zeros({model->input_layer->weight.size(0), ckpt_input_dim}, device));
  if (model->input_layer->bias.defined()) {
    src_g1.push_back(torch::zeros_like(model->input_layer->bias, device));
  }
  // res1..res8 (64)
  for (int b = 0; b < 8; ++b) {
    for (auto& p : model->res_blocks[b]->parameters()) {
      src_g1.push_back(torch::zeros_like(p, device));
    }
  }
  // value_head (2)
  for (auto& p : model->value_head->parameters()) {
    src_g1.push_back(torch::zeros_like(p, device));
  }
  // semantic_scorer (18 inherited params)
  auto add_scorer_params = [&](auto& mod) {
    for (auto& p : mod->parameters()) {
      src_g1.push_back(torch::zeros_like(p, device));
    }
  };
  if (model->semantic_scorer_) {
    add_scorer_params(model->semantic_scorer_->card_embedding);
    add_scorer_params(model->semantic_scorer_->space_embedding);
    add_scorer_params(model->semantic_scorer_->desc_proj);
    add_scorer_params(model->semantic_scorer_->desc_ln);
    add_scorer_params(model->semantic_scorer_->trunk_proj);
    add_scorer_params(model->semantic_scorer_->trunk_ln);
    add_scorer_params(model->semantic_scorer_->mlp1);
    add_scorer_params(model->semantic_scorer_->mlp1_ln);
    add_scorer_params(model->semantic_scorer_->out_layer);
    add_scorer_params(model->semantic_scorer_->out_layer_ext);
  }

  std::vector<torch::Tensor> src_g2;
  // res9..res<dst_num_blocks> (8 * (dst_num_blocks - 8))
  for (int b = 8; b < dst_num_blocks; ++b) {
    for (auto& p : model->res_blocks[b]->parameters()) {
      src_g2.push_back(torch::zeros_like(p, device));
    }
  }
  if (ckpt_f_params == 2) {
    int64_t ckpt_f_out_dim = 1;
    for (const auto& key : state_keys) {
      torch::serialize::InputArchive p_arch;
      if (state_archive.try_read(key, p_arch)) {
        torch::Tensor exp_avg;
        if (p_arch.try_read("exp_avg", exp_avg) &&
            exp_avg.dim() == 2 &&
            exp_avg.size(0) == 2 &&
            exp_avg.size(1) == dune_semantic::kScorerHiddenDim) {
          ckpt_f_out_dim = 2;
          break;
        }
      }
    }
    src_g2.push_back(torch::zeros({ckpt_f_out_dim, dune_semantic::kScorerHiddenDim}, device));
    src_g2.push_back(torch::zeros({ckpt_f_out_dim}, device));
  }

  std::vector<torch::optim::OptimizerParamGroup> src_groups;
  src_groups.emplace_back(src_g0);
  src_groups.emplace_back(src_g1);
  src_groups.emplace_back(src_g2);

  torch::optim::AdamW src_optimizer(src_groups, torch::optim::AdamWOptions(lr).eps(1e-5));
  try {
    torch::load(src_optimizer, path, device);
  } catch (const c10::Error& e) {
    std::cerr << "[MIGRATION ERROR] Source optimizer torch::load failed: " << e.msg() << std::endl;
    if (requires_width_or_block_migration) {
      SpielFatalError(absl::StrCat("LoadOptimizerCheckpointMigrating: source optimizer torch::load failed: ", e.msg()));
    }
    return false;
  }

  int moved = 0;
  auto move_state = [&](const torch::Tensor& src_t, const torch::Tensor& dst_t) {
    auto src_key = src_t.unsafeGetTensorImpl();
    auto it = src_optimizer.state().find(src_key);
    if (it != src_optimizer.state().end()) {
      optimizer.state()[dst_t.unsafeGetTensorImpl()] = std::move(it->second);
      ++moved;
      return true;
    }
    return false;
  };

  // Group 0: policy
  move_state(src_g0[0], model->policy_head->weight);
  if (model->policy_head->bias.defined() && src_g0.size() > 1) {
    move_state(src_g0[1], model->policy_head->bias);
  }

  // Group 1: torso
  auto move_input_layer_state = [&](const torch::Tensor& src_t, const torch::Tensor& dst_t) {
    auto src_key = src_t.unsafeGetTensorImpl();
    auto it = src_optimizer.state().find(src_key);
    if (it != src_optimizer.state().end()) {
      if (ckpt_input_dim != target_input_dim) {
        auto* src_adam = dynamic_cast<torch::optim::AdamWParamState*>(it->second.get());
        if (src_adam) {
          auto new_state = std::make_unique<torch::optim::AdamWParamState>();
          new_state->step(src_adam->step());
          const int64_t target_hidden = model->input_layer->weight.size(0);
          if (src_adam->exp_avg().defined()) {
            torch::Tensor new_exp_avg = torch::zeros(
                {target_hidden, target_input_dim}, src_adam->exp_avg().options());
            const int64_t copy_cols = std::min(ckpt_input_dim, target_input_dim);
            new_exp_avg.slice(1, 0, copy_cols).copy_(src_adam->exp_avg().slice(1, 0, copy_cols));
            new_state->exp_avg(new_exp_avg);
          }
          if (src_adam->exp_avg_sq().defined()) {
            torch::Tensor new_exp_avg_sq = torch::zeros(
                {target_hidden, target_input_dim}, src_adam->exp_avg_sq().options());
            const int64_t copy_cols = std::min(ckpt_input_dim, target_input_dim);
            new_exp_avg_sq.slice(1, 0, copy_cols).copy_(src_adam->exp_avg_sq().slice(1, 0, copy_cols));
            new_state->exp_avg_sq(new_exp_avg_sq);
          }
          if (src_adam->max_exp_avg_sq().defined()) {
            torch::Tensor new_max = torch::zeros(
                {target_hidden, target_input_dim}, src_adam->max_exp_avg_sq().options());
            const int64_t copy_cols = std::min(ckpt_input_dim, target_input_dim);
            new_max.slice(1, 0, copy_cols).copy_(src_adam->max_exp_avg_sq().slice(1, 0, copy_cols));
            new_state->max_exp_avg_sq(new_max);
          }
          optimizer.state()[dst_t.unsafeGetTensorImpl()] = std::move(new_state);
          ++moved;
          return true;
        }
      }
      optimizer.state()[dst_t.unsafeGetTensorImpl()] = std::move(it->second);
      ++moved;
      return true;
    }
    return false;
  };
  move_input_layer_state(src_g1[0], model->input_layer->weight);
  if (model->input_layer->bias.defined() && src_g1.size() > 1) {
    move_state(src_g1[1], model->input_layer->bias);
  }
  size_t g1_idx = 1 + (model->input_layer->bias.defined() ? 1 : 0);
  for (int b = 0; b < 8; ++b) {
    for (auto& p : model->res_blocks[b]->parameters()) {
      move_state(src_g1[g1_idx++], p);
    }
  }
  for (auto& p : model->value_head->parameters()) {
    move_state(src_g1[g1_idx++], p);
  }
  if (model->semantic_scorer_) {
    auto move_scorer_mod = [&](auto& mod) {
      for (auto& p : mod->parameters()) {
        move_state(src_g1[g1_idx++], p);
      }
    };
    move_scorer_mod(model->semantic_scorer_->card_embedding);
    move_scorer_mod(model->semantic_scorer_->space_embedding);
    move_scorer_mod(model->semantic_scorer_->desc_proj);
    move_scorer_mod(model->semantic_scorer_->desc_ln);
    move_scorer_mod(model->semantic_scorer_->trunk_proj);
    move_scorer_mod(model->semantic_scorer_->trunk_ln);
    move_scorer_mod(model->semantic_scorer_->mlp1);
    move_scorer_mod(model->semantic_scorer_->mlp1_ln);
    move_scorer_mod(model->semantic_scorer_->out_layer);
    move_scorer_mod(model->semantic_scorer_->out_layer_ext);
  }

  // Group 2: res9..res<dst_num_blocks> (and out_layer_f if kept in v4 dst; discarded when dst is v5)
  size_t g2_idx = 0;
  for (int b = 8; b < dst_num_blocks; ++b) {
    for (auto& p : model->res_blocks[b]->parameters()) {
      move_state(src_g2[g2_idx++], p);
    }
  }
  if (keep_f_in_dst) {
    for (auto& p : model->semantic_scorer_->out_layer_f->parameters()) {
      move_state(src_g2[g2_idx++], p);
    }
  }

  if (moved != expected_moved) {
    SpielFatalError(absl::StrCat(
        "LoadOptimizerCheckpointMigrating: expected ", expected_moved,
        " moved parameter states, but moved ", moved));
  }

  ZeroDynamicSlotPolicyHeadMoments(optimizer, model);

  // New parameters in Prototype F / v4 / v5 start with empty optimizer state as specified.
  std::cout << "[MIGRATION] Optimizer successfully migrated from " << path << ": moved "
            << moved << " parameter states (expected " << expected_moved
            << "), leaving new parameters with empty optimizer state." << std::endl;
  return true;
}

}  // namespace optimizer_migration_internal

inline bool LoadOptimizerCheckpointMigrating(
    std::shared_ptr<SharedDunePolicyValueNetImpl> model,
    torch::optim::AdamW& optimizer, const std::string& path,
    torch::Device device, double lr = 2.5e-4) {
  torch::serialize::InputArchive archive;
  try {
    archive.load_from(path, device);
  } catch (const c10::Error& e) {
    return false;
  }

  torch::serialize::InputArchive pg_archive;
  size_t ckpt_num_groups = 0;
  std::vector<int64_t> group_param_counts;
  if (archive.try_read("param_groups", pg_archive)) {
    while (true) {
      std::string g_name = c10::to_string(ckpt_num_groups);
      std::string g_alt = "param_groups/" + g_name;
      torch::serialize::InputArchive sub_g;
      if (pg_archive.try_read(g_alt, sub_g) || pg_archive.try_read(g_name, sub_g)) {
        torch::Tensor p_sz_tensor;
        int64_t p_sz = 0;
        if (sub_g.try_read("params/size", p_sz_tensor)) {
          p_sz = p_sz_tensor.item<int64_t>();
        }
        group_param_counts.push_back(p_sz);
        ++ckpt_num_groups;
      } else {
        break;
      }
    }
  }
  if (ckpt_num_groups == 0) {
    return false;
  }

  torch::serialize::InputArchive state_archive;
  if (!archive.try_read("state", state_archive)) {
    return false;
  }

  const auto state_keys = state_archive.keys();
  const size_t total_ckpt_params = state_keys.size();

  int64_t ckpt_input_dim = 0;
  const int64_t hidden_dim = model->input_layer->weight.size(0);
  const int64_t target_input_dim = model->input_layer->weight.size(1);

  for (const auto& key : state_keys) {
    torch::serialize::InputArchive p_arch;
    if (state_archive.try_read(key, p_arch)) {
      torch::Tensor exp_avg;
      if (p_arch.try_read("exp_avg", exp_avg)) {
        if (exp_avg.dim() == 2 && exp_avg.size(0) == hidden_dim) {
          int64_t cols = exp_avg.size(1);
          if ((cols == 5580 || cols == 6215 || cols == 6255 ||
               cols == dune_imperium::kFullPublicInformationStateSize ||
               cols == dune_imperium::kFullPublicInformationV4StateSize ||
               cols == dune_imperium::kFullPublicInformationV5StateSize ||
               cols == target_input_dim) &&
              exp_avg.size(0) == hidden_dim) {
            ckpt_input_dim = cols;
            break;
          }
        }
      }
    }
  }
  if (ckpt_input_dim == 0) {
    if (!state_keys.empty()) {
      return false;
    }
    ckpt_input_dim = target_input_dim;
  }

  const int target_blocks = static_cast<int>(model->res_blocks.size());
  int ckpt_blocks = target_blocks;
  if (ckpt_num_groups == 3 && group_param_counts.size() == 3) {
    ckpt_blocks = 8 + static_cast<int>(group_param_counts[2] / 8);
  } else if (group_param_counts.size() > 1) {
    int64_t ckpt_torso_params = group_param_counts[1];
    if (ckpt_torso_params == 86 || ckpt_torso_params == 84 || ckpt_torso_params == 68) {
      ckpt_blocks = 8;
    }
  }
  const bool requires_width_or_block_migration =
      (ckpt_input_dim != target_input_dim) || (ckpt_blocks != target_blocks);

  bool groups_match = (ckpt_num_groups == optimizer.param_groups().size());
  bool input_dim_matches = (ckpt_input_dim == target_input_dim);
  bool params_match = true;
  if (groups_match) {
    for (size_t g = 0; g < ckpt_num_groups; ++g) {
      if (g < group_param_counts.size() &&
          group_param_counts[g] != static_cast<int64_t>(optimizer.param_groups()[g].params().size())) {
        params_match = false;
        break;
      }
    }
  } else {
    params_match = false;
  }

  if (groups_match && input_dim_matches && params_match) {
    std::cout << "[INFO] Optimizer checkpoint layout matches target model ("
              << ckpt_num_groups << " groups, " << total_ckpt_params
              << " parameters). Taking direct torch::load path (no migration).\n";
    return false;
  }

  if (groups_match && params_match && !input_dim_matches) {
    return optimizer_migration_internal::TryMigrateSameGroupsAcrossInputDim(
        model, optimizer, path, device, ckpt_input_dim, target_input_dim,
        requires_width_or_block_migration);
  }

  const int dst_num_blocks = static_cast<int>(model->res_blocks.size());
  const bool use_nonlinear = model->use_nonlinear_value_head_;
  int64_t ckpt_torso_params = (group_param_counts.size() > 1) ? group_param_counts[1] : 0;

  // Collect v5 parameters if enabled
  std::vector<torch::Tensor> v5_params;
  if (model->semantic_scorer_ && model->semantic_scorer_->enable_v5_) {
    if (model->semantic_scorer_->v5_item_emb) {
      for (auto& p : model->semantic_scorer_->v5_item_emb->parameters()) {
        v5_params.push_back(p);
      }
    }
    if (model->semantic_scorer_->v5_card_adapter) {
      for (auto& p : model->semantic_scorer_->v5_card_adapter->parameters()) {
        v5_params.push_back(p);
      }
    }
    if (model->semantic_scorer_->v5_ctx_proj) {
      for (auto& p : model->semantic_scorer_->v5_ctx_proj->parameters()) {
        v5_params.push_back(p);
      }
    }
    if (model->semantic_scorer_->v5_query_proj) {
      for (auto& p : model->semantic_scorer_->v5_query_proj->parameters()) {
        v5_params.push_back(p);
      }
    }
  }

  // Detect (2, 91, 64) -> (2, 86, 69) migration where v5 parameters were legacy-placed in Group 1
  bool is_v5_group1_to_group2_migration = false;
  size_t dst_g0_n = 0;
  size_t dst_g1_n = 0;
  size_t dst_g2_n = 0;
  if (ckpt_num_groups == 3 && group_param_counts.size() == 3 &&
      optimizer.param_groups().size() == 3 && !v5_params.empty()) {
    dst_g0_n = optimizer.param_groups()[0].params().size();
    dst_g1_n = optimizer.param_groups()[1].params().size();
    dst_g2_n = optimizer.param_groups()[2].params().size();
    if (dst_g2_n >= v5_params.size() &&
        group_param_counts[0] == static_cast<int64_t>(dst_g0_n) &&
        group_param_counts[1] == static_cast<int64_t>(dst_g1_n + v5_params.size()) &&
        group_param_counts[2] == static_cast<int64_t>(dst_g2_n - v5_params.size())) {
      is_v5_group1_to_group2_migration = true;
    }
  }

  if (is_v5_group1_to_group2_migration) {
    return optimizer_migration_internal::TryMigrateV5Group1ToGroup2(
        model, optimizer, path, device, lr, ckpt_input_dim, target_input_dim,
        total_ckpt_params, requires_width_or_block_migration,
        group_param_counts, v5_params, dst_g0_n, dst_g1_n, dst_g2_n);
  }

  // Explicit 3-group layout migration for v3/v4 -> v4/v5 (or U31400 style):
  // Source layout has 3 groups: Group 0 = policy (2), Group 1 = torso (86),
  // Group 2 = res9..res<N> (8 * (dst_num_blocks - 8)) + ckpt_f_params (0 for v3, 2 for v4 out_layer_f).
  const size_t base_g2_src = (dst_num_blocks > 8) ? static_cast<size_t>(8 * (dst_num_blocks - 8)) : 0;
  const int64_t ckpt_g2_count = (group_param_counts.size() > 2) ? group_param_counts[2] : -1;
  const int64_t ckpt_f_params =
      (dst_num_blocks > 8 &&
       (ckpt_g2_count == static_cast<int64_t>(base_g2_src) ||
        ckpt_g2_count == static_cast<int64_t>(base_g2_src) + 2))
          ? (ckpt_g2_count - static_cast<int64_t>(base_g2_src))
          : -1;
  const bool is_u31400_layout = (ckpt_num_groups == 3 && group_param_counts.size() == 3 &&
                                 group_param_counts[0] == 2 && group_param_counts[1] == 86 &&
                                 dst_num_blocks > 8 && ckpt_f_params >= 0 &&
                                 model->semantic_scorer_ &&
                                 (model->semantic_scorer_->out_layer_f || model->semantic_scorer_->enable_v5_));
  if (is_u31400_layout) {
    return optimizer_migration_internal::TryMigrateU31400ThreeGroupLayout(
        model, optimizer, path, device, lr, ckpt_input_dim, target_input_dim,
        dst_num_blocks, ckpt_g2_count, ckpt_f_params,
        requires_width_or_block_migration, state_archive, state_keys);
  }

  // Explicit 8->12/16 migration path: build source layout from source block count
  int src_num_blocks = dst_num_blocks;
  if (ckpt_num_groups == 2 && dst_num_blocks != 8 &&
      (ckpt_torso_params == 86 || ckpt_torso_params == 84 || ckpt_torso_params == 68)) {
    src_num_blocks = 8;
  }

  const size_t base_torso_params = 2 + (src_num_blocks * 8) + (use_nonlinear ? 4 : 2);
  const size_t policy_params_count = model->policy_head->bias.defined() ? 2 : 1;
  const size_t aux_params_count = 6;

  bool source_has_aux = (ckpt_num_groups >= 3 && group_param_counts.size() > 2 &&
                          group_param_counts[2] == static_cast<int64_t>(aux_params_count));
  bool source_has_scorer = false;
  bool source_has_scorer_ext = false;

  if (ckpt_torso_params >= static_cast<int64_t>(base_torso_params + 18)) {
    source_has_scorer = true;
    source_has_scorer_ext = true;
  } else if (ckpt_torso_params >= static_cast<int64_t>(base_torso_params + 16)) {
    source_has_scorer = true;
    source_has_scorer_ext = false;
  } else {
    source_has_scorer = false;
    source_has_scorer_ext = false;
  }

  std::vector<torch::Tensor> src_policy_params;
  src_policy_params.push_back(torch::zeros_like(model->policy_head->weight, device));
  if (model->policy_head->bias.defined()) {
    src_policy_params.push_back(torch::zeros_like(model->policy_head->bias, device));
  }

  std::vector<torch::Tensor> src_other_params;
  torch::Tensor src_input_w = torch::zeros({model->input_layer->weight.size(0), ckpt_input_dim}, device);
  src_other_params.push_back(src_input_w);
  if (model->input_layer->bias.defined()) {
    src_other_params.push_back(torch::zeros_like(model->input_layer->bias, device));
  }

  for (int b = 0; b < src_num_blocks; ++b) {
    for (auto& p : model->res_blocks[b]->parameters()) {
      src_other_params.push_back(torch::zeros_like(p, device));
    }
  }

  for (auto& p : model->value_head->parameters()) {
    src_other_params.push_back(torch::zeros_like(p, device));
  }
  if (use_nonlinear && model->value_head2) {
    for (auto& p : model->value_head2->parameters()) {
      src_other_params.push_back(torch::zeros_like(p, device));
    }
  }

  if (source_has_scorer && model->with_semantic_scorer_ && model->semantic_scorer_) {
    auto add_module_params = [&](auto& mod) {
      for (auto& p : mod->parameters()) {
        src_other_params.push_back(torch::zeros_like(p, device));
      }
    };
    add_module_params(model->semantic_scorer_->card_embedding);
    add_module_params(model->semantic_scorer_->space_embedding);
    add_module_params(model->semantic_scorer_->desc_proj);
    add_module_params(model->semantic_scorer_->desc_ln);
    add_module_params(model->semantic_scorer_->trunk_proj);
    add_module_params(model->semantic_scorer_->trunk_ln);
    add_module_params(model->semantic_scorer_->mlp1);
    add_module_params(model->semantic_scorer_->mlp1_ln);
    add_module_params(model->semantic_scorer_->out_layer);
    if (source_has_scorer_ext) {
      for (auto& p : model->semantic_scorer_->out_layer_ext->parameters()) {
        src_other_params.push_back(torch::zeros_like(p, device));
      }
    }
  }

  std::vector<torch::Tensor> src_aux_params;
  if (source_has_aux) {
    const int64_t action_dim = model->policy_head->weight.size(0);
    src_aux_params = {
        torch::zeros({1, hidden_dim}, device),
        torch::zeros({1}, device),
        torch::zeros({3, hidden_dim}, device),
        torch::zeros({3}, device),
        torch::zeros({action_dim, hidden_dim}, device),
        torch::zeros({action_dim}, device)};
  }

  std::vector<torch::optim::OptimizerParamGroup> src_groups;
  src_groups.emplace_back(src_policy_params);
  src_groups.emplace_back(src_other_params);
  if (source_has_aux) {
    src_groups.emplace_back(src_aux_params);
  }

  torch::optim::AdamW src_optimizer(
      src_groups,
      torch::optim::AdamWOptions(lr).eps(1e-5));

  try {
    torch::load(src_optimizer, path, device);
  } catch (const c10::Error& e) {
    std::cerr << "[MIGRATION ERROR] Temporary source optimizer torch::load failed: " << e.msg() << std::endl;
    if (requires_width_or_block_migration) {
      SpielFatalError(absl::StrCat("LoadOptimizerCheckpointMigrating: Temporary source optimizer torch::load failed: ", e.msg()));
    }
    return false;
  }

  int moved = 0;
  auto move_state = [&](const torch::Tensor& src_t, const torch::Tensor& dst_t) {
    auto src_key = src_t.unsafeGetTensorImpl();
    auto it = src_optimizer.state().find(src_key);
    if (it != src_optimizer.state().end()) {
      optimizer.state()[dst_t.unsafeGetTensorImpl()] = std::move(it->second);
      ++moved;
      return true;
    }
    return false;
  };

  move_state(src_policy_params[0], model->policy_head->weight);
  if (model->policy_head->bias.defined() && src_policy_params.size() > 1) {
    move_state(src_policy_params[1], model->policy_head->bias);
  }

  {
    auto src_key = src_input_w.unsafeGetTensorImpl();
    auto it = src_optimizer.state().find(src_key);
    if (it != src_optimizer.state().end()) {
      auto* adam_state = dynamic_cast<torch::optim::AdamWParamState*>(it->second.get());
      if (adam_state && adam_state->exp_avg().defined()) {
        const int64_t old_cols = adam_state->exp_avg().size(1);
        const int64_t new_cols = model->input_layer->weight.size(1);
        if (old_cols < new_cols) {
          torch::Tensor new_exp_avg = torch::zeros(
              {adam_state->exp_avg().size(0), new_cols}, adam_state->exp_avg().options());
          new_exp_avg.slice(1, 0, old_cols).copy_(adam_state->exp_avg());
          adam_state->exp_avg() = new_exp_avg;

          torch::Tensor new_exp_avg_sq = torch::zeros(
              {adam_state->exp_avg_sq().size(0), new_cols}, adam_state->exp_avg_sq().options());
          new_exp_avg_sq.slice(1, 0, old_cols).copy_(adam_state->exp_avg_sq());
          adam_state->exp_avg_sq() = new_exp_avg_sq;
        }
      }
      optimizer.state()[model->input_layer->weight.unsafeGetTensorImpl()] = std::move(it->second);
      ++moved;
    }
  }
  if (model->input_layer->bias.defined() && src_other_params.size() > 1) {
    move_state(src_other_params[1], model->input_layer->bias);
  }

  size_t src_res_idx = 1 + (model->input_layer->bias.defined() ? 1 : 0);
  for (int b = 0; b < src_num_blocks; ++b) {
    for (auto& p : model->res_blocks[b]->parameters()) {
      move_state(src_other_params[src_res_idx++], p);
    }
  }

  for (auto& p : model->value_head->parameters()) {
    move_state(src_other_params[src_res_idx++], p);
  }
  if (use_nonlinear && model->value_head2) {
    for (auto& p : model->value_head2->parameters()) {
      move_state(src_other_params[src_res_idx++], p);
    }
  }

  if (source_has_scorer && model->with_semantic_scorer_ && model->semantic_scorer_) {
    auto move_module_params = [&](auto& mod) {
      for (auto& p : mod->parameters()) {
        move_state(src_other_params[src_res_idx++], p);
      }
    };
    move_module_params(model->semantic_scorer_->card_embedding);
    move_module_params(model->semantic_scorer_->space_embedding);
    move_module_params(model->semantic_scorer_->desc_proj);
    move_module_params(model->semantic_scorer_->desc_ln);
    move_module_params(model->semantic_scorer_->trunk_proj);
    move_module_params(model->semantic_scorer_->trunk_ln);
    move_module_params(model->semantic_scorer_->mlp1);
    move_module_params(model->semantic_scorer_->mlp1_ln);
    move_module_params(model->semantic_scorer_->out_layer);
    if (source_has_scorer_ext) {
      for (auto& p : model->semantic_scorer_->out_layer_ext->parameters()) {
        move_state(src_other_params[src_res_idx++], p);
      }
    }
  }

  if (source_has_aux && model->has_aux_heads_) {
    size_t aux_i = 0;
    for (auto& p : model->final_vp_head->parameters()) move_state(src_aux_params[aux_i++], p);
    for (auto& p : model->terminal_round_head->parameters()) move_state(src_aux_params[aux_i++], p);
    for (auto& p : model->next_own_action_head->parameters()) move_state(src_aux_params[aux_i++], p);
  }

  int initialized_new = 0;
  for (auto& group : optimizer.param_groups()) {
    for (auto& param : group.params()) {
      auto key = param.unsafeGetTensorImpl();
      if (optimizer.state().find(key) == optimizer.state().end()) {
        auto new_state = std::make_unique<torch::optim::AdamWParamState>();
        new_state->step(0);
        new_state->exp_avg(torch::zeros_like(param, torch::MemoryFormat::Preserve));
        new_state->exp_avg_sq(torch::zeros_like(param, torch::MemoryFormat::Preserve));
        optimizer.state()[key] = std::move(new_state);
        ++initialized_new;
      }
    }
  }

  ZeroDynamicSlotPolicyHeadMoments(optimizer, model);

  std::cout << "[MIGRATION] Optimizer migrated from " << path << ": moved " << moved
            << " parameter states, initialized " << initialized_new << " new parameter states." << std::endl;
  return true;
}

}  // namespace open_spiel

#endif  // OPEN_SPIEL_EXAMPLES_DUNE_NETWORK_OPTIMIZER_H_
