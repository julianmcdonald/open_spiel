#include "dune_surgery_12block_internal.h"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <memory>
#include <string>

#include "absl/strings/str_format.h"
#include "open_spiel/games/dune_imperium/dune_imperium.h"
#include "open_spiel/spiel_utils.h"

#include "dune_evaluator.h"
#include "dune_sha256.h"

namespace open_spiel::dune_surgery_12block {
namespace {

void CopyCommonModelLayers(
    const std::shared_ptr<open_spiel::SharedDunePolicyValueNetImpl>& src,
    const std::shared_ptr<open_spiel::SharedDunePolicyValueNetImpl>& dst,
    int src_blocks,
    const std::string& mask_mismatch_error) {
  // Input layer
  dst->input_layer->weight.copy_(src->input_layer->weight);
  if (src->input_layer->bias.defined()) {
    dst->input_layer->bias.copy_(src->input_layer->bias);
  }

  // Inherited residual blocks (0 .. src_blocks - 1)
  for (int b = 0; b < src_blocks; ++b) {
    dst->res_blocks[b]->fc1->weight.copy_(src->res_blocks[b]->fc1->weight);
    dst->res_blocks[b]->fc1->bias.copy_(src->res_blocks[b]->fc1->bias);
    dst->res_blocks[b]->fc2->weight.copy_(src->res_blocks[b]->fc2->weight);
    dst->res_blocks[b]->fc2->bias.copy_(src->res_blocks[b]->fc2->bias);
    dst->res_blocks[b]->ln1->weight.copy_(src->res_blocks[b]->ln1->weight);
    dst->res_blocks[b]->ln1->bias.copy_(src->res_blocks[b]->ln1->bias);
    dst->res_blocks[b]->ln2->weight.copy_(src->res_blocks[b]->ln2->weight);
    dst->res_blocks[b]->ln2->bias.copy_(src->res_blocks[b]->ln2->bias);
  }

  // Policy head
  dst->policy_head->weight.copy_(src->policy_head->weight);
  if (src->policy_head->bias.defined()) {
    dst->policy_head->bias.copy_(src->policy_head->bias);
  }

  // Value head
  dst->value_head->weight.copy_(src->value_head->weight);
  if (src->value_head->bias.defined()) {
    dst->value_head->bias.copy_(src->value_head->bias);
  }

  // Semantic scorer (all parameters and buffers)
  if (src->semantic_scorer_ && dst->semantic_scorer_) {
    for (const auto& pair : src->semantic_scorer_->named_parameters()) {
      dst->semantic_scorer_->named_parameters()[pair.key()].copy_(pair.value());
    }
    for (const auto& pair : src->semantic_scorer_->named_buffers()) {
      dst->semantic_scorer_->named_buffers()[pair.key()].copy_(pair.value());
    }
  }

  // Top-level buffers
  for (const auto& pair : src->named_buffers()) {
    if (dst->named_buffers().contains(pair.key())) {
      dst->named_buffers()[pair.key()].copy_(pair.value());
    }
  }
  if (src->v5_dynamic_slot_mask_.defined()) {
    dst->v5_dynamic_slot_mask_ = src->v5_dynamic_slot_mask_.clone();
    if (!torch::equal(src->v5_dynamic_slot_mask_, dst->v5_dynamic_slot_mask_)) {
      open_spiel::SpielFatalError(mask_mismatch_error);
    }
  }
}

}  // namespace

ManifestMetadata ParseManifestMetadata(const open_spiel::json::Object& obj) {
  ManifestMetadata m;
  auto it_obs = obj.find("observation_dim");
  if (it_obs != obj.end() && it_obs->second.IsInt()) {
    m.observation_dim = it_obs->second.GetInt();
  }

  auto it_mode = obj.find("market_appendix_mode");
  if (it_mode != obj.end() && it_mode->second.IsString()) {
    m.market_appendix_mode_str = it_mode->second.GetString();
    m.market_appendix_mode =
        open_spiel::dune_imperium::ParseMarketAppendixMode(m.market_appendix_mode_str);
  }

  auto it_dps = obj.find("deck_pool_semantics");
  if (it_dps != obj.end() && it_dps->second.IsString()) {
    m.deck_pool_semantics_str = it_dps->second.GetString();
    m.deck_pool_semantics =
        open_spiel::dune_imperium::ParseDeckPoolSemanticsOrDie(m.deck_pool_semantics_str);
  }

  auto it_sem = obj.find("semantic_descriptor_schema");
  if (it_sem != obj.end() && it_sem->second.IsString()) {
    m.semantic_descriptor_schema = it_sem->second.GetString();
  }

  auto it_blk = obj.find("num_blocks");
  if (it_blk != obj.end() && it_blk->second.IsInt()) {
    m.num_blocks = it_blk->second.GetInt();
  }

  auto it_up = obj.find("global_update");
  if (it_up != obj.end() && it_up->second.IsInt()) {
    m.global_update = it_up->second.GetInt();
  }

  auto it_sha = obj.find("feature_schema_sha256");
  if (it_sha != obj.end() && it_sha->second.IsString()) {
    m.feature_schema_sha256 = it_sha->second.GetString();
  }

  auto it_lbl = obj.find("feature_schema_label");
  if (it_lbl != obj.end() && it_lbl->second.IsString()) {
    m.feature_schema_label = it_lbl->second.GetString();
  }

  auto it_ramp = obj.find("rollout_amp");
  if (it_ramp != obj.end() && it_ramp->second.IsBool()) {
    m.rollout_amp = it_ramp->second.GetBool();
  }

  auto it_tf32 = obj.find("allow_tf32");
  if (it_tf32 != obj.end() && it_tf32->second.IsBool()) {
    m.allow_tf32 = it_tf32->second.GetBool();
  }

  auto it_tamp = obj.find("train_amp");
  if (it_tamp != obj.end() && it_tamp->second.IsBool()) {
    m.train_amp = it_tamp->second.GetBool();
  }

  auto it_spec = obj.find("specimen_exchange_penalty");
  if (it_spec != obj.end() && it_spec->second.IsDouble()) {
    m.specimen_exchange_penalty = it_spec->second.GetDouble();
  }

  auto it_rg = obj.find("rollout_games");
  if (it_rg != obj.end() && it_rg->second.IsInt()) {
    m.rollout_games = it_rg->second.GetInt();
  }

  auto it_pool = obj.find("training_opponent_pool");
  if (it_pool != obj.end() && it_pool->second.IsString()) {
    m.training_opponent_pool = it_pool->second.GetString();
  }

  auto it_pgf = obj.find("pool_game_fraction");
  if (it_pgf != obj.end() && it_pgf->second.IsDouble()) {
    m.pool_game_fraction = it_pgf->second.GetDouble();
  }

  auto it_popg = obj.find("pool_opponents_per_game");
  if (it_popg != obj.end() && it_popg->second.IsInt()) {
    m.pool_opponents_per_game = it_popg->second.GetInt();
  }

  auto it_fa = obj.find("family_atomics_penalty");
  if (it_fa != obj.end() && it_fa->second.IsDouble()) {
    m.family_atomics_penalty = it_fa->second.GetDouble();
  }

  auto it_pip = obj.find("plot_intrigue_penalty");
  if (it_pip != obj.end() && it_pip->second.IsDouble()) {
    m.plot_intrigue_penalty = it_pip->second.GetDouble();
  }

  auto it_piet = obj.find("plot_intrigue_exemption_threshold");
  if (it_piet != obj.end() && it_piet->second.IsInt()) {
    m.plot_intrigue_exemption_threshold = it_piet->second.GetInt();
  }

  auto it_mag = obj.find("ema_magnet_kl_coef");
  if (it_mag != obj.end() && it_mag->second.IsDouble()) {
    m.ema_magnet_kl_coef = it_mag->second.GetDouble();
  }

  auto it_fp = obj.find("config_fingerprint");
  if (it_fp != obj.end() && it_fp->second.IsString()) {
    m.config_fingerprint = it_fp->second.GetString();
  }

  return m;
}

ManifestMetadata LoadManifestForModel(const std::string& model_path,
                                      const std::string& fallback_manifest) {
  std::string json_path = "";
  if (model_path.size() >= 3 && model_path.substr(model_path.size() - 3) == ".pt") {
    std::string candidate = model_path.substr(0, model_path.size() - 3) + ".json";
    if (std::filesystem::exists(candidate)) {
      json_path = candidate;
    }
  }
  if (json_path.empty() && !fallback_manifest.empty() &&
      std::filesystem::exists(fallback_manifest)) {
    json_path = fallback_manifest;
  }
  if (!json_path.empty()) {
    std::ifstream jf(json_path);
    if (jf.is_open()) {
      std::string content((std::istreambuf_iterator<char>(jf)),
                          std::istreambuf_iterator<char>());
      auto parsed = open_spiel::json::FromString(content);
      if (parsed && parsed->IsObject()) {
        return ParseManifestMetadata(parsed->GetObject());
      }
    }
  }
  return ManifestMetadata();
}

std::shared_ptr<open_spiel::SharedDunePolicyValueNetImpl> MakeModel(
    int64_t obs_dim, int blocks, const std::string& sem_schema,
    open_spiel::dune_imperium::MarketAppendixMode mkt_mode,
    open_spiel::dune_imperium::DeckPoolSemantics semantics) {
  auto m = std::make_shared<open_spiel::SharedDunePolicyValueNetImpl>(
      obs_dim, 2048, 2391, blocks,
      /*use_nonlinear=*/false, /*with_aux_heads=*/false, /*head_init_seed=*/0,
      /*with_semantic_scorer=*/true, /*enable_zone_encoder=*/false, sem_schema);
  if (!sem_schema.empty()) {
    m->SetSemanticDescriptorSchema(sem_schema);
  }
  m->market_appendix_mode_ = mkt_mode;
  m->deck_pool_semantics_ = semantics;
  return m;
}

std::string BuildPrePrecisionConfigJsonString(
    int num_blocks,
    double base_lr,
    const ManifestMetadata& meta,
    const std::string& target_opponent_pool) {
  open_spiel::json::Object cfg;
  cfg["game"] = std::string("dune_imperium");
  cfg["ppo_minibatch_size"] = int64_t(2048);
  cfg["ppo_update_epochs"] = int64_t(4);
  cfg["learning_rate"] = base_lr;
  cfg["anneal_lr"] = false;
  cfg["gamma"] = 1.0;
  cfg["gae_lambda"] = 0.95;
  cfg["normalize_advantages"] = true;
  cfg["ppo_clip_epsilon"] = 0.2;
  cfg["ppo_clip_value_loss"] = true;
  cfg["entropy_coef"] = 0.015;
  cfg["value_coef"] = 0.5;
  cfg["grad_clip_norm"] = 0.5;
  cfg["target_kl"] = 0.005;
  cfg["weight_decay"] = 0.0;
  cfg["policy_weight_decay"] = 0.0;
  cfg["hidden_dim"] = int64_t(2048);
  cfg["num_blocks"] = int64_t(num_blocks);
  cfg["logit_cap"] = 10.0;
  cfg["shaped_reward_weight"] = 0.0;
  cfg["tleilaxu_breadcrumb_weight"] = 0.0;
  cfg["tleilaxu_level7_breadcrumb_weight"] = 0.0;
  cfg["specimen_exchange_penalty"] = meta.specimen_exchange_penalty;
  cfg["shaping_start_env_steps"] = int64_t(206830543);
  cfg["shaping_decay_env_steps"] = int64_t(0);
  cfg["reward_scale"] = 4.0;
  cfg["terminal_reward_mode"] = std::string("placement");
  cfg["round7_speed_bonus"] = 0.0;
  cfg["train_amp"] = meta.train_amp;
  cfg["pipeline"] = false;
  cfg["rollout_games"] = meta.rollout_games;
  cfg["rollout_transitions"] = int64_t(0);
  cfg["search_lambda"] = 0.5;
  cfg["search_minibatches_per_update"] = int64_t(2);
  cfg["search_minibatch_size"] = int64_t(512);
  cfg["online_search_collection"] = false;
  cfg["auxiliary_games"] = int64_t(16);
  cfg["auxiliary_search_seed_domain"] = int64_t(0);
  cfg["collector_dirichlet_epsilon"] = 0.25;
  cfg["search_loss_coef"] = 0.10;
  cfg["search_loss_warmup_update"] = int64_t(25);
  cfg["abort_grad_norm_ratio"] = 0.50;
  cfg["target_sharpen_exponent"] = 1.0;
  cfg["swordmaster_grant_fraction"] = 0.0;
  cfg["swordmaster_grant_round"] = int64_t(2);
  cfg["collector_acceptance_prior"] = std::string("raw_network_prior");
  cfg["deterministic"] = true;
  cfg["deterministic_rollout_eval"] = false;
  cfg["train_value_only"] = false;
  cfg["unfreeze_trunk"] = false;
  cfg["nonlinear_value_head"] = false;
  cfg["sample_counterfactual_states"] = false;
  cfg["counterfactual_samples_per_game"] = int64_t(4);
  cfg["counterfactual_replay_weight"] = int64_t(8);
  cfg["policy_kl_anchor_coeff"] = 0.0;

  std::string opp_pool =
      target_opponent_pool.empty() ? meta.training_opponent_pool : target_opponent_pool;
  if (meta.pool_game_fraction > 0.0 && meta.pool_opponents_per_game > 0) {
    cfg["pool_game_fraction"] = meta.pool_game_fraction;
    cfg["pool_opponents_per_game"] = meta.pool_opponents_per_game;
    cfg["training_opponent_pool"] = opp_pool;
  }

  if (meta.market_appendix_mode_str == "full_public_information_v5") {
    const auto schema = open_spiel::dune_imperium::GetInformationStateSchema(
        open_spiel::dune_imperium::MarketAppendixMode::kFullPublicInformationV5,
        meta.deck_pool_semantics);
    cfg["market_appendix_mode"] = std::string("full_public_information_v5");
    cfg["observation_dim"] = schema.size;
    cfg["feature_schema_sha256"] = std::string(schema.sha256);
  } else if (meta.market_appendix_mode_str == "full_public_information_v4") {
    const auto schema = open_spiel::dune_imperium::GetInformationStateSchema(
        open_spiel::dune_imperium::MarketAppendixMode::kFullPublicInformationV4,
        meta.deck_pool_semantics);
    cfg["market_appendix_mode"] = std::string("full_public_information_v4");
    cfg["observation_dim"] = schema.size;
    cfg["feature_schema_sha256"] = std::string(schema.sha256);
  } else if (meta.market_appendix_mode_str == "full_public_information_v3") {
    const auto schema = open_spiel::dune_imperium::GetInformationStateSchema(
        open_spiel::dune_imperium::MarketAppendixMode::kFullPublicInformationV3,
        meta.deck_pool_semantics);
    cfg["market_appendix_mode"] = std::string("full_public_information_v3");
    cfg["observation_dim"] = schema.size;
    cfg["feature_schema_sha256"] = std::string(schema.sha256);
  } else if (meta.market_appendix_mode_str == "ordered_card_slots_v2") {
    const auto schema = open_spiel::dune_imperium::GetInformationStateSchema(
        open_spiel::dune_imperium::MarketAppendixMode::kOrderedCardSlotsV2,
        meta.deck_pool_semantics);
    cfg["market_appendix_mode"] = std::string("ordered_card_slots_v2");
    cfg["observation_dim"] = schema.size;
    cfg["feature_schema_sha256"] = std::string(schema.sha256);
  }

  if (meta.deck_pool_semantics == open_spiel::dune_imperium::DeckPoolSemantics::kFixed) {
    cfg["deck_pool_semantics"] = std::string("fixed");
  }

  cfg["enable_semantic_scorer"] = true;
  cfg["semantic_descriptor_schema"] = meta.semantic_descriptor_schema;

  if (meta.family_atomics_penalty != 0.0) {
    cfg["family_atomics_penalty"] = meta.family_atomics_penalty;
  }
  if (meta.plot_intrigue_penalty != 0.0) {
    cfg["plot_intrigue_penalty"] = meta.plot_intrigue_penalty;
    cfg["plot_intrigue_exemption_threshold"] = meta.plot_intrigue_exemption_threshold;
  }
  if (meta.ema_magnet_kl_coef > 0.0) {
    cfg["ema_magnet_kl_coef"] = meta.ema_magnet_kl_coef;
  }

  // Precision fields
  cfg["rollout_amp"] = meta.rollout_amp;
  cfg["allow_tf32"] = meta.allow_tf32;

  return open_spiel::json::ToString(open_spiel::json::Value(cfg));
}

std::string ComputeManifestConfigFingerprint(
    int num_blocks,
    double base_lr,
    const ManifestMetadata& meta,
    const std::string& target_opponent_pool) {
  std::string json_str =
      BuildPrePrecisionConfigJsonString(num_blocks, base_lr, meta, target_opponent_pool);
  return open_spiel::ComputeStringSHA256(json_str);
}

void VerifyOptimizerGroups(
    const torch::optim::AdamW& opt_dst,
    int dst_blocks,
    int dst_nb_first_idx,
    double base_lr,
    double nb_lr,
    const std::string& semantic_descriptor_schema) {
  if (opt_dst.param_groups().size() != 3) {
    open_spiel::SpielFatalError("Expected 3 optimizer groups, found " +
                                std::to_string(opt_dst.param_groups().size()));
  }
  const auto& opt0 =
      static_cast<const torch::optim::AdamWOptions&>(opt_dst.param_groups()[0].options());
  const auto& opt1 =
      static_cast<const torch::optim::AdamWOptions&>(opt_dst.param_groups()[1].options());
  const auto& opt2 =
      static_cast<const torch::optim::AdamWOptions&>(opt_dst.param_groups()[2].options());

  std::cout << absl::StrFormat(
      "Group 0: lr=%.6e, weight_decay=%.6e, tensors=%zu  [policy_head.* (%zu)]\n",
      opt0.lr(), opt0.weight_decay(), opt_dst.param_groups()[0].params().size(),
      opt_dst.param_groups()[0].params().size());
  std::cout << absl::StrFormat(
      "Group 1: lr=%.6e, weight_decay=%.6e, tensors=%zu [input_layer.* (2), res1-res%d.* (%d), value_head.* (2), semantic_scorer.* (18)]\n",
      opt1.lr(), opt1.weight_decay(), opt_dst.param_groups()[1].params().size(),
      dst_nb_first_idx, dst_nb_first_idx * 8);
  std::cout << absl::StrFormat(
      "Group 2: lr=%.6e, weight_decay=%.6e, tensors=%zu [res%d-res%d.* (%d)%s]\n",
      opt2.lr(), opt2.weight_decay(), opt_dst.param_groups()[2].params().size(),
      dst_nb_first_idx + 1, dst_blocks, (dst_blocks - dst_nb_first_idx) * 8,
      (semantic_descriptor_schema == "semantic_action_v5" ? " + v5_params (5)" : ""));

  if (std::abs(opt0.lr() - base_lr) > 1e-12) {
    open_spiel::SpielFatalError("Group 0 lr mismatch!");
  }
  if (std::abs(opt1.lr() - base_lr) > 1e-12) {
    open_spiel::SpielFatalError("Group 1 lr mismatch!");
  }
  if (std::abs(opt2.lr() - nb_lr) > 1e-12) {
    open_spiel::SpielFatalError("Group 2 lr mismatch!");
  }

  const size_t expected_g0_size = 2;
  const size_t expected_g1_size = 2 + 8 * dst_nb_first_idx + 2 + 18;
  const size_t expected_g2_size =
      8 * (dst_blocks - dst_nb_first_idx) +
      (semantic_descriptor_schema == "semantic_action_v5" ? 5 : 0);

  if (opt_dst.param_groups()[0].params().size() != expected_g0_size) {
    open_spiel::SpielFatalError(absl::StrFormat(
        "Group 0 size mismatch: got %zu, expected %zu",
        opt_dst.param_groups()[0].params().size(), expected_g0_size));
  }
  if (opt_dst.param_groups()[1].params().size() != expected_g1_size) {
    open_spiel::SpielFatalError(absl::StrFormat(
        "Group 1 size mismatch: got %zu, expected %zu",
        opt_dst.param_groups()[1].params().size(), expected_g1_size));
  }
  if (opt_dst.param_groups()[2].params().size() != expected_g2_size) {
    open_spiel::SpielFatalError(absl::StrFormat(
        "Group 2 size mismatch: got %zu, expected %zu",
        opt_dst.param_groups()[2].params().size(), expected_g2_size));
  }
}

LiveSurgeryResult RunLiveModelSurgery(
    const std::string& src_m_path,
    const std::string& out_dir,
    int src_blocks,
    int dst_blocks,
    int64_t init_seed,
    int64_t global_update,
    const ManifestMetadata& manifest_meta,
    const torch::Device& device) {
  LiveSurgeryResult res;

  // 1. Build Source Model and Load
  std::cout << "\n[B1 SURGERY] Loading source " << src_blocks << "-block model...\n";
  res.model_src = MakeModel(manifest_meta.observation_dim, src_blocks,
                            manifest_meta.semantic_descriptor_schema,
                            manifest_meta.market_appendix_mode,
                            manifest_meta.deck_pool_semantics);
  open_spiel::LoadModelCheckpointRobust(res.model_src, src_m_path, device);
  std::cout << "Source " << src_blocks << "-block model loaded successfully.\n";

  // 2. Build Destination Model with deterministic seed
  std::cout << "[B1 SURGERY] Constructing destination " << dst_blocks
            << "-block model with seed=" << init_seed << "...\n";
  torch::manual_seed(init_seed);
  res.model_dst = MakeModel(manifest_meta.observation_dim, dst_blocks,
                            manifest_meta.semantic_descriptor_schema,
                            manifest_meta.market_appendix_mode,
                            manifest_meta.deck_pool_semantics);

  // 3. Copy layers exactly
  {
    torch::NoGradGuard no_grad;
    CopyCommonModelLayers(res.model_src, res.model_dst, src_blocks,
                          "v5_dynamic_slot_mask_ mismatch between model_src and model_dst");

    // Append new blocks (src_blocks .. dst_blocks - 1):
    // fc1, fc2, ln1 remain at deterministic initialization from init_seed.
    // ln2 weight = 0, ln2 bias = 0.
    for (int b = src_blocks; b < dst_blocks; ++b) {
      res.model_dst->res_blocks[b]->ln2->weight.zero_();
      res.model_dst->res_blocks[b]->ln2->bias.zero_();
    }
  }
  std::cout << "[B1 SURGERY] Layers copied: input_layer, res1-" << src_blocks
            << ", policy_head, value_head, semantic_scorer.\n";
  std::cout << "[B1 SURGERY] res" << (src_blocks + 1) << "-res" << dst_blocks
            << " appended with deterministic init on fc1/fc2/ln1, and ln2 weight=0, bias=0.\n";

  // 4. Save destination model
  res.dst_model_path =
      out_dir + "/ppo_model_update_" + std::to_string(global_update) + ".pt";
  torch::save(res.model_dst, res.dst_model_path);
  res.dst_model_sha256 =
      open_spiel::ComputeFileSHA256(res.dst_model_path, &res.dst_model_size);
  std::cout << "[B1 SURGERY] Saved " << dst_blocks << "-block model: "
            << res.dst_model_path << " (" << res.dst_model_size
            << " bytes, SHA=" << res.dst_model_sha256 << ")\n";
  return res;
}

EmaSurgeryResult RunEmaModelSurgery(
    const std::string& src_ema_m_path,
    const std::string& src_ema_json_path,
    const std::string& out_dir,
    int src_blocks,
    int dst_blocks,
    int64_t global_update,
    const ManifestMetadata& manifest_meta,
    const std::shared_ptr<open_spiel::SharedDunePolicyValueNetImpl>& model_dst,
    const torch::Device& device) {
  EmaSurgeryResult res;
  if (src_ema_m_path.empty() || !std::filesystem::exists(src_ema_m_path)) {
    return res;
  }

  std::cout << "\n[B1 EMA SURGERY] Loading source " << src_blocks
            << "-block EMA model: " << src_ema_m_path << "\n";
  res.model_ema_src = MakeModel(manifest_meta.observation_dim, src_blocks,
                                manifest_meta.semantic_descriptor_schema,
                                manifest_meta.market_appendix_mode,
                                manifest_meta.deck_pool_semantics);
  open_spiel::LoadModelCheckpointRobust(res.model_ema_src, src_ema_m_path, device);
  std::cout << "[B1 EMA SURGERY] Source " << src_blocks
            << "-block EMA model loaded successfully.\n";

  std::cout << "[B1 EMA SURGERY] Constructing destination " << dst_blocks
            << "-block EMA model...\n";
  res.model_ema_dst = MakeModel(manifest_meta.observation_dim, dst_blocks,
                                manifest_meta.semantic_descriptor_schema,
                                manifest_meta.market_appendix_mode,
                                manifest_meta.deck_pool_semantics);

  {
    torch::NoGradGuard no_grad;
    CopyCommonModelLayers(
        res.model_ema_src, res.model_ema_dst, src_blocks,
        "v5_dynamic_slot_mask_ mismatch between model_ema_src and model_ema_dst");

    // Copy the EXACT SAME newly initialized res_blocks[src_blocks..dst_blocks-1] from model_dst
    for (int b = src_blocks; b < dst_blocks; ++b) {
      res.model_ema_dst->res_blocks[b]->fc1->weight.copy_(
          model_dst->res_blocks[b]->fc1->weight);
      res.model_ema_dst->res_blocks[b]->fc1->bias.copy_(
          model_dst->res_blocks[b]->fc1->bias);
      res.model_ema_dst->res_blocks[b]->fc2->weight.copy_(
          model_dst->res_blocks[b]->fc2->weight);
      res.model_ema_dst->res_blocks[b]->fc2->bias.copy_(
          model_dst->res_blocks[b]->fc2->bias);
      res.model_ema_dst->res_blocks[b]->ln1->weight.copy_(
          model_dst->res_blocks[b]->ln1->weight);
      res.model_ema_dst->res_blocks[b]->ln1->bias.copy_(
          model_dst->res_blocks[b]->ln1->bias);
      res.model_ema_dst->res_blocks[b]->ln2->weight.copy_(
          model_dst->res_blocks[b]->ln2->weight);
      res.model_ema_dst->res_blocks[b]->ln2->bias.copy_(
          model_dst->res_blocks[b]->ln2->bias);
    }
  }
  std::cout << "[B1 EMA SURGERY] Layers copied: input_layer, res1-" << src_blocks
            << ", policy_head, value_head, semantic_scorer from source EMA.\n";
  std::cout << "[B1 EMA SURGERY] res" << (src_blocks + 1) << "-res" << dst_blocks
            << " copied identically from destination live model.\n";

  res.dst_ema_model_path =
      out_dir + "/ppo_model_ema_update_" + std::to_string(global_update) + ".pt";
  torch::save(res.model_ema_dst, res.dst_ema_model_path);
  res.dst_ema_model_sha256 =
      open_spiel::ComputeFileSHA256(res.dst_ema_model_path, &res.dst_ema_model_size);
  std::cout << "[B1 EMA SURGERY] Saved " << dst_blocks << "-block EMA model: "
            << res.dst_ema_model_path << " (" << res.dst_ema_model_size
            << " bytes, SHA=" << res.dst_ema_model_sha256 << ")\n";

  if (!src_ema_json_path.empty() && std::filesystem::exists(src_ema_json_path)) {
    std::ifstream src_ema_jf(src_ema_json_path);
    if (!src_ema_jf.is_open()) {
      open_spiel::SpielFatalError("Cannot open src_ema_manifest: " + src_ema_json_path);
    }
    std::string j_ema_content((std::istreambuf_iterator<char>(src_ema_jf)),
                              std::istreambuf_iterator<char>());
    src_ema_jf.close();

    auto j_ema_opt = open_spiel::json::FromString(j_ema_content);
    if (!j_ema_opt) {
      open_spiel::SpielFatalError("Failed parsing source EMA manifest JSON");
    }
    res.j_ema_obj = j_ema_opt->GetObject();
    res.has_ema_manifest = true;

    res.j_ema_obj["num_blocks"] =
        open_spiel::json::Value(static_cast<int64_t>(dst_blocks));
    res.j_ema_obj["model_filename"] =
        open_spiel::json::Value("ppo_model_ema_update_" + std::to_string(global_update) + ".pt");
    res.j_ema_obj["model_file_size"] =
        open_spiel::json::Value(static_cast<int64_t>(res.dst_ema_model_size));
    res.j_ema_obj["model_sha256"] = open_spiel::json::Value(res.dst_ema_model_sha256);
  }
  return res;
}

OptimizerSurgeryResult RunOptimizerSurgery(
    const std::shared_ptr<open_spiel::SharedDunePolicyValueNetImpl>& model_src,
    const std::shared_ptr<open_spiel::SharedDunePolicyValueNetImpl>& model_dst,
    const std::string& src_opt_path,
    const std::string& out_dir,
    int src_blocks,
    int dst_blocks,
    int src_nb_first_idx,
    int dst_nb_first_idx,
    double base_lr,
    double nb_lr,
    int64_t global_update,
    const ManifestMetadata& manifest_meta,
    const torch::Device& device) {
  OptimizerSurgeryResult res;
  std::cout << "\n[B2 OPTIMIZER] Constructing source and destination AdamW optimizers via MakeDuneOptimizer...\n";
  res.opt_src = open_spiel::MakeDuneOptimizer(
      model_src, base_lr, /*weight_decay=*/0.0, /*policy_weight_decay=*/0.0,
      nb_lr, src_nb_first_idx);
  torch::optim::AdamW& opt_src = *res.opt_src;
  torch::load(opt_src, src_opt_path, device);
  std::cout << "[B2 OPTIMIZER] Source optimizer loaded successfully from " << src_opt_path << "\n";

  res.opt_dst = open_spiel::MakeDuneOptimizer(
      model_dst, base_lr, /*weight_decay=*/0.0, /*policy_weight_decay=*/0.0,
      nb_lr, dst_nb_first_idx);
  torch::optim::AdamW& opt_dst = *res.opt_dst;

  // Verify group counts on opt_dst
  VerifyOptimizerGroups(opt_dst, dst_blocks, dst_nb_first_idx, base_lr, nb_lr,
                        manifest_meta.semantic_descriptor_schema);

  // Migrate state from opt_src to opt_dst
  auto copy_state = [&](const torch::Tensor& t_src, const torch::Tensor& t_dst) {
    auto it = opt_src.state().find(t_src.unsafeGetTensorImpl());
    if (it == opt_src.state().end()) {
      open_spiel::SpielFatalError("Source param missing from source optimizer state!");
    }
    auto* s_src = dynamic_cast<torch::optim::AdamWParamState*>(it->second.get());
    auto s_dst = std::make_unique<torch::optim::AdamWParamState>();
    s_dst->step(s_src->step());
    s_dst->exp_avg(s_src->exp_avg().clone());
    s_dst->exp_avg_sq(s_src->exp_avg_sq().clone());
    opt_dst.state()[t_dst.unsafeGetTensorImpl()] = std::move(s_dst);
  };

  // Policy head
  copy_state(model_src->policy_head->weight, model_dst->policy_head->weight);
  if (model_src->policy_head->bias.defined()) {
    copy_state(model_src->policy_head->bias, model_dst->policy_head->bias);
  }
  // Input layer
  copy_state(model_src->input_layer->weight, model_dst->input_layer->weight);
  if (model_src->input_layer->bias.defined()) {
    copy_state(model_src->input_layer->bias, model_dst->input_layer->bias);
  }
  // Inherited res blocks
  for (int b = 0; b < src_blocks; ++b) {
    copy_state(model_src->res_blocks[b]->fc1->weight, model_dst->res_blocks[b]->fc1->weight);
    copy_state(model_src->res_blocks[b]->fc1->bias, model_dst->res_blocks[b]->fc1->bias);
    copy_state(model_src->res_blocks[b]->fc2->weight, model_dst->res_blocks[b]->fc2->weight);
    copy_state(model_src->res_blocks[b]->fc2->bias, model_dst->res_blocks[b]->fc2->bias);
    copy_state(model_src->res_blocks[b]->ln1->weight, model_dst->res_blocks[b]->ln1->weight);
    copy_state(model_src->res_blocks[b]->ln1->bias, model_dst->res_blocks[b]->ln1->bias);
    copy_state(model_src->res_blocks[b]->ln2->weight, model_dst->res_blocks[b]->ln2->weight);
    copy_state(model_src->res_blocks[b]->ln2->bias, model_dst->res_blocks[b]->ln2->bias);
  }
  // Value head
  copy_state(model_src->value_head->weight, model_dst->value_head->weight);
  if (model_src->value_head->bias.defined()) {
    copy_state(model_src->value_head->bias, model_dst->value_head->bias);
  }
  // Semantic scorer: ALL named parameters including v5 tensors
  if (model_src->semantic_scorer_ && model_dst->semantic_scorer_) {
    const auto& p_src = model_src->semantic_scorer_->named_parameters();
    const auto& p_dst = model_dst->semantic_scorer_->named_parameters();
    for (const auto& pair : p_src) {
      copy_state(pair.value(), p_dst[pair.key()]);
    }
  }

  // Initialize fresh state for new blocks
  for (int b = src_blocks; b < dst_blocks; ++b) {
    for (const auto& p : model_dst->res_blocks[b]->parameters()) {
      auto s_new = std::make_unique<torch::optim::AdamWParamState>();
      s_new->step(0);
      s_new->exp_avg(torch::zeros_like(p));
      s_new->exp_avg_sq(torch::zeros_like(p));
      opt_dst.state()[p.unsafeGetTensorImpl()] = std::move(s_new);
    }
  }

  res.dst_opt_path =
      out_dir + "/ppo_optimizer_update_" + std::to_string(global_update) + ".pt";
  torch::save(opt_dst, res.dst_opt_path);
  res.dst_opt_sha256 =
      open_spiel::ComputeFileSHA256(res.dst_opt_path, &res.dst_opt_size);
  std::cout << "[B2 OPTIMIZER] Saved migrated " << dst_blocks << "-block optimizer: "
            << res.dst_opt_path << " (" << res.dst_opt_size
            << " bytes, SHA=" << res.dst_opt_sha256 << ")\n";
  return res;
}

void UpdateAndSaveManifests(
    open_spiel::json::Object& j_obj,
    EmaSurgeryResult& ema_res,
    const ManifestMetadata& manifest_meta,
    int src_blocks,
    int dst_blocks,
    double base_lr,
    const std::string& target_opp_pool,
    bool update_config_fingerprint,
    int64_t global_update,
    const std::string& out_dir,
    const LiveSurgeryResult& live_res,
    const OptimizerSurgeryResult& opt_res) {
  // 6. Config Fingerprint Verification & Recomputation
  std::string dst_fp = "";
  if (update_config_fingerprint) {
    std::cout << "\n[CONFIG FINGERPRINT] Verifying source manifest config fingerprint...\n";
    std::string src_fp =
        ComputeManifestConfigFingerprint(src_blocks, base_lr, manifest_meta);
    std::cout << "Computed source fingerprint: " << src_fp << "\n";
    std::cout << "Manifest source fingerprint: " << manifest_meta.config_fingerprint << "\n";
    if (!manifest_meta.config_fingerprint.empty() &&
        src_fp != manifest_meta.config_fingerprint) {
      open_spiel::SpielFatalError(absl::StrFormat(
          "MANDATORY SELF-VERIFICATION FAILED: Source config_fingerprint mismatch! Computed %s vs Manifest %s",
          src_fp, manifest_meta.config_fingerprint));
    }
    std::cout << "[CONFIG FINGERPRINT PASS] Exact match verified on source config fingerprint!\n";

    dst_fp = ComputeManifestConfigFingerprint(
        dst_blocks, base_lr, manifest_meta, target_opp_pool);
    std::cout << "Recomputed destination fingerprint: " << dst_fp << "\n";
    j_obj["config_fingerprint"] = open_spiel::json::Value(dst_fp);
    if (!target_opp_pool.empty()) {
      j_obj["training_opponent_pool"] = open_spiel::json::Value(target_opp_pool);
    }

    if (ema_res.has_ema_manifest) {
      ema_res.j_ema_obj["config_fingerprint"] = open_spiel::json::Value(dst_fp);
      if (!target_opp_pool.empty()) {
        ema_res.j_ema_obj["training_opponent_pool"] =
            open_spiel::json::Value(target_opp_pool);
      }
    }
  }

  // 6b. Update & Save JSON Manifests
  std::cout << "\n[B1 MANIFEST] Updating JSON manifest...\n";
  j_obj["num_blocks"] = open_spiel::json::Value(static_cast<int64_t>(dst_blocks));
  j_obj["model_filename"] =
      open_spiel::json::Value("ppo_model_update_" + std::to_string(global_update) + ".pt");
  j_obj["model_file_size"] =
      open_spiel::json::Value(static_cast<int64_t>(live_res.dst_model_size));
  j_obj["model_sha256"] = open_spiel::json::Value(live_res.dst_model_sha256);
  j_obj["optimizer_filename"] =
      open_spiel::json::Value("ppo_optimizer_update_" + std::to_string(global_update) + ".pt");
  j_obj["optimizer_file_size"] =
      open_spiel::json::Value(static_cast<int64_t>(opt_res.dst_opt_size));
  j_obj["optimizer_sha256"] = open_spiel::json::Value(opt_res.dst_opt_sha256);

  std::string dst_json_path =
      out_dir + "/ppo_model_update_" + std::to_string(global_update) + ".json";
  std::ofstream out_jf(dst_json_path);
  out_jf << open_spiel::json::ToString(open_spiel::json::Value(j_obj));
  out_jf.close();
  std::cout << "[B1 MANIFEST] Saved updated manifest with num_blocks=" << dst_blocks
            << ": " << dst_json_path << "\n";

  if (ema_res.has_ema_manifest) {
    std::string dst_ema_json_path =
        out_dir + "/ppo_model_ema_update_" + std::to_string(global_update) + ".json";
    std::ofstream out_ema_jf(dst_ema_json_path);
    out_ema_jf << open_spiel::json::ToString(open_spiel::json::Value(ema_res.j_ema_obj));
    out_ema_jf.close();
    std::cout << "[B1 EMA MANIFEST] Saved updated EMA manifest with num_blocks="
              << dst_blocks << ": " << dst_ema_json_path << "\n";
  }
}

}  // namespace open_spiel::dune_surgery_12block
