#include "dune_ppo_train_internal.h"

namespace open_spiel::ppo_train_internal {

#ifdef OPEN_SPIEL_BUILD_WITH_LIBTORCH

void UpdateEmaWeights(
    const std::shared_ptr<SharedDunePolicyValueNetImpl>& model,
    std::shared_ptr<SharedDunePolicyValueNetImpl>& ema_model,
    double decay) {
  torch::NoGradGuard no_grad;
  auto model_params = model->named_parameters();
  auto ema_params = ema_model->named_parameters();
  for (const auto& kv : model_params) {
    auto it = ema_params.find(kv.key());
    if (it != nullptr) {
      it->mul_(decay).add_(kv.value().detach(), 1.0 - decay);
    }
  }
  auto model_buffers = model->named_buffers();
  auto ema_buffers = ema_model->named_buffers();
  for (const auto& kv : model_buffers) {
    auto it = ema_buffers.find(kv.key());
    if (it != nullptr) {
      it->copy_(kv.value().detach());
    }
  }
  if (ema_model->bf16_inference_cache_ready()) {
    ema_model->RefreshBf16InferenceCache();
  }
}

void SaveCheckpoint(std::shared_ptr<SharedDunePolicyValueNetImpl> model,
                    torch::optim::AdamW& optimizer,
                    const std::string& model_path,
                    const std::string& optim_path,
                    int global_update,
                    int target_end_update,
                    uint64_t total_env_steps,
                    uint64_t next_episode_id,
                    uint64_t base_seed,
                    int seed_scheme_version,
                    const std::string& config_fingerprint,
                    const std::string& search_label_fingerprint,
                    const std::string& run_uuid,
                    const open_spiel::OnlineCollectionState* aux_state,
                    std::shared_ptr<SharedDunePolicyValueNetImpl> ema_model,
                    int64_t ema_updates_absorbed,
                    double ema_half_life,
                    double ema_decay) {
  if (model_path.empty() || optim_path.empty()) {
    SpielFatalError("SaveCheckpoint called with empty path.");
  }
  std::string input_model = absl::GetFlag(FLAGS_model_checkpoint);
  if (!input_model.empty()) {
    std::error_code ec;
    std::filesystem::path in_model = std::filesystem::absolute(input_model, ec);
    std::filesystem::path out_model = std::filesystem::absolute(model_path, ec);
    if (in_model == out_model) {
      SpielFatalError("CRITICAL BASELINE OVERWRITE ERROR: SaveCheckpoint attempted to overwrite input model checkpoint: " + input_model);
    }
    if (std::filesystem::exists(input_model) && std::filesystem::exists(model_path)) {
      if (std::filesystem::equivalent(model_path, input_model, ec)) {
        SpielFatalError("CRITICAL BASELINE OVERWRITE ERROR: SaveCheckpoint attempted to overwrite input model checkpoint: " + input_model);
      }
    }
  }
  std::string input_optim = absl::GetFlag(FLAGS_optim_checkpoint);
  if (!input_optim.empty()) {
    std::error_code ec;
    std::filesystem::path in_optim = std::filesystem::absolute(input_optim, ec);
    std::filesystem::path out_optim = std::filesystem::absolute(optim_path, ec);
    if (in_optim == out_optim) {
      SpielFatalError("CRITICAL BASELINE OVERWRITE ERROR: SaveCheckpoint attempted to overwrite input optimizer checkpoint: " + input_optim);
    }
    if (std::filesystem::exists(input_optim) && std::filesystem::exists(optim_path)) {
      if (std::filesystem::equivalent(optim_path, input_optim, ec)) {
        SpielFatalError("CRITICAL BASELINE OVERWRITE ERROR: SaveCheckpoint attempted to overwrite input optimizer checkpoint: " + input_optim);
      }
    }
  }

  std::string model_tmp = model_path + ".tmp";
  std::string optim_tmp = optim_path + ".tmp";

  std::filesystem::path manifest_path = model_path;
  manifest_path.replace_extension(".json");
  std::string manifest_path_str = manifest_path.string();
  std::string manifest_tmp = manifest_path_str + ".tmp";

  std::string ema_model_path;
  std::string ema_tmp;
  std::string ema_manifest_path;
  std::string ema_manifest_tmp;

  try {
    torch::save(model, model_tmp);
    torch::save(optimizer, optim_tmp);

    size_t model_size = 0;
    std::string model_hash = ComputeFileSHA256(model_tmp, &model_size);

    size_t optim_size = 0;
    std::string optim_hash = ComputeFileSHA256(optim_tmp, &optim_size);

    std::string checkpoint_uuid = GenerateUUID();
    json::Object manifest_obj;
    manifest_obj["schema_version"] = static_cast<int64_t>(2);
    manifest_obj["checkpoint_uuid"] = checkpoint_uuid;
    manifest_obj["global_update"] = static_cast<int64_t>(global_update);
    manifest_obj["target_end_update"] = static_cast<int64_t>(target_end_update);
    manifest_obj["total_env_steps"] = static_cast<int64_t>(total_env_steps);
    manifest_obj["next_episode_id"] = static_cast<int64_t>(next_episode_id);
    manifest_obj["base_seed"] = static_cast<int64_t>(base_seed);
    manifest_obj["seed_scheme_version"] = static_cast<int64_t>(seed_scheme_version);
    manifest_obj["config_fingerprint"] = config_fingerprint;
    WritePpoPrecisionManifestFields(
        manifest_obj, absl::GetFlag(FLAGS_rollout_amp),
        absl::GetFlag(FLAGS_allow_tf32));
    manifest_obj["train_amp"] = absl::GetFlag(FLAGS_train_amp);
    manifest_obj["search_label_fingerprint"] = search_label_fingerprint;
    manifest_obj["run_uuid"] = run_uuid;
    manifest_obj["model_filename"] = std::filesystem::path(model_path).filename().string();
    manifest_obj["model_file_size"] = static_cast<int64_t>(model_size);
    manifest_obj["model_sha256"] = model_hash;
    manifest_obj["optimizer_filename"] = std::filesystem::path(optim_path).filename().string();
    manifest_obj["optimizer_file_size"] = static_cast<int64_t>(optim_size);
    manifest_obj["optimizer_sha256"] = optim_hash;
    manifest_obj["hidden_dim"] = static_cast<int64_t>(absl::GetFlag(FLAGS_hidden_dim));
    manifest_obj["num_blocks"] = static_cast<int64_t>(absl::GetFlag(FLAGS_num_blocks));
    manifest_obj["observation_dim"] = static_cast<int64_t>(model->input_layer->weight.size(1));
    manifest_obj["market_appendix_mode"] = absl::GetFlag(FLAGS_market_appendix_mode);
    const auto resolved_deck_pool_semantics = dune_imperium::ParseDeckPoolSemanticsOrDie(absl::GetFlag(FLAGS_deck_pool_semantics));
    manifest_obj["deck_pool_semantics"] = dune_imperium::DeckPoolSemanticsToString(resolved_deck_pool_semantics);
    const auto schema = dune_imperium::GetInformationStateSchema(
        dune_imperium::ParseMarketAppendixMode(absl::GetFlag(FLAGS_market_appendix_mode)),
        resolved_deck_pool_semantics);
    if (static_cast<int64_t>(model->input_layer->weight.size(1)) == schema.size) {
      manifest_obj["feature_schema_label"] = std::string(schema.label);
      manifest_obj["feature_schema_sha256"] = std::string(schema.sha256);
    }
    manifest_obj["enable_semantic_scorer"] = absl::GetFlag(FLAGS_enable_semantic_scorer);
    if (absl::GetFlag(FLAGS_enable_semantic_scorer)) {
      manifest_obj["semantic_descriptor_schema"] = model->semantic_descriptor_schema_;
      if (model->semantic_scorer_ != nullptr && model->semantic_scorer_->slot_indexed_f_) {
        manifest_obj["slot_indexed_out_layer_f"] = true;
      }
    }
    manifest_obj["enable_zone_encoder"] = model->enable_zone_encoder_;
    manifest_obj["specimen_exchange_penalty"] = absl::GetFlag(FLAGS_specimen_exchange_penalty);
    manifest_obj["family_atomics_penalty"] = absl::GetFlag(FLAGS_family_atomics_penalty);
    manifest_obj["plot_intrigue_penalty"] = absl::GetFlag(FLAGS_plot_intrigue_penalty);
    manifest_obj["plot_intrigue_exemption_threshold"] = static_cast<int64_t>(absl::GetFlag(FLAGS_plot_intrigue_exemption_threshold));
    if (absl::GetFlag(FLAGS_round7_speed_bonus) != 0.0) {
      manifest_obj["round7_speed_bonus"] = absl::GetFlag(FLAGS_round7_speed_bonus);
    }
    if (absl::GetFlag(FLAGS_win_speed_bonus_per_round) != 0.0) {
      manifest_obj["win_speed_bonus_per_round"] = absl::GetFlag(FLAGS_win_speed_bonus_per_round);
    }
    if (absl::GetFlag(FLAGS_terminal_reward_mode) != "placement") {
      manifest_obj["terminal_reward_mode"] = absl::GetFlag(FLAGS_terminal_reward_mode);
    }
    if (absl::GetFlag(FLAGS_v5_appendix_lr_mult) > 0.0) {
      manifest_obj["v5_appendix_lr_mult"] = absl::GetFlag(FLAGS_v5_appendix_lr_mult);
    }
    manifest_obj["rollout_games"] = static_cast<int64_t>(absl::GetFlag(FLAGS_rollout_games));
    manifest_obj["random_leader_draft"] = absl::GetFlag(FLAGS_random_leader_draft);
    if (!g_reward_transition_source_fingerprint.empty()) {
      manifest_obj["reward_transition_source_fingerprint"] = g_reward_transition_source_fingerprint;
      if (g_reward_transition_source_update >= 0) {
        manifest_obj["reward_transition_source_update"] = g_reward_transition_source_update;
      }
    }
    if (!g_gae_transition_source_fingerprint.empty()) {
      manifest_obj["gae_transition_source_fingerprint"] = g_gae_transition_source_fingerprint;
      if (g_gae_transition_source_update >= 0) {
        manifest_obj["gae_transition_source_update"] = g_gae_transition_source_update;
      }
    }
    if (!g_learning_rate_transition_source_fingerprint.empty()) {
      manifest_obj["learning_rate_transition_source_fingerprint"] = g_learning_rate_transition_source_fingerprint;
      if (g_learning_rate_transition_source_update >= 0) {
        manifest_obj["learning_rate_transition_source_update"] = g_learning_rate_transition_source_update;
      }
    }
    if (!g_target_kl_transition_source_fingerprint.empty()) {
      manifest_obj["target_kl_transition_source_fingerprint"] = g_target_kl_transition_source_fingerprint;
      if (g_target_kl_transition_source_update >= 0) {
        manifest_obj["target_kl_transition_source_update"] = g_target_kl_transition_source_update;
      }
    }
    if (!g_single_learner_transition_source_fingerprint.empty()) {
      manifest_obj["single_learner_transition_source_fingerprint"] = g_single_learner_transition_source_fingerprint;
      if (g_single_learner_transition_source_update >= 0) {
        manifest_obj["single_learner_transition_source_update"] = g_single_learner_transition_source_update;
      }
    }
    if (!g_ema_magnet_transition_source_fingerprint.empty()) {
      manifest_obj["ema_magnet_transition_source_fingerprint"] = g_ema_magnet_transition_source_fingerprint;
      if (g_ema_magnet_transition_source_update >= 0) {
        manifest_obj["ema_magnet_transition_source_update"] = g_ema_magnet_transition_source_update;
      }
    }
    if (absl::GetFlag(FLAGS_ema_magnet_kl_coef) > 0.0) {
      manifest_obj["ema_magnet_kl_coef"] = absl::GetFlag(FLAGS_ema_magnet_kl_coef);
    }
    if (!g_pool_transition_source_fingerprint.empty()) {
      manifest_obj["pool_transition_source_fingerprint"] = g_pool_transition_source_fingerprint;
      if (g_pool_transition_source_update >= 0) {
        manifest_obj["pool_transition_source_update"] = g_pool_transition_source_update;
      }
    }
    if (!g_market_appendix_transition_source_fingerprint.empty()) {
      manifest_obj["market_appendix_transition_source_fingerprint"] = g_market_appendix_transition_source_fingerprint;
      if (g_market_appendix_transition_source_update >= 0) {
        manifest_obj["market_appendix_transition_source_update"] = g_market_appendix_transition_source_update;
      }
    }
    if (!g_deck_pool_semantics_transition_source_fingerprint.empty()) {
      manifest_obj["deck_pool_semantics_transition_source_fingerprint"] = g_deck_pool_semantics_transition_source_fingerprint;
      if (g_deck_pool_semantics_transition_source_update >= 0) {
        manifest_obj["deck_pool_semantics_transition_source_update"] = g_deck_pool_semantics_transition_source_update;
      }
    }
    if (!g_tf32_transition_source_fingerprint.empty()) {
      manifest_obj["tf32_transition_source_fingerprint"] = g_tf32_transition_source_fingerprint;
      if (g_tf32_transition_source_update >= 0) {
        manifest_obj["tf32_transition_source_update"] = g_tf32_transition_source_update;
      }
    }
    if (!g_semantic_schema_transition_source_fingerprint.empty()) {
      manifest_obj["semantic_schema_transition_source_fingerprint"] = g_semantic_schema_transition_source_fingerprint;
      if (g_semantic_schema_transition_source_update >= 0) {
        manifest_obj["semantic_schema_transition_source_update"] = g_semantic_schema_transition_source_update;
      }
    }
    if (absl::GetFlag(FLAGS_pool_game_fraction) > 0.0 &&
        absl::GetFlag(FLAGS_pool_opponents_per_game) > 0) {
      manifest_obj["pool_game_fraction"] = absl::GetFlag(FLAGS_pool_game_fraction);
      manifest_obj["pool_opponents_per_game"] = absl::GetFlag(FLAGS_pool_opponents_per_game);
      manifest_obj["training_opponent_pool"] = absl::GetFlag(FLAGS_training_opponent_pool);
    }
    // Phase 18B: persist online-collection state for exact resume (plan §18C).
    if (aux_state != nullptr) {
      open_spiel::WriteOnlineCollectionState(manifest_obj, *aux_state);
    }

    {
      std::ofstream ofs(manifest_tmp);
      if (!ofs) {
        throw std::runtime_error("Could not open manifest temp file for writing: " + manifest_tmp);
      }
      ofs << json::ToString(manifest_obj, true);
    }

    std::filesystem::rename(model_tmp, model_path);
    std::filesystem::rename(optim_tmp, optim_path);
    std::filesystem::rename(manifest_tmp, manifest_path);

    std::cout << "Saved checkpoint successfully:\n"
              << "  Model: " << model_path << " (" << model_size << " bytes, sha256=" << model_hash << ")\n"
              << "  Optimizer: " << optim_path << " (" << optim_size << " bytes, sha256=" << optim_hash << ")\n"
              << "  Manifest: " << manifest_path << "\n";

    if (ema_model != nullptr) {
      std::string filename = std::filesystem::path(model_path).filename().string();
      std::string parent_dir = std::filesystem::path(model_path).parent_path().string();
      std::string ema_filename;
      if (filename.find("_model_update_") != std::string::npos) {
        ema_filename = filename;
        size_t pos = ema_filename.find("_model_update_");
        ema_filename.replace(pos, 14, "_model_ema_update_");
      } else if (filename.find("_model_final") != std::string::npos) {
        ema_filename = filename;
        size_t pos = ema_filename.find("_model_final");
        ema_filename.replace(pos, 12, "_model_ema_final");
      } else {
        ema_filename = "ema_" + filename;
      }
      ema_model_path = parent_dir.empty() ? ema_filename : (parent_dir + "/" + ema_filename);
      ema_tmp = ema_model_path + ".tmp";
      std::filesystem::path emp(ema_model_path);
      emp.replace_extension(".json");
      ema_manifest_path = emp.string();
      ema_manifest_tmp = ema_manifest_path + ".tmp";

      torch::save(ema_model, ema_tmp);
      size_t ema_size = 0;
      std::string ema_hash = ComputeFileSHA256(ema_tmp, &ema_size);

      json::Object ema_manifest_obj = manifest_obj;
      ema_manifest_obj["checkpoint_uuid"] = GenerateUUID();
      ema_manifest_obj["model_filename"] = ema_filename;
      ema_manifest_obj["model_file_size"] = static_cast<int64_t>(ema_size);
      ema_manifest_obj["model_sha256"] = ema_hash;
      ema_manifest_obj["ema_updates_absorbed"] = static_cast<int64_t>(ema_updates_absorbed);
      ema_manifest_obj["ema_half_life"] = ema_half_life;
      ema_manifest_obj["ema_decay"] = ema_decay;
      ema_manifest_obj["source_model_filename"] = std::filesystem::path(model_path).filename().string();
      ema_manifest_obj.erase("optimizer_filename");
      ema_manifest_obj.erase("optimizer_file_size");
      ema_manifest_obj.erase("optimizer_sha256");

      {
        std::ofstream ofs(ema_manifest_tmp);
        if (!ofs) throw std::runtime_error("Could not open ema manifest temp file: " + ema_manifest_tmp);
        ofs << json::ToString(ema_manifest_obj, true);
      }

      std::filesystem::rename(ema_tmp, ema_model_path);
      std::filesystem::rename(ema_manifest_tmp, ema_manifest_path);

      std::cout << "  EMA Model: " << ema_model_path << " (" << ema_size << " bytes, sha256=" << ema_hash
                << ", absorbed=" << ema_updates_absorbed << ")\n"
                << "  EMA Manifest: " << ema_manifest_path << "\n";
    }
  } catch (const std::exception& e) {
    std::cerr << "CRITICAL ERROR: SaveCheckpoint failed: " << e.what() << "\n";
    if (std::filesystem::exists(model_tmp)) std::filesystem::remove(model_tmp);
    if (std::filesystem::exists(optim_tmp)) std::filesystem::remove(optim_tmp);
    if (std::filesystem::exists(manifest_tmp)) std::filesystem::remove(manifest_tmp);
    if (!ema_tmp.empty() && std::filesystem::exists(ema_tmp)) std::filesystem::remove(ema_tmp);
    if (!ema_manifest_tmp.empty() && std::filesystem::exists(ema_manifest_tmp)) std::filesystem::remove(ema_manifest_tmp);
    SpielFatalError("SaveCheckpoint failed, aborting training to prevent corrupted states.");
  }
}

// Reuses an existing completed checkpoint bundle (update) to publish the final
// aliases (ppo_model_final.*) atomically without calling torch::save again.
bool PublishFinalCheckpointFromExisting(
    const std::string& prefix, int update,
    bool separate_actor_critic) {
  std::error_code ec;
  if (separate_actor_critic) {
    std::string src_m = absl::StrCat(prefix, "_model_update_", update, ".pt");
    std::string src_o = absl::StrCat(prefix, "_optimizer_update_", update, ".pt");
    std::string src_cm = absl::StrCat(prefix, "_critic_model_update_", update, ".pt");
    std::string src_co = absl::StrCat(prefix, "_critic_optimizer_update_", update, ".pt");
    std::string src_j = absl::StrCat(prefix, "_model_update_", update, ".json");

    if (!std::filesystem::exists(src_m, ec) || !std::filesystem::exists(src_o, ec) ||
        !std::filesystem::exists(src_cm, ec) || !std::filesystem::exists(src_co, ec) ||
        !std::filesystem::exists(src_j, ec)) {
      return false;
    }

    std::string dst_m = absl::StrCat(prefix, "_model_final.pt");
    std::string dst_o = absl::StrCat(prefix, "_optimizer_final.pt");
    std::string dst_cm = absl::StrCat(prefix, "_critic_model_final.pt");
    std::string dst_co = absl::StrCat(prefix, "_critic_optimizer_final.pt");
    std::string dst_j = absl::StrCat(prefix, "_model_final.json");

    auto link_or_copy = [](const std::string& src, const std::string& dst) {
      if (std::filesystem::exists(dst)) {
        std::error_code eq_ec;
        if (std::filesystem::equivalent(src, dst, eq_ec)) return;
      }
      std::string tmp = dst + ".tmp";
      std::error_code ec;
      if (std::filesystem::exists(tmp, ec)) std::filesystem::remove(tmp, ec);
      std::filesystem::create_hard_link(src, tmp, ec);
      if (ec) {
        std::filesystem::copy_file(src, tmp, std::filesystem::copy_options::overwrite_existing, ec);
        if (ec) {
          SpielFatalError("Failed to link or copy bundle artifact (" + src + " -> " + tmp + "): " + ec.message());
        }
      }
      std::filesystem::rename(tmp, dst);
    };

    link_or_copy(src_m, dst_m);
    link_or_copy(src_o, dst_o);
    link_or_copy(src_cm, dst_cm);
    link_or_copy(src_co, dst_co);

    std::ifstream ifs(src_j);
    if (!ifs) return false;
    std::string json_str((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
    ifs.close();
    auto parsed = json::FromString(json_str);
    if (parsed && parsed->IsObject()) {
      auto obj = parsed->GetObject();
      obj["model_filename"] = std::filesystem::path(dst_m).filename().string();
      obj["optimizer_filename"] = std::filesystem::path(dst_o).filename().string();
      obj["critic_model_filename"] = std::filesystem::path(dst_cm).filename().string();
      obj["critic_optimizer_filename"] = std::filesystem::path(dst_co).filename().string();
      std::string tmp_j = dst_j + ".tmp";
      std::ofstream ofs(tmp_j);
      ofs << json::ToString(obj, true);
      ofs.close();
      std::filesystem::rename(tmp_j, dst_j);
    }

    std::string src_ema_m = absl::StrCat(prefix, "_model_ema_update_", update, ".pt");
    std::string src_ema_j = absl::StrCat(prefix, "_model_ema_update_", update, ".json");
    if (std::filesystem::exists(src_ema_m, ec) && std::filesystem::exists(src_ema_j, ec)) {
      std::string dst_ema_m = absl::StrCat(prefix, "_model_ema_final.pt");
      std::string dst_ema_j = absl::StrCat(prefix, "_model_ema_final.json");
      link_or_copy(src_ema_m, dst_ema_m);
      std::ifstream eifs(src_ema_j);
      if (eifs) {
        std::string ejson_str((std::istreambuf_iterator<char>(eifs)), std::istreambuf_iterator<char>());
        eifs.close();
        auto e_parsed = json::FromString(ejson_str);
        if (e_parsed && e_parsed->IsObject()) {
          auto e_obj = e_parsed->GetObject();
          e_obj["model_filename"] = std::filesystem::path(dst_ema_m).filename().string();
          std::string tmp_ej = dst_ema_j + ".tmp";
          std::ofstream eofs(tmp_ej);
          eofs << json::ToString(e_obj, true);
          eofs.close();
          std::filesystem::rename(tmp_ej, dst_ema_j);
        }
      }
    }
    return true;
  } else {
    std::string src_m = absl::StrCat(prefix, "_model_update_", update, ".pt");
    std::string src_o = absl::StrCat(prefix, "_optimizer_update_", update, ".pt");
    std::string src_j = absl::StrCat(prefix, "_model_update_", update, ".json");

    if (!std::filesystem::exists(src_m, ec) || !std::filesystem::exists(src_o, ec) ||
        !std::filesystem::exists(src_j, ec)) {
      return false;
    }

    std::string dst_m = absl::StrCat(prefix, "_model_final.pt");
    std::string dst_o = absl::StrCat(prefix, "_optimizer_final.pt");
    std::string dst_j = absl::StrCat(prefix, "_model_final.json");

    auto link_or_copy = [](const std::string& src, const std::string& dst) {
      if (std::filesystem::exists(dst)) {
        std::error_code eq_ec;
        if (std::filesystem::equivalent(src, dst, eq_ec)) return;
      }
      std::string tmp = dst + ".tmp";
      std::error_code ec;
      if (std::filesystem::exists(tmp, ec)) std::filesystem::remove(tmp, ec);
      std::filesystem::create_hard_link(src, tmp, ec);
      if (ec) {
        std::filesystem::copy_file(src, tmp, std::filesystem::copy_options::overwrite_existing, ec);
        if (ec) {
          SpielFatalError("Failed to link or copy bundle artifact (" + src + " -> " + tmp + "): " + ec.message());
        }
      }
      std::filesystem::rename(tmp, dst);
    };

    link_or_copy(src_m, dst_m);
    link_or_copy(src_o, dst_o);

    std::ifstream ifs(src_j);
    if (!ifs) return false;
    std::string json_str((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
    ifs.close();
    auto parsed = json::FromString(json_str);
    if (parsed && parsed->IsObject()) {
      auto obj = parsed->GetObject();
      obj["model_filename"] = std::filesystem::path(dst_m).filename().string();
      obj["optimizer_filename"] = std::filesystem::path(dst_o).filename().string();
      std::string tmp_j = dst_j + ".tmp";
      std::ofstream ofs(tmp_j);
      ofs << json::ToString(obj, true);
      ofs.close();
      std::filesystem::rename(tmp_j, dst_j);
    }

    std::string src_ema_m = absl::StrCat(prefix, "_model_ema_update_", update, ".pt");
    std::string src_ema_j = absl::StrCat(prefix, "_model_ema_update_", update, ".json");
    if (std::filesystem::exists(src_ema_m, ec) && std::filesystem::exists(src_ema_j, ec)) {
      std::string dst_ema_m = absl::StrCat(prefix, "_model_ema_final.pt");
      std::string dst_ema_j = absl::StrCat(prefix, "_model_ema_final.json");
      link_or_copy(src_ema_m, dst_ema_m);
      std::ifstream eifs(src_ema_j);
      if (eifs) {
        std::string ejson_str((std::istreambuf_iterator<char>(eifs)), std::istreambuf_iterator<char>());
        eifs.close();
        auto e_parsed = json::FromString(ejson_str);
        if (e_parsed && e_parsed->IsObject()) {
          auto e_obj = e_parsed->GetObject();
          e_obj["model_filename"] = std::filesystem::path(dst_ema_m).filename().string();
          std::string tmp_ej = dst_ema_j + ".tmp";
          std::ofstream eofs(tmp_ej);
          eofs << json::ToString(e_obj, true);
          eofs.close();
          std::filesystem::rename(tmp_ej, dst_ema_j);
        }
      }
    }
    return true;
  }
}





#endif  // OPEN_SPIEL_BUILD_WITH_LIBTORCH

}  // namespace open_spiel::ppo_train_internal
