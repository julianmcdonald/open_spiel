#include "dune_ppo_train_run_context.h"

namespace open_spiel::ppo_train_internal {

#ifdef OPEN_SPIEL_BUILD_WITH_LIBTORCH

namespace {

void RunRandomInit(PpoTrainRunContext& ctx) {
  std::string model_path = absl::GetFlag(FLAGS_model_checkpoint);
  std::string optim_path = absl::GetFlag(FLAGS_optim_checkpoint);
  std::filesystem::path m_path = model_path;
  m_path.replace_extension(".json");
  std::string manifest_path = m_path.string();
  if (std::filesystem::exists(model_path) ||
      std::filesystem::exists(optim_path) ||
      std::filesystem::exists(manifest_path)) {
    SpielFatalError("init_mode=random but checkpoint or manifest file already exists. Refusing to overwrite.");
  }
  std::cout << "Starting fresh training run (init_mode=random).\n";
}

void RunBootstrapInit(PpoTrainRunContext& ctx) {
  std::string model_path = absl::GetFlag(FLAGS_model_checkpoint);
  std::string optim_path = absl::GetFlag(FLAGS_optim_checkpoint);
  if (model_path.find("artifacts/baselines") != std::string::npos ||
      model_path.find("artifacts/branch_a_frozen") != std::string::npos) {
    SpielFatalError("Refusing to bootstrap directly inside the frozen baseline or branch_a_frozen directory: " + model_path +
                    "\nRun bootstrap beside a BRANCH COPY in a separate output directory.");
  }
  std::filesystem::path m_path = model_path;
  m_path.replace_extension(".json");
  std::string manifest_path = m_path.string();

  // If active bundle was left in a mixed/interrupted state and valid .bak exists,
  // restore the clean prior bundle FIRST so retry doesn't backup mixed files!
  open_spiel::RecoverInterruptedBundleIfNeeded(model_path, optim_path, manifest_path);

  if (!std::filesystem::exists(model_path)) {
    SpielFatalError("Model file not found for bootstrap: " + model_path);
  }
  if (!absl::GetFlag(FLAGS_fresh_optimizer) && !std::filesystem::exists(optim_path)) {
    SpielFatalError("Optimizer file not found for bootstrap: " + optim_path);
  }

  try {
    LoadModelCheckpoint(ctx.training_model, model_path, ctx.device);
    ctx.training_model->deck_pool_semantics_ = ctx.resolved_target_deck_pool_semantics;
    if (ctx.anchor_model) {
      LoadModelCheckpoint(ctx.anchor_model, model_path, ctx.device);
      ctx.anchor_model->deck_pool_semantics_ = ctx.resolved_target_deck_pool_semantics;
      ctx.anchor_model->eval();
      for (auto& p : ctx.anchor_model->parameters()) {
        p.set_requires_grad(false);
      }
      std::cout << "[INFO] Loaded anchor model for policy KL penalty.\n";
    }
    if (absl::GetFlag(FLAGS_fresh_optimizer)) {
      std::cout << "[INFO] fresh_optimizer is true: keeping fresh AdamW optimizer. Skipping baseline optimizer load.\n";
    } else if (!absl::GetFlag(FLAGS_train_value_only)) {
      if (!open_spiel::LoadOptimizerCheckpointMigrating(
              ctx.training_model, *ctx.optimizer, optim_path, ctx.device)) {
        torch::load(*ctx.optimizer, optim_path, ctx.device);
        if (ctx.training_model->semantic_descriptor_schema_ ==
            dune_semantic::kDescriptorSchemaVersionV5) {
          open_spiel::ZeroDynamicSlotPolicyHeadMoments(*ctx.optimizer, ctx.training_model);
        }
      }
    } else {
      std::cout << "[INFO] train_value_only is true: using fresh value-only optimizer (skipping baseline optimizer load).\n";
    }
  } catch (const c10::Error& e) {
    SpielFatalError("LibTorch load failed during bootstrap: " + std::string(e.msg()));
  }

  std::string model_tmp = model_path + ".tmp";
  std::string optim_tmp = optim_path + ".tmp";

  torch::save(ctx.training_model, model_tmp);
  if (ctx.optimizer) {
    SetOptimizerLearningRate(
        *ctx.optimizer, absl::GetFlag(FLAGS_learning_rate), absl::GetFlag(FLAGS_new_block_lr));
    for (size_t g = 0; g < ctx.optimizer->param_groups().size(); ++g) {
      auto& options =
          static_cast<torch::optim::AdamWOptions&>(
              ctx.optimizer->param_groups()[g].options());
      options.weight_decay(g == 0 ? absl::GetFlag(FLAGS_policy_weight_decay)
                          : g == 1 ? absl::GetFlag(FLAGS_weight_decay)
                                   : 0.0);
    }
    for (size_t g = 0; g < ctx.optimizer->param_groups().size(); ++g) {
      const auto& options =
          static_cast<const torch::optim::AdamWOptions&>(
              ctx.optimizer->param_groups()[g].options());
      const double expected_lr = GetExpectedGroupLearningRate(
          g, ctx.optimizer->param_groups().size(),
          absl::GetFlag(FLAGS_learning_rate),
          absl::GetFlag(FLAGS_new_block_lr));
      if (std::abs(options.lr() - expected_lr) > 1e-12) {
        SpielFatalError(absl::StrFormat(
            "Optimizer parameter group %zu effective learning rate mismatch during bootstrap: expected %.6e, got %.6e",
            g, expected_lr, options.lr()));
      }
    }
    std::cout << absl::StrFormat(
        "[INFO] Verified effective learning rate in all %zu optimizer parameter groups during bootstrap\n",
        ctx.optimizer->param_groups().size());
    PrintOptimizerGroupSummary(*ctx.optimizer, ctx.training_model);
    torch::save(*ctx.optimizer, optim_tmp);
  }

  size_t model_size = 0;
  std::string model_hash = open_spiel::ComputeFileSHA256(model_tmp, &model_size);
  size_t optim_size = 0;
  std::string optim_hash = "";
  if (ctx.optimizer && std::filesystem::exists(optim_tmp)) {
    optim_hash = open_spiel::ComputeFileSHA256(optim_tmp, &optim_size);
  }

  std::string manifest_tmp = manifest_path + ".tmp";

  json::Object manifest_obj;
  manifest_obj["schema_version"] = static_cast<int64_t>(2);
  manifest_obj["global_update"] = static_cast<int64_t>(absl::GetFlag(FLAGS_start_update));
  manifest_obj["target_end_update"] = static_cast<int64_t>(absl::GetFlag(FLAGS_target_end_update));
  manifest_obj["total_env_steps"] = static_cast<int64_t>(absl::GetFlag(FLAGS_start_env_steps));
  manifest_obj["next_episode_id"] = static_cast<int64_t>(absl::GetFlag(FLAGS_start_episode_id));
  manifest_obj["base_seed"] = static_cast<int64_t>(absl::GetFlag(FLAGS_seed));
  manifest_obj["seed_scheme_version"] = static_cast<int64_t>(absl::GetFlag(FLAGS_seed_scheme_version));
  manifest_obj["config_fingerprint"] = ctx.config_fingerprint;
  WritePpoPrecisionManifestFields(
      manifest_obj, absl::GetFlag(FLAGS_rollout_amp),
      absl::GetFlag(FLAGS_allow_tf32));
  manifest_obj["train_amp"] = absl::GetFlag(FLAGS_train_amp);
  manifest_obj["search_label_fingerprint"] = ctx.search_label_fingerprint;
  manifest_obj["run_uuid"] = GenerateUUID();
  manifest_obj["checkpoint_uuid"] = GenerateUUID();
  manifest_obj["model_filename"] = std::filesystem::path(model_path).filename().string();
  manifest_obj["model_file_size"] = static_cast<int64_t>(model_size);
  manifest_obj["model_sha256"] = model_hash;
  manifest_obj["optimizer_filename"] = std::filesystem::path(optim_path).filename().string();
  manifest_obj["optimizer_file_size"] = static_cast<int64_t>(optim_size);
  manifest_obj["optimizer_sha256"] = optim_hash;
  manifest_obj["hidden_dim"] = static_cast<int64_t>(absl::GetFlag(FLAGS_hidden_dim));
  manifest_obj["num_blocks"] = static_cast<int64_t>(absl::GetFlag(FLAGS_num_blocks));
  manifest_obj["observation_dim"] = static_cast<int64_t>(ctx.training_model->input_layer->weight.size(1));
  manifest_obj["market_appendix_mode"] = absl::GetFlag(FLAGS_market_appendix_mode);
  const auto resolved_deck_pool_semantics =
      dune_imperium::ParseDeckPoolSemanticsOrDie(absl::GetFlag(FLAGS_deck_pool_semantics));
  manifest_obj["deck_pool_semantics"] =
      dune_imperium::DeckPoolSemanticsToString(resolved_deck_pool_semantics);
  const auto schema = dune_imperium::GetInformationStateSchema(
      dune_imperium::ParseMarketAppendixMode(absl::GetFlag(FLAGS_market_appendix_mode)),
      resolved_deck_pool_semantics);
  if (static_cast<int64_t>(ctx.training_model->input_layer->weight.size(1)) == schema.size) {
    manifest_obj["feature_schema_label"] = std::string(schema.label);
    manifest_obj["feature_schema_sha256"] = std::string(schema.sha256);
  }
  manifest_obj["enable_semantic_scorer"] = absl::GetFlag(FLAGS_enable_semantic_scorer);
  if (absl::GetFlag(FLAGS_enable_semantic_scorer)) {
    manifest_obj["semantic_descriptor_schema"] = ctx.training_model->semantic_descriptor_schema_;
    if (ctx.training_model->semantic_scorer_ != nullptr &&
        ctx.training_model->semantic_scorer_->slot_indexed_f_) {
      manifest_obj["slot_indexed_out_layer_f"] = true;
    }
  }
  manifest_obj["enable_zone_encoder"] = ctx.training_model->enable_zone_encoder_;
  manifest_obj["legacy_migration_provenance"] = "Synthesized via init_mode=bootstrap";
  manifest_obj["random_leader_draft"] = absl::GetFlag(FLAGS_random_leader_draft);

  {
    std::ofstream ofs(manifest_tmp);
    if (!ofs) {
      SpielFatalError("Could not open manifest temp file for writing: " + manifest_tmp);
    }
    ofs << json::ToString(manifest_obj, true);
  }
  // Coherent publication: preserve prior valid bundle under coherent names until manifest commit.
  std::string model_bak = model_path + ".bak";
  std::string optim_bak = optim_path + ".bak";
  std::string manifest_bak = manifest_path + ".bak";

  // 1. Create backups first with strict error checking.
  std::error_code ec;
  if (std::filesystem::exists(model_path, ec)) {
    std::filesystem::copy_file(model_path, model_bak,
                               std::filesystem::copy_options::overwrite_existing, ec);
    if (ec) {
      SpielFatalError("Failed to create model backup (" + model_path + " -> " + model_bak + "): " + ec.message());
    }
  }
  if (ctx.optimizer && std::filesystem::exists(optim_path, ec)) {
    std::filesystem::copy_file(optim_path, optim_bak,
                               std::filesystem::copy_options::overwrite_existing, ec);
    if (ec) {
      SpielFatalError("Failed to create optimizer backup (" + optim_path + " -> " + optim_bak + "): " + ec.message());
    }
  }
  if (std::filesystem::exists(manifest_path, ec)) {
    std::filesystem::copy_file(manifest_path, manifest_bak,
                               std::filesystem::copy_options::overwrite_existing, ec);
    if (ec) {
      SpielFatalError("Failed to create manifest backup (" + manifest_path + " -> " + manifest_bak + "): " + ec.message());
    }
  }

  // 2. Verify that the newly created backup bundle is valid before touching active files!
  if (std::filesystem::exists(manifest_bak, ec)) {
    std::string backup_err;
    if (!open_spiel::CheckBundleIntegrity(manifest_bak, model_bak, ctx.optimizer ? optim_bak : "", &backup_err)) {
      SpielFatalError("Backup bundle integrity check failed prior to artifact replacement: " + backup_err);
    }
  }

  // 3. Atomically replace active artifacts: model and optimizer first, manifest last.
  try {
    std::filesystem::rename(model_tmp, model_path);
    if (ctx.optimizer && std::filesystem::exists(optim_tmp, ec)) {
      std::filesystem::rename(optim_tmp, optim_path);
    }
    std::filesystem::rename(manifest_tmp, manifest_path);
  } catch (const std::exception& e) {
    std::string rollback_err;
    std::error_code rb_ec;
    bool rb_failed = false;

    auto safe_copy_rollback = [&](const std::string& src, const std::string& dst, const std::string& label) {
      if (src.empty() || dst.empty() || src == dst) return;
      std::error_code eq_ec;
      if (std::filesystem::exists(dst, eq_ec) && std::filesystem::equivalent(src, dst, eq_ec)) {
        return;
      }
      std::error_code cp_ec;
      std::filesystem::copy_file(src, dst, std::filesystem::copy_options::overwrite_existing, cp_ec);
      if (cp_ec) {
        rollback_err += " [" + label + " rollback failed: " + cp_ec.message() + "]";
        rb_failed = true;
      }
    };

    if (std::filesystem::exists(model_bak, rb_ec)) {
      safe_copy_rollback(model_bak, model_path, "model");
    }
    if (ctx.optimizer && std::filesystem::exists(optim_bak, rb_ec)) {
      safe_copy_rollback(optim_bak, optim_path, "optim");
    }
    if (std::filesystem::exists(manifest_bak, rb_ec)) {
      safe_copy_rollback(manifest_bak, manifest_path, "manifest");
    }
    std::string restored_verify_err;
    if (!open_spiel::CheckBundleIntegrity(manifest_path, model_path, ctx.optimizer ? optim_path : "", &restored_verify_err)) {
      rollback_err += " [restored active bundle failed verification: " + restored_verify_err + "]";
      rb_failed = true;
    }
    if (rb_failed) {
      SpielFatalError("Bootstrap publication failed (" + std::string(e.what()) +
                      ") AND rollback to backup failed:" + rollback_err);
    }
    if (std::filesystem::exists(model_tmp, rb_ec)) std::filesystem::remove(model_tmp, rb_ec);
    if (std::filesystem::exists(optim_tmp, rb_ec)) std::filesystem::remove(optim_tmp, rb_ec);
    if (std::filesystem::exists(manifest_tmp, rb_ec)) std::filesystem::remove(manifest_tmp, rb_ec);

    if (std::filesystem::exists(model_bak, rb_ec)) std::filesystem::remove(model_bak, rb_ec);
    if (std::filesystem::exists(optim_bak, rb_ec)) std::filesystem::remove(optim_bak, rb_ec);
    if (std::filesystem::exists(manifest_bak, rb_ec)) std::filesystem::remove(manifest_bak, rb_ec);

    SpielFatalError("Bootstrap publication failed: " + std::string(e.what()) +
                    " (active files restored from prior backup)");
  }

  // 4. Verify the newly committed active bundle before unlinking backups!
  std::string active_err;
  if (!open_spiel::CheckBundleIntegrity(manifest_path, model_path, ctx.optimizer ? optim_path : "", &active_err)) {
    SpielFatalError("Committed active bundle failed integrity verification: " + active_err);
  }

  // 5. Clean up .bak files now that publication is fully committed and verified.
  std::error_code ec_clean;
  if (std::filesystem::exists(model_bak, ec_clean)) std::filesystem::remove(model_bak, ec_clean);
  if (std::filesystem::exists(optim_bak, ec_clean)) std::filesystem::remove(optim_bak, ec_clean);
  if (std::filesystem::exists(manifest_bak, ec_clean)) std::filesystem::remove(manifest_bak, ec_clean);

  std::cout << "Successfully bootstrapped migrated model, optimizer, and manifest at " << manifest_path << "\n";
  exit(0);
}

void RunValidateLegacyInit(PpoTrainRunContext& ctx) {
  std::string model_path = absl::GetFlag(FLAGS_model_checkpoint);
  std::string optim_path = absl::GetFlag(FLAGS_optim_checkpoint);
  std::filesystem::path manifest_path;
  std::string artifact_manifest_flag = absl::GetFlag(FLAGS_artifact_manifest);
  if (!artifact_manifest_flag.empty()) {
    manifest_path = artifact_manifest_flag;
  } else {
    manifest_path = std::filesystem::path(model_path).parent_path() / "manifest.json";
  }

  if (!std::filesystem::exists(manifest_path)) {
    SpielFatalError("init_mode=validate_legacy but Phase 1 manifest file not found at: " + manifest_path.string());
  }

  std::ifstream ifs(manifest_path.string());
  if (!ifs) {
    SpielFatalError("Could not open manifest file at: " + manifest_path.string());
  }
  std::string content((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
  auto val_opt = open_spiel::json::FromString(content);
  if (!val_opt.has_value() || !val_opt->IsObject()) {
    SpielFatalError("Manifest file is malformed JSON at: " + manifest_path.string());
  }
  const auto& manifest_obj = val_opt->GetObject();

  auto get_object_field = [&](const std::string& key) -> const json::Object& {
    auto it = manifest_obj.find(key);
    if (it == manifest_obj.end() || !it->second.IsObject()) {
      SpielFatalError("Manifest missing required object field: " + key);
    }
    return it->second.GetObject();
  };

  auto get_nested_string = [](const json::Object& obj, const std::string& key) -> std::string {
    auto it = obj.find(key);
    if (it == obj.end() || !it->second.IsString()) {
      SpielFatalError("Nested object missing required string field: " + key);
    }
    return it->second.GetString();
  };

  auto get_nested_int = [](const json::Object& obj, const std::string& key) -> int64_t {
    auto it = obj.find(key);
    if (it == obj.end() || !it->second.IsInt()) {
      SpielFatalError("Nested object missing required int field: " + key);
    }
    return it->second.GetInt();
  };

  const auto& model_obj = get_object_field("model");
  const auto& optim_obj = get_object_field("optimizer");
  const auto& arch_obj = get_object_field("architecture");

  std::string expected_model_sha256 = get_nested_string(model_obj, "sha256");
  size_t expected_model_size = get_nested_int(model_obj, "size_bytes");

  std::string expected_optim_sha256 = get_nested_string(optim_obj, "sha256");
  size_t expected_optim_size = get_nested_int(optim_obj, "size_bytes");

  int64_t manifest_hidden_dim = get_nested_int(arch_obj, "hidden_dim");
  int64_t manifest_num_blocks = get_nested_int(arch_obj, "num_blocks");
  int64_t manifest_obs_size = get_nested_int(arch_obj, "observation_size");
  int64_t manifest_action_size = get_nested_int(arch_obj, "action_size");

  if (!std::filesystem::exists(model_path)) {
    SpielFatalError("Model file not found: " + model_path);
  }
  if (!std::filesystem::exists(optim_path)) {
    SpielFatalError("Optimizer file not found: " + optim_path);
  }

  size_t actual_model_size = 0;
  std::string actual_model_hash = open_spiel::ComputeFileSHA256(model_path, &actual_model_size);
  if (actual_model_size != expected_model_size) {
    SpielFatalError(absl::StrFormat("Model file size mismatch. Manifest: %d, Actual: %d", expected_model_size, actual_model_size));
  }
  if (actual_model_hash != expected_model_sha256) {
    SpielFatalError("Model SHA-256 hash mismatch. Manifest: " + expected_model_sha256 + ", Actual: " + actual_model_hash);
  }

  size_t actual_optim_size = 0;
  std::string actual_optim_hash = open_spiel::ComputeFileSHA256(optim_path, &actual_optim_size);
  if (actual_optim_size != expected_optim_size) {
    SpielFatalError(absl::StrFormat("Optimizer file size mismatch. Manifest: %d, Actual: %d", expected_optim_size, actual_optim_size));
  }
  if (actual_optim_hash != expected_optim_sha256) {
    SpielFatalError("Optimizer SHA-256 hash mismatch. Manifest: " + expected_optim_sha256 + ", Actual: " + actual_optim_hash);
  }

  if (absl::GetFlag(FLAGS_hidden_dim) != manifest_hidden_dim) {
    SpielFatalError(absl::StrFormat("hidden_dim mismatch. Flags: %d, Manifest: %d", absl::GetFlag(FLAGS_hidden_dim), manifest_hidden_dim));
  }
  if (absl::GetFlag(FLAGS_num_blocks) != manifest_num_blocks) {
    SpielFatalError(absl::StrFormat("num_blocks mismatch. Flags: %d, Manifest: %d", absl::GetFlag(FLAGS_num_blocks), manifest_num_blocks));
  }
  if (ctx.obs_size != manifest_obs_size) {
    SpielFatalError(absl::StrFormat("observation_size mismatch. Game: %d, Manifest: %d", ctx.obs_size, manifest_obs_size));
  }
  if (ctx.action_size != manifest_action_size) {
    SpielFatalError(absl::StrFormat("action_size mismatch. Game: %d, Manifest: %d", ctx.action_size, manifest_action_size));
  }

  try {
    LoadModelCheckpoint(ctx.training_model, model_path, ctx.device);
  } catch (const c10::Error& e) {
    SpielFatalError("LibTorch load failed: " + std::string(e.msg()));
  }

  int64_t actual_param_count = 0;
  for (const auto& param : ctx.training_model->parameters()) {
    actual_param_count += param.numel();
  }

  auto dummy_model = std::make_shared<open_spiel::SharedDunePolicyValueNetImpl>(
      manifest_obs_size, manifest_hidden_dim, manifest_action_size, manifest_num_blocks, absl::GetFlag(FLAGS_nonlinear_value_head));
  int64_t expected_param_count = 0;
  for (const auto& param : dummy_model->parameters()) {
    expected_param_count += param.numel();
  }

  if (actual_param_count != expected_param_count) {
    SpielFatalError(absl::StrFormat("Parameter count mismatch. Expected: %d (calculated from architecture), Actual: %d", expected_param_count, actual_param_count));
  }

  std::cout << absl::StrFormat("Legacy checkpoint validation successful!\n"
                               "  Model parameter count: %d\n"
                               "  Model size and hash match Phase 1 manifest.\n"
                               "  Optimizer size and hash match Phase 1 manifest.\n", actual_param_count);
  exit(0);
}

void RunStudyInit(PpoTrainRunContext& ctx) {
  std::string model_path = absl::GetFlag(FLAGS_model_checkpoint);
  if (!std::filesystem::exists(model_path)) {
    SpielFatalError("Model file not found for study: " + model_path);
  }
  LoadModelCheckpoint(ctx.training_model, model_path, ctx.device);
  ctx.training_model->deck_pool_semantics_ = ctx.resolved_target_deck_pool_semantics;
  if (ctx.anchor_model) {
    LoadModelCheckpoint(ctx.anchor_model, model_path, ctx.device);
    ctx.anchor_model->deck_pool_semantics_ = ctx.resolved_target_deck_pool_semantics;
    ctx.anchor_model->eval();
    for (auto& p : ctx.anchor_model->parameters()) p.set_requires_grad(false);
  }

  if (absl::GetFlag(FLAGS_separate_actor_critic) && ctx.training_critic != nullptr) {
    std::string critic_path = absl::GetFlag(FLAGS_critic_checkpoint);
    if (critic_path.empty()) critic_path = model_path;
    if (!std::filesystem::exists(critic_path)) {
      SpielFatalError("Critic file not found for study: " + critic_path);
    }
    LoadModelCheckpoint(ctx.training_critic, critic_path, ctx.device);
    ctx.training_critic->deck_pool_semantics_ = ctx.resolved_target_deck_pool_semantics;

    for (auto& item : ctx.training_model->named_parameters()) {
      if (item.key().find("value_head") != std::string::npos) {
        item.value().set_requires_grad(false);
      }
    }
    for (auto& item : ctx.training_critic->named_parameters()) {
      if (item.key().rfind("policy_head", 0) == 0) {
        item.value().set_requires_grad(false);
      }
    }
  }

  std::filesystem::path m_path = model_path;
  m_path.replace_extension(".json");
  if (std::filesystem::exists(m_path)) {
    std::ifstream m_in(m_path);
    std::string m_text((std::istreambuf_iterator<char>(m_in)),
                       std::istreambuf_iterator<char>());
    auto m_json = json::FromString(m_text);
    if (m_json.has_value() && m_json->IsObject()) {
      auto obj = m_json->GetObject();
      if (obj.find("total_env_steps") != obj.end() && obj["total_env_steps"].IsInt()) {
        ctx.total_env_steps.store(obj["total_env_steps"].GetInt());
      }
      if (obj.find("next_episode_id") != obj.end() && obj["next_episode_id"].IsInt()) {
        ctx.next_episode_id.store(obj["next_episode_id"].GetInt());
      }
      if (obj.find("global_update") != obj.end() && obj["global_update"].IsInt()) {
        ctx.start_update = obj["global_update"].GetInt() + 1;
      }
    }
  }

  int start_override = absl::GetFlag(FLAGS_start_update_override);
  if (start_override > 0) {
    ctx.start_update = start_override;
  }
  int study_n_updates = absl::GetFlag(FLAGS_study_updates);
  if (study_n_updates > 0) {
    ctx.target_end_update = ctx.start_update + study_n_updates - 1;
  }

  std::cout << "[STUDY] Starting controlled study (separate="
            << (absl::GetFlag(FLAGS_separate_actor_critic) ? "true" : "false")
            << ") from update " << ctx.start_update << " to " << ctx.target_end_update
            << " (" << (ctx.target_end_update - ctx.start_update + 1) << " updates).\n";
}

}  // namespace

void RunBootstrapPhase(PpoTrainRunContext& ctx) {
  if (ctx.init_mode == "random") {
    RunRandomInit(ctx);
  } else if (ctx.init_mode == "bootstrap") {
    RunBootstrapInit(ctx);
  } else if (ctx.init_mode == "validate_legacy") {
    RunValidateLegacyInit(ctx);
  } else if (ctx.init_mode == "study") {
    RunStudyInit(ctx);
  } else {
    SpielFatalError("Unsupported init_mode: " + ctx.init_mode +
                    " (expected random, checkpoint, bootstrap, "
                    "validate_legacy, diagnostic, study, or study_resume)");
  }
}

#endif  // OPEN_SPIEL_BUILD_WITH_LIBTORCH

}  // namespace open_spiel::ppo_train_internal
