#include "dune_ppo_train_run_context.h"

#include "open_spiel/abseil-cpp/absl/flags/reflection.h"

namespace open_spiel::ppo_train_internal {

#ifdef OPEN_SPIEL_BUILD_WITH_LIBTORCH

namespace {

std::string ResolvePermittedTransitionFingerprint(
    const std::string& manifest_path,
    const std::string& config_fingerprint,
    bool allow_nested_semantic_in_market) {
  std::string fp = ValidateAndComputeRewardTransitionFingerprint(
      manifest_path, config_fingerprint);
  if (!fp.empty()) {
    g_reward_transition_source_fingerprint = fp;
    return fp;
  }
  fp = ValidateAndComputeGaeTransitionFingerprint(
      manifest_path, config_fingerprint);
  if (!fp.empty()) {
    g_gae_transition_source_fingerprint = fp;
    return fp;
  }
  fp = ValidateAndComputeLearningRateTransitionFingerprint(
      manifest_path, config_fingerprint);
  if (!fp.empty()) {
    g_learning_rate_transition_source_fingerprint = fp;
    return fp;
  }
  fp = ValidateAndComputeTargetKlTransitionFingerprint(
      manifest_path, config_fingerprint);
  if (!fp.empty()) {
    g_target_kl_transition_source_fingerprint = fp;
    return fp;
  }
  fp = ValidateAndComputeSingleLearnerTransitionFingerprint(
      manifest_path, config_fingerprint);
  if (!fp.empty()) {
    g_single_learner_transition_source_fingerprint = fp;
    return fp;
  }
  fp = ValidateAndComputeEmaMagnetTransitionFingerprint(
      manifest_path, config_fingerprint);
  if (!fp.empty()) {
    g_ema_magnet_transition_source_fingerprint = fp;
    return fp;
  }
  fp = ValidateAndComputePoolTransitionFingerprint(
      manifest_path, config_fingerprint);
  if (!fp.empty()) {
    g_pool_transition_source_fingerprint = fp;
    return fp;
  }
  fp = ValidateAndComputeMarketAppendixTransitionFingerprint(
      manifest_path, config_fingerprint);
  if (!fp.empty()) {
    g_market_appendix_transition_source_fingerprint = fp;
    if (allow_nested_semantic_in_market &&
        absl::GetFlag(FLAGS_allow_semantic_schema_transition)) {
      g_semantic_schema_transition_source_fingerprint = fp;
    }
    return fp;
  }
  fp = ValidateAndComputeDeckPoolSemanticsTransitionFingerprint(
      manifest_path, config_fingerprint);
  if (!fp.empty()) {
    g_deck_pool_semantics_transition_source_fingerprint = fp;
    return fp;
  }
  fp = ValidateAndComputeTf32TransitionFingerprint(
      manifest_path, config_fingerprint);
  if (!fp.empty()) {
    g_tf32_transition_source_fingerprint = fp;
    return fp;
  }
  fp = ValidateAndComputeSemanticSchemaTransitionFingerprint(
      manifest_path, config_fingerprint);
  if (!fp.empty()) {
    g_semantic_schema_transition_source_fingerprint = fp;
  }
  return fp;
}

void RecordTransitionSourceUpdate(int64_t global_update) {
  if (!g_reward_transition_source_fingerprint.empty()) {
    g_reward_transition_source_update = global_update;
  }
  if (!g_gae_transition_source_fingerprint.empty()) {
    g_gae_transition_source_update = global_update;
  }
  if (!g_learning_rate_transition_source_fingerprint.empty()) {
    g_learning_rate_transition_source_update = global_update;
  }
  if (!g_target_kl_transition_source_fingerprint.empty()) {
    g_target_kl_transition_source_update = global_update;
  }
  if (!g_single_learner_transition_source_fingerprint.empty()) {
    g_single_learner_transition_source_update = global_update;
  }
  if (!g_ema_magnet_transition_source_fingerprint.empty()) {
    g_ema_magnet_transition_source_update = global_update;
  }
  if (!g_pool_transition_source_fingerprint.empty()) {
    g_pool_transition_source_update = global_update;
  }
  if (!g_market_appendix_transition_source_fingerprint.empty()) {
    g_market_appendix_transition_source_update = global_update;
  }
  if (!g_deck_pool_semantics_transition_source_fingerprint.empty()) {
    g_deck_pool_semantics_transition_source_update = global_update;
  }
  if (!g_tf32_transition_source_fingerprint.empty()) {
    g_tf32_transition_source_update = global_update;
  }
  if (!g_semantic_schema_transition_source_fingerprint.empty()) {
    g_semantic_schema_transition_source_update = global_update;
  }
}

std::string DeriveSiblingCheckpointPath(
    const std::string& base_path,
    const std::string& base_update_token,
    const std::string& base_final_token,
    const std::string& target_update_token,
    const std::string& target_final_token) {
  if (base_path.empty()) return "";
  std::filesystem::path p(base_path);
  std::string filename = p.filename().string();
  size_t pos = filename.rfind(base_update_token);
  if (pos != std::string::npos) {
    filename.replace(pos, base_update_token.size(), target_update_token);
    return (p.parent_path() / filename).string();
  }
  pos = filename.rfind(base_final_token);
  if (pos != std::string::npos) {
    filename.replace(pos, base_final_token.size(), target_final_token);
    return (p.parent_path() / filename).string();
  }
  return "";
}

void ApplyAndVerifyOptimizerConfig(
    torch::optim::AdamW& optimizer,
    const std::shared_ptr<SharedDunePolicyValueNetImpl>& training_model,
    torch::optim::AdamW* critic_optimizer,
    bool print_summary) {
  SetOptimizerLearningRate(
      optimizer, absl::GetFlag(FLAGS_learning_rate), absl::GetFlag(FLAGS_new_block_lr));
  if (critic_optimizer != nullptr) {
    SetOptimizerLearningRate(*critic_optimizer, absl::GetFlag(FLAGS_learning_rate));
  }
  for (size_t g = 0; g < optimizer.param_groups().size(); ++g) {
    auto& options =
        static_cast<torch::optim::AdamWOptions&>(
            optimizer.param_groups()[g].options());
    // PWO-5 section 7.4: group 2 is the auxiliary-head group and must stay
    // at weight_decay = 0.0.
    options.weight_decay(g == 0 ? absl::GetFlag(FLAGS_policy_weight_decay)
                        : g == 1 ? absl::GetFlag(FLAGS_weight_decay)
                                 : 0.0);
  }
  for (size_t g = 0; g < optimizer.param_groups().size(); ++g) {
    const auto& options =
        static_cast<const torch::optim::AdamWOptions&>(
            optimizer.param_groups()[g].options());
    const double expected_lr = GetExpectedGroupLearningRate(
        g, optimizer.param_groups().size(),
        absl::GetFlag(FLAGS_learning_rate),
        absl::GetFlag(FLAGS_new_block_lr));
    if (std::abs(options.lr() - expected_lr) > 1e-12) {
      SpielFatalError(absl::StrFormat(
          "Optimizer parameter group %zu effective learning rate mismatch: expected %.6e, got %.6e",
          g, expected_lr, options.lr()));
    }
  }
  std::cout << absl::StrFormat(
      "[INFO] Verified effective learning rate in all %zu optimizer parameter groups\n",
      optimizer.param_groups().size());
  if (print_summary) {
    PrintOptimizerGroupSummary(optimizer, training_model);
  }
}

void MoveAdamWStateToDevice(torch::optim::AdamW& opt, const torch::Device& device) {
  for (auto& pair : opt.state()) {
    if (auto* state =
            dynamic_cast<torch::optim::AdamWParamState*>(pair.second.get())) {
      if (state->exp_avg().defined()) state->exp_avg() = state->exp_avg().to(device);
      if (state->exp_avg_sq().defined()) state->exp_avg_sq() = state->exp_avg_sq().to(device);
      if (state->max_exp_avg_sq().defined()) {
        state->max_exp_avg_sq() = state->max_exp_avg_sq().to(device);
      }
    }
  }
}

void RunDiagnosticResume(PpoTrainRunContext& ctx) {
  if (!ctx.numerical_parity) {
    SpielFatalError(
        "init_mode=diagnostic is reserved for numerical parity");
  }
  const std::string model_path = absl::GetFlag(FLAGS_model_checkpoint);
  ctx.parity_source_manifest = absl::GetFlag(FLAGS_artifact_manifest);
  if (!std::filesystem::is_regular_file(model_path) ||
      !std::filesystem::is_regular_file(ctx.parity_source_manifest)) {
    SpielFatalError(
        "Diagnostic model checkpoint and provenance manifest must both be regular files");
  }
  size_t manifest_size = 0;
  ctx.parity_source_manifest_sha256 = ComputeFileSHA256(
      ctx.parity_source_manifest, &manifest_size);
  std::ifstream source_stream(ctx.parity_source_manifest);
  const std::string source_text((std::istreambuf_iterator<char>(source_stream)),
                                std::istreambuf_iterator<char>());
  auto parsed = json::FromString(source_text);
  if (!parsed.has_value() || !parsed->IsObject()) {
    SpielFatalError("Numerical parity provenance manifest is malformed JSON");
  }
  const json::Object& source = parsed->GetObject();
  auto required_string = [&](const char* key) -> std::string {
    auto it = source.find(key);
    if (it == source.end() || !it->second.IsString()) {
      SpielFatalError(std::string("Numerical parity manifest missing string '") +
                      key + "'");
    }
    return it->second.GetString();
  };
  auto required_int = [&](const char* key) -> int64_t {
    auto it = source.find(key);
    if (it == source.end() || !it->second.IsInt()) {
      SpielFatalError(std::string("Numerical parity manifest missing integer '") +
                      key + "'");
    }
    return it->second.GetInt();
  };
  size_t model_size = 0;
  const std::string model_sha256 = ComputeFileSHA256(model_path, &model_size);
  if (model_sha256 != required_string("model_sha256") ||
      static_cast<int64_t>(model_size) != required_int("model_file_size")) {
    SpielFatalError(
        "Numerical parity model bytes do not match the provenance manifest");
  }
  if (absl::GetFlag(FLAGS_hidden_dim) != required_int("hidden_dim") ||
      absl::GetFlag(FLAGS_num_blocks) != required_int("num_blocks")) {
    SpielFatalError(
        "Numerical parity architecture flags do not match the provenance manifest");
  }
  ctx.parity_source_global_update = required_int("global_update");
  ctx.parity_source_checkpoint_uuid = required_string("checkpoint_uuid");
  LoadModelCheckpoint(ctx.training_model, model_path, ctx.device);
  ctx.start_update = static_cast<int>(ctx.parity_source_global_update + 1);
  ctx.next_episode_id.store(
      static_cast<uint64_t>(absl::GetFlag(FLAGS_start_episode_id)));
  ctx.total_env_steps.store(
      static_cast<uint64_t>(absl::GetFlag(FLAGS_start_env_steps)));
  std::cout << "Loaded read-only numerical-parity source model "
            << model_path << " sha256=" << model_sha256
            << "; optimizer checkpoint was not loaded.\n";
}

void RunCheckpointResume(PpoTrainRunContext& ctx) {
  if (absl::GetFlag(FLAGS_pipeline)) {
    SpielFatalError("init_mode=checkpoint with pipeline=true is rejected because prefetched rollout is not persisted.");
  }
  std::string model_path = absl::GetFlag(FLAGS_model_checkpoint);
  std::string optim_path = absl::GetFlag(FLAGS_optim_checkpoint);
  std::filesystem::path m_path = model_path;
  m_path.replace_extension(".json");
  std::string manifest_path = m_path.string();

  // WO-20 acceptance-contract guard, deliberately BEFORE the fingerprint check.
  if (absl::GetFlag(FLAGS_online_search_collection)) {
    OnlineCollectionState prior_state;
    std::string oc_probe_err;
    if (ReadOnlineCollectionState(manifest_path, prior_state, oc_probe_err) &&
        prior_state.present) {
      const std::string effective =
          AcceptancePriorSourceName(EffectiveAcceptancePriorSource());
      const bool stated = !absl::GetFlag(FLAGS_collector_acceptance_prior).empty();
      if (prior_state.acceptance_prior_source.empty() && !stated) {
        SpielFatalError(absl::StrCat(
            "Refusing to resume: ", manifest_path, " predates WO-20 and has "
            "no online_collection.acceptance_prior_source, so its "
            "cum_accepted=", prior_state.cum_accepted, " / cum_rejected=",
            prior_state.cum_rejected, " were accumulated against the "
            "POST-NOISE tree prior. Continuing under the new default ('",
            effective, "') would mix two coverage contracts in one counter "
            "set and one accepted-target hash chain.\n"
            "  To continue that run's original contract: "
            "--collector_acceptance_prior=tree_prior\n"
            "  To adopt the new contract knowingly (counters and hash chain "
            "then span both): --collector_acceptance_prior=", effective, "\n"
            "  Note either choice changes the config fingerprint relative to "
            "the stored one, which is intended — the semantics did change."));
      }
      if (!prior_state.acceptance_prior_source.empty() &&
          prior_state.acceptance_prior_source != effective) {
        SpielFatalError(absl::StrCat(
            "Refusing to resume: ", manifest_path, " collected under "
            "acceptance_prior_source='", prior_state.acceptance_prior_source,
            "' but this run resolves to '", effective, "'. Its counters and "
            "hash chain cannot be extended across a change of coverage "
            "contract. Pass --collector_acceptance_prior=",
            prior_state.acceptance_prior_source, " to continue it, or start "
            "a fresh run."));
      }
    }
  }

  CheckpointManifest manifest;
  std::string err;

  open_spiel::RecoverInterruptedBundleIfNeeded(model_path, optim_path, manifest_path);

  std::string permitted_transition_source_fingerprint =
      ResolvePermittedTransitionFingerprint(
          manifest_path, ctx.config_fingerprint, /*allow_nested_semantic_in_market=*/true);

  std::string legacy_fingerprint = absl::GetFlag(FLAGS_train_value_only)
      ? ""
      : ComputeLegacyConfigFingerprint();
  if (!ParseAndValidateManifest(manifest_path, model_path, optim_path,
                                ctx.master, ctx.target_end_update,
                                absl::GetFlag(FLAGS_seed_scheme_version),
                                ctx.config_fingerprint,
                                ctx.pre_precision_config_fingerprint,
                                absl::GetFlag(FLAGS_rollout_amp),
                                absl::GetFlag(FLAGS_allow_tf32),
                                ctx.search_label_fingerprint,
                                absl::GetFlag(FLAGS_hidden_dim),
                                absl::GetFlag(FLAGS_num_blocks),
                                manifest, err, legacy_fingerprint,
                                permitted_transition_source_fingerprint)) {
    SpielFatalError(err);
  }
  if (!permitted_transition_source_fingerprint.empty()) {
    RecordTransitionSourceUpdate(manifest.global_update);
  }

  // Verify market_appendix_mode matches the current run's flag and capture
  // optional critic filenames if present in the manifest JSON.
  std::string manifest_critic_model_filename;
  std::string manifest_critic_optim_filename;
  {
    std::ifstream mfs(manifest_path);
    std::string content((std::istreambuf_iterator<char>(mfs)),
                        std::istreambuf_iterator<char>());
    auto parsed = open_spiel::json::FromString(content);
    if (parsed && parsed->IsObject()) {
      const auto& obj = parsed->GetObject();
      auto cm_it = obj.find("critic_model_filename");
      if (cm_it != obj.end() && cm_it->second.IsString()) {
        manifest_critic_model_filename = cm_it->second.GetString();
      }
      auto co_it = obj.find("critic_optimizer_filename");
      if (co_it != obj.end() && co_it->second.IsString()) {
        manifest_critic_optim_filename = co_it->second.GetString();
      }
      auto it = obj.find("market_appendix_mode");
      std::string manifest_mode = "none";
      if (it != obj.end() && it->second.IsString()) {
        manifest_mode = it->second.GetString();
      }
      std::string current_mode = absl::GetFlag(FLAGS_market_appendix_mode);
      if (manifest_mode != current_mode) {
        if (!absl::GetFlag(FLAGS_allow_market_appendix_transition)) {
          SpielFatalError(absl::StrFormat(
              "market_appendix_mode mismatch on resume. Manifest: '%s', Flag: '%s'",
              manifest_mode, current_mode));
        }
      }
      open_spiel::ValidateResumeDeckPoolSemanticsOrDie(
          obj, absl::GetFlag(FLAGS_deck_pool_semantics), "resume");
      auto sem_it = obj.find("semantic_descriptor_schema");
      std::string manifest_schema = "semantic_action_v3";
      if (sem_it != obj.end() && sem_it->second.IsString()) {
        manifest_schema = sem_it->second.GetString();
      }
      std::string current_schema = absl::GetFlag(FLAGS_semantic_descriptor_schema);
      if (manifest_schema != current_schema) {
        if (!absl::GetFlag(FLAGS_allow_semantic_schema_transition)) {
          SpielFatalError(absl::StrFormat(
              "semantic_descriptor_schema mismatch on resume. Manifest: '%s', Flag: '%s'",
              manifest_schema, current_schema));
        }
      }
    }
  }

  int global_update = manifest.global_update;
  if (global_update >= ctx.target_end_update) {
    SpielFatalError(absl::StrFormat(
        "global_update (%d) >= target_end_update (%d). "
        "Resuming is not possible as training is already finished.",
        global_update, ctx.target_end_update));
  }
  uint64_t manifest_env_steps = manifest.total_env_steps;
  uint64_t manifest_episode_id = manifest.next_episode_id;
  std::string manifest_run_uuid = manifest.run_uuid;

  // Phase 18B exact resume
  {
    open_spiel::OnlineCollectionState restored;
    std::string oc_err;
    if (!open_spiel::ReadOnlineCollectionState(manifest_path, restored, oc_err)) {
      SpielFatalError("Failed to read online_collection manifest state: " + oc_err);
    }
    if (restored.present) {
      ctx.aux_next_episode_id_persist = restored.next_auxiliary_episode_id;
      ctx.aux_cum_accepted = restored.cum_accepted;
      ctx.aux_cum_rejected = restored.cum_rejected;
      ctx.aux_cum_granted = restored.cum_granted;
      ctx.aux_cum_organic = restored.cum_organic;
      for (int r = 0; r < open_spiel::OnlineCollectionState::kNumSearchRoles;
           ++r) {
        ctx.aux_cum_role_searches[r] = restored.cum_role_searches[r];
        ctx.aux_cum_role_accepted[r] = restored.cum_role_accepted[r];
      }
      ctx.aux_hash_chain = restored.accepted_hash_chain;
      std::cout << absl::StrFormat(
          "[18B] Restored online-collection state: next_aux_episode_id=%llu "
          "cum_accepted=%lld cum_granted=%lld cum_organic=%lld\n",
          (unsigned long long)ctx.aux_next_episode_id_persist,
          (long long)ctx.aux_cum_accepted, (long long)ctx.aux_cum_granted,
          (long long)ctx.aux_cum_organic);
    }
  }

  try {
    LoadModelCheckpoint(ctx.training_model, model_path, ctx.device);
    if (absl::GetFlag(FLAGS_enable_ema_tracking) && ctx.ema_model != nullptr) {
      std::string ema_path = absl::GetFlag(FLAGS_ema_model_checkpoint);
      if (ema_path.empty()) {
        std::string stem_ema = DeriveSiblingCheckpointPath(
            model_path, "_model_update_", "_model_final",
            "_model_ema_update_", "_model_ema_final");
        if (!stem_ema.empty() && std::filesystem::exists(stem_ema)) {
          ema_path = stem_ema;
        } else {
          std::string legacy_ema = (std::filesystem::path(model_path).parent_path() /
              ("ppo_model_ema_update_" + std::to_string(manifest.global_update) + ".pt")).string();
          if (std::filesystem::exists(legacy_ema)) {
            ema_path = legacy_ema;
          }
        }
      }
      if (!ema_path.empty() && std::filesystem::exists(ema_path)) {
        LoadModelCheckpoint(ctx.ema_model, ema_path, ctx.device);
        std::string ema_json_path = (std::filesystem::path(ema_path).parent_path() /
            (std::filesystem::path(ema_path).stem().string() + ".json")).string();
        if (std::filesystem::exists(ema_json_path)) {
          std::ifstream ejf(ema_json_path);
          if (ejf) {
            std::string content((std::istreambuf_iterator<char>(ejf)), std::istreambuf_iterator<char>());
            auto parsed = open_spiel::json::FromString(content);
            if (parsed && parsed->IsObject()) {
              auto obj = parsed->GetObject();
              auto it_u = obj.find("ema_updates_absorbed");
              if (it_u != obj.end() && it_u->second.IsInt()) {
                ctx.ema_updates_absorbed = it_u->second.GetInt();
              }
            }
          }
        }
        std::cout << absl::StrFormat(
            "[EMA] Resumed EMA model from %s (updates_absorbed=%lld)\n",
            ema_path, (long long)ctx.ema_updates_absorbed);
      } else {
        CopyModelWeights(ctx.training_model, ctx.ema_model);
        ctx.ema_updates_absorbed = 0;
        std::cout << "[EMA] No prior EMA checkpoint found; initialized EMA weights from training model.\n";
      }
    }
    if (ctx.anchor_model) {
      LoadModelCheckpoint(ctx.anchor_model, model_path, ctx.device);
      ctx.anchor_model->eval();
      for (auto& p : ctx.anchor_model->parameters()) {
        p.set_requires_grad(false);
      }
      std::cout << "[INFO] Loaded anchor model for policy KL penalty.\n";
    }
    if (absl::GetFlag(FLAGS_separate_actor_critic) && ctx.training_critic != nullptr) {
      std::string critic_path = absl::GetFlag(FLAGS_critic_checkpoint);
      if (critic_path.empty()) {
        const std::filesystem::path model_dir =
            std::filesystem::path(model_path).parent_path();
        if (!manifest_critic_model_filename.empty()) {
          std::string manifest_critic =
              (model_dir / manifest_critic_model_filename).string();
          if (std::filesystem::exists(manifest_critic)) {
            critic_path = manifest_critic;
          }
        }
        if (critic_path.empty()) {
          std::string stem_critic = DeriveSiblingCheckpointPath(
              model_path, "_model_update_", "_model_final",
              "_critic_model_update_", "_critic_model_final");
          if (!stem_critic.empty() && std::filesystem::exists(stem_critic)) {
            critic_path = stem_critic;
          }
        }
        if (critic_path.empty()) {
          std::string legacy_critic = (model_dir /
              ("ppo_critic_model_update_" + std::to_string(manifest.global_update) + ".pt")).string();
          if (std::filesystem::exists(legacy_critic)) {
            critic_path = legacy_critic;
          }
        }
        if (critic_path.empty()) {
          critic_path = model_path;
        }
      }
      if (!std::filesystem::exists(critic_path)) {
        SpielFatalError("Critic checkpoint not found: " + critic_path);
      }
      LoadModelCheckpoint(ctx.training_critic, critic_path, ctx.device, /*allow_skip_semantic_scorer=*/true);
      std::cout << "[INFO] Loaded critic checkpoint from " << critic_path << "\n";

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
    if (absl::GetFlag(FLAGS_fresh_optimizer)) {
      std::cout << "[INFO] fresh_optimizer is true: keeping fresh AdamW optimizer. Skipping optimizer checkpoint load.\n";
    } else if (!absl::GetFlag(FLAGS_train_value_only)) {
      if (!open_spiel::LoadOptimizerCheckpointMigrating(
              ctx.training_model, *ctx.optimizer, optim_path, ctx.device)) {
        torch::load(*ctx.optimizer, optim_path, ctx.device);
        if (ctx.training_model->semantic_descriptor_schema_ ==
            dune_semantic::kDescriptorSchemaVersionV5) {
          open_spiel::ZeroDynamicSlotPolicyHeadMoments(*ctx.optimizer, ctx.training_model);
        }
      }
      if (absl::GetFlag(FLAGS_separate_actor_critic) && ctx.critic_optimizer != nullptr) {
        std::string critic_opt_path = absl::GetFlag(FLAGS_critic_optim_checkpoint);
        if (critic_opt_path.empty()) {
          const std::filesystem::path optim_dir =
              std::filesystem::path(optim_path).parent_path();
          if (!manifest_critic_optim_filename.empty()) {
            std::string manifest_opt =
                (optim_dir / manifest_critic_optim_filename).string();
            if (std::filesystem::exists(manifest_opt)) {
              critic_opt_path = manifest_opt;
            }
          }
          if (critic_opt_path.empty()) {
            std::string stem_opt = DeriveSiblingCheckpointPath(
                optim_path, "_optimizer_update_", "_optimizer_final",
                "_critic_optimizer_update_", "_critic_optimizer_final");
            if (!stem_opt.empty() && std::filesystem::exists(stem_opt)) {
              critic_opt_path = stem_opt;
            }
          }
          if (critic_opt_path.empty()) {
            std::string legacy_opt = (optim_dir /
                ("ppo_critic_optimizer_update_" + std::to_string(manifest.global_update) + ".pt")).string();
            if (std::filesystem::exists(legacy_opt)) {
              critic_opt_path = legacy_opt;
            }
          }
        }
        if (!critic_opt_path.empty() && std::filesystem::exists(critic_opt_path)) {
          torch::load(*ctx.critic_optimizer, critic_opt_path, ctx.device);
          std::cout << "[INFO] Loaded critic optimizer from " << critic_opt_path << "\n";
        } else {
          std::cout << "[INFO] Starting fresh critic optimizer for separate actor-critic.\n";
        }
      }
    } else {
      std::cout << "[INFO] train_value_only is true: keeping fresh value-only optimizer. Skipping optimizer checkpoint load.\n";
    }
  } catch (const c10::Error& e) {
    SpielFatalError("LibTorch load failed: " + std::string(e.msg()));
  }

  ApplyAndVerifyOptimizerConfig(
      *ctx.optimizer, ctx.training_model, ctx.critic_optimizer.get(), /*print_summary=*/true);
  MoveAdamWStateToDevice(*ctx.optimizer, ctx.device);
  if (ctx.critic_optimizer != nullptr) {
    MoveAdamWStateToDevice(*ctx.critic_optimizer, ctx.device);
  }

  ctx.start_update = global_update + 1;
  ctx.total_env_steps.store(manifest_env_steps);
  ctx.next_episode_id.store(manifest_episode_id);
  ctx.run_uuid = manifest_run_uuid;

  std::cout << "Successfully verified and loaded checkpoint manifest. Resuming from update "
            << ctx.start_update << " to " << ctx.target_end_update << ".\n";
}

void RunStudyResume(PpoTrainRunContext& ctx) {
  std::string model_path = absl::GetFlag(FLAGS_model_checkpoint);
  std::string optim_path = absl::GetFlag(FLAGS_optim_checkpoint);
  std::filesystem::path m_path = model_path;
  m_path.replace_extension(".json");
  std::string manifest_path = m_path.string();

  open_spiel::RecoverInterruptedBundleIfNeeded(model_path, optim_path, manifest_path);

  if (!std::filesystem::exists(manifest_path)) {
    SpielFatalError("Study resume: manifest not found: " + manifest_path);
  }

  {
    std::ifstream m_in(manifest_path);
    std::string m_text((std::istreambuf_iterator<char>(m_in)),
                       std::istreambuf_iterator<char>());
    auto m_json = json::FromString(m_text);
    if (m_json.has_value() && m_json->IsObject()) {
      const auto& obj = m_json->GetObject();
      open_spiel::ValidateResumeDeckPoolSemanticsOrDie(
          obj, absl::GetFlag(FLAGS_deck_pool_semantics), "study_resume");
    }
  }

  if (absl::GetFlag(FLAGS_separate_actor_critic) && ctx.training_critic != nullptr) {
    std::string critic_path = absl::GetFlag(FLAGS_critic_checkpoint);
    std::string critic_opt_path = absl::GetFlag(FLAGS_critic_optim_checkpoint);

    open_spiel::DualCheckpointManifest dual_manifest;
    std::string err;
    if (!open_spiel::ParseAndValidateDualManifest(
            manifest_path, model_path, optim_path, critic_path, critic_opt_path,
            ctx.master, ctx.target_end_update,
            absl::GetFlag(FLAGS_seed_scheme_version),
            ctx.config_fingerprint,
            absl::GetFlag(FLAGS_rollout_amp),
            absl::GetFlag(FLAGS_allow_tf32),
            absl::GetFlag(FLAGS_hidden_dim),
            absl::GetFlag(FLAGS_num_blocks),
            dual_manifest, err)) {
      SpielFatalError("Study resume dual validation failed: " + err);
    }

    LoadModelCheckpoint(ctx.training_model, model_path, ctx.device);
    torch::load(*ctx.optimizer, optim_path, ctx.device);
    if (ctx.training_model->semantic_descriptor_schema_ ==
        dune_semantic::kDescriptorSchemaVersionV5) {
      open_spiel::ZeroDynamicSlotPolicyHeadMoments(*ctx.optimizer, ctx.training_model);
    }
    LoadModelCheckpoint(ctx.training_critic, critic_path, ctx.device);
    torch::load(*ctx.critic_optimizer, critic_opt_path, ctx.device);

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

    auto* te_flag = absl::FindCommandLineFlag("target_end_update");
    if (te_flag && te_flag->CurrentValue() != te_flag->DefaultValue() &&
        absl::GetFlag(FLAGS_target_end_update) != dual_manifest.target_end_update) {
      SpielFatalError(absl::StrFormat(
          "CLI --target_end_update=%d does not match manifest target_end_update=%d",
          absl::GetFlag(FLAGS_target_end_update), dual_manifest.target_end_update));
    }
    auto* seed_flag = absl::FindCommandLineFlag("seed");
    if (seed_flag && seed_flag->CurrentValue() != seed_flag->DefaultValue() &&
        absl::GetFlag(FLAGS_seed) != 0 &&
        absl::GetFlag(FLAGS_seed) != dual_manifest.base_seed) {
      SpielFatalError(absl::StrFormat(
          "CLI --seed=%llu does not match manifest base_seed=%llu",
          static_cast<unsigned long long>(absl::GetFlag(FLAGS_seed)),
          static_cast<unsigned long long>(dual_manifest.base_seed)));
    }

    ctx.start_update = dual_manifest.global_update + 1;
    ctx.total_env_steps.store(dual_manifest.total_env_steps);
    ctx.next_episode_id.store(dual_manifest.next_episode_id);
    ctx.run_uuid = dual_manifest.run_uuid;
    ctx.master = dual_manifest.base_seed;
    ctx.target_end_update = dual_manifest.target_end_update;
  } else {
    open_spiel::CheckpointManifest shared_manifest;
    std::string err;
    std::string permitted_transition_source_fingerprint =
        ResolvePermittedTransitionFingerprint(
            manifest_path, ctx.config_fingerprint, /*allow_nested_semantic_in_market=*/false);
    std::string legacy_fingerprint = absl::GetFlag(FLAGS_train_value_only)
        ? ""
        : ComputeLegacyConfigFingerprint();
    if (!open_spiel::ParseAndValidateManifest(
            manifest_path, model_path, optim_path,
            ctx.master, ctx.target_end_update,
            absl::GetFlag(FLAGS_seed_scheme_version),
            ctx.config_fingerprint,
            ctx.pre_precision_config_fingerprint,
            absl::GetFlag(FLAGS_rollout_amp),
            absl::GetFlag(FLAGS_allow_tf32),
            ctx.search_label_fingerprint,
            absl::GetFlag(FLAGS_hidden_dim),
            absl::GetFlag(FLAGS_num_blocks),
            shared_manifest, err, legacy_fingerprint,
            permitted_transition_source_fingerprint)) {
      SpielFatalError("Study resume shared validation failed: " + err);
    }
    if (!permitted_transition_source_fingerprint.empty()) {
      RecordTransitionSourceUpdate(shared_manifest.global_update);
    }

    // Verify market_appendix_mode and feature_schema match the current run's flags
    {
      std::ifstream mfs(manifest_path);
      std::string content((std::istreambuf_iterator<char>(mfs)),
                          std::istreambuf_iterator<char>());
      auto parsed = open_spiel::json::FromString(content);
      if (parsed && parsed->IsObject()) {
        const auto& obj = parsed->GetObject();
        auto it = obj.find("market_appendix_mode");
        std::string manifest_mode = "none";
        if (it != obj.end() && it->second.IsString()) {
          manifest_mode = it->second.GetString();
        }
        std::string current_mode = absl::GetFlag(FLAGS_market_appendix_mode);
        if (manifest_mode != current_mode) {
          if (!absl::GetFlag(FLAGS_allow_market_appendix_transition)) {
            SpielFatalError(absl::StrFormat(
                "market_appendix_mode mismatch on study_resume. Manifest: '%s', Flag: '%s'",
                manifest_mode, current_mode));
          }
        }
        std::string manifest_sem = open_spiel::ValidateResumeDeckPoolSemanticsOrDie(
            obj, absl::GetFlag(FLAGS_deck_pool_semantics), "study_resume");
        auto sem_it = obj.find("semantic_descriptor_schema");
        std::string manifest_schema = "semantic_action_v3";
        if (sem_it != obj.end() && sem_it->second.IsString()) {
          manifest_schema = sem_it->second.GetString();
        }
        std::string current_schema = absl::GetFlag(FLAGS_semantic_descriptor_schema);
        if (manifest_schema != current_schema) {
          if (!absl::GetFlag(FLAGS_allow_semantic_schema_transition)) {
            SpielFatalError(absl::StrFormat(
                "semantic_descriptor_schema mismatch on study_resume. Manifest: '%s', Flag: '%s'",
                manifest_schema, current_schema));
          }
        }
        if (current_mode != "none" && !absl::GetFlag(FLAGS_allow_market_appendix_transition)) {
          auto it_schema = obj.find("feature_schema_sha256");
          std::string manifest_schema_sha = (it_schema != obj.end() && it_schema->second.IsString())
              ? it_schema->second.GetString() : "";
          const auto schema = dune_imperium::GetInformationStateSchema(
              dune_imperium::ParseMarketAppendixMode(current_mode),
              dune_imperium::ParseDeckPoolSemanticsOrDie(manifest_sem));
          if (manifest_schema_sha != schema.sha256) {
            SpielFatalError(absl::StrFormat(
                "feature_schema_sha256 mismatch on study_resume. Manifest: '%s', Expected: '%s'",
                manifest_schema_sha, schema.sha256));
          }
        }
      }
    }

    LoadModelCheckpoint(ctx.training_model, model_path, ctx.device);
    if (!open_spiel::LoadOptimizerCheckpointMigrating(
            ctx.training_model, *ctx.optimizer, optim_path, ctx.device)) {
      torch::load(*ctx.optimizer, optim_path, ctx.device);
      if (ctx.training_model->semantic_descriptor_schema_ ==
          dune_semantic::kDescriptorSchemaVersionV5) {
        open_spiel::ZeroDynamicSlotPolicyHeadMoments(*ctx.optimizer, ctx.training_model);
      }
    }

    if (!absl::GetFlag(FLAGS_train_value_only) && ctx.optimizer) {
      ApplyAndVerifyOptimizerConfig(
          *ctx.optimizer, ctx.training_model, ctx.critic_optimizer.get(), /*print_summary=*/false);
    }

    auto* te_flag = absl::FindCommandLineFlag("target_end_update");
    if (te_flag && te_flag->CurrentValue() != te_flag->DefaultValue() &&
        absl::GetFlag(FLAGS_target_end_update) != shared_manifest.target_end_update) {
      SpielFatalError(absl::StrFormat(
          "CLI --target_end_update=%d does not match manifest target_end_update=%d",
          absl::GetFlag(FLAGS_target_end_update), shared_manifest.target_end_update));
    }
    auto* seed_flag = absl::FindCommandLineFlag("seed");
    if (seed_flag && seed_flag->CurrentValue() != seed_flag->DefaultValue() &&
        absl::GetFlag(FLAGS_seed) != 0 &&
        absl::GetFlag(FLAGS_seed) != shared_manifest.base_seed) {
      SpielFatalError(absl::StrFormat(
          "CLI --seed=%llu does not match manifest base_seed=%llu",
          static_cast<unsigned long long>(absl::GetFlag(FLAGS_seed)),
          static_cast<unsigned long long>(shared_manifest.base_seed)));
    }

    ctx.start_update = shared_manifest.global_update + 1;
    ctx.total_env_steps.store(shared_manifest.total_env_steps);
    ctx.next_episode_id.store(shared_manifest.next_episode_id);
    ctx.run_uuid = shared_manifest.run_uuid;
    ctx.master = shared_manifest.base_seed;
    ctx.target_end_update = shared_manifest.target_end_update;
  }

  std::cout << "[STUDY] Resuming study from update " << ctx.start_update
            << " to " << ctx.target_end_update << ".\n";
}

}  // namespace

void RunResumePhase(PpoTrainRunContext& ctx) {
  if (ctx.init_mode == "diagnostic") {
    RunDiagnosticResume(ctx);
  } else if (ctx.init_mode == "checkpoint") {
    RunCheckpointResume(ctx);
  } else if (ctx.init_mode == "study_resume") {
    RunStudyResume(ctx);
  } else {
    SpielFatalError("RunResumePhase called for unsupported init_mode: " + ctx.init_mode);
  }
}

#endif  // OPEN_SPIEL_BUILD_WITH_LIBTORCH

}  // namespace open_spiel::ppo_train_internal
