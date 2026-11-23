#include "dune_ppo_train_run_context.h"

namespace open_spiel::ppo_train_internal {

#ifdef OPEN_SPIEL_BUILD_WITH_LIBTORCH

void PpoTrainRunContext::RunAuxCollection(int update_id, CollectResult& cr) {
  if (!online_search_collection) return;
  aux_collector->CollectUpdate(update_id, game, aux_evaluator, &cr.aux_examples,
                               &cr.aux_stats);
}

void PpoTrainRunContext::AccountAuxConsumed(const CollectResult& cr) {
  if (!online_search_collection) return;
  const open_spiel::OnlineSearchCollectionStats& st = cr.aux_stats;
  aux_cum_accepted += st.accepted_targets;
  aux_cum_rejected += st.rejected_incomplete;
  aux_cum_granted += st.swordmaster_granted_games;
  aux_cum_organic += st.swordmaster_organic_games;
  for (int r = 0; r < open_spiel::OnlineCollectionState::kNumSearchRoles; ++r) {
    aux_cum_role_searches[r] += st.by_role[r].searches;
    aux_cum_role_accepted[r] += st.by_role[r].accepted;
  }
  aux_next_episode_id_persist = static_cast<uint64_t>(st.next_episode_id);
  std::string digest;
  for (const auto& ex : cr.aux_examples) {
    digest.append(reinterpret_cast<const char*>(&ex.episode_id), sizeof(ex.episode_id));
    int64_t did = ex.decision_id;
    digest.append(reinterpret_cast<const char*>(&did), sizeof(did));
    int32_t pl = ex.player;
    digest.append(reinterpret_cast<const char*>(&pl), sizeof(pl));
    for (Action a : ex.legal_actions)
      digest.append(reinterpret_cast<const char*>(&a), sizeof(a));
    for (double v : ex.normalized_visits)
      digest.append(reinterpret_cast<const char*>(&v), sizeof(v));
    digest.append(reinterpret_cast<const char*>(&ex.value_target), sizeof(ex.value_target));
  }
  aux_hash_chain = open_spiel::ComputeStringSHA256(aux_hash_chain + digest);
}

open_spiel::OnlineCollectionState PpoTrainRunContext::BuildAuxState() const {
  open_spiel::OnlineCollectionState s;
  s.present = true;
  s.auxiliary_games = aux_config.auxiliary_games;
  s.auxiliary_search_seed_domain = aux_config.auxiliary_search_seed_domain;
  s.collector_dirichlet_epsilon = aux_config.dirichlet_epsilon;
  s.target_sharpen_exponent = aux_config.target_sharpen_exponent;
  s.acceptance_prior_source =
      AcceptancePriorSourceName(aux_config.acceptance_prior_source);
  s.swordmaster_grant_fraction = aux_config.swordmaster_grant_fraction;
  s.swordmaster_grant_round = aux_config.swordmaster_grant_round;
  s.search_loss_coef_target = aux_search_loss_coef_target;
  s.search_loss_warmup_update = aux_search_loss_warmup;
  s.abort_grad_norm_ratio = aux_abort_ratio;
  s.next_auxiliary_episode_id = aux_next_episode_id_persist;
  s.cum_accepted = aux_cum_accepted;
  s.cum_rejected = aux_cum_rejected;
  for (int r = 0; r < open_spiel::OnlineCollectionState::kNumSearchRoles; ++r) {
    s.cum_role_searches[r] = aux_cum_role_searches[r];
    s.cum_role_accepted[r] = aux_cum_role_accepted[r];
  }
  s.cum_granted = aux_cum_granted;
  s.cum_organic = aux_cum_organic;
  s.accepted_hash_chain = aux_hash_chain;
  return s;
}

void RunSetupPhase(PpoTrainRunContext& ctx) {
  if (ctx.training_model) ctx.training_model->deck_pool_semantics_ = ctx.resolved_target_deck_pool_semantics;
  if (ctx.inference_model) ctx.inference_model->deck_pool_semantics_ = ctx.resolved_target_deck_pool_semantics;
  if (ctx.ema_model) ctx.ema_model->deck_pool_semantics_ = ctx.resolved_target_deck_pool_semantics;
  if (ctx.collection_model) ctx.collection_model->deck_pool_semantics_ = ctx.resolved_target_deck_pool_semantics;
  if (ctx.anchor_model) ctx.anchor_model->deck_pool_semantics_ = ctx.resolved_target_deck_pool_semantics;
  if (ctx.training_critic) ctx.training_critic->deck_pool_semantics_ = ctx.resolved_target_deck_pool_semantics;
  if (ctx.inference_critic) ctx.inference_critic->deck_pool_semantics_ = ctx.resolved_target_deck_pool_semantics;

  if (absl::GetFlag(FLAGS_separate_actor_critic) && ctx.training_critic != nullptr) {
    SyncDualModels(ctx.training_model, ctx.inference_model,
                   ctx.training_critic, ctx.inference_critic, &ctx.sync_mutex);
  } else {
    SyncModels(ctx.training_model, ctx.inference_model, &ctx.sync_mutex);
  }
  if (ctx.collection_model != nullptr) {
    CopyModelWeights(ctx.training_model, ctx.collection_model);
  }
  if (ctx.ema_model != nullptr && ctx.ema_updates_absorbed == 0) {
    CopyModelWeights(ctx.training_model, ctx.ema_model);
  }

  if (absl::GetFlag(FLAGS_deterministic_rollout_eval)) {
    ctx.evaluator = std::make_shared<open_spiel::DeterministicEvaluator>(
        ctx.inference_model, ctx.device, &ctx.eval_mutex, &ctx.sync_mutex,
        (absl::GetFlag(FLAGS_separate_actor_critic) && !absl::GetFlag(FLAGS_privileged_critic))
            ? ctx.inference_critic : nullptr);
  } else {
    ctx.evaluator = std::make_shared<open_spiel::BatchedEvaluator>(
        ctx.inference_model, absl::GetFlag(FLAGS_eval_batch_size),
        absl::GetFlag(FLAGS_eval_timeout_ms), ctx.device, &ctx.sync_mutex, 0.0f,
        absl::GetFlag(FLAGS_evaluator_device_synchronize),
        /*high_priority_stream=*/false,
        /*emit_batch_membership=*/ctx.numerical_parity,
        absl::GetFlag(FLAGS_rollout_amp),
        absl::GetFlag(FLAGS_allow_tf32));
    if (std::getenv("DUNE_ENABLE_ASYNC_STAGE_TIMERS") != nullptr) {
      if (auto be = std::dynamic_pointer_cast<open_spiel::BatchedEvaluator>(ctx.evaluator)) {
        be->EnableAsyncStageTimers(true);
      }
    }
  }

  const bool is_single_learner = absl::GetFlag(FLAGS_single_learner_training);
  const bool is_mixed_pool = (absl::GetFlag(FLAGS_pool_game_fraction) > 0.0 &&
                              absl::GetFlag(FLAGS_pool_opponents_per_game) > 0);

  if (is_single_learner || is_mixed_pool) {
    std::string pool_str = absl::GetFlag(FLAGS_training_opponent_pool);
    if (!pool_str.empty()) {
      std::vector<std::string> opp_paths = dune_opponent_pool::ParseOpponentPoolPaths(pool_str);
      int opp_batch_size = absl::GetFlag(FLAGS_eval_batch_size);
      if (is_mixed_pool && !opp_paths.empty()) {
        double expected_active =
            static_cast<double>(absl::GetFlag(FLAGS_threads)) *
            absl::GetFlag(FLAGS_pool_game_fraction) *
            (static_cast<double>(absl::GetFlag(FLAGS_pool_opponents_per_game)) / 4.0) /
            static_cast<double>(opp_paths.size());
        int scaled_batch = std::max(1, static_cast<int>(std::ceil(expected_active)));
        opp_batch_size = std::min(opp_batch_size, scaled_batch);
      }
      for (const auto& opp_path : opp_paths) {
        if (!std::filesystem::exists(opp_path)) {
          SpielFatalError("Training opponent pool checkpoint does not exist: " + opp_path);
        }
        int opp_detected_hidden = absl::GetFlag(FLAGS_hidden_dim);
        int opp_detected_blocks = absl::GetFlag(FLAGS_num_blocks);
        int opp_detected_input_dim = static_cast<int>(ctx.obs_size);
        std::string opp_detected_market_mode_str = "none";
        dune_imperium::DeckPoolSemantics opp_detected_semantics =
            dune_imperium::DeckPoolSemantics::kLegacy;
        std::string opp_detected_schema =
            dune_semantic::kDescriptorSchemaVersionV3;
        bool opp_slot_indexed_f = false;
        if (!open_spiel::DetectCheckpointModelDimensions(
                opp_path, &opp_detected_hidden, &opp_detected_blocks,
                &opp_detected_input_dim, &opp_detected_market_mode_str,
                &opp_detected_semantics, &opp_detected_schema,
                &opp_slot_indexed_f)) {
          SpielFatalError("Failed to detect model dimensions for opponent checkpoint: " + opp_path);
        }
        const auto opp_market_mode =
            dune_imperium::ParseMarketAppendixMode(opp_detected_market_mode_str);
        if (dune_imperium::IsExtendedInformationStateSize(opp_detected_input_dim)) {
          if (opp_market_mode == dune_imperium::MarketAppendixMode::kNone) {
            SpielFatalError(absl::StrFormat(
                "Expanded opponent checkpoint %s has absent or invalid market_appendix_mode: '%s'",
                opp_path, opp_detected_market_mode_str));
          }
          SPIEL_CHECK_EQ(
              opp_detected_input_dim,
              dune_imperium::GetInformationStateSchema(opp_market_mode, opp_detected_semantics).size);
        }
        const bool opp_has_scorer = CheckpointHasSemanticScorer(opp_path);
        if (dune_imperium::IsFullPublicInformationStateSize(opp_detected_input_dim) && !opp_has_scorer) {
          SpielFatalError(
              "Opponent pool full public information actor requires an active semantic scorer with schema v3, v4, or v5: " +
              opp_path);
        }
        auto opp_model = std::make_shared<open_spiel::SharedDunePolicyValueNetImpl>(
            opp_detected_input_dim, opp_detected_hidden, ctx.action_size,
            opp_detected_blocks, /*use_nonlinear=*/false,
            /*with_aux_heads=*/false, /*head_init_seed=*/0,
            /*with_semantic_scorer=*/opp_has_scorer,
            /*enable_zone_encoder=*/false, opp_detected_schema);
        opp_model->SetSemanticDescriptorSchema(opp_detected_schema);
        if (opp_slot_indexed_f && opp_model->semantic_scorer_ != nullptr) {
          opp_model->semantic_scorer_->EnableF(true);
        }
        opp_model->market_appendix_mode_ = opp_market_mode;
        opp_model->deck_pool_semantics_ = opp_detected_semantics;
        opp_model->to(ctx.device);
        open_spiel::LoadModelCheckpoint(opp_model, opp_path, ctx.device);
        opp_model->to(ctx.device);
        opp_model->eval();
        for (auto& p : opp_model->parameters()) {
          p.set_requires_grad(false);
        }
        std::string opp_hash = dune_opponent_pool::HashModelParametersAndBuffers(opp_model);
        ctx.opponent_hashes_before.push_back(opp_hash);
        ctx.opponent_models.push_back(opp_model);

        std::shared_ptr<open_spiel::IGameEvaluator> opp_eval;
        if (absl::GetFlag(FLAGS_deterministic_rollout_eval)) {
          opp_eval = std::make_shared<open_spiel::DeterministicEvaluator>(
              opp_model, ctx.device, &ctx.eval_mutex);
        } else {
          opp_eval = std::make_shared<open_spiel::BatchedEvaluator>(
              opp_model, opp_batch_size,
              absl::GetFlag(FLAGS_eval_timeout_ms), ctx.device, &ctx.sync_mutex, 0.0f,
              absl::GetFlag(FLAGS_evaluator_device_synchronize),
              /*high_priority_stream=*/false,
              /*emit_batch_membership=*/false,
              absl::GetFlag(FLAGS_rollout_amp),
              absl::GetFlag(FLAGS_allow_tf32));
        }
        ctx.opponent_evaluators.push_back(opp_eval);
      }
      if (is_mixed_pool) {
        std::cout << "[OPPONENT POOL] Loaded " << ctx.opponent_evaluators.size()
                  << " frozen opponent models for mixed-episode multi-learner training (fraction="
                  << absl::GetFlag(FLAGS_pool_game_fraction) << ", opponents="
                  << absl::GetFlag(FLAGS_pool_opponents_per_game) << ")." << std::endl;
      } else {
        std::cout << "[OPPONENT POOL] Loaded " << ctx.opponent_evaluators.size()
                  << " frozen opponent models for single-learner training." << std::endl;
      }
    } else {
      if (is_mixed_pool) {
        std::cout << "[OPPONENT POOL] Mixed-episode training active in current-policy control mode (Arm 1 Control: 0 pool models, fraction="
                  << absl::GetFlag(FLAGS_pool_game_fraction) << ", opponents="
                  << absl::GetFlag(FLAGS_pool_opponents_per_game) << ")." << std::endl;
      } else {
        std::cout << "[OPPONENT POOL] Single-learner training active in current-policy control mode (Arm A: 0 pool models, snapshot opponent)." << std::endl;
      }
    }
  }

  // --- Phase 18B online auxiliary-search collection setup ---
  ctx.online_search_collection = absl::GetFlag(FLAGS_online_search_collection);
  if (ctx.online_search_collection && !absl::GetFlag(FLAGS_search_label_dir).empty()) {
    SpielFatalError(
        "--online_search_collection and --search_label_dir are mutually "
        "exclusive (online collection vs legacy offline distillation).");
  }
  ctx.aux_search_loss_coef_target = absl::GetFlag(FLAGS_search_loss_coef);
  ctx.aux_search_loss_warmup = absl::GetFlag(FLAGS_search_loss_warmup_update);
  ctx.aux_abort_ratio = absl::GetFlag(FLAGS_abort_grad_norm_ratio);
  if (ctx.online_search_collection) {
    if (absl::GetFlag(FLAGS_auxiliary_search_seed_domain) == 0) {
      SpielFatalError(
          "--auxiliary_search_seed_domain must be set nonzero when "
          "--online_search_collection is enabled (no silent default).");
    }
    ctx.aux_config.auxiliary_games = absl::GetFlag(FLAGS_auxiliary_games);
    ctx.aux_config.auxiliary_search_seed_domain =
        absl::GetFlag(FLAGS_auxiliary_search_seed_domain);
    ctx.aux_config.dirichlet_epsilon = absl::GetFlag(FLAGS_collector_dirichlet_epsilon);
    ctx.aux_config.acceptance_prior_source = EffectiveAcceptancePriorSource();
    ctx.aux_config.target_sharpen_exponent =
        absl::GetFlag(FLAGS_target_sharpen_exponent);
    ctx.aux_config.swordmaster_grant_fraction =
        absl::GetFlag(FLAGS_swordmaster_grant_fraction);
    ctx.aux_config.swordmaster_grant_round =
        absl::GetFlag(FLAGS_swordmaster_grant_round);
    ctx.aux_config.search_leader_draft = absl::GetFlag(FLAGS_search_leader_draft);
    ctx.aux_config.leader_draft_simulations =
        absl::GetFlag(FLAGS_leader_draft_simulations);
    ctx.aux_config.leader_mass_only_coverage =
        absl::GetFlag(FLAGS_leader_mass_only_coverage);
    if (ctx.aux_config.search_leader_draft &&
        ctx.aux_config.leader_draft_simulations <= 0) {
      SpielFatalError(
          "--leader_draft_simulations must be positive when "
          "--search_leader_draft is enabled.");
    }
    ctx.aux_config.next_auxiliary_episode_id =
        static_cast<int64_t>(ctx.aux_next_episode_id_persist);
    ctx.aux_logit_cap = static_cast<float>(absl::GetFlag(FLAGS_logit_cap));
    ctx.aux_evaluator = std::make_shared<open_spiel::DuneNNEvaluator>(
        ctx.inference_model, ctx.device, ctx.aux_logit_cap,
        ctx.inference_model->market_appendix_mode_);
    ctx.aux_collector = std::make_shared<open_spiel::OnlineSearchCollector>(
        ctx.aux_config, ctx.config_fingerprint);
    std::cout << absl::StrFormat(
        "[18B] Online collection ON | aux_games=%d seed_domain=%llu "
        "dirichlet_eps=%.3f grant_frac=%.3f grant_round=%d loss_coef=%.3f/warmup%d "
        "abort_ratio=%.3f sharpen=%.3f resume_ep=%llu logit_cap=%.3f "
        "accept_prior=%s\n",
        ctx.aux_config.auxiliary_games,
        (unsigned long long)ctx.aux_config.auxiliary_search_seed_domain,
        ctx.aux_config.dirichlet_epsilon, ctx.aux_config.swordmaster_grant_fraction,
        ctx.aux_config.swordmaster_grant_round, ctx.aux_search_loss_coef_target,
        ctx.aux_search_loss_warmup, ctx.aux_abort_ratio, ctx.aux_config.target_sharpen_exponent,
        (unsigned long long)ctx.aux_next_episode_id_persist, ctx.aux_logit_cap,
        open_spiel::AcceptancePriorSourceName(ctx.aux_config.acceptance_prior_source));
  }

  std::cout << absl::StrFormat(
      "Initialized dune_ppo_train | obs=%d actions=%d hidden=%d blocks=%d "
      "rollout_transitions=%d rollout_games=%d minibatch=%d epochs=%d clip=%.3f vf_coef=%.3f "
      "ent_coef=%.4f gamma=%.3f gae_lambda=%.3f\n",
      ctx.obs_size, ctx.action_size, absl::GetFlag(FLAGS_hidden_dim),
      absl::GetFlag(FLAGS_num_blocks),
      absl::GetFlag(FLAGS_rollout_transitions),
      absl::GetFlag(FLAGS_rollout_games),
      absl::GetFlag(FLAGS_ppo_minibatch_size),
      absl::GetFlag(FLAGS_ppo_update_epochs),
      absl::GetFlag(FLAGS_ppo_clip_epsilon),
      absl::GetFlag(FLAGS_value_coef), absl::GetFlag(FLAGS_entropy_coef),
      absl::GetFlag(FLAGS_gamma), absl::GetFlag(FLAGS_gae_lambda));

  ctx.total_games = 0;
  ctx.total_moves = 0;
  ctx.training_start = std::chrono::high_resolution_clock::now();

  ctx.base_lr = absl::GetFlag(FLAGS_learning_rate);
  ctx.pipeline = absl::GetFlag(FLAGS_pipeline);
  ctx.num_threads = absl::GetFlag(FLAGS_threads);

  ctx.search_buffer.SetExpectedDimensions(ctx.obs_size, ctx.action_size);
  ctx.search_label_dir = absl::GetFlag(FLAGS_search_label_dir);
  if (!ctx.search_label_dir.empty()) {
    ctx.search_buffer.LoadFromDirectory(ctx.search_label_dir);
  }

  // Search-in-Training: teacher evaluator pool
  if (absl::GetFlag(FLAGS_search_in_training)) {
    if (absl::GetFlag(FLAGS_online_search_collection)) {
      SpielFatalError("--search_in_training and --online_search_collection are mutually exclusive.");
    }
    const std::string teacher_ckpt = absl::GetFlag(FLAGS_search_teacher_checkpoint);
    const int num_evals = absl::GetFlag(FLAGS_search_num_evaluators);
    const int target_batch = absl::GetFlag(FLAGS_search_target_batch_size);
    std::cout << "[SEARCH-IN-TRAINING] Initializing " << num_evals
              << " teacher evaluators (target batch " << target_batch
              << ") from " << (teacher_ckpt.empty() ? "training_model snapshot" : teacher_ckpt) << std::endl;
    ctx.search_teacher_models.resize(num_evals);
    ctx.search_teacher_mutexes.reserve(num_evals);
    ctx.search_teacher_coords.resize(num_evals);
    ctx.search_teacher_evaluators.resize(num_evals);

    const float teacher_logit_cap = static_cast<float>(absl::GetFlag(FLAGS_logit_cap));
    for (int e = 0; e < num_evals; ++e) {
      if (!teacher_ckpt.empty() && teacher_ckpt != "None") {
        int t_hidden = absl::GetFlag(FLAGS_hidden_dim);
        int t_blocks = absl::GetFlag(FLAGS_num_blocks);
        int t_input_dim = static_cast<int>(ctx.obs_size);
        std::string t_market_mode_str = "none";
        dune_imperium::DeckPoolSemantics t_semantics =
            dune_imperium::DeckPoolSemantics::kLegacy;
        std::string t_schema = dune_semantic::kDescriptorSchemaVersionV3;
        bool t_slot_indexed_f = false;
        if (!open_spiel::DetectCheckpointModelDimensions(
                teacher_ckpt, &t_hidden, &t_blocks, &t_input_dim,
                &t_market_mode_str, &t_semantics, &t_schema,
                &t_slot_indexed_f)) {
          SpielFatalError("Failed to detect model dimensions for search teacher checkpoint: " + teacher_ckpt);
        }
        const auto t_market_mode =
            dune_imperium::ParseMarketAppendixMode(t_market_mode_str);
        const bool t_has_scorer = CheckpointHasSemanticScorer(teacher_ckpt);
        ctx.search_teacher_models[e] = std::make_shared<SharedDunePolicyValueNetImpl>(
            t_input_dim, t_hidden, ctx.action_size, t_blocks,
            /*use_nonlinear=*/false, /*with_aux_heads=*/false,
            /*head_init_seed=*/0, /*with_semantic_scorer=*/t_has_scorer,
            /*enable_zone_encoder=*/false, t_schema);
        ctx.search_teacher_models[e]->SetSemanticDescriptorSchema(t_schema);
        if (t_slot_indexed_f && ctx.search_teacher_models[e]->semantic_scorer_ != nullptr) {
          ctx.search_teacher_models[e]->semantic_scorer_->EnableF(true);
        }
        ctx.search_teacher_models[e]->market_appendix_mode_ = t_market_mode;
        ctx.search_teacher_models[e]->deck_pool_semantics_ = t_semantics;
        ctx.search_teacher_models[e]->to(ctx.device);
        open_spiel::LoadModelCheckpoint(ctx.search_teacher_models[e], teacher_ckpt, ctx.device);
        ctx.search_teacher_models[e]->to(ctx.device);
      } else {
        ctx.search_teacher_models[e] = std::make_shared<SharedDunePolicyValueNetImpl>(
            ctx.obs_size, absl::GetFlag(FLAGS_hidden_dim), ctx.action_size,
            absl::GetFlag(FLAGS_num_blocks),
            ctx.training_model->use_nonlinear_value_head_,
            /*with_aux_heads=*/false, /*head_init_seed=*/0,
            /*with_semantic_scorer=*/ctx.training_model->with_semantic_scorer_,
            /*enable_zone_encoder=*/ctx.training_model->enable_zone_encoder_,
            ctx.training_model->semantic_descriptor_schema_);
        ctx.search_teacher_models[e]->SetSemanticDescriptorSchema(
            ctx.training_model->semantic_descriptor_schema_);
        if (ctx.training_model->semantic_scorer_ != nullptr &&
            ctx.training_model->semantic_scorer_->slot_indexed_f_ &&
            ctx.search_teacher_models[e]->semantic_scorer_ != nullptr) {
          ctx.search_teacher_models[e]->semantic_scorer_->EnableF(true);
        }
        ctx.search_teacher_models[e]->market_appendix_mode_ =
            ctx.training_model->market_appendix_mode_;
        ctx.search_teacher_models[e]->deck_pool_semantics_ =
            ctx.training_model->deck_pool_semantics_;
        ctx.search_teacher_models[e]->to(ctx.device);
        CopyModelWeights(ctx.training_model, ctx.search_teacher_models[e]);
      }
      ctx.search_teacher_models[e]->eval();

      ctx.search_teacher_mutexes.push_back(std::make_unique<std::shared_mutex>());
      bool teacher_amp = (ctx.device.is_cuda() && (absl::GetFlag(FLAGS_search_teacher_amp) || absl::GetFlag(FLAGS_train_amp)));
      bool teacher_tf32 = absl::GetFlag(FLAGS_search_teacher_amp) || absl::GetFlag(FLAGS_allow_tf32);
      ctx.search_teacher_coords[e] = std::make_shared<open_spiel::BatchedEvaluator>(
          ctx.search_teacher_models[e], target_batch, /*timeout_ms=*/1, ctx.device,
          ctx.search_teacher_mutexes.back().get(), teacher_logit_cap,
          /*device_synchronize=*/false, /*high_priority_stream=*/true,
          /*emit_batch_membership=*/false,
          /*rollout_amp=*/teacher_amp,
          /*allow_tf32=*/teacher_tf32);
      ctx.search_teacher_evaluators[e] = std::make_shared<open_spiel::BatchedNNEvaluator>(
          ctx.search_teacher_coords[e], teacher_logit_cap,
          ctx.search_teacher_models[e]->market_appendix_mode_);
    }
    absl::SetFlag(&FLAGS_aux_full_batch_presentation, true);
    if (absl::GetFlag(FLAGS_aux_value_coef) < 0.0) {
      absl::SetFlag(&FLAGS_aux_value_coef, 0.0);
    }
  }

  ctx.rollout_games = absl::GetFlag(FLAGS_rollout_games);
  if (absl::GetFlag(FLAGS_train_value_only)) {
    ctx.initial_non_value_hash = HashNonValueParameters(ctx.training_model);
    std::cout << "[INFO] Initial non-value parameters SHA256: " << ctx.initial_non_value_hash << "\n";
  }
}


std::string PpoTrainRunContext::ResolveInputSidecarPath(
    const std::string& filename) const {
  const std::string model_ckpt = absl::GetFlag(FLAGS_model_checkpoint);
  if (model_ckpt.empty()) return "";
  std::filesystem::path dir1 = std::filesystem::path(model_ckpt).parent_path();
  if (dir1.empty()) dir1 = ".";
  std::filesystem::path p1 = dir1 / filename;
  if (std::filesystem::exists(p1)) return p1.string();
  std::error_code ec;
  if (std::filesystem::is_symlink(model_ckpt, ec)) {
    auto resolved = std::filesystem::canonical(model_ckpt, ec);
    if (!ec) {
      std::filesystem::path p2 = resolved.parent_path() / filename;
      if (std::filesystem::exists(p2)) return p2.string();
    }
  }
  return p1.string();
}

void PpoTrainRunContext::WriteVerificationSidecarIfEnabled(
    int update_id,
    const CollectResult& collect,
    const open_spiel::PpoUpdateStats& stats) const {
  std::string verification_sidecar_path =
      absl::GetFlag(FLAGS_verification_sidecar_path);
  if (verification_sidecar_path.empty()) return;
  std::string semantic_digest =
      open_spiel::ComputeRolloutSemanticDigest(collect.rollout);
  std::ofstream v_file(verification_sidecar_path, std::ios::app);
  if (v_file.is_open()) {
    v_file << open_spiel::SerializeVerificationSidecarRecord(
        static_cast<int64_t>(update_id),
        static_cast<int64_t>(collect.rollout.size()),
        semantic_digest,
        stats.rollout_hash,
        stats.policy_kl_before,
        stats.epoch_kls,
        stats.measured_transitions,
        stats.fraction_critic_near_1,
        stats.minibatches,
        stats.policy_loss,
        stats.value_loss,
        stats.explained_variance,
        stats.early_stopped,
        static_cast<int64_t>(stats.kl_early_stop_epoch)) << "\n";
  }
}

namespace {

int RunNumericalParityGate(
    PpoTrainRunContext& ctx,
    const std::string& parity_model_hash_before,
    const std::string& parity_inference_hash_before,
    bool parity_tf32_cublas_before,
    bool parity_tf32_cudnn_before) {
  const auto batch_evaluator =
      std::dynamic_pointer_cast<open_spiel::BatchedEvaluator>(ctx.evaluator);
  if (batch_evaluator == nullptr) {
    SpielFatalError("v4 numerical parity requires BatchedEvaluator");
  }
  const open_spiel::EvaluatorStats evaluator_before_r =
      batch_evaluator->GetStats();
  std::vector<open_spiel::PpoParityBatchMembership> membership;
  membership.reserve(ctx.current_collect.rollout.size());
  for (const auto& transition : ctx.current_collect.rollout) {
    membership.push_back({transition.behavior_physical_batch_id,
                          transition.behavior_physical_batch_size,
                          transition.behavior_physical_batch_row});
  }
  const open_spiel::PpoParityBatchGeometry geometry =
      open_spiel::ReconstructPpoParityBatchGeometry(
          membership, absl::GetFlag(FLAGS_eval_batch_size));
  const std::string geometry_sha256 =
      open_spiel::ComputeStringSHA256(geometry.canonical_payload);

  PpoParityCellResult integrity_cell;
  integrity_cell.name = "rollout_cpu_recompute_integrity";
  integrity_cell.module_train_mode = false;
  integrity_cell.autocast_enabled = false;
  PpoParityIntegrityStats integrity_stats;
  const auto a_start = std::chrono::steady_clock::now();
  integrity_cell.rows = ReplayPpoCpuIntegrityCell(
      ctx.current_collect.rollout,
      static_cast<float>(absl::GetFlag(FLAGS_logit_cap)),
      absl::GetFlag(FLAGS_rollout_amp),
      &integrity_cell.schema_errors, &integrity_cell.row_evidence_sha256,
      &integrity_stats.raw_values_total,
      &integrity_stats.raw_values_bf16_exact,
      &integrity_stats.vector_width_rows_ok,
      &integrity_stats.behavior_vector_exact_rows,
      &integrity_stats.chosen_scalar_exact_rows);
  integrity_cell.wall_time_s = std::chrono::duration<double>(
      std::chrono::steady_clock::now() - a_start).count();
  integrity_cell.raw_logit_sha256 =
      PpoParityCapturedRawSha256(ctx.current_collect.rollout);
  integrity_cell.policy_sha256 =
      PpoParityCapturedPolicySha256(ctx.current_collect.rollout);
  const int64_t transitions =
      static_cast<int64_t>(ctx.current_collect.rollout.size());
  const bool hard_capture_valid =
      ctx.current_collect.parity_capture_errors.empty() &&
      ctx.current_collect.parity_decisions == transitions &&
      ctx.current_collect.parity_full_width_ok == transitions &&
      ctx.current_collect.parity_full_finite_ok == transitions &&
      ctx.current_collect.parity_legal_ids_unique_in_range == transitions &&
      ctx.current_collect.parity_chosen_once == transitions &&
      integrity_cell.schema_errors.empty() &&
      static_cast<int64_t>(integrity_cell.rows.size()) == transitions &&
      integrity_stats.vector_width_rows_ok == transitions &&
      integrity_stats.behavior_vector_exact_rows == transitions &&
      integrity_stats.chosen_scalar_exact_rows == transitions &&
      geometry.valid && parity_tf32_cublas_before &&
      parity_tf32_cudnn_before &&
      evaluator_before_r.requests == static_cast<uint64_t>(transitions) &&
      evaluator_before_r.batches ==
          static_cast<uint64_t>(geometry.groups) &&
      evaluator_before_r.max_batch_size ==
          static_cast<uint64_t>(geometry.max_batch_size);

  PpoParityCellResult replay_cell;
  PpoParityReplayGeometryStats replay_stats;
  PpoParityCellResult learner_cell;
  learner_cell.name = "actual_learner_full_fp32_tf32_2048";
  learner_cell.module_train_mode = true;
  learner_cell.autocast_enabled = false;
  PpoParityCellResult geometry_model_cell;
  PpoParityOriginalGeometryModelStats geometry_model_stats;
  if (hard_capture_valid) {
    const auto r_start = std::chrono::steady_clock::now();
    replay_cell = ReplayPpoInferenceExactGeometry(
        batch_evaluator.get(), ctx.current_collect.rollout, geometry,
        ctx.action_size, static_cast<float>(absl::GetFlag(FLAGS_logit_cap)),
        evaluator_before_r, &replay_stats);
    replay_cell.wall_time_s = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - r_start).count();
    const bool r_valid = replay_cell.schema_errors.empty() &&
        replay_stats.returned_metadata_exact_rows == transitions &&
        replay_stats.raw_legal_exact_rows == transitions &&
        replay_stats.cpu_policy_exact_rows == transitions &&
        replay_stats.captured_raw_sha256 == replay_stats.replay_raw_sha256 &&
        replay_stats.after.requests - replay_stats.before.requests ==
            static_cast<uint64_t>(transitions) &&
        replay_stats.after.batches - replay_stats.before.batches ==
            static_cast<uint64_t>(geometry.groups) &&
        replay_stats.after.max_batch_size ==
            replay_stats.before.max_batch_size;
    if (r_valid) {
      const auto b_start = std::chrono::steady_clock::now();
      learner_cell.rows = ReplayPpoNumericalParityCell(
          ctx.training_model, ctx.current_collect.rollout, ctx.obs_size, ctx.action_size,
          ctx.device, /*model_train_mode=*/true,
          /*learner_autocast=*/false, learner_cell.name,
          &learner_cell.schema_errors, &learner_cell.row_evidence_sha256,
          &learner_cell.model_forward_calls,
          &learner_cell.dense_mask_rows_ok,
          &learner_cell.raw_logit_sha256,
          &learner_cell.policy_sha256);
      learner_cell.wall_time_s = std::chrono::duration<double>(
          std::chrono::steady_clock::now() - b_start).count();
      const auto d_start = std::chrono::steady_clock::now();
      geometry_model_cell = ReplayPpoTrainingModelOriginalGeometry(
          ctx.training_model, ctx.current_collect.rollout, geometry, ctx.obs_size,
          ctx.action_size, ctx.device,
          static_cast<float>(absl::GetFlag(FLAGS_logit_cap)),
          /*learner_autocast=*/false,
          &geometry_model_stats);
      geometry_model_cell.wall_time_s = std::chrono::duration<double>(
          std::chrono::steady_clock::now() - d_start).count();
    } else {
      learner_cell.schema_errors.push_back(
          "skipped because R exact replay validity failed");
      geometry_model_cell.schema_errors.push_back(
          "skipped because R exact replay validity failed");
    }
  } else {
    replay_cell.name = "live_inference_exact_original_batches";
    replay_cell.schema_errors.push_back(
        "skipped because A/capture/geometry validity failed");
    learner_cell.schema_errors.push_back(
        "skipped because A/capture/geometry validity failed");
    geometry_model_cell.name = "training_model_exact_original_batches";
    geometry_model_cell.schema_errors.push_back(
        "skipped because A/capture/geometry validity failed");
  }
  const std::string parity_model_hash_after =
      HashAllModelState(ctx.training_model);
  const std::string parity_inference_hash_after =
      HashAllModelState(ctx.inference_model);
  const bool parity_tf32_cublas_after =
      at::globalContext().allowTF32CuBLAS();
  const bool parity_tf32_cudnn_after =
      at::globalContext().allowTF32CuDNN();
  const bool valid = WritePpoNumericalParityArtifactV5(
      absl::GetFlag(FLAGS_numerical_parity_output), integrity_cell,
      integrity_stats, replay_cell, replay_stats, learner_cell,
      geometry_model_cell, geometry_model_stats, geometry, geometry_sha256,
      ctx.current_collect,
      PpoParitySharedIdentitySha256(ctx.current_collect.rollout),
      parity_model_hash_before,
      parity_model_hash_after, parity_inference_hash_before,
      parity_inference_hash_after, ctx.parity_source_manifest,
      ctx.parity_source_manifest_sha256, ctx.parity_source_global_update,
      ctx.parity_source_checkpoint_uuid, ctx.config_fingerprint, ctx.device,
      ctx.obs_size, ctx.action_size, ctx.parity_command_line,
      ctx.parity_source_provenance,
      parity_tf32_cublas_before, parity_tf32_cudnn_before,
      parity_tf32_cublas_after, parity_tf32_cudnn_after);
  std::cout << "Raw-PPO numerical parity v5-A "
            << (valid ? "VALID classification" : "INVALID instrument")
            << ": "
            << absl::GetFlag(FLAGS_numerical_parity_output) << "\n"
            << "No optimizer checkpoint was loaded and optimizer.step was "
               "not called; no backward pass or training update ran.\n";
  return valid ? 0 : 3;
}

void LoadR765CollateralPanel(PpoTrainRunContext& ctx) {
  const std::string panel_json_path = absl::GetFlag(FLAGS_r765_collateral_panel_json);
  if (panel_json_path.empty()) return;
  std::ifstream f(panel_json_path);
  if (!f.is_open()) {
    open_spiel::SpielFatalError("Failed to open --r765_collateral_panel_json: " + panel_json_path);
  }
  std::string content((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
  auto parsed = open_spiel::json::FromString(content);
  if (!parsed.has_value() || !parsed->IsArray() || parsed->GetArray().empty()) {
    open_spiel::SpielFatalError("Malformed or empty JSON in --r765_collateral_panel_json: " + panel_json_path);
  }
  const auto& arr = parsed->GetArray();
  const auto eval_market_mode = ctx.training_model->market_appendix_mode_;
  const auto eval_deck_semantics = ctx.training_model->deck_pool_semantics_;
  const int active_obs_size = dune_imperium::GetInformationStateSchema(
      eval_market_mode, eval_deck_semantics).size;
  int64_t hp65_count = 0;
  ctx.r765_collateral_panel.reserve(arr.size());
  for (size_t i = 0; i < arr.size(); ++i) {
    if (!arr[i].IsObject()) {
      open_spiel::SpielFatalError("Expected object in --r765_collateral_panel_json array at index " + std::to_string(i));
    }
    const auto& obj = arr[i].GetObject();
    auto it_ser = obj.find("serialized_state");
    auto it_ga = obj.find("baseline_greedy_action");
    auto it_probs = obj.find("baseline_probs");
    if (it_ser == obj.end() || it_ga == obj.end() || it_probs == obj.end()) {
      open_spiel::SpielFatalError("Missing required keys in --r765_collateral_panel_json at index " + std::to_string(i));
    }
    std::string ser_state = it_ser->second.GetString();
    int base_ga = it_ga->second.IsInt() ? static_cast<int>(it_ga->second.GetInt())
                                        : static_cast<int>(it_ga->second.GetDouble());
    double max_p = 0.0;
    if (it_probs->second.IsArray()) {
      for (const auto& pv : it_probs->second.GetArray()) {
        double p = pv.IsDouble() ? pv.GetDouble() : (pv.IsInt() ? static_cast<double>(pv.GetInt()) : 0.0);
        max_p = std::max(max_p, p);
      }
    }
    bool is_hp65 = (max_p >= 0.65);
    if (is_hp65) hp65_count++;

    std::unique_ptr<State> st = ctx.game->DeserializeState(ser_state);
    if (!st) {
      open_spiel::SpielFatalError("Failed to deserialize state in --r765_collateral_panel_json at index " + std::to_string(i));
    }
    Player current_player = st->CurrentPlayer();
    const auto* dune_state = dynamic_cast<const dune_imperium::DuneImperiumState*>(st.get());
    std::vector<float> obs(active_obs_size, 0.0f);
    if (dune_state != nullptr && current_player >= 0) {
      dune_state->InformationStateTensorWithAppendix(
          current_player, eval_market_mode, absl::MakeSpan(obs), eval_deck_semantics);
    } else if (current_player >= 0) {
      st->InformationStateTensor(current_player, absl::MakeSpan(obs));
    }
    std::vector<Action> legals = st->LegalActions();
    dune_semantic::CandidateActionData cand_data;
    if (dune_state != nullptr && ctx.training_model->with_semantic_scorer_) {
      dune_semantic::ExtractCandidateDescriptors(
          *dune_state, legals, &cand_data, ctx.training_model->semantic_descriptor_schema_);
    }
    R765CollateralPanelItem item;
    item.obs = std::move(obs);
    item.legal_actions = std::move(legals);
    item.candidate_data = std::move(cand_data);
    item.baseline_greedy_action = base_ga;
    item.is_hp65 = is_hp65;
    ctx.r765_collateral_panel.push_back(std::move(item));
  }
  std::cout << "[R765-PANEL] Loaded " << ctx.r765_collateral_panel.size()
            << " states (" << hp65_count << " hp65 states) from " << panel_json_path << "\n";
}

}  // namespace

std::optional<int> RunSetupPreloopPhase(PpoTrainRunContext& ctx) {
  // Collect first rollout synchronously.
  const std::string parity_model_hash_before =
      ctx.numerical_parity ? HashAllModelState(ctx.training_model) : "";
  const std::string parity_inference_hash_before =
      ctx.numerical_parity ? HashAllModelState(ctx.inference_model) : "";
  const bool parity_tf32_cublas_before =
      ctx.numerical_parity && at::globalContext().allowTF32CuBLAS();
  const bool parity_tf32_cudnn_before =
      ctx.numerical_parity && at::globalContext().allowTF32CuDNN();
  ctx.reward_lambda = ComputeRewardLambda(ctx.total_env_steps.load(),
                                          absl::GetFlag(FLAGS_shaping_start_env_steps),
                                          absl::GetFlag(FLAGS_shaping_decay_env_steps));
  if (!absl::GetFlag(FLAGS_r755_gate1_output).empty()) {
    RunR755Gate1Audit(
        ctx.game.get(), ctx.evaluator, ctx.obs_size, ctx.master, ctx.rollout_games,
        absl::GetFlag(FLAGS_r755_gate1_output));
    std::exit(0);
  }
  if (absl::GetFlag(FLAGS_r755_hybrid_mode)) {
    absl::SetFlag(&FLAGS_aux_full_batch_presentation, false);
  }
  ctx.current_collect = CollectRollout(
      ctx.game.get(), ctx.evaluator, ctx.obs_size, &ctx.total_env_steps,
      ctx.num_threads, &ctx.next_episode_id, ctx.rollout_games, ctx.reward_lambda,
      ctx.opponent_evaluators, /*global_update=*/ctx.start_update);

  if (ctx.numerical_parity) {
    return RunNumericalParityGate(
        ctx, parity_model_hash_before, parity_inference_hash_before,
        parity_tf32_cublas_before, parity_tf32_cudnn_before);
  }

  if (absl::GetFlag(FLAGS_diagnostics_only)) {
    open_spiel::PpoUpdateStats stats =
        (absl::GetFlag(FLAGS_separate_actor_critic) && ctx.training_critic != nullptr)
            ? open_spiel::TrainPpoUpdateSeparate(
                  ctx.training_model, *ctx.optimizer, ctx.training_critic, *ctx.critic_optimizer,
                  ctx.current_collect.rollout, ctx.obs_size, ctx.action_size, ctx.device,
                  ctx.master, ctx.start_update, /*compute_diagnostics=*/true,
                  absl::GetFlag(FLAGS_gamma),
                  (absl::GetFlag(FLAGS_train_value_only) ? 1.0 : absl::GetFlag(FLAGS_gae_lambda)))
            : open_spiel::TrainPpoUpdate(
                  ctx.training_model, *ctx.optimizer, ctx.current_collect.rollout,
                  ctx.obs_size, ctx.action_size, ctx.device, ctx.master, ctx.start_update,
                  ctx.anchor_model,
                  /*search_examples=*/{}, /*search_loss_coef=*/0.0,
                  /*abort_grad_norm_ratio=*/0.0,
                  ctx.collection_model, absl::GetFlag(FLAGS_reverse_kl_coef),
                  /*compute_diagnostics=*/true,
                  ctx.ema_model, absl::GetFlag(FLAGS_ema_magnet_kl_coef));
    stats.episode_ids_unique = ctx.current_collect.episode_ids_unique;
    stats.forced_buys = ctx.current_collect.forced_buys;
    AttachPrecapAbszStats(&stats, ctx.current_collect);
    AttachCanaryStats(&stats, ctx.current_collect);

    std::string diagnostics_path = absl::GetFlag(FLAGS_diagnostics_path);
    if (!diagnostics_path.empty()) {
      double val_kl = ctx.search_buffer.ComputeValidationKL(
          ctx.training_model, ctx.device, static_cast<float>(absl::GetFlag(FLAGS_logit_cap)));
      open_spiel::WriteDiagnostics(diagnostics_path, ctx.start_update, stats,
                                   ctx.current_collect.conflict_vp_generated,
                                   ctx.current_collect.conflict_vp_attributed,
                                   ctx.current_collect.conflict_vp_unattributed,
                                   ctx.master, ctx.run_uuid, absl::GetFlag(FLAGS_run_prefix),
                                   ctx.config_fingerprint,
                                   ctx.current_collect.raw_conflict_vp,
                                   ctx.current_collect.raw_noncombat_vp,
                                   ctx.current_collect.raw_total_vp,
                                   val_kl,
                                   absl::GetFlag(FLAGS_emit_canary_columns));
    }
    ctx.WriteVerificationSidecarIfEnabled(ctx.start_update, ctx.current_collect, stats);
    std::cout << "Diagnostics-only run complete. Exiting.\n";
    exit(0);
  }

  ctx.total_games += ctx.current_collect.games;
  ctx.total_moves += ctx.current_collect.moves;

  ctx.card_telemetry_path = absl::GetFlag(FLAGS_card_telemetry_path);
  if (!ctx.card_telemetry_path.empty()) {
    dune_imperium::LoadCardTelemetry(
        ctx.card_telemetry_path, &ctx.cum_card11, &ctx.cum_card13, ctx.start_update - 1);
  }

  ctx.cutoff_str = absl::GetFlag(FLAGS_cutoff_timestamp);
  ctx.cutoff_epoch = absl::GetFlag(FLAGS_cutoff_epoch);
  ctx.last_completed_update = ctx.start_update - 1;
  ctx.final_checkpoint_saved = false;

  // Phase 18B: aux examples for the first update, from the same frozen snapshot
  // the pre-loop rollout used (sequential, main thread).
  ctx.RunAuxCollection(ctx.start_update, ctx.current_collect);

  auto resolve_input_run_stem = []() -> std::string {
    const std::string model_ckpt = absl::GetFlag(FLAGS_model_checkpoint);
    if (!model_ckpt.empty()) {
      const std::string model_stem =
          std::filesystem::path(model_ckpt).stem().string();
      const size_t pos = model_stem.rfind("_model_");
      if (pos != std::string::npos && pos > 0) {
        return model_stem.substr(0, pos);
      }
    }
    std::string run_stem =
        std::filesystem::path(absl::GetFlag(FLAGS_run_prefix)).filename().string();
    return run_stem.empty() ? std::string("ppo") : run_stem;
  };
  const std::string input_stem = resolve_input_run_stem();
  auto resolve_sidecar_with_legacy =
      [&ctx](const std::string& primary_name,
             const std::string& legacy_name) -> std::string {
    const std::string primary = ctx.ResolveInputSidecarPath(primary_name);
    if (!primary.empty() && std::filesystem::exists(primary)) {
      return primary;
    }
    if (legacy_name != primary_name) {
      const std::string legacy = ctx.ResolveInputSidecarPath(legacy_name);
      if (!legacy.empty() && std::filesystem::exists(legacy)) {
        return legacy;
      }
    }
    return primary;
  };

  if ((ctx.init_mode == "checkpoint" || ctx.init_mode == "study_resume") &&
      absl::GetFlag(FLAGS_r765_replay_window_updates) > 1 && ctx.start_update > 1) {
    const std::string primary_replay_filename =
        absl::StrCat(input_stem, "_r765_replay_update_", ctx.start_update - 1, ".bin");
    const std::string legacy_replay_filename =
        absl::StrCat("ppo_r765_replay_update_", ctx.start_update - 1, ".bin");
    const std::string replay_sidecar =
        resolve_sidecar_with_legacy(primary_replay_filename, legacy_replay_filename);
    if (!replay_sidecar.empty() && std::filesystem::exists(replay_sidecar)) {
      if (!ctx.r765_replay_mgr.LoadState(replay_sidecar, &ctx.r765_preserve_guard)) {
        SpielFatalError("Failed to load R7-65 replay sidecar at: " + replay_sidecar);
      }
      std::cout << "[R765-REPLAY] Restored " << ctx.r765_replay_mgr.active_train_cohorts()
                << " live cohorts (" << ctx.r765_replay_mgr.active_diag_cohorts()
                << " diag cohorts) from " << replay_sidecar << "\n";
    } else {
      std::cout << "[R765-REPLAY] No sidecar found at " << replay_sidecar
                << "; starting with empty replay buffer.\n";
    }
  }

  LoadR765CollateralPanel(ctx);

  if (ctx.training_model->semantic_descriptor_schema_ ==
      dune_semantic::kDescriptorSchemaVersionV5) {
    const int v5_anneal_n = absl::GetFlag(FLAGS_v5_anneal_legacy_slot_residual_updates);
    bool loaded_v5_sidecar = false;
    if (v5_anneal_n > 0 && (ctx.init_mode == "checkpoint" || ctx.init_mode == "study_resume")) {
      const std::string v5_sidecar_in = resolve_sidecar_with_legacy(
          absl::StrCat(input_stem, "_v5_anneal_state.json"),
          "v5_anneal_state.json");
      if (!v5_sidecar_in.empty() && std::filesystem::exists(v5_sidecar_in)) {
        if (ctx.v5_slot_residual_state.LoadSidecar(v5_sidecar_in, ctx.device) &&
            ctx.v5_slot_residual_state.N == v5_anneal_n) {
          loaded_v5_sidecar = true;
          std::cout << "[INFO] Restored V5 legacy slot residual sidecar from "
                    << v5_sidecar_in << " (N=" << ctx.v5_slot_residual_state.N
                    << ", start_global_update="
                    << ctx.v5_slot_residual_state.start_global_update
                    << ", active=" << (ctx.v5_slot_residual_state.active ? "true" : "false")
                    << ").\n";
        }
      }
    }
    if (!loaded_v5_sidecar) {
      ctx.v5_slot_residual_state.Initialize(ctx.training_model, v5_anneal_n);
      ctx.v5_slot_residual_state.start_global_update =
          (g_semantic_schema_transition_source_update >= 0)
              ? static_cast<int>(g_semantic_schema_transition_source_update)
              : (ctx.start_update - 1);
      if (ctx.v5_slot_residual_state.active) {
        std::cout << "[INFO] Initialized V5 legacy slot residual handler (N="
                  << ctx.v5_slot_residual_state.N << ", start_global_update="
                  << ctx.v5_slot_residual_state.start_global_update
                  << ", active=true).\n";
      }
    }
  }

  return std::nullopt;
}

#endif  // OPEN_SPIEL_BUILD_WITH_LIBTORCH

}  // namespace open_spiel::ppo_train_internal
