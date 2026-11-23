#include "dune_ppo_train_run_context.h"

#include "open_spiel/abseil-cpp/absl/flags/parse.h"

namespace open_spiel::ppo_train_internal {

std::optional<int> RunStartupPhase(PpoTrainRunContext& ctx, int argc, char** argv) {
  setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8", 1);
  std::ostringstream parity_command_line_stream;
  for (int i = 0; i < argc; ++i) {
    if (i) parity_command_line_stream << ' ';
    parity_command_line_stream << std::quoted(std::string(argv[i]));
  }
  ctx.parity_command_line = parity_command_line_stream.str();
  absl::ParseCommandLine(argc, argv);
  open_spiel::dune_targeted_buy::g_forced_norm_csv_path =
      absl::GetFlag(FLAGS_targeted_buy_forced_csv);
  open_spiel::dune_targeted_buy::g_forced_telemetry_csv_path =
      absl::GetFlag(FLAGS_targeted_buy_telemetry_csv);
  std::signal(SIGTERM, open_spiel::ppo_train_internal::SignalHandler);
  std::signal(SIGINT, open_spiel::ppo_train_internal::SignalHandler);
  const bool numerical_parity =
      !absl::GetFlag(FLAGS_numerical_parity_output).empty();
  ctx.numerical_parity = numerical_parity;
  if (numerical_parity) {
    auto parity_stop = [](const std::string& message) {
      open_spiel::SpielFatalError(
          "Raw-PPO numerical parity configuration rejected: " + message);
    };
    if (!absl::GetFlag(FLAGS_diagnostics_only)) {
      parity_stop("--diagnostics_only must be true");
    }
    if (absl::GetFlag(FLAGS_init_mode) != "diagnostic") {
      parity_stop("--init_mode must be diagnostic");
    }
    if (absl::GetFlag(FLAGS_artifact_manifest).empty()) {
      parity_stop("--artifact_manifest is required");
    }
    if (absl::GetFlag(FLAGS_numerical_parity_registration_id).empty()) {
      parity_stop("--numerical_parity_registration_id is required");
    }
    if (absl::GetFlag(FLAGS_numerical_parity_source_root).empty()) {
      parity_stop("--numerical_parity_source_root is required");
    }
    const std::string registered_source_sha =
        absl::GetFlag(FLAGS_numerical_parity_source_sha256);
    if (registered_source_sha.size() != 64 ||
        !std::all_of(registered_source_sha.begin(), registered_source_sha.end(),
                     [](char c) {
                       return (c >= '0' && c <= '9') ||
                              (c >= 'a' && c <= 'f');
                     })) {
      parity_stop(
          "--numerical_parity_source_sha256 must be an explicit lowercase SHA-256");
    }
    const int games = absl::GetFlag(FLAGS_rollout_games);
    if (games <= 0 || games > 64) {
      parity_stop("--rollout_games must be in [1,64] (hard ceiling 320000 transitions)");
    }
    if (absl::GetFlag(FLAGS_rollout_amp) ||
        absl::GetFlag(FLAGS_train_amp) ||
        !absl::GetFlag(FLAGS_allow_tf32)) {
      parity_stop(
          "v5-A requires rollout_amp=false, train_amp=false, and "
          "allow_tf32=true");
    }
    if (absl::GetFlag(FLAGS_ppo_minibatch_size) != 2048) {
      parity_stop(
          "v5-A requires --ppo_minibatch_size=2048");
    }
    if (absl::GetFlag(FLAGS_deterministic_rollout_eval) ||
        !absl::GetFlag(FLAGS_evaluator_device_synchronize)) {
      parity_stop(
          "v4 requires BatchedEvaluator with deterministic_rollout_eval=false "
          "and evaluator_device_synchronize=true");
    }
    if (absl::GetFlag(FLAGS_pipeline) ||
        absl::GetFlag(FLAGS_online_search_collection) ||
        !absl::GetFlag(FLAGS_search_label_dir).empty() ||
        absl::GetFlag(FLAGS_train_value_only) ||
        absl::GetFlag(FLAGS_sample_counterfactual_states)) {
      parity_stop("pipeline, search, distillation, value-only, and counterfactual paths must all be disabled");
    }
    const double max_logprob_delta =
        absl::GetFlag(FLAGS_numerical_parity_max_abs_logprob_delta);
    const double max_full_kl =
        absl::GetFlag(FLAGS_numerical_parity_max_full_kl);
    const double max_raw_mass_residual =
        absl::GetFlag(FLAGS_numerical_parity_max_raw_mass_residual);
    const double min_ratio =
        absl::GetFlag(FLAGS_numerical_parity_min_ratio);
    const double max_ratio =
        absl::GetFlag(FLAGS_numerical_parity_max_ratio);
    if (absl::GetFlag(FLAGS_numerical_parity_min_rows) <= 0 ||
        absl::GetFlag(FLAGS_numerical_parity_min_nontrivial_rows) <= 0 ||
        !std::isfinite(max_logprob_delta) || max_logprob_delta <= 0.0 ||
        !std::isfinite(max_full_kl) || max_full_kl <= 0.0 ||
        !std::isfinite(max_raw_mass_residual) ||
        max_raw_mass_residual != 1e-5 ||
        !std::isfinite(min_ratio) || !(min_ratio > 0.0 && min_ratio <= 1.0) ||
        !std::isfinite(max_ratio) || max_ratio < 1.0) {
      parity_stop("all sample floors and numerical bounds must be stated explicitly and be valid");
    }
    if (absl::GetFlag(FLAGS_numerical_parity_min_rows) > games * 5000 ||
        absl::GetFlag(FLAGS_numerical_parity_min_nontrivial_rows) >
            games * 5000) {
      parity_stop("a sample floor exceeds the hard transition ceiling");
    }
    if (std::filesystem::exists(
            absl::GetFlag(FLAGS_numerical_parity_output)) ||
        std::filesystem::exists(
            absl::GetFlag(FLAGS_numerical_parity_output) + ".tmp")) {
      parity_stop("output or temp path already exists (fresh output required)");
    }
  }
  // PWO-5 section 10.2: the reserved final-gate base-seed range, enforced at
  // the launcher so the exclusion is mechanical rather than a matter of
  // operator care. Checked before ANY other work.
  {
    const std::string stop = dune_seed::ReservedFinalGateSeedStop(
        "--seed", static_cast<long long>(absl::GetFlag(FLAGS_seed)));
    if (!stop.empty()) {
      std::cerr << stop << "\n";
      return 1;
    }
  }
  // PWO-5 gate 2 item (a): reject a NEGATIVE --specimen_exchange_penalty
  // fatally, after argument parsing and before anything else runs.
  if (absl::GetFlag(FLAGS_specimen_exchange_penalty) < 0.0) {
    std::cerr << "Fatal: --specimen_exchange_penalty="
              << absl::GetFlag(FLAGS_specimen_exchange_penalty)
              << " is NEGATIVE. This flag is a MAGNITUDE THAT IS SUBTRACTED "
                 "from the reward of a ConvertSpecimenToTroop transition, so a "
                 "negative value is a BONUS on the breadcrumb behaviour the "
                 "term exists to penalize (this is the u175-lineage defect). "
                 "Pass a value >= 0; 0.0 disables the term."
              << std::endl;
    return -1;
  }
  if (absl::GetFlag(FLAGS_family_atomics_penalty) < 0.0) {
    std::cerr << "Fatal: --family_atomics_penalty="
              << absl::GetFlag(FLAGS_family_atomics_penalty)
              << " is NEGATIVE. This flag is a MAGNITUDE THAT IS SUBTRACTED "
                 "from the reward, so a negative value is a BONUS. "
                 "Pass a value >= 0; 0.0 disables the term."
              << std::endl;
    return -1;
  }
  if (absl::GetFlag(FLAGS_plot_intrigue_penalty) < 0.0) {
    std::cerr << "Fatal: --plot_intrigue_penalty="
              << absl::GetFlag(FLAGS_plot_intrigue_penalty)
              << " is NEGATIVE. This flag is a MAGNITUDE THAT IS SUBTRACTED "
                 "from the reward, so a negative value is a BONUS. "
                 "Pass a value >= 0; 0.0 disables the term."
              << std::endl;
    return -1;
  }
  if (absl::GetFlag(FLAGS_shaped_reward_weight) != 0.0 ||
      absl::GetFlag(FLAGS_tleilaxu_breadcrumb_weight) != 0.0 ||
      absl::GetFlag(FLAGS_tleilaxu_level7_breadcrumb_weight) != 0.0 ||
      absl::GetFlag(FLAGS_specimen_exchange_penalty) != 0.0 ||
      absl::GetFlag(FLAGS_family_atomics_penalty) != 0.0 ||
      absl::GetFlag(FLAGS_plot_intrigue_penalty) != 0.0) {
    if (!absl::GetFlag(FLAGS_allow_shaping)) {
      std::cerr << "Fatal: Reward shaping weights are non-zero but --allow_shaping is not set. "
                << "To run with reward shaping, pass --allow_shaping=true." << std::endl;
      return -1;
    } else {
      std::cout << "Warning: Running with non-zero experimental reward shaping weights!" << std::endl;
    }
  }
  if (absl::GetFlag(FLAGS_deterministic)) {
    at::globalContext().setDeterministicAlgorithms(true, /*silent=*/true);
  }

#ifndef OPEN_SPIEL_BUILD_WITH_LIBTORCH
  std::cerr << "dune_ppo_train requires OPEN_SPIEL_BUILD_WITH_LIBTORCH.\n";
  return 1;
#else
  at::set_num_threads(1);
  at::set_num_interop_threads(1);

  ctx.master = static_cast<uint64_t>(absl::GetFlag(FLAGS_seed));
  torch::manual_seed(dune_seed::DeriveSeed(ctx.master, dune_seed::kStreamModelInit));

  ctx.game = open_spiel::LoadGame(absl::GetFlag(FLAGS_game));
  ctx.obs_size = ctx.game->GetType().provides_information_state_tensor
                     ? ctx.game->InformationStateTensorSize()
                     : ctx.game->ObservationTensorSize();
  const std::string market_mode_str = absl::GetFlag(FLAGS_market_appendix_mode);
  dune_imperium::MarketAppendixMode market_mode =
      dune_imperium::ParseMarketAppendixMode(market_mode_str);
  if (market_mode != dune_imperium::MarketAppendixMode::kNone) {
    ctx.obs_size = dune_imperium::GetInformationStateSchema(market_mode).size;
  }
  ctx.action_size = ctx.game->NumDistinctActions();

  ctx.device = torch::Device(torch::kCPU);
  if (torch::cuda::is_available()) {
    ctx.device = torch::Device(torch::kCUDA);
    at::globalContext().setAllowTF32CuBLAS(
        absl::GetFlag(FLAGS_allow_tf32));
    at::globalContext().setAllowTF32CuDNN(
        absl::GetFlag(FLAGS_allow_tf32));
    at::autocast::set_autocast_dtype(at::kCUDA, at::ScalarType::BFloat16);
    std::cout << "CUDA available. PPO training on GPU.\n";
  }
  if (numerical_parity && !ctx.device.is_cuda()) {
    SpielFatalError(
        "Raw-PPO numerical parity requires CUDA: without it the rollout side "
        "does not execute BF16 autocast and cannot test the registered gate.");
  }
  if (numerical_parity) {
    // Source identity is verified before model load and, critically, before a
    // rollout worker can be created. A mismatch cannot produce a partial
    // collection that looks like evidence from the registered implementation.
    ctx.parity_source_provenance =
        open_spiel::ppo_train_internal::LoadPpoNumericalParitySourceProvenance(
            absl::GetFlag(FLAGS_numerical_parity_source_root),
            absl::GetFlag(FLAGS_numerical_parity_source_sha256));
  }

  ctx.enable_scorer = absl::GetFlag(FLAGS_enable_semantic_scorer);
  ctx.enable_zone_enc = absl::GetFlag(FLAGS_enable_zone_encoder);
  ctx.sem_schema = absl::GetFlag(FLAGS_semantic_descriptor_schema);
  if (dune_imperium::IsFullPublicInformationStateSize(ctx.obs_size) && !ctx.enable_scorer) {
    SpielFatalError("Full public information actor requires --enable_semantic_scorer=true");
  }
  if ((absl::GetFlag(FLAGS_market_appendix_mode) == "full_public_information_v4" ||
       absl::GetFlag(FLAGS_market_appendix_mode) == "full_public_information_v5") &&
      ctx.sem_schema != "semantic_action_v4" &&
      ctx.sem_schema != "semantic_action_v5" &&
      !absl::GetFlag(FLAGS_allow_legacy_v3_on_v4_obs)) {
    SpielFatalError(absl::StrFormat(
        "market_appendix_mode=%s requires semantic_descriptor_schema=semantic_action_v4 or semantic_action_v5, "
        "but found '%s'. Pass --allow_legacy_v3_on_v4_obs=true to override.",
        absl::GetFlag(FLAGS_market_appendix_mode),
        ctx.sem_schema));
  }
  ctx.training_model = std::make_shared<open_spiel::SharedDunePolicyValueNetImpl>(
      ctx.obs_size, absl::GetFlag(FLAGS_hidden_dim), ctx.action_size,
      absl::GetFlag(FLAGS_num_blocks), absl::GetFlag(FLAGS_nonlinear_value_head),
      /*with_aux_heads=*/false, /*head_init_seed=*/0,
      ctx.enable_scorer, ctx.enable_zone_enc, ctx.sem_schema);
  ctx.inference_model = std::make_shared<open_spiel::SharedDunePolicyValueNetImpl>(
      ctx.obs_size, absl::GetFlag(FLAGS_hidden_dim), ctx.action_size,
      absl::GetFlag(FLAGS_num_blocks), absl::GetFlag(FLAGS_nonlinear_value_head),
      /*with_aux_heads=*/false, /*head_init_seed=*/0,
      ctx.enable_scorer, ctx.enable_zone_enc, ctx.sem_schema);
  if (!ctx.sem_schema.empty()) {
    ctx.training_model->SetSemanticDescriptorSchema(ctx.sem_schema);
    ctx.inference_model->SetSemanticDescriptorSchema(ctx.sem_schema);
    if (absl::GetFlag(FLAGS_slot_indexed_out_layer_f)) {
      if (ctx.training_model->semantic_scorer_) {
        ctx.training_model->semantic_scorer_->EnableF(true);
      }
      if (ctx.inference_model->semantic_scorer_) {
        ctx.inference_model->semantic_scorer_->EnableF(true);
      }
    }
  }
  ctx.training_model->to(ctx.device);
  ctx.inference_model->to(ctx.device);

  const std::string m_mode_str = absl::GetFlag(FLAGS_market_appendix_mode);
  if (m_mode_str != "none") {
    auto parsed_mode = dune_imperium::ParseMarketAppendixMode(m_mode_str);
    ctx.training_model->market_appendix_mode_ = parsed_mode;
    ctx.inference_model->market_appendix_mode_ = parsed_mode;
  }

  ctx.resolved_target_deck_pool_semantics =
      dune_imperium::ParseDeckPoolSemanticsOrDie(absl::GetFlag(FLAGS_deck_pool_semantics));
  ctx.training_model->deck_pool_semantics_ = ctx.resolved_target_deck_pool_semantics;
  ctx.inference_model->deck_pool_semantics_ = ctx.resolved_target_deck_pool_semantics;

  ctx.ema_model = nullptr;
  ctx.ema_updates_absorbed = 0;
  ctx.ema_half_life = absl::GetFlag(FLAGS_ema_half_life);
  ctx.ema_decay = absl::GetFlag(FLAGS_ema_decay);
  if (ctx.ema_decay <= 0.0 && ctx.ema_half_life > 0.0) {
    ctx.ema_decay = std::pow(0.5, 1.0 / ctx.ema_half_life);
  }
  if (absl::GetFlag(FLAGS_enable_ema_tracking)) {
    ctx.ema_model = std::make_shared<open_spiel::SharedDunePolicyValueNetImpl>(
        ctx.obs_size, absl::GetFlag(FLAGS_hidden_dim), ctx.action_size,
        absl::GetFlag(FLAGS_num_blocks), absl::GetFlag(FLAGS_nonlinear_value_head),
        /*with_aux_heads=*/false, /*head_init_seed=*/0,
        ctx.enable_scorer, ctx.enable_zone_enc, ctx.sem_schema);
    if (!ctx.sem_schema.empty()) {
      ctx.ema_model->SetSemanticDescriptorSchema(ctx.sem_schema);
      if (absl::GetFlag(FLAGS_slot_indexed_out_layer_f) && ctx.ema_model->semantic_scorer_) {
        ctx.ema_model->semantic_scorer_->EnableF(true);
      }
    }
    ctx.ema_model->to(ctx.device);
    if (m_mode_str != "none") {
      ctx.ema_model->market_appendix_mode_ = ctx.training_model->market_appendix_mode_;
    }
    ctx.ema_model->deck_pool_semantics_ = ctx.resolved_target_deck_pool_semantics;
    std::cout << absl::StrFormat(
        "[EMA] Initialized EMA tracking model: half_life=%.1f, decay=%.8f\n",
        ctx.ema_half_life, ctx.ema_decay);
  }
  if (ctx.obs_size == dune_imperium::kExpandedInformationStateSize) {
    if (absl::GetFlag(FLAGS_init_mode) == "random") {
      if (m_mode_str != "zeros" && m_mode_str != "ordered_market") {
        SpielFatalError("Fresh 6,215 model requires explicit --market_appendix_mode ('zeros' or 'ordered_market')");
      }
    }
  }

  const std::string save_init_path = absl::GetFlag(FLAGS_save_initial_model_and_exit);
  if (!save_init_path.empty()) {
    std::cout << "[INIT-EXPORT] Saving initial model to: " << save_init_path << std::endl;
    torch::save(ctx.training_model, save_init_path);
    std::cout << "[INIT-EXPORT] Initial model saved successfully. Exiting." << std::endl;
    return 0;
  }

  if (absl::GetFlag(FLAGS_separate_actor_critic)) {
    if (absl::GetFlag(FLAGS_sample_counterfactual_states)) {
      SpielFatalError("--separate_actor_critic cannot be combined with --sample_counterfactual_states");
    }
    const int64_t critic_obs_size =
        absl::GetFlag(FLAGS_privileged_critic)
            ? (ctx.obs_size + open_spiel::kPrivilegedCriticAppendixSize)
            : ctx.obs_size;
    ctx.training_critic = std::make_shared<open_spiel::SharedDunePolicyValueNetImpl>(
        critic_obs_size, absl::GetFlag(FLAGS_hidden_dim), ctx.action_size,
        absl::GetFlag(FLAGS_num_blocks), absl::GetFlag(FLAGS_nonlinear_value_head));
    ctx.inference_critic = std::make_shared<open_spiel::SharedDunePolicyValueNetImpl>(
        critic_obs_size, absl::GetFlag(FLAGS_hidden_dim), ctx.action_size,
        absl::GetFlag(FLAGS_num_blocks), absl::GetFlag(FLAGS_nonlinear_value_head));
    ctx.training_critic->to(ctx.device);
    ctx.inference_critic->to(ctx.device);
    ctx.training_critic->deck_pool_semantics_ = ctx.resolved_target_deck_pool_semantics;
    ctx.inference_critic->deck_pool_semantics_ = ctx.resolved_target_deck_pool_semantics;
  }

  if (absl::GetFlag(FLAGS_reverse_kl_coef) > 0.0) {
    ctx.collection_model = std::make_shared<open_spiel::SharedDunePolicyValueNetImpl>(
        ctx.obs_size, absl::GetFlag(FLAGS_hidden_dim), ctx.action_size,
        absl::GetFlag(FLAGS_num_blocks), absl::GetFlag(FLAGS_nonlinear_value_head),
        /*with_aux_heads=*/false, /*head_init_seed=*/0, ctx.enable_scorer);
    ctx.collection_model->to(ctx.device);
    ctx.collection_model->deck_pool_semantics_ = ctx.resolved_target_deck_pool_semantics;
  }

  if (absl::GetFlag(FLAGS_sample_counterfactual_states) && !absl::GetFlag(FLAGS_train_value_only)) {
    SpielFatalError("Counterfactual states sampling can only be enabled during value-only training (train_value_only=true).");
  }
  if (absl::GetFlag(FLAGS_sample_counterfactual_states) &&
      (absl::GetFlag(FLAGS_counterfactual_samples_per_game) <= 0 ||
       absl::GetFlag(FLAGS_counterfactual_replay_weight) <= 0)) {
    SpielFatalError("Counterfactual sample and replay counts must both be positive.");
  }

  if (absl::GetFlag(FLAGS_train_value_only)) {
    if (absl::GetFlag(FLAGS_reward_scale) != 4.0) {
      SpielFatalError("reward_scale must be exactly 4.0 when train_value_only is active.");
    }
    {
      std::string corpus_path = "data/dune_diagnostic_corpus.json";
      std::ifstream f(corpus_path);
      if (!f.good()) {
        SpielFatalError(absl::StrCat("CRITICAL EXCEPTION: Prohibited histories corpus file ", corpus_path, " is missing! Fail-closed boundary triggered."));
      }
      std::string content((std::istreambuf_iterator<char>(f)),
                           std::istreambuf_iterator<char>());
      auto json_parsed = open_spiel::json::FromString(content);
      if (!json_parsed) {
        SpielFatalError(absl::StrCat("CRITICAL EXCEPTION: Failed to parse prohibited histories corpus from ", corpus_path, "!"));
      }
      auto json_arr = json_parsed.value().GetArray();
      for (const auto& item_val : json_arr) {
        auto item = item_val.GetObject();
        auto history_arr = item.at("history").GetArray();
        std::vector<open_spiel::Action> hist;
        for (const auto& act_val : history_arr) {
          hist.push_back(static_cast<open_spiel::Action>(act_val.GetInt()));
        }
        prohibited_histories.insert(hist);
      }
      std::cout << "[INFO] Loaded " << prohibited_histories.size()
                << " prohibited histories from official corpus to prevent training leakage.\n";
    }
    std::cout << "[INFO] train_value_only is true: freezing trunk & policy head parameters.\n";
    if (absl::GetFlag(FLAGS_unfreeze_trunk)) {
      if (absl::GetFlag(FLAGS_policy_kl_anchor_coeff) <= 0.0) {
        SpielFatalError("CRITICAL SAFETY ERROR: policy_kl_anchor_coeff must be positive when unfreezing the trunk layer in value-only training mode.");
      }
    } else {
      for (auto& p : ctx.training_model->input_layer->parameters()) {
        p.set_requires_grad(false);
      }
      for (auto& block : ctx.training_model->res_blocks) {
        for (auto& p : block->parameters()) {
          p.set_requires_grad(false);
        }
      }
    }
    for (auto& p : ctx.training_model->policy_head->parameters()) {
      p.set_requires_grad(false);
    }
  }

  ctx.training_model->train();
  ctx.inference_model->eval();

  ctx.optimizer = MakeOptimizer(ctx.training_model);
  if (absl::GetFlag(FLAGS_separate_actor_critic) && ctx.training_critic != nullptr) {
    ctx.critic_optimizer = MakeOptimizer(ctx.training_critic);
  }

  if (absl::GetFlag(FLAGS_train_value_only) && absl::GetFlag(FLAGS_unfreeze_trunk)) {
    ctx.anchor_model = std::make_shared<open_spiel::SharedDunePolicyValueNetImpl>(
        ctx.obs_size, absl::GetFlag(FLAGS_hidden_dim), ctx.action_size,
        absl::GetFlag(FLAGS_num_blocks), absl::GetFlag(FLAGS_nonlinear_value_head),
        /*with_aux_heads=*/false, /*head_init_seed=*/0,
        ctx.enable_scorer, ctx.enable_zone_enc, ctx.sem_schema);
    if (!ctx.sem_schema.empty()) {
      ctx.anchor_model->SetSemanticDescriptorSchema(ctx.sem_schema);
      if (absl::GetFlag(FLAGS_slot_indexed_out_layer_f) && ctx.anchor_model->semantic_scorer_) {
        ctx.anchor_model->semantic_scorer_->EnableF(true);
      }
    }
    ctx.anchor_model->to(ctx.device);
    ctx.anchor_model->deck_pool_semantics_ = ctx.resolved_target_deck_pool_semantics;
  }

  ctx.init_mode = absl::GetFlag(FLAGS_init_mode);
  if (ctx.init_mode.empty()) {
    SpielFatalError("Required flag --init_mode is missing. Must be 'random', 'checkpoint', 'bootstrap', or 'validate_legacy'.");
  }

  // WO-PERF-TIMING: --pipeline=true shares the compute stream with CollectRollout.
  if (absl::GetFlag(FLAGS_phase_timing_mode) == "phases" &&
      absl::GetFlag(FLAGS_pipeline)) {
    SpielFatalError(
        "--phase_timing_mode=phases is incompatible with --pipeline=true: the "
        "background collection thread shares the compute stream, so every "
        "per-phase device time would silently include collector kernels. Run "
        "timed arms with --pipeline=false.");
  }

  ctx.config_fingerprint = open_spiel::ppo_train_internal::ComputeConfigFingerprint();
  ctx.pre_precision_config_fingerprint =
      open_spiel::ppo_train_internal::ComputeCurrentPrePrecisionConfigFingerprint();
  ctx.search_label_fingerprint = open_spiel::ppo_train_internal::GetSearchLabelFingerprint(
      absl::GetFlag(FLAGS_search_label_dir));
  ctx.run_uuid = open_spiel::ppo_train_internal::GenerateUUID();
  ctx.gate_session_nonce = open_spiel::ppo_train_internal::GenerateUUID();
  if (absl::GetFlag(FLAGS_pause_for_gate_at_update) > 0) {
    const std::string gdf = absl::GetFlag(FLAGS_gate_decision_file);
    if (gdf.empty()) {
      open_spiel::SpielFatalError(
          "--pause_for_gate_at_update > 0 requires a non-empty --gate_decision_file.");
    }
    if (std::filesystem::exists(gdf)) {
      open_spiel::SpielFatalError(
          "--gate_decision_file already exists at launch; refusing stale decision file: " + gdf);
    }
  }

  ctx.total_env_steps.store(0);
  ctx.next_episode_id.store(0);
  ctx.start_update = 1;
  ctx.target_end_update = absl::GetFlag(FLAGS_target_end_update);
  return std::nullopt;
#endif  // OPEN_SPIEL_BUILD_WITH_LIBTORCH
}

}  // namespace open_spiel::ppo_train_internal
