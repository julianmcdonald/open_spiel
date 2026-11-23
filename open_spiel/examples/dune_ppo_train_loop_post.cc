#include "dune_ppo_train_run_context.h"

#if defined(__GLIBC__)
#include <malloc.h>
#endif

namespace open_spiel::ppo_train_internal {

#ifdef OPEN_SPIEL_BUILD_WITH_LIBTORCH

void PpoTrainRunContext::SaveAuxiliarySidecars(int ckpt_update) {
  std::filesystem::path run_pfx(absl::GetFlag(FLAGS_run_prefix));
  std::filesystem::path ckpt_dir = run_pfx.parent_path();
  if (ckpt_dir.empty()) ckpt_dir = ".";
  std::string run_stem = run_pfx.filename().string();
  if (run_stem.empty()) run_stem = "ppo";
  if (absl::GetFlag(FLAGS_r765_replay_window_updates) > 1) {
    const std::string replay_out =
        (ckpt_dir / absl::StrCat(run_stem, "_r765_replay_update_", ckpt_update, ".bin")).string();
    if (!r765_replay_mgr.SaveState(replay_out, &r765_preserve_guard)) {
      SpielFatalError("Failed to save R7-65 replay sidecar at: " + replay_out);
    }
  }
  if (v5_slot_residual_state.N > 0 &&
      !v5_slot_residual_state.init_w_residuals.empty()) {
    const std::string v5_sidecar_out =
        (ckpt_dir / absl::StrCat(run_stem, "_v5_anneal_state.json")).string();
    if (!v5_slot_residual_state.SaveSidecar(v5_sidecar_out)) {
      SpielFatalError("Failed to save V5 anneal sidecar at: " + v5_sidecar_out);
    }
  }
}

namespace {

void SaveCheckpointBundle(
    PpoTrainRunContext& ctx,
    int ckpt_update,
    bool sep_ac,
    bool is_final) {
  std::string prefix = absl::GetFlag(FLAGS_run_prefix);
  std::string suffix = is_final ? "_final.pt" : absl::StrCat("_update_", ckpt_update, ".pt");
  if (sep_ac) {
    std::string actor_model_path = absl::StrCat(prefix, "_model", suffix);
    std::string actor_optim_path = absl::StrCat(prefix, "_optimizer", suffix);
    std::string critic_model_path = absl::StrCat(prefix, "_critic_model", suffix);
    std::string critic_optim_path = absl::StrCat(prefix, "_critic_optimizer", suffix);
    open_spiel::SaveDualCheckpoint(
        ctx.training_model, *ctx.optimizer, ctx.training_critic, *ctx.critic_optimizer,
        actor_model_path, actor_optim_path, critic_model_path, critic_optim_path,
        ckpt_update, ctx.target_end_update, ctx.total_env_steps.load(),
        ctx.next_episode_id.load(), ctx.master,
        absl::GetFlag(FLAGS_seed_scheme_version), ctx.config_fingerprint,
        ctx.search_label_fingerprint, ctx.run_uuid,
        absl::GetFlag(FLAGS_model_checkpoint), absl::GetFlag(FLAGS_optim_checkpoint));
  } else {
    std::string model_path = absl::StrCat(prefix, "_model", suffix);
    std::string optim_path = absl::StrCat(prefix, "_optimizer", suffix);
    open_spiel::OnlineCollectionState aux_ckpt = ctx.BuildAuxState();
    SaveCheckpoint(
        ctx.training_model, *ctx.optimizer, model_path, optim_path,
        ckpt_update, ctx.target_end_update, ctx.total_env_steps.load(),
        ctx.next_episode_id.load(), ctx.master,
        absl::GetFlag(FLAGS_seed_scheme_version), ctx.config_fingerprint,
        ctx.search_label_fingerprint, ctx.run_uuid,
        ctx.online_search_collection ? &aux_ckpt : nullptr,
        ctx.ema_model, ctx.ema_updates_absorbed,
        ctx.ema_half_life, ctx.ema_decay);
  }
}

void PrintUpdateTelemetry(const PpoTrainRunContext& ctx, const PpoTrainStepState& step) {
  const int update = step.update;
  auto wall_end = std::chrono::high_resolution_clock::now();

  double collect_elapsed = ctx.current_collect.elapsed_seconds;
  double wall_elapsed =
      std::chrono::duration<double>(wall_end - step.wall_start).count() +
      (step.have_bg ? 0.0 : collect_elapsed);
  double sps = collect_elapsed > 0.0
                   ? ctx.current_collect.rollout.size() / collect_elapsed
                   : 0.0;

  std::cout << absl::StrFormat(
      "Update %5d/%d | Transitions: %6zu | Games: %llu | "
      "Collect: %.2fs (%.0f t/s) | PPO: %.2fs | Wall: %.2fs | "
      "PolicyLoss: %.6f | ValueLoss: %.6f | Entropy: %.4f | "
      "ApproxKL: %.6f | ClipFrac: %.3f | ExplVar: %.3f%s%s%s\n",
      update, ctx.target_end_update, ctx.current_collect.rollout.size(),
      static_cast<unsigned long long>(ctx.total_games),
      collect_elapsed, sps, step.ppo_elapsed, wall_elapsed,
      step.stats.policy_loss, step.stats.value_loss, step.stats.entropy, step.stats.approx_kl,
      step.stats.clip_fraction, step.stats.explained_variance,
      (absl::GetFlag(FLAGS_reverse_kl_coef) > 0.0 ? absl::StrFormat(" | RevKL: %.6f", step.stats.reverse_kl) : ""),
      (ctx.ema_model != nullptr ? absl::StrFormat(" | MagnetKL: %.6f", step.stats.ema_magnet_kl) : ""),
      step.stats.early_stopped ? " | early-stop" : "");

  // Phase 18B combined-optimization + collector diagnostics.
  if (ctx.online_search_collection) {
    const open_spiel::OnlineSearchCollectionStats& cst = ctx.current_collect.aux_stats;
    auto acc = [](const open_spiel::PerRoleSearchStats& r) {
      return r.searches > 0 ? static_cast<double>(r.accepted) / r.searches : 0.0;
    };
    auto mkl = [](const open_spiel::PerRoleSearchStats& r) {
      return r.searches > 0 ? r.sum_kl / r.searches : 0.0;
    };
    std::cout << absl::StrFormat(
        "  [18B Aux] used=%d coef=%.4f ce=%.4f vmse=%.4f | aux_norm=%.4f "
        "ppo_norm=%.4f ratio=%.3f\n",
        step.stats.aux_examples_used, step.this_search_coef, step.stats.aux_ce,
        step.stats.aux_value_mse, step.stats.aux_grad_norm_mean, step.stats.ppo_grad_norm_mean,
        step.stats.aux_ppo_norm_ratio);
    std::cout << absl::StrFormat(
        "  [18B Collector] accepted=%d rejected=%d wall=%.1fs | acc/KL "
        "primary %.2f/%.2f cont %.2f/%.2f purch %.2f/%.2f | SM granted=%d organic=%d\n",
        cst.accepted_targets, cst.rejected_incomplete, cst.collection_wall_time_s,
        acc(cst.by_role[0]), mkl(cst.by_role[0]), acc(cst.by_role[1]),
        mkl(cst.by_role[1]), acc(cst.by_role[2]), mkl(cst.by_role[2]),
        static_cast<int>(cst.swordmaster_granted_games),
        static_cast<int>(cst.swordmaster_organic_games));
  }

  if (absl::GetFlag(FLAGS_search_in_training)) {
    std::cout << absl::StrFormat(
        "  [Search-in-Training] eligible_decisions=%lu selected_roots=%d used_by_optimizer=%d (%.2f%% eligible, %.4f%% transitions) | "
        "turns=%d decisions=%d overrides=%d (%.1f%%) fallbacks=%d | "
        "time=%.2fs (%.2f turns/s) | lambda=%.4f aux_ce=%.4f\n",
        ctx.current_collect.search_eligible_decisions,
        step.search_sup_res.roots_evaluated,
        step.stats.aux_examples_used,
        ctx.current_collect.search_eligible_decisions > 0
            ? (100.0 * step.search_sup_res.roots_evaluated / ctx.current_collect.search_eligible_decisions)
            : 0.0,
        ctx.current_collect.rollout.size() > 0
            ? (100.0 * step.search_sup_res.roots_evaluated / ctx.current_collect.rollout.size())
            : 0.0,
        step.search_sup_res.roots_evaluated,
        step.search_sup_res.total_decisions_evaluated,
        step.search_sup_res.overrides,
        step.search_sup_res.mean_override_rate * 100.0,
        step.search_sup_res.fallbacks_to_mu,
        step.search_sup_res.elapsed_seconds,
        step.search_sup_res.elapsed_seconds > 0.0
            ? step.search_sup_res.roots_evaluated / step.search_sup_res.elapsed_seconds : 0.0,
        step.this_search_coef,
        step.stats.aux_ce);
  }

  std::cout << absl::StrFormat(
      "  [Conflict VP Stats] Generated: %.2f | Attributed: %.2f | Unattributed: %.2f\n",
      ctx.current_collect.conflict_vp_generated,
      ctx.current_collect.conflict_vp_attributed,
      ctx.current_collect.conflict_vp_unattributed);

  {
    uint64_t total_acquisitions = 0;
    for (int i = 0; i < 4; ++i) total_acquisitions += ctx.current_collect.sm_acquisitions_by_seat[i];
    if (total_acquisitions > 0) {
      std::cout << absl::StrFormat(
          "  [Natural SM Acquisitions] Total: %d | Seat P0: %d | P1: %d | P2: %d | P3: %d\n",
          total_acquisitions, ctx.current_collect.sm_acquisitions_by_seat[0],
          ctx.current_collect.sm_acquisitions_by_seat[1], ctx.current_collect.sm_acquisitions_by_seat[2],
          ctx.current_collect.sm_acquisitions_by_seat[3]);
      std::cout << absl::StrFormat(
          "  [SM Acquisitions by Round] R1: %d | R2: %d | R3: %d | R4: %d | R5: %d | R6-10: %d\n",
          ctx.current_collect.sm_acquisitions_by_round[1], ctx.current_collect.sm_acquisitions_by_round[2],
          ctx.current_collect.sm_acquisitions_by_round[3], ctx.current_collect.sm_acquisitions_by_round[4],
          ctx.current_collect.sm_acquisitions_by_round[5],
          ctx.current_collect.sm_acquisitions_by_round[6] + ctx.current_collect.sm_acquisitions_by_round[7] +
          ctx.current_collect.sm_acquisitions_by_round[8] + ctx.current_collect.sm_acquisitions_by_round[9] +
          ctx.current_collect.sm_acquisitions_by_round[10]);
      std::cout << absl::StrFormat(
          "  [SM Acquisitions by Leader] Leto (4): %d | Rabban (6): %d | Other Leaders: %d\n",
          ctx.current_collect.sm_acquisitions_by_leader[4], ctx.current_collect.sm_acquisitions_by_leader[6],
          total_acquisitions - ctx.current_collect.sm_acquisitions_by_leader[4] - ctx.current_collect.sm_acquisitions_by_leader[6]);
    }

    const double specimen_exchange_penalty = absl::GetFlag(FLAGS_specimen_exchange_penalty);
    const double family_atomics_penalty = absl::GetFlag(FLAGS_family_atomics_penalty);
    const double plot_intrigue_penalty = absl::GetFlag(FLAGS_plot_intrigue_penalty);
    const int plot_intrigue_exemption_threshold = absl::GetFlag(FLAGS_plot_intrigue_exemption_threshold);

    if (specimen_exchange_penalty != 0.0 || family_atomics_penalty != 0.0 || plot_intrigue_penalty != 0.0 ||
        ctx.current_collect.specimen_conversions > 0 || ctx.current_collect.atomics_penalized > 0 ||
        ctx.current_collect.atomics_exempt > 0 || ctx.current_collect.plot_penalized > 0 ||
        ctx.current_collect.plot_exempt > 0) {
      std::cout << absl::StrFormat(
          "  [Penalty Telemetry] Specimen: %d conv, -%.4f ded (coef=%.4f, mult=%.2f) | "
          "Atomics: %d pen, %d exempt, -%.4f ded (coef=%.4f, mult=%.2f) | "
          "Plot: %d pen, %d exempt, -%.4f ded (coef=%.4f, thresh=%d, mult=%.2f)\n",
          ctx.current_collect.specimen_conversions,
          ctx.current_collect.specimen_penalty_deducted,
          specimen_exchange_penalty, ctx.reward_lambda,
          ctx.current_collect.atomics_penalized,
          ctx.current_collect.atomics_exempt,
          ctx.current_collect.atomics_penalty_deducted,
          family_atomics_penalty, ctx.reward_lambda,
          ctx.current_collect.plot_penalized,
          ctx.current_collect.plot_exempt,
          ctx.current_collect.plot_penalty_deducted,
          plot_intrigue_penalty,
          plot_intrigue_exemption_threshold,
          ctx.reward_lambda);
    }
    if (absl::GetFlag(FLAGS_purchase_exploration) || ctx.current_collect.forced_buys > 0) {
      std::cout << absl::StrFormat("  [Purchase Exploration] Forced buys: %d\n", ctx.current_collect.forced_buys);
    }
    if (absl::GetFlag(FLAGS_targeted_buy_exploration) ||
        ctx.current_collect.targeted_buy_achieved_card11 > 0 ||
        ctx.current_collect.targeted_buy_achieved_card13 > 0 ||
        ctx.current_collect.targeted_buy_designated_opps_card11 > 0 ||
        ctx.current_collect.targeted_buy_designated_opps_card13 > 0) {
      std::cout << absl::StrFormat(
          "  [Targeted Buy Exploration] Biased exploration active (policy log-prob retained)\n"
          "    Scientific Breakthrough (Tleilaxu 11 / Card 118): Quota: %d/%d distinct games | Designated Opps (R<=%d): %d | Total Opps: %d\n"
          "    Stitched Horror         (Tleilaxu 13 / Card 120): Quota: %d/%d distinct games | Designated Opps (R<=%d): %d | Total Opps: %d\n",
          ctx.current_collect.targeted_buy_achieved_card11, absl::GetFlag(FLAGS_targeted_buy_quota_per_card),
          absl::GetFlag(FLAGS_targeted_buy_max_round), ctx.current_collect.targeted_buy_designated_opps_card11,
          ctx.current_collect.targeted_buy_total_opps_card11,
          ctx.current_collect.targeted_buy_achieved_card13, absl::GetFlag(FLAGS_targeted_buy_quota_per_card),
          absl::GetFlag(FLAGS_targeted_buy_max_round), ctx.current_collect.targeted_buy_designated_opps_card13,
          ctx.current_collect.targeted_buy_total_opps_card13);

      std::cout << "    Breakdown by Card and Round:\n";
      for (int c = 0; c < 2; ++c) {
        std::string card_name = (c == 0) ? "Scientific Breakthrough" : "Stitched Horror";
        for (int r = 1; r <= 6; ++r) {
          uint64_t u_opps = ctx.current_collect.targeted_buy_telemetry.unforced_opps[c][r];
          double u_prob = (u_opps > 0) ? (ctx.current_collect.targeted_buy_telemetry.unforced_sum_prob[c][r] / u_opps) : 0.0;
          uint64_t u_buys = ctx.current_collect.targeted_buy_telemetry.unforced_buys[c][r];
          double u_adv = (u_buys > 0) ? (ctx.current_collect.targeted_buy_telemetry.unforced_sum_adv[c][r] / u_buys) : 0.0;
          uint64_t f_buys = ctx.current_collect.targeted_buy_telemetry.forced_buys[c][r];
          double f_adv = (f_buys > 0) ? (ctx.current_collect.targeted_buy_telemetry.forced_sum_adv[c][r] / f_buys) : 0.0;
          if (u_opps > 0 || f_buys > 0) {
            std::cout << absl::StrFormat(
                "      [%s R%d] Unforced: %d opps, avg_prob=%.4f, %d buys, avg_adv=%+.4f | Forced: %d buys, avg_adv=%+.4f\n",
                card_name, r, u_opps, u_prob, u_buys, u_adv, f_buys, f_adv);
          }
        }
      }
    }
  }

  if (step.search_lambda > 0.0 && ctx.search_buffer.Size() > 0) {
    double avg_search_kl = (step.search_minibatches_per_update > 0)
        ? (step.search_kl_sum / step.search_minibatches_per_update) : 0.0;
    double avg_search_grad = (step.search_minibatches_per_update > 0)
        ? (step.search_grad_sum / step.search_minibatches_per_update) : 0.0;
    double avg_ppo_grad = (step.stats.grad_norm_count > 0)
        ? (step.stats.grad_norm_sum / step.stats.grad_norm_count) : 0.0;
    double ratio = (avg_ppo_grad > 0.0) ? (avg_search_grad / avg_ppo_grad) : 0.0;
    std::cout << absl::StrFormat(
        "SearchKL: %.3f | SearchGrad: %.3f | PPOGrad: %.3f | Ratio: %.2f | EV: %.3f\n",
        avg_search_kl, avg_search_grad, avg_ppo_grad, ratio, step.stats.explained_variance);
  }
}

}  // namespace

void RunLoopPostPhase(PpoTrainRunContext& ctx, PpoTrainStepState& step) {
  const int update = step.update;

  // Join background collection before syncing models.
  if (step.have_bg) {
    step.bg_collect_thread.join();
    ctx.total_games += step.next_collect.games;
    ctx.total_moves += step.next_collect.moves;
  }

  if (absl::GetFlag(FLAGS_separate_actor_critic) && ctx.training_critic != nullptr) {
    SyncDualModels(ctx.training_model, ctx.inference_model,
                   ctx.training_critic, ctx.inference_critic, &ctx.sync_mutex);
  } else {
    SyncModels(ctx.training_model, ctx.inference_model, &ctx.sync_mutex);
  }
  if (ctx.collection_model != nullptr) {
    CopyModelWeights(ctx.training_model, ctx.collection_model);
  }

  PrintUpdateTelemetry(ctx, step);

  ctx.cum_card11 += ctx.current_collect.card11_metrics;
  ctx.cum_card13 += ctx.current_collect.card13_metrics;

  // Release completed trajectories before saving or collecting their replacement.
  ctx.current_collect = CollectResult{};
#if defined(__GLIBC__)
  // Worker arenas retain freed trajectory pages across updates unless trimmed.
  malloc_trim(0);
#endif

  int checkpoint_interval = absl::GetFlag(FLAGS_checkpoint_interval);
  int training_step = update - ctx.start_update + 1;
  bool is_pilot_update = absl::GetFlag(FLAGS_save_pilot_checkpoints) &&
      (training_step == 10 || training_step == 25 || training_step == 50);
  bool is_target_end = (update == ctx.target_end_update);
  step.is_checkpoint = is_pilot_update ||
      (checkpoint_interval > 0 && update % checkpoint_interval == 0) ||
      is_target_end;
  if (step.is_checkpoint) {
    bool sep_ac = absl::GetFlag(FLAGS_separate_actor_critic) && (ctx.training_critic != nullptr);
    SaveCheckpointBundle(ctx, update, sep_ac, /*is_final=*/false);
    ctx.SaveAuxiliarySidecars(update);
    if (!ctx.card_telemetry_path.empty()) {
      dune_imperium::SaveCardTelemetry(
          ctx.card_telemetry_path, update, ctx.cum_card11, ctx.cum_card13, /*is_checkpoint=*/true);
    }
  }

  for (size_t opp_i = 0; opp_i < ctx.opponent_models.size(); ++opp_i) {
    std::string current_hash = dune_opponent_pool::HashModelParametersAndBuffers(ctx.opponent_models[opp_i]);
    if (current_hash != ctx.opponent_hashes_before[opp_i]) {
      SpielFatalError(absl::StrFormat("CRITICAL ERROR: Opponent model %d mutated during training!", opp_i));
    }
  }

  if (!ctx.card_telemetry_path.empty()) {
    dune_imperium::SaveCardTelemetry(
        ctx.card_telemetry_path, update, ctx.cum_card11, ctx.cum_card13, /*is_checkpoint=*/false);
  }

  ctx.last_completed_update = update;

  const int gate_pause_upd = absl::GetFlag(FLAGS_pause_for_gate_at_update);
  const std::string gate_dec_path = absl::GetFlag(FLAGS_gate_decision_file);
  if (gate_pause_upd > 0 && update == gate_pause_upd && !gate_dec_path.empty()) {
    if (step.have_bg && step.bg_collect_thread.joinable()) {
      step.bg_collect_thread.join();
      step.have_bg = false;
    }
    const double max_wait_s = static_cast<double>(std::max(1, absl::GetFlag(FLAGS_gate_timeout_seconds)));
    std::cout << absl::StrFormat(
        "[GATE-BARRIER] Paused after completing and checkpointing update %d (nonce=%s). Waiting up to %ds for decision in %s ...\n",
        update, ctx.gate_session_nonce, std::max(1, absl::GetFlag(FLAGS_gate_timeout_seconds)), gate_dec_path) << std::flush;
    const auto barrier_start = std::chrono::steady_clock::now();
    while (!g_stop_requested.load(std::memory_order_relaxed)) {
      if (std::chrono::duration<double>(std::chrono::steady_clock::now() - barrier_start).count() >= max_wait_s) {
        std::cout << absl::StrFormat(
            "[GATE-BARRIER] Timeout waiting for gate decision (%ds elapsed); failing closed and releasing GPU.\n",
            static_cast<int>(max_wait_s)) << std::flush;
        g_stop_requested.store(true, std::memory_order_relaxed);
        break;
      }
      std::ifstream dfs(gate_dec_path);
      if (dfs.is_open()) {
        std::string tok, file_nonce;
        int file_upd = -1;
        dfs >> tok >> file_nonce >> file_upd;
        if (tok == "CONTINUE" && file_nonce == ctx.gate_session_nonce && file_upd == update) {
          std::cout << absl::StrFormat(
              "[GATE-BARRIER] Received valid CONTINUE at update %d (nonce=%s). Proceeding to update %d.\n",
              update, file_nonce, update + 1) << std::flush;
          break;
        } else if (!tok.empty()) {
          std::cout << absl::StrFormat(
              "[GATE-BARRIER] Received ABORT or invalid gate token (tok='%s' nonce='%s' upd=%d; expected 'CONTINUE %s %d') at update %d. Stopping cleanly.\n",
              tok, file_nonce, file_upd, ctx.gate_session_nonce, update, update) << std::flush;
          g_stop_requested.store(true, std::memory_order_relaxed);
          break;
        }
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(250));
    }
  }

  int interval = absl::GetFlag(FLAGS_checkpoint_interval);
  bool cutoff_reached = false;
  if (ctx.cutoff_epoch > 0.0) {
    double now_epoch = std::chrono::duration<double>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    if (now_epoch >= ctx.cutoff_epoch && (interval <= 0 || update % interval == 0)) {
      cutoff_reached = true;
    }
  }
  if (g_stop_requested.load(std::memory_order_relaxed)) {
    cutoff_reached = true;
  }

  if (cutoff_reached) {
    std::cout << absl::StrFormat(
        "\n[dune_ppo_train] Stopping endpoint reached at update %d (cutoff=%s, epoch=%.1f, g_stop=%d).\n",
        update, ctx.cutoff_str, ctx.cutoff_epoch, g_stop_requested.load()) << std::flush;

    if (step.have_bg) {
      step.bg_collect_thread.join();
      step.have_bg = false;
    }

    if (!step.is_checkpoint) {
      bool sep_ac = absl::GetFlag(FLAGS_separate_actor_critic) && (ctx.critic_optimizer != nullptr);
      SaveCheckpointBundle(ctx, update, sep_ac, /*is_final=*/false);
      ctx.SaveAuxiliarySidecars(update);
    }

    if (absl::GetFlag(FLAGS_save_final_checkpoint)) {
      std::string prefix = absl::GetFlag(FLAGS_run_prefix);
      bool sep_ac = absl::GetFlag(FLAGS_separate_actor_critic) && (ctx.critic_optimizer != nullptr);
      if (!PublishFinalCheckpointFromExisting(prefix, update, sep_ac)) {
        SaveCheckpointBundle(ctx, update, sep_ac, /*is_final=*/true);
      } else {
        std::cout << "[dune_ppo_train] Reused completed endpoint checkpoint for final export (zero re-serialization).\n";
      }
      ctx.final_checkpoint_saved = true;
    }
    if (!ctx.card_telemetry_path.empty()) {
      dune_imperium::SaveCardTelemetry(
          ctx.card_telemetry_path, update, ctx.cum_card11, ctx.cum_card13, /*is_checkpoint=*/true);
    }
    step.should_break = true;
    return;
  }

  // Advance to next rollout.
  if (step.have_bg) {
    ctx.current_collect = std::move(step.next_collect);
  } else if (update < ctx.target_end_update) {
    float next_reward_lambda = ComputeRewardLambda(ctx.total_env_steps.load(),
                                                   absl::GetFlag(FLAGS_shaping_start_env_steps),
                                                   absl::GetFlag(FLAGS_shaping_decay_env_steps));
    ctx.current_collect = CollectRollout(
        ctx.game.get(), ctx.evaluator, ctx.obs_size, &ctx.total_env_steps,
        ctx.num_threads, &ctx.next_episode_id, ctx.rollout_games,
        next_reward_lambda, ctx.opponent_evaluators, /*global_update=*/update + 1);
    ctx.total_games += ctx.current_collect.games;
    ctx.total_moves += ctx.current_collect.moves;
    // Phase 18B (non-pipelined path): aux for the next update, same snapshot.
    ctx.RunAuxCollection(update + 1, ctx.current_collect);
  }
}

void RunShutdownPhase(PpoTrainRunContext& ctx) {
  auto training_end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = training_end - ctx.training_start;
  auto eval_stats = ctx.evaluator->GetStats();
  std::cout << absl::StrFormat(
      "\n=== PPO Training Complete ===\nElapsed: %.2fs\nEnv Steps: %llu\n"
      "Games: %llu\nMoves: %llu\nEvaluator Requests: %llu | Batches: %llu | "
      "Avg Batch: %.2f | Max Batch: %llu\n",
      elapsed.count(),
      static_cast<unsigned long long>(ctx.total_env_steps.load()),
      static_cast<unsigned long long>(ctx.total_games),
      static_cast<unsigned long long>(ctx.total_moves),
      static_cast<unsigned long long>(eval_stats.requests),
      static_cast<unsigned long long>(eval_stats.batches),
      eval_stats.avg_batch_size,
      static_cast<unsigned long long>(eval_stats.max_batch_size));

  if (absl::GetFlag(FLAGS_save_final_checkpoint) &&
      !ctx.final_checkpoint_saved &&
      ctx.last_completed_update >= ctx.start_update) {
    std::string prefix = absl::GetFlag(FLAGS_run_prefix);
    bool sep_ac = absl::GetFlag(FLAGS_separate_actor_critic) && (ctx.training_critic != nullptr);
    if (!PublishFinalCheckpointFromExisting(prefix, ctx.last_completed_update, sep_ac)) {
      SaveCheckpointBundle(ctx, ctx.last_completed_update, sep_ac, /*is_final=*/true);
    } else {
      std::cout << "[dune_ppo_train] Reused completed endpoint checkpoint for final export (zero re-serialization).\n";
    }
  }
  if (!ctx.card_telemetry_path.empty() && ctx.last_completed_update >= ctx.start_update) {
    dune_imperium::SaveCardTelemetry(
        ctx.card_telemetry_path, ctx.last_completed_update,
        ctx.cum_card11, ctx.cum_card13, /*is_checkpoint=*/true);
  }
  if (!absl::GetFlag(FLAGS_r755_gate2_output).empty()) {
    WriteR755Gate2Json(absl::GetFlag(FLAGS_r755_gate2_output), g_gate2_diagnostics);
  }
}

#endif  // OPEN_SPIEL_BUILD_WITH_LIBTORCH

}  // namespace open_spiel::ppo_train_internal
