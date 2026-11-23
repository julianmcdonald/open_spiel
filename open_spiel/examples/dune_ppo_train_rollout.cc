#include "dune_ppo_train_internal.h"

namespace open_spiel::ppo_train_internal {

#ifdef OPEN_SPIEL_BUILD_WITH_LIBTORCH

size_t PpoRolloutBuffer::PushTrajectory(std::vector<PpoTransition>&& trajectory) {
  std::lock_guard<std::mutex> lock(mu_);
  size_t size = trajectory.size();
  trajectories_.push_back(std::move(trajectory));
  num_transitions_ += size;
  return num_transitions_;
}

size_t PpoRolloutBuffer::Size() const {
  std::lock_guard<std::mutex> lock(mu_);
  return num_transitions_;
}

std::vector<PpoTransition> PpoRolloutBuffer::TakeAll(bool* out_episode_ids_unique) {
  std::lock_guard<std::mutex> lock(mu_);
  if (out_episode_ids_unique) {
    *out_episode_ids_unique = true;
    std::unordered_set<uint64_t> seen_ids;
    for (const auto& traj : trajectories_) {
      if (!traj.empty()) {
        uint64_t ep_id = traj[0].episode_id;
        if (seen_ids.count(ep_id)) {
          *out_episode_ids_unique = false;
        }
        seen_ids.insert(ep_id);
      }
    }
  }
  std::sort(trajectories_.begin(), trajectories_.end(),
            [](const std::vector<PpoTransition>& a,
               const std::vector<PpoTransition>& b) {
              uint64_t id_a = a.empty() ? 0 : a[0].episode_id;
              uint64_t id_b = b.empty() ? 0 : b[0].episode_id;
              return id_a < id_b;
            });
  std::vector<PpoTransition> out;
  out.reserve(num_transitions_);
  for (auto& traj : trajectories_) {
    for (auto& transition : traj) {
      out.push_back(std::move(transition));
    }
  }
  trajectories_.clear();
  num_transitions_ = 0;
  return out;
}

void RolloutWorker(int thread_id, const Game* game,
                    std::shared_ptr<IGameEvaluator> evaluator, int64_t obs_size,
                    PpoRolloutBuffer* rollout_buffer,
                    std::atomic<bool>* stop_collection,
                    std::atomic<uint64_t>* total_env_steps,
                    std::vector<WorkerStats>* worker_stats,
                    std::atomic<uint64_t>* local_episode_id,
                    uint64_t start_episode_id,
                    int rollout_games,
                    float reward_lambda,
                    const std::vector<std::shared_ptr<IGameEvaluator>>& opponent_evaluators,
                    std::vector<open_spiel::dune_imperium::SearchSnapshot>* local_snapshots,
                    int global_update,
                    std::vector<open_spiel::SearchTrainingExample>* worker_aux_examples) {
  uint64_t master = absl::GetFlag(FLAGS_seed);
  WorkerStats local_stats;
  uint64_t search_decision_count = 0;
  std::mt19937_64 search_rng;
  if (local_snapshots != nullptr) {
    uint64_t search_seed = dune_seed::DeriveSeed(master, dune_seed::kDomainTrain, global_update, /*stream=*/0x53454152ULL + thread_id);
    search_rng = std::mt19937_64(search_seed);
  }
  while (true) {
    if (rollout_games == 0 && stop_collection->load(std::memory_order_relaxed)) {
      break;
    }
    uint64_t episode_id = local_episode_id->fetch_add(1, std::memory_order_relaxed);
    if (rollout_games > 0 && episode_id >= start_episode_id + rollout_games) {
      break;
    }
    std::vector<PpoTransition> trajectory;
    int moves = PpoSimulation(master, episode_id, *game, evaluator, obs_size, &trajectory,
                              total_env_steps, reward_lambda, &local_stats,
                              opponent_evaluators,
                              local_snapshots,
                              local_snapshots != nullptr ? &search_decision_count : nullptr,
                              local_snapshots != nullptr ? &search_rng : nullptr,
                              worker_aux_examples);
    local_stats.games += 1;
    local_stats.moves += moves;
    size_t size = rollout_buffer->PushTrajectory(std::move(trajectory));
    if (rollout_games == 0) {
      if (size >= static_cast<size_t>(absl::GetFlag(FLAGS_rollout_transitions))) {
        stop_collection->store(true, std::memory_order_relaxed);
      }
    }
  }
  (*worker_stats)[thread_id] = local_stats;
}

// Phase A: folds the rollout-side pre-cap |z| canaries into the update's stats
// so every WriteDiagnostics site persists them. Called immediately after
// episode_ids_unique is copied, which is the existing seam for carrying
// collection-side facts into the diagnostics row.
// PWO-5 section 14.1c: fold the rollout-side canaries into the update's stats,
// and CHECK the partition identity rather than assuming it.
//
// Two halves, both registered:
//   Sum_r n_r == norm_entropy_n            EXACTLY (integer equality) -- the
//     stronger half; it proves no qualifying decision was dropped or
//     double-counted.
//   |Sum_r (n_r * NE_r) / Sum_r n_r - NE_global| <= 1e-9  -- float slack only.
//
// A violation is a gate-3 failure, not a rounding artifact, so it is fatal here
// rather than reported.
void AttachCanaryStats(open_spiel::PpoUpdateStats* stats,
                       const CollectResult& collect) {
  stats->norm_entropy_n = collect.canary_ne_n;
  stats->norm_entropy =
      collect.canary_ne_n > 0 ? collect.canary_ne_sum / collect.canary_ne_n : 0.0;
  stats->max_action_prob =
      collect.canary_ne_n > 0 ? collect.canary_maxprob_sum / collect.canary_ne_n
                              : 0.0;
  stats->frac_legal_absz_ge_cap_n = collect.canary_sat_logits;
  stats->frac_legal_absz_ge_cap =
      collect.canary_sat_logits > 0
          ? static_cast<double>(collect.canary_sat_ge_cap) /
                static_cast<double>(collect.canary_sat_logits)
          : 0.0;

  int64_t role_n_total = 0;
  double weighted_sum = 0.0;
  for (int r = 0; r < 7; ++r) {
    stats->norm_entropy_n_role[r] = collect.canary_ne_n_role[r];
    // Zero-support semantics: mean is 0.0 when the count is 0, and 0.0 there
    // means "no support this update", never "entropy collapsed to zero".
    stats->norm_entropy_role[r] =
        collect.canary_ne_n_role[r] > 0
            ? collect.canary_ne_sum_role[r] / collect.canary_ne_n_role[r]
            : 0.0;
    role_n_total += collect.canary_ne_n_role[r];
    weighted_sum += collect.canary_ne_sum_role[r];
  }

  if (role_n_total != collect.canary_ne_n) {
    SpielFatalError(absl::StrFormat(
        "PWO-5 section 14.1c partition identity FAILED (count half): "
        "sum of role denominators = %lld but norm_entropy_n = %lld. A "
        "qualifying decision was dropped or double-counted.",
        static_cast<long long>(role_n_total),
        static_cast<long long>(collect.canary_ne_n)));
  }
  if (role_n_total > 0) {
    const double reconstructed = weighted_sum / role_n_total;
    if (std::abs(reconstructed - stats->norm_entropy) > 1e-9) {
      SpielFatalError(absl::StrFormat(
          "PWO-5 section 14.1c partition identity FAILED (mean half): "
          "row-count-weighted role mean %.17g vs norm_entropy %.17g, "
          "difference %.3g exceeds 1e-9.",
          reconstructed, stats->norm_entropy,
          std::abs(reconstructed - stats->norm_entropy)));
    }
  }
  // The seventh role must be identically empty at this site (section 14.1c's
  // proof). Checked, because the proof depends on the measurement site and a
  // future move of that site would silently invalidate it.
  if (collect.canary_ne_n_role[static_cast<int>(
          DuneDecisionRole::kForcedOrBookkeeping)] != 0) {
    SpielFatalError(
        "PWO-5 section 14.1c: kForcedOrBookkeeping carried a canary decision. "
        "That role is provably empty at the registered rollout measurement "
        "site (chance nodes continue before classification, and the population "
        "requires n_legal >= 2), so a nonzero count means the canaries are no "
        "longer being measured where section 14.1b froze them.");
  }
}

void AttachPrecapAbszStats(open_spiel::PpoUpdateStats* stats,
                           const CollectResult& collect) {
  stats->precap_absz_all =
      open_spiel::SummarizePrecapAbsz(collect.precap_absz_all);
  stats->precap_absz_primary =
      open_spiel::SummarizePrecapAbsz(collect.precap_absz_primary);
  stats->precap_absz_continuation =
      open_spiel::SummarizePrecapAbsz(collect.precap_absz_continuation);
  stats->precap_absz_purchase =
      open_spiel::SummarizePrecapAbsz(collect.precap_absz_purchase);
}



CollectResult CollectRollout(const Game* game,
                             std::shared_ptr<IGameEvaluator> evaluator,
                             int64_t obs_size,
                             std::atomic<uint64_t>* total_env_steps,
                             int num_threads,
                             std::atomic<uint64_t>* next_episode_id,
                             int rollout_games,
                             float reward_lambda,
                             const std::vector<std::shared_ptr<IGameEvaluator>>& opponent_evaluators,
                             int global_update) {
  CollectResult result;
  open_spiel::dune_targeted_buy::g_targeted_buy_tracker.Reset(
      absl::GetFlag(FLAGS_targeted_buy_exploration),
      absl::GetFlag(FLAGS_targeted_buy_quota_per_card),
      absl::GetFlag(FLAGS_targeted_buy_quota_per_card),
      absl::GetFlag(FLAGS_targeted_buy_max_round));
  PpoRolloutBuffer rollout_buffer;
  std::atomic<bool> stop_collection{false};
  std::vector<WorkerStats> worker_stats(num_threads);
  std::vector<std::thread> workers;
  workers.reserve(num_threads);

  const bool search_in_training = absl::GetFlag(FLAGS_search_in_training);
  std::vector<std::vector<open_spiel::dune_imperium::SearchSnapshot>> worker_snapshots(
      search_in_training ? num_threads : 0);
  const bool r755_hybrid = absl::GetFlag(FLAGS_r755_hybrid_mode);
  std::vector<std::vector<open_spiel::SearchTrainingExample>> worker_aux_examples(
      r755_hybrid ? num_threads : 0);

  uint64_t actual_start_ep = next_episode_id->load();
  std::atomic<uint64_t> local_episode_id{actual_start_ep};

  auto batched_eval = std::dynamic_pointer_cast<open_spiel::BatchedEvaluator>(evaluator);
  if (batched_eval && std::getenv("DUNE_ENABLE_ASYNC_STAGE_TIMERS") != nullptr) {
    batched_eval->EnableAsyncStageTimers(true);
  }
  open_spiel::BatcherTelemetry telem_before;
  if (batched_eval) {
    telem_before = batched_eval->GetBatcherTelemetry();
  }

  std::atomic<int> completed_workers{0};
  std::atomic<int64_t> first_worker_finish_ns{0};

  auto start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < num_threads; ++i) {
    workers.emplace_back([&, i]() {
      RolloutWorker(i, game, evaluator, obs_size,
                    &rollout_buffer, &stop_collection, total_env_steps,
                    &worker_stats, &local_episode_id, actual_start_ep,
                    rollout_games, reward_lambda,
                    std::cref(opponent_evaluators),
                    search_in_training ? &worker_snapshots[i] : nullptr,
                    global_update,
                    r755_hybrid ? &worker_aux_examples[i] : nullptr);
      if (completed_workers.fetch_add(1, std::memory_order_relaxed) == 0) {
        first_worker_finish_ns.store(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::high_resolution_clock::now().time_since_epoch()).count(),
            std::memory_order_relaxed);
      }
    });
  }
  for (auto& worker : workers) worker.join();

  if (r755_hybrid) {
    for (int i = 0; i < num_threads; ++i) {
      for (auto& ex : worker_aux_examples[i]) {
        result.aux_examples.push_back(std::move(ex));
      }
    }
  }

  if (search_in_training) {
    std::vector<open_spiel::dune_imperium::SearchSnapshot> all_candidates;
    for (int i = 0; i < num_threads; ++i) {
      for (auto& s : worker_snapshots[i]) {
        all_candidates.push_back(std::move(s));
      }
    }
    int budget = absl::GetFlag(FLAGS_search_budget_per_update);
    if (static_cast<int>(all_candidates.size()) <= budget) {
      result.search_snapshots = std::move(all_candidates);
    } else {
      uint64_t master_rseed = dune_seed::DeriveSeed(
          absl::GetFlag(FLAGS_seed), dune_seed::kDomainTrain, global_update, /*stream=*/0x53454152ULL);
      std::mt19937_64 master_rng(master_rseed);
      std::shuffle(all_candidates.begin(), all_candidates.end(), master_rng);
      all_candidates.resize(budget);
      result.search_snapshots = std::move(all_candidates);
    }
  }

  if (rollout_games > 0) {
    next_episode_id->store(actual_start_ep + rollout_games);
  } else {
    next_episode_id->store(local_episode_id.load());
  }
  auto end = std::chrono::high_resolution_clock::now();

  bool episode_ids_unique = true;
  result.rollout = rollout_buffer.TakeAll(&episode_ids_unique);
  result.episode_ids_unique = episode_ids_unique;

  for (const auto& stats : worker_stats) {
    result.games += stats.games;
    result.moves += stats.moves;
    result.search_eligible_decisions += stats.search_eligible_decisions;

    result.specimen_conversions += stats.specimen_conversions;
    result.specimen_penalty_deducted += stats.specimen_penalty_deducted;
    result.atomics_penalized += stats.atomics_penalized;
    result.atomics_exempt += stats.atomics_exempt;
    result.atomics_penalty_deducted += stats.atomics_penalty_deducted;
    result.plot_penalized += stats.plot_penalized;
    result.plot_exempt += stats.plot_exempt;
    result.plot_penalty_deducted += stats.plot_penalty_deducted;
    result.forced_buys += stats.forced_buys;
    result.nontrivial_transitions += stats.nontrivial_transitions;
    result.high_stakes_pre_transitions += stats.high_stakes_pre_transitions;
    result.high_stakes_post_transitions += stats.high_stakes_post_transitions;

    for (int i = 0; i < 4; ++i) result.sm_acquisitions_by_seat[i] += stats.sm_acquisitions_by_seat[i];
    for (int i = 0; i < 14; ++i) result.sm_acquisitions_by_leader[i] += stats.sm_acquisitions_by_leader[i];
    for (int i = 0; i < 11; ++i) result.sm_acquisitions_by_round[i] += stats.sm_acquisitions_by_round[i];

    result.conflict_vp_generated += stats.conflict_vp_generated;
    result.conflict_vp_attributed += stats.conflict_vp_attributed;
    result.conflict_vp_unattributed += stats.conflict_vp_unattributed;

    result.raw_conflict_vp += stats.raw_conflict_vp;
    result.raw_noncombat_vp += stats.raw_noncombat_vp;
    result.raw_total_vp += stats.raw_total_vp;

    for (int c = 0; c < 6; ++c) {
      result.r755_searched_by_cat[c] += stats.r755_searched_by_cat[c];
      result.r755_executed_overrides_by_cat[c] += stats.r755_executed_overrides_by_cat[c];
    }
    result.r755_imperium_buys += stats.r755_imperium_buys;
    result.r755_tleilaxu_buys += stats.r755_tleilaxu_buys;
    result.r755_tech_buys += stats.r755_tech_buys;
    result.r755_reveal_turns += stats.r755_reveal_turns;
    result.r755_unspent_persuasion_sum += stats.r755_unspent_persuasion_sum;
    for (int b = 0; b < 9; ++b) {
      result.r755_cat3_i_sum[b] += stats.r755_cat3_i_sum[b];
      result.r755_cat3_ii_sum[b] += stats.r755_cat3_ii_sum[b];
      result.r755_cat3_iii_sum[b] += stats.r755_cat3_iii_sum[b];
      result.r755_cat3_bucket_count[b] += stats.r755_cat3_bucket_count[b];
    }

    result.parity_decisions += stats.parity_decisions;
    result.parity_full_width_ok += stats.parity_full_width_ok;
    result.parity_full_finite_ok += stats.parity_full_finite_ok;
    result.parity_legal_ids_unique_in_range +=
        stats.parity_legal_ids_unique_in_range;
    result.parity_chosen_once += stats.parity_chosen_once;
    result.parity_capture_errors.insert(
        result.parity_capture_errors.end(),
        stats.parity_capture_errors.begin(),
        stats.parity_capture_errors.end());

    // Phase A canary: concatenate in thread-index order. Downstream only sorts,
    // so order is irrelevant to the reported percentiles.
    result.precap_absz_all.insert(result.precap_absz_all.end(),
                                  stats.precap_absz_all.begin(),
                                  stats.precap_absz_all.end());
    result.precap_absz_primary.insert(result.precap_absz_primary.end(),
                                      stats.precap_absz_primary.begin(),
                                      stats.precap_absz_primary.end());
    result.precap_absz_continuation.insert(result.precap_absz_continuation.end(),
                                           stats.precap_absz_continuation.begin(),
                                           stats.precap_absz_continuation.end());
    result.precap_absz_purchase.insert(result.precap_absz_purchase.end(),
                                       stats.precap_absz_purchase.begin(),
                                       stats.precap_absz_purchase.end());

    // PWO-5 section 14.1b canaries.
    result.canary_ne_sum += stats.canary_ne_sum;
    result.canary_ne_n += stats.canary_ne_n;
    for (int r = 0; r < 7; ++r) {
      result.canary_ne_sum_role[r] += stats.canary_ne_sum_role[r];
      result.canary_ne_n_role[r] += stats.canary_ne_n_role[r];
    }
    result.canary_maxprob_sum += stats.canary_maxprob_sum;
    result.canary_sat_ge_cap += stats.canary_sat_ge_cap;
    result.canary_sat_logits += stats.canary_sat_logits;

    result.card11_metrics += stats.card11_metrics;
    result.card13_metrics += stats.card13_metrics;
    result.targeted_buy_telemetry += stats.targeted_buy_telemetry;

    for (const auto& pr : stats.r765_override_probes) {
      result.r765_override_probes.push_back(pr);
    }
    for (const auto& ap : stats.r765_agreed_probes) {
      if (result.r765_agreed_probes.size() < 64) {
        result.r765_agreed_probes.push_back(ap);
      }
    }
  }

  result.targeted_buy_achieved_card11 = open_spiel::dune_targeted_buy::g_targeted_buy_tracker.achieved_games_card11.load(std::memory_order_relaxed);
  result.targeted_buy_achieved_card13 = open_spiel::dune_targeted_buy::g_targeted_buy_tracker.achieved_games_card13.load(std::memory_order_relaxed);
  result.targeted_buy_designated_opps_card11 = open_spiel::dune_targeted_buy::g_targeted_buy_tracker.designated_opps_card11.load(std::memory_order_relaxed);
  result.targeted_buy_designated_opps_card13 = open_spiel::dune_targeted_buy::g_targeted_buy_tracker.designated_opps_card13.load(std::memory_order_relaxed);
  result.targeted_buy_total_opps_card11 = open_spiel::dune_targeted_buy::g_targeted_buy_tracker.total_opps_card11.load(std::memory_order_relaxed);
  result.targeted_buy_total_opps_card13 = open_spiel::dune_targeted_buy::g_targeted_buy_tracker.total_opps_card13.load(std::memory_order_relaxed);

  // Verify shaped reward conservation invariant
  if (std::abs(result.conflict_vp_generated - (result.conflict_vp_attributed + result.conflict_vp_unattributed)) > 1e-4) {
    open_spiel::SpielFatalError(absl::StrFormat("Shaped reward conservation violated: generated = %f, attributed = %f, unattributed = %f",
                                                result.conflict_vp_generated, result.conflict_vp_attributed, result.conflict_vp_unattributed));
  }

  // Verify signed raw VP conservation invariant
  if (std::abs(result.raw_conflict_vp + result.raw_noncombat_vp - result.raw_total_vp) > 1e-4) {
    open_spiel::SpielFatalError(absl::StrFormat("Signed raw VP conservation violated: raw_conflict = %f, raw_noncombat = %f, raw_total = %f",
                                                result.raw_conflict_vp, result.raw_noncombat_vp, result.raw_total_vp));
  }
  result.elapsed_seconds =
      std::chrono::duration<double>(end - start).count();

  double tail_drain_s = 0.0;
  int64_t f_ns = first_worker_finish_ns.load(std::memory_order_relaxed);
  if (f_ns > 0) {
    auto f_time = std::chrono::high_resolution_clock::time_point(
        std::chrono::nanoseconds(f_ns));
    tail_drain_s = std::max(0.0, std::chrono::duration<double>(end - f_time).count());
  }

  if (batched_eval && std::getenv("DUNE_LOG_BATCHER_TELEM") != nullptr) {
    open_spiel::BatcherTelemetry telem_after = batched_eval->GetBatcherTelemetry();
    uint64_t d_batches = telem_after.physical_batches - telem_before.physical_batches;
    uint64_t d_requests = telem_after.submitted_rows - telem_before.submitted_rows;
    uint64_t d_dev_timed = telem_after.device_timed_batches - telem_before.device_timed_batches;
    double d_h2d_ms = telem_after.h2d_ms - telem_before.h2d_ms;
    double d_trunk_ms = telem_after.trunk_ms - telem_before.trunk_ms;
    double d_scorer_ms = telem_after.scorer_ms - telem_before.scorer_ms;
    double d_post_cap_ms = telem_after.post_cap_softmax_ms - telem_before.post_cap_softmax_ms;
    double d_d2h_ms = telem_after.d2h_ms - telem_before.d2h_ms;
    double d_gap_ms = telem_after.inter_slot_idle_gap_ms - telem_before.inter_slot_idle_gap_ms;
    uint64_t d_gaps = telem_after.inter_slot_gap_count - telem_before.inter_slot_gap_count;
    double d_cpu_wait_ms = telem_after.cpu_wait_ms - telem_before.cpu_wait_ms;
    double d_cpu_pack_ms = telem_after.cpu_pack_ms - telem_before.cpu_pack_ms;
    double d_cpu_dist_ms = telem_after.cpu_distribute_ms - telem_before.cpu_distribute_ms;
    uint64_t d_wants_batches = telem_after.wants_logits_batches - telem_before.wants_logits_batches;
    uint64_t d_wants_rows = telem_after.wants_logits_rows - telem_before.wants_logits_rows;
    uint64_t d_d2h_bytes = telem_after.d2h_bytes - telem_before.d2h_bytes;
    uint64_t d_greedy_calls = telem_after.greedy_action_calls - telem_before.greedy_action_calls;
    uint64_t d_greedy_ties = telem_after.greedy_action_ties - telem_before.greedy_action_ties;
    uint64_t d_full = telem_after.full_batches - telem_before.full_batches;
    uint64_t d_ge128 = telem_after.ge128_batches - telem_before.ge128_batches;
    uint64_t d_ge64 = telem_after.ge64_batches - telem_before.ge64_batches;
    uint64_t d_under64 = telem_after.under64_batches - telem_before.under64_batches;

    double h2d_ms_batch = d_dev_timed > 0 ? (d_h2d_ms / d_dev_timed) : 0.0;
    double trunk_ms_batch = d_dev_timed > 0 ? (d_trunk_ms / d_dev_timed) : 0.0;
    double scorer_ms_batch = d_dev_timed > 0 ? (d_scorer_ms / d_dev_timed) : 0.0;
    double post_cap_ms_batch = d_dev_timed > 0 ? (d_post_cap_ms / d_dev_timed) : 0.0;
    double d2h_ms_batch = d_dev_timed > 0 ? (d_d2h_ms / d_dev_timed) : 0.0;
    double gap_ms_batch = d_gaps > 0 ? (d_gap_ms / d_gaps) : 0.0;
    double cpu_wait_ms_batch = d_batches > 0 ? (d_cpu_wait_ms / d_batches) : 0.0;
    double cpu_pack_ms_batch = d_batches > 0 ? (d_cpu_pack_ms / d_batches) : 0.0;
    double cpu_dist_ms_batch = d_batches > 0 ? (d_cpu_dist_ms / d_batches) : 0.0;
    double mean_batch_size = d_batches > 0 ? (static_cast<double>(d_requests) / d_batches) : 0.0;
    double d2h_mb = static_cast<double>(d_d2h_bytes) / (1024.0 * 1024.0);

    std::cout << absl::StrFormat(
        "[BATCHER-TELEM] u=%d collect_s=%.2f tail_drain_s=%.2f batches=%llu (full=%llu ge128=%llu ge64=%llu u64=%llu) "
        "rows=%llu mean_b=%.1f | GPU ms/b: h2d=%.3f trunk=%.3f scorer=%.3f post_cap=%.3f d2h=%.3f gap=%.3f | "
        "CPU ms/b: wait=%.3f pack=%.3f dist=%.3f | D2H: %.1f MB (wants_logits: %llu b, %llu rows) | "
        "greedy: %llu calls, %llu ties (%.2f%%)",
        global_update,
        result.elapsed_seconds,
        tail_drain_s,
        static_cast<unsigned long long>(d_batches),
        static_cast<unsigned long long>(d_full),
        static_cast<unsigned long long>(d_ge128),
        static_cast<unsigned long long>(d_ge64),
        static_cast<unsigned long long>(d_under64),
        static_cast<unsigned long long>(d_requests),
        mean_batch_size,
        h2d_ms_batch, trunk_ms_batch, scorer_ms_batch, post_cap_ms_batch, d2h_ms_batch, gap_ms_batch,
        cpu_wait_ms_batch, cpu_pack_ms_batch, cpu_dist_ms_batch,
        d2h_mb,
        static_cast<unsigned long long>(d_wants_batches),
        static_cast<unsigned long long>(d_wants_rows),
        static_cast<unsigned long long>(d_greedy_calls),
        static_cast<unsigned long long>(d_greedy_ties),
        d_greedy_calls > 0 ? (100.0 * d_greedy_ties / d_greedy_calls) : 0.0)
        << std::endl;
  }
  return result;
}


#endif  // OPEN_SPIEL_BUILD_WITH_LIBTORCH

}  // namespace open_spiel::ppo_train_internal
