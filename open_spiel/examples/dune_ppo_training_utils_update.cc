#include "dune_ppo_training_utils_update_internal.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <string>
#include <unordered_set>
#include <vector>

#ifdef OPEN_SPIEL_BUILD_WITH_LIBTORCH
#include <c10/cuda/CUDACachingAllocator.h>
#include "dune_seed_utils.h"
#include "open_spiel/abseil-cpp/absl/flags/flag.h"

namespace open_spiel::ppo_utils_internal {

bool RunTrainPpoUpdatePrepass(TrainPpoUpdateContext& ctx) {
  auto& batch = *ctx.batch;
  auto& model = ctx.model;
  auto& stats = ctx.stats;
  const int64_t n = static_cast<int64_t>(batch.size());
  ctx.n = n;
  const int64_t obs_size = ctx.obs_size;
  const int64_t action_dim = ctx.action_dim;
  const torch::Device device = ctx.device;

  auto cpu_float = torch::TensorOptions().dtype(torch::kFloat32);
  auto cpu_bool = torch::TensorOptions().dtype(torch::kBool);
  auto cpu_long = torch::TensorOptions().dtype(torch::kInt64);

  ctx.returns_cpu = torch::empty({n}, cpu_float);
  ctx.old_values_cpu = torch::empty({n}, cpu_float);
  ctx.returns_ptr = ctx.returns_cpu.data_ptr<float>();
  ctx.old_values_ptr = ctx.old_values_cpu.data_ptr<float>();
  for (int64_t i = 0; i < n; ++i) {
    ctx.returns_ptr[i] = batch[i].return_value;
    ctx.old_values_ptr[i] = batch[i].value;
  }

  ctx.minibatch_size =
      std::min<int64_t>(::absl::GetFlag(::FLAGS_ppo_minibatch_size), n);
  ctx.max_mb = ctx.minibatch_size;
  const int64_t minibatch_size = ctx.minibatch_size;
  const int64_t max_mb = ctx.max_mb;

  // Pre-allocate reusable 2-slot double-buffered CPU staging tensors sized to the active minibatch capacity.
  ctx.use_pinned = device.is_cuda();
  auto staging_float_opts = cpu_float.pinned_memory(ctx.use_pinned);
  auto staging_bool_opts = cpu_bool.pinned_memory(ctx.use_pinned);
  auto staging_long_opts = cpu_long.pinned_memory(ctx.use_pinned);

  for (int slot = 0; slot < 2; ++slot) {
    ctx.staging_states_cpu[slot] =
        torch::empty({max_mb, obs_size}, staging_float_opts);
    ctx.staging_masks_cpu[slot] =
        torch::empty({max_mb, action_dim}, staging_bool_opts);
    ctx.staging_actions_cpu[slot] =
        torch::empty({max_mb}, staging_long_opts);
    ctx.staging_old_log_probs_cpu[slot] =
        torch::empty({max_mb}, staging_float_opts);
    ctx.staging_advantages_cpu[slot] =
        torch::empty({max_mb}, staging_float_opts);
    ctx.staging_returns_cpu[slot] =
        torch::empty({max_mb}, staging_float_opts);
    ctx.staging_old_values_cpu[slot] =
        torch::empty({max_mb}, staging_float_opts);
    ctx.staging_nontrivial_cpu[slot] =
        torch::empty({max_mb}, staging_bool_opts);
    ctx.staging_off_policy_cpu[slot] =
        torch::empty({max_mb}, staging_bool_opts);
  }

  ctx.staging_slot = 0;
  ctx.staging_slot_recorded[0] = false;
  ctx.staging_slot_recorded[1] = false;
  if (device.is_cuda() && ctx.use_pinned) {
    ctx.staging_upload_done[0] = std::make_unique<c10::Event>(
        c10::DeviceType::CUDA, c10::EventFlag::BACKEND_DEFAULT);
    ctx.staging_upload_done[1] = std::make_unique<c10::Event>(
        c10::DeviceType::CUDA, c10::EventFlag::BACKEND_DEFAULT);
  }
  ctx.train_batch_ptr = &batch;

  ctx.update_epochs = std::max(1, ::absl::GetFlag(::FLAGS_ppo_update_epochs));
  ctx.clip_epsilon = static_cast<float>(::absl::GetFlag(::FLAGS_ppo_clip_epsilon));
  ctx.normalize_advantages = ::absl::GetFlag(::FLAGS_normalize_advantages);
  ctx.clip_value_loss = ::absl::GetFlag(::FLAGS_ppo_clip_value_loss);
  ctx.entropy_coef = static_cast<float>(::absl::GetFlag(::FLAGS_entropy_coef));
  ctx.value_coef = static_cast<float>(::absl::GetFlag(::FLAGS_value_coef));
  ctx.logit_cap = static_cast<float>(::absl::GetFlag(::FLAGS_logit_cap));
  ctx.target_kl = ::absl::GetFlag(::FLAGS_target_kl);

  // --- Phase 5 Diagnostics Pre-pass ---

  // 1. Episode-ID Uniqueness
  std::unordered_set<uint64_t> seen_episodes;
  bool episode_ids_unique = true;
  if (!batch.empty()) {
    uint64_t last_ep_id = batch[0].episode_id;
    seen_episodes.insert(last_ep_id);
    for (const auto& trans : batch) {
      if (trans.episode_id != last_ep_id) {
        last_ep_id = trans.episode_id;
        if (seen_episodes.find(last_ep_id) != seen_episodes.end()) {
          episode_ids_unique = false;
        }
        seen_episodes.insert(last_ep_id);
      }
    }
  }
  stats.episode_ids_unique = episode_ids_unique;

  // 2. Nontrivial Mask
  int64_t nontrivial_count = 0;
  for (int64_t i = 0; i < n; ++i) {
    if (batch[i].legal_actions.size() > 1) {
      nontrivial_count++;
    }
  }
  ctx.nontrivial_count = nontrivial_count;
  stats.total_transitions = n;
  stats.nontrivial_transitions = nontrivial_count;
  stats.forced_transitions = n - nontrivial_count;
  stats.single_action_transitions = stats.forced_transitions;
  stats.single_action_fraction =
      n > 0 ? (static_cast<double>(stats.single_action_transitions) / n) : 0.0;

  // 3. Policy KL Before & Saturation Checks
  double kl_before_sum = 0.0;
  int64_t kl_before_nontrivial_count = 0;
  bool kl_measured_this_update = true;

  if (ctx.compute_diagnostics) {
    ctx.phase_timer->Begin(kPhaseDiagPrepass);
    if (!ctx.prepass_cadenced) {
      double value_near_one_count = 0;

      bool was_training = model->is_training();
      model->eval();
      {
        torch::NoGradGuard no_grad;
        for (int64_t start = 0; start < n; start += minibatch_size) {
          int64_t end = std::min(start + minibatch_size, n);
          int64_t mb_len = end - start;
          torch::Tensor mb_states, mb_masks, mb_actions, mb_old_log_probs, mb_nontrivial;
          std::vector<const dune_semantic::CandidateActionData*> mb_cands;
          StageAndUploadMinibatch(
              ctx, /*mb_indices=*/nullptr, start, mb_len, /*full_ppo_fields=*/false,
              mb_states, mb_masks, mb_actions, mb_old_log_probs, mb_nontrivial,
              (model->with_semantic_scorer_ && model->semantic_scorer_) ? &mb_cands : nullptr);

          auto outputs = model->forward(mb_states);
          if (model->with_semantic_scorer_ && model->semantic_scorer_) {
            dune_semantic::ApplySemanticScorerBatch(
                model->semantic_scorer_, outputs.trunk, mb_cands, outputs.logits, device);
          }
          if (!::absl::GetFlag(::FLAGS_train_value_only)) {
            torch::Tensor logits = CenterAndCapLogitsTensor(outputs.logits, mb_masks, ctx.logit_cap);
            torch::Tensor masked_logits = logits.masked_fill(mb_masks.logical_not(), -1e9f);
            torch::Tensor log_probs = torch::log_softmax(masked_logits, -1);
            torch::Tensor selected_log_probs = log_probs.gather(1, mb_actions.unsqueeze(1)).squeeze(1);

            torch::Tensor log_ratio = selected_log_probs - mb_old_log_probs;
            torch::Tensor ratio = torch::exp(log_ratio);
            torch::Tensor approx_kl = (ratio - 1.0f) - log_ratio;

            torch::Tensor masked_kl = approx_kl.masked_select(mb_nontrivial);
            if (masked_kl.numel() > 0) {
              kl_before_sum += masked_kl.sum().item<double>();
              kl_before_nontrivial_count += masked_kl.numel();
            }
          }

          torch::Tensor mb_values = outputs.values.squeeze(1);
          torch::Tensor mb_near_one = mb_values.abs() >= 0.99f;
          value_near_one_count += mb_near_one.sum().item<double>();
        }
      }
      if (was_training) {
        model->train();
      }
      stats.policy_kl_before =
          (kl_before_nontrivial_count > 0) ? (kl_before_sum / kl_before_nontrivial_count) : 0.0;
      stats.fraction_critic_near_1 = value_near_one_count / n;
    } else {
      int64_t stored_near_one_count = 0;
      for (int64_t i = 0; i < n; ++i) {
        if (std::abs(ctx.old_values_ptr[i]) >= 0.99f) ++stored_near_one_count;
      }
      stats.fraction_critic_near_1 = static_cast<double>(stored_near_one_count) / n;

      kl_measured_this_update =
          ctx.expensive_checks_now && !::absl::GetFlag(::FLAGS_train_value_only);
      if (kl_measured_this_update) {
        bool was_training = model->is_training();
        model->eval();
        {
          torch::NoGradGuard no_grad;
          AutocastGuard autocast_guard(
              device.type(), device.is_cuda() && ::absl::GetFlag(::FLAGS_rollout_amp));
          for (int64_t start = 0; start < n; start += minibatch_size) {
            int64_t end = std::min(start + minibatch_size, n);
            int64_t mb_len = end - start;
            torch::Tensor mb_states, mb_masks, mb_actions, mb_old_log_probs, mb_nontrivial;
            std::vector<const dune_semantic::CandidateActionData*> mb_cands;
            StageAndUploadMinibatch(
                ctx, /*mb_indices=*/nullptr, start, mb_len, /*full_ppo_fields=*/false,
                mb_states, mb_masks, mb_actions, mb_old_log_probs, mb_nontrivial,
                (model->with_semantic_scorer_ && model->semantic_scorer_) ? &mb_cands : nullptr);

            auto outputs = model->forward(mb_states);
            if (model->with_semantic_scorer_ && model->semantic_scorer_) {
              dune_semantic::ApplySemanticScorerBatch(
                  model->semantic_scorer_, outputs.trunk, mb_cands, outputs.logits, device);
            }
            torch::Tensor logits = CenterAndCapLogitsTensor(outputs.logits, mb_masks, ctx.logit_cap);
            torch::Tensor masked_logits = logits.masked_fill(mb_masks.logical_not(), -1e9f);
            torch::Tensor log_probs = torch::log_softmax(masked_logits, -1);
            torch::Tensor selected_log_probs = log_probs.gather(1, mb_actions.unsqueeze(1)).squeeze(1);

            torch::Tensor log_ratio = selected_log_probs - mb_old_log_probs;
            torch::Tensor ratio = torch::exp(log_ratio);
            torch::Tensor approx_kl = (ratio - 1.0f) - log_ratio;

            torch::Tensor masked_kl = approx_kl.masked_select(mb_nontrivial);
            if (masked_kl.numel() > 0) {
              kl_before_sum += masked_kl.sum().item<double>();
              kl_before_nontrivial_count += masked_kl.numel();
            }
          }
        }
        if (was_training) {
          model->train();
        }
      }
      stats.policy_kl_before =
          (kl_before_nontrivial_count > 0) ? (kl_before_sum / kl_before_nontrivial_count) : 0.0;
    }
    ctx.phase_timer->End(kPhaseDiagPrepass);
    stats.measured_transitions =
        kl_measured_this_update ? kl_before_nontrivial_count : int64_t{-1};

    // 4. Return Calibration Diagnostics
    std::vector<float> returns_vec(ctx.returns_ptr, ctx.returns_ptr + n);
    std::sort(returns_vec.begin(), returns_vec.end());
    stats.return_min = returns_vec.front();
    stats.return_max = returns_vec.back();
    stats.return_p50 = returns_vec[n * 50 / 100];
    stats.return_p95 = returns_vec[n * 95 / 100];
    stats.return_p99 = returns_vec[n * 99 / 100];

    std::vector<float> abs_returns_vec(n);
    int64_t count_outside = 0;
    for (int64_t i = 0; i < n; ++i) {
      abs_returns_vec[i] = std::abs(ctx.returns_ptr[i]);
      if (ctx.returns_ptr[i] < -1.0f || ctx.returns_ptr[i] > 1.0f) {
        count_outside++;
      }
    }
    std::sort(abs_returns_vec.begin(), abs_returns_vec.end());
    stats.abs_return_p99 = abs_returns_vec[n * 99 / 100];
    stats.fraction_targets_outside_1 = static_cast<double>(count_outside) / n;
  } else {
    stats.measured_transitions = int64_t{-1};
  }

  if (absl::GetFlag(FLAGS_diagnostics_only)) {
    // Populates phases 1-2 for a caller that wants them. NOTE: the only
    // --diagnostics_only call site (dune_ppo_train.cc) writes diagnostics and
    // then exits WITHOUT calling WritePhaseTiming, so no sidecar record is
    // emitted on this path. Stated plainly rather than left to look like
    // coverage that does not exist.
    ctx.phase_timer->Finalize(&stats.phase_timings);
    return true;
  }

  // --- Phase 18B online-search auxiliary examples (combined optimization) ---
  const auto& search_examples = *ctx.search_examples;
  ctx.aux_active = !search_examples.empty() && ctx.search_loss_coef > 0.0 &&
                   !::absl::GetFlag(::FLAGS_train_value_only) &&
                   !::absl::GetFlag(::FLAGS_search_aux_decoupled);
  stats.aux_search_loss_coef = ctx.search_loss_coef;
  {
    const bool is_v5 = model->with_semantic_scorer_ && model->semantic_scorer_ &&
                       model->semantic_scorer_->enable_v5_;
    const std::string purchase_target = is_v5 ? "semantic_scorer.v5_" : "out_layer";
    auto named = model->named_parameters();
    ctx.need_snapshot_param.assign(named.size(), 0);
    for (size_t pi = 0; pi < named.size(); ++pi) {
      const std::string& nm = named[pi].key();
      if (nm.find(purchase_target) != std::string::npos) {
        ctx.purchase_param_indices.push_back(pi);
        ctx.need_snapshot_param[pi] = 1;
      }
      if (nm.find("semantic_scorer.v5_") != std::string::npos ||
          nm.find("out_layer") != std::string::npos ||
          nm.find("policy_head") != std::string::npos) {
        ctx.action_head_param_indices.push_back(pi);
        ctx.need_snapshot_param[pi] = 1;
      }
    }
  }
  ctx.aux_n =
      ctx.aux_active ? static_cast<int64_t>(search_examples.size()) : 0;
  const int64_t aux_n = ctx.aux_n;
  ctx.all_aux_row_weights.assign(ctx.aux_active ? aux_n : 0, ctx.search_loss_coef);
  ctx.all_aux_ages.assign(ctx.aux_active ? aux_n : 0, 0);
  ctx.aux_visited_mask.assign(ctx.aux_active ? aux_n : 0, 0);
  if (ctx.aux_active) {
    stats.aux_examples_used = static_cast<int>(aux_n);
    for (int64_t i = 0; i < aux_n; ++i) {
      const SearchTrainingExample& ex = search_examples[i];
      ctx.all_aux_row_weights[i] =
          (ex.row_loss_weight >= 0.0) ? ex.row_loss_weight : ctx.search_loss_coef;
      ctx.all_aux_ages[i] =
          std::clamp(ex.replay_age, 0, PpoUpdateStats::kMaxR765ReplayAges - 1);
    }
    auto af = torch::TensorOptions().dtype(torch::kFloat32);
    auto ab = torch::TensorOptions().dtype(torch::kBool);
    torch::Tensor s = torch::zeros({aux_n, obs_size}, af);
    torch::Tensor m = torch::zeros({aux_n, action_dim}, ab);
    torch::Tensor t = torch::zeros({aux_n, action_dim}, af);
    torch::Tensor v = torch::zeros({aux_n}, af);
    float* sp = s.data_ptr<float>();
    bool* mp = m.data_ptr<bool>();
    float* tp = t.data_ptr<float>();
    float* vp = v.data_ptr<float>();
    for (int64_t i = 0; i < aux_n; ++i) {
      const SearchTrainingExample& ex = search_examples[i];
      int64_t copy = std::min<int64_t>(
          obs_size, static_cast<int64_t>(ex.observation.size()));
      if (copy > 0) {
        std::memcpy(sp + i * obs_size, ex.observation.data(),
                    copy * sizeof(float));
      }
      // Legal mask + normalized-visit CE targets, aligned to legal_actions.
      for (size_t j = 0; j < ex.legal_actions.size(); ++j) {
        Action a = ex.legal_actions[j];
        if (a >= 0 && a < action_dim) {
          mp[i * action_dim + a] = true;
          tp[i * action_dim + a] = static_cast<float>(
              (j < ex.normalized_visits.size()) ? ex.normalized_visits[j] : 0.0);
        }
      }
      vp[i] = static_cast<float>(ex.value_target);
    }
    ctx.all_aux_cands.assign(aux_n, nullptr);
    ctx.all_aux_mpo_pairs.resize(aux_n, {kInvalidAction, kInvalidAction});
    for (int64_t i = 0; i < aux_n; ++i) {
      ctx.all_aux_mpo_pairs[i] = {search_examples[i].best_action, search_examples[i].raw_action};
    }
    if (model->with_semantic_scorer_) {
      for (int64_t i = 0; i < aux_n; ++i) {
        if (search_examples[i].candidate_data.empty()) {
          open_spiel::SpielFatalError("TrainPpoUpdateSeparate: semantic scorer is enabled but candidate descriptors are missing or empty on search examples.");
        }
        ctx.all_aux_cands[i] = &search_examples[i].candidate_data;
      }
    }
    ctx.aux_states = s.to(device);
    ctx.aux_masks = m.to(device);
    ctx.aux_targets = t.to(device);
    ctx.aux_values = v.to(device);
    int64_t num_mb = (n + minibatch_size - 1) / minibatch_size;
    auto slices = ComputeAuxSlices(aux_n, num_mb);
    ctx.aux_slice_start.assign(num_mb, 0);
    ctx.aux_slice_len.assign(num_mb, 0);
    for (int64_t k = 0; k < num_mb && k < static_cast<int64_t>(slices.size()); ++k) {
      ctx.aux_slice_start[k] = slices[k].first;
      ctx.aux_slice_len[k] = slices[k].second;
    }
  }

  // WO-PERF-1 Part B: per-module gradient-telemetry mode.
  const std::string grad_telemetry_mode =
      ::absl::GetFlag(::FLAGS_grad_telemetry_mode);
  ctx.grad_telemetry_accumulated = (grad_telemetry_mode == "accumulated");
  if (!ctx.grad_telemetry_accumulated && grad_telemetry_mode != "per_param") {
    SpielFatalError("Unknown --grad_telemetry_mode: '" + grad_telemetry_mode +
                    "' (expected 'per_param' or 'accumulated').");
  }
  if (ctx.compute_diagnostics && ctx.grad_telemetry_accumulated) {
    ctx.head_norm_accum = torch::zeros(
        {6}, torch::TensorOptions().dtype(torch::kFloat64).device(device));
  }

  // --- Reverse KL penalty (Arm B: PPO with reverse-KL to collection policy) ---
  ctx.reverse_kl_active =
      (ctx.collection_model != nullptr && ctx.reverse_kl_coef > 0.0 &&
       !::absl::GetFlag(::FLAGS_train_value_only));
  if (ctx.reverse_kl_active) {
    auto& collection_model = ctx.collection_model;
    ctx.collection_log_probs = torch::empty(
        {n, action_dim},
        torch::TensorOptions().device(device).dtype(torch::kFloat32));
    bool coll_was_training = collection_model->is_training();
    collection_model->eval();
    {
      torch::NoGradGuard no_grad;
      for (int64_t start = 0; start < n; start += minibatch_size) {
        int64_t end = std::min(start + minibatch_size, n);
        int64_t mb_len = end - start;
        torch::Tensor mb_s, mb_m, dummy_actions, dummy_olp, dummy_nt;
        std::vector<const dune_semantic::CandidateActionData*> coll_cands;
        StageAndUploadMinibatch(
            ctx, /*mb_indices=*/nullptr, start, mb_len, /*full_ppo_fields=*/false,
            mb_s, mb_m, dummy_actions, dummy_olp, dummy_nt,
            (collection_model->with_semantic_scorer_ && collection_model->semantic_scorer_) ? &coll_cands : nullptr);
        auto coll_out = collection_model->forward(mb_s);
        if (collection_model->with_semantic_scorer_ && collection_model->semantic_scorer_) {
          dune_semantic::ApplySemanticScorerBatch(
              collection_model->semantic_scorer_, coll_out.trunk, coll_cands, coll_out.logits, device);
        }
        torch::Tensor coll_logits =
            CenterAndCapLogitsTensor(coll_out.logits, mb_m, ctx.logit_cap);
        torch::Tensor coll_masked_logits =
            coll_logits.masked_fill(mb_m.logical_not(), -1e9f);
        torch::Tensor coll_lp = torch::log_softmax(coll_masked_logits, -1);
        ctx.collection_log_probs.narrow(0, start, end - start).copy_(coll_lp);
      }
    }
    if (coll_was_training) {
      collection_model->train();
    }
  }

  // --- EMAgnet forward KL regularization (Arm 2: Treatment) ---
  ctx.ema_magnet_active =
      (ctx.ema_model != nullptr && ctx.ema_magnet_kl_coef > 0.0 &&
       !::absl::GetFlag(::FLAGS_train_value_only));
  if (ctx.ema_model != nullptr) {
    ctx.ema_model->eval();
  }

  // Step 3 update-path telemetry: collect forced row indices and pre-update log probabilities.
  for (size_t i = 0; i < batch.size(); ++i) {
    if (batch[i].targeted_buy_forced) {
      ctx.forced_indices.push_back(i);
    }
  }
  if (!ctx.forced_indices.empty()) {
    ctx.log_pi_before = ComputeForcedLogProbs(ctx, ctx.forced_indices);
  }

  // --- 12-block H/J study: pre-update diagnostics and filtering ---
  ctx.do_filtering = ::absl::GetFlag(::FLAGS_advantage_filtering);
  ctx.filter_quantile = ::absl::GetFlag(::FLAGS_advantage_filtering_quantile);
  ctx.study_readouts = ::absl::GetFlag(::FLAGS_enable_study_readouts) || ctx.do_filtering;
  ctx.actor_only_filtering = ::absl::GetFlag(::FLAGS_actor_only_advantage_filtering);
  ctx.actor_adv_threshold = 0.0;
  ctx.phase_stratified_norm = ::absl::GetFlag(::FLAGS_phase_stratified_advantage_norm);

  if (ctx.study_readouts) {
    for (int64_t i = 0; i < n; ++i) {
      if (batch[i].legal_actions.size() <= 1) {
        ctx.sa_indices.push_back(i);
      }
    }
    if (ctx.sa_indices.size() > 1024) {
      uint64_t sa_seed = dune_seed::DeriveSeed(
          ctx.master, dune_seed::kDomainTrain, ctx.global_update, 0, 99991);
      std::mt19937_64 sa_rng(sa_seed);
      std::shuffle(ctx.sa_indices.begin(), ctx.sa_indices.end(), sa_rng);
      ctx.sa_indices.resize(1024);
      std::sort(ctx.sa_indices.begin(), ctx.sa_indices.end());
    }
    stats.single_action_sample_size = static_cast<int64_t>(ctx.sa_indices.size());
    stats.single_action_value_mse_before = ComputeValueMse(ctx, ctx.sa_indices);
  }

  if (ctx.do_filtering) {
    std::vector<int64_t> mc_indices;
    mc_indices.reserve(nontrivial_count);
    for (int64_t i = 0; i < n; ++i) {
      if (batch[i].legal_actions.size() > 1 && !batch[i].mask_policy_loss) {
        mc_indices.push_back(i);
      }
    }
    if (!mc_indices.empty()) {
      std::vector<std::pair<double, int64_t>> abs_std_adv_and_idx;
      abs_std_adv_and_idx.reserve(mc_indices.size());
      if (ctx.phase_stratified_norm) {
        StandardizeAdvantagesPhaseStratified(&batch, mc_indices, /*min_stratum_count=*/2);
        for (int64_t idx : mc_indices) {
          abs_std_adv_and_idx.push_back({std::abs(static_cast<double>(batch[idx].advantage)), idx});
        }
      } else {
        double adv_sum = 0.0;
        for (int64_t idx : mc_indices) {
          adv_sum += batch[idx].advantage;
        }
        double adv_mean = adv_sum / mc_indices.size();
        double adv_sq_sum = 0.0;
        for (int64_t idx : mc_indices) {
          double diff = batch[idx].advantage - adv_mean;
          adv_sq_sum += diff * diff;
        }
        double adv_std = std::sqrt(adv_sq_sum / mc_indices.size()) + 1e-8;

        for (int64_t idx : mc_indices) {
          double std_adv = (batch[idx].advantage - adv_mean) / adv_std;
          batch[idx].advantage = static_cast<float>(std_adv);
          abs_std_adv_and_idx.push_back({std::abs(std_adv), idx});
        }
      }

      int64_t k_retained = std::max<int64_t>(1, std::round(ctx.filter_quantile * mc_indices.size()));
      if (ctx.actor_only_filtering) {
        if (k_retained < static_cast<int64_t>(abs_std_adv_and_idx.size())) {
          std::nth_element(
              abs_std_adv_and_idx.begin(),
              abs_std_adv_and_idx.begin() + (k_retained - 1),
              abs_std_adv_and_idx.end(),
              [](const std::pair<double, int64_t>& a, const std::pair<double, int64_t>& b) {
                return a.first > b.first;
              });
          ctx.actor_adv_threshold = abs_std_adv_and_idx[k_retained - 1].first;
        } else {
          ctx.actor_adv_threshold = 0.0;
        }
        stats.retained_transitions = k_retained;
        ctx.normalize_advantages = false;
      } else {
        if (k_retained < static_cast<int64_t>(abs_std_adv_and_idx.size())) {
          std::nth_element(
              abs_std_adv_and_idx.begin(),
              abs_std_adv_and_idx.begin() + k_retained,
              abs_std_adv_and_idx.end(),
              [](const std::pair<double, int64_t>& a, const std::pair<double, int64_t>& b) {
                return a.first > b.first;
              });
          abs_std_adv_and_idx.resize(k_retained);
        }
        std::vector<int64_t> retained_indices;
        retained_indices.reserve(abs_std_adv_and_idx.size());
        for (const auto& item : abs_std_adv_and_idx) {
          retained_indices.push_back(item.second);
        }
        std::sort(retained_indices.begin(), retained_indices.end());

        ctx.filtered_batch.reserve(retained_indices.size());
        for (int64_t idx : retained_indices) {
          ctx.filtered_batch.push_back(batch[idx]);
        }
        ctx.train_batch_ptr = &ctx.filtered_batch;
        stats.retained_transitions = static_cast<int64_t>(ctx.filtered_batch.size());
        ctx.normalize_advantages = false;

        if (!retained_indices.empty() && static_cast<int64_t>(retained_indices.size()) < n) {
          if (ctx.reverse_kl_active) {
            auto retained_idx_t = torch::tensor(
                retained_indices,
                torch::TensorOptions().dtype(torch::kInt64).device(device));
            ctx.collection_log_probs = ctx.collection_log_probs.index_select(0, retained_idx_t);
          }
          const int64_t num_mb_train =
              (static_cast<int64_t>(ctx.filtered_batch.size()) + minibatch_size - 1) / minibatch_size;
          if (ctx.aux_active) {
            auto slices = ComputeAuxSlices(aux_n, num_mb_train);
            ctx.aux_slice_start.assign(num_mb_train, 0);
            ctx.aux_slice_len.assign(num_mb_train, 0);
            for (int64_t k = 0; k < num_mb_train && k < static_cast<int64_t>(slices.size()); ++k) {
              ctx.aux_slice_start[k] = slices[k].first;
              ctx.aux_slice_len[k] = slices[k].second;
            }
          }
        }
      }
    } else {
      stats.retained_transitions = 0;
    }
  } else {
    stats.retained_transitions = n;
    if (ctx.phase_stratified_norm && ctx.normalize_advantages) {
      std::vector<int64_t> mc_indices;
      mc_indices.reserve(nontrivial_count);
      for (int64_t i = 0; i < n; ++i) {
        if (batch[i].legal_actions.size() > 1 && !batch[i].mask_policy_loss) {
          mc_indices.push_back(i);
        }
      }
      if (!mc_indices.empty()) {
        StandardizeAdvantagesPhaseStratified(&batch, mc_indices, /*min_stratum_count=*/2);
        ctx.normalize_advantages = false;
      }
    }
  }

  ctx.n_train = static_cast<int64_t>(ctx.train_batch_ptr->size());
  ctx.use_ema_cache =
      (ctx.ema_magnet_active && ctx.update_epochs > 1 &&
       ::absl::GetFlag(::FLAGS_cache_ema_log_probs) &&
       !::absl::GetFlag(::FLAGS_use_legacy_unoptimized_reads));

  ctx.fused_step_caches.resize(ctx.optimizer->param_groups().size());
  ctx.reuse_foreach_norms =
      (device.is_cuda() &&
       !::absl::GetFlag(::FLAGS_use_legacy_unoptimized_reads) &&
       !::absl::GetFlag(::FLAGS_use_wave1_ppo_update) &&
       ::absl::GetFlag(::FLAGS_use_grad_norm_reuse));

  return false;
}

}  // namespace open_spiel::ppo_utils_internal

namespace open_spiel {

PpoUpdateStats TrainPpoUpdate(
    std::shared_ptr<SharedDunePolicyValueNetImpl> model,
    torch::optim::AdamW& optimizer, std::vector<PpoTransition>& batch,
    int64_t obs_size, int64_t action_dim, torch::Device device,
    uint64_t master, int global_update,
    std::shared_ptr<SharedDunePolicyValueNetImpl> anchor_model,
    const std::vector<SearchTrainingExample>& search_examples,
    double search_loss_coef, double abort_grad_norm_ratio,
    std::shared_ptr<SharedDunePolicyValueNetImpl> collection_model,
    double reverse_kl_coef,
    bool compute_diagnostics,
    std::shared_ptr<SharedDunePolicyValueNetImpl> ema_model,
    double ema_magnet_kl_coef) {
  if (model != nullptr) {
    SPIEL_CHECK_FALSE(model->bf16_inference_cache_ready());
  }
  PpoUpdateStats stats;
  if (batch.empty()) return stats;
  const bool r755_hybrid = ::absl::GetFlag(::FLAGS_r755_hybrid_mode);
  if (device.is_cuda() && r755_hybrid) {
    c10::cuda::CUDACachingAllocator::emptyCache();
  }

  // WO-PERF-TIMING. Validated on EVERY path, including the default, so a typo
  // is fatal at the first update rather than silently timing nothing for a
  // whole run. Construction of a disabled timer allocates nothing.
  const std::string phase_timing_mode =
      ::absl::GetFlag(::FLAGS_phase_timing_mode);
  const bool phase_timing_on = (phase_timing_mode == "phases");
  if (!phase_timing_on && phase_timing_mode != "off") {
    SpielFatalError("Unknown --phase_timing_mode: '" + phase_timing_mode +
                    "' (expected 'off' or 'phases').");
  }

  const std::string diag_prepass_mode =
      ::absl::GetFlag(::FLAGS_diag_prepass_mode);
  const bool prepass_cadenced = (diag_prepass_mode == "cadenced");
  if (!prepass_cadenced && diag_prepass_mode != "full") {
    SpielFatalError("Unknown --diag_prepass_mode: '" + diag_prepass_mode +
                    "' (expected 'full' or 'cadenced').");
  }
  const int prepass_interval = ::absl::GetFlag(::FLAGS_diag_prepass_interval);
  const bool first_update_for_model =
      prepass_cadenced &&
      ppo_utils_internal::ConsumeFirstPrepassForModel(
          static_cast<const void*>(model.get()));
  const bool expensive_checks_now =
      !prepass_cadenced || first_update_for_model ||
      (prepass_interval > 0 && global_update % prepass_interval == 0);

  if (compute_diagnostics) {
    if (expensive_checks_now) {
      const auto hash_start = std::chrono::steady_clock::now();
      stats.rollout_hash = ComputeRolloutHash(batch);
      if (phase_timing_on) {
        stats.phase_timings.rollout_hash_host_s =
            std::chrono::duration<double>(
                std::chrono::steady_clock::now() - hash_start)
                .count();
      }
    } else {
      stats.rollout_hash = "unmeasured";
      stats.phase_timings.rollout_hash_host_s = 0.0;
    }
  } else {
    stats.rollout_hash.clear();
    stats.phase_timings.rollout_hash_host_s = 0.0;
  }

  // NOTE: the --pipeline incompatibility is enforced at startup in
  // dune_ppo_train.cc, where both flags are defined. It cannot live here --
  // FLAGS_pipeline is defined in the trainer's own TU, and six other targets
  // link this file without it.
  PhaseTimer phase_timer(phase_timing_on, device);

  ppo_utils_internal::TrainPpoUpdateContext ctx;
  ctx.model = std::move(model);
  ctx.optimizer = &optimizer;
  ctx.batch = &batch;
  ctx.obs_size = obs_size;
  ctx.action_dim = action_dim;
  ctx.device = device;
  ctx.master = master;
  ctx.global_update = global_update;
  ctx.anchor_model = std::move(anchor_model);
  ctx.search_examples = &search_examples;
  ctx.search_loss_coef = search_loss_coef;
  ctx.abort_grad_norm_ratio = abort_grad_norm_ratio;
  ctx.collection_model = std::move(collection_model);
  ctx.reverse_kl_coef = reverse_kl_coef;
  ctx.compute_diagnostics = compute_diagnostics;
  ctx.ema_model = std::move(ema_model);
  ctx.ema_magnet_kl_coef = ema_magnet_kl_coef;
  ctx.phase_timer = &phase_timer;
  ctx.stats = std::move(stats);
  ctx.r755_hybrid = r755_hybrid;
  ctx.phase_timing_on = phase_timing_on;
  ctx.prepass_cadenced = prepass_cadenced;
  ctx.expensive_checks_now = expensive_checks_now;

  if (ppo_utils_internal::RunTrainPpoUpdatePrepass(ctx)) {
    return ctx.stats;
  }
  ppo_utils_internal::RunTrainPpoUpdateEpochLoop(ctx);
  ppo_utils_internal::RunTrainPpoUpdateFinishPass(ctx);
  return ctx.stats;
}

}  // namespace open_spiel

#endif  // OPEN_SPIEL_BUILD_WITH_LIBTORCH
