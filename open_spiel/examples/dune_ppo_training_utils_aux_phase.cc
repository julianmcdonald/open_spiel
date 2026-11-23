#include "dune_ppo_training_utils_internal.h"
#include "open_spiel/abseil-cpp/absl/flags/flag.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <numeric>
#include <random>
#include <vector>

#ifdef OPEN_SPIEL_BUILD_WITH_LIBTORCH

namespace open_spiel {

SearchAuxPhaseResult TrainSearchAuxiliaryPhase(
    std::shared_ptr<SharedDunePolicyValueNetImpl> model,
    torch::optim::AdamW& optimizer,
    const std::vector<SearchTrainingExample>& search_examples,
    int64_t obs_size, int64_t action_dim,
    torch::Device device,
    int max_steps,
    int batch_size,
    double max_cumulative_kl,
    double search_loss_coef,
    const std::vector<PpoTransition>& reference_rollout_sample,
    uint64_t aux_seed,
    int epochs,
    const std::vector<SearchTrainingExample>& agreement_examples,
    const std::vector<PpoTransition>& all_rollout_transitions,
    const SearchAuxAnchoringConfig& anchoring_config) {
  SearchAuxPhaseResult res;
  res.optimizer_states_before = static_cast<int64_t>(optimizer.state().size());
  res.optimizer_states_after = res.optimizer_states_before;
  if (model) {
    SPIEL_CHECK_FALSE(model->bf16_inference_cache_ready());
  }
  struct WeightDecayRestoreGuard {
    torch::optim::AdamW& opt;
    std::vector<double> saved_wd;
    explicit WeightDecayRestoreGuard(torch::optim::AdamW& o) : opt(o) {
      saved_wd.reserve(opt.param_groups().size());
      for (auto& group : opt.param_groups()) {
        auto& opts = static_cast<torch::optim::AdamWOptions&>(group.options());
        saved_wd.push_back(opts.weight_decay());
        opts.weight_decay(0.0);
      }
    }
    ~WeightDecayRestoreGuard() {
      size_t idx = 0;
      for (auto& group : opt.param_groups()) {
        if (idx < saved_wd.size()) {
          static_cast<torch::optim::AdamWOptions&>(group.options())
              .weight_decay(saved_wd[idx++]);
        }
      }
    }
  } wd_guard(optimizer);
  const double aux_logit_cap =
      (anchoring_config.logit_cap > 0.0)
          ? anchoring_config.logit_cap
          : ::absl::GetFlag(::FLAGS_logit_cap);
  if (!model || search_examples.empty() || search_loss_coef <= 0.0) {
    return res;
  }

  const int64_t aux_n = static_cast<int64_t>(search_examples.size());
  const int mb_size = (batch_size > 0 && batch_size < aux_n) ? batch_size : aux_n;
  const int num_mb = static_cast<int>((aux_n + mb_size - 1) / mb_size);
  if (max_steps <= 0) {
    max_steps = std::max(1, epochs) * num_mb;
  }
  res.steps_budgeted = max_steps;
  if (max_steps <= 0) {
    return res;
  }

  // 1. Mandatory Semantic-Policy Parity Validation
  if (model->with_semantic_scorer_) {
    for (int64_t i = 0; i < aux_n; ++i) {
      if (search_examples[i].candidate_data.empty()) {
        open_spiel::SpielFatalError(
            "TrainSearchAuxiliaryPhase: semantic scorer is enabled but candidate descriptors "
            "are missing or empty on search examples.");
      }
    }
  }

  const int64_t r_n = static_cast<int64_t>(reference_rollout_sample.size());
  std::vector<const dune_semantic::CandidateActionData*> ref_cands(r_n, nullptr);
  if (model->with_semantic_scorer_ && r_n > 0) {
    for (int64_t i = 0; i < r_n; ++i) {
      if (reference_rollout_sample[i].candidate_data.empty()) {
        open_spiel::SpielFatalError(
            "TrainSearchAuxiliaryPhase: semantic scorer is enabled but candidate descriptors "
            "are missing or empty on reference rollout sample.");
      }
      ref_cands[i] = &reference_rollout_sample[i].candidate_data;
    }
  }

  // 2. Snapshot initial value head parameters to test bitwise immutability (Correction 3)
  std::vector<torch::Tensor> initial_value_params;
  for (const auto& p : model->value_head->parameters()) {
    initial_value_params.push_back(p.detach().clone());
  }
  if (model->use_nonlinear_value_head_ && model->value_head2) {
    for (const auto& p : model->value_head2->parameters()) {
      initial_value_params.push_back(p.detach().clone());
    }
  }

  // 3. Shuffle search training labels using seeded permutation and pre-pack tensors
  std::vector<int64_t> perm(aux_n);
  std::iota(perm.begin(), perm.end(), 0);
  std::mt19937_64 rng(aux_seed != 0 ? aux_seed : 1234567ULL);
  std::shuffle(perm.begin(), perm.end(), rng);

  std::vector<const dune_semantic::CandidateActionData*> aux_cands(aux_n, nullptr);
  auto af = torch::TensorOptions().dtype(torch::kFloat32);
  auto ab = torch::TensorOptions().dtype(torch::kBool);
  torch::Tensor s = torch::zeros({aux_n, obs_size}, af);
  torch::Tensor m = torch::zeros({aux_n, action_dim}, ab);
  torch::Tensor t = torch::zeros({aux_n, action_dim}, af);
  float* sp = s.data_ptr<float>();
  bool* mp = m.data_ptr<bool>();
  float* tp = t.data_ptr<float>();

  for (int64_t i = 0; i < aux_n; ++i) {
    int64_t src_idx = perm[i];
    const SearchTrainingExample& ex = search_examples[src_idx];
    int64_t copy = std::min<int64_t>(obs_size, static_cast<int64_t>(ex.observation.size()));
    if (copy > 0) {
      std::memcpy(sp + i * obs_size, ex.observation.data(), copy * sizeof(float));
    }
    for (size_t j = 0; j < ex.legal_actions.size(); ++j) {
      Action a = ex.legal_actions[j];
      if (a >= 0 && a < action_dim) {
        mp[i * action_dim + a] = true;
        tp[i * action_dim + a] = static_cast<float>(
            (j < ex.normalized_visits.size()) ? ex.normalized_visits[j] : 0.0);
      }
    }
    if (model->with_semantic_scorer_) {
      aux_cands[i] = &search_examples[src_idx].candidate_data;
    }
  }

  torch::Tensor aux_states = s.to(device);
  torch::Tensor aux_masks = m.to(device);
  torch::Tensor aux_targets = t.to(device);

  // 4. Pre-compute reference log-probs using the full deployed semantic policy
  torch::Tensor ref_alogp;
  {
    torch::NoGradGuard no_grad;
    AutocastGuard autocast_guard(c10::DeviceType::CUDA, device.is_cuda() && ::absl::GetFlag(::FLAGS_train_amp));
    auto ref_ao = model->forward(aux_states);
    if (model->with_semantic_scorer_ && model->semantic_scorer_) {
      dune_semantic::ApplySemanticScorerBatch(
          model->semantic_scorer_, ref_ao.trunk, aux_cands, ref_ao.logits, device);
    }
    torch::Tensor ref_alogits = CenterAndCapLogitsTensor(ref_ao.logits, aux_masks, aux_logit_cap);
    torch::Tensor ref_amasked = ref_alogits.masked_fill(aux_masks.logical_not(), -1e9f);
    ref_alogp = torch::log_softmax(ref_amasked, -1).detach();
  }

  // Record initial CE across the full auxiliary dataset before any gradient updates
  res.initial_aux_ce = -(aux_targets * ref_alogp).sum(-1).mean().item<double>();
  res.final_aux_ce = res.initial_aux_ce;

  // Pre-pack reference rollout sample if provided
  torch::Tensor ref_rollout_states, ref_rollout_masks, ref_rollout_alogp, ref_rollout_values;
  if (!reference_rollout_sample.empty()) {
    torch::Tensor rs = torch::zeros({r_n, obs_size}, af);
    torch::Tensor rm = torch::zeros({r_n, action_dim}, ab);
    float* rsp = rs.data_ptr<float>();
    bool* rmp = rm.data_ptr<bool>();
    for (int64_t i = 0; i < r_n; ++i) {
      const auto& tr = reference_rollout_sample[i];
      int64_t c = std::min<int64_t>(obs_size, static_cast<int64_t>(tr.state.size()));
      if (c > 0) std::memcpy(rsp + i * obs_size, tr.state.data(), c * sizeof(float));
      for (Action a : tr.legal_actions) {
        if (a >= 0 && a < action_dim) rmp[i * action_dim + a] = true;
      }
    }
    ref_rollout_states = rs.to(device);
    ref_rollout_masks = rm.to(device);
    torch::NoGradGuard no_grad;
    AutocastGuard autocast_guard(c10::DeviceType::CUDA, device.is_cuda() && ::absl::GetFlag(::FLAGS_train_amp));
    auto r_ao = model->forward(ref_rollout_states);
    if (model->with_semantic_scorer_ && model->semantic_scorer_) {
      dune_semantic::ApplySemanticScorerBatch(
          model->semantic_scorer_, r_ao.trunk, ref_cands, r_ao.logits, device);
    }
    torch::Tensor r_logits = CenterAndCapLogitsTensor(r_ao.logits, ref_rollout_masks, aux_logit_cap);
    torch::Tensor r_masked = r_logits.masked_fill(ref_rollout_masks.logical_not(), -1e9f);
    ref_rollout_alogp = torch::log_softmax(r_masked, -1).detach();
    ref_rollout_values = r_ao.values.squeeze(1).detach();
  }

  // 5. Explicit Null Supervision Detection: skip optimization if target is identical to reference
  {
    torch::NoGradGuard no_grad;
    torch::Tensor ref_probs = torch::exp(ref_alogp).masked_fill(aux_masks.logical_not(), 0.0f);
    double target_diff = (ref_probs - aux_targets).abs().max().item<double>();
    if (target_diff < 1e-5) {
      res.null_supervision_skipped = true;
      res.steps_accepted = 0;
      res.final_aux_ce = res.initial_aux_ce;
      res.cumulative_kl_searched = 0.0;
      res.cumulative_kl_rollout = 0.0;
      res.mean_aux_grad_norm = 0.0;
      res.value_head_immutable = true;
      res.critic_probe_mse_shift = 0.0;
      res.max_prob_movement = 0.0;
      res.optimizer_states_after = static_cast<int64_t>(optimizer.state().size());
      return res;
    }
  }

  // 6. Bounded optimizer step loop (at most max_steps accepted steps)
  double grad_norm_accum = 0.0;
  int grad_norm_count = 0;

  if (!anchoring_config.enabled) {
    for (int step = 0; step < max_steps; ++step) {
      // Snapshot model parameters and optimizer state before this step
      auto step_model_backup = ppo_utils_internal::SnapshotModel(model);
      auto step_opt_backup = ppo_utils_internal::SnapshotOptState(optimizer);

      // Slice minibatch
      int64_t mb_in_epoch = (num_mb > 0) ? (step % num_mb) : 0;
      int64_t start_idx = mb_in_epoch * mb_size;
      int64_t cur_len = std::min<int64_t>(mb_size, aux_n - start_idx);
      torch::Tensor mb_s = (cur_len == aux_n) ? aux_states : aux_states.narrow(0, start_idx, cur_len);
      torch::Tensor mb_m = (cur_len == aux_n) ? aux_masks : aux_masks.narrow(0, start_idx, cur_len);
      torch::Tensor mb_t = (cur_len == aux_n) ? aux_targets : aux_targets.narrow(0, start_idx, cur_len);
      std::vector<const dune_semantic::CandidateActionData*> mb_cands;
      if (model->with_semantic_scorer_) {
        mb_cands.reserve(cur_len);
        for (int64_t k = 0; k < cur_len; ++k) {
          mb_cands.push_back(aux_cands[start_idx + k]);
        }
      }

      optimizer.zero_grad();
      torch::Tensor alogp;
      torch::Tensor ce;
      torch::Tensor loss;
      {
        AutocastGuard autocast_guard(c10::DeviceType::CUDA, device.is_cuda() && ::absl::GetFlag(::FLAGS_train_amp));
        auto ao = model->forward(mb_s);
        if (model->with_semantic_scorer_ && model->semantic_scorer_) {
          dune_semantic::ApplySemanticScorerBatch(
              model->semantic_scorer_, ao.trunk, mb_cands, ao.logits, device);
        }
        torch::Tensor alogits = CenterAndCapLogitsTensor(ao.logits, mb_m, aux_logit_cap);
        torch::Tensor amasked = alogits.masked_fill(mb_m.logical_not(), -1e9f);
        alogp = torch::log_softmax(amasked, -1);
        ce = -(mb_t * alogp).sum(-1).mean();
        loss = static_cast<float>(search_loss_coef) * ce;
      }

      if (!torch::isfinite(loss).item<bool>()) {
        ppo_utils_internal::RestoreModel(model, step_model_backup);
        ppo_utils_internal::RestoreOptState(optimizer, step_opt_backup);
        res.step_rejected_nonfinite = true;
        if (res.steps_accepted == 0) res.phase_rejected = true;
        break;
      }

      loss.backward();

      // Zero out any value head gradients (Correction 3)
      for (auto& p : model->value_head->parameters()) {
        if (p.grad().defined()) p.grad().zero_();
      }
      if (model->use_nonlinear_value_head_ && model->value_head2) {
        for (auto& p : model->value_head2->parameters()) {
          if (p.grad().defined()) p.grad().zero_();
        }
      }

      double g_norm = torch::nn::utils::clip_grad_norm_(model->parameters(), 0.5);
      if (!std::isfinite(g_norm)) {
        ppo_utils_internal::RestoreModel(model, step_model_backup);
        ppo_utils_internal::RestoreOptState(optimizer, step_opt_backup);
        res.step_rejected_nonfinite = true;
        if (res.steps_accepted == 0) res.phase_rejected = true;
        break;
      }

      bool grads_finite = true;
      for (const auto& p : model->parameters()) {
        if (p.grad().defined() && !torch::isfinite(p.grad()).all().item<bool>()) {
          grads_finite = false;
          break;
        }
      }
      if (!grads_finite) {
        ppo_utils_internal::RestoreModel(model, step_model_backup);
        ppo_utils_internal::RestoreOptState(optimizer, step_opt_backup);
        res.step_rejected_nonfinite = true;
        if (res.steps_accepted == 0) res.phase_rejected = true;
        break;
      }

      grad_norm_accum += g_norm;
      ++grad_norm_count;

      optimizer.step();

      // Parameter finiteness guard
      bool params_finite = true;
      for (const auto& p : model->parameters()) {
        if (!torch::isfinite(p).all().item<bool>()) {
          params_finite = false;
          break;
        }
      }
      if (!params_finite) {
        ppo_utils_internal::RestoreModel(model, step_model_backup);
        ppo_utils_internal::RestoreOptState(optimizer, step_opt_backup);
        res.step_rejected_nonfinite = true;
        if (res.steps_accepted == 0) res.phase_rejected = true;
        break;
      }

      // Restore value head parameters bitwise (Correction 3)
      size_t vi = 0;
      for (auto& p : model->value_head->parameters()) {
        p.data().copy_(initial_value_params[vi++]);
      }
      if (model->use_nonlinear_value_head_ && model->value_head2) {
        for (auto& p : model->value_head2->parameters()) {
          p.data().copy_(initial_value_params[vi++]);
        }
      }

      // Measure cumulative KL with deployed semantic policy (Correction 2)
      double cand_kl_searched = 0.0;
      double cand_kl_rollout = 0.0;
      bool kl_finite = true;
      {
        torch::NoGradGuard no_grad;
        AutocastGuard autocast_guard(c10::DeviceType::CUDA, device.is_cuda() && ::absl::GetFlag(::FLAGS_train_amp));
        auto cand_ao = model->forward(aux_states);
        if (model->with_semantic_scorer_ && model->semantic_scorer_) {
          dune_semantic::ApplySemanticScorerBatch(
              model->semantic_scorer_, cand_ao.trunk, aux_cands, cand_ao.logits, device);
        }
        torch::Tensor c_logits = CenterAndCapLogitsTensor(cand_ao.logits, aux_masks, aux_logit_cap);
        torch::Tensor c_masked = c_logits.masked_fill(aux_masks.logical_not(), -1e9f);
        torch::Tensor c_logp = torch::log_softmax(c_masked, -1);
        torch::Tensor p_ref = torch::exp(ref_alogp).masked_fill(aux_masks.logical_not(), 0.0f);
        torch::Tensor diff = (ref_alogp - c_logp).masked_fill(aux_masks.logical_not(), 0.0f);
        torch::Tensor kl_t = (p_ref * diff).sum(-1).mean();
        double r_s_kl = kl_t.item<double>();
        if (!std::isfinite(r_s_kl)) {
          kl_finite = false;
        } else {
          cand_kl_searched = std::max(0.0, r_s_kl);
        }

        if (kl_finite && ref_rollout_states.defined() && ref_rollout_states.size(0) > 0) {
          auto r_cand_ao = model->forward(ref_rollout_states);
          if (model->with_semantic_scorer_ && model->semantic_scorer_) {
            dune_semantic::ApplySemanticScorerBatch(
                model->semantic_scorer_, r_cand_ao.trunk, ref_cands, r_cand_ao.logits, device);
          }
          torch::Tensor r_c_logits = CenterAndCapLogitsTensor(r_cand_ao.logits, ref_rollout_masks, aux_logit_cap);
          torch::Tensor r_c_masked = r_c_logits.masked_fill(ref_rollout_masks.logical_not(), -1e9f);
          torch::Tensor r_c_logp = torch::log_softmax(r_c_masked, -1);
          torch::Tensor r_p_ref = torch::exp(ref_rollout_alogp).masked_fill(ref_rollout_masks.logical_not(), 0.0f);
          torch::Tensor r_diff = (ref_rollout_alogp - r_c_logp).masked_fill(ref_rollout_masks.logical_not(), 0.0f);
          torch::Tensor r_kl_t = (r_p_ref * r_diff).sum(-1).mean();
          double r_r_kl = r_kl_t.item<double>();
          if (!std::isfinite(r_r_kl)) {
            kl_finite = false;
          } else {
            cand_kl_rollout = std::max(0.0, r_r_kl);
          }
        }
      }

      if (!kl_finite) {
        ppo_utils_internal::RestoreModel(model, step_model_backup);
        ppo_utils_internal::RestoreOptState(optimizer, step_opt_backup);
        res.step_rejected_nonfinite = true;
        if (res.steps_accepted == 0) res.phase_rejected = true;
        break;
      }

      // Strict KL ceiling check: reject step and rollback to state before this step
      if (max_cumulative_kl > 0.0 &&
          (cand_kl_searched > max_cumulative_kl || cand_kl_rollout > max_cumulative_kl)) {
        ppo_utils_internal::RestoreModel(model, step_model_backup);
        ppo_utils_internal::RestoreOptState(optimizer, step_opt_backup);
        res.step_rejected_on_kl = true;
        if (res.steps_accepted == 0) res.phase_rejected = true;
        break;
      }

      // Step accepted!
      res.steps_accepted++;
      res.accepted_override_presentations += static_cast<int>(cur_len);
      res.cumulative_kl_searched = cand_kl_searched;
      res.cumulative_kl_rollout = cand_kl_rollout;
    }

    if (grad_norm_count > 0) {
      res.mean_aux_grad_norm = grad_norm_accum / grad_norm_count;
    }
    res.optimizer_states_after = static_cast<int64_t>(optimizer.state().size());

    // Measure maximum probability movement and final cross-entropy across full dataset
    {
      torch::NoGradGuard no_grad;
      AutocastGuard autocast_guard(c10::DeviceType::CUDA, device.is_cuda() && ::absl::GetFlag(::FLAGS_train_amp));
      auto final_ao = model->forward(aux_states);
      if (model->with_semantic_scorer_ && model->semantic_scorer_) {
        dune_semantic::ApplySemanticScorerBatch(
            model->semantic_scorer_, final_ao.trunk, aux_cands, final_ao.logits, device);
      }
      torch::Tensor final_logits = CenterAndCapLogitsTensor(final_ao.logits, aux_masks, aux_logit_cap);
      torch::Tensor final_masked = final_logits.masked_fill(aux_masks.logical_not(), -1e9f);
      torch::Tensor final_probs = torch::softmax(final_masked, -1);
      torch::Tensor ref_probs = torch::exp(ref_alogp).masked_fill(aux_masks.logical_not(), 0.0f);
      res.max_prob_movement = (final_probs - ref_probs).abs().max().item<double>();
      if (res.steps_accepted > 0) {
        torch::Tensor final_logp = torch::log_softmax(final_masked, -1);
        res.final_aux_ce = -(aux_targets * final_logp).sum(-1).mean().item<double>();
      }
    }

    if (ref_rollout_states.defined() && ref_rollout_states.size(0) > 0) {
      torch::NoGradGuard no_grad;
      AutocastGuard autocast_guard(c10::DeviceType::CUDA, device.is_cuda() && ::absl::GetFlag(::FLAGS_train_amp));
      auto final_ao = model->forward(ref_rollout_states);
      auto v_final = final_ao.values.squeeze(1);
      res.critic_probe_mse_shift = (v_final - ref_rollout_values).pow(2).mean().item<double>();
    }
  } else {
    // -------------------------------------------------------------
    // Anchored Aux Supervision Phase (Step 3)
    // -------------------------------------------------------------
    auto anchored = ppo_utils_internal::PrepareAnchoredAuxBatches(
        model, agreement_examples, all_rollout_transitions,
        obs_size, action_dim, device, aux_logit_cap, &rng);

    double accum_override_ce = 0.0;
    double accum_beta_kl = 0.0;
    double accum_alpha_mse = 0.0;

    for (int step = 0; step < max_steps; ++step) {
      auto step_model_backup = ppo_utils_internal::SnapshotModel(model);
      auto step_opt_backup = ppo_utils_internal::SnapshotOptState(optimizer);

      int64_t mb_in_epoch = (num_mb > 0) ? (step % num_mb) : 0;
      int64_t start_idx = mb_in_epoch * mb_size;
      int64_t cur_len = std::min<int64_t>(mb_size, aux_n - start_idx);
      torch::Tensor mb_s = (cur_len == aux_n) ? aux_states : aux_states.narrow(0, start_idx, cur_len);
      torch::Tensor mb_m = (cur_len == aux_n) ? aux_masks : aux_masks.narrow(0, start_idx, cur_len);
      torch::Tensor mb_t = (cur_len == aux_n) ? aux_targets : aux_targets.narrow(0, start_idx, cur_len);
      std::vector<const dune_semantic::CandidateActionData*> mb_cands;
      if (model->with_semantic_scorer_) {
        mb_cands.reserve(cur_len);
        for (int64_t k = 0; k < cur_len; ++k) {
          mb_cands.push_back(aux_cands[start_idx + k]);
        }
      }

      optimizer.zero_grad();
      torch::Tensor ovr_ce;
      {
        AutocastGuard autocast_guard(c10::DeviceType::CUDA, device.is_cuda() && ::absl::GetFlag(::FLAGS_train_amp));
        auto ao = model->forward(mb_s);
        if (model->with_semantic_scorer_ && model->semantic_scorer_) {
          dune_semantic::ApplySemanticScorerBatch(
              model->semantic_scorer_, ao.trunk, mb_cands, ao.logits, device);
        }
        torch::Tensor alogits = CenterAndCapLogitsTensor(ao.logits, mb_m, aux_logit_cap);
        torch::Tensor amasked = alogits.masked_fill(mb_m.logical_not(), -1e9f);
        torch::Tensor alogp = torch::log_softmax(amasked, -1);
        ovr_ce = -(mb_t * alogp).sum(-1).mean();
      }

      torch::Tensor anc_kl = torch::tensor(0.0f, torch::TensorOptions().dtype(torch::kFloat32).device(device));
      torch::Tensor anc_v_mse = torch::tensor(0.0f, torch::TensorOptions().dtype(torch::kFloat32).device(device));

      if (anchored.anchor_batch.count > 0) {
        int64_t anc_sample_n = std::min<int64_t>(anchoring_config.anchor_sample_size, anchored.anchor_batch.count);
        std::vector<int64_t> sample_indices(anc_sample_n);
        for (int64_t i = 0; i < anc_sample_n; ++i) {
          sample_indices[i] = rng() % anchored.anchor_batch.count;
        }
        torch::Tensor idx_t = torch::tensor(sample_indices, torch::TensorOptions().dtype(torch::kInt64).device(device));
        torch::Tensor anc_s = anchored.anchor_batch.states.index_select(0, idx_t);
        torch::Tensor anc_m = anchored.anchor_batch.masks.index_select(0, idx_t);
        torch::Tensor anc_ref_logp = anchored.anchor_batch.ref_alogp.index_select(0, idx_t);
        torch::Tensor anc_ref_probs = anchored.anchor_batch.ref_probs.index_select(0, idx_t);
        torch::Tensor anc_ref_vals = anchored.anchor_batch.ref_values.index_select(0, idx_t);
        std::vector<const dune_semantic::CandidateActionData*> anc_cands_sub;
        if (model->with_semantic_scorer_) {
          anc_cands_sub.reserve(anc_sample_n);
          for (int64_t idx : sample_indices) {
            anc_cands_sub.push_back(anchored.anchor_batch.cands[idx]);
          }
        }

        AutocastGuard autocast_guard(c10::DeviceType::CUDA, device.is_cuda() && ::absl::GetFlag(::FLAGS_train_amp));
        auto a_ao = model->forward(anc_s);
        if (model->with_semantic_scorer_ && model->semantic_scorer_) {
          dune_semantic::ApplySemanticScorerBatch(
              model->semantic_scorer_, a_ao.trunk, anc_cands_sub, a_ao.logits, device);
        }
        torch::Tensor a_logits = CenterAndCapLogitsTensor(a_ao.logits, anc_m, aux_logit_cap);
        torch::Tensor a_masked = a_logits.masked_fill(anc_m.logical_not(), -1e9f);
        torch::Tensor a_logp = torch::log_softmax(a_masked, -1);
        torch::Tensor diff = (anc_ref_logp - a_logp).masked_fill(anc_m.logical_not(), 0.0f);
        anc_kl = (anc_ref_probs * diff).sum(-1).mean();
        anc_v_mse = (4.0 * a_ao.values.view({-1}) - 4.0 * anc_ref_vals).pow(2).mean();
      }

      torch::Tensor loss = static_cast<float>(search_loss_coef) * (
          ovr_ce +
          static_cast<float>(anchoring_config.beta) * anc_kl +
          static_cast<float>(anchoring_config.alpha) * anc_v_mse);

      if (!torch::isfinite(loss).item<bool>()) {
        ppo_utils_internal::RestoreModel(model, step_model_backup);
        ppo_utils_internal::RestoreOptState(optimizer, step_opt_backup);
        res.step_rejected_nonfinite = true;
        if (res.steps_accepted == 0) res.phase_rejected = true;
        break;
      }

      loss.backward();

      for (auto& p : model->value_head->parameters()) {
        if (p.grad().defined()) p.grad().zero_();
      }
      if (model->use_nonlinear_value_head_ && model->value_head2) {
        for (auto& p : model->value_head2->parameters()) {
          if (p.grad().defined()) p.grad().zero_();
        }
      }

      double g_norm = torch::nn::utils::clip_grad_norm_(model->parameters(), 0.5);
      if (!std::isfinite(g_norm)) {
        ppo_utils_internal::RestoreModel(model, step_model_backup);
        ppo_utils_internal::RestoreOptState(optimizer, step_opt_backup);
        res.step_rejected_nonfinite = true;
        if (res.steps_accepted == 0) res.phase_rejected = true;
        break;
      }

      bool grads_finite = true;
      for (const auto& p : model->parameters()) {
        if (p.grad().defined() && !torch::isfinite(p.grad()).all().item<bool>()) {
          grads_finite = false;
          break;
        }
      }
      if (!grads_finite) {
        ppo_utils_internal::RestoreModel(model, step_model_backup);
        ppo_utils_internal::RestoreOptState(optimizer, step_opt_backup);
        res.step_rejected_nonfinite = true;
        if (res.steps_accepted == 0) res.phase_rejected = true;
        break;
      }

      grad_norm_accum += g_norm;
      ++grad_norm_count;

      optimizer.step();

      bool params_finite = true;
      for (const auto& p : model->parameters()) {
        if (!torch::isfinite(p).all().item<bool>()) {
          params_finite = false;
          break;
        }
      }
      if (!params_finite) {
        ppo_utils_internal::RestoreModel(model, step_model_backup);
        ppo_utils_internal::RestoreOptState(optimizer, step_opt_backup);
        res.step_rejected_nonfinite = true;
        if (res.steps_accepted == 0) res.phase_rejected = true;
        break;
      }

      size_t vi = 0;
      for (auto& p : model->value_head->parameters()) {
        p.data().copy_(initial_value_params[vi++]);
      }
      if (model->use_nonlinear_value_head_ && model->value_head2) {
        for (auto& p : model->value_head2->parameters()) {
          p.data().copy_(initial_value_params[vi++]);
        }
      }

      // Rail rejection check
      double cand_rail_agr_kl = ppo_utils_internal::EvaluateBatchKL(model, anchored.rail_agr_batch, device, aux_logit_cap);
      double cand_rail_roll_kl = ppo_utils_internal::EvaluateBatchKL(model, anchored.rail_roll_batch, device, aux_logit_cap);
      int64_t total_rail = anchored.rail_agr_batch.count + anchored.rail_roll_batch.count;
      double cand_rail_total_kl = 0.0;
      if (total_rail > 0) {
        cand_rail_total_kl = (cand_rail_agr_kl * anchored.rail_agr_batch.count + cand_rail_roll_kl * anchored.rail_roll_batch.count) / total_rail;
      }

      // Override labels KL ceiling check (0.05 max cumulative KL)
      double cand_override_kl = 0.0;
      {
        torch::NoGradGuard no_grad;
        AutocastGuard autocast_guard(c10::DeviceType::CUDA, device.is_cuda() && ::absl::GetFlag(::FLAGS_train_amp));
        auto cand_ao = model->forward(aux_states);
        if (model->with_semantic_scorer_ && model->semantic_scorer_) {
          dune_semantic::ApplySemanticScorerBatch(
              model->semantic_scorer_, cand_ao.trunk, aux_cands, cand_ao.logits, device);
        }
        torch::Tensor c_logits = CenterAndCapLogitsTensor(cand_ao.logits, aux_masks, aux_logit_cap);
        torch::Tensor c_masked = c_logits.masked_fill(aux_masks.logical_not(), -1e9f);
        torch::Tensor c_logp = torch::log_softmax(c_masked, -1);
        torch::Tensor p_ref = torch::exp(ref_alogp).masked_fill(aux_masks.logical_not(), 0.0f);
        torch::Tensor diff = (ref_alogp - c_logp).masked_fill(aux_masks.logical_not(), 0.0f);
        cand_override_kl = (p_ref * diff).sum(-1).mean().item<double>();
      }

      if (!std::isfinite(cand_rail_total_kl) || !std::isfinite(cand_override_kl) ||
          cand_rail_total_kl > anchoring_config.rail_kl_target ||
          (max_cumulative_kl > 0.0 && cand_override_kl > max_cumulative_kl)) {
        ppo_utils_internal::RestoreModel(model, step_model_backup);
        ppo_utils_internal::RestoreOptState(optimizer, step_opt_backup);
        res.step_rejected_on_kl = true;
        if (res.steps_accepted == 0) res.phase_rejected = true;
        break;
      }

      res.steps_accepted++;
      res.accepted_override_presentations += static_cast<int>(cur_len);
      res.rail_kl_total = cand_rail_total_kl;
      res.rail_kl_agreement = cand_rail_agr_kl;
      res.rail_kl_non_high_stakes = cand_rail_roll_kl;
      accum_override_ce += ovr_ce.item<double>();
      accum_beta_kl += (anchoring_config.beta * anc_kl).item<double>();
      accum_alpha_mse += (anchoring_config.alpha * anc_v_mse).item<double>();
    }

    if (grad_norm_count > 0) {
      res.mean_aux_grad_norm = grad_norm_accum / grad_norm_count;
    }
    res.optimizer_states_after = static_cast<int64_t>(optimizer.state().size());

    if (res.steps_accepted > 0) {
      res.mean_override_ce_loss = accum_override_ce / res.steps_accepted;
      res.mean_anchor_kl_loss = accum_beta_kl / res.steps_accepted;
      res.mean_anchor_value_mse_loss = accum_alpha_mse / res.steps_accepted;

      torch::NoGradGuard no_grad;
      AutocastGuard autocast_guard(c10::DeviceType::CUDA, device.is_cuda() && ::absl::GetFlag(::FLAGS_train_amp));
      auto final_ao = model->forward(aux_states);
      if (model->with_semantic_scorer_ && model->semantic_scorer_) {
        dune_semantic::ApplySemanticScorerBatch(
            model->semantic_scorer_, final_ao.trunk, aux_cands, final_ao.logits, device);
      }
      torch::Tensor final_logits = CenterAndCapLogitsTensor(final_ao.logits, aux_masks, aux_logit_cap);
      torch::Tensor final_masked = final_logits.masked_fill(aux_masks.logical_not(), -1e9f);
      torch::Tensor final_probs = torch::softmax(final_masked, -1);
      torch::Tensor ref_probs = torch::exp(ref_alogp).masked_fill(aux_masks.logical_not(), 0.0f);
      res.max_prob_movement = (final_probs - ref_probs).abs().max().item<double>();
      torch::Tensor final_logp = torch::log_softmax(final_masked, -1);
      res.final_aux_ce = -(aux_targets * final_logp).sum(-1).mean().item<double>();
      torch::Tensor diff = (ref_alogp - final_logp).masked_fill(aux_masks.logical_not(), 0.0f);
      res.override_kl = (ref_probs * diff).sum(-1).mean().item<double>();

      if (anchored.rail_roll_batch.count > 0) {
        double cur_rail_ent = ppo_utils_internal::EvaluateBatchEntropy(model, anchored.rail_roll_batch, device, aux_logit_cap);
        res.delta_entropy_rail = cur_rail_ent - anchored.rail_roll_batch.ref_mean_entropy;
      }
      if (anchored.hs_batch.count > 0) {
        double cur_hs_ent = ppo_utils_internal::EvaluateBatchEntropy(model, anchored.hs_batch, device, aux_logit_cap);
        res.delta_entropy_high_stakes = cur_hs_ent - anchored.hs_batch.ref_mean_entropy;
      }
      if (anchored.anchor_batch.count > 0) {
        torch::NoGradGuard no_grad;
        AutocastGuard autocast_guard(c10::DeviceType::CUDA, device.is_cuda() && ::absl::GetFlag(::FLAGS_train_amp));
        auto a_ao = model->forward(anchored.anchor_batch.states);
        res.critic_probe_mse_shift_anchors = (a_ao.values.view({-1}) - anchored.anchor_batch.ref_values).pow(2).mean().item<double>();
        res.critic_probe_mse_shift = res.critic_probe_mse_shift_anchors;
      }
      {
        torch::NoGradGuard no_grad;
        AutocastGuard autocast_guard(c10::DeviceType::CUDA, device.is_cuda() && ::absl::GetFlag(::FLAGS_train_amp));
        double agr_mse = 0.0;
        if (anchored.rail_agr_batch.count > 0) {
          auto ao = model->forward(anchored.rail_agr_batch.states);
          agr_mse = (ao.values.view({-1}) - anchored.rail_agr_batch.ref_values).pow(2).mean().item<double>();
        }
        double roll_mse = 0.0;
        if (anchored.rail_roll_batch.count > 0) {
          auto ao = model->forward(anchored.rail_roll_batch.states);
          roll_mse = (ao.values.view({-1}) - anchored.rail_roll_batch.ref_values).pow(2).mean().item<double>();
        }
        int64_t total_rail_count = anchored.rail_agr_batch.count + anchored.rail_roll_batch.count;
        if (total_rail_count > 0) {
          res.critic_probe_mse_shift_rail = (agr_mse * anchored.rail_agr_batch.count + roll_mse * anchored.rail_roll_batch.count) / total_rail_count;
        }
      }
    }
  }

  // 8. Final post-phase value head immutability assert (Correction 3)
  bool val_immut = true;
  size_t v_idx = 0;
  for (const auto& p : model->value_head->parameters()) {
    if (!torch::equal(p, initial_value_params[v_idx++])) val_immut = false;
  }
  if (model->use_nonlinear_value_head_ && model->value_head2) {
    for (const auto& p : model->value_head2->parameters()) {
      if (!torch::equal(p, initial_value_params[v_idx++])) val_immut = false;
    }
  }
  res.value_head_immutable = val_immut;

  return res;
}

}  // namespace open_spiel

#endif  // OPEN_SPIEL_BUILD_WITH_LIBTORCH
