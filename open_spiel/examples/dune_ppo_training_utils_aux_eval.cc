#include "dune_ppo_training_utils_internal.h"
#include "open_spiel/abseil-cpp/absl/flags/flag.h"

#include <algorithm>
#include <cstring>
#include <numeric>
#include <random>
#include <unordered_map>
#include <vector>

#ifdef OPEN_SPIEL_BUILD_WITH_LIBTORCH

namespace open_spiel {
namespace ppo_utils_internal {

PackedEvaluationBatch PackEvaluationBatch(
    const std::vector<const std::vector<float>*>& obs_ptrs,
    const std::vector<const std::vector<open_spiel::Action>*>& legals_ptrs,
    const std::vector<const dune_semantic::CandidateActionData*>& cand_ptrs,
    int64_t obs_size, int64_t action_dim,
    const std::shared_ptr<SharedDunePolicyValueNetImpl>& model,
    torch::Device device,
    bool compute_entropy,
    bool compute_values,
    double logit_cap) {
  PackedEvaluationBatch batch;
  const int64_t n = static_cast<int64_t>(obs_ptrs.size());
  batch.count = n;
  if (n == 0) return batch;
  batch.cands = cand_ptrs;

  auto af = torch::TensorOptions().dtype(torch::kFloat32);
  auto ab = torch::TensorOptions().dtype(torch::kBool);
  torch::Tensor s = torch::zeros({n, obs_size}, af);
  torch::Tensor m = torch::zeros({n, action_dim}, ab);
  float* sp = s.data_ptr<float>();
  bool* mp = m.data_ptr<bool>();

  for (int64_t i = 0; i < n; ++i) {
    if (obs_ptrs[i]) {
      int64_t c = std::min<int64_t>(obs_size, static_cast<int64_t>(obs_ptrs[i]->size()));
      if (c > 0) std::memcpy(sp + i * obs_size, obs_ptrs[i]->data(), c * sizeof(float));
    }
    if (legals_ptrs[i]) {
      for (Action a : *legals_ptrs[i]) {
        if (a >= 0 && a < action_dim) mp[i * action_dim + a] = true;
      }
    }
  }
  batch.states = s.to(device);
  batch.masks = m.to(device);

  torch::NoGradGuard no_grad;
  AutocastGuard autocast_guard(c10::DeviceType::CUDA, device.is_cuda() && ::absl::GetFlag(::FLAGS_train_amp));
  auto ao = model->forward(batch.states);
  if (model->with_semantic_scorer_ && model->semantic_scorer_) {
    dune_semantic::ApplySemanticScorerBatch(
        model->semantic_scorer_, ao.trunk, batch.cands, ao.logits, device);
  }
  torch::Tensor logits = CenterAndCapLogitsTensor(ao.logits, batch.masks, logit_cap);
  torch::Tensor masked = logits.masked_fill(batch.masks.logical_not(), -1e9f);
  batch.ref_alogp = torch::log_softmax(masked, -1).detach();
  batch.ref_probs = torch::exp(batch.ref_alogp).masked_fill(batch.masks.logical_not(), 0.0f).detach();
  if (compute_values) {
    batch.ref_values = ao.values.view({-1}).detach();
  }
  if (compute_entropy) {
    torch::Tensor p = batch.ref_probs;
    torch::Tensor logp = batch.ref_alogp.masked_fill(batch.masks.logical_not(), 0.0f);
    torch::Tensor ent = -(p * logp).sum(-1);
    batch.ref_mean_entropy = ent.mean().item<double>();
  }
  return batch;
}

double EvaluateBatchKL(
    const std::shared_ptr<SharedDunePolicyValueNetImpl>& model,
    const PackedEvaluationBatch& batch,
    torch::Device device,
    double logit_cap) {
  if (batch.count == 0) return 0.0;
  torch::NoGradGuard no_grad;
  AutocastGuard autocast_guard(c10::DeviceType::CUDA, device.is_cuda() && ::absl::GetFlag(::FLAGS_train_amp));
  auto ao = model->forward(batch.states);
  if (model->with_semantic_scorer_ && model->semantic_scorer_) {
    dune_semantic::ApplySemanticScorerBatch(
        model->semantic_scorer_, ao.trunk, batch.cands, ao.logits, device);
  }
  torch::Tensor logits = CenterAndCapLogitsTensor(ao.logits, batch.masks, logit_cap);
  torch::Tensor masked = logits.masked_fill(batch.masks.logical_not(), -1e9f);
  torch::Tensor logp = torch::log_softmax(masked, -1);
  torch::Tensor diff = (batch.ref_alogp - logp).masked_fill(batch.masks.logical_not(), 0.0f);
  torch::Tensor kl = (batch.ref_probs * diff).sum(-1).mean();
  return kl.item<double>();
}

double EvaluateBatchEntropy(
    const std::shared_ptr<SharedDunePolicyValueNetImpl>& model,
    const PackedEvaluationBatch& batch,
    torch::Device device,
    double logit_cap) {
  if (batch.count == 0) return 0.0;
  torch::NoGradGuard no_grad;
  AutocastGuard autocast_guard(c10::DeviceType::CUDA, device.is_cuda() && ::absl::GetFlag(::FLAGS_train_amp));
  auto ao = model->forward(batch.states);
  if (model->with_semantic_scorer_ && model->semantic_scorer_) {
    dune_semantic::ApplySemanticScorerBatch(
        model->semantic_scorer_, ao.trunk, batch.cands, ao.logits, device);
  }
  torch::Tensor logits = CenterAndCapLogitsTensor(ao.logits, batch.masks, logit_cap);
  torch::Tensor masked = logits.masked_fill(batch.masks.logical_not(), -1e9f);
  torch::Tensor logp = torch::log_softmax(masked, -1);
  torch::Tensor p = torch::exp(logp).masked_fill(batch.masks.logical_not(), 0.0f);
  torch::Tensor ent = -(p * logp.masked_fill(batch.masks.logical_not(), 0.0f)).sum(-1);
  return ent.mean().item<double>();
}

std::unordered_map<void*, OptParamState> SnapshotOptState(
    torch::optim::AdamW& opt) {
  std::unordered_map<void*, OptParamState> backup;
  for (auto& group : opt.param_groups()) {
    for (auto& param : group.params()) {
      auto key = param.unsafeGetTensorImpl();
      auto it = opt.state().find(key);
      if (it != opt.state().end()) {
        auto* adam_state = dynamic_cast<torch::optim::AdamWParamState*>(it->second.get());
        if (adam_state) {
          OptParamState st;
          st.step = adam_state->step();
          st.exp_avg = adam_state->exp_avg().clone();
          st.exp_avg_sq = adam_state->exp_avg_sq().clone();
          if (adam_state->max_exp_avg_sq().defined()) {
            st.max_exp_avg_sq = adam_state->max_exp_avg_sq().clone();
          }
          backup[key] = std::move(st);
        }
      }
    }
  }
  return backup;
}

void RestoreOptState(
    torch::optim::AdamW& opt,
    const std::unordered_map<void*, OptParamState>& backup) {
  // Remove newly created states
  std::vector<void*> keys_to_remove;
  for (const auto& kv : opt.state()) {
    if (backup.find(kv.first) == backup.end()) {
      keys_to_remove.push_back(kv.first);
    }
  }
  for (void* k : keys_to_remove) {
    opt.state().erase(k);
  }
  // Restore values for pre-existing keys
  for (const auto& pair : backup) {
    auto it = opt.state().find(pair.first);
    if (it != opt.state().end()) {
      auto* adam_state = dynamic_cast<torch::optim::AdamWParamState*>(it->second.get());
      if (adam_state) {
        adam_state->step(pair.second.step);
        adam_state->exp_avg().copy_(pair.second.exp_avg);
        adam_state->exp_avg_sq().copy_(pair.second.exp_avg_sq);
        if (pair.second.max_exp_avg_sq.defined() && adam_state->max_exp_avg_sq().defined()) {
          adam_state->max_exp_avg_sq().copy_(pair.second.max_exp_avg_sq);
        }
      }
    }
  }
}

std::vector<torch::Tensor> SnapshotModel(
    const std::shared_ptr<SharedDunePolicyValueNetImpl>& m) {
  std::vector<torch::Tensor> backup;
  for (const auto& p : m->parameters()) {
    backup.push_back(p.detach().clone());
  }
  return backup;
}

void RestoreModel(
    const std::shared_ptr<SharedDunePolicyValueNetImpl>& m,
    const std::vector<torch::Tensor>& backup) {
  size_t p_idx = 0;
  for (auto& p : m->parameters()) {
    p.data().copy_(backup[p_idx++]);
  }
}

AnchoredAuxBatches PrepareAnchoredAuxBatches(
    const std::shared_ptr<SharedDunePolicyValueNetImpl>& model,
    const std::vector<SearchTrainingExample>& agreement_examples,
    const std::vector<PpoTransition>& all_rollout_transitions,
    int64_t obs_size, int64_t action_dim,
    torch::Device device,
    double aux_logit_cap,
    std::mt19937_64* rng) {
    std::vector<size_t> agr_perm(agreement_examples.size());
    std::iota(agr_perm.begin(), agr_perm.end(), 0);
    std::shuffle(agr_perm.begin(), agr_perm.end(), *rng);
    size_t n_agr_anc = agreement_examples.size() / 2;

    std::vector<const std::vector<float>*> anc_obs;
    std::vector<const std::vector<Action>*> anc_legals;
    std::vector<const dune_semantic::CandidateActionData*> anc_cands;

    std::vector<const std::vector<float>*> rail_agr_obs;
    std::vector<const std::vector<Action>*> rail_agr_legals;
    std::vector<const dune_semantic::CandidateActionData*> rail_agr_cands;

    for (size_t i = 0; i < agreement_examples.size(); ++i) {
      size_t idx = agr_perm[i];
      if (i < n_agr_anc) {
        anc_obs.push_back(&agreement_examples[idx].observation);
        anc_legals.push_back(&agreement_examples[idx].legal_actions);
        anc_cands.push_back(&agreement_examples[idx].candidate_data);
      } else {
        rail_agr_obs.push_back(&agreement_examples[idx].observation);
        rail_agr_legals.push_back(&agreement_examples[idx].legal_actions);
        rail_agr_cands.push_back(&agreement_examples[idx].candidate_data);
      }
    }

    std::vector<const PpoTransition*> rollout_nontrivial;
    std::vector<const PpoTransition*> rollout_non_hs;
    std::vector<const PpoTransition*> rollout_hs;
    for (const auto& tr : all_rollout_transitions) {
      if (tr.legal_actions.size() > 1) {
        rollout_nontrivial.push_back(&tr);
        if (tr.is_high_stakes) {
          rollout_hs.push_back(&tr);
        } else {
          rollout_non_hs.push_back(&tr);
        }
      }
    }

    std::vector<size_t> r_nt_perm(rollout_nontrivial.size());
    std::iota(r_nt_perm.begin(), r_nt_perm.end(), 0);
    std::shuffle(r_nt_perm.begin(), r_nt_perm.end(), *rng);
    size_t n_anc_rollout = std::min<size_t>(4096, rollout_nontrivial.size());
    for (size_t i = 0; i < n_anc_rollout; ++i) {
      const auto* tr = rollout_nontrivial[r_nt_perm[i]];
      anc_obs.push_back(&tr->state);
      anc_legals.push_back(&tr->legal_actions);
      anc_cands.push_back(&tr->candidate_data);
    }

    std::vector<const std::vector<float>*> rail_roll_obs;
    std::vector<const std::vector<Action>*> rail_roll_legals;
    std::vector<const dune_semantic::CandidateActionData*> rail_roll_cands;

    std::vector<size_t> r_non_hs_perm(rollout_non_hs.size());
    std::iota(r_non_hs_perm.begin(), r_non_hs_perm.end(), 0);
    std::shuffle(r_non_hs_perm.begin(), r_non_hs_perm.end(), *rng);
    size_t n_rail_rollout = std::min<size_t>(1024, rollout_non_hs.size());
    for (size_t i = 0; i < n_rail_rollout; ++i) {
      const auto* tr = rollout_non_hs[r_non_hs_perm[i]];
      rail_roll_obs.push_back(&tr->state);
      rail_roll_legals.push_back(&tr->legal_actions);
      rail_roll_cands.push_back(&tr->candidate_data);
    }

    std::vector<const std::vector<float>*> hs_obs;
    std::vector<const std::vector<Action>*> hs_legals;
    std::vector<const dune_semantic::CandidateActionData*> hs_cands;

    std::vector<size_t> r_hs_perm(rollout_hs.size());
    std::iota(r_hs_perm.begin(), r_hs_perm.end(), 0);
    std::shuffle(r_hs_perm.begin(), r_hs_perm.end(), *rng);
    size_t n_hs = std::min<size_t>(1024, rollout_hs.size());
    for (size_t i = 0; i < n_hs; ++i) {
      const auto* tr = rollout_hs[r_hs_perm[i]];
      hs_obs.push_back(&tr->state);
      hs_legals.push_back(&tr->legal_actions);
      hs_cands.push_back(&tr->candidate_data);
    }

  AnchoredAuxBatches out;
  out.anchor_batch = PackEvaluationBatch(
        anc_obs, anc_legals, anc_cands, obs_size, action_dim, model, device,
        /*compute_entropy=*/false, /*compute_values=*/true, aux_logit_cap);

  out.rail_agr_batch = PackEvaluationBatch(
        rail_agr_obs, rail_agr_legals, rail_agr_cands, obs_size, action_dim, model, device,
        /*compute_entropy=*/true, /*compute_values=*/true, aux_logit_cap);

  out.rail_roll_batch = PackEvaluationBatch(
        rail_roll_obs, rail_roll_legals, rail_roll_cands, obs_size, action_dim, model, device,
        /*compute_entropy=*/true, /*compute_values=*/true, aux_logit_cap);

  out.hs_batch = PackEvaluationBatch(
        hs_obs, hs_legals, hs_cands, obs_size, action_dim, model, device,
        /*compute_entropy=*/true, /*compute_values=*/false, aux_logit_cap);
  return out;
}

}  // namespace ppo_utils_internal

SearchExamplesMetrics EvaluateSearchExamplesMetrics(
    const std::shared_ptr<SharedDunePolicyValueNetImpl>& model,
    const std::vector<SearchTrainingExample>& examples,
    int64_t obs_size, int64_t action_dim,
    torch::Device device) {
  SearchExamplesMetrics metrics;
  if (!model || examples.empty()) return metrics;
  torch::NoGradGuard no_grad;
  const int64_t total_n = static_cast<int64_t>(examples.size());
  metrics.count = total_n;
  const int64_t batch_size = 128;
  double total_ce_sum = 0.0;
  double total_p_search_sum = 0.0;
  int64_t greedy_agreements = 0;
  int64_t total_evaluated = 0;

  auto af = torch::TensorOptions().dtype(torch::kFloat32);
  auto ab = torch::TensorOptions().dtype(torch::kBool);

  for (int64_t start = 0; start < total_n; start += batch_size) {
    int64_t cur_len = std::min<int64_t>(batch_size, total_n - start);
    torch::Tensor s = torch::zeros({cur_len, obs_size}, af);
    torch::Tensor m = torch::zeros({cur_len, action_dim}, ab);
    torch::Tensor t = torch::zeros({cur_len, action_dim}, af);
    float* sp = s.data_ptr<float>();
    bool* mp = m.data_ptr<bool>();
    float* tp = t.data_ptr<float>();

    std::vector<const dune_semantic::CandidateActionData*> mb_cands;
    if (model->with_semantic_scorer_) {
      mb_cands.reserve(cur_len);
    }

    std::vector<Action> search_moves(cur_len, -1);

    for (int64_t i = 0; i < cur_len; ++i) {
      const auto& ex = examples[start + i];
      int64_t copy = std::min<int64_t>(obs_size, static_cast<int64_t>(ex.observation.size()));
      if (copy > 0) {
        std::memcpy(sp + i * obs_size, ex.observation.data(), copy * sizeof(float));
      }
      Action s_move = -1;
      double max_t_val = -1.0;
      for (size_t j = 0; j < ex.legal_actions.size(); ++j) {
        Action a = ex.legal_actions[j];
        if (a >= 0 && a < action_dim) {
          mp[i * action_dim + a] = true;
          double t_val = (j < ex.normalized_visits.size()) ? ex.normalized_visits[j] : 0.0;
          tp[i * action_dim + a] = static_cast<float>(t_val);
          if (t_val > max_t_val) {
            max_t_val = t_val;
            s_move = a;
          }
        }
      }
      search_moves[i] = s_move;
      if (model->with_semantic_scorer_) {
        mb_cands.push_back(&ex.candidate_data);
      }
    }

    torch::Tensor b_s = s.to(device);
    torch::Tensor b_m = m.to(device);
    torch::Tensor b_t = t.to(device);

    AutocastGuard autocast_guard(c10::DeviceType::CUDA, device.is_cuda() && ::absl::GetFlag(::FLAGS_train_amp));
    auto ao = model->forward(b_s);
    if (model->with_semantic_scorer_ && model->semantic_scorer_) {
      dune_semantic::ApplySemanticScorerBatch(
          model->semantic_scorer_, ao.trunk, mb_cands, ao.logits, device);
    }
    torch::Tensor logits = CenterAndCapLogitsTensor(
        ao.logits, b_m, ::absl::GetFlag(::FLAGS_logit_cap));
    torch::Tensor masked = logits.masked_fill(b_m.logical_not(), -1e9f);
    torch::Tensor logp = torch::log_softmax(masked, -1);
    torch::Tensor p = torch::exp(logp);

    torch::Tensor ce_per_sample = -(b_t * logp).sum(-1);
    total_ce_sum += ce_per_sample.sum().item<double>();

    torch::Tensor argmax_action = masked.argmax(-1).to(torch::kCPU);
    torch::Tensor p_cpu = p.to(torch::kCPU);

    auto p_acc = p_cpu.accessor<float, 2>();
    auto argmax_acc = argmax_action.accessor<int64_t, 1>();

    for (int64_t i = 0; i < cur_len; ++i) {
      Action sm = search_moves[i];
      if (sm >= 0 && sm < action_dim) {
        total_p_search_sum += p_acc[i][sm];
        if (argmax_acc[i] == sm) {
          greedy_agreements++;
        }
      }
    }
    total_evaluated += cur_len;
  }

  if (total_evaluated > 0) {
    metrics.mean_ce = total_ce_sum / total_evaluated;
    metrics.mean_p_search_move = total_p_search_sum / total_evaluated;
    metrics.greedy_agreement_pct = 100.0 * static_cast<double>(greedy_agreements) / total_evaluated;
  }
  return metrics;
}

double EvaluateSearchExamplesCrossEntropy(
    const std::shared_ptr<SharedDunePolicyValueNetImpl>& model,
    const std::vector<SearchTrainingExample>& examples,
    int64_t obs_size, int64_t action_dim,
    torch::Device device) {
  return EvaluateSearchExamplesMetrics(model, examples, obs_size, action_dim, device).mean_ce;
}

}  // namespace open_spiel

#endif  // OPEN_SPIEL_BUILD_WITH_LIBTORCH
