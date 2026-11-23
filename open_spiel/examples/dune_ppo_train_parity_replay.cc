#include "dune_ppo_train_internal.h"

namespace open_spiel::ppo_train_internal {

#ifdef OPEN_SPIEL_BUILD_WITH_LIBTORCH

std::string PpoParityRowIdentitySha256(
    int64_t row_index, const PpoTransition& transition) {
  std::stringstream bytes;
  bytes.write(reinterpret_cast<const char*>(&row_index), sizeof(row_index));
  bytes.write(reinterpret_cast<const char*>(&transition.episode_id),
              sizeof(transition.episode_id));
  bytes.write(reinterpret_cast<const char*>(&transition.player_id),
              sizeof(transition.player_id));
  bytes.write(reinterpret_cast<const char*>(&transition.action),
              sizeof(transition.action));
  bytes.write(reinterpret_cast<const char*>(&transition.decision_role),
              sizeof(transition.decision_role));
  bytes.write(reinterpret_cast<const char*>(&transition.advantage),
              sizeof(transition.advantage));
  bytes.write(reinterpret_cast<const char*>(&transition.old_log_prob),
              sizeof(transition.old_log_prob));
  const int64_t legal_count =
      static_cast<int64_t>(transition.legal_actions.size());
  bytes.write(reinterpret_cast<const char*>(&legal_count), sizeof(legal_count));
  for (size_t i = 0; i < transition.legal_actions.size(); ++i) {
    bytes.write(reinterpret_cast<const char*>(&transition.legal_actions[i]),
                sizeof(transition.legal_actions[i]));
    const float raw = i < transition.behavior_raw_legal_logits.size()
                          ? transition.behavior_raw_legal_logits[i]
                          : std::numeric_limits<float>::quiet_NaN();
    const float behavior = i < transition.behavior_legal_log_probs.size()
                               ? transition.behavior_legal_log_probs[i]
                               : std::numeric_limits<float>::quiet_NaN();
    bytes.write(reinterpret_cast<const char*>(&raw), sizeof(raw));
    bytes.write(reinterpret_cast<const char*>(&behavior), sizeof(behavior));
  }
  return ComputeStringSHA256(bytes.str());
}

std::string PpoParitySharedIdentitySha256(
    const std::vector<PpoTransition>& batch) {
  std::string payload;
  for (int64_t i = 0; i < static_cast<int64_t>(batch.size()); ++i) {
    payload.append(PpoParityRowIdentitySha256(i, batch[i]));
    payload.push_back('\n');
  }
  return ComputeStringSHA256(payload);
}

std::string PpoParityCapturedRawSha256(
    const std::vector<PpoTransition>& batch) {
  std::stringstream bytes;
  for (const auto& transition : batch) {
    for (float value : transition.behavior_raw_legal_logits) {
      bytes.write(reinterpret_cast<const char*>(&value), sizeof(value));
    }
  }
  return ComputeStringSHA256(bytes.str());
}

std::string PpoParityCapturedPolicySha256(
    const std::vector<PpoTransition>& batch) {
  std::stringstream bytes;
  for (const auto& transition : batch) {
    for (float value : transition.behavior_legal_log_probs) {
      bytes.write(reinterpret_cast<const char*>(&value), sizeof(value));
    }
  }
  return ComputeStringSHA256(bytes.str());
}

std::vector<PpoNumericalParityRow> ReplayPpoCpuIntegrityCell(
    const std::vector<PpoTransition>& batch, float logit_cap,
    bool require_bf16_grid,
    std::vector<std::string>* schema_errors,
    std::string* row_evidence_sha256, int64_t* raw_values_total,
    int64_t* raw_values_bf16_exact, int64_t* vector_width_rows_ok,
    int64_t* behavior_vector_exact_rows, int64_t* chosen_scalar_exact_rows) {
  if (raw_values_total != nullptr) *raw_values_total = 0;
  if (raw_values_bf16_exact != nullptr) *raw_values_bf16_exact = 0;
  if (vector_width_rows_ok != nullptr) *vector_width_rows_ok = 0;
  if (behavior_vector_exact_rows != nullptr) *behavior_vector_exact_rows = 0;
  if (chosen_scalar_exact_rows != nullptr) *chosen_scalar_exact_rows = 0;
  std::vector<PpoNumericalParityRow> rows;
  rows.reserve(batch.size());
  std::stringstream evidence;
  evidence << "dune_raw_ppo_numerical_parity_v3";
  evidence.put('\0');
  evidence << "rollout_cpu_recompute_integrity";
  evidence.put('\0');

  for (int64_t row_index = 0;
       row_index < static_cast<int64_t>(batch.size()); ++row_index) {
    const PpoTransition& transition = batch[row_index];
    const std::string identity =
        PpoParityRowIdentitySha256(row_index, transition);
    bool row_valid = true;
    auto error = [&](const std::string& message) {
      schema_errors->push_back(absl::StrFormat(
          "row %lld: %s", static_cast<long long>(row_index), message));
      row_valid = false;
    };
    const size_t legal_count = transition.legal_actions.size();
    if (legal_count == 0 ||
        transition.behavior_raw_legal_logits.size() != legal_count ||
        transition.behavior_legal_log_probs.size() != legal_count) {
      error("legal/raw/logprob vector widths differ or are empty");
    } else if (vector_width_rows_ok != nullptr) {
      ++*vector_width_rows_ok;
    }
    if (!PpoParityCapturedRawLogitsValid(
            transition.behavior_raw_legal_logits, require_bf16_grid,
            raw_values_total, raw_values_bf16_exact)) {
      error(require_bf16_grid
                ? "captured raw legal logits are not finite BF16-grid values"
                : "captured raw legal logits are not finite");
    }

    std::vector<float> recomputed;
    std::string recompute_error;
    if (!RecomputePpoBehaviorLegalLogProbs(
            transition.behavior_raw_legal_logits, logit_cap, &recomputed,
            &recompute_error)) {
      error("CPU behavior recompute failed: " + recompute_error);
    }
    bool behavior_exact = recomputed.size() ==
                          transition.behavior_legal_log_probs.size();
    if (behavior_exact) {
      for (size_t i = 0; i < recomputed.size(); ++i) {
        if (PpoParityFloatBits(recomputed[i]) != PpoParityFloatBits(
                transition.behavior_legal_log_probs[i])) {
          behavior_exact = false;
          break;
        }
      }
    }
    if (behavior_exact) {
      if (behavior_vector_exact_rows != nullptr) {
        ++*behavior_vector_exact_rows;
      }
    } else {
      error("CPU recompute does not bit-match stored behavior vector");
    }
    int chosen_index = -1;
    int chosen_count = 0;
    for (size_t i = 0; i < legal_count; ++i) {
      if (transition.legal_actions[i] == transition.action) {
        chosen_index = static_cast<int>(i);
        ++chosen_count;
      }
    }
    bool chosen_exact = chosen_count == 1 && chosen_index >= 0 &&
                        chosen_index < static_cast<int>(
                            transition.behavior_legal_log_probs.size()) &&
                        PpoParityFloatBits(transition.old_log_prob) ==
                            PpoParityFloatBits(
                                transition.behavior_legal_log_probs[
                                    chosen_index]);
    if (chosen_exact) {
      if (chosen_scalar_exact_rows != nullptr) ++*chosen_scalar_exact_rows;
    } else {
      error("chosen scalar does not bit-match stored behavior vector");
    }

    PpoNumericalParityInput input;
    input.legal_count = static_cast<int>(legal_count);
    input.decision_role = transition.decision_role;
    input.advantage = transition.advantage;
    input.chosen_index = chosen_index;
    input.stored_chosen_log_prob = transition.old_log_prob;
    for (float value : transition.behavior_legal_log_probs) {
      input.old_log_probs.push_back(value);
    }
    for (float value : recomputed) input.new_log_probs.push_back(value);
    PpoNumericalParityRow row;
    std::string parity_error;
    if (!ComputePpoNumericalParityRow(input, &row, &parity_error)) {
      error("parity row failed: " + parity_error);
    }
    if (!row_valid) {
      row.schema_errors = std::max<int64_t>(1, row.schema_errors);
      row.nonfinite_values = std::max<int64_t>(1, row.nonfinite_values);
    }
    row.row_identity_sha256 = identity;
    rows.push_back(row);
    evidence << identity;
    evidence.put('\0');
    for (float value : recomputed) {
      evidence.write(reinterpret_cast<const char*>(&value), sizeof(value));
    }
  }
  if (row_evidence_sha256 != nullptr) {
    *row_evidence_sha256 = ComputeStringSHA256(evidence.str());
  }
  return rows;
}

std::vector<PpoNumericalParityRow> ReplayPpoNumericalParityCell(
    const std::shared_ptr<SharedDunePolicyValueNetImpl>& model,
    const std::vector<PpoTransition>& batch, int64_t obs_size,
    int64_t action_dim, torch::Device device, bool model_train_mode,
    bool learner_autocast,
    const std::string& cell_label,
    std::vector<std::string>* schema_errors,
    std::string* row_evidence_sha256, int64_t* model_forward_calls,
    int64_t* dense_mask_rows_ok, std::string* raw_logit_sha256,
    std::string* policy_sha256) {
  std::vector<PpoNumericalParityRow> rows;
  rows.reserve(batch.size());
  std::stringstream evidence;
  std::stringstream raw_bytes, policy_bytes;
  evidence << "dune_raw_ppo_numerical_parity_v3";
  evidence.put('\0');
  evidence << cell_label;
  evidence.put('\0');
  if (model_forward_calls != nullptr) *model_forward_calls = 0;
  if (dense_mask_rows_ok != nullptr) *dense_mask_rows_ok = 0;
  const int64_t chunk_size = std::max<int64_t>(
      1, std::min<int64_t>(absl::GetFlag(FLAGS_ppo_minibatch_size),
                           static_cast<int64_t>(batch.size())));
  const float logit_cap = static_cast<float>(absl::GetFlag(FLAGS_logit_cap));
  const bool was_training = model->is_training();
  if (model_train_mode) {
    model->train();  // exact TrainPpoUpdate module mode
  } else {
    model->eval();   // v1 descriptive-control convention
  }
  torch::NoGradGuard no_grad;

  for (int64_t start = 0; start < static_cast<int64_t>(batch.size());
       start += chunk_size) {
    const int64_t count = std::min<int64_t>(
        chunk_size, static_cast<int64_t>(batch.size()) - start);
    torch::Tensor states_cpu = torch::zeros(
        {count, obs_size}, torch::TensorOptions().dtype(torch::kFloat32));
    torch::Tensor masks_cpu = torch::zeros(
        {count, action_dim}, torch::TensorOptions().dtype(torch::kBool));
    float* state_ptr = states_cpu.data_ptr<float>();
    bool* mask_ptr = masks_cpu.data_ptr<bool>();
    std::vector<bool> packed_row_valid(count, true);
    for (int64_t i = 0; i < count; ++i) {
      const PpoTransition& transition = batch[start + i];
      if (static_cast<int64_t>(transition.state.size()) != obs_size) {
        schema_errors->push_back(absl::StrFormat(
            "row %lld: observation width %lld != %lld",
            static_cast<long long>(start + i),
            static_cast<long long>(transition.state.size()),
            static_cast<long long>(obs_size)));
        packed_row_valid[i] = false;
        continue;
      }
      std::memcpy(state_ptr + i * obs_size, transition.state.data(),
                  obs_size * sizeof(float));
      for (Action action : transition.legal_actions) {
        if (action >= 0 && action < action_dim) {
          mask_ptr[i * action_dim + action] = true;
        }
      }
    }

    const torch::Tensor mask_popcounts =
        masks_cpu.sum(1).to(torch::kCPU).to(torch::kInt64).contiguous();
    const int64_t* mask_popcount_ptr = mask_popcounts.data_ptr<int64_t>();
    for (int64_t i = 0; i < count; ++i) {
      const PpoTransition& transition = batch[start + i];
      if (mask_popcount_ptr[i] ==
          static_cast<int64_t>(transition.legal_actions.size())) {
        if (dense_mask_rows_ok != nullptr) ++*dense_mask_rows_ok;
      } else {
        packed_row_valid[i] = false;
        schema_errors->push_back(absl::StrFormat(
            "row %lld: dense mask popcount %lld != legal count %lld",
            static_cast<long long>(start + i),
            static_cast<long long>(mask_popcount_ptr[i]),
            static_cast<long long>(transition.legal_actions.size())));
      }
    }

    const torch::Tensor states = states_cpu.to(device);
    const torch::Tensor masks = masks_cpu.to(device);
    if (states_cpu.scalar_type() != torch::kFloat32 ||
        states.scalar_type() != torch::kFloat32) {
      schema_errors->push_back("cell input/pre-forward dtype is not Float32");
    }
    torch::Tensor replay_raw_logits, replay_log_probs;
    auto replay_policy_path = [&]() {
      if (model_forward_calls != nullptr) ++*model_forward_calls;
      auto outputs = model->forward(states);
      if (model->with_semantic_scorer_ && model->semantic_scorer_) {
        std::vector<const dune_semantic::CandidateActionData*> cands;
        cands.reserve(count);
        for (int64_t i = 0; i < count; ++i) {
          cands.push_back(&batch[start + i].candidate_data);
        }
        dune_semantic::ApplySemanticScorerBatch(
            model->semantic_scorer_, outputs.trunk, cands, outputs.logits, device);
      }
      replay_raw_logits = outputs.logits;
      const torch::Tensor centered =
          CenterAndCapLogitsTensor(replay_raw_logits, masks, logit_cap);
      const torch::Tensor masked_logits =
          centered.masked_fill(masks.logical_not(), -1e9f);
      replay_log_probs = torch::log_softmax(masked_logits, -1);
    };
    if (learner_autocast) {
      // Exact production PPO policy path: TrainPpoUpdate wraps compute_loss --
      // including forward, CenterAndCapLogitsTensor, masking and log_softmax --
      // in this CUDA BF16 AutocastGuard when --train_amp=true.
      AutocastGuard autocast_guard(c10::DeviceType::CUDA, true);
      replay_policy_path();
    } else {
      // Descriptive cross-precision control retained from v1.
      replay_policy_path();
    }
    const torch::Tensor log_probs_cpu =
        replay_log_probs.to(torch::kCPU).to(torch::kFloat32).contiguous();
    const torch::Tensor raw_logits_cpu =
        replay_raw_logits.to(torch::kCPU).to(torch::kFloat32).contiguous();
    const float* new_log_probs = log_probs_cpu.data_ptr<float>();
    const float* raw_logits = raw_logits_cpu.data_ptr<float>();

    for (int64_t i = 0; i < count; ++i) {
      const int64_t row_index = start + i;
      const PpoTransition& transition = batch[row_index];
      const std::string row_identity =
          PpoParityRowIdentitySha256(row_index, transition);
      PpoNumericalParityInput input;
      input.legal_count = static_cast<int>(transition.legal_actions.size());
      input.decision_role = transition.decision_role;
      input.advantage = transition.advantage;
      input.stored_chosen_log_prob = transition.old_log_prob;
      input.old_log_probs.reserve(input.legal_count);
      input.new_log_probs.reserve(input.legal_count);
      for (int j = 0; j < input.legal_count; ++j) {
        const Action action = transition.legal_actions[j];
        input.old_log_probs.push_back(
            j < static_cast<int>(transition.behavior_legal_log_probs.size())
                ? transition.behavior_legal_log_probs[j]
                : std::numeric_limits<double>::quiet_NaN());
        input.new_log_probs.push_back(
            (action >= 0 && action < action_dim)
                ? new_log_probs[i * action_dim + action]
                : std::numeric_limits<double>::quiet_NaN());
        if (action == transition.action) input.chosen_index = j;
        const float replay_raw =
            (action >= 0 && action < action_dim)
                ? raw_logits[i * action_dim + action]
                : std::numeric_limits<float>::quiet_NaN();
        raw_bytes.write(reinterpret_cast<const char*>(&replay_raw),
                        sizeof(replay_raw));
        const float replay_policy = static_cast<float>(
            input.new_log_probs.back());
        policy_bytes.write(reinterpret_cast<const char*>(&replay_policy),
                           sizeof(replay_policy));
      }
      // Hash the exact row identity and BOTH aligned legal distributions. The
      // artifact's summaries can therefore be tied back to the measured row
      // evidence without embedding a many-megabyte raw dump in JSON.
      evidence.write(reinterpret_cast<const char*>(&row_index),
                     sizeof(row_index));
      evidence.write(reinterpret_cast<const char*>(&transition.episode_id),
                     sizeof(transition.episode_id));
      evidence.write(reinterpret_cast<const char*>(&transition.player_id),
                     sizeof(transition.player_id));
      evidence.write(reinterpret_cast<const char*>(&transition.action),
                     sizeof(transition.action));
      evidence.write(reinterpret_cast<const char*>(&transition.decision_role),
                     sizeof(transition.decision_role));
      evidence.write(reinterpret_cast<const char*>(&transition.advantage),
                     sizeof(transition.advantage));
      evidence.write(reinterpret_cast<const char*>(&transition.old_log_prob),
                     sizeof(transition.old_log_prob));
      for (int j = 0; j < input.legal_count; ++j) {
        evidence.write(
            reinterpret_cast<const char*>(&transition.legal_actions[j]),
            sizeof(transition.legal_actions[j]));
        evidence.write(reinterpret_cast<const char*>(&input.old_log_probs[j]),
                       sizeof(input.old_log_probs[j]));
        evidence.write(reinterpret_cast<const char*>(&input.new_log_probs[j]),
                       sizeof(input.new_log_probs[j]));
        const float raw =
            j < static_cast<int>(transition.behavior_raw_legal_logits.size())
                ? transition.behavior_raw_legal_logits[j]
                : std::numeric_limits<float>::quiet_NaN();
        evidence.write(reinterpret_cast<const char*>(&raw), sizeof(raw));
      }
      if (!packed_row_valid[i]) {
        PpoNumericalParityRow bad;
        bad.legal_count = static_cast<int>(transition.legal_actions.size());
        bad.decision_role = transition.decision_role;
        bad.advantage = transition.advantage;
        bad.schema_errors = 1;
        bad.nonfinite_values = 1;
        bad.row_identity_sha256 = row_identity;
        rows.push_back(bad);
        continue;
      }
      PpoNumericalParityRow row;
      std::string error;
      if (!ComputePpoNumericalParityRow(input, &row, &error)) {
        schema_errors->push_back(absl::StrFormat(
            "row %lld: %s", static_cast<long long>(row_index), error));
      }
      row.row_identity_sha256 = row_identity;
      // V2 never drops a bad row. Schema-error rows remain in every partition
      // and are excluded only from finite metric denominators.
      rows.push_back(row);
    }
  }
  if (was_training) {
    model->train();
  } else {
    model->eval();
  }
  if (row_evidence_sha256 != nullptr) {
    *row_evidence_sha256 = ComputeStringSHA256(evidence.str());
  }
  if (raw_logit_sha256 != nullptr) {
    *raw_logit_sha256 = ComputeStringSHA256(raw_bytes.str());
  }
  if (policy_sha256 != nullptr) {
    *policy_sha256 = ComputeStringSHA256(policy_bytes.str());
  }
  return rows;
}

std::vector<PpoNumericalParityRow> ReplayPpoCapturedLogitsBf16Cell(
    const std::vector<PpoTransition>& batch, int64_t action_dim,
    torch::Device device, float logit_cap,
    std::vector<std::string>* schema_errors,
    std::string* row_evidence_sha256, int64_t* model_forward_calls,
    int64_t* dense_mask_rows_ok) {
  if (model_forward_calls != nullptr) *model_forward_calls = 0;
  if (dense_mask_rows_ok != nullptr) *dense_mask_rows_ok = 0;
  std::vector<PpoNumericalParityRow> rows;
  rows.reserve(batch.size());
  std::stringstream evidence;
  evidence << "dune_raw_ppo_numerical_parity_v3";
  evidence.put('\0');
  evidence << "captured_logits_dense_cuda_bf16_postprocess";
  evidence.put('\0');
  const int64_t chunk_size = std::max<int64_t>(
      1, std::min<int64_t>(absl::GetFlag(FLAGS_ppo_minibatch_size),
                           static_cast<int64_t>(batch.size())));

  for (int64_t start = 0; start < static_cast<int64_t>(batch.size());
       start += chunk_size) {
    const int64_t count = std::min<int64_t>(
        chunk_size, static_cast<int64_t>(batch.size()) - start);
    torch::Tensor dense_cpu = torch::zeros(
        {count, action_dim}, torch::TensorOptions().dtype(torch::kFloat32));
    torch::Tensor masks_cpu = torch::zeros(
        {count, action_dim}, torch::TensorOptions().dtype(torch::kBool));
    float* dense = dense_cpu.data_ptr<float>();
    bool* mask = masks_cpu.data_ptr<bool>();
    std::vector<bool> packed_row_valid(count, true);
    for (int64_t i = 0; i < count; ++i) {
      const PpoTransition& transition = batch[start + i];
      if (transition.behavior_raw_legal_logits.size() !=
          transition.legal_actions.size()) {
        packed_row_valid[i] = false;
        schema_errors->push_back(absl::StrFormat(
            "row %lld: raw/legal vector width mismatch",
            static_cast<long long>(start + i)));
      }
      std::unordered_set<Action> unique;
      for (size_t j = 0; j < transition.legal_actions.size(); ++j) {
        const Action action = transition.legal_actions[j];
        const bool valid = action >= 0 && action < action_dim &&
                           unique.insert(action).second &&
                           j < transition.behavior_raw_legal_logits.size() &&
                           std::isfinite(
                               transition.behavior_raw_legal_logits[j]);
        if (!valid) {
          packed_row_valid[i] = false;
          continue;
        }
        dense[i * action_dim + action] =
            transition.behavior_raw_legal_logits[j];
        mask[i * action_dim + action] = true;
      }
    }
    const torch::Tensor mask_popcounts =
        masks_cpu.sum(1).to(torch::kCPU).to(torch::kInt64).contiguous();
    const int64_t* popcount = mask_popcounts.data_ptr<int64_t>();
    for (int64_t i = 0; i < count; ++i) {
      if (popcount[i] == static_cast<int64_t>(
                             batch[start + i].legal_actions.size())) {
        if (dense_mask_rows_ok != nullptr) ++*dense_mask_rows_ok;
      } else {
        packed_row_valid[i] = false;
        schema_errors->push_back(absl::StrFormat(
            "row %lld: dense mask popcount %lld != legal count %lld",
            static_cast<long long>(start + i),
            static_cast<long long>(popcount[i]),
            static_cast<long long>(
                batch[start + i].legal_actions.size())));
      }
    }

    const torch::Tensor dense_bf16 =
        dense_cpu.to(device).to(torch::kBFloat16);
    const torch::Tensor masks = masks_cpu.to(device);
    torch::Tensor replay_log_probs;
    {
      AutocastGuard autocast_guard(c10::DeviceType::CUDA, true);
      const torch::Tensor centered =
          CenterAndCapLogitsTensor(dense_bf16, masks, logit_cap);
      const torch::Tensor masked_logits =
          centered.masked_fill(masks.logical_not(), -1e9f);
      replay_log_probs = torch::log_softmax(masked_logits, -1);
    }
    const torch::Tensor log_probs_cpu =
        replay_log_probs.to(torch::kCPU).to(torch::kFloat32).contiguous();
    const float* new_log_probs = log_probs_cpu.data_ptr<float>();

    for (int64_t i = 0; i < count; ++i) {
      const int64_t row_index = start + i;
      const PpoTransition& transition = batch[row_index];
      const std::string identity =
          PpoParityRowIdentitySha256(row_index, transition);
      PpoNumericalParityInput input;
      input.legal_count = static_cast<int>(transition.legal_actions.size());
      input.decision_role = transition.decision_role;
      input.advantage = transition.advantage;
      input.stored_chosen_log_prob = transition.old_log_prob;
      for (int j = 0; j < input.legal_count; ++j) {
        const Action action = transition.legal_actions[j];
        input.old_log_probs.push_back(
            j < static_cast<int>(transition.behavior_legal_log_probs.size())
                ? transition.behavior_legal_log_probs[j]
                : std::numeric_limits<double>::quiet_NaN());
        input.new_log_probs.push_back(
            (action >= 0 && action < action_dim)
                ? new_log_probs[i * action_dim + action]
                : std::numeric_limits<double>::quiet_NaN());
        if (action == transition.action) input.chosen_index = j;
      }
      PpoNumericalParityRow row;
      std::string error;
      if (!ComputePpoNumericalParityRow(input, &row, &error)) {
        schema_errors->push_back(absl::StrFormat(
            "row %lld: %s", static_cast<long long>(row_index), error));
      }
      if (!packed_row_valid[i]) {
        row.schema_errors = std::max<int64_t>(1, row.schema_errors);
        row.nonfinite_values = std::max<int64_t>(1, row.nonfinite_values);
      }
      row.row_identity_sha256 = identity;
      rows.push_back(row);
      evidence << identity;
      evidence.put('\0');
      for (int j = 0; j < input.legal_count; ++j) {
        evidence.write(reinterpret_cast<const char*>(&input.new_log_probs[j]),
                       sizeof(input.new_log_probs[j]));
      }
    }
  }
  if (row_evidence_sha256 != nullptr) {
    *row_evidence_sha256 = ComputeStringSHA256(evidence.str());
  }
  return rows;
}

json::Object PpoNumericalParitySummaryJson(
    const PpoNumericalParitySummary& s) {
  json::Object out;
  out["rows"] = s.rows;
  out["finite_rows"] = s.finite_rows;
  out["legal_logits"] = s.legal_logits;
  out["old_probability_underflows"] = s.old_probability_underflows;
  out["new_probability_underflows"] = s.new_probability_underflows;
  out["nonfinite_values"] = s.nonfinite_values;
  out["schema_error_rows"] = s.schema_error_rows;
  out["mass_residual_rows"] = s.mass_residual_rows;
  out["mean_old_raw_mass_residual"] = s.mean_old_raw_mass_residual;
  out["max_old_raw_mass_residual"] = s.max_old_raw_mass_residual;
  out["mean_new_raw_mass_residual"] = s.mean_new_raw_mass_residual;
  out["max_new_raw_mass_residual"] = s.max_new_raw_mass_residual;
  out["mean_abs_chosen_log_prob_delta"] =
      s.mean_abs_chosen_log_prob_delta;
  out["max_abs_chosen_log_prob_delta"] =
      s.max_abs_chosen_log_prob_delta;
  out["mean_full_kl_old_new"] = s.mean_kl_old_new;
  out["max_full_kl_old_new"] = s.max_kl_old_new;
  out["mean_full_kl_new_old"] = s.mean_kl_new_old;
  out["max_full_kl_new_old"] = s.max_kl_new_old;
  out["ratio_min"] = s.ratio_min;
  out["ratio_p01"] = s.ratio_p01;
  out["ratio_p50"] = s.ratio_p50;
  out["ratio_p99"] = s.ratio_p99;
  out["ratio_max"] = s.ratio_max;
  out["ratio_lt_0_8"] = s.ratio_lt_0_8;
  out["ratio_gt_1_2"] = s.ratio_gt_1_2;
  return out;
}

template <typename Predicate>
json::Object PpoNumericalParitySubsetJson(
    const std::vector<PpoNumericalParityRow>& rows, Predicate predicate) {
  std::vector<PpoNumericalParityRow> subset;
  for (const auto& row : rows) {
    if (predicate(row)) subset.push_back(row);
  }
  return PpoNumericalParitySummaryJson(
      SummarizePpoNumericalParityRows(subset));
}

std::string PpoDecisionRoleName(int role) {
  switch (static_cast<DuneDecisionRole>(role)) {
    case DuneDecisionRole::kForcedOrBookkeeping: return "forced_or_bookkeeping";
    case DuneDecisionRole::kLeaderSelection: return "leader_selection";
    case DuneDecisionRole::kAgentPrimary: return "agent_primary";
    case DuneDecisionRole::kAgentContinuation: return "agent_continuation";
    case DuneDecisionRole::kPurchase: return "purchase";
    case DuneDecisionRole::kCombatIntrigue: return "combat_intrigue";
    case DuneDecisionRole::kOtherOptional: return "other_optional";
  }
  return "invalid_role_" + std::to_string(role);
}

json::Object PpoNumericalParitySplitsJson(
    const std::vector<PpoNumericalParityRow>& rows) {
  json::Object legal_splits;
  std::set<int> legal_counts;
  for (const auto& row : rows) legal_counts.insert(row.legal_count);
  for (int legal_count : legal_counts) {
    legal_splits[std::to_string(legal_count)] = json::Value(
        PpoNumericalParitySubsetJson(rows, [=](const auto& row) {
          return row.legal_count == legal_count;
        }));
  }
  json::Object probability_splits;
  for (const std::string& bucket : {
           "p_lt_1e-6", "p_1e-6_to_1e-4", "p_1e-4_to_1e-2",
           "p_1e-2_to_0_1", "p_0_1_to_0_5", "p_ge_0_5", "invalid"}) {
    probability_splits[bucket] = json::Value(PpoNumericalParitySubsetJson(
        rows, [&](const auto& row) {
          return PpoNumericalParityOldProbabilityBucket(row) == bucket;
        }));
  }
  json::Object advantage_splits;
  advantage_splits["negative"] = json::Value(PpoNumericalParitySubsetJson(
      rows, [](const auto& r) { return r.advantage < 0.0; }));
  advantage_splits["zero"] = json::Value(PpoNumericalParitySubsetJson(
      rows, [](const auto& r) { return r.advantage == 0.0; }));
  advantage_splits["positive"] = json::Value(PpoNumericalParitySubsetJson(
      rows, [](const auto& r) { return r.advantage > 0.0; }));
  advantage_splits["nonfinite"] = json::Value(PpoNumericalParitySubsetJson(
      rows, [](const auto& r) { return !std::isfinite(r.advantage); }));
  json::Object role_splits;
  for (int role = 0; role < 7; ++role) {
    role_splits[PpoDecisionRoleName(role)] = json::Value(
        PpoNumericalParitySubsetJson(rows, [=](const auto& row) {
          return row.decision_role == role;
        }));
  }
  role_splits["invalid"] = json::Value(PpoNumericalParitySubsetJson(
      rows, [](const auto& row) {
        return row.decision_role < 0 || row.decision_role >= 7;
      }));
  json::Object splits;
  splits["legal_count_exact"] = json::Value(legal_splits);
  splits["old_chosen_probability"] = json::Value(probability_splits);
  splits["advantage_sign"] = json::Value(advantage_splits);
  splits["decision_role"] = json::Value(role_splits);
  return splits;
}

PpoParityViolationMap BuildPpoParityViolationMap(
    const std::vector<PpoNumericalParityRow>& rows) {
  PpoParityViolationMap violations;
  for (const std::string& name : {
           "schema_errors", "probability_underflow", "nonfinite_values",
           "raw_mass_residual", "chosen_logprob_delta", "full_legal_kl",
           "ratio_min", "ratio_max"}) {
    violations[name] = {};
  }
  const double mass_bound =
      absl::GetFlag(FLAGS_numerical_parity_max_raw_mass_residual);
  const double delta_bound =
      absl::GetFlag(FLAGS_numerical_parity_max_abs_logprob_delta);
  const double kl_bound =
      absl::GetFlag(FLAGS_numerical_parity_max_full_kl);
  const double ratio_min = absl::GetFlag(FLAGS_numerical_parity_min_ratio);
  const double ratio_max = absl::GetFlag(FLAGS_numerical_parity_max_ratio);
  for (const auto& row : rows) {
    const std::string& identity = row.row_identity_sha256;
    if (row.schema_errors != 0) violations["schema_errors"].push_back(identity);
    if (row.old_probability_underflows != 0 ||
        row.new_probability_underflows != 0) {
      violations["probability_underflow"].push_back(identity);
    }
    if (row.nonfinite_values != 0) {
      violations["nonfinite_values"].push_back(identity);
    }
    if (!row.mass_residual_valid ||
        row.old_raw_mass_residual > mass_bound ||
        row.new_raw_mass_residual > mass_bound) {
      violations["raw_mass_residual"].push_back(identity);
    }
    if (std::abs(row.chosen_log_prob_delta) > delta_bound) {
      violations["chosen_logprob_delta"].push_back(identity);
    }
    if (row.kl_old_new > kl_bound || row.kl_new_old > kl_bound) {
      violations["full_legal_kl"].push_back(identity);
    }
    if (row.ratio < ratio_min) violations["ratio_min"].push_back(identity);
    if (row.ratio > ratio_max) violations["ratio_max"].push_back(identity);
  }
  return violations;
}

bool PpoParityViolationMapPasses(const PpoParityViolationMap& violations) {
  for (const auto& item : violations) {
    if (!item.second.empty()) return false;
  }
  return true;
}

json::Object PpoParityViolationSetJson(
    const std::vector<std::string>& identities) {
  std::string payload;
  std::string error;
  if (!CanonicalPpoParityViolationIdentityPayload(
          identities, &payload, &error)) {
    SpielFatalError("Violation identity set is malformed: " + error);
  }
  json::Object out;
  out["count"] = static_cast<int64_t>(identities.size());
  out["row_identity_set_sha256"] = ComputeStringSHA256(payload);
  return out;
}

json::Object PpoParityViolationMapJson(
    const PpoParityViolationMap& violations) {
  json::Object out;
  for (const auto& item : violations) {
    out[item.first] = json::Value(PpoParityViolationSetJson(item.second));
  }
  return out;
}

json::Object PpoParityViolationIntersectionsJson(
    const PpoParityViolationMap& first,
    const PpoParityViolationMap& second) {
  json::Object out;
  for (const auto& item : first) {
    const auto found = second.find(item.first);
    if (found == second.end()) {
      SpielFatalError("Violation maps have different threshold names");
    }
    const std::vector<std::string> intersection =
        IntersectPpoParityViolationIdentities(item.second, found->second);
    out[item.first] = json::Value(PpoParityViolationSetJson(intersection));
  }
  return out;
}


#endif  // OPEN_SPIEL_BUILD_WITH_LIBTORCH

}  // namespace open_spiel::ppo_train_internal
