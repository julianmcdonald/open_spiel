#include "dune_ppo_train_internal.h"

namespace open_spiel::ppo_train_internal {

#ifdef OPEN_SPIEL_BUILD_WITH_LIBTORCH

PpoNumericalParitySourceProvenance LoadPpoNumericalParitySourceProvenance(
    const std::string& source_root, const std::string& registered_sha256) {
  std::error_code ec;
  const std::filesystem::path canonical_root =
      std::filesystem::canonical(source_root, ec);
  if (ec || !std::filesystem::is_directory(canonical_root)) {
    SpielFatalError(
        "Numerical parity source root is not a readable directory: " +
        source_root);
  }
  PpoNumericalParitySourceProvenance out;
  out.root = canonical_root.string();
  std::vector<std::pair<std::string, std::string>> digest_records;
  for (const std::string& relative :
       PpoNumericalParitySourceRelativePaths()) {
    const std::filesystem::path path = canonical_root / relative;
    if (!std::filesystem::is_regular_file(path)) {
      SpielFatalError(
          "Numerical parity required source is not a regular file: " +
          path.string());
    }
    size_t size = 0;
    const std::string digest = ComputeFileSHA256(path.string(), &size);
    if (digest.empty()) {
      SpielFatalError("Numerical parity could not hash required source: " +
                      path.string());
    }
    out.files.push_back({relative, path.string(),
                         static_cast<int64_t>(size), digest});
    digest_records.push_back({relative, digest});
  }
  std::string payload;
  std::string payload_error;
  if (!CanonicalPpoNumericalParitySourcePayload(
          digest_records, &payload, &payload_error)) {
    SpielFatalError("Numerical parity source canonicalization failed: " +
                    payload_error);
  }
  out.combined_sha256 = ComputeStringSHA256(payload);
  if (out.combined_sha256 != registered_sha256) {
    SpielFatalError(
        "Numerical parity registered source SHA-256 mismatch: expected " +
        registered_sha256 + " computed " + out.combined_sha256);
  }
  return out;
}

PpoParityCellResult ReplayPpoInferenceExactGeometry(
    BatchedEvaluator* evaluator, const std::vector<PpoTransition>& batch,
    const PpoParityBatchGeometry& geometry, int64_t action_dim,
    float logit_cap, const EvaluatorStats& before,
    PpoParityReplayGeometryStats* stats) {
  PpoParityCellResult cell;
  cell.name = "live_inference_exact_original_batches";
  cell.module_train_mode = false;
  cell.autocast_enabled = false;
  cell.rows.resize(batch.size());
  stats->before = before;
  std::stringstream evidence, captured_raw, replay_raw, replay_policy_bytes;
  evidence << "dune_raw_ppo_numerical_parity_v4";
  evidence.put('\0');
  evidence << cell.name;
  evidence.put('\0');
  for (size_t group_index = 0;
       group_index < geometry.row_indices_by_group.size(); ++group_index) {
    const auto& indices = geometry.row_indices_by_group[group_index];
    std::vector<std::vector<float>> observations;
    observations.reserve(indices.size());
    for (size_t index : indices) observations.push_back(batch[index].state);
    const std::vector<EvalResult> replay = evaluator->EvaluateBatch(observations);
    ++stats->model_forward_calls;
    if (replay.size() != indices.size()) {
      cell.schema_errors.push_back("R returned row count differs from group");
      continue;
    }
    for (size_t position = 0; position < indices.size(); ++position) {
      const size_t index = indices[position];
      const PpoTransition& transition = batch[index];
      const EvalResult& result = replay[position];
      const std::string identity =
          PpoParityRowIdentitySha256(static_cast<int64_t>(index), transition);
      bool valid = true;
      auto error = [&](const std::string& message) {
        cell.schema_errors.push_back("row " + std::to_string(index) +
                                     ": " + message);
        valid = false;
      };
      const int64_t expected_replay_batch_id =
          static_cast<int64_t>(before.batches + group_index);
      if (result.physical_batch_id == expected_replay_batch_id &&
          result.physical_batch_size == static_cast<int32_t>(indices.size()) &&
          result.physical_batch_row == static_cast<int32_t>(position)) {
        ++stats->returned_metadata_exact_rows;
      } else {
        error("returned replay batch metadata differs from exact group");
      }
      if (static_cast<int64_t>(result.logits.size()) != action_dim ||
          !std::all_of(result.logits.begin(), result.logits.end(),
                       [](float value) { return std::isfinite(value); })) {
        error("returned full logits have wrong width or nonfinite value");
      }
      std::vector<float> replay_raw_legal;
      replay_raw_legal.reserve(transition.legal_actions.size());
      for (Action action : transition.legal_actions) {
        replay_raw_legal.push_back(
            action >= 0 && action < action_dim &&
                    action < static_cast<Action>(result.logits.size())
                ? result.logits[action]
                : std::numeric_limits<float>::quiet_NaN());
      }
      bool raw_exact = replay_raw_legal.size() ==
                       transition.behavior_raw_legal_logits.size();
      if (raw_exact) {
        for (size_t j = 0; j < replay_raw_legal.size(); ++j) {
          if (PpoParityFloatBits(replay_raw_legal[j]) != PpoParityFloatBits(
                  transition.behavior_raw_legal_logits[j])) {
            raw_exact = false;
            break;
          }
        }
      }
      if (raw_exact) ++stats->raw_legal_exact_rows;
      else error("returned raw legal logits are not bit-exact captured logits");
      std::vector<float> replay_policy;
      std::string recompute_error;
      if (!RecomputePpoBehaviorLegalLogProbs(
              replay_raw_legal, logit_cap, &replay_policy,
              &recompute_error)) {
        error("returned-logit CPU policy recompute failed: " + recompute_error);
      }
      bool policy_exact = replay_policy.size() ==
                          transition.behavior_legal_log_probs.size();
      if (policy_exact) {
        for (size_t j = 0; j < replay_policy.size(); ++j) {
          if (PpoParityFloatBits(replay_policy[j]) != PpoParityFloatBits(
                  transition.behavior_legal_log_probs[j])) {
            policy_exact = false;
            break;
          }
        }
      }
      if (policy_exact) ++stats->cpu_policy_exact_rows;
      else error("returned-logit CPU policy is not bit-exact captured policy");

      PpoNumericalParityInput input;
      input.legal_count = static_cast<int>(transition.legal_actions.size());
      input.decision_role = transition.decision_role;
      input.advantage = transition.advantage;
      input.stored_chosen_log_prob = transition.old_log_prob;
      for (size_t j = 0; j < transition.legal_actions.size(); ++j) {
        input.old_log_probs.push_back(
            j < transition.behavior_legal_log_probs.size()
                ? transition.behavior_legal_log_probs[j]
                : std::numeric_limits<double>::quiet_NaN());
        input.new_log_probs.push_back(
            j < replay_policy.size()
                ? replay_policy[j]
                : std::numeric_limits<double>::quiet_NaN());
        if (transition.legal_actions[j] == transition.action) {
          input.chosen_index = static_cast<int>(j);
        }
        const float captured =
            j < transition.behavior_raw_legal_logits.size()
                ? transition.behavior_raw_legal_logits[j]
                : std::numeric_limits<float>::quiet_NaN();
        const float returned =
            j < replay_raw_legal.size()
                ? replay_raw_legal[j]
                : std::numeric_limits<float>::quiet_NaN();
        captured_raw.write(reinterpret_cast<const char*>(&captured),
                           sizeof(captured));
        replay_raw.write(reinterpret_cast<const char*>(&returned),
                         sizeof(returned));
        const float replay_policy_value =
            j < replay_policy.size()
                ? replay_policy[j]
                : std::numeric_limits<float>::quiet_NaN();
        replay_policy_bytes.write(
            reinterpret_cast<const char*>(&replay_policy_value),
            sizeof(replay_policy_value));
      }
      PpoNumericalParityRow row;
      std::string row_error;
      if (!ComputePpoNumericalParityRow(input, &row, &row_error)) {
        error("R parity row failed: " + row_error);
      }
      if (!valid) {
        row.schema_errors = std::max<int64_t>(1, row.schema_errors);
        row.nonfinite_values = std::max<int64_t>(1, row.nonfinite_values);
      }
      row.row_identity_sha256 = identity;
      cell.rows[index] = row;
      evidence << identity;
      evidence.put('\0');
      for (float value : replay_raw_legal) {
        evidence.write(reinterpret_cast<const char*>(&value), sizeof(value));
      }
    }
  }
  stats->after = evaluator->GetStats();
  stats->captured_raw_sha256 = ComputeStringSHA256(captured_raw.str());
  stats->replay_raw_sha256 = ComputeStringSHA256(replay_raw.str());
  cell.model_forward_calls = stats->model_forward_calls;
  cell.row_evidence_sha256 = ComputeStringSHA256(evidence.str());
  cell.raw_logit_sha256 = stats->replay_raw_sha256;
  cell.policy_sha256 = ComputeStringSHA256(replay_policy_bytes.str());
  return cell;
}

PpoParityCellResult ReplayPpoTrainingModelOriginalGeometry(
    const std::shared_ptr<SharedDunePolicyValueNetImpl>& model,
    const std::vector<PpoTransition>& batch,
    const PpoParityBatchGeometry& geometry, int64_t obs_size,
    int64_t action_dim, torch::Device device, float logit_cap,
    bool learner_autocast,
    PpoParityOriginalGeometryModelStats* stats) {
  PpoParityCellResult cell;
  cell.name = "training_model_exact_original_batches";
  cell.module_train_mode = true;
  cell.autocast_enabled = learner_autocast;
  cell.rows.resize(batch.size());
  std::stringstream evidence, raw_bytes, policy_bytes;
  evidence << "dune_raw_ppo_numerical_parity_v4";
  evidence.put('\0');
  evidence << cell.name;
  evidence.put('\0');
  const bool was_training = model->is_training();
  model->train();
  torch::NoGradGuard no_grad;
  for (const auto& indices : geometry.row_indices_by_group) {
    const int64_t count = static_cast<int64_t>(indices.size());
    torch::Tensor states_cpu = torch::zeros(
        {count, obs_size}, torch::TensorOptions().dtype(torch::kFloat32));
    torch::Tensor masks_cpu = torch::zeros(
        {count, action_dim}, torch::TensorOptions().dtype(torch::kBool));
    float* state_data = states_cpu.data_ptr<float>();
    bool* mask_data = masks_cpu.data_ptr<bool>();
    std::vector<bool> valid(count, true);
    for (int64_t i = 0; i < count; ++i) {
      const PpoTransition& transition = batch[indices[i]];
      if (static_cast<int64_t>(transition.state.size()) != obs_size) {
        valid[i] = false;
        cell.schema_errors.push_back("D observation width mismatch");
      } else {
        std::memcpy(state_data + i * obs_size, transition.state.data(),
                    obs_size * sizeof(float));
      }
      for (Action action : transition.legal_actions) {
        if (action >= 0 && action < action_dim) {
          mask_data[i * action_dim + action] = true;
        }
      }
    }
    const torch::Tensor popcounts =
        masks_cpu.sum(1).to(torch::kCPU).to(torch::kInt64).contiguous();
    const int64_t* popcount = popcounts.data_ptr<int64_t>();
    for (int64_t i = 0; i < count; ++i) {
      if (popcount[i] == static_cast<int64_t>(
                             batch[indices[i]].legal_actions.size())) {
        ++cell.dense_mask_rows_ok;
      } else {
        valid[i] = false;
        cell.schema_errors.push_back("D dense-mask popcount mismatch");
      }
    }
    const torch::Tensor states = states_cpu.to(device);
    const torch::Tensor masks = masks_cpu.to(device);
    if (states_cpu.scalar_type() != torch::kFloat32 ||
        states.scalar_type() != torch::kFloat32) {
      cell.schema_errors.push_back(
          "D input/pre-forward dtype is not Float32");
    }
    torch::Tensor raw_logits, replay_log_probs;
    auto compute = [&]() {
      auto outputs = model->forward(states);
      if (model->with_semantic_scorer_ && model->semantic_scorer_) {
        std::vector<const dune_semantic::CandidateActionData*> cands;
        cands.reserve(count);
        for (int64_t i = 0; i < count; ++i) {
          cands.push_back(&batch[indices[i]].candidate_data);
        }
        dune_semantic::ApplySemanticScorerBatch(
            model->semantic_scorer_, outputs.trunk, cands, outputs.logits, device);
      }
      raw_logits = outputs.logits;
      const torch::Tensor centered =
          CenterAndCapLogitsTensor(raw_logits, masks, logit_cap);
      replay_log_probs = torch::log_softmax(
          centered.masked_fill(masks.logical_not(), -1e9f), -1);
    };
    if (learner_autocast) {
      AutocastGuard autocast_guard(c10::DeviceType::CUDA, true);
      compute();
    } else {
      compute();
    }
    ++cell.model_forward_calls;
    const torch::Tensor raw_cpu =
        raw_logits.to(torch::kCPU).to(torch::kFloat32).contiguous();
    const torch::Tensor log_probs_cpu =
        replay_log_probs.to(torch::kCPU).to(torch::kFloat32).contiguous();
    const float* raw_data = raw_cpu.data_ptr<float>();
    const float* logprob_data = log_probs_cpu.data_ptr<float>();
    for (int64_t i = 0; i < count; ++i) {
      const size_t index = indices[i];
      const PpoTransition& transition = batch[index];
      const std::string identity =
          PpoParityRowIdentitySha256(static_cast<int64_t>(index), transition);
      bool raw_exact = transition.behavior_raw_legal_logits.size() ==
                       transition.legal_actions.size();
      PpoNumericalParityInput input;
      input.legal_count = static_cast<int>(transition.legal_actions.size());
      input.decision_role = transition.decision_role;
      input.advantage = transition.advantage;
      input.stored_chosen_log_prob = transition.old_log_prob;
      for (size_t j = 0; j < transition.legal_actions.size(); ++j) {
        const Action action = transition.legal_actions[j];
        const float replay_raw =
            action >= 0 && action < action_dim
                ? raw_data[i * action_dim + action]
                : std::numeric_limits<float>::quiet_NaN();
        const float captured_raw =
            j < transition.behavior_raw_legal_logits.size()
                ? transition.behavior_raw_legal_logits[j]
                : std::numeric_limits<float>::quiet_NaN();
        if (PpoParityFloatBits(replay_raw) !=
            PpoParityFloatBits(captured_raw)) {
          raw_exact = false;
        }
        if (std::isfinite(replay_raw) && std::isfinite(captured_raw)) {
          stats->raw_legal_max_abs_delta = std::max(
              stats->raw_legal_max_abs_delta,
              std::abs(static_cast<double>(replay_raw) - captured_raw));
        }
        input.old_log_probs.push_back(
            j < transition.behavior_legal_log_probs.size()
                ? transition.behavior_legal_log_probs[j]
                : std::numeric_limits<double>::quiet_NaN());
        input.new_log_probs.push_back(
            action >= 0 && action < action_dim
                ? logprob_data[i * action_dim + action]
                : std::numeric_limits<double>::quiet_NaN());
        if (action == transition.action) input.chosen_index = j;
        evidence.write(reinterpret_cast<const char*>(&replay_raw),
                       sizeof(replay_raw));
        raw_bytes.write(reinterpret_cast<const char*>(&replay_raw),
                        sizeof(replay_raw));
        const float replay_policy = static_cast<float>(
            input.new_log_probs.back());
        policy_bytes.write(reinterpret_cast<const char*>(&replay_policy),
                           sizeof(replay_policy));
      }
      if (raw_exact) ++stats->raw_legal_exact_rows;
      PpoNumericalParityRow row;
      std::string error;
      if (!ComputePpoNumericalParityRow(input, &row, &error)) {
        cell.schema_errors.push_back("row " + std::to_string(index) +
                                     ": " + error);
        valid[i] = false;
      }
      if (!valid[i]) {
        row.schema_errors = std::max<int64_t>(1, row.schema_errors);
        row.nonfinite_values = std::max<int64_t>(1, row.nonfinite_values);
      }
      row.row_identity_sha256 = identity;
      cell.rows[index] = row;
      evidence << identity;
      evidence.put('\0');
    }
  }
  if (was_training) model->train(); else model->eval();
  cell.row_evidence_sha256 = ComputeStringSHA256(evidence.str());
  cell.raw_logit_sha256 = ComputeStringSHA256(raw_bytes.str());
  cell.policy_sha256 = ComputeStringSHA256(policy_bytes.str());
  return cell;
}

bool WritePpoNumericalParityArtifact(
    const std::string& output_path,
    const std::vector<PpoNumericalParityRow>& primary_rows,
    const std::vector<std::string>& primary_schema_errors,
    const std::string& primary_row_evidence_sha256,
    const std::vector<PpoNumericalParityRow>& control_rows,
    const std::vector<std::string>& control_schema_errors,
    const std::string& control_row_evidence_sha256,
    const CollectResult& collect,
    const std::string& model_hash_before,
    const std::string& model_hash_after,
    const std::string& inference_hash_before,
    const std::string& inference_hash_after,
    const std::string& source_manifest,
    const std::string& source_manifest_sha256, int64_t source_global_update,
    const std::string& source_checkpoint_uuid,
    const std::string& config_fingerprint, torch::Device device,
    int64_t obs_size, int64_t action_size, const std::string& command_line,
    const PpoNumericalParitySourceProvenance& source_provenance) {
  const PpoNumericalParitySummary primary =
      SummarizePpoNumericalParityRows(primary_rows);
  const PpoNumericalParitySummary control =
      SummarizePpoNumericalParityRows(control_rows);
  int64_t nontrivial_rows = 0;
  for (const auto& row : primary_rows) {
    if (row.legal_count > 1) ++nontrivial_rows;
  }

  json::Array failures;
  auto reject = [&](const std::string& reason) { failures.emplace_back(reason); };
  if (!primary_schema_errors.empty()) {
    reject(absl::StrFormat("%d malformed parity rows",
                           static_cast<int>(primary_schema_errors.size())));
  }
  if (primary.rows != static_cast<int64_t>(collect.rollout.size())) {
    reject("primary replay row count does not match collected transitions");
  }
  if (primary.mass_residual_rows != primary.rows) {
    reject("primary raw mass residual coverage is incomplete");
  }
  if (!collect.episode_ids_unique) reject("episode IDs are not unique");
  if (primary.rows < absl::GetFlag(FLAGS_numerical_parity_min_rows)) {
    reject("total row floor not met");
  }
  if (nontrivial_rows <
      absl::GetFlag(FLAGS_numerical_parity_min_nontrivial_rows)) {
    reject("nontrivial row floor not met");
  }
  if (primary.nonfinite_values != 0) reject("nonfinite parity values observed");
  if (primary.old_probability_underflows != 0) {
    reject("rollout behavior probability underflow observed");
  }
  if (primary.new_probability_underflows != 0) {
    reject("learner autocast probability underflow observed");
  }
  const double max_raw_mass_residual =
      absl::GetFlag(FLAGS_numerical_parity_max_raw_mass_residual);
  if (primary.mass_residual_rows == primary.rows &&
      !PpoNumericalParityRawMassWithinBound(primary,
                                             max_raw_mass_residual)) {
    reject("raw legal probability mass residual exceeds bound");
  }
  if (primary.max_abs_chosen_log_prob_delta >
      absl::GetFlag(FLAGS_numerical_parity_max_abs_logprob_delta)) {
    reject("chosen-action log-probability delta exceeds bound");
  }
  if (primary.max_kl_old_new >
          absl::GetFlag(FLAGS_numerical_parity_max_full_kl) ||
      primary.max_kl_new_old >
          absl::GetFlag(FLAGS_numerical_parity_max_full_kl)) {
    reject("full legal categorical KL exceeds bound");
  }
  if (primary.ratio_min <
      absl::GetFlag(FLAGS_numerical_parity_min_ratio)) {
    reject("chosen-action ratio minimum is below bound");
  }
  if (primary.ratio_max >
      absl::GetFlag(FLAGS_numerical_parity_max_ratio)) {
    reject("chosen-action ratio maximum is above bound");
  }
  if (model_hash_before != model_hash_after) {
    reject("model state changed during read-only diagnostic");
  }
  if (inference_hash_before != inference_hash_after) {
    reject("rollout inference model changed during read-only diagnostic");
  }
  if (model_hash_before != inference_hash_before) {
    reject("learner and rollout inference models differed before collection");
  }

  json::Object root;
  root["schema"] = "dune_raw_ppo_numerical_parity_v2";
  root["epistemic_label"] =
      "no_training_actual_learner_autocast_parity_gate_with_fp32_control";
  root["measurement_intent"] =
      "Gate unchanged-weight parity on the historical production learner "
      "CUDA BF16 autocast policy path; retain CUDA FP32 replay as a "
      "descriptive cross-precision control only.";
  root["prior_v1_disposition"] =
      "The prior v1 cross-precision REJECT is a separate valid artifact; v2 "
      "does not overwrite or reinterpret it.";
  root["gate_cell"] = "actual_learner_cuda_bf16_autocast";
  root["control_cell"] = "descriptive_cuda_fp32_no_autocast";
  root["control_can_affect_status"] = false;
  root["registration_id"] =
      absl::GetFlag(FLAGS_numerical_parity_registration_id);
  root["scope"] = "raw_ppo_only";
  root["optimizer_steps"] = int64_t{0};
  root["optimizer_checkpoint_loaded"] = false;
  root["command_line"] = command_line;
  root["config_fingerprint"] = config_fingerprint;
  root["device"] = device.str();
  root["rollout_precision"] =
      device.is_cuda() ? "cuda_bf16_autocast" : "cpu_fp32";
  root["primary_replay_precision"] = "cuda_bf16_autocast";
  root["descriptive_control_replay_precision"] =
      "cuda_fp32_no_autocast_tf32_allowed";
  root["train_amp"] = absl::GetFlag(FLAGS_train_amp);
  root["ppo_minibatch_size"] = static_cast<int64_t>(
      absl::GetFlag(FLAGS_ppo_minibatch_size));
  root["logit_cap"] = absl::GetFlag(FLAGS_logit_cap);
  root["seed"] = static_cast<int64_t>(absl::GetFlag(FLAGS_seed));
  root["threads"] = static_cast<int64_t>(absl::GetFlag(FLAGS_threads));
  root["eval_batch_size"] =
      static_cast<int64_t>(absl::GetFlag(FLAGS_eval_batch_size));
  root["eval_timeout_ms"] =
      static_cast<int64_t>(absl::GetFlag(FLAGS_eval_timeout_ms));
  root["rollout_games"] =
      static_cast<int64_t>(absl::GetFlag(FLAGS_rollout_games));
  root["hard_transition_ceiling"] = static_cast<int64_t>(
      absl::GetFlag(FLAGS_rollout_games)) * 5000;
  root["games_collected"] = static_cast<int64_t>(collect.games);
  root["transitions_collected"] =
      static_cast<int64_t>(collect.rollout.size());
  root["nontrivial_rows"] = nontrivial_rows;
  root["observation_size"] = obs_size;
  root["action_size"] = action_size;
  root["rollout_hash"] = ComputeRolloutHash(collect.rollout);
  root["model_state_sha256_before"] = model_hash_before;
  root["model_state_sha256_after"] = model_hash_after;
  root["inference_state_sha256_before"] = inference_hash_before;
  root["inference_state_sha256_after"] = inference_hash_after;

  size_t model_file_size = 0;
  const std::string model_path = absl::GetFlag(FLAGS_model_checkpoint);
  json::Object source;
  source["model_path"] = std::filesystem::absolute(model_path).string();
  source["model_size"] = static_cast<int64_t>(model_file_size);
  source["model_sha256"] =
      ComputeFileSHA256(model_path, &model_file_size);
  source["model_size"] = static_cast<int64_t>(model_file_size);
  source["manifest_path"] = std::filesystem::absolute(source_manifest).string();
  source["manifest_sha256"] = source_manifest_sha256;
  source["global_update"] = source_global_update;
  source["checkpoint_uuid"] = source_checkpoint_uuid;
  size_t binary_size = 0;
  std::error_code ec;
  const std::filesystem::path binary_path =
      std::filesystem::read_symlink("/proc/self/exe", ec);
  if (ec || binary_path.empty()) {
    reject("could not resolve executed binary path");
  } else {
    source["binary_path"] = binary_path.string();
    source["binary_sha256"] =
        ComputeFileSHA256(binary_path.string(), &binary_size);
    source["binary_size"] = static_cast<int64_t>(binary_size);
  }
  root["source"] = json::Value(source);

  json::Object source_code;
  source_code["root"] = source_provenance.root;
  source_code["combined_sha256"] = source_provenance.combined_sha256;
  source_code["canonical_contract"] =
      "fixed ordered relative_path + NUL + lowercase file_sha256 + newline";
  json::Array source_files;
  for (const auto& file : source_provenance.files) {
    json::Object record;
    record["relative_path"] = file.relative_path;
    record["absolute_path"] = file.absolute_path;
    record["size"] = file.size;
    record["sha256"] = file.sha256;
    source_files.emplace_back(json::Value(record));
  }
  source_code["files"] = json::Value(source_files);
  root["source_code"] = json::Value(source_code);

  json::Object thresholds;
  thresholds["min_rows"] = static_cast<int64_t>(
      absl::GetFlag(FLAGS_numerical_parity_min_rows));
  thresholds["min_nontrivial_rows"] = static_cast<int64_t>(
      absl::GetFlag(FLAGS_numerical_parity_min_nontrivial_rows));
  thresholds["max_abs_chosen_logprob_delta"] =
      absl::GetFlag(FLAGS_numerical_parity_max_abs_logprob_delta);
  thresholds["max_full_legal_kl_each_direction"] =
      absl::GetFlag(FLAGS_numerical_parity_max_full_kl);
  thresholds["max_raw_mass_residual"] =
      absl::GetFlag(FLAGS_numerical_parity_max_raw_mass_residual);
  thresholds["min_ratio"] =
      absl::GetFlag(FLAGS_numerical_parity_min_ratio);
  thresholds["max_ratio"] =
      absl::GetFlag(FLAGS_numerical_parity_max_ratio);
  thresholds["require_zero_underflows"] = true;
  thresholds["require_zero_nonfinite"] = true;
  thresholds["require_unchanged_model_hash"] = true;
  root["thresholds"] = json::Value(thresholds);

  auto cell_json = [&](const std::string& label, bool gate_eligible,
                       const std::string& precision,
                       const std::vector<PpoNumericalParityRow>& cell_rows,
                       const PpoNumericalParitySummary& summary,
                       const std::vector<std::string>& cell_errors,
                       const std::string& evidence_sha256) {
    json::Object cell;
    cell["label"] = label;
    cell["gate_eligible"] = gate_eligible;
    cell["precision"] = precision;
    cell["train_amp"] = gate_eligible;
    cell["ppo_minibatch_size"] = static_cast<int64_t>(
        absl::GetFlag(FLAGS_ppo_minibatch_size));
    cell["row_evidence_sha256"] = evidence_sha256;
    cell["row_evidence_contract"] =
        "v2 cell label then ordered row_index,episode_id,player_id,"
        "chosen_action,decision_role,advantage,stored_chosen_logprob,then "
        "each aligned (legal_action,rollout_raw_logprob,replay_raw_logprob),"
        "native fixed-width bytes";
    cell["overall"] = json::Value(PpoNumericalParitySummaryJson(summary));
    cell["splits"] = json::Value(PpoNumericalParitySplitsJson(cell_rows));
    json::Array errors;
    for (const std::string& error : cell_errors) errors.emplace_back(error);
    cell["schema_errors"] = json::Value(errors);
    return cell;
  };
  json::Object cells;
  cells["actual_learner_cuda_bf16_autocast"] = json::Value(cell_json(
      "actual_learner_cuda_bf16_autocast", true, "cuda_bf16_autocast",
      primary_rows, primary, primary_schema_errors,
      primary_row_evidence_sha256));
  cells["descriptive_cuda_fp32_no_autocast"] = json::Value(cell_json(
      "descriptive_cuda_fp32_no_autocast", false,
      "cuda_fp32_no_autocast_tf32_allowed", control_rows, control,
      control_schema_errors, control_row_evidence_sha256));
  root["cells"] = json::Value(cells);
  root["failure_reasons"] = json::Value(failures);
  // Status is assigned LAST: every provenance check above is allowed to reject.
  root["status"] = failures.empty() ? "PASS" : "REJECT";

  const std::filesystem::path output(output_path);
  const std::filesystem::path tmp = output_path + ".tmp";
  if (std::filesystem::exists(output) || std::filesystem::exists(tmp)) {
    SpielFatalError("Numerical parity output already exists; refusing overwrite: " +
                    output_path);
  }
  if (!output.parent_path().empty()) {
    std::filesystem::create_directories(output.parent_path());
  }
  {
    std::ofstream out(tmp, std::ios::trunc);
    if (!out) SpielFatalError("Cannot create numerical parity temp artifact");
    out << json::ToString(root, true) << "\n";
    out.flush();
    if (!out) SpielFatalError("Numerical parity artifact write failed");
  }
  std::filesystem::rename(tmp, output);
  return failures.empty();
}


bool WritePpoNumericalParityArtifactV3(
    const std::string& output_path, const PpoParityCellResult& integrity_cell,
    const PpoParityIntegrityStats& integrity_stats,
    const PpoParityCellResult& learner_cell,
    const PpoParityCellResult& postprocess_cell,
    const CollectResult& collect, const std::string& shared_identity_sha256,
    const std::string& model_hash_before,
    const std::string& model_hash_after,
    const std::string& inference_hash_before,
    const std::string& inference_hash_after,
    const std::string& source_manifest,
    const std::string& source_manifest_sha256, int64_t source_global_update,
    const std::string& source_checkpoint_uuid,
    const std::string& config_fingerprint, torch::Device device,
    int64_t obs_size, int64_t action_size, const std::string& command_line,
    const PpoNumericalParitySourceProvenance& source_provenance) {
  const int64_t transitions = static_cast<int64_t>(collect.rollout.size());
  const PpoNumericalParitySummary a =
      SummarizePpoNumericalParityRows(integrity_cell.rows);
  const PpoNumericalParitySummary b =
      SummarizePpoNumericalParityRows(learner_cell.rows);
  const PpoNumericalParitySummary c =
      SummarizePpoNumericalParityRows(postprocess_cell.rows);
  auto lower_hex64 = [](const std::string& value) {
    if (value.size() != 64) return false;
    for (char ch : value) {
      if (!((ch >= '0' && ch <= '9') || (ch >= 'a' && ch <= 'f'))) {
        return false;
      }
    }
    return true;
  };
  std::vector<std::string> validity_errors = collect.parity_capture_errors;
  auto require = [&](bool condition, const std::string& message) {
    if (!condition) validity_errors.push_back(message);
  };
  require(collect.episode_ids_unique, "episode IDs are not unique");
  require(static_cast<int64_t>(collect.games) ==
              absl::GetFlag(FLAGS_rollout_games),
          "collected game count differs from exact rollout_games");
  require(transitions >= absl::GetFlag(FLAGS_numerical_parity_min_rows),
          "collected transition row floor is not met");
  int64_t nontrivial_rows = 0;
  for (const auto& row : integrity_cell.rows) {
    if (row.legal_count > 1) ++nontrivial_rows;
  }
  require(nontrivial_rows >=
              absl::GetFlag(FLAGS_numerical_parity_min_nontrivial_rows),
          "nontrivial transition row floor is not met");
  require(collect.parity_decisions == transitions,
          "capture decision count differs from transitions");
  require(collect.parity_full_width_ok == transitions,
          "not every evaluator output has full action width");
  require(collect.parity_full_finite_ok == transitions,
          "not every full evaluator output is finite");
  require(collect.parity_legal_ids_unique_in_range == transitions,
          "not every legal-ID vector is unique and in range");
  require(collect.parity_chosen_once == transitions,
          "not every chosen action occurs exactly once");
  require(a.rows == transitions && b.rows == transitions && c.rows == transitions,
          "A/B/C row counts do not equal collected transitions");
  require(a.schema_error_rows == 0 && integrity_cell.schema_errors.empty(),
          "cell A has schema errors");
  require(b.schema_error_rows == 0 && learner_cell.schema_errors.empty(),
          "cell B has schema errors");
  require(c.schema_error_rows == 0 && postprocess_cell.schema_errors.empty(),
          "cell C has schema errors");
  require(a.finite_rows == a.rows, "cell A finite rows do not equal rows");
  require(a.max_abs_chosen_log_prob_delta == 0.0 &&
              a.max_kl_old_new == 0.0 && a.max_kl_new_old == 0.0 &&
              a.ratio_min == 1.0 && a.ratio_max == 1.0,
          "cell A recompute metrics are not bit-exact identity");
  require(integrity_cell.model_forward_calls == 0,
          "cell A executed a model forward");
  require(postprocess_cell.model_forward_calls == 0,
          "cell C executed a model forward");
  require(learner_cell.model_forward_calls > 0,
          "cell B executed no model forward");
  require(learner_cell.dense_mask_rows_ok == transitions &&
              postprocess_cell.dense_mask_rows_ok == transitions,
          "B/C dense-mask popcount coverage is incomplete");
  require(integrity_stats.vector_width_rows_ok == transitions,
          "A vector-width coverage is incomplete");
  require(integrity_stats.behavior_vector_exact_rows == transitions,
          "A behavior-vector bit-exact coverage is incomplete");
  require(integrity_stats.chosen_scalar_exact_rows == transitions,
          "A chosen-scalar bit-exact coverage is incomplete");
  require(integrity_stats.raw_values_total == a.legal_logits,
          "A raw-value count differs from legal-logit count");
  require(integrity_stats.raw_values_total > 0,
          "A captured no raw legal-logit values");
  require(integrity_stats.raw_values_bf16_exact ==
              integrity_stats.raw_values_total,
          "captured raw legal logits do not all round-trip FP32-BF16-FP32 exactly");
  require(lower_hex64(shared_identity_sha256),
          "shared row-identity hash is malformed");
  require(lower_hex64(integrity_cell.row_evidence_sha256) &&
              lower_hex64(learner_cell.row_evidence_sha256) &&
              lower_hex64(postprocess_cell.row_evidence_sha256),
          "one or more cell evidence hashes are malformed");
  require(integrity_cell.row_evidence_sha256 !=
              learner_cell.row_evidence_sha256 &&
              learner_cell.row_evidence_sha256 !=
                  postprocess_cell.row_evidence_sha256 &&
              integrity_cell.row_evidence_sha256 !=
                  postprocess_cell.row_evidence_sha256,
          "cell evidence hashes are not distinct");
  bool shared_rows = integrity_cell.rows.size() == learner_cell.rows.size() &&
                     learner_cell.rows.size() == postprocess_cell.rows.size();
  if (shared_rows) {
    for (size_t i = 0; i < integrity_cell.rows.size(); ++i) {
      if (integrity_cell.rows[i].row_identity_sha256 !=
              learner_cell.rows[i].row_identity_sha256 ||
          learner_cell.rows[i].row_identity_sha256 !=
              postprocess_cell.rows[i].row_identity_sha256 ||
          !lower_hex64(integrity_cell.rows[i].row_identity_sha256)) {
        shared_rows = false;
        break;
      }
    }
  }
  require(shared_rows, "A/B/C row identities differ");
  require(model_hash_before == model_hash_after &&
              inference_hash_before == inference_hash_after &&
              model_hash_before == inference_hash_before,
          "model immutability or initial learner/inference equality failed");

  const bool instrument_valid = validity_errors.empty();
  const PpoParityViolationMap b_violations =
      BuildPpoParityViolationMap(learner_cell.rows);
  const PpoParityViolationMap c_violations =
      BuildPpoParityViolationMap(postprocess_cell.rows);
  const bool b_pass = instrument_valid &&
                      PpoParityViolationMapPasses(b_violations);
  const bool c_pass = instrument_valid &&
                      PpoParityViolationMapPasses(c_violations);
  const PpoParityV3Classification classification =
      ClassifyPpoParityV3(instrument_valid, b_pass, c_pass);

  json::Object root;
  root["schema"] = "dune_raw_ppo_numerical_parity_v3";
  root["status"] = instrument_valid ? "VALID" : "INVALID";
  root["classification"] = PpoParityV3ClassificationName(classification);
  root["training_authorized"] = false;
  root["registration_id"] =
      absl::GetFlag(FLAGS_numerical_parity_registration_id);
  root["scope"] = "raw_ppo_only";
  root["epistemic_label"] =
      "no_training_three_cell_rollout_forward_postprocess_causal_diagnostic";
  root["optimizer_steps"] = int64_t{0};
  root["optimizer_checkpoint_loaded"] = false;
  root["backward_calls"] = int64_t{0};
  root["training_updates"] = int64_t{0};
  root["command_line"] = command_line;
  root["config_fingerprint"] = config_fingerprint;
  root["device"] = device.str();
  root["rollout_amp"] = absl::GetFlag(FLAGS_rollout_amp);
  root["train_amp"] = absl::GetFlag(FLAGS_train_amp);
  root["allow_tf32"] = absl::GetFlag(FLAGS_allow_tf32);
  root["ppo_minibatch_size"] = static_cast<int64_t>(
      absl::GetFlag(FLAGS_ppo_minibatch_size));
  root["logit_cap"] = absl::GetFlag(FLAGS_logit_cap);
  root["seed"] = static_cast<int64_t>(absl::GetFlag(FLAGS_seed));
  root["threads"] = static_cast<int64_t>(absl::GetFlag(FLAGS_threads));
  root["eval_batch_size"] =
      static_cast<int64_t>(absl::GetFlag(FLAGS_eval_batch_size));
  root["eval_timeout_ms"] =
      static_cast<int64_t>(absl::GetFlag(FLAGS_eval_timeout_ms));
  root["rollout_games"] =
      static_cast<int64_t>(absl::GetFlag(FLAGS_rollout_games));
  root["hard_transition_ceiling"] = static_cast<int64_t>(
      absl::GetFlag(FLAGS_rollout_games)) * 5000;
  root["games_collected"] = static_cast<int64_t>(collect.games);
  root["transitions_collected"] = transitions;
  root["nontrivial_rows"] = nontrivial_rows;
  root["observation_size"] = obs_size;
  root["action_size"] = action_size;
  root["rollout_hash"] = ComputeRolloutHash(collect.rollout);
  root["shared_row_identity_sha256"] = shared_identity_sha256;
  root["model_state_sha256_before"] = model_hash_before;
  root["model_state_sha256_after"] = model_hash_after;
  root["inference_state_sha256_before"] = inference_hash_before;
  root["inference_state_sha256_after"] = inference_hash_after;

  json::Object capture;
  capture["decisions"] = collect.parity_decisions;
  capture["full_width_ok"] = collect.parity_full_width_ok;
  capture["full_finite_ok"] = collect.parity_full_finite_ok;
  capture["legal_ids_unique_in_range"] =
      collect.parity_legal_ids_unique_in_range;
  capture["chosen_once"] = collect.parity_chosen_once;
  capture["raw_values_total"] = integrity_stats.raw_values_total;
  capture["raw_values_bf16_exact"] =
      integrity_stats.raw_values_bf16_exact;
  capture["raw_values_bf16_exact_fraction"] =
      integrity_stats.raw_values_total > 0
          ? static_cast<double>(integrity_stats.raw_values_bf16_exact) /
                integrity_stats.raw_values_total
          : 0.0;
  json::Array capture_errors;
  for (const std::string& error : validity_errors) {
    capture_errors.emplace_back(error);
  }
  capture["validity_errors"] = json::Value(capture_errors);
  root["hard_validity"] = json::Value(capture);

  auto cell_json = [&](const PpoParityCellResult& cell,
                       const std::string& role,
                       const std::string& precision,
                       const PpoNumericalParitySummary& summary,
                       const PpoParityViolationMap* violations) {
    json::Object out;
    out["name"] = cell.name;
    out["role"] = role;
    out["precision"] = precision;
    out["model_forward_calls"] = cell.model_forward_calls;
    out["dense_mask_rows_ok"] = cell.dense_mask_rows_ok;
    out["row_evidence_sha256"] = cell.row_evidence_sha256;
    out["overall"] = json::Value(PpoNumericalParitySummaryJson(summary));
    out["splits"] = json::Value(PpoNumericalParitySplitsJson(cell.rows));
    json::Array errors;
    for (const std::string& error : cell.schema_errors) {
      errors.emplace_back(error);
    }
    out["schema_errors"] = json::Value(errors);
    if (violations != nullptr) {
      out["threshold_pass"] = PpoParityViolationMapPasses(*violations);
      out["violations"] = json::Value(PpoParityViolationMapJson(*violations));
    }
    return out;
  };
  json::Object cells;
  cells["A"] = json::Value(cell_json(
      integrity_cell, "hard_validity", "rollout_cpu_float_arithmetic", a,
      nullptr));
  cells["B"] = json::Value(cell_json(
      learner_cell, "actual_learner_phenotype", "cuda_bf16_autocast", b,
      &b_violations));
  cells["C"] = json::Value(cell_json(
      postprocess_cell, "captured_logits_postprocess_isolate",
      "captured_logits_dense_cuda_bf16", c, &c_violations));
  root["cells"] = json::Value(cells);
  root["B_C_violation_intersections"] = json::Value(
      PpoParityViolationIntersectionsJson(b_violations, c_violations));

  json::Object thresholds;
  thresholds["min_rows"] = static_cast<int64_t>(
      absl::GetFlag(FLAGS_numerical_parity_min_rows));
  thresholds["min_nontrivial_rows"] = static_cast<int64_t>(
      absl::GetFlag(FLAGS_numerical_parity_min_nontrivial_rows));
  thresholds["max_abs_chosen_logprob_delta"] =
      absl::GetFlag(FLAGS_numerical_parity_max_abs_logprob_delta);
  thresholds["max_full_legal_kl_each_direction"] =
      absl::GetFlag(FLAGS_numerical_parity_max_full_kl);
  thresholds["max_raw_mass_residual"] =
      absl::GetFlag(FLAGS_numerical_parity_max_raw_mass_residual);
  thresholds["min_ratio"] =
      absl::GetFlag(FLAGS_numerical_parity_min_ratio);
  thresholds["max_ratio"] =
      absl::GetFlag(FLAGS_numerical_parity_max_ratio);
  root["thresholds"] = json::Value(thresholds);
  size_t model_file_size = 0;
  const std::string model_path = absl::GetFlag(FLAGS_model_checkpoint);
  json::Object source;
  source["model_path"] = std::filesystem::absolute(model_path).string();
  source["model_sha256"] = ComputeFileSHA256(model_path, &model_file_size);
  source["model_size"] = static_cast<int64_t>(model_file_size);
  source["manifest_path"] = std::filesystem::absolute(source_manifest).string();
  source["manifest_sha256"] = source_manifest_sha256;
  source["global_update"] = source_global_update;
  source["checkpoint_uuid"] = source_checkpoint_uuid;
  size_t binary_size = 0;
  std::error_code ec;
  const std::filesystem::path binary_path =
      std::filesystem::read_symlink("/proc/self/exe", ec);
  if (ec || binary_path.empty()) {
    validity_errors.push_back("could not resolve executed binary path");
    root["status"] = "INVALID";
    root["classification"] = "INVALID";
    json::Array updated_errors;
    for (const std::string& error : validity_errors) {
      updated_errors.emplace_back(error);
    }
    root["hard_validity"].GetObject()["validity_errors"] =
        json::Value(updated_errors);
  } else {
    source["binary_path"] = binary_path.string();
    source["binary_sha256"] =
        ComputeFileSHA256(binary_path.string(), &binary_size);
    source["binary_size"] = static_cast<int64_t>(binary_size);
  }
  root["source"] = json::Value(source);
  json::Object source_code;
  source_code["root"] = source_provenance.root;
  source_code["combined_sha256"] = source_provenance.combined_sha256;
  source_code["canonical_contract"] =
      "fixed ordered relative_path + NUL + lowercase file_sha256 + newline";
  json::Array source_files;
  for (const auto& file : source_provenance.files) {
    json::Object record;
    record["relative_path"] = file.relative_path;
    record["absolute_path"] = file.absolute_path;
    record["size"] = file.size;
    record["sha256"] = file.sha256;
    source_files.emplace_back(json::Value(record));
  }
  source_code["files"] = json::Value(source_files);
  root["source_code"] = json::Value(source_code);

  const std::filesystem::path output(output_path);
  const std::filesystem::path tmp = output_path + ".tmp";
  if (std::filesystem::exists(output) || std::filesystem::exists(tmp)) {
    SpielFatalError("Numerical parity output already exists; refusing overwrite: " +
                    output_path);
  }
  if (!output.parent_path().empty()) {
    std::filesystem::create_directories(output.parent_path());
  }
  {
    std::ofstream out(tmp, std::ios::trunc);
    if (!out) SpielFatalError("Cannot create numerical parity temp artifact");
    out << json::ToString(root, true) << "\n";
    out.flush();
    if (!out) SpielFatalError("Numerical parity artifact write failed");
  }
  std::filesystem::rename(tmp, output);
  return root["status"] == "VALID";
}


bool WritePpoNumericalParityArtifactV5(
    const std::string& output_path, const PpoParityCellResult& cell_a,
    const PpoParityIntegrityStats& a_stats,
    const PpoParityCellResult& cell_r,
    const PpoParityReplayGeometryStats& r_stats,
    const PpoParityCellResult& cell_b,
    const PpoParityCellResult& cell_d,
    const PpoParityOriginalGeometryModelStats& d_stats,
    const PpoParityBatchGeometry& geometry,
    const std::string& geometry_sha256, const CollectResult& collect,
    const std::string& shared_identity_sha256,
    const std::string& model_hash_before,
    const std::string& model_hash_after,
    const std::string& inference_hash_before,
    const std::string& inference_hash_after,
    const std::string& source_manifest,
    const std::string& source_manifest_sha256, int64_t source_global_update,
    const std::string& source_checkpoint_uuid,
    const std::string& config_fingerprint, torch::Device device,
    int64_t obs_size, int64_t action_size, const std::string& command_line,
    const PpoNumericalParitySourceProvenance& source_provenance,
    bool tf32_cublas_before, bool tf32_cudnn_before,
    bool tf32_cublas_after, bool tf32_cudnn_after) {
  const int64_t transitions = static_cast<int64_t>(collect.rollout.size());
  const auto sa = SummarizePpoNumericalParityRows(cell_a.rows);
  const auto sr = SummarizePpoNumericalParityRows(cell_r.rows);
  const auto sb = SummarizePpoNumericalParityRows(cell_b.rows);
  const auto sd = SummarizePpoNumericalParityRows(cell_d.rows);
  auto hex64 = [](const std::string& value) {
    if (value.size() != 64) return false;
    return std::all_of(value.begin(), value.end(), [](char c) {
      return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
    });
  };
  std::vector<std::string> validity_errors = collect.parity_capture_errors;
  validity_errors.insert(validity_errors.end(), geometry.errors.begin(),
                         geometry.errors.end());
  auto require = [&](bool ok, const std::string& error) {
    if (!ok) validity_errors.push_back(error);
  };
  require(collect.episode_ids_unique, "episode IDs are not unique");
  require(static_cast<int64_t>(collect.games) ==
              absl::GetFlag(FLAGS_rollout_games),
          "exact game count mismatch");
  require(transitions >= absl::GetFlag(FLAGS_numerical_parity_min_rows),
          "row floor not met");
  int64_t nontrivial_rows = 0;
  for (const auto& row : cell_a.rows) if (row.legal_count > 1) ++nontrivial_rows;
  require(nontrivial_rows >=
              absl::GetFlag(FLAGS_numerical_parity_min_nontrivial_rows),
          "nontrivial row floor not met");
  require(collect.parity_decisions == transitions &&
              collect.parity_full_width_ok == transitions &&
              collect.parity_full_finite_ok == transitions &&
              collect.parity_legal_ids_unique_in_range == transitions &&
              collect.parity_chosen_once == transitions,
          "capture counters do not cover every transition");
  require(geometry.valid && geometry.rows == transitions &&
              geometry.groups > 0 && geometry.max_batch_size <=
                  absl::GetFlag(FLAGS_eval_batch_size),
          "physical-batch geometry is invalid");
  require(r_stats.before.requests == static_cast<uint64_t>(transitions) &&
              r_stats.before.batches == static_cast<uint64_t>(geometry.groups) &&
              r_stats.before.max_batch_size ==
                  static_cast<uint64_t>(geometry.max_batch_size) &&
              std::abs(r_stats.before.avg_batch_size -
                       static_cast<double>(transitions) / geometry.groups) <
                  1e-12,
          "pre-R evaluator stats do not match captured geometry");
  require(r_stats.after.requests - r_stats.before.requests ==
                  static_cast<uint64_t>(transitions) &&
              r_stats.after.batches - r_stats.before.batches ==
                  static_cast<uint64_t>(geometry.groups) &&
              r_stats.after.max_batch_size == r_stats.before.max_batch_size,
          "post-R evaluator stats deltas do not match exact replay");
  require(r_stats.returned_metadata_exact_rows == transitions &&
              r_stats.raw_legal_exact_rows == transitions &&
              r_stats.cpu_policy_exact_rows == transitions &&
              r_stats.model_forward_calls == geometry.groups &&
              r_stats.captured_raw_sha256 == r_stats.replay_raw_sha256,
          "R exact metadata/raw/policy/hash replay failed");
  require(sa.rows == transitions && sr.rows == transitions &&
              sb.rows == transitions && sd.rows == transitions,
          "A/R/B/D row counts differ");
  require(cell_a.schema_errors.empty() && cell_r.schema_errors.empty() &&
              cell_b.schema_errors.empty() && cell_d.schema_errors.empty(),
          "one or more cells have schema errors");
  require(sa.finite_rows == transitions && sr.finite_rows == transitions &&
              sb.finite_rows == transitions && sd.finite_rows == transitions,
          "one or more cells have incomplete finite rows");
  require(sa.max_abs_chosen_log_prob_delta == 0.0 &&
              sa.max_kl_old_new == 0.0 && sa.max_kl_new_old == 0.0 &&
              sa.ratio_min == 1.0 && sa.ratio_max == 1.0,
          "A is not exact identity");
  require(cell_a.model_forward_calls == 0 && cell_b.model_forward_calls > 0 &&
              cell_d.model_forward_calls == geometry.groups,
          "model-forward call counts are invalid");
  require(cell_b.dense_mask_rows_ok == transitions &&
              cell_d.dense_mask_rows_ok == transitions,
          "B/D dense-mask coverage incomplete");
  require(a_stats.raw_values_total == sa.legal_logits &&
              a_stats.vector_width_rows_ok == transitions &&
              a_stats.behavior_vector_exact_rows == transitions &&
              a_stats.chosen_scalar_exact_rows == transitions,
          "A raw/vector proof incomplete");
  PpoParityV5PrecisionConfig precision_config;
  precision_config.rollout_amp = absl::GetFlag(FLAGS_rollout_amp);
  precision_config.train_amp = absl::GetFlag(FLAGS_train_amp);
  precision_config.allow_tf32 = absl::GetFlag(FLAGS_allow_tf32);
  precision_config.tf32_cublas_before = tf32_cublas_before;
  precision_config.tf32_cudnn_before = tf32_cudnn_before;
  precision_config.tf32_cublas_after = tf32_cublas_after;
  precision_config.tf32_cudnn_after = tf32_cudnn_after;
  std::string precision_error;
  require(ValidatePpoParityV5PrecisionConfig(
              precision_config, &precision_error),
          precision_error.empty() ? "v5 precision runtime contract failed"
                                  : precision_error);
  require(!cell_a.module_train_mode && !cell_a.autocast_enabled &&
              !cell_r.module_train_mode && !cell_r.autocast_enabled &&
              cell_b.module_train_mode && !cell_b.autocast_enabled &&
              cell_d.module_train_mode && !cell_d.autocast_enabled,
          "A/R/B/D module-mode or autocast configuration mismatch");
  for (const auto* cell : {&cell_a, &cell_r, &cell_b, &cell_d}) {
    require(cell->input_dtype == "Float32" &&
                cell->pre_forward_dtype == "Float32",
            "cell input/pre-forward dtype is not Float32");
  }
  require(hex64(geometry_sha256) && hex64(shared_identity_sha256),
          "geometry/shared hash malformed");
  const std::vector<const PpoParityCellResult*> cell_list =
      {&cell_a, &cell_r, &cell_b, &cell_d};
  std::set<std::string> evidence_hashes;
  bool cell_hashes_valid = true;
  for (const auto* cell : cell_list) {
    cell_hashes_valid = cell_hashes_valid &&
        hex64(cell->row_evidence_sha256) && hex64(cell->raw_logit_sha256) &&
        hex64(cell->policy_sha256);
    evidence_hashes.insert(cell->row_evidence_sha256);
  }
  require(cell_hashes_valid && evidence_hashes.size() == cell_list.size(),
          "cell raw/policy/evidence hashes malformed or evidence not distinct");
  const bool shared_rows = PpoParityV5RowsShareIdentities(
      cell_a.rows, cell_r.rows, cell_b.rows, cell_d.rows, transitions);
  require(shared_rows, "A/R/B/D row identities differ");
  require(model_hash_before == model_hash_after &&
              inference_hash_before == inference_hash_after &&
              model_hash_before == inference_hash_before,
          "model immutability/equality failure");

  const bool instrument_valid = validity_errors.empty();
  const auto b_violations = BuildPpoParityViolationMap(cell_b.rows);
  const auto d_violations = BuildPpoParityViolationMap(cell_d.rows);
  const bool b_pass = instrument_valid &&
                      PpoParityViolationMapPasses(b_violations);
  const bool d_pass = instrument_valid &&
                      PpoParityViolationMapPasses(d_violations);
  const auto classification = ClassifyPpoParityV5(
      instrument_valid, b_pass, d_pass);

  json::Object root;
  root["schema"] = "dune_raw_ppo_numerical_parity_v5a";
  root["status"] = instrument_valid ? "VALID" : "INVALID";
  root["classification"] = PpoParityV5ClassificationName(classification);
  root["training_authorized"] = false;
  root["matched_2x2_preflight_admitted"] =
      instrument_valid &&
      classification ==
          PpoParityV5Classification::kFp32Tf32AllowedCandidateAdmitted;
  root["registration_id"] =
      absl::GetFlag(FLAGS_numerical_parity_registration_id);
  root["scope"] = "raw_ppo_only";
  root["epistemic_label"] =
      "no_training_fp32_tf32_allowed_candidate_screen";
  root["optimizer_steps"] = int64_t{0};
  root["optimizer_checkpoint_loaded"] = false;
  root["backward_calls"] = int64_t{0};
  root["training_updates"] = int64_t{0};
  root["command_line"] = command_line;
  root["config_fingerprint"] = config_fingerprint;
  root["device"] = device.str();
  root["rollout_amp"] = absl::GetFlag(FLAGS_rollout_amp);
  root["train_amp"] = absl::GetFlag(FLAGS_train_amp);
  root["allow_tf32"] = absl::GetFlag(FLAGS_allow_tf32);
  root["ppo_minibatch_size"] = static_cast<int64_t>(
      absl::GetFlag(FLAGS_ppo_minibatch_size));
  root["logit_cap"] = absl::GetFlag(FLAGS_logit_cap);
  root["seed"] = static_cast<int64_t>(absl::GetFlag(FLAGS_seed));
  root["threads"] = static_cast<int64_t>(absl::GetFlag(FLAGS_threads));
  root["eval_batch_size"] = static_cast<int64_t>(
      absl::GetFlag(FLAGS_eval_batch_size));
  root["eval_timeout_ms"] = static_cast<int64_t>(
      absl::GetFlag(FLAGS_eval_timeout_ms));
  root["rollout_games"] = static_cast<int64_t>(
      absl::GetFlag(FLAGS_rollout_games));
  root["hard_transition_ceiling"] = static_cast<int64_t>(
      absl::GetFlag(FLAGS_rollout_games)) * 5000;
  root["games_collected"] = static_cast<int64_t>(collect.games);
  root["transitions_collected"] = transitions;
  root["nontrivial_rows"] = nontrivial_rows;
  root["observation_size"] = obs_size;
  root["action_size"] = action_size;
  root["rollout_hash"] = ComputeRolloutHash(collect.rollout);
  root["shared_row_identity_sha256"] = shared_identity_sha256;
  root["physical_batch_geometry_sha256"] = geometry_sha256;
  root["model_state_sha256_before"] = model_hash_before;
  root["model_state_sha256_after"] = model_hash_after;
  root["inference_state_sha256_before"] = inference_hash_before;
  root["inference_state_sha256_after"] = inference_hash_after;
  json::Object thresholds;
  thresholds["min_rows"] = static_cast<int64_t>(
      absl::GetFlag(FLAGS_numerical_parity_min_rows));
  thresholds["min_nontrivial_rows"] = static_cast<int64_t>(
      absl::GetFlag(FLAGS_numerical_parity_min_nontrivial_rows));
  thresholds["max_abs_chosen_logprob_delta"] =
      absl::GetFlag(FLAGS_numerical_parity_max_abs_logprob_delta);
  thresholds["max_full_legal_kl_each_direction"] =
      absl::GetFlag(FLAGS_numerical_parity_max_full_kl);
  thresholds["max_raw_mass_residual"] =
      absl::GetFlag(FLAGS_numerical_parity_max_raw_mass_residual);
  thresholds["min_ratio"] =
      absl::GetFlag(FLAGS_numerical_parity_min_ratio);
  thresholds["max_ratio"] =
      absl::GetFlag(FLAGS_numerical_parity_max_ratio);
  root["thresholds"] = json::Value(thresholds);
  json::Object precision;
  precision["input_dtype"] = "Float32";
  precision["pre_forward_dtype"] = "Float32";
  precision["rollout_autocast"] = absl::GetFlag(FLAGS_rollout_amp);
  precision["learner_autocast"] = absl::GetFlag(FLAGS_train_amp);
  precision["tf32_cublas_before"] = tf32_cublas_before;
  precision["tf32_cudnn_before"] = tf32_cudnn_before;
  precision["tf32_cublas_after"] = tf32_cublas_after;
  precision["tf32_cudnn_after"] = tf32_cudnn_after;
  precision["runtime_state_unchanged"] =
      tf32_cublas_before == tf32_cublas_after &&
      tf32_cudnn_before == tf32_cudnn_after;
  root["precision_runtime"] = json::Value(precision);
  json::Object capture;
  capture["decisions"] = collect.parity_decisions;
  capture["full_width_ok"] = collect.parity_full_width_ok;
  capture["full_finite_ok"] = collect.parity_full_finite_ok;
  capture["legal_ids_unique_in_range"] =
      collect.parity_legal_ids_unique_in_range;
  capture["chosen_once"] = collect.parity_chosen_once;
  capture["raw_values_total"] = a_stats.raw_values_total;
  capture["raw_values_bf16_exact_descriptive"] =
      a_stats.raw_values_bf16_exact;
  root["hard_capture"] = json::Value(capture);

  auto stats_json = [](const EvaluatorStats& stats) {
    json::Object out;
    out["requests"] = static_cast<int64_t>(stats.requests);
    out["batches"] = static_cast<int64_t>(stats.batches);
    out["max_batch_size"] = static_cast<int64_t>(stats.max_batch_size);
    out["mean_batch_size"] = stats.avg_batch_size;
    return out;
  };
  json::Object metadata;
  metadata["groups"] = geometry.groups;
  metadata["rows"] = geometry.rows;
  metadata["max_batch_size"] = geometry.max_batch_size;
  metadata["pre_R_stats"] = json::Value(stats_json(r_stats.before));
  metadata["post_R_stats"] = json::Value(stats_json(r_stats.after));
  metadata["returned_metadata_exact_rows"] =
      r_stats.returned_metadata_exact_rows;
  metadata["R_captured_raw_sha256"] = r_stats.captured_raw_sha256;
  metadata["R_replay_raw_sha256"] = r_stats.replay_raw_sha256;
  json::Array geometry_errors;
  for (const auto& error : validity_errors) geometry_errors.emplace_back(error);
  metadata["validity_errors"] = json::Value(geometry_errors);
  root["physical_batch_metadata"] = json::Value(metadata);

  auto cell_json = [&](const PpoParityCellResult& cell,
                       const std::string& role,
                       const PpoNumericalParitySummary& summary,
                       const PpoParityViolationMap* violations) {
    json::Object out;
    out["name"] = cell.name;
    out["role"] = role;
    out["model_forward_calls"] = cell.model_forward_calls;
    out["dense_mask_rows_ok"] = cell.dense_mask_rows_ok;
    out["module_mode"] = cell.module_train_mode ? "train" : "eval";
    out["autocast_enabled"] = cell.autocast_enabled;
    out["input_dtype"] = cell.input_dtype;
    out["pre_forward_dtype"] = cell.pre_forward_dtype;
    out["wall_time_s"] = cell.wall_time_s;
    out["device_time_s"] = cell.device_time_s;
    out["device_timing_available"] = cell.device_time_s >= 0.0;
    out["row_evidence_sha256"] = cell.row_evidence_sha256;
    out["raw_logit_sha256"] = cell.raw_logit_sha256;
    out["policy_sha256"] = cell.policy_sha256;
    out["overall"] = json::Value(PpoNumericalParitySummaryJson(summary));
    out["splits"] = json::Value(PpoNumericalParitySplitsJson(cell.rows));
    json::Array errors;
    for (const auto& error : cell.schema_errors) errors.emplace_back(error);
    out["schema_errors"] = json::Value(errors);
    if (violations != nullptr) {
      out["threshold_pass"] = PpoParityViolationMapPasses(*violations);
      out["violations"] = json::Value(PpoParityViolationMapJson(*violations));
    }
    return out;
  };
  json::Object cells;
  cells["A"] = json::Value(cell_json(cell_a, "hard_validity", sa, nullptr));
  cells["R"] = json::Value(cell_json(
      cell_r, "live_inference_exact_batch_replay", sr, nullptr));
  cells["B"] = json::Value(cell_json(
      cell_b, "full_fp32_tf32_2048_candidate", sb, &b_violations));
  json::Object d_json = cell_json(
      cell_d, "fp32_tf32_original_batch_geometry", sd, &d_violations);
  d_json["raw_legal_bit_exact_rows"] = d_stats.raw_legal_exact_rows;
  d_json["raw_legal_max_abs_delta"] = d_stats.raw_legal_max_abs_delta;
  cells["D"] = json::Value(d_json);
  root["cells"] = json::Value(cells);
  root["B_D_violation_intersections"] = json::Value(
      PpoParityViolationIntersectionsJson(b_violations, d_violations));

  size_t model_size = 0;
  const std::string model_path = absl::GetFlag(FLAGS_model_checkpoint);
  json::Object source;
  source["model_path"] = std::filesystem::absolute(model_path).string();
  source["model_sha256"] = ComputeFileSHA256(model_path, &model_size);
  source["model_size"] = static_cast<int64_t>(model_size);
  source["manifest_path"] = std::filesystem::absolute(source_manifest).string();
  source["manifest_sha256"] = source_manifest_sha256;
  source["global_update"] = source_global_update;
  source["checkpoint_uuid"] = source_checkpoint_uuid;
  size_t binary_size = 0;
  std::error_code ec;
  const auto binary_path = std::filesystem::read_symlink("/proc/self/exe", ec);
  if (ec || binary_path.empty()) {
    root["status"] = "INVALID";
    root["classification"] = "INVALID";
  } else {
    source["binary_path"] = binary_path.string();
    source["binary_sha256"] =
        ComputeFileSHA256(binary_path.string(), &binary_size);
    source["binary_size"] = static_cast<int64_t>(binary_size);
  }
  root["source"] = json::Value(source);
  json::Object source_code;
  source_code["root"] = source_provenance.root;
  source_code["combined_sha256"] = source_provenance.combined_sha256;
  json::Array source_files;
  for (const auto& file : source_provenance.files) {
    json::Object record;
    record["relative_path"] = file.relative_path;
    record["absolute_path"] = file.absolute_path;
    record["size"] = file.size;
    record["sha256"] = file.sha256;
    source_files.emplace_back(json::Value(record));
  }
  source_code["files"] = json::Value(source_files);
  root["source_code"] = json::Value(source_code);
  const std::filesystem::path output(output_path);
  const std::filesystem::path tmp = output_path + ".tmp";
  if (std::filesystem::exists(output) || std::filesystem::exists(tmp)) {
    SpielFatalError("Numerical parity output already exists; refusing overwrite: " +
                    output_path);
  }
  if (!output.parent_path().empty()) {
    std::filesystem::create_directories(output.parent_path());
  }
  {
    std::ofstream out(tmp, std::ios::trunc);
    if (!out) SpielFatalError("Cannot create numerical parity temp artifact");
    out << json::ToString(root, true) << "\n";
    out.flush();
    if (!out) SpielFatalError("Numerical parity artifact write failed");
  }
  std::filesystem::rename(tmp, output);
  return root["status"] == "VALID";
}


#endif  // OPEN_SPIEL_BUILD_WITH_LIBTORCH

}  // namespace open_spiel::ppo_train_internal
