#pragma once

// Numerical parity structs and declarations included by dune_ppo_train_internal.h.

#ifdef OPEN_SPIEL_BUILD_WITH_LIBTORCH

namespace open_spiel::ppo_train_internal {

using PpoParityViolationMap =
    std::map<std::string, std::vector<std::string>>;

struct PpoNumericalParitySourceRecord {
  std::string relative_path;
  std::string absolute_path;
  int64_t size = 0;
  std::string sha256;
};

struct PpoNumericalParitySourceProvenance {
  std::string root;
  std::string combined_sha256;
  std::vector<PpoNumericalParitySourceRecord> files;
};

struct PpoParityCellResult {
  std::string name;
  std::vector<PpoNumericalParityRow> rows;
  std::vector<std::string> schema_errors;
  std::string row_evidence_sha256;
  std::string raw_logit_sha256;
  std::string policy_sha256;
  int64_t model_forward_calls = 0;
  int64_t dense_mask_rows_ok = 0;
  bool module_train_mode = false;
  bool autocast_enabled = false;
  std::string input_dtype = "Float32";
  std::string pre_forward_dtype = "Float32";
  double wall_time_s = 0.0;
  double device_time_s = -1.0;
};

struct PpoParityIntegrityStats {
  int64_t raw_values_total = 0;
  int64_t raw_values_bf16_exact = 0;
  int64_t vector_width_rows_ok = 0;
  int64_t behavior_vector_exact_rows = 0;
  int64_t chosen_scalar_exact_rows = 0;
};

struct PpoParityReplayGeometryStats {
  EvaluatorStats before;
  EvaluatorStats after;
  int64_t returned_metadata_exact_rows = 0;
  int64_t raw_legal_exact_rows = 0;
  int64_t cpu_policy_exact_rows = 0;
  int64_t model_forward_calls = 0;
  std::string captured_raw_sha256;
  std::string replay_raw_sha256;
};

struct PpoParityOriginalGeometryModelStats {
  int64_t raw_legal_exact_rows = 0;
  double raw_legal_max_abs_delta = 0.0;
};

// --- Parity replay & write (dune_ppo_train_parity_replay.cc / _write*.cc) ---
std::string PpoParityRowIdentitySha256(
    int64_t row_index, const PpoTransition& transition);
std::string PpoParitySharedIdentitySha256(const std::vector<PpoTransition>& batch);
std::string PpoParityCapturedRawSha256(const std::vector<PpoTransition>& batch);
std::string PpoParityCapturedPolicySha256(const std::vector<PpoTransition>& batch);
std::vector<PpoNumericalParityRow> ReplayPpoCpuIntegrityCell(
    const std::vector<PpoTransition>& batch, float logit_cap,
    bool require_bf16_grid,
    std::vector<std::string>* schema_errors,
    std::string* row_evidence_sha256, int64_t* raw_values_total,
    int64_t* raw_values_bf16_exact, int64_t* vector_width_rows_ok,
    int64_t* behavior_vector_exact_rows, int64_t* chosen_scalar_exact_rows);
std::vector<PpoNumericalParityRow> ReplayPpoNumericalParityCell(
    const std::shared_ptr<SharedDunePolicyValueNetImpl>& model,
    const std::vector<PpoTransition>& batch, int64_t obs_size,
    int64_t action_dim, torch::Device device, bool model_train_mode,
    bool learner_autocast, const std::string& cell_label,
    std::vector<std::string>* schema_errors,
    std::string* row_evidence_sha256, int64_t* model_forward_calls,
    int64_t* dense_mask_rows_ok, std::string* raw_logit_sha256,
    std::string* policy_sha256);
std::vector<PpoNumericalParityRow> ReplayPpoCapturedLogitsBf16Cell(
    const std::vector<PpoTransition>& batch, int64_t action_dim,
    torch::Device device, float logit_cap,
    std::vector<std::string>* schema_errors,
    std::string* row_evidence_sha256, int64_t* model_forward_calls,
    int64_t* dense_mask_rows_ok);
json::Object PpoNumericalParitySummaryJson(const PpoNumericalParitySummary& s);
std::string PpoDecisionRoleName(int role);
json::Object PpoNumericalParitySplitsJson(const std::vector<PpoNumericalParityRow>& rows);
PpoParityViolationMap BuildPpoParityViolationMap(const std::vector<PpoNumericalParityRow>& rows);
bool PpoParityViolationMapPasses(const PpoParityViolationMap& violations);
json::Object PpoParityViolationSetJson(const std::vector<std::string>& identities);
json::Object PpoParityViolationMapJson(const PpoParityViolationMap& violations);
json::Object PpoParityViolationIntersectionsJson(
    const PpoParityViolationMap& first, const PpoParityViolationMap& second);
PpoNumericalParitySourceProvenance LoadPpoNumericalParitySourceProvenance(
    const std::string& source_root, const std::string& registered_sha256);
PpoParityCellResult ReplayPpoInferenceExactGeometry(
    BatchedEvaluator* evaluator, const std::vector<PpoTransition>& batch,
    const PpoParityBatchGeometry& geometry, int64_t action_dim,
    float logit_cap, const EvaluatorStats& before,
    PpoParityReplayGeometryStats* stats);
PpoParityCellResult ReplayPpoTrainingModelOriginalGeometry(
    const std::shared_ptr<SharedDunePolicyValueNetImpl>& model,
    const std::vector<PpoTransition>& batch,
    const PpoParityBatchGeometry& geometry, int64_t obs_size,
    int64_t action_dim, torch::Device device, float logit_cap,
    bool learner_autocast, PpoParityOriginalGeometryModelStats* stats);
bool WritePpoNumericalParityArtifact(
    const std::string& output_path,
    const std::vector<PpoNumericalParityRow>& primary_rows,
    const std::vector<std::string>& primary_schema_errors,
    const std::string& primary_row_evidence_sha256,
    const std::vector<PpoNumericalParityRow>& control_rows,
    const std::vector<std::string>& control_schema_errors,
    const std::string& control_row_evidence_sha256,
    const CollectResult& collect, const std::string& model_hash_before,
    const std::string& model_hash_after,
    const std::string& inference_hash_before,
    const std::string& inference_hash_after, const std::string& source_manifest,
    const std::string& source_manifest_sha256, int64_t source_global_update,
    const std::string& source_checkpoint_uuid,
    const std::string& config_fingerprint, torch::Device device,
    int64_t obs_size, int64_t action_size, const std::string& command_line,
    const PpoNumericalParitySourceProvenance& source_provenance);
bool WritePpoNumericalParityArtifactV3(
    const std::string& output_path, const PpoParityCellResult& integrity_cell,
    const PpoParityIntegrityStats& integrity_stats,
    const PpoParityCellResult& learner_cell,
    const PpoParityCellResult& postprocess_cell, const CollectResult& collect,
    const std::string& shared_identity_sha256,
    const std::string& model_hash_before, const std::string& model_hash_after,
    const std::string& inference_hash_before,
    const std::string& inference_hash_after, const std::string& source_manifest,
    const std::string& source_manifest_sha256, int64_t source_global_update,
    const std::string& source_checkpoint_uuid,
    const std::string& config_fingerprint, torch::Device device,
    int64_t obs_size, int64_t action_size, const std::string& command_line,
    const PpoNumericalParitySourceProvenance& source_provenance);
bool WritePpoNumericalParityArtifactV5(
    const std::string& output_path, const PpoParityCellResult& cell_a,
    const PpoParityIntegrityStats& a_stats, const PpoParityCellResult& cell_r,
    const PpoParityReplayGeometryStats& r_stats,
    const PpoParityCellResult& cell_b, const PpoParityCellResult& cell_d,
    const PpoParityOriginalGeometryModelStats& d_stats,
    const PpoParityBatchGeometry& geometry, const std::string& geometry_sha256,
    const CollectResult& collect, const std::string& shared_identity_sha256,
    const std::string& model_hash_before, const std::string& model_hash_after,
    const std::string& inference_hash_before,
    const std::string& inference_hash_after, const std::string& source_manifest,
    const std::string& source_manifest_sha256, int64_t source_global_update,
    const std::string& source_checkpoint_uuid,
    const std::string& config_fingerprint, torch::Device device,
    int64_t obs_size, int64_t action_size, const std::string& command_line,
    const PpoNumericalParitySourceProvenance& source_provenance,
    bool tf32_cublas_before, bool tf32_cudnn_before, bool tf32_cublas_after,
    bool tf32_cudnn_after);

}  // namespace open_spiel::ppo_train_internal

#endif  // OPEN_SPIEL_BUILD_WITH_LIBTORCH
