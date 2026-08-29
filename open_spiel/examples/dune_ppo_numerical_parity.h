#ifndef OPEN_SPIEL_EXAMPLES_DUNE_PPO_NUMERICAL_PARITY_H_
#define OPEN_SPIEL_EXAMPLES_DUNE_PPO_NUMERICAL_PARITY_H_

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include <c10/util/BFloat16.h>

namespace open_spiel {

// Source inputs whose exact bytes determine the parity measurement semantics.
// The order is part of the v1 provenance contract: callers hash the canonical
// payload returned below, never directory iteration order.
inline std::vector<std::string> PpoNumericalParitySourceRelativePaths() {
  return {
      "open_spiel/examples/dune_ppo_train.cc",
      "open_spiel/examples/dune_ppo_numerical_parity.h",
      "open_spiel/examples/dune_ppo_training_utils.h",
      "open_spiel/examples/dune_ppo_training_utils.cc",
      "open_spiel/examples/dune_network.h",
      "open_spiel/examples/dune_search_routing.cc",
      "open_spiel/examples/dune_search_routing.h",
  };
}

// Canonical input to the combined source SHA-256: for each required source in
// the fixed order, UTF-8 relative_path, one NUL byte, lowercase file SHA-256,
// and one newline. Exact path/order validation prevents a caller from hashing a
// convenient subset or accepting directory-enumeration drift.
inline bool CanonicalPpoNumericalParitySourcePayload(
    const std::vector<std::pair<std::string, std::string>>& records,
    std::string* payload, std::string* error) {
  auto fail = [&](const std::string& message) {
    if (error != nullptr) *error = message;
    return false;
  };
  if (payload == nullptr) return fail("null canonical payload output");
  const std::vector<std::string> expected =
      PpoNumericalParitySourceRelativePaths();
  if (records.size() != expected.size()) {
    return fail("source record count does not match fixed provenance list");
  }
  auto valid_sha256 = [](const std::string& digest) {
    if (digest.size() != 64) return false;
    for (char c : digest) {
      if (!((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'))) return false;
    }
    return true;
  };
  std::string canonical;
  for (size_t i = 0; i < expected.size(); ++i) {
    if (records[i].first != expected[i]) {
      return fail("source path/order differs from fixed provenance list at index " +
                  std::to_string(i));
    }
    if (!valid_sha256(records[i].second)) {
      return fail("source file digest is not lowercase SHA-256 at index " +
                  std::to_string(i));
    }
    canonical.append(records[i].first);
    canonical.push_back('\0');
    canonical.append(records[i].second);
    canonical.push_back('\n');
  }
  *payload = std::move(canonical);
  return true;
}

// One frozen-policy comparison row. `old_log_probs` is the legal categorical
// distribution actually used by rollout sampling (CUDA BF16 inference followed
// by the rollout's CPU legal softmax). `new_log_probs` is the same unchanged
// model replayed through one explicitly labelled learner cell (production BF16
// autocast or descriptive FP32). Both vectors are aligned to the transition's
// legal-action order.
struct PpoNumericalParityInput {
  int legal_count = 0;
  int decision_role = -1;
  double advantage = 0.0;
  int chosen_index = -1;
  double stored_chosen_log_prob = 0.0;
  std::vector<double> old_log_probs;
  std::vector<double> new_log_probs;
};

struct PpoNumericalParityRow {
  int legal_count = 0;
  int decision_role = -1;
  double advantage = 0.0;
  double old_chosen_probability = 0.0;
  double chosen_log_prob_delta = 0.0;  // labelled replay - stored rollout
  double ratio = 1.0;
  double kl_old_new = 0.0;             // full legal KL(old || new)
  double kl_new_old = 0.0;             // full legal KL(new || old)
  // Absolute residual of the RAW float-stored legal distribution before the
  // v2 double-logsumexp renormalization used for KL. These are independently
  // gated: renormalization fixes the measurement, it does not hide bad input.
  double old_raw_mass_residual = 0.0;
  double new_raw_mass_residual = 0.0;
  bool mass_residual_valid = false;
  int64_t old_probability_underflows = 0;
  int64_t new_probability_underflows = 0;
  int64_t nonfinite_values = 0;
  int64_t schema_errors = 0;
  std::string row_identity_sha256;
};

inline uint32_t PpoParityFloatBits(float value) {
  uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

inline bool PpoParityBf16RoundTripsBitExactly(float value) {
  const float roundtrip = static_cast<float>(c10::BFloat16(value));
  return PpoParityFloatBits(value) == PpoParityFloatBits(roundtrip);
}

// Exact CPU postprocessing used by rollout after the evaluator returns its
// full FP32 container of BF16-grid logits. The input is legal-only and already
// ordered like state.LegalActions(); subtracting one shared legal mean means no
// illegal value is needed.
inline bool RecomputePpoBehaviorLegalLogProbs(
    const std::vector<float>& raw_legal_logits, float logit_cap,
    std::vector<float>* out, std::string* error) {
  auto fail = [&](const std::string& message) {
    if (error != nullptr) *error = message;
    return false;
  };
  if (out == nullptr) return fail("null behavior recompute output");
  if (raw_legal_logits.empty()) return fail("empty raw legal-logit vector");
  double sum = 0.0;
  for (float value : raw_legal_logits) {
    if (!std::isfinite(value)) return fail("nonfinite raw legal logit");
    sum += value;
  }
  const float mean = static_cast<float>(
      sum / static_cast<double>(raw_legal_logits.size()));
  std::vector<float> logits = raw_legal_logits;
  for (float& value : logits) {
    value -= mean;
    if (logit_cap > 0.0f) {
      value = logit_cap * std::tanh(value / logit_cap);
    }
  }
  float maximum = -std::numeric_limits<float>::infinity();
  for (float value : logits) maximum = std::max(maximum, value);
  std::vector<double> weights(logits.size(), 0.0);
  double total = 0.0;
  for (size_t i = 0; i < logits.size(); ++i) {
    weights[i] = std::exp(static_cast<double>(logits[i] - maximum));
    if (!std::isfinite(weights[i]) || weights[i] < 0.0) weights[i] = 0.0;
    total += weights[i];
  }
  if (!(total > 0.0) || !std::isfinite(total)) {
    weights.assign(logits.size(), 1.0);
    total = static_cast<double>(logits.size());
  }
  std::vector<float> result;
  result.reserve(logits.size());
  for (double weight : weights) {
    result.push_back(weight > 0.0
                         ? static_cast<float>(std::log(weight / total))
                         : -std::numeric_limits<float>::infinity());
  }
  *out = std::move(result);
  return true;
}

struct PpoParityPrecapCaptureValidation {
  bool full_width_ok = false;
  bool full_finite_ok = false;
  bool legal_ids_unique_in_range = false;
};

inline PpoParityPrecapCaptureValidation ValidateAndCapturePpoParityPrecap(
    const std::vector<float>& full_logits, int64_t action_dim,
    const std::vector<int64_t>& legal_actions,
    std::vector<float>* raw_legal_logits, std::string* error) {
  PpoParityPrecapCaptureValidation out;
  if (raw_legal_logits == nullptr) {
    if (error != nullptr) *error = "null raw legal-logit output";
    return out;
  }
  out.full_width_ok =
      static_cast<int64_t>(full_logits.size()) == action_dim;
  out.full_finite_ok = std::all_of(
      full_logits.begin(), full_logits.end(),
      [](float value) { return std::isfinite(value); });
  std::set<int64_t> unique;
  out.legal_ids_unique_in_range = !legal_actions.empty();
  raw_legal_logits->clear();
  raw_legal_logits->reserve(legal_actions.size());
  for (int64_t action : legal_actions) {
    const bool in_range = action >= 0 && action < action_dim &&
                          action < static_cast<int64_t>(full_logits.size());
    if (!in_range || !unique.insert(action).second) {
      out.legal_ids_unique_in_range = false;
    }
    raw_legal_logits->push_back(
        in_range ? full_logits[action]
                 : std::numeric_limits<float>::quiet_NaN());
  }
  if (error != nullptr) {
    std::string message;
    if (!out.full_width_ok) message += "full_width;";
    if (!out.full_finite_ok) message += "full_nonfinite;";
    if (!out.legal_ids_unique_in_range) message += "legal_ids;";
    *error = message;
  }
  return out;
}

inline bool ValidatePpoParityBehaviorCaptureWidthsAndChoice(
    const std::vector<int64_t>& legal_actions, int64_t chosen_action,
    const std::vector<float>& raw_legal_logits,
    const std::vector<float>& behavior_log_probs, std::string* error) {
  if (legal_actions.empty() || raw_legal_logits.size() != legal_actions.size() ||
      behavior_log_probs.size() != legal_actions.size()) {
    if (error != nullptr) *error = "legal/raw/behavior vector width mismatch";
    return false;
  }
  const int count = static_cast<int>(std::count(
      legal_actions.begin(), legal_actions.end(), chosen_action));
  if (count != 1) {
    if (error != nullptr) *error = "chosen action does not occur exactly once";
    return false;
  }
  return true;
}

enum class PpoParityV3Classification {
  kInvalid,
  kPostprocessSufficient,
  kForwardBatchComponentNecessary,
  kInconclusive,
};

inline PpoParityV3Classification ClassifyPpoParityV3(
    bool instrument_valid, bool learner_bf16_pass,
    bool captured_logits_postprocess_pass) {
  if (!instrument_valid) return PpoParityV3Classification::kInvalid;
  if (learner_bf16_pass) return PpoParityV3Classification::kInconclusive;
  return captured_logits_postprocess_pass
             ? PpoParityV3Classification::kForwardBatchComponentNecessary
             : PpoParityV3Classification::kPostprocessSufficient;
}

inline std::string PpoParityV3ClassificationName(
    PpoParityV3Classification classification) {
  switch (classification) {
    case PpoParityV3Classification::kInvalid: return "INVALID";
    case PpoParityV3Classification::kPostprocessSufficient:
      return "POSTPROCESS_SUFFICIENT";
    case PpoParityV3Classification::kForwardBatchComponentNecessary:
      return "FORWARD_BATCH_COMPONENT_NECESSARY";
    case PpoParityV3Classification::kInconclusive: return "INCONCLUSIVE";
  }
  return "INVALID";
}

enum class PpoParityV4Classification {
  kInvalid,
  kInconclusivePhenotypeNotReproduced,
  kBatchGeometrySufficient,
  kBatchGeometryInsufficient,
};

inline PpoParityV4Classification ClassifyPpoParityV4(
    bool instrument_valid, bool learner_bf16_pass,
    bool original_geometry_replay_pass) {
  if (!instrument_valid) return PpoParityV4Classification::kInvalid;
  if (learner_bf16_pass) {
    return PpoParityV4Classification::kInconclusivePhenotypeNotReproduced;
  }
  return original_geometry_replay_pass
             ? PpoParityV4Classification::kBatchGeometrySufficient
             : PpoParityV4Classification::kBatchGeometryInsufficient;
}

inline std::string PpoParityV4ClassificationName(
    PpoParityV4Classification classification) {
  switch (classification) {
    case PpoParityV4Classification::kInvalid: return "INVALID";
    case PpoParityV4Classification::kInconclusivePhenotypeNotReproduced:
      return "INCONCLUSIVE_PHENOTYPE_NOT_REPRODUCED";
    case PpoParityV4Classification::kBatchGeometrySufficient:
      return "BATCH_GEOMETRY_SUFFICIENT";
    case PpoParityV4Classification::kBatchGeometryInsufficient:
      return "BATCH_GEOMETRY_INSUFFICIENT";
  }
  return "INVALID";
}

enum class PpoParityV5Classification {
  kInvalid,
  kFp32Tf32AllowedCandidateAdmitted,
  kFp32Tf32AllowedBatchGeometryReject,
  kInconclusiveCommonOrInteraction,
};

inline PpoParityV5Classification ClassifyPpoParityV5(
    bool instrument_valid, bool learner_2048_pass,
    bool original_geometry_pass) {
  if (!instrument_valid) return PpoParityV5Classification::kInvalid;
  if (learner_2048_pass && original_geometry_pass) {
    return PpoParityV5Classification::kFp32Tf32AllowedCandidateAdmitted;
  }
  if (!learner_2048_pass && original_geometry_pass) {
    return PpoParityV5Classification::kFp32Tf32AllowedBatchGeometryReject;
  }
  return PpoParityV5Classification::kInconclusiveCommonOrInteraction;
}

inline std::string PpoParityV5ClassificationName(
    PpoParityV5Classification classification) {
  switch (classification) {
    case PpoParityV5Classification::kInvalid: return "INVALID";
    case PpoParityV5Classification::kFp32Tf32AllowedCandidateAdmitted:
      return "FP32_TF32_ALLOWED_CANDIDATE_ADMITTED";
    case PpoParityV5Classification::kFp32Tf32AllowedBatchGeometryReject:
      return "FP32_TF32_ALLOWED_BATCH_GEOMETRY_REJECT";
    case PpoParityV5Classification::kInconclusiveCommonOrInteraction:
      return "INCONCLUSIVE_COMMON_OR_INTERACTION";
  }
  return "INVALID";
}

struct PpoParityV5PrecisionConfig {
  bool rollout_amp = true;
  bool train_amp = true;
  bool allow_tf32 = true;
  bool tf32_cublas_before = true;
  bool tf32_cudnn_before = true;
  bool tf32_cublas_after = true;
  bool tf32_cudnn_after = true;
  std::string input_dtype = "Float32";
  std::string pre_forward_dtype = "Float32";
};

inline bool ValidatePpoParityV5PrecisionConfig(
    const PpoParityV5PrecisionConfig& config, std::string* error) {
  const bool valid = !config.rollout_amp && !config.train_amp &&
      config.allow_tf32 && config.tf32_cublas_before &&
      config.tf32_cudnn_before && config.tf32_cublas_after &&
      config.tf32_cudnn_after && config.input_dtype == "Float32" &&
      config.pre_forward_dtype == "Float32";
  if (!valid && error != nullptr) {
    *error = "v5 requires Float32 input/pre-forward, autocast off, TF32 on and unchanged";
  }
  return valid;
}

inline std::vector<std::string> IntersectPpoParityViolationIdentities(
    const std::vector<std::string>& first,
    const std::vector<std::string>& second) {
  const std::set<std::string> second_set(second.begin(), second.end());
  std::vector<std::string> out;
  for (const std::string& identity : first) {
    if (second_set.find(identity) != second_set.end()) out.push_back(identity);
  }
  return out;
}

inline bool CanonicalPpoParityViolationIdentityPayload(
    const std::vector<std::string>& identities, std::string* payload,
    std::string* error) {
  auto valid_sha256 = [](const std::string& digest) {
    if (digest.size() != 64) return false;
    for (char c : digest) {
      if (!((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'))) return false;
    }
    return true;
  };
  if (payload == nullptr) {
    if (error != nullptr) *error = "null violation payload output";
    return false;
  }
  std::string out;
  for (const std::string& identity : identities) {
    if (!valid_sha256(identity)) {
      if (error != nullptr) *error = "invalid row-identity SHA-256";
      return false;
    }
    out.append(identity);
    out.push_back('\n');
  }
  *payload = std::move(out);
  return true;
}

struct PpoParityBatchMembership {
  int64_t batch_id = -1;
  int32_t batch_size = -1;
  int32_t batch_row = -1;
};

struct PpoParityBatchGeometry {
  bool valid = false;
  int64_t groups = 0;
  int64_t rows = 0;
  int64_t max_batch_size = 0;
  std::vector<std::vector<size_t>> row_indices_by_group;
  std::string canonical_payload;
  std::vector<std::string> errors;
};

inline PpoParityBatchGeometry ReconstructPpoParityBatchGeometry(
    const std::vector<PpoParityBatchMembership>& membership,
    int64_t configured_max_batch_size) {
  PpoParityBatchGeometry out;
  out.rows = static_cast<int64_t>(membership.size());
  std::map<int64_t, std::vector<std::pair<int32_t, size_t>>> groups;
  std::map<int64_t, int32_t> sizes;
  for (size_t i = 0; i < membership.size(); ++i) {
    const auto& item = membership[i];
    if (item.batch_id < 0 || item.batch_size < 1 ||
        item.batch_size > configured_max_batch_size || item.batch_row < 0 ||
        item.batch_row >= item.batch_size) {
      out.errors.push_back("row " + std::to_string(i) +
                           " has invalid membership sentinel/range");
      continue;
    }
    auto found = sizes.find(item.batch_id);
    if (found == sizes.end()) {
      sizes[item.batch_id] = item.batch_size;
    } else if (found->second != item.batch_size) {
      out.errors.push_back("batch " + std::to_string(item.batch_id) +
                           " carries inconsistent sizes");
    }
    groups[item.batch_id].push_back({item.batch_row, i});
    out.max_batch_size = std::max<int64_t>(out.max_batch_size,
                                           item.batch_size);
  }
  if (groups.empty() && !membership.empty()) {
    out.errors.push_back("no valid batch groups reconstructed");
  }
  int64_t expected_id = 0;
  int64_t member_sum = 0;
  for (auto& item : groups) {
    const int64_t id = item.first;
    if (id != expected_id) {
      out.errors.push_back("batch IDs are not contiguous from zero");
    }
    ++expected_id;
    auto& members = item.second;
    std::sort(members.begin(), members.end());
    const int32_t expected_size = sizes[id];
    if (static_cast<int32_t>(members.size()) != expected_size) {
      out.errors.push_back("batch " + std::to_string(id) +
                           " member count differs from declared size");
    }
    std::vector<size_t> row_indices;
    for (size_t position = 0; position < members.size(); ++position) {
      if (members[position].first != static_cast<int32_t>(position)) {
        out.errors.push_back("batch " + std::to_string(id) +
                             " has duplicate/missing row positions");
      }
      row_indices.push_back(members[position].second);
      out.canonical_payload.append(std::to_string(id));
      out.canonical_payload.push_back(':');
      out.canonical_payload.append(std::to_string(expected_size));
      out.canonical_payload.push_back(':');
      out.canonical_payload.append(std::to_string(position));
      out.canonical_payload.push_back(':');
      out.canonical_payload.append(std::to_string(members[position].second));
      out.canonical_payload.push_back('\n');
      ++member_sum;
    }
    out.row_indices_by_group.push_back(std::move(row_indices));
  }
  out.groups = static_cast<int64_t>(groups.size());
  if (member_sum != out.rows) {
    out.errors.push_back("sum of reconstructed members differs from rows");
  }
  out.valid = out.errors.empty();
  return out;
}

inline std::string PpoNumericalParityOldProbabilityBucket(
    const PpoNumericalParityRow& row) {
  if (row.schema_errors != 0 || row.nonfinite_values != 0 ||
      !std::isfinite(row.old_chosen_probability) ||
      row.old_chosen_probability < 0.0) {
    return "invalid";
  }
  if (row.old_chosen_probability < 1e-6) return "p_lt_1e-6";
  if (row.old_chosen_probability < 1e-4) return "p_1e-6_to_1e-4";
  if (row.old_chosen_probability < 1e-2) return "p_1e-4_to_1e-2";
  if (row.old_chosen_probability < 0.1) return "p_1e-2_to_0_1";
  if (row.old_chosen_probability < 0.5) return "p_0_1_to_0_5";
  return "p_ge_0_5";
}

struct PpoNumericalParitySummary {
  int64_t rows = 0;
  int64_t finite_rows = 0;
  int64_t legal_logits = 0;
  int64_t old_probability_underflows = 0;
  int64_t new_probability_underflows = 0;
  int64_t nonfinite_values = 0;
  int64_t schema_error_rows = 0;
  int64_t mass_residual_rows = 0;
  double mean_old_raw_mass_residual = 0.0;
  double max_old_raw_mass_residual = 0.0;
  double mean_new_raw_mass_residual = 0.0;
  double max_new_raw_mass_residual = 0.0;
  double mean_abs_chosen_log_prob_delta = 0.0;
  double max_abs_chosen_log_prob_delta = 0.0;
  double mean_kl_old_new = 0.0;
  double max_kl_old_new = 0.0;
  double mean_kl_new_old = 0.0;
  double max_kl_new_old = 0.0;
  double ratio_min = 1.0;
  double ratio_p01 = 1.0;
  double ratio_p50 = 1.0;
  double ratio_p99 = 1.0;
  double ratio_max = 1.0;
  int64_t ratio_lt_0_8 = 0;
  int64_t ratio_gt_1_2 = 0;
};

inline bool IsNegativeInfinity(double x) {
  return std::isinf(x) && x < 0.0;
}

inline bool LogSumExpFiniteOrNegativeInfinity(
    const std::vector<double>& values, double* out) {
  double maximum = -std::numeric_limits<double>::infinity();
  for (double value : values) {
    if ((!std::isfinite(value) && !IsNegativeInfinity(value))) return false;
    maximum = std::max(maximum, value);
  }
  if (!std::isfinite(maximum)) return false;  // empty support / all -inf
  double shifted_sum = 0.0;
  for (double value : values) {
    if (!IsNegativeInfinity(value)) shifted_sum += std::exp(value - maximum);
  }
  if (!(shifted_sum > 0.0) || !std::isfinite(shifted_sum)) return false;
  *out = maximum + std::log(shifted_sum);
  return std::isfinite(*out);
}

// Computes the selected-action ratio from the raw values actually consumed by
// PPO, while computing both directions of the FULL legal categorical KL after
// independently normalizing old and new float-stored log-probability vectors
// with double logsumexp. The raw mass residual remains in the row for a
// separate fail-closed gate; normalization repairs KL arithmetic but never
// launders malformed input. On every error `out` is still populated with a
// schema-error row so the caller can retain, count and split it rather than
// silently dropping evidence.
inline bool ComputePpoNumericalParityRow(
    const PpoNumericalParityInput& in, PpoNumericalParityRow* out,
    std::string* error) {
  if (out == nullptr) {
    if (error != nullptr) *error = "null output row";
    return false;
  }
  PpoNumericalParityRow row;
  row.legal_count = in.legal_count;
  row.decision_role = in.decision_role;
  row.advantage = in.advantage;
  auto fail = [&](const std::string& message) {
    if (error != nullptr) *error = message;
    row.schema_errors = 1;
    row.nonfinite_values = std::max<int64_t>(1, row.nonfinite_values);
    *out = row;
    return false;
  };
  if (in.legal_count <= 0 ||
      static_cast<int>(in.old_log_probs.size()) != in.legal_count ||
      static_cast<int>(in.new_log_probs.size()) != in.legal_count) {
    return fail("legal-count/distribution-size mismatch");
  }
  if (in.chosen_index < 0 || in.chosen_index >= in.legal_count) {
    return fail("chosen action is absent from the legal-action order");
  }
  if (!std::isfinite(in.advantage) ||
      !std::isfinite(in.stored_chosen_log_prob)) {
    return fail("nonfinite advantage or stored chosen log-probability");
  }

  // Integrity is against the RAW behavior vector, before normalization.
  const double old_chosen_lp = in.old_log_probs[in.chosen_index];
  const double new_chosen_lp = in.new_log_probs[in.chosen_index];
  if (!std::isfinite(old_chosen_lp) || !std::isfinite(new_chosen_lp)) {
    return fail("chosen action has zero or nonfinite probability");
  }
  if (std::abs(old_chosen_lp - in.stored_chosen_log_prob) > 2e-6) {
    return fail("stored chosen log-probability disagrees with raw stored legal distribution");
  }

  double old_log_normalizer = 0.0;
  double new_log_normalizer = 0.0;
  if (!LogSumExpFiniteOrNegativeInfinity(in.old_log_probs,
                                         &old_log_normalizer) ||
      !LogSumExpFiniteOrNegativeInfinity(in.new_log_probs,
                                         &new_log_normalizer)) {
    return fail("legal distribution has empty, NaN, or positive-infinite support");
  }
  const double old_raw_mass = std::exp(old_log_normalizer);
  const double new_raw_mass = std::exp(new_log_normalizer);
  if (!std::isfinite(old_raw_mass) || !std::isfinite(new_raw_mass)) {
    return fail("raw legal distribution mass is nonfinite");
  }
  row.old_raw_mass_residual = std::abs(old_raw_mass - 1.0);
  row.new_raw_mass_residual = std::abs(new_raw_mass - 1.0);
  row.mass_residual_valid = true;

  for (int i = 0; i < in.legal_count; ++i) {
    const double old_lp = in.old_log_probs[i];
    const double new_lp = in.new_log_probs[i];
    if ((!std::isfinite(old_lp) && !IsNegativeInfinity(old_lp)) ||
        (!std::isfinite(new_lp) && !IsNegativeInfinity(new_lp))) {
      return fail("legal distribution contains NaN or positive infinity");
    }
    const double old_p_raw =
        IsNegativeInfinity(old_lp) ? 0.0 : std::exp(old_lp);
    const double new_p_raw =
        IsNegativeInfinity(new_lp) ? 0.0 : std::exp(new_lp);
    const double old_p = IsNegativeInfinity(old_lp)
                             ? 0.0
                             : std::exp(old_lp - old_log_normalizer);
    const double new_p = IsNegativeInfinity(new_lp)
                             ? 0.0
                             : std::exp(new_lp - new_log_normalizer);
    if (!std::isfinite(old_p) || !std::isfinite(new_p) || old_p < 0.0 ||
        new_p < 0.0 || !std::isfinite(old_p_raw) ||
        !std::isfinite(new_p_raw)) {
      return fail("legal distribution exponentiation is nonfinite");
    }
    if (old_p_raw == 0.0) ++row.old_probability_underflows;
    if (new_p_raw == 0.0) ++row.new_probability_underflows;

    if (old_p > 0.0) {
      if (new_p == 0.0) {
        row.kl_old_new = std::numeric_limits<double>::infinity();
      } else if (std::isfinite(row.kl_old_new)) {
        row.kl_old_new += old_p *
            ((old_lp - old_log_normalizer) -
             (new_lp - new_log_normalizer));
      }
    }
    if (new_p > 0.0) {
      if (old_p == 0.0) {
        row.kl_new_old = std::numeric_limits<double>::infinity();
      } else if (std::isfinite(row.kl_new_old)) {
        row.kl_new_old += new_p *
            ((new_lp - new_log_normalizer) -
             (old_lp - old_log_normalizer));
      }
    }
  }
  if (row.kl_old_new < -1e-12 || row.kl_new_old < -1e-12) {
    return fail("categorical KL is materially negative");
  }
  row.kl_old_new = std::max(0.0, row.kl_old_new);
  row.kl_new_old = std::max(0.0, row.kl_new_old);

  row.old_chosen_probability = std::exp(in.stored_chosen_log_prob);
  row.chosen_log_prob_delta = new_chosen_lp - in.stored_chosen_log_prob;
  row.ratio = std::exp(row.chosen_log_prob_delta);
  if (!std::isfinite(row.old_chosen_probability) ||
      !std::isfinite(row.chosen_log_prob_delta) ||
      !std::isfinite(row.ratio) || !std::isfinite(row.kl_old_new) ||
      !std::isfinite(row.kl_new_old)) {
    ++row.nonfinite_values;
  }
  *out = row;
  return true;
}

inline double PpoParityQuantile(const std::vector<double>& sorted,
                                double fraction) {
  if (sorted.empty()) return 1.0;
  const double bounded = std::max(0.0, std::min(1.0, fraction));
  const size_t index = static_cast<size_t>(
      std::floor(bounded * static_cast<double>(sorted.size() - 1)));
  return sorted[index];
}

inline PpoNumericalParitySummary SummarizePpoNumericalParityRows(
    const std::vector<PpoNumericalParityRow>& rows) {
  PpoNumericalParitySummary out;
  out.rows = static_cast<int64_t>(rows.size());
  if (rows.empty()) return out;
  std::vector<double> ratios;
  ratios.reserve(rows.size());
  double abs_delta_sum = 0.0;
  double kl_old_new_sum = 0.0;
  double kl_new_old_sum = 0.0;
  double old_mass_residual_sum = 0.0;
  double new_mass_residual_sum = 0.0;
  for (const auto& row : rows) {
    out.legal_logits += row.legal_count;
    out.old_probability_underflows += row.old_probability_underflows;
    out.new_probability_underflows += row.new_probability_underflows;
    out.nonfinite_values += row.nonfinite_values;
    out.schema_error_rows += row.schema_errors != 0 ? 1 : 0;
    if (row.mass_residual_valid &&
        std::isfinite(row.old_raw_mass_residual) &&
        std::isfinite(row.new_raw_mass_residual)) {
      ++out.mass_residual_rows;
      old_mass_residual_sum += row.old_raw_mass_residual;
      new_mass_residual_sum += row.new_raw_mass_residual;
      out.max_old_raw_mass_residual = std::max(
          out.max_old_raw_mass_residual, row.old_raw_mass_residual);
      out.max_new_raw_mass_residual = std::max(
          out.max_new_raw_mass_residual, row.new_raw_mass_residual);
    }
    if (row.schema_errors != 0 || row.nonfinite_values != 0 ||
        !std::isfinite(row.chosen_log_prob_delta) ||
        !std::isfinite(row.ratio) || !std::isfinite(row.kl_old_new) ||
        !std::isfinite(row.kl_new_old)) {
      continue;
    }
    ++out.finite_rows;
    const double abs_delta = std::abs(row.chosen_log_prob_delta);
    abs_delta_sum += abs_delta;
    kl_old_new_sum += row.kl_old_new;
    kl_new_old_sum += row.kl_new_old;
    out.max_abs_chosen_log_prob_delta =
        std::max(out.max_abs_chosen_log_prob_delta, abs_delta);
    out.max_kl_old_new = std::max(out.max_kl_old_new, row.kl_old_new);
    out.max_kl_new_old = std::max(out.max_kl_new_old, row.kl_new_old);
    if (row.ratio < 0.8) ++out.ratio_lt_0_8;
    if (row.ratio > 1.2) ++out.ratio_gt_1_2;
    ratios.push_back(row.ratio);
  }
  if (out.mass_residual_rows > 0) {
    out.mean_old_raw_mass_residual =
        old_mass_residual_sum / out.mass_residual_rows;
    out.mean_new_raw_mass_residual =
        new_mass_residual_sum / out.mass_residual_rows;
  }
  if (out.finite_rows == 0) return out;
  out.mean_abs_chosen_log_prob_delta = abs_delta_sum / out.finite_rows;
  out.mean_kl_old_new = kl_old_new_sum / out.finite_rows;
  out.mean_kl_new_old = kl_new_old_sum / out.finite_rows;
  std::sort(ratios.begin(), ratios.end());
  out.ratio_min = ratios.front();
  out.ratio_p01 = PpoParityQuantile(ratios, 0.01);
  out.ratio_p50 = PpoParityQuantile(ratios, 0.50);
  out.ratio_p99 = PpoParityQuantile(ratios, 0.99);
  out.ratio_max = ratios.back();
  return out;
}

inline bool PpoNumericalParityRawMassWithinBound(
    const PpoNumericalParitySummary& summary, double bound) {
  return std::isfinite(bound) && bound >= 0.0 &&
         summary.mass_residual_rows == summary.rows &&
         summary.max_old_raw_mass_residual <= bound &&
         summary.max_new_raw_mass_residual <= bound;
}

}  // namespace open_spiel

#endif  // OPEN_SPIEL_EXAMPLES_DUNE_PPO_NUMERICAL_PARITY_H_
