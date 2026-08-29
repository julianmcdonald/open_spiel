#ifndef OPEN_SPIEL_EXAMPLES_DUNE_PPO_NUMERICAL_PARITY_H_
#define OPEN_SPIEL_EXAMPLES_DUNE_PPO_NUMERICAL_PARITY_H_

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <utility>
#include <vector>

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
};

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
