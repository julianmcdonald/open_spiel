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
// model replayed through the learner's non-autocast forward. Both vectors are
// aligned to the transition's legal-action order.
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
  double chosen_log_prob_delta = 0.0;  // learner FP32 - stored rollout
  double ratio = 1.0;
  double kl_old_new = 0.0;             // full legal KL(old || new)
  double kl_new_old = 0.0;             // full legal KL(new || old)
  int64_t old_probability_underflows = 0;
  int64_t new_probability_underflows = 0;
  int64_t nonfinite_values = 0;
};

struct PpoNumericalParitySummary {
  int64_t rows = 0;
  int64_t finite_rows = 0;
  int64_t legal_logits = 0;
  int64_t old_probability_underflows = 0;
  int64_t new_probability_underflows = 0;
  int64_t nonfinite_values = 0;
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

// Computes the selected-action ratio and both directions of the FULL legal
// categorical KL. It refuses malformed or unnormalised inputs instead of
// manufacturing a finite result. Exact-zero probabilities caused by exp
// underflow are counted; a support mismatch makes the corresponding KL
// infinite and therefore fails the diagnostic downstream.
inline bool ComputePpoNumericalParityRow(
    const PpoNumericalParityInput& in, PpoNumericalParityRow* out,
    std::string* error) {
  auto fail = [&](const std::string& message) {
    if (error != nullptr) *error = message;
    return false;
  };
  if (out == nullptr) return fail("null output row");
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

  PpoNumericalParityRow row;
  row.legal_count = in.legal_count;
  row.decision_role = in.decision_role;
  row.advantage = in.advantage;

  double old_mass = 0.0;
  double new_mass = 0.0;
  for (int i = 0; i < in.legal_count; ++i) {
    const double old_lp = in.old_log_probs[i];
    const double new_lp = in.new_log_probs[i];
    if ((!std::isfinite(old_lp) && !IsNegativeInfinity(old_lp)) ||
        (!std::isfinite(new_lp) && !IsNegativeInfinity(new_lp))) {
      return fail("legal distribution contains NaN or positive infinity");
    }
    const double old_p = IsNegativeInfinity(old_lp) ? 0.0 : std::exp(old_lp);
    const double new_p = IsNegativeInfinity(new_lp) ? 0.0 : std::exp(new_lp);
    if (!std::isfinite(old_p) || !std::isfinite(new_p) || old_p < 0.0 ||
        new_p < 0.0) {
      return fail("legal distribution exponentiation is nonfinite");
    }
    if (old_p == 0.0) ++row.old_probability_underflows;
    if (new_p == 0.0) ++row.new_probability_underflows;
    old_mass += old_p;
    new_mass += new_p;

    if (old_p > 0.0) {
      if (new_p == 0.0) {
        row.kl_old_new = std::numeric_limits<double>::infinity();
      } else if (std::isfinite(row.kl_old_new)) {
        row.kl_old_new += old_p * (old_lp - new_lp);
      }
    }
    if (new_p > 0.0) {
      if (old_p == 0.0) {
        row.kl_new_old = std::numeric_limits<double>::infinity();
      } else if (std::isfinite(row.kl_new_old)) {
        row.kl_new_old += new_p * (new_lp - old_lp);
      }
    }
  }
  // Float-stored log-probabilities should still normalise much more closely
  // than this. A wider error means this is not a categorical distribution.
  if (std::abs(old_mass - 1.0) > 1e-5 || std::abs(new_mass - 1.0) > 1e-5) {
    return fail("legal distribution does not normalise to one");
  }
  if (row.kl_old_new < -1e-7 || row.kl_new_old < -1e-7) {
    return fail("categorical KL is materially negative");
  }
  row.kl_old_new = std::max(0.0, row.kl_old_new);
  row.kl_new_old = std::max(0.0, row.kl_new_old);

  const double old_chosen_lp = in.old_log_probs[in.chosen_index];
  const double new_chosen_lp = in.new_log_probs[in.chosen_index];
  if (!std::isfinite(old_chosen_lp) || !std::isfinite(new_chosen_lp)) {
    return fail("chosen action has zero or nonfinite probability");
  }
  // This is a schema-integrity check, not a parity threshold: the stored scalar
  // and stored full distribution came from the same rollout decision.
  if (std::abs(old_chosen_lp - in.stored_chosen_log_prob) > 2e-6) {
    return fail("stored chosen log-probability disagrees with stored legal distribution");
  }
  row.old_chosen_probability = std::exp(old_chosen_lp);
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
  for (const auto& row : rows) {
    out.legal_logits += row.legal_count;
    out.old_probability_underflows += row.old_probability_underflows;
    out.new_probability_underflows += row.new_probability_underflows;
    out.nonfinite_values += row.nonfinite_values;
    if (row.nonfinite_values != 0 ||
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

}  // namespace open_spiel

#endif  // OPEN_SPIEL_EXAMPLES_DUNE_PPO_NUMERICAL_PARITY_H_
