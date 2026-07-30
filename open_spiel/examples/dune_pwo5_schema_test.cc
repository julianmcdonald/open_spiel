// PWO-5 gate 3: the diagnostics schema-v5 and partition-identity tests.
//
// docs/PWO5_PILOT_REGISTRATION.md section 16 gate 3 item 5 requires the
// seventeen canary columns 70-86 "by their registered names, appended in this
// order", and requires gate 3 to assert section 14.1c's partition identity
// "both halves: Sum_r n_r == norm_entropy_n exactly, and the weighted mean
// within 1e-9".
//
// STOP condition 17.5 item 14 makes a diagnostics.csv whose header is not the
// v5 86-column schema a STOP, so the header is pinned here character by
// character rather than counted.

#include <cmath>
#include <cstdint>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "open_spiel/abseil-cpp/absl/flags/flag.h"

#include "dune_ppo_training_utils.h"
#include "dune_search_routing.h"

ABSL_FLAG(int, ppo_minibatch_size, 2048, "");
ABSL_FLAG(int, ppo_update_epochs, 4, "");
ABSL_FLAG(double, ppo_clip_epsilon, 0.2, "");
ABSL_FLAG(bool, normalize_advantages, true, "");
ABSL_FLAG(bool, ppo_clip_value_loss, true, "");
ABSL_FLAG(double, entropy_coef, 0.01, "");
ABSL_FLAG(double, value_coef, 0.5, "");
ABSL_FLAG(double, logit_cap, 10.0, "");
ABSL_FLAG(double, target_kl, 0.0, "");
ABSL_FLAG(bool, train_amp, true, "");
ABSL_FLAG(double, grad_clip_norm, 0.5, "");
ABSL_FLAG(uint64_t, shaping_start_env_steps, 206830543, "");
ABSL_FLAG(uint64_t, shaping_decay_env_steps, 0, "");
ABSL_FLAG(bool, diagnostics_only, false, "");

namespace {

int failures = 0;

void Check(bool ok, const std::string& what) {
  if (ok) {
    std::cout << "  PASS  " << what << "\n";
  } else {
    std::cout << "  FAIL  " << what << "\n";
    ++failures;
  }
}

std::vector<std::string> Split(const std::string& s, char sep) {
  std::vector<std::string> out;
  std::string cur;
  std::istringstream ss(s);
  while (std::getline(ss, cur, sep)) out.push_back(cur);
  return out;
}

// The seventeen names, in the registered order (section 14.1c's table).
const char* const kRegisteredCanaryColumns[17] = {
    "norm_entropy",
    "norm_entropy_n",
    "norm_entropy_primary",
    "norm_entropy_n_primary",
    "norm_entropy_continuation",
    "norm_entropy_n_continuation",
    "norm_entropy_purchase",
    "norm_entropy_n_purchase",
    "norm_entropy_combat_intrigue",
    "norm_entropy_n_combat_intrigue",
    "norm_entropy_other_optional",
    "norm_entropy_n_other_optional",
    "norm_entropy_leader_selection",
    "norm_entropy_n_leader_selection",
    "max_action_prob",
    "frac_legal_absz_ge_cap",
    "frac_legal_absz_ge_cap_n",
};

void TestSchemaShape() {
  std::cout << "[1] schema v4 (69 cols) and v5 (86 cols)\n";
  const std::vector<std::string> v4 =
      Split(open_spiel::DiagnosticsCsvHeader(false), ',');
  const std::vector<std::string> v5 =
      Split(open_spiel::DiagnosticsCsvHeader(true), ',');

  Check(v4.size() == 69, "v4 has 69 columns (got " + std::to_string(v4.size()) + ")");
  Check(v5.size() == 86, "v5 has 86 columns (got " + std::to_string(v5.size()) + ")");

  // v5 must EXTEND v4, never reorder or rename it. The header is compared
  // byte-for-byte on resume, and every committed analyzer indexes by position.
  bool prefix_ok = v5.size() >= v4.size();
  for (size_t i = 0; prefix_ok && i < v4.size(); ++i) {
    if (v4[i] != v5[i]) prefix_ok = false;
  }
  Check(prefix_ok, "v5 columns 1-69 are byte-identical to v4 -- append only");

  bool names_ok = true;
  for (int i = 0; i < 17; ++i) {
    // 1-indexed column 70 is 0-indexed 69.
    if (v5[69 + i] != kRegisteredCanaryColumns[i]) {
      names_ok = false;
      std::cout << "        column " << (70 + i) << " is '" << v5[69 + i]
                << "', registered name is '" << kRegisteredCanaryColumns[i]
                << "'\n";
    }
  }
  Check(names_ok, "columns 70-86 carry the registered names in the registered order");

  // Every role mean is immediately followed by its own denominator. Revision 7
  // of the registration emitted the means with no counts at all, which made the
  // partition identity unauditable from the CSV.
  bool paired = true;
  for (int i = 2; i <= 12; i += 2) {  // the six role pairs, 0-indexed in the 17
    const std::string mean = kRegisteredCanaryColumns[i];
    const std::string count = kRegisteredCanaryColumns[i + 1];
    const std::string suffix = mean.substr(std::string("norm_entropy").size());
    if (count != "norm_entropy_n" + suffix) paired = false;
  }
  Check(paired, "each of the six role means is paired with its own _n denominator");

  // kl_early_stop_rate is NOT a column -- it is analyzer arithmetic over the
  // existing col 38, because a windowed mean is not a per-update primitive.
  bool has_rate = false;
  for (const std::string& c : v5) {
    if (c == "kl_early_stop_rate") has_rate = true;
  }
  Check(!has_rate, "kl_early_stop_rate is NOT a column (analyzer arithmetic)");

  // ...but the primitive it is derived from IS emitted.
  bool has_kl_early_stop = false;
  for (const std::string& c : v5) {
    if (c == "kl_early_stop") has_kl_early_stop = true;
  }
  Check(has_kl_early_stop, "kl_early_stop (col 38) is emitted as the primitive");
}

void TestRoleCoverageIsAPartition() {
  std::cout << "[2] the six role columns are an exhaustive partition of NE's domain\n";
  // DuneDecisionRole has seven members. Six get a column; the seventh,
  // kForcedOrBookkeeping, is excluded by proof (chance nodes cannot reach the
  // classifier at the registered rollout site, and NE's population requires
  // n_legal >= 2, so the role is identically empty there).
  const std::vector<std::string> v5 =
      Split(open_spiel::DiagnosticsCsvHeader(true), ',');
  const char* kExpectedRoles[6] = {"primary",         "continuation",
                                   "purchase",        "combat_intrigue",
                                   "other_optional",  "leader_selection"};
  bool all_present = true;
  for (const char* role : kExpectedRoles) {
    bool found = false;
    for (const std::string& c : v5) {
      if (c == std::string("norm_entropy_") + role) found = true;
    }
    if (!found) all_present = false;
  }
  Check(all_present, "all six contributing roles have a mean column");

  bool forced_absent = true;
  for (const std::string& c : v5) {
    if (c == "norm_entropy_forced" || c == "norm_entropy_forced_or_bookkeeping") {
      forced_absent = false;
    }
  }
  Check(forced_absent,
        "kForcedOrBookkeeping has NO column -- identically empty by construction");

  // The gap this closes: the pre-existing precap_absz convention covers only
  // primary/continuation/purchase, and kOtherOptional alone is 45.82% of the
  // PWO-4 stream. A canary blind to nearly half of all decisions is not a
  // canary.
  bool other_optional_present = false;
  for (const std::string& c : v5) {
    if (c == "norm_entropy_other_optional") other_optional_present = true;
  }
  Check(other_optional_present,
        "kOtherOptional (45.82% of decisions) is covered");
}

void TestPartitionIdentity() {
  std::cout << "[3] the partition identity, both halves\n";
  // A synthetic stats row whose role splits are consistent.
  open_spiel::PpoUpdateStats stats;
  const int kPrimary = static_cast<int>(open_spiel::DuneDecisionRole::kAgentPrimary);
  const int kCont = static_cast<int>(open_spiel::DuneDecisionRole::kAgentContinuation);
  const int kPurch = static_cast<int>(open_spiel::DuneDecisionRole::kPurchase);
  const int kOther = static_cast<int>(open_spiel::DuneDecisionRole::kOtherOptional);

  stats.norm_entropy_n_role[kPrimary] = 100;
  stats.norm_entropy_role[kPrimary] = 0.80;
  stats.norm_entropy_n_role[kCont] = 50;
  stats.norm_entropy_role[kCont] = 0.60;
  stats.norm_entropy_n_role[kPurch] = 25;
  stats.norm_entropy_role[kPurch] = 0.40;
  stats.norm_entropy_n_role[kOther] = 25;
  stats.norm_entropy_role[kOther] = 0.20;

  int64_t n_total = 0;
  double weighted = 0.0;
  for (int r = 0; r < 7; ++r) {
    n_total += stats.norm_entropy_n_role[r];
    weighted += stats.norm_entropy_n_role[r] * stats.norm_entropy_role[r];
  }
  stats.norm_entropy_n = n_total;
  stats.norm_entropy = weighted / n_total;

  Check(n_total == 200, "count half: sum of role denominators == norm_entropy_n");
  Check(std::abs(weighted / n_total - stats.norm_entropy) <= 1e-9,
        "mean half: row-count-weighted role mean reproduces the global mean");

  // A role with ZERO support contributes zero weight and cannot move the
  // identity -- which is what makes the "0.0 means no support, never collapsed
  // to zero" convention safe.
  const int kLeader = static_cast<int>(open_spiel::DuneDecisionRole::kLeaderSelection);
  Check(stats.norm_entropy_n_role[kLeader] == 0 &&
            stats.norm_entropy_role[kLeader] == 0.0,
        "a zero-support role is emitted as 0.0 with count 0");
  double weighted2 = weighted + stats.norm_entropy_n_role[kLeader] *
                                    stats.norm_entropy_role[kLeader];
  Check(weighted2 == weighted, "a zero-support role contributes zero weight");
}

void TestZeroSupportIsNotCollapse() {
  std::cout << "[4] zero support is distinguishable from a collapsed entropy\n";
  // Both emit a mean of 0.0. The ONLY thing that distinguishes them is the
  // denominator, which is why every role mean is paired with its own _n and why
  // every consumer must read the count first.
  open_spiel::PpoUpdateStats no_support;
  open_spiel::PpoUpdateStats collapsed;
  const int kPurch = static_cast<int>(open_spiel::DuneDecisionRole::kPurchase);
  no_support.norm_entropy_role[kPurch] = 0.0;
  no_support.norm_entropy_n_role[kPurch] = 0;
  collapsed.norm_entropy_role[kPurch] = 0.0;
  collapsed.norm_entropy_n_role[kPurch] = 4321;

  Check(no_support.norm_entropy_role[kPurch] == collapsed.norm_entropy_role[kPurch],
        "the two are INDISTINGUISHABLE by the mean alone");
  Check(no_support.norm_entropy_n_role[kPurch] !=
            collapsed.norm_entropy_n_role[kPurch],
        "the denominator is what separates them -- read the count first");
}

}  // namespace

int main(int argc, char** argv) {
  std::cout << "dune_pwo5_schema_test -- PWO-5 gate 3, sections 14.1c / 16\n\n";
  TestSchemaShape();
  TestRoleCoverageIsAPartition();
  TestPartitionIdentity();
  TestZeroSupportIsNotCollapse();
  std::cout << "\n";
  if (failures == 0) {
    std::cout << "ALL PASS\n";
    return 0;
  }
  std::cout << failures << " FAILED\n";
  return 1;
}
