// Frozen offline changed-state weighting experiment helpers.
//
// This lane is intentionally separate from dune_ppo_train.  It consumes only
// retained SearchPiRow shards, never collects games, and never restores an
// optimizer checkpoint.

#ifndef OPEN_SPIEL_EXAMPLES_DUNE_OFFLINE_CHANGED_STATE_H_
#define OPEN_SPIEL_EXAMPLES_DUNE_OFFLINE_CHANGED_STATE_H_

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <torch/torch.h>

#include "dune_network.h"
#include "dune_search_pi.h"

namespace open_spiel {

inline constexpr int64_t kOfflineChangedStateRows = 92379;
inline constexpr uint64_t kOfflineChangedStateSeed = 8260001;
inline constexpr int64_t kOfflineChangedStateBatchSize = 256;
inline constexpr double kOfflineChangedStateLearningRate = 1e-5;
inline constexpr double kOfflineChangedStateAdamEps = 1e-5;
inline constexpr double kOfflineChangedStateGradientClip = 0.5;
inline constexpr double kOfflineChangedStateLogitCap = 10.0;
inline constexpr double kOfflineChangedStateWeightDecay = 0.0;
inline constexpr double kOfflineChangedStatePolicyWeightDecay = 0.0;

struct OfflineRoleSpec {
  DuneDecisionRole role;
  const char* name;
  int64_t total;
  int64_t changed;
  int64_t preserved;
};

const std::array<OfflineRoleSpec, 5>& RegisteredOfflineRoleSpecs();
const char* OfflineRoleName(DuneDecisionRole role);

struct OfflineRoleWeightSummary {
  OfflineRoleSpec spec;
  int64_t parsed_total = 0;
  int64_t parsed_changed = 0;
  int64_t parsed_preserved = 0;
  double changed_weight = 0.0;
  double preserved_weight = 0.0;
  double changed_loss_mass = 0.0;
  double preserved_loss_mass = 0.0;
  double total_loss_mass = 0.0;
};

struct OfflineWeightPlan {
  std::vector<double> control;
  std::vector<double> treatment;
  std::vector<OfflineRoleWeightSummary> roles;
  double control_mass = 0.0;
  double treatment_mass = 0.0;
  double extended_precision_sequential_treatment_mass = 0.0;
  double binary64_sequential_treatment_mass = 0.0;
};

bool BuildOfflineWeightPlan(const std::vector<SearchPiRow>& rows,
                            const std::vector<OfflineRoleSpec>& specs,
                            OfflineWeightPlan* out, std::string* error);

struct OfflineBatchBoundary {
  int64_t start = 0;
  int64_t count = 0;
};

std::vector<OfflineBatchBoundary> BuildOfflineBatchBoundaries(
    int64_t row_count, int64_t batch_size);

std::string OfflinePresentationDigest(
    const std::string& extended_row_hash,
    const std::vector<OfflineBatchBoundary>& boundaries, uint64_t seed);

std::string CanonicalOfflineModuleDigest(torch::nn::Module& model);

struct OfflineAdamWGroupContract {
  std::vector<std::string> parameter_names;
  double learning_rate = 0.0;
  double beta1 = 0.0;
  double beta2 = 0.0;
  double epsilon = 0.0;
  double weight_decay = 0.0;
  bool amsgrad = false;
};

struct OfflineAdamWContract {
  std::vector<OfflineAdamWGroupContract> groups;
};

std::unique_ptr<torch::optim::AdamW> MakeOfflineFreshAdamW(
    const std::shared_ptr<SharedDunePolicyValueNetImpl>& model,
    double learning_rate = kOfflineChangedStateLearningRate,
    double epsilon = kOfflineChangedStateAdamEps,
    double policy_weight_decay = kOfflineChangedStatePolicyWeightDecay,
    double other_weight_decay = kOfflineChangedStateWeightDecay);

OfflineAdamWContract CaptureOfflineAdamWContract(
    const SharedDunePolicyValueNetImpl& model,
    const torch::optim::AdamW& optimizer);

bool OfflineAdamWContractsEqual(const OfflineAdamWContract& lhs,
                                const OfflineAdamWContract& rhs,
                                std::string* error);

bool ValidateFreshOfflineBoundary(
    SharedDunePolicyValueNetImpl& control_model,
    const torch::optim::AdamW& control_optimizer,
    SharedDunePolicyValueNetImpl& treatment_model,
    const torch::optim::AdamW& treatment_optimizer,
    const std::string& expected_start_digest, std::string* error);

bool RefuseOfflineOptimizerCheckpoint(const std::string& path,
                                      std::string* error);

struct OfflineTrainingStats {
  int64_t presentations = 0;
  int64_t minibatches = 0;
  int64_t final_minibatch_size = 0;
  double presentation_weight_mass = 0.0;
  double weighted_ce_sum = 0.0;
  double mean_presented_weighted_ce = 0.0;
};

OfflineTrainingStats TrainOfflineChangedStateArm(
    const std::shared_ptr<SharedDunePolicyValueNetImpl>& model,
    torch::optim::AdamW& optimizer, const std::vector<SearchPiRow>& rows,
    const std::vector<double>& weights,
    const std::vector<OfflineBatchBoundary>& boundaries, int64_t obs_size,
    int64_t action_dim, torch::Device device, uint64_t seed,
    double gradient_clip = kOfflineChangedStateGradientClip,
    double logit_cap = kOfflineChangedStateLogitCap);

struct OfflineModelMetrics {
  std::vector<double> ce;
  std::vector<uint8_t> adopted;
};

OfflineModelMetrics EvaluateOfflineChangedStateRows(
    const std::shared_ptr<SharedDunePolicyValueNetImpl>& model,
    const std::vector<SearchPiRow>& rows,
    const std::vector<OfflineBatchBoundary>& boundaries, int64_t obs_size,
    int64_t action_dim, torch::Device device,
    double logit_cap = kOfflineChangedStateLogitCap);

struct OfflineMeanInterval {
  int64_t n = 0;
  double mean = 0.0;
  double sd = 0.0;
  double se = 0.0;
  double lower = 0.0;
  double upper = 0.0;
};

struct OfflineMetricLevel {
  int64_t n = 0;
  double ce_mean = 0.0;
  int64_t adopted = 0;
  double adoption = 0.0;
};

struct OfflinePairedComparison {
  OfflineMeanInterval ce_t_minus_c;
  OfflineMeanInterval adoption_t_minus_c;
  int64_t adoption_gained = 0;
  int64_t adoption_lost = 0;
  int64_t adoption_discordant = 0;
  double adoption_one_sided_exact_p = 1.0;
};

struct OfflineRoleGateResult {
  std::string role;
  OfflinePairedComparison changed;
  bool lower_changed_ce = false;
};

struct OfflineGateResult {
  OfflineMetricLevel parent_all;
  OfflineMetricLevel control_all;
  OfflineMetricLevel treatment_all;
  OfflineMetricLevel parent_changed;
  OfflineMetricLevel control_changed;
  OfflineMetricLevel treatment_changed;
  OfflineMetricLevel parent_preserved;
  OfflineMetricLevel control_preserved;
  OfflineMetricLevel treatment_preserved;
  OfflinePairedComparison changed;
  OfflinePairedComparison preserved;
  std::vector<OfflineRoleGateResult> roles;
  bool condition_1_changed_fit_and_adoption = false;
  bool condition_2_all_roles_lower_changed_ce = false;
  bool condition_3_preserved_noninferiority = false;
  bool condition_4_integrity_assertions = false;
  bool passed = false;
};

OfflineGateResult ApplyOfflineChangedStateGate(
    const std::vector<SearchPiRow>& rows,
    const OfflineModelMetrics& parent,
    const OfflineModelMetrics& control_child,
    const OfflineModelMetrics& treatment_child,
    bool integrity_assertions_passed);

double OneSidedPairedExactP(int64_t gained, int64_t lost);

}  // namespace open_spiel

#endif  // OPEN_SPIEL_EXAMPLES_DUNE_OFFLINE_CHANGED_STATE_H_
