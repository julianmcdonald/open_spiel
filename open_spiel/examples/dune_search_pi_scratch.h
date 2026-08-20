#ifndef OPEN_SPIEL_EXAMPLES_DUNE_SEARCH_PI_SCRATCH_H_
#define OPEN_SPIEL_EXAMPLES_DUNE_SEARCH_PI_SCRATCH_H_

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <torch/torch.h>

#include "dune_network.h"
#include "dune_search_pi.h"

namespace open_spiel {

inline constexpr char kScratchSearchPiProfile[] = "scratch_q_v1";
inline constexpr char kScratchVisitSearchPiProfile[] = "scratch_visit_v1";
inline constexpr char kWarmSearchPiProfile[] = "ppo_warm_q_v1";
inline constexpr char kWarmNormalizedCmpoProfile[] = "ppo_warm_cmpo_norm_v1";

struct NormalizedCmpoGenerationStats {
  double sigma = 1e-6;
  int64_t full_rows = 0;
  int64_t cheap_rows = 0;
  int64_t advantage_count = 0;
  int64_t clipped_low = 0;
  int64_t clipped_high = 0;
  std::vector<double> target_kls;
  double prior_expected_q_mean = 0.0;
  double target_expected_q_mean = 0.0;
  double q_range_mean = 0.0;
  double direct_visit_count_mean = 0.0;
  double raw_entropy_mean = 0.0;
  double target_entropy_mean = 0.0;
};

RegularizedQTargetResult BuildNormalizedCmpoTarget(
    const std::vector<Action>& legal_actions,
    const std::vector<double>& raw_prior, const std::vector<int>& visits,
    const std::vector<double>& q_values, double root_value, double sigma,
    double max_kl = 0.10);

bool ApplyNormalizedCmpoTargets(
    std::vector<SearchPiRow>* rows, NormalizedCmpoGenerationStats* stats,
    std::string* error);

struct ScratchVisitGenerationStats {
  int64_t full_rows = 0;
  int64_t cheap_rows = 0;
  double target_entropy_mean = 0.0;
  double target_kl_mean = 0.0;
  double prior_expected_q_mean = 0.0;
  double target_expected_q_mean = 0.0;
  double q_range_mean = 0.0;
  double direct_visit_count_mean = 0.0;
};

bool ApplyScratchVisitTargets(
    std::vector<SearchPiRow>* rows, ScratchVisitGenerationStats* stats,
    std::string* error);

std::vector<std::vector<SearchPiRow>> FilterNormalizedCmpoReplayWindow(
    const std::vector<std::vector<SearchPiRow>>& replay_window);

struct ScratchSearchPiLearnerStats {
  double policy_ce = 0.0;
  double value_mse = 0.0;
  double policy_grad_norm = 0.0;
  double value_grad_norm = 0.0;
  double shared_trunk_cosine = 0.0;
  bool shared_trunk_cosine_defined = false;
  bool policy_backward_executed = false;
  bool value_backward_executed = false;
  int64_t policy_backward_minibatches = 0;
  int64_t value_backward_minibatches = 0;
  int64_t distinct_rows = 0;
  int64_t policy_weight_rows = 0;
  int64_t presentations = 0;
  int64_t minibatches = 0;
  double value_target_mean = 0.0;
  double value_target_sd = 0.0;
  double critic_pred_mean_pre = 0.0;
  double critic_pred_sd_pre = 0.0;
  double critic_pred_mean_post = 0.0;
  double critic_pred_sd_post = 0.0;
  double critic_saturation_pre = 0.0;
  double critic_saturation_post = 0.0;
  double learner_precap_max_legal_logit = 0.0;
  double learner_postcap_max_legal_logit = 0.0;
  int64_t learner_logit_cap_decisions = 0;
  int64_t learner_logit_cap_legal_logits = 0;
  int64_t learner_logit_cap_saturated = 0;
};

ScratchSearchPiLearnerStats RunScratchSearchPiLearner(
    const std::shared_ptr<SharedDunePolicyValueNetImpl>& model,
    torch::optim::Optimizer& optimizer, const std::vector<SearchPiRow>& rows,
    int64_t obs_size, int64_t action_dim, torch::Device device,
    uint64_t master_seed, int generation, const SearchPiLearnerConfig& cfg);

void CopySearchPiModel(
    const std::shared_ptr<SharedDunePolicyValueNetImpl>& source,
    const std::shared_ptr<SharedDunePolicyValueNetImpl>& destination);

bool ScratchCollectorOwnedByOptimizer(
    const std::shared_ptr<SharedDunePolicyValueNetImpl>& collector,
    const torch::optim::Optimizer& optimizer);

}  // namespace open_spiel

#endif  // OPEN_SPIEL_EXAMPLES_DUNE_SEARCH_PI_SCRATCH_H_
