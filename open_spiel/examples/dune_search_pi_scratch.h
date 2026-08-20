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
inline constexpr char kWarmSearchPiProfile[] = "ppo_warm_q_v1";

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
