#ifndef OPEN_SPIEL_EXAMPLES_DUNE_ACTION_VALUE_STUDY_H_
#define OPEN_SPIEL_EXAMPLES_DUNE_ACTION_VALUE_STUDY_H_

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <random>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "open_spiel/abseil-cpp/absl/strings/str_format.h"
#include "open_spiel/games/dune_imperium/dune_imperium.h"
#include "open_spiel/spiel.h"
#include "open_spiel/utils/json.h"

#ifdef OPEN_SPIEL_BUILD_WITH_LIBTORCH
#include <torch/torch.h>
#include "dune_network.h"
#include "dune_pwo2_common.h"
#include "dune_search_routing.h"
#include "dune_seed_utils.h"
#include "dune_sha256.h"
#include "dune_vrpo.h"
#endif

namespace open_spiel {
namespace action_value_study {

// ===========================================================================
// Domain and Stream Constants for Strict Domain Separation
// ===========================================================================
constexpr uint64_t kDomainPilot = 0x00A1;
constexpr uint64_t kDomainTrain = 0x00A2;
constexpr uint64_t kDomainDev   = 0x00A3;
constexpr uint64_t kDomainTest  = 0x00A4;

constexpr uint64_t kStreamSourceGame              = 0x0011;
constexpr uint64_t kStreamCandidateActionSample   = 0x0051;
constexpr uint64_t kStreamContinuationChance      = 0x00A1;
constexpr uint64_t kStreamContinuationPolicy      = 0x00A2;
constexpr uint64_t kStreamCriticInit              = 0x0071;
constexpr uint64_t kStreamBootstrap               = 0x0081;

// ===========================================================================
// Study Dimensions and Specifications
// ===========================================================================
inline constexpr int kNumPlayers = 4;
inline constexpr int kActorInputDim = 5580;
inline constexpr int kActorHiddenDim = 2048;
inline constexpr int kActorNumBlocks = 8;
inline constexpr int kActionDim = 2391;

inline constexpr int kPrivilegedCurrentPosDim = 9012;
inline constexpr int kMarketAppendixDim = 635;
inline constexpr int kCriticInputDim = 9647; // 9012 + 635
inline constexpr int kCriticHiddenDim = 512;
inline constexpr int kCriticNumResBlocks = 2;

inline constexpr uint64_t kRegisteredCriticInitSeed = 19;
inline constexpr double kCriticLearningRate = 2.5e-4;
inline constexpr double kCriticAdamWEpsilon = 1e-5;
inline constexpr double kCriticWeightDecay = 0.0;
inline constexpr double kCriticGradClipNorm = 1.0;
inline constexpr int kCriticMinibatchSizeRoots = 128;
inline constexpr int kCriticTotalEpochs = 100;
inline constexpr int kCriticEvalIntervalEpochs = 5;

inline constexpr int kRootsTrain = 2048;
inline constexpr int kRootsDev = 256;
inline constexpr int kRootsTest = 512;
inline constexpr int kRootsTotal = kRootsTrain + kRootsDev + kRootsTest; // 2816

inline constexpr int kContinuationsTrain = 16;
inline constexpr int kContinuationsDev = 32;
inline constexpr int kContinuationsTest = 64;

inline constexpr double kUtilityDivisor = 4.0;
inline constexpr int kBootstrapResamples = 10000;
inline constexpr uint64_t kBootstrapSeed = 20260911;

inline constexpr int64_t kTotalDeadlineSeconds = 28800; // 8 hours
inline constexpr int64_t kReportingReserveSeconds = 1800; // 30 minutes

inline const char kExpectedActorSha256[] =
    "68febee771509f88446286cc50983afd22d64f7772f6866def77bafd4aae36d2";
inline const char kActorModelPathDefault[] =
    "/home/warcr/projects/dune_drl/calibration_results_v2/pf_c_run2/s1_ctl_b/ppo_model_update_15828.pt";

// ===========================================================================
// Data Structures
// ===========================================================================

enum class Partition {
  kPilot,
  kTrain,
  kDev,
  kTest
};

inline const char* PartitionToString(Partition p) {
  switch (p) {
    case Partition::kPilot: return "pilot";
    case Partition::kTrain: return "train";
    case Partition::kDev:   return "dev";
    case Partition::kTest:  return "test";
  }
  return "unknown";
}

inline Partition StringToPartition(const std::string& str) {
  if (str == "pilot") return Partition::kPilot;
  if (str == "train") return Partition::kTrain;
  if (str == "dev") return Partition::kDev;
  if (str == "test") return Partition::kTest;
  SpielFatalError("Unknown partition: " + str);
}

inline uint64_t PartitionDomain(Partition p) {
  switch (p) {
    case Partition::kPilot: return kDomainPilot;
    case Partition::kTrain: return kDomainTrain;
    case Partition::kDev:   return kDomainDev;
    case Partition::kTest:  return kDomainTest;
  }
  return 0;
}

struct RootRecord {
  std::string root_id;
  Partition partition = Partition::kTrain;
  int source_episode_id = 0;
  Player acting_player = kInvalidPlayer;
  int round = 0;
  std::string stratum;
  DuneDecisionRole role = DuneDecisionRole::kForcedOrBookkeeping;
  std::vector<Action> history;
  std::vector<Action> legal_actions;
  std::vector<Action> candidate_actions; // 1: raw argmax, 2: 2nd highest, 3: uniform remaining
  Action reference_action = kInvalidAction; // candidate_actions[0]
  std::vector<double> candidate_actor_probs;
  std::vector<float> critic_input_9647; // privileged 9012 + market 635
};

struct ContinuationRecord {
  std::string root_id;
  Partition partition = Partition::kTrain;
  Action action = kInvalidAction;
  int replicate = 0;
  uint64_t chance_seed = 0;
  uint64_t policy_seed = 0;
  std::array<double, kNumPlayers> absolute_returns = {0.0, 0.0, 0.0, 0.0};
  std::array<double, kNumPlayers> actor_relative_scaled_returns = {0.0, 0.0, 0.0, 0.0};
};

struct ActionDifferenceMetric {
  double mse_m = 0.0;
  double mse_s = 0.0;
  double mse_zero = 0.0;
  double reduction_m_vs_s = 0.0;
  double reduction_m_vs_zero = 0.0;
  double reduction_s_vs_zero = 0.0;
  double paired_diff_m_minus_zero_mean = 0.0;
  double paired_diff_m_minus_zero_ci95_low = 0.0;
  double paired_diff_m_minus_zero_ci95_high = 0.0;
  double paired_diff_m_minus_s_mean = 0.0;
  double paired_diff_m_minus_s_ci95_low = 0.0;
  double paired_diff_m_minus_s_ci95_high = 0.0;
};

struct LocalUsefulnessMetric {
  double mean_utility_gain = 0.0;
  double utility_gain_ci95_low = 0.0;
  double utility_gain_ci95_high = 0.0;
  double changed_choice_fraction = 0.0;
  int changed_count = 0;
  int total_roots = 0;
  std::map<int, int> changed_by_round;
  std::map<int, int> changed_by_seat;
};

struct TargetReliabilityMetric {
  double correlation = 0.0;
  double mean_abs_diff = 0.0;
  double agreement_mse = 0.0;
};

enum class StudyVerdict {
  kPassQLearningScreen,
  kNoDemonstratedUsefulQGain,
  kIncompleteOrInvalid
};

inline const char* VerdictToString(StudyVerdict v) {
  switch (v) {
    case StudyVerdict::kPassQLearningScreen:
      return "PASS_Q_LEARNING_SCREEN";
    case StudyVerdict::kNoDemonstratedUsefulQGain:
      return "NO_DEMONSTRATED_USEFUL_Q_GAIN";
    case StudyVerdict::kIncompleteOrInvalid:
      return "INCOMPLETE_OR_INVALID";
  }
  return "INCOMPLETE_OR_INVALID";
}

// ===========================================================================
// Seed Helpers
// ===========================================================================

inline uint64_t DeriveSourceGameSeed(Partition p, int episode_id) {
  return dune_seed::DeriveSeed(PartitionDomain(p), kStreamSourceGame, static_cast<uint64_t>(episode_id));
}

inline uint64_t DeriveCandidateActionSampleSeed(Partition p, const std::string& root_id) {
  return dune_seed::DeriveSeed(PartitionDomain(p), kStreamCandidateActionSample, pwo2::HexPrefix64(root_id));
}

inline uint64_t DeriveContinuationChanceSeed(Partition p, const std::string& root_id, int replicate) {
  return dune_seed::DeriveSeed(PartitionDomain(p), kStreamContinuationChance,
                               pwo2::HexPrefix64(root_id), static_cast<uint64_t>(replicate));
}

inline uint64_t DeriveContinuationPolicySeed(Partition p, const std::string& root_id, int replicate) {
  return dune_seed::DeriveSeed(PartitionDomain(p), kStreamContinuationPolicy,
                               pwo2::HexPrefix64(root_id), static_cast<uint64_t>(replicate));
}

// ===========================================================================
// State Reconstruction and Input Extraction
// ===========================================================================

inline std::unique_ptr<State> ReconstructState(const std::shared_ptr<const Game>& game,
                                               const std::vector<Action>& history) {
  auto state = game->NewInitialState();
  for (Action a : history) {
    state->ApplyAction(a);
  }
  return state;
}

inline std::vector<float> ExtractCriticInput(const State& state, Player acting_player) {
  const auto* dune_state = dynamic_cast<const dune_imperium::DuneImperiumState*>(&state);
  SPIEL_CHECK_TRUE(dune_state != nullptr);

  // 1. Privileged current-position tensor (9012)
  std::vector<float> input = dune_state->VrpoCentralCriticTensor(acting_player);
  SPIEL_CHECK_EQ(input.size(), kPrivilegedCurrentPosDim);

  // 2. Ordered public Imperium market appendix (635)
  std::vector<float> appendix(kMarketAppendixDim, 0.0f);
  dune_state->WriteImperiumMarketAppendix(absl::MakeSpan(appendix));
  SPIEL_CHECK_EQ(appendix.size(), kMarketAppendixDim);

  // Concatenate: 9012 + 635 = 9647
  input.insert(input.end(), appendix.begin(), appendix.end());
  SPIEL_CHECK_EQ(input.size(), kCriticInputDim);

  for (float v : input) {
    SPIEL_CHECK_TRUE(std::isfinite(v));
  }
  return input;
}

// Actor-relative vs absolute conversion for returns
inline std::array<double, kNumPlayers> ConvertAbsoluteReturnsToActorRelative(
    Player actor, const std::array<double, kNumPlayers>& absolute_returns) {
  SPIEL_CHECK_GE(actor, 0);
  SPIEL_CHECK_LT(actor, kNumPlayers);
  std::array<double, kNumPlayers> rel{};
  for (int slot = 0; slot < kNumPlayers; ++slot) {
    rel[slot] = absolute_returns[(actor + slot) % kNumPlayers];
  }
  return rel;
}

// ===========================================================================
// Candidate Action Selection
// ===========================================================================
inline std::vector<Action> SelectCandidateActions(
    const std::vector<Action>& legal_actions,
    const std::vector<float>& logits,
    Partition partition,
    const std::string& root_id,
    std::vector<double>* out_probs = nullptr) {
  SPIEL_CHECK_GE(legal_actions.size(), 2u);
  if (out_probs != nullptr) out_probs->clear();

  // 1. Center and cap legal logits (cap = 10.0)
  std::vector<float> capped_logits = logits;
  CenterAndCapLegalLogits(capped_logits, legal_actions, 10.0f);

  // 2. Softmax over legal actions
  float max_logit = -std::numeric_limits<float>::infinity();
  for (Action a : legal_actions) {
    if (a >= 0 && static_cast<size_t>(a) < capped_logits.size()) {
      max_logit = std::max(max_logit, capped_logits[a]);
    }
  }

  std::vector<double> probs(legal_actions.size(), 0.0);
  double sum_exp = 0.0;
  for (size_t i = 0; i < legal_actions.size(); ++i) {
    Action a = legal_actions[i];
    if (a >= 0 && static_cast<size_t>(a) < capped_logits.size() && std::isfinite(max_logit)) {
      probs[i] = std::exp(static_cast<double>(capped_logits[a] - max_logit));
      sum_exp += probs[i];
    }
  }
  if (sum_exp > 0.0) {
    for (double& p : probs) p /= sum_exp;
  } else {
    for (double& p : probs) p = 1.0 / legal_actions.size();
  }

  // 3. Candidate 1: Raw-policy argmax, breaking ties by lowest action ID
  size_t best_idx = 0;
  double best_prob = probs[0];
  Action best_action = legal_actions[0];
  for (size_t i = 1; i < legal_actions.size(); ++i) {
    if (probs[i] > best_prob || (probs[i] == best_prob && legal_actions[i] < best_action)) {
      best_idx = i;
      best_prob = probs[i];
      best_action = legal_actions[i];
    }
  }

  // 4. Candidate 2: Highest-probability remaining action, breaking ties by lowest action ID
  size_t second_idx = (best_idx == 0) ? 1 : 0;
  double second_prob = probs[second_idx];
  Action second_action = legal_actions[second_idx];
  for (size_t i = 0; i < legal_actions.size(); ++i) {
    if (i == best_idx) continue;
    if (probs[i] > second_prob || (probs[i] == second_prob && legal_actions[i] < second_action)) {
      second_idx = i;
      second_prob = probs[i];
      second_action = legal_actions[i];
    }
  }

  std::vector<Action> candidates = {best_action, second_action};
  if (out_probs != nullptr) {
    out_probs->push_back(best_prob);
    out_probs->push_back(second_prob);
  }

  // 5. Candidate 3: One uniformly sampled action from remaining legal actions, if any
  if (legal_actions.size() > 2) {
    std::vector<Action> remaining;
    std::vector<double> remaining_probs;
    for (size_t i = 0; i < legal_actions.size(); ++i) {
      if (i != best_idx && i != second_idx) {
        remaining.push_back(legal_actions[i]);
        remaining_probs.push_back(probs[i]);
      }
    }
    SPIEL_CHECK_FALSE(remaining.empty());

    uint64_t sample_seed = DeriveCandidateActionSampleSeed(partition, root_id);
    std::mt19937_64 sample_rng(sample_seed);
    std::uniform_int_distribution<size_t> dist(0, remaining.size() - 1);
    size_t chosen_rem = dist(sample_rng);
    candidates.push_back(remaining[chosen_rem]);
    if (out_probs != nullptr) {
      out_probs->push_back(remaining_probs[chosen_rem]);
    }
  }

  return candidates;
}

// ===========================================================================
// Manifest JSON Serializer & Contract Checker
// ===========================================================================
struct ExecutableManifest {
  std::string study_name = "action_value_learnability_study_20260911";
  std::string actor_model_path = kActorModelPathDefault;
  std::string actor_model_sha256 = kExpectedActorSha256;
  int actor_input_dim = kActorInputDim;
  int actor_hidden_dim = kActorHiddenDim;
  int actor_num_blocks = kActorNumBlocks;
  int actor_action_dim = kActionDim;

  uint64_t critic_init_seed = kRegisteredCriticInitSeed;
  int critic_expected_input_dim = kCriticInputDim;
  int critic_hidden_dim = kCriticHiddenDim;
  int critic_num_res_blocks = kCriticNumResBlocks;
  double critic_lr = kCriticLearningRate;
  double critic_adamw_eps = kCriticAdamWEpsilon;
  double critic_weight_decay = kCriticWeightDecay;
  double critic_grad_clip_norm = kCriticGradClipNorm;
  int critic_batch_size_roots = kCriticMinibatchSizeRoots;
  int critic_epochs = kCriticTotalEpochs;
  int critic_eval_interval_epochs = kCriticEvalIntervalEpochs;

  int roots_train = kRootsTrain;
  int roots_dev = kRootsDev;
  int roots_test = kRootsTest;
  int continuations_per_action_train = kContinuationsTrain;
  int continuations_per_action_dev = kContinuationsDev;
  int continuations_per_action_test = kContinuationsTest;

  int target_s_replicate_index = 0;
  int target_m_replicate_count = kContinuationsTrain;
  bool equal_weighting_per_root = true;
  double utility_divisor = kUtilityDivisor;

  int bootstrap_resamples = kBootstrapResamples;
  uint64_t bootstrap_seed = kBootstrapSeed;

  int64_t deadline_seconds = kTotalDeadlineSeconds;
  int64_t reporting_reserve_seconds = kReportingReserveSeconds;

  double min_m_mse_reduction_vs_s = 0.10;
  double min_m_mse_reduction_vs_zero = 0.10;
  double m_minus_baseline_ci95_upper_bound = 0.0;
  double min_m_selected_utility_gain = 0.05;
  double m_selected_utility_gain_ci95_lower_bound = 0.0;

  std::string ToJsonString() const {
    json::Object root;
    root["study_name"] = study_name;
    root["actor_model_path"] = actor_model_path;
    root["actor_model_sha256"] = actor_model_sha256;
    root["actor_input_dim"] = static_cast<int64_t>(actor_input_dim);
    root["actor_hidden_dim"] = static_cast<int64_t>(actor_hidden_dim);
    root["actor_num_blocks"] = static_cast<int64_t>(actor_num_blocks);
    root["actor_action_dim"] = static_cast<int64_t>(actor_action_dim);

    root["critic_init_seed"] = static_cast<int64_t>(critic_init_seed);
    root["critic_expected_input_dim"] = static_cast<int64_t>(critic_expected_input_dim);
    root["critic_hidden_dim"] = static_cast<int64_t>(critic_hidden_dim);
    root["critic_num_res_blocks"] = static_cast<int64_t>(critic_num_res_blocks);
    root["critic_lr"] = critic_lr;
    root["critic_adamw_eps"] = critic_adamw_eps;
    root["critic_weight_decay"] = critic_weight_decay;
    root["critic_grad_clip_norm"] = critic_grad_clip_norm;
    root["critic_batch_size_roots"] = static_cast<int64_t>(critic_batch_size_roots);
    root["critic_epochs"] = static_cast<int64_t>(critic_epochs);
    root["critic_eval_interval_epochs"] = static_cast<int64_t>(critic_eval_interval_epochs);

    root["roots_train"] = static_cast<int64_t>(roots_train);
    root["roots_dev"] = static_cast<int64_t>(roots_dev);
    root["roots_test"] = static_cast<int64_t>(roots_test);
    root["continuations_per_action_train"] = static_cast<int64_t>(continuations_per_action_train);
    root["continuations_per_action_dev"] = static_cast<int64_t>(continuations_per_action_dev);
    root["continuations_per_action_test"] = static_cast<int64_t>(continuations_per_action_test);

    root["target_s_replicate_index"] = static_cast<int64_t>(target_s_replicate_index);
    root["target_m_replicate_count"] = static_cast<int64_t>(target_m_replicate_count);
    root["equal_weighting_per_root"] = equal_weighting_per_root;
    root["utility_divisor"] = utility_divisor;

    root["bootstrap_resamples"] = static_cast<int64_t>(bootstrap_resamples);
    root["bootstrap_seed"] = static_cast<uint64_t>(bootstrap_seed);

    root["deadline_seconds"] = deadline_seconds;
    root["reporting_reserve_seconds"] = reporting_reserve_seconds;

    root["min_m_mse_reduction_vs_s"] = min_m_mse_reduction_vs_s;
    root["min_m_mse_reduction_vs_zero"] = min_m_mse_reduction_vs_zero;
    root["m_minus_baseline_ci95_upper_bound"] = m_minus_baseline_ci95_upper_bound;
    root["min_m_selected_utility_gain"] = min_m_selected_utility_gain;
    root["m_selected_utility_gain_ci95_lower_bound"] = m_selected_utility_gain_ci95_lower_bound;

    return json::ToString(root);
  }

  static ExecutableManifest FromJson(const json::Object& obj) {
    ExecutableManifest m;
    m.study_name = obj.at("study_name").GetString();
    m.actor_model_path = obj.at("actor_model_path").GetString();
    m.actor_model_sha256 = obj.at("actor_model_sha256").GetString();
    m.actor_input_dim = static_cast<int>(obj.at("actor_input_dim").GetInt());
    m.actor_hidden_dim = static_cast<int>(obj.at("actor_hidden_dim").GetInt());
    m.actor_num_blocks = static_cast<int>(obj.at("actor_num_blocks").GetInt());
    m.actor_action_dim = static_cast<int>(obj.at("actor_action_dim").GetInt());

    m.critic_init_seed = static_cast<uint64_t>(obj.at("critic_init_seed").GetInt());
    m.critic_expected_input_dim = static_cast<int>(obj.at("critic_expected_input_dim").GetInt());
    m.critic_hidden_dim = static_cast<int>(obj.at("critic_hidden_dim").GetInt());
    m.critic_num_res_blocks = static_cast<int>(obj.at("critic_num_res_blocks").GetInt());
    m.critic_lr = obj.at("critic_lr").GetDouble();
    m.critic_adamw_eps = obj.at("critic_adamw_eps").GetDouble();
    m.critic_weight_decay = obj.at("critic_weight_decay").GetDouble();
    m.critic_grad_clip_norm = obj.at("critic_grad_clip_norm").GetDouble();
    m.critic_batch_size_roots = static_cast<int>(obj.at("critic_batch_size_roots").GetInt());
    m.critic_epochs = static_cast<int>(obj.at("critic_epochs").GetInt());
    m.critic_eval_interval_epochs = static_cast<int>(obj.at("critic_eval_interval_epochs").GetInt());

    m.roots_train = static_cast<int>(obj.at("roots_train").GetInt());
    m.roots_dev = static_cast<int>(obj.at("roots_dev").GetInt());
    m.roots_test = static_cast<int>(obj.at("roots_test").GetInt());
    m.continuations_per_action_train = static_cast<int>(obj.at("continuations_per_action_train").GetInt());
    m.continuations_per_action_dev = static_cast<int>(obj.at("continuations_per_action_dev").GetInt());
    m.continuations_per_action_test = static_cast<int>(obj.at("continuations_per_action_test").GetInt());

    m.target_s_replicate_index = static_cast<int>(obj.at("target_s_replicate_index").GetInt());
    m.target_m_replicate_count = static_cast<int>(obj.at("target_m_replicate_count").GetInt());
    m.equal_weighting_per_root = obj.at("equal_weighting_per_root").GetBool();
    m.utility_divisor = obj.at("utility_divisor").GetDouble();

    m.bootstrap_resamples = static_cast<int>(obj.at("bootstrap_resamples").GetInt());
    m.bootstrap_seed = static_cast<uint64_t>(obj.at("bootstrap_seed").GetInt());

    m.deadline_seconds = obj.at("deadline_seconds").GetInt();
    m.reporting_reserve_seconds = obj.at("reporting_reserve_seconds").GetInt();

    m.min_m_mse_reduction_vs_s = obj.at("min_m_mse_reduction_vs_s").GetDouble();
    m.min_m_mse_reduction_vs_zero = obj.at("min_m_mse_reduction_vs_zero").GetDouble();
    m.m_minus_baseline_ci95_upper_bound = obj.at("m_minus_baseline_ci95_upper_bound").GetDouble();
    m.min_m_selected_utility_gain = obj.at("min_m_selected_utility_gain").GetDouble();
    m.m_selected_utility_gain_ci95_lower_bound = obj.at("m_selected_utility_gain_ci95_lower_bound").GetDouble();

    return m;
  }
};

// ===========================================================================
// Bootstrap Confidence Interval Calculation (Resampling whole roots)
// ===========================================================================
inline std::pair<double, double> Bootstrap95CI(
    const std::vector<double>& per_root_values,
    int resamples = kBootstrapResamples,
    uint64_t seed = kBootstrapSeed) {
  if (per_root_values.empty()) return {0.0, 0.0};
  if (per_root_values.size() == 1) return {per_root_values[0], per_root_values[0]};

  std::mt19937_64 rng(seed);
  std::uniform_int_distribution<size_t> dist(0, per_root_values.size() - 1);
  const size_t n = per_root_values.size();

  std::vector<double> means(resamples);
  for (int r = 0; r < resamples; ++r) {
    double sum = 0.0;
    for (size_t i = 0; i < n; ++i) {
      sum += per_root_values[dist(rng)];
    }
    means[r] = sum / n;
  }
  std::sort(means.begin(), means.end());
  const size_t low_idx = static_cast<size_t>(0.025 * resamples);
  const size_t high_idx = static_cast<size_t>(0.975 * resamples);
  return {means[low_idx], means[high_idx]};
}

}  // namespace action_value_study
}  // namespace open_spiel

#endif  // OPEN_SPIEL_EXAMPLES_DUNE_ACTION_VALUE_STUDY_H_
