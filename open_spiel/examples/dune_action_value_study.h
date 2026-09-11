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
#include <set>
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
constexpr uint64_t kStreamRootReservoirSelection  = 0x0052;
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

inline uint64_t DeriveRootSelectionSeed(Partition p, int episode_id) {
  return dune_seed::DeriveSeed(PartitionDomain(p), kStreamRootReservoirSelection, static_cast<uint64_t>(episode_id));
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

  bool Validate(std::string* error = nullptr) const {
    auto fail = [&](const std::string& msg) {
      if (error != nullptr) *error = "Manifest Validation Error: " + msg;
      return false;
    };
    if (study_name != "action_value_learnability_study_20260911") return fail("study_name mismatch");
    if (actor_model_sha256 != kExpectedActorSha256) return fail("actor_model_sha256 mismatch");
    if (actor_input_dim != kActorInputDim) return fail("actor_input_dim mismatch");
    if (actor_hidden_dim != kActorHiddenDim) return fail("actor_hidden_dim mismatch");
    if (actor_num_blocks != kActorNumBlocks) return fail("actor_num_blocks mismatch");
    if (actor_action_dim != kActionDim) return fail("actor_action_dim mismatch");

    if (critic_init_seed != kRegisteredCriticInitSeed) return fail("critic_init_seed mismatch");
    if (critic_expected_input_dim != kCriticInputDim) return fail("critic_expected_input_dim mismatch");
    if (critic_hidden_dim != kCriticHiddenDim) return fail("critic_hidden_dim mismatch");
    if (critic_num_res_blocks != kCriticNumResBlocks) return fail("critic_num_res_blocks mismatch");
    if (std::abs(critic_lr - kCriticLearningRate) > 1e-9) return fail("critic_lr mismatch");
    if (std::abs(critic_adamw_eps - kCriticAdamWEpsilon) > 1e-9) return fail("critic_adamw_eps mismatch");
    if (std::abs(critic_weight_decay - kCriticWeightDecay) > 1e-9) return fail("critic_weight_decay mismatch");
    if (std::abs(critic_grad_clip_norm - kCriticGradClipNorm) > 1e-9) return fail("critic_grad_clip_norm mismatch");
    if (critic_batch_size_roots != kCriticMinibatchSizeRoots) return fail("critic_batch_size_roots mismatch");
    if (critic_epochs != kCriticTotalEpochs) return fail("critic_epochs mismatch");
    if (critic_eval_interval_epochs != kCriticEvalIntervalEpochs) return fail("critic_eval_interval_epochs mismatch");

    if (roots_train != kRootsTrain) return fail("roots_train mismatch");
    if (roots_dev != kRootsDev) return fail("roots_dev mismatch");
    if (roots_test != kRootsTest) return fail("roots_test mismatch");
    if (continuations_per_action_train != kContinuationsTrain) return fail("continuations_per_action_train mismatch");
    if (continuations_per_action_dev != kContinuationsDev) return fail("continuations_per_action_dev mismatch");
    if (continuations_per_action_test != kContinuationsTest) return fail("continuations_per_action_test mismatch");

    if (target_s_replicate_index != 0) return fail("target_s_replicate_index must be 0");
    if (target_m_replicate_count != kContinuationsTrain) return fail("target_m_replicate_count mismatch");
    if (!equal_weighting_per_root) return fail("equal_weighting_per_root must be true");
    if (std::abs(utility_divisor - kUtilityDivisor) > 1e-9) return fail("utility_divisor must be 4.0");

    if (bootstrap_resamples != kBootstrapResamples) return fail("bootstrap_resamples mismatch");
    if (bootstrap_seed != kBootstrapSeed) return fail("bootstrap_seed mismatch");
    if (deadline_seconds != kTotalDeadlineSeconds) return fail("deadline_seconds mismatch");
    if (reporting_reserve_seconds != kReportingReserveSeconds) return fail("reporting_reserve_seconds mismatch");

    return true;
  }

  static ExecutableManifest LoadFromFile(const std::filesystem::path& path, std::string* error = nullptr) {
    std::ifstream f(path);
    if (!f.is_open()) {
      if (error != nullptr) *error = "Cannot open manifest file: " + path.string();
      return ExecutableManifest{};
    }
    std::string text((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    auto parsed = json::FromString(text);
    if (!parsed.has_value()) {
      if (error != nullptr) *error = "Failed to parse manifest JSON: " + path.string();
      return ExecutableManifest{};
    }
    ExecutableManifest m = FromJson(parsed.value().GetObject());
    if (!m.Validate(error)) {
      return ExecutableManifest{};
    }
    return m;
  }
};

// ===========================================================================
// Real Boundary Validators
// ===========================================================================

// Validates a single root record against state reconstruction and legal actions
inline bool ValidateRootRecord(
    const RootRecord& r,
    const std::shared_ptr<const Game>& game,
    std::string* error = nullptr) {
  auto fail = [&](const std::string& msg) {
    if (error != nullptr) *error = absl::StrFormat("Root [%s]: %s", r.root_id, msg);
    return false;
  };

  if (r.root_id.empty() || r.root_id.length() != 16) {
    return fail("root_id must be 16-hex characters");
  }
  std::string expected_hash = pwo2::HistoryHash(r.history).substr(0, 16);
  if (r.root_id != expected_hash) {
    return fail(absl::StrFormat("root_id %s does not match history hash %s", r.root_id, expected_hash));
  }
  if (r.acting_player < 0 || r.acting_player >= kNumPlayers) {
    return fail(absl::StrFormat("acting_player %d is out of range [0, 3]", r.acting_player));
  }
  if (r.round < 2 || r.round > 10) {
    return fail(absl::StrFormat("round %d is out of range [2, 10]", r.round));
  }
  if (r.role != DuneDecisionRole::kAgentPrimary) {
    return fail("role is not kAgentPrimary");
  }
  if (r.candidate_actions.size() < 2 || r.candidate_actions.size() > 3) {
    return fail("candidate_actions count must be 2 or 3");
  }
  if (r.reference_action != r.candidate_actions[0]) {
    return fail("reference_action must equal candidate_actions[0]");
  }
  if (r.critic_input_9647.size() != static_cast<size_t>(kCriticInputDim)) {
    return fail(absl::StrFormat("critic_input_9647 width %d != %d", r.critic_input_9647.size(), kCriticInputDim));
  }
  for (size_t i = 0; i < r.critic_input_9647.size(); ++i) {
    if (!std::isfinite(r.critic_input_9647[i])) {
      return fail(absl::StrFormat("critic_input_9647 contains non-finite value at %d", i));
    }
  }

  // Reconstruct state
  std::unique_ptr<State> state;
  try {
    state = ReconstructState(game, r.history);
  } catch (const std::exception& e) {
    return fail(std::string("state reconstruction failed: ") + e.what());
  }

  if (state->IsTerminal()) return fail("reconstructed state is terminal");
  if (state->IsChanceNode()) return fail("reconstructed state is a chance node");
  if (state->CurrentPlayer() != r.acting_player) {
    return fail(absl::StrFormat("reconstructed current player %d != acting_player %d",
                                state->CurrentPlayer(), r.acting_player));
  }

  auto true_legal = state->LegalActions();
  if (true_legal.size() != r.legal_actions.size()) {
    return fail(absl::StrFormat("legal_actions size %d != reconstructed size %d",
                                r.legal_actions.size(), true_legal.size()));
  }
  for (size_t i = 0; i < true_legal.size(); ++i) {
    if (true_legal[i] != r.legal_actions[i]) {
      return fail(absl::StrFormat("legal_action mismatch at %d: %d != %d",
                                  i, r.legal_actions[i], true_legal[i]));
    }
  }

  // Verify candidate actions are in legal actions
  for (Action ca : r.candidate_actions) {
    if (std::find(true_legal.begin(), true_legal.end(), ca) == true_legal.end()) {
      return fail(absl::StrFormat("candidate action %d is not in legal actions", ca));
    }
  }

  // Verify recomputed critic tensor matches recorded critic input
  auto true_critic_input = ExtractCriticInput(*state, r.acting_player);
  if (true_critic_input.size() != r.critic_input_9647.size()) {
    return fail("extracted critic input dimension mismatch");
  }
  for (size_t i = 0; i < true_critic_input.size(); ++i) {
    if (std::abs(true_critic_input[i] - r.critic_input_9647[i]) > 1e-5f) {
      return fail(absl::StrFormat("critic input value mismatch at index %d: rec=%f, true=%f",
                                  i, r.critic_input_9647[i], true_critic_input[i]));
    }
  }

  return true;
}

// Validates a single continuation record against expected root and derived seeds
inline bool ValidateContinuationRecord(
    const ContinuationRecord& cr,
    const RootRecord& root,
    int expected_replicates,
    std::string* error = nullptr) {
  auto fail = [&](const std::string& msg) {
    if (error != nullptr) *error = absl::StrFormat("Continuation [%s, act=%d, rep=%d]: %s",
                                                   cr.root_id, cr.action, cr.replicate, msg);
    return false;
  };

  if (cr.root_id != root.root_id) {
    return fail(absl::StrFormat("root_id %s != root.root_id %s", cr.root_id, root.root_id));
  }
  if (cr.partition != root.partition) {
    return fail("partition does not match root");
  }
  if (std::find(root.candidate_actions.begin(), root.candidate_actions.end(), cr.action) ==
      root.candidate_actions.end()) {
    return fail(absl::StrFormat("action %d is not among candidate actions", cr.action));
  }
  if (cr.replicate < 0 || cr.replicate >= expected_replicates) {
    return fail(absl::StrFormat("replicate %d is out of range [0, %d)", cr.replicate, expected_replicates));
  }

  uint64_t expected_chance = DeriveContinuationChanceSeed(cr.partition, cr.root_id, cr.replicate);
  if (cr.chance_seed != expected_chance) {
    return fail(absl::StrFormat("chance seed %llu != expected %llu", cr.chance_seed, expected_chance));
  }

  uint64_t expected_policy = DeriveContinuationPolicySeed(cr.partition, cr.root_id, cr.replicate);
  if (cr.policy_seed != expected_policy) {
    return fail(absl::StrFormat("policy seed %llu != expected %llu", cr.policy_seed, expected_policy));
  }

  for (int p = 0; p < kNumPlayers; ++p) {
    if (!std::isfinite(cr.absolute_returns[p])) {
      return fail(absl::StrFormat("absolute return for player %d is non-finite", p));
    }
    if (!std::isfinite(cr.actor_relative_scaled_returns[p])) {
      return fail(absl::StrFormat("actor-relative return for slot %d is non-finite", p));
    }
  }

  // Verify actor-relative mapping and scaling (/ 4.0)
  for (int slot = 0; slot < kNumPlayers; ++slot) {
    int abs_player = (root.acting_player + slot) % kNumPlayers;
    double expected_scaled = cr.absolute_returns[abs_player] / kUtilityDivisor;
    if (std::abs(cr.actor_relative_scaled_returns[slot] - expected_scaled) > 1e-6) {
      return fail(absl::StrFormat("actor_relative_scaled_returns[%d]=%f != expected %f",
                                  slot, cr.actor_relative_scaled_returns[slot], expected_scaled));
    }
  }

  return true;
}

// Validates the entire corpus of roots
inline bool ValidateCorpus(
    const std::vector<RootRecord>& roots,
    const std::shared_ptr<const Game>& game,
    const ExecutableManifest& manifest,
    std::string* error = nullptr) {
  auto fail = [&](const std::string& msg) {
    if (error != nullptr) *error = "Corpus Validation Error: " + msg;
    return false;
  };

  int total_expected = manifest.roots_train + manifest.roots_dev + manifest.roots_test;
  if (roots.size() != static_cast<size_t>(total_expected)) {
    return fail(absl::StrFormat("total roots %d != expected %d", roots.size(), total_expected));
  }

  std::map<Partition, int> part_counts;
  std::map<Partition, std::array<int, 4>> seat_counts;
  std::map<int, int> round_counts;
  std::set<std::string> seen_roots;
  std::map<Partition, std::set<int>> seen_episodes;

  for (const auto& r : roots) {
    if (seen_roots.count(r.root_id) > 0) {
      return fail("duplicate root_id: " + r.root_id);
    }
    seen_roots.insert(r.root_id);

    if (seen_episodes[r.partition].count(r.source_episode_id) > 0) {
      return fail(absl::StrFormat("duplicate source_episode_id %d in partition %s",
                                  r.source_episode_id, PartitionToString(r.partition)));
    }
    seen_episodes[r.partition].insert(r.source_episode_id);

    part_counts[r.partition]++;
    seat_counts[r.partition][r.acting_player]++;
    round_counts[r.round]++;

    std::string root_err;
    if (!ValidateRootRecord(r, game, &root_err)) {
      return fail(root_err);
    }
  }

  if (part_counts[Partition::kTrain] != manifest.roots_train) {
    return fail(absl::StrFormat("train roots %d != %d", part_counts[Partition::kTrain], manifest.roots_train));
  }
  if (part_counts[Partition::kDev] != manifest.roots_dev) {
    return fail(absl::StrFormat("dev roots %d != %d", part_counts[Partition::kDev], manifest.roots_dev));
  }
  if (part_counts[Partition::kTest] != manifest.roots_test) {
    return fail(absl::StrFormat("test roots %d != %d", part_counts[Partition::kTest], manifest.roots_test));
  }

  // Check seat balance
  for (int s = 0; s < 4; ++s) {
    if (seat_counts[Partition::kTrain][s] != manifest.roots_train / 4) {
      return fail(absl::StrFormat("train seat %d count %d != %d", s, seat_counts[Partition::kTrain][s], manifest.roots_train / 4));
    }
    if (seat_counts[Partition::kDev][s] != manifest.roots_dev / 4) {
      return fail(absl::StrFormat("dev seat %d count %d != %d", s, seat_counts[Partition::kDev][s], manifest.roots_dev / 4));
    }
    if (seat_counts[Partition::kTest][s] != manifest.roots_test / 4) {
      return fail(absl::StrFormat("test seat %d count %d != %d", s, seat_counts[Partition::kTest][s], manifest.roots_test / 4));
    }
  }

  // Check round coverage: every round 2..10 must have roots
  for (int rd = 2; rd <= 10; ++rd) {
    if (round_counts[rd] == 0) {
      return fail(absl::StrFormat("round %d has 0 roots", rd));
    }
  }

  return true;
}

// Validates loaded continuation map for a partition
inline bool ValidateContinuations(
    const std::vector<RootRecord>& roots,
    const std::map<std::pair<std::string, Action>, std::vector<ContinuationRecord>>& conts,
    Partition partition,
    int expected_replicates,
    std::string* error = nullptr) {
  auto fail = [&](const std::string& msg) {
    if (error != nullptr) *error = "Continuations Validation Error: " + msg;
    return false;
  };

  std::map<std::string, const RootRecord*> root_map;
  for (const auto& r : roots) {
    if (r.partition == partition) {
      root_map[r.root_id] = &r;
    }
  }

  for (const auto& kv : root_map) {
    const auto& r = *kv.second;
    for (Action a : r.candidate_actions) {
      auto it = conts.find({r.root_id, a});
      if (it == conts.end()) {
        return fail(absl::StrFormat("missing continuations for root %s action %d", r.root_id, a));
      }
      const auto& reps = it->second;
      if (reps.size() != static_cast<size_t>(expected_replicates)) {
        return fail(absl::StrFormat("root %s action %d has %d reps != expected %d",
                                    r.root_id, a, reps.size(), expected_replicates));
      }

      std::set<int> seen_reps;
      for (const auto& cr : reps) {
        if (seen_reps.count(cr.replicate) > 0) {
          return fail(absl::StrFormat("duplicate replicate %d for root %s action %d", cr.replicate, r.root_id, a));
        }
        seen_reps.insert(cr.replicate);

        std::string cr_err;
        if (!ValidateContinuationRecord(cr, r, expected_replicates, &cr_err)) {
          return fail(cr_err);
        }
      }

      for (int k = 0; k < expected_replicates; ++k) {
        if (seen_reps.count(k) == 0) {
          return fail(absl::StrFormat("missing replicate %d for root %s action %d", k, r.root_id, a));
        }
      }
    }
  }

  return true;
}

// Validates selected checkpoints file and matches file hashes on disk
inline bool ValidateSelectedCheckpoints(
    const std::filesystem::path& sel_path,
    std::string* error = nullptr) {
  auto fail = [&](const std::string& msg) {
    if (error != nullptr) *error = "Selected Checkpoints Validation Error: " + msg;
    return false;
  };

  std::ifstream f(sel_path);
  if (!f.is_open()) return fail("cannot open " + sel_path.string());
  std::string text((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
  auto parsed = json::FromString(text);
  if (!parsed.has_value()) return fail("failed to parse selected_checkpoints.json");

  const auto& obj = parsed.value().GetObject();
  if (obj.find("critic_s") == obj.end() || obj.find("critic_m") == obj.end()) {
    return fail("selected_checkpoints.json missing critic_s or critic_m");
  }

  for (const std::string& name : {"critic_s", "critic_m"}) {
    const auto& c_obj = obj.at(name).GetObject();
    std::string path = c_obj.at("path").GetString();
    std::string expected_sha = c_obj.at("sha256").GetString();

    if (!std::filesystem::exists(path)) {
      return fail(absl::StrFormat("%s checkpoint path does not exist: %s", name, path));
    }
    std::string actual_sha = ComputeFileSHA256(path);
    if (actual_sha != expected_sha) {
      return fail(absl::StrFormat("%s hash mismatch on disk: expected %s, actual %s",
                                  name, expected_sha, actual_sha));
    }
  }

  return true;
}

// Validates actor immutability: requires_grad=false, and hash unchanged
inline bool ValidateActorImmutability(
    const std::shared_ptr<SharedDunePolicyValueNetImpl>& actor,
    const std::string& actor_path,
    std::string* error = nullptr) {
  auto fail = [&](const std::string& msg) {
    if (error != nullptr) *error = "Actor Immutability Error: " + msg;
    return false;
  };

  if (!actor) return fail("actor model pointer is null");

  for (const auto& p : actor->parameters()) {
    if (p.requires_grad()) {
      return fail("actor parameter has requires_grad == true");
    }
  }

  std::string disk_sha = ComputeFileSHA256(actor_path);
  if (disk_sha != kExpectedActorSha256) {
    return fail(absl::StrFormat("actor disk SHA256 %s != expected %s", disk_sha, kExpectedActorSha256));
  }

  return true;
}

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
