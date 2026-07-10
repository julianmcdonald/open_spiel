// Native LibTorch PPO trainer for Dune Imperium.
//
// This executable is intentionally separate from dune_benchmark.cc. It follows
// the OpenSpiel PyTorch PPO structure: collect an on-policy rollout batch,
// freeze it, run shuffled PPO epochs/minibatches, then sync the inference model.

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
#include <random>
#include <shared_mutex>
#include <string>
#include <thread>
#include <vector>

#include "open_spiel/abseil-cpp/absl/flags/flag.h"
#include "open_spiel/abseil-cpp/absl/flags/parse.h"
#include "open_spiel/abseil-cpp/absl/strings/str_format.h"
#include "open_spiel/games/dune_imperium/dune_imperium.h"
#include "open_spiel/games/dune_imperium/dune_imperium_board.h"
#include "open_spiel/games/dune_imperium/dune_imperium_util.h"
#include "open_spiel/spiel.h"

#ifdef OPEN_SPIEL_BUILD_WITH_LIBTORCH
#include <torch/torch.h>

#include "dune_network.h"
#include "dune_seed_utils.h"
#endif

ABSL_FLAG(std::string, game, "dune_imperium", "The OpenSpiel game to train.");
ABSL_FLAG(int, threads, 64, "Rollout worker threads.");
ABSL_FLAG(int, total_updates, 1000, "Number of PPO collect/update cycles.");
ABSL_FLAG(int, rollout_transitions, 32768,
          "Minimum on-policy transitions collected before each PPO update.");
ABSL_FLAG(int, rollout_games, 0,
          "Exact complete games collected per PPO update rollout batch. "
          "If > 0, overrides transition-threshold mode.");
ABSL_FLAG(int, ppo_minibatch_size, 2048, "PPO minibatch size.");
ABSL_FLAG(int, ppo_update_epochs, 4, "PPO epochs per rollout batch.");
ABSL_FLAG(double, learning_rate, 2.5e-4, "AdamW learning rate.");
ABSL_FLAG(bool, anneal_lr, true, "Linearly anneal learning rate over updates.");
ABSL_FLAG(double, gamma, 1.0, "Discount factor.");
ABSL_FLAG(double, gae_lambda, 0.95, "GAE lambda.");
ABSL_FLAG(bool, normalize_advantages, true,
          "Normalize advantages within each PPO minibatch.");
ABSL_FLAG(double, ppo_clip_epsilon, 0.2, "PPO surrogate/value clip epsilon.");
ABSL_FLAG(bool, ppo_clip_value_loss, true, "Use clipped value loss.");
ABSL_FLAG(double, entropy_coef, 0.01, "Entropy bonus coefficient.");
ABSL_FLAG(double, value_coef, 0.5, "Value loss coefficient.");
ABSL_FLAG(double, grad_clip_norm, 0.5, "Maximum gradient norm.");
ABSL_FLAG(double, target_kl, 0.0,
          "Stop PPO epochs early if approximate KL exceeds this value. <=0 disables.");
ABSL_FLAG(double, weight_decay, 0.0, "AdamW weight decay for torso/value params.");
ABSL_FLAG(double, policy_weight_decay, 0.0,
          "AdamW weight decay for policy-head params.");

ABSL_FLAG(int, eval_batch_size, 64, "Target inference batch size.");
ABSL_FLAG(int, eval_timeout_ms, 2, "Inference batch timeout in milliseconds.");
ABSL_FLAG(int, hidden_dim, 2048, "Network hidden dimension.");
ABSL_FLAG(int, num_blocks, 8, "Network residual block count.");
ABSL_FLAG(double, logit_cap, 10.0,
          "Smooth tanh cap on legal-centered policy logits. <=0 disables.");
ABSL_FLAG(bool, train_amp, true, "Use CUDA BF16 autocast for PPO updates.");
ABSL_FLAG(bool, evaluator_device_synchronize, true,
          "Use whole-device CUDA synchronize after evaluator D2H copies.");

ABSL_FLAG(double, shaped_reward_weight, 0.2,
          "Weight for intermediate VP shaped rewards.");
ABSL_FLAG(double, tleilaxu_breadcrumb_weight, 0.0,
          "Weight for Tleilaxu levels 5/6 breadcrumbs.");
ABSL_FLAG(double, tleilaxu_level7_breadcrumb_weight, 0.0,
          "Weight for Tleilaxu level 7 breadcrumb.");
ABSL_FLAG(int, decay_horizon, 12000000,
          "Transitions over which shaped reward lambda decays.");
ABSL_FLAG(double, reward_scale, 4.0,
          "Divide shaped plus terminal rewards by this value.");

ABSL_FLAG(std::string, model_checkpoint, "dune_ppo_model.pt",
          "Model checkpoint to load/save.");
ABSL_FLAG(std::string, optim_checkpoint, "dune_ppo_optimizer.pt",
          "Optimizer checkpoint to load/save.");
ABSL_FLAG(std::string, run_prefix, "dune_ppo",
          "Prefix for rotating checkpoints.");
ABSL_FLAG(int, checkpoint_interval, 10,
          "Save rotating checkpoints every N PPO updates. <=0 disables.");
ABSL_FLAG(bool, save_final_checkpoint, true,
          "Save final model and optimizer checkpoints.");
ABSL_FLAG(int, seed, 1, "Base random seed.");
ABSL_FLAG(bool, pipeline, false,
          "Overlap next rollout collection with current PPO training. "
          "Beneficial with separate inference/training GPUs. On a single GPU, "
          "both workloads compete for compute and pipelining may be slower.");

ABSL_FLAG(std::string, search_label_dir, "",
          "Directory containing search teacher labels. Empty disables.");
ABSL_FLAG(double, search_lambda, 0.5,
          "Weight for search teacher KL distillation loss.");
ABSL_FLAG(int, search_minibatches_per_update, 2,
          "Search distillation minibatches per PPO update.");
ABSL_FLAG(int, search_minibatch_size, 512,
          "Size of each search distillation minibatch.");

namespace open_spiel {
namespace {

#ifdef OPEN_SPIEL_BUILD_WITH_LIBTORCH

struct PpoTransition {
  std::vector<float> state;
  std::vector<Action> legal_actions;
  int64_t action;
  float old_log_prob;
  float reward;
  float value;
  float advantage;
  float return_value;
  int player_id;
  uint64_t episode_id;
};

class PpoRolloutBuffer {
 public:
  size_t PushTrajectory(std::vector<PpoTransition>&& trajectory) {
    std::lock_guard<std::mutex> lock(mu_);
    size_t size = trajectory.size();
    trajectories_.push_back(std::move(trajectory));
    num_transitions_ += size;
    return num_transitions_;
  }

  size_t Size() const {
    std::lock_guard<std::mutex> lock(mu_);
    return num_transitions_;
  }

  std::vector<PpoTransition> TakeAll() {
    std::lock_guard<std::mutex> lock(mu_);
    std::sort(trajectories_.begin(), trajectories_.end(),
              [](const std::vector<PpoTransition>& a,
                 const std::vector<PpoTransition>& b) {
                uint64_t id_a = a.empty() ? 0 : a[0].episode_id;
                uint64_t id_b = b.empty() ? 0 : b[0].episode_id;
                return id_a < id_b;
              });
    std::vector<PpoTransition> out;
    out.reserve(num_transitions_);
    for (auto& traj : trajectories_) {
      for (auto& transition : traj) {
        out.push_back(std::move(transition));
      }
    }
    trajectories_.clear();
    num_transitions_ = 0;
    return out;
  }

 private:
  mutable std::mutex mu_;
  std::vector<std::vector<PpoTransition>> trajectories_;
  size_t num_transitions_ = 0;
};

struct SearchLabel {
  std::vector<float> state;
  int32_t num_legal_actions;
  std::vector<std::pair<int64_t, float>> teacher_probs;
  std::vector<std::pair<int64_t, float>> ppo_probs;
  float teacher_kl;
  int32_t num_covered_actions;
  float eta;
  uint8_t eta_capped;
};

class SearchLabelBuffer {
 public:
  void SetExpectedDimensions(int64_t obs_size, int64_t action_dim) {
    expected_obs_size_ = obs_size;
    expected_action_dim_ = action_dim;
  }

  void LoadFromDirectory(const std::string& dir) {
    if (dir.empty() || !std::filesystem::exists(dir)) return;
    LoadNewFiles(dir);
  }

  void LoadNewFiles(const std::string& dir) {
    if (dir.empty() || !std::filesystem::exists(dir)) return;
    std::lock_guard<std::mutex> lock(mu_);
    for (const auto& entry : std::filesystem::directory_iterator(dir)) {
      if (entry.path().extension() == ".bin") {
        std::string path_str = entry.path().string();
        if (loaded_files_.find(path_str) == loaded_files_.end()) {
          LoadFile(path_str);
          loaded_files_.insert(path_str);
        }
      }
    }
  }

  std::vector<SearchLabel> Sample(int n, std::mt19937* rng) const {
    std::lock_guard<std::mutex> lock(mu_);
    std::vector<SearchLabel> batch;
    if (labels_.empty()) return batch;
    batch.reserve(n);
    std::uniform_int_distribution<size_t> dist(0, labels_.size() - 1);
    for (int i = 0; i < n; ++i) {
      batch.push_back(labels_[dist(*rng)]);
    }
    return batch;
  }

  size_t Size() const {
    std::lock_guard<std::mutex> lock(mu_);
    return labels_.size();
  }

 private:
  void LoadFile(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
      std::cerr << "SearchLabelBuffer: Failed to open file: " << path << "\n";
      return;
    }

    uint32_t magic = 0;
    uint32_t schema = 0;
    int32_t obs_size = 0;
    int32_t action_dim = 0;
    int32_t max_simulations = 0;
    float value_scale = 0;
    float puct_c = 0;
    float target_teacher_kl = 0;
    int32_t min_visits = 0;
    int32_t min_coverage = 0;
    float blueprint_temp = 0;
    uint64_t fingerprint = 0;
    uint32_t reserved = 0;

    in.read(reinterpret_cast<char*>(&magic), 4);
    if (magic != 0x4c545344) {
      std::cerr << "SearchLabelBuffer: Invalid magic in " << path << "\n";
      return;
    }
    in.read(reinterpret_cast<char*>(&schema), 4);
    if (schema != 1) {
      std::cerr << "SearchLabelBuffer: Unsupported schema version " << schema
                << " in " << path << " (expected 1)\n";
      return;
    }
    in.read(reinterpret_cast<char*>(&obs_size), 4);
    in.read(reinterpret_cast<char*>(&action_dim), 4);
    if (expected_obs_size_ > 0 && obs_size != expected_obs_size_) {
      std::cerr << "SearchLabelBuffer: obs_size mismatch in " << path
                << ": file=" << obs_size << " expected=" << expected_obs_size_ << "\n";
      return;
    }
    if (expected_action_dim_ > 0 && action_dim != expected_action_dim_) {
      std::cerr << "SearchLabelBuffer: action_dim mismatch in " << path
                << ": file=" << action_dim << " expected=" << expected_action_dim_ << "\n";
      return;
    }
    in.read(reinterpret_cast<char*>(&max_simulations), 4);
    in.read(reinterpret_cast<char*>(&value_scale), 4);
    in.read(reinterpret_cast<char*>(&puct_c), 4);
    in.read(reinterpret_cast<char*>(&target_teacher_kl), 4);
    in.read(reinterpret_cast<char*>(&min_visits), 4);
    in.read(reinterpret_cast<char*>(&min_coverage), 4);
    in.read(reinterpret_cast<char*>(&blueprint_temp), 4);
    in.read(reinterpret_cast<char*>(&fingerprint), 8);
    if (!has_fingerprint_) {
      expected_fingerprint_ = fingerprint;
      has_fingerprint_ = true;
    } else if (fingerprint != expected_fingerprint_) {
      std::cerr << "Warning: mixed label fingerprints in " << path
                << ": file fingerprint=0x" << std::hex << fingerprint
                << " expected=0x" << expected_fingerprint_ << std::dec << "\n";
    }
    in.read(reinterpret_cast<char*>(&reserved), 4);

    int labels_before = labels_.size();
    while (in.peek() != EOF) {
      SearchLabel label;
      label.state.resize(obs_size);
      in.read(reinterpret_cast<char*>(label.state.data()), obs_size * sizeof(float));
      if (!in) break;

      in.read(reinterpret_cast<char*>(&label.num_legal_actions), sizeof(int32_t));
      if (label.num_legal_actions <= 0 || label.num_legal_actions > action_dim) {
        std::cerr << "SearchLabelBuffer: Invalid num_legal_actions " << label.num_legal_actions
                  << " in " << path << "\n";
        break;
      }
      label.teacher_probs.resize(label.num_legal_actions);
      label.ppo_probs.resize(label.num_legal_actions);

      bool valid = true;
      for (int32_t i = 0; i < label.num_legal_actions; ++i) {
        int32_t action_id = 0;
        float t_prob = 0.0f;
        float p_prob = 0.0f;
        in.read(reinterpret_cast<char*>(&action_id), sizeof(int32_t));
        in.read(reinterpret_cast<char*>(&t_prob), sizeof(float));
        in.read(reinterpret_cast<char*>(&p_prob), sizeof(float));
        if (action_id < 0 || action_id >= action_dim ||
            !std::isfinite(t_prob) || !std::isfinite(p_prob)) {
          valid = false;
          break;
        }
        label.teacher_probs[i] = {action_id, t_prob};
        label.ppo_probs[i] = {action_id, p_prob};
      }
      if (!valid || !in) break;
      in.read(reinterpret_cast<char*>(&label.teacher_kl), sizeof(float));
      in.read(reinterpret_cast<char*>(&label.num_covered_actions), sizeof(int32_t));
      in.read(reinterpret_cast<char*>(&label.eta), sizeof(float));
      in.read(reinterpret_cast<char*>(&label.eta_capped), sizeof(uint8_t));
      uint8_t padding[3];
      in.read(reinterpret_cast<char*>(padding), 3);
      if (!in) break;

      labels_.push_back(std::move(label));
    }
    int labels_loaded = labels_.size() - labels_before;
    std::cout << "SearchLabelBuffer: Loaded " << labels_loaded
              << " labels from " << path << "\n";
  }

  int64_t expected_obs_size_ = 0;
  int64_t expected_action_dim_ = 0;
  uint64_t expected_fingerprint_ = 0;
  bool has_fingerprint_ = false;
  std::vector<SearchLabel> labels_;
  std::set<std::string> loaded_files_;
  mutable std::mutex mu_;
};

struct WorkerStats {
  uint64_t games = 0;
  uint64_t moves = 0;

  // Natural SM Acquisitions
  uint64_t sm_acquisitions_by_seat[4] = {0};
  uint64_t sm_acquisitions_by_leader[14] = {0};
  uint64_t sm_acquisitions_by_round[11] = {0};
};

struct PpoUpdateStats {
  double policy_loss = 0.0;
  double value_loss = 0.0;
  double entropy = 0.0;
  double approx_kl = 0.0;
  double clip_fraction = 0.0;
  double explained_variance = 0.0;
  int minibatches = 0;
  bool early_stopped = false;
  double grad_norm_sum = 0.0;
  int grad_norm_count = 0;
};

torch::Tensor LegalLogitMean(const torch::Tensor& logits,
                             const torch::Tensor& legal_mask) {
  torch::Tensor mask_f = legal_mask.to(logits.dtype());
  torch::Tensor legal_counts = mask_f.sum(1, true).clamp_min(1.0);
  return (logits * mask_f).sum(1, true) / legal_counts;
}

torch::Tensor ApplyLogitCapTensor(const torch::Tensor& logits,
                                  float logit_cap) {
  if (logit_cap <= 0.0f) return logits;
  return logit_cap * torch::tanh(logits / logit_cap);
}

torch::Tensor CenterAndCapLogitsTensor(const torch::Tensor& logits,
                                       const torch::Tensor& legal_mask,
                                       float logit_cap) {
  return ApplyLogitCapTensor(logits - LegalLogitMean(logits, legal_mask),
                             logit_cap);
}

void CopyModelWeights(std::shared_ptr<SharedDunePolicyValueNetImpl> source,
                      std::shared_ptr<SharedDunePolicyValueNetImpl> target) {
  torch::NoGradGuard no_grad;
  auto source_params = source->parameters();
  auto target_params = target->parameters();
  auto source_buffers = source->buffers();
  auto target_buffers = target->buffers();
  for (size_t i = 0; i < source_params.size(); ++i) {
    target_params[i].copy_(source_params[i].to(target_params[i].device()));
  }
  for (size_t i = 0; i < source_buffers.size(); ++i) {
    target_buffers[i].copy_(source_buffers[i].to(target_buffers[i].device()));
  }
}

void SyncModels(std::shared_ptr<SharedDunePolicyValueNetImpl> training_model,
                std::shared_ptr<SharedDunePolicyValueNetImpl> inference_model,
                std::shared_mutex* sync_mutex) {
  std::unique_lock<std::shared_mutex> lock(*sync_mutex);
  CopyModelWeights(training_model, inference_model);
}

void SetOptimizerLearningRate(torch::optim::Optimizer& optimizer, double lr) {
  for (auto& group : optimizer.param_groups()) {
    auto& options = static_cast<torch::optim::AdamWOptions&>(group.options());
    options.lr(lr);
  }
}

std::unique_ptr<torch::optim::AdamW> MakeOptimizer(
    std::shared_ptr<SharedDunePolicyValueNetImpl> model) {
  std::vector<torch::Tensor> policy_params;
  std::vector<torch::Tensor> other_params;
  auto policy_params_set = model->policy_head->parameters();
  for (auto& param : model->parameters()) {
    bool is_policy = false;
    for (auto& policy_param : policy_params_set) {
      if (param.is_same(policy_param)) {
        is_policy = true;
        break;
      }
    }
    if (is_policy) {
      policy_params.push_back(param);
    } else {
      other_params.push_back(param);
    }
  }

  std::vector<torch::optim::OptimizerParamGroup> groups;
  groups.emplace_back(policy_params);
  groups.emplace_back(other_params);
  auto optimizer = std::make_unique<torch::optim::AdamW>(
      groups, torch::optim::AdamWOptions(absl::GetFlag(FLAGS_learning_rate))
                  .eps(1e-5));
  static_cast<torch::optim::AdamWOptions&>(
      optimizer->param_groups()[0].options())
      .weight_decay(absl::GetFlag(FLAGS_policy_weight_decay));
  static_cast<torch::optim::AdamWOptions&>(
      optimizer->param_groups()[1].options())
      .weight_decay(absl::GetFlag(FLAGS_weight_decay));
  return optimizer;
}

void LoadCheckpoint(std::shared_ptr<SharedDunePolicyValueNetImpl> model,
                    torch::optim::AdamW& optimizer,
                    const std::string& model_path,
                    const std::string& optim_path,
                    torch::Device device) {
  if (std::filesystem::exists(model_path)) {
    try {
      torch::load(model, model_path, device);
      std::cout << "Loaded PPO model checkpoint: " << model_path << "\n";
    } catch (const c10::Error& e) {
      std::cerr << "Error loading model checkpoint: " << e.msg() << "\n";
    }
  } else {
    std::cout << "No model checkpoint at " << model_path
              << ". Starting PPO model from scratch.\n";
  }
  model->to(device);

  if (std::filesystem::exists(optim_path)) {
    try {
      torch::load(optimizer, optim_path, device);
      std::cout << "Loaded PPO optimizer checkpoint: " << optim_path << "\n";
      SetOptimizerLearningRate(optimizer, absl::GetFlag(FLAGS_learning_rate));
      for (size_t g = 0; g < optimizer.param_groups().size(); ++g) {
        auto& options =
            static_cast<torch::optim::AdamWOptions&>(
                optimizer.param_groups()[g].options());
        options.weight_decay(g == 0 ? absl::GetFlag(FLAGS_policy_weight_decay)
                                    : absl::GetFlag(FLAGS_weight_decay));
      }
      for (auto& pair : optimizer.state()) {
        if (auto* state =
                dynamic_cast<torch::optim::AdamWParamState*>(pair.second.get())) {
          if (state->exp_avg().defined()) state->exp_avg() = state->exp_avg().to(device);
          if (state->exp_avg_sq().defined()) state->exp_avg_sq() = state->exp_avg_sq().to(device);
          if (state->max_exp_avg_sq().defined()) {
            state->max_exp_avg_sq() = state->max_exp_avg_sq().to(device);
          }
        }
      }
    } catch (const c10::Error& e) {
      std::cerr << "Error loading optimizer checkpoint: " << e.msg() << "\n";
    }
  } else {
    std::cout << "No optimizer checkpoint at " << optim_path
              << ". Starting PPO optimizer from scratch.\n";
  }
}

void SaveCheckpoint(std::shared_ptr<SharedDunePolicyValueNetImpl> model,
                    torch::optim::AdamW& optimizer,
                    const std::string& model_path,
                    const std::string& optim_path) {
  try {
    torch::save(model, model_path);
    torch::save(optimizer, optim_path);
    std::cout << "Saved checkpoint: " << model_path << "\n";
  } catch (const c10::Error& e) {
    std::cerr << "Error saving checkpoint: " << e.msg() << "\n";
  }
}

template <typename RngType>
std::pair<Action, float> SamplePolicyAction(
    RngType* rng, const std::vector<float>& logits,
    const std::vector<Action>& legal_actions) {
  Action action = legal_actions.front();
  float max_logit = -std::numeric_limits<float>::infinity();
  for (Action legal_action : legal_actions) {
    if (legal_action >= 0 &&
        static_cast<size_t>(legal_action) < logits.size()) {
      max_logit = std::max(max_logit, logits[legal_action]);
    }
  }

  std::vector<double> weights;
  weights.reserve(legal_actions.size());
  double total_weight = 0.0;
  for (Action legal_action : legal_actions) {
    double weight = 1.0;
    if (legal_action >= 0 &&
        static_cast<size_t>(legal_action) < logits.size() &&
        std::isfinite(max_logit)) {
      weight = std::exp(static_cast<double>(logits[legal_action] - max_logit));
    }
    if (!std::isfinite(weight) || weight <= 0.0) weight = 1.0;
    weights.push_back(weight);
    total_weight += weight;
  }

  std::discrete_distribution<size_t> dist(weights.begin(), weights.end());
  size_t sampled_index = dist(*rng);
  action = legal_actions[sampled_index];
  double prob = total_weight > 0.0 ? weights[sampled_index] / total_weight
                                   : 1.0 / legal_actions.size();
  return {action, static_cast<float>(std::log(std::max(prob, 1e-12)))};
}



int PpoSimulation(uint64_t master, uint64_t episode_id, const Game& game,
                  std::shared_ptr<BatchedEvaluator> evaluator, int64_t obs_size,
                  std::vector<PpoTransition>* trajectory,
                  std::atomic<uint64_t>* total_env_steps,
                  WorkerStats* local_stats) {
  uint64_t env_steps_at_start = total_env_steps->load(std::memory_order_relaxed);
  int decay_horizon = std::max(1, absl::GetFlag(FLAGS_decay_horizon));
  float reward_lambda =
      env_steps_at_start < static_cast<uint64_t>(decay_horizon)
          ? 1.0f - static_cast<float>(env_steps_at_start) / decay_horizon
          : 0.0f;

  auto chance_rng = dune_seed::MakeRng64(dune_seed::DeriveSeed(master, dune_seed::kDomainTrain, episode_id, dune_seed::kStreamChance));
  std::mt19937_64 policy_rng[4];
  for (int p = 0; p < 4; ++p) {
    uint64_t seed = dune_seed::DeriveSeed(master, dune_seed::kDomainTrain, episode_id, dune_seed::kStreamPolicyPlayer0 + p);
    policy_rng[p] = dune_seed::MakeRng64(seed);
  }

  while (true) {
    std::unique_ptr<State> state = game.NewInitialState();
    bool provides_info_state_tensor =
        game.GetType().provides_information_state_tensor;
    bool provides_observations_tensor =
        game.GetType().provides_observation_tensor;

    const auto* dune_state =
        dynamic_cast<const dune_imperium::DuneImperiumState*>(state.get());

    std::vector<int> current_vps(game.NumPlayers(), 0);
    std::vector<int> current_tleilaxu(game.NumPlayers(), 0);
    std::vector<bool> had_swordmaster(game.NumPlayers(), false);
    if (dune_state != nullptr) {
      for (int p = 0; p < game.NumPlayers(); ++p) {
        current_vps[p] = dune_state->GetPlayerVpForTesting(p);
        current_tleilaxu[p] = dune_state->GetTleilaxuTrackForTesting(p);
        had_swordmaster[p] = dune_state->HasSwordmaster(p);
      }
    }

    struct CombatCreditEvent {
      int transition_index;
      int strength_delta;
    };
    std::vector<int> last_transition_index(game.NumPlayers(), -1);
    std::vector<float> shaped_bonus_by_player(game.NumPlayers(), 0.0f);
    std::vector<std::vector<CombatCreditEvent>> combat_credit(game.NumPlayers());
    std::vector<int> pre_combat_strength(game.NumPlayers(), 0);
    dune_imperium::GamePhase pre_action_phase = dune_imperium::GamePhase::kDeal;

    float logit_cap = static_cast<float>(absl::GetFlag(FLAGS_logit_cap));
    double shaped_weight = absl::GetFlag(FLAGS_shaped_reward_weight);
    double tleilaxu_breadcrumb_weight =
        absl::GetFlag(FLAGS_tleilaxu_breadcrumb_weight);
    double tleilaxu_level7_breadcrumb_weight =
        absl::GetFlag(FLAGS_tleilaxu_level7_breadcrumb_weight);

    torch::NoGradGuard no_grad;
    int game_length = 0;
    while (!state->IsTerminal()) {
      ++game_length;
      if (game_length > 5000) {
        std::cerr << "Possible infinite loop detected. State:\n"
                  << state->ToString() << "\n";
        std::abort();
      }

      if (state->IsChanceNode()) {
        std::vector<std::pair<Action, double>> outcomes = state->ChanceOutcomes();
        Action action = game.GetType().chance_mode == GameType::ChanceMode::kSampledStochastic
                    ? outcomes.front().first
                    : SampleAction(outcomes, chance_rng).first;
        state->ApplyAction(action);
        continue;
      }

      if (state->CurrentPlayer() == kSimultaneousPlayerId) {
        std::vector<Action> joint_action;
        for (int p = 0; p < game.NumPlayers(); ++p) {
          std::vector<Action> actions = state->LegalActions(p);
          if (actions.empty()) {
            joint_action.push_back(0);
          } else {
            std::uniform_int_distribution<int> dis(0, actions.size() - 1);
            joint_action.push_back(actions[dis(policy_rng[p])]);
          }
        }
        state->ApplyActions(joint_action);
        continue;
      }

      Player current_player = state->CurrentPlayer();
      std::vector<Action> actions = state->LegalActions();
      if (actions.empty()) {
        std::cerr << "Non-terminal state has no legal actions. State:\n"
                  << state->ToString() << "\n";
        std::abort();
      }

      std::vector<float> obs(obs_size, 0.0f);
      if (provides_info_state_tensor && current_player >= 0) {
        state->InformationStateTensor(current_player, absl::MakeSpan(obs));
      } else if (provides_observations_tensor && current_player >= 0) {
        state->ObservationTensor(current_player, absl::MakeSpan(obs));
      }

      EvalResult result = evaluator->Evaluate(obs);
      std::vector<float> logits = std::move(result.logits);
      CenterAndCapLegalLogits(logits, actions, logit_cap);

      auto policy_sample = SamplePolicyAction(&policy_rng[current_player], logits, actions);
      Action action = policy_sample.first;
      float old_log_prob = policy_sample.second;

      size_t transition_index = trajectory->size();

      if (dune_state != nullptr) {
        pre_action_phase = dune_state->phase();
        if (current_player >= 0 && current_player < game.NumPlayers()) {
          pre_combat_strength[current_player] =
              dune_imperium::CombatStrength(*dune_state, current_player);
        }
      }

      state->ApplyAction(action);
      total_env_steps->fetch_add(1, std::memory_order_relaxed);

      if (dune_state != nullptr) {
        for (int p = 0; p < game.NumPlayers(); ++p) {
          if (!had_swordmaster[p] && dune_state->HasSwordmaster(p)) {
            had_swordmaster[p] = true;
            int purchase_round = dune_state->GetCurrentRound();
            int leader = dune_state->PlayerLeader(p);
            local_stats->sm_acquisitions_by_seat[p]++;
            if (leader >= 0 && leader < 14) {
              local_stats->sm_acquisitions_by_leader[leader]++;
            }
            if (purchase_round >= 1 && purchase_round <= 10) {
              local_stats->sm_acquisitions_by_round[purchase_round]++;
            }
          }
        }
      }

      std::fill(shaped_bonus_by_player.begin(), shaped_bonus_by_player.end(),
                0.0f);
      if (dune_state != nullptr) {
        for (int p = 0; p < game.NumPlayers(); ++p) {
          int new_vp = dune_state->GetPlayerVpForTesting(p);
          int vp_delta = new_vp - current_vps[p];
          current_vps[p] = new_vp;
          if (vp_delta != 0) {
            shaped_bonus_by_player[p] +=
                vp_delta * static_cast<float>(shaped_weight) * reward_lambda;
          }

          int new_tleilaxu = dune_state->GetTleilaxuTrackForTesting(p);
          int tleilaxu_delta = new_tleilaxu - current_tleilaxu[p];
          current_tleilaxu[p] = new_tleilaxu;
          if (tleilaxu_delta > 0 &&
              (tleilaxu_breadcrumb_weight > 0.0 ||
               tleilaxu_level7_breadcrumb_weight > 0.0)) {
            int start_l = new_tleilaxu - tleilaxu_delta;
            for (int l = start_l + 1; l <= new_tleilaxu; ++l) {
              if (l >= 1 && l <= 6) {
                shaped_bonus_by_player[p] +=
                    static_cast<float>(tleilaxu_breadcrumb_weight) *
                    reward_lambda;
              } else if (l == 7) {
                shaped_bonus_by_player[p] +=
                    static_cast<float>(tleilaxu_level7_breadcrumb_weight) *
                    reward_lambda;
              }
            }
          }
        }
      }

      if (dune_state != nullptr &&
          pre_action_phase != dune_imperium::GamePhase::kCombat &&
          dune_state->phase() == dune_imperium::GamePhase::kCombat) {
        for (auto& events : combat_credit) events.clear();
      }

      if (dune_state != nullptr && current_player >= 0 &&
          current_player < game.NumPlayers() &&
          action != dune_imperium::kActionCombatPass) {
        int post_strength =
            dune_imperium::CombatStrength(*dune_state, current_player);
        int delta = post_strength - pre_combat_strength[current_player];
        if (delta > 0) {
          combat_credit[current_player].push_back(
              {static_cast<int>(transition_index), delta});
        }
      }

      float own_reward =
          current_player >= 0 &&
                  current_player < static_cast<int>(shaped_bonus_by_player.size())
              ? shaped_bonus_by_player[current_player]
              : 0.0f;

      if (dune_state != nullptr &&
          action == dune_imperium::kActionCombatPass && own_reward != 0.0f &&
          current_player >= 0 && current_player < game.NumPlayers()) {
        if (own_reward > 0.0f && !combat_credit[current_player].empty()) {
          int total_delta = 0;
          for (const auto& ev : combat_credit[current_player]) {
            total_delta += ev.strength_delta;
          }
          if (total_delta > 0) {
            for (const auto& ev : combat_credit[current_player]) {
              if (ev.transition_index >= 0 &&
                  ev.transition_index < static_cast<int>(trajectory->size())) {
                (*trajectory)[ev.transition_index].reward +=
                    own_reward * static_cast<float>(ev.strength_delta) /
                    static_cast<float>(total_delta);
              }
            }
          }
        }
        own_reward = 0.0f;
      }

      PpoTransition transition;
      transition.state = std::move(obs);
      transition.legal_actions = std::move(actions);
      transition.action = action;
      transition.old_log_prob = old_log_prob;
      transition.reward = own_reward;
      transition.value = result.value;
      transition.advantage = 0.0f;
      transition.return_value = 0.0f;
      transition.player_id = current_player;
      transition.episode_id = episode_id;
      trajectory->push_back(std::move(transition));

      if (current_player >= 0 && current_player < game.NumPlayers()) {
        last_transition_index[current_player] = static_cast<int>(transition_index);
      }

      for (int p = 0; p < game.NumPlayers(); ++p) {
        if (p == current_player || shaped_bonus_by_player[p] == 0.0f) continue;

        if (dune_state != nullptr &&
            action == dune_imperium::kActionCombatPass) {
          if (shaped_bonus_by_player[p] > 0.0f && !combat_credit[p].empty()) {
            int total_delta = 0;
            for (const auto& ev : combat_credit[p]) total_delta += ev.strength_delta;
            if (total_delta > 0) {
              for (const auto& ev : combat_credit[p]) {
                if (ev.transition_index >= 0 &&
                    ev.transition_index < static_cast<int>(trajectory->size())) {
                  (*trajectory)[ev.transition_index].reward +=
                      shaped_bonus_by_player[p] *
                      static_cast<float>(ev.strength_delta) /
                      static_cast<float>(total_delta);
                }
              }
            }
          }
        } else {
          int idx = last_transition_index[p];
          if (idx >= 0 && idx < static_cast<int>(trajectory->size())) {
            (*trajectory)[idx].reward += shaped_bonus_by_player[p];
          }
        }
      }
    }

    std::vector<double> terminal_returns = state->Returns();
    double gamma = absl::GetFlag(FLAGS_gamma);
    double gae_lambda = absl::GetFlag(FLAGS_gae_lambda);
    float reward_scale = static_cast<float>(
        std::max(1e-6, absl::GetFlag(FLAGS_reward_scale)));
    std::vector<float> last_value(game.NumPlayers(), 0.0f);
    std::vector<float> last_gae(game.NumPlayers(), 0.0f);
    std::vector<bool> seen_last_action(game.NumPlayers(), false);

    for (auto it = trajectory->rbegin(); it != trajectory->rend(); ++it) {
      if (it->player_id < 0 || it->player_id >= game.NumPlayers()) continue;
      int p = it->player_id;
      float reward = it->reward;
      if (!seen_last_action[p]) {
        reward += static_cast<float>(terminal_returns[p]);
        seen_last_action[p] = true;
      }
      reward = std::clamp(reward / reward_scale, -1.0f, 1.0f);

      float delta = reward + static_cast<float>(gamma) * last_value[p] -
                    it->value;
      float advantage =
          delta + static_cast<float>(gamma * gae_lambda) * last_gae[p];
      it->advantage = advantage;
      it->return_value = advantage + it->value;
      last_value[p] = it->value;
      last_gae[p] = advantage;
    }

    return game_length;
  }
}

void RolloutWorker(int thread_id, const Game* game,
                   std::shared_ptr<BatchedEvaluator> evaluator, int64_t obs_size,
                   PpoRolloutBuffer* rollout_buffer,
                   std::atomic<bool>* stop_collection,
                   std::atomic<uint64_t>* total_env_steps,
                   std::vector<WorkerStats>* worker_stats,
                   std::atomic<uint64_t>* next_episode_id,
                   uint64_t start_episode_id,
                   int rollout_games) {
  uint64_t master = absl::GetFlag(FLAGS_seed);
  WorkerStats local_stats;
  while (true) {
    if (rollout_games == 0 && stop_collection->load(std::memory_order_relaxed)) {
      break;
    }
    uint64_t episode_id = next_episode_id->fetch_add(1, std::memory_order_relaxed);
    if (rollout_games > 0 && episode_id >= start_episode_id + rollout_games) {
      break;
    }
    std::vector<PpoTransition> trajectory;
    int moves = PpoSimulation(master, episode_id, *game, evaluator, obs_size, &trajectory,
                              total_env_steps, &local_stats);
    local_stats.games += 1;
    local_stats.moves += moves;
    size_t size = rollout_buffer->PushTrajectory(std::move(trajectory));
    if (rollout_games == 0) {
      if (size >= static_cast<size_t>(absl::GetFlag(FLAGS_rollout_transitions))) {
        stop_collection->store(true, std::memory_order_relaxed);
      }
    }
  }
  (*worker_stats)[thread_id] = local_stats;
}

struct CollectResult {
  std::vector<PpoTransition> rollout;
  uint64_t games = 0;
  uint64_t moves = 0;
  double elapsed_seconds = 0.0;

  uint64_t sm_acquisitions_by_seat[4] = {0};
  uint64_t sm_acquisitions_by_leader[14] = {0};
  uint64_t sm_acquisitions_by_round[11] = {0};
};

CollectResult CollectRollout(const Game* game,
                             std::shared_ptr<BatchedEvaluator> evaluator,
                             int64_t obs_size,
                             std::atomic<uint64_t>* total_env_steps,
                             int num_threads,
                             std::atomic<uint64_t>* next_episode_id,
                             uint64_t start_episode_id,
                             int rollout_games) {
  CollectResult result;
  PpoRolloutBuffer rollout_buffer;
  std::atomic<bool> stop_collection{false};
  std::vector<WorkerStats> worker_stats(num_threads);
  std::vector<std::thread> workers;
  workers.reserve(num_threads);

  std::atomic<uint64_t> local_episode_id{start_episode_id};
  std::atomic<uint64_t>* ep_id_ptr = (rollout_games > 0) ? &local_episode_id : next_episode_id;

  auto start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < num_threads; ++i) {
    workers.emplace_back(RolloutWorker, i, game, evaluator, obs_size,
                         &rollout_buffer, &stop_collection, total_env_steps,
                         &worker_stats, ep_id_ptr, start_episode_id, rollout_games);
  }
  for (auto& worker : workers) worker.join();
  auto end = std::chrono::high_resolution_clock::now();

  result.rollout = rollout_buffer.TakeAll();
  for (const auto& stats : worker_stats) {
    result.games += stats.games;
    result.moves += stats.moves;

    for (int i = 0; i < 4; ++i) result.sm_acquisitions_by_seat[i] += stats.sm_acquisitions_by_seat[i];
    for (int i = 0; i < 14; ++i) result.sm_acquisitions_by_leader[i] += stats.sm_acquisitions_by_leader[i];
    for (int i = 0; i < 11; ++i) result.sm_acquisitions_by_round[i] += stats.sm_acquisitions_by_round[i];
  }
  result.elapsed_seconds =
      std::chrono::duration<double>(end - start).count();
  return result;
}

PpoUpdateStats TrainPpoUpdate(
    std::shared_ptr<SharedDunePolicyValueNetImpl> model,
    torch::optim::AdamW& optimizer, std::vector<PpoTransition>& batch,
    int64_t obs_size, int64_t action_dim, torch::Device device,
    uint64_t master, int global_update) {
  PpoUpdateStats stats;
  if (batch.empty()) return stats;

  int64_t n = static_cast<int64_t>(batch.size());
  auto cpu_float = torch::TensorOptions().dtype(torch::kFloat32);
  auto cpu_bool = torch::TensorOptions().dtype(torch::kBool);
  auto cpu_long = torch::TensorOptions().dtype(torch::kInt64);

  torch::Tensor states_cpu = torch::empty({n, obs_size}, cpu_float);
  torch::Tensor masks_cpu = torch::zeros({n, action_dim}, cpu_bool);
  torch::Tensor actions_cpu = torch::empty({n}, cpu_long);
  torch::Tensor old_log_probs_cpu = torch::empty({n}, cpu_float);
  torch::Tensor advantages_cpu = torch::empty({n}, cpu_float);
  torch::Tensor returns_cpu = torch::empty({n}, cpu_float);
  torch::Tensor old_values_cpu = torch::empty({n}, cpu_float);

  float* states_ptr = states_cpu.data_ptr<float>();
  bool* masks_ptr = masks_cpu.data_ptr<bool>();
  int64_t* actions_ptr = actions_cpu.data_ptr<int64_t>();
  float* old_log_probs_ptr = old_log_probs_cpu.data_ptr<float>();
  float* advantages_ptr = advantages_cpu.data_ptr<float>();
  float* returns_ptr = returns_cpu.data_ptr<float>();
  float* old_values_ptr = old_values_cpu.data_ptr<float>();

  for (int64_t i = 0; i < n; ++i) {
    const PpoTransition& transition = batch[i];
    std::memcpy(states_ptr + i * obs_size, transition.state.data(),
                obs_size * sizeof(float));
    for (Action action : transition.legal_actions) {
      if (action >= 0 && action < action_dim) {
        masks_ptr[i * action_dim + action] = true;
      }
    }
    actions_ptr[i] = transition.action;
    old_log_probs_ptr[i] = transition.old_log_prob;
    advantages_ptr[i] = transition.advantage;
    returns_ptr[i] = transition.return_value;
    old_values_ptr[i] = transition.value;
  }

  torch::Tensor states = states_cpu.to(device);
  torch::Tensor masks = masks_cpu.to(device);
  torch::Tensor actions = actions_cpu.to(device);
  torch::Tensor old_log_probs = old_log_probs_cpu.to(device);
  torch::Tensor advantages = advantages_cpu.to(device);
  torch::Tensor returns = returns_cpu.to(device);
  torch::Tensor old_values = old_values_cpu.to(device);

  int64_t minibatch_size =
      std::min<int64_t>(absl::GetFlag(FLAGS_ppo_minibatch_size), n);
  int update_epochs = std::max(1, absl::GetFlag(FLAGS_ppo_update_epochs));
  float clip_epsilon = static_cast<float>(absl::GetFlag(FLAGS_ppo_clip_epsilon));
  bool normalize_advantages = absl::GetFlag(FLAGS_normalize_advantages);
  bool clip_value_loss = absl::GetFlag(FLAGS_ppo_clip_value_loss);
  float entropy_coef = static_cast<float>(absl::GetFlag(FLAGS_entropy_coef));
  float value_coef = static_cast<float>(absl::GetFlag(FLAGS_value_coef));
  float logit_cap = static_cast<float>(absl::GetFlag(FLAGS_logit_cap));
  double target_kl = absl::GetFlag(FLAGS_target_kl);

  double policy_loss_sum = 0.0;
  double value_loss_sum = 0.0;
  double entropy_sum = 0.0;
  double kl_sum = 0.0;
  double clip_fraction_sum = 0.0;

  for (int epoch = 0; epoch < update_epochs; ++epoch) {
    uint64_t perm_seed = dune_seed::DeriveSeed(master, dune_seed::kDomainTrain, global_update, epoch, dune_seed::kStreamPPOPermutation);
    at::Generator gen = dune_seed::MakeTorchCPUGenerator(perm_seed);
    torch::Tensor permutation =
        torch::randperm(n, gen, torch::TensorOptions().device(torch::kCPU).dtype(torch::kInt64)).to(device);
    for (int64_t start = 0; start < n; start += minibatch_size) {
      int64_t end = std::min(start + minibatch_size, n);
      torch::Tensor mb_idx = permutation.narrow(0, start, end - start);

      torch::Tensor mb_states = states.index_select(0, mb_idx);
      torch::Tensor mb_masks = masks.index_select(0, mb_idx);
      torch::Tensor mb_actions = actions.index_select(0, mb_idx);
      torch::Tensor mb_old_log_probs = old_log_probs.index_select(0, mb_idx);
      torch::Tensor mb_advantages = advantages.index_select(0, mb_idx);
      torch::Tensor mb_returns = returns.index_select(0, mb_idx);
      torch::Tensor mb_old_values = old_values.index_select(0, mb_idx);

      optimizer.zero_grad();
      torch::Tensor policy_loss, value_loss, entropy, approx_kl, clip_fraction;
      torch::Tensor total_loss;

      auto compute_loss = [&]() {
        auto outputs = model->forward(mb_states);
        torch::Tensor logits =
            CenterAndCapLogitsTensor(outputs.logits, mb_masks, logit_cap);
        torch::Tensor masked_logits =
            logits.masked_fill(mb_masks.logical_not(), -1e9f);
        torch::Tensor log_probs = torch::log_softmax(masked_logits, -1);
        torch::Tensor probs = torch::softmax(masked_logits, -1);
        torch::Tensor selected_log_probs =
            log_probs.gather(1, mb_actions.unsqueeze(1)).squeeze(1);

        torch::Tensor log_ratio = selected_log_probs - mb_old_log_probs;
        torch::Tensor ratio = torch::exp(log_ratio);
        torch::Tensor mb_adv = mb_advantages;
        if (normalize_advantages && mb_adv.numel() > 1) {
          mb_adv = (mb_adv - mb_adv.mean()) /
                   (mb_adv.std(/*unbiased=*/false) + 1e-8);
        }
        mb_adv = mb_adv.detach();

        torch::Tensor pg_loss1 = -mb_adv * ratio;
        torch::Tensor pg_loss2 =
            -mb_adv * ratio.clamp(1.0f - clip_epsilon,
                                  1.0f + clip_epsilon);
        policy_loss = torch::max(pg_loss1, pg_loss2).mean();

        torch::Tensor new_values = outputs.values.squeeze(1);
        if (clip_value_loss) {
          torch::Tensor value_loss_unclipped =
              (new_values - mb_returns).pow(2);
          torch::Tensor value_clipped =
              mb_old_values +
              (new_values - mb_old_values)
                  .clamp(-clip_epsilon, clip_epsilon);
          torch::Tensor value_loss_clipped =
              (value_clipped - mb_returns).pow(2);
          value_loss =
              0.5f * torch::max(value_loss_unclipped, value_loss_clipped).mean();
        } else {
          value_loss = 0.5f * (new_values - mb_returns).pow(2).mean();
        }

        entropy = -(probs * log_probs).sum(-1).mean();
        approx_kl = ((ratio - 1.0f) - log_ratio).mean();
        clip_fraction =
            ((ratio - 1.0f).abs() > clip_epsilon).to(torch::kFloat32).mean();
        total_loss = policy_loss + value_coef * value_loss -
                     entropy_coef * entropy;
      };

      if (device.is_cuda() && absl::GetFlag(FLAGS_train_amp)) {
        AutocastGuard autocast_guard(c10::DeviceType::CUDA, true);
        compute_loss();
      } else {
        compute_loss();
      }

      total_loss.backward();
      double grad_norm =
          torch::nn::utils::clip_grad_norm_(
              model->parameters(), absl::GetFlag(FLAGS_grad_clip_norm));
      if (std::isnan(grad_norm) || std::isinf(grad_norm)) {
        std::cerr << "Fatal PPO gradient norm: " << grad_norm << "\n";
        std::exit(EXIT_FAILURE);
      }
      stats.grad_norm_sum += grad_norm;
      stats.grad_norm_count += 1;
      optimizer.step();

      double kl = approx_kl.item<double>();
      policy_loss_sum += policy_loss.item<double>();
      value_loss_sum += value_loss.item<double>();
      entropy_sum += entropy.item<double>();
      kl_sum += kl;
      clip_fraction_sum += clip_fraction.item<double>();
      stats.minibatches += 1;

      if (target_kl > 0.0 && kl > target_kl) {
        stats.early_stopped = true;
        break;
      }
    }
    if (stats.early_stopped) break;
  }

  if (stats.minibatches > 0) {
    stats.policy_loss = policy_loss_sum / stats.minibatches;
    stats.value_loss = value_loss_sum / stats.minibatches;
    stats.entropy = entropy_sum / stats.minibatches;
    stats.approx_kl = kl_sum / stats.minibatches;
    stats.clip_fraction = clip_fraction_sum / stats.minibatches;
  }

  torch::Tensor returns_cpu_flat = returns_cpu;
  torch::Tensor old_values_cpu_flat = old_values_cpu;
  double return_var = returns_cpu_flat.var(/*unbiased=*/false).item<double>();
  if (return_var > 1e-12) {
    double residual_var =
        (returns_cpu_flat - old_values_cpu_flat)
            .var(/*unbiased=*/false)
            .item<double>();
    stats.explained_variance = 1.0 - residual_var / return_var;
  } else {
    stats.explained_variance = 0.0;
  }

  return stats;
}

#endif  // OPEN_SPIEL_BUILD_WITH_LIBTORCH

}  // namespace
}  // namespace open_spiel

int main(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);
  using namespace open_spiel;

#ifndef OPEN_SPIEL_BUILD_WITH_LIBTORCH
  std::cerr << "dune_ppo_train requires OPEN_SPIEL_BUILD_WITH_LIBTORCH.\n";
  return 1;
#else
  at::set_num_threads(1);
  at::set_num_interop_threads(1);

  uint64_t master = static_cast<uint64_t>(absl::GetFlag(FLAGS_seed));
  torch::manual_seed(dune_seed::DeriveSeed(master, dune_seed::kStreamModelInit));

  auto game = open_spiel::LoadGame(absl::GetFlag(FLAGS_game));
  int64_t obs_size = game->GetType().provides_information_state_tensor
                         ? game->InformationStateTensorSize()
                         : game->ObservationTensorSize();
  int64_t action_size = game->NumDistinctActions();

  torch::Device device(torch::kCPU);
  if (torch::cuda::is_available()) {
    device = torch::Device(torch::kCUDA);
    at::globalContext().setAllowTF32CuBLAS(true);
    at::globalContext().setAllowTF32CuDNN(true);
    at::autocast::set_autocast_gpu_dtype(at::ScalarType::BFloat16);
    std::cout << "CUDA available. PPO training on GPU.\n";
  }

  auto training_model =
      std::make_shared<open_spiel::SharedDunePolicyValueNetImpl>(
          obs_size, absl::GetFlag(FLAGS_hidden_dim), action_size,
          absl::GetFlag(FLAGS_num_blocks));
  auto inference_model =
      std::make_shared<open_spiel::SharedDunePolicyValueNetImpl>(
          obs_size, absl::GetFlag(FLAGS_hidden_dim), action_size,
          absl::GetFlag(FLAGS_num_blocks));
  training_model->to(device);
  inference_model->to(device);
  training_model->train();
  inference_model->eval();

  auto optimizer = open_spiel::MakeOptimizer(training_model);
  open_spiel::LoadCheckpoint(training_model, *optimizer,
                             absl::GetFlag(FLAGS_model_checkpoint),
                             absl::GetFlag(FLAGS_optim_checkpoint), device);

  std::shared_mutex sync_mutex;
  open_spiel::SyncModels(training_model, inference_model, &sync_mutex);

  auto evaluator = std::make_shared<open_spiel::BatchedEvaluator>(
      inference_model, absl::GetFlag(FLAGS_eval_batch_size),
      absl::GetFlag(FLAGS_eval_timeout_ms), device, &sync_mutex, 0.0f,
      absl::GetFlag(FLAGS_evaluator_device_synchronize));

  std::cout << absl::StrFormat(
      "Initialized dune_ppo_train | obs=%d actions=%d hidden=%d blocks=%d "
      "rollout_transitions=%d rollout_games=%d minibatch=%d epochs=%d clip=%.3f vf_coef=%.3f "
      "ent_coef=%.4f gamma=%.3f gae_lambda=%.3f\n",
      obs_size, action_size, absl::GetFlag(FLAGS_hidden_dim),
      absl::GetFlag(FLAGS_num_blocks),
      absl::GetFlag(FLAGS_rollout_transitions),
      absl::GetFlag(FLAGS_rollout_games),
      absl::GetFlag(FLAGS_ppo_minibatch_size),
      absl::GetFlag(FLAGS_ppo_update_epochs),
      absl::GetFlag(FLAGS_ppo_clip_epsilon),
      absl::GetFlag(FLAGS_value_coef), absl::GetFlag(FLAGS_entropy_coef),
      absl::GetFlag(FLAGS_gamma), absl::GetFlag(FLAGS_gae_lambda));

  std::atomic<uint64_t> total_env_steps{0};
  std::atomic<uint64_t> next_episode_id{0};
  uint64_t total_games = 0;
  uint64_t total_moves = 0;
  auto training_start = std::chrono::high_resolution_clock::now();

  int total_updates = absl::GetFlag(FLAGS_total_updates);
  double base_lr = absl::GetFlag(FLAGS_learning_rate);
  bool pipeline = absl::GetFlag(FLAGS_pipeline);
  int num_threads = absl::GetFlag(FLAGS_threads);

  open_spiel::SearchLabelBuffer search_buffer;
  search_buffer.SetExpectedDimensions(obs_size, action_size);
  std::string search_label_dir = absl::GetFlag(FLAGS_search_label_dir);
  if (!search_label_dir.empty()) {
    search_buffer.LoadFromDirectory(search_label_dir);
  }

  int start_update = 1;
  int rollout_games = absl::GetFlag(FLAGS_rollout_games);
  uint64_t start_episode_id = (rollout_games > 0) ? (start_update - 1) * rollout_games : 0;

  // Collect first rollout synchronously.
  open_spiel::CollectResult current_collect = open_spiel::CollectRollout(
      game.get(), evaluator, obs_size, &total_env_steps, num_threads, &next_episode_id,
      start_episode_id, rollout_games);
  total_games += current_collect.games;
  total_moves += current_collect.moves;

  for (int update = 1; update <= total_updates; ++update) {
    if (absl::GetFlag(FLAGS_anneal_lr)) {
      double frac = 1.0 - static_cast<double>(update - 1) /
                              std::max(1, total_updates);
      open_spiel::SetOptimizerLearningRate(*optimizer,
                                           std::max(1e-8, frac * base_lr));
    }

    auto wall_start = std::chrono::high_resolution_clock::now();

    // Launch background collection for the next update (pipelined).
    // The inference model is frozen during PPO training, so the background
    // collection safely uses the current policy snapshot.
    open_spiel::CollectResult next_collect;
    std::thread bg_collect_thread;
    bool have_bg = pipeline && update < total_updates;
    if (have_bg) {
      uint64_t next_start_episode_id = (rollout_games > 0) ? update * rollout_games : 0;
      bg_collect_thread = std::thread([&, next_start_episode_id]() {
        next_collect = open_spiel::CollectRollout(
            game.get(), evaluator, obs_size, &total_env_steps, num_threads, &next_episode_id,
            next_start_episode_id, rollout_games);
      });
    }

    auto ppo_start = std::chrono::high_resolution_clock::now();
    open_spiel::PpoUpdateStats stats =
        open_spiel::TrainPpoUpdate(training_model, *optimizer,
                                   current_collect.rollout, obs_size,
                                   action_size, device,
                                   master, update);
    auto ppo_end = std::chrono::high_resolution_clock::now();

    // --- Search auxiliary distillation steps ---
    double search_kl_sum = 0.0;
    double search_grad_sum = 0.0;
    double search_lambda = absl::GetFlag(FLAGS_search_lambda);
    int search_minibatches_per_update = absl::GetFlag(FLAGS_search_minibatches_per_update);
    int search_minibatch_size = absl::GetFlag(FLAGS_search_minibatch_size);
    float logit_cap = static_cast<float>(absl::GetFlag(FLAGS_logit_cap));

    if (!search_label_dir.empty()) {
      search_buffer.LoadNewFiles(search_label_dir);
    }

    if (search_lambda > 0.0 && search_buffer.Size() > 0) {
      auto cpu_float = torch::TensorOptions().dtype(torch::kFloat32);
      auto cpu_bool = torch::TensorOptions().dtype(torch::kBool);

      for (int aux = 0; aux < search_minibatches_per_update; ++aux) {
        uint64_t aux_seed = dune_seed::DeriveSeed(master, dune_seed::kDomainTrain, update, aux, dune_seed::kStreamSearchSampling);
        std::mt19937 aux_rng = dune_seed::MakeRng32(aux_seed);
        std::vector<SearchLabel> search_batch = search_buffer.Sample(search_minibatch_size, &aux_rng);
        if (search_batch.empty()) continue;

        int64_t sb_size = search_batch.size();
        torch::Tensor search_states_cpu = torch::empty({sb_size, obs_size}, cpu_float);
        torch::Tensor search_masks_cpu = torch::zeros({sb_size, action_size}, cpu_bool);
        torch::Tensor search_teacher_probs_cpu = torch::zeros({sb_size, action_size}, cpu_float);

        float* states_ptr = search_states_cpu.data_ptr<float>();
        bool* masks_ptr = search_masks_cpu.data_ptr<bool>();
        float* teacher_ptr = search_teacher_probs_cpu.data_ptr<float>();

        for (int64_t i = 0; i < sb_size; ++i) {
          const auto& label = search_batch[i];
          std::memcpy(states_ptr + i * obs_size, label.state.data(), obs_size * sizeof(float));
          for (const auto& ap : label.teacher_probs) {
            int64_t action_id = ap.first;
            float prob = ap.second;
            if (action_id >= 0 && action_id < action_size) {
              masks_ptr[i * action_size + action_id] = true;
              teacher_ptr[i * action_size + action_id] = prob;
            }
          }
        }

        torch::Tensor search_states = search_states_cpu.to(device);
        torch::Tensor search_masks = search_masks_cpu.to(device);
        torch::Tensor search_teacher_probs = search_teacher_probs_cpu.to(device);

        optimizer->zero_grad();

        torch::Tensor mean_kl;
        auto compute_distill_loss = [&]() {
          auto outputs = training_model->forward(search_states);
          torch::Tensor logits = CenterAndCapLogitsTensor(outputs.logits, search_masks, logit_cap);
          torch::Tensor masked_logits = logits.masked_fill(search_masks.logical_not(), -1e9f);
          torch::Tensor log_probs = torch::log_softmax(masked_logits, -1);

          torch::Tensor log_teacher = torch::log(search_teacher_probs.clamp_min(1e-12f));
          torch::Tensor kl_loss = search_teacher_probs * (log_teacher - log_probs);
          mean_kl = kl_loss.sum(-1).mean();
        };

        if (device.is_cuda() && absl::GetFlag(FLAGS_train_amp)) {
          AutocastGuard autocast_guard(c10::DeviceType::CUDA, true);
          compute_distill_loss();
        } else {
          compute_distill_loss();
        }

        torch::Tensor search_loss = search_lambda * mean_kl;
        search_loss.backward();

        double search_grad_norm = torch::nn::utils::clip_grad_norm_(
            training_model->parameters(), std::numeric_limits<double>::infinity());
        if (!std::isfinite(search_grad_norm) || !std::isfinite(mean_kl.item<double>())) {
          std::cerr << "Warning: Non-finite search distillation gradient (" << search_grad_norm
                    << ") or loss. Skipping optimizer step.\n";
          optimizer->zero_grad();
          continue;
        }
        torch::nn::utils::clip_grad_norm_(
            training_model->parameters(), absl::GetFlag(FLAGS_grad_clip_norm));
        optimizer->step();

        search_kl_sum += mean_kl.item<double>();
        search_grad_sum += search_grad_norm;
      }
    }

    // Join background collection before syncing models.
    if (have_bg) {
      bg_collect_thread.join();
      total_games += next_collect.games;
      total_moves += next_collect.moves;
    }

    open_spiel::SyncModels(training_model, inference_model, &sync_mutex);

    auto wall_end = std::chrono::high_resolution_clock::now();

    double collect_elapsed = current_collect.elapsed_seconds;
    double ppo_elapsed =
        std::chrono::duration<double>(ppo_end - ppo_start).count();
    double wall_elapsed =
        std::chrono::duration<double>(wall_end - wall_start).count();
    double sps = collect_elapsed > 0.0
                     ? current_collect.rollout.size() / collect_elapsed
                     : 0.0;

    std::cout << absl::StrFormat(
        "Update %5d/%d | Transitions: %6zu | Games: %llu | "
        "Collect: %.2fs (%.0f t/s) | PPO: %.2fs | Wall: %.2fs | "
        "PolicyLoss: %.6f | ValueLoss: %.6f | Entropy: %.4f | "
        "ApproxKL: %.6f | ClipFrac: %.3f | ExplVar: %.3f%s\n",
        update, total_updates, current_collect.rollout.size(),
        static_cast<unsigned long long>(total_games),
        collect_elapsed, sps, ppo_elapsed, wall_elapsed,
        stats.policy_loss, stats.value_loss, stats.entropy, stats.approx_kl,
        stats.clip_fraction, stats.explained_variance,
        stats.early_stopped ? " | early-stop" : "");



    {
      uint64_t total_acquisitions = 0;
      for (int i = 0; i < 4; ++i) total_acquisitions += current_collect.sm_acquisitions_by_seat[i];
      if (total_acquisitions > 0) {
        std::cout << absl::StrFormat(
            "  [Natural SM Acquisitions] Total: %d | Seat P0: %d | P1: %d | P2: %d | P3: %d\n",
            total_acquisitions, current_collect.sm_acquisitions_by_seat[0],
            current_collect.sm_acquisitions_by_seat[1], current_collect.sm_acquisitions_by_seat[2],
            current_collect.sm_acquisitions_by_seat[3]);
        std::cout << absl::StrFormat(
            "  [SM Acquisitions by Round] R1: %d | R2: %d | R3: %d | R4: %d | R5: %d | R6-10: %d\n",
            current_collect.sm_acquisitions_by_round[1], current_collect.sm_acquisitions_by_round[2],
            current_collect.sm_acquisitions_by_round[3], current_collect.sm_acquisitions_by_round[4],
            current_collect.sm_acquisitions_by_round[5],
            current_collect.sm_acquisitions_by_round[6] + current_collect.sm_acquisitions_by_round[7] +
            current_collect.sm_acquisitions_by_round[8] + current_collect.sm_acquisitions_by_round[9] +
            current_collect.sm_acquisitions_by_round[10]);
        std::cout << absl::StrFormat(
            "  [SM Acquisitions by Leader] Leto (4): %d | Rabban (6): %d | Other Leaders: %d\n",
            current_collect.sm_acquisitions_by_leader[4], current_collect.sm_acquisitions_by_leader[6],
            total_acquisitions - current_collect.sm_acquisitions_by_leader[4] - current_collect.sm_acquisitions_by_leader[6]);
      }
    }

    if (search_lambda > 0.0 && search_buffer.Size() > 0) {
      double avg_search_kl = (search_minibatches_per_update > 0) ? (search_kl_sum / search_minibatches_per_update) : 0.0;
      double avg_search_grad = (search_minibatches_per_update > 0) ? (search_grad_sum / search_minibatches_per_update) : 0.0;
      double avg_ppo_grad = (stats.grad_norm_count > 0) ? (stats.grad_norm_sum / stats.grad_norm_count) : 0.0;
      double ratio = (avg_ppo_grad > 0.0) ? (avg_search_grad / avg_ppo_grad) : 0.0;
      std::cout << absl::StrFormat(
          "SearchKL: %.3f | SearchGrad: %.3f | PPOGrad: %.3f | Ratio: %.2f | EV: %.3f\n",
          avg_search_kl, avg_search_grad, avg_ppo_grad, ratio, stats.explained_variance);
    }

    int checkpoint_interval = absl::GetFlag(FLAGS_checkpoint_interval);
    if (checkpoint_interval > 0 && update % checkpoint_interval == 0) {
      std::string prefix = absl::GetFlag(FLAGS_run_prefix);
      std::string model_path =
          absl::StrCat(prefix, "_model_update_", update, ".pt");
      std::string optim_path =
          absl::StrCat(prefix, "_optimizer_update_", update, ".pt");
      open_spiel::SaveCheckpoint(training_model, *optimizer, model_path,
                                 optim_path);
    }

    // Advance to next rollout.
    if (have_bg) {
      current_collect = std::move(next_collect);
    } else if (update < total_updates) {
      uint64_t next_start_episode_id = (rollout_games > 0) ? update * rollout_games : 0;
      current_collect = open_spiel::CollectRollout(
          game.get(), evaluator, obs_size, &total_env_steps, num_threads, &next_episode_id,
          next_start_episode_id, rollout_games);
      total_games += current_collect.games;
      total_moves += current_collect.moves;
    }
  }

  auto training_end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = training_end - training_start;
  auto eval_stats = evaluator->GetStats();
  std::cout << absl::StrFormat(
      "\n=== PPO Training Complete ===\nElapsed: %.2fs\nEnv Steps: %llu\n"
      "Games: %llu\nMoves: %llu\nEvaluator Requests: %llu | Batches: %llu | "
      "Avg Batch: %.2f | Max Batch: %llu\n",
      elapsed.count(),
      static_cast<unsigned long long>(total_env_steps.load()),
      static_cast<unsigned long long>(total_games),
      static_cast<unsigned long long>(total_moves),
      static_cast<unsigned long long>(eval_stats.requests),
      static_cast<unsigned long long>(eval_stats.batches),
      eval_stats.avg_batch_size,
      static_cast<unsigned long long>(eval_stats.max_batch_size));



  if (absl::GetFlag(FLAGS_save_final_checkpoint)) {
    open_spiel::SaveCheckpoint(training_model, *optimizer,
                               absl::GetFlag(FLAGS_model_checkpoint),
                               absl::GetFlag(FLAGS_optim_checkpoint));
  }
  return 0;
#endif
}
