// Timed, multi-threaded Dune Imperium self-play benchmark driver
// Supports both a pure C++ engine baseline and a LibTorch tensor inference benchmark.

#include <iostream>
#include <random>
#include <vector>
#include <thread>
#include <atomic>
#include <chrono>
#include <memory>
#include <deque>
#include <mutex>
#include <shared_mutex>
#include <iterator>
#include <filesystem>
#include <optional>
#include <limits>
#include <string>
#include <sstream>

#include "open_spiel/abseil-cpp/absl/flags/flag.h"
#include "open_spiel/abseil-cpp/absl/flags/parse.h"
#include "open_spiel/abseil-cpp/absl/strings/str_format.h"
#include "open_spiel/spiel.h"
#ifdef OPEN_SPIEL_BUILD_WITH_LIBTORCH
#include <torch/torch.h>
#include "dune_network.h"
#include <cmath>
#include <algorithm>
#include "open_spiel/games/dune_imperium/dune_imperium.h"
#include "open_spiel/games/dune_imperium/dune_imperium_util.h"
#endif

ABSL_FLAG(std::string, game, "dune_imperium", "The name of the game to play.");
ABSL_FLAG(int, games, 1000, "How many games to play in total.");
ABSL_FLAG(int, threads, 8, "How many threads to run.");
ABSL_FLAG(std::string, model_checkpoint, "dune_stage_a_model.pt", "Path to load/save model checkpoint.");
ABSL_FLAG(std::string, optim_checkpoint, "dune_stage_a_optimizer.pt", "Path to load/save optimizer checkpoint.");
ABSL_FLAG(bool, async_mode, false, "Whether to run optimization asynchronously in a background thread.");
ABSL_FLAG(int, min_train_size, 1024, "Minimum buffer transitions before starting SGD training.");
ABSL_FLAG(int, train_batch_size, 32, "Batch size for background SGD updates.");
ABSL_FLAG(int, sync_interval, 100, "Number of training steps between weight synchronizations.");
ABSL_FLAG(int, buffer_capacity, 30000, "Maximum capacity of the global replay buffer.");
ABSL_FLAG(int, train_ratio, 64, "The number of NEW moves the CPU must generate before the GPU is allowed to pull 1 training batch. Set <= 0 to disable throttling.");
ABSL_FLAG(int, decay_horizon, 12000000, "Number of training steps over which to linearly decay the shaped reward lambda.");
ABSL_FLAG(double, shaped_reward_weight, 0.2, "Weight multiplier for each VP gained or lost in intermediate shaped rewards.");
ABSL_FLAG(double, tleilaxu_breadcrumb_weight, 0.0, "Weight multiplier for Tleilaxu progress breadcrumbs (levels 4->5 and 5->6).");
ABSL_FLAG(double, tleilaxu_level7_breadcrumb_weight, 0.0, "Weight multiplier for the Tleilaxu level 7 VP milestone breadcrumb.");
ABSL_FLAG(double, swordmaster_breadcrumb_weight, 0.0, "Weight multiplier for one-time Swordmaster acquisition breadcrumb.");
ABSL_FLAG(double, temperature, 1.0, "Softmax temperature for action selection (1.0 = standard, >1.0 = explore, 0.0 = greedy).");
ABSL_FLAG(double, learning_rate, 1e-4, "Learning rate for the Adam optimizer.");
ABSL_FLAG(double, mmd_eta, 0.2, "MMD entropy parameter eta.");
ABSL_FLAG(double, mmd_alpha, 0.1, "MMD entropy parameter alpha.");
ABSL_FLAG(std::string, mmd_reference_mode, "uniform", "MMD reference policy mode: uniform, periodic, or ema.");
ABSL_FLAG(int, mmd_reference_interval, 50000, "Training steps between reference-policy refreshes when --mmd_reference_mode=periodic.");
ABSL_FLAG(double, mmd_reference_ema_decay, 0.999, "Parameter EMA decay for --mmd_reference_mode=ema.");
ABSL_FLAG(std::string, policy_update_mode, "mmd", "Policy update mode: mmd or ppo.");
ABSL_FLAG(double, ppo_clip_epsilon, 0.2, "PPO ratio clipping epsilon when --policy_update_mode=ppo.");
ABSL_FLAG(int, eval_batch_size, 64, "Target batch size for the evaluation queue.");
ABSL_FLAG(int, eval_timeout_ms, 2, "Timeout in milliseconds for the evaluation queue.");
ABSL_FLAG(int, hidden_dim, 2048, "Hidden dimension of the network.");
ABSL_FLAG(int, num_blocks, 8, "Number of residual blocks.");
ABSL_FLAG(double, grad_clip_norm, 1.0, "Maximum gradient norm for clipping.");
ABSL_FLAG(double, gae_lambda, 1.0, "Lambda parameter for GAE.");
ABSL_FLAG(int, start_step, 0, "Global training step to resume from to keep decay math aligned.");
ABSL_FLAG(std::string, run_prefix, "dune_stage_c", "Prefix for saving rotating checkpoints.");
ABSL_FLAG(double, logit_cap, 10.0, "Smooth tanh cap on policy logits to prevent runaway growth.");
ABSL_FLAG(double, entropy_coef, 0.01, "Entropy bonus coefficient (maximizes policy entropy to prevent collapse).");
ABSL_FLAG(double, weight_decay, 1e-4, "AdamW decoupled weight decay coefficient.");
ABSL_FLAG(double, policy_weight_decay, 1e-2, "AdamW decoupled weight decay coefficient for the policy head only.");
ABSL_FLAG(double, mmd_importance_clip, 20.0, "Maximum 1 / behavior_prob multiplier for sampled MMD targets. Set <= 0 to disable clipping.");
ABSL_FLAG(int, max_train_steps, 0, "Maximum local async SGD steps to execute before stopping collection. Set <= 0 to disable.");
ABSL_FLAG(bool, save_final_checkpoint, true, "Whether to save model and optimizer checkpoints when the benchmark exits.");
ABSL_FLAG(bool, train_amp, false, "Use CUDA BF16 autocast for the training forward/loss path.");
ABSL_FLAG(int, loss_read_interval, 1, "Read CPU loss scalars every N local train steps. 1 preserves per-step NaN checks/log values.");
ABSL_FLAG(int, train_diagnostic_interval, 1000, "Print expensive training diagnostics every N TrainStep calls. Set <= 0 to disable.");
ABSL_FLAG(bool, evaluator_device_synchronize, true, "Use whole-device CUDA synchronize after evaluator D2H copies. False uses blocking D2H copies instead.");

ABSL_FLAG(std::string, opponent_checkpoints, "", "Comma-separated list of paths to opponent model checkpoints.");
ABSL_FLAG(std::string, opponent_mix, "", "Comma-separated list of probabilities/weights for each opponent checkpoint.");
ABSL_FLAG(std::string, opponent_hidden_dims, "", "Comma-separated list of hidden dimensions for each opponent checkpoint.");
ABSL_FLAG(std::string, opponent_num_blocks, "", "Comma-separated list of block counts for each opponent checkpoint.");
ABSL_FLAG(std::string, learner_seat_mode, "self_play", "Mode of seat assignment: self_play or rotate.");
ABSL_FLAG(double, opponent_temperature, 0.0, "Softmax temperature for opponent action selection.");

namespace open_spiel {

std::atomic<uint64_t> recorded_transitions_by_seat[4] = {0, 0, 0, 0};
std::atomic<uint64_t> opponent_decisions_count{0};
std::atomic<uint64_t> learner_seat_games_count[4] = {0, 0, 0, 0};

struct Transition {
  std::vector<float> spatial_tensor;     // Flattened observation tensor
  std::vector<Action> legal_actions;     // Sparse list of legal action IDs (saves 99% RAM)
  int64_t action_taken;                  // Action ID chosen
  std::vector<float> prev_logits;        // Legal-centered, soft-capped behavior policy logits (z_k)
  float q_value;                         // Sampled policy signal for MMD
  float advantage;                       // Raw GAE advantage before MMD importance correction
  float reward_target;                   // TD(lambda) value target
  float v_value;                         // Value Network estimate V(s)
  float behavior_prob;                   // Probability of action_taken under the sampling policy
  float importance_multiplier;           // Clipped sampled-MMD importance multiplier
  int64_t training_step_inserted;         // Optimizer step when trajectory entered replay
  int player_id;                         // Seat ID of the deciding player
};

class GlobalReplayBuffer {
 private:
  std::deque<std::shared_ptr<Transition>> buffer_;
  std::mutex buffer_mutex_;
  size_t max_capacity_;
  std::atomic<uint64_t> total_inserted_{0};

 public:
  GlobalReplayBuffer(size_t capacity) : max_capacity_(capacity) {}

  void PushTrajectory(std::vector<Transition>& local_trajectory,
                      int64_t training_step_inserted = 0) {
    std::lock_guard<std::mutex> lock(buffer_mutex_);
    size_t num_inserted = local_trajectory.size();
    for (auto& transition : local_trajectory) {
      transition.training_step_inserted = training_step_inserted;
      buffer_.push_back(std::make_shared<Transition>(std::move(transition)));
    }
    while (buffer_.size() > max_capacity_) {
      buffer_.pop_front();
    }
    total_inserted_.fetch_add(num_inserted, std::memory_order_relaxed);
  }

  std::vector<std::shared_ptr<Transition>> SampleBatch(size_t batch_size) {
    std::lock_guard<std::mutex> lock(buffer_mutex_);
    std::vector<std::shared_ptr<Transition>> batch;
    if (buffer_.empty()) return batch;
    
    batch.reserve(batch_size);
    std::uniform_int_distribution<size_t> dis(0, buffer_.size() - 1);
    static thread_local std::mt19937 rng(std::random_device{}());
    
    for (size_t i = 0; i < batch_size; ++i) {
      batch.push_back(buffer_[dis(rng)]);
    }
    return batch;
  }

  size_t Size() {
    std::lock_guard<std::mutex> lock(buffer_mutex_);
    return buffer_.size();
  }

  uint64_t GetTotalInserted() const {
    return total_inserted_.load(std::memory_order_relaxed);
  }
};

std::vector<std::string> SplitCommaSeparated(const std::string& value) {
  std::vector<std::string> items;
  std::stringstream ss(value);
  std::string item;
  while (std::getline(ss, item, ',')) {
    item.erase(item.begin(), std::find_if(item.begin(), item.end(), [](unsigned char ch) {
      return !std::isspace(ch);
    }));
    item.erase(std::find_if(item.rbegin(), item.rend(), [](unsigned char ch) {
      return !std::isspace(ch);
    }).base(), item.end());
    if (!item.empty()) items.push_back(item);
  }
  return items;
}

bool IsLearnerTurn(Player current_player, Player learner_seat, const std::string& learner_seat_mode) {
  if (learner_seat_mode == "rotate") {
    return current_player == learner_seat;
  }
  return true; // Default self_play mode: all seats are learners
}

namespace {

#ifdef OPEN_SPIEL_BUILD_WITH_LIBTORCH
// Shared network architecture imported from dune_network.h

void LoadCheckpoint(std::shared_ptr<SharedDunePolicyValueNetImpl> model, 
                    std::unique_ptr<torch::optim::AdamW>& optimizer, 
                    const std::string& model_path, 
                    const std::string& optim_path,
                    torch::Device device) {
  if (std::filesystem::exists(model_path)) {
    try {
      torch::load(model, model_path, device);
      std::cout << "Successfully loaded model weights from " << model_path << "\n";
    } catch (const c10::Error& e) {
      std::cerr << "Error loading checkpoint: " << e.msg() << "\n";
    }
  } else {
    std::cout << "No existing checkpoint found at " << model_path << ". Starting from scratch.\n";
  }

  // Move model to its target hardware device AFTER loading weights
  model->to(device);

  double learning_rate = absl::GetFlag(FLAGS_learning_rate);
  double weight_decay = absl::GetFlag(FLAGS_weight_decay);
  double policy_weight_decay = absl::GetFlag(FLAGS_policy_weight_decay);

  std::vector<torch::Tensor> policy_params;
  std::vector<torch::Tensor> other_params;
  auto policy_params_set = model->policy_head->parameters();
  for (auto& param : model->parameters()) {
    bool is_policy = false;
    for (auto& p : policy_params_set) {
      if (param.is_same(p)) {
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
  groups.push_back(torch::optim::OptimizerParamGroup(policy_params)); // Group 0: Policy Head
  groups.push_back(torch::optim::OptimizerParamGroup(other_params));  // Group 1: Torso/Value Head

  // Initialize AdamW Optimizer with parameter groups
  optimizer = std::make_unique<torch::optim::AdamW>(groups, torch::optim::AdamWOptions(learning_rate));

  // Explicitly apply weight decays on the groups
  static_cast<torch::optim::AdamWOptions&>(optimizer->param_groups()[0].options()).weight_decay(policy_weight_decay);
  static_cast<torch::optim::AdamWOptions&>(optimizer->param_groups()[1].options()).weight_decay(weight_decay);

  if (std::filesystem::exists(optim_path)) {
    try {
      torch::load(*optimizer, optim_path, device);
      std::cout << "Successfully loaded optimizer state from " << optim_path << "\n";

      // CRITICAL FIX: PyTorch's load() overwrites the options with the ones saved in the file.
      // We must explicitly override learning rate and weight decays to ensure the groups are correct!
      for (size_t g = 0; g < optimizer->param_groups().size(); ++g) {
        auto& param_group = optimizer->param_groups()[g];
        if (param_group.has_options()) {
          auto& options = static_cast<torch::optim::AdamWOptions&>(param_group.options());
          options.lr(learning_rate);
          if (g == 0) {
            options.weight_decay(policy_weight_decay);
          } else {
            options.weight_decay(weight_decay);
          }
        }
      }
      
      // Move all loaded optimizer state tensors (momentum buffers) to target device to prevent CPU-GPU mismatches
      for (auto& pair : optimizer->state()) {
        auto& state_ptr = pair.second;
        if (auto adam_state = dynamic_cast<torch::optim::AdamWParamState*>(state_ptr.get())) {
          if (adam_state->exp_avg().defined()) {
            adam_state->exp_avg() = adam_state->exp_avg().to(device);
          }
          if (adam_state->exp_avg_sq().defined()) {
            adam_state->exp_avg_sq() = adam_state->exp_avg_sq().to(device);
          }
          if (adam_state->max_exp_avg_sq().defined()) {
            adam_state->max_exp_avg_sq() = adam_state->max_exp_avg_sq().to(device);
          }
        }
      }
    } catch (const c10::Error& e) {
      std::cerr << "Error loading optimizer checkpoint: " << e.msg() << "\n";
    }
  } else {
    std::cout << "Optimizer checkpoint not found at " << optim_path << ". Starting with default optimizer state.\n";
  }
}

void SaveCheckpoint(std::shared_ptr<SharedDunePolicyValueNetImpl> model, 
                    torch::optim::AdamW& optimizer, 
                    const std::string& model_path, 
                    const std::string& optim_path) {
  try {
    torch::save(model, model_path);
    torch::save(optimizer, optim_path);
    std::cout << "Checkpoint saved successfully.\n";
  } catch (const c10::Error& e) {
    std::cerr << "Error saving checkpoint: " << e.msg() << "\n";
  }
}
struct PersistentTrainBuffers {
  // Pinned host memory
  torch::Tensor states, masks, rewards, prev_logits, q_values;
  torch::Tensor advantages, behavior_probs, actions_taken;
  float *p_states, *p_masks, *p_rewards, *p_prev, *p_q;
  float *p_advantages, *p_behavior_probs;
  int64_t *p_actions;

  // Persistent device-static tensors for zero-allocation H2D transfers
  torch::Tensor d_states, d_masks, d_rewards, d_prev_logits, d_q_values;
  torch::Tensor d_advantages, d_behavior_probs, d_actions_taken;

  bool is_cuda_ = false;

  void Initialize(int64_t max_batch_size, int64_t obs_size, int64_t action_dim, bool is_cuda) {
    is_cuda_ = is_cuda;
    auto float_opts = torch::TensorOptions().dtype(torch::kFloat32).pinned_memory(is_cuda);
    auto long_opts = torch::TensorOptions().dtype(torch::kInt64).pinned_memory(is_cuda);

    states = torch::empty({max_batch_size, obs_size}, float_opts);
    masks = torch::empty({max_batch_size, action_dim}, float_opts);
    rewards = torch::empty({max_batch_size, 1}, float_opts);
    prev_logits = torch::empty({max_batch_size, action_dim}, float_opts);
    q_values = torch::empty({max_batch_size, 1}, float_opts);
    advantages = torch::empty({max_batch_size, 1}, float_opts);
    behavior_probs = torch::empty({max_batch_size, 1}, float_opts);
    actions_taken = torch::empty({max_batch_size, 1}, long_opts);

    p_states = states.data_ptr<float>();
    p_masks = masks.data_ptr<float>();
    p_rewards = rewards.data_ptr<float>();
    p_prev = prev_logits.data_ptr<float>();
    p_q = q_values.data_ptr<float>();
    p_advantages = advantages.data_ptr<float>();
    p_behavior_probs = behavior_probs.data_ptr<float>();
    p_actions = actions_taken.data_ptr<int64_t>();

    if (is_cuda) {
      auto d_f = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
      auto d_l = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA);
      
      d_states = torch::empty({max_batch_size, obs_size}, d_f);
      d_masks = torch::empty({max_batch_size, action_dim}, d_f);
      d_rewards = torch::empty({max_batch_size, 1}, d_f);
      d_prev_logits = torch::empty({max_batch_size, action_dim}, d_f);
      d_q_values = torch::empty({max_batch_size, 1}, d_f);
      d_advantages = torch::empty({max_batch_size, 1}, d_f);
      d_behavior_probs = torch::empty({max_batch_size, 1}, d_f);
      d_actions_taken = torch::empty({max_batch_size, 1}, d_l);
    }
  }
};

torch::Tensor LegalLogitMean(const torch::Tensor& logits,
                             const torch::Tensor& masks) {
  torch::Tensor legal_counts = masks.sum(1, true).clamp_min(1.0);
  return (logits * masks).sum(1, true) / legal_counts;
}

torch::Tensor CenterLegalLogitsTensor(const torch::Tensor& logits,
                                      const torch::Tensor& masks) {
  return logits - LegalLogitMean(logits, masks);
}

torch::Tensor ApplyLogitCapTensor(const torch::Tensor& logits,
                                  float logit_cap) {
  if (logit_cap <= 0.0f) return logits;
  return logit_cap * torch::tanh(logits / logit_cap);
}

float LegalMaskedMaxAbs(const torch::Tensor& logits,
                        const torch::Tensor& masks) {
  torch::Tensor legal_values = logits.abs().masked_select(masks == 1.0f);
  if (legal_values.numel() == 0) return 0.0f;
  return legal_values.max().item<float>();
}

float Percentile(std::vector<float> values, double quantile) {
  if (values.empty()) return 0.0f;
  std::sort(values.begin(), values.end());
  quantile = std::clamp(quantile, 0.0, 1.0);
  size_t index = static_cast<size_t>(std::llround(quantile * (values.size() - 1)));
  index = std::min(index, values.size() - 1);
  return values[index];
}

bool IsReferenceModeEnabled(const std::string& mode) {
  return mode == "periodic" || mode == "ema";
}

bool IsPolicyUpdateModeValid(const std::string& mode) {
  return mode == "mmd" || mode == "ppo";
}

void CopyModelWeights(std::shared_ptr<SharedDunePolicyValueNetImpl> source_model,
                      std::shared_ptr<SharedDunePolicyValueNetImpl> target_model) {
  torch::NoGradGuard no_grad;

  auto source_params = source_model->parameters();
  auto target_params = target_model->parameters();
  auto source_buffers = source_model->buffers();
  auto target_buffers = target_model->buffers();

  for (size_t i = 0; i < source_params.size(); ++i) {
    target_params[i].copy_(source_params[i].to(target_params[i].device()));
  }
  for (size_t i = 0; i < source_buffers.size(); ++i) {
    target_buffers[i].copy_(source_buffers[i].to(target_buffers[i].device()));
  }
}

void UpdateReferenceModelEma(
    std::shared_ptr<SharedDunePolicyValueNetImpl> source_model,
    std::shared_ptr<SharedDunePolicyValueNetImpl> reference_model,
    double decay) {
  torch::NoGradGuard no_grad;
  decay = std::clamp(decay, 0.0, 1.0);
  double update_weight = 1.0 - decay;

  auto source_params = source_model->parameters();
  auto reference_params = reference_model->parameters();
  auto source_buffers = source_model->buffers();
  auto reference_buffers = reference_model->buffers();

  for (size_t i = 0; i < source_params.size(); ++i) {
    torch::Tensor source = source_params[i].to(reference_params[i].device());
    reference_params[i].mul_(decay);
    reference_params[i].add_(source, update_weight);
  }
  for (size_t i = 0; i < source_buffers.size(); ++i) {
    reference_buffers[i].copy_(source_buffers[i].to(reference_buffers[i].device()));
  }
}

std::pair<float, float> TrainStep(std::shared_ptr<SharedDunePolicyValueNetImpl> model, 
                                  std::shared_ptr<SharedDunePolicyValueNetImpl> reference_model,
                                  torch::optim::Optimizer& optimizer, 
                                  const std::vector<std::shared_ptr<Transition>>& batch,
                                  PersistentTrainBuffers& buffers,
                                  int64_t obs_size,
                                  int64_t action_dim,
                                  torch::Device device,
                                  int64_t current_training_step,
                                  bool read_loss_scalars);

void SyncModels(std::shared_ptr<SharedDunePolicyValueNetImpl> training_model, 
                std::shared_ptr<SharedDunePolicyValueNetImpl> inference_model,
                std::shared_mutex* sync_mutex) {
  std::unique_lock<std::shared_mutex> lock(*sync_mutex);
  CopyModelWeights(training_model, inference_model);
}

void OptimizationWorker(
    std::shared_ptr<SharedDunePolicyValueNetImpl> training_model,
    std::shared_ptr<SharedDunePolicyValueNetImpl> inference_model,
    std::shared_ptr<SharedDunePolicyValueNetImpl> reference_model,
    torch::optim::AdamW* optimizer,
    GlobalReplayBuffer* global_buffer,
    int64_t obs_size,
    int64_t action_size,
    std::shared_mutex* sync_mutex,
    std::atomic<bool>& stop_training,
    std::atomic<bool>& stop_collection,
    std::atomic<int>& training_steps,
    int min_train_size,
    int train_batch_size,
    int sync_interval,
    int max_train_steps,
    const std::string& model_path,
    const std::string& optim_path,
    torch::Device device,
    std::atomic<float>* reward_lambda) {
  try {
    // 1. Warmup Phase: wait until the global replay buffer has enough transitions
    while (global_buffer->Size() < static_cast<size_t>(min_train_size) && !stop_training.load()) {
      std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }

    if (!stop_training.load()) {
      std::cout << "Replay buffer warmup completed. Starting background optimization loop...\n" << std::flush;
    }

    std::string reference_mode = absl::GetFlag(FLAGS_mmd_reference_mode);
    std::string policy_update_mode = absl::GetFlag(FLAGS_policy_update_mode);
    bool use_mmd = policy_update_mode == "mmd";
    bool reference_enabled =
        use_mmd && IsReferenceModeEnabled(reference_mode) &&
        reference_model != nullptr;
    int reference_interval = std::max(1, absl::GetFlag(FLAGS_mmd_reference_interval));
    double reference_ema_decay = absl::GetFlag(FLAGS_mmd_reference_ema_decay);
    if (reference_enabled) {
      CopyModelWeights(training_model, reference_model);
      reference_model->eval();
      std::cout << absl::StrFormat(
          "MMD reference policy enabled: mode=%s, interval=%d, ema_decay=%.6f\n",
          reference_mode, reference_interval, reference_ema_decay);
    }

    // 2. Training Loop
    PersistentTrainBuffers train_buffers;
    train_buffers.Initialize(train_batch_size, obs_size, action_size, device.is_cuda());
    float last_v_loss = std::numeric_limits<float>::quiet_NaN();
    float last_p_loss = std::numeric_limits<float>::quiet_NaN();
    auto optimization_start_time = std::chrono::high_resolution_clock::now();

    while (!stop_training.load()) {
      int train_ratio = absl::GetFlag(FLAGS_train_ratio);
      if (train_ratio > 0) {
        uint64_t current_inserted = global_buffer->GetTotalInserted();
        if (current_inserted >= static_cast<uint64_t>(min_train_size)) {
          uint64_t allowed_steps = (current_inserted - min_train_size) / train_ratio;
          uint64_t local_steps = static_cast<uint64_t>(training_steps.load()) - absl::GetFlag(FLAGS_start_step);
          if (local_steps >= allowed_steps) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
          }
        }
      }

      std::vector<std::shared_ptr<Transition>> batch = global_buffer->SampleBatch(train_batch_size);
      if (batch.empty()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        continue;
      }

      int next_step = training_steps.load(std::memory_order_relaxed) + 1;
      int loss_read_interval = absl::GetFlag(FLAGS_loss_read_interval);
      bool read_loss_scalars =
          loss_read_interval <= 1 || (next_step % loss_read_interval == 0);

      // Lock-free optimization step on training_model
      auto [v_loss, p_loss] = TrainStep(
          training_model, reference_model, *optimizer, batch, train_buffers,
          obs_size, action_size, device, next_step, read_loss_scalars);
      if (read_loss_scalars) {
        last_v_loss = v_loss;
        last_p_loss = p_loss;
      }

      int step = training_steps.fetch_add(1) + 1;
      int local_steps = step - absl::GetFlag(FLAGS_start_step);

      if (reference_enabled) {
        if (reference_mode == "ema") {
          UpdateReferenceModelEma(training_model, reference_model,
                                  reference_ema_decay);
        } else if (reference_mode == "periodic" &&
                   local_steps > 0 &&
                   local_steps % reference_interval == 0) {
          CopyModelWeights(training_model, reference_model);
          reference_model->eval();
          std::cout << absl::StrFormat(
              "[MMD Reference] Refreshed periodic reference at step %d "
              "(local %d).\n",
              step, local_steps);
        }
      }

      // Update linear decay for the intermediate reward shaping lambda
      if (reward_lambda != nullptr) {
        int decay_horizon = absl::GetFlag(FLAGS_decay_horizon);
        if (step < decay_horizon) {
          float new_lambda = 1.0f - (static_cast<float>(step) / decay_horizon);
          reward_lambda->store(new_lambda, std::memory_order_relaxed);
        } else {
          reward_lambda->store(0.0f, std::memory_order_relaxed);
        }
      }

      // Periodic model synchronization
      if (step % sync_interval == 0) {
        SyncModels(training_model, inference_model, sync_mutex);
        float current_lambda = (reward_lambda != nullptr) ? reward_lambda->load(std::memory_order_relaxed) : 0.0f;
        std::cout << absl::StrFormat("Step %5d | Buffer Size: %6zu | Lambda: %.4f | Value Loss (MSE): %.6f | Policy Loss (KL): %.6f\n",
                                     step, global_buffer->Size(), current_lambda, last_v_loss, last_p_loss) << std::flush;
      }

      if (max_train_steps > 0 && local_steps >= max_train_steps) {
        std::cout << absl::StrFormat(
            "Reached max local training steps (%d). Stopping collection after active games finish.\n",
            max_train_steps) << std::flush;
        stop_collection.store(true, std::memory_order_relaxed);
        stop_training.store(true, std::memory_order_relaxed);
        break;
      }

      // Periodic checkpoint saving (runs in background, no sync_mutex needed)
      if (step % 50000 == 0) {
        // ROTATING CHECKPOINTS: Enforce saving to a new numbered file every time
        std::string run_prefix = absl::GetFlag(FLAGS_run_prefix);
        std::string step_model_path = absl::StrCat(run_prefix, "_model_step_", step, ".pt");
        std::string step_optim_path = absl::StrCat(run_prefix, "_optimizer_step_", step, ".pt");
        SaveCheckpoint(training_model, *optimizer, step_model_path, step_optim_path);
      }
    }
    
    // Final synchronization at shutdown to ensure inference_model has latest weights
    SyncModels(training_model, inference_model, sync_mutex);
    auto optimization_end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> optimization_elapsed =
        optimization_end_time - optimization_start_time;
    int local_steps_done = training_steps.load() - absl::GetFlag(FLAGS_start_step);
    double optimization_seconds = optimization_elapsed.count();
    if (optimization_seconds > 0.0) {
      std::cout << absl::StrFormat(
          "Optimizer Timing: %.3f sec | Local Steps: %d | Steps/sec: %.2f | Samples/sec: %.1f\n",
          optimization_seconds, local_steps_done,
          static_cast<double>(local_steps_done) / optimization_seconds,
          static_cast<double>(local_steps_done) * train_batch_size / optimization_seconds);
    }
    std::cout << "Background optimization worker stopped cleanly. Total steps: " << training_steps.load() << "\n";
  } catch (const c10::Error& e) {
    std::cerr << "\n============================================\n"
              << "CRITICAL PyTorch error in Asynchronous Optimization Worker thread:\n"
              << e.what() << "\n"
              << "============================================\n" << std::endl;
    std::abort();
  } catch (const std::exception& e) {
    std::cerr << "\n============================================\n"
              << "CRITICAL Exception in Asynchronous Optimization Worker thread:\n"
              << e.what() << "\n"
              << "============================================\n" << std::endl;
    std::abort();
  }
}

std::pair<float, float> TrainStep(std::shared_ptr<SharedDunePolicyValueNetImpl> model, 
                                  std::shared_ptr<SharedDunePolicyValueNetImpl> reference_model,
                                  torch::optim::Optimizer& optimizer, 
                                  const std::vector<std::shared_ptr<Transition>>& batch,
                                  PersistentTrainBuffers& buffers,
                                  int64_t obs_size,
                                  int64_t action_dim,
                                  torch::Device device,
                                  int64_t current_training_step,
                                  bool read_loss_scalars) {
  size_t batch_size = batch.size();
  if (batch_size == 0) return {0.0f, 0.0f};

  for (size_t i = 0; i < batch_size; ++i) {
    const auto& t = batch[i];
    std::memcpy(buffers.p_states + i * obs_size, t->spatial_tensor.data(), obs_size * sizeof(float));
    std::memcpy(buffers.p_prev + i * action_dim, t->prev_logits.data(), action_dim * sizeof(float));
    std::fill(buffers.p_masks + i * action_dim, buffers.p_masks + (i + 1) * action_dim, 0.0f);
    for (Action a : t->legal_actions) {
      if (a >= 0 && a < action_dim) buffers.p_masks[i * action_dim + a] = 1.0f;
    }
    buffers.p_rewards[i] = t->reward_target;
    buffers.p_q[i] = t->q_value;
    buffers.p_advantages[i] = t->advantage;
    buffers.p_behavior_probs[i] = std::max(t->behavior_prob, 1e-8f);
    buffers.p_actions[i] = t->action_taken;
  }

  // Zero-Allocation H2D Transfer: copy from pinned host into persistent device memory
  if (buffers.is_cuda_) {
    buffers.d_states.slice(0, 0, batch_size).copy_(buffers.states.slice(0, 0, batch_size), /*non_blocking=*/true);
    buffers.d_masks.slice(0, 0, batch_size).copy_(buffers.masks.slice(0, 0, batch_size), /*non_blocking=*/true);
    buffers.d_rewards.slice(0, 0, batch_size).copy_(buffers.rewards.slice(0, 0, batch_size), /*non_blocking=*/true);
    buffers.d_prev_logits.slice(0, 0, batch_size).copy_(buffers.prev_logits.slice(0, 0, batch_size), /*non_blocking=*/true);
    buffers.d_q_values.slice(0, 0, batch_size).copy_(buffers.q_values.slice(0, 0, batch_size), /*non_blocking=*/true);
    buffers.d_advantages.slice(0, 0, batch_size).copy_(buffers.advantages.slice(0, 0, batch_size), /*non_blocking=*/true);
    buffers.d_behavior_probs.slice(0, 0, batch_size).copy_(buffers.behavior_probs.slice(0, 0, batch_size), /*non_blocking=*/true);
    buffers.d_actions_taken.slice(0, 0, batch_size).copy_(buffers.actions_taken.slice(0, 0, batch_size), /*non_blocking=*/true);
  }

  // Use persistent device tensors (zero-alloc) on CUDA, or sliced pinned tensors on CPU
  torch::Tensor states = buffers.is_cuda_ ? buffers.d_states.slice(0, 0, batch_size) : buffers.states.slice(0, 0, batch_size);
  torch::Tensor masks = buffers.is_cuda_ ? buffers.d_masks.slice(0, 0, batch_size) : buffers.masks.slice(0, 0, batch_size);
  torch::Tensor rewards = buffers.is_cuda_ ? buffers.d_rewards.slice(0, 0, batch_size) : buffers.rewards.slice(0, 0, batch_size);
  torch::Tensor prev_logits = buffers.is_cuda_ ? buffers.d_prev_logits.slice(0, 0, batch_size) : buffers.prev_logits.slice(0, 0, batch_size);
  torch::Tensor q_values = buffers.is_cuda_ ? buffers.d_q_values.slice(0, 0, batch_size) : buffers.q_values.slice(0, 0, batch_size);
  torch::Tensor advantages = buffers.is_cuda_ ? buffers.d_advantages.slice(0, 0, batch_size) : buffers.advantages.slice(0, 0, batch_size);
  torch::Tensor behavior_probs_tensor = buffers.is_cuda_ ? buffers.d_behavior_probs.slice(0, 0, batch_size) : buffers.behavior_probs.slice(0, 0, batch_size);
  torch::Tensor actions_taken = buffers.is_cuda_ ? buffers.d_actions_taken.slice(0, 0, batch_size) : buffers.actions_taken.slice(0, 0, batch_size);

  optimizer.zero_grad();
  torch::Tensor raw_pred_logits, centered_raw_logits, pred_logits, pred_values;
  torch::Tensor masked_logits, log_probs, value_loss, q_vector, target_logits;
  torch::Tensor masked_target_logits, target_probs, policy_loss, probs;
  torch::Tensor entropy_per_sample, mean_entropy, total_loss;
  torch::Tensor raw_legal_mean, centered_prev_logits, reference_logits;
  torch::Tensor centered_reference_logits;
  float logit_cap = static_cast<float>(absl::GetFlag(FLAGS_logit_cap));
  float eta = static_cast<float>(absl::GetFlag(FLAGS_mmd_eta));
  float alpha = static_cast<float>(absl::GetFlag(FLAGS_mmd_alpha));
  float entropy_coef = static_cast<float>(absl::GetFlag(FLAGS_entropy_coef));
  float ppo_clip_epsilon =
      static_cast<float>(absl::GetFlag(FLAGS_ppo_clip_epsilon));
  std::string policy_update_mode = absl::GetFlag(FLAGS_policy_update_mode);
  bool use_mmd = policy_update_mode == "mmd";
  bool use_ppo = policy_update_mode == "ppo";
  std::string reference_mode = absl::GetFlag(FLAGS_mmd_reference_mode);
  bool use_reference =
      use_mmd && IsReferenceModeEnabled(reference_mode) && reference_model != nullptr;

  auto compute_loss = [&]() {
    auto outputs = model->forward(states);
    raw_pred_logits = outputs.logits;
    pred_values = outputs.values;

    // Policy logits are invariant to a per-state additive constant. Center over
    // legal actions before applying the cap so a harmless common offset cannot
    // push the whole policy head into tanh saturation.
    raw_legal_mean = LegalLogitMean(raw_pred_logits, masks);
    centered_raw_logits = raw_pred_logits - raw_legal_mean;
    pred_logits = ApplyLogitCapTensor(centered_raw_logits, logit_cap);

    masked_logits = pred_logits.masked_fill(masks == 0.0f, -1e9f);
    log_probs = torch::log_softmax(masked_logits, -1);
    value_loss = torch::nn::functional::mse_loss(pred_values, rewards);

    centered_prev_logits = CenterLegalLogitsTensor(prev_logits, masks);
    if (use_mmd) {
      q_vector = torch::zeros({(int64_t)batch_size, action_dim},
                              torch::TensorOptions().device(device));
      q_vector.scatter_(1, actions_taken, q_values);

      target_logits = centered_prev_logits + eta * q_vector;
      if (use_reference) {
        torch::NoGradGuard no_grad;
        auto reference_outputs = reference_model->forward(states);
        reference_logits = reference_outputs.logits;
        centered_reference_logits =
            ApplyLogitCapTensor(CenterLegalLogitsTensor(reference_logits, masks),
                                logit_cap);
        target_logits = target_logits + eta * alpha * centered_reference_logits;
      }
      target_logits = target_logits / (1.0f + eta * alpha);
      target_logits = CenterLegalLogitsTensor(target_logits, masks);
      masked_target_logits = target_logits.masked_fill(masks == 0.0f, -1e9f);
      target_probs = torch::softmax(masked_target_logits, -1).detach();

      policy_loss = torch::nn::functional::kl_div(
          log_probs, target_probs,
          torch::nn::functional::KLDivFuncOptions().reduction(torch::kBatchMean));
    } else if (use_ppo) {
      torch::Tensor selected_log_probs = log_probs.gather(1, actions_taken);
      torch::Tensor old_log_probs = behavior_probs_tensor.clamp_min(1e-8).log();
      torch::Tensor ratios = torch::exp(selected_log_probs - old_log_probs);
      torch::Tensor centered_advantages = advantages - advantages.mean();
      torch::Tensor advantage_std =
          centered_advantages.pow(2).mean().sqrt().clamp_min(1e-6);
      torch::Tensor normalized_advantages =
          (centered_advantages / advantage_std).detach();
      torch::Tensor clipped_ratios =
          ratios.clamp(1.0f - ppo_clip_epsilon, 1.0f + ppo_clip_epsilon);
      policy_loss =
          -torch::min(ratios * normalized_advantages,
                      clipped_ratios * normalized_advantages)
               .mean();
    } else {
      SPIEL_CHECK_TRUE(false);
    }

    // ENTROPY BONUS: Maximize entropy over legal actions to prevent policy collapse
    probs = torch::softmax(masked_logits, -1);
    // Entropy = -sum(p * log(p)) over legal actions; masked slots contribute ~0
    entropy_per_sample = -(probs * log_probs).sum(-1); // [batch]
    mean_entropy = entropy_per_sample.mean();

    total_loss = policy_loss + value_loss - entropy_coef * mean_entropy;
  };

  if (device.is_cuda() && absl::GetFlag(FLAGS_train_amp)) {
    AutocastGuard autocast_guard(c10::DeviceType::CUDA, true);
    compute_loss();
  } else {
    compute_loss();
  }

  float p_loss_val = std::numeric_limits<float>::quiet_NaN();
  float v_loss_val = std::numeric_limits<float>::quiet_NaN();
  bool have_loss_scalars = false;
  auto read_loss_values = [&]() {
    if (!have_loss_scalars) {
      p_loss_val = policy_loss.item<float>();
      v_loss_val = value_loss.item<float>();
      have_loss_scalars = true;
    }
  };
  if (read_loss_scalars) {
    read_loss_values();
  }

  // Diagnostic for eta scaling and logit health
  static int log_counter = 0;
  int diagnostic_step = ++log_counter;
  int diagnostic_interval = absl::GetFlag(FLAGS_train_diagnostic_interval);
  if (diagnostic_interval > 0 && diagnostic_step % diagnostic_interval == 0) {
      read_loss_values();
      float mean_abs_policy_signal = q_values.abs().mean().item<float>();
      float std_policy_signal = q_values.std().item<float>();
      float raw_all_max_abs_logit = raw_pred_logits.abs().max().item<float>();
      float raw_legal_offset_abs = raw_legal_mean.abs().mean().item<float>();
      float raw_centered_legal_max_abs = LegalMaskedMaxAbs(centered_raw_logits, masks);
      float capped_legal_max_abs = LegalMaskedMaxAbs(pred_logits, masks);
      float target_legal_max_abs =
          use_mmd ? LegalMaskedMaxAbs(target_logits, masks) : 0.0f;
      float reference_legal_max_abs =
          use_reference ? LegalMaskedMaxAbs(centered_reference_logits, masks)
                        : 0.0f;
      torch::Tensor behavior_masked_logits =
          centered_prev_logits.masked_fill(masks == 0.0f, -1e9f);
      torch::Tensor behavior_log_probs =
          torch::log_softmax(behavior_masked_logits, -1);
      torch::Tensor behavior_probs =
          torch::softmax(behavior_masked_logits, -1);
      float behavior_current_kl =
          (behavior_probs * (behavior_log_probs - log_probs))
              .sum(-1)
              .mean()
              .item<float>();
      // Top-1 softmax confidence (want < ~0.9, not ~0.97)
      float mean_top1_prob = std::get<0>(probs.max(-1)).mean().item<float>();
      float mean_ent = mean_entropy.item<float>();

      std::vector<float> abs_q_values;
      std::vector<float> abs_advantages;
      std::vector<float> abs_reward_targets;
      std::vector<float> importance_multipliers;
      std::vector<float> replay_ages;
      abs_q_values.reserve(batch_size);
      abs_advantages.reserve(batch_size);
      abs_reward_targets.reserve(batch_size);
      importance_multipliers.reserve(batch_size);
      replay_ages.reserve(batch_size);
      for (const auto& transition : batch) {
        abs_q_values.push_back(std::abs(transition->q_value));
        abs_advantages.push_back(std::abs(transition->advantage));
        abs_reward_targets.push_back(std::abs(transition->reward_target));
        importance_multipliers.push_back(transition->importance_multiplier);
        int64_t age =
            std::max<int64_t>(0, current_training_step -
                                     transition->training_step_inserted);
        replay_ages.push_back(static_cast<float>(age));
      }

      double total_std_logits = 0.0;
      int count = 0;
      for (size_t i = 0; i < batch_size; ++i) {
          auto mask_i = masks[i];
          auto logits_i = pred_logits[i];
          auto legal_logits_i = logits_i.masked_select(mask_i == 1.0f);
          if (legal_logits_i.numel() > 1) {
              total_std_logits += legal_logits_i.std().item<float>();
              count++;
          }
      }
      float mean_std_logits = (count > 0) ? (total_std_logits / count) : 0.0f;
      float policy_weight_norm = model->policy_head->weight.norm().item<float>();

      std::cout << "\n[Diagnostic] Mean|PolicySignal|: " << mean_abs_policy_signal
                << " | Std|PolicySignal|: " << std_policy_signal
                << " | PolicyMode: " << policy_update_mode
                << " | Eta: " << eta
                << " | RawAllMax|Logit|: " << raw_all_max_abs_logit
                << " | RawLegalOffset: " << raw_legal_offset_abs
                << " | RawMax|LegalCentered|: " << raw_centered_legal_max_abs
                << " | CapMax|Legal|: " << capped_legal_max_abs
                << " | TargetMax|Legal|: " << target_legal_max_abs
                << " | RefMax|Legal|: " << reference_legal_max_abs
                << " | LogitStd: " << mean_std_logits
                << " | PolNorm: " << policy_weight_norm
                << " | MeanTop1: " << mean_top1_prob
                << " | Entropy: " << mean_ent
                << " | PolicyLoss: " << p_loss_val
                << " | BehaviorToCurrentKL: " << behavior_current_kl
                << " | ReplayAge(p50/p95/p99): "
                << Percentile(replay_ages, 0.50) << "/"
                << Percentile(replay_ages, 0.95) << "/"
                << Percentile(replay_ages, 0.99)
                << " | |Q|(p50/p95/p99): "
                << Percentile(abs_q_values, 0.50) << "/"
                << Percentile(abs_q_values, 0.95) << "/"
                << Percentile(abs_q_values, 0.99)
                << " | |Adv|(p50/p95/p99): "
                << Percentile(abs_advantages, 0.50) << "/"
                << Percentile(abs_advantages, 0.95) << "/"
                << Percentile(abs_advantages, 0.99)
                << " | |Target|(p50/p95/p99): "
                << Percentile(abs_reward_targets, 0.50) << "/"
                << Percentile(abs_reward_targets, 0.95) << "/"
                << Percentile(abs_reward_targets, 0.99)
                << " | Importance(p50/p95/p99): "
                << Percentile(importance_multipliers, 0.50) << "/"
                << Percentile(importance_multipliers, 0.95) << "/"
                << Percentile(importance_multipliers, 0.99)
                << "\n";
  }

  // 1. FATAL SAFETY GUARD & TENSOR AUTOPSY
  if (have_loss_scalars && (std::isnan(p_loss_val) || std::isnan(v_loss_val))) {
    std::cerr << "\n================ TENSOR AUTOPSY =================\n";
    std::cerr << "[FATAL] NaN loss detected! Isolating the source...\n";
    std::cerr << "Losses -> Policy (KL): " << p_loss_val << " | Value (MSE): " << v_loss_val << "\n";
    
    auto check_nan = [](const torch::Tensor& t, const std::string& name) {
        if (!t.defined()) return;
        bool has_nan = t.isnan().any().item<bool>();
        bool has_inf = t.isinf().any().item<bool>();
        std::cerr << " - '" << name << "' has NaN/Inf: " << (has_nan || has_inf ? "YES" : "NO") << "\n";
    };

    std::cerr << "\nINPUTS (From C++ Replay Buffer):\n";
    check_nan(states, "states");
    check_nan(rewards, "rewards");
    check_nan(prev_logits, "prev_logits");
    check_nan(q_values, "q_values");
    check_nan(advantages, "advantages");
    check_nan(behavior_probs_tensor, "behavior_probs");
    
    std::cerr << "\nMODEL OUTPUTS (Forward Pass):\n";
    check_nan(raw_pred_logits, "raw_pred_logits");
    check_nan(centered_raw_logits, "centered_raw_logits");
    check_nan(pred_logits, "pred_logits");
    check_nan(pred_values, "pred_values");
    
    std::cerr << "\nLOSS MATH INTERMEDIATES:\n";
    check_nan(log_probs, "log_probs");
    check_nan(target_probs, "target_probs");
    
    std::cerr << "\nMODEL WEIGHTS:\n";
    bool weights_bad = false;
    for (const auto& param : model->parameters()) {
        if (param.isnan().any().item<bool>() || param.isinf().any().item<bool>()) {
            weights_bad = true;
            break;
        }
    }
    std::cerr << " - Model weights have NaN/Inf: " << (weights_bad ? "YES" : "NO") << "\n";

    std::cerr << "\nEDGE CASES:\n";
    std::cerr << " - Any state has ZERO valid actions? " << ((masks.sum(1) == 0.0f).any().item<bool>() ? "YES" : "NO") << "\n";
    
    std::cerr << "=================================================\n";
    std::cerr << "Aborting to protect healthy checkpoints on disk.\n";
    std::exit(EXIT_FAILURE);
  }

  total_loss.backward();

  // 2. GRADIENT CLIPPING & SECONDARY GUARD
  double grad_norm = torch::nn::utils::clip_grad_norm_(model->parameters(), absl::GetFlag(FLAGS_grad_clip_norm));
  
  if (std::isnan(grad_norm) || std::isinf(grad_norm)) {
      std::cerr << "\n=========================================\n";
      std::cerr << "[FATAL] NaN/Inf detected in Gradient Norm: " << grad_norm << "\n";
      std::cerr << "The backward pass produced invalid gradients (likely BF16 overflow).\n";
      std::cerr << "Aborting optimizer step to prevent injecting NaNs into model weights.\n";
      std::cerr << "=========================================\n";
      std::exit(EXIT_FAILURE);
  }

  optimizer.step();

  return {v_loss_val, p_loss_val};
}

int TorchSimulation(std::mt19937* rng, const Game& game, std::shared_ptr<BatchedEvaluator> evaluator, int64_t obs_size, std::vector<Transition>& trajectory, std::atomic<float>* reward_lambda,
                    const std::vector<std::shared_ptr<BatchedEvaluator>>& opponent_evaluators,
                    const std::vector<double>& opponent_mix,
                    const std::string& learner_seat_mode,
                    int game_id) {
  std::unique_ptr<State> state = game.NewInitialState();

  Player learner_seat = -1;
  if (learner_seat_mode == "rotate") {
    learner_seat = game_id % 4;
    learner_seat_games_count[learner_seat].fetch_add(1, std::memory_order_relaxed);
  }

  std::vector<std::shared_ptr<BatchedEvaluator>> seat_evaluators(game.NumPlayers(), nullptr);
  if (learner_seat_mode == "rotate") {
    std::discrete_distribution<size_t> opponent_dist(opponent_mix.begin(), opponent_mix.end());
    for (int p = 0; p < game.NumPlayers(); ++p) {
      if (p == learner_seat) {
        seat_evaluators[p] = evaluator;
      } else {
        size_t opp_idx = opponent_dist(*rng);
        SPIEL_CHECK_LT(opp_idx, opponent_evaluators.size());
        seat_evaluators[p] = opponent_evaluators[opp_idx];
      }
    }
  } else {
    for (int p = 0; p < game.NumPlayers(); ++p) {
      seat_evaluators[p] = evaluator;
    }
  }

  bool provides_info_state_tensor =
      game.GetType().provides_information_state_tensor;
  bool provides_observations_tensor =
      game.GetType().provides_observation_tensor;

  // CRITICAL: Disable gradient tracking for self-play generation
  torch::NoGradGuard no_grad;

  // Initialize VP tracking for all players (before the action loop)
  const dune_imperium::DuneImperiumState* dune_state = 
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
  std::vector<int> last_transition_index(game.NumPlayers(), -1);
  std::vector<float> shaped_bonus_by_player(game.NumPlayers(), 0.0f);

  // Combat-causal credit tracking: per-player list of {transition_index,
  // strength_delta} pairs for transitions where CombatStrength() increased.
  // Combat VP is distributed proportionally by delta weight, avoiding
  // crediting the trivially forced CombatPass action.
  struct CombatCreditEvent {
    int transition_index;
    int strength_delta;
  };
  std::vector<std::vector<CombatCreditEvent>> combat_credit(game.NumPlayers());
  std::vector<int> pre_combat_strength(game.NumPlayers(), 0);
  dune_imperium::GamePhase pre_action_phase = dune_imperium::GamePhase::kDeal;

  // Micro-optimized flag retrieval outside the state progression loop
  double temp = absl::GetFlag(FLAGS_temperature);
  double opponent_temp = absl::GetFlag(FLAGS_opponent_temperature);
  double shaped_weight = absl::GetFlag(FLAGS_shaped_reward_weight);
  double tleilaxu_breadcrumb_weight = absl::GetFlag(FLAGS_tleilaxu_breadcrumb_weight);
  double tleilaxu_level7_breadcrumb_weight = absl::GetFlag(FLAGS_tleilaxu_level7_breadcrumb_weight);
  double swordmaster_breadcrumb_weight = absl::GetFlag(FLAGS_swordmaster_breadcrumb_weight);
  float logit_cap = static_cast<float>(absl::GetFlag(FLAGS_logit_cap));

  int game_length = 0;
  while (!state->IsTerminal()) {
    ++game_length;
    if (game_length > 5000) {
      std::cerr << "Possible infinite loop detected! Game length: " << game_length
                << " Player: " << state->CurrentPlayer()
                << "\nState string:\n" << state->ToString() << std::endl;
      std::abort();
    }
    if (state->IsChanceNode()) {
      std::vector<std::pair<Action, double>> outcomes = state->ChanceOutcomes();
      Action action;
      if (game.GetType().chance_mode ==
          GameType::ChanceMode::kSampledStochastic) {
        action = outcomes.front().first;
      } else {
        action = SampleAction(outcomes, *rng).first;
      }
      state->ApplyAction(action);
    } else if (state->CurrentPlayer() == kSimultaneousPlayerId) {
      std::vector<Action> joint_action;
      for (int p = 0; p < game.NumPlayers(); p++) {
        std::vector<Action> actions = state->LegalActions(p);
        Action action = 0;
        if (!actions.empty()) {
          std::uniform_int_distribution<int> dis(0, actions.size() - 1);
          action = actions[dis(*rng)];
        }
        joint_action.push_back(action);
      }
      state->ApplyActions(joint_action);
    } else {
      Player current_player = state->CurrentPlayer();
      std::vector<Action> actions = state->LegalActions();
      if (actions.empty()) {
        std::cerr << "Spiel Fatal Error: Non-terminal state has empty LegalActions(). Player: " 
                  << state->CurrentPlayer() << "\nState string:\n" << state->ToString() << std::endl;
        std::abort();
      }

      // 1. Allocate and populate observation vector via span
      std::vector<float> obs(obs_size, 0.0f);
      if (provides_info_state_tensor && current_player >= 0) {
        state->InformationStateTensor(current_player, absl::MakeSpan(obs));
      } else if (provides_observations_tensor && current_player >= 0) {
        state->ObservationTensor(current_player, absl::MakeSpan(obs));
      }

      int64_t action_size = game.NumDistinctActions();

      std::shared_ptr<BatchedEvaluator> current_evaluator = seat_evaluators[current_player];
      SPIEL_CHECK_TRUE(current_evaluator != nullptr);

      if (learner_seat_mode == "rotate" && current_player != learner_seat) {
        opponent_decisions_count.fetch_add(1, std::memory_order_relaxed);
      }

      // Query the batched evaluator (actor thread goes to sleep, OS multiplexes)
      EvalResult result = current_evaluator->Evaluate(obs);
      float current_value = result.value;
      std::vector<float> prev_logits_vec = std::move(result.logits);
      CenterAndCapLegalLogits(prev_logits_vec, actions, logit_cap);

      // Fast CPU Softmax Action Selection using the configured temperature
      double curr_temp = IsLearnerTurn(current_player, learner_seat, learner_seat_mode) ? temp : opponent_temp;
      Action action = actions.front();
      float behavior_prob = 1.0f;
      if (curr_temp > 0.001) {
        std::vector<double> action_probs;
        action_probs.reserve(actions.size());
        
        // Find max logit among LEGAL actions for numerical stability
        float max_logit = -1e9f;
        for (Action a : actions) {
          if (prev_logits_vec[a] > max_logit) max_logit = prev_logits_vec[a];
        }
        
        for (Action a : actions) {
          action_probs.push_back(std::exp((prev_logits_vec[a] - max_logit) / curr_temp));
        }
        
        // Sample action probabilistically from the network's beliefs
        std::discrete_distribution<size_t> dist(action_probs.begin(), action_probs.end());
        size_t sampled_idx = dist(*rng);
        action = actions[sampled_idx];

        double total_weight = 0.0;
        for (double weight : action_probs) {
          total_weight += weight;
        }
        if (total_weight > 0.0 && std::isfinite(total_weight)) {
          behavior_prob = static_cast<float>(action_probs[sampled_idx] / total_weight);
        } else {
          behavior_prob = 1.0f / static_cast<float>(actions.size());
        }
      } else {
        // Fallback Argmax (Greedy selection when Temp is 0)
        float max_logit = -1e9f;
        for (Action a : actions) {
          if (prev_logits_vec[a] > max_logit) {
            max_logit = prev_logits_vec[a];
            action = a;
          }
        }
      }

      size_t current_transition_index = trajectory.size();

      // Snapshot phase and current player's combat strength before action
      if (dune_state != nullptr) {
        pre_action_phase = dune_state->phase();
        if (current_player >= 0 && current_player < game.NumPlayers()) {
          pre_combat_strength[current_player] =
              dune_imperium::CombatStrength(*dune_state, current_player);
        }
      }

      state->ApplyAction(action);

      // Shaped Reward VP Gain Detection
      //
      // VP can be awarded by deferred resolution triggered on another player's
      // action, e.g. combat rewards on the final CombatPass. Track every
      // player's VP after every decision and retro-credit positive deltas to
      // that player's most recent transition.
      std::fill(shaped_bonus_by_player.begin(), shaped_bonus_by_player.end(), 0.0f);
      if (dune_state != nullptr) {
        float current_lambda = reward_lambda != nullptr
                                   ? reward_lambda->load(std::memory_order_relaxed)
                                   : 0.0f;
        for (int p = 0; p < game.NumPlayers(); ++p) {
          int new_vp = dune_state->GetPlayerVpForTesting(p);
          int vp_delta = new_vp - current_vps[p];
          current_vps[p] = new_vp;

          if (vp_delta != 0 && reward_lambda != nullptr) {
            shaped_bonus_by_player[p] +=
                vp_delta * static_cast<float>(shaped_weight) * current_lambda;
          }

          // Tleilaxu breadcrumb reward calculation
          int new_tleilaxu = dune_state->GetTleilaxuTrackForTesting(p);
          int tleilaxu_delta = new_tleilaxu - current_tleilaxu[p];
          current_tleilaxu[p] = new_tleilaxu;

          if (tleilaxu_delta > 0 && reward_lambda != nullptr &&
              (tleilaxu_breadcrumb_weight > 0.0 ||
               tleilaxu_level7_breadcrumb_weight > 0.0)) {
            int approach_steps = 0;
            int level7_steps = 0;
            int start_l = new_tleilaxu - tleilaxu_delta;
            for (int l = start_l + 1; l <= new_tleilaxu; ++l) {
              if (l == 5 || l == 6) {
                approach_steps++;
              } else if (l == 7) {
                level7_steps++;
              }
            }
            if (approach_steps > 0) {
              shaped_bonus_by_player[p] +=
                  approach_steps * static_cast<float>(tleilaxu_breadcrumb_weight) * current_lambda;
            }
            if (level7_steps > 0) {
              shaped_bonus_by_player[p] +=
                  level7_steps * static_cast<float>(tleilaxu_level7_breadcrumb_weight) * current_lambda;
            }
          }

          // Swordmaster breadcrumb: round-decayed reward when acquired
          if (swordmaster_breadcrumb_weight > 0.0 && !had_swordmaster[p] &&
              dune_state->HasSwordmaster(p)) {
            had_swordmaster[p] = true;
            int round = dune_state->GetCurrentRound();
            double round_multiplier;
            if (round <= 2)      round_multiplier = 1.00;
            else if (round == 3) round_multiplier = 0.85;
            else if (round == 4) round_multiplier = 0.65;
            else if (round == 5) round_multiplier = 0.45;
            else if (round == 6) round_multiplier = 0.25;
            else if (round == 7) round_multiplier = 0.15;
            else if (round == 8) round_multiplier = 0.075;
            else                 round_multiplier = 0.00;  // R9+ no breadcrumb
            shaped_bonus_by_player[p] +=
                static_cast<float>(swordmaster_breadcrumb_weight *
                                   round_multiplier) *
                current_lambda;
          }
        }
      }

      // Clear combat tracking when a new combat phase begins
      if (dune_state != nullptr &&
          pre_action_phase != dune_imperium::GamePhase::kCombat &&
          dune_state->phase() == dune_imperium::GamePhase::kCombat) {
        for (auto& cc : combat_credit) cc.clear();
      }

      if (IsLearnerTurn(current_player, learner_seat, learner_seat_mode)) {
        // Track combat-causal transitions: record when CombatStrength increases.
        // Captures troop/dread deployments AND sword-boosting combat intrigues.
        // Excludes CombatPass, Harvest Cells, Economic Positioning, etc.
        if (dune_state != nullptr && current_player >= 0 &&
            current_player < game.NumPlayers() &&
            action != dune_imperium::kActionCombatPass) {
          int post_strength =
              dune_imperium::CombatStrength(*dune_state, current_player);
          int delta = post_strength - pre_combat_strength[current_player];
          if (delta > 0) {
            combat_credit[current_player].push_back(
                {static_cast<int>(current_transition_index), delta});
          }
        }

        // Push to trajectory
        Transition transition;
        transition.spatial_tensor = std::move(obs);
        transition.legal_actions = std::move(actions); // Store sparse legal actions list directly
        transition.action_taken = action;
        transition.prev_logits = std::move(prev_logits_vec);
        float own_shaped_bonus =
            (current_player >= 0 &&
             current_player < static_cast<int>(shaped_bonus_by_player.size()))
                 ? shaped_bonus_by_player[current_player]
                 : 0.0f;

        // If CombatPass triggered own VP: redistribute positive VP to
        // combat-causal transitions weighted by strength delta.
        // Suppress negative VP on CombatPass (alliance theft side effects).
        // Suppress positive VP if no combat events exist (GAE handles it).
        if (action == dune_imperium::kActionCombatPass && own_shaped_bonus != 0.0f &&
            current_player >= 0 && current_player < game.NumPlayers()) {
          if (own_shaped_bonus > 0.0f &&
              !combat_credit[current_player].empty()) {
            int total_delta = 0;
            for (const auto& ev : combat_credit[current_player])
              total_delta += ev.strength_delta;
            if (total_delta > 0) {
              for (const auto& ev : combat_credit[current_player]) {
                float weight = static_cast<float>(ev.strength_delta) /
                               static_cast<float>(total_delta);
                float bonus = own_shaped_bonus * weight;
                if (ev.transition_index >= 0 &&
                    ev.transition_index < static_cast<int>(trajectory.size())) {
                  trajectory[ev.transition_index].q_value += bonus;
                  trajectory[ev.transition_index].advantage += bonus;
                  trajectory[ev.transition_index].reward_target += bonus;
                }
              }
            }
          }
          own_shaped_bonus = 0.0f;
        }

        transition.q_value = own_shaped_bonus;
        transition.advantage = own_shaped_bonus;
        transition.reward_target = own_shaped_bonus;
        transition.v_value = current_value; // Store the V(s) value
        transition.behavior_prob = behavior_prob;
        transition.importance_multiplier = 1.0f;
        transition.training_step_inserted = 0;
        transition.player_id = current_player;
        
        recorded_transitions_by_seat[current_player].fetch_add(1, std::memory_order_relaxed);

        trajectory.push_back(std::move(transition));
        if (current_player >= 0 && current_player < game.NumPlayers()) {
          last_transition_index[current_player] =
              static_cast<int>(current_transition_index);
        }
      }

      // Retro-credit VP to other players.
      for (int p = 0; p < game.NumPlayers(); ++p) {
        if (p == current_player || shaped_bonus_by_player[p] == 0.0f) {
          continue;
        }
        if (action == dune_imperium::kActionCombatPass) {
          // CombatPass triggered VP for another player: distribute positive
          // VP across that player's combat-causal transitions by delta weight.
          // Suppress negative VP (alliance theft) and positive VP with no
          // combat events (GAE handles it via terminal returns).
          if (shaped_bonus_by_player[p] > 0.0f &&
              !combat_credit[p].empty()) {
            int total_delta = 0;
            for (const auto& ev : combat_credit[p])
              total_delta += ev.strength_delta;
            if (total_delta > 0) {
              for (const auto& ev : combat_credit[p]) {
                float weight = static_cast<float>(ev.strength_delta) /
                               static_cast<float>(total_delta);
                float bonus = shaped_bonus_by_player[p] * weight;
                if (ev.transition_index >= 0 &&
                    ev.transition_index < static_cast<int>(trajectory.size())) {
                  trajectory[ev.transition_index].q_value += bonus;
                  trajectory[ev.transition_index].advantage += bonus;
                  trajectory[ev.transition_index].reward_target += bonus;
                }
              }
            }
          }
        } else {
          // Non-CombatPass VP: credit to most recent transition as before.
          int idx = last_transition_index[p];
          if (idx >= 0 && idx < static_cast<int>(trajectory.size())) {
            trajectory[idx].q_value += shaped_bonus_by_player[p];
            trajectory[idx].advantage += shaped_bonus_by_player[p];
            trajectory[idx].reward_target += shaped_bonus_by_player[p];
          }
        }
      }
    }
  }

  // Per-Player GAE Computation
  std::vector<double> returns = state->Returns();
  double gae_lambda = absl::GetFlag(FLAGS_gae_lambda);
  std::vector<float> last_val(game.NumPlayers(), 0.0f);
  std::vector<float> last_gae(game.NumPlayers(), 0.0f);
  std::vector<bool> seen_last_action(game.NumPlayers(), false);

  for (auto it = trajectory.rbegin(); it != trajectory.rend(); ++it) {
    if (it->player_id >= 0 && it->player_id < game.NumPlayers()) {
      int p = it->player_id;
      float r_i = it->reward_target; // holds shaped_bonus VP gain

      // Chronologically last action check (first step seen going backward)
      if (!seen_last_action[p]) {
        r_i += static_cast<float>(returns[p]);
        seen_last_action[p] = true;
      }

      // Normalize total reward to [-1.0, 1.0] scale
      r_i = r_i / 4.0f;
      r_i = std::clamp(r_i, -1.0f, 1.0f);

      float V_i = it->v_value;
      
      // δ_i = r_i + γ * V_{i+1} - V_i
      // Episodic finite horizon: γ = 1.0
      float delta_i = r_i + last_val[p] - V_i;

      // A_i = δ_i + γ * λ * A_{i+1}
      // Episodic finite horizon: γ = 1.0
      float A_i = delta_i + gae_lambda * last_gae[p];

      // Value learning wants the TD(lambda) return. The sparse sampled MMD
      // policy update wants an estimator of the full legal-action update. Use
      // advantage plus behavior-probability correction; using the raw return
      // here reinforces every sampled action in good states, even bad ones.
      it->reward_target = A_i + V_i;
      float importance = 1.0f / std::max(it->behavior_prob, 1e-6f);
      float importance_clip = static_cast<float>(absl::GetFlag(FLAGS_mmd_importance_clip));
      if (importance_clip > 0.0f) {
        importance = std::min(importance, importance_clip);
      }
      it->importance_multiplier = importance;
      it->advantage = A_i;
      it->q_value = A_i * importance;


      // Update trackers
      last_val[p] = V_i;
      last_gae[p] = A_i;
    }
  }

  return game_length;
}
#endif

int RandomSimulation(std::mt19937* rng, const Game& game, int64_t obs_size, std::vector<Transition>& trajectory) {
  std::unique_ptr<State> state = game.NewInitialState();

  bool provides_info_state_tensor =
      game.GetType().provides_information_state_tensor;
  bool provides_observations_tensor =
      game.GetType().provides_observation_tensor;

  int game_length = 0;
  while (!state->IsTerminal()) {
    ++game_length;
    if (game_length > 5000) {
      std::cerr << "Possible infinite loop detected! Game length: " << game_length
                << " Player: " << state->CurrentPlayer()
                << "\nState string:\n" << state->ToString() << std::endl;
      std::abort();
    }
    if (state->IsChanceNode()) {
      std::vector<std::pair<Action, double>> outcomes = state->ChanceOutcomes();
      Action action;
      if (game.GetType().chance_mode ==
          GameType::ChanceMode::kSampledStochastic) {
        action = outcomes.front().first;
      } else {
        action = SampleAction(outcomes, *rng).first;
      }
      state->ApplyAction(action);
    } else if (state->CurrentPlayer() == kSimultaneousPlayerId) {
      std::vector<Action> joint_action;
      for (int p = 0; p < game.NumPlayers(); p++) {
        std::vector<Action> actions = state->LegalActions(p);
        Action action = 0;
        if (!actions.empty()) {
          std::uniform_int_distribution<int> dis(0, actions.size() - 1);
          action = actions[dis(*rng)];
        }
        joint_action.push_back(action);
      }
      state->ApplyActions(joint_action);
    } else {
      Player current_player = state->CurrentPlayer();
      std::vector<Action> actions = state->LegalActions();
      if (actions.empty()) {
        std::cerr << "Spiel Fatal Error: Non-terminal state has empty LegalActions(). Player: " 
                  << state->CurrentPlayer() << "\nState string:\n" << state->ToString() << std::endl;
        std::abort();
      }
      std::uniform_int_distribution<int> dis(0, actions.size() - 1);
      Action action = actions[dis(*rng)];

      // 1. Allocate and populate observation vector via span
      std::vector<float> obs(obs_size, 0.0f);
      if (provides_info_state_tensor && current_player >= 0) {
        state->InformationStateTensor(current_player, absl::MakeSpan(obs));
      } else if (provides_observations_tensor && current_player >= 0) {
        state->ObservationTensor(current_player, absl::MakeSpan(obs));
      }

      int64_t action_size = game.NumDistinctActions();

      // Apply action
      state->ApplyAction(action);

      // Push to trajectory
      Transition transition;
      transition.spatial_tensor = std::move(obs);
      transition.legal_actions = std::move(actions); // Store sparse legal actions list directly
      transition.action_taken = action;
      transition.prev_logits = std::vector<float>(action_size, 0.0f);
      transition.q_value = 0.0f;
      transition.advantage = 0.0f;
      transition.reward_target = 0.0f;
      transition.v_value = 0.0f; // Store V(s) = 0.0f explicitly
      transition.behavior_prob = 1.0f / static_cast<float>(actions.size());
      transition.importance_multiplier = 1.0f;
      transition.training_step_inserted = 0;
      transition.player_id = current_player;
      trajectory.push_back(std::move(transition));
    }
  }

  // Populate rewards at game terminal state
  std::vector<double> returns = state->Returns();
  for (auto& transition : trajectory) {
    if (transition.player_id >= 0 && transition.player_id < static_cast<int>(returns.size())) {
      float reward = static_cast<float>(returns[transition.player_id]);
      transition.reward_target = reward / 4.0f; // Scale consistently
      transition.q_value = reward / 4.0f;        // Scale consistently
      transition.advantage = reward / 4.0f;      // Scale consistently
    }
  }

  return game_length;
}

void ThreadWorker(int thread_id, const Game* game, std::atomic<int>& games_completed,
                  int total_games, std::atomic<int>& total_moves,
                  GlobalReplayBuffer* global_buffer, int64_t obs_size,
                  std::atomic<bool>* stop_collection
#ifdef OPEN_SPIEL_BUILD_WITH_LIBTORCH
                  , std::shared_ptr<BatchedEvaluator> evaluator
                  , std::atomic<float>* reward_lambda
                  , std::atomic<int>* training_steps
                  , const std::vector<std::shared_ptr<BatchedEvaluator>>& opponent_evaluators
                  , const std::vector<double>& opponent_mix
                  , const std::string& learner_seat_mode
#endif
) {
  std::mt19937 rng(std::random_device{}() + thread_id);
  int moves = 0;
  while (true) {
    if (stop_collection != nullptr && stop_collection->load(std::memory_order_relaxed)) {
      break;
    }
    int prev_completed = games_completed.fetch_add(1);
    if (prev_completed >= total_games) {
      games_completed.fetch_sub(1);
      break;
    }
    std::vector<Transition> local_trajectory;
#ifdef OPEN_SPIEL_BUILD_WITH_LIBTORCH
    if (evaluator != nullptr) {
      moves += TorchSimulation(&rng, *game, evaluator, obs_size, local_trajectory, reward_lambda,
                               opponent_evaluators, opponent_mix, learner_seat_mode, prev_completed);
    } else {
      moves += RandomSimulation(&rng, *game, obs_size, local_trajectory);
    }
#else
    moves += RandomSimulation(&rng, *game, obs_size, local_trajectory);
#endif
    int64_t inserted_step = 0;
#ifdef OPEN_SPIEL_BUILD_WITH_LIBTORCH
    if (training_steps != nullptr) {
      inserted_step = training_steps->load(std::memory_order_relaxed);
    }
#endif
    global_buffer->PushTrajectory(local_trajectory, inserted_step);
  }
  total_moves.fetch_add(moves);
}

} // namespace
} // namespace open_spiel

int main(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);

#ifdef OPEN_SPIEL_BUILD_WITH_LIBTORCH
  // Clamp internal LibTorch CPU threads to 1 to prevent severe thread pool thrashing
  at::set_num_threads(1);
  at::set_num_interop_threads(1);
#endif

  std::string game_name = absl::GetFlag(FLAGS_game);
  int total_games = absl::GetFlag(FLAGS_games);
  int num_threads = absl::GetFlag(FLAGS_threads);

  std::cout << absl::StrFormat("Initializing Multi-threaded Benchmark...\n");
  std::cout << absl::StrFormat("Game: %s\n", game_name);
  std::cout << absl::StrFormat("Total games to simulate: %d\n", total_games);
  std::cout << absl::StrFormat("Number of threads: %d\n", num_threads);

#ifdef OPEN_SPIEL_BUILD_WITH_LIBTORCH
  std::cout << "LibTorch Inference Mode: ENABLED\n";
#else
  std::cout << "LibTorch Inference Mode: DISABLED (Pure C++ Random Baseline)\n";
#endif

  auto game = open_spiel::LoadGame(game_name);

  // General observation shape retrieval
  int64_t obs_size = 0;
  bool provides_info_state_tensor =
      game->GetType().provides_information_state_tensor;
  bool provides_observations_tensor =
      game->GetType().provides_observation_tensor;
  if (provides_info_state_tensor) {
    obs_size = game->InformationStateTensorSize();
  } else if (provides_observations_tensor) {
    obs_size = game->ObservationTensorSize();
  }

#ifdef OPEN_SPIEL_BUILD_WITH_LIBTORCH
  int64_t action_size = game->NumDistinctActions();
  
  torch::Device train_device(torch::kCPU);
  if (torch::cuda::is_available()) {
    train_device = torch::Device(torch::kCUDA);
    std::cout << "CUDA is available! Optimization Worker will execute on GPU.\n";
    
    // Enable TF32 globally
    at::globalContext().setAllowTF32CuBLAS(true);
    at::globalContext().setAllowTF32CuDNN(true);
    
    // CRITICAL: Force Autocast to use BF16 natively on Ada Lovelace
    at::autocast::set_autocast_gpu_dtype(at::ScalarType::BFloat16);
  }

  std::shared_ptr<open_spiel::SharedDunePolicyValueNetImpl> training_model = nullptr;
  std::shared_ptr<open_spiel::SharedDunePolicyValueNetImpl> inference_model = nullptr;
  std::shared_ptr<open_spiel::SharedDunePolicyValueNetImpl> reference_model = nullptr;
  std::unique_ptr<torch::optim::AdamW> optimizer = nullptr;
  std::string model_path = absl::GetFlag(FLAGS_model_checkpoint);
  std::string optim_path = absl::GetFlag(FLAGS_optim_checkpoint);

  std::shared_mutex sync_mutex;
  std::thread optimization_thread;
  std::atomic<bool> stop_training(false);
  std::atomic<int> training_steps(absl::GetFlag(FLAGS_start_step));
  std::atomic<float> reward_lambda{1.0f};
  std::shared_ptr<open_spiel::BatchedEvaluator> evaluator = nullptr;
  std::vector<std::shared_ptr<open_spiel::BatchedEvaluator>> opponent_evaluators;
  std::vector<double> opponent_mix_normalized;
  std::string learner_seat_mode = absl::GetFlag(FLAGS_learner_seat_mode);

  if (obs_size > 0) {
    int hidden_dim = absl::GetFlag(FLAGS_hidden_dim);
    int num_blocks = absl::GetFlag(FLAGS_num_blocks);
    int eval_batch_size = absl::GetFlag(FLAGS_eval_batch_size);
    int eval_timeout_ms = absl::GetFlag(FLAGS_eval_timeout_ms);

    training_model = std::make_shared<open_spiel::SharedDunePolicyValueNetImpl>(obs_size, hidden_dim, action_size, num_blocks);
    inference_model = std::make_shared<open_spiel::SharedDunePolicyValueNetImpl>(obs_size, hidden_dim, action_size, num_blocks);
    std::string reference_mode = absl::GetFlag(FLAGS_mmd_reference_mode);
    std::string policy_update_mode = absl::GetFlag(FLAGS_policy_update_mode);
    if (!open_spiel::IsPolicyUpdateModeValid(policy_update_mode)) {
      std::cerr << "Invalid --policy_update_mode='" << policy_update_mode
                << "'. Expected one of: mmd, ppo.\n";
      return 1;
    }
    if (reference_mode != "uniform" && reference_mode != "periodic" &&
        reference_mode != "ema") {
      std::cerr << "Invalid --mmd_reference_mode='" << reference_mode
                << "'. Expected one of: uniform, periodic, ema.\n";
      return 1;
    }
    if (policy_update_mode == "mmd" &&
        open_spiel::IsReferenceModeEnabled(reference_mode)) {
      reference_model = std::make_shared<open_spiel::SharedDunePolicyValueNetImpl>(
          obs_size, hidden_dim, action_size, num_blocks);
      reference_model->to(train_device);
      reference_model->eval();
    }
    
    inference_model->to(train_device);

    training_model->train(); // training model stays in train mode permanently
    inference_model->eval(); // inference model stays in eval mode permanently
    
    std::cout << absl::StrFormat("Initialized Double-Buffered SharedDunePolicyValueNet (Input Dim: %d, Hidden Dim: %d, Action Dim: %d, Blocks: %d)\n", 
                                 obs_size, hidden_dim, action_size, num_blocks);
    
    // Load existing checkpoint and set up the training model and optimizer correctly on the target hardware device
    open_spiel::LoadCheckpoint(training_model, optimizer, model_path, optim_path, train_device);
    // Initial weight synchronization
    open_spiel::SyncModels(training_model, inference_model, &sync_mutex);
    if (reference_model != nullptr) {
      open_spiel::CopyModelWeights(training_model, reference_model);
      std::cout << absl::StrFormat(
          "Initialized MMD reference policy model (mode=%s, interval=%d, ema_decay=%.6f)\n",
          reference_mode, absl::GetFlag(FLAGS_mmd_reference_interval),
          absl::GetFlag(FLAGS_mmd_reference_ema_decay));
    }

    // Step-0 Logit Health Check: this raw all-action probe is informational.
    // Actual policy logits are centered over legal actions before any cap.
    float logit_cap = static_cast<float>(absl::GetFlag(FLAGS_logit_cap));
    if (logit_cap > 0.0f) {
      torch::NoGradGuard no_grad;
      torch::Tensor probe_input = torch::zeros({1, obs_size}, torch::TensorOptions().device(train_device));
      auto probe_out = training_model->forward(probe_input);
      float max_abs_logit = probe_out.logits.abs().max().item<float>();
      std::cout << absl::StrFormat("[Step-0 Check] Max |raw logit|: %.2f (cap: %.1f)\n", max_abs_logit, logit_cap);
      if (max_abs_logit > logit_cap) {
        std::cout << "[Step-0 Check] Raw all-action logits exceed the cap. Continuing because legal-centered diagnostics are now authoritative.\n";
      }
    }

    bool evaluator_device_synchronize = absl::GetFlag(FLAGS_evaluator_device_synchronize);

    // Instantiate the BatchedEvaluator with target batch size and timeout. It returns
    // raw logits; self-play applies legal-action centering plus the configured cap.
    evaluator = std::make_shared<open_spiel::BatchedEvaluator>(
        inference_model, eval_batch_size, eval_timeout_ms, train_device, &sync_mutex,
        0.0f, evaluator_device_synchronize);
    std::cout << absl::StrFormat(
        "BatchedEvaluator initialized with target batch size: %d, timeout: %d ms, legal-centered logit cap: %.1f, device synchronize: %s\n",
        eval_batch_size, eval_timeout_ms,
        logit_cap,
        evaluator_device_synchronize ? "true" : "false");
    std::cout << absl::StrFormat(
        "Policy update mode: %s%s\n",
        absl::GetFlag(FLAGS_policy_update_mode),
        absl::GetFlag(FLAGS_policy_update_mode) == "ppo"
            ? absl::StrFormat(" (clip_epsilon=%.3f)",
                              absl::GetFlag(FLAGS_ppo_clip_epsilon))
            : "");

    // Validate learner_seat_mode
    SPIEL_CHECK_TRUE(learner_seat_mode == "self_play" || learner_seat_mode == "rotate");
    
    std::string opponent_checkpoints_str = absl::GetFlag(FLAGS_opponent_checkpoints);
    if (learner_seat_mode == "rotate") {
      SPIEL_CHECK_TRUE(!opponent_checkpoints_str.empty());
    }

    if (!opponent_checkpoints_str.empty()) {
      std::vector<std::string> opponent_paths = open_spiel::SplitCommaSeparated(opponent_checkpoints_str);
      size_t num_opponents = opponent_paths.size();
      
      std::string opponent_hidden_dims_str = absl::GetFlag(FLAGS_opponent_hidden_dims);
      std::string opponent_num_blocks_str = absl::GetFlag(FLAGS_opponent_num_blocks);
      
      std::vector<std::string> h_dims_str = open_spiel::SplitCommaSeparated(opponent_hidden_dims_str);
      std::vector<std::string> n_blocks_str = open_spiel::SplitCommaSeparated(opponent_num_blocks_str);
      
      std::vector<int> opponent_hidden_dims(num_opponents);
      std::vector<int> opponent_num_blocks(num_opponents);
      
      // Resolve hidden dims
      if (h_dims_str.empty()) {
        for (size_t i = 0; i < num_opponents; ++i) {
          opponent_hidden_dims[i] = hidden_dim;
        }
      } else if (h_dims_str.size() == 1) {
        int single_val = std::stoi(h_dims_str[0]);
        for (size_t i = 0; i < num_opponents; ++i) {
          opponent_hidden_dims[i] = single_val;
        }
      } else {
        SPIEL_CHECK_EQ(h_dims_str.size(), num_opponents);
        for (size_t i = 0; i < num_opponents; ++i) {
          opponent_hidden_dims[i] = std::stoi(h_dims_str[i]);
        }
      }
      
      // Resolve num blocks
      if (n_blocks_str.empty()) {
        for (size_t i = 0; i < num_opponents; ++i) {
          opponent_num_blocks[i] = num_blocks;
        }
      } else if (n_blocks_str.size() == 1) {
        int single_val = std::stoi(n_blocks_str[0]);
        for (size_t i = 0; i < num_opponents; ++i) {
          opponent_num_blocks[i] = single_val;
        }
      } else {
        SPIEL_CHECK_EQ(n_blocks_str.size(), num_opponents);
        for (size_t i = 0; i < num_opponents; ++i) {
          opponent_num_blocks[i] = std::stoi(n_blocks_str[i]);
        }
      }
      
      // Parse and validate opponent mix
      std::string opponent_mix_str = absl::GetFlag(FLAGS_opponent_mix);
      std::vector<std::string> mix_str_vals = open_spiel::SplitCommaSeparated(opponent_mix_str);
      std::vector<double> raw_weights;
      
      if (mix_str_vals.empty()) {
        raw_weights.assign(num_opponents, 1.0);
      } else {
        SPIEL_CHECK_EQ(mix_str_vals.size(), num_opponents);
        for (const auto& w_str : mix_str_vals) {
          double w = std::stod(w_str);
          SPIEL_CHECK_TRUE(std::isfinite(w));
          SPIEL_CHECK_GE(w, 0.0);
          raw_weights.push_back(w);
        }
      }
      
      double sum_weights = 0.0;
      for (double w : raw_weights) sum_weights += w;
      SPIEL_CHECK_GT(sum_weights, 0.0);
      
      opponent_mix_normalized.reserve(num_opponents);
      for (double w : raw_weights) {
        opponent_mix_normalized.push_back(w / sum_weights);
      }
      
      // Load opponent models and wrap in BatchedEvaluators
      opponent_evaluators.reserve(num_opponents);
      for (size_t i = 0; i < num_opponents; ++i) {
        std::cout << absl::StrFormat("Loading opponent %d: %s (Architecture: %dx%d, Weight: %.4f)\n",
                                     i, opponent_paths[i], opponent_hidden_dims[i], opponent_num_blocks[i],
                                     opponent_mix_normalized[i]);
        
        auto opp_model = std::make_shared<open_spiel::SharedDunePolicyValueNetImpl>(
            obs_size, opponent_hidden_dims[i], action_size, opponent_num_blocks[i]);
        opp_model->eval();
        
        try {
          torch::serialize::InputArchive archive;
          archive.load_from(opponent_paths[i], train_device);
          opp_model->load(archive);
        } catch (const c10::Error& e) {
          std::cerr << "Failed to load opponent checkpoint: " << opponent_paths[i] << "\n" << e.msg() << "\n";
          std::exit(1);
        }
        
        opp_model->to(train_device);
        
        opponent_evaluators.push_back(std::make_shared<open_spiel::BatchedEvaluator>(
            opp_model, eval_batch_size, eval_timeout_ms, train_device, &sync_mutex,
            0.0f, evaluator_device_synchronize));
      }
    }
  }
#endif

  // Construct global thread-safe replay buffer
  open_spiel::GlobalReplayBuffer global_buffer(absl::GetFlag(FLAGS_buffer_capacity));

  std::atomic<int> games_completed(0);
  std::atomic<int> total_moves(0);
  std::atomic<bool> stop_collection(false);

#ifdef OPEN_SPIEL_BUILD_WITH_LIBTORCH
  // Start background optimization thread if async_mode is enabled
  bool async_mode = absl::GetFlag(FLAGS_async_mode);
  if (async_mode && training_model != nullptr) {
    int min_train_size = absl::GetFlag(FLAGS_min_train_size);
    int train_batch_size = absl::GetFlag(FLAGS_train_batch_size);
    int sync_interval = absl::GetFlag(FLAGS_sync_interval);
    int train_ratio = absl::GetFlag(FLAGS_train_ratio);
    int max_train_steps = absl::GetFlag(FLAGS_max_train_steps);
    bool train_amp = absl::GetFlag(FLAGS_train_amp);
    int loss_read_interval = absl::GetFlag(FLAGS_loss_read_interval);
    int diagnostic_interval = absl::GetFlag(FLAGS_train_diagnostic_interval);

    std::cout << absl::StrFormat(
        "Starting Asynchronous Optimization Worker (Min Train Size: %d, Batch Size: %d, Sync Interval: %d, Train Ratio: %d, Max Local Steps: %d, Train AMP: %s, Loss Read Interval: %d, Diagnostic Interval: %d)...\n",
        min_train_size, train_batch_size, sync_interval, train_ratio, max_train_steps,
        train_amp ? "true" : "false", loss_read_interval, diagnostic_interval);

    optimization_thread = std::thread(
        open_spiel::OptimizationWorker,
        training_model, inference_model, reference_model, optimizer.get(), &global_buffer,
        obs_size, action_size, &sync_mutex,
        std::ref(stop_training), std::ref(stop_collection), std::ref(training_steps),
        min_train_size, train_batch_size, sync_interval, max_train_steps,
        model_path, optim_path, train_device,
        &reward_lambda
    );
  }
#endif

  auto start_time = std::chrono::high_resolution_clock::now();

  std::vector<std::thread> threads;
  threads.reserve(num_threads);
  for (int i = 0; i < num_threads; ++i) {
    threads.emplace_back(open_spiel::ThreadWorker, i, game.get(),
                         std::ref(games_completed), total_games, std::ref(total_moves),
                         &global_buffer, obs_size, &stop_collection
#ifdef OPEN_SPIEL_BUILD_WITH_LIBTORCH
                         , evaluator, &reward_lambda, &training_steps
                         , opponent_evaluators, opponent_mix_normalized, learner_seat_mode
#endif
    );
  }

  for (auto& t : threads) {
    t.join();
  }

#ifdef OPEN_SPIEL_BUILD_WITH_LIBTORCH
  // Stop background training and join thread
  if (async_mode && optimization_thread.joinable()) {
    stop_training.store(true);
    optimization_thread.join();
  }
#endif

  auto end_time = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = end_time - start_time;

  double seconds = elapsed.count();
  int moves = total_moves.load();
  int completed = games_completed.load();

#ifdef OPEN_SPIEL_BUILD_WITH_LIBTORCH
  if (evaluator != nullptr) {
    auto eval_stats = evaluator->GetStats();
    std::cout << absl::StrFormat(
        "\nEvaluator Stats: Requests=%llu | Batches=%llu | Avg Batch=%.2f | Max Batch=%llu\n",
        static_cast<unsigned long long>(eval_stats.requests),
        static_cast<unsigned long long>(eval_stats.batches),
        eval_stats.avg_batch_size,
        static_cast<unsigned long long>(eval_stats.max_batch_size));
  }

  if (training_model != nullptr && global_buffer.Size() > 0 && optimizer != nullptr) {
    if (async_mode) {
      std::cout << absl::StrFormat("\nAsynchronous training completed. Total background SGD steps executed: %d\n", training_steps.load());
      // Save checkpoint of training_model at program termination
      if (absl::GetFlag(FLAGS_save_final_checkpoint)) {
        open_spiel::SaveCheckpoint(training_model, *optimizer, model_path, optim_path);
      } else {
        std::cout << "Final checkpoint save skipped (--save_final_checkpoint=false).\n";
      }
    } else {
      std::cout << "\nStarting optimization validation phase (Synchronous Mode)...\n";
      // Sample a single training batch of size 32
      size_t batch_size = 32;
      std::vector<std::shared_ptr<open_spiel::Transition>> batch = global_buffer.SampleBatch(batch_size);
      std::cout << absl::StrFormat("Sampled %zu transitions from replay buffer.\n", batch.size());

      // Execute backprop step using the startup-initialized optimizer
      open_spiel::PersistentTrainBuffers sync_buffers;
      sync_buffers.Initialize(batch_size, obs_size, action_size, train_device.is_cuda());
      open_spiel::TrainStep(training_model, reference_model, *optimizer, batch,
                            sync_buffers, obs_size, action_size, train_device,
                            training_steps.load(), true);
      std::cout << "Optimization Step successfully completed (Forward + Loss + Backward + Weight Update)!\n";
      
      // Sync training weights to inference weights for completeness
      open_spiel::SyncModels(training_model, inference_model, &sync_mutex);

      // Save updated checkpoint
      if (absl::GetFlag(FLAGS_save_final_checkpoint)) {
        open_spiel::SaveCheckpoint(training_model, *optimizer, model_path, optim_path);
      } else {
        std::cout << "Final checkpoint save skipped (--save_final_checkpoint=false).\n";
      }
    }
  }
#endif

  std::cout << absl::StrFormat("\n=== Benchmark Completed ===\n");
  std::cout << absl::StrFormat("Elapsed Time: %.3f seconds\n", seconds);
  std::cout << absl::StrFormat("Games Completed: %d\n", completed);
  std::cout << absl::StrFormat("Total Moves Executed: %d\n", moves);
  std::cout << absl::StrFormat("Total Replay Transitions Inserted: %llu\n",
                               static_cast<unsigned long long>(global_buffer.GetTotalInserted()));
  std::cout << absl::StrFormat("Final Global Buffer Size: %zu transitions\n", global_buffer.Size());
  std::cout << absl::StrFormat("Games Per Second (GPS): %.2f\n", completed / seconds);
  std::cout << absl::StrFormat("Moves Per Second (MPS): %.2f\n", moves / seconds);

#ifdef OPEN_SPIEL_BUILD_WITH_LIBTORCH
  if (learner_seat_mode == "rotate") {
    std::cout << absl::StrFormat("\n=== Frozen-Population Diagnostics ===\n");
    std::cout << absl::StrFormat("Learner Seat Games Played: P0:%d | P1:%d | P2:%d | P3:%d\n",
                                 open_spiel::learner_seat_games_count[0].load(),
                                 open_spiel::learner_seat_games_count[1].load(),
                                 open_spiel::learner_seat_games_count[2].load(),
                                 open_spiel::learner_seat_games_count[3].load());
    std::cout << absl::StrFormat("Learner Transitions Recorded: P0:%llu | P1:%llu | P2:%llu | P3:%llu (Total: %llu)\n",
                                 static_cast<unsigned long long>(open_spiel::recorded_transitions_by_seat[0].load()),
                                 static_cast<unsigned long long>(open_spiel::recorded_transitions_by_seat[1].load()),
                                 static_cast<unsigned long long>(open_spiel::recorded_transitions_by_seat[2].load()),
                                 static_cast<unsigned long long>(open_spiel::recorded_transitions_by_seat[3].load()),
                                 static_cast<unsigned long long>(open_spiel::recorded_transitions_by_seat[0].load() +
                                                                  open_spiel::recorded_transitions_by_seat[1].load() +
                                                                  open_spiel::recorded_transitions_by_seat[2].load() +
                                                                  open_spiel::recorded_transitions_by_seat[3].load()));
    std::cout << absl::StrFormat("Opponent Decisions Made: %llu\n",
                                 static_cast<unsigned long long>(open_spiel::opponent_decisions_count.load()));
  }
#endif

  return 0;
}
