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
ABSL_FLAG(int, buffer_capacity, 100000, "Maximum capacity of the global replay buffer.");
ABSL_FLAG(int, train_ratio, 64, "The number of NEW moves the CPU must generate before the GPU is allowed to pull 1 training batch. Set <= 0 to disable throttling.");
ABSL_FLAG(int, decay_horizon, 12000000, "Number of training steps over which to linearly decay the shaped reward lambda.");
ABSL_FLAG(double, shaped_reward_weight, 0.2, "Weight multiplier for each VP gained in intermediate shaped rewards.");
ABSL_FLAG(double, temperature, 1.0, "Softmax temperature for action selection (1.0 = standard, >1.0 = explore, 0.0 = greedy).");
ABSL_FLAG(double, learning_rate, 1e-4, "Learning rate for the Adam optimizer.");
ABSL_FLAG(double, mmd_eta, 0.2, "MMD entropy parameter eta.");
ABSL_FLAG(double, mmd_alpha, 0.1, "MMD entropy parameter alpha.");
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

namespace open_spiel {

struct Transition {
  std::vector<float> spatial_tensor;     // Flattened observation tensor
  std::vector<Action> legal_actions;     // Sparse list of legal action IDs (saves 99% RAM)
  int64_t action_taken;                  // Action ID chosen
  std::vector<float> prev_logits;        // Raw policy logits (z_k)
  float q_value;                         // Estimated action-value Q(s, a)
  float reward_target;                   // Mapped zero-sum reward
  float v_value;                         // Value Network estimate V(s)
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

  void PushTrajectory(std::vector<Transition>& local_trajectory) {
    std::lock_guard<std::mutex> lock(buffer_mutex_);
    size_t num_inserted = local_trajectory.size();
    for (auto& transition : local_trajectory) {
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

  // Initialize AdamW Optimizer AFTER model weights have been moved to the target hardware device
  optimizer = std::make_unique<torch::optim::AdamW>(model->parameters(), torch::optim::AdamWOptions(learning_rate).weight_decay(weight_decay));

  if (std::filesystem::exists(optim_path)) {
    try {
      torch::load(*optimizer, optim_path, device);
      std::cout << "Successfully loaded optimizer state from " << optim_path << "\n";

      // CRITICAL FIX: PyTorch's load() overwrites the learning rate with the one saved in the file.
      // We must explicitly override it to ensure the Abseil flag is respected!
      for (auto& param_group : optimizer->param_groups()) {
        if (param_group.has_options()) {
          static_cast<torch::optim::AdamWOptions&>(param_group.options()).lr(learning_rate);
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
  torch::Tensor states, masks, rewards, prev_logits, q_values, actions_taken;
  float *p_states, *p_masks, *p_rewards, *p_prev, *p_q;
  int64_t *p_actions;

  // Persistent device-static tensors for zero-allocation H2D transfers
  torch::Tensor d_states, d_masks, d_rewards, d_prev_logits, d_q_values, d_actions_taken;

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
    actions_taken = torch::empty({max_batch_size, 1}, long_opts);

    p_states = states.data_ptr<float>();
    p_masks = masks.data_ptr<float>();
    p_rewards = rewards.data_ptr<float>();
    p_prev = prev_logits.data_ptr<float>();
    p_q = q_values.data_ptr<float>();
    p_actions = actions_taken.data_ptr<int64_t>();

    if (is_cuda) {
      auto d_f = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
      auto d_l = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA);
      
      d_states = torch::empty({max_batch_size, obs_size}, d_f);
      d_masks = torch::empty({max_batch_size, action_dim}, d_f);
      d_rewards = torch::empty({max_batch_size, 1}, d_f);
      d_prev_logits = torch::empty({max_batch_size, action_dim}, d_f);
      d_q_values = torch::empty({max_batch_size, 1}, d_f);
      d_actions_taken = torch::empty({max_batch_size, 1}, d_l);
    }
  }
};

std::pair<float, float> TrainStep(std::shared_ptr<SharedDunePolicyValueNetImpl> model, 
                                  torch::optim::Optimizer& optimizer, 
                                  const std::vector<std::shared_ptr<Transition>>& batch,
                                  PersistentTrainBuffers& buffers,
                                  int64_t obs_size,
                                  int64_t action_dim,
                                  torch::Device device);

void SyncModels(std::shared_ptr<SharedDunePolicyValueNetImpl> training_model, 
                std::shared_ptr<SharedDunePolicyValueNetImpl> inference_model,
                std::shared_mutex* sync_mutex) {
  std::unique_lock<std::shared_mutex> lock(*sync_mutex);
  torch::NoGradGuard no_grad;

  auto train_params = training_model->parameters();
  auto infer_params = inference_model->parameters();
  auto train_buffers = training_model->buffers();
  auto infer_buffers = inference_model->buffers();

  for (size_t i = 0; i < train_params.size(); ++i) {
    infer_params[i].copy_(train_params[i].to(infer_params[i].device()));
  }
  for (size_t i = 0; i < train_buffers.size(); ++i) {
    infer_buffers[i].copy_(train_buffers[i].to(infer_buffers[i].device()));
  }
}

void OptimizationWorker(
    std::shared_ptr<SharedDunePolicyValueNetImpl> training_model,
    std::shared_ptr<SharedDunePolicyValueNetImpl> inference_model,
    torch::optim::AdamW* optimizer,
    GlobalReplayBuffer* global_buffer,
    int64_t obs_size,
    int64_t action_size,
    std::shared_mutex* sync_mutex,
    std::atomic<bool>& stop_training,
    std::atomic<int>& training_steps,
    int min_train_size,
    int train_batch_size,
    int sync_interval,
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

    // 2. Training Loop
    PersistentTrainBuffers train_buffers;
    train_buffers.Initialize(train_batch_size, obs_size, action_size, device.is_cuda());

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

      // Lock-free optimization step on training_model
      auto [v_loss, p_loss] = TrainStep(training_model, *optimizer, batch, train_buffers, obs_size, action_size, device);

      int step = training_steps.fetch_add(1) + 1;

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
                                     step, global_buffer->Size(), current_lambda, v_loss, p_loss) << std::flush;
      }

      // Periodic checkpoint saving (runs in background, no sync_mutex needed)
      if (step % 100000 == 0) {
        // ROTATING CHECKPOINTS: Enforce saving to a new numbered file every time
        std::string run_prefix = absl::GetFlag(FLAGS_run_prefix);
        std::string step_model_path = absl::StrCat(run_prefix, "_model_step_", step, ".pt");
        std::string step_optim_path = absl::StrCat(run_prefix, "_optimizer_step_", step, ".pt");
        SaveCheckpoint(training_model, *optimizer, step_model_path, step_optim_path);
      }
    }
    
    // Final synchronization at shutdown to ensure inference_model has latest weights
    SyncModels(training_model, inference_model, sync_mutex);
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
                                  torch::optim::Optimizer& optimizer, 
                                  const std::vector<std::shared_ptr<Transition>>& batch,
                                  PersistentTrainBuffers& buffers,
                                  int64_t obs_size,
                                  int64_t action_dim,
                                  torch::Device device) {
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
    buffers.p_actions[i] = t->action_taken;
  }

  // Zero-Allocation H2D Transfer: copy from pinned host into persistent device memory
  if (buffers.is_cuda_) {
    buffers.d_states.slice(0, 0, batch_size).copy_(buffers.states.slice(0, 0, batch_size), /*non_blocking=*/true);
    buffers.d_masks.slice(0, 0, batch_size).copy_(buffers.masks.slice(0, 0, batch_size), /*non_blocking=*/true);
    buffers.d_rewards.slice(0, 0, batch_size).copy_(buffers.rewards.slice(0, 0, batch_size), /*non_blocking=*/true);
    buffers.d_prev_logits.slice(0, 0, batch_size).copy_(buffers.prev_logits.slice(0, 0, batch_size), /*non_blocking=*/true);
    buffers.d_q_values.slice(0, 0, batch_size).copy_(buffers.q_values.slice(0, 0, batch_size), /*non_blocking=*/true);
    buffers.d_actions_taken.slice(0, 0, batch_size).copy_(buffers.actions_taken.slice(0, 0, batch_size), /*non_blocking=*/true);
  }

  // Use persistent device tensors (zero-alloc) on CUDA, or sliced pinned tensors on CPU
  torch::Tensor states = buffers.is_cuda_ ? buffers.d_states.slice(0, 0, batch_size) : buffers.states.slice(0, 0, batch_size);
  torch::Tensor masks = buffers.is_cuda_ ? buffers.d_masks.slice(0, 0, batch_size) : buffers.masks.slice(0, 0, batch_size);
  torch::Tensor rewards = buffers.is_cuda_ ? buffers.d_rewards.slice(0, 0, batch_size) : buffers.rewards.slice(0, 0, batch_size);
  torch::Tensor prev_logits = buffers.is_cuda_ ? buffers.d_prev_logits.slice(0, 0, batch_size) : buffers.prev_logits.slice(0, 0, batch_size);
  torch::Tensor q_values = buffers.is_cuda_ ? buffers.d_q_values.slice(0, 0, batch_size) : buffers.q_values.slice(0, 0, batch_size);
  torch::Tensor actions_taken = buffers.is_cuda_ ? buffers.d_actions_taken.slice(0, 0, batch_size) : buffers.actions_taken.slice(0, 0, batch_size);

  optimizer.zero_grad();
  torch::Tensor pred_logits, pred_values;
  {
    auto outputs = model->forward(states);
    pred_logits = outputs.logits;
    pred_values = outputs.values;
  }

  // LOGIT SOFT-CAP: Prevent unbounded logit growth using smooth tanh saturation
  float logit_cap = static_cast<float>(absl::GetFlag(FLAGS_logit_cap));
  if (logit_cap > 0.0f) {
    pred_logits = logit_cap * torch::tanh(pred_logits / logit_cap);
  }

  torch::Tensor masked_logits = pred_logits.masked_fill(masks == 0.0f, -1e9f);
  torch::Tensor log_probs = torch::log_softmax(masked_logits, -1);
  torch::Tensor value_loss = torch::nn::functional::mse_loss(pred_values, rewards);


  float eta = static_cast<float>(absl::GetFlag(FLAGS_mmd_eta));
  float alpha = static_cast<float>(absl::GetFlag(FLAGS_mmd_alpha));

  torch::Tensor q_vector = torch::zeros({(int64_t)batch_size, action_dim}, torch::TensorOptions().device(device));
  q_vector.scatter_(1, actions_taken, q_values);

  torch::Tensor target_logits = (prev_logits + eta * q_vector) / (1.0f + eta * alpha);
  torch::Tensor masked_target_logits = target_logits.masked_fill(masks == 0.0f, -1e9f);
  torch::Tensor target_probs = torch::softmax(masked_target_logits, -1).detach();

  torch::Tensor policy_loss = torch::nn::functional::kl_div(
      log_probs, target_probs, torch::nn::functional::KLDivFuncOptions().reduction(torch::kBatchMean));

  // ENTROPY BONUS: Maximize entropy over legal actions to prevent policy collapse
  float entropy_coef = static_cast<float>(absl::GetFlag(FLAGS_entropy_coef));
  torch::Tensor probs = torch::softmax(masked_logits, -1);
  // Entropy = -sum(p * log(p)) over legal actions; masked slots contribute ~0
  torch::Tensor entropy_per_sample = -(probs * log_probs).sum(-1); // [batch]
  torch::Tensor mean_entropy = entropy_per_sample.mean();

  torch::Tensor total_loss = policy_loss + value_loss - entropy_coef * mean_entropy;

  float p_loss_val = policy_loss.item<float>();
  float v_loss_val = value_loss.item<float>();

  // Diagnostic for eta scaling and logit health
  static int log_counter = 0;
  if (++log_counter % 1000 == 0) {
      float mean_abs_q = q_values.abs().mean().item<float>();
      float max_abs_logit = pred_logits.abs().max().item<float>();
      // Top-1 softmax confidence (want < ~0.9, not ~0.97)
      float mean_top1_prob = std::get<0>(probs.max(-1)).mean().item<float>();
      float mean_ent = mean_entropy.item<float>();
      std::cout << "\n[Diagnostic] Mean|Q|: " << mean_abs_q 
                << " | Eta: " << eta
                << " | Max|Logit|: " << max_abs_logit
                << " | MeanTop1: " << mean_top1_prob
                << " | Entropy: " << mean_ent
                << " | PolicyKL: " << p_loss_val << "\n";
  }

  // 1. FATAL SAFETY GUARD & TENSOR AUTOPSY
  if (std::isnan(p_loss_val) || std::isnan(v_loss_val)) {
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
    
    std::cerr << "\nMODEL OUTPUTS (Forward Pass):\n";
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

int TorchSimulation(std::mt19937* rng, const Game& game, std::shared_ptr<BatchedEvaluator> evaluator, int64_t obs_size, std::vector<Transition>& trajectory, std::atomic<float>* reward_lambda) {
  std::unique_ptr<State> state = game.NewInitialState();

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
  if (dune_state != nullptr) {
    for (int p = 0; p < game.NumPlayers(); ++p) {
      current_vps[p] = dune_state->GetPlayerVpForTesting(p);
    }
  }

  // Micro-optimized flag retrieval outside the state progression loop
  double temp = absl::GetFlag(FLAGS_temperature);
  double shaped_weight = absl::GetFlag(FLAGS_shaped_reward_weight);

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

      // Query the batched evaluator (actor thread goes to sleep, OS multiplexes)
      EvalResult result = evaluator->Evaluate(obs);
      float current_value = result.value;
      std::vector<float> prev_logits_vec = std::move(result.logits);

      // Fast CPU Softmax Action Selection using the configured temperature
      Action action = actions.front();
      if (temp > 0.001) {
        std::vector<double> action_probs;
        action_probs.reserve(actions.size());
        
        // Find max logit among LEGAL actions for numerical stability
        float max_logit = -1e9f;
        for (Action a : actions) {
          if (prev_logits_vec[a] > max_logit) max_logit = prev_logits_vec[a];
        }
        
        for (Action a : actions) {
          action_probs.push_back(std::exp((prev_logits_vec[a] - max_logit) / temp));
        }
        
        // Sample action probabilistically from the network's beliefs
        std::discrete_distribution<size_t> dist(action_probs.begin(), action_probs.end());
        action = actions[dist(*rng)];
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

      state->ApplyAction(action);

      // Shaped Reward VP Gain Detection
      float shaped_bonus = 0.0f;
      if (dune_state != nullptr) {
        int new_vp = dune_state->GetPlayerVpForTesting(current_player);
        int vp_gained = new_vp - current_vps[current_player];
        current_vps[current_player] = new_vp;

        if (vp_gained > 0 && reward_lambda != nullptr) {
          float current_lambda = reward_lambda->load(std::memory_order_relaxed);
          shaped_bonus = vp_gained * static_cast<float>(shaped_weight) * current_lambda;
        }
      }

      // Push to trajectory
      Transition transition;
      transition.spatial_tensor = std::move(obs);
      transition.legal_actions = std::move(actions); // Store sparse legal actions list directly
      transition.action_taken = action;
      transition.prev_logits = std::move(prev_logits_vec);
      transition.q_value = shaped_bonus;
      transition.reward_target = shaped_bonus;
      transition.v_value = current_value; // Store the V(s) value
      transition.player_id = current_player;
      trajectory.push_back(std::move(transition));
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

      // Wiring the signals: q_value must be the TD(λ) return estimate (Q-value), not advantage, to avoid MMD target logit bias.
      it->q_value = A_i + V_i;
      it->reward_target = A_i + V_i; // TD(λ) return target


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
      transition.reward_target = 0.0f;
      transition.v_value = 0.0f; // Store V(s) = 0.0f explicitly
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
    }
  }

  return game_length;
}

void ThreadWorker(int thread_id, const Game* game, std::atomic<int>& games_completed,
                  int total_games, std::atomic<int>& total_moves,
                  GlobalReplayBuffer* global_buffer, int64_t obs_size
#ifdef OPEN_SPIEL_BUILD_WITH_LIBTORCH
                  , std::shared_ptr<BatchedEvaluator> evaluator
                  , std::atomic<float>* reward_lambda
#endif
) {
  std::mt19937 rng(std::random_device{}() + thread_id);
  int moves = 0;
  while (true) {
    int prev_completed = games_completed.fetch_add(1);
    if (prev_completed >= total_games) {
      games_completed.fetch_sub(1);
      break;
    }
    std::vector<Transition> local_trajectory;
#ifdef OPEN_SPIEL_BUILD_WITH_LIBTORCH
    if (evaluator != nullptr) {
      moves += TorchSimulation(&rng, *game, evaluator, obs_size, local_trajectory, reward_lambda);
    } else {
      moves += RandomSimulation(&rng, *game, obs_size, local_trajectory);
    }
#else
    moves += RandomSimulation(&rng, *game, obs_size, local_trajectory);
#endif
    global_buffer->PushTrajectory(local_trajectory);
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
    at::autocast::set_autocast_dtype(c10::DeviceType::CUDA, at::ScalarType::BFloat16);
  }

  std::shared_ptr<open_spiel::SharedDunePolicyValueNetImpl> training_model = nullptr;
  std::shared_ptr<open_spiel::SharedDunePolicyValueNetImpl> inference_model = nullptr;
  std::unique_ptr<torch::optim::AdamW> optimizer = nullptr;
  std::string model_path = absl::GetFlag(FLAGS_model_checkpoint);
  std::string optim_path = absl::GetFlag(FLAGS_optim_checkpoint);

  std::shared_mutex sync_mutex;
  std::thread optimization_thread;
  std::atomic<bool> stop_training(false);
  std::atomic<int> training_steps(absl::GetFlag(FLAGS_start_step));
  std::atomic<float> reward_lambda{1.0f};
  std::shared_ptr<open_spiel::BatchedEvaluator> evaluator = nullptr;

  if (obs_size > 0) {
    int hidden_dim = absl::GetFlag(FLAGS_hidden_dim);
    int num_blocks = absl::GetFlag(FLAGS_num_blocks);
    int eval_batch_size = absl::GetFlag(FLAGS_eval_batch_size);
    int eval_timeout_ms = absl::GetFlag(FLAGS_eval_timeout_ms);

    training_model = std::make_shared<open_spiel::SharedDunePolicyValueNetImpl>(obs_size, hidden_dim, action_size, num_blocks);
    inference_model = std::make_shared<open_spiel::SharedDunePolicyValueNetImpl>(obs_size, hidden_dim, action_size, num_blocks);
    
    inference_model->to(train_device);

    training_model->train(); // training model stays in train mode permanently
    inference_model->eval(); // inference model stays in eval mode permanently
    
    std::cout << absl::StrFormat("Initialized Double-Buffered SharedDunePolicyValueNet (Input Dim: %d, Hidden Dim: %d, Action Dim: %d, Blocks: %d)\n", 
                                 obs_size, hidden_dim, action_size, num_blocks);
    
    // Load existing checkpoint and set up the training model and optimizer correctly on the target hardware device
    open_spiel::LoadCheckpoint(training_model, optimizer, model_path, optim_path, train_device);
    // Initial weight synchronization
    open_spiel::SyncModels(training_model, inference_model, &sync_mutex);

    // Step-0 Logit Health Check: probe the loaded model with a dummy forward pass
    // and abort if raw logits already exceed the cap (indicates an exploded checkpoint)
    float logit_cap = static_cast<float>(absl::GetFlag(FLAGS_logit_cap));
    if (logit_cap > 0.0f) {
      torch::NoGradGuard no_grad;
      torch::Tensor probe_input = torch::zeros({1, obs_size}, torch::TensorOptions().device(train_device));
      auto probe_out = training_model->forward(probe_input);
      float max_abs_logit = probe_out.logits.abs().max().item<float>();
      std::cout << absl::StrFormat("[Step-0 Check] Max |raw logit|: %.2f (cap: %.1f)\n", max_abs_logit, logit_cap);
      if (max_abs_logit > logit_cap) {
        std::cerr << "\n=========================================\n"
                  << "[FATAL] Loaded checkpoint has exploded logits!\n"
                  << "Max |logit| = " << max_abs_logit << " > logit_cap = " << logit_cap << "\n"
                  << "This checkpoint cannot be safely resumed with logit capping.\n"
                  << "Start fresh or use an earlier checkpoint with healthy logits.\n"
                  << "=========================================\n";
        std::exit(EXIT_FAILURE);
      }
    }

    // Instantiate the BatchedEvaluator with target batch size, timeout, train_device, and logit cap
    evaluator = std::make_shared<open_spiel::BatchedEvaluator>(
        inference_model, eval_batch_size, eval_timeout_ms, train_device, &sync_mutex, logit_cap);
    std::cout << absl::StrFormat("BatchedEvaluator initialized with target batch size: %d, timeout: %d ms\n", 
                                 eval_batch_size, eval_timeout_ms);
  }
#endif

  // Construct global thread-safe replay buffer
  open_spiel::GlobalReplayBuffer global_buffer(absl::GetFlag(FLAGS_buffer_capacity));

  std::atomic<int> games_completed(0);
  std::atomic<int> total_moves(0);

#ifdef OPEN_SPIEL_BUILD_WITH_LIBTORCH
  // Start background optimization thread if async_mode is enabled
  bool async_mode = absl::GetFlag(FLAGS_async_mode);
  if (async_mode && training_model != nullptr) {
    int min_train_size = absl::GetFlag(FLAGS_min_train_size);
    int train_batch_size = absl::GetFlag(FLAGS_train_batch_size);
    int sync_interval = absl::GetFlag(FLAGS_sync_interval);

    std::cout << absl::StrFormat("Starting Asynchronous Optimization Worker (Min Train Size: %d, Batch Size: %d, Sync Interval: %d)...\n", 
                                 min_train_size, train_batch_size, sync_interval);

    optimization_thread = std::thread(
        open_spiel::OptimizationWorker,
        training_model, inference_model, optimizer.get(), &global_buffer,
        obs_size, action_size, &sync_mutex,
        std::ref(stop_training), std::ref(training_steps),
        min_train_size, train_batch_size, sync_interval,
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
                         &global_buffer, obs_size
#ifdef OPEN_SPIEL_BUILD_WITH_LIBTORCH
                         , evaluator, &reward_lambda
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
  if (training_model != nullptr && global_buffer.Size() > 0 && optimizer != nullptr) {
    if (async_mode) {
      std::cout << absl::StrFormat("\nAsynchronous training completed. Total background SGD steps executed: %d\n", training_steps.load());
      // Save checkpoint of training_model at program termination
      open_spiel::SaveCheckpoint(training_model, *optimizer, model_path, optim_path);
    } else {
      std::cout << "\nStarting optimization validation phase (Synchronous Mode)...\n";
      // Sample a single training batch of size 32
      size_t batch_size = 32;
      std::vector<std::shared_ptr<open_spiel::Transition>> batch = global_buffer.SampleBatch(batch_size);
      std::cout << absl::StrFormat("Sampled %zu transitions from replay buffer.\n", batch.size());

      // Execute backprop step using the startup-initialized optimizer
      open_spiel::PersistentTrainBuffers sync_buffers;
      sync_buffers.Initialize(batch_size, obs_size, action_size, train_device.is_cuda());
      open_spiel::TrainStep(training_model, *optimizer, batch, sync_buffers, obs_size, action_size, train_device);
      std::cout << "Optimization Step successfully completed (Forward + Loss + Backward + Weight Update)!\n";
      
      // Sync training weights to inference weights for completeness
      open_spiel::SyncModels(training_model, inference_model, &sync_mutex);

      // Save updated checkpoint
      open_spiel::SaveCheckpoint(training_model, *optimizer, model_path, optim_path);
    }
  }
#endif

  std::cout << absl::StrFormat("\n=== Benchmark Completed ===\n");
  std::cout << absl::StrFormat("Elapsed Time: %.3f seconds\n", seconds);
  std::cout << absl::StrFormat("Games Completed: %d\n", completed);
  std::cout << absl::StrFormat("Total Moves Executed: %d\n", moves);
  std::cout << absl::StrFormat("Final Global Buffer Size: %zu transitions\n", global_buffer.Size());
  std::cout << absl::StrFormat("Games Per Second (GPS): %.2f\n", completed / seconds);
  std::cout << absl::StrFormat("Moves Per Second (MPS): %.2f\n", moves / seconds);

  return 0;
}
