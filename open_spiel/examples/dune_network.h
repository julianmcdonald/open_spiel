#pragma once

#include <torch/torch.h>
#include <memory>
#include <vector>
#include <deque>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <chrono>
#include <shared_mutex>
#include <ATen/autocast_mode.h>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

namespace open_spiel {

// 1. Residual Block Definition
struct ResBlockImpl : torch::nn::Module {
  torch::nn::Linear fc1{nullptr};
  torch::nn::Linear fc2{nullptr};
  torch::nn::LayerNorm ln1{nullptr};
  torch::nn::LayerNorm ln2{nullptr};

  ResBlockImpl(int64_t dim) {
    fc1 = register_module("fc1", torch::nn::Linear(dim, dim));
    fc2 = register_module("fc2", torch::nn::Linear(dim, dim));
    ln1 = register_module("ln1", torch::nn::LayerNorm(torch::nn::LayerNormOptions({dim})));
    ln2 = register_module("ln2", torch::nn::LayerNorm(torch::nn::LayerNormOptions({dim})));
  }

  torch::Tensor forward(torch::Tensor x) {
    torch::Tensor residual = x;
    x = torch::relu(ln1->forward(fc1->forward(x)));
    x = ln2->forward(fc2->forward(x));
    return torch::relu(x + residual);
  }
};
TORCH_MODULE(ResBlock);

// 2. Main Dual-Headed Policy-Value Network Definition
struct SharedDunePolicyValueNetImpl : torch::nn::Module {
  torch::nn::Linear input_layer{nullptr};
  std::vector<std::shared_ptr<ResBlockImpl>> res_blocks;
  torch::nn::Linear policy_head{nullptr};
  torch::nn::Linear value_head{nullptr};

  SharedDunePolicyValueNetImpl(int64_t input_dim, int64_t hidden_dim = 2048, int64_t action_dim = 2391, int num_blocks = 8) {
    input_layer = register_module("input_layer", torch::nn::Linear(input_dim, hidden_dim));
    
    // Register custom submodules dynamically (res1, res2, etc.)
    for (int i = 0; i < num_blocks; ++i) {
      auto block = std::make_shared<ResBlockImpl>(hidden_dim);
      res_blocks.push_back(block);
      register_module("res" + std::to_string(i + 1), block);
    }
    
    policy_head = register_module("policy_head", torch::nn::Linear(hidden_dim, action_dim));
    value_head = register_module("value_head", torch::nn::Linear(hidden_dim, 1));
  }

  struct ModelOutputs {
    torch::Tensor logits;
    torch::Tensor values;
  };

  ModelOutputs forward(torch::Tensor x) {
    x = torch::relu(input_layer->forward(x));
    for (auto& block : res_blocks) {
      x = block->forward(x);
    }
    
    torch::Tensor logits = policy_head->forward(x);
    torch::Tensor values = torch::tanh(value_head->forward(x));
    return {logits, values};
  }
};
TORCH_MODULE(SharedDunePolicyValueNet);

struct AutocastGuard {
    c10::DeviceType device_type_;
    bool previous_state_;
    AutocastGuard(c10::DeviceType device_type, bool enabled)
        : device_type_(device_type) {
        previous_state_ = at::autocast::is_autocast_enabled(device_type_);
        at::autocast::set_autocast_enabled(device_type_, enabled);
    }
    ~AutocastGuard() {
        at::autocast::set_autocast_enabled(device_type_, previous_state_);
    }
};

struct EvalResult {
    std::vector<float> logits;
    float value; 
};

class BatchedEvaluator {
public:
    BatchedEvaluator(std::shared_ptr<SharedDunePolicyValueNetImpl> model, 
                     int target_batch_size, 
                     int timeout_ms, 
                     torch::Device device,
                     std::shared_mutex* sync_mutex)
        : model_(model), target_batch_size_(target_batch_size), 
          timeout_ms_(timeout_ms), device_(device), sync_mutex_(sync_mutex), stop_(false) {
        
        // Enable TF32 for Ada Lovelace (RTX 4080 Super) speedup
        if (device_.is_cuda()) {
            at::globalContext().setAllowTF32CuBLAS(true);
            at::globalContext().setAllowTF32CuDNN(true);
        }
        
        model_->eval(); // Defensive hygiene: ResBlocks use LayerNorm so this is a no-op, but protects future Dropout additions.
        runner_thread_ = std::thread(&BatchedEvaluator::Runner, this);
    }

    ~BatchedEvaluator() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stop_ = true;
        }
        cv_.notify_all();
        if (runner_thread_.joinable()) runner_thread_.join();
    }

    // Called by the actor threads
    EvalResult Evaluate(const std::vector<float>& obs) {
        EvalResult result; // Stack allocated!
        std::atomic<bool> ready{false};

        {
            std::lock_guard<std::mutex> lock(mutex_);
            requests_.push_back({&obs, &result, &ready});
            
            // Only wake the Runner on the first arrival or when the batch is full
            if (requests_.size() == 1 || requests_.size() >= (size_t)target_batch_size_) {
                cv_.notify_one(); 
            }
        }
        
        // HYBRID WAIT: Spin briefly to catch ultra-fast GPU responses, then park.
        int spin_count = 0;
        while (!ready.load(std::memory_order_acquire)) {
            if (spin_count < 4000) {
#if defined(__x86_64__) || defined(_M_X64)
                _mm_pause(); // Emits PAUSE instruction to prevent pipeline flushing
#else
                std::this_thread::yield();
#endif
                spin_count++;
            } else {
                // Park safely (Zero CPU burn)
                std::unique_lock<std::mutex> park_lock(park_mutex_);
                park_cv_.wait(park_lock, [&] { return ready.load(std::memory_order_acquire); });
            }
        }
        
        return result; // Move semantics
    }

private:
    struct Request {
        const std::vector<float>* obs;
        EvalResult* result_dest;
        std::atomic<bool>* ready_flag;
    };
    
    std::mutex park_mutex_;
    std::condition_variable park_cv_;

    std::shared_ptr<SharedDunePolicyValueNetImpl> model_;
    int target_batch_size_;
    int timeout_ms_;
    torch::Device device_;
    std::shared_mutex* sync_mutex_;
    
    std::mutex mutex_;
    std::condition_variable cv_;
    std::deque<Request> requests_;
    std::chrono::steady_clock::time_point first_request_ts_;
    
    bool stop_;
    std::thread runner_thread_;

    void Runner() {
        torch::NoGradGuard no_grad;
        torch::Tensor pinned_stacked_obs;
        bool pinned_allocated = false;

        while (true) {
            std::vector<Request> batch;
            
            {
                std::unique_lock<std::mutex> lock(mutex_);
                
                // 1. Wait until we have at least one request or we are shutting down
                cv_.wait(lock, [this]() { return stop_ || !requests_.empty(); });
                if (stop_ && requests_.empty()) break;
                
                // 2. We have requests. Calculate the absolute timeout deadline
                auto deadline = first_request_ts_ + std::chrono::milliseconds(timeout_ms_);
                
                // 3. Wait until the batch is full OR the timeout expires (Deadlock fix!)
                cv_.wait_until(lock, deadline, [this, deadline]() {
                    return stop_ || requests_.size() >= (size_t)target_batch_size_ || 
                           std::chrono::steady_clock::now() >= deadline;
                });

                if (stop_ && requests_.empty()) break;

                // 4. Scoop up the batch
                size_t actual_batch = std::min((size_t)target_batch_size_, requests_.size());
                batch.reserve(actual_batch);
                for (size_t i = 0; i < actual_batch; ++i) {
                    batch.push_back(std::move(requests_.front()));
                    requests_.pop_front();
                }
                
                // Reset the timer if there are leftovers
                if (!requests_.empty()) {
                    first_request_ts_ = std::chrono::steady_clock::now();
                }
            }

            if (batch.empty()) continue;

            size_t batch_size = batch.size();
            size_t obs_size = batch[0].obs->size(); // Add pointer arrow

            // A. PINNED MEMORY: Allocate exactly ONCE outside the hot loop
            if (!pinned_allocated) {
                if (device_.is_cuda()) {
                    auto options = torch::TensorOptions().dtype(torch::kFloat32).pinned_memory(true);
                    pinned_stacked_obs = torch::empty({(long)target_batch_size_, (long)obs_size}, options);
                } else {
                    pinned_stacked_obs = torch::empty({(long)target_batch_size_, (long)obs_size}, torch::kFloat32);
                }
                pinned_allocated = true;
            }
            
            float* dest_ptr = pinned_stacked_obs.data_ptr<float>();
            for (size_t i = 0; i < batch_size; ++i) {
                // Add pointer arrow to obs->data()
                std::memcpy(dest_ptr + i * obs_size, batch[i].obs->data(), obs_size * sizeof(float));
            }

            // Non-blocking H2D transfer using .slice() for partial batches
            torch::Tensor device_obs = device_.is_cuda() 
                ? pinned_stacked_obs.slice(0, 0, batch_size).to(device_, /*non_blocking=*/true) 
                : pinned_stacked_obs.slice(0, 0, batch_size);

            // B. FORWARD PASS WITH AMP (FP16/TF32) AND WRITE PROTECTION
            torch::Tensor pred_logits, pred_values;
            {
                std::shared_lock<std::shared_mutex> lock(*sync_mutex_);
                if (device_.is_cuda()) {
                    AutocastGuard autocast_guard(c10::DeviceType::CUDA, true);
                    auto outputs = model_->forward(device_obs);
                    pred_logits = outputs.logits;
                    pred_values = outputs.values;
                } else {
                    auto outputs = model_->forward(device_obs);
                    pred_logits = outputs.logits;
                    pred_values = outputs.values;
                }
            }

            // C. NON-BLOCKING D2H TRANSFER (cast AMP FP16 outputs back to Float32)
            if (device_.is_cuda()) {
                pred_logits = pred_logits.to(torch::kFloat32).to(torch::kCPU, /*non_blocking=*/true).contiguous();
                pred_values = pred_values.to(torch::kFloat32).to(torch::kCPU, /*non_blocking=*/true).contiguous();
                
                // D. CRITICAL: Wait for asynchronous transfers to complete before reading!
                torch::cuda::synchronize();
            } else {
                pred_logits = pred_logits.contiguous();
                pred_values = pred_values.contiguous();
            }

            // --- DISTRIBUTE RESULTS & WAKE THREADS ---
            float* logits_ptr = pred_logits.data_ptr<float>();
            float* values_ptr = pred_values.data_ptr<float>();
            size_t action_dim = pred_logits.size(1);

            for (size_t i = 0; i < batch_size; ++i) {
                batch[i].result_dest->logits.assign(logits_ptr + i * action_dim, logits_ptr + (i + 1) * action_dim);
                batch[i].result_dest->value = values_ptr[i];
                
                // Set the atomic flag (fast spinning threads will catch this instantly)
                batch[i].ready_flag->store(true, std::memory_order_release);
            }
            
            // Memory barrier to guarantee parked threads see the ready flag, 
            // followed by a SINGLE broadcast syscall (instead of 64 individual futex wakes)
            { std::lock_guard<std::mutex> park_lock(park_mutex_); }
            park_cv_.notify_all();
        }
    }
};

} // namespace open_spiel
