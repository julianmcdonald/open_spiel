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

#include <random>
#include <shared_mutex>
#include <string>
#include <thread>
#include <vector>

#include "open_spiel/abseil-cpp/absl/flags/flag.h"
#include "open_spiel/abseil-cpp/absl/flags/parse.h"
#include "open_spiel/abseil-cpp/absl/strings/str_format.h"
#include "open_spiel/games/dune_imperium/dune_imperium.h"

#include "open_spiel/games/dune_imperium/dune_imperium_util.h"
#include "open_spiel/spiel.h"

#ifdef OPEN_SPIEL_BUILD_WITH_LIBTORCH
#include <torch/torch.h>

#include "dune_network.h"
#include "dune_seed_utils.h"
#include "dune_sha256.h"
#include "dune_ppo_training_utils.h"
#include "open_spiel/utils/json.h"
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
ABSL_FLAG(double, gae_lambda, 1.0, "GAE lambda.");
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
ABSL_FLAG(bool, deterministic, true, "Enable strict PyTorch/LibTorch deterministic algorithms.");
ABSL_FLAG(bool, deterministic_rollout_eval, false, "Use deterministic batch-1 rollout evaluation.");
ABSL_FLAG(bool, diagnostics_only, false, "Collect one rollout, write diagnostics, and exit without optimization.");

ABSL_FLAG(double, shaped_reward_weight, 0.2,
          "Weight for intermediate VP shaped rewards.");
ABSL_FLAG(double, tleilaxu_breadcrumb_weight, 0.0,
          "Weight for Tleilaxu levels 5/6 breadcrumbs.");
ABSL_FLAG(double, tleilaxu_level7_breadcrumb_weight, 0.0,
          "Weight for Tleilaxu level 7 breadcrumb.");
ABSL_FLAG(double, reward_scale, 4.0,
          "Divide shaped plus terminal rewards by this value.");

ABSL_FLAG(std::string, model_checkpoint, "dune_ppo_model.pt",
          "Model checkpoint to load/save.");
ABSL_FLAG(std::string, optim_checkpoint, "dune_ppo_optimizer.pt",
          "Optimizer checkpoint to load/save.");
ABSL_FLAG(std::string, artifact_manifest, "",
          "Path to the Phase 1 baseline artifact manifest.json for validation.");
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

ABSL_FLAG(std::string, init_mode, "",
          "Initialization mode (required): random, checkpoint, bootstrap, validate_legacy.");
ABSL_FLAG(uint64_t, shaping_start_env_steps, 206830543,
          "Environment steps at which shaped reward decay starts.");
ABSL_FLAG(uint64_t, shaping_decay_env_steps, 0,
          "Environment steps duration for shaped reward decay (0 means no decay).");
ABSL_FLAG(int, seed_scheme_version, 2, "Seed scheme version.");
ABSL_FLAG(int, target_end_update, 2450, "Absolute target update number to train until.");
ABSL_FLAG(int, start_update, 1, "Fallback start update for bootstrap mode.");
ABSL_FLAG(uint64_t, start_env_steps, 0, "Fallback start environment steps for bootstrap mode.");
ABSL_FLAG(uint64_t, start_episode_id, 0, "Fallback start episode ID for bootstrap mode.");
ABSL_FLAG(std::string, diagnostics_path, "",
          "Path to write structured diagnostics JSON/CSV. Extension determines format.");

namespace open_spiel {
namespace {

#ifdef OPEN_SPIEL_BUILD_WITH_LIBTORCH

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

  std::vector<PpoTransition> TakeAll(bool* out_episode_ids_unique = nullptr) {
    std::lock_guard<std::mutex> lock(mu_);
    if (out_episode_ids_unique) {
      *out_episode_ids_unique = true;
      std::unordered_set<uint64_t> seen_ids;
      for (const auto& traj : trajectories_) {
        if (!traj.empty()) {
          uint64_t ep_id = traj[0].episode_id;
          if (seen_ids.count(ep_id)) {
            *out_episode_ids_unique = false;
          }
          seen_ids.insert(ep_id);
        }
      }
    }
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
  uint8_t player_id;
};

class SearchLabelBuffer {
 public:
  void SetExpectedDimensions(int64_t obs_size, int64_t action_dim) {
    expected_obs_size_ = obs_size;
    expected_action_dim_ = action_dim;
  }

  void LoadFromDirectory(const std::string& dir) {
    if (dir.empty() || !std::filesystem::exists(dir)) return;
    
    // Verify manifest.json
    std::filesystem::path manifest_path = std::filesystem::path(dir) / "manifest.json";
    if (!std::filesystem::exists(manifest_path)) {
      SpielFatalError("SearchLabelBuffer: manifest.json not found in " + dir);
    }
    
    std::ifstream ifs(manifest_path.string());
    if (!ifs) {
      SpielFatalError("SearchLabelBuffer: Failed to open manifest.json in " + dir);
    }
    std::string content((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
    ifs.close();
    
    auto val_opt = open_spiel::json::FromString(content);
    if (!val_opt) {
      SpielFatalError("SearchLabelBuffer: Failed to parse manifest.json in " + dir);
    }
    
    const auto& manifest_obj = val_opt->GetObject();
    
    // Verify schema_version
    auto schema_it = manifest_obj.find("schema_version");
    if (schema_it == manifest_obj.end() || static_cast<int>(schema_it->second.GetInt()) != 2) {
      SpielFatalError("SearchLabelBuffer: Unsupported manifest schema version in " + dir + " (expected 2)");
    }
    
    // Verify training_label_count & validation_label_count
    auto train_cnt_it = manifest_obj.find("training_label_count");
    auto val_cnt_it = manifest_obj.find("validation_label_count");
    if (train_cnt_it == manifest_obj.end() || val_cnt_it == manifest_obj.end()) {
      SpielFatalError("SearchLabelBuffer: Missing training/validation counts in manifest.json");
    }
    
    int64_t training_count = train_cnt_it->second.GetInt();
    int64_t validation_count = val_cnt_it->second.GetInt();
    if (training_count < 8192 || validation_count < 1024) {
      SpielFatalError("SearchLabelBuffer: Manifest training count (" + std::to_string(training_count) + 
                     ") or validation count (" + std::to_string(validation_count) + ") is insufficient.");
    }
    
    // Verify files & SHA-256 hashes
    auto files_it = manifest_obj.find("files");
    if (files_it == manifest_obj.end()) {
      SpielFatalError("SearchLabelBuffer: Missing 'files' list in manifest.json");
    }
    const auto& files_arr = files_it->second.GetArray();
    for (const auto& f_val : files_arr) {
      const auto& f_obj = f_val.GetObject();
      auto fn_it = f_obj.find("filename");
      auto hash_it = f_obj.find("sha256");
      if (fn_it == f_obj.end() || hash_it == f_obj.end()) {
        SpielFatalError("SearchLabelBuffer: Missing filename or sha256 in file entry");
      }
      
      std::string filename = fn_it->second.GetString();
      std::string expected_sha256 = hash_it->second.GetString();
      std::filesystem::path bin_path = std::filesystem::path(dir) / filename;
      if (!std::filesystem::exists(bin_path)) {
        SpielFatalError("SearchLabelBuffer: Label file " + filename + " listed in manifest does not exist.");
      }
      
      std::string actual_sha256 = open_spiel::ComputeFileSHA256(bin_path.string());
      if (actual_sha256 != expected_sha256) {
        SpielFatalError("SearchLabelBuffer: Hash mismatch for file " + filename + 
                       ": expected=" + expected_sha256 + " actual=" + actual_sha256);
      }
    }
    
    // Reconstruct semantic object for fingerprint verification
    open_spiel::json::Object semantic_obj;
    semantic_obj["schema_version"] = manifest_obj.at("schema_version");
    semantic_obj["base_seed"] = manifest_obj.at("base_seed");
    semantic_obj["model_checkpoint_sha256"] = manifest_obj.at("model_checkpoint_sha256");
    semantic_obj["effective_search_config"] = manifest_obj.at("effective_search_config");
    semantic_obj["architecture"] = manifest_obj.at("architecture");
    semantic_obj["training_label_count"] = manifest_obj.at("training_label_count");
    semantic_obj["validation_label_count"] = manifest_obj.at("validation_label_count");
    
    std::string semantic_json = open_spiel::json::ToString(semantic_obj);
    std::string expected_fingerprint = open_spiel::ComputeStringSHA256(semantic_json);
    
    auto fp_it = manifest_obj.find("search_label_fingerprint");
    if (fp_it == manifest_obj.end() || fp_it->second.GetString() != expected_fingerprint) {
      SpielFatalError("SearchLabelBuffer: Manifest search_label_fingerprint mismatch!");
    }
    
    std::cout << "SearchLabelBuffer: manifest.json verified successfully. Fingerprint: " 
              << expected_fingerprint << "\n";
              
    LoadNewFiles(dir);
  }

  void LoadNewFiles(const std::string& dir) {
    if (dir.empty() || !std::filesystem::exists(dir)) return;
    std::lock_guard<std::mutex> lock(mu_);
    std::vector<std::string> paths;
    for (const auto& entry : std::filesystem::directory_iterator(dir)) {
      if (entry.path().extension() == ".bin") {
        paths.push_back(entry.path().string());
      }
    }
    std::sort(paths.begin(), paths.end());
    for (const auto& path_str : paths) {
      if (loaded_files_.find(path_str) == loaded_files_.end()) {
        LoadFile(path_str);
        loaded_files_.insert(path_str);
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

  double ComputeValidationKL(const std::shared_ptr<open_spiel::SharedDunePolicyValueNetImpl>& model, 
                             const torch::Device& device, float logit_cap) const {
    std::lock_guard<std::mutex> lock(mu_);
    if (validation_labels_.empty()) return 0.0;
    
    torch::NoGradGuard no_grad;
    const size_t batch_size = 512;
    double kl_sum = 0.0;
    size_t total_count = 0;
    
    for (size_t i = 0; i < validation_labels_.size(); i += batch_size) {
      size_t current_batch_size = std::min(batch_size, validation_labels_.size() - i);
      
      torch::Tensor states_cpu = torch::zeros({static_cast<int64_t>(current_batch_size), expected_obs_size_}, torch::kFloat);
      torch::Tensor masks_cpu = torch::zeros({static_cast<int64_t>(current_batch_size), expected_action_dim_}, torch::kBool);
      torch::Tensor teacher_probs_cpu = torch::zeros({static_cast<int64_t>(current_batch_size), expected_action_dim_}, torch::kFloat);
      
      float* states_ptr = states_cpu.data_ptr<float>();
      bool* masks_ptr = masks_cpu.data_ptr<bool>();
      float* teacher_ptr = teacher_probs_cpu.data_ptr<float>();
      
      for (size_t j = 0; j < current_batch_size; ++j) {
        const auto& label = validation_labels_[i + j];
        std::copy(label.state.begin(), label.state.end(), states_ptr + j * expected_obs_size_);
        for (const auto& ap : label.teacher_probs) {
          int action_id = ap.first;
          float prob = ap.second;
          masks_ptr[j * expected_action_dim_ + action_id] = true;
          teacher_ptr[j * expected_action_dim_ + action_id] = prob;
        }
      }
      
      torch::Tensor states = states_cpu.to(device);
      torch::Tensor masks = masks_cpu.to(device);
      torch::Tensor teacher_probs = teacher_probs_cpu.to(device);
      
      auto outputs = model->forward(states);
      torch::Tensor logits = CenterAndCapLogitsTensor(outputs.logits, masks, logit_cap);
      torch::Tensor masked_logits = logits.masked_fill(masks.logical_not(), -1e9f);
      torch::Tensor log_probs = torch::log_softmax(masked_logits, -1);
      
      torch::Tensor log_teacher = torch::log(teacher_probs.clamp_min(1e-12f));
      torch::Tensor kl_loss = teacher_probs * (log_teacher - log_probs);
      torch::Tensor mean_kl = kl_loss.sum(-1);
      
      kl_sum += mean_kl.sum().item<double>();
      total_count += current_batch_size;
    }
    
    return total_count > 0 ? (kl_sum / total_count) : 0.0;
  }

  size_t ValidationSize() const {
    std::lock_guard<std::mutex> lock(mu_);
    return validation_labels_.size();
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
    if (schema != 1 && schema != 2) {
      std::cerr << "SearchLabelBuffer: Unsupported schema version " << schema
                << " in " << path << " (expected 1 or 2)\n";
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
    int val_before = validation_labels_.size();
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
      
      uint8_t pid = 0;
      uint8_t padding[2];
      in.read(reinterpret_cast<char*>(&pid), sizeof(uint8_t));
      in.read(reinterpret_cast<char*>(padding), 2);
      if (!in) break;
      label.player_id = pid;

      std::vector<int32_t> legal_actions;
      for (const auto& ap : label.teacher_probs) {
        legal_actions.push_back(static_cast<int32_t>(ap.first));
      }

      if (IsValidationLabel(label.state, legal_actions, label.player_id)) {
        validation_labels_.push_back(std::move(label));
      } else {
        labels_.push_back(std::move(label));
      }
    }
    int labels_loaded = labels_.size() - labels_before;
    int val_loaded = validation_labels_.size() - val_before;
    std::cout << "SearchLabelBuffer: Loaded " << labels_loaded
              << " train, " << val_loaded << " val labels from " << path << "\n";
  }

  int64_t expected_obs_size_ = 0;
  int64_t expected_action_dim_ = 0;
  uint64_t expected_fingerprint_ = 0;
  bool has_fingerprint_ = false;
  std::vector<SearchLabel> labels_;
  std::vector<SearchLabel> validation_labels_;
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

  double conflict_vp_generated = 0.0;
  double conflict_vp_attributed = 0.0;
  double conflict_vp_unattributed = 0.0;

  // Raw VPs for conservation checks
  double raw_conflict_vp = 0.0;
  double raw_noncombat_vp = 0.0;
  double raw_total_vp = 0.0;
};


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

std::string GenerateUUID() {
  std::random_device rd;
  std::mt19937_64 generator(rd());
  uint64_t r1 = generator();
  uint64_t r2 = generator();
  // Set version 4 (random) and variant (RFC 4122)
  uint32_t a = (r1 >> 32);
  uint16_t b = (r1 >> 16) & 0xFFFF;
  uint16_t c = (r1 & 0x0FFF) | 0x4000;
  uint16_t d = ((r2 >> 48) & 0x3FFF) | 0x8000;
  uint64_t e = r2 & 0xFFFFFFFFFFFFULL;
  return absl::StrFormat("%08x-%04x-%04x-%04x-%012llx", a, b, c, d, e);
}

std::string ComputeLegacyConfigFingerprint() {
  open_spiel::json::Object config_obj;
  config_obj["game"] = absl::GetFlag(FLAGS_game);
  config_obj["ppo_minibatch_size"] = absl::GetFlag(FLAGS_ppo_minibatch_size);
  config_obj["ppo_update_epochs"] = absl::GetFlag(FLAGS_ppo_update_epochs);
  config_obj["learning_rate"] = absl::GetFlag(FLAGS_learning_rate);
  config_obj["anneal_lr"] = absl::GetFlag(FLAGS_anneal_lr);
  config_obj["gamma"] = absl::GetFlag(FLAGS_gamma);
  config_obj["gae_lambda"] = absl::GetFlag(FLAGS_gae_lambda);
  config_obj["normalize_advantages"] = absl::GetFlag(FLAGS_normalize_advantages);
  config_obj["ppo_clip_epsilon"] = absl::GetFlag(FLAGS_ppo_clip_epsilon);
  config_obj["ppo_clip_value_loss"] = absl::GetFlag(FLAGS_ppo_clip_value_loss);
  config_obj["entropy_coef"] = absl::GetFlag(FLAGS_entropy_coef);
  config_obj["value_coef"] = absl::GetFlag(FLAGS_value_coef);
  config_obj["grad_clip_norm"] = absl::GetFlag(FLAGS_grad_clip_norm);
  config_obj["target_kl"] = absl::GetFlag(FLAGS_target_kl);
  config_obj["weight_decay"] = absl::GetFlag(FLAGS_weight_decay);
  config_obj["policy_weight_decay"] = absl::GetFlag(FLAGS_policy_weight_decay);
  config_obj["hidden_dim"] = absl::GetFlag(FLAGS_hidden_dim);
  config_obj["num_blocks"] = absl::GetFlag(FLAGS_num_blocks);
  config_obj["logit_cap"] = absl::GetFlag(FLAGS_logit_cap);
  config_obj["shaped_reward_weight"] = absl::GetFlag(FLAGS_shaped_reward_weight);
  config_obj["tleilaxu_breadcrumb_weight"] = absl::GetFlag(FLAGS_tleilaxu_breadcrumb_weight);
  config_obj["tleilaxu_level7_breadcrumb_weight"] = absl::GetFlag(FLAGS_tleilaxu_level7_breadcrumb_weight);
  config_obj["shaping_start_env_steps"] = static_cast<int64_t>(absl::GetFlag(FLAGS_shaping_start_env_steps));
  config_obj["shaping_decay_env_steps"] = static_cast<int64_t>(absl::GetFlag(FLAGS_shaping_decay_env_steps));
  config_obj["reward_scale"] = absl::GetFlag(FLAGS_reward_scale);
  config_obj["train_amp"] = absl::GetFlag(FLAGS_train_amp);
  config_obj["pipeline"] = absl::GetFlag(FLAGS_pipeline);
  config_obj["rollout_games"] = absl::GetFlag(FLAGS_rollout_games);
  config_obj["rollout_transitions"] = absl::GetFlag(FLAGS_rollout_transitions);
  config_obj["search_lambda"] = absl::GetFlag(FLAGS_search_lambda);
  config_obj["search_minibatches_per_update"] = absl::GetFlag(FLAGS_search_minibatches_per_update);
  config_obj["search_minibatch_size"] = absl::GetFlag(FLAGS_search_minibatch_size);

  std::string json_str = open_spiel::json::ToString(config_obj);
  return open_spiel::ComputeStringSHA256(json_str);
}

std::string ComputeConfigFingerprint() {
  open_spiel::json::Object config_obj;
  config_obj["game"] = absl::GetFlag(FLAGS_game);
  config_obj["ppo_minibatch_size"] = absl::GetFlag(FLAGS_ppo_minibatch_size);
  config_obj["ppo_update_epochs"] = absl::GetFlag(FLAGS_ppo_update_epochs);
  config_obj["learning_rate"] = absl::GetFlag(FLAGS_learning_rate);
  config_obj["anneal_lr"] = absl::GetFlag(FLAGS_anneal_lr);
  config_obj["gamma"] = absl::GetFlag(FLAGS_gamma);
  config_obj["gae_lambda"] = absl::GetFlag(FLAGS_gae_lambda);
  config_obj["normalize_advantages"] = absl::GetFlag(FLAGS_normalize_advantages);
  config_obj["ppo_clip_epsilon"] = absl::GetFlag(FLAGS_ppo_clip_epsilon);
  config_obj["ppo_clip_value_loss"] = absl::GetFlag(FLAGS_ppo_clip_value_loss);
  config_obj["entropy_coef"] = absl::GetFlag(FLAGS_entropy_coef);
  config_obj["value_coef"] = absl::GetFlag(FLAGS_value_coef);
  config_obj["grad_clip_norm"] = absl::GetFlag(FLAGS_grad_clip_norm);
  config_obj["target_kl"] = absl::GetFlag(FLAGS_target_kl);
  config_obj["weight_decay"] = absl::GetFlag(FLAGS_weight_decay);
  config_obj["policy_weight_decay"] = absl::GetFlag(FLAGS_policy_weight_decay);
  config_obj["hidden_dim"] = absl::GetFlag(FLAGS_hidden_dim);
  config_obj["num_blocks"] = absl::GetFlag(FLAGS_num_blocks);
  config_obj["logit_cap"] = absl::GetFlag(FLAGS_logit_cap);
  config_obj["shaped_reward_weight"] = absl::GetFlag(FLAGS_shaped_reward_weight);
  config_obj["tleilaxu_breadcrumb_weight"] = absl::GetFlag(FLAGS_tleilaxu_breadcrumb_weight);
  config_obj["tleilaxu_level7_breadcrumb_weight"] = absl::GetFlag(FLAGS_tleilaxu_level7_breadcrumb_weight);
  config_obj["shaping_start_env_steps"] = static_cast<int64_t>(absl::GetFlag(FLAGS_shaping_start_env_steps));
  config_obj["shaping_decay_env_steps"] = static_cast<int64_t>(absl::GetFlag(FLAGS_shaping_decay_env_steps));
  config_obj["reward_scale"] = absl::GetFlag(FLAGS_reward_scale);
  config_obj["train_amp"] = absl::GetFlag(FLAGS_train_amp);
  config_obj["pipeline"] = absl::GetFlag(FLAGS_pipeline);
  config_obj["rollout_games"] = absl::GetFlag(FLAGS_rollout_games);
  config_obj["rollout_transitions"] = absl::GetFlag(FLAGS_rollout_transitions);
  config_obj["search_lambda"] = absl::GetFlag(FLAGS_search_lambda);
  config_obj["search_minibatches_per_update"] = absl::GetFlag(FLAGS_search_minibatches_per_update);
  config_obj["search_minibatch_size"] = absl::GetFlag(FLAGS_search_minibatch_size);

  // New flags added for complete config fingerprint validation
  config_obj["deterministic"] = absl::GetFlag(FLAGS_deterministic);
  config_obj["deterministic_rollout_eval"] = absl::GetFlag(FLAGS_deterministic_rollout_eval);

  std::string json_str = open_spiel::json::ToString(config_obj);
  return open_spiel::ComputeStringSHA256(json_str);
}

std::string GetSearchLabelFingerprint(const std::string& search_label_dir) {
  if (search_label_dir.empty()) {
    return "";
  }
  std::filesystem::path manifest_path = std::filesystem::path(search_label_dir) / "manifest.json";
  if (!std::filesystem::exists(manifest_path)) {
    std::cout << "Warning: search_label_dir is specified but manifest.json not found at " << manifest_path << "\n";
    return "";
  }
  try {
    std::ifstream ifs(manifest_path);
    std::string content((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
    auto val_opt = open_spiel::json::FromString(content);
    if (val_opt.has_value() && val_opt->IsObject()) {
      const auto& obj = val_opt->GetObject();
      auto it = obj.find("search_label_fingerprint");
      if (it != obj.end() && it->second.IsString()) {
        return it->second.GetString();
      }
    }
  } catch (const std::exception& e) {
    std::cerr << "Error reading search label manifest at " << manifest_path << ": " << e.what() << "\n";
  }
  return "";
}

void SaveCheckpoint(std::shared_ptr<SharedDunePolicyValueNetImpl> model,
                    torch::optim::AdamW& optimizer,
                    const std::string& model_path,
                    const std::string& optim_path,
                    int global_update,
                    int target_end_update,
                    uint64_t total_env_steps,
                    uint64_t next_episode_id,
                    uint64_t base_seed,
                    int seed_scheme_version,
                    const std::string& config_fingerprint,
                    const std::string& search_label_fingerprint,
                    const std::string& run_uuid) {
  std::string model_tmp = model_path + ".tmp";
  std::string optim_tmp = optim_path + ".tmp";
  
  std::filesystem::path manifest_path = model_path;
  manifest_path.replace_extension(".json");
  std::string manifest_path_str = manifest_path.string();
  std::string manifest_tmp = manifest_path_str + ".tmp";

  try {
    torch::save(model, model_tmp);
    torch::save(optimizer, optim_tmp);

    size_t model_size = 0;
    std::string model_hash = ComputeFileSHA256(model_tmp, &model_size);

    size_t optim_size = 0;
    std::string optim_hash = ComputeFileSHA256(optim_tmp, &optim_size);

    std::string checkpoint_uuid = GenerateUUID();
    json::Object manifest_obj;
    manifest_obj["schema_version"] = static_cast<int64_t>(2);
    manifest_obj["checkpoint_uuid"] = checkpoint_uuid;
    manifest_obj["global_update"] = static_cast<int64_t>(global_update);
    manifest_obj["target_end_update"] = static_cast<int64_t>(target_end_update);
    manifest_obj["total_env_steps"] = static_cast<int64_t>(total_env_steps);
    manifest_obj["next_episode_id"] = static_cast<int64_t>(next_episode_id);
    manifest_obj["base_seed"] = static_cast<int64_t>(base_seed);
    manifest_obj["seed_scheme_version"] = static_cast<int64_t>(seed_scheme_version);
    manifest_obj["config_fingerprint"] = config_fingerprint;
    manifest_obj["search_label_fingerprint"] = search_label_fingerprint;
    manifest_obj["run_uuid"] = run_uuid;
    manifest_obj["model_filename"] = std::filesystem::path(model_path).filename().string();
    manifest_obj["model_file_size"] = static_cast<int64_t>(model_size);
    manifest_obj["model_sha256"] = model_hash;
    manifest_obj["optimizer_filename"] = std::filesystem::path(optim_path).filename().string();
    manifest_obj["optimizer_file_size"] = static_cast<int64_t>(optim_size);
    manifest_obj["optimizer_sha256"] = optim_hash;
    manifest_obj["hidden_dim"] = static_cast<int64_t>(absl::GetFlag(FLAGS_hidden_dim));
    manifest_obj["num_blocks"] = static_cast<int64_t>(absl::GetFlag(FLAGS_num_blocks));

    {
      std::ofstream ofs(manifest_tmp);
      if (!ofs) {
        throw std::runtime_error("Could not open manifest temp file for writing: " + manifest_tmp);
      }
      ofs << json::ToString(manifest_obj, true);
    }

    std::filesystem::rename(model_tmp, model_path);
    std::filesystem::rename(optim_tmp, optim_path);
    std::filesystem::rename(manifest_tmp, manifest_path);

    std::cout << "Saved checkpoint successfully:\n"
              << "  Model: " << model_path << " (" << model_size << " bytes, sha256=" << model_hash << ")\n"
              << "  Optimizer: " << optim_path << " (" << optim_size << " bytes, sha256=" << optim_hash << ")\n"
              << "  Manifest: " << manifest_path << "\n";
  } catch (const std::exception& e) {
    std::cerr << "CRITICAL ERROR: SaveCheckpoint failed: " << e.what() << "\n";
    if (std::filesystem::exists(model_tmp)) std::filesystem::remove(model_tmp);
    if (std::filesystem::exists(optim_tmp)) std::filesystem::remove(optim_tmp);
    if (std::filesystem::exists(manifest_tmp)) std::filesystem::remove(manifest_tmp);
    SpielFatalError("SaveCheckpoint failed, aborting training to prevent corrupted states.");
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
                  std::shared_ptr<IGameEvaluator> evaluator, int64_t obs_size,
                  std::vector<PpoTransition>* trajectory,
                  std::atomic<uint64_t>* total_env_steps,
                  float reward_lambda,
                  WorkerStats* local_stats) {

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

    std::vector<int> current_tleilaxu(game.NumPlayers(), 0);
    std::vector<bool> had_swordmaster(game.NumPlayers(), false);
    std::vector<int> prev_conflict_vp(game.NumPlayers(), 0);
    std::vector<int> prev_total_vp(game.NumPlayers(), 0);
    if (dune_state != nullptr) {
      for (int p = 0; p < game.NumPlayers(); ++p) {
        current_tleilaxu[p] = dune_state->GetTleilaxuTrackForTesting(p);
        had_swordmaster[p] = dune_state->HasSwordmaster(p);
        prev_conflict_vp[p] = dune_state->ConflictVpDelta(p);
        prev_total_vp[p] = dune_state->GetPlayerVp(p);
      }
    }

    CombatCreditAccumulator combat_accumulator;
    std::vector<int> last_transition_index(game.NumPlayers(), -1);
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

      PpoTransition transition;
      transition.state = std::move(obs);
      transition.legal_actions = std::move(actions);
      transition.action = action;
      transition.old_log_prob = old_log_prob;
      transition.reward = 0.0f;
      transition.value = result.value;
      transition.advantage = 0.0f;
      transition.return_value = 0.0f;
      transition.player_id = current_player;
      transition.episode_id = episode_id;
      trajectory->push_back(std::move(transition));

      if (current_player >= 0 && current_player < game.NumPlayers()) {
        last_transition_index[current_player] = static_cast<int>(trajectory->size() - 1);
      }

      if (dune_state != nullptr) {
        if (current_player >= 0 && current_player < game.NumPlayers() &&
            action != dune_imperium::kActionCombatPass) {
          int post_strength = dune_imperium::CombatStrength(*dune_state, current_player);
          int strength_delta = post_strength - pre_combat_strength[current_player];
          if (strength_delta > 0) {
            combat_accumulator.RecordDeployment(current_player, trajectory->size() - 1, strength_delta);
          }
        }

        for (int p = 0; p < game.NumPlayers(); ++p) {
          int new_tleilaxu = dune_state->GetTleilaxuTrackForTesting(p);
          int tleilaxu_delta = new_tleilaxu - current_tleilaxu[p];
          current_tleilaxu[p] = new_tleilaxu;
          if (tleilaxu_delta > 0 &&
              (tleilaxu_breadcrumb_weight > 0.0 ||
               tleilaxu_level7_breadcrumb_weight > 0.0)) {
            float tleilaxu_reward = 0.0f;
            int start_l = new_tleilaxu - tleilaxu_delta;
            for (int l = start_l + 1; l <= new_tleilaxu; ++l) {
              if (l >= 1 && l <= 6) {
                tleilaxu_reward += static_cast<float>(tleilaxu_breadcrumb_weight) * reward_lambda;
              } else if (l == 7) {
                tleilaxu_reward += static_cast<float>(tleilaxu_level7_breadcrumb_weight) * reward_lambda;
              }
            }
            int idx = last_transition_index[p];
            if (idx >= 0 && idx < static_cast<int>(trajectory->size())) {
              (*trajectory)[idx].reward += tleilaxu_reward;
            }
          }
        }

        for (int p = 0; p < game.NumPlayers(); ++p) {
          int raw_conflict_vp_delta = dune_state->ConflictVpDelta(p) - prev_conflict_vp[p];
          prev_conflict_vp[p] = dune_state->ConflictVpDelta(p);

          int raw_total_vp_delta = dune_state->GetPlayerVp(p) - prev_total_vp[p];
          prev_total_vp[p] = dune_state->GetPlayerVp(p);

          int raw_noncombat = raw_total_vp_delta - raw_conflict_vp_delta;

          local_stats->raw_conflict_vp += raw_conflict_vp_delta;
          local_stats->raw_noncombat_vp += raw_noncombat;
          local_stats->raw_total_vp += raw_total_vp_delta;

          float combat_shape = std::max(raw_conflict_vp_delta, 0)
                                * static_cast<float>(shaped_weight) * reward_lambda;
          float noncombat_shape = raw_noncombat
                                  * static_cast<float>(shaped_weight) * reward_lambda;

          if (noncombat_shape != 0.0f) {
            int idx = last_transition_index[p];
            if (idx >= 0 && idx < static_cast<int>(trajectory->size())) {
              (*trajectory)[idx].reward += noncombat_shape;
            }
          }

          if (combat_shape > 0.0f) {
            local_stats->conflict_vp_generated += combat_shape;
            int total_investment = combat_accumulator.GetTotalInvestment(p);
            if (total_investment > 0) {
              for (const auto& ev : combat_accumulator.GetEvents(p)) {
                if (ev.transition_index >= 0 && ev.transition_index < static_cast<int>(trajectory->size())) {
                  float attributed_portion = combat_shape * static_cast<float>(ev.strength_delta) /
                                             static_cast<float>(total_investment);
                  (*trajectory)[ev.transition_index].reward += attributed_portion;
                  local_stats->conflict_vp_attributed += attributed_portion;
                }
              }
            } else {
              local_stats->conflict_vp_unattributed += combat_shape;
            }
          }
        }

        if (pre_action_phase == dune_imperium::GamePhase::kCombat &&
            dune_state->phase() != dune_imperium::GamePhase::kCombat) {
          combat_accumulator.ClearAll();
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
                    std::shared_ptr<IGameEvaluator> evaluator, int64_t obs_size,
                    PpoRolloutBuffer* rollout_buffer,
                    std::atomic<bool>* stop_collection,
                    std::atomic<uint64_t>* total_env_steps,
                    std::vector<WorkerStats>* worker_stats,
                    std::atomic<uint64_t>* local_episode_id,
                    uint64_t start_episode_id,
                    int rollout_games,
                    float reward_lambda) {
  uint64_t master = absl::GetFlag(FLAGS_seed);
  WorkerStats local_stats;
  while (true) {
    if (rollout_games == 0 && stop_collection->load(std::memory_order_relaxed)) {
      break;
    }
    uint64_t episode_id = local_episode_id->fetch_add(1, std::memory_order_relaxed);
    if (rollout_games > 0 && episode_id >= start_episode_id + rollout_games) {
      break;
    }
    std::vector<PpoTransition> trajectory;
    int moves = PpoSimulation(master, episode_id, *game, evaluator, obs_size, &trajectory,
                              total_env_steps, reward_lambda, &local_stats);
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

  double conflict_vp_generated = 0.0;
  double conflict_vp_attributed = 0.0;
  double conflict_vp_unattributed = 0.0;

  bool episode_ids_unique = true;
  double raw_conflict_vp = 0.0;
  double raw_noncombat_vp = 0.0;
  double raw_total_vp = 0.0;
};



CollectResult CollectRollout(const Game* game,
                             std::shared_ptr<IGameEvaluator> evaluator,
                             int64_t obs_size,
                             std::atomic<uint64_t>* total_env_steps,
                             int num_threads,
                             std::atomic<uint64_t>* next_episode_id,
                             int rollout_games,
                             float reward_lambda) {
  CollectResult result;
  PpoRolloutBuffer rollout_buffer;
  std::atomic<bool> stop_collection{false};
  std::vector<WorkerStats> worker_stats(num_threads);
  std::vector<std::thread> workers;
  workers.reserve(num_threads);

  uint64_t actual_start_ep = next_episode_id->load();
  std::atomic<uint64_t> local_episode_id{actual_start_ep};

  auto start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < num_threads; ++i) {
    workers.emplace_back(RolloutWorker, i, game, evaluator, obs_size,
                         &rollout_buffer, &stop_collection, total_env_steps,
                         &worker_stats, &local_episode_id, actual_start_ep, rollout_games, reward_lambda);
  }
  for (auto& worker : workers) worker.join();
  
  if (rollout_games > 0) {
    next_episode_id->store(actual_start_ep + rollout_games);
  } else {
    next_episode_id->store(local_episode_id.load());
  }
  auto end = std::chrono::high_resolution_clock::now();

  bool episode_ids_unique = true;
  result.rollout = rollout_buffer.TakeAll(&episode_ids_unique);
  result.episode_ids_unique = episode_ids_unique;

  for (const auto& stats : worker_stats) {
    result.games += stats.games;
    result.moves += stats.moves;

    for (int i = 0; i < 4; ++i) result.sm_acquisitions_by_seat[i] += stats.sm_acquisitions_by_seat[i];
    for (int i = 0; i < 14; ++i) result.sm_acquisitions_by_leader[i] += stats.sm_acquisitions_by_leader[i];
    for (int i = 0; i < 11; ++i) result.sm_acquisitions_by_round[i] += stats.sm_acquisitions_by_round[i];

    result.conflict_vp_generated += stats.conflict_vp_generated;
    result.conflict_vp_attributed += stats.conflict_vp_attributed;
    result.conflict_vp_unattributed += stats.conflict_vp_unattributed;

    result.raw_conflict_vp += stats.raw_conflict_vp;
    result.raw_noncombat_vp += stats.raw_noncombat_vp;
    result.raw_total_vp += stats.raw_total_vp;
  }

  // Verify shaped reward conservation invariant
  if (std::abs(result.conflict_vp_generated - (result.conflict_vp_attributed + result.conflict_vp_unattributed)) > 1e-4) {
    open_spiel::SpielFatalError(absl::StrFormat("Shaped reward conservation violated: generated = %f, attributed = %f, unattributed = %f",
                                                result.conflict_vp_generated, result.conflict_vp_attributed, result.conflict_vp_unattributed));
  }

  // Verify signed raw VP conservation invariant
  if (std::abs(result.raw_conflict_vp + result.raw_noncombat_vp - result.raw_total_vp) > 1e-4) {
    open_spiel::SpielFatalError(absl::StrFormat("Signed raw VP conservation violated: raw_conflict = %f, raw_noncombat = %f, raw_total = %f",
                                                result.raw_conflict_vp, result.raw_noncombat_vp, result.raw_total_vp));
  }
  result.elapsed_seconds =
      std::chrono::duration<double>(end - start).count();
  return result;
}


#endif  // OPEN_SPIEL_BUILD_WITH_LIBTORCH

}  // namespace
}  // namespace open_spiel

int main(int argc, char** argv) {
  setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8", 1);
  absl::ParseCommandLine(argc, argv);
  if (absl::GetFlag(FLAGS_deterministic)) {
    at::globalContext().setDeterministicAlgorithms(true, /*silent=*/true);
  }
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

  std::string init_mode = absl::GetFlag(FLAGS_init_mode);
  if (init_mode.empty()) {
    SpielFatalError("Required flag --init_mode is missing. Must be 'random', 'checkpoint', 'bootstrap', or 'validate_legacy'.");
  }

  std::string config_fingerprint = open_spiel::ComputeConfigFingerprint();
  std::string search_label_fingerprint = open_spiel::GetSearchLabelFingerprint(absl::GetFlag(FLAGS_search_label_dir));
  std::string run_uuid = open_spiel::GenerateUUID();

  std::atomic<uint64_t> total_env_steps{0};
  std::atomic<uint64_t> next_episode_id{0};
  int start_update = 1;
  int target_end_update = absl::GetFlag(FLAGS_target_end_update);

  if (init_mode == "random") {
    std::string model_path = absl::GetFlag(FLAGS_model_checkpoint);
    std::string optim_path = absl::GetFlag(FLAGS_optim_checkpoint);
    std::filesystem::path m_path = model_path;
    m_path.replace_extension(".json");
    std::string manifest_path = m_path.string();
    if (std::filesystem::exists(model_path) || std::filesystem::exists(optim_path) || std::filesystem::exists(manifest_path)) {
      SpielFatalError("init_mode=random but checkpoint or manifest file already exists. Refusing to overwrite.");
    }
    std::cout << "Starting fresh training run (init_mode=random).\n";
  } else if (init_mode == "checkpoint") {
    if (absl::GetFlag(FLAGS_pipeline)) {
      SpielFatalError("init_mode=checkpoint with pipeline=true is rejected because prefetched rollout is not persisted.");
    }
    std::string model_path = absl::GetFlag(FLAGS_model_checkpoint);
    std::string optim_path = absl::GetFlag(FLAGS_optim_checkpoint);
    std::filesystem::path m_path = model_path;
    m_path.replace_extension(".json");
    std::string manifest_path = m_path.string();

    CheckpointManifest manifest;
    std::string err;
    std::string legacy_fingerprint = ComputeLegacyConfigFingerprint();
    if (!ParseAndValidateManifest(manifest_path, model_path, optim_path,
                                  master, target_end_update,
                                  absl::GetFlag(FLAGS_seed_scheme_version),
                                  config_fingerprint, search_label_fingerprint,
                                  absl::GetFlag(FLAGS_hidden_dim),
                                  absl::GetFlag(FLAGS_num_blocks),
                                  manifest, err, legacy_fingerprint)) {
      SpielFatalError(err);
    }

    int global_update = manifest.global_update;
    if (global_update >= target_end_update) {
      SpielFatalError(absl::StrFormat(
          "global_update (%d) >= target_end_update (%d). "
          "Resuming is not possible as training is already finished.",
          global_update, target_end_update));
    }
    uint64_t manifest_env_steps = manifest.total_env_steps;
    uint64_t manifest_episode_id = manifest.next_episode_id;
    std::string manifest_run_uuid = manifest.run_uuid;

    try {
      torch::load(training_model, model_path, device);
      torch::load(*optimizer, optim_path, device);
    } catch (const c10::Error& e) {
      SpielFatalError("LibTorch load failed: " + std::string(e.msg()));
    }

    open_spiel::SetOptimizerLearningRate(*optimizer, absl::GetFlag(FLAGS_learning_rate));
    for (size_t g = 0; g < optimizer->param_groups().size(); ++g) {
      auto& options =
          static_cast<torch::optim::AdamWOptions&>(
              optimizer->param_groups()[g].options());
      options.weight_decay(g == 0 ? absl::GetFlag(FLAGS_policy_weight_decay)
                                  : absl::GetFlag(FLAGS_weight_decay));
    }
    for (auto& pair : optimizer->state()) {
      if (auto* state =
              dynamic_cast<torch::optim::AdamWParamState*>(pair.second.get())) {
        if (state->exp_avg().defined()) state->exp_avg() = state->exp_avg().to(device);
        if (state->exp_avg_sq().defined()) state->exp_avg_sq() = state->exp_avg_sq().to(device);
        if (state->max_exp_avg_sq().defined()) {
          state->max_exp_avg_sq() = state->max_exp_avg_sq().to(device);
        }
      }
    }

    start_update = global_update + 1;
    total_env_steps.store(manifest_env_steps);
    next_episode_id.store(manifest_episode_id);
    run_uuid = manifest_run_uuid;

    std::cout << "Successfully verified and loaded checkpoint manifest. Resuming from update "
              << start_update << " to " << target_end_update << ".\n";
  } else if (init_mode == "bootstrap") {
    std::string model_path = absl::GetFlag(FLAGS_model_checkpoint);
    std::string optim_path = absl::GetFlag(FLAGS_optim_checkpoint);
    if (model_path.find("artifacts/baselines") != std::string::npos) {
      SpielFatalError("Refusing to bootstrap directly inside the frozen baseline directory: " + model_path + 
                      "\nRun bootstrap beside a BRANCH COPY in a separate output directory.");
    }
    if (!std::filesystem::exists(model_path)) {
      SpielFatalError("Model file not found for bootstrap: " + model_path);
    }
    if (!std::filesystem::exists(optim_path)) {
      SpielFatalError("Optimizer file not found for bootstrap: " + optim_path);
    }

    try {
      torch::load(training_model, model_path, device);
      torch::load(*optimizer, optim_path, device);
    } catch (const c10::Error& e) {
      SpielFatalError("LibTorch load failed during bootstrap: " + std::string(e.msg()));
    }

    size_t model_size = 0;
    std::string model_hash = open_spiel::ComputeFileSHA256(model_path, &model_size);
    size_t optim_size = 0;
    std::string optim_hash = open_spiel::ComputeFileSHA256(optim_path, &optim_size);

    std::filesystem::path m_path = model_path;
    m_path.replace_extension(".json");
    std::string manifest_path = m_path.string();
    std::string manifest_tmp = manifest_path + ".tmp";

    json::Object manifest_obj;
    manifest_obj["schema_version"] = static_cast<int64_t>(2);
    manifest_obj["global_update"] = static_cast<int64_t>(absl::GetFlag(FLAGS_start_update));
    manifest_obj["target_end_update"] = static_cast<int64_t>(absl::GetFlag(FLAGS_target_end_update));
    manifest_obj["total_env_steps"] = static_cast<int64_t>(absl::GetFlag(FLAGS_start_env_steps));
    manifest_obj["next_episode_id"] = static_cast<int64_t>(absl::GetFlag(FLAGS_start_episode_id));
    manifest_obj["base_seed"] = static_cast<int64_t>(absl::GetFlag(FLAGS_seed));
    manifest_obj["seed_scheme_version"] = static_cast<int64_t>(absl::GetFlag(FLAGS_seed_scheme_version));
    manifest_obj["config_fingerprint"] = config_fingerprint;
    manifest_obj["search_label_fingerprint"] = search_label_fingerprint;
    manifest_obj["run_uuid"] = open_spiel::GenerateUUID();
    manifest_obj["checkpoint_uuid"] = open_spiel::GenerateUUID();
    manifest_obj["model_filename"] = std::filesystem::path(model_path).filename().string();
    manifest_obj["model_file_size"] = static_cast<int64_t>(model_size);
    manifest_obj["model_sha256"] = model_hash;
    manifest_obj["optimizer_filename"] = std::filesystem::path(optim_path).filename().string();
    manifest_obj["optimizer_file_size"] = static_cast<int64_t>(optim_size);
    manifest_obj["optimizer_sha256"] = optim_hash;
    manifest_obj["hidden_dim"] = static_cast<int64_t>(absl::GetFlag(FLAGS_hidden_dim));
    manifest_obj["num_blocks"] = static_cast<int64_t>(absl::GetFlag(FLAGS_num_blocks));
    manifest_obj["legacy_migration_provenance"] = "Synthesized via init_mode=bootstrap";

    {
      std::ofstream ofs(manifest_tmp);
      if (!ofs) {
        SpielFatalError("Could not open manifest temp file for writing: " + manifest_tmp);
      }
      ofs << json::ToString(manifest_obj, true);
    }
    std::filesystem::rename(manifest_tmp, manifest_path);
    std::cout << "Successfully bootstrapped manifest at " << manifest_path << "\n";
    exit(0);
  } else if (init_mode == "validate_legacy") {
    std::string model_path = absl::GetFlag(FLAGS_model_checkpoint);
    std::string optim_path = absl::GetFlag(FLAGS_optim_checkpoint);
    std::filesystem::path manifest_path;
    std::string artifact_manifest_flag = absl::GetFlag(FLAGS_artifact_manifest);
    if (!artifact_manifest_flag.empty()) {
      manifest_path = artifact_manifest_flag;
    } else {
      manifest_path = std::filesystem::path(model_path).parent_path() / "manifest.json";
    }

    if (!std::filesystem::exists(manifest_path)) {
      SpielFatalError("init_mode=validate_legacy but Phase 1 manifest file not found at: " + manifest_path.string());
    }

    std::ifstream ifs(manifest_path.string());
    if (!ifs) {
      SpielFatalError("Could not open manifest file at: " + manifest_path.string());
    }
    std::string content((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
    auto val_opt = open_spiel::json::FromString(content);
    if (!val_opt.has_value() || !val_opt->IsObject()) {
      SpielFatalError("Manifest file is malformed JSON at: " + manifest_path.string());
    }
    const auto& manifest_obj = val_opt->GetObject();

    auto get_object_field = [&](const std::string& key) -> const json::Object& {
      auto it = manifest_obj.find(key);
      if (it == manifest_obj.end() || !it->second.IsObject()) {
        SpielFatalError("Manifest missing required object field: " + key);
      }
      return it->second.GetObject();
    };

    auto get_nested_string = [](const json::Object& obj, const std::string& key) -> std::string {
      auto it = obj.find(key);
      if (it == obj.end() || !it->second.IsString()) {
        SpielFatalError("Nested object missing required string field: " + key);
      }
      return it->second.GetString();
    };

    auto get_nested_int = [](const json::Object& obj, const std::string& key) -> int64_t {
      auto it = obj.find(key);
      if (it == obj.end() || !it->second.IsInt()) {
        SpielFatalError("Nested object missing required int field: " + key);
      }
      return it->second.GetInt();
    };

    const auto& model_obj = get_object_field("model");
    const auto& optim_obj = get_object_field("optimizer");
    const auto& arch_obj = get_object_field("architecture");

    std::string expected_model_sha256 = get_nested_string(model_obj, "sha256");
    size_t expected_model_size = get_nested_int(model_obj, "size_bytes");

    std::string expected_optim_sha256 = get_nested_string(optim_obj, "sha256");
    size_t expected_optim_size = get_nested_int(optim_obj, "size_bytes");

    int64_t manifest_hidden_dim = get_nested_int(arch_obj, "hidden_dim");
    int64_t manifest_num_blocks = get_nested_int(arch_obj, "num_blocks");
    int64_t manifest_obs_size = get_nested_int(arch_obj, "observation_size");
    int64_t manifest_action_size = get_nested_int(arch_obj, "action_size");

    if (!std::filesystem::exists(model_path)) {
      SpielFatalError("Model file not found: " + model_path);
    }
    if (!std::filesystem::exists(optim_path)) {
      SpielFatalError("Optimizer file not found: " + optim_path);
    }

    size_t actual_model_size = 0;
    std::string actual_model_hash = open_spiel::ComputeFileSHA256(model_path, &actual_model_size);
    if (actual_model_size != expected_model_size) {
      SpielFatalError(absl::StrFormat("Model file size mismatch. Manifest: %d, Actual: %d", expected_model_size, actual_model_size));
    }
    if (actual_model_hash != expected_model_sha256) {
      SpielFatalError("Model SHA-256 hash mismatch. Manifest: " + expected_model_sha256 + ", Actual: " + actual_model_hash);
    }

    size_t actual_optim_size = 0;
    std::string actual_optim_hash = open_spiel::ComputeFileSHA256(optim_path, &actual_optim_size);
    if (actual_optim_size != expected_optim_size) {
      SpielFatalError(absl::StrFormat("Optimizer file size mismatch. Manifest: %d, Actual: %d", expected_optim_size, actual_optim_size));
    }
    if (actual_optim_hash != expected_optim_sha256) {
      SpielFatalError("Optimizer SHA-256 hash mismatch. Manifest: " + expected_optim_sha256 + ", Actual: " + actual_optim_hash);
    }

    if (absl::GetFlag(FLAGS_hidden_dim) != manifest_hidden_dim) {
      SpielFatalError(absl::StrFormat("hidden_dim mismatch. Flags: %d, Manifest: %d", absl::GetFlag(FLAGS_hidden_dim), manifest_hidden_dim));
    }
    if (absl::GetFlag(FLAGS_num_blocks) != manifest_num_blocks) {
      SpielFatalError(absl::StrFormat("num_blocks mismatch. Flags: %d, Manifest: %d", absl::GetFlag(FLAGS_num_blocks), manifest_num_blocks));
    }
    if (obs_size != manifest_obs_size) {
      SpielFatalError(absl::StrFormat("observation_size mismatch. Game: %d, Manifest: %d", obs_size, manifest_obs_size));
    }
    if (action_size != manifest_action_size) {
      SpielFatalError(absl::StrFormat("action_size mismatch. Game: %d, Manifest: %d", action_size, manifest_action_size));
    }

    try {
      torch::load(training_model, model_path, device);
    } catch (const c10::Error& e) {
      SpielFatalError("LibTorch load failed: " + std::string(e.msg()));
    }

    int64_t actual_param_count = 0;
    for (const auto& param : training_model->parameters()) {
      actual_param_count += param.numel();
    }

    auto dummy_model = std::make_shared<open_spiel::SharedDunePolicyValueNetImpl>(
        manifest_obs_size, manifest_hidden_dim, manifest_action_size, manifest_num_blocks);
    int64_t expected_param_count = 0;
    for (const auto& param : dummy_model->parameters()) {
      expected_param_count += param.numel();
    }

    if (actual_param_count != expected_param_count) {
      SpielFatalError(absl::StrFormat("Parameter count mismatch. Expected: %d (calculated from architecture), Actual: %d", expected_param_count, actual_param_count));
    }

    std::cout << absl::StrFormat("Legacy checkpoint validation successful!\n"
                                 "  Model parameter count: %d\n"
                                 "  Model size and hash match Phase 1 manifest.\n"
                                 "  Optimizer size and hash match Phase 1 manifest.\n", actual_param_count);
    exit(0);
  } else {
    SpielFatalError("Unsupported init_mode: " + init_mode);
  }

  std::shared_mutex sync_mutex;
  std::mutex eval_mutex;
  open_spiel::SyncModels(training_model, inference_model, &sync_mutex);

  std::shared_ptr<open_spiel::IGameEvaluator> evaluator;
  if (absl::GetFlag(FLAGS_deterministic_rollout_eval)) {
    evaluator = std::make_shared<open_spiel::DeterministicEvaluator>(
        inference_model, device, &eval_mutex, &sync_mutex);
  } else {
    evaluator = std::make_shared<open_spiel::BatchedEvaluator>(
        inference_model, absl::GetFlag(FLAGS_eval_batch_size),
        absl::GetFlag(FLAGS_eval_timeout_ms), device, &sync_mutex, 0.0f,
        absl::GetFlag(FLAGS_evaluator_device_synchronize));
  }

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

  uint64_t total_games = 0;
  uint64_t total_moves = 0;
  auto training_start = std::chrono::high_resolution_clock::now();

  double base_lr = absl::GetFlag(FLAGS_learning_rate);
  bool pipeline = absl::GetFlag(FLAGS_pipeline);
  int num_threads = absl::GetFlag(FLAGS_threads);

  open_spiel::SearchLabelBuffer search_buffer;
  search_buffer.SetExpectedDimensions(obs_size, action_size);
  std::string search_label_dir = absl::GetFlag(FLAGS_search_label_dir);
  if (!search_label_dir.empty()) {
    search_buffer.LoadFromDirectory(search_label_dir);
  }

  int rollout_games = absl::GetFlag(FLAGS_rollout_games);

  // Collect first rollout synchronously.
  float reward_lambda = ComputeRewardLambda(total_env_steps.load(),
                                            absl::GetFlag(FLAGS_shaping_start_env_steps),
                                            absl::GetFlag(FLAGS_shaping_decay_env_steps));
  open_spiel::CollectResult current_collect = open_spiel::CollectRollout(
      game.get(), evaluator, obs_size, &total_env_steps, num_threads, &next_episode_id,
      rollout_games, reward_lambda);
  if (absl::GetFlag(FLAGS_diagnostics_only)) {
    open_spiel::PpoUpdateStats stats =
        open_spiel::TrainPpoUpdate(training_model, *optimizer, current_collect.rollout,
                                   obs_size, action_size, device, master, start_update);
    stats.episode_ids_unique = current_collect.episode_ids_unique;
    std::string diagnostics_path = absl::GetFlag(FLAGS_diagnostics_path);
    if (!diagnostics_path.empty()) {
      double val_kl = search_buffer.ComputeValidationKL(training_model, device, static_cast<float>(absl::GetFlag(FLAGS_logit_cap)));
      open_spiel::WriteDiagnostics(diagnostics_path, start_update, stats,
                                   current_collect.conflict_vp_generated,
                                   current_collect.conflict_vp_attributed,
                                   current_collect.conflict_vp_unattributed,
                                   master, run_uuid, absl::GetFlag(FLAGS_run_prefix), config_fingerprint,
                                   current_collect.raw_conflict_vp,
                                   current_collect.raw_noncombat_vp,
                                   current_collect.raw_total_vp,
                                   val_kl);
    }
    std::cout << "Diagnostics-only run complete. Exiting.\n";
    exit(0);
  }

  total_games += current_collect.games;
  total_moves += current_collect.moves;

  for (int update = start_update; update <= target_end_update; ++update) {
    if (absl::GetFlag(FLAGS_anneal_lr)) {
      double frac = 1.0 - static_cast<double>(update - 1) /
                              std::max(1, target_end_update);
      open_spiel::SetOptimizerLearningRate(*optimizer,
                                           std::max(1e-8, frac * base_lr));
    }

    auto wall_start = std::chrono::high_resolution_clock::now();

    // Launch background collection for the next update (pipelined).
    // The inference model is frozen during PPO training, so the background
    // collection safely uses the current policy snapshot.
    open_spiel::CollectResult next_collect;
    std::thread bg_collect_thread;
    bool have_bg = pipeline && update < target_end_update;
    if (have_bg) {
      float next_reward_lambda = ComputeRewardLambda(total_env_steps.load(),
                                                     absl::GetFlag(FLAGS_shaping_start_env_steps),
                                                     absl::GetFlag(FLAGS_shaping_decay_env_steps));
      bg_collect_thread = std::thread([&, next_reward_lambda]() {
        next_collect = open_spiel::CollectRollout(
            game.get(), evaluator, obs_size, &total_env_steps, num_threads, &next_episode_id,
            rollout_games, next_reward_lambda);
      });
    }

    auto ppo_start = std::chrono::high_resolution_clock::now();
    open_spiel::PpoUpdateStats stats =
        open_spiel::TrainPpoUpdate(training_model, *optimizer, current_collect.rollout,
                                   obs_size, action_size, device, master, update);
    stats.episode_ids_unique = current_collect.episode_ids_unique;

    double ppo_elapsed = std::chrono::duration<double>(
        std::chrono::high_resolution_clock::now() - ppo_start).count();

    // Log PPO updates to diagnostics CSV
    std::string diagnostics_path = absl::GetFlag(FLAGS_diagnostics_path);
    if (!diagnostics_path.empty()) {
      double val_kl = search_buffer.ComputeValidationKL(training_model, device, static_cast<float>(absl::GetFlag(FLAGS_logit_cap)));
      open_spiel::WriteDiagnostics(diagnostics_path, update, stats,
                                   current_collect.conflict_vp_generated,
                                   current_collect.conflict_vp_attributed,
                                   current_collect.conflict_vp_unattributed,
                                   master, run_uuid, absl::GetFlag(FLAGS_run_prefix), config_fingerprint,
                                   current_collect.raw_conflict_vp,
                                   current_collect.raw_noncombat_vp,
                                   current_collect.raw_total_vp,
                                   val_kl);
    }

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
        update, target_end_update, current_collect.rollout.size(),
        static_cast<unsigned long long>(total_games),
        collect_elapsed, sps, ppo_elapsed, wall_elapsed,
        stats.policy_loss, stats.value_loss, stats.entropy, stats.approx_kl,
        stats.clip_fraction, stats.explained_variance,
        stats.early_stopped ? " | early-stop" : "");



    std::cout << absl::StrFormat(
        "  [Conflict VP Stats] Generated: %.2f | Attributed: %.2f | Unattributed: %.2f\n",
        current_collect.conflict_vp_generated,
        current_collect.conflict_vp_attributed,
        current_collect.conflict_vp_unattributed);

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
                                 optim_path, update, target_end_update,
                                 total_env_steps.load(), next_episode_id.load(),
                                 master, absl::GetFlag(FLAGS_seed_scheme_version),
                                 config_fingerprint, search_label_fingerprint,
                                 run_uuid);
    }

    // Advance to next rollout.
    if (have_bg) {
      current_collect = std::move(next_collect);
    } else if (update < target_end_update) {
      float next_reward_lambda = ComputeRewardLambda(total_env_steps.load(),
                                                     absl::GetFlag(FLAGS_shaping_start_env_steps),
                                                     absl::GetFlag(FLAGS_shaping_decay_env_steps));
      current_collect = open_spiel::CollectRollout(
          game.get(), evaluator, obs_size, &total_env_steps, num_threads, &next_episode_id,
          rollout_games, next_reward_lambda);
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
                               absl::GetFlag(FLAGS_optim_checkpoint),
                               target_end_update, target_end_update,
                               total_env_steps.load(), next_episode_id.load(),
                               master, absl::GetFlag(FLAGS_seed_scheme_version),
                               config_fingerprint, search_label_fingerprint,
                               run_uuid);
  }
  return 0;
#endif
}
