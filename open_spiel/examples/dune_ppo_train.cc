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
#include "open_spiel/games/dune_imperium/dune_imperium_util.h"
#include "open_spiel/spiel.h"

#ifdef OPEN_SPIEL_BUILD_WITH_LIBTORCH
#include <torch/torch.h>

#include "dune_network.h"
#endif

ABSL_FLAG(std::string, game, "dune_imperium", "The OpenSpiel game to train.");
ABSL_FLAG(int, threads, 64, "Rollout worker threads.");
ABSL_FLAG(int, total_updates, 1000, "Number of PPO collect/update cycles.");
ABSL_FLAG(int, rollout_transitions, 32768,
          "Minimum on-policy transitions collected before each PPO update.");
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
};

class PpoRolloutBuffer {
 public:
  size_t PushTrajectory(std::vector<PpoTransition>&& trajectory) {
    std::lock_guard<std::mutex> lock(mu_);
    for (auto& transition : trajectory) {
      transitions_.push_back(std::move(transition));
    }
    return transitions_.size();
  }

  size_t Size() const {
    std::lock_guard<std::mutex> lock(mu_);
    return transitions_.size();
  }

  std::vector<PpoTransition> TakeAll() {
    std::lock_guard<std::mutex> lock(mu_);
    std::vector<PpoTransition> out;
    out.swap(transitions_);
    return out;
  }

 private:
  mutable std::mutex mu_;
  std::vector<PpoTransition> transitions_;
};

struct WorkerStats {
  uint64_t games = 0;
  uint64_t moves = 0;
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

std::pair<Action, float> SamplePolicyAction(
    std::mt19937* rng, const std::vector<float>& logits,
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

int PpoSimulation(std::mt19937* rng, const Game& game,
                  std::shared_ptr<BatchedEvaluator> evaluator, int64_t obs_size,
                  std::vector<PpoTransition>* trajectory,
                  std::atomic<uint64_t>* total_env_steps) {
  std::unique_ptr<State> state = game.NewInitialState();
  bool provides_info_state_tensor =
      game.GetType().provides_information_state_tensor;
  bool provides_observations_tensor =
      game.GetType().provides_observation_tensor;

  const auto* dune_state =
      dynamic_cast<const dune_imperium::DuneImperiumState*>(state.get());
  std::vector<int> current_vps(game.NumPlayers(), 0);
  std::vector<int> current_tleilaxu(game.NumPlayers(), 0);
  if (dune_state != nullptr) {
    for (int p = 0; p < game.NumPlayers(); ++p) {
      current_vps[p] = dune_state->GetPlayerVpForTesting(p);
      current_tleilaxu[p] = dune_state->GetTleilaxuTrackForTesting(p);
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
  int decay_horizon = std::max(1, absl::GetFlag(FLAGS_decay_horizon));

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
      Action action = game.GetType().chance_mode ==
                              GameType::ChanceMode::kSampledStochastic
                          ? outcomes.front().first
                          : SampleAction(outcomes, *rng).first;
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
          joint_action.push_back(actions[dis(*rng)]);
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
    auto [action, old_log_prob] = SamplePolicyAction(rng, logits, actions);

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

    std::fill(shaped_bonus_by_player.begin(), shaped_bonus_by_player.end(),
              0.0f);
    if (dune_state != nullptr) {
      uint64_t env_steps = total_env_steps->load(std::memory_order_relaxed);
      float reward_lambda =
          env_steps < static_cast<uint64_t>(decay_horizon)
              ? 1.0f - static_cast<float>(env_steps) / decay_horizon
              : 0.0f;

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

void RolloutWorker(int thread_id, const Game* game,
                   std::shared_ptr<BatchedEvaluator> evaluator, int64_t obs_size,
                   PpoRolloutBuffer* rollout_buffer,
                   std::atomic<bool>* stop_collection,
                   std::atomic<uint64_t>* total_env_steps,
                   std::vector<WorkerStats>* worker_stats) {
  std::mt19937 rng(absl::GetFlag(FLAGS_seed) + 9973 * thread_id);
  WorkerStats local_stats;
  while (!stop_collection->load(std::memory_order_relaxed)) {
    std::vector<PpoTransition> trajectory;
    int moves = PpoSimulation(&rng, *game, evaluator, obs_size, &trajectory,
                              total_env_steps);
    local_stats.games += 1;
    local_stats.moves += moves;
    size_t size = rollout_buffer->PushTrajectory(std::move(trajectory));
    if (size >= static_cast<size_t>(absl::GetFlag(FLAGS_rollout_transitions))) {
      stop_collection->store(true, std::memory_order_relaxed);
    }
  }
  (*worker_stats)[thread_id] = local_stats;
}

struct CollectResult {
  std::vector<PpoTransition> rollout;
  uint64_t games = 0;
  uint64_t moves = 0;
  double elapsed_seconds = 0.0;
};

CollectResult CollectRollout(const Game* game,
                             std::shared_ptr<BatchedEvaluator> evaluator,
                             int64_t obs_size,
                             std::atomic<uint64_t>* total_env_steps,
                             int num_threads) {
  CollectResult result;
  PpoRolloutBuffer rollout_buffer;
  std::atomic<bool> stop_collection{false};
  std::vector<WorkerStats> worker_stats(num_threads);
  std::vector<std::thread> workers;
  workers.reserve(num_threads);

  auto start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < num_threads; ++i) {
    workers.emplace_back(RolloutWorker, i, game, evaluator, obs_size,
                         &rollout_buffer, &stop_collection, total_env_steps,
                         &worker_stats);
  }
  for (auto& worker : workers) worker.join();
  auto end = std::chrono::high_resolution_clock::now();

  result.rollout = rollout_buffer.TakeAll();
  for (const auto& stats : worker_stats) {
    result.games += stats.games;
    result.moves += stats.moves;
  }
  result.elapsed_seconds =
      std::chrono::duration<double>(end - start).count();
  return result;
}

PpoUpdateStats TrainPpoUpdate(
    std::shared_ptr<SharedDunePolicyValueNetImpl> model,
    torch::optim::AdamW& optimizer, std::vector<PpoTransition>& batch,
    int64_t obs_size, int64_t action_dim, torch::Device device) {
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
    torch::Tensor permutation =
        torch::randperm(n, torch::TensorOptions().device(device).dtype(torch::kInt64));
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

#ifndef OPEN_SPIEL_BUILD_WITH_LIBTORCH
  std::cerr << "dune_ppo_train requires OPEN_SPIEL_BUILD_WITH_LIBTORCH.\n";
  return 1;
#else
  at::set_num_threads(1);
  at::set_num_interop_threads(1);

  torch::manual_seed(absl::GetFlag(FLAGS_seed));

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
      "rollout_transitions=%d minibatch=%d epochs=%d clip=%.3f vf_coef=%.3f "
      "ent_coef=%.4f gamma=%.3f gae_lambda=%.3f\n",
      obs_size, action_size, absl::GetFlag(FLAGS_hidden_dim),
      absl::GetFlag(FLAGS_num_blocks),
      absl::GetFlag(FLAGS_rollout_transitions),
      absl::GetFlag(FLAGS_ppo_minibatch_size),
      absl::GetFlag(FLAGS_ppo_update_epochs),
      absl::GetFlag(FLAGS_ppo_clip_epsilon),
      absl::GetFlag(FLAGS_value_coef), absl::GetFlag(FLAGS_entropy_coef),
      absl::GetFlag(FLAGS_gamma), absl::GetFlag(FLAGS_gae_lambda));

  std::atomic<uint64_t> total_env_steps{0};
  uint64_t total_games = 0;
  uint64_t total_moves = 0;
  auto training_start = std::chrono::high_resolution_clock::now();

  int total_updates = absl::GetFlag(FLAGS_total_updates);
  double base_lr = absl::GetFlag(FLAGS_learning_rate);
  bool pipeline = absl::GetFlag(FLAGS_pipeline);
  int num_threads = absl::GetFlag(FLAGS_threads);

  // Collect first rollout synchronously.
  open_spiel::CollectResult current_collect = open_spiel::CollectRollout(
      game.get(), evaluator, obs_size, &total_env_steps, num_threads);
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
      bg_collect_thread = std::thread([&]() {
        next_collect = open_spiel::CollectRollout(
            game.get(), evaluator, obs_size, &total_env_steps, num_threads);
      });
    }

    auto ppo_start = std::chrono::high_resolution_clock::now();
    open_spiel::PpoUpdateStats stats =
        open_spiel::TrainPpoUpdate(training_model, *optimizer,
                                   current_collect.rollout, obs_size,
                                   action_size, device);
    auto ppo_end = std::chrono::high_resolution_clock::now();

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
      current_collect = open_spiel::CollectRollout(
          game.get(), evaluator, obs_size, &total_env_steps, num_threads);
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
