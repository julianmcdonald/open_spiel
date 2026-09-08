// Opponent pool routing and assignment for Dune Imperium PPO training.
#pragma once

#include <array>
#include <atomic>
#include <cstdint>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include <torch/torch.h>

#include "open_spiel/abseil-cpp/absl/strings/str_cat.h"
#include "open_spiel/abseil-cpp/absl/strings/str_split.h"
#include "open_spiel/spiel.h"
#include "dune_network.h"
#include "dune_seed_utils.h"
#include "dune_sha256.h"
#include "dune_ppo_training_utils.h"

namespace open_spiel {
namespace dune_opponent_pool {

struct SeatAssignment {
  int learner_seat = 0;
  // Per-seat assigned opponent index in [0, pool_size - 1] for seats != learner_seat.
  std::array<size_t, 4> opponent_pool_indices = {0, 0, 0, 0};
};

// Deterministically compute seat assignment for an episode using kStreamOpponentAssign.
inline SeatAssignment ComputeSeatAssignment(
    uint64_t master_seed, uint64_t episode_id, size_t pool_size) {
  SeatAssignment assignment;
  assignment.learner_seat = static_cast<int>(episode_id % 4);
  if (pool_size > 0) {
    uint64_t opp_assign_seed = dune_seed::DeriveSeed(
        master_seed, dune_seed::kDomainTrain, episode_id,
        dune_seed::kStreamOpponentAssign);
    std::mt19937_64 opp_rng = dune_seed::MakeRng64(opp_assign_seed);
    std::uniform_int_distribution<size_t> dist(0, pool_size - 1);
    for (int p = 0; p < 4; ++p) {
      if (p != assignment.learner_seat) {
        assignment.opponent_pool_indices[p] = dist(opp_rng);
      }
    }
  }
  return assignment;
}

// Resolve evaluator for current player given seat assignment.
inline std::shared_ptr<IGameEvaluator> ResolveEvaluator(
    int current_player,
    const SeatAssignment& assignment,
    std::shared_ptr<IGameEvaluator> learner_evaluator,
    const std::vector<std::shared_ptr<IGameEvaluator>>& opponent_evaluators) {
  if (current_player == assignment.learner_seat || opponent_evaluators.empty()) {
    return learner_evaluator;
  }
  if (current_player < 0 || current_player >= 4) {
    return learner_evaluator;
  }
  size_t opp_idx = assignment.opponent_pool_indices[current_player];
  if (opp_idx >= opponent_evaluators.size()) {
    SpielFatalError(absl::StrCat("Opponent index out of range: ", opp_idx,
                                 " >= pool size ", opponent_evaluators.size()));
  }
  return opponent_evaluators[opp_idx];
}

// Parse comma-separated checkpoint paths.
inline std::vector<std::string> ParseOpponentPoolPaths(const std::string& csv) {
  std::vector<std::string> paths;
  for (absl::string_view piece : absl::StrSplit(csv, ',', absl::SkipEmpty())) {
    paths.push_back(std::string(piece));
  }
  return paths;
}

// Compute hash of all named parameters and buffers for immutability verification.
inline std::string HashModelParametersAndBuffers(
    const std::shared_ptr<SharedDunePolicyValueNetImpl>& model) {
  std::stringstream ss;
  torch::NoGradGuard no_grad;
  auto append = [&](const std::string& name, const torch::Tensor& source) {
    const torch::Tensor tensor =
        source.detach().contiguous().cpu().to(torch::kFloat32);
    ss << name << ':';
    for (int64_t dim : tensor.sizes()) ss << dim << ',';
    ss << ';';
    ss.write(reinterpret_cast<const char*>(tensor.data_ptr<float>()),
             tensor.numel() * sizeof(float));
  };
  for (const auto& item : model->named_parameters()) {
    append("P/" + item.key(), item.value());
  }
  for (const auto& item : model->named_buffers()) {
    append("B/" + item.key(), item.value());
  }
  return ComputeStringSHA256(ss.str());
}

// Reusable standalone simulation of a single-learner episode with opponent pool routing.
inline int SimulateSingleLearnerRollout(
    uint64_t master, uint64_t episode_id, const Game& game,
    std::shared_ptr<IGameEvaluator> learner_evaluator,
    const std::vector<std::shared_ptr<IGameEvaluator>>& opponent_evaluators,
    int64_t obs_size,
    std::vector<PpoTransition>* trajectory,
    std::atomic<uint64_t>* total_env_steps,
    float reward_scale = 4.0f) {
  const auto seat_assignment =
      ComputeSeatAssignment(master, episode_id, opponent_evaluators.size());
  const int learner_seat = seat_assignment.learner_seat;

  auto chance_rng = dune_seed::MakeRng64(
      dune_seed::DeriveSeed(master, dune_seed::kDomainTrain, episode_id, dune_seed::kStreamChance));
  std::mt19937_64 policy_rng[4];
  for (int p = 0; p < 4; ++p) {
    uint64_t seed = dune_seed::DeriveSeed(master, dune_seed::kDomainTrain, episode_id,
                                          dune_seed::kStreamPolicyPlayer0 + p);
    policy_rng[p] = dune_seed::MakeRng64(seed);
  }

  std::unique_ptr<State> state = game.NewInitialState();
  bool provides_info_state_tensor = game.GetType().provides_information_state_tensor;
  bool provides_observations_tensor = game.GetType().provides_observation_tensor;

  int game_length = 0;
  while (!state->IsTerminal()) {
    ++game_length;
    if (game_length > 5000) std::abort();

    if (state->IsChanceNode()) {
      auto outcomes = state->ChanceOutcomes();
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
    if (actions.empty()) std::abort();

    const bool is_learner = (current_player == learner_seat);

    std::vector<float> obs(obs_size, 0.0f);
    if (provides_info_state_tensor && current_player >= 0) {
      state->InformationStateTensor(current_player, absl::MakeSpan(obs));
    } else if (provides_observations_tensor && current_player >= 0) {
      state->ObservationTensor(current_player, absl::MakeSpan(obs));
    }

    std::shared_ptr<IGameEvaluator> active_evaluator =
        ResolveEvaluator(current_player, seat_assignment, learner_evaluator, opponent_evaluators);
    EvalResult result = active_evaluator->Evaluate(obs);
    std::vector<float> logits = std::move(result.logits);
    CenterAndCapLegalLogits(logits, actions, 10.0f);

    const PolicyDistributionSample policy_sample =
        SamplePolicyDistribution(&policy_rng[current_player], logits, actions, nullptr);
    Action action = policy_sample.action;
    float old_log_prob = policy_sample.chosen_log_probability;

    state->ApplyAction(action);
    if (total_env_steps) total_env_steps->fetch_add(1, std::memory_order_relaxed);

    if (is_learner) {
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
    }
  }

  const std::vector<double> terminal_returns =
      open_spiel::ComputeTerminalReturns(*state, "placement", 0.0);

  float last_value = 0.0f;
  float last_gae = 0.0f;
  bool seen_last = false;
  for (auto it = trajectory->rbegin(); it != trajectory->rend(); ++it) {
    float reward = it->reward;
    if (!seen_last) {
      reward += static_cast<float>(terminal_returns[learner_seat]);
      seen_last = true;
    }
    reward = std::clamp(reward / reward_scale, -1.0f, 1.0f);
    float delta = reward + 1.0f * last_value - it->value;
    float advantage = delta + 1.0f * last_gae;
    it->advantage = advantage;
    it->return_value = advantage + it->value;
    last_value = it->value;
    last_gae = advantage;
  }

  return game_length;
}

}  // namespace dune_opponent_pool
}  // namespace open_spiel
