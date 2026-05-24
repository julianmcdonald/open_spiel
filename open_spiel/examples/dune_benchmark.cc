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
#include <iterator>

#include "open_spiel/abseil-cpp/absl/flags/flag.h"
#include "open_spiel/abseil-cpp/absl/flags/parse.h"
#include "open_spiel/abseil-cpp/absl/strings/str_format.h"
#include "open_spiel/spiel.h"

#ifdef OPEN_SPIEL_BUILD_WITH_LIBTORCH
#include <torch/torch.h>
#endif

ABSL_FLAG(std::string, game, "dune_imperium", "The name of the game to play.");
ABSL_FLAG(int, games, 1000, "How many games to play in total.");
ABSL_FLAG(int, threads, 8, "How many threads to run.");

namespace open_spiel {

struct Transition {
  std::vector<float> spatial_tensor;     // Flattened observation tensor
  std::vector<int> legal_actions_mask;   // 1 for legal, 0 for illegal
  int64_t action_taken;                  // Action ID chosen
  std::vector<float> prev_logits;        // Raw policy logits (z_k)
  std::vector<float> base_logits;        // Raw logits (z_base)
  float q_value;                         // Estimated action-value Q(s, a)
  float reward_target;                   // Mapped zero-sum reward
  int player_id;                         // Seat ID of the deciding player
};

class GlobalReplayBuffer {
 private:
  std::deque<Transition> buffer_;
  std::mutex buffer_mutex_;
  size_t max_capacity_;

 public:
  GlobalReplayBuffer(size_t capacity) : max_capacity_(capacity) {}

  void PushTrajectory(std::vector<Transition>& local_trajectory) {
    std::lock_guard<std::mutex> lock(buffer_mutex_);
    
    buffer_.insert(buffer_.end(), 
                   std::make_move_iterator(local_trajectory.begin()), 
                   std::make_move_iterator(local_trajectory.end()));
                   
    while (buffer_.size() > max_capacity_) {
      buffer_.pop_front();
    }
  }

  std::vector<Transition> SampleBatch(size_t batch_size) {
    std::lock_guard<std::mutex> lock(buffer_mutex_);
    std::vector<Transition> batch;
    if (buffer_.empty()) return batch;
    
    batch.reserve(batch_size);
    for (size_t i = 0; i < batch_size; ++i) {
      size_t idx = rand() % buffer_.size();
      batch.push_back(buffer_[idx]);
    }
    return batch;
  }

  size_t Size() {
    std::lock_guard<std::mutex> lock(buffer_mutex_);
    return buffer_.size();
  }
};

namespace {

#ifdef OPEN_SPIEL_BUILD_WITH_LIBTORCH
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

struct SharedDunePolicyValueNetImpl : torch::nn::Module {
  torch::nn::Linear input_layer{nullptr};
  std::shared_ptr<ResBlockImpl> res1{nullptr};
  std::shared_ptr<ResBlockImpl> res2{nullptr};
  torch::nn::Linear policy_head{nullptr};
  torch::nn::Linear value_head{nullptr};

  SharedDunePolicyValueNetImpl(int64_t input_dim, int64_t hidden_dim, int64_t action_dim) {
    input_layer = register_module("input_layer", torch::nn::Linear(input_dim, hidden_dim));
    
    // Register the underlying implementation object wrapped in shared_ptr to solve template deduction
    res1 = register_module("res1", std::make_shared<ResBlockImpl>(hidden_dim));
    res2 = register_module("res2", std::make_shared<ResBlockImpl>(hidden_dim));
    
    policy_head = register_module("policy_head", torch::nn::Linear(hidden_dim, action_dim));
    value_head = register_module("value_head", torch::nn::Linear(hidden_dim, 1));
  }

  struct ModelOutputs {
    torch::Tensor logits;
    torch::Tensor values;
  };

  ModelOutputs forward(torch::Tensor x) {
    x = torch::relu(input_layer->forward(x));
    x = res1->forward(x);
    x = res2->forward(x);
    
    torch::Tensor logits = policy_head->forward(x);
    torch::Tensor values = torch::tanh(value_head->forward(x));
    return {logits, values};
  }
};
TORCH_MODULE(SharedDunePolicyValueNet);

void TrainStep(std::shared_ptr<SharedDunePolicyValueNetImpl> model, 
               torch::optim::Optimizer& optimizer, 
               const std::vector<Transition>& batch, 
               int64_t obs_size,
               int64_t action_dim) {
  size_t batch_size = batch.size();
  if (batch_size == 0) return;

  std::vector<float> flat_states;
  std::vector<float> flat_masks;
  std::vector<float> flat_rewards;
  flat_states.reserve(batch_size * obs_size);
  flat_masks.reserve(batch_size * action_dim);
  flat_rewards.reserve(batch_size);

  for (const auto& transition : batch) {
    flat_states.insert(flat_states.end(), transition.spatial_tensor.begin(), transition.spatial_tensor.end());
    for (int mask_val : transition.legal_actions_mask) {
      flat_masks.push_back(static_cast<float>(mask_val));
    }
    // Scale target reward: divide by 2.25 to bound target in [-0.77, 1.00]
    flat_rewards.push_back(transition.reward_target / 2.25f);
  }

  torch::Tensor states = torch::from_blob(flat_states.data(), {(int64_t)batch_size, obs_size}, torch::kFloat).clone();
  torch::Tensor masks = torch::from_blob(flat_masks.data(), {(int64_t)batch_size, action_dim}, torch::kFloat).clone();
  torch::Tensor rewards = torch::from_blob(flat_rewards.data(), {(int64_t)batch_size, 1}, torch::kFloat).clone();

  optimizer.zero_grad();
  auto outputs = model->forward(states);
  torch::Tensor pred_logits = outputs.logits;
  torch::Tensor pred_values = outputs.values;

  // Mask illegal action logits (mask == 0) with a large negative value (-1e9f)
  torch::Tensor masked_logits = pred_logits.masked_fill(masks == 0.0f, -1e9f);

  // Compute log_softmax over masked logits
  torch::Tensor log_probs = torch::log_softmax(masked_logits, /*dim=*/-1);

  // Value Loss (MSE between scaled zero-sum returns and Tanh bounded prediction)
  torch::Tensor value_loss = torch::nn::functional::mse_loss(pred_values, rewards);

  // Mock Policy Loss for optimization verification (KL divergence against a softmax distribution over masked logits)
  torch::Tensor mock_target = torch::softmax(masked_logits, /*dim=*/-1).detach();
  torch::Tensor policy_loss = torch::nn::functional::kl_div(
      log_probs, 
      mock_target, 
      torch::nn::functional::KLDivFuncOptions().reduction(torch::kBatchMean)
  );

  torch::Tensor total_loss = policy_loss + value_loss;
  total_loss.backward();
  optimizer.step();
}

int TorchSimulation(std::mt19937* rng, const Game& game, SharedDunePolicyValueNetImpl* model, int64_t obs_size, std::vector<Transition>& trajectory) {
  std::unique_ptr<State> state = game.NewInitialState();

  bool provides_info_state_tensor =
      game.GetType().provides_information_state_tensor;
  bool provides_observations_tensor =
      game.GetType().provides_observation_tensor;

  // CRITICAL: Disable gradient tracking for self-play generation
  torch::NoGradGuard no_grad;

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

      // 2. Assemble legal actions mask
      int64_t action_size = game.NumDistinctActions();
      std::vector<int> legal_actions_mask(action_size, 0);
      for (Action a : actions) {
        if (a >= 0 && a < action_size) {
          legal_actions_mask[a] = 1;
        }
      }

      // Convert C++ Vector to LibTorch Tensor (Zero-copy mapping)
      torch::Tensor state_tensor = torch::from_blob(obs.data(), {1, obs_size}, torch::kFloat);
      
      // Forward Pass
      torch::Tensor logits = model->forward(state_tensor).logits;

      // Select action randomly to guarantee state progression, mirroring RandomSimulation
      std::uniform_int_distribution<int> dis(0, actions.size() - 1);
      Action action = actions[dis(*rng)];
      state->ApplyAction(action);

      // Push to trajectory
      Transition transition;
      transition.spatial_tensor = std::move(obs);
      transition.legal_actions_mask = std::move(legal_actions_mask);
      transition.action_taken = action;
      transition.prev_logits = std::vector<float>(action_size, 0.0f);
      transition.base_logits = std::vector<float>(action_size, 0.0f);
      transition.q_value = 0.0f;
      transition.reward_target = 0.0f;
      transition.player_id = current_player;
      trajectory.push_back(std::move(transition));
    }
  }

  // Populate rewards at game terminal state
  std::vector<double> returns = state->Returns();
  for (auto& transition : trajectory) {
    if (transition.player_id >= 0 && transition.player_id < static_cast<int>(returns.size())) {
      transition.reward_target = static_cast<float>(returns[transition.player_id]);
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

      // 2. Assemble legal actions mask
      int64_t action_size = game.NumDistinctActions();
      std::vector<int> legal_actions_mask(action_size, 0);
      for (Action a : actions) {
        if (a >= 0 && a < action_size) {
          legal_actions_mask[a] = 1;
        }
      }

      // Apply action
      state->ApplyAction(action);

      // Push to trajectory
      Transition transition;
      transition.spatial_tensor = std::move(obs);
      transition.legal_actions_mask = std::move(legal_actions_mask);
      transition.action_taken = action;
      transition.prev_logits = std::vector<float>(action_size, 0.0f);
      transition.base_logits = std::vector<float>(action_size, 0.0f);
      transition.q_value = 0.0f;
      transition.reward_target = 0.0f;
      transition.player_id = current_player;
      trajectory.push_back(std::move(transition));
    }
  }

  // Populate rewards at game terminal state
  std::vector<double> returns = state->Returns();
  for (auto& transition : trajectory) {
    if (transition.player_id >= 0 && transition.player_id < static_cast<int>(returns.size())) {
      transition.reward_target = static_cast<float>(returns[transition.player_id]);
    }
  }

  return game_length;
}

void ThreadWorker(int thread_id, const Game* game, std::atomic<int>& games_completed,
                  int total_games, std::atomic<int>& total_moves,
                  GlobalReplayBuffer* global_buffer, int64_t obs_size
#ifdef OPEN_SPIEL_BUILD_WITH_LIBTORCH
                  , SharedDunePolicyValueNetImpl* model
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
    if (model != nullptr) {
      moves += TorchSimulation(&rng, *game, model, obs_size, local_trajectory);
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
  
  std::shared_ptr<open_spiel::SharedDunePolicyValueNetImpl> model = nullptr;
  if (obs_size > 0) {
    model = std::make_shared<open_spiel::SharedDunePolicyValueNetImpl>(obs_size, 1024, action_size);
    model->eval(); // Put into evaluation mode
    std::cout << absl::StrFormat("Initialized SharedDunePolicyValueNet (Input Dim: %d, Hidden Dim: 1024, Action Dim: %d)\n", obs_size, action_size);
  }
#endif

  // Construct global thread-safe replay buffer
  open_spiel::GlobalReplayBuffer global_buffer(1000000);

  std::atomic<int> games_completed(0);
  std::atomic<int> total_moves(0);

  auto start_time = std::chrono::high_resolution_clock::now();

  std::vector<std::thread> threads;
  threads.reserve(num_threads);
  for (int i = 0; i < num_threads; ++i) {
    threads.emplace_back(open_spiel::ThreadWorker, i, game.get(),
                         std::ref(games_completed), total_games, std::ref(total_moves),
                         &global_buffer, obs_size
#ifdef OPEN_SPIEL_BUILD_WITH_LIBTORCH
                         , model.get()
#endif
    );
  }

  for (auto& t : threads) {
    t.join();
  }

  auto end_time = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = end_time - start_time;

  double seconds = elapsed.count();
  int moves = total_moves.load();
  int completed = games_completed.load();

#ifdef OPEN_SPIEL_BUILD_WITH_LIBTORCH
  if (model != nullptr && global_buffer.Size() > 0) {
    std::cout << "\nStarting optimization validation phase...\n";
    model->train(); // Toggle to training mode to enable gradient tracking

    // Initialize Adam Optimizer
    torch::optim::Adam optimizer(model->parameters(), torch::optim::AdamOptions(1e-3));

    // Sample a single training batch of size 32
    size_t batch_size = 32;
    std::vector<open_spiel::Transition> batch = global_buffer.SampleBatch(batch_size);
    std::cout << absl::StrFormat("Sampled %zu transitions from replay buffer.\n", batch.size());

    // Execute backprop step
    open_spiel::TrainStep(model, optimizer, batch, obs_size, action_size);
    std::cout << "Optimization Step successfully completed (Forward + Loss + Backward + Weight Update)!\n";
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
