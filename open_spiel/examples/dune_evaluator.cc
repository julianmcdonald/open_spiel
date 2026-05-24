#include <iostream>
#include <memory>
#include <vector>
#include <string>
#include <algorithm>
#include <iomanip>
#include <filesystem>

#include "open_spiel/spiel.h"
#include "open_spiel/games/dune_imperium/dune_imperium.h"
#include <torch/torch.h>

// Import the synced shared architecture
#include "dune_network.h" 

using namespace open_spiel;

struct ActionProb {
  Action action;
  float probability;
  std::string name;
};

int main(int argc, char* argv[]) {
  std::cout << "=== Dune: Imperium AI Evaluator ===\n";

  // 1. Initialize the Game
  std::shared_ptr<const Game> game = LoadGame("dune_imperium");
  std::unique_ptr<State> state = game->NewInitialState();

  // 2. Initialize the Network
  int64_t obs_size = 5584; // Must match InformationStateTensor shape
  int64_t hidden_dim = 1024;
  int64_t action_size = 2391;
  
  auto inference_model = std::make_shared<SharedDunePolicyValueNetImpl>(obs_size, hidden_dim, action_size);
  inference_model->eval(); // Disable Dropout/BatchNorm variance

  // 3. Robust Multi-Path Checkpoint Loading
  std::string model_checkpoint = "dune_stage_a_model.pt";
  if (argc > 1) {
    model_checkpoint = argv[1];
  } else if (!std::filesystem::exists(model_checkpoint)) {
    std::vector<std::string> paths = {
      "../../../dune_stage_a_model.pt",
      "../../dune_stage_a_model.pt",
      "../dune_stage_a_model.pt",
      "/home/warcr/projects/dune_drl/dune_stage_a_model.pt"
    };
    for (const auto& path : paths) {
      if (std::filesystem::exists(path)) {
        model_checkpoint = path;
        break;
      }
    }
  }

  try {
    torch::load(inference_model, model_checkpoint);
    std::cout << "Successfully loaded brain from " << model_checkpoint << "\n";
  } catch (const c10::Error& e) {
    std::cerr << "\n[ERROR] Failed to load checkpoint.\n";
    std::cerr << "PyTorch says: " << e.msg() << "\n";
    return 1;
  }

  // 4. Device Management (Map Model and Tensors to CUDA if available, CPU otherwise)
  torch::Device device(torch::kCPU);
  if (torch::cuda::is_available()) {
    device = torch::Device(torch::kCUDA);
    std::cout << "CUDA is available! Running inference on CUDA device.\n\n";
  } else {
    std::cout << "Running inference on CPU.\n\n";
  }
  inference_model->to(device);

  // 5. The Interactive Step-by-Step Loop
  torch::NoGradGuard no_grad; // Crucial: prevents memory leaks during evaluation
  
  while (!state->IsTerminal()) {
    
    // --- CHANCE NODES (Decks, Market Reveals) ---
    if (state->IsChanceNode()) {
      auto outcomes = state->ChanceOutcomes();
      // Take the first chance outcome for evaluation stability
      if (!outcomes.empty()) {
        state->ApplyAction(outcomes.front().first);
      }
      continue; 
    }

    std::cout << "----------------------------------------------------\n";
    Player current_player = state->CurrentPlayer();
    std::cout << "Current Player: P" << current_player << "\n";
    
    // Print native OpenSpiel board visualization
    std::cout << "Board State:\n" << state->ToString() << "\n";

    // --- TENSOR PREPARATION ---
    // Use InformationStateTensor to match the 5584-float dimension exactly
    std::vector<float> obs_tensor = state->InformationStateTensor();
    torch::Tensor input_tensor = torch::from_blob(obs_tensor.data(), {1, obs_size}, torch::kFloat).to(device);

    // --- QUERY THE NETWORK ---
    auto output = inference_model->forward(input_tensor);
    
    // Extract Value (Win Probability) - Move back to CPU for extraction
    float raw_value = output.values.to(torch::kCPU).item<float>();
    // Convert Tanh [-1.0, 1.0] to Percentage [0%, 100%]
    float win_pct = ((raw_value + 1.0f) / 2.0f) * 100.0f;
    
    std::cout << "\n>> AI Evaluation: " << std::fixed << std::setprecision(1) 
              << win_pct << "% estimated chance to win.\n";

    // Extract Policy (Action Probabilities)
    std::vector<Action> legal_actions = state->LegalActions();
    torch::Tensor policy_logits = output.logits.squeeze(0); // Remains on the model's device
    
    // Mask illegal actions on the active device
    torch::Tensor mask = torch::full({action_size}, -1e9f, torch::TensorOptions().device(device));
    for (Action a : legal_actions) {
      mask[a] = 0.0f;
    }
    
    torch::Tensor masked_logits = policy_logits + mask;
    torch::Tensor probabilities = torch::softmax(masked_logits, /*dim=*/0).to(torch::kCPU); // Move back to CPU for iteration

    // Rank the actions
    std::vector<ActionProb> ranked_actions;
    for (Action a : legal_actions) {
      ranked_actions.push_back({a, probabilities[a].item<float>(), state->ActionToString(current_player, a)});
    }

    std::sort(ranked_actions.begin(), ranked_actions.end(), 
      [](const ActionProb& a, const ActionProb& b) {
        return a.probability > b.probability;
      });

    // Print Top Decisions
    std::cout << ">> Top Considered Moves:\n";
    int display_count = std::min(3, (int)ranked_actions.size());
    for (int i = 0; i < display_count; ++i) {
      std::cout << "   " << i + 1 << ". " << ranked_actions[i].name 
                << " (" << std::fixed << std::setprecision(1) << ranked_actions[i].probability * 100.0f << "%)\n";
    }

    // Execute the #1 chosen action
    Action chosen_action = ranked_actions.front().action;
    std::cout << "\n>> AI Executes: " << ranked_actions.front().name << "\n\n";
    state->ApplyAction(chosen_action);
    
    std::cout << "Press Enter to advance to the next turn...";
    std::cin.ignore();
  }

  std::cout << "\n=== Game Over ===\n";
  std::cout << "Final Returns (Scores): ";
  auto returns = state->Returns();
  for (size_t i = 0; i < returns.size(); ++i) {
    std::cout << "P" << i << ": " << returns[i] << "  ";
  }
  std::cout << "\n";

  return 0;
}
