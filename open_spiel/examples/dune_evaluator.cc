#include <iostream>
#include <fstream>
#include <memory>
#include <vector>
#include <string>
#include <algorithm>
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

static int ParseRound(const std::string& state_str) {
  size_t pos = state_str.find("round=");
  if (pos != std::string::npos) {
    size_t end_pos = state_str.find(" ", pos);
    if (end_pos != std::string::npos) {
      std::string round_str = state_str.substr(pos + 6, end_pos - (pos + 6));
      try {
        return std::stoi(round_str);
      } catch (...) {
        return 1;
      }
    }
  }
  return 1;
}

static std::string CleanActionName(const std::string& raw_name) {
  // 1. Agent card selection
  if (raw_name.rfind("SelectAgentCard(", 0) == 0) {
    std::string card = raw_name.substr(16);
    if (!card.empty() && card.back() == ')') {
      card.pop_back();
    }
    return "plays " + card;
  }

  // 2. Agent space placement
  if (raw_name.rfind("PlaceAgent[", 0) == 0) {
    std::string space = raw_name.substr(11);
    if (!space.empty() && space.back() == ']') {
      space.pop_back();
    }
    return "to " + space;
  }

  // 3. Card Reveals
  if (raw_name == "Reveal") {
    return "reveals cards";
  }

  // 4. Purchases from Imperium Row
  if (raw_name.rfind("BuyImperiumRow[", 0) == 0) {
    size_t colon_pos = raw_name.find(":");
    if (colon_pos != std::string::npos) {
      std::string card = raw_name.substr(colon_pos + 1);
      if (!card.empty() && card.back() == ']') {
        card.pop_back();
      }
      return "buys " + card;
    }
  }

  // 5. Tech / Tleilaxu acquisitions
  if (raw_name.rfind("AcquireTech[", 0) == 0) {
    size_t colon_pos = raw_name.find(":");
    if (colon_pos != std::string::npos) {
      std::string tech = raw_name.substr(colon_pos + 1);
      if (!tech.empty() && tech.back() == ']') {
        tech.pop_back();
      }
      return "acquires Tech " + tech;
    } else {
      std::string tech = raw_name.substr(12);
      if (!tech.empty() && tech.back() == ']') {
        tech.pop_back();
      }
      return "acquires Tech " + tech;
    }
  }
  if (raw_name.rfind("AcquireTechWithSolari[", 0) == 0) {
    size_t colon_pos = raw_name.find(":");
    if (colon_pos != std::string::npos) {
      std::string tech = raw_name.substr(colon_pos + 1);
      if (!tech.empty() && tech.back() == ']') {
        tech.pop_back();
      }
      return "acquires Tech " + tech + " (with Solari)";
    } else {
      std::string tech = raw_name.substr(22);
      if (!tech.empty() && tech.back() == ']') {
        tech.pop_back();
      }
      return "acquires Tech " + tech + " (with Solari)";
    }
  }
  if (raw_name.rfind("AcquireTleilaxu[", 0) == 0) {
    size_t colon_pos = raw_name.find(":");
    if (colon_pos != std::string::npos) {
      std::string card = raw_name.substr(colon_pos + 1);
      if (!card.empty() && card.back() == ']') {
        card.pop_back();
      }
      return "acquires Tleilaxu " + card;
    } else {
      std::string card = raw_name.substr(16);
      if (!card.empty() && card.back() == ']') {
        card.pop_back();
      }
      return "acquires Tleilaxu " + card;
    }
  }

  return "";
}

int main(int argc, char* argv[]) {
  // Parse command-line arguments
  std::string model_checkpoint = "dune_stage_a_run11_model.pt";
  bool interactive = true;
  std::string output_file = "";

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--non-interactive" || arg == "-n") {
      interactive = false;
    } else if (arg == "--output" || arg == "-o") {
      if (i + 1 < argc) {
        output_file = argv[++i];
      } else {
        std::cerr << "Error: --output requires a file path.\n";
        return 1;
      }
    } else {
      model_checkpoint = arg;
    }
  }

  std::ofstream out_file;
  std::streambuf* cout_buf = nullptr;
  if (!output_file.empty()) {
    out_file.open(output_file);
    if (!out_file.is_open()) {
      std::cerr << "Failed to open output file: " << output_file << "\n";
      return 1;
    }
    cout_buf = std::cout.rdbuf();
    std::cout.rdbuf(out_file.rdbuf());
    std::cerr << "[INFO] Running evaluation... Output is being saved to: " << output_file << "\n";
  }

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
  if (!std::filesystem::exists(model_checkpoint)) {
    std::vector<std::string> paths = {
      "../../../dune_stage_a_run11_model.pt",
      "../../dune_stage_a_run11_model.pt",
      "../dune_stage_a_run11_model.pt",
      "/home/warcr/projects/dune_drl/dune_stage_a_run11_model.pt"
    };
    for (const auto& path : paths) {
      if (std::filesystem::exists(path)) {
        model_checkpoint = path;
        break;
      }
    }
  }

  // Device Management
  torch::Device device(torch::kCPU);
  try {
    if (torch::cuda::is_available()) {
      device = torch::Device(torch::kCUDA);
    }
  } catch (...) {
    device = torch::Device(torch::kCPU);
  }

  // Load Model weights with explicit device mapping
  try {
    std::cout << "Loading model checkpoint onto " << device << "...\n";
    torch::serialize::InputArchive archive;
    archive.load_from(model_checkpoint, device);
    inference_model->load(archive);
    std::cout << "Successfully loaded model weights from " << model_checkpoint << "\n";
  } catch (const c10::Error& e) {
    if (device.is_cuda()) {
      std::cerr << "\n[WARNING] Failed to load checkpoint onto CUDA device. Falling back to CPU.\n";
      std::cerr << "Error details: " << e.msg() << "\n";
      device = torch::Device(torch::kCPU);
      try {
        torch::serialize::InputArchive archive;
        archive.load_from(model_checkpoint, device);
        inference_model->load(archive);
        std::cout << "Successfully loaded model weights onto CPU as fallback.\n";
      } catch (const c10::Error& fallback_err) {
        std::cerr << "\n[ERROR] Failed to load checkpoint on CPU: " << model_checkpoint << "\n";
        std::cerr << "PyTorch says: " << fallback_err.msg() << "\n";
        return 1;
      }
    } else {
      std::cerr << "\n[ERROR] Failed to load checkpoint: " << model_checkpoint << "\n";
      std::cerr << "PyTorch says: " << e.msg() << "\n";
      return 1;
    }
  }

  inference_model->to(device);

  // 5. The Interactive Step-by-Step Loop
  torch::NoGradGuard no_grad; // Crucial: prevents memory leaks during evaluation
  
  std::string last_state_str = "";
  Action last_action = -1;
  int last_round_tracker = 0;

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

    Player current_player = state->CurrentPlayer();

    // Track round transition and print score at the end of every round
    int current_round = ParseRound(state->ToString());
    if (current_round > last_round_tracker) {
      if (last_round_tracker > 0) {
        const auto* dune_state = dynamic_cast<const dune_imperium::DuneImperiumState*>(state.get());
        std::cout << "\n>>> Round " << last_round_tracker << " Completed! Victory Points: ";
        if (dune_state) {
          for (int i = 0; i < 4; ++i) {
            std::cout << "P" << i << ": " << dune_state->GetPlayerVpForTesting(i) << "  ";
          }
        } else {
          auto returns = state->Returns();
          for (size_t i = 0; i < returns.size(); ++i) {
            std::cout << "P" << i << ": " << (int)returns[i] << "  ";
          }
        }
        std::cout << "\n\n" << std::flush;
      }
      std::cout << ">>> Round " << current_round << " Start\n" << std::flush;
      last_round_tracker = current_round;
    }

    // --- TENSOR PREPARATION ---
    // Use InformationStateTensor to match the 5584-float dimension exactly
    std::vector<float> obs_tensor = state->InformationStateTensor();
    torch::Tensor input_tensor = torch::from_blob(obs_tensor.data(), {1, obs_size}, torch::kFloat).to(device);

    // --- QUERY THE NETWORK ---
    auto output = inference_model->forward(input_tensor);

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

    // Execute the #1 chosen action
    std::string current_state_str = state->ToString();
    Action chosen_action = ranked_actions.front().action;
    std::string chosen_action_name = ranked_actions.front().name;

    if (current_state_str == last_state_str && chosen_action == last_action) {
      std::cerr << "[WARNING] Deterministic loop detected on action: " << chosen_action_name 
                << ". Falling back to the next best considered move.\n";
      if (ranked_actions.size() > 1) {
        chosen_action = ranked_actions[1].action;
        chosen_action_name = ranked_actions[1].name;
      } else {
        std::cerr << "[ERROR] No alternative legal actions available to break the loop!\n";
      }
    }

    last_state_str = current_state_str;
    last_action = chosen_action;

    // Only log if it's an agent placement action
    std::string clean_act = CleanActionName(chosen_action_name);
    if (!clean_act.empty()) {
      std::cout << "[Player P" << current_player << "] " << clean_act << "\n" << std::flush;
    }

    state->ApplyAction(chosen_action);
    
    if (interactive) {
      std::cout << "Press Enter to advance to the next turn..." << std::flush;
      std::cin.ignore();
    } else {
      std::cout << std::flush;
    }
  }

  std::cout << "\n=== Game Over ===\n";
  const auto* dune_state = dynamic_cast<const dune_imperium::DuneImperiumState*>(state.get());
  if (dune_state) {
    std::cout << "Final Victory Points: ";
    for (int i = 0; i < 4; ++i) {
      std::cout << "P" << i << ": " << dune_state->GetPlayerVpForTesting(i) << "  ";
    }
    std::cout << "\n";
  }
  std::cout << "Final Returns (Utility): ";
  auto returns = state->Returns();
  for (size_t i = 0; i < returns.size(); ++i) {
    std::cout << "P" << i << ": " << returns[i] << "  ";
  }
  std::cout << "\n" << std::flush;

  if (cout_buf) {
    std::cout.rdbuf(cout_buf);
    std::cout << "[INFO] Evaluation completed successfully! Logs saved to: " << output_file << "\n";
  }

  return 0;
}
