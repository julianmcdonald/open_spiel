#include <iostream>
#include <fstream>
#include <memory>
#include <vector>
#include <string>
#include <algorithm>
#include <filesystem>

#include "open_spiel/spiel.h"
#include "open_spiel/games/dune_imperium/dune_imperium.h"
#include "open_spiel/games/dune_imperium/dune_imperium_content.h"
#include "open_spiel/games/dune_imperium/dune_imperium_cards.h"
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

  // 1b. Graft partner selection
  if (raw_name.rfind("SelectGraftPartner(", 0) == 0) {
    std::string card = raw_name.substr(19);
    if (!card.empty() && card.back() == ')') {
      card.pop_back();
    }
    return "grafts " + card;
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
      return "acquires " + card;
    } else {
      std::string card = raw_name.substr(16);
      if (!card.empty() && card.back() == ']') {
        card.pop_back();
      }
      return "acquires " + card;
    }
  }

  // 6. Leader Draft
  if (raw_name.rfind("LeaderPick[", 0) == 0) {
    std::string leader_id_str = raw_name.substr(11);
    if (!leader_id_str.empty() && leader_id_str.back() == ']') {
      leader_id_str.pop_back();
    }
    try {
      int leader_id = std::stoi(leader_id_str);
      if (leader_id >= 0 && leader_id < dune_imperium::kNumLeaders) {
        return "drafts " + std::string(dune_imperium::kLeaders[leader_id].name);
      }
    } catch (...) {}
    return "drafts leader " + leader_id_str;
  }

  // 7. Shipping Track Actions
  if (raw_name == "ShippingAdvance") {
    return "advances on the shipping track";
  }
  if (raw_name == "ShippingRecall") {
    return "recalls from the shipping track";
  }
  if (raw_name == "ShippingRecallSkip") {
    return "skips shipping recall";
  }
  if (raw_name == "ShippingLevel1Spice") {
    return "chooses Spice from shipping recall";
  }
  if (raw_name == "ShippingLevel1Dividends") {
    return "chooses Dividends from shipping recall";
  }
  if (raw_name == "ShippingInfluenceEmperor") {
    return "gains Emperor influence from shipping recall";
  }
  if (raw_name == "ShippingInfluenceSpacingGuild") {
    return "gains Spacing Guild influence from shipping recall";
  }
  if (raw_name == "ShippingInfluenceBeneGesserit") {
    return "gains Bene Gesserit influence from shipping recall";
  }
  if (raw_name == "ShippingInfluenceFremen") {
    return "gains Fremen influence from shipping recall";
  }
  if (raw_name == "ShippingLevel2Troops") {
    return "gains troops from shipping recall";
  }

  // 8. Combat Commit Action
  if (raw_name.rfind("CombatCommit[d=", 0) == 0) {
    size_t d_pos = raw_name.find("d=");
    size_t t_pos = raw_name.find("t=");
    if (d_pos != std::string::npos && t_pos != std::string::npos) {
      std::string d_str = raw_name.substr(d_pos + 2, raw_name.find(",", d_pos) - (d_pos + 2));
      std::string t_str = raw_name.substr(t_pos + 2, raw_name.find("]", t_pos) - (t_pos + 2));
      return "commits " + d_str + " dreadnoughts and " + t_str + " troops to combat";
    }
  }

  return raw_name;
}

struct PlayerSnapshot {
  int vp = 0;
  int influence[4] = {0, 0, 0, 0};
  int shipping_level = 0;
  int tleilaxu_track = 0;
  int spice = 0;
  int solari = 0;
  int water = 0;
  int garrison_troops = 0;
  int combat_troops = 0;
  int garrison_dreads = 0;
  int combat_dreads = 0;
  int intrigue_hand_size = 0;
};

struct BoardSnapshot {
  dune_imperium::GamePhase phase;
  PlayerSnapshot players[4];
  int alliance_owner[4] = {-1, -1, -1, -1};
};

static BoardSnapshot GetBoardSnapshot(const dune_imperium::DuneImperiumState* dune_state) {
  BoardSnapshot snap;
  if (!dune_state) return snap;
  snap.phase = dune_state->phase();
  for (int i = 0; i < 4; ++i) {
    snap.players[i].vp = dune_state->GetPlayerVpForTesting(i);
    for (int f = 0; f < 4; ++f) {
      snap.players[i].influence[f] = dune_state->GetPlayerInfluenceForTesting(i, static_cast<dune_imperium::Faction>(f));
    }
    snap.players[i].shipping_level = dune_state->PlayerShippingLevelForTesting(i);
    snap.players[i].tleilaxu_track = dune_state->GetTleilaxuTrackForTesting(i);
    snap.players[i].spice = dune_state->GetPlayerSpiceForTesting(i);
    snap.players[i].solari = dune_state->GetPlayerSolariForTesting(i);
    snap.players[i].water = dune_state->GetPlayerWaterForTesting(i);
    snap.players[i].garrison_troops = dune_state->GetPlayerTroopsInGarrisonForTesting(i);
    snap.players[i].combat_troops = dune_state->TroopsInCombat(i);
    snap.players[i].garrison_dreads = dune_state->GetPlayerDreadnoughtsInGarrisonForTesting(i);
    snap.players[i].combat_dreads = dune_state->DreadnoughtsInCombat(i);
    snap.players[i].intrigue_hand_size = dune_state->GetIntrigueHandForTesting(i).size();
  }
  for (int f = 0; f < 4; ++f) {
    snap.alliance_owner[f] = dune_state->GetAllianceOwnerForTesting(f);
  }
  return snap;
}

static void CompareSnapshotsAndLog(const BoardSnapshot& old_snap, const BoardSnapshot& new_snap, const std::string& action_context) {
  // 1. Check alliance changes
  for (int f = 0; f < 4; ++f) {
    if (old_snap.alliance_owner[f] != new_snap.alliance_owner[f]) {
      std::string faction_name = (f == 0 ? "Emperor" : (f == 1 ? "Spacing Guild" : (f == 2 ? "Bene Gesserit" : "Fremen")));
      if (old_snap.alliance_owner[f] != -1) {
        std::cout << "  [VP Update] Player P" << old_snap.alliance_owner[f] << " loses alliance with " << faction_name << " (-1 VP)\n";
      }
      if (new_snap.alliance_owner[f] != -1) {
        std::cout << "  [VP Update] Player P" << new_snap.alliance_owner[f] << " gains alliance with " << faction_name << " (+1 VP)\n";
      }
    }
  }

  // 2. Check individual player changes
  for (int p = 0; p < 4; ++p) {
    // 2a. Check influence changes
    for (int f = 0; f < 4; ++f) {
      if (old_snap.players[p].influence[f] != new_snap.players[p].influence[f]) {
        std::string faction_name = (f == 0 ? "Emperor" : (f == 1 ? "Spacing Guild" : (f == 2 ? "Bene Gesserit" : "Fremen")));
        std::cout << "  [Influence Update] Player P" << p << " influence with " << faction_name << " goes from "
                  << old_snap.players[p].influence[f] << " to " << new_snap.players[p].influence[f];
        if (old_snap.players[p].influence[f] < 2 && new_snap.players[p].influence[f] >= 2) {
          std::cout << " (+1 VP for reaching 2 influence)";
        }
        std::cout << "\n";
      }
    }

    // 2b. Check Tleilaxu track level changes
    if (old_snap.players[p].tleilaxu_track != new_snap.players[p].tleilaxu_track) {
      std::cout << "  [Tleilaxu Update] Player P" << p << " Tleilaxu track level goes from "
                << old_snap.players[p].tleilaxu_track << " to " << new_snap.players[p].tleilaxu_track << "\n";
    }

    // 2c. Check overall VP changes
    int vp_diff = new_snap.players[p].vp - old_snap.players[p].vp;
    if (vp_diff != 0) {
      std::vector<std::string> reasons;

      for (int f = 0; f < 4; ++f) {
        if (old_snap.players[p].influence[f] < 2 && new_snap.players[p].influence[f] >= 2) {
          std::string faction_name = (f == 0 ? "Emperor" : (f == 1 ? "Spacing Guild" : (f == 2 ? "Bene Gesserit" : "Fremen")));
          reasons.push_back("reaching 2 influence on " + faction_name + " track");
        }
        if (old_snap.alliance_owner[f] != p && new_snap.alliance_owner[f] == p) {
          std::string faction_name = (f == 0 ? "Emperor" : (f == 1 ? "Spacing Guild" : (f == 2 ? "Bene Gesserit" : "Fremen")));
          reasons.push_back("gaining " + faction_name + " Alliance");
        }
        if (old_snap.alliance_owner[f] == p && new_snap.alliance_owner[f] != p) {
          std::string faction_name = (f == 0 ? "Emperor" : (f == 1 ? "Spacing Guild" : (f == 2 ? "Bene Gesserit" : "Fremen")));
          reasons.push_back("losing " + faction_name + " Alliance");
        }
      }

      if (old_snap.players[p].tleilaxu_track != new_snap.players[p].tleilaxu_track) {
        if (new_snap.players[p].tleilaxu_track == 3 || new_snap.players[p].tleilaxu_track == 7) {
          reasons.push_back("reaching Tleilaxu track milestone");
        }
      }

      if (action_context.find("The Spice Must Flow") != std::string::npos || action_context.find("BuyReserveTheSpiceMustFlow") != std::string::npos) {
        reasons.push_back("purchasing The Spice Must Flow card");
      }

      if (reasons.empty()) {
        if (old_snap.phase == dune_imperium::GamePhase::kCombat && new_snap.phase != dune_imperium::GamePhase::kCombat) {
          reasons.push_back("combat rewards / conflict resolution");
        } else {
          reasons.push_back("card/intrigue/action effect (" + action_context + ")");
        }
      }

      std::cout << "  [VP Update] Player P" << p << " VP goes from " << old_snap.players[p].vp
                << " to " << new_snap.players[p].vp << " (diff: " << (vp_diff > 0 ? "+" : "") << vp_diff << ") due to: ";
      for (size_t r = 0; r < reasons.size(); ++r) {
        if (r > 0) std::cout << ", ";
        std::cout << reasons[r];
      }
      std::cout << "\n";
    }
  }
}

int main(int argc, char* argv[]) {
  // Parse command-line arguments
  std::string model_checkpoint = "dune_stage_a_run11_model.pt";
  bool interactive = true;
  std::string output_file = "";
  int64_t hidden_dim = 2048;
  int num_blocks = 8;

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
    } else if (arg == "--hidden_dim" || arg == "--hidden-dim") {
      if (i + 1 < argc) {
        hidden_dim = std::stoll(argv[++i]);
      } else {
        std::cerr << "Error: --hidden_dim requires an integer.\n";
        return 1;
      }
    } else if (arg == "--num_blocks" || arg == "--num-blocks") {
      if (i + 1 < argc) {
        num_blocks = std::stoi(argv[++i]);
      } else {
        std::cerr << "Error: --num_blocks requires an integer.\n";
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
  int64_t obs_size = game->InformationStateTensorShape()[0];
  int64_t action_size = 2391;
  
  auto inference_model = std::make_shared<SharedDunePolicyValueNetImpl>(obs_size, hidden_dim, action_size, num_blocks);
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
        const auto* dune_state = dynamic_cast<const dune_imperium::DuneImperiumState*>(state.get());
        BoardSnapshot old_snap = GetBoardSnapshot(dune_state);

        state->ApplyAction(outcomes.front().first);

        const auto* post_dune_state = dynamic_cast<const dune_imperium::DuneImperiumState*>(state.get());
        BoardSnapshot new_snap = GetBoardSnapshot(post_dune_state);
        CompareSnapshotsAndLog(old_snap, new_snap, "ChanceNode");
      }
      continue; 
    }

    Player current_player = state->CurrentPlayer();

    // Track round transition and print score at the end of every round
    int current_round = ParseRound(state->ToString());
    if (current_round > last_round_tracker) {
      if (last_round_tracker > 0) {
        const auto* dune_state = dynamic_cast<const dune_imperium::DuneImperiumState*>(state.get());
        std::cout << "\n>>> Round " << last_round_tracker << " Completed!\n";
        if (dune_state) {
          for (int i = 0; i < 4; ++i) {
            std::cout << "  P" << i << ": " << dune_state->GetPlayerVpForTesting(i) << " VP"
                      << " (Spice: " << dune_state->GetPlayerSpiceForTesting(i)
                      << ", Solari: " << dune_state->GetPlayerSolariForTesting(i)
                      << ", Water: " << dune_state->GetPlayerWaterForTesting(i) << ")\n"
                      << "       Shipping Level: " << dune_state->PlayerShippingLevelForTesting(i) << "\n"
                      << "       Influence: Emperor=" << dune_state->GetPlayerInfluenceForTesting(i, dune_imperium::Faction::kEmperor)
                      << ", Spacing Guild=" << dune_state->GetPlayerInfluenceForTesting(i, dune_imperium::Faction::kSpacingGuild)
                      << ", Bene Gesserit=" << dune_state->GetPlayerInfluenceForTesting(i, dune_imperium::Faction::kBeneGesserit)
                      << ", Fremen=" << dune_state->GetPlayerInfluenceForTesting(i, dune_imperium::Faction::kFremen) << "\n";
            std::vector<std::string> alliances;
            for (int f = 0; f < 4; ++f) {
              if (dune_state->GetAllianceOwnerForTesting(f) == i) {
                alliances.push_back(f == 0 ? "Emperor" : (f == 1 ? "Spacing Guild" : (f == 2 ? "Bene Gesserit" : "Fremen")));
              }
            }
            if (!alliances.empty()) {
              std::cout << "       Alliances Owned: ";
              for (size_t a = 0; a < alliances.size(); ++a) {
                if (a > 0) std::cout << ", ";
                std::cout << alliances[a];
              }
              std::cout << "\n";
            }
          }
        } else {
          std::cout << "Victory Points: ";
          auto returns = state->Returns();
          for (size_t i = 0; i < returns.size(); ++i) {
            std::cout << "P" << i << ": " << (int)returns[i] << "  ";
          }
          std::cout << "\n";
        }
        std::cout << "\n" << std::flush;
      }
      std::cout << ">>> Round " << current_round << " Start\n" << std::flush;
      last_round_tracker = current_round;
    }

    // --- TENSOR PREPARATION ---
    // Use InformationStateTensor to match the dynamic tensor shape
    std::vector<float> obs_tensor = state->InformationStateTensor();
    torch::Tensor input_tensor = torch::from_blob(obs_tensor.data(), {1, obs_size}, torch::kFloat).to(device);

    // --- QUERY THE NETWORK ---
    auto output = inference_model->forward(input_tensor);

    // Extract Policy (Action Probabilities)
    std::vector<Action> legal_actions = state->LegalActions();
    torch::Tensor policy_logits = output.logits.squeeze(0); // Remains on the model's device

    std::vector<int64_t> legal_indices;
    legal_indices.reserve(legal_actions.size());
    for (Action a : legal_actions) {
      legal_indices.push_back(static_cast<int64_t>(a));
    }
    torch::Tensor legal_index_tensor = torch::tensor(
        legal_indices, torch::TensorOptions().dtype(torch::kInt64).device(device));
    torch::Tensor legal_mean = policy_logits.index_select(0, legal_index_tensor).mean();
    policy_logits = policy_logits - legal_mean;
    constexpr float kEvaluatorLogitCap = 10.0f;
    policy_logits = kEvaluatorLogitCap * torch::tanh(policy_logits / kEvaluatorLogitCap);
    
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

    const auto* dune_state_before = dynamic_cast<const dune_imperium::DuneImperiumState*>(state.get());
    BoardSnapshot old_snap = GetBoardSnapshot(dune_state_before);

    state->ApplyAction(chosen_action);

    const auto* dune_state_after = dynamic_cast<const dune_imperium::DuneImperiumState*>(state.get());
    BoardSnapshot new_snap = GetBoardSnapshot(dune_state_after);
    CompareSnapshotsAndLog(old_snap, new_snap, clean_act);

    // Check if a graft partner was automatically or interactively selected
    if (chosen_action >= dune_imperium::kActionSelectAgentCard0 &&
        chosen_action < dune_imperium::kActionSelectAgentCard0 + 256) {
      int card_id = static_cast<int>(chosen_action - dune_imperium::kActionSelectAgentCard0);
      if (dune_state_after) {
        int partner_id = dune_state_after->GraftedPartnerCardForTesting(current_player, card_id);
        if (partner_id != dune_imperium::kInvalidCard) {
          const auto* partner_card = dune_imperium::FindImperiumCardById(partner_id);
          if (partner_card) {
            std::cout << "[Player P" << current_player << "] grafts " << partner_card->name << "\n" << std::flush;
          }
        }
      }
    }
    
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
    dune_imperium::DuneImperiumState* mutable_state = const_cast<dune_imperium::DuneImperiumState*>(dune_state);

    std::cout << "Final Research Columns: ";
    for (int i = 0; i < 4; ++i) {
      std::cout << "P" << i << ": " << dune_state->GetResearchBottomColForTesting(i) << "  ";
    }
    std::cout << "\nFinal Research Rows: ";
    for (int i = 0; i < 4; ++i) {
      std::cout << "P" << i << ": " << dune_state->GetResearchBottomRowForTesting(i) << "  ";
    }
    std::cout << "\nFinal Tleilaxu Track: ";
    for (int i = 0; i < 4; ++i) {
      std::cout << "P" << i << ": " << dune_state->GetTleilaxuTrackForTesting(i) << "  ";
    }

    std::cout << "\n\n=== Endgame scoring breakdown ===\n";
    std::vector<int> true_final_vp(4, 0);
    for (int p = 0; p < 4; ++p) {
      int base_vp = dune_state->GetPlayerVpForTesting(p);
      int endgame_vp = 0;
      int endgame_spice = 0;
      std::cout << "Player P" << p << ":\n";
      std::cout << "  Base In-Game VPs: " << base_vp << "\n";

      // 1. Endgame Intrigues
      const auto& hand = dune_state->GetIntrigueHandForTesting(p);
      if (!hand.empty()) {
        for (int intrigue_id : hand) {
          const auto* intrigue = dune_imperium::FindIntrigueCardById(intrigue_id);
          if (intrigue && (intrigue->phase_mask & dune_imperium::kIntriguePhaseEndgameMask) != 0) {
            int vp_bonus = dune_state->EndgameIntrigueVpBonusForTesting(p, intrigue_id);
            int spice_bonus = dune_state->EndgameIntrigueSpiceBonusForTesting(p, intrigue_id);
            endgame_vp += vp_bonus;
            endgame_spice += spice_bonus;
            std::cout << "  - Endgame Intrigue: " << intrigue->name << " -> Scored " << vp_bonus << " VP";
            if (spice_bonus > 0) std::cout << " and " << spice_bonus << " Spice";
            std::cout << "\n";

            // Explain requirements for the ones in play
            if (intrigue_id == dune_imperium::kIntriguePlansWithinPlans) {
              int count_3 = 0;
              for (int f = 0; f < 4; ++f) {
                if (dune_state->GetPlayerInfluenceForTesting(p, static_cast<dune_imperium::Faction>(f)) >= 3) {
                  count_3++;
                }
              }
              std::cout << "      [Plans Within Plans] Gained " << vp_bonus << " VP because player had 3+ influence with " << count_3 << " factions (needs 3 factions for 1 VP, 4 factions for 2 VP).\n";
            } else if (intrigue_id == dune_imperium::kIntrigueCornerTheMarket) {
              int my_tsmf = 0;
              auto count_tsmf_cards = [&](const std::vector<int>& cards) {
                for (int id : cards) {
                  if (id == dune_imperium::kCardTheSpiceMustFlow) my_tsmf++;
                }
              };
              count_tsmf_cards(mutable_state->GetPlayerDrawDeckForTesting(p));
              count_tsmf_cards(mutable_state->GetPlayerDiscardForTesting(p));
              count_tsmf_cards(mutable_state->GetPlayerHandForTesting(p));
              count_tsmf_cards(dune_state->GetPlayedAgentCardsForTesting(p));
              count_tsmf_cards(dune_state->GetRevealedCardsForTesting(p));
              std::cout << "      [Corner The Market] Gained " << vp_bonus << " VP because player owned " << my_tsmf << " The Spice Must Flow cards (needs >=2 for 1 VP, strictly more than anyone else for +1 VP).\n";
            }
          }
        }
      }

      // 2. Tech tile 6: Holtzman Engine
      const auto& tech_tiles = dune_state->GetPlayerTechTilesForTesting(p);
      if (std::find(tech_tiles.begin(), tech_tiles.end(), 6) != tech_tiles.end()) {
        int tsmf_count = 0;
        auto count_tsmf = [&](const std::vector<int>& cards) {
          for (int id : cards) {
            if (id == dune_imperium::kCardTheSpiceMustFlow) tsmf_count++;
          }
        };
        count_tsmf(mutable_state->GetPlayerDrawDeckForTesting(p));
        count_tsmf(mutable_state->GetPlayerDiscardForTesting(p));
        count_tsmf(mutable_state->GetPlayerHandForTesting(p));
        count_tsmf(dune_state->GetPlayedAgentCardsForTesting(p));
        count_tsmf(dune_state->GetRevealedCardsForTesting(p));
        if (tsmf_count >= 2) {
          endgame_vp += 1;
          std::cout << "  - Tech Tile Holtzman Engine (TSMF bonus) -> Scored 1 VP (owned " << tsmf_count << " The Spice Must Flow cards)\n";
        }
      }

      // 3. Faction Influence milestone (>=3 in all 4 factions)
      bool all_3 = true;
      for (int f = 0; f < 4; ++f) {
        if (dune_state->GetPlayerInfluenceForTesting(p, static_cast<dune_imperium::Faction>(f)) < 3) {
          all_3 = false;
        }
      }
      if (all_3) {
        endgame_vp += 1;
        std::cout << "  - Faction Influence milestone (>=3 in all 4 factions) -> Scored 1 VP\n";
      }

      // 4. Tech tile 14: Spy Satellites
      if (std::find(tech_tiles.begin(), tech_tiles.end(), 14) != tech_tiles.end()) {
        int low_influence_factions = 0;
        for (int f = 0; f < 4; ++f) {
          if (dune_state->GetPlayerInfluenceForTesting(p, static_cast<dune_imperium::Faction>(f)) <= 1) {
            low_influence_factions++;
          }
        }
        if (low_influence_factions > 0) {
          endgame_vp += low_influence_factions;
          std::cout << "  - Tech Tile Spy Satellites (Low influence bonus) -> Scored " << low_influence_factions << " VP (had <=1 influence with " << low_influence_factions << " factions)\n";
        }
      }

      true_final_vp[p] = base_vp + endgame_vp;
      std::cout << "  True Final Victory Points: " << true_final_vp[p] << "\n";
    }

    std::cout << "\nFinal Victory Points (including Endgame): ";
    for (int i = 0; i < 4; ++i) {
      std::cout << "P" << i << ": " << true_final_vp[i] << "  ";
    }

    std::cout << "\nFinal Intrigue Hands:\n";
    for (int i = 0; i < 4; ++i) {
      std::cout << "  P" << i << ": ";
      const auto& hand = dune_state->GetIntrigueHandForTesting(i);
      if (hand.empty()) {
        std::cout << "(empty)\n";
      } else {
        for (size_t c = 0; c < hand.size(); ++c) {
          if (c > 0) std::cout << ", ";
          const auto* intrigue = dune_imperium::FindIntrigueCardById(hand[c]);
          if (intrigue) {
            std::cout << intrigue->name;
            if ((intrigue->phase_mask & dune_imperium::kIntriguePhaseEndgameMask) != 0) {
              std::cout << " [Endgame]";
            }
          } else {
            std::cout << "UnknownCard(" << hand[c] << ")";
          }
        }
        std::cout << "\n";
      }
    }
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
