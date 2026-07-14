#include <iostream>
#include <fstream>
#include <memory>
#include <vector>
#include <string>
#include <algorithm>
#include <filesystem>
#include <random>
#include <set>
#include <mutex>
#include <thread>
#include <atomic>

#include "open_spiel/spiel.h"
#include "open_spiel/games/dune_imperium/dune_imperium.h"
#include "open_spiel/games/dune_imperium/dune_imperium_common.h"
#include "open_spiel/games/dune_imperium/dune_imperium_board.h"
#include "open_spiel/abseil-cpp/absl/flags/flag.h"
#include "open_spiel/abseil-cpp/absl/flags/parse.h"
#include "open_spiel/abseil-cpp/absl/strings/str_format.h"
#include "open_spiel/abseil-cpp/absl/random/distributions.h"
#include "open_spiel/utils/json.h"
#include <torch/torch.h>

#include "dune_network.h"
#include "dune_evaluator.h"
#include "dune_puct_is_mcts.h"
#include "dune_warmstart_helpers.h"

ABSL_FLAG(std::string, model_checkpoint, "artifacts/branch_a_frozen/branch_a_seed11_model_update_2450.pt", "Path to the model checkpoint");
ABSL_FLAG(std::string, output_path, "data/dune_diagnostic_corpus.json", "Output JSON path for the corpus");
ABSL_FLAG(int, seed, 42, "Base seed");
ABSL_FLAG(int, hidden_dim, 2048, "Hidden dimension");
ABSL_FLAG(int, num_blocks, 8, "Block count");
ABSL_FLAG(int, threads, 16, "Number of threads");

using namespace open_spiel;

// Struct to hold collected candidate states
struct CandidateState {
  std::string category;
  Player player;
  int round;
  std::vector<Action> history;
  std::vector<float> observation;
  std::string info_state_str;
  int seed;
  int decision_index;
  
  // Sort deterministically to avoid scheduling non-determinism
  bool operator<(const CandidateState& o) const {
    if (category != o.category) return category < o.category;
    if (seed != o.seed) return seed < o.seed;
    return decision_index < o.decision_index;
  }
};

// Helper to sample action stochastically from prior
Action SampleActionFromPrior(const ActionsAndProbs& prior, std::mt19937& rng) {
  if (prior.empty()) return kInvalidAction;
  std::vector<double> weights;
  weights.reserve(prior.size());
  for (const auto& ap : prior) {
    weights.push_back(ap.second);
  }
  std::discrete_distribution<size_t> dist(weights.begin(), weights.end());
  return prior[dist(rng)].first;
}

// Replay state history to reconstruct state
std::unique_ptr<State> ReconstructState(const std::shared_ptr<const Game>& game, const std::vector<Action>& history) {
  auto state = game->NewInitialState();
  for (Action action : history) {
    state->ApplyAction(action);
  }
  return state;
}

int main(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);
  at::set_num_threads(1);
  
  std::string model_checkpoint = absl::GetFlag(FLAGS_model_checkpoint);
  std::string output_path = absl::GetFlag(FLAGS_output_path);
  int seed = absl::GetFlag(FLAGS_seed);
  int num_threads = absl::GetFlag(FLAGS_threads);
  
  auto game = open_spiel::LoadGame("dune_imperium");
  int64_t obs_size = game->GetType().provides_information_state_tensor
                         ? game->InformationStateTensorSize()
                         : game->ObservationTensorSize();
  int64_t action_size = game->NumDistinctActions();
  
  torch::Device device = torch::cuda::is_available() ? torch::Device(torch::kCUDA) : torch::Device(torch::kCPU);
  std::cout << "Using device: " << (device.is_cuda() ? "CUDA" : "CPU") << "\n";
  auto model = std::make_shared<SharedDunePolicyValueNetImpl>(
      obs_size, absl::GetFlag(FLAGS_hidden_dim), action_size, absl::GetFlag(FLAGS_num_blocks));
  try {
    torch::load(model, model_checkpoint, device);
  } catch (const std::exception& e) {
    std::cerr << "Failed to load model checkpoint: " << e.what() << "\n";
    return 1;
  }
  model->to(device);
  model->eval();
  
  std::cout << absl::StrFormat("Starting parallel corpus generation on %d threads...\n", num_threads);
  
  std::vector<std::vector<CandidateState>> thread_candidates(num_threads);
  
  auto worker_fn = [&](int thread_id) {
    auto thread_eval = std::make_shared<DuneNNEvaluator>(model, device);
    int total_games = 1000;
    int games_per_thread = total_games / num_threads;
    int start_game = thread_id * games_per_thread;
    int end_game = (thread_id == num_threads - 1) ? total_games : (thread_id + 1) * games_per_thread;
    
    for (int g = start_game; g < end_game; ++g) {
      int game_seed = seed + g;
      std::mt19937 rng(game_seed);
      auto state = game->NewInitialState();
      Player guided_player = game_seed % 4; // Rotate seat ownership deterministically
      
      int step_count = 0;
      while (!state->IsTerminal()) {
        step_count++;
        if (state->IsChanceNode()) {
          auto outcomes = state->ChanceOutcomes();
          Action choice = SampleAction(outcomes, absl::Uniform(rng, 0.0, 1.0)).first;
          state->ApplyAction(choice);
        } else if (state->CurrentPlayer() == kSimultaneousPlayerId) {
          std::vector<Action> joint_action;
          for (int p = 0; p < game->NumPlayers(); ++p) {
            auto actions = state->LegalActions(p);
            std::uniform_int_distribution<int> dis(0, actions.size() - 1);
            joint_action.push_back(actions[dis(rng)]);
          }
          state->ApplyActions(joint_action);
        } else {
          Player player = state->CurrentPlayer();
          const auto* dune_state = dynamic_cast<const dune_imperium::DuneImperiumState*>(state.get());
          int round = dune_state ? dune_state->GetCurrentRound() : 1;
          bool is_strategic = IsStrategicState(*state, player);
          
          // Collect strategic
          if (is_strategic && round <= 5) {
            CandidateState cs;
            cs.category = "strategic";
            cs.player = player;
            cs.round = round;
            cs.history = state->History();
            cs.observation = state->InformationStateTensor(player);
            cs.info_state_str = state->InformationStateString(player);
            cs.seed = game_seed;
            cs.decision_index = step_count;
            thread_candidates[thread_id].push_back(cs);
          }
          
          // Collect opportunity
          bool has_sm = dune_state && dune_state->HasSwordmaster(player);
          int solari = dune_state ? dune_state->GetPlayerSolari(player) : 0;
          auto legal = state->LegalActions();
          bool sm_is_legal = std::find(legal.begin(), legal.end(), dune_imperium::kActionAgentSpaceSwordmaster) != legal.end();
          if (is_strategic && round >= 2 && round <= 5 && !has_sm && solari >= 6 && sm_is_legal) {
            CandidateState cs;
            cs.category = "opportunity";
            cs.player = player;
            cs.round = round;
            cs.history = state->History();
            cs.observation = state->InformationStateTensor(player);
            cs.info_state_str = state->InformationStateString(player);
            cs.seed = game_seed;
            cs.decision_index = step_count;
            thread_candidates[thread_id].push_back(cs);
          }
          
          // Collect planner
          if (player == guided_player && is_strategic && round >= 2 && round <= 5 &&
              dune_state && !dune_state->HasSwordmaster(guided_player)) {
            CandidateState cs;
            cs.category = "planner";
            cs.player = player;
            cs.round = round;
            cs.history = state->History();
            cs.observation = state->InformationStateTensor(player);
            cs.info_state_str = state->InformationStateString(player);
            cs.seed = game_seed;
            cs.decision_index = step_count;
            thread_candidates[thread_id].push_back(cs);
          }
          
          Action action = kInvalidAction;
          if (player == guided_player) {
            action = ChooseHeuristicAcquisitionAction(*state, legal, guided_player, &rng);
          } else {
            ActionsAndProbs prior = thread_eval->Prior(*state);
            action = SampleActionFromPrior(prior, rng);
          }
          state->ApplyAction(action);
        }
      }
    }
  };
  
  std::vector<std::thread> workers;
  for (int i = 0; i < num_threads; ++i) {
    workers.emplace_back(worker_fn, i);
  }
  for (auto& w : workers) {
    w.join();
  }
  
  // Merge candidates across threads
  std::vector<CandidateState> all_candidates;
  for (const auto& tc : thread_candidates) {
    all_candidates.insert(all_candidates.end(), tc.begin(), tc.end());
  }
  
  // Sort deterministically
  std::sort(all_candidates.begin(), all_candidates.end());
  
  // Buckets for round-robin stratification
  std::vector<std::vector<CandidateState>> strategic_buckets(20);
  std::vector<std::vector<CandidateState>> opportunity_buckets(4);
  std::vector<std::vector<CandidateState>> planner_buckets(4);
  
  for (const auto& c : all_candidates) {
    if (c.category == "strategic") {
      int b_idx = (c.round - 1) * 4 + c.player;
      strategic_buckets[b_idx].push_back(c);
    } else if (c.category == "opportunity") {
      opportunity_buckets[c.player].push_back(c);
    } else if (c.category == "planner") {
      planner_buckets[c.player].push_back(c);
    }
  }
  
  std::set<std::string> unique_state_keys;
  std::vector<CandidateState> selected_strategic;
  std::vector<CandidateState> selected_opportunity;
  std::vector<CandidateState> selected_planner;
  
  // 1. Stratify Strategic (Target 64)
  size_t strategic_target = 64;
  std::vector<size_t> strat_ptrs(20, 0);
  bool progress = true;
  while (selected_strategic.size() < strategic_target && progress) {
    progress = false;
    for (int b = 0; b < 20; ++b) {
      if (selected_strategic.size() >= strategic_target) break;
      while (strat_ptrs[b] < strategic_buckets[b].size()) {
        const auto& cand = strategic_buckets[b][strat_ptrs[b]++];
        if (unique_state_keys.find(cand.info_state_str) == unique_state_keys.end()) {
          unique_state_keys.insert(cand.info_state_str);
          selected_strategic.push_back(cand);
          progress = true;
          break;
        }
      }
    }
  }
  
  // 2. Stratify Opportunity (Target 32)
  size_t opportunity_target = 32;
  std::vector<size_t> opp_ptrs(4, 0);
  progress = true;
  while (selected_opportunity.size() < opportunity_target && progress) {
    progress = false;
    for (int b = 0; b < 4; ++b) {
      if (selected_opportunity.size() >= opportunity_target) break;
      while (opp_ptrs[b] < opportunity_buckets[b].size()) {
        const auto& cand = opportunity_buckets[b][opp_ptrs[b]++];
        if (unique_state_keys.find(cand.info_state_str) == unique_state_keys.end()) {
          unique_state_keys.insert(cand.info_state_str);
          selected_opportunity.push_back(cand);
          progress = true;
          break;
        }
      }
    }
  }
  
  // 3. Stratify Planner (Target 32)
  size_t planner_target = 32;
  std::vector<size_t> plan_ptrs(4, 0);
  progress = true;
  while (selected_planner.size() < planner_target && progress) {
    progress = false;
    for (int b = 0; b < 4; ++b) {
      if (selected_planner.size() >= planner_target) break;
      while (plan_ptrs[b] < planner_buckets[b].size()) {
        const auto& cand = planner_buckets[b][plan_ptrs[b]++];
        if (unique_state_keys.find(cand.info_state_str) == unique_state_keys.end()) {
          unique_state_keys.insert(cand.info_state_str);
          selected_planner.push_back(cand);
          progress = true;
          break;
        }
      }
    }
  }
  
  if (selected_strategic.size() < strategic_target) {
    std::cerr << absl::StrFormat("Error: Stratified strategic selection short of target: %d vs %d\n", selected_strategic.size(), strategic_target);
    return 1;
  }
  if (selected_opportunity.size() < opportunity_target) {
    std::cerr << absl::StrFormat("Error: Stratified opportunity selection short of target: %d vs %d\n", selected_opportunity.size(), opportunity_target);
    return 1;
  }
  if (selected_planner.size() < planner_target) {
    std::cerr << absl::StrFormat("Error: Stratified planner selection short of target: %d vs %d\n", selected_planner.size(), planner_target);
    return 1;
  }
  
  // Verify state integrity via reconstruction and history replay
  auto check_state = [&](const CandidateState& cand) {
    auto rec_state = ReconstructState(game, cand.history);
    SPIEL_CHECK_EQ(rec_state->CurrentPlayer(), cand.player);
    const auto* dune_rec = dynamic_cast<const dune_imperium::DuneImperiumState*>(rec_state.get());
    int rec_round = dune_rec ? dune_rec->GetCurrentRound() : 1;
    SPIEL_CHECK_EQ(rec_round, cand.round);
    std::vector<float> rec_obs = rec_state->InformationStateTensor(cand.player);
    SPIEL_CHECK_EQ(rec_obs.size(), cand.observation.size());
    for (size_t i = 0; i < rec_obs.size(); ++i) {
      SPIEL_CHECK_FLOAT_NEAR(rec_obs[i], cand.observation[i], 1e-5);
    }
  };
  
  for (const auto& cand : selected_strategic) check_state(cand);
  for (const auto& cand : selected_opportunity) check_state(cand);
  for (const auto& cand : selected_planner) check_state(cand);
  
  std::cout << "All corpus states validated successfully via replay.\n";
  
  // Save output JSON
  open_spiel::json::Array corpus_arr;
  auto add_to_corpus_arr = [&](const CandidateState& cand) {
    open_spiel::json::Object obj;
    obj["category"] = cand.category;
    obj["player"] = static_cast<int64_t>(cand.player);
    obj["round"] = static_cast<int64_t>(cand.round);
    
    open_spiel::json::Array hist_arr;
    for (Action a : cand.history) hist_arr.push_back(static_cast<int64_t>(a));
    obj["history"] = hist_arr;
    
    open_spiel::json::Array obs_arr;
    for (float v : cand.observation) obs_arr.push_back(v);
    obj["observation"] = obs_arr;
    
    corpus_arr.push_back(obj);
  };
  
  for (const auto& cand : selected_strategic) add_to_corpus_arr(cand);
  for (const auto& cand : selected_opportunity) add_to_corpus_arr(cand);
  for (const auto& cand : selected_planner) add_to_corpus_arr(cand);
  
  std::filesystem::create_directories(std::filesystem::path(output_path).parent_path());
  std::ofstream out(output_path);
  if (!out) {
    std::cerr << "Failed to open output path: " << output_path << "\n";
    return 1;
  }
  out << open_spiel::json::ToString(corpus_arr, true) << "\n";
  std::cout << "Successfully saved deterministic 128-state diagnostic corpus to: " << output_path << "\n";
  
  return 0;
}
