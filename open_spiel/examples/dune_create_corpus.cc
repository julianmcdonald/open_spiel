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
#include "dune_search_routing.h"
#include "dune_sha256.h"

ABSL_FLAG(std::string, model_checkpoint, "artifacts/branch_a_frozen/branch_a_seed11_model_update_2450.pt", "Path to the model checkpoint");
ABSL_FLAG(std::string, output_path, "data/dune_diagnostic_corpus.json", "Output JSON path for the corpus");
ABSL_FLAG(std::string, legacy_corpus_path, "", "Path to legacy corpus to retain strategic states from");
ABSL_FLAG(int, seed, 42, "Base seed");
ABSL_FLAG(int, hidden_dim, 2048, "Hidden dimension");
ABSL_FLAG(int, num_blocks, 8, "Block count");
ABSL_FLAG(int, threads, 16, "Number of threads");
ABSL_FLAG(bool, nonlinear_value_head, false,
          "Use the versioned nonlinear value-head architecture.");

namespace {
using namespace open_spiel;
using namespace open_spiel::dune_imperium;

bool SpaceIsReachable(State *state, Action space_action) {
  auto legal = state->LegalActions();
  if (std::find(legal.begin(), legal.end(), space_action) != legal.end()) {
    return true;
  }

  for (Action a : legal) {
    if (a >= dune_imperium::kActionSelectAgentCard0 &&
        a < dune_imperium::kActionSelectAgentCard0 + 256) {
      std::unique_ptr<State> clone = state->Clone();
      clone->ApplyAction(a);

      // Drain forced nodes using the exact logic of DrainForcedNodesForPlanner
      int safety = 0;
      while (!clone->IsTerminal()) {
        ++safety;
        if (safety > 150) break;
        if (clone->IsChanceNode()) {
          auto outcomes = clone->ChanceOutcomes();
          if (outcomes.empty()) break;
          clone->ApplyAction(outcomes.front().first);
          continue;
        }
        auto clone_legal = clone->LegalActions();
        if (clone_legal.empty()) break;
        if (clone_legal.size() == 1) {
          clone->ApplyAction(clone_legal[0]);
          continue;
        }
        if (clone_legal.size() > 1) {
          bool has_ack = false;
          for (Action act : clone_legal) {
            if (act == dune_imperium::kActionAcknowledgeChance) {
              clone->ApplyAction(act);
              has_ack = true;
              break;
            }
          }
          if (has_ack) continue;
        }
        break;
      }

      auto clone_legal = clone->LegalActions();
      if (std::find(clone_legal.begin(), clone_legal.end(), space_action) != clone_legal.end()) {
        return true;
      }
    }
  }
  return false;
}
} // namespace

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
  std::vector<Action> legal_actions;
  DuneDecisionRole role;

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

std::string ComputeHistoryHash(const std::vector<Action>& history) {
  open_spiel::SHA256 sha;
  std::ostringstream oss;
  for (size_t i = 0; i < history.size(); ++i) {
    if (i > 0) oss << ",";
    oss << history[i];
  }
  sha.Update(oss.str());
  return sha.Final();
}

std::string DecisionRoleToString(DuneDecisionRole role) {
  switch (role) {
    case DuneDecisionRole::kForcedOrBookkeeping: return "FORCED_OR_BOOKKEEPING";
    case DuneDecisionRole::kLeaderSelection: return "LEADER_SELECTION";
    case DuneDecisionRole::kAgentPrimary: return "AGENT_PRIMARY";
    case DuneDecisionRole::kAgentContinuation: return "AGENT_CONTINUATION";
    case DuneDecisionRole::kPurchase: return "PURCHASE";
    case DuneDecisionRole::kCombatIntrigue: return "COMBAT_INTRIGUE";
    case DuneDecisionRole::kOtherOptional: return "OTHER_OPTIONAL";
    default: return "UNKNOWN";
  }
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
      obs_size, absl::GetFlag(FLAGS_hidden_dim), action_size,
      absl::GetFlag(FLAGS_num_blocks),
      absl::GetFlag(FLAGS_nonlinear_value_head));
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

      // Part A: Collect strategic and opportunity candidates from raw-policy games
      {
        std::mt19937 rng(game_seed);
        auto state = game->NewInitialState();
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
              cs.legal_actions = state->LegalActions();
              cs.role = ClassifyDuneDecisionRole(*state, player, false);
              thread_candidates[thread_id].push_back(cs);
            }

            // Collect opportunity
            bool has_sm = dune_state && dune_state->HasSwordmaster(player);
            int solari = dune_state ? dune_state->GetPlayerSolari(player) : 0;

            bool is_agent_primary = ClassifyDuneDecisionRole(*state, player, false) == DuneDecisionRole::kAgentPrimary;

            if (is_agent_primary && round >= 2 && round <= 5 && !has_sm && solari >= 8 &&
                SpaceIsReachable(state.get(), dune_imperium::kActionAgentSpaceSwordmaster)) {
              CandidateState cs;
              cs.category = "opportunity";
              cs.player = player;
              cs.round = round;
              cs.history = state->History();
              cs.observation = state->InformationStateTensor(player);
              cs.info_state_str = state->InformationStateString(player);
              cs.seed = game_seed;
              cs.decision_index = step_count;
              cs.legal_actions = state->LegalActions();
              cs.role = DuneDecisionRole::kAgentPrimary;
              thread_candidates[thread_id].push_back(cs);
            }

            ActionsAndProbs prior = thread_eval->Prior(*state);
            Action action = SampleActionFromPrior(prior, rng);
            state->ApplyAction(action);
          }
        }
      }

      // Part B: Collect planner candidates from guided heuristic games
      {
        std::mt19937 rng(game_seed + 50000);
        auto state = game->NewInitialState();
        Player guided_player = game_seed % 4;
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
            auto legal = state->LegalActions();

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
              cs.legal_actions = legal;
              cs.role = ClassifyDuneDecisionRole(*state, player, false);
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

  std::string legacy_corpus_path = absl::GetFlag(FLAGS_legacy_corpus_path);
  if (!legacy_corpus_path.empty()) {
    std::cout << "Loading strategic states from legacy corpus: " << legacy_corpus_path << "\n";
    std::ifstream lf(legacy_corpus_path);
    if (!lf.good()) {
      std::cerr << "Failed to open legacy corpus: " << legacy_corpus_path << "\n";
      return 1;
    }
    std::string content((std::istreambuf_iterator<char>(lf)), std::istreambuf_iterator<char>());
    auto json_parsed = open_spiel::json::FromString(content);
    if (!json_parsed) {
      std::cerr << "Failed to parse legacy corpus from: " << legacy_corpus_path << "\n";
      return 1;
    }
    auto json_arr = json_parsed.value().GetArray();
    for (const auto& item_val : json_arr) {
      auto item = item_val.GetObject();
      if (item.at("category").GetString() == "strategic") {
        CandidateState cs;
        cs.category = "strategic";
        cs.player = static_cast<Player>(item.at("player").GetInt());
        cs.round = static_cast<int>(item.at("round").GetInt());

        auto history_arr = item.at("history").GetArray();
        for (const auto& act_val : history_arr) {
          cs.history.push_back(static_cast<Action>(act_val.GetInt()));
        }

        auto obs_arr = item.at("observation").GetArray();
        for (const auto& val : obs_arr) {
          cs.observation.push_back(static_cast<float>(val.GetDouble()));
        }

        // Reconstruct state to compute info_state_str and legal_actions
        auto state = ReconstructState(game, cs.history);
        cs.info_state_str = state->InformationStateString(cs.player);
        cs.legal_actions = state->LegalActions();
        cs.role = ClassifyDuneDecisionRole(*state, cs.player, false);

        if (item.find("source_seed") != item.end()) {
          cs.seed = static_cast<int>(item.at("source_seed").GetInt());
        } else {
          cs.seed = seed;
        }
        if (item.find("decision_index") != item.end()) {
          cs.decision_index = static_cast<int>(item.at("decision_index").GetInt());
        } else {
          cs.decision_index = 0;
        }

        selected_strategic.push_back(cs);
        unique_state_keys.insert(cs.info_state_str);
      }
    }
    std::cout << "Loaded " << selected_strategic.size() << " strategic states from legacy corpus.\n";
  }

  // 1. Stratify Strategic (Target 64)
  size_t strategic_target = 64;
  if (selected_strategic.empty()) {
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
  }

  // 2. Stratify Opportunity (Target 32, exactly 8 per player seat)
  std::vector<size_t> opp_ptrs(4, 0);
  for (int b = 0; b < 4; ++b) {
    size_t chosen_from_bucket = 0;
    while (chosen_from_bucket < 8 && opp_ptrs[b] < opportunity_buckets[b].size()) {
      const auto& cand = opportunity_buckets[b][opp_ptrs[b]++];
      if (unique_state_keys.find(cand.info_state_str) == unique_state_keys.end()) {
        unique_state_keys.insert(cand.info_state_str);
        selected_opportunity.push_back(cand);
        chosen_from_bucket++;
      }
    }
    SPIEL_CHECK_EQ(chosen_from_bucket, 8);
  }

  // 3. Stratify Planner (Target 32, exactly 8 per player seat, prioritizing PURCHASE/COMBAT roles)
  int strategic_purchase = 0;
  int strategic_combat = 0;
  for (const auto& cs : selected_strategic) {
    auto state = ReconstructState(game, cs.history);
    DuneDecisionRole role = ClassifyDuneDecisionRole(*state, cs.player, false);
    if (role == DuneDecisionRole::kPurchase) {
      strategic_purchase++;
    } else if (role == DuneDecisionRole::kCombatIntrigue) {
      strategic_combat++;
    }
  }
  int needed_purchase = std::max(0, 3 - strategic_purchase);
  int needed_combat = std::max(0, 9 - strategic_combat);
  std::cout << "Strategic roles: purchase=" << strategic_purchase << ", combat=" << strategic_combat << "\n";
  std::cout << "Needed from planner: purchase=" << needed_purchase << ", combat=" << needed_combat << "\n";

  for (int b = 0; b < 4; ++b) {
    size_t chosen_from_bucket = 0;
    std::vector<bool> candidate_used(planner_buckets[b].size(), false);

    // Pass 1: Prioritize PURCHASE and COMBAT_INTRIGUE roles
    for (size_t i = 0; i < planner_buckets[b].size() && chosen_from_bucket < 8; ++i) {
      const auto& cand = planner_buckets[b][i];
      if (unique_state_keys.find(cand.info_state_str) != unique_state_keys.end()) {
        continue;
      }
      auto state = ReconstructState(game, cand.history);
      DuneDecisionRole role = ClassifyDuneDecisionRole(*state, cand.player, false);

      bool select = false;
      if (needed_purchase > 0 && role == DuneDecisionRole::kPurchase) {
        select = true;
        needed_purchase--;
      } else if (needed_combat > 0 && role == DuneDecisionRole::kCombatIntrigue) {
        select = true;
        needed_combat--;
      }

      if (select) {
        unique_state_keys.insert(cand.info_state_str);
        selected_planner.push_back(cand);
        candidate_used[i] = true;
        chosen_from_bucket++;
      }
    }

    // Pass 2: Fill the rest of the 8 slots round-robin
    for (size_t i = 0; i < planner_buckets[b].size() && chosen_from_bucket < 8; ++i) {
      if (candidate_used[i]) continue;
      const auto& cand = planner_buckets[b][i];
      if (unique_state_keys.find(cand.info_state_str) == unique_state_keys.end()) {
        unique_state_keys.insert(cand.info_state_str);
        selected_planner.push_back(cand);
        chosen_from_bucket++;
      }
    }
    SPIEL_CHECK_EQ(chosen_from_bucket, 8);
  }

  if (selected_strategic.size() < strategic_target) {
    std::cerr << absl::StrFormat("Error: Stratified strategic selection short of target: %d vs %d\n", selected_strategic.size(), strategic_target);
    return 1;
  }
  if (selected_opportunity.size() < 32) {
    std::cerr << absl::StrFormat("Error: Stratified opportunity selection short of target: %d vs 32\n", selected_opportunity.size());
    return 1;
  }
  if (selected_planner.size() < 32) {
    std::cerr << absl::StrFormat("Error: Stratified planner selection short of target: %d vs 32\n", selected_planner.size());
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

    std::vector<Action> rec_legal = rec_state->LegalActions();
    SPIEL_CHECK_EQ(rec_legal.size(), cand.legal_actions.size());
    for (size_t i = 0; i < rec_legal.size(); ++i) {
      SPIEL_CHECK_EQ(rec_legal[i], cand.legal_actions[i]);
    }

    // Corpus Verification Gate: verify role
    DuneDecisionRole role = ClassifyDuneDecisionRole(*rec_state, cand.player, false);
    SPIEL_CHECK_TRUE(role == cand.role);
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

    open_spiel::json::Array legal_arr;
    for (Action a : cand.legal_actions) legal_arr.push_back(static_cast<int64_t>(a));
    obj["legal_actions"] = legal_arr;

    // Provenance fields
    obj["source_seed"] = static_cast<int64_t>(cand.seed);
    obj["episode_id"] = static_cast<int64_t>(cand.seed - seed); // Seed maps to episode offset from base seed
    obj["seat"] = static_cast<int64_t>(cand.player);
    obj["decision_index"] = static_cast<int64_t>(cand.decision_index);

    // Reconstruct state to get role and perform self-check
    auto rec_state = ReconstructState(game, cand.history);
    if (rec_state->CurrentPlayer() != cand.player) {
      std::cerr << "Self-check failed: player mismatch on reconstruction.\n";
      std::exit(1);
    }
    if (rec_state->LegalActions() != cand.legal_actions) {
      std::cerr << "Self-check failed: legal actions mismatch on reconstruction.\n";
      std::exit(1);
    }
    const auto& rec_obs = rec_state->InformationStateTensor(cand.player);
    if (rec_obs.size() != cand.observation.size()) {
      std::cerr << "Self-check failed: observation size mismatch on reconstruction.\n";
      std::exit(1);
    }
    for (size_t idx = 0; idx < rec_obs.size(); ++idx) {
      if (std::abs(rec_obs[idx] - cand.observation[idx]) > 1e-5) {
        std::cerr << "Self-check failed: observation value mismatch on reconstruction at index " << idx << ".\n";
        std::exit(1);
      }
    }
    DuneDecisionRole role = ClassifyDuneDecisionRole(*rec_state, cand.player, false);
    if (role != cand.role) {
      std::cerr << "Self-check failed: role mismatch on reconstruction.\n";
      std::exit(1);
    }
    obj["role"] = DecisionRoleToString(role);
    obj["history_hash"] = ComputeHistoryHash(cand.history);
    obj["corpus_schema_version"] = "v2";

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
