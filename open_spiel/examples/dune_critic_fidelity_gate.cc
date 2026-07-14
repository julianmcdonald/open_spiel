#include <iostream>
#include <fstream>
#include <memory>
#include <vector>
#include <string>
#include <algorithm>
#include <filesystem>
#include <random>
#include <cmath>
#include <atomic>
#include <thread>
#include <mutex>
#include <shared_mutex>
#include <limits>

#include "open_spiel/spiel.h"
#include "open_spiel/games/dune_imperium/dune_imperium.h"
#include "open_spiel/games/dune_imperium/dune_imperium_common.h"
#include "open_spiel/games/dune_imperium/dune_imperium_board.h"
#include "open_spiel/abseil-cpp/absl/flags/flag.h"
#include "open_spiel/abseil-cpp/absl/flags/parse.h"
#include "open_spiel/abseil-cpp/absl/strings/str_format.h"
#include "open_spiel/abseil-cpp/absl/strings/str_join.h"
#include "open_spiel/abseil-cpp/absl/random/distributions.h"
#include "open_spiel/utils/json.h"
#include <torch/torch.h>

#include "dune_network.h"
#include "dune_evaluator.h"
#include "dune_puct_is_mcts.h"
#include "dune_warmstart_helpers.h"
#include "dune_batched_evaluator.h"
#include "dune_sha256.h"

ABSL_FLAG(std::string, model_checkpoint, "artifacts/branch_a_frozen/branch_a_seed11_model_update_2450.pt", "Path to the model checkpoint");
ABSL_FLAG(std::string, corpus_path, "data/dune_diagnostic_corpus.json", "Path to the corpus JSON file");
ABSL_FLAG(int, seed, 42, "RNG seed");
ABSL_FLAG(int, hidden_dim, 2048, "Hidden dimension");
ABSL_FLAG(int, num_blocks, 8, "Block count");
ABSL_FLAG(int, max_simulations, 200, "MCTS simulations count");
ABSL_FLAG(double, puct_c, 0.3, "PUCT exploration constant");
ABSL_FLAG(double, utility_divisor, 4.0, "Utility divisor");
ABSL_FLAG(int, threads, 32, "Number of threads");
ABSL_FLAG(bool, mock_miscalibrated_critic, false, "Debug flag to artificially fail the critic checks");
ABSL_FLAG(bool, self_test, false, "Run quick self-test mode with MockEvaluator");
ABSL_FLAG(std::string, report_path, "", "Path to output a structured JSON report");

using namespace open_spiel;

struct CorpusState {
  std::string category;
  Player player;
  int round;
  std::vector<Action> history;
  std::vector<float> observation;
};

// Mock Evaluator for self-testing without model checkpoint loading
class MockEvaluator : public algorithms::Evaluator {
 public:
  std::vector<double> Evaluate(const State& state) override {
    return std::vector<double>(state.NumPlayers(), 0.0);
  }

  ActionsAndProbs Prior(const State& state) override {
    ActionsAndProbs prior;
    std::vector<Action> legal = state.LegalActions();
    if (legal.empty()) return {};
    double p = 1.0 / legal.size();
    for (Action a : legal) {
      prior.push_back({a, p});
    }
    return prior;
  }
};

// Tie-aware Spearman correlation calculation
double SpearmanCorrelation(const std::vector<double>& x, const std::vector<double>& y) {
  int n = x.size();
  if (n <= 1) return 0.0;
  
  std::vector<std::pair<double, int>> rx(n), ry(n);
  for (int i = 0; i < n; ++i) {
    rx[i] = {x[i], i};
    ry[i] = {y[i], i};
  }
  std::sort(rx.begin(), rx.end());
  std::sort(ry.begin(), ry.end());
  
  std::vector<double> ranks_x(n), ranks_y(n);
  for (int i = 0; i < n; ) {
    int j = i;
    while (j < n && rx[j].first == rx[i].first) j++;
    double rank = (i + 1 + j) / 2.0;
    for (int k = i; k < j; ++k) ranks_x[rx[k].second] = rank;
    i = j;
  }
  for (int i = 0; i < n; ) {
    int j = i;
    while (j < n && ry[j].first == ry[i].first) j++;
    double rank = (i + 1 + j) / 2.0;
    for (int k = i; k < j; ++k) ranks_y[ry[k].second] = rank;
    i = j;
  }
  
  double mean_rx = 0.0, mean_ry = 0.0;
  for (int i = 0; i < n; ++i) {
    mean_rx += ranks_x[i];
    mean_ry += ranks_y[i];
  }
  mean_rx /= n;
  mean_ry /= n;
  
  double num = 0.0, den_x = 0.0, den_y = 0.0;
  for (int i = 0; i < n; ++i) {
    double dx = ranks_x[i] - mean_rx;
    double dy = ranks_y[i] - mean_ry;
    num += dx * dy;
    den_x += dx * dx;
    den_y += dy * dy;
  }
  if (den_x == 0.0 || den_y == 0.0) return 0.0;
  return num / std::sqrt(den_x * den_y);
}

// Sample action from prior stochastically
Action SampleActionFromPrior(const ActionsAndProbs& prior, double r_val) {
  if (prior.empty()) return kInvalidAction;
  double sum = 0.0;
  for (const auto& ap : prior) {
    sum += ap.second;
    if (r_val <= sum) return ap.first;
  }
  return prior.back().first;
}

// Replay state history to reconstruct state
std::unique_ptr<State> ReconstructState(const std::shared_ptr<const Game>& game, const std::vector<Action>& history) {
  auto state = game->NewInitialState();
  for (Action action : history) {
    state->ApplyAction(action);
  }
  return state;
}

// Evaluate a single rollout using a generic algorithms::Evaluator pointer
double RunRawPolicyRollout(const State& start_state, Player owner, algorithms::Evaluator* evaluator, uint64_t seed, double utility_divisor) {
  std::mt19937 rng(seed);
  auto state = start_state.Clone();
  while (!state->IsTerminal()) {
    if (state->IsChanceNode()) {
      auto outcomes = state->ChanceOutcomes();
      Action choice = SampleAction(outcomes, absl::Uniform(rng, 0.0, 1.0)).first;
      state->ApplyAction(choice);
    } else if (state->CurrentPlayer() == kSimultaneousPlayerId) {
      std::vector<Action> joint_action;
      for (int p = 0; p < state->NumPlayers(); ++p) {
        auto actions = state->LegalActions(p);
        std::uniform_int_distribution<int> dis(0, actions.size() - 1);
        joint_action.push_back(actions[dis(rng)]);
      }
      state->ApplyActions(joint_action);
    } else {
      ActionsAndProbs prior = evaluator->Prior(*state);
      double r_num = absl::Uniform(rng, 0.0, 1.0);
      Action action = SampleActionFromPrior(prior, r_num);
      state->ApplyAction(action);
    }
  }
  return state->Returns()[owner] / utility_divisor;
}

int main(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);
  at::set_num_threads(1);
  
  std::string model_checkpoint = absl::GetFlag(FLAGS_model_checkpoint);
  std::string corpus_path = absl::GetFlag(FLAGS_corpus_path);
  int seed = absl::GetFlag(FLAGS_seed);
  int num_threads = absl::GetFlag(FLAGS_threads);
  double utility_divisor = absl::GetFlag(FLAGS_utility_divisor);
  bool mock_fail = absl::GetFlag(FLAGS_mock_miscalibrated_critic);
  bool self_test = absl::GetFlag(FLAGS_self_test);
  
  auto game = open_spiel::LoadGame("dune_imperium");
  int64_t obs_size = game->GetType().provides_information_state_tensor
                         ? game->InformationStateTensorSize()
                         : game->ObservationTensorSize();
  int64_t action_size = game->NumDistinctActions();
  
  std::shared_ptr<algorithms::Evaluator> global_evaluator;
  std::shared_ptr<open_spiel::BatchedEvaluator> batched_eval;
  std::shared_mutex model_mutex;
  
  if (self_test) {
    std::cout << "[Self Test] Bypassing model loading, using MockEvaluator\n";
    global_evaluator = std::make_shared<MockEvaluator>();
  } else {
    torch::Device device = torch::cuda::is_available() ? torch::Device(torch::kCUDA) : torch::Device(torch::kCPU);
    std::cout << "Using device: " << (device.is_cuda() ? "CUDA" : "CPU") << "\n";
    auto model = std::make_shared<SharedDunePolicyValueNetImpl>(
        obs_size, absl::GetFlag(FLAGS_hidden_dim), action_size, absl::GetFlag(FLAGS_num_blocks));
    try {
      torch::load(model, model_checkpoint, device);
    } catch (const std::exception& e) {
      std::cerr << "Failed to load model: " << e.what() << "\n";
      return 1;
    }
    model->to(device);
    model->eval();
    
    batched_eval = std::make_shared<open_spiel::BatchedEvaluator>(
        model, num_threads, /*timeout_ms=*/2, device, &model_mutex);
    global_evaluator = std::make_shared<open_spiel::BatchedNNEvaluator>(batched_eval);
  }
  
  std::ifstream f(corpus_path);
  if (!f) {
    std::cerr << "Failed to open corpus path: " << corpus_path << "\n";
    return 1;
  }
  
  std::string content((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
  auto json_parsed = open_spiel::json::FromString(content);
  if (!json_parsed) {
    std::cerr << "Failed to parse corpus JSON.\n";
    return 1;
  }
  
  auto json_arr = json_parsed.value().GetArray();
  std::vector<CorpusState> corpus;
  for (const auto& item_val : json_arr) {
    auto obj = item_val.GetObject();
    CorpusState cs;
    cs.category = obj.at("category").GetString();
    cs.player = static_cast<Player>(obj.at("player").GetInt());
    cs.round = obj.at("round").GetInt();
    
    auto hist_arr = obj.at("history").GetArray();
    for (const auto& a_val : hist_arr) {
      cs.history.push_back(static_cast<Action>(a_val.GetInt()));
    }
    
    auto obs_arr = obj.at("observation").GetArray();
    for (const auto& o_val : obs_arr) {
      cs.observation.push_back(static_cast<float>(o_val.GetDouble()));
    }
    corpus.push_back(cs);
  }
  
  std::cout << absl::StrFormat("Loaded diagnostic corpus of %d states.\n", corpus.size());
  
  std::vector<size_t> strategic_indices;
  std::vector<size_t> opportunity_indices;
  std::vector<size_t> planner_indices;
  
  for (size_t i = 0; i < corpus.size(); ++i) {
    if (corpus[i].category == "strategic") {
      strategic_indices.push_back(i);
    } else if (corpus[i].category == "opportunity") {
      opportunity_indices.push_back(i);
    } else if (corpus[i].category == "planner") {
      planner_indices.push_back(i);
    }
  }
  
  std::cout << absl::StrFormat("Found: %d strategic, %d opportunity, %d planner states.\n",
                               strategic_indices.size(), opportunity_indices.size(), planner_indices.size());
  
  size_t target_g1 = self_test ? 2 : 64;
  size_t target_g2 = self_test ? 2 : 32;
  
  std::vector<size_t> gate1_indices(strategic_indices.begin(), strategic_indices.begin() + std::min<size_t>(target_g1, strategic_indices.size()));
  std::vector<size_t> gate2_indices(opportunity_indices.begin(), opportunity_indices.begin() + std::min<size_t>(target_g2, opportunity_indices.size()));
  
  std::vector<double> v_critic_g1(gate1_indices.size(), 0.0);
  std::vector<double> v_true_g1(gate1_indices.size(), 0.0);
  
  std::vector<double> q_raw_g2(gate2_indices.size(), 0.0);
  std::vector<double> q_search_g2(gate2_indices.size(), 0.0);
  std::vector<double> q_sm_g2(gate2_indices.size(), 0.0);
  std::vector<bool> has_sm_choice(gate2_indices.size(), false);
  
  std::mutex eval_mutex;
  std::atomic<int> completed_g1{0};
  std::atomic<int> completed_g2{0};
  
  // Worker for Gate 1 (Critic Fidelity)
  std::atomic<int> next_gate1_idx{0};
  auto worker_g1 = [&]() {
    int total_g1 = gate1_indices.size();
    while (true) {
      int idx = next_gate1_idx.fetch_add(1);
      if (idx >= total_g1) break;
      
      size_t corpus_idx = gate1_indices[idx];
      const auto& cs = corpus[corpus_idx];
      auto state = ReconstructState(game, cs.history);
      
      double critic_val = global_evaluator->Evaluate(*state)[cs.player];
      if (mock_fail) {
        critic_val += 0.10;
      }
      
      double true_sum = 0.0;
      int rollouts = self_test ? 2 : 32;
      for (int k = 1; k <= rollouts; ++k) {
        // Aligned seeds with Gate 1 space prefix 100000
        uint64_t roll_seed = seed + 100000 + 1000 * idx + k;
        true_sum += RunRawPolicyRollout(*state, cs.player, global_evaluator.get(), roll_seed, utility_divisor);
      }
      double true_val = true_sum / rollouts;
      
      {
        std::lock_guard<std::mutex> lock(eval_mutex);
        v_critic_g1[idx] = critic_val;
        v_true_g1[idx] = true_val;
      }
      completed_g1.fetch_add(1);
    }
  };
  
  // Worker for Gate 2 (Choice Advantages)
  std::atomic<int> next_gate2_idx{0};
  auto worker_g2 = [&]() {
    int total_g2 = gate2_indices.size();
    while (true) {
      int idx = next_gate2_idx.fetch_add(1);
      if (idx >= total_g2) break;
      
      size_t corpus_idx = gate2_indices[idx];
      const auto& cs = corpus[corpus_idx];
      auto state = ReconstructState(game, cs.history);
      
      auto actions = state->LegalActions();
      ActionsAndProbs ppo_prior = global_evaluator->Prior(*state);
      
      // 1. Raw Policy Choice
      Action a_raw = kInvalidAction;
      double max_prior = -1.0;
      for (const auto& ap : ppo_prior) {
        if (ap.second > max_prior) {
          max_prior = ap.second;
          a_raw = ap.first;
        }
      }
      
      // 2. Search Choice
      DuneSearchConfig bot_cfg;
      bot_cfg.max_simulations = self_test ? 5 : absl::GetFlag(FLAGS_max_simulations);
      bot_cfg.puct_c = absl::GetFlag(FLAGS_puct_c);
      // Opponent mode & greedy selection settings matching the training winner
      bot_cfg.opponent_mode = SearchOpponentMode::kPolicy;
      bot_cfg.opponent_temperature = 1.0;
      bot_cfg.temperature = 0.0;
      bot_cfg.relative_time_budget_ms = std::numeric_limits<double>::infinity();
      bot_cfg.seed = seed + 200000 + 1000 * idx;
      
      DunePUCTISMCTSBot bot(bot_cfg, global_evaluator);
      DuneSearchResult result = bot.RunSearch(*state);
      
      if (!self_test) {
        SPIEL_CHECK_EQ(result.simulations_completed, absl::GetFlag(FLAGS_max_simulations));
        if (result.used_fallback) {
          SPIEL_CHECK_EQ(result.fallback_reason, "low_coverage");
        }
      }
      
      Action a_search = kInvalidAction;
      int max_visits = -1;
      for (size_t k = 0; k < result.diagnostics.actions.size(); ++k) {
        if (result.diagnostics.visit_counts[k] > max_visits) {
          max_visits = result.diagnostics.visit_counts[k];
          a_search = result.diagnostics.actions[k];
        }
      }
      
      if (!self_test && a_search != kInvalidAction) {
        SPIEL_CHECK_TRUE(std::find(actions.begin(), actions.end(), a_search) != actions.end());
      }
      
      // 3. Swordmaster Heuristic Choice
      std::mt19937 rng(seed + 300000 + 1000 * idx);
      Action a_sm = FindActionOrCardPathToSpace(*state, cs.player, dune_imperium::kActionAgentSpaceSwordmaster, &rng);
      bool sm_applicable = (a_sm != kInvalidAction && std::find(actions.begin(), actions.end(), a_sm) != actions.end());
      
      // Compute Rollouts
      double q_raw_sum = 0.0;
      double q_search_sum = 0.0;
      double q_sm_sum = 0.0;
      int rollouts = self_test ? 2 : 32;
      
      // Choice 1 continuations
      if (a_raw != kInvalidAction) {
        auto raw_state = state->Clone();
        raw_state->ApplyAction(a_raw);
        for (int k = 1; k <= rollouts; ++k) {
          uint64_t roll_seed = seed + 200000 + 1000 * idx + k; // aligned seeds using prefix 2
          q_raw_sum += RunRawPolicyRollout(*raw_state, cs.player, global_evaluator.get(), roll_seed, utility_divisor);
        }
      }
      
      // Choice 2 continuations
      if (a_search != kInvalidAction) {
        auto search_state = state->Clone();
        search_state->ApplyAction(a_search);
        for (int k = 1; k <= rollouts; ++k) {
          uint64_t roll_seed = seed + 200000 + 1000 * idx + k; // aligned seeds using prefix 2
          q_search_sum += RunRawPolicyRollout(*search_state, cs.player, global_evaluator.get(), roll_seed, utility_divisor);
        }
      }
      
      // Choice 3 continuations
      if (sm_applicable) {
        auto sm_state = state->Clone();
        sm_state->ApplyAction(a_sm);
        for (int k = 1; k <= rollouts; ++k) {
          uint64_t roll_seed = seed + 200000 + 1000 * idx + k; // aligned seeds using prefix 2
          q_sm_sum += RunRawPolicyRollout(*sm_state, cs.player, global_evaluator.get(), roll_seed, utility_divisor);
        }
      }
      
      {
        std::lock_guard<std::mutex> lock(eval_mutex);
        q_raw_g2[idx] = q_raw_sum / rollouts;
        q_search_g2[idx] = q_search_sum / rollouts;
        if (sm_applicable) {
          q_sm_g2[idx] = q_sm_sum / rollouts;
          has_sm_choice[idx] = true;
        }
      }
      completed_g2.fetch_add(1);
    }
  };
  
  std::atomic<bool> heartbeat_stop{false};
  std::thread heartbeat_thread([&]() {
    while (!heartbeat_stop.load()) {
      std::this_thread::sleep_for(std::chrono::seconds(5));
      if (heartbeat_stop.load()) break;
      int g1 = completed_g1.load();
      int g2 = completed_g2.load();
      std::cout << absl::StrFormat("[Heartbeat] Progress: Gate 1 %d/%d states, Gate 2 %d/%d states\n",
                                   g1, gate1_indices.size(), g2, gate2_indices.size()) << std::flush;
    }
  });
  
  // Launch Gate 1
  std::cout << "Starting sequential thread workers for Gate 1...\n" << std::flush;
  std::vector<std::thread> workers_g1;
  for (int i = 0; i < num_threads; ++i) {
    workers_g1.emplace_back(worker_g1);
  }
  for (auto& w : workers_g1) {
    w.join();
  }
  
  // Launch Gate 2
  std::cout << "Gate 1 completed. Starting Gate 2...\n" << std::flush;
  std::vector<std::thread> workers_g2;
  for (int i = 0; i < num_threads; ++i) {
    workers_g2.emplace_back(worker_g2);
  }
  for (auto& w : workers_g2) {
    w.join();
  }
  
  // Stop heartbeat thread
  heartbeat_stop = true;
  if (heartbeat_thread.joinable()) {
    heartbeat_thread.join();
  }
  
  std::cout << "Fidelity simulation rollouts complete.\n";
  
  // ==========================================
  // Report Gate 1 (Critic Fidelity)
  // ==========================================
  double total_bias = 0.0;
  double total_sq_err = 0.0;
  for (size_t i = 0; i < gate1_indices.size(); ++i) {
    double diff = v_critic_g1[i] - v_true_g1[i];
    total_bias += diff;
    total_sq_err += diff * diff;
  }
  double mean_bias = total_bias / gate1_indices.size();
  double rmse = std::sqrt(total_sq_err / gate1_indices.size());
  double rank_corr = SpearmanCorrelation(v_critic_g1, v_true_g1);
  
  std::cout << "\n============================================\n";
  std::cout << "      GATE 1: CRITIC FIDELITY REPORT        \n";
  std::cout << "============================================\n";
  std::cout << absl::StrFormat("States Evaluated:     %d\n", gate1_indices.size());
  std::cout << absl::StrFormat("Normalized Mean Bias: %.4f (Limit <= 0.05)\n", mean_bias);
  std::cout << absl::StrFormat("RMSE:                 %.4f\n", rmse);
  std::cout << absl::StrFormat("Rank Correlation:     %.4f\n", rank_corr);
  
  // Group by round
  std::vector<int> round_counts(6, 0);
  std::vector<double> round_bias(6, 0.0);
  std::vector<double> round_rmse(6, 0.0);
  
  for (size_t i = 0; i < gate1_indices.size(); ++i) {
    int r = corpus[gate1_indices[i]].round;
    if (r >= 1 && r <= 5) {
      round_counts[r]++;
      double diff = v_critic_g1[i] - v_true_g1[i];
      round_bias[r] += diff;
      round_rmse[r] += diff * diff;
    }
  }
  
  std::cout << "\nCalibration by Round:\n";
  for (int r = 1; r <= 5; ++r) {
    if (round_counts[r] > 0) {
      double rb = round_bias[r] / round_counts[r];
      double rrmse = std::sqrt(round_rmse[r] / round_counts[r]);
      std::cout << absl::StrFormat("  Round %d: Count=%2d, Bias=% .4f, RMSE=%.4f\n", r, round_counts[r], rb, rrmse);
    }
  }
  
  // Reliability tables (5 predicted bins)
  std::vector<int> bin_counts(5, 0);
  std::vector<double> bin_pred_sum(5, 0.0);
  std::vector<double> bin_act_sum(5, 0.0);
  
  for (size_t i = 0; i < gate1_indices.size(); ++i) {
    double pred = v_critic_g1[i];
    int bin_idx = 0;
    if (pred < -0.6) bin_idx = 0;
    else if (pred < -0.2) bin_idx = 1;
    else if (pred < 0.2) bin_idx = 2;
    else if (pred < 0.6) bin_idx = 3;
    else bin_idx = 4;
    
    bin_counts[bin_idx]++;
    bin_pred_sum[bin_idx] += pred;
    bin_act_sum[bin_idx] += v_true_g1[i];
  }
  
  std::cout << "\nBinned Predicted vs. Actual Calibration Table:\n";
  std::cout << "  Bin boundaries   | Count | Mean Predicted | Mean Actual | Mean Bias \n";
  std::cout << "  -----------------+-------+----------------+-------------+-----------\n";
  std::vector<std::string> bin_labels = {"[-1.0, -0.6)", "[-0.6, -0.2)", "[-0.2,  0.2)", "[ 0.2,  0.6)", "[ 0.6,  1.0]"};
  for (int b = 0; b < 5; ++b) {
    if (bin_counts[b] > 0) {
      double mp = bin_pred_sum[b] / bin_counts[b];
      double ma = bin_act_sum[b] / bin_counts[b];
      std::cout << absl::StrFormat("  %-16s | %5d | % 14.4f | %11.4f | % .4f\n", bin_labels[b], bin_counts[b], mp, ma, mp - ma);
    } else {
      std::cout << absl::StrFormat("  %-16s |     0 |            N/A |         N/A |       N/A\n", bin_labels[b]);
    }
  }
  
  // ==========================================
  // Report Gate 2 (Choice Advantages)
  // ==========================================
  std::vector<double> paired_advantages(gate2_indices.size(), 0.0);
  double total_adv = 0.0;
  int sm_adv_counts = 0;
  double sm_adv_sum = 0.0;
  
  for (size_t i = 0; i < gate2_indices.size(); ++i) {
    double adv = q_search_g2[i] - q_raw_g2[i];
    paired_advantages[i] = adv;
    total_adv += adv;
    
    if (has_sm_choice[i]) {
      sm_adv_sum += q_sm_g2[i] - q_raw_g2[i];
      sm_adv_counts++;
    }
  }
  double mean_adv = total_adv / gate2_indices.size();
  
  std::cout << "\n============================================\n";
  std::cout << "      GATE 2: CHOICE ADVANTAGES REPORT      \n";
  std::cout << "============================================\n";
  std::cout << absl::StrFormat("Opportunity States:    %d\n", gate2_indices.size());
  std::cout << absl::StrFormat("Search vs. Raw mean Q: % .4f\n", mean_adv);
  if (sm_adv_counts > 0) {
    std::cout << absl::StrFormat("SM vs. Raw mean Q:     % .4f (on %d applicable states)\n",
                                 sm_adv_sum / sm_adv_counts, sm_adv_counts);
  }
  
  int num_boots = 10000;
  std::vector<double> boot_means(num_boots, 0.0);
  std::mt19937 boot_rng(seed + 9999);
  std::uniform_int_distribution<int> boot_dist(0, gate2_indices.size() - 1);
  
  for (int b = 0; b < num_boots; ++b) {
    double sum = 0.0;
    for (size_t i = 0; i < gate2_indices.size(); ++i) {
      int idx = boot_dist(boot_rng);
      sum += paired_advantages[idx];
    }
    boot_means[b] = sum / gate2_indices.size();
  }
  std::sort(boot_means.begin(), boot_means.end());
  
  double bootstrap_lcb = boot_means[static_cast<int>(0.05 * num_boots)];
  if (mock_fail) {
    bootstrap_lcb = -0.01;
  }
  
  std::cout << absl::StrFormat("95%% Bootstrap LCB:      % .4f (Limit > 0.0)\n", bootstrap_lcb);
  std::cout << "============================================\n";
  
  bool bias_passed = (std::abs(mean_bias) <= 0.05);
  bool bootstrap_passed = (bootstrap_lcb > 0.0);
  
  if (!bias_passed) {
    std::cerr << absl::StrFormat("\nCRITICAL FAILURE: Critic absolute mean bias %.4f exceeds 0.05 limit!\n", std::abs(mean_bias));
  }
  if (!bootstrap_passed) {
    std::cerr << absl::StrFormat("\nCRITICAL FAILURE: 95%% Bootstrap LCB %.4f is not positive!\n", bootstrap_lcb);
  }
  
  // Save structured JSON report if requested
  std::string report_path = absl::GetFlag(FLAGS_report_path);
  if (!report_path.empty()) {
    std::filesystem::create_directories(std::filesystem::path(report_path).parent_path());
    std::ofstream out(report_path);
    if (out) {
      open_spiel::json::Object root;
      root["command_line"] = absl::StrJoin(std::vector<std::string>(argv, argv + argc), " ");
      root["self_test"] = self_test;
      
      try {
        root["corpus_fingerprint"] = open_spiel::ComputeFileSHA256(corpus_path);
      } catch (...) {
        root["corpus_fingerprint"] = "unknown";
      }
      
      if (!self_test) {
        root["model_checkpoint_path"] = model_checkpoint;
        try {
          root["model_checkpoint_hash"] = open_spiel::ComputeFileSHA256(model_checkpoint);
        } catch (...) {
          root["model_checkpoint_hash"] = "unknown";
        }
      } else {
        root["model_checkpoint_path"] = "self_test";
        root["model_checkpoint_hash"] = "self_test";
      }
      
      root["passed"] = bias_passed && bootstrap_passed;
      
      open_spiel::json::Object g1;
      g1["states_evaluated"] = static_cast<int64_t>(gate1_indices.size());
      g1["mean_bias"] = mean_bias;
      g1["rmse"] = rmse;
      g1["rank_correlation"] = rank_corr;
      root["gate1"] = g1;
      
      open_spiel::json::Object g2;
      g2["opportunity_states"] = static_cast<int64_t>(gate2_indices.size());
      g2["search_vs_raw_mean_q"] = mean_adv;
      g2["bootstrap_lcb"] = bootstrap_lcb;
      root["gate2"] = g2;
      
      out << open_spiel::json::ToString(root, true) << "\n";
      std::cout << "Saved structured JSON report to: " << report_path << "\n";
    }
  }
  
  if (self_test && !mock_fail) {
    std::cout << "\nSUCCESS: Self-test completed successfully.\n";
    return 0;
  }
  
  if (bias_passed && bootstrap_passed) {
    std::cout << "\nSUCCESS: Critic fidelity gate PASSED.\n";
    return 0;
  } else {
    std::cerr << "\nFAILURE: Gating checks FAILED. Do not begin search training.\n";
    return 1;
  }
}
