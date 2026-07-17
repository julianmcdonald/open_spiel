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
#include "dune_search_session.h"
#include "dune_warmstart_helpers.h"
#include "dune_batched_evaluator.h"
#include "dune_sha256.h"

ABSL_FLAG(std::string, model_checkpoint, "artifacts/branch_a_frozen/branch_a_seed11_model_update_2450.pt", "Path to the model checkpoint");
ABSL_FLAG(std::string, corpus_path, "data/dune_diagnostic_corpus.json", "Path to the corpus JSON file");
ABSL_FLAG(int, seed, 42, "RNG seed");
ABSL_FLAG(int, hidden_dim, 2048, "Hidden dimension");
ABSL_FLAG(int, num_blocks, 8, "Block count");
ABSL_FLAG(int, max_simulations, 64, "MCTS simulations count");
ABSL_FLAG(double, puct_c, 1.0, "PUCT exploration constant");
ABSL_FLAG(double, root_prior_temperature, 1.0, "Root prior temperature");
ABSL_FLAG(double, utility_divisor, 4.0, "Utility divisor");
ABSL_FLAG(int, threads, 32, "Number of threads");
ABSL_FLAG(double, relative_time_budget_ms, std::numeric_limits<double>::infinity(), "Time limit per move in ms");
ABSL_FLAG(int, limit_states, 0, "Limit count of states evaluated");
ABSL_FLAG(std::string, report_path, "", "Path to output a structured JSON report");
ABSL_FLAG(int, fixed_continuation_reserve, 0, "Continuation reserve simulations.");
ABSL_FLAG(int, purchase_combat_budget, 16, "Purchase/combat short-window simulation budget.");
ABSL_FLAG(std::string, target_role, "primary", "Role to calibrate: 'primary', 'purchase', or 'combat'");
ABSL_FLAG(int, opponent_mode, 1,
          "Search opponent mode: 0=kMaxN, 1=policy sampling.");
ABSL_FLAG(bool, nonlinear_value_head, false,
          "Use the versioned nonlinear value-head architecture.");
ABSL_FLAG(bool, conservative_override_enabled, false, "Enforce conservative override selection protocol");
ABSL_FLAG(double, conservative_covered_prior_threshold, 0.95, "Minimum covered prior mass to avoid fallback");
ABSL_FLAG(int, conservative_meaningful_visit_threshold, 10, "Minimum visits required for raw and argmax actions");
ABSL_FLAG(double, conservative_q_margin_threshold, 0.03, "Q margin threshold for MCTS to override raw");
ABSL_FLAG(double, conservative_stability_checkpoint_fraction, 0.5, "Fraction of budget at which to check stability");
ABSL_FLAG(bool, conservative_continuation_overrides_disabled, true, "Disable overrides during continuation decisions");

using namespace open_spiel;

struct CorpusState {
  std::string category;
  Player player;
  int round;
  std::vector<Action> history;
  std::vector<float> observation;
};

struct StateDiagnostic {
  int idx = 0;
  Action chosen_action = kInvalidAction;
  int simulations_completed = 0;
  double elapsed_time_ms = 0.0;
  double mean_depth = 0.0;
  double max_depth = 0.0;
  double terminal_leaf_fraction = 0.0;
};

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
  double prior_temp = absl::GetFlag(FLAGS_root_prior_temperature);
  double puct_c = absl::GetFlag(FLAGS_puct_c);
  int max_simulations = absl::GetFlag(FLAGS_max_simulations);
  double time_budget_ms = absl::GetFlag(FLAGS_relative_time_budget_ms);
  int limit_states = absl::GetFlag(FLAGS_limit_states);
  std::string report_path = absl::GetFlag(FLAGS_report_path);

  auto game = open_spiel::LoadGame("dune_imperium");
  int64_t obs_size = game->GetType().provides_information_state_tensor
                         ? game->InformationStateTensorSize()
                         : game->ObservationTensorSize();
  int64_t action_size = game->NumDistinctActions();

  torch::Device device = torch::cuda::is_available() ? torch::Device(torch::kCUDA) : torch::Device(torch::kCPU);
  std::cout << "Using device: " << (device.is_cuda() ? "CUDA" : "CPU") << "\n";
  std::shared_mutex model_mutex;

  auto model = std::make_shared<SharedDunePolicyValueNetImpl>(
      obs_size, absl::GetFlag(FLAGS_hidden_dim), action_size,
      absl::GetFlag(FLAGS_num_blocks),
      absl::GetFlag(FLAGS_nonlinear_value_head));
  try {
    torch::load(model, model_checkpoint, device);
  } catch (const std::exception& e) {
    std::cerr << "Failed to load model: " << e.what() << "\n";
    return 1;
  }
  model->to(device);
  model->eval();

  auto batched_eval = std::make_shared<open_spiel::BatchedEvaluator>(
      model, num_threads, /*timeout_ms=*/2, device, &model_mutex);
  auto global_evaluator = std::make_shared<open_spiel::BatchedNNEvaluator>(batched_eval);

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

  std::string target_role = absl::GetFlag(FLAGS_target_role);
  open_spiel::DuneDecisionRole expected_role = open_spiel::DuneDecisionRole::kAgentPrimary;
  if (target_role == "purchase") {
    expected_role = open_spiel::DuneDecisionRole::kPurchase;
  } else if (target_role == "combat") {
    expected_role = open_spiel::DuneDecisionRole::kCombatIntrigue;
  }

  std::vector<size_t> strategic_indices;
  for (size_t i = 0; i < corpus.size(); ++i) {
    auto state = ReconstructState(game, corpus[i].history);
    open_spiel::DuneDecisionRole role = open_spiel::ClassifyDuneDecisionRole(*state, state->CurrentPlayer(), false);
    if (role == expected_role) {
      strategic_indices.push_back(i);
    }
  }

  if (limit_states > 0) {
    SPIEL_CHECK_GE(strategic_indices.size(), static_cast<size_t>(limit_states));
    strategic_indices.erase(strategic_indices.begin() + limit_states, strategic_indices.end());
    SPIEL_CHECK_EQ(strategic_indices.size(), static_cast<size_t>(limit_states));
  }

  std::cout << absl::StrFormat("Loaded %d strategic states for evaluation.\n", strategic_indices.size());

  std::vector<StateDiagnostic> detailed_diagnostics(strategic_indices.size());
  std::atomic<int> next_strategic_idx{0};
  std::atomic<int> total_accepted{0};
  std::atomic<double> total_paired_advantage{0.0};
  std::atomic<double> total_discovered_useful{0.0};

  auto worker_fn = [&]() {
    int total_states = strategic_indices.size();
    while (true) {
      int idx = next_strategic_idx.fetch_add(1);
      if (idx >= total_states) break;

      size_t corpus_idx = strategic_indices[idx];
      const auto& cs = corpus[corpus_idx];
      auto state = ReconstructState(game, cs.history);
      auto actions = state->LegalActions();

      ActionsAndProbs ppo_prior = global_evaluator->Prior(*state);

      // 1. Raw Policy Choice (argmax)
      Action a_raw = kInvalidAction;
      double max_prior = -1.0;
      for (const auto& ap : ppo_prior) {
        if (ap.second > max_prior) {
          max_prior = ap.second;
          a_raw = ap.first;
        }
      }

      // 2. Search Choice under root prior temperature
      DuneSearchConfig bot_cfg;
      bot_cfg.max_simulations = max_simulations;
      bot_cfg.puct_c = puct_c;
      const int opponent_mode = absl::GetFlag(FLAGS_opponent_mode);
      SPIEL_CHECK_TRUE(opponent_mode == 0 || opponent_mode == 1);
      bot_cfg.opponent_mode = opponent_mode == 0
                                  ? SearchOpponentMode::kMaxN
                                  : SearchOpponentMode::kPolicy;
      bot_cfg.opponent_temperature = 1.0;
      bot_cfg.temperature = 0.0;
      bot_cfg.relative_time_budget_ms = time_budget_ms;
      bot_cfg.seed = seed + 200000 + 1000 * idx;
      bot_cfg.root_prior_temperature = prior_temp;
      bot_cfg.fixed_session_limit = max_simulations;
      bot_cfg.fixed_continuation_reserve = absl::GetFlag(FLAGS_fixed_continuation_reserve);
      bot_cfg.purchase_combat_budget = absl::GetFlag(FLAGS_purchase_combat_budget);
      bot_cfg.model_checkpoint_path = absl::GetFlag(FLAGS_model_checkpoint);
      bot_cfg.conservative_override_enabled = absl::GetFlag(FLAGS_conservative_override_enabled);
      bot_cfg.conservative_covered_prior_threshold = absl::GetFlag(FLAGS_conservative_covered_prior_threshold);
      bot_cfg.conservative_meaningful_visit_threshold = absl::GetFlag(FLAGS_conservative_meaningful_visit_threshold);
      bot_cfg.conservative_q_margin_threshold = absl::GetFlag(FLAGS_conservative_q_margin_threshold);
      bot_cfg.conservative_stability_checkpoint_fraction = absl::GetFlag(FLAGS_conservative_stability_checkpoint_fraction);
      bot_cfg.conservative_continuation_overrides_disabled = absl::GetFlag(FLAGS_conservative_continuation_overrides_disabled);

      DuneSearchSession session(bot_cfg, global_evaluator, DuneSearchBudgetMode::kFixedSessionSimulations);
      DuneSearchResult result = session.Search(*state);
      std::mt19937 step_rng(bot_cfg.seed);
      double r_val = absl::Uniform(step_rng, 0.0, 1.0);
      ControllerDecision decision = session.SelectControllerAction(*state, result, r_val);
      result = session.CommitAction(decision);

      Action a_search = result.diagnostics.selected_action;

      // Compute rollout advantage (32 continuations)
      double q_raw_sum = 0.0;
      double q_search_sum = 0.0;
      int rollouts = 32;

      // Choice 1: Raw
      if (a_raw != kInvalidAction) {
        auto raw_state = state->Clone();
        raw_state->ApplyAction(a_raw);
        for (int k = 1; k <= rollouts; ++k) {
          uint64_t roll_seed = seed + 200000 + 1000 * idx + k;
          q_raw_sum += RunRawPolicyRollout(*raw_state, cs.player, global_evaluator.get(), roll_seed, utility_divisor);
        }
      }

      // Choice 2: Search (if choices differ, compute continuations; else reuse)
      if (a_search != kInvalidAction) {
        if (a_search == a_raw) {
          q_search_sum = q_raw_sum;
        } else {
          auto search_state = state->Clone();
          search_state->ApplyAction(a_search);
          for (int k = 1; k <= rollouts; ++k) {
            uint64_t roll_seed = seed + 200000 + 1000 * idx + k;
            q_search_sum += RunRawPolicyRollout(*search_state, cs.player, global_evaluator.get(), roll_seed, utility_divisor);
          }
        }
      }

      double q_raw = q_raw_sum / rollouts;
      double q_search = q_search_sum / rollouts;
      double state_advantage = q_search - q_raw;

      open_spiel::SearchDiagnostics diag = result.diagnostics;
      int required_coverage = std::min(3, static_cast<int>(diag.actions.size()));
      bool has_coverage = (diag.num_covered_actions >= required_coverage);
      bool has_mass = (diag.covered_prior_mass >= 0.50);
      bool accepted = (has_coverage && has_mass);

      // Discovery rate using true unscaled ppo_prior
      int discovered = 0;
      for (size_t k = 0; k < diag.actions.size(); ++k) {
        double true_raw_p = 0.0;
        for (const auto& ap : ppo_prior) {
          if (ap.first == diag.actions[k]) {
            true_raw_p = ap.second;
            break;
          }
        }
        double visit_prob = static_cast<double>(diag.visit_counts[k]) / std::max(1, diag.total_root_visits);
        if (true_raw_p < 0.01 && visit_prob >= 0.05) {
          discovered++;
        }
      }

      if (accepted) total_accepted.fetch_add(1);

      double prev_adv = total_paired_advantage.load(std::memory_order_relaxed);
      while (!total_paired_advantage.compare_exchange_weak(prev_adv, prev_adv + state_advantage)) {}

      double prev_disc = total_discovered_useful.load(std::memory_order_relaxed);
      while (!total_discovered_useful.compare_exchange_weak(prev_disc, prev_disc + discovered)) {}

      // Log detailed state diagnostic
      StateDiagnostic sd;
      sd.idx = idx;
      sd.chosen_action = a_search;
      sd.simulations_completed = result.simulations_completed;
      sd.elapsed_time_ms = result.elapsed_time_ms;
      sd.mean_depth = result.diagnostics.mean_depth;
      sd.max_depth = result.diagnostics.max_depth;
      sd.terminal_leaf_fraction = result.diagnostics.terminal_leaf_fraction;
      detailed_diagnostics[idx] = sd;
    }
  };

  std::vector<std::thread> workers;
  for (int i = 0; i < num_threads; ++i) {
    workers.emplace_back(worker_fn);
  }
  for (auto& w : workers) w.join();

  double yield = (100.0 * total_accepted.load()) / strategic_indices.size();
  double avg_adv = total_paired_advantage.load() / strategic_indices.size();
  double avg_disc = total_discovered_useful.load() / strategic_indices.size();

  std::cout << "\n============================================\n";
  std::cout << absl::StrFormat("Calibration finished.\n");
  std::cout << absl::StrFormat("Accepted Yield:                     %.2f%% (%d/%d states)\n", yield, total_accepted.load(), strategic_indices.size());
  std::cout << absl::StrFormat("Paired Rollout Action Advantage:    %+.6f\n", avg_adv);
  std::cout << "============================================\n\n";

  if (!report_path.empty()) {
    open_spiel::json::Object rep;
    rep["root_prior_temperature"] = prior_temp;
    rep["max_simulations"] = static_cast<int64_t>(max_simulations);
    rep["puct_c"] = puct_c;
    rep["opponent_mode"] =
        static_cast<int64_t>(absl::GetFlag(FLAGS_opponent_mode));
    rep["nonlinear_value_head"] =
        absl::GetFlag(FLAGS_nonlinear_value_head);
    rep["relative_time_budget_ms"] = time_budget_ms;
    rep["states_evaluated"] = static_cast<int64_t>(strategic_indices.size());
    rep["accepted_states"] = static_cast<int64_t>(total_accepted.load());
    rep["accepted_target_yield"] = yield;
    rep["paired_rollout_action_advantage"] = avg_adv;
    rep["discovery_of_useful_actions"] = avg_disc;

    open_spiel::json::Array diag_arr;
    for (const auto& sd : detailed_diagnostics) {
      open_spiel::json::Object sd_obj;
      sd_obj["state_idx"] = static_cast<int64_t>(sd.idx);
      sd_obj["chosen_action"] = static_cast<int64_t>(sd.chosen_action);
      sd_obj["simulations_completed"] = static_cast<int64_t>(sd.simulations_completed);
      sd_obj["elapsed_time_ms"] = sd.elapsed_time_ms;
      sd_obj["mean_depth"] = sd.mean_depth;
      sd_obj["max_depth"] = sd.max_depth;
      sd_obj["terminal_leaf_fraction"] = sd.terminal_leaf_fraction;
      diag_arr.push_back(sd_obj);
    }
    rep["detailed_diagnostics"] = diag_arr;

    std::ofstream out(report_path);
    if (out) {
      out << open_spiel::json::ToString(rep, true);
    }
  }

  return 0;
}
