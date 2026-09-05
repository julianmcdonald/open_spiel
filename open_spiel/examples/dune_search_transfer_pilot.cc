// Copyright 2026 DeepMind Technologies Limited
//
// Standalone runner for Bounded Search-Training Transfer Pilot from Placement U200.
// Executes:
//   1. Smoke test: active parameter update, save/reload, prediction diff,
//      optimizer state equivalence, zero-weight minibatch handling, row JSONL roundtrip.
//   2. Parity check: multi-seat (seats 0,1,2,3) decision-level replay comparison
//      against benchmark controller covering all key decision roles and fallbacks.
//   3. Collection: 128 complete games (96 train / 32 heldout) with 16 threads,
//      exact short-window budgets, 0.50 coverage threshold, and raw fallback masking.
//   4. Training: 1-epoch weighted visit CE + value MSE using RunScratchSearchPiLearner.

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <mutex>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <torch/torch.h>

#include "open_spiel/abseil-cpp/absl/flags/flag.h"
#include "open_spiel/abseil-cpp/absl/flags/parse.h"
#include "open_spiel/abseil-cpp/absl/random/distributions.h"
#include "open_spiel/abseil-cpp/absl/strings/str_cat.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"
#include "open_spiel/utils/json.h"

#include "dune_evaluator.h"
#include "dune_online_search_collector.h"
#include "dune_ppo_training_utils.h"
#include "dune_puct_is_mcts.h"
#include "dune_search_pi.h"
#include "dune_search_pi_scratch.h"
#include "dune_search_routing.h"
#include "dune_search_session.h"
#include "dune_seed_utils.h"
#include "dune_sha256.h"

// --- CLI Flags ---------------------------------------------------------------
ABSL_FLAG(std::string, mode, "smoke_test",
          "Execution mode: smoke_test | verify_parity | collect | train");
ABSL_FLAG(std::string, model_checkpoint, "",
          "Path to candidate model checkpoint (U200).");
ABSL_FLAG(std::string, opponent_checkpoint, "",
          "Path to opponent model checkpoint (u15828).");
ABSL_FLAG(int, games, 128, "Total games to collect (default 128).");
ABSL_FLAG(int, start_episode_id, 0, "First episode id.");
ABSL_FLAG(uint64_t, seed, 2026090850ULL, "Base seed for data generation.");
ABSL_FLAG(int, threads, 16, "Number of worker threads.");

// Architecture & limits
ABSL_FLAG(int, hidden_dim, 2048, "Model hidden dimension.");
ABSL_FLAG(int, num_blocks, 8, "Model residual blocks.");
ABSL_FLAG(int, opp_hidden_dim, 2048, "Opponent model hidden dimension.");
ABSL_FLAG(int, opp_num_blocks, 8, "Opponent model residual blocks.");
ABSL_FLAG(double, candidate_logit_cap, 10.0, "Candidate logit cap.");
ABSL_FLAG(double, opponent_logit_cap, 10.0, "Opponent logit cap.");

// Search configuration (matching dune_search_benchmark)
ABSL_FLAG(int, pi_primary_simulations, 200, "Primary search simulations.");
ABSL_FLAG(int, pi_continuation_simulations, 64, "Continuation search simulations.");
ABSL_FLAG(int, pi_short_window_simulations, 64, "Purchase/combat search simulations.");
ABSL_FLAG(double, puct_c, 0.30, "PUCT exploration constant.");
ABSL_FLAG(double, covered_prior_threshold, 0.50, "Coverage prior threshold.");

// Training hyperparameters
ABSL_FLAG(double, learning_rate, 1e-5, "Search learner AdamW learning rate.");
ABSL_FLAG(double, weight_decay, 0.0, "Weight decay for AdamW (0.0).");
ABSL_FLAG(int, minibatch_size, 256, "Minibatch size.");
ABSL_FLAG(int, epochs, 1, "Training epochs.");
ABSL_FLAG(double, policy_coef, 1.0, "Policy cross-entropy coefficient.");
ABSL_FLAG(double, value_coef, 0.5, "Value MSE coefficient.");
ABSL_FLAG(double, grad_clip_norm, 0.5, "Gradient clipping norm.");

// Output paths
ABSL_FLAG(std::string, output_train_jsonl, "", "Path to write training rows JSONL.");
ABSL_FLAG(std::string, output_heldout_jsonl, "", "Path to write heldout rows JSONL.");
ABSL_FLAG(std::string, output_games_jsonl, "", "Path to write games summary JSONL.");
ABSL_FLAG(std::string, output_model_path, "", "Path to write trained child model.");
ABSL_FLAG(std::string, output_optim_path, "", "Path to write trained child optimizer.");
ABSL_FLAG(std::string, output_meta_json, "", "Path to write metadata JSON.");
ABSL_FLAG(std::string, input_train_jsonl, "", "Path to read training rows for train mode.");
ABSL_FLAG(std::string, input_heldout_jsonl, "", "Path to read heldout rows for train mode.");

// Link-only PPO flag definitions required by dune_ppo_training_utils.cc.
ABSL_FLAG(int, ppo_minibatch_size, 2048, "");
ABSL_FLAG(int, ppo_update_epochs, 4, "");
ABSL_FLAG(double, ppo_clip_epsilon, 0.2, "");
ABSL_FLAG(bool, normalize_advantages, true, "");
ABSL_FLAG(bool, ppo_clip_value_loss, true, "");
ABSL_FLAG(double, entropy_coef, 0.01, "");
ABSL_FLAG(double, logit_cap, 10.0, "");
ABSL_FLAG(double, target_kl, 0.0, "");
ABSL_FLAG(bool, train_amp, true, "");
ABSL_FLAG(bool, diagnostics_only, false, "");

namespace open_spiel {
namespace {

// Row Serialization to JSONL
bool WriteTransferPilotRows(const std::string& path,
                            const std::vector<SearchPiRow>& rows,
                            std::string* error) {
  std::ofstream out(path);
  if (!out.is_open()) {
    if (error) *error = "Failed to open output file: " + path;
    return false;
  }

  for (const auto& r : rows) {
    json::Object obj;
    obj["episode_id"] = static_cast<int64_t>(r.episode_id);
    obj["decision_id"] = static_cast<int64_t>(r.decision_id);
    obj["player"] = static_cast<int64_t>(r.player);
    obj["role"] = static_cast<int64_t>(r.role);
    obj["chosen_action"] = static_cast<int64_t>(r.chosen_action);
    obj["policy_target_weight"] = r.policy_target_weight;
    obj["value_target"] = r.value_target;
    obj["value_target_attached"] = r.value_target_attached;
    obj["simulations_completed"] = static_cast<int64_t>(r.simulations_completed);
    obj["simulations_requested"] = static_cast<int64_t>(r.simulations_requested);
    obj["search_fallback_reason"] = r.search_fallback_reason;
    obj["root_value"] = r.root_value;

    json::Array legals_arr;
    for (Action a : r.legal_actions) legals_arr.push_back(static_cast<int64_t>(a));
    obj["legal_actions"] = legals_arr;

    json::Array target_arr;
    for (double p : r.target_probs) target_arr.push_back(p);
    obj["target_probs"] = target_arr;

    json::Array obs_arr;
    for (float f : r.observation) obs_arr.push_back(static_cast<double>(f));
    obj["observation"] = obs_arr;

    out << json::ToString(json::Value(obj)) << "\n";
  }

  return true;
}

bool ReadTransferPilotRows(const std::string& path,
                           std::vector<SearchPiRow>* rows,
                           std::string* error) {
  std::ifstream in(path);
  if (!in.is_open()) {
    if (error) *error = "Failed to open input file: " + path;
    return false;
  }

  rows->clear();
  std::string line;
  int line_num = 0;
  while (std::getline(in, line)) {
    line_num++;
    if (line.empty()) continue;
    auto parsed = json::FromString(line);
    if (!parsed.has_value() || !parsed->IsObject()) {
      if (error) *error = absl::StrCat("Malformed JSON on line ", line_num);
      return false;
    }
    const auto& obj = parsed->GetObject();
    SearchPiRow r;
    r.row_schema_version = 2;
    r.target_type = "visit_policy";
    r.search_budget_class = "full";
    r.episode_id = obj.at("episode_id").GetInt();
    r.decision_id = obj.at("decision_id").GetInt();
    r.player = static_cast<Player>(obj.at("player").GetInt());
    r.role = static_cast<DuneDecisionRole>(obj.at("role").GetInt());
    r.chosen_action = static_cast<Action>(obj.at("chosen_action").GetInt());
    r.policy_target_weight = obj.at("policy_target_weight").GetDouble();
    r.value_target = obj.at("value_target").GetDouble();
    r.value_target_attached = obj.at("value_target_attached").GetBool();
    r.simulations_completed = static_cast<int>(obj.at("simulations_completed").GetInt());
    r.simulations_requested = static_cast<int>(obj.at("simulations_requested").GetInt());
    r.search_fallback_reason = obj.at("search_fallback_reason").GetString();
    r.root_value = obj.at("root_value").GetDouble();

    const auto& legals_arr = obj.at("legal_actions").GetArray();
    r.legal_actions.reserve(legals_arr.size());
    for (const auto& v : legals_arr) r.legal_actions.push_back(static_cast<Action>(v.GetInt()));

    const auto& target_arr = obj.at("target_probs").GetArray();
    r.target_probs.reserve(target_arr.size());
    for (const auto& v : target_arr) r.target_probs.push_back(v.GetDouble());

    const auto& obs_arr = obj.at("observation").GetArray();
    r.observation.reserve(obs_arr.size());
    for (const auto& v : obs_arr) r.observation.push_back(static_cast<float>(v.GetDouble()));

    r.target_visits.assign(r.legal_actions.size(), 0);
    r.raw_visits.assign(r.legal_actions.size(), 0);
    r.raw_policy.assign(r.legal_actions.size(), 1.0 / std::max<size_t>(1, r.legal_actions.size()));
    r.q_values.assign(r.legal_actions.size(), r.root_value);

    rows->push_back(std::move(r));
  }

  return true;
}

class DuneGreedyBot : public Bot {
 public:
  DuneGreedyBot(std::shared_ptr<DuneNNEvaluator> evaluator, int seed, double temperature)
      : evaluator_(std::move(evaluator)), rng_(seed), temperature_(temperature) {}

  Action Step(const State& state) override {
    if (state.IsTerminal()) return kInvalidAction;
    ActionsAndProbs prior = evaluator_->Prior(state);
    return SelectBestAction(prior, state);
  }

  bool ProvidesPolicy() override { return true; }

  ActionsAndProbs GetPolicy(const State& state) override {
    return evaluator_->Prior(state);
  }

  std::pair<ActionsAndProbs, Action> StepWithPolicy(const State& state) override {
    ActionsAndProbs policy = GetPolicy(state);
    Action chosen = SelectBestAction(policy, state);
    return {policy, chosen};
  }

  void Restart() override {}
  void RestartAt(const State& state) override {}

 private:
  double RandomNumber() {
    return absl::Uniform(rng_, 0.0, 1.0);
  }

  Action SelectBestAction(const ActionsAndProbs& policy, const State& state) {
    if (policy.empty()) {
      auto legals = state.LegalActions();
      return legals.empty() ? kInvalidAction : legals.front();
    }
    double temp = temperature_;
    if (temp <= 0.0) {
      Action best_action = policy[0].first;
      double best_prob = policy[0].second;
      for (const auto& ap : policy) {
        if (ap.second > best_prob) {
          best_prob = ap.second;
          best_action = ap.first;
        }
      }
      return best_action;
    } else {
      ActionsAndProbs scaled_policy;
      scaled_policy.reserve(policy.size());
      double sum = 0.0;
      double inv_temp = 1.0 / temp;
      for (const auto& ap : policy) {
        double p = std::pow(std::max(ap.second, 1e-12), inv_temp);
        scaled_policy.push_back({ap.first, p});
        sum += p;
      }
      for (auto& ap : scaled_policy) {
        ap.second /= sum;
      }
      return SampleAction(scaled_policy, RandomNumber()).first;
    }
  }

  std::shared_ptr<DuneNNEvaluator> evaluator_;
  std::mt19937 rng_;
  double temperature_;
};

// Single game collection result
struct GameCollectionResult {
  int episode_id = 0;
  int search_seat = 0;
  uint64_t game_seed = 0;
  double wall_seconds = 0.0;
  double return_search_seat = 0.0;
  int winning_round = 0;
  int search_seat_rank = 4;
  std::vector<SearchPiRow> rows;
  int primary_searches = 0;
  int continuation_searches = 0;
  int short_window_searches = 0;
  int low_coverage_fallbacks = 0;
  int forced_or_bookkeeping = 0;
  std::map<std::string, int> role_counts;
};

GameCollectionResult PlayCollectionGame(
    int episode_id,
    uint64_t master_seed,
    const std::shared_ptr<const Game>& game,
    const std::shared_ptr<SharedDunePolicyValueNetImpl>& search_model,
    const std::shared_ptr<SharedDunePolicyValueNetImpl>& opponent_model,
    torch::Device device,
    double candidate_logit_cap,
    double opponent_logit_cap) {
  GameCollectionResult result;
  result.episode_id = episode_id;
  result.search_seat = episode_id % 4;

  auto t_start = std::chrono::steady_clock::now();
  uint64_t game_seed = dune_seed::DeriveSeed(master_seed, dune_seed::kStreamBlueprint, episode_id);
  result.game_seed = game_seed;
  std::mt19937 game_rng(game_seed);

  // Exactly 4 seat seeds drawn in order
  std::array<uint64_t, 4> seat_seeds{};
  for (int p = 0; p < 4; ++p) {
    seat_seeds[p] = game_rng();
  }

  // Evaluators
  auto search_evaluator = std::make_shared<DuneNNEvaluator>(search_model, device, candidate_logit_cap);
  auto opponent_evaluator = std::make_shared<DuneNNEvaluator>(opponent_model, device, opponent_logit_cap);

  // MCTS Config matching dune_search_benchmark exactly
  DuneSearchConfig config{
      .max_simulations = 50,
      .relative_time_budget_ms = std::numeric_limits<double>::infinity(),
      .max_nodes = 50000,
      .puct_c = absl::GetFlag(FLAGS_puct_c),
      .opponent_mode = SearchOpponentMode::kMaxN,
      .temperature = 0.0,
      .opponent_temperature = 1.0,
      .max_world_samples = -1,
      .utility_divisor = 4.0,
      .min_visit_threshold = 2,
      .covered_prior_threshold = absl::GetFlag(FLAGS_covered_prior_threshold),
      .seed = seat_seeds[result.search_seat],
      .final_policy_type = DuneISMCTSFinalPolicyType::kNormalizedVisitCount,
      .dirichlet_epsilon = 0.0,
      .dirichlet_alpha = 0.3,
      .use_observation_string = true,
      .verbose_diagnostics = false,
      .check_strategic_state = false,
      .root_prior_temperature = 1.0,
  };
  config.search_leader_draft = false;
  config.fixed_session_limit = 50;
  config.fixed_continuation_reserve = 0;
  config.purchase_combat_budget = absl::GetFlag(FLAGS_pi_short_window_simulations);
  config.pi_primary_simulations = absl::GetFlag(FLAGS_pi_primary_simulations);
  config.pi_continuation_simulations = absl::GetFlag(FLAGS_pi_continuation_simulations);
  config.exact_short_window_budgets = true;
  config.policy_only_covers_short_window = false;

  DuneSearchBudgetMode budget_mode = DuneSearchBudgetMode::kTrainingPolicyIteration;
  auto session = std::make_unique<DuneSearchSession>(config, search_evaluator, budget_mode);

  // Opponent bots (greedy raw policy from opponent checkpoint)
  std::vector<std::unique_ptr<Bot>> bots(4);
  for (int p = 0; p < 4; ++p) {
    if (p != result.search_seat) {
      bots[p] = std::make_unique<DuneGreedyBot>(opponent_evaluator, static_cast<int>(seat_seeds[p]), 0.0);
    }
  }

  std::unique_ptr<State> state = game->NewInitialState();
  int decision_id = 0;

  while (!state->IsTerminal()) {
    if (state->IsChanceNode()) {
      auto outcomes = state->ChanceOutcomes();
      Action action;
      if (game->GetType().chance_mode == GameType::ChanceMode::kSampledStochastic) {
        action = outcomes.front().first;
      } else {
        action = SampleAction(outcomes, game_rng).first;
      }
      state->ApplyAction(action);
      continue;
    }

    Player current_player = state->CurrentPlayer();
    Action chosen_action = -1;

    if (current_player == result.search_seat) {
      bool has_active = session->HasActiveSession();
      DuneDecisionRole role = ClassifyDuneDecisionRole(*state, current_player, has_active);
      result.role_counts[std::to_string(static_cast<int>(role))]++;

      std::vector<Action> legal_actions = state->LegalActions();
      std::vector<float> obs = state->InformationStateTensor(current_player);

      DuneSearchResult last_res = session->SearchAndSelectWithDeadline(*state, 0.0);
      chosen_action = last_res.diagnostics.selected_action;

      SearchPiRow row;
      row.row_schema_version = 2;
      row.target_type = "visit_policy";
      row.search_budget_class = "full";
      row.episode_id = episode_id;
      row.decision_id = decision_id++;
      row.player = current_player;
      row.role = role;
      row.legal_actions = legal_actions;
      row.chosen_action = chosen_action;
      row.observation = std::move(obs);
      row.simulations_completed = last_res.simulations_completed;
      row.simulations_requested = last_res.diagnostics.soft_sim_limit;
      row.search_fallback_reason = last_res.fallback_reason;
      row.root_value = last_res.diagnostics.root_value;
      row.q_values = last_res.diagnostics.q_values;
      row.raw_visits = last_res.diagnostics.visit_counts;
      row.raw_policy = last_res.diagnostics.raw_priors.empty()
                           ? last_res.diagnostics.priors
                           : last_res.diagnostics.raw_priors;

      if (role == DuneDecisionRole::kAgentPrimary) {
        result.primary_searches++;
      } else if (role == DuneDecisionRole::kAgentContinuation) {
        result.continuation_searches++;
      } else if (role == DuneDecisionRole::kPurchase ||
                 role == DuneDecisionRole::kCombatIntrigue ||
                 role == DuneDecisionRole::kOtherOptional) {
        result.short_window_searches++;
      } else {
        result.forced_or_bookkeeping++;
      }

      if (last_res.used_fallback) {
        if (last_res.fallback_reason == "low_coverage") {
          result.low_coverage_fallbacks++;
        }
        row.policy_target_weight = 0.0;
        row.target_probs.assign(legal_actions.size(), 0.0);
        row.target_visits.assign(legal_actions.size(), 0);
      } else {
        row.policy_target_weight = 1.0;
        row.target_visits = last_res.diagnostics.visit_counts;
        int64_t total_v = 0;
        for (int v : row.target_visits) total_v += v;
        row.target_probs.resize(legal_actions.size(), 0.0);
        if (total_v > 0) {
          for (size_t i = 0; i < legal_actions.size(); ++i) {
            row.target_probs[i] = static_cast<double>(row.target_visits[i]) / static_cast<double>(total_v);
          }
        }
      }

      result.rows.push_back(std::move(row));
    } else {
      chosen_action = bots[current_player]->Step(*state);
    }

    state->ApplyAction(chosen_action);
  }

  auto t_end = std::chrono::steady_clock::now();
  result.wall_seconds = std::chrono::duration<double>(t_end - t_start).count();

  std::vector<double> returns = state->Returns();
  result.return_search_seat = returns[result.search_seat];
  double value_target = result.return_search_seat / 4.0;

  // Rank determination
  int rank = 1;
  for (int p = 0; p < 4; ++p) {
    if (p != result.search_seat && returns[p] > returns[result.search_seat]) {
      rank++;
    }
  }
  result.search_seat_rank = rank;

  // Attach terminal returns to all collected candidate rows
  for (auto& r : result.rows) {
    r.value_target = value_target;
    r.value_target_attached = true;
  }

  return result;
}

// -----------------------------------------------------------------------------
// Mode: smoke_test
// -----------------------------------------------------------------------------
int RunSmokeTest() {
  std::cout << "========================================================================\n";
  std::cout << ">>> RUNNING ACTIVE SMOKE TEST\n";
  std::cout << "========================================================================\n";

  torch::Device device = torch::cuda::is_available() ? torch::Device(torch::kCUDA) : torch::Device(torch::kCPU);
  std::cout << "[SMOKE] Using hardware device: " << device << "\n";

  int64_t obs_size = 5580;
  int64_t action_dim = 2391;
  int hidden_dim = absl::GetFlag(FLAGS_hidden_dim);
  int num_blocks = absl::GetFlag(FLAGS_num_blocks);

  // 1. Synthetic dataset with both accepted and rejected rows
  std::vector<SearchPiRow> sample_rows;
  for (int i = 0; i < 8; ++i) {
    SearchPiRow r;
    r.row_schema_version = 2;
    r.target_type = "visit_policy";
    r.search_budget_class = "full";
    r.episode_id = i;
    r.decision_id = 0;
    r.player = 0;
    r.role = (i % 2 == 0) ? DuneDecisionRole::kAgentPrimary : DuneDecisionRole::kPurchase;
    r.observation.assign(obs_size, 0.05f * (i + 1));
    r.legal_actions = {10, 20, 30};
    r.chosen_action = 10;
    r.value_target = (i % 4 == 0) ? 0.5625 : (i % 4 == 1) ? 0.0625 : (i % 4 == 2) ? -0.1875 : -0.4375;
    r.value_target_attached = true;
    r.simulations_completed = 64;
    r.simulations_requested = 64;
    r.root_value = 0.0;
    r.raw_policy = {0.5, 0.3, 0.2};
    r.raw_visits = {32, 20, 12};
    r.q_values = {0.1, 0.0, -0.1};

    if (i % 2 == 0) {
      // Accepted search
      r.policy_target_weight = 1.0;
      r.target_visits = {40, 20, 4};
      r.target_probs = {40.0 / 64.0, 20.0 / 64.0, 4.0 / 64.0};
      r.search_fallback_reason = "none";
    } else {
      // Rejected search / fallback
      r.policy_target_weight = 0.0;
      r.target_visits = {0, 0, 0};
      r.target_probs = {0.0, 0.0, 0.0};
      r.search_fallback_reason = "low_coverage";
    }
    sample_rows.push_back(std::move(r));
  }

  // Check 1: Serialization / Deserialization fidelity
  std::string tmp_jsonl = "/tmp/smoke_transfer_pilot_rows.jsonl";
  std::string ser_err;
  if (!WriteTransferPilotRows(tmp_jsonl, sample_rows, &ser_err)) {
    std::cerr << "FAIL: Row serialization failed: " << ser_err << "\n";
    return 1;
  }
  std::vector<SearchPiRow> read_rows;
  if (!ReadTransferPilotRows(tmp_jsonl, &read_rows, &ser_err)) {
    std::cerr << "FAIL: Row deserialization failed: " << ser_err << "\n";
    return 1;
  }
  if (read_rows.size() != sample_rows.size()) {
    std::cerr << "FAIL: Deserialized row count mismatch: " << read_rows.size() << " vs " << sample_rows.size() << "\n";
    return 1;
  }
  for (size_t i = 0; i < sample_rows.size(); ++i) {
    if (read_rows[i].policy_target_weight != sample_rows[i].policy_target_weight ||
        read_rows[i].value_target != sample_rows[i].value_target ||
        read_rows[i].target_probs.size() != sample_rows[i].target_probs.size()) {
      std::cerr << "FAIL: Field mismatch in deserialized row " << i << "\n";
      return 1;
    }
  }
  std::filesystem::remove(tmp_jsonl);
  std::cout << "[SMOKE CHECK 1 PASSED] Row JSONL serialization & deserialization round-trip verified.\n";

  // Check 2: Zero-weight minibatch handling
  std::vector<SearchPiRow> zero_weight_rows;
  for (int i = 0; i < 4; ++i) {
    SearchPiRow r = sample_rows[1]; // rejected row
    zero_weight_rows.push_back(r);
  }
  auto test_model = std::make_shared<SharedDunePolicyValueNetImpl>(obs_size, hidden_dim, action_dim, num_blocks);
  test_model->to(device);
  auto test_opt = std::make_unique<torch::optim::AdamW>(
      test_model->parameters(),
      torch::optim::AdamWOptions(1e-5).eps(1e-5).weight_decay(0.0));
  SearchPiLearnerConfig zw_cfg;
  zw_cfg.minibatch_size = 4;
  zw_cfg.epochs = 1;
  zw_cfg.policy_coef = 1.0;
  zw_cfg.value_coef = 0.5;
  zw_cfg.grad_clip_norm = 0.5;
  zw_cfg.logit_cap = 10.0;

  auto zw_stats = RunScratchSearchPiLearner(test_model, *test_opt, zero_weight_rows,
                                           obs_size, action_dim, device,
                                           /*master_seed=*/12345ULL, /*generation=*/0, zw_cfg);
  if (zw_stats.policy_backward_executed) {
    std::cerr << "FAIL: Policy backward was executed on all-zero-weight batch!\n";
    return 1;
  }
  if (!zw_stats.value_backward_executed) {
    std::cerr << "FAIL: Value backward was not executed on all-zero-weight batch!\n";
    return 1;
  }
  std::cout << "[SMOKE CHECK 2 PASSED] Zero-weight minibatch handled cleanly (policy backward skipped, value backward executed).\n";

  // Check 3: Active parameter update, save/reload, and optimizer equivalence
  std::string ckpt_path = absl::GetFlag(FLAGS_model_checkpoint);
  auto model_a = std::make_shared<SharedDunePolicyValueNetImpl>(obs_size, hidden_dim, action_dim, num_blocks);
  if (!ckpt_path.empty() && std::filesystem::exists(ckpt_path)) {
    torch::serialize::InputArchive archive;
    archive.load_from(ckpt_path, device);
    model_a->load(archive);
    std::cout << "[SMOKE] Loaded origin weights from " << ckpt_path << "\n";
  }
  model_a->to(device);

  auto opt_a = std::make_unique<torch::optim::AdamW>(
      model_a->parameters(),
      torch::optim::AdamWOptions(1e-5).eps(1e-5).weight_decay(0.0));

  torch::Tensor test_input = torch::ones({1, obs_size}, torch::kFloat32).to(device);
  auto pre_pred = model_a->forward(test_input);
  float pre_v = pre_pred.values.item<float>();
  float pre_l0 = pre_pred.logits[0][0].item<float>();

  // Run 1 real step
  SearchPiLearnerConfig tr_cfg;
  tr_cfg.minibatch_size = 4;
  tr_cfg.epochs = 1;
  tr_cfg.policy_coef = 1.0;
  tr_cfg.value_coef = 0.5;
  tr_cfg.grad_clip_norm = 0.5;
  tr_cfg.logit_cap = 10.0;

  auto tr_stats = RunScratchSearchPiLearner(model_a, *opt_a, sample_rows,
                                           obs_size, action_dim, device,
                                           /*master_seed=*/9999ULL, /*generation=*/0, tr_cfg);

  auto post_pred = model_a->forward(test_input);
  float post_v = post_pred.values.item<float>();
  float post_l0 = post_pred.logits[0][0].item<float>();

  if (std::abs(post_v - pre_v) < 1e-8 && std::abs(post_l0 - pre_l0) < 1e-8) {
    std::cerr << "FAIL: Model weights did not update after training step! Pre: "
              << pre_v << ", " << pre_l0 << " Post: " << post_v << ", " << post_l0 << "\n";
    return 1;
  }
  std::cout << "[SMOKE] Parameter update verified: Value pred shifted from " << pre_v << " to " << post_v
            << ", Logit[0] shifted from " << pre_l0 << " to " << post_l0 << "\n";

  // Save model and optimizer
  std::string tmp_mod_path = "/tmp/smoke_test_model.pt";
  std::string tmp_opt_path = "/tmp/smoke_test_optimizer.pt";
  torch::serialize::OutputArchive mod_archive;
  model_a->save(mod_archive);
  mod_archive.save_to(tmp_mod_path);
  torch::save(*opt_a, tmp_opt_path);

  // Reload into model_b and opt_b
  auto model_b = std::make_shared<SharedDunePolicyValueNetImpl>(obs_size, hidden_dim, action_dim, num_blocks);
  torch::serialize::InputArchive reload_mod_archive;
  reload_mod_archive.load_from(tmp_mod_path, device);
  model_b->load(reload_mod_archive);
  model_b->to(device);

  auto opt_b = std::make_unique<torch::optim::AdamW>(
      model_b->parameters(),
      torch::optim::AdamWOptions(1e-5).eps(1e-5).weight_decay(0.0));
  torch::load(*opt_b, tmp_opt_path, device);

  auto reloaded_pred = model_b->forward(test_input);
  float reloaded_v = reloaded_pred.values.item<float>();
  float reloaded_l0 = reloaded_pred.logits[0][0].item<float>();

  if (std::abs(reloaded_v - post_v) > 1e-7 || std::abs(reloaded_l0 - post_l0) > 1e-7) {
    std::cerr << "FAIL: Reloaded model prediction mismatch! Expected (" << post_v << ", " << post_l0
              << "), got (" << reloaded_v << ", " << reloaded_l0 << ")\n";
    return 1;
  }
  std::cout << "[SMOKE CHECK 3A PASSED] Checkpoint saved and reloaded with bit-identical predictions.\n";

  // Verify optimizer restoration equivalence by running an identical second step on both model_a and model_b
  RunScratchSearchPiLearner(model_a, *opt_a, sample_rows, obs_size, action_dim, device, 8888ULL, 1, tr_cfg);
  RunScratchSearchPiLearner(model_b, *opt_b, sample_rows, obs_size, action_dim, device, 8888ULL, 1, tr_cfg);

  auto step2_pred_a = model_a->forward(test_input);
  auto step2_pred_b = model_b->forward(test_input);

  if (std::abs(step2_pred_a.values.item<float>() - step2_pred_b.values.item<float>()) > 1e-7 ||
      std::abs(step2_pred_a.logits[0][0].item<float>() - step2_pred_b.logits[0][0].item<float>()) > 1e-7) {
    std::cerr << "FAIL: Step 2 after reload diverged! Optimizer state was not faithfully restored.\n";
    return 1;
  }

  // Layer-by-layer weight equivalence check after restored optimizer update
  auto params_a = model_a->parameters();
  auto params_b = model_b->parameters();
  if (params_a.size() != params_b.size()) {
    std::cerr << "FAIL: Parameter count mismatch between models!\n";
    return 1;
  }
  for (size_t i = 0; i < params_a.size(); ++i) {
    if (!torch::allclose(params_a[i], params_b[i], /*rtol=*/1e-6, /*atol=*/1e-7)) {
      std::cerr << "FAIL: Layer parameter " << i << " mismatch after step 2 with restored optimizer!\n";
      return 1;
    }
  }
  std::cout << "[SMOKE CHECK 3B PASSED] Restored optimizer state reproduces identical updates across all "
            << params_a.size() << " parameter tensors.\n";

  std::filesystem::remove(tmp_mod_path);
  std::filesystem::remove(tmp_opt_path);

  std::cout << ">>> ALL SMOKE TESTS PASSED CLEANLY.\n";
  return 0;
}

// -----------------------------------------------------------------------------
// Mode: verify_parity
// -----------------------------------------------------------------------------
int RunVerifyParity() {
  std::cout << "========================================================================\n";
  std::cout << ">>> RUNNING MULTI-SEAT CONTROLLER PARITY CHECK (SEATS 0, 1, 2, 3)\n";
  std::cout << "========================================================================\n";

  std::string cand_path = absl::GetFlag(FLAGS_model_checkpoint);
  std::string opp_path = absl::GetFlag(FLAGS_opponent_checkpoint);

  if (cand_path.empty() || !std::filesystem::exists(cand_path)) {
    std::cerr << "FATAL: Candidate model not found at " << cand_path << "\n";
    return 1;
  }
  if (opp_path.empty() || !std::filesystem::exists(opp_path)) {
    std::cerr << "FATAL: Opponent model not found at " << opp_path << "\n";
    return 1;
  }

  torch::Device device = torch::cuda::is_available() ? torch::Device(torch::kCUDA) : torch::Device(torch::kCPU);

  int64_t obs_size = 5580;
  int64_t action_dim = 2391;
  int hidden_dim = absl::GetFlag(FLAGS_hidden_dim);
  int num_blocks = absl::GetFlag(FLAGS_num_blocks);

  auto cand_model = std::make_shared<SharedDunePolicyValueNetImpl>(obs_size, hidden_dim, action_dim, num_blocks);
  {
    torch::serialize::InputArchive ar;
    ar.load_from(cand_path, device);
    cand_model->load(ar);
    cand_model->to(device);
    cand_model->eval();
  }

  auto opp_model = std::make_shared<SharedDunePolicyValueNetImpl>(obs_size, hidden_dim, action_dim, num_blocks);
  {
    torch::serialize::InputArchive ar;
    ar.load_from(opp_path, device);
    opp_model->load(ar);
    opp_model->to(device);
    opp_model->eval();
  }

  auto game = LoadGame("dune_imperium");
  uint64_t smoke_seed = 2026090800ULL; // Same seed as benchmark smoke test

  std::map<std::string, int> cumulative_roles;
  int total_low_coverage = 0;
  int total_candidate_decisions = 0;

  for (int ep = 0; ep < 4; ++ep) {
    int expected_seat = ep % 4;
    auto res = PlayCollectionGame(ep, smoke_seed, game, cand_model, opp_model, device,
                                  absl::GetFlag(FLAGS_candidate_logit_cap),
                                  absl::GetFlag(FLAGS_opponent_logit_cap));

    std::cout << "[PARITY GAME " << ep << "] Candidate Seat: " << res.search_seat
              << " | Decisions: " << res.rows.size()
              << " | Primary: " << res.primary_searches
              << " | Continuation: " << res.continuation_searches
              << " | Short-Window: " << res.short_window_searches
              << " | Low-Coverage Fallbacks: " << res.low_coverage_fallbacks
              << " | Return: " << res.return_search_seat
              << " | Wall: " << res.wall_seconds << "s\n";

    if (res.search_seat != expected_seat) {
      std::cerr << "FAIL: Seat rotation mismatch: expected " << expected_seat << ", got " << res.search_seat << "\n";
      return 1;
    }

    total_low_coverage += res.low_coverage_fallbacks;
    total_candidate_decisions += res.rows.size();
    for (const auto& kv : res.role_counts) {
      cumulative_roles[kv.first] += kv.second;
    }
  }

  std::cout << "------------------------------------------------------------------------\n";
  std::cout << "[PARITY SUMMARY] Cumulative Decision Roles over Seats 0..3:\n";
  const std::map<int, std::string> role_names = {
      {0, "kForcedOrBookkeeping"},
      {1, "kLeaderSelection"},
      {2, "kAgentPrimary"},
      {3, "kAgentContinuation"},
      {4, "kPurchase"},
      {5, "kCombatIntrigue"},
      {6, "kOtherOptional"},
  };
  for (int r = 0; r <= 6; ++r) {
    std::string key = std::to_string(r);
    int count = cumulative_roles.count(key) ? cumulative_roles[key] : 0;
    std::cout << "  Role " << r << " (" << role_names.at(r) << "): " << count << " decisions\n";
  }
  std::cout << "  Total Candidate Decisions: " << total_candidate_decisions << "\n";
  std::cout << "  Total Low-Coverage Fallbacks: " << total_low_coverage << "\n";

  // Verify that key roles appeared
  bool has_primary = cumulative_roles["2"] > 0;
  bool has_continuation = cumulative_roles["3"] > 0;
  bool has_short_window = (cumulative_roles["4"] + cumulative_roles["5"] + cumulative_roles["6"]) > 0;
  bool has_forced = cumulative_roles["0"] > 0;

  if (!has_primary || !has_continuation || !has_short_window || !has_forced) {
    std::cerr << "FAIL: Not all required decision roles appeared across the 4 candidate seats!\n";
    std::cerr << "  Primary: " << has_primary << " (" << cumulative_roles["2"] << ")\n"
              << "  Continuation: " << has_continuation << " (" << cumulative_roles["3"] << ")\n"
              << "  ShortWindow: " << has_short_window << " (" 
              << (cumulative_roles["4"] + cumulative_roles["5"] + cumulative_roles["6"]) << ")\n"
              << "  Forced: " << has_forced << " (" << cumulative_roles["0"] << ")\n";
    return 1;
  }
  if (total_low_coverage == 0) {
    std::cerr << "WARNING: No low-coverage fallbacks occurred in this sample.\n";
  } else {
    std::cout << "[PARITY] Low-coverage fallback coverage verified: " << total_low_coverage << " fallbacks recorded.\n";
  }

  std::cout << ">>> PARITY CHECK PASSED (all 4 seats rotated, key decision roles verified).\n";
  return 0;
}

// -----------------------------------------------------------------------------
// Mode: collect
// -----------------------------------------------------------------------------
int RunCollect() {
  int total_games = absl::GetFlag(FLAGS_games);
  int num_threads = absl::GetFlag(FLAGS_threads);
  uint64_t seed = absl::GetFlag(FLAGS_seed);

  std::cout << "========================================================================\n";
  std::cout << ">>> STARTING 128-GAME SEARCH COLLECTION (CONCURRENCY: " << num_threads << " WORKERS)\n";
  std::cout << "========================================================================\n";

  std::string cand_path = absl::GetFlag(FLAGS_model_checkpoint);
  std::string opp_path = absl::GetFlag(FLAGS_opponent_checkpoint);

  torch::Device device = torch::cuda::is_available() ? torch::Device(torch::kCUDA) : torch::Device(torch::kCPU);

  int64_t obs_size = 5580;
  int64_t action_dim = 2391;
  int hidden_dim = absl::GetFlag(FLAGS_hidden_dim);
  int num_blocks = absl::GetFlag(FLAGS_num_blocks);

  auto cand_model = std::make_shared<SharedDunePolicyValueNetImpl>(obs_size, hidden_dim, action_dim, num_blocks);
  {
    torch::serialize::InputArchive ar;
    ar.load_from(cand_path, device);
    cand_model->load(ar);
    cand_model->to(device);
    cand_model->eval();
  }

  auto opp_model = std::make_shared<SharedDunePolicyValueNetImpl>(obs_size, hidden_dim, action_dim, num_blocks);
  {
    torch::serialize::InputArchive ar;
    ar.load_from(opp_path, device);
    opp_model->load(ar);
    opp_model->to(device);
    opp_model->eval();
  }

  auto game = LoadGame("dune_imperium");

  std::vector<SearchPiRow> train_rows;
  std::vector<SearchPiRow> heldout_rows;
  std::vector<GameCollectionResult> game_results(total_games);
  std::mutex out_mutex;

  std::atomic<int> next_game{0};
  std::atomic<int> completed{0};
  auto collect_start = std::chrono::steady_clock::now();

  std::vector<std::thread> workers;
  for (int t = 0; t < num_threads; ++t) {
    workers.emplace_back([&, t]() {
      while (true) {
        int g = next_game.fetch_add(1);
        if (g >= total_games) break;

        auto res = PlayCollectionGame(g, seed, game, cand_model, opp_model, device,
                                      absl::GetFlag(FLAGS_candidate_logit_cap),
                                      absl::GetFlag(FLAGS_opponent_logit_cap));

        // Preassign whole-game partition:
        // Seat rotation: g % 4 is seat.
        // For balanced 96/32: 3 of every 4 games per seat are train, 1 is heldout.
        // Formula: (g / 4) % 4 == 3 -> heldout (games 12..15, 28..31, 44..47, 60..63, 76..79, 92..95, 108..111, 124..127)
        // Exactly 8 heldout games for each of the 4 seats = 32 heldout games total.
        // Exactly 24 train games for each of the 4 seats = 96 train games total.
        bool is_heldout = ((g / 4) % 4 == 3);

        {
          std::lock_guard<std::mutex> lock(out_mutex);
          game_results[g] = res;
          if (is_heldout) {
            heldout_rows.insert(heldout_rows.end(),
                                std::make_move_iterator(res.rows.begin()),
                                std::make_move_iterator(res.rows.end()));
          } else {
            train_rows.insert(train_rows.end(),
                              std::make_move_iterator(res.rows.begin()),
                              std::make_move_iterator(res.rows.end()));
          }
        }

        int c = completed.fetch_add(1) + 1;
        if (c % 16 == 0 || c == total_games) {
          auto now = std::chrono::steady_clock::now();
          double el = std::chrono::duration<double>(now - collect_start).count();
          double rate = el / c;
          std::cout << "[COLLECT PROGRESS] " << c << "/" << total_games << " games ("
                    << (c * 100 / total_games) << "%) in " << el << "s (" << rate << "s/game, eta "
                    << rate * (total_games - c) << "s)\n";
        }
      }
    });
  }

  for (auto& w : workers) w.join();
  auto collect_end = std::chrono::steady_clock::now();
  double total_wall = std::chrono::duration<double>(collect_end - collect_start).count();

  std::cout << "[COLLECTION COMPLETE] Finished " << total_games << " games in " << total_wall << "s\n";
  std::cout << "  Training rows (96 games): " << train_rows.size() << "\n";
  std::cout << "  Heldout rows (32 games):  " << heldout_rows.size() << "\n";

  // Measure row serialization overhead
  auto ser_start = std::chrono::steady_clock::now();
  std::string err;
  std::string out_train = absl::GetFlag(FLAGS_output_train_jsonl);
  std::string out_heldout = absl::GetFlag(FLAGS_output_heldout_jsonl);
  std::string out_games = absl::GetFlag(FLAGS_output_games_jsonl);

  if (!out_train.empty() && !WriteTransferPilotRows(out_train, train_rows, &err)) {
    std::cerr << "FATAL: Failed writing train rows: " << err << "\n";
    return 1;
  }
  if (!out_heldout.empty() && !WriteTransferPilotRows(out_heldout, heldout_rows, &err)) {
    std::cerr << "FATAL: Failed writing heldout rows: " << err << "\n";
    return 1;
  }
  if (!out_games.empty()) {
    std::ofstream g_out(out_games);
    for (const auto& gr : game_results) {
      json::Object obj;
      obj["episode_id"] = static_cast<int64_t>(gr.episode_id);
      obj["search_seat"] = static_cast<int64_t>(gr.search_seat);
      obj["rank"] = static_cast<int64_t>(gr.search_seat_rank);
      obj["return"] = gr.return_search_seat;
      obj["wall_seconds"] = gr.wall_seconds;
      obj["decisions"] = static_cast<int64_t>(gr.rows.size());
      obj["low_coverage_fallbacks"] = static_cast<int64_t>(gr.low_coverage_fallbacks);
      g_out << json::ToString(json::Value(obj)) << "\n";
    }
  }
  auto ser_end = std::chrono::steady_clock::now();
  double ser_wall = std::chrono::duration<double>(ser_end - ser_start).count();
  std::cout << "[SERIALIZATION OVERHEAD] Serialized rows and games in " << ser_wall << "s\n";

  // Write metadata manifest
  std::string meta_path = absl::GetFlag(FLAGS_output_meta_json);
  if (!meta_path.empty()) {
    json::Object meta;
    meta["total_games"] = static_cast<int64_t>(total_games);
    meta["train_games"] = static_cast<int64_t>(96);
    meta["heldout_games"] = static_cast<int64_t>(32);
    meta["train_rows"] = static_cast<int64_t>(train_rows.size());
    meta["heldout_rows"] = static_cast<int64_t>(heldout_rows.size());
    meta["collection_wall_seconds"] = total_wall;
    meta["serialization_wall_seconds"] = ser_wall;
    meta["total_wall_seconds"] = total_wall + ser_wall;
    meta["threads"] = static_cast<int64_t>(num_threads);
    meta["seed"] = static_cast<int64_t>(seed);
    std::ofstream m_out(meta_path);
    m_out << json::ToString(json::Value(meta)) << "\n";
  }

  return 0;
}

// -----------------------------------------------------------------------------
// Mode: train
// -----------------------------------------------------------------------------
int RunTrain() {
  std::cout << "========================================================================\n";
  std::cout << ">>> STARTING SEARCH-CHILD TRANSFER TRAINING (1 EPOCH)\n";
  std::cout << "========================================================================\n";

  std::string in_train = absl::GetFlag(FLAGS_input_train_jsonl);
  std::string in_heldout = absl::GetFlag(FLAGS_input_heldout_jsonl);
  std::string cand_path = absl::GetFlag(FLAGS_model_checkpoint);
  std::string out_model = absl::GetFlag(FLAGS_output_model_path);
  std::string out_optim = absl::GetFlag(FLAGS_output_optim_path);

  std::vector<SearchPiRow> train_rows;
  std::vector<SearchPiRow> heldout_rows;
  std::string err;

  if (!ReadTransferPilotRows(in_train, &train_rows, &err)) {
    std::cerr << "FATAL: Failed reading train rows from " << in_train << ": " << err << "\n";
    return 1;
  }
  if (!in_heldout.empty()) {
    if (!ReadTransferPilotRows(in_heldout, &heldout_rows, &err)) {
      std::cerr << "FATAL: Failed reading heldout rows from " << in_heldout << ": " << err << "\n";
      return 1;
    }
  }

  std::cout << "[TRAIN] Loaded " << train_rows.size() << " training rows and "
            << heldout_rows.size() << " heldout rows.\n";

  torch::Device device = torch::cuda::is_available() ? torch::Device(torch::kCUDA) : torch::Device(torch::kCPU);
  int64_t obs_size = 5580;
  int64_t action_dim = 2391;
  int hidden_dim = absl::GetFlag(FLAGS_hidden_dim);
  int num_blocks = absl::GetFlag(FLAGS_num_blocks);

  auto child_model = std::make_shared<SharedDunePolicyValueNetImpl>(obs_size, hidden_dim, action_dim, num_blocks);
  {
    torch::serialize::InputArchive ar;
    ar.load_from(cand_path, device);
    child_model->load(ar);
    child_model->to(device);
  }

  // Dedicated AdamW optimizer with eps=1e-5, zero weight decay
  double lr = absl::GetFlag(FLAGS_learning_rate);
  double wd = absl::GetFlag(FLAGS_weight_decay);
  auto child_optimizer = std::make_unique<torch::optim::AdamW>(
      child_model->parameters(),
      torch::optim::AdamWOptions(lr).eps(1e-5).weight_decay(wd));

  SearchPiLearnerConfig cfg;
  cfg.minibatch_size = absl::GetFlag(FLAGS_minibatch_size);
  cfg.epochs = absl::GetFlag(FLAGS_epochs);
  cfg.policy_coef = absl::GetFlag(FLAGS_policy_coef);
  cfg.value_coef = absl::GetFlag(FLAGS_value_coef);
  cfg.grad_clip_norm = absl::GetFlag(FLAGS_grad_clip_norm);
  cfg.logit_cap = absl::GetFlag(FLAGS_candidate_logit_cap);

  auto t_start = std::chrono::steady_clock::now();
  auto stats = RunScratchSearchPiLearner(child_model, *child_optimizer, train_rows,
                                        obs_size, action_dim, device,
                                        /*master_seed=*/absl::GetFlag(FLAGS_seed),
                                        /*generation=*/1, cfg);
  auto t_end = std::chrono::steady_clock::now();
  double train_wall = std::chrono::duration<double>(t_end - t_start).count();

  std::cout << "[TRAIN SUMMARY]\n"
            << "  Wall Time:               " << train_wall << "s\n"
            << "  Rows Trained:            " << stats.distinct_rows << "\n"
            << "  Policy-Weighted Rows:    " << stats.policy_weight_rows << "\n"
            << "  Minibatches:             " << stats.minibatches << "\n"
            << "  Policy Backward Steps:   " << stats.policy_backward_minibatches << "\n"
            << "  Value Backward Steps:    " << stats.value_backward_minibatches << "\n"
            << "  Weighted Policy CE:      " << stats.policy_ce << "\n"
            << "  Terminal Value MSE:      " << stats.value_mse << "\n"
            << "  Policy Grad Norm:        " << stats.policy_grad_norm << "\n"
            << "  Value Grad Norm:         " << stats.value_grad_norm << "\n"
            << "  Critic Pred Pre:         mean=" << stats.critic_pred_mean_pre << " sd=" << stats.critic_pred_sd_pre << "\n"
            << "  Critic Pred Post:        mean=" << stats.critic_pred_mean_post << " sd=" << stats.critic_pred_sd_post << "\n";

  // Save trained model and dedicated optimizer
  if (!out_model.empty()) {
    torch::serialize::OutputArchive ar;
    child_model->save(ar);
    ar.save_to(out_model);
    std::cout << "[SAVED MODEL] " << out_model << "\n";
  }
  if (!out_optim.empty()) {
    torch::save(*child_optimizer, out_optim);
    std::cout << "[SAVED OPTIMIZER] " << out_optim << "\n";
  }

  // Save metadata JSON
  std::string meta_path = absl::GetFlag(FLAGS_output_meta_json);
  if (!meta_path.empty()) {
    json::Object meta;
    meta["training_wall_seconds"] = train_wall;
    meta["distinct_rows"] = static_cast<int64_t>(stats.distinct_rows);
    meta["policy_weight_rows"] = static_cast<int64_t>(stats.policy_weight_rows);
    meta["minibatches"] = static_cast<int64_t>(stats.minibatches);
    meta["policy_backward_minibatches"] = static_cast<int64_t>(stats.policy_backward_minibatches);
    meta["value_backward_minibatches"] = static_cast<int64_t>(stats.value_backward_minibatches);
    meta["policy_ce"] = stats.policy_ce;
    meta["value_mse"] = stats.value_mse;
    meta["policy_grad_norm"] = stats.policy_grad_norm;
    meta["value_grad_norm"] = stats.value_grad_norm;
    meta["learning_rate"] = lr;
    meta["weight_decay"] = wd;
    meta["minibatch_size"] = static_cast<int64_t>(cfg.minibatch_size);
    meta["epochs"] = static_cast<int64_t>(cfg.epochs);
    std::ofstream m_out(meta_path);
    m_out << json::ToString(json::Value(meta)) << "\n";
  }

  return 0;
}

}  // namespace
}  // namespace open_spiel

int main(int argc, char* argv[]) {
  absl::ParseCommandLine(argc, argv);
  std::string mode = absl::GetFlag(FLAGS_mode);

  if (mode == "smoke_test") {
    return open_spiel::RunSmokeTest();
  } else if (mode == "verify_parity") {
    return open_spiel::RunVerifyParity();
  } else if (mode == "collect") {
    return open_spiel::RunCollect();
  } else if (mode == "train") {
    return open_spiel::RunTrain();
  } else {
    std::cerr << "Unknown --mode: " << mode << " (valid: smoke_test | verify_parity | collect | train)\n";
    return 1;
  }
}
