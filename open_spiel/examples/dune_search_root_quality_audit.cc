// Incremental fixed-root Search-PI teacher-quality audit.
//
// Search is run on the registered fixed root set. Changed-action roots are
// selected deterministically, then raw-vs-search terminal rollouts are written
// one JSONL record at a time with an immediate flush. This makes interruption
// useful rather than leaving only an empty final summary.

#include <algorithm>
#include <atomic>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <numeric>
#include <set>
#include <string>
#include <thread>
#include <vector>

#include "open_spiel/abseil-cpp/absl/flags/flag.h"
#include "open_spiel/abseil-cpp/absl/flags/parse.h"
#include "open_spiel/abseil-cpp/absl/random/distributions.h"
#include "open_spiel/games/dune_imperium/dune_imperium.h"
#include "open_spiel/spiel.h"
#include "open_spiel/utils/json.h"
#include <torch/torch.h>

#include "dune_evaluator.h"
#include "dune_network.h"
#include "dune_puct_is_mcts.h"
#include "dune_pwo2_common.h"
#include "dune_pwo3_common.h"
#include "dune_search_session.h"
#include "dune_sha256.h"

ABSL_FLAG(std::string, model_checkpoint, "", "Model checkpoint.");
ABSL_FLAG(int, hidden_dim, 2048, "Model hidden dimension.");
ABSL_FLAG(int, num_blocks, 4, "Model residual block count.");
ABSL_FLAG(bool, nonlinear_value_head, false, "Value-head variant.");
ABSL_FLAG(double, candidate_logit_cap, 10.0, "Legal-centered logit cap.");
ABSL_FLAG(std::string, root_corpus, "data/pwo2_root_corpus.json", "Fixed root corpus.");
ABSL_FLAG(std::string, root_selection_path, "data/pwo3_selections.json", "Root selection JSON.");
ABSL_FLAG(std::string, root_selection_key, "oracle_replication_subset_32", "Root selection key.");
ABSL_FLAG(std::string, root_ids_path, "", "Optional JSON array of exact root IDs to reuse.");
ABSL_FLAG(bool, rollout_all_roots, false,
          "Roll out every loaded root, preserving an externally supplied fixed set.");
ABSL_FLAG(std::string, output_dir, "", "Fresh, empty output directory.");
ABSL_FLAG(std::string, tier, "low", "low or high.");
ABSL_FLAG(int, changed_root_count, 4, "Number of changed-action roots to roll out.");
ABSL_FLAG(int, rollout_count, 8, "Matched rollouts per action.");
ABSL_FLAG(int, threads, 8, "Search/rollout worker threads.");
ABSL_FLAG(int, search_seed, 145, "Root-search seed coordinate.");
ABSL_FLAG(int, max_nodes, 200000, "Fixed-root node ceiling.");
ABSL_FLAG(double, puct_c, 0.30, "PUCT exploration constant.");
ABSL_FLAG(int, min_visit_threshold, 2, "Minimum visits for empirical Q.");
ABSL_FLAG(double, utility_divisor, 4.0, "Search/rollout utility divisor.");
ABSL_FLAG(double, simulated_opponent_temperature, 1.0, "Simulated policy temperature.");
ABSL_FLAG(int, low_primary_simulations, 200, "Low-tier primary budget.");
ABSL_FLAG(int, low_other_simulations, 64, "Low-tier non-primary budget.");
ABSL_FLAG(int, high_primary_simulations, 800, "High-tier primary budget.");
ABSL_FLAG(int, high_other_simulations, 256, "High-tier non-primary budget.");

namespace open_spiel {
namespace {

struct Root {
  std::string root_id;
  std::string stratum;
  std::vector<Action> history;
  std::vector<Action> legal_actions;
  Player player = kInvalidPlayer;
  int round = 0;
  int source_episode_id = 0;
  int decision_index = 0;
};

struct SearchCell {
  Root root;
  int requested = 0;
  int completed = 0;
  bool timeout = false;
  bool fallback = false;
  std::string fallback_reason;
  std::vector<Action> actions;
  std::vector<double> raw_priors;
  std::vector<int> visits;
  std::vector<double> q_values;
  Action raw_action = kInvalidAction;
  Action selected_action = kInvalidAction;
  double q_raw = 0.0;
  double q_selected = 0.0;
  std::vector<double> raw_outcomes;
  std::vector<double> selected_outcomes;
};

std::unique_ptr<State> Reconstruct(const std::shared_ptr<const Game>& game,
                                   const std::vector<Action>& history) {
  auto state = game->NewInitialState();
  for (Action action : history) state->ApplyAction(action);
  return state;
}

std::shared_ptr<SharedDunePolicyValueNetImpl> LoadModel(
    const std::string& path, int64_t observation_size, int64_t action_size,
    const torch::Device& device) {
  auto model = std::make_shared<SharedDunePolicyValueNetImpl>(
      observation_size, absl::GetFlag(FLAGS_hidden_dim), action_size,
      absl::GetFlag(FLAGS_num_blocks),
      absl::GetFlag(FLAGS_nonlinear_value_head));
  torch::serialize::InputArchive archive;
  archive.load_from(path, device);
  model->load(archive);
  model->to(device);
  model->eval();
  return model;
}

Action ArgmaxLowestId(const std::vector<Action>& actions,
                      const std::vector<double>& weights) {
  SPIEL_CHECK_EQ(actions.size(), weights.size());
  SPIEL_CHECK_GT(actions.size(), 0u);
  Action best = actions.front();
  double best_weight = weights.front();
  for (size_t i = 1; i < actions.size(); ++i) {
    if (weights[i] > best_weight ||
        (weights[i] == best_weight && actions[i] < best)) {
      best = actions[i];
      best_weight = weights[i];
    }
  }
  return best;
}

Action ArgmaxPolicy(const ActionsAndProbs& policy) {
  SPIEL_CHECK_GT(policy.size(), 0u);
  Action best = policy.front().first;
  double best_weight = policy.front().second;
  for (const auto& pair : policy) {
    if (pair.second > best_weight ||
        (pair.second == best_weight && pair.first < best)) {
      best = pair.first;
      best_weight = pair.second;
    }
  }
  return best;
}

double QFor(const std::vector<Action>& actions, const std::vector<double>& q,
            Action action) {
  for (size_t i = 0; i < actions.size(); ++i)
    if (actions[i] == action) return q[i];
  SpielFatalError("Q vector omitted legal action");
}

std::string Exact(double value) { return absl::StrFormat("%.17g", value); }

template <typename T>
json::Array JsonArray(const std::vector<T>& values) {
  json::Array result;
  for (const T& value : values) result.push_back(value);
  return result;
}

void LoadRoots(const std::shared_ptr<const Game>& game,
               const std::string& corpus_path, const std::string& selection_path,
               const std::string& selection_key, const std::string& ids_path,
               std::vector<Root>* roots) {
  std::ifstream corpus_file(corpus_path);
  if (!corpus_file) SpielFatalError("cannot open root corpus: " + corpus_path);
  std::string corpus_text((std::istreambuf_iterator<char>(corpus_file)),
                         std::istreambuf_iterator<char>());
  auto corpus = json::FromString(corpus_text);
  if (!corpus) SpielFatalError("invalid root corpus JSON");
  std::map<std::string, Root> by_id;
  for (const auto& value : corpus.value().GetArray()) {
    auto object = value.GetObject();
    Root root;
    root.root_id = object.at("root_id").GetString();
    root.stratum = object.at("stratum").GetString();
    root.player = static_cast<Player>(object.at("player").GetInt());
    root.round = static_cast<int>(object.at("round").GetInt());
    root.source_episode_id = static_cast<int>(object.at("source_episode_id").GetInt());
    root.decision_index = static_cast<int>(object.at("decision_index").GetInt());
    for (const auto& a : object.at("history").GetArray())
      root.history.push_back(static_cast<Action>(a.GetInt()));
    for (const auto& a : object.at("legal_actions").GetArray())
      root.legal_actions.push_back(static_cast<Action>(a.GetInt()));
    if (!by_id.emplace(root.root_id, root).second)
      SpielFatalError("duplicate root ID in corpus: " + root.root_id);
  }

  std::vector<std::string> selected_ids;
  if (!ids_path.empty()) {
    std::ifstream ids_file(ids_path);
    if (!ids_file) SpielFatalError("cannot open root IDs: " + ids_path);
    std::string text((std::istreambuf_iterator<char>(ids_file)),
                     std::istreambuf_iterator<char>());
    auto parsed = json::FromString(text);
    if (!parsed) SpielFatalError("invalid root IDs JSON");
    for (const auto& value : parsed.value().GetArray())
      selected_ids.push_back(value.GetString());
  } else {
    std::ifstream selection_file(selection_path);
    if (!selection_file) SpielFatalError("cannot open root selection: " + selection_path);
    std::string text((std::istreambuf_iterator<char>(selection_file)),
                     std::istreambuf_iterator<char>());
    auto parsed = json::FromString(text);
    if (!parsed) SpielFatalError("invalid root selection JSON");
    auto object = parsed.value().GetObject();
    auto it = object.find(selection_key);
    if (it == object.end()) SpielFatalError("missing selection key: " + selection_key);
    for (const auto& value : it->second.GetArray())
      selected_ids.push_back(value.GetString());
  }
  std::set<std::string> seen;
  for (const std::string& id : selected_ids) {
    if (!seen.insert(id).second) SpielFatalError("duplicate selected root: " + id);
    auto it = by_id.find(id);
    if (it == by_id.end()) SpielFatalError("selected root missing: " + id);
    Root root = it->second;
    auto state = Reconstruct(game, root.history);
    SPIEL_CHECK_EQ(pwo2::HistoryHash(root.history), root.root_id);
    SPIEL_CHECK_TRUE(pwo3::IsAscending(root.legal_actions));
    SPIEL_CHECK_EQ(state->CurrentPlayer(), root.player);
    SPIEL_CHECK_EQ(state->LegalActions(), root.legal_actions);
    roots->push_back(std::move(root));
  }
  if (roots->empty()) SpielFatalError("empty fixed root selection");
}

std::vector<double> Rollouts(const State& after_action, Player owner,
                             algorithms::Evaluator* evaluator,
                             const std::string& root_id, int count,
                             double utility_divisor) {
  std::vector<double> values(count);
  for (int k = 0; k < count; ++k) {
    std::mt19937 rng = dune_seed::MakeRng32(
        pwo2::OracleContinuationSeed(root_id, k));
    auto state = after_action.Clone();
    while (!state->IsTerminal()) {
      if (state->IsChanceNode()) {
        state->ApplyAction(SampleAction(
            state->ChanceOutcomes(), absl::Uniform(rng, 0.0, 1.0)).first);
      } else if (state->CurrentPlayer() == kSimultaneousPlayerId) {
        std::vector<Action> joint;
        for (int p = 0; p < state->NumPlayers(); ++p) {
          auto actions = state->LegalActions(p);
          std::uniform_int_distribution<int> dist(0, actions.size() - 1);
          joint.push_back(actions[dist(rng)]);
        }
        state->ApplyActions(joint);
      } else {
        state->ApplyAction(SampleActionFromPrior(
            evaluator->Prior(*state), absl::Uniform(rng, 0.0, 1.0)));
      }
    }
    values[k] = state->Returns()[owner] / utility_divisor;
  }
  return values;
}

double Mean(const std::vector<double>& values) {
  return values.empty() ? 0.0
                        : std::accumulate(values.begin(), values.end(), 0.0) /
                              values.size();
}

json::Object Stats(const std::vector<double>& values) {
  json::Object result;
  const double mean = Mean(values);
  double ss = 0.0;
  for (double v : values) ss += (v - mean) * (v - mean);
  const double sd = values.size() > 1 ? std::sqrt(ss / (values.size() - 1)) : 0.0;
  const double se = values.empty() ? 0.0 : sd / std::sqrt(values.size());
  result["n"] = static_cast<int64_t>(values.size());
  result["mean"] = mean;
  result["sd"] = sd;
  result["se"] = se;
  result["ci95_low"] = mean - 1.96 * se;
  result["ci95_high"] = mean + 1.96 * se;
  return result;
}

double Pearson(const std::vector<double>& x, const std::vector<double>& y) {
  if (x.size() != y.size() || x.size() < 2) return 0.0;
  const double mx = Mean(x), my = Mean(y);
  double num = 0.0, xx = 0.0, yy = 0.0;
  for (size_t i = 0; i < x.size(); ++i) {
    const double a = x[i] - mx, b = y[i] - my;
    num += a * b;
    xx += a * a;
    yy += b * b;
  }
  return xx > 0.0 && yy > 0.0 ? num / std::sqrt(xx * yy) : 0.0;
}

json::Object Correlation(const std::vector<double>& x,
                         const std::vector<double>& y) {
  json::Object result;
  result["n"] = static_cast<int64_t>(x.size());
  result["pearson"] = Pearson(x, y);
  return result;
}

void PrepareOutput(const std::filesystem::path& output_dir) {
  if (output_dir.empty()) SpielFatalError("--output_dir is required");
  std::error_code ec;
  if (std::filesystem::exists(output_dir, ec))
    SpielFatalError("refusing existing/nonempty output directory: " + output_dir.string());
  if (!std::filesystem::create_directory(output_dir, ec) || ec)
    SpielFatalError("cannot create output directory: " + output_dir.string());
  std::ofstream lock(output_dir / ".writer.lock");
  lock << "incremental_fixed_root_quality_audit\n";
}

}  // namespace
}  // namespace open_spiel

using namespace open_spiel;

int main(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);
  at::set_num_threads(1);
  at::set_num_interop_threads(1);
  const std::string model_path = absl::GetFlag(FLAGS_model_checkpoint);
  if (model_path.empty()) SpielFatalError("--model_checkpoint is required");
  if (absl::GetFlag(FLAGS_tier) != "low" && absl::GetFlag(FLAGS_tier) != "high")
    SpielFatalError("--tier must be low or high");
  if (absl::GetFlag(FLAGS_rollout_count) <= 0 ||
      (!absl::GetFlag(FLAGS_rollout_all_roots) &&
       absl::GetFlag(FLAGS_changed_root_count) <= 0) ||
      absl::GetFlag(FLAGS_threads) <= 0)
    SpielFatalError("rollout_count, changed_root_count and threads must be positive");
  PrepareOutput(absl::GetFlag(FLAGS_output_dir));
  const std::filesystem::path output_dir = absl::GetFlag(FLAGS_output_dir);

  auto game = LoadGame("dune_imperium");
  const auto device = torch::cuda::is_available() ? torch::Device(torch::kCUDA)
                                                  : torch::Device(torch::kCPU);
  auto model = LoadModel(model_path, game->InformationStateTensorSize(),
                         game->NumDistinctActions(), device);
  const std::string model_sha = ComputeFileSHA256(model_path);
  std::vector<Root> roots;
  LoadRoots(game, absl::GetFlag(FLAGS_root_corpus),
            absl::GetFlag(FLAGS_root_selection_path),
            absl::GetFlag(FLAGS_root_selection_key),
            absl::GetFlag(FLAGS_root_ids_path), &roots);

  const bool high = absl::GetFlag(FLAGS_tier) == "high";
  const int primary_budget = high ? absl::GetFlag(FLAGS_high_primary_simulations)
                                  : absl::GetFlag(FLAGS_low_primary_simulations);
  const int other_budget = high ? absl::GetFlag(FLAGS_high_other_simulations)
                                : absl::GetFlag(FLAGS_low_other_simulations);

  json::Object preflight;
  preflight["protocol"] = "incremental_fixed_root_teacher_quality_audit";
  preflight["model_checkpoint"] = model_path;
  preflight["model_sha256"] = model_sha;
  preflight["tier"] = absl::GetFlag(FLAGS_tier);
  preflight["root_count_searched"] = static_cast<int64_t>(roots.size());
  preflight["changed_root_count_requested"] =
      static_cast<int64_t>(absl::GetFlag(FLAGS_changed_root_count));
  preflight["rollout_all_roots"] = absl::GetFlag(FLAGS_rollout_all_roots);
  preflight["rollout_count_per_action"] =
      static_cast<int64_t>(absl::GetFlag(FLAGS_rollout_count));
  const int rollout_roots = absl::GetFlag(FLAGS_rollout_all_roots)
                                ? static_cast<int>(roots.size())
                                : absl::GetFlag(FLAGS_changed_root_count);
  preflight["expected_rollout_records"] = static_cast<int64_t>(
      2 * rollout_roots *
      absl::GetFlag(FLAGS_rollout_count));
  preflight["rollout_seed_scheme"] = "pwo2::OracleContinuationSeed(root_id,k)";
  preflight["primary_budget"] = static_cast<int64_t>(primary_budget);
  preflight["other_budget"] = static_cast<int64_t>(other_budget);
  preflight["opponent_mode"] = "policy";
  preflight["candidate_logit_cap"] = absl::GetFlag(FLAGS_candidate_logit_cap);
  {
    std::ofstream file(output_dir / "preflight.json");
    file << json::ToString(preflight, true) << std::endl;
  }

  std::vector<SearchCell> cells(roots.size());
  std::atomic<size_t> next{0};
  std::mutex error_mutex;
  std::string error;
  auto worker = [&]() {
    auto evaluator = std::make_shared<DuneNNEvaluator>(
        model, device, static_cast<float>(absl::GetFlag(FLAGS_candidate_logit_cap)));
    while (true) {
      const size_t i = next.fetch_add(1);
      if (i >= roots.size()) break;
      SearchCell& cell = cells[i];
      cell.root = roots[i];
      const bool primary = cell.root.stratum == "agent_primary";
      cell.requested = primary ? primary_budget : other_budget;
      auto state = Reconstruct(game, cell.root.history);
      DuneSearchConfig config;
      config.max_simulations = cell.requested;
      config.relative_time_budget_ms = std::numeric_limits<double>::infinity();
      config.max_nodes = absl::GetFlag(FLAGS_max_nodes);
      config.puct_c = absl::GetFlag(FLAGS_puct_c);
      config.opponent_mode = SearchOpponentMode::kPolicy;
      config.temperature = 0.0;
      config.opponent_temperature = absl::GetFlag(FLAGS_simulated_opponent_temperature);
      config.utility_divisor = absl::GetFlag(FLAGS_utility_divisor);
      config.min_visit_threshold = absl::GetFlag(FLAGS_min_visit_threshold);
      config.covered_prior_threshold = 0.0;
      config.root_prior_temperature = 1.0;
      config.seed = pwo2::SearchRngSeed(absl::GetFlag(FLAGS_search_seed), cell.root.root_id);
      DunePUCTISMCTSBot bot(config, evaluator);
      DuneSearchResult result = bot.RunSearch(
          *state, cell.requested, std::numeric_limits<double>::infinity(), 0);
      if (result.simulations_completed != cell.requested || result.timeout_status ||
          result.diagnostics.inherited_root_visits != 0) {
        std::lock_guard<std::mutex> lock(error_mutex);
        if (error.empty())
          error = absl::StrFormat("search health failure at root %s: %d/%d",
                                  cell.root.root_id, result.simulations_completed,
                                  cell.requested);
        continue;
      }
      cell.completed = result.simulations_completed;
      cell.timeout = result.timeout_status;
      cell.fallback = result.used_fallback;
      cell.fallback_reason = result.fallback_reason;
      cell.actions = result.diagnostics.actions;
      cell.visits = result.diagnostics.visit_counts;
      cell.q_values = result.diagnostics.q_values;
      cell.raw_priors = result.diagnostics.raw_priors.empty()
                            ? result.diagnostics.priors
                            : result.diagnostics.raw_priors;
      cell.raw_action = ArgmaxLowestId(cell.actions, cell.raw_priors);
      cell.selected_action = ArgmaxPolicy(result.policy);
      cell.q_raw = QFor(cell.actions, cell.q_values, cell.raw_action);
      cell.q_selected = QFor(cell.actions, cell.q_values, cell.selected_action);
    }
  };
  std::vector<std::thread> workers;
  for (int i = 0; i < absl::GetFlag(FLAGS_threads); ++i) workers.emplace_back(worker);
  for (auto& thread : workers) thread.join();
  if (!error.empty()) SpielFatalError(error);

  std::ofstream search_file(output_dir / "search.jsonl");
  std::ofstream selected_file(output_dir / "selected_roots.json");
  std::vector<size_t> selected_indices;
  for (size_t i = 0; i < cells.size(); ++i) {
    if (cells[i].selected_action != cells[i].raw_action)
      selected_indices.push_back(i);
  }
  const int wanted = absl::GetFlag(FLAGS_rollout_all_roots)
                         ? static_cast<int>(cells.size())
                         : absl::GetFlag(FLAGS_changed_root_count);
  if (absl::GetFlag(FLAGS_rollout_all_roots)) {
    selected_indices.clear();
    for (size_t i = 0; i < cells.size(); ++i) selected_indices.push_back(i);
  } else {
    if (static_cast<int>(selected_indices.size()) < wanted)
      SpielFatalError(absl::StrFormat("only %d changed roots found, need %d",
                                      selected_indices.size(), wanted));
    selected_indices.resize(wanted);
  }
  json::Array selected_ids;
  for (size_t i : selected_indices) selected_ids.push_back(cells[i].root.root_id);
  selected_file << json::ToString(selected_ids, true) << std::endl;
  for (const SearchCell& cell : cells) {
    json::Object row;
    row["tier"] = absl::GetFlag(FLAGS_tier);
    row["root_id"] = cell.root.root_id;
    row["stratum"] = cell.root.stratum;
    row["requested_simulations"] = static_cast<int64_t>(cell.requested);
    row["simulations_completed"] = static_cast<int64_t>(cell.completed);
    row["fallback"] = cell.fallback;
    row["fallback_reason"] = cell.fallback_reason;
    row["raw_action"] = static_cast<int64_t>(cell.raw_action);
    row["selected_action"] = static_cast<int64_t>(cell.selected_action);
    row["action_changed"] = cell.raw_action != cell.selected_action;
    row["q_raw"] = cell.q_raw;
    row["q_selected"] = cell.q_selected;
    row["q_delta"] = cell.q_selected - cell.q_raw;
    row["root_actions"] = JsonArray(cell.actions);
    row["q_values"] = JsonArray(cell.q_values);
    row["visit_counts"] = JsonArray(cell.visits);
    search_file << json::ToString(row, false) << std::endl;
  }

  std::ofstream rollout_file(output_dir / "rollouts.jsonl", std::ios::app);
  std::ofstream progress_file(output_dir / "progress.jsonl", std::ios::app);
  std::mutex write_mutex;
  std::atomic<int> completed_rollouts{0};
  const int total_rollouts = wanted * 2 * absl::GetFlag(FLAGS_rollout_count);
  std::atomic<size_t> next_cell{0};
  auto rollout_worker = [&]() {
    auto evaluator = std::make_shared<DuneNNEvaluator>(
        model, device, static_cast<float>(absl::GetFlag(FLAGS_candidate_logit_cap)));
    while (true) {
      const size_t job = next_cell.fetch_add(1);
      if (job >= selected_indices.size()) break;
      SearchCell& cell = cells[selected_indices[job]];
      cell.raw_outcomes.resize(absl::GetFlag(FLAGS_rollout_count));
      cell.selected_outcomes.resize(absl::GetFlag(FLAGS_rollout_count));
      for (int action_kind = 0; action_kind < 2; ++action_kind) {
        const Action action = action_kind == 0 ? cell.raw_action : cell.selected_action;
        auto state = Reconstruct(game, cell.root.history);
        state->ApplyAction(action);
        const std::vector<double> values = Rollouts(
            *state, cell.root.player, evaluator.get(), cell.root.root_id,
            absl::GetFlag(FLAGS_rollout_count), absl::GetFlag(FLAGS_utility_divisor));
        for (int k = 0; k < static_cast<int>(values.size()); ++k) {
          if (action_kind == 0) cell.raw_outcomes[k] = values[k];
          else cell.selected_outcomes[k] = values[k];
          const int done = ++completed_rollouts;
          json::Object row;
          row["tier"] = absl::GetFlag(FLAGS_tier);
          row["root_id"] = cell.root.root_id;
          row["action_kind"] = action_kind == 0 ? "raw" : "selected";
          row["action"] = static_cast<int64_t>(action);
          row["rollout_index"] = static_cast<int64_t>(k);
          row["seed"] = static_cast<int64_t>(pwo2::OracleContinuationSeed(
              cell.root.root_id, k));
          row["outcome"] = values[k];
          std::lock_guard<std::mutex> lock(write_mutex);
          rollout_file << json::ToString(row, false) << std::endl;
          json::Object progress;
          progress["completed_rollouts"] = static_cast<int64_t>(done);
          progress["total_rollouts"] = static_cast<int64_t>(total_rollouts);
          progress["root_id"] = cell.root.root_id;
          progress["action_kind"] = action_kind == 0 ? "raw" : "selected";
          progress_file << json::ToString(progress, false) << std::endl;
        }
      }
    }
  };
  workers.clear();
  for (int i = 0; i < absl::GetFlag(FLAGS_threads); ++i)
    workers.emplace_back(rollout_worker);
  for (auto& thread : workers) thread.join();

  if (completed_rollouts != total_rollouts)
    SpielFatalError("rollout count mismatch after workers joined");

  std::vector<double> q_delta, outcome_delta, q_cells, outcome_cells;
  int positive = 0, negative = 0, sign_agree = 0;
  for (size_t index : selected_indices) {
    const SearchCell& cell = cells[index];
    const double raw_mean = Mean(cell.raw_outcomes);
    const double selected_mean = Mean(cell.selected_outcomes);
    const double delta = selected_mean - raw_mean;
    const double qadv = cell.q_selected - cell.q_raw;
    q_delta.push_back(qadv);
    outcome_delta.push_back(delta);
    q_cells.push_back(cell.q_raw);
    outcome_cells.push_back(raw_mean);
    q_cells.push_back(cell.q_selected);
    outcome_cells.push_back(selected_mean);
    if (delta > 0.0) ++positive;
    if (delta < 0.0) ++negative;
    if (qadv != 0.0 && delta != 0.0 && (qadv > 0.0) == (delta > 0.0)) ++sign_agree;
  }
  json::Object summary;
  summary["protocol"] = "incremental_fixed_root_teacher_quality_audit";
  summary["model_checkpoint"] = model_path;
  summary["model_sha256"] = model_sha;
  summary["tier"] = absl::GetFlag(FLAGS_tier);
  summary["searched_roots"] = static_cast<int64_t>(roots.size());
  summary["selected_roots"] = static_cast<int64_t>(selected_indices.size());
  summary["selected_changed_roots"] = static_cast<int64_t>(std::count_if(
      selected_indices.begin(), selected_indices.end(),
      [&](size_t i) { return cells[i].raw_action != cells[i].selected_action; }));
  summary["rollout_count_per_action"] = static_cast<int64_t>(absl::GetFlag(FLAGS_rollout_count));
  summary["rollout_records"] = static_cast<int64_t>(completed_rollouts);
  summary["selected_minus_raw_outcome"] = Stats(outcome_delta);
  summary["changed_outcome_positive"] = static_cast<int64_t>(positive);
  summary["changed_outcome_negative"] = static_cast<int64_t>(negative);
  summary["q_vs_rollout_action_cells"] = Correlation(q_cells, outcome_cells);
  summary["q_advantage_vs_rollout_delta"] = Correlation(q_delta, outcome_delta);
  summary["q_advantage_sign_agreement"] =
      (positive + negative) == 0 ? 0.0 : static_cast<double>(sign_agree) / (positive + negative);
  std::ofstream summary_file(output_dir / "summary.json");
  summary_file << json::ToString(summary, true) << std::endl;
  return 0;
}
