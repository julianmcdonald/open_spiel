// C4 held-out prediction check: do opponents' public actions predict their
// hidden cards better than the conditional-uniform baseline?
//
// At evaluation points in held-out games, three predictors estimate, for each
// opponent, which hidden cards are in its hand and in its intrigue hand. All
// three see only the observer's information:
//   resampler      today's search worlds (ResampleFromInfostate of the state);
//   uniform_filter the particle filter with a uniform opponent model, i.e. the
//                  same consistency constraints with no belief;
//   belief_filter  the particle filter weighted by the opponent model.
// The private cards of the real game are used only as scoring targets.
// The belief effect is belief_filter minus uniform_filter; the total change
// against today's search is belief_filter minus resampler.
//
// Games come from either --roots_json (saved roots with full action histories,
// as in census_sampled_roots.json) or fresh greedy self-play (--games).

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <random>
#include <shared_mutex>
#include <string>
#include <thread>
#include <vector>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include <torch/torch.h>

#include "open_spiel/examples/dune_batched_evaluator.h"
#include "open_spiel/examples/dune_belief_particles.h"
#include "open_spiel/examples/dune_network.h"
#include "open_spiel/examples/dune_seed_utils.h"
#include "open_spiel/games/dune_imperium/dune_imperium.h"
#include "open_spiel/spiel.h"
#include "open_spiel/utils/json.h"

// Link satisfaction flags
ABSL_FLAG(int, ppo_minibatch_size, 2048, "");
ABSL_FLAG(int, ppo_update_epochs, 4, "");
ABSL_FLAG(double, ppo_clip_epsilon, 0.2, "");
ABSL_FLAG(bool, normalize_advantages, true, "");
ABSL_FLAG(bool, ppo_clip_value_loss, true, "");
ABSL_FLAG(double, entropy_coef, 0.01, "");
ABSL_FLAG(double, value_coef, 0.5, "");
ABSL_FLAG(double, target_kl, 0.0, "");
ABSL_FLAG(bool, train_amp, true, "");
ABSL_FLAG(double, grad_clip_norm, 0.5, "");
ABSL_FLAG(double, logit_cap, 10.0, "");
ABSL_FLAG(uint64_t, shaping_start_env_steps, 0, "");
ABSL_FLAG(uint64_t, shaping_decay_env_steps, 0, "");
ABSL_FLAG(bool, diagnostics_only, false, "");

ABSL_FLAG(std::string, model_checkpoint, "", "Opponent model used for the belief (and for play).");
ABSL_FLAG(int, num_blocks, 12, "Residual blocks in --model_checkpoint (8 or 12).");
ABSL_FLAG(std::string, opponent_checkpoint, "",
          "Optional different checkpoint that the observer's opponents play with, to "
          "test a mismatched opponent model. Empty = opponents play --model_checkpoint.");
ABSL_FLAG(int, opponent_num_blocks, 12, "Residual blocks in --opponent_checkpoint.");
ABSL_FLAG(std::string, roots_json, "", "Saved roots with action histories; empty = self-play.");
ABSL_FLAG(int, max_roots, -1, "Maximum roots to score; -1 = all.");
ABSL_FLAG(int, games, 64, "Self-play games when --roots_json is empty.");
ABSL_FLAG(uint64_t, master_seed, 2026101307, "Seed for self-play games and filters.");
ABSL_FLAG(int, eval_every, 8, "Score every Nth observer agent-turn decision in self-play.");
ABSL_FLAG(int, particles, 64, "Particles per filter.");
ABSL_FLAG(int, resampler_samples, 64, "Worlds drawn from the uniform resampler.");
ABSL_FLAG(double, temperature, 1.0, "Belief exponent on the opponent model probability.");
ABSL_FLAG(double, epsilon, 0.05, "Uniform mix into the opponent model probability.");
ABSL_FLAG(bool, repair, false,
          "Both filters repair particles an opponent's action would kill by rebinding "
          "the shown cards into that opponent's hand (approximate; default off).");
ABSL_FLAG(int, threads, 8, "Concurrent games or roots.");
ABSL_FLAG(int, filter_tasks, 4, "Concurrent particle-update tasks per filter.");
ABSL_FLAG(int, num_evaluators, 2, "Batched evaluator coordinators per model.");
ABSL_FLAG(int, target_batch_size, 128, "Evaluator target batch size.");
ABSL_FLAG(bool, amp, true, "Use AMP for network evaluation.");
ABSL_FLAG(int, bootstrap, 2000, "Game-clustered bootstrap resamples for the summary.");
ABSL_FLAG(std::string, output_jsonl, "", "Per-point rows (required).");
ABSL_FLAG(std::string, output_summary, "", "Summary JSON (required).");

namespace {
using namespace open_spiel;
using namespace open_spiel::dune_imperium;
using open_spiel::dune_belief::BeliefOptions;
using open_spiel::dune_belief::BeliefParticleFilter;

constexpr uint64_t kStreamBeliefCheck = 0x42454C43;  // "BELC"
const std::vector<std::string> kPredictors = {"resampler", "uniform_filter", "belief_filter"};
const std::vector<std::string> kCategories = {"hand", "intrigue"};

struct ModelPool {
  std::vector<std::shared_ptr<SharedDunePolicyValueNetImpl>> models;
  std::vector<std::unique_ptr<std::shared_mutex>> mutexes;
  std::vector<std::shared_ptr<BatchedEvaluator>> coordinators;
  std::vector<std::shared_ptr<BatchedNNEvaluator>> evaluators;
};

ModelPool LoadPool(const std::string& path, int num_blocks, int count, int batch,
                   bool amp, torch::Device device) {
  ModelPool pool;
  for (int e = 0; e < count; ++e) {
    pool.models.push_back(std::make_shared<SharedDunePolicyValueNetImpl>(
        kFullPublicInformationStateSize, 2048, 2391, num_blocks,
        /*nonlinear_value_head=*/false, /*with_aux_heads=*/false,
        /*head_init_seed=*/0, /*with_semantic_scorer=*/true));
    pool.models.back()->to(device);
    torch::load(pool.models.back(), path, device);
    pool.models.back()->eval();
    pool.mutexes.push_back(std::make_unique<std::shared_mutex>());
    pool.coordinators.push_back(std::make_shared<BatchedEvaluator>(
        pool.models.back(), batch, /*timeout_ms=*/1, device, pool.mutexes.back().get(), 10.0f,
        /*device_synchronize=*/false, /*high_priority_stream=*/true,
        /*emit_batch_membership=*/false, /*rollout_amp=*/amp, /*allow_tf32=*/amp));
    pool.evaluators.push_back(std::make_shared<BatchedNNEvaluator>(pool.coordinators.back(), 10.0f));
  }
  return pool;
}

double PriorOf(BatchedNNEvaluator* evaluator, const State& state, Action action) {
  for (const auto& ap : evaluator->Prior(state)) {
    if (ap.first == action) return ap.second;
  }
  return 0.0;
}

Action Greedy(BatchedNNEvaluator* evaluator, const State& state) {
  const auto legal = state.LegalActions();
  if (legal.size() == 1) return legal.front();
  const auto prior = evaluator->Prior(state);
  Action best = legal.front();
  double best_p = -1.0;
  for (Action a : legal) {
    for (const auto& ap : prior) {
      if (ap.first == a && ap.second > best_p) { best_p = ap.second; best = a; }
    }
  }
  return best;
}

// Expected count of each card in each opponent's hand and intrigue hand.
struct Prediction {
  std::array<std::map<int, double>, kNumPlayers> hand;
  std::array<std::map<int, double>, kNumPlayers> intrigue;
};

void AddWorld(const State& world, Player observer, double weight, Prediction* out) {
  const auto& d = static_cast<const DuneImperiumState&>(world);
  for (Player q = 0; q < kNumPlayers; ++q) {
    if (q == observer) continue;
    for (int c : d.GetPlayerHandForTesting(q)) out->hand[q][c] += weight;
    for (int c : d.GetPlayerIntrigues(q)) out->intrigue[q][c] += weight;
  }
}

Prediction FromFilter(const BeliefParticleFilter& filter) {
  Prediction p;
  const auto w = filter.Weights();
  for (int i = 0; i < filter.size(); ++i) {
    if (w[i] > 0.0) AddWorld(filter.particle(i), filter.observer(), w[i], &p);
  }
  return p;
}

Prediction FromResampler(const State& truth, Player observer, int samples, uint64_t seed) {
  Prediction p;
  const auto& d = static_cast<const DuneImperiumState&>(truth);
  for (int k = 0; k < samples; ++k) {
    std::mt19937_64 rng(dune_seed::DeriveSeed(seed, k));
    auto draw = [&rng]() { return std::generate_canonical<double, 53>(rng); };
    auto world = d.ResampleFromInfostate(observer, draw);
    AddWorld(*world, observer, 1.0 / samples, &p);
  }
  return p;
}

struct Score {
  double brier = 0.0;
  double log_loss = 0.0;
  int cards = 0;
};

// Scores one category over every opponent. For each hidden card id c with n_c
// copies in the hidden universe, the target is (copies in the zone) / n_c and
// the prediction is (expected copies in the zone) / n_c.
Score ScoreCategory(const std::array<std::map<int, double>, kNumPlayers>& predicted,
                    const std::array<std::map<int, int>, kNumPlayers>& truth_zone,
                    const std::array<std::map<int, int>, kNumPlayers>& universe,
                    Player observer) {
  Score s;
  for (Player q = 0; q < kNumPlayers; ++q) {
    if (q == observer) continue;
    for (const auto& [card, copies] : universe[q]) {
      auto t = truth_zone[q].find(card);
      const double y = (t == truth_zone[q].end() ? 0.0 : t->second) / copies;
      auto pr = predicted[q].find(card);
      const double p = (pr == predicted[q].end() ? 0.0 : pr->second) / copies;
      const double pc = std::min(1.0 - 1e-3, std::max(1e-3, p));
      s.brier += (p - y) * (p - y);
      s.log_loss -= y * std::log(pc) + (1.0 - y) * std::log(1.0 - pc);
      ++s.cards;
    }
  }
  if (s.cards > 0) { s.brier /= s.cards; s.log_loss /= s.cards; }
  return s;
}

struct PointRow {
  int game_id;
  int observer;
  int round;
  std::array<std::array<Score, 2>, 3> scores;  // [predictor][category]
  double belief_ess;
  int64_t belief_rebuilds;
  int64_t uniform_rebuilds;
  int64_t belief_repaired;
  int64_t uniform_repaired;
};

PointRow ScorePoint(int game_id, const State& truth, Player observer,
                    const BeliefParticleFilter& uniform_filter,
                    const BeliefParticleFilter& belief_filter, int resampler_samples,
                    uint64_t seed) {
  const auto& d = static_cast<const DuneImperiumState&>(truth);
  std::array<std::map<int, int>, kNumPlayers> hand_truth, hand_universe;
  std::array<std::map<int, int>, kNumPlayers> intrigue_truth, intrigue_universe;
  std::map<int, int> hidden_intrigues;
  for (int c : d.GetIntrigueDrawDeckForTesting()) ++hidden_intrigues[c];
  for (Player q = 0; q < kNumPlayers; ++q) {
    if (q == observer) continue;
    for (int c : d.GetPlayerHandForTesting(q)) { ++hand_truth[q][c]; ++hand_universe[q][c]; }
    for (int c : d.GetPlayerDrawDeckForTesting(q)) ++hand_universe[q][c];
    for (int c : d.GetPlayerIntrigues(q)) { ++intrigue_truth[q][c]; ++hidden_intrigues[c]; }
  }
  for (Player q = 0; q < kNumPlayers; ++q) {
    if (q != observer) intrigue_universe[q] = hidden_intrigues;
  }
  const std::array<Prediction, 3> predictions = {
      FromResampler(truth, observer, resampler_samples, seed),
      FromFilter(uniform_filter), FromFilter(belief_filter)};
  PointRow row;
  row.game_id = game_id;
  row.observer = observer;
  row.round = d.GetCurrentRound();
  for (int k = 0; k < 3; ++k) {
    row.scores[k][0] = ScoreCategory(predictions[k].hand, hand_truth, hand_universe, observer);
    row.scores[k][1] = ScoreCategory(predictions[k].intrigue, intrigue_truth, intrigue_universe, observer);
  }
  row.belief_ess = belief_filter.EffectiveSampleSize();
  row.belief_rebuilds = belief_filter.stats().rebuilds;
  row.uniform_rebuilds = uniform_filter.stats().rebuilds;
  row.belief_repaired = belief_filter.stats().repaired;
  row.uniform_repaired = uniform_filter.stats().repaired;
  return row;
}

json::Object RowJson(const PointRow& r) {
  json::Object o{{"game_id", r.game_id}, {"observer", r.observer}, {"round", r.round},
                 {"belief_ess", r.belief_ess},
                 {"belief_rebuilds", r.belief_rebuilds},
                 {"uniform_rebuilds", r.uniform_rebuilds},
                 {"belief_repaired", r.belief_repaired},
                 {"uniform_repaired", r.uniform_repaired}};
  for (int k = 0; k < 3; ++k) {
    for (int c = 0; c < 2; ++c) {
      const std::string key = kPredictors[k] + "_" + kCategories[c];
      o[key + "_brier"] = r.scores[k][c].brier;
      o[key + "_log_loss"] = r.scores[k][c].log_loss;
      o[key + "_cards"] = r.scores[k][c].cards;
    }
  }
  return o;
}

struct RootCase {
  int root_id;
  int game_id;
  int seat;
  std::vector<Action> history;
};

std::vector<RootCase> ReadRoots(const std::string& path, int max_roots) {
  std::ifstream input(path);
  if (!input) SpielFatalError("Cannot read " + path);
  std::string content((std::istreambuf_iterator<char>(input)), {});
  auto parsed = json::FromString(content);
  if (!parsed || !parsed->IsObject()) SpielFatalError("Invalid roots JSON: " + path);
  const auto& roots = parsed->GetObject().at("roots").GetArray();
  std::vector<RootCase> out;
  for (const auto& item : roots) {
    const auto& row = item.GetObject();
    RootCase root{static_cast<int>(row.at("root_id").GetInt()),
                  static_cast<int>(row.at("game_id").GetInt()),
                  static_cast<int>(row.at("search_seat").GetInt()), {}};
    for (const auto& a : row.at("action_history").GetArray()) root.history.push_back(a.GetInt());
    out.push_back(std::move(root));
    if (max_roots > 0 && static_cast<int>(out.size()) == max_roots) break;
  }
  return out;
}

struct Filters {
  std::unique_ptr<BeliefParticleFilter> uniform;
  std::unique_ptr<BeliefParticleFilter> belief;
};

Filters MakeFilters(const State& initial, Player observer, uint64_t seed,
                    BatchedNNEvaluator* model) {
  BeliefOptions options;
  options.num_particles = absl::GetFlag(FLAGS_particles);
  options.temperature = absl::GetFlag(FLAGS_temperature);
  options.epsilon = absl::GetFlag(FLAGS_epsilon);
  options.parallel_tasks = absl::GetFlag(FLAGS_filter_tasks);
  options.repair_opponent_hands = absl::GetFlag(FLAGS_repair);
  Filters f;
  // Both filters share a seed so hidden draws match until weighting differs.
  f.uniform = std::make_unique<BeliefParticleFilter>(
      initial, observer, options, seed, open_spiel::dune_belief::UniformOpponentModel());
  f.belief = std::make_unique<BeliefParticleFilter>(
      initial, observer, options, seed,
      [model](const State& s, Action a) { return PriorOf(model, s, a); });
  return f;
}

// Bootstrap mean difference of per-game means, clustered by game.
std::array<double, 3> ClusteredDiff(const std::vector<PointRow>& rows, int a, int b,
                                    int category, bool brier, int resamples, uint64_t seed) {
  std::map<int, std::pair<double, int>> per_game;
  for (const auto& r : rows) {
    const auto& sa = r.scores[a][category];
    const auto& sb = r.scores[b][category];
    if (sa.cards == 0) continue;
    const double d = brier ? sa.brier - sb.brier : sa.log_loss - sb.log_loss;
    per_game[r.game_id].first += d;
    per_game[r.game_id].second += 1;
  }
  std::vector<double> means;
  for (const auto& [g, v] : per_game) means.push_back(v.first / v.second);
  if (means.empty()) return {0.0, 0.0, 0.0};
  double mean = 0.0;
  for (double m : means) mean += m;
  mean /= means.size();
  std::mt19937_64 rng(seed);
  std::uniform_int_distribution<size_t> pick(0, means.size() - 1);
  std::vector<double> boots;
  for (int k = 0; k < resamples; ++k) {
    double s = 0.0;
    for (size_t i = 0; i < means.size(); ++i) s += means[pick(rng)];
    boots.push_back(s / means.size());
  }
  std::sort(boots.begin(), boots.end());
  return {mean, boots[static_cast<size_t>(0.025 * (boots.size() - 1))],
          boots[static_cast<size_t>(0.975 * (boots.size() - 1))]};
}

}  // namespace

int main(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);
  const std::string model_path = absl::GetFlag(FLAGS_model_checkpoint);
  const std::string jsonl_path = absl::GetFlag(FLAGS_output_jsonl);
  const std::string summary_path = absl::GetFlag(FLAGS_output_summary);
  if (model_path.empty() || jsonl_path.empty() || summary_path.empty()) {
    std::cerr << "[ERROR] --model_checkpoint, --output_jsonl and --output_summary are required\n";
    return 1;
  }
  torch::Device device = torch::cuda::is_available() ? torch::Device(torch::kCUDA, 0)
                                                     : torch::Device(torch::kCPU);
  const int evals = std::max(1, absl::GetFlag(FLAGS_num_evaluators));
  const int batch = absl::GetFlag(FLAGS_target_batch_size);
  const bool amp = absl::GetFlag(FLAGS_amp);
  ModelPool model = LoadPool(model_path, absl::GetFlag(FLAGS_num_blocks), evals, batch, amp, device);
  ModelPool opponent_pool;
  const std::string opponent_path = absl::GetFlag(FLAGS_opponent_checkpoint);
  if (!opponent_path.empty()) {
    opponent_pool = LoadPool(opponent_path, absl::GetFlag(FLAGS_opponent_num_blocks),
                             evals, batch, amp, device);
  }
  auto game = LoadGame("dune_imperium");
  const uint64_t master_seed = absl::GetFlag(FLAGS_master_seed);
  const int resampler_samples = absl::GetFlag(FLAGS_resampler_samples);
  const int eval_every = std::max(1, absl::GetFlag(FLAGS_eval_every));
  const std::string roots_path = absl::GetFlag(FLAGS_roots_json);
  const std::vector<RootCase> roots =
      roots_path.empty() ? std::vector<RootCase>{} : ReadRoots(roots_path, absl::GetFlag(FLAGS_max_roots));
  const int jobs = roots_path.empty() ? absl::GetFlag(FLAGS_games) : static_cast<int>(roots.size());

  std::ofstream jsonl(jsonl_path);
  if (!jsonl) SpielFatalError("Cannot write " + jsonl_path);
  std::vector<PointRow> rows;
  std::mutex mu;
  std::atomic<int> next{0}, done{0}, skipped_roots{0};

  auto worker = [&](int thread_id) {
    BatchedNNEvaluator* model_eval = model.evaluators[thread_id % evals].get();
    BatchedNNEvaluator* opponent_eval = opponent_pool.evaluators.empty()
        ? model_eval : opponent_pool.evaluators[thread_id % evals].get();
    while (true) {
      const int job = next.fetch_add(1);
      if (job >= jobs) break;
      std::vector<PointRow> local;
      auto state = game->NewInitialState();
      if (!roots_path.empty()) {
        const RootCase& root = roots[job];
        const uint64_t seed = dune_seed::DeriveSeed(master_seed, kStreamBeliefCheck, root.root_id);
        Filters f = MakeFilters(*state, root.seat, seed, model_eval);
        bool legal = true;
        for (Action a : root.history) {
          const auto legal_now = state->IsChanceNode() ? std::vector<Action>{} : state->LegalActions();
          bool ok = false;
          if (state->IsChanceNode()) {
            for (const auto& [o, p] : state->ChanceOutcomes()) ok |= (o == a && p > 0.0);
          } else {
            ok = std::find(legal_now.begin(), legal_now.end(), a) != legal_now.end();
          }
          if (!ok) { legal = false; break; }
          auto before = state->Clone();
          state->ApplyAction(a);
          f.uniform->Observe(*before, a, *state);
          f.belief->Observe(*before, a, *state);
        }
        if (!legal || state->IsTerminal()) {
          skipped_roots.fetch_add(1);
        } else {
          local.push_back(ScorePoint(root.game_id, *state, root.seat, *f.uniform, *f.belief,
                                     resampler_samples, seed));
        }
      } else {
        const Player observer = job % kNumPlayers;
        const uint64_t seed = dune_seed::DeriveSeed(master_seed, kStreamBeliefCheck, job);
        std::mt19937_64 chance_rng(dune_seed::DeriveSeed(seed, 1));
        Filters f = MakeFilters(*state, observer, seed, model_eval);
        int observer_decisions = 0;
        while (!state->IsTerminal()) {
          Action a;
          if (state->IsChanceNode()) {
            a = SampleAction(state->ChanceOutcomes(),
                             std::generate_canonical<double, 53>(chance_rng)).first;
          } else {
            const Player cur = state->CurrentPlayer();
            const auto& d = static_cast<const DuneImperiumState&>(*state);
            if (cur == observer && d.phase() == GamePhase::kAgentTurns &&
                state->LegalActions().size() > 1 && observer_decisions++ % eval_every == 0) {
              local.push_back(ScorePoint(job, *state, observer, *f.uniform, *f.belief,
                                         resampler_samples,
                                         dune_seed::DeriveSeed(seed, 2, observer_decisions)));
            }
            a = Greedy(cur == observer ? model_eval : opponent_eval, *state);
          }
          auto before = state->Clone();
          state->ApplyAction(a);
          f.uniform->Observe(*before, a, *state);
          f.belief->Observe(*before, a, *state);
        }
      }
      std::lock_guard<std::mutex> lock(mu);
      for (const auto& r : local) {
        jsonl << json::ToString(RowJson(r)) << '\n';
        rows.push_back(r);
      }
      jsonl.flush();
      std::cout << "[JOB] " << ++done << '/' << jobs << " points=" << rows.size() << '\n'
                << std::flush;
    }
  };
  std::vector<std::thread> workers;
  for (int t = 0; t < absl::GetFlag(FLAGS_threads); ++t) workers.emplace_back(worker, t);
  for (auto& w : workers) w.join();

  json::Object summary{{"model_checkpoint", model_path},
                       {"opponent_checkpoint", opponent_path},
                       {"roots_json", roots_path},
                       {"jobs", jobs},
                       {"skipped_roots", skipped_roots.load()},
                       {"points", static_cast<int64_t>(rows.size())},
                       {"master_seed", std::to_string(master_seed)},
                       {"particles", absl::GetFlag(FLAGS_particles)},
                       {"resampler_samples", resampler_samples},
                       {"temperature", absl::GetFlag(FLAGS_temperature)},
                       {"epsilon", absl::GetFlag(FLAGS_epsilon)},
                       {"repair", absl::GetFlag(FLAGS_repair)}};
  const int resamples = absl::GetFlag(FLAGS_bootstrap);
  for (int c = 0; c < 2; ++c) {
    for (int k = 0; k < 3; ++k) {
      double brier = 0.0, log_loss = 0.0;
      int n = 0;
      for (const auto& r : rows) {
        if (r.scores[k][c].cards == 0) continue;
        brier += r.scores[k][c].brier;
        log_loss += r.scores[k][c].log_loss;
        ++n;
      }
      const std::string key = kPredictors[k] + "_" + kCategories[c];
      summary[key + "_mean_brier"] = n ? brier / n : 0.0;
      summary[key + "_mean_log_loss"] = n ? log_loss / n : 0.0;
      summary[key + "_points"] = n;
    }
    const std::vector<std::array<int, 2>> contrasts = {{2, 1}, {2, 0}, {1, 0}};
    for (const auto& [a, b] : contrasts) {
      for (bool brier : {true, false}) {
        const auto d = ClusteredDiff(rows, a, b, c, brier, resamples,
                                     dune_seed::DeriveSeed(master_seed, a, b, c, brier));
        const std::string key = kPredictors[a] + "_minus_" + kPredictors[b] + "_" +
                                kCategories[c] + (brier ? "_brier" : "_log_loss");
        summary[key] = json::Array{d[0], d[1], d[2]};
      }
    }
  }
  std::ofstream out(summary_path);
  out << json::ToString(json::Value(summary), true) << '\n';
  std::cout << json::ToString(json::Value(summary), true) << '\n';
  return 0;
}
