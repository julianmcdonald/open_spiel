#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <memory>
#include <chrono>
#include <mutex>
#include <shared_mutex>
#include <atomic>
#include <thread>
#include <future>
#include <numeric>
#include <cmath>
#include <iomanip>
#include <algorithm>
#include <random>
#include <queue>
#include <condition_variable>
#include <filesystem>
#include <unordered_map>

#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"
#include "open_spiel/utils/json.h"
#include "open_spiel/games/dune_imperium/dune_imperium.h"
#include "open_spiel/games/dune_imperium/dune_imperium_common.h"
#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/strings/str_format.h"

#include <torch/torch.h>

#include "dune_network.h"
#include "dune_evaluator.h"
#include "dune_batched_evaluator.h"
#include "dune_semantic_action_scorer.h"
#include "dune_seed_utils.h"
#include "dune_sha256.h"

ABSL_FLAG(std::string, model_checkpoint,
          "/run/media/warcr/Storage/dune_drl_runtime/round7/u27900_continuation_20260930/checkpoints/ppo_model_update_28700.pt",
          "Path to U28700 anchor model checkpoint.");
ABSL_FLAG(std::string, output_dir,
          "/home/warcr/dune_drl_runtime/round7/tleilaxu_card_check_20261003",
          "Output directory for study JSON artifacts.");
ABSL_FLAG(int, num_roots, 200, "Number of roots in discovery and confirmation sets.");
ABSL_FLAG(int, num_rollouts, 256, "Number of paired rollouts per candidate.");
ABSL_FLAG(int, threads, 64, "Number of rollout worker threads.");
ABSL_FLAG(int, num_evaluators, 2, "Number of batched evaluator coordinators.");
ABSL_FLAG(int, target_batch_size, 64, "GPU evaluator batch size.");
ABSL_FLAG(uint64_t, master_seed, 20261003ULL, "Master seed.");
ABSL_FLAG(uint64_t, discovery_start_seed, 0ULL, "Episode offset for discovery games.");
ABSL_FLAG(uint64_t, confirmation_start_seed, 100000ULL, "Episode offset for confirmation games.");
ABSL_FLAG(bool, confirmation_only, false,
          "Run only confirmation set using existing roots_confirmation.json.");

namespace {

using namespace open_spiel;
using namespace open_spiel::dune_imperium;

constexpr int kScientificBreakthroughTleilaxuId = 11;
constexpr int kScientificBreakthroughCardId = 118;
constexpr int kStitchedHorrorTleilaxuId = 13;
constexpr int kStitchedHorrorCardId = 120;
constexpr int kSpecimenCost = 3;

constexpr Action kActionReclaimedForces = 90; // kActionTleilaxuAcquire0
constexpr Action kActionAcquireSlot0 = 91;    // kActionTleilaxuAcquire0 + 1
constexpr Action kActionAcquireSlot1 = 92;    // kActionTleilaxuAcquire0 + 2

struct RootRecord {
  int root_id = 0;
  uint64_t episode_id = 0;
  int search_seat = 0;
  int round = 0;
  std::string round_band; // "R1-3", "R4-6", "R7+"
  std::vector<Action> legal_actions;
  std::vector<int64_t> action_history;
  std::vector<float> obs;

  // Legality
  bool sb_legal = false;
  Action sb_action = kInvalidAction;
  bool sh_legal = false;
  Action sh_action = kInvalidAction;
  bool rf_legal = false;
  Action rf_action = kInvalidAction;

  // Policy greedy choice and probabilities
  Action greedy_choice = kInvalidAction;
  double greedy_prob = 0.0;
  double sb_prob = 0.0;
  double sh_prob = 0.0;
  double rf_prob = 0.0;

  // Logit diagnostics (A2)
  double z_sb = 0.0;
  double capped_sb = 0.0;
  double sech2_sb = 0.0;
  double z_sh = 0.0;
  double capped_sh = 0.0;
  double sech2_sh = 0.0;
};

struct WorldRolloutRecord {
  int world_idx = 0;
  Action action = kInvalidAction;
  double return_val = 0.0;
  bool is_terminal = false;
  std::string exit_reason; // "terminal", "step_cap_reached", "empty_legals"
  int steps = 0;
  int final_vp = 0;
  bool resample_fallback = false;
  bool sb_played = false;
  bool sh_played = false;
  bool sb_trashed_for_vp = false;
  std::unordered_map<std::string, int> vp_by_source;
};

struct AuditCounters {
  std::atomic<size_t> total_simulations{0};
  std::atomic<size_t> terminal_exits{0};
  std::atomic<size_t> step_cap_exits{0};
  std::atomic<size_t> empty_legals_exits{0};
  std::atomic<size_t> total_resample_calls{0};
  std::atomic<size_t> resample_fallbacks{0};

  std::atomic<size_t> sb_rollouts{0};
  std::atomic<size_t> sb_played_rollouts{0};
  std::atomic<size_t> sb_trash_vp_rollouts{0};

  std::atomic<size_t> sh_rollouts{0};
  std::atomic<size_t> sh_played_rollouts{0};

  std::atomic<size_t> rf_rollouts{0};
  std::atomic<size_t> rf_played_rollouts{0};

  std::unordered_map<std::string, double> sum_vp_greedy;
  std::unordered_map<std::string, double> sum_vp_sb;
  std::unordered_map<std::string, double> sum_vp_sh;
  std::unordered_map<std::string, double> sum_vp_rf;
  double sum_final_vp_greedy = 0.0;
  double sum_final_vp_sb = 0.0;
  double sum_final_vp_sh = 0.0;
  double sum_final_vp_rf = 0.0;
  size_t count_eval_sb = 0;
  size_t count_eval_sh = 0;
  size_t count_eval_rf = 0;
  size_t count_eval_greedy = 0;
};

struct PerRootRolloutResult {
  int root_id = 0;
  int round = 0;
  std::string round_band;
  Action greedy_choice = kInvalidAction;
  double q_greedy = 0.0;

  bool sb_legal = false;
  Action sb_action = kInvalidAction;
  double q_sb = 0.0;
  double sb_gap_vs_greedy = 0.0;
  double sb_se_vs_greedy = 0.0;
  double sb_gap_vs_rf = 0.0;
  double sb_se_vs_rf = 0.0;

  bool sh_legal = false;
  Action sh_action = kInvalidAction;
  double q_sh = 0.0;
  double sh_gap_vs_greedy = 0.0;
  double sh_se_vs_greedy = 0.0;
  double sh_gap_vs_rf = 0.0;
  double sh_se_vs_rf = 0.0;

  bool rf_legal = false;
  Action rf_action = kInvalidAction;
  double q_rf = 0.0;

  std::unordered_map<Action, std::vector<double>> raw_rollout_returns;
  std::unordered_map<Action, std::vector<WorldRolloutRecord>> detailed_world_rollouts;
};

struct SummaryStats {
  double mean = 0.0;
  double median = 0.0;
  double p10 = 0.0;
  double p90 = 0.0;
};

SummaryStats ComputePercentiles(std::vector<double> vals) {
  SummaryStats s;
  if (vals.empty()) return s;
  std::sort(vals.begin(), vals.end());
  double sum = 0.0;
  for (double v : vals) sum += v;
  s.mean = sum / vals.size();
  auto get_p = [&](double p) {
    if (vals.size() == 1) return vals[0];
    double idx = p * (vals.size() - 1);
    size_t lo = static_cast<size_t>(std::floor(idx));
    size_t hi = static_cast<size_t>(std::ceil(idx));
    double frac = idx - lo;
    return (1.0 - frac) * vals[lo] + frac * vals[hi];
  };
  s.p10 = get_p(0.10);
  s.median = get_p(0.50);
  s.p90 = get_p(0.90);
  return s;
}

std::pair<double, double> BootstrapMeanCI(
    const std::vector<double>& data,
    int num_resamples = 10000,
    double ci = 0.95,
    uint64_t seed = 42) {
  if (data.empty()) return {0.0, 0.0};
  if (data.size() == 1) return {data[0], data[0]};

  std::mt19937_64 rng(seed);
  std::uniform_int_distribution<size_t> dist(0, data.size() - 1);
  std::vector<double> boot_means(num_resamples);

  for (int b = 0; b < num_resamples; ++b) {
    double sum = 0.0;
    for (size_t i = 0; i < data.size(); ++i) {
      sum += data[dist(rng)];
    }
    boot_means[b] = sum / data.size();
  }
  std::sort(boot_means.begin(), boot_means.end());
  double alpha = (1.0 - ci) / 2.0;
  size_t idx_lower = static_cast<size_t>(std::floor(alpha * num_resamples));
  size_t idx_upper = static_cast<size_t>(std::ceil((1.0 - alpha) * num_resamples)) - 1;
  return {boot_means[idx_lower], boot_means[idx_upper]};
}

class SimpleThreadPool {
 public:
  explicit SimpleThreadPool(size_t threads) : stop_(false) {
    for (size_t i = 0; i < threads; ++i) {
      workers_.emplace_back([this]() {
        while (true) {
          std::function<void()> task;
          {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            cv_.wait(lock, [this]() { return stop_ || !tasks_.empty(); });
            if (stop_ && tasks_.empty()) return;
            task = std::move(tasks_.front());
            tasks_.pop();
          }
          task();
        }
      });
    }
  }

  void Enqueue(std::function<void()> task) {
    {
      std::unique_lock<std::mutex> lock(queue_mutex_);
      tasks_.push(std::move(task));
    }
    cv_.notify_one();
  }

  void WaitAll(size_t expected_total_tasks, const std::atomic<size_t>& completed_counter) {
    while (completed_counter.load(std::memory_order_relaxed) < expected_total_tasks) {
      std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
  }

  ~SimpleThreadPool() {
    {
      std::unique_lock<std::mutex> lock(queue_mutex_);
      stop_ = true;
    }
    cv_.notify_all();
    for (std::thread& worker : workers_) {
      if (worker.joinable()) worker.join();
    }
  }

 private:
  std::vector<std::thread> workers_;
  std::queue<std::function<void()>> tasks_;
  std::mutex queue_mutex_;
  std::condition_variable cv_;
  bool stop_;
};

struct RolloutOutcome {
  double ret = 0.0;
  bool is_terminal = false;
  std::string exit_reason; // "terminal", "step_cap_reached", "empty_legals"
  int steps = 0;
  int final_vp = 0;
  bool sb_played = false;
  bool sh_played = false;
  bool sb_trashed_for_vp = false;
  std::unordered_map<std::string, int> vp_by_source;
};

inline RolloutOutcome SimulateGreedyRollout(
    open_spiel::State* sim_state,
    open_spiel::Player search_player,
    open_spiel::BatchedNNEvaluator* evaluator,
    uint64_t seed) {
  RolloutOutcome outcome;
  std::mt19937 rng(seed);
  int steps = 0;
  const int max_steps = 2500;
  const auto* dune_sim = dynamic_cast<const open_spiel::dune_imperium::DuneImperiumState*>(sim_state);

  while (!sim_state->IsTerminal() && steps < max_steps) {
    steps++;
    if (sim_state->IsChanceNode()) {
      auto chance_outcomes = sim_state->ChanceOutcomes();
      double u = std::generate_canonical<double, 53>(rng);
      open_spiel::Action ca = open_spiel::SampleAction(chance_outcomes, u).first;
      sim_state->ApplyAction(ca);
      continue;
    }
    std::vector<open_spiel::Action> legals = sim_state->LegalActions();
    if (legals.empty()) {
      outcome.exit_reason = "empty_legals";
      break;
    }
    if (legals.size() == 1) {
      sim_state->ApplyAction(legals.front());
    } else {
      open_spiel::ActionsAndProbs prior = evaluator->Prior(*sim_state);
      open_spiel::Action a = legals.front();
      double best_p = -1.0;
      for (const auto& ap : prior) {
        if (ap.second > best_p) {
          best_p = ap.second;
          a = ap.first;
        }
      }
      sim_state->ApplyAction(a);
    }
    if (dune_sim) {
      for (int c : dune_sim->GetPlayedAgentCardsForTesting(search_player)) {
        if (c == kScientificBreakthroughCardId) outcome.sb_played = true;
        if (c == kStitchedHorrorCardId) outcome.sh_played = true;
      }
      for (int c : dune_sim->GetRevealedCardsForTesting(search_player)) {
        if (c == kScientificBreakthroughCardId) outcome.sb_played = true;
        if (c == kStitchedHorrorCardId) outcome.sh_played = true;
      }
    }
  }

  outcome.steps = steps;
  if (sim_state->IsTerminal()) {
    outcome.is_terminal = true;
    outcome.exit_reason = "terminal";
  } else if (outcome.exit_reason.empty()) {
    outcome.exit_reason = "step_cap_reached";
  }

  std::vector<double> ret = sim_state->Returns();
  if (static_cast<size_t>(search_player) < ret.size()) {
    outcome.ret = ret[search_player];
  }

  if (dune_sim) {
    outcome.final_vp = dune_sim->GetPlayerVp(search_player);
    const auto& events = dune_sim->GetVpEvents(search_player);
    for (const auto& ev : events) {
      std::string s_name = open_spiel::dune_imperium::VpSourceName(ev.source);
      outcome.vp_by_source[s_name] += ev.delta;
      if (ev.source == open_spiel::dune_imperium::VpSource::kScientificBreakthrough) {
        outcome.sb_trashed_for_vp = true;
        outcome.sb_played = true;
      }
    }
  }

  return outcome;
}

// Evaluate raw logits & logit diagnostics at a root
void PopulateRootLogits(
    open_spiel::SharedDunePolicyValueNetImpl& model,
    torch::Device device,
    RootRecord& rec,
    const dune_semantic::CandidateActionData& cand_data) {
  torch::NoGradGuard no_grad;
  torch::Tensor x = torch::from_blob(rec.obs.data(), {1, static_cast<int64_t>(rec.obs.size())}, torch::kFloat32).to(device);
  auto outputs = model.forward(x);

  std::vector<const dune_semantic::CandidateActionData*> batch_cands = {&cand_data};
  dune_semantic::ApplySemanticScorerBatch(
      model.semantic_scorer_, outputs.trunk, batch_cands, outputs.logits, device);

  torch::Tensor l_cpu = outputs.logits.squeeze(0).to(torch::kCPU);
  const float* l_ptr = l_cpu.data_ptr<float>();

  double legal_sum = 0.0;
  for (Action a : rec.legal_actions) {
    legal_sum += l_ptr[a];
  }
  double legal_mean = legal_sum / rec.legal_actions.size();

  double max_c = -1e9;
  std::vector<double> capped(rec.legal_actions.size());
  for (size_t i = 0; i < rec.legal_actions.size(); ++i) {
    Action a = rec.legal_actions[i];
    double z = static_cast<double>(l_ptr[a]) - legal_mean;
    double c = 10.0 * std::tanh(z / 10.0);
    capped[i] = c;
    if (c > max_c) max_c = c;
  }
  double sum_exp = 0.0;
  for (double c : capped) sum_exp += std::exp(c - max_c);

  std::vector<double> probs(rec.legal_actions.size());
  for (size_t i = 0; i < rec.legal_actions.size(); ++i) {
    probs[i] = std::exp(capped[i] - max_c) / sum_exp;
  }

  // Find greedy choice
  Action greedy_a = rec.legal_actions[0];
  double greedy_p = probs[0];
  for (size_t i = 1; i < rec.legal_actions.size(); ++i) {
    if (probs[i] > greedy_p) {
      greedy_p = probs[i];
      greedy_a = rec.legal_actions[i];
    }
  }
  rec.greedy_choice = greedy_a;
  rec.greedy_prob = greedy_p;

  if (rec.sb_legal) {
    double z = static_cast<double>(l_ptr[rec.sb_action]) - legal_mean;
    rec.z_sb = z;
    rec.capped_sb = 10.0 * std::tanh(z / 10.0);
    rec.sech2_sb = 1.0 - std::pow(std::tanh(z / 10.0), 2);
    for (size_t i = 0; i < rec.legal_actions.size(); ++i) {
      if (rec.legal_actions[i] == rec.sb_action) {
        rec.sb_prob = probs[i];
        break;
      }
    }
  }

  if (rec.sh_legal) {
    double z = static_cast<double>(l_ptr[rec.sh_action]) - legal_mean;
    rec.z_sh = z;
    rec.capped_sh = 10.0 * std::tanh(z / 10.0);
    rec.sech2_sh = 1.0 - std::pow(std::tanh(z / 10.0), 2);
    for (size_t i = 0; i < rec.legal_actions.size(); ++i) {
      if (rec.legal_actions[i] == rec.sh_action) {
        rec.sh_prob = probs[i];
        break;
      }
    }
  }

  if (rec.rf_legal) {
    for (size_t i = 0; i < rec.legal_actions.size(); ++i) {
      if (rec.legal_actions[i] == rec.rf_action) {
        rec.rf_prob = probs[i];
        break;
      }
    }
  }
}

std::vector<RootRecord> CollectRootsSet(
    const std::shared_ptr<const open_spiel::Game>& game,
    const std::shared_ptr<open_spiel::SharedDunePolicyValueNetImpl>& model,
    const std::vector<std::shared_ptr<open_spiel::BatchedNNEvaluator>>& evaluators,
    torch::Device device,
    uint64_t master_seed,
    uint64_t start_episode_id,
    int target_total_roots,
    int num_threads,
    const std::string& set_name) {
  std::cout << "\n======================================================================\n";
  std::cout << "[A1 ROOTS] Collecting " << target_total_roots << " roots for " << set_name << " set ("
            << num_threads << " worker threads)...\n";
  std::cout << "======================================================================\n" << std::flush;

  std::vector<RootRecord> roots;
  roots.reserve(target_total_roots);
  std::mutex roots_mutex;

  std::atomic<uint64_t> global_ep{start_episode_id};
  std::atomic<int> games_played{0};
  int count_r1_3 = 0;
  int count_r4_6 = 0;
  int count_r7_plus = 0;

  auto t0 = std::chrono::steady_clock::now();

  std::vector<std::thread> workers;
  workers.reserve(num_threads);

  for (int th = 0; th < num_threads; ++th) {
    workers.emplace_back([&, th]() {
      auto* evaluator = evaluators[th % evaluators.size()].get();
      const size_t obs_size = open_spiel::dune_imperium::kFullPublicInformationStateSize;
      std::vector<float> local_obs(obs_size, 0.0f);

      while (true) {
        {
          std::lock_guard<std::mutex> lk(roots_mutex);
          if (roots.size() >= static_cast<size_t>(target_total_roots)) {
            break;
          }
        }

        uint64_t episode_id = global_ep.fetch_add(1);
        int g_count = games_played.fetch_add(1) + 1;

        auto chance_rng = dune_seed::MakeRng64(dune_seed::DeriveSeed(
            master_seed, dune_seed::kDomainTrain, episode_id, dune_seed::kStreamChance));

        std::unique_ptr<open_spiel::State> state = game->NewInitialState();
        auto* dune_state = dynamic_cast<open_spiel::dune_imperium::DuneImperiumState*>(state.get());
        SPIEL_CHECK_TRUE(dune_state != nullptr);

        std::vector<int64_t> action_history;
        bool game_root_selected = false;

        int steps = 0;
        while (!state->IsTerminal() && steps < 2500) {
          steps++;
          if (state->IsChanceNode()) {
            auto outcomes = state->ChanceOutcomes();
            Action a = SampleAction(outcomes, chance_rng).first;
            action_history.push_back(a);
            state->ApplyAction(a);
            continue;
          }

          std::vector<Action> legal_actions = state->LegalActions();
          if (legal_actions.empty()) break;

          Player cur_player = state->CurrentPlayer();

          // Check legality of target cards
          const auto& row = dune_state->GetTleilaxuRowForTesting();
          int row0 = (row.size() > 0) ? row[0] : kInvalidCard;
          int row1 = (row.size() > 1) ? row[1] : kInvalidCard;

          bool can_buy_0 = std::find(legal_actions.begin(), legal_actions.end(), kActionAcquireSlot0) != legal_actions.end();
          bool can_buy_1 = std::find(legal_actions.begin(), legal_actions.end(), kActionAcquireSlot1) != legal_actions.end();
          bool can_buy_rf = std::find(legal_actions.begin(), legal_actions.end(), kActionReclaimedForces) != legal_actions.end();

          bool sb_legal = false;
          Action sb_action = kInvalidAction;
          if (can_buy_0 && row0 == kScientificBreakthroughTleilaxuId) {
            sb_legal = true;
            sb_action = kActionAcquireSlot0;
          } else if (can_buy_1 && row1 == kScientificBreakthroughTleilaxuId) {
            sb_legal = true;
            sb_action = kActionAcquireSlot1;
          }

          bool sh_legal = false;
          Action sh_action = kInvalidAction;
          if (can_buy_0 && row0 == kStitchedHorrorTleilaxuId) {
            sh_legal = true;
            sh_action = kActionAcquireSlot0;
          } else if (can_buy_1 && row1 == kStitchedHorrorTleilaxuId) {
            sh_legal = true;
            sh_action = kActionAcquireSlot1;
          }

          bool is_eligible = (sb_legal || sh_legal);
          if (is_eligible && !game_root_selected) {
            int round = dune_state->GetCurrentRound();
            std::string band = (round <= 3) ? "R1-3" : ((round <= 6) ? "R4-6" : "R7+");

            std::unique_lock<std::mutex> lk(roots_mutex);
            if (roots.size() < static_cast<size_t>(target_total_roots) && !game_root_selected) {
              bool accept = false;
              if (band == "R1-3" && count_r1_3 < 80) accept = true;
              else if (band == "R4-6" && count_r4_6 < 80) accept = true;
              else if (band == "R7+" && count_r7_plus < 60) accept = true;
              else if (g_count > 600 || (count_r1_3 + count_r4_6 >= 150)) {
                accept = true;
              }

              if (accept) {
                game_root_selected = true;
                if (band == "R1-3") count_r1_3++;
                else if (band == "R4-6") count_r4_6++;
                else count_r7_plus++;

                RootRecord rec;
                rec.root_id = static_cast<int>(roots.size());
                rec.episode_id = episode_id;
                rec.search_seat = cur_player;
                rec.round = round;
                rec.round_band = band;
                rec.legal_actions = legal_actions;
                rec.action_history = action_history;

                std::fill(local_obs.begin(), local_obs.end(), 0.0f);
                dune_state->InformationStateTensorWithAppendix(
                    cur_player, open_spiel::dune_imperium::MarketAppendixMode::kFullPublicInformationV3,
                    absl::MakeSpan(local_obs));
                rec.obs = local_obs;

                dune_semantic::CandidateActionData cand_data;
                dune_semantic::ExtractCandidateDescriptors(
                    *dune_state, legal_actions, &cand_data,
                    open_spiel::dune_semantic::kDescriptorSchemaVersionV3);

                rec.sb_legal = sb_legal;
                rec.sb_action = sb_action;
                rec.sh_legal = sh_legal;
                rec.sh_action = sh_action;
                rec.rf_legal = can_buy_rf;
                rec.rf_action = can_buy_rf ? kActionReclaimedForces : kInvalidAction;

                PopulateRootLogits(*model, device, rec, cand_data);
                roots.push_back(std::move(rec));

                if (roots.size() % 25 == 0 || roots.size() == static_cast<size_t>(target_total_roots)) {
                  std::cout << absl::StrFormat("  [%s] Collected %zu/%d roots (R1-3: %d, R4-6: %d, R7+: %d) across %d games\n",
                                               set_name.c_str(), roots.size(), target_total_roots,
                                               count_r1_3, count_r4_6, count_r7_plus, g_count) << std::flush;
                }
              }
            }
          }

          // Greedy policy continuation
          Action chosen_action = kInvalidAction;
          if (legal_actions.size() == 1) {
            chosen_action = legal_actions.front();
          } else {
            open_spiel::ActionsAndProbs prior = evaluator->Prior(*state);
            chosen_action = legal_actions.front();
            double best_p = -1.0;
            for (const auto& ap : prior) {
              if (ap.second > best_p) {
                best_p = ap.second;
                chosen_action = ap.first;
              }
            }
          }
          action_history.push_back(chosen_action);
          state->ApplyAction(chosen_action);
        }
      }
    });
  }

  for (auto& w : workers) {
    if (w.joinable()) w.join();
  }

  auto t1 = std::chrono::steady_clock::now();
  double elapsed = std::chrono::duration<double>(t1 - t0).count();
  std::cout << absl::StrFormat("[A1 ROOTS DONE] %zu roots collected in %.2f s (R1-3: %d, R4-6: %d, R7+: %d). Played %d games.\n",
                               roots.size(), elapsed, count_r1_3, count_r4_6, count_r7_plus, games_played.load()) << std::flush;
  return roots;
}

// Write Roots to JSON
void SaveRootsToJson(const std::vector<RootRecord>& roots, const std::string& path) {
  open_spiel::json::Array arr;
  for (const auto& r : roots) {
    open_spiel::json::Object obj;
    obj["root_id"] = open_spiel::json::Value(static_cast<int64_t>(r.root_id));
    obj["episode_id"] = open_spiel::json::Value(static_cast<int64_t>(r.episode_id));
    obj["search_seat"] = open_spiel::json::Value(static_cast<int64_t>(r.search_seat));
    obj["round"] = open_spiel::json::Value(static_cast<int64_t>(r.round));
    obj["round_band"] = open_spiel::json::Value(r.round_band);

    obj["sb_legal"] = open_spiel::json::Value(r.sb_legal);
    obj["sb_action"] = open_spiel::json::Value(static_cast<int64_t>(r.sb_action));
    obj["sh_legal"] = open_spiel::json::Value(r.sh_legal);
    obj["sh_action"] = open_spiel::json::Value(static_cast<int64_t>(r.sh_action));
    obj["rf_legal"] = open_spiel::json::Value(r.rf_legal);
    obj["rf_action"] = open_spiel::json::Value(static_cast<int64_t>(r.rf_action));

    obj["greedy_choice"] = open_spiel::json::Value(static_cast<int64_t>(r.greedy_choice));
    obj["greedy_prob"] = open_spiel::json::Value(r.greedy_prob);
    obj["sb_prob"] = open_spiel::json::Value(r.sb_prob);
    obj["sh_prob"] = open_spiel::json::Value(r.sh_prob);
    obj["rf_prob"] = open_spiel::json::Value(r.rf_prob);

    obj["z_sb"] = open_spiel::json::Value(r.z_sb);
    obj["capped_sb"] = open_spiel::json::Value(r.capped_sb);
    obj["sech2_sb"] = open_spiel::json::Value(r.sech2_sb);

    obj["z_sh"] = open_spiel::json::Value(r.z_sh);
    obj["capped_sh"] = open_spiel::json::Value(r.capped_sh);
    obj["sech2_sh"] = open_spiel::json::Value(r.sech2_sh);

    open_spiel::json::Array legals_arr;
    for (Action a : r.legal_actions) legals_arr.push_back(open_spiel::json::Value(static_cast<int64_t>(a)));
    obj["legal_actions"] = open_spiel::json::Value(legals_arr);

    open_spiel::json::Array hist_arr;
    for (int64_t a : r.action_history) hist_arr.push_back(open_spiel::json::Value(a));
    obj["action_history"] = open_spiel::json::Value(hist_arr);

    arr.push_back(open_spiel::json::Value(obj));
  }
  std::ofstream f(path);
  f << open_spiel::json::ToString(open_spiel::json::Value(arr));
  f.close();
  std::cout << "Saved " << roots.size() << " roots to " << path << "\n";
}

std::vector<RootRecord> LoadRootsFromJsonFile(const std::string& path) {
  std::ifstream f(path);
  SPIEL_CHECK_TRUE(f.is_open());
  std::stringstream ss;
  ss << f.rdbuf();
  auto val = open_spiel::json::FromString(ss.str());
  SPIEL_CHECK_TRUE(val.has_value());
  const auto& arr = val.value().GetArray();
  std::vector<RootRecord> roots;
  roots.reserve(arr.size());
  for (const auto& item : arr) {
    const auto& obj = item.GetObject();
    RootRecord r;
    r.root_id = static_cast<int>(obj.at("root_id").GetInt());
    r.episode_id = static_cast<uint64_t>(obj.at("episode_id").GetInt());
    r.search_seat = static_cast<int>(obj.at("search_seat").GetInt());
    r.round = static_cast<int>(obj.at("round").GetInt());
    r.round_band = obj.at("round_band").GetString();
    r.greedy_choice = static_cast<Action>(obj.at("greedy_choice").GetInt());
    r.greedy_prob = obj.at("greedy_prob").GetDouble();
    r.sb_legal = obj.at("sb_legal").GetBool();
    r.sb_action = static_cast<Action>(obj.at("sb_action").GetInt());
    r.sb_prob = obj.at("sb_prob").GetDouble();
    r.z_sb = obj.at("z_sb").GetDouble();
    r.capped_sb = obj.at("capped_sb").GetDouble();
    r.sech2_sb = obj.at("sech2_sb").GetDouble();
    r.sh_legal = obj.at("sh_legal").GetBool();
    r.sh_action = static_cast<Action>(obj.at("sh_action").GetInt());
    r.sh_prob = obj.at("sh_prob").GetDouble();
    r.z_sh = obj.at("z_sh").GetDouble();
    r.capped_sh = obj.at("capped_sh").GetDouble();
    r.sech2_sh = obj.at("sech2_sh").GetDouble();
    r.rf_legal = obj.at("rf_legal").GetBool();
    r.rf_action = static_cast<Action>(obj.at("rf_action").GetInt());
    r.rf_prob = obj.at("rf_prob").GetDouble();
    for (const auto& la : obj.at("legal_actions").GetArray()) {
      r.legal_actions.push_back(static_cast<Action>(la.GetInt()));
    }
    for (const auto& ah : obj.at("action_history").GetArray()) {
      r.action_history.push_back(ah.GetInt());
    }
    roots.push_back(r);
  }
  std::cout << "Loaded " << roots.size() << " roots from " << path << "\n";
  return roots;
}

// Compute A2 Logit Diagnostics
open_spiel::json::Object ComputeAndPrintA2Logits(const std::vector<RootRecord>& roots) {
  std::cout << "\n======================================================================\n";
  std::cout << "[A2 LOGITS] Logit Diagnostics on Discovery Set (N = " << roots.size() << " roots)\n";
  std::cout << "======================================================================\n";

  std::vector<double> z_all, capped_all, prob_all, sech2_all;
  std::vector<double> z_sb, capped_sb, prob_sb, sech2_sb;
  std::vector<double> z_sh, capped_sh, prob_sh, sech2_sh;

  for (const auto& r : roots) {
    if (r.sb_legal) {
      z_all.push_back(r.z_sb);
      capped_all.push_back(r.capped_sb);
      prob_all.push_back(r.sb_prob);
      sech2_all.push_back(r.sech2_sb);

      z_sb.push_back(r.z_sb);
      capped_sb.push_back(r.capped_sb);
      prob_sb.push_back(r.sb_prob);
      sech2_sb.push_back(r.sech2_sb);
    }
    if (r.sh_legal) {
      z_all.push_back(r.z_sh);
      capped_all.push_back(r.capped_sh);
      prob_all.push_back(r.sh_prob);
      sech2_all.push_back(r.sech2_sh);

      z_sh.push_back(r.z_sh);
      capped_sh.push_back(r.capped_sh);
      prob_sh.push_back(r.sh_prob);
      sech2_sh.push_back(r.sech2_sh);
    }
  }

  auto report_group = [](const std::string& title,
                         const std::vector<double>& z,
                         const std::vector<double>& capped,
                         const std::vector<double>& prob,
                         const std::vector<double>& sech2) -> open_spiel::json::Object {
    auto sz = ComputePercentiles(z);
    auto sc = ComputePercentiles(capped);
    auto sp = ComputePercentiles(prob);
    auto ss = ComputePercentiles(sech2);

    std::cout << "\n--- " << title << " (N = " << z.size() << " opportunities) ---\n";
    std::cout << absl::StrFormat("  Legal-centred pre-cap logit z:  median = %+.4f, p10 = %+.4f, p90 = %+.4f\n",
                                 sz.median, sz.p10, sz.p90);
    std::cout << absl::StrFormat("  Capped logit (tanh(z/10)*10):   median = %+.4f, p10 = %+.4f, p90 = %+.4f\n",
                                 sc.median, sc.p10, sc.p90);
    std::cout << absl::StrFormat("  Action probability (softmax):   median = %.6f, p10 = %.6f, p90 = %.6f\n",
                                 sp.median, sp.p10, sp.p90);
    std::cout << absl::StrFormat("  Cap derivative sech2(z/10):     median = %.6f, p10 = %.6f, p90 = %.6f\n",
                                 ss.median, ss.p10, ss.p90);

    open_spiel::json::Object grp;
    grp["count"] = open_spiel::json::Value(static_cast<int64_t>(z.size()));
    auto pack_p = [](const SummaryStats& s) {
      open_spiel::json::Object o;
      o["mean"] = open_spiel::json::Value(s.mean);
      o["median"] = open_spiel::json::Value(s.median);
      o["p10"] = open_spiel::json::Value(s.p10);
      o["p90"] = open_spiel::json::Value(s.p90);
      return open_spiel::json::Value(o);
    };
    grp["z_pre_cap"] = pack_p(sz);
    grp["capped_logit"] = pack_p(sc);
    grp["probability"] = pack_p(sp);
    grp["sech2_derivative"] = pack_p(ss);
    return grp;
  };

  open_spiel::json::Object res_obj;
  res_obj["combined"] = open_spiel::json::Value(report_group("Combined Target Cards", z_all, capped_all, prob_all, sech2_all));
  res_obj["scientific_breakthrough"] = open_spiel::json::Value(report_group("Scientific Breakthrough (ID 11)", z_sb, capped_sb, prob_sb, sech2_sb));
  res_obj["stitched_horror"] = open_spiel::json::Value(report_group("Stitched Horror (ID 13)", z_sh, capped_sh, prob_sh, sech2_sh));

  return res_obj;
}

// Run Paired Rollouts (A3 & A4)
std::vector<PerRootRolloutResult> RunPairedRolloutsForRoots(
    const std::shared_ptr<const open_spiel::Game>& game,
    const std::vector<std::shared_ptr<open_spiel::BatchedNNEvaluator>>& evaluators,
    const std::vector<RootRecord>& roots,
    int rollouts_per_candidate,
    int num_threads,
    uint64_t master_seed,
    const std::string& set_name,
    AuditCounters& audit) {
  std::cout << "\n======================================================================\n";
  std::cout << "[PAIRED ROLLOUTS] Running " << rollouts_per_candidate << " rollouts/candidate on "
            << roots.size() << " roots (" << set_name << " set, " << num_threads << " worker threads)\n";
  std::cout << "======================================================================\n";

  SimpleThreadPool pool(num_threads);
  std::vector<PerRootRolloutResult> results(roots.size());
  std::vector<std::unique_ptr<open_spiel::State>> root_states(roots.size());
  std::vector<std::vector<Action>> root_candidates(roots.size());

  for (size_t r_idx = 0; r_idx < roots.size(); ++r_idx) {
    const auto& root = roots[r_idx];
    auto& res = results[r_idx];
    res.root_id = root.root_id;
    res.round = root.round;
    res.round_band = root.round_band;
    res.greedy_choice = root.greedy_choice;
    res.sb_legal = root.sb_legal;
    res.sb_action = root.sb_action;
    res.sh_legal = root.sh_legal;
    res.sh_action = root.sh_action;
    res.rf_legal = root.rf_legal;
    res.rf_action = root.rf_action;

    // Build unique candidate list
    std::vector<Action> candidates;
    candidates.push_back(root.greedy_choice);
    if (root.sb_legal && std::find(candidates.begin(), candidates.end(), root.sb_action) == candidates.end()) {
      candidates.push_back(root.sb_action);
    }
    if (root.sh_legal && std::find(candidates.begin(), candidates.end(), root.sh_action) == candidates.end()) {
      candidates.push_back(root.sh_action);
    }
    if (root.rf_legal && std::find(candidates.begin(), candidates.end(), root.rf_action) == candidates.end()) {
      candidates.push_back(root.rf_action);
    }
    root_candidates[r_idx] = candidates;

    // Prepare return storage: map Action -> vector<double>(rollouts_per_candidate)
    for (Action c : candidates) {
      res.raw_rollout_returns[c].assign(rollouts_per_candidate, 0.0);
      res.detailed_world_rollouts[c].assign(rollouts_per_candidate, WorldRolloutRecord());
    }

    // Replay state from action history
    std::unique_ptr<open_spiel::State> root_state = game->NewInitialState();
    for (int64_t a : root.action_history) {
      root_state->ApplyAction(a);
    }
    const auto* dune_root = dynamic_cast<const open_spiel::dune_imperium::DuneImperiumState*>(root_state.get());
    SPIEL_CHECK_TRUE(dune_root != nullptr);
    root_states[r_idx] = std::move(root_state);
  }

  auto t_start = std::chrono::steady_clock::now();
  size_t total_jobs = roots.size() * rollouts_per_candidate;
  std::atomic<size_t> completed_jobs{0};

  for (size_t job_idx = 0; job_idx < total_jobs; ++job_idx) {
    size_t r_idx = job_idx / rollouts_per_candidate;
    int r = static_cast<int>(job_idx % rollouts_per_candidate);

    pool.Enqueue([r_idx, r, &roots, &root_states, &root_candidates, &results, &evaluators, &audit,
                  master_seed, &completed_jobs, total_jobs, t_start]() {
      const auto& root = roots[r_idx];
      const auto* dune_root = dynamic_cast<const open_spiel::dune_imperium::DuneImperiumState*>(root_states[r_idx].get());
      const auto& candidates = root_candidates[r_idx];
      auto* res_ptr = &results[r_idx];

      uint64_t decision_seed = dune_seed::DeriveSeed(master_seed, root.episode_id, root.search_seat, root.round);
      uint64_t wseed = dune_seed::DeriveSeed(decision_seed, r);
      std::mt19937 wrng(wseed);
      auto rng_func = [&wrng]() {
        return std::generate_canonical<double, 53>(wrng);
      };

      audit.total_resample_calls.fetch_add(1, std::memory_order_relaxed);
      std::unique_ptr<open_spiel::State> world = dune_root->ResampleFromInfostate(root.search_seat, rng_func);
      bool fallback = false;
      if (!world) {
        audit.resample_fallbacks.fetch_add(1, std::memory_order_relaxed);
        world = root_states[r_idx]->Clone();
        fallback = true;
      }

      size_t eval_idx = (r_idx * 256 + r) % evaluators.size();
      auto* active_eval = evaluators[eval_idx].get();

      for (Action c : candidates) {
        std::unique_ptr<open_spiel::State> sim = world->Clone();
        sim->ApplyAction(c);
        RolloutOutcome outcome = SimulateGreedyRollout(sim.get(), root.search_seat, active_eval, wseed + 999);
        res_ptr->raw_rollout_returns[c][r] = outcome.ret;

        WorldRolloutRecord wrec;
        wrec.world_idx = r;
        wrec.action = c;
        wrec.return_val = outcome.ret;
        wrec.is_terminal = outcome.is_terminal;
        wrec.exit_reason = outcome.exit_reason;
        wrec.steps = outcome.steps;
        wrec.final_vp = outcome.final_vp;
        wrec.resample_fallback = fallback;
        wrec.sb_played = outcome.sb_played;
        wrec.sh_played = outcome.sh_played;
        wrec.sb_trashed_for_vp = outcome.sb_trashed_for_vp;
        wrec.vp_by_source = outcome.vp_by_source;
        res_ptr->detailed_world_rollouts[c][r] = wrec;

        audit.total_simulations.fetch_add(1, std::memory_order_relaxed);
        if (outcome.is_terminal) {
          audit.terminal_exits.fetch_add(1, std::memory_order_relaxed);
        } else if (outcome.exit_reason == "step_cap_reached") {
          audit.step_cap_exits.fetch_add(1, std::memory_order_relaxed);
        } else {
          audit.empty_legals_exits.fetch_add(1, std::memory_order_relaxed);
        }

        if (c == root.sb_action && root.sb_legal) {
          audit.sb_rollouts.fetch_add(1, std::memory_order_relaxed);
          if (outcome.sb_played) audit.sb_played_rollouts.fetch_add(1, std::memory_order_relaxed);
          if (outcome.sb_trashed_for_vp) audit.sb_trash_vp_rollouts.fetch_add(1, std::memory_order_relaxed);
        } else if (c == root.sh_action && root.sh_legal) {
          audit.sh_rollouts.fetch_add(1, std::memory_order_relaxed);
          if (outcome.sh_played) audit.sh_played_rollouts.fetch_add(1, std::memory_order_relaxed);
        } else if (c == root.rf_action && root.rf_legal) {
          audit.rf_rollouts.fetch_add(1, std::memory_order_relaxed);
        }
      }

      size_t done = completed_jobs.fetch_add(1, std::memory_order_relaxed) + 1;
      if (done % 500 == 0 || done == total_jobs) {
        double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_start).count();
        double rate = done / (elapsed > 0.0 ? elapsed : 1.0);
        double remaining = (total_jobs - done) / (rate > 0.0 ? rate : 1.0);
        std::cout << absl::StrFormat("  [Progress] %5zu/%5zu jobs (%5.1f%%) | %6.1f s | %5.1f jobs/s | ETA: %4.1f min\n",
                                     done, total_jobs, 100.0 * done / total_jobs, elapsed, rate, remaining / 60.0) << std::flush;
      }
    });
  }

  pool.WaitAll(total_jobs, completed_jobs);

  // Compute Q values and paired differences for each root
  for (size_t r_idx = 0; r_idx < roots.size(); ++r_idx) {
    auto& res = results[r_idx];
    const auto& root = roots[r_idx];

    auto get_q = [&](Action a) -> double {
      const auto& vals = res.raw_rollout_returns[a];
      double sum = 0.0;
      for (double v : vals) sum += v;
      return sum / vals.size();
    };

    res.q_greedy = get_q(root.greedy_choice);
    if (res.sb_legal) res.q_sb = get_q(res.sb_action);
    if (res.sh_legal) res.q_sh = get_q(res.sh_action);
    if (res.rf_legal) res.q_rf = get_q(res.rf_action);

    auto compute_paired_diff = [&](Action cand_a, Action base_a, double& out_gap, double& out_se) {
      const auto& c_vals = res.raw_rollout_returns[cand_a];
      const auto& b_vals = res.raw_rollout_returns[base_a];
      double sum_d = 0.0;
      for (int i = 0; i < rollouts_per_candidate; ++i) {
        sum_d += (c_vals[i] - b_vals[i]);
      }
      out_gap = sum_d / rollouts_per_candidate;
      double var_d = 0.0;
      for (int i = 0; i < rollouts_per_candidate; ++i) {
        double diff = (c_vals[i] - b_vals[i]) - out_gap;
        var_d += diff * diff;
      }
      var_d /= (rollouts_per_candidate - 1);
      out_se = std::sqrt(var_d / rollouts_per_candidate);
    };

    if (res.sb_legal) {
      compute_paired_diff(res.sb_action, root.greedy_choice, res.sb_gap_vs_greedy, res.sb_se_vs_greedy);
      if (res.rf_legal) {
        compute_paired_diff(res.sb_action, res.rf_action, res.sb_gap_vs_rf, res.sb_se_vs_rf);
      }
    }

    if (res.sh_legal) {
      compute_paired_diff(res.sh_action, root.greedy_choice, res.sh_gap_vs_greedy, res.sh_se_vs_greedy);
      if (res.rf_legal) {
        compute_paired_diff(res.sh_action, res.rf_action, res.sh_gap_vs_rf, res.sh_se_vs_rf);
      }
    }

    // Accumulate mechanism VP statistics across rollouts
    for (int r = 0; r < rollouts_per_candidate; ++r) {
      if (res.detailed_world_rollouts.count(root.greedy_choice)) {
        const auto& w_gr = res.detailed_world_rollouts.at(root.greedy_choice)[r];
        audit.sum_final_vp_greedy += w_gr.final_vp;
        audit.count_eval_greedy++;
        for (const auto& [src, delta] : w_gr.vp_by_source) {
          audit.sum_vp_greedy[src] += delta;
        }
      }
      if (root.sb_legal && res.detailed_world_rollouts.count(root.sb_action)) {
        const auto& w_sb = res.detailed_world_rollouts.at(root.sb_action)[r];
        audit.sum_final_vp_sb += w_sb.final_vp;
        audit.count_eval_sb++;
        for (const auto& [src, delta] : w_sb.vp_by_source) {
          audit.sum_vp_sb[src] += delta;
        }
      }
      if (root.sh_legal && res.detailed_world_rollouts.count(root.sh_action)) {
        const auto& w_sh = res.detailed_world_rollouts.at(root.sh_action)[r];
        audit.sum_final_vp_sh += w_sh.final_vp;
        audit.count_eval_sh++;
        for (const auto& [src, delta] : w_sh.vp_by_source) {
          audit.sum_vp_sh[src] += delta;
        }
      }
      if (root.rf_legal && res.detailed_world_rollouts.count(root.rf_action)) {
        const auto& w_rf = res.detailed_world_rollouts.at(root.rf_action)[r];
        audit.sum_final_vp_rf += w_rf.final_vp;
        audit.count_eval_rf++;
        for (const auto& [src, delta] : w_rf.vp_by_source) {
          audit.sum_vp_rf[src] += delta;
        }
      }
    }
  }

  return results;
}

// Compute aggregate report table (mean gap, 95% CI, fraction >= 0.15)
open_spiel::json::Object SummarizeRolloutMetrics(
    const std::vector<PerRootRolloutResult>& results,
    const AuditCounters& audit,
    const std::string& set_name,
    const std::string& out_dir) {
  std::cout << "\n======================================================================\n";
  std::cout << "SUMMARY RESULTS: PAIRED ROLLOUT GAPS (" << set_name << ")\n";
  std::cout << "======================================================================\n";

  struct SlicedData {
    std::vector<double> sb_gaps_vs_greedy;
    std::vector<double> sb_gaps_vs_rf;
    std::vector<double> sh_gaps_vs_greedy;
    std::vector<double> sh_gaps_vs_rf;
  };

  SlicedData all;
  SlicedData r1_3;
  SlicedData r4_6;
  SlicedData r7_plus;

  for (const auto& r : results) {
    SlicedData* target_band = nullptr;
    if (r.round_band == "R1-3") target_band = &r1_3;
    else if (r.round_band == "R4-6") target_band = &r4_6;
    else target_band = &r7_plus;

    if (r.sb_legal) {
      all.sb_gaps_vs_greedy.push_back(r.sb_gap_vs_greedy);
      target_band->sb_gaps_vs_greedy.push_back(r.sb_gap_vs_greedy);
      if (r.rf_legal) {
        all.sb_gaps_vs_rf.push_back(r.sb_gap_vs_rf);
        target_band->sb_gaps_vs_rf.push_back(r.sb_gap_vs_rf);
      }
    }

    if (r.sh_legal) {
      all.sh_gaps_vs_greedy.push_back(r.sh_gap_vs_greedy);
      target_band->sh_gaps_vs_greedy.push_back(r.sh_gap_vs_greedy);
      if (r.rf_legal) {
        all.sh_gaps_vs_rf.push_back(r.sh_gap_vs_rf);
        target_band->sh_gaps_vs_rf.push_back(r.sh_gap_vs_rf);
      }
    }
  }

  auto format_slice = [](const std::string& card_name, const std::string& slice_name, const std::vector<double>& gaps) -> open_spiel::json::Object {
    open_spiel::json::Object obj;
    obj["card"] = open_spiel::json::Value(card_name);
    obj["slice"] = open_spiel::json::Value(slice_name);
    obj["count"] = open_spiel::json::Value(static_cast<int64_t>(gaps.size()));

    if (gaps.empty()) {
      std::cout << absl::StrFormat("  %-25s | %-6s | N=  0 | (no opportunities)\n", card_name.c_str(), slice_name.c_str());
      return obj;
    }

    double sum = 0.0;
    int ge_015_count = 0;
    for (double g : gaps) {
      sum += g;
      if (g >= 0.15) ++ge_015_count;
    }
    double mean_gap = sum / gaps.size();
    double frac_ge_015 = static_cast<double>(ge_015_count) / gaps.size();

    auto [ci_lo, ci_hi] = BootstrapMeanCI(gaps, 10000, 0.95);

    std::cout << absl::StrFormat("  %-25s | %-6s | N=%3zu | Mean Gap: %+.4f, 95%% CI [%+.4f, %+.4f] | Frac >= 0.15: %5.1f%% (%d/%zu)\n",
                                 card_name.c_str(), slice_name.c_str(), gaps.size(),
                                 mean_gap, ci_lo, ci_hi,
                                 frac_ge_015 * 100.0, ge_015_count, gaps.size());

    obj["mean_gap"] = open_spiel::json::Value(mean_gap);
    obj["ci_lower"] = open_spiel::json::Value(ci_lo);
    obj["ci_upper"] = open_spiel::json::Value(ci_hi);
    obj["ci_positive"] = open_spiel::json::Value(ci_lo > 0.0);
    obj["frac_ge_015"] = open_spiel::json::Value(frac_ge_015);
    obj["count_ge_015"] = open_spiel::json::Value(static_cast<int64_t>(ge_015_count));
    return obj;
  };

  std::cout << "\n--- Q(Card) - Q(Policy Greedy Choice) ---\n";
  open_spiel::json::Object summary_json;
  open_spiel::json::Array slices_arr;

  slices_arr.push_back(open_spiel::json::Value(format_slice("Scientific Breakthrough", "All", all.sb_gaps_vs_greedy)));
  slices_arr.push_back(open_spiel::json::Value(format_slice("Scientific Breakthrough", "R1-3", r1_3.sb_gaps_vs_greedy)));
  slices_arr.push_back(open_spiel::json::Value(format_slice("Scientific Breakthrough", "R4-6", r4_6.sb_gaps_vs_greedy)));
  slices_arr.push_back(open_spiel::json::Value(format_slice("Scientific Breakthrough", "R7+", r7_plus.sb_gaps_vs_greedy)));

  slices_arr.push_back(open_spiel::json::Value(format_slice("Stitched Horror", "All", all.sh_gaps_vs_greedy)));
  slices_arr.push_back(open_spiel::json::Value(format_slice("Stitched Horror", "R1-3", r1_3.sh_gaps_vs_greedy)));
  slices_arr.push_back(open_spiel::json::Value(format_slice("Stitched Horror", "R4-6", r4_6.sh_gaps_vs_greedy)));
  slices_arr.push_back(open_spiel::json::Value(format_slice("Stitched Horror", "R7+", r7_plus.sh_gaps_vs_greedy)));

  std::cout << "\n--- Q(Card) - Q(Reclaimed Forces) [where Reclaimed Forces legal] ---\n";
  slices_arr.push_back(open_spiel::json::Value(format_slice("SB vs Reclaimed Forces", "All", all.sb_gaps_vs_rf)));
  slices_arr.push_back(open_spiel::json::Value(format_slice("SH vs Reclaimed Forces", "All", all.sh_gaps_vs_rf)));

  summary_json["slices"] = open_spiel::json::Value(slices_arr);

  // Rollout Audit Output
  std::cout << "\n----------------------------------------------------------------------\n";
  std::cout << "ROLLOUT AUDIT & VALIDITY (" << set_name << "):\n";
  std::cout << "----------------------------------------------------------------------\n";
  size_t total_sims = audit.total_simulations.load();
  size_t term_exits = audit.terminal_exits.load();
  size_t cap_exits = audit.step_cap_exits.load();
  size_t empty_exits = audit.empty_legals_exits.load();
  size_t total_resamples = audit.total_resample_calls.load();
  size_t fallbacks = audit.resample_fallbacks.load();
  double fallback_pct = total_resamples > 0 ? (100.0 * fallbacks / total_resamples) : 0.0;

  std::cout << absl::StrFormat("  Total Simulations:        %zu\n", total_sims);
  std::cout << absl::StrFormat("  Terminal Exits:           %zu (%5.2f%%)\n", term_exits, total_sims > 0 ? (100.0 * term_exits / total_sims) : 0.0);
  std::cout << absl::StrFormat("  Step-Cap Exits (2500):    %zu\n", cap_exits);
  std::cout << absl::StrFormat("  Empty Legal Exits:        %zu\n", empty_exits);
  std::cout << absl::StrFormat("  Infostate Resamples:      %zu\n", total_resamples);
  std::cout << absl::StrFormat("  True-State Fallbacks:     %zu (%5.2f%%)\n", fallbacks, fallback_pct);

  open_spiel::json::Object audit_json;
  audit_json["total_simulations"] = open_spiel::json::Value(static_cast<int64_t>(total_sims));
  audit_json["terminal_exits"] = open_spiel::json::Value(static_cast<int64_t>(term_exits));
  audit_json["step_cap_exits"] = open_spiel::json::Value(static_cast<int64_t>(cap_exits));
  audit_json["empty_legals_exits"] = open_spiel::json::Value(static_cast<int64_t>(empty_exits));
  audit_json["total_resample_calls"] = open_spiel::json::Value(static_cast<int64_t>(total_resamples));
  audit_json["resample_fallbacks"] = open_spiel::json::Value(static_cast<int64_t>(fallbacks));
  audit_json["resample_fallback_rate"] = open_spiel::json::Value(fallback_pct / 100.0);
  summary_json["rollout_audit"] = open_spiel::json::Value(audit_json);

  // Mechanism Tracking Output
  std::cout << "\n----------------------------------------------------------------------\n";
  std::cout << "MECHANISM TRACKING (" << set_name << "):\n";
  std::cout << "----------------------------------------------------------------------\n";
  open_spiel::json::Object mech_json;

  size_t sb_n = audit.sb_rollouts.load();
  size_t sb_play_n = audit.sb_played_rollouts.load();
  size_t sb_trash_n = audit.sb_trash_vp_rollouts.load();
  double sb_play_rate = sb_n > 0 ? (static_cast<double>(sb_play_n) / sb_n) : 0.0;
  double sb_trash_rate = sb_n > 0 ? (static_cast<double>(sb_trash_n) / sb_n) : 0.0;
  double mean_vp_sb = audit.count_eval_sb > 0 ? (audit.sum_final_vp_sb / audit.count_eval_sb) : 0.0;
  double mean_vp_greedy = audit.count_eval_greedy > 0 ? (audit.sum_final_vp_greedy / audit.count_eval_greedy) : 0.0;

  std::cout << "  Scientific Breakthrough (ID 11):\n";
  std::cout << absl::StrFormat("    Card Played Rate:       %5.1f%% (%zu / %zu)\n", sb_play_rate * 100.0, sb_play_n, sb_n);
  std::cout << absl::StrFormat("    Self-Trash for VP Rate: %5.1f%% (%zu / %zu)\n", sb_trash_rate * 100.0, sb_trash_n, sb_n);
  std::cout << absl::StrFormat("    Mean Final VP:          %.3f (vs Greedy %.3f, Delta %+.3f)\n", mean_vp_sb, mean_vp_greedy, mean_vp_sb - mean_vp_greedy);
  std::cout << "    VP Delta Breakdown by Source (SB vs Greedy):\n";

  open_spiel::json::Object sb_mech;
  sb_mech["total_rollouts"] = open_spiel::json::Value(static_cast<int64_t>(sb_n));
  sb_mech["card_play_rate"] = open_spiel::json::Value(sb_play_rate);
  sb_mech["trash_for_vp_rate"] = open_spiel::json::Value(sb_trash_rate);
  sb_mech["mean_final_vp"] = open_spiel::json::Value(mean_vp_sb);
  sb_mech["mean_final_vp_greedy"] = open_spiel::json::Value(mean_vp_greedy);
  sb_mech["mean_final_vp_delta"] = open_spiel::json::Value(mean_vp_sb - mean_vp_greedy);

  open_spiel::json::Object sb_vp_deltas;
  std::set<std::string> all_sources;
  for (const auto& [src, _] : audit.sum_vp_sb) all_sources.insert(src);
  for (const auto& [src, _] : audit.sum_vp_greedy) all_sources.insert(src);
  for (const auto& [src, _] : audit.sum_vp_sh) all_sources.insert(src);
  for (const auto& [src, _] : audit.sum_vp_rf) all_sources.insert(src);

  for (const auto& src : all_sources) {
    double v_sb = (audit.count_eval_sb > 0 && audit.sum_vp_sb.count(src)) ? (audit.sum_vp_sb.at(src) / audit.count_eval_sb) : 0.0;
    double v_gr = (audit.count_eval_greedy > 0 && audit.sum_vp_greedy.count(src)) ? (audit.sum_vp_greedy.at(src) / audit.count_eval_greedy) : 0.0;
    double delta = v_sb - v_gr;
    if (std::abs(delta) >= 0.005) {
      std::cout << absl::StrFormat("      %-25s: %+.3f VP\n", src.c_str(), delta);
      sb_vp_deltas[src] = open_spiel::json::Value(delta);
    }
  }
  sb_mech["vp_delta_by_source"] = open_spiel::json::Value(sb_vp_deltas);
  mech_json["scientific_breakthrough"] = open_spiel::json::Value(sb_mech);

  size_t sh_n = audit.sh_rollouts.load();
  size_t sh_play_n = audit.sh_played_rollouts.load();
  double sh_play_rate = sh_n > 0 ? (static_cast<double>(sh_play_n) / sh_n) : 0.0;
  double mean_vp_sh = audit.count_eval_sh > 0 ? (audit.sum_final_vp_sh / audit.count_eval_sh) : 0.0;

  std::cout << "  Stitched Horror (ID 13):\n";
  std::cout << absl::StrFormat("    Card Played Rate:       %5.1f%% (%zu / %zu)\n", sh_play_rate * 100.0, sh_play_n, sh_n);
  std::cout << absl::StrFormat("    Mean Final VP:          %.3f (vs Greedy %.3f, Delta %+.3f)\n", mean_vp_sh, mean_vp_greedy, mean_vp_sh - mean_vp_greedy);
  std::cout << "    VP Delta Breakdown by Source (SH vs Greedy):\n";

  open_spiel::json::Object sh_mech;
  sh_mech["total_rollouts"] = open_spiel::json::Value(static_cast<int64_t>(sh_n));
  sh_mech["card_play_rate"] = open_spiel::json::Value(sh_play_rate);
  sh_mech["mean_final_vp"] = open_spiel::json::Value(mean_vp_sh);
  sh_mech["mean_final_vp_greedy"] = open_spiel::json::Value(mean_vp_greedy);
  sh_mech["mean_final_vp_delta"] = open_spiel::json::Value(mean_vp_sh - mean_vp_greedy);

  open_spiel::json::Object sh_vp_deltas;
  for (const auto& src : all_sources) {
    double v_sh = (audit.count_eval_sh > 0 && audit.sum_vp_sh.count(src)) ? (audit.sum_vp_sh.at(src) / audit.count_eval_sh) : 0.0;
    double v_gr = (audit.count_eval_greedy > 0 && audit.sum_vp_greedy.count(src)) ? (audit.sum_vp_greedy.at(src) / audit.count_eval_greedy) : 0.0;
    double delta = v_sh - v_gr;
    if (std::abs(delta) >= 0.005) {
      std::cout << absl::StrFormat("      %-25s: %+.3f VP\n", src.c_str(), delta);
      sh_vp_deltas[src] = open_spiel::json::Value(delta);
    }
  }
  sh_mech["vp_delta_by_source"] = open_spiel::json::Value(sh_vp_deltas);
  mech_json["stitched_horror"] = open_spiel::json::Value(sh_mech);

  summary_json["mechanism_tracking"] = open_spiel::json::Value(mech_json);

  // Persist detailed per-world rollout records to disk
  if (!out_dir.empty()) {
    std::string det_path = out_dir + "/a4_rollouts_" + set_name + "_detailed.json";
    open_spiel::json::Array det_arr;
    for (const auto& res : results) {
      open_spiel::json::Object r_obj;
      r_obj["root_id"] = open_spiel::json::Value(static_cast<int64_t>(res.root_id));
      r_obj["round"] = open_spiel::json::Value(static_cast<int64_t>(res.round));
      r_obj["round_band"] = open_spiel::json::Value(res.round_band);
      r_obj["greedy_choice"] = open_spiel::json::Value(static_cast<int64_t>(res.greedy_choice));
      r_obj["q_greedy"] = open_spiel::json::Value(res.q_greedy);
      r_obj["sb_legal"] = open_spiel::json::Value(res.sb_legal);
      if (res.sb_legal) {
        r_obj["sb_action"] = open_spiel::json::Value(static_cast<int64_t>(res.sb_action));
        r_obj["q_sb"] = open_spiel::json::Value(res.q_sb);
        r_obj["sb_gap_vs_greedy"] = open_spiel::json::Value(res.sb_gap_vs_greedy);
      }
      r_obj["sh_legal"] = open_spiel::json::Value(res.sh_legal);
      if (res.sh_legal) {
        r_obj["sh_action"] = open_spiel::json::Value(static_cast<int64_t>(res.sh_action));
        r_obj["q_sh"] = open_spiel::json::Value(res.q_sh);
        r_obj["sh_gap_vs_greedy"] = open_spiel::json::Value(res.sh_gap_vs_greedy);
      }
      r_obj["rf_legal"] = open_spiel::json::Value(res.rf_legal);
      if (res.rf_legal) {
        r_obj["rf_action"] = open_spiel::json::Value(static_cast<int64_t>(res.rf_action));
        r_obj["q_rf"] = open_spiel::json::Value(res.q_rf);
      }

      open_spiel::json::Object worlds_obj;
      for (const auto& [act, wlist] : res.detailed_world_rollouts) {
        open_spiel::json::Array w_arr;
        for (const auto& w : wlist) {
          open_spiel::json::Object w_item;
          w_item["world_idx"] = open_spiel::json::Value(static_cast<int64_t>(w.world_idx));
          w_item["return"] = open_spiel::json::Value(w.return_val);
          w_item["is_terminal"] = open_spiel::json::Value(w.is_terminal);
          w_item["exit_reason"] = open_spiel::json::Value(w.exit_reason);
          w_item["steps"] = open_spiel::json::Value(static_cast<int64_t>(w.steps));
          w_item["final_vp"] = open_spiel::json::Value(static_cast<int64_t>(w.final_vp));
          w_item["resample_fallback"] = open_spiel::json::Value(w.resample_fallback);
          w_item["sb_played"] = open_spiel::json::Value(w.sb_played);
          w_item["sh_played"] = open_spiel::json::Value(w.sh_played);
          w_item["sb_trashed_for_vp"] = open_spiel::json::Value(w.sb_trashed_for_vp);
          open_spiel::json::Object vps_obj;
          for (const auto& [src, delta] : w.vp_by_source) {
            vps_obj[src] = open_spiel::json::Value(static_cast<int64_t>(delta));
          }
          w_item["vp_by_source"] = open_spiel::json::Value(vps_obj);
          w_arr.push_back(open_spiel::json::Value(w_item));
        }
        worlds_obj[std::to_string(act)] = open_spiel::json::Value(w_arr);
      }
      r_obj["worlds"] = open_spiel::json::Value(worlds_obj);
      det_arr.push_back(open_spiel::json::Value(r_obj));
    }
    std::ofstream det_f(det_path);
    det_f << open_spiel::json::ToString(open_spiel::json::Value(det_arr));
    det_f.close();
    std::cout << "[DETAILS PERSISTED] Detailed per-world rollouts saved to: " << det_path << "\n";
  }

  return summary_json;
}

} // namespace

int main(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);

  std::string ckpt_path = absl::GetFlag(FLAGS_model_checkpoint);
  std::string out_dir = absl::GetFlag(FLAGS_output_dir);
  int num_roots = absl::GetFlag(FLAGS_num_roots);
  int num_rollouts = absl::GetFlag(FLAGS_num_rollouts);
  int num_threads = absl::GetFlag(FLAGS_threads);
  int num_evals = absl::GetFlag(FLAGS_num_evaluators);
  int target_batch_size = absl::GetFlag(FLAGS_target_batch_size);
  uint64_t master_seed = absl::GetFlag(FLAGS_master_seed);
  uint64_t disc_start_seed = absl::GetFlag(FLAGS_discovery_start_seed);
  uint64_t conf_start_seed = absl::GetFlag(FLAGS_confirmation_start_seed);

  std::cout << "======================================================================\n";
  std::cout << "PART A: OFFLINE CHECK ON STARVED TLEILAXU CARDS\n";
  std::cout << "Scientific Breakthrough (ID 11) & Stitched Horror (ID 13)\n";
  std::cout << "======================================================================\n";
  std::cout << "Checkpoint:          " << ckpt_path << "\n";
  std::cout << "Output Directory:    " << out_dir << "\n";
  std::cout << "Num Roots / Set:     " << num_roots << "\n";
  std::cout << "Rollouts / Cand:     " << num_rollouts << "\n";
  std::cout << "Threads:             " << num_threads << "\n";
  std::cout << "Evaluators:          " << num_evals << "\n";
  std::cout << "Target Batch Size:   " << target_batch_size << "\n";
  std::cout << "Master Seed:         " << master_seed << "\n";

  // Verify Checkpoint Integrity & SHA-256
  if (!std::filesystem::exists(ckpt_path)) {
    open_spiel::SpielFatalError("Model checkpoint does not exist: " + ckpt_path);
  }
  size_t ckpt_size = 0;
  std::string ckpt_sha = open_spiel::ComputeFileSHA256(ckpt_path, &ckpt_size);
  std::cout << "Checkpoint Size:     " << ckpt_size << " bytes\n";
  std::cout << "Checkpoint SHA-256:  " << ckpt_sha << "\n";

  std::filesystem::create_directories(out_dir);

  // Initialize Game with Immortality expansion
  std::shared_ptr<const open_spiel::Game> game = open_spiel::LoadGame("dune_imperium(enable_immortality=true)");
  if (!game) {
    open_spiel::SpielFatalError("Failed to load dune_imperium(enable_immortality=true)");
  }

  // Device & Evaluator setup
  torch::Device device(torch::kCUDA, 0);
  if (!torch::cuda::is_available()) {
    device = torch::Device(torch::kCPU);
  }
  std::cout << "Inference Device:    " << (device.is_cuda() ? "CUDA (GPU)" : "CPU") << "\n";

  std::vector<std::shared_ptr<open_spiel::SharedDunePolicyValueNetImpl>> models(num_evals);
  std::vector<std::unique_ptr<std::shared_mutex>> model_mutexes(num_evals);
  std::vector<std::shared_ptr<open_spiel::BatchedEvaluator>> coords(num_evals);
  std::vector<std::shared_ptr<open_spiel::BatchedNNEvaluator>> evaluators(num_evals);

  for (int e = 0; e < num_evals; ++e) {
    models[e] = std::make_shared<open_spiel::SharedDunePolicyValueNetImpl>(
        open_spiel::dune_imperium::kFullPublicInformationStateSize, 2048, 2391, 8,
        /*nonlinear=*/false, /*aux=*/false, /*seed=*/0, /*scorer=*/true);
    models[e]->to(device);
    open_spiel::LoadModelCheckpointRobust(models[e], ckpt_path, device);
    models[e]->eval();

    model_mutexes[e] = std::make_unique<std::shared_mutex>();
    coords[e] = std::make_shared<open_spiel::BatchedEvaluator>(
        models[e], target_batch_size, /*timeout_ms=*/1, device, model_mutexes[e].get(), 10.0f,
        /*device_synchronize=*/true, /*high_priority_stream=*/true,
        /*emit_batch_membership=*/false, /*rollout_amp=*/true, /*allow_tf32=*/true);
    evaluators[e] = std::make_shared<open_spiel::BatchedNNEvaluator>(coords[e], 10.0f);
  }

  if (absl::GetFlag(FLAGS_confirmation_only)) {
    std::cout << "\n======================================================================\n";
    std::cout << "[CONFIRMATION ONLY MODE] Running Instrumented Rollouts on Confirmation Roots\n";
    std::cout << "======================================================================\n";
    std::string conf_json_path = out_dir + "/roots_confirmation.json";
    if (!std::filesystem::exists(conf_json_path)) {
      open_spiel::SpielFatalError("roots_confirmation.json does not exist in " + out_dir);
    }
    std::vector<RootRecord> confirmation_roots = LoadRootsFromJsonFile(conf_json_path);

    AuditCounters conf_audit;
    std::vector<PerRootRolloutResult> confirmation_rollouts = RunPairedRolloutsForRoots(
        game, evaluators, confirmation_roots, num_rollouts, num_threads, master_seed + 1000, "confirmation", conf_audit);

    open_spiel::json::Object conf_summary_json = SummarizeRolloutMetrics(confirmation_rollouts, conf_audit, "confirmation", out_dir);
    std::string a4_summary_path = out_dir + "/a4_rollouts_confirmation.json";
    std::ofstream a4_f(a4_summary_path);
    a4_f << open_spiel::json::ToString(open_spiel::json::Value(conf_summary_json));
    a4_f.close();

    // Update receipt if exists
    std::string receipt_path = out_dir + "/tleilaxu_card_check_receipt.json";
    if (std::filesystem::exists(receipt_path)) {
      std::ifstream rf_in(receipt_path);
      std::stringstream ss;
      ss << rf_in.rdbuf();
      auto r_val = open_spiel::json::FromString(ss.str());
      if (r_val.has_value()) {
        auto r_obj = r_val.value().GetObject();
        r_obj["a4_confirmation_summary"] = open_spiel::json::Value(conf_summary_json);
        std::ofstream rf_out(receipt_path);
        rf_out << open_spiel::json::ToString(open_spiel::json::Value(r_obj));
        rf_out.close();
      }
    }
    std::cout << "\n[CONFIRMATION COMPLETE] Master receipt updated: " << receipt_path << "\n";
    return 0;
  }

  // A1: Collect Discovery and Confirmation Roots (load if already exist)
  std::vector<RootRecord> discovery_roots;
  std::string disc_json_path = out_dir + "/roots_discovery.json";
  if (std::filesystem::exists(disc_json_path)) {
    std::cout << "[A1 ROOTS] Loading existing discovery roots from " << disc_json_path << "\n";
    discovery_roots = LoadRootsFromJsonFile(disc_json_path);
  } else {
    discovery_roots = CollectRootsSet(
        game, models[0], evaluators, device, master_seed, disc_start_seed, num_roots, num_threads, "discovery");
    SaveRootsToJson(discovery_roots, disc_json_path);
  }

  std::vector<RootRecord> confirmation_roots;
  std::string conf_json_path = out_dir + "/roots_confirmation.json";
  if (std::filesystem::exists(conf_json_path)) {
    std::cout << "[A1 ROOTS] Loading existing confirmation roots from " << conf_json_path << "\n";
    confirmation_roots = LoadRootsFromJsonFile(conf_json_path);
  } else {
    confirmation_roots = CollectRootsSet(
        game, models[0], evaluators, device, master_seed, conf_start_seed, num_roots, num_threads, "confirmation");
    SaveRootsToJson(confirmation_roots, conf_json_path);
  }

  // A2: Logits Diagnostics on Discovery Set
  open_spiel::json::Object a2_logits_json = ComputeAndPrintA2Logits(discovery_roots);
  std::string a2_json_path = out_dir + "/a2_logits_discovery.json";
  std::ofstream a2_f(a2_json_path);
  a2_f << open_spiel::json::ToString(open_spiel::json::Value(a2_logits_json));
  a2_f.close();

  // A3: Paired Rollouts on Discovery Set
  AuditCounters disc_audit;
  std::vector<PerRootRolloutResult> discovery_rollouts = RunPairedRolloutsForRoots(
      game, evaluators, discovery_roots, num_rollouts, num_threads, master_seed, "discovery", disc_audit);

  open_spiel::json::Object disc_summary_json = SummarizeRolloutMetrics(discovery_rollouts, disc_audit, "discovery", out_dir);
  std::string a3_summary_path = out_dir + "/a3_rollouts_discovery.json";
  std::ofstream a3_f(a3_summary_path);
  a3_f << open_spiel::json::ToString(open_spiel::json::Value(disc_summary_json));
  a3_f.close();

  // A4: Confirmation check
  // Check if any card or round band has discovery CI lower bound > 0
  bool any_positive = false;
  const auto& slices_arr = disc_summary_json.at("slices").GetArray();
  for (const auto& sl_val : slices_arr) {
    const auto& sl_obj = sl_val.GetObject();
    if (sl_obj.find("ci_positive") != sl_obj.end() && sl_obj.at("ci_positive").GetBool()) {
      any_positive = true;
      std::cout << "[CONFIRMATION TRIGGERED] " << sl_obj.at("card").GetString()
                << " in " << sl_obj.at("slice").GetString() << " has positive CI lower bound ("
                << sl_obj.at("ci_lower").GetDouble() << ")!\n";
    }
  }

  open_spiel::json::Object conf_summary_json;
  if (any_positive) {
    std::cout << "\n======================================================================\n";
    std::cout << "[A4 CONFIRMATION] Running Confirmation Paired Rollouts...\n";
    std::cout << "======================================================================\n";
    AuditCounters conf_audit;
    std::vector<PerRootRolloutResult> confirmation_rollouts = RunPairedRolloutsForRoots(
        game, evaluators, confirmation_roots, num_rollouts, num_threads, master_seed + 1000, "confirmation", conf_audit);

    conf_summary_json = SummarizeRolloutMetrics(confirmation_rollouts, conf_audit, "confirmation", out_dir);
    std::string a4_summary_path = out_dir + "/a4_rollouts_confirmation.json";
    std::ofstream a4_f(a4_summary_path);
    a4_f << open_spiel::json::ToString(open_spiel::json::Value(conf_summary_json));
    a4_f.close();
  } else {
    std::cout << "\n======================================================================\n";
    std::cout << "[A4 CONFIRMATION] No card or round band had discovery CI lower bound > 0.\n";
    std::cout << "Per agreed protocol: Report and stop (no confirmation rollouts needed).\n";
    std::cout << "======================================================================\n";
  }

  // Master Receipt JSON
  open_spiel::json::Object receipt;
  receipt["study_name"] = open_spiel::json::Value("starved_tleilaxu_cards_offline_check");
  receipt["model_checkpoint"] = open_spiel::json::Value(ckpt_path);
  receipt["model_sha256"] = open_spiel::json::Value(ckpt_sha);
  receipt["discovery_roots_count"] = open_spiel::json::Value(static_cast<int64_t>(discovery_roots.size()));
  receipt["confirmation_roots_count"] = open_spiel::json::Value(static_cast<int64_t>(confirmation_roots.size()));
  receipt["rollouts_per_candidate"] = open_spiel::json::Value(static_cast<int64_t>(num_rollouts));
  receipt["a2_logits_discovery"] = open_spiel::json::Value(a2_logits_json);
  receipt["a3_discovery_summary"] = open_spiel::json::Value(disc_summary_json);
  receipt["confirmation_triggered"] = open_spiel::json::Value(any_positive);
  if (any_positive) {
    receipt["a4_confirmation_summary"] = open_spiel::json::Value(conf_summary_json);
  }

  std::string receipt_path = out_dir + "/tleilaxu_card_check_receipt.json";
  std::ofstream rf(receipt_path);
  rf << open_spiel::json::ToString(open_spiel::json::Value(receipt));
  rf.close();

  std::cout << "\n[STUDY COMPLETE] Master receipt written to: " << receipt_path << "\n";
  return 0;
}
