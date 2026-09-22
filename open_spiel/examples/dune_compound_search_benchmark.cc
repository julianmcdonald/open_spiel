#include <iostream>
#include <vector>
#include <string>
#include <memory>
#include <chrono>
#include <thread>
#include <iomanip>
#include <filesystem>

#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"
#include "open_spiel/games/dune_imperium/dune_imperium.h"
#include "open_spiel/games/dune_imperium/dune_imperium_common.h"
#include "absl/flags/flag.h"
#include "absl/flags/parse.h"

#include <torch/torch.h>

#include "dune_network.h"
#include "dune_evaluator.h"
#include "dune_batched_evaluator.h"
#include "dune_seed_utils.h"
#include "dune_compound_turn_search.h"

using namespace open_spiel;
using namespace open_spiel::dune_imperium;

// Link satisfaction flags for dune_ppo_training_utils
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
ABSL_FLAG(uint64_t, shaping_start_env_steps, 206830543, "");
ABSL_FLAG(uint64_t, shaping_decay_env_steps, 0, "");
ABSL_FLAG(bool, diagnostics_only, false, "");

ABSL_FLAG(std::string, model_checkpoint,
          "/run/media/warcr/Storage/dune_drl_runtime/round7/u21328_continuation_20260921_004103/checkpoints/ppo_model_update_22263.pt",
          "Path to U22263 teacher checkpoint.");
ABSL_FLAG(int, target_batch_size, 128, "GPU evaluator target batch size.");
ABSL_FLAG(int, num_roots, 32, "Number of roots to benchmark.");

struct BenchmarkResult {
  std::string name;
  int threads = 0;
  int coordinators = 0;
  double elapsed_seconds = 0.0;
  double roots_per_sec = 0.0;
  int overrides = 0;
  int fallbacks = 0;
};

std::vector<SearchSnapshot> GenerateBenchmarkSnapshots(
    int num_roots, const Game& game, std::shared_ptr<BatchedNNEvaluator> eval, uint64_t master_seed) {
  std::cout << "[BENCHMARK] Generating " << num_roots << " decision snapshots from self-play...\n";
  std::vector<SearchSnapshot> snapshots;
  snapshots.reserve(num_roots);

  uint64_t ep_counter = 0;
  while (static_cast<int>(snapshots.size()) < num_roots) {
    std::unique_ptr<State> state = game.NewInitialState();
    auto dune_state = dynamic_cast<const DuneImperiumState*>(state.get());
    std::mt19937_64 rng(dune_seed::DeriveSeed(master_seed, 0x534E4150ULL, ep_counter++));

    while (!state->IsTerminal() && static_cast<int>(snapshots.size()) < num_roots) {
      if (state->IsChanceNode()) {
        auto outcomes = state->ChanceOutcomes();
        Action a = SampleAction(outcomes, rng).first;
        state->ApplyAction(a);
        continue;
      }
      Player cur = state->CurrentPlayer();
      auto legals = state->LegalActions();
      if (legals.empty()) break;

      if (legals.size() > 1) {
        SearchSnapshot snap;
        snap.sim_state = state->Clone();
        snap.player = cur;
        snap.legal_actions = legals;
        snap.chosen_action = legals.front();
        snap.observation = eval->GetConsumedObservation(*state, cur);
        snap.root_id = snapshots.size();

        ActionsAndProbs prior = eval->Prior(*state);
        snap.raw_policy.reserve(legals.size());
        for (Action a : legals) {
          double p = 0.0;
          for (const auto& ap : prior) {
            if (ap.first == a) { p = ap.second; break; }
          }
          snap.raw_policy.push_back({a, p});
        }
        snapshots.push_back(std::move(snap));
      }

      ActionsAndProbs prior = eval->Prior(*state);
      Action a = PickGreedyAction(prior, legals);
      state->ApplyAction(a);
    }
  }
  std::cout << "[BENCHMARK] Generated " << snapshots.size() << " valid pre-action decision snapshots.\n";
  return snapshots;
}

BenchmarkResult RunConfig(
    const std::string& name,
    int threads,
    int num_evals,
    int target_batch_size,
    const std::vector<SearchSnapshot>& base_snapshots,
    const std::string& model_path,
    torch::Device device) {
  std::cout << "\n--------------------------------------------------------------------------------\n";
  std::cout << "Testing Config: " << name << " (" << threads << " threads / " << num_evals << " coordinators, batch=" << target_batch_size << ")\n";
  std::cout << "--------------------------------------------------------------------------------\n";

  // Create coordinator pool
  std::vector<std::shared_ptr<SharedDunePolicyValueNetImpl>> models(num_evals);
  std::vector<std::unique_ptr<std::shared_mutex>> mutexes(num_evals);
  std::vector<std::shared_ptr<BatchedEvaluator>> coords(num_evals);
  std::vector<std::shared_ptr<BatchedNNEvaluator>> evaluators(num_evals);

  for (int e = 0; e < num_evals; ++e) {
    models[e] = std::make_shared<SharedDunePolicyValueNetImpl>(
        kFullPublicInformationStateSize, 2048, 2391, 8,
        /*nonlinear_value_head=*/false, /*with_aux_heads=*/false,
        /*head_init_seed=*/0, /*with_semantic_scorer=*/true);
    models[e]->to(device);
    torch::load(models[e], model_path, device);
    models[e]->eval();

    mutexes[e] = std::make_unique<std::shared_mutex>();
    coords[e] = std::make_shared<BatchedEvaluator>(
        models[e], target_batch_size, /*timeout_ms=*/1, device, mutexes[e].get(), 10.0f,
        /*device_synchronize=*/false, /*high_priority_stream=*/true,
        /*emit_batch_membership=*/false, /*rollout_amp=*/true, /*allow_tf32=*/true);
    evaluators[e] = std::make_shared<BatchedNNEvaluator>(coords[e], 10.0f);
  }

  // Clone snapshots for independent run
  std::vector<SearchSnapshot> test_snapshots;
  test_snapshots.reserve(base_snapshots.size());
  for (const auto& s : base_snapshots) {
    SearchSnapshot c;
    c.sim_state = s.sim_state->Clone();
    c.player = s.player;
    c.observation = s.observation;
    c.legal_actions = s.legal_actions;
    c.raw_policy = s.raw_policy;
    c.chosen_action = s.chosen_action;
    c.behavior_log_prob = s.behavior_log_prob;
    c.root_id = s.root_id;
    test_snapshots.push_back(std::move(c));
  }

  // Warmup 1 root
  std::vector<SearchSnapshot> warmup_snaps;
  warmup_snaps.push_back(SearchSnapshot{
      test_snapshots[0].sim_state->Clone(),
      test_snapshots[0].player,
      test_snapshots[0].observation,
      test_snapshots[0].legal_actions,
      test_snapshots[0].raw_policy,
      test_snapshots[0].chosen_action,
      test_snapshots[0].behavior_log_prob,
      0});
  BatchCompoundSearchSupervision(warmup_snaps, evaluators, 3, 64, 0.15, 0.50, 0.80, 12345, 1);

  // Execute benchmark
  auto t_start = std::chrono::steady_clock::now();
  auto sup_res = BatchCompoundSearchSupervision(
      test_snapshots, evaluators, 3, 64, 0.15, 0.50, 0.80, 20260922, threads);
  auto t_end = std::chrono::steady_clock::now();

  double elapsed = std::chrono::duration<double>(t_end - t_start).count();
  double rps = sup_res.roots_evaluated / std::max(1e-4, elapsed);

  std::cout << "Results for " << name << ":\n"
            << "  Roots Evaluated: " << sup_res.roots_evaluated << "\n"
            << "  Elapsed Time:    " << std::fixed << std::setprecision(3) << elapsed << " s\n"
            << "  Throughput:      " << std::fixed << std::setprecision(2) << rps << " roots/sec\n"
            << "  Overrides:       " << sup_res.overrides << " ("
            << std::fixed << std::setprecision(1) << (sup_res.mean_override_rate * 100.0) << "%)\n"
            << "  Fallbacks to mu: " << sup_res.fallbacks_to_mu << "\n";

  BenchmarkResult res;
  res.name = name;
  res.threads = threads;
  res.coordinators = num_evals;
  res.elapsed_seconds = elapsed;
  res.roots_per_sec = rps;
  res.overrides = sup_res.overrides;
  res.fallbacks = sup_res.fallbacks_to_mu;
  return res;
}

int main(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);
  std::string model_path = absl::GetFlag(FLAGS_model_checkpoint);
  int target_batch = absl::GetFlag(FLAGS_target_batch_size);
  int num_roots = absl::GetFlag(FLAGS_num_roots);

  if (!std::filesystem::exists(model_path)) {
    std::cerr << "Checkpoint does not exist: " << model_path << "\n";
    return 1;
  }

  torch::Device device(torch::kCPU);
  if (torch::cuda::is_available()) {
    device = torch::Device(torch::kCUDA, 0);
    std::cout << "[CUDA] Initialized on device: " << device << "\n";
  } else {
    std::cerr << "CUDA is required for benchmark!\n";
    return 1;
  }

  std::shared_ptr<const Game> game = LoadGame("dune_imperium");

  // Create single generator evaluator
  auto gen_model = std::make_shared<SharedDunePolicyValueNetImpl>(
      kFullPublicInformationStateSize, 2048, 2391, 8,
      /*nonlinear_value_head=*/false, /*with_aux_heads=*/false,
      /*head_init_seed=*/0, /*with_semantic_scorer=*/true);
  gen_model->to(device);
  torch::load(gen_model, model_path, device);
  gen_model->eval();

  std::shared_mutex gen_mutex;
  auto gen_coord = std::make_shared<BatchedEvaluator>(
      gen_model, 32, /*timeout_ms=*/1, device, &gen_mutex, 10.0f,
      /*device_synchronize=*/false, /*high_priority_stream=*/true,
      /*emit_batch_membership=*/false, /*rollout_amp=*/true, /*allow_tf32=*/true);
  auto gen_eval = std::make_shared<BatchedNNEvaluator>(gen_coord, 10.0f);

  auto snapshots = GenerateBenchmarkSnapshots(num_roots, *game, gen_eval, 20260921);

  std::vector<BenchmarkResult> results;
  results.push_back(RunConfig("22t / 6c (Current Best)", 22, 6, target_batch, snapshots, model_path, device));
  results.push_back(RunConfig("32t / 6c",                32, 6, target_batch, snapshots, model_path, device));
  results.push_back(RunConfig("32t / 8c",                32, 8, target_batch, snapshots, model_path, device));
  results.push_back(RunConfig("48t / 6c",                48, 6, target_batch, snapshots, model_path, device));
  results.push_back(RunConfig("48t / 8c",                48, 8, target_batch, snapshots, model_path, device));

  std::cout << "\n================================================================================\n";
  std::cout << "BENCHMARK SUMMARY (" << num_roots << " ROOTS, BATCH " << target_batch << ")\n";
  std::cout << "================================================================================\n";
  std::cout << std::left << std::setw(25) << "Configuration"
            << std::setw(15) << "Time (s)"
            << std::setw(18) << "Roots/sec"
            << std::setw(15) << "Overrides"
            << "Speedup vs Baseline\n";
  std::cout << "--------------------------------------------------------------------------------\n";

  double base_rps = results[0].roots_per_sec;
  const BenchmarkResult* best = &results[0];

  for (const auto& r : results) {
    double speedup = r.roots_per_sec / base_rps;
    std::cout << std::left << std::setw(25) << r.name
              << std::fixed << std::setprecision(3) << std::setw(15) << r.elapsed_seconds
              << std::fixed << std::setprecision(2) << std::setw(18) << r.roots_per_sec
              << std::setw(15) << r.overrides
              << std::fixed << std::setprecision(2) << speedup << "x\n";
    if (r.roots_per_sec > best->roots_per_sec) {
      best = &r;
    }
  }
  std::cout << "================================================================================\n";
  std::cout << "WINNER: " << best->name << " with " << std::fixed << std::setprecision(2)
            << best->roots_per_sec << " roots/sec (" << best->threads << " threads / "
            << best->coordinators << " coordinators)\n";
  std::cout << "================================================================================\n";

  return 0;
}
