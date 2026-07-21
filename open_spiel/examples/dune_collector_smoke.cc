// Phase 18B online-collector smoke harness (training-side only).
//
// Loads the real frozen search model, constructs an OnlineSearchCollector, runs
// ONE CollectUpdate, and dumps per-role acceptance/KL telemetry plus the
// Swordmaster endowment-curriculum grant/organic counters. Used to (a) measure
// noise-ON per-role acceptance and KL for the decider, and (b) exercise the
// grant path under the real model. It restates NO frozen defaults: everything
// except the handful of flags below comes from OnlineSearchConfig's defaults
// (which hold the pilot values). Model/evaluator construction mirrors
// dune_search_benchmark.cc. Exit 0 on success.

#include <cstdint>
#include <exception>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"
#include "open_spiel/utils/json.h"
#include "open_spiel/abseil-cpp/absl/flags/flag.h"
#include "open_spiel/abseil-cpp/absl/flags/parse.h"
#include "open_spiel/abseil-cpp/absl/strings/str_format.h"
#include <torch/torch.h>

#include "open_spiel/examples/dune_online_search_collector.h"
#include "open_spiel/algorithms/mcts.h"  // algorithms::Evaluator
#include "dune_network.h"                 // SharedDunePolicyValueNetImpl
#include "dune_evaluator.h"               // DuneNNEvaluator (: algorithms::Evaluator)
#include "dune_sha256.h"                  // ComputeFileSHA256

// The branch_a_frozen checkpoint is a 2048-wide / 8-block baseline-critic net
// (hidden_dim/num_blocks are fixed for this checkpoint — see the benchmark
// defaults and scripts/eval/run_task10_calibration.py). Not exposed as flags:
// this smoke targets that one checkpoint family.
constexpr int64_t kHiddenDim = 2048;
constexpr int64_t kNumBlocks = 8;

ABSL_FLAG(std::string, model_checkpoint,
          "artifacts/branch_a_frozen/branch_a_seed11_model_update_2450.pt",
          "Path to the frozen search model checkpoint.");
ABSL_FLAG(int, games, 8,
          "Auxiliary games for the single CollectUpdate (multiple of 4).");
ABSL_FLAG(int, seed_domain, 1801,
          "auxiliary_search_seed_domain (isolated collection stream).");
ABSL_FLAG(double, dirichlet_epsilon, 0.25,
          "Root Dirichlet noise weight; activates the KataGo package when > 0.");
ABSL_FLAG(double, swordmaster_grant_fraction, 0.0,
          "Fraction of games granting the searched seat a free Swordmaster.");
ABSL_FLAG(int, swordmaster_grant_round, 2,
          "Round at which a selected game's grant fires.");
ABSL_FLAG(std::string, out_json, "", "Output path for the stats JSON (required).");

namespace {

open_spiel::json::Object RoleObject(const open_spiel::PerRoleSearchStats& r) {
  const double denom = r.searches > 0 ? static_cast<double>(r.searches) : 0.0;
  open_spiel::json::Object o;
  o["roots_seen"] = static_cast<int64_t>(r.roots_seen);
  o["searches"] = static_cast<int64_t>(r.searches);
  o["accepted"] = static_cast<int64_t>(r.accepted);
  o["acceptance_rate"] = denom > 0 ? static_cast<double>(r.accepted) / denom : 0.0;
  o["mean_kl"] = denom > 0 ? r.sum_kl / denom : 0.0;
  o["override_rate"] =
      denom > 0 ? static_cast<double>(r.prior_argmax_overrides) / denom : 0.0;
  return o;
}

// Echo of every OnlineSearchConfig field actually used (audit the effective
// config; do not trust the flags alone). Mirrors the struct field-for-field.
open_spiel::json::Object ConfigEcho(const open_spiel::OnlineSearchConfig& c) {
  open_spiel::json::Object o;
  o["auxiliary_games"] = static_cast<int64_t>(c.auxiliary_games);
  o["search_probability"] = c.search_probability;
  o["workers"] = static_cast<int64_t>(c.workers);
  o["max_simulations"] = static_cast<int64_t>(c.max_simulations);
  o["fixed_continuation_reserve"] =
      static_cast<int64_t>(c.fixed_continuation_reserve);
  o["purchase_combat_budget"] = static_cast<int64_t>(c.purchase_combat_budget);
  o["max_search_decision_depth"] =
      static_cast<int64_t>(c.max_search_decision_depth);
  o["puct_c"] = c.puct_c;
  o["use_opponent_model"] = c.use_opponent_model;
  o["simulated_opponent_temperature"] = c.simulated_opponent_temperature;
  o["root_prior_temperature"] = c.root_prior_temperature;
  o["utility_divisor"] = c.utility_divisor;
  o["nonlinear_value_head"] = c.nonlinear_value_head;
  o["dirichlet_epsilon"] = c.dirichlet_epsilon;
  o["dirichlet_alpha_total"] = c.dirichlet_alpha_total;
  o["forced_playouts_k"] = c.forced_playouts_k;
  o["root_noise_fpu_zero"] = c.root_noise_fpu_zero;
  o["swordmaster_grant_fraction"] = c.swordmaster_grant_fraction;
  o["swordmaster_grant_round"] = static_cast<int64_t>(c.swordmaster_grant_round);
  o["non_search_temperature"] = c.non_search_temperature;
  o["min_coverage"] = static_cast<int64_t>(c.min_coverage);
  o["min_visits_per_action"] = static_cast<int64_t>(c.min_visits_per_action);
  o["min_prior_mass"] = c.min_prior_mass;
  o["accepted_action_temperature"] = c.accepted_action_temperature;
  o["search_loss_coef_target"] = c.search_loss_coef_target;
  o["search_loss_coef_warmup_update"] =
      static_cast<int64_t>(c.search_loss_coef_warmup_update);
  o["abort_grad_norm_ratio"] = c.abort_grad_norm_ratio;
  o["auxiliary_search_seed_domain"] =
      static_cast<int64_t>(c.auxiliary_search_seed_domain);
  o["next_auxiliary_episode_id"] =
      static_cast<int64_t>(c.next_auxiliary_episode_id);
  o["per_search_timeout_ms"] = c.per_search_timeout_ms;
  return o;
}

}  // namespace

int main(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);

  const std::string model_ckpt = absl::GetFlag(FLAGS_model_checkpoint);
  const std::string out_json = absl::GetFlag(FLAGS_out_json);
  const int games = absl::GetFlag(FLAGS_games);

  if (out_json.empty()) {
    std::cerr << "Error: --out_json is required.\n";
    return 1;
  }
  if (model_ckpt.empty()) {
    std::cerr << "Error: --model_checkpoint is required.\n";
    return 1;
  }
  SPIEL_CHECK_EQ(games % 4, 0);  // collector seat-balance guard (fail loud)

  std::shared_ptr<const open_spiel::Game> game =
      open_spiel::LoadGame("dune_imperium");
  const int64_t obs_size = game->InformationStateTensorSize();
  const int64_t action_size = game->NumDistinctActions();

  torch::Device device = torch::cuda::is_available() ? torch::Device(torch::kCUDA)
                                                     : torch::Device(torch::kCPU);
  std::cout << "Loading model from " << model_ckpt << " on device " << device
            << " (obs=" << obs_size << " actions=" << action_size
            << " hidden=" << kHiddenDim << " blocks=" << kNumBlocks << ")\n";

  auto model = std::make_shared<open_spiel::SharedDunePolicyValueNetImpl>(
      obs_size, kHiddenDim, action_size, kNumBlocks,
      /*nonlinear_value_head=*/false);  // baseline critic (collector invariant)
  model->eval();
  try {
    torch::serialize::InputArchive archive;
    archive.load_from(model_ckpt, device);
    model->load(archive);
  } catch (const c10::Error& e) {
    std::cerr << "Failed to load model weights:\n" << e.msg() << "\n";
    return 1;
  }
  model->to(device);

  std::shared_ptr<open_spiel::algorithms::Evaluator> evaluator =
      std::make_shared<open_spiel::DuneNNEvaluator>(model, device, 10.0f);

  // Provenance hash of the frozen snapshot (reuses the eval-side helper).
  std::string checkpoint_hash = "smoke";
  try {
    checkpoint_hash = open_spiel::ComputeFileSHA256(model_ckpt);
  } catch (const std::exception& e) {
    std::cerr << "Warning: SHA-256 of checkpoint failed (" << e.what()
              << "); using \"smoke\".\n";
  }

  // Only the flagged knobs deviate from OnlineSearchConfig's frozen defaults.
  open_spiel::OnlineSearchConfig config;
  config.auxiliary_games = games;
  config.auxiliary_search_seed_domain =
      static_cast<uint64_t>(absl::GetFlag(FLAGS_seed_domain));
  config.dirichlet_epsilon = absl::GetFlag(FLAGS_dirichlet_epsilon);
  config.swordmaster_grant_fraction =
      absl::GetFlag(FLAGS_swordmaster_grant_fraction);
  config.swordmaster_grant_round = absl::GetFlag(FLAGS_swordmaster_grant_round);

  open_spiel::OnlineSearchCollector collector(config, checkpoint_hash);
  std::vector<open_spiel::SearchTrainingExample> out;
  open_spiel::OnlineSearchCollectionStats stats;

  std::cout << "Running CollectUpdate: games=" << games
            << " seed_domain=" << config.auxiliary_search_seed_domain
            << " dirichlet_epsilon=" << config.dirichlet_epsilon
            << " grant_fraction=" << config.swordmaster_grant_fraction
            << " grant_round=" << config.swordmaster_grant_round << " ...\n";
  std::cout.flush();
  collector.CollectUpdate(/*update_id=*/1, game, evaluator, &out, &stats);

  const double wall_per_game =
      games > 0 ? stats.collection_wall_time_s / games : 0.0;

  // --- JSON out ---
  open_spiel::json::Object root;
  root["checkpoint"] = model_ckpt;
  root["checkpoint_hash"] = checkpoint_hash;
  root["device"] = device.is_cuda() ? std::string("cuda") : std::string("cpu");
  root["emitted_examples"] = static_cast<int64_t>(out.size());
  root["config"] = ConfigEcho(config);

  open_spiel::json::Object roles;
  roles["primary"] = RoleObject(stats.by_role[0]);
  roles["continuation"] = RoleObject(stats.by_role[1]);
  roles["purchase"] = RoleObject(stats.by_role[2]);
  root["by_role"] = roles;

  open_spiel::json::Object agg;
  agg["strategic_roots_seen"] = static_cast<int64_t>(stats.strategic_roots_seen);
  agg["searches_selected"] = static_cast<int64_t>(stats.searches_selected);
  agg["accepted_targets"] = static_cast<int64_t>(stats.accepted_targets);
  agg["rejected_incomplete"] = static_cast<int64_t>(stats.rejected_incomplete);
  agg["fallback_raw_policy"] = static_cast<int64_t>(stats.fallback_raw_policy);
  agg["timeouts"] = static_cast<int64_t>(stats.timeouts);
  agg["mean_simulations_completed"] = stats.mean_simulations_completed;
  agg["mean_covered_prior_mass"] = stats.mean_covered_prior_mass;
  agg["inference_calls"] = static_cast<int64_t>(stats.inference_calls);
  agg["collection_wall_time_s"] = stats.collection_wall_time_s;
  agg["wall_time_per_game"] = wall_per_game;
  agg["first_episode_id"] = static_cast<int64_t>(stats.first_episode_id);
  agg["next_episode_id"] = static_cast<int64_t>(stats.next_episode_id);
  root["aggregates"] = agg;

  open_spiel::json::Object sm;
  sm["swordmaster_granted_games"] =
      static_cast<int64_t>(stats.swordmaster_granted_games);
  sm["swordmaster_organic_games"] =
      static_cast<int64_t>(stats.swordmaster_organic_games);
  root["swordmaster"] = sm;

  std::ofstream f(out_json);
  if (!f) {
    std::cerr << "Error: could not open --out_json for writing: " << out_json
              << "\n";
    return 1;
  }
  f << open_spiel::json::ToString(root, /*wrap=*/true);
  f.close();

  // --- Human summary ---
  auto role_line = [](const char* name,
                      const open_spiel::PerRoleSearchStats& r) {
    const double denom = r.searches > 0 ? static_cast<double>(r.searches) : 1.0;
    std::cout << absl::StrFormat(
        "  %-13s roots=%-4d searches=%-4d accepted=%-4d "
        "accept_rate=%.3f mean_kl=%.3f override_rate=%.3f\n",
        name, r.roots_seen, r.searches, r.accepted,
        r.searches > 0 ? r.accepted / denom : 0.0,
        r.searches > 0 ? r.sum_kl / denom : 0.0,
        r.searches > 0 ? r.prior_argmax_overrides / denom : 0.0);
  };
  std::cout << "\n=== Collector smoke summary ===\n";
  std::cout << absl::StrFormat(
      "games=%d  wall=%.1fs  (%.1fs/game)  emitted=%d  inference_calls=%d\n",
      games, stats.collection_wall_time_s, wall_per_game,
      static_cast<int>(out.size()), static_cast<int>(stats.inference_calls));
  role_line("primary", stats.by_role[0]);
  role_line("continuation", stats.by_role[1]);
  role_line("purchase", stats.by_role[2]);
  std::cout << absl::StrFormat(
      "  aggregate    accepted=%d rejected=%d fallback=%d timeouts=%d "
      "mean_sims=%.1f mean_coverage=%.3f\n",
      stats.accepted_targets, stats.rejected_incomplete,
      stats.fallback_raw_policy, stats.timeouts,
      stats.mean_simulations_completed, stats.mean_covered_prior_mass);
  std::cout << absl::StrFormat(
      "  swordmaster  granted_games=%d organic_games=%d\n",
      static_cast<int>(stats.swordmaster_granted_games),
      static_cast<int>(stats.swordmaster_organic_games));
  std::cout << "wrote " << out_json << "\n";
  return 0;
}
