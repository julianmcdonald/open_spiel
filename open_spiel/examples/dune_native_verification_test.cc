// Standalone native verification test for the Mature Market Information Study.
//
// Strictly verifies:
// 1. Native C++ forward / evaluator parity on isolated models (u15828 vs boot_u15828).
// 2. Exact finite legal-distribution checks with CenterAndCapLegalLogitsWithStats,
//    legal softmax, TV distance, and masked-safe KL divergence across 100 realistic states.
// 3. Single (DeterministicEvaluator) vs Batched (BatchedEvaluator) parity for collection/replay.
// 4. Treatment sensitivity in Arm B (pilot_b distinguishes market states).

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <memory>
#include <cmath>
#include <cassert>
#include <mutex>
#include <shared_mutex>
#include <filesystem>
#include <iomanip>
#include <thread>
#include <atomic>

#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"
#include <torch/torch.h>

#include "dune_network.h"
#include "dune_evaluator.h"
#include "open_spiel/games/dune_imperium/dune_imperium.h"

using namespace open_spiel;

struct SampledStateRecord {
  std::vector<float> obs_5580;
  std::vector<float> obs_6215_zeros;
  std::vector<float> obs_6215_market;
  std::vector<Action> legal_actions;
};

int main(int argc, char** argv) {
  std::cout << "================================================================================\n";
  std::cout << "Starting Native C++ Forward / Evaluator Parity & Distribution Verification\n";
  std::cout << "================================================================================\n";

  torch::Device device(torch::kCPU);
  if (torch::cuda::is_available()) {
    device = torch::Device(torch::kCUDA);
    std::cout << "[DEVICE] CUDA is available. Running native verification on GPU (" << device << ").\n";
  } else {
    std::cout << "[DEVICE] CUDA not available. Running native verification on CPU.\n";
  }

  // Strictly enforce production precision settings (FP32, TF32 disabled)
  if (device.is_cuda()) {
    at::globalContext().setAllowTF32CuBLAS(false);
    at::globalContext().setAllowTF32CuDNN(false);
    std::cout << "[PRECISION] TF32 cuBLAS and cuDNN strictly disabled for production FP32 parity.\n";
  }

  const std::string path_5580 = "/home/warcr/projects/dune_drl/calibration_results_v2/pf_c_run2/s1_ctl_b/ppo_model_update_15828.pt";
  const std::string path_6215_pilot_b = "/run/media/warcr/Storage/dune_drl_runtime/round7/market_information_20260910_195200/pilot_arm_b/ppo_model_update_1.pt";

  if (!std::filesystem::exists(path_5580) || !std::filesystem::exists(path_6215_pilot_b)) {
    std::cerr << "ERROR: One or more required models missing for native verification.\n";
    return 1;
  }

  std::cout << "[MODEL] Loading base 5,580 model (u15828)...\n";
  auto model_5580 = std::make_shared<SharedDunePolicyValueNetImpl>(5580, 2048, 2391, 8);
  {
    torch::serialize::InputArchive arch;
    arch.load_from(path_5580, device);
    model_5580->load(arch);
  }
  model_5580->to(device);
  model_5580->eval();

  assert(model_5580->input_layer->weight.size(0) == 2048);
  assert(model_5580->input_layer->weight.size(1) == 5580);
  std::cout << "[MODEL] Verified model_5580 shape: [" << model_5580->input_layer->weight.size(0)
            << ", " << model_5580->input_layer->weight.size(1) << "].\n";

  std::cout << "[MODEL] Performing production expansion migration to 6,215 model...\n";
  auto model_6215_boot = std::make_shared<SharedDunePolicyValueNetImpl>(6215, 2048, 2391, 8);
  model_6215_boot->to(device);

  {
    torch::NoGradGuard no_grad;
    // 1. Trunk input layer: copy inherited 5580 columns and strictly zero the 635 added appendix columns
    model_6215_boot->input_layer->weight.slice(1, 0, 5580).copy_(model_5580->input_layer->weight);
    model_6215_boot->input_layer->weight.slice(1, 5580, 6215).zero_();
    if (model_6215_boot->input_layer->bias.defined() && model_5580->input_layer->bias.defined()) {
      model_6215_boot->input_layer->bias.copy_(model_5580->input_layer->bias);
    }
    // 2. Residual blocks:
    for (size_t i = 0; i < model_6215_boot->res_blocks.size(); ++i) {
      auto source_params = model_5580->res_blocks[i]->parameters();
      auto target_params = model_6215_boot->res_blocks[i]->parameters();
      for (size_t j = 0; j < source_params.size(); ++j) {
        target_params[j].copy_(source_params[j]);
      }
      auto source_buffers = model_5580->res_blocks[i]->buffers();
      auto target_buffers = model_6215_boot->res_blocks[i]->buffers();
      for (size_t j = 0; j < source_buffers.size(); ++j) {
        target_buffers[j].copy_(source_buffers[j]);
      }
    }
    // 3. Policy head:
    model_6215_boot->policy_head->weight.copy_(model_5580->policy_head->weight);
    if (model_6215_boot->policy_head->bias.defined() && model_5580->policy_head->bias.defined()) {
      model_6215_boot->policy_head->bias.copy_(model_5580->policy_head->bias);
    }
    // 4. Value head:
    model_6215_boot->value_head->weight.copy_(model_5580->value_head->weight);
    if (model_6215_boot->value_head->bias.defined() && model_5580->value_head->bias.defined()) {
      model_6215_boot->value_head->bias.copy_(model_5580->value_head->bias);
    }
  }
  model_6215_boot->eval();

  // Assert structural shape and exact migration
  assert(model_6215_boot->input_layer->weight.size(0) == 2048);
  assert(model_6215_boot->input_layer->weight.size(1) == 6215);
  assert(torch::equal(model_6215_boot->input_layer->weight.slice(1, 0, 5580),
                      model_5580->input_layer->weight));
  assert(model_6215_boot->input_layer->weight.slice(1, 5580, 6215).abs().max().item<float>() == 0.0f);
  if (model_6215_boot->input_layer->bias.defined()) {
    assert(torch::equal(model_6215_boot->input_layer->bias, model_5580->input_layer->bias));
  }
  assert(torch::equal(model_6215_boot->policy_head->weight, model_5580->policy_head->weight));
  assert(torch::equal(model_6215_boot->value_head->weight, model_5580->value_head->weight));
  std::cout << "[MODEL] Verified model_6215_boot retained 6,215 input width and exactly zero added columns.\n";

  std::cout << "[MODEL] Loading native Arm B pilot model...\n";
  auto model_6215_pilot_b = std::make_shared<SharedDunePolicyValueNetImpl>(6215, 2048, 2391, 8);
  {
    torch::serialize::InputArchive arch;
    arch.load_from(path_6215_pilot_b, device);
    model_6215_pilot_b->load(arch);
  }
  model_6215_pilot_b->to(device);
  model_6215_pilot_b->eval();

  assert(model_6215_pilot_b->input_layer->weight.size(0) == 2048);
  assert(model_6215_pilot_b->input_layer->weight.size(1) == 6215);
  assert(model_6215_pilot_b->input_layer->weight.slice(1, 5580, 6215).abs().max().item<float>() > 0.0f);
  std::cout << "[MODEL] Verified model_6215_pilot_b has 6,215 input width and non-zero learned treatment weights.\n";

  std::mutex eval_mutex;
  std::shared_mutex sync_mutex;

  // Single deterministic evaluators (matching production FP32: rollout_amp=false)
  DeterministicEvaluator eval_5580(model_5580, device, &eval_mutex, &sync_mutex, nullptr, /*rollout_amp=*/false);
  DeterministicEvaluator eval_6215_det(model_6215_boot, device, &eval_mutex, &sync_mutex, nullptr, /*rollout_amp=*/false);
  DeterministicEvaluator eval_pilot_b(model_6215_pilot_b, device, &eval_mutex, &sync_mutex, nullptr, /*rollout_amp=*/false);
  assert(!eval_5580.RolloutAmpForTesting());
  assert(!eval_6215_det.RolloutAmpForTesting());
  assert(!eval_pilot_b.RolloutAmpForTesting());

  // Batched evaluator configured with production parameters:
  // target_batch_size=32, timeout_ms=2, FP32 (rollout_amp=false), TF32 disabled (allow_tf32=false)
  BatchedEvaluator eval_6215_batch(
      model_6215_boot,
      /*target_batch_size=*/32,
      /*timeout_ms=*/2,
      device,
      &sync_mutex,
      /*logit_cap=*/0.0f,
      /*device_synchronize=*/true,
      /*high_priority_stream=*/false,
      /*emit_batch_membership=*/false,
      /*rollout_amp=*/false,
      /*allow_tf32=*/false);

  assert(!eval_6215_batch.RolloutAmpForTesting());
  assert(!eval_6215_batch.AllowTf32ForTesting());
  eval_6215_batch.EnableBatcherTelemetry();
  std::cout << "[EVALUATOR] BatchedEvaluator initialized: target_batch=32, timeout=2ms, FP32, no TF32, telemetry enabled.\n";

  std::cout << "[GAME] Initializing Dune: Imperium game engine...\n";
  std::shared_ptr<const Game> game = LoadGame("dune_imperium");

  const int num_target_states = 100;
  int games_played = 0;
  std::vector<SampledStateRecord> sampled_states;
  sampled_states.reserve(num_target_states);

  std::mt19937 rng(1337);

  while (static_cast<int>(sampled_states.size()) < num_target_states) {
    std::unique_ptr<State> state = game->NewInitialState();
    games_played++;

    while (!state->IsTerminal() && static_cast<int>(sampled_states.size()) < num_target_states) {
      if (state->IsChanceNode()) {
        auto outcomes = state->ChanceOutcomes();
        std::uniform_int_distribution<size_t> dist(0, outcomes.size() - 1);
        state->ApplyAction(outcomes[dist(rng)].first);
        continue;
      }

      Player player = state->CurrentPlayer();
      std::vector<float> obs_5580 = state->InformationStateTensor(player);
      std::vector<Action> legal_actions = state->LegalActions();

      if (legal_actions.empty()) {
        continue;
      }

      SampledStateRecord record;
      record.obs_5580 = obs_5580;
      record.legal_actions = legal_actions;

      record.obs_6215_zeros.assign(6215, 0.0f);
      std::memcpy(record.obs_6215_zeros.data(), obs_5580.data(), 5580 * sizeof(float));

      record.obs_6215_market.assign(6215, 0.0f);
      std::memcpy(record.obs_6215_market.data(), obs_5580.data(), 5580 * sizeof(float));
      const auto* dune_state = dynamic_cast<const dune_imperium::DuneImperiumState*>(state.get());
      assert(dune_state != nullptr);
      dune_state->WriteImperiumMarketAppendix(absl::MakeSpan(&record.obs_6215_market[5580], 635));

      sampled_states.push_back(std::move(record));

      // Advance game
      std::uniform_int_distribution<size_t> act_dist(0, legal_actions.size() - 1);
      state->ApplyAction(legal_actions[act_dist(rng)]);
    }
  }

  std::cout << "[SAMPLE] Sampled " << sampled_states.size() << " realistic decision states across "
            << games_played << " games.\n";

  // Evaluate single deterministic evaluators
  std::vector<EvalResult> res_5580_list(num_target_states);
  std::vector<EvalResult> res_6215_det_list(num_target_states);
  std::vector<EvalResult> res_pilot_b_market_list(num_target_states);
  std::vector<EvalResult> res_pilot_b_zeros_list(num_target_states);

  for (size_t i = 0; i < sampled_states.size(); ++i) {
    res_5580_list[i] = eval_5580.Evaluate(sampled_states[i].obs_5580);
    res_6215_det_list[i] = eval_6215_det.Evaluate(sampled_states[i].obs_6215_zeros);
    res_pilot_b_market_list[i] = eval_pilot_b.Evaluate(sampled_states[i].obs_6215_market);
    res_pilot_b_zeros_list[i] = eval_pilot_b.Evaluate(sampled_states[i].obs_6215_zeros);
  }

  // Multi-threaded concurrent batching evaluation across 16 worker threads
  std::cout << "[CONCURRENCY] Launching 16 concurrent workers to exercise BatchedEvaluator...\n";
  const int num_workers = 16;
  std::vector<EvalResult> res_6215_batch_list(num_target_states);
  std::vector<std::thread> workers;
  workers.reserve(num_workers);

  std::atomic<bool> start_gate{false};
  std::atomic<size_t> next_state_idx{0};

  for (int w = 0; w < num_workers; ++w) {
    workers.emplace_back([&]() {
      while (!start_gate.load(std::memory_order_acquire)) {
        std::this_thread::yield();
      }
      while (true) {
        size_t idx = next_state_idx.fetch_add(1, std::memory_order_relaxed);
        if (idx >= sampled_states.size()) break;
        res_6215_batch_list[idx] = eval_6215_batch.Evaluate(sampled_states[idx].obs_6215_zeros);
      }
    });
  }

  // Release workers simultaneously to trigger true batching
  start_gate.store(true, std::memory_order_release);

  for (auto& w : workers) {
    w.join();
  }

  auto batch_stats = eval_6215_batch.GetStats();
  auto batch_timing = eval_6215_batch.GetBatcherTelemetry();

  std::cout << "[BATCH TELEMETRY] Evaluator Requests: " << batch_stats.requests
            << " | Physical Batches: " << batch_stats.batches
            << " | Max Batch Size: " << batch_stats.max_batch_size
            << " | Mean Batch Size: " << std::fixed << std::setprecision(2) << batch_stats.avg_batch_size
            << " | Device Timed Batches: " << batch_timing.device_timed_batches << "\n";

  // Assert true concurrent batching occurred
  assert(batch_stats.requests == static_cast<uint64_t>(num_target_states));
  assert(batch_stats.max_batch_size > 1);
  assert(batch_stats.avg_batch_size > 1.0);
  if (device.is_cuda()) {
    assert(batch_timing.device_timed_batches > 0);
  }

  // Distribution and Parity Metrics
  double max_raw_logit_diff = 0.0;
  double max_legal_logit_diff = 0.0;
  double max_policy_diff = 0.0;
  double max_tv_distance = 0.0;
  double max_kl_divergence = 0.0;
  double max_value_diff = 0.0;
  double max_batch_logit_diff = 0.0;
  double max_batch_value_diff = 0.0;
  double max_pilot_b_market_divergence = 0.0;

  for (size_t s = 0; s < sampled_states.size(); ++s) {
    const auto& legal_actions = sampled_states[s].legal_actions;
    const auto& res_5580 = res_5580_list[s];
    const auto& res_6215_det = res_6215_det_list[s];
    const auto& res_6215_batch = res_6215_batch_list[s];
    const auto& res_pilot_b_market = res_pilot_b_market_list[s];
    const auto& res_pilot_b_zeros = res_pilot_b_zeros_list[s];

    // 1. Single vs Batched Parity on model_6215_boot
    double b_vdiff = std::abs(res_6215_det.value - res_6215_batch.value);
    if (b_vdiff > max_batch_value_diff) max_batch_value_diff = b_vdiff;
    for (size_t a = 0; a < res_6215_det.logits.size(); ++a) {
      double d = std::abs(res_6215_det.logits[a] - res_6215_batch.logits[a]);
      if (d > max_batch_logit_diff) max_batch_logit_diff = d;
    }

    // 2. Parity between 5580 and 6215 zeros (raw outputs)
    double vdiff = std::abs(res_6215_det.value - res_5580.value);
    if (vdiff > max_value_diff) max_value_diff = vdiff;
    for (size_t a = 0; a < res_5580.logits.size(); ++a) {
      double d = std::abs(res_6215_det.logits[a] - res_5580.logits[a]);
      if (d > max_raw_logit_diff) max_raw_logit_diff = d;
    }

    // 3. Arm B Treatment Sensitivity (market appendix vs zeros)
    for (Action a : legal_actions) {
      double d = std::abs(res_pilot_b_market.logits[a] - res_pilot_b_zeros.logits[a]);
      if (d > max_pilot_b_market_divergence) max_pilot_b_market_divergence = d;
    }

    // 4. Production Legal-Centering and Capping
    std::vector<float> capped_logits_5580 = res_5580.logits;
    std::vector<float> capped_logits_6215 = res_6215_det.logits;
    CenterAndCapLegalLogitsWithStats(capped_logits_5580, legal_actions, 10.0f);
    CenterAndCapLegalLogitsWithStats(capped_logits_6215, legal_actions, 10.0f);

    for (Action a : legal_actions) {
      double d = std::abs(capped_logits_6215[a] - capped_logits_5580[a]);
      if (d > max_legal_logit_diff) max_legal_logit_diff = d;
    }

    // 5. Legal Softmax Policy Distributions
    double max_l_5580 = -1e9;
    double max_l_6215 = -1e9;
    for (Action a : legal_actions) {
      if (capped_logits_5580[a] > max_l_5580) max_l_5580 = capped_logits_5580[a];
      if (capped_logits_6215[a] > max_l_6215) max_l_6215 = capped_logits_6215[a];
    }

    double sum_exp_5580 = 0.0;
    double sum_exp_6215 = 0.0;
    std::vector<double> pi_5580(2391, 0.0);
    std::vector<double> pi_6215(2391, 0.0);

    for (Action a : legal_actions) {
      pi_5580[a] = std::exp(capped_logits_5580[a] - max_l_5580);
      sum_exp_5580 += pi_5580[a];
      pi_6215[a] = std::exp(capped_logits_6215[a] - max_l_6215);
      sum_exp_6215 += pi_6215[a];
    }

    double check_sum_5580 = 0.0;
    double check_sum_6215 = 0.0;
    for (Action a : legal_actions) {
      pi_5580[a] /= sum_exp_5580;
      pi_6215[a] /= sum_exp_6215;

      assert(std::isfinite(pi_5580[a]));
      assert(std::isfinite(pi_6215[a]));
      assert(pi_5580[a] > 0.0);
      assert(pi_6215[a] > 0.0);

      check_sum_5580 += pi_5580[a];
      check_sum_6215 += pi_6215[a];
    }

    assert(std::abs(check_sum_5580 - 1.0) < 1e-5);
    assert(std::abs(check_sum_6215 - 1.0) < 1e-5);

    // Verify all illegal actions are strictly zero
    for (size_t a = 0; a < 2391; ++a) {
      bool is_legal = (std::find(legal_actions.begin(), legal_actions.end(), a) != legal_actions.end());
      if (!is_legal) {
        assert(pi_5580[a] == 0.0);
        assert(pi_6215[a] == 0.0);
      }
    }

    // 6. Masked-safe Legal KL and Total Variation Distance
    double kl = 0.0;
    double tv = 0.0;
    for (Action a : legal_actions) {
      double p = pi_6215[a];
      double q = pi_5580[a];
      double diff = std::abs(p - q);
      if (diff > max_policy_diff) max_policy_diff = diff;
      tv += 0.5 * diff;
      if (p > 0.0 && q > 0.0) {
        kl += p * std::log(p / q);
      }
    }

    assert(std::isfinite(kl));
    assert(std::isfinite(tv));
    assert(kl >= -1e-9);
    if (kl < 0.0) kl = 0.0; // clamp minor numerical negative float error

    if (kl > max_kl_divergence) max_kl_divergence = kl;
    if (tv > max_tv_distance) max_tv_distance = tv;
  }

  std::cout << "\n[RESULTS] Sampled " << sampled_states.size() << " decision states across " << games_played << " games:\n";
  std::cout << "  Max Raw Logit Diff (6215 zeros vs 5580):       " << std::scientific << max_raw_logit_diff << "\n";
  std::cout << "  Max Legal Logit Diff (after centering & cap):  " << std::scientific << max_legal_logit_diff << "\n";
  std::cout << "  Max Legal Policy Diff (Softmax):               " << std::scientific << max_policy_diff << "\n";
  std::cout << "  Max Total Variation Distance:                  " << std::scientific << max_tv_distance << "\n";
  std::cout << "  Max Legal KL Divergence:                       " << std::scientific << max_kl_divergence << "\n";
  std::cout << "  Max Value Difference (Tanh head):              " << std::scientific << max_value_diff << "\n";
  std::cout << "  Max Single vs Batched Logit Diff:              " << std::scientific << max_batch_logit_diff << "\n";
  std::cout << "  Max Single vs Batched Value Diff:              " << std::scientific << max_batch_value_diff << "\n";
  std::cout << "  Max Arm B Market Sensitivity Logit Divergence: " << std::fixed << std::setprecision(6) << max_pilot_b_market_divergence << "\n";

  // Pre-registered meaningful numerical tolerance for FP32 GEMM across different matrix dimensions
  const double kParityTolerance = 1e-4;

  assert(std::isfinite(max_raw_logit_diff));
  assert(max_raw_logit_diff < kParityTolerance);

  assert(std::isfinite(max_legal_logit_diff));
  assert(max_legal_logit_diff < kParityTolerance);

  assert(std::isfinite(max_policy_diff));
  assert(max_policy_diff < kParityTolerance);

  assert(std::isfinite(max_tv_distance));
  assert(max_tv_distance < kParityTolerance);

  assert(std::isfinite(max_kl_divergence));
  assert(max_kl_divergence < kParityTolerance);

  assert(std::isfinite(max_value_diff));
  assert(max_value_diff < kParityTolerance);

  assert(std::isfinite(max_batch_logit_diff));
  assert(max_batch_logit_diff < kParityTolerance);

  assert(std::isfinite(max_batch_value_diff));
  assert(max_batch_value_diff < kParityTolerance);

  // Treatment sensitivity assert (treatment features produce substantial divergence > 1e-3)
  assert(max_pilot_b_market_divergence > 1e-3);

  std::cout << "\n[PASS] All Native C++ Parity and Distribution Checks Succeeded within tolerance ("
            << kParityTolerance << ")!\n";

  // Write results JSON
  std::string out_json_path = "/run/media/warcr/Storage/dune_drl_runtime/round7/market_information_20260910_195200/native_verification_results.json";
  std::ofstream out(out_json_path);
  if (out.is_open()) {
    out << "{\n"
        << "  \"verified\": true,\n"
        << "  \"states_sampled\": " << sampled_states.size() << ",\n"
        << "  \"tolerance\": " << kParityTolerance << ",\n"
        << "  \"max_raw_logit_diff\": " << max_raw_logit_diff << ",\n"
        << "  \"max_legal_logit_diff\": " << max_legal_logit_diff << ",\n"
        << "  \"max_policy_diff\": " << max_policy_diff << ",\n"
        << "  \"max_tv_distance\": " << max_tv_distance << ",\n"
        << "  \"max_kl_divergence\": " << max_kl_divergence << ",\n"
        << "  \"max_value_diff\": " << max_value_diff << ",\n"
        << "  \"max_batch_logit_diff\": " << max_batch_logit_diff << ",\n"
        << "  \"max_batch_value_diff\": " << max_batch_value_diff << ",\n"
        << "  \"max_pilot_b_market_divergence\": " << max_pilot_b_market_divergence << ",\n"
        << "  \"batch_requests\": " << batch_stats.requests << ",\n"
        << "  \"batch_physical_batches\": " << batch_stats.batches << ",\n"
        << "  \"batch_max_batch_size\": " << batch_stats.max_batch_size << ",\n"
        << "  \"batch_mean_batch_size\": " << batch_stats.avg_batch_size << ",\n"
        << "  \"batch_device_timed_batches\": " << batch_timing.device_timed_batches << ",\n"
        << "  \"rollout_amp\": false,\n"
        << "  \"allow_tf32\": false,\n"
        << "  \"status\": \"PASS\"\n"
        << "}\n";
    std::cout << "[SAVED] Verification results saved to: " << out_json_path << "\n";
  }

  return 0;
}
