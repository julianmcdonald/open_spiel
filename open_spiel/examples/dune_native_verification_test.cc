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

#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"
#include <torch/torch.h>

#include "dune_network.h"
#include "dune_evaluator.h"
#include "open_spiel/games/dune_imperium/dune_imperium.h"

using namespace open_spiel;

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

  const std::string path_5580 = "/home/warcr/projects/dune_drl/calibration_results_v2/pf_c_run2/s1_ctl_b/ppo_model_update_15828.pt";
  const std::string path_6215_boot = "/run/media/warcr/Storage/dune_drl_runtime/round7/market_information_20260910_195200/arm_a/boot_u15828.pt";
  const std::string path_6215_pilot_b = "/run/media/warcr/Storage/dune_drl_runtime/round7/market_information_20260910_195200/pilot_arm_b/ppo_model_update_1.pt";

  if (!std::filesystem::exists(path_5580) || !std::filesystem::exists(path_6215_boot) || !std::filesystem::exists(path_6215_pilot_b)) {
    std::cerr << "ERROR: One or more required models missing for native verification.\n";
    return 1;
  }

  std::cout << "[MODEL] Loading native C++ LibTorch modules...\n";
  auto model_5580 = std::make_shared<SharedDunePolicyValueNetImpl>(5580, 2048, 2391, 8);
  auto model_6215_boot = std::make_shared<SharedDunePolicyValueNetImpl>(6215, 2048, 2391, 8);
  auto model_6215_pilot_b = std::make_shared<SharedDunePolicyValueNetImpl>(6215, 2048, 2391, 8);

  {
    torch::serialize::InputArchive arch;
    arch.load_from(path_5580, device);
    model_5580->load(arch);
  }
  {
    torch::serialize::InputArchive arch;
    arch.load_from(path_6215_boot, device);
    model_6215_boot->load(arch);
  }
  {
    torch::serialize::InputArchive arch;
    arch.load_from(path_6215_pilot_b, device);
    model_6215_pilot_b->load(arch);
  }

  model_5580->to(device);
  model_6215_boot->to(device);
  model_6215_pilot_b->to(device);

  model_5580->eval();
  model_6215_boot->eval();
  model_6215_pilot_b->eval();

  std::mutex eval_mutex;
  std::shared_mutex sync_mutex;

  // Single evaluators
  DeterministicEvaluator eval_5580(model_5580, device, &eval_mutex, &sync_mutex);
  DeterministicEvaluator eval_6215_det(model_6215_boot, device, &eval_mutex, &sync_mutex);
  DeterministicEvaluator eval_pilot_b(model_6215_pilot_b, device, &eval_mutex, &sync_mutex);

  // Batched evaluator (max_batch_size=32, timeout=2ms)
  BatchedEvaluator eval_6215_batch(model_6215_boot, 32, 2, device, &sync_mutex);

  std::cout << "[GAME] Initializing Dune: Imperium game engine...\n";
  std::shared_ptr<const Game> game = LoadGame("dune_imperium");

  const int num_target_states = 100;
  int states_sampled = 0;
  int games_played = 0;

  double max_raw_logit_diff = 0.0;
  double max_legal_logit_diff = 0.0;
  double max_policy_diff = 0.0;
  double max_tv_distance = 0.0;
  double max_kl_divergence = 0.0;
  double max_value_diff = 0.0;
  double max_batch_logit_diff = 0.0;
  double max_batch_value_diff = 0.0;
  double max_pilot_b_market_divergence = 0.0;

  std::mt19937 rng(1337);

  while (states_sampled < num_target_states) {
    std::unique_ptr<State> state = game->NewInitialState();
    games_played++;

    while (!state->IsTerminal() && states_sampled < num_target_states) {
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

      // Construct 6215 observations
      std::vector<float> obs_6215_zeros(6215, 0.0f);
      std::memcpy(obs_6215_zeros.data(), obs_5580.data(), 5580 * sizeof(float));

      std::vector<float> obs_6215_market(6215, 0.0f);
      std::memcpy(obs_6215_market.data(), obs_5580.data(), 5580 * sizeof(float));
      const auto* dune_state = dynamic_cast<const dune_imperium::DuneImperiumState*>(state.get());
      assert(dune_state != nullptr);
      dune_state->WriteImperiumMarketAppendix(absl::MakeSpan(&obs_6215_market[5580], 635));

      // 1. Native C++ evaluation
      EvalResult res_5580 = eval_5580.Evaluate(obs_5580);
      EvalResult res_6215_det = eval_6215_det.Evaluate(obs_6215_zeros);
      EvalResult res_6215_batch = eval_6215_batch.Evaluate(obs_6215_zeros);
      EvalResult res_pilot_b_market = eval_pilot_b.Evaluate(obs_6215_market);
      EvalResult res_pilot_b_zeros = eval_pilot_b.Evaluate(obs_6215_zeros);

      // Verify single vs batched parity on model_6215
      double b_vdiff = std::abs(res_6215_det.value - res_6215_batch.value);
      if (b_vdiff > max_batch_value_diff) max_batch_value_diff = b_vdiff;
      for (size_t a = 0; a < res_6215_det.logits.size(); ++a) {
        double d = std::abs(res_6215_det.logits[a] - res_6215_batch.logits[a]);
        if (d > max_batch_logit_diff) max_batch_logit_diff = d;
      }

      // Check raw output parity between 5580 and 6215 zeros
      double vdiff = std::abs(res_6215_det.value - res_5580.value);
      if (vdiff > max_value_diff) max_value_diff = vdiff;
      for (size_t a = 0; a < res_5580.logits.size(); ++a) {
        double d = std::abs(res_6215_det.logits[a] - res_5580.logits[a]);
        if (d > max_raw_logit_diff) max_raw_logit_diff = d;
      }

      // Check Arm B treatment sensitivity (market cards vs zeros)
      for (Action a : legal_actions) {
        double d = std::abs(res_pilot_b_market.logits[a] - res_pilot_b_zeros.logits[a]);
        if (d > max_pilot_b_market_divergence) max_pilot_b_market_divergence = d;
      }

      // 2. Production Legal-Centering and Capping
      CenterAndCapLegalLogitsWithStats(res_5580.logits, legal_actions, 10.0f);
      CenterAndCapLegalLogitsWithStats(res_6215_det.logits, legal_actions, 10.0f);

      for (Action a : legal_actions) {
        double d = std::abs(res_6215_det.logits[a] - res_5580.logits[a]);
        if (d > max_legal_logit_diff) max_legal_logit_diff = d;
      }

      // 3. Compute Legal Softmax Distributions
      double max_l_5580 = -1e9;
      double max_l_6215 = -1e9;
      for (Action a : legal_actions) {
        if (res_5580.logits[a] > max_l_5580) max_l_5580 = res_5580.logits[a];
        if (res_6215_det.logits[a] > max_l_6215) max_l_6215 = res_6215_det.logits[a];
      }

      double sum_exp_5580 = 0.0;
      double sum_exp_6215 = 0.0;
      std::vector<double> pi_5580(2391, 0.0);
      std::vector<double> pi_6215(2391, 0.0);

      for (Action a : legal_actions) {
        pi_5580[a] = std::exp(res_5580.logits[a] - max_l_5580);
        sum_exp_5580 += pi_5580[a];
        pi_6215[a] = std::exp(res_6215_det.logits[a] - max_l_6215);
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

      // Verify all illegal actions are strictly 0.0
      for (size_t a = 0; a < 2391; ++a) {
        bool is_legal = (std::find(legal_actions.begin(), legal_actions.end(), a) != legal_actions.end());
        if (!is_legal) {
          assert(pi_5580[a] == 0.0);
          assert(pi_6215[a] == 0.0);
        }
      }

      // 4. Compute safe legal KL and TV distance
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

      assert(!std::isnan(kl));
      assert(!std::isinf(kl));
      if (kl > max_kl_divergence) max_kl_divergence = kl;
      if (tv > max_tv_distance) max_tv_distance = tv;

      states_sampled++;

      // Advance game
      std::uniform_int_distribution<size_t> act_dist(0, legal_actions.size() - 1);
      state->ApplyAction(legal_actions[act_dist(rng)]);
    }
  }

  std::cout << "\n[RESULTS] Sampled " << states_sampled << " decision states across " << games_played << " games:\n";
  std::cout << "  Max Raw Logit Diff (6215 zeros vs 5580):       " << std::scientific << max_raw_logit_diff << "\n";
  std::cout << "  Max Legal Logit Diff (after centering & cap):  " << std::scientific << max_legal_logit_diff << "\n";
  std::cout << "  Max Legal Policy Diff (Softmax):               " << std::scientific << max_policy_diff << "\n";
  std::cout << "  Max Total Variation Distance:                  " << std::scientific << max_tv_distance << "\n";
  std::cout << "  Max Legal KL Divergence:                       " << std::scientific << max_kl_divergence << "\n";
  std::cout << "  Max Value Difference (Tanh head):              " << std::scientific << max_value_diff << "\n";
  std::cout << "  Max Single vs Batched Logit Diff:              " << std::scientific << max_batch_logit_diff << "\n";
  std::cout << "  Max Single vs Batched Value Diff:              " << std::scientific << max_batch_value_diff << "\n";
  std::cout << "  Max Arm B Market Sensitivity Logit Divergence: " << std::fixed << std::setprecision(6) << max_pilot_b_market_divergence << "\n";

  // Assertions: 6215 zeros vs 5580 must be EXACT bitwise match
  assert(max_legal_logit_diff == 0.0);
  assert(max_policy_diff == 0.0);
  assert(max_tv_distance == 0.0);
  assert(max_kl_divergence == 0.0);
  assert(max_value_diff == 0.0);
  assert(max_batch_logit_diff == 0.0);
  assert(max_batch_value_diff == 0.0);
  assert(max_pilot_b_market_divergence > 0.0001);

  std::cout << "\n[PASS] All Native C++ Parity and Distribution Checks Succeeded!\n";

  // Write results JSON
  std::string out_json_path = "/run/media/warcr/Storage/dune_drl_runtime/round7/market_information_20260910_195200/native_verification_results.json";
  std::ofstream out(out_json_path);
  if (out.is_open()) {
    out << "{\n"
        << "  \"verified\": true,\n"
        << "  \"states_sampled\": " << states_sampled << ",\n"
        << "  \"max_raw_logit_diff\": " << max_raw_logit_diff << ",\n"
        << "  \"max_legal_logit_diff\": " << max_legal_logit_diff << ",\n"
        << "  \"max_policy_diff\": " << max_policy_diff << ",\n"
        << "  \"max_tv_distance\": " << max_tv_distance << ",\n"
        << "  \"max_kl_divergence\": " << max_kl_divergence << ",\n"
        << "  \"max_value_diff\": " << max_value_diff << ",\n"
        << "  \"max_batch_logit_diff\": " << max_batch_logit_diff << ",\n"
        << "  \"max_batch_value_diff\": " << max_batch_value_diff << ",\n"
        << "  \"max_pilot_b_market_divergence\": " << max_pilot_b_market_divergence << ",\n"
        << "  \"status\": \"PASS\"\n"
        << "}\n";
    std::cout << "[SAVED] Verification results saved to: " << out_json_path << "\n";
  }

  return 0;
}
