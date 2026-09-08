// Numerical Parity Audit: Fused DuneNNEvaluator vs Unfused SplitPolicyValueEvaluator.
// Compares:
//   1. Per-state Prior() and Evaluate() values on identical states.
//   2. MCTS search visit distributions and move selection under identical seeds.

#include <cmath>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"
#include "open_spiel/games/dune_imperium/dune_imperium.h"
#include <torch/torch.h>

#include "dune_evaluator.h"
#include "dune_network.h"
#include "dune_puct_is_mcts.h"
#include "dune_split_evaluator.h"

namespace open_spiel {

struct ParityDiffStats {
  double max_prior_diff = 0.0;
  double max_val_diff = 0.0;
  int prior_argmax_mismatches = 0;
  int total_states_checked = 0;
};

ParityDiffStats CheckParityOnStates(
    const std::string& label,
    const std::shared_ptr<algorithms::Evaluator>& fused,
    const std::shared_ptr<algorithms::Evaluator>& split,
    const std::vector<std::unique_ptr<State>>& states) {
  ParityDiffStats stats;
  for (const auto& state : states) {
    if (state->IsTerminal() || state->IsChanceNode()) continue;

    stats.total_states_checked++;
    auto [fused_prior, fused_vals] = fused->PriorAndEvaluate(*state);
    auto [split_prior, split_vals] = split->PriorAndEvaluate(*state);

    // Value comparison
    for (size_t p = 0; p < fused_vals.size(); ++p) {
      double diff = std::abs(fused_vals[p] - split_vals[p]);
      if (diff > stats.max_val_diff) stats.max_val_diff = diff;
    }

    // Prior comparison
    Action fused_best = -1;
    double fused_best_p = -1.0;
    Action split_best = -1;
    double split_best_p = -1.0;

    for (size_t i = 0; i < fused_prior.size(); ++i) {
      double diff = std::abs(fused_prior[i].second - split_prior[i].second);
      if (diff > stats.max_prior_diff) stats.max_prior_diff = diff;

      if (fused_prior[i].second > fused_best_p) {
        fused_best_p = fused_prior[i].second;
        fused_best = fused_prior[i].first;
      }
      if (split_prior[i].second > split_best_p) {
        split_best_p = split_prior[i].second;
        split_best = split_prior[i].first;
      }
    }

    if (fused_best != split_best) {
      stats.prior_argmax_mismatches++;
    }
  }

  std::cout << "--- " << label << " Parity Audit (" << stats.total_states_checked << " states) ---\n";
  std::cout << "  Max Value Absolute Diff:  " << std::scientific << std::setprecision(6) << stats.max_val_diff << "\n";
  std::cout << "  Max Prior Absolute Diff:  " << std::scientific << std::setprecision(6) << stats.max_prior_diff << "\n";
  std::cout << "  Prior Argmax Mismatches:  " << stats.prior_argmax_mismatches << " / " << stats.total_states_checked << "\n";
  std::cout << std::defaultfloat;
  return stats;
}

} // namespace open_spiel

int main(int argc, char** argv) {
  using namespace open_spiel;

  std::shared_ptr<const Game> game = LoadGame("dune_imperium");
  const int64_t obs_size = game->InformationStateTensorSize();
  const int64_t action_size = game->NumDistinctActions();
  const int hidden_dim = 2048;
  const int num_blocks = 8;
  const float candidate_logit_cap = 10.0f;

  torch::Device device = torch::cuda::is_available() ? torch::Device(torch::kCUDA) : torch::Device(torch::kCPU);
  std::cout << "Running parity audit on device: " << device << "\n";

  const std::string parent_ckpt = "/run/media/warcr/Storage/dune_drl_runtime/round7/placement_u200/ppo_model_update_200.pt";
  const std::string child_ckpt = "/run/media/warcr/Storage/dune_drl_runtime/round7/search_transfer_pilot_20260905_203537/search_child.pt";

  // Load parent
  auto parent_net = std::make_shared<SharedDunePolicyValueNetImpl>(obs_size, hidden_dim, action_size, num_blocks, false);
  parent_net->eval();
  {
    torch::serialize::InputArchive arc;
    arc.load_from(parent_ckpt, device);
    parent_net->load(arc);
    parent_net->to(device);
  }

  // Load child
  auto child_net = std::make_shared<SharedDunePolicyValueNetImpl>(obs_size, hidden_dim, action_size, num_blocks, false);
  child_net->eval();
  {
    torch::serialize::InputArchive arc;
    arc.load_from(child_ckpt, device);
    child_net->load(arc);
    child_net->to(device);
  }

  // Generate 100 diverse game states
  std::vector<std::unique_ptr<State>> sample_states;
  std::mt19937 rng(42);
  for (int g = 0; g < 5; ++g) {
    std::unique_ptr<State> state = game->NewInitialState();
    while (!state->IsTerminal() && sample_states.size() < 100) {
      if (state->IsChanceNode()) {
        auto outcomes = state->ChanceOutcomes();
        state->ApplyAction(outcomes.front().first);
      } else {
        sample_states.push_back(state->Clone());
        auto legals = state->LegalActions();
        std::uniform_int_distribution<int> dist(0, legals.size() - 1);
        state->ApplyAction(legals[dist(rng)]);
      }
    }
  }

  // Create evaluators
  auto fused_parent = std::make_shared<DuneNNEvaluator>(parent_net, device, candidate_logit_cap);
  auto split_parent = std::make_shared<SplitPolicyValueEvaluator>(
      std::make_shared<DuneNNEvaluator>(parent_net, device, candidate_logit_cap),
      std::make_shared<DuneNNEvaluator>(parent_net, device, candidate_logit_cap));

  auto fused_child = std::make_shared<DuneNNEvaluator>(child_net, device, candidate_logit_cap);
  auto split_child = std::make_shared<SplitPolicyValueEvaluator>(
      std::make_shared<DuneNNEvaluator>(child_net, device, candidate_logit_cap),
      std::make_shared<DuneNNEvaluator>(child_net, device, candidate_logit_cap));

  // Run Per-State Checks
  CheckParityOnStates("Parent/Parent", fused_parent, split_parent, sample_states);
  CheckParityOnStates("Child/Child", fused_child, split_child, sample_states);

  // Run MCTS Search Parity on a decision state
  std::cout << "\n--- MCTS 200-Simulation Search Parity Check ---\n";
  const State* test_state = nullptr;
  for (const auto& s : sample_states) {
    if (s->LegalActions().size() >= 3) {
      test_state = s.get();
      break;
    }
  }
  SPIEL_CHECK_TRUE(test_state != nullptr);

  DuneSearchConfig cfg;
  cfg.max_simulations = 200;
  cfg.puct_c = 0.30;
  cfg.seed = 2026090501ULL;
  cfg.utility_divisor = 4.0;
  cfg.temperature = 0.0;

  DunePUCTISMCTSBot fused_bot(cfg, fused_parent);
  DunePUCTISMCTSBot split_bot(cfg, split_parent);

  Action fused_action = fused_bot.Step(*test_state);
  DuneSearchResult fused_res = fused_bot.GetLastSearchResult();

  Action split_action = split_bot.Step(*test_state);
  DuneSearchResult split_res = split_bot.GetLastSearchResult();

  std::cout << "Fused Bot Sims: " << fused_res.simulations_completed << " | Action: " << fused_action << "\n";
  std::cout << "Split Bot Sims: " << split_res.simulations_completed << " | Action: " << split_action << "\n";

  int matching_visits = 0;
  int total_actions = 0;
  for (const auto& [act, prob] : fused_res.policy) {
    total_actions++;
    for (const auto& [s_act, s_prob] : split_res.policy) {
      if (s_act == act && std::abs(s_prob - prob) < 1e-4) {
        matching_visits++;
        break;
      }
    }
  }
  std::cout << "Visit distribution exact matches: " << matching_visits << " / " << total_actions << "\n";
  std::cout << "Selected Action Match: " << (fused_action == split_action ? "YES" : "NO") << "\n";

  return 0;
}
