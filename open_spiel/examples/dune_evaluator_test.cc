#include <iostream>
#include <string>
#include <vector>
#include <memory>
#include <cmath>
#include <cassert>
#include <filesystem>

#include "open_spiel/spiel.h"
#include "open_spiel/games/dune_imperium/dune_imperium.h"
#include <torch/torch.h>

#include "dune_network.h"
#include "dune_evaluator.h"

using namespace open_spiel;

void AssertAlmostEqual(double a, double b, double tol = 1e-5) {
  if (std::abs(a - b) > tol) {
    std::cerr << "Assertion failed: " << a << " != " << b << " (diff: " << std::abs(a - b) << ")\n";
    std::abort();
  }
}

int main(int argc, char* argv[]) {
  std::string checkpoint_path = "";
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--checkpoint" && i + 1 < argc) {
      checkpoint_path = argv[++i];
    }
  }

  std::cout << "Loading dune_imperium game...\n";
  std::shared_ptr<const Game> game = LoadGame("dune_imperium");
  std::unique_ptr<State> state = game->NewInitialState();

  int64_t obs_size = game->InformationStateTensorShape()[0];
  int64_t action_size = 2391;
  int64_t hidden_dim = 2048;
  int num_blocks = 8;

  std::cout << "Initializing SharedDunePolicyValueNet with " << num_blocks << " blocks...\n";
  auto model = std::make_shared<SharedDunePolicyValueNetImpl>(obs_size, hidden_dim, action_size, num_blocks);
  model->eval();

  torch::Device device(torch::kCPU);
  if (torch::cuda::is_available()) {
    device = torch::Device(torch::kCUDA);
    std::cout << "CUDA is available. Using CUDA device.\n";
  } else {
    std::cout << "CUDA is not available. Using CPU.\n";
  }

  if (!checkpoint_path.empty()) {
    std::cout << "Loading checkpoint: " << checkpoint_path << " on " << device << "\n";
    try {
      torch::serialize::InputArchive archive;
      archive.load_from(checkpoint_path, device);
      model->load(archive);
      std::cout << "Checkpoint successfully loaded!\n";
    } catch (const c10::Error& e) {
      std::cerr << "Failed to load checkpoint on " << device << ": " << e.msg() << "\n";
      if (device.is_cuda()) {
        std::cerr << "Retrying load on CPU...\n";
        device = torch::Device(torch::kCPU);
        try {
          torch::serialize::InputArchive archive;
          archive.load_from(checkpoint_path, device);
          model->load(archive);
          std::cout << "Checkpoint successfully loaded on CPU!\n";
        } catch (const c10::Error& e_cpu) {
          std::cerr << "Failed to load checkpoint on CPU: " << e_cpu.msg() << "\n";
          return 1;
        }
      } else {
        return 1;
      }
    }
  }
  model->to(device);

  // Apply chance actions to get to first player decision state
  std::mt19937 rng(42);
  std::cout << "Rolling state forward through setup chance nodes...\n";
  while (state->IsChanceNode()) {
    auto outcomes = state->ChanceOutcomes();
    state->ApplyAction(outcomes.front().first);
  }

  // Now we are at a non-chance decision node.
  assert(!state->IsChanceNode());
  assert(!state->IsTerminal());

  Player acting_player = state->CurrentPlayer();
  std::cout << "Acting player: P" << acting_player << "\n";

  // Test 1: Instantiation and Value Evaluation
  {
    std::cout << "=== Test 1: Value Evaluation & Scaling ===\n";
    double scale1 = 1.0;
    double scale2 = 4.0;
    
    DuneNNEvaluator eval1(model, device, scale1);
    DuneNNEvaluator eval2(model, device, scale2);

    auto values1 = eval1.Evaluate(*state);
    auto values2 = eval2.Evaluate(*state);

    assert(values1.size() == 4);
    assert(values2.size() == 4);

    std::cout << "Scale 1.0 values: ";
    for (double v : values1) std::cout << v << " ";
    std::cout << "\n";

    std::cout << "Scale 4.0 values: ";
    for (double v : values2) std::cout << v << " ";
    std::cout << "\n";

    for (int p = 0; p < 4; ++p) {
      // Check that scaling is applied linearly
      AssertAlmostEqual(values2[p], values1[p] * scale2, 1e-4);
      // Check that values are reasonable and not NaN
      assert(!std::isnan(values1[p]));
      assert(!std::isinf(values1[p]));
    }
    std::cout << "Test 1 Passed!\n\n";
  }

  // Test 2: Policy Prior Correctness
  {
    std::cout << "=== Test 2: Policy Prior Correctness ===\n";
    DuneNNEvaluator evaluator(model, device, 1.0);
    
    auto prior = evaluator.Prior(*state);
    auto legal_actions = state->LegalActions();

    assert(prior.size() == legal_actions.size());

    double prob_sum = 0.0;
    std::cout << "Prior actions and probabilities:\n";
    for (const auto& action_prob : prior) {
      Action action = action_prob.first;
      double prob = action_prob.second;
      std::cout << "  Action " << action << " (" << state->ActionToString(acting_player, action) << "): " << prob << "\n";
      
      // Probability must be strictly positive (since it's a softmax over legal actions)
      assert(prob > 0.0);
      assert(prob <= 1.0);
      
      // Action must be in the list of legal actions
      auto it = std::find(legal_actions.begin(), legal_actions.end(), action);
      assert(it != legal_actions.end());
      
      prob_sum += prob;
    }

    // Probabilities must sum to 1.0
    AssertAlmostEqual(prob_sum, 1.0, 1e-5);
    std::cout << "Test 2 Passed! Probability sum: " << prob_sum << "\n\n";
  }

  // Test 3: Sequential Evaluation Verification
  {
    std::cout << "=== Test 3: Sequential Evaluation Verification ===\n";
    DuneNNEvaluator evaluator(model, device, 1.0, 10.0f);

    auto values = evaluator.Evaluate(*state);

    assert(values.size() == 4);
    std::cout << "Evaluated values: ";
    for (double v : values) {
      std::cout << v << " ";
      assert(!std::isnan(v));
      assert(!std::isinf(v));
    }
    std::cout << "\n";

    std::cout << "Test 3 Passed!\n\n";
  }

  // Test 4: Chance Node Safety
  {
    std::cout << "=== Test 4: Chance Node Safety ===\n";
    std::unique_ptr<State> chance_state = game->NewInitialState();
    assert(chance_state->IsChanceNode());

    DuneNNEvaluator evaluator(model, device, 1.0);
    
    // Evaluate on a chance state should run safely and return 4 values
    auto values = evaluator.Evaluate(*chance_state);
    assert(values.size() == 4);
    for (double v : values) {
      assert(!std::isnan(v));
      assert(!std::isinf(v));
    }

    // Prior on a chance state should return an empty list
    auto prior = evaluator.Prior(*chance_state);
    assert(prior.empty());

    std::cout << "Test 4 Passed!\n\n";
  }

  std::cout << "All DuneNNEvaluator tests completed successfully!\n";
  return 0;
}
