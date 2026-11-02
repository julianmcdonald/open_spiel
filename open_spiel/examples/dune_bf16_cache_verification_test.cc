#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "open_spiel/spiel.h"
#include "open_spiel/games/dune_imperium/dune_imperium.h"
#include "dune_network.h"
#include "dune_semantic_action_scorer.h"
#include "dune_ppo_training_utils.h"

using namespace open_spiel;

#define CHECK_TRUE(cond, msg)                                         \
  do {                                                                \
    if (!(cond)) {                                                    \
      std::cerr << "FAILED: " << msg << "\n  Assertion: " #cond       \
                << "\n  at " << __FILE__ << ":" << __LINE__ << "\n";  \
      std::abort();                                                   \
    }                                                                 \
  } while (0)

int main(int argc, char** argv) {
  if (!torch::cuda::is_available()) {
    std::cerr << "CUDA is not available. Skipping test.\n";
    return 0;
  }

  torch::Device device(torch::kCUDA, 0);
  std::cout << "Running Dune BF16 Linear Cache Verification on CUDA device: "
            << device << std::endl;

  const int64_t obs_size = 9941;
  const int64_t hidden_dim = 2048;
  const int64_t action_dim = 2391;
  const int num_blocks = 16;
  const bool use_nonlinear = false;
  const bool with_aux_heads = false;
  const bool with_semantic_scorer = true;
  const bool enable_zone_encoder = false;
  const std::string semantic_schema = "semantic_action_v3";

  torch::manual_seed(1337);
  auto model = std::make_shared<SharedDunePolicyValueNetImpl>(
      obs_size, hidden_dim, action_dim, num_blocks, use_nonlinear,
      with_aux_heads, /*head_init_seed=*/0, with_semantic_scorer,
      enable_zone_encoder, semantic_schema);
  model->to(device);

  if (argc > 1 && std::string(argv[1]).size() > 0) {
    const std::string checkpoint_path = argv[1];
    std::cout << "Loading checkpoint from argv[1]: " << checkpoint_path << std::endl;
    LoadModelCheckpointRobust(model, checkpoint_path, device);
  } else {
    std::cout << "Using deterministic seeded model weights (pass checkpoint path as argv[1] to override)."
              << std::endl;
  }
  model->eval();

  // Create fixed deterministic batch of B=128 observations
  const int64_t B = 128;
  torch::manual_seed(42);
  torch::Tensor obs = torch::randn({B, obs_size}, torch::TensorOptions().dtype(torch::kFloat32).device(device));

  // =========================================================================
  // Part 1: Bitwise Numerical Parity Test (Baseline BF16 vs Persistent BF16)
  // =========================================================================
  std::cout << "\n=== Part 1: Baseline BF16 vs Persistent BF16 Parity Test (B=128) ===" << std::endl;

  // 1a. Baseline BF16: cache disabled (on-the-fly autocast conversion)
  model->DisableBf16InferenceCache();
  CHECK_TRUE(!model->bf16_inference_cache_ready(), "Cache must be disabled");

  SharedDunePolicyValueNetImpl::ModelOutputs baseline_out;
  {
    torch::NoGradGuard no_grad;
    AutocastGuard guard(c10::DeviceType::CUDA, /*enabled=*/true);
    baseline_out = model->forward(obs);
  }
  torch::Tensor base_logits = baseline_out.logits.clone();
  torch::Tensor base_values = baseline_out.values.clone();
  torch::Tensor base_trunk = baseline_out.trunk.clone();

  // 1b. Persistent BF16: refresh inference cache and run with persistent weights
  model->RefreshBf16InferenceCache();
  CHECK_TRUE(model->bf16_inference_cache_ready(), "Cache must be ready");

  SharedDunePolicyValueNetImpl::ModelOutputs cached_out;
  {
    torch::NoGradGuard no_grad;
    AutocastGuard guard(c10::DeviceType::CUDA, /*enabled=*/true);
    cached_out = model->forward(obs);
  }
  torch::Tensor cached_logits = cached_out.logits.clone();
  torch::Tensor cached_values = cached_out.values.clone();
  torch::Tensor cached_trunk = cached_out.trunk.clone();

  float diff_logits = (base_logits - cached_logits).abs().max().item<float>();
  float diff_values = (base_values - cached_values).abs().max().item<float>();
  float diff_trunk = (base_trunk - cached_trunk).abs().max().item<float>();

  std::cout << "  Logits max abs diff: " << diff_logits << std::endl;
  std::cout << "  Values max abs diff: " << diff_values << std::endl;
  std::cout << "  Trunk  max abs diff: " << diff_trunk << std::endl;

  CHECK_TRUE(diff_logits == 0.0f, "Logits diff must be exactly 0.0");
  CHECK_TRUE(diff_values == 0.0f, "Values diff must be exactly 0.0");
  CHECK_TRUE(diff_trunk == 0.0f, "Trunk diff must be exactly 0.0");
  std::cout << ">> SUCCESS: Bitwise output parity verified (max_abs_diff == 0.0)!\n";

  // =========================================================================
  // Part 2: CUDAGraph In-Place Weight Refresh Parity Test
  // =========================================================================
  std::cout << "\n=== Part 2: CUDAGraph In-Place Weight Mutation & Refresh Test ===" << std::endl;

  c10::cuda::CUDAStream capture_stream = c10::cuda::getStreamFromPool();
  at::cuda::CUDAGraph graph;
  SharedDunePolicyValueNetImpl::ModelOutputs graph_out;

  {
    c10::cuda::CUDAStreamGuard stream_guard(capture_stream);
    // Warmup iterations
    for (int i = 0; i < 3; ++i) {
      torch::NoGradGuard no_grad;
      AutocastGuard guard(c10::DeviceType::CUDA, /*enabled=*/true);
      graph_out = model->forward(obs);
    }
    c10::cuda::getCurrentCUDAStream().synchronize();
    at::autocast::clear_cache();

    // Graph capture
    {
      torch::NoGradGuard no_grad;
      AutocastGuard guard(c10::DeviceType::CUDA, /*enabled=*/true);
      graph.capture_begin();
      graph_out = model->forward(obs);
      graph.capture_end();
    }
  }
  c10::cuda::getCurrentCUDAStream().synchronize();

  // Replay graph with initial weights
  graph.replay();
  c10::cuda::getCurrentCUDAStream().synchronize();
  torch::Tensor pre_mutation_logits = graph_out.logits.clone();
  torch::Tensor pre_mutation_values = graph_out.values.clone();

  float replay1_diff = (pre_mutation_logits - cached_logits).abs().max().item<float>();
  std::cout << "  Replay 1 vs uncaptured diff: " << replay1_diff << std::endl;
  CHECK_TRUE(replay1_diff == 0.0f, "Initial graph replay must match uncaptured forward pass");

  // Mutate FP32 master weights in place (simulating PPO learner gradient update)
  std::cout << "  Mutating master weights in place..." << std::endl;
  {
    torch::NoGradGuard no_grad;
    model->input_layer->weight.add_(0.05f);
    model->policy_head->weight.add_(0.05f);
    model->value_head->weight.add_(0.05f);
    for (auto& block : model->res_blocks) {
      block->fc1->weight.add_(0.02f);
      block->fc2->weight.add_(0.02f);
    }
  }

  // Refresh persistent BF16 cache in place
  std::cout << "  Refreshing persistent BF16 cache..." << std::endl;
  model->RefreshBf16InferenceCache();
  c10::cuda::getCurrentCUDAStream().synchronize();

  // Replay graph WITHOUT recapturing
  std::cout << "  Replaying CUDAGraph without recapturing..." << std::endl;
  graph.replay();
  c10::cuda::getCurrentCUDAStream().synchronize();
  torch::Tensor post_mutation_logits = graph_out.logits.clone();
  torch::Tensor post_mutation_values = graph_out.values.clone();

  float diff_from_pre = (post_mutation_logits - pre_mutation_logits).abs().max().item<float>();
  std::cout << "  Diff from pre-mutation: " << diff_from_pre << std::endl;
  CHECK_TRUE(diff_from_pre > 0.0f, "Replayed graph output must change after weight refresh");

  // Run fresh uncaptured forward pass with new weights to verify graph accuracy
  SharedDunePolicyValueNetImpl::ModelOutputs fresh_out;
  {
    torch::NoGradGuard no_grad;
    AutocastGuard guard(c10::DeviceType::CUDA, /*enabled=*/true);
    fresh_out = model->forward(obs);
  }
  float diff_from_fresh_logits = (post_mutation_logits - fresh_out.logits).abs().max().item<float>();
  float diff_from_fresh_values = (post_mutation_values - fresh_out.values).abs().max().item<float>();
  std::cout << "  Diff from fresh uncaptured logits: " << diff_from_fresh_logits << std::endl;
  std::cout << "  Diff from fresh uncaptured values: " << diff_from_fresh_values << std::endl;

  CHECK_TRUE(diff_from_fresh_logits == 0.0f, "Replayed graph logits must exactly match fresh forward pass");
  CHECK_TRUE(diff_from_fresh_values == 0.0f, "Replayed graph values must exactly match fresh forward pass");

  std::cout << ">> SUCCESS: CUDAGraph weight-refresh parity verified (max_abs_diff == 0.0)!\n";

  // =========================================================================
  // Part 3: CopyModelWeights & Semantic Scorer Test
  // =========================================================================
  std::cout << "\n=== Part 3: CopyModelWeights & Semantic Scorer BF16 Cache Test ===" << std::endl;

  auto target_model = std::make_shared<SharedDunePolicyValueNetImpl>(
      obs_size, hidden_dim, action_dim, num_blocks, use_nonlinear,
      with_aux_heads, /*head_init_seed=*/0, with_semantic_scorer,
      enable_zone_encoder, semantic_schema);
  target_model->to(device);
  target_model->eval();
  target_model->RefreshBf16InferenceCache();

  CopyModelWeights(model, target_model);
  CHECK_TRUE(target_model->bf16_inference_cache_ready(), "Target model cache must be ready after copy");

  SharedDunePolicyValueNetImpl::ModelOutputs target_out;
  {
    torch::NoGradGuard no_grad;
    AutocastGuard guard(c10::DeviceType::CUDA, /*enabled=*/true);
    target_out = target_model->forward(obs);
  }
  float copy_diff = (fresh_out.logits - target_out.logits).abs().max().item<float>();
  std::cout << "  Target model copy parity diff: " << copy_diff << std::endl;
  CHECK_TRUE(copy_diff == 0.0f, "Copied model with persistent cache must match source model bitwise");

  std::cout << ">> SUCCESS: CopyModelWeights cache synchronization verified!\n";

  // =========================================================================
  // Part 4: Independent Uncached Master Reference, Scorer, Grad & Evaluator Checks
  // =========================================================================
  std::cout << "\n=== Part 4: Uncached Master Reference, Semantic Scorer, Grad & Evaluator Checks ===\n";
  model->DisableBf16InferenceCache();
  {
    torch::InferenceMode guard;
    AutocastGuard amp(c10::DeviceType::CUDA, true);
    auto master_out = model->forward(obs);
    CHECK_TRUE(torch::equal(post_mutation_logits, master_out.logits),
               "Refreshed graph must match uncached master weights");
    CHECK_TRUE(torch::equal(post_mutation_values, master_out.values),
               "Refreshed graph values must match uncached master weights");
  }
  std::cout << "  Uncached master reference after mutation: PASS\n";

  dune_semantic::CandidateActionData candidates;
  candidates.actions = {10, 20, 30};
  candidates.features.assign(3 * dune_semantic::kSemanticFeatDim, 0.25f);
  candidates.card_ids = {1, 2, 3};
  candidates.space_ids = {0, 1, 2};
  candidates.supported = {1, 1, 1};
  candidates.roles = {dune_semantic::ActionRole::kPrimaryCard,
                      dune_semantic::ActionRole::kTleilaxuPurchase,
                      dune_semantic::ActionRole::kBoardSpace};
  std::vector<const dune_semantic::CandidateActionData*> actions(B, &candidates);
  auto scorer_result = [&]() {
    torch::InferenceMode guard;
    AutocastGuard amp(c10::DeviceType::CUDA, true);
    auto out = model->forward(obs);
    out.logits = torch::zeros_like(out.logits);
    dune_semantic::ApplySemanticScorerBatch(model->semantic_scorer_, out.trunk,
                                            actions, out.logits, device);
    return out.logits.clone();
  };
  auto uncached_scorer = scorer_result();
  model->RefreshBf16InferenceCache();
  CHECK_TRUE(torch::equal(uncached_scorer, scorer_result()), "Scorer parity");
  {
    torch::NoGradGuard guard;
    model->semantic_scorer_->out_layer->bias.add_(1.0f);
    model->semantic_scorer_->out_layer_ext->bias.add_(2.0f);
  }
  model->RefreshBf16InferenceCache();
  auto changed_scorer = scorer_result();
  CHECK_TRUE(!torch::equal(changed_scorer, uncached_scorer), "Scorer mutation visible");
  model->DisableBf16InferenceCache();
  CHECK_TRUE(torch::equal(changed_scorer, scorer_result()), "Scorer refresh parity");
  std::cout << "  Semantic scorer inference parity and refresh: PASS\n";

  auto fc = torch::nn::Linear(16, 16);
  fc->to(device);
  dune_semantic::Bf16LinearCache linear_cache;
  linear_cache.Refresh(fc);
  auto input = torch::randn({4, 16}, torch::TensorOptions().device(device));
  {
    torch::NoGradGuard guard;
    fc->weight.add_(1.0f);
  }
  auto gradient = [&](bool cached) {
    fc->zero_grad();
    AutocastGuard amp(c10::DeviceType::CUDA, true);
    linear_cache.Forward(fc, input, cached).sum().backward();
    return fc->weight.grad().clone();
  };
  CHECK_TRUE(torch::equal(gradient(false), gradient(true)), "Gradient cache bypass");
  std::cout << "  Gradient-enabled master weight path: PASS\n";

  auto cpu_obs = obs[0].to(torch::kCPU).contiguous();
  std::vector<float> observation(cpu_obs.data_ptr<float>(),
                                 cpu_obs.data_ptr<float>() + obs_size);
  for (bool synchronize : {false, true}) {
    std::shared_mutex mutex;
    BatchedEvaluator evaluator(target_model, 1, 1, device, &mutex, 0.0f,
                               synchronize, false, false, true, true);
    evaluator.EvaluateWithActions(observation, &candidates);
    evaluator.EvaluateWithActions(observation, &candidates);
    {
      std::unique_lock<std::shared_mutex> lock(mutex);
      CopyModelWeights(model, target_model);
    }
    for (int slot = 0; slot < 2; ++slot) {
      auto actual = evaluator.EvaluateWithActions(observation, &candidates);
      torch::InferenceMode guard;
      AutocastGuard amp(c10::DeviceType::CUDA, true);
      auto expected = model->forward(obs.narrow(0, 0, 1));
      std::vector<const dune_semantic::CandidateActionData*> one{&candidates};
      dune_semantic::ApplySemanticScorerBatch(model->semantic_scorer_, expected.trunk,
                                              one, expected.logits, device);
      auto logits = expected.logits.to(torch::kFloat32).to(torch::kCPU).contiguous();
      CHECK_TRUE(actual.logits.size() == action_dim, "Evaluator output size");
      for (int64_t j = 0; j < action_dim; ++j) {
        CHECK_TRUE(actual.logits[j] == logits.data_ptr<float>()[j],
                   "Evaluator after copy matches uncached master");
      }
    }
    std::cout << "  Evaluator refresh synchronize=" << synchronize << ": PASS\n";
  }

  std::cout << "\nALL BF16 LINEAR CACHE VERIFICATION TESTS PASSED!\n";
  return 0;
}
