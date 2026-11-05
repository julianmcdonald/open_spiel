#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include <ATen/cuda/CUDAContext.h>

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
  candidates.roles = {
      dune_semantic::ActionRole::kPrimaryCard,
      model->semantic_scorer_->is_f_enabled()
          ? dune_semantic::ActionRole::kTleilaxuPurchase
          : dune_semantic::ActionRole::kIndexedIntrigue,
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

  // =========================================================================
  // Part 5: FP8 (E4M3) Trunk Cache Verification Test
  // =========================================================================
  std::cout << "\n=== Part 5: FP8 (E4M3) Trunk Cache Verification Test ===\n";

  const auto* dev_props = at::cuda::getCurrentDeviceProperties();
  const bool skip_fp8 = (dev_props->major < 8 || (dev_props->major == 8 && dev_props->minor < 9));
  if (skip_fp8) {
    std::cout << "  GPU compute capability " << dev_props->major << "." << dev_props->minor
              << " < 8.9: skipping FP8 E4M3 verification on this GPU.\n";
  } else {
    std::cout << "  GPU compute capability " << dev_props->major << "." << dev_props->minor
              << " >= 8.9: running FP8 E4M3 verification.\n";

    torch::manual_seed(1337);
    auto fp8_model = std::make_shared<SharedDunePolicyValueNetImpl>(
        obs_size, hidden_dim, action_dim, num_blocks, use_nonlinear,
        with_aux_heads, /*head_init_seed=*/0, with_semantic_scorer,
        enable_zone_encoder, semantic_schema);
    fp8_model->to(device);
    if (argc > 1 && std::string(argv[1]).size() > 0) {
      LoadModelCheckpointRobust(fp8_model, argv[1], device);
    }
    fp8_model->eval();

    // Collect 80 in-distribution dune_imperium observations so Part 5 validates
    // FP8 numerical agreement on both seeded weights and trained checkpoints (argv[1]).
    auto dune_game = LoadGame("dune_imperium");
    const int64_t fp8_b = 80;
    torch::Tensor fp8_obs_cpu = torch::zeros(
        {fp8_b, obs_size}, torch::TensorOptions().dtype(torch::kFloat32));
    float* fp8_obs_ptr = fp8_obs_cpu.data_ptr<float>();
    int64_t collected = 0;
    for (int g_idx = 0; collected < fp8_b && g_idx < 32; ++g_idx) {
      auto st = dune_game->NewInitialState();
      int step = 0;
      while (!st->IsTerminal() && collected < fp8_b) {
        if (st->IsChanceNode()) {
          auto outcomes = st->ChanceOutcomes();
          st->ApplyAction(outcomes[(g_idx * 131 + step * 17) % outcomes.size()].first);
        } else {
          Player p = st->CurrentPlayer();
          std::vector<float> raw_obs(dune_game->ObservationTensorSize(), 0.0f);
          st->ObservationTensor(p, absl::MakeSpan(raw_obs));
          const int64_t copy_n = std::min<int64_t>(obs_size, static_cast<int64_t>(raw_obs.size()));
          std::memcpy(fp8_obs_ptr + collected * obs_size, raw_obs.data(), copy_n * sizeof(float));
          ++collected;
          auto legals = st->LegalActions();
          st->ApplyAction(legals[(g_idx * 37 + step * 31) % legals.size()]);
        }
        ++step;
      }
    }
    CHECK_TRUE(collected == fp8_b, "Collected 80 real dune_imperium observations");
    torch::Tensor fp8_obs = fp8_obs_cpu.to(device);

    // 1. M = 1 and M = 16 (< 32 threshold): PrepareFp8Cache() produces bitwise
    // identical logits and values (torch::equal) to RefreshBf16InferenceCache()
    // under AutocastGuard(c10::DeviceType::CUDA, true), confirming root priors
    // (M = 1) remain in exact BF16.
    for (int64_t small_m : {1, 16}) {
      auto small_obs = fp8_obs.narrow(0, 0, small_m);

      fp8_model->DisableFp8Cache();
      CHECK_TRUE(!fp8_model->fp8_cache_ready(), "FP8 cache must be disabled before BF16 baseline");
      fp8_model->RefreshBf16InferenceCache();
      SharedDunePolicyValueNetImpl::ModelOutputs bf16_out;
      {
        torch::InferenceMode guard;
        AutocastGuard amp(c10::DeviceType::CUDA, true);
        bf16_out = fp8_model->forward(small_obs);
      }

      fp8_model->PrepareFp8Cache();
      CHECK_TRUE(fp8_model->fp8_cache_ready(), "FP8 cache must be ready after PrepareFp8Cache()");
      SharedDunePolicyValueNetImpl::ModelOutputs fp8_out;
      {
        torch::InferenceMode guard;
        AutocastGuard amp(c10::DeviceType::CUDA, true);
        fp8_out = fp8_model->forward(small_obs);
      }

      CHECK_TRUE(torch::equal(bf16_out.logits, fp8_out.logits),
                 "M < 32 logits must match BF16 exactly (bitwise identical)");
      CHECK_TRUE(torch::equal(bf16_out.values, fp8_out.values),
                 "M < 32 values must match BF16 exactly (bitwise identical)");
      std::cout << "  M = " << small_m << " (< 32 fallback to exact BF16): PASS (bitwise identical)\n";
    }

    // 2. M = 64 (16-aligned) and M = 70 (unaligned M % 16 != 0 zero-padding path):
    // PrepareFp8Cache() executes cleanly with finite outputs (torch::isfinite(...).all()),
    // close agreement with BF16 (cosine_similarity > 0.98f on logits and max abs value diff < 0.08f),
    // and DisableFp8Cache() restores exact torch::equal bitwise parity with BF16.
    for (int64_t batch_m : {64, 70}) {
      auto batch_obs = fp8_obs.narrow(0, 0, batch_m);

      fp8_model->DisableFp8Cache();
      CHECK_TRUE(!fp8_model->fp8_cache_ready(), "FP8 cache must be disabled before BF16 baseline");
      fp8_model->RefreshBf16InferenceCache();
      SharedDunePolicyValueNetImpl::ModelOutputs bf16_out;
      {
        torch::InferenceMode guard;
        AutocastGuard amp(c10::DeviceType::CUDA, true);
        bf16_out = fp8_model->forward(batch_obs);
      }

      fp8_model->PrepareFp8Cache();
      SharedDunePolicyValueNetImpl::ModelOutputs fp8_out;
      {
        torch::InferenceMode guard;
        AutocastGuard amp(c10::DeviceType::CUDA, true);
        fp8_out = fp8_model->forward(batch_obs);
      }

      CHECK_TRUE(torch::isfinite(fp8_out.logits).all().item<bool>(), "FP8 logits must be finite");
      CHECK_TRUE(torch::isfinite(fp8_out.values).all().item<bool>(), "FP8 values must be finite");

      float cos_sim = torch::cosine_similarity(
          bf16_out.logits.to(torch::kFloat32).flatten(),
          fp8_out.logits.to(torch::kFloat32).flatten(),
          /*dim=*/0).item<float>();
      auto abs_val_diffs = (bf16_out.values.to(torch::kFloat32) -
                            fp8_out.values.to(torch::kFloat32)).abs();
      float mean_val_diff = abs_val_diffs.mean().item<float>();
      float max_val_diff = abs_val_diffs.max().item<float>();

      std::cout << "  M = " << batch_m << " FP8 agreement: cosine_sim=" << cos_sim
                << ", mean_val_diff=" << mean_val_diff
                << ", max_val_diff=" << max_val_diff << std::endl;
      CHECK_TRUE(cos_sim > 0.98f, "FP8 logits cosine similarity with BF16 must be > 0.98");
      CHECK_TRUE(mean_val_diff < 0.03f, "FP8 values mean abs diff with BF16 must be < 0.03");
      CHECK_TRUE(max_val_diff < 0.08f, "FP8 values max abs diff with BF16 must be < 0.08");

      fp8_model->DisableFp8Cache();
      CHECK_TRUE(!fp8_model->fp8_cache_ready(), "FP8 cache must be disabled after DisableFp8Cache()");
      SharedDunePolicyValueNetImpl::ModelOutputs restored_out;
      {
        torch::InferenceMode guard;
        AutocastGuard amp(c10::DeviceType::CUDA, true);
        restored_out = fp8_model->forward(batch_obs);
      }
      CHECK_TRUE(torch::equal(bf16_out.logits, restored_out.logits),
                 "DisableFp8Cache must restore exact bitwise parity with BF16 logits");
      CHECK_TRUE(torch::equal(bf16_out.values, restored_out.values),
                 "DisableFp8Cache must restore exact bitwise parity with BF16 values");
      std::cout << "  M = " << batch_m << " DisableFp8Cache bitwise BF16 restoration: PASS\n";
    }

    // 3. BatchedEvaluator SetEnableFp8Trunk(true) -> WarmupCudaGraphs({1, 64})
    // -> batch evaluation -> SetEnableFp8Trunk(false) -> WarmupCudaGraphs({1, 64})
    // executes cleanly with finite outputs and restores exact BF16 output after disabling FP8.
    {
      std::shared_mutex eval_mutex;
      BatchedEvaluator evaluator(fp8_model, /*target_batch_size=*/64, /*timeout_ms=*/1, device, &eval_mutex,
                                 /*logit_cap=*/10.0f, /*device_synchronize=*/false, /*high_priority_stream=*/true,
                                 /*emit_batch_membership=*/false, /*rollout_amp=*/true, /*allow_tf32=*/true);

      evaluator.SetEnableFp8Trunk(false);
      evaluator.WarmupCudaGraphs({1, 64});

      auto base_single = evaluator.EvaluateWithActions(observation, &candidates);
      CHECK_TRUE(base_single.ok, "Base single eval ok");
      CHECK_TRUE(base_single.logits.size() == action_dim, "Base single logits size");

      std::vector<std::vector<float>> batch_obs_64(64, observation);
      auto base_batch = evaluator.EvaluateBatch(batch_obs_64);
      CHECK_TRUE(base_batch.size() == 64, "Base batch size 64");
      for (const auto& r : base_batch) {
        CHECK_TRUE(r.ok, "Base batch item ok");
      }

      evaluator.SetEnableFp8Trunk(true);
      CHECK_TRUE(fp8_model->use_fp8_trunk(), "Model use_fp8_trunk must be true");
      evaluator.WarmupCudaGraphs({1, 64});

      auto fp8_single = evaluator.EvaluateWithActions(observation, &candidates);
      CHECK_TRUE(fp8_single.ok, "FP8 single eval ok");
      CHECK_TRUE(std::isfinite(fp8_single.value), "FP8 single value finite");
      for (float l : fp8_single.logits) {
        CHECK_TRUE(std::isfinite(l), "FP8 single logit finite");
      }

      auto fp8_batch = evaluator.EvaluateBatch(batch_obs_64);
      CHECK_TRUE(fp8_batch.size() == 64, "FP8 batch size 64");
      for (const auto& r : fp8_batch) {
        CHECK_TRUE(r.ok, "FP8 batch item ok");
        CHECK_TRUE(std::isfinite(r.value), "FP8 batch value finite");
      }

      evaluator.SetEnableFp8Trunk(false);
      CHECK_TRUE(!fp8_model->use_fp8_trunk(), "Model use_fp8_trunk must be false");
      evaluator.WarmupCudaGraphs({1, 64});

      auto restored_single = evaluator.EvaluateWithActions(observation, &candidates);
      CHECK_TRUE(restored_single.ok, "Restored single eval ok");
      CHECK_TRUE(restored_single.value == base_single.value,
                 "Restored single value bitwise matches base BF16");
      CHECK_TRUE(restored_single.logits.size() == base_single.logits.size(),
                 "Restored single logits size matches base BF16");
      for (size_t j = 0; j < base_single.logits.size(); ++j) {
        CHECK_TRUE(restored_single.logits[j] == base_single.logits[j],
                   "Restored single logit bitwise matches base BF16");
      }

      auto restored_batch = evaluator.EvaluateBatch(batch_obs_64);
      CHECK_TRUE(restored_batch.size() == 64, "Restored batch size 64");
      for (size_t i = 0; i < 64; ++i) {
        CHECK_TRUE(restored_batch[i].ok, "Restored batch item ok");
        CHECK_TRUE(restored_batch[i].value == base_batch[i].value,
                   "Restored batch value bitwise matches base BF16");
      }
      std::cout << "  BatchedEvaluator FP8 toggle and CudaGraph invalidation: PASS\n";
    }
  }

  std::cout << "\nALL BF16 LINEAR CACHE VERIFICATION TESTS PASSED!\n";
  return 0;
}
