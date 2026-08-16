#include <iostream>
#include <string>
#include <vector>
#include <memory>
#include <cmath>
#include <cassert>
#include <atomic>
#include <mutex>
#include <shared_mutex>
#include <stdexcept>
#include <thread>

#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"
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

// Compares two EvalResults (value + full logits vector) within a tolerance.
void AssertResultsEqual(const open_spiel::EvalResult& a,
                        const open_spiel::EvalResult& b, double tol = 1e-4) {
  assert(a.logits.size() == b.logits.size());
  AssertAlmostEqual(a.value, b.value, tol);
  for (size_t i = 0; i < a.logits.size(); ++i) {
    AssertAlmostEqual(a.logits[i], b.logits[i], tol);
  }
}

// A stand-in for OpenSpiel's default (process-exiting) error handler, used to
// restore normal fatal-error behavior after a scoped throwing handler.
void ExitingErrorHandler(const std::string& msg) {
  std::cerr << "Spiel Fatal Error: " << msg << std::endl;
  std::exit(1);
}

// Evaluates one observation of the given length on a DeterministicEvaluator
// under a throwing error handler; returns true iff the observation-size contract
// rejected it. Used to pin down the accept/reject boundary.
bool EvalRejectsObsSize(
    std::shared_ptr<open_spiel::SharedDunePolicyValueNetImpl> model,
    torch::Device device, int64_t obs_len) {
  std::mutex eval_mutex;
  std::shared_mutex sync_mutex;
  open_spiel::DeterministicEvaluator de(model, device, &eval_mutex,
                                        &sync_mutex);
  std::vector<float> obs(static_cast<size_t>(obs_len), 0.5f);
  bool threw = false;
  open_spiel::SetErrorHandler(
      [](const std::string& msg) { throw std::runtime_error(msg); });
  try {
    de.Evaluate(obs);
  } catch (const std::exception&) {
    threw = true;
  }
  open_spiel::SetErrorHandler(ExitingErrorHandler);
  return threw;
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
    
    DuneNNEvaluator eval1(model, device);
    DuneNNEvaluator eval2(model, device);

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
      // Check that values are unscaled by divisor
      AssertAlmostEqual(values2[p], values1[p], 1e-4);
      // Check that values are reasonable and not NaN
      assert(!std::isnan(values1[p]));
      assert(!std::isinf(values1[p]));
    }
    std::cout << "Test 1 Passed!\n\n";
  }

  // Test 2: Policy Prior Correctness
  {
    std::cout << "=== Test 2: Policy Prior Correctness ===\n";
    DuneNNEvaluator evaluator(model, device);
    
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
    DuneNNEvaluator evaluator(model, device, 10.0f);

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

    DuneNNEvaluator evaluator(model, device);
    
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

  // Test 5: Combined Evaluation Parity
  {
    std::cout << "=== Test 5: Combined Evaluation Parity ===\n";
    DuneNNEvaluator evaluator(model, device, 10.0f);

    auto prior_seq = evaluator.Prior(*state);
    auto values_seq = evaluator.Evaluate(*state);

    auto combined = evaluator.PriorAndEvaluate(*state);
    auto prior_comb = combined.first;
    auto values_comb = combined.second;

    assert(values_comb.size() == values_seq.size());
    for (size_t i = 0; i < values_seq.size(); ++i) {
      AssertAlmostEqual(values_comb[i], values_seq[i], 1e-2);
    }

    assert(prior_comb.size() == prior_seq.size());
    for (size_t i = 0; i < prior_seq.size(); ++i) {
      assert(prior_comb[i].first == prior_seq[i].first);
      AssertAlmostEqual(prior_comb[i].second, prior_seq[i].second, 1e-2);
    }

    std::cout << "Test 5 Passed!\n\n";
  }

  // Test 6: BatchedEvaluator mixed-size batch (WO-02, search finding 6).
  // A batch mixing the two engine observation layouts must size every request
  // from its OWN vector, not from batch[0]. Pre-fix, batch[0]'s length was used
  // for all requests: a shorter obs behind a longer batch[0] was over-read, a
  // longer obs behind a shorter batch[0] was truncated. Each request must yield
  // the same result it gets when evaluated alone, regardless of batch position.
  {
    std::cout << "=== Test 6: Mixed-size batch per-request sizing ===\n";
    // The only batch that legitimately mixes observation lengths is the
    // supported 5580 <-> 5584 dual layout; use exactly that so the request also
    // passes the (now strict) size contract.
    const int64_t kFull = 5584;
    const int64_t kShort = 5580;
    torch::manual_seed(20260723);
    auto small_model = std::make_shared<SharedDunePolicyValueNetImpl>(
        kFull, /*hidden_dim=*/32, /*action_dim=*/8, /*num_blocks=*/1);
    small_model->eval();
    torch::Device cpu(torch::kCPU);
    small_model->to(cpu);
    std::shared_mutex sync_mutex;
    BatchedEvaluator be(small_model, /*target_batch_size=*/4, /*timeout_ms=*/20,
                        cpu, &sync_mutex);

    std::vector<float> obs_full(kFull), obs_short(kShort);
    for (int64_t i = 0; i < kFull; ++i)
      obs_full[i] = 0.01f * static_cast<float>((i % 17) + 1);
    // Make the 4 trailing floats (indices 5580..5583, dropped by pre-fix
    // truncation when a shorter obs sits at batch[0]) a large, distinctive
    // signal so the truncation is unmistakably detectable.
    for (int64_t i = kShort; i < kFull; ++i) obs_full[i] = 3.0f;
    for (int64_t i = 0; i < kShort; ++i)
      obs_short[i] = -0.01f * static_cast<float>((i % 19) + 1);

    // Reference: each obs evaluated alone (a single-element batch sizes correctly
    // in every version, so these are the ground-truth per-request results).
    EvalResult ref_full = be.EvaluateBatch({obs_full})[0];
    EvalResult ref_short = be.EvaluateBatch({obs_short})[0];

    // Short first: pre-fix sizes the batch from the short obs and truncates
    // obs_full (deterministic mismatch). Long first: pre-fix over-reads obs_short.
    auto mixed_short_first = be.EvaluateBatch({obs_short, obs_full});
    AssertResultsEqual(mixed_short_first[0], ref_short);
    AssertResultsEqual(mixed_short_first[1], ref_full);

    auto mixed_long_first = be.EvaluateBatch({obs_full, obs_short});
    AssertResultsEqual(mixed_long_first[0], ref_full);
    AssertResultsEqual(mixed_long_first[1], ref_short);

    std::cout << "Test 6 Passed!\n\n";
  }

  // Test 7: Observation-size contract is exact match or the 5580/5584 pair only
  // (WO-02). Near misses (+/-1, +/-2) and gross mismatches must be rejected --
  // never silently zero-padded or truncated. A throwing error handler makes the
  // fatal error observable from the caller thread.
  {
    std::cout << "=== Test 7: Observation-size contract (exact or 5580/5584) ===\n";
    torch::Device cpu(torch::kCPU);
    torch::manual_seed(20260724);

    // Model whose input matches neither dual-layout value: only the exact size
    // is valid, everything else (including +/-1) must be rejected.
    auto m16 = std::make_shared<SharedDunePolicyValueNetImpl>(
        16, /*hidden_dim=*/32, /*action_dim=*/8, /*num_blocks=*/1);
    m16->eval();
    m16->to(cpu);
    assert(!EvalRejectsObsSize(m16, cpu, 16));  // exact match accepted
    for (int64_t bad : {14, 15, 17, 18, 24}) {  // +/-1, +/-2, gross: all rejected
      assert(EvalRejectsObsSize(m16, cpu, bad));
    }

    // A 5584-input model: the 5580 obs is the one tolerated non-exact case;
    // 5581/5583/5585 (a single element off) must still be rejected.
    auto m5584 = std::make_shared<SharedDunePolicyValueNetImpl>(
        5584, /*hidden_dim=*/8, /*action_dim=*/4, /*num_blocks=*/1);
    m5584->eval();
    m5584->to(cpu);
    assert(!EvalRejectsObsSize(m5584, cpu, 5584));  // exact
    assert(!EvalRejectsObsSize(m5584, cpu, 5580));  // supported pair
    assert(EvalRejectsObsSize(m5584, cpu, 5581));   // near miss rejected
    assert(EvalRejectsObsSize(m5584, cpu, 5583));   // near miss rejected
    assert(EvalRejectsObsSize(m5584, cpu, 5585));   // near miss rejected

    std::cout << "Test 7 Passed!\n\n";
  }

  // Test 8: CUDA deterministic-evaluator concurrency stress (WO-02 finding 1).
  // Many threads share ONE DeterministicEvaluator (hence one pinned staging
  // buffer). Each thread repeatedly evaluates its own distinct observation and
  // must always get that observation's reference result. Pre-fix, the async H2D
  // copy let the next thread overwrite the shared pinned buffer mid-DMA, handing
  // one thread another game's logits/values. Skips clearly without CUDA.
  {
    std::cout << "=== Test 8: CUDA deterministic concurrency stress ===\n";
    if (!torch::cuda::is_available()) {
      std::cout << "SKIPPED: CUDA not available (cannot exercise the pinned "
                   "staging-buffer race on CPU).\n\n";
    } else {
      torch::Device cuda(torch::kCUDA);
      // A large observation makes each host->device copy slow enough (hundreds
      // of KB over PCIe) that its DMA is still reading the shared pinned buffer
      // when the next thread's memcpy overwrites it -- the exact window the
      // finding-1 race exploits. A tiny obs completes the H2D faster than the
      // mutex handoff and hides the bug. The forward is kept lean so the test
      // stays fast; the race is about the staging copy, not model size.
      const int64_t kInput = 65536;
      torch::manual_seed(20260725);
      auto stress_model = std::make_shared<SharedDunePolicyValueNetImpl>(
          kInput, /*hidden_dim=*/128, /*action_dim=*/32, /*num_blocks=*/1);
      stress_model->eval();
      stress_model->to(cuda);

      std::mutex eval_mutex;
      std::shared_mutex sync_mutex;
      DeterministicEvaluator shared_eval(stress_model, cuda, &eval_mutex,
                                         &sync_mutex);

      const int num_threads = 32;
      const int iters = 100;

      // Distinct observation per thread (different sinusoid frequencies).
      std::vector<std::vector<float>> obs(num_threads,
                                          std::vector<float>(kInput));
      for (int t = 0; t < num_threads; ++t) {
        for (int64_t i = 0; i < kInput; ++i) {
          obs[t][i] = std::sin(0.05f * static_cast<float>(t + 1) *
                               static_cast<float>(i + 1));
        }
      }

      // Uncontended reference result for each thread's observation.
      std::vector<EvalResult> ref(num_threads);
      for (int t = 0; t < num_threads; ++t) ref[t] = shared_eval.Evaluate(obs[t]);

      // Sanity: references must be distinguishable, else contamination would be
      // invisible to the comparison below.
      double vmin = ref[0].value, vmax = ref[0].value;
      for (int t = 1; t < num_threads; ++t) {
        vmin = std::min<double>(vmin, ref[t].value);
        vmax = std::max<double>(vmax, ref[t].value);
      }
      assert(vmax - vmin > 1e-2);

      std::atomic<int> mismatches{0};
      std::vector<std::thread> threads;
      for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back([&, t]() {
          for (int it = 0; it < iters; ++it) {
            EvalResult r = shared_eval.Evaluate(obs[t]);
            bool ok = std::abs(r.value - ref[t].value) <= 1e-2;
            for (size_t k = 0; ok && k < r.logits.size(); ++k) {
              if (std::abs(r.logits[k] - ref[t].logits[k]) > 1e-2) ok = false;
            }
            if (!ok) mismatches.fetch_add(1, std::memory_order_relaxed);
          }
        });
      }
      for (auto& th : threads) th.join();

      std::cout << "Concurrent evaluations: " << (num_threads * iters)
                << ", cross-contamination mismatches: " << mismatches.load()
                << "\n";
      assert(mismatches.load() == 0);
      std::cout << "Test 8 Passed!\n\n";
    }
  }

  // Test 9: BatchedEvaluator device-timing telemetry (WO-PERF-TIMING-BATCH).
  //
  // Test 8 does NOT cover this: it stresses the DeterministicEvaluator, which
  // has no batcher and no telemetry, so it cannot show whether the timing
  // fields are wired, gated, or zeroed correctly. This test drives the
  // BatchedEvaluator directly.
  //
  // Three claims, on CPU so it runs everywhere:
  //   (a) with telemetry DISABLED the timing fields stay exactly zero;
  //   (b) EnableBatcherTelemetry() arms the per-batch bookkeeping, so
  //       physical_batches becomes nonzero and matches the rows submitted;
  //   (c) on CPU the four DEVICE intervals stay zero even when armed, because
  //       there is no CUDA stream to record events on -- "armed" must not be
  //       confused with "measured".
  {
    std::cout << "=== Test 9: BatchedEvaluator telemetry gating and counts ===\n";
    const int64_t kObs = 5580;
    torch::manual_seed(20277001);
    auto model = std::make_shared<SharedDunePolicyValueNetImpl>(
        kObs, /*hidden_dim=*/32, /*action_dim=*/8, /*num_blocks=*/1);
    model->eval();
    torch::Device cpu(torch::kCPU);
    model->to(cpu);

    std::vector<float> obs(kObs, 0.25f);

    // (a) DISABLED: no EnableBatcherTelemetry() call at all.
    {
      std::shared_mutex sync_mutex;
      BatchedEvaluator be(model, /*target_batch_size=*/4, /*timeout_ms=*/5,
                          cpu, &sync_mutex);
      for (int i = 0; i < 8; ++i) (void)be.Evaluate(obs);
      BatcherTelemetry t = be.GetBatcherTelemetry();
      assert(t.h2d_ms == 0.0);
      assert(t.forward_ms == 0.0);
      assert(t.output_cast_ms == 0.0);
      assert(t.d2h_ms == 0.0);
      assert(t.sync_ms == 0.0);
      assert(t.device_timed_batches == 0);
      // The always-on counters still move; only the opt-in sections are dark.
      assert(t.submitted_rows == 8);
      std::cout << "  disabled: all timing zero, submitted_rows="
                << t.submitted_rows << "\n";
    }

    // (b)+(c) ENABLED on CPU.
    {
      std::shared_mutex sync_mutex;
      BatchedEvaluator be(model, /*target_batch_size=*/4, /*timeout_ms=*/5,
                          cpu, &sync_mutex);
      be.EnableBatcherTelemetry();
      const int kCalls = 12;
      for (int i = 0; i < kCalls; ++i) (void)be.Evaluate(obs);
      BatcherTelemetry t = be.GetBatcherTelemetry();

      assert(t.submitted_rows == static_cast<uint64_t>(kCalls));
      assert(t.single_row_calls == static_cast<uint64_t>(kCalls));
      assert(t.physical_batches > 0);
      // Every row lands in exactly one physical batch, so batches can never
      // exceed rows and the mean must be a sane positive number.
      assert(t.physical_batches <= t.submitted_rows);
      assert(t.mean_batch_size > 0.0);
      assert(t.max_batch_size >= 1);

      // CPU: armed, but no CUDA events exist, so device time stays zero and
      // device_timed_batches stays zero. This is the distinction the field was
      // added for -- "measured zero" vs "never measured".
      assert(t.h2d_ms == 0.0);
      assert(t.forward_ms == 0.0);
      assert(t.output_cast_ms == 0.0);
      assert(t.d2h_ms == 0.0);
      assert(t.device_timed_batches == 0);

      std::cout << "  enabled(CPU): rows=" << t.submitted_rows
                << " batches=" << t.physical_batches
                << " mean=" << t.mean_batch_size
                << " device_timed_batches=" << t.device_timed_batches
                << " (0 expected on CPU)\n";
    }

    // On CUDA, device_timed_batches must equal physical_batches: every batch
    // that runs is a batch that gets timed.
    if (torch::cuda::is_available()) {
      torch::Device gpu(torch::kCUDA, 0);
      auto gmodel = std::make_shared<SharedDunePolicyValueNetImpl>(
          kObs, /*hidden_dim=*/32, /*action_dim=*/8, /*num_blocks=*/1);
      gmodel->eval();
      gmodel->to(gpu);
      std::shared_mutex sync_mutex;
      BatchedEvaluator be(gmodel, /*target_batch_size=*/4, /*timeout_ms=*/5,
                          gpu, &sync_mutex);
      be.EnableBatcherTelemetry();
      for (int i = 0; i < 12; ++i) (void)be.Evaluate(obs);
      BatcherTelemetry t = be.GetBatcherTelemetry();
      assert(t.physical_batches > 0);
      assert(t.device_timed_batches == t.physical_batches);
      assert(t.forward_ms > 0.0);   // a real forward took real time
      assert(t.h2d_ms >= 0.0);
      assert(t.output_cast_ms >= 0.0);
      assert(t.d2h_ms >= 0.0);
      assert(t.sync_ms >= 0.0);
      std::cout << "  enabled(CUDA): batches=" << t.physical_batches
                << " timed=" << t.device_timed_batches
                << " h2d=" << t.h2d_ms << "ms forward=" << t.forward_ms
                << "ms cast=" << t.output_cast_ms << "ms d2h=" << t.d2h_ms
                << "ms sync=" << t.sync_ms << "ms (sync OVERLAPS the rest)\n";
    } else {
      std::cout << "  CUDA unavailable: device-interval assertions skipped\n";
    }
    std::cout << "Test 9 Passed!\n\n";
  }

  std::cout << "All DuneNNEvaluator tests completed successfully!\n";
  return 0;
}
