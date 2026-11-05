#include <iostream>
#include <string>
#include <vector>
#include <memory>
#include <cmath>
#include <cassert>
#include <atomic>
#include <chrono>
#include <mutex>
#include <shared_mutex>
#include <stdexcept>
#include <thread>

#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"
#include <torch/torch.h>
#include <cstdlib>

#include "dune_network.h"
#include "dune_evaluator.h"
#include "dune_batched_evaluator.h"

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

// The group-cap guard must fire synchronously in the caller, before any rows
// are queued. A throwing handler makes that fatal contract observable here.
bool BatchRejectsOversizedGroup(
    std::shared_ptr<open_spiel::SharedDunePolicyValueNetImpl> model,
    torch::Device device, int target, int group_size, int64_t obs_len) {
  std::shared_mutex sync_mutex;
  open_spiel::BatchedEvaluator be(model, target, /*timeout_ms=*/5, device,
                                  &sync_mutex);
  std::vector<std::vector<float>> observations(
      static_cast<size_t>(group_size),
      std::vector<float>(static_cast<size_t>(obs_len), 0.5f));
  bool threw = false;
  open_spiel::SetErrorHandler(
      [](const std::string& msg) { throw std::runtime_error(msg); });
  try {
    (void)be.EvaluateBatch(observations);
  } catch (const std::exception& e) {
    threw = std::string(e.what()).find("exceeds configured target") !=
            std::string::npos;
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

  // Test 10: warm-up must not contaminate scored telemetry
  // (WO-PERF-TIMING-BATCH regression).
  //
  // EnableBatcherTelemetry() arms the DETAILED per-batch sections. It does NOT
  // reset the always-on counters. A warm-up run through the evaluator that then
  // serves the scored run therefore leaves submitted_rows, physical_batches and
  // the class counters already advanced, and -- because timing arms afterwards
  // -- leaves device_timed_batches permanently short of physical_batches.
  //
  // This test asserts BOTH halves: that the contamination is real on a shared
  // instance (so the fix is not cargo-culted), and that a fresh instance is
  // clean (so the throwaway-evaluator pattern actually works).
  {
    std::cout << "=== Test 10: warm-up isolation and timing coverage ===\n";
    const int64_t kObs = 5580;
    torch::manual_seed(20277002);
    auto model = std::make_shared<SharedDunePolicyValueNetImpl>(
        kObs, /*hidden_dim=*/32, /*action_dim=*/8, /*num_blocks=*/1);
    model->eval();
    torch::Device cpu(torch::kCPU);
    model->to(cpu);
    std::vector<float> obs(kObs, 0.25f);
    const int kWarm = 5, kScored = 7;

    // (a) SHARED instance -- the bug. Arming after use does not rewind.
    {
      std::shared_mutex m;
      BatchedEvaluator be(model, /*target_batch_size=*/4, /*timeout_ms=*/5,
                          cpu, &m);
      for (int i = 0; i < kWarm; ++i) (void)be.Evaluate(obs);   // "warm-up"
      be.EnableBatcherTelemetry();                              // arm AFTER
      for (int i = 0; i < kScored; ++i) (void)be.Evaluate(obs); // "scored"
      BatcherTelemetry t = be.GetBatcherTelemetry();
      // The counters carry the warm-up rows: this is the contamination.
      assert(t.submitted_rows == static_cast<uint64_t>(kWarm + kScored));
      assert(t.single_row_calls == static_cast<uint64_t>(kWarm + kScored));
      std::cout << "  shared instance: submitted_rows=" << t.submitted_rows
                << " (warm " << kWarm << " + scored " << kScored
                << ") -- EnableBatcherTelemetry does NOT reset\n";
    }

    // (b) FRESH instance -- the fix. Warm-up ran on a throwaway that is gone.
    {
      std::shared_mutex mw;
      {
        BatchedEvaluator warm(model, /*target_batch_size=*/4, /*timeout_ms=*/5,
                              cpu, &mw);
        for (int i = 0; i < kWarm; ++i) (void)warm.Evaluate(obs);
      }  // throwaway destroyed here
      std::shared_mutex m;
      BatchedEvaluator be(model, /*target_batch_size=*/4, /*timeout_ms=*/5,
                          cpu, &m);
      be.EnableBatcherTelemetry();
      for (int i = 0; i < kScored; ++i) (void)be.Evaluate(obs);
      BatcherTelemetry t = be.GetBatcherTelemetry();
      // Scored counters see ONLY scored work: warm-up contributed zero.
      assert(t.submitted_rows == static_cast<uint64_t>(kScored));
      assert(t.single_row_calls == static_cast<uint64_t>(kScored));
      assert(t.physical_batches > 0);
      assert(t.physical_batches <= t.submitted_rows);
      std::cout << "  fresh instance:  submitted_rows=" << t.submitted_rows
                << " batches=" << t.physical_batches
                << " -- warm-up contributed ZERO scored rows\n";
    }

    // (c) On CUDA the coverage gate the sweep enforces must hold exactly.
    if (torch::cuda::is_available()) {
      torch::Device gpu(torch::kCUDA, 0);
      auto gmodel = std::make_shared<SharedDunePolicyValueNetImpl>(
          kObs, /*hidden_dim=*/32, /*action_dim=*/8, /*num_blocks=*/1);
      gmodel->eval();
      gmodel->to(gpu);
      std::shared_mutex mw;
      {
        BatchedEvaluator warm(gmodel, 4, 5, gpu, &mw);
        for (int i = 0; i < kWarm; ++i) (void)warm.Evaluate(obs);
      }
      std::shared_mutex m;
      BatchedEvaluator be(gmodel, 4, 5, gpu, &m);
      be.EnableBatcherTelemetry();
      for (int i = 0; i < kScored; ++i) (void)be.Evaluate(obs);
      BatcherTelemetry t = be.GetBatcherTelemetry();
      assert(t.submitted_rows == static_cast<uint64_t>(kScored));
      // S6: every physical batch timed. This is the assertion that fails if
      // warm-up ever shares the scored evaluator again.
      assert(t.device_timed_batches == t.physical_batches);
      assert(t.forward_ms > 0.0);
      std::cout << "  fresh instance (CUDA): batches=" << t.physical_batches
                << " timed=" << t.device_timed_batches
                << " -- full coverage holds\n";
    } else {
      std::cout << "  CUDA unavailable: coverage assertion skipped\n";
    }
    std::cout << "Test 10 Passed!\n\n";
  }

  // Test 11: a physical target boundary must never split one logical
  // EvaluateBatch() leaf group. Two singles followed by one four-row group
  // create a six-row queue against target four. The correct dispatch is 2+4;
  // the former min(target, queue_size) scoop dispatched 4+2 and split the
  // logical group across both forwards.
  {
    std::cout << "=== Test 11: leaf groups survive target boundaries ===\n";
    const int64_t kObs = 5580;
    torch::manual_seed(20277003);
    auto model = std::make_shared<SharedDunePolicyValueNetImpl>(
        kObs, /*hidden_dim=*/32, /*action_dim=*/8, /*num_blocks=*/1);
    model->eval();
    torch::Device cpu(torch::kCPU);
    model->to(cpu);
    std::shared_mutex m;
    BatchedEvaluator be(model, /*target_batch_size=*/4, /*timeout_ms=*/100,
                        cpu, &m);
    be.EnableBatcherTelemetry();
    std::vector<float> obs(kObs, 0.25f);
    std::thread single_a([&] { (void)be.Evaluate(obs); });
    std::thread single_b([&] { (void)be.Evaluate(obs); });
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    std::thread group([&] {
      (void)be.EvaluateBatch({obs, obs, obs, obs});
    });
    single_a.join();
    single_b.join();
    group.join();
    BatcherTelemetry t = be.GetBatcherTelemetry();
    assert(t.submitted_rows == 6);
    assert(t.physical_batches == 2);
    assert(t.max_batch_size == 4);
    assert(t.leaf_groups_split == 0);
    std::cout << "  rows=6 batches=2 max=4 split=0\n";
    std::cout << "Test 11 Passed!\n\n";
  }

  // Test 12: one logical group larger than the physical target is invalid.
  // It cannot be made whole by the runner and must never be admitted and
  // silently split across forwards.
  {
    std::cout << "=== Test 12: oversized leaf group is rejected ===\n";
    const int64_t kObs = 16;
    auto model = std::make_shared<SharedDunePolicyValueNetImpl>(
        kObs, /*hidden_dim=*/8, /*action_dim=*/4, /*num_blocks=*/1);
    model->eval();
    torch::Device cpu(torch::kCPU);
    model->to(cpu);
    assert(!BatchRejectsOversizedGroup(model, cpu, /*target=*/4,
                                       /*group_size=*/4, kObs));
    assert(BatchRejectsOversizedGroup(model, cpu, /*target=*/4,
                                      /*group_size=*/5, kObs));
    std::cout << "Test 12 Passed!\n\n";
  }

  // Test 13: CUDA BatchedEvaluator RunnerAsync mixed-request & slot-reorder stress
  {
    std::cout << "=== Test 13: CUDA BatchedEvaluator RunnerAsync mixed-request & slot-reorder stress ===\n";
    if (!torch::cuda::is_available()) {
      std::cout << "  CUDA unavailable: skipping Test 13\n";
    } else {
      torch::Device device(torch::kCUDA, 0);
      const int64_t kObs = 4096;
      const int64_t kHidden = 128;
      const int64_t kAction = 32;
      const int64_t kBlocks = 1;
      torch::manual_seed(20261102);
      auto model = std::make_shared<SharedDunePolicyValueNetImpl>(
          kObs, kHidden, kAction, kBlocks);
      model->eval();
      model->to(device);

      std::shared_mutex sync_mutex;
      BatchedEvaluator be(model,
                          /*target_batch_size=*/8,
                          /*timeout_ms=*/50,
                          device,
                          &sync_mutex,
                          /*logit_cap=*/0.0f,
                          /*device_synchronize=*/false);

      const int num_threads = 32;
      const int iters = 40;
      std::vector<Action> legals = {0, 3, 7, 12, 19, 25, 31};

      std::vector<std::vector<float>> obs(num_threads, std::vector<float>(kObs));
      for (int t = 0; t < num_threads; ++t) {
        for (int64_t i = 0; i < kObs; ++i) {
          obs[t][i] = std::sin(static_cast<float>(t * 100 + i + 1));
        }
      }

      std::vector<EvalResult> ref_full(num_threads);
      std::vector<CompactEvalResult> ref_compact(num_threads);
      std::vector<EvalResult> ref_val(num_threads);
      for (int t = 0; t < num_threads; ++t) {
        ref_full[t] = be.Evaluate(obs[t]);
        ref_compact[t] = be.EvaluateCompact(obs[t], legals);
        ref_val[t] = be.EvaluateValues(obs[t]);
      }

      std::atomic<int> mismatches{0};
      std::vector<std::thread> threads;
      threads.reserve(num_threads);
      for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back([&, t]() {
          for (int it = 0; it < iters; ++it) {
            const int mode = (t + it) % 4;
            if (mode == 0) {
              EvalResult r = be.Evaluate(obs[t]);
              bool ok = r.ok && (std::abs(r.value - ref_full[t].value) <= 1e-2);
              if (ok && r.logits.size() == ref_full[t].logits.size()) {
                for (size_t k = 0; k < r.logits.size(); ++k) {
                  if (std::abs(r.logits[k] - ref_full[t].logits[k]) > 1e-2) {
                    ok = false;
                    break;
                  }
                }
              } else {
                ok = false;
              }
              if (!ok) mismatches.fetch_add(1, std::memory_order_relaxed);
            } else if (mode == 1) {
              CompactEvalResult r = be.EvaluateCompact(obs[t], legals);
              bool ok = r.ok && (std::abs(r.value - ref_compact[t].value) <= 1e-2);
              if (ok && r.probabilities.size() == ref_compact[t].probabilities.size()) {
                for (size_t k = 0; k < r.probabilities.size(); ++k) {
                  if (std::abs(r.probabilities[k] - ref_compact[t].probabilities[k]) > 1e-2) {
                    ok = false;
                    break;
                  }
                }
              } else {
                ok = false;
              }
              if (!ok) mismatches.fetch_add(1, std::memory_order_relaxed);
            } else if (mode == 2) {
              EvalResult r = be.EvaluateValues(obs[t]);
              bool ok = r.ok && (std::abs(r.value - ref_val[t].value) <= 1e-2);
              if (!ok) mismatches.fetch_add(1, std::memory_order_relaxed);
            } else {
              std::vector<EvalResult> r_vec = be.EvaluateBatchValues({obs[t]});
              bool ok = (r_vec.size() == 1) && r_vec[0].ok &&
                        (std::abs(r_vec[0].value - ref_val[t].value) <= 1e-2);
              if (!ok) mismatches.fetch_add(1, std::memory_order_relaxed);
            }
          }
        });
      }
      for (auto& th : threads) th.join();

      assert(mismatches.load() == 0);

      // Sub-batch tail of 5 requests (< target_batch_size = 8) across
      // Evaluate, EvaluateCompact, and EvaluateValues
      std::atomic<int> tail_mismatches{0};
      std::vector<std::thread> tail_threads;
      tail_threads.reserve(5);
      tail_threads.emplace_back([&]() {
        EvalResult r = be.Evaluate(obs[0]);
        if (!r.ok || std::abs(r.value - ref_full[0].value) > 1e-2) tail_mismatches.fetch_add(1);
      });
      tail_threads.emplace_back([&]() {
        CompactEvalResult r = be.EvaluateCompact(obs[1], legals);
        if (!r.ok || std::abs(r.value - ref_compact[1].value) > 1e-2) tail_mismatches.fetch_add(1);
      });
      tail_threads.emplace_back([&]() {
        EvalResult r = be.EvaluateValues(obs[2]);
        if (!r.ok || std::abs(r.value - ref_val[2].value) > 1e-2) tail_mismatches.fetch_add(1);
      });
      tail_threads.emplace_back([&]() {
        EvalResult r = be.Evaluate(obs[3]);
        if (!r.ok || std::abs(r.value - ref_full[3].value) > 1e-2) tail_mismatches.fetch_add(1);
      });
      tail_threads.emplace_back([&]() {
        CompactEvalResult r = be.EvaluateCompact(obs[4], legals);
        if (!r.ok || std::abs(r.value - ref_compact[4].value) > 1e-2) tail_mismatches.fetch_add(1);
      });
      for (auto& th : tail_threads) th.join();
      assert(tail_mismatches.load() == 0);

      std::cout << "Test 13 Passed!\n\n";
    }
  }

  // Test 14: Forced-interleaving notification lifetime handshake
  {
    std::cout << "=== Test 14: Forced-interleaving notification lifetime handshake ===\n";

    std::atomic<bool> destroyed{false};
    auto custom_deleter = [&](std::atomic<uint32_t>* ptr) {
      destroyed.store(true, std::memory_order_release);
      delete ptr;
    };
    using FlagPtr = std::unique_ptr<std::atomic<uint32_t>, decltype(custom_deleter)>;

    std::mutex mtx;
    std::condition_variable cv_parked;
    bool waiter_parked = false;

    std::condition_variable cv_resume;
    bool allow_waiter_resume = false;

    std::atomic<uint32_t>* raw_flag_ptr = nullptr;

    std::thread waiter([&]() {
      FlagPtr flag(new std::atomic<uint32_t>(BatchedEvaluator::kWaitSpinning), custom_deleter);
      raw_flag_ptr = flag.get();

      BatchedEvaluator::WaitOnReadyFlagWithHook(*flag, [&]() {
        {
          std::lock_guard<std::mutex> lk(mtx);
          waiter_parked = true;
        }
        cv_parked.notify_one();

        {
          std::unique_lock<std::mutex> lk(mtx);
          cv_resume.wait(lk, [&]() { return allow_waiter_resume; });
        }
      });
    });

    {
      std::unique_lock<std::mutex> lk(mtx);
      cv_parked.wait(lk, [&]() { return waiter_parked; });
    }

    BatchedEvaluator::SignalReadyFlagWithHook(raw_flag_ptr, [&]() {
      {
        std::lock_guard<std::mutex> lk(mtx);
        allow_waiter_resume = true;
      }
      cv_resume.notify_one();

      std::this_thread::sleep_for(std::chrono::milliseconds(5));
      assert(!destroyed.load(std::memory_order_acquire));
    });

    waiter.join();
    assert(destroyed.load(std::memory_order_acquire));

    std::cout << "Test 14 Passed!\n\n";
  }

  // Test 15: Bug 7 group-size guard on EvaluateBatchValuesWithCompactPrior
  {
    std::cout << "=== Test 15: Bug 7 group-size guard on EvaluateBatchValuesWithCompactPrior ===\n";
    std::shared_mutex sync_mutex;
    open_spiel::BatchedEvaluator be_small(model, /*target_batch_size=*/2, /*timeout_ms=*/5, device, &sync_mutex);
    std::vector<std::vector<float>> obs_4(4, std::vector<float>(static_cast<size_t>(obs_size), 0.5f));
    std::vector<Action> legal_actions = {0, 1, 2};

    bool threw = false;
    open_spiel::SetErrorHandler([](const std::string& msg) { throw std::runtime_error(msg); });
    try {
      (void)be_small.EvaluateBatchValuesWithCompactPrior(obs_4, /*prior_index=*/0, legal_actions);
    } catch (const std::exception& e) {
      threw = std::string(e.what()).find("exceeds configured target") != std::string::npos;
    }
    open_spiel::SetErrorHandler(ExitingErrorHandler);
    assert(threw);

    // Verify target_batch_size = 4 succeeds with 4 observations
    open_spiel::BatchedEvaluator be_ok(model, /*target_batch_size=*/4, /*timeout_ms=*/5, device, &sync_mutex);
    auto res_ok = be_ok.EvaluateBatchValuesWithCompactPrior(obs_4, /*prior_index=*/0, legal_actions);
    assert(res_ok.first.size() == 4);
    assert(res_ok.second.actions.size() == legal_actions.size());

    std::cout << "Test 15 Passed!\n\n";
  }

  // Test 16: Bitwise parity max |diff| == 0.0 across Steps 3, 5, 7 + fail-closed v3 kTleilaxuPurchase
  {
    std::cout << "=== Test 16: Bitwise parity max |diff| == 0.0 across Steps 3, 5, 7 + fail-closed v3 kTleilaxuPurchase ===\n";
    const int64_t v4_input_dim = dune_imperium::kFullPublicInformationV4StateSize;
    auto model_v3 = std::make_shared<SharedDunePolicyValueNetImpl>(
        v4_input_dim, /*hidden_dim=*/2048, /*action_dim=*/2391, /*num_blocks=*/2,
        /*use_nonlinear=*/false, /*with_aux_heads=*/false, /*head_init_seed=*/0,
        /*with_semantic_scorer=*/true, /*enable_zone_encoder=*/false,
        /*semantic_descriptor_schema=*/dune_semantic::kDescriptorSchemaVersionV3);
    model_v3->eval();
    {
      torch::NoGradGuard no_grad;
      model_v3->semantic_scorer_->out_layer->weight.normal_(0.0, 0.2);
      model_v3->semantic_scorer_->out_layer_ext->weight.normal_(0.0, 0.2);
    }
    model_v3->to(device);

    std::shared_mutex sync_mutex;
    setenv("DUNE_DISABLE_PREWAIT_H2D", "1", 1);
    setenv("DUNE_DISABLE_CAP_SUBSAMPLE", "1", 1);
    setenv("DUNE_DISABLE_SELECTIVE_LOGITS_D2H", "1", 1);
    setenv("DUNE_DISABLE_SCORER_INPLACE", "1", 1);
    setenv("DUNE_DISABLE_TAIL_GRAPH", "1", 1);
    open_spiel::BatchedEvaluator be_legacy(
        model_v3, /*target_batch_size=*/8, /*timeout_ms=*/500, device, &sync_mutex,
        /*logit_cap=*/10.0f, /*device_synchronize=*/false, /*high_priority_stream=*/false,
        /*emit_batch_membership=*/false, /*rollout_amp=*/device.is_cuda(), /*allow_tf32=*/true);

    unsetenv("DUNE_DISABLE_PREWAIT_H2D");
    unsetenv("DUNE_DISABLE_CAP_SUBSAMPLE");
    unsetenv("DUNE_DISABLE_SELECTIVE_LOGITS_D2H");
    unsetenv("DUNE_DISABLE_SCORER_INPLACE");
    unsetenv("DUNE_DISABLE_TAIL_GRAPH");
    open_spiel::BatchedEvaluator be_opt(
        model_v3, /*target_batch_size=*/8, /*timeout_ms=*/500, device, &sync_mutex,
        /*logit_cap=*/10.0f, /*device_synchronize=*/false, /*high_priority_stream=*/false,
        /*emit_batch_membership=*/false, /*rollout_amp=*/device.is_cuda(), /*allow_tf32=*/true);

    auto make_cands = [](int batch_idx, int row) -> dune_semantic::CandidateActionData {
      dune_semantic::CandidateActionData cd;
      cd.actions = {10, 20, 30};
      cd.card_ids = {1, 2, 3};
      cd.space_ids = {4, 5, 6};
      cd.supported = {true, true, true};
      if (batch_idx == 1 && (row == 2 || row == 4)) {
        cd.roles = {dune_semantic::ActionRole::kPrimaryCard,
                    dune_semantic::ActionRole::kIndexedIntrigue,
                    dune_semantic::ActionRole::kBoardSpace};
      } else {
        cd.roles = {dune_semantic::ActionRole::kPrimaryCard,
                    dune_semantic::ActionRole::kMarketPurchase,
                    dune_semantic::ActionRole::kBoardSpace};
      }
      cd.features.assign(3 * dune_semantic::kSemanticFeatDim, 0.1f * (batch_idx + 1) + 0.01f * row);
      return cd;
    };

    struct BatchOutput {
      EvalResult eval_0;
      EvalResult eval_1;
      CompactEvalResult compact_2;
      CompactEvalResult compact_3;
      GreedyEvalResult greedy_4;
      GreedyEvalResult greedy_5;
      EvalResult val_6;
      EvalResult val_7;
    };

    auto run_batch = [&](open_spiel::BatchedEvaluator& be, int batch_idx) -> BatchOutput {
      BatchOutput out;
      std::vector<std::vector<float>> batch_obs(8, std::vector<float>(v4_input_dim));
      for (int r = 0; r < 8; ++r) {
        for (size_t i = 0; i < batch_obs[r].size(); ++i) {
          batch_obs[r][i] = std::sin(static_cast<float>((batch_idx + 1) * 1000 + r * 100 + i));
        }
      }
      std::vector<Action> legal_actions = {10, 20, 30, 40, 50};
      dune_semantic::CandidateActionData cd0 = make_cands(batch_idx, 0);
      dune_semantic::CandidateActionData cd2 = make_cands(batch_idx, 2);
      dune_semantic::CandidateActionData cd3 = make_cands(batch_idx, 3);
      dune_semantic::CandidateActionData cd4 = make_cands(batch_idx, 4);
      dune_semantic::CandidateActionData cd5 = make_cands(batch_idx, 5);

      std::vector<std::thread> ths;
      ths.reserve(8);
      ths.emplace_back([&]() { out.eval_0 = be.EvaluateWithActions(batch_obs[0], (batch_idx % 2 == 0) ? &cd0 : nullptr); });
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
      ths.emplace_back([&]() { out.eval_1 = be.Evaluate(batch_obs[1]); });
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
      ths.emplace_back([&]() { out.compact_2 = be.EvaluateCompact(batch_obs[2], legal_actions, &cd2); });
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
      ths.emplace_back([&]() { out.compact_3 = be.EvaluateCompact(batch_obs[3], legal_actions, &cd3); });
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
      ths.emplace_back([&]() { out.greedy_4 = be.EvaluateGreedyAction(batch_obs[4], legal_actions, &cd4); });
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
      ths.emplace_back([&]() { out.greedy_5 = be.EvaluateGreedyAction(batch_obs[5], legal_actions, &cd5); });
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
      ths.emplace_back([&]() { out.val_6 = be.EvaluateValues(batch_obs[6]); });
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
      ths.emplace_back([&]() { out.val_7 = be.EvaluateValues(batch_obs[7]); });

      for (auto& t : ths) t.join();
      return out;
    };

    for (int b = 0; b < 4; ++b) {
      BatchOutput legacy_out = run_batch(be_legacy, b);
      BatchOutput opt_out = run_batch(be_opt, b);

      assert(legacy_out.eval_0.ok && opt_out.eval_0.ok);
      AssertAlmostEqual(legacy_out.eval_0.value, opt_out.eval_0.value, 0.0);
      assert(legacy_out.eval_0.logits.size() == opt_out.eval_0.logits.size());
      for (size_t k = 0; k < legacy_out.eval_0.logits.size(); ++k) {
        AssertAlmostEqual(legacy_out.eval_0.logits[k], opt_out.eval_0.logits[k], 0.0);
      }

      assert(legacy_out.eval_1.ok && opt_out.eval_1.ok);
      AssertAlmostEqual(legacy_out.eval_1.value, opt_out.eval_1.value, 0.0);
      assert(legacy_out.eval_1.logits.size() == opt_out.eval_1.logits.size());
      for (size_t k = 0; k < legacy_out.eval_1.logits.size(); ++k) {
        AssertAlmostEqual(legacy_out.eval_1.logits[k], opt_out.eval_1.logits[k], 0.0);
      }

      assert(legacy_out.compact_2.ok && opt_out.compact_2.ok);
      AssertAlmostEqual(legacy_out.compact_2.value, opt_out.compact_2.value, 0.0);
      assert(legacy_out.compact_2.probabilities.size() == opt_out.compact_2.probabilities.size());
      for (size_t k = 0; k < legacy_out.compact_2.probabilities.size(); ++k) {
        AssertAlmostEqual(legacy_out.compact_2.probabilities[k], opt_out.compact_2.probabilities[k], 0.0);
      }

      assert(legacy_out.compact_3.ok && opt_out.compact_3.ok);
      AssertAlmostEqual(legacy_out.compact_3.value, opt_out.compact_3.value, 0.0);
      assert(legacy_out.compact_3.probabilities.size() == opt_out.compact_3.probabilities.size());
      for (size_t k = 0; k < legacy_out.compact_3.probabilities.size(); ++k) {
        AssertAlmostEqual(legacy_out.compact_3.probabilities[k], opt_out.compact_3.probabilities[k], 0.0);
      }

      assert(legacy_out.greedy_4.ok && opt_out.greedy_4.ok);
      assert(legacy_out.greedy_4.action == opt_out.greedy_4.action);

      assert(legacy_out.greedy_5.ok && opt_out.greedy_5.ok);
      assert(legacy_out.greedy_5.action == opt_out.greedy_5.action);

      assert(legacy_out.val_6.ok && opt_out.val_6.ok);
      AssertAlmostEqual(legacy_out.val_6.value, opt_out.val_6.value, 0.0);

      assert(legacy_out.val_7.ok && opt_out.val_7.ok);
      AssertAlmostEqual(legacy_out.val_7.value, opt_out.val_7.value, 0.0);
    }

    // Explicit exact tie test
    auto tie_model = std::make_shared<SharedDunePolicyValueNetImpl>(
        v4_input_dim, hidden_dim, action_size, /*num_blocks=*/1,
        /*use_nonlinear=*/false, /*with_aux_heads=*/false, /*head_init_seed=*/0,
        /*with_semantic_scorer=*/true, /*enable_zone_encoder=*/false,
        /*semantic_descriptor_schema=*/dune_semantic::kDescriptorSchemaVersionV3);
    tie_model->eval();
    {
      torch::NoGradGuard no_grad;
      tie_model->policy_head->weight.zero_();
      tie_model->policy_head->bias.fill_(-500.0f);
      tie_model->policy_head->bias[10] = 500.0f;
      tie_model->policy_head->bias[20] = 500.0f;
      tie_model->semantic_scorer_->out_layer->weight.zero_();
      if (tie_model->semantic_scorer_->out_layer->bias.defined()) {
        tie_model->semantic_scorer_->out_layer->bias.zero_();
      }
      tie_model->semantic_scorer_->out_layer_ext->weight.zero_();
      if (tie_model->semantic_scorer_->out_layer_ext->bias.defined()) {
        tie_model->semantic_scorer_->out_layer_ext->bias.zero_();
      }
    }
    tie_model->to(device);
    const char* prev_fast_greedy = std::getenv("DUNE_ENABLE_FAST_GREEDY");
    setenv("DUNE_ENABLE_FAST_GREEDY", "1", 1);
    auto tie_be = std::make_shared<open_spiel::BatchedEvaluator>(
        tie_model, /*target_batch_size=*/1, /*timeout_ms=*/5, device, &sync_mutex, /*logit_cap=*/10.0f);
    open_spiel::BatchedNNEvaluator tie_nn_eval(tie_be, 10.0f);
    assert(tie_nn_eval.FastGreedyEnabled());

    std::vector<Action> tie_legals = {10, 20};
    tie_nn_eval.SetFastGreedyEnabled(false);
    Action act_legacy = tie_nn_eval.GreedyActionWithLegals(*state, tie_legals);

    tie_nn_eval.SetFastGreedyEnabled(true);
    Action act_fast = tie_nn_eval.GreedyActionWithLegals(*state, tie_legals);

    assert(act_legacy == act_fast);
    assert(act_fast == 10 || act_fast == 20);
    if (device.is_cuda()) {
      assert(tie_be->GetBatcherTelemetry().greedy_action_ties >= 1);
    }
    if (prev_fast_greedy != nullptr) {
      setenv("DUNE_ENABLE_FAST_GREEDY", prev_fast_greedy, 1);
    } else {
      unsetenv("DUNE_ENABLE_FAST_GREEDY");
    }

    // Fail-closed H2 guard verification on v3 scorer with ActionRole::kTleilaxuPurchase
    dune_semantic::CandidateActionData h2_data;
    h2_data.actions = {91};
    h2_data.card_ids = {118};
    h2_data.space_ids = {0};
    h2_data.supported = {true};
    h2_data.roles = {dune_semantic::ActionRole::kTleilaxuPurchase};
    h2_data.features.assign(dune_semantic::kSemanticFeatDim, 0.5f);

    std::vector<const dune_semantic::CandidateActionData*> h2_batch = {&h2_data};
    torch::Tensor dummy_trunk = torch::zeros({1, 2048}, torch::TensorOptions().device(device).dtype(torch::kFloat32));
    torch::Tensor dummy_logits = torch::zeros({1, 2391}, torch::TensorOptions().device(device).dtype(torch::kFloat32));

    bool h2_threw = false;
    open_spiel::SetErrorHandler([](const std::string& msg) { throw std::runtime_error(msg); });
    try {
      dune_semantic::ApplySemanticScorerBatch(
          model_v3->semantic_scorer_, dummy_trunk, h2_batch, dummy_logits, device);
    } catch (const std::exception& e) {
      h2_threw = std::string(e.what()).find("enable_f_ is false") != std::string::npos;
    }
    open_spiel::SetErrorHandler(ExitingErrorHandler);
    assert(h2_threw);

    std::cout << "Test 16 Passed!\n\n";
  }

  std::cout << "All DuneNNEvaluator tests completed successfully!\n";
  return 0;
}
