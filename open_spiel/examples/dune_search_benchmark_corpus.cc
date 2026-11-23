#include "dune_search_benchmark_internal.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <random>
#include <shared_mutex>
#include <sstream>
#include <thread>

#include "dune_sha256.h"
#include "open_spiel/abseil-cpp/absl/flags/flag.h"
#include "open_spiel/abseil-cpp/absl/random/distributions.h"
#include "open_spiel/abseil-cpp/absl/strings/str_format.h"
#include "open_spiel/spiel_utils.h"
#include "open_spiel/utils/json.h"

namespace open_spiel {

// ===========================================================================
// WO-PERF-3: corpus-root parity + batcher-telemetry instrument.
//
// This is the permanent, opt-in equivalent of the temporary Phase-2 diagnostic
// extension. It reproduces the Phase-2 CPU shape exactly:
//
//   one frozen model
//     -> one shared BatchedEvaluator            (batched mode only)
//       -> one BatchedNNEvaluator per worker
//         -> one sequential DuneSearchSession per root
//
// The DIRECT arm is identical except each worker holds its own DuneNNEvaluator.
// The session, tree, opponent cache, simulation counters and RNG are per-root,
// so the ONLY difference between the two arms is how network rows are physically
// grouped -- which is the exact A/B the deferred GPU sweep needs.
//
// Search path: this drives the DuneSearchSession (kFixedSessionSimulations) --
// the same session/fresh path the PF collector uses -- NOT Path B
// (--use_session=false, the July-14 single-seat reference protocol). Session-
// only diagnostic fields (inherited_root_visits, decision_role, is_strategic,
// root_coverage) are populated by the session here; the parity evidence below
// keys on `actions` + `visit_counts`, which are populated on every root.
// ===========================================================================

std::unique_ptr<State> ReconstructCorpusRootState(
    const std::shared_ptr<const Game>& game,
    const std::vector<Action>& history) {
  auto state = game->NewInitialState();
  for (Action a : history) state->ApplyAction(a);
  return state;
}

int RunCorpusRootBenchmark(
    std::shared_ptr<const Game> game,
    std::shared_ptr<SharedDunePolicyValueNetImpl> search_model,
    torch::Device device) {
  const std::string mode = absl::GetFlag(FLAGS_evaluator_mode);
  if (mode != "direct" && mode != "batched") {
    std::cerr << "Error: --evaluator_mode must be 'direct' or 'batched' (got '"
              << mode << "').\n";
    return 1;
  }
  const bool batched = (mode == "batched");
  const int workers = std::max(1, absl::GetFlag(FLAGS_corpus_workers));
  const int target = std::max(1, absl::GetFlag(FLAGS_batch_target_rows));
  const int timeout_ms = std::max(0, absl::GetFlag(FLAGS_batch_max_queue_wait_ms));
  const int root_count = absl::GetFlag(FLAGS_corpus_root_count);
  const float candidate_logit_cap =
      static_cast<float>(absl::GetFlag(FLAGS_candidate_logit_cap));
  const int max_sims = absl::GetFlag(FLAGS_max_simulations);
  const int seed = absl::GetFlag(FLAGS_seed);
  const double puct_c = absl::GetFlag(FLAGS_puct_c);
  const double prior_temp = absl::GetFlag(FLAGS_root_prior_temperature);
  const double utility_divisor = absl::GetFlag(FLAGS_utility_divisor);

  DuneDecisionRole expected_role = DuneDecisionRole::kAgentPrimary;
  const std::string role_flag = absl::GetFlag(FLAGS_corpus_root_role);
  if (role_flag == "purchase") {
    expected_role = DuneDecisionRole::kPurchase;
  } else if (role_flag == "combat") {
    expected_role = DuneDecisionRole::kCombatIntrigue;
  } else if (role_flag != "primary") {
    std::cerr << "Error: --corpus_root_role must be primary|purchase|combat.\n";
    return 1;
  }

  const std::string corpus_path = absl::GetFlag(FLAGS_corpus_root_path);
  std::ifstream f(corpus_path);
  if (!f) {
    std::cerr << "Error: failed to open --corpus_root_path: " << corpus_path << "\n";
    return 1;
  }
  std::string content((std::istreambuf_iterator<char>(f)),
                      std::istreambuf_iterator<char>());
  auto parsed = open_spiel::json::FromString(content);
  if (!parsed) {
    std::cerr << "Error: failed to parse corpus JSON.\n";
    return 1;
  }
  auto arr = parsed.value().GetArray();

  // Collect the LEADING strategic roots in corpus order (matches the Phase-2
  // "first 14 primary/strategic roots" root set).
  std::vector<CorpusRootRecord> roots;
  for (size_t i = 0; i < arr.size(); ++i) {
    auto obj = arr[i].GetObject();
    CorpusRootRecord cs;
    cs.corpus_idx = static_cast<int>(i);
    cs.category = obj.at("category").GetString();
    cs.player = static_cast<Player>(obj.at("player").GetInt());
    cs.round = obj.at("round").GetInt();
    auto seed_it = obj.find("seed");
    if (seed_it != obj.end()) {
      cs.seed = static_cast<uint64_t>(seed_it->second.GetInt());
    }
    auto legals_it = obj.find("legal_actions");
    if (legals_it != obj.end()) {
      for (const auto& a : legals_it->second.GetArray()) {
        cs.recorded_legals.push_back(static_cast<Action>(a.GetInt()));
      }
    }
    for (const auto& a : obj.at("history").GetArray()) {
      cs.history.push_back(static_cast<Action>(a.GetInt()));
    }
    auto st = ReconstructCorpusRootState(game, cs.history);
    // Native validation against recorded player and legal actions
    SPIEL_CHECK_EQ(st->CurrentPlayer(), cs.player);
    if (!cs.recorded_legals.empty()) {
      std::vector<Action> state_legals = st->LegalActions();
      std::sort(state_legals.begin(), state_legals.end());
      std::vector<Action> rec_legals = cs.recorded_legals;
      std::sort(rec_legals.begin(), rec_legals.end());
      if (state_legals != rec_legals) {
        SpielFatalError(absl::StrFormat(
            "Native validation failed for root %d (corpus_idx %d): legal actions mismatch! Reconstructed has %d legals, recorded has %d legals.",
            static_cast<int>(roots.size()), cs.corpus_idx,
            state_legals.size(), rec_legals.size()));
      }
    }
    DuneDecisionRole role =
        ClassifyDuneDecisionRole(*st, st->CurrentPlayer(), false);
    if (role == expected_role) {
      roots.push_back(std::move(cs));
      if (root_count > 0 && static_cast<int>(roots.size()) >= root_count) break;
    }
  }
  if (root_count > 0 && static_cast<int>(roots.size()) < root_count) {
    std::cerr << "Error: found only " << roots.size() << " '" << role_flag
              << "' roots, requested " << root_count << ".\n";
    return 1;
  }
  std::cout << "  native_validation: PASS (" << roots.size()
            << " roots verified against recorded player and legal actions)\n";

  // WO-PERF-R3 (deliverable 2, WO-3 review F2 forward-fix): self-record the
  // selected root set so every run carries its own root-set identity. The
  // Phase-2 root-set hash recipe was lost and could not be reproduced; this
  // prevents a recurrence. The serialization is deliberately simple, stable, and
  // documented here so a future run can recompute and compare it by hand:
  //
  //   line 1:            "<root_count>"
  //   one line per root, IN SELECTION ORDER:
  //                      "<corpus_idx>|<category>|<player>|<round>|<a0,a1,...,ak>"
  //
  // The fields are exactly the identifying fields the driver reads from
  // data/dune_diagnostic_corpus.json to select and reconstruct each root
  // (corpus_idx pins provenance; history is the reconstruction input; category/
  // player/round pin the decision). Every line is '\n'-terminated. The root-set
  // hash is the sha256 of that byte string. It is emitted to stdout (below), to
  // each per-root JSONL record, and to the batcher-telemetry JSON.
  std::string root_set_serialization;
  {
    std::ostringstream ss;
    ss << roots.size() << "\n";
    for (const auto& cs : roots) {
      ss << cs.corpus_idx << "|" << cs.category << "|"
         << static_cast<int>(cs.player) << "|" << cs.round << "|";
      for (size_t k = 0; k < cs.history.size(); ++k) {
        if (k != 0) ss << ",";
        ss << static_cast<long long>(cs.history[k]);
      }
      ss << "\n";
    }
    root_set_serialization = ss.str();
  }
  const std::string root_set_sha256 =
      open_spiel::ComputeStringSHA256(root_set_serialization);

  std::cout << "\n=== WO-PERF-3 corpus-root benchmark ===\n"
            << "  evaluator_mode:  " << mode << "\n"
            << "  roots:           " << roots.size() << " (" << role_flag << ")\n"
            << "  corpus:          " << corpus_path << "\n"
            << "  workers:         " << workers << "\n"
            << "  max_simulations: " << max_sims << "\n";
  if (batched) {
    std::cout << "  batch_target:    " << target << " rows\n"
              << "  max_queue_wait:  " << timeout_ms << " ms\n";
  }
  std::cout << "  checkpoint:      " << absl::GetFlag(FLAGS_model_checkpoint)
            << "\n";
  // Leader provenance is printed ALWAYS, including when off, so a run log
  // records which Leader semantics produced it rather than leaving a reader to
  // infer it from the absence of a line.
  std::cout << "  leader_search:   "
            << (absl::GetFlag(FLAGS_search_leader_draft) ? "ON" : "off")
            << " sims=" << absl::GetFlag(FLAGS_leader_draft_simulations)
            << " mass_only="
            << (absl::GetFlag(FLAGS_leader_mass_only_coverage) ? "true" : "false")
            << "\n";
  std::cout << "  root_set_count:  " << roots.size() << "\n"
            << "  root_set_sha256: " << root_set_sha256 << "\n\n";

  std::shared_mutex model_mutex;

  // --- Registered warm-up, on a THROWAWAY evaluator ------------------------
  //
  // The warm-up MUST NOT share the evaluator the scored run uses.
  // EnableBatcherTelemetry() arms the detailed per-batch sections; it does NOT
  // reset the always-on counters. So a warm-up that ran through the scored
  // BatchedEvaluator would leave total_requests, physical_batches,
  // single_row_calls, group_rows and the timeout counts already advanced before
  // a single scored row was submitted. Three checks break at once:
  //
  //   * group_rows would exceed the leaf-evaluation delta, because that delta is
  //     snapshotted after warm-up while group_rows still carries warm-up rows
  //     -- the B3 crosscheck fails;
  //   * device_timed_batches < physical_batches, because timing arms after
  //     warm-up while the batch counter does not reset -- the S6 coverage gate
  //     fails;
  //   * so the FIRST batched arm of every sweep fails S2/S6, and would look
  //     like an instrument fault rather than a warm-up accounting error.
  //
  // The temporary evaluator is therefore destroyed before the scored one is
  // constructed, which also releases its pinned staging buffers rather than
  // holding them while the scored evaluator allocates its own.
  const std::string controller_mode_flag = absl::GetFlag(FLAGS_corpus_controller);
  if (controller_mode_flag != "session" && controller_mode_flag != "fresh") {
    std::cerr << "Error: --corpus_controller must be 'session' or 'fresh' (got '"
              << controller_mode_flag << "').\n";
    return 1;
  }
  const bool fresh_controller = (controller_mode_flag == "fresh");

  const int warmup_n = absl::GetFlag(FLAGS_corpus_warmup_searches);
  if (warmup_n > 0 && !roots.empty()) {
    torch::InferenceMode inference_guard;
    std::shared_ptr<open_spiel::BatchedEvaluator> warm_batched;
    std::shared_ptr<algorithms::Evaluator> wu_eval;
    if (batched) {
      warm_batched = std::make_shared<open_spiel::BatchedEvaluator>(
          search_model, target, timeout_ms, device, &model_mutex,
          candidate_logit_cap, /*device_synchronize=*/true,
          /*high_priority_stream=*/false, /*emit_batch_membership=*/false,
          /*rollout_amp=*/true, /*allow_tf32=*/false);
      wu_eval = std::make_shared<open_spiel::BatchedNNEvaluator>(
          warm_batched, candidate_logit_cap);
    } else {
      wu_eval = std::make_shared<open_spiel::DuneNNEvaluator>(
          search_model, device, candidate_logit_cap);
    }
    for (int w = 0; w < warmup_n; ++w) {
      const CorpusRootRecord& cs = roots[w % roots.size()];
      auto st = ReconstructCorpusRootState(game, cs.history);
      if (fresh_controller) {
        DuneSearchConfig wcfg;
        wcfg.max_simulations = max_sims;
        wcfg.relative_time_budget_ms = std::numeric_limits<double>::infinity();
        wcfg.max_nodes = absl::GetFlag(FLAGS_max_nodes);
        wcfg.puct_c = puct_c;
        wcfg.opponent_mode = SearchOpponentMode::kMaxN;
        wcfg.temperature = 0.0;
        wcfg.opponent_temperature = 0.0;
        wcfg.utility_divisor = utility_divisor;
        wcfg.min_visit_threshold = 2;
        wcfg.covered_prior_threshold = 0.50;
        wcfg.seed = seed + 900000 + w;
        wcfg.final_policy_type = DuneISMCTSFinalPolicyType::kNormalizedVisitCount;
        wcfg.dirichlet_epsilon = 0.0;
        wcfg.use_observation_string = true;
        wcfg.verbose_diagnostics = false;
        wcfg.check_strategic_state = false;
        wcfg.root_prior_temperature = prior_temp;
        wcfg.model_checkpoint_path = absl::GetFlag(FLAGS_model_checkpoint);
        DunePUCTISMCTSBot wbot(wcfg, wu_eval);
        (void)wbot.Step(*st);
      } else {
        DuneSearchConfig wcfg;
        wcfg.max_simulations = max_sims;
        wcfg.relative_time_budget_ms = std::numeric_limits<double>::infinity();
        wcfg.puct_c = puct_c;
        wcfg.opponent_mode = SearchOpponentMode::kPolicy;
        wcfg.opponent_temperature = 1.0;
        wcfg.temperature = 0.0;
        wcfg.utility_divisor = utility_divisor;
        wcfg.seed = seed + 900000 + w;
        wcfg.root_prior_temperature = prior_temp;
        wcfg.fixed_session_limit = max_sims;
        wcfg.fixed_continuation_reserve =
            absl::GetFlag(FLAGS_fixed_continuation_reserve);
        wcfg.purchase_combat_budget = absl::GetFlag(FLAGS_purchase_combat_budget);
        wcfg.model_checkpoint_path = absl::GetFlag(FLAGS_model_checkpoint);
        DuneSearchSession wsession(wcfg, wu_eval,
                                   DuneSearchBudgetMode::kFixedSessionSimulations);
        (void)wsession.Search(*st);
      }
    }
    // Explicit teardown before the scored evaluator exists.
    wu_eval.reset();
    warm_batched.reset();
    std::cout << "  warm-up:         " << warmup_n
              << " discarded searches on a throwaway evaluator\n";
  }

  // The shared coordinator (batched mode only) -- constructed FRESH, after
  // warm-up, so every counter it reports starts at zero for the scored run.
  std::shared_ptr<open_spiel::BatchedEvaluator> batched_eval;
  if (batched) {
    batched_eval = std::make_shared<open_spiel::BatchedEvaluator>(
        search_model, target, timeout_ms, device, &model_mutex,
        candidate_logit_cap, /*device_synchronize=*/true,
        /*high_priority_stream=*/false, /*emit_batch_membership=*/false,
        /*rollout_amp=*/true, /*allow_tf32=*/false);
    // R2: the corpus driver is the one consumer that reads the detailed batcher
    // telemetry, so it alone arms the per-batch bookkeeping.
    batched_eval->EnableBatcherTelemetry();
  }

  // Snapshot the evaluator-side leaf-value counter so its DELTA independently
  // cross-checks the batcher's group_rows (B3).
  const uint64_t leaf_evals_before =
      open_spiel::DuneNNEvaluator::global_num_leaf_evaluations.load();

  std::vector<CorpusRootOutcome> outcomes(roots.size());
  std::atomic<int> next_idx{0};
  std::atomic<uint64_t> benchmark_raw_prior_calls{0};

  auto worker_fn = [&]() {
    torch::InferenceMode inference_guard;
    // One evaluator per worker. In batched mode every worker's evaluator wraps
    // the SAME shared coordinator; in direct mode each is an independent direct
    // evaluator over the shared model. Reused across roots this worker pulls.
    std::shared_ptr<algorithms::Evaluator> evaluator;
    if (batched) {
      evaluator = std::make_shared<open_spiel::BatchedNNEvaluator>(
          batched_eval, candidate_logit_cap);
    } else {
      evaluator = std::make_shared<open_spiel::DuneNNEvaluator>(
          search_model, device, candidate_logit_cap);
    }

    while (true) {
      int idx = next_idx.fetch_add(1);
      if (idx >= static_cast<int>(roots.size())) break;
      const auto root_t0 = std::chrono::steady_clock::now();
      const CorpusRootRecord& cs = roots[idx];
      auto state = ReconstructCorpusRootState(game, cs.history);

      if (fresh_controller) {
        DuneSearchConfig cfg;
        cfg.max_simulations = max_sims;
        cfg.relative_time_budget_ms = std::numeric_limits<double>::infinity();
        cfg.max_nodes = absl::GetFlag(FLAGS_max_nodes);
        cfg.puct_c = puct_c;
        cfg.opponent_mode = SearchOpponentMode::kMaxN;
        cfg.temperature = 0.0;
        cfg.opponent_temperature = 0.0;
        cfg.utility_divisor = utility_divisor;
        cfg.min_visit_threshold = 2;
        cfg.covered_prior_threshold = 0.50;
        cfg.seed = (cs.seed != 0) ? cs.seed : (seed + 200000 + 1000 * idx);
        cfg.final_policy_type = DuneISMCTSFinalPolicyType::kNormalizedVisitCount;
        cfg.dirichlet_epsilon = 0.0;
        cfg.use_observation_string = true;
        cfg.verbose_diagnostics = false;
        cfg.check_strategic_state = false;
        cfg.root_prior_temperature = prior_temp;
        cfg.model_checkpoint_path = absl::GetFlag(FLAGS_model_checkpoint);

        DunePUCTISMCTSBot bot(cfg, evaluator);
        Action chosen_action = bot.Step(*state);
        DuneSearchResult sr = bot.GetLastSearchResult();
        SPIEL_CHECK_EQ(sr.simulations_completed, max_sims);

        CorpusRootOutcome oc;
        oc.root_idx = idx;
        oc.corpus_idx = cs.corpus_idx;
        oc.player = cs.player;
        oc.round = cs.round;
        oc.selected_action = chosen_action;
        oc.actions = sr.diagnostics.actions;
        oc.visit_counts = sr.diagnostics.visit_counts;
        oc.q_values = sr.diagnostics.q_values;
        oc.priors = sr.diagnostics.priors;
        oc.raw_priors = sr.diagnostics.raw_priors;
        oc.total_root_visits = sr.diagnostics.total_root_visits;
        oc.num_covered_actions = sr.diagnostics.num_covered_actions;
        oc.covered_prior_mass = sr.diagnostics.covered_prior_mass;
        oc.used_fallback = sr.used_fallback;
        oc.fallback_reason = sr.fallback_reason;
        oc.root_value = sr.diagnostics.root_value;
        oc.simulations_completed = sr.simulations_completed;
        oc.elapsed_time_ms = sr.elapsed_time_ms;
        oc.time_nn_inference_ms = sr.diagnostics.time_nn_inference_ms;
        oc.time_tree_engine_ms = sr.diagnostics.time_tree_engine_ms;
        oc.time_sample_root_ms = sr.diagnostics.time_sample_root_ms;
        oc.time_other_overhead_ms = sr.diagnostics.time_other_overhead_ms;

        Action a_raw = kInvalidAction;
        double best_p = -1.0;
        for (size_t k = 0; k < oc.actions.size() && k < oc.raw_priors.size(); ++k) {
          if (oc.raw_priors[k] > best_p) {
            best_p = oc.raw_priors[k];
            a_raw = oc.actions[k];
          }
        }
        oc.raw_argmax_action = a_raw;
        oc.root_wall_ms = std::chrono::duration<double, std::milli>(
                              std::chrono::steady_clock::now() - root_t0).count();
        oc.populated = true;
        outcomes[idx] = std::move(oc);
      } else {
        // Out-of-MCTS raw prior #1 (pre-search). Mirrors the first of the two
        // extra one-row raw-prior calls per root in the Phase-2 accounting.
        ActionsAndProbs ppo_prior = evaluator->Prior(*state);
        benchmark_raw_prior_calls.fetch_add(1, std::memory_order_relaxed);
        Action a_raw = kInvalidAction;
        double best_p = -1.0;
        for (const auto& ap : ppo_prior) {
          if (ap.second > best_p) { best_p = ap.second; a_raw = ap.first; }
        }

        // Config mirrors dune_search_calibration.cc (the Phase-2 reference): the
        // frozen policy-opponent workload at a fixed session simulation budget.
        DuneSearchConfig cfg;
        cfg.max_simulations = max_sims;
        cfg.relative_time_budget_ms = std::numeric_limits<double>::infinity();
        cfg.puct_c = puct_c;
        cfg.opponent_mode = SearchOpponentMode::kPolicy;
        cfg.opponent_temperature = 1.0;
        cfg.temperature = 0.0;
        cfg.utility_divisor = utility_divisor;
        cfg.seed = seed + 200000 + 1000 * idx;
        cfg.root_prior_temperature = prior_temp;
        cfg.fixed_session_limit = max_sims;
        cfg.fixed_continuation_reserve =
            absl::GetFlag(FLAGS_fixed_continuation_reserve);
        cfg.purchase_combat_budget = absl::GetFlag(FLAGS_purchase_combat_budget);
        cfg.model_checkpoint_path = absl::GetFlag(FLAGS_model_checkpoint);

        DuneSearchSession session(cfg, evaluator,
                                  DuneSearchBudgetMode::kFixedSessionSimulations);
        DuneSearchResult sr = session.Search(*state);

        CorpusRootOutcome oc;
        oc.root_idx = idx;
        oc.corpus_idx = cs.corpus_idx;
        oc.player = cs.player;
        oc.round = cs.round;
        oc.raw_argmax_action = a_raw;
        oc.actions = sr.diagnostics.actions;
        oc.visit_counts = sr.diagnostics.visit_counts;
        oc.total_root_visits = sr.diagnostics.total_root_visits;
        oc.simulations_completed = sr.simulations_completed;
        oc.elapsed_time_ms = sr.elapsed_time_ms;

        // Controller selection + commit. SelectControllerAction issues the second
        // out-of-MCTS raw-prior call per root (dune_search_session.cc:679).
        std::mt19937 step_rng(cfg.seed);
        double r_val = absl::Uniform(step_rng, 0.0, 1.0);
        ControllerDecision decision =
            session.SelectControllerAction(*state, sr, r_val);
        if (sr.raw_policy.empty()) {
          benchmark_raw_prior_calls.fetch_add(1, std::memory_order_relaxed);
        }
        DuneSearchResult committed = session.CommitAction(decision);
        oc.selected_action = committed.diagnostics.selected_action;
        oc.root_wall_ms = std::chrono::duration<double, std::milli>(
                              std::chrono::steady_clock::now() - root_t0).count();
        oc.populated = true;

        outcomes[idx] = std::move(oc);
      }
    }
  };

  auto wall_start = std::chrono::steady_clock::now();
  // Absolute wall-clock bounds of the SCORED window, so an external sampler
  // (nvidia-smi, /proc/stat) can be cropped to exactly this interval. Without
  // it those samplers necessarily span model load and warm-up as well, because
  // they must be running before the process starts.
  const double scored_start_unix =
      std::chrono::duration<double>(
          std::chrono::system_clock::now().time_since_epoch()).count();
  std::vector<std::thread> threads;
  threads.reserve(workers);
  for (int i = 0; i < workers; ++i) threads.emplace_back(worker_fn);
  for (auto& t : threads) t.join();
  double wall_s =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - wall_start)
          .count();
  const double scored_end_unix =
      std::chrono::duration<double>(
          std::chrono::system_clock::now().time_since_epoch()).count();
  std::cout << absl::StrFormat("scored_window_unix: %.3f %.3f\n",
                               scored_start_unix, scored_end_unix);

  const uint64_t leaf_evals_delta =
      open_spiel::DuneNNEvaluator::global_num_leaf_evaluations.load() -
      leaf_evals_before;

  // --- Per-root output + parity summary -----------------------------------
  int total_sims = 0;
  for (const auto& oc : outcomes) total_sims += oc.simulations_completed;
  // --- Per-root latency percentiles (the registered latency bound) ---------
  // Computed over root_wall_ms, the end-to-end per-root figure, because that is
  // what a consumer of the search waits for. Reported for every arm, direct and
  // batched alike, so the bound is checkable without a batcher telemetry file.
  double lat_p50 = 0.0, lat_p95 = 0.0, lat_p99 = 0.0, lat_max = 0.0;
  {
    std::vector<double> lat;
    lat.reserve(outcomes.size());
    for (const auto& oc : outcomes) {
      if (oc.populated) lat.push_back(oc.root_wall_ms);
    }
    if (!lat.empty()) {
      std::sort(lat.begin(), lat.end());
      auto pct = [&lat](double q) {
        size_t i = static_cast<size_t>(std::round(q * (lat.size() - 1)));
        return lat[std::min(i, lat.size() - 1)];
      };
      lat_p50 = pct(0.50);
      lat_p95 = pct(0.95);
      lat_p99 = pct(0.99);
      lat_max = lat.back();
      std::cout << "Per-root latency (root_wall_ms): P50 "
                << absl::StrFormat("%.2f", lat_p50) << "  P95 "
                << absl::StrFormat("%.2f", lat_p95) << "  P99 "
                << absl::StrFormat("%.2f", lat_p99) << "  max "
                << absl::StrFormat("%.2f", lat_max) << "  (n=" << lat.size()
                << ")\n";
    }
  }

  std::cout << "Completed " << outcomes.size() << " roots, " << total_sims
            << " simulations in " << absl::StrFormat("%.3f", wall_s) << "s ("
            << absl::StrFormat("%.3f", total_sims / std::max(1e-9, wall_s))
            << " sim/s).\n\n";

  std::cout << "Per-root results (root_idx corpus_idx selected_action "
               "total_root_visits sims):\n";
  for (const auto& oc : outcomes) {
    std::cout << absl::StrFormat(
        "  root %2d  corpus %4d  action %5d  visits %5d  sims %4d\n",
        oc.root_idx, oc.corpus_idx, static_cast<int>(oc.selected_action),
        oc.total_root_visits, oc.simulations_completed);
  }

  const std::string jsonl_path = absl::GetFlag(FLAGS_corpus_root_jsonl_path);
  if (!jsonl_path.empty()) {
    std::ofstream out(jsonl_path);
    if (!out) {
      std::cerr << "Warning: could not open --corpus_root_jsonl_path: "
                << jsonl_path << "\n";
    } else {
      for (const auto& oc : outcomes) {
        open_spiel::json::Object o;
        o["evaluator_mode"] = mode;
        // WO-PERF-R3: every record self-records the root-set identity so a
        // direct-mode run (which emits no telemetry JSON) still carries it.
        o["root_set_sha256"] = root_set_sha256;
        o["root_set_count"] = static_cast<int64_t>(roots.size());
        o["root_idx"] = static_cast<int64_t>(oc.root_idx);
        o["corpus_idx"] = static_cast<int64_t>(oc.corpus_idx);
        o["player"] = static_cast<int64_t>(oc.player);
        o["round"] = static_cast<int64_t>(oc.round);
        o["selected_action"] = static_cast<int64_t>(oc.selected_action);
        o["raw_argmax_action"] = static_cast<int64_t>(oc.raw_argmax_action);
        o["total_root_visits"] = static_cast<int64_t>(oc.total_root_visits);
        o["simulations_completed"] =
            static_cast<int64_t>(oc.simulations_completed);
        o["root_wall_ms"] = oc.root_wall_ms;
        o["elapsed_time_ms"] = oc.elapsed_time_ms;
        o["time_nn_inference_ms"] = oc.time_nn_inference_ms;
        o["time_tree_engine_ms"] = oc.time_tree_engine_ms;
        o["time_sample_root_ms"] = oc.time_sample_root_ms;
        o["time_other_overhead_ms"] = oc.time_other_overhead_ms;
        o["num_covered_actions"] = static_cast<int64_t>(oc.num_covered_actions);
        o["covered_prior_mass"] = oc.covered_prior_mass;
        o["used_fallback"] = oc.used_fallback;
        o["fallback_reason"] = oc.fallback_reason;
        o["root_value"] = oc.root_value;
        open_spiel::json::Array acts;
        for (Action a : oc.actions) acts.push_back(static_cast<int64_t>(a));
        o["actions"] = acts;
        open_spiel::json::Array vis;
        for (int v : oc.visit_counts) vis.push_back(static_cast<int64_t>(v));
        o["visit_counts"] = vis;
        open_spiel::json::Array qvals;
        for (double q : oc.q_values) qvals.push_back(q);
        o["q_values"] = qvals;
        open_spiel::json::Array priors;
        for (double p : oc.priors) priors.push_back(p);
        o["priors"] = priors;
        open_spiel::json::Array raw_priors;
        for (double p : oc.raw_priors) raw_priors.push_back(p);
        o["raw_priors"] = raw_priors;
        out << open_spiel::json::ToString(o, /*wrap=*/false) << "\n";
      }
      std::cout << "\nWrote per-root JSONL to " << jsonl_path << "\n";
    }
  }

  // --- Batcher telemetry (batched mode) -----------------------------------
  if (batched) {
    open_spiel::BatcherTelemetry t = batched_eval->GetBatcherTelemetry();
    const uint64_t raw_priors = benchmark_raw_prior_calls.load();
    // single_row_calls = MCTS opponent Prior calls + benchmark raw-prior calls.
    const int64_t opponent_calls =
        static_cast<int64_t>(t.single_row_calls) -
        static_cast<int64_t>(raw_priors);
    const bool rows_identity =
        (t.submitted_rows == t.single_row_calls + t.group_rows);
    const bool leaf_rows_crosscheck = (t.group_rows == leaf_evals_delta);

    std::cout << "\n=== WO-PERF-3 batcher telemetry ===\n";
    std::cout << absl::StrFormat("  submitted_rows:        %d\n", t.submitted_rows);
    std::cout << absl::StrFormat("  physical_batches:      %d\n", t.physical_batches);
    std::cout << absl::StrFormat("  mean_batch_size:       %.4f\n", t.mean_batch_size);
    std::cout << absl::StrFormat("  p50 / p95 / max batch: %d / %d / %d\n",
                                 t.p50_batch_size, t.p95_batch_size, t.max_batch_size);
    std::cout << absl::StrFormat("  target_occupancy:      %.4f (target %d rows)\n",
                                 t.target_occupancy, t.target_batch_size);
    std::cout << absl::StrFormat("  timeout_flush_batches: %d\n", t.timeout_flush_batches);
    std::cout << absl::StrFormat("  single_row_calls:      %d (opponent Prior + %d raw prior)\n",
                                 t.single_row_calls, raw_priors);
    std::cout << absl::StrFormat("  group_calls (leaf):    %d\n", t.group_calls);
    std::cout << absl::StrFormat("  group_rows (leaf):     %d\n", t.group_rows);
    std::cout << absl::StrFormat("  leaf_groups_split:     %d\n", t.leaf_groups_split);
    std::cout << absl::StrFormat("  queue-wait single:     mean %.4f ms  max %.4f ms  (n=%d)\n",
                                 t.single_wait_ms_mean, t.single_wait_ms_max, t.single_wait_n);
    std::cout << absl::StrFormat("  queue-wait leaf:       mean %.4f ms  max %.4f ms  (n=%d)\n",
                                 t.group_wait_ms_mean, t.group_wait_ms_max, t.group_wait_n);
    std::cout << "\n  Self-consistency (B3):\n";
    std::cout << absl::StrFormat(
        "    rows == single_row_calls + group_rows : %s (%d == %d + %d)\n",
        rows_identity ? "OK" : "MISMATCH", t.submitted_rows, t.single_row_calls,
        t.group_rows);
    std::cout << absl::StrFormat(
        "    group_rows == leaf-value records      : %s (%d == %d)\n",
        leaf_rows_crosscheck ? "OK" : "MISMATCH", t.group_rows, leaf_evals_delta);
    std::cout << absl::StrFormat(
        "    implied MCTS opponent Prior calls     : %d\n"
        "    documented extra raw-prior calls      : %d (2 per root)\n",
        opponent_calls, raw_priors);

    const std::string tel_path = absl::GetFlag(FLAGS_batcher_telemetry_json_path);
    if (!tel_path.empty()) {
      open_spiel::json::Object o;
      o["evaluator_mode"] = mode;
      // WO-PERF-R3 (deliverable 2): root-set identity, self-recorded.
      o["root_set_sha256"] = root_set_sha256;
      // Latency percentiles and the warm-up count, so a consumer of this file
      // can check the registered latency bound and confirm the arm warmed up
      // without parsing stdout. (An earlier revision of the registration
      // claimed these were emitted here when they were not.)
      o["latency_p50_ms"] = lat_p50;
      o["latency_p95_ms"] = lat_p95;
      o["latency_p99_ms"] = lat_p99;
      o["latency_max_ms"] = lat_max;
      o["warmup_searches"] = static_cast<int64_t>(warmup_n);
      o["scored_window_unix_start"] = scored_start_unix;
      o["scored_window_unix_end"] = scored_end_unix;
      o["root_set_count"] = static_cast<int64_t>(roots.size());
      o["roots"] = static_cast<int64_t>(roots.size());
      o["workers"] = static_cast<int64_t>(workers);
      o["max_simulations"] = static_cast<int64_t>(max_sims);
      o["batch_target_rows"] = static_cast<int64_t>(t.target_batch_size);
      o["max_queue_wait_ms"] = static_cast<int64_t>(t.timeout_ms);
      o["submitted_rows"] = static_cast<int64_t>(t.submitted_rows);
      o["physical_batches"] = static_cast<int64_t>(t.physical_batches);
      o["mean_batch_size"] = t.mean_batch_size;
      o["p50_batch_size"] = static_cast<int64_t>(t.p50_batch_size);
      o["p95_batch_size"] = static_cast<int64_t>(t.p95_batch_size);
      o["max_batch_size"] = static_cast<int64_t>(t.max_batch_size);
      o["target_occupancy"] = t.target_occupancy;
      o["timeout_flush_batches"] = static_cast<int64_t>(t.timeout_flush_batches);
      o["single_row_calls"] = static_cast<int64_t>(t.single_row_calls);
      o["group_calls"] = static_cast<int64_t>(t.group_calls);
      o["group_rows"] = static_cast<int64_t>(t.group_rows);
      o["leaf_groups_split"] = static_cast<int64_t>(t.leaf_groups_split);
      o["single_wait_ms_mean"] = t.single_wait_ms_mean;
      o["single_wait_ms_max"] = t.single_wait_ms_max;
      o["group_wait_ms_mean"] = t.group_wait_ms_mean;
      o["group_wait_ms_max"] = t.group_wait_ms_max;
      // WO-PERF-TIMING-BATCH: the device-side split the Phase-2 promotion gate
      // asks for. Zero on CPU and zero without EnableBatcherTelemetry();
      // `device_timed_batches` says how many physical batches contributed, so a
      // reader can tell "measured zero" from "never measured".
      o["h2d_ms"] = t.h2d_ms;
      o["forward_ms"] = t.forward_ms;
      o["output_cast_ms"] = t.output_cast_ms;
      o["d2h_ms"] = t.d2h_ms;
      // Reported SEPARATELY and never folded into a percentage split with the
      // four device intervals above: sync_ms is host wall time blocked in
      // torch::cuda::synchronize(), which OVERLAPS the device work it waits on.
      o["sync_ms"] = t.sync_ms;
      o["device_timing_note"] =
          "h2d/forward/output_cast/d2h are CUDA-event intervals on the compute "
          "stream; sync_ms is host blocked time and OVERLAPS them. Their sum is "
          "not a total and percentages of that sum are meaningless.";
      o["device_timed_batches"] =
          static_cast<int64_t>(t.device_timed_batches);
      o["benchmark_raw_prior_calls"] = static_cast<int64_t>(raw_priors);
      o["implied_opponent_prior_calls"] = opponent_calls;
      o["leaf_value_records_delta"] = static_cast<int64_t>(leaf_evals_delta);
      o["rows_identity_ok"] = rows_identity;
      o["leaf_rows_crosscheck_ok"] = leaf_rows_crosscheck;
      std::ofstream out(tel_path);
      if (out) {
        out << open_spiel::json::ToString(o, /*wrap=*/true);
        std::cout << "\nWrote batcher telemetry to " << tel_path << "\n";
      } else {
        std::cerr << "Warning: could not open --batcher_telemetry_json_path: "
                  << tel_path << "\n";
      }
    }
  } else {
    std::cout << "\n(direct mode: no shared batcher; run with "
                 "--evaluator_mode=batched for batcher telemetry)\n";
  }

  return 0;
}

}  // namespace open_spiel
