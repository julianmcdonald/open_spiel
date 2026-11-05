#pragma once

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <numeric>
#include <random>
#include <shared_mutex>
#include <string>
#include <vector>

#include "dune_batched_evaluator.h"
#include "dune_network.h"
#include "dune_search_routing.h"
#include "dune_search_session.h"
#include "dune_seed_utils.h"
#include "dune_wide_rollout_search.h"
#include "open_spiel/games/dune_imperium/dune_imperium.h"
#include "open_spiel/games/dune_imperium/dune_imperium_content.h"
#include "open_spiel/spiel.h"

struct Fp8BenchRoot {
  int game_id = 0;
  int round = 1;
  std::string round_band;
  std::string decision_type;
  std::shared_ptr<open_spiel::State> state;
  int player = 0;
  double stratum_weight = 1.0;
  open_spiel::Action raw_action = 0;
  open_spiel::Action a_bf16 = 0;
  open_spiel::Action a_fp8 = 0;
  float max_prior_diff = 0.0f;
  double bf16_ms = 0.0;
  double fp8_ms = 0.0;
  int bf16_sims = 0;
  int fp8_sims = 0;
  double u_raw = 0.0;
  double u_bf16 = 0.0;
  double u_fp8 = 0.0;
};

struct StatSummary {
  double unweighted_mean = 0.0;
  double u_ci_low = 0.0;
  double u_ci_high = 0.0;
  double weighted_mean = 0.0;
  double w_ci_low = 0.0;
  double w_ci_high = 0.0;
};

inline StatSummary ComputeMetricStats(
    const std::vector<Fp8BenchRoot>& roots,
    const std::vector<size_t>& subset,
    const std::vector<int>& unique_games,
    const std::map<int, std::vector<size_t>>& game_map,
    int metric_id) {  // 0: bf16-raw, 1: fp8-raw, 2: fp8-bf16
  StatSummary s;
  if (subset.empty() || unique_games.empty()) return s;
  auto get_val = [](const Fp8BenchRoot& r, int mid) {
    if (mid == 0) return r.u_bf16 - r.u_raw;
    if (mid == 1) return r.u_fp8 - r.u_raw;
    return r.u_fp8 - r.u_bf16;
  };
  double sum_v = 0.0, sum_wv = 0.0, sum_w = 0.0;
  std::vector<bool> in_subset(roots.size(), false);
  for (size_t idx : subset) {
    in_subset[idx] = true;
    double v = get_val(roots[idx], metric_id);
    sum_v += v;
    sum_wv += v * roots[idx].stratum_weight;
    sum_w += roots[idx].stratum_weight;
  }
  s.unweighted_mean = sum_v / static_cast<double>(subset.size());
  s.weighted_mean = sum_w > 0.0 ? (sum_wv / sum_w) : 0.0;

  std::mt19937_64 rng(20261104ULL + static_cast<uint64_t>(metric_id) * 17ULL);
  std::uniform_int_distribution<size_t> dist(0, unique_games.size() - 1);
  std::vector<double> boot_u(10000, 0.0);
  std::vector<double> boot_w(10000, 0.0);
  for (int b = 0; b < 10000; ++b) {
    double b_uv = 0.0, b_uc = 0.0, b_wv = 0.0, b_wc = 0.0;
    for (size_t g = 0; g < unique_games.size(); ++g) {
      int gid = unique_games[dist(rng)];
      auto it = game_map.find(gid);
      if (it == game_map.end()) continue;
      for (size_t idx : it->second) {
        if (!in_subset[idx]) continue;
        double v = get_val(roots[idx], metric_id);
        b_uv += v;
        b_uc += 1.0;
        b_wv += v * roots[idx].stratum_weight;
        b_wc += roots[idx].stratum_weight;
      }
    }
    boot_u[b] = b_uc > 0.0 ? (b_uv / b_uc) : 0.0;
    boot_w[b] = b_wc > 0.0 ? (b_wv / b_wc) : 0.0;
  }
  std::sort(boot_u.begin(), boot_u.end());
  std::sort(boot_w.begin(), boot_w.end());
  s.u_ci_low = boot_u[250];
  s.u_ci_high = boot_u[9750];
  s.w_ci_low = boot_w[250];
  s.w_ci_high = boot_w[9750];
  return s;
}

inline void RunFp8Ref2048Benchmark(
    const std::string& raw_model_path, const std::string& json_out_path,
    int target_roots, int census_games, int ref_rollouts) {
  const auto bench_t0 = std::chrono::steady_clock::now();
  std::string model_path = raw_model_path.empty()
      ? "/home/warcr/projects/dune_drl/models/champion_policy.pt"
      : raw_model_path;
  std::cout << "Starting FP8 Ref2048 Benchmark with model: " << model_path
            << " (target_roots=" << target_roots
            << ", census_games=" << census_games
            << ", ref_rollouts=" << ref_rollouts << ")\n";

  open_spiel::DuneAdaptiveRolloutSession sess_bf16(
      model_path, "cuda", 2048, 16, 256, 32, 256, 20261104ULL,
      true, true, 96, 1, 576, true, false);
  open_spiel::DuneAdaptiveRolloutSession sess_fp8(
      model_path, "cuda", 2048, 16, 256, 32, 256, 20261104ULL,
      true, true, 96, 1, 576, true, true);
  sess_bf16.set_prior_log_ratio_weight(0.02);
  sess_bf16.set_stage1_high_conf_prior_skip(0.95);
  sess_bf16.set_use_event_keyed_crn(true);
  sess_fp8.set_prior_log_ratio_weight(0.02);
  sess_fp8.set_stage1_high_conf_prior_skip(0.95);
  sess_fp8.set_use_event_keyed_crn(true);

  auto ref_model = std::make_shared<open_spiel::SharedDunePolicyValueNetImpl>(
      open_spiel::dune_imperium::kFullPublicInformationV4StateSize, 2048,
      open_spiel::dune_imperium::kNumDistinctPlayerActions, 16,
      false, false, 0, true, false,
      open_spiel::dune_semantic::kDescriptorSchemaVersionV3);
  ref_model->market_appendix_mode_ =
      open_spiel::dune_imperium::MarketAppendixMode::kFullPublicInformationV4;
  ref_model->deck_pool_semantics_ =
      open_spiel::dune_imperium::DeckPoolSemantics::kFixed;
  open_spiel::LoadModelCheckpointRobust(
      ref_model, model_path, torch::Device(torch::kCPU));
  ref_model->to(torch::Device(torch::kCUDA, 0));
  ref_model->eval();
  for (auto& parameter : ref_model->parameters()) {
    parameter.set_requires_grad(false);
  }
  ref_model->RefreshBf16InferenceCache();
  std::shared_mutex ref_mtx;
  auto ref_coord = std::make_shared<open_spiel::BatchedEvaluator>(
      ref_model, 256, 1, torch::Device(torch::kCUDA, 0), &ref_mtx,
      10.0f, false, true, false, true, true);
  auto ref_eval = std::make_shared<open_spiel::BatchedNNEvaluator>(
      ref_coord, 10.0f,
      open_spiel::dune_imperium::MarketAppendixMode::kFullPublicInformationV4);
  ref_coord->WarmupCudaGraphs({1, 4, 96, 256});
  std::vector<std::shared_ptr<open_spiel::BatchedNNEvaluator>> ref_evals = {
      ref_eval};

  auto game = open_spiel::LoadGame("dune_imperium");
  std::vector<Fp8BenchRoot> candidates;
  for (int g = 0; g < census_games; ++g) {
    auto st = game->NewInitialState();
    std::mt19937_64 rng(20261104ULL + static_cast<uint64_t>(g) * 9973ULL);
    while (!st->IsTerminal()) {
      if (st->IsChanceNode()) {
        auto outcomes = st->ChanceOutcomes();
        std::vector<double> weights;
        weights.reserve(outcomes.size());
        for (const auto& o : outcomes) weights.push_back(o.second);
        std::discrete_distribution<size_t> d(weights.begin(), weights.end());
        st->ApplyAction(outcomes[d(rng)].first);
      } else {
        auto legal = st->LegalActions();
        if (legal.empty()) break;
        int p = st->CurrentPlayer();
        const auto& ds =
            dynamic_cast<const open_spiel::dune_imperium::DuneImperiumState&>(
                *st);
        if (legal.size() > 1 &&
            open_spiel::dune_imperium::IsHighStakesStrategicState(*st, p)) {
          Fp8BenchRoot cand;
          cand.game_id = g;
          cand.round = ds.GetCurrentRound();
          cand.round_band =
              (cand.round <= 3) ? "R1-3" : ((cand.round <= 6) ? "R4-6" : "R7+");
          bool has_tleilaxu = false;
          for (auto a : legal) {
            if (a >= open_spiel::dune_imperium::kActionTleilaxuAcquire0 &&
                a <= open_spiel::dune_imperium::kActionTleilaxuAcquire0 + 2) {
              has_tleilaxu = true;
              break;
            }
          }
          if (open_spiel::dune_imperium::IsPurchaseDecisionState(*st)) {
            cand.decision_type =
                has_tleilaxu ? "Purchase:Tleilaxu" : "Purchase:Imperium";
          } else if (ds.phase() ==
                     open_spiel::dune_imperium::GamePhase::kCombat) {
            cand.decision_type = "CombatIntrigue";
          } else if (open_spiel::ClassifyDuneDecisionRole(*st, p, false) ==
                     open_spiel::DuneDecisionRole::kAgentPrimary) {
            cand.decision_type = "AgentPrimary";
          } else {
            cand.decision_type = "AgentContinuation";
          }
          cand.state = st->Clone();
          cand.player = p;
          candidates.push_back(std::move(cand));
        }
        open_spiel::Action chosen = legal[0];
        if (legal.size() > 1) {
          auto pd = sess_bf16.PriorWithDetails(*st);
          float best_p = -1.0f;
          for (size_t k = 0; k < pd.actions.size(); ++k) {
            if (pd.probabilities[k] > best_p &&
                std::find(legal.begin(), legal.end(), pd.actions[k]) !=
                    legal.end()) {
              best_p = pd.probabilities[k];
              chosen = pd.actions[k];
            }
          }
        }
        st->ApplyAction(chosen);
      }
    }
  }

  std::map<std::string, std::vector<Fp8BenchRoot>> strata;
  for (auto& c : candidates) {
    strata[c.round_band + "/" + c.decision_type].push_back(std::move(c));
  }
  std::mt19937_64 strat_rng(20261018ULL);
  for (auto& kv : strata) {
    std::shuffle(kv.second.begin(), kv.second.end(), strat_rng);
  }

  std::vector<std::string> strat_keys;
  for (const auto& kv : strata) strat_keys.push_back(kv.first);
  std::sort(strat_keys.begin(), strat_keys.end(),
            [&](const std::string& a, const std::string& b) {
              if (strata[a].size() != strata[b].size()) {
                return strata[a].size() > strata[b].size();
              }
              return a < b;
            });
  int K = static_cast<int>(strat_keys.size());
  int base_q = target_roots / std::max(1, K);
  int rem_q = target_roots % std::max(1, K);
  std::map<std::string, int> quota;
  int allocated = 0;
  for (int i = 0; i < K; ++i) {
    int q = std::min(static_cast<int>(strata[strat_keys[i]].size()),
                     base_q + (i < rem_q ? 1 : 0));
    quota[strat_keys[i]] = q;
    allocated += q;
  }
  while (allocated < target_roots) {
    bool progress = false;
    for (const auto& k : strat_keys) {
      if (quota[k] < static_cast<int>(strata[k].size())) {
        quota[k]++;
        allocated++;
        progress = true;
        if (allocated == target_roots) break;
      }
    }
    if (!progress) break;
  }

  std::vector<Fp8BenchRoot> roots;
  for (const auto& k : strat_keys) {
    int q = quota[k];
    for (int i = 0; i < q; ++i) {
      roots.push_back(std::move(strata[k][i]));
      roots.back().stratum_weight =
          static_cast<double>(strata[k].size()) / std::max(1, q);
    }
  }
  double sum_raw_w = 0.0;
  for (const auto& r : roots) sum_raw_w += r.stratum_weight;
  double N = static_cast<double>(roots.size());
  if (sum_raw_w > 0.0) {
    for (auto& r : roots) r.stratum_weight *= (N / sum_raw_w);
  }
  std::cout << "Census extracted " << candidates.size()
            << " high-stakes candidates across " << K << " strata. Sampled "
            << roots.size() << " roots.\n";
  for (const auto& k : strat_keys) {
    std::cout << "  Stratum " << k << ": pool=" << strata[k].size()
              << ", sampled=" << quota[k] << "\n";
  }

  if (!roots.empty()) {
    auto pdb0 = sess_bf16.PriorWithDetails(*roots[0].state);
    auto pdr0 = ref_eval->PriorWithDetails(*roots[0].state);
    std::map<open_spiel::Action, float> mb0, mr0;
    for (size_t k = 0; k < pdb0.actions.size(); ++k) {
      mb0[pdb0.actions[k]] = pdb0.probabilities[k];
    }
    for (size_t k = 0; k < pdr0.actions.size(); ++k) {
      mr0[pdr0.actions[k]] = pdr0.probabilities[k];
    }
    float ref_diff = 0.0f;
    for (auto act : roots[0].state->LegalActions()) {
      ref_diff = std::max(ref_diff, std::abs(mb0[act] - mr0[act]));
    }
    std::cout << "Ref2048 vs Session BF16 prior parity max_diff=" << ref_diff
              << "\n";
    if (ref_diff > 1e-5f) {
      std::cerr << "FATAL: ref_model prior mismatch vs sess_bf16: " << ref_diff
                << "\n";
      std::exit(1);
    }
  }

  float global_max_prior_diff = 0.0f;
  int ref2048_eval_roots = 0;
  for (size_t i = 0; i < roots.size(); ++i) {
    auto& r = roots[i];
    auto pdb = sess_bf16.PriorWithDetails(*r.state);
    auto pdf = sess_fp8.PriorWithDetails(*r.state);
    std::map<open_spiel::Action, float> mb, mf;
    for (size_t k = 0; k < pdb.actions.size(); ++k) {
      mb[pdb.actions[k]] = pdb.probabilities[k];
    }
    for (size_t k = 0; k < pdf.actions.size(); ++k) {
      mf[pdf.actions[k]] = pdf.probabilities[k];
    }
    float max_d = 0.0f, best_pb = -1.0f;
    for (auto act : r.state->LegalActions()) {
      max_d = std::max(max_d, std::abs(mb[act] - mf[act]));
      if (mb[act] > best_pb) {
        best_pb = mb[act];
        r.raw_action = act;
      }
    }
    r.max_prior_diff = max_d;
    global_max_prior_diff = std::max(global_max_prior_diff, max_d);

    uint64_t seed = dune_seed::DeriveSeed(20261104ULL, 0x0701ULL, i);
    sess_bf16.ResetSession("bench_root");
    sess_bf16.set_seed(seed);
    auto rb = sess_bf16.Search(*r.state, -1.0);
    sess_bf16.DiscardPendingAction();
    r.a_bf16 = rb.diagnostics.selected_action;
    r.bf16_ms = rb.elapsed_time_ms;
    r.bf16_sims = rb.simulations_completed;

    sess_fp8.ResetSession("bench_root");
    sess_fp8.set_seed(seed);
    auto rf = sess_fp8.Search(*r.state, -1.0);
    sess_fp8.DiscardPendingAction();
    r.a_fp8 = rf.diagnostics.selected_action;
    r.fp8_ms = rf.elapsed_time_ms;
    r.fp8_sims = rf.simulations_completed;

    std::vector<open_spiel::Action> distinct = {r.raw_action};
    if (std::find(distinct.begin(), distinct.end(), r.a_bf16) ==
        distinct.end()) {
      distinct.push_back(r.a_bf16);
    }
    if (std::find(distinct.begin(), distinct.end(), r.a_fp8) ==
        distinct.end()) {
      distinct.push_back(r.a_fp8);
    }

    if (distinct.size() >= 2) {
      ++ref2048_eval_roots;
      const auto r_t0 = std::chrono::steady_clock::now();
      auto ref_res =
          open_spiel::dune_imperium::PairedPriorSortedRolloutSearchDecision(
              *r.state, r.player, nullptr, ref_evals, distinct, ref_rollouts,
              0.0, 0.0, dune_seed::DeriveSeed(20261104ULL, 0x0703ULL, i),
              open_spiel::dune_imperium::SearchRolloutCrnMode::kEventKeyedCrn,
              false, -1.0, 128, true, true);
      const double r_sec =
          std::chrono::duration<double>(std::chrono::steady_clock::now() - r_t0)
              .count();
      std::map<open_spiel::Action, double> u_map;
      for (const auto& cu : ref_res.candidate_utilities) {
        u_map[cu.first] = cu.second;
      }
      r.u_raw = u_map[r.raw_action];
      r.u_bf16 = u_map[r.a_bf16];
      r.u_fp8 = u_map[r.a_fp8];
      std::cout << "[Ref2048] Root " << i << "/" << roots.size()
                << " (" << r.round_band << "/" << r.decision_type << ")"
                << " distinct=" << distinct.size()
                << " raw=" << r.raw_action
                << " bf16=" << r.a_bf16
                << " fp8=" << r.a_fp8
                << " d(fp8-bf16)=" << (r.u_fp8 - r.u_bf16)
                << " (" << r_sec << "s)\n";
    }

    if ((i + 1) % 25 == 0 || i + 1 == roots.size()) {
      const double elapsed_s =
          std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                        bench_t0)
              .count();
      std::cout << "[Roots Progress] " << (i + 1) << "/" << roots.size()
                << " evaluated (ref2048_roots=" << ref2048_eval_roots
                << ", elapsed=" << elapsed_s << "s)\n";
    }
  }

  std::vector<int> unique_games;
  std::map<int, std::vector<size_t>> game_map;
  for (size_t i = 0; i < roots.size(); ++i) {
    if (game_map.find(roots[i].game_id) == game_map.end()) {
      unique_games.push_back(roots[i].game_id);
    }
    game_map[roots[i].game_id].push_back(i);
  }

  std::vector<size_t> all_idx, r13_idx, r46_idx, r7_idx, purch_idx,
      nonpurch_idx;
  int agree_count = 0, bf16_override = 0, fp8_override = 0,
      both_override_same = 0, union_override = 0, searched_count = 0;
  std::vector<double> steady_bf16_ms, steady_fp8_ms, s_bf16_ms, s_fp8_ms;
  for (size_t i = 0; i < roots.size(); ++i) {
    all_idx.push_back(i);
    if (roots[i].round_band == "R1-3") {
      r13_idx.push_back(i);
    } else if (roots[i].round_band == "R4-6") {
      r46_idx.push_back(i);
    } else {
      r7_idx.push_back(i);
    }
    if (roots[i].decision_type.rfind("Purchase", 0) == 0) {
      purch_idx.push_back(i);
    } else {
      nonpurch_idx.push_back(i);
    }

    if (roots[i].bf16_sims > 0) searched_count++;
    if (roots[i].a_bf16 == roots[i].a_fp8) agree_count++;
    bool bo = (roots[i].a_bf16 != roots[i].raw_action);
    bool fo = (roots[i].a_fp8 != roots[i].raw_action);
    if (bo) bf16_override++;
    if (fo) fp8_override++;
    if (bo || fo) union_override++;
    if (bo && fo && roots[i].a_bf16 == roots[i].a_fp8) both_override_same++;

    if (i >= 1) {
      steady_bf16_ms.push_back(roots[i].bf16_ms);
      steady_fp8_ms.push_back(roots[i].fp8_ms);
      if (roots[i].bf16_sims > 0) {
        s_bf16_ms.push_back(roots[i].bf16_ms);
        s_fp8_ms.push_back(roots[i].fp8_ms);
      }
    }
  }

  auto get_p = [](std::vector<double> v, double q) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    size_t idx = static_cast<size_t>(q * static_cast<double>(v.size() - 1));
    return v[idx];
  };
  double mean_b_ms = steady_bf16_ms.empty()
      ? 0.0
      : std::accumulate(steady_bf16_ms.begin(), steady_bf16_ms.end(), 0.0) /
            static_cast<double>(steady_bf16_ms.size());
  double mean_f_ms = steady_fp8_ms.empty()
      ? 0.0
      : std::accumulate(steady_fp8_ms.begin(), steady_fp8_ms.end(), 0.0) /
            static_cast<double>(steady_fp8_ms.size());
  double speedup =
      mean_b_ms > 0.0 ? (mean_b_ms - mean_f_ms) / mean_b_ms * 100.0 : 0.0;

  double mean_sb_ms = s_bf16_ms.empty()
      ? 0.0
      : std::accumulate(s_bf16_ms.begin(), s_bf16_ms.end(), 0.0) /
            static_cast<double>(s_bf16_ms.size());
  double mean_sf_ms = s_fp8_ms.empty()
      ? 0.0
      : std::accumulate(s_fp8_ms.begin(), s_fp8_ms.end(), 0.0) /
            static_cast<double>(s_fp8_ms.size());
  double searched_speedup =
      mean_sb_ms > 0.0 ? (mean_sb_ms - mean_sf_ms) / mean_sb_ms * 100.0 : 0.0;

  auto overall_b_raw =
      ComputeMetricStats(roots, all_idx, unique_games, game_map, 0);
  auto overall_f_raw =
      ComputeMetricStats(roots, all_idx, unique_games, game_map, 1);
  auto overall_f_b =
      ComputeMetricStats(roots, all_idx, unique_games, game_map, 2);
  auto purch_f_b =
      ComputeMetricStats(roots, purch_idx, unique_games, game_map, 2);
  auto nonpurch_f_b =
      ComputeMetricStats(roots, nonpurch_idx, unique_games, game_map, 2);

  const double total_wall_s =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - bench_t0)
          .count();

  std::cout << "\n=== FP8 Ref2048 Benchmark Summary ===\n"
            << "Total roots: " << roots.size()
            << " (searched=" << searched_count
            << ", ref2048_evaluated=" << ref2048_eval_roots
            << ", unique_games=" << unique_games.size() << ")\n"
            << "Max M=1 prior prob diff: " << global_max_prior_diff << "\n"
            << "Move agreement (FP8 vs BF16): "
            << (agree_count * 100.0 / std::max<size_t>(1, roots.size())) << "% ("
            << agree_count << "/" << roots.size() << ")\n"
            << "Overrides: BF16=" << bf16_override << ", FP8=" << fp8_override
            << ", both_same=" << both_override_same
            << ", union=" << union_override << "\n"
            << "Steady Speedup: " << speedup << "% (BF16 " << mean_b_ms
            << " ms vs FP8 " << mean_f_ms << " ms)\n"
            << "Searched-only Speedup: " << searched_speedup << "%\n"
            << "BF16 - Raw (weighted): " << overall_b_raw.weighted_mean << " ["
            << overall_b_raw.w_ci_low << ", " << overall_b_raw.w_ci_high
            << "] | unweighted: " << overall_b_raw.unweighted_mean << " ["
            << overall_b_raw.u_ci_low << ", " << overall_b_raw.u_ci_high << "]\n"
            << "FP8 - Raw (weighted): " << overall_f_raw.weighted_mean << " ["
            << overall_f_raw.w_ci_low << ", " << overall_f_raw.w_ci_high
            << "] | unweighted: " << overall_f_raw.unweighted_mean << " ["
            << overall_f_raw.u_ci_low << ", " << overall_f_raw.u_ci_high << "]\n"
            << "FP8 - BF16 (weighted): " << overall_f_b.weighted_mean << " ["
            << overall_f_b.w_ci_low << ", " << overall_f_b.w_ci_high
            << "] | unweighted: " << overall_f_b.unweighted_mean << " ["
            << overall_f_b.u_ci_low << ", " << overall_f_b.u_ci_high << "]\n"
            << "Total wall time: " << total_wall_s << " s\n";

  if (!json_out_path.empty()) {
    auto write_stat = [](std::ofstream& o, const StatSummary& st) {
      o << "{\"weighted\": " << st.weighted_mean
        << ", \"w_ci_low\": " << st.w_ci_low
        << ", \"w_ci_high\": " << st.w_ci_high
        << ", \"unweighted\": " << st.unweighted_mean
        << ", \"u_ci_low\": " << st.u_ci_low
        << ", \"u_ci_high\": " << st.u_ci_high << "}";
    };
    std::ofstream out(json_out_path);
    out << std::fixed << std::setprecision(6) << "{\n"
        << "  \"roots_count\": " << roots.size() << ",\n"
        << "  \"searched_roots_count\": " << searched_count << ",\n"
        << "  \"ref2048_evaluated_roots\": " << ref2048_eval_roots << ",\n"
        << "  \"unique_games_count\": " << unique_games.size() << ",\n"
        << "  \"strata_count\": " << K << ",\n"
        << "  \"max_m1_prior_prob_diff\": " << global_max_prior_diff << ",\n"
        << "  \"move_agreement_count\": " << agree_count << ",\n"
        << "  \"move_agreement_pct\": "
        << (agree_count * 100.0 / std::max<size_t>(1, roots.size())) << ",\n"
        << "  \"bf16_overrides\": " << bf16_override << ",\n"
        << "  \"fp8_overrides\": " << fp8_override << ",\n"
        << "  \"both_override_same\": " << both_override_same << ",\n"
        << "  \"union_overrides\": " << union_override << ",\n"
        << "  \"steady_speedup_pct\": " << speedup << ",\n"
        << "  \"searched_only_steady_speedup_pct\": " << searched_speedup << ",\n"
        << "  \"total_wall_seconds\": " << total_wall_s << ",\n"
        << "  \"steady_latency_ms\": {\n"
        << "    \"bf16\": {\"mean\": " << mean_b_ms
        << ", \"median\": " << get_p(steady_bf16_ms, 0.5)
        << ", \"p90\": " << get_p(steady_bf16_ms, 0.9) << "},\n"
        << "    \"fp8\": {\"mean\": " << mean_f_ms
        << ", \"median\": " << get_p(steady_fp8_ms, 0.5)
        << ", \"p90\": " << get_p(steady_fp8_ms, 0.9) << "}\n"
        << "  },\n"
        << "  \"metrics\": {\n"
        << "    \"fp8_minus_bf16\": ";
    write_stat(out, overall_f_b);
    out << ",\n    \"bf16_minus_raw\": ";
    write_stat(out, overall_b_raw);
    out << ",\n    \"fp8_minus_raw\": ";
    write_stat(out, overall_f_raw);
    out << ",\n    \"purchase_fp8_minus_bf16\": ";
    write_stat(out, purch_f_b);
    out << ",\n    \"nonpurchase_fp8_minus_bf16\": ";
    write_stat(out, nonpurch_f_b);
    out << "\n  }\n}\n";
  }
}
