#include "open_spiel/examples/dune_fixed_root_search_benchmark_internal.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <iostream>

#include "open_spiel/abseil-cpp/absl/flags/flag.h"
#include "open_spiel/abseil-cpp/absl/strings/str_format.h"
#include "open_spiel/examples/dune_seed_utils.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel::dune_imperium::fixed_root_benchmark {
namespace {

template <typename SearchResultT>
VariantResult BuildPairedRefVariantResult(
    uint64_t dec_seed,
    Action raw_a,
    const SearchResultT& s_res) {
  VariantResult vr;
  vr.decision_seed = dec_seed;
  vr.chosen_action = s_res.action;
  vr.overridden = s_res.overridden;
  vr.raw_mean_utility = s_res.raw_mean_utility;
  vr.chosen_mean_utility = s_res.chosen_mean_utility;
  for (const auto& cu : s_res.candidate_utilities) {
    vr.candidate_utilities.push_back({cu.first, cu.second});
  }
  for (const auto& cru : s_res.candidate_rollout_utilities) {
    vr.candidate_rollout_utilities.push_back({cru.first, cru.second});
  }
  vr.stage1_planned_worlds = s_res.stage1_planned_worlds;
  vr.stage1_kept_worlds = s_res.stage1_kept_worlds;
  vr.stage2_planned_worlds = s_res.stage2_planned_worlds;
  vr.stage2_kept_worlds = s_res.stage2_kept_worlds;
  vr.deadline_bound = (s_res.stage1_kept_worlds < s_res.stage1_planned_worlds);

  const std::vector<double>& raw_rollouts = s_res.candidate_rollout_utilities.empty()
      ? std::vector<double>{}
      : s_res.candidate_rollout_utilities.front().second;
  for (const auto& cru : s_res.candidate_rollout_utilities) {
    Action cand_a = cru.first;
    if (cand_a == raw_a) {
      vr.candidate_pair_sd_vs_raw.push_back({static_cast<int>(cand_a), 0.0});
    } else {
      double sd = ComputePairedRolloutSdVsRaw(raw_rollouts, cru.second);
      vr.candidate_pair_sd_vs_raw.push_back({static_cast<int>(cand_a), sd});
    }
  }
  return vr;
}

void VerifyBaseReceiptRef2048Sanity(
    const open_spiel::Game& game,
    const std::vector<std::shared_ptr<open_spiel::BatchedNNEvaluator>>& evaluators,
    const std::vector<SampledRoot>& sampled_roots,
    const std::vector<VariantConfig>& variants,
    const std::vector<std::vector<VariantResult>>& all_results,
    const std::vector<std::vector<bool>>& has_base_variant,
    double min_override_margin) {
  size_t ref_v_idx = variants.size();
  for (size_t v = 0; v < variants.size(); ++v) {
    if (variants[v].type == BenchmarkVariantType::kPairedReferenceOnChosenActions) {
      ref_v_idx = v;
      break;
    }
  }
  if (ref_v_idx >= variants.size()) {
    return;
  }

  int sanity_root_idx = -1;
  for (size_t r = 0; r < sampled_roots.size(); ++r) {
    if (has_base_variant[r][ref_v_idx] &&
        all_results[r][ref_v_idx].evaluations > 0 &&
        all_results[r][ref_v_idx].stage1_kept_worlds > 0) {
      sanity_root_idx = static_cast<int>(r);
      break;
    }
  }
  if (sanity_root_idx < 0) {
    return;
  }

  const auto& root = sampled_roots[sanity_root_idx];
  const auto& cached_ref_vr = all_results[sanity_root_idx][ref_v_idx];
  std::unique_ptr<open_spiel::State> sanity_state = game.NewInitialState();
  for (int64_t a : root.action_history) {
    sanity_state->ApplyAction(a);
  }
  std::vector<Action> distinct_actions;
  for (const auto& cu : cached_ref_vr.candidate_utilities) {
    distinct_actions.push_back(cu.first);
  }
  auto s_res = PairedPriorSortedRolloutSearchDecision(
      *sanity_state, root.search_seat, /*turn_tree=*/nullptr, evaluators,
      distinct_actions, /*rollouts_per_action=*/2048, min_override_margin,
      /*min_t_stat=*/1.0, cached_ref_vr.decision_seed, SearchRolloutCrnMode::kSequential,
      /*sample_opponent_leader_draft=*/false, /*max_time_ms=*/900000.0,
      /*max_inflight_worlds=*/128);
  double max_u_diff = std::abs(s_res.raw_mean_utility - cached_ref_vr.raw_mean_utility);
  SPIEL_CHECK_EQ(s_res.candidate_utilities.size(), cached_ref_vr.candidate_utilities.size());
  for (size_t i = 0; i < s_res.candidate_utilities.size(); ++i) {
    SPIEL_CHECK_EQ(s_res.candidate_utilities[i].first, cached_ref_vr.candidate_utilities[i].first);
    double diff = std::abs(s_res.candidate_utilities[i].second - cached_ref_vr.candidate_utilities[i].second);
    if (diff > max_u_diff) max_u_diff = diff;
  }
  if (max_u_diff > 1e-5) {
    open_spiel::SpielFatalError(absl::StrFormat(
        "[SANITY-CHECK] Ref2048 reproduction failed on root %d: max_u_diff=%e > 1e-5",
        root.root_id, max_u_diff));
  }
  std::cout << absl::StrFormat(
      "[SANITY-CHECK] Verified Ref2048 reproduction on root %d (seed=%llu): max_u_diff=%e\n",
      root.root_id, static_cast<unsigned long long>(cached_ref_vr.decision_seed), max_u_diff);
}

bool HandleCachedBasePairedRefExtension(
    const open_spiel::State& state,
    const SampledRoot& root,
    const VariantConfig& v_cfg,
    size_t r_idx,
    size_t v_idx,
    uint64_t dec_seed,
    double min_override_margin,
    const std::vector<std::shared_ptr<open_spiel::BatchedNNEvaluator>>& evaluators,
    std::vector<std::vector<VariantResult>>& all_results) {
  const Action raw_a = all_results[r_idx][0].chosen_action;
  std::vector<Action> distinct_actions;
  distinct_actions.push_back(raw_a);
  for (size_t prev_v = 0; prev_v < v_idx; ++prev_v) {
    Action a = all_results[r_idx][prev_v].chosen_action;
    if (std::find(distinct_actions.begin(), distinct_actions.end(), a) == distinct_actions.end()) {
      distinct_actions.push_back(a);
    }
  }

  auto cached_ref = all_results[r_idx][v_idx];
  // Case 1: distinct_actions.size() <= 1
  if (distinct_actions.size() <= 1) {
    return true;
  }

  // Case 2: distinct_actions.size() > 1 AND (cached_ref.evaluations == 0 || cached_ref.stage1_kept_worlds == 0)
  if (cached_ref.evaluations == 0 || cached_ref.stage1_kept_worlds == 0) {
    uint64_t evals_before = open_spiel::GetTotalNetworkEvaluations();
    auto t_start = std::chrono::steady_clock::now();

    auto s_res = PairedPriorSortedRolloutSearchDecision(
        state, root.search_seat, /*turn_tree=*/nullptr, evaluators,
        distinct_actions, /*rollouts_per_action=*/v_cfg.rollouts, min_override_margin,
        /*min_t_stat=*/1.0, dec_seed, SearchRolloutCrnMode::kSequential,
        /*sample_opponent_leader_draft=*/false, /*max_time_ms=*/900000.0,
        /*max_inflight_worlds=*/128);

    auto t_end = std::chrono::steady_clock::now();
    uint64_t evals_after = open_spiel::GetTotalNetworkEvaluations();
    double dur_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();

    VariantResult vr = BuildPairedRefVariantResult(dec_seed, raw_a, s_res);
    vr.evaluations = evals_after - evals_before;
    vr.duration_ms = dur_ms;
    all_results[r_idx][v_idx] = std::move(vr);
    std::cout << absl::StrFormat(
        "[REF2048-EXTEND] Root %d: populated previously-empty Ref2048 for override actions\n",
        root.root_id);
    return true;
  }

  // Case 3: distinct_actions.size() > 1 AND cached_ref.evaluations > 0
  std::vector<Action> missing_actions;
  for (Action a : distinct_actions) {
    bool found = false;
    for (const auto& cu : cached_ref.candidate_utilities) {
      if (cu.first == static_cast<int>(a)) {
        found = true;
        break;
      }
    }
    if (!found) {
      missing_actions.push_back(a);
    }
  }

  if (missing_actions.empty()) {
    return true;
  }

  std::vector<Action> eval_actions = {raw_a};
  for (Action ma : missing_actions) {
    eval_actions.push_back(ma);
  }

  uint64_t evals_before = open_spiel::GetTotalNetworkEvaluations();
  auto t_start = std::chrono::steady_clock::now();

  auto s_res = PairedPriorSortedRolloutSearchDecision(
      state, root.search_seat, /*turn_tree=*/nullptr, evaluators,
      eval_actions, /*rollouts_per_action=*/v_cfg.rollouts, min_override_margin,
      /*min_t_stat=*/1.0, dec_seed, SearchRolloutCrnMode::kSequential,
      /*sample_opponent_leader_draft=*/false, /*max_time_ms=*/900000.0,
      /*max_inflight_worlds=*/128);

  auto t_end = std::chrono::steady_clock::now();
  uint64_t evals_after = open_spiel::GetTotalNetworkEvaluations();
  double dur_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();

  double raw_diff = std::abs(s_res.raw_mean_utility - cached_ref.raw_mean_utility);
  if (raw_diff > 1e-5) {
    open_spiel::SpielFatalError(absl::StrFormat(
        "[REF2048-EXTEND] Raw mean utility diff %e > 1e-5 on root %d during extension",
        raw_diff, root.root_id));
  }

  const std::vector<double>& raw_rollouts = s_res.candidate_rollout_utilities.empty()
      ? std::vector<double>{}
      : s_res.candidate_rollout_utilities.front().second;

  for (const auto& cru : s_res.candidate_rollout_utilities) {
    Action cand_a = cru.first;
    if (cand_a == raw_a) continue;

    for (const auto& cu : s_res.candidate_utilities) {
      if (cu.first == cand_a) {
        cached_ref.candidate_utilities.push_back({static_cast<int>(cand_a), cu.second});
        break;
      }
    }

    cached_ref.candidate_rollout_utilities.push_back({static_cast<int>(cand_a), cru.second});
    double sd = ComputePairedRolloutSdVsRaw(raw_rollouts, cru.second);
    cached_ref.candidate_pair_sd_vs_raw.push_back({static_cast<int>(cand_a), sd});
  }

  ReevaluateExtendedRef2048Choice(min_override_margin, cached_ref);
  cached_ref.evaluations += (evals_after - evals_before);
  cached_ref.duration_ms += dur_ms;

  all_results[r_idx][v_idx] = cached_ref;
  std::cout << absl::StrFormat(
      "[REF2048-EXTEND] Root %d: appended %zu new action(s) to populated Ref2048 (raw_u diff=%e)\n",
      root.root_id, missing_actions.size(), raw_diff);
  return true;
}

void RunFixedRootVariantsLoop(
    const open_spiel::Game& game,
    const std::vector<std::shared_ptr<open_spiel::BatchedNNEvaluator>>& evaluators,
    const std::vector<SampledRoot>& sampled_roots,
    const std::vector<VariantConfig>& variants,
    const std::vector<std::vector<bool>>& has_base_variant,
    size_t start_root_idx,
    int top_k,
    int reference_top_k,
    double min_override_margin,
    uint64_t master_seed,
    const std::string& checkpoint_path,
    std::vector<std::vector<VariantResult>>& all_results) {
  auto get_effective_top_k = [](const VariantConfig& vc, const SampledRoot& rt) -> int {
    if (vc.type != BenchmarkVariantType::kSuccessiveHalvingFixedTop5) return 0;
    if (vc.all_buy && rt.decision_type.rfind("Purchase", 0) == 0) {
      return static_cast<int>(rt.legal_actions.size());
    }
    return vc.fixed_top_k;
  };
  bool verified_k_equivalence = false;

  for (size_t r_idx = start_root_idx; r_idx < sampled_roots.size(); ++r_idx) {
    const auto& root = sampled_roots[r_idx];

    // Replay state
    std::unique_ptr<open_spiel::State> state = game.NewInitialState();
    for (int64_t a : root.action_history) {
      state->ApplyAction(a);
    }

    if (r_idx % 25 == 0 || r_idx + 1 == sampled_roots.size()) {
      std::cout << absl::StrFormat("  Processing root %zu / %zu (Game %d, Rnd %d, %s)...\n",
                                   r_idx + 1, sampled_roots.size(), root.game_id, root.round, root.decision_type);
    }

    for (size_t v_idx = 0; v_idx < variants.size(); ++v_idx) {
      const auto& v_cfg = variants[v_idx];

      if (v_cfg.type != BenchmarkVariantType::kPairedReferenceOnChosenActions) {
        if (has_base_variant[r_idx][v_idx]) {
          continue;
        }
      }

      uint64_t dec_seed = (v_cfg.type == BenchmarkVariantType::kPairedReferenceOnChosenActions &&
                           has_base_variant[r_idx][v_idx] &&
                           all_results[r_idx][v_idx].decision_seed != 0)
          ? all_results[r_idx][v_idx].decision_seed
          : dune_seed::DeriveSeed(master_seed, v_cfg.seed_domain, root.root_id);

      if (v_cfg.type == BenchmarkVariantType::kPairedReferenceOnChosenActions &&
          has_base_variant[r_idx][v_idx]) {
        HandleCachedBasePairedRefExtension(
            *state, root, v_cfg, r_idx, v_idx, dec_seed,
            min_override_margin, evaluators, all_results);
        continue;
      }

      uint64_t evals_before = open_spiel::GetTotalNetworkEvaluations();
      auto t_start = std::chrono::steady_clock::now();

      VariantResult vr;
      vr.decision_seed = dec_seed;

      if (v_cfg.type == BenchmarkVariantType::kRaw) {
        const auto legal = state->LegalActions();
        const ActionsAndProbs raw_prior = evaluators.front()->Prior(*state);
        const auto sorted_by_prior = GetSortedCandidatesByPrior(raw_prior, legal);
        Action raw_a = sorted_by_prior.empty() ? (legal.empty() ? kInvalidAction : legal.front()) : sorted_by_prior.front().first;
        vr.chosen_action = raw_a;
        vr.overridden = false;
        vr.raw_mean_utility = 0.0;
        vr.chosen_mean_utility = 0.0;
      } else if (v_cfg.type == BenchmarkVariantType::kSuccessiveHalvingDynamic ||
                 v_cfg.type == BenchmarkVariantType::kSuccessiveHalvingFixedTop5) {
        int eff_top_k = 0;
        if (v_cfg.type == BenchmarkVariantType::kSuccessiveHalvingFixedTop5) {
          eff_top_k = get_effective_top_k(v_cfg, root);
          const int eff_K = std::min<int>(eff_top_k, static_cast<int>(root.legal_actions.size()));

          int matched_prev_v = -1;
          for (size_t prev_v = 0; prev_v < v_idx; ++prev_v) {
            if (variants[prev_v].type == BenchmarkVariantType::kSuccessiveHalvingFixedTop5 &&
                variants[prev_v].reference_rollouts_per_action == v_cfg.reference_rollouts_per_action &&
                variants[prev_v].seed_domain == v_cfg.seed_domain &&
                std::min<int>(get_effective_top_k(variants[prev_v], root), static_cast<int>(root.legal_actions.size())) == eff_K &&
                all_results[r_idx][prev_v].stage1_planned_worlds > 0) {
              matched_prev_v = static_cast<int>(prev_v);
              break;
            }
          }

          if (matched_prev_v >= 0) {
            if (!verified_k_equivalence) {
              auto s_res = SuccessiveHalvingRolloutSearchDecision(
                  *state, root.search_seat, /*turn_tree=*/nullptr, evaluators,
                  v_cfg.reference_rollouts_per_action, min_override_margin,
                  dec_seed, SearchRolloutCrnMode::kSequential,
                  /*unbiased_stage2_gate=*/false, /*max_time_ms=*/-1.0,
                  eff_top_k);
              if (s_res.action != all_results[r_idx][matched_prev_v].chosen_action ||
                  std::abs(s_res.chosen_mean_utility - all_results[r_idx][matched_prev_v].chosen_mean_utility) > 1e-6) {
                open_spiel::SpielFatalError(absl::StrFormat(
                    "[SANITY-CHECK] K-equivalence mismatch between %s and %s on root %d: action %d vs %d, utility %f vs %f",
                    v_cfg.name, variants[matched_prev_v].name, root.root_id,
                    s_res.action, all_results[r_idx][matched_prev_v].chosen_action,
                    s_res.chosen_mean_utility, all_results[r_idx][matched_prev_v].chosen_mean_utility));
              }
              std::cout << absl::StrFormat(
                  "[SANITY-CHECK] Verified exact top-k equivalence between %s and %s on root %d (eff_K=%d)\n",
                  v_cfg.name, variants[matched_prev_v].name, root.root_id, eff_K);
              verified_k_equivalence = true;
            }
            all_results[r_idx][v_idx] = all_results[r_idx][matched_prev_v];
            continue;
          }
        }

        auto s_res = SuccessiveHalvingRolloutSearchDecision(
            *state, root.search_seat, /*turn_tree=*/nullptr, evaluators,
            v_cfg.reference_rollouts_per_action, min_override_margin,
            dec_seed, SearchRolloutCrnMode::kSequential,
            /*unbiased_stage2_gate=*/false, /*max_time_ms=*/-1.0,
            eff_top_k);
        vr.chosen_action = s_res.action;
        vr.overridden = s_res.overridden;
        vr.raw_mean_utility = s_res.raw_mean_utility;
        vr.chosen_mean_utility = s_res.chosen_mean_utility;
        for (const auto& cu : s_res.candidate_utilities) {
          vr.candidate_utilities.push_back({cu.first, cu.second});
        }
        for (const auto& cru : s_res.candidate_rollout_utilities) {
          vr.candidate_rollout_utilities.push_back({cru.first, cru.second});
        }
        vr.stage1_planned_worlds = s_res.stage1_planned_worlds;
        vr.stage1_kept_worlds = s_res.stage1_kept_worlds;
        vr.stage2_planned_worlds = s_res.stage2_planned_worlds;
        vr.stage2_kept_worlds = s_res.stage2_kept_worlds;
        vr.deadline_bound = false;
      } else if (v_cfg.type == BenchmarkVariantType::kPairedReferenceOnChosenActions) {
        const Action raw_a = all_results[r_idx][0].chosen_action;
        std::vector<Action> distinct_actions;
        distinct_actions.push_back(raw_a);
        for (size_t prev_v = 0; prev_v < v_idx; ++prev_v) {
          Action a = all_results[r_idx][prev_v].chosen_action;
          if (std::find(distinct_actions.begin(), distinct_actions.end(), a) == distinct_actions.end()) {
            distinct_actions.push_back(a);
          }
        }

        if (distinct_actions.size() <= 1) {
          vr.chosen_action = raw_a;
          vr.overridden = false;
          vr.raw_mean_utility = 0.0;
          vr.chosen_mean_utility = 0.0;
          vr.candidate_utilities.push_back({static_cast<int>(raw_a), 0.0});
          vr.candidate_rollout_utilities.push_back({static_cast<int>(raw_a), {}});
          vr.candidate_pair_sd_vs_raw.push_back({static_cast<int>(raw_a), 0.0});
          vr.stage1_planned_worlds = 0;
          vr.stage1_kept_worlds = 0;
          vr.stage2_planned_worlds = 0;
          vr.stage2_kept_worlds = 0;
          vr.deadline_bound = false;
        } else {
          auto s_res = PairedPriorSortedRolloutSearchDecision(
              *state, root.search_seat, /*turn_tree=*/nullptr, evaluators,
              distinct_actions, /*rollouts_per_action=*/v_cfg.rollouts, min_override_margin,
              /*min_t_stat=*/1.0, dec_seed, SearchRolloutCrnMode::kSequential,
              /*sample_opponent_leader_draft=*/false, /*max_time_ms=*/900000.0,
              /*max_inflight_worlds=*/128);
          vr = BuildPairedRefVariantResult(dec_seed, raw_a, s_res);
        }
      } else {
        const int eff_top_k = v_cfg.name == "Ref512" ? reference_top_k : top_k;
        auto s_res = CompoundRolloutSearchDecision(
            *state, root.search_seat, /*turn_tree=*/nullptr, evaluators,
            eff_top_k, v_cfg.rollouts, min_override_margin,
            TreeReuseMode::kTargetFill, dec_seed,
            v_cfg.pairing, v_cfg.truncation);
        vr.chosen_action = s_res.action;
        vr.overridden = s_res.overridden;
        vr.raw_mean_utility = s_res.raw_mean_utility;
        vr.chosen_mean_utility = s_res.chosen_mean_utility;
        vr.stage1_planned_worlds = v_cfg.rollouts;
        vr.stage1_kept_worlds = v_cfg.rollouts;
        vr.stage2_planned_worlds = 0;
        vr.stage2_kept_worlds = 0;
        vr.deadline_bound = false;
        for (const auto& cu : s_res.candidate_utilities) {
          vr.candidate_utilities.push_back({cu.first, cu.second});
        }
        for (const auto& cru : s_res.candidate_rollout_utilities) {
          vr.candidate_rollout_utilities.push_back({cru.first, cru.second});
        }
      }

      auto t_end = std::chrono::steady_clock::now();
      uint64_t evals_after = open_spiel::GetTotalNetworkEvaluations();
      double dur_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();

      vr.evaluations = evals_after - evals_before;
      vr.duration_ms = dur_ms;
      all_results[r_idx][v_idx] = std::move(vr);
    }

    if ((r_idx + 1) % 10 == 0 || r_idx + 1 == sampled_roots.size()) {
      open_spiel::json::Object temp_summary;
      SaveBenchmarkReceipt(checkpoint_path, temp_summary, sampled_roots, variants, all_results, {}, r_idx + 1);
    }
  }
}

std::vector<VariantResult> RunDisagreementRecheck1024(
    const open_spiel::Game& game,
    const std::vector<std::shared_ptr<open_spiel::BatchedNNEvaluator>>& evaluators,
    const std::vector<SampledRoot>& sampled_roots,
    const std::vector<VariantConfig>& variants,
    const std::vector<std::vector<VariantResult>>& all_results,
    const open_spiel::json::Object& root_out_json,
    const std::string& output_path,
    size_t ref_idx,
    int reference_top_k,
    double min_override_margin,
    uint64_t master_seed) {
  // 1,024-Rollout Re-check Pass on Disagreement Roots
  std::vector<int> disagreement_root_indices;
  for (size_t r_idx = 0; r_idx < sampled_roots.size(); ++r_idx) {
    const auto& ref_vr = all_results[r_idx][ref_idx];
    bool has_disagreement = false;
    for (size_t v_idx = 0; v_idx < ref_idx; ++v_idx) {
      if (all_results[r_idx][v_idx].chosen_action != ref_vr.chosen_action) {
        has_disagreement = true;
        break;
      }
    }
    if (has_disagreement) {
      disagreement_root_indices.push_back(static_cast<int>(r_idx));
    }
  }

  std::cout << absl::StrFormat(
      "[RE-CHECK] Identified %zu disagreement roots. Running 1,024-rollout paired re-check (domain 0x050B)...\n",
      disagreement_root_indices.size());

  std::vector<VariantResult> recheck_1024_results(sampled_roots.size());
  for (size_t d_idx = 0; d_idx < disagreement_root_indices.size(); ++d_idx) {
    int r_idx = disagreement_root_indices[d_idx];
    const auto& root = sampled_roots[r_idx];
    std::unique_ptr<open_spiel::State> state = game.NewInitialState();
    for (int64_t a : root.action_history) {
      state->ApplyAction(a);
    }

    if (d_idx % 10 == 0 || d_idx + 1 == disagreement_root_indices.size()) {
      std::cout << absl::StrFormat("  Re-checking disagreement root %zu / %zu (Root %d, Game %d)...\n",
                                   d_idx + 1, disagreement_root_indices.size(), root.root_id, root.game_id);
    }

    uint64_t dec_seed = dune_seed::DeriveSeed(master_seed, benchmark_domains::kDomainRecheck1024, root.root_id);
    uint64_t evals_before = open_spiel::GetTotalNetworkEvaluations();
    auto t_start = std::chrono::steady_clock::now();
    auto s_res = CompoundRolloutSearchDecision(
        *state, root.search_seat, /*turn_tree=*/nullptr, evaluators,
        reference_top_k, 1024, min_override_margin,
        TreeReuseMode::kTargetFill, dec_seed,
        SearchPairingMode::kPaired, SearchTruncationMode::kTerminal);
    auto t_end = std::chrono::steady_clock::now();
    uint64_t evals_after = open_spiel::GetTotalNetworkEvaluations();

    VariantResult vr;
    vr.chosen_action = s_res.action;
    vr.overridden = s_res.overridden;
    vr.raw_mean_utility = s_res.raw_mean_utility;
    vr.chosen_mean_utility = s_res.chosen_mean_utility;
    vr.decision_seed = dec_seed;
    vr.evaluations = evals_after - evals_before;
    vr.duration_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
    for (const auto& cu : s_res.candidate_utilities) {
      vr.candidate_utilities.push_back({cu.first, cu.second});
    }
    for (const auto& cru : s_res.candidate_rollout_utilities) {
      vr.candidate_rollout_utilities.push_back({cru.first, cru.second});
    }
    recheck_1024_results[r_idx] = std::move(vr);

    if ((d_idx + 1) % 5 == 0 || d_idx + 1 == disagreement_root_indices.size()) {
      SaveBenchmarkReceipt(output_path, root_out_json, sampled_roots, variants, all_results, recheck_1024_results, sampled_roots.size());
    }
  }
  std::cout << "[RE-CHECK] 1,024-rollout re-check complete!\n\n";
  return recheck_1024_results;
}

}  // namespace

int RunFixedRootBenchmark() {
  const std::string model_path = absl::GetFlag(FLAGS_model_checkpoint);
  const std::string roots_path = absl::GetFlag(FLAGS_sampled_roots_json);
  const std::string obs_path = absl::GetFlag(FLAGS_obs_binary);
  const std::string output_path = absl::GetFlag(FLAGS_output_json);
  const int num_evals = absl::GetFlag(FLAGS_num_evaluators);
  const int target_batch_size = absl::GetFlag(FLAGS_target_batch_size);
  const int top_k = absl::GetFlag(FLAGS_top_k_actions);
  const int reference_top_k = (absl::GetFlag(FLAGS_sh_offline_check) || absl::GetFlag(FLAGS_sh_budget_check))
      ? std::max(top_k, 5) : top_k;
  const double min_override_margin = absl::GetFlag(FLAGS_min_override_margin);
  const uint64_t master_seed = absl::GetFlag(FLAGS_master_seed);

  if (model_path.empty() || roots_path.empty() || obs_path.empty() || output_path.empty()) {
    std::cerr << "Usage: dune_fixed_root_search_benchmark --model_checkpoint=... --sampled_roots_json=... --obs_binary=... --output_json=...\n";
    return 1;
  }

  std::cout << "================================================================================\n";
  std::cout << "STEP 1: FIXED-ROOT SEARCH BENCHMARK HARNESS\n";
  std::cout << "================================================================================\n";
  std::cout << "Model checkpoint:   " << model_path << "\n";
  std::cout << "Sampled roots JSON: " << roots_path << "\n";
  std::cout << "Observation bin:    " << obs_path << "\n";
  std::cout << "Master seed:        " << master_seed << "\n";

  // Load Game
  std::shared_ptr<const open_spiel::Game> game = open_spiel::LoadGame("dune_imperium");
  std::string current_game_str = game->ToString();

  RootLoadVerifyOptions root_opts;
  root_opts.verify_game_string = true;
  root_opts.filter_placement_only = false;
  root_opts.check_legal_during_replay = true;
  root_opts.check_obs_stream_good = true;
  root_opts.fill_missing_pi_parent = false;
  root_opts.require_non_empty = true;
  root_opts.log_loaded_count = true;
  root_opts.verbose_parity_messages = true;

  std::vector<SampledRoot> sampled_roots = LoadSampledRootsJson(roots_path, current_game_str, root_opts);

  // Initialize Torch Device & Evaluators
  EvaluatorBundle bundle = InitEvaluatorBundle(
      num_evals, target_batch_size, absl::GetFlag(FLAGS_num_blocks),
      model_path, absl::GetFlag(FLAGS_rollout_amp), absl::GetFlag(FLAGS_allow_tf32),
      /*disable_global_tf32=*/false, /*log_device=*/true);
  std::cout << "[MODEL] Successfully initialized " << num_evals << " evaluator coordinators.\n";

  if (absl::GetFlag(FLAGS_sh_budget_check)) {
    open_spiel::dune_imperium::DuneImperiumState::SetAutoPassEmptyCombatIntrigueHands(false);
  }

  // Replay check & observation verification for all roots
  VerifySampledRootsObsParity(*game, obs_path, bundle.evaluators[0], root_opts, sampled_roots);

  // Configure Variants
  std::vector<VariantConfig> variants = BuildBenchmarkVariants();
  std::cout << "\n[BENCHMARK] Executing " << variants.size() << " variants across " << sampled_roots.size() << " roots...\n";

  // Data storage: results[root_idx][variant_idx]
  std::vector<std::vector<VariantResult>> all_results(
      sampled_roots.size(), std::vector<VariantResult>(variants.size()));
  std::vector<std::vector<bool>> has_base_variant(
      sampled_roots.size(), std::vector<bool>(variants.size(), false));

  const std::string base_receipt_json = absl::GetFlag(FLAGS_base_receipt_json);
  if (!base_receipt_json.empty()) {
    if (!LoadBaseReceiptVariants(base_receipt_json, sampled_roots, variants, all_results, has_base_variant)) {
      open_spiel::SpielFatalError(absl::StrFormat("Failed to load base receipt JSON: %s", base_receipt_json));
    }
  }

  const std::string checkpoint_path = output_path + ".checkpoint.json";
  size_t start_root_idx = 0;
  if (LoadBenchmarkCheckpoint(checkpoint_path, variants, all_results, start_root_idx)) {
    std::cout << absl::StrFormat("[RESUME] Resuming from checkpoint: %zu / %zu roots already completed!\n",
                                 start_root_idx, sampled_roots.size());
  }

  if (!base_receipt_json.empty()) {
    VerifyBaseReceiptRef2048Sanity(
        *game, bundle.evaluators, sampled_roots, variants, all_results, has_base_variant, min_override_margin);
  }

  RunFixedRootVariantsLoop(
      *game, bundle.evaluators, sampled_roots, variants, has_base_variant,
      start_root_idx, top_k, reference_top_k, min_override_margin,
      master_seed, checkpoint_path, all_results);

  // If leaf_check_only, compute and report V2_16_Paired vs Ref512 correlation and exit
  if (absl::GetFlag(FLAGS_leaf_check_only)) {
    LeafCheckStats leaf_stats = ComputeLeafCheckStats(sampled_roots, all_results);
    ReportLeafCheckAndSave(output_path, checkpoint_path, leaf_stats, sampled_roots, variants, all_results);
    return 0;
  }

  const size_t ref_idx = variants.size() - 1;  // Reference is the last variant
  SPIEL_CHECK_TRUE(variants[ref_idx].name == "Ref512" || variants[ref_idx].name == "Ref2048");

  std::cout << absl::StrFormat("[BENCHMARK] All roots evaluated! Analyzing true gain against %s...\n\n",
                               variants[ref_idx].name);

  BenchmarkSummaryReportData summary_data =
      ComputeBenchmarkSummaryReportData(sampled_roots, variants, all_results, ref_idx);
  open_spiel::json::Object root_out_json =
      PrintAndBuildMainBenchmarkReport(output_path, summary_data, sampled_roots, variants, all_results);

  std::vector<VariantResult> recheck_1024_results;
  if (!absl::GetFlag(FLAGS_skip_recheck_1024) && !absl::GetFlag(FLAGS_sh_budget_check)) {
    recheck_1024_results = RunDisagreementRecheck1024(
        *game, bundle.evaluators, sampled_roots, variants, all_results,
        root_out_json, output_path, ref_idx, reference_top_k, min_override_margin, master_seed);
  } else {
    std::cout << "[RE-CHECK] Skipping 1,024-rollout re-check pass (--skip_recheck_1024=true).\n\n";
  }

  SaveBenchmarkReceipt(output_path, root_out_json, sampled_roots, variants, all_results, recheck_1024_results, sampled_roots.size());
  std::cout << "[RECEIPT] Saved finalized benchmark receipt JSON to " << output_path << "\n";
  std::remove(checkpoint_path.c_str());

  return 0;
}

}  // namespace open_spiel::dune_imperium::fixed_root_benchmark
