#include "open_spiel/examples/dune_fixed_root_search_benchmark_internal.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <unordered_map>

#include "open_spiel/abseil-cpp/absl/flags/flag.h"
#include "open_spiel/abseil-cpp/absl/strings/str_format.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel::dune_imperium::fixed_root_benchmark {

double ComputePairedRolloutSdVsRaw(
    const std::vector<double>& raw_rollouts,
    const std::vector<double>& cand_rollouts) {
  const size_t n = std::min(raw_rollouts.size(), cand_rollouts.size());
  if (n < 2) {
    return 0.0;
  }
  double sum_diff = 0.0;
  for (size_t i = 0; i < n; ++i) {
    sum_diff += (cand_rollouts[i] - raw_rollouts[i]);
  }
  const double mean_diff = sum_diff / static_cast<double>(n);
  double sum_sq = 0.0;
  for (size_t i = 0; i < n; ++i) {
    double d = (cand_rollouts[i] - raw_rollouts[i]) - mean_diff;
    sum_sq += d * d;
  }
  const double var = sum_sq / static_cast<double>(n - 1);
  return std::sqrt(std::max(0.0, var));
}

void ReevaluateExtendedRef2048Choice(
    double min_override_margin,
    VariantResult& cached_ref) {
  const size_t K = cached_ref.candidate_rollout_utilities.size();
  const auto& base_raw_rollouts = cached_ref.candidate_rollout_utilities.front().second;
  Action best_action = static_cast<Action>(cached_ref.candidate_rollout_utilities.front().first);
  double best_mean_u = cached_ref.raw_mean_utility;
  bool overridden = false;

  for (size_t c_idx = 1; c_idx < K; ++c_idx) {
    Action cand_a = static_cast<Action>(cached_ref.candidate_rollout_utilities[c_idx].first);
    const auto& cand_rollouts = cached_ref.candidate_rollout_utilities[c_idx].second;
    double sum_u = 0.0;
    const int n_c = static_cast<int>(cand_rollouts.size());
    for (int r = 0; r < n_c; ++r) {
      sum_u += cand_rollouts[r];
    }
    double mean_u = (n_c > 0) ? (sum_u / n_c) : 0.0;
    double mean_d = mean_u - cached_ref.raw_mean_utility;

    const size_t n_pairs = std::min(base_raw_rollouts.size(), cand_rollouts.size());
    double sum_sq_diff = 0.0;
    for (size_t r = 0; r < n_pairs; ++r) {
      double d_r = cand_rollouts[r] - base_raw_rollouts[r];
      double diff = d_r - mean_d;
      sum_sq_diff += diff * diff;
    }
    double var_d = (n_pairs > 1) ? (sum_sq_diff / (n_pairs - 1)) : 0.0;
    double se_d = (n_pairs > 1) ? std::sqrt(var_d / n_pairs) : 0.0;
    const double t_stat = (n_pairs >= 2) ? ((se_d > 1e-9) ? (mean_d / se_d) : (mean_d > 0.0 ? 100.0 : 0.0)) : 0.0;

    if (mean_d > min_override_margin && t_stat >= 1.0 && mean_u > best_mean_u) {
      best_mean_u = mean_u;
      best_action = cand_a;
      overridden = true;
    }
  }

  cached_ref.chosen_action = best_action;
  cached_ref.chosen_mean_utility = best_mean_u;
  cached_ref.overridden = overridden;
}

LeafCheckStats ComputeLeafCheckStats(
    const std::vector<SampledRoot>& sampled_roots,
    const std::vector<std::vector<VariantResult>>& all_results) {
  std::vector<double> critic_x;
  std::vector<double> critic_y;
  for (size_t r_idx = 0; r_idx < sampled_roots.size(); ++r_idx) {
    const auto& root = sampled_roots[r_idx];
    Action raw_a = static_cast<Action>(root.raw_action);
    const auto& v2_res = all_results[r_idx][0];
    const auto& ref_res = all_results[r_idx][1];

    std::unordered_map<Action, double> v2_cand_utils;
    for (const auto& cu : v2_res.candidate_utilities) v2_cand_utils[cu.first] = cu.second;
    double q_v2_raw = v2_cand_utils.count(raw_a) ? v2_cand_utils[raw_a] : v2_res.raw_mean_utility;

    std::unordered_map<Action, double> ref_cand_utils;
    for (const auto& cu : ref_res.candidate_utilities) ref_cand_utils[cu.first] = cu.second;
    double q_ref_raw = ref_cand_utils.count(raw_a) ? ref_cand_utils[raw_a] : ref_res.raw_mean_utility;

    for (const auto& cu : v2_cand_utils) {
      Action c = cu.first;
      if (c == raw_a || !ref_cand_utils.count(c)) continue;
      critic_x.push_back(cu.second - q_v2_raw);
      critic_y.push_back(ref_cand_utils[c] - q_ref_raw);
    }
  }
  double v2_ref_corr = 0.0;
  if (critic_x.size() >= 2) {
    double mx = std::accumulate(critic_x.begin(), critic_x.end(), 0.0) / critic_x.size();
    double my = std::accumulate(critic_y.begin(), critic_y.end(), 0.0) / critic_y.size();
    double cov = 0.0, vx = 0.0, vy = 0.0;
    for (size_t i = 0; i < critic_x.size(); ++i) {
      cov += (critic_x[i] - mx) * (critic_y[i] - my);
      vx += (critic_x[i] - mx) * (critic_x[i] - mx);
      vy += (critic_y[i] - my) * (critic_y[i] - my);
    }
    double denom = std::sqrt(vx * vy);
    v2_ref_corr = (denom > 1e-12) ? (cov / denom) : 0.0;
  }
  return LeafCheckStats{v2_ref_corr, static_cast<int64_t>(critic_x.size())};
}

BenchmarkSummaryReportData ComputeBenchmarkSummaryReportData(
    const std::vector<SampledRoot>& sampled_roots,
    const std::vector<VariantConfig>& variants,
    const std::vector<std::vector<VariantResult>>& all_results,
    size_t ref_idx) {
  BenchmarkSummaryReportData report;
  double total_weight_sum = 0.0;
  for (const auto& r : sampled_roots) total_weight_sum += r.stratum_weight;

  for (size_t v_idx = 0; v_idx < ref_idx; ++v_idx) {
    const auto& v_cfg = variants[v_idx];
    double weighted_agree = 0.0;
    double weighted_true_gain = 0.0;
    uint64_t total_evals = 0;
    double total_time_ms = 0.0;
    int override_count = 0;

    for (size_t r_idx = 0; r_idx < sampled_roots.size(); ++r_idx) {
      const auto& root = sampled_roots[r_idx];
      const auto& vr = all_results[r_idx][v_idx];
      const auto& ref_vr = all_results[r_idx][ref_idx];
      double w = root.stratum_weight;

      total_evals += vr.evaluations;
      total_time_ms += vr.duration_ms;
      if (vr.overridden) override_count++;

      // Agreement with reference action
      if (vr.chosen_action == ref_vr.chosen_action) {
        weighted_agree += w;
      }

      // MUST FIX 4(b): Fatal check if variant's chosen action is not in reference candidate_utilities
      bool found_chosen = false;
      double ref_chosen_u = 0.0;
      double ref_raw_u = ref_vr.raw_mean_utility;
      for (const auto& r_cu : ref_vr.candidate_utilities) {
        if (r_cu.first == vr.chosen_action) {
          ref_chosen_u = r_cu.second;
          found_chosen = true;
          break;
        }
      }
      if (!found_chosen) {
        open_spiel::SpielFatalError(absl::StrFormat(
            "Fatal: Variant %s chosen action %d on root %d not found in %s candidate_utilities! "
            "%s must evaluate all top candidate actions.",
            v_cfg.name, vr.chosen_action, root.root_id,
            variants[ref_idx].name, variants[ref_idx].name));
      }
      double true_gain = ref_chosen_u - ref_raw_u;
      weighted_true_gain += w * true_gain;
    }

    BenchmarkVariantSummary row;
    row.name = v_cfg.name;
    row.rollouts = v_cfg.rollouts;
    row.pairing_str = (v_cfg.pairing == SearchPairingMode::kPaired) ? "Paired" : "Unpaired";
    row.mode_str = (v_cfg.truncation == SearchTruncationMode::kTerminal) ? "Terminal" : "Truncated";
    row.agree_pct = 100.0 * weighted_agree / total_weight_sum;
    row.mean_gain = weighted_true_gain / total_weight_sum;
    row.avg_evals = static_cast<double>(total_evals) / sampled_roots.size();
    row.avg_time_ms = total_time_ms / sampled_roots.size();
    row.override_count = override_count;
    report.variant_summaries.push_back(std::move(row));
  }

  // Reference statistics reported separately
  uint64_t ref_total_evals = 0;
  double ref_total_time_ms = 0.0;
  for (size_t r_idx = 0; r_idx < sampled_roots.size(); ++r_idx) {
    ref_total_evals += all_results[r_idx][ref_idx].evaluations;
    ref_total_time_ms += all_results[r_idx][ref_idx].duration_ms;
  }
  report.ref_name = variants[ref_idx].name;
  report.ref_rollouts = variants[ref_idx].rollouts;
  report.ref_avg_evals = static_cast<double>(ref_total_evals) / sampled_roots.size();
  report.ref_avg_time_ms = ref_total_time_ms / sampled_roots.size();
  return report;
}

void AccumulateHeldOutReplayDecision(
    int round,
    int64_t obs_row,
    bool is_override,
    Action raw_a,
    Action best_cand_action,
    Action argmax_mu,
    Action model_argmax,
    double policy_entropy,
    const std::vector<Action>& legal_actions,
    const std::vector<float>& capped_logits,
    const std::unordered_map<Action, double>& model_probs,
    const std::unordered_map<Action, double>& mu_map,
    const std::vector<std::pair<Action, double>>& candidate_utilities,
    const std::unordered_map<Action, double>& cand_util_map,
    const std::vector<double>& target_dist,
    HeldOutReplayStats& stats) {
  if (absl::GetFlag(FLAGS_dump_per_state)) {
    open_spiel::json::Object dec_obj;
    dec_obj["obs_row"] = open_spiel::json::Value(static_cast<int64_t>(obs_row));
    dec_obj["is_override"] = open_spiel::json::Value(is_override);
    dec_obj["entropy"] = open_spiel::json::Value(policy_entropy);
    dec_obj["model_argmax"] = open_spiel::json::Value(static_cast<int64_t>(model_argmax));
    dec_obj["argmax_mu"] = open_spiel::json::Value(static_cast<int64_t>(argmax_mu));
    dec_obj["raw_action"] = open_spiel::json::Value(static_cast<int64_t>(raw_a));
    dec_obj["best_cand_action"] = open_spiel::json::Value(static_cast<int64_t>(best_cand_action));

    open_spiel::json::Array cand_arr;
    open_spiel::json::Array cand_probs_arr;
    open_spiel::json::Array cand_logits_arr;
    for (const auto& cu : candidate_utilities) {
      Action ca = cu.first;
      cand_arr.push_back(open_spiel::json::Value(static_cast<int64_t>(ca)));
      auto p_it = model_probs.find(ca);
      cand_probs_arr.push_back(open_spiel::json::Value(p_it != model_probs.end() ? p_it->second : 0.0));
      cand_logits_arr.push_back(open_spiel::json::Value(static_cast<double>(capped_logits[ca])));
    }
    dec_obj["candidate_actions"] = open_spiel::json::Value(cand_arr);
    dec_obj["candidate_probs"] = open_spiel::json::Value(cand_probs_arr);
    dec_obj["candidate_capped_logits"] = open_spiel::json::Value(cand_logits_arr);
    stats.dumped_decisions.push_back(open_spiel::json::Value(dec_obj));
  }

  // max |p - mu| over legal actions
  double max_p_diff_mu = 0.0;
  for (Action a : legal_actions) {
    double p_val = model_probs.count(a) ? model_probs.at(a) : 0.0;
    double mu_val = mu_map.count(a) ? mu_map.at(a) : 0.0;
    max_p_diff_mu = std::max(max_p_diff_mu, std::abs(p_val - mu_val));
  }

  // Value of changed choices:
  // where the student's greedy move differs from argmax(mu) and both are among recorded candidates,
  // utility difference = cand_util_map[model_argmax] - cand_util_map[argmax_mu].
  bool val_covered = false;
  double val_diff = 0.0;
  if (model_argmax != argmax_mu &&
      cand_util_map.count(model_argmax) &&
      cand_util_map.count(argmax_mu)) {
    val_covered = true;
    val_diff = cand_util_map.at(model_argmax) - cand_util_map.at(argmax_mu);
  }

  // Compute cross-entropy against MPO relative target
  double ce = 0.0;
  Action target_argmax = -1;
  double max_q = -1.0;
  double max_movement = 0.0;
  for (size_t i = 0; i < legal_actions.size(); ++i) {
    Action a = legal_actions[i];
    double q = target_dist[i];
    if (q > max_q) {
      max_q = q;
      target_argmax = a;
    }
    double p = model_probs.count(a) ? model_probs.at(a) : 0.0;
    ce -= q * std::log(std::max(p, 1e-12));
    max_movement = std::max(max_movement, std::abs(p - q));
  }

  stats.total_ce_all += ce;
  if (model_argmax == target_argmax) {
    ++stats.top1_matches;
  }
  stats.total_max_prob_diff += max_movement;
  stats.round_ce_sum[round] += ce;
  stats.round_dec_count[round]++;
  ++stats.fresh_search_decisions;

  ReplaySubgroupMetrics& g = is_override ? stats.ovr_stats : stats.non_ovr_stats;
  g.count++;
  if (model_argmax != argmax_mu) {
    g.greedy_changed_count++;
  }
  g.sum_max_p_diff_mu += max_p_diff_mu;
  g.sum_policy_entropy += policy_entropy;
  if (val_covered) {
    g.value_covered_count++;
    g.sum_value_diff += val_diff;
  }
  g.sum_ce += ce;

  if (is_override) {
    if (model_argmax == best_cand_action) {
      g.greedy_agree_search_count++;
    }
    double teacher_prob = model_probs.count(best_cand_action) ? model_probs.at(best_cand_action) : 0.0;
    g.sum_prob_search_move += teacher_prob;
  }
}

HeldOutReplayComputedMetrics ComputeHeldOutReplayMetrics(const HeldOutReplayStats& stats) {
  HeldOutReplayComputedMetrics m;
  const int64_t n = stats.fresh_search_decisions;
  m.mean_ce_all = stats.total_ce_all / n;
  m.top1_acc = 100.0 * static_cast<double>(stats.top1_matches) / n;
  m.mean_max_diff = stats.total_max_prob_diff / n;

  const auto& ovr = stats.ovr_stats;
  m.ovr_greedy_change_rate = (ovr.count > 0) ? (100.0 * ovr.greedy_changed_count / ovr.count) : 0.0;
  m.ovr_greedy_agree_search = (ovr.count > 0) ? (100.0 * ovr.greedy_agree_search_count / ovr.count) : 0.0;
  m.ovr_mean_prob_search = (ovr.count > 0) ? (ovr.sum_prob_search_move / ovr.count) : 0.0;
  m.ovr_mean_max_p_diff = (ovr.count > 0) ? (ovr.sum_max_p_diff_mu / ovr.count) : 0.0;
  m.ovr_mean_entropy = (ovr.count > 0) ? (ovr.sum_policy_entropy / ovr.count) : 0.0;
  m.ovr_val_changed_diff = (ovr.value_covered_count > 0) ? (ovr.sum_value_diff / ovr.value_covered_count) : 0.0;
  m.mean_ce_override = (ovr.count > 0) ? (ovr.sum_ce / ovr.count) : 0.0;
  m.override_pct = 100.0 * static_cast<double>(ovr.count) / n;

  const auto& non_ovr = stats.non_ovr_stats;
  m.non_ovr_greedy_change_rate = (non_ovr.count > 0) ? (100.0 * non_ovr.greedy_changed_count / non_ovr.count) : 0.0;
  m.non_ovr_mean_max_p_diff = (non_ovr.count > 0) ? (non_ovr.sum_max_p_diff_mu / non_ovr.count) : 0.0;
  m.non_ovr_mean_entropy = (non_ovr.count > 0) ? (non_ovr.sum_policy_entropy / non_ovr.count) : 0.0;
  m.non_ovr_val_changed_diff = (non_ovr.value_covered_count > 0) ? (non_ovr.sum_value_diff / non_ovr.value_covered_count) : 0.0;
  m.mean_ce_non_override = (non_ovr.count > 0) ? (non_ovr.sum_ce / non_ovr.count) : 0.0;
  return m;
}

DistillLabelStats ComputeDistillLabelStats(
    const std::vector<LabelledResult>& results,
    double total_wall_s) {
  DistillLabelStats s;
  s.total_wall_s = total_wall_s;
  s.roots_per_s = results.size() / std::max(1e-4, total_wall_s);
  double sum_w = 0.0;
  for (const auto& r : results) {
    if (r.overridden) s.total_overrides++;
    sum_w += r.w;
  }
  s.override_rate = 100.0 * s.total_overrides / std::max<size_t>(1, results.size());
  s.mean_w = sum_w / std::max<size_t>(1, results.size());
  return s;
}

TieBreakerStats ComputeTieBreakerBootstrapStats(
    size_t num_games,
    uint64_t master_seed,
    const std::vector<std::vector<ChangedDecision>>& per_game_changed,
    const std::vector<ChangedDecision>& sampled_decisions,
    const std::vector<double>& deltas) {
  const size_t M = sampled_decisions.size();
  double sum_delta = 0.0;
  for (double d : deltas) sum_delta += d;
  double mean_delta = sum_delta / static_cast<double>(M);

  size_t total_changed = 0;
  for (const auto& v : per_game_changed) total_changed += v.size();
  double r = static_cast<double>(total_changed) / static_cast<double>(num_games);
  double dU = r * mean_delta;

  // Game-clustered bootstrap
  std::vector<double> sampled_delta_sum(num_games, 0.0);
  std::vector<size_t> sampled_delta_cnt(num_games, 0);
  for (size_t i = 0; i < M; ++i) {
    int g = sampled_decisions[i].game_idx;
    sampled_delta_sum[g] += deltas[i];
    sampled_delta_cnt[g]++;
  }

  std::mt19937 boot_rng(master_seed + 12345);
  std::uniform_int_distribution<size_t> g_dist(0, num_games - 1);
  std::vector<double> boot_dU(10000);
  for (int b = 0; b < 10000; ++b) {
    double boot_c_sum = 0.0;
    double boot_delta_sum = 0.0;
    size_t boot_delta_cnt = 0;
    for (size_t k = 0; k < num_games; ++k) {
      size_t g = g_dist(boot_rng);
      boot_c_sum += static_cast<double>(per_game_changed[g].size());
      boot_delta_sum += sampled_delta_sum[g];
      boot_delta_cnt += sampled_delta_cnt[g];
    }
    double r_boot = boot_c_sum / static_cast<double>(num_games);
    double mean_delta_boot = (boot_delta_cnt > 0) ? (boot_delta_sum / static_cast<double>(boot_delta_cnt)) : mean_delta;
    boot_dU[b] = r_boot * mean_delta_boot;
  }
  std::sort(boot_dU.begin(), boot_dU.end());
  double ci_low = boot_dU[250];
  double ci_high = boot_dU[9750];
  double boot_mean = std::accumulate(boot_dU.begin(), boot_dU.end(), 0.0) / 10000.0;
  double boot_var = 0.0;
  for (double d : boot_dU) {
    double diff = d - boot_mean;
    boot_var += diff * diff;
  }
  double boot_se = std::sqrt(boot_var / 9999.0);

  return TieBreakerStats{mean_delta, dU, boot_se, ci_low, ci_high};
}

}  // namespace open_spiel::dune_imperium::fixed_root_benchmark
