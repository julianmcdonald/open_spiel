#include "open_spiel/examples/dune_fixed_root_search_benchmark_internal.h"

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>

#include "open_spiel/abseil-cpp/absl/flags/flag.h"
#include "open_spiel/abseil-cpp/absl/strings/str_format.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel::dune_imperium::fixed_root_benchmark {

void SaveBenchmarkReceipt(
    const std::string& output_path,
    const open_spiel::json::Object& root_out_json,
    const std::vector<SampledRoot>& sampled_roots,
    const std::vector<VariantConfig>& variants,
    const std::vector<std::vector<VariantResult>>& all_results,
    const std::vector<VariantResult>& recheck_1024_results,
    size_t num_roots_completed) {
  open_spiel::json::Object current_root_json = root_out_json;
  open_spiel::json::Array roots_results_arr;
  roots_results_arr.reserve(num_roots_completed);

  for (size_t r_idx = 0; r_idx < num_roots_completed; ++r_idx) {
    const auto& root = sampled_roots[r_idx];
    open_spiel::json::Object r_obj;
    r_obj["root_id"] = open_spiel::json::Value(static_cast<int64_t>(root.root_id));
    r_obj["game_id"] = open_spiel::json::Value(static_cast<int64_t>(root.game_id));
    r_obj["round"] = open_spiel::json::Value(static_cast<int64_t>(root.round));
    r_obj["round_band"] = open_spiel::json::Value(root.round_band);
    r_obj["decision_type"] = open_spiel::json::Value(root.decision_type);
    r_obj["stratum_weight"] = open_spiel::json::Value(root.stratum_weight);
    r_obj["raw_action"] = open_spiel::json::Value(static_cast<int64_t>(root.raw_action));

    open_spiel::json::Object var_map;
    for (size_t v_idx = 0; v_idx < variants.size(); ++v_idx) {
      const auto& vr = all_results[r_idx][v_idx];
      open_spiel::json::Object vr_obj;
      vr_obj["chosen_action"] = open_spiel::json::Value(static_cast<int64_t>(vr.chosen_action));
      vr_obj["overridden"] = open_spiel::json::Value(vr.overridden);
      vr_obj["raw_mean_utility"] = open_spiel::json::Value(vr.raw_mean_utility);
      vr_obj["chosen_mean_utility"] = open_spiel::json::Value(vr.chosen_mean_utility);
      vr_obj["decision_seed"] = open_spiel::json::Value(static_cast<int64_t>(vr.decision_seed));
      vr_obj["evaluations"] = open_spiel::json::Value(static_cast<int64_t>(vr.evaluations));
      vr_obj["duration_ms"] = open_spiel::json::Value(vr.duration_ms);
      vr_obj["stage1_planned_worlds"] = open_spiel::json::Value(static_cast<int64_t>(vr.stage1_planned_worlds));
      vr_obj["stage1_kept_worlds"] = open_spiel::json::Value(static_cast<int64_t>(vr.stage1_kept_worlds));
      vr_obj["stage2_planned_worlds"] = open_spiel::json::Value(static_cast<int64_t>(vr.stage2_planned_worlds));
      vr_obj["stage2_kept_worlds"] = open_spiel::json::Value(static_cast<int64_t>(vr.stage2_kept_worlds));
      vr_obj["deadline_bound"] = open_spiel::json::Value(vr.deadline_bound);

      open_spiel::json::Array sd_arr;
      for (const auto& sd_pair : vr.candidate_pair_sd_vs_raw) {
        open_spiel::json::Array pair_arr;
        pair_arr.push_back(open_spiel::json::Value(static_cast<int64_t>(sd_pair.first)));
        pair_arr.push_back(open_spiel::json::Value(sd_pair.second));
        sd_arr.push_back(open_spiel::json::Value(pair_arr));
      }
      vr_obj["candidate_pair_sd_vs_raw"] = open_spiel::json::Value(sd_arr);

      open_spiel::json::Array cu_arr;
      for (const auto& cu : vr.candidate_utilities) {
        open_spiel::json::Array pair_arr;
        pair_arr.push_back(open_spiel::json::Value(static_cast<int64_t>(cu.first)));
        pair_arr.push_back(open_spiel::json::Value(cu.second));
        cu_arr.push_back(open_spiel::json::Value(pair_arr));
      }
      vr_obj["candidate_utilities"] = open_spiel::json::Value(cu_arr);

      open_spiel::json::Array cru_arr;
      for (const auto& cru : vr.candidate_rollout_utilities) {
        open_spiel::json::Array pair_arr;
        pair_arr.push_back(open_spiel::json::Value(static_cast<int64_t>(cru.first)));
        open_spiel::json::Array vals_arr;
        vals_arr.reserve(cru.second.size());
        for (double u : cru.second) {
          vals_arr.push_back(open_spiel::json::Value(u));
        }
        pair_arr.push_back(open_spiel::json::Value(vals_arr));
        cru_arr.push_back(open_spiel::json::Value(pair_arr));
      }
      vr_obj["candidate_rollout_utilities"] = open_spiel::json::Value(cru_arr);

      var_map[variants[v_idx].name] = open_spiel::json::Value(vr_obj);
    }

    if (r_idx < recheck_1024_results.size() && recheck_1024_results[r_idx].chosen_action != 0) {
      const auto& rk = recheck_1024_results[r_idx];
      open_spiel::json::Object rk_obj;
      rk_obj["chosen_action"] = open_spiel::json::Value(static_cast<int64_t>(rk.chosen_action));
      rk_obj["overridden"] = open_spiel::json::Value(rk.overridden);
      rk_obj["raw_mean_utility"] = open_spiel::json::Value(rk.raw_mean_utility);
      rk_obj["chosen_mean_utility"] = open_spiel::json::Value(rk.chosen_mean_utility);
      rk_obj["decision_seed"] = open_spiel::json::Value(static_cast<int64_t>(rk.decision_seed));
      rk_obj["evaluations"] = open_spiel::json::Value(static_cast<int64_t>(rk.evaluations));
      rk_obj["duration_ms"] = open_spiel::json::Value(rk.duration_ms);

      open_spiel::json::Array cu_arr;
      for (const auto& cu : rk.candidate_utilities) {
        open_spiel::json::Array pair_arr;
        pair_arr.push_back(open_spiel::json::Value(static_cast<int64_t>(cu.first)));
        pair_arr.push_back(open_spiel::json::Value(cu.second));
        cu_arr.push_back(open_spiel::json::Value(pair_arr));
      }
      rk_obj["candidate_utilities"] = open_spiel::json::Value(cu_arr);

      open_spiel::json::Array cru_arr;
      for (const auto& cru : rk.candidate_rollout_utilities) {
        open_spiel::json::Array pair_arr;
        pair_arr.push_back(open_spiel::json::Value(static_cast<int64_t>(cru.first)));
        open_spiel::json::Array vals_arr;
        vals_arr.reserve(cru.second.size());
        for (double u : cru.second) {
          vals_arr.push_back(open_spiel::json::Value(u));
        }
        pair_arr.push_back(open_spiel::json::Value(vals_arr));
        cru_arr.push_back(open_spiel::json::Value(pair_arr));
      }
      rk_obj["candidate_rollout_utilities"] = open_spiel::json::Value(cru_arr);

      var_map["Ref1024_Recheck"] = open_spiel::json::Value(rk_obj);
    }

    r_obj["variants"] = open_spiel::json::Value(var_map);
    roots_results_arr.push_back(open_spiel::json::Value(r_obj));
  }
  current_root_json["roots_results"] = open_spiel::json::Value(roots_results_arr);

  std::string tmp_path = output_path + ".tmp";
  std::ofstream out_file(tmp_path);
  if (out_file.is_open()) {
    out_file << open_spiel::json::ToString(current_root_json);
    out_file.close();
    std::rename(tmp_path.c_str(), output_path.c_str());
  }
}

void ReportLeafCheckAndSave(
    const std::string& output_path,
    const std::string& checkpoint_path,
    const LeafCheckStats& leaf_stats,
    const std::vector<SampledRoot>& sampled_roots,
    const std::vector<VariantConfig>& variants,
    const std::vector<std::vector<VariantResult>>& all_results) {
  std::cout << "\n====================================================================================================\n";
  std::cout << absl::StrFormat(
      "LEAF CHECK (N=%zu roots, N=%lld candidate pairs): V2_16_Paired vs Ref512 correlation = %.4f (U28700 baseline: ~0.8)\n",
      sampled_roots.size(), static_cast<long long>(leaf_stats.candidate_pairs), leaf_stats.correlation);
  std::cout << "====================================================================================================\n";
  open_spiel::json::Object leaf_out_json;
  leaf_out_json["leaf_check_v2_ref_correlation"] = open_spiel::json::Value(leaf_stats.correlation);
  leaf_out_json["leaf_check_candidate_pairs"] = open_spiel::json::Value(leaf_stats.candidate_pairs);
  SaveBenchmarkReceipt(output_path, leaf_out_json, sampled_roots, variants, all_results, {}, sampled_roots.size());
  std::remove(checkpoint_path.c_str());
}

open_spiel::json::Object PrintAndBuildMainBenchmarkReport(
    const std::string& output_path,
    const BenchmarkSummaryReportData& summary_data,
    const std::vector<SampledRoot>& sampled_roots,
    const std::vector<VariantConfig>& variants,
    const std::vector<std::vector<VariantResult>>& all_results) {
  // MUST FIX 4(b): Summary table drops / labels reference baseline and reports Evals/Decision
  std::cout << "====================================================================================================\n";
  std::cout << "| Candidate Variant | Rollouts | Pairing  | Mode      | Agreement (%) | Mean True Gain* | Evals/Dec |\n";
  std::cout << "| :---              | :---     | :---     | :---      | :---          | :---            | :---      |\n";

  open_spiel::json::Object root_out_json;
  open_spiel::json::Array variant_summaries;

  for (const auto& row : summary_data.variant_summaries) {
    std::cout << absl::StrFormat(
        "| %-17s | %-8d | %-8s | %-9s | %12.1f%% | %14.3f* | %9.1f |\n",
        row.name, row.rollouts, row.pairing_str, row.mode_str,
        row.agree_pct, row.mean_gain, row.avg_evals);

    open_spiel::json::Object v_sum_obj;
    v_sum_obj["name"] = open_spiel::json::Value(row.name);
    v_sum_obj["rollouts"] = open_spiel::json::Value(static_cast<int64_t>(row.rollouts));
    v_sum_obj["pairing"] = open_spiel::json::Value(row.pairing_str);
    v_sum_obj["truncation"] = open_spiel::json::Value(row.mode_str);
    v_sum_obj["agreement_pct"] = open_spiel::json::Value(row.agree_pct);
    v_sum_obj["mean_true_gain"] = open_spiel::json::Value(row.mean_gain);
    v_sum_obj["avg_evaluations"] = open_spiel::json::Value(row.avg_evals);
    v_sum_obj["avg_time_ms"] = open_spiel::json::Value(row.avg_time_ms);
    v_sum_obj["overrides"] = open_spiel::json::Value(static_cast<int64_t>(row.override_count));
    variant_summaries.push_back(open_spiel::json::Value(v_sum_obj));
  }

  std::cout << "====================================================================================================\n";
  std::cout << absl::StrFormat(
      "[Reference Baseline] %s: %d rollouts | Paired | Terminal | Avg Evals/Dec: %.1f | Avg Time: %.1f ms\n",
      summary_data.ref_name, summary_data.ref_rollouts,
      summary_data.ref_avg_evals, summary_data.ref_avg_time_ms);
  std::cout << "====================================================================================================\n";
  std::cout << absl::StrFormat("* True Gain is measured against independent %d-rollout paired reference utilities.\n\n",
                               summary_data.ref_rollouts);

  root_out_json["variant_summaries"] = open_spiel::json::Value(variant_summaries);

  // Save complete main sweep immediately so results are preserved before recheck begins
  SaveBenchmarkReceipt(output_path, root_out_json, sampled_roots, variants, all_results, {}, sampled_roots.size());
  std::cout << "[RECEIPT] Saved complete main sweep (all variants across all roots) to " << output_path << "\n";
  return root_out_json;
}

void PrintAndSaveHeldOutReplayReport(
    const std::string& model_path,
    const std::string& trajectory_jsonl_path,
    const std::string& obs_bin_path,
    const std::string& output_json_path,
    const HeldOutReplayStats& stats,
    const HeldOutReplayComputedMetrics& metrics) {
  std::cout << "================================================================================\n";
  std::cout << "HELD-OUT EVALUATION AGAINST RECORDED U28700 PRIOR (mu)\n";
  std::cout << "================================================================================\n";
  std::cout << absl::StrFormat("Total fresh_search decisions evaluated: %lld\n\n",
                               static_cast<long long>(stats.fresh_search_decisions));

  std::cout << absl::StrFormat("--- OVERRIDE STATES (N = %lld, %.2f%%) ---\n",
                               static_cast<long long>(stats.ovr_stats.count), metrics.override_pct);
  std::cout << absl::StrFormat("  Greedy change rate [argmax(p) != argmax(mu)]:       %6.2f%% (%lld / %lld)\n",
                               metrics.ovr_greedy_change_rate,
                               static_cast<long long>(stats.ovr_stats.greedy_changed_count),
                               static_cast<long long>(stats.ovr_stats.count));
  std::cout << absl::StrFormat("  Greedy agreement with search move:                  %6.2f%% (%lld / %lld)\n",
                               metrics.ovr_greedy_agree_search,
                               static_cast<long long>(stats.ovr_stats.greedy_agree_search_count),
                               static_cast<long long>(stats.ovr_stats.count));
  std::cout << absl::StrFormat("  Mean probability on search move:                    %6.4f (U28700 baseline: 0.0887)\n",
                               metrics.ovr_mean_prob_search);
  std::cout << absl::StrFormat("  Mean max |p - mu|:                                  %6.4f\n",
                               metrics.ovr_mean_max_p_diff);
  std::cout << absl::StrFormat("  Mean policy entropy:                                %6.4f\n",
                               metrics.ovr_mean_entropy);
  std::cout << absl::StrFormat("  Value of changed choices (mean utility diff):       %+6.4f ladder units (N=%lld covered)\n",
                               metrics.ovr_val_changed_diff,
                               static_cast<long long>(stats.ovr_stats.value_covered_count));
  std::cout << absl::StrFormat("  Mean CE (MPO target):                               %6.4f\n\n",
                               metrics.mean_ce_override);

  std::cout << absl::StrFormat("--- NON-OVERRIDE STATES (N = %lld, %.2f%%) ---\n",
                               static_cast<long long>(stats.non_ovr_stats.count), 100.0 - metrics.override_pct);
  std::cout << absl::StrFormat("  Greedy change rate [argmax(p) != argmax(mu)]:       %6.2f%% (%lld / %lld)\n",
                               metrics.non_ovr_greedy_change_rate,
                               static_cast<long long>(stats.non_ovr_stats.greedy_changed_count),
                               static_cast<long long>(stats.non_ovr_stats.count));
  std::cout << absl::StrFormat("  Mean max |p - mu|:                                  %6.4f\n",
                               metrics.non_ovr_mean_max_p_diff);
  std::cout << absl::StrFormat("  Mean policy entropy:                                %6.4f\n",
                               metrics.non_ovr_mean_entropy);
  std::cout << absl::StrFormat("  Value of changed choices (mean utility diff):       %+6.4f ladder units (N=%lld covered)\n",
                               metrics.non_ovr_val_changed_diff,
                               static_cast<long long>(stats.non_ovr_stats.value_covered_count));
  std::cout << absl::StrFormat("  Mean CE (MPO target):                               %6.4f\n",
                               metrics.mean_ce_non_override);
  std::cout << "--------------------------------------------------------------------------------\n";
  std::cout << absl::StrFormat("  All States Mean CE:                                 %6.4f\n", metrics.mean_ce_all);
  std::cout << absl::StrFormat("  All States MPO Target Top-1 Match:                  %6.2f%%\n", metrics.top1_acc);
  std::cout << absl::StrFormat("  All States Mean Max Prob Movement (vs MPO target):  %6.4f\n", metrics.mean_max_diff);
  std::cout << "--------------------------------------------------------------------------------\n";
  std::cout << "Per-Round Cross-Entropy (All States):\n";
  for (int r = 1; r <= 10; ++r) {
    auto cnt_it = stats.round_dec_count.find(r);
    if (cnt_it != stats.round_dec_count.end() && cnt_it->second > 0) {
      std::cout << absl::StrFormat("  Round %2d: CE = %.4f (%lld decisions)\n",
                                   r, stats.round_ce_sum.at(r) / cnt_it->second,
                                   static_cast<long long>(cnt_it->second));
    }
  }
  std::cout << "================================================================================\n";

  if (!output_json_path.empty()) {
    open_spiel::json::Object out_obj;
    out_obj["model_checkpoint"] = open_spiel::json::Value(model_path);
    out_obj["trajectory_meta_jsonl"] = open_spiel::json::Value(trajectory_jsonl_path);
    out_obj["trajectory_obs_bin"] = open_spiel::json::Value(obs_bin_path);
    out_obj["fresh_search_decisions"] = open_spiel::json::Value(stats.fresh_search_decisions);

    // Override subgroup
    open_spiel::json::Object ovr_json;
    ovr_json["count"] = open_spiel::json::Value(stats.ovr_stats.count);
    ovr_json["override_pct"] = open_spiel::json::Value(metrics.override_pct);
    ovr_json["greedy_change_rate_pct"] = open_spiel::json::Value(metrics.ovr_greedy_change_rate);
    ovr_json["greedy_agree_search_pct"] = open_spiel::json::Value(metrics.ovr_greedy_agree_search);
    ovr_json["mean_prob_search_move"] = open_spiel::json::Value(metrics.ovr_mean_prob_search);
    ovr_json["mean_max_p_diff_mu"] = open_spiel::json::Value(metrics.ovr_mean_max_p_diff);
    ovr_json["mean_policy_entropy"] = open_spiel::json::Value(metrics.ovr_mean_entropy);
    ovr_json["value_changed_utility_diff"] = open_spiel::json::Value(metrics.ovr_val_changed_diff);
    ovr_json["value_changed_count_covered"] = open_spiel::json::Value(stats.ovr_stats.value_covered_count);
    ovr_json["mean_cross_entropy"] = open_spiel::json::Value(metrics.mean_ce_override);
    out_obj["override_states"] = open_spiel::json::Value(ovr_json);

    // Non-override subgroup
    open_spiel::json::Object non_ovr_json;
    non_ovr_json["count"] = open_spiel::json::Value(stats.non_ovr_stats.count);
    non_ovr_json["pct"] = open_spiel::json::Value(100.0 - metrics.override_pct);
    non_ovr_json["greedy_change_rate_pct"] = open_spiel::json::Value(metrics.non_ovr_greedy_change_rate);
    non_ovr_json["mean_max_p_diff_mu"] = open_spiel::json::Value(metrics.non_ovr_mean_max_p_diff);
    non_ovr_json["mean_policy_entropy"] = open_spiel::json::Value(metrics.non_ovr_mean_entropy);
    non_ovr_json["value_changed_utility_diff"] = open_spiel::json::Value(metrics.non_ovr_val_changed_diff);
    non_ovr_json["value_changed_count_covered"] = open_spiel::json::Value(stats.non_ovr_stats.value_covered_count);
    non_ovr_json["mean_cross_entropy"] = open_spiel::json::Value(metrics.mean_ce_non_override);
    out_obj["non_override_states"] = open_spiel::json::Value(non_ovr_json);

    out_obj["temperature"] = open_spiel::json::Value(absl::GetFlag(FLAGS_temperature));
    out_obj["mean_policy_entropy_non_override"] = open_spiel::json::Value(metrics.non_ovr_mean_entropy);
    out_obj["mean_policy_entropy_override"] = open_spiel::json::Value(metrics.ovr_mean_entropy);
    if (absl::GetFlag(FLAGS_dump_per_state)) {
      out_obj["decisions"] = open_spiel::json::Value(stats.dumped_decisions);
    }

    // Legacy fields for backward compatibility
    out_obj["override_decisions"] = open_spiel::json::Value(stats.ovr_stats.count);
    out_obj["override_pct"] = open_spiel::json::Value(metrics.override_pct);
    out_obj["mean_teacher_prob_override"] = open_spiel::json::Value(metrics.ovr_mean_prob_search);
    out_obj["mean_cross_entropy_all"] = open_spiel::json::Value(metrics.mean_ce_all);
    out_obj["mean_cross_entropy_override"] = open_spiel::json::Value(metrics.mean_ce_override);
    out_obj["top1_accuracy_pct"] = open_spiel::json::Value(metrics.top1_acc);
    out_obj["mean_max_prob_diff"] = open_spiel::json::Value(metrics.mean_max_diff);

    open_spiel::json::Object rounds_obj;
    for (const auto& kv : stats.round_dec_count) {
      open_spiel::json::Object r_info;
      r_info["count"] = open_spiel::json::Value(kv.second);
      r_info["mean_ce"] = open_spiel::json::Value(stats.round_ce_sum.at(kv.first) / kv.second);
      rounds_obj[std::to_string(kv.first)] = open_spiel::json::Value(r_info);
    }
    out_obj["round_breakdown"] = open_spiel::json::Value(rounds_obj);

    std::ofstream out_file(output_json_path);
    if (out_file.is_open()) {
      out_file << open_spiel::json::ToString(open_spiel::json::Value(out_obj)) << "\n";
      out_file.close();
      std::cout << "[OUTPUT] Saved receipt to: " << output_json_path << "\n";
    }
  }
}

void WritePreservationItem(std::ofstream& ofs, const CapturedPreservationItem& item) {
  WritePod(ofs, item.game_id);
  WritePod(ofs, item.split_id);
  WritePod(ofs, item.player);
  WriteVector(ofs, item.obs);
  WriteVector(ofs, item.cand_data.actions);
  WriteVector(ofs, item.cand_data.features);
  WriteVector(ofs, item.cand_data.card_ids);
  WriteVector(ofs, item.cand_data.space_ids);
  WriteVector(ofs, item.cand_data.supported);
  WriteVector(ofs, item.cand_data.roles);
  WriteVector(ofs, item.legal_actions);
  WriteVector(ofs, item.pi_parent);
}

void SavePreservationCheckpoint(
    const std::string& out_dir,
    int num_games,
    const std::atomic<uint8_t>* game_completed,
    const std::vector<std::vector<CapturedPreservationItem>>& per_game_pres) {
  std::string tmp_ckpt = out_dir + "/preservation_pool.bin.ckpt.tmp";
  std::string ckpt_file = out_dir + "/preservation_pool.bin.ckpt";
  std::ofstream ofs(tmp_ckpt, std::ios::binary);
  if (!ofs.is_open()) return;

  WritePod(ofs, 0x50434B50U);
  WritePod(ofs, 1U);
  uint64_t completed_count = 0;
  for (int i = 0; i < num_games; ++i) {
    if (game_completed[i].load(std::memory_order_acquire) != 0) {
      completed_count++;
    }
  }
  WritePod(ofs, completed_count);
  for (int i = 0; i < num_games; ++i) {
    if (game_completed[i].load(std::memory_order_acquire) != 0) {
      WritePod(ofs, i);
      uint64_t n_items = per_game_pres[i].size();
      WritePod(ofs, n_items);
      for (const auto& item : per_game_pres[i]) {
        WritePreservationItem(ofs, item);
      }
    }
  }
  ofs.close();
  std::error_code ec;
  std::filesystem::rename(tmp_ckpt, ckpt_file, ec);
  if (!ec) {
    std::cout << absl::StrFormat("[CAPTURE] Checkpointed %zu completed games to %s\n",
                                 completed_count, ckpt_file);
  }
}

void SaveCaptureOutputs(
    const open_spiel::Game& game,
    const std::string& out_dir,
    int num_games,
    uint64_t base_seed,
    const std::vector<CapturedPlacementRoot>& all_sampled_roots,
    const std::vector<CapturedPreservationItem>& all_pres,
    double elapsed_s) {
  std::filesystem::create_directories(out_dir);

  // 1. Write sampled_roots.json
  open_spiel::json::Object meta;
  meta["game_string"] = open_spiel::json::Value(game.ToString());
  meta["total_games"] = open_spiel::json::Value(static_cast<int64_t>(num_games));
  meta["total_roots"] = open_spiel::json::Value(static_cast<int64_t>(all_sampled_roots.size()));
  meta["master_seed"] = open_spiel::json::Value(static_cast<int64_t>(base_seed));

  open_spiel::json::Array roots_arr;
  roots_arr.reserve(all_sampled_roots.size());
  for (size_t i = 0; i < all_sampled_roots.size(); ++i) {
    const auto& r = all_sampled_roots[i];
    open_spiel::json::Object r_obj;
    r_obj["root_id"] = open_spiel::json::Value(static_cast<int64_t>(i));
    r_obj["game_id"] = open_spiel::json::Value(static_cast<int64_t>(r.game_id));
    r_obj["decision_ordinal"] = open_spiel::json::Value(static_cast<int64_t>(r.decision_ordinal));
    r_obj["obs_row"] = open_spiel::json::Value(static_cast<int64_t>(i));
    r_obj["search_seat"] = open_spiel::json::Value(static_cast<int64_t>(r.search_seat));
    r_obj["round"] = open_spiel::json::Value(static_cast<int64_t>(r.round));
    r_obj["round_band"] = open_spiel::json::Value(r.round_band);
    r_obj["decision_type"] = open_spiel::json::Value(r.decision_type);
    r_obj["stratum_weight"] = open_spiel::json::Value(1.0);
    r_obj["raw_action"] = open_spiel::json::Value(static_cast<int64_t>(r.raw_action));
    r_obj["census_chosen_action"] = open_spiel::json::Value(static_cast<int64_t>(r.raw_action));
    r_obj["census_overridden"] = open_spiel::json::Value(false);

    open_spiel::json::Array la_arr;
    for (int a : r.legal_actions) la_arr.push_back(open_spiel::json::Value(static_cast<int64_t>(a)));
    r_obj["legal_actions"] = open_spiel::json::Value(la_arr);

    open_spiel::json::Array ah_arr;
    for (int64_t a : r.action_history) ah_arr.push_back(open_spiel::json::Value(a));
    r_obj["action_history"] = open_spiel::json::Value(ah_arr);

    r_obj["split"] = open_spiel::json::Value(r.split);

    open_spiel::json::Array pi_arr;
    for (double p : r.pi_parent) pi_arr.push_back(open_spiel::json::Value(p));
    r_obj["pi_parent"] = open_spiel::json::Value(pi_arr);

    roots_arr.push_back(open_spiel::json::Value(r_obj));
  }

  open_spiel::json::Object root_out_json;
  root_out_json["metadata"] = open_spiel::json::Value(meta);
  root_out_json["roots"] = open_spiel::json::Value(roots_arr);

  std::string roots_json_path = out_dir + "/sampled_roots.json";
  {
    std::ofstream ofs(roots_json_path);
    if (!ofs.is_open()) open_spiel::SpielFatalError("Failed to write to " + roots_json_path);
    ofs << open_spiel::json::ToString(open_spiel::json::Value(root_out_json), true) << "\n";
  }
  std::cout << "[SAVED] " << roots_json_path << " (" << all_sampled_roots.size() << " roots)\n";

  // 2. Write sampled_obs.bin
  std::string obs_bin_path = out_dir + "/sampled_obs.bin";
  {
    std::ofstream ofs(obs_bin_path, std::ios::binary);
    if (!ofs.is_open()) open_spiel::SpielFatalError("Failed to write to " + obs_bin_path);
    for (size_t i = 0; i < all_sampled_roots.size(); ++i) {
      ofs.write(reinterpret_cast<const char*>(all_sampled_roots[i].obs.data()),
                all_sampled_roots[i].obs.size() * sizeof(float));
    }
  }
  std::cout << "[SAVED] " << obs_bin_path << "\n";

  // 3. Write sampled_cand.bin
  std::string cand_bin_path = out_dir + "/sampled_cand.bin";
  {
    std::ofstream ofs(cand_bin_path, std::ios::binary);
    if (!ofs.is_open()) open_spiel::SpielFatalError("Failed to write to " + cand_bin_path);
    WritePod(ofs, 0x43414E44U);
    WritePod(ofs, 1U);
    uint64_t nr = all_sampled_roots.size();
    WritePod(ofs, nr);
    for (size_t i = 0; i < all_sampled_roots.size(); ++i) {
      const auto& cd = all_sampled_roots[i].cand_data;
      WriteVector(ofs, cd.actions);
      WriteVector(ofs, cd.features);
      WriteVector(ofs, cd.card_ids);
      WriteVector(ofs, cd.space_ids);
      WriteVector(ofs, cd.supported);
      WriteVector(ofs, cd.roles);
    }
  }
  std::cout << "[SAVED] " << cand_bin_path << "\n";

  // 4. Write preservation_pool.bin
  std::string pres_bin_path = out_dir + "/preservation_pool.bin";
  {
    std::ofstream ofs(pres_bin_path, std::ios::binary);
    if (!ofs.is_open()) open_spiel::SpielFatalError("Failed to write to " + pres_bin_path);
    WritePod(ofs, 0x50524553U);
    WritePod(ofs, 1U);
    uint64_t np = all_pres.size();
    WritePod(ofs, np);
    for (size_t i = 0; i < all_pres.size(); ++i) {
      WritePreservationItem(ofs, all_pres[i]);
    }
  }
  std::cout << "[SAVED] " << pres_bin_path << " (" << all_pres.size() << " preservation decisions)\n";

  std::remove((out_dir + "/preservation_pool.bin.ckpt").c_str());
  std::remove((out_dir + "/preservation_pool.bin.ckpt.tmp").c_str());
  (void)elapsed_s;
}

open_spiel::json::Object LabelledResultToJson(const LabelledResult& lr) {
  open_spiel::json::Object r_obj;
  r_obj["root_id"] = open_spiel::json::Value(static_cast<int64_t>(lr.root_id));
  r_obj["game_id"] = open_spiel::json::Value(static_cast<int64_t>(lr.game_id));
  r_obj["obs_row"] = open_spiel::json::Value(static_cast<int64_t>(lr.obs_row));
  r_obj["search_seat"] = open_spiel::json::Value(static_cast<int64_t>(lr.search_seat));
  r_obj["round"] = open_spiel::json::Value(static_cast<int64_t>(lr.round));
  r_obj["decision_type"] = open_spiel::json::Value(lr.decision_type);
  r_obj["split"] = open_spiel::json::Value(lr.split);
  r_obj["raw_action"] = open_spiel::json::Value(static_cast<int64_t>(lr.raw_action));
  r_obj["chosen_action"] = open_spiel::json::Value(static_cast<int64_t>(lr.chosen_action));
  r_obj["overridden"] = open_spiel::json::Value(lr.overridden);
  r_obj["d2"] = open_spiel::json::Value(lr.d2);
  r_obj["se2"] = open_spiel::json::Value(lr.se2);
  r_obj["w"] = open_spiel::json::Value(lr.w);
  open_spiel::json::Array c_arr;
  for (int a : lr.candidates) c_arr.push_back(open_spiel::json::Value(static_cast<int64_t>(a)));
  r_obj["candidates"] = open_spiel::json::Value(c_arr);
  open_spiel::json::Array la_arr;
  for (int a : lr.legal_actions) la_arr.push_back(open_spiel::json::Value(static_cast<int64_t>(a)));
  r_obj["legal_actions"] = open_spiel::json::Value(la_arr);
  open_spiel::json::Array pi_arr;
  for (double p : lr.pi_parent) pi_arr.push_back(open_spiel::json::Value(p));
  r_obj["pi_parent"] = open_spiel::json::Value(pi_arr);
  r_obj["duration_ms"] = open_spiel::json::Value(lr.duration_ms);
  r_obj["stage1_kept_worlds"] = open_spiel::json::Value(static_cast<int64_t>(lr.stage1_kept_worlds));
  r_obj["stage2_kept_worlds"] = open_spiel::json::Value(static_cast<int64_t>(lr.stage2_kept_worlds));
  return r_obj;
}

void SaveDistillLabelCheckpoint(
    const std::string& path,
    size_t total_roots,
    int completed_count,
    const std::vector<LabelledResult>& results,
    const std::atomic<uint8_t>* completed) {
  open_spiel::json::Object c_obj;
  c_obj["total_roots"] = open_spiel::json::Value(static_cast<int64_t>(total_roots));
  c_obj["completed_roots"] = open_spiel::json::Value(static_cast<int64_t>(completed_count));
  open_spiel::json::Array arr;
  for (size_t i = 0; i < results.size(); ++i) {
    if (completed[i].load(std::memory_order_acquire) == 0) continue;
    arr.push_back(open_spiel::json::Value(LabelledResultToJson(results[i])));
  }
  c_obj["roots"] = open_spiel::json::Value(arr);
  std::string tmp_path = path + ".tmp";
  std::ofstream ofs(tmp_path);
  if (ofs.is_open()) {
    ofs << open_spiel::json::ToString(open_spiel::json::Value(c_obj), true);
    ofs.close();
    std::filesystem::rename(tmp_path, path);
  }
}

void SaveDistillLabelFinalOutput(
    const std::string& output_path,
    const std::string& ckpt_path,
    const std::vector<LabelledResult>& results,
    const DistillLabelStats& stats) {
  open_spiel::json::Object meta_out;
  meta_out["total_roots"] = open_spiel::json::Value(static_cast<int64_t>(results.size()));
  meta_out["wall_time_sec"] = open_spiel::json::Value(stats.total_wall_s);
  meta_out["roots_per_sec"] = open_spiel::json::Value(stats.roots_per_s);
  meta_out["override_rate_pct"] = open_spiel::json::Value(stats.override_rate);
  meta_out["total_overrides"] = open_spiel::json::Value(static_cast<int64_t>(stats.total_overrides));
  meta_out["mean_weight_w"] = open_spiel::json::Value(stats.mean_w);

  open_spiel::json::Array arr_out;
  for (const auto& lr : results) {
    arr_out.push_back(open_spiel::json::Value(LabelledResultToJson(lr)));
  }

  open_spiel::json::Object final_out;
  final_out["metadata"] = open_spiel::json::Value(meta_out);
  final_out["roots"] = open_spiel::json::Value(arr_out);

  {
    std::ofstream ofs(output_path);
    if (!ofs.is_open()) open_spiel::SpielFatalError("Failed to open " + output_path);
    ofs << open_spiel::json::ToString(open_spiel::json::Value(final_out), true) << "\n";
  }
  std::remove(ckpt_path.c_str());
  std::cout << absl::StrFormat(
      "[LABEL] Finished labelling %zu roots in %.1f s (%.2f roots/s). Overrides: %d (%.1f%%), mean w = %.4f. Saved to %s\n",
      results.size(), stats.total_wall_s, stats.roots_per_s,
      stats.total_overrides, stats.override_rate, stats.mean_w, output_path);
}

void PrintAndSaveTieBreakerSummary(
    const std::string& output_path,
    size_t num_games,
    size_t num_changed,
    double r_changed,
    bool has_rollout_stats,
    size_t sampled_count,
    int tie_breaker_rollouts,
    const TieBreakerStats& tb_stats) {
  open_spiel::json::Object summary;
  summary["total_games"] = open_spiel::json::Value(static_cast<int64_t>(num_games));
  summary["total_changed_decisions"] = open_spiel::json::Value(static_cast<int64_t>(num_changed));
  summary["r_changed_per_game"] = open_spiel::json::Value(r_changed);
  summary["replay_verified"] = open_spiel::json::Value(true);

  if (has_rollout_stats) {
    std::cout << "================================================================================\n";
    std::cout << absl::StrFormat("  Tie-Breaker dU: %.4f (SE: %.4f, 95%% CI: [%.4f, %.4f])\n",
                                 tb_stats.dU, tb_stats.boot_se, tb_stats.ci_low, tb_stats.ci_high);
    std::cout << absl::StrFormat("  Mean Delta:     %.4f over %zu moves\n", tb_stats.mean_delta, sampled_count);
    std::cout << "================================================================================\n";

    summary["sampled_changed_decisions"] = open_spiel::json::Value(static_cast<int64_t>(sampled_count));
    summary["mean_delta"] = open_spiel::json::Value(tb_stats.mean_delta);
    summary["du_per_seat_game"] = open_spiel::json::Value(tb_stats.dU);
    summary["du_bootstrap_se"] = open_spiel::json::Value(tb_stats.boot_se);
    summary["du_bootstrap_ci_low"] = open_spiel::json::Value(tb_stats.ci_low);
    summary["du_bootstrap_ci_high"] = open_spiel::json::Value(tb_stats.ci_high);
    summary["rollouts_per_action"] = open_spiel::json::Value(static_cast<int64_t>(tie_breaker_rollouts));
  }

  if (!output_path.empty()) {
    std::ofstream ofs(output_path);
    if (ofs.is_open()) {
      ofs << open_spiel::json::ToString(open_spiel::json::Value(summary), true) << "\n";
      std::cout << "[SAVED] Summary written to " << output_path << "\n";
    }
  }
}

}  // namespace open_spiel::dune_imperium::fixed_root_benchmark
