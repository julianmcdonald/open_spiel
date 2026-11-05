// Copyright 2026 DeepMind Technologies Limited.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// dune_ord1a_cpu_audit: CPU Diagnostics & Symmetry/Session Audit (Ord 1a).
// Performs:
// 1. Part A: Contender Saturation Audit on natural self-play trajectories.
//    - Evaluates z_1, z_2, z_3, z_min, cap derivatives sech^2(z_i / 10), pre/post gaps.
//    - Distinguishes overall and per-role saturation (AgentPrimary, AgentContinuation, Purchase, CombatIntrigue, etc.).
//    - Measures false-alarm rate of legacy pre-cap max-abs saturation vs actual top-contender saturation.
// 2. Part B: Market Slot-Swap Symmetry Audit across market families:
//    - Imperium Row (BuyImperiumRow[0..4], 4 non-identity permutations)
//    - Tleilaxu Row (AcquireTleilaxu[0..1], 91 <-> 92 swap, natural & controlled specimen slices, 2-slot avg policy)
//    - Tech Stacks (AcquireTech[0..2] / AcquireTechWithSolari[0..2], 5 non-identity permutations, market + stack sizes)
//    - Helena Reserve (BuyHelenaReserve[0..k-1], reversal/permutation alignment)

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <numeric>
#include <random>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

#include <torch/torch.h>

#include "open_spiel/spiel.h"
#include "open_spiel/games/dune_imperium/dune_imperium.h"
#include "open_spiel/games/dune_imperium/dune_imperium_board.h"
#include "open_spiel/games/dune_imperium/dune_imperium_common.h"
#include "open_spiel/games/dune_imperium/dune_imperium_content.h"
#include "open_spiel/games/dune_imperium/dune_imperium_test_utils.h"

#include "dune_network.h"
#include "dune_semantic_action_scorer.h"
#include "dune_search_routing.h"
#include "open_spiel/utils/json.h"
#include "open_spiel/abseil-cpp/absl/flags/flag.h"
#include "open_spiel/abseil-cpp/absl/flags/parse.h"

ABSL_FLAG(std::string, model_path,
          "/home/warcr/projects/dune_drl/models/champion_policy.pt",
          "Path to policy checkpoint (.pt)");
ABSL_FLAG(int, num_games, 50,
          "Number of self-play games for saturation audit");
ABSL_FLAG(int, threads, 8,
          "Number of CPU threads for PyTorch operations");
ABSL_FLAG(int, seed, 20261104,
          "Random seed for reproducible self-play trajectories");
ABSL_FLAG(std::string, output_json_path,
          "/home/warcr/dune_drl_runtime/round7/_ord1a_cpu_audit/cpp_audit_summary.json",
          "Path to export audit results as JSON");

namespace open_spiel {
namespace dune_imperium {
namespace {

inline bool HasAction(const std::vector<Action>& acts, Action a) {
  return std::find(acts.begin(), acts.end(), a) != acts.end();
}

inline double CalcSech2(double z) {
  const double t = std::tanh(z / 10.0);
  return 1.0 - t * t;
}

double Percentile(std::vector<double>& vals, double pct) {
  if (vals.empty()) return 0.0;
  std::sort(vals.begin(), vals.end());
  const double idx = pct * (vals.size() - 1);
  const size_t lower = static_cast<size_t>(std::floor(idx));
  const size_t upper = static_cast<size_t>(std::ceil(idx));
  if (lower == upper) return vals[lower];
  const double frac = idx - static_cast<double>(lower);
  return vals[lower] * (1.0 - frac) + vals[upper] * frac;
}

std::string RoleToString(DuneDecisionRole role) {
  switch (role) {
    case DuneDecisionRole::kAgentPrimary: return "AgentPrimary";
    case DuneDecisionRole::kAgentContinuation: return "AgentContinuation";
    case DuneDecisionRole::kPurchase: return "Purchase";
    case DuneDecisionRole::kCombatIntrigue: return "CombatIntrigue";
    case DuneDecisionRole::kLeaderSelection: return "LeaderSelection";
    case DuneDecisionRole::kOtherOptional: return "OtherOptional";
    case DuneDecisionRole::kForcedOrBookkeeping: return "ForcedOrBookkeeping";
  }
  return "Unknown";
}

struct PolicyEvalResult {
  std::vector<Action> legal_actions;
  std::vector<float> raw_legal_logits;
  float mean_legal = 0.0f;
  std::vector<float> z;     // Centered pre-cap logits: raw - mean_legal
  std::vector<float> c;     // Capped logits: 10.0 * tanh(z / 10.0)
  std::vector<float> probs; // Softmax over c
  float value = 0.0f;
  float pre_max_abs = 0.0f;
};

PolicyEvalResult EvaluatePolicyAtState(
    const std::shared_ptr<SharedDunePolicyValueNetImpl>& model,
    const DuneImperiumState& state,
    Player p,
    const std::vector<Action>& legal,
    torch::Device device) {
  PolicyEvalResult res;
  res.legal_actions = legal;
  if (legal.empty()) return res;

  const std::vector<float> obs = state.InformationStateTensorWithAppendix(
      p, MarketAppendixMode::kFullPublicInformationV4, DeckPoolSemantics::kFixed);
  torch::Tensor obs_tensor = torch::from_blob(
      const_cast<float*>(obs.data()),
      {1, static_cast<long>(obs.size())},
      torch::kFloat32).to(device);

  dune_semantic::CandidateActionData cand_data;
  dune_semantic::ExtractCandidateDescriptors(
      state, legal, &cand_data, "semantic_action_v3");

  std::vector<float> full_raw_logits(2391, 0.0f);
  {
    torch::NoGradGuard no_grad;
    auto out = model->forward(obs_tensor);
    std::vector<const dune_semantic::CandidateActionData*> cands = {&cand_data};
    dune_semantic::ApplySemanticScorerBatch(
        model->semantic_scorer_, out.trunk, cands, out.logits, device);

    torch::Tensor logits_cpu = out.logits.to(torch::kCPU).contiguous();
    std::memcpy(full_raw_logits.data(), logits_cpu.data_ptr<float>(),
                2391 * sizeof(float));
    res.value = out.values[0][0].item<float>();
  }

  res.raw_legal_logits.resize(legal.size());
  double legal_sum = 0.0;
  for (size_t i = 0; i < legal.size(); ++i) {
    const Action a = legal[i];
    const float val = (a >= 0 && a < 2391) ? full_raw_logits[a] : 0.0f;
    res.raw_legal_logits[i] = val;
    legal_sum += val;
  }
  res.mean_legal = static_cast<float>(legal_sum / legal.size());

  res.z.resize(legal.size());
  res.c.resize(legal.size());
  res.pre_max_abs = 0.0f;
  float max_c = -1e9f;
  for (size_t i = 0; i < legal.size(); ++i) {
    const float z_val = res.raw_legal_logits[i] - res.mean_legal;
    res.z[i] = z_val;
    if (std::abs(z_val) > res.pre_max_abs) {
      res.pre_max_abs = std::abs(z_val);
    }
    const float c_val = 10.0f * std::tanh(z_val / 10.0f);
    res.c[i] = c_val;
    if (c_val > max_c) {
      max_c = c_val;
    }
  }

  res.probs.resize(legal.size());
  double exp_sum = 0.0;
  for (size_t i = 0; i < legal.size(); ++i) {
    const float exp_c = std::exp(res.c[i] - max_c);
    res.probs[i] = exp_c;
    exp_sum += exp_c;
  }
  const float norm = (exp_sum > 0.0) ? static_cast<float>(exp_sum) : 1.0f;
  for (size_t i = 0; i < legal.size(); ++i) {
    res.probs[i] /= norm;
  }

  return res;
}

struct NumericSummary {
  int count = 0;
  double mean = 0.0;
  double p10 = 0.0;
  double p50 = 0.0;
  double p90 = 0.0;
  double p99 = 0.0;
  double max_val = 0.0;

  static NumericSummary FromVector(std::vector<double> vals, bool include_p10 = false) {
    NumericSummary s;
    s.count = static_cast<int>(vals.size());
    if (vals.empty()) return s;
    double sum = 0.0;
    double max_v = -1e30;
    for (double v : vals) {
      sum += v;
      if (v > max_v) max_v = v;
    }
    s.mean = sum / s.count;
    s.max_val = max_v;
    s.p50 = Percentile(vals, 0.50);
    s.p90 = Percentile(vals, 0.90);
    s.p99 = Percentile(vals, 0.99);
    if (include_p10) {
      s.p10 = Percentile(vals, 0.10);
    }
    return s;
  }

  json::Object ToJson(bool include_p10 = false) const {
    json::Object obj;
    obj["count"] = static_cast<int64_t>(count);
    obj["mean"] = mean;
    if (include_p10) {
      obj["p10"] = p10;
    }
    obj["p50"] = p50;
    obj["p90"] = p90;
    obj["p99"] = p99;
    obj["max"] = max_val;
    return obj;
  }
};

struct RoleSaturationStats {
  int total_decisions = 0;
  std::vector<double> z1_vals;
  std::vector<double> z2_vals;
  std::vector<double> z3_vals;
  std::vector<double> zmin_vals;

  std::vector<double> d1_vals;
  std::vector<double> d2_vals;
  std::vector<double> d3_vals;

  std::vector<double> gap_z1_z2_vals;
  std::vector<double> gap_z1_z3_vals;
  std::vector<double> gap_c1_c2_vals;
  std::vector<double> gap_c1_c3_vals;

  std::vector<double> pre_max_abs_vals;

  int count_d1_lt_005 = 0;
  int count_d1_lt_020 = 0;
  int count_d2_lt_005 = 0;
  int count_d2_lt_020 = 0;
  int count_pre_max_abs_ge_229 = 0;
  int count_false_alarm = 0;

  void Accumulate(const PolicyEvalResult& eval) {
    if (eval.z.size() < 2) return;
    total_decisions++;

    std::vector<double> sorted_z;
    sorted_z.reserve(eval.z.size());
    for (float val : eval.z) sorted_z.push_back(static_cast<double>(val));
    std::sort(sorted_z.begin(), sorted_z.end(), std::greater<double>());

    const double z1 = sorted_z[0];
    const double z2 = sorted_z[1];
    const double zmin = sorted_z.back();

    z1_vals.push_back(z1);
    z2_vals.push_back(z2);
    zmin_vals.push_back(zmin);

    const double d1 = CalcSech2(z1);
    const double d2 = CalcSech2(z2);
    d1_vals.push_back(d1);
    d2_vals.push_back(d2);

    if (d1 < 0.05) count_d1_lt_005++;
    if (d1 < 0.20) count_d1_lt_020++;
    if (d2 < 0.05) count_d2_lt_005++;
    if (d2 < 0.20) count_d2_lt_020++;

    gap_z1_z2_vals.push_back(z1 - z2);
    const double c1 = 10.0 * std::tanh(z1 / 10.0);
    const double c2 = 10.0 * std::tanh(z2 / 10.0);
    gap_c1_c2_vals.push_back(c1 - c2);

    if (sorted_z.size() >= 3) {
      const double z3 = sorted_z[2];
      const double d3 = CalcSech2(z3);
      const double c3 = 10.0 * std::tanh(z3 / 10.0);
      z3_vals.push_back(z3);
      d3_vals.push_back(d3);
      gap_z1_z3_vals.push_back(z1 - z3);
      gap_c1_c3_vals.push_back(c1 - c3);
    }

    const double pre_max = static_cast<double>(eval.pre_max_abs);
    pre_max_abs_vals.push_back(pre_max);
    if (pre_max >= 22.9) {
      count_pre_max_abs_ge_229++;
      if (std::abs(z1) < 15.0 && std::abs(z2) < 15.0) {
        count_false_alarm++;
      }
    }
  }

  json::Object ToJson() const {
    json::Object obj;
    obj["total_decisions"] = static_cast<int64_t>(total_decisions);

    obj["z1"] = NumericSummary::FromVector(z1_vals).ToJson();
    obj["z2"] = NumericSummary::FromVector(z2_vals).ToJson();
    obj["z3"] = NumericSummary::FromVector(z3_vals).ToJson();
    obj["z_min"] = NumericSummary::FromVector(zmin_vals).ToJson();

    obj["d1_sech2"] = NumericSummary::FromVector(d1_vals, /*include_p10=*/true).ToJson(true);
    obj["d2_sech2"] = NumericSummary::FromVector(d2_vals, /*include_p10=*/true).ToJson(true);
    obj["d3_sech2"] = NumericSummary::FromVector(d3_vals, /*include_p10=*/true).ToJson(true);

    obj["gap_z1_z2"] = NumericSummary::FromVector(gap_z1_z2_vals).ToJson();
    obj["gap_z1_z3"] = NumericSummary::FromVector(gap_z1_z3_vals).ToJson();
    obj["gap_c1_c2"] = NumericSummary::FromVector(gap_c1_c2_vals).ToJson();
    obj["gap_c1_c3"] = NumericSummary::FromVector(gap_c1_c3_vals).ToJson();

    obj["pre_max_abs"] = NumericSummary::FromVector(pre_max_abs_vals).ToJson();

    const double tot = (total_decisions > 0) ? static_cast<double>(total_decisions) : 1.0;
    obj["frac_d1_lt_005"] = static_cast<double>(count_d1_lt_005) / tot;
    obj["frac_d1_lt_020"] = static_cast<double>(count_d1_lt_020) / tot;
    obj["frac_d2_lt_005"] = static_cast<double>(count_d2_lt_005) / tot;
    obj["frac_d2_lt_020"] = static_cast<double>(count_d2_lt_020) / tot;
    obj["frac_pre_max_abs_ge_229"] = static_cast<double>(count_pre_max_abs_ge_229) / tot;
    obj["false_alarm_rate_total"] = static_cast<double>(count_false_alarm) / tot;
    obj["false_alarm_rate_conditional"] = (count_pre_max_abs_ge_229 > 0)
        ? static_cast<double>(count_false_alarm) / static_cast<double>(count_pre_max_abs_ge_229)
        : 0.0;
    return obj;
  }
};

struct MarketSymmetryStats {
  int natural_roots = 0;
  int augmented_roots = 0;
  int total_roots = 0;
  int total_permutations_tested = 0;

  std::vector<double> kl_full_vals;
  std::vector<double> kl_cond_vals;
  std::vector<double> max_abs_diff_vals;

  int overall_top1_flips = 0;
  int best_family_buy_flips = 0;
  int total_family_evals = 0;

  // Tleilaxu 2-slot average policy stats
  std::vector<double> entropy_orig_vals;
  std::vector<double> entropy_avg_vals;
  std::vector<double> buy_mass_orig_vals;
  std::vector<double> buy_mass_avg_vals;
  int slot0_argmax_flips = 0;
  int within_tleilaxu_rank_flips = 0;
  int total_tleilaxu_pairs = 0;

  std::string note = "";

  void AccumulatePermutation(
      const std::vector<Action>& orig_legal,
      const std::vector<float>& p_orig,
      const std::vector<float>& p_perm_aligned,
      const std::vector<Action>& family_actions) {
    total_permutations_tested++;

    // Full distribution KL(p_orig || p_perm_aligned)
    double kl_full = 0.0;
    for (size_t i = 0; i < orig_legal.size(); ++i) {
      if (p_orig[i] > 1e-12f) {
        const double q = std::max(static_cast<double>(p_perm_aligned[i]), 1e-12);
        kl_full += static_cast<double>(p_orig[i]) * std::log(static_cast<double>(p_orig[i]) / q);
      }
    }
    kl_full_vals.push_back(std::max(0.0, kl_full));

    // Overall greedy top-1 action flip
    size_t argmax_orig = 0;
    float max_p_orig = p_orig[0];
    size_t argmax_perm = 0;
    float max_p_perm = p_perm_aligned[0];
    for (size_t i = 1; i < orig_legal.size(); ++i) {
      if (p_orig[i] > max_p_orig) {
        max_p_orig = p_orig[i];
        argmax_orig = i;
      }
      if (p_perm_aligned[i] > max_p_perm) {
        max_p_perm = p_perm_aligned[i];
        argmax_perm = i;
      }
    }
    if (orig_legal[argmax_orig] != orig_legal[argmax_perm]) {
      overall_top1_flips++;
    }

    // Family-conditional KL and max-abs-prob-diff
    double mass_orig = 0.0;
    double mass_perm = 0.0;
    double max_diff = 0.0;
    std::unordered_map<Action, size_t> act_to_idx;
    for (size_t i = 0; i < orig_legal.size(); ++i) {
      act_to_idx[orig_legal[i]] = i;
    }

    Action best_fam_orig = -1;
    float best_fam_p_orig = -1.0f;
    Action best_fam_perm = -1;
    float best_fam_p_perm = -1.0f;

    for (Action a : family_actions) {
      auto it = act_to_idx.find(a);
      if (it == act_to_idx.end()) continue;
      const size_t idx = it->second;
      const float po = p_orig[idx];
      const float pp = p_perm_aligned[idx];
      mass_orig += po;
      mass_perm += pp;
      const double d = std::abs(static_cast<double>(po) - static_cast<double>(pp));
      if (d > max_diff) max_diff = d;

      if (po > best_fam_p_orig) {
        best_fam_p_orig = po;
        best_fam_orig = a;
      }
      if (pp > best_fam_p_perm) {
        best_fam_p_perm = pp;
        best_fam_perm = a;
      }
    }

    max_abs_diff_vals.push_back(max_diff);

    if (mass_orig > 1e-6 && mass_perm > 1e-6) {
      double kl_cond = 0.0;
      for (Action a : family_actions) {
        auto it = act_to_idx.find(a);
        if (it == act_to_idx.end()) continue;
        const size_t idx = it->second;
        const double po_norm = static_cast<double>(p_orig[idx]) / mass_orig;
        const double pp_norm = static_cast<double>(p_perm_aligned[idx]) / mass_perm;
        if (po_norm > 1e-12) {
          const double q = std::max(pp_norm, 1e-12);
          kl_cond += po_norm * std::log(po_norm / q);
        }
      }
      kl_cond_vals.push_back(std::max(0.0, kl_cond));
    }

    if (best_fam_orig != -1 && best_fam_perm != -1) {
      total_family_evals++;
      if (best_fam_orig != best_fam_perm) {
        best_family_buy_flips++;
      }
    }
  }

  void AccumulateTleilaxuAveragedPolicy(
      const std::vector<Action>& orig_legal,
      const std::vector<float>& p_orig,
      const std::vector<float>& p_swap_aligned) {
    total_tleilaxu_pairs++;

    std::vector<float> p_avg(orig_legal.size());
    double ent_orig = 0.0;
    double ent_avg = 0.0;
    for (size_t i = 0; i < orig_legal.size(); ++i) {
      const float avg_val = 0.5f * (p_orig[i] + p_swap_aligned[i]);
      p_avg[i] = avg_val;
      if (p_orig[i] > 1e-12f) {
        ent_orig -= static_cast<double>(p_orig[i]) * std::log(static_cast<double>(p_orig[i]));
      }
      if (avg_val > 1e-12f) {
        ent_avg -= static_cast<double>(avg_val) * std::log(static_cast<double>(avg_val));
      }
    }
    entropy_orig_vals.push_back(ent_orig);
    entropy_avg_vals.push_back(ent_avg);

    size_t idx91 = orig_legal.size();
    size_t idx92 = orig_legal.size();
    size_t argmax_orig = 0;
    size_t argmax_avg = 0;
    for (size_t i = 0; i < orig_legal.size(); ++i) {
      if (orig_legal[i] == 91) idx91 = i;
      if (orig_legal[i] == 92) idx92 = i;
      if (p_orig[i] > p_orig[argmax_orig]) argmax_orig = i;
      if (p_avg[i] > p_avg[argmax_avg]) argmax_avg = i;
    }

    if (argmax_orig != argmax_avg) {
      slot0_argmax_flips++;
    }

    if (idx91 < orig_legal.size() && idx92 < orig_legal.size()) {
      buy_mass_orig_vals.push_back(p_orig[idx91] + p_orig[idx92]);
      buy_mass_avg_vals.push_back(p_avg[idx91] + p_avg[idx92]);

      const bool orig_prefers_91 = (p_orig[idx91] > p_orig[idx92]);
      const bool avg_prefers_91 = (p_avg[idx91] > p_avg[idx92]);
      if (orig_prefers_91 != avg_prefers_91) {
        within_tleilaxu_rank_flips++;
      }
    }
  }

  json::Object ToJson() const {
    json::Object obj;
    obj["natural_roots"] = static_cast<int64_t>(natural_roots);
    obj["augmented_roots"] = static_cast<int64_t>(augmented_roots);
    obj["total_roots"] = static_cast<int64_t>(total_roots);
    obj["total_permutations_tested"] = static_cast<int64_t>(total_permutations_tested);

    obj["kl_full"] = NumericSummary::FromVector(kl_full_vals).ToJson();
    obj["kl_conditional"] = NumericSummary::FromVector(kl_cond_vals).ToJson();
    obj["max_abs_prob_diff"] = NumericSummary::FromVector(max_abs_diff_vals).ToJson();

    const double tot_perm = (total_permutations_tested > 0)
        ? static_cast<double>(total_permutations_tested) : 1.0;
    obj["overall_top1_flip_rate"] = static_cast<double>(overall_top1_flips) / tot_perm;

    const double tot_fam = (total_family_evals > 0)
        ? static_cast<double>(total_family_evals) : 1.0;
    obj["best_family_buy_flip_rate"] = static_cast<double>(best_family_buy_flips) / tot_fam;

    if (total_tleilaxu_pairs > 0) {
      json::Object tleilaxu_avg_obj;
      tleilaxu_avg_obj["entropy_orig"] = NumericSummary::FromVector(entropy_orig_vals).ToJson();
      tleilaxu_avg_obj["entropy_avg"] = NumericSummary::FromVector(entropy_avg_vals).ToJson();
      tleilaxu_avg_obj["buy_mass_orig"] = NumericSummary::FromVector(buy_mass_orig_vals).ToJson();
      tleilaxu_avg_obj["buy_mass_avg"] = NumericSummary::FromVector(buy_mass_avg_vals).ToJson();
      tleilaxu_avg_obj["slot0_argmax_flip_rate"] =
          static_cast<double>(slot0_argmax_flips) / static_cast<double>(total_tleilaxu_pairs);
      tleilaxu_avg_obj["within_tleilaxu_rank_flip_rate"] =
          static_cast<double>(within_tleilaxu_rank_flips) / static_cast<double>(total_tleilaxu_pairs);
      obj["tleilaxu_slot_averaged_policy"] = tleilaxu_avg_obj;
    }

    if (!note.empty()) {
      obj["note"] = note;
    }
    return obj;
  }
};

}  // namespace

int RunAudit(int argc, char* argv[]) {
  absl::ParseCommandLine(argc, argv);

  const std::string model_path = absl::GetFlag(FLAGS_model_path);
  const int num_games = absl::GetFlag(FLAGS_num_games);
  const int threads = absl::GetFlag(FLAGS_threads);
  const int seed = absl::GetFlag(FLAGS_seed);
  const std::string output_json_path = absl::GetFlag(FLAGS_output_json_path);

  std::cout << "========================================================\n";
  std::cout << "Dune: Imperium - Ord 1a CPU Diagnostics & Symmetry Audit\n";
  std::cout << "========================================================\n";
  std::cout << "Model path:        " << model_path << "\n";
  std::cout << "Self-play games:   " << num_games << "\n";
  std::cout << "CPU Threads:       " << threads << "\n";
  std::cout << "Random seed:       " << seed << "\n";
  std::cout << "Output JSON path:  " << output_json_path << "\n";
  std::cout << "--------------------------------------------------------\n";

  at::set_num_threads(threads);
  torch::Device device(torch::kCPU);

  const int input_dim = 9941;
  const int hidden_dim = 2048;
  const int action_dim = 2391;
  const int num_blocks = 16;

  std::cout << "Loading policy network on CPU...\n";
  auto model = std::make_shared<SharedDunePolicyValueNetImpl>(
      input_dim, hidden_dim, action_dim, num_blocks,
      /*use_nonlinear=*/false, /*with_aux_heads=*/false,
      /*head_init_seed=*/0, /*with_semantic_scorer=*/true,
      /*enable_zone_encoder=*/false,
      /*semantic_descriptor_schema=*/dune_semantic::kDescriptorSchemaVersionV3);
  model->market_appendix_mode_ = MarketAppendixMode::kFullPublicInformationV4;
  model->deck_pool_semantics_ = DeckPoolSemantics::kFixed;

  LoadModelCheckpointRobust(model, model_path, device);
  model->to(device);
  model->eval();
  std::cout << "Model successfully loaded and set to eval mode.\n";

  std::shared_ptr<const Game> game = LoadGame("dune_imperium(enable_immortality=true)");
  std::mt19937_64 rng(static_cast<uint64_t>(seed));

  RoleSaturationStats overall_saturation;
  std::map<std::string, RoleSaturationStats> role_saturation;

  MarketSymmetryStats imperium_symmetry;
  MarketSymmetryStats tleilaxu_natural_symmetry;
  MarketSymmetryStats tleilaxu_controlled_symmetry;
  MarketSymmetryStats tech_symmetry;
  MarketSymmetryStats helena_symmetry;

  std::vector<std::unique_ptr<State>> tech_augmentation_candidates;
  std::vector<int> completed_game_rounds;

  const std::vector<std::vector<int>> kImperiumPerms = {
      {1, 2, 3, 4, 0},
      {2, 3, 4, 0, 1},
      {3, 4, 0, 1, 2},
      {4, 3, 2, 1, 0}
  };

  const std::vector<std::vector<int>> kTechPerms = {
      {1, 2, 0},
      {2, 0, 1},
      {0, 2, 1},
      {2, 1, 0},
      {1, 0, 2}
  };

  std::cout << "\nStarting " << num_games << " natural self-play games for diagnostics...\n";
  auto start_time = std::chrono::steady_clock::now();

  for (int g = 0; g < num_games; ++g) {
    std::unique_ptr<State> st = game->NewInitialState();
    bool in_agent_turn[4] = {false, false, false, false};

    while (!st->IsTerminal()) {
      if (st->IsChanceNode()) {
        const auto outcomes = st->ChanceOutcomes();
        std::vector<double> weights;
        weights.reserve(outcomes.size());
        for (const auto& o : outcomes) weights.push_back(o.second);
        std::discrete_distribution<size_t> ch_dist(weights.begin(), weights.end());
        st->ApplyAction(outcomes[ch_dist(rng)].first);
        continue;
      }

      auto legal = st->LegalActions();
      if (legal.empty()) break;

      auto* dune_st = dynamic_cast<DuneImperiumState*>(st.get());
      const Player cur_p = dune_st->CurrentPlayer();

      if (dune_st->phase() != GamePhase::kAgentTurns) {
        for (int p = 0; p < 4; ++p) in_agent_turn[p] = false;
      }

      if (legal.size() == 1) {
        const Action a = legal[0];
        if (dune_st->phase() == GamePhase::kAgentTurns) {
          if ((a >= kActionSelectAgentCard0 && a < kActionSelectAgentCard0 + 256) ||
              a == kActionPlayKwisatzHaderach) {
            in_agent_turn[cur_p] = true;
          } else if (a == kActionEndTurn) {
            in_agent_turn[cur_p] = false;
          }
        }
        st->ApplyAction(a);
        continue;
      }

      // Non-trivial decision point (legal.size() >= 2)
      const DuneDecisionRole role =
          ClassifyDuneDecisionRole(*st, cur_p, in_agent_turn[cur_p]);
      const std::string role_str = RoleToString(role);

      PolicyEvalResult eval =
          EvaluatePolicyAtState(model, *dune_st, cur_p, legal, device);

      overall_saturation.Accumulate(eval);
      role_saturation[role_str].Accumulate(eval);

      // --- Part B: Check for natural market roots ---

      // 1. Imperium Row Purchase Roots
      if (role == DuneDecisionRole::kPurchase ||
          dune_st->phase() == GamePhase::kRevealTurns) {
        const auto imp_row = dune_st->GetImperiumRowForTesting();
        std::set<int> distinct_row(imp_row.begin(), imp_row.end());
        distinct_row.erase(kInvalidCard);

        std::vector<Action> legal_imp_buys;
        for (Action a : legal) {
          if (a >= kActionBuyImperiumRow0 && a <= kActionBuyImperiumRow0 + 4) {
            legal_imp_buys.push_back(a);
          }
        }

        if (distinct_row.size() == 5 && legal_imp_buys.size() >= 2) {
          imperium_symmetry.natural_roots++;
          imperium_symmetry.total_roots++;

          for (const auto& perm : kImperiumPerms) {
            auto perm_st = dune_st->Clone();
            auto* perm_dune = dynamic_cast<DuneImperiumState*>(perm_st.get());

            std::vector<int> permuted_row(5);
            for (int s = 0; s < 5; ++s) {
              permuted_row[s] = imp_row[perm[s]];
            }
            perm_dune->SetImperiumRowForTesting(permuted_row);

            const auto perm_legal = perm_dune->LegalActions();
            PolicyEvalResult perm_eval =
                EvaluatePolicyAtState(model, *perm_dune, cur_p, perm_legal, device);

            // Align perm_eval back to original action space
            // permuted_row[new_s] = imp_row[perm[new_s]]
            // Buying new_s (action 10 + new_s) in perm buys card perm[new_s]
            // which was action (10 + perm[new_s]) in orig state.
            std::vector<float> p_perm_aligned(legal.size(), 0.0f);
            for (size_t pi = 0; pi < perm_legal.size(); ++pi) {
              Action pa = perm_legal[pi];
              Action aligned_a = pa;
              if (pa >= kActionBuyImperiumRow0 && pa <= kActionBuyImperiumRow0 + 4) {
                int new_s = pa - kActionBuyImperiumRow0;
                int old_s = perm[new_s];
                aligned_a = kActionBuyImperiumRow0 + old_s;
              }
              for (size_t oi = 0; oi < legal.size(); ++oi) {
                if (legal[oi] == aligned_a) {
                  p_perm_aligned[oi] = perm_eval.probs[pi];
                  break;
                }
              }
            }

            imperium_symmetry.AccumulatePermutation(
                legal, eval.probs, p_perm_aligned, legal_imp_buys);
          }
        }
      }

      // 2. Tleilaxu Row Purchase Roots
      const auto t_row = dune_st->GetTleilaxuRow();
      if (t_row.size() == 2 && t_row[0] != t_row[1] &&
          t_row[0] != kInvalidAction && t_row[1] != kInvalidAction) {
        const bool has_91 = HasAction(legal, 91);
        const bool has_92 = HasAction(legal, 92);

        // (i) Natural Purchase roots where both 91 and 92 are legal
        if (has_91 && has_92) {
          tleilaxu_natural_symmetry.natural_roots++;
          tleilaxu_natural_symmetry.total_roots++;

          auto swap_st = dune_st->Clone();
          auto* swap_dune = dynamic_cast<DuneImperiumState*>(swap_st.get());
          swap_dune->SetTleilaxuRowForTesting({t_row[1], t_row[0]});

          const auto swap_legal = swap_dune->LegalActions();
          PolicyEvalResult swap_eval =
              EvaluatePolicyAtState(model, *swap_dune, cur_p, swap_legal, device);

          std::vector<float> p_swap_aligned(legal.size(), 0.0f);
          for (size_t si = 0; si < swap_legal.size(); ++si) {
            Action sa = swap_legal[si];
            Action aligned_a = sa;
            if (sa == 91) aligned_a = 92;
            else if (sa == 92) aligned_a = 91;

            for (size_t oi = 0; oi < legal.size(); ++oi) {
              if (legal[oi] == aligned_a) {
                p_swap_aligned[oi] = swap_eval.probs[si];
                break;
              }
            }
          }

          std::vector<Action> t_actions = {91, 92};
          tleilaxu_natural_symmetry.AccumulatePermutation(
              legal, eval.probs, p_swap_aligned, t_actions);
          tleilaxu_natural_symmetry.AccumulateTleilaxuAveragedPolicy(
              legal, eval.probs, p_swap_aligned);
        }

        // (ii) Controlled specimens >= 3 slice of Reveal Purchase roots
        if (role == DuneDecisionRole::kPurchase ||
            dune_st->phase() == GamePhase::kRevealTurns) {
          auto ctrl_st = dune_st->Clone();
          auto* ctrl_dune = dynamic_cast<DuneImperiumState*>(ctrl_st.get());
          ctrl_dune->SetSpecimensForTesting(cur_p, 5);

          const auto ctrl_legal = ctrl_dune->LegalActions();
          if (HasAction(ctrl_legal, 91) && HasAction(ctrl_legal, 92)) {
            tleilaxu_controlled_symmetry.augmented_roots++;
            tleilaxu_controlled_symmetry.total_roots++;

            PolicyEvalResult ctrl_eval =
                EvaluatePolicyAtState(model, *ctrl_dune, cur_p, ctrl_legal, device);

            auto swap_ctrl_st = ctrl_dune->Clone();
            auto* swap_ctrl_dune = dynamic_cast<DuneImperiumState*>(swap_ctrl_st.get());
            swap_ctrl_dune->SetTleilaxuRowForTesting({t_row[1], t_row[0]});

            const auto swap_ctrl_legal = swap_ctrl_dune->LegalActions();
            PolicyEvalResult swap_ctrl_eval =
                EvaluatePolicyAtState(model, *swap_ctrl_dune, cur_p, swap_ctrl_legal, device);

            std::vector<float> p_ctrl_swap_aligned(ctrl_legal.size(), 0.0f);
            for (size_t si = 0; si < swap_ctrl_legal.size(); ++si) {
              Action sa = swap_ctrl_legal[si];
              Action aligned_a = sa;
              if (sa == 91) aligned_a = 92;
              else if (sa == 92) aligned_a = 91;

              for (size_t oi = 0; oi < ctrl_legal.size(); ++oi) {
                if (ctrl_legal[oi] == aligned_a) {
                  p_ctrl_swap_aligned[oi] = swap_ctrl_eval.probs[si];
                  break;
                }
              }
            }

            std::vector<Action> t_actions = {91, 92};
            tleilaxu_controlled_symmetry.AccumulatePermutation(
                ctrl_legal, ctrl_eval.probs, p_ctrl_swap_aligned, t_actions);
            tleilaxu_controlled_symmetry.AccumulateTleilaxuAveragedPolicy(
                ctrl_legal, ctrl_eval.probs, p_ctrl_swap_aligned);
          }
        }
      }

      // 3. Tech Market Roots
      const auto& tech_market = dune_st->GetTechMarket();
      std::set<int> valid_tech_tiles;
      for (int t : tech_market) {
        if (t != kInvalidAction && t >= 0 && t < static_cast<int>(kTechTiles.size())) {
          valid_tech_tiles.insert(t);
        }
      }
      std::vector<Action> legal_tech_acts;
      for (Action a : legal) {
        if ((a >= kActionTechAcquire0 && a <= kActionTechAcquire0 + 2) ||
            (a >= kActionTechAcquireWithSolari0 && a <= kActionTechAcquireWithSolari0 + 2)) {
          legal_tech_acts.push_back(a);
        }
      }

      if (valid_tech_tiles.size() >= 2 && legal_tech_acts.size() >= 2) {
        tech_symmetry.natural_roots++;
        tech_symmetry.total_roots++;

        const std::vector<int> orig_depth = {
            dune_st->GetTechStackSizeForTesting(0),
            dune_st->GetTechStackSizeForTesting(1),
            dune_st->GetTechStackSizeForTesting(2)
        };

        for (const auto& perm : kTechPerms) {
          auto perm_st = dune_st->Clone();
          auto* perm_dune = dynamic_cast<DuneImperiumState*>(perm_st.get());

          for (int new_s = 0; new_s < 3; ++new_s) {
            const int old_s = perm[new_s];
            perm_dune->SetTechMarketForTesting(new_s, tech_market[old_s]);
            perm_dune->SetTechStackForTesting(
                new_s, std::vector<int>(orig_depth[old_s], tech_market[old_s]));
          }

          const auto perm_legal = perm_dune->LegalActions();
          PolicyEvalResult perm_eval =
              EvaluatePolicyAtState(model, *perm_dune, cur_p, perm_legal, device);

          std::vector<float> p_perm_aligned(legal.size(), 0.0f);
          for (size_t pi = 0; pi < perm_legal.size(); ++pi) {
            Action pa = perm_legal[pi];
            Action aligned_a = pa;
            if (pa >= kActionTechAcquire0 && pa <= kActionTechAcquire0 + 2) {
              int new_s = pa - kActionTechAcquire0;
              int old_s = perm[new_s];
              aligned_a = kActionTechAcquire0 + old_s;
            } else if (pa >= kActionTechAcquireWithSolari0 &&
                       pa <= kActionTechAcquireWithSolari0 + 2) {
              int new_s = pa - kActionTechAcquireWithSolari0;
              int old_s = perm[new_s];
              aligned_a = kActionTechAcquireWithSolari0 + old_s;
            }
            for (size_t oi = 0; oi < legal.size(); ++oi) {
              if (legal[oi] == aligned_a) {
                p_perm_aligned[oi] = perm_eval.probs[pi];
                break;
              }
            }
          }

          tech_symmetry.AccumulatePermutation(
              legal, eval.probs, p_perm_aligned, legal_tech_acts);
        }
      } else if (tech_augmentation_candidates.size() < 100 &&
                 (role == DuneDecisionRole::kAgentPrimary ||
                  role == DuneDecisionRole::kPurchase)) {
        tech_augmentation_candidates.push_back(st->Clone());
      }

      // Sample next action and progress natural game
      std::discrete_distribution<size_t> act_dist(eval.probs.begin(), eval.probs.end());
      const Action chosen_action = legal[act_dist(rng)];

      if (dune_st->phase() == GamePhase::kAgentTurns) {
        if ((chosen_action >= kActionSelectAgentCard0 &&
             chosen_action < kActionSelectAgentCard0 + 256) ||
            chosen_action == kActionPlayKwisatzHaderach) {
          in_agent_turn[cur_p] = true;
        } else if (chosen_action == kActionEndTurn) {
          in_agent_turn[cur_p] = false;
        }
      }

      st->ApplyAction(chosen_action);
    }

    const auto* dune_fin = dynamic_cast<const DuneImperiumState*>(st.get());
    completed_game_rounds.push_back(std::max(1, dune_fin->GetCurrentRound() - 1));

    if ((g + 1) % 10 == 0 || g == num_games - 1) {
      std::cout << "  Completed game " << (g + 1) << " / " << num_games << "\n";
    }
  }

  // --- Augmentation for Tech if natural roots < 50 ---
  if (tech_symmetry.natural_roots < 50 && !tech_augmentation_candidates.empty()) {
    std::cout << "\nNatural Tech purchase roots (" << tech_symmetry.natural_roots
              << ") < 50; synthesizing augmented tech roots from natural states...\n";
    for (size_t i = 0;
         i < tech_augmentation_candidates.size() && tech_symmetry.total_roots < 60;
         ++i) {
      auto aug_st = tech_augmentation_candidates[i]->Clone();
      auto* aug_dune = dynamic_cast<DuneImperiumState*>(aug_st.get());
      const Player p = aug_dune->CurrentPlayer();

      aug_dune->SetPlayerSpiceForTesting(p, 6);
      aug_dune->SetPendingShippingNeedsTechChoiceForTesting(true);

      // Place 3 distinct tiles in slots 0, 1, 2
      aug_dune->SetTechMarketForTesting(0, 0);
      aug_dune->SetTechStackForTesting(0, {0, 0});
      aug_dune->SetTechMarketForTesting(1, 1);
      aug_dune->SetTechStackForTesting(1, {1, 1});
      aug_dune->SetTechMarketForTesting(2, 2);
      aug_dune->SetTechStackForTesting(2, {2, 2});

      const auto aug_legal = aug_dune->LegalActions();
      std::vector<Action> tech_buys = {70, 71, 72};
      if (HasAction(aug_legal, 70) && HasAction(aug_legal, 71) && HasAction(aug_legal, 72)) {
        tech_symmetry.augmented_roots++;
        tech_symmetry.total_roots++;

        PolicyEvalResult aug_eval =
            EvaluatePolicyAtState(model, *aug_dune, p, aug_legal, device);

        const std::vector<int> depth = {2, 2, 2};
        const std::vector<int> market = {0, 1, 2};

        for (const auto& perm : kTechPerms) {
          auto perm_st = aug_dune->Clone();
          auto* perm_dune = dynamic_cast<DuneImperiumState*>(perm_st.get());

          for (int new_s = 0; new_s < 3; ++new_s) {
            const int old_s = perm[new_s];
            perm_dune->SetTechMarketForTesting(new_s, market[old_s]);
            perm_dune->SetTechStackForTesting(
                new_s, std::vector<int>(depth[old_s], market[old_s]));
          }

          const auto perm_legal = perm_dune->LegalActions();
          PolicyEvalResult perm_eval =
              EvaluatePolicyAtState(model, *perm_dune, p, perm_legal, device);

          std::vector<float> p_perm_aligned(aug_legal.size(), 0.0f);
          for (size_t pi = 0; pi < perm_legal.size(); ++pi) {
            Action pa = perm_legal[pi];
            Action aligned_a = pa;
            if (pa >= kActionTechAcquire0 && pa <= kActionTechAcquire0 + 2) {
              int new_s = pa - kActionTechAcquire0;
              int old_s = perm[new_s];
              aligned_a = kActionTechAcquire0 + old_s;
            }
            for (size_t oi = 0; oi < aug_legal.size(); ++oi) {
              if (aug_legal[oi] == aligned_a) {
                p_perm_aligned[oi] = perm_eval.probs[pi];
                break;
              }
            }
          }

          tech_symmetry.AccumulatePermutation(
              aug_legal, aug_eval.probs, p_perm_aligned, tech_buys);
        }
      }
    }
  }

  // Helena Reserve note
  helena_symmetry.natural_roots = 0;
  helena_symmetry.augmented_roots = 0;
  helena_symmetry.total_roots = 0;
  helena_symmetry.note =
      "Engine rule enforces kMaxHelenaReserveCardsPerRound = 1 (single reserved card buyable per round; slot 1..7 actions do not exist in LegalActions). Simultaneous multi-card Helena market permutations are physically non-existent under the official rule set.";

  auto end_time = std::chrono::steady_clock::now();
  const double elapsed_sec =
      std::chrono::duration<double>(end_time - start_time).count();

  // Compute overall round statistics across completed games
  double round_sum = 0.0;
  std::map<int, int> round_histogram;
  for (int r : completed_game_rounds) {
    round_sum += r;
    round_histogram[r]++;
  }
  const double avg_round_finished = completed_game_rounds.empty()
      ? 0.0 : round_sum / completed_game_rounds.size();

  std::cout << "\n--------------------------------------------------------\n";
  std::cout << "Audit Execution Summary:\n";
  std::cout << "  Wall time elapsed:           " << std::fixed << std::setprecision(2)
            << elapsed_sec << " seconds\n";
  std::cout << "  Games finished:              " << completed_game_rounds.size() << "\n";
  std::cout << "  Overall avg round finished:  " << std::setprecision(2)
            << avg_round_finished << "\n";
  std::cout << "  Total decisions audited:     " << overall_saturation.total_decisions << "\n";
  std::cout << "  Imperium symmetry roots:     " << imperium_symmetry.total_roots
            << " (natural: " << imperium_symmetry.natural_roots << ")\n";
  std::cout << "  Tleilaxu natural roots:      " << tleilaxu_natural_symmetry.total_roots << "\n";
  std::cout << "  Tleilaxu controlled roots:   " << tleilaxu_controlled_symmetry.total_roots << "\n";
  std::cout << "  Tech stack symmetry roots:   " << tech_symmetry.total_roots
            << " (natural: " << tech_symmetry.natural_roots
            << ", augmented: " << tech_symmetry.augmented_roots << ")\n";

  // Construct JSON report
  json::Object root;
  root["benchmark_name"] = "dune_ord1a_cpu_audit";
  root["model_path"] = model_path;
  root["num_games"] = static_cast<int64_t>(num_games);
  root["threads"] = static_cast<int64_t>(threads);
  root["seed"] = static_cast<int64_t>(seed);
  root["elapsed_wall_seconds"] = elapsed_sec;

  json::Object game_stats;
  game_stats["completed_games_count"] = static_cast<int64_t>(completed_game_rounds.size());
  game_stats["avg_round_finished"] = avg_round_finished;
  json::Object hist_obj;
  for (const auto& [r, count] : round_histogram) {
    hist_obj[std::to_string(r)] = static_cast<int64_t>(count);
  }
  game_stats["round_histogram"] = hist_obj;
  root["game_stats"] = game_stats;

  json::Object part_a;
  part_a["overall"] = overall_saturation.ToJson();
  json::Object per_role;
  for (const auto& [rname, rstat] : role_saturation) {
    per_role[rname] = rstat.ToJson();
  }
  part_a["per_role"] = per_role;
  root["part_a_contender_saturation"] = part_a;

  json::Object part_b;
  part_b["imperium_row"] = imperium_symmetry.ToJson();
  part_b["tleilaxu_row_natural"] = tleilaxu_natural_symmetry.ToJson();
  part_b["tleilaxu_row_controlled"] = tleilaxu_controlled_symmetry.ToJson();
  part_b["tech_stacks"] = tech_symmetry.ToJson();
  part_b["helena_reserve"] = helena_symmetry.ToJson();
  root["part_b_market_slot_swap_symmetry"] = part_b;

  std::filesystem::create_directories(
      std::filesystem::path(output_json_path).parent_path());
  std::ofstream out_file(output_json_path);
  if (out_file.is_open()) {
    out_file << json::ToString(root, /*pretty=*/true) << "\n";
    std::cout << "Successfully exported JSON audit report to: "
              << output_json_path << "\n";
  } else {
    std::cerr << "WARNING: Could not open " << output_json_path << " for writing!\n";
  }

  std::cout << "========================================================\n";
  return 0;
}

}  // namespace dune_imperium
}  // namespace open_spiel

int main(int argc, char* argv[]) {
  return open_spiel::dune_imperium::RunAudit(argc, argv);
}
