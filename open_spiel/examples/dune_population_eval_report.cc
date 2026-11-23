// Statistics aggregation, telemetry serialization, and console/JSON reporting
// for dune_population_eval.

#include "dune_population_eval_internal.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <mutex>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#include "open_spiel/abseil-cpp/absl/flags/flag.h"
#include "open_spiel/abseil-cpp/absl/strings/str_format.h"

namespace open_spiel {
namespace dune_pop_eval {

// ---------------------------------------------------------------------------
// Wilson score interval
// ---------------------------------------------------------------------------
WilsonCI WilsonScore(int successes, int trials, double z) {
  if (trials == 0) return {0.0, 0.0, 0.0};
  double n = static_cast<double>(trials);
  double p_hat = static_cast<double>(successes) / n;
  double z2 = z * z;
  double denom = 1.0 + z2 / n;
  double center = (p_hat + z2 / (2.0 * n)) / denom;
  double margin = z * std::sqrt((p_hat * (1.0 - p_hat) + z2 / (4.0 * n)) / n)
                  / denom;
  return {p_hat, std::max(0.0, center - margin),
                 std::min(1.0, center + margin)};
}

// ---------------------------------------------------------------------------
// Phase-3 telemetry JSON helpers
// ---------------------------------------------------------------------------
namespace {

std::string JsonVpArrayOrNull(bool has, const std::array<int, kNumPlayers>& v) {
  if (!has) return "null";
  std::string s = "[";
  for (int p = 0; p < kNumPlayers; ++p) {
    if (p) s += ",";
    s += std::to_string(v[p]);
  }
  s += "]";
  return s;
}

std::string JsonIntOrNull(int x) {
  return x < 0 ? std::string("null") : std::to_string(x);
}

std::string JsonBoolOrNull(int tri) {
  if (tri < 0) return "null";
  return tri ? "true" : "false";
}

// --- WO-1 Phase 3 emission helpers ---
// [[r1p0,r1p1,r1p2,r1p3],[r2...],...] for rounds 1..rounds_captured.
std::string JsonVpByRound(
    int rounds_captured,
    const std::array<std::array<int, kNumPlayers>, kMaxRounds>& v) {
  std::string s = "[";
  for (int rd = 0; rd < rounds_captured; ++rd) {
    if (rd) s += ",";
    s += "[";
    for (int p = 0; p < kNumPlayers; ++p) {
      if (p) s += ",";
      s += std::to_string(v[rd][p]);
    }
    s += "]";
  }
  return s + "]";
}

// Per-seat ints where a negative entry means "never" and emits JSON null.
std::string JsonIntArrayWithNulls(const std::array<int, kNumPlayers>& v) {
  std::string s = "[";
  for (int p = 0; p < kNumPlayers; ++p) {
    if (p) s += ",";
    s += v[p] < 0 ? std::string("null") : std::to_string(v[p]);
  }
  return s + "]";
}

std::string JsonIntArray(const std::array<int, kNumPlayers>& v) {
  std::string s = "[";
  for (int p = 0; p < kNumPlayers; ++p) {
    if (p) s += ",";
    s += std::to_string(v[p]);
  }
  return s + "]";
}

std::string JsonIntVector(const std::vector<int>& v) {
  std::string s = "[";
  for (size_t i = 0; i < v.size(); ++i) {
    if (i) s += ",";
    s += std::to_string(v[i]);
  }
  return s + "]";
}

std::string JsonActionVector(const std::vector<open_spiel::Action>& v) {
  std::string s = "[";
  for (size_t i = 0; i < v.size(); ++i) {
    if (i) s += ",";
    s += std::to_string(v[i]);
  }
  return s + "]";
}

std::string JsonBoolArray(const std::array<bool, kNumPlayers>& v) {
  std::string s = "[";
  for (int p = 0; p < kNumPlayers; ++p) {
    if (p) s += ",";
    s += (v[p] ? "true" : "false");
  }
  return s + "]";
}

std::string JsonNestedIntVector(const std::array<std::vector<int>, kNumPlayers>& v) {
  std::string s = "[";
  for (int p = 0; p < kNumPlayers; ++p) {
    if (p) s += ",";
    s += "[";
    for (size_t i = 0; i < v[p].size(); ++i) {
      if (i) s += ",";
      s += std::to_string(v[p][i]);
    }
    s += "]";
  }
  return s + "]";
}

std::string JsonResearchPaths(
    const std::array<std::vector<std::pair<int, int>>, kNumPlayers>& v) {
  std::string s = "[";
  for (int p = 0; p < kNumPlayers; ++p) {
    if (p) s += ",";
    s += "[";
    for (size_t i = 0; i < v[p].size(); ++i) {
      if (i) s += ",";
      s += "[" + std::to_string(v[p][i].first) + "," +
           std::to_string(v[p][i].second) + "]";
    }
    s += "]";
  }
  s += "]";
  return s;
}

std::mutex g_logit_dump_mutex;

double LogitStatsPercentile(std::vector<double> v, double q) {
  if (v.empty()) return 0.0;
  std::sort(v.begin(), v.end());
  if (v.size() == 1) return v[0];
  double idx = q / 100.0 * (static_cast<double>(v.size()) - 1.0);
  size_t lo = static_cast<size_t>(std::floor(idx));
  size_t hi = static_cast<size_t>(std::ceil(idx));
  double frac = idx - static_cast<double>(lo);
  return v[lo] * (1.0 - frac) + v[hi] * frac;
}

}  // namespace

// ---------------------------------------------------------------------------
// Phase-4.2: optional per-candidate-decision logit-stat dump (CSV)
// ---------------------------------------------------------------------------
const char* GamePhaseRole(dune_imperium::GamePhase ph) {
  switch (ph) {
    case dune_imperium::GamePhase::kLeaderOfferChance: return "leader_offer";
    case dune_imperium::GamePhase::kLeaderDraft:        return "leader_draft";
    case dune_imperium::GamePhase::kDeal:               return "deal";
    case dune_imperium::GamePhase::kRoundStart:         return "round_start";
    case dune_imperium::GamePhase::kAgentTurns:         return "agent";
    case dune_imperium::GamePhase::kRevealTurns:        return "reveal";
    case dune_imperium::GamePhase::kCombat:             return "combat";
    case dune_imperium::GamePhase::kMakers:             return "makers";
    case dune_imperium::GamePhase::kRecall:             return "recall";
    case dune_imperium::GamePhase::kTerminal:           return "terminal";
  }
  return "unknown";
}

// Appends one CSV row of PRE-cap legal-centered logit statistics for a single
// candidate decision. Center first (z = raw - legal_mean), then measure; caps
// are applied only for the capped-variant columns (soft tanh, matching
// CenterAndCapLegalLogits). Thread-safe (serialized on g_logit_dump_mutex).
void AppendLogitStatsRow(std::ofstream& out, int episode_id, int round,
                         const std::string& role,
                         const std::vector<Action>& legal,
                         const std::vector<float>& raw_logits,
                         double configured_cap) {
  const int n = static_cast<int>(legal.size());
  if (n == 0) return;
  double sum = 0.0;
  for (Action a : legal) sum += raw_logits[a];
  const double mean = sum / n;
  std::vector<double> z(n), absz(n), c10(n), c30(n);
  for (int i = 0; i < n; ++i) {
    z[i] = static_cast<double>(raw_logits[legal[i]]) - mean;
    absz[i] = std::fabs(z[i]);
    c10[i] = 10.0 * std::tanh(z[i] / 10.0);
    c30[i] = 30.0 * std::tanh(z[i] / 30.0);
  }
  auto log_softmax = [](const std::vector<double>& x) {
    double m = *std::max_element(x.begin(), x.end());
    double s = 0.0;
    for (double v : x) s += std::exp(v - m);
    double lse = m + std::log(s);
    std::vector<double> lp(x.size());
    for (size_t i = 0; i < x.size(); ++i) lp[i] = x[i] - lse;
    return lp;
  };
  auto entropy = [](const std::vector<double>& lp) {
    double h = 0.0;
    for (double l : lp) { double p = std::exp(l); if (p > 0.0) h -= p * l; }
    return h;
  };
  auto kl = [](const std::vector<double>& lp, const std::vector<double>& lq) {
    double d = 0.0;
    for (size_t i = 0; i < lp.size(); ++i) {
      double p = std::exp(lp[i]);
      if (p > 0.0) d += p * (lp[i] - lq[i]);
    }
    return d;
  };
  std::vector<double> lp_unc = log_softmax(z);
  std::vector<double> lp_10 = log_softmax(c10);
  std::vector<double> lp_30 = log_softmax(c30);
  double H_unc = entropy(lp_unc), H_10 = entropy(lp_10), H_30 = entropy(lp_30);
  double kl_10_unc = kl(lp_10, lp_unc), kl_10_30 = kl(lp_10, lp_30);
  double p90 = LogitStatsPercentile(absz, 90.0);
  double max_abs = 0.0;
  int ge5 = 0, ge10 = 0, ge20 = 0;
  for (double a : absz) {
    max_abs = std::max(max_abs, a);
    if (a >= 5.0) ++ge5;
    if (a >= 10.0) ++ge10;
    if (a >= 20.0) ++ge20;
  }
  const double dn = static_cast<double>(n);
  // --- WO-1 Phase 3 per-decision policy-shape stats ---
  // Normalized entropy puts decisions with different legal-set sizes on one
  // scale: H/log(n) is 1.0 for a uniform policy and 0.0 for a deterministic
  // one. n == 1 has log(n) == 0, so it is defined to 0.0 (a forced decision
  // carries no choice entropy) rather than emitting a NaN.
  const double norm_entropy = (n > 1) ? (H_unc / std::log(dn)) : 0.0;
  const double eff_action_count = std::exp(H_unc);
  double max_prob = 0.0;
  for (double l : lp_unc) max_prob = std::max(max_prob, std::exp(l));
  // Saturation against the cap this run is ACTUALLY configured with, unlike
  // the fixed 5/10/20 columns above which predate the cap being a flag.
  int ge_cap = 0, ge_2cap = 0;
  for (double a : absz) {
    if (a >= configured_cap) ++ge_cap;
    if (a >= 2.0 * configured_cap) ++ge_2cap;
  }
  std::lock_guard<std::mutex> lock(g_logit_dump_mutex);
  out << episode_id << ',' << round << ',' << role << ',' << n << ','
      << absl::StrFormat("%.6f", max_abs) << ',' << absl::StrFormat("%.6f", p90)
      << ',' << absl::StrFormat("%.6f", H_unc) << ','
      << absl::StrFormat("%.6f", H_10) << ',' << absl::StrFormat("%.6f", H_30)
      << ',' << absl::StrFormat("%.6f", kl_10_unc) << ','
      << absl::StrFormat("%.6f", kl_10_30) << ','
      << absl::StrFormat("%.6f", ge5 / dn) << ','
      << absl::StrFormat("%.6f", ge10 / dn) << ','
      << absl::StrFormat("%.6f", ge20 / dn) << ','
      // WO-1 Phase 3 (appended; all columns above are unchanged)
      << absl::StrFormat("%.6f", norm_entropy) << ','
      << absl::StrFormat("%.6f", max_prob) << ','
      << absl::StrFormat("%.6f", eff_action_count) << ','
      << absl::StrFormat("%.6f", configured_cap) << ','
      << absl::StrFormat("%.6f", ge_cap / dn) << ','
      << absl::StrFormat("%.6f", ge_2cap / dn) << '\n';
}

// ---------------------------------------------------------------------------
// Market diagnostics and aggregate reporting
// ---------------------------------------------------------------------------
void ReportMarketDiagnostics(
    const EvalConfig& cfg,
    const std::vector<dune_market_diag::CompactPerGameMarketDiagnostics>& market_results) {
  if (!cfg.record_market_diagnostics) return;
  auto report = dune_market_diag::MarketDiagnosticsReport::AggregateReport(
      market_results, cfg.total_games);
  if (!cfg.market_diag_json_path.empty()) {
    if (!report.WriteJson(cfg.market_diag_json_path)) {
      open_spiel::SpielFatalError("Failed writing market diagnostics JSON to " + cfg.market_diag_json_path);
    }
    std::cout << "Market diagnostics JSON written to " << cfg.market_diag_json_path << "\n";
  }
  if (!cfg.market_diag_csv_path.empty()) {
    if (!report.WriteCsv(cfg.market_diag_csv_path)) {
      open_spiel::SpielFatalError("Failed writing market diagnostics CSV to " + cfg.market_diag_csv_path);
    }
    std::cout << "Market diagnostics CSV written to " << cfg.market_diag_csv_path << "\n";
  }
  if (!cfg.market_diag_games_path.empty()) {
    if (!dune_market_diag::WritePerGameJsonl(cfg.market_diag_games_path, market_results)) {
      open_spiel::SpielFatalError("Failed writing market diagnostics games JSONL to " + cfg.market_diag_games_path);
    }
    std::cout << "Market diagnostics games JSONL written to " << cfg.market_diag_games_path << "\n";

    // Re-read and assert exact bit-for-bit reconciliation with in-memory aggregate report
    auto read_games = dune_market_diag::ReadPerGameJsonl(cfg.market_diag_games_path);
    auto recomputed_report = dune_market_diag::MarketDiagnosticsReport::AggregateReport(read_games, cfg.total_games);
    dune_market_diag::VerifyReconciliation(report, recomputed_report);
    std::cout << "Market diagnostics per-game reconciliation verified successfully (" << read_games.size() << " games).\n";
  }
}

void ReportAggregateResults(const EvalConfig& cfg,
                            const EvalRuntime& rt,
                            double elapsed_secs,
                            const std::vector<GameResult>& results) {
  const int total_games = cfg.total_games;

  // --- Write per-game JSONL ---
  std::ofstream jsonl_out;
  if (!cfg.output_dir.empty()) {
    std::filesystem::create_directories(cfg.output_dir);
    std::string jsonl_path =
        (std::filesystem::path(cfg.output_dir) / "games.jsonl").string();
    jsonl_out.open(jsonl_path);
    if (!jsonl_out.is_open()) {
      std::cerr << "Error: cannot open " << jsonl_path << " for writing.\n";
      std::exit(1);
    }
  }

  // Compute aggregates while writing JSONL
  int first_place_count = 0;
  double return_sum = 0.0;
  double vp_sum = 0.0;
  double cand_r7_track_vp_sum = 0.0;
  double max4_r7_track_vp_sum = 0.0;
  int r7_track_vp_games = 0;
  int ending_round_le7 = 0;
  int ending_round_le8 = 0;
  int placement_counts[4] = {0, 0, 0, 0};
  int wins_by_seat[4] = {0, 0, 0, 0};
  int games_by_seat[4] = {0, 0, 0, 0};

  for (int i = 0; i < total_games; ++i) {
    const GameResult& gr = results[i];

    // Aggregation
    if (gr.placement == 1) {
      first_place_count++;
      wins_by_seat[gr.seat]++;
    }
    return_sum += gr.game_return;
    vp_sum += gr.final_scored_vp;
    if (gr.has_r7_track_vp && gr.seat >= 0 && gr.seat < kNumPlayers) {
      cand_r7_track_vp_sum += static_cast<double>(gr.r7_track_vp[gr.seat]);
      int max4_vp = *std::max_element(gr.r7_track_vp.begin(), gr.r7_track_vp.end());
      max4_r7_track_vp_sum += static_cast<double>(max4_vp);
      r7_track_vp_games++;
    }
    if (gr.ending_round <= 7) ending_round_le7++;
    if (gr.ending_round <= 8) ending_round_le8++;
    if (gr.placement >= 1 && gr.placement <= 4) {
      placement_counts[gr.placement - 1]++;
    }
    games_by_seat[gr.seat]++;

    // JSONL output with opponents serialized as a JSON array of strings
    if (jsonl_out.is_open()) {
      jsonl_out << "{\"episode_id\":" << gr.episode_id
                << ",\"seat\":" << gr.seat
                << ",\"opponents\":[";
      for (size_t idx = 0; idx < gr.opponents.size(); ++idx) {
        if (idx > 0) jsonl_out << ",";
        jsonl_out << "\"" << gr.opponents[idx] << "\"";
      }
      jsonl_out << "]"
                << ",\"placement\":" << gr.placement
                << ",\"return\":" << absl::StrFormat("%.4f", gr.game_return)
                << ",\"ending_round\":" << gr.ending_round
                << ",\"track_vp\":" << gr.current_vp
                << ",\"final_scored_vp\":" << gr.final_scored_vp
                // PWO-5 section 13.5. All four seats' TRUE final VP, from the
                // engine's FinalScoredVp, and the exact registered margin.
                // `%.17g` is round-trip precision: `vp_margin` has denominator
                // 3, so 1/3 and 2/3 are not exactly representable and a
                // fixed-decimal format would quietly change the estimand.
                << ",\"final_scored_vp_all\":"
                << JsonIntArray(gr.final_scored_vp_all)
                << ",\"vp_margin\":"
                << (gr.vp_margin_valid
                        ? absl::StrFormat("%.17g", gr.vp_margin)
                        : std::string("null"))
                << ",\"vp_end_r5\":" << JsonVpArrayOrNull(gr.has_r5, gr.vp_end_r5)
                << ",\"vp_end_r6\":" << JsonVpArrayOrNull(gr.has_r6, gr.vp_end_r6)
                << ",\"vp_end_r7\":" << JsonVpArrayOrNull(gr.has_r7, gr.vp_end_r7)
                << ",\"end_trigger_player\":" << JsonIntOrNull(gr.end_trigger_player)
                << ",\"end_trigger_round\":" << JsonIntOrNull(gr.end_trigger_round)
                << ",\"winner_triggered\":" << JsonBoolOrNull(gr.winner_triggered)
                // --- WO-1 Phase 3: pace/tempo (new keys only; every key above
                // is byte-identical to the pre-WO-1 emission) ---
                << ",\"candidate_seat\":" << gr.seat
                << ",\"candidate_leader\":"
                << (gr.seat >= 0 && gr.seat < kNumPlayers ? gr.player_leaders[gr.seat] : -1)
                << ",\"leaders\":" << JsonIntArray(gr.player_leaders)
                << ",\"vp_end_by_round\":"
                << JsonVpByRound(gr.rounds_captured, gr.vp_end_by_round)
                << ",\"first_round_vp_ge_11\":"
                << JsonIntArrayWithNulls(gr.first_round_vp_ge_11)
                << ",\"terminal_threshold_set\":"
                << JsonIntVector(gr.terminal_threshold_set)
                << ",\"candidate_in_terminal_threshold_set\":"
                << (gr.candidate_in_terminal_threshold_set ? "true" : "false")
                << ",\"terminal_reason\":\"" << gr.terminal_reason << "\""
                << ",\"specimen_conversions\":"
                << JsonIntArray(gr.specimen_conversions)
                << ",\"atomics_used\":"
                << JsonBoolArray(gr.atomics_used)
                << ",\"atomics_in_reveal\":"
                << JsonIntArrayWithNulls(gr.atomics_in_reveal)
                << ",\"atomics_persuasion\":"
                << JsonIntArrayWithNulls(gr.atomics_persuasion)
                << ",\"atomics_round\":"
                << JsonIntArrayWithNulls(gr.atomics_round)
                << ",\"plot_plays\":"
                << JsonIntArray(gr.plot_plays)
                << ",\"plot_pre_hand_sizes\":"
                << JsonNestedIntVector(gr.plot_pre_hand_sizes)
                << ",\"research_col\":"
                << JsonIntArray(gr.research_col)
                << ",\"research_row\":"
                << JsonIntArray(gr.research_row)
                << ",\"research_paths\":"
                << JsonResearchPaths(gr.research_paths)
                << ",\"reveals_with_agents_remaining\":"
                << JsonIntArray(gr.reveals_with_agents_remaining)
                << ",\"solari_end_by_round\":"
                << JsonVpByRound(gr.rounds_captured, gr.solari_end_by_round)
                << ",\"seed\":" << gr.chance_seed;
      if (cfg.record_history) {
        jsonl_out << ",\"history\":" << JsonActionVector(gr.history);
      }
      jsonl_out << "}\n";
    }
  }

  if (jsonl_out.is_open()) {
    jsonl_out.close();
    std::cout << "Per-game results written to "
              << (std::filesystem::path(cfg.output_dir) / "games.jsonl").string()
              << "\n\n";
  }

  // --- Compute aggregate statistics ---
  double mean_return = total_games > 0 ? return_sum / total_games : 0.0;
  double mean_vp = total_games > 0 ? vp_sum / total_games : 0.0;

  // Confidence intervals
  WilsonCI first_place_ci = WilsonScore(first_place_count, total_games);
  WilsonCI round_le7_ci = WilsonScore(ending_round_le7, total_games);
  WilsonCI round_le8_ci = WilsonScore(ending_round_le8, total_games);

  // Return CI (normal approximation)
  double return_var = 0.0;
  double vp_var = 0.0;
  for (int i = 0; i < total_games; ++i) {
    double d = results[i].game_return - mean_return;
    return_var += d * d;
    double dv = results[i].final_scored_vp - mean_vp;
    vp_var += dv * dv;
  }
  double return_se = total_games > 1
      ? std::sqrt(return_var / (total_games - 1)) / std::sqrt(total_games)
      : 0.0;
  double vp_se = total_games > 1
      ? std::sqrt(vp_var / (total_games - 1)) / std::sqrt(total_games)
      : 0.0;

  // --- Print aggregate results ---
  std::cout << absl::StrFormat("Completed %d games in %.1f seconds (%.1f g/s)\n\n",
                               total_games, elapsed_secs,
                               total_games / elapsed_secs);

  std::cout << "=== Aggregate Results ===\n\n";

  std::cout << absl::StrFormat(
      "First-place rate:     %.2f%%   Wilson 95%% CI [%.2f%%, %.2f%%]\n",
      first_place_ci.point * 100, first_place_ci.lower * 100,
      first_place_ci.upper * 100);

  std::cout << absl::StrFormat(
      "Mean final_scored_vp: %.3f   95%% CI [%.3f, %.3f]\n",
      mean_vp, mean_vp - 1.96 * vp_se, mean_vp + 1.96 * vp_se);

  std::cout << absl::StrFormat(
      "Mean return:          %.4f   95%% CI [%.4f, %.4f]\n",
      mean_return, mean_return - 1.96 * return_se,
      mean_return + 1.96 * return_se);

  const double cand_r7_track_vp_mean =
      r7_track_vp_games > 0 ? cand_r7_track_vp_sum / r7_track_vp_games : 0.0;
  const double max4_r7_track_vp_mean =
      r7_track_vp_games > 0 ? max4_r7_track_vp_sum / r7_track_vp_games : 0.0;
  const double frac_rounds_le_7 = round_le7_ci.point;

  std::cout << absl::StrFormat(
      "P(round <= 7):        %.2f%%   Wilson 95%% CI [%.2f%%, %.2f%%]\n",
      round_le7_ci.point * 100, round_le7_ci.lower * 100,
      round_le7_ci.upper * 100);

  std::cout << absl::StrFormat(
      "End-of-R7 track VP:   cand=%.3f   max4=%.3f   frac_rounds_le_7=%.4f\n",
      cand_r7_track_vp_mean, max4_r7_track_vp_mean, frac_rounds_le_7);

  std::cout << absl::StrFormat(
      "P(round <= 8):        %.2f%%   Wilson 95%% CI [%.2f%%, %.2f%%]\n",
      round_le8_ci.point * 100, round_le8_ci.lower * 100,
      round_le8_ci.upper * 100);

  std::cout << "\nPlacement distribution:\n";
  for (int r = 0; r < 4; ++r) {
    WilsonCI pci = WilsonScore(placement_counts[r], total_games);
    std::cout << absl::StrFormat(
        "  %dst: %5d / %d  (%.2f%%)  Wilson [%.2f%%, %.2f%%]\n",
        r + 1, placement_counts[r], total_games,
        pci.point * 100, pci.lower * 100, pci.upper * 100);
  }

  std::cout << "\nWin rate by seat:\n";
  for (int s = 0; s < 4; ++s) {
    if (games_by_seat[s] > 0) {
      WilsonCI sci = WilsonScore(wins_by_seat[s], games_by_seat[s]);
      std::cout << absl::StrFormat(
          "  Seat P%d: %d/%d  (%.2f%%)  Wilson [%.2f%%, %.2f%%]\n",
          s, wins_by_seat[s], games_by_seat[s],
          sci.point * 100, sci.lower * 100, sci.upper * 100);
    }
  }

  if (absl::GetFlag(FLAGS_candidate_r758_teacher)) {
    int64_t searched = g_teacher_searched_roots.load();
    int64_t overrides = g_teacher_overrides.load();
    double roots_per_game = total_games > 0 ? static_cast<double>(searched) / total_games : 0.0;
    double overrides_per_game = total_games > 0 ? static_cast<double>(overrides) / total_games : 0.0;
    double yield_pct = searched > 0 ? (100.0 * overrides / searched) : 0.0;
    std::cout << absl::StrFormat(
        "\nTeacher Search: %d roots (%.2f/game), %d overrides (%.2f/game, %.2f%% yield)\n",
        searched, roots_per_game, overrides, overrides_per_game, yield_pct);
  }

  // --- Write aggregate JSON with seat-wise and placement counts ---
  if (!cfg.output_dir.empty()) {
    std::string agg_path =
        (std::filesystem::path(cfg.output_dir) / "aggregate.json").string();
    std::ofstream agg_out(agg_path);
    if (agg_out.is_open()) {
      agg_out << "{\n"
              << "  \"model_checkpoint\": \""
              << cfg.model_checkpoint << "\",\n"
              << "  \"opponent_checkpoints\": \""
              << (cfg.use_opponent_model ? cfg.opponent_str : "random") << "\",\n"
              << "  \"domain\": \"" << cfg.domain_str << "\",\n"
              << "  \"base_seed\": " << cfg.base_seed << ",\n"
              << "  \"num_games\": " << total_games << ",\n"
              << "  \"start_episode_id\": " << cfg.start_episode_id << ",\n"
              << "  \"greedy\": " << (cfg.greedy ? "true" : "false") << ",\n"
              << "  \"opponent_greedy\": "
              << (cfg.has_opponent_greedy ? (cfg.opponent_greedy ? "true" : "false")
                                          : "null") << ",\n"
              << "  \"opponent_temperature\": "
              << (cfg.has_opponent_temperature
                      ? absl::StrFormat("%.6f", cfg.opponent_temperature)
                      : std::string("null")) << ",\n"
              << "  \"temperature\": "
              << absl::StrFormat("%.2f", cfg.temperature) << ",\n"
              << "  \"candidate_logit_cap\": "
              << absl::StrFormat("%.4f", cfg.candidate_logit_cap) << ",\n"
              << "  \"opponent_logit_cap\": "
              << absl::StrFormat("%.4f", cfg.opponent_logit_cap) << ",\n"
              << "  \"hidden_dim\": " << rt.main_hidden_dim << ",\n"
              << "  \"num_blocks\": " << rt.main_num_blocks << ",\n"
              << "  \"candidate_input_dim\": " << rt.main_input_dim << ",\n"
              << "  \"candidate_market_mode\": \"" << MarketModeToString(rt.cand_market_mode) << "\",\n"
              << "  \"opp_hidden_dim\": " << (rt.opp_metadata.empty() ? -1 : rt.opp_metadata[0].hidden_dim) << ",\n"
              << "  \"opp_num_blocks\": " << (rt.opp_metadata.empty() ? -1 : rt.opp_metadata[0].num_blocks) << ",\n"
              << "  \"execution_mode\": \"" << (cfg.deterministic ? "deterministic" : "batched") << "\",\n"
              << "  \"threads\": " << rt.num_threads << ",\n"
              << "  \"detected_opponent_architectures\": [\n";
      for (size_t i = 0; i < rt.opp_metadata.size(); ++i) {
        agg_out << "    {\n"
                << "      \"checkpoint\": \"" << rt.opp_metadata[i].path << "\",\n"
                << "      \"hidden_dim\": " << rt.opp_metadata[i].hidden_dim << ",\n"
                << "      \"num_blocks\": " << rt.opp_metadata[i].num_blocks << ",\n"
                << "      \"input_dim\": " << rt.opp_metadata[i].input_dim << ",\n"
                << "      \"market_mode\": \"" << rt.opp_metadata[i].market_mode_str << "\"\n"
                << "    }" << (i + 1 < rt.opp_metadata.size() ? "," : "") << "\n";
      }
      agg_out << "  ],\n"
              << "  \"elapsed_seconds\": "
              << absl::StrFormat("%.1f", elapsed_secs) << ",\n"
              << "  \"first_place_rate\": "
              << absl::StrFormat("%.6f", first_place_ci.point) << ",\n"
              << "  \"first_place_wilson_lower\": "
              << absl::StrFormat("%.6f", first_place_ci.lower) << ",\n"
              << "  \"first_place_wilson_upper\": "
              << absl::StrFormat("%.6f", first_place_ci.upper) << ",\n"
              << "  \"mean_final_scored_vp\": "
              << absl::StrFormat("%.4f", mean_vp) << ",\n"
              << "  \"mean_final_scored_vp_ci_lower\": "
              << absl::StrFormat("%.4f", mean_vp - 1.96 * vp_se) << ",\n"
              << "  \"mean_final_scored_vp_ci_upper\": "
              << absl::StrFormat("%.4f", mean_vp + 1.96 * vp_se) << ",\n"
              << "  \"mean_return\": "
              << absl::StrFormat("%.6f", mean_return) << ",\n"
              << "  \"mean_return_ci_lower\": "
              << absl::StrFormat("%.6f", mean_return - 1.96 * return_se)
              << ",\n"
              << "  \"mean_return_ci_upper\": "
              << absl::StrFormat("%.6f", mean_return + 1.96 * return_se)
              << ",\n"
              << "  \"cand_r7_track_vp_mean\": "
              << absl::StrFormat("%.4f", cand_r7_track_vp_mean) << ",\n"
              << "  \"max4_r7_track_vp_mean\": "
              << absl::StrFormat("%.4f", max4_r7_track_vp_mean) << ",\n"
              << "  \"frac_rounds_le_7\": "
              << absl::StrFormat("%.6f", frac_rounds_le_7) << ",\n"
              << "  \"p_round_le7\": "
              << absl::StrFormat("%.6f", round_le7_ci.point) << ",\n"
              << "  \"p_round_le7_wilson_lower\": "
              << absl::StrFormat("%.6f", round_le7_ci.lower) << ",\n"
              << "  \"p_round_le7_wilson_upper\": "
              << absl::StrFormat("%.6f", round_le7_ci.upper) << ",\n"
              << "  \"p_round_le8\": "
              << absl::StrFormat("%.6f", round_le8_ci.point) << ",\n"
              << "  \"p_round_le8_wilson_lower\": "
              << absl::StrFormat("%.6f", round_le8_ci.lower) << ",\n"
              << "  \"p_round_le8_wilson_upper\": "
              << absl::StrFormat("%.6f", round_le8_ci.upper) << ",\n"
              << "  \"placement_1st_count\": " << placement_counts[0] << ",\n"
              << "  \"placement_2nd_count\": " << placement_counts[1] << ",\n"
              << "  \"placement_3rd_count\": " << placement_counts[2] << ",\n"
              << "  \"placement_4th_count\": " << placement_counts[3] << ",\n"
              << "  \"seat_0_wins\": " << wins_by_seat[0] << ",\n"
              << "  \"seat_0_games\": " << games_by_seat[0] << ",\n"
              << "  \"seat_1_wins\": " << wins_by_seat[1] << ",\n"
              << "  \"seat_1_games\": " << games_by_seat[1] << ",\n"
              << "  \"seat_2_wins\": " << wins_by_seat[2] << ",\n"
              << "  \"seat_2_games\": " << games_by_seat[2] << ",\n"
              << "  \"seat_3_wins\": " << wins_by_seat[3] << ",\n"
              << "  \"seat_3_games\": " << games_by_seat[3];
      if (absl::GetFlag(FLAGS_candidate_r758_teacher)) {
        int64_t searched = g_teacher_searched_roots.load();
        int64_t overrides = g_teacher_overrides.load();
        double roots_per_game = total_games > 0 ? static_cast<double>(searched) / total_games : 0.0;
        double overrides_per_game = total_games > 0 ? static_cast<double>(overrides) / total_games : 0.0;
        double yield_pct = searched > 0 ? (100.0 * overrides / searched) : 0.0;
        agg_out << ",\n"
                << "  \"teacher_searched_roots\": " << searched << ",\n"
                << "  \"teacher_overrides\": " << overrides << ",\n"
                << "  \"teacher_roots_per_game\": " << absl::StrFormat("%.4f", roots_per_game) << ",\n"
                << "  \"teacher_overrides_per_game\": " << absl::StrFormat("%.4f", overrides_per_game) << ",\n"
                << "  \"teacher_override_yield_pct\": " << absl::StrFormat("%.2f", yield_pct);
      }
      agg_out << "\n}\n";
      agg_out.close();
      if (absl::GetFlag(FLAGS_candidate_r758_teacher)) {
        std::string sum_path = (std::filesystem::path(cfg.output_dir) / "summary.json").string();
        std::error_code ec;
        std::filesystem::copy_file(agg_path, sum_path, std::filesystem::copy_options::overwrite_existing, ec);
      }
      std::cout << "\nAggregate results written to " << agg_path << "\n";
    }
  }
}

}  // namespace dune_pop_eval
}  // namespace open_spiel
