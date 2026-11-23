#include "dune_ppo_training_utils_internal.h"
#include "dune_search_routing.h"
#include "open_spiel/utils/json.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <limits>
#include <string>
#include <vector>

#ifdef OPEN_SPIEL_BUILD_WITH_LIBTORCH
#include "open_spiel/abseil-cpp/absl/flags/flag.h"

namespace open_spiel {

// Linear-interpolated percentile over a sorted vector, matching the convention
// the frozen scripts/eval/logit_cap_audit.py uses so the two agree on the same
// data. `sorted` must be non-empty and ascending.
double SortedPercentile(const std::vector<float>& sorted, double q) {
  if (sorted.empty()) return 0.0;
  if (sorted.size() == 1) return sorted[0];
  const double pos = (q / 100.0) * (static_cast<double>(sorted.size()) - 1.0);
  const size_t lo = static_cast<size_t>(std::floor(pos));
  const size_t hi = static_cast<size_t>(std::ceil(pos));
  if (lo == hi) return sorted[lo];
  const double frac = pos - static_cast<double>(lo);
  return sorted[lo] * (1.0 - frac) + sorted[hi] * frac;
}

PrecapAbszStats SummarizePrecapAbsz(std::vector<float> values) {
  PrecapAbszStats out;
  out.decisions = static_cast<int64_t>(values.size());
  if (values.empty()) return out;
  std::sort(values.begin(), values.end());
  out.p95 = SortedPercentile(values, 95.0);
  out.p99 = SortedPercentile(values, 99.0);
  out.max = values.back();
  int64_t ge10 = 0;
  for (float v : values) {
    if (v >= 10.0f) ++ge10;
  }
  out.frac_ge10 = static_cast<double>(ge10) / static_cast<double>(values.size());
  return out;
}

// Diagnostics CSV schema v3 (Phase A, 2026-07 screens): v2 plus the trailing
// durable PPO-side canary columns. v1 -> v2 (WO-17) added the eight aux
// columns; v2 -> v3 adds only PPO-side columns and changes no existing one, so
// column meanings are stable across versions and only the tail grows.
//
// Appending is only safe against a file carrying this exact header, so a
// resumed run pointed at a v1 or v2 file fails loudly instead of writing rows
// the header cannot describe. JSONL is self-describing per line and needs no
// such gate.
//
// Deliberately NOT reusing the WO-17 column `ppo_grad_norm_mean` as the PPO
// grad-norm canary: that column is populated only when online collection fed
// aux examples and reads 0.0 with collection off. `grad_norm_mean` /
// `grad_norm_max` below are always populated.
const char* const kDiagnosticsCsvHeader =
    "seed,run_uuid,run_prefix,config_fingerprint,update,rollout_hash,episode_ids_unique,policy_kl_before,return_min,return_max,return_p50,"
    "return_p95,return_p99,abs_return_p99,fraction_targets_outside_1,fraction_critic_near_1,"
    "total_transitions,nontrivial_transitions,forced_transitions,epoch_kls,"
    "conflict_vp_generated,conflict_vp_attributed,conflict_vp_unattributed,"
    "raw_conflict_vp,raw_noncombat_vp,raw_total_vp,validation_kl,"
    "aux_examples_used,aux_search_loss_coef,aux_ce,aux_value_mse,"
    "aux_grad_norm_mean,ppo_grad_norm_mean,aux_ppo_norm_ratio,aux_ratio_abort,"
    "entropy,approx_kl,kl_early_stop,kl_early_stop_epoch,"
    "grad_norm_mean,grad_norm_max,value_loss,nonfinite_abort,"
    "precap_absz_n,precap_absz_p95,precap_absz_p99,precap_absz_max,frac_decisions_absz_ge10,"
    "precap_absz_n_primary,precap_absz_p95_primary,precap_absz_p99_primary,"
    "precap_absz_max_primary,frac_decisions_absz_ge10_primary,"
    "precap_absz_n_continuation,precap_absz_p95_continuation,precap_absz_p99_continuation,"
    "precap_absz_max_continuation,frac_decisions_absz_ge10_continuation,"
    "precap_absz_n_purchase,precap_absz_p95_purchase,precap_absz_p99_purchase,"
    "precap_absz_max_purchase,frac_decisions_absz_ge10_purchase,"
    // --- v4: WO-1 Phase 5 scratch-anomaly bundle (log-only) ---
    "value_clip_fraction,explained_variance,policy_head_grad_norm,"
    "value_head_grad_norm,trunk_grad_norm,minibatches_executed";

// --- PWO-5 section 14.1c: schema v5 = v4 (69 cols) + 17 canary cols 70-86. --
//
// The names, the ORDER and the pairing of every role mean with its own `_n`
// denominator are all frozen by section 14.1c. The six role denominators are
// what makes the partition identity AUDITABLE; revision 7 of the registration
// registered the identity while emitting no counts at all, which made it
// decoration rather than a check.
//
// Role coverage is a COMPLETE PARTITION of NE's domain at the registered
// measurement site, not a convenient subset: `DuneDecisionRole` has SEVEN
// members and kOtherOptional ALONE is 45.82% of the PWO-4 stream, so the
// existing precap_absz convention of primary/continuation/purchase would leave
// the canary blind to nearly half of all decisions. The seventh role,
// kForcedOrBookkeeping, is excluded by proof: the classifier returns it iff the
// searched player is the chance player OR n_legal <= 1; chance nodes `continue`
// in the rollout loop BEFORE any policy evaluation or role classification, so
// the first disjunct is unsatisfiable at this site; and NE's population already
// requires n_legal >= 2. A norm_entropy_forced column would be identically
// empty by construction.
const char* const kDiagnosticsCsvHeaderV5Suffix =
    ",norm_entropy,norm_entropy_n,"
    "norm_entropy_primary,norm_entropy_n_primary,"
    "norm_entropy_continuation,norm_entropy_n_continuation,"
    "norm_entropy_purchase,norm_entropy_n_purchase,"
    "norm_entropy_combat_intrigue,norm_entropy_n_combat_intrigue,"
    "norm_entropy_other_optional,norm_entropy_n_other_optional,"
    "norm_entropy_leader_selection,norm_entropy_n_leader_selection,"
    "max_action_prob,frac_legal_absz_ge_cap,frac_legal_absz_ge_cap_n";

// --- v6 (WO-PERF-1): one column, gated on --diag_prepass_mode=cadenced. ----
//
// Gated, not unconditional, for the same reason v5 is gated on
// --emit_canary_columns: the v4/69 and v5/86 shapes are pinned character by
// character (dune_pwo5_schema_test; PWO-5 registration section 17.5 item 14
// makes any other header a STOP for runs under that registration), so a mode
// that reproduces today's behavior must also reproduce today's header. In
// cadenced mode the column is what makes a skipped KL measurement (-1)
// distinguishable from a measured KL of zero (>= 0; 0 itself means "measured
// over zero nontrivial transitions"). Appended AFTER the v5 canary
// block so the frozen positions 70-86 never move.
const char* const kDiagnosticsCsvHeaderV6Suffix = ",measured_transitions";

const char* const kDiagnosticsCsvHeaderStudySuffix =
    ",single_action_transitions,single_action_fraction,"
    "single_action_value_mse_before,single_action_value_mse_after,"
    "single_action_sample_size,unfiltered_kl,unfiltered_kl_sample_size,"
    "retained_transitions,optimizer_minibatches";

std::string DiagnosticsCsvHeader(bool emit_canary_columns) {
  std::string header(kDiagnosticsCsvHeader);
  if (emit_canary_columns) header += kDiagnosticsCsvHeaderV5Suffix;
  if (::absl::GetFlag(::FLAGS_diag_prepass_mode) == "cadenced") {
    header += kDiagnosticsCsvHeaderV6Suffix;
  }
  if (::absl::GetFlag(::FLAGS_purchase_exploration)) {
    header += ",forced_buys";
  }
  if (::absl::GetFlag(::FLAGS_enable_study_readouts) ||
      ::absl::GetFlag(::FLAGS_advantage_filtering)) {
    header += kDiagnosticsCsvHeaderStudySuffix;
  }
  if (::absl::GetFlag(::FLAGS_enable_ema_tracking)) {
    header += ",ema_magnet_kl";
  }
  return header;
}

// Returns true if `filepath` already holds a header line, and sets `out_header`
// to it (trailing CR/LF stripped). An absent or zero-length file has no header.
bool ReadDiagnosticsCsvHeader(const std::string& filepath,
                              std::string* out_header) {
  std::ifstream ifs(filepath);
  if (!ifs) return false;
  std::string line;
  if (!std::getline(ifs, line)) return false;
  while (!line.empty() && (line.back() == '\r' || line.back() == '\n')) {
    line.pop_back();
  }
  *out_header = line;
  return true;
}

namespace {

std::string F17(double v) {
  char buf[64];
  std::snprintf(buf, sizeof(buf), "%.17g", v);
  return std::string(buf);
}

}  // namespace

// --- WO-PERF-TIMING sidecar ------------------------------------------------

std::string PhaseTimingPath(const std::string& diagnostics_path) {
  if (diagnostics_path.empty()) return std::string();
  std::filesystem::path p(diagnostics_path);
  std::filesystem::path dir = p.parent_path();
  return (dir / "phase_timing.jsonl").string();
}

std::string PhaseTimingContract() {
  std::string s =
      "{\"record\":\"header\",\"schema\":\"phase_timing.v2\","
      "\"float_format\":\"%.17g\",\"phases\":[";
  for (int i = 0; i < kPhaseTimingNumPhases; ++i) {
    if (i) s += ",";
    s += "\"";
    s += kPhaseTimingNames[i];
    s += "\"";
  }
  s += "]";
  return s;
}

// %.17g of a non-finite double emits `inf`/`nan`, which is NOT valid JSON --
// `json.loads('{"x":inf}')` rejects it. Gate T3 asks for finite values; this is
// what makes a violation visible as data rather than as an unparseable file.
std::string F17Finite(double v, bool* saw_nonfinite) {
  if (!std::isfinite(v)) {
    *saw_nonfinite = true;
    return "null";
  }
  return F17(v);
}

// run_prefix and run_uuid are caller-supplied and land inside JSON strings.
std::string JsonEscape(const std::string& in) {
  std::string out;
  out.reserve(in.size() + 8);
  for (char c : in) {
    switch (c) {
      case '"':  out += "\\\""; break;
      case '\\': out += "\\\\"; break;
      case '\n': out += "\\n"; break;
      case '\r': out += "\\r"; break;
      case '\t': out += "\\t"; break;
      default:
        if (static_cast<unsigned char>(c) < 0x20) {
          char buf[8];
          std::snprintf(buf, sizeof(buf), "\\u%04x", c);
          out += buf;
        } else {
          out += c;
        }
    }
  }
  return out;
}

void WritePhaseTimingToPath(const std::string& path, int update,
                            const PpoUpdateStats& stats, double ppo_elapsed_s,
                            const std::string& run_uuid,
                            const std::string& run_prefix) {
  if (!stats.phase_timings.enabled) return;
  if (path.empty()) return;

  const std::string contract = PhaseTimingContract();
  bool needs_header = true;
  {
    std::ifstream in(path);
    std::string first;
    if (in && std::getline(in, first) && !first.empty()) {
      needs_header = false;
      if (first.compare(0, contract.size(), contract) != 0) {
        SpielFatalError(
            "Phase-timing schema mismatch at: " + path +
            "\n  Existing header contract: " + first.substr(0, contract.size()) +
            "\n  This binary writes:       " + contract +
            "\n  Appending would produce records the header cannot describe.");
      }
    }
  }

  std::ofstream out(path, std::ios::app);
  if (!out) {
    SpielFatalError("Could not open phase-timing sidecar: " + path);
  }
  if (needs_header) {
    out << contract << ",\"run_uuid\":\"" << JsonEscape(run_uuid)
        << "\",\"run_prefix\":\"" << JsonEscape(run_prefix) << "\"}\n";
  }

  const PpoPhaseTimings& t = stats.phase_timings;
  bool nonfinite = false;

  // run_uuid is repeated on EVERY record, not just the header. The header is
  // written once per FILE, so a resume or a second run pointed at the same
  // --diagnostics_path directory would otherwise append its updates under the
  // first run's header with no per-record identity -- and update numbers repeat
  // across runs. WriteDiagnostics puts run_uuid on every CSV row for exactly
  // this reason. It matters more here: --phase_timing_mode is deliberately
  // absent from ComputeConfigFingerprint(), so a manifest cannot detect a
  // mid-run mode flip and this field is the only provenance a record carries.
  out << "{\"record\":\"update\",\"update\":" << update
      << ",\"run_uuid\":\"" << JsonEscape(run_uuid) << "\""
      << ",\"mode\":\"" << JsonEscape(::absl::GetFlag(::FLAGS_phase_timing_mode))
      << "\",\"timer\":\"" << (t.cuda ? "cuda_event" : "steady_clock") << "\"";

  out << ",\"device_s\":{";
  for (int i = 0; i < kPhaseTimingNumPhases; ++i) {
    if (i) out << ",";
    out << "\"" << kPhaseTimingNames[i] << "\":"
        << F17Finite(t.device_s[i], &nonfinite);
  }
  out << "}";

  out << ",\"host_s\":{";
  for (int i = 0; i < kPhaseTimingNumPhases; ++i) {
    if (i) out << ",";
    out << "\"" << kPhaseTimingNames[i] << "\":"
        << F17Finite(t.host_s[i], &nonfinite);
  }
  out << "}";

  // The metadata that lets a reader see the attribution discrepancy directly
  // rather than infer it (work order §3.3, gate T4).
  //
  // T4 asks for total_attributed_s to be within a declared tolerance of the
  // trainer's PPO figure. DECLARED DEVIATION: on CUDA, total_attributed_s is
  // sum(device_s), which deliberately excludes host-bound time and therefore
  // CANNOT approach ppo_elapsed_s. The quantity that is comparable is
  // total_host_attributed_s, and `unattributed_host_s` is its signed residual
  // against the trainer's own figure. Declared tolerance: the residual is
  // expected to be POSITIVE and to account for the unbracketed work named in
  // the registration (the auxiliary-loss sections, the permutation draws, the
  // per-minibatch index_select gathers and mb_nontrivial sync, the epoch
  // bookkeeping, and this instrument's own sync). A NEGATIVE residual would
  // mean a phase was double-counted and is a defect.
  out << ",\"total_attributed_s\":" << F17Finite(t.total_attributed_s, &nonfinite)
      << ",\"total_host_attributed_s\":"
      << F17Finite(t.total_host_attributed_s, &nonfinite)
      << ",\"rollout_hash_host_s\":"
      << F17Finite(t.rollout_hash_host_s, &nonfinite)
      << ",\"sync_s\":" << F17Finite(t.sync_s, &nonfinite)
      << ",\"ppo_elapsed_s\":" << F17Finite(ppo_elapsed_s, &nonfinite)
      << ",\"unattributed_host_s\":"
      << F17Finite(ppo_elapsed_s - t.total_host_attributed_s, &nonfinite)
      << ",\"bracket_count\":" << t.bracket_count
      << ",\"events_recorded\":" << t.events_recorded
      << ",\"minibatches\":" << stats.minibatches
      << ",\"nonfinite\":" << (nonfinite ? "true" : "false") << "}\n";

  if (!out.good()) {
    SpielFatalError("Failed to write phase-timing sidecar to: " + path);
  }
  out.close();
  if (out.fail()) {
    SpielFatalError("Failed to close phase-timing sidecar at: " + path);
  }
}

void WritePhaseTiming(const std::string& diagnostics_path, int update,
                      const PpoUpdateStats& stats, double ppo_elapsed_s,
                      const std::string& run_uuid,
                      const std::string& run_prefix) {
  if (!stats.phase_timings.enabled) return;
  const std::string path = PhaseTimingPath(diagnostics_path);
  if (path.empty()) return;
  WritePhaseTimingToPath(path, update, stats, ppo_elapsed_s, run_uuid, run_prefix);
}

void WriteDiagnostics(const std::string& filepath, int update, const PpoUpdateStats& stats,
                      double conflict_vp_generated, double conflict_vp_attributed, double conflict_vp_unattributed,
                      uint64_t seed, const std::string& run_uuid, const std::string& run_prefix, const std::string& config_fingerprint,
                      double raw_conflict_vp, double raw_noncombat_vp, double raw_total_vp,
                      double validation_kl, bool emit_canary_columns) {
  if (filepath.empty()) return;
  bool is_csv = (filepath.size() >= 4 && filepath.substr(filepath.size() - 4) == ".csv");

  if (is_csv) {
    std::string existing_header;
    const bool has_header = ReadDiagnosticsCsvHeader(filepath, &existing_header);
    const std::string expected_header = DiagnosticsCsvHeader(emit_canary_columns);
    if (has_header && existing_header != expected_header) {
      SpielFatalError(
          "Diagnostics CSV schema mismatch at: " + filepath +
          "\n  Existing header: " + existing_header +
          "\n  This binary writes: " + expected_header +
          "\n  Appending would produce rows the header cannot describe. Move "
          "the old file aside or point --diagnostics_path at a new file.");
    }
    std::ofstream ofs(filepath, std::ios::app);
    if (!ofs) {
      SpielFatalError("Could not open diagnostics path: " + filepath);
    }
    if (!has_header) {
      ofs << expected_header << "\n";
    }
    std::string epoch_kls_str;
    for (size_t i = 0; i < stats.epoch_kls.size(); ++i) {
      epoch_kls_str += std::to_string(stats.epoch_kls[i]);
      if (i + 1 < stats.epoch_kls.size()) epoch_kls_str += ";";
    }
    ofs << seed << ","
        << run_uuid << ","
        << run_prefix << ","
        << config_fingerprint << ","
        << update << ","
        << stats.rollout_hash << ","
        << (stats.episode_ids_unique ? 1 : 0) << ","
        << stats.policy_kl_before << ","
        << stats.return_min << ","
        << stats.return_max << ","
        << stats.return_p50 << ","
        << stats.return_p95 << ","
        << stats.return_p99 << ","
        << stats.abs_return_p99 << ","
        << stats.fraction_targets_outside_1 << ","
        << stats.fraction_critic_near_1 << ","
        << stats.total_transitions << ","
        << stats.nontrivial_transitions << ","
        << stats.forced_transitions << ","
        << epoch_kls_str << ","
        << conflict_vp_generated << ","
        << conflict_vp_attributed << ","
        << conflict_vp_unattributed << ","
        << raw_conflict_vp << ","
        << raw_noncombat_vp << ","
        << raw_total_vp << ","
        << validation_kl << ","
        << stats.aux_examples_used << ","
        << stats.aux_search_loss_coef << ","
        << stats.aux_ce << ","
        << stats.aux_value_mse << ","
        << stats.aux_grad_norm_mean << ","
        << stats.ppo_grad_norm_mean << ","
        << stats.aux_ppo_norm_ratio << ","
        << (stats.aux_ratio_abort ? 1 : 0) << ","
        // --- v3: durable PPO-side canaries (Phase A) ---
        << stats.entropy << ","
        << stats.approx_kl << ","
        << (stats.early_stopped ? 1 : 0) << ","
        << stats.kl_early_stop_epoch << ","
        << stats.GradNormMean() << ","
        << stats.grad_norm_max << ","
        << stats.value_loss << ","
        << (stats.nonfinite_abort ? 1 : 0) << ","
        << stats.precap_absz_all.decisions << ","
        << stats.precap_absz_all.p95 << ","
        << stats.precap_absz_all.p99 << ","
        << stats.precap_absz_all.max << ","
        << stats.precap_absz_all.frac_ge10 << ","
        << stats.precap_absz_primary.decisions << ","
        << stats.precap_absz_primary.p95 << ","
        << stats.precap_absz_primary.p99 << ","
        << stats.precap_absz_primary.max << ","
        << stats.precap_absz_primary.frac_ge10 << ","
        << stats.precap_absz_continuation.decisions << ","
        << stats.precap_absz_continuation.p95 << ","
        << stats.precap_absz_continuation.p99 << ","
        << stats.precap_absz_continuation.max << ","
        << stats.precap_absz_continuation.frac_ge10 << ","
        << stats.precap_absz_purchase.decisions << ","
        << stats.precap_absz_purchase.p95 << ","
        << stats.precap_absz_purchase.p99 << ","
        << stats.precap_absz_purchase.max << ","
        << stats.precap_absz_purchase.frac_ge10 << ","
        // --- v4: WO-1 Phase 5 scratch-anomaly bundle ---
        << stats.value_clip_fraction << ","
        << stats.explained_variance << ","
        << stats.PolicyHeadGradNormMean() << ","
        << stats.ValueHeadGradNormMean() << ","
        << stats.TrunkGradNormMean() << ","
        // Minibatches actually executed, i.e. BEFORE the KL early stop cut the
        // update short. Pairs with kl_early_stop/kl_early_stop_epoch: a short
        // count with early_stop=0 means something else ended the update.
        << stats.minibatches;
    if (emit_canary_columns) {
      // --- v5: PWO-5 section 14.1c, columns 70-86 in registered order. -----
      //
      // Round-trip precision, per section 13.5's rule as applied to CSV: a
      // normalized entropy of 3e-7 must not print as 0.000000. The stream's
      // default precision is 6 SIGNIFICANT digits (not 6 decimals), which
      // switches to scientific notation rather than flushing a small value to
      // zero -- but 6 significant digits still loses information, so the
      // canary columns are written at max_digits10 and restored afterwards.
      const std::streamsize prev_precision = ofs.precision(
          std::numeric_limits<double>::max_digits10);
      // Role order is the registered column order, which is the section 8.2
      // strategic set (primary, continuation, purchase, combat_intrigue,
      // other_optional) followed by leader_selection -- NOT the enum's
      // declaration order.
      const int kRoleColumnOrder[6] = {
          static_cast<int>(DuneDecisionRole::kAgentPrimary),
          static_cast<int>(DuneDecisionRole::kAgentContinuation),
          static_cast<int>(DuneDecisionRole::kPurchase),
          static_cast<int>(DuneDecisionRole::kCombatIntrigue),
          static_cast<int>(DuneDecisionRole::kOtherOptional),
          static_cast<int>(DuneDecisionRole::kLeaderSelection)};
      ofs << "," << stats.norm_entropy << "," << stats.norm_entropy_n;
      for (int i = 0; i < 6; ++i) {
        const int r = kRoleColumnOrder[i];
        ofs << "," << stats.norm_entropy_role[r]
            << "," << stats.norm_entropy_n_role[r];
      }
      ofs << "," << stats.max_action_prob
          << "," << stats.frac_legal_absz_ge_cap
          << "," << stats.frac_legal_absz_ge_cap_n;
      ofs.precision(prev_precision);
    }
    // --- v6 (WO-PERF-1): the KL-measurement denominator, cadenced mode only
    // (matching DiagnosticsCsvHeader's gate so header and rows never diverge).
    if (::absl::GetFlag(::FLAGS_diag_prepass_mode) == "cadenced") {
      ofs << "," << stats.measured_transitions;
    }
    if (::absl::GetFlag(::FLAGS_purchase_exploration)) {
      ofs << "," << stats.forced_buys;
    }
    if (::absl::GetFlag(::FLAGS_enable_study_readouts) ||
        ::absl::GetFlag(::FLAGS_advantage_filtering)) {
      ofs << "," << stats.single_action_transitions
          << "," << stats.single_action_fraction
          << "," << stats.single_action_value_mse_before
          << "," << stats.single_action_value_mse_after
          << "," << stats.single_action_sample_size
          << "," << stats.unfiltered_kl
          << "," << stats.unfiltered_kl_sample_size
          << "," << stats.retained_transitions
          << "," << stats.optimizer_minibatches;
    }
    if (::absl::GetFlag(::FLAGS_enable_ema_tracking)) {
      ofs << "," << stats.ema_magnet_kl;
    }
    ofs << "\n";
    ofs.flush();
    if (!ofs) {
      SpielFatalError("Failed to write diagnostics CSV data to: " + filepath);
    }
    ofs.close();
    if (!ofs) {
      SpielFatalError("Failed to close diagnostics CSV file at: " + filepath);
    }
  } else {
    // JSONL format
    std::ofstream ofs(filepath, std::ios::app);
    if (!ofs) {
      SpielFatalError("Could not open diagnostics path: " + filepath);
    }
    open_spiel::json::Object obj;
    obj["seed"] = static_cast<int64_t>(seed);
    obj["run_uuid"] = run_uuid;
    obj["run_prefix"] = run_prefix;
    obj["config_fingerprint"] = config_fingerprint;
    obj["update"] = update;
    obj["rollout_hash"] = stats.rollout_hash;
    obj["episode_ids_unique"] = stats.episode_ids_unique;
    obj["policy_kl_before"] = stats.policy_kl_before;
    obj["return_min"] = stats.return_min;
    obj["return_max"] = stats.return_max;
    obj["return_p50"] = stats.return_p50;
    obj["return_p95"] = stats.return_p95;
    obj["return_p99"] = stats.return_p99;
    obj["abs_return_p99"] = stats.abs_return_p99;
    obj["fraction_targets_outside_1"] = stats.fraction_targets_outside_1;
    obj["fraction_critic_near_1"] = stats.fraction_critic_near_1;
    obj["total_transitions"] = static_cast<int64_t>(stats.total_transitions);
    obj["nontrivial_transitions"] = static_cast<int64_t>(stats.nontrivial_transitions);
    obj["forced_transitions"] = static_cast<int64_t>(stats.forced_transitions);

    open_spiel::json::Array kls_arr;
    for (double kl : stats.epoch_kls) kls_arr.push_back(kl);
    obj["epoch_kls"] = kls_arr;

    obj["conflict_vp_generated"] = conflict_vp_generated;
    obj["conflict_vp_attributed"] = conflict_vp_attributed;
    obj["conflict_vp_unattributed"] = conflict_vp_unattributed;
    obj["raw_conflict_vp"] = raw_conflict_vp;
    obj["raw_noncombat_vp"] = raw_noncombat_vp;
    obj["raw_total_vp"] = raw_total_vp;
    obj["validation_kl"] = validation_kl;

    // Phase 18B auxiliary-loss trajectory (WO-17): same names as the CSV
    // columns, so the two formats stay analysable interchangeably.
    obj["aux_examples_used"] = static_cast<int64_t>(stats.aux_examples_used);
    obj["aux_search_loss_coef"] = stats.aux_search_loss_coef;
    obj["aux_ce"] = stats.aux_ce;
    obj["aux_value_mse"] = stats.aux_value_mse;
    obj["aux_grad_norm_mean"] = stats.aux_grad_norm_mean;
    obj["ppo_grad_norm_mean"] = stats.ppo_grad_norm_mean;
    obj["aux_ppo_norm_ratio"] = stats.aux_ppo_norm_ratio;
    obj["aux_ratio_abort"] = stats.aux_ratio_abort;
    obj["aux_finish_pass_steps"] = stats.aux_finish_pass_steps;
    obj["aux_finish_pass_rows"] = stats.aux_finish_pass_rows;

    // Durable PPO-side canaries (Phase A): same names as the CSV columns, so
    // the two formats stay analysable interchangeably.
    obj["entropy"] = stats.entropy;
    obj["approx_kl"] = stats.approx_kl;
    obj["kl_early_stop"] = stats.early_stopped;
    obj["kl_early_stop_epoch"] = stats.kl_early_stop_epoch;
    obj["grad_norm_mean"] = stats.GradNormMean();
    obj["grad_norm_max"] = stats.grad_norm_max;
    obj["value_loss"] = stats.value_loss;
    obj["nonfinite_abort"] = stats.nonfinite_abort;
    // WO-1 Phase 5: same names as the CSV columns, so the two formats stay
    // analysable interchangeably.
    obj["value_clip_fraction"] = stats.value_clip_fraction;
    obj["explained_variance"] = stats.explained_variance;
    obj["policy_head_grad_norm"] = stats.PolicyHeadGradNormMean();
    obj["value_head_grad_norm"] = stats.ValueHeadGradNormMean();
    obj["trunk_grad_norm"] = stats.TrunkGradNormMean();
    obj["minibatches_executed"] = stats.minibatches;
    const auto add_absz = [&obj](const std::string& suffix,
                                 const PrecapAbszStats& z) {
      obj["precap_absz_n" + suffix] = static_cast<int64_t>(z.decisions);
      obj["precap_absz_p95" + suffix] = z.p95;
      obj["precap_absz_p99" + suffix] = z.p99;
      obj["precap_absz_max" + suffix] = z.max;
      obj["frac_decisions_absz_ge10" + suffix] = z.frac_ge10;
    };
    add_absz("", stats.precap_absz_all);
    add_absz("_primary", stats.precap_absz_primary);
    add_absz("_continuation", stats.precap_absz_continuation);
    add_absz("_purchase", stats.precap_absz_purchase);

    if (emit_canary_columns) {
      const char* const kRoleSuffixes[6] = {
          "_primary", "_continuation", "_purchase",
          "_combat_intrigue", "_other_optional", "_leader_selection"};
      const int kRoleColumnOrder[6] = {
          static_cast<int>(DuneDecisionRole::kAgentPrimary),
          static_cast<int>(DuneDecisionRole::kAgentContinuation),
          static_cast<int>(DuneDecisionRole::kPurchase),
          static_cast<int>(DuneDecisionRole::kCombatIntrigue),
          static_cast<int>(DuneDecisionRole::kOtherOptional),
          static_cast<int>(DuneDecisionRole::kLeaderSelection)};
      obj["norm_entropy"] = stats.norm_entropy;
      obj["norm_entropy_n"] = static_cast<int64_t>(stats.norm_entropy_n);
      for (int i = 0; i < 6; ++i) {
        const int r = kRoleColumnOrder[i];
        obj[std::string("norm_entropy") + kRoleSuffixes[i]] = stats.norm_entropy_role[r];
        obj[std::string("norm_entropy_n") + kRoleSuffixes[i]] =
            static_cast<int64_t>(stats.norm_entropy_n_role[r]);
      }
      obj["max_action_prob"] = stats.max_action_prob;
      obj["frac_legal_absz_ge_cap"] = stats.frac_legal_absz_ge_cap;
      obj["frac_legal_absz_ge_cap_n"] = static_cast<int64_t>(stats.frac_legal_absz_ge_cap_n);
    }

    // --- v6 (WO-PERF-1): same name and the same cadenced-mode gate as the
    // CSV column, so the two formats stay analysable interchangeably.
    if (::absl::GetFlag(::FLAGS_diag_prepass_mode) == "cadenced") {
      obj["measured_transitions"] =
          static_cast<int64_t>(stats.measured_transitions);
    }
    if (::absl::GetFlag(::FLAGS_purchase_exploration)) {
      obj["forced_buys"] = static_cast<int64_t>(stats.forced_buys);
    }
    if (::absl::GetFlag(::FLAGS_enable_study_readouts) ||
        ::absl::GetFlag(::FLAGS_advantage_filtering)) {
      obj["single_action_transitions"] = static_cast<int64_t>(stats.single_action_transitions);
      obj["single_action_fraction"] = stats.single_action_fraction;
      obj["single_action_value_mse_before"] = stats.single_action_value_mse_before;
      obj["single_action_value_mse_after"] = stats.single_action_value_mse_after;
      obj["single_action_sample_size"] = static_cast<int64_t>(stats.single_action_sample_size);
      obj["unfiltered_kl"] = stats.unfiltered_kl;
      obj["unfiltered_kl_sample_size"] = static_cast<int64_t>(stats.unfiltered_kl_sample_size);
      obj["retained_transitions"] = static_cast<int64_t>(stats.retained_transitions);
      obj["optimizer_minibatches"] = static_cast<int64_t>(stats.optimizer_minibatches);
    }
    if (::absl::GetFlag(::FLAGS_enable_ema_tracking)) {
      obj["ema_magnet_kl"] = stats.ema_magnet_kl;
      obj["ema_magnet_kl_post"] = stats.ema_magnet_kl_post;
    }

    ofs << open_spiel::json::ToString(obj, false) << "\n";
    ofs.flush();
    if (!ofs) {
      SpielFatalError("Failed to write diagnostics JSONL data to: " + filepath);
    }
    ofs.close();
    if (!ofs) {
      SpielFatalError("Failed to close diagnostics JSONL file at: " + filepath);
    }
  }
}

}  // namespace open_spiel

#endif  // OPEN_SPIEL_BUILD_WITH_LIBTORCH
