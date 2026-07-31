// Reproducible population evaluator for Dune Imperium.
//
// Replaces the ad-hoc dune_eval_1000.cc with deterministic, domain-separated
// seeding via dune_seed_utils.h. Produces per-game JSONL and aggregate JSON
// with Wilson confidence intervals.
//
// Key properties:
//   - Per-game seeds derived from (base_seed, domain, episode_id, stream)
//   - Deterministic seat rotation and opponent assignment
//   - Heterogeneous opponent architectures supported
//   - Greedy / stochastic policy selection
//   - Thread-count reproducibility (results identical regardless of --threads)

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <memory>
#include <mutex>
#include <optional>
#include <random>
#include <shared_mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>
#include <unordered_map>

#include "open_spiel/spiel.h"
#include "open_spiel/games/dune_imperium/dune_imperium.h"
#include "open_spiel/abseil-cpp/absl/flags/flag.h"
#include "open_spiel/abseil-cpp/absl/flags/parse.h"
#include "open_spiel/abseil-cpp/absl/strings/str_format.h"
#include <torch/torch.h>

#include "dune_network.h"
#include "dune_seed_utils.h"
#include "dune_eval_action_selection.h"
#include "dune_specimen_conversion.h"
#include "open_spiel/utils/json.h"

// ---------------------------------------------------------------------------
// Flags
// ---------------------------------------------------------------------------
ABSL_FLAG(std::string, model_checkpoint, "",
          "Path to the model checkpoint to evaluate.");
ABSL_FLAG(std::string, opponent_checkpoints, "",
          "Comma-separated paths to opponent model checkpoints. "
          "Empty or \"random\" for random opponents.");
ABSL_FLAG(int, num_games, 5000,
          "Total number of evaluation games to play.");
ABSL_FLAG(uint64_t, base_seed, 42,
          "Base seed for deterministic evaluation.");
ABSL_FLAG(std::string, domain, "EVAL_BASELINE",
          "Evaluation domain: EVAL_BASELINE, EVAL_DEV, or EVAL_FINAL.");
ABSL_FLAG(bool, greedy, true,
          "If true, use argmax policy; if false, sample from softmax.");
ABSL_FLAG(int, threads, 0,
          "Number of game worker threads. 0 = auto-detect.");
ABSL_FLAG(int, hidden_dim, 2048,
          "Hidden dimension of the evaluated model.");
ABSL_FLAG(int, num_blocks, 8,
          "Number of residual blocks in the evaluated model.");
ABSL_FLAG(int, opp_hidden_dim, -1,
          "Opponent hidden dim. -1 = same as --hidden_dim.");
ABSL_FLAG(int, opp_num_blocks, -1,
          "Opponent num blocks. -1 = same as --num_blocks.");
ABSL_FLAG(std::string, output_dir, "",
          "Directory for per-game JSONL and aggregate JSON. "
          "Empty = stdout only.");
ABSL_FLAG(float, temperature, 1.0f,
          "Softmax temperature for stochastic policy (--greedy=false).");
// PWO-5 gate 2 item (d). OPTIONAL flags: unset (the default) means INHERIT the
// global --greedy / --temperature, which is exactly the pre-gate-2 behaviour for
// every seat. A plain bool defaulting to true would NOT be backward compatible,
// because a run passing --greedy=false would then keep greedy opponents.
ABSL_FLAG(std::optional<bool>, opponent_greedy, std::nullopt,
          "Action selection for OPPONENT seats only. Unset (default) = inherit "
          "--greedy, i.e. identical to pre-existing behaviour. Set true to hold "
          "opponents at argmax while the candidate samples -- required to "
          "express the frozen strong population (raw prior, GREEDY) against a "
          "candidate sampled at temperature 1.0.");
ABSL_FLAG(std::optional<float>, opponent_temperature, std::nullopt,
          "Softmax temperature for OPPONENT seats only. Unset (default) = "
          "inherit --temperature. Inert when opponents are greedy.");
ABSL_FLAG(bool, deterministic_eval, true,
          "If true, use batch-1 mutex-serialized inference for strict bitwise "
          "thread-count reproducibility. Much slower than batched mode.");
ABSL_FLAG(bool, nonlinear_value_head, false,
          "Use the versioned nonlinear value head for the evaluated model.");
ABSL_FLAG(bool, opponent_nonlinear_value_head, false,
          "Use the versioned nonlinear value head for opponent checkpoints.");
ABSL_FLAG(double, candidate_logit_cap, 10.0,
          "Cap on legal-centered logits for the CANDIDATE (evaluated) model, "
          "applied by CenterAndCapLegalLogits. Default 10.0 preserves prior "
          "behavior (formerly the hard-coded kEvalLogitCap).");
ABSL_FLAG(double, opponent_logit_cap, 10.0,
          "Cap on legal-centered logits for OPPONENT models. Default 10.0 "
          "preserves prior behavior (formerly the hard-coded kEvalLogitCap).");
ABSL_FLAG(std::string, dump_logit_stats, "",
          "If non-empty, append one CSV row per CANDIDATE decision with pre-cap "
          "legal-centered logit statistics to this path (audit; off by default).");

namespace open_spiel {
namespace {

using dune_imperium::DuneImperiumState;
using dune_imperium::kNumPlayers;
using dune_imperium::kMaxRounds;
using dune_imperium::kTargetVp;

// ---------------------------------------------------------------------------
// Domain resolution
// ---------------------------------------------------------------------------
uint64_t ResolveDomain(const std::string& domain_str) {
  if (domain_str == "EVAL_BASELINE") return dune_seed::kDomainEvalBaseline;
  if (domain_str == "EVAL_DEV")      return dune_seed::kDomainEvalDev;
  if (domain_str == "EVAL_FINAL")    return dune_seed::kDomainEvalFinal;
  std::cerr << "Unknown domain: " << domain_str
            << ". Must be EVAL_BASELINE, EVAL_DEV, or EVAL_FINAL.\n";
  std::exit(1);
}

// ---------------------------------------------------------------------------
// Wilson score interval
// ---------------------------------------------------------------------------
struct WilsonCI {
  double point;
  double lower;
  double upper;
};

WilsonCI WilsonScore(int successes, int trials, double z = 1.96) {
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
// Per-game result
// ---------------------------------------------------------------------------
struct GameResult {
  int episode_id;
  int seat;
  std::vector<std::string> opponents;  // Exact size 3
  int placement;                       // 1-based (mapped directly from returns)
  double game_return;
  int ending_round;
  int current_vp;
  int final_scored_vp;

  // PWO-5 section 13.5. The registered VP-margin estimand is
  //
  //     vp_margin = candidate's TRUE final VP - MEAN of the three opponents'
  //
  // -- true final VP, not track VP; the mean of three, not the best. Before
  // this, popeval emitted only the CANDIDATE's `final_scored_vp`, so the
  // quantity section 15.2 gate 2 asks for could not be computed from its
  // output at all; a reconstruction from `vp_end_by_round` computes a
  // different thing.
  //
  // `FinalScoredVp` is the ENGINE's function. It is NOT `GetTrueFinalVp`, the
  // REPORTING helper that drops the tech-tile-8 guard and disagrees with the
  // engine on 212 of 1,600 seat values -- the defect that became the PWO-5
  // gate-3 STOP. Every seat here goes through the engine.
  std::array<int, kNumPlayers> final_scored_vp_all{};
  // Exactly representable only as a rational with denominator 3, so it is
  // carried as a double and SERIALIZED AT ROUND-TRIP PRECISION (%.17g). It is
  // never routed through open_spiel's JSON writer, which emits doubles as %f
  // at six decimal places and would silently floor anything below 5e-7 to 0.0.
  double vp_margin = 0.0;
  bool vp_margin_valid = false;
  // --- Phase-3 pace/threshold telemetry (new keys only) ---
  // vp_end_rN[p] = player p's running (track) VP at the end of round N.
  // has_rN=false => the game ended before round N completed (emitted as null).
  std::array<int, kNumPlayers> vp_end_r5{};
  std::array<int, kNumPlayers> vp_end_r6{};
  std::array<int, kNumPlayers> vp_end_r7{};
  bool has_r5 = false;
  bool has_r6 = false;
  bool has_r7 = false;
  int end_trigger_player = -1;  // -1 => null
  int end_trigger_round = -1;   // -1 => null
  int winner_triggered = -1;    // -1 => null, 0 => false, 1 => true

  // --- WO-1 Phase 3 pace/tempo telemetry (new keys only) ---
  // Track VP at the end of EVERY round, 1..terminal. Supersedes the r5/r6/r7
  // triple above, which stays for backward compatibility. vp_end_by_round[N-1]
  // is round N; only the first `rounds_captured` entries are meaningful.
  std::array<std::array<int, kNumPlayers>, kMaxRounds> vp_end_by_round{};
  int rounds_captured = 0;
  // First round at whose END this seat's track VP was >= kTargetVp; -1 => null.
  std::array<int, kNumPlayers> first_round_vp_ge_11{};
  // Every seat at track VP >= kTargetVp in the terminal round. May hold more
  // than one seat: crossing is not exclusive and no causal trigger is assigned.
  std::vector<int> terminal_threshold_set;
  bool candidate_in_terminal_threshold_set = false;
  // "threshold" | "round_limit" | "other". Derived from the same accessors the
  // engine's own end test uses (dune_imperium.cc:8891-8904): end_game is set
  // when round_ > kMaxRounds OR any vp_[p] >= kTargetVp. "threshold" wins when
  // both hold, matching end_trigger_player's existing precedence. "other" is
  // unreachable by construction and is emitted as a tripwire.
  std::string terminal_reason = "other";
  // Applied ConvertSpecimenToTroop actions per seat over the whole game.
  std::array<int, kNumPlayers> specimen_conversions{};
};

// ---------------------------------------------------------------------------
// Phase-3 telemetry JSON helpers
// ---------------------------------------------------------------------------
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

// ---------------------------------------------------------------------------
// Phase-4.2: optional per-candidate-decision logit-stat dump (CSV)
// ---------------------------------------------------------------------------
std::mutex g_logit_dump_mutex;

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
// Comma-separated path splitting
// ---------------------------------------------------------------------------
std::vector<std::string> SplitCommaSeparated(const std::string& value) {
  std::vector<std::string> items;
  std::stringstream ss(value);
  std::string item;
  while (std::getline(ss, item, ',')) {
    // Trim whitespace
    item.erase(item.begin(), std::find_if(item.begin(), item.end(),
               [](unsigned char ch) { return !std::isspace(ch); }));
    item.erase(std::find_if(item.rbegin(), item.rend(),
               [](unsigned char ch) { return !std::isspace(ch); }).base(),
               item.end());
    if (!item.empty()) items.push_back(item);
  }
  return items;
}



// ---------------------------------------------------------------------------
// Auto-detect model hidden_dim and num_blocks from JSON manifest or weight keys
// ---------------------------------------------------------------------------
bool DetectModelDimensions(const std::string& model_path, int* hidden_dim, int* num_blocks) {
  // 1. Try to find a JSON file
  std::string json_path = "";
  std::filesystem::path p(model_path);
  std::filesystem::path p_json = p;
  p_json.replace_extension(".json");
  if (std::filesystem::exists(p_json)) {
    json_path = p_json.string();
  } else {
    std::filesystem::path p_parent_manifest = p.parent_path() / "manifest.json";
    if (std::filesystem::exists(p_parent_manifest)) {
      json_path = p_parent_manifest.string();
    }
  }

  if (!json_path.empty()) {
    try {
      std::ifstream f(json_path);
      if (f) {
        std::string content((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
        auto val_opt = json::FromString(content);
        if (val_opt.has_value() && val_opt->IsObject()) {
          const auto& obj = val_opt->GetObject();
          auto it_hd = obj.find("hidden_dim");
          auto it_nb = obj.find("num_blocks");
          if (it_hd != obj.end() && it_hd->second.IsInt() &&
              it_nb != obj.end() && it_nb->second.IsInt()) {
            *hidden_dim = it_hd->second.GetInt();
            *num_blocks = it_nb->second.GetInt();
            return true;
          }
          auto it_arch = obj.find("architecture");
          if (it_arch != obj.end() && it_arch->second.IsObject()) {
            const auto& arch_obj = it_arch->second.GetObject();
            auto it_arch_hd = arch_obj.find("hidden_dim");
            auto it_arch_nb = arch_obj.find("num_blocks");
            if (it_arch_hd != arch_obj.end() && it_arch_hd->second.IsInt() &&
                it_arch_nb != arch_obj.end() && it_arch_nb->second.IsInt()) {
              *hidden_dim = it_arch_hd->second.GetInt();
              *num_blocks = it_arch_nb->second.GetInt();
              return true;
            }
          }
        }
      }
    } catch (...) {
      // Fallback
    }
  }

  // 2. Fall back to weights key-based inspection
  try {
    torch::serialize::InputArchive archive;
    archive.load_from(model_path, torch::kCPU);

    torch::serialize::InputArchive input_layer_archive;
    archive.read("input_layer", input_layer_archive);
    torch::Tensor weight;
    input_layer_archive.read("weight", weight);
    *hidden_dim = weight.size(0);

    int blocks = 0;
    while (true) {
      torch::serialize::InputArchive res_archive;
      std::string block_name = "res" + std::to_string(blocks + 1);
      try {
        archive.read(block_name, res_archive);
        blocks++;
      } catch (...) {
        break;
      }
    }
    *num_blocks = blocks;

    std::cerr << "WARNING: Manifest JSON not found for model checkpoint " << model_path
              << ". Auto-detected architecture (hidden_dim=" << *hidden_dim
              << ", num_blocks=" << *num_blocks << ") from weight keys. "
              << "Note: this inspection logic is coupled to the SharedDunePolicyValueNetImpl class architecture.\n";
    return true;
  } catch (const std::exception& e) {
    std::cerr << "ERROR: Failed to detect model dimensions or load archive from " << model_path
              << ": " << e.what() << "\n";
    return false;
  }
}

// ---------------------------------------------------------------------------
// Worker thread: plays games with deterministic per-game seeds.
// Results are stored in a pre-allocated vector indexed by episode_id,
// guaranteeing thread-count-independent output.
// ---------------------------------------------------------------------------
void WorkerThread(
    int /*thread_id*/,
    std::shared_ptr<const Game> game,
    std::shared_ptr<IGameEvaluator> model_evaluator,
    const std::vector<std::shared_ptr<IGameEvaluator>>& opponent_evaluators,
    const std::vector<std::string>& opponent_names,
    int64_t obs_size,
    bool provides_info_state_tensor,
    bool provides_observations_tensor,
    std::atomic<int>& next_game_id,
    int total_games,
    uint64_t base_seed,
    uint64_t domain,
    bool greedy,
    float temperature,
    // PWO-5 gate 2 item (d). `has_*` false = inherit the global value above.
    bool has_opponent_greedy,
    bool opponent_greedy,
    bool has_opponent_temperature,
    float opponent_temperature,
    float candidate_logit_cap,
    float opponent_logit_cap,
    std::ofstream* dump_out,
    std::vector<GameResult>& results) {

  // Pre-allocate observation buffer reused across all games
  std::vector<float> obs(obs_size, 0.0f);

  while (true) {
    int episode_id = next_game_id++;
    if (episode_id >= total_games) break;

    // Log progress every 100 completed games
    int completed = next_game_id.load();
    if (completed % 100 == 0) {
      static std::mutex progress_mutex;
      static int last_printed = -1;
      std::lock_guard<std::mutex> lock(progress_mutex);
      if (completed > last_printed) {
        std::cout << "Progress: " << completed << " / " << total_games << " games completed..." << std::endl;
        last_printed = completed;
      }
    }

    // --- Round-robin seat assignment for exact balance ---
    int model_player = episode_id % kNumPlayers;

    // --- Per-game chance RNG ---
    uint64_t chance_seed = dune_seed::DeriveSeed(
        base_seed, domain, static_cast<uint64_t>(episode_id),
        dune_seed::kStreamChance);
    std::mt19937 chance_rng = dune_seed::MakeRng32(chance_seed);

    // --- Per-player policy RNGs (for stochastic mode) ---
    std::array<std::mt19937_64, 4> policy_rngs;
    for (int p = 0; p < kNumPlayers; ++p) {
      uint64_t policy_stream = dune_seed::kStreamPolicyPlayer0 +
                               static_cast<uint64_t>(p);
      uint64_t pseed = dune_seed::DeriveSeed(
          base_seed, domain, static_cast<uint64_t>(episode_id),
          policy_stream);
      policy_rngs[p] = dune_seed::MakeRng64(pseed);
    }

    // --- Per-game opponent assignment RNG ---
    std::vector<size_t> player_opp_idx(kNumPlayers, 0);
    if (!opponent_evaluators.empty()) {
      uint64_t opp_assign_seed = dune_seed::DeriveSeed(
          base_seed, domain, static_cast<uint64_t>(episode_id),
          dune_seed::kStreamOpponentAssign);
      std::mt19937_64 opp_rng = dune_seed::MakeRng64(opp_assign_seed);
      std::uniform_int_distribution<size_t> opp_dist(0, opponent_evaluators.size() - 1);

      for (int p = 0; p < kNumPlayers; ++p) {
        if (p == model_player) continue;
        player_opp_idx[p] = opp_dist(opp_rng);
      }
    }

    std::unique_ptr<State> state = game->NewInitialState();
    int game_length = 0;

    // Phase-3: observe running VP at each round boundary via public accessors.
    const DuneImperiumState* dune_state =
        dynamic_cast<const DuneImperiumState*>(state.get());
    std::array<std::array<int, kNumPlayers>, kMaxRounds + 2> vp_at_round_end{};
    std::array<bool, kMaxRounds + 2> round_end_seen{};
    std::array<int, kNumPlayers> specimen_conversions{};
    int last_round = dune_state ? dune_state->GetCurrentRound() : -1;
    auto snapshot_round_ends = [&]() {
      if (!dune_state) return;
      int cur = dune_state->GetCurrentRound();
      while (last_round >= 0 && last_round < cur) {
        if (last_round >= 1 && last_round <= kMaxRounds) {
          for (int p = 0; p < kNumPlayers; ++p) {
            vp_at_round_end[last_round][p] = dune_state->GetPlayerVp(p);
          }
          round_end_seen[last_round] = true;
        }
        ++last_round;
      }
    };

    while (!state->IsTerminal()) {
      snapshot_round_ends();
      ++game_length;
      if (game_length > 5000) {
        std::cerr << "Possible infinite loop in episode " << episode_id
                  << "! Length: " << game_length
                  << " Player: " << state->CurrentPlayer() << std::endl;
        std::abort();
      }

      if (state->IsChanceNode()) {
        auto outcomes = state->ChanceOutcomes();
        Action action;
        if (game->GetType().chance_mode ==
            GameType::ChanceMode::kSampledStochastic) {
          action = outcomes.front().first;
        } else {
          action = SampleAction(outcomes, chance_rng).first;
        }
        state->ApplyAction(action);
        continue;
      }

      Player current_player = state->CurrentPlayer();
      std::vector<Action> legal_actions = state->LegalActions();
      if (legal_actions.empty()) {
        std::cerr << "Empty legal actions in episode " << episode_id << "!\n";
        break;
      }

      Action chosen_action = -1;
      bool use_model = (current_player == model_player);
      bool use_opponent_model = (!use_model && !opponent_evaluators.empty());

      if (use_model || use_opponent_model) {
        // Fill observation buffer
        std::fill(obs.begin(), obs.end(), 0.0f);
        if (provides_info_state_tensor) {
          state->InformationStateTensor(current_player, absl::MakeSpan(obs));
        } else if (provides_observations_tensor) {
          state->ObservationTensor(current_player, absl::MakeSpan(obs));
        }

        // Select evaluator
        std::shared_ptr<IGameEvaluator> evaluator = model_evaluator;
        if (!use_model) {
          size_t idx = player_opp_idx[current_player];
          evaluator = opponent_evaluators[idx];
        }

        EvalResult result = evaluator->Evaluate(obs);
        if (use_model && dump_out != nullptr && dune_state != nullptr) {
          AppendLogitStatsRow(*dump_out, episode_id,
                              dune_state->GetCurrentRound(),
                              GamePhaseRole(dune_state->phase()),
                              legal_actions, result.logits,
                              static_cast<double>(candidate_logit_cap));
        }
        const float logit_cap =
            use_model ? candidate_logit_cap : opponent_logit_cap;
        CenterAndCapLegalLogits(result.logits, legal_actions, logit_cap);

        // PWO-5 gate 2 item (d): the candidate seat and the opponent seats can
        // now select differently. With the overrides unset this resolves to the
        // global (greedy, temperature) for BOTH, which is the pre-gate-2
        // behaviour bit for bit -- see dune_eval_action_selection.h.
        const dune_eval::SelectionPolicy policy =
            dune_eval::ResolveSelectionPolicy(
                use_model, greedy, temperature, has_opponent_greedy,
                opponent_greedy, has_opponent_temperature,
                opponent_temperature);
        chosen_action = dune_eval::SelectActionFromLogits(
            result.logits, legal_actions, policy,
            policy_rngs[current_player]);
      } else {
        // Random opponent (no model loaded)
        std::uniform_int_distribution<size_t> dist(
            0, legal_actions.size() - 1);
        chosen_action = legal_actions[dist(policy_rngs[current_player])];
      }

      // WO-1 Phase 3 addendum: count applied specimen->troop conversions per
      // seat, at application time, by engine action-ID range. Measurement only
      // -- no shaping is enabled anywhere by this counter.
      //
      // PWO-5 gate 2 item (c): the range is 741-752, not 740-752. This site was
      // NOT named in the registration's known list (which named the trainer's
      // guard and dune_search_benchmark's counter); the registered "sweep for
      // other 740-752 sites" found it. Numerically a no-op -- 740 is an unused
      // base constant and is never legal -- so the committed PWO-2 popeval
      // conversion baselines (4.730 / 5.232 / 4.165 candidate-seat) do not move.
      if (dune_shaping::IsSpecimenConversionAction(chosen_action) &&
          current_player >= 0 && current_player < kNumPlayers) {
        ++specimen_conversions[current_player];
      }
      state->ApplyAction(chosen_action);
    }

    // Phase-3: capture the final (terminal) round's VP snapshot.
    snapshot_round_ends();

    // --- Collect results ---
    std::vector<double> returns = state->Returns();

    // Map returns cleanly directly to placement:
    // 2.25 -> 1st, 0.25 -> 2nd, -0.75 -> 3rd, -1.75 -> 4th
    int placement = 4;
    double r = returns[model_player];
    if (std::abs(r - 2.25) < 1e-4) {
      placement = 1;
    } else if (std::abs(r - 0.25) < 1e-4) {
      placement = 2;
    } else if (std::abs(r - (-0.75)) < 1e-4) {
      placement = 3;
    } else if (std::abs(r - (-1.75)) < 1e-4) {
      placement = 4;
    }

    GameResult& gr = results[episode_id];
    gr.episode_id = episode_id;
    gr.seat = model_player;

    // Track opponent names for seats other than model_player
    gr.opponents.clear();
    for (int p = 0; p < kNumPlayers; ++p) {
      if (p == model_player) continue;
      if (opponent_names.empty()) {
        gr.opponents.push_back("random");
      } else {
        size_t opp_idx = player_opp_idx[p];
        gr.opponents.push_back(opponent_names[opp_idx]);
      }
    }

    gr.placement = placement;
    gr.game_return = returns[model_player];
    if (dune_state) {
      gr.ending_round = dune_state->IsTerminal() ? (dune_state->GetCurrentRound() - 1)
                                                 : dune_state->GetCurrentRound();
    } else {
      gr.ending_round = -1;
    }
    gr.current_vp = dune_state
                    ? dune_state->GetPlayerVpForTesting(model_player)
                    : -1;
    gr.final_scored_vp = dune_state
                         ? dune_state->FinalScoredVp(model_player)
                         : -1;

    // PWO-5 section 13.5: every seat's TRUE final VP, and the exact margin.
    if (dune_state) {
      long long opponent_sum = 0;
      for (int p = 0; p < kNumPlayers; ++p) {
        gr.final_scored_vp_all[p] = dune_state->FinalScoredVp(p);
        if (p != model_player) opponent_sum += gr.final_scored_vp_all[p];
      }
      // The mean of the THREE opponents, not the best of them.
      gr.vp_margin = static_cast<double>(gr.final_scored_vp_all[model_player]) -
                     static_cast<double>(opponent_sum) /
                         static_cast<double>(kNumPlayers - 1);
      gr.vp_margin_valid = true;
    } else {
      gr.final_scored_vp_all.fill(-1);
      gr.vp_margin = 0.0;
      gr.vp_margin_valid = false;
    }

    // --- Phase-3 pace/threshold telemetry ---
    gr.has_r5 = round_end_seen[5];
    gr.has_r6 = round_end_seen[6];
    gr.has_r7 = round_end_seen[7];
    if (gr.has_r5) gr.vp_end_r5 = vp_at_round_end[5];
    if (gr.has_r6) gr.vp_end_r6 = vp_at_round_end[6];
    if (gr.has_r7) gr.vp_end_r7 = vp_at_round_end[7];
    gr.end_trigger_player = -1;
    gr.end_trigger_round = -1;
    if (dune_state) {
      // Game end is checked only at round boundaries: at the terminal round-end
      // a player triggered the end iff running VP >= kTargetVp. Among crossers
      // pick the leader (max VP; tie -> lowest seat).
      int best_p = -1, best_vp = -1;
      for (int p = 0; p < kNumPlayers; ++p) {
        int v = dune_state->GetPlayerVp(p);
        if (v >= kTargetVp && v > best_vp) {
          best_vp = v;
          best_p = p;
        }
      }
      if (best_p >= 0) {
        gr.end_trigger_player = best_p;
        gr.end_trigger_round = gr.ending_round;
      }
    }
    // First place = argmax of placement returns (tie -> lowest seat).
    int first_place = 0;
    for (int p = 1; p < kNumPlayers; ++p) {
      if (returns[p] > returns[first_place]) first_place = p;
    }
    gr.winner_triggered = (gr.end_trigger_player >= 0)
                              ? (first_place == gr.end_trigger_player ? 1 : 0)
                              : -1;

    // --- WO-1 Phase 3 pace/tempo telemetry ---------------------------------
    // Dense round-end VP, replacing the r5/r6/r7 ceiling. The snapshot hook
    // already captured rounds 1..kMaxRounds; only the emission was truncated.
    gr.rounds_captured = 0;
    for (int rd = 1; rd <= kMaxRounds; ++rd) {
      if (!round_end_seen[rd]) break;
      gr.vp_end_by_round[rd - 1] = vp_at_round_end[rd];
      gr.rounds_captured = rd;
    }
    for (int p = 0; p < kNumPlayers; ++p) {
      gr.first_round_vp_ge_11[p] = -1;
      for (int rd = 1; rd <= gr.rounds_captured; ++rd) {
        if (vp_at_round_end[rd][p] >= kTargetVp) {
          gr.first_round_vp_ge_11[p] = rd;
          break;
        }
      }
    }
    gr.specimen_conversions = specimen_conversions;
    gr.terminal_threshold_set.clear();
    gr.candidate_in_terminal_threshold_set = false;
    gr.terminal_reason = "other";
    if (dune_state) {
      for (int p = 0; p < kNumPlayers; ++p) {
        if (dune_state->GetPlayerVp(p) >= kTargetVp) {
          gr.terminal_threshold_set.push_back(p);
          if (p == model_player) gr.candidate_in_terminal_threshold_set = true;
        }
      }
      // Mirrors the engine's own end test (dune_imperium.cc:8891-8904):
      // end_game <- (round_ > kMaxRounds) OR (any vp_[p] >= kTargetVp).
      // Threshold takes precedence when both hold, matching the precedence
      // end_trigger_player already uses. "other" is unreachable; if it ever
      // appears, the engine's end condition changed under a frozen engine.
      if (!gr.terminal_threshold_set.empty()) {
        gr.terminal_reason = "threshold";
      } else if (dune_state->GetCurrentRound() > kMaxRounds) {
        gr.terminal_reason = "round_limit";
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Main evaluation driver
// ---------------------------------------------------------------------------
void RunEvaluation() {
  const std::string model_checkpoint = absl::GetFlag(FLAGS_model_checkpoint);
  const std::string opponent_str = absl::GetFlag(FLAGS_opponent_checkpoints);
  const int total_games = absl::GetFlag(FLAGS_num_games);
  const uint64_t base_seed = absl::GetFlag(FLAGS_base_seed);
  const std::string domain_str = absl::GetFlag(FLAGS_domain);
  const bool greedy = absl::GetFlag(FLAGS_greedy);
  const int requested_threads = absl::GetFlag(FLAGS_threads);
  const int hidden_dim = absl::GetFlag(FLAGS_hidden_dim);
  const int num_blocks = absl::GetFlag(FLAGS_num_blocks);
  int opp_hidden_dim = absl::GetFlag(FLAGS_opp_hidden_dim);
  int opp_num_blocks = absl::GetFlag(FLAGS_opp_num_blocks);
  const std::string output_dir = absl::GetFlag(FLAGS_output_dir);
  const float temperature = absl::GetFlag(FLAGS_temperature);
  // PWO-5 gate 2 item (d). Unset => inherit the global value; the resolution
  // itself lives in dune_eval::ResolveSelectionPolicy so it is unit-testable.
  const std::optional<bool> opponent_greedy_opt =
      absl::GetFlag(FLAGS_opponent_greedy);
  const std::optional<float> opponent_temperature_opt =
      absl::GetFlag(FLAGS_opponent_temperature);
  const bool has_opponent_greedy = opponent_greedy_opt.has_value();
  const bool opponent_greedy = opponent_greedy_opt.value_or(false);
  const bool has_opponent_temperature = opponent_temperature_opt.has_value();
  const float opponent_temperature = opponent_temperature_opt.value_or(0.0f);
  const double candidate_logit_cap = absl::GetFlag(FLAGS_candidate_logit_cap);
  const double opponent_logit_cap = absl::GetFlag(FLAGS_opponent_logit_cap);
  const std::string dump_logit_stats_path = absl::GetFlag(FLAGS_dump_logit_stats);

  if (model_checkpoint.empty()) {
    std::cerr << "Error: --model_checkpoint is required.\n";
    std::exit(1);
  }

  uint64_t domain = ResolveDomain(domain_str);
  if (opp_hidden_dim < 0) opp_hidden_dim = hidden_dim;
  if (opp_num_blocks < 0) opp_num_blocks = num_blocks;

  // Parse opponent paths
  std::vector<std::string> opponent_paths;
  if (!opponent_str.empty() && opponent_str != "random") {
    opponent_paths = SplitCommaSeparated(opponent_str);
  }
  bool use_opponent_model = !opponent_paths.empty();

  std::vector<std::string> opponent_names;
  if (use_opponent_model) {
    for (const std::string& opp_path : opponent_paths) {
      opponent_names.push_back(
          std::filesystem::path(opp_path).stem().string());
    }
  }

  // Initialize game
  std::shared_ptr<const Game> game = LoadGame("dune_imperium");

  int64_t obs_size = 0;
  bool provides_info_state_tensor =
      game->GetType().provides_information_state_tensor;
  bool provides_observations_tensor =
      game->GetType().provides_observation_tensor;
  if (provides_info_state_tensor) {
    obs_size = game->InformationStateTensorSize();
  } else if (provides_observations_tensor) {
    obs_size = game->ObservationTensorSize();
  }
  if (obs_size == 0) {
    std::cerr << "Error: observation size is 0.\n";
    std::exit(1);
  }
  int64_t action_size = game->NumDistinctActions();

  // Determine thread count
  unsigned int hw_threads = std::thread::hardware_concurrency();
  if (hw_threads == 0) hw_threads = 4;
  unsigned int num_threads;
  if (requested_threads > 0) {
    num_threads = static_cast<unsigned int>(requested_threads);
  } else {
    num_threads = hw_threads * 4;
  }

  // Device setup
  torch::InferenceMode inference_guard;
  torch::Device device = torch::cuda::is_available()
                         ? torch::Device(torch::kCUDA)
                         : torch::Device(torch::kCPU);

  int eval_batch_size;
  int eval_timeout_ms;
  if (device.is_cuda()) {
    eval_batch_size = std::min(64u, num_threads);
    eval_timeout_ms = 1;
  } else {
    eval_batch_size = std::min(32u, num_threads);
    eval_timeout_ms = 2;
  }

  std::string device_name = device.is_cuda() ? "CUDA (GPU)" : "CPU";
  bool deterministic = absl::GetFlag(FLAGS_deterministic_eval);
  if (deterministic) {
    std::cout << "INFO: Running in deterministic mode (default). Use --deterministic_eval=false for faster batched evaluation.\n";
  }

  // Synchronization primitives (only used for the active evaluator mode)
  std::shared_mutex sync_mutex;  // For BatchedEvaluator (shared read lock)
  std::mutex eval_mutex;         // For DeterministicEvaluator (exclusive lock)

  // Load eval model with auto-detected dimensions
  int main_hidden_dim = hidden_dim;
  int main_num_blocks = num_blocks;
  if (!DetectModelDimensions(model_checkpoint, &main_hidden_dim, &main_num_blocks)) {
    SpielFatalError("Failed to detect model dimensions for main checkpoint: " + model_checkpoint);
  }

  auto model = std::make_shared<SharedDunePolicyValueNetImpl>(
      obs_size, main_hidden_dim, action_size, main_num_blocks,
      absl::GetFlag(FLAGS_nonlinear_value_head));
  model->eval();
  {
    torch::serialize::InputArchive archive;
    archive.load_from(model_checkpoint, device);
    model->load(archive);
  }
  model->to(device);

  std::shared_ptr<IGameEvaluator> model_evaluator;
  if (deterministic) {
    model_evaluator = std::make_shared<DeterministicEvaluator>(
        model, device, &eval_mutex);
  } else {
    model_evaluator = std::make_shared<BatchedEvaluator>(
        model, eval_batch_size, eval_timeout_ms, device,
        &sync_mutex, 0.0f);
  }

  // Load opponent models with auto-detected dimensions
  struct OpponentMetadata {
    std::string path;
    int hidden_dim;
    int num_blocks;
  };
  std::vector<OpponentMetadata> opp_metadata;
  std::vector<std::shared_ptr<IGameEvaluator>> opponent_evaluators;
  for (const std::string& opp_path : opponent_paths) {
    int opp_detected_hidden = opp_hidden_dim;
    int opp_detected_blocks = opp_num_blocks;
    if (!DetectModelDimensions(opp_path, &opp_detected_hidden, &opp_detected_blocks)) {
      SpielFatalError("Failed to detect model dimensions for opponent checkpoint: " + opp_path);
    }
    opp_metadata.push_back({opp_path, opp_detected_hidden, opp_detected_blocks});

    auto opp_model = std::make_shared<SharedDunePolicyValueNetImpl>(
        obs_size, opp_detected_hidden, action_size, opp_detected_blocks,
        absl::GetFlag(FLAGS_opponent_nonlinear_value_head));
    opp_model->eval();
    {
      torch::serialize::InputArchive archive;
      archive.load_from(opp_path, device);
      opp_model->load(archive);
    }
    opp_model->to(device);

    if (deterministic) {
      opponent_evaluators.push_back(std::make_shared<DeterministicEvaluator>(
          opp_model, device, &eval_mutex));
    } else {
      opponent_evaluators.push_back(std::make_shared<BatchedEvaluator>(
          opp_model, eval_batch_size, eval_timeout_ms, device,
          &sync_mutex, 0.0f));
    }
  }

  // --- Print configuration ---
  std::cout << "=== Dune Population Evaluator ===\n"
            << "Model:      " << model_checkpoint << "\n"
            << "Opponents:  "
            << (use_opponent_model ? opponent_str : "Random Agents") << "\n"
            << "Domain:     " << domain_str << "\n"
            << "Base seed:  " << base_seed << "\n"
            << "Games:      " << total_games << "\n"
            << "Greedy:     " << (greedy ? "true" : "false") << "\n"
            << "Temperature:" << temperature << "\n"
            << "OppGreedy:  "
            << (has_opponent_greedy ? (opponent_greedy ? "true" : "false")
                                    : "(inherit --greedy)") << "\n"
            << "OppTemp:    "
            << (has_opponent_temperature ? std::to_string(opponent_temperature)
                                         : std::string("(inherit --temperature)"))
            << "\n"
            << "CandCap:    " << candidate_logit_cap << "\n"
            << "OppCap:     " << opponent_logit_cap << "\n"
            << "Threads:    " << num_threads << "\n"
            << "Eval mode:  " << (deterministic ? "Deterministic (batch-1)" : "Batched") << "\n"
            << "Nonlinear: "
            << (absl::GetFlag(FLAGS_nonlinear_value_head) ? "true" : "false")
            << "\n"
            << "Batch size: " << eval_batch_size << "\n"
            << "Device:     " << device_name << "\n"
            << "Hidden dim: " << hidden_dim << " / Blocks: " << num_blocks
            << "\n";
  if (use_opponent_model && (opp_hidden_dim != hidden_dim ||
                              opp_num_blocks != num_blocks)) {
    std::cout << "Opp dim:    " << opp_hidden_dim
              << " / Blocks: " << opp_num_blocks << "\n";
  }
  std::cout << std::endl;

  // --- Optional per-candidate-decision logit-stat dump (audit) ---
  std::ofstream logit_dump;
  std::ofstream* logit_dump_ptr = nullptr;
  if (!dump_logit_stats_path.empty()) {
    logit_dump.open(dump_logit_stats_path);
    if (logit_dump.is_open()) {
      logit_dump << "episode_id,round,decision_role,n_legal,max_abs_z,p90_abs_z,"
                    "entropy_uncapped,entropy_cap10,entropy_cap30,"
                    "kl_cap10_uncapped,kl_cap10_cap30,"
                    "frac_legal_absz_ge5,frac_legal_absz_ge10,frac_legal_absz_ge20,"
                    // WO-1 Phase 3 (appended; columns above unchanged)
                    "norm_entropy,max_action_prob,eff_action_count,"
                    "configured_logit_cap,frac_legal_absz_ge_cap,"
                    "frac_legal_absz_ge_2cap\n";
      logit_dump_ptr = &logit_dump;
      std::cout << "Dumping candidate logit stats to " << dump_logit_stats_path
                << "\n";
    } else {
      std::cerr << "WARNING: could not open --dump_logit_stats path: "
                << dump_logit_stats_path << "\n";
    }
  }

  // --- Pre-allocate results (indexed by episode_id for determinism) ---
  std::vector<GameResult> results(total_games);

  auto start_time = std::chrono::steady_clock::now();

  // Launch worker threads
  std::atomic<int> next_game_id{0};
  std::vector<std::thread> threads;
  for (unsigned int t = 0; t < num_threads; ++t) {
    threads.emplace_back(
        WorkerThread, static_cast<int>(t), game, model_evaluator,
        std::cref(opponent_evaluators), std::cref(opponent_names), obs_size,
        provides_info_state_tensor, provides_observations_tensor,
        std::ref(next_game_id), total_games,
        base_seed, domain, greedy, temperature,
        has_opponent_greedy, opponent_greedy,
        has_opponent_temperature, opponent_temperature,
        static_cast<float>(candidate_logit_cap),
        static_cast<float>(opponent_logit_cap),
        logit_dump_ptr,
        std::ref(results));
  }
  for (auto& th : threads) {
    if (th.joinable()) th.join();
  }
  if (logit_dump.is_open()) logit_dump.close();

  auto end_time = std::chrono::steady_clock::now();
  double elapsed_secs =
      std::chrono::duration<double>(end_time - start_time).count();

  // --- Write per-game JSONL ---
  std::ofstream jsonl_out;
  if (!output_dir.empty()) {
    std::filesystem::create_directories(output_dir);
    std::string jsonl_path =
        (std::filesystem::path(output_dir) / "games.jsonl").string();
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
                << "}\n";
    }
  }

  if (jsonl_out.is_open()) {
    jsonl_out.close();
    std::cout << "Per-game results written to "
              << (std::filesystem::path(output_dir) / "games.jsonl").string()
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

  std::cout << absl::StrFormat(
      "P(round <= 7):        %.2f%%   Wilson 95%% CI [%.2f%%, %.2f%%]\n",
      round_le7_ci.point * 100, round_le7_ci.lower * 100,
      round_le7_ci.upper * 100);

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

  // --- Write aggregate JSON with seat-wise and placement counts ---
  if (!output_dir.empty()) {
    std::string agg_path =
        (std::filesystem::path(output_dir) / "aggregate.json").string();
    std::ofstream agg_out(agg_path);
    if (agg_out.is_open()) {
      agg_out << "{\n"
              << "  \"model_checkpoint\": \""
              << model_checkpoint << "\",\n"
              << "  \"opponent_checkpoints\": \""
              << (use_opponent_model ? opponent_str : "random") << "\",\n"
              << "  \"domain\": \"" << domain_str << "\",\n"
              << "  \"base_seed\": " << base_seed << ",\n"
              << "  \"num_games\": " << total_games << ",\n"
              << "  \"greedy\": " << (greedy ? "true" : "false") << ",\n"
              << "  \"opponent_greedy\": "
              << (has_opponent_greedy ? (opponent_greedy ? "true" : "false")
                                      : "null") << ",\n"
              << "  \"opponent_temperature\": "
              << (has_opponent_temperature
                      ? absl::StrFormat("%.6f", opponent_temperature)
                      : std::string("null")) << ",\n"
              << "  \"temperature\": "
              << absl::StrFormat("%.2f", temperature) << ",\n"
              << "  \"candidate_logit_cap\": "
              << absl::StrFormat("%.4f", candidate_logit_cap) << ",\n"
              << "  \"opponent_logit_cap\": "
              << absl::StrFormat("%.4f", opponent_logit_cap) << ",\n"
              << "  \"hidden_dim\": " << main_hidden_dim << ",\n"
              << "  \"num_blocks\": " << main_num_blocks << ",\n"
              << "  \"opp_hidden_dim\": " << (opp_metadata.empty() ? -1 : opp_metadata[0].hidden_dim) << ",\n"
              << "  \"opp_num_blocks\": " << (opp_metadata.empty() ? -1 : opp_metadata[0].num_blocks) << ",\n"
              << "  \"execution_mode\": \"" << (deterministic ? "deterministic" : "batched") << "\",\n"
              << "  \"threads\": " << num_threads << ",\n"
              << "  \"detected_opponent_architectures\": [\n";
      for (size_t i = 0; i < opp_metadata.size(); ++i) {
        agg_out << "    {\n"
                << "      \"checkpoint\": \"" << opp_metadata[i].path << "\",\n"
                << "      \"hidden_dim\": " << opp_metadata[i].hidden_dim << ",\n"
                << "      \"num_blocks\": " << opp_metadata[i].num_blocks << "\n"
                << "    }" << (i + 1 < opp_metadata.size() ? "," : "") << "\n";
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
              << "  \"seat_3_games\": " << games_by_seat[3] << "\n"
              << "}\n";
      agg_out.close();
      std::cout << "\nAggregate results written to " << agg_path << "\n";
    }
  }
}

}  // namespace
}  // namespace open_spiel

int main(int argc, char* argv[]) {
  setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8", 1);
  absl::ParseCommandLine(argc, argv);

  // PWO-5 section 10.2: the reserved final-gate base-seed range. No training
  // OR EVALUATION seed may enter [9000000, 9999999]; checked before any work.
  {
    const std::string stop = dune_seed::ReservedFinalGateSeedStop(
        "--base_seed", static_cast<long long>(absl::GetFlag(FLAGS_base_seed)));
    if (!stop.empty()) {
      std::cerr << stop << "\n";
      return 1;
    }
  }

  // Set PyTorch to use deterministic algorithms globally
  at::globalContext().setDeterministicAlgorithms(true, /*silent=*/true);

  // On GPU: game threads only do engine work. On CPU: Runner needs all cores.
  if (torch::cuda::is_available()) {
    at::set_num_threads(1);
  } else {
    at::set_num_threads(
        std::max(1, static_cast<int>(std::thread::hardware_concurrency())));
  }
  at::set_num_interop_threads(1);

  open_spiel::RunEvaluation();
  return 0;
}
