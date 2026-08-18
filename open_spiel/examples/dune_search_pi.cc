#include "dune_search_pi.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <mutex>
#include <numeric>
#include <set>
#include <sstream>

#include "open_spiel/abseil-cpp/absl/strings/str_cat.h"
#include "open_spiel/abseil-cpp/absl/strings/str_format.h"
#include "open_spiel/algorithms/mcts.h"
#include "open_spiel/games/dune_imperium/dune_imperium.h"  // FinalScoredVp,
                                                           // GetCurrentRound
#include "open_spiel/spiel_utils.h"
#include "dune_online_search_collector.h"  // PruneForcedPlayouts, SharpenVisitTarget,
                                           // ComputeSearchAcceptance (telemetry only)
#include "dune_ppo_training_utils.h"       // CenterAndCapLogitsTensor
#include "dune_seed_utils.h"
#include "dune_sha256.h"

namespace open_spiel {
namespace {

// --- Deterministic stream derivation -------------------------------------
//
// Local to this lane and domain-separated from PPO's streams and from the PF-C
// collector's (which uses its own 0x11..0x66 block inside its own translation
// unit). Distinct constants here mean a search-PI run and an 18B run driven
// from the same base seed never draw a correlated stream.
constexpr uint64_t kStreamPiChance = 0x5001;
constexpr uint64_t kStreamPiOpponentPolicy = 0x5002;
constexpr uint64_t kStreamPiSearchSeed = 0x5003;
constexpr uint64_t kStreamPiBehavior = 0x5004;
constexpr uint64_t kStreamPiUnsearchedSeat = 0x5005;

uint64_t Splitmix64(uint64_t x) {
  x += 0x9E3779B97F4A7C15ULL;
  x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
  x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
  return x ^ (x >> 31);
}

uint64_t DeriveStream(uint64_t seed_domain, int64_t episode_id, int64_t index,
                      uint64_t stream) {
  uint64_t h = Splitmix64(seed_domain ^ stream);
  h = Splitmix64(h ^ static_cast<uint64_t>(episode_id));
  h = Splitmix64(h ^ static_cast<uint64_t>(index));
  return h;
}

// Uniform double in [0,1) from the top 53 bits.
double UnitDouble(uint64_t bits) {
  return static_cast<double>(bits >> 11) * (1.0 / 9007199254740992.0);
}

// Raw-policy action, temperature-aware. Temperature 0 is argmax. This is the
// same shape as the fresh path's SelectRawPriorAction: it must NEVER return a
// uniform draw, because the standing parity invariant is that starved paths
// degrade to the raw network prior.
Action SampleRawPolicyAction(const ActionsAndProbs& prior, double temperature,
                             uint64_t seed_bits,
                             const std::vector<Action>& legal_actions) {
  if (prior.empty()) {
    return legal_actions.empty() ? kInvalidAction : legal_actions.front();
  }
  if (temperature <= 0.0) {
    Action best = prior.front().first;
    double best_p = prior.front().second;
    for (const auto& ap : prior) {
      if (ap.second > best_p) {
        best_p = ap.second;
        best = ap.first;
      }
    }
    return best;
  }
  const double r = UnitDouble(seed_bits);
  std::vector<double> w;
  w.reserve(prior.size());
  const double inv_t = 1.0 / temperature;
  double total = 0.0;
  for (const auto& ap : prior) {
    const double v = (std::abs(temperature - 1.0) < 1e-12)
                         ? ap.second
                         : std::pow(std::max(ap.second, 1e-12), inv_t);
    w.push_back(v);
    total += v;
  }
  if (!(total > 0.0)) {
    return prior.front().first;
  }
  double acc = 0.0;
  for (size_t i = 0; i < w.size(); ++i) {
    acc += w[i] / total;
    if (r < acc) return prior[i].first;
  }
  return prior.back().first;
}

// Manifest form. The value becomes a json::Value string, which the serializer
// quotes for us, so a non-finite token needs no quoting of its own.
std::string F17(double v) {
  if (!std::isfinite(v)) {
    return std::isnan(v) ? "nan" : (v > 0 ? "inf" : "-inf");
  }
  return absl::StrFormat("%.17g", v);
}

// Sidecar form. This one is streamed RAW into a JSONL line, so a non-finite
// value has to carry its own quotes or the line stops being JSON -- and the
// generation whose learner diverged is exactly the one whose telemetry must
// still parse.
std::string F17Json(double v) {
  if (!std::isfinite(v)) {
    return std::isnan(v) ? "\"nan\"" : (v > 0 ? "\"inf\"" : "\"-inf\"");
  }
  return absl::StrFormat("%.17g", v);
}

void AppendBytes(std::string* out, const void* p, size_t n) {
  out->append(static_cast<const char*>(p), n);
}

}  // namespace

// ---------------------------------------------------------------------------
// Names
// ---------------------------------------------------------------------------

const char* SearchPiFallbackName(SearchPiFallback reason) {
  switch (reason) {
    case SearchPiFallback::kNone: return "none";
    case SearchPiFallback::kNoSearchRun: return "no_search_run";
    case SearchPiFallback::kZeroVisits: return "zero_visits";
    case SearchPiFallback::kInvalidAlignment: return "invalid_alignment";
    case SearchPiFallback::kIllegalAction: return "illegal_action";
    case SearchPiFallback::kTimeoutBeforeUsefulSearch:
      return "timeout_before_useful_search";
    case SearchPiFallback::kEmptyPolicy: return "empty_policy";
  }
  return "unknown";
}

const char* SearchPiEarlyExitName(SearchPiEarlyExit reason) {
  switch (reason) {
    case SearchPiEarlyExit::kNone: return "none";
    case SearchPiEarlyExit::kNoSearchRun: return "no_search_run";
    case SearchPiEarlyExit::kTimeout: return "timeout";
    case SearchPiEarlyExit::kMaxNodes: return "max_nodes";
    case SearchPiEarlyExit::kSingleLegalAction: return "single_legal_action";
    case SearchPiEarlyExit::kEmptyOrZeroVisits: return "empty_or_zero_visits";
    case SearchPiEarlyExit::kSessionBudget: return "session_budget";
    case SearchPiEarlyExit::kUnclassified: return "unclassified";
  }
  return "unknown";
}

SearchPiEarlyExit ClassifySearchPiEarlyExit(
    const DuneSearchResult& result, const std::vector<Action>& legal_actions,
    int requested_simulations) {
  // A session-level clamp is reported first even when the (clamped) budget was
  // then spent in full: "the session cut this search short" is the cause, and
  // the count it was cut to is not evidence against it. Inert under
  // kTrainingPolicyIteration, whose whole point is that the two role budgets are
  // never drawn from a shared pool -- carried explicitly so that if it ever DOES
  // fire, it is named rather than silently reclassified as a timeout.
  const std::string& limit = result.diagnostics.budget_limit_reason;
  if (!limit.empty() && limit != "none") return SearchPiEarlyExit::kSessionBudget;

  if (requested_simulations <= 0) return SearchPiEarlyExit::kNoSearchRun;
  if (result.simulations_completed >= requested_simulations) {
    return SearchPiEarlyExit::kNone;
  }

  // Short. Name the cause. The loop is `for (; sim < actual_max_sims; ++sim)`
  // and reports `simulations_completed = sim`, so short is equivalent to "a
  // break fired, or RunSearch returned before the loop".
  const std::string& r = result.fallback_reason;
  if (result.timeout_status || r == "timeout") return SearchPiEarlyExit::kTimeout;
  if (r == "max_nodes") return SearchPiEarlyExit::kMaxNodes;

  // From here down the bot gave NO cause, so anything we conclude is inferred
  // from the result's shape. Guard that inference on the reason string actually
  // being absent: a short search carrying an unrecognised reason must reach
  // kUnclassified, because kUnclassified means "the instrument does not know"
  // and an unknown cause is exactly that. Without this guard the
  // `non_strategic_state` return -- short, unrecognised reason, and a
  // DEFAULT-CONSTRUCTED diagnostics block whose actions are therefore empty --
  // would be absorbed into kEmptyOrZeroVisits and read as a known case.
  // (Unreachable today: SearchPiSearchConfigFor sets check_strategic_state
  // false. Guarded anyway, because the enum's contract is what makes a zero
  // unclassified count mean anything.)
  if (!r.empty() && r != "none") return SearchPiEarlyExit::kUnclassified;

  // Checked AFTER the reason strings because RunSearch's forced-root return
  // leaves fallback_reason at its "none" default (dune_puct_is_mcts.cc:809-816),
  // so this case is invisible in the string and would otherwise land in
  // kUnclassified -- which must mean "the instrument does not know", not "a
  // known case the instrument forgot to ask about".
  if (legal_actions.size() <= 1) return SearchPiEarlyExit::kSingleLegalAction;
  if (result.diagnostics.actions.empty()) {
    return SearchPiEarlyExit::kEmptyOrZeroVisits;
  }
  int64_t total_visits = 0;
  for (int v : result.diagnostics.visit_counts) total_visits += v;
  if (total_visits <= 0) return SearchPiEarlyExit::kEmptyOrZeroVisits;

  return SearchPiEarlyExit::kUnclassified;
}

const char* SearchPiArmName(SearchPiArm arm) {
  return arm == SearchPiArm::kRawArgmax ? "raw_argmax" : "searched";
}

int SearchPiPlacementFromReturn(double engine_return, double utility_divisor) {
  if (!(utility_divisor > 0.0)) return 0;
  // dune_imperium.cc:2383-2447. Scaled the same way SearchPiRow::value_target
  // is, so one helper reads a raw return and a value target alike.
  static const double kLadder[4] = {2.25, 0.25, -0.75, -1.75};
  for (int k = 0; k < 4; ++k) {
    if (std::abs(engine_return - kLadder[k] / utility_divisor) < 1e-9) {
      return k + 1;
    }
  }
  return 0;
}

const char* SearchPiContinuationTargetName(SearchPiContinuationTarget mode) {
  return mode == SearchPiContinuationTarget::kNewVisitsOnly ? "new_visits_only"
                                                            : "total_visits";
}

bool ParseSearchPiContinuationTarget(const std::string& name,
                                     SearchPiContinuationTarget* out) {
  if (name == "total_visits") {
    *out = SearchPiContinuationTarget::kTotalVisits;
    return true;
  }
  if (name == "new_visits_only") {
    *out = SearchPiContinuationTarget::kNewVisitsOnly;
    return true;
  }
  return false;
}

// ---------------------------------------------------------------------------
// Search configuration
// ---------------------------------------------------------------------------

DuneSearchConfig SearchPiSearchConfigFor(const SearchPiConfig& config,
                                         uint64_t search_seed) {
  DuneSearchConfig cfg;
  cfg.seed = search_seed;
  cfg.puct_c = config.puct_c;
  cfg.max_search_decision_depth = config.max_search_decision_depth;
  cfg.opponent_mode = config.use_opponent_model ? SearchOpponentMode::kPolicy
                                                : SearchOpponentMode::kMaxN;
  cfg.opponent_temperature = config.opponent_model_temperature;
  cfg.root_prior_temperature = config.root_prior_temperature;
  cfg.training_root_prior_temperature = config.root_prior_temperature;
  cfg.utility_divisor = config.utility_divisor;
  cfg.dirichlet_epsilon = config.dirichlet_epsilon;
  cfg.dirichlet_alpha_total = config.dirichlet_alpha_total;
  cfg.forced_playouts_k = config.forced_playouts_k;
  cfg.root_noise_fpu_zero = config.root_noise_fpu_zero;

  // The lane owns action selection; the session's own controller temperature is
  // never consulted, because SelectControllerAction is not on this lane's path.
  cfg.temperature = config.behavior_temperature;

  // The two budgets the new mode reads. fixed_session_limit is deliberately
  // left at its default and is NOT consulted by kTrainingPolicyIteration.
  cfg.pi_primary_simulations = config.primary_simulations;
  cfg.pi_continuation_simulations = config.continuation_simulations;

  // Simulations at the three roles the NARROW teacher must not search.
  //
  // purchase_combat_budget <= 0 makes DuneSearchSession::Search return the
  // policy-only "policy_only_purchase_combat" result before any budget is
  // resolved (dune_search_session.cc:375, the `policy_only_purchase_combat`
  // return; the previous :354-355 pointed at a different branch's comment),
  // which covers kPurchase,
  // kCombatIntrigue and kOtherOptional. search_leader_draft == false sends
  // kLeaderSelection down the "forced_or_bookkeeping" early return
  // (dune_search_session.cc:356-361, the two `forced_or_bookkeeping` returns;
  // :339-341 was a misread and is inside get_policy_only_result's body).
  // kForcedOrBookkeeping returns there too.
  //
  // This was a hard-coded 0 through rung 3a. It is now read from the config so
  // the WIDE arm can register a nonzero dose -- the SearchPiConfig default is
  // still 0, so every caller that does not set it is unchanged bit-for-bit.
  // search_leader_draft stays structurally false regardless: leader picks are
  // not part of the scope axis under test, and SearchPiGenerator's constructor
  // fatals if it is ever true here.
  cfg.purchase_combat_budget = config.purchase_combat_budget;
  // This lane is an AUDIT of an exact-simulation teacher, so the short-window
  // roles get exact per-root budgets under the registered watchdog rather than
  // a shared pool under a hard-coded 500 ms wall. Inert for the narrow teacher,
  // whose budget is 0 and whose short-window roles return policy-only before
  // any budget is resolved.
  cfg.exact_short_window_budgets = true;
  cfg.fixed_continuation_reserve = 0;
  cfg.search_leader_draft = false;
  cfg.leader_mass_only_coverage = false;

  // The runaway watchdog. Previously never assigned, so it inherited
  // DuneSearchConfig's own 10,000 ms default; SearchPiConfig's default is that
  // same 10,000 ms, which makes this assignment a no-op for every historical
  // caller and keeps the rung-2 parity chains reproducible. The batched teacher
  // registers 60,000 ms.
  cfg.relative_time_budget_ms = config.relative_time_budget_ms;

  // The search result itself must not be discarded for covering few actions --
  // that is exactly the PF-C behaviour this lane exists to remove. A
  // concentrated search is a finding, not a defect. min_visit_threshold stays
  // at its default because it only defines "covered" for telemetry and for
  // which Q values are trusted; the coverage GATE is what we neutralize.
  cfg.covered_prior_threshold = 0.0;

  cfg.check_strategic_state = false;
  cfg.use_observation_string = true;
  return cfg;
}

// ---------------------------------------------------------------------------
// Result classification
// ---------------------------------------------------------------------------

SearchPiFallback ClassifySearchPiResult(
    const DuneSearchResult& result, const std::vector<Action>& legal_actions) {
  const SearchDiagnostics& d = result.diagnostics;
  if (d.actions.empty() || legal_actions.empty()) {
    return SearchPiFallback::kEmptyPolicy;
  }
  if (d.actions.size() != d.visit_counts.size()) {
    return SearchPiFallback::kInvalidAlignment;
  }
  if (d.actions.size() != legal_actions.size()) {
    return SearchPiFallback::kInvalidAlignment;
  }
  for (size_t i = 0; i < d.actions.size(); ++i) {
    if (d.actions[i] != legal_actions[i]) {
      return SearchPiFallback::kInvalidAlignment;
    }
  }
  int64_t total = 0;
  for (int v : d.visit_counts) {
    if (v < 0) return SearchPiFallback::kInvalidAlignment;
    total += v;
  }
  if (total <= 0) {
    // No visits at all. A timeout that produced nothing usable reports itself
    // as a timeout so the two causes stay distinguishable in telemetry.
    return result.timeout_status ? SearchPiFallback::kTimeoutBeforeUsefulSearch
                                 : SearchPiFallback::kZeroVisits;
  }
  return SearchPiFallback::kNone;
}

// ---------------------------------------------------------------------------
// Distribution scalars
// ---------------------------------------------------------------------------

double NormalizedEntropy(const std::vector<double>& probs) {
  const size_t n = probs.size();
  if (n <= 1) return 0.0;
  double h = 0.0;
  for (double p : probs) {
    if (p > 0.0) h -= p * std::log(p);
  }
  const double denom = std::log(static_cast<double>(n));
  if (!(denom > 0.0)) return 0.0;
  const double v = h / denom;
  // Numerical guard only: an exactly-uniform distribution can land a few ulp
  // above 1.0 through the division.
  return std::min(1.0, std::max(0.0, v));
}

double KlTargetGivenRaw(const std::vector<double>& target,
                        const std::vector<double>& raw, bool* floored) {
  if (floored != nullptr) *floored = false;
  if (target.size() != raw.size() || target.empty()) return 0.0;
  // json.cc floors any float below 5e-7 to exactly 0.0, so a serialized raw
  // prior can read as zero where the live one was merely tiny. This value is
  // therefore computed once, here, from the in-memory doubles.
  constexpr double kRawFloor = 1e-12;
  double kl = 0.0;
  for (size_t i = 0; i < target.size(); ++i) {
    const double t = target[i];
    if (!(t > 0.0)) continue;
    double r = raw[i];
    if (!(r > 0.0)) {
      r = kRawFloor;
      if (floored != nullptr) *floored = true;
    }
    kl += t * std::log(t / r);
  }
  return kl;
}

void FillSearchPiRowScalars(SearchPiRow* row) {
  SPIEL_CHECK_TRUE(row != nullptr);
  row->raw_policy_entropy_norm = NormalizedEntropy(row->raw_policy);
  row->target_entropy_norm = NormalizedEntropy(row->target_probs);

  row->raw_policy_max_prob = 0.0;
  int raw_argmax = -1;
  for (size_t i = 0; i < row->raw_policy.size(); ++i) {
    if (raw_argmax < 0 || row->raw_policy[i] > row->raw_policy_max_prob) {
      row->raw_policy_max_prob = row->raw_policy[i];
      raw_argmax = static_cast<int>(i);
    }
  }
  row->target_max_prob = 0.0;
  int tgt_argmax = -1;
  for (size_t i = 0; i < row->target_probs.size(); ++i) {
    if (tgt_argmax < 0 || row->target_probs[i] > row->target_max_prob) {
      row->target_max_prob = row->target_probs[i];
      tgt_argmax = static_cast<int>(i);
    }
  }
  bool floored = false;
  row->kl_target_given_raw =
      KlTargetGivenRaw(row->target_probs, row->raw_policy, &floored);
  row->kl_raw_prior_floored = floored;
  row->target_argmax_differs_from_raw =
      (raw_argmax >= 0 && tgt_argmax >= 0 && raw_argmax != tgt_argmax);
}

// ---------------------------------------------------------------------------
// Target construction
// ---------------------------------------------------------------------------

bool BuildSearchPiTarget(const SearchPiConfig& config, DuneDecisionRole role,
                         const SearchDiagnostics& diag,
                         std::vector<int>* out_target_visits,
                         std::vector<double>* out_target_probs) {
  const size_t n = diag.visit_counts.size();
  if (n == 0 || diag.actions.size() != n) return false;

  // Which visit vector the target normalizes.
  std::vector<int> base(diag.visit_counts);
  if (role == DuneDecisionRole::kAgentContinuation &&
      config.continuation_target == SearchPiContinuationTarget::kNewVisitsOnly) {
    // new = final - pre_search, available only when the retention snapshots are
    // valid. When they are not, fall back to total rather than inventing a
    // difference against an absent baseline.
    if (diag.retention_snapshots_valid &&
        diag.visit_counts_final.size() == n &&
        diag.visit_counts_pre_search.size() == n) {
      for (size_t i = 0; i < n; ++i) {
        base[i] = std::max(0, diag.visit_counts_final[i] -
                                  diag.visit_counts_pre_search[i]);
      }
    }
  }

  int64_t base_total = 0;
  for (int v : base) base_total += v;
  if (base_total <= 0) return false;

  // KataGo visit-target pruning, applied to the TARGET only and only when the
  // exploration package is actually armed. With epsilon 0 there are no forced
  // playouts, so an armed pruner would subtract organic visits.
  std::vector<int> target_visits = base;
  const bool package_active = config.forced_playouts_k > 0.0 &&
                              config.dirichlet_epsilon > 0.0 &&
                              diag.q_values.size() == n &&
                              diag.priors.size() == n;
  if (package_active) {
    int64_t total_root = diag.total_root_visits > 0
                             ? diag.total_root_visits
                             : base_total;
    std::vector<int> pruned = PruneForcedPlayouts(
        base, diag.priors, diag.q_values, static_cast<int>(total_root),
        config.puct_c, config.forced_playouts_k);
    int64_t ptot = 0;
    for (int v : pruned) ptot += v;
    if (ptot > 0) target_visits = std::move(pruned);
  }

  int64_t total = 0;
  for (int v : target_visits) total += v;
  if (total <= 0) return false;

  std::vector<double> probs(n);
  for (size_t i = 0; i < n; ++i) {
    probs[i] = static_cast<double>(target_visits[i]) / static_cast<double>(total);
  }
  probs = SharpenVisitTarget(probs, config.target_sharpen_exponent);

  // Renormalize defensively so the emitted target sums to 1 exactly enough for
  // the CE to be a proper cross-entropy. SharpenVisitTarget already normalizes,
  // but a target that silently failed to sum to one is the kind of defect that
  // only shows up as a slow bias.
  double s = 0.0;
  for (double p : probs) s += p;
  if (!(s > 0.0)) return false;
  for (double& p : probs) p /= s;

  *out_target_visits = std::move(target_visits);
  *out_target_probs = std::move(probs);
  return true;
}

int SelectSearchPiActionIndex(const std::vector<int>& raw_visits,
                              double behavior_temperature, double r_val) {
  if (raw_visits.empty()) return -1;
  if (behavior_temperature <= 0.0) {
    int best = 0;
    for (size_t i = 1; i < raw_visits.size(); ++i) {
      if (raw_visits[i] > raw_visits[best]) best = static_cast<int>(i);
    }
    return raw_visits[best] > 0 ? best : -1;
  }
  std::vector<double> w(raw_visits.size());
  const double inv_t = 1.0 / behavior_temperature;
  double total = 0.0;
  for (size_t i = 0; i < raw_visits.size(); ++i) {
    const double v = static_cast<double>(raw_visits[i]);
    w[i] = (std::abs(behavior_temperature - 1.0) < 1e-12)
               ? v
               : (v > 0.0 ? std::pow(v, inv_t) : 0.0);
    total += w[i];
  }
  if (!(total > 0.0)) return -1;
  double acc = 0.0;
  for (size_t i = 0; i < w.size(); ++i) {
    acc += w[i] / total;
    if (r_val < acc) return static_cast<int>(i);
  }
  return static_cast<int>(w.size()) - 1;
}

Action RawPriorArgmaxAction(const ActionsAndProbs& prior,
                            const std::vector<Action>& legal_actions) {
  if (prior.empty()) {
    return legal_actions.empty() ? kInvalidAction : legal_actions.front();
  }
  Action best = prior.front().first;
  double best_p = prior.front().second;
  for (const auto& ap : prior) {
    if (ap.second > best_p) {
      best_p = ap.second;
      best = ap.first;
    }
  }
  return best;
}

Action RawPriorArgmaxFromDiagnostics(const SearchDiagnostics& d) {
  // The raw-prior argmax the MATCHED CONTROL would have played at this root,
  // recomputed from the search's own diagnostics. Deliberately not a second
  // Prior() call: an extra inference per searched root would perturb both
  // inference_calls and the throughput this audit is gated on.
  //
  // Same source expression the emitted row uses for `raw_policy` (raw_priors
  // when present, priors otherwise) and the same FIRST-WINS tie-break as
  // RawPriorArgmaxAction, so on the raw arm -- where diagnostics.priors is
  // filled from the identical Prior() call, in the identical order -- the two
  // agree exactly rather than approximately.
  const std::vector<double>& p =
      d.raw_priors.empty() ? d.priors : d.raw_priors;
  if (d.actions.empty() || p.size() != d.actions.size()) return kInvalidAction;
  size_t best = 0;
  for (size_t i = 1; i < p.size(); ++i) {
    if (p[i] > p[best]) best = i;
  }
  return d.actions[best];
}

uint64_t MixPlayedAction(uint64_t h, int64_t decision_id, Action action) {
  // FNV-1a 64 over the (decision_id, action) pair. A divergence detector; the
  // lane's bitwise claims live in the sha256 row chains and nowhere else.
  const uint64_t vals[2] = {static_cast<uint64_t>(decision_id),
                            static_cast<uint64_t>(action)};
  for (uint64_t v : vals) {
    for (int b = 0; b < 8; ++b) {
      h ^= (v >> (8 * b)) & 0xFFull;
      h *= 0x100000001b3ull;
    }
  }
  return h;
}

// ---------------------------------------------------------------------------
// Generator
// ---------------------------------------------------------------------------

SearchPiGenerator::SearchPiGenerator(const SearchPiConfig& config)
    : config_(config) {
  SPIEL_CHECK_GT(config_.games_per_generation, 0);
  // Seat balance: the searched seat rotates by episode id, so a generation that
  // is not a multiple of four searches some seats more than others.
  SPIEL_CHECK_EQ(config_.games_per_generation % 4, 0);
  SPIEL_CHECK_GT(config_.primary_simulations, 0);
  SPIEL_CHECK_GT(config_.continuation_simulations, 0);
  SPIEL_CHECK_GT(config_.seed_domain, 0u);
  SPIEL_CHECK_GT(config_.utility_divisor, 0.0);
  SPIEL_CHECK_GE(config_.behavior_temperature, 0.0);
  SPIEL_CHECK_GE(config_.target_sharpen_exponent, 0.0);
  // The Leader teacher must not leak into this lane. It is not a tunable knob
  // here: the lane's scope is agent-turn decisions.
  //
  // Deliberately NOT SPIEL_CHECK_FALSE(config_.search_leader_draft). That macro
  // stringifies its argument, emitting a literal "config_.search_leader_draft"
  // -- and the linker TAIL-MERGES the standalone ABSL flag-name literal
  // "search_leader_draft" into it, because the longer string ends with the
  // shorter one. tests/test_production_training_pins.py greps `strings` on the
  // built trainer for that flag name as a whole line, so the macro form silently
  // broke a registered fail-closed control while the flag itself still worked.
  // Any future literal ENDING in a pinned flag name will do the same.
  if (config_.search_leader_draft) {
    SpielFatalError(
        "search-PI does not search the Leader draft: the lane's scope is "
        "kAgentPrimary and kAgentContinuation only.");
  }
  // Forced playouts without noise would prune organic visits out of the target.
  if (config_.dirichlet_epsilon <= 0.0) {
    SPIEL_CHECK_EQ(config_.forced_playouts_k, 0.0);
  }
}

Player SearchPiGenerator::SearchedSeatForEpisode(int64_t episode_id,
                                                 int num_players) {
  SPIEL_CHECK_GT(num_players, 0);
  return static_cast<Player>(
      ((episode_id % num_players) + num_players) % num_players);
}

void SearchPiGenerator::GenerateGeneration(
    int generation, const std::shared_ptr<const Game>& game,
    const std::shared_ptr<algorithms::Evaluator>& evaluator,
    std::vector<SearchPiRow>* out, SearchPiGenerationStats* stats,
    SearchPiArm arm) {
  SPIEL_CHECK_TRUE(game != nullptr);
  SPIEL_CHECK_TRUE(evaluator != nullptr);
  SPIEL_CHECK_TRUE(out != nullptr);
  SPIEL_CHECK_TRUE(stats != nullptr);

  const auto wall_start = std::chrono::steady_clock::now();
  const bool provides_istate =
      game->GetType().provides_information_state_tensor;
  const int num_players = game->NumPlayers();
  const int64_t first_episode_id = config_.next_episode_id;
  // `out` is APPENDED to, so every per-generation aggregate below must be taken
  // over the slice this call added, not over the whole vector. Summing the whole
  // vector would double-count `rows_total` for a caller that reuses one buffer
  // and -- worse -- would re-hash the previous generation's rows into this
  // generation's target chain, breaking the resume verification the chain is for.
  const size_t out_base = out->size();

  *stats = SearchPiGenerationStats();
  stats->generation = generation;
  stats->games = config_.games_per_generation;
  stats->first_episode_id = first_episode_id;
  stats->arm = arm;
  stats->games_played.reserve(config_.games_per_generation);

  // Short searches should be zero; rung 2 saw 29 missing simulations across 14
  // cells. The cap is generous enough that a real occurrence is fully named and
  // small enough that a pathological generation cannot bury run.log.
  constexpr int kMaxShortSearchLogLines = 32;
  int short_searches_logged = 0;

  for (int g = 0; g < config_.games_per_generation; ++g) {
    const int64_t episode_id = first_episode_id + g;
    const Player searched_seat =
        SearchedSeatForEpisode(episode_id, num_players);

    // ONE persistent session for the whole game. Its own lifecycle handles the
    // activation boundaries: it starts/resets at kAgentPrimary
    // (dune_search_session.cc:202-213), re-roots through kAgentContinuation via
    // the history-prefix descendant test (:173-199) and the node-key lookup
    // (:479-496), preserves the tree across chance and intermediate decisions
    // because nothing prunes the node map, and resets at the real
    // agent-turn/seat/round boundary (:174-176).
    DuneSearchConfig scfg = SearchPiSearchConfigFor(
        config_,
        DeriveStream(config_.seed_domain, episode_id, 0, kStreamPiSearchSeed));
    // The matched control keeps the SESSION and changes only the budget mode.
    // It is not an optimisation to drop the session: role classification reads
    // has_active_session() to tell kAgentPrimary from kAgentContinuation
    // (dune_search_routing.cc:9), and the two roles are exactly the ones this
    // arm plays differently from the seat's off-scope decisions. Without a
    // session every agent-turn decision would classify as a primary and the
    // control would stop being matched. kPolicyOnly returns before the budget
    // is resolved (dune_search_session.cc:399-401) while the session start,
    // re-root and commit bookkeeping above it all still run.
    DuneSearchSession session(scfg, evaluator,
                              arm == SearchPiArm::kRawArgmax
                                  ? DuneSearchBudgetMode::kPolicyOnly
                                  : DuneSearchBudgetMode::kTrainingPolicyIteration);
    session.SetEpisodeId(static_cast<int>(episode_id));
    session.SetUpdateId(generation);

    std::unique_ptr<State> state = game->NewInitialState();
    std::vector<size_t> emitted_indices;
    int64_t decision_index = 0;
    int64_t chance_index = 0;

    SearchPiGameOutcome outcome;
    outcome.episode_id = episode_id;
    outcome.searched_seat = searched_seat;

    while (!state->IsTerminal()) {
      if (state->IsChanceNode()) {
        ActionsAndProbs outcomes = state->ChanceOutcomes();
        const double r = UnitDouble(DeriveStream(
            config_.seed_domain, episode_id, chance_index++, kStreamPiChance));
        state->ApplyAction(SampleActionFromPrior(outcomes, r));
        continue;
      }

      const Player cur = state->CurrentPlayer();
      const int64_t this_decision = decision_index++;
      const std::vector<Action> legal_actions = state->LegalActions();

      if (cur != searched_seat) {
        // Every other seat plays the frozen raw policy at temperature 1.0.
        ActionsAndProbs prior = evaluator->Prior(*state);
        state->ApplyAction(SampleRawPolicyAction(
            prior, config_.non_search_temperature,
            DeriveStream(config_.seed_domain, episode_id, this_decision,
                         kStreamPiOpponentPolicy),
            legal_actions));
        continue;
      }

      // --- The searched seat ------------------------------------------------
      //
      // EVERY decision of this seat goes through the session, including the
      // ones this lane does not search. Skipping them would leave the session's
      // re-root anchor (last_input_history_, advanced only by CommitAction)
      // pointing at a stale prefix, and the next continuation would re-root
      // against a history that never happened. The non-agent roles come back as
      // zero-simulation policy-only results.
      const DuneDecisionRole role =
          ClassifyDuneDecisionRole(*state, cur, session.HasActiveSession());
      const int role_idx = static_cast<int>(role);
      stats->decisions_by_role[role_idx]++;
      outcome.searched_seat_decisions++;

      DuneSearchResult res = session.Search(*state);
      stats->simulations_by_role[role_idx] += res.simulations_completed;
      if (!IsSearchPiSearchRole(role, config_.purchase_combat_budget)) {
        outcome.off_scope_simulations += res.simulations_completed;
      }
      stats->inference_calls += res.inference_count;

      const bool is_search_role =
          IsSearchPiSearchRole(role, config_.purchase_combat_budget);
      // Every searched role now has a real bucket. The three short-window roles
      // resolve to their own stats only when the wide budget is armed; with the
      // narrow default IsSearchPiSearchRole is false for them, is_search_role
      // gates every use below, and these buckets stay untouched at zero.
      SearchPiRoleStats* rs =
          role == DuneDecisionRole::kAgentPrimary        ? &stats->primary
          : role == DuneDecisionRole::kAgentContinuation ? &stats->continuation
          : role == DuneDecisionRole::kPurchase          ? &stats->purchase
          : role == DuneDecisionRole::kCombatIntrigue    ? &stats->combat_intrigue
          : role == DuneDecisionRole::kOtherOptional     ? &stats->other_optional
                                                         : nullptr;

      Action chosen = kInvalidAction;
      SearchPiFallback fallback = SearchPiFallback::kNoSearchRun;

      if (is_search_role) {
        SPIEL_CHECK_TRUE(rs != nullptr);
        rs->roots_seen++;
        if (role == DuneDecisionRole::kAgentPrimary) {
          outcome.primary_roots++;
          outcome.primary_simulations += res.simulations_completed;
        } else if (role == DuneDecisionRole::kAgentContinuation) {
          outcome.continuation_roots++;
          outcome.continuation_simulations += res.simulations_completed;
        } else {
          // kPurchase / kCombatIntrigue / kOtherOptional, reachable only when
          // the wide budget is armed. Before this branch existed they fell into
          // the continuation arm of a two-way split, so a wide episode reported
          // ~55 phantom continuation roots and its per-episode counters could
          // not be reconciled against the summary role buckets.
          outcome.wide_roots++;
          outcome.wide_simulations += res.simulations_completed;
        }
      }

      if (is_search_role && arm == SearchPiArm::kRawArgmax) {
        // The matched control. kPolicyOnly returned the frozen network's own
        // prior with no simulation spent, so there is no search result to
        // classify, no target to build and no row to emit -- the arm exists to
        // measure a controller, not to teach one. `chosen` is left invalid on
        // purpose: the shared path below plays the raw-prior ARGMAX for a
        // searched role, which is exactly the one behaviour this arm changes.
        //
        // Recorded as an early exit rather than as a fallback: a fallback is a
        // technical failure of a search that ran, and no search ran here.
        rs->early_exit_counts[
            static_cast<int>(SearchPiEarlyExit::kNoSearchRun)]++;
      } else if (is_search_role) {
        rs->searches_run++;
        rs->simulations_completed += res.simulations_completed;
        rs->inherited_visits += res.diagnostics.inherited_root_visits;
        if (res.diagnostics.re_root_status == "hit") rs->re_root_hits++;
        if (res.diagnostics.re_root_status == "miss") rs->re_root_misses++;

        // --- Early-exit accounting, BEFORE the fallback classification ---
        //
        // Deliberately covers every searched root, including the ones that go
        // on to fall back: a search that was BOTH truncated and unusable is two
        // facts, and folding it into the fallback count loses the first one.
        // The requested count comes from the session's own soft_sim_limit, so
        // this arithmetic cannot agree with itself by construction.
        const int requested = res.diagnostics.soft_sim_limit;
        const SearchPiEarlyExit early_exit =
            ClassifySearchPiEarlyExit(res, legal_actions, requested);
        rs->simulations_requested += requested;
        rs->early_exit_counts[static_cast<int>(early_exit)]++;

        // The same quantity per episode, incremented here rather than derived
        // downstream, so the per-episode and summary views cannot disagree.
        if (role == DuneDecisionRole::kAgentPrimary) {
          outcome.primary_simulations_requested += requested;
        } else if (role == DuneDecisionRole::kAgentContinuation) {
          outcome.continuation_simulations_requested += requested;
        } else {
          outcome.wide_simulations_requested += requested;
        }

        // Deadline headroom. A zero shortfall says the wall-clock guard did not
        // fire; these say by how much, which is what distinguishes "it held" from
        // "it holds". The configured limit is read back from the session rather
        // than assumed, so the margin is against the deadline actually in force.
        const double elapsed = res.diagnostics.elapsed_search_time_ms;
        rs->sum_search_elapsed_ms += elapsed;
        if (elapsed > rs->max_search_elapsed_ms) {
          rs->max_search_elapsed_ms = elapsed;
        }
        rs->configured_time_limit_ms = res.diagnostics.soft_time_limit_ms;

        if (res.simulations_completed < requested) {
          rs->searches_short_of_budget++;
          rs->simulation_shortfall += requested - res.simulations_completed;
          outcome.searches_short_of_budget++;
          if (rs->first_short_episode_id < 0) {
            rs->first_short_episode_id = episode_id;
            rs->first_short_decision_id = this_decision;
            rs->first_short_simulations_completed = res.simulations_completed;
            rs->first_short_simulations_requested = requested;
            rs->first_short_bot_reason = res.fallback_reason;
            rs->first_short_session_reason = res.diagnostics.budget_limit_reason;
          }
          // One line per short search, capped, so a forensic run NAMES the root
          // in its own run.log instead of leaving the aggregate to be divided
          // by an assumed per-root budget. The cap exists because the aggregate
          // is already in telemetry; this is the attribution, not the count.
          //
          // Built as ONE string under a lock and emitted in a single insertion.
          // Chained operator<< synchronises each field separately, so on the
          // multi-threaded audit path the fields of concurrent lines interleave
          // -- and a line whose entire purpose is to name one root is worth
          // nothing shuffled with another root's.
          if (short_searches_logged < kMaxShortSearchLogLines) {
            ++short_searches_logged;
            std::ostringstream line;
            line << "[search-PI] SHORT SEARCH gen=" << generation
                 << " episode=" << episode_id
                 << " decision=" << this_decision
                 << " role=" << static_cast<int>(role)
                 << " sims=" << res.simulations_completed << "/" << requested
                 << " early_exit=" << SearchPiEarlyExitName(early_exit)
                 << " bot_reason=" << res.fallback_reason
                 << " session_reason=" << res.diagnostics.budget_limit_reason
                 << " elapsed_ms=" << elapsed
                 << " time_limit_ms=" << res.diagnostics.soft_time_limit_ms
                 << " tree_nodes=" << res.diagnostics.tree_node_count
                 << " legal=" << legal_actions.size()
                 << " inherited=" << res.diagnostics.inherited_root_visits
                 << " re_root=" << res.diagnostics.re_root_status << "\n";
            static std::mutex short_log_mutex;
            std::lock_guard<std::mutex> lk(short_log_mutex);
            std::cout << line.str() << std::flush;
          } else if (short_searches_logged == kMaxShortSearchLogLines) {
            ++short_searches_logged;
            static std::mutex cap_log_mutex;
            std::lock_guard<std::mutex> lk(cap_log_mutex);
            std::cout << "[search-PI] SHORT SEARCH log capped at "
                      << kMaxShortSearchLogLines
                      << " lines; the per-role totals in the telemetry sidecar "
                         "carry the rest.\n" << std::flush;
          }
        }

        fallback = ClassifySearchPiResult(res, legal_actions);

        std::vector<int> target_visits;
        std::vector<double> target_probs;
        if (fallback == SearchPiFallback::kNone &&
            !BuildSearchPiTarget(config_, role, res.diagnostics, &target_visits,
                                 &target_probs)) {
          fallback = SearchPiFallback::kZeroVisits;
        }

        if (fallback == SearchPiFallback::kNone) {
          // Execute from the UNPRUNED visits, independently of the pruning that
          // shaped the target.
          const double r_beh = UnitDouble(DeriveStream(
              config_.seed_domain, episode_id, this_decision, kStreamPiBehavior));
          const int idx = SelectSearchPiActionIndex(
              res.diagnostics.visit_counts, config_.behavior_temperature, r_beh);
          if (idx < 0 ||
              idx >= static_cast<int>(res.diagnostics.actions.size())) {
            fallback = SearchPiFallback::kZeroVisits;
          } else {
            const Action candidate = res.diagnostics.actions[idx];
            if (std::find(legal_actions.begin(), legal_actions.end(),
                          candidate) == legal_actions.end()) {
              fallback = SearchPiFallback::kIllegalAction;
            } else {
              chosen = candidate;
            }
          }
        }

        if (fallback == SearchPiFallback::kNone) {
          const std::vector<double>& raw_priors =
              res.diagnostics.raw_priors.empty() ? res.diagnostics.priors
                                                 : res.diagnostics.raw_priors;

          SearchPiRow row;
          row.observation = provides_istate ? state->InformationStateTensor(cur)
                                            : state->ObservationTensor(cur);
          row.observation_is_information_state = provides_istate;
          row.player = cur;
          row.role = role;
          row.legal_actions = res.diagnostics.actions;
          row.raw_policy.assign(raw_priors.begin(), raw_priors.end());
          row.raw_visits = res.diagnostics.visit_counts;
          row.target_visits = std::move(target_visits);
          row.target_probs = std::move(target_probs);
          row.chosen_action = chosen;
          row.generation = generation;
          row.episode_id = episode_id;
          row.decision_id = static_cast<int>(this_decision);
          row.simulations_completed = res.simulations_completed;
          row.simulations_requested = requested;
          row.early_exit = early_exit;
          row.inherited_root_visits = res.diagnostics.inherited_root_visits;
          row.retention_snapshots_valid =
              res.diagnostics.retention_snapshots_valid;
          row.visits_pre_search = res.diagnostics.visit_counts_pre_search;
          row.re_root_status = res.diagnostics.re_root_status;
          row.fallback = SearchPiFallback::kNone;
          row.search_fallback_reason = res.fallback_reason;
          row.root_value = res.diagnostics.root_value;
          row.q_values = res.diagnostics.q_values;

          // Coverage: recorded, never acted on. A search this lane keeps but the
          // PF-C gate would have discarded is counted, because that difference
          // is the whole behavioural change and it should be measurable.
          if (row.raw_policy.size() == row.raw_visits.size()) {
            int covered = 0;
            double covered_mass = 0.0;
            row.would_pass_legacy_coverage_gate = ComputeSearchAcceptance(
                row.raw_visits, row.raw_policy,
                static_cast<int>(row.legal_actions.size()),
                config_.telemetry_min_coverage,
                config_.telemetry_min_visits_per_action,
                config_.telemetry_min_prior_mass, &covered, &covered_mass);
            row.num_covered_actions = covered;
            row.covered_prior_mass = covered_mass;
            if (!row.would_pass_legacy_coverage_gate) {
              rs->kept_despite_legacy_gate++;
            }
          }

          FillSearchPiRowScalars(&row);
          rs->sum_target_entropy_norm += row.target_entropy_norm;
          rs->sum_raw_entropy_norm += row.raw_policy_entropy_norm;
          rs->sum_kl_target_given_raw += row.kl_target_given_raw;
          if (row.target_argmax_differs_from_raw) rs->target_argmax_overrides++;

          emitted_indices.push_back(out->size());
          out->push_back(std::move(row));
          rs->rows_emitted++;
          outcome.rows_emitted++;
          if (role == DuneDecisionRole::kLeaderSelection) {
            stats->leader_rows_emitted++;  // structurally unreachable
          }
        } else {
          rs->fallbacks++;
          outcome.fallbacks++;
        }
      }

      if (chosen == kInvalidAction) {
        // Two distinct populations land here, and they get DIFFERENT
        // temperatures on purpose:
        //   - a searched role whose search technically failed: raw-prior ARGMAX
        //     at behaviour temperature 0, so a failed search does not silently
        //     inject a sampled move into a trajectory that is meant to reflect
        //     the stronger controller;
        //   - the searched seat's non-searched decisions (leader, purchase,
        //     combat, other-optional): raw policy at T=1, matching the
        //     opponents, so those decisions are drawn from the same
        //     distribution everyone else plays.
        ActionsAndProbs prior = evaluator->Prior(*state);
        if (is_search_role) {
          chosen = RawPriorArgmaxAction(prior, legal_actions);
        } else {
          chosen = SampleRawPolicyAction(
              prior, config_.searched_seat_unsearched_temperature,
              DeriveStream(config_.seed_domain, episode_id, this_decision,
                           kStreamPiUnsearchedSeat),
              legal_actions);
        }
      }

      SPIEL_CHECK_NE(chosen, kInvalidAction);

      // --- The played-action manipulation check (Fable, 2026-08-18) ---------
      //
      // Counted HERE, on the action about to be applied to the state, and not
      // anywhere upstream: the defect this exists to catch is precisely a
      // search whose result never reaches this line. Searched roles only --
      // the narrow arm's off-scope decisions are T=1 samples and their
      // divergence from argmax measures the sampler, not the teacher.
      if (is_search_role) {
        const Action raw_ref = RawPriorArgmaxFromDiagnostics(res.diagnostics);
        if (raw_ref != kInvalidAction && chosen != raw_ref) {
          if (role == DuneDecisionRole::kAgentPrimary ||
              role == DuneDecisionRole::kAgentContinuation) {
            outcome.agent_roots_played_differs_from_raw++;
          } else {
            outcome.wide_roots_played_differs_from_raw++;
          }
        }
      }
      outcome.played_action_digest =
          MixPlayedAction(outcome.played_action_digest, this_decision, chosen);

      ControllerDecision dec;
      dec.selected_action = chosen;
      dec.raw_reference_action = chosen;
      dec.mcts_proposed_action = res.diagnostics.selected_action;
      session.CommitAction(dec);
      state->ApplyAction(chosen);
    }

    // Terminal value targets, from the engine's own Returns() and nothing else.
    const std::vector<double> returns = state->Returns();
    for (size_t idx : emitted_indices) {
      SearchPiRow& row = (*out)[idx];
      SPIEL_CHECK_GE(row.player, 0);
      SPIEL_CHECK_LT(row.player, static_cast<int>(returns.size()));
      row.value_target = returns[row.player] / config_.utility_divisor;
      row.value_target_attached = true;
    }

    // The same Returns() the value targets come from, kept per game so a paired
    // strength test has something to join on. Placement is read off the ladder
    // at divisor 1.0 because these are raw returns, not scaled targets.
    outcome.returns = returns;

    // Game-shape telemetry, from GAME TRUTH rather than the convenient reads.
    // See SearchPiGameOutcome for why each of these two has a wrong derivation
    // that looks right: the round counter has already advanced past the round
    // actually played, and the widely-copied final-VP reporting helper
    // overstates by 1 whenever a seat holds all four factions at >= 3 influence
    // without tech tile 8.
    {
      const auto* dune =
          dynamic_cast<const dune_imperium::DuneImperiumState*>(state.get());
      // A dynamic_cast rather than a down_cast, and checked: this lane is
      // game-agnostic everywhere else, and a silent nullptr deref here would
      // be a crash in the middle of a multi-hour audit.
      SPIEL_CHECK_TRUE(dune != nullptr);
      outcome.final_round = dune->GetCurrentRound() - 1;
      outcome.final_vp.resize(returns.size());
      for (int p = 0; p < static_cast<int>(returns.size()); ++p) {
        outcome.final_vp[p] = dune->FinalScoredVp(p);
      }
    }

    SPIEL_CHECK_GE(searched_seat, 0);
    SPIEL_CHECK_LT(searched_seat, static_cast<int>(returns.size()));
    outcome.searched_seat_return = returns[searched_seat];
    outcome.searched_seat_placement =
        SearchPiPlacementFromReturn(outcome.searched_seat_return, 1.0);
    stats->games_played.push_back(std::move(outcome));
  }

  config_.next_episode_id = first_episode_id + config_.games_per_generation;
  stats->next_episode_id = config_.next_episode_id;
  stats->rows_total = static_cast<int64_t>(out->size() - out_base);
  stats->collection_wall_time_s =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - wall_start)
          .count();

  // Two chains over the same rows in the same order. The legacy one is the
  // provenance tie to every generation recorded before this field existed; the
  // extended one is the only one that can carry a cross-arm row-identity claim.
  std::string chain;
  std::string ext_chain;
  for (size_t i = out_base; i < out->size(); ++i) {
    chain = ChainSearchPiTargetHash(chain, (*out)[i]);
    ext_chain = ChainSearchPiExtendedRowHash(ext_chain, (*out)[i]);
  }
  stats->target_hash_chain = chain;
  stats->extended_hash_chain = ext_chain;
}

// ---------------------------------------------------------------------------
// Dedicated learner
// ---------------------------------------------------------------------------
namespace {

enum class ParamGroup { kPolicyHead, kValueHead, kTrunk };

ParamGroup ClassifyParam(const std::string& name) {
  // Order matters: next_own_action_head does not contain "value_head", but the
  // policy test is anchored at position 0 so it cannot match a suffix.
  if (name.rfind("policy_head", 0) == 0) return ParamGroup::kPolicyHead;
  if (name.find("value_head") != std::string::npos) return ParamGroup::kValueHead;
  return ParamGroup::kTrunk;
}

struct GroupAccum {
  double dot = 0.0;
  double a2 = 0.0;
  double b2 = 0.0;
  void Add(double d, double aa, double bb) {
    dot += d;
    a2 += aa;
    b2 += bb;
  }
  bool Defined() const { return a2 > 0.0 && b2 > 0.0; }
  // 0.0 when undefined, but callers must consult Defined() before reporting it:
  // a zero denominator means "this group carries no gradient from one of the two
  // objectives", which is a different statement from "the two are orthogonal".
  double Cosine() const {
    const double denom = std::sqrt(a2) * std::sqrt(b2);
    return denom > 0.0 ? dot / denom : 0.0;
  }
};

// Mean of a per-minibatch quantity together with how many minibatches actually
// contributed one. Keeping the count separate is what lets an undefined group
// report "never measured" instead of a mean of zeros.
struct CosineMean {
  double sum = 0.0;
  int64_t n = 0;
  void Add(const GroupAccum& g) {
    if (!g.Defined()) return;
    sum += g.Cosine();
    ++n;
  }
  double Mean() const { return n > 0 ? sum / static_cast<double>(n) : 0.0; }
  bool Defined() const { return n > 0; }
};

// Population mean and sd of a flat float tensor, at double precision. sd is the
// population form (/n): these are complete descriptions of the generation's own
// rows, not estimates of a wider distribution.
void MeanSd(const torch::Tensor& flat, double* mean, double* sd) {
  const int64_t n = flat.numel();
  if (n <= 0) {
    *mean = 0.0;
    *sd = 0.0;
    return;
  }
  const torch::Tensor d = flat.to(torch::kDouble).cpu();
  const double m = d.mean().item<double>();
  *mean = m;
  *sd = n > 1 ? std::sqrt((d - m).pow(2).sum().item<double>() /
                          static_cast<double>(n))
              : 0.0;
}

// One no-grad forward over every row, returning the critic's scalar prediction
// per row. Chunked at the learner's own minibatch size so the measurement never
// has a larger memory profile than the training step it brackets.
//
// No RNG is consumed and no autocast guard is entered, so inserting this pass
// cannot perturb the trained result -- which matters, because the anchor arm
// has to reproduce a run recorded before this instrument existed.
torch::Tensor PredictValues(
    const std::shared_ptr<SharedDunePolicyValueNetImpl>& model,
    const torch::Tensor& states, int64_t chunk) {
  torch::NoGradGuard no_grad;
  const int64_t n = states.size(0);
  std::vector<torch::Tensor> parts;
  for (int64_t start = 0; start < n; start += chunk) {
    const int64_t len = std::min<int64_t>(chunk, n - start);
    parts.push_back(
        model->forward(states.narrow(0, start, len)).values.squeeze(1).detach());
  }
  return parts.empty() ? torch::zeros({0}) : torch::cat(parts, 0);
}

}  // namespace

SearchPiLearnerStats RunSearchPiLearner(
    std::shared_ptr<SharedDunePolicyValueNetImpl> model,
    torch::optim::Optimizer& optimizer, const std::vector<SearchPiRow>& rows,
    int64_t obs_size, int64_t action_dim, torch::Device device,
    uint64_t master_seed, int generation, const SearchPiLearnerConfig& cfg) {
  SearchPiLearnerStats stats;
  stats.distinct_rows = static_cast<int64_t>(rows.size());
  // Structural: this function never builds a PPO surrogate. The field exists so
  // a test can assert the negative directly instead of inferring it.
  stats.ppo_policy_loss_constructed = false;
  if (rows.empty()) return stats;

  SPIEL_CHECK_GT(cfg.minibatch_size, 0);
  SPIEL_CHECK_GT(cfg.epochs, 0);

  const int64_t n = static_cast<int64_t>(rows.size());
  auto f32 = torch::TensorOptions().dtype(torch::kFloat32);
  auto boolopt = torch::TensorOptions().dtype(torch::kBool);

  torch::Tensor states = torch::zeros({n, obs_size}, f32);
  torch::Tensor masks = torch::zeros({n, action_dim}, boolopt);
  torch::Tensor targets = torch::zeros({n, action_dim}, f32);
  torch::Tensor values = torch::zeros({n}, f32);
  float* sp = states.data_ptr<float>();
  bool* mp = masks.data_ptr<bool>();
  float* tp = targets.data_ptr<float>();
  float* vp = values.data_ptr<float>();

  for (int64_t i = 0; i < n; ++i) {
    const SearchPiRow& row = rows[i];
    // A row whose value target was never attached would train the critic on a
    // default-constructed 0.0. That is a silent corruption, so it is fatal.
    SPIEL_CHECK_TRUE(row.value_target_attached);
    const int64_t copy = std::min<int64_t>(
        obs_size, static_cast<int64_t>(row.observation.size()));
    if (copy > 0) {
      std::memcpy(sp + i * obs_size, row.observation.data(),
                  copy * sizeof(float));
    }
    for (size_t j = 0; j < row.legal_actions.size(); ++j) {
      const Action a = row.legal_actions[j];
      if (a >= 0 && a < action_dim) {
        mp[i * action_dim + a] = true;
        tp[i * action_dim + a] = static_cast<float>(
            j < row.target_probs.size() ? row.target_probs[j] : 0.0);
      }
    }
    vp[i] = static_cast<float>(row.value_target);
  }

  states = states.to(device);
  masks = masks.to(device);
  targets = targets.to(device);
  values = values.to(device);

  auto named = model->named_parameters();
  std::vector<ParamGroup> groups;
  std::vector<torch::Tensor> params;
  groups.reserve(named.size());
  params.reserve(named.size());
  for (const auto& item : named) {
    groups.push_back(ClassifyParam(item.key()));
    params.push_back(item.value());
  }

  // Coefficient 0 SKIPS an objective's backward; it does not scale it to zero.
  // See the header: a scaled-to-zero pass still walks the graph and installs a
  // defined all-zeros gradient, which is not the same state as no gradient, and
  // 0 * non-finite is NaN rather than 0. The skip is the elimination.
  const bool policy_active = (cfg.policy_coef != 0.0);
  const bool value_active = (cfg.value_coef != 0.0);
  stats.policy_backward_executed = policy_active;
  stats.value_backward_executed = value_active;

  // --- Value targets and the searched seat's realized outcome mix ---
  MeanSd(values, &stats.value_target_mean, &stats.value_target_sd);
  {
    // Placement recovered from the terminal ladder {+2.25, +0.25, -0.75,
    // -1.75} / utility_divisor 4 (dune_imperium.cc:2383-2447). Deduplicated to
    // one entry per EPISODE: the rows repeat a game's outcome once per
    // decision, and the count that matters is how many independent outcomes the
    // value regression actually saw.
    static const double kLadder[4] = {0.5625, 0.0625, -0.1875, -0.4375};
    std::set<int64_t> seen;
    for (const SearchPiRow& row : rows) {
      if (!seen.insert(row.episode_id).second) continue;
      ++stats.outcome_episodes;
      int rung = -1;
      for (int k = 0; k < 4; ++k) {
        if (std::abs(row.value_target - kLadder[k]) < 1e-9) {
          rung = k;
          break;
        }
      }
      // Not snapped to the nearest rung: a target off the ladder means the
      // value pipeline changed, and rounding it away would hide that.
      if (rung < 0) {
        ++stats.outcome_unmapped;
      } else {
        ++stats.outcome_placements[rung];
      }
    }
  }

  // --- Critic predictions BEFORE any optimizer step ---
  MeanSd(PredictValues(model, states, cfg.minibatch_size),
         &stats.critic_pred_mean_pre, &stats.critic_pred_sd_pre);

  CosineMean overall, g_policy, g_trunk, g_value;
  double ce_sum = 0.0, mse_sum = 0.0;
  double pol_norm_sum = 0.0, val_norm_sum = 0.0;
  // Per-group squared-norm accumulators, summed over minibatches as norms (not
  // as squares) so they average the same way the totals above do.
  double pol_trunk_sum = 0.0, pol_phead_sum = 0.0, pol_vhead_sum = 0.0;
  double val_trunk_sum = 0.0, val_phead_sum = 0.0, val_vhead_sum = 0.0;
  int64_t mb_count = 0;

  for (int epoch = 0; epoch < cfg.epochs; ++epoch) {
    const uint64_t perm_seed = dune_seed::DeriveSeed(
        master_seed, dune_seed::kDomainTrain, generation, epoch,
        kStreamPiBehavior);
    at::Generator gen = dune_seed::MakeTorchCPUGenerator(perm_seed);
    torch::Tensor perm =
        torch::randperm(n, gen,
                        torch::TensorOptions().device(torch::kCPU).dtype(
                            torch::kInt64))
            .to(device);

    for (int64_t start = 0; start < n; start += cfg.minibatch_size) {
      const int64_t end = std::min<int64_t>(start + cfg.minibatch_size, n);
      torch::Tensor idx = perm.narrow(0, start, end - start);
      torch::Tensor mb_states = states.index_select(0, idx);
      torch::Tensor mb_masks = masks.index_select(0, idx);
      torch::Tensor mb_targets = targets.index_select(0, idx);
      torch::Tensor mb_values = values.index_select(0, idx);

      // Deliberately NOT wrapped in an autocast guard. The two-pass gradient
      // decomposition below reads .grad between backwards, and autocast keeps a
      // process-lifetime weight cache that has previously gone stale across
      // exactly this kind of split. Determinism and an honest gradient split
      // matter more here than the throughput.
      auto outputs = model->forward(mb_states);
      torch::Tensor logits =
          CenterAndCapLogitsTensor(outputs.logits, mb_masks,
                                   static_cast<float>(cfg.logit_cap));
      // -1e9f, not -inf: targets are exactly 0 at illegal actions, and
      // 0 * (-inf) is NaN while 0 * (finite) is 0.
      torch::Tensor masked = logits.masked_fill(mb_masks.logical_not(), -1e9f);
      torch::Tensor logp = torch::log_softmax(masked, -1);

      // THE objective. Mean legal-action CE against the search target, plus
      // value_coef * mean MSE against the terminal return. No ratio, no clip,
      // no entropy bonus, no advantage, no KL stop.
      torch::Tensor ce = -(mb_targets * logp).sum(-1).mean();
      torch::Tensor pred_v = outputs.values.squeeze(1);
      torch::Tensor vmse = (pred_v - mb_values).pow(2).mean();

      // At exactly 1.0 no multiply node is inserted, so the anchor arm's
      // gradients are bit-identical to the pilot's, which ran before this
      // coefficient existed. The value side keeps its original expression for
      // the same reason.
      torch::Tensor policy_term =
          (cfg.policy_coef == 1.0) ? ce
                                   : static_cast<float>(cfg.policy_coef) * ce;
      torch::Tensor value_term = static_cast<float>(cfg.value_coef) * vmse;

      // Pass 1: the policy CE alone. Skipped entirely at coefficient 0, which
      // leaves every ce_grads entry undefined -- the structural elimination.
      optimizer.zero_grad();
      std::vector<torch::Tensor> ce_grads(params.size());
      if (policy_active) {
        // retain_graph only matters when a second backward follows.
        policy_term.backward(torch::Tensor(), /*retain_graph=*/value_active);
        for (size_t pi = 0; pi < params.size(); ++pi) {
          auto g = params[pi].grad();
          if (g.defined()) ce_grads[pi] = g.detach().clone();
        }
        optimizer.zero_grad();
      }

      // Pass 2: the (coefficient-scaled) value MSE alone. Symmetrically skipped.
      std::vector<torch::Tensor> v_grads(params.size());
      if (value_active) {
        value_term.backward();
        for (size_t pi = 0; pi < params.size(); ++pi) {
          auto g = params[pi].grad();
          if (g.defined()) v_grads[pi] = g.detach().clone();
        }
        optimizer.zero_grad();
      }

      // Norms, dots and the per-group cosines -- measured, not inferred.
      double pol_sq = 0.0, val_sq = 0.0;
      // Same squares, kept per group: this is what turns "the value objective
      // dominated the shared trunk" from an inference off total norms into a
      // reading.
      double pol_sq_trunk = 0.0, pol_sq_ph = 0.0, pol_sq_vh = 0.0;
      double val_sq_trunk = 0.0, val_sq_ph = 0.0, val_sq_vh = 0.0;
      GroupAccum mb_overall, mb_policy, mb_trunk, mb_value;
      for (size_t pi = 0; pi < params.size(); ++pi) {
        const bool have_a = ce_grads[pi].defined();
        const bool have_b = v_grads[pi].defined();
        if (!have_a && !have_b) continue;
        const double aa =
            have_a ? ce_grads[pi].pow(2).sum().item<double>() : 0.0;
        const double bb =
            have_b ? v_grads[pi].pow(2).sum().item<double>() : 0.0;
        const double dd = (have_a && have_b)
                              ? (ce_grads[pi] * v_grads[pi]).sum().item<double>()
                              : 0.0;
        pol_sq += aa;
        val_sq += bb;
        mb_overall.Add(dd, aa, bb);
        switch (groups[pi]) {
          case ParamGroup::kPolicyHead:
            mb_policy.Add(dd, aa, bb);
            pol_sq_ph += aa;
            val_sq_ph += bb;
            break;
          case ParamGroup::kValueHead:
            mb_value.Add(dd, aa, bb);
            pol_sq_vh += aa;
            val_sq_vh += bb;
            break;
          case ParamGroup::kTrunk:
            mb_trunk.Add(dd, aa, bb);
            pol_sq_trunk += aa;
            val_sq_trunk += bb;
            break;
        }
      }
      pol_norm_sum += std::sqrt(pol_sq);
      val_norm_sum += std::sqrt(val_sq);
      pol_trunk_sum += std::sqrt(pol_sq_trunk);
      pol_phead_sum += std::sqrt(pol_sq_ph);
      pol_vhead_sum += std::sqrt(pol_sq_vh);
      val_trunk_sum += std::sqrt(val_sq_trunk);
      val_phead_sum += std::sqrt(val_sq_ph);
      val_vhead_sum += std::sqrt(val_sq_vh);
      overall.Add(mb_overall);
      g_policy.Add(mb_policy);
      g_trunk.Add(mb_trunk);
      g_value.Add(mb_value);

      // The combined step: grads add, so the sum of the two snapshots is
      // exactly the gradient of (ce + value_coef * vmse).
      for (size_t pi = 0; pi < params.size(); ++pi) {
        const bool have_a = ce_grads[pi].defined();
        const bool have_b = v_grads[pi].defined();
        if (!have_a && !have_b) continue;
        if (have_a && have_b) {
          params[pi].mutable_grad() = ce_grads[pi] + v_grads[pi];
        } else if (have_a) {
          params[pi].mutable_grad() = ce_grads[pi];
        } else {
          params[pi].mutable_grad() = v_grads[pi];
        }
      }
      if (cfg.grad_clip_norm > 0.0) {
        torch::nn::utils::clip_grad_norm_(model->parameters(),
                                          cfg.grad_clip_norm);
      }
      optimizer.step();

      ce_sum += ce.item<double>();
      mse_sum += vmse.item<double>();
      stats.presentations += (end - start);
      ++mb_count;
    }
  }

  if (mb_count > 0) {
    const double d = static_cast<double>(mb_count);
    stats.policy_ce = ce_sum / d;
    stats.value_mse = mse_sum / d;
    stats.policy_grad_norm = pol_norm_sum / d;
    stats.value_grad_norm = val_norm_sum / d;
    stats.policy_grad_norm_trunk = pol_trunk_sum / d;
    stats.policy_grad_norm_policy_head = pol_phead_sum / d;
    stats.policy_grad_norm_value_head = pol_vhead_sum / d;  // structurally 0
    stats.value_grad_norm_trunk = val_trunk_sum / d;
    stats.value_grad_norm_policy_head = val_phead_sum / d;  // structurally 0
    stats.value_grad_norm_value_head = val_vhead_sum / d;
    // Mean of the per-minibatch cosines -- the typical alignment, not the
    // alignment of the summed gradients -- over the minibatches where the group
    // had gradient from BOTH objectives. The *_defined flags say which of these
    // four numbers is a measurement; policy-head and value-head are structurally
    // undefined, because neither objective reaches the other's head.
    stats.grad_cosine_overall = overall.Mean();
    stats.grad_cosine_policy_head = g_policy.Mean();
    stats.grad_cosine_trunk = g_trunk.Mean();
    stats.grad_cosine_value_head = g_value.Mean();
    stats.grad_cosine_overall_defined = overall.Defined();
    stats.grad_cosine_policy_head_defined = g_policy.Defined();
    stats.grad_cosine_trunk_defined = g_trunk.Defined();
    stats.grad_cosine_value_head_defined = g_value.Defined();
  }
  stats.minibatches = mb_count;

  // --- Critic predictions AFTER the last optimizer step ---
  // Paired with the pre-update pass over the SAME rows, so the difference is
  // movement on a fixed state set rather than a level on two different ones.
  MeanSd(PredictValues(model, states, cfg.minibatch_size),
         &stats.critic_pred_mean_post, &stats.critic_pred_sd_post);
  stats.critic_pred_measured = true;
  return stats;
}

// ---------------------------------------------------------------------------
// Manifest state
// ---------------------------------------------------------------------------

json::Object WriteSearchPiState(const SearchPiState& s) {
  const SearchPiConfig& c = s.config;
  const SearchPiLearnerConfig& l = s.learner;
  json::Object o;
  // v2 adds extended_hash_chain and learner_policy_coef. The reader accepts v1
  // too and defaults both, so every checkpoint the pilot wrote stays readable:
  // a run must not lose its resume path because an instrument was added.
  o["schema_version"] = static_cast<int64_t>(2);
  o["generation"] = static_cast<int64_t>(s.generation);
  o["next_episode_id"] = static_cast<int64_t>(s.next_episode_id);
  o["cum_rows"] = static_cast<int64_t>(s.cum_rows);
  o["cum_primary_rows"] = static_cast<int64_t>(s.cum_primary_rows);
  o["cum_continuation_rows"] = static_cast<int64_t>(s.cum_continuation_rows);
  o["cum_primary_simulations"] =
      static_cast<int64_t>(s.cum_primary_simulations);
  o["cum_continuation_simulations"] =
      static_cast<int64_t>(s.cum_continuation_simulations);
  o["target_hash_chain"] = s.target_hash_chain;
  o["extended_hash_chain"] = s.extended_hash_chain;

  // Every search-configuration field, so a run's own artifact answers what it
  // searched with. Stored as strings at full %.17g precision where the value is
  // a double: json.cc floors floats under 5e-7 to exactly 0.0.
  o["games_per_generation"] = static_cast<int64_t>(c.games_per_generation);
  o["primary_simulations"] = static_cast<int64_t>(c.primary_simulations);
  o["continuation_simulations"] =
      static_cast<int64_t>(c.continuation_simulations);
  o["puct_c"] = F17(c.puct_c);
  o["max_search_decision_depth"] =
      static_cast<int64_t>(c.max_search_decision_depth);
  o["use_opponent_model"] = c.use_opponent_model;
  o["opponent_model_temperature"] = F17(c.opponent_model_temperature);
  o["root_prior_temperature"] = F17(c.root_prior_temperature);
  o["utility_divisor"] = F17(c.utility_divisor);
  o["dirichlet_epsilon"] = F17(c.dirichlet_epsilon);
  o["dirichlet_alpha_total"] = F17(c.dirichlet_alpha_total);
  o["forced_playouts_k"] = F17(c.forced_playouts_k);
  o["root_noise_fpu_zero"] = c.root_noise_fpu_zero;
  o["target_sharpen_exponent"] = F17(c.target_sharpen_exponent);
  o["continuation_target"] =
      std::string(SearchPiContinuationTargetName(c.continuation_target));
  o["behavior_temperature"] = F17(c.behavior_temperature);
  o["non_search_temperature"] = F17(c.non_search_temperature);
  o["searched_seat_unsearched_temperature"] =
      F17(c.searched_seat_unsearched_temperature);
  // Recorded even though it is structurally false, so the answer to "did the
  // Leader teacher leak into this lane" is legible from the artifact.
  o["search_leader_draft"] = c.search_leader_draft;
  o["seed_domain"] = absl::StrCat(c.seed_domain);
  o["telemetry_min_coverage"] = static_cast<int64_t>(c.telemetry_min_coverage);
  o["telemetry_min_visits_per_action"] =
      static_cast<int64_t>(c.telemetry_min_visits_per_action);
  o["telemetry_min_prior_mass"] = F17(c.telemetry_min_prior_mass);

  o["learner_learning_rate"] = F17(l.learning_rate);
  o["learner_minibatch_size"] = static_cast<int64_t>(l.minibatch_size);
  o["learner_epochs"] = static_cast<int64_t>(l.epochs);
  o["learner_value_coef"] = F17(l.value_coef);
  o["learner_policy_coef"] = F17(l.policy_coef);
  o["learner_grad_clip_norm"] = F17(l.grad_clip_norm);
  o["learner_logit_cap"] = F17(l.logit_cap);
  o["learner_weight_decay"] = F17(l.weight_decay);
  o["learner_policy_weight_decay"] = F17(l.policy_weight_decay);

  o["config_fingerprint"] = SearchPiConfigFingerprint(c, l);
  return o;
}

namespace {

bool GetInt(const json::Object& o, const char* key, int64_t* out,
            std::string* err) {
  auto it = o.find(key);
  if (it == o.end() || !it->second.IsInt()) {
    if (err != nullptr) *err = absl::StrCat("missing/!int search_pi.", key);
    return false;
  }
  *out = it->second.GetInt();
  return true;
}

bool GetStr(const json::Object& o, const char* key, std::string* out,
            std::string* err) {
  auto it = o.find(key);
  if (it == o.end() || !it->second.IsString()) {
    if (err != nullptr) *err = absl::StrCat("missing/!string search_pi.", key);
    return false;
  }
  *out = it->second.GetString();
  return true;
}

bool GetBool(const json::Object& o, const char* key, bool* out,
             std::string* err) {
  auto it = o.find(key);
  if (it == o.end() || !it->second.IsBool()) {
    if (err != nullptr) *err = absl::StrCat("missing/!bool search_pi.", key);
    return false;
  }
  *out = it->second.GetBool();
  return true;
}

bool GetF17(const json::Object& o, const char* key, double* out,
            std::string* err) {
  std::string s;
  if (!GetStr(o, key, &s, err)) return false;
  *out = std::strtod(s.c_str(), nullptr);
  return true;
}

}  // namespace

bool ReadSearchPiState(const json::Object& o, SearchPiState* out,
                       std::string* error) {
  SPIEL_CHECK_TRUE(out != nullptr);
  int64_t v = 0;
  if (!GetInt(o, "schema_version", &v, error)) return false;
  if (v != 1 && v != 2) {
    if (error != nullptr) *error = "search_pi.schema_version not in {1, 2}";
    return false;
  }
  const bool v2 = (v >= 2);
  SearchPiState s;
  if (!GetInt(o, "generation", &v, error)) return false;
  s.generation = static_cast<int>(v);
  if (!GetInt(o, "next_episode_id", &s.next_episode_id, error)) return false;
  if (!GetInt(o, "cum_rows", &s.cum_rows, error)) return false;
  if (!GetInt(o, "cum_primary_rows", &s.cum_primary_rows, error)) return false;
  if (!GetInt(o, "cum_continuation_rows", &s.cum_continuation_rows, error)) {
    return false;
  }
  if (!GetInt(o, "cum_primary_simulations", &s.cum_primary_simulations, error)) {
    return false;
  }
  if (!GetInt(o, "cum_continuation_simulations", &s.cum_continuation_simulations,
              error)) {
    return false;
  }
  if (!GetStr(o, "target_hash_chain", &s.target_hash_chain, error)) return false;
  // A v1 manifest predates the extended chain. Empty is the honest value there:
  // it says "this lineage recorded no extended chain", which a resume can carry
  // forward, rather than inventing one that would chain against nothing.
  if (v2 && !GetStr(o, "extended_hash_chain", &s.extended_hash_chain, error)) {
    return false;
  }

  SearchPiConfig& c = s.config;
  if (!GetInt(o, "games_per_generation", &v, error)) return false;
  c.games_per_generation = static_cast<int>(v);
  if (!GetInt(o, "primary_simulations", &v, error)) return false;
  c.primary_simulations = static_cast<int>(v);
  if (!GetInt(o, "continuation_simulations", &v, error)) return false;
  c.continuation_simulations = static_cast<int>(v);
  if (!GetF17(o, "puct_c", &c.puct_c, error)) return false;
  if (!GetInt(o, "max_search_decision_depth", &v, error)) return false;
  c.max_search_decision_depth = static_cast<int>(v);
  if (!GetBool(o, "use_opponent_model", &c.use_opponent_model, error)) {
    return false;
  }
  if (!GetF17(o, "opponent_model_temperature", &c.opponent_model_temperature,
              error)) {
    return false;
  }
  if (!GetF17(o, "root_prior_temperature", &c.root_prior_temperature, error)) {
    return false;
  }
  if (!GetF17(o, "utility_divisor", &c.utility_divisor, error)) return false;
  if (!GetF17(o, "dirichlet_epsilon", &c.dirichlet_epsilon, error)) return false;
  if (!GetF17(o, "dirichlet_alpha_total", &c.dirichlet_alpha_total, error)) {
    return false;
  }
  if (!GetF17(o, "forced_playouts_k", &c.forced_playouts_k, error)) return false;
  if (!GetBool(o, "root_noise_fpu_zero", &c.root_noise_fpu_zero, error)) {
    return false;
  }
  if (!GetF17(o, "target_sharpen_exponent", &c.target_sharpen_exponent, error)) {
    return false;
  }
  std::string mode;
  if (!GetStr(o, "continuation_target", &mode, error)) return false;
  if (!ParseSearchPiContinuationTarget(mode, &c.continuation_target)) {
    if (error != nullptr) {
      *error = absl::StrCat("unrecognized search_pi.continuation_target: ", mode);
    }
    return false;
  }
  if (!GetF17(o, "behavior_temperature", &c.behavior_temperature, error)) {
    return false;
  }
  if (!GetF17(o, "non_search_temperature", &c.non_search_temperature, error)) {
    return false;
  }
  if (!GetF17(o, "searched_seat_unsearched_temperature",
              &c.searched_seat_unsearched_temperature, error)) {
    return false;
  }
  if (!GetBool(o, "search_leader_draft", &c.search_leader_draft, error)) {
    return false;
  }
  std::string seed_domain;
  if (!GetStr(o, "seed_domain", &seed_domain, error)) return false;
  c.seed_domain = std::strtoull(seed_domain.c_str(), nullptr, 10);
  if (!GetInt(o, "telemetry_min_coverage", &v, error)) return false;
  c.telemetry_min_coverage = static_cast<int>(v);
  if (!GetInt(o, "telemetry_min_visits_per_action", &v, error)) return false;
  c.telemetry_min_visits_per_action = static_cast<int>(v);
  if (!GetF17(o, "telemetry_min_prior_mass", &c.telemetry_min_prior_mass,
              error)) {
    return false;
  }
  c.next_episode_id = s.next_episode_id;

  SearchPiLearnerConfig& l = s.learner;
  if (!GetF17(o, "learner_learning_rate", &l.learning_rate, error)) return false;
  if (!GetInt(o, "learner_minibatch_size", &v, error)) return false;
  l.minibatch_size = static_cast<int>(v);
  if (!GetInt(o, "learner_epochs", &v, error)) return false;
  l.epochs = static_cast<int>(v);
  if (!GetF17(o, "learner_value_coef", &l.value_coef, error)) return false;
  // v1 predates the coefficient; 1.0 is what those runs actually applied, so
  // defaulting to it reconstructs their objective rather than approximating it.
  if (v2) {
    if (!GetF17(o, "learner_policy_coef", &l.policy_coef, error)) return false;
  } else {
    l.policy_coef = 1.0;
  }
  if (!GetF17(o, "learner_grad_clip_norm", &l.grad_clip_norm, error)) {
    return false;
  }
  if (!GetF17(o, "learner_logit_cap", &l.logit_cap, error)) return false;
  if (!GetF17(o, "learner_weight_decay", &l.weight_decay, error)) return false;
  if (!GetF17(o, "learner_policy_weight_decay", &l.policy_weight_decay, error)) {
    return false;
  }

  *out = s;
  return true;
}

std::string SearchPiConfigFingerprint(const SearchPiConfig& c,
                                      const SearchPiLearnerConfig& l) {
  std::ostringstream ss;
  // v2: policy_coef joined the objective. The version string moves with the
  // algorithm so a changed hash reads as a changed recipe rather than as an
  // unexplained mismatch against a v1 artifact.
  ss << "search_pi_v2"
     << "|games=" << c.games_per_generation
     << "|prim=" << c.primary_simulations
     << "|cont=" << c.continuation_simulations
     << "|puct=" << F17(c.puct_c)
     << "|depth=" << c.max_search_decision_depth
     << "|oppmodel=" << (c.use_opponent_model ? 1 : 0)
     << "|opptemp=" << F17(c.opponent_model_temperature)
     << "|rootprior=" << F17(c.root_prior_temperature)
     << "|util=" << F17(c.utility_divisor)
     << "|eps=" << F17(c.dirichlet_epsilon)
     << "|alpha=" << F17(c.dirichlet_alpha_total)
     << "|fpk=" << F17(c.forced_playouts_k)
     << "|fpu0=" << (c.root_noise_fpu_zero ? 1 : 0)
     << "|sharpen=" << F17(c.target_sharpen_exponent)
     << "|conttarget=" << SearchPiContinuationTargetName(c.continuation_target)
     << "|behtemp=" << F17(c.behavior_temperature)
     << "|nstemp=" << F17(c.non_search_temperature)
     << "|unsearched=" << F17(c.searched_seat_unsearched_temperature)
     << "|leader=" << (c.search_leader_draft ? 1 : 0)
     << "|domain=" << c.seed_domain
     << "|lr=" << F17(l.learning_rate)
     << "|mb=" << l.minibatch_size
     << "|epochs=" << l.epochs
     << "|vcoef=" << F17(l.value_coef)
     << "|pcoef=" << F17(l.policy_coef)
     << "|clip=" << F17(l.grad_clip_norm)
     << "|logitcap=" << F17(l.logit_cap)
     << "|wd=" << F17(l.weight_decay)
     << "|polwd=" << F17(l.policy_weight_decay);
  return ComputeStringSHA256(ss.str());
}

std::string ChainSearchPiTargetHash(const std::string& prev,
                                    const SearchPiRow& row) {
  std::string digest;
  const int64_t ep = row.episode_id;
  const int64_t dec = static_cast<int64_t>(row.decision_id);
  const int64_t pl = static_cast<int64_t>(row.player);
  const int64_t rl = static_cast<int64_t>(row.role);
  const int64_t act = static_cast<int64_t>(row.chosen_action);
  AppendBytes(&digest, &ep, sizeof(ep));
  AppendBytes(&digest, &dec, sizeof(dec));
  AppendBytes(&digest, &pl, sizeof(pl));
  AppendBytes(&digest, &rl, sizeof(rl));
  AppendBytes(&digest, &act, sizeof(act));
  for (Action a : row.legal_actions) {
    const int64_t v = static_cast<int64_t>(a);
    AppendBytes(&digest, &v, sizeof(v));
  }
  for (double p : row.target_probs) AppendBytes(&digest, &p, sizeof(p));
  AppendBytes(&digest, &row.value_target, sizeof(row.value_target));
  return ComputeStringSHA256(prev + digest);
}

std::string ChainSearchPiExtendedRowHash(const std::string& prev,
                                         const SearchPiRow& row) {
  std::string digest;
  // A tag, so an extended chain can never be mistaken for a legacy one even if
  // some future row shape made the two byte streams coincide.
  digest.append("spi_ext_v1");

  // Every length is hashed before its payload. Without that, {[1],[2,3]} and
  // {[1,2],[3]} serialize identically and the chain would call two different
  // rows equal.
  auto put_i64 = [&digest](int64_t v) { AppendBytes(&digest, &v, sizeof(v)); };
  auto put_f64 = [&digest](double v) { AppendBytes(&digest, &v, sizeof(v)); };

  // --- Identity ---
  put_i64(row.generation);
  put_i64(row.episode_id);
  put_i64(row.decision_id);
  put_i64(static_cast<int64_t>(row.player));
  put_i64(static_cast<int64_t>(row.role));

  // --- Model input: the exact tensor forward() consumes, and which one it is.
  // The observation is the input the legacy chain omits entirely, and it is the
  // one a "same rows" claim most depends on: identical targets attached to
  // different states is precisely the failure the assertion must catch.
  put_i64(row.observation_is_information_state ? 1 : 0);
  put_i64(static_cast<int64_t>(row.observation.size()));
  for (float f : row.observation) AppendBytes(&digest, &f, sizeof(f));

  // --- Legal actions (the mask, in its aligned form) ---
  put_i64(static_cast<int64_t>(row.legal_actions.size()));
  for (Action a : row.legal_actions) put_i64(static_cast<int64_t>(a));

  // --- Raw snapshot prior: the frozen network's own read of this state. Two
  // arms that collected from the same snapshot must agree on it exactly.
  put_i64(static_cast<int64_t>(row.raw_policy.size()));
  for (double p : row.raw_policy) put_f64(p);

  // --- Target and executed action ---
  put_i64(static_cast<int64_t>(row.target_probs.size()));
  for (double p : row.target_probs) put_f64(p);
  put_i64(static_cast<int64_t>(row.chosen_action));

  // --- Value target ---
  put_i64(row.value_target_attached ? 1 : 0);
  put_f64(row.value_target);

  // --- Search provenance: how this row was produced, not just what it says.
  // A row reached by a different budget or a re-root miss is a different row
  // even when its target happens to match.
  put_i64(row.simulations_completed);
  put_i64(row.inherited_root_visits);
  put_i64(static_cast<int64_t>(row.fallback));
  put_i64(static_cast<int64_t>(row.re_root_status.size()));
  digest.append(row.re_root_status);

  return ComputeStringSHA256(prev + digest);
}

// ---------------------------------------------------------------------------
// Telemetry sidecar
// ---------------------------------------------------------------------------

std::string SearchPiTelemetryPath(const std::string& diagnostics_path) {
  if (diagnostics_path.empty()) return "";
  const size_t slash = diagnostics_path.find_last_of('/');
  const size_t dot = diagnostics_path.find_last_of('.');
  if (dot != std::string::npos && (slash == std::string::npos || dot > slash)) {
    return diagnostics_path.substr(0, dot) + "_search_pi.jsonl";
  }
  return diagnostics_path + "_search_pi.jsonl";
}

namespace {

// A JSON string literal with the two characters that would break the sidecar
// escaped. The reason strings are a fixed identifier set today; a future one
// carrying a quote would otherwise emit a line no parser can read, and a
// telemetry file that cannot be parsed is a lost measurement.
std::string JsonQuoted(const std::string& s) {
  std::string out = "\"";
  for (char c : s) {
    if (c == '"' || c == '\\') out += '\\';
    out += c;
  }
  out += '"';
  return out;
}

void EmitRole(std::ostream& os, const char* prefix,
              const SearchPiRoleStats& r) {
  // A role with no rows emits its means as null, not 0: "no rows this
  // generation" and "mean zero" are different facts and must not print alike.
  const bool has = r.rows_emitted > 0;
  const double d = has ? static_cast<double>(r.rows_emitted) : 1.0;
  auto mean = [&](double sum) -> std::string {
    return has ? F17Json(sum / d) : std::string("null");
  };
  os << ",\"" << prefix << "_roots_seen\":" << r.roots_seen
     << ",\"" << prefix << "_searches_run\":" << r.searches_run
     << ",\"" << prefix << "_rows_emitted\":" << r.rows_emitted
     << ",\"" << prefix << "_fallbacks\":" << r.fallbacks
     << ",\"" << prefix << "_simulations_completed\":" << r.simulations_completed
     << ",\"" << prefix << "_inherited_visits\":" << r.inherited_visits
     << ",\"" << prefix << "_re_root_hits\":" << r.re_root_hits
     << ",\"" << prefix << "_re_root_misses\":" << r.re_root_misses
     << ",\"" << prefix << "_kept_despite_legacy_gate\":"
     << r.kept_despite_legacy_gate
     << ",\"" << prefix << "_mean_target_entropy_norm\":"
     << mean(r.sum_target_entropy_norm)
     << ",\"" << prefix << "_mean_raw_entropy_norm\":"
     << mean(r.sum_raw_entropy_norm)
     << ",\"" << prefix << "_mean_kl_target_given_raw\":"
     << mean(r.sum_kl_target_given_raw)
     << ",\"" << prefix << "_target_argmax_overrides\":"
     << r.target_argmax_overrides
     // --- Early exits. `simulations_requested` is what the session asked for,
     // so `_simulation_shortfall` is a subtraction and not a comparison against
     // an assumed per-root budget. A truncated search used to emit a normal row
     // and report nothing; these fields are what it now announces.
     << ",\"" << prefix << "_simulations_requested\":" << r.simulations_requested
     << ",\"" << prefix << "_searches_short_of_budget\":"
     << r.searches_short_of_budget
     << ",\"" << prefix << "_simulation_shortfall\":" << r.simulation_shortfall;
  for (int i = 0; i < kSearchPiEarlyExitCount; ++i) {
    os << ",\"" << prefix << "_early_exit_"
       << SearchPiEarlyExitName(static_cast<SearchPiEarlyExit>(i)) << "\":"
       << r.early_exit_counts[i];
  }
  // The first short search, NAMED. null rather than a sentinel when none
  // occurred: -1 for an episode id reads as data, and "there was no such root"
  // is not a root at index -1.
  // Deadline headroom beside the shortfall: a run that spent its budget with
  // 200 ms to spare and one that spent it with 7,000 ms to spare report the
  // same zero, and only the second says the guard is not about to fire.
  os << ",\"" << prefix << "_max_search_elapsed_ms\":"
     << F17Json(r.max_search_elapsed_ms)
     << ",\"" << prefix << "_mean_search_elapsed_ms\":"
     << (r.searches_run > 0
             ? F17Json(r.sum_search_elapsed_ms /
                       static_cast<double>(r.searches_run))
             : std::string("null"))
     << ",\"" << prefix << "_configured_time_limit_ms\":"
     << F17Json(r.configured_time_limit_ms);

  const bool short_seen = r.first_short_episode_id >= 0;
  auto or_null = [&](int64_t v) -> std::string {
    return short_seen ? absl::StrCat(v) : std::string("null");
  };
  os << ",\"" << prefix << "_first_short_episode_id\":"
     << or_null(r.first_short_episode_id)
     << ",\"" << prefix << "_first_short_decision_id\":"
     << or_null(r.first_short_decision_id)
     << ",\"" << prefix << "_first_short_simulations_completed\":"
     << or_null(r.first_short_simulations_completed)
     << ",\"" << prefix << "_first_short_simulations_requested\":"
     << or_null(r.first_short_simulations_requested)
     << ",\"" << prefix << "_first_short_bot_reason\":"
     << (short_seen ? JsonQuoted(r.first_short_bot_reason) : std::string("null"))
     << ",\"" << prefix << "_first_short_session_reason\":"
     << (short_seen ? JsonQuoted(r.first_short_session_reason)
                    : std::string("null"));
}

}  // namespace

void WriteSearchPiTelemetry(const std::string& diagnostics_path,
                            const SearchPiConfig& c,
                            const SearchPiLearnerConfig& l,
                            const SearchPiGenerationStats& g,
                            const SearchPiLearnerStats& s) {
  const std::string path = SearchPiTelemetryPath(diagnostics_path);
  if (path.empty()) return;
  std::ofstream os(path, std::ios::app);
  if (!os) return;
  os << "{\"generation\":" << g.generation
     << ",\"games\":" << g.games
     << ",\"first_episode_id\":" << g.first_episode_id
     << ",\"next_episode_id\":" << g.next_episode_id
     << ",\"rows_total\":" << g.rows_total
     << ",\"collection_wall_time_s\":" << F17Json(g.collection_wall_time_s)
     << ",\"inference_calls\":" << g.inference_calls
     << ",\"target_hash_chain\":\"" << g.target_hash_chain << "\""
     // The legacy chain above ties this generation to runs recorded before the
     // extended one existed; the extended chain below is the one a cross-arm
     // row-identity assertion tests.
     << ",\"extended_hash_chain\":\"" << g.extended_hash_chain << "\"";
  EmitRole(os, "primary", g.primary);
  EmitRole(os, "continuation", g.continuation);
  for (int i = 0; i < 7; ++i) {
    os << ",\"sims_role_" << i << "\":" << g.simulations_by_role[i]
       << ",\"decisions_role_" << i << "\":" << g.decisions_by_role[i];
  }
  os << ",\"leader_rows_emitted\":" << g.leader_rows_emitted
     // Learner: CE and value MSE separately, their gradient norms separately,
     // and the cosines overall and by module.
     << ",\"policy_ce\":" << F17Json(s.policy_ce)
     << ",\"value_mse\":" << F17Json(s.value_mse)
     << ",\"policy_grad_norm\":" << F17Json(s.policy_grad_norm)
     << ",\"value_grad_norm\":" << F17Json(s.value_grad_norm)
     // Each objective's norm split across the three parameter groups. The two
     // cross terms (policy at the value head, value at the policy head) are
     // structural zeros; a nonzero there would mean the grouping is wrong.
     << ",\"policy_grad_norm_trunk\":" << F17Json(s.policy_grad_norm_trunk)
     << ",\"policy_grad_norm_policy_head\":"
     << F17Json(s.policy_grad_norm_policy_head)
     << ",\"policy_grad_norm_value_head\":"
     << F17Json(s.policy_grad_norm_value_head)
     << ",\"value_grad_norm_trunk\":" << F17Json(s.value_grad_norm_trunk)
     << ",\"value_grad_norm_policy_head\":"
     << F17Json(s.value_grad_norm_policy_head)
     << ",\"value_grad_norm_value_head\":"
     << F17Json(s.value_grad_norm_value_head)
     // Whether each backward RAN. A zero norm and a skipped pass are different
     // facts and the coefficient ablation depends on telling them apart.
     << ",\"policy_backward_executed\":"
     << (s.policy_backward_executed ? "true" : "false")
     << ",\"value_backward_executed\":"
     << (s.value_backward_executed ? "true" : "false")
     // What the critic was asked to fit, and what it predicted before and
     // after. null when unmeasured, so an absent measurement never reads as 0.
     << ",\"value_target_mean\":" << F17Json(s.value_target_mean)
     << ",\"value_target_sd\":" << F17Json(s.value_target_sd)
     << ",\"critic_pred_mean_pre\":"
     << (s.critic_pred_measured ? F17Json(s.critic_pred_mean_pre)
                                : std::string("null"))
     << ",\"critic_pred_sd_pre\":"
     << (s.critic_pred_measured ? F17Json(s.critic_pred_sd_pre)
                                : std::string("null"))
     << ",\"critic_pred_mean_post\":"
     << (s.critic_pred_measured ? F17Json(s.critic_pred_mean_post)
                                : std::string("null"))
     << ",\"critic_pred_sd_post\":"
     << (s.critic_pred_measured ? F17Json(s.critic_pred_sd_post)
                                : std::string("null"))
     // The searched seat's realized outcome mix, one entry per episode.
     << ",\"outcome_first\":" << s.outcome_placements[0]
     << ",\"outcome_second\":" << s.outcome_placements[1]
     << ",\"outcome_third\":" << s.outcome_placements[2]
     << ",\"outcome_fourth\":" << s.outcome_placements[3]
     << ",\"outcome_episodes\":" << s.outcome_episodes
     << ",\"outcome_unmapped\":" << s.outcome_unmapped
     // null where the group carries gradient from only one objective, so an
     // undefined cosine can never be read as a measured orthogonality.
     << ",\"grad_cosine_overall\":"
     << (s.grad_cosine_overall_defined ? F17Json(s.grad_cosine_overall)
                                       : std::string("null"))
     << ",\"grad_cosine_policy_head\":"
     << (s.grad_cosine_policy_head_defined
             ? F17Json(s.grad_cosine_policy_head)
             : std::string("null"))
     << ",\"grad_cosine_trunk\":"
     << (s.grad_cosine_trunk_defined ? F17Json(s.grad_cosine_trunk)
                                     : std::string("null"))
     << ",\"grad_cosine_value_head\":"
     << (s.grad_cosine_value_head_defined
             ? F17Json(s.grad_cosine_value_head)
             : std::string("null"))
     << ",\"distinct_rows\":" << s.distinct_rows
     << ",\"presentations\":" << s.presentations
     << ",\"minibatches\":" << s.minibatches
     << ",\"ppo_policy_loss_constructed\":"
     << (s.ppo_policy_loss_constructed ? "true" : "false")
     // Full search configuration on every line, so a single telemetry row is
     // self-describing.
     << ",\"cfg_primary_simulations\":" << c.primary_simulations
     << ",\"cfg_continuation_simulations\":" << c.continuation_simulations
     << ",\"cfg_puct_c\":" << F17Json(c.puct_c)
     << ",\"cfg_forced_playouts_k\":" << F17Json(c.forced_playouts_k)
     << ",\"cfg_dirichlet_epsilon\":" << F17Json(c.dirichlet_epsilon)
     << ",\"cfg_target_sharpen_exponent\":" << F17Json(c.target_sharpen_exponent)
     << ",\"cfg_behavior_temperature\":" << F17Json(c.behavior_temperature)
     << ",\"cfg_continuation_target\":\""
     << SearchPiContinuationTargetName(c.continuation_target) << "\""
     << ",\"cfg_search_leader_draft\":"
     << (c.search_leader_draft ? "true" : "false")
     << ",\"cfg_learner_lr\":" << F17Json(l.learning_rate)
     << ",\"cfg_learner_value_coef\":" << F17Json(l.value_coef)
     << ",\"cfg_learner_policy_coef\":" << F17Json(l.policy_coef)
     << ",\"cfg_fingerprint\":\"" << SearchPiConfigFingerprint(c, l) << "\""
     << "}\n";
}

}  // namespace open_spiel
