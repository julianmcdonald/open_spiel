// PWO-3 Phase 1: conversion and Swordmaster enrichment strata.
//
// Implements docs/PWO3_REGISTRATION.md sections 5.1, 5.2, 5.3, 5.4 and 4.0
// exactly.
//
// Why a fork of dune_pwo2_root_corpus.cc rather than a flag on it: the PWO-2
// corpus is FROZEN (sha256 pinned, registration section 1) and must never be
// rewritten by a tool run. This binary regenerates the SAME 256 source episodes
// (128 per arm), proves they reproduce (the section 5.1 fidelity gate), and then
// mines two NEW corpora out of them. The PWO-2 selection machinery
// (registration 3.6-A: exact quota per micro-cell, outcome-blind hash ranking,
// decision_index demoted to a last-resort tie-break) is reused verbatim in
// shape; the strata, the quotas and the shortfall rule are new.
//
// NEVER uses --legacy_corpus_path semantics: that flag copies stored observation
// arrays verbatim and carries staleness forward. Every observation here is
// produced by, and re-verified against, a live replay under the frozen engine.
//
// Output: two JSON arrays whose per-root objects carry EVERY field the PWO-2
// corpus carries, plus the PWO-3 fields of registration 4.0/4.4. Note in
// particular that `stratum` keeps its PWO-2 meaning on every root -- it is the
// DECISION-ROLE stratum, never "conversion" and never "sm". The corpus a root
// was mined from is the separate `corpus` axis. Registration 4.0 registers that
// separation so an enrichment root cannot silently widen a registered estimand.

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <iterator>
#include <map>
#include <memory>
#include <numeric>
#include <random>
#include <set>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "open_spiel/abseil-cpp/absl/flags/flag.h"
#include "open_spiel/abseil-cpp/absl/flags/parse.h"
#include "open_spiel/abseil-cpp/absl/random/distributions.h"
#include "open_spiel/abseil-cpp/absl/strings/str_format.h"
#include "open_spiel/games/dune_imperium/dune_imperium.h"
#include "open_spiel/spiel.h"
#include "open_spiel/utils/json.h"
#include <torch/torch.h>

#include "dune_evaluator.h"
#include "dune_network.h"
#include "dune_pwo2_common.h"
#include "dune_pwo3_common.h"
#include "dune_search_routing.h"
#include "dune_seed_utils.h"
#include "dune_sha256.h"

ABSL_FLAG(std::string, branch_a_checkpoint,
          "artifacts/branch_a_frozen/branch_a_seed11_model_update_2450.pt",
          "Branch-A u2450 checkpoint (arm 0).");
ABSL_FLAG(std::string, u175_checkpoint,
          "calibration_results_v2/pilot300_search_seed12/ppo_model_update_25.pt",
          "u175 checkpoint (arm 1). Registration 1.2: corpus-regeneration "
          "self-play driver ONLY -- never searched, never evaluated.");
ABSL_FLAG(std::string, pwo2_corpus_path, "data/pwo2_root_corpus.json",
          "Frozen PWO-2 root corpus. Read-only: supplies the section 5.1 "
          "fidelity gate's expectations, the section 5.4 half assert, and the "
          "history hashes both new strata must not duplicate.");
ABSL_FLAG(std::string, conversion_output, "data/pwo3_conversion_corpus.json",
          "Output path, conversion stratum (registration 5.2).");
ABSL_FLAG(std::string, conversion_manifest,
          "data/pwo3_conversion_corpus_manifest.json",
          "Output manifest path, conversion stratum.");
ABSL_FLAG(std::string, sm_output, "data/pwo3_sm_corpus.json",
          "Output path, Swordmaster stratum (registration 5.3).");
ABSL_FLAG(std::string, sm_manifest, "data/pwo3_sm_corpus_manifest.json",
          "Output manifest path, Swordmaster stratum.");
ABSL_FLAG(int, branch_a_base_seed, 777000, "Base seed, Branch-A arm.");
ABSL_FLAG(int, u175_base_seed, 778000, "Base seed, u175 arm.");
ABSL_FLAG(int, episodes_per_arm, 128, "Source episodes per arm (first attempt).");
ABSL_FLAG(int, max_episodes_per_arm, 256,
          "Extra-episode budget ceiling per arm (registration 5.2 step 4).");
ABSL_FLAG(int, hidden_dim, 2048, "Hidden dimension.");
ABSL_FLAG(int, num_blocks, 8, "Residual block count.");
ABSL_FLAG(double, logit_cap, 10.0, "Evaluator logit cap.");
ABSL_FLAG(bool, nonlinear_value_head, false, "Versioned nonlinear value head.");
ABSL_FLAG(int, threads, 16, "Worker threads (episodes are independent; results are "
                            "sorted canonically, so this never affects output).");

using namespace open_spiel;

namespace {

constexpr int kNumArms = 2;
constexpr int kNumHalves = 2;
constexpr int kNumBuckets = 4;
constexpr int kNumSeats = 4;

const char* ArmName(int arm) { return arm == 0 ? "branch_a" : "u175"; }
const char* HalfName(int h) { return h == 0 ? "calibration" : "validation"; }
const char* BucketName(int b) {
  static const char* kNames[] = {"b0", "b1", "b2", "b3"};
  return kNames[b];
}
int ArmIndex(const std::string& name) {
  if (name == "branch_a") return 0;
  if (name == "u175") return 1;
  SpielFatalError("unknown source_arm in the PWO-2 corpus: " + name);
}

// Registration 5.4 / 3.1: the episodes 0..127 half assignment is IMMUTABLE. It
// is a frozen constant here, deliberately NOT a flag -- "no flag default defines
// a measurement" is satisfied more strongly by a value that cannot be passed at
// all than by one that can be passed wrongly.
constexpr int kPwo2HalfAssignSeed = 20260726;
constexpr int kPwo2BaseEpisodesPerArm = 128;
constexpr int kExtensionEpisodesPerArm = 128;  // ids 128..255

// Registered hard guards (registration 5.2). The per-episode pair below is the
// PWO-2 G12/G3 pair carried over under registration 4.0's vocabulary: "within
// this stratum" is within a DECISION-ROLE stratum (agent_primary, ...), "within
// this corpus" is within the conversion or SM corpus being written. Read the
// other way the two constraints would collapse into one and the ">= 16 distinct
// compound episode keys" guard -- which is exactly the bound implied by 32 roots
// at <= 2 per key -- would be dead.
constexpr int kGuardMinEpisodesPerCorpus = 16;
constexpr int kGuardMaxRootsPerEpisodeWithinCorpus = 2;
constexpr int kGuardMaxRootsPerEpisodeWithinStratum = 1;

constexpr int kMaxReportedMismatches = 20;   // registration 5.1
constexpr int kDecisionIndexBinWidth = 100;  // histogram bin, reporting only

// Registration 4.0: `stratum` is the DECISION-ROLE stratum, on every root,
// including enrichment roots. Empty string = a role PWO-2 never included.
const char* StratumNameForRole(DuneDecisionRole role) {
  switch (role) {
    case DuneDecisionRole::kAgentPrimary:      return "agent_primary";
    case DuneDecisionRole::kAgentContinuation: return "agent_continuation";
    case DuneDecisionRole::kPurchase:          return "purchase";
    case DuneDecisionRole::kCombatIntrigue:    return "combat_intrigue";
    default:                                   return "";
  }
}

// ---------------------------------------------------------------------------
// The two registered corpora (registration 5.2 and 5.3).
//
// `quota[b]` is the per-(arm, half, bucket) micro-cell quota and `seats[b]` is
// the within-cell seat schedule; the two always have the same size, so a cell is
// exactly one root per scheduled seat. That is the PWO-2 3.6-A shape.
// ---------------------------------------------------------------------------
struct CorpusSpec {
  const char* name;                     // "conversion" | "sm"
  uint64_t rank_tag;                    // pwo3::kTagConversionRank / kTagSmRank
  int quota[kNumBuckets];
  std::vector<std::vector<int>> seats;  // per round bucket
  int target;
  int floor;
  bool floor_is_stop;
  const char* registration_section;
};

const CorpusSpec& ConversionSpec() {
  // Registration 5.2: quota 2 per cell = 32 roots; the seat-pair schedule is
  // EXACTLY PWO-2 3.6-A's combat_intrigue schedule, so each seat appears in
  // exactly two of the four buckets.
  static const CorpusSpec kSpec = {
      "conversion", pwo3::kTagConversionRank, {2, 2, 2, 2},
      {{0, 1}, {2, 3}, {0, 2}, {1, 3}},
      /*target=*/32, /*floor=*/32, /*floor_is_stop=*/true, "5.2"};
  return kSpec;
}

const CorpusSpec& SmSpec() {
  // Registration 5.3: weighted toward the plan section 4 curriculum question
  // ("does 800-sim / 50 s search find EARLY-SM routes"), so b0 and b1 are filled
  // first. Seat totals across the stratum are 2/2/2/2 per (arm x half).
  static const CorpusSpec kSpec = {
      "sm", pwo3::kTagSmRank, {3, 3, 1, 1},
      {{1, 2, 3}, {0, 2, 3}, {0}, {1}},
      /*target=*/32, /*floor=*/16, /*floor_is_stop=*/false, "5.3"};
  return kSpec;
}

int QuotaPerArmHalf(const CorpusSpec& spec) {
  int t = 0;
  for (int b = 0; b < kNumBuckets; ++b) t += spec.quota[b];
  return t;
}

// The registered underpowered-SM outcome, verbatim from registration 5.3.
constexpr const char* kSmUnderpoweredOutcome =
    "SM slice underpowered: no curriculum decision; a plan amendment is "
    "required before any SM oversampling in PWO-4/5.";

// ---------------------------------------------------------------------------
// A mined candidate root.
// ---------------------------------------------------------------------------
struct Root {
  int arm = 0;
  int episode_id = 0;
  int decision_index = 0;
  Player player = kInvalidPlayer;
  int round = 0;
  std::string stratum;  // decision-role stratum (registration 4.0)
  DuneDecisionRole role = DuneDecisionRole::kForcedOrBookkeeping;
  std::vector<Action> history;
  std::vector<Action> legal_actions;
  std::vector<float> observation;
  std::string history_hash;
  // Registration 4.4 tags, derived from the legal set alone.
  bool conversion_legal = false;
  bool sm_legal = false;
  std::vector<Action> legal_conversion_actions;

  // Canonical order: (source_arm, source_episode_id, decision_index). Used for
  // stable merging and for the FINAL corpus ordering only -- decision_index is
  // never a ranking key (PWO-2 Amendment 1).
  bool operator<(const Root& o) const {
    if (arm != o.arm) return arm < o.arm;
    if (episode_id != o.episode_id) return episode_id < o.episode_id;
    return decision_index < o.decision_index;
  }
};

// A root paired with the rank key of the stratum currently being selected. The
// same root can be a candidate for both strata under two different tags, so the
// key cannot live on Root.
struct Candidate {
  const Root* r = nullptr;
  uint64_t rank_key = 0;
};

// Registration 5.2 within-cell ranking: lowest RankKey wins; canonical
// tie-breaks in order history_hash lexicographic, source_arm,
// source_episode_id, then decision_index as a LAST RESORT ONLY -- never a
// ranking key, which is what caused PWO-2's Amendment-1 premise failure.
bool RanksBefore(const Candidate& a, const Candidate& b) {
  if (a.rank_key != b.rank_key) return a.rank_key < b.rank_key;
  if (a.r->history_hash != b.r->history_hash)
    return a.r->history_hash < b.r->history_hash;
  if (a.r->arm != b.r->arm) return a.r->arm < b.r->arm;
  if (a.r->episode_id != b.r->episode_id)
    return a.r->episode_id < b.r->episode_id;
  return a.r->decision_index < b.r->decision_index;
}

Action SampleFromPrior(const ActionsAndProbs& prior, std::mt19937& rng) {
  if (prior.empty()) return kInvalidAction;
  std::vector<double> w;
  w.reserve(prior.size());
  for (const auto& ap : prior) w.push_back(ap.second);
  std::discrete_distribution<size_t> dist(w.begin(), w.end());
  return prior[dist(rng)].first;
}

std::unique_ptr<State> ReconstructState(const std::shared_ptr<const Game>& game,
                                        const std::vector<Action>& history) {
  auto state = game->NewInitialState();
  for (Action a : history) state->ApplyAction(a);
  return state;
}

std::shared_ptr<SharedDunePolicyValueNetImpl> LoadModel(
    const std::string& path, int64_t obs_size, int64_t action_size,
    torch::Device device) {
  auto model = std::make_shared<SharedDunePolicyValueNetImpl>(
      obs_size, absl::GetFlag(FLAGS_hidden_dim), action_size,
      absl::GetFlag(FLAGS_num_blocks), absl::GetFlag(FLAGS_nonlinear_value_head));
  try {
    torch::load(model, path, device);
  } catch (const std::exception& e) {
    std::cerr << "Failed to load checkpoint " << path << ": " << e.what() << "\n";
    std::exit(1);
  }
  model->to(device);
  model->eval();
  return model;
}

// ---------------------------------------------------------------------------
// The frozen PWO-2 corpus, read-only.
// ---------------------------------------------------------------------------
struct Pwo2Row {
  int arm = 0;
  int episode_id = 0;
  int decision_index = 0;
  int round = 0;
  Player player = kInvalidPlayer;
  std::string stratum;
  std::string half;
  std::string history_hash;
};

std::vector<Pwo2Row> LoadPwo2Corpus(const std::string& path) {
  std::ifstream f(path);
  if (!f) SpielFatalError("cannot open --pwo2_corpus_path: " + path);
  std::string content((std::istreambuf_iterator<char>(f)),
                      std::istreambuf_iterator<char>());
  auto parsed = json::FromString(content);
  if (!parsed) SpielFatalError("PWO-2 corpus is not valid JSON: " + path);
  std::vector<Pwo2Row> rows;
  for (const auto& v : parsed.value().GetArray()) {
    const auto& o = v.GetObject();
    Pwo2Row r;
    r.arm = ArmIndex(o.at("source_arm").GetString());
    r.episode_id = static_cast<int>(o.at("source_episode_id").GetInt());
    r.decision_index = static_cast<int>(o.at("decision_index").GetInt());
    r.round = static_cast<int>(o.at("round").GetInt());
    r.player = static_cast<Player>(o.at("player").GetInt());
    r.stratum = o.at("stratum").GetString();
    r.half = o.at("half").GetString();
    r.history_hash = o.at("history_hash").GetString();
    // Fail closed: the frozen corpus's own hash must still be self-consistent
    // with its stored history under this build.
    std::vector<Action> history;
    for (const auto& a : o.at("history").GetArray())
      history.push_back(static_cast<Action>(a.GetInt()));
    SPIEL_CHECK_EQ(pwo2::HistoryHash(history), r.history_hash);
    rows.push_back(std::move(r));
  }
  return rows;
}

// ---------------------------------------------------------------------------
// Registration 5.4: half assignment, in TWO independently seeded blocks.
//
// PWO-2's AssignHalves(n, seed) shuffles ids 0..n-1, so calling it with n = 256
// would silently REASSIGN the original 128. Both blocks are therefore always
// built in full and the extension block never touches ids 0..127.
// ---------------------------------------------------------------------------
std::map<std::pair<int, int>, std::string> BuildHalves() {
  std::map<std::pair<int, int>, std::string> half;
  for (int arm = 0; arm < kNumArms; ++arm) {
    // Block 1, ids 0..127: IMMUTABLE. Reproduces PWO-2's
    // AssignHalves(128, 20260726) construction exactly.
    std::vector<int> ids(kPwo2BaseEpisodesPerArm);
    std::iota(ids.begin(), ids.end(), 0);
    auto rng = dune_seed::MakeRng64(dune_seed::DeriveSeed(kPwo2HalfAssignSeed, arm));
    std::shuffle(ids.begin(), ids.end(), rng);
    for (int i = 0; i < kPwo2BaseEpisodesPerArm; ++i) {
      half[{arm, ids[i]}] =
          (i < kPwo2BaseEpisodesPerArm / 2) ? "calibration" : "validation";
    }

    // Block 2, ids 128..255: seeded independently under the registered
    // extension tag. Always built in full so that the assignment of any given
    // extension id does not depend on how many extension episodes were used.
    std::vector<int> ext(kExtensionEpisodesPerArm);
    std::iota(ext.begin(), ext.end(), kPwo2BaseEpisodesPerArm);
    auto ext_rng = dune_seed::MakeRng64(dune_seed::DeriveSeed(
        pwo3::kNewSelectionSeed, pwo3::kTagExtensionHalfBase + arm));
    std::shuffle(ext.begin(), ext.end(), ext_rng);
    for (int i = 0; i < kExtensionEpisodesPerArm; ++i) {
      half[{arm, ext[i]}] =
          (i < kExtensionEpisodesPerArm / 2) ? "calibration" : "validation";
    }
  }
  return half;
}

// ---------------------------------------------------------------------------
// Selection (registration 5.2, reused unchanged by 5.3).
//
// Stage 1 is the PWO-2 3.6-A machinery: one root per (arm, half, bucket, seat)
// slot, chosen by lowest RankKey among the admissible candidates at those exact
// coordinates.
//
// Stage 2 is the registered shortfall rule, AS AMENDED (registration 5.2 step
// 1, amended 2026-07-28 before any data existed). A slot that cannot fill
// redistributes its quota WITHIN the same (arm, half) pair, donors ordered by
// |b' - b| ascending then b' ascending -- which puts the shortfall's OWN bucket
// FIRST, at distance 0, and only then the neighbouring buckets.
//
//   - Distance-0 donor (b' == b): the extra root is drawn from a seat OUTSIDE
//     that cell's registered seat schedule. The scheduled seats are by
//     definition already consumed or unavailable, so this relaxes SEAT while
//     holding the round bucket fixed.
//   - Every other donor (b' != b): ascending RankKey over ALL its admissible
//     candidates REGARDLESS OF SEAT.
//
// Why the amendment: a cell that fails only because one SEAT is unavailable
// must not surrender its root to a neighbouring ROUND bucket. The round
// distribution is the axis both enrichment strata exist to control -- the SM
// stratum's entire point is its b0/b1 weighting. Seat balance is the cheaper
// thing to relax: it is reported, never gated.
//
// Arm and half totals stay exact either way. The achieved seat histogram and
// the bucket histogram are both emitted unconditionally, so any relaxation the
// donors forced is visible rather than smoothed over.
// ---------------------------------------------------------------------------
struct CellAvail {
  int arm = 0, half = 0, bucket = 0;
  int quota = 0;
  int n_available = 0;              // candidates at these cell coordinates, any seat
  int n_admissible_remaining = 0;   // ... still admissible after all selection
  int seat_available[kNumSeats] = {0, 0, 0, 0};
  int n_scheduled_fills = 0;        // roots taken at this cell's scheduled seats
  int n_offschedule_fills = 0;      // own quota slots filled HERE, off-schedule
                                    // seat (the amended distance-0 donor)
  int n_extra_as_donor = 0;         // roots taken here to cover ANOTHER bucket
  int n_quota_exported = 0;         // own quota slots covered by other buckets
  int n_quota_unmet = 0;            // own quota slots no donor could cover
  std::vector<int> unfilled_seats;
};

struct SlotShortfall {
  int arm = 0, half = 0, bucket = 0, seat = 0;
  int n_available = 0;   // candidates matching the slot coordinates
  int n_admissible = 0;  // ... that also passed the episode/duplicate constraints
};

struct Redistribution {
  int arm = 0, half = 0, donor_bucket = 0, recipient_bucket = 0, count = 0;
};

struct SelectResult {
  std::vector<const Root*> selected;
  std::vector<CellAvail> cells;
  std::vector<SlotShortfall> shortfalls;
  std::vector<Redistribution> redistributions;
  int n_candidates = 0;
  int arm_half_total[kNumArms][kNumHalves] = {{0, 0}, {0, 0}};
  bool arm_half_complete = false;  // every (arm, half) reached its quota total
};

SelectResult Select(const CorpusSpec& spec, const std::vector<Root>& all,
                    const std::function<bool(const Root&)>& eligible,
                    const std::map<std::pair<int, int>, std::string>& halves,
                    const std::set<std::string>& excluded_hashes) {
  SelectResult sr;

  // Candidate list in canonical corpus order, annotated with THIS stratum's
  // rank key. Order matters only for the deterministic tie-break; RanksBefore
  // is a total order on distinct roots.
  std::vector<Candidate> cands;
  for (const Root& r : all) {
    if (!eligible(r)) continue;
    cands.push_back({&r, pwo3::RankKey(spec.rank_tag, r.history_hash)});
  }
  sr.n_candidates = static_cast<int>(cands.size());

  // Live selection state. `used` starts loaded with every hash this corpus is
  // forbidden to duplicate (the PWO-2 corpus, and the other new stratum when it
  // has already been selected), so cross-corpus duplication is impossible by
  // construction and is re-checked as a guard afterwards anyway.
  std::set<std::string> used = excluded_hashes;
  std::map<std::pair<int, int>, int> per_ep_corpus;                        // <= 2
  std::map<std::pair<std::string, std::pair<int, int>>, int> per_ep_stratum;  // <= 1

  auto admissible = [&](const Candidate& c) {
    if (used.count(c.r->history_hash)) return false;
    const std::pair<int, int> ep{c.r->arm, c.r->episode_id};
    auto it = per_ep_corpus.find(ep);
    if (it != per_ep_corpus.end() &&
        it->second >= kGuardMaxRootsPerEpisodeWithinCorpus)
      return false;
    auto it2 = per_ep_stratum.find({c.r->stratum, ep});
    if (it2 != per_ep_stratum.end() &&
        it2->second >= kGuardMaxRootsPerEpisodeWithinStratum)
      return false;
    return true;
  };
  auto take = [&](const Candidate& c) {
    used.insert(c.r->history_hash);
    per_ep_corpus[{c.r->arm, c.r->episode_id}]++;
    per_ep_stratum[{c.r->stratum, {c.r->arm, c.r->episode_id}}]++;
    sr.selected.push_back(c.r);
  };
  auto in_cell = [&](const Candidate& c, int arm, int half, int bucket) {
    return c.r->arm == arm && pwo3::RoundBucket(c.r->round) == bucket &&
           halves.at({c.r->arm, c.r->episode_id}) == HalfName(half);
  };
  // Used only by the amended distance-0 donor, which draws from OUTSIDE the
  // cell's registered seat schedule.
  auto seat_is_scheduled = [&](int bucket, int seat) {
    const std::vector<int>& s = spec.seats[bucket];
    return std::find(s.begin(), s.end(), seat) != s.end();
  };

  // Availability pre-pass. Independent of selection order, so it is a fixed
  // property of the collected episodes and can be trusted in a STOP report.
  auto cell_at = [&](int arm, int half, int bucket) -> CellAvail& {
    return sr.cells[((arm * kNumHalves) + half) * kNumBuckets + bucket];
  };
  sr.cells.resize(kNumArms * kNumHalves * kNumBuckets);
  for (int arm = 0; arm < kNumArms; ++arm)
    for (int half = 0; half < kNumHalves; ++half)
      for (int bucket = 0; bucket < kNumBuckets; ++bucket) {
        CellAvail& cell = cell_at(arm, half, bucket);
        cell.arm = arm;
        cell.half = half;
        cell.bucket = bucket;
        cell.quota = spec.quota[bucket];
        for (const auto& c : cands) {
          if (!in_cell(c, arm, half, bucket)) continue;
          ++cell.n_available;
          ++cell.seat_available[c.r->player];
        }
      }

  // ---- Stage 1: exact quota per (cell, seat) slot, deterministic cell order.
  for (int arm = 0; arm < kNumArms; ++arm) {
    for (int half = 0; half < kNumHalves; ++half) {
      for (int bucket = 0; bucket < kNumBuckets; ++bucket) {
        CellAvail& cell = cell_at(arm, half, bucket);
        const std::vector<int>& seats = spec.seats[bucket];
        SPIEL_CHECK_EQ(static_cast<int>(seats.size()), spec.quota[bucket]);
        for (int seat : seats) {
          const Candidate* best = nullptr;
          int n_available = 0, n_admissible = 0;
          for (const auto& c : cands) {
            if (!in_cell(c, arm, half, bucket)) continue;
            if (c.r->player != seat) continue;
            ++n_available;
            if (!admissible(c)) continue;
            ++n_admissible;
            if (best == nullptr || RanksBefore(c, *best)) best = &c;
          }
          if (best == nullptr) {
            sr.shortfalls.push_back({arm, half, bucket, seat, n_available,
                                     n_admissible});
            cell.unfilled_seats.push_back(seat);
            continue;
          }
          take(*best);
          ++cell.n_scheduled_fills;
        }
      }
    }
  }

  // ---- Stage 2: registered shortfall redistribution. Deficits are snapshotted
  // from stage 1 BEFORE any donation, so a donor cell's own deficit cannot be
  // silently erased by the extra root it just supplied for a neighbour -- and,
  // under the amended rule, so that a cell filling its OWN deficit off-schedule
  // (the distance-0 donor) does not re-read a deficit it has just reduced.
  int deficit[kNumArms][kNumHalves][kNumBuckets];
  for (int arm = 0; arm < kNumArms; ++arm)
    for (int half = 0; half < kNumHalves; ++half)
      for (int bucket = 0; bucket < kNumBuckets; ++bucket)
        deficit[arm][half][bucket] =
            spec.quota[bucket] - cell_at(arm, half, bucket).n_scheduled_fills;

  for (int arm = 0; arm < kNumArms; ++arm) {
    for (int half = 0; half < kNumHalves; ++half) {
      for (int b = 0; b < kNumBuckets; ++b) {
        // Amended registered donor order: nearest bucket first, ties toward
        // earlier buckets -- and b ITSELF is the first donor, at distance 0.
        std::vector<int> donors;
        for (int d = 0; d < kNumBuckets; ++d) donors.push_back(d);
        std::stable_sort(donors.begin(), donors.end(), [b](int x, int y) {
          const int dx = std::abs(x - b), dy = std::abs(y - b);
          if (dx != dy) return dx < dy;
          return x < y;
        });
        SPIEL_CHECK_EQ(donors.front(), b);  // the amendment, asserted

        for (int unit = 0; unit < deficit[arm][half][b]; ++unit) {
          const Candidate* best = nullptr;
          int donor_bucket = -1;
          for (int d : donors) {
            for (const auto& c : cands) {
              if (!in_cell(c, arm, half, d)) continue;
              // At the distance-0 donor the extra root must come from a seat
              // OUTSIDE this cell's registered seat schedule: the scheduled
              // seats are already consumed or unavailable, and taking a second
              // root at a scheduled seat is not what the amendment authorises.
              // At every other donor all seats are eligible.
              if (d == b && seat_is_scheduled(d, c.r->player)) continue;
              if (!admissible(c)) continue;
              if (best == nullptr || RanksBefore(c, *best)) best = &c;
            }
            if (best != nullptr) {
              donor_bucket = d;
              break;  // nearest donor that can supply wins; do not shop around
            }
          }
          if (best == nullptr) {
            cell_at(arm, half, b).n_quota_unmet++;
            continue;
          }
          take(*best);
          if (donor_bucket == b) {
            // Seat relaxed, round bucket held: nothing left this cell.
            cell_at(arm, half, b).n_offschedule_fills++;
          } else {
            cell_at(arm, half, donor_bucket).n_extra_as_donor++;
            cell_at(arm, half, b).n_quota_exported++;
          }
          // One row per redistributed root, INCLUDING the distance-0 ones
          // (donor == recipient): the manifest carries the full ledger, not a
          // summary, and "the seat schedule was relaxed here" is exactly the
          // fact a reader needs to see.
          sr.redistributions.push_back({arm, half, donor_bucket, b, 1});
        }
      }
    }
  }

  // ---- Final admissibility remaining, for the STOP report: "how many usable
  // candidates are left in this cell now", not "how many there were".
  for (auto& cell : sr.cells) {
    for (const auto& c : cands) {
      if (!in_cell(c, cell.arm, cell.half, cell.bucket)) continue;
      if (!admissible(c)) continue;
      ++cell.n_admissible_remaining;
    }
  }

  for (const Root* r : sr.selected) {
    const int h = halves.at({r->arm, r->episode_id}) == "calibration" ? 0 : 1;
    sr.arm_half_total[r->arm][h]++;
  }
  const int want = QuotaPerArmHalf(spec);
  sr.arm_half_complete = true;
  for (int arm = 0; arm < kNumArms; ++arm)
    for (int half = 0; half < kNumHalves; ++half)
      if (sr.arm_half_total[arm][half] != want) sr.arm_half_complete = false;

  std::sort(sr.selected.begin(), sr.selected.end(),
            [](const Root* a, const Root* b) { return *a < *b; });
  return sr;
}

void PrintAvailability(std::ostream& os, const CorpusSpec& spec,
                       const SelectResult& sr) {
  for (const auto& c : sr.cells) {
    os << absl::StrFormat(
        "    CELL %-11s arm=%-9s half=%-11s %s  quota=%d roots=%d "
        "(sched=%d offsched=%d donated_out=%d) exported=%d unmet=%d  "
        "available=%d admissible_left=%d  by_seat= %d/%d/%d/%d\n",
        spec.name, ArmName(c.arm), HalfName(c.half), BucketName(c.bucket),
        c.quota,
        c.n_scheduled_fills + c.n_offschedule_fills + c.n_extra_as_donor,
        c.n_scheduled_fills, c.n_offschedule_fills, c.n_extra_as_donor,
        c.n_quota_exported, c.n_quota_unmet, c.n_available,
        c.n_admissible_remaining, c.seat_available[0], c.seat_available[1],
        c.seat_available[2], c.seat_available[3]);
  }
  for (const auto& s : sr.shortfalls) {
    os << absl::StrFormat(
        "    SLOT SHORTFALL %-11s arm=%-9s half=%-11s %s seat=%d -> "
        "available=%d admissible=%d\n",
        spec.name, ArmName(s.arm), HalfName(s.half), BucketName(s.bucket),
        s.seat, s.n_available, s.n_admissible);
  }
  for (const auto& r : sr.redistributions) {
    os << absl::StrFormat(
        "    REDISTRIBUTION %-11s arm=%-9s half=%-11s donor=%s -> recipient=%s "
        "count=%d  %s\n",
        spec.name, ArmName(r.arm), HalfName(r.half), BucketName(r.donor_bucket),
        BucketName(r.recipient_bucket), r.count,
        r.donor_bucket == r.recipient_bucket ? "SEAT_RELAXED_SAME_BUCKET"
                                             : "CROSS_BUCKET");
  }
}

// ---------------------------------------------------------------------------
// Guards (registration 5.2; the same set is computed for the SM stratum so a
// floor miss is never confused with a structural violation).
//
// `structural_pass` is the subset that must hold no matter how many roots were
// found: no duplicate hashes anywhere, the two per-episode caps, and the replay
// verification. Those drive the exit code for BOTH strata. The count-shaped
// guards (target, arm balance, half balance, minimum distinct episodes) drive
// the exit code for the conversion stratum only -- registration 5.3 pre-declares
// that an underpowered SM slice is a reported outcome, not a stop.
// ---------------------------------------------------------------------------
struct GuardSummary {
  int n = 0;
  int per_arm[kNumArms] = {0, 0};
  int per_half[kNumHalves] = {0, 0};
  int n_distinct_episodes = 0;
  int max_roots_per_episode = 0;
  int max_roots_per_episode_within_stratum = 0;
  int n_shared_episodes_with_pwo2 = 0;  // permitted; recorded (registration 5.2)

  bool g_count_target = false;
  bool g_arm_balance = false;
  bool g_half_balance = false;
  bool g_min_episodes = false;
  bool g_max_per_episode_within_corpus = false;
  bool g_max_per_episode_within_stratum = false;
  bool g_no_duplicate_within = false;
  bool g_no_duplicate_vs_pwo2 = false;
  bool g_no_duplicate_vs_other_stratum = false;
  bool g_replay_verified = false;  // set true only after the replay block runs

  bool structural_pass = false;
  bool all_pass = false;
};

GuardSummary ComputeGuards(
    const CorpusSpec& spec, const std::vector<const Root*>& sel,
    const std::map<std::pair<int, int>, std::string>& halves,
    const std::set<std::string>& pwo2_hashes,
    const std::set<std::pair<int, int>>& pwo2_episodes,
    const std::set<std::string>& other_stratum_hashes) {
  GuardSummary g;
  g.n = static_cast<int>(sel.size());

  std::set<std::string> hashes;
  std::map<std::pair<int, int>, int> per_ep;
  std::map<std::pair<std::string, std::pair<int, int>>, int> per_ep_stratum;
  bool dup_within = false, dup_pwo2 = false, dup_other = false;
  for (const Root* r : sel) {
    ++g.per_arm[r->arm];
    g.per_half[halves.at({r->arm, r->episode_id}) == "calibration" ? 0 : 1]++;
    if (!hashes.insert(r->history_hash).second) dup_within = true;
    if (pwo2_hashes.count(r->history_hash)) dup_pwo2 = true;
    if (other_stratum_hashes.count(r->history_hash)) dup_other = true;
    per_ep[{r->arm, r->episode_id}]++;
    per_ep_stratum[{r->stratum, {r->arm, r->episode_id}}]++;
  }
  g.n_distinct_episodes = static_cast<int>(per_ep.size());
  for (const auto& [k, c] : per_ep) {
    g.max_roots_per_episode = std::max(g.max_roots_per_episode, c);
    if (pwo2_episodes.count(k)) ++g.n_shared_episodes_with_pwo2;
  }
  for (const auto& [k, c] : per_ep_stratum)
    g.max_roots_per_episode_within_stratum =
        std::max(g.max_roots_per_episode_within_stratum, c);

  const int want_half = spec.target / kNumHalves;
  const int want_arm = spec.target / kNumArms;
  g.g_count_target = (g.n == spec.target);
  g.g_arm_balance = (g.per_arm[0] == want_arm && g.per_arm[1] == want_arm);
  g.g_half_balance = (g.per_half[0] == want_half && g.per_half[1] == want_half);
  g.g_min_episodes = (g.n_distinct_episodes >= kGuardMinEpisodesPerCorpus);
  g.g_max_per_episode_within_corpus =
      (g.max_roots_per_episode <= kGuardMaxRootsPerEpisodeWithinCorpus);
  g.g_max_per_episode_within_stratum =
      (g.max_roots_per_episode_within_stratum <=
       kGuardMaxRootsPerEpisodeWithinStratum);
  g.g_no_duplicate_within = !dup_within;
  g.g_no_duplicate_vs_pwo2 = !dup_pwo2;
  g.g_no_duplicate_vs_other_stratum = !dup_other;
  return g;
}

void FinalizeGuards(GuardSummary& g) {
  g.structural_pass = g.g_max_per_episode_within_corpus &&
                      g.g_max_per_episode_within_stratum &&
                      g.g_no_duplicate_within && g.g_no_duplicate_vs_pwo2 &&
                      g.g_no_duplicate_vs_other_stratum && g.g_replay_verified;
  g.all_pass = g.structural_pass && g.g_count_target && g.g_arm_balance &&
               g.g_half_balance && g.g_min_episodes;
}

json::Object GuardObject(const GuardSummary& g, const CorpusSpec& spec) {
  json::Object o;
  o["n_roots"] = static_cast<int64_t>(g.n);
  o["target"] = static_cast<int64_t>(spec.target);
  o["floor"] = static_cast<int64_t>(spec.floor);
  o["n_roots_branch_a"] = static_cast<int64_t>(g.per_arm[0]);
  o["n_roots_u175"] = static_cast<int64_t>(g.per_arm[1]);
  o["n_roots_calibration"] = static_cast<int64_t>(g.per_half[0]);
  o["n_roots_validation"] = static_cast<int64_t>(g.per_half[1]);
  o["n_distinct_compound_episodes"] =
      static_cast<int64_t>(g.n_distinct_episodes);
  o["max_roots_per_compound_episode_within_corpus"] =
      static_cast<int64_t>(g.max_roots_per_episode);
  o["max_roots_per_compound_episode_within_stratum"] =
      static_cast<int64_t>(g.max_roots_per_episode_within_stratum);
  o["n_compound_episodes_shared_with_pwo2_corpus"] =
      static_cast<int64_t>(g.n_shared_episodes_with_pwo2);
  o["G_exact_target_roots"] = g.g_count_target;
  o["G_arm_balance_exact"] = g.g_arm_balance;
  o["G_half_balance_exact"] = g.g_half_balance;
  o["G_min_16_distinct_compound_episodes"] = g.g_min_episodes;
  o["G_max_2_roots_per_compound_episode_within_corpus"] =
      g.g_max_per_episode_within_corpus;
  o["G_max_1_root_per_compound_episode_within_stratum"] =
      g.g_max_per_episode_within_stratum;
  o["G_no_duplicate_history_hash_within_corpus"] = g.g_no_duplicate_within;
  o["G_no_duplicate_history_hash_vs_pwo2_corpus"] = g.g_no_duplicate_vs_pwo2;
  o["G_no_duplicate_history_hash_vs_other_new_stratum"] =
      g.g_no_duplicate_vs_other_stratum;
  o["G6_G7_all_roots_replay_verified"] = g.g_replay_verified;
  o["STRUCTURAL_GUARDS_PASS"] = g.structural_pass;
  o["ALL_GUARDS_PASS"] = g.all_pass;
  return o;
}

// ---------------------------------------------------------------------------
// Distributions.
//
// AMENDMENT-1 LESSON, ENFORCED: these are computed and emitted to BOTH the
// manifest and stdout UNCONDITIONALLY -- even when every guard passes. PWO-2's
// rejected corpus passed every guard it had; the round/seat confound was
// invisible only because nothing made anyone look. Guards only catch what they
// encode, so the histograms are not conditioned on a guard failing.
// ---------------------------------------------------------------------------
struct Dist {
  std::map<int, int> round;          // exact round
  std::map<int, int> bucket;
  std::map<int, int> seat;
  std::map<int, int> n_legal;
  std::map<int, int> n_legal_conversion_actions;
  std::map<int, int> decision_index_bin;
  std::map<std::string, int> stratum;  // decision-role stratum
  std::vector<int> decision_index;     // sorted
  int n_conversion_legal = 0;
  int n_sm_legal = 0;
};

Dist ComputeDist(const std::vector<const Root*>& sel) {
  Dist d;
  for (const Root* r : sel) {
    d.round[r->round]++;
    d.bucket[pwo3::RoundBucket(r->round)]++;
    d.seat[r->player]++;
    d.n_legal[static_cast<int>(r->legal_actions.size())]++;
    d.n_legal_conversion_actions[static_cast<int>(
        r->legal_conversion_actions.size())]++;
    d.decision_index_bin[(r->decision_index / kDecisionIndexBinWidth) *
                         kDecisionIndexBinWidth]++;
    d.stratum[r->stratum]++;
    d.decision_index.push_back(r->decision_index);
    if (r->conversion_legal) ++d.n_conversion_legal;
    if (r->sm_legal) ++d.n_sm_legal;
  }
  std::sort(d.decision_index.begin(), d.decision_index.end());
  return d;
}

json::Object IntHistToJson(const std::map<int, int>& h) {
  json::Object o;
  for (const auto& [k, c] : h) o[std::to_string(k)] = static_cast<int64_t>(c);
  return o;
}

void AddDistToManifest(json::Object& man, const Dist& d) {
  man["round_histogram"] = IntHistToJson(d.round);
  json::Object bh;
  for (const auto& [b, c] : d.bucket) bh[BucketName(b)] = static_cast<int64_t>(c);
  man["round_bucket_histogram"] = bh;
  man["seat_histogram"] = IntHistToJson(d.seat);
  man["n_legal_histogram"] = IntHistToJson(d.n_legal);
  man["n_legal_conversion_actions_histogram"] =
      IntHistToJson(d.n_legal_conversion_actions);
  json::Object dib;
  for (const auto& [lo, c] : d.decision_index_bin)
    dib[absl::StrFormat("%d-%d", lo, lo + kDecisionIndexBinWidth - 1)] =
        static_cast<int64_t>(c);
  man["decision_index_histogram"] = dib;
  json::Object sh;
  for (const auto& [s, c] : d.stratum) sh[s] = static_cast<int64_t>(c);
  man["decision_role_stratum_histogram"] = sh;
  man["n_conversion_legal_roots"] = static_cast<int64_t>(d.n_conversion_legal);
  man["n_sm_legal_roots"] = static_cast<int64_t>(d.n_sm_legal);
  json::Array div;
  for (int v : d.decision_index) div.push_back(static_cast<int64_t>(v));
  man["decision_index_values_sorted"] = div;
  json::Object dio;
  if (!d.decision_index.empty()) {
    const auto& v = d.decision_index;
    dio["min"] = static_cast<int64_t>(v.front());
    dio["p25"] = static_cast<int64_t>(v[v.size() / 4]);
    dio["median"] = static_cast<int64_t>(v[v.size() / 2]);
    dio["p75"] = static_cast<int64_t>(v[3 * v.size() / 4]);
    dio["max"] = static_cast<int64_t>(v.back());
  }
  man["decision_index"] = dio;
  man["distribution_reporting_note"] = std::string(
      "Amendment-1 lesson: these histograms are emitted UNCONDITIONALLY, "
      "including when every guard passes. PWO-2's rejected corpus passed every "
      "guard it had and the round/seat confound was invisible only because "
      "nothing made anyone look.");
}

void PrintDist(std::ostream& os, const char* label, const Dist& d) {
  os << "\nDistributions for the " << label
     << " corpus (ALWAYS reported, guards or no guards):\n  round buckets :";
  for (const auto& [b, c] : d.bucket) os << " " << BucketName(b) << "=" << c;
  os << "\n  exact rounds  :";
  for (const auto& [r, c] : d.round) os << " " << r << ":" << c;
  os << "\n  seats         :";
  for (const auto& [s, c] : d.seat) os << " " << s << ":" << c;
  os << "\n  role strata   :";
  for (const auto& [s, c] : d.stratum) os << " " << s << "=" << c;
  os << "\n  n_legal       :";
  for (const auto& [n, c] : d.n_legal) os << " " << n << ":" << c;
  os << "\n  n_legal_conversion_actions:";
  for (const auto& [n, c] : d.n_legal_conversion_actions)
    os << " " << n << ":" << c;
  os << "\n  decision_index bins:";
  for (const auto& [lo, c] : d.decision_index_bin)
    os << " " << lo << "-" << (lo + kDecisionIndexBinWidth - 1) << ":" << c;
  if (!d.decision_index.empty()) {
    const auto& v = d.decision_index;
    os << absl::StrFormat(
        "\n  decision_index: min=%d p25=%d median=%d p75=%d max=%d\n", v.front(),
        v[v.size() / 4], v[v.size() / 2], v[3 * v.size() / 4], v.back());
  } else {
    os << "\n  decision_index: (empty corpus)\n";
  }
  os << absl::StrFormat("  conversion_legal=%d  sm_legal=%d\n",
                        d.n_conversion_legal, d.n_sm_legal);
}

}  // namespace

int main(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);
  at::set_num_threads(1);

  const int num_threads = absl::GetFlag(FLAGS_threads);
  const int base_seeds[kNumArms] = {absl::GetFlag(FLAGS_branch_a_base_seed),
                                    absl::GetFlag(FLAGS_u175_base_seed)};
  const std::string ckpts[kNumArms] = {absl::GetFlag(FLAGS_branch_a_checkpoint),
                                       absl::GetFlag(FLAGS_u175_checkpoint)};
  const int max_episodes = absl::GetFlag(FLAGS_max_episodes_per_arm);
  int episodes_per_arm = absl::GetFlag(FLAGS_episodes_per_arm);
  SPIEL_CHECK_GE(episodes_per_arm, kPwo2BaseEpisodesPerArm);
  SPIEL_CHECK_LE(max_episodes, kPwo2BaseEpisodesPerArm + kExtensionEpisodesPerArm);
  SPIEL_CHECK_LE(episodes_per_arm, max_episodes);

  std::cout << "PWO-3 Phase 1 enrichment strata (registration 5.1-5.4)\n"
            << "  new-selection seed        = " << pwo3::kNewSelectionSeed << "\n"
            << "  conversion rank tag       = " << pwo3::kTagConversionRank
            << "  -> DeriveSeed = "
            << dune_seed::DeriveSeed(pwo3::kNewSelectionSeed,
                                     pwo3::kTagConversionRank)
            << "\n"
            << "  sm rank tag               = " << pwo3::kTagSmRank
            << "  -> DeriveSeed = "
            << dune_seed::DeriveSeed(pwo3::kNewSelectionSeed, pwo3::kTagSmRank)
            << "\n"
            << "  extension half tag base   = " << pwo3::kTagExtensionHalfBase
            << "\n"
            << "  pwo2 half-assign seed     = " << kPwo2HalfAssignSeed
            << " (IMMUTABLE, not a flag)\n"
            << "  conversion action range   = [" << pwo3::kConversionMin << ", "
            << pwo3::kConversionMax << "]  (base " << pwo3::kConversionBase
            << " is a sentinel and is NEVER a conversion)\n"
            << "  swordmaster action        = " << pwo3::kSwordmasterAction << "\n";

  auto game = LoadGame("dune_imperium");
  const int64_t obs_size = game->GetType().provides_information_state_tensor
                               ? game->InformationStateTensorSize()
                               : game->ObservationTensorSize();
  const int64_t action_size = game->NumDistinctActions();
  torch::Device device = torch::cuda::is_available() ? torch::Device(torch::kCUDA)
                                                     : torch::Device(torch::kCPU);
  std::cout << "  device: " << (device.is_cuda() ? "CUDA" : "CPU") << "\n";

  std::shared_ptr<SharedDunePolicyValueNetImpl> models[kNumArms];
  std::string model_shas[kNumArms];
  for (int a = 0; a < kNumArms; ++a) {
    models[a] = LoadModel(ckpts[a], obs_size, action_size, device);
    model_shas[a] = pwo2::Sha256File(ckpts[a]);
    std::cout << "  arm " << ArmName(a) << ": " << ckpts[a] << "\n    sha256 "
              << model_shas[a] << "\n";
  }

  const std::string pwo2_path = absl::GetFlag(FLAGS_pwo2_corpus_path);
  const std::vector<Pwo2Row> pwo2_rows = LoadPwo2Corpus(pwo2_path);
  std::set<std::string> pwo2_hashes;
  std::set<std::pair<int, int>> pwo2_episodes;
  for (const auto& r : pwo2_rows) {
    pwo2_hashes.insert(r.history_hash);
    pwo2_episodes.insert({r.arm, r.episode_id});
  }
  std::cout << "  PWO-2 corpus: " << pwo2_path << "\n    sha256 "
            << pwo2::Sha256File(pwo2_path) << "\n    " << pwo2_rows.size()
            << " roots over " << pwo2_episodes.size()
            << " compound episode keys\n";

  // -------------------------------------------------------------------------
  // Collection. Reused essentially unchanged from dune_pwo2_root_corpus.cc: one
  // episode is handled entirely by one thread and is seeded by its own game
  // seed, so the collected set is independent of thread count. Nothing added
  // here touches the RNG stream, so the regenerated episodes are bit-for-bit
  // the PWO-2 episodes -- which is exactly what the 5.1 gate then proves.
  //
  // Two outputs per episode, not one:
  //   (a) every non-chance, non-simultaneous decision's history hash, keyed by
  //       decision_index. The 5.1 gate needs ALL of them, not just the eligible
  //       ones: a PWO-2 root may sit at a decision_index whose role or
  //       eligibility differs under this tool's stratum filter, and matching
  //       only eligible decisions would report a spurious mismatch.
  //   (b) the enrichment candidates.
  // -------------------------------------------------------------------------
  struct EpisodeOut {
    int arm = 0;
    int episode_id = 0;
    std::map<int, std::string> hash_by_decision_index;
  };
  struct Collection {
    std::vector<Root> candidates;  // canonical order
    std::map<std::pair<int, int>, std::map<int, std::string>> decisions;
    int64_t n_player_decisions = 0;
  };

  auto collect = [&](int n_episodes_per_arm) {
    std::vector<std::vector<Root>> per_thread_roots(num_threads);
    std::vector<std::vector<EpisodeOut>> per_thread_eps(num_threads);
    std::atomic<int> next_job{0};
    const int total_jobs = kNumArms * n_episodes_per_arm;

    auto worker = [&](int tid) {
      std::shared_ptr<algorithms::Evaluator> evals[kNumArms];
      for (int a = 0; a < kNumArms; ++a) {
        evals[a] = std::make_shared<DuneNNEvaluator>(
            models[a], device, static_cast<float>(absl::GetFlag(FLAGS_logit_cap)));
      }
      while (true) {
        int job = next_job.fetch_add(1);
        if (job >= total_jobs) break;
        const int arm = job / n_episodes_per_arm;
        const int episode_id = job % n_episodes_per_arm;
        const int game_seed = base_seeds[arm] + episode_id;

        EpisodeOut eo;
        eo.arm = arm;
        eo.episode_id = episode_id;

        std::mt19937 rng(game_seed);
        auto state = game->NewInitialState();
        int decision_index = 0;
        while (!state->IsTerminal()) {
          ++decision_index;
          if (state->IsChanceNode()) {
            auto outcomes = state->ChanceOutcomes();
            state->ApplyAction(
                SampleAction(outcomes, absl::Uniform(rng, 0.0, 1.0)).first);
            continue;
          }
          if (state->CurrentPlayer() == kSimultaneousPlayerId) {
            std::vector<Action> joint;
            for (int p = 0; p < game->NumPlayers(); ++p) {
              auto acts = state->LegalActions(p);
              std::uniform_int_distribution<int> d(0, acts.size() - 1);
              joint.push_back(acts[d(rng)]);
            }
            state->ApplyActions(joint);
            continue;
          }

          const Player player = state->CurrentPlayer();
          // has_active_session=false: corpus roots are standalone
          // reconstructions with no inherited session, and the qualification
          // runner searches them exactly the same way (PWO-2 registration 3.2).
          const DuneDecisionRole role =
              ClassifyDuneDecisionRole(*state, player, /*has_active_session=*/false);
          auto legal = state->LegalActions();
          const std::vector<Action> history = state->History();
          const std::string history_hash = pwo2::HistoryHash(history);
          eo.hash_by_decision_index[decision_index] = history_hash;

          const char* stratum = StratumNameForRole(role);
          const bool conv = pwo3::HasConversion(legal);
          const bool sm = pwo3::HasSwordmaster(legal);
          if (stratum[0] != '\0' && legal.size() >= 2 && (conv || sm)) {
            Root r;
            r.arm = arm;
            r.episode_id = episode_id;
            r.decision_index = decision_index;
            r.player = player;
            const auto* ds =
                dynamic_cast<const dune_imperium::DuneImperiumState*>(state.get());
            r.round = ds ? ds->GetCurrentRound() : 1;
            r.stratum = stratum;
            r.role = role;
            r.history = history;
            r.legal_actions = legal;
            r.observation = state->InformationStateTensor(player);
            r.history_hash = history_hash;
            r.conversion_legal = conv;
            r.sm_legal = sm;
            r.legal_conversion_actions = pwo3::LegalConversionActions(legal);
            per_thread_roots[tid].push_back(std::move(r));
          }

          // Four-copy raw-policy self-play at sampling temperature 1.0.
          ActionsAndProbs prior = evals[arm]->Prior(*state);
          state->ApplyAction(SampleFromPrior(prior, rng));
        }
        per_thread_eps[tid].push_back(std::move(eo));
      }
    };

    std::vector<std::thread> ws;
    for (int i = 0; i < num_threads; ++i) ws.emplace_back(worker, i);
    for (auto& w : ws) w.join();

    Collection c;
    for (auto& t : per_thread_roots)
      c.candidates.insert(c.candidates.end(), t.begin(), t.end());
    std::sort(c.candidates.begin(), c.candidates.end());  // canonical order
    for (auto& t : per_thread_eps)
      for (auto& eo : t) {
        c.n_player_decisions +=
            static_cast<int64_t>(eo.hash_by_decision_index.size());
        c.decisions[{eo.arm, eo.episode_id}] =
            std::move(eo.hash_by_decision_index);
      }
    return c;
  };

  std::cout << "\nGenerating " << episodes_per_arm << " episodes per arm ("
            << kNumArms * episodes_per_arm << " total)...\n";
  Collection col = collect(episodes_per_arm);
  std::cout << "  " << col.n_player_decisions
            << " player decisions recorded; " << col.candidates.size()
            << " conversion/SM-legal candidate roots\n";

  // -------------------------------------------------------------------------
  // (1) REGENERATION FIDELITY GATE (registration 5.1 step 2). Runs BEFORE any
  // selection. A failure is a STOP: the regeneration premise (frozen engine +
  // same models + same seeds => same episodes) is broken, and nothing measured
  // downstream would mean anything.
  //
  // Registration 5.1 step 1 (re-running scripts/eval/pwo2_corpus_validation.py
  // for G1-G12 under the new binary) is a separate, Python-side obligation and
  // is NOT performed here; this tool implements step 2 and additionally
  // fail-closes on each PWO-2 row's own history/hash self-consistency at load.
  // -------------------------------------------------------------------------
  struct GateMismatch {
    int arm = 0, episode_id = 0, decision_index = 0, round = 0;
    Player player = kInvalidPlayer;
    std::string stratum, expected_hash, got_hash, reason;
  };
  std::vector<GateMismatch> gate_mismatches;
  int gate_checked = 0, gate_matched = 0;
  for (const auto& row : pwo2_rows) {
    ++gate_checked;
    GateMismatch m;
    m.arm = row.arm;
    m.episode_id = row.episode_id;
    m.decision_index = row.decision_index;
    m.round = row.round;
    m.player = row.player;
    m.stratum = row.stratum;
    m.expected_hash = row.history_hash;

    auto ep_it = col.decisions.find({row.arm, row.episode_id});
    if (ep_it == col.decisions.end()) {
      m.reason = "EPISODE_NOT_REGENERATED";
      gate_mismatches.push_back(m);
      continue;
    }
    auto di_it = ep_it->second.find(row.decision_index);
    if (di_it == ep_it->second.end()) {
      m.reason = "DECISION_INDEX_ABSENT";
      gate_mismatches.push_back(m);
      continue;
    }
    if (di_it->second != row.history_hash) {
      m.reason = "HISTORY_HASH_DIFFERS";
      m.got_hash = di_it->second;
      gate_mismatches.push_back(m);
      continue;
    }
    ++gate_matched;
  }
  const bool gate_pass = gate_mismatches.empty();
  std::cout << "\nRegeneration fidelity gate (registration 5.1): checked "
            << gate_checked << " PWO-2 roots, matched " << gate_matched
            << " -> " << (gate_pass ? "PASS" : "FAIL") << "\n";
  if (!gate_pass) {
    std::cerr << "\nSTOP: the regeneration fidelity gate FAILED. "
              << gate_mismatches.size() << " of " << gate_checked
              << " PWO-2 roots did not reproduce. The premise (frozen engine + "
                 "same models + same seeds => same episodes) is broken. Report; "
                 "do not improvise.\nFirst "
              << std::min<size_t>(kMaxReportedMismatches, gate_mismatches.size())
              << " mismatches:\n";
    for (size_t i = 0; i < gate_mismatches.size() &&
                       i < static_cast<size_t>(kMaxReportedMismatches);
         ++i) {
      const auto& m = gate_mismatches[i];
      std::cerr << absl::StrFormat(
          "    %-22s arm=%-9s episode=%3d decision_index=%5d round=%2d "
          "player=%d stratum=%-19s\n      expected %s\n      got      %s\n",
          m.reason, ArmName(m.arm), m.episode_id, m.decision_index, m.round,
          static_cast<int>(m.player), m.stratum, m.expected_hash,
          m.got_hash.empty() ? std::string("(absent)") : m.got_hash);
    }
    return 1;
  }

  // -------------------------------------------------------------------------
  // HALF ASSIGNMENT (registration 5.4). Two independently seeded blocks; the
  // 0..127 block is immutable and is asserted against the frozen corpus.
  // -------------------------------------------------------------------------
  const auto halves = BuildHalves();
  {
    std::vector<std::string> half_disagreements;
    for (const auto& row : pwo2_rows) {
      const std::string got = halves.at({row.arm, row.episode_id});
      if (got != row.half)
        half_disagreements.push_back(absl::StrFormat(
            "arm=%s episode=%d: PWO-2 says %s, recomputed %s", ArmName(row.arm),
            row.episode_id, row.half, got));
    }
    if (!half_disagreements.empty()) {
      std::cerr << "\nSTOP: the recomputed episodes 0-127 half assignment "
                   "DISAGREES with data/pwo2_root_corpus.json on "
                << half_disagreements.size()
                << " root(s). Registration 5.4 declares those assignments "
                   "immutable; a disagreement means the reconstruction is "
                   "wrong, not that the corpus is.\n";
      for (const auto& d : half_disagreements) std::cerr << "    " << d << "\n";
      SPIEL_CHECK_TRUE(half_disagreements.empty());
      return 1;
    }
    std::cout << "  half assignment: episodes 0-127 reproduce all "
              << pwo2_rows.size() << " PWO-2 half tags exactly (registration "
                                     "5.4 hard assert)\n";
  }

  // -------------------------------------------------------------------------
  // (2) and (3): the two strata. Conversion is selected FIRST; the SM stratum
  // then excludes both the PWO-2 corpus and the conversion selection, so the
  // registered no-duplicate-across-corpora rule holds by construction and is
  // re-checked as a guard afterwards regardless.
  // -------------------------------------------------------------------------
  auto conversion_eligible = [](const Root& r) { return r.conversion_legal; };
  auto sm_eligible = [](const Root& r) { return r.sm_legal; };

  auto run_selection = [&](SelectResult* conv, SelectResult* sm) {
    *conv = Select(ConversionSpec(), col.candidates, conversion_eligible, halves,
                   pwo2_hashes);
    std::set<std::string> sm_excluded = pwo2_hashes;
    for (const Root* r : conv->selected) sm_excluded.insert(r->history_hash);
    *sm = Select(SmSpec(), col.candidates, sm_eligible, halves, sm_excluded);
  };

  SelectResult conv_sel, sm_sel;
  run_selection(&conv_sel, &sm_sel);
  std::cout << "\nAt " << episodes_per_arm << " episodes/arm:\n"
            << absl::StrFormat(
                   "  conversion: %d candidates -> %d roots (arm x half totals "
                   "%d/%d/%d/%d, want %d each)\n",
                   conv_sel.n_candidates,
                   static_cast<int>(conv_sel.selected.size()),
                   conv_sel.arm_half_total[0][0], conv_sel.arm_half_total[0][1],
                   conv_sel.arm_half_total[1][0], conv_sel.arm_half_total[1][1],
                   QuotaPerArmHalf(ConversionSpec()))
            << absl::StrFormat(
                   "  sm        : %d candidates -> %d roots (arm x half totals "
                   "%d/%d/%d/%d, want %d each)\n",
                   sm_sel.n_candidates, static_cast<int>(sm_sel.selected.size()),
                   sm_sel.arm_half_total[0][0], sm_sel.arm_half_total[0][1],
                   sm_sel.arm_half_total[1][0], sm_sel.arm_half_total[1][1],
                   QuotaPerArmHalf(SmSpec()));

  // Registered extension budget (registration 5.2 step 4): if any (arm, half)
  // total still cannot reach its quota, extend to `--max_episodes_per_arm` and
  // repeat the WHOLE selection. The fidelity gate is NOT re-run: episodes 0-127
  // are seeded by their own game seeds and are regenerated identically at any
  // episode count, so the gate's result carries over unchanged.
  bool extension_spent = false;
  if ((!conv_sel.arm_half_complete || !sm_sel.arm_half_complete) &&
      episodes_per_arm < max_episodes) {
    std::cout << "\nAt least one (source_arm, half) total is short; spending "
                 "the registered extra-episode budget (-> "
              << max_episodes << "/arm, seeds " << base_seeds[0] << "+"
              << episodes_per_arm << " / " << base_seeds[1] << "+"
              << episodes_per_arm << ").\n";
    PrintAvailability(std::cout, ConversionSpec(), conv_sel);
    PrintAvailability(std::cout, SmSpec(), sm_sel);
    episodes_per_arm = max_episodes;
    extension_spent = true;
    std::cout << "\nGenerating " << episodes_per_arm << " episodes per arm ("
              << kNumArms * episodes_per_arm << " total)...\n";
    col = collect(episodes_per_arm);
    std::cout << "  " << col.n_player_decisions
              << " player decisions recorded; " << col.candidates.size()
              << " conversion/SM-legal candidate roots\n";
    run_selection(&conv_sel, &sm_sel);
    std::cout << absl::StrFormat(
        "  conversion: %d candidates -> %d roots\n  sm        : %d candidates "
        "-> %d roots\n",
        conv_sel.n_candidates, static_cast<int>(conv_sel.selected.size()),
        sm_sel.n_candidates, static_cast<int>(sm_sel.selected.size()));
  }

  // Conversion FLOOR = TARGET = 32. Fewer than 32 after the extension budget is
  // a STOP, not a degraded ship (registration 5.2).
  const int conv_n = static_cast<int>(conv_sel.selected.size());
  SPIEL_CHECK_TRUE(ConversionSpec().floor_is_stop);
  if (conv_n < ConversionSpec().floor) {
    std::cerr << "\nSTOP: the conversion stratum reached " << conv_n
              << " roots; the registered FLOOR = TARGET = "
              << ConversionSpec().floor
              << " and the extra-episode budget is spent. Do NOT silently drop, "
                 "merge, or re-quota a cell (registration 5.2). Per-cell "
                 "availability:\n";
    PrintAvailability(std::cerr, ConversionSpec(), conv_sel);
    return 1;
  }

  // SM FLOOR = 16, TARGET = 32. A miss here is NOT a stop: registration 5.3
  // pre-declares the outcome verbatim and the WO carries on.
  SPIEL_CHECK_FALSE(SmSpec().floor_is_stop);
  const int sm_n = static_cast<int>(sm_sel.selected.size());
  const bool sm_floor_met = sm_n >= SmSpec().floor;

  // -------------------------------------------------------------------------
  // G6 / G7: every root reconstructs exactly. Same replay block as the PWO-2
  // tool, plus the registration 14.2 ascending-legal_actions assert and a
  // re-derivation of the registration 4.4 conversion/SM tags from the REPLAYED
  // legal set -- a tag that was merely carried along would prove nothing.
  // -------------------------------------------------------------------------
  auto replay_verify = [&](const std::vector<const Root*>& sel,
                           const char* label) {
    std::cout << "\nVerifying " << sel.size() << " " << label
              << " roots by replay (G6, G7)...\n";
    for (const Root* r : sel) {
      auto st = ReconstructState(game, r->history);
      SPIEL_CHECK_EQ(st->CurrentPlayer(), r->player);
      const auto* ds =
          dynamic_cast<const dune_imperium::DuneImperiumState*>(st.get());
      SPIEL_CHECK_EQ(ds ? ds->GetCurrentRound() : 1, r->round);
      SPIEL_CHECK_TRUE(st->LegalActions() == r->legal_actions);
      SPIEL_CHECK_GE(r->legal_actions.size(), 2u);            // G7
      SPIEL_CHECK_TRUE(pwo3::IsAscending(r->legal_actions));  // registration 14.2
      SPIEL_CHECK_TRUE(ClassifyDuneDecisionRole(*st, r->player, false) == r->role);
      SPIEL_CHECK_EQ(std::string(StratumNameForRole(r->role)), r->stratum);
      auto obs = st->InformationStateTensor(r->player);
      SPIEL_CHECK_EQ(obs.size(), r->observation.size());
      for (size_t i = 0; i < obs.size(); ++i)
        SPIEL_CHECK_FLOAT_NEAR(obs[i], r->observation[i], 1e-5);
      SPIEL_CHECK_EQ(pwo2::HistoryHash(r->history), r->history_hash);
      SPIEL_CHECK_EQ(pwo3::HasConversion(r->legal_actions), r->conversion_legal);
      SPIEL_CHECK_EQ(pwo3::HasSwordmaster(r->legal_actions), r->sm_legal);
      SPIEL_CHECK_TRUE(pwo3::LegalConversionActions(r->legal_actions) ==
                       r->legal_conversion_actions);
    }
    std::cout << "  all " << label << " roots replay-verified (G6, G7).\n";
  };
  replay_verify(conv_sel.selected, "conversion");
  replay_verify(sm_sel.selected, "sm");

  // -------------------------------------------------------------------------
  // Guards. Checked on the FINAL selections and emitted whether they pass or
  // fail. The cross-stratum duplicate guard is checked in BOTH directions even
  // though the selection order makes it impossible to violate -- a guard that
  // is only asserted where it cannot fail is not a guard.
  // -------------------------------------------------------------------------
  std::set<std::string> conv_hashes, sm_hashes;
  for (const Root* r : conv_sel.selected) conv_hashes.insert(r->history_hash);
  for (const Root* r : sm_sel.selected) sm_hashes.insert(r->history_hash);

  GuardSummary conv_guards = ComputeGuards(ConversionSpec(), conv_sel.selected,
                                           halves, pwo2_hashes, pwo2_episodes,
                                           sm_hashes);
  GuardSummary sm_guards = ComputeGuards(SmSpec(), sm_sel.selected, halves,
                                         pwo2_hashes, pwo2_episodes,
                                         conv_hashes);
  conv_guards.g_replay_verified = true;  // the block above aborts on failure
  sm_guards.g_replay_verified = true;
  FinalizeGuards(conv_guards);
  FinalizeGuards(sm_guards);

  const Dist conv_dist = ComputeDist(conv_sel.selected);
  const Dist sm_dist = ComputeDist(sm_sel.selected);

  // -------------------------------------------------------------------------
  // Emit.
  // -------------------------------------------------------------------------
  auto emit_corpus = [&](const CorpusSpec& spec,
                         const std::vector<const Root*>& sel,
                         const std::string& path) {
    json::Array corpus;
    for (const Root* r : sel) {
      json::Object o;
      // ---- Every field the PWO-2 corpus carries, unchanged in meaning.
      // `category` stays schema-compatible with dune_search_calibration
      // --corpus_path, and `stratum` keeps its PWO-2 meaning (registration
      // 4.0): it is the DECISION-ROLE stratum on every root and is never
      // "conversion" or "sm".
      o["category"] = r->stratum;
      o["player"] = static_cast<int64_t>(r->player);
      o["round"] = static_cast<int64_t>(r->round);
      json::Array h;
      for (Action a : r->history) h.push_back(static_cast<int64_t>(a));
      o["history"] = h;
      json::Array obs;
      for (float v : r->observation) obs.push_back(v);
      o["observation"] = obs;
      json::Array la;
      for (Action a : r->legal_actions) la.push_back(static_cast<int64_t>(a));
      o["legal_actions"] = la;
      o["root_id"] = r->history_hash;
      o["history_hash"] = r->history_hash;
      o["stratum"] = r->stratum;
      o["role"] = std::string(pwo2::RoleName(r->role));
      o["source_arm"] = std::string(ArmName(r->arm));
      o["source_episode_id"] = static_cast<int64_t>(r->episode_id);
      o["source_game_seed"] =
          static_cast<int64_t>(base_seeds[r->arm] + r->episode_id);
      o["half"] = halves.at({r->arm, r->episode_id});
      o["decision_index"] = static_cast<int64_t>(r->decision_index);
      o["n_legal"] = static_cast<int64_t>(r->legal_actions.size());
      // PWO-2's registration-3.7 live-50 tag. Definitionally false on an
      // enrichment root; emitted anyway so a concatenation of main +
      // conversion + sm has one uniform schema (registration 8.2 pools
      // corpus in {main, conversion}). It is NOT a PWO-3 live-tier flag --
      // PWO-3's live root set is registered separately in section 6.
      o["in_live50_subset"] = false;
      // ---- PWO-3 additions (registration 4.0, 4.4).
      o["corpus"] = std::string(spec.name);
      json::Array lca;
      for (Action a : r->legal_conversion_actions)
        lca.push_back(static_cast<int64_t>(a));
      o["legal_conversion_actions"] = lca;
      o["n_legal_conversion_actions"] =
          static_cast<int64_t>(r->legal_conversion_actions.size());
      o["conversion_legal"] = r->conversion_legal;
      o["sm_legal"] = r->sm_legal;
      o["corpus_schema_version"] = std::string("pwo3_v1");
      corpus.push_back(o);
    }
    const auto parent = std::filesystem::path(path).parent_path();
    if (!parent.empty()) std::filesystem::create_directories(parent);
    std::ofstream out(path);
    if (!out) {
      std::cerr << "Failed to open output path: " << path << "\n";
      std::exit(1);
    }
    out << json::ToString(corpus, true) << "\n";
  };

  // The 5.1 gate result, identical in both manifests: one gate, two corpora.
  json::Object gate_obj;
  gate_obj["registration_section"] = std::string("5.1");
  gate_obj["n_pwo2_roots_checked"] = static_cast<int64_t>(gate_checked);
  gate_obj["n_matched"] = static_cast<int64_t>(gate_matched);
  gate_obj["n_mismatched"] = static_cast<int64_t>(gate_mismatches.size());
  gate_obj["PASS"] = gate_pass;
  gate_obj["episodes_per_arm_at_gate"] =
      static_cast<int64_t>(absl::GetFlag(FLAGS_episodes_per_arm));
  gate_obj["note"] = std::string(
      "Matched on ALL non-chance, non-simultaneous decisions of the "
      "regenerated episode, not only the ones this tool considers eligible: a "
      "PWO-2 root may sit at a decision_index whose role or eligibility "
      "differs here. Registration 5.1 step 1 (pwo2_corpus_validation.py, "
      "G1-G12) is a separate Python-side obligation and is not performed by "
      "this binary. The gate is not re-run after the extra-episode budget is "
      "spent: episodes 0-127 are seeded by their own game seeds and regenerate "
      "identically at any episode count.");

  const std::string self_sha = pwo2::Sha256File(argv[0]);
  const std::string pwo2_sha = pwo2::Sha256File(pwo2_path);

  auto emit_manifest = [&](const CorpusSpec& spec, const SelectResult& sr,
                           const GuardSummary& g, const Dist& d, bool is_sm,
                           const std::string& corpus_path,
                           const std::string& path) {
    json::Object man;
    man["tool"] = std::string("dune_pwo3_strata");
    man["registration"] = std::string("docs/PWO3_REGISTRATION.md");
    man["registration_section"] = std::string(spec.registration_section);
    man["corpus"] = std::string(spec.name);
    man["corpus_path"] = corpus_path;
    man["corpus_schema_version"] = std::string("pwo3_v1");
    man["binary_sha256"] = self_sha;
    json::Object shas;
    shas["branch_a"] = model_shas[0];
    shas["u175"] = model_shas[1];
    man["model_sha256"] = shas;
    man["pwo2_corpus_path"] = pwo2_path;
    man["pwo2_corpus_sha256"] = pwo2_sha;

    // ---- Provenance of the source episodes.
    man["branch_a_base_seed"] = static_cast<int64_t>(base_seeds[0]);
    man["u175_base_seed"] = static_cast<int64_t>(base_seeds[1]);
    man["episodes_per_arm"] = static_cast<int64_t>(episodes_per_arm);
    man["episodes_per_arm_first_attempt"] =
        static_cast<int64_t>(absl::GetFlag(FLAGS_episodes_per_arm));
    man["max_episodes_per_arm"] = static_cast<int64_t>(max_episodes);
    man["extra_episode_budget_spent"] = extension_spent;
    man["sampling_temperature"] = 1.0;
    man["logit_cap"] = absl::GetFlag(FLAGS_logit_cap);
    man["hidden_dim"] = static_cast<int64_t>(absl::GetFlag(FLAGS_hidden_dim));
    man["num_blocks"] = static_cast<int64_t>(absl::GetFlag(FLAGS_num_blocks));
    man["nonlinear_value_head"] = absl::GetFlag(FLAGS_nonlinear_value_head);
    man["threads"] = static_cast<int64_t>(num_threads);
    man["n_player_decisions_regenerated"] =
        static_cast<int64_t>(col.n_player_decisions);
    man["n_candidate_roots_collected"] =
        static_cast<int64_t>(col.candidates.size());
    man["n_candidates_for_this_stratum"] = static_cast<int64_t>(sr.n_candidates);

    // ---- Registered RNG.
    man["new_selection_seed"] =
        static_cast<int64_t>(pwo3::kNewSelectionSeed);
    man["rank_tag"] = static_cast<int64_t>(spec.rank_tag);
    man["rank_tag_derived_seed"] = static_cast<int64_t>(
        dune_seed::DeriveSeed(pwo3::kNewSelectionSeed, spec.rank_tag));
    man["pwo2_half_assign_seed"] = static_cast<int64_t>(kPwo2HalfAssignSeed);
    man["extension_half_tag_base"] =
        static_cast<int64_t>(pwo3::kTagExtensionHalfBase);
    man["half_assignment_rule"] = std::string(
        "registration 5.4: ids 0-127 per arm shuffled with "
        "MakeRng64(DeriveSeed(20260726, arm)), first 64 calibration; ids "
        "128-255 per arm shuffled INDEPENDENTLY with "
        "MakeRng64(DeriveSeed(20260728, 300 + arm)), first 64 calibration. "
        "PWO-2's AssignHalves(n, seed) is NOT called with n=256 -- that would "
        "silently reassign the original 128. The 0-127 tags are hard-asserted "
        "against data/pwo2_root_corpus.json.");
    man["selection_rule"] = std::string(
        "registration 5.2 (5.3 reuses it unchanged): exact quota per micro-cell "
        "(source_arm, half, round_bucket) with a registered within-cell seat "
        "schedule; within-cell rank = "
        "DeriveSeed(DeriveSeed(20260728, <rank_tag>), "
        "HexPrefix64(history_hash)), lowest wins; tie-breaks history_hash "
        "lexicographic, source_arm, source_episode_id, then decision_index as a "
        "LAST RESORT ONLY -- decision_index is never a ranking key.");
    man["shortfall_rule"] = std::string(
        "registration 5.2 step 1 AS AMENDED (2026-07-28, before any data "
        "existed): an unfillable slot redistributes its quota WITHIN the same "
        "(source_arm, half) pair, donors ordered by |b'-b| ascending then b' "
        "ascending -- which puts the shortfall's OWN bucket first, at distance "
        "0, filled from a seat OUTSIDE that cell's registered seat schedule; "
        "then the other buckets, each filled by ascending RankKey over all its "
        "admissible candidates REGARDLESS OF SEAT. Rationale on the record: a "
        "cell that fails only because one SEAT is unavailable must not "
        "surrender its root to a neighbouring ROUND bucket, because the round "
        "distribution is the axis both enrichment strata exist to control (the "
        "SM stratum's entire point is its b0/b1 weighting). Seat balance is the "
        "cheaper thing to relax: it is reported, never gated. Every "
        "redistribution is recorded, including the distance-0 ones where donor "
        "== recipient.");
    man["eligibility_rule"] = std::string(
        is_sm ? "registration 5.3: role in {kAgentPrimary, kAgentContinuation, "
                "kPurchase, kCombatIntrigue} (has_active_session=false), "
                "n_legal >= 2, action 610 legal, history_hash absent from both "
                "the PWO-2 corpus and the conversion stratum."
              : "registration 5.2: role in {kAgentPrimary, "
                "kAgentContinuation, kPurchase, kCombatIntrigue} "
                "(has_active_session=false), n_legal >= 2, at least one of "
                "741-752 legal, history_hash absent from the PWO-2 corpus.");
    man["episode_constraint_rule"] = std::string(
        "at most 1 root per compound episode key (source_arm, "
        "source_episode_id) WITHIN a decision-role stratum, and at most 2 per "
        "compound key WITHIN this corpus. 'stratum' and 'corpus' are the two "
        "registration-4.0 axes; reading both as the same axis would collapse "
        "the pair and make the >= 16 distinct-episode guard dead.");
    man["conversion_action_range"] = std::string(absl::StrFormat(
        "%d-%d inclusive; %d is an unused base constant and is NEVER a "
        "conversion",
        static_cast<int>(pwo3::kConversionMin),
        static_cast<int>(pwo3::kConversionMax),
        static_cast<int>(pwo3::kConversionBase)));
    man["swordmaster_action"] =
        static_cast<int64_t>(pwo3::kSwordmasterAction);

    man["fidelity_gate"] = gate_obj;
    man["guard_table"] = GuardObject(g, spec);
    man["n_roots"] = static_cast<int64_t>(g.n);

    // ---- Per-cell availability table, ALWAYS emitted.
    json::Array cells;
    for (const auto& c : sr.cells) {
      json::Object o;
      o["source_arm"] = std::string(ArmName(c.arm));
      o["half"] = std::string(HalfName(c.half));
      o["round_bucket"] = std::string(BucketName(c.bucket));
      o["quota"] = static_cast<int64_t>(c.quota);
      o["n_roots_in_cell"] = static_cast<int64_t>(
          c.n_scheduled_fills + c.n_offschedule_fills + c.n_extra_as_donor);
      o["n_filled_at_scheduled_seat"] =
          static_cast<int64_t>(c.n_scheduled_fills);
      o["n_filled_off_schedule_seat_same_bucket"] =
          static_cast<int64_t>(c.n_offschedule_fills);
      o["n_extra_supplied_as_donor"] = static_cast<int64_t>(c.n_extra_as_donor);
      o["n_quota_exported_to_other_buckets"] =
          static_cast<int64_t>(c.n_quota_exported);
      o["n_quota_unmet"] = static_cast<int64_t>(c.n_quota_unmet);
      o["n_available_any_seat"] = static_cast<int64_t>(c.n_available);
      o["n_admissible_remaining"] =
          static_cast<int64_t>(c.n_admissible_remaining);
      json::Object by_seat;
      for (int s = 0; s < kNumSeats; ++s)
        by_seat[std::to_string(s)] = static_cast<int64_t>(c.seat_available[s]);
      o["available_by_seat"] = by_seat;
      json::Array sched;
      for (int s : spec.seats[c.bucket]) sched.push_back(static_cast<int64_t>(s));
      o["scheduled_seats"] = sched;
      json::Array unf;
      for (int s : c.unfilled_seats) unf.push_back(static_cast<int64_t>(s));
      o["unfilled_scheduled_seats"] = unf;
      cells.push_back(o);
    }
    man["cell_availability_table"] = cells;

    json::Array shorts;
    for (const auto& s : sr.shortfalls) {
      json::Object o;
      o["source_arm"] = std::string(ArmName(s.arm));
      o["half"] = std::string(HalfName(s.half));
      o["round_bucket"] = std::string(BucketName(s.bucket));
      o["seat"] = static_cast<int64_t>(s.seat);
      o["n_available"] = static_cast<int64_t>(s.n_available);
      o["n_admissible"] = static_cast<int64_t>(s.n_admissible);
      shorts.push_back(o);
    }
    man["unfillable_scheduled_slots"] = shorts;

    json::Array redis;
    for (const auto& r : sr.redistributions) {
      json::Object o;
      o["source_arm"] = std::string(ArmName(r.arm));
      o["half"] = std::string(HalfName(r.half));
      o["donor_round_bucket"] = std::string(BucketName(r.donor_bucket));
      o["recipient_round_bucket"] =
          std::string(BucketName(r.recipient_bucket));
      o["count"] = static_cast<int64_t>(r.count);
      // The amended rule's distance-0 donor is the shortfall's own bucket, so
      // donor == recipient means "seat schedule relaxed, round bucket held".
      o["kind"] = std::string(r.donor_bucket == r.recipient_bucket
                                  ? "seat_relaxed_same_bucket"
                                  : "cross_bucket");
      redis.push_back(o);
    }
    man["redistributions"] = redis;
    man["n_redistributed_roots"] =
        static_cast<int64_t>(sr.redistributions.size());

    json::Object ah;
    for (int arm = 0; arm < kNumArms; ++arm)
      for (int half = 0; half < kNumHalves; ++half)
        ah[absl::StrFormat("%s/%s", ArmName(arm), HalfName(half))] =
            static_cast<int64_t>(sr.arm_half_total[arm][half]);
    man["roots_per_arm_half"] = ah;
    man["quota_per_arm_half"] = static_cast<int64_t>(QuotaPerArmHalf(spec));
    man["arm_half_totals_complete"] = sr.arm_half_complete;

    man["cross_corpus_note"] = std::string(
        "Overlap with the PWO-2 corpus on the compound episode key is "
        "PERMITTED and recorded (registration 5.2); the cluster bootstrap "
        "handles the correlation via that key. Overlap on history_hash is "
        "forbidden against the PWO-2 corpus and against the other new "
        "stratum, and is guarded in both directions.");

    AddDistToManifest(man, d);

    if (is_sm) {
      man["sm_corpus_n"] = static_cast<int64_t>(g.n);
      man["sm_floor_met"] = sm_floor_met;
      man["sm_target"] = static_cast<int64_t>(spec.target);
      man["sm_floor"] = static_cast<int64_t>(spec.floor);
      if (!sm_floor_met) {
        man["registered_outcome"] = std::string(kSmUnderpoweredOutcome);
      }
      man["sm_scope_note"] = std::string(
          "Registration 5.3: this stratum is search-only -- fixed_400 / "
          "fixed_800 seed-1301 rows, no seed 1302, no live rows, and NO oracle "
          "cells, ever.");
    }

    man["STRUCTURAL_GUARDS_PASS"] = g.structural_pass;
    man["ALL_GUARDS_PASS"] = g.all_pass;

    const auto parent = std::filesystem::path(path).parent_path();
    if (!parent.empty()) std::filesystem::create_directories(parent);
    std::ofstream out(path);
    if (!out) {
      std::cerr << "Failed to open manifest path: " << path << "\n";
      std::exit(1);
    }
    out << json::ToString(man, true) << "\n";
  };

  const std::string conv_out = absl::GetFlag(FLAGS_conversion_output);
  const std::string conv_man = absl::GetFlag(FLAGS_conversion_manifest);
  const std::string sm_out = absl::GetFlag(FLAGS_sm_output);
  const std::string sm_man = absl::GetFlag(FLAGS_sm_manifest);
  emit_corpus(ConversionSpec(), conv_sel.selected, conv_out);
  emit_manifest(ConversionSpec(), conv_sel, conv_guards, conv_dist,
                /*is_sm=*/false, conv_out, conv_man);
  emit_corpus(SmSpec(), sm_sel.selected, sm_out);
  emit_manifest(SmSpec(), sm_sel, sm_guards, sm_dist, /*is_sm=*/true, sm_out,
                sm_man);

  // -------------------------------------------------------------------------
  // Report. Guard tables, availability tables, redistributions and ALL the
  // histograms go to stdout unconditionally -- see the Amendment-1 note above
  // AddDistToManifest. Nothing here is gated on a guard having failed.
  // -------------------------------------------------------------------------
  auto print_guards = [](const char* label, const CorpusSpec& spec,
                         const GuardSummary& g) {
    std::cout << absl::StrFormat(
        "\nGuard table, %s corpus (target %d, floor %d):\n"
        "  n_roots                                   = %d\n"
        "  arms   branch_a/u175                      = %d/%d\n"
        "  halves calibration/validation             = %d/%d\n"
        "  distinct compound episode keys            = %d\n"
        "  max roots per compound key (corpus)       = %d\n"
        "  max roots per compound key (role stratum) = %d\n"
        "  compound keys shared with the PWO-2 corpus= %d (permitted, recorded)\n"
        "  G exact target roots                      %s\n"
        "  G arm balance exact                       %s\n"
        "  G half balance exact                      %s\n"
        "  G >= 16 distinct compound episodes        %s\n"
        "  G <= 2 roots per compound key (corpus)    %s\n"
        "  G <= 1 root  per compound key (stratum)   %s\n"
        "  G no duplicate hash within corpus         %s\n"
        "  G no duplicate hash vs PWO-2 corpus       %s\n"
        "  G no duplicate hash vs other new stratum  %s\n"
        "  G6/G7 all roots replay-verified           %s\n"
        "  STRUCTURAL_GUARDS_PASS = %s   ALL_GUARDS_PASS = %s\n",
        label, spec.target, spec.floor, g.n, g.per_arm[0], g.per_arm[1],
        g.per_half[0], g.per_half[1], g.n_distinct_episodes,
        g.max_roots_per_episode, g.max_roots_per_episode_within_stratum,
        g.n_shared_episodes_with_pwo2,
        g.g_count_target ? "PASS" : "FAIL",
        g.g_arm_balance ? "PASS" : "FAIL",
        g.g_half_balance ? "PASS" : "FAIL",
        g.g_min_episodes ? "PASS" : "FAIL",
        g.g_max_per_episode_within_corpus ? "PASS" : "FAIL",
        g.g_max_per_episode_within_stratum ? "PASS" : "FAIL",
        g.g_no_duplicate_within ? "PASS" : "FAIL",
        g.g_no_duplicate_vs_pwo2 ? "PASS" : "FAIL",
        g.g_no_duplicate_vs_other_stratum ? "PASS" : "FAIL",
        g.g_replay_verified ? "PASS" : "FAIL",
        g.structural_pass ? "true" : "false", g.all_pass ? "true" : "false");
  };

  print_guards("conversion", ConversionSpec(), conv_guards);
  std::cout << "\nConversion cell availability / redistribution ledger:\n";
  PrintAvailability(std::cout, ConversionSpec(), conv_sel);
  PrintDist(std::cout, "conversion", conv_dist);

  print_guards("sm", SmSpec(), sm_guards);
  std::cout << "\nSM cell availability / redistribution ledger:\n";
  PrintAvailability(std::cout, SmSpec(), sm_sel);
  PrintDist(std::cout, "sm", sm_dist);

  std::cout << absl::StrFormat(
      "\nSM slice: n=%d, target=%d, floor=%d, sm_floor_met=%s\n", sm_n,
      SmSpec().target, SmSpec().floor, sm_floor_met ? "true" : "false");
  if (!sm_floor_met) {
    std::cout << "\n"
              << "  ****************************************************\n"
              << "  ** REGISTERED OUTCOME (registration 5.3), verbatim:\n"
              << "  ** " << kSmUnderpoweredOutcome << "\n"
              << "  ** achieved SM roots: " << sm_n << " of a floor of "
              << SmSpec().floor << "; SM-legal candidates found: "
              << sm_sel.n_candidates << " over " << episodes_per_arm
              << " episodes/arm.\n"
              << "  ** This is NOT a stop. The WO carries on; the rarity of\n"
              << "  ** early SM-legal states in raw self-play is itself a\n"
              << "  ** finding the curriculum decision needs.\n"
              << "  ****************************************************\n";
  }

  std::cout << "\nWrote:\n  " << conv_out << "\n  " << conv_man << "\n  "
            << sm_out << "\n  " << sm_man << "\n";
  std::cout << "\n  conversion ALL_GUARDS_PASS = "
            << (conv_guards.all_pass ? "true" : "false")
            << "\n  sm         ALL_GUARDS_PASS = "
            << (sm_guards.all_pass ? "true" : "false")
            << "  (its count-shaped guards are advisory: registration 5.3 "
               "pre-declares an underpowered SM slice as a reported outcome, "
               "not a stop)\n";

  // Exit code: the conversion stratum's full guard set, plus the STRUCTURAL
  // guards of the SM stratum. An SM count shortfall never makes this non-zero.
  const bool ok = conv_guards.all_pass && sm_guards.structural_pass;
  return ok ? 0 : 1;
}
