// PWO-5 offline preparation pass -- replay-fidelity gate, whole-trajectory
// split, auxiliary-target artifact, and the legacy distillation label pack.
//
// Registration of record: docs/PWO5_PILOT_REGISTRATION.md (DRAFT revision 9),
// sections 5.3, 5.4, 6.1, 8.3, 8.5 and section 16 gate 3 items 3 and 4.
//
// ---------------------------------------------------------------------------
// WHY THIS BINARY EXISTS
// ---------------------------------------------------------------------------
// Section 5.1: the PWO-4 stream persists NO observation tensor. Every PWO-5
// auxiliary head, and the distillation loss itself, therefore needs the
// observation reconstructed by replaying each game's recorded engine action
// `history` through the frozen engine. Section 5.4 requires that the auxiliary
// targets be derived in the SAME replay pass that feeds the section 5.3
// fidelity gate, "so no target can be computed against a state the gate did not
// verify". That is one pass, and this is it.
//
// The tool runs no search, loads no model and touches no GPU. It is a pure
// function of (games.jsonl, labels.jsonl, the engine).
//
// ASSERTION 5 IS THREE ASSERTIONS, per amendment 1 ruling 3
// (docs/PWO5_AMENDMENT_1_TARGET_EXPOSURE_TELEMETRY_2026_07_31.md section 4).
// Registration 5.3 assertion 5 as literally written required the replayed
// terminal `FinalScoredVp` by seat to equal games.jsonl's `final_vp`. It does
// not hold, on 212 of 1,600 (game, seat) pairs, because the recorded field was
// written by a different function -- and that equality requirement has been
// WITHDRAWN. What replaces it:
//   5a  GetCurrentRound() - 1 == rounds_played              (hard STOP)
//   5b  recomputed PWO-4 legacy reported VP == final_vp     (hard STOP)
//   5c  FinalScoredVp recomputed INDEPENDENTLY for the head target, NOT
//       required to equal the recorded field; the 212/1,600 and 59/400
//       divergences are PINNED and a different count is a hard STOP
// The mechanism is documented at Pwo4LegacyReportedVp below.
//
// There is no --strict_final_scored_vp flag. It was deleted: it made a
// knowingly-failing assertion optional while the tool still reported success.
//
// ---------------------------------------------------------------------------
// WHAT IT PRODUCES, in <out_dir>
// ---------------------------------------------------------------------------
//   heldout_games.json          the section 8.5 whole-trajectory split
//   aux_targets.bin             the section 6.1-item-1 "parallel artifact":
//                               observation + three targets, keyed by
//                               (game_index, decision_index), ALL label rows
//   aux_targets_manifest.json   its sha256 and its counts
//   search_labels/labels_000.bin  the legacy distillation pack
//   search_labels/manifest.json   its loader-verified manifest
//
// The pack lives in its own SUBDIRECTORY on purpose: SearchLabelBuffer::
// LoadNewFiles (dune_ppo_train.cc:461-467) globs *every* `.bin` in the
// directory it is pointed at, so aux_targets.bin must not share a directory
// with the pack or the loader would try to parse it as labels.
//
// ---------------------------------------------------------------------------
// aux_targets.bin FORMAT  (little-endian; x86-64 native layout, written field
// by field so no struct padding participates)
// ---------------------------------------------------------------------------
//   header:
//     uint32 magic       = 0x50573541
//     uint32 schema      = 1
//     uint32 obs_size
//     uint32 action_dim          (= NumDistinctActions() = 2391)
//     uint64 num_rows
//     uint64 reserved    = 0
//   record, repeated num_rows times, in canonical order
//   (game_index ASCENDING, then decision_index ASCENDING):
//     int32  game_index
//     int32  decision_index
//     int32  acting_player
//     int32  terminal_round_class     0: rounds_played <= 8, 1: == 9, 2: == 10
//     float  final_vp_target          FinalScoredVp(acting_player) / 20.0
//     int32  next_own_action          -1 = no target (final label row of a game)
//     uint8  weight                   0 or 1, copied from the label row
//     uint8  is_heldout               1 if the row's game is one of the 60
//     uint8  pad[2]      = 0
//     float  observation[obs_size]
//
// Header is 32 bytes; each record is 28 + 4*obs_size bytes.
//
// ---------------------------------------------------------------------------
// THE DISTILLATION PACK AND THE `% 11` TRAP  (registration 6.1 / 8.5)
// ---------------------------------------------------------------------------
// Section 8.5 calls this "the single most likely silent-violation path in the
// whole implementation", and section 16 gate 3 checks it explicitly. The
// hazard: the legacy loader partitions the pack itself, per row, with
// IsValidationLabel = ComputeLabelFnv1a(observation, legal_actions, player_id)
// % 11 == 0 (dune_ppo_training_utils.h:361-365). That is a WITHIN-trajectory
// split, which plan section 9.6 forbids for terminal_round_head because every
// row of a game shares one terminal round.
//
// ComputeLabelFnv1a is a pure function of data we do not control, so the pack
// cannot be arranged to make the loader's partition coincide with the
// whole-trajectory split. What this tool does instead, and it is the option
// section 8.5 leaves open ("or the loader's partition must be bypassed"):
//
//   *** NO ROW OF A HELD-OUT GAME IS WRITTEN INTO THE PACK AT ALL. ***
//
// The pack contains exactly the training population: weight-one rows whose game
// is NOT one of the 60 held out. The loader's `% 11` therefore never sees a
// held-out row, and cannot promote one into a head's held-out set. What it does
// do is carve roughly 1/11 of the TRAINING rows into its own "validation"
// bucket, which the distillation loss never trains on. That is a loss of
// distillation exposure, entirely inside the training portion, and it is a
// different and harmless thing from a split violation -- but it is NOT free,
// so the exact diverted count is measured here and printed loudly rather than
// absorbed. The disposition of that loss is a principal's call, not this
// tool's.
//
// The pack's manifest reports training_label_count / validation_label_count as
// the loader will ACTUALLY compute them (this tool runs the identical function
// over the identical bytes), not as the totals written.
//
// ---------------------------------------------------------------------------
// JSON
// ---------------------------------------------------------------------------
// labels.jsonl is 375 MB over 48,443 lines of 177 keys. Rather than build a
// full json::Value tree per line, this file carries a small depth-tracked
// top-level field scanner: it walks each line once, skipping nested values
// wholesale, and records the raw text span of the ~13 fields that are needed.
// Because nesting is skipped as a unit, a key inside a nested value can never
// shadow a top-level one. Parse correctness is then cross-checked against the
// data's own redundancy: n_legal == |legal_actions| == |policy_target_exact|
// == |raw_prior_vector_exact| on every row, and each game's row count must
// equal games.jsonl's n_label_rows.
//
// --labels_path must be a PLAIN .jsonl. Decompress labels.jsonl.gz first; this
// binary links no decompressor and shells out to nothing.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

#include "open_spiel/abseil-cpp/absl/flags/flag.h"
#include "open_spiel/abseil-cpp/absl/flags/parse.h"
#include "open_spiel/abseil-cpp/absl/types/span.h"
#include "open_spiel/games/dune_imperium/dune_imperium.h"
#include "open_spiel/games/dune_imperium/dune_imperium_cards.h"
#include "open_spiel/spiel.h"
#include "open_spiel/utils/json.h"

#include "dune_ppo_training_utils.h"  // ComputeLabelFnv1a / IsValidationLabel
#include "dune_sha256.h"              // ComputeFileSHA256 / ComputeStringSHA256

ABSL_FLAG(std::string, labels_path, "",
          "REQUIRED. PLAIN (already decompressed) labels.jsonl.");
ABSL_FLAG(std::string, games_path, "",
          "REQUIRED. games.jsonl carrying each game's full action history.");
ABSL_FLAG(std::string, out_dir, "", "REQUIRED. Where the artifacts are written.");
ABSL_FLAG(int, expect_label_rows, 48443, "Registration 4.1 row count. Asserted.");
ABSL_FLAG(int, expect_games, 400, "Registration 4.1 game count. Asserted.");
ABSL_FLAG(int, heldout_games, 60, "Registration 8.5: 15% of 400.");
namespace open_spiel {
namespace {

// ---------------------------------------------------------------------------
// The pinned corpus-audit divergence (amendment 1 ruling 3)
// ---------------------------------------------------------------------------
// `--strict_final_scored_vp` is GONE. It made a knowingly-failing assertion
// optional while the tool still printed "ALL GATES PASSED", which is exactly
// the shape this program forbids. Amendment 1 replaces the equality
// requirement with a PIN: the divergence between the engine's FinalScoredVp
// and the legacy reported field is a measured property of the frozen corpus,
// and these are its measured values. The tool asserts them, so an unexpected
// count -- in EITHER direction -- is a STOP.
constexpr int64_t kExpectedVpDivergences = 212;            // of 1,600 pairs
constexpr int64_t kExpectedVpDivergencesActingSeat = 59;   // of 400 games

// ---------------------------------------------------------------------------
// STOP
// ---------------------------------------------------------------------------
// Section 5.3 is "a STOP on any mismatch, any row". Every gate failure exits
// non-zero after naming the offending game_index/decision_index.
[[noreturn]] void Stop(const std::string& msg) {
  std::cout.flush();
  std::cerr << "\n*** STOP: " << msg << "\n";
  std::cerr.flush();
  std::exit(2);
}

void Require(bool cond, const std::string& msg) {
  if (!cond) Stop(msg);
}

// Amendment 1 ruling 2 requires the COMPLETE histogram to be recorded, not a
// summary of it. Emitted as a sparse {"vp": count} object -- integer counts, so
// json.cc's six-decimal float floor cannot touch them.
std::string HistogramJson(const std::array<int64_t, 41>& h) {
  std::string out;
  bool first = true;
  for (int v = 0; v < static_cast<int>(h.size()); ++v) {
    if (h[v] == 0) continue;
    if (!first) out += ", ";
    first = false;
    out += "\"" + std::to_string(v) + "\": " + std::to_string(h[v]);
  }
  return out;
}

// ---------------------------------------------------------------------------
// Buffered binary writer. Endianness is x86-64 little-endian; every POD is
// written by its native representation, field by field, so no struct padding
// enters the file.
// ---------------------------------------------------------------------------
class BinWriter {
 public:
  explicit BinWriter(const std::string& path) : path_(path) {
    f_ = std::fopen(path.c_str(), "wb");
    if (f_ == nullptr) Stop("cannot open for writing: " + path);
    std::setvbuf(f_, nullptr, _IOFBF, 1 << 22);
  }
  ~BinWriter() { Close(); }
  BinWriter(const BinWriter&) = delete;
  BinWriter& operator=(const BinWriter&) = delete;

  void Raw(const void* p, size_t n) {
    if (n == 0) return;
    if (std::fwrite(p, 1, n, f_) != n) Stop("short write to " + path_);
    bytes_ += n;
  }
  void U32(uint32_t v) { Raw(&v, 4); }
  void I32(int32_t v) { Raw(&v, 4); }
  void U64(uint64_t v) { Raw(&v, 8); }
  void F32(float v) { Raw(&v, 4); }
  void U8(uint8_t v) { Raw(&v, 1); }
  void Floats(const std::vector<float>& v) { Raw(v.data(), v.size() * sizeof(float)); }

  void Close() {
    if (f_ != nullptr) {
      std::FILE* f = f_;
      f_ = nullptr;
      if (std::fclose(f) != 0) Stop("close/flush failed for " + path_);
    }
  }
  uint64_t bytes() const { return bytes_; }
  const std::string& path() const { return path_; }

 private:
  std::string path_;
  std::FILE* f_ = nullptr;
  uint64_t bytes_ = 0;
};

// ---------------------------------------------------------------------------
// Top-level JSON field scanner (see the header comment for why).
// ---------------------------------------------------------------------------
struct Span {
  const char* b = nullptr;
  const char* e = nullptr;
  bool present() const { return b != nullptr; }
  std::string_view view() const { return std::string_view(b, e - b); }
};

inline const char* SkipWs(const char* p, const char* e) {
  while (p < e && (*p == ' ' || *p == '\t' || *p == '\r' || *p == '\n')) ++p;
  return p;
}

// p points at the opening quote. Returns just past the closing quote, or
// nullptr if the string is unterminated.
inline const char* ScanString(const char* p, const char* e, const char** cb,
                              const char** ce) {
  if (p >= e || *p != '"') return nullptr;
  ++p;
  if (cb != nullptr) *cb = p;
  while (p < e) {
    if (*p == '\\') {
      p += 2;
      continue;
    }
    if (*p == '"') {
      if (ce != nullptr) *ce = p;
      return p + 1;
    }
    ++p;
  }
  return nullptr;
}

// p points at the first character of a value. Returns just past the value.
// Nested objects/arrays are consumed as a unit with string-aware depth
// tracking, which is what makes top-level keys unshadowable.
inline const char* ScanValue(const char* p, const char* e) {
  if (p >= e) return nullptr;
  if (*p == '"') return ScanString(p, e, nullptr, nullptr);
  if (*p == '{' || *p == '[') {
    int depth = 0;
    while (p < e) {
      const char c = *p;
      if (c == '"') {
        p = ScanString(p, e, nullptr, nullptr);
        if (p == nullptr) return nullptr;
        continue;
      }
      if (c == '{' || c == '[') {
        ++depth;
      } else if (c == '}' || c == ']') {
        --depth;
        if (depth == 0) return p + 1;
        if (depth < 0) return nullptr;
      }
      ++p;
    }
    return nullptr;
  }
  while (p < e && *p != ',' && *p != '}' && *p != ']' && *p != ' ' &&
         *p != '\t' && *p != '\r' && *p != '\n') {
    ++p;
  }
  return p;
}

// A fixed list of wanted keys; the scanner fills the matching spans and
// discards everything else without allocating.
class FieldSet {
 public:
  explicit FieldSet(std::vector<std::string> keys) : keys_(std::move(keys)) {
    spans_.resize(keys_.size());
  }
  void Reset() {
    for (auto& s : spans_) s = Span{};
  }
  // Returns false if the line is not a well-formed single JSON object.
  bool Scan(const std::string& line) {
    Reset();
    const char* p = line.data();
    const char* e = p + line.size();
    p = SkipWs(p, e);
    if (p >= e || *p != '{') return false;
    ++p;
    p = SkipWs(p, e);
    if (p < e && *p == '}') return true;
    while (p < e) {
      const char* kb = nullptr;
      const char* ke = nullptr;
      p = ScanString(p, e, &kb, &ke);
      if (p == nullptr) return false;
      p = SkipWs(p, e);
      if (p >= e || *p != ':') return false;
      ++p;
      p = SkipWs(p, e);
      const char* vb = p;
      p = ScanValue(p, e);
      if (p == nullptr) return false;
      const std::string_view key(kb, ke - kb);
      for (size_t i = 0; i < keys_.size(); ++i) {
        if (keys_[i].size() == key.size() &&
            std::memcmp(keys_[i].data(), key.data(), key.size()) == 0) {
          spans_[i] = Span{vb, p};
          break;
        }
      }
      p = SkipWs(p, e);
      if (p >= e) return false;
      if (*p == ',') {
        ++p;
        p = SkipWs(p, e);
        continue;
      }
      if (*p == '}') return true;
      return false;
    }
    return false;
  }
  const Span& At(size_t i) const { return spans_[i]; }
  const std::string& Key(size_t i) const { return keys_[i]; }

 private:
  std::vector<std::string> keys_;
  std::vector<Span> spans_;
};

int64_t ParseInt(const Span& s, const std::string& what) {
  if (!s.present()) Stop("missing field " + what);
  char* end = nullptr;
  const int64_t v = std::strtoll(s.b, &end, 10);
  if (end == s.b) Stop("field " + what + " is not an integer: " + std::string(s.view()));
  return v;
}

// The stream writes every registered double as a decimal STRING (%.17g). Parse
// the string, then narrow -- never read the *_legacy6 sibling, which json.cc
// has already floored to six decimals.
double ParseQuotedDouble(const Span& s, const std::string& what) {
  if (!s.present()) Stop("missing field " + what);
  const char* p = s.b;
  if (p >= s.e || *p != '"') Stop("field " + what + " is not a quoted decimal");
  ++p;
  char* end = nullptr;
  const double v = std::strtod(p, &end);
  if (end == p) Stop("field " + what + " is not a decimal: " + std::string(s.view()));
  return v;
}

void ParseIntArray(const Span& s, const std::string& what, std::vector<int32_t>* out) {
  out->clear();
  if (!s.present()) Stop("missing field " + what);
  const char* p = s.b;
  const char* e = s.e;
  p = SkipWs(p, e);
  if (p >= e || *p != '[') Stop("field " + what + " is not an array");
  ++p;
  p = SkipWs(p, e);
  if (p < e && *p == ']') return;
  while (p < e) {
    char* end = nullptr;
    const long long v = std::strtoll(p, &end, 10);
    if (end == p) Stop("field " + what + " has a non-integer element");
    out->push_back(static_cast<int32_t>(v));
    p = SkipWs(end, e);
    if (p >= e) Stop("field " + what + " is truncated");
    if (*p == ',') {
      p = SkipWs(p + 1, e);
      continue;
    }
    if (*p == ']') return;
    Stop("field " + what + " is malformed");
  }
  Stop("field " + what + " is truncated");
}

void ParseInt64Array(const Span& s, const std::string& what, std::vector<Action>* out) {
  out->clear();
  if (!s.present()) Stop("missing field " + what);
  const char* p = s.b;
  const char* e = s.e;
  p = SkipWs(p, e);
  if (p >= e || *p != '[') Stop("field " + what + " is not an array");
  ++p;
  p = SkipWs(p, e);
  if (p < e && *p == ']') return;
  while (p < e) {
    char* end = nullptr;
    const long long v = std::strtoll(p, &end, 10);
    if (end == p) Stop("field " + what + " has a non-integer element");
    out->push_back(static_cast<Action>(v));
    p = SkipWs(end, e);
    if (p >= e) Stop("field " + what + " is truncated");
    if (*p == ',') {
      p = SkipWs(p + 1, e);
      continue;
    }
    if (*p == ']') return;
    Stop("field " + what + " is malformed");
  }
  Stop("field " + what + " is truncated");
}

// Array of decimal STRINGS, e.g. ["0.005", "0.905", ...].
void ParseQuotedDoubleArray(const Span& s, const std::string& what,
                            std::vector<double>* out) {
  out->clear();
  if (!s.present()) Stop("missing field " + what);
  const char* p = s.b;
  const char* e = s.e;
  p = SkipWs(p, e);
  if (p >= e || *p != '[') Stop("field " + what + " is not an array");
  ++p;
  p = SkipWs(p, e);
  if (p < e && *p == ']') return;
  while (p < e) {
    if (*p != '"') Stop("field " + what + " element is not a quoted decimal");
    ++p;
    char* end = nullptr;
    const double v = std::strtod(p, &end);
    if (end == p) Stop("field " + what + " element is not a decimal");
    out->push_back(v);
    p = end;
    if (p >= e || *p != '"') Stop("field " + what + " element is unterminated");
    ++p;
    p = SkipWs(p, e);
    if (p >= e) Stop("field " + what + " is truncated");
    if (*p == ',') {
      p = SkipWs(p + 1, e);
      continue;
    }
    if (*p == ']') return;
    Stop("field " + what + " is malformed");
  }
  Stop("field " + what + " is truncated");
}

// ---------------------------------------------------------------------------
// Records
// ---------------------------------------------------------------------------
struct GameRec {
  int32_t game_index = -1;
  int32_t rounds_played = -1;
  int32_t searched_seat = -1;
  int32_t n_label_rows = -1;
  std::array<int32_t, 4> final_vp{{0, 0, 0, 0}};
  std::vector<Action> history;
};

struct LabelRow {
  int32_t game_index = -1;
  int32_t decision_index = -1;
  int64_t history_length = -1;
  int32_t acting_player = -1;
  int32_t searched_seat = -1;
  int32_t n_legal = -1;
  int32_t weight = -1;
  int32_t executed_action = -1;
  int32_t num_covered_actions = -1;
  float teacher_kl = 0.0f;
  std::vector<int32_t> legal_actions;
  std::vector<float> policy_target;
  std::vector<float> raw_prior;
};

// ---------------------------------------------------------------------------
// WHAT games.jsonl's `final_vp` ACTUALLY IS -- and why it is not FinalScoredVp
// ---------------------------------------------------------------------------
// Registration 5.3 assertion 5 says the replayed terminal `FinalScoredVp` by
// seat must equal games.jsonl's `final_vp`. IT DOES NOT, and the reason is in
// the generator, not in the replay:
//
//   games.jsonl's final_vp was emitted by GetTrueFinalVp
//   (dune_pwo4_trajectory.cc:401-445, itself a verbatim copy of
//   dune_search_benchmark.cc:147-202), NOT by DuneImperiumState::FinalScoredVp.
//
// The two functions differ in one clause. `FinalScoredVp(p)` is
// `vp_[p] + ComputeEndgameVp(p)` (dune_imperium.cc:2398-2403), and
// ComputeEndgameVp awards the "all four factions at influence >= 3" bonus
// ONLY IF the player owns tech tile 8 (dune_imperium.cc, the tech-tile-8
// branch). GetTrueFinalVp awards that same +1 UNCONDITIONALLY -- it never
// tests for tech tile 8. So the recorded field OVERSTATES by 1 for any seat
// that reached >= 3 influence with all four factions without owning tech
// tile 8.
//
// Which one is the game's real score is not in doubt: DuneImperiumState::
// Returns() -- the placement the engine actually pays out -- ranks on
// `vp_[p] + ComputeEndgameVp(p)`, i.e. on FinalScoredVp. GetTrueFinalVp is a
// reporting-only helper.
//
// AMENDMENT 1 RULING 3 restates assertion 5 as three parts, and this tool
// implements exactly that:
//   * 5a -- GetCurrentRound() - 1 == games.jsonl rounds_played. Hard STOP.
//   * 5b -- the recomputed PWO-4 legacy reported VP equals the recorded
//     final_vp field. This is the REPLAY-FIDELITY content: it compares the
//     function whose output was actually recorded, so a mismatch really does
//     mean the replay landed on a different terminal state. Hard STOP.
//   * 5c -- FinalScoredVp is recomputed INDEPENDENTLY for the head target and
//     is NOT required to equal the recorded field. The 212/1,600 all-seat and
//     59/400 searched-seat divergences are PINNED as corpus-audit facts
//     (kExpectedVpDivergences above); a different count is a STOP.
//   * the final_vp_head target uses FinalScoredVp -- amendment 1 ruling 1.
//
// THE SOURCE BUG IS FIXED ELSEWHERE, AND THIS COPY SURVIVES ON PURPOSE.
// dune_pwo4_trajectory.cc and dune_search_benchmark.cc no longer carry their
// own endgame arithmetic; both now report through
// DuneImperiumState::FinalScoredVp. The function below is the FROZEN
// HISTORICAL REPRODUCTION of the arithmetic that wrote the frozen corpus,
// bug included, retained for one purpose only: auditing that corpus. It is
// NOT a scoring function, it is not what any future artifact will contain,
// and nothing else in this file may call it.
int Pwo4LegacyReportedVp(const dune_imperium::DuneImperiumState* dune_state,
                             int p) {
  int base_vp = dune_state->GetPlayerVpForTesting(p);
  int endgame_vp = 0;

  const auto& hand = dune_state->GetIntrigueHandForTesting(p);
  for (int intrigue_id : hand) {
    const auto* intrigue = dune_imperium::FindIntrigueCardById(intrigue_id);
    if (intrigue &&
        (intrigue->phase_mask & dune_imperium::kIntriguePhaseEndgameMask) != 0) {
      endgame_vp += dune_state->EndgameIntrigueVpBonusForTesting(p, intrigue_id);
    }
  }

  const auto& tech_tiles = dune_state->GetPlayerTechTilesForTesting(p);
  if (std::find(tech_tiles.begin(), tech_tiles.end(), 6) != tech_tiles.end()) {
    int tsmf_count = 0;
    auto count_tsmf = [&](const std::vector<int>& cards) {
      for (int id : cards) {
        if (id == dune_imperium::kCardTheSpiceMustFlow) tsmf_count++;
      }
    };
    auto* mutable_state = const_cast<dune_imperium::DuneImperiumState*>(dune_state);
    count_tsmf(mutable_state->GetPlayerDrawDeckForTesting(p));
    count_tsmf(mutable_state->GetPlayerDiscardForTesting(p));
    count_tsmf(mutable_state->GetPlayerHandForTesting(p));
    count_tsmf(dune_state->GetPlayedAgentCardsForTesting(p));
    count_tsmf(dune_state->GetRevealedCardsForTesting(p));
    if (tsmf_count >= 2) endgame_vp += 1;
  }

  // NOTE: no tech-tile-8 test here. This is the divergence.
  bool all_3 = true;
  for (int f = 0; f < 4; ++f) {
    if (dune_state->GetPlayerInfluenceForTesting(
            p, static_cast<dune_imperium::Faction>(f)) < 3) {
      all_3 = false;
    }
  }
  if (all_3) endgame_vp += 1;

  if (std::find(tech_tiles.begin(), tech_tiles.end(), 14) != tech_tiles.end()) {
    int low_influence_factions = 0;
    for (int f = 0; f < 4; ++f) {
      if (dune_state->GetPlayerInfluenceForTesting(
              p, static_cast<dune_imperium::Faction>(f)) <= 1) {
        low_influence_factions++;
      }
    }
    endgame_vp += low_influence_factions;
  }
  return base_vp + endgame_vp;
}

int TerminalClass(int rounds_played) {
  if (rounds_played <= 8) return 0;
  if (rounds_played == 9) return 1;
  return 2;
}
std::string JoinInts(const std::vector<int32_t>& v) {
  std::ostringstream oss;
  for (size_t i = 0; i < v.size(); ++i) {
    if (i != 0) oss << ",";
    oss << v[i];
  }
  return oss.str();
}

void ReadGames(const std::string& path, std::map<int32_t, GameRec>* out) {
  std::ifstream f(path);
  if (!f) Stop("cannot open games_path: " + path);
  FieldSet fs({"game_index", "history", "rounds_played", "final_vp",
               "searched_seat", "n_label_rows", "history_length"});
  enum { kGi, kHist, kRounds, kFinalVp, kSeat, kNRows, kHistLen };
  std::string line;
  int64_t lineno = 0;
  std::vector<int32_t> vp;
  while (std::getline(f, line)) {
    ++lineno;
    if (line.empty()) continue;
    if (!fs.Scan(line))
      Stop("games.jsonl line " + std::to_string(lineno) + " is not a JSON object");
    GameRec g;
    g.game_index = static_cast<int32_t>(ParseInt(fs.At(kGi), "games.game_index"));
    g.rounds_played = static_cast<int32_t>(ParseInt(fs.At(kRounds), "games.rounds_played"));
    g.searched_seat = static_cast<int32_t>(ParseInt(fs.At(kSeat), "games.searched_seat"));
    g.n_label_rows = static_cast<int32_t>(ParseInt(fs.At(kNRows), "games.n_label_rows"));
    ParseInt64Array(fs.At(kHist), "games.history", &g.history);
    ParseIntArray(fs.At(kFinalVp), "games.final_vp", &vp);
    Require(vp.size() == 4, "games.final_vp is not length 4 on game " +
                                std::to_string(g.game_index));
    for (int i = 0; i < 4; ++i) g.final_vp[i] = vp[i];
    const int64_t hl = ParseInt(fs.At(kHistLen), "games.history_length");
    Require(static_cast<int64_t>(g.history.size()) == hl,
            "games.history_length disagrees with |history| on game " +
                std::to_string(g.game_index));
    Require(out->find(g.game_index) == out->end(),
            "duplicate game_index " + std::to_string(g.game_index));
    (*out)[g.game_index] = std::move(g);
  }
}

void ReadLabels(const std::string& path,
                std::map<int32_t, std::vector<LabelRow>>* out, int64_t* n_rows) {
  std::ifstream f(path);
  if (!f) Stop("cannot open labels_path: " + path);
  FieldSet fs({"game_index", "decision_index", "history_length", "acting_player",
               "searched_seat", "legal_actions", "n_legal", "weight",
               "executed_action", "policy_target_exact", "raw_prior_vector_exact",
               "raw_to_search_policy_kl_exact", "num_covered_actions"});
  enum {
    kGi, kDi, kHl, kAp, kSeat, kLegal, kNLegal, kW, kExec, kPt, kRp, kKl, kCov
  };
  std::string line;
  int64_t lineno = 0;
  std::vector<double> tmp;
  *n_rows = 0;
  while (std::getline(f, line)) {
    ++lineno;
    if (line.empty()) continue;
    if (!fs.Scan(line))
      Stop("labels.jsonl line " + std::to_string(lineno) + " is not a JSON object");
    LabelRow r;
    r.game_index = static_cast<int32_t>(ParseInt(fs.At(kGi), "labels.game_index"));
    r.decision_index = static_cast<int32_t>(ParseInt(fs.At(kDi), "labels.decision_index"));
    r.history_length = ParseInt(fs.At(kHl), "labels.history_length");
    r.acting_player = static_cast<int32_t>(ParseInt(fs.At(kAp), "labels.acting_player"));
    r.searched_seat = static_cast<int32_t>(ParseInt(fs.At(kSeat), "labels.searched_seat"));
    r.n_legal = static_cast<int32_t>(ParseInt(fs.At(kNLegal), "labels.n_legal"));
    r.weight = static_cast<int32_t>(ParseInt(fs.At(kW), "labels.weight"));
    r.executed_action = static_cast<int32_t>(ParseInt(fs.At(kExec), "labels.executed_action"));
    r.num_covered_actions =
        static_cast<int32_t>(ParseInt(fs.At(kCov), "labels.num_covered_actions"));
    r.teacher_kl = static_cast<float>(
        ParseQuotedDouble(fs.At(kKl), "labels.raw_to_search_policy_kl_exact"));
    ParseIntArray(fs.At(kLegal), "labels.legal_actions", &r.legal_actions);
    ParseQuotedDoubleArray(fs.At(kPt), "labels.policy_target_exact", &tmp);
    r.policy_target.assign(tmp.begin(), tmp.end());
    ParseQuotedDoubleArray(fs.At(kRp), "labels.raw_prior_vector_exact", &tmp);
    r.raw_prior.assign(tmp.begin(), tmp.end());

    const std::string where = "game " + std::to_string(r.game_index) +
                              " decision " + std::to_string(r.decision_index);
    // Parse self-check: the stream's own redundancy. A field scanner that
    // grabbed the wrong span would fail here immediately.
    Require(r.n_legal > 0, "n_legal <= 0 at " + where);
    Require(static_cast<int32_t>(r.legal_actions.size()) == r.n_legal,
            "ASSERTION 2 (data side): |legal_actions| != n_legal at " + where);
    Require(static_cast<int32_t>(r.policy_target.size()) == r.n_legal,
            "|policy_target_exact| != n_legal at " + where);
    Require(static_cast<int32_t>(r.raw_prior.size()) == r.n_legal,
            "|raw_prior_vector_exact| != n_legal at " + where);
    Require(r.weight == 0 || r.weight == 1, "weight not in {0,1} at " + where);
    // Registration 5.3 assertion 3 / 8.3(b): one searched seat per game.
    Require(r.acting_player == r.searched_seat,
            "acting_player != searched_seat at " + where);

    (*out)[r.game_index].push_back(std::move(r));
    ++(*n_rows);
  }
}

// ---------------------------------------------------------------------------
// Read the written pack back with the READER's own walk, not the writer's.
//
// SearchLabelBuffer::LoadFile (dune_ppo_train.cc:466-597) `break`s out of its
// record loop on any short read or malformed field and then silently keeps
// whatever it had -- it does not fail. A one-byte layout error would therefore
// show up at training time as a quietly smaller buffer, not as an error. This
// re-reads the file with the identical field order and the identical
// IsValidationLabel call, so the counts written into manifest.json are the
// counts the loader will produce, verified rather than predicted.
// ---------------------------------------------------------------------------
void VerifyPackAgainstReader(const std::string& path, int64_t expect_obs_size,
                             int64_t expect_action_dim, int64_t* out_rows,
                             int64_t* out_validation) {
  std::ifstream in(path, std::ios::binary);
  if (!in) Stop("verification: cannot reopen " + path);

  uint32_t magic = 0, schema = 0, reserved = 0;
  int32_t obs_size = 0, action_dim = 0, max_simulations = 0, min_visits = 0,
          min_coverage = 0;
  float utility_divisor = 0, puct_c = 0, target_teacher_kl = 0, blueprint_temp = 0;
  uint64_t fingerprint = 0;
  in.read(reinterpret_cast<char*>(&magic), 4);
  Require(magic == 0x4c545344u, "verification: bad pack magic");
  in.read(reinterpret_cast<char*>(&schema), 4);
  Require(schema == 1 || schema == 2, "verification: bad pack schema");
  in.read(reinterpret_cast<char*>(&obs_size), 4);
  in.read(reinterpret_cast<char*>(&action_dim), 4);
  Require(obs_size == expect_obs_size, "verification: pack obs_size mismatch");
  Require(action_dim == expect_action_dim, "verification: pack action_dim mismatch");
  in.read(reinterpret_cast<char*>(&max_simulations), 4);
  in.read(reinterpret_cast<char*>(&utility_divisor), 4);
  Require(utility_divisor == 4.0f,
          "verification: pack utility_divisor != 4.0 -- the loader would reject it");
  in.read(reinterpret_cast<char*>(&puct_c), 4);
  in.read(reinterpret_cast<char*>(&target_teacher_kl), 4);
  in.read(reinterpret_cast<char*>(&min_visits), 4);
  in.read(reinterpret_cast<char*>(&min_coverage), 4);
  in.read(reinterpret_cast<char*>(&blueprint_temp), 4);
  in.read(reinterpret_cast<char*>(&fingerprint), 8);
  in.read(reinterpret_cast<char*>(&reserved), 4);
  Require(static_cast<bool>(in), "verification: truncated pack header");

  int64_t rows = 0, validation = 0;
  std::vector<float> state(static_cast<size_t>(obs_size));
  std::vector<int32_t> legal_actions;
  while (in.peek() != EOF) {
    in.read(reinterpret_cast<char*>(state.data()), obs_size * sizeof(float));
    Require(static_cast<bool>(in), "verification: truncated observation at row " +
                                       std::to_string(rows));
    int32_t num_legal = 0;
    in.read(reinterpret_cast<char*>(&num_legal), sizeof(int32_t));
    Require(num_legal > 0 && num_legal <= action_dim,
            "verification: invalid num_legal_actions " + std::to_string(num_legal) +
                " at row " + std::to_string(rows));
    legal_actions.clear();
    for (int32_t i = 0; i < num_legal; ++i) {
      int32_t action_id = 0;
      float t_prob = 0.0f, p_prob = 0.0f;
      in.read(reinterpret_cast<char*>(&action_id), sizeof(int32_t));
      in.read(reinterpret_cast<char*>(&t_prob), sizeof(float));
      in.read(reinterpret_cast<char*>(&p_prob), sizeof(float));
      Require(action_id >= 0 && action_id < action_dim && std::isfinite(t_prob) &&
                  std::isfinite(p_prob),
              "verification: the loader would reject the triple at row " +
                  std::to_string(rows));
      legal_actions.push_back(action_id);
    }
    float teacher_kl = 0.0f, eta = 0.0f;
    int32_t num_covered = 0;
    uint8_t eta_capped = 0, pid = 0, padding[2] = {0, 0};
    in.read(reinterpret_cast<char*>(&teacher_kl), sizeof(float));
    in.read(reinterpret_cast<char*>(&num_covered), sizeof(int32_t));
    in.read(reinterpret_cast<char*>(&eta), sizeof(float));
    in.read(reinterpret_cast<char*>(&eta_capped), sizeof(uint8_t));
    in.read(reinterpret_cast<char*>(&pid), sizeof(uint8_t));
    in.read(reinterpret_cast<char*>(padding), 2);
    Require(static_cast<bool>(in),
            "verification: truncated record tail at row " + std::to_string(rows));
    if (IsValidationLabel(state, legal_actions, pid)) ++validation;
    ++rows;
  }
  *out_rows = rows;
  *out_validation = validation;
}

}  // namespace
}  // namespace open_spiel

int main(int argc, char** argv) {
  using namespace open_spiel;
  absl::ParseCommandLine(argc, argv);

  const std::string labels_path = absl::GetFlag(FLAGS_labels_path);
  const std::string games_path = absl::GetFlag(FLAGS_games_path);
  const std::string out_dir = absl::GetFlag(FLAGS_out_dir);
  const int expect_label_rows = absl::GetFlag(FLAGS_expect_label_rows);
  const int expect_games = absl::GetFlag(FLAGS_expect_games);
  const int heldout_games = absl::GetFlag(FLAGS_heldout_games);

  Require(!labels_path.empty(), "--labels_path is required");
  Require(!games_path.empty(), "--games_path is required");
  Require(!out_dir.empty(), "--out_dir is required");
  Require(labels_path.size() < 3 ||
              labels_path.substr(labels_path.size() - 3) != ".gz",
          "--labels_path must be a PLAIN .jsonl. Decompress labels.jsonl.gz "
          "first; this binary links no decompressor.");

  std::filesystem::create_directories(out_dir);
  const std::filesystem::path pack_dir = std::filesystem::path(out_dir) / "search_labels";
  std::filesystem::create_directories(pack_dir);

  std::cout << "=========================================================\n"
            << "dune_pwo5_prepare -- PWO-5 offline preparation pass\n"
            << "registration: docs/PWO5_PILOT_REGISTRATION.md 5.3/5.4/6.1/8.5\n"
            << "=========================================================\n"
            << "labels_path : " << labels_path << "\n"
            << "games_path  : " << games_path << "\n"
            << "out_dir     : " << out_dir << "\n\n";

  // -------------------------------------------------------------------------
  // Engine + tensor geometry. Identical derivation to dune_ppo_train.cc:1790.
  // -------------------------------------------------------------------------
  std::shared_ptr<const Game> game = LoadGame("dune_imperium");
  const bool provides_info_state_tensor =
      game->GetType().provides_information_state_tensor;
  const bool provides_observations_tensor =
      game->GetType().provides_observation_tensor;
  const int64_t obs_size = provides_info_state_tensor
                               ? game->InformationStateTensorSize()
                               : game->ObservationTensorSize();
  const int64_t action_dim = game->NumDistinctActions();
  Require(provides_info_state_tensor || provides_observations_tensor,
          "the game provides neither an information-state nor an observation tensor");
  std::cout << "obs_size   = " << obs_size
            << " (information_state_tensor=" << (provides_info_state_tensor ? "yes" : "no")
            << ")\naction_dim = " << action_dim << "\n\n";
  // Registration 8.3(e): the next_own_action vocabulary is the full action space.
  Require(action_dim == 2391,
          "NumDistinctActions() != 2391, which registration 8.3(e) pins");

  // -------------------------------------------------------------------------
  // Read the stream.
  // -------------------------------------------------------------------------
  std::map<int32_t, GameRec> games;
  ReadGames(games_path, &games);
  std::cout << "games.jsonl : " << games.size() << " games\n";
  Require(static_cast<int>(games.size()) == expect_games,
          "games.jsonl has " + std::to_string(games.size()) + " games, expected " +
              std::to_string(expect_games));

  std::map<int32_t, std::vector<LabelRow>> rows_by_game;
  int64_t n_rows = 0;
  ReadLabels(labels_path, &rows_by_game, &n_rows);
  std::cout << "labels.jsonl: " << n_rows << " rows over " << rows_by_game.size()
            << " games\n\n";
  Require(n_rows == expect_label_rows,
          "labels.jsonl has " + std::to_string(n_rows) + " rows, expected " +
              std::to_string(expect_label_rows));
  Require(rows_by_game.size() == games.size(),
          "label rows cover " + std::to_string(rows_by_game.size()) +
              " games but games.jsonl has " + std::to_string(games.size()));

  for (auto& kv : rows_by_game) {
    auto& rows = kv.second;
    std::sort(rows.begin(), rows.end(),
              [](const LabelRow& a, const LabelRow& b) {
                return a.decision_index < b.decision_index;
              });
    auto git = games.find(kv.first);
    Require(git != games.end(),
            "label rows for unknown game_index " + std::to_string(kv.first));
    Require(static_cast<int32_t>(rows.size()) == git->second.n_label_rows,
            "game " + std::to_string(kv.first) + " has " +
                std::to_string(rows.size()) + " label rows but games.jsonl says " +
                std::to_string(git->second.n_label_rows));
    for (size_t i = 0; i < rows.size(); ++i) {
      Require(rows[i].searched_seat == git->second.searched_seat,
              "searched_seat disagrees between labels and games on game " +
                  std::to_string(kv.first));
      if (i > 0) {
        Require(rows[i].decision_index > rows[i - 1].decision_index,
                "duplicate decision_index in game " + std::to_string(kv.first));
        Require(rows[i].history_length > rows[i - 1].history_length,
                "history_length not strictly increasing in game " +
                    std::to_string(kv.first));
      }
    }
  }
  // Registration 8.4: rounds 1-6 are empty at every denominator; 10 is an exact
  // terminal class, not an overflow (kMaxRounds = 10).
  for (const auto& kv : games) {
    Require(kv.second.rounds_played >= 7 && kv.second.rounds_played <= 10,
            "game " + std::to_string(kv.first) + " has rounds_played " +
                std::to_string(kv.second.rounds_played) + ", outside [7,10]");
  }

  // -------------------------------------------------------------------------
  // (2) The whole-trajectory held-out split -- registration 8.5.
  //
  // Deterministic, no RNG, no seed. Stratified on the three terminal classes.
  //
  // HOW THE PER-CLASS HELD-OUT COUNT IS OBTAINED: it is COMPUTED as
  // round(n_class * heldout_games / n_games), NOT hard-coded. On the registered
  // corpus that yields round(5.25)=5, round(16.2)=16, round(38.55)=39 -- exactly
  // the registered 5 / 16 / 39 -- and the tool asserts both the class sizes
  // (35 / 108 / 257) and the derived counts against those registered values, so
  // the computation is guarded on both sides.
  // -------------------------------------------------------------------------
  std::array<std::vector<int32_t>, 3> class_games;  // ascending, by construction
  for (const auto& kv : games) {
    class_games[TerminalClass(kv.second.rounds_played)].push_back(kv.first);
  }
  const bool registered_config =
      (expect_games == 400 && heldout_games == 60 && expect_label_rows == 48443);
  if (!registered_config) {
    std::cout << "*** WARNING: this is NOT the registered configuration "
                 "(400 games / 60 held out / 48,443 rows). The registered "
                 "class-size and split-size assertions are SKIPPED.\n\n";
  } else {
    Require(class_games[0].size() == 35 && class_games[1].size() == 108 &&
                class_games[2].size() == 257,
            "terminal-class sizes are not the registered 35 / 108 / 257");
  }

  std::array<int, 3> n_held{{0, 0, 0}};
  for (int c = 0; c < 3; ++c) {
    n_held[c] = static_cast<int>(std::llround(
        static_cast<double>(class_games[c].size()) *
        static_cast<double>(heldout_games) / static_cast<double>(games.size())));
  }
  Require(n_held[0] + n_held[1] + n_held[2] == heldout_games,
          "proportional rounding does not sum to --heldout_games");
  if (registered_config) {
    Require(n_held[0] == 5 && n_held[1] == 16 && n_held[2] == 39,
            "derived per-class held-out counts are not the registered 5 / 16 / 39");
  }

  // Registration 8.5 algorithm, verbatim: within each class, sort the class's
  // game_index values ascending, then take positions floor(k*n_class/n_held)
  // for k = 0 .. n_held-1.
  std::vector<int32_t> heldout;
  for (int c = 0; c < 3; ++c) {
    const int64_t n_class = static_cast<int64_t>(class_games[c].size());
    for (int k = 0; k < n_held[c]; ++k) {
      const int64_t pos = (static_cast<int64_t>(k) * n_class) / n_held[c];
      heldout.push_back(class_games[c][static_cast<size_t>(pos)]);
    }
  }
  std::sort(heldout.begin(), heldout.end());
  Require(static_cast<int>(heldout.size()) == heldout_games,
          "held-out selection did not produce --heldout_games games");
  Require(std::adjacent_find(heldout.begin(), heldout.end()) == heldout.end(),
          "held-out selection produced a duplicate game_index");
  std::map<int32_t, bool> is_heldout;
  for (int32_t g : heldout) is_heldout[g] = true;

  // The canonical serialization is the MEMBERSHIP LIST ONLY -- the 60 ascending
  // game_index values joined by "," with no spaces and no trailing comma, utf-8
  // -- NOT the JSON file that carries it. The digest therefore survives any
  // reformatting of heldout_games.json, which is the point: it pins the split,
  // not the file.
  const std::string membership_canonical = JoinInts(heldout);
  const std::string membership_sha256 = ComputeStringSHA256(membership_canonical);

  {
    std::ostringstream j;
    j << "{\n  \"heldout_game_indices\": [";
    for (size_t i = 0; i < heldout.size(); ++i) {
      j << (i ? ", " : "") << heldout[i];
    }
    j << "],\n  \"n_heldout\": " << heldout.size()
      << ",\n  \"class_counts\": {\"le8\": " << class_games[0].size()
      << ", \"r9\": " << class_games[1].size()
      << ", \"r10\": " << class_games[2].size()
      << "},\n  \"class_heldout\": {\"le8\": " << n_held[0]
      << ", \"r9\": " << n_held[1] << ", \"r10\": " << n_held[2]
      << "},\n  \"membership_sha256\": \"" << membership_sha256 << "\"\n}\n";
    const std::string p = (std::filesystem::path(out_dir) / "heldout_games.json").string();
    std::ofstream o(p);
    if (!o) Stop("cannot write " + p);
    o << j.str();
    o.close();
    std::cout << "--- section 8.5 whole-trajectory split ---\n"
              << "class sizes      : le8=" << class_games[0].size()
              << " r9=" << class_games[1].size() << " r10=" << class_games[2].size()
              << "\nheld out / class : le8=" << n_held[0] << " r9=" << n_held[1]
              << " r10=" << n_held[2] << " (total " << heldout.size() << ")\n"
              << "membership       : " << membership_canonical << "\n"
              << "membership_sha256: " << membership_sha256 << "\n"
              << "written          : " << p << "\n\n";
  }

  // -------------------------------------------------------------------------
  // (1)+(3)+(4) ONE replay pass: the fidelity gate, the auxiliary targets, and
  // the distillation pack. Registration 5.4: the targets are derived in the
  // same pass that feeds the gate, so no target can be computed against a state
  // the gate did not verify.
  // -------------------------------------------------------------------------
  const std::string aux_path = (std::filesystem::path(out_dir) / "aux_targets.bin").string();
  // Amendment 1 ruling 5: TWO packs with EXPLICIT roles, not one pack split by
  // a per-row hash.
  //
  //   labels_train_000.bin      role "train"      -- the weight-one rows of the
  //                                                  340 NON-held-out games
  //   labels_validation_000.bin role "validation" -- the weight-one rows of the
  //                                                  60 HELD-OUT games
  //
  // The train pack bypasses IsValidationLabel entirely, restoring the
  // registered 20,582 rows and 14.93 expected passes. The validation pack
  // buys a genuine whole-trajectory-held-out distillation KL, which the %11
  // bucket never measured -- and which is DESCRIPTIVE ONLY: no gate,
  // threshold or nomination criterion reads it (ruling 5, section 6.4).
  const std::string pack_train_name = "labels_train_000.bin";
  const std::string pack_val_name = "labels_validation_000.bin";
  const std::string pack_train_path = (pack_dir / pack_train_name).string();
  const std::string pack_val_path = (pack_dir / pack_val_name).string();

  auto aux = std::make_unique<BinWriter>(aux_path);
  aux->U32(0x50573541u);                          // magic
  aux->U32(1u);                                   // schema
  aux->U32(static_cast<uint32_t>(obs_size));      // obs_size
  aux->U32(static_cast<uint32_t>(action_dim));    // action_dim
  aux->U64(static_cast<uint64_t>(n_rows));        // num_rows
  aux->U64(0u);                                   // reserved

  // Legacy pack header -- byte for byte as SearchLabelBuffer::LoadFile reads it.
  // The descriptive fields carry PWO-4's own controller pin
  // (calibration_results_v2/pwo4_trajectory/manifest.json controller_pin), so
  // the file describes the teacher that produced it. Only magic / schema /
  // obs_size / action_dim / utility_divisor are checked by the reader;
  // utility_divisor MUST be exactly 4.0f or the file is rejected.
  auto pack = std::make_unique<BinWriter>(pack_train_path);
  auto pack_val = std::make_unique<BinWriter>(pack_val_path);
  {
    // The 64-bit header fingerprint is a file-identity tag the loader uses only
    // to detect MIXED fingerprints across several .bin files in one directory.
    // BOTH packs carry the SAME value on purpose: they are two role-partitioned
    // halves of one teacher stream, so a "mixed fingerprints" warning between
    // them would be a false positive. The value must be written before the row
    // counts are known, so it is derived from the count-independent teacher
    // identity. It is NOT the manifest's search_label_fingerprint, which is a
    // sha256 over a semantic object that does include the counts and the roles.
    const std::string fp_domain =
        "PWO5|pwo4_trajectory|model=4bffd2b7d73fc684344947d2064560d5c367c63e28"
        "a124fc87010de04e021f79|engine=a67f1925305d35d9a2f30fd9658b9f01fe21bdd6";
    const std::string fp_hex = ComputeStringSHA256(fp_domain);
    uint64_t fp = 0;
    for (int i = 0; i < 16; ++i) {
      const char ch = fp_hex[i];
      const uint64_t nib = (ch >= '0' && ch <= '9') ? (ch - '0') : (ch - 'a' + 10);
      fp = (fp << 4) | nib;
    }
    for (BinWriter* w : {pack.get(), pack_val.get()}) {
      w->U32(0x4c545344u);                        // magic "DSTL"
      w->U32(2u);                                 // schema (loader accepts 1 or 2)
      w->I32(static_cast<int32_t>(obs_size));
      w->I32(static_cast<int32_t>(action_dim));
      w->I32(200);                                // max_simulations (PWO-4 pin)
      w->F32(4.0f);                               // utility_divisor -- ASSERTED == 4.0
      w->F32(0.3f);                               // puct_c (PWO-4 pin)
      w->F32(0.0f);                               // target_teacher_kl (unused offline)
      w->I32(2);                                  // min_visits (PWO-4 min_visit_threshold)
      w->I32(0);                                  // min_coverage (PWO-4 uses a prior-mass gate)
      w->F32(1.0f);                               // blueprint_temp (root_prior_temperature)
      w->U64(fp);
      w->U32(0u);                                 // reserved
    }
  }

  // Counters
  int64_t rows_written = 0;
  int64_t rows_heldout = 0;
  int64_t rows_training = 0;
  int64_t w1_total = 0;
  int64_t w1_heldout = 0;
  int64_t w1_training = 0;
  int64_t rows_no_next = 0;
  int64_t pack_rows = 0;               // rows written to the "train"-role pack
  int64_t pack_val_rows = 0;           // rows written to the "validation"-role pack
  // What the loader's %11 rule WOULD have diverted out of the train pack. Under
  // ruling 5 the train pack declares role "train", so this diversion no longer
  // happens -- the number is measured and reported so the restored exposure is
  // visible rather than asserted.
  int64_t pack_loader_validation = 0;
  std::array<int64_t, 3> rows_by_class{{0, 0, 0}};
  std::array<int64_t, 3> games_by_class{{0, 0, 0}};
  int64_t games_checked = 0;
  // Assertion 5c bookkeeping (see Pwo4LegacyReportedVp).
  int64_t vp_divergences = 0;
  int64_t vp_divergences_acting_seat = 0;
  std::vector<std::string> vp_divergence_examples;
  int64_t acting_vp_sum = 0, acting_vp_recorded_sum = 0;
  int acting_vp_min = 1 << 30, acting_vp_max = -(1 << 30);
  int acting_vp_recorded_min = 1 << 30, acting_vp_recorded_max = -(1 << 30);
  // Amendment 1 ruling 2: the COMPLETE histogram of the acting seat's VP, on
  // BOTH functions, is a registered output -- the Huber-delta justification is
  // re-derived against the FinalScoredVp distribution and a bare min/max/mean
  // cannot support that. Indexed by VP; 0..40 covers the reachable range with
  // room to spare (the /20 divisor is registered on the premise that no game
  // reaches 20, which the histogram now makes checkable rather than asserted).
  std::array<int64_t, 41> acting_vp_hist{};
  std::array<int64_t, 41> acting_vp_recorded_hist{};

  std::vector<float> obs(static_cast<size_t>(obs_size), 0.0f);
  std::vector<std::vector<float>> game_obs;
  std::vector<int32_t> legal_i32;

  std::cout << "--- section 5.3 replay-fidelity gate (ALL rows) ---\n";

  for (const auto& gkv : games) {
    const GameRec& g = gkv.second;
    auto rit = rows_by_game.find(g.game_index);
    Require(rit != rows_by_game.end(),
            "game " + std::to_string(g.game_index) + " has no label rows");
    const std::vector<LabelRow>& rows = rit->second;

    game_obs.assign(rows.size(), std::vector<float>());

    std::unique_ptr<State> state = game->NewInitialState();
    size_t li = 0;
    for (size_t pos = 0; pos <= g.history.size(); ++pos) {
      while (li < rows.size() &&
             rows[li].history_length == static_cast<int64_t>(pos)) {
        const LabelRow& r = rows[li];
        const std::string where = "game " + std::to_string(r.game_index) +
                                  " decision_index " + std::to_string(r.decision_index) +
                                  " (history_length " + std::to_string(r.history_length) + ")";
        // Assertion 4: the replayed position index equals history_length.
        // Checked against the state's own history, not against the loop
        // counter, so it is an independent statement about the replay.
        Require(static_cast<int64_t>(state->History().size()) == r.history_length,
                "GATE assertion 4 FAILED: replayed position " +
                    std::to_string(state->History().size()) +
                    " != history_length at " + where);
        Require(!state->IsTerminal(),
                "GATE FAILED: replay is terminal at a label row -- " + where);
        // Assertion 3: current player equals acting_player. A chance node has
        // CurrentPlayer() == kChancePlayerId (-1) and fails this too.
        const Player cp = state->CurrentPlayer();
        Require(cp == r.acting_player,
                "GATE assertion 3 FAILED: replayed CurrentPlayer()=" +
                    std::to_string(cp) + " != acting_player=" +
                    std::to_string(r.acting_player) + " at " + where);
        // Assertion 1 (load-bearing): the legal action set matches exactly, as
        // a set AND in order.
        const std::vector<Action> legal = state->LegalActions();
        Require(legal.size() == r.legal_actions.size(),
                "GATE assertion 1 FAILED: replayed |LegalActions()|=" +
                    std::to_string(legal.size()) + " != recorded " +
                    std::to_string(r.legal_actions.size()) + " at " + where);
        for (size_t i = 0; i < legal.size(); ++i) {
          if (legal[i] != static_cast<Action>(r.legal_actions[i])) {
            Stop("GATE assertion 1 FAILED: legal action order/content differs at "
                 "index " + std::to_string(i) + " (replayed " +
                 std::to_string(legal[i]) + " vs recorded " +
                 std::to_string(r.legal_actions[i]) + ") at " + where);
          }
        }
        // Assertion 2: |legal_actions| == n_legal (checked on the data at parse
        // time too; here it closes against the replay).
        Require(static_cast<int32_t>(legal.size()) == r.n_legal,
                "GATE assertion 2 FAILED: replayed |LegalActions()| != n_legal at " + where);

        // The observation the trainer would feed the net. Identical branch and
        // identical fill to dune_ppo_train.cc:1549-1554.
        std::fill(obs.begin(), obs.end(), 0.0f);
        if (provides_info_state_tensor && cp >= 0) {
          state->InformationStateTensor(cp, absl::MakeSpan(obs));
        } else if (provides_observations_tensor && cp >= 0) {
          state->ObservationTensor(cp, absl::MakeSpan(obs));
        }
        game_obs[li] = obs;
        ++li;
      }
      if (pos == g.history.size()) break;
      state->ApplyAction(g.history[pos]);
    }

    Require(li == rows.size(),
            "GATE FAILED: game " + std::to_string(g.game_index) + " has " +
                std::to_string(rows.size() - li) +
                " label row(s) whose history_length exceeds the recorded history");

    // Assertion 5a: the replayed game is terminal and its terminal round equals
    // rounds_played. The VP half of registration 5.3 assertion 5 is split into
    // 5b and 5c below -- see Pwo4LegacyReportedVp for why it has to be.
    Require(state->IsTerminal(),
            "GATE assertion 5a FAILED: game " + std::to_string(g.game_index) +
                " is not terminal after replaying its full history");
    const auto* ds = dynamic_cast<const dune_imperium::DuneImperiumState*>(state.get());
    Require(ds != nullptr, "replayed state is not a DuneImperiumState");

    // WHAT IS COMPARED, AND WHY IT IS OFF BY ONE. The engine's round counter
    // OVERSHOOTS at terminal: DuneImperiumState::ApplyAction does `round_ += 1`
    // at the end of the Recall phase and only THEN tests for game end
    // (dune_imperium.cc:8890-8903), on both termination paths (round_ > 10 and
    // any seat at >= 11 VP). So at a terminal state GetCurrentRound() is
    // (last round actually played) + 1. games.jsonl's `rounds_played` was
    // emitted as exactly `GetCurrentRound() - 1` at the terminal state
    // (dune_pwo4_trajectory.cc:1490), so the comparison below is
    // GetCurrentRound() - 1 == rounds_played, i.e. the number of rounds PLAYED,
    // and the registered terminal classes (7,8,9,10 with kMaxRounds = 10) are
    // on that same scale.
    const int terminal_round_counter = ds->GetCurrentRound();
    Require(terminal_round_counter - 1 == g.rounds_played,
            "GATE assertion 5a FAILED: game " + std::to_string(g.game_index) +
                " replayed terminal round counter " +
                std::to_string(terminal_round_counter) + " (= " +
                std::to_string(terminal_round_counter - 1) +
                " rounds played) != games.jsonl rounds_played " +
                std::to_string(g.rounds_played));
    std::array<int, 4> replayed_vp{{0, 0, 0, 0}};
    for (Player p = 0; p < 4; ++p) {
      replayed_vp[p] = ds->FinalScoredVp(p);
      // Assertion 5b: reproduce the field the generator actually wrote. This
      // is the replay-fidelity content -- a mismatch means the replay reached
      // a different terminal state. Hard STOP.
      const int as_recorded = Pwo4LegacyReportedVp(ds, p);
      Require(as_recorded == g.final_vp[p],
              "GATE assertion 5b FAILED: game " + std::to_string(g.game_index) +
                  " recomputed GetTrueFinalVp(" + std::to_string(p) + ")=" +
                  std::to_string(as_recorded) + " != games.jsonl final_vp " +
                  std::to_string(g.final_vp[p]) +
                  " -- the replay did not reach the generator's terminal state");
      // Assertion 5c (amendment 1 ruling 3c): FinalScoredVp is recomputed
      // INDEPENDENTLY here, for the head target. It is NOT required to equal
      // the recorded field -- the amendment removed that requirement, because
      // the recorded field is a different function. The divergence is counted
      // and pinned after the loop against the amendment's audited values.
      if (replayed_vp[p] != g.final_vp[p]) {
        ++vp_divergences;
        if (p == g.searched_seat) ++vp_divergences_acting_seat;
        if (vp_divergence_examples.size() < 12) {
          vp_divergence_examples.push_back(
              "game " + std::to_string(g.game_index) + " seat " +
              std::to_string(p) + ": FinalScoredVp=" +
              std::to_string(replayed_vp[p]) + " games.jsonl final_vp=" +
              std::to_string(g.final_vp[p]));
        }
        // The divergence mechanism is exactly one clause (the tech-tile-8
        // guard), so the legacy field can only ever OVERSTATE by exactly 1.
        // Asserting the magnitude turns "we know why they differ" from a
        // comment into a check.
        Require(g.final_vp[p] - replayed_vp[p] == 1,
                "GATE assertion 5c FAILED: game " +
                    std::to_string(g.game_index) + " seat " +
                    std::to_string(p) + " differs by " +
                    std::to_string(g.final_vp[p] - replayed_vp[p]) +
                    ", but the only known divergence is the tech-tile-8 guard, "
                    "which can only overstate by exactly 1");
      }
    }
    {
      const int acting_vp = replayed_vp[g.searched_seat];
      acting_vp_sum += acting_vp;
      acting_vp_min = std::min(acting_vp_min, acting_vp);
      acting_vp_max = std::max(acting_vp_max, acting_vp);
      const int rec = g.final_vp[g.searched_seat];
      acting_vp_recorded_sum += rec;
      acting_vp_recorded_min = std::min(acting_vp_recorded_min, rec);
      acting_vp_recorded_max = std::max(acting_vp_recorded_max, rec);
      Require(acting_vp >= 0 && acting_vp < static_cast<int>(acting_vp_hist.size()),
              "acting-seat FinalScoredVp out of histogram range in game " +
                  std::to_string(g.game_index));
      Require(rec >= 0 && rec < static_cast<int>(acting_vp_recorded_hist.size()),
              "acting-seat recorded final_vp out of histogram range in game " +
                  std::to_string(g.game_index));
      ++acting_vp_hist[acting_vp];
      ++acting_vp_recorded_hist[rec];
    }

    // ----- targets + artifacts for this game -----
    const int tclass = TerminalClass(g.rounds_played);
    ++games_by_class[tclass];
    ++games_checked;

    for (size_t i = 0; i < rows.size(); ++i) {
      const LabelRow& r = rows[i];
      const uint8_t held = is_heldout.count(r.game_index) ? 1 : 0;
      // Registration 8.3: the target is the executed_action of the NEXT label
      // row of the same game by decision_index. The final label row of each
      // game has no target and is written as -1; its loss is omitted from BOTH
      // numerator and denominator by the trainer.
      const int32_t next_action =
          (i + 1 < rows.size()) ? rows[i + 1].executed_action : -1;
      if (next_action < 0) ++rows_no_next;
      // Registration 5.4: FinalScoredVp(acting_player) / 20, taken from the
      // REPLAYED terminal state (cross-checked against games.jsonl above).
      const float vp_target =
          static_cast<float>(replayed_vp[r.acting_player]) / 20.0f;

      aux->I32(r.game_index);
      aux->I32(r.decision_index);
      aux->I32(r.acting_player);
      aux->I32(static_cast<int32_t>(tclass));
      aux->F32(vp_target);
      aux->I32(next_action);
      aux->U8(static_cast<uint8_t>(r.weight));
      aux->U8(held);
      aux->U8(0);
      aux->U8(0);
      aux->Floats(game_obs[i]);

      ++rows_written;
      ++rows_by_class[tclass];
      if (held) ++rows_heldout; else ++rows_training;
      if (r.weight == 1) {
        ++w1_total;
        if (held) ++w1_heldout; else ++w1_training;
      }

      // ----- the two role-partitioned distillation packs -----
      // Every weight-one row goes to exactly one pack, chosen by WHOLE-GAME
      // held-out membership -- never by a per-row property. That is the whole
      // content of the section 8.5 split.
      if (r.weight == 1) {
        BinWriter* w = held ? pack_val.get() : pack.get();
        w->Floats(game_obs[i]);
        w->I32(r.n_legal);
        for (int32_t k = 0; k < r.n_legal; ++k) {
          w->I32(r.legal_actions[k]);
          w->F32(r.policy_target[k]);
          w->F32(r.raw_prior[k]);
        }
        w->F32(r.teacher_kl);
        w->I32(r.num_covered_actions);
        w->F32(0.0f);                                   // eta
        w->U8(0);                                       // eta_capped
        w->U8(static_cast<uint8_t>(r.acting_player));   // player_id
        w->U8(0);                                       // padding[0]
        w->U8(0);                                       // padding[1]
        if (held) {
          ++pack_val_rows;
        } else {
          ++pack_rows;
          // What the legacy %11 rule WOULD have diverted out of the train
          // pack, computed over EXACTLY the bytes the loader would hash:
          // legal_actions rebuilt from the stored teacher_probs action ids, in
          // the stored order, and player_id the uint8 widened to int32.
          // Measured so the exposure ruling 5 restores is a number in the
          // manifest rather than a claim in a comment.
          legal_i32.assign(r.legal_actions.begin(), r.legal_actions.end());
          if (IsValidationLabel(game_obs[i], legal_i32, r.acting_player)) {
            ++pack_loader_validation;
          }
        }
      }
    }

    if (games_checked % 50 == 0) {
      std::cout << "  replayed " << games_checked << " / " << games.size()
                << " games, " << rows_written << " rows checked\n"
                << std::flush;
    }
  }

  aux->Close();
  pack->Close();
  pack_val->Close();

  Require(rows_written == n_rows, "wrote fewer aux rows than were read");
  Require(games_checked == static_cast<int64_t>(games.size()),
          "did not replay every game");

  // Progress only. THE PASS BANNER IS AT THE END OF main(), after every check.
  // Printing it here -- which is what this line used to do -- meant the tool
  // announced PASS before the 5c pins, the histogram totals, the saturation
  // check, the pack read-back and the loader floors had run.
  std::cout << "\nreplay complete: " << games_checked << " games, "
            << rows_written << " label rows; assertions 1-4 and 5a-5b clean on "
               "every row. Running the remaining checks...\n\n";

  // -------------------------------------------------------------------------
  // Assertion 5c under amendment 1 ruling 3c: FinalScoredVp is recomputed
  // independently for the head target and is NOT required to equal the
  // recorded field. What IS required is that the divergence equals the pinned
  // audited values. See Pwo4LegacyReportedVp for the mechanism.
  // -------------------------------------------------------------------------
  std::cout << "--- assertion 5c: FinalScoredVp vs the legacy reported field ---\n";
  if (vp_divergences == 0) {
    std::cout << "  no divergence: FinalScoredVp == final_vp on all "
              << (games_checked * 4) << " (game, seat) pairs.\n\n";
  } else {
    std::cout
        << "  PINNED CORPUS-AUDIT FACT (amendment 1 ruling 3c). The two\n"
        << "  functions differ, they are EXPECTED to differ, and the cause is\n"
        << "  the generator, not the replay:\n"
        << "  games.jsonl final_vp was written by the since-REMOVED\n"
        << "  GetTrueFinalVp reporting helper, which awarded the 'all four\n"
        << "  factions at influence >= 3' +1 UNCONDITIONALLY, while\n"
        << "  DuneImperiumState::ComputeEndgameVp -- and therefore\n"
        << "  FinalScoredVp, and therefore Returns() and the engine's actual\n"
        << "  placement -- awards it ONLY to a seat owning tech tile 8\n"
        << "  (Memocorders). The recorded field OVERSTATES those seats by\n"
        << "  exactly 1 VP. That helper is now fixed at source in both\n"
        << "  dune_pwo4_trajectory.cc and dune_search_benchmark.cc; the frozen\n"
        << "  corpus is preserved unrewritten as historical evidence, and the\n"
        << "  arithmetic survives here ONLY as Pwo4LegacyReportedVp so this\n"
        << "  audit can still reproduce it.\n"
        << "  Assertion 5b (recomputed legacy VP == recorded) passed on\n"
        << "  every seat of every game, so the REPLAY is exact; what differs\n"
        << "  is which VP function the recorded field represents.\n"
        << "  divergent (game, seat) pairs : " << vp_divergences << " of "
        << (games_checked * 4) << "\n"
        << "  of which the SEARCHED seat   : " << vp_divergences_acting_seat
        << " of " << games_checked << "  <-- these feed final_vp_head\n"
        << "  examples:\n";
    for (const std::string& ex : vp_divergence_examples) {
      std::cout << "    " << ex << "\n";
    }
    std::cout
        << "  final_vp_head targets use FinalScoredVp (registration 5.4's\n"
        << "  literal wording, and the engine's real score).\n"
        << "  Registration 5.4 quotes the acting seat's range from the\n"
        << "  RECORDED field. Both are recomputed here:\n"
        << "    recorded final_vp[searched_seat] : min "
        << acting_vp_recorded_min << " max " << acting_vp_recorded_max
        << " mean "
        << (static_cast<double>(acting_vp_recorded_sum) / games_checked)
        << "  (registration 5.4 states min 5 / max 16 / mean 10.13)\n"
        << "    FinalScoredVp(searched_seat)     : min " << acting_vp_min
        << " max " << acting_vp_max << " mean "
        << (static_cast<double>(acting_vp_sum) / games_checked) << "  -> /20 = ["
        << (acting_vp_min / 20.0) << ", " << (acting_vp_max / 20.0)
        << "], mean " << (static_cast<double>(acting_vp_sum) / games_checked / 20.0)
        << "\n\n";
  }

  // -------------------------------------------------------------------------
  // Amendment 1 ruling 3: the divergence is a PINNED CORPUS-AUDIT FACT.
  // -------------------------------------------------------------------------
  // Assertion 5c no longer asserts equality -- the amendment removed that
  // requirement, because the two functions are not the same function. What it
  // asserts instead is that the corpus's divergence is EXACTLY the one that was
  // audited and ruled on: 212 of 1,600 (game, seat) pairs and 59 of the 400
  // searched seats. A different count means the corpus, the engine, or the
  // legacy reproduction moved, and that is a STOP -- so the check has teeth in
  // both directions rather than being a report nobody can fail.
  Require(vp_divergences == kExpectedVpDivergences,
          "GATE assertion 5c FAILED: divergent (game, seat) pairs = " +
              std::to_string(vp_divergences) + ", pinned by amendment 1 at " +
              std::to_string(kExpectedVpDivergences) +
              " -- the corpus, the engine, or Pwo4LegacyReportedVp has moved");
  Require(vp_divergences_acting_seat == kExpectedVpDivergencesActingSeat,
          "GATE assertion 5c FAILED: divergent searched seats = " +
              std::to_string(vp_divergences_acting_seat) +
              ", pinned by amendment 1 at " +
              std::to_string(kExpectedVpDivergencesActingSeat) +
              " -- the corpus, the engine, or Pwo4LegacyReportedVp has moved");
  std::cout << "  assertion 5c (pinned divergence) : " << vp_divergences
            << " pairs / " << vp_divergences_acting_seat
            << " searched seats -- MATCHES the amendment-1 pins\n\n";

  // -------------------------------------------------------------------------
  // Amendment 1 ruling 2: the COMPLETE acting-seat histograms.
  // -------------------------------------------------------------------------
  auto print_hist = [&](const char* label, const std::array<int64_t, 41>& h) {
    std::cout << "  " << label << ":\n    ";
    int64_t total = 0;
    for (int v = 0; v < static_cast<int>(h.size()); ++v) {
      if (h[v] == 0) continue;
      std::cout << v << "->" << h[v] << "  ";
      total += h[v];
    }
    std::cout << "\n    total " << total << "\n";
    Require(total == games_checked,
            std::string(label) + " histogram total != games replayed");
  };
  std::cout << "--- acting-seat VP histograms (amendment 1 ruling 2) ---\n";
  print_hist("FinalScoredVp(searched_seat)   [the registered target source]",
             acting_vp_hist);
  print_hist("recorded final_vp[searched_seat] [the legacy reported field]",
             acting_vp_recorded_hist);
  // The /20 divisor is registered on the premise that no game reaches 20 VP.
  // With the full histogram in hand that premise is checkable, so it is checked.
  Require(acting_vp_max < 20,
          "GATE FAILED: acting-seat FinalScoredVp reached " +
              std::to_string(acting_vp_max) +
              " -- the /20 divisor saturates and registration 5.4's "
              "'never saturates' claim is false");
  std::cout << "\n";

  // -------------------------------------------------------------------------
  // Realized split sizes vs the registration. WARNING, never a hard failure:
  // the registration derived these numbers, so a disagreement is information
  // about the registration, not necessarily about the data.
  // -------------------------------------------------------------------------
  auto compare = [](const char* name, int64_t got, int64_t reg) {
    if (got != reg) {
      std::cout << "*** WARNING: " << name << " = " << got
                << " but the registration derived " << reg << " ***\n";
    } else {
      std::cout << "  " << name << " = " << got << "  (matches registration)\n";
    }
  };
  std::cout << "--- realized split sizes vs registration 8.5 ---\n";
  compare("label rows held out      ", rows_heldout, 7311);
  compare("weight-one rows held out ", w1_heldout, 3541);
  compare("training label rows      ", rows_training, 41132);
  compare("training weight-one rows ", w1_training, 20582);
  std::cout << "  (training games = " << (games.size() - heldout.size())
            << ", total weight-one = " << w1_total << ")\n\n";

  // aux_targets.bin size arithmetic: 32-byte header + fixed-width records.
  {
    const uint64_t expect_bytes =
        32ull + static_cast<uint64_t>(rows_written) *
                    (28ull + 4ull * static_cast<uint64_t>(obs_size));
    Require(aux->bytes() == expect_bytes,
            "aux_targets.bin is " + std::to_string(aux->bytes()) +
                " bytes, expected " + std::to_string(expect_bytes));
  }

  // Read BOTH packs back with the reader's own walk (VerifyPackAgainstReader).
  {
    int64_t vrows = 0, vval = 0;
    std::cout << "verifying the train pack against the reader's own record walk...\n"
              << std::flush;
    VerifyPackAgainstReader(pack_train_path, obs_size, action_dim, &vrows, &vval);
    Require(vrows == pack_rows,
            "train pack read-back found " + std::to_string(vrows) +
                " records but " + std::to_string(pack_rows) + " were written");
    Require(vval == pack_loader_validation,
            "train pack read-back %11 count " + std::to_string(vval) +
                " != the writer's " + std::to_string(pack_loader_validation));
    std::cout << "  read-back OK: " << vrows << " records\n" << std::flush;

    int64_t vrows2 = 0, vval2 = 0;
    std::cout << "verifying the validation pack against the reader's own walk...\n"
              << std::flush;
    VerifyPackAgainstReader(pack_val_path, obs_size, action_dim, &vrows2, &vval2);
    Require(vrows2 == pack_val_rows,
            "validation pack read-back found " + std::to_string(vrows2) +
                " records but " + std::to_string(pack_val_rows) + " were written");
    std::cout << "  read-back OK: " << vrows2 << " records\n\n";
  }

  // -------------------------------------------------------------------------
  // Ruling 5: the roles, the restored exposure, and the %11 rule the train
  // pack now BYPASSES.
  // -------------------------------------------------------------------------
  const int64_t manifest_training = pack_rows;
  const int64_t manifest_validation = pack_val_rows;

  // The registered realized counts. A different number is a STOP, not a note:
  // ruling 5 registers 20,582 / 3,541 and section 6.3's 14.93 expected passes
  // is arithmetic over the first of them.
  Require(manifest_training == 20582,
          "train-role pack holds " + std::to_string(manifest_training) +
              " rows; amendment 1 ruling 5 registers 20582");
  Require(manifest_validation == 3541,
          "validation-role pack holds " + std::to_string(manifest_validation) +
              " rows; amendment 1 ruling 5 registers 3541");

  const double passes = 300.0 * 1024.0 / static_cast<double>(manifest_training);
  std::cout << "--- role-partitioned distillation packs (ruling 5) ---\n"
            << "  " << pack_train_name << "  role=train       rows = "
            << manifest_training << "  (the 340 NON-held-out games)\n"
            << "  " << pack_val_name << "  role=validation  rows = "
            << manifest_validation << "  (the 60 HELD-OUT games)\n"
            << "  Because the train pack declares a role, the loader does NOT\n"
            << "  consult IsValidationLabel for it. The %11 rule WOULD have\n"
            << "  diverted " << pack_loader_validation << " of those "
            << manifest_training << " rows ("
            << (manifest_training > 0
                    ? 100.0 * pack_loader_validation / manifest_training : 0.0)
            << "%) into the loader's own\n"
            << "  validation bucket, cutting the registered exposure to "
            << (manifest_training - pack_loader_validation) << " rows.\n"
            << "  Restored: 300 updates x 1024 presentations / "
            << manifest_training << " rows = " << passes
            << " expected passes\n"
            << "  (registration section 6.2 registers 14.93).\n"
            << "  NO HELD-OUT ROW IS IN THE TRAIN PACK, and the validation pack\n"
            << "  is loaded validation-only, so no held-out row can receive a\n"
            << "  training gradient. The held-out distillation KL it buys is\n"
            << "  DESCRIPTIVE ONLY -- no gate, threshold or nomination\n"
            << "  criterion reads it (ruling 5, section 6.4).\n\n";

  if (manifest_training < 8192 || manifest_validation < 1024) {
    Stop("legacy-loader floors violated: training_label_count=" +
         std::to_string(manifest_training) + " (needs >= 8192), "
         "validation_label_count=" + std::to_string(manifest_validation) +
         " (needs >= 1024). The loader would fatally reject this pack.");
  }
  std::cout << "  loader floors: training " << manifest_training
            << " >= 8192 OK; validation " << manifest_validation
            << " >= 1024 OK\n\n";

  // -------------------------------------------------------------------------
  // Hashes and manifests.
  // -------------------------------------------------------------------------
  std::cout << "hashing artifacts (this reads ~"
            << ((aux->bytes() + pack->bytes() + pack_val->bytes()) >> 20)
            << " MiB)...\n" << std::flush;
  const std::string aux_sha = ComputeFileSHA256(aux_path);
  const std::string pack_sha = ComputeFileSHA256(pack_train_path);
  const std::string pack_val_sha = ComputeFileSHA256(pack_val_path);

  // ----- the legacy pack manifest -----
  // Built as json::Object so the semantic sub-object below is byte-identical to
  // what SearchLabelBuffer::LoadFromDirectory reconstructs after parsing this
  // file. json::Object is a std::map, so ToString emits keys in lexicographic
  // order regardless of insertion order; and every value here is an int or a
  // string, so nothing goes through json.cc's six-decimal double formatter.
  {
    json::Object semantic;
    semantic["schema_version"] = json::Value(static_cast<int64_t>(2));
    semantic["base_seed"] = json::Value(static_cast<int64_t>(20260729));
    semantic["model_checkpoint_sha256"] = json::Value(
        std::string("4bffd2b7d73fc684344947d2064560d5c367c63e28a124fc87010de04e021f79"));
    semantic["effective_search_config"] = json::Value(std::string(
        "PWO-4 trajectory teacher, frozen: max_simulations=200 puct_c=0.3 "
        "utility_divisor=4 temperature=0 use_session=false "
        "check_strategic_state=true final_policy_type=kNormalizedVisitCount "
        "covered_prior_threshold=0.5 min_visit_threshold=2 "
        "stability_checkpoint_sim=100 logit_cap=10 root_prior_temperature=1 "
        "engine_tree=a67f1925305d35d9a2f30fd9658b9f01fe21bdd6"));
    semantic["architecture"] = json::Value(std::string(
        "SharedDunePolicyValueNet hidden_dim=2048 num_blocks=8 "
        "nonlinear_value_head=false"));
    semantic["training_label_count"] = json::Value(manifest_training);
    semantic["validation_label_count"] = json::Value(manifest_validation);
    // Ruling 5: a ROLE-AWARE manifest binds the (filename, sha256, role)
    // mapping into the fingerprint. Without it the roles would be the one part
    // of the contract nothing authenticates, and an edited manifest could move
    // the held-out pack into the training bucket with every hash still
    // verifying. The string is order-sensitive and MUST match
    // SearchLabelBuffer::CanonicalFileRoles byte for byte -- the loader
    // recomputes exactly this and compares.
    const std::string file_roles =
        pack_train_name + ":" + pack_sha + ":train;" +
        pack_val_name + ":" + pack_val_sha + ":validation;";
    semantic["file_roles"] = json::Value(file_roles);

    const std::string semantic_json = json::ToString(semantic);
    const std::string fingerprint = ComputeStringSHA256(semantic_json);

    json::Object manifest = semantic;
    json::Array files;
    {
      json::Object e;
      e["filename"] = json::Value(pack_train_name);
      e["sha256"] = json::Value(pack_sha);
      e["role"] = json::Value(std::string("train"));
      files.push_back(json::Value(e));
    }
    {
      json::Object e;
      e["filename"] = json::Value(pack_val_name);
      e["sha256"] = json::Value(pack_val_sha);
      e["role"] = json::Value(std::string("validation"));
      files.push_back(json::Value(e));
    }
    manifest["files"] = json::Value(files);
    manifest["search_label_fingerprint"] = json::Value(fingerprint);
    // Non-semantic provenance. The loader ignores unknown keys, and they are
    // NOT part of the fingerprint's semantic object, so adding them cannot
    // change the fingerprint.
    manifest["source"] = json::Value(std::string(
        "calibration_results_v2/pwo4_trajectory (PWO-4 CP4 accepted corpus)"));
    manifest["pack_population"] = json::Value(std::string(
        "role=train: the weight-one rows of the 340 NON-held-out games "
        "(gradient-eligible). role=validation: the weight-one rows of the 60 "
        "PWO-5 section 8.5 HELD-OUT games (held out from ALL gradients; the "
        "held-out distillation KL they yield is DESCRIPTIVE ONLY)."));
    manifest["rows_written_train"] = json::Value(pack_rows);
    manifest["rows_written_validation"] = json::Value(pack_val_rows);
    manifest["legacy_pct11_would_have_diverted"] =
        json::Value(pack_loader_validation);
    manifest["heldout_membership_sha256"] = json::Value(membership_sha256);
    manifest["produced_by"] = json::Value(std::string("dune_pwo5_prepare"));

    const std::string p = (pack_dir / "manifest.json").string();
    std::ofstream o(p);
    if (!o) Stop("cannot write " + p);
    o << json::ToString(manifest, /*wrap=*/true) << "\n";
    o.close();
    std::cout << "pack manifest fingerprint: " << fingerprint << "\n";
  }

  // ----- the auxiliary-target manifest -----
  {
    std::ostringstream j;
    j << "{\n"
      << "  \"schema\": 1,\n"
      << "  \"magic\": \"0x50573541\",\n"
      << "  \"obs_size\": " << obs_size << ",\n"
      << "  \"action_dim\": " << action_dim << ",\n"
      << "  \"num_rows\": " << rows_written << ",\n"
      << "  \"file\": \"aux_targets.bin\",\n"
      << "  \"file_bytes\": " << aux->bytes() << ",\n"
      << "  \"aux_targets_sha256\": \"" << aux_sha << "\",\n"
      << "  \"heldout_membership_sha256\": \"" << membership_sha256 << "\",\n"
      << "  \"record_order\": \"game_index ascending, then decision_index ascending\",\n"
      << "  \"counts\": {\n"
      << "    \"rows\": " << rows_written << ",\n"
      << "    \"heldout_rows\": " << rows_heldout << ",\n"
      << "    \"training_rows\": " << rows_training << ",\n"
      << "    \"weight_one_rows\": " << w1_total << ",\n"
      << "    \"weight_one_heldout_rows\": " << w1_heldout << ",\n"
      << "    \"weight_one_training_rows\": " << w1_training << ",\n"
      << "    \"rows_without_next_own_action\": " << rows_no_next << ",\n"
      << "    \"games\": " << games_checked << ",\n"
      << "    \"heldout_games\": " << heldout.size() << "\n"
      << "  },\n"
      << "  \"final_vp_target_source\": \"DuneImperiumState::FinalScoredVp("
         "acting_player) / 20\",\n"
      << "  \"gate_assertion_5c\": {\n"
      << "    \"rule\": \"amendment 1 ruling 3c: FinalScoredVp is recomputed "
         "INDEPENDENTLY for the head target and is NOT required to equal the "
         "recorded final_vp field. The divergence below is a PINNED "
         "corpus-audit fact; a different count is a STOP.\",\n"
      << "    \"pinned_divergent_game_seat_pairs\": " << kExpectedVpDivergences
      << ",\n"
      << "    \"pinned_divergent_searched_seat_games\": "
      << kExpectedVpDivergencesActingSeat << ",\n"
      << "    \"measured_divergent_game_seat_pairs\": " << vp_divergences << ",\n"
      << "    \"measured_divergent_searched_seat_games\": "
      << vp_divergences_acting_seat << ",\n"
      << "    \"pins_match\": "
      << ((vp_divergences == kExpectedVpDivergences &&
           vp_divergences_acting_seat == kExpectedVpDivergencesActingSeat)
              ? "true"
              : "false")
      << ",\n"
      << "    \"divergence_magnitude\": \"the legacy field overstates by "
         "exactly 1, asserted per divergent seat\",\n"
      << "    \"cause\": \"games.jsonl final_vp was emitted by the "
         "since-removed GetTrueFinalVp reporting helper, which awarded the "
         "all-factions>=3 +1 unconditionally; ComputeEndgameVp awards it only "
         "with tech tile 8 (Memocorders). The helper is now fixed at source in "
         "dune_pwo4_trajectory.cc and dune_search_benchmark.cc, both of which "
         "report through DuneImperiumState::FinalScoredVp; the frozen corpus is "
         "preserved unrewritten as historical evidence and the legacy "
         "arithmetic survives only as dune_pwo5_prepare.cc's "
         "Pwo4LegacyReportedVp for this audit. Replay fidelity itself is "
         "unaffected: assertion 5b passed on every seat of every game.\"\n"
      << "  },\n"
      << "  \"acting_seat_final_scored_vp\": {\"min\": " << acting_vp_min
      << ", \"max\": " << acting_vp_max << ", \"mean\": "
      << std::setprecision(17) << (static_cast<double>(acting_vp_sum) / games_checked)
      << ", \"histogram\": {" << HistogramJson(acting_vp_hist) << "}},\n"
      << "  \"acting_seat_recorded_final_vp\": {\"min\": " << acting_vp_recorded_min
      << ", \"max\": " << acting_vp_recorded_max << ", \"mean\": "
      << std::setprecision(17)
      << (static_cast<double>(acting_vp_recorded_sum) / games_checked)
      << ", \"histogram\": {" << HistogramJson(acting_vp_recorded_hist) << "}},\n"
      << "  \"terminal_round_class_rows\": {\"le8\": " << rows_by_class[0]
      << ", \"r9\": " << rows_by_class[1] << ", \"r10\": " << rows_by_class[2] << "},\n"
      << "  \"terminal_round_class_games\": {\"le8\": " << games_by_class[0]
      << ", \"r9\": " << games_by_class[1] << ", \"r10\": " << games_by_class[2] << "},\n"
      << "  \"pack\": {\n"
      << "    \"dir\": \"search_labels\",\n"
      << "    \"role_aware\": true,\n"
      << "    \"train\": {\n"
      << "      \"file\": \"" << pack_train_name << "\",\n"
      << "      \"role\": \"train\",\n"
      << "      \"sha256\": \"" << pack_sha << "\",\n"
      << "      \"file_bytes\": " << pack->bytes() << ",\n"
      << "      \"rows_written\": " << pack_rows << "\n"
      << "    },\n"
      << "    \"validation\": {\n"
      << "      \"file\": \"" << pack_val_name << "\",\n"
      << "      \"role\": \"validation\",\n"
      << "      \"sha256\": \"" << pack_val_sha << "\",\n"
      << "      \"file_bytes\": " << pack_val->bytes() << ",\n"
      << "      \"rows_written\": " << pack_val_rows << "\n"
      << "    },\n"
      << "    \"loader_training_label_count\": " << manifest_training << ",\n"
      << "    \"loader_validation_label_count\": " << manifest_validation << ",\n"
      << "    \"legacy_pct11_would_have_diverted\": " << pack_loader_validation
      << "\n"
      << "  }\n"
      << "}\n";
    const std::string p =
        (std::filesystem::path(out_dir) / "aux_targets_manifest.json").string();
    std::ofstream o(p);
    if (!o) Stop("cannot write " + p);
    o << j.str();
    o.close();
  }

  // -------------------------------------------------------------------------
  // Summary.
  // -------------------------------------------------------------------------
  std::cout << "\n--- artifacts ---\n"
            << "  aux_targets.bin            " << aux->bytes() << " bytes\n"
            << "    sha256                   " << aux_sha << "\n"
            << "  search_labels/" << pack_train_name << "  role=train       "
            << pack->bytes() << " bytes, " << pack_rows << " rows\n"
            << "    sha256                   " << pack_sha << "\n"
            << "  search_labels/" << pack_val_name << "  role=validation  "
            << pack_val->bytes() << " bytes, " << pack_val_rows << " rows\n"
            << "    sha256                   " << pack_val_sha << "\n"
            << "  heldout membership sha256  " << membership_sha256 << "\n\n"
            << "--- auxiliary-target counts ---\n"
            << "  rows                       " << rows_written << "\n"
            << "  rows with next_own_action = -1 (expect 400): " << rows_no_next
            << (rows_no_next == games_checked ? "  (== one per game)" : "  *** NOT one per game ***")
            << "\n"
            << "  terminal_round_class rows  le8=" << rows_by_class[0]
            << " r9=" << rows_by_class[1] << " r10=" << rows_by_class[2] << "\n"
            << "  terminal_round_class games le8=" << games_by_class[0]
            << " r9=" << games_by_class[1] << " r10=" << games_by_class[2] << "\n";

  Require(rows_no_next == games_checked,
          "rows with next_own_action == -1 (" + std::to_string(rows_no_next) +
              ") is not one per game (" + std::to_string(games_checked) + ")");

  // -------------------------------------------------------------------------
  // THE SUCCESS BANNERS, AND THEY ARE LAST ON PURPOSE.
  // -------------------------------------------------------------------------
  // Nothing below this line may fail, because nothing below this line runs a
  // check. Every assertion in this tool executes ABOVE here.
  //
  // This ordering is the whole point and it was WRONG until an adversarial
  // review demonstrated it by execution: the banners used to print before the
  // 5c pins, the histogram totals, the /20 saturation check, the pack read-back
  // and the loader floors, so the tool would emit
  //     GATE RESULT: PASS (assertions 1, 2, 3, 4, 5a, 5b, 5c)
  //     ALL GATES PASSED.
  // on stdout and then STOP on stderr with exit 2. A caller reading the exit
  // code was safe; a human, or anything grepping stdout, was not. That is the
  // same "reports success while a check fails" shape amendment 1 section 4.4
  // removed the --strict_final_scored_vp flag to eliminate, reintroduced by
  // print ORDER rather than by an optional assertion.
  //
  // If you add a check to this tool, add it ABOVE this block.
  std::cout << "\nGATE RESULT: PASS (assertions 1, 2, 3, 4, 5a, 5b, 5c)\n"
            << "  games replayed        : " << games_checked << "\n"
            << "  label rows checked    : " << rows_written << "\n"
            << "  assertion 1 (legal set, order+content) : no mismatch on any row\n"
            << "  assertion 2 (|legal| == n_legal)       : no mismatch on any row\n"
            << "  assertion 3 (CurrentPlayer == acting)  : no mismatch on any row\n"
            << "  assertion 4 (position == history_len)  : no mismatch on any row\n"
            << "  assertion 5a (terminal round)          : no mismatch on any game\n"
            << "  assertion 5b (recomputed PWO-4 legacy reported VP\n"
            << "                == games.jsonl final_vp)  : no mismatch on any seat\n"
            << "  assertion 5c (pinned divergence)       : " << vp_divergences
            << " pairs / " << vp_divergences_acting_seat << " searched seats\n"
            << "  /20 saturation (max < 20)              : max " << acting_vp_max
            << "\n\n"
            << "ALL GATES PASSED.\n";
  return 0;
}
