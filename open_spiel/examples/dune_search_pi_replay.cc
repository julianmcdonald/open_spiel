#include "dune_search_pi_replay.h"

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <ios>
#include <limits>
#include <numeric>
#include <random>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#include "open_spiel/abseil-cpp/absl/strings/str_cat.h"
#include "open_spiel/abseil-cpp/absl/strings/str_format.h"
#include "open_spiel/spiel_utils.h"
#include "dune_seed_utils.h"

namespace open_spiel {
namespace {

// --- Representation assumptions, made explicit ------------------------------
//
// The format writes raw object bytes. That is what makes the float round-trip
// exact, and it is also what makes the format architecture-bound: a shard is
// portable only between builds that agree on these. They are asserted rather
// than assumed so a hypothetical big-endian or non-IEEE target fails to compile
// instead of silently writing shards nothing can read back.
static_assert(sizeof(float) == 4, "shard format writes 4-byte binary32 floats");
static_assert(sizeof(double) == 8,
              "shard format writes 8-byte binary64 doubles");
static_assert(sizeof(int) == 4, "shard format writes 4-byte ints");
static_assert(std::numeric_limits<float>::is_iec559,
              "shard format assumes IEEE-754 float");
static_assert(std::numeric_limits<double>::is_iec559,
              "shard format assumes IEEE-754 double");
static_assert(sizeof(Action) == 8, "shard format writes 8-byte Action");
static_assert(sizeof(Player) == 4, "shard format writes 4-byte Player");

// --- Deterministic stream derivation ---------------------------------------
//
// Continues this lane's OWN 0x5000 stream block (dune_search_pi.cc:35-39 holds
// 0x5001..0x5005). A distinct constant here is what stops a replay sample from
// correlating with the chance, opponent-policy, search-seed, behaviour or
// unsearched-seat streams a generation draws from the same base seed. The
// shared kStream* constants in dune_seed_utils.h are deliberately NOT reused:
// that block is registered, and the note on kStreamAuxSampling records why
// sharing a stream between two samplers drawing different populations is
// forbidden. This sampler draws yet a third population (rows across cohorts).
constexpr uint64_t kStreamPiReplaySample = 0x5006;
constexpr uint64_t kStreamPiScratchReplaySample = 0x5008;

// --- Row size floor ---------------------------------------------------------
//
// The fixed (payload-independent) cost of one serialized row: 8 vector length
// prefixes at 8 bytes, 3 string length prefixes at 4 bytes, 6 bool bytes,
// 10 int32 fields, 2 int64 fields and 8 double fields. Used ONLY as a lower
// bound, to reject a header row_count that could not possibly fit in the file
// before any per-row allocation happens. Because it is only ever a lower bound,
// a value left stale by ADDING a field stays sound; removing a field is the
// case that would require updating it, and that is a format version bump.
constexpr uint64_t kMinSerializedRowBytes =
    8 * 8 + 3 * 4 + 6 * 1 + 10 * 4 + 2 * 8 + 8 * 8;

std::string ErrnoText(const char* what, const std::string& path) {
  // The errno text goes LAST inside the sentence and the sentence ends in a
  // period. Never end a diagnostic literal with a bare identifier: this
  // codebase's linker tail-merges string literals, and a literal ending in a
  // pinned flag name gets absorbed into that flag's own literal.
  return absl::StrCat(what, " '", path, "' failed: ", std::strerror(errno),
                      ".");
}

void SetError(std::string* error, std::string text) {
  if (error != nullptr) *error = std::move(text);
}

// ===========================================================================
// Writer
// ===========================================================================
//
// Buffers into memory and flushes in fixed chunks straight to a file
// descriptor. The descriptor (rather than an std::ofstream) is the point: the
// atomicity contract needs an fsync on the DATA before the rename, and an
// ofstream gives no portable handle to fsync. Buffering in chunks rather than
// building the whole shard in memory keeps peak memory flat -- a cohort's
// observations alone run to order 10^8 bytes, and doubling that just to write
// it out would be a real cost.
class ShardWriter {
 public:
  explicit ShardWriter(int fd, std::string path) : fd_(fd), path_(std::move(path)) {
    buf_.reserve(kChunkBytes + 1024);
  }

  void PutBytes(const void* src, size_t n) {
    if (failed_) return;
    buf_.append(static_cast<const char*>(src), n);
    if (buf_.size() >= kChunkBytes) FlushBuffer();
  }

  void PutU8(uint8_t v) { PutBytes(&v, sizeof(v)); }
  void PutBool(bool v) { PutU8(v ? 1 : 0); }
  void PutU32(uint32_t v) { PutBytes(&v, sizeof(v)); }
  void PutU64(uint64_t v) { PutBytes(&v, sizeof(v)); }
  void PutI32(int32_t v) { PutBytes(&v, sizeof(v)); }
  void PutI64(int64_t v) { PutBytes(&v, sizeof(v)); }
  void PutF64(double v) { PutBytes(&v, sizeof(v)); }

  // Enums go out as their int32 ordinal. The stable NAMES exist
  // (SearchPiFallbackName / SearchPiEarlyExitName) and are what telemetry
  // persists, but a shard is an internal store read back by this exact build,
  // so the ordinal is both smaller and exact. The header's version field is
  // what protects against a future reordering of either enum.
  template <typename Enum>
  void PutEnum(Enum v) {
    PutI32(static_cast<int32_t>(v));
  }

  // Length-prefixed raw payload. sizeof(T) bytes per element, memcpy semantics,
  // no conversion -- this is the float exactness guarantee.
  template <typename T>
  void PutPodVector(const std::vector<T>& v) {
    PutU64(static_cast<uint64_t>(v.size()));
    if (!v.empty()) PutBytes(v.data(), v.size() * sizeof(T));
  }

  // uint32 length prefix: a fingerprint is 64 hex characters and the two status
  // strings are short words, so 4 bytes is ample and 8 would be waste per row.
  void PutString(const std::string& s) {
    PutU32(static_cast<uint32_t>(s.size()));
    if (!s.empty()) PutBytes(s.data(), s.size());
  }

  bool Flush() {
    if (!failed_ && !buf_.empty()) FlushBuffer();
    return !failed_;
  }

  const std::string& error() const { return error_; }

 private:
  static constexpr size_t kChunkBytes = 1u << 20;  // 1 MiB

  void FlushBuffer() {
    const char* p = buf_.data();
    size_t left = buf_.size();
    while (left > 0) {
      // A short write is legal and not an error; EINTR is a retry, not a
      // failure. Treating either as fatal would truncate shards under signal
      // load rather than under any real fault.
      const ssize_t n = ::write(fd_, p, left);
      if (n < 0) {
        if (errno == EINTR) continue;
        failed_ = true;
        error_ = ErrnoText("write to", path_);
        buf_.clear();
        return;
      }
      p += n;
      left -= static_cast<size_t>(n);
    }
    buf_.clear();
  }

  int fd_;
  std::string path_;
  std::string buf_;
  std::string error_;
  bool failed_ = false;
};

void SerializeRow(ShardWriter* w, const SearchPiRow& row) {
  // EVERY field of SearchPiRow, in declaration order. The learner reads only
  // observation / legal_actions / target_probs / value_target /
  // value_target_attached / episode_id, but a shard is the ONLY surviving copy
  // of a cohort once its process exits, and the fields it does not read are the
  // ones every audit, hash chain and post-hoc diagnosis joins on. Dropping them
  // to save roughly 5% of a row -- the observation is 22 KB of the ~23 KB -- would
  // trade an unrecoverable loss for nothing.
  w->PutPodVector(row.observation);
  w->PutBool(row.observation_is_information_state);
  w->PutI32(static_cast<int32_t>(row.player));
  w->PutEnum(row.role);

  w->PutPodVector(row.legal_actions);
  w->PutPodVector(row.raw_policy);
  w->PutPodVector(row.raw_visits);
  w->PutPodVector(row.target_visits);
  w->PutPodVector(row.target_probs);
  w->PutI64(static_cast<int64_t>(row.chosen_action));

  w->PutF64(row.value_target);
  w->PutBool(row.value_target_attached);

  w->PutI32(row.generation);
  w->PutI64(row.episode_id);
  w->PutI32(row.decision_id);
  w->PutString(row.state_fingerprint);

  w->PutI32(row.simulations_completed);
  w->PutI32(row.simulations_requested);
  w->PutEnum(row.early_exit);
  w->PutI32(row.inherited_root_visits);
  w->PutPodVector(row.visits_pre_search);
  w->PutBool(row.retention_snapshots_valid);
  w->PutString(row.re_root_status);
  w->PutEnum(row.fallback);
  w->PutString(row.search_fallback_reason);
  w->PutF64(row.root_value);
  w->PutPodVector(row.q_values);

  // The distribution scalars are carried, never recomputed on read. They were
  // computed once at full double precision before anything was serialized
  // precisely so a reader would not have to reconstruct them; recomputing
  // KL(target || raw) from a stored row is the operation the header of
  // dune_search_pi.h warns manufactures an infinity.
  w->PutF64(row.raw_policy_entropy_norm);
  w->PutF64(row.target_entropy_norm);
  w->PutF64(row.raw_policy_max_prob);
  w->PutF64(row.target_max_prob);
  w->PutF64(row.kl_target_given_raw);
  w->PutBool(row.kl_raw_prior_floored);
  w->PutBool(row.target_argmax_differs_from_raw);

  w->PutI32(row.num_covered_actions);
  w->PutF64(row.covered_prior_mass);
  w->PutBool(row.would_pass_legacy_coverage_gate);
}

void SerializeScratchRowV2(ShardWriter* w, const SearchPiRow& row) {
  SerializeRow(w, row);
  w->PutI32(row.row_schema_version);
  w->PutString(row.target_type);
  w->PutString(row.search_budget_class);
  w->PutU64(row.search_budget_draw);
  w->PutF64(row.policy_target_weight);
  w->PutBool(row.regularized_q_valid);
  w->PutString(row.regularized_q_error);
  w->PutPodVector(row.regularized_q_target);
  w->PutF64(row.regularized_q_beta);
  w->PutF64(row.regularized_q_kl);
  w->PutF64(row.regularized_q_prior_expected_q);
  w->PutF64(row.regularized_q_target_expected_q);
  w->PutF64(row.regularized_q_q_range);
  w->PutI32(row.regularized_q_direct_visit_count);
  w->PutF64(row.regularized_q_entropy_norm);
}

// ===========================================================================
// Reader
// ===========================================================================
//
// Every read is bounded by the bytes REMAINING in the file, which is read once
// up front. That single invariant is what makes a corrupted shard a clean
// `false` instead of a crash: a length prefix damaged to 2^63 is rejected by
// comparison before any resize() is attempted, so there is no huge-allocation
// path to reach.
class ShardReader {
 public:
  ShardReader(std::ifstream* in, uint64_t size, std::string path)
      : in_(in), remaining_(size), path_(std::move(path)) {}

  bool Bytes(void* dst, uint64_t n, const char* field) {
    if (failed_) return false;
    if (n > remaining_) {
      return Fail(absl::StrCat("field '", field, "' needs ", n,
                               " bytes but only ", remaining_,
                               " remain in the shard."));
    }
    in_->read(static_cast<char*>(dst), static_cast<std::streamsize>(n));
    if (!in_->good()) {
      return Fail(absl::StrCat("read of field '", field, "' short by ",
                               n - static_cast<uint64_t>(in_->gcount()),
                               " bytes."));
    }
    remaining_ -= n;
    return true;
  }

  bool U8(uint8_t* v, const char* f) { return Bytes(v, 1, f); }
  bool U32(uint32_t* v, const char* f) { return Bytes(v, 4, f); }
  bool U64(uint64_t* v, const char* f) { return Bytes(v, 8, f); }
  bool I32(int32_t* v, const char* f) { return Bytes(v, 4, f); }
  bool I64(int64_t* v, const char* f) { return Bytes(v, 8, f); }
  bool F64(double* v, const char* f) { return Bytes(v, 8, f); }

  bool Bool(bool* v, const char* f) {
    uint8_t byte = 0;
    if (!U8(&byte, f)) return false;
    // Anything other than the two values the writer emits means the byte stream
    // has drifted out of alignment, and a drifted stream that keeps parsing is
    // exactly how a corrupt shard turns into plausible-looking garbage rows.
    if (byte > 1) {
      return Fail(absl::StrCat("boolean field '", f, "' holds ",
                               static_cast<int>(byte),
                               ", which is not 0 or 1."));
    }
    *v = (byte != 0);
    return true;
  }

  template <typename Enum>
  bool Enum32(Enum* v, int32_t count, const char* f) {
    int32_t raw = 0;
    if (!I32(&raw, f)) return false;
    if (raw < 0 || raw >= count) {
      return Fail(absl::StrCat("enum field '", f, "' holds ordinal ", raw,
                               ", outside [0, ", count, ")."));
    }
    *v = static_cast<Enum>(raw);
    return true;
  }

  template <typename T>
  bool PodVector(std::vector<T>* out, const char* f) {
    uint64_t count = 0;
    if (!U64(&count, f)) return false;
    // Division, not multiplication: `count * sizeof(T)` on a corrupted count
    // overflows and can compare small, which would defeat the bound it is
    // supposed to enforce.
    if (count > remaining_ / sizeof(T)) {
      return Fail(absl::StrCat("vector field '", f, "' declares ", count,
                               " elements, more than the ", remaining_,
                               " bytes left in the shard allow."));
    }
    out->assign(count, T{});
    if (count == 0) return true;
    return Bytes(out->data(), count * sizeof(T), f);
  }

  bool Str(std::string* out, const char* f) {
    uint32_t len = 0;
    if (!U32(&len, f)) return false;
    if (static_cast<uint64_t>(len) > remaining_) {
      return Fail(absl::StrCat("string field '", f, "' declares ", len,
                               " bytes, more than the ", remaining_,
                               " left in the shard."));
    }
    out->assign(len, '\0');
    if (len == 0) return true;
    return Bytes(&(*out)[0], len, f);
  }

  bool Fail(std::string text) {
    if (!failed_) {
      failed_ = true;
      error_ = absl::StrCat("shard '", path_, "' is malformed: ", text);
    }
    return false;
  }

  uint64_t remaining() const { return remaining_; }
  bool failed() const { return failed_; }
  const std::string& error() const { return error_; }

 private:
  std::ifstream* in_;
  uint64_t remaining_;
  std::string path_;
  std::string error_;
  bool failed_ = false;
};

bool DeserializeRow(ShardReader* r, SearchPiRow* row) {
  int32_t i32 = 0;
  if (!r->PodVector(&row->observation, "observation")) return false;
  if (!r->Bool(&row->observation_is_information_state,
               "observation_is_information_state")) {
    return false;
  }
  if (!r->I32(&i32, "player")) return false;
  row->player = static_cast<Player>(i32);
  if (!r->Enum32(&row->role, 7, "role")) return false;

  if (!r->PodVector(&row->legal_actions, "legal_actions")) return false;
  if (!r->PodVector(&row->raw_policy, "raw_policy")) return false;
  if (!r->PodVector(&row->raw_visits, "raw_visits")) return false;
  if (!r->PodVector(&row->target_visits, "target_visits")) return false;
  if (!r->PodVector(&row->target_probs, "target_probs")) return false;
  int64_t i64 = 0;
  if (!r->I64(&i64, "chosen_action")) return false;
  row->chosen_action = static_cast<Action>(i64);

  if (!r->F64(&row->value_target, "value_target")) return false;
  if (!r->Bool(&row->value_target_attached, "value_target_attached")) {
    return false;
  }

  if (!r->I32(&row->generation, "generation")) return false;
  if (!r->I64(&row->episode_id, "episode_id")) return false;
  if (!r->I32(&row->decision_id, "decision_id")) return false;
  if (!r->Str(&row->state_fingerprint, "state_fingerprint")) return false;

  if (!r->I32(&row->simulations_completed, "simulations_completed")) {
    return false;
  }
  if (!r->I32(&row->simulations_requested, "simulations_requested")) {
    return false;
  }
  if (!r->Enum32(&row->early_exit, kSearchPiEarlyExitCount, "early_exit")) {
    return false;
  }
  if (!r->I32(&row->inherited_root_visits, "inherited_root_visits")) {
    return false;
  }
  if (!r->PodVector(&row->visits_pre_search, "visits_pre_search")) return false;
  if (!r->Bool(&row->retention_snapshots_valid, "retention_snapshots_valid")) {
    return false;
  }
  if (!r->Str(&row->re_root_status, "re_root_status")) return false;
  if (!r->Enum32(&row->fallback, 7, "fallback")) return false;
  if (!r->Str(&row->search_fallback_reason, "search_fallback_reason")) {
    return false;
  }
  if (!r->F64(&row->root_value, "root_value")) return false;
  if (!r->PodVector(&row->q_values, "q_values")) return false;

  if (!r->F64(&row->raw_policy_entropy_norm, "raw_policy_entropy_norm")) {
    return false;
  }
  if (!r->F64(&row->target_entropy_norm, "target_entropy_norm")) return false;
  if (!r->F64(&row->raw_policy_max_prob, "raw_policy_max_prob")) return false;
  if (!r->F64(&row->target_max_prob, "target_max_prob")) return false;
  if (!r->F64(&row->kl_target_given_raw, "kl_target_given_raw")) return false;
  if (!r->Bool(&row->kl_raw_prior_floored, "kl_raw_prior_floored")) {
    return false;
  }
  if (!r->Bool(&row->target_argmax_differs_from_raw,
               "target_argmax_differs_from_raw")) {
    return false;
  }

  if (!r->I32(&row->num_covered_actions, "num_covered_actions")) return false;
  if (!r->F64(&row->covered_prior_mass, "covered_prior_mass")) return false;
  if (!r->Bool(&row->would_pass_legacy_coverage_gate,
               "would_pass_legacy_coverage_gate")) {
    return false;
  }
  return true;
}

bool DeserializeScratchRowV2(ShardReader* r, SearchPiRow* row) {
  if (!DeserializeRow(r, row)) return false;
  if (!r->I32(&row->row_schema_version, "row_schema_version")) return false;
  if (!r->Str(&row->target_type, "target_type")) return false;
  if (!r->Str(&row->search_budget_class, "search_budget_class")) return false;
  if (!r->U64(&row->search_budget_draw, "search_budget_draw")) return false;
  if (!r->F64(&row->policy_target_weight, "policy_target_weight")) {
    return false;
  }
  if (!r->Bool(&row->regularized_q_valid, "regularized_q_valid")) {
    return false;
  }
  if (!r->Str(&row->regularized_q_error, "regularized_q_error")) return false;
  if (!r->PodVector(&row->regularized_q_target,
                    "regularized_q_target")) {
    return false;
  }
  if (!r->F64(&row->regularized_q_beta, "regularized_q_beta")) return false;
  if (!r->F64(&row->regularized_q_kl, "regularized_q_kl")) return false;
  if (!r->F64(&row->regularized_q_prior_expected_q,
              "regularized_q_prior_expected_q")) {
    return false;
  }
  if (!r->F64(&row->regularized_q_target_expected_q,
              "regularized_q_target_expected_q")) {
    return false;
  }
  if (!r->F64(&row->regularized_q_q_range,
              "regularized_q_q_range")) {
    return false;
  }
  if (!r->I32(&row->regularized_q_direct_visit_count,
              "regularized_q_direct_visit_count")) {
    return false;
  }
  return r->F64(&row->regularized_q_entropy_norm,
                "regularized_q_entropy_norm");
}

bool AllFinite(const std::vector<double>& values) {
  return std::all_of(values.begin(), values.end(),
                     [](double value) { return std::isfinite(value); });
}

bool AllFinite(const std::vector<float>& values) {
  return std::all_of(values.begin(), values.end(),
                     [](float value) { return std::isfinite(value); });
}

bool ValidateScratchRowV2(const SearchPiRow& row, std::string* error) {
  auto fail = [error](const std::string& message) {
    SetError(error, message);
    return false;
  };
  const size_t n = row.legal_actions.size();
  if (row.row_schema_version != 2) return fail("scratch row schema is not 2.");
  if (row.target_type != "regularized_q_kl" &&
      row.target_type != "normalized_cmpo" &&
      row.target_type != "visit_policy") {
    return fail("scratch row target_type is not supported.");
  }
  if (row.search_budget_class != "full" &&
      row.search_budget_class != "cheap") {
    return fail("scratch row budget class is neither full nor cheap.");
  }
  if (n == 0 || row.raw_policy.size() != n || row.raw_visits.size() != n ||
      row.target_visits.size() != n || row.target_probs.size() != n ||
      row.q_values.size() != n) {
    return fail("scratch row legacy vectors are not aligned.");
  }
  if (!AllFinite(row.observation) || !AllFinite(row.raw_policy) ||
      !AllFinite(row.target_probs) || !AllFinite(row.q_values) ||
      !std::isfinite(row.root_value) || !std::isfinite(row.value_target) ||
      !std::isfinite(row.policy_target_weight) ||
      !std::isfinite(row.regularized_q_beta) ||
      !std::isfinite(row.regularized_q_kl) ||
      !std::isfinite(row.regularized_q_prior_expected_q) ||
      !std::isfinite(row.regularized_q_target_expected_q) ||
      !std::isfinite(row.regularized_q_q_range) ||
      !std::isfinite(row.regularized_q_entropy_norm)) {
    return fail("scratch row contains a non-finite numeric field.");
  }
  if (!row.value_target_attached) {
    return fail("scratch row lacks its terminal value target.");
  }
  if (row.simulations_requested < 0 || row.simulations_completed < 0 ||
      row.simulations_completed > row.simulations_requested) {
    return fail("scratch row simulation accounting is invalid.");
  }
  if (row.policy_target_weight != 0.0 && row.policy_target_weight != 1.0) {
    return fail("scratch row policy weight is not 0 or 1.");
  }
  if (row.target_type == "visit_policy") {
    double visit_sum = std::accumulate(row.target_probs.begin(),
                                       row.target_probs.end(), 0.0);
    const int64_t target_visit_total = std::accumulate(
        row.target_visits.begin(), row.target_visits.end(), int64_t{0});
    if (target_visit_total <= 0 || std::abs(visit_sum - 1.0) > 1e-10) {
      return fail("visit-policy target is not normalized.");
    }
    for (size_t i = 0; i < n; ++i) {
      if (row.target_visits[i] < 0) {
        return fail("visit-policy target visits contain a negative count.");
      }
      const double expected = static_cast<double>(row.target_visits[i]) /
                              static_cast<double>(target_visit_total);
      if (std::abs(row.target_probs[i] - expected) > 1e-10) {
        return fail("visit-policy target does not match target visits.");
      }
    }
    if (row.regularized_q_valid) {
      if (row.regularized_q_error != "none" ||
          row.regularized_q_target.size() != n ||
          !AllFinite(row.regularized_q_target)) {
        return fail("valid diagnostic Q target is malformed.");
      }
      double q_sum = std::accumulate(row.regularized_q_target.begin(),
                                     row.regularized_q_target.end(), 0.0);
      if (std::abs(q_sum - 1.0) > 1e-12 ||
          row.regularized_q_kl > 0.10 + 1e-12) {
        return fail("valid diagnostic Q target violates its contract.");
      }
    } else if (!row.regularized_q_target.empty() ||
               row.regularized_q_error.empty()) {
      return fail("invalid diagnostic Q target is not named.");
    }
  } else if (row.regularized_q_valid) {
    if (row.regularized_q_error != "none" ||
        row.regularized_q_target.size() != n ||
        !AllFinite(row.regularized_q_target)) {
      return fail("valid scratch target is malformed.");
    }
    double sum = std::accumulate(row.regularized_q_target.begin(),
                                 row.regularized_q_target.end(), 0.0);
    if (std::abs(sum - 1.0) > 1e-12 ||
        (row.target_type != "visit_policy" &&
         row.regularized_q_kl > 0.10 + 1e-12)) {
      return fail("valid scratch target violates normalization or KL cap.");
    }
  } else if (!row.regularized_q_target.empty() ||
             row.regularized_q_error.empty() ||
             row.policy_target_weight != 0.0) {
    return fail("invalid scratch target is not a named zero-weight row.");
  }
  if (row.search_budget_class == "cheap" && row.policy_target_weight != 0.0) {
    return fail("cheap scratch row has nonzero policy weight.");
  }
  if (row.search_budget_class == "full" && row.regularized_q_valid &&
      row.policy_target_weight != 1.0) {
    return fail("valid full scratch row lacks unit policy weight.");
  }
  return true;
}

// --- Bounded uniform draw ---------------------------------------------------
//
// Unbiased [0, bound) by rejecting the residue tail. Written out rather than
// delegated to std::uniform_int_distribution because that class's output
// sequence for a given engine state is implementation-defined -- and this
// lane's contract is that a seed reproduces a sample exactly, which a
// standard-library upgrade must not be able to break.
//
// 2^64 mod bound == ((2^64 - 1) mod bound + 1) mod bound, computed in 64 bits.
// Discarding draws above (2^64 - 1) - that residue leaves a count divisible by
// `bound`, so the modulo is exactly uniform.
uint64_t BoundedUniform(std::mt19937_64* rng, uint64_t bound) {
  SPIEL_CHECK_GT(bound, 0u);
  constexpr uint64_t kMax = std::numeric_limits<uint64_t>::max();
  const uint64_t residue = (kMax % bound + 1) % bound;
  const uint64_t accept_at_or_below = kMax - residue;
  uint64_t draw = (*rng)();
  while (draw > accept_at_or_below) draw = (*rng)();
  return draw % bound;
}

constexpr char kShardPrefix[] = "searchpi_cohort_";
constexpr char kShardSuffix[] = ".bin";

}  // namespace

// ===========================================================================
// Public: shard I/O
// ===========================================================================

bool WriteSearchPiRowShard(const std::string& path,
                           const std::vector<SearchPiRow>& rows,
                           std::string* error) {
  namespace fs = std::filesystem;
  const fs::path target(path);
  // An empty parent means a bare filename in the process's cwd. Naming it "."
  // explicitly keeps the temp file a SIBLING of the target in every case, which
  // is what makes the rename below a same-filesystem (and therefore atomic)
  // rename rather than a cross-device copy.
  fs::path dir = target.parent_path();
  if (dir.empty()) dir = fs::path(".");

  // The pid is in the temp name so two processes writing the same generation --
  // which the two-arm design makes possible if arms ever run as separate
  // processes -- cannot scribble on one another's partial file. Whoever renames
  // last wins, and both files were complete.
  const std::string tmp_path =
      (dir / absl::StrCat(target.filename().string(), ".tmp.",
                          static_cast<long long>(::getpid())))
          .string();

  const int fd = ::open(tmp_path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
  if (fd < 0) {
    SetError(error, ErrnoText("opening shard temp file", tmp_path));
    return false;
  }

  // Any exit below this point that is not a success must remove the temp file:
  // a stray .tmp. left in the cohort directory is harmless to the retention
  // scan (it does not parse as a canonical shard name) but it is still litter
  // that accumulates once per failure.
  auto abort_write = [&](std::string text) {
    ::close(fd);
    std::error_code ec;
    fs::remove(tmp_path, ec);
    SetError(error, std::move(text));
    return false;
  };

  ShardWriter w(fd, tmp_path);
  w.PutU32(kSearchPiShardMagic);
  w.PutU32(kSearchPiShardVersion);
  w.PutU64(static_cast<uint64_t>(rows.size()));
  for (const SearchPiRow& row : rows) SerializeRow(&w, row);
  w.PutU32(kSearchPiShardTrailerMagic);
  w.PutU64(static_cast<uint64_t>(rows.size()));
  if (!w.Flush()) return abort_write(w.error());

  // fsync BEFORE the rename. Without it a crash can leave the directory entry
  // pointing at a file whose data blocks never reached the device -- the exact
  // half-written shard the atomicity contract promises never to expose.
  while (::fsync(fd) != 0) {
    if (errno == EINTR) continue;
    return abort_write(ErrnoText("fsync of shard temp file", tmp_path));
  }
  if (::close(fd) != 0) {
    std::error_code ec;
    fs::remove(tmp_path, ec);
    SetError(error, ErrnoText("closing shard temp file", tmp_path));
    return false;
  }

  std::error_code ec;
  fs::rename(tmp_path, path, ec);
  if (ec) {
    // The message is captured BEFORE the cleanup: fs::remove overwrites `ec`,
    // and reading it afterwards would report the removal's status as if it were
    // the rename's.
    const std::string why = ec.message();
    std::error_code cleanup_ec;
    fs::remove(tmp_path, cleanup_ec);
    SetError(error, absl::StrCat("renaming '", tmp_path, "' to '", path,
                                 "' failed: ", why, "."));
    return false;
  }

  // Best-effort durability of the rename itself. Deliberately NOT fatal: the
  // data file is already fsynced and the rename has already made the shard
  // either wholly visible or not visible at all, so a filesystem that refuses
  // a directory fsync weakens crash durability without breaking the
  // never-observe-a-partial-shard property this function actually promises.
  const int dir_fd = ::open(dir.c_str(), O_RDONLY | O_DIRECTORY);
  if (dir_fd >= 0) {
    while (::fsync(dir_fd) != 0 && errno == EINTR) {
    }
    ::close(dir_fd);
  }
  return true;
}

bool ReadSearchPiRowShard(const std::string& path,
                          std::vector<SearchPiRow>* rows, std::string* error) {
  SPIEL_CHECK_TRUE(rows != nullptr);
  rows->clear();

  std::ifstream in(path, std::ios::binary | std::ios::ate);
  if (!in) {
    SetError(error, absl::StrCat("cannot open shard '", path,
                                 "' for reading; it is missing or"
                                 " unreadable."));
    return false;
  }
  const std::streamoff end = in.tellg();
  if (end < 0) {
    SetError(error,
             absl::StrCat("cannot determine the size of shard '", path, "'."));
    return false;
  }
  const uint64_t size = static_cast<uint64_t>(end);
  in.seekg(0, std::ios::beg);
  if (!in.good()) {
    SetError(error, absl::StrCat("cannot rewind shard '", path, "'."));
    return false;
  }

  ShardReader r(&in, size, path);

  uint32_t magic = 0;
  uint32_t version = 0;
  uint64_t row_count = 0;
  if (!r.U32(&magic, "magic") || !r.U32(&version, "version") ||
      !r.U64(&row_count, "row_count")) {
    SetError(error, r.error());
    return false;
  }
  if (magic != kSearchPiShardMagic) {
    r.Fail(absl::StrCat("magic is 0x", absl::Hex(magic), ", expected 0x",
                        absl::Hex(kSearchPiShardMagic), "."));
    SetError(error, r.error());
    return false;
  }
  if (version != kSearchPiShardVersion) {
    // No forward compatibility on purpose. A shard from another version is not
    // "probably fine": the layout IS the contract, and reading one with the
    // wrong field order would produce rows that look valid and train wrong.
    r.Fail(absl::StrCat("version is ", version, ", but this build reads only"
                                                " version ",
                        kSearchPiShardVersion, "."));
    SetError(error, r.error());
    return false;
  }
  // Reject an impossible count before allocating anything at all. Without this
  // a header claiming 2^60 rows would reserve its way into the OOM killer
  // before the first per-row bound could fire.
  if (row_count > r.remaining() / kMinSerializedRowBytes) {
    r.Fail(absl::StrCat("header declares ", row_count, " rows, which cannot fit"
                        " in the ",
                        r.remaining(), " bytes that follow it."));
    SetError(error, r.error());
    return false;
  }

  std::vector<SearchPiRow> parsed;
  parsed.resize(static_cast<size_t>(row_count));
  for (uint64_t i = 0; i < row_count; ++i) {
    if (!DeserializeRow(&r, &parsed[static_cast<size_t>(i)])) {
      SetError(error, absl::StrCat(r.error(), " (at row index ", i, ")."));
      return false;
    }
  }

  uint32_t trailer = 0;
  uint64_t trailer_count = 0;
  if (!r.U32(&trailer, "trailer_magic") ||
      !r.U64(&trailer_count, "trailer_row_count")) {
    SetError(error, r.error());
    return false;
  }
  if (trailer != kSearchPiShardTrailerMagic) {
    r.Fail(absl::StrCat("trailer magic is 0x", absl::Hex(trailer),
                        ", expected 0x", absl::Hex(kSearchPiShardTrailerMagic),
                        "; the shard is truncated or damaged."));
    SetError(error, r.error());
    return false;
  }
  if (trailer_count != row_count) {
    r.Fail(absl::StrCat("trailer row count ", trailer_count,
                        " disagrees with the header's ", row_count, "."));
    SetError(error, r.error());
    return false;
  }
  if (r.remaining() != 0) {
    r.Fail(absl::StrCat(r.remaining(),
                        " unexpected bytes follow the trailer."));
    SetError(error, r.error());
    return false;
  }

  *rows = std::move(parsed);
  return true;
}

bool WriteScratchSearchPiRowShardV2(const std::string& path,
                                    const std::vector<SearchPiRow>& rows,
                                    std::string* error) {
  namespace fs = std::filesystem;
  for (size_t i = 0; i < rows.size(); ++i) {
    std::string why;
    if (!ValidateScratchRowV2(rows[i], &why)) {
      SetError(error, absl::StrCat("scratch shard row ", i,
                                   " is invalid: ", why));
      return false;
    }
  }

  const fs::path target(path);
  fs::path dir = target.parent_path();
  if (dir.empty()) dir = fs::path(".");
  const std::string tmp_path =
      (dir / absl::StrCat(target.filename().string(), ".tmp.",
                          static_cast<long long>(::getpid())))
          .string();
  const int fd = ::open(tmp_path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
  if (fd < 0) {
    SetError(error, ErrnoText("opening scratch shard temp file", tmp_path));
    return false;
  }
  auto abort_write = [&](std::string message) {
    ::close(fd);
    std::error_code ec;
    fs::remove(tmp_path, ec);
    SetError(error, std::move(message));
    return false;
  };

  ShardWriter w(fd, tmp_path);
  w.PutU32(kSearchPiShardMagic);
  w.PutU32(kScratchSearchPiShardVersion);
  w.PutU64(static_cast<uint64_t>(rows.size()));
  for (const SearchPiRow& row : rows) SerializeScratchRowV2(&w, row);
  w.PutU32(kSearchPiShardTrailerMagic);
  w.PutU64(static_cast<uint64_t>(rows.size()));
  if (!w.Flush()) return abort_write(w.error());
  while (::fsync(fd) != 0) {
    if (errno == EINTR) continue;
    return abort_write(ErrnoText("fsync of scratch shard temp file", tmp_path));
  }
  if (::close(fd) != 0) {
    std::error_code ec;
    fs::remove(tmp_path, ec);
    SetError(error, ErrnoText("closing scratch shard temp file", tmp_path));
    return false;
  }
  std::error_code ec;
  fs::rename(tmp_path, path, ec);
  if (ec) {
    const std::string why = ec.message();
    std::error_code cleanup_ec;
    fs::remove(tmp_path, cleanup_ec);
    SetError(error, absl::StrCat("renaming '", tmp_path, "' to '", path,
                                 "' failed: ", why, "."));
    return false;
  }
  const int dir_fd = ::open(dir.c_str(), O_RDONLY | O_DIRECTORY);
  if (dir_fd >= 0) {
    while (::fsync(dir_fd) != 0 && errno == EINTR) {
    }
    ::close(dir_fd);
  }
  return true;
}

bool ReadScratchSearchPiRowShardV2(const std::string& path,
                                   std::vector<SearchPiRow>* rows,
                                   std::string* error) {
  SPIEL_CHECK_TRUE(rows != nullptr);
  rows->clear();
  std::ifstream in(path, std::ios::binary | std::ios::ate);
  if (!in) {
    SetError(error, absl::StrCat("cannot open scratch shard '", path,
                                 "' for reading; it is missing or unreadable."));
    return false;
  }
  const std::streamoff end = in.tellg();
  if (end < 0) {
    SetError(error, absl::StrCat("cannot determine the size of scratch shard '",
                                 path, "'."));
    return false;
  }
  const uint64_t size = static_cast<uint64_t>(end);
  in.seekg(0, std::ios::beg);
  if (!in.good()) {
    SetError(error, absl::StrCat("cannot rewind scratch shard '", path, "'."));
    return false;
  }

  ShardReader r(&in, size, path);
  uint32_t magic = 0;
  uint32_t version = 0;
  uint64_t row_count = 0;
  if (!r.U32(&magic, "magic") || !r.U32(&version, "version") ||
      !r.U64(&row_count, "row_count")) {
    SetError(error, r.error());
    return false;
  }
  if (magic != kSearchPiShardMagic) {
    r.Fail(absl::StrCat("magic is 0x", absl::Hex(magic), ", expected 0x",
                        absl::Hex(kSearchPiShardMagic), "."));
  } else if (version != kScratchSearchPiShardVersion) {
    r.Fail(absl::StrCat("version is ", version,
                        ", but Search-PI Q profiles read only version ",
                        kScratchSearchPiShardVersion, "."));
  } else if (row_count > r.remaining() / kMinSerializedRowBytes) {
    r.Fail(absl::StrCat("header declares ", row_count,
                        " rows, which cannot fit in the remaining bytes."));
  }
  if (r.failed()) {
    SetError(error, r.error());
    return false;
  }

  std::vector<SearchPiRow> parsed(static_cast<size_t>(row_count));
  for (uint64_t i = 0; i < row_count; ++i) {
    if (!DeserializeScratchRowV2(&r, &parsed[static_cast<size_t>(i)])) {
      SetError(error, absl::StrCat(r.error(), " (at row index ", i, ")."));
      return false;
    }
    std::string why;
    if (!ValidateScratchRowV2(parsed[static_cast<size_t>(i)], &why)) {
      SetError(error, absl::StrCat("scratch shard row ", i,
                                   " failed validation: ", why));
      return false;
    }
  }
  uint32_t trailer = 0;
  uint64_t trailer_count = 0;
  if (!r.U32(&trailer, "trailer_magic") ||
      !r.U64(&trailer_count, "trailer_row_count")) {
    SetError(error, r.error());
    return false;
  }
  if (trailer != kSearchPiShardTrailerMagic) {
    r.Fail("scratch shard trailer magic is invalid or truncated.");
  } else if (trailer_count != row_count) {
    r.Fail("scratch shard trailer row count disagrees with its header.");
  } else if (r.remaining() != 0) {
    r.Fail(absl::StrCat(r.remaining(),
                        " unexpected bytes follow the scratch trailer."));
  }
  if (r.failed()) {
    SetError(error, r.error());
    return false;
  }
  *rows = std::move(parsed);
  return true;
}

// ===========================================================================
// Public: uniform replay-window sampler
// ===========================================================================

std::vector<SearchPiRow> SampleUniformReplayWindow(
    const std::vector<std::vector<SearchPiRow>>& window, size_t sample_count,
    uint64_t seed) {
  // One flat pointer table over every cohort, cohort-major. This is what makes
  // the draw uniform over the UNION: there is no per-cohort step anywhere below,
  // so cohort age cannot enter the probability. The table costs 8 bytes a row
  // against the ~23 KB a row occupies, so materialising it is free next to any
  // scheme that indexed cohorts separately.
  std::vector<const SearchPiRow*> flat;
  size_t total = 0;
  for (const std::vector<SearchPiRow>& cohort : window) total += cohort.size();
  flat.reserve(total);
  for (const std::vector<SearchPiRow>& cohort : window) {
    for (const SearchPiRow& row : cohort) flat.push_back(&row);
  }

  // The caller's guarantee, checked rather than trusted. Sampling more rows
  // than the window holds has no correct answer -- without replacement it is
  // simply impossible -- and silently clamping would hand arm W fewer rows than
  // arm F, which is precisely the confound the equal-row-count design exists to
  // rule out.
  SPIEL_CHECK_LE(sample_count, total);

  // Domain separation before the seed reaches the generator: DeriveSeed mixes
  // the caller's seed with this lane's replay stream, so the same base seed
  // driving a generation's chance/behaviour streams cannot produce a correlated
  // replay sample.
  std::mt19937_64 rng =
      dune_seed::MakeRng64(dune_seed::DeriveSeed(seed, kStreamPiReplaySample));

  // Partial Fisher-Yates: exactly `sample_count` swaps, each drawing uniformly
  // from the not-yet-selected suffix. This is uniform without replacement by
  // construction and costs one pass, where rejection-resampling a set would
  // degrade badly as sample_count approaches total.
  std::vector<size_t> index(total);
  std::iota(index.begin(), index.end(), size_t{0});
  for (size_t i = 0; i < sample_count; ++i) {
    const size_t j =
        i + static_cast<size_t>(BoundedUniform(&rng, total - i));
    std::swap(index[i], index[j]);
  }

  // Canonical order, not draw order -- see the header. Sorting the CHOSEN
  // indices (not the rows) keeps this cheap and keeps two samples of the same
  // window directly comparable position by position.
  std::vector<size_t> chosen(index.begin(), index.begin() + sample_count);
  std::sort(chosen.begin(), chosen.end());

  std::vector<SearchPiRow> out;
  out.reserve(sample_count);
  for (size_t g : chosen) out.push_back(*flat[g]);
  return out;
}

std::vector<SearchPiRow> SampleScratchSearchPiReplayWindow(
    const std::vector<std::vector<SearchPiRow>>& window,
    int current_generation, size_t max_rows, uint64_t seed,
    std::string* error) {
  if (error != nullptr) error->clear();
  std::vector<const SearchPiRow*> flat;
  for (const auto& cohort : window) {
    for (const SearchPiRow& row : cohort) {
      if (row.row_schema_version != 2) {
        SetError(error, "scratch replay window contains a non-v2 row.");
        return {};
      }
      flat.push_back(&row);
    }
  }
  const size_t target_count = std::min(max_rows, flat.size());
  std::vector<bool> selected(flat.size(), false);
  size_t required = 0;
  for (size_t i = 0; i < flat.size(); ++i) {
    const SearchPiRow& row = *flat[i];
    if (row.generation == current_generation &&
        row.search_budget_class == "full" &&
        row.policy_target_weight > 0.0) {
      selected[i] = true;
      ++required;
    }
  }
  if (required > target_count) {
    SetError(error, absl::StrCat(
                        "current full rows (", required,
                        ") exceed the scratch replay cap (", target_count,
                        ")."));
    return {};
  }

  std::vector<size_t> remaining;
  remaining.reserve(flat.size() - required);
  for (size_t i = 0; i < flat.size(); ++i) {
    if (!selected[i]) remaining.push_back(i);
  }
  const size_t need = target_count - required;
  std::mt19937_64 rng = dune_seed::MakeRng64(
      dune_seed::DeriveSeed(seed, kStreamPiScratchReplaySample,
                            current_generation));
  for (size_t i = 0; i < need; ++i) {
    const size_t j = i + static_cast<size_t>(
                             BoundedUniform(&rng, remaining.size() - i));
    std::swap(remaining[i], remaining[j]);
    selected[remaining[i]] = true;
  }

  std::vector<SearchPiRow> out;
  out.reserve(target_count);
  for (size_t i = 0; i < flat.size(); ++i) {
    if (selected[i]) out.push_back(*flat[i]);
  }
  return out;
}

// ===========================================================================
// Public: retention
// ===========================================================================

std::string SearchPiShardPathForGeneration(const std::string& dir,
                                           int generation) {
  return (std::filesystem::path(dir) /
          absl::StrFormat("%s%06d%s", kShardPrefix, generation, kShardSuffix))
      .string();
}

std::string ScratchSearchPiShardPathForGeneration(const std::string& dir,
                                                  int generation) {
  return (std::filesystem::path(dir) /
          absl::StrFormat("scratch_q_v1_%06d.bin", generation))
      .string();
}

bool ParseSearchPiShardGeneration(const std::string& path, int* generation) {
  SPIEL_CHECK_TRUE(generation != nullptr);
  const std::string name = std::filesystem::path(path).filename().string();
  const size_t prefix_len = std::strlen(kShardPrefix);
  const size_t suffix_len = std::strlen(kShardSuffix);
  if (name.size() <= prefix_len + suffix_len) return false;
  if (name.compare(0, prefix_len, kShardPrefix) != 0) return false;
  if (name.compare(name.size() - suffix_len, suffix_len, kShardSuffix) != 0) {
    return false;
  }
  const std::string digits =
      name.substr(prefix_len, name.size() - prefix_len - suffix_len);
  // Digits only. Accepting a leading '+'/'-' or stray whitespace here would let
  // "searchpi_cohort_-00001.bin" into the window with a negative generation,
  // and accepting an overlong run of digits would silently wrap.
  if (digits.empty() || digits.size() > 9) return false;
  for (char c : digits) {
    if (c < '0' || c > '9') return false;
  }
  const int parsed = std::stoi(digits);
  // Round-trip against the writer's own format. Without this, "…_7.bin" and
  // "…_000007.bin" both parse to generation 7, and two files claiming one
  // cohort would make the retention sort ambiguous -- which of them survives
  // would depend on directory order, exactly the nondeterminism the sort was
  // added to remove. Accepting only names this build would itself write makes
  // the generation a KEY rather than a coincidence.
  if (digits != absl::StrFormat("%06d", parsed)) return false;
  *generation = parsed;
  return true;
}

SearchPiReplayRetention PlanSearchPiReplayRetention(
    const std::string& dir, const std::string& new_shard_path,
    size_t max_window_cohorts) {
  namespace fs = std::filesystem;
  SearchPiReplayRetention out;
  SPIEL_CHECK_GT(max_window_cohorts, 0u);

  int new_generation = 0;
  if (!ParseSearchPiShardGeneration(new_shard_path, &new_generation)) {
    out.error = absl::StrCat("'", new_shard_path,
                             "' is not a canonical shard name of the form ",
                             kShardPrefix, "NNNNNN", kShardSuffix, ".");
    return out;
  }

  std::error_code ec;
  if (!fs::is_directory(dir, ec) || ec) {
    out.error = absl::StrCat("'", dir, "' is not an existing directory.");
    return out;
  }

  fs::directory_iterator it(dir, ec);
  if (ec) {
    out.error = absl::StrCat("listing '", dir, "' failed: ", ec.message(), ".");
    return out;
  }

  std::vector<std::pair<int, std::string>> shards;
  for (const fs::directory_entry& entry : it) {
    std::error_code entry_ec;
    if (!entry.is_regular_file(entry_ec) || entry_ec) continue;
    int generation = 0;
    // Non-canonical names are IGNORED, not an error. A cohort directory legally
    // holds manifests and logs beside its shards, and a half-written .tmp.<pid>
    // from a crashed writer must never be mistaken for a cohort -- which is
    // exactly what the name filter buys.
    if (!ParseSearchPiShardGeneration(entry.path().string(), &generation)) {
      continue;
    }
    if (generation == new_generation) continue;  // replaced below
    shards.emplace_back(generation, entry.path().string());
  }
  // The caller-supplied path is authoritative for its own generation, whether
  // or not the directory scan already saw a file there.
  shards.emplace_back(new_generation, new_shard_path);

  // Ordered by the PARSED generation. Directory iteration order is unspecified,
  // so this sort is load-bearing, not cosmetic: an unsorted window would evict
  // whichever cohort the filesystem happened to hand back first.
  std::sort(shards.begin(), shards.end(),
            [](const std::pair<int, std::string>& a,
               const std::pair<int, std::string>& b) {
              return a.first < b.first;
            });

  const size_t keep = std::min(max_window_cohorts, shards.size());
  const size_t first_retained = shards.size() - keep;
  out.retained.reserve(keep);
  out.evicted.reserve(first_retained);
  for (size_t i = 0; i < shards.size(); ++i) {
    if (i < first_retained) {
      out.evicted.push_back(shards[i].second);
    } else {
      out.retained.push_back(shards[i].second);
    }
  }
  out.ok = true;
  return out;
}

}  // namespace open_spiel
