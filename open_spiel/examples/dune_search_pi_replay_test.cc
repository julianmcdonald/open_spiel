// Acceptance tests for the search-PI replay store and replay-window sampler.
//
// Six numbered tests, one per property the two-arm replay experiment depends
// on. Everything runs on CPU, touches no model, and calls nothing from
// dune_search_pi.cc -- SearchPiRow is a plain struct, so this binary needs only
// the replay translation unit plus OpenSpiel's core.
//
// The file tests use a temp directory under /tmp and remove it on the way out.
// A SPIEL_CHECK failure aborts the process and therefore leaks that directory;
// that is the deliberate trade, since the surviving shard bytes are the only
// evidence of what a format failure actually wrote.

#include "dune_search_pi_replay.h"

#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <set>
#include <string>
#include <system_error>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/flags/flag.h"
#include "open_spiel/spiel_utils.h"

// Link-only PPO flag definitions, exactly as dune_search_pi_test.cc carries
// them (:38-51). This target pulls dune_search_pi.cc, which reaches
// dune_ppo_training_utils.cc, which REFERENCES these flags without defining
// them -- they belong to the trainer binary. None of them is read on any path
// this test exercises; they exist so the link resolves. Values are the
// trainer's defaults so an accidental read would at least be unsurprising.
ABSL_FLAG(int, ppo_minibatch_size, 2048, "");
ABSL_FLAG(int, ppo_update_epochs, 4, "");
ABSL_FLAG(double, ppo_clip_epsilon, 0.2, "");
ABSL_FLAG(bool, normalize_advantages, true, "");
ABSL_FLAG(bool, ppo_clip_value_loss, true, "");
ABSL_FLAG(double, entropy_coef, 0.01, "");
ABSL_FLAG(double, value_coef, 0.5, "");
ABSL_FLAG(double, logit_cap, 10.0, "");
ABSL_FLAG(double, target_kl, 0.0, "");
ABSL_FLAG(bool, train_amp, true, "");
ABSL_FLAG(double, grad_clip_norm, 0.5, "");
ABSL_FLAG(uint64_t, shaping_start_env_steps, 206830543, "");
ABSL_FLAG(uint64_t, shaping_decay_env_steps, 0, "");
ABSL_FLAG(bool, diagnostics_only, false, "");

namespace open_spiel {
namespace {

// --- Bit-level comparison ---------------------------------------------------
//
// Floats are compared as BIT PATTERNS, not as numbers. That is the whole claim
// the binary format makes: `==` on floats would call two different bit patterns
// equal for -0.0 vs +0.0, and would call a NaN unequal to itself, so `==` can
// neither confirm nor refute exactness. memcpy to an unsigned integer is the
// only comparison that can.
uint32_t FloatBits(float v) {
  uint32_t bits = 0;
  std::memcpy(&bits, &v, sizeof(bits));
  return bits;
}

uint64_t DoubleBits(double v) {
  uint64_t bits = 0;
  std::memcpy(&bits, &v, sizeof(bits));
  return bits;
}

void CheckFloatVectorsBitEqual(const std::vector<float>& a,
                               const std::vector<float>& b,
                               const char* field) {
  SPIEL_CHECK_EQ(a.size(), b.size());
  for (size_t i = 0; i < a.size(); ++i) {
    if (FloatBits(a[i]) != FloatBits(b[i])) {
      std::cerr << "float mismatch in " << field << " at index " << i << "\n";
    }
    SPIEL_CHECK_EQ(FloatBits(a[i]), FloatBits(b[i]));
  }
}

void CheckDoubleVectorsBitEqual(const std::vector<double>& a,
                                const std::vector<double>& b,
                                const char* field) {
  SPIEL_CHECK_EQ(a.size(), b.size());
  for (size_t i = 0; i < a.size(); ++i) {
    if (DoubleBits(a[i]) != DoubleBits(b[i])) {
      std::cerr << "double mismatch in " << field << " at index " << i << "\n";
    }
    SPIEL_CHECK_EQ(DoubleBits(a[i]), DoubleBits(b[i]));
  }
}

void CheckIntVectorsEqual(const std::vector<int>& a, const std::vector<int>& b) {
  SPIEL_CHECK_EQ(a.size(), b.size());
  for (size_t i = 0; i < a.size(); ++i) SPIEL_CHECK_EQ(a[i], b[i]);
}

void CheckActionVectorsEqual(const std::vector<Action>& a,
                             const std::vector<Action>& b) {
  SPIEL_CHECK_EQ(a.size(), b.size());
  for (size_t i = 0; i < a.size(); ++i) SPIEL_CHECK_EQ(a[i], b[i]);
}

// EVERY field of SearchPiRow, in declaration order. Enumerated by hand rather
// than hashed: a hash tells you two rows differ, this tells you which field,
// and a field silently dropped by the serializer is exactly the failure this
// test exists to name.
void CheckRowsBitwiseEqual(const SearchPiRow& a, const SearchPiRow& b) {
  CheckFloatVectorsBitEqual(a.observation, b.observation, "observation");
  SPIEL_CHECK_EQ(a.observation_is_information_state,
                 b.observation_is_information_state);
  SPIEL_CHECK_EQ(a.player, b.player);
  SPIEL_CHECK_EQ(static_cast<int>(a.role), static_cast<int>(b.role));

  CheckActionVectorsEqual(a.legal_actions, b.legal_actions);
  CheckDoubleVectorsBitEqual(a.raw_policy, b.raw_policy, "raw_policy");
  CheckIntVectorsEqual(a.raw_visits, b.raw_visits);
  CheckIntVectorsEqual(a.target_visits, b.target_visits);
  CheckDoubleVectorsBitEqual(a.target_probs, b.target_probs, "target_probs");
  SPIEL_CHECK_EQ(a.chosen_action, b.chosen_action);

  SPIEL_CHECK_EQ(DoubleBits(a.value_target), DoubleBits(b.value_target));
  SPIEL_CHECK_EQ(a.value_target_attached, b.value_target_attached);

  SPIEL_CHECK_EQ(a.generation, b.generation);
  SPIEL_CHECK_EQ(a.episode_id, b.episode_id);
  SPIEL_CHECK_EQ(a.decision_id, b.decision_id);
  SPIEL_CHECK_EQ(a.state_fingerprint, b.state_fingerprint);

  SPIEL_CHECK_EQ(a.simulations_completed, b.simulations_completed);
  SPIEL_CHECK_EQ(a.simulations_requested, b.simulations_requested);
  SPIEL_CHECK_EQ(static_cast<int>(a.early_exit), static_cast<int>(b.early_exit));
  SPIEL_CHECK_EQ(a.inherited_root_visits, b.inherited_root_visits);
  CheckIntVectorsEqual(a.visits_pre_search, b.visits_pre_search);
  SPIEL_CHECK_EQ(a.retention_snapshots_valid, b.retention_snapshots_valid);
  SPIEL_CHECK_EQ(a.re_root_status, b.re_root_status);
  SPIEL_CHECK_EQ(static_cast<int>(a.fallback), static_cast<int>(b.fallback));
  SPIEL_CHECK_EQ(a.search_fallback_reason, b.search_fallback_reason);
  SPIEL_CHECK_EQ(DoubleBits(a.root_value), DoubleBits(b.root_value));
  CheckDoubleVectorsBitEqual(a.q_values, b.q_values, "q_values");

  SPIEL_CHECK_EQ(DoubleBits(a.raw_policy_entropy_norm),
                 DoubleBits(b.raw_policy_entropy_norm));
  SPIEL_CHECK_EQ(DoubleBits(a.target_entropy_norm),
                 DoubleBits(b.target_entropy_norm));
  SPIEL_CHECK_EQ(DoubleBits(a.raw_policy_max_prob),
                 DoubleBits(b.raw_policy_max_prob));
  SPIEL_CHECK_EQ(DoubleBits(a.target_max_prob), DoubleBits(b.target_max_prob));
  SPIEL_CHECK_EQ(DoubleBits(a.kl_target_given_raw),
                 DoubleBits(b.kl_target_given_raw));
  SPIEL_CHECK_EQ(a.kl_raw_prior_floored, b.kl_raw_prior_floored);
  SPIEL_CHECK_EQ(a.target_argmax_differs_from_raw,
                 b.target_argmax_differs_from_raw);

  SPIEL_CHECK_EQ(a.num_covered_actions, b.num_covered_actions);
  SPIEL_CHECK_EQ(DoubleBits(a.covered_prior_mass),
                 DoubleBits(b.covered_prior_mass));
  SPIEL_CHECK_EQ(a.would_pass_legacy_coverage_gate,
                 b.would_pass_legacy_coverage_gate);
}

// --- Fixtures ---------------------------------------------------------------

// The float values a text format cannot carry back unchanged. 1/3 and 1/7 are
// not exactly representable in any short decimal; the denormal and
// std::numeric_limits<float>::min() sit below the 5e-7 floor OpenSpiel's
// json.cc rounds to zero; -0.0f and the two infinities and the NaN have no
// decimal spelling at all that survives a naive parse. If any single one of
// these comes back with a different bit pattern the store is lossy.
const std::vector<float>& AwkwardFloats() {
  static const std::vector<float> kValues = {
      1.0f / 3.0f,
      1.0f / 7.0f,
      -0.0f,
      0.0f,
      std::numeric_limits<float>::denorm_min(),
      -std::numeric_limits<float>::denorm_min(),
      std::numeric_limits<float>::min(),
      std::numeric_limits<float>::max(),
      std::numeric_limits<float>::lowest(),
      std::numeric_limits<float>::epsilon(),
      std::numeric_limits<float>::infinity(),
      -std::numeric_limits<float>::infinity(),
      std::numeric_limits<float>::quiet_NaN(),
      std::nextafter(1.0f, 2.0f),
      std::nextafter(1.0f, 0.0f),
      1e-30f,
  };
  return kValues;
}

const std::vector<double>& AwkwardDoubles() {
  static const std::vector<double> kValues = {
      1.0 / 3.0,
      1.0 / 7.0,
      -0.0,
      std::numeric_limits<double>::denorm_min(),
      std::numeric_limits<double>::min(),
      std::numeric_limits<double>::max(),
      std::numeric_limits<double>::epsilon(),
      std::nextafter(1.0, 2.0),
      1e-300,
      -1.0 / 3.0,
  };
  return kValues;
}

// A fully populated row. Every field gets a value derived from the identity
// triple, so a serializer that wrote a field twice, skipped one, or transposed
// two of the same width would land on a different value and fail the
// comparison rather than pass by coincidence.
SearchPiRow MakeRow(int generation, int64_t episode, int decision,
                    size_t obs_len, bool awkward) {
  SearchPiRow row;
  row.observation.resize(obs_len);
  for (size_t i = 0; i < obs_len; ++i) {
    row.observation[i] =
        static_cast<float>(generation * 1000 + decision) + static_cast<float>(i);
  }
  if (awkward) {
    // The awkward values go at the FRONT so a truncating serializer that kept a
    // prefix would still be caught, and the tail keeps ordinary values so a
    // reader that mixed up element order is caught too.
    const std::vector<float>& awk = AwkwardFloats();
    for (size_t i = 0; i < awk.size() && i < obs_len; ++i) {
      row.observation[i] = awk[i];
    }
  }
  row.observation_is_information_state = ((decision % 2) == 0);
  row.player = static_cast<Player>(decision % 4);
  row.role = static_cast<DuneDecisionRole>((generation + decision) % 7);

  const size_t n_legal = 5;
  row.legal_actions.resize(n_legal);
  row.raw_policy.resize(n_legal);
  row.raw_visits.resize(n_legal);
  row.target_visits.resize(n_legal);
  row.target_probs.resize(n_legal);
  row.q_values.resize(n_legal);
  row.visits_pre_search.resize(n_legal);
  const std::vector<double>& awkd = AwkwardDoubles();
  for (size_t j = 0; j < n_legal; ++j) {
    row.legal_actions[j] =
        static_cast<Action>(1000000000LL * static_cast<int64_t>(j + 1) + episode);
    row.raw_policy[j] = awkward ? awkd[j % awkd.size()]
                                : 0.2 + 0.01 * static_cast<double>(j);
    row.raw_visits[j] = 10 + static_cast<int>(j) + decision;
    row.target_visits[j] = 20 + static_cast<int>(j) + decision;
    row.target_probs[j] = awkward ? awkd[(j + 3) % awkd.size()]
                                  : 0.2 - 0.01 * static_cast<double>(j);
    row.q_values[j] = awkward ? awkd[(j + 6) % awkd.size()]
                              : -0.5 + 0.1 * static_cast<double>(j);
    row.visits_pre_search[j] = 3 + static_cast<int>(j);
  }
  row.chosen_action = row.legal_actions[static_cast<size_t>(decision) % n_legal];

  // Not a ladder rung on purpose: 0.5625 round-trips through text unchanged, so
  // it would prove nothing about the format.
  row.value_target = awkward ? (1.0 / 3.0) - static_cast<double>(generation)
                             : 0.0625 * static_cast<double>(generation + 1);
  row.value_target_attached = true;

  row.generation = generation;
  row.episode_id = episode;
  row.decision_id = decision;
  row.state_fingerprint =
      "fp_" + std::to_string(generation) + "_" + std::to_string(episode) + "_" +
      std::to_string(decision);

  row.simulations_completed = 200 - decision;
  row.simulations_requested = 200;
  row.early_exit =
      static_cast<SearchPiEarlyExit>(decision % kSearchPiEarlyExitCount);
  row.inherited_root_visits = 7 * decision;
  row.retention_snapshots_valid = ((decision % 3) == 0);
  row.re_root_status = (decision % 3 == 0) ? "hit"
                       : (decision % 3 == 1) ? "miss"
                                             : "none";
  row.fallback = static_cast<SearchPiFallback>((decision + 1) % 7);
  row.search_fallback_reason = (decision % 2 == 0) ? "none" : "timeout";
  row.root_value = awkward ? -1.0 / 7.0 : 0.125 * static_cast<double>(decision);

  row.raw_policy_entropy_norm = awkward ? 1.0 / 3.0 : 0.91;
  row.target_entropy_norm = awkward ? 1.0 / 7.0 : 0.42;
  row.raw_policy_max_prob = awkward ? std::nextafter(0.5, 1.0) : 0.51;
  row.target_max_prob = awkward ? std::nextafter(0.5, 0.0) : 0.61;
  row.kl_target_given_raw =
      awkward ? std::numeric_limits<double>::denorm_min() : 0.037;
  row.kl_raw_prior_floored = ((decision % 5) == 0);
  row.target_argmax_differs_from_raw = ((decision % 7) == 0);

  row.num_covered_actions = static_cast<int>(n_legal) - 1;
  row.covered_prior_mass = awkward ? 1.0 - std::numeric_limits<double>::epsilon()
                                   : 0.97;
  row.would_pass_legacy_coverage_gate = ((decision % 4) == 0);
  return row;
}

// Cheap rows for the sampler tests: identity is what those assert on, so the
// observation stays at four floats and 4,000 rows cost nothing.
std::vector<SearchPiRow> MakeCohort(int generation, int rows) {
  std::vector<SearchPiRow> cohort;
  cohort.reserve(rows);
  for (int i = 0; i < rows; ++i) {
    cohort.push_back(MakeRow(generation, 1000LL * generation + i, i,
                             /*obs_len=*/4, /*awkward=*/false));
  }
  return cohort;
}

// The (cohort, position) identity a sampled row carries. MakeCohort sets
// generation to the cohort index and decision_id to the position, so this pair
// recovers the row's exact place in the flattened window.
using RowKey = std::pair<int, int>;

std::vector<RowKey> KeysOf(const std::vector<SearchPiRow>& rows) {
  std::vector<RowKey> keys;
  keys.reserve(rows.size());
  for (const SearchPiRow& r : rows) keys.emplace_back(r.generation, r.decision_id);
  return keys;
}

// --- Raw file helpers for the corruption test -------------------------------

std::string ReadWholeFile(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  SPIEL_CHECK_TRUE(in.good());
  return std::string((std::istreambuf_iterator<char>(in)),
                     std::istreambuf_iterator<char>());
}

void WriteWholeFile(const std::string& path, const std::string& bytes) {
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  SPIEL_CHECK_TRUE(out.good());
  out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
  out.close();
  SPIEL_CHECK_TRUE(out.good());
}

void PatchU32(std::string* bytes, size_t offset, uint32_t value) {
  SPIEL_CHECK_LE(offset + sizeof(value), bytes->size());
  std::memcpy(&(*bytes)[offset], &value, sizeof(value));
}

void PatchU64(std::string* bytes, size_t offset, uint64_t value) {
  SPIEL_CHECK_LE(offset + sizeof(value), bytes->size());
  std::memcpy(&(*bytes)[offset], &value, sizeof(value));
}

// Every corruption case asserts the SAME three things: false, a non-empty
// reason, and an EMPTY output vector. The output vector is pre-filled with
// junk first, because "returned false but left three rows behind" is the
// failure mode that would quietly shrink a cohort rather than announce itself.
void ExpectShardReadFails(const std::string& path, const char* label) {
  std::vector<SearchPiRow> rows;
  rows.resize(3);
  std::string error = "untouched";
  const bool ok = ReadSearchPiRowShard(path, &rows, &error);
  SPIEL_CHECK_FALSE(ok);
  SPIEL_CHECK_FALSE(error.empty());
  SPIEL_CHECK_NE(error, std::string("untouched"));
  SPIEL_CHECK_TRUE(rows.empty());
  std::cout << "    " << label << ": " << error << "\n";
}

// ===========================================================================
// Test 1 -- shard round-trip is bit-exact
// ===========================================================================
void Test01_ShardRoundTripIsBitExact(const std::string& dir) {
  std::cout << "Test01 -- shard round-trip is bit-exact\n";

  // A realistic observation width: the information_state tensor is 5,580
  // floats, and a format that quietly capped or chunked a vector would show up
  // only at a realistic length.
  std::vector<SearchPiRow> rows;
  rows.push_back(MakeRow(3, 900001, 0, /*obs_len=*/5580, /*awkward=*/true));
  rows.push_back(MakeRow(3, 900001, 1, /*obs_len=*/5576, /*awkward=*/true));
  rows.push_back(MakeRow(3, 900002, 2, /*obs_len=*/17, /*awkward=*/false));
  // An empty observation is legal in the struct and must not be a special case
  // in the format; a length-prefixed vector of zero elements is the boundary
  // where an off-by-one in the writer would surface.
  SearchPiRow empty_obs = MakeRow(4, 900003, 3, /*obs_len=*/0, false);
  empty_obs.legal_actions.clear();
  empty_obs.raw_policy.clear();
  empty_obs.raw_visits.clear();
  empty_obs.target_visits.clear();
  empty_obs.target_probs.clear();
  empty_obs.q_values.clear();
  empty_obs.visits_pre_search.clear();
  empty_obs.state_fingerprint.clear();
  empty_obs.re_root_status.clear();
  empty_obs.search_fallback_reason.clear();
  rows.push_back(empty_obs);

  const std::string path = (std::filesystem::path(dir) / "roundtrip.bin").string();
  std::string error;
  SPIEL_CHECK_TRUE(WriteSearchPiRowShard(path, rows, &error));
  SPIEL_CHECK_TRUE(error.empty());

  std::vector<SearchPiRow> back;
  SPIEL_CHECK_TRUE(ReadSearchPiRowShard(path, &back, &error));
  SPIEL_CHECK_EQ(back.size(), rows.size());
  for (size_t i = 0; i < rows.size(); ++i) CheckRowsBitwiseEqual(rows[i], back[i]);

  // The awkward payload really did survive, checked directly rather than only
  // through the row comparison: this is the assertion that a text format fails.
  const std::vector<float>& awk = AwkwardFloats();
  for (size_t i = 0; i < awk.size(); ++i) {
    SPIEL_CHECK_EQ(FloatBits(back[0].observation[i]), FloatBits(awk[i]));
  }
  // -0.0f specifically: numerically equal to +0.0f, bitwise not.
  SPIEL_CHECK_EQ(FloatBits(back[0].observation[2]), FloatBits(-0.0f));
  SPIEL_CHECK_NE(FloatBits(back[0].observation[2]), FloatBits(0.0f));
  // The NaN survived as a NaN with its exact payload, which `==` cannot say.
  SPIEL_CHECK_EQ(FloatBits(back[0].observation[12]),
                 FloatBits(std::numeric_limits<float>::quiet_NaN()));

  // A zero-row shard is the degenerate cohort and must round-trip to empty
  // rather than to a read error.
  const std::string empty_path =
      (std::filesystem::path(dir) / "empty.bin").string();
  SPIEL_CHECK_TRUE(
      WriteSearchPiRowShard(empty_path, std::vector<SearchPiRow>(), &error));
  std::vector<SearchPiRow> none;
  SPIEL_CHECK_TRUE(ReadSearchPiRowShard(empty_path, &none, &error));
  SPIEL_CHECK_TRUE(none.empty());

  // Byte-for-byte determinism of the writer itself: the same rows must produce
  // the same shard, or two arms could not be compared on a shard hash.
  const std::string path2 = (std::filesystem::path(dir) / "roundtrip2.bin").string();
  SPIEL_CHECK_TRUE(WriteSearchPiRowShard(path2, rows, &error));
  SPIEL_CHECK_EQ(ReadWholeFile(path), ReadWholeFile(path2));

  std::cout << "  " << rows.size() << " rows, " << ReadWholeFile(path).size()
            << " bytes, every field bit-identical after read-back\n";
  std::cout << "Test01 Passed!\n\n";
}

// ===========================================================================
// Test 2 -- a damaged shard fails cleanly instead of crashing
// ===========================================================================
void Test02_CorruptShardFailsCleanly(const std::string& dir) {
  std::cout << "Test02 -- damaged shards fail through *error, never by crash\n";

  // One row with an 8-float observation, so the byte offsets below are exact:
  //   0  magic   4  version   8  row_count   16  observation length
  //   24 observation payload (8 floats)      56  first bool byte
  std::vector<SearchPiRow> rows;
  rows.push_back(MakeRow(1, 5, 0, /*obs_len=*/8, /*awkward=*/false));
  const std::string good = (std::filesystem::path(dir) / "good.bin").string();
  std::string error;
  SPIEL_CHECK_TRUE(WriteSearchPiRowShard(good, rows, &error));
  const std::string bytes = ReadWholeFile(good);
  // The offsets used below only mean what the comment says if the row really is
  // large: the fixture row serializes to several hundred bytes.
  SPIEL_CHECK_GT(bytes.size(), 300u);

  auto variant = [&](const char* name, const std::string& content) {
    const std::string p = (std::filesystem::path(dir) / name).string();
    WriteWholeFile(p, content);
    return p;
  };

  ExpectShardReadFails((std::filesystem::path(dir) / "no_such_shard.bin").string(),
                       "missing file");
  ExpectShardReadFails(variant("c_empty.bin", std::string()), "zero-length file");
  ExpectShardReadFails(variant("c_stub.bin", bytes.substr(0, 10)),
                       "shorter than the header");

  {
    std::string b = bytes;
    PatchU32(&b, 0, 0xDEADBEEFu);
    ExpectShardReadFails(variant("c_magic.bin", b), "wrong magic");
  }
  {
    std::string b = bytes;
    PatchU32(&b, 4, 99u);
    ExpectShardReadFails(variant("c_version.bin", b), "wrong version");
  }
  {
    // A header claiming far more rows than the file can hold. The read must
    // refuse on arithmetic BEFORE allocating per-row storage.
    std::string b = bytes;
    PatchU64(&b, 8, 1ull << 40);
    ExpectShardReadFails(variant("c_rowcount_high.bin", b),
                         "row count larger than the file");
  }
  {
    // A row count corrupted DOWNWARD parses fewer rows than the file holds and
    // would otherwise be accepted as a short-but-valid cohort. The trailer is
    // what turns it into a failure.
    std::string b = bytes;
    PatchU64(&b, 8, 0);
    ExpectShardReadFails(variant("c_rowcount_low.bin", b),
                         "row count zeroed in the header");
  }
  {
    // The allocation guard: an observation length of ~2^64 must be rejected by
    // comparison against the bytes remaining, not attempted.
    std::string b = bytes;
    PatchU64(&b, 16, 0xFFFFFFFFFFFFFF00ull);
    ExpectShardReadFails(variant("c_veclen.bin", b),
                         "vector length beyond the file");
  }
  {
    // A bool byte that is neither 0 nor 1 means the stream has drifted out of
    // alignment; parsing on from there would manufacture plausible rows.
    std::string b = bytes;
    b[kSearchPiShardHeaderBytes + 40] = static_cast<char>(7);
    ExpectShardReadFails(variant("c_bool.bin", b), "non-boolean bool byte");
  }
  // Cutting 60 bytes removes the 12-byte trailer and lands 48 bytes inside the
  // row payload, so the header's row count still passes the size floor and the
  // failure is a genuine mid-row exhaustion rather than an early header reject.
  ExpectShardReadFails(variant("c_midrow.bin", bytes.substr(0, bytes.size() - 60)),
                       "truncated mid-row");
  ExpectShardReadFails(
      variant("c_no_trailer.bin", bytes.substr(0, bytes.size() - 12)),
      "truncated at the row boundary, trailer gone");
  {
    std::string b = bytes;
    PatchU64(&b, b.size() - 8, 77);
    ExpectShardReadFails(variant("c_trailer_count.bin", b),
                         "trailer count disagrees with the header");
  }
  ExpectShardReadFails(variant("c_tail_junk.bin", bytes + std::string(5, 'X')),
                       "trailing bytes after the trailer");

  // A null `error` must be tolerated on the failure path too -- a caller that
  // does not want the text must not be the caller that segfaults.
  std::vector<SearchPiRow> sink;
  SPIEL_CHECK_FALSE(ReadSearchPiRowShard(
      (std::filesystem::path(dir) / "c_magic.bin").string(), &sink, nullptr));
  SPIEL_CHECK_TRUE(sink.empty());

  // ...and the undamaged fixture still reads, so the cases above failed for
  // their damage and not because the fixture was broken to begin with.
  std::vector<SearchPiRow> ok_rows;
  SPIEL_CHECK_TRUE(ReadSearchPiRowShard(good, &ok_rows, &error));
  SPIEL_CHECK_EQ(ok_rows.size(), 1u);
  CheckRowsBitwiseEqual(rows[0], ok_rows[0]);

  std::cout << "Test02 Passed!\n\n";
}

// ===========================================================================
// Test 3 -- sampler determinism
// ===========================================================================
void Test03_SamplerDeterminism() {
  std::cout << "Test03 -- same seed reproduces a sample; a different seed does"
               " not\n";

  std::vector<std::vector<SearchPiRow>> window;
  for (int g = 0; g < 8; ++g) window.push_back(MakeCohort(g, 200));
  const size_t total = 8 * 200;
  const size_t sample = 400;

  const std::vector<SearchPiRow> a = SampleUniformReplayWindow(window, sample, 12345);
  const std::vector<SearchPiRow> b = SampleUniformReplayWindow(window, sample, 12345);
  const std::vector<SearchPiRow> c = SampleUniformReplayWindow(window, sample, 999);

  SPIEL_CHECK_EQ(a.size(), sample);
  SPIEL_CHECK_EQ(b.size(), sample);
  SPIEL_CHECK_EQ(c.size(), sample);

  // Same seed: identical sequence, and identical down to the payload bits, not
  // merely to the identity fields.
  const std::vector<RowKey> ka = KeysOf(a);
  const std::vector<RowKey> kb = KeysOf(b);
  const std::vector<RowKey> kc = KeysOf(c);
  SPIEL_CHECK_TRUE(ka == kb);
  for (size_t i = 0; i < a.size(); ++i) CheckRowsBitwiseEqual(a[i], b[i]);

  // Different seed: a different sequence. Choosing 400 of 1,600 leaves a
  // collision probability of 1/C(1600,400), which is far past astronomical, so
  // this is a hard assertion and not a flaky one.
  SPIEL_CHECK_FALSE(ka == kc);

  // The documented canonical order: strictly increasing in the flattened index,
  // i.e. cohort-major and in-cohort ascending. This pins "deterministic order"
  // to an actual order rather than to whatever the draw happened to produce.
  for (size_t i = 1; i < ka.size(); ++i) {
    SPIEL_CHECK_TRUE(ka[i - 1] < ka[i]);
  }

  // Degenerate boundary: sample_count == 0 and sample_count == total.
  SPIEL_CHECK_TRUE(SampleUniformReplayWindow(window, 0, 7).empty());
  const std::vector<SearchPiRow> everything =
      SampleUniformReplayWindow(window, total, 7);
  SPIEL_CHECK_EQ(everything.size(), total);

  std::cout << "  seed 12345 reproduced " << sample << " of " << total
            << " rows exactly; seed 999 drew a different set\n";
  std::cout << "Test03 Passed!\n\n";
}

// ===========================================================================
// Test 4 -- uniformity across cohorts (the recency-bias detector)
// ===========================================================================
void Test04_SamplerUniformityAcrossCohorts() {
  std::cout << "Test04 -- every cohort in the window contributes its share\n";

  constexpr int kCohorts = 8;
  constexpr int kRowsPerCohort = 500;
  constexpr size_t kSample = 2000;
  std::vector<std::vector<SearchPiRow>> window;
  for (int g = 0; g < kCohorts; ++g) window.push_back(MakeCohort(g, kRowsPerCohort));

  const std::vector<SearchPiRow> drawn =
      SampleUniformReplayWindow(window, kSample, 20260819);
  SPIEL_CHECK_EQ(drawn.size(), kSample);

  std::vector<int> per_cohort(kCohorts, 0);
  for (const SearchPiRow& r : drawn) {
    SPIEL_CHECK_GE(r.generation, 0);
    SPIEL_CHECK_LT(r.generation, kCohorts);
    ++per_cohort[r.generation];
  }

  // Expected 250 per cohort. The draw is hypergeometric with sd ~10.5, so a
  // +/-25% band sits about six sd out: wide enough never to flake, tight enough
  // that a recency-weighted sampler cannot slip through. The oldest cohort of a
  // sampler that favoured the newest would come in far under 187.
  const int expected = static_cast<int>(kSample) / kCohorts;
  const int lo = expected - expected / 4;
  const int hi = expected + expected / 4;
  for (int g = 0; g < kCohorts; ++g) {
    SPIEL_CHECK_GE(per_cohort[g], lo);
    SPIEL_CHECK_LE(per_cohort[g], hi);
  }

  // The directional statement the band alone does not make: the OLDEST cohort
  // must not be systematically starved relative to the NEWEST one.
  const int oldest = per_cohort[0];
  const int newest = per_cohort[kCohorts - 1];
  SPIEL_CHECK_LT(std::abs(oldest - newest), expected / 2);

  std::cout << "  per-cohort counts:";
  for (int g = 0; g < kCohorts; ++g) std::cout << " " << per_cohort[g];
  std::cout << " (expected " << expected << ", band [" << lo << ", " << hi
            << "])\n";
  std::cout << "Test04 Passed!\n\n";
}

// ===========================================================================
// Test 5 -- sampling is without replacement
// ===========================================================================
void Test05_SamplingWithoutReplacement() {
  std::cout << "Test05 -- no row is drawn twice in one sample\n";

  std::vector<std::vector<SearchPiRow>> window;
  for (int g = 0; g < 8; ++g) window.push_back(MakeCohort(g, 300));
  constexpr size_t kSample = 900;

  const std::vector<SearchPiRow> drawn =
      SampleUniformReplayWindow(window, kSample, 4242);
  SPIEL_CHECK_EQ(drawn.size(), kSample);

  std::set<std::tuple<int, int64_t, int>> identities;
  std::set<std::string> fingerprints;
  for (const SearchPiRow& r : drawn) {
    identities.insert(
        std::make_tuple(r.generation, r.episode_id, r.decision_id));
    fingerprints.insert(r.state_fingerprint);
  }
  SPIEL_CHECK_EQ(identities.size(), kSample);
  SPIEL_CHECK_EQ(fingerprints.size(), kSample);

  // The hardest case for a partial-shuffle bug: drawing the ENTIRE window must
  // still yield every row exactly once, with nothing repeated and nothing lost.
  const size_t total = 8 * 300;
  const std::vector<SearchPiRow> all =
      SampleUniformReplayWindow(window, total, 11);
  std::set<std::tuple<int, int64_t, int>> all_ids;
  for (const SearchPiRow& r : all) {
    all_ids.insert(std::make_tuple(r.generation, r.episode_id, r.decision_id));
  }
  SPIEL_CHECK_EQ(all.size(), total);
  SPIEL_CHECK_EQ(all_ids.size(), total);

  std::cout << "  " << kSample << " distinct identities out of " << kSample
            << " drawn; full-window draw covered all " << total << "\n";
  std::cout << "Test05 Passed!\n\n";
}

// ===========================================================================
// Test 6 -- retention keeps the most recent 8 and reports the rest
// ===========================================================================
void Test06_RetentionWindow(const std::string& parent) {
  std::cout << "Test06 -- retention keeps the newest 8 cohorts and reports the"
               " rest\n";

  const std::string dir = (std::filesystem::path(parent) / "cohorts").string();
  std::filesystem::create_directories(dir);

  // The canonical name and its parser must agree in both directions.
  const std::string p7 = SearchPiShardPathForGeneration(dir, 7);
  int parsed = -1;
  SPIEL_CHECK_TRUE(ParseSearchPiShardGeneration(p7, &parsed));
  SPIEL_CHECK_EQ(parsed, 7);
  SPIEL_CHECK_FALSE(ParseSearchPiShardGeneration(
      (std::filesystem::path(dir) / "manifest.json").string(), &parsed));
  SPIEL_CHECK_FALSE(ParseSearchPiShardGeneration(
      (std::filesystem::path(dir) / "searchpi_cohort_00000x.bin").string(),
      &parsed));
  // Non-canonical digit runs that would otherwise ALIAS onto generation 7 and
  // make the retention order depend on directory order.
  SPIEL_CHECK_FALSE(ParseSearchPiShardGeneration(
      (std::filesystem::path(dir) / "searchpi_cohort_7.bin").string(), &parsed));
  SPIEL_CHECK_FALSE(ParseSearchPiShardGeneration(
      (std::filesystem::path(dir) / "searchpi_cohort_0000007.bin").string(),
      &parsed));
  // A shard mid-write must never look like a cohort.
  SPIEL_CHECK_FALSE(ParseSearchPiShardGeneration(
      (std::filesystem::path(dir) / "searchpi_cohort_000007.bin.tmp.99").string(),
      &parsed));

  std::string error;
  auto write_generation = [&](int g) {
    const std::string p = SearchPiShardPathForGeneration(dir, g);
    SPIEL_CHECK_TRUE(
        WriteSearchPiRowShard(p, std::vector<SearchPiRow>{MakeRow(g, g, 0, 4, false)},
                              &error));
    return p;
  };

  // Under-full window first: fewer cohorts than the limit evicts nothing.
  for (int g = 1; g <= 3; ++g) write_generation(g);
  {
    const std::string next = write_generation(4);
    const SearchPiReplayRetention r = PlanSearchPiReplayRetention(dir, next, 8);
    SPIEL_CHECK_TRUE(r.ok);
    SPIEL_CHECK_EQ(r.retained.size(), 4u);
    SPIEL_CHECK_TRUE(r.evicted.empty());
  }

  // Fill past the limit: generations 1..10 on disk, then append 11.
  for (int g = 5; g <= 10; ++g) write_generation(g);
  // Files that are not canonical shards must be invisible to the scan: a
  // manifest beside the shards, and a temp file from a writer that died.
  WriteWholeFile((std::filesystem::path(dir) / "manifest.json").string(), "{}");
  WriteWholeFile(
      (std::filesystem::path(dir) / "searchpi_cohort_000011.bin.tmp.4242").string(),
      "partial");

  const std::string newest = write_generation(11);
  const SearchPiReplayRetention plan =
      PlanSearchPiReplayRetention(dir, newest, kSearchPiReplayWindowCohorts);
  SPIEL_CHECK_TRUE(plan.ok);
  SPIEL_CHECK_TRUE(plan.error.empty());

  // EXACTLY the newest eight, in cohort order oldest-first.
  SPIEL_CHECK_EQ(plan.retained.size(), kSearchPiReplayWindowCohorts);
  for (size_t i = 0; i < plan.retained.size(); ++i) {
    const int expected_generation = static_cast<int>(i) + 4;
    SPIEL_CHECK_EQ(plan.retained[i],
                   SearchPiShardPathForGeneration(dir, expected_generation));
  }
  // ...and EXACTLY the three that fell out, also oldest-first. Reported, not
  // deleted: every one of them is still on disk afterwards.
  SPIEL_CHECK_EQ(plan.evicted.size(), 3u);
  for (size_t i = 0; i < plan.evicted.size(); ++i) {
    const int expected_generation = static_cast<int>(i) + 1;
    SPIEL_CHECK_EQ(plan.evicted[i],
                   SearchPiShardPathForGeneration(dir, expected_generation));
    SPIEL_CHECK_TRUE(std::filesystem::exists(plan.evicted[i]));
  }

  // The retained shards are readable, which is what makes them a usable window
  // rather than a list of names.
  std::vector<std::vector<SearchPiRow>> window;
  for (const std::string& p : plan.retained) {
    std::vector<SearchPiRow> cohort;
    SPIEL_CHECK_TRUE(ReadSearchPiRowShard(p, &cohort, &error));
    SPIEL_CHECK_EQ(cohort.size(), 1u);
    window.push_back(std::move(cohort));
  }
  SPIEL_CHECK_EQ(SampleUniformReplayWindow(window, 8, 5).size(), 8u);

  // A non-canonical "new" path is a caller error and is reported, not guessed.
  const SearchPiReplayRetention bad_name =
      PlanSearchPiReplayRetention(dir, (std::filesystem::path(dir) / "x.bin").string(), 8);
  SPIEL_CHECK_FALSE(bad_name.ok);
  SPIEL_CHECK_FALSE(bad_name.error.empty());

  // A missing directory is reported the same way.
  const SearchPiReplayRetention bad_dir = PlanSearchPiReplayRetention(
      (std::filesystem::path(parent) / "not_a_dir").string(),
      SearchPiShardPathForGeneration("/nowhere", 1), 8);
  SPIEL_CHECK_FALSE(bad_dir.ok);
  SPIEL_CHECK_FALSE(bad_dir.error.empty());

  std::cout << "  retained generations 4..11, evicted 1..3, all 11 files still"
               " present on disk\n";
  std::cout << "Test06 Passed!\n\n";
}

std::string MakeTempDir() {
  const std::string dir = "/tmp/dune_searchpi_replay_test_" +
                          std::to_string(static_cast<long long>(::getpid()));
  std::error_code ec;
  std::filesystem::remove_all(dir, ec);
  std::filesystem::create_directories(dir, ec);
  SPIEL_CHECK_FALSE(static_cast<bool>(ec));
  return dir;
}

}  // namespace
}  // namespace open_spiel

int main(int argc, char** argv) {
  using namespace open_spiel;
  const std::string dir = MakeTempDir();
  std::cout << "temp directory: " << dir << "\n\n";

  Test01_ShardRoundTripIsBitExact(dir);
  Test02_CorruptShardFailsCleanly(dir);
  Test03_SamplerDeterminism();
  Test04_SamplerUniformityAcrossCohorts();
  Test05_SamplingWithoutReplacement();
  Test06_RetentionWindow(dir);

  std::error_code ec;
  std::filesystem::remove_all(dir, ec);
  SPIEL_CHECK_FALSE(static_cast<bool>(ec));

  std::cout << "All dune search-PI replay tests passed!\n";
  return 0;
}
