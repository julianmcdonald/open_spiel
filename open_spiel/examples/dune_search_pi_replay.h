// Persistent training-row store and uniform replay-window sampler for the
// two-arm search-PI replay experiment.
//
// WHY THIS EXISTS. Each generation collects ONE shared 512-game target set
// against a FROZEN teacher, and the outcome-blind first-64-game training prefix
// (the "cohort") is handed to two learners:
//
//   Arm F -- trains fresh-only, on exactly this generation's cohort.
//   Arm W -- trains on exactly the SAME NUMBER of rows, sampled uniformly from
//            a recent replay window of the last 8 cohorts.
//
// The equal row count is the whole point of the design. If arm W trained on the
// union of eight cohorts it would take eight times the SGD steps per generation,
// and any difference between the arms would confound "replay" with "more
// optimisation" -- an uninterpretable contrast. Holding the row count fixed
// means the ONLY thing that varies is where the rows came from.
//
// Both arms call RunSearchPiLearner(model, optimizer, rows, ...) unchanged: it
// consumes a plain std::vector<SearchPiRow>, so neither arm needs a learner
// variant and neither arm can accidentally diverge in the objective. This
// header adds only (1) a way to make a cohort outlive its process and (2) a way
// to draw from several cohorts without a recency bias.
//
// WHY A BINARY STORE AND NOT JSON. A SearchPiRow carries a
// std::vector<float> observation of ~5580 elements and a 64-game cohort is
// thousands of rows, so a cohort is order 10^8 bytes of float payload. JSON
// costs roughly 10x that in bytes, and -- decisively -- OpenSpiel's json.cc
// floors any float under 5e-7 to exactly 0.0 (see the note on the distribution
// scalars in dune_search_pi.h). A replay store that silently rounded its rows
// would make arm W train on different numbers than arm F for a reason that has
// nothing to do with replay. This format writes raw IEEE-754 bytes: the
// round-trip is bit-exact, including denormals, signed zero and NaN.

#ifndef OPEN_SPIEL_EXAMPLES_DUNE_SEARCH_PI_REPLAY_H_
#define OPEN_SPIEL_EXAMPLES_DUNE_SEARCH_PI_REPLAY_H_

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "dune_search_pi.h"  // SearchPiRow

namespace open_spiel {

// ---------------------------------------------------------------------------
// Shard format constants
// ---------------------------------------------------------------------------
//
// The header is a fixed 16 bytes, all little-endian, in this exact order:
//
//   offset  0  uint32  magic          kSearchPiShardMagic
//   offset  4  uint32  version        kSearchPiShardVersion
//   offset  8  uint64  row_count
//   offset 16  rows...
//   (tail)     uint32  trailer magic  kSearchPiShardTrailerMagic
//   (tail)     uint64  row_count echo
//
// The TRAILER is not redundant. A shard truncated at a row boundary parses
// every row it contains without complaint, and a reader that stopped at
// row_count would accept it as a short-but-valid file. The trailer turns that
// into a clean failure, and the echoed count catches a header whose row_count
// was corrupted downward. Trailing bytes after the trailer are also rejected:
// a shard is exactly its content and nothing else.
//
// The offsets above are part of the contract because the corruption test writes
// deliberate damage at known offsets; changing the layout means bumping the
// version and updating that test together.
inline constexpr uint32_t kSearchPiShardMagic = 0x52505344u;         // "DSPR"
inline constexpr uint32_t kSearchPiShardTrailerMagic = 0x444E4553u;  // "SEND"
inline constexpr uint32_t kSearchPiShardVersion = 1u;
inline constexpr uint32_t kScratchSearchPiShardVersion = 2u;
inline constexpr size_t kSearchPiShardHeaderBytes = 16;

// The retained replay window, in cohorts. Eight is the registered window for
// this experiment; the retention helper takes it as an argument anyway so a
// test can exercise the boundary without redefining the constant.
inline constexpr size_t kSearchPiReplayWindowCohorts = 8;

// ---------------------------------------------------------------------------
// Shard I/O
// ---------------------------------------------------------------------------

// Writes `rows` to `path` as one shard.
//
// ATOMIC. The bytes go to a sibling temp file, which is fsynced and closed
// before std::filesystem::rename() moves it into place. rename() within a
// filesystem is atomic, so a concurrent reader -- or a reader in the next
// process after a crash -- either sees the complete previous shard or the
// complete new one, never a half-written prefix. This matters because the
// retention window is scanned by directory listing: a partially written file
// with the final name would be picked up as a cohort.
//
// Returns false and sets `*error` (if non-null) on any failure; a failed write
// leaves no file at `path` and removes its temp file. `error` may be null.
bool WriteSearchPiRowShard(const std::string& path,
                           const std::vector<SearchPiRow>& rows,
                           std::string* error);

// Reads a shard written by WriteSearchPiRowShard.
//
// Verifies the magic, the version, the row count and the trailer, and bounds
// every length prefix against the bytes actually remaining in the file, so a
// corrupted length can neither over-read nor provoke an enormous allocation.
// Every failure is reported through the return value and `*error`; this
// function does not abort the process on a damaged file.
//
// On failure `*rows` is left EMPTY rather than partially filled: a partial
// cohort silently entering the replay window would be a data-quantity bug that
// no downstream assertion could see. `rows` must be non-null; `error` may be
// null.
bool ReadSearchPiRowShard(const std::string& path,
                          std::vector<SearchPiRow>* rows, std::string* error);

// Isolated scratch_q_v1 codec. The legacy version-1 functions above remain
// byte-for-byte on their original format and reject these version-2 shards.
bool WriteScratchSearchPiRowShardV2(const std::string& path,
                                    const std::vector<SearchPiRow>& rows,
                                    std::string* error);
bool ReadScratchSearchPiRowShardV2(const std::string& path,
                                   std::vector<SearchPiRow>* rows,
                                   std::string* error);

// ---------------------------------------------------------------------------
// Uniform replay-window sampler
// ---------------------------------------------------------------------------

// Draws `sample_count` rows uniformly WITHOUT REPLACEMENT from the FLATTENED
// union of every cohort in `window`.
//
// FLATTENED is the operative word. The draw is over rows, not over cohorts, so
// a row in the oldest cohort is exactly as likely to be drawn as a row in the
// newest one. That is what "uniform recent replay window" means here, and it is
// the property the experiment's contrast depends on: any recency weighting
// would make arm W a partially-fresh learner and blur the comparison against
// arm F. (Cohorts of unequal size therefore contribute in proportion to their
// row counts, which is the same statement.)
//
// DETERMINISM. Identical `window` contents plus identical `seed` produce an
// identical row sequence. The seed is domain-separated through the lane's own
// stream block before it reaches the generator, so a replay sample never
// correlates with the chance, opponent-policy, search or behaviour streams
// drawn from the same base seed. The bounded draw is implemented here rather
// than taken from std::uniform_int_distribution, whose output sequence is not
// specified across standard-library implementations.
//
// ORDER. Rows come back in canonical flattened order -- cohort-major, and in
// each cohort in its stored order -- not in draw order. The learner permutes
// its rows every epoch anyway, so draw order buys nothing, while a canonical
// order makes two samples directly comparable. One consequence to know: at
// sample_count == total the result is the entire window in canonical order for
// EVERY seed, because the sampled set is then the whole set.
//
// PRECONDITION, checked: `sample_count` must not exceed the total number of
// rows in `window`. The caller guarantees this -- for the experiment that is
// the statement that eight cohorts hold at least one cohort's worth of rows,
// which is trivially true once the window has filled, and before it has filled
// the caller must clamp the request to what the window actually holds.
std::vector<SearchPiRow> SampleUniformReplayWindow(
    const std::vector<std::vector<SearchPiRow>>& window, size_t sample_count,
    uint64_t seed);

// scratch_q_v1 replay selection: retain every current-generation full-policy
// row, then fill uniformly without replacement from every remaining row in the
// flattened recent window. Returns empty and sets error if required full rows
// alone exceed max_rows or if any row is not schema version 2.
std::vector<SearchPiRow> SampleScratchSearchPiReplayWindow(
    const std::vector<std::vector<SearchPiRow>>& window,
    int current_generation, size_t max_rows, uint64_t seed,
    std::string* error);

std::string ScratchSearchPiShardPathForGeneration(const std::string& dir,
                                                  int generation);

// ---------------------------------------------------------------------------
// Retention
// ---------------------------------------------------------------------------

// Canonical shard name for a generation: "searchpi_cohort_%06d.bin".
//
// The generation is zero-padded to six digits so a lexicographic listing and a
// numeric one agree, but the retention scan still sorts on the PARSED integer.
// Sorting on the name would be a latent off-by-one-hundred-thousand at
// generation 1,000,000, and sorting on mtime would be plainly wrong -- rewriting
// a shard would reorder the window.
std::string SearchPiShardPathForGeneration(const std::string& dir,
                                           int generation);

// Recovers the generation from a shard path. Returns false if the file name is
// not in canonical form, which is how the retention scan ignores unrelated
// files sitting in the same directory (manifests, logs, leftover temp files)
// instead of guessing at them.
//
// "Canonical" means EXACTLY what SearchPiShardPathForGeneration would emit, not
// merely "digits between the prefix and the suffix": the digit run must
// round-trip through the writer's own %06d. That is what makes the generation a
// key -- otherwise "…_7.bin" and "…_000007.bin" would both claim cohort 7 and
// the retention order between them would depend on directory order.
bool ParseSearchPiShardGeneration(const std::string& path, int* generation);

struct SearchPiReplayRetention {
  bool ok = false;
  std::string error;
  // Cohort order, oldest first. At most `max_window_cohorts` entries.
  std::vector<std::string> retained;
  // Everything that fell outside the window, also oldest first. REPORTED ONLY.
  std::vector<std::string> evicted;
};

// Lists the canonical shards already in `dir`, appends `new_shard_path`, orders
// them by generation, and splits them into the retained window and the rest.
//
// This function DELETES NOTHING. Reporting and deleting are separated on
// purpose: a retention bug that only miscounts is recoverable, whereas one that
// also unlinks has already destroyed a cohort that took a 512-game collection
// to produce. The caller decides what, if anything, happens to `evicted`.
//
// `new_shard_path` may already be present in `dir` (the normal case -- it was
// just written there). It is matched by generation, not by string, and the
// caller-supplied path wins for that generation.
SearchPiReplayRetention PlanSearchPiReplayRetention(
    const std::string& dir, const std::string& new_shard_path,
    size_t max_window_cohorts = kSearchPiReplayWindowCohorts);

}  // namespace open_spiel

#endif  // OPEN_SPIEL_EXAMPLES_DUNE_SEARCH_PI_REPLAY_H_
