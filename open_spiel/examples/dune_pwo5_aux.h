#ifndef OPEN_SPIEL_EXAMPLES_DUNE_PWO5_AUX_H_
#define OPEN_SPIEL_EXAMPLES_DUNE_PWO5_AUX_H_

// PWO-5 auxiliary-head data path: the target artifact, the whole-trajectory
// held-out split, and the hierarchical sampler.
//
// docs/PWO5_PILOT_REGISTRATION.md sections 4.4, 5.2, 8.5 and 8.6.
//
// Kept out of dune_ppo_train.cc so the sampler's exact algorithm is testable
// without a GPU, a checkpoint or the engine -- section 8.6 pins that algorithm
// bit for bit and the registration requires an update-1 sampler digest in every
// arm's manifest, so "reproducible in principle" is not enough.

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <map>
#include <random>
#include <string>
#include <vector>

#include "open_spiel/spiel.h"

namespace open_spiel {
namespace pwo5 {

// ---------------------------------------------------------------------------
// The exact without-replacement draw, pinned by section 8.6.
// ---------------------------------------------------------------------------
//
// "uniformly without replacement" admits many non-equivalent implementations
// and they do not agree bit for bit, so the registration pins ONE: a partial
// Fisher-Yates (selection sample) over a working copy of the canonically
// ordered array, returning the first k entries IN DRAW ORDER.
//
//   for i in 0 .. k-1:
//       j = i + (rng() % (N - i))
//       swap(A[i], A[j])
//   return A[0 .. k-1]
//
// TWO TRAPS, both registered:
//
//  1. `std::uniform_int_distribution` is FORBIDDEN here. Its mapping from
//     generator output to range is IMPLEMENTATION-DEFINED -- libstdc++ and
//     libc++ give different draws from the same seeded generator -- so a split
//     using it is not reproducible across toolchains and could not be verified
//     from a persisted digest. `rng() % (N - i)` is specified DELIBERATELY: the
//     modulo bias is bounded by (N-i)/2^64 < 2e-17 for N <= 340 and is
//     ACCEPTED, because the property being bought is exact reproducibility, not
//     perfect uniformity.
//
//  2. Draw ORDER is significant -- the batch partition reads it -- so the
//     result is the first k of the working copy, not a sorted set.
template <typename T>
inline std::vector<T> SampleWithoutReplacement(std::vector<T> a, int k,
                                               std::mt19937_64* rng) {
  const int n = static_cast<int>(a.size());
  if (k > n) k = n;
  for (int i = 0; i < k; ++i) {
    const uint64_t span = static_cast<uint64_t>(n - i);
    const int j = i + static_cast<int>((*rng)() % span);
    std::swap(a[i], a[j]);
  }
  a.resize(k);
  return a;
}

// ---------------------------------------------------------------------------
// The target artifact, as written by dune_pwo5_prepare.
// ---------------------------------------------------------------------------
struct AuxTargetHeader {
  uint32_t magic;
  uint32_t schema;
  uint32_t obs_size;
  uint32_t action_dim;
  uint64_t num_rows;
  uint64_t reserved;
};
inline constexpr uint32_t kAuxTargetMagic = 0x50573541u;
inline constexpr uint32_t kAuxTargetSchema = 1u;

struct AuxTargetRow {
  int32_t game_index = 0;
  int32_t decision_index = 0;
  int32_t acting_player = 0;
  int32_t terminal_round_class = 0;  // 0 = <=8, 1 = 9, 2 = 10
  float final_vp_target = 0.0f;      // FinalScoredVp(acting_player) / 20
  int32_t next_own_action = -1;      // -1 = no later action in this trajectory
  uint8_t weight = 0;
  uint8_t is_heldout = 0;
};

// One drawn row: an index into the store's flat observation buffer plus its
// targets.
struct AuxSample {
  int64_t row;        // index into AuxTargetStore::rows()
  int32_t game_index; // carried so the within-game grouping is explicit
};

// A per-update draw: `batches` batches, each a list of games in draw order,
// each game carrying exactly `rows_per_game` row indices.
struct AuxDraw {
  // batch -> game -> rows
  std::vector<std::vector<std::vector<int64_t>>> batches;
  std::vector<int32_t> drawn_games;  // all G games, in draw order
  std::string digest;               // canonical digest of the realized draw
};

class AuxTargetStore {
 public:
  // Loads the artifact and applies the held-out split. Returns false and fills
  // `error` on any structural problem -- the caller makes it fatal, because
  // section 8.2 registers a missing target as a STOP at gate 3, never a
  // training-time skip.
  bool Load(const std::string& path,
            const std::vector<int32_t>& heldout_games,
            std::string* error) {
    std::ifstream in(path, std::ios::binary);
    if (!in) { *error = "cannot open aux target artifact: " + path; return false; }
    AuxTargetHeader h{};
    in.read(reinterpret_cast<char*>(&h.magic), 4);
    in.read(reinterpret_cast<char*>(&h.schema), 4);
    in.read(reinterpret_cast<char*>(&h.obs_size), 4);
    in.read(reinterpret_cast<char*>(&h.action_dim), 4);
    in.read(reinterpret_cast<char*>(&h.num_rows), 8);
    in.read(reinterpret_cast<char*>(&h.reserved), 8);
    if (!in) { *error = "truncated aux target header"; return false; }
    if (h.magic != kAuxTargetMagic) {
      *error = "aux target magic mismatch";
      return false;
    }
    if (h.schema != kAuxTargetSchema) {
      *error = "aux target schema " + std::to_string(h.schema) + " != 1";
      return false;
    }
    obs_size_ = h.obs_size;
    action_dim_ = h.action_dim;

    std::vector<int32_t> heldout_sorted = heldout_games;
    std::sort(heldout_sorted.begin(), heldout_sorted.end());

    rows_.reserve(h.num_rows);
    obs_.reserve(static_cast<size_t>(h.num_rows) * obs_size_);
    std::vector<float> scratch(obs_size_);
    for (uint64_t i = 0; i < h.num_rows; ++i) {
      AuxTargetRow r;
      uint8_t pad[2];
      in.read(reinterpret_cast<char*>(&r.game_index), 4);
      in.read(reinterpret_cast<char*>(&r.decision_index), 4);
      in.read(reinterpret_cast<char*>(&r.acting_player), 4);
      in.read(reinterpret_cast<char*>(&r.terminal_round_class), 4);
      in.read(reinterpret_cast<char*>(&r.final_vp_target), 4);
      in.read(reinterpret_cast<char*>(&r.next_own_action), 4);
      in.read(reinterpret_cast<char*>(&r.weight), 1);
      in.read(reinterpret_cast<char*>(&r.is_heldout), 1);
      in.read(reinterpret_cast<char*>(pad), 2);
      in.read(reinterpret_cast<char*>(scratch.data()),
              static_cast<std::streamsize>(obs_size_) * 4);
      if (!in) {
        *error = "truncated aux target row " + std::to_string(i);
        return false;
      }
      if (r.terminal_round_class < 0 || r.terminal_round_class > 2) {
        *error = "aux target row " + std::to_string(i) +
                 " has terminal_round_class " +
                 std::to_string(r.terminal_round_class);
        return false;
      }
      // The artifact's own is_heldout must agree with the membership list the
      // arm was launched with. Disagreement means the two were produced from
      // different splits, which section 8.5 forbids across arms and across the
      // u600 extension.
      const bool in_list = std::binary_search(heldout_sorted.begin(),
                                              heldout_sorted.end(), r.game_index);
      if (in_list != (r.is_heldout != 0)) {
        *error = "aux target row " + std::to_string(i) + " (game " +
                 std::to_string(r.game_index) +
                 ") disagrees with the held-out membership list";
        return false;
      }
      const int64_t row_index = static_cast<int64_t>(rows_.size());
      rows_.push_back(r);
      obs_.insert(obs_.end(), scratch.begin(), scratch.end());
      if (!in_list) {
        game_rows_[r.game_index].push_back(row_index);
      } else {
        // Held-out rows are grouped SEPARATELY and are never reachable from
        // game_rows_, so the sampler cannot draw one. They exist here only so
        // the update-300 evaluator can compute the registered
        // whole-trajectory-HELD-OUT loss, which is a no-gradient measurement.
        heldout_game_rows_[r.game_index].push_back(row_index);
      }
    }

    // Canonical ordering, section 8.6: games ascending by game_index, rows
    // ascending by decision_index. The artifact is written in that order, but
    // the sampler's reproducibility must not depend on the writer, so it is
    // established here rather than assumed.
    training_games_.reserve(game_rows_.size());
    for (auto& kv : game_rows_) {
      std::sort(kv.second.begin(), kv.second.end(),
                [this](int64_t a, int64_t b) {
                  return rows_[a].decision_index < rows_[b].decision_index;
                });
      training_games_.push_back(kv.first);
    }
    std::sort(training_games_.begin(), training_games_.end());
    heldout_games_.reserve(heldout_game_rows_.size());
    for (auto& kv : heldout_game_rows_) {
      std::sort(kv.second.begin(), kv.second.end(),
                [this](int64_t a, int64_t b) {
                  return rows_[a].decision_index < rows_[b].decision_index;
                });
      heldout_games_.push_back(kv.first);
    }
    std::sort(heldout_games_.begin(), heldout_games_.end());
    return true;
  }

  int64_t obs_size() const { return obs_size_; }
  int64_t action_dim() const { return action_dim_; }
  const std::vector<AuxTargetRow>& rows() const { return rows_; }
  const float* observation(int64_t row) const {
    return obs_.data() + static_cast<size_t>(row) * obs_size_;
  }
  const std::vector<int32_t>& training_games() const { return training_games_; }
  size_t training_row_count() const {
    size_t n = 0;
    for (const auto& kv : game_rows_) n += kv.second.size();
    return n;
  }
  // The two whole-trajectory splits, as (game -> canonically ordered rows).
  // Used by the update-300 evaluator, which reports EVERY row of both, and by
  // nothing that computes a gradient.
  const std::map<int32_t, std::vector<int64_t>>& training_game_rows() const {
    return game_rows_;
  }
  const std::map<int32_t, std::vector<int64_t>>& heldout_game_rows() const {
    return heldout_game_rows_;
  }
  const std::vector<int32_t>& heldout_games() const { return heldout_games_; }
  size_t heldout_row_count() const {
    size_t n = 0;
    for (const auto& kv : heldout_game_rows_) n += kv.second.size();
    return n;
  }
  int MinRowsPerTrainingGame() const {
    int m = -1;
    for (const auto& kv : game_rows_) {
      const int n = static_cast<int>(kv.second.size());
      if (m < 0 || n < m) m = n;
    }
    return m;
  }

  // Section 8.6's draw: G games uniformly without replacement ONCE per update,
  // then R rows from each in draw order, then partition the G games into
  // `batches` equal blocks IN DRAW ORDER.
  //
  // Drawing once and partitioning -- rather than drawing each batch
  // independently -- is what makes "1,024 distinct rows" TRUE. Two independent
  // 32-game draws give E[games in both] = 340*(32/340)^2 = 3.012 and
  // E[duplicated rows] = 6.532, i.e. ~1,017.47 distinct rows, not 1,024.
  AuxDraw Draw(uint64_t seed, int games_per_update, int rows_per_game,
               int batches) const {
    std::mt19937_64 rng(seed);
    AuxDraw out;
    out.drawn_games = SampleWithoutReplacement(training_games_,
                                               games_per_update, &rng);
    std::vector<std::vector<int64_t>> per_game;
    per_game.reserve(out.drawn_games.size());
    for (int32_t g : out.drawn_games) {
      const auto it = game_rows_.find(g);
      per_game.push_back(
          SampleWithoutReplacement(it->second, rows_per_game, &rng));
    }
    const int games_per_batch =
        batches > 0 ? static_cast<int>(out.drawn_games.size()) / batches : 0;
    out.batches.resize(batches);
    for (int b = 0; b < batches; ++b) {
      for (int i = b * games_per_batch; i < (b + 1) * games_per_batch; ++i) {
        out.batches[b].push_back(per_game[i]);
      }
    }
    out.digest = DigestOfDraw(out);
    return out;
  }

  // A canonical, order-sensitive serialization of the realized draw, so two
  // arms sharing a triplet seed can be checked to have drawn identically
  // without persisting 300 updates of row lists. Order-SENSITIVE on purpose:
  // the draw order determines the batch partition, so two draws with the same
  // membership but different order are different draws.
  std::string DigestOfDraw(const AuxDraw& draw) const {
    std::string s;
    char buf[64];
    for (size_t b = 0; b < draw.batches.size(); ++b) {
      for (const auto& game_rows : draw.batches[b]) {
        for (int64_t r : game_rows) {
          std::snprintf(buf, sizeof(buf), "%d:%d;", rows_[r].game_index,
                        rows_[r].decision_index);
          s += buf;
        }
      }
      s += "|";
    }
    return s;
  }

 private:
  int64_t obs_size_ = 0;
  int64_t action_dim_ = 0;
  std::vector<AuxTargetRow> rows_;
  std::vector<float> obs_;
  // Only NON-held-out games appear here: section 8.5 excludes held-out games
  // from ALL offline losses, so a held-out row can never be drawn.
  std::map<int32_t, std::vector<int64_t>> game_rows_;
  std::vector<int32_t> training_games_;
  // Held-out games, kept strictly apart from game_rows_ so no draw can reach
  // them. Read only by the update-300 evaluator.
  std::map<int32_t, std::vector<int64_t>> heldout_game_rows_;
  std::vector<int32_t> heldout_games_;
};

}  // namespace pwo5
}  // namespace open_spiel

#endif  // OPEN_SPIEL_EXAMPLES_DUNE_PWO5_AUX_H_
