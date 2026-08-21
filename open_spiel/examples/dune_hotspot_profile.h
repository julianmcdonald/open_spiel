#ifndef OPEN_SPIEL_EXAMPLES_DUNE_HOTSPOT_PROFILE_H_
#define OPEN_SPIEL_EXAMPLES_DUNE_HOTSPOT_PROFILE_H_

#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>

namespace open_spiel::dune_hotspot {

enum class Kind : int {
  kStateClone = 0,
  kLegalActions = 1,
  kGameTransition = 2,
  kObservationConstruction = 3,
  kMctsSearch = 4,
  kBatcherResultDistribution = 5,
  kCount = 6,
};

struct Snapshot {
  std::array<uint64_t, static_cast<size_t>(Kind::kCount)> nanoseconds{};
  std::array<uint64_t, static_cast<size_t>(Kind::kCount)> calls{};
};

inline std::atomic<bool> enabled{false};
constexpr size_t kMaxShards = 1024;
struct Shard {
  std::array<uint64_t, static_cast<size_t>(Kind::kCount)> nanoseconds{};
  std::array<uint64_t, static_cast<size_t>(Kind::kCount)> calls{};
};
inline std::array<Shard, kMaxShards> shards{};
inline std::atomic<size_t> next_shard{0};
inline thread_local size_t local_shard_index = kMaxShards;

inline Shard& LocalShard() {
  if (local_shard_index == kMaxShards) {
    local_shard_index = next_shard.fetch_add(1, std::memory_order_relaxed);
    if (local_shard_index >= kMaxShards) local_shard_index = kMaxShards - 1;
  }
  return shards[local_shard_index];
}

inline void Reset() {
  enabled.store(false, std::memory_order_release);
  next_shard.store(0, std::memory_order_relaxed);
  for (Shard& shard : shards) shard = Shard{};
  local_shard_index = kMaxShards;
  enabled.store(true, std::memory_order_release);
}

inline Snapshot Read() {
  Snapshot snapshot;
  for (const Shard& shard : shards) {
    for (size_t i = 0; i < static_cast<size_t>(Kind::kCount); ++i) {
      snapshot.nanoseconds[i] += shard.nanoseconds[i];
      snapshot.calls[i] += shard.calls[i];
    }
  }
  return snapshot;
}

class ScopedTimer {
 public:
  explicit ScopedTimer(Kind kind)
      : kind_(kind), active_(enabled.load(std::memory_order_relaxed)),
        shard_(active_ ? &LocalShard() : nullptr),
        start_(active_ ? std::chrono::steady_clock::now()
                       : std::chrono::steady_clock::time_point{}) {}

  ~ScopedTimer() {
    if (!active_) return;
    const size_t index = static_cast<size_t>(kind_);
    const uint64_t elapsed = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - start_)
            .count());
    shard_->nanoseconds[index] += elapsed;
    shard_->calls[index] += 1;
  }

 private:
  Kind kind_;
  bool active_;
  Shard* shard_;
  std::chrono::steady_clock::time_point start_;
};

}  // namespace open_spiel::dune_hotspot

#endif  // OPEN_SPIEL_EXAMPLES_DUNE_HOTSPOT_PROFILE_H_
