// Shared primitives for the PWO-2 tools (root corpus, qualification runner,
// rollout oracle).
//
// Everything here is frozen by docs/PWO2_QUALIFICATION_REGISTRATION.md section
// 1.1. The two derivations below are the whole reason this header exists: a
// row's RNG must be a function of its registered coordinates ONLY, so that any
// root's search or any (root, continuation) oracle playout is reproducible in
// isolation, independent of candidate, tier, thread count, and visit order.

#ifndef OPEN_SPIEL_EXAMPLES_DUNE_PWO2_COMMON_H_
#define OPEN_SPIEL_EXAMPLES_DUNE_PWO2_COMMON_H_

#include <cstdint>
#include <sstream>
#include <string>
#include <vector>

#include "open_spiel/spiel.h"

#include "dune_search_routing.h"
#include "dune_seed_utils.h"
#include "dune_sha256.h"

namespace pwo2 {

// ---------------------------------------------------------------------------
// 64-bit FNV-1a over ASCII. Used ONLY to turn a registered domain-tag string
// into a uint64_t for dune_seed::DeriveSeed. constexpr so the two tag constants
// below are compile-time values that cannot drift.
// ---------------------------------------------------------------------------
constexpr uint64_t Fnv1a64(const char* s) {
  uint64_t h = 0xcbf29ce484222325ULL;
  while (*s != '\0') {
    h ^= static_cast<uint64_t>(static_cast<unsigned char>(*s++));
    h *= 0x00000100000001b3ULL;
  }
  return h;
}

constexpr uint64_t kDomainSearch = Fnv1a64("PWO2_SEARCH");
constexpr uint64_t kDomainOracle = Fnv1a64("PWO2_ORACLE");

// ---------------------------------------------------------------------------
// Root id: the first 16 hex characters of the 64-hex-char SHA256 history hash,
// parsed big-endian.
// ---------------------------------------------------------------------------
inline uint64_t HexPrefix64(const std::string& hex) {
  SPIEL_CHECK_GE(hex.size(), 16u);
  uint64_t v = 0;
  for (int i = 0; i < 16; ++i) {
    const char c = hex[i];
    uint64_t d;
    if (c >= '0' && c <= '9') {
      d = c - '0';
    } else if (c >= 'a' && c <= 'f') {
      d = 10 + (c - 'a');
    } else if (c >= 'A' && c <= 'F') {
      d = 10 + (c - 'A');
    } else {
      open_spiel::SpielFatalError("history hash is not hex: " + hex);
    }
    v = (v << 4) | d;
  }
  return v;
}

// ---------------------------------------------------------------------------
// The two registered derivations.
//
// search: a function of (search seed, root) ONLY -- NOT candidate, tier, thread
//   count, or visit order.
// oracle: a function of (root, continuation index) ONLY -- NOT candidate, tier,
//   or ACTION. That last exclusion is the pairing: at a given root, continuation
//   k draws the same stream whichever action was applied, so oracle values are
//   common-random-number paired and identical chosen actions give delta exactly 0.
// ---------------------------------------------------------------------------
inline uint64_t SearchRngSeed(int search_seed, const std::string& history_hash) {
  return dune_seed::DeriveSeed(kDomainSearch, static_cast<uint64_t>(search_seed),
                               HexPrefix64(history_hash));
}

inline uint64_t OracleContinuationSeed(const std::string& history_hash,
                                       int continuation_index) {
  return dune_seed::DeriveSeed(kDomainOracle, HexPrefix64(history_hash),
                               static_cast<uint64_t>(continuation_index));
}

// ---------------------------------------------------------------------------
// History hash: SHA256 over the comma-joined action ids. Same construction as
// dune_create_corpus.cc:143 so the two corpora remain comparable.
// ---------------------------------------------------------------------------
inline std::string HistoryHash(const std::vector<open_spiel::Action>& history) {
  std::ostringstream oss;
  for (size_t i = 0; i < history.size(); ++i) {
    if (i > 0) oss << ",";
    oss << history[i];
  }
  return open_spiel::ComputeStringSHA256(oss.str());
}

inline std::string Sha256File(const std::string& path) {
  try {
    return open_spiel::ComputeFileSHA256(path);
  } catch (const std::exception&) {
    return "";
  }
}

inline const char* RoleName(open_spiel::DuneDecisionRole role) {
  switch (role) {
    case open_spiel::DuneDecisionRole::kForcedOrBookkeeping: return "FORCED_OR_BOOKKEEPING";
    case open_spiel::DuneDecisionRole::kLeaderSelection:     return "LEADER_SELECTION";
    case open_spiel::DuneDecisionRole::kAgentPrimary:        return "AGENT_PRIMARY";
    case open_spiel::DuneDecisionRole::kAgentContinuation:   return "AGENT_CONTINUATION";
    case open_spiel::DuneDecisionRole::kPurchase:            return "PURCHASE";
    case open_spiel::DuneDecisionRole::kCombatIntrigue:      return "COMBAT_INTRIGUE";
    case open_spiel::DuneDecisionRole::kOtherOptional:       return "OTHER_OPTIONAL";
  }
  return "UNKNOWN";
}

}  // namespace pwo2

#endif  // OPEN_SPIEL_EXAMPLES_DUNE_PWO2_COMMON_H_
