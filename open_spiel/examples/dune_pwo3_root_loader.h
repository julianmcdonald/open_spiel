#ifndef OPEN_SPIEL_EXAMPLES_DUNE_PWO3_ROOT_LOADER_H_
#define OPEN_SPIEL_EXAMPLES_DUNE_PWO3_ROOT_LOADER_H_

// PWO-3 Amendment 2 (docs/PWO3_AMENDMENT_2_PRECISION_RECOVERY.md): the corpus
// loader and the state reconstruction that `dune_pwo3_teacher_audit --mode=search`
// used to produce every grid row.
//
// WHY THIS IS A LIFT AND NOT A REFACTOR OF THE AUDIT TOOL
// ------------------------------------------------------
// `dune_pwo3_teacher_audit` is the BINARY OF RECORD for every PWO-3 measurement
// (sha256 02874e8f...). Amendment 2 requires it to survive byte-for-byte, so its
// translation unit is not touched -- not even to factor code out of it, because
// the next `make` would then relink a binary that no committed measurement was
// taken with. These three definitions are therefore lifted VERBATIM from
// dune_pwo3_teacher_audit.cc's anonymous namespace, and
// tests/test_pwo3_recovery_instrument.py scans BOTH files and hard-fails if the
// lifted text ever diverges. One behaviour, two locations, an enforced identity.
//
// The lift is `Root`, `Reconstruct` and `LoadCorpusInto`. Nothing else: the
// recovery instrument runs no search budget, emits no grid row, and computes no
// registered quantity.

#include <fstream>
#include <iterator>
#include <memory>
#include <string>
#include <vector>

#include "open_spiel/games/dune_imperium/dune_imperium.h"
#include "open_spiel/spiel.h"
#include "open_spiel/utils/json.h"

#include "dune_pwo2_common.h"
#include "dune_pwo3_common.h"

// The lifted bodies name `Player`, `Action`, `State`, `Game` and `json`
// unqualified, exactly as the audit tool does under its file-scope
// `using namespace open_spiel;`. Nesting inside open_spiel keeps the text
// byte-identical instead of "verbatim apart from the qualifiers".
namespace open_spiel {
namespace pwo3_recovery {

// --- BEGIN LIFT dune_pwo3_teacher_audit.cc: struct Root ---------------------
struct Root {
  std::string root_id;
  std::string stratum;   // decision-role stratum; PWO-2 meaning on EVERY root
  std::string corpus;    // main | conversion | sm
  std::string source_arm;
  std::string half;
  int source_episode_id = 0;
  int round = 0;
  int decision_index = 0;
  Player player = kInvalidPlayer;
  std::vector<Action> history;
  std::vector<Action> legal_actions;
};
// --- END LIFT ---------------------------------------------------------------

// --- BEGIN LIFT dune_pwo3_teacher_audit.cc: Reconstruct ---------------------
inline std::unique_ptr<State> Reconstruct(const std::shared_ptr<const Game>& game,
                                          const std::vector<Action>& history) {
  auto st = game->NewInitialState();
  for (Action a : history) st->ApplyAction(a);
  return st;
}
// --- END LIFT ---------------------------------------------------------------

// --- BEGIN LIFT dune_pwo3_teacher_audit.cc: LoadCorpusInto ------------------
inline void LoadCorpusInto(const std::string& path, const std::string& corpus_tag,
                           std::vector<Root>* out) {
  if (path.empty()) return;
  std::ifstream f(path);
  if (!f) SpielFatalError("cannot open corpus: " + path);
  std::string content((std::istreambuf_iterator<char>(f)),
                      std::istreambuf_iterator<char>());
  auto parsed = json::FromString(content);
  if (!parsed) SpielFatalError("corpus is not valid JSON: " + path);
  for (const auto& v : parsed.value().GetArray()) {
    auto o = v.GetObject();
    Root r;
    r.root_id = o.at("root_id").GetString();
    r.stratum = o.at("stratum").GetString();
    // The main corpus predates the `corpus` field; tag it here.
    r.corpus = (o.find("corpus") != o.end()) ? o.at("corpus").GetString() : corpus_tag;
    SPIEL_CHECK_EQ(r.corpus, corpus_tag);
    r.source_arm = o.at("source_arm").GetString();
    r.half = o.at("half").GetString();
    r.source_episode_id = static_cast<int>(o.at("source_episode_id").GetInt());
    r.round = static_cast<int>(o.at("round").GetInt());
    r.decision_index = static_cast<int>(o.at("decision_index").GetInt());
    r.player = static_cast<Player>(o.at("player").GetInt());
    for (const auto& a : o.at("history").GetArray())
      r.history.push_back(static_cast<Action>(a.GetInt()));
    for (const auto& a : o.at("legal_actions").GetArray())
      r.legal_actions.push_back(static_cast<Action>(a.GetInt()));
    // Fail closed: the corpus must still replay under the frozen engine.
    SPIEL_CHECK_EQ(pwo2::HistoryHash(r.history), r.root_id);
    SPIEL_CHECK_TRUE(pwo3::IsAscending(r.legal_actions));
    out->push_back(std::move(r));
  }
}
// --- END LIFT ---------------------------------------------------------------

}  // namespace pwo3_recovery
}  // namespace open_spiel

#endif  // OPEN_SPIEL_EXAMPLES_DUNE_PWO3_ROOT_LOADER_H_
