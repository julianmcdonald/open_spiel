// PWO-4 independent decision_role verifier — replay, do not re-call.
//
// Registration of record: docs/PWO4_TRAJECTORY_REGISTRATION.md revision 3.
// Work order: docs/PWO4_IMPLEMENTATION_PROMPT_2026_07_28.md CP0.6 guard 3, CP1.1a.
//
// ---------------------------------------------------------------------------
// WHY THIS BINARY EXISTS
// ---------------------------------------------------------------------------
// CP0.6 requires three guards on every emitted decision_role. Two of them --
// non-empty, and a member of the registered role set -- are cheap local
// assertions the generator makes at emission time. The third is not:
//
//   "it EQUALS ClassifyDuneDecisionRole recomputed at the same state and player
//    by the analyzer's independent call path."
//
// Discharging that inside the generator, on the live `state` object it just
// classified, would prove only that ClassifyDuneDecisionRole is a deterministic
// function -- which is true by inspection and worth nothing. It would not catch
// the failure modes that actually threaten the section 9.3 floors:
//
//   * the role recorded against the wrong acting player;
//   * the role recorded against a stale or mutated state;
//   * a decision's row bound to the wrong position in the game;
//   * a recorded history that does not replay to the state it claims.
//
// Every one of those survives a double call and dies here. This tool reconstructs
// each decision state from the PERSISTED game history -- fresh initial state, the
// recorded actions applied in order, stopping at the recorded history_length --
// and classifies THAT state. The classifier is the same
// ClassifyDuneDecisionRole (dune_search_routing.h:18); the STATE is arrived at by
// an independent route, which is the part that carries the evidence.
//
// It is a separate binary, not a mode, so the analyzer invokes it as its own
// process on committed files. It loads no model, runs no search and touches no
// GPU: the classifier is a pure function of (state, player, has_active_session).
//
// has_active_session is FALSE, matching dune_search_benchmark.cc:784-785 on the
// fresh-search path (there is no session, so `cur_session` is null). That
// argument is load-bearing -- it selects the kAgentContinuation branch at
// dune_search_routing.cc's agent-turn test -- so it is pinned here rather than
// defaulted.
//
// While the state is reconstructed anyway, four further recorded facts are
// checked for free, and each one is an independent hazard:
//   acting_player, n_legal, legal_actions, and the CP0.4 search_expected
//   predicate (n_legal >= 2 && IsStrategicState). A stream whose history replays
//   to a different player or a different legal-action set is corrupt regardless
//   of whether its roles happen to agree.

#include <algorithm>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "open_spiel/abseil-cpp/absl/flags/flag.h"
#include "open_spiel/abseil-cpp/absl/flags/parse.h"
#include "open_spiel/games/dune_imperium/dune_imperium.h"
#include "open_spiel/spiel.h"
#include "open_spiel/utils/json.h"

#include "dune_puct_is_mcts.h"   // IsStrategicState
#include "dune_search_routing.h"  // ClassifyDuneDecisionRole

ABSL_FLAG(std::string, games_jsonl_path, "", "REQUIRED. Per-game records carrying the full action history.");
ABSL_FLAG(std::string, audit_jsonl_path, "", "REQUIRED. Audit telemetry: one record per searched-seat decision.");
ABSL_FLAG(std::string, output_json_path, "", "REQUIRED. The verdict.");
ABSL_FLAG(bool, check_strategic_state, true, "The controller pin, needed to recompute search_expected.");
ABSL_FLAG(int, max_mismatch_examples, 20, "How many mismatches to enumerate in the verdict.");

namespace open_spiel {
namespace {

// Mirrors dune_pwo4_trajectory.cc's RegisteredRoleName: the registration's
// spelling, so the verdict and the stream are directly comparable without a
// translation step.
const char* RegisteredRoleName(DuneDecisionRole role) {
  switch (role) {
    case DuneDecisionRole::kForcedOrBookkeeping: return "ForcedOrBookkeeping";
    case DuneDecisionRole::kLeaderSelection:     return "LeaderSelection";
    case DuneDecisionRole::kAgentPrimary:        return "AgentPrimary";
    case DuneDecisionRole::kAgentContinuation:   return "AgentContinuation";
    case DuneDecisionRole::kPurchase:            return "Purchase";
    case DuneDecisionRole::kCombatIntrigue:      return "CombatIntrigue";
    case DuneDecisionRole::kOtherOptional:       return "OtherOptional";
  }
  return "UNKNOWN";
}

// One recorded decision, as read from audit.jsonl.
struct AuditDecision {
  int64_t game_index = -1;
  int64_t decision_index = -1;
  int64_t history_length = -1;
  int64_t acting_player = -1;
  int64_t n_legal = -1;
  bool search_expected = false;
  bool is_strategic = false;
  std::string decision_role;
  std::vector<Action> legal_actions;
};

std::vector<Action> ReadActionArray(const json::Object& o, const std::string& key) {
  std::vector<Action> out;
  auto it = o.find(key);
  if (it == o.end()) return out;
  for (const auto& v : it->second.GetArray()) out.push_back(static_cast<Action>(v.GetInt()));
  return out;
}

// Reads a JSONL file one object per line. json::FromString parses a single
// document, so the file is walked line by line rather than slurped.
bool ForEachJsonLine(const std::string& path,
                     const std::function<bool(const json::Object&, int)>& fn) {
  std::ifstream f(path);
  if (!f) {
    std::cerr << "STOP: cannot open " << path << "\n";
    return false;
  }
  std::string line;
  int lineno = 0;
  while (std::getline(f, line)) {
    ++lineno;
    if (line.empty()) continue;
    auto parsed = json::FromString(line);
    if (!parsed) {
      std::cerr << "STOP: " << path << ":" << lineno << " is not valid JSON.\n";
      return false;
    }
    if (!fn(parsed.value().GetObject(), lineno)) return false;
  }
  return true;
}

}  // namespace
}  // namespace open_spiel

using namespace open_spiel;

int main(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);
  const std::string games_path = absl::GetFlag(FLAGS_games_jsonl_path);
  const std::string audit_path = absl::GetFlag(FLAGS_audit_jsonl_path);
  const std::string out_path = absl::GetFlag(FLAGS_output_json_path);
  if (games_path.empty() || audit_path.empty() || out_path.empty()) {
    std::cerr << "STOP: --games_jsonl_path, --audit_jsonl_path and "
                 "--output_json_path are all required.\n";
    return 1;
  }
  const bool check_strategic = absl::GetFlag(FLAGS_check_strategic_state);
  const int max_examples = absl::GetFlag(FLAGS_max_mismatch_examples);

  // --- Load the histories. -------------------------------------------------
  std::map<int64_t, std::vector<Action>> histories;
  if (!ForEachJsonLine(games_path, [&](const json::Object& o, int lineno) {
        const int64_t gi = o.at("game_index").GetInt();
        if (histories.count(gi)) {
          std::cerr << "STOP: duplicate game_index " << gi << " at " << games_path
                    << ":" << lineno << ".\n";
          return false;
        }
        std::vector<Action> h = ReadActionArray(o, "history");
        const int64_t declared = o.at("history_length").GetInt();
        if (static_cast<int64_t>(h.size()) != declared) {
          std::cerr << "STOP: game " << gi << " declares history_length " << declared
                    << " but carries " << h.size() << " actions.\n";
          return false;
        }
        histories[gi] = std::move(h);
        return true;
      })) {
    return 1;
  }

  // --- Load the decisions, grouped by game and ordered by position. --------
  std::map<int64_t, std::vector<AuditDecision>> by_game;
  int64_t n_audit = 0;
  if (!ForEachJsonLine(audit_path, [&](const json::Object& o, int lineno) {
        AuditDecision d;
        d.game_index = o.at("game_index").GetInt();
        d.decision_index = o.at("decision_index").GetInt();
        d.history_length = o.at("history_length").GetInt();
        d.acting_player = o.at("acting_player").GetInt();
        d.n_legal = o.at("n_legal").GetInt();
        d.search_expected = o.at("search_expected").GetBool();
        d.is_strategic = o.at("is_strategic").GetBool();
        d.decision_role = o.at("decision_role").GetString();
        d.legal_actions = ReadActionArray(o, "legal_actions");
        if (!histories.count(d.game_index)) {
          std::cerr << "STOP: " << audit_path << ":" << lineno << " references game "
                    << d.game_index << ", which has no history record.\n";
          return false;
        }
        by_game[d.game_index].push_back(std::move(d));
        ++n_audit;
        return true;
      })) {
    return 1;
  }

  // --- Replay. -------------------------------------------------------------
  std::shared_ptr<const Game> game = LoadGame("dune_imperium");

  int64_t n_checked = 0;
  int64_t n_role_mismatch = 0, n_player_mismatch = 0, n_legal_mismatch = 0;
  int64_t n_strategic_mismatch = 0, n_search_expected_mismatch = 0;
  std::map<std::string, int64_t> role_histogram;
  json::Array examples;

  for (auto& [game_index, decisions] : by_game) {
    std::sort(decisions.begin(), decisions.end(),
              [](const AuditDecision& a, const AuditDecision& b) {
                return a.history_length < b.history_length;
              });
    for (size_t i = 1; i < decisions.size(); ++i) {
      if (decisions[i].history_length == decisions[i - 1].history_length) {
        std::cerr << "STOP: game " << game_index << " has two decisions at "
                     "history_length " << decisions[i].history_length << ".\n";
        return 1;
      }
    }

    const std::vector<Action>& history = histories[game_index];
    auto state = game->NewInitialState();
    size_t pos = 0;

    for (const AuditDecision& d : decisions) {
      if (d.history_length < 0 || static_cast<size_t>(d.history_length) > history.size()) {
        std::cerr << "STOP: game " << game_index << " decision " << d.decision_index
                  << " has history_length " << d.history_length << ", outside the "
                  << history.size() << "-action history.\n";
        return 1;
      }
      // Advance the INDEPENDENTLY replayed state to this decision's position.
      while (pos < static_cast<size_t>(d.history_length)) {
        state->ApplyAction(history[pos]);
        ++pos;
      }
      if (state->IsTerminal() || state->IsChanceNode()) {
        std::cerr << "STOP: game " << game_index << " decision " << d.decision_index
                  << " replays to a "
                  << (state->IsTerminal() ? "terminal" : "chance")
                  << " state, which cannot be a searched-seat decision.\n";
        return 1;
      }

      const Player replayed_player = state->CurrentPlayer();
      const std::vector<Action> replayed_legal = state->LegalActions();
      // has_active_session = false: the fresh-search path has no session
      // (dune_search_benchmark.cc:784-785). The argument selects a branch, so it
      // is pinned rather than defaulted.
      const DuneDecisionRole replayed_role =
          ClassifyDuneDecisionRole(*state, replayed_player, false);
      const std::string replayed_role_name = RegisteredRoleName(replayed_role);
      const bool replayed_strategic = IsStrategicState(*state, replayed_player);
      const bool replayed_search_expected =
          (replayed_legal.size() >= 2) && (!check_strategic || replayed_strategic);

      ++n_checked;
      role_histogram[replayed_role_name]++;

      const bool player_ok = (replayed_player == static_cast<Player>(d.acting_player));
      const bool legal_ok = (replayed_legal == d.legal_actions) &&
                            (static_cast<int64_t>(replayed_legal.size()) == d.n_legal);
      const bool role_ok = (replayed_role_name == d.decision_role);
      const bool strategic_ok = (replayed_strategic == d.is_strategic);
      const bool se_ok = (replayed_search_expected == d.search_expected);

      if (!player_ok) ++n_player_mismatch;
      if (!legal_ok) ++n_legal_mismatch;
      if (!role_ok) ++n_role_mismatch;
      if (!strategic_ok) ++n_strategic_mismatch;
      if (!se_ok) ++n_search_expected_mismatch;

      if ((!player_ok || !legal_ok || !role_ok || !strategic_ok || !se_ok) &&
          static_cast<int>(examples.size()) < max_examples) {
        json::Object ex;
        ex["game_index"] = game_index;
        ex["decision_index"] = d.decision_index;
        ex["history_length"] = d.history_length;
        ex["recorded_acting_player"] = d.acting_player;
        ex["replayed_acting_player"] = static_cast<int64_t>(replayed_player);
        ex["recorded_n_legal"] = d.n_legal;
        ex["replayed_n_legal"] = static_cast<int64_t>(replayed_legal.size());
        ex["recorded_decision_role"] = d.decision_role;
        ex["replayed_decision_role"] = replayed_role_name;
        ex["recorded_is_strategic"] = d.is_strategic;
        ex["replayed_is_strategic"] = replayed_strategic;
        ex["recorded_search_expected"] = d.search_expected;
        ex["replayed_search_expected"] = replayed_search_expected;
        examples.push_back(ex);
      }
    }
  }

  // --- The verdict. --------------------------------------------------------
  const bool ok = (n_role_mismatch == 0 && n_player_mismatch == 0 &&
                   n_legal_mismatch == 0 && n_strategic_mismatch == 0 &&
                   n_search_expected_mismatch == 0 && n_checked == n_audit);

  json::Object v;
  v["tool"] = std::string("dune_pwo4_role_verify");
  v["method"] = std::string(
      "Each decision state is reconstructed by replaying the persisted per-game "
      "action history from a fresh initial state up to the recorded "
      "history_length, then classified with ClassifyDuneDecisionRole(state, "
      "player, has_active_session=false). The classifier is shared with the "
      "emitter; the STATE is reached by an independent route, which is what makes "
      "this a check rather than a second call.");
  v["games"] = static_cast<int64_t>(by_game.size());
  v["audit_rows_read"] = n_audit;
  v["decisions_checked"] = n_checked;
  v["role_mismatches"] = n_role_mismatch;
  v["acting_player_mismatches"] = n_player_mismatch;
  v["legal_actions_mismatches"] = n_legal_mismatch;
  v["is_strategic_mismatches"] = n_strategic_mismatch;
  v["search_expected_mismatches"] = n_search_expected_mismatch;
  json::Object hist;
  for (const auto& kv : role_histogram) hist[kv.first] = kv.second;
  v["replayed_role_histogram"] = hist;
  v["mismatch_examples"] = examples;
  v["verdict"] = std::string(ok ? "PASS" : "FAIL");

  std::ofstream out(out_path);
  if (!out) {
    std::cerr << "STOP: cannot open --output_json_path.\n";
    return 1;
  }
  out << json::ToString(v) << "\n";

  std::cerr << "role verification: " << n_checked << " decisions replayed, "
            << n_role_mismatch << " role / " << n_player_mismatch << " player / "
            << n_legal_mismatch << " legal-action / " << n_strategic_mismatch
            << " is_strategic / " << n_search_expected_mismatch
            << " search_expected mismatches -> " << (ok ? "PASS" : "FAIL") << "\n";
  return ok ? 0 : 1;
}
