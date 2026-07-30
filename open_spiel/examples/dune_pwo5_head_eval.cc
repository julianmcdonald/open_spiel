// PWO-5 update-300 whole-dataset auxiliary-head evaluator.
//
// docs/PWO5_AMENDMENT_1_TARGET_EXPOSURE_TELEMETRY_2026_07_31.md section 8
// (RULING 7), against docs/PWO5_PILOT_REGISTRATION.md sections 8.2, 8.4, 8.5,
// 8.6 and 9.5.
//
// ---------------------------------------------------------------------------
// WHAT THIS IS, AND WHY IT IS A SEPARATE BINARY
// ---------------------------------------------------------------------------
//
// Ruling 7 registers that at pilot-local update 300 each arm reports, for all
// three heads:
//
//   * the per-head TRAIN loss on the training population;
//   * the per-head WHOLE-TRAJECTORY-HELD-OUT loss on section 8.5's 60 held-out
//     games -- the measurement section 9.5 requires and which no earlier gate
//     produced;
//   * the TRAJECTORY DENOMINATOR behind every one of those six numbers,
//     reported beside it and never inferred;
//   * TERMINAL-CLASS SUPPORT in BOTH splits, at the trajectory denominator
//     (section 8.4 item iii: reported whether or not it looks healthy).
//
// This is a whole-dataset pass -- all 41,132 non-held-out rows across 340
// games, and all 7,311 held-out rows across 60 games -- not the 1,024-row
// sample the trainer's per-update telemetry reports. It is a SEPARATE BINARY
// because Appendix A.1's closing rule makes any gate-3 trainer flag outside its
// table a STOP, and a whole-dataset evaluation pass has no registered flag.
//
// ---------------------------------------------------------------------------
// TWO REGISTERED SUBTLETIES THIS TOOL MUST NOT GET WRONG
// ---------------------------------------------------------------------------
//
//  1. TRAJECTORY WEIGHTING, not row weighting (section 8.4 item i and section
//     8.6). The per-row losses of a game are averaged WITHIN the game, and the
//     reported loss is the mean over contributing GAMES. A per-row mean would
//     weight each game by its label count -- and round-10 games are both
//     over-represented among games AND carry more rows each, so it is the same
//     imbalance twice over.
//
//  2. `next_own_action_head`'s OMISSION SEMANTICS (section 8.3 item d). The
//     final label row of each game has no later action; its target is -1. Such
//     a row is out of the numerator AND the denominator -- it is NOT
//     zero-filled and NOT given a uniform target, either of which would train
//     the head toward a fiction. A game with no contributing row at all is out
//     of the GAME denominator too. The affected games and rows are REPORTED
//     rather than silently absorbed.
//
// The five held-out `<= 8` games are DESCRIPTIVE ONLY: no gate, threshold or
// nomination criterion reads their per-class cross-entropy, and `n = 5` is
// printed every time it appears (section 8.5).
//
// Auxiliary-loss improvement is never by itself a nomination criterion
// (sections 9.5 and 15.2). This tool reports; it rules on nothing.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "open_spiel/abseil-cpp/absl/flags/flag.h"
#include "open_spiel/abseil-cpp/absl/flags/parse.h"
#include "open_spiel/spiel.h"

#include "dune_network.h"
#include "dune_pwo5_aux.h"
#include "dune_sha256.h"

ABSL_FLAG(std::string, model_checkpoint, "",
          "REQUIRED. The arm's pilot-local update-300 model checkpoint.");
ABSL_FLAG(std::string, aux_target_path, "",
          "REQUIRED. The section 5.2 auxiliary-target artifact.");
ABSL_FLAG(std::string, aux_target_sha256, "",
          "If set, asserted against the artifact (section 5.2).");
ABSL_FLAG(std::string, aux_heldout_games_path, "",
          "REQUIRED. The section 8.5 60-game membership list.");
ABSL_FLAG(std::string, aux_heldout_sha256, "",
          "If set, asserted against the canonical membership digest.");
ABSL_FLAG(double, huber_delta_final_vp, 0.10,
          "Section 8.2 / amendment ruling 2. On the /20 scale.");
ABSL_FLAG(int, hidden_dim, 2048, "Must match the checkpoint.");
ABSL_FLAG(int, num_blocks, 8, "Must match the checkpoint.");
ABSL_FLAG(bool, nonlinear_value_head, false, "Must match the checkpoint.");
ABSL_FLAG(int, batch_size, 512, "Forward batch size. Numerics-neutral.");
ABSL_FLAG(std::string, output_json, "", "REQUIRED. Where the report is written.");
ABSL_FLAG(std::string, arm_label, "", "Recorded verbatim in the report.");
ABSL_FLAG(int, expect_training_rows, 41132, "Section 8.5. Asserted.");
ABSL_FLAG(int, expect_training_games, 340, "Section 8.5. Asserted.");
ABSL_FLAG(int, expect_heldout_rows, 7311, "Section 8.5. Asserted.");
ABSL_FLAG(int, expect_heldout_games, 60, "Section 8.5. Asserted.");

namespace {

using open_spiel::pwo5::AuxTargetStore;

[[noreturn]] void Stop(const std::string& why) {
  std::cerr << "\nSTOP: " << why << "\n";
  std::exit(2);
}

void Require(bool ok, const std::string& why) {
  if (!ok) Stop(why);
}

// %.17g. Registered by section 13.5's round-trip-precision rule and by ruling
// 6's note on open_spiel's json.cc writer, whose `%f` six-decimal floor turns
// anything below 5e-7 into exactly 0.0 -- which a converged head loss trips.
std::string F17(double v) {
  char buf[64];
  std::snprintf(buf, sizeof(buf), "%.17g", v);
  return std::string(buf);
}

// One head's trajectory-weighted result: the mean over contributing games of
// the within-game mean, plus every denominator behind it.
struct HeadResult {
  double loss = 0.0;
  int64_t games = 0;          // contributing GAMES -- the operative denominator
  int64_t rows = 0;           // contributing rows, reported for completeness
  int64_t excluded_rows = 0;  // rows with no target (next_own_action only)
  int64_t games_with_exclusions = 0;
  int64_t games_with_no_contributing_row = 0;
};

struct SplitResult {
  HeadResult final_vp;
  HeadResult terminal_round;
  HeadResult next_own_action;
  int64_t games = 0;
  int64_t rows = 0;
  // Terminal-class support at BOTH denominators. Section 8.4 item iii requires
  // both be reported; the TRAJECTORY one is the operative denominator under
  // trajectory-weighted averaging, and the row counts are reported only to show
  // the two are tracked separately -- a healthy-looking row count must never
  // stand in for a thin game count.
  std::array<int64_t, 3> class_games{{0, 0, 0}};
  std::array<int64_t, 3> class_rows{{0, 0, 0}};
  // Per-class terminal_round cross-entropy, trajectory-weighted within class.
  std::array<double, 3> class_terminal_round_ce{{0.0, 0.0, 0.0}};
};

// Evaluates one whole split. `game_rows` is (game_index -> canonically ordered
// row indices); every row of every game is evaluated exactly once.
SplitResult EvaluateSplit(
    const std::shared_ptr<open_spiel::SharedDunePolicyValueNetImpl>& model,
    const AuxTargetStore& store,
    const std::map<int32_t, std::vector<int64_t>>& game_rows,
    const torch::Device& device, int batch_size, double huber_delta) {
  torch::NoGradGuard no_grad;
  model->eval();
  SplitResult out;

  const int64_t obs_size = store.obs_size();

  // Per-game accumulators, so the within-game mean is exact regardless of how
  // the rows happen to fall across forward batches.
  struct GameAcc {
    double fv_sum = 0.0;
    int64_t fv_n = 0;
    double tr_sum = 0.0;
    int64_t tr_n = 0;
    double na_sum = 0.0;
    int64_t na_n = 0;
    int64_t na_excluded = 0;
    int32_t tclass = 0;
  };
  std::map<int32_t, GameAcc> acc;

  // A flat evaluation order over the whole split. Order does not affect the
  // result -- every row lands in its own game's accumulator -- it only affects
  // batching.
  std::vector<int64_t> flat;
  for (const auto& kv : game_rows) {
    GameAcc a;
    a.tclass = store.rows()[kv.second.front()].terminal_round_class;
    acc[kv.first] = a;
    for (int64_t r : kv.second) flat.push_back(r);
  }
  out.games = static_cast<int64_t>(game_rows.size());
  out.rows = static_cast<int64_t>(flat.size());

  std::vector<float> obs_buf;
  obs_buf.reserve(static_cast<size_t>(batch_size) * obs_size);

  for (size_t start = 0; start < flat.size();
       start += static_cast<size_t>(batch_size)) {
    const size_t end =
        std::min(flat.size(), start + static_cast<size_t>(batch_size));
    const int64_t n = static_cast<int64_t>(end - start);
    obs_buf.clear();
    for (size_t i = start; i < end; ++i) {
      const float* o = store.observation(flat[i]);
      obs_buf.insert(obs_buf.end(), o, o + obs_size);
    }
    torch::Tensor x =
        torch::from_blob(obs_buf.data(), {n, obs_size},
                         torch::TensorOptions().dtype(torch::kFloat32))
            .clone()
            .to(device);
    auto ao = model->ForwardAux(x);

    // final_vp: Huber on the /20 scale.
    torch::Tensor pred = ao.final_vp.squeeze(1).to(torch::kFloat32).cpu();
    // terminal_round: 3-class CE.
    torch::Tensor tr_logp =
        torch::log_softmax(ao.terminal_round.to(torch::kFloat32), -1).cpu();
    // next_own_action: FULL-VOCABULARY softmax, NO MASK. The current state's
    // legal mask is not valid for a FUTURE action -- the target is illegal at
    // the predicting state on 74.57% of rows, so masking would make the correct
    // answer unreachable on three rows in four (section 8.3 item f).
    torch::Tensor na_logp =
        torch::log_softmax(ao.next_own_action.to(torch::kFloat32), -1).cpu();

    auto pred_a = pred.accessor<float, 1>();
    auto tr_a = tr_logp.accessor<float, 2>();
    auto na_a = na_logp.accessor<float, 2>();

    for (int64_t k = 0; k < n; ++k) {
      const int64_t row = flat[start + static_cast<size_t>(k)];
      const auto& r = store.rows()[row];
      GameAcc& a = acc[r.game_index];

      const double diff = static_cast<double>(pred_a[k]) - r.final_vp_target;
      const double ad = std::fabs(diff);
      const double huber = (ad <= huber_delta)
                               ? 0.5 * diff * diff
                               : huber_delta * (ad - 0.5 * huber_delta);
      a.fv_sum += huber;
      ++a.fv_n;

      a.tr_sum += -static_cast<double>(tr_a[k][r.terminal_round_class]);
      ++a.tr_n;

      if (r.next_own_action >= 0) {
        a.na_sum += -static_cast<double>(na_a[k][r.next_own_action]);
        ++a.na_n;
      } else {
        ++a.na_excluded;
      }
    }
  }

  // Trajectory weighting: mean over contributing GAMES of the within-game mean.
  double fv_total = 0.0, tr_total = 0.0, na_total = 0.0;
  std::array<double, 3> class_tr_total{{0.0, 0.0, 0.0}};
  for (const auto& kv : acc) {
    const GameAcc& a = kv.second;
    ++out.class_games[a.tclass];
    out.class_rows[a.tclass] += a.fv_n;

    if (a.fv_n > 0) {
      fv_total += a.fv_sum / static_cast<double>(a.fv_n);
      ++out.final_vp.games;
      out.final_vp.rows += a.fv_n;
    }
    if (a.tr_n > 0) {
      const double per_game = a.tr_sum / static_cast<double>(a.tr_n);
      tr_total += per_game;
      class_tr_total[a.tclass] += per_game;
      ++out.terminal_round.games;
      out.terminal_round.rows += a.tr_n;
    }
    if (a.na_n > 0) {
      na_total += a.na_sum / static_cast<double>(a.na_n);
      ++out.next_own_action.games;
      out.next_own_action.rows += a.na_n;
    } else {
      // Registered: a game with no contributing row is out of the GAME
      // denominator too, and the fact is reported rather than absorbed.
      ++out.next_own_action.games_with_no_contributing_row;
    }
    if (a.na_excluded > 0) {
      out.next_own_action.excluded_rows += a.na_excluded;
      ++out.next_own_action.games_with_exclusions;
    }
  }
  out.final_vp.loss =
      out.final_vp.games > 0 ? fv_total / out.final_vp.games : 0.0;
  out.terminal_round.loss =
      out.terminal_round.games > 0 ? tr_total / out.terminal_round.games : 0.0;
  out.next_own_action.loss =
      out.next_own_action.games > 0 ? na_total / out.next_own_action.games : 0.0;
  for (int c = 0; c < 3; ++c) {
    out.class_terminal_round_ce[c] =
        out.class_games[c] > 0 ? class_tr_total[c] / out.class_games[c] : 0.0;
  }
  return out;
}

std::string HeadJson(const std::string& name, const HeadResult& h) {
  std::ostringstream j;
  j << "      \"" << name << "\": {\n"
    << "        \"loss\": " << F17(h.loss) << ",\n"
    << "        \"trajectory_denominator_games\": " << h.games << ",\n"
    << "        \"contributing_rows\": " << h.rows << ",\n"
    << "        \"excluded_rows_no_target\": " << h.excluded_rows << ",\n"
    << "        \"games_with_excluded_rows\": " << h.games_with_exclusions << ",\n"
    << "        \"games_with_no_contributing_row\": "
    << h.games_with_no_contributing_row << "\n"
    << "      }";
  return j.str();
}

std::string SplitJson(const std::string& name, const SplitResult& s,
                      bool heldout) {
  std::ostringstream j;
  j << "    \"" << name << "\": {\n"
    << "      \"games\": " << s.games << ",\n"
    << "      \"rows\": " << s.rows << ",\n"
    << HeadJson("final_vp_head", s.final_vp) << ",\n"
    << HeadJson("terminal_round_head", s.terminal_round) << ",\n"
    << HeadJson("next_own_action_head", s.next_own_action) << ",\n"
    << "      \"terminal_class_support\": {\n"
    << "        \"note\": \"The TRAJECTORY denominator is the operative one "
       "under section 8.4 item i's trajectory-weighted averaging. Row counts "
       "are reported only to show the two are tracked separately.\",\n"
    << "        \"games\": {\"le8\": " << s.class_games[0]
    << ", \"r9\": " << s.class_games[1] << ", \"r10\": " << s.class_games[2]
    << "},\n"
    << "        \"rows\": {\"le8\": " << s.class_rows[0]
    << ", \"r9\": " << s.class_rows[1] << ", \"r10\": " << s.class_rows[2]
    << "}\n"
    << "      },\n"
    << "      \"terminal_round_ce_by_class\": {\n"
    << "        \"le8\": {\"ce\": " << F17(s.class_terminal_round_ce[0])
    << ", \"n_games\": " << s.class_games[0];
  if (heldout) {
    j << ", \"status\": \"DESCRIPTIVE ONLY -- no gate, threshold or nomination "
         "criterion reads this cell (section 8.5). n = "
      << s.class_games[0] << ".\"";
  }
  j << "},\n"
    << "        \"r9\": {\"ce\": " << F17(s.class_terminal_round_ce[1])
    << ", \"n_games\": " << s.class_games[1] << "},\n"
    << "        \"r10\": {\"ce\": " << F17(s.class_terminal_round_ce[2])
    << ", \"n_games\": " << s.class_games[2] << "}\n"
    << "      }\n"
    << "    }";
  return j.str();
}

void PrintSplit(const std::string& title, const SplitResult& s, bool heldout) {
  std::cout << "--- " << title << " ---\n"
            << "  games " << s.games << ", rows " << s.rows << "\n"
            << "  final_vp_head        loss " << F17(s.final_vp.loss)
            << "   trajectory denominator " << s.final_vp.games << " games ("
            << s.final_vp.rows << " rows)\n"
            << "  terminal_round_head  loss " << F17(s.terminal_round.loss)
            << "   trajectory denominator " << s.terminal_round.games
            << " games (" << s.terminal_round.rows << " rows)\n"
            << "  next_own_action_head loss " << F17(s.next_own_action.loss)
            << "   trajectory denominator " << s.next_own_action.games
            << " games (" << s.next_own_action.rows << " rows)\n"
            << "    rows with NO later action, excluded from BOTH numerator "
               "and denominator: "
            << s.next_own_action.excluded_rows << " across "
            << s.next_own_action.games_with_exclusions << " games"
            << (s.next_own_action.games_with_no_contributing_row > 0
                    ? "  *** " +
                          std::to_string(
                              s.next_own_action
                                  .games_with_no_contributing_row) +
                          " game(s) contributed NO row and are out of the game "
                          "denominator ***"
                    : "")
            << "\n"
            << "  terminal-class support (TRAJECTORY denominator -- the "
               "operative one):\n"
            << "    <=8 " << s.class_games[0] << " games / " << s.class_rows[0]
            << " rows    9 " << s.class_games[1] << " / " << s.class_rows[1]
            << "    10 " << s.class_games[2] << " / " << s.class_rows[2] << "\n"
            << "  terminal_round CE by class: <=8 "
            << F17(s.class_terminal_round_ce[0]) << " (n=" << s.class_games[0]
            << (heldout ? ", DESCRIPTIVE ONLY" : "") << ")   9 "
            << F17(s.class_terminal_round_ce[1]) << " (n=" << s.class_games[1]
            << ")   10 " << F17(s.class_terminal_round_ce[2])
            << " (n=" << s.class_games[2] << ")\n\n";
}

}  // namespace

int main(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);

  const std::string model_path = absl::GetFlag(FLAGS_model_checkpoint);
  const std::string target_path = absl::GetFlag(FLAGS_aux_target_path);
  const std::string heldout_path = absl::GetFlag(FLAGS_aux_heldout_games_path);
  const std::string out_json = absl::GetFlag(FLAGS_output_json);
  Require(!model_path.empty(), "--model_checkpoint is required");
  Require(!target_path.empty(), "--aux_target_path is required");
  Require(!heldout_path.empty(), "--aux_heldout_games_path is required");
  Require(!out_json.empty(), "--output_json is required");

  std::cout << "PWO-5 update-300 whole-dataset head evaluation (ruling 7)\n"
            << "  arm         : " << absl::GetFlag(FLAGS_arm_label) << "\n"
            << "  checkpoint  : " << model_path << "\n";

  // ----- the held-out membership list, and its registered digest -----
  std::vector<int32_t> heldout;
  {
    std::ifstream hin(heldout_path);
    Require(static_cast<bool>(hin), "cannot open " + heldout_path);
    std::string text((std::istreambuf_iterator<char>(hin)),
                     std::istreambuf_iterator<char>());
    const std::size_t key = text.find("heldout_game_indices");
    Require(key != std::string::npos,
            heldout_path + " has no heldout_game_indices");
    const std::size_t lb = text.find('[', key);
    const std::size_t rb = text.find(']', lb);
    std::string body = text.substr(lb + 1, rb - lb - 1);
    for (char& c : body) if (c == ',') c = ' ';
    std::istringstream bs(body);
    int32_t v;
    while (bs >> v) heldout.push_back(v);
  }
  std::sort(heldout.begin(), heldout.end());
  {
    std::string canonical;
    for (std::size_t i = 0; i < heldout.size(); ++i) {
      if (i) canonical += ",";
      canonical += std::to_string(heldout[i]);
    }
    const std::string digest = open_spiel::ComputeStringSHA256(canonical);
    const std::string expect = absl::GetFlag(FLAGS_aux_heldout_sha256);
    Require(expect.empty() || expect == digest,
            "held-out split digest mismatch: expected " + expect +
                " computed " + digest);
    std::cout << "  heldout     : " << heldout.size() << " games, digest "
              << digest << "\n";
  }

  // ----- the target artifact, and its registered digest -----
  {
    const std::string expect = absl::GetFlag(FLAGS_aux_target_sha256);
    if (!expect.empty()) {
      const std::string digest = open_spiel::ComputeFileSHA256(target_path);
      Require(expect == digest, "--aux_target_sha256 mismatch for " +
                                    target_path + ": expected " + expect +
                                    " computed " + digest);
    }
  }
  AuxTargetStore store;
  {
    std::string err;
    Require(store.Load(target_path, heldout, &err),
            "auxiliary target artifact rejected: " + err);
  }

  // Section 8.5's realized split sizes, ASSERTED rather than printed.
  Require(static_cast<int>(store.training_row_count()) ==
              absl::GetFlag(FLAGS_expect_training_rows),
          "training rows " + std::to_string(store.training_row_count()) +
              " != the registered " +
              std::to_string(absl::GetFlag(FLAGS_expect_training_rows)));
  Require(static_cast<int>(store.training_games().size()) ==
              absl::GetFlag(FLAGS_expect_training_games),
          "training games " + std::to_string(store.training_games().size()) +
              " != the registered " +
              std::to_string(absl::GetFlag(FLAGS_expect_training_games)));
  Require(static_cast<int>(store.heldout_row_count()) ==
              absl::GetFlag(FLAGS_expect_heldout_rows),
          "held-out rows " + std::to_string(store.heldout_row_count()) +
              " != the registered " +
              std::to_string(absl::GetFlag(FLAGS_expect_heldout_rows)));
  Require(static_cast<int>(store.heldout_games().size()) ==
              absl::GetFlag(FLAGS_expect_heldout_games),
          "held-out games " + std::to_string(store.heldout_games().size()) +
              " != the registered " +
              std::to_string(absl::GetFlag(FLAGS_expect_heldout_games)));
  std::cout << "  population  : train " << store.training_row_count()
            << " rows / " << store.training_games().size() << " games;  "
            << "held out " << store.heldout_row_count() << " rows / "
            << store.heldout_games().size() << " games\n\n";

  torch::Device device(torch::cuda::is_available() ? torch::kCUDA
                                                   : torch::kCPU);
  auto model = std::make_shared<open_spiel::SharedDunePolicyValueNetImpl>(
      store.obs_size(), absl::GetFlag(FLAGS_hidden_dim), store.action_dim(),
      absl::GetFlag(FLAGS_num_blocks),
      absl::GetFlag(FLAGS_nonlinear_value_head), /*with_aux_heads=*/true,
      /*head_init_seed=*/0);
  torch::load(model, model_path, device);
  model->to(device);

  const double delta = absl::GetFlag(FLAGS_huber_delta_final_vp);
  const int batch = absl::GetFlag(FLAGS_batch_size);

  const SplitResult train = EvaluateSplit(model, store, store.training_game_rows(),
                                          device, batch, delta);
  const SplitResult held = EvaluateSplit(model, store, store.heldout_game_rows(),
                                         device, batch, delta);

  PrintSplit("TRAIN split (all non-held-out rows and games)", train, false);
  PrintSplit("WHOLE-TRAJECTORY-HELD-OUT split (section 8.5's 60 games)", held,
             true);

  const std::string model_sha = open_spiel::ComputeFileSHA256(model_path);
  std::ostringstream j;
  j << "{\n"
    << "  \"tool\": \"dune_pwo5_head_eval\",\n"
    << "  \"registration\": \"PWO-5 amendment 1 ruling 7 (update-300 head "
       "evaluation)\",\n"
    << "  \"arm_label\": \"" << absl::GetFlag(FLAGS_arm_label) << "\",\n"
    << "  \"model_checkpoint\": \"" << model_path << "\",\n"
    << "  \"model_sha256\": \"" << model_sha << "\",\n"
    << "  \"aux_target_path\": \"" << target_path << "\",\n"
    << "  \"huber_delta_final_vp\": " << F17(delta) << ",\n"
    << "  \"weighting\": \"TRAJECTORY -- mean over contributing games of the "
       "within-game mean (sections 8.4 item i, 8.6). NOT row-weighted.\",\n"
    << "  \"next_own_action_omission\": \"A row with no later action is out of "
       "the numerator AND the denominator; it is not zero-filled and not given "
       "a uniform target (section 8.3 item d).\",\n"
    << "  \"nomination_status\": \"REPORTED ONLY. Auxiliary-loss improvement is "
       "never by itself a nomination criterion (sections 9.5, 15.2).\",\n"
    << "  \"splits\": {\n"
    << SplitJson("train", train, false) << ",\n"
    << SplitJson("heldout", held, true) << "\n"
    << "  }\n"
    << "}\n";
  {
    std::ofstream o(out_json);
    Require(static_cast<bool>(o), "cannot write " + out_json);
    o << j.str();
  }

  // The banner is LAST, after every assertion above has passed.
  std::cout << "update-300 head evaluation COMPLETE for arm '"
            << absl::GetFlag(FLAGS_arm_label) << "'\n"
            << "  all six numbers reported with their trajectory denominators; "
               "terminal-class support reported in both splits\n"
            << "  report: " << out_json << "\n";
  return 0;
}
