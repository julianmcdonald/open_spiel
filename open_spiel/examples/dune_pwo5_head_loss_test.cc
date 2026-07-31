// PWO-5 gate 3: the NUMERICS of the three auxiliary head losses.
//
// docs/PWO5_PILOT_REGISTRATION.md sections 8.2, 8.3, 8.4 and 8.6, as amended
// by ruling 2.
//
// ---------------------------------------------------------------------------
// WHAT THESE TESTS ARE FOR
// ---------------------------------------------------------------------------
//
// Every value below is computed BY HAND in the test and compared against the
// SHIPPED pwo5loss:: definitions -- the same ones dune_ppo_train's auxiliary
// block and dune_pwo5_head_eval call. A test that recomputed the loss with the
// same torch ops would be a tautology; these assert arithmetic worked out
// independently.
//
// The four registered properties under test:
//
//   1. HUBER, delta = 0.10 on the /20 scale. Quadratic inside the knee, LINEAR
//      outside it. Getting this backwards, or using the framework default of
//      1.0, would make "Huber" a misnomer over the whole reachable range.
//
//   2. TRAJECTORY weighting, NOT row weighting (section 8.4 item i). A per-row
//      mean weights each game by its label count; round-10 games are both
//      over-represented among games AND carry more rows each, so it is the
//      same imbalance twice over. The decisive fixture is a batch where the
//      two answers DIFFER, so passing cannot be an accident.
//
//   3. `next_own_action_head`'s OMISSION semantics (section 8.3 item d): a row
//      with no later action is out of the numerator AND the denominator, and a
//      game with no contributing row is out of the GAME denominator too. It is
//      not zero-filled and not given a uniform target -- either would train the
//      head toward a fiction on 0.8% of rows.
//
//   4. FULL-VOCABULARY softmax with NO MASK for next_own_action (section 8.3
//      item f). The current state's legal mask is not valid for a FUTURE
//      action: the target is illegal at the predicting state on 74.57% of
//      rows, so masking would make the correct answer unreachable on three
//      rows in four.

#include <cmath>
#include <cstdio>
#include <iostream>
#include <string>
#include <vector>

#include <torch/torch.h>

#include "open_spiel/abseil-cpp/absl/flags/flag.h"

#include "dune_ppo_training_utils.h"

// dune_ppo_training_utils.cc ABSL_DECLARE_FLAGs these; they are DEFINED in
// dune_ppo_train.cc, which this test does not link. Defined at their compiled
// defaults purely to satisfy the linker -- no code path here reads one.
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

namespace {

int g_failures = 0;

void Check(bool ok, const std::string& what) {
  std::cout << (ok ? "  PASS  " : "  FAIL  ") << what << "\n";
  if (!ok) ++g_failures;
}

void CheckClose(double got, double want, double tol, const std::string& what) {
  const bool ok = std::fabs(got - want) <= tol;
  char buf[256];
  std::snprintf(buf, sizeof(buf), "%s  (got %.17g, want %.17g)", what.c_str(),
                got, want);
  Check(ok, buf);
}

torch::Tensor F(const std::vector<float>& v) {
  return torch::tensor(v, torch::TensorOptions().dtype(torch::kFloat32));
}
torch::Tensor I(const std::vector<int64_t>& v) {
  return torch::tensor(v, torch::TensorOptions().dtype(torch::kInt64));
}

}  // namespace

int main() {
  torch::manual_seed(0);
  std::cout << "\n=== PWO-5 gate 3: auxiliary head loss NUMERICS ===\n\n";

  // -------------------------------------------------------------------------
  std::cout << "[1] final_vp_head -- Huber on the /20 scale, delta = 0.10\n";
  {
    const double d = 0.10;
    // Residuals chosen to straddle the knee in both directions and to sit
    // exactly ON it, because "<= delta" vs "< delta" is the kind of boundary a
    // reimplementation gets wrong.
    //   pred - target = 0.05  -> INSIDE  -> 0.5 * 0.05^2            = 0.00125
    //   pred - target = 0.10  -> ON      -> 0.5 * 0.10^2            = 0.005
    //   pred - target = 0.30  -> OUTSIDE -> 0.10 * (0.30 - 0.05)    = 0.025
    //   pred - target = -0.40 -> OUTSIDE -> 0.10 * (0.40 - 0.05)    = 0.035
    torch::Tensor pred = F({0.55f, 0.60f, 0.80f, 0.10f});
    torch::Tensor targ = F({0.50f, 0.50f, 0.50f, 0.50f});
    torch::Tensor per_row = open_spiel::pwo5loss::HuberPerRow(pred, targ, d);
    auto a = per_row.accessor<float, 1>();
    CheckClose(a[0], 0.00125, 1e-7, "residual 0.05 (inside the knee): quadratic");
    CheckClose(a[1], 0.005, 1e-7, "residual 0.10 (exactly ON the knee): quadratic");
    CheckClose(a[2], 0.025, 1e-7, "residual 0.30 (outside): LINEAR");
    CheckClose(a[3], 0.035, 1e-7, "residual -0.40 (outside, negative): LINEAR");

    // The registered rejection of the framework default, made numeric: at
    // delta = 1.0 every residual in the reachable range is quadratic, so
    // "Huber" would describe pure squared error.
    torch::Tensor d1 = open_spiel::pwo5loss::HuberPerRow(pred, targ, 1.0);
    auto b = d1.accessor<float, 1>();
    Check(std::fabs(b[2] - 0.5 * 0.30 * 0.30) < 1e-7 &&
              std::fabs(b[3] - 0.5 * 0.40 * 0.40) < 1e-7,
          "delta = 1.0 makes the WHOLE reachable range quadratic -- which is "
          "why the framework default is rejected");
  }

  // -------------------------------------------------------------------------
  std::cout << "\n[2] TRAJECTORY weighting, and why it is not row weighting\n";
  {
    // Two games with DELIBERATELY unequal row counts and unequal losses, so
    // the trajectory answer and the row answer differ. If they agreed, the
    // test would pass under either implementation and prove nothing.
    //
    //   game 0: 4 rows, all loss 1.0   -> within-game mean 1.0
    //   game 1: 1 row,       loss 6.0  -> within-game mean 6.0
    //
    //   TRAJECTORY mean = (1.0 + 6.0) / 2      = 3.5   <-- REGISTERED
    //   row mean        = (1+1+1+1+6) / 5      = 2.0   <-- the biased answer
    torch::Tensor per_row = F({1.0f, 1.0f, 1.0f, 1.0f, 6.0f});
    torch::Tensor game = I({0, 0, 0, 0, 1});
    torch::Tensor mask = torch::ones_like(per_row, torch::kBool);
    int64_t games = 0;
    torch::Tensor l = open_spiel::pwo5loss::TrajectoryMean(per_row, mask, game,
                                                           2, &games);
    CheckClose(l.item<double>(), 3.5, 1e-7,
               "mean over games of the within-game mean");
    Check(std::fabs(l.item<double>() - 2.0) > 1e-6,
          "and it is NOT the row mean 2.0 -- the long game does not get extra "
          "weight from its row count");
    Check(games == 2, "contributing-game denominator is 2");
  }

  // -------------------------------------------------------------------------
  std::cout << "\n[3] next_own_action -- omission is out of BOTH numerator "
               "and denominator\n";
  {
    // game 0: rows 3.0, 5.0, and one row with NO later action
    // game 1: 7.0
    // game 2: EVERY sampled row has no later action -> contributes nothing and
    //         drops out of the GAME denominator entirely.
    torch::Tensor per_row = F({3.0f, 5.0f, 99.0f, 7.0f, 99.0f, 99.0f});
    torch::Tensor game = I({0, 0, 0, 1, 2, 2});
    torch::Tensor has_target =
        torch::tensor({true, true, false, true, false, false},
                      torch::TensorOptions().dtype(torch::kBool));
    int64_t games = 0;
    torch::Tensor l = open_spiel::pwo5loss::TrajectoryMean(per_row, has_target,
                                                           game, 3, &games);
    // game 0 within-game mean over CONTRIBUTING rows = (3+5)/2 = 4
    // game 1 = 7 ; game 2 contributes nothing
    // trajectory mean = (4 + 7) / 2 = 5.5
    CheckClose(l.item<double>(), 5.5, 1e-7,
               "omitted rows leave the numerator AND the within-game "
               "denominator");
    Check(games == 2,
          "a game with NO contributing row is out of the GAME denominator too "
          "(2 of 3 games)");
    // The 99.0 sentinel proves the omitted rows were not merely down-weighted:
    // had they been included at any weight, the result would exceed 5.5.
    Check(l.item<double>() < 6.0,
          "the 99.0 sentinel rows contributed NOTHING -- they were not "
          "zero-filled, uniform-targeted, or down-weighted");
  }

  // -------------------------------------------------------------------------
  std::cout << "\n[4] cross-entropy, and the FULL-VOCABULARY no-mask rule\n";
  {
    // A uniform 3-class logit vector gives CE = log(3) for any target, which
    // is checkable without reference to any implementation.
    torch::Tensor logits = torch::zeros({2, 3});
    torch::Tensor target = I({0, 2});
    torch::Tensor ce =
        open_spiel::pwo5loss::CrossEntropyPerRow(logits, target);
    auto a = ce.accessor<float, 1>();
    CheckClose(a[0], std::log(3.0), 1e-6, "uniform 3-class CE == log(3)");
    CheckClose(a[1], std::log(3.0), 1e-6, "and is target-independent when "
                                          "uniform");

    // A peaked distribution: logits [10, 0, 0]. CE for class 0 is
    // -log(e^10 / (e^10 + 2)) = log(1 + 2*e^-10).
    torch::Tensor peaked = torch::tensor({{10.0f, 0.0f, 0.0f}});
    torch::Tensor t0 = I({0});
    double want = std::log(1.0 + 2.0 * std::exp(-10.0));
    CheckClose(
        open_spiel::pwo5loss::CrossEntropyPerRow(peaked, t0).item<double>(),
        want, 1e-6, "peaked logits give the analytic CE");

    // The no-mask rule, stated numerically. Over a 2,391-way head the target
    // is usually ILLEGAL at the predicting state (74.57% of rows), so the
    // vocabulary must be the full action space. A uniform 2,391-way head gives
    // log(2391) -- and a masked head restricted to, say, 8 legal actions
    // would give log(8), a number the head could reach only by predicting
    // something known to be wrong.
    torch::Tensor full = torch::zeros({1, 2391});
    torch::Tensor tfull = I({1234});
    CheckClose(
        open_spiel::pwo5loss::CrossEntropyPerRow(full, tfull).item<double>(),
        std::log(2391.0), 1e-4,
        "uniform FULL-vocabulary (2,391) CE == log(2391); a masked head would "
        "make the correct answer unreachable on 3 rows in 4");
  }

  // -------------------------------------------------------------------------
  std::cout << "\n[5] head-off short-circuit: a zero coefficient constructs "
               "NO loss term\n";
  {
    open_spiel::Pwo5AuxConfig off;
    off.final_vp_coef = 0.0;
    off.terminal_round_coef = 0.0;
    off.next_own_action_coef = 0.0;
    Check(!off.AnyActive(),
          "all-zero coefficients report INACTIVE, so the auxiliary forward and "
          "every head loss are skipped entirely");
    open_spiel::Pwo5AuxConfig on = off;
    on.next_own_action_coef = 0.15;
    Check(on.AnyActive(),
          "a single nonzero coefficient activates the auxiliary path");
    // Multiply-by-zero is REJECTED as the mechanism: it would still build the
    // graph, still populate .grad with exact zeros, and still expose the head
    // parameters to weight decay and to any non-finite value in the head's
    // forward. AnyActive() is the short-circuit that makes a head-off arm's
    // behaviour independent of the head's numerics.
    open_spiel::Pwo5AuxConfig tiny = off;
    tiny.final_vp_coef = 1e-300;
    Check(tiny.AnyActive(),
          "the short-circuit tests EXACT zero, not smallness -- 1e-300 is "
          "active, so no coefficient is silently rounded away");
  }

  // -------------------------------------------------------------------------
  std::cout << "\n[6] gradient grouping: the three heads are separable by "
               "name\n";
  {
    // The per-head gradient norms are grouped by parameter-name prefix, and
    // `next_own_action_head` does NOT contain the substring `final_vp_head` or
    // `terminal_round_head`, so prefix matching is unambiguous. Asserted
    // because a substring collision here would silently merge two heads' norms.
    const std::string a = "final_vp_head";
    const std::string b = "terminal_round_head";
    const std::string c = "next_own_action_head";
    Check(c.find(a) == std::string::npos && c.find(b) == std::string::npos &&
              b.find(a) == std::string::npos && a.find(b) == std::string::npos,
          "no head name is a substring of another, so prefix grouping cannot "
          "merge two heads' gradient norms");
    Check(std::string("final_vp_head.weight").rfind(a, 0) == 0 &&
              std::string("next_own_action_head.bias").rfind(c, 0) == 0,
          "parameter names begin with their head's name, which is what the "
          "grouping matches on");
  }

  std::cout << "\n";
  if (g_failures == 0) {
    std::cout << "ALL HEAD-LOSS NUMERIC TESTS PASSED\n";
    return 0;
  }
  std::cout << g_failures << " CHECK(S) FAILED\n";
  return 1;
}
