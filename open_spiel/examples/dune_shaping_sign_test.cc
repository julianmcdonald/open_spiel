// PWO-5 gate 2 item (b): a unit test asserting that a POSITIVE
// --specimen_exchange_penalty DECREASES the shaped reward on a conversion
// transition.
//
// docs/PWO5_PILOT_REGISTRATION.md section 16 gate 2 item (b) and section 0.3
// item 2: before this file, `grep -rl "specimen_exchange_penalty"` over the
// repo's C++/Python/shell sources returned ONLY six committed launch.sh files.
// No test anywhere covered the sign, and the only validation of the flag was the
// --allow_shaping gate, which tests `!= 0.0` and accepted negatives.
//
// That gap has a measured cost: the u175 lineage trained with
// `--specimen_exchange_penalty=-0.02`, i.e. a +0.02 BONUS on the specimen->troop
// breadcrumb the term exists to suppress. See
// calibration_results_v2/pilot300_search_seed12/launch.sh.
//
// This test exercises `ApplySpecimenExchangeShaping`, which is the function
// dune_ppo_train.cc's rollout loop actually calls -- not a transcription of it.

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

#include "open_spiel/abseil-cpp/absl/flags/flag.h"

#include "dune_ppo_training_utils.h"
#include "dune_specimen_conversion.h"
#include "open_spiel/games/dune_imperium/dune_imperium_content.h"

// dune_ppo_training_utils.cc ABSL_DECLARE_FLAGs these; they are DEFINED in
// dune_ppo_train.cc, which is not linked into a test binary. Same list and same
// values as dune_ppo_training_utils_test.cc:63-76, for the same reason. None is
// read by the code under test -- they only have to exist for the link.
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

int failures = 0;

void Check(bool ok, const std::string& what) {
  if (ok) {
    std::cout << "  PASS  " << what << "\n";
  } else {
    std::cout << "  FAIL  " << what << "\n";
    ++failures;
  }
}

constexpr open_spiel::Action kBase =
    open_spiel::dune_imperium::kActionConvertSpecimenToTroop0;

// --- Test 1: the sign. A positive penalty must DECREASE the reward. ----------
void TestPositivePenaltyDecreasesConversionReward() {
  std::cout << "[1] positive penalty DECREASES a conversion transition's reward\n";
  const float reward_lambda = 1.0f;

  // Every real conversion action, 741..752.
  for (open_spiel::Action a = kBase + 1; a <= kBase + 12; ++a) {
    const float before = 0.5f;
    const float after = open_spiel::ApplySpecimenExchangeShaping(
        before, a, /*specimen_exchange_penalty=*/0.02, reward_lambda);
    Check(after < before,
          "action " + std::to_string(a) + ": " + std::to_string(after) +
              " < " + std::to_string(before));
  }

  // The magnitude is exactly penalty * lambda.
  const float after = open_spiel::ApplySpecimenExchangeShaping(
      0.5f, kBase + 1, 0.02, 1.0f);
  Check(std::fabs((0.5f - after) - 0.02f) < 1e-7f,
        "decrement equals penalty * reward_lambda (0.02)");

  // ... and it scales with reward_lambda.
  const float after_half = open_spiel::ApplySpecimenExchangeShaping(
      0.5f, kBase + 1, 0.02, 0.5f);
  Check(std::fabs((0.5f - after_half) - 0.01f) < 1e-7f,
        "decrement scales with reward_lambda (0.5 -> 0.01)");

  // Negative rewards move down too -- "decreases" is not "moves toward zero".
  const float neg_after = open_spiel::ApplySpecimenExchangeShaping(
      -1.0f, kBase + 5, 0.02, 1.0f);
  Check(neg_after < -1.0f, "a negative reward is decreased further, not shrunk");
}

// --- Test 2: the inverted sign is what the u175 lineage did. -----------------
// The trainer now REJECTS a negative value at startup (gate 2 item (a)), so this
// documents the arithmetic that made the rejection necessary. If this assertion
// ever flips, the flag has stopped being "a magnitude that is subtracted" and
// the fatal check in main() would be guarding the wrong direction.
void TestNegativePenaltyWouldBeABonus() {
  std::cout << "[2] a NEGATIVE penalty would INCREASE the reward (the u175 defect)\n";
  const float before = 0.5f;
  const float after = open_spiel::ApplySpecimenExchangeShaping(
      before, kBase + 1, /*penalty=*/-0.02, 1.0f);
  Check(after > before,
        "-0.02 is a BONUS (" + std::to_string(after) + " > " +
            std::to_string(before) + ") -- this is why main() rejects it");
}

// --- Test 3: the range is 741-752, and 740 is NOT a conversion. --------------
void TestRangeIs741To752() {
  std::cout << "[3] the conversion range is 741-752; 740 is an unused base constant\n";
  Check(kBase == 740, "kActionConvertSpecimenToTroop0 == 740");

  Check(!open_spiel::dune_shaping::IsSpecimenConversionAction(kBase),
        "740 is NOT a conversion action");
  Check(open_spiel::dune_shaping::IsSpecimenConversionAction(kBase + 1),
        "741 IS a conversion action");
  Check(open_spiel::dune_shaping::IsSpecimenConversionAction(kBase + 12),
        "752 IS a conversion action");
  Check(!open_spiel::dune_shaping::IsSpecimenConversionAction(kBase + 13),
        "753 is NOT a conversion action");

  // The shaping must therefore leave a 740 transition BITWISE unchanged, which
  // is the whole content of the 740->741 correction: the old guard would have
  // penalized it.
  const float before = 0.5f;
  const float after =
      open_spiel::ApplySpecimenExchangeShaping(before, kBase, 0.02, 1.0f);
  Check(after == before, "740 transition is bitwise unchanged by the shaping");

  const float after_753 =
      open_spiel::ApplySpecimenExchangeShaping(before, kBase + 13, 0.02, 1.0f);
  Check(after_753 == before, "753 transition is bitwise unchanged");

  // Amount decoding: action_id - 740, over the real range only.
  Check(open_spiel::dune_shaping::SpecimenConversionAmount(kBase + 1) == 1,
        "741 converts 1");
  Check(open_spiel::dune_shaping::SpecimenConversionAmount(kBase + 12) == 12,
        "752 converts 12");
}

// --- Test 4: a zero penalty is bitwise inert on every action. ---------------
void TestZeroPenaltyIsInert() {
  std::cout << "[4] a zero penalty is bitwise inert\n";
  bool all_equal = true;
  for (open_spiel::Action a = kBase - 2; a <= kBase + 14; ++a) {
    const float before = 0.123456f;
    if (open_spiel::ApplySpecimenExchangeShaping(before, a, 0.0, 1.0f) !=
        before) {
      all_equal = false;
    }
  }
  Check(all_equal, "penalty 0.0 returns the reward unchanged on 740-754");
}

}  // namespace

int main(int argc, char** argv) {
  std::cout << "dune_shaping_sign_test -- PWO-5 gate 2 item (b)\n\n";
  TestPositivePenaltyDecreasesConversionReward();
  TestNegativePenaltyWouldBeABonus();
  TestRangeIs741To752();
  TestZeroPenaltyIsInert();
  std::cout << "\n";
  if (failures == 0) {
    std::cout << "ALL PASS\n";
    return 0;
  }
  std::cout << failures << " FAILED\n";
  return 1;
}
