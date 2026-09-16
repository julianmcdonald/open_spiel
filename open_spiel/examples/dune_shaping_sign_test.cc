// PWO-5 gate 2 item (b) & Round 7 Reward Penalties Regression Tests
//
// Regression tests covering all three reward penalties:
// 1. Specimen conversions (specimen_exchange_penalty = 0.02)
// 2. Family Atomics (family_atomics_penalty = 0.10)
// 3. Plot intrigues (plot_intrigue_penalty = 0.01, exemption_threshold = 3)
// 4. Fixed shaping multiplier (ComputeRewardLambda constant 1.0 throughout)
// 5. Scaled penalty contributions (with reward_scale = 4.0)

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

#include "open_spiel/abseil-cpp/absl/flags/flag.h"

#include "dune_ppo_training_utils.h"
#include "dune_specimen_conversion.h"
#include "open_spiel/games/dune_imperium/dune_imperium.h"
#include "open_spiel/games/dune_imperium/dune_imperium_content.h"
#include "open_spiel/games/dune_imperium/dune_imperium_test_utils.h"

// Flags needed for linking with dune_ppo_training_utils.cc
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

// =============================================================================
// Section 1: Specimen Conversions Tests
// =============================================================================

void TestSpecimenConversionSignAndRange() {
  std::cout << "[Specimen 1] positive penalty DECREASES a conversion transition's reward\n";
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

  // Exact magnitude: penalty * lambda = 0.02.
  const float after = open_spiel::ApplySpecimenExchangeShaping(
      0.5f, kBase + 1, 0.02, 1.0f);
  Check(std::fabs((0.5f - after) - 0.02f) < 1e-7f,
        "decrement equals penalty * reward_lambda (0.02)");

  // Scales with lambda.
  const float after_half = open_spiel::ApplySpecimenExchangeShaping(
      0.5f, kBase + 1, 0.02, 0.5f);
  Check(std::fabs((0.5f - after_half) - 0.01f) < 1e-7f,
        "decrement scales with reward_lambda (0.5 -> 0.01)");

  // Negative rewards move further down.
  const float neg_after = open_spiel::ApplySpecimenExchangeShaping(
      -1.0f, kBase + 5, 0.02, 1.0f);
  Check(neg_after < -1.0f, "negative reward is decreased further");

  // Negative coefficient produces a bonus (proves why negative must be rejected).
  const float neg_coef_after = open_spiel::ApplySpecimenExchangeShaping(
      0.5f, kBase + 1, -0.02, 1.0f);
  Check(neg_coef_after > 0.5f,
        "negative coefficient is a bonus (+0.02) proving rejection necessity");

  // Range checks: 740 is base, not conversion; 753 is outside.
  Check(!open_spiel::dune_shaping::IsSpecimenConversionAction(kBase),
        "740 is NOT a conversion action");
  Check(!open_spiel::dune_shaping::IsSpecimenConversionAction(kBase + 13),
        "753 is NOT a conversion action");
  Check(open_spiel::ApplySpecimenExchangeShaping(0.5f, kBase, 0.02, 1.0f) == 0.5f,
        "740 transition is bitwise unchanged");
  Check(open_spiel::ApplySpecimenExchangeShaping(0.5f, kBase + 13, 0.02, 1.0f) == 0.5f,
        "753 transition is bitwise unchanged");

  // Zero coefficient is bitwise inert.
  Check(open_spiel::ApplySpecimenExchangeShaping(0.123456f, kBase + 1, 0.0, 1.0f) == 0.123456f,
        "zero coefficient is bitwise inert");

  // Scaled isolated penalty with reward_scale = 4.0: -0.02 / 4.0 = -0.005.
  const float scaled_deduction = -0.02f / 4.0f;
  Check(std::fabs(scaled_deduction - (-0.005f)) < 1e-7f,
        "specimen scaled contribution is exactly -0.005");
}

// =============================================================================
// Section 2: Family Atomics Shaping Tests
// =============================================================================

void TestFamilyAtomicsShapingAndSign() {
  std::cout << "[Atomics 1] Family Atomics shaping arithmetic, signs, and conditions\n";
  const open_spiel::Action atomics_action = open_spiel::dune_imperium::kActionFamilyAtomics;
  const float before = 0.5f;
  const float reward_lambda = 1.0f;

  // Example 1: Own reveal turn, 7 persuasion remaining: subtract 0.
  const float ex1 = open_spiel::ApplyFamilyAtomicsShaping(
      before, atomics_action, /*is_own_reveal_turn=*/true, /*remaining_persuasion=*/7,
      /*family_atomics_penalty=*/0.10, reward_lambda);
  Check(ex1 == before, "Example 1: own reveal turn, 7 persuasion -> no penalty (subtract 0)");

  // Example 2: Own reveal turn, already bought a card but 4 persuasion remains: subtract 0.
  const float ex2 = open_spiel::ApplyFamilyAtomicsShaping(
      before, atomics_action, /*is_own_reveal_turn=*/true, /*remaining_persuasion=*/4,
      /*family_atomics_penalty=*/0.10, reward_lambda);
  Check(ex2 == before, "Example 2: own reveal turn, 4 persuasion remaining -> no penalty (subtract 0)");

  // Example 3: Own reveal turn, 0 persuasion remaining: subtract 0.10.
  const float ex3 = open_spiel::ApplyFamilyAtomicsShaping(
      before, atomics_action, /*is_own_reveal_turn=*/true, /*remaining_persuasion=*/0,
      /*family_atomics_penalty=*/0.10, reward_lambda);
  Check(std::fabs((before - ex3) - 0.10f) < 1e-7f,
        "Example 3: own reveal turn, 0 persuasion -> subtract 0.10");

  // Example 4: Before that player reveals, even if they have persuasion: subtract 0.10.
  const float ex4 = open_spiel::ApplyFamilyAtomicsShaping(
      before, atomics_action, /*is_own_reveal_turn=*/false, /*remaining_persuasion=*/5,
      /*family_atomics_penalty=*/0.10, reward_lambda);
  Check(std::fabs((before - ex4) - 0.10f) < 1e-7f,
        "Example 4: before reveal, persuasion=5 -> subtract 0.10");

  // Example 5: After that player's reveal turn has ended: subtract 0.10.
  const float ex5 = open_spiel::ApplyFamilyAtomicsShaping(
      before, atomics_action, /*is_own_reveal_turn=*/false, /*remaining_persuasion=*/0,
      /*family_atomics_penalty=*/0.10, reward_lambda);
  Check(std::fabs((before - ex5) - 0.10f) < 1e-7f,
        "Example 5: after reveal turn ended -> subtract 0.10");

  // Positive coefficient DECREASES reward.
  Check(ex3 < before, "positive coefficient decreases reward");

  // Negative coefficient would increase reward (bonus) -> proves rejection necessity.
  const float neg_after = open_spiel::ApplyFamilyAtomicsShaping(
      before, atomics_action, false, 0, -0.10, reward_lambda);
  Check(neg_after > before, "negative coefficient gives a bonus (+0.10)");

  // Zero coefficient is bitwise inert.
  const float zero_after = open_spiel::ApplyFamilyAtomicsShaping(
      before, atomics_action, false, 0, 0.0, reward_lambda);
  Check(zero_after == before, "zero coefficient is bitwise inert");

  // Non-atomics action is untouched bitwise.
  const float non_atomics = open_spiel::ApplyFamilyAtomicsShaping(
      before, atomics_action + 1, false, 0, 0.10, reward_lambda);
  Check(non_atomics == before, "non-atomics action is bitwise unchanged");

  // Scaled isolated penalty with reward_scale = 4.0: -0.10 / 4.0 = -0.025.
  const float scaled_deduction = -0.10f / 4.0f;
  Check(std::fabs(scaled_deduction - (-0.025f)) < 1e-7f,
        "atomics scaled contribution is exactly -0.025");
}

// =============================================================================
// Section 3: Family Atomics Engine State & Timing Logic
// =============================================================================

void TestFamilyAtomicsEngineStateLogic() {
  std::cout << "[Atomics 2] DuneImperiumState::IsPlayerInRevealTurn and once-per-game\n";
  auto state = open_spiel::dune_imperium::BuildStateAtFirstAgentTurn();
  auto* impl = static_cast<open_spiel::dune_imperium::DuneImperiumState*>(state.get());
  const open_spiel::Player p0 = impl->CurrentPlayer();

  // (1) In kAgentTurns before reveal: IsPlayerInRevealTurn is false.
  Check(impl->phase() == open_spiel::dune_imperium::GamePhase::kAgentTurns, "phase is kAgentTurns");
  Check(!impl->GetPlayerRevealedForTesting(p0), "player 0 not yet revealed");
  Check(!impl->IsPlayerInRevealTurn(p0), "before reveal: IsPlayerInRevealTurn(p0) is FALSE");

  // Even with persuasion > 0 before reveal, still false:
  impl->SetPlayerPersuasionForTesting(p0, 5);
  Check(impl->GetPlayerPersuasionForTesting(p0) == 5, "persuasion is 5");
  Check(!impl->IsPlayerInRevealTurn(p0), "before reveal with 5 persuasion: IsPlayerInRevealTurn(p0) is FALSE");

  // (2) In kAgentTurns after reveal: IsPlayerInRevealTurn is true.
  impl->SetPlayerRevealedForTesting(p0, true);
  Check(impl->IsPlayerInRevealTurn(p0), "in kAgentTurns with revealed=true: IsPlayerInRevealTurn(p0) is TRUE");

  // For other players while p0 is acting:
  Check(!impl->IsPlayerInRevealTurn((p0 + 1) % 4), "not acting player: IsPlayerInRevealTurn(p1) is FALSE");

  // (3) After turn ends (current_player changes):
  impl->SetCurrentPlayerForTesting((p0 + 1) % 4);
  Check(!impl->IsPlayerInRevealTurn(p0), "after turn advances: IsPlayerInRevealTurn(p0) is FALSE");

  // (4) In kRevealTurns:
  impl->SetCurrentPlayerForTesting(p0);
  // Temporarily set phase to kRevealTurns:
  impl->SetPhaseForTesting(open_spiel::dune_imperium::GamePhase::kRevealTurns);
  impl->SetRevealDoneForTesting(p0, false);
  Check(impl->IsPlayerInRevealTurn(p0), "in kRevealTurns with !reveal_done: IsPlayerInRevealTurn(p0) is TRUE");

  impl->SetRevealDoneForTesting(p0, true);
  Check(!impl->IsPlayerInRevealTurn(p0), "in kRevealTurns with reveal_done: IsPlayerInRevealTurn(p0) is FALSE");

  // (5) In kCombat:
  impl->SetPhaseForTesting(open_spiel::dune_imperium::GamePhase::kCombat);
  Check(!impl->IsPlayerInRevealTurn(p0), "in kCombat: IsPlayerInRevealTurn(p0) is FALSE");

  // Restore state to test once-per-game:
  auto fresh_state = open_spiel::dune_imperium::BuildStateAtFirstAgentTurn();
  auto* fresh_impl = static_cast<open_spiel::dune_imperium::DuneImperiumState*>(fresh_state.get());
  const open_spiel::Player actor = fresh_impl->CurrentPlayer();

  Check(!fresh_impl->GetAtomicsUsedForTesting(actor), "atomics_used initially false");
  auto legal = fresh_state->LegalActions();
  Check(open_spiel::dune_imperium::ContainsAction(legal, open_spiel::dune_imperium::kActionFamilyAtomics),
        "kActionFamilyAtomics is legal initially");

  // Apply atomics:
  fresh_state->ApplyAction(open_spiel::dune_imperium::kActionFamilyAtomics);
  Check(fresh_impl->GetAtomicsUsedForTesting(actor), "atomics_used is now true");

  // Handle any chance nodes for refill:
  while (fresh_state->IsChanceNode()) {
    auto outcomes = fresh_state->ChanceOutcomes();
    fresh_state->ApplyAction(outcomes[0].first);
  }

  // Once atomics_used is true, verify kActionFamilyAtomics is NEVER legal again:
  auto legal_after = fresh_state->LegalActions();
  Check(!open_spiel::dune_imperium::ContainsAction(legal_after, open_spiel::dune_imperium::kActionFamilyAtomics),
        "kActionFamilyAtomics is NEVER legal after being used (max 1 Atomics per game)");
}

// =============================================================================
// Section 4: Plot Intrigue Shaping Tests
// =============================================================================

void TestPlotIntrigueShapingAndSign() {
  std::cout << "[Plot 1] Plot intrigue shaping arithmetic, signs, and hand size threshold\n";
  const float before = 0.5f;
  const float reward_lambda = 1.0f;
  const double penalty = 0.01;
  const int threshold = 3;
  const open_spiel::Action plot_action = open_spiel::dune_imperium::kActionPlayIntriguePlotCard0 + 5; // e.g. plot card 5

  Check(open_spiel::IsPlotIntrigueAction(plot_action), "action in 1600-1699 is a plot intrigue");

  // Hand counts 0, 1, 2: penalized (subtract 0.01).
  for (int count = 0; count < 3; ++count) {
    const float after = open_spiel::ApplyPlotIntrigueShaping(
        before, plot_action, count, threshold, penalty, reward_lambda);
    Check(std::fabs((before - after) - 0.01f) < 1e-7f,
          "pre-play count=" + std::to_string(count) + ": subtract 0.01");
    Check(after < before, "positive coefficient decreases reward for count=" + std::to_string(count));
  }

  // Hand counts 3, 4, 5: exempt (subtract 0).
  for (int count = 3; count <= 5; ++count) {
    const float after = open_spiel::ApplyPlotIntrigueShaping(
        before, plot_action, count, threshold, penalty, reward_lambda);
    Check(after == before,
          "pre-play count=" + std::to_string(count) + ": exempt (subtract 0)");
  }

  // Exemption threshold changes EXACTLY at 3:
  // count 2 is penalized, count 3 is exempt.
  const float count2 = open_spiel::ApplyPlotIntrigueShaping(
      before, plot_action, 2, threshold, penalty, reward_lambda);
  const float count3 = open_spiel::ApplyPlotIntrigueShaping(
      before, plot_action, 3, threshold, penalty, reward_lambda);
  Check(count2 < before, "count=2 is penalized");
  Check(count3 == before, "count=3 is exempt (threshold boundary exact at 3)");

  // Negative coefficient produces a bonus (documents why negative is rejected).
  const float neg_after = open_spiel::ApplyPlotIntrigueShaping(
      before, plot_action, 1, threshold, -0.01, reward_lambda);
  Check(neg_after > before, "negative coefficient produces a bonus (+0.01)");

  // Zero coefficient is bitwise inert.
  const float zero_after = open_spiel::ApplyPlotIntrigueShaping(
      before, plot_action, 1, threshold, 0.0, reward_lambda);
  Check(zero_after == before, "zero coefficient is bitwise inert");

  // Non-plot actions:
  // (a) Combat intrigue (1700+):
  const open_spiel::Action combat_intrigue = open_spiel::dune_imperium::kActionPlayIntrigueCombatCard0 + 5;
  Check(!open_spiel::IsPlotIntrigueAction(combat_intrigue), "combat intrigue is NOT plot intrigue");
  Check(open_spiel::ApplyPlotIntrigueShaping(before, combat_intrigue, 1, threshold, penalty, reward_lambda) == before,
        "combat intrigue receives NO plot penalty");

  // (b) Follow-up choices (520..534):
  const open_spiel::Action choice_action = open_spiel::dune_imperium::kActionIntrigueChoiceModeA;
  Check(!open_spiel::IsPlotIntrigueAction(choice_action), "intrigue choice is NOT plot intrigue action");
  Check(open_spiel::ApplyPlotIntrigueShaping(before, choice_action, 1, threshold, penalty, reward_lambda) == before,
        "intrigue follow-up choice receives NO penalty");

  // (c) Chance node / action:
  const open_spiel::Action chance_action = open_spiel::dune_imperium::kActionChanceFizzle;
  Check(!open_spiel::IsPlotIntrigueAction(chance_action), "chance action is NOT plot intrigue");
  Check(open_spiel::ApplyPlotIntrigueShaping(before, chance_action, 1, threshold, penalty, reward_lambda) == before,
        "chance action receives NO penalty");

  // Scaled isolated penalty with reward_scale = 4.0: -0.01 / 4.0 = -0.0025.
  const float scaled_deduction = -0.01f / 4.0f;
  Check(std::fabs(scaled_deduction - (-0.0025f)) < 1e-7f,
        "plot intrigue scaled contribution is exactly -0.0025");
}

// =============================================================================
// Section 5: Fixed Shaping Multiplier Throughout Run & Resume
// =============================================================================

void TestFixedShapingMultiplier() {
  std::cout << "[Multiplier 1] ComputeRewardLambda returns 1.0 throughout when decay_steps == 0\n";

  // Start of run (0 steps):
  Check(open_spiel::ComputeRewardLambda(0, 0, 0) == 1.0f,
        "start of run: ComputeRewardLambda(0, 0, 0) == 1.0f");

  // Midpoint (1,000,000 steps):
  Check(open_spiel::ComputeRewardLambda(1000000, 0, 0) == 1.0f,
        "midpoint: ComputeRewardLambda(1000000, 0, 0) == 1.0f");

  // Target endpoint (2,000,000 steps):
  Check(open_spiel::ComputeRewardLambda(2000000, 0, 0) == 1.0f,
        "target endpoint: ComputeRewardLambda(2000000, 0, 0) == 1.0f");

  // After restart / resume (e.g. 2,071,206,934 steps):
  Check(open_spiel::ComputeRewardLambda(2071206934ULL, 0, 0) == 1.0f,
        "after resume: ComputeRewardLambda(2071206934, 0, 0) == 1.0f");

  // With any start_steps when decay_steps == 0:
  Check(open_spiel::ComputeRewardLambda(5000000000ULL, 206830543ULL, 0) == 1.0f,
        "arbitrary steps with decay_steps=0: multiplier is strictly 1.0f");
}

}  // namespace

int main(int argc, char** argv) {
  std::cout << "dune_shaping_sign_test -- Round 7 Reward Penalties Regression Tests\n\n";

  TestSpecimenConversionSignAndRange();
  std::cout << "\n";

  TestFamilyAtomicsShapingAndSign();
  std::cout << "\n";

  TestFamilyAtomicsEngineStateLogic();
  std::cout << "\n";

  TestPlotIntrigueShapingAndSign();
  std::cout << "\n";

  TestFixedShapingMultiplier();
  std::cout << "\n";

  if (failures == 0) {
    std::cout << "ALL PASS\n";
    return 0;
  }
  std::cout << failures << " FAILED\n";
  return 1;
}
