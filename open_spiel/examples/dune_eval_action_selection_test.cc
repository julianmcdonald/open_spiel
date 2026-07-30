// PWO-5 gate 2 item (d): a unit test asserting that `--opponent_greedy` and
// `--opponent_temperature` LEAVE THE DEFAULT PATH BITWISE UNCHANGED.
//
// docs/PWO5_PILOT_REGISTRATION.md section 16 gate 2 item (d):
//
//   "dune_population_eval gains opponent-specific action selection --
//    --opponent_greedy and --opponent_temperature, DEFAULTING TO THE CURRENT
//    GLOBAL BEHAVIOUR SO NO COMMITTED RESULT CHANGES. ... A unit test must
//    assert that the new flags leave the default path bitwise unchanged."
//
// "Bitwise unchanged" is asserted two ways, because the flags could break it in
// two independent places:
//
//   TEST A -- RESOLUTION. With the overrides unset, an opponent seat must resolve
//     to EXACTLY the global (greedy, temperature), for every combination of
//     global values -- including `--greedy=false`, which is the case a plain
//     `ABSL_FLAG(bool, opponent_greedy, true)` would have silently broken.
//
//   TEST B -- SELECTION. The refactor moved the argmax/softmax-sample block out
//     of dune_population_eval.cc's worker loop and into a shared function. This
//     test pins that function against an INDEPENDENT TRANSCRIPTION of the
//     ORIGINAL pre-gate-2 code (`OriginalSelect` below, copied verbatim from
//     dune_population_eval.cc:647-686 before the change) over a randomized grid,
//     requiring bitwise identical chosen actions AND identical RNG state
//     afterwards. Same action from a different number of RNG draws would
//     desynchronize every later decision in the episode, so the RNG check is not
//     redundant.
//
// Neither test needs a GPU, a checkpoint, or the engine.

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "dune_eval_action_selection.h"

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

using open_spiel::Action;
using open_spiel::dune_eval::ResolveSelectionPolicy;
using open_spiel::dune_eval::SelectActionFromLogits;
using open_spiel::dune_eval::SelectionPolicy;

// ---------------------------------------------------------------------------
// The ORIGINAL selection code, transcribed verbatim from
// dune_population_eval.cc:647-686 as it stood at commit 905fb3c (gate 1), before
// gate 2 touched it. This is the reference the shared function must reproduce.
// It is deliberately NOT refactored -- including the `-1e9f` / `-1e30` sentinels
// and the `legal_actions.back()` fallback.
// ---------------------------------------------------------------------------
template <typename Rng>
Action OriginalSelect(const std::vector<float>& logits,
                      const std::vector<Action>& legal_actions, bool greedy,
                      float temperature, Rng& rng) {
  Action chosen_action = -1;
  if (greedy || legal_actions.size() == 1) {
    // Argmax
    chosen_action = legal_actions.front();
    float max_logit = -1e9f;
    for (Action a : legal_actions) {
      if (logits[a] > max_logit) {
        max_logit = logits[a];
        chosen_action = a;
      }
    }
  } else {
    // Stochastic: softmax with temperature, then sample
    double max_logit = -1e30;
    for (Action a : legal_actions) {
      if (logits[a] > max_logit) {
        max_logit = logits[a];
      }
    }
    std::vector<double> probs(legal_actions.size());
    double sum = 0.0;
    for (size_t i = 0; i < legal_actions.size(); ++i) {
      probs[i] = std::exp((static_cast<double>(logits[legal_actions[i]]) -
                           max_logit) /
                          static_cast<double>(temperature));
      sum += probs[i];
    }
    for (auto& p : probs) p /= sum;

    std::uniform_real_distribution<double> dist(0.0, 1.0);
    double r = dist(rng);
    double cumulative = 0.0;
    chosen_action = legal_actions.back();
    for (size_t i = 0; i < legal_actions.size(); ++i) {
      cumulative += probs[i];
      if (r < cumulative) {
        chosen_action = legal_actions[i];
        break;
      }
    }
  }
  return chosen_action;
}

// ---------------------------------------------------------------------------
// TEST A -- the resolution rule.
// ---------------------------------------------------------------------------
void TestUnsetOverridesInheritTheGlobals() {
  std::cout << "[A] with the overrides UNSET, an opponent resolves to the globals\n";

  const bool globals_greedy[] = {true, false};
  const float globals_temp[] = {0.0f, 0.5f, 1.0f, 2.0f};

  bool all_ok = true;
  for (bool g : globals_greedy) {
    for (float t : globals_temp) {
      // Opponent seat, both overrides unset.
      SelectionPolicy opp = ResolveSelectionPolicy(
          /*is_candidate=*/false, g, t,
          /*has_opponent_greedy=*/false, /*opponent_greedy=*/true,
          /*has_opponent_temperature=*/false, /*opponent_temperature=*/0.0f);
      if (opp.greedy != g || opp.temperature != t) all_ok = false;

      // Candidate seat, both overrides unset.
      SelectionPolicy cand = ResolveSelectionPolicy(
          /*is_candidate=*/true, g, t, false, true, false, 0.0f);
      if (cand.greedy != g || cand.temperature != t) all_ok = false;
    }
  }
  Check(all_ok,
        "unset overrides => opponent AND candidate both equal (greedy, temperature), "
        "over 8 global combinations");

  // The specific regression a plain `bool opponent_greedy = true` would cause:
  // a committed run passing --greedy=false must NOT get greedy opponents.
  SelectionPolicy opp_sampling = ResolveSelectionPolicy(
      false, /*global_greedy=*/false, /*global_temperature=*/1.0f, false, true,
      false, 0.0f);
  Check(opp_sampling.greedy == false,
        "--greedy=false with no override keeps opponents SAMPLING "
        "(the defaulted-bool regression)");

  // The candidate NEVER reads the overrides, even when they are set.
  SelectionPolicy cand_with_overrides = ResolveSelectionPolicy(
      /*is_candidate=*/true, /*global_greedy=*/false,
      /*global_temperature=*/1.0f,
      /*has_opponent_greedy=*/true, /*opponent_greedy=*/true,
      /*has_opponent_temperature=*/true, /*opponent_temperature=*/0.0f);
  Check(cand_with_overrides.greedy == false &&
            cand_with_overrides.temperature == 1.0f,
        "the CANDIDATE ignores the opponent overrides entirely");
}

// ---------------------------------------------------------------------------
// TEST A2 -- the registered PWO-5 configuration is actually expressible.
// ---------------------------------------------------------------------------
void TestRegisteredPwo5ConfigurationIsExpressible() {
  std::cout << "[A2] the section 13.2 configuration resolves as registered\n";
  // --greedy=false --temperature=1.0 --opponent_greedy=true
  // --opponent_temperature=0.0
  SelectionPolicy cand = ResolveSelectionPolicy(true, false, 1.0f, true, true,
                                                true, 0.0f);
  SelectionPolicy opp = ResolveSelectionPolicy(false, false, 1.0f, true, true,
                                               true, 0.0f);
  Check(cand.greedy == false && cand.temperature == 1.0f,
        "candidate SAMPLES at temperature 1.0");
  Check(opp.greedy == true,
        "opponents are GREEDY -- the frozen strong population (section 13.1)");
  // Before this gate, one global pair had to serve both, so this pairing was not
  // expressible at all.
  Check(cand.greedy != opp.greedy,
        "candidate and opponents differ -- NOT expressible before gate 2");
}

// ---------------------------------------------------------------------------
// TEST B -- selection is bitwise identical to the original code.
// ---------------------------------------------------------------------------
void TestSelectionIsBitwiseIdenticalToTheOriginal() {
  std::cout << "[B] the shared selector reproduces the original bitwise\n";

  std::mt19937_64 gen(20260801ULL);
  std::uniform_real_distribution<float> logit_dist(-30.0f, 30.0f);
  std::uniform_int_distribution<int> nlegal_dist(1, 40);

  const int kTrials = 4000;
  int mismatched_action = 0;
  int mismatched_rng = 0;
  int greedy_trials = 0;
  int sampling_trials = 0;
  int forced_trials = 0;

  const int kActionDim = 2391;

  for (int trial = 0; trial < kTrials; ++trial) {
    // A logit vector over the full action space, and a legal subset.
    std::vector<float> logits(kActionDim);
    for (float& v : logits) v = logit_dist(gen);

    const int n_legal = nlegal_dist(gen);
    std::vector<Action> legal_actions;
    {
      std::uniform_int_distribution<int> a_dist(0, kActionDim - 1);
      std::vector<bool> seen(kActionDim, false);
      while (static_cast<int>(legal_actions.size()) < n_legal) {
        int a = a_dist(gen);
        if (seen[a]) continue;
        seen[a] = true;
        legal_actions.push_back(a);
      }
    }

    // Exercise ties: with probability ~1/4, flatten the legal logits so several
    // actions share the maximum. This is where an argmax refactor most easily
    // changes tie-breaking.
    if (trial % 4 == 0) {
      for (Action a : legal_actions) logits[a] = 3.25f;
    }

    const bool greedy = (trial % 3 == 0);
    const float temperature =
        (trial % 5 == 0) ? 0.25f : ((trial % 5 == 1) ? 2.0f : 1.0f);

    if (n_legal == 1) {
      ++forced_trials;
    } else if (greedy) {
      ++greedy_trials;
    } else {
      ++sampling_trials;
    }

    // Two RNGs seeded identically, so any difference in the number of draws
    // shows up as a difference in final state.
    const uint64_t seed = 0x9E3779B97F4A7C15ULL ^ static_cast<uint64_t>(trial);
    std::mt19937_64 rng_original(seed);
    std::mt19937_64 rng_shared(seed);

    const Action original =
        OriginalSelect(logits, legal_actions, greedy, temperature,
                       rng_original);

    // The shared path, driven through the RESOLVER with the overrides unset --
    // i.e. exactly what an unmodified command line produces.
    const SelectionPolicy policy = ResolveSelectionPolicy(
        /*is_candidate=*/false, greedy, temperature,
        /*has_opponent_greedy=*/false, /*opponent_greedy=*/true,
        /*has_opponent_temperature=*/false, /*opponent_temperature=*/0.0f);
    const Action shared =
        SelectActionFromLogits(logits, legal_actions, policy, rng_shared);

    if (original != shared) ++mismatched_action;
    if (rng_original != rng_shared) ++mismatched_rng;
  }

  Check(mismatched_action == 0,
        "chosen action identical on all " + std::to_string(kTrials) +
            " trials (" + std::to_string(mismatched_action) + " mismatches)");
  Check(mismatched_rng == 0,
        "RNG state identical afterwards on all trials -- same number of draws (" +
            std::to_string(mismatched_rng) + " mismatches)");

  // The grid must actually have covered all three branches, or the test above
  // is vacuous on the ones it missed.
  Check(greedy_trials > 100, "argmax branch exercised (" +
                                 std::to_string(greedy_trials) + " trials)");
  Check(sampling_trials > 100, "softmax-sample branch exercised (" +
                                   std::to_string(sampling_trials) +
                                   " trials)");
  Check(forced_trials > 0, "forced (n_legal == 1) branch exercised (" +
                               std::to_string(forced_trials) + " trials)");
}

// ---------------------------------------------------------------------------
// TEST B2 -- the greedy path consumes no randomness.
// ---------------------------------------------------------------------------
// The original took the argmax branch on `greedy || legal_actions.size() == 1`
// and never touched the RNG there, which is what keeps a seat's RNG stream
// aligned across --greedy settings. If the shared function drew even once on
// that branch, every later decision in the episode would shift.
void TestGreedyPathDrawsNoRandomness() {
  std::cout << "[B2] the argmax branch consumes no randomness\n";
  std::vector<float> logits(2391, 0.0f);
  logits[7] = 1.0f;
  std::vector<Action> legal_actions = {3, 7, 11};

  std::mt19937_64 rng(12345);
  const std::mt19937_64 before = rng;
  const Action a = SelectActionFromLogits(
      logits, legal_actions, SelectionPolicy{true, 1.0f}, rng);
  Check(a == 7, "argmax picks the max-logit action");
  Check(rng == before, "RNG untouched on the greedy branch");

  // A forced decision takes the argmax branch even when greedy is false.
  std::vector<Action> forced = {42};
  std::mt19937_64 rng2(999);
  const std::mt19937_64 before2 = rng2;
  const Action f = SelectActionFromLogits(
      logits, forced, SelectionPolicy{false, 1.0f}, rng2);
  Check(f == 42, "a single legal action is returned");
  Check(rng2 == before2, "RNG untouched on a forced decision even when sampling");
}

}  // namespace

int main(int argc, char** argv) {
  std::cout << "dune_eval_action_selection_test -- PWO-5 gate 2 item (d)\n\n";
  TestUnsetOverridesInheritTheGlobals();
  TestRegisteredPwo5ConfigurationIsExpressible();
  TestSelectionIsBitwiseIdenticalToTheOriginal();
  TestGreedyPathDrawsNoRandomness();
  std::cout << "\n";
  if (failures == 0) {
    std::cout << "ALL PASS\n";
    return 0;
  }
  std::cout << failures << " FAILED\n";
  return 1;
}
