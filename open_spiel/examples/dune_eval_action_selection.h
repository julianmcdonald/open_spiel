#ifndef OPEN_SPIEL_EXAMPLES_DUNE_EVAL_ACTION_SELECTION_H_
#define OPEN_SPIEL_EXAMPLES_DUNE_EVAL_ACTION_SELECTION_H_

// Population-evaluation action selection, extracted so that PWO-5 gate 2 item
// (d) can be UNIT TESTED rather than reasoned about.
//
// docs/PWO5_PILOT_REGISTRATION.md section 16 gate 2 item (d) and section 13.2:
// the PWO-5 inferential readout evaluates a candidate SAMPLED at temperature 1.0
// against the frozen strong population, which section 13.1 pins as
// `3x Branch-A u2450 seed11, raw network prior, GREEDY`. `dune_population_eval`
// exposed only a GLOBAL `--greedy` and a GLOBAL `--temperature`, so that pairing
// -- candidate sampled, opponents greedy -- was NOT EXPRESSIBLE in the binary.
//
// The gate therefore adds `--opponent_greedy` and `--opponent_temperature`,
// "defaulting to the current global behaviour so no committed result changes".
//
// TWO THINGS MAKE THAT DEFAULT CLAIM TRUE RATHER THAN INTENDED, and both are
// what this header exists for:
//
//  1. The new flags are `std::optional`, defaulting to `std::nullopt` =
//     INHERIT. A plain `ABSL_FLAG(bool, opponent_greedy, true)` would NOT be
//     backward compatible: a committed run passing `--greedy=false` would keep
//     greedy opponents under it, silently changing the opponent population.
//     `ResolveSelectionPolicy` below is the one place that resolution happens.
//
//  2. The selection arithmetic itself is ONE function used by BOTH the candidate
//     and the opponent path, byte-for-byte the pre-gate-2 code: the same
//     `float max_logit = -1e9f` argmax with strictly-greater tie-breaking (first
//     maximum wins), the same `double max_logit = -1e30` softmax shift, the same
//     `std::uniform_real_distribution<double>(0.0, 1.0)` draw against a running
//     cumulative, and the same `legal_actions.back()` fallback when floating
//     point leaves the cumulative just short of `r`. A reimplementation would
//     have been free to "clean up" any of those and change results.
//
// The test `dune_eval_action_selection_test.cc` asserts (1) directly and
// asserts (2) by running this function against an independent transcription of
// the ORIGINAL pre-gate-2 code over a randomized grid, requiring bitwise
// identical chosen actions.

#include <cmath>
#include <cstddef>
#include <random>
#include <vector>

#include "open_spiel/spiel.h"  // Action

namespace open_spiel {
namespace dune_eval {

// How one seat selects its action from logits.
struct SelectionPolicy {
  bool greedy;
  float temperature;
};

// Resolve the effective policy for a seat.
//
// `is_candidate` is the existing `use_model` discriminator in
// dune_population_eval.cc -- true for the evaluated model's seat, false for an
// opponent seat.
//
// When the opponent overrides are unset (`has_opponent_*` false), an opponent
// seat resolves to EXACTLY the global `(greedy, temperature)`, which is the
// pre-gate-2 behaviour for every seat. The candidate seat NEVER reads the
// opponent overrides.
inline SelectionPolicy ResolveSelectionPolicy(bool is_candidate,
                                              bool global_greedy,
                                              float global_temperature,
                                              bool has_opponent_greedy,
                                              bool opponent_greedy,
                                              bool has_opponent_temperature,
                                              float opponent_temperature) {
  if (is_candidate) {
    return SelectionPolicy{global_greedy, global_temperature};
  }
  return SelectionPolicy{
      has_opponent_greedy ? opponent_greedy : global_greedy,
      has_opponent_temperature ? opponent_temperature : global_temperature};
}

// Choose an action from `logits` (indexed by action id) over `legal_actions`.
//
// `logits` must already be legal-centered and capped by CenterAndCapLegalLogits.
// `rng` is the caller's per-episode, per-player generator and is advanced ONLY
// on the sampling branch -- exactly as before, so the greedy path consumes no
// randomness and the RNG draw sequence is unchanged for any configuration that
// was expressible before this gate.
template <typename Rng>
inline Action SelectActionFromLogits(const std::vector<float>& logits,
                                     const std::vector<Action>& legal_actions,
                                     const SelectionPolicy& policy, Rng& rng) {
  if (policy.greedy || legal_actions.size() == 1) {
    // Argmax. First maximum wins (strict `>`), matching the original.
    Action chosen_action = legal_actions.front();
    float max_logit = -1e9f;
    for (Action a : legal_actions) {
      if (logits[a] > max_logit) {
        max_logit = logits[a];
        chosen_action = a;
      }
    }
    return chosen_action;
  }

  // Stochastic: softmax with temperature, then sample.
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
                        static_cast<double>(policy.temperature));
    sum += probs[i];
  }
  for (auto& p : probs) p /= sum;

  std::uniform_real_distribution<double> dist(0.0, 1.0);
  double r = dist(rng);
  double cumulative = 0.0;
  Action chosen_action = legal_actions.back();
  for (size_t i = 0; i < legal_actions.size(); ++i) {
    cumulative += probs[i];
    if (r < cumulative) {
      chosen_action = legal_actions[i];
      break;
    }
  }
  return chosen_action;
}

}  // namespace dune_eval
}  // namespace open_spiel

#endif  // OPEN_SPIEL_EXAMPLES_DUNE_EVAL_ACTION_SELECTION_H_
