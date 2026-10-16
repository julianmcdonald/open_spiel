// Tests for the belief particle filter. No network: opponent models are
// uniform or a fixed deterministic function, and play is uniformly random.

#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "open_spiel/examples/dune_belief_particles.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace dune_belief {
namespace {

constexpr int kMaxSteps = 4000;

Action RandomAction(const State& state, std::mt19937_64& rng) {
  if (state.IsChanceNode()) {
    return SampleAction(state.ChanceOutcomes(),
                        std::generate_canonical<double, 53>(rng)).first;
  }
  const auto legal = state.LegalActions();
  return legal[std::uniform_int_distribution<size_t>(0, legal.size() - 1)(rng)];
}

// Every live particle must show the observer exactly what the real game shows,
// at every step, and at least one particle must stay alive.
void TestParticlesStayConsistent(Player observer, uint64_t seed) {
  auto game = LoadGame("dune_imperium");
  auto state = game->NewInitialState();
  BeliefOptions options;
  options.num_particles = 8;
  options.parallel_tasks = 4;
  BeliefParticleFilter filter(*state, observer, options, seed, UniformOpponentModel());
  std::mt19937_64 rng(seed);
  int steps = 0;
  while (!state->IsTerminal() && steps++ < kMaxSteps) {
    const Action action = RandomAction(*state, rng);
    auto before = state->Clone();
    state->ApplyAction(action);
    filter.Observe(*before, action, *state);
    const std::string truth = state->ObservationString(observer);
    int alive = 0;
    for (int i = 0; i < filter.size(); ++i) {
      if (!filter.alive(i)) continue;
      ++alive;
      SPIEL_CHECK_EQ(filter.particle(i).ObservationString(observer), truth);
    }
    SPIEL_CHECK_GT(alive, 0);
  }
  SPIEL_CHECK_GT(filter.stats().hidden_chance_nodes, 0);
  std::cout << "observer " << observer << ": steps=" << steps
            << " hidden_chance=" << filter.stats().hidden_chance_nodes
            << " opponent_decisions=" << filter.stats().opponent_decisions
            << " killed_illegal=" << filter.stats().killed_illegal
            << " killed_inconsistent=" << filter.stats().killed_inconsistent
            << " resamples=" << filter.stats().resamples
            << " rebuilds=" << filter.stats().rebuilds << "\n";
}

// A world seed must always give the same world, and two filters run with the
// same seed over the same game must agree exactly.
void TestDeterminism(uint64_t seed) {
  auto game = LoadGame("dune_imperium");
  auto state = game->NewInitialState();
  BeliefOptions options;
  options.num_particles = 8;
  options.parallel_tasks = 4;
  // A model that depends on the particle, so weights genuinely differ.
  OpponentActionProb model = [](const State& s, Action a) {
    const auto legal = s.LegalActions();
    return (a == legal.front() ? 0.7 : 0.3) / legal.size();
  };
  BeliefParticleFilter first(*state, 0, options, seed, model);
  options.parallel_tasks = 1;
  BeliefParticleFilter second(*state, 0, options, seed, model);
  std::mt19937_64 rng(seed);
  for (int steps = 0; steps < 600 && !state->IsTerminal(); ++steps) {
    const Action action = RandomAction(*state, rng);
    auto before = state->Clone();
    state->ApplyAction(action);
    first.Observe(*before, action, *state);
    second.Observe(*before, action, *state);
  }
  const auto w1 = first.Weights();
  const auto w2 = second.Weights();
  SPIEL_CHECK_EQ(w1.size(), w2.size());
  for (size_t i = 0; i < w1.size(); ++i) {
    SPIEL_CHECK_FLOAT_EQ(w1[i], w2[i]);
    if (first.alive(i)) {
      SPIEL_CHECK_EQ(first.particle(i).ToString(), second.particle(i).ToString());
    }
  }
  for (uint64_t world_seed : {1ULL, 2ULL, 99ULL}) {
    auto a = first.SampleWorld(world_seed);
    auto b = first.SampleWorld(world_seed);
    SPIEL_CHECK_TRUE(a != nullptr);
    SPIEL_CHECK_EQ(a->ToString(), b->ToString());
  }
}

// Temperature 0 leaves live particles equally weighted, so the filter reduces
// to the consistency constraints alone. Full epsilon ignores the model: every
// legal action gets 1/|legal|, so a skewed model and a uniform one give the
// same weights (not equal ones, since particles can differ in legal actions).
void TestUniformLimits(uint64_t seed) {
  auto game = LoadGame("dune_imperium");
  OpponentActionProb skewed = [](const State& s, Action a) {
    return a == s.LegalActions().front() ? 1.0 : 0.0;
  };
  {
    auto state = game->NewInitialState();
    BeliefOptions options;
    options.num_particles = 6;
    options.temperature = 0.0;
    BeliefParticleFilter filter(*state, 1, options, seed, skewed);
    std::mt19937_64 rng(seed);
    for (int steps = 0; steps < 400 && !state->IsTerminal(); ++steps) {
      const Action action = RandomAction(*state, rng);
      auto before = state->Clone();
      state->ApplyAction(action);
      filter.Observe(*before, action, *state);
      const auto w = filter.Weights();
      double live = 0.0;
      for (int i = 0; i < filter.size(); ++i) live += filter.alive(i);
      for (int i = 0; i < filter.size(); ++i) {
        if (filter.alive(i)) SPIEL_CHECK_FLOAT_NEAR(w[i], 1.0 / live, 1e-12);
      }
    }
  }
  {
    auto state = game->NewInitialState();
    BeliefOptions options;
    options.num_particles = 6;
    options.epsilon = 1.0;
    BeliefParticleFilter with_skewed(*state, 1, options, seed, skewed);
    BeliefParticleFilter with_uniform(*state, 1, options, seed, UniformOpponentModel());
    std::mt19937_64 rng(seed + 1);
    for (int steps = 0; steps < 400 && !state->IsTerminal(); ++steps) {
      const Action action = RandomAction(*state, rng);
      auto before = state->Clone();
      state->ApplyAction(action);
      with_skewed.Observe(*before, action, *state);
      with_uniform.Observe(*before, action, *state);
      const auto a = with_skewed.Weights();
      const auto b = with_uniform.Weights();
      for (size_t i = 0; i < a.size(); ++i) SPIEL_CHECK_FLOAT_NEAR(a[i], b[i], 1e-12);
    }
  }
}

// A model certain of one action must only down-weight particles when the
// opponent does something else: epsilon keeps every live weight positive.
void TestWeightsStayFiniteUnderSurprise(uint64_t seed) {
  auto game = LoadGame("dune_imperium");
  auto state = game->NewInitialState();
  BeliefOptions options;
  options.num_particles = 6;
  OpponentActionProb certain = [](const State& s, Action a) {
    return a == s.LegalActions().back() ? 1.0 : 0.0;
  };
  BeliefParticleFilter filter(*state, 2, options, seed, certain);
  std::mt19937_64 rng(seed);
  for (int steps = 0; steps < 400 && !state->IsTerminal(); ++steps) {
    const Action action = RandomAction(*state, rng);
    auto before = state->Clone();
    state->ApplyAction(action);
    filter.Observe(*before, action, *state);
    const auto w = filter.Weights();
    double total = 0.0;
    for (int i = 0; i < filter.size(); ++i) {
      if (filter.alive(i)) SPIEL_CHECK_GT(w[i], 0.0);
      total += w[i];
    }
    SPIEL_CHECK_FLOAT_NEAR(total, 1.0, 1e-9);
  }
  SPIEL_CHECK_GT(filter.stats().opponent_decisions, 0);
}

// With repair on, an opponent card a particle did not hold is moved into
// that particle's hands instead of killing it. Repaired particles must still
// match the observer's view, and repair must actually happen over a game.
void TestRepairKeepsParticlesConsistent(uint64_t seed) {
  auto game = LoadGame("dune_imperium");
  int64_t repaired = 0;
  for (Player observer = 0; observer < 4; ++observer) {
    auto state = game->NewInitialState();
    BeliefOptions options;
    options.num_particles = 8;
    options.parallel_tasks = 4;
    options.repair_opponent_hands = true;
    BeliefParticleFilter filter(*state, observer, options, seed + observer,
                                UniformOpponentModel());
    std::mt19937_64 rng(seed + observer);
    int steps = 0;
    while (!state->IsTerminal() && steps++ < kMaxSteps) {
      const Action action = RandomAction(*state, rng);
      auto before = state->Clone();
      state->ApplyAction(action);
      filter.Observe(*before, action, *state);
      const std::string truth = state->ObservationString(observer);
      int alive = 0;
      for (int i = 0; i < filter.size(); ++i) {
        if (!filter.alive(i)) continue;
        ++alive;
        SPIEL_CHECK_EQ(filter.particle(i).ObservationString(observer), truth);
      }
      SPIEL_CHECK_GT(alive, 0);
    }
    repaired += filter.stats().repaired;
    std::cout << "repair observer " << observer << ": repaired=" << filter.stats().repaired
              << " killed_illegal=" << filter.stats().killed_illegal
              << " killed_inconsistent=" << filter.stats().killed_inconsistent << "\n";
  }
  SPIEL_CHECK_GT(repaired, 0);
}

}  // namespace
}  // namespace dune_belief
}  // namespace open_spiel

int main() {
  using namespace open_spiel::dune_belief;
  for (int observer = 0; observer < 4; ++observer) {
    TestParticlesStayConsistent(observer, 20261013 + observer);
  }
  TestDeterminism(7);
  TestUniformLimits(11);
  TestWeightsStayFiniteUnderSurprise(13);
  TestRepairKeepsParticlesConsistent(17);
  std::cout << "dune_belief_particles_test passed\n";
  return 0;
}
