// Belief-weighted search worlds (C4) without engine changes.
//
// The search samples opponents' hidden cards with ResampleFromInfostate, which
// is uniform over the legally consistent hidden pools. This filter instead
// carries N complete game states ("particles") alongside the real game:
//
// - Every real action is applied to every particle with the native ApplyAction.
// - A chance outcome the observer can see (its own draws, market refills,
//   conflict cards) is copied. A hidden one (an opponent's draw) is sampled
//   independently inside each particle, so no hidden truth enters a particle.
// - After every action a particle must give the observer exactly the true
//   ObservationString (WO-31: exact and privacy-filtered); otherwise it dies.
// - At each opponent decision a particle's weight is multiplied by the opponent
//   model's probability of the action the opponent really took, evaluated on
//   that opponent's view of the particle. With a uniform model the filter keeps
//   only the consistency constraints, which is the matched baseline.
// - When every particle dies (an opponent's reveal often does this), the filter
//   rebuilds from the true state with ResampleFromInfostate, so it degrades to
//   today's uniform resampler and never to anything worse.
// - Optional repair (off by default): a particle that an opponent's action
//   would kill first moves the cards that action made public (played or
//   revealed Imperium cards, discarded intrigues) into that opponent's hand
//   from the same hidden pool, then retries the action. It survives only if
//   the retry is legal and matches the true observation. This is an
//   approximation: strict conditioning would discard the particle.
//
// SampleWorld(seed) is deterministic in its seed, so candidates that share a
// world seed share a world, which keeps C1's paired worlds.

#ifndef OPEN_SPIEL_EXAMPLES_DUNE_BELIEF_PARTICLES_H_
#define OPEN_SPIEL_EXAMPLES_DUNE_BELIEF_PARTICLES_H_

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <future>
#include <iterator>
#include <limits>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "open_spiel/examples/dune_seed_utils.h"
#include "open_spiel/games/dune_imperium/dune_imperium.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace dune_belief {

struct BeliefOptions {
  int num_particles = 128;
  // Exponent on the opponent model's probability; 0 reduces to the uniform filter.
  double temperature = 1.0;
  // Mix the model with uniform over legal actions so one surprising move cannot
  // zero a particle's weight.
  double epsilon = 0.05;
  // Attempts to draw a hidden chance outcome consistent with the observation.
  int max_hidden_draw_tries = 16;
  // Resample when the effective sample size falls below this fraction of N.
  double resample_ess_fraction = 0.5;
  // Number of concurrent tasks used to update particles.
  int parallel_tasks = 16;
  // Repair particles an opponent's action would kill (see the header comment).
  bool repair_opponent_hands = false;
};

struct BeliefStats {
  int64_t observed_actions = 0;
  int64_t hidden_chance_nodes = 0;
  int64_t opponent_decisions = 0;
  int64_t killed_illegal = 0;
  int64_t killed_inconsistent = 0;
  int64_t resamples = 0;
  int64_t rebuilds = 0;
  int64_t repaired = 0;
};

// Probability of `action` for the current player of `state` under the opponent
// model. Must be safe to call concurrently.
using OpponentActionProb = std::function<double(const State& state, Action action)>;

inline OpponentActionProb UniformOpponentModel() {
  return [](const State& state, Action) {
    return 1.0 / static_cast<double>(std::max<size_t>(1, state.LegalActions().size()));
  };
}

class BeliefParticleFilter {
 public:
  // `initial_state` must contain nothing hidden from `observer` that later
  // play depends on (the game's NewInitialState qualifies). Use
  // RebuildFromTruth to start mid-game.
  BeliefParticleFilter(const State& initial_state, Player observer,
                       BeliefOptions options, uint64_t seed,
                       OpponentActionProb model)
      : observer_(observer), options_(options), seed_(seed),
        model_(std::move(model)) {
    SPIEL_CHECK_GT(options_.num_particles, 0);
    particles_.reserve(options_.num_particles);
    for (int i = 0; i < options_.num_particles; ++i) {
      particles_.push_back(initial_state.Clone());
    }
    log_weights_.assign(options_.num_particles, 0.0);
    ReseedRngs();
  }

  Player observer() const { return observer_; }
  const BeliefStats& stats() const { return stats_; }
  int size() const { return static_cast<int>(particles_.size()); }
  bool alive(int i) const { return std::isfinite(log_weights_[i]); }
  const State& particle(int i) const { return *particles_[i]; }

  // Normalized weights; dead particles get zero.
  std::vector<double> Weights() const {
    std::vector<double> w(particles_.size(), 0.0);
    double max_lw = -std::numeric_limits<double>::infinity();
    for (double lw : log_weights_) max_lw = std::max(max_lw, lw);
    if (!std::isfinite(max_lw)) return w;
    double total = 0.0;
    for (size_t i = 0; i < w.size(); ++i) {
      w[i] = std::isfinite(log_weights_[i]) ? std::exp(log_weights_[i] - max_lw) : 0.0;
      total += w[i];
    }
    for (double& x : w) x /= total;
    return w;
  }

  double EffectiveSampleSize() const {
    double sq = 0.0;
    for (double x : Weights()) sq += x * x;
    return sq > 0.0 ? 1.0 / sq : 0.0;
  }

  // Applies one real action. `before` and `after` are the true states around it.
  void Observe(const State& before, Action action, const State& after) {
    ++stats_.observed_actions;
    const std::string truth_obs = after.ObservationString(observer_);
    const bool chance = before.IsChanceNode();
    const bool hidden = chance && IsHiddenChance(before, action, truth_obs);
    if (hidden) ++stats_.hidden_chance_nodes;
    const Player actor = chance ? kChancePlayerId : before.CurrentPlayer();
    const bool weigh = !chance && actor != observer_ &&
                       before.LegalActions().size() > 1 &&
                       options_.temperature != 0.0;
    if (weigh) ++stats_.opponent_decisions;
    const bool repair = options_.repair_opponent_hands && !chance && actor != observer_;
    const ShownCards shown = repair ? CardsShownBy(before, after, actor) : ShownCards{};

    // 0 ok, 1 illegal, 2 inconsistent, 3 repaired
    std::vector<int> outcome(particles_.size(), 0);
    auto update = [&](size_t begin, size_t end) {
      for (size_t i = begin; i < end; ++i) {
        if (!std::isfinite(log_weights_[i])) continue;
        outcome[i] = UpdateParticle(i, action, chance, hidden, actor, weigh, truth_obs,
                                    repair ? &shown : nullptr);
      }
    };
    RunParallel(update);

    for (size_t i = 0; i < outcome.size(); ++i) {
      if (outcome[i] == 1) ++stats_.killed_illegal;
      if (outcome[i] == 2) ++stats_.killed_inconsistent;
      if (outcome[i] == 3) ++stats_.repaired;
    }
    ResampleOrRebuild(after);
  }

  // Replaces every particle with a uniform resample of the true state. This is
  // exactly today's world distribution.
  void RebuildFromTruth(const State& truth) {
    ++stats_.rebuilds;
    const auto* dune = dynamic_cast<const dune_imperium::DuneImperiumState*>(&truth);
    for (size_t i = 0; i < particles_.size(); ++i) {
      std::mt19937_64 rng(dune_seed::DeriveSeed(seed_, kRebuildStream, stats_.rebuilds, i));
      auto draw = [&rng]() { return std::generate_canonical<double, 53>(rng); };
      std::unique_ptr<State> world = dune ? dune->ResampleFromInfostate(observer_, draw) : nullptr;
      particles_[i] = world ? std::move(world) : truth.Clone();
      log_weights_[i] = 0.0;
    }
    ReseedRngs();
  }

  // One world drawn by weight. The same seed always gives the same world.
  // Returns nullptr only if no particle is alive.
  std::unique_ptr<State> SampleWorld(uint64_t world_seed) const {
    const std::vector<double> w = Weights();
    std::mt19937_64 rng(dune_seed::DeriveSeed(seed_, kWorldStream, world_seed));
    const double u = std::generate_canonical<double, 53>(rng);
    double cumulative = 0.0;
    int last_alive = -1;
    for (size_t i = 0; i < w.size(); ++i) {
      if (w[i] <= 0.0) continue;
      last_alive = static_cast<int>(i);
      cumulative += w[i];
      if (u < cumulative) return particles_[i]->Clone();
    }
    return last_alive >= 0 ? particles_[last_alive]->Clone() : nullptr;
  }

 private:
  static constexpr uint64_t kRebuildStream = 0x42454C52;   // "BELR"
  static constexpr uint64_t kResampleStream = 0x42454C53;  // "BELS"
  static constexpr uint64_t kWorldStream = 0x42454C57;     // "BELW"
  static constexpr uint64_t kParticleStream = 0x42454C50;  // "BELP"

  // A chance outcome is hidden from the observer when a different legal
  // outcome would leave the observer's exact observation unchanged.
  bool IsHiddenChance(const State& before, Action action,
                      const std::string& truth_obs) const {
    int alternatives = 0;
    for (const auto& [outcome, probability] : before.ChanceOutcomes()) {
      if (outcome == action || probability <= 0.0) continue;
      auto alternative = before.Clone();
      alternative->ApplyAction(outcome);
      if (alternative->ObservationString(observer_) != truth_obs) return false;
      if (++alternatives == 2) break;
    }
    return alternatives > 0;
  }

  // Cards one opponent action made public, read only from public zones of the
  // true states: Imperium cards that entered the actor's played or revealed
  // zones, and intrigues that entered the shared intrigue discard.
  struct ShownCards {
    std::vector<int> hand;
    std::vector<int> intrigues;
  };

  static std::vector<int> AddedCards(std::vector<int> before, std::vector<int> after) {
    std::sort(before.begin(), before.end());
    std::sort(after.begin(), after.end());
    std::vector<int> added;
    std::set_difference(after.begin(), after.end(), before.begin(), before.end(),
                        std::back_inserter(added));
    return added;
  }

  static ShownCards CardsShownBy(const State& before, const State& after, Player actor) {
    const auto* b = dynamic_cast<const dune_imperium::DuneImperiumState*>(&before);
    const auto* a = dynamic_cast<const dune_imperium::DuneImperiumState*>(&after);
    if (b == nullptr || a == nullptr) return {};
    auto public_cards = [actor](const dune_imperium::DuneImperiumState& s) {
      std::vector<int> cards = s.GetPlayedAgentCardsForTesting(actor);
      const std::vector<int>& revealed = s.GetRevealedCardsForTesting(actor);
      cards.insert(cards.end(), revealed.begin(), revealed.end());
      return cards;
    };
    return {AddedCards(public_cards(*b), public_cards(*a)),
            AddedCards(b->GetIntrigueDiscardForTesting(), a->GetIntrigueDiscardForTesting())};
  }

  // Moves `shown` into `actor`'s hands. False if nothing was shown or a
  // rebind fails; the state may then be partly changed and must be dropped.
  static bool RebindShownCards(State* state, Player actor, const ShownCards& shown) {
    auto* dune = dynamic_cast<dune_imperium::DuneImperiumState*>(state);
    if (dune == nullptr || (shown.hand.empty() && shown.intrigues.empty())) return false;
    if (!shown.hand.empty() && !dune->RebindHiddenHandForBelief(actor, shown.hand)) return false;
    return shown.intrigues.empty() || dune->RebindHiddenIntriguesForBelief(actor, shown.intrigues);
  }

  int UpdateParticle(size_t i, Action action, bool chance, bool hidden,
                     Player actor, bool weigh, const std::string& truth_obs,
                     const ShownCards* shown) {
    State& particle = *particles_[i];
    if (chance != particle.IsChanceNode() ||
        (!chance && particle.CurrentPlayer() != actor)) {
      log_weights_[i] = -std::numeric_limits<double>::infinity();
      return 1;
    }
    if (hidden) {
      // Draw this particle's own outcome, conditioned on staying consistent.
      for (int attempt = 0; attempt < options_.max_hidden_draw_tries; ++attempt) {
        auto candidate = particle.Clone();
        const double u = std::generate_canonical<double, 53>(rngs_[i]);
        candidate->ApplyAction(SampleAction(candidate->ChanceOutcomes(), u).first);
        if (candidate->ObservationString(observer_) == truth_obs) {
          particles_[i] = std::move(candidate);
          return 0;
        }
      }
      log_weights_[i] = -std::numeric_limits<double>::infinity();
      return 2;
    }
    // Kept only when a repair may follow, since a failed apply mutates the particle.
    std::unique_ptr<State> original = shown != nullptr ? particle.Clone() : nullptr;
    double log_weight = 0.0;
    int result = ApplyVisible(&particle, action, chance, weigh, truth_obs, &log_weight);
    if (result != 0 && original != nullptr &&
        RebindShownCards(original.get(), actor, *shown) &&
        ApplyVisible(original.get(), action, chance, weigh, truth_obs, &log_weight) == 0) {
      particles_[i] = std::move(original);
      result = 3;
    }
    if (result == 1 || result == 2) {
      log_weights_[i] = -std::numeric_limits<double>::infinity();
      return result;
    }
    log_weights_[i] += log_weight;
    return result;
  }

  // Applies a visible action to `state`. Returns 0 if it stays consistent, 1 if
  // the action is illegal there, 2 if the observation no longer matches.
  // `log_weight` receives the opponent-model term when `weigh` is set.
  int ApplyVisible(State* state, Action action, bool chance, bool weigh,
                   const std::string& truth_obs, double* log_weight) const {
    *log_weight = 0.0;
    const std::vector<Action> legal =
        chance ? std::vector<Action>{} : state->LegalActions();
    if (chance) {
      bool legal_outcome = false;
      for (const auto& [outcome, probability] : state->ChanceOutcomes()) {
        if (outcome == action && probability > 0.0) legal_outcome = true;
      }
      if (!legal_outcome) return 1;
    } else if (std::find(legal.begin(), legal.end(), action) == legal.end()) {
      return 1;
    }
    if (weigh) {
      const double p = model_(*state, action);
      const double mixed = (1.0 - options_.epsilon) * std::max(0.0, p) +
                           options_.epsilon / static_cast<double>(legal.size());
      *log_weight = options_.temperature * std::log(std::max(mixed, 1e-300));
    }
    state->ApplyAction(action);
    return state->ObservationString(observer_) == truth_obs ? 0 : 2;
  }

  void ResampleOrRebuild(const State& truth) {
    const std::vector<double> w = Weights();
    double total = 0.0, sq = 0.0;
    for (double x : w) { total += x; sq += x * x; }
    if (total <= 0.0) {
      RebuildFromTruth(truth);
      return;
    }
    const double ess = 1.0 / sq;
    if (ess >= options_.resample_ess_fraction * particles_.size()) return;
    // Systematic resampling.
    ++stats_.resamples;
    std::mt19937_64 rng(dune_seed::DeriveSeed(seed_, kResampleStream, stats_.resamples));
    const size_t n = particles_.size();
    const double start = std::generate_canonical<double, 53>(rng) / n;
    std::vector<std::unique_ptr<State>> next;
    next.reserve(n);
    double cumulative = w[0];
    size_t source = 0;
    for (size_t k = 0; k < n; ++k) {
      const double target = start + static_cast<double>(k) / n;
      while (cumulative < target && source + 1 < n) cumulative += w[++source];
      while (w[source] <= 0.0 && source + 1 < n) cumulative += w[++source];
      next.push_back(particles_[source]->Clone());
    }
    particles_ = std::move(next);
    log_weights_.assign(n, 0.0);
    ReseedRngs();
  }

  void ReseedRngs() {
    rngs_.clear();
    for (size_t i = 0; i < particles_.size(); ++i) {
      rngs_.emplace_back(dune_seed::DeriveSeed(seed_, kParticleStream,
                                               stats_.resamples, stats_.rebuilds, i));
    }
  }

  template <typename Fn>
  void RunParallel(Fn& fn) {
    const size_t n = particles_.size();
    const size_t tasks = std::max<size_t>(1, std::min<size_t>(options_.parallel_tasks, n));
    if (tasks == 1) { fn(0, n); return; }
    std::vector<std::future<void>> futures;
    const size_t chunk = (n + tasks - 1) / tasks;
    for (size_t begin = 0; begin < n; begin += chunk) {
      const size_t end = std::min(n, begin + chunk);
      futures.push_back(std::async(std::launch::async, [&fn, begin, end]() { fn(begin, end); }));
    }
    for (auto& f : futures) f.get();
  }

  Player observer_;
  BeliefOptions options_;
  uint64_t seed_;
  OpponentActionProb model_;
  std::vector<std::unique_ptr<State>> particles_;
  std::vector<double> log_weights_;
  std::vector<std::mt19937_64> rngs_;
  BeliefStats stats_;
};

}  // namespace dune_belief
}  // namespace open_spiel

#endif  // OPEN_SPIEL_EXAMPLES_DUNE_BELIEF_PARTICLES_H_
