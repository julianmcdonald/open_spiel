// Phase 18B online auxiliary-search collector — see the header and
// docs/PHASE_18B_ONLINE_COLLECTOR_DESIGN.md.
//
// Status: the deterministic/pure logic (seat rotation, domain-separated 0.25
// Bernoulli, acceptance test, loss-coef warmup) is implemented and unit-tested
// (dune_online_search_collector_test.cc). CollectUpdate — which drives the
// frozen search session over auxiliary games — is scaffolded; wiring it to the
// DuneSearchSession + frozen evaluator is the next increment (see the design
// doc "CollectUpdate wiring" section).

#include "open_spiel/examples/dune_online_search_collector.h"

#include <algorithm>
#include <cmath>
#include <utility>

#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace {

// Deterministic, seed-domain-separated mixing. splitmix64 finalizer.
uint64_t Splitmix64(uint64_t x) {
  x += 0x9E3779B97F4A7C15ULL;
  x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
  x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
  return x ^ (x >> 31);
}

// Uniform double in [0, 1) from a 64-bit state (top 53 bits).
double UnitDouble(uint64_t bits) {
  return static_cast<double>(bits >> 11) * (1.0 / 9007199254740992.0);  // 2^53
}

}  // namespace

OnlineSearchCollector::OnlineSearchCollector(const OnlineSearchConfig& config,
                                             std::string checkpoint_hash)
    : config_(config), checkpoint_hash_(std::move(checkpoint_hash)) {
  SPIEL_CHECK_GT(config_.auxiliary_games, 0);
  SPIEL_CHECK_EQ(config_.auxiliary_games % 4, 0);  // seat balance
  SPIEL_CHECK_GE(config_.search_probability, 0.0);
  SPIEL_CHECK_LE(config_.search_probability, 1.0);
  SPIEL_CHECK_GT(config_.max_simulations, config_.fixed_continuation_reserve);
  SPIEL_CHECK_LT(config_.max_search_decision_depth, 0);  // 18B is uncapped
  SPIEL_CHECK_EQ(config_.nonlinear_value_head, false);   // 18B uses baseline critic
}

Player OnlineSearchCollector::SearchedSeatForEpisode(int64_t episode_id,
                                                     int num_players) {
  SPIEL_CHECK_GT(num_players, 0);
  // Rotate one searched seat by auxiliary episode id.
  return static_cast<Player>(((episode_id % num_players) + num_players) %
                             num_players);
}

bool OnlineSearchCollector::ShouldSearchAtRoot(int64_t episode_id,
                                               int decision_id) const {
  // Domain-separated deterministic Bernoulli(search_probability), keyed by
  // (auxiliary_search_seed_domain, episode_id, decision_id). Independent of the
  // PPO, evaluation, chance, and opponent seed streams.
  uint64_t h = Splitmix64(config_.auxiliary_search_seed_domain);
  h = Splitmix64(h ^ static_cast<uint64_t>(episode_id));
  h = Splitmix64(h ^ static_cast<uint64_t>(decision_id));
  return UnitDouble(h) < config_.search_probability;
}

bool OnlineSearchCollector::AcceptSearch(const std::vector<int>& visit_counts,
                                         const std::vector<double>& root_priors,
                                         int legal_count, int* num_covered_out,
                                         double* covered_prior_mass_out) const {
  SPIEL_CHECK_EQ(visit_counts.size(), root_priors.size());
  int num_covered = 0;
  double covered_prior_mass = 0.0;
  for (size_t i = 0; i < visit_counts.size(); ++i) {
    if (visit_counts[i] >= config_.min_visits_per_action) {
      ++num_covered;
      covered_prior_mass += root_priors[i];
    }
  }
  if (num_covered_out != nullptr) *num_covered_out = num_covered;
  if (covered_prior_mass_out != nullptr) {
    *covered_prior_mass_out = covered_prior_mass;
  }
  const int required = std::min(config_.min_coverage, legal_count);
  return num_covered >= required && covered_prior_mass >= config_.min_prior_mass;
}

void OnlineSearchCollector::CollectUpdate(
    int update_id, std::vector<SearchTrainingExample>* out,
    OnlineSearchCollectionStats* stats) {
  SPIEL_CHECK_TRUE(out != nullptr);
  SPIEL_CHECK_TRUE(stats != nullptr);
  // Algorithm (see design doc "CollectUpdate wiring"):
  //   for g in [0, auxiliary_games):
  //     episode_id = config_.next_auxiliary_episode_id + g
  //     searched_seat = SearchedSeatForEpisode(episode_id, num_players)
  //     play a full game from the frozen snapshot; at each decision:
  //       - non-searched seats: raw stochastic policy @ non_search_temperature
  //       - searched seat, non-strategic root: raw policy
  //       - searched seat, strategic root:
  //           if ShouldSearchAtRoot(episode_id, decision_id):
  //             run DuneSearchSession (64/8, puct 0.30, policy opponent @1.0,
  //               uncapped, baseline critic); read diag.visit_counts/priors
  //             if AcceptSearch(...): emit SearchTrainingExample (normalized
  //               visits as CE target; value_target attached at terminal), then
  //               execute action sampled from normalized visits @ temp 1.0
  //             else: raw policy (no example); stats->rejected_incomplete++
  //           else: raw policy
  //     at terminal: attach value_target = terminal_utility(seat)/utility_divisor
  //       to that game's emitted examples.
  //   stats->next_episode_id = next_auxiliary_episode_id + auxiliary_games
  SpielFatalError(
      "OnlineSearchCollector::CollectUpdate is not wired yet: connect the "
      "frozen evaluator + DuneSearchSession per the design doc before use.");
}

double SearchLossCoefForUpdate(int update_id, double target, int warmup_update) {
  if (warmup_update <= 1) return target;
  if (update_id <= 1) return 0.0;
  if (update_id >= warmup_update) return target;
  // Linear from 0.0 at update 1 to `target` at `warmup_update`.
  return target * static_cast<double>(update_id - 1) /
         static_cast<double>(warmup_update - 1);
}

}  // namespace open_spiel
