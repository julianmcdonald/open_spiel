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
#include <chrono>
#include <cmath>
#include <limits>
#include <memory>
#include <random>
#include <utility>
#include <vector>

#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"
// DuneImperiumState + SetSwordmasterForTesting/HasSwordmaster/GetCurrentRound
// for the Swordmaster endowment curriculum (engine is symlinked into the tree;
// same path dune_search_benchmark.cc uses).
#include "open_spiel/games/dune_imperium/dune_imperium.h"
// Pulls in DuneSearchSession/DuneSearchConfig/DuneSearchResult/SearchDiagnostics,
// IsStrategicState, SampleActionFromPrior, DuneDecisionRole/ClassifyDuneDecisionRole,
// and (transitively) open_spiel::algorithms::Evaluator.
#include "dune_search_session.h"

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

// Per-stream seed domains within an auxiliary game. Keeps chance, raw-policy,
// search, and executed-action randomness on independent streams, all separated
// from the PPO and evaluation seed domains by `auxiliary_search_seed_domain`.
constexpr uint64_t kStreamChance = 0x11;
constexpr uint64_t kStreamRawPolicy = 0x22;
constexpr uint64_t kStreamSearchSeed = 0x33;
constexpr uint64_t kStreamControllerR = 0x44;
constexpr uint64_t kStreamAcceptedAction = 0x55;
constexpr uint64_t kStreamSwordmasterGrant = 0x66;

// Deterministic 64-bit draw for (seed_domain, episode_id, index, stream).
uint64_t DeriveStream(uint64_t seed_domain, int64_t episode_id, int64_t index,
                      uint64_t stream) {
  uint64_t h = Splitmix64(seed_domain);
  h = Splitmix64(h ^ static_cast<uint64_t>(episode_id));
  h = Splitmix64(h ^ static_cast<uint64_t>(index));
  h = Splitmix64(h ^ stream);
  return h;
}

// Samples an index into `weights` with the given temperature using r_unit in
// [0, 1). temperature == 1 samples proportionally; temperature <= 0 is greedy
// (argmax). Negative/zero weights are clamped; an all-zero vector falls back to
// uniform. `weights` must be non-empty.
int SampleIndexWithTemperature(const std::vector<double>& weights,
                               double temperature, double r_unit) {
  const size_t n = weights.size();
  SPIEL_CHECK_GT(n, 0);
  if (temperature <= 0.0) {
    int best = 0;
    for (size_t i = 1; i < n; ++i) {
      if (weights[i] > weights[best]) best = static_cast<int>(i);
    }
    return best;
  }
  std::vector<double> w(n);
  double sum = 0.0;
  const bool unit_temp = std::abs(temperature - 1.0) < 1e-12;
  for (size_t i = 0; i < n; ++i) {
    double v = std::max(0.0, weights[i]);
    w[i] = unit_temp ? v : std::pow(v, 1.0 / temperature);
    sum += w[i];
  }
  if (!(sum > 0.0)) {  // all zero (or NaN) -> uniform
    for (size_t i = 0; i < n; ++i) w[i] = 1.0;
    sum = static_cast<double>(n);
  }
  const double threshold = r_unit * sum;
  double acc = 0.0;
  for (size_t i = 0; i < n; ++i) {
    acc += w[i];
    if (threshold <= acc) return static_cast<int>(i);
  }
  return static_cast<int>(n - 1);
}

// Samples a legal action from the raw policy `prior` at `temperature`, using the
// deterministic draw in `seed_bits`. Falls back to the first legal action if the
// prior is empty or degenerate. `temperature == 1` samples directly from prior.
Action SampleRawPolicyAction(const ActionsAndProbs& prior, double temperature,
                             uint64_t seed_bits, const State& state) {
  const double r = UnitDouble(seed_bits);
  Action a = kInvalidAction;
  if (!prior.empty()) {
    if (std::abs(temperature - 1.0) < 1e-12) {
      a = SampleActionFromPrior(prior, r);
    } else {
      std::vector<double> weights;
      weights.reserve(prior.size());
      for (const auto& ap : prior) weights.push_back(ap.second);
      a = prior[SampleIndexWithTemperature(weights, temperature, r)].first;
    }
  }
  if (a == kInvalidAction) {
    std::vector<Action> legal = state.LegalActions();
    if (!legal.empty()) a = legal.front();
  }
  return a;
}

// Item-4 telemetry: KL(normalized visits || prior) and whether the search's most
// visited action differs from the prior's argmax. `visits` and `priors` aligned.
struct VisitPriorDivergence {
  double kl = 0.0;
  bool prior_argmax_override = false;
};
VisitPriorDivergence ComputeVisitPriorDivergence(const std::vector<int>& visits,
                                                 const std::vector<double>& priors) {
  VisitPriorDivergence d;
  if (visits.empty() || visits.size() != priors.size()) return d;
  double total = 0.0;
  for (int v : visits) total += v;
  if (total <= 0.0) return d;
  size_t vmax = 0, pmax = 0;
  for (size_t i = 0; i < visits.size(); ++i) {
    double p = static_cast<double>(visits[i]) / total;
    if (p > 0.0) d.kl += p * std::log(p / std::max(priors[i], 1e-12));
    if (visits[i] > visits[vmax]) vmax = i;
    if (priors[i] > priors[pmax]) pmax = i;
  }
  d.prior_argmax_override = (vmax != pmax);
  return d;
}

}  // namespace

// KataGo policy-target pruning (§3.2 of 1902.10565). Returns pruned root visit
// counts to renormalize into the CE target. For every child except the
// most-visited c*, remove up to n_forced(c) = floor(sqrt(k * P_noised(c) *
// N_root)) visits, but never so many that the child's PUCT value would reach
// c*'s at the final stats (those visits were earned, not forced); then drop any
// non-c* child left with a single visit. `priors` are the tree (post-noise)
// priors, aligned with `visits` and `q_values`. Only meaningful when the
// exploration package ran (forced_k > 0); callers gate on that.
std::vector<int> PruneForcedPlayouts(const std::vector<int>& visits,
                                     const std::vector<double>& priors,
                                     const std::vector<double>& q_values,
                                     int total_root_visits, double puct_c,
                                     double forced_k) {
  const size_t n = visits.size();
  std::vector<int> pruned = visits;
  if (n <= 1) return pruned;

  int cstar = 0;
  double sum_v = 0.0;
  for (size_t i = 0; i < n; ++i) {
    sum_v += visits[i];
    if (visits[i] > visits[cstar]) cstar = static_cast<int>(i);
  }
  double N = (total_root_visits > 0) ? static_cast<double>(total_root_visits) : sum_v;
  const double sqrtN = std::sqrt(std::max(1.0, N));

  auto puct = [&](int idx, int v) {
    return q_values[idx] +
           puct_c * priors[idx] * sqrtN / (1.0 + static_cast<double>(v));
  };
  const double puct_cstar = puct(cstar, visits[cstar]);

  for (size_t c = 0; c < n; ++c) {
    if (static_cast<int>(c) == cstar || visits[c] <= 0) continue;
    const int n_forced =
        static_cast<int>(std::floor(std::sqrt(forced_k * priors[c] * N)));
    int v = visits[c];
    int removed = 0;
    // Remove forced playouts one at a time; stop before the child would become
    // PUCT-competitive with c* (removing a visit shrinks the denominator, raising
    // U and hence PUCT).
    while (removed < n_forced && v > 1 &&
           puct(static_cast<int>(c), v - 1) < puct_cstar) {
      --v;
      ++removed;
    }
    pruned[c] = v;
  }
  // Drop non-c* children reduced to a single visit (single-playout children are
  // exploration noise, not signal).
  for (size_t c = 0; c < n; ++c) {
    if (static_cast<int>(c) != cstar && pruned[c] == 1) pruned[c] = 0;
  }
  return pruned;
}

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
  SPIEL_CHECK_GE(config_.swordmaster_grant_fraction, 0.0);
  SPIEL_CHECK_LE(config_.swordmaster_grant_fraction, 1.0);
  SPIEL_CHECK_GE(config_.swordmaster_grant_round, 1);
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
    int update_id, const std::shared_ptr<const Game>& game,
    const std::shared_ptr<algorithms::Evaluator>& evaluator,
    std::vector<SearchTrainingExample>* out,
    OnlineSearchCollectionStats* stats) {
  SPIEL_CHECK_TRUE(out != nullptr);
  SPIEL_CHECK_TRUE(stats != nullptr);
  SPIEL_CHECK_TRUE(game != nullptr);
  SPIEL_CHECK_TRUE(evaluator != nullptr);

  const auto wall_start = std::chrono::steady_clock::now();
  const int num_players = game->NumPlayers();
  SPIEL_CHECK_GT(num_players, 0);
  const bool provides_istate = game->GetType().provides_information_state_tensor;

  // Reset stats; record the episode window collected this update.
  *stats = OnlineSearchCollectionStats();
  stats->auxiliary_games = config_.auxiliary_games;
  const int64_t first_episode_id = config_.next_auxiliary_episode_id;
  stats->first_episode_id = first_episode_id;

  // Frozen teacher search config, mirroring the Step 2 benchmark's
  // DuneSearchConfig field-for-field (dune_search_benchmark.cc:301-335) with the
  // 64/8 *training* budget rather than Step 2's 200/0 confirmation budget. The
  // baseline critic is the evaluator itself (no `nonlinear_value_head` knob
  // exists here; the ctor enforces the collector-config guard). We deliberately
  // do NOT call LoadCalibratedParameters(): the frozen benchmark sets every
  // field explicitly and so do we, so no manifest can silently move puct etc.
  DuneSearchConfig base_cfg;
  base_cfg.max_simulations = config_.max_simulations;                 // 64
  base_cfg.fixed_session_limit = config_.max_simulations;            // 64
  base_cfg.fixed_continuation_reserve = config_.fixed_continuation_reserve;  // 0 (uniform 64)
  base_cfg.relative_time_budget_ms = std::numeric_limits<double>::infinity();
  base_cfg.max_nodes = 50000;
  base_cfg.puct_c = config_.puct_c;                                  // 0.30
  base_cfg.opponent_mode = config_.use_opponent_model
                               ? SearchOpponentMode::kPolicy
                               : SearchOpponentMode::kMaxN;
  base_cfg.temperature = 1.0;
  base_cfg.opponent_temperature = config_.simulated_opponent_temperature;  // 1.0
  base_cfg.max_world_samples = -1;
  base_cfg.utility_divisor = config_.utility_divisor;                // 4.0
  base_cfg.min_visit_threshold = config_.min_visits_per_action;      // 2
  base_cfg.covered_prior_threshold = config_.min_prior_mass;         // 0.50
  base_cfg.final_policy_type = DuneISMCTSFinalPolicyType::kNormalizedVisitCount;
  base_cfg.use_observation_string = true;
  base_cfg.verbose_diagnostics = false;
  // Matches the frozen Step 2 config. It only bypasses search at NON-strategic
  // states; every kAgentPrimary root we search is strategic (it offers a
  // SelectAgentCard/PlayKwisatzHaderach action), so this is a no-op at our roots.
  base_cfg.check_strategic_state = true;
  base_cfg.root_prior_temperature = config_.root_prior_temperature;  // 1.0
  base_cfg.max_search_decision_depth = config_.max_search_decision_depth;  // -1 (uncapped)
  base_cfg.purchase_combat_budget = config_.purchase_combat_budget;   // 64 (uniform)
  base_cfg.conservative_override_enabled = false;
  // KataGo root-exploration package (Phase 18B). A unit with the collector-side
  // visit-target pruning below. Disabled when dirichlet_epsilon == 0 (Step-3 probe).
  base_cfg.dirichlet_epsilon = config_.dirichlet_epsilon;          // 0.25
  base_cfg.dirichlet_alpha_total = config_.dirichlet_alpha_total;  // 10.83 -> alpha=total/N_legal
  base_cfg.forced_playouts_k = config_.forced_playouts_k;          // 2.0
  base_cfg.root_noise_fpu_zero = config_.root_noise_fpu_zero;      // true

  double sum_sims = 0.0;
  double sum_covered_mass = 0.0;

  for (int g = 0; g < config_.auxiliary_games; ++g) {
    const int64_t episode_id = first_episode_id + g;
    const Player searched_seat = SearchedSeatForEpisode(episode_id, num_players);

    // Swordmaster endowment curriculum: deterministic per-episode selection on
    // its own stream (no draw at all when the fraction is 0, so the existing
    // chance/raw-policy/search/controller/accepted-action streams and the whole
    // collection are byte-identical to today). Selected games grant the searched
    // seat a free Swordmaster once, at the hook below.
    bool grant_selected = false;
    bool grant_fired = false;
    if (config_.swordmaster_grant_fraction > 0.0) {
      grant_selected =
          UnitDouble(DeriveStream(config_.auxiliary_search_seed_domain, episode_id,
                                  /*index=*/0, kStreamSwordmasterGrant)) <
          config_.swordmaster_grant_fraction;
    }

    std::unique_ptr<State> state = game->NewInitialState();
    int64_t decision_index = 0;  // global index over non-chance decisions (any seat)
    int64_t chance_index = 0;
    std::vector<size_t> emitted_indices;  // positions in *out emitted this game
    int64_t guard = 0;

    while (!state->IsTerminal()) {
      if (++guard > 200000) {
        SpielFatalError(
            "OnlineSearchCollector::CollectUpdate exceeded the per-game decision "
            "guard; the game did not terminate.");
      }

      // --- Chance node: sample from a private, domain-separated stream. ---
      if (state->IsChanceNode()) {
        ActionsAndProbs outcomes = state->ChanceOutcomes();
        double r = UnitDouble(DeriveStream(config_.auxiliary_search_seed_domain,
                                           episode_id, chance_index, kStreamChance));
        ++chance_index;
        state->ApplyAction(SampleAction(outcomes, r).first);
        continue;
      }

      const Player cur = state->CurrentPlayer();
      SPIEL_CHECK_GE(cur, 0);  // Dune is sequential: no simultaneous nodes.
      const int64_t this_decision = decision_index++;

      // Swordmaster endowment curriculum: grant the searched seat a free
      // Swordmaster at its FIRST decision of the grant round, before it places
      // any agent that round. Fires once; persistence is the engine's permanent
      // third agent (c8b3bf6). No solari deducted, no action forced.
      if (grant_selected && !grant_fired && cur == searched_seat) {
        if (auto* gr_state =
                dynamic_cast<dune_imperium::DuneImperiumState*>(state.get())) {
          if (gr_state->GetCurrentRound() == config_.swordmaster_grant_round &&
              !gr_state->HasSwordmaster(searched_seat)) {
            gr_state->SetSwordmasterForTesting(searched_seat, true);
            grant_fired = true;
            ++stats->swordmaster_granted_games;
          }
        }
      }

      // --- Non-searched seat: raw stochastic policy @ non_search_temperature. ---
      if (cur != searched_seat) {
        ActionsAndProbs prior = evaluator->Prior(*state);
        state->ApplyAction(SampleRawPolicyAction(
            prior, config_.non_search_temperature,
            DeriveStream(config_.auxiliary_search_seed_domain, episode_id,
                         this_decision, kStreamRawPolicy),
            *state));
        continue;
      }

      // --- Searched seat. ---
      const DuneDecisionRole role =
          ClassifyDuneDecisionRole(*state, cur, /*has_active_session=*/false);
      // Phase 18B search surface: agent-phase primaries + continuations AND
      // purchases. The arm-4 decomposition (25.0% = policy-only, 0% of the
      // policy->full gap recovered) showed the search lever lives in the
      // non-agent-phase decisions, so purchases stay in. Combat/other-optional
      // are a later tranche, not this pilot.
      const bool is_search_target =
          role == DuneDecisionRole::kAgentPrimary ||
          role == DuneDecisionRole::kAgentContinuation ||
          role == DuneDecisionRole::kPurchase;
      // Per-role telemetry index: 0=primary, 1=continuation, 2=purchase, else -1.
      const int role_idx =
          role == DuneDecisionRole::kAgentPrimary      ? 0
          : role == DuneDecisionRole::kAgentContinuation ? 1
          : role == DuneDecisionRole::kPurchase          ? 2
                                                         : -1;

      Action chosen = kInvalidAction;  // set by an accepted search; else raw below.

      if (is_search_target) {
        ++stats->strategic_roots_seen;
        if (role_idx >= 0) ++stats->by_role[role_idx].roots_seen;
        if (ShouldSearchAtRoot(episode_id, this_decision)) {
          ++stats->searches_selected;
          if (role_idx >= 0) ++stats->by_role[role_idx].searches;

          // Fresh, cold session per root: one UNIFORM 64-sim search regardless of
          // role (reserve 0, purchase_combat_budget 64). A cold session inherits
          // nothing, so this reproduces the teacher's search at that root exactly
          // while keeping the Bernoulli-sampled roots independent — no
          // persistent-session re-root state to corrupt across skipped roots.
          DuneSearchConfig cfg = base_cfg;
          cfg.seed = DeriveStream(config_.auxiliary_search_seed_domain, episode_id,
                                  this_decision, kStreamSearchSeed);
          DuneSearchSession session(cfg, evaluator,
                                    DuneSearchBudgetMode::kFixedSessionSimulations);
          session.SetEpisodeId(static_cast<int>(episode_id));
          session.SetUpdateId(update_id);

          double r_ctrl = UnitDouble(DeriveStream(config_.auxiliary_search_seed_domain,
                                                  episode_id, this_decision,
                                                  kStreamControllerR));
          DuneSearchResult res = session.SearchAndSelect(*state, r_ctrl);
          const SearchDiagnostics& diag = res.diagnostics;

          stats->inference_calls += res.inference_count;
          if (res.timeout_status) ++stats->timeouts;
          sum_sims += res.simulations_completed;

          // Item-4 per-role telemetry: KL(visits||prior) + prior-argmax override,
          // over every searched root (accepted or not) for comparability with the
          // 200-sim role baselines (primary 0.43/24%, cont 0.51/25%, purch 0.62/29%).
          // The baselines are vs the RAW network prior, so compare against the
          // pre-noise prior (diag.raw_priors) when the exploration package noised
          // the root; diag.priors (post-noise) would inflate KL and corrupt the
          // budget decision in item 4. Falls back to diag.priors when no noise ran
          // (then priors already equals the raw prior).
          if (role_idx >= 0) {
            const std::vector<double>& telemetry_priors =
                diag.raw_priors.empty() ? diag.priors : diag.raw_priors;
            VisitPriorDivergence d =
                ComputeVisitPriorDivergence(diag.visit_counts, telemetry_priors);
            stats->by_role[role_idx].sum_kl += d.kl;
            if (d.prior_argmax_override) {
              ++stats->by_role[role_idx].prior_argmax_overrides;
            }
          }

          // Acceptance is recomputed from the visit counts + root priors so the
          // collector, not the controller, owns the accept/reject decision.
          const int legal_count = static_cast<int>(diag.actions.size());
          int covered = 0;
          double covered_mass = 0.0;
          bool accept = false;
          if (legal_count > 0 && diag.visit_counts.size() == diag.priors.size() &&
              diag.actions.size() == diag.visit_counts.size()) {
            accept = AcceptSearch(diag.visit_counts, diag.priors, legal_count,
                                  &covered, &covered_mass);
          }
          sum_covered_mass += covered_mass;

          if (accept) {
            // Raw normalized visit distribution: used to EXECUTE the auxiliary-game
            // move (faithful to the search's actual policy).
            double total = 0.0;
            for (int v : diag.visit_counts) total += v;
            SPIEL_CHECK_GT(total, 0.0);  // acceptance implies visited actions.
            std::vector<double> raw_normalized(diag.visit_counts.size());
            for (size_t i = 0; i < diag.visit_counts.size(); ++i) {
              raw_normalized[i] = static_cast<double>(diag.visit_counts[i]) / total;
            }

            // CE target: KataGo visit-target pruning when the exploration package
            // ran (forced playouts present); otherwise the raw visit distribution.
            std::vector<double> target = raw_normalized;
            const bool package_active =
                config_.forced_playouts_k > 0.0 && config_.dirichlet_epsilon > 0.0 &&
                diag.q_values.size() == diag.visit_counts.size();
            if (package_active) {
              std::vector<int> pruned = PruneForcedPlayouts(
                  diag.visit_counts, diag.priors, diag.q_values,
                  diag.total_root_visits, config_.puct_c, config_.forced_playouts_k);
              double ptot = 0.0;
              for (int v : pruned) ptot += v;
              if (ptot > 0.0) {
                for (size_t i = 0; i < pruned.size(); ++i) {
                  target[i] = static_cast<double>(pruned[i]) / ptot;
                }
              }
            }

            SearchTrainingExample ex;
            ex.observation = provides_istate ? state->InformationStateTensor(cur)
                                             : state->ObservationTensor(cur);
            ex.player = cur;
            ex.legal_actions = diag.actions;      // aligned to normalized_visits
            ex.normalized_visits = target;        // CE target (pruned when noised)
            ex.value_target = 0.0;                // attached at terminal
            ex.value_target_attached = false;
            ex.checkpoint_hash = checkpoint_hash_;
            ex.update_id = update_id;
            ex.episode_id = episode_id;
            ex.decision_id = static_cast<int>(this_decision);
            ex.simulations_completed = res.simulations_completed;
            ex.num_covered_actions = covered;
            ex.covered_prior_mass = covered_mass;
            ex.mean_depth = diag.mean_depth;
            ex.terminal_leaf_fraction = diag.terminal_leaf_fraction;

            emitted_indices.push_back(out->size());
            out->push_back(std::move(ex));
            ++stats->accepted_targets;
            if (role_idx >= 0) ++stats->by_role[role_idx].accepted;

            // Execute an action sampled from the RAW visit distribution (the
            // search's actual policy), independent of the target pruning above.
            double r_act = UnitDouble(DeriveStream(config_.auxiliary_search_seed_domain,
                                                  episode_id, this_decision,
                                                  kStreamAcceptedAction));
            int idx = SampleIndexWithTemperature(
                raw_normalized, config_.accepted_action_temperature, r_act);
            chosen = diag.actions[idx];
          } else {
            ++stats->rejected_incomplete;
            ++stats->fallback_raw_policy;  // rejected search -> raw policy executed
          }
        }
      }

      // Raw policy for: non-strategic roles, unselected roots, rejected searches.
      if (chosen == kInvalidAction) {
        ActionsAndProbs prior = evaluator->Prior(*state);
        chosen = SampleRawPolicyAction(
            prior, config_.non_search_temperature,
            DeriveStream(config_.auxiliary_search_seed_domain, episode_id,
                         this_decision, kStreamRawPolicy),
            *state);
      }

      state->ApplyAction(chosen);
    }

    // Terminal: attach the placement value target to this game's examples.
    std::vector<double> returns = state->Returns();
    for (size_t idx : emitted_indices) {
      SearchTrainingExample& ex = (*out)[idx];
      SPIEL_CHECK_GE(ex.player, 0);
      SPIEL_CHECK_LT(ex.player, static_cast<int>(returns.size()));
      ex.value_target = returns[ex.player] / config_.utility_divisor;
      ex.value_target_attached = true;
    }

    // Swordmaster endowment curriculum ignition metric: the searched seat owns
    // Swordmaster at terminal WITHOUT a grant firing this game (organic
    // acquisition). Read-only; gated on the feature being active so at fraction
    // 0.0 no engine call is made and behavior is byte-identical to today.
    if (config_.swordmaster_grant_fraction > 0.0) {
      if (auto* gr_state = dynamic_cast<const dune_imperium::DuneImperiumState*>(
              state.get())) {
        if (gr_state->HasSwordmaster(searched_seat) && !grant_fired) {
          ++stats->swordmaster_organic_games;
        }
      }
    }
  }

  // Advance the episode cursor (persisted for resume) and finalize stats.
  stats->next_episode_id = first_episode_id + config_.auxiliary_games;
  config_.next_auxiliary_episode_id = stats->next_episode_id;
  if (stats->searches_selected > 0) {
    stats->mean_simulations_completed =
        sum_sims / static_cast<double>(stats->searches_selected);
    stats->mean_covered_prior_mass =
        sum_covered_mass / static_cast<double>(stats->searches_selected);
  }
  stats->collection_wall_time_s =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - wall_start)
          .count();
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
