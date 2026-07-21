// Phase 18B online auxiliary-search collector.
//
// Collects search-guided training examples online, during PPO training, using
// the frozen pre-update inference snapshot. Replaces the legacy offline static
// label path (SearchLabel / --search_label_dir in dune_ppo_train.cc), which is
// retained only as an explicitly-disabled legacy offline-distillation mode and
// is mutually exclusive with online collection.
//
// Teacher configuration is FROZEN to the Phase 18A Step 1/Step 2 winner:
// baseline critic, uncapped search. No critic swap (nonlinear_value_head=false),
// no depth cap (max_search_decision_depth=-1). See
// docs/PHASE_18A_STEP1_SCREEN_RESULTS.md and
// docs/PHASE_18B_ONLINE_COLLECTOR_DESIGN.md.

#ifndef OPEN_SPIEL_EXAMPLES_DUNE_ONLINE_SEARCH_COLLECTOR_H_
#define OPEN_SPIEL_EXAMPLES_DUNE_ONLINE_SEARCH_COLLECTOR_H_

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "open_spiel/spiel.h"

namespace open_spiel {

namespace algorithms {
class Evaluator;  // frozen inference snapshot; defined in algorithms/mcts.h
}  // namespace algorithms

// One accepted online search example. Policy target is the normalized visit
// distribution over legal_actions (cross-entropy); value target is the terminal
// placement utility / utility_divisor, attached after the auxiliary game ends.
struct SearchTrainingExample {
  std::vector<float> observation;       // information-state / observation tensor
  Player player = kInvalidPlayer;       // searched seat at this decision
  std::vector<Action> legal_actions;    // legal actions at the root
  std::vector<double> normalized_visits;  // CE target, aligned to legal_actions
  double value_target = 0.0;            // terminal_utility / utility_divisor
  bool value_target_attached = false;   // set true once the game terminates

  // Provenance / reproducibility.
  std::string checkpoint_hash;          // frozen inference snapshot hash
  int update_id = -1;                   // PPO update index
  int64_t episode_id = -1;              // auxiliary episode id (drives seat rotation)
  int decision_id = -1;                 // decision index within the auxiliary game

  // Search diagnostics (accepted-target audit).
  int simulations_completed = 0;
  int num_covered_actions = 0;
  double covered_prior_mass = 0.0;
  double mean_depth = 0.0;
  double terminal_leaf_fraction = 0.0;
};

// Frozen 18B collection configuration. Defaults encode the baseline/uncapped
// teacher and the 64/8 training budget screened in Step 3.
struct OnlineSearchConfig {
  // --- Collection shape ---
  int auxiliary_games = 16;             // per update; multiple of 4 (seat balance)
  double search_probability = 0.25;     // deterministic Bernoulli at strategic roots
  int workers = 1;                      // auxiliary games collected sequentially

  // --- Frozen teacher search budget: a UNIFORM 64 sims at every searched root. ---
  // The collector builds one cold session per root and discards it, so the
  // continuation reserve was dead weight (never spent) and purchases would
  // otherwise inherit the 16-sim short-window default. Reserve 0 + a 64-sim
  // short-window budget give primary, continuation, AND purchase roots exactly 64
  // sims each (see dune_search_session.cc:264-344). Item-4 telemetry monitors
  // per-role KL vs the 200-sim baselines; if a role teaches far below, raise ITS
  // budget (for purchases that means purchase_combat_budget, not max_simulations).
  int max_simulations = 64;             // fixed_session_limit
  int fixed_continuation_reserve = 0;   // reserve unused (one cold session per root)
  int purchase_combat_budget = 64;      // short-window (purchase) roots also get 64
  int max_search_decision_depth = -1;   // UNCAPPED (Step 1 winner). Do not cap.
  double puct_c = 0.30;                 // calibrated (Stage 3 / Step 1 winner)
  bool use_opponent_model = true;       // policy opponent model
  double simulated_opponent_temperature = 1.0;
  double root_prior_temperature = 1.0;
  double utility_divisor = 4.0;
  bool nonlinear_value_head = false;    // baseline critic. Do not swap.

  // --- KataGo root-exploration package (Phase 18B; §3.1-3.2 of 1902.10565). ---
  // A UNIT: root Dirichlet noise + forced playouts + FPU=0 at the noised root
  // (in the search core), paired with visit-target pruning in the collector
  // (noise without pruning pollutes the CE target at the 64-sim budget). Active
  // ONLY when dirichlet_epsilon > 0. Default OFF: the frozen Step-2 teacher is
  // noise-free, so this matches it (and the Step-3 probe, which runs noise-off).
  // The pilot opts in by setting dirichlet_epsilon = 0.25. The other three fields
  // hold pilot-ready values but are inert while dirichlet_epsilon == 0.
  double dirichlet_epsilon = 0.0;        // root Dirichlet noise weight (pilot: 0.25)
  double dirichlet_alpha_total = 10.83;  // per-root alpha = total / N_legal
  double forced_playouts_k = 2.0;        // n_forced(c) = sqrt(k * P_noised(c) * N_root)
  bool root_noise_fpu_zero = true;       // FPU = 0 at the noised root (footnote 3)

  // --- Swordmaster endowment curriculum (Phase 18B follow-on to the arm-B-endow
  // probe). If swordmaster_grant_fraction > 0, a deterministic per-episode draw
  // (domain-separated stream; no draw at all when the fraction is 0) selects
  // that fraction of games; in selected games the SEARCHED seat is granted
  // Swordmaster free via SetSwordmasterForTesting at its first decision of
  // swordmaster_grant_round. Purpose: the critic was trained pre-c8b3bf6 and
  // has never seen third-agent states, so search cannot credit the buy; granted
  // games carry honest terminal value targets for those states. Persistence
  // across rounds is the engine's permanent third agent — no logic here.
  // Default 0.0 = structurally inert.
  double swordmaster_grant_fraction = 0.0;
  int swordmaster_grant_round = 2;  // matches the endowment probe (arm B-endow)

  // --- Non-search seats (in-game opponents during auxiliary games) ---
  double non_search_temperature = 1.0;  // stochastic policy at temperature 1.0

  // --- Acceptance rule (matches the search teacher) ---
  int min_coverage = 3;                 // min(3, legal_count) actions...
  int min_visits_per_action = 2;        // ...with >= 2 visits each
  double min_prior_mass = 0.50;         // covered prior mass >= 0.50
  double accepted_action_temperature = 1.0;  // sample executed action from visits

  // --- Combined-optimization coupling (consumed by the trainer) ---
  double search_loss_coef_target = 0.10;   // hold value after warmup
  int search_loss_coef_warmup_update = 25; // linear 0.0 (update 1) -> target (update 25)
  double abort_grad_norm_ratio = 0.50;     // abort if aux/PPO grad-norm ratio exceeds

  // --- Seeds / domains (isolated from PPO and evaluation domains) ---
  uint64_t auxiliary_search_seed_domain = 0;  // set per run; distinct stream
  int64_t next_auxiliary_episode_id = 0;      // persisted in checkpoint manifest

  // --- Timing / safety ---
  double per_search_timeout_ms = 0.0;   // 0 = no per-search wall timeout
};

// Aggregate collection diagnostics for one update (persisted + logged).
// Item-4 per-role search telemetry: KL(normalized_visits || prior) and the
// prior-argmax override rate, compared against the 200-sim role baselines
// (primary 0.43/24%, continuation 0.51/25%, purchase 0.62/29%). Accumulated over
// every SEARCHED root of that role (not only accepted ones), for comparability.
struct PerRoleSearchStats {
  int roots_seen = 0;              // search-target roots of this role encountered
  int searches = 0;               // Bernoulli-selected (a search ran)
  int accepted = 0;               // passed acceptance -> emitted as a CE target
  double sum_kl = 0.0;            // sum over `searches` of KL(visits || prior)
  int prior_argmax_overrides = 0; // searches where argmax(visits) != argmax(prior)
};

struct OnlineSearchCollectionStats {
  int auxiliary_games = 0;
  int strategic_roots_seen = 0;         // search-target roots (primary+cont+purchase)
  int searches_selected = 0;            // Bernoulli(0.25) selected
  int accepted_targets = 0;
  int rejected_incomplete = 0;          // searched but failed acceptance
  // Per-role telemetry, index 0=primary, 1=continuation, 2=purchase.
  PerRoleSearchStats by_role[3];
  // Swordmaster endowment curriculum telemetry.
  int64_t swordmaster_granted_games = 0;  // grants that actually fired
  int64_t swordmaster_organic_games = 0;  // searched seat owns SM at terminal
                                          // WITHOUT a grant — the curriculum
                                          // ignition metric
  int64_t first_episode_id = -1;
  int64_t next_episode_id = -1;         // to persist for resume

  double collection_wall_time_s = 0.0;
  double mean_simulations_completed = 0.0;
  double mean_covered_prior_mass = 0.0;
  int64_t inference_calls = 0;
  int timeouts = 0;
  int fallback_raw_policy = 0;          // rejected/incomplete -> raw policy executed
};

// Collects one update's worth of auxiliary search examples from the frozen
// inference snapshot. Determinism: seat rotation is a pure function of
// episode_id; the per-root search/skip decision is a domain-separated
// deterministic Bernoulli keyed by (auxiliary_search_seed_domain, episode_id,
// decision_id); rollout/chance/opponent streams are separated from PPO's.
class OnlineSearchCollector {
 public:
  OnlineSearchCollector(const OnlineSearchConfig& config,
                        std::string checkpoint_hash);

  // Collects `config.auxiliary_games` games starting at
  // `config.next_auxiliary_episode_id`, appends accepted examples to `out`,
  // fills `stats`, and advances the episode cursor (both `stats->next_episode_id`
  // and the collector's own `config_.next_auxiliary_episode_id`). `update_id`
  // stamps the emitted examples.
  //
  // `game` is the environment (e.g. dune_imperium); `evaluator` is the FROZEN
  // pre-update inference snapshot, shared by all seats (self-play) and handed to
  // the search session. Both are injected so unit tests can drive the collector
  // with a mock evaluator and never touch the GPU.
  //
  // Semantics (see docs/PHASE_18B_ONLINE_COLLECTOR_DESIGN.md "CollectUpdate
  // wiring"): the searched seat runs a fresh frozen-config search session only at
  // its strategic ROOTS (ClassifyDuneDecisionRole == kAgentPrimary) selected by
  // the 0.25 Bernoulli; accepted searches emit one example (normalized visits as
  // CE target, terminal value attached at game end) and execute a visit-sampled
  // action. Non-searched seats, non-strategic decisions, unselected roots, and
  // rejected searches all execute the raw stochastic policy.
  void CollectUpdate(int update_id,
                     const std::shared_ptr<const Game>& game,
                     const std::shared_ptr<algorithms::Evaluator>& evaluator,
                     std::vector<SearchTrainingExample>* out,
                     OnlineSearchCollectionStats* stats);

  // Seat searched at a given auxiliary episode: rotates one seat by episode id.
  static Player SearchedSeatForEpisode(int64_t episode_id, int num_players);

  // Domain-separated deterministic search/skip decision at a strategic root.
  bool ShouldSearchAtRoot(int64_t episode_id, int decision_id) const;

  // Acceptance test on a completed search: >= min(min_coverage, legal_count)
  // actions with >= min_visits_per_action visits AND covered prior mass >=
  // min_prior_mass. Returns covered-action count and covered prior mass via out.
  bool AcceptSearch(const std::vector<int>& visit_counts,
                    const std::vector<double>& root_priors, int legal_count,
                    int* num_covered_out, double* covered_prior_mass_out) const;

 private:
  OnlineSearchConfig config_;
  std::string checkpoint_hash_;
};

// The linear search_loss_coef warmup: 0.0 on update 1, linearly to
// `target` on `warmup_update`, held at `target` thereafter. Free function so
// the trainer and its tests share one definition.
double SearchLossCoefForUpdate(int update_id, double target, int warmup_update);

// KataGo policy-target pruning (§3.2 of 1902.10565), exposed for unit testing.
// Given FINAL root stats, returns pruned visit counts to renormalize into the CE
// target: for every child except the most-visited c*, remove up to
// n_forced(c) = floor(sqrt(forced_k * priors[c] * total_root_visits)) visits, but
// never so many that the child's PUCT value would reach c*'s at the final stats;
// then drop any non-c* child left with a single visit. `priors` are the tree
// (post-noise) priors, aligned with `visits` and `q_values`.
std::vector<int> PruneForcedPlayouts(const std::vector<int>& visits,
                                     const std::vector<double>& priors,
                                     const std::vector<double>& q_values,
                                     int total_root_visits, double puct_c,
                                     double forced_k);

}  // namespace open_spiel

#endif  // OPEN_SPIEL_EXAMPLES_DUNE_ONLINE_SEARCH_COLLECTOR_H_
