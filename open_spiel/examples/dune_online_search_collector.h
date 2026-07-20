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
#include <string>
#include <vector>

#include "open_spiel/spiel.h"

namespace open_spiel {

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

  // --- Frozen teacher search budget (64 primary / 8 continuation) ---
  int max_simulations = 72;             // fixed_session_limit; primary = limit - reserve
  int fixed_continuation_reserve = 8;   // -> 64 primary simulations
  int max_search_decision_depth = -1;   // UNCAPPED (Step 1 winner). Do not cap.
  double puct_c = 0.30;                 // calibrated (Stage 3 / Step 1 winner)
  bool use_opponent_model = true;       // policy opponent model
  double simulated_opponent_temperature = 1.0;
  double root_prior_temperature = 1.0;
  double utility_divisor = 4.0;
  bool nonlinear_value_head = false;    // baseline critic. Do not swap.

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
struct OnlineSearchCollectionStats {
  int auxiliary_games = 0;
  int strategic_roots_seen = 0;
  int searches_selected = 0;            // Bernoulli(0.25) selected
  int accepted_targets = 0;
  int rejected_incomplete = 0;          // searched but failed acceptance
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
  // fills `stats`, and advances the episode cursor. `update_id` stamps the
  // emitted examples. Requires the frozen inference model/evaluator to be wired
  // (see the .cc); non-search seats and rejected roots execute the raw policy.
  void CollectUpdate(int update_id,
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

}  // namespace open_spiel

#endif  // OPEN_SPIEL_EXAMPLES_DUNE_ONLINE_SEARCH_COLLECTOR_H_
