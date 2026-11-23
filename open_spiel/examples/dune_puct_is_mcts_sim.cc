#include "dune_puct_is_mcts_internal.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <memory>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "open_spiel/abseil-cpp/absl/random/distributions.h"
#include "open_spiel/abseil-cpp/absl/strings/str_format.h"
#include "open_spiel/games/dune_imperium/dune_imperium.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"
#include "dune_seed_utils.h"

namespace open_spiel {

std::vector<double> DunePUCTISMCTSBot::RunSimulation(
    State* state, int depth, int decision_depth, int sim_index) {
  max_depth_this_search_ = std::max(max_depth_this_search_, depth);
  max_decision_depth_this_search_ =
      std::max(max_decision_depth_this_search_, decision_depth);

  if (state->IsTerminal()) {
    std::vector<double> returns = state->Returns();
    for (double& r : returns) {
      r /= config_.utility_divisor;
    }
    sum_depth_this_search_ += depth;
    sum_decision_depth_this_search_ += decision_depth;
    num_sims_this_search_++;
    simulation_depths_this_search_.push_back(depth);
    terminal_leaf_simulations_count_++;
    const auto* dune_state = dynamic_cast<const dune_imperium::DuneImperiumState*>(state);
    if (dune_state) {
      max_round_this_search_ = std::max(max_round_this_search_, dune_state->GetCurrentRound());
    }
    return returns;
  } else if (state->IsChanceNode()) {
    // Chance outcome RNG derived deterministically from stream seed
    uint64_t chance_seed = dune_seed::DeriveSeed(config_.seed, dune_seed::kStreamChance, sim_index, depth);
    std::mt19937 chance_rng(chance_seed);
    double r_num = absl::Uniform(chance_rng, 0.0, 1.0);
    Action chance_action = SampleAction(state->ChanceOutcomes(), r_num).first;
    state->ApplyAction(chance_action);
    return RunSimulation(state, depth + 1, decision_depth, sim_index);
  }

  std::vector<Action> legal_actions = state->LegalActions();
  if (legal_actions.size() == 1) {
    state->ApplyAction(legal_actions[0]);
    return RunSimulation(state, depth + 1, decision_depth, sim_index);
  }
  Player cur_player = state->CurrentPlayer();

  if (config_.max_search_decision_depth >= 0 &&
      decision_depth >= config_.max_search_decision_depth) {
    return EvaluateCappedLeaf(*state, depth, decision_depth);
  }

  if (config_.opponent_mode == SearchOpponentMode::kPolicy && cur_player != searching_player_) {
    auto key = GetStateKey(*state);
    auto iter = opponent_prior_cache_.find(key);
    ActionsAndProbs prior;
    if (iter != opponent_prior_cache_.end()) {
      prior = iter->second;
    } else {
      auto t_inf_start = std::chrono::steady_clock::now();
      prior = evaluators_[cur_player]->Prior(*state);
      auto t_inf_end = std::chrono::steady_clock::now();
      accumulated_inference_time_ms_ += std::chrono::duration<double, std::milli>(t_inf_end - t_inf_start).count();
      inference_count_this_search_++;
      opponent_prior_cache_[key] = prior;
    }

    // Filter cached/evaluated prior by current concrete legal actions
    ActionsAndProbs filtered_prior;
    double sum_prob = 0.0;
    for (const auto& ap : prior) {
      if (std::find(legal_actions.begin(), legal_actions.end(), ap.first) != legal_actions.end()) {
        filtered_prior.push_back(ap);
        sum_prob += ap.second;
      }
    }
    if (filtered_prior.empty()) {
      for (Action a : legal_actions) {
        filtered_prior.push_back({a, 1.0 / legal_actions.size()});
      }
    } else {
      for (auto& ap : filtered_prior) {
        ap.second = (sum_prob > 0.0) ? (ap.second / sum_prob) : (1.0 / filtered_prior.size());
      }
    }
    prior = filtered_prior;

    Action chosen_action = kInvalidAction;
    if (prior.empty()) {
      if (!legal_actions.empty()) {
        uint64_t tie_seed = dune_seed::DeriveSeed(config_.seed, dune_seed::kStreamMCTS, sim_index, depth);
        std::mt19937 tie_rng(tie_seed);
        chosen_action = legal_actions[absl::Uniform(tie_rng, 0u, legal_actions.size())];
      }
    } else {
      double temp = config_.opponent_temperature;
      if (temp <= 0.0) {
        chosen_action = prior[0].first;
        double best_prob = prior[0].second;
        for (const auto& ap : prior) {
          if (ap.second > best_prob) {
            best_prob = ap.second;
            chosen_action = ap.first;
          }
        }
      } else {
        uint64_t sample_seed = dune_seed::DeriveSeed(config_.seed, dune_seed::kStreamMCTS, sim_index, depth);
        std::mt19937 sample_rng(sample_seed);
        ActionsAndProbs scaled;
        scaled.reserve(prior.size());
        double max_log_p = -std::numeric_limits<double>::infinity();
        for (const auto& ap : prior) {
          double lp = std::log(std::max(ap.second, 1e-12));
          if (lp > max_log_p) max_log_p = lp;
        }
        double sum = 0.0;
        double inv_temp = 1.0 / temp;
        for (const auto& ap : prior) {
          double lp = std::log(std::max(ap.second, 1e-12));
          double p = std::exp((lp - max_log_p) * inv_temp);
          scaled.push_back({ap.first, p});
          sum += p;
        }
        if (sum > 0.0) {
          for (auto& ap : scaled) {
            ap.second /= sum;
          }
          chosen_action = SampleAction(scaled, absl::Uniform(sample_rng, 0.0, 1.0)).first;
        } else {
          chosen_action = prior[0].first;
        }
      }
    }

    if (chosen_action != kInvalidAction) {
      state->ApplyAction(chosen_action);
      std::vector<double> returns =
          RunSimulation(state, depth + 1, decision_depth + 1, sim_index);
      if (returns.empty()) {
        return {};
      }
      return returns;
    } else {
      std::vector<double> returns = state->Returns();
      for (double& r : returns) {
        r /= config_.utility_divisor;
      }
      sum_depth_this_search_ += depth;
      sum_decision_depth_this_search_ += decision_depth;
      num_sims_this_search_++;
      simulation_depths_this_search_.push_back(depth);
      const auto* dune_state = dynamic_cast<const dune_imperium::DuneImperiumState*>(state);
      if (dune_state) {
        max_round_this_search_ = std::max(max_round_this_search_, dune_state->GetCurrentRound());
      }
      return returns;
    }
  }

  if (has_deadline_ && std::chrono::steady_clock::now() >= deadline_) {
    return {};
  }

  DuneISMCTSNode* node = LookupOrCreateNode(*state);
  SPIEL_CHECK_TRUE(node != nullptr);

  if (has_deadline_ && std::chrono::steady_clock::now() >= deadline_) {
    return {};
  }

  InitializePriorsAndValue(node, *state); // Combined evaluation expansion!

  if (node->total_visits == -1) {
    node->total_visits = 0;
    sum_depth_this_search_ += depth;
    sum_decision_depth_this_search_ += decision_depth;
    num_sims_this_search_++;
    simulation_depths_this_search_.push_back(depth);
    const auto* dune_state = dynamic_cast<const dune_imperium::DuneImperiumState*>(state);
    if (dune_state) {
      max_round_this_search_ = std::max(max_round_this_search_, dune_state->GetCurrentRound());
    }
    return node->cached_values;
  }

  Action chosen_action = SelectActionTreePolicy(node, legal_actions);
  SPIEL_CHECK_NE(chosen_action, kInvalidAction);

  node->total_visits++;
  node->child_info[chosen_action].visits++;

  state->ApplyAction(chosen_action);
  std::vector<double> returns =
      RunSimulation(state, depth + 1, decision_depth + 1, sim_index);
  if (returns.empty()) {
    node->total_visits--;
    node->child_info[chosen_action].visits--;
    return {};
  }

// Multi-player Max-N backup
  node->child_info[chosen_action].return_sum += returns[cur_player];
  return returns;
}

DuneSearchResult DunePUCTISMCTSBot::RunSearch(const State& state, int max_sims, double max_time_ms, int start_sim_index) {
  root_samples_.clear();
  opponent_prior_cache_.clear();
  inference_count_this_search_ = 0;
  root_node_ = nullptr;
  if (start_sim_index == 0) {
    current_search_leaf_histories_.clear();
    current_search_sampled_leaf_states_.clear();
    diagnostic_leaf_states_seen_ = 0;
  }

  visited_nodes_this_search_.clear();
  simulation_depths_this_search_.clear();
  terminal_leaf_simulations_count_ = 0;
  max_round_this_search_ = 0;
  max_depth_this_search_ = 0;
  sum_depth_this_search_ = 0.0;
  num_sims_this_search_ = 0;
  max_decision_depth_this_search_ = 0;
  sum_decision_depth_this_search_ = 0.0;
  is_continuation_ = (start_sim_index > 0);
  // PWO-4 Amendment 2 section 5.1: was an existing tree left in place for this
  // decision's root? Captured exactly where the amendment defines it -- after
  // the reset-or-retain block resolves, before LookupOrCreateNode. Emission
  // only: nothing below reads this flag, and the two assignments added here sit
  // in `else` arms that previously did nothing, so no branch is taken
  // differently than before.
  bool root_reused_pre_search = false;
  if (!in_session_) {
    if (evaluators_ != last_evaluators_ ||
        state.CurrentPlayer() != last_searching_player_ ||
        last_searching_player_ == kInvalidPlayer) {
      Reset();
      last_evaluators_ = evaluators_;
      last_searching_player_ = state.CurrentPlayer();
    } else {
      auto root_key = GetStateKey(state);
      if (nodes_.find(root_key) == nodes_.end()) {
        Reset();
      } else {
        root_reused_pre_search = true;
      }
    }
  } else {
    // The session path does not run the block above; the session owns the tree.
    // Computed explicitly so the field is never a silent default there -- the
    // exact defect (a plausible zero reading as data) that made
    // inherited_root_visits useless on the fresh path.
    root_reused_pre_search = (nodes_.find(GetStateKey(state)) != nodes_.end());
  }

  int actual_max_sims = (max_sims >= 0) ? max_sims : config_.max_simulations;
  double actual_max_time_ms = (max_time_ms >= 0.0) ? max_time_ms : config_.relative_time_budget_ms;

  accumulated_inference_time_ms_ = 0.0;
  accumulated_sample_root_ms_ = 0.0;
  accumulated_tree_engine_ms_ = 0.0;

  auto start_time = std::chrono::steady_clock::now();
  has_deadline_ = (actual_max_time_ms != std::numeric_limits<double>::infinity());
  if (has_deadline_) {
    deadline_ = start_time + std::chrono::milliseconds(static_cast<int64_t>(actual_max_time_ms));
  }

  DuneSearchResult result;
  result.timeout_status = false;

  std::vector<Action> legal_actions = state.LegalActions();
  if (legal_actions.size() == 1) {
    result.policy = {{legal_actions[0], 1.0}};
    result.raw_policy = result.policy;
    result.simulations_completed = 0;
    result.elapsed_time_ms = 0.0;
    result.inference_count = inference_count_this_search_;
    result.diagnostics.actions = legal_actions;
    result.diagnostics.priors = {1.0};
    result.diagnostics.raw_priors = {1.0};
    result.diagnostics.incumbent_action = legal_actions[0];
    result.diagnostics.raw_prior_max_prob = 1.0;
    result.diagnostics.raw_prior_max_prob_hex = absl::StrFormat("%a", 1.0);
    result.diagnostics.time_sample_root_ms = 0.0;
    result.diagnostics.time_nn_inference_ms = 0.0;
    result.diagnostics.time_tree_engine_ms = 0.0;
    result.diagnostics.time_other_overhead_ms = 0.0;
    last_search_result_ = result;
    return result;
  }

  if (config_.check_strategic_state && !IsStrategicState(state, state.CurrentPlayer())) {
    auto t_inf_start = std::chrono::steady_clock::now();
    result.policy = evaluators_[state.CurrentPlayer()]->Prior(state);
    result.raw_policy = result.policy;
    auto t_inf_end = std::chrono::steady_clock::now();
    accumulated_inference_time_ms_ += std::chrono::duration<double, std::milli>(t_inf_end - t_inf_start).count();
    inference_count_this_search_++;
    result.simulations_completed = 0;
    result.elapsed_time_ms = 0.0;
    result.fallback_reason = "non_strategic_state";
    result.inference_count = inference_count_this_search_;
    result.diagnostics.actions = legal_actions;
    result.diagnostics.priors.reserve(result.policy.size());
    result.diagnostics.raw_priors.reserve(result.policy.size());
    for (const auto& ap : result.policy) {
      result.diagnostics.priors.push_back(ap.second);
      result.diagnostics.raw_priors.push_back(ap.second);
    }
    double max_prob = 0.0;
    result.diagnostics.incumbent_action =
        ComputeIncumbentFromPriors(result.policy, legal_actions, &max_prob);
    result.diagnostics.raw_prior_max_prob = max_prob;
    result.diagnostics.raw_prior_max_prob_hex = absl::StrFormat("%a", max_prob);
    result.diagnostics.time_sample_root_ms = 0.0;
    result.diagnostics.time_nn_inference_ms = accumulated_inference_time_ms_;
    result.diagnostics.time_tree_engine_ms = 0.0;
    result.diagnostics.time_other_overhead_ms = 0.0;
    last_search_result_ = result;
    return result;
  }

  searching_player_ = state.CurrentPlayer();

  auto check_time = std::chrono::steady_clock::now();
  double check_elapsed_ms = std::chrono::duration<double, std::milli>(check_time - start_time).count();
  double safety_margin = std::min(50.0, std::max(5.0, actual_max_time_ms - 10.0));
  if (check_elapsed_ms + safety_margin >= actual_max_time_ms) {
    // Defense in depth: an already-expired deadline must NOT yield a uniform
    // policy. Uniform-over-legal-actions was played verbatim by callers and,
    // even under argmax selection, collapses to "always the lowest legal
    // action id". Degrade instead to the raw network prior on the true current
    // state (one inference, identical to the session-level policy-only
    // fallback) and set used_fallback so callers can see the search never ran.
    auto t_inf_start = std::chrono::steady_clock::now();
    result.policy = evaluators_[state.CurrentPlayer()]->Prior(state);
    result.raw_policy = result.policy;
    auto t_inf_end = std::chrono::steady_clock::now();
    accumulated_inference_time_ms_ += std::chrono::duration<double, std::milli>(t_inf_end - t_inf_start).count();
    inference_count_this_search_++;
    result.simulations_completed = 0;
    result.elapsed_time_ms = check_elapsed_ms;
    result.timeout_status = true;
    result.used_fallback = true;
    result.fallback_reason = "timeout";
    result.inference_count = inference_count_this_search_;
    result.diagnostics.actions = legal_actions;
    result.diagnostics.priors.reserve(result.policy.size());
    result.diagnostics.raw_priors.reserve(result.policy.size());
    for (const auto& ap : result.policy) {
      result.diagnostics.priors.push_back(ap.second);
      result.diagnostics.raw_priors.push_back(ap.second);
    }
    double max_prob = 0.0;
    result.diagnostics.incumbent_action =
        ComputeIncumbentFromPriors(result.policy, legal_actions, &max_prob);
    result.diagnostics.raw_prior_max_prob = max_prob;
    result.diagnostics.raw_prior_max_prob_hex = absl::StrFormat("%a", max_prob);
    result.diagnostics.time_sample_root_ms = 0.0;
    result.diagnostics.time_nn_inference_ms = accumulated_inference_time_ms_;
    result.diagnostics.time_tree_engine_ms = 0.0;
    result.diagnostics.time_other_overhead_ms = std::max(0.0, result.elapsed_time_ms - accumulated_inference_time_ms_);
    last_search_result_ = result;
    return result;
  }

  root_node_ = LookupOrCreateNode(state);
  InitializePriorsAndValue(root_node_, state);

  // WO-15 / search finding 7: mark the root EXPANDED here, where it is actually
  // expanded, instead of leaving total_visits at -1 for the sim loop to trip
  // over. It used to stay -1, so simulation 0 re-reached the root, flipped it to
  // 0 and returned the cached value without ever descending: a cold N-simulation
  // root did N-1 descents while reporting N against the budget. The under-funded
  // 16-sim purchase window lost 1/16 of its search to this. Continuations
  // re-enter an already-expanded root and are unaffected.
  if (root_node_->total_visits < 0) {
    root_node_->total_visits = 0;
  }

  // PWO-4 Amendment 2 section 5.1: the PRE-SEARCH snapshot, taken HERE -- after
  // LookupOrCreateNode, after InitializePriorsAndValue, and after the WO-15
  // cold-root expansion bookkeeping immediately above -- and before the
  // simulation loop below. Everything this decision's 200 simulations add is
  // measured against it.
  std::vector<int> visit_counts_pre_search;
  std::vector<int> visit_counts_checkpoint;
  std::vector<int> visit_counts_final;
  dune_puct_internal::SnapshotRootVisits(root_node_, legal_actions, &visit_counts_pre_search);

  // Apply Dirichlet noise exactly once at the root node's priors (if enabled).
  if (config_.dirichlet_epsilon > 0.0 && !is_continuation_ && !root_node_->dirichlet_noise_applied) {
    if (legal_actions.size() > 1) {
      // Per-root Dirichlet concentration (KataGo inverse-legal-count scaling):
      // alpha = dirichlet_alpha_total / N_legal when configured, else the fixed
      // legacy alpha. alpha <= 0 disables the draw (matches the old alpha>0 guard).
      double alpha = config_.dirichlet_alpha_total > 0.0
          ? config_.dirichlet_alpha_total / static_cast<double>(legal_actions.size())
          : config_.dirichlet_alpha;
      if (alpha > 0.0) {
        // Per-root noise stream (bug fix): key the stream on this root's identity
        // (player + state key) plus the search counter, so noise no longer repeats
        // across roots handled by one bot instance. The previous constant position
        // argument (0) seeded every root identically.
        std::pair<Player, std::string> root_key = GetStateKey(state);
        uint64_t noise_seed = DeriveRootNoiseSeed(
            config_.seed, static_cast<uint64_t>(search_count_), root_key.first,
            root_key.second);
        std::mt19937 noise_rng(noise_seed);
        std::gamma_distribution<double> gamma_dist(alpha, 1.0);
        std::vector<double> noise(legal_actions.size());
        double sum_noise = 0.0;
        for (size_t i = 0; i < legal_actions.size(); ++i) {
          noise[i] = gamma_dist(noise_rng);
          sum_noise += noise[i];
        }
        for (size_t i = 0; i < legal_actions.size(); ++i) {
          Action action = legal_actions[i];
          double d_noise = sum_noise > 0.0 ? (noise[i] / sum_noise) : (1.0 / legal_actions.size());
          auto& child = root_node_->child_info[action];
          // Only the TREE prior is noised. child.raw_prior already holds the
          // untransformed network prior from expansion, which is what item-4
          // telemetry and the starvation fallbacks read.
          child.prior = (1.0 - config_.dirichlet_epsilon) * child.prior + config_.dirichlet_epsilon * d_noise;
        }
        root_node_->dirichlet_noise_applied = true;
      }
    }
  }

  int stability_checkpoint = std::floor(actual_max_sims * config_.conservative_stability_checkpoint_fraction);
  Action stability_checkpoint_action = kInvalidAction;
  bool stability_checkpoint_reached = false;
  // PWO-3 telemetry (docs/PWO3_REGISTRATION.md sections 4.2, 14.3a). Read-only
  // snapshots at checkpoints the loop already visits: no RNG is drawn and no tree
  // state is mutated. On a FIXED tier that makes them behaviour-neutral outright
  // (the search stops on a COUNT). The half-TIME snapshot below sits inside the
  // TIMED loop, so on a live tier it is MEASUREMENT OVERHEAD -- bounded, fires at
  // most once, cannot change the tree, but can shift simulations_completed
  // slightly. Do not describe it as bitwise unchanged.
  const int pwo3_min_visits = EffectiveMinVisitThreshold(config_.min_visit_threshold);
  int stability_checkpoint_num_covered = 0;
  double stability_checkpoint_prior_mass = 0.0;
  Action half_time_checkpoint_action = kInvalidAction;
  bool half_time_checkpoint_reached = false;
  int half_time_checkpoint_sim = -1;
  int half_time_checkpoint_num_covered = 0;
  double half_time_checkpoint_prior_mass = 0.0;
  const double half_time_ms = 0.5 * actual_max_time_ms;
  const bool has_finite_time_budget = std::isfinite(actual_max_time_ms);
  // The end-of-search coverage computation (below, "Coverage Check & Fallback
  // Logic"), lifted verbatim so a checkpoint measures exactly what the gate does.
  auto pwo3_snapshot_coverage = [&](int* num_covered, double* prior_mass) {
    double mass = 0.0;
    int covered = 0;
    ActionsAndProbs np = FilterAndNormalizePriors(root_node_, legal_actions);
    for (size_t i = 0; i < legal_actions.size(); ++i) {
      auto it = root_node_->child_info.find(legal_actions[i]);
      int v = (it != root_node_->child_info.end()) ? it->second.visits : 0;
      if (v >= pwo3_min_visits) {
        mass += np[i].second;
        covered++;
      }
    }
    *num_covered = covered;
    *prior_mass = mass;
  };

  int sim = 0;
  double max_sim_duration_ms = 10.0;
  for (; sim < actual_max_sims; ++sim) {
    // Checkpoint stability snapshot
    if (sim == stability_checkpoint) {
      stability_checkpoint_action = GetRootArgmaxAction(root_node_, legal_actions);
      stability_checkpoint_reached = true;
      pwo3_snapshot_coverage(&stability_checkpoint_num_covered,
                             &stability_checkpoint_prior_mass);
      // PWO-4 Amendment 2 section 5.1: the CHECKPOINT snapshot, at the same
      // point the existing PWO-3 checkpoint telemetry above already reads.
      dune_puct_internal::SnapshotRootVisits(root_node_, legal_actions, &visit_counts_checkpoint);
    }

    // Check relative time budget limit with measured max simulation latency safety margin
    auto now = std::chrono::steady_clock::now();
    double elapsed_ms = std::chrono::duration<double, std::milli>(now - start_time).count();
    // PWO-3 half-TIME checkpoint: fires once, at the first simulation at or past
    // half the deadline. Only meaningful under a finite time budget (live tiers).
    if (has_finite_time_budget && !half_time_checkpoint_reached &&
        elapsed_ms >= half_time_ms) {
      half_time_checkpoint_action = GetRootArgmaxAction(root_node_, legal_actions);
      half_time_checkpoint_reached = true;
      half_time_checkpoint_sim = sim;
      pwo3_snapshot_coverage(&half_time_checkpoint_num_covered,
                             &half_time_checkpoint_prior_mass);
    }
    if (elapsed_ms + max_sim_duration_ms >= actual_max_time_ms) {
      result.timeout_status = true;
      result.fallback_reason = "timeout";
      break;
    }

    // Check Node/Memory limit
    if (static_cast<int>(node_pool_.size()) >= config_.max_nodes) {
      result.fallback_reason = "max_nodes";
      break;
    }

    auto t_sample_start = std::chrono::steady_clock::now();
    std::unique_ptr<State> sampled_root_state = SampleRootState(state, start_sim_index + sim);
    auto t_sample_end = std::chrono::steady_clock::now();
    accumulated_sample_root_ms_ += std::chrono::duration<double, std::milli>(t_sample_end - t_sample_start).count();

    double inf_before_sim = accumulated_inference_time_ms_;
    auto t_sim_start = std::chrono::steady_clock::now();
    std::vector<double> returns =
        RunSimulation(sampled_root_state.get(), 0, 0,
                      start_sim_index + sim);
    auto t_sim_end = std::chrono::steady_clock::now();
    double sim_wall_ms = std::chrono::duration<double, std::milli>(t_sim_end - t_sim_start).count();
    double inf_in_sim_ms = accumulated_inference_time_ms_ - inf_before_sim;
    accumulated_tree_engine_ms_ += std::max(0.0, sim_wall_ms - inf_in_sim_ms);

    if (returns.empty()) {
      result.timeout_status = true;
      result.fallback_reason = "timeout";
      break;
    }

    double sim_duration = std::chrono::duration<double, std::milli>(t_sim_end - t_sample_start).count();
    max_sim_duration_ms = std::max(max_sim_duration_ms, sim_duration);
  }

  auto end_time = std::chrono::steady_clock::now();
  result.elapsed_time_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();
  result.simulations_completed = sim;

  // Coverage Check & Fallback Logic
  const int min_visits = EffectiveMinVisitThreshold(config_.min_visit_threshold);
  double covered_prior_mass = 0.0;
  int num_covered_actions = 0;
  ActionsAndProbs normalized_priors = FilterAndNormalizePriors(root_node_, legal_actions);
  for (size_t i = 0; i < legal_actions.size(); ++i) {
    Action action = legal_actions[i];
    double prior = normalized_priors[i].second;
    auto iter = root_node_->child_info.find(action);
    int visits = (iter != root_node_->child_info.end()) ? iter->second.visits : 0;
    if (visits >= min_visits) {
      covered_prior_mass += prior;
      num_covered_actions++;
    }
  }

  // Leader-specific mass-only coverage. Scoped by BOTH the opt-in flag and the
  // root's actual phase, so an identical coverage pattern at any non-Leader
  // root keeps the generic three-action rule and is still rejected.
  bool leader_root = false;
  if (config_.leader_mass_only_coverage) {
    const auto& dune_state =
        static_cast<const dune_imperium::DuneImperiumState&>(state);
    const dune_imperium::GamePhase phase = dune_state.phase();
    leader_root = (phase == dune_imperium::GamePhase::kLeaderDraft ||
                   phase == dune_imperium::GamePhase::kLeaderOfferChance);
  }
  const int req_covered_actions =
      leader_root ? 0 : std::min<int>(3, legal_actions.size());
  if (num_covered_actions < req_covered_actions || covered_prior_mass < config_.covered_prior_threshold) {
    // WO-15 / search finding 4: degrade to the RAW network prior, not to the
    // tree prior. At a noised root the tree prior is a 25% Dirichlet mixture
    // (and may carry root_prior_temperature flattening), so a consumer that
    // plays this policy — e.g. the legacy kTrainingFullFast path in
    // dune_search_session.cc — was executing exploration noise a quarter of the
    // time on exactly the roots the search had failed to cover. Matches what
    // the expired-deadline fallback above and the session policy-only fallback
    // already play.
    result.policy = FilterAndNormalizeRawPriors(root_node_, legal_actions);
    result.used_fallback = true;
    if (result.fallback_reason == "none") {
      result.fallback_reason = "low_coverage";
    }
  } else {
    result.policy = GetFinalPolicy(state, root_node_);
    // Check if fallback to uniform happened due to zero root simulation visits
    double sum_visits = 0.0;
    for (Action action : legal_actions) {
      auto iter = root_node_->child_info.find(action);
      if (iter != root_node_->child_info.end()) {
        sum_visits += iter->second.visits;
      }
    }
    if (sum_visits == 0.0) {
      result.used_fallback = true;
      if (result.fallback_reason == "none") {
        result.fallback_reason = "zero_visits";
      }
    }
  }

  if (root_node_ != nullptr) {
    result.raw_policy = root_node_->raw_policy;
  }

  Action final_proposed_action = GetRootArgmaxAction(root_node_, legal_actions);

  result.diagnostics = GetRootDiagnostics(state, min_visits);
  double max_raw_prob = 0.0;
  result.diagnostics.incumbent_action =
      ComputeIncumbentFromNode(root_node_, legal_actions, &max_raw_prob);
  result.diagnostics.raw_prior_max_prob = max_raw_prob;
  result.diagnostics.raw_prior_max_prob_hex = absl::StrFormat("%a", max_raw_prob);
  result.diagnostics.max_sim_duration_ms = max_sim_duration_ms;
  result.diagnostics.leaf_histories = current_search_leaf_histories_;
  result.diagnostics.sampled_leaf_states = current_search_sampled_leaf_states_;
  result.diagnostics.stability_checkpoint_action = stability_checkpoint_action;
  result.diagnostics.stability_checkpoint_reached = stability_checkpoint_reached;
  result.diagnostics.stability_agreement = stability_checkpoint_reached && (stability_checkpoint_action == final_proposed_action);
  // PWO-3 telemetry (docs/PWO3_REGISTRATION.md section 4.2).
  result.diagnostics.stability_checkpoint_num_covered_actions = stability_checkpoint_num_covered;
  result.diagnostics.stability_checkpoint_covered_prior_mass = stability_checkpoint_prior_mass;
  result.diagnostics.half_time_checkpoint_action = half_time_checkpoint_action;
  result.diagnostics.half_time_checkpoint_reached = half_time_checkpoint_reached;
  result.diagnostics.half_time_checkpoint_sim = half_time_checkpoint_sim;
  result.diagnostics.half_time_checkpoint_num_covered_actions = half_time_checkpoint_num_covered;
  result.diagnostics.half_time_checkpoint_covered_prior_mass = half_time_checkpoint_prior_mass;
  // PWO-4 Amendment 2 section 5.1: the FINAL snapshot and the retention block.
  // Taken here, after GetRootDiagnostics has read the same node, so that
  // requiring visit_counts_final == visit_counts is a genuine cross-check of two
  // independent reads rather than a copy compared with itself.
  dune_puct_internal::SnapshotRootVisits(root_node_, legal_actions, &visit_counts_final);
  result.diagnostics.retention_snapshots_valid = true;
  result.diagnostics.root_reused_pre_search = root_reused_pre_search;
  result.diagnostics.visit_counts_pre_search = visit_counts_pre_search;
  result.diagnostics.visit_counts_checkpoint = visit_counts_checkpoint;
  result.diagnostics.visit_counts_final = visit_counts_final;
  result.inference_count = inference_count_this_search_;

  result.diagnostics.time_sample_root_ms = accumulated_sample_root_ms_;
  result.diagnostics.time_nn_inference_ms = accumulated_inference_time_ms_;
  result.diagnostics.time_tree_engine_ms = accumulated_tree_engine_ms_;
  double explained_time = accumulated_sample_root_ms_ + accumulated_inference_time_ms_ + accumulated_tree_engine_ms_;
  result.diagnostics.time_other_overhead_ms = std::max(0.0, result.elapsed_time_ms - explained_time);

  last_search_result_ = result;
  return result;
}

}  // namespace open_spiel
