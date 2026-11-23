#include "dune_puct_is_mcts.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <numeric>
#include <random>
#include <utility>
#include <vector>

#include "open_spiel/abseil-cpp/absl/container/flat_hash_map.h"
#include "open_spiel/abseil-cpp/absl/random/distributions.h"
#include "open_spiel/abseil-cpp/absl/strings/str_cat.h"
#include "open_spiel/abseil-cpp/absl/strings/str_format.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"
#include "dune_seed_utils.h"

namespace open_spiel {

// ---------------------------------------------------------------------------
// DunePUCTISMCTSBot Implementation
// ---------------------------------------------------------------------------

DunePUCTISMCTSBot::DunePUCTISMCTSBot(
    const DuneSearchConfig& config,
    const std::vector<std::shared_ptr<algorithms::Evaluator>>& evaluators)
    : rng_(config.seed),
      config_(config),
      evaluators_(evaluators),
      root_node_(nullptr) {
  SPIEL_CHECK_EQ(evaluators_.size(), 4);
  SPIEL_CHECK_TRUE(std::isfinite(config_.temperature));
  SPIEL_CHECK_GE(config_.temperature, 0.0);
  SPIEL_CHECK_TRUE(std::isfinite(config_.root_prior_temperature));
  SPIEL_CHECK_GE(config_.root_prior_temperature, 0.0);
  SPIEL_CHECK_TRUE(std::isfinite(config_.opponent_temperature));
  SPIEL_CHECK_GE(config_.opponent_temperature, 0.0);
  SPIEL_CHECK_TRUE(config_.max_search_decision_depth == -1 || config_.max_search_decision_depth > 0);
}

DunePUCTISMCTSBot::DunePUCTISMCTSBot(
    uint64_t seed, std::shared_ptr<algorithms::Evaluator> evaluator,
    double puct_c, int max_simulations,
    int max_world_samples,
    double temperature,
    double dirichlet_epsilon,
    double dirichlet_alpha,
    double utility_divisor,
    bool use_observation_string,
    DuneISMCTSFinalPolicyType final_policy_type,
    bool use_opponent_model,
    double opponent_temperature,
    bool verbose_diagnostics)
    // WO-15 / search finding 2: this used to be a POSITIONAL braced-init of
    // DuneSearchConfig. Phase 18B inserted dirichlet_alpha_total,
    // forced_playouts_k and root_noise_fpu_zero ahead of use_observation_string
    // in the struct, so the trailing bools silently slid into the KataGo
    // doubles (bool->double is only a -Wnarrowing warning, not an error):
    // use_observation_string=true became dirichlet_alpha_total=1.0, which
    // switches on KataGo inverse-legal-count alpha scaling the moment anyone
    // enables noise on this path. Designated initializers bind by NAME, so
    // inserting a field can no longer shift anyone's value; every field the
    // legacy signature does not carry keeps its declared default.
    : DunePUCTISMCTSBot(
          DuneSearchConfig{
              .max_simulations = max_simulations,
              .relative_time_budget_ms = 10000.0,
              .max_nodes = 50000,
              .puct_c = puct_c,
              .opponent_mode = use_opponent_model ? SearchOpponentMode::kPolicy
                                                  : SearchOpponentMode::kMaxN,
              .temperature = temperature,
              .opponent_temperature = opponent_temperature,
              .max_world_samples = max_world_samples,
              .utility_divisor = utility_divisor,
              .min_visit_threshold = 2,
              .covered_prior_threshold = 0.50,
              .seed = seed,
              .final_policy_type = final_policy_type,
              .dirichlet_epsilon = dirichlet_epsilon,
              .dirichlet_alpha = dirichlet_alpha,
              .use_observation_string = use_observation_string,
              .verbose_diagnostics = verbose_diagnostics,
          },
          evaluator) {}

double DunePUCTISMCTSBot::RandomNumber() {
  return absl::Uniform(rng_, 0.0, 1.0);
}

void DunePUCTISMCTSBot::Reset() {
  nodes_.clear();
  node_pool_.clear();
  root_samples_.clear();
  root_node_ = nullptr;
  total_lookups_ = 0;
  reused_lookups_ = 0;
  opponent_prior_cache_.clear();
  current_search_leaf_histories_.clear();
  current_search_sampled_leaf_states_.clear();
  diagnostic_leaf_states_seen_ = 0;
}

const DuneSearchResult& DunePUCTISMCTSBot::GetLastSearchResult() const {
  return last_search_result_;
}

Action DunePUCTISMCTSBot::ComputeIncumbentFromNode(
    DuneISMCTSNode* node, const std::vector<Action>& legal_actions,
    double* max_prob_out) const {
  if (legal_actions.empty()) {
    if (max_prob_out) *max_prob_out = 0.0;
    return kInvalidAction;
  }
  Action best_action = legal_actions.front();
  double max_prob = -1.0;
  if (node != nullptr && node->child_info.count(best_action)) {
    max_prob = node->child_info.at(best_action).raw_prior;
  }
  SPIEL_CHECK_TRUE(std::isfinite(max_prob));
  for (size_t i = 1; i < legal_actions.size(); ++i) {
    Action a = legal_actions[i];
    double p = 0.0;
    if (node != nullptr && node->child_info.count(a)) {
      p = node->child_info.at(a).raw_prior;
    }
    SPIEL_CHECK_TRUE(std::isfinite(p));
    if (p > max_prob) {
      max_prob = p;
      best_action = a;
    }
  }
  if (max_prob_out) *max_prob_out = max_prob;
  return best_action;
}

Action DunePUCTISMCTSBot::ComputeIncumbentFromPriors(
    const ActionsAndProbs& priors, const std::vector<Action>& legal_actions,
    double* max_prob_out) const {
  if (legal_actions.empty()) {
    if (max_prob_out) *max_prob_out = 0.0;
    return kInvalidAction;
  }
  absl::flat_hash_map<Action, double> prob_map;
  for (const auto& ap : priors) prob_map[ap.first] = ap.second;
  Action best_action = legal_actions.front();
  double max_prob = prob_map.count(best_action) ? prob_map[best_action] : 0.0;
  SPIEL_CHECK_TRUE(std::isfinite(max_prob));
  for (size_t i = 1; i < legal_actions.size(); ++i) {
    Action a = legal_actions[i];
    double p = prob_map.count(a) ? prob_map[a] : 0.0;
    SPIEL_CHECK_TRUE(std::isfinite(p));
    if (p > max_prob) {
      max_prob = p;
      best_action = a;
    }
  }
  if (max_prob_out) *max_prob_out = max_prob;
  return best_action;
}

Action DunePUCTISMCTSBot::GetRootRawPriorArgmax() const {
  if (last_search_result_.diagnostics.incumbent_action != kInvalidAction) {
    return last_search_result_.diagnostics.incumbent_action;
  }
  if (root_node_ != nullptr && !root_node_->child_info.empty()) {
    return ComputeIncumbentFromNode(
        root_node_, last_search_result_.diagnostics.actions, nullptr);
  }
  if (!last_search_result_.policy.empty()) {
    return ComputeIncumbentFromPriors(
        last_search_result_.policy, last_search_result_.diagnostics.actions, nullptr);
  }
  return kInvalidAction;
}

ActionsAndProbs DunePUCTISMCTSBot::GetFinalPolicy(const State& state, DuneISMCTSNode* node) const {
  ActionsAndProbs policy;
  SPIEL_CHECK_TRUE(node != nullptr);
  std::vector<Action> legal_actions = state.LegalActions();

  std::vector<std::pair<Action, double>> action_visits;
  action_visits.reserve(legal_actions.size());
  double total_legal_visits = 0.0;

  for (Action action : legal_actions) {
    double visits = 0.0;
    auto iter = node->child_info.find(action);
    if (iter != node->child_info.end()) {
      visits = static_cast<double>(iter->second.visits);
    }
    action_visits.push_back({action, visits});
    total_legal_visits += visits;
  }

  if (total_legal_visits == 0.0) {
    // WO-15 / search finding 3: the historical starvation bug lived here. A
    // root with no visits used to yield the UNIFORM policy, which callers
    // played verbatim and which, under argmax selection, collapses to "always
    // the lowest legal action id". There is nothing to read from the visit
    // counts, so degrade to the raw network prior — the same policy every other
    // starved path in this file plays.
    return FilterAndNormalizeRawPriors(node, legal_actions);
  }

  if (config_.final_policy_type == DuneISMCTSFinalPolicyType::kMaxValue) {
    policy.reserve(legal_actions.size());
    Action max_action = kInvalidAction;
    double max_val = -std::numeric_limits<double>::infinity();
    for (Action action : legal_actions) {
      double val = -std::numeric_limits<double>::infinity();
      auto iter = node->child_info.find(action);
      if (iter != node->child_info.end() && iter->second.visits > 0) {
        val = iter->second.value();
      }
      if (val > max_val) {
        max_val = val;
        max_action = action;
      }
    }
    for (Action action : legal_actions) {
      policy.push_back({action, (action == max_action) ? 1.0 : 0.0});
    }
    return policy;
  }

  if (config_.final_policy_type == DuneISMCTSFinalPolicyType::kMaxVisitCount || config_.temperature == 0.0) {
    Action max_action = kInvalidAction;
    double max_visits = -1.0;
    for (const auto& av : action_visits) {
      if (av.second > max_visits) {
        max_visits = av.second;
        max_action = av.first;
      }
    }
    for (Action action : legal_actions) {
      policy.push_back({action, (action == max_action) ? 1.0 : 0.0});
    }
  } else if (config_.temperature == 1.0) {
    for (const auto& av : action_visits) {
      policy.push_back({av.first, av.second / total_legal_visits});
    }
  } else {
    double max_log_visits = -std::numeric_limits<double>::infinity();
    for (const auto& av : action_visits) {
      if (av.second > 0.0) {
        double lv = std::log(av.second);
        if (lv > max_log_visits) max_log_visits = lv;
      }
    }
    if (!std::isfinite(max_log_visits)) {
      return FilterAndNormalizeRawPriors(node, legal_actions);
    }
    double inv_temp = 1.0 / config_.temperature;
    double sum_w = 0.0;
    std::vector<double> pow_visits(action_visits.size(), 0.0);
    for (size_t i = 0; i < action_visits.size(); ++i) {
      if (action_visits[i].second > 0.0) {
        pow_visits[i] = std::exp((std::log(action_visits[i].second) - max_log_visits) * inv_temp);
        sum_w += pow_visits[i];
      }
    }
    if (sum_w <= 0.0) {
      return FilterAndNormalizeRawPriors(node, legal_actions);
    }
    for (size_t i = 0; i < action_visits.size(); ++i) {
      policy.push_back({action_visits[i].first, pow_visits[i] / sum_w});
    }
  }

  return policy;
}

DunePUCTISMCTSBot::ControllerSelection DunePUCTISMCTSBot::SelectAction(
    const State& state, const DuneSearchResult& res) {
  ControllerSelection sel;
  sel.selection_mode = DuneActionSelectionModeToString(config_.action_selection_mode);

  std::vector<Action> legal_actions = state.LegalActions();
  if (legal_actions.empty()) {
    sel.selected_action = kInvalidAction;
    sel.incumbent_action = kInvalidAction;
    sel.selection_reason = "no_legal_actions";
    return sel;
  }

  // 1. Advance search counter and draw step RNG identically to historical progression
  search_count_++;
  uint64_t step_seed = dune_seed::Combine(config_.seed, dune_seed::kStreamBlueprint, search_count_);
  std::mt19937 step_rng(step_seed);
  double r_val = absl::Uniform(step_rng, 0.0, 1.0);

  // 2. Identify incumbent action and maximum raw prior probability from the current decision
  Action incumbent = res.diagnostics.incumbent_action;
  double max_raw_prob = res.diagnostics.raw_prior_max_prob;
  if (incumbent == kInvalidAction) {
    if (legal_actions.size() == 1) {
      incumbent = legal_actions[0];
      max_raw_prob = 1.0;
    } else if (root_node_ != nullptr && !root_node_->child_info.empty()) {
      incumbent = ComputeIncumbentFromNode(root_node_, legal_actions, &max_raw_prob);
    } else if (!res.policy.empty()) {
      incumbent = ComputeIncumbentFromPriors(res.policy, legal_actions, &max_raw_prob);
    } else {
      ActionsAndProbs raw_eval = evaluators_[state.CurrentPlayer()]->Prior(state);
      incumbent = ComputeIncumbentFromPriors(raw_eval, legal_actions, &max_raw_prob);
    }
  }

  sel.incumbent_action = incumbent;
  sel.raw_prior_max_prob = max_raw_prob;
  sel.raw_prior_max_prob_hex = absl::StrFormat("%a", max_raw_prob);

  // 3. Search acceptance classification
  bool is_forced = (legal_actions.size() <= 1);
  bool search_accepted = (!is_forced &&
                          !res.used_fallback &&
                          res.fallback_reason == "none" &&
                          res.simulations_completed > 0 &&
                          !res.timeout_status);
  sel.search_accepted = search_accepted;

  // 4. Selection reason
  if (is_forced) {
    sel.selection_reason = "forced_single_action";
  } else if (search_accepted) {
    sel.selection_reason = "accepted_search";
  } else if (res.fallback_reason == "non_strategic_state") {
    sel.selection_reason = "bypass_non_strategic";
  } else if (res.fallback_reason == "low_coverage") {
    sel.selection_reason = "fallback_low_coverage";
  } else if (res.fallback_reason == "zero_visits") {
    sel.selection_reason = "fallback_zero_visits";
  } else if (res.fallback_reason == "timeout" || res.timeout_status) {
    sel.selection_reason = "fallback_timeout";
  } else if (res.fallback_reason == "max_nodes") {
    sel.selection_reason = "fallback_max_nodes";
  } else {
    sel.selection_reason = absl::StrCat("fallback_", res.fallback_reason);
  }

  // 5. Final action selection
  if (config_.action_selection_mode == DuneActionSelectionMode::kHistoricalSample) {
    sel.selected_action = SampleAction(res.policy, r_val).first;
  } else {
    // kPreserveRawGreedy
    if (config_.temperature > 0.0) {
      sel.selected_action = SampleAction(res.policy, r_val).first;
    } else {
      // temperature == 0.0
      if (search_accepted) {
        sel.selected_action = SampleAction(res.policy, r_val).first;
      } else {
        // Fallback / bypass: return exact raw-greedy incumbent action
        sel.selected_action = incumbent;
      }
    }
  }

  // 6. Selected raw probability
  if (sel.selected_action == incumbent) {
    sel.raw_prior_selected_prob = max_raw_prob;
  } else {
    double sel_p = 0.0;
    if (root_node_ != nullptr && root_node_->child_info.count(sel.selected_action)) {
      sel_p = root_node_->child_info.at(sel.selected_action).raw_prior;
    } else {
      for (const auto& ap : res.policy) {
        if (ap.first == sel.selected_action) {
          sel_p = ap.second;
          break;
        }
      }
    }
    sel.raw_prior_selected_prob = sel_p;
  }
  sel.raw_prior_selected_prob_hex = absl::StrFormat("%a", sel.raw_prior_selected_prob);

  // 7. Update diagnostics on last_search_result_
  last_search_result_.diagnostics.selected_action = sel.selected_action;
  last_search_result_.diagnostics.incumbent_action = sel.incumbent_action;
  last_search_result_.diagnostics.raw_prior_max_prob = sel.raw_prior_max_prob;
  last_search_result_.diagnostics.raw_prior_selected_prob = sel.raw_prior_selected_prob;
  last_search_result_.diagnostics.raw_prior_max_prob_hex = sel.raw_prior_max_prob_hex;
  last_search_result_.diagnostics.raw_prior_selected_prob_hex = sel.raw_prior_selected_prob_hex;
  last_search_result_.diagnostics.selection_mode = sel.selection_mode;
  last_search_result_.diagnostics.selection_reason = sel.selection_reason;
  last_search_result_.diagnostics.search_accepted = sel.search_accepted;

  return sel;
}

Action DunePUCTISMCTSBot::Step(const State& state) {
  DuneSearchResult res = RunSearch(state);
  ControllerSelection sel = SelectAction(state, res);
  return sel.selected_action;
}

ActionsAndProbs DunePUCTISMCTSBot::GetPolicy(const State& state) {
  return RunSearch(state).policy;
}

std::pair<ActionsAndProbs, Action> DunePUCTISMCTSBot::StepWithPolicy(const State& state) {
  DuneSearchResult res = RunSearch(state);
  ControllerSelection sel = SelectAction(state, res);
  return {res.policy, sel.selected_action};
}

SearchDiagnostics DunePUCTISMCTSBot::GetRootDiagnostics(const State& state, int min_visit_threshold, Action chosen_action) const {
  // Clamp here too, not just at the call site: this is a public entry point and
  // a threshold below 1 would let a zero-visit action reach `return_sum /
  // visits` (search finding 5).
  const int min_visits = EffectiveMinVisitThreshold(min_visit_threshold);
  SearchDiagnostics diag;
  diag.root_value = 0.0;
  diag.total_root_visits = 0;
  diag.num_covered_actions = 0;
  diag.covered_prior_mass = 0.0;

  auto key = GetStateKey(state);
  auto iter = nodes_.find(key);

  std::vector<Action> legal_actions = state.LegalActions();
  diag.actions = legal_actions;
  diag.visit_counts.reserve(legal_actions.size());
  diag.q_values.reserve(legal_actions.size());
  diag.priors.reserve(legal_actions.size());

  if (iter != nodes_.end()) {
    DuneISMCTSNode* node = iter->second;
    diag.root_value = node->cached_value;
    diag.total_root_visits = node->total_visits;

    ActionsAndProbs normalized_priors = FilterAndNormalizePriors(node, legal_actions);
    for (size_t i = 0; i < legal_actions.size(); ++i) {
      Action action = legal_actions[i];
      double prior = normalized_priors[i].second;
      diag.priors.push_back(prior);

      int visits = 0;
      double return_sum = 0.0;
      auto child_iter = node->child_info.find(action);
      if (child_iter != node->child_info.end()) {
        visits = child_iter->second.visits;
        return_sum = child_iter->second.return_sum;
      }
      diag.visit_counts.push_back(visits);

      // visits >= min_visits >= 1, so this division can never be 0/0. NaN Q
      // values used to reach both telemetry and PruneForcedPlayouts' PUCT
      // comparisons, where they compare false and silently disable pruning.
      double q_value = diag.root_value;
      if (visits >= min_visits) {
        q_value = return_sum / visits;
        diag.num_covered_actions++;
        diag.covered_prior_mass += prior;
      }
      diag.q_values.push_back(q_value);
    }

    // Item-4 telemetry baseline: the untransformed network prior, normalized
    // identically to `priors` over the same legal-action set, so
    // KL(visits‖raw_prior) is comparable to the noise-free 200-sim role
    // baselines. Populated unconditionally (WO-15): it previously appeared only
    // under Dirichlet noise, which left consumers reading the temperature-scaled
    // tree prior as the "raw" baseline whenever root_prior_temperature != 1.
    ActionsAndProbs normalized_raw = FilterAndNormalizeRawPriors(node, legal_actions);
    diag.raw_priors.reserve(normalized_raw.size());
    for (const auto& ap : normalized_raw) diag.raw_priors.push_back(ap.second);
  } else {
    auto t_inf_start = std::chrono::steady_clock::now();
    ActionsAndProbs raw_priors = evaluators_[state.CurrentPlayer()]->Prior(state);
    auto t_inf_end = std::chrono::steady_clock::now();
    accumulated_inference_time_ms_ += std::chrono::duration<double, std::milli>(t_inf_end - t_inf_start).count();
    inference_count_this_search_++;
    absl::flat_hash_map<Action, double> prior_map;
    for (const auto& ap : raw_priors) {
      prior_map[ap.first] = ap.second;
    }
    double sum = 0.0;
    for (Action action : legal_actions) {
      double p = 1e-5;
      auto p_iter = prior_map.find(action);
      if (p_iter != prior_map.end()) {
        p = p_iter->second;
      }
      diag.priors.push_back(p);
      sum += p;
    }
    if (sum > 0.0) {
      for (double& p : diag.priors) p /= sum;
    } else {
      for (double& p : diag.priors) p = 1.0 / legal_actions.size();
    }

    diag.visit_counts.assign(legal_actions.size(), 0);
    diag.q_values.assign(legal_actions.size(), 0.0);
  }

  // 18A diagnostics calculations
  if (!simulation_depths_this_search_.empty()) {
    double max_d = 0.0;
    double sum_d = 0.0;
    for (int d : simulation_depths_this_search_) {
      max_d = std::max(max_d, static_cast<double>(d));
      sum_d += d;
    }
    diag.max_depth = max_d;
    diag.mean_depth = sum_d / simulation_depths_this_search_.size();

    std::vector<int> sorted_depths = simulation_depths_this_search_;
    std::sort(sorted_depths.begin(), sorted_depths.end());
    size_t p95_idx = std::min<size_t>(
        sorted_depths.size() - 1,
        static_cast<size_t>(0.95 * (sorted_depths.size() - 1)));
    diag.p95_depth = sorted_depths[p95_idx];
  } else {
    diag.max_depth = 0.0;
    diag.mean_depth = 0.0;
    diag.p95_depth = 0.0;
  }

  diag.deepest_simulated_round = max_round_this_search_;
  diag.max_decision_depth = max_decision_depth_this_search_;
  diag.mean_decision_depth = num_sims_this_search_ > 0
      ? sum_decision_depth_this_search_ / num_sims_this_search_
      : 0.0;
  diag.terminal_leaf_fraction = num_sims_this_search_ > 0
      ? static_cast<double>(terminal_leaf_simulations_count_) / num_sims_this_search_
      : 0.0;
  diag.unique_nodes = visited_nodes_this_search_.size();
  diag.inference_count = inference_count_this_search_;

  diag.forced_visit_counts.assign(legal_actions.size(), 0);
  diag.pruned_visit_counts.assign(legal_actions.size(), 0);

  double total_visits = 0.0;
  for (int v : diag.visit_counts) {
    total_visits += v;
  }
  std::vector<double> search_probs(legal_actions.size());
  if (total_visits > 0.0) {
    for (size_t i = 0; i < legal_actions.size(); ++i) {
      search_probs[i] = diag.visit_counts[i] / total_visits;
    }
  } else {
    for (size_t i = 0; i < legal_actions.size(); ++i) {
      search_probs[i] = 1.0 / legal_actions.size();
    }
  }

  // Every `raw_*`/`vs_raw` diagnostic below is defined against the UNTRANSFORMED
  // network prior, so they must read `raw_priors`, not the tree prior: `priors`
  // is the post-Dirichlet mixture at a noised root and carries
  // `root_prior_temperature` scaling, either of which would make a field named
  // "raw" report against a reshaped distribution (inflating KL, and moving the
  // prior rank/argmax the override flags are derived from). `raw_priors` is
  // empty only when the root was not in the tree, and on that path `priors` is
  // itself built straight from the evaluator's prior, so it IS raw. Same
  // fallback idiom as dune_online_search_collector.cc's item-4 telemetry.
  const std::vector<double>& raw_baseline =
      diag.raw_priors.size() == legal_actions.size() ? diag.raw_priors
                                                     : diag.priors;

  diag.raw_to_search_policy_kl = 0.0;
  for (size_t i = 0; i < legal_actions.size(); ++i) {
    double p_search = search_probs[i];
    double p_raw = raw_baseline[i];
    if (p_search > 0.0) {
      diag.raw_to_search_policy_kl += p_search * std::log(p_search / std::max(p_raw, 1e-12));
    }
  }

  size_t chosen_idx = 0;
  bool found_chosen = false;
  if (chosen_action != kInvalidAction) {
    for (size_t i = 0; i < legal_actions.size(); ++i) {
      if (legal_actions[i] == chosen_action) {
        chosen_idx = i;
        found_chosen = true;
        break;
      }
    }
  }

  if (!found_chosen) {
    double max_prob = -1.0;
    for (size_t i = 0; i < legal_actions.size(); ++i) {
      if (search_probs[i] > max_prob) {
        max_prob = search_probs[i];
        chosen_idx = i;
      }
    }
  }

  diag.chosen_action_raw_prior_probability = raw_baseline[chosen_idx];

  int rank = 1;
  for (size_t i = 0; i < legal_actions.size(); ++i) {
    if (raw_baseline[i] > diag.chosen_action_raw_prior_probability) {
      rank++;
    }
  }
  diag.chosen_action_raw_prior_rank = rank;

  size_t raw_argmax_idx = 0;
  double max_raw_prob = -1.0;
  for (size_t i = 0; i < legal_actions.size(); ++i) {
    if (raw_baseline[i] > max_raw_prob) {
      max_raw_prob = raw_baseline[i];
      raw_argmax_idx = i;
    }
  }
  diag.action_changed_vs_raw_argmax = (chosen_idx != raw_argmax_idx);

  return diag;
}

}  // namespace open_spiel
