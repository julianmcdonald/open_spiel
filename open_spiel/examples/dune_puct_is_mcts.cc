#include "dune_puct_is_mcts.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <utility>
#include <vector>
#include <chrono>

#include "open_spiel/abseil-cpp/absl/random/discrete_distribution.h"
#include "open_spiel/abseil-cpp/absl/random/distributions.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"
#include "open_spiel/games/dune_imperium/dune_imperium.h"
#include "dune_seed_utils.h"

namespace open_spiel {

// ---------------------------------------------------------------------------
// Strategic State Classification Implementation
// ---------------------------------------------------------------------------

bool IsStrategicAction(const std::string& action_str) {
  if (action_str.rfind("PlaceAgent", 0) == 0 ||
      action_str.rfind("Deploy ", 0) == 0 ||
      action_str.rfind("CombatCommit", 0) == 0 ||
      action_str.rfind("Buy", 0) == 0 ||
      action_str.rfind("AcquireTech", 0) == 0 ||
      action_str.rfind("AcquireTleilaxu", 0) == 0 ||
      action_str.rfind("Shipping", 0) == 0 ||
      action_str.rfind("SignetRing", 0) == 0 ||
      action_str.rfind("tech_", 0) == 0 ||
      action_str.rfind("ResearchBranch", 0) == 0 ||
      action_str.rfind("SelectAgentCard", 0) == 0 ||
      action_str.rfind("SelectGraftPartner", 0) == 0 ||
      action_str.rfind("PlayAgentSolo", 0) == 0 ||
      action_str.rfind("PlayPlotIntrigue", 0) == 0 ||
      action_str.rfind("PlayCombatIntrigue", 0) == 0 ||
      action_str.rfind("IntrigueChoice", 0) == 0 ||
      action_str.rfind("CombatPass", 0) == 0 ||
      action_str.rfind("AgentPass", 0) == 0 ||
      action_str.rfind("Reveal", 0) == 0 ||
      action_str.rfind("ConvertSpecimenToTroop", 0) == 0 ||
      action_str.rfind("TrashHand[", 0) == 0 ||
      action_str.rfind("TrashDiscard[", 0) == 0 ||
      action_str.rfind("TrashInPlay[", 0) == 0 ||
      action_str.rfind("Trash Intrigue ", 0) == 0 ||
      action_str.rfind("StealIntrigue[", 0) == 0 ||
      action_str == "TrashSkip" ||
      action_str == "PlayKwisatzHaderach" ||
      action_str == "FamilyAtomics" ||
      action_str == "AgentPlayGeneric") {
    return true;
  }
  return false;
}

bool IsStrategicState(const State& state, Player searched_player) {
  if (state.CurrentPlayer() != searched_player) return false;
  std::vector<Action> legal_actions = state.LegalActions();
  if (legal_actions.size() < 2) return false; // Must be a choice

  const dune_imperium::DuneImperiumState& dune_state =
      static_cast<const dune_imperium::DuneImperiumState&>(state);
  for (Action a : legal_actions) {
    std::string action_str = dune_state.ActionToString(state.CurrentPlayer(), a);
    if (IsStrategicAction(action_str)) return true;
  }
  return false;
}

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
    : DunePUCTISMCTSBot(
          DuneSearchConfig{
              max_simulations,
              10000.0, // relative_time_budget_ms
              50000,   // max_nodes
              puct_c,
              use_opponent_model ? SearchOpponentMode::kPolicy : SearchOpponentMode::kMaxN,
              temperature,
              opponent_temperature,
              max_world_samples,
              utility_divisor, // utility_divisor = utility_divisor
              2,           // min_visit_threshold
              0.50,        // covered_prior_threshold
              seed,
              final_policy_type,
              dirichlet_epsilon,
              dirichlet_alpha,
              use_observation_string,
              verbose_diagnostics
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
  max_depth_this_search_ = 0;
  sum_depth_this_search_ = 0.0;
  num_sims_this_search_ = 0;
  total_lookups_ = 0;
  reused_lookups_ = 0;
  opponent_prior_cache_.clear();
}

const DuneSearchResult& DunePUCTISMCTSBot::GetLastSearchResult() const {
  return last_search_result_;
}

std::pair<Player, std::string> DunePUCTISMCTSBot::GetStateKey(const State& state) const {
  if (config_.use_observation_string) {
    return {state.CurrentPlayer(), state.ObservationString()};
  } else {
    return {state.CurrentPlayer(), state.InformationStateString()};
  }
}

std::unique_ptr<State> DunePUCTISMCTSBot::SampleRootState(const State& state, int sim_index) {
  if (config_.max_world_samples == -1) {
    return ResampleFromInfostate(state, sim_index);
  } else if (root_samples_.size() < static_cast<size_t>(config_.max_world_samples)) {
    root_samples_.push_back(ResampleFromInfostate(state, sim_index));
    return root_samples_.back()->Clone();
  } else {
    uint64_t sample_seed = dune_seed::Combine(config_.seed, dune_seed::kStreamSearchSampling, sim_index);
    std::mt19937 sample_rng(sample_seed);
    int idx = absl::Uniform(sample_rng, 0u, root_samples_.size());
    return root_samples_[idx]->Clone();
  }
}

std::unique_ptr<State> DunePUCTISMCTSBot::ResampleFromInfostate(const State& state, int sim_index) {
  uint64_t resample_seed = dune_seed::Combine(config_.seed, dune_seed::kStreamSearchSampling, sim_index);
  std::mt19937 resample_rng(resample_seed);
  return state.ResampleFromInfostate(state.CurrentPlayer(),
                                     [&resample_rng]() { return absl::Uniform(resample_rng, 0.0, 1.0); });
}

DuneISMCTSNode* DunePUCTISMCTSBot::CreateNewNode(const State& state) {
  auto key = GetStateKey(state);
  node_pool_.push_back(std::make_unique<DuneISMCTSNode>());
  DuneISMCTSNode* node = node_pool_.back().get();
  nodes_[key] = node;
  return node;
}

DuneISMCTSNode* DunePUCTISMCTSBot::LookupNode(const State& state) {
  total_lookups_++;
  auto key = GetStateKey(state);
  auto iter = nodes_.find(key);
  if (iter == nodes_.end()) {
    return nullptr;
  }
  reused_lookups_++;
  return iter->second;
}

DuneISMCTSNode* DunePUCTISMCTSBot::LookupOrCreateNode(const State& state) {
  DuneISMCTSNode* node = LookupNode(state);
  if (node != nullptr) {
    return node;
  }
  return CreateNewNode(state);
}

void DunePUCTISMCTSBot::InitializePriorsAndValue(DuneISMCTSNode* node, const State& state) {
  if (node->priors_initialized) {
    return;
  }
  Player cur_player = state.CurrentPlayer();
  auto eval_res = evaluators_[cur_player]->PriorAndEvaluate(state);
  inference_count_this_search_++;

  const ActionsAndProbs& priors = eval_res.first;
  for (const auto& action_prob : priors) {
    Action action = action_prob.first;
    double prob = action_prob.second;
    node->child_info[action] = DuneChildInfo{0, 0.0, prob};
  }

  node->cached_values.resize(state.NumPlayers());
  bool evaluators_differ = false;
  for (int p = 0; p < state.NumPlayers(); ++p) {
    if (evaluators_[p] != evaluators_[cur_player]) {
      evaluators_differ = true;
      break;
    }
  }

  if (!evaluators_differ) {
    node->cached_values = eval_res.second;
  } else {
    for (int p = 0; p < state.NumPlayers(); ++p) {
      if (p == cur_player) {
        node->cached_values[p] = eval_res.second[p];
      } else {
        node->cached_values[p] = evaluators_[p]->Evaluate(state)[p];
        inference_count_this_search_++;
      }
    }
  }

  if (cur_player >= 0 && cur_player < static_cast<int>(node->cached_values.size())) {
    node->cached_value = node->cached_values[cur_player];
  } else {
    node->cached_value = 0.0;
  }

  node->priors_initialized = true;
}

ActionsAndProbs DunePUCTISMCTSBot::FilterAndNormalizePriors(
    DuneISMCTSNode* node, const std::vector<Action>& legal_actions) const {
  ActionsAndProbs result;
  result.reserve(legal_actions.size());
  double sum_prior = 0.0;
  
  for (Action action : legal_actions) {
    double prior = 1e-5; // fallback prior
    auto iter = node->child_info.find(action);
    if (iter != node->child_info.end()) {
      prior = iter->second.prior;
    }
    result.push_back({action, prior});
    sum_prior += prior;
  }
  
  if (sum_prior > 0.0) {
    for (auto& action_prob : result) {
      action_prob.second /= sum_prior;
    }
  } else {
    double uniform_prob = 1.0 / legal_actions.size();
    for (auto& action_prob : result) {
      action_prob.second = uniform_prob;
    }
  }
  return result;
}

Action DunePUCTISMCTSBot::SelectActionTreePolicy(
    DuneISMCTSNode* node, const std::vector<Action>& legal_actions) {
  ActionsAndProbs normalized = FilterAndNormalizePriors(node, legal_actions);

  double fpu_val = node->cached_value;

  Action best_action = kInvalidAction;
  double best_puct_val = -std::numeric_limits<double>::infinity();
  double tie_tolerance = 1e-5;
  std::vector<Action> best_actions;

  double parent_visits_sqrt = std::sqrt(std::max(1, node->total_visits));

  for (const auto& action_prob : normalized) {
    Action a = action_prob.first;
    double prior = action_prob.second;
    auto& child = node->child_info[a];
    
    if (child.prior == 0.0) {
      child.prior = prior;
    }

    double q_val = (child.visits > 0) ? child.value() : fpu_val;
    double u = config_.puct_c * prior * parent_visits_sqrt / (1.0 + child.visits);
    double puct_val = q_val + u;

    if (puct_val > best_puct_val + tie_tolerance) {
      best_puct_val = puct_val;
      best_actions = {a};
    } else if (puct_val >= best_puct_val - tie_tolerance) {
      best_actions.push_back(a);
    }
  }

  SPIEL_CHECK_GE(best_actions.size(), 1);
  if (best_actions.size() == 1) {
    return best_actions[0];
  } else {
    // Deterministic tie-breaking RNG derived from stream seed
    uint64_t tie_seed = dune_seed::Combine(config_.seed, dune_seed::kStreamMCTS, node->total_visits);
    std::mt19937 tie_rng(tie_seed);
    return best_actions[absl::Uniform(tie_rng, 0u, best_actions.size())];
  }
}

std::vector<double> DunePUCTISMCTSBot::RunSimulation(State* state, int depth, int sim_index) {
  max_depth_this_search_ = std::max(max_depth_this_search_, depth);

  if (state->IsTerminal()) {
    std::vector<double> returns = state->Returns();
    for (double& r : returns) {
      r /= config_.utility_divisor;
    }
    sum_depth_this_search_ += depth;
    num_sims_this_search_++;
    return returns;
  } else if (state->IsChanceNode()) {
    // Chance outcome RNG derived deterministically from stream seed
    uint64_t chance_seed = dune_seed::DeriveSeed(config_.seed, dune_seed::kStreamChance, sim_index, depth);
    std::mt19937 chance_rng(chance_seed);
    double r_num = absl::Uniform(chance_rng, 0.0, 1.0);
    Action chance_action = SampleAction(state->ChanceOutcomes(), r_num).first;
    state->ApplyAction(chance_action);
    return RunSimulation(state, depth + 1, sim_index);
  }

  std::vector<Action> legal_actions = state->LegalActions();
  if (legal_actions.size() == 1) {
    state->ApplyAction(legal_actions[0]);
    return RunSimulation(state, depth + 1, sim_index);
  }
  Player cur_player = state->CurrentPlayer();

  if (config_.opponent_mode == SearchOpponentMode::kPolicy && cur_player != searching_player_) {
    auto key = GetStateKey(*state);
    auto iter = opponent_prior_cache_.find(key);
    ActionsAndProbs prior;
    if (iter != opponent_prior_cache_.end()) {
      prior = iter->second;
    } else {
      prior = evaluators_[cur_player]->Prior(*state);
      inference_count_this_search_++;
      opponent_prior_cache_[key] = prior;
    }
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
        double sum = 0.0;
        double inv_temp = 1.0 / temp;
        for (const auto& ap : prior) {
          double p = std::pow(std::max(ap.second, 1e-12), inv_temp);
          scaled.push_back({ap.first, p});
          sum += p;
        }
        for (auto& ap : scaled) {
          ap.second /= sum;
        }
        chosen_action = SampleAction(scaled, absl::Uniform(sample_rng, 0.0, 1.0)).first;
      }
    }

    if (chosen_action != kInvalidAction) {
      state->ApplyAction(chosen_action);
      return RunSimulation(state, depth + 1, sim_index);
    } else {
      std::vector<double> returns = state->Returns();
      for (double& r : returns) {
        r /= config_.utility_divisor;
      }
      sum_depth_this_search_ += depth;
      num_sims_this_search_++;
      return returns;
    }
  }

  DuneISMCTSNode* node = LookupOrCreateNode(*state);
  SPIEL_CHECK_TRUE(node != nullptr);

  InitializePriorsAndValue(node, *state); // Combined evaluation expansion!

  if (node->total_visits == -1) {
    node->total_visits = 0;
    sum_depth_this_search_ += depth;
    num_sims_this_search_++;
    return node->cached_values;
  }

  Action chosen_action = SelectActionTreePolicy(node, legal_actions);
  SPIEL_CHECK_NE(chosen_action, kInvalidAction);

  node->total_visits++;
  node->child_info[chosen_action].visits++;

  state->ApplyAction(chosen_action);
  std::vector<double> returns = RunSimulation(state, depth + 1, sim_index);
  
  // Multi-player Max-N backup
  node->child_info[chosen_action].return_sum += returns[cur_player];
  return returns;
}

DuneSearchResult DunePUCTISMCTSBot::RunSearch(const State& state) {
  root_samples_.clear();
  opponent_prior_cache_.clear();
  inference_count_this_search_ = 0;

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
    }
  }

  auto start_time = std::chrono::steady_clock::now();
  DuneSearchResult result;
  result.timeout_status = false;

  std::vector<Action> legal_actions = state.LegalActions();
  if (legal_actions.size() == 1) {
    result.policy = {{legal_actions[0], 1.0}};
    result.simulations_completed = 0;
    result.elapsed_time_ms = 0.0;
    result.inference_count = inference_count_this_search_;
    last_search_result_ = result;
    return result;
  }

  if (config_.check_strategic_state && !IsStrategicState(state, state.CurrentPlayer())) {
    result.policy = evaluators_[state.CurrentPlayer()]->Prior(state);
    inference_count_this_search_++;
    result.simulations_completed = 0;
    result.elapsed_time_ms = 0.0;
    result.fallback_reason = "non_strategic_state";
    result.inference_count = inference_count_this_search_;
    last_search_result_ = result;
    return result;
  }

  searching_player_ = state.CurrentPlayer();

  root_node_ = LookupOrCreateNode(state);
  InitializePriorsAndValue(root_node_, state);

  // Apply Dirichlet noise exactly once at the root node's priors (if enabled)
  if (config_.dirichlet_epsilon > 0.0 && config_.dirichlet_alpha > 0.0) {
    if (legal_actions.size() > 1) {
      uint64_t noise_seed = dune_seed::Combine(config_.seed, dune_seed::kStreamBlueprint, 0);
      std::mt19937 noise_rng(noise_seed);
      std::gamma_distribution<double> gamma_dist(config_.dirichlet_alpha, 1.0);
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
        child.prior = (1.0 - config_.dirichlet_epsilon) * child.prior + config_.dirichlet_epsilon * d_noise;
      }
    }
  }

  int sim = 0;
  for (; sim < config_.max_simulations; ++sim) {
    // Check relative time budget limit
    auto now = std::chrono::steady_clock::now();
    double elapsed_ms = std::chrono::duration<double, std::milli>(now - start_time).count();
    if (elapsed_ms >= config_.relative_time_budget_ms) {
      result.timeout_status = true;
      result.fallback_reason = "timeout";
      break;
    }

    // Check Node/Memory limit
    if (static_cast<int>(node_pool_.size()) >= config_.max_nodes) {
      result.fallback_reason = "max_nodes";
      break;
    }

    std::unique_ptr<State> sampled_root_state = SampleRootState(state, sim);
    RunSimulation(sampled_root_state.get(), 0, sim);
  }

  auto end_time = std::chrono::steady_clock::now();
  result.elapsed_time_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();
  result.simulations_completed = sim;

  // Coverage Check & Fallback Logic
  double covered_prior_mass = 0.0;
  int num_covered_actions = 0;
  ActionsAndProbs normalized_priors = FilterAndNormalizePriors(root_node_, legal_actions);
  for (size_t i = 0; i < legal_actions.size(); ++i) {
    Action action = legal_actions[i];
    double prior = normalized_priors[i].second;
    auto iter = root_node_->child_info.find(action);
    int visits = (iter != root_node_->child_info.end()) ? iter->second.visits : 0;
    if (visits >= config_.min_visit_threshold) {
      covered_prior_mass += prior;
      num_covered_actions++;
    }
  }

  int req_covered_actions = std::min<int>(3, legal_actions.size());
  if (num_covered_actions < req_covered_actions || covered_prior_mass < config_.covered_prior_threshold) {
    result.policy = normalized_priors;
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

  result.diagnostics = GetRootDiagnostics(state, config_.min_visit_threshold);
  result.inference_count = inference_count_this_search_;
  last_search_result_ = result;
  return result;
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
    double uniform_prob = 1.0 / legal_actions.size();
    policy.reserve(legal_actions.size());
    for (Action action : legal_actions) {
      policy.push_back({action, uniform_prob});
    }
    return policy;
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
    double sum_pow = 0.0;
    std::vector<double> pow_visits(action_visits.size());
    double inv_temp = 1.0 / config_.temperature;
    for (size_t i = 0; i < action_visits.size(); ++i) {
      pow_visits[i] = std::pow(action_visits[i].second, inv_temp);
      sum_pow += pow_visits[i];
    }
    for (size_t i = 0; i < action_visits.size(); ++i) {
      policy.push_back({action_visits[i].first, sum_pow > 0.0 ? (pow_visits[i] / sum_pow) : (1.0 / legal_actions.size())});
    }
  }

  return policy;
}

Action DunePUCTISMCTSBot::Step(const State& state) {
  DuneSearchResult res = RunSearch(state);
  search_count_++;
  uint64_t step_seed = dune_seed::Combine(config_.seed, dune_seed::kStreamBlueprint, search_count_);
  std::mt19937 step_rng(step_seed);
  double r_val = absl::Uniform(step_rng, 0.0, 1.0);
  return SampleAction(res.policy, r_val).first;
}

ActionsAndProbs DunePUCTISMCTSBot::GetPolicy(const State& state) {
  return RunSearch(state).policy;
}

std::pair<ActionsAndProbs, Action> DunePUCTISMCTSBot::StepWithPolicy(const State& state) {
  DuneSearchResult res = RunSearch(state);
  search_count_++;
  uint64_t step_seed = dune_seed::Combine(config_.seed, dune_seed::kStreamBlueprint, search_count_);
  std::mt19937 step_rng(step_seed);
  double r_val = absl::Uniform(step_rng, 0.0, 1.0);
  Action sampled_action = SampleAction(res.policy, r_val).first;
  return {res.policy, sampled_action};
}

SearchDiagnostics DunePUCTISMCTSBot::GetRootDiagnostics(const State& state, int min_visit_threshold) const {
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

      double q_value = diag.root_value;
      if (visits >= min_visit_threshold) {
        q_value = return_sum / visits;
        diag.num_covered_actions++;
        diag.covered_prior_mass += prior;
      }
      diag.q_values.push_back(q_value);
    }
  } else {
    ActionsAndProbs raw_priors = evaluators_[state.CurrentPlayer()]->Prior(state);
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
  return diag;
}

}  // namespace open_spiel
