#include "dune_puct_is_mcts.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <utility>
#include <vector>

#include "open_spiel/abseil-cpp/absl/random/discrete_distribution.h"
#include "open_spiel/abseil-cpp/absl/random/distributions.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"
#include "dune_seed_utils.h"

namespace open_spiel {

DunePUCTISMCTSBot::DunePUCTISMCTSBot(
    uint64_t seed, std::shared_ptr<algorithms::Evaluator> evaluator, double puct_c,
    int max_simulations, int max_world_samples, double temperature,
    double dirichlet_epsilon, double dirichlet_alpha, double value_scale,
    bool use_observation_string,
    DuneISMCTSFinalPolicyType final_policy_type,
    bool use_opponent_model, double opponent_temperature,
    bool verbose_diagnostics)
    : rng_(dune_seed::MakeRng32(seed)),
      evaluator_(evaluator),
      puct_c_(puct_c),
      max_simulations_(max_simulations),
      max_world_samples_(max_world_samples),
      temperature_(temperature),
      dirichlet_epsilon_(dirichlet_epsilon),
      dirichlet_alpha_(dirichlet_alpha),
      value_scale_(value_scale),
      use_observation_string_(use_observation_string),
      final_policy_type_(final_policy_type),
      use_opponent_model_(use_opponent_model),
      opponent_temperature_(opponent_temperature),
      verbose_diagnostics_(verbose_diagnostics),
      root_node_(nullptr) {}

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
}

std::pair<Player, std::string> DunePUCTISMCTSBot::GetStateKey(const State& state) const {
  if (use_observation_string_) {
    return {state.CurrentPlayer(), state.ObservationString()};
  } else {
    return {state.CurrentPlayer(), state.InformationStateString()};
  }
}

std::unique_ptr<State> DunePUCTISMCTSBot::SampleRootState(const State& state) {
  if (max_world_samples_ == -1) {
    return ResampleFromInfostate(state);
  } else if (root_samples_.size() < static_cast<size_t>(max_world_samples_)) {
    root_samples_.push_back(ResampleFromInfostate(state));
    return root_samples_.back()->Clone();
  } else {
    int idx = absl::Uniform(rng_, 0u, root_samples_.size());
    return root_samples_[idx]->Clone();
  }
}

std::unique_ptr<State> DunePUCTISMCTSBot::ResampleFromInfostate(const State& state) {
  return state.ResampleFromInfostate(state.CurrentPlayer(),
                                     [this]() { return RandomNumber(); });
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

void DunePUCTISMCTSBot::InitializePriors(DuneISMCTSNode* node, const State& state) {
  if (node->priors_initialized) {
    return;
  }
  ActionsAndProbs priors = evaluator_->Prior(state);
  for (const auto& action_prob : priors) {
    Action action = action_prob.first;
    double prob = action_prob.second;
    node->child_info[action] = DuneChildInfo{0, 0.0, prob};
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
  // Filter and normalize priors dynamically
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
    
    // Remember normalized prior for child visits tracking if not already set
    if (child.prior == 0.0) {
      child.prior = prior;
    }

    double q_val = (child.visits > 0) ? child.value() : fpu_val;
    double u = puct_c_ * prior * parent_visits_sqrt / (1.0 + child.visits);
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
    return best_actions[absl::Uniform(rng_, 0u, best_actions.size())];
  }
}

std::vector<double> DunePUCTISMCTSBot::RunSimulation(State* state, int depth) {
  max_depth_this_search_ = std::max(max_depth_this_search_, depth);

  if (state->IsTerminal()) {
    std::vector<double> returns = state->Returns();
    for (double& r : returns) {
      r /= value_scale_;
    }
    sum_depth_this_search_ += depth;
    num_sims_this_search_++;
    return returns;
  } else if (state->IsChanceNode()) {
    Action chance_action = SampleAction(state->ChanceOutcomes(), RandomNumber()).first;
    state->ApplyAction(chance_action);
    return RunSimulation(state, depth + 1);
  }

  std::vector<Action> legal_actions = state->LegalActions();
  if (legal_actions.size() == 1) {
    state->ApplyAction(legal_actions[0]);
    return RunSimulation(state, depth + 1);
  }
  Player cur_player = state->CurrentPlayer();

  if (use_opponent_model_ && cur_player != searching_player_) {
    ActionsAndProbs prior = evaluator_->Prior(*state);
    Action chosen_action = kInvalidAction;
    if (prior.empty()) {
      if (!legal_actions.empty()) {
        chosen_action = legal_actions[absl::Uniform(rng_, 0u, legal_actions.size())];
      }
    } else {
      double temp = opponent_temperature_;
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
        chosen_action = SampleAction(scaled, RandomNumber()).first;
      }
    }

    if (chosen_action != kInvalidAction) {
      state->ApplyAction(chosen_action);
      return RunSimulation(state, depth + 1);
    } else {
      std::vector<double> returns = state->Returns();
      for (double& r : returns) {
        r /= value_scale_;
      }
      sum_depth_this_search_ += depth;
      num_sims_this_search_++;
      return returns;
    }
  }
  
  DuneISMCTSNode* node = LookupOrCreateNode(*state);
  SPIEL_CHECK_TRUE(node != nullptr);

  InitializePriors(node, *state);

  if (node->total_visits == -1) {
    node->total_visits = 0;
    sum_depth_this_search_ += depth;
    num_sims_this_search_++;
    std::vector<double> values = evaluator_->Evaluate(*state);
    node->cached_value = values[state->CurrentPlayer()];
    return values;
  }

  Action chosen_action = SelectActionTreePolicy(node, legal_actions);
  SPIEL_CHECK_NE(chosen_action, kInvalidAction);

  node->total_visits++;
  node->child_info[chosen_action].visits++;

  state->ApplyAction(chosen_action);
  std::vector<double> returns = RunSimulation(state, depth + 1);
  
  // Multi-player Max-N backup
  node->child_info[chosen_action].return_sum += returns[cur_player];
  return returns;
}

ActionsAndProbs DunePUCTISMCTSBot::RunSearch(const State& state) {
  Reset();
  SPIEL_CHECK_EQ(state.GetGame()->GetType().dynamics, GameType::Dynamics::kSequential);

  std::vector<Action> legal_actions = state.LegalActions();
  if (legal_actions.size() == 1) {
    return {{legal_actions[0], 1.0}};
  }

  searching_player_ = state.CurrentPlayer();

  root_node_ = CreateNewNode(state);
  InitializePriors(root_node_, state);

  // Apply Dirichlet noise exactly once at the root node's priors
  if (dirichlet_epsilon_ > 0.0 && dirichlet_alpha_ > 0.0) {
    if (legal_actions.size() > 1) {
      std::gamma_distribution<double> gamma_dist(dirichlet_alpha_, 1.0);
      std::vector<double> noise(legal_actions.size());
      double sum_noise = 0.0;
      for (size_t i = 0; i < legal_actions.size(); ++i) {
        noise[i] = gamma_dist(rng_);
        sum_noise += noise[i];
      }
      for (size_t i = 0; i < legal_actions.size(); ++i) {
        Action action = legal_actions[i];
        double d_noise = sum_noise > 0.0 ? (noise[i] / sum_noise) : (1.0 / legal_actions.size());
        auto& child = root_node_->child_info[action];
        child.prior = (1.0 - dirichlet_epsilon_) * child.prior + dirichlet_epsilon_ * d_noise;
      }
    }
  }

  auto root_key = GetStateKey(state);

  for (int sim = 0; sim < max_simulations_; ++sim) {
    std::unique_ptr<State> sampled_root_state = SampleRootState(state);
    SPIEL_CHECK_TRUE(root_key == GetStateKey(*sampled_root_state));
    RunSimulation(sampled_root_state.get(), 0);
  }

  if (verbose_diagnostics_ && (search_count_++ % 50 == 0)) {
    double avg_depth = num_sims_this_search_ > 0 ? (sum_depth_this_search_ / num_sims_this_search_) : 0.0;
    double reuse_rate = total_lookups_ > 0 ? (100.0 * reused_lookups_ / total_lookups_) : 0.0;
    std::cout << "[IS-MCTS Diagnostics] Search " << search_count_
              << " | Player P" << searching_player_
              << " | Tree size: " << nodes_.size()
              << " | Max depth: " << max_depth_this_search_
              << " | Avg depth: " << avg_depth
              << " | Node reuse: " << reuse_rate << "%\n" << std::flush;
  }

  return GetFinalPolicy(state, root_node_);
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

  switch (final_policy_type_) {
    case DuneISMCTSFinalPolicyType::kNormalizedVisitCount: {
      policy.reserve(legal_actions.size());
      if (temperature_ == 0.0) {
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
      } else if (temperature_ == 1.0) {
        for (const auto& av : action_visits) {
          policy.push_back({av.first, av.second / total_legal_visits});
        }
      } else {
        double sum_pow = 0.0;
        std::vector<double> pow_visits(action_visits.size());
        double inv_temp = 1.0 / temperature_;
        for (size_t i = 0; i < action_visits.size(); ++i) {
          pow_visits[i] = std::pow(action_visits[i].second, inv_temp);
          sum_pow += pow_visits[i];
        }
        for (size_t i = 0; i < action_visits.size(); ++i) {
          policy.push_back({action_visits[i].first, sum_pow > 0.0 ? (pow_visits[i] / sum_pow) : (1.0 / legal_actions.size())});
        }
      }
    } break;

    case DuneISMCTSFinalPolicyType::kMaxVisitCount: {
      policy.reserve(legal_actions.size());
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
    } break;

    case DuneISMCTSFinalPolicyType::kMaxValue: {
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
    } break;
  }

  return policy;
}

Action DunePUCTISMCTSBot::Step(const State& state) {
  ActionsAndProbs policy = RunSearch(state);
  return SampleAction(policy, RandomNumber()).first;
}

ActionsAndProbs DunePUCTISMCTSBot::GetPolicy(const State& state) {
  return RunSearch(state);
}

std::pair<ActionsAndProbs, Action> DunePUCTISMCTSBot::StepWithPolicy(const State& state) {
  ActionsAndProbs policy = GetPolicy(state);
  Action sampled_action = SampleAction(policy, RandomNumber()).first;
  return {policy, sampled_action};
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
    ActionsAndProbs raw_priors = evaluator_->Prior(state);
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
