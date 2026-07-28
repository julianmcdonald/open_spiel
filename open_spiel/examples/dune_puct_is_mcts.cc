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

namespace {

// FNV-1a over a state key gives a stable identity for a position that is cheap
// to compute and does not depend on iteration order or pointer values.
uint64_t FnvHash64(const std::string& s) {
  uint64_t h = 1469598103934665603ULL;  // FNV-1a offset basis
  for (char ch : s) {
    h ^= static_cast<uint64_t>(static_cast<unsigned char>(ch));
    h *= 1099511628211ULL;              // FNV-1a prime
  }
  return h;
}

}  // namespace

uint64_t DeriveNodeKeyHash(int player, const std::string& key_str) {
  return dune_seed::Combine(FnvHash64(key_str),
                            static_cast<uint64_t>(player), /*position=*/0);
}

uint64_t DeriveRootNoiseSeed(uint64_t config_seed, uint64_t search_count,
                             int root_player, const std::string& root_key_str) {
  // Hashing the root state key gives a stable per-root identity so noise varies
  // by root regardless of caller (fresh-per-root sessions, persistent sessions,
  // or Path B bots that reuse one config seed across many roots).
  return dune_seed::DeriveSeed(config_seed, dune_seed::kStreamBlueprint,
                               search_count, static_cast<uint64_t>(root_player),
                               FnvHash64(root_key_str));
}

uint64_t DeriveTieBreakSeed(uint64_t config_seed, uint64_t node_key_hash,
                            int total_visits) {
  return dune_seed::DeriveSeed(config_seed, dune_seed::kStreamMCTS,
                               node_key_hash,
                               static_cast<uint64_t>(total_visits));
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

std::pair<Player, std::string> DunePUCTISMCTSBot::GetStateKey(const State& state) const {
  std::string obs = config_.use_observation_string ? state.ObservationString() : state.InformationStateString();
  std::vector<Action> actions = state.LegalActions();
  std::sort(actions.begin(), actions.end());
  std::string actions_sig;
  for (Action a : actions) {
    absl::StrAppend(&actions_sig, a, ",");
  }
  std::string full_key = absl::StrCat(
      "obs=", obs,
      "|model=", config_.model_checkpoint_path,
      "|utility=", config_.utility_divisor,
      "|puct_c=", config_.puct_c,
      "|decision_depth_cap=", config_.max_search_decision_depth,
      "|actions=", actions_sig
  );
  return {state.CurrentPlayer(), full_key};
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
  node->key_hash = DeriveNodeKeyHash(key.first, key.second);
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
  DuneISMCTSNode* node = iter->second;
  if (node->priors_initialized) {
    std::vector<Action> state_actions = state.LegalActions();
    for (Action a : state_actions) {
      SPIEL_CHECK_TRUE(node->child_info.count(a) > 0);
    }
    SPIEL_CHECK_EQ(node->child_info.size(), state_actions.size());
  }
  reused_lookups_++;
  return node;
}

DuneISMCTSNode* DunePUCTISMCTSBot::LookupOrCreateNode(const State& state) {
  visited_nodes_this_search_.insert(GetStateKey(state));
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
  RecordSampledLeaf(state);

  // Initialize all legal actions with 0.0 prior first. Fields are assigned by
  // name, not position, so adding a DuneChildInfo member cannot silently
  // reassign these values (the defect class of search finding 2).
  for (Action a : state.LegalActions()) {
    DuneChildInfo& child = node->child_info[a];
    child.visits = 0;
    child.return_sum = 0.0;
    child.prior = 0.0;
    child.raw_prior = 0.0;
  }

  const ActionsAndProbs& priors = eval_res.first;
  // Root-prior temperature reshapes the prior the TREE searches with. The
  // untransformed network prior is kept alongside it in `raw_prior` for every
  // node, because that — not the reshaped tree prior — is what starvation
  // fallbacks play and what the item-4 KL baseline is defined against.
  const bool scale_root_prior =
      node == root_node_ && !is_continuation_ &&
      config_.root_prior_temperature != 1.0 &&
      config_.root_prior_temperature > 0.0;
  std::vector<double> tree_probs;
  tree_probs.reserve(priors.size());
  if (scale_root_prior) {
    double sum = 0.0;
    for (const auto& action_prob : priors) {
      double p = std::pow(std::max(action_prob.second, 1e-12),
                          1.0 / config_.root_prior_temperature);
      tree_probs.push_back(p);
      sum += p;
    }
    for (double& p : tree_probs) {
      p = sum > 0.0 ? (p / sum) : (1.0 / priors.size());
    }
  } else {
    for (const auto& action_prob : priors) {
      tree_probs.push_back(action_prob.second);
    }
  }
  for (size_t i = 0; i < priors.size(); ++i) {
    DuneChildInfo& child = node->child_info[priors[i].first];
    child.visits = 0;
    child.return_sum = 0.0;
    child.prior = tree_probs[i];
    child.raw_prior = priors[i].second;
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

void DunePUCTISMCTSBot::RecordSampledLeaf(const State& state) {
  current_search_leaf_histories_.push_back(state.History());
  ++diagnostic_leaf_states_seen_;
  auto leaf_clone = std::shared_ptr<State>(state.Clone().release());
  if (current_search_sampled_leaf_states_.size() < 16) {
    current_search_sampled_leaf_states_.push_back(std::move(leaf_clone));
  } else {
    // Deterministic reservoir sampling keeps representative leaves without
    // perturbing the search RNG or retaining up to 100,000 cloned states.
    uint64_t sample_seed = dune_seed::Combine(
        config_.seed, dune_seed::kStreamSearchSampling,
        diagnostic_leaf_states_seen_ + 0x4c454146ULL);
    size_t slot = sample_seed % diagnostic_leaf_states_seen_;
    if (slot < current_search_sampled_leaf_states_.size()) {
      current_search_sampled_leaf_states_[slot] = std::move(leaf_clone);
    }
  }
}

std::vector<double> DunePUCTISMCTSBot::EvaluateCappedLeaf(
    const State& state, int depth, int decision_depth) {
  Player cur_player = state.CurrentPlayer();
  std::vector<double> values = evaluators_[cur_player]->Evaluate(state);
  inference_count_this_search_++;
  for (int player = 0; player < state.NumPlayers(); ++player) {
    if (evaluators_[player] != evaluators_[cur_player]) {
      values[player] = evaluators_[player]->Evaluate(state)[player];
      inference_count_this_search_++;
    }
  }
  RecordSampledLeaf(state);
  max_depth_this_search_ = std::max(max_depth_this_search_, depth);
  max_decision_depth_this_search_ =
      std::max(max_decision_depth_this_search_, decision_depth);
  sum_depth_this_search_ += depth;
  sum_decision_depth_this_search_ += decision_depth;
  num_sims_this_search_++;
  simulation_depths_this_search_.push_back(depth);
  const auto* dune_state =
      dynamic_cast<const dune_imperium::DuneImperiumState*>(&state);
  if (dune_state) {
    max_round_this_search_ =
        std::max(max_round_this_search_, dune_state->GetCurrentRound());
  }
  return values;
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

ActionsAndProbs DunePUCTISMCTSBot::FilterAndNormalizeRawPriors(
    DuneISMCTSNode* node, const std::vector<Action>& legal_actions) const {
  ActionsAndProbs result;
  result.reserve(legal_actions.size());
  double sum_prior = 0.0;

  for (Action action : legal_actions) {
    double prior = 1e-5;  // same missing-child fallback as the tree prior
    auto iter = node->child_info.find(action);
    if (iter != node->child_info.end()) {
      prior = iter->second.raw_prior;
    }
    result.push_back({action, prior});
    sum_prior += prior;
  }

  if (sum_prior > 0.0) {
    for (auto& action_prob : result) {
      action_prob.second /= sum_prior;
    }
    return result;
  }
  // No raw prior recorded at all (node never expanded). The tree prior still
  // carries strictly more information than uniform, so degrade to it rather
  // than to the uniform policy this whole path exists to avoid.
  return FilterAndNormalizePriors(node, legal_actions);
}

Action DunePUCTISMCTSBot::SelectActionTreePolicy(
    DuneISMCTSNode* node, const std::vector<Action>& legal_actions) {
  ActionsAndProbs normalized = FilterAndNormalizePriors(node, legal_actions);

  // KataGo root-exploration behaviors apply only at the noised root node.
  const bool at_noised_root = (node == root_node_) && node->dirichlet_noise_applied;

  // Forced playouts (KataGo §3.2): at the noised root, any child under its quota
  // n_forced(c) = sqrt(k * P_noised(c) * N_root) has infinite PUCT urgency. Pick
  // the most-deficient such child so enforcement is deterministic and
  // self-limiting as N_root grows. P_noised(c) is the tree prior (post-noise).
  if (at_noised_root && config_.forced_playouts_k > 0.0) {
    double n_root = static_cast<double>(std::max(0, node->total_visits));
    Action forced_action = kInvalidAction;
    double best_deficit = 0.0;
    for (const auto& action_prob : normalized) {
      auto ci = node->child_info.find(action_prob.first);
      int cv = (ci != node->child_info.end()) ? ci->second.visits : 0;
      if (cv <= 0) continue;  // forcing applies only after a child's first visit
      double quota = std::sqrt(config_.forced_playouts_k * action_prob.second * n_root);
      double deficit = quota - cv;
      if (deficit > best_deficit) {
        best_deficit = deficit;
        forced_action = action_prob.first;
      }
    }
    if (forced_action != kInvalidAction) return forced_action;
  }

  // FPU = 0 at the noised root (KataGo footnote 3); otherwise the node's value.
  double fpu_val = (at_noised_root && config_.root_noise_fpu_zero) ? 0.0
                                                                   : node->cached_value;

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
    // Deterministic tie-breaking RNG derived from stream seed. Keyed on the
    // NODE as well as its local visit count — see DeriveTieBreakSeed.
    uint64_t tie_seed =
        DeriveTieBreakSeed(config_.seed, node->key_hash, node->total_visits);
    std::mt19937 tie_rng(tie_seed);
    return best_actions[absl::Uniform(tie_rng, 0u, best_actions.size())];
  }
}

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
      prior = evaluators_[cur_player]->Prior(*state);
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

Action GetRootArgmaxAction(const DuneISMCTSNode* root_node, const std::vector<Action>& legal_actions) {
  if (root_node == nullptr) return kInvalidAction;
  Action best_action = kInvalidAction;
  int max_visits = -1;
  for (Action action : legal_actions) {
    auto iter = root_node->child_info.find(action);
    if (iter != root_node->child_info.end()) {
      if (iter->second.visits > max_visits) {
        max_visits = iter->second.visits;
        best_action = action;
      }
    }
  }
  return best_action;
}

DuneSearchResult DunePUCTISMCTSBot::RunSearch(const State& state, int max_sims, double max_time_ms, int start_sim_index) {
  root_samples_.clear();
  opponent_prior_cache_.clear();
  inference_count_this_search_ = 0;
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
      }
    }
  }

  int actual_max_sims = (max_sims >= 0) ? max_sims : config_.max_simulations;
  double actual_max_time_ms = (max_time_ms >= 0.0) ? max_time_ms : config_.relative_time_budget_ms;

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
    result.policy = evaluators_[state.CurrentPlayer()]->Prior(state);
    inference_count_this_search_++;
    result.simulations_completed = 0;
    result.elapsed_time_ms = check_elapsed_ms;
    result.timeout_status = true;
    result.used_fallback = true;
    result.fallback_reason = "timeout";
    result.inference_count = inference_count_this_search_;
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
  // PWO-3 telemetry (docs/PWO3_REGISTRATION.md section 4.2). Read-only snapshots
  // at checkpoints the loop already visits: no RNG is drawn and no tree state is
  // mutated, so search behaviour is bitwise unchanged.
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

    auto sim_start = std::chrono::steady_clock::now();
    std::unique_ptr<State> sampled_root_state = SampleRootState(state, start_sim_index + sim);
    std::vector<double> returns =
        RunSimulation(sampled_root_state.get(), 0, 0,
                      start_sim_index + sim);
    auto sim_end = std::chrono::steady_clock::now();

    if (returns.empty()) {
      result.timeout_status = true;
      result.fallback_reason = "timeout";
      break;
    }

    double sim_duration = std::chrono::duration<double, std::milli>(sim_end - sim_start).count();
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

  int req_covered_actions = std::min<int>(3, legal_actions.size());
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

  Action final_proposed_action = GetRootArgmaxAction(root_node_, legal_actions);

  result.diagnostics = GetRootDiagnostics(state, min_visits);
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
