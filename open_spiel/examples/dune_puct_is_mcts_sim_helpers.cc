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
#include "open_spiel/abseil-cpp/absl/strings/str_cat.h"
#include "open_spiel/games/dune_imperium/dune_imperium.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"
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

namespace dune_puct_internal {

uint64_t FnvHash64(const std::string& s) {
  uint64_t h = 1469598103934665603ULL;  // FNV-1a offset basis
  for (char ch : s) {
    h ^= static_cast<uint64_t>(static_cast<unsigned char>(ch));
    h *= 1099511628211ULL;              // FNV-1a prime
  }
  return h;
}

void SnapshotRootVisits(const DuneISMCTSNode* root_node,
                        const std::vector<Action>& legal_actions,
                        std::vector<int>* out) {
  out->clear();
  if (root_node == nullptr) return;
  out->reserve(legal_actions.size());
  for (Action action : legal_actions) {
    auto iter = root_node->child_info.find(action);
    out->push_back(iter != root_node->child_info.end() ? iter->second.visits : 0);
  }
}

}  // namespace dune_puct_internal

uint64_t DeriveNodeKeyHash(int player, const std::string& key_str) {
  return dune_seed::Combine(dune_puct_internal::FnvHash64(key_str),
                            static_cast<uint64_t>(player), /*position=*/0);
}

uint64_t DeriveRootNoiseSeed(uint64_t config_seed, uint64_t search_count,
                             int root_player, const std::string& root_key_str) {
  // Hashing the root state key gives a stable per-root identity so noise varies
  // by root regardless of caller (fresh-per-root sessions, persistent sessions,
  // or Path B bots that reuse one config seed across many roots).
  return dune_seed::DeriveSeed(config_seed, dune_seed::kStreamBlueprint,
                               search_count, static_cast<uint64_t>(root_player),
                               dune_puct_internal::FnvHash64(root_key_str));
}

uint64_t DeriveTieBreakSeed(uint64_t config_seed, uint64_t node_key_hash,
                            int total_visits) {
  return dune_seed::DeriveSeed(config_seed, dune_seed::kStreamMCTS,
                               node_key_hash,
                               static_cast<uint64_t>(total_visits));
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
  if (const auto* dune =
          dynamic_cast<const dune_imperium::DuneImperiumState*>(&state)) {
    absl::StrAppend(&full_key, "|public_card_knowledge=",
                    dune->PublicCardKnowledgeRecordDigest());
  }
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

DuneISMCTSNode* DunePUCTISMCTSBot::CreateNewNodeWithKey(const std::pair<Player, std::string>& key) {
  node_pool_.push_back(std::make_unique<DuneISMCTSNode>());
  DuneISMCTSNode* node = node_pool_.back().get();
  node->key_hash = DeriveNodeKeyHash(key.first, key.second);
  nodes_[key] = node;
  return node;
}

DuneISMCTSNode* DunePUCTISMCTSBot::CreateNewNode(const State& state) {
  return CreateNewNodeWithKey(GetStateKey(state));
}

DuneISMCTSNode* DunePUCTISMCTSBot::LookupNodeByKey(const std::pair<Player, std::string>& key) {
  total_lookups_++;
  auto iter = nodes_.find(key);
  if (iter == nodes_.end()) {
    return nullptr;
  }
  reused_lookups_++;
  return iter->second;
}

DuneISMCTSNode* DunePUCTISMCTSBot::LookupNode(const State& state) {
  return LookupNodeByKey(GetStateKey(state));
}

DuneISMCTSNode* DunePUCTISMCTSBot::LookupOrCreateNode(const State& state) {
  auto key = GetStateKey(state);
  visited_nodes_this_search_.insert(key);
  DuneISMCTSNode* node = LookupNodeByKey(key);
  if (node != nullptr) {
    return node;
  }
  return CreateNewNodeWithKey(key);
}

void DunePUCTISMCTSBot::InitializePriorsAndValue(DuneISMCTSNode* node, const State& state) {
  if (node->priors_initialized) {
    return;
  }
  Player cur_player = state.CurrentPlayer();
  auto t_inf_start = std::chrono::steady_clock::now();
  auto eval_res = evaluators_[cur_player]->PriorAndEvaluate(state);
  auto t_inf_end = std::chrono::steady_clock::now();
  accumulated_inference_time_ms_ += std::chrono::duration<double, std::milli>(t_inf_end - t_inf_start).count();
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
  node->raw_policy = eval_res.first;
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
    double max_log_p = -std::numeric_limits<double>::infinity();
    for (const auto& action_prob : priors) {
      double lp = std::log(std::max(action_prob.second, 1e-12));
      if (lp > max_log_p) max_log_p = lp;
    }
    double sum = 0.0;
    double inv_temp = 1.0 / config_.root_prior_temperature;
    for (const auto& action_prob : priors) {
      double lp = std::log(std::max(action_prob.second, 1e-12));
      double p = std::exp((lp - max_log_p) * inv_temp);
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
        auto t_inf_start = std::chrono::steady_clock::now();
        node->cached_values[p] = evaluators_[p]->Evaluate(state)[p];
        auto t_inf_end = std::chrono::steady_clock::now();
        accumulated_inference_time_ms_ += std::chrono::duration<double, std::milli>(t_inf_end - t_inf_start).count();
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
  if (current_search_sampled_leaf_states_.size() < 16) {
    current_search_sampled_leaf_states_.push_back(
        std::shared_ptr<State>(state.Clone().release()));
  } else {
    // Deterministic reservoir sampling keeps representative leaves without
    // perturbing the search RNG or retaining up to 100,000 cloned states.
    uint64_t sample_seed = dune_seed::Combine(
        config_.seed, dune_seed::kStreamSearchSampling,
        diagnostic_leaf_states_seen_ + 0x4c454146ULL);
    size_t slot = sample_seed % diagnostic_leaf_states_seen_;
    if (slot < current_search_sampled_leaf_states_.size()) {
      current_search_sampled_leaf_states_[slot] =
          std::shared_ptr<State>(state.Clone().release());
    }
  }
}

std::vector<double> DunePUCTISMCTSBot::EvaluateCappedLeaf(
    const State& state, int depth, int decision_depth) {
  Player cur_player = state.CurrentPlayer();
  auto t_inf_start = std::chrono::steady_clock::now();
  std::vector<double> values = evaluators_[cur_player]->Evaluate(state);
  auto t_inf_end = std::chrono::steady_clock::now();
  accumulated_inference_time_ms_ += std::chrono::duration<double, std::milli>(t_inf_end - t_inf_start).count();
  inference_count_this_search_++;
  for (int player = 0; player < state.NumPlayers(); ++player) {
    if (evaluators_[player] != evaluators_[cur_player]) {
      auto t_inf_p_start = std::chrono::steady_clock::now();
      values[player] = evaluators_[player]->Evaluate(state)[player];
      auto t_inf_p_end = std::chrono::steady_clock::now();
      accumulated_inference_time_ms_ += std::chrono::duration<double, std::milli>(t_inf_p_end - t_inf_p_start).count();
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

}  // namespace open_spiel
