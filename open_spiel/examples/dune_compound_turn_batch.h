// Copyright 2026 DeepMind Technologies Limited.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef OPEN_SPIEL_EXAMPLES_DUNE_COMPOUND_TURN_BATCH_H_
#define OPEN_SPIEL_EXAMPLES_DUNE_COMPOUND_TURN_BATCH_H_

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include "open_spiel/examples/dune_compound_turn_decision.h"
#include "open_spiel/examples/dune_online_search_collector.h"
#include "open_spiel/games/dune_imperium/dune_imperium_util.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace dune_imperium {

// Compute softmax target distribution over evaluated actions at temperature T
inline std::vector<std::pair<Action, double>> ComputeSoftmaxTarget(
    const std::vector<std::pair<Action, double>>& action_utilities,
    const std::vector<Action>& legal_actions,
    double temperature) {
  std::vector<std::pair<Action, double>> probs;
  if (action_utilities.empty() || legal_actions.empty()) return probs;

  double max_u = -1e9;
  for (const auto& au : action_utilities) {
    if (au.second > max_u) max_u = au.second;
  }

  double sum_exp = 0.0;
  std::unordered_map<Action, double> exp_map;
  for (const auto& au : action_utilities) {
    double e = std::exp((au.second - max_u) / temperature);
    exp_map[au.first] = e;
    sum_exp += e;
  }

  probs.reserve(legal_actions.size());
  for (Action a : legal_actions) {
    auto it = exp_map.find(a);
    double p = (it != exp_map.end() && sum_exp > 0.0) ? (it->second / sum_exp) : 0.0;
    probs.push_back({a, p});
  }
  return probs;
}

// Blend blueprint prior mu and search target q_search: (1 - eta) * mu + eta * q_search
inline std::vector<std::pair<Action, double>> BlendTargets(
    const std::vector<std::pair<Action, double>>& blueprint_prior,
    const std::vector<std::pair<Action, double>>& search_target,
    double eta) {
  std::unordered_map<Action, double> q_map;
  for (const auto& st : search_target) q_map[st.first] = st.second;

  std::vector<std::pair<Action, double>> blended;
  blended.reserve(blueprint_prior.size());
  for (const auto& bp : blueprint_prior) {
    Action a = bp.first;
    double mu = bp.second;
    auto it = q_map.find(a);
    double q = (it != q_map.end()) ? it->second : 0.0;
    blended.push_back({a, (1.0 - eta) * mu + eta * q});
  }
  return blended;
}

// Override-gated target: (1 - eta) * prior + eta * onehot(best_candidate)
inline std::vector<double> ComputeOverrideGatedTarget(
    const std::vector<std::pair<Action, double>>& raw_policy,
    const std::vector<Action>& legal_actions,
    Action best_candidate,
    double override_eta) {
  std::vector<double> target(legal_actions.size(), 0.0);
  double prior_sum = 0.0;
  for (size_t i = 0; i < legal_actions.size(); ++i) {
    Action a = legal_actions[i];
    double p = 0.0;
    for (const auto& ap : raw_policy) {
      if (ap.first == a) { p = ap.second; break; }
    }
    target[i] = p;
    prior_sum += p;
  }
  if (prior_sum > 1e-12) {
    for (double& p : target) p /= prior_sum;
  } else {
    std::fill(target.begin(), target.end(), 1.0 / legal_actions.size());
  }

  for (size_t i = 0; i < legal_actions.size(); ++i) {
    Action a = legal_actions[i];
    double onehot = (a == best_candidate) ? 1.0 : 0.0;
    target[i] = (1.0 - override_eta) * target[i] + override_eta * onehot;
  }
  return target;
}

// Check if state_after gained new private information (card drawn or intrigue drawn)
// compared to state_before for player.
inline bool HasNewInformation(
    const DuneImperiumState& state_before,
    const DuneImperiumState& state_after,
    Player player) {
  // 1. Intrigue card drawn
  const auto& int_before = state_before.GetIntrigueHandForTesting(player);
  const auto& int_after = state_after.GetIntrigueHandForTesting(player);
  if (int_after.size() > int_before.size()) return true;

  // 2. Card drawn from deck
  const auto& draw_before = state_before.GetPlayerDrawDeckForTesting(player);
  const auto& draw_after = state_after.GetPlayerDrawDeckForTesting(player);
  if (draw_after.size() != draw_before.size()) return true;

  return false;
}

// Check for high-stakes decision points: Agent Placement, Combat Deployment, Reveal/Tech/Tleilaxu Acquisition
inline bool IsHighStakesStrategicState(const State& state, Player searched_player) {
  if (state.CurrentPlayer() != searched_player) return false;
  std::vector<Action> legal_actions = state.LegalActions();
  if (legal_actions.size() < 2) return false;

  const auto* dune_state = dynamic_cast<const dune_imperium::DuneImperiumState*>(&state);
  if (!dune_state) return false;

  for (Action a : legal_actions) {
    if (IsCardOrTechAcquisitionAction(a)) {
      return true;
    }
    std::string s = dune_state->ActionToString(searched_player, a);
    if (s.rfind("PlaceAgent", 0) == 0 ||
        s.rfind("SelectAgentCard", 0) == 0 ||
        s.rfind("Deploy ", 0) == 0 ||
        s.rfind("CombatCommit", 0) == 0 ||
        s.rfind("Buy", 0) == 0) {
      return true;
    }
  }
  return false;
}

inline bool IsBuyDecisionState(const State& state) {
  const std::vector<Action> legal_actions = state.LegalActions();
  if (legal_actions.size() < 2) return false;
  for (Action a : legal_actions) {
    if (IsCardOrTechAcquisitionAction(a)) {
      return true;
    }
  }
  return false;
}

inline bool IsPurchaseDecisionState(const State& state) {
  return IsBuyDecisionState(state);
}

// Owned immutable snapshot of a pre-action decision visited by the learner
struct SearchSnapshot {
  std::unique_ptr<State> sim_state;
  Player player = kInvalidPlayer;
  std::vector<float> observation;
  std::vector<Action> legal_actions;
  std::vector<std::pair<Action, double>> raw_policy;
  Action chosen_action = kInvalidAction;
  float behavior_log_prob = 0.0f;
  uint64_t root_id = 0;
  // Candidate action descriptors for the deployed semantic action scorer.
  dune_semantic::CandidateActionData candidate_data;
};

// Audited Fallback-Aligned Target Rule:
// q = (1 - eta) * mu + eta * q_search if overridden or if D(q_search) == a_mu;
// q = mu if not overridden and D(q_search) != a_mu.
inline std::vector<double> ComputeFallbackAlignedTarget(
    const std::vector<std::pair<Action, double>>& raw_policy,
    const std::vector<Action>& legal_actions,
    const CompoundSearchDecisionResult& search_res,
    Action raw_action,
    double softmax_temperature,
    double eta_blend,
    bool* fell_back_to_mu) {
  auto q_search_probs = ComputeSoftmaxTarget(search_res.candidate_utilities, legal_actions, softmax_temperature);

  Action search_argmax = raw_action;
  double max_q = -1.0;
  for (const auto& qp : q_search_probs) {
    if (qp.second > max_q) {
      max_q = qp.second;
      search_argmax = qp.first;
    }
  }

  bool fallback = (!search_res.overridden) && (search_argmax != raw_action);
  if (fell_back_to_mu != nullptr) {
    *fell_back_to_mu = fallback;
  }

  std::vector<double> target_distribution(legal_actions.size(), 0.0);
  if (fallback) {
    for (size_t i = 0; i < legal_actions.size(); ++i) {
      Action a = legal_actions[i];
      for (const auto& ap : raw_policy) {
        if (ap.first == a) {
          target_distribution[i] = ap.second;
          break;
        }
      }
    }
  } else {
    auto blended = BlendTargets(raw_policy, q_search_probs, eta_blend);
    for (size_t i = 0; i < legal_actions.size(); ++i) {
      Action a = legal_actions[i];
      for (const auto& ap : blended) {
        if (ap.first == a) {
          target_distribution[i] = ap.second;
          break;
        }
      }
    }
  }
  return target_distribution;
}

// MPO-Style Policy-Relative Reweighting with Temperature Floor and KL Ceiling:
// Outside candidate set C: q(a | s) = pi_ref(a | s).
// Inside candidate set C: q(a | s) = P_C * (pi_ref(a | s) * exp((Q(s, a) - Q_max) / tau)) / sum_{c in C} (pi_ref(c | s) * exp((Q(s, c) - Q_max) / tau)),
// where P_C = sum_{c in C} pi_ref(c | s).
// Base temperature tau_base acts as a floor. If KL(q || pi_ref) <= max_kl, tau_base is kept.
// If KL(q || pi_ref) > max_kl, tau is increased via bisection until KL(q || pi_ref) <= max_kl.
inline std::vector<double> ComputeMpoRelativeTarget(
    const std::vector<std::pair<Action, double>>& raw_policy,
    const std::vector<Action>& legal_actions,
    const std::vector<std::pair<Action, double>>& candidate_utilities,
    double base_temperature = 0.50,
    double max_target_kl = 0.05,
    double* out_realized_kl = nullptr,
    double* out_realized_temperature = nullptr) {
  std::unordered_map<Action, double> pi_map;
  for (const auto& ap : raw_policy) {
    pi_map[ap.first] = std::max(0.0, ap.second);
  }

  std::vector<double> fallback_dist(legal_actions.size(), 0.0);
  for (size_t i = 0; i < legal_actions.size(); ++i) {
    auto it = pi_map.find(legal_actions[i]);
    fallback_dist[i] = (it != pi_map.end()) ? it->second : 0.0;
  }

  if (candidate_utilities.size() <= 1 || legal_actions.empty()) {
    if (out_realized_kl) *out_realized_kl = 0.0;
    if (out_realized_temperature) *out_realized_temperature = base_temperature;
    return fallback_dist;
  }

  std::vector<std::pair<Action, double>> valid_cands;
  double p_c = 0.0;
  double q_max = -1e9;
  for (const auto& cu : candidate_utilities) {
    auto it = pi_map.find(cu.first);
    if (it != pi_map.end() && it->second > 0.0) {
      valid_cands.push_back(cu);
      p_c += it->second;
      if (cu.second > q_max) q_max = cu.second;
    }
  }

  if (valid_cands.size() <= 1 || p_c <= 1e-12) {
    if (out_realized_kl) *out_realized_kl = 0.0;
    if (out_realized_temperature) *out_realized_temperature = base_temperature;
    return fallback_dist;
  }

  auto eval_cand_dist_and_kl = [&](double tau, std::vector<double>& cand_q) -> double {
    double sum_w = 0.0;
    std::vector<double> weights(valid_cands.size(), 0.0);
    for (size_t i = 0; i < valid_cands.size(); ++i) {
      Action a = valid_cands[i].first;
      double pi_a = pi_map[a];
      double u = valid_cands[i].second;
      double w = pi_a * std::exp((u - q_max) / tau);
      weights[i] = w;
      sum_w += w;
    }
    if (sum_w <= 1e-12) {
      for (size_t i = 0; i < valid_cands.size(); ++i) {
        cand_q[i] = pi_map[valid_cands[i].first];
      }
      return 0.0;
    }
    double kl = 0.0;
    for (size_t i = 0; i < valid_cands.size(); ++i) {
      Action a = valid_cands[i].first;
      double pi_a = pi_map[a];
      double q_a = p_c * (weights[i] / sum_w);
      cand_q[i] = q_a;
      if (q_a > 1e-15 && pi_a > 1e-15) {
        kl += q_a * std::log(q_a / pi_a);
      }
    }
    return kl;
  };

  double tau = std::max(1e-4, base_temperature);
  std::vector<double> cand_q(valid_cands.size(), 0.0);
  double current_kl = eval_cand_dist_and_kl(tau, cand_q);

  if (current_kl > max_target_kl) {
    double tau_low = tau;
    double tau_high = 1000.0;
    for (int iter = 0; iter < 25; ++iter) {
      double tau_mid = 0.5 * (tau_low + tau_high);
      double kl_mid = eval_cand_dist_and_kl(tau_mid, cand_q);
      if (kl_mid > max_target_kl) {
        tau_low = tau_mid;
      } else {
        tau_high = tau_mid;
      }
    }
    tau = tau_high;
    current_kl = eval_cand_dist_and_kl(tau, cand_q);
  }

  if (out_realized_kl) *out_realized_kl = current_kl;
  if (out_realized_temperature) *out_realized_temperature = tau;

  std::unordered_map<Action, double> cand_q_map;
  for (size_t i = 0; i < valid_cands.size(); ++i) {
    cand_q_map[valid_cands[i].first] = cand_q[i];
  }

  std::vector<double> target_distribution(legal_actions.size(), 0.0);
  for (size_t i = 0; i < legal_actions.size(); ++i) {
    Action a = legal_actions[i];
    auto it = cand_q_map.find(a);
    if (it != cand_q_map.end()) {
      target_distribution[i] = it->second;
    } else {
      auto pit = pi_map.find(a);
      target_distribution[i] = (pit != pi_map.end()) ? pit->second : 0.0;
    }
  }

  return target_distribution;
}

inline std::vector<double> ComputePairwiseMpoTarget(
    const std::vector<std::pair<Action, double>>& raw_policy,
    const std::vector<Action>& legal_actions,
    Action raw_action, double raw_q,
    Action best_action, double best_q,
    double base_temperature = 0.15,
    double max_target_kl = 0.05,
    double* out_realized_kl = nullptr,
    double* out_realized_temperature = nullptr) {
  std::vector<std::pair<Action, double>> pair_utilities = {
      {raw_action, raw_q},
      {best_action, best_q},
  };
  return ComputeMpoRelativeTarget(
      raw_policy, legal_actions, pair_utilities,
      base_temperature, max_target_kl,
      out_realized_kl, out_realized_temperature);
}

inline std::vector<float> ExtractObservationTensor(
    const State& state,
    Player player,
    dune_imperium::MarketAppendixMode mode = dune_imperium::MarketAppendixMode::kFullPublicInformationV3,
    dune_imperium::DeckPoolSemantics deck_pool_semantics = dune_imperium::DeckPoolSemantics::kLegacy) {
  const auto* dune = dynamic_cast<const dune_imperium::DuneImperiumState*>(&state);
  if (dune) {
    return dune->InformationStateTensorWithAppendix(player, mode, deck_pool_semantics);
  }
  return state.InformationStateTensor(player);
}

struct SearchSupervisionResult {
  std::vector<open_spiel::SearchTrainingExample> examples;
  std::vector<open_spiel::SearchTrainingExample> agreement_examples;
  int roots_evaluated = 0;
  int overrides = 0;
  int fallbacks_to_mu = 0;
  double elapsed_seconds = 0.0;
  double mean_override_rate = 0.0;
  int total_decisions_evaluated = 0;
  double mean_target_prob_search_move = 0.0;
};

inline SearchSupervisionResult BatchCompoundSearchSupervision(
    std::vector<SearchSnapshot>& snapshots,
    const std::vector<std::shared_ptr<BatchedNNEvaluator>>& evaluators,
    int top_k = 3,
    int rollouts_per_action = 64,
    double min_override_margin = 0.15,
    double softmax_temperature = 0.50,
    double eta_blend = 0.80,
    uint64_t search_seed = 0,
    int num_workers = 1,
    bool full_turn_supervision = true,
    bool use_mpo_target = true,
    double max_target_kl = 0.05,
    SearchPairingMode pairing_mode = SearchPairingMode::kUnpaired,
    SearchTruncationMode truncation_mode = SearchTruncationMode::kTerminal,
    const std::string& target_mode = "mpo",
    double override_eta = 0.50) {
  auto start_time = std::chrono::steady_clock::now();
  SearchSupervisionResult result;
  if (snapshots.empty() || evaluators.empty()) return result;

  struct TurnResult {
    bool valid = false;
    int overrides = 0;
    int fallbacks = 0;
    int rollouts = 0;
    int decisions_evaluated = 0;
    double target_prob_search_move_sum = 0.0;
    std::vector<open_spiel::SearchTrainingExample> examples;
    std::vector<open_spiel::SearchTrainingExample> agreement_examples;
  };
  std::vector<TurnResult> turn_results(snapshots.size());

  auto populate_obs_and_candidates = [&](const SearchSnapshot& snap,
                                         bool reuse_snap_obs,
                                         const std::vector<Action>& legals,
                                         open_spiel::SearchTrainingExample* ex) {
    if (reuse_snap_obs) {
      ex->observation = snap.observation;
      ex->candidate_data = snap.candidate_data;
      return;
    }
    const auto active_mode = (!evaluators.empty() && evaluators.front())
                                 ? evaluators.front()->MarketMode()
                                 : dune_imperium::MarketAppendixMode::kFullPublicInformationV3;
    const auto active_semantics = (!evaluators.empty() && evaluators.front())
                                      ? evaluators.front()->DeckPoolSemantics()
                                      : dune_imperium::DeckPoolSemantics::kLegacy;
    ex->observation = ExtractObservationTensor(*snap.sim_state, snap.player, active_mode, active_semantics);
    const auto* dune_s = dynamic_cast<const dune_imperium::DuneImperiumState*>(snap.sim_state.get());
    if (dune_s) {
      const std::string cand_schema = (!evaluators.empty() && evaluators.front())
                                          ? evaluators.front()->SemanticDescriptorSchema()
                                          : dune_semantic::kDescriptorSchemaVersionV3;
      dune_semantic::ExtractCandidateDescriptors(*dune_s, legals, &ex->candidate_data, cand_schema);
    }
  };

  auto record_decision = [&](const SearchSnapshot& snap,
                             const std::vector<Action>& legals,
                             const CompoundSearchDecisionResult& search_res,
                             bool allow_cached_mu,
                             bool reuse_snap_obs,
                             TurnResult* tr) {
    std::vector<std::pair<Action, double>> mu;
    if (allow_cached_mu && !snap.raw_policy.empty()) {
      mu = snap.raw_policy;
    } else if (!evaluators.empty()) {
      ActionsAndProbs raw_prior = evaluators.front()->Prior(*snap.sim_state);
      mu.reserve(legals.size());
      for (Action a : legals) {
        double p = 0.0;
        for (const auto& ap : raw_prior) {
          if (ap.first == a) { p = ap.second; break; }
        }
        mu.push_back({a, p});
      }
    } else {
      mu = snap.raw_policy;
    }
    Action raw_action = PickGreedyAction(mu, legals);

    bool fell_back = false;
    std::vector<double> target_dist;
    bool keep_example = true;
    double target_prob_search_move = 0.0;

    auto find_search_prob = [&](const std::vector<double>& dist) {
      for (size_t i = 0; i < legals.size(); ++i) {
        if (legals[i] == search_res.action) return dist[i];
      }
      return 0.0;
    };

    if (target_mode == "override_gated") {
      if (search_res.overridden) {
        target_dist = ComputeOverrideGatedTarget(mu, legals, search_res.action, override_eta);
        target_prob_search_move = find_search_prob(target_dist);
      } else {
        // Drop non-override states from auxiliary training set; collect as agreement state
        keep_example = false;
        open_spiel::SearchTrainingExample agr_ex;
        populate_obs_and_candidates(snap, reuse_snap_obs, legals, &agr_ex);
        agr_ex.player = snap.player;
        agr_ex.legal_actions = legals;
        tr->agreement_examples.push_back(std::move(agr_ex));
      }
    } else if (use_mpo_target) {
      double r_kl = 0.0;
      double r_tau = softmax_temperature;
      target_dist = ComputeMpoRelativeTarget(
          mu, legals, search_res.candidate_utilities,
          softmax_temperature, max_target_kl, &r_kl, &r_tau);
      target_prob_search_move = find_search_prob(target_dist);
    } else {
      target_dist = ComputeFallbackAlignedTarget(
          mu, legals, search_res, raw_action,
          softmax_temperature, eta_blend, &fell_back);
      target_prob_search_move = find_search_prob(target_dist);
    }

    if (keep_example) {
      open_spiel::SearchTrainingExample ex;
      populate_obs_and_candidates(snap, reuse_snap_obs, legals, &ex);
      ex.player = snap.player;
      ex.legal_actions = legals;
      ex.normalized_visits = std::move(target_dist);
      ex.value_target = 0.0;
      ex.value_target_attached = false;
      ex.simulations_completed = search_res.new_rollouts_executed;
      ex.raw_action = raw_action;
      ex.best_action = search_res.action;
      if (snap.sim_state) {
        ex.state_snapshot = snap.sim_state->Clone();
      }
      tr->target_prob_search_move_sum += target_prob_search_move;
      tr->examples.push_back(std::move(ex));
    }

    tr->decisions_evaluated++;
    tr->rollouts += search_res.new_rollouts_executed;
    if (search_res.overridden) tr->overrides++;
    if (fell_back) tr->fallbacks++;
  };

  auto process_root = [&](size_t r_idx) {
    auto& snap = snapshots[r_idx];
    if (!snap.sim_state) return;

    uint64_t root_seed = dune_seed::DeriveSeed(search_seed, r_idx);
    TurnResult& tr = turn_results[r_idx];
    tr.valid = true;

    if (!full_turn_supervision) {
      // Single-decision supervision at the root state
      auto current_legals = snap.sim_state->LegalActions();
      SPIEL_CHECK_EQ(current_legals.size(), snap.legal_actions.size());
      for (size_t i = 0; i < current_legals.size(); ++i) {
        SPIEL_CHECK_EQ(current_legals[i], snap.legal_actions[i]);
      }

      CompoundSearchDecisionResult search_res = CompoundRolloutSearchDecision(
          *snap.sim_state,
          snap.player,
          /*turn_tree=*/nullptr,
          evaluators,
          top_k,
          rollouts_per_action,
          min_override_margin,
          TreeReuseMode::kTargetFill,
          root_seed,
          pairing_mode,
          truncation_mode);
      record_decision(snap, snap.legal_actions, search_res, !use_mpo_target, true, &tr);
      return;
    }

    // Full-turn supervision: search takes over from turn start to kActionEndTurn
    TurnSearchTree turn_tree;
    int intra_step = 0;
    const int max_turn_steps = 20;

    // Use a persistent per-root RNG across consecutive chance outcomes
    uint64_t chance_root_seed = dune_seed::DeriveSeed(root_seed, /*stream=*/0x4348414EULL);
    std::mt19937_64 chance_rng(chance_root_seed);

    while (!snap.sim_state->IsTerminal() &&
           (snap.sim_state->CurrentPlayer() == snap.player || snap.sim_state->IsChanceNode()) &&
           intra_step < max_turn_steps) {
      if (snap.sim_state->IsChanceNode()) {
        auto chance_outcomes = snap.sim_state->ChanceOutcomes();
        double u = std::generate_canonical<double, 53>(chance_rng);
        Action ca = open_spiel::SampleAction(chance_outcomes, u).first;
        snap.sim_state->ApplyAction(ca);
        // Random chance event (card draw/market refill) invalidates cached search tree
        turn_tree.Reset();
        continue;
      }

      std::vector<Action> current_legals = snap.sim_state->LegalActions();
      if (current_legals.empty()) break;

      if (current_legals.size() == 1) {
        Action forced_a = current_legals.front();
        turn_tree.Advance(forced_a);
        const auto* dune_before = dynamic_cast<const dune_imperium::DuneImperiumState*>(snap.sim_state.get());
        std::unique_ptr<dune_imperium::DuneImperiumState> dune_snap_before =
            dune_before ? std::unique_ptr<dune_imperium::DuneImperiumState>(
                              dynamic_cast<dune_imperium::DuneImperiumState*>(dune_before->Clone().release()))
                        : nullptr;
        snap.sim_state->ApplyAction(forced_a);
        if (dune_snap_before) {
          const auto* dune_after = dynamic_cast<const dune_imperium::DuneImperiumState*>(snap.sim_state.get());
          if (dune_after && HasNewInformation(*dune_snap_before, *dune_after, snap.player)) {
            turn_tree.Reset();
          }
        }
        if (forced_a == dune_imperium::kActionEndTurn) {
          break;
        }
        continue;
      }

      uint64_t step_seed = dune_seed::DeriveSeed(root_seed, intra_step);
      intra_step++;

      CompoundSearchDecisionResult search_res = CompoundRolloutSearchDecision(
          *snap.sim_state,
          snap.player,
          &turn_tree,
          evaluators,
          top_k,
          rollouts_per_action,
          min_override_margin,
          TreeReuseMode::kTargetFill,
          step_seed,
          pairing_mode,
          truncation_mode);

      const bool allow_cached_mu =
          !use_mpo_target && intra_step == 1 && current_legals.size() == snap.legal_actions.size();
      const bool reuse_snap_obs = (intra_step == 1 && !snap.observation.empty());
      record_decision(snap, current_legals, search_res, allow_cached_mu, reuse_snap_obs, &tr);

      Action chosen_action = search_res.action;
      turn_tree.Advance(chosen_action);
      const auto* dune_before = dynamic_cast<const dune_imperium::DuneImperiumState*>(snap.sim_state.get());
      std::unique_ptr<dune_imperium::DuneImperiumState> dune_snap_before =
          dune_before ? std::unique_ptr<dune_imperium::DuneImperiumState>(
                            dynamic_cast<dune_imperium::DuneImperiumState*>(dune_before->Clone().release()))
                      : nullptr;
      snap.sim_state->ApplyAction(chosen_action);
      if (dune_snap_before) {
        const auto* dune_after = dynamic_cast<const dune_imperium::DuneImperiumState*>(snap.sim_state.get());
        if (dune_after && HasNewInformation(*dune_snap_before, *dune_after, snap.player)) {
          turn_tree.Reset();
        }
      }

      if (chosen_action == dune_imperium::kActionEndTurn) {
        break;
      }
    }
  };

  int actual_workers = std::min(num_workers, static_cast<int>(snapshots.size()));
  if (actual_workers <= 1) {
    for (size_t r_idx = 0; r_idx < snapshots.size(); ++r_idx) {
      process_root(r_idx);
    }
  } else {
    std::atomic<size_t> next_root{0};
    std::vector<std::thread> workers;
    workers.reserve(actual_workers);
    for (int w = 0; w < actual_workers; ++w) {
      workers.emplace_back([&]() {
        while (true) {
          size_t r_idx = next_root.fetch_add(1);
          if (r_idx >= snapshots.size()) break;
          process_root(r_idx);
        }
      });
    }
    for (auto& worker : workers) {
      worker.join();
    }
  }

  double total_target_prob_search_move = 0.0;
  for (size_t r_idx = 0; r_idx < snapshots.size(); ++r_idx) {
    if (!turn_results[r_idx].valid) continue;
    result.roots_evaluated++;
    result.overrides += turn_results[r_idx].overrides;
    result.fallbacks_to_mu += turn_results[r_idx].fallbacks;
    result.total_decisions_evaluated += turn_results[r_idx].decisions_evaluated;
    total_target_prob_search_move += turn_results[r_idx].target_prob_search_move_sum;
    for (auto& ex : turn_results[r_idx].examples) {
      result.examples.push_back(std::move(ex));
    }
    for (auto& ex : turn_results[r_idx].agreement_examples) {
      result.agreement_examples.push_back(std::move(ex));
    }
  }

  auto end_time = std::chrono::steady_clock::now();
  result.elapsed_seconds = std::chrono::duration<double>(end_time - start_time).count();
  if (result.total_decisions_evaluated > 0) {
    result.mean_override_rate = static_cast<double>(result.overrides) / result.total_decisions_evaluated;
  }
  if (!result.examples.empty()) {
    result.mean_target_prob_search_move = total_target_prob_search_move / result.examples.size();
  }
  return result;
}

}  // namespace dune_imperium
}  // namespace open_spiel

#endif  // OPEN_SPIEL_EXAMPLES_DUNE_COMPOUND_TURN_BATCH_H_
