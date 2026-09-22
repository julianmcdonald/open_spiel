// Factorial Continuation Diagnostic for Search Policy Iteration (Search-PI).
//
// Evaluates candidate actions vs teacher recommendations vs baseline actions
// across 4 continuation conditions with paired sampled worlds:
// Condition 1 (Q^mu): Greedy U22263 baseline continuation without search.
// Condition 2 (Plan): Compound-plan completion using cached suffix, then U22263.
// Condition 3 (Q^T) : Production compound search on subsequent focal decisions.
// Condition 4 (Q^nu): Greedy Candidate nu continuation without search.
//
// All opponents in all 4 conditions play greedy U22263 without search.
// Roots are stratified across 12 strata (4 decision roles x 3 state origins)
// with balanced deal blocks and independent evaluation randomness.

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <future>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mutex>
#include <numeric>
#include <random>
#include <shared_mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"
#include "open_spiel/utils/json.h"
#include "open_spiel/games/dune_imperium/dune_imperium.h"
#include "open_spiel/games/dune_imperium/dune_imperium_common.h"
#include "absl/flags/flag.h"
#include "absl/flags/parse.h"

#include <torch/torch.h>

#include "dune_network.h"
#include "dune_evaluator.h"
#include "dune_batched_evaluator.h"
#include "dune_semantic_action_scorer.h"
#include "dune_eval_action_selection.h"
#include "dune_seed_utils.h"
#include "dune_sha256.h"
#include "dune_puct_is_mcts.h"
#include "dune_compound_turn_search.h"

using namespace open_spiel;
using namespace open_spiel::dune_imperium;

// Link satisfaction flags
ABSL_FLAG(int, ppo_minibatch_size, 2048, "");
ABSL_FLAG(int, ppo_update_epochs, 4, "");
ABSL_FLAG(double, ppo_clip_epsilon, 0.2, "");
ABSL_FLAG(bool, normalize_advantages, true, "");
ABSL_FLAG(bool, ppo_clip_value_loss, true, "");
ABSL_FLAG(double, entropy_coef, 0.01, "");
ABSL_FLAG(double, value_coef, 0.5, "");
ABSL_FLAG(double, target_kl, 0.0, "");
ABSL_FLAG(bool, train_amp, true, "");
ABSL_FLAG(double, grad_clip_norm, 0.5, "");
ABSL_FLAG(double, logit_cap, 10.0, "");
ABSL_FLAG(uint64_t, shaping_start_env_steps, 206830543, "");
ABSL_FLAG(uint64_t, shaping_decay_env_steps, 0, "");
ABSL_FLAG(bool, diagnostics_only, false, "");

// Diagnostic Flags
ABSL_FLAG(std::string, mode, "run_diagnostic",
          "Mode: 'generate_roots' to sample & freeze 32 roots, or 'run_diagnostic' to execute.");
ABSL_FLAG(std::string, model_checkpoint,
          "/run/media/warcr/Storage/dune_drl_runtime/round7/u21328_continuation_20260921_004103/checkpoints/ppo_model_update_22263.pt",
          "Path to Champion model checkpoint (U22263).");
ABSL_FLAG(std::string, candidate_checkpoint,
          "/home/warcr/dune_drl_runtime/round7/search_pi_stage3_gen1/search_pi_candidate_gen1.pt",
          "Path to Candidate model checkpoint (search_pi_candidate_gen1.pt).");
ABSL_FLAG(std::string, roots_manifest,
          "/home/warcr/projects/dune_drl/docs/experiment_records/diagnostic_32_roots_manifest.json",
          "Path to read/write the 32 stratified diagnostic roots manifest JSON.");
ABSL_FLAG(std::string, output_json,
          "/home/warcr/projects/dune_drl/docs/experiment_records/factorial_continuation_diagnostic_receipt.json",
          "Path to write structured diagnostic results receipt JSON.");
ABSL_FLAG(int, num_paired_worlds, 32, "Outer evaluation budget (sampled worlds per action/condition).");
ABSL_FLAG(int, threads, 14, "Number of concurrent worker threads.");
ABSL_FLAG(int, num_evaluators, 6, "Number of parallel batched evaluator coordinators per network.");
ABSL_FLAG(int, target_batch_size, 128, "GPU evaluator target batch size.");
ABSL_FLAG(int, top_k_actions, 3, "Number of top candidate actions evaluated in teacher search.");
ABSL_FLAG(int, rollouts_per_action, 64, "Number of Monte Carlo rollouts per candidate action in teacher search.");
ABSL_FLAG(double, min_override_margin, 0.15, "Minimum mean utility advantage for teacher override.");
ABSL_FLAG(uint64_t, master_seed, 20260921, "Master random seed for independent evaluation streams.");
ABSL_FLAG(double, softmax_temperature, 0.5, "Temperature for search target distribution.");
ABSL_FLAG(double, eta_blend, 0.80, "Blend parameter for search target: (1-eta)*mu + eta*q_search.");

// Check for high-stakes decision points
inline const char* GetGamePhaseString(GamePhase phase) {
  switch (phase) {
    case GamePhase::kLeaderOfferChance: return "leader_offer_chance";
    case GamePhase::kLeaderDraft: return "leader_draft";
    case GamePhase::kDeal: return "deal";
    case GamePhase::kRoundStart: return "round_start";
    case GamePhase::kAgentTurns: return "agent_turns";
    case GamePhase::kRevealTurns: return "reveal_turns";
    case GamePhase::kCombat: return "combat";
    case GamePhase::kMakers: return "makers";
    case GamePhase::kRecall: return "recall";
    case GamePhase::kTerminal: return "terminal";
    default: return "unknown";
  }
}

inline bool IsStrategicDecision(const State& state, Player p) {
  if (state.CurrentPlayer() != p) return false;
  std::vector<Action> legal_actions = state.LegalActions();
  if (legal_actions.size() < 2) return false;

  const auto* dune_state = dynamic_cast<const dune_imperium::DuneImperiumState*>(&state);
  if (!dune_state) return false;

  for (Action a : legal_actions) {
    std::string s = dune_state->ActionToString(p, a);
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

// Canonical argmax using legal actions array ordering for deterministic tie-breaking
inline Action PickCanonicalArgmax(
    const std::vector<std::pair<Action, double>>& dist,
    const std::vector<Action>& legal_actions) {
  if (legal_actions.empty()) return kInvalidAction;
  double best_p = -1e9;
  Action best_a = legal_actions.front();
  for (Action a : legal_actions) {
    for (const auto& ap : dist) {
      if (ap.first == a) {
        if (ap.second > best_p) {
          best_p = ap.second;
          best_a = a;
        }
        break;
      }
    }
  }
  return best_a;
}

// Root definition representing a frozen decision state
struct DiagnosticRoot {
  int root_id = 0;
  std::string stratum;         // e.g. "teacher_override", "u22263_sub_threshold"
  std::string state_origin;    // "teacher", "u22263", "candidate"
  std::string decision_role;   // "override", "sub_threshold", "agreed", "continuation"
  int focal_player = 0;
  int round = 1;
  std::string phase = "UNKNOWN";
  std::vector<Action> history;
  std::vector<Action> legal_actions;
  std::vector<Action> cached_suffix;
  Action a_mu = kInvalidAction;
  Action a_T = kInvalidAction;
  Action a_raw = kInvalidAction;
  Action a_target = kInvalidAction;
  Action a_nu = kInvalidAction;
  int deal_block = 0;
};

// Simulation outcome for a single paired rollout
struct PairedSimulationOutcome {
  int root_id = 0;
  int condition = 0;  // 1: Q^mu, 2: Plan, 3: Q^T, 4: Q^nu
  Action action = kInvalidAction;
  std::string action_name;  // "mu", "T", "raw", "target", "nu"
  int world_id = 0;
  double utility = 0.0;
  int placement = 4;
  double win = 0.0;
  int vp = 0;
  int ending_round = 0;
};

// Evaluator pool for a network
struct EvaluatorPool {
  std::vector<std::shared_ptr<SharedDunePolicyValueNetImpl>> models;
  std::vector<std::unique_ptr<std::shared_mutex>> mutexes;
  std::vector<std::shared_ptr<BatchedEvaluator>> coords;
  std::vector<std::shared_ptr<BatchedNNEvaluator>> evaluators;

  void Init(const std::string& path, int num_evals, int batch_size, torch::Device device) {
    models.resize(num_evals);
    mutexes.resize(num_evals);
    coords.resize(num_evals);
    evaluators.resize(num_evals);

    for (int e = 0; e < num_evals; ++e) {
      models[e] = std::make_shared<SharedDunePolicyValueNetImpl>(
          kFullPublicInformationStateSize, 2048, 2391, 8,
          /*nonlinear_value_head=*/false, /*with_aux_heads=*/false,
          /*head_init_seed=*/0, /*with_semantic_scorer=*/true);
      models[e]->to(device);
      torch::load(models[e], path, device);
      models[e]->eval();

      mutexes[e] = std::make_unique<std::shared_mutex>();
      coords[e] = std::make_shared<BatchedEvaluator>(
          models[e], batch_size, /*timeout_ms=*/1, device, mutexes[e].get(), 10.0f,
          /*device_synchronize=*/false, /*high_priority_stream=*/true,
          /*emit_batch_membership=*/false, /*rollout_amp=*/true, /*allow_tf32=*/true);
      evaluators[e] = std::make_shared<BatchedNNEvaluator>(coords[e], 10.0f);
    }
  }
};

// Map returns directly to placement:
// 2.25 -> 1st (Win), 0.25 -> 2nd, -0.75 -> 3rd, -1.75 -> 4th
inline int ReturnToPlacement(double ret) {
  if (std::abs(ret - 2.25) < 1e-3) return 1;
  if (std::abs(ret - 0.25) < 1e-3) return 2;
  if (std::abs(ret - (-0.75)) < 1e-3) return 3;
  return 4;
}

// Extract greedily followed plan from turn tree under action a
inline std::vector<Action> ExtractPlannedSuffix(TurnSearchTree* turn_tree, Action a) {
  std::vector<Action> suffix;
  if (!turn_tree || !turn_tree->current_node() || !turn_tree->current_node()->children.count(a)) {
    return suffix;
  }
  const auto* node = turn_tree->current_node()->children.at(a).get();
  while (node && !node->children.empty()) {
    Action best_child = kInvalidAction;
    double best_mean_u = -1e9;
    int best_visits = -1;
    for (const auto& kv : node->children) {
      if (kv.second->visits > best_visits ||
          (kv.second->visits == best_visits && kv.second->MeanUtility() > best_mean_u)) {
        best_visits = kv.second->visits;
        best_mean_u = kv.second->MeanUtility();
        best_child = kv.first;
      }
    }
    if (best_child == kInvalidAction) break;
    suffix.push_back(best_child);
    if (best_child == dune_imperium::kActionEndTurn) break;
    node = node->children.at(best_child).get();
  }
  return suffix;
}

// -----------------------------------------------------------------------------
// Step 1: Root Generation & Stratified Sampling
// -----------------------------------------------------------------------------
void GenerateAndFreezeRoots(
    const std::shared_ptr<const Game>& game,
    EvaluatorPool& pool_mu,
    EvaluatorPool& pool_nu,
    const std::string& manifest_path,
    uint64_t master_seed,
    int top_k,
    int rollouts_per_action,
    double min_override_margin,
    double softmax_temp,
    double eta_blend) {
  std::cout << "\n================================================================================\n";
  std::cout << "[ROOT GENERATION] Generating and stratifying 32 diagnostic roots...\n";
  std::cout << "================================================================================\n";

  std::vector<DiagnosticRoot> candidates_pool;
  std::vector<std::string> origins = {"teacher", "u22263", "candidate"};

  struct GameJob {
    std::string origin;
    int seat;
  };
  std::vector<GameJob> jobs;
  for (const auto& origin : origins) {
    for (int seat = 0; seat < 4; ++seat) {
      jobs.push_back({origin, seat});
    }
  }

  std::mutex pool_mutex;
  std::atomic<size_t> next_job{0};
  std::vector<std::thread> workers;
  int num_workers = std::min(static_cast<int>(jobs.size()), 12);

  auto worker = [&](int thread_id) {
    while (true) {
      size_t j_idx = next_job.fetch_add(1);
      if (j_idx >= jobs.size()) break;
      const auto& job = jobs[j_idx];
      const std::string& origin = job.origin;
      int seat = job.seat;

      uint64_t driver_stream = (origin == "teacher") ? 0x7EA : ((origin == "u22263") ? 0x222 : 0xC4D);
      uint64_t gseed = dune_seed::DeriveSeed(master_seed, driver_stream, seat);

      std::unique_ptr<State> state = game->NewInitialState();
      std::mt19937 chance_rng(gseed);
      TurnSearchTree turn_tree;
      Player last_player = kInvalidPlayer;
      uint64_t search_seed_counter = 0;
      std::vector<DiagnosticRoot> local_candidates;

      while (!state->IsTerminal()) {
        if (state->IsChanceNode()) {
          const auto* dune_before = dynamic_cast<const DuneImperiumState*>(state.get());
          int hand_before = dune_before ? dune_before->GetPlayerCardsInHand(seat) : 0;
          int int_before = dune_before ? static_cast<int>(dune_before->GetPlayerIntrigues(seat).size()) : 0;

          auto chance_outcomes = state->ChanceOutcomes();
          double u = std::generate_canonical<double, 53>(chance_rng);
          Action ca = open_spiel::SampleAction(chance_outcomes, u).first;
          state->ApplyAction(ca);

          const auto* dune_after = dynamic_cast<const DuneImperiumState*>(state.get());
          if (dune_after) {
            int hand_after = dune_after->GetPlayerCardsInHand(seat);
            int int_after = static_cast<int>(dune_after->GetPlayerIntrigues(seat).size());
            if (hand_after > hand_before || int_after > int_before) {
              turn_tree.Reset();
            }
          }
          continue;
        }

        Player cur = state->CurrentPlayer();
        if (cur != last_player) {
          turn_tree.Reset();
          last_player = cur;
        }

        std::vector<Action> legals = state->LegalActions();
        if (legals.empty()) break;

        Action action_to_apply = kInvalidAction;

        if (cur == seat) {
          bool is_strategic = IsStrategicDecision(*state, cur);

          // Get baseline prior
          ActionsAndProbs raw_prior_mu = pool_mu.evaluators[thread_id % pool_mu.evaluators.size()]->Prior(*state);
          Action a_mu = PickGreedyAction(raw_prior_mu, legals);

          // Get candidate prior
          ActionsAndProbs raw_prior_nu = pool_nu.evaluators[thread_id % pool_nu.evaluators.size()]->Prior(*state);
          Action a_nu = PickGreedyAction(raw_prior_nu, legals);

          std::vector<std::pair<Action, double>> mu_legal;
          mu_legal.reserve(legals.size());
          for (Action a : legals) {
            double p = 0.0;
            for (const auto& ap : raw_prior_mu) {
              if (ap.first == a) { p = ap.second; break; }
            }
            mu_legal.push_back({a, p});
          }

          if (is_strategic && legals.size() > 1) {
            bool had_continuation = (turn_tree.current_node() != nullptr && !turn_tree.current_node()->children.empty());
            uint64_t dseed = dune_seed::DeriveSeed(gseed, cur, search_seed_counter++);

            auto s_res = CompoundRolloutSearchDecision(
                *state, cur, &turn_tree, pool_mu.evaluators, top_k, rollouts_per_action,
                min_override_margin, TreeReuseMode::kAdditive, dseed);

            Action a_T = s_res.action;
            auto q_search = ComputeSoftmaxTarget(s_res.candidate_utilities, legals, softmax_temp);
            auto q_blend = BlendTargets(mu_legal, q_search, eta_blend);
            Action a_raw = PickCanonicalArgmax(q_search, legals);
            Action a_target = PickCanonicalArgmax(q_blend, legals);

            std::string role;
            if (had_continuation) {
              role = "continuation";
            } else if (s_res.overridden) {
              role = "override";
            } else if (a_raw != a_mu) {
              role = "sub_threshold";
            } else {
              role = "agreed";
            }

            std::vector<Action> plan_suffix = ExtractPlannedSuffix(&turn_tree, a_T);

            const auto* dune_s = dynamic_cast<const DuneImperiumState*>(state.get());
            int r_num = dune_s ? dune_s->GetCurrentRound() : 1;
            std::string p_name = dune_s ? GetGamePhaseString(dune_s->phase()) : "UNKNOWN";

            DiagnosticRoot r;
            r.state_origin = origin;
            r.decision_role = role;
            r.stratum = origin + "_" + role;
            r.focal_player = cur;
            r.round = r_num;
            r.phase = p_name;
            r.history = state->History();
            r.legal_actions = legals;
            r.cached_suffix = plan_suffix;
            r.a_mu = a_mu;
            r.a_T = a_T;
            r.a_raw = a_raw;
            r.a_target = a_target;
            r.a_nu = a_nu;
            r.deal_block = seat;

            local_candidates.push_back(std::move(r));

            // Select live driver action
            if (origin == "teacher") {
              action_to_apply = a_T;
            } else if (origin == "candidate") {
              action_to_apply = a_nu;
            } else {
              action_to_apply = a_mu;
            }

            turn_tree.Advance(action_to_apply);
            if (action_to_apply == dune_imperium::kActionEndTurn) {
              turn_tree.Reset();
            }
          } else {
            // Routine / non-strategic move
            if (origin == "candidate") {
              action_to_apply = a_nu;
            } else {
              action_to_apply = a_mu;
            }
            turn_tree.Reset();
          }
        } else {
          // Opponents play greedy U22263 prior
          ActionsAndProbs opp_prior = pool_mu.evaluators[thread_id % pool_mu.evaluators.size()]->Prior(*state);
          action_to_apply = PickGreedyAction(opp_prior, legals);
        }

        const auto* dune_before = dynamic_cast<const DuneImperiumState*>(state.get());
        int hand_before = (cur == seat && dune_before) ? dune_before->GetPlayerCardsInHand(seat) : 0;
        int int_before = (cur == seat && dune_before) ? static_cast<int>(dune_before->GetPlayerIntrigues(seat).size()) : 0;

        state->ApplyAction(action_to_apply);

        if (cur == seat && action_to_apply != dune_imperium::kActionEndTurn) {
          const auto* dune_after = dynamic_cast<const DuneImperiumState*>(state.get());
          if (dune_after) {
            int hand_after = dune_after->GetPlayerCardsInHand(seat);
            int int_after = static_cast<int>(dune_after->GetPlayerIntrigues(seat).size());
            if (hand_after > hand_before || int_after > int_before) {
              turn_tree.Reset();
            }
          }
        }
      }

      {
        std::lock_guard<std::mutex> lock(pool_mutex);
        candidates_pool.insert(candidates_pool.end(), local_candidates.begin(), local_candidates.end());
        std::cout << "[ROOT GEN] Finished driver=" << origin << " Seat=" << seat
                  << " (collected: " << local_candidates.size() << " candidates)\n" << std::flush;
      }
    }
  };

  for (int t = 0; t < num_workers; ++t) {
    workers.emplace_back(worker, t);
  }
  for (auto& w : workers) {
    w.join();
  }

  std::cout << "[ROOT GEN] Collected " << candidates_pool.size() << " candidate decisions in pool.\n";

  // Save all candidates to disk for archival and inspection
  std::string pool_path = std::filesystem::path(manifest_path).parent_path().string() + "/diagnostic_all_candidates_pool.json";
  json::Object pool_obj;
  pool_obj["total_candidates"] = json::Value(static_cast<int64_t>(candidates_pool.size()));
  json::Array pool_arr;
  for (const auto& r : candidates_pool) {
    json::Object r_obj;
    r_obj["stratum"] = json::Value(r.stratum);
    r_obj["state_origin"] = json::Value(r.state_origin);
    r_obj["decision_role"] = json::Value(r.decision_role);
    r_obj["focal_player"] = json::Value(static_cast<int64_t>(r.focal_player));
    r_obj["round"] = json::Value(static_cast<int64_t>(r.round));
    r_obj["phase"] = json::Value(r.phase);
    r_obj["deal_block"] = json::Value(static_cast<int64_t>(r.deal_block));
    r_obj["a_mu"] = json::Value(static_cast<int64_t>(r.a_mu));
    r_obj["a_T"] = json::Value(static_cast<int64_t>(r.a_T));
    r_obj["a_raw"] = json::Value(static_cast<int64_t>(r.a_raw));
    r_obj["a_target"] = json::Value(static_cast<int64_t>(r.a_target));
    r_obj["a_nu"] = json::Value(static_cast<int64_t>(r.a_nu));
    pool_arr.push_back(json::Value(r_obj));
  }
  pool_obj["candidates"] = json::Value(pool_arr);
  std::ofstream pool_file(pool_path);
  if (pool_file.is_open()) {
    pool_file << json::ToString(pool_obj);
    pool_file.close();
    std::cout << "[ROOT GEN] Saved full candidates pool (" << candidates_pool.size() << " items) to: " << pool_path << "\n";
  }

  // 32 slots balanced across 12 strata and exactly 8 roots per seat (seats 0, 1, 2, 3)
  struct StratumSlot {
    std::string stratum;
    int target_seat;
  };

  std::vector<StratumSlot> slots = {
    // Teacher: 3 override, 3 sub_threshold, 3 agreed, 2 continuation (= 11)
    {"teacher_override", 0}, {"teacher_override", 1}, {"teacher_override", 2},
    {"teacher_sub_threshold", 3}, {"teacher_sub_threshold", 0}, {"teacher_sub_threshold", 1},
    {"teacher_agreed", 2}, {"teacher_agreed", 3}, {"teacher_agreed", 0},
    {"teacher_continuation", 1}, {"teacher_continuation", 2},

    // U22263: 3 override, 3 sub_threshold, 3 agreed, 2 continuation (= 11)
    {"u22263_override", 3}, {"u22263_override", 0}, {"u22263_override", 1},
    {"u22263_sub_threshold", 2}, {"u22263_sub_threshold", 3}, {"u22263_sub_threshold", 0},
    {"u22263_agreed", 1}, {"u22263_agreed", 2}, {"u22263_agreed", 3},
    {"u22263_continuation", 0}, {"u22263_continuation", 3},

    // Candidate: 3 override, 3 sub_threshold, 2 agreed, 2 continuation (= 10)
    {"candidate_override", 1}, {"candidate_override", 2}, {"candidate_override", 3},
    {"candidate_sub_threshold", 0}, {"candidate_sub_threshold", 1}, {"candidate_sub_threshold", 2},
    {"candidate_agreed", 3}, {"candidate_agreed", 0},
    {"candidate_continuation", 1}, {"candidate_continuation", 2}
  };

  std::vector<DiagnosticRoot> selected_32;
  int root_counter = 0;
  std::unordered_set<size_t> used_candidate_indices;

  for (const auto& slot : slots) {
    // Find best match for (slot.stratum, slot.target_seat)
    // Prefer round >= 2, and higher rounds for strategic depth
    int best_idx = -1;
    int best_score = -1;

    for (size_t i = 0; i < candidates_pool.size(); ++i) {
      if (used_candidate_indices.count(i)) continue;
      const auto& c = candidates_pool[i];
      if (c.stratum == slot.stratum && c.focal_player == slot.target_seat) {
        // Scoring: heavily prioritize round >= 2 (score = 100 + round), else round
        int score = (c.round >= 2) ? (100 + c.round) : c.round;
        if (score > best_score) {
          best_score = score;
          best_idx = static_cast<int>(i);
        }
      }
    }

    // Fallback if no matching seat candidate found: pick any matching stratum
    if (best_idx < 0) {
      for (size_t i = 0; i < candidates_pool.size(); ++i) {
        if (used_candidate_indices.count(i)) continue;
        const auto& c = candidates_pool[i];
        if (c.stratum == slot.stratum) {
          int score = (c.round >= 2) ? (100 + c.round) : c.round;
          if (score > best_score) {
            best_score = score;
            best_idx = static_cast<int>(i);
          }
        }
      }
    }

    SPIEL_CHECK_GE(best_idx, 0);
    used_candidate_indices.insert(best_idx);
    DiagnosticRoot sel = candidates_pool[best_idx];
    sel.root_id = root_counter++;
    sel.deal_block = sel.focal_player;
    selected_32.push_back(sel);
  }

  SPIEL_CHECK_EQ(selected_32.size(), 32);

  // Write manifest JSON
  json::Object root_obj;
  root_obj["schema_version"] = json::Value("1.0");
  root_obj["total_roots"] = json::Value(static_cast<int64_t>(32));
  root_obj["master_seed"] = json::Value(static_cast<int64_t>(master_seed));

  json::Array roots_arr;
  for (const auto& r : selected_32) {
    json::Object r_obj;
    r_obj["root_id"] = json::Value(static_cast<int64_t>(r.root_id));
    r_obj["stratum"] = json::Value(r.stratum);
    r_obj["state_origin"] = json::Value(r.state_origin);
    r_obj["decision_role"] = json::Value(r.decision_role);
    r_obj["focal_player"] = json::Value(static_cast<int64_t>(r.focal_player));
    r_obj["round"] = json::Value(static_cast<int64_t>(r.round));
    r_obj["phase"] = json::Value(r.phase);
    r_obj["deal_block"] = json::Value(static_cast<int64_t>(r.deal_block));
    r_obj["a_mu"] = json::Value(static_cast<int64_t>(r.a_mu));
    r_obj["a_T"] = json::Value(static_cast<int64_t>(r.a_T));
    r_obj["a_raw"] = json::Value(static_cast<int64_t>(r.a_raw));
    r_obj["a_target"] = json::Value(static_cast<int64_t>(r.a_target));
    r_obj["a_nu"] = json::Value(static_cast<int64_t>(r.a_nu));

    json::Array hist_arr;
    for (Action a : r.history) hist_arr.push_back(json::Value(static_cast<int64_t>(a)));
    r_obj["history"] = json::Value(hist_arr);

    json::Array leg_arr;
    for (Action a : r.legal_actions) leg_arr.push_back(json::Value(static_cast<int64_t>(a)));
    r_obj["legal_actions"] = json::Value(leg_arr);

    json::Array suf_arr;
    for (Action a : r.cached_suffix) suf_arr.push_back(json::Value(static_cast<int64_t>(a)));
    r_obj["cached_suffix"] = json::Value(suf_arr);

    roots_arr.push_back(json::Value(r_obj));
  }
  root_obj["roots"] = json::Value(roots_arr);

  std::filesystem::create_directories(std::filesystem::path(manifest_path).parent_path());
  std::ofstream out_file(manifest_path);
  SPIEL_CHECK_TRUE(out_file.is_open());
  out_file << json::ToString(root_obj);
  out_file.close();

  std::cout << "[ROOT GEN] Successfully froze 32 roots across 12 strata to: " << manifest_path << "\n";
}

// -----------------------------------------------------------------------------
// Step 2: Load Manifest
// -----------------------------------------------------------------------------
std::vector<DiagnosticRoot> LoadRootsManifest(const std::string& manifest_path) {
  std::ifstream f(manifest_path);
  SPIEL_CHECK_TRUE(f.is_open());
  std::stringstream ss;
  ss << f.rdbuf();
  auto root_val = json::FromString(ss.str());
  SPIEL_CHECK_TRUE(root_val.has_value());
  auto root_obj = root_val.value().GetObject();
  auto roots_arr = root_obj["roots"].GetArray();

  std::vector<DiagnosticRoot> roots;
  roots.reserve(roots_arr.size());

  for (const auto& rv : roots_arr) {
    auto ro = rv.GetObject();
    DiagnosticRoot r;
    r.root_id = static_cast<int>(ro["root_id"].GetInt());
    r.stratum = ro["stratum"].GetString();
    r.state_origin = ro["state_origin"].GetString();
    r.decision_role = ro["decision_role"].GetString();
    r.focal_player = static_cast<int>(ro["focal_player"].GetInt());
    r.round = static_cast<int>(ro["round"].GetInt());
    r.phase = ro["phase"].GetString();
    r.deal_block = static_cast<int>(ro["deal_block"].GetInt());
    r.a_mu = static_cast<Action>(ro["a_mu"].GetInt());
    r.a_T = static_cast<Action>(ro["a_T"].GetInt());
    r.a_raw = static_cast<Action>(ro["a_raw"].GetInt());
    r.a_target = static_cast<Action>(ro["a_target"].GetInt());
    r.a_nu = static_cast<Action>(ro["a_nu"].GetInt());

    for (const auto& a : ro["history"].GetArray()) {
      r.history.push_back(static_cast<Action>(a.GetInt()));
    }
    for (const auto& a : ro["legal_actions"].GetArray()) {
      r.legal_actions.push_back(static_cast<Action>(a.GetInt()));
    }
    for (const auto& a : ro["cached_suffix"].GetArray()) {
      r.cached_suffix.push_back(static_cast<Action>(a.GetInt()));
    }
    roots.push_back(std::move(r));
  }
  return roots;
}

// -----------------------------------------------------------------------------
// Step 3: Run Continuation Rollout for a Condition
// -----------------------------------------------------------------------------
PairedSimulationOutcome ExecuteContinuationRollout(
    int condition,
    const State& resampled_world_state,
    Action evaluated_action,
    const std::string& action_name,
    const DiagnosticRoot& root,
    int world_id,
    uint64_t chance_seed,
    EvaluatorPool& pool_mu,
    EvaluatorPool& pool_nu,
    int thread_id,
    int top_k,
    int rollouts_per_action,
    double min_override_margin) {
  std::unique_ptr<State> state = resampled_world_state.Clone();
  std::mt19937 chance_rng(chance_seed);
  Player focal_player = root.focal_player;

  // Apply root evaluated action first
  state->ApplyAction(evaluated_action);

  TurnSearchTree tree;
  bool tree_valid = false;
  size_t suffix_idx = 0;

  if (condition == 2 && !root.cached_suffix.empty()) {
    tree_valid = true;
  }

  uint64_t search_seed_counter = 0;

  while (!state->IsTerminal()) {
    if (state->IsChanceNode()) {
      const auto* dune_before = dynamic_cast<const DuneImperiumState*>(state.get());
      int hand_before = dune_before ? dune_before->GetPlayerCardsInHand(focal_player) : 0;
      int int_before = dune_before ? static_cast<int>(dune_before->GetPlayerIntrigues(focal_player).size()) : 0;

      auto chance_outcomes = state->ChanceOutcomes();
      double u = std::generate_canonical<double, 53>(chance_rng);
      Action ca = open_spiel::SampleAction(chance_outcomes, u).first;
      state->ApplyAction(ca);

      const auto* dune_after = dynamic_cast<const DuneImperiumState*>(state.get());
      if (dune_after) {
        int hand_after = dune_after->GetPlayerCardsInHand(focal_player);
        int int_after = static_cast<int>(dune_after->GetPlayerIntrigues(focal_player).size());
        if (hand_after > hand_before || int_after > int_before) {
          tree.Reset();
          tree_valid = false;
        }
      }
      continue;
    }

    Player cur = state->CurrentPlayer();
    std::vector<Action> legals = state->LegalActions();
    if (legals.empty()) break;

    Action next_action = kInvalidAction;

    if (cur == focal_player) {
      if (condition == 1) {
        // Condition 1: Greedy U22263 prior
        ActionsAndProbs prior = pool_mu.evaluators[thread_id % pool_mu.evaluators.size()]->Prior(*state);
        next_action = PickGreedyAction(prior, legals);
      } else if (condition == 2) {
        // Condition 2: Plan completion using cached suffix
        if (tree_valid && suffix_idx < root.cached_suffix.size()) {
          Action planned = root.cached_suffix[suffix_idx++];
          bool is_legal = (std::find(legals.begin(), legals.end(), planned) != legals.end());
          if (is_legal) {
            next_action = planned;
            if (next_action == dune_imperium::kActionEndTurn) {
              tree_valid = false;
            }
          } else {
            // Divergence: declare fallback to greedy prior
            tree_valid = false;
            ActionsAndProbs prior = pool_mu.evaluators[thread_id % pool_mu.evaluators.size()]->Prior(*state);
            next_action = PickGreedyAction(prior, legals);
          }
        } else {
          // Fallback to greedy prior
          ActionsAndProbs prior = pool_mu.evaluators[thread_id % pool_mu.evaluators.size()]->Prior(*state);
          next_action = PickGreedyAction(prior, legals);
        }
      } else if (condition == 3) {
        // Condition 3: Full production compound search on subsequent focal player decisions
        if (IsStrategicDecision(*state, cur) && legals.size() > 1) {
          uint64_t dseed = dune_seed::DeriveSeed(chance_seed, cur, search_seed_counter++);
          auto s_res = CompoundRolloutSearchDecision(
              *state, cur, &tree, pool_mu.evaluators, top_k, rollouts_per_action,
              min_override_margin, TreeReuseMode::kAdditive, dseed);
          next_action = s_res.action;
          tree.Advance(next_action);
          if (next_action == dune_imperium::kActionEndTurn) tree.Reset();
        } else if (tree.current_node() != nullptr && !tree.current_node()->children.empty()) {
          // Continuation move from turn tree
          double best_u = -1e9;
          ActionsAndProbs raw_prior = pool_mu.evaluators[thread_id % pool_mu.evaluators.size()]->Prior(*state);
          Action best_a = PickGreedyAction(raw_prior, legals);
          for (Action a : legals) {
            if (tree.HasPriorData(a) && tree.GetMeanUtility(a) > best_u) {
              best_u = tree.GetMeanUtility(a);
              best_a = a;
            }
          }
          next_action = best_a;
          tree.Advance(next_action);
          if (next_action == dune_imperium::kActionEndTurn) tree.Reset();
        } else {
          ActionsAndProbs prior = pool_mu.evaluators[thread_id % pool_mu.evaluators.size()]->Prior(*state);
          next_action = PickGreedyAction(prior, legals);
          tree.Reset();
        }
      } else if (condition == 4) {
        // Condition 4: Greedy Candidate nu prior
        ActionsAndProbs prior = pool_nu.evaluators[thread_id % pool_nu.evaluators.size()]->Prior(*state);
        next_action = PickGreedyAction(prior, legals);
      }
    } else {
      // Opponents always play greedy U22263 prior
      ActionsAndProbs opp_prior = pool_mu.evaluators[thread_id % pool_mu.evaluators.size()]->Prior(*state);
      next_action = PickGreedyAction(opp_prior, legals);
    }

    state->ApplyAction(next_action);
  }

  // Calculate terminal metrics
  std::vector<double> returns = state->Returns();
  double u = returns[focal_player];
  int place = ReturnToPlacement(u);
  double win = (place == 1) ? 1.0 : 0.0;

  const auto* dune_s = dynamic_cast<const DuneImperiumState*>(state.get());
  int vp = dune_s ? dune_s->GetPlayerVp(focal_player) : 0;
  int end_r = dune_s ? dune_s->GetCurrentRound() : 10;

  PairedSimulationOutcome out;
  out.root_id = root.root_id;
  out.condition = condition;
  out.action = evaluated_action;
  out.action_name = action_name;
  out.world_id = world_id;
  out.utility = u;
  out.placement = place;
  out.win = win;
  out.vp = vp;
  out.ending_round = end_r;
  return out;
}

// -----------------------------------------------------------------------------
// Step 4: Run Full Factorial Diagnostic Suite
// -----------------------------------------------------------------------------
void RunFactorialDiagnostic(
    const std::shared_ptr<const Game>& game,
    EvaluatorPool& pool_mu,
    EvaluatorPool& pool_nu,
    const std::vector<DiagnosticRoot>& roots,
    int num_paired_worlds,
    int threads_count,
    const std::string& output_json_path,
    uint64_t master_seed,
    int top_k,
    int rollouts_per_action,
    double min_override_margin) {
  std::cout << "\n================================================================================\n";
  std::cout << "[FACTORIAL DIAGNOSTIC] Executing 32 Roots x 4 Conditions x 32 Paired Worlds...\n";
  std::cout << "================================================================================\n";

  auto start_time = std::chrono::steady_clock::now();
  std::vector<PairedSimulationOutcome> all_outcomes;
  std::mutex out_mutex;
  std::atomic<int> completed_roots(0);

  // We process roots in parallel across threads
  std::vector<std::thread> workers;
  std::atomic<size_t> next_root_idx{0};

  auto worker = [&](int thread_id) {
    while (true) {
      size_t r_idx = next_root_idx.fetch_add(1);
      if (r_idx >= roots.size()) break;

      const auto& root = roots[r_idx];

      // Reconstruct root state from history
      std::unique_ptr<State> root_state = game->NewInitialState();
      for (Action a : root.history) {
        root_state->ApplyAction(a);
      }

      SPIEL_CHECK_FALSE(root_state->IsTerminal());
      SPIEL_CHECK_EQ(root_state->CurrentPlayer(), root.focal_player);

      // Collect unique candidate actions
      // map action -> name (e.g. "mu", "T", "raw", "target", "nu")
      std::unordered_map<Action, std::string> distinct_actions;
      distinct_actions[root.a_mu] = "mu";
      if (!distinct_actions.count(root.a_T)) distinct_actions[root.a_T] = "T";
      if (!distinct_actions.count(root.a_raw)) distinct_actions[root.a_raw] = "raw";
      if (!distinct_actions.count(root.a_target)) distinct_actions[root.a_target] = "target";
      if (!distinct_actions.count(root.a_nu)) distinct_actions[root.a_nu] = "nu";

      std::vector<PairedSimulationOutcome> root_results;

      // Sample paired worlds
      const auto* dune_root = dynamic_cast<const DuneImperiumState*>(root_state.get());

      for (int w = 0; w < num_paired_worlds; ++w) {
        uint64_t wseed = dune_seed::DeriveSeed(master_seed, root.root_id, w);
        std::mt19937 wrng(wseed);
        auto rng_func = [&wrng]() {
          return std::generate_canonical<double, 53>(wrng);
        };

        std::unique_ptr<State> world_state = dune_root
            ? dune_root->ResampleFromInfostate(root.focal_player, rng_func)
            : root_state->Clone();

        uint64_t chance_seed = dune_seed::DeriveSeed(wseed, 0xC8A4CE, 0);

        // Evaluate each condition
        for (int c = 1; c <= 4; ++c) {
          // If Condition 2 and plan is empty, mark unavailable / identical to Cond 1
          for (const auto& act_pair : distinct_actions) {
            Action a = act_pair.first;
            const std::string& a_name = act_pair.second;

            auto res = ExecuteContinuationRollout(
                c, *world_state, a, a_name, root, w, chance_seed,
                pool_mu, pool_nu, thread_id, top_k, rollouts_per_action, min_override_margin);

            root_results.push_back(res);
          }
        }
      }

      int done = completed_roots.fetch_add(1) + 1;
      {
        std::lock_guard<std::mutex> lock(out_mutex);
        all_outcomes.insert(all_outcomes.end(), root_results.begin(), root_results.end());
        std::cout << "Root [" << std::setw(2) << done << "/32] ID=" << std::setw(2) << root.root_id
                  << " Stratum=" << std::setw(25) << std::left << root.stratum << std::right
                  << " Seat=" << root.focal_player << " R=" << root.round
                  << " DistinctActions=" << distinct_actions.size()
                  << " (Simulations: " << root_results.size() << ")\n" << std::flush;
      }
    }
  };

  for (int t = 0; t < threads_count; ++t) {
    workers.emplace_back(worker, t);
  }
  for (auto& w : workers) {
    w.join();
  }

  auto end_time = std::chrono::steady_clock::now();
  double total_time = std::chrono::duration<double>(end_time - start_time).count();

  std::cout << "\n================================================================================\n";
  std::cout << "[FACTORIAL DIAGNOSTIC] Completed " << all_outcomes.size()
            << " paired rollouts in " << std::fixed << std::setprecision(1) << total_time << "s ("
            << (total_time / 60.0) << " min)\n";
  std::cout << "================================================================================\n";

  // Build JSON Receipt with raw paired simulations and aggregated contrasts
  json::Object receipt;
  receipt["tournament_type"] = json::Value("factorial_continuation_diagnostic");
  receipt["model_checkpoint"] = json::Value(pool_mu.models.front() ? "U22263" : "");
  receipt["candidate_checkpoint"] = json::Value(pool_nu.models.front() ? "search_pi_candidate_gen1.pt" : "");
  receipt["total_roots"] = json::Value(static_cast<int64_t>(roots.size()));
  receipt["num_paired_worlds"] = json::Value(static_cast<int64_t>(num_paired_worlds));
  receipt["total_simulations"] = json::Value(static_cast<int64_t>(all_outcomes.size()));
  receipt["total_wall_time_seconds"] = json::Value(total_time);

  json::Array raw_arr;
  for (const auto& sim : all_outcomes) {
    json::Object s_obj;
    s_obj["root_id"] = json::Value(static_cast<int64_t>(sim.root_id));
    s_obj["condition"] = json::Value(static_cast<int64_t>(sim.condition));
    s_obj["action"] = json::Value(static_cast<int64_t>(sim.action));
    s_obj["action_name"] = json::Value(sim.action_name);
    s_obj["world_id"] = json::Value(static_cast<int64_t>(sim.world_id));
    s_obj["utility"] = json::Value(sim.utility);
    s_obj["placement"] = json::Value(static_cast<int64_t>(sim.placement));
    s_obj["win"] = json::Value(sim.win);
    s_obj["vp"] = json::Value(static_cast<int64_t>(sim.vp));
    s_obj["ending_round"] = json::Value(static_cast<int64_t>(sim.ending_round));
    raw_arr.push_back(json::Value(s_obj));
  }
  receipt["simulations"] = json::Value(raw_arr);

  std::filesystem::create_directories(std::filesystem::path(output_json_path).parent_path());
  std::ofstream out_f(output_json_path);
  SPIEL_CHECK_TRUE(out_f.is_open());
  out_f << json::ToString(receipt);
  out_f.close();

  std::cout << "[RECEIPT] Factorial diagnostic raw receipts saved to: " << output_json_path << "\n";
}

// -----------------------------------------------------------------------------
// Main Entry Point
// -----------------------------------------------------------------------------
int main(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);

  std::string mode = absl::GetFlag(FLAGS_mode);
  std::string model_path = absl::GetFlag(FLAGS_model_checkpoint);
  std::string cand_path = absl::GetFlag(FLAGS_candidate_checkpoint);
  std::string manifest_path = absl::GetFlag(FLAGS_roots_manifest);
  std::string output_json_path = absl::GetFlag(FLAGS_output_json);
  int num_worlds = absl::GetFlag(FLAGS_num_paired_worlds);
  int threads_count = absl::GetFlag(FLAGS_threads);
  int num_evals = absl::GetFlag(FLAGS_num_evaluators);
  int batch_size = absl::GetFlag(FLAGS_target_batch_size);
  int top_k = absl::GetFlag(FLAGS_top_k_actions);
  int rollouts_per_action = absl::GetFlag(FLAGS_rollouts_per_action);
  double min_override_margin = absl::GetFlag(FLAGS_min_override_margin);
  uint64_t master_seed = absl::GetFlag(FLAGS_master_seed);
  double softmax_temp = absl::GetFlag(FLAGS_softmax_temperature);
  double eta_blend = absl::GetFlag(FLAGS_eta_blend);

  std::cout << "================================================================================\n";
  std::cout << "DUNE: IMPERIUM -- FACTORIAL CONTINUATION DIAGNOSTIC\n";
  std::cout << "================================================================================\n";
  std::cout << "Mode:                  " << mode << "\n";
  std::cout << "Baseline Checkpoint:   " << model_path << "\n";
  std::cout << "Candidate Checkpoint:  " << cand_path << "\n";
  std::cout << "Roots Manifest:        " << manifest_path << "\n";
  std::cout << "Output JSON Receipt:   " << output_json_path << "\n";
  std::cout << "Paired Worlds:         " << num_worlds << " per action/condition\n";
  std::cout << "Inner Search Budget:   Top-" << top_k << ", " << rollouts_per_action << " rollouts/act, margin " << min_override_margin << "\n";
  std::cout << "Hardware Threads:      " << threads_count << " threads, " << num_evals << " evaluators/net\n";
  std::cout << "================================================================================\n";

  torch::Device device(torch::kCPU);
  if (torch::cuda::is_available()) {
    device = torch::Device(torch::kCUDA, 0);
    std::cout << "[CUDA] Initialized on device: " << device << "\n";
  } else {
    std::cout << "[CPU] CUDA not available, falling back to CPU.\n";
  }

  std::shared_ptr<const Game> game = LoadGame("dune_imperium");

  // Initialize Evaluators for U22263 and Candidate nu
  std::cout << "[MODEL] Loading Baseline U22263 evaluators...\n";
  EvaluatorPool pool_mu;
  pool_mu.Init(model_path, num_evals, batch_size, device);

  std::cout << "[MODEL] Loading Candidate Gen 1 evaluators...\n";
  EvaluatorPool pool_nu;
  pool_nu.Init(cand_path, num_evals, batch_size, device);

  if (mode == "generate_roots") {
    GenerateAndFreezeRoots(
        game, pool_mu, pool_nu, manifest_path, master_seed,
        top_k, rollouts_per_action, min_override_margin, softmax_temp, eta_blend);
  } else if (mode == "run_diagnostic") {
    // If manifest does not exist, generate it first
    if (!std::filesystem::exists(manifest_path)) {
      std::cout << "[MANIFEST] Manifest not found, generating roots first...\n";
      GenerateAndFreezeRoots(
          game, pool_mu, pool_nu, manifest_path, master_seed,
          top_k, rollouts_per_action, min_override_margin, softmax_temp, eta_blend);
    }
    auto roots = LoadRootsManifest(manifest_path);
    std::cout << "[MANIFEST] Loaded " << roots.size() << " frozen diagnostic roots from " << manifest_path << "\n";

    RunFactorialDiagnostic(
        game, pool_mu, pool_nu, roots, num_worlds, threads_count,
        output_json_path, master_seed, top_k, rollouts_per_action, min_override_margin);
  } else {
    std::cerr << "[ERROR] Unknown mode: " << mode << ". Use 'generate_roots' or 'run_diagnostic'.\n";
    return 1;
  }

  return 0;
}
