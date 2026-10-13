// Copyright 2026 DeepMind Technologies Limited
//
// Native Actor-Only Search-Supervision Pilot for Dune: Imperium
// Implements:
//   1. Frozen-actor collection (all 4 seats play learner's sampled policy, uniform random legal leader draft).
//   2. 25% random position search excluding leader draft positions, with candidates (top 3 + 1 random remaining).
//   3. High-concurrency BatchedEvaluator with rollout_amp=false and allow_tf32=false (pure FP32).
//   4. Decoupled concurrent rollout pool feeding wide GPU batches (e.g. batch 64/128).
//   5. 32 paired rollouts per candidate with truncated horizon (kNextRoundSearchPlayer) and immutable U31900 critic.
//   6. 32 fresh paired confirmation rollouts with SHARED continuation seeds (cseed), independent from discovery streams.
//   7. Target q = (1 - w)*pi + w*onehot(selected_move), w = clip((d - se)/0.15, 0, 1).
//   8. Broad pool preservation targets from cycle-start policy (including leader draft positions).
//   9. True batched tensor actor learning on GPU: CE(q, learner) + 0.1 * KL(cycle_start_policy || learner).
//  10. Preserved inherited optimizer learning rates (5e-6, 5e-6, 5e-5 across 3 groups).
//  11. EMA tracking with decay 2^(-11/S) and dynamic S estimation with balanced Cycle 2 replay.
//  12. Complete focused checks: pre-search parity, zero preservation KL, direct vs batched critic check,
//      and checkpoint/EMA/optimizer reload verification.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <future>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mutex>
#include <numeric>
#include <queue>
#include <random>
#include <string>
#include <thread>
#include <vector>
#include <sys/resource.h>

#include <torch/torch.h>

#include "open_spiel/abseil-cpp/absl/flags/flag.h"
#include "open_spiel/abseil-cpp/absl/flags/parse.h"
#include "open_spiel/abseil-cpp/absl/strings/str_format.h"
#include "open_spiel/games/dune_imperium/dune_imperium.h"
#include "open_spiel/spiel.h"
#include "open_spiel/utils/json.h"

#include "dune_compound_turn_search.h"
#include "dune_eval_action_selection.h"
#include "dune_network.h"
#include "dune_ppo_training_flags.h"
#include "dune_ppo_training_utils.h"
#include "dune_search_routing.h"
#include "dune_seed_utils.h"
#include "dune_semantic_action_scorer.h"
#include "dune_sha256.h"
#include "dune_split_evaluator.h"

ABSL_FLAG(std::string, actor_checkpoint, "", "Path to starting actor model checkpoint.");
ABSL_FLAG(std::string, actor_optimizer, "", "Path to starting actor optimizer checkpoint.");
ABSL_FLAG(std::string, actor_ema_checkpoint, "", "Path to starting actor EMA checkpoint.");
ABSL_FLAG(std::string, actor_ema_manifest, "", "Path to starting actor EMA manifest.");
ABSL_FLAG(std::string, critic_checkpoint, "", "Path to separate immutable critic checkpoint.");
ABSL_FLAG(std::string, source_manifest, "", "Path to source manifest JSON.");
ABSL_FLAG(std::string, output_dir, "", "Output directory for preflight and pilot artifacts.");
ABSL_FLAG(std::string, device, "cpu", "Device to use: 'cuda' or 'cpu'.");
ABSL_FLAG(int, threads, 16, "Number of concurrent threads for collection.");
ABSL_FLAG(int, rollout_workers, 128, "Number of concurrent rollout worker threads for search.");
ABSL_FLAG(int, search_batch_size, 128, "Target batch size for batched evaluators in search phase.");
ABSL_FLAG(int, evaluator_count, 1, "Number of evaluator instances for rollout evaluation (1 or 2).");
ABSL_FLAG(bool, search_amp, false, "Run search-phase rollout and critic evaluators with BF16 autocast and TF32 (default: pure FP32).");
ABSL_FLAG(int, search_root_window, 0, "Root positions searched concurrently (0 = min(16, max(4, rollout_workers / 8))).");
ABSL_FLAG(bool, benchmark_concurrency, false, "Run concurrency micro-benchmark across configurations.");
ABSL_FLAG(int, games_per_cycle, 8, "Number of games per cycle (8 for preflight).");
ABSL_FLAG(double, search_sample_rate, 0.25, "Fraction of non-draft multi-action decisions to search.");
ABSL_FLAG(int, rollouts_per_candidate, 32, "Paired rollouts per candidate move.");
ABSL_FLAG(int, confirmation_rollouts, 32, "Fresh paired confirmation rollouts.");
ABSL_FLAG(int, epochs, 20, "Passes through search examples.");
ABSL_FLAG(int, batch_size, 1024, "Minibatch size for search and preservation examples.");
ABSL_FLAG(double, learning_rate, 5e-6, "Base learning rate (unused: preserved from checkpoint).");
ABSL_FLAG(uint64_t, seed, 11, "Base random seed.");
ABSL_FLAG(bool, preflight_check, false, "Run only the eight-game timing collection and learning check.");
ABSL_FLAG(int, pilot_cycle, 0, "Pilot cycle (0=preflight, 1=cycle 1, 2=cycle 2).");
ABSL_FLAG(double, planned_s, 0.0, "Planned pilot steps S for decay calculation (0.0 = compute dynamically).");
ABSL_FLAG(std::string, preservation_ref_checkpoint, "", "Path to preservation reference model checkpoint (defaults to actor_checkpoint).");
ABSL_FLAG(std::string, save_replay_path, "", "Path to save collected dataset and search examples for cycle 2 replay.");
ABSL_FLAG(std::string, load_replay_path, "", "Path to load cycle 1 dataset and search examples for cycle 2 replay.");

// Diagnostic flags:
ABSL_FLAG(bool, run_terminal_diagnostic, false, "Run changed-move terminal-continuation diagnostic mode.");
ABSL_FLAG(std::string, diag_candidate_checkpoint, "", "Candidate EMA model for diagnostic.");
ABSL_FLAG(std::string, diag_preceding_checkpoint, "", "Preceding EMA model for diagnostic.");
ABSL_FLAG(std::string, diag_u31900_checkpoint, "", "Starting U31900 EMA model for continuation.");
ABSL_FLAG(std::string, diag_opponent_checkpoint, "", "Opponent model for diagnostic (U31400).");
ABSL_FLAG(int, diag_games, 64, "Number of games for diagnostic.");
ABSL_FLAG(int, diag_continuations, 64, "Number of paired terminal continuations per changed move.");
ABSL_FLAG(int, diag_max_changed, 2000, "Maximum changed positions to evaluate.");
ABSL_FLAG(std::string, diag_output_dir, "", "Output directory for diagnostic results.");


// Link-only flag definitions required by dune_ppo_training_utils.cc
ABSL_FLAG(int, ppo_minibatch_size, 2048, "INERT here (link satisfaction)");
ABSL_FLAG(int, ppo_update_epochs, 4, "INERT here (link satisfaction)");
ABSL_FLAG(double, ppo_clip_epsilon, 0.2, "INERT here (link satisfaction)");
ABSL_FLAG(bool, normalize_advantages, true, "INERT here (link satisfaction)");
ABSL_FLAG(bool, ppo_clip_value_loss, true, "INERT here (link satisfaction)");
ABSL_FLAG(double, entropy_coef, 0.01, "INERT here (link satisfaction)");
ABSL_FLAG(double, value_coef, 0.5, "INERT here (link satisfaction)");
ABSL_FLAG(double, target_kl, 0.0, "INERT here (link satisfaction)");
ABSL_FLAG(bool, train_amp, true, "INERT here (link satisfaction)");
ABSL_FLAG(double, grad_clip_norm, 0.5, "INERT here (link satisfaction)");
ABSL_FLAG(double, logit_cap, 10.0, "INERT here (link satisfaction)");
ABSL_FLAG(uint64_t, shaping_start_env_steps, 206830543, "INERT here (link satisfaction)");
ABSL_FLAG(uint64_t, shaping_decay_env_steps, 0, "INERT here (link satisfaction)");
ABSL_FLAG(bool, diagnostics_only, false, "INERT here (link satisfaction)");

namespace open_spiel {
namespace dune_pilot {

using dune_imperium::DuneImperiumState;
using dune_imperium::GamePhase;

// Stream constants for independent reproducible random derivation
constexpr uint64_t kStreamSearchPosSelect    = 0x00C0;
constexpr uint64_t kStreamSearchCandSelect   = 0x00C1;
constexpr uint64_t kStreamSearchHiddenWorld  = 0x00C2;
constexpr uint64_t kStreamSearchContinuation = 0x00C3;
constexpr uint64_t kStreamConfHiddenWorld    = 0x00C4;
constexpr uint64_t kStreamConfContinuation   = 0x00C5;

// Requirement 5: Store each observation and descriptor exactly ONCE
struct CollectedDecision {
  int cycle_idx = 1;
  int game_id = 0;
  int decision_idx = 0;
  Player player = 0;
  bool is_leader_draft = false;
  std::vector<float> obs;  // Stored once
  dune_semantic::CandidateActionData cand_data;  // Stored once
  std::vector<Action> legal_actions;
  std::vector<double> collecting_policy;  // Normalized softmax probabilities over legal actions under 10.0 cap
  std::vector<Action> action_history;    // Complete game history to reconstruct state
};

// Requirement 5: Search labels reference CollectedDecision by index (zero-copy)
struct SearchExample {
  size_t decision_store_idx = 0;  // Index into all_decisions
  std::vector<double> q;          // Target distribution over legal actions
  Action selected_move = -1;
  Action raw_action = -1;
  double advantage_d = 0.0;
  double se = 0.0;
  double teaching_weight_w = 0.0;
};

// Memory monitoring (RSS in megabytes)
double GetCurrentMemoryUsageMB() {
  struct rusage ru;
  if (getrusage(RUSAGE_SELF, &ru) == 0) {
    return static_cast<double>(ru.ru_maxrss) / 1024.0;
  }
  return 0.0;
}

// Compute policy probabilities via BatchedEvaluator (pure FP32, centered and capped at 10.0)
std::vector<double> ComputePolicyProbabilities(
    const std::shared_ptr<open_spiel::BatchedEvaluator>& actor_batcher,
    const std::vector<float>& obs,
    const dune_semantic::CandidateActionData& cand_data,
    const std::vector<Action>& legal_actions) {
  if (legal_actions.empty()) return {};
  if (legal_actions.size() == 1) return {1.0};

  open_spiel::CompactEvalResult compact = actor_batcher->EvaluateCompact(obs, legal_actions, &cand_data);
  std::vector<double> probs(compact.probabilities.begin(), compact.probabilities.end());
  return probs;
}

// Evaluate leaf value via BatchedEvaluator (pure FP32, value-only path)
double EvaluateCriticLeafValue(
    const std::shared_ptr<open_spiel::BatchedEvaluator>& critic_batcher,
    const State& state,
    Player player) {
  const auto* dune = dynamic_cast<const DuneImperiumState*>(&state);
  std::vector<float> obs(9182, 0.0f);
  if (dune != nullptr) {
    dune->InformationStateTensorWithAppendix(
        player, dune_imperium::MarketAppendixMode::kFullPublicInformationV3, absl::MakeSpan(obs));
  } else {
    obs = state.InformationStateTensor(player);
  }

  open_spiel::EvalResult res = critic_batcher->EvaluateValues(obs);
  return static_cast<double>(res.value);
}

// Direct unbatched evaluation for probe states (exact before/after bitwise comparison)
double EvaluateCriticDirect(
    const std::shared_ptr<SharedDunePolicyValueNetImpl>& critic,
    const State& state,
    Player player,
    torch::Device device) {
  const auto* dune = dynamic_cast<const DuneImperiumState*>(&state);
  std::vector<float> obs(9182, 0.0f);
  if (dune != nullptr) {
    dune->InformationStateTensorWithAppendix(
        player, dune_imperium::MarketAppendixMode::kFullPublicInformationV3, absl::MakeSpan(obs));
  } else {
    obs = state.InformationStateTensor(player);
  }

  torch::NoGradGuard no_grad;
  torch::Tensor obs_t = torch::from_blob(
      obs.data(), {1, static_cast<int64_t>(obs.size())}, torch::kFloat32).to(device);
  auto outputs = critic->forward(obs_t);
  torch::Tensor cpu_val = outputs.values.to(torch::kCPU);
  return static_cast<double>(cpu_val.data_ptr<float>()[0]);
}

// Truncated rollout using collecting actor for rollout policies and immutable critic for leaf values
double SimulateTruncatedRollout(
    State* sim_state,
    Player search_player,
    const std::shared_ptr<open_spiel::BatchedEvaluator>& actor_batcher,
    const std::shared_ptr<open_spiel::BatchedEvaluator>& critic_batcher,
    uint64_t rollout_seed) {
  std::mt19937 rng(rollout_seed);
  int steps = 0;
  const int max_steps = 2500;

  const auto* dune_init = dynamic_cast<const DuneImperiumState*>(sim_state);
  int start_round = dune_init ? dune_init->GetCurrentRound() : 1;

  while (!sim_state->IsTerminal() && steps < max_steps) {
    steps++;
    if (sim_state->IsChanceNode()) {
      auto chance_outcomes = sim_state->ChanceOutcomes();
      double u = std::generate_canonical<double, 53>(rng);
      Action ca = open_spiel::SampleAction(chance_outcomes, u).first;
      sim_state->ApplyAction(ca);
      continue;
    }

    Player cur = sim_state->CurrentPlayer();
    const auto* cur_dune = dynamic_cast<const DuneImperiumState*>(sim_state);
    int cur_round = cur_dune ? cur_dune->GetCurrentRound() : 1;

    // Truncation condition: search player's first decision in subsequent round
    if (cur == search_player && cur_round > start_round) {
      double raw_val = EvaluateCriticLeafValue(critic_batcher, *sim_state, search_player);
      return 4.0 * raw_val;
    }

    std::vector<Action> legals = sim_state->LegalActions();
    if (legals.empty()) break;

    Action chosen = legals.front();
    if (legals.size() > 1) {
      dune_semantic::CandidateActionData cand_data;
      std::vector<float> obs(9182, 0.0f);
      if (cur_dune != nullptr) {
        cur_dune->InformationStateTensorWithAppendix(
            cur, dune_imperium::MarketAppendixMode::kFullPublicInformationV3, absl::MakeSpan(obs));
        dune_semantic::ExtractCandidateDescriptors(
            *cur_dune, legals, &cand_data, dune_semantic::kDescriptorSchemaVersionV3);
      }
      auto probs = ComputePolicyProbabilities(actor_batcher, obs, cand_data, legals);
      // Greedy action during rollout
      double best_p = -1.0;
      for (size_t i = 0; i < legals.size(); ++i) {
        if (probs[i] > best_p) {
          best_p = probs[i];
          chosen = legals[i];
        }
      }
    }

    sim_state->ApplyAction(chosen);
  }

  if (!sim_state->IsTerminal()) {
    double raw_val = EvaluateCriticLeafValue(critic_batcher, *sim_state, search_player);
    return 4.0 * raw_val;
  }

  std::vector<double> ret = sim_state->Returns();
  return (static_cast<size_t>(search_player) < ret.size()) ? ret[search_player] : 0.0;
}

// Generate candidate moves: top 3 by prior + 1 uniformly sampled remaining using deterministic stream
std::vector<Action> GenerateCandidates(
    const std::vector<Action>& legal_actions,
    const std::vector<double>& prior_probs,
    std::mt19937_64& rng) {
  if (legal_actions.size() <= 3) {
    return legal_actions;
  }

  std::vector<std::pair<double, Action>> scored;
  scored.reserve(legal_actions.size());
  for (size_t i = 0; i < legal_actions.size(); ++i) {
    scored.push_back({prior_probs[i], legal_actions[i]});
  }
  std::sort(scored.begin(), scored.end(), [](const auto& a, const auto& b) {
    return a.first > b.first;
  });

  std::vector<Action> cands;
  cands.push_back(scored[0].second);
  cands.push_back(scored[1].second);
  cands.push_back(scored[2].second);

  std::uniform_int_distribution<size_t> dist(3, scored.size() - 1);
  cands.push_back(scored[dist(rng)].second);
  return cands;
}

void UpdateEmaWeights(
    const std::shared_ptr<SharedDunePolicyValueNetImpl>& model,
    std::shared_ptr<SharedDunePolicyValueNetImpl>& ema_model,
    double decay) {
  torch::NoGradGuard no_grad;
  auto model_params = model->named_parameters();
  auto ema_params = ema_model->named_parameters();

  for (const auto& kv : model_params) {
    const std::string& name = kv.key();
    const torch::Tensor& m_tensor = kv.value();
    auto ema_it = ema_params.find(name);
    if (ema_it != nullptr) {
      torch::Tensor& ema_tensor = *ema_it;
      ema_tensor.mul_(decay).add_(m_tensor, 1.0 - decay);
    }
  }
}


// --- Replay Serialization Helpers ---
template <typename T>
void WritePod(std::ofstream& ofs, const T& val) {
  ofs.write(reinterpret_cast<const char*>(&val), sizeof(T));
}

template <typename T>
void ReadPod(std::ifstream& ifs, T& val) {
  ifs.read(reinterpret_cast<char*>(&val), sizeof(T));
}

template <typename T>
void WriteVector(std::ofstream& ofs, const std::vector<T>& vec) {
  uint64_t sz = vec.size();
  WritePod(ofs, sz);
  if (sz > 0) {
    ofs.write(reinterpret_cast<const char*>(vec.data()), sz * sizeof(T));
  }
}

template <typename T>
void ReadVector(std::ifstream& ifs, std::vector<T>& vec) {
  uint64_t sz = 0;
  ReadPod(ifs, sz);
  vec.resize(sz);
  if (sz > 0) {
    ifs.read(reinterpret_cast<char*>(vec.data()), sz * sizeof(T));
  }
}

bool SaveReplayData(
    const std::string& path,
    const std::vector<CollectedDecision>& decisions,
    const std::vector<SearchExample>& search_examples,
    const std::vector<size_t>& train_search,
    const std::vector<size_t>& holdout_search,
    const std::vector<size_t>& train_pres,
    const std::vector<size_t>& holdout_pres) {
  std::ofstream ofs(path, std::ios::binary);
  if (!ofs) {
    std::cerr << "Failed to open " << path << " for writing replay data.\n";
    return false;
  }
  uint32_t magic = 0x44554E45;
  uint32_t version = 1;
  WritePod(ofs, magic);
  WritePod(ofs, version);

  uint64_t num_dec = decisions.size();
  WritePod(ofs, num_dec);
  for (const auto& dec : decisions) {
    WritePod(ofs, dec.cycle_idx);
    WritePod(ofs, dec.game_id);
    WritePod(ofs, dec.decision_idx);
    WritePod(ofs, dec.player);
    uint8_t draft = dec.is_leader_draft ? 1 : 0;
    WritePod(ofs, draft);
    WriteVector(ofs, dec.obs);
    WriteVector(ofs, dec.cand_data.actions);
    WriteVector(ofs, dec.cand_data.features);
    WriteVector(ofs, dec.cand_data.card_ids);
    WriteVector(ofs, dec.cand_data.space_ids);
    WriteVector(ofs, dec.cand_data.supported);
    WriteVector(ofs, dec.cand_data.roles);
    WriteVector(ofs, dec.legal_actions);
    WriteVector(ofs, dec.collecting_policy);
    WriteVector(ofs, dec.action_history);
  }

  uint64_t num_ex = search_examples.size();
  WritePod(ofs, num_ex);
  for (const auto& ex : search_examples) {
    uint64_t store_idx = ex.decision_store_idx;
    WritePod(ofs, store_idx);
    WriteVector(ofs, ex.q);
    WritePod(ofs, ex.selected_move);
    WritePod(ofs, ex.raw_action);
    WritePod(ofs, ex.advantage_d);
    WritePod(ofs, ex.se);
    WritePod(ofs, ex.teaching_weight_w);
  }

  WriteVector(ofs, train_search);
  WriteVector(ofs, holdout_search);
  WriteVector(ofs, train_pres);
  WriteVector(ofs, holdout_pres);

  return ofs.good();
}

bool LoadReplayData(
    const std::string& path,
    std::vector<CollectedDecision>& decisions,
    std::vector<SearchExample>& search_examples,
    std::vector<size_t>& train_search,
    std::vector<size_t>& holdout_search,
    std::vector<size_t>& train_pres,
    std::vector<size_t>& holdout_pres) {
  std::ifstream ifs(path, std::ios::binary);
  if (!ifs) {
    std::cerr << "Failed to open " << path << " for reading replay data.\n";
    return false;
  }
  uint32_t magic = 0;
  uint32_t version = 0;
  ReadPod(ifs, magic);
  ReadPod(ifs, version);
  if (magic != 0x44554E45 || version != 1) {
    std::cerr << "Invalid replay data file format: " << path << "\n";
    return false;
  }

  uint64_t num_dec = 0;
  ReadPod(ifs, num_dec);
  decisions.resize(num_dec);
  for (uint64_t i = 0; i < num_dec; ++i) {
    auto& dec = decisions[i];
    ReadPod(ifs, dec.cycle_idx);
    ReadPod(ifs, dec.game_id);
    ReadPod(ifs, dec.decision_idx);
    ReadPod(ifs, dec.player);
    uint8_t draft = 0;
    ReadPod(ifs, draft);
    dec.is_leader_draft = (draft != 0);
    ReadVector(ifs, dec.obs);
    ReadVector(ifs, dec.cand_data.actions);
    ReadVector(ifs, dec.cand_data.features);
    ReadVector(ifs, dec.cand_data.card_ids);
    ReadVector(ifs, dec.cand_data.space_ids);
    ReadVector(ifs, dec.cand_data.supported);
    ReadVector(ifs, dec.cand_data.roles);
    ReadVector(ifs, dec.legal_actions);
    ReadVector(ifs, dec.collecting_policy);
    ReadVector(ifs, dec.action_history);
  }

  uint64_t num_ex = 0;
  ReadPod(ifs, num_ex);
  search_examples.resize(num_ex);
  for (uint64_t i = 0; i < num_ex; ++i) {
    auto& ex = search_examples[i];
    uint64_t store_idx = 0;
    ReadPod(ifs, store_idx);
    ex.decision_store_idx = store_idx;
    ReadVector(ifs, ex.q);
    ReadPod(ifs, ex.selected_move);
    ReadPod(ifs, ex.raw_action);
    ReadPod(ifs, ex.advantage_d);
    ReadPod(ifs, ex.se);
    ReadPod(ifs, ex.teaching_weight_w);
  }

  ReadVector(ifs, train_search);
  ReadVector(ifs, holdout_search);
  ReadVector(ifs, train_pres);
  ReadVector(ifs, holdout_pres);

  return ifs.good();
}

// Simulate full game from state to terminal outcome using greedy evaluation
double SimulateTerminalRollout(
    const Game* game,
    const std::vector<Action>& history,
    Action first_action,
    Player cand_player,
    const std::shared_ptr<open_spiel::BatchedEvaluator>& u31900_batcher,
    const std::shared_ptr<open_spiel::BatchedEvaluator>& opp_batcher,
    uint64_t seed) {
  std::mt19937 rng(seed);
  std::unique_ptr<State> state = game->NewInitialState();
  for (Action a : history) {
    state->ApplyAction(a);
  }
  state->ApplyAction(first_action);

  const auto* dune = dynamic_cast<const DuneImperiumState*>(state.get());
  int steps = 0;
  const int max_steps = 3000;

  while (!state->IsTerminal() && steps < max_steps) {
    steps++;
    if (state->IsChanceNode()) {
      auto outcomes = state->ChanceOutcomes();
      double u = std::generate_canonical<double, 53>(rng);
      Action ca = open_spiel::SampleAction(outcomes, u).first;
      state->ApplyAction(ca);
      continue;
    }

    Player cur = state->CurrentPlayer();
    std::vector<Action> legals = state->LegalActions();
    if (legals.empty()) break;

    Action chosen = legals[0];
    if (legals.size() > 1) {
      std::vector<float> obs(9182, 0.0f);
      dune_semantic::CandidateActionData cand_data;
      if (dune != nullptr) {
        dune->InformationStateTensorWithAppendix(
            cur, dune_imperium::MarketAppendixMode::kFullPublicInformationV3, absl::MakeSpan(obs));
        dune_semantic::ExtractCandidateDescriptors(
            *dune, legals, &cand_data, dune_semantic::kDescriptorSchemaVersionV3);
      }
      auto batcher = (cur == cand_player) ? u31900_batcher : opp_batcher;
      auto probs = ComputePolicyProbabilities(batcher, obs, cand_data, legals);
      double best_p = -1.0;
      for (size_t k = 0; k < legals.size(); ++k) {
        if (probs[k] > best_p) {
          best_p = probs[k];
          chosen = legals[k];
        }
      }
    }
    state->ApplyAction(chosen);
  }

  std::vector<double> rets = state->Returns();
  if (cand_player >= 0 && cand_player < static_cast<Player>(rets.size())) {
    return rets[cand_player];
  }
  return 0.0;
}

std::shared_ptr<open_spiel::SharedDunePolicyValueNetImpl> LoadActorModel(
    const std::string& path, torch::Device device) {
  auto model = std::make_shared<open_spiel::SharedDunePolicyValueNetImpl>(
      9182, 2048, 2391, 12, false, false, 0, true);
  model->semantic_descriptor_schema_ = open_spiel::dune_semantic::kDescriptorSchemaVersionV3;
  model->market_appendix_mode_ = open_spiel::dune_imperium::MarketAppendixMode::kFullPublicInformationV3;
  open_spiel::LoadModelCheckpointRobust(model, path, device);
  return model;
}

int RunTerminalContinuationDiagnostic(
    const std::string& cand_ckpt,
    const std::string& prec_ckpt,
    const std::string& u31900_ckpt,
    const std::string& opp_ckpt,
    int num_games,
    int num_continuations,
    int max_changed,
    const std::string& output_dir,
    const std::string& dev_str,
    uint64_t base_seed,
    int num_threads) {
  std::cout << "\n=================================================================\n";
  std::cout << "  Changed-Move Terminal-Continuation Diagnostic\n";
  std::cout << "=================================================================\n";
  std::cout << "  Candidate model:            " << cand_ckpt << "\n";
  std::cout << "  Preceding model:            " << prec_ckpt << "\n";
  std::cout << "  U31900 continuation model:  " << u31900_ckpt << "\n";
  std::cout << "  Opponent model:             " << opp_ckpt << "\n";
  std::cout << "  Diagnostic games:           " << num_games << "\n";
  std::cout << "  Continuations per changed:  " << num_continuations << "\n";
  std::cout << "  Output dir:                 " << output_dir << "\n";

  std::filesystem::create_directories(output_dir);

  torch::Device device = (dev_str == "cuda" && torch::cuda::is_available())
      ? torch::Device(torch::kCUDA, 0)
      : torch::Device(torch::kCPU);

  auto game = open_spiel::LoadGame("dune_imperium");
  SPIEL_CHECK_TRUE(game != nullptr);

  auto cand_model = open_spiel::dune_pilot::LoadActorModel(cand_ckpt, device);
  auto prec_model = open_spiel::dune_pilot::LoadActorModel(prec_ckpt, device);
  auto u31900_model = open_spiel::dune_pilot::LoadActorModel(u31900_ckpt, device);
  auto opp_model = open_spiel::dune_pilot::LoadActorModel(opp_ckpt, device);

  cand_model->eval();
  prec_model->eval();
  u31900_model->eval();
  opp_model->eval();

  std::shared_mutex cand_mutex, prec_mutex, u31900_mutex, opp_mutex;
  auto cand_batcher = std::make_shared<open_spiel::BatchedEvaluator>(
      cand_model, 64, 1, device, &cand_mutex, 10.0f, false, true, false, false, false);
  auto prec_batcher = std::make_shared<open_spiel::BatchedEvaluator>(
      prec_model, 64, 1, device, &prec_mutex, 10.0f, false, true, false, false, false);
  auto u31900_batcher = std::make_shared<open_spiel::BatchedEvaluator>(
      u31900_model, 64, 1, device, &u31900_mutex, 10.0f, false, true, false, false, false);
  auto opp_batcher = std::make_shared<open_spiel::BatchedEvaluator>(
      opp_model, 64, 1, device, &opp_mutex, 10.0f, false, true, false, false, false);

  struct ChangedPosition {
    int game_id;
    int decision_idx;
    Player candidate_player;
    std::vector<Action> action_history;
    Action action_cand;
    Action action_prec;
    double diff_mean = 0.0;
    double sampling_weight = 1.0;
  };

  std::vector<ChangedPosition> all_changed_positions;
  std::vector<int> total_decisions_per_game(num_games, 0);
  std::vector<int> total_changed_per_game(num_games, 0);
  std::vector<int> eval_changed_per_game(num_games, 0);
  int total_decisions = 0;
  int total_changed_positions = 0;

  for (int g = 0; g < num_games; ++g) {
    Player cand_player = g % 4;
    uint64_t game_seed = dune_seed::DeriveSeed(base_seed, 20261012, 8888, g);
    std::mt19937 chance_rng(game_seed);

    std::unique_ptr<State> state = game->NewInitialState();
    auto* dune_state = dynamic_cast<DuneImperiumState*>(state.get());

    int dec_idx = 0;
    while (!state->IsTerminal()) {
      if (state->IsChanceNode()) {
        auto outcomes = state->ChanceOutcomes();
        double u = std::generate_canonical<double, 53>(chance_rng);
        Action ca = open_spiel::SampleAction(outcomes, u).first;
        state->ApplyAction(ca);
        continue;
      }

      Player cur = state->CurrentPlayer();
      std::vector<Action> legals = state->LegalActions();
      if (legals.empty()) break;

      if (cur == cand_player) {
        total_decisions++;
        total_decisions_per_game[g]++;
        dec_idx++;

        Action a_cand = legals[0];
        Action a_prec = legals[0];

        if (legals.size() > 1) {
          std::vector<float> obs(9182, 0.0f);
          dune_semantic::CandidateActionData cand_data;
          if (dune_state != nullptr) {
            dune_state->InformationStateTensorWithAppendix(
                cur, dune_imperium::MarketAppendixMode::kFullPublicInformationV3, absl::MakeSpan(obs));
            dune_semantic::ExtractCandidateDescriptors(
                *dune_state, legals, &cand_data, dune_semantic::kDescriptorSchemaVersionV3);
          }
          auto p_cand = ComputePolicyProbabilities(cand_batcher, obs, cand_data, legals);
          auto p_prec = ComputePolicyProbabilities(prec_batcher, obs, cand_data, legals);

          double best_cand = -1.0;
          double best_prec = -1.0;
          for (size_t i = 0; i < legals.size(); ++i) {
            if (p_cand[i] > best_cand) { best_cand = p_cand[i]; a_cand = legals[i]; }
            if (p_prec[i] > best_prec) { best_prec = p_prec[i]; a_prec = legals[i]; }
          }
        }

        if (a_cand != a_prec) {
          total_changed_positions++;
          total_changed_per_game[g]++;
          ChangedPosition cp;
          cp.game_id = g;
          cp.decision_idx = dec_idx;
          cp.candidate_player = cand_player;
          cp.action_history = state->History();
          cp.action_cand = a_cand;
          cp.action_prec = a_prec;
          all_changed_positions.push_back(std::move(cp));
        }

        state->ApplyAction(a_cand);
      } else {
        Action a_opp = legals[0];
        if (legals.size() > 1) {
          std::vector<float> obs(9182, 0.0f);
          dune_semantic::CandidateActionData cand_data;
          if (dune_state != nullptr) {
            dune_state->InformationStateTensorWithAppendix(
                cur, dune_imperium::MarketAppendixMode::kFullPublicInformationV3, absl::MakeSpan(obs));
            dune_semantic::ExtractCandidateDescriptors(
                *dune_state, legals, &cand_data, dune_semantic::kDescriptorSchemaVersionV3);
          }
          auto p_opp = ComputePolicyProbabilities(opp_batcher, obs, cand_data, legals);
          double best_opp = -1.0;
          for (size_t i = 0; i < legals.size(); ++i) {
            if (p_opp[i] > best_opp) { best_opp = p_opp[i]; a_opp = legals[i]; }
          }
        }
        state->ApplyAction(a_opp);
      }
    }
  }

  std::cout << absl::StrFormat("[DIAG] Evaluated %d candidate decisions across %d games.\n",
                               total_decisions, num_games);
  std::cout << absl::StrFormat("[DIAG] Found %zu changed positions across entire bank (%.2f%%).\n",
                               all_changed_positions.size(),
                               (total_decisions > 0) ? (100.0 * all_changed_positions.size() / total_decisions) : 0.0);

  // Sample across the entire diagnostic bank if total changed positions exceed max_changed
  std::vector<ChangedPosition> changed_positions;
  if (static_cast<int>(all_changed_positions.size()) <= max_changed) {
    changed_positions = std::move(all_changed_positions);
    for (auto& cp : changed_positions) {
      cp.sampling_weight = 1.0;
      eval_changed_per_game[cp.game_id]++;
    }
  } else {
    size_t bank_size = all_changed_positions.size();
    double sampling_weight = static_cast<double>(bank_size) / max_changed;
    std::cout << absl::StrFormat(
        "[DIAG] Diagnostic bank size (%zu) exceeds cap (%d). Sampling %d positions across entire bank (sampling weight = %.4f).\n",
        bank_size, max_changed, max_changed, sampling_weight);

    uint64_t sample_seed = dune_seed::DeriveSeed(base_seed, 20261012, 9999);
    std::mt19937 sample_rng(sample_seed);

    std::vector<size_t> indices(bank_size);
    std::iota(indices.begin(), indices.end(), 0);
    std::shuffle(indices.begin(), indices.end(), sample_rng);
    indices.resize(max_changed);
    std::sort(indices.begin(), indices.end());

    changed_positions.reserve(max_changed);
    for (size_t idx : indices) {
      auto cp = std::move(all_changed_positions[idx]);
      cp.sampling_weight = sampling_weight;
      eval_changed_per_game[cp.game_id]++;
      changed_positions.push_back(std::move(cp));
    }
  }

  // Paired continuations parallelized across worker threads
  int num_diag_workers = std::max(1, num_threads);
  std::atomic<size_t> next_cp_idx{0};
  std::vector<std::thread> workers;
  for (int th = 0; th < num_diag_workers; ++th) {
    workers.emplace_back([&]() {
      while (true) {
        size_t idx = next_cp_idx.fetch_add(1);
        if (idx >= changed_positions.size()) break;
        auto& cp = changed_positions[idx];
        double sum_diff = 0.0;
        for (int r = 0; r < num_continuations; ++r) {
          uint64_t cont_seed = dune_seed::DeriveSeed(base_seed, 20261012, 7777, idx, r);
          double u_cand = SimulateTerminalRollout(
              game.get(), cp.action_history, cp.action_cand, cp.candidate_player,
              u31900_batcher, opp_batcher, cont_seed);
          double u_prec = SimulateTerminalRollout(
              game.get(), cp.action_history, cp.action_prec, cp.candidate_player,
              u31900_batcher, opp_batcher, cont_seed);
          sum_diff += (u_cand - u_prec);
        }
        cp.diff_mean = (num_continuations > 0) ? (sum_diff / num_continuations) : 0.0;
      }
    });
  }
  for (auto& w : workers) {
    if (w.joinable()) w.join();
  }

  double sum_weighted_diff = 0.0;
  for (const auto& cp : changed_positions) {
    sum_weighted_diff += cp.diff_mean * cp.sampling_weight;
  }
  double conditional_gain = (total_changed_positions > 0)
      ? (sum_weighted_diff / total_changed_positions)
      : 0.0;

  // Game gains aggregated with inverse-probability sampling weights across all games
  std::vector<double> game_gains(num_games, 0.0);
  for (const auto& cp : changed_positions) {
    game_gains[cp.game_id] += cp.diff_mean * cp.sampling_weight;
  }

  double total_attributed_diff = 0.0;
  for (double gg : game_gains) total_attributed_diff += gg;
  double freq_weighted_gain = (total_decisions > 0) ? (total_attributed_diff / total_decisions) : 0.0;

  // Clustered ratio-estimator standard error:
  // Cluster size M_g = total_decisions_per_game[g]
  // Cluster total Y_g = game_gains[g]
  // Residual e_g = Y_g - freq_weighted_gain * M_g
  double sum_sq_residuals = 0.0;
  for (int g = 0; g < num_games; ++g) {
    double e_g = game_gains[g] - freq_weighted_gain * total_decisions_per_game[g];
    sum_sq_residuals += e_g * e_g;
  }
  double se_game_grouped = 0.0;
  if (num_games > 1 && total_decisions > 0) {
    double var = (static_cast<double>(num_games) / (num_games - 1)) * sum_sq_residuals /
                 (static_cast<double>(total_decisions) * total_decisions);
    se_game_grouped = std::sqrt(var);
  }
  double ci_95_lcb = freq_weighted_gain - 1.96 * se_game_grouped;
  double ci_95_ucb = freq_weighted_gain + 1.96 * se_game_grouped;

  std::cout << absl::StrFormat("[DIAG] Frequency-weighted gain: %+.6f | SE: %.6f | 95%% CI: [%+.6f, %+.6f]\n",
                               freq_weighted_gain, se_game_grouped, ci_95_lcb, ci_95_ucb);
  std::cout << absl::StrFormat("[DIAG] Conditional gain on changed moves: %+.6f\n", conditional_gain);
  std::cout << absl::StrFormat("[DIAG] Positive 95%% Interval: %s\n", (ci_95_lcb > 0.0) ? "YES" : "NO");

  open_spiel::json::Object diag_json;
  diag_json["diagnostic_games"] = num_games;
  diag_json["total_decisions"] = total_decisions;
  diag_json["total_changed_positions"] = static_cast<int64_t>(total_changed_positions);
  diag_json["sampled_changed_positions"] = static_cast<int64_t>(changed_positions.size());
  diag_json["changed_positions"] = static_cast<int64_t>(changed_positions.size());
  diag_json["sampling_weight"] = (!changed_positions.empty()) ? changed_positions[0].sampling_weight : 1.0;
  diag_json["changed_move_fraction"] = (total_decisions > 0) ? (static_cast<double>(total_changed_positions) / total_decisions) : 0.0;
  diag_json["continuations_per_changed"] = num_continuations;
  diag_json["frequency_weighted_gain"] = freq_weighted_gain;
  diag_json["se_game_grouped"] = se_game_grouped;
  open_spiel::json::Array ci_arr;
  ci_arr.push_back(ci_95_lcb);
  ci_arr.push_back(ci_95_ucb);
  diag_json["ci_95"] = ci_arr;
  diag_json["is_positive_95"] = (ci_95_lcb > 0.0);
  diag_json["conditional_gain_changed_moves"] = conditional_gain;

  std::string out_file = output_dir + "/terminal_continuation_diagnostic.json";
  std::ofstream ofs(out_file);
  ofs << open_spiel::json::ToString(diag_json, true);
  std::cout << "[DIAG] Saved diagnostic record to " << out_file << "\n";

  return 0;
}

// Requirement 2: High-throughput concurrent rollout worker pool
class ConcurrentRolloutPool {
 public:
  explicit ConcurrentRolloutPool(int num_workers) : stop_(false) {
    for (int i = 0; i < num_workers; ++i) {
      workers_.emplace_back([this]() {
        while (true) {
          std::function<void()> task;
          {
            std::unique_lock<std::mutex> lock(mutex_);
            cv_.wait(lock, [this]() { return stop_ || !tasks_.empty(); });
            if (stop_ && tasks_.empty()) return;
            task = std::move(tasks_.front());
            tasks_.pop();
          }
          task();
        }
      });
    }
  }

  void Enqueue(std::function<void()> task) {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      tasks_.push(std::move(task));
    }
    cv_.notify_one();
  }

  ~ConcurrentRolloutPool() {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      stop_ = true;
    }
    cv_.notify_all();
    for (auto& w : workers_) {
      if (w.joinable()) w.join();
    }
  }

 private:
  std::vector<std::thread> workers_;
  std::queue<std::function<void()>> tasks_;
  std::mutex mutex_;
  std::condition_variable cv_;
  bool stop_;
};

// Completion tracker ensuring worker-completion race freedom
struct CompletionTracker {
  std::mutex mutex;
  std::condition_variable cv;
  int remaining;
  explicit CompletionTracker(int count) : remaining(count) {}

  void FinishOne() {
    std::lock_guard<std::mutex> lock(mutex);
    if (--remaining == 0) {
      cv.notify_all();
    }
  }

  void Wait() {
    std::unique_lock<std::mutex> lock(mutex);
    cv.wait(lock, [this]() { return remaining == 0; });
  }
};

struct SearchConfig {
  std::string name;
  int rollout_workers = 128;
  int search_batch_size = 128;
  int evaluator_count = 1;
  bool amp = false;
  int root_window = 0;  // 0 = derived from rollout_workers

  int EffectiveRootWindow() const {
    return root_window > 0 ? root_window
                           : std::min(16, std::max(4, rollout_workers / 8));
  }
};

struct SearchWorkflowResult {
  std::vector<SearchExample> examples;
  int positive_weight_count = 0;
  double total_teaching_weight = 0.0;
  int agreement_count = 0;
  int override_count = 0;
  double elapsed_seconds = 0.0;
};

SearchWorkflowResult ExecuteSearchWorkflow(
    const SearchConfig& cfg,
    const std::vector<size_t>& root_indices,
    const std::shared_ptr<const open_spiel::Game>& game,
    const std::vector<CollectedDecision>& all_decisions,
    std::shared_ptr<open_spiel::SharedDunePolicyValueNetImpl> cycle_start_model,
    std::shared_ptr<open_spiel::SharedDunePolicyValueNetImpl> critic_model,
    torch::Device device,
    int rollouts_per_candidate,
    int confirmation_rollouts,
    uint64_t base_seed,
    int pilot_cycle,
    bool log_progress) {
  auto t_search_start = std::chrono::steady_clock::now();
  std::vector<std::shared_ptr<open_spiel::BatchedEvaluator>> a_batchers;
  std::vector<std::shared_ptr<open_spiel::BatchedEvaluator>> c_batchers;
  std::vector<std::unique_ptr<std::shared_mutex>> a_mutexes;
  std::vector<std::unique_ptr<std::shared_mutex>> c_mutexes;

  for (int ev = 0; ev < cfg.evaluator_count; ++ev) {
    a_mutexes.push_back(std::make_unique<std::shared_mutex>());
    a_batchers.push_back(std::make_shared<open_spiel::BatchedEvaluator>(
        cycle_start_model, cfg.search_batch_size, /*timeout_ms=*/1, device,
        a_mutexes.back().get(), /*logit_cap=*/10.0f, /*device_synchronize=*/false,
        /*high_priority_stream=*/true, /*emit_batch_membership=*/false,
        /*rollout_amp=*/cfg.amp, /*allow_tf32=*/cfg.amp));

    c_mutexes.push_back(std::make_unique<std::shared_mutex>());
    c_batchers.push_back(std::make_shared<open_spiel::BatchedEvaluator>(
        critic_model, cfg.search_batch_size, /*timeout_ms=*/1, device,
        c_mutexes.back().get(), /*logit_cap=*/0.0f, /*device_synchronize=*/false,
        /*high_priority_stream=*/true, /*emit_batch_membership=*/false,
        /*rollout_amp=*/cfg.amp, /*allow_tf32=*/cfg.amp));
  }

  ConcurrentRolloutPool rollout_pool(cfg.rollout_workers);
  SearchWorkflowResult res;
  res.examples.resize(root_indices.size());

  std::atomic<size_t> next_search_idx{0};
  std::atomic<size_t> completed_searches{0};
  std::mutex progress_log_mutex;
  std::atomic<int> positive_weight_count{0};
  std::atomic<double> total_teaching_weight{0.0};
  std::atomic<int> agreement_count{0};
  std::atomic<int> override_count{0};

  const int active_root_window = cfg.EffectiveRootWindow();
  std::vector<std::thread> root_coordinators;

  for (int c_id = 0; c_id < active_root_window; ++c_id) {
    root_coordinators.emplace_back([&, c_id]() {
      while (true) {
        size_t s_idx = next_search_idx.fetch_add(1);
        if (s_idx >= root_indices.size()) break;

        const size_t store_idx = root_indices[s_idx];
        const auto& dec = all_decisions[store_idx];

        std::unique_ptr<open_spiel::State> root_state = game->NewInitialState();
        for (open_spiel::Action a : dec.action_history) {
          root_state->ApplyAction(a);
        }
        auto* dune_root = dynamic_cast<open_spiel::dune_imperium::DuneImperiumState*>(root_state.get());

        // Derive candidate selection seed
        uint64_t cand_seed = dune_seed::DeriveSeed(base_seed, pilot_cycle, dec.game_id, dec.decision_idx, kStreamSearchCandSelect);
        std::mt19937_64 cand_rng(cand_seed);
        std::vector<open_spiel::Action> candidates = GenerateCandidates(
            dec.legal_actions, dec.collecting_policy, cand_rng);

        open_spiel::Action raw_action = dec.legal_actions[0];
        double best_prior = -1.0;
        for (size_t i = 0; i < dec.legal_actions.size(); ++i) {
          if (dec.collecting_policy[i] > best_prior) {
            best_prior = dec.collecting_policy[i];
            raw_action = dec.legal_actions[i];
          }
        }

        // Discovery rollouts
        const int num_rollouts = rollouts_per_candidate;
        int total_disc_rollouts = num_rollouts * candidates.size();
        auto disc_tracker = std::make_shared<CompletionTracker>(total_disc_rollouts);
        auto cand_sum_u = std::make_shared<std::vector<std::atomic<double>>>(candidates.size());
        for (auto& a : *cand_sum_u) a.store(0.0);

        uint64_t decision_seed = dune_seed::DeriveSeed(base_seed, pilot_cycle, dec.game_id, dec.decision_idx, kStreamSearchPosSelect);

        for (int r = 0; r < num_rollouts; ++r) {
          uint64_t hw_seed = dune_seed::DeriveSeed(decision_seed, r, kStreamSearchHiddenWorld);
          uint64_t cont_seed = dune_seed::DeriveSeed(decision_seed, r, kStreamSearchContinuation);

          std::unique_ptr<open_spiel::State> sim_state = nullptr;
          if (dune_root != nullptr) {
            std::mt19937 wrng(hw_seed);
            auto rng_func = [&wrng]() { return std::generate_canonical<double, 53>(wrng); };
            sim_state = dune_root->ResampleFromInfostate(dec.player, rng_func);
          }
          if (!sim_state) sim_state = root_state->Clone();

          for (size_t c_idx = 0; c_idx < candidates.size(); ++c_idx) {
            std::shared_ptr<open_spiel::State> branch_state(sim_state->Clone().release());
            branch_state->ApplyAction(candidates[c_idx]);

            int ev_idx = (cfg.evaluator_count > 1) ? ((c_id + c_idx) % cfg.evaluator_count) : 0;
            auto a_batcher = a_batchers[ev_idx];
            auto c_batcher = c_batchers[ev_idx];

            rollout_pool.Enqueue([branch_state, player = dec.player, c_idx, cont_seed,
                                  a_batcher, c_batcher, cand_sum_u, disc_tracker]() {
              double u = open_spiel::dune_pilot::SimulateTruncatedRollout(
                  branch_state.get(), player, a_batcher, c_batcher, cont_seed);
              double cur = (*cand_sum_u)[c_idx].load();
              while (!(*cand_sum_u)[c_idx].compare_exchange_weak(cur, cur + u)) {}
              disc_tracker->FinishOne();
            });
          }
        }

        // Wait for all discovery rollouts of this root to complete
        disc_tracker->Wait();

        open_spiel::Action selected_move = raw_action;
        double best_cand_u = -1e9;
        for (size_t c_idx = 0; c_idx < candidates.size(); ++c_idx) {
          double mean_u = (*cand_sum_u)[c_idx].load() / num_rollouts;
          if (mean_u > best_cand_u) {
            best_cand_u = mean_u;
            selected_move = candidates[c_idx];
          }
        }

        double d = 0.0;
        double se = 0.0;
        double w = 0.0;

        if (selected_move == raw_action) {
          agreement_count.fetch_add(1);
          d = 0.0;
          se = 0.0;
          w = 0.0;
        } else {
          override_count.fetch_add(1);
          const int num_conf = confirmation_rollouts;
          auto conf_tracker = std::make_shared<CompletionTracker>(num_conf);
          auto diffs = std::make_shared<std::vector<double>>(num_conf, 0.0);

          for (int r = 0; r < num_conf; ++r) {
            uint64_t conf_hw_seed = dune_seed::DeriveSeed(decision_seed, 50000 + r, kStreamConfHiddenWorld);
            uint64_t conf_cont_seed = dune_seed::DeriveSeed(decision_seed, 50000 + r, kStreamConfContinuation);

            std::unique_ptr<open_spiel::State> sim_state = nullptr;
            if (dune_root != nullptr) {
              std::mt19937 wrng(conf_hw_seed);
              auto rng_func = [&wrng]() { return std::generate_canonical<double, 53>(wrng); };
              sim_state = dune_root->ResampleFromInfostate(dec.player, rng_func);
            }
            if (!sim_state) sim_state = root_state->Clone();

            std::shared_ptr<open_spiel::State> sel_state(sim_state->Clone().release());
            sel_state->ApplyAction(selected_move);
            std::shared_ptr<open_spiel::State> raw_state(sim_state->Clone().release());
            raw_state->ApplyAction(raw_action);

            int ev_idx = (cfg.evaluator_count > 1) ? ((c_id + r) % cfg.evaluator_count) : 0;
            auto a_batcher = a_batchers[ev_idx];
            auto c_batcher = c_batchers[ev_idx];

            rollout_pool.Enqueue([sel_state, raw_state,
                                  player = dec.player, r, conf_cont_seed, a_batcher, c_batcher,
                                  diffs, conf_tracker]() {
              double u_sel = open_spiel::dune_pilot::SimulateTruncatedRollout(
                  sel_state.get(), player, a_batcher, c_batcher, conf_cont_seed);
              double u_raw = open_spiel::dune_pilot::SimulateTruncatedRollout(
                  raw_state.get(), player, a_batcher, c_batcher, conf_cont_seed);
              (*diffs)[r] = u_sel - u_raw;
              conf_tracker->FinishOne();
            });
          }

          conf_tracker->Wait();

          double sum_diff = 0.0;
          for (double diff : *diffs) sum_diff += diff;
          d = sum_diff / num_conf;

          double var_diff = 0.0;
          for (double diff : *diffs) var_diff += (diff - d) * (diff - d);
          var_diff /= std::max(1, num_conf - 1);
          se = std::sqrt(var_diff / num_conf);

          w = std::clamp((d - se) / 0.15, 0.0, 1.0);
        }

        if (w > 0.0) {
          positive_weight_count.fetch_add(1);
          double cur_w = total_teaching_weight.load();
          while (!total_teaching_weight.compare_exchange_weak(cur_w, cur_w + w)) {}
        }

        // Target distribution q = (1 - w)*pi + w*onehot(selected_move)
        std::vector<double> q(dec.legal_actions.size(), 0.0);
        for (size_t i = 0; i < dec.legal_actions.size(); ++i) {
          double pi_a = dec.collecting_policy[i];
          double ind = (dec.legal_actions[i] == selected_move) ? 1.0 : 0.0;
          q[i] = (1.0 - w) * pi_a + w * ind;
        }

        open_spiel::dune_pilot::SearchExample ex;
        ex.decision_store_idx = store_idx;
        ex.q = std::move(q);
        ex.selected_move = selected_move;
        ex.raw_action = raw_action;
        ex.advantage_d = d;
        ex.se = se;
        ex.teaching_weight_w = w;

        res.examples[s_idx] = std::move(ex);

        size_t completed = ++completed_searches;
        if (log_progress) {
          size_t report_interval = std::max<size_t>(5, root_indices.size() / 10);
          if (completed % report_interval == 0 || completed == root_indices.size()) {
            std::lock_guard<std::mutex> lock(progress_log_mutex);
            auto now = std::chrono::steady_clock::now();
            double el = std::chrono::duration<double>(now - t_search_start).count();
            double pct = 100.0 * completed / root_indices.size();
            double rate = completed / std::max(0.001, el);
            double eta = (root_indices.size() - completed) / std::max(0.001, rate);
            std::cout << absl::StrFormat("[SEARCH-PROGRESS] Completed %zu/%zu root searches (%.1f%%) in %.1fs | Rate: %.2f roots/s | ETA: %.1fs\n",
                                         completed, root_indices.size(), pct, el, rate, eta) << std::flush;
          }
        }
      }
    });
  }

  for (auto& w : root_coordinators) w.join();

  auto t_search_end = std::chrono::steady_clock::now();
  res.elapsed_seconds = std::chrono::duration<double>(t_search_end - t_search_start).count();
  res.positive_weight_count = positive_weight_count.load();
  res.total_teaching_weight = total_teaching_weight.load();
  res.agreement_count = agreement_count.load();
  res.override_count = override_count.load();
  return res;
}

SearchConfig RunConcurrencyBenchmark(
    std::shared_ptr<open_spiel::SharedDunePolicyValueNetImpl> cycle_start_model,
    std::shared_ptr<open_spiel::SharedDunePolicyValueNetImpl> critic_model,
    const std::shared_ptr<const open_spiel::Game>& game,
    const std::vector<CollectedDecision>& all_decisions,
    const std::vector<size_t>& searched_indices,
    torch::Device device,
    int rollouts_per_candidate,
    int confirmation_rollouts,
    uint64_t base_seed,
    int pilot_cycle,
    double& out_benchmark_elapsed_seconds) {
  std::cout << "\n=================================================================\n";
  std::cout << "  Requirement 2: Search Concurrency Micro-Benchmark\n";
  std::cout << "=================================================================\n";
  std::cout << "  Benchmarking configurations on 4 identical roots & seeds...\n";

  std::vector<SearchConfig> configs = {
    {"B64-W64-E1",   64,  64,  1},
    {"B128-W128-E1", 128, 128, 1},
    {"B64-W64-E2",   64,  64,  2},
    {"B128-W128-E2", 128, 128, 2},
  };

  const size_t num_bench_roots = std::min<size_t>(4, searched_indices.size());
  std::vector<size_t> bench_roots(searched_indices.begin(), searched_indices.begin() + num_bench_roots);

  std::cout << "  -----------------------------------------------------------------\n";
  std::cout << "  Config         Workers  Batch  Evals  Time (s)  Roots/s\n";
  std::cout << "  -----------------------------------------------------------------\n";

  SearchConfig best_cfg = configs[0];
  double best_pos_per_s = -1.0;
  out_benchmark_elapsed_seconds = 0.0;

  for (const auto& cfg : configs) {
    auto res = ExecuteSearchWorkflow(
        cfg, bench_roots, game, all_decisions, cycle_start_model, critic_model,
        device, rollouts_per_candidate, confirmation_rollouts,
        base_seed, pilot_cycle, /*log_progress=*/false);

    out_benchmark_elapsed_seconds += res.elapsed_seconds;
    double pos_per_s = num_bench_roots / std::max(1e-4, res.elapsed_seconds);

    std::cout << absl::StrFormat("  %-14s %-8d %-6d %-6d %-9.2f %-6.2f\n",
                                 cfg.name, cfg.rollout_workers, cfg.search_batch_size, cfg.evaluator_count,
                                 res.elapsed_seconds, pos_per_s);

    if (pos_per_s > best_pos_per_s) {
      best_pos_per_s = pos_per_s;
      best_cfg = cfg;
    }
  }
  std::cout << "  -----------------------------------------------------------------\n";
  std::cout << absl::StrFormat("  [BENCHMARK-WINNER] Selected: %s (Workers=%d, Batch=%d, Evals=%d) -> %.2f roots/s\n\n",
                               best_cfg.name, best_cfg.rollout_workers, best_cfg.search_batch_size, best_cfg.evaluator_count, best_pos_per_s);
  return best_cfg;
}

}  // namespace dune_pilot
}  // namespace open_spiel

int main(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);

  if (absl::GetFlag(FLAGS_run_terminal_diagnostic)) {
    return open_spiel::dune_pilot::RunTerminalContinuationDiagnostic(
        absl::GetFlag(FLAGS_diag_candidate_checkpoint),
        absl::GetFlag(FLAGS_diag_preceding_checkpoint),
        absl::GetFlag(FLAGS_diag_u31900_checkpoint),
        absl::GetFlag(FLAGS_diag_opponent_checkpoint),
        absl::GetFlag(FLAGS_diag_games),
        absl::GetFlag(FLAGS_diag_continuations),
        absl::GetFlag(FLAGS_diag_max_changed),
        absl::GetFlag(FLAGS_diag_output_dir),
        absl::GetFlag(FLAGS_device),
        absl::GetFlag(FLAGS_seed),
        absl::GetFlag(FLAGS_threads));
  }

  const std::string actor_ckpt = absl::GetFlag(FLAGS_actor_checkpoint);
  const std::string actor_opt = absl::GetFlag(FLAGS_actor_optimizer);
  const std::string actor_ema = absl::GetFlag(FLAGS_actor_ema_checkpoint);
  const std::string actor_ema_man = absl::GetFlag(FLAGS_actor_ema_manifest);
  const std::string critic_ckpt = absl::GetFlag(FLAGS_critic_checkpoint);
  const std::string src_man = absl::GetFlag(FLAGS_source_manifest);
  const std::string out_dir = absl::GetFlag(FLAGS_output_dir);
  const std::string dev_str = absl::GetFlag(FLAGS_device);
  const int num_threads = absl::GetFlag(FLAGS_threads);
  const int rollout_workers = absl::GetFlag(FLAGS_rollout_workers);
  const int search_batch_size = absl::GetFlag(FLAGS_search_batch_size);
  const int evaluator_count = absl::GetFlag(FLAGS_evaluator_count);
  const int games_per_cycle = absl::GetFlag(FLAGS_games_per_cycle);
  const uint64_t base_seed = absl::GetFlag(FLAGS_seed);
  const int pilot_cycle = absl::GetFlag(FLAGS_pilot_cycle);
  const double planned_s = absl::GetFlag(FLAGS_planned_s);

  std::cout << "=================================================================\n";
  std::cout << "  Actor-Only Search-Supervision Pilot (High-Concurrency Architecture)\n";
  std::cout << "=================================================================\n";
  std::cout << "  Actor model:       " << actor_ckpt << "\n";
  std::cout << "  Actor optimizer:   " << actor_opt << "\n";
  std::cout << "  Actor EMA:         " << actor_ema << "\n";
  std::cout << "  Actor EMA manifest:" << actor_ema_man << "\n";
  std::cout << "  Immutable critic:  " << critic_ckpt << "\n";
  std::cout << "  Output dir:        " << out_dir << "\n";
  std::cout << "  Collection threads:" << num_threads << "\n";
  std::cout << "  Rollout workers:   " << rollout_workers << "\n";
  std::cout << "  Search batch size: " << search_batch_size << "\n";
  std::cout << "  Evaluator count:   " << evaluator_count << "\n";
  std::cout << "  Games per cycle:   " << games_per_cycle << "\n";
  std::cout << "  Base seed:         " << base_seed << "\n";
  std::cout << "  Pilot cycle:       " << pilot_cycle << "\n";
  std::cout << "  Device:            " << dev_str << "\n" << std::flush;

  std::filesystem::create_directories(out_dir + "/checkpoints");

  torch::Device device = (dev_str == "cuda" && torch::cuda::is_available())
      ? torch::Device(torch::kCUDA, 0)
      : torch::Device(torch::kCPU);

  auto game = open_spiel::LoadGame("dune_imperium");
  SPIEL_CHECK_TRUE(game != nullptr);

  // Model instantiation: 12-block ResNet with semantic scorer and FullPublicInformationV3
  auto learner_model = std::make_shared<open_spiel::SharedDunePolicyValueNetImpl>(
      9182, 2048, 2391, 12, false, false, 0, true);
  learner_model->semantic_descriptor_schema_ = open_spiel::dune_semantic::kDescriptorSchemaVersionV3;
  learner_model->market_appendix_mode_ = open_spiel::dune_imperium::MarketAppendixMode::kFullPublicInformationV3;
  open_spiel::LoadModelCheckpointRobust(learner_model, actor_ckpt, device);
  learner_model->eval();

  auto cycle_start_model = std::make_shared<open_spiel::SharedDunePolicyValueNetImpl>(
      9182, 2048, 2391, 12, false, false, 0, true);
  cycle_start_model->semantic_descriptor_schema_ = open_spiel::dune_semantic::kDescriptorSchemaVersionV3;
  cycle_start_model->market_appendix_mode_ = open_spiel::dune_imperium::MarketAppendixMode::kFullPublicInformationV3;
  open_spiel::LoadModelCheckpointRobust(cycle_start_model, actor_ckpt, device);
  cycle_start_model->eval();

  auto ema_model = std::make_shared<open_spiel::SharedDunePolicyValueNetImpl>(
      9182, 2048, 2391, 12, false, false, 0, true);
  ema_model->semantic_descriptor_schema_ = open_spiel::dune_semantic::kDescriptorSchemaVersionV3;
  ema_model->market_appendix_mode_ = open_spiel::dune_imperium::MarketAppendixMode::kFullPublicInformationV3;
  if (!actor_ema.empty() && std::filesystem::exists(actor_ema)) {
    open_spiel::LoadModelCheckpointRobust(ema_model, actor_ema, device);
    std::cout << "[INIT] Loaded starting EMA model from: " << actor_ema << "\n";
  } else {
    std::cerr << "FATAL: Required EMA model missing: " << actor_ema << "\n";
    return 1;
  }

  auto critic_model = std::make_shared<open_spiel::SharedDunePolicyValueNetImpl>(
      9182, 2048, 2391, 12, false, false, 0, true);
  critic_model->semantic_descriptor_schema_ = open_spiel::dune_semantic::kDescriptorSchemaVersionV3;
  critic_model->market_appendix_mode_ = open_spiel::dune_imperium::MarketAppendixMode::kFullPublicInformationV3;
  open_spiel::LoadModelCheckpointRobust(critic_model, critic_ckpt, device);
  critic_model->eval();

  // Optimizer setup matching inherited 3 parameter groups:
  // Group 0: policy head (5e-6)
  // Group 1: torso (input layer, first 8 res blocks, value head, semantic scorer) (5e-6)
  // Group 2: remaining blocks (last 4 res blocks) (5e-5)
  std::vector<torch::Tensor> grp0_params = learner_model->policy_head->parameters();
  std::vector<torch::Tensor> grp1_params;
  for (auto& p : learner_model->input_layer->parameters()) grp1_params.push_back(p);
  for (int b = 0; b < 8 && b < static_cast<int>(learner_model->res_blocks.size()); ++b) {
    for (auto& p : learner_model->res_blocks[b]->parameters()) grp1_params.push_back(p);
  }
  for (auto& p : learner_model->value_head->parameters()) grp1_params.push_back(p);
  if (learner_model->semantic_scorer_) {
    for (auto& p : learner_model->semantic_scorer_->parameters()) grp1_params.push_back(p);
  }
  std::vector<torch::Tensor> grp2_params;
  for (size_t b = 8; b < learner_model->res_blocks.size(); ++b) {
    for (auto& p : learner_model->res_blocks[b]->parameters()) grp2_params.push_back(p);
  }

  std::vector<torch::optim::OptimizerParamGroup> param_groups = {
      torch::optim::OptimizerParamGroup(grp0_params, std::make_unique<torch::optim::AdamWOptions>(5e-6)),
      torch::optim::OptimizerParamGroup(grp1_params, std::make_unique<torch::optim::AdamWOptions>(5e-6)),
      torch::optim::OptimizerParamGroup(grp2_params, std::make_unique<torch::optim::AdamWOptions>(5e-5)),
  };

  auto optimizer = std::make_shared<torch::optim::AdamW>(
      param_groups, torch::optim::AdamWOptions(5e-6).weight_decay(1e-4).eps(1e-8));

  if (!actor_opt.empty() && std::filesystem::exists(actor_opt)) {
    try {
      torch::load(*optimizer, actor_opt, device);
      std::cout << "[INIT] Successfully loaded inherited optimizer state from: " << actor_opt << "\n";
    } catch (const std::exception& e) {
      std::cout << "[INIT] Optimizer GPU load fallback to CPU: " << e.what() << "\n";
      torch::load(*optimizer, actor_opt, torch::kCPU);
    }
  } else {
    std::cerr << "FATAL: Actor optimizer checkpoint missing: " << actor_opt << "\n";
    return 1;
  }

  SPIEL_CHECK_EQ(optimizer->param_groups().size(), 3);
  double lr0 = static_cast<const torch::optim::AdamWOptions&>(optimizer->param_groups()[0].options()).lr();
  double lr1 = static_cast<const torch::optim::AdamWOptions&>(optimizer->param_groups()[1].options()).lr();
  double lr2 = static_cast<const torch::optim::AdamWOptions&>(optimizer->param_groups()[2].options()).lr();
  std::cout << absl::StrFormat("[INIT] Preserved optimizer learning rates: grp0=%.2e, grp1=%.2e, grp2=%.2e\n",
                               lr0, lr1, lr2);
  SPIEL_CHECK_FLOAT_EQ(lr0, 5e-6);
  SPIEL_CHECK_FLOAT_EQ(lr1, 5e-6);
  SPIEL_CHECK_FLOAT_EQ(lr2, 5e-5);

  // Initialize BatchedEvaluators for collection
  std::shared_mutex coll_actor_mutex;
  auto coll_actor_batcher = std::make_shared<open_spiel::BatchedEvaluator>(
      cycle_start_model, /*target_batch_size=*/16, /*timeout_ms=*/1, device,
      &coll_actor_mutex, /*logit_cap=*/10.0f, /*device_synchronize=*/false,
      /*high_priority_stream=*/true, /*emit_batch_membership=*/false,
      /*rollout_amp=*/false, /*allow_tf32=*/false);

  // Phase 1: Timed Collection of Games
  std::cout << "\n=================================================================\n";
  std::cout << absl::StrFormat("  Phase 1: Timed Collection of %d Games (Batched FP32)\n", games_per_cycle);
  std::cout << "=================================================================\n" << std::flush;

  std::vector<open_spiel::dune_pilot::CollectedDecision> all_decisions;
  std::vector<open_spiel::dune_pilot::SearchExample> search_examples;
  std::vector<size_t> c1_train_search, c1_holdout_search, c1_train_pres, c1_holdout_pres;

  const std::string load_replay = absl::GetFlag(FLAGS_load_replay_path);
  if (!load_replay.empty()) {
    std::cout << "[REPLAY] Loading Cycle 1 replay data from: " << load_replay << "...\n";
    if (!open_spiel::dune_pilot::LoadReplayData(load_replay, all_decisions, search_examples,
                                                c1_train_search, c1_holdout_search,
                                                c1_train_pres, c1_holdout_pres)) {
      std::cerr << "FATAL: Failed to load replay data from " << load_replay << "\n";
      return 1;
    }
    std::cout << absl::StrFormat("[REPLAY] Loaded %zu Cycle 1 decisions and %zu search examples.\n",
                                 all_decisions.size(), search_examples.size());
  }
  size_t c1_dec_offset = all_decisions.size();
  size_t c1_ex_offset = search_examples.size();

  auto t_collect_start = std::chrono::steady_clock::now();
  std::mutex collect_mutex;

  std::vector<std::thread> workers;
  std::atomic<int> next_game_idx{0};
  std::atomic<int> completed_games{0};

  for (int th = 0; th < num_threads; ++th) {
    workers.emplace_back([&, th]() {
      while (true) {
        int g_idx = next_game_idx.fetch_add(1);
        if (g_idx >= games_per_cycle) break;

        uint64_t game_seed = dune_seed::DeriveSeed(base_seed, pilot_cycle, 20261012, g_idx, 0);
        auto chance_rng = dune_seed::MakeRng64(
            dune_seed::DeriveSeed(game_seed, 0, 0, dune_seed::kStreamChance));
        std::mt19937_64 policy_rng[4];
        for (int p = 0; p < 4; ++p) {
          policy_rng[p] = dune_seed::MakeRng64(
              dune_seed::DeriveSeed(game_seed, 0, 0, dune_seed::kStreamPolicyPlayer0 + p));
        }

        std::unique_ptr<open_spiel::State> state = game->NewInitialState();
        auto* dune_state = dynamic_cast<open_spiel::dune_imperium::DuneImperiumState*>(state.get());

        std::vector<open_spiel::Action> game_history;
        std::vector<open_spiel::dune_pilot::CollectedDecision> game_decisions;
        int dec_idx = 0;

        while (!state->IsTerminal()) {
          if (state->IsChanceNode()) {
            auto outcomes = state->ChanceOutcomes();
            open_spiel::Action ca = open_spiel::SampleAction(outcomes, chance_rng).first;
            game_history.push_back(ca);
            state->ApplyAction(ca);
            continue;
          }

          open_spiel::Player cur = state->CurrentPlayer();
          std::vector<open_spiel::Action> legals = state->LegalActions();
          if (legals.empty()) break;

          open_spiel::Action chosen = legals.front();

          if (legals.size() == 1) {
            chosen = legals.front();
          } else {
            open_spiel::dune_pilot::CollectedDecision dec;
            dec.cycle_idx = (pilot_cycle > 0 ? pilot_cycle : 1);
            dec.game_id = g_idx;
            dec.decision_idx = dec_idx++;
            dec.player = cur;
            dec.legal_actions = legals;
            dec.action_history = game_history;

            dec.obs.assign(9182, 0.0f);
            if (dune_state != nullptr) {
              dune_state->InformationStateTensorWithAppendix(
                  cur, open_spiel::dune_imperium::MarketAppendixMode::kFullPublicInformationV3,
                  absl::MakeSpan(dec.obs));
              open_spiel::dune_semantic::ExtractCandidateDescriptors(
                  *dune_state, legals, &dec.cand_data,
                  open_spiel::dune_semantic::kDescriptorSchemaVersionV3);
            }

            dec.collecting_policy = open_spiel::dune_pilot::ComputePolicyProbabilities(
                coll_actor_batcher, dec.obs, dec.cand_data, legals);

            bool is_draft = (dune_state != nullptr &&
                             dune_state->phase() == open_spiel::dune_imperium::GamePhase::kLeaderDraft);
            dec.is_leader_draft = is_draft;

            // Requirement 4: Keep uniform random legal leader drafting during collection
            if (is_draft) {
              std::uniform_int_distribution<size_t> pick(0, legals.size() - 1);
              chosen = legals[pick(policy_rng[cur])];
            } else {
              std::discrete_distribution<size_t> dist(dec.collecting_policy.begin(), dec.collecting_policy.end());
              chosen = legals[dist(policy_rng[cur])];
            }

            game_decisions.push_back(std::move(dec));
          }

          game_history.push_back(chosen);
          state->ApplyAction(chosen);
        }

        {
          std::lock_guard<std::mutex> lock(collect_mutex);
          all_decisions.insert(all_decisions.end(),
                               std::make_move_iterator(game_decisions.begin()),
                               std::make_move_iterator(game_decisions.end()));
          int c_games = ++completed_games;
          auto now = std::chrono::steady_clock::now();
          double el = std::chrono::duration<double>(now - t_collect_start).count();
          double rate = c_games / std::max(0.001, el);
          double eta = (games_per_cycle - c_games) / std::max(0.001, rate);
          std::cout << absl::StrFormat("[COLLECT-PROGRESS] Completed %d/%d games (%.1f%%) in %.1fs | Rate: %.2f games/s | ETA: %.1fs\n",
                                       c_games, games_per_cycle, 100.0 * c_games / games_per_cycle, el, rate, eta) << std::flush;
        }
      }
    });
  }

  for (auto& w : workers) w.join();
  workers.clear();
  coll_actor_batcher.reset();

  // Canonical stable sort by (game_id, decision_idx) to ensure strict scheduling independence
  std::sort(all_decisions.begin() + c1_dec_offset, all_decisions.end(), [](const auto& a, const auto& b) {
    if (a.game_id != b.game_id) return a.game_id < b.game_id;
    return a.decision_idx < b.decision_idx;
  });

  auto t_collect_end = std::chrono::steady_clock::now();
  double collect_elapsed_s = std::chrono::duration<double>(t_collect_end - t_collect_start).count();

  std::cout << absl::StrFormat("[COLLECT] Completed %d games in %.2f s (%.2f s/game).\n",
                               games_per_cycle, collect_elapsed_s, collect_elapsed_s / games_per_cycle);
  std::cout << absl::StrFormat("[COLLECT] Collected %zu nontrivial multi-action decisions.\n",
                               all_decisions.size());
  std::cout << "[COLLECT] RAM usage: " << absl::StrFormat("%.2f MB", open_spiel::dune_pilot::GetCurrentMemoryUsageMB()) << "\n";

  // Pre-Search Diagnostic: Verify Collection vs Learner Policy Parity
  std::cout << "\n=================================================================\n";
  std::cout << "  Pre-Search Diagnostic: Verify Collection vs Learner Policy Parity\n";
  std::cout << "=================================================================\n";
  double max_parity_prob_diff = 0.0;
  double max_parity_kl = 0.0;
  size_t start_check = c1_dec_offset;
  size_t parity_check_count = std::min<size_t>(20, all_decisions.size() - c1_dec_offset);
  for (size_t i = start_check; i < start_check + parity_check_count; ++i) {
    const auto& dec = all_decisions[i];
    torch::Tensor obs_t = torch::from_blob(
        const_cast<float*>(dec.obs.data()),
        {1, static_cast<int64_t>(dec.obs.size())}, torch::kFloat32).to(device);
    torch::Tensor mask_t = torch::zeros({1, 2391}, torch::TensorOptions().dtype(torch::kBool).device(device));
    for (open_spiel::Action a : dec.legal_actions) {
      if (a >= 0 && a < 2391) mask_t[0][a] = true;
    }
    std::vector<const open_spiel::dune_semantic::CandidateActionData*> cands = {&dec.cand_data};

    torch::NoGradGuard no_grad;
    auto outputs = learner_model->forward(obs_t);
    if (learner_model->with_semantic_scorer_ && learner_model->semantic_scorer_) {
      open_spiel::dune_semantic::ApplySemanticScorerBatch(
          learner_model->semantic_scorer_, outputs.trunk, cands, outputs.logits, device);
    }
    torch::Tensor capped = open_spiel::CenterAndCapLogitsTensor(outputs.logits, mask_t, 10.0f);
    torch::Tensor masked = torch::where(mask_t, capped, torch::full_like(capped, -1e9f));
    torch::Tensor log_p = torch::log_softmax(masked, -1).to(torch::kCPU);
    const float* log_p_ptr = log_p.data_ptr<float>();

    double kl = 0.0;
    for (size_t j = 0; j < dec.legal_actions.size(); ++j) {
      open_spiel::Action a = dec.legal_actions[j];
      double ref_p = dec.collecting_policy[j];
      double learner_p = (a >= 0 && a < 2391) ? std::exp(static_cast<double>(log_p_ptr[a])) : 0.0;
      double prob_diff = std::abs(ref_p - learner_p);
      if (prob_diff > max_parity_prob_diff) {
        max_parity_prob_diff = prob_diff;
        if (prob_diff > 1e-4) {
          std::cout << absl::StrFormat("[DEBUG-PARITY-FAIL] Dec %zu (game %d, dec %d, is_draft %d, act %d): ref_p=%.6f, learner_p=%.6f, diff=%.6f\n",
                                       i, dec.game_id, dec.decision_idx, (int)dec.is_leader_draft, (int)a, ref_p, learner_p, prob_diff);
          for (size_t k = 0; k < dec.legal_actions.size(); ++k) {
            open_spiel::Action ak = dec.legal_actions[k];
            double r_p = dec.collecting_policy[k];
            double l_p = (ak >= 0 && ak < 2391) ? std::exp(static_cast<double>(log_p_ptr[ak])) : 0.0;
            std::cout << absl::StrFormat("  legal[%zu]=act %d: stored_ref=%.6f, direct_learner=%.6f, diff=%.6f\n",
                                         k, (int)ak, r_p, l_p, std::abs(r_p - l_p));
          }
        }
      }
      if (ref_p > 1e-9 && learner_p > 1e-12) {
        kl += ref_p * (std::log(ref_p) - std::log(learner_p));
      }
    }
    if (kl > max_parity_kl) max_parity_kl = kl;
  }
  std::cout << absl::StrFormat("[PRE-SEARCH PARITY] Evaluated %zu decisions: Max prob diff = %.8f, Max KL = %.8f\n",
                               parity_check_count, max_parity_prob_diff, max_parity_kl);
  SPIEL_CHECK_LE(max_parity_prob_diff, 1e-4);
  SPIEL_CHECK_LE(max_parity_kl, 1e-4);
  std::cout << "[PRE-SEARCH PARITY] Parity proven: Collection probabilities match learner model under 10.0 cap.\n" << std::flush;

  // Requirement 3 & 4: Select 25% non-draft multi-action positions deterministically from current cycle decisions
  std::vector<size_t> searched_indices;
  size_t total_nondraft_decisions = 0;
  for (size_t i = c1_dec_offset; i < all_decisions.size(); ++i) {
    const auto& dec = all_decisions[i];
    // Requirement 4: Exclude leader drafting from search supervision
    if (dec.is_leader_draft) continue;
    total_nondraft_decisions++;

    // Requirement 3: Derive sampling stably from (base_seed, cycle, game_id, decision_idx, stream)
    uint64_t pos_seed = dune_seed::DeriveSeed(base_seed, pilot_cycle, dec.game_id, dec.decision_idx,
                                              open_spiel::dune_pilot::kStreamSearchPosSelect);
    auto pos_rng = dune_seed::MakeRng64(pos_seed);
    double u = std::generate_canonical<double, 53>(pos_rng);
    if (u < absl::GetFlag(FLAGS_search_sample_rate)) {
      searched_indices.push_back(i);
    }
  }

  std::cout << "\n=================================================================\n";
  std::cout << "  Phase 2: Timed Batched Search (25% Sampling, 32+32 Paired Rollouts)\n";
  std::cout << "=================================================================\n";
  std::cout << absl::StrFormat("[SEARCH] Selected %zu / %zu non-draft decisions for search (%.2f%%) [%zu total incl draft].\n",
                               searched_indices.size(), total_nondraft_decisions,
                               100.0 * searched_indices.size() / std::max<size_t>(1, total_nondraft_decisions),
                               all_decisions.size()) << std::flush;

  // Requirement 6: Compare values returned by actual batched search critic path with direct values
  std::shared_mutex probe_critic_mutex;
  auto probe_critic_batcher = std::make_shared<open_spiel::BatchedEvaluator>(
      critic_model, 64, /*timeout_ms=*/1, device,
      &probe_critic_mutex, /*logit_cap=*/0.0f, /*device_synchronize=*/false,
      /*high_priority_stream=*/true, /*emit_batch_membership=*/false,
      /*rollout_amp=*/false, /*allow_tf32=*/false);

  std::vector<std::unique_ptr<open_spiel::State>> critic_probe_states;
  std::vector<open_spiel::Player> critic_probe_players;
  std::vector<double> critic_probe_vals_before;

  for (size_t i = 0; i < std::min<size_t>(50, searched_indices.size()); ++i) {
    const auto& dec = all_decisions[searched_indices[i]];
    std::unique_ptr<open_spiel::State> probe_s = game->NewInitialState();
    for (open_spiel::Action a : dec.action_history) {
      probe_s->ApplyAction(a);
    }
    double direct_val = open_spiel::dune_pilot::EvaluateCriticDirect(critic_model, *probe_s, dec.player, device);
    double batched_val = open_spiel::dune_pilot::EvaluateCriticLeafValue(probe_critic_batcher, *probe_s, dec.player);
    double diff = std::abs(direct_val - batched_val);
    SPIEL_CHECK_LE(diff, 1e-4);

    critic_probe_vals_before.push_back(direct_val);
    critic_probe_states.push_back(std::move(probe_s));
    critic_probe_players.push_back(dec.player);
  }
  std::cout << "[VERIFY-CRITIC] Batched search critic path matches direct critic within 1e-4 tolerance across probe states.\n" << std::flush;

  const int rollouts_per_cand = absl::GetFlag(FLAGS_rollouts_per_candidate);
  const int num_conf = absl::GetFlag(FLAGS_confirmation_rollouts);

  open_spiel::dune_pilot::SearchConfig selected_search_cfg;
  selected_search_cfg.name = "CLI-Default";
  selected_search_cfg.rollout_workers = rollout_workers;
  selected_search_cfg.search_batch_size = search_batch_size;
  selected_search_cfg.evaluator_count = evaluator_count;
  double benchmark_elapsed_s = 0.0;

  if (absl::GetFlag(FLAGS_benchmark_concurrency)) {
    selected_search_cfg = open_spiel::dune_pilot::RunConcurrencyBenchmark(
        cycle_start_model, critic_model, game, all_decisions, searched_indices,
        device, rollouts_per_cand, num_conf, base_seed, pilot_cycle, benchmark_elapsed_s);
  }

  selected_search_cfg.amp = absl::GetFlag(FLAGS_search_amp);
  selected_search_cfg.root_window = absl::GetFlag(FLAGS_search_root_window);

  // Requirement 2: Execute search using selected measured configuration
  std::cout << absl::StrFormat("\n[SEARCH] Launching production search using %s (Workers=%d, Batch=%d, Evaluators=%d, RootWindow=%d, Precision=%s)...\n",
                               selected_search_cfg.name, selected_search_cfg.rollout_workers,
                               selected_search_cfg.search_batch_size, selected_search_cfg.evaluator_count,
                               selected_search_cfg.EffectiveRootWindow(),
                               selected_search_cfg.amp ? "BF16-autocast+TF32" : "FP32") << std::flush;

  auto search_res = open_spiel::dune_pilot::ExecuteSearchWorkflow(
      selected_search_cfg, searched_indices, game, all_decisions,
      cycle_start_model, critic_model, device, rollouts_per_cand, num_conf,
      base_seed, pilot_cycle, /*log_progress=*/true);
  if (selected_search_cfg.amp && device.is_cuda()) {
    // BatchedEvaluator sets TF32 process-wide; keep learning in pure FP32.
    at::globalContext().setAllowTF32CuBLAS(false);
    at::globalContext().setAllowTF32CuDNN(false);
  }

  double search_elapsed_s = search_res.elapsed_seconds;
  auto new_search_examples = std::move(search_res.examples);
  for (auto& ex : new_search_examples) {
    search_examples.push_back(std::move(ex));
  }

  int positive_weight_count = search_res.positive_weight_count;
  double total_teaching_weight = search_res.total_teaching_weight;
  int agreement_count = search_res.agreement_count;
  int override_count = search_res.override_count;

  std::cout << absl::StrFormat("[SEARCH] Completed %zu new searches in %.2f s (%.2f s/search) [%zu total].\n",
                               new_search_examples.size(), search_elapsed_s,
                               search_elapsed_s / std::max<size_t>(1, new_search_examples.size()),
                               search_examples.size());
  std::cout << absl::StrFormat("[SEARCH] Agreement positions: %d, Overrides evaluated: %d\n",
                               agreement_count, override_count);
  std::cout << absl::StrFormat("[SEARCH] Positive-weight targets (w > 0): %d / %zu (%.2f%%)\n",
                               positive_weight_count, new_search_examples.size(),
                               100.0 * positive_weight_count / std::max<size_t>(1, new_search_examples.size()));
  std::cout << absl::StrFormat("[SEARCH] Total teaching weight: %.4f (mean w = %.4f)\n",
                               total_teaching_weight,
                               total_teaching_weight / std::max<size_t>(1, new_search_examples.size()));
  std::cout << "[SEARCH] RAM usage: " << absl::StrFormat("%.2f MB", open_spiel::dune_pilot::GetCurrentMemoryUsageMB()) << "\n";

  // Partition newly searched and collected decisions
  // In the full pilot, train on ALL selected search positions and preserve ALL collected decisions.
  // Holdout partition is kept ONLY for the dedicated preflight learning check.
  const bool is_preflight = absl::GetFlag(FLAGS_preflight_check) || (pilot_cycle == 0);
  std::vector<size_t> c2_train_search, c2_holdout_search;
  for (size_t i = c1_ex_offset; i < search_examples.size(); ++i) {
    int gid = all_decisions[search_examples[i].decision_store_idx].game_id;
    if (is_preflight && (gid % 4 == 0)) {
      c2_holdout_search.push_back(i);
    } else {
      c2_train_search.push_back(i);
    }
  }

  std::vector<size_t> c2_train_pres, c2_holdout_pres;
  for (size_t i = c1_dec_offset; i < all_decisions.size(); ++i) {
    if (is_preflight && (all_decisions[i].game_id % 4 == 0)) {
      c2_holdout_pres.push_back(i);
    } else {
      c2_train_pres.push_back(i);
    }
  }

  std::vector<size_t> train_search_indices = c1_train_search;
  train_search_indices.insert(train_search_indices.end(), c2_train_search.begin(), c2_train_search.end());
  std::vector<size_t> hold_search_indices = c1_holdout_search;
  hold_search_indices.insert(hold_search_indices.end(), c2_holdout_search.begin(), c2_holdout_search.end());
  std::vector<size_t> train_pres_indices = c1_train_pres;
  train_pres_indices.insert(train_pres_indices.end(), c2_train_pres.begin(), c2_train_pres.end());
  std::vector<size_t> hold_pres_indices = c1_holdout_pres;
  hold_pres_indices.insert(hold_pres_indices.end(), c2_holdout_pres.begin(), c2_holdout_pres.end());

  std::cout << "\n=================================================================\n";
  std::cout << "  Phase 3: Dataset Partitions (Whole-Game Split, Zero-Copy Indexing)\n";
  std::cout << "=================================================================\n";
  std::cout << absl::StrFormat("  Search examples:       %zu total (Train: %zu, Holdout: %zu)\n",
                               search_examples.size(), train_search_indices.size(), hold_search_indices.size());
  std::cout << absl::StrFormat("  Preservation examples: %zu total (Train: %zu, Holdout: %zu)\n",
                               all_decisions.size(), train_pres_indices.size(), hold_pres_indices.size());

  // Load refreshed preservation reference model (frozen cycle-start policy)
  std::shared_ptr<open_spiel::SharedDunePolicyValueNetImpl> pres_ref_model = nullptr;
  std::string pres_ref_ckpt = absl::GetFlag(FLAGS_preservation_ref_checkpoint);
  if (!pres_ref_ckpt.empty() && std::filesystem::exists(pres_ref_ckpt)) {
    std::cout << "[PRESERVATION-REF] Loading refreshed preservation reference from: " << pres_ref_ckpt << "\n";
    pres_ref_model = open_spiel::dune_pilot::LoadActorModel(pres_ref_ckpt, device);
  } else {
    std::cout << "[PRESERVATION-REF] Using starting actor as preservation reference.\n";
    pres_ref_model = cycle_start_model;
  }
  pres_ref_model->eval();

  // Evaluate metrics on partitions using references to all_decisions
  auto EvaluatePartition = [&](
      const std::vector<size_t>& s_indices,
      const std::vector<size_t>& p_indices,
      std::shared_ptr<open_spiel::SharedDunePolicyValueNetImpl> eval_model,
      double& out_total_loss, double& out_ce, double& out_kl,
      double& out_pos_target_prob, double& out_greedy_adopt) {
    eval_model->eval();
    torch::NoGradGuard no_grad;

    double sum_ce = 0.0;
    double sum_target_p = 0.0;
    int target_p_count = 0;
    int greedy_adopt_count = 0;
    int pos_weight_count = 0;

    for (size_t s_idx : s_indices) {
      const auto& ex = search_examples[s_idx];
      const auto& dec = all_decisions[ex.decision_store_idx];

      torch::Tensor obs_t = torch::from_blob(
          const_cast<float*>(dec.obs.data()), {1, static_cast<int64_t>(dec.obs.size())}, torch::kFloat32).to(device);
      auto outputs = eval_model->forward(obs_t);

      std::vector<const open_spiel::dune_semantic::CandidateActionData*> cands = {&dec.cand_data};
      if (eval_model->with_semantic_scorer_ && eval_model->semantic_scorer_) {
        open_spiel::dune_semantic::ApplySemanticScorerBatch(
            eval_model->semantic_scorer_, outputs.trunk, cands, outputs.logits, device);
      }

      torch::Tensor mask_t = torch::zeros({1, 2391}, torch::TensorOptions().dtype(torch::kBool).device(device));
      for (open_spiel::Action a : dec.legal_actions) {
        if (a >= 0 && a < 2391) mask_t[0][a] = true;
      }

      torch::Tensor capped = open_spiel::CenterAndCapLogitsTensor(outputs.logits, mask_t, 10.0f);
      torch::Tensor cpu_capped = capped.to(torch::kCPU);
      const float* p_logits = cpu_capped.data_ptr<float>();

      double max_l = -1e9;
      for (open_spiel::Action a : dec.legal_actions) {
        float val = (a >= 0 && a < 2391) ? p_logits[a] : -1e9f;
        if (val > max_l) max_l = val;
      }
      double sum_exp = 0.0;
      std::vector<double> probs(dec.legal_actions.size());
      for (size_t i = 0; i < dec.legal_actions.size(); ++i) {
        open_spiel::Action a = dec.legal_actions[i];
        float val = (a >= 0 && a < 2391) ? p_logits[a] : -1e9f;
        double e = std::exp(static_cast<double>(val) - max_l);
        probs[i] = e;
        sum_exp += e;
      }
      for (size_t i = 0; i < dec.legal_actions.size(); ++i) {
        probs[i] /= std::max(1e-12, sum_exp);
        if (ex.q[i] > 1e-9) {
          sum_ce += -ex.q[i] * std::log(std::max(1e-12, probs[i]));
        }
      }

      if (ex.teaching_weight_w > 0.0) {
        pos_weight_count++;
        open_spiel::Action greedy_a = dec.legal_actions[0];
        double best_l = -1e9;
        for (size_t i = 0; i < dec.legal_actions.size(); ++i) {
          if (probs[i] > best_l) {
            best_l = probs[i];
            greedy_a = dec.legal_actions[i];
          }
          if (dec.legal_actions[i] == ex.selected_move) {
            sum_target_p += probs[i];
            target_p_count++;
          }
        }
        if (greedy_a == ex.selected_move) {
          greedy_adopt_count++;
        }
      }
    }

    out_ce = s_indices.empty() ? 0.0 : (sum_ce / s_indices.size());
    out_pos_target_prob = (target_p_count > 0) ? (sum_target_p / target_p_count) : 0.0;
    out_greedy_adopt = (pos_weight_count > 0) ? (100.0 * greedy_adopt_count / pos_weight_count) : 0.0;

    double sum_kl = 0.0;
    if (eval_model == pres_ref_model || eval_model == cycle_start_model) {
      sum_kl = 0.0;
    } else {
      for (size_t p_idx : p_indices) {
        const auto& dec = all_decisions[p_idx];
        torch::Tensor obs_t = torch::from_blob(
            const_cast<float*>(dec.obs.data()), {1, static_cast<int64_t>(dec.obs.size())}, torch::kFloat32).to(device);
        auto outputs = eval_model->forward(obs_t);
        auto ref_outputs = pres_ref_model->forward(obs_t);

        std::vector<const open_spiel::dune_semantic::CandidateActionData*> cands = {&dec.cand_data};
        if (eval_model->with_semantic_scorer_ && eval_model->semantic_scorer_) {
          open_spiel::dune_semantic::ApplySemanticScorerBatch(
              eval_model->semantic_scorer_, outputs.trunk, cands, outputs.logits, device);
        }
        if (pres_ref_model->with_semantic_scorer_ && pres_ref_model->semantic_scorer_) {
          open_spiel::dune_semantic::ApplySemanticScorerBatch(
              pres_ref_model->semantic_scorer_, ref_outputs.trunk, cands, ref_outputs.logits, device);
        }

        torch::Tensor mask_t = torch::zeros({1, 2391}, torch::TensorOptions().dtype(torch::kBool).device(device));
        for (open_spiel::Action a : dec.legal_actions) {
          if (a >= 0 && a < 2391) mask_t[0][a] = true;
        }

        torch::Tensor capped = open_spiel::CenterAndCapLogitsTensor(outputs.logits, mask_t, 10.0f);
        torch::Tensor ref_capped = open_spiel::CenterAndCapLogitsTensor(ref_outputs.logits, mask_t, 10.0f);
        torch::Tensor cpu_capped = capped.to(torch::kCPU);
        torch::Tensor cpu_ref_capped = ref_capped.to(torch::kCPU);
        const float* p_logits = cpu_capped.data_ptr<float>();
        const float* r_logits = cpu_ref_capped.data_ptr<float>();

        double max_l = -1e9, max_r = -1e9;
        for (open_spiel::Action a : dec.legal_actions) {
          float val = (a >= 0 && a < 2391) ? p_logits[a] : -1e9f;
          if (val > max_l) max_l = val;
          float r_val = (a >= 0 && a < 2391) ? r_logits[a] : -1e9f;
          if (r_val > max_r) max_r = r_val;
        }
        double sum_exp_l = 0.0, sum_exp_r = 0.0;
        std::vector<double> probs(dec.legal_actions.size());
        std::vector<double> r_probs(dec.legal_actions.size());
        for (size_t i = 0; i < dec.legal_actions.size(); ++i) {
          open_spiel::Action a = dec.legal_actions[i];
          float val = (a >= 0 && a < 2391) ? p_logits[a] : -1e9f;
          probs[i] = std::exp(static_cast<double>(val) - max_l);
          sum_exp_l += probs[i];
          float r_val = (a >= 0 && a < 2391) ? r_logits[a] : -1e9f;
          r_probs[i] = std::exp(static_cast<double>(r_val) - max_r);
          sum_exp_r += r_probs[i];
        }
        for (size_t i = 0; i < dec.legal_actions.size(); ++i) {
          probs[i] /= std::max(1e-12, sum_exp_l);
          r_probs[i] /= std::max(1e-12, sum_exp_r);
          if (r_probs[i] > 1e-9) {
            sum_kl += r_probs[i] * (std::log(std::max(1e-12, r_probs[i])) - std::log(std::max(1e-12, probs[i])));
          }
        }
      }
    }

    out_kl = p_indices.empty() ? 0.0 : (sum_kl / p_indices.size());
    out_total_loss = out_ce + 0.1 * out_kl;
  };

  std::cout << "\n=================================================================\n";
  std::cout << "  Phase 4: Baseline Pre-Learning Metric Evaluation\n";
  std::cout << "=================================================================\n";

  double pre_train_loss = 0.0, pre_train_ce = 0.0, pre_train_kl = 0.0;
  double pre_train_tp = 0.0, pre_train_ga = 0.0;
  EvaluatePartition(train_search_indices, train_pres_indices, learner_model,
                    pre_train_loss, pre_train_ce, pre_train_kl, pre_train_tp, pre_train_ga);

  double pre_hold_loss = 0.0, pre_hold_ce = 0.0, pre_hold_kl = 0.0;
  double pre_hold_tp = 0.0, pre_hold_ga = 0.0;
  EvaluatePartition(hold_search_indices, hold_pres_indices, learner_model,
                    pre_hold_loss, pre_hold_ce, pre_hold_kl, pre_hold_tp, pre_hold_ga);

  std::cout << absl::StrFormat("Pre-Train:   Loss=%.4f (CE=%.4f, KL=%.4f) | P(target)=%.4f | Greedy Adopt=%.1f%%\n",
                               pre_train_loss, pre_train_ce, pre_train_kl, pre_train_tp, pre_train_ga);
  std::cout << absl::StrFormat("Pre-Holdout: Loss=%.4f (CE=%.4f, KL=%.4f) | P(target)=%.4f | Greedy Adopt=%.1f%%\n",
                               pre_hold_loss, pre_hold_ce, pre_hold_kl, pre_hold_tp, pre_hold_ga);

  // Requirement 6: Verify approximately zero preservation KL before learning
  SPIEL_CHECK_LE(pre_train_kl, 1e-4);
  SPIEL_CHECK_LE(pre_hold_kl, 1e-4);
  std::cout << "[VERIFY-PRESERVATION] Pre-learning preservation KL verified approximately zero (<= 1e-4).\n";

  // Requirement 7: Pilot parameters and dynamic S with balanced Cycle 2 replay
  int est_search_per_game = std::max(1, static_cast<int>(search_examples.size()) / games_per_cycle);
  int est_c1_searches = est_search_per_game * games_per_cycle;
  int est_steps_epoch1 = std::max(1, (est_c1_searches + 1023) / 1024);
  int est_s1 = 20 * est_steps_epoch1;
  int est_c2_searches = 2 * est_c1_searches;
  int est_steps_epoch2 = std::max(1, (est_c2_searches + 1023) / 1024);
  int est_s2 = 20 * est_steps_epoch2;
  double pilot_S = static_cast<double>(est_s1 + est_s2);
  if (planned_s > 0.0) pilot_S = planned_s;
  double ema_decay = std::pow(2.0, -11.0 / pilot_S);

  std::cout << "\n=================================================================\n";
  std::cout << "  Phase 5: Native Batched Tensor Learning Check (20 Passes)\n";
  std::cout << "=================================================================\n";
  std::cout << absl::StrFormat("[PILOT-PARAM] Dynamic S = %.0f steps (Cycle 1: %d, Cycle 2: %d), Decay = %.6f\n",
                               pilot_S, est_s1, est_s2, ema_decay);

  auto t_learn_start = std::chrono::steady_clock::now();
  learner_model->train();

  const int num_epochs = absl::GetFlag(FLAGS_epochs);
  const size_t target_batch_size = absl::GetFlag(FLAGS_batch_size);
  const bool use_balanced_replay = !c1_train_search.empty() && !c2_train_search.empty();

  size_t total_train_search = train_search_indices.size();
  size_t steps_per_epoch = std::max<size_t>(1, (total_train_search + target_batch_size - 1) / target_batch_size);
  size_t total_learning_steps = num_epochs * steps_per_epoch;

  std::cout << absl::StrFormat("[PILOT-PARAM] Training %zu search examples across %d passes (%zu steps/pass, %zu total steps)\n",
                               total_train_search, num_epochs, steps_per_epoch, total_learning_steps);
  if (use_balanced_replay) {
    std::cout << absl::StrFormat("[REPLAY] Balanced replay active: 50%% Cycle 1 (%zu examples) + 50%% Cycle 2 (%zu examples)\n",
                                 c1_train_search.size(), c2_train_search.size());
  }

  std::mt19937_64 learn_rng(base_seed + 9999);
  size_t step_counter = 0;

  for (int ep = 1; ep <= num_epochs; ++ep) {
    for (size_t s_in_ep = 0; s_in_ep < steps_per_epoch; ++s_in_ep) {
      step_counter++;
      size_t s_bs = std::min<size_t>(target_batch_size, total_train_search);
      if (s_bs == 0 || train_pres_indices.empty()) break;

      std::vector<size_t> s_batch_idx(s_bs);
      std::vector<size_t> p_batch_idx(s_bs);

      if (use_balanced_replay) {
        size_t half_s = s_bs / 2;
        std::uniform_int_distribution<size_t> dist_c1_s(0, c1_train_search.size() - 1);
        std::uniform_int_distribution<size_t> dist_c2_s(0, c2_train_search.size() - 1);
        for (size_t b = 0; b < half_s; ++b) s_batch_idx[b] = c1_train_search[dist_c1_s(learn_rng)];
        for (size_t b = half_s; b < s_bs; ++b) s_batch_idx[b] = c2_train_search[dist_c2_s(learn_rng)];

        size_t half_p = s_bs / 2;
        std::uniform_int_distribution<size_t> dist_c1_p(0, c1_train_pres.size() - 1);
        std::uniform_int_distribution<size_t> dist_c2_p(0, c2_train_pres.size() - 1);
        for (size_t b = 0; b < half_p; ++b) p_batch_idx[b] = c1_train_pres[dist_c1_p(learn_rng)];
        for (size_t b = half_p; b < s_bs; ++b) p_batch_idx[b] = c2_train_pres[dist_c2_p(learn_rng)];
      } else {
        std::uniform_int_distribution<size_t> s_dist(0, train_search_indices.size() - 1);
        for (size_t b = 0; b < s_bs; ++b) s_batch_idx[b] = train_search_indices[s_dist(learn_rng)];
        std::uniform_int_distribution<size_t> p_dist(0, train_pres_indices.size() - 1);
        for (size_t b = 0; b < s_bs; ++b) p_batch_idx[b] = train_pres_indices[p_dist(learn_rng)];
      }

      // 1. Materialize Batched Search Tensors [s_bs, 9182]
      torch::Tensor s_obs = torch::empty({static_cast<int64_t>(s_bs), 9182}, torch::kFloat32);
      float* s_obs_ptr = s_obs.data_ptr<float>();
      std::vector<const open_spiel::dune_semantic::CandidateActionData*> s_cands(s_bs);
      torch::Tensor s_mask = torch::zeros({static_cast<int64_t>(s_bs), 2391},
                                          torch::TensorOptions().dtype(torch::kBool).device(device));
      torch::Tensor s_q = torch::zeros({static_cast<int64_t>(s_bs), 2391},
                                       torch::TensorOptions().dtype(torch::kFloat32).device(device));

      for (size_t b = 0; b < s_bs; ++b) {
        const auto& ex = search_examples[s_batch_idx[b]];
        const auto& dec = all_decisions[ex.decision_store_idx];
        std::memcpy(s_obs_ptr + b * 9182, dec.obs.data(), 9182 * sizeof(float));
        s_cands[b] = &dec.cand_data;
        for (size_t i = 0; i < dec.legal_actions.size(); ++i) {
          open_spiel::Action a = dec.legal_actions[i];
          if (a >= 0 && a < 2391) {
            s_mask[b][a] = true;
            s_q[b][a] = static_cast<float>(ex.q[i]);
          }
        }
      }

      torch::Tensor s_obs_dev = s_obs.to(device);
      auto s_outputs = learner_model->forward(s_obs_dev);
      if (learner_model->with_semantic_scorer_ && learner_model->semantic_scorer_) {
        open_spiel::dune_semantic::ApplySemanticScorerBatch(
            learner_model->semantic_scorer_, s_outputs.trunk, s_cands, s_outputs.logits, device);
      }
      torch::Tensor s_capped = open_spiel::CenterAndCapLogitsTensor(s_outputs.logits, s_mask, 10.0f);
      torch::Tensor s_masked = torch::where(s_mask, s_capped, torch::full_like(s_capped, -1e9f));
      torch::Tensor s_log_p = torch::log_softmax(s_masked, -1);
      torch::Tensor ce_loss = -torch::sum(s_q * s_log_p, -1).mean();

      // 2. Materialize Batched Preservation Tensors [p_bs, 9182]
      size_t p_bs = s_bs;
      torch::Tensor p_obs = torch::empty({static_cast<int64_t>(p_bs), 9182}, torch::kFloat32);
      float* p_obs_ptr = p_obs.data_ptr<float>();
      std::vector<const open_spiel::dune_semantic::CandidateActionData*> p_cands(p_bs);
      torch::Tensor p_mask = torch::zeros({static_cast<int64_t>(p_bs), 2391},
                                          torch::TensorOptions().dtype(torch::kBool).device(device));

      for (size_t b = 0; b < p_bs; ++b) {
        const auto& dec = all_decisions[p_batch_idx[b]];
        std::memcpy(p_obs_ptr + b * 9182, dec.obs.data(), 9182 * sizeof(float));
        p_cands[b] = &dec.cand_data;
        for (size_t i = 0; i < dec.legal_actions.size(); ++i) {
          open_spiel::Action a = dec.legal_actions[i];
          if (a >= 0 && a < 2391) {
            p_mask[b][a] = true;
          }
        }
      }

      torch::Tensor p_obs_dev = p_obs.to(device);
      torch::Tensor p_ref;
      {
        torch::NoGradGuard no_grad;
        auto ref_outputs = pres_ref_model->forward(p_obs_dev);
        if (pres_ref_model->with_semantic_scorer_ && pres_ref_model->semantic_scorer_) {
          open_spiel::dune_semantic::ApplySemanticScorerBatch(
              pres_ref_model->semantic_scorer_, ref_outputs.trunk, p_cands, ref_outputs.logits, device);
        }
        torch::Tensor ref_capped = open_spiel::CenterAndCapLogitsTensor(ref_outputs.logits, p_mask, 10.0f);
        torch::Tensor ref_masked = torch::where(p_mask, ref_capped, torch::full_like(ref_capped, -1e9f));
        p_ref = torch::softmax(ref_masked, -1);
      }

      auto p_outputs = learner_model->forward(p_obs_dev);
      if (learner_model->with_semantic_scorer_ && learner_model->semantic_scorer_) {
        open_spiel::dune_semantic::ApplySemanticScorerBatch(
            learner_model->semantic_scorer_, p_outputs.trunk, p_cands, p_outputs.logits, device);
      }
      torch::Tensor p_capped = open_spiel::CenterAndCapLogitsTensor(p_outputs.logits, p_mask, 10.0f);
      torch::Tensor p_masked = torch::where(p_mask, p_capped, torch::full_like(p_capped, -1e9f));
      torch::Tensor p_log_p = torch::log_softmax(p_masked, -1);
      torch::Tensor p_ref_clamped = torch::clamp(p_ref, 1e-12, 1.0);
      torch::Tensor p_log_ref = torch::log(p_ref_clamped);
      torch::Tensor kl_loss = torch::sum(torch::where(p_mask, p_ref * (p_log_ref - p_log_p), torch::zeros_like(p_ref)), -1).mean();

      torch::Tensor total_loss = ce_loss + 0.1 * kl_loss;

      optimizer->zero_grad();
      total_loss.backward();
      torch::nn::utils::clip_grad_norm_(learner_model->parameters(), 0.5);
      optimizer->step();

      open_spiel::dune_pilot::UpdateEmaWeights(learner_model, ema_model, ema_decay);

      if (step_counter % 10 == 0 || step_counter == total_learning_steps) {
        std::cout << absl::StrFormat("[LEARN] Pass %2d/%2d (Step %d/%d): Total=%.4f (CE=%.4f, KL=%.4f, Weight=%.4f)\n",
                                     ep, num_epochs, step_counter, total_learning_steps,
                                     total_loss.item<double>(), ce_loss.item<double>(), kl_loss.item<double>(),
                                     total_teaching_weight) << std::flush;
      }
    }
  }

  auto t_learn_end = std::chrono::steady_clock::now();
  double learn_elapsed_s = std::chrono::duration<double>(t_learn_end - t_learn_start).count();
  std::cout << absl::StrFormat("[LEARN] Completed %zu batched optimizer steps across %d passes in %.2f s (%.4f s/step).\n",
                               step_counter, num_epochs, learn_elapsed_s, learn_elapsed_s / std::max<size_t>(1, step_counter));

  // Step 7: Post-Learning Metric Evaluation
  double post_train_loss = 0.0, post_train_ce = 0.0, post_train_kl = 0.0;
  double post_train_tp = 0.0, post_train_ga = 0.0;
  EvaluatePartition(train_search_indices, train_pres_indices, learner_model,
                    post_train_loss, post_train_ce, post_train_kl, post_train_tp, post_train_ga);

  double post_hold_loss = 0.0, post_hold_ce = 0.0, post_hold_kl = 0.0;
  double post_hold_tp = 0.0, post_hold_ga = 0.0;
  EvaluatePartition(hold_search_indices, hold_pres_indices, learner_model,
                    post_hold_loss, post_hold_ce, post_hold_kl, post_hold_tp, post_hold_ga);

  std::cout << "\n=================================================================\n";
  std::cout << "  Phase 6: Post-Learning Metric Evaluation & Deltas\n";
  std::cout << "=================================================================\n";
  std::cout << absl::StrFormat("Post-Train:   Loss=%.4f (CE=%.4f, KL=%.4f) | P(target)=%.4f | Greedy Adopt=%.1f%%\n",
                               post_train_loss, post_train_ce, post_train_kl, post_train_tp, post_train_ga);
  std::cout << absl::StrFormat("Post-Holdout: Loss=%.4f (CE=%.4f, KL=%.4f) | P(target)=%.4f | Greedy Adopt=%.1f%%\n",
                               post_hold_loss, post_hold_ce, post_hold_kl, post_hold_tp, post_hold_ga);

  std::cout << absl::StrFormat("Delta Train Loss:   %+.4f | Delta P(target)=%+.4f | Delta Greedy Adopt=%+.1f pp\n",
                               post_train_loss - pre_train_loss, post_train_tp - pre_train_tp, post_train_ga - pre_train_ga);
  std::cout << absl::StrFormat("Delta Holdout Loss: %+.4f | Delta P(target)=%+.4f | Delta Greedy Adopt=%+.1f pp\n",
                               post_hold_loss - pre_hold_loss, post_hold_tp - pre_hold_tp, post_hold_ga - pre_hold_ga);

  // Phase 7: Critic Immutability Verification (True Player Index)
  std::cout << "\n=================================================================\n";
  std::cout << "  Phase 7: Critic Immutability Verification (True Player Index)\n";
  std::cout << "=================================================================\n";

  double max_critic_diff = 0.0;
  for (size_t i = 0; i < critic_probe_states.size(); ++i) {
    double val_after = open_spiel::dune_pilot::EvaluateCriticDirect(
        critic_model, *(critic_probe_states[i]), critic_probe_players[i], device);
    double diff = std::abs(val_after - critic_probe_vals_before[i]);
    if (diff > max_critic_diff) max_critic_diff = diff;
  }
  std::cout << absl::StrFormat("[VERIFY-CRITIC] Probed %zu states before and after actor learning.\n",
                               critic_probe_states.size());
  std::cout << absl::StrFormat("[VERIFY-CRITIC] Maximum leaf value divergence: %.10f\n", max_critic_diff);
  if (max_critic_diff > 1e-7) {
    std::cerr << "FATAL: Critic leaf predictions changed after actor learning! Critic was contaminated.\n";
    return 1;
  }
  std::cout << "[VERIFY-CRITIC] Parity proven: Critic leaf values are 100% bit-exact and immutable.\n";

  // Phase 8: Save Checkpoints and Verify Full Reload Parity (Model, EMA, and Optimizer)
  std::cout << "\n=================================================================\n";
  std::cout << "  Phase 8: Checkpoint Save and Parity Verification\n";
  std::cout << "=================================================================\n";

  std::string ckpt_tag = (pilot_cycle == 1) ? "cycle1" : ((pilot_cycle == 2) ? "cycle2" : "preflight");
  std::string pf_model_path = out_dir + "/checkpoints/ppo_model_update_" + ckpt_tag + ".pt";
  std::string pf_opt_path = out_dir + "/checkpoints/ppo_optimizer_update_" + ckpt_tag + ".pt";
  std::string pf_ema_path = out_dir + "/checkpoints/ppo_model_ema_update_" + ckpt_tag + ".pt";
  std::string pf_manifest_path = out_dir + "/checkpoints/ppo_model_update_" + ckpt_tag + ".json";
  std::string pf_ema_manifest_path = out_dir + "/checkpoints/ppo_model_ema_update_" + ckpt_tag + ".json";

  torch::save(learner_model, pf_model_path);
  torch::save(*optimizer, pf_opt_path);
  torch::save(ema_model, pf_ema_path);

  // If preflight, also write legacy names if needed
  if (pilot_cycle == 0) {
    torch::save(learner_model, out_dir + "/checkpoints/ppo_model_update_preflight.pt");
    torch::save(ema_model, out_dir + "/checkpoints/ppo_model_ema_update_preflight.pt");
  }

  const std::string save_replay = absl::GetFlag(FLAGS_save_replay_path);
  if (!save_replay.empty()) {
    std::cout << "[REPLAY] Saving collected dataset and search examples to: " << save_replay << "...\n";
    if (!open_spiel::dune_pilot::SaveReplayData(
            save_replay, all_decisions, search_examples,
            train_search_indices, hold_search_indices,
            train_pres_indices, hold_pres_indices)) {
      std::cerr << "WARNING: Failed to save replay data to " << save_replay << "\n";
    } else {
      std::cout << "[REPLAY] Replay data saved successfully (" << all_decisions.size() << " decisions, "
                << search_examples.size() << " examples).\n";
    }
  }

  std::string pf_m_hash = open_spiel::ComputeFileSHA256(pf_model_path);
  std::string pf_opt_hash = open_spiel::ComputeFileSHA256(pf_opt_path);
  std::string pf_ema_hash = open_spiel::ComputeFileSHA256(pf_ema_path);
  size_t pf_m_sz = std::filesystem::file_size(pf_model_path);
  size_t pf_opt_sz = std::filesystem::file_size(pf_opt_path);
  size_t pf_ema_sz = std::filesystem::file_size(pf_ema_path);

  open_spiel::json::Object m_man;
  m_man["schema_version"] = 2;
  m_man["num_blocks"] = 12;
  m_man["hidden_dim"] = 2048;
  m_man["observation_dim"] = 9182;
  m_man["enable_semantic_scorer"] = true;
  m_man["enable_zone_encoder"] = false;
  m_man["semantic_descriptor_schema"] = "semantic_action_v3";
  m_man["market_appendix_mode"] = "full_public_information_v3";
  m_man["feature_schema_sha256"] = "1f91ede6656d792545589b69b39ed1209cb8255c3ba8dcae49726acde7388e7f";
  m_man["rollout_amp"] = false;
  m_man["allow_tf32"] = false;
  m_man["model_filename"] = "ppo_model_update_" + ckpt_tag + ".pt";
  m_man["model_file_size"] = static_cast<int64_t>(pf_m_sz);
  m_man["model_sha256"] = pf_m_hash;
  m_man["optimizer_filename"] = "ppo_optimizer_update_" + ckpt_tag + ".pt";
  m_man["optimizer_file_size"] = static_cast<int64_t>(pf_opt_sz);
  m_man["optimizer_sha256"] = pf_opt_hash;
  m_man["pilot_optimizer_steps"] = static_cast<int64_t>(step_counter);
  m_man["planned_pilot_steps"] = static_cast<int64_t>(pilot_S);
  {
    std::ofstream ofs(pf_manifest_path);
    ofs << open_spiel::json::ToString(m_man, true);
  }

  open_spiel::json::Object e_man = m_man;
  e_man["model_filename"] = "ppo_model_ema_update_" + ckpt_tag + ".pt";
  e_man["model_file_size"] = static_cast<int64_t>(pf_ema_sz);
  e_man["model_sha256"] = pf_ema_hash;
  e_man["ema_decay"] = ema_decay;
  e_man["pilot_optimizer_steps"] = static_cast<int64_t>(step_counter);
  e_man["planned_pilot_steps"] = static_cast<int64_t>(pilot_S);
  {
    std::ofstream ofs(pf_ema_manifest_path);
    ofs << open_spiel::json::ToString(e_man, true);
  }

  // Reload saved checkpoints and verify bit-for-bit parity
  auto reloaded_model = std::make_shared<open_spiel::SharedDunePolicyValueNetImpl>(
      9182, 2048, 2391, 12, false, false, 0, true);
  reloaded_model->semantic_descriptor_schema_ = open_spiel::dune_semantic::kDescriptorSchemaVersionV3;
  reloaded_model->market_appendix_mode_ = open_spiel::dune_imperium::MarketAppendixMode::kFullPublicInformationV3;
  open_spiel::LoadModelCheckpointRobust(reloaded_model, pf_model_path, device);

  auto reloaded_ema = std::make_shared<open_spiel::SharedDunePolicyValueNetImpl>(
      9182, 2048, 2391, 12, false, false, 0, true);
  reloaded_ema->semantic_descriptor_schema_ = open_spiel::dune_semantic::kDescriptorSchemaVersionV3;
  reloaded_ema->market_appendix_mode_ = open_spiel::dune_imperium::MarketAppendixMode::kFullPublicInformationV3;
  open_spiel::LoadModelCheckpointRobust(reloaded_ema, pf_ema_path, device);

  std::vector<torch::Tensor> r_grp0 = reloaded_model->policy_head->parameters();
  std::vector<torch::Tensor> r_grp1;
  for (auto& p : reloaded_model->input_layer->parameters()) r_grp1.push_back(p);
  for (int b = 0; b < 8 && b < static_cast<int>(reloaded_model->res_blocks.size()); ++b) {
    for (auto& p : reloaded_model->res_blocks[b]->parameters()) r_grp1.push_back(p);
  }
  for (auto& p : reloaded_model->value_head->parameters()) r_grp1.push_back(p);
  if (reloaded_model->semantic_scorer_) {
    for (auto& p : reloaded_model->semantic_scorer_->parameters()) r_grp1.push_back(p);
  }
  std::vector<torch::Tensor> r_grp2;
  for (size_t b = 8; b < reloaded_model->res_blocks.size(); ++b) {
    for (auto& p : reloaded_model->res_blocks[b]->parameters()) r_grp2.push_back(p);
  }

  std::vector<torch::optim::OptimizerParamGroup> r_param_groups = {
      torch::optim::OptimizerParamGroup(r_grp0, std::make_unique<torch::optim::AdamWOptions>(5e-6)),
      torch::optim::OptimizerParamGroup(r_grp1, std::make_unique<torch::optim::AdamWOptions>(5e-6)),
      torch::optim::OptimizerParamGroup(r_grp2, std::make_unique<torch::optim::AdamWOptions>(5e-5)),
  };

  auto reloaded_optimizer = std::make_shared<torch::optim::AdamW>(
      r_param_groups, torch::optim::AdamWOptions(5e-6).weight_decay(1e-4).eps(1e-8));

  torch::load(*reloaded_optimizer, pf_opt_path, device);

  SPIEL_CHECK_EQ(reloaded_optimizer->param_groups().size(), 3);
  double r_lr0 = static_cast<const torch::optim::AdamWOptions&>(reloaded_optimizer->param_groups()[0].options()).lr();
  double r_lr1 = static_cast<const torch::optim::AdamWOptions&>(reloaded_optimizer->param_groups()[1].options()).lr();
  double r_lr2 = static_cast<const torch::optim::AdamWOptions&>(reloaded_optimizer->param_groups()[2].options()).lr();
  SPIEL_CHECK_FLOAT_EQ(r_lr0, 5e-6);
  SPIEL_CHECK_FLOAT_EQ(r_lr1, 5e-6);
  SPIEL_CHECK_FLOAT_EQ(r_lr2, 5e-5);

  double max_m_diff = 0.0;
  for (const auto& kv : learner_model->named_parameters()) {
    auto r_p = reloaded_model->named_parameters().find(kv.key());
    if (r_p != nullptr) {
      double diff = torch::max(torch::abs(kv.value() - *r_p)).item<double>();
      if (diff > max_m_diff) max_m_diff = diff;
    }
  }

  double max_ema_diff = 0.0;
  for (const auto& kv : ema_model->named_parameters()) {
    auto r_p = reloaded_ema->named_parameters().find(kv.key());
    if (r_p != nullptr) {
      double diff = torch::max(torch::abs(kv.value() - *r_p)).item<double>();
      if (diff > max_ema_diff) max_ema_diff = diff;
    }
  }

  SPIEL_CHECK_EQ(optimizer->state().size(), reloaded_optimizer->state().size());
  SPIEL_CHECK_FALSE(reloaded_optimizer->state().empty());

  double max_exp_avg_diff = 0.0;
  double max_exp_avg_sq_diff = 0.0;
  int64_t max_step_diff = 0;

  for (size_t g = 0; g < optimizer->param_groups().size(); ++g) {
    const auto& orig_params = optimizer->param_groups()[g].params();
    const auto& relo_params = reloaded_optimizer->param_groups()[g].params();
    SPIEL_CHECK_EQ(orig_params.size(), relo_params.size());
    for (size_t p = 0; p < orig_params.size(); ++p) {
      auto orig_key = orig_params[p].unsafeGetTensorImpl();
      auto relo_key = relo_params[p].unsafeGetTensorImpl();

      auto orig_it = optimizer->state().find(orig_key);
      auto relo_it = reloaded_optimizer->state().find(relo_key);
      if (orig_it != optimizer->state().end() && relo_it != reloaded_optimizer->state().end()) {
        const auto* orig_st = dynamic_cast<const torch::optim::AdamWParamState*>(orig_it->second.get());
        const auto* relo_st = dynamic_cast<const torch::optim::AdamWParamState*>(relo_it->second.get());
        if (orig_st != nullptr && relo_st != nullptr) {
          int64_t s_diff = std::abs(orig_st->step() - relo_st->step());
          if (s_diff > max_step_diff) max_step_diff = s_diff;
          if (orig_st->exp_avg().defined() && relo_st->exp_avg().defined()) {
            double d = torch::max(torch::abs(orig_st->exp_avg() - relo_st->exp_avg())).item<double>();
            if (d > max_exp_avg_diff) max_exp_avg_diff = d;
          }
          if (orig_st->exp_avg_sq().defined() && relo_st->exp_avg_sq().defined()) {
            double d = torch::max(torch::abs(orig_st->exp_avg_sq() - relo_st->exp_avg_sq())).item<double>();
            if (d > max_exp_avg_sq_diff) max_exp_avg_sq_diff = d;
          }
        }
      }
    }
  }

  std::cout << absl::StrFormat("[VERIFY-RELOAD] Reloaded model tensor max diff: %.10f\n", max_m_diff);
  std::cout << absl::StrFormat("[VERIFY-RELOAD] Reloaded EMA tensor max diff:   %.10f\n", max_ema_diff);
  std::cout << absl::StrFormat("[VERIFY-RELOAD] Reloaded optimizer state: %zu tensors verified\n", optimizer->state().size());
  std::cout << absl::StrFormat("[VERIFY-RELOAD] Optimizer step max diff: %ld\n", max_step_diff);
  std::cout << absl::StrFormat("[VERIFY-RELOAD] Optimizer exp_avg max diff: %.10f\n", max_exp_avg_diff);
  std::cout << absl::StrFormat("[VERIFY-RELOAD] Optimizer exp_avg_sq max diff: %.10f\n", max_exp_avg_sq_diff);
  std::cout << absl::StrFormat("[VERIFY-RELOAD] Reloaded optimizer learning rates: grp0=%.2e, grp1=%.2e, grp2=%.2e\n",
                               r_lr0, r_lr1, r_lr2);
  if (max_m_diff > 1e-7 || max_ema_diff > 1e-7 || max_step_diff != 0 || max_exp_avg_diff > 1e-7 || max_exp_avg_sq_diff > 1e-7) {
    std::cerr << "FATAL: Checkpoint reload failed bitwise preservation!\n";
    return 1;
  }
  std::cout << "[VERIFY-RELOAD] Parity proven: Checkpoints, EMA, and optimizer moments/steps preserved exactly.\n";

  // Requirement 7: Accurate budget calculation including Cycle 2 balanced replay
  const double total_budget_seconds = 37551.32;
  double t_collect_per_game = collect_elapsed_s / games_per_cycle;
  double t_search_per_pos = search_elapsed_s / std::max<size_t>(1, search_examples.size());
  double p_search_per_game = static_cast<double>(search_examples.size()) / games_per_cycle;
  double t_search_per_game = p_search_per_game * t_search_per_pos;
  double sec_per_learn_step = learn_elapsed_s / std::max<size_t>(1, total_learning_steps);

  // Search max G <= 256 such that T_total(G) <= 37551.32
  int best_g = -1;
  int best_s1 = 0, best_s2 = 0;
  double best_total_time = 0.0;
  for (int g = 256; g >= 8; --g) {
    int n1 = static_cast<int>(std::round(g * p_search_per_game));
    int s1 = 20 * std::max(1, (n1 + 1023) / 1024);
    int n2_total = 2 * n1;  // Cycle 2 balanced replay
    int s2 = 20 * std::max(1, (n2_total + 1023) / 1024);
    int total_s = s1 + s2;
    double t_learn_total = total_s * sec_per_learn_step;
    double t_total = 2.0 * g * t_collect_per_game + 2.0 * g * t_search_per_game + t_learn_total + 60.0; // 60s save overhead
    if (t_total <= total_budget_seconds) {
      best_g = g;
      best_s1 = s1;
      best_s2 = s2;
      best_total_time = t_total;
      break;
    }
  }

  double pilot_planned_S = 0.0;
  double pilot_planned_decay = 0.0;
  if (best_g >= 8) {
    pilot_planned_S = static_cast<double>(best_s1 + best_s2);
    pilot_planned_decay = std::pow(2.0, -11.0 / pilot_planned_S);
  } else {
    std::cerr << "WARNING: No game count (>= 8) fits the 37,551.32-second budget!\n";
  }

  std::cout << "\n=================================================================\n";
  std::cout << "  Preflight Summary Report\n";
  std::cout << "=================================================================\n";
  std::cout << absl::StrFormat("  Games timed:                      %d\n", games_per_cycle);
  std::cout << absl::StrFormat("  Total nontrivial decisions:       %zu\n", all_decisions.size());
  std::cout << absl::StrFormat("  Total non-draft decisions:        %zu\n", total_nondraft_decisions);
  std::cout << absl::StrFormat("  Searched positions (25%%):         %zu\n", search_examples.size());
  std::cout << absl::StrFormat("  Selected concurrency config:      %s (Workers=%d, Batch=%d, Evals=%d)\n",
                               selected_search_cfg.name, selected_search_cfg.rollout_workers,
                               selected_search_cfg.search_batch_size, selected_search_cfg.evaluator_count);
  std::cout << absl::StrFormat("  Positive-weight targets (w > 0):  %d (%.2f%%)\n",
                               positive_weight_count,
                               100.0 * positive_weight_count / std::max<size_t>(1, search_examples.size()));
  std::cout << absl::StrFormat("  Total teaching weight:            %.4f (mean w = %.4f)\n",
                               total_teaching_weight,
                               total_teaching_weight / std::max<size_t>(1, search_examples.size()));
  std::cout << absl::StrFormat("  Collection elapsed time:          %.2f s (%.2f s/game)\n",
                               collect_elapsed_s, t_collect_per_game);
  if (benchmark_elapsed_s > 0.0) {
    std::cout << absl::StrFormat("  Benchmark elapsed time:           %.2f s\n", benchmark_elapsed_s);
  }
  std::cout << absl::StrFormat("  Production search elapsed time:   %.2f s (%.2f s/game, %.2f s/search)\n",
                               search_elapsed_s, t_search_per_game, t_search_per_pos);
  std::cout << absl::StrFormat("  Learning elapsed time:            %.2f s (%.4f s/step)\n",
                               learn_elapsed_s, sec_per_learn_step);
  std::cout << absl::StrFormat("  Peak local RSS memory:            %.2f MB\n", open_spiel::dune_pilot::GetCurrentMemoryUsageMB());
  std::cout << absl::StrFormat("  Proposed games per cycle:         %d\n", best_g);
  std::cout << absl::StrFormat("  Smoke test steps S:               %.0f (Decay = %.6f)\n", pilot_S, ema_decay);
  std::cout << absl::StrFormat("  Planned pilot steps S:            %.0f (Decay = %.6f)\n", pilot_planned_S, pilot_planned_decay);
  std::cout << absl::StrFormat("  Projected pilot total time:       %.2f s / %.2f s budget\n", best_total_time, total_budget_seconds);

  open_spiel::json::Object summary_json;
  summary_json["games_timed"] = games_per_cycle;
  summary_json["total_nontrivial_decisions"] = static_cast<int64_t>(all_decisions.size());
  summary_json["total_nondraft_decisions"] = static_cast<int64_t>(total_nondraft_decisions);
  summary_json["searched_positions"] = static_cast<int64_t>(search_examples.size());
  summary_json["selected_concurrency_config"] = selected_search_cfg.name;
  summary_json["selected_rollout_workers"] = selected_search_cfg.rollout_workers;
  summary_json["selected_search_batch_size"] = selected_search_cfg.search_batch_size;
  summary_json["selected_evaluator_count"] = selected_search_cfg.evaluator_count;
  summary_json["search_amp"] = selected_search_cfg.amp;
  summary_json["search_root_window"] = selected_search_cfg.EffectiveRootWindow();
  summary_json["benchmark_elapsed_seconds"] = benchmark_elapsed_s;
  summary_json["positive_weight_count"] = positive_weight_count;
  summary_json["total_teaching_weight"] = total_teaching_weight;
  summary_json["agreement_count"] = agreement_count;
  summary_json["override_count"] = override_count;
  summary_json["collection_elapsed_seconds"] = collect_elapsed_s;
  summary_json["search_elapsed_seconds"] = search_elapsed_s;
  summary_json["learning_elapsed_seconds"] = learn_elapsed_s;
  summary_json["time_per_game_seconds"] = t_collect_per_game + t_search_per_game;
  summary_json["peak_ram_mb"] = open_spiel::dune_pilot::GetCurrentMemoryUsageMB();
  summary_json["total_budget_seconds"] = total_budget_seconds;
  summary_json["proposed_games_per_cycle"] = best_g;
  summary_json["smoke_test_steps_s"] = pilot_S;
  summary_json["smoke_ema_decay"] = ema_decay;
  summary_json["planned_pilot_steps_s"] = pilot_planned_S;
  summary_json["planned_ema_decay"] = pilot_planned_decay;
  summary_json["learning_seconds_per_step"] = sec_per_learn_step;
  summary_json["projected_pilot_seconds"] = best_total_time;

  summary_json["pre_train_loss"] = pre_train_loss;
  summary_json["post_train_loss"] = post_train_loss;
  summary_json["pre_train_target_prob"] = pre_train_tp;
  summary_json["post_train_target_prob"] = post_train_tp;
  summary_json["pre_train_greedy_adoption"] = pre_train_ga;
  summary_json["post_train_greedy_adoption"] = post_train_ga;

  summary_json["pre_holdout_loss"] = pre_hold_loss;
  summary_json["post_holdout_loss"] = post_hold_loss;
  summary_json["pre_holdout_target_prob"] = pre_hold_tp;
  summary_json["post_holdout_target_prob"] = post_hold_tp;
  summary_json["pre_holdout_greedy_adoption"] = pre_hold_ga;
  summary_json["post_holdout_greedy_adoption"] = post_hold_ga;

  summary_json["max_critic_difference"] = max_critic_diff;
  summary_json["max_model_reload_difference"] = max_m_diff;
  summary_json["max_ema_reload_difference"] = max_ema_diff;
  summary_json["max_optimizer_step_difference"] = static_cast<int64_t>(max_step_diff);
  summary_json["max_optimizer_exp_avg_difference"] = max_exp_avg_diff;
  summary_json["max_optimizer_exp_avg_sq_difference"] = max_exp_avg_sq_diff;

  std::string summary_path = out_dir + "/preflight_summary.json";
  {
    std::ofstream ofs(summary_path);
    ofs << open_spiel::json::ToString(summary_json, true);
  }
  if (!ckpt_tag.empty() && ckpt_tag != "preflight") {
    std::string cycle_summary_path = out_dir + "/" + ckpt_tag + "_summary.json";
    std::ofstream ofs(cycle_summary_path);
    ofs << open_spiel::json::ToString(summary_json, true);
  }
  std::cout << "\nSaved summary to: " << summary_path << "\n";

  return 0;
}
