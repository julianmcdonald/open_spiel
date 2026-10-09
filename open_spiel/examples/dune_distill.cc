// Copyright 2026 DeepMind Technologies Limited
//
// Native C++ Knowledge Distillation Pipeline for Dune: Imperium
// Repairs chance sampling, bounds memory via chunk streaming,
// implements chunk-boundary resume, and evaluates held-out fidelity.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include <torch/torch.h>

#include "open_spiel/abseil-cpp/absl/flags/flag.h"
#include "open_spiel/abseil-cpp/absl/flags/parse.h"
#include "open_spiel/abseil-cpp/absl/strings/str_format.h"
#include "open_spiel/games/dune_imperium/dune_imperium.h"
#include "open_spiel/spiel.h"
#include "open_spiel/utils/json.h"

#include "dune_eval_action_selection.h"
#include "dune_network.h"
#include "dune_ppo_training_flags.h"
#include "dune_ppo_training_utils.h"
#include "dune_search_routing.h"
#include "dune_semantic_action_scorer.h"
#include "dune_sha256.h"

ABSL_FLAG(std::string, teacher_checkpoint,
          "/run/media/warcr/Storage/dune_drl_runtime/round7/u30650_12block_random_draft_overnight_20261006/checkpoints/ppo_model_update_31400.pt",
          "Path to teacher model checkpoint.");
ABSL_FLAG(std::string, student_checkpoint,
          "/home/warcr/dune_drl_runtime/round7/u31400_distill_streamed_4096_20261008/checkpoints/ppo_model_chunk_16.pt",
          "Path to initial student model checkpoint.");
ABSL_FLAG(std::string, student_optim_checkpoint,
          "/home/warcr/dune_drl_runtime/round7/u31400_distill_streamed_4096_20261008/checkpoints/ppo_optimizer_chunk_16.pt",
          "Path to initial student optimizer checkpoint.");
ABSL_FLAG(std::string, output_dir,
          "/home/warcr/dune_drl_runtime/round7/u31400_distill_streamed_12288_20261008",
          "Directory where distilled checkpoint bundles, progress and fidelity outputs will be saved.");
ABSL_FLAG(int, start_chunk, 16, "Initial completed chunks before this phase starts.");
ABSL_FLAG(int, total_chunks, 48, "Total cumulative chunks at completion of this phase.");
ABSL_FLAG(int, phase_chunks, 32, "Total chunks in the current learning rate phase.");
ABSL_FLAG(int, games_per_chunk, 256, "Total games per chunk (half teacher, half student).");
ABSL_FLAG(int, epochs_per_chunk, 2, "Training epochs per chunk.");
ABSL_FLAG(int, held_out_teacher_games, 40, "Fixed held-out games from teacher.");
ABSL_FLAG(int, held_out_student_games, 40, "Fixed held-out games from starting student.");
ABSL_FLAG(int, batch_size, 256, "Training and inference minibatch size.");
ABSL_FLAG(double, learning_rate, 1e-4, "Starting AdamW learning rate for student in this phase.");
ABSL_FLAG(double, learning_rate_end, 2.5e-5, "Ending AdamW learning rate for student across this phase.");
ABSL_FLAG(int, threads, 32, "Number of concurrent game simulation threads.");
ABSL_FLAG(uint64_t, seed, 42, "Base random seed.");
ABSL_FLAG(std::string, held_out_dataset_path,
          "/home/warcr/dune_drl_runtime/round7/u31400_distill_streamed_4096_20261008/held_out_dataset.bin",
          "Path to existing held-out dataset bin to reuse.");
ABSL_FLAG(uint64_t, initial_global_step, 26434, "Initial cumulative global optimization steps.");
ABSL_FLAG(double, initial_best_mean_kl, 0.068247, "Initial best combined mean KL score from previous phase.");
ABSL_FLAG(int, initial_best_chunk, 16, "Initial best chunk index from previous phase.");
ABSL_FLAG(std::string, initial_best_model,
          "/home/warcr/dune_drl_runtime/round7/u31400_distill_streamed_4096_20261008/checkpoints/ppo_model_chunk_16.pt",
          "Initial best model checkpoint to seed this phase.");
ABSL_FLAG(std::string, initial_best_optim,
          "/home/warcr/dune_drl_runtime/round7/u31400_distill_streamed_4096_20261008/checkpoints/ppo_optimizer_chunk_16.pt",
          "Initial best optimizer checkpoint to seed this phase.");
ABSL_FLAG(std::string, distillation_provenance,
          "champion_distillation_streamed_12288_20261008",
          "Distillation provenance string for manifest.");

// Link-only definitions required by dune_ppo_training_utils.cc
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
namespace {

using dune_imperium::DuneImperiumState;
using dune_imperium::GamePhase;

enum class DistillDecisionType {
  kLeaderSelection,
  kPurchase,
  kAgentPlacement,
  kCombat,
  kReveal,
  kOther
};

const char* DecisionTypeToString(DistillDecisionType dt) {
  switch (dt) {
    case DistillDecisionType::kLeaderSelection: return "leader selection";
    case DistillDecisionType::kPurchase:        return "purchase";
    case DistillDecisionType::kAgentPlacement:  return "agent placement";
    case DistillDecisionType::kCombat:          return "combat";
    case DistillDecisionType::kReveal:          return "reveal";
    case DistillDecisionType::kOther:           return "other";
  }
  return "other";
}

DistillDecisionType CategorizeDecision(
    const DuneImperiumState& state,
    const std::vector<Action>& legal_actions) {
  // 1. Leader selection:
  if (state.phase() == GamePhase::kLeaderDraft ||
      state.phase() == GamePhase::kLeaderOfferChance) {
    return DistillDecisionType::kLeaderSelection;
  }
  for (Action a : legal_actions) {
    if (a >= dune_imperium::kActionLeaderPick0 &&
        a < dune_imperium::kActionLeaderPick0 + 14) {
      return DistillDecisionType::kLeaderSelection;
    }
  }

  // 2. Purchase:
  bool has_purchase = false;
  for (Action a : legal_actions) {
    if ((a >= dune_imperium::kActionBuyImperiumRow0 && a < dune_imperium::kActionBuyImperiumRow0 + 8) ||
        (a >= dune_imperium::kActionBuyHelenaReserve0 && a < dune_imperium::kActionBuyHelenaReserve0 + 8) ||
        (a == dune_imperium::kActionBuyReserveArrakisLiaison) ||
        (a == dune_imperium::kActionBuyReserveTheSpiceMustFlow) ||
        (a >= dune_imperium::kActionTechAcquire0 && a <= dune_imperium::kActionTechAcquireSkip) ||
        (a >= dune_imperium::kActionTleilaxuAcquire0 && a <= dune_imperium::kActionTleilaxuAcquireSkip) ||
        (a >= dune_imperium::kActionTechAcquireWithSolari0 && a <= dune_imperium::kActionTechAcquireWithSolariSkip)) {
      has_purchase = true;
      break;
    }
  }
  if (has_purchase) {
    return DistillDecisionType::kPurchase;
  }

  // 3. Combat:
  if (state.phase() == GamePhase::kCombat) {
    return DistillDecisionType::kCombat;
  }
  bool has_combat = false;
  for (Action a : legal_actions) {
    if (a == dune_imperium::kActionCombatPass ||
        (a >= dune_imperium::kActionPlayIntrigueCombatCard0 &&
         a < dune_imperium::kActionPlayIntrigueCombatCard0 + dune_imperium::kMaxIntrigueCards) ||
        (a >= dune_imperium::kActionCombatCommit0 &&
         a < dune_imperium::kActionCombatCommit0 + 48)) {
      has_combat = true;
      break;
    }
  }
  if (has_combat) {
    return DistillDecisionType::kCombat;
  }

  // 4. Agent Placement:
  if (state.phase() == GamePhase::kAgentTurns) {
    return DistillDecisionType::kAgentPlacement;
  }
  for (Action a : legal_actions) {
    if ((a >= dune_imperium::kActionAgentSpaceConspire &&
         a <= dune_imperium::kActionAgentSpaceSietchTabr) ||
        (a == dune_imperium::kActionPlayKwisatzHaderach)) {
      return DistillDecisionType::kAgentPlacement;
    }
  }

  // 5. Reveal:
  if (state.phase() == GamePhase::kRevealTurns) {
    return DistillDecisionType::kReveal;
  }
  for (Action a : legal_actions) {
    if (a == dune_imperium::kActionReveal) {
      return DistillDecisionType::kReveal;
    }
  }

  // 6. Other:
  return DistillDecisionType::kOther;
}

struct StateTransition {
  int game_id = 0;
  int decision_index = 0;
  Player player = 0;
  DistillDecisionType dec_type = DistillDecisionType::kOther;
  bool is_single_action = false;
  int source_group = 0; // 0 = teacher-generated, 1 = student-generated
  std::vector<float> obs;
  std::vector<Action> legal_actions;
  dune_semantic::CandidateActionData cand_data;

  // Teacher targets:
  std::vector<float> teacher_legal_logits;
  float teacher_value = 0.0f;
};

void SerializeCandidateActionData(std::ostream& os, const dune_semantic::CandidateActionData& cad) {
  uint32_t n = static_cast<uint32_t>(cad.actions.size());
  os.write(reinterpret_cast<const char*>(&n), sizeof(n));
  if (n > 0) {
    os.write(reinterpret_cast<const char*>(cad.actions.data()), n * sizeof(Action));
    uint32_t n_feat = static_cast<uint32_t>(cad.features.size());
    os.write(reinterpret_cast<const char*>(&n_feat), sizeof(n_feat));
    if (n_feat > 0) {
      os.write(reinterpret_cast<const char*>(cad.features.data()), n_feat * sizeof(float));
    }
    os.write(reinterpret_cast<const char*>(cad.card_ids.data()), n * sizeof(int64_t));
    os.write(reinterpret_cast<const char*>(cad.space_ids.data()), n * sizeof(int64_t));
    os.write(reinterpret_cast<const char*>(cad.supported.data()), n * sizeof(uint8_t));
    os.write(reinterpret_cast<const char*>(cad.roles.data()), n * sizeof(dune_semantic::ActionRole));
  }
}

void DeserializeCandidateActionData(std::istream& is, dune_semantic::CandidateActionData& cad) {
  uint32_t n = 0;
  is.read(reinterpret_cast<char*>(&n), sizeof(n));
  if (n > 0) {
    cad.actions.resize(n);
    is.read(reinterpret_cast<char*>(cad.actions.data()), n * sizeof(Action));
    uint32_t n_feat = 0;
    is.read(reinterpret_cast<char*>(&n_feat), sizeof(n_feat));
    if (n_feat > 0) {
      cad.features.resize(n_feat);
      is.read(reinterpret_cast<char*>(cad.features.data()), n_feat * sizeof(float));
    }
    cad.card_ids.resize(n);
    is.read(reinterpret_cast<char*>(cad.card_ids.data()), n * sizeof(int64_t));
    cad.space_ids.resize(n);
    is.read(reinterpret_cast<char*>(cad.space_ids.data()), n * sizeof(int64_t));
    cad.supported.resize(n);
    is.read(reinterpret_cast<char*>(cad.supported.data()), n * sizeof(uint8_t));
    cad.roles.resize(n);
    is.read(reinterpret_cast<char*>(cad.roles.data()), n * sizeof(dune_semantic::ActionRole));
  }
}

void SaveHeldOutDataset(const std::string& path, const std::vector<StateTransition>& data) {
  std::string tmp_path = path + ".tmp";
  std::ofstream os(tmp_path, std::ios::binary);
  uint64_t count = data.size();
  os.write(reinterpret_cast<const char*>(&count), sizeof(count));
  for (const auto& t : data) {
    os.write(reinterpret_cast<const char*>(&t.game_id), sizeof(t.game_id));
    os.write(reinterpret_cast<const char*>(&t.decision_index), sizeof(t.decision_index));
    os.write(reinterpret_cast<const char*>(&t.player), sizeof(t.player));
    int dec = static_cast<int>(t.dec_type);
    os.write(reinterpret_cast<const char*>(&dec), sizeof(dec));
    os.write(reinterpret_cast<const char*>(&t.is_single_action), sizeof(t.is_single_action));
    os.write(reinterpret_cast<const char*>(&t.source_group), sizeof(t.source_group));
    os.write(reinterpret_cast<const char*>(&t.teacher_value), sizeof(t.teacher_value));

    uint32_t obs_sz = static_cast<uint32_t>(t.obs.size());
    os.write(reinterpret_cast<const char*>(&obs_sz), sizeof(obs_sz));
    os.write(reinterpret_cast<const char*>(t.obs.data()), obs_sz * sizeof(float));

    uint32_t legals_sz = static_cast<uint32_t>(t.legal_actions.size());
    os.write(reinterpret_cast<const char*>(&legals_sz), sizeof(legals_sz));
    os.write(reinterpret_cast<const char*>(t.legal_actions.data()), legals_sz * sizeof(Action));

    uint32_t logits_sz = static_cast<uint32_t>(t.teacher_legal_logits.size());
    os.write(reinterpret_cast<const char*>(&logits_sz), sizeof(logits_sz));
    os.write(reinterpret_cast<const char*>(t.teacher_legal_logits.data()), logits_sz * sizeof(float));

    SerializeCandidateActionData(os, t.cand_data);
  }
  os.close();
  std::filesystem::rename(tmp_path, path);
  std::cout << "[INFO] Saved " << count << " held-out transitions to " << path << "\n";
}

bool LoadHeldOutDataset(const std::string& path, std::vector<StateTransition>& data) {
  if (!std::filesystem::exists(path)) return false;
  std::ifstream is(path, std::ios::binary);
  if (!is) return false;
  uint64_t count = 0;
  is.read(reinterpret_cast<char*>(&count), sizeof(count));
  data.clear();
  data.reserve(count);
  for (uint64_t i = 0; i < count; ++i) {
    StateTransition t;
    is.read(reinterpret_cast<char*>(&t.game_id), sizeof(t.game_id));
    is.read(reinterpret_cast<char*>(&t.decision_index), sizeof(t.decision_index));
    is.read(reinterpret_cast<char*>(&t.player), sizeof(t.player));
    int dec = 0;
    is.read(reinterpret_cast<char*>(&dec), sizeof(dec));
    t.dec_type = static_cast<DistillDecisionType>(dec);
    is.read(reinterpret_cast<char*>(&t.is_single_action), sizeof(t.is_single_action));
    is.read(reinterpret_cast<char*>(&t.source_group), sizeof(t.source_group));
    is.read(reinterpret_cast<char*>(&t.teacher_value), sizeof(t.teacher_value));

    uint32_t obs_sz = 0;
    is.read(reinterpret_cast<char*>(&obs_sz), sizeof(obs_sz));
    t.obs.resize(obs_sz);
    is.read(reinterpret_cast<char*>(t.obs.data()), obs_sz * sizeof(float));

    uint32_t legals_sz = 0;
    is.read(reinterpret_cast<char*>(&legals_sz), sizeof(legals_sz));
    t.legal_actions.resize(legals_sz);
    is.read(reinterpret_cast<char*>(t.legal_actions.data()), legals_sz * sizeof(Action));

    uint32_t logits_sz = 0;
    is.read(reinterpret_cast<char*>(&logits_sz), sizeof(logits_sz));
    t.teacher_legal_logits.resize(logits_sz);
    is.read(reinterpret_cast<char*>(t.teacher_legal_logits.data()), logits_sz * sizeof(float));

    DeserializeCandidateActionData(is, t.cand_data);
    data.push_back(std::move(t));
  }
  std::cout << "[INFO] Loaded " << data.size() << " held-out transitions from cache: " << path << "\n";
  return true;
}

// Verification function testing chance outcome sampling on a nonuniform chance node
void VerifyNonuniformChanceSampling(const Game& game) {
  std::cout << "\n=================================================================\n";
  std::cout << "  Verification: Chance Outcome Sampling on Nonuniform State\n";
  std::cout << "=================================================================\n";

  std::unique_ptr<State> state = game.NewInitialState();
  SPIEL_CHECK_TRUE(state->IsChanceNode());
  auto outcomes = state->ChanceOutcomes();
  SPIEL_CHECK_GT(outcomes.size(), 1);

  double min_p = 1.0;
  double max_p = 0.0;
  Action min_act = -1, max_act = -1;
  for (const auto& [act, p] : outcomes) {
    if (p < min_p) { min_p = p; min_act = act; }
    if (p > max_p) { max_p = p; max_act = act; }
  }

  std::cout << absl::StrFormat(
      "[VERIFY-CHANCE] Initial game chance node has %zu outcomes.\n"
      "[VERIFY-CHANCE] Nonuniformity probe: min_p=%.6f (action %d), max_p=%.6f (action %d), ratio=%.2f\n",
      outcomes.size(), min_p, min_act, max_p, max_act, max_p / min_p);
  SPIEL_CHECK_GT(max_p / min_p, 1.5);

  const int num_samples = 100000;
  std::mt19937_64 rng(999);
  std::map<Action, int> counts;

  for (int s = 0; s < num_samples; ++s) {
    Action sampled = open_spiel::SampleAction(outcomes, rng).first;
    counts[sampled]++;
  }

  double obs_min_p = static_cast<double>(counts[min_act]) / num_samples;
  double obs_max_p = static_cast<double>(counts[max_act]) / num_samples;
  double z_min = std::abs(obs_min_p - min_p) / std::sqrt(min_p * (1.0 - min_p) / num_samples);
  double z_max = std::abs(obs_max_p - max_p) / std::sqrt(max_p * (1.0 - max_p) / num_samples);

  double chi_sq = 0.0;
  for (const auto& [act, p] : outcomes) {
    double exp_cnt = num_samples * p;
    double obs_cnt = counts[act];
    chi_sq += (obs_cnt - exp_cnt) * (obs_cnt - exp_cnt) / exp_cnt;
  }

  std::cout << absl::StrFormat(
      "[VERIFY-CHANCE] %d samples evaluated via open_spiel::SampleAction():\n"
      "  Action %3d (min prob): expected=%.6f, observed=%.6f (z=%.2f)\n"
      "  Action %3d (max prob): expected=%.6f, observed=%.6f (z=%.2f)\n"
      "  Chi-Square statistic: %.2f (df=%zu)\n",
      num_samples, min_act, min_p, obs_min_p, z_min,
      max_act, max_p, obs_max_p, z_max, chi_sq, outcomes.size() - 1);

  SPIEL_CHECK_LT(z_min, 3.5);
  SPIEL_CHECK_LT(z_max, 3.5);

  // Counterfactual comparison against uniform sampling (the prior defect)
  double uniform_p = 1.0 / outcomes.size();
  double z_uniform_defect = std::abs(uniform_p - max_p) / std::sqrt(max_p * (1.0 - max_p) / num_samples);
  std::cout << absl::StrFormat(
      "[VERIFY-CHANCE] Counterfactual uniform sampler (prior defect):\n"
      "  Action %3d: true=%.6f vs uniform=%.6f (z=%.2f => defect refuted)\n",
      max_act, max_p, uniform_p, z_uniform_defect);
  SPIEL_CHECK_GT(z_uniform_defect, 10.0);

  std::cout << "[VERIFY-CHANCE] Chance outcome sampling test passed.\n\n";
}

// Batch inference helper running the exact production forward path on model
void RunProductionForwardBatch(
    std::shared_ptr<SharedDunePolicyValueNetImpl> model,
    const std::vector<const std::vector<float>*>& obs_ptrs,
    const std::vector<const std::vector<Action>*>& legal_ptrs,
    const std::vector<const dune_semantic::CandidateActionData*>& cand_ptrs,
    torch::Device device,
    std::vector<std::vector<float>>& out_legal_logits,
    std::vector<float>& out_values) {
  const size_t batch_size = obs_ptrs.size();
  if (batch_size == 0) return;

  const int64_t obs_dim = 9182;
  const int64_t action_dim = 2391;

  torch::Tensor batch_states = torch::empty(
      {static_cast<int64_t>(batch_size), obs_dim},
      torch::TensorOptions().dtype(torch::kFloat32).device(device));
  torch::Tensor batch_masks = torch::zeros(
      {static_cast<int64_t>(batch_size), action_dim},
      torch::TensorOptions().dtype(torch::kBool).device(device));

  std::vector<float> flat_obs(batch_size * obs_dim, 0.0f);
  std::vector<uint8_t> flat_masks(batch_size * action_dim, 0);

  for (size_t b = 0; b < batch_size; ++b) {
    std::memcpy(&flat_obs[b * obs_dim], obs_ptrs[b]->data(), obs_dim * sizeof(float));
    for (Action a : *(legal_ptrs[b])) {
      if (a >= 0 && a < action_dim) {
        flat_masks[b * action_dim + a] = 1;
      }
    }
  }

  torch::Tensor cpu_states = torch::from_blob(
      flat_obs.data(), {static_cast<int64_t>(batch_size), obs_dim}, torch::kFloat32);
  torch::Tensor cpu_masks = torch::from_blob(
      flat_masks.data(), {static_cast<int64_t>(batch_size), action_dim}, torch::kUInt8).to(torch::kBool);

  batch_states.copy_(cpu_states, /*non_blocking=*/false);
  batch_masks.copy_(cpu_masks, /*non_blocking=*/false);

  torch::NoGradGuard no_grad;
  auto outputs = model->forward(batch_states);
  if (model->with_semantic_scorer_ && model->semantic_scorer_) {
    dune_semantic::ApplySemanticScorerBatch(
        model->semantic_scorer_, outputs.trunk, cand_ptrs, outputs.logits, device);
  }

  torch::Tensor capped_logits = CenterAndCapLogitsTensor(outputs.logits, batch_masks, 10.0f);

  torch::Tensor cpu_capped = capped_logits.to(torch::kCPU);
  torch::Tensor cpu_vals = outputs.values.to(torch::kCPU);

  const float* p_logits = cpu_capped.data_ptr<float>();
  const float* p_vals = cpu_vals.data_ptr<float>();

  out_legal_logits.resize(batch_size);
  out_values.resize(batch_size);

  for (size_t b = 0; b < batch_size; ++b) {
    out_values[b] = p_vals[b];
    const auto& legals = *(legal_ptrs[b]);
    out_legal_logits[b].resize(legals.size());
    for (size_t i = 0; i < legals.size(); ++i) {
      Action a = legals[i];
      out_legal_logits[b][i] = (a >= 0 && a < action_dim) ? p_logits[b * action_dim + a] : -1e9f;
    }
  }
}

// Label all transitions in a dataset with the teacher network using production forward path
void LabelTransitionsWithTeacher(
    std::shared_ptr<SharedDunePolicyValueNetImpl> teacher,
    std::vector<StateTransition>& transitions,
    torch::Device device,
    int batch_size = 256) {
  const size_t total = transitions.size();
  teacher->eval();

  for (size_t i = 0; i < total; i += batch_size) {
    size_t end = std::min(total, i + batch_size);
    std::vector<const std::vector<float>*> obs_ptrs;
    std::vector<const std::vector<Action>*> legal_ptrs;
    std::vector<const dune_semantic::CandidateActionData*> cand_ptrs;
    for (size_t j = i; j < end; ++j) {
      obs_ptrs.push_back(&transitions[j].obs);
      legal_ptrs.push_back(&transitions[j].legal_actions);
      cand_ptrs.push_back(&transitions[j].cand_data);
    }

    std::vector<std::vector<float>> batch_legal_logits;
    std::vector<float> batch_values;
    RunProductionForwardBatch(
        teacher, obs_ptrs, legal_ptrs, cand_ptrs, device, batch_legal_logits, batch_values);

    for (size_t j = i; j < end; ++j) {
      size_t b = j - i;
      transitions[j].teacher_legal_logits = std::move(batch_legal_logits[b]);
      transitions[j].teacher_value = batch_values[b];
    }
  }
}

// Rollout collection for games using correct chance outcome probabilities
std::vector<StateTransition> CollectRolloutGames(
    const Game& game,
    std::shared_ptr<IGameEvaluator> acting_evaluator,
    int num_games,
    int num_threads,
    uint64_t base_seed,
    int game_id_offset,
    int source_group) {
  std::vector<StateTransition> all_transitions;
  std::mutex collect_mutex;
  std::atomic<int> next_game_idx{0};

  auto worker = [&](int thread_id) {
    while (true) {
      int local_idx = next_game_idx.fetch_add(1);
      if (local_idx >= num_games) break;

      int game_id = game_id_offset + local_idx;
      uint64_t game_seed = dune_seed::DeriveSeed(base_seed, dune_seed::kDomainTrain, game_id, 0);
      auto chance_rng = dune_seed::MakeRng64(dune_seed::DeriveSeed(game_seed, 0, 0, dune_seed::kStreamChance));
      std::mt19937_64 policy_rng[4];
      for (int p = 0; p < 4; ++p) {
        policy_rng[p] = dune_seed::MakeRng64(
            dune_seed::DeriveSeed(game_seed, 0, 0, dune_seed::kStreamPolicyPlayer0 + p));
      }

      std::unique_ptr<State> state = game.NewInitialState();
      auto* dune_state = dynamic_cast<DuneImperiumState*>(state.get());

      std::vector<StateTransition> game_trans;
      int decision_counter = 0;

      while (!state->IsTerminal()) {
        if (state->IsChanceNode()) {
          std::vector<std::pair<Action, double>> outcomes = state->ChanceOutcomes();
          Action action = game.GetType().chance_mode == GameType::ChanceMode::kSampledStochastic
                      ? outcomes.front().first
                      : open_spiel::SampleAction(outcomes, chance_rng).first;
          state->ApplyAction(action);
          continue;
        }

        if (state->IsSimultaneousNode()) {
          std::vector<Action> joint_action;
          for (int p = 0; p < game.NumPlayers(); ++p) {
            std::vector<Action> actions = state->LegalActions(p);
            if (actions.empty()) {
              joint_action.push_back(0);
            } else {
              std::uniform_int_distribution<int> dis(0, actions.size() - 1);
              joint_action.push_back(actions[dis(policy_rng[p])]);
            }
          }
          state->ApplyActions(joint_action);
          continue;
        }

        Player current_player = state->CurrentPlayer();
        std::vector<Action> legal_actions = state->LegalActions();
        if (legal_actions.empty()) break;

        StateTransition trans;
        trans.game_id = game_id;
        trans.decision_index = decision_counter++;
        trans.player = current_player;
        trans.legal_actions = legal_actions;
        trans.is_single_action = (legal_actions.size() <= 1);
        trans.source_group = source_group;
        trans.dec_type = dune_state != nullptr
            ? CategorizeDecision(*dune_state, legal_actions)
            : DistillDecisionType::kOther;

        trans.obs.assign(9182, 0.0f);
        if (dune_state != nullptr) {
          dune_state->InformationStateTensorWithAppendix(
              current_player,
              dune_imperium::MarketAppendixMode::kFullPublicInformationV3,
              absl::MakeSpan(trans.obs));
          dune_semantic::ExtractCandidateDescriptors(
              *dune_state, legal_actions, &trans.cand_data,
              dune_semantic::kDescriptorSchemaVersionV3);
        }

        EvalResult res = acting_evaluator->EvaluateWithActions(trans.obs, &trans.cand_data);
        CenterAndCapLegalLogits(res.logits, legal_actions, 10.0f);

        Action chosen_action;
        if (dune_state != nullptr &&
            dune_state->phase() == GamePhase::kLeaderDraft) {
          // Uniform random legal leader draft during collection
          std::uniform_int_distribution<size_t> dist(0, legal_actions.size() - 1);
          chosen_action = legal_actions[dist(policy_rng[current_player])];
        } else {
          // Sampled policy actions at temperature 1.0
          chosen_action = dune_eval::SelectActionFromLogits(
              res.logits, legal_actions,
              dune_eval::SelectionPolicy{/*greedy=*/false, /*temperature=*/1.0f},
              policy_rng[current_player]);
        }

        game_trans.push_back(std::move(trans));
        state->ApplyAction(chosen_action);
      }

      std::lock_guard<std::mutex> lock(collect_mutex);
      all_transitions.insert(all_transitions.end(),
                             std::make_move_iterator(game_trans.begin()),
                             std::make_move_iterator(game_trans.end()));
    }
  };

  std::vector<std::thread> workers;
  for (int t = 0; t < num_threads; ++t) {
    workers.emplace_back(worker, t);
  }
  for (auto& w : workers) {
    w.join();
  }

  return all_transitions;
}

// Structure holding fidelity metrics for a decision category
struct FidelityStats {
  std::string name;
  int64_t count = 0;
  int64_t top1_agreed = 0;
  double kl_sum = 0.0;
  std::vector<double> kl_list;
  std::vector<double> teacher_vals;
  std::vector<double> student_vals;

  FidelityStats() = default;
  explicit FidelityStats(const std::string& n) : name(n) {}
};

// Compute fidelity metrics on a partition of held-out data
json::Object EvaluateFidelityPartition(
    std::shared_ptr<SharedDunePolicyValueNetImpl> student,
    const std::vector<StateTransition>& held_out_data,
    int target_source_group,
    torch::Device device,
    const std::string& source_label,
    std::ostream& out_stream,
    double* out_total_nontrivial_kl,
    int64_t* out_total_nontrivial_count) {
  student->eval();

  std::vector<const StateTransition*> subset;
  for (const auto& t : held_out_data) {
    if (t.source_group == target_source_group) {
      subset.push_back(&t);
    }
  }

  const size_t total = subset.size();
  const size_t batch_size = 256;
  std::vector<std::vector<float>> student_legal_logits;
  std::vector<float> student_values;
  student_legal_logits.reserve(total);
  student_values.reserve(total);

  for (size_t i = 0; i < total; i += batch_size) {
    size_t end = std::min(total, i + batch_size);
    std::vector<const std::vector<float>*> obs_ptrs;
    std::vector<const std::vector<Action>*> legal_ptrs;
    std::vector<const dune_semantic::CandidateActionData*> cand_ptrs;
    for (size_t j = i; j < end; ++j) {
      obs_ptrs.push_back(&subset[j]->obs);
      legal_ptrs.push_back(&subset[j]->legal_actions);
      cand_ptrs.push_back(&subset[j]->cand_data);
    }

    std::vector<std::vector<float>> mb_logits;
    std::vector<float> mb_vals;
    RunProductionForwardBatch(
        student, obs_ptrs, legal_ptrs, cand_ptrs, device, mb_logits, mb_vals);

    for (size_t b = 0; b < mb_logits.size(); ++b) {
      student_legal_logits.push_back(std::move(mb_logits[b]));
      student_values.push_back(mb_vals[b]);
    }
  }

  std::map<DistillDecisionType, FidelityStats> type_map;
  type_map[DistillDecisionType::kLeaderSelection] = FidelityStats{"leader selection"};
  type_map[DistillDecisionType::kPurchase]        = FidelityStats{"purchase"};
  type_map[DistillDecisionType::kAgentPlacement]  = FidelityStats{"agent placement"};
  type_map[DistillDecisionType::kCombat]          = FidelityStats{"combat"};
  type_map[DistillDecisionType::kReveal]          = FidelityStats{"reveal"};
  type_map[DistillDecisionType::kOther]           = FidelityStats{"other"};

  int64_t single_action_count = 0;
  double total_kl = 0.0;
  int64_t total_nontrivial = 0;

  for (size_t i = 0; i < total; ++i) {
    const auto& trans = *subset[i];
    const size_t k = trans.legal_actions.size();

    if (k <= 1 || trans.is_single_action) {
      ++single_action_count;
      continue;
    }

    auto& stats = type_map[trans.dec_type];
    const auto& t_logits = trans.teacher_legal_logits;
    const auto& s_logits = student_legal_logits[i];
    float t_val = trans.teacher_value;
    float s_val = student_values[i];

    int best_t = 0;
    float max_t = t_logits[0];
    for (size_t j = 1; j < k; ++j) {
      if (t_logits[j] > max_t) { max_t = t_logits[j]; best_t = static_cast<int>(j); }
    }

    int best_s = 0;
    float max_s = s_logits[0];
    for (size_t j = 1; j < k; ++j) {
      if (s_logits[j] > max_s) { max_s = s_logits[j]; best_s = static_cast<int>(j); }
    }

    if (best_t == best_s) {
      ++stats.top1_agreed;
    }

    double max_t_l = -1e30, max_s_l = -1e30;
    for (float v : t_logits) if (v > max_t_l) max_t_l = v;
    for (float v : s_logits) if (v > max_s_l) max_s_l = v;

    double sum_t = 0.0, sum_s = 0.0;
    for (float v : t_logits) sum_t += std::exp(v - max_t_l);
    for (float v : s_logits) sum_s += std::exp(v - max_s_l);

    double log_z_t = max_t_l + std::log(sum_t);
    double log_z_s = max_s_l + std::log(sum_s);

    double kl = 0.0;
    for (size_t j = 0; j < k; ++j) {
      double p_t = std::exp(t_logits[j] - log_z_t);
      double log_p_t = t_logits[j] - log_z_t;
      double log_p_s = s_logits[j] - log_z_s;
      kl += p_t * (log_p_t - log_p_s);
    }
    if (kl < 0.0) kl = 0.0;

    stats.count++;
    stats.kl_sum += kl;
    stats.kl_list.push_back(kl);
    stats.teacher_vals.push_back(t_val);
    stats.student_vals.push_back(s_val);

    total_kl += kl;
    total_nontrivial++;
  }

  if (out_total_nontrivial_kl) *out_total_nontrivial_kl = total_kl;
  if (out_total_nontrivial_count) *out_total_nontrivial_count = total_nontrivial;

  out_stream << "\n### Fidelity Table: " << source_label << "\n\n";
  out_stream << "| Decision Type    | Decisions | Top-1 Agr (%) | Mean KL | 95th% KL | 99th% KL | Val Corr (r) | Val MSE  |\n";
  out_stream << "|:-----------------|----------:|--------------:|--------:|---------:|---------:|-------------:|---------:|\n";

  json::Object json_summary;
  json_summary["source"] = source_label;
  json_summary["single_legal_action_count"] = single_action_count;
  json_summary["total_nontrivial_decisions"] = total_nontrivial;
  json_summary["total_nontrivial_kl_sum"] = total_kl;
  json_summary["overall_mean_kl"] = total_nontrivial > 0 ? (total_kl / total_nontrivial) : 0.0;
  json::Object categories_obj;

  std::vector<DistillDecisionType> ordered_types = {
      DistillDecisionType::kLeaderSelection,
      DistillDecisionType::kPurchase,
      DistillDecisionType::kAgentPlacement,
      DistillDecisionType::kCombat,
      DistillDecisionType::kReveal,
      DistillDecisionType::kOther};

  for (auto dt : ordered_types) {
    auto& st = type_map[dt];
    json::Object cat_obj;
    cat_obj["decision_count"] = st.count;

    if (st.count == 0) {
      out_stream << absl::StrFormat(
          "| %-16s | %9d | %13s | %7s | %8s | %8s | %12s | %8s |\n",
          st.name.c_str(), 0, "N/A", "N/A", "N/A", "N/A", "N/A", "N/A");
      categories_obj[st.name] = cat_obj;
      continue;
    }

    double top1_pct = 100.0 * st.top1_agreed / st.count;
    double mean_kl = st.kl_sum / st.count;

    std::sort(st.kl_list.begin(), st.kl_list.end());
    size_t idx95 = static_cast<size_t>(0.95 * (st.count - 1));
    size_t idx99 = static_cast<size_t>(0.99 * (st.count - 1));
    double p95_kl = st.kl_list[idx95];
    double p99_kl = st.kl_list[idx99];

    double mean_t = std::accumulate(st.teacher_vals.begin(), st.teacher_vals.end(), 0.0) / st.count;
    double mean_s = std::accumulate(st.student_vals.begin(), st.student_vals.end(), 0.0) / st.count;
    double var_t = 0.0, var_s = 0.0, cov_ts = 0.0, mse = 0.0;
    for (size_t j = 0; j < static_cast<size_t>(st.count); ++j) {
      double dt_v = st.teacher_vals[j] - mean_t;
      double ds_v = st.student_vals[j] - mean_s;
      var_t += dt_v * dt_v;
      var_s += ds_v * ds_v;
      cov_ts += dt_v * ds_v;
      double diff = st.student_vals[j] - st.teacher_vals[j];
      mse += diff * diff;
    }
    mse /= st.count;
    double std_t = std::sqrt(var_t / st.count);
    double std_s = std::sqrt(var_s / st.count);
    double corr = (std_t > 1e-6 && std_s > 1e-6) ? (cov_ts / st.count) / (std_t * std_s) : 0.0;

    cat_obj["top1_agreement_pct"] = top1_pct;
    cat_obj["mean_kl"] = mean_kl;
    cat_obj["p95_kl"] = p95_kl;
    cat_obj["p99_kl"] = p99_kl;
    cat_obj["value_correlation"] = corr;
    cat_obj["value_mse"] = mse;
    categories_obj[st.name] = cat_obj;

    out_stream << absl::StrFormat(
        "| %-16s | %9d | %12.2f%% | %7.4f | %8.4f | %8.4f | %12.4f | %8.5f |\n",
        st.name.c_str(), st.count, top1_pct, mean_kl, p95_kl, p99_kl, corr, mse);
  }

  out_stream << "\nSingle-legal-action states excluded from agreement and KL: "
             << single_action_count << " (Total evaluated: " << total << ")\n\n";

  json_summary["categories"] = categories_obj;
  return json_summary;
}

// Write the complete checkpoint bundle (model, optimizer, manifest) in trainer's format
void SaveDistillCheckpointBundle(
    std::shared_ptr<SharedDunePolicyValueNetImpl> model,
    torch::optim::AdamW& optimizer,
    const std::string& checkpoints_dir,
    const std::string& bundle_tag,
    int global_update,
    uint64_t total_steps,
    uint64_t total_games,
    double combined_held_out_mean_kl,
    const std::string& teacher_checkpoint_path,
    const std::string& distillation_provenance) {
  std::filesystem::create_directories(checkpoints_dir);

  std::string model_filename = absl::StrFormat("ppo_model_%s.pt", bundle_tag.c_str());
  std::string optim_filename = absl::StrFormat("ppo_optimizer_%s.pt", bundle_tag.c_str());
  std::string manifest_filename = absl::StrFormat("ppo_model_%s.json", bundle_tag.c_str());

  std::string model_path = checkpoints_dir + "/" + model_filename;
  std::string optim_path = checkpoints_dir + "/" + optim_filename;
  std::string manifest_path = checkpoints_dir + "/" + manifest_filename;

  std::string model_tmp = model_path + ".tmp";
  std::string optim_tmp = optim_path + ".tmp";
  std::string manifest_tmp = manifest_path + ".tmp";

  torch::save(model, model_tmp);
  torch::save(optimizer, optim_tmp);

  size_t model_size = 0;
  std::string model_hash = open_spiel::ComputeFileSHA256(model_tmp, &model_size);

  size_t optim_size = 0;
  std::string optim_hash = open_spiel::ComputeFileSHA256(optim_tmp, &optim_size);

  std::string schema_label = "dune_imperium_full_public_information_v3|prefix=ordered_card_slots_v2:6255|opp_revealed=3x126|consumed_conflicts=22|ordered_topdeck=4x4x127|water_overflow=4|pending_graft=127|pending_hundro=2x63|pending_tech=4|pending_ceremony=2x63|pending_intrigue=108|size=9182";
  std::string schema_sha256 = "1f91ede6656d792545589b69b39ed1209cb8255c3ba8dcae49726acde7388e7f";

  json::Object manifest_obj;
  manifest_obj["schema_version"] = static_cast<int64_t>(2);
  manifest_obj["checkpoint_uuid"] = open_spiel::GenerateUUID();
  manifest_obj["global_update"] = static_cast<int64_t>(global_update);
  manifest_obj["target_end_update"] = static_cast<int64_t>(global_update);
  manifest_obj["total_env_steps"] = static_cast<int64_t>(total_steps);
  manifest_obj["next_episode_id"] = static_cast<int64_t>(total_games);
  manifest_obj["base_seed"] = static_cast<int64_t>(11);
  manifest_obj["seed_scheme_version"] = static_cast<int64_t>(2);
  manifest_obj["distillation_provenance"] = distillation_provenance;
  manifest_obj["teacher_source"] = teacher_checkpoint_path;
  manifest_obj["combined_held_out_mean_kl"] = combined_held_out_mean_kl;
  manifest_obj["rollout_amp"] = false;
  manifest_obj["allow_tf32"] = false;
  manifest_obj["search_label_fingerprint"] = "";
  manifest_obj["run_uuid"] = open_spiel::GenerateUUID();
  manifest_obj["model_filename"] = model_filename;
  manifest_obj["model_file_size"] = static_cast<int64_t>(model_size);
  manifest_obj["model_sha256"] = model_hash;
  manifest_obj["optimizer_filename"] = optim_filename;
  manifest_obj["optimizer_file_size"] = static_cast<int64_t>(optim_size);
  manifest_obj["optimizer_sha256"] = optim_hash;
  manifest_obj["hidden_dim"] = static_cast<int64_t>(2048);
  manifest_obj["num_blocks"] = static_cast<int64_t>(12);
  manifest_obj["observation_dim"] = static_cast<int64_t>(9182);
  manifest_obj["market_appendix_mode"] = "full_public_information_v3";
  manifest_obj["feature_schema_label"] = schema_label;
  manifest_obj["feature_schema_sha256"] = schema_sha256;
  manifest_obj["enable_semantic_scorer"] = true;
  manifest_obj["semantic_descriptor_schema"] = "semantic_action_v3";
  manifest_obj["specimen_exchange_penalty"] = 0.500000;
  manifest_obj["family_atomics_penalty"] = 0.100000;
  manifest_obj["plot_intrigue_penalty"] = 0.030000;
  manifest_obj["plot_intrigue_exemption_threshold"] = static_cast<int64_t>(3);
  manifest_obj["rollout_games"] = static_cast<int64_t>(256);
  manifest_obj["random_leader_draft"] = true;

  {
    std::ofstream ofs(manifest_tmp);
    ofs << json::ToString(manifest_obj, true);
  }

  std::filesystem::rename(model_tmp, model_path);
  std::filesystem::rename(optim_tmp, optim_path);
  std::filesystem::rename(manifest_tmp, manifest_path);
}

// Atomically aliases an existing bundle (src_tag) to a new bundle name (dst_tag)
// without re-serializing model or optimizer tensors.
void AliasDistillCheckpointBundle(
    const std::string& checkpoints_dir,
    const std::string& src_tag,
    const std::string& dst_tag) {
  std::string src_m = checkpoints_dir + "/ppo_model_" + src_tag + ".pt";
  std::string src_o = checkpoints_dir + "/ppo_optimizer_" + src_tag + ".pt";
  std::string src_j = checkpoints_dir + "/ppo_model_" + src_tag + ".json";

  std::string dst_m = checkpoints_dir + "/ppo_model_" + dst_tag + ".pt";
  std::string dst_o = checkpoints_dir + "/ppo_optimizer_" + dst_tag + ".pt";
  std::string dst_j = checkpoints_dir + "/ppo_model_" + dst_tag + ".json";

  std::string dst_m_tmp = dst_m + ".tmp";
  std::string dst_o_tmp = dst_o + ".tmp";
  std::string dst_j_tmp = dst_j + ".tmp";

  auto link_or_copy = [](const std::string& src, const std::string& tmp) {
    std::error_code ec;
    if (std::filesystem::exists(tmp, ec)) std::filesystem::remove(tmp, ec);
    std::filesystem::create_hard_link(src, tmp, ec);
    if (ec) {
      std::filesystem::copy_file(src, tmp, std::filesystem::copy_options::overwrite_existing, ec);
      if (ec) {
        SpielFatalError("Failed to link or copy bundle artifact (" + src + " -> " + tmp + "): " + ec.message());
      }
    }
  };

  link_or_copy(src_m, dst_m_tmp);
  link_or_copy(src_o, dst_o_tmp);

  std::ifstream ifs(src_j);
  if (!ifs) {
    SpielFatalError("Failed to open source manifest for aliasing: " + src_j);
  }
  std::string json_str((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
  ifs.close();

  auto parsed = json::FromString(json_str);
  if (!parsed || !parsed->IsObject()) {
    SpielFatalError("Failed to parse source manifest JSON for aliasing: " + src_j);
  }
  auto obj = parsed->GetObject();
  obj["model_filename"] = std::filesystem::path(dst_m).filename().string();
  obj["optimizer_filename"] = std::filesystem::path(dst_o).filename().string();

  {
    std::ofstream ofs(dst_j_tmp);
    if (!ofs) {
      SpielFatalError("Failed to open dest manifest temp file: " + dst_j_tmp);
    }
    ofs << json::ToString(obj, true);
  }

  std::filesystem::rename(dst_m_tmp, dst_m);
  std::filesystem::rename(dst_o_tmp, dst_o);
  std::filesystem::rename(dst_j_tmp, dst_j);
}

// Preserve latest two resume bundles (current_chunk and current_chunk - 1),
// and best_chunk. Prune older intermediate chunk bundles.
void PruneIntermediateCheckpoints(
    const std::string& checkpoints_dir,
    int current_chunk,
    int phase_start_chunk,
    int best_chunk) {
  for (int c = phase_start_chunk + 1; c < current_chunk - 1; ++c) {
    if (c == best_chunk) continue;
    std::string tag = absl::StrFormat("chunk_%d", c);
    std::string m_path = checkpoints_dir + "/ppo_model_" + tag + ".pt";
    std::string o_path = checkpoints_dir + "/ppo_optimizer_" + tag + ".pt";
    std::string j_path = checkpoints_dir + "/ppo_model_" + tag + ".json";
    std::error_code ec;
    if (std::filesystem::exists(m_path, ec)) std::filesystem::remove(m_path, ec);
    if (std::filesystem::exists(o_path, ec)) std::filesystem::remove(o_path, ec);
    if (std::filesystem::exists(j_path, ec)) std::filesystem::remove(j_path, ec);
  }
}

}  // namespace
}  // namespace open_spiel

int main(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);

  const std::string teacher_path = absl::GetFlag(FLAGS_teacher_checkpoint);
  const std::string student_path = absl::GetFlag(FLAGS_student_checkpoint);
  const std::string student_optim_path = absl::GetFlag(FLAGS_student_optim_checkpoint);
  const std::string out_dir = absl::GetFlag(FLAGS_output_dir);
  const int flag_start_chunk = absl::GetFlag(FLAGS_start_chunk);
  const int flag_total_chunks = absl::GetFlag(FLAGS_total_chunks);
  const int flag_phase_chunks = absl::GetFlag(FLAGS_phase_chunks);
  const int games_per_chunk = absl::GetFlag(FLAGS_games_per_chunk);
  const int epochs_per_chunk = absl::GetFlag(FLAGS_epochs_per_chunk);
  const int held_out_teacher_games = absl::GetFlag(FLAGS_held_out_teacher_games);
  const int held_out_student_games = absl::GetFlag(FLAGS_held_out_student_games);
  const int batch_size = absl::GetFlag(FLAGS_batch_size);
  const double lr_start = absl::GetFlag(FLAGS_learning_rate);
  const double lr_end = absl::GetFlag(FLAGS_learning_rate_end);
  const int num_threads = absl::GetFlag(FLAGS_threads);
  const uint64_t seed = absl::GetFlag(FLAGS_seed);
  const std::string held_out_path = absl::GetFlag(FLAGS_held_out_dataset_path);
  const uint64_t flag_initial_global_step = absl::GetFlag(FLAGS_initial_global_step);
  const double flag_initial_best_mean_kl = absl::GetFlag(FLAGS_initial_best_mean_kl);
  const int flag_initial_best_chunk = absl::GetFlag(FLAGS_initial_best_chunk);
  const std::string initial_best_model = absl::GetFlag(FLAGS_initial_best_model);
  const std::string initial_best_optim = absl::GetFlag(FLAGS_initial_best_optim);
  const std::string distillation_provenance = absl::GetFlag(FLAGS_distillation_provenance);

  std::filesystem::create_directories(out_dir);
  const std::string ckpt_dir = out_dir + "/checkpoints";
  std::filesystem::create_directories(ckpt_dir);

  torch::Device device(torch::kCPU);
  if (torch::cuda::is_available()) {
    device = torch::Device(torch::kCUDA, 0);
    std::cout << "[INFO] Using CUDA device: cuda:0\n";
  } else {
    std::cout << "[INFO] CUDA not available, using CPU.\n";
  }

  std::cout << "=================================================================\n";
  std::cout << "  Dune: Imperium Streamed Knowledge Distillation Pipeline\n";
  std::cout << "  Teacher Checkpoint: " << teacher_path << "\n";
  std::cout << "  Student Checkpoint: " << student_path << "\n";
  std::cout << "  Output Directory:   " << out_dir << "\n";
  std::cout << absl::StrFormat("  Phase Chunks:       %d..%d (%d additional chunks, %d games)\n",
                               flag_start_chunk + 1, flag_total_chunks,
                               flag_total_chunks - flag_start_chunk,
                               (flag_total_chunks - flag_start_chunk) * games_per_chunk);
  std::cout << absl::StrFormat("  Cosine LR Cycle:    %.6e -> %.6e across %d chunks\n",
                               lr_start, lr_end, flag_phase_chunks);
  std::cout << "=================================================================\n";

  std::shared_ptr<const open_spiel::Game> game = open_spiel::LoadGame("dune_imperium");

  // Step 1: Verify chance outcome sampling correction on nonuniform chance state
  open_spiel::VerifyNonuniformChanceSampling(*game);

  // Step 2: Load Teacher Model
  std::cout << "--- Loading Teacher Model ---\n";
  auto teacher = std::make_shared<open_spiel::SharedDunePolicyValueNetImpl>(
      /*input_dim=*/9182, /*hidden_dim=*/2048, /*action_dim=*/2391,
      /*num_blocks=*/12, /*use_nonlinear=*/false, /*with_aux_heads=*/false,
      /*head_init_seed=*/0, /*with_semantic_scorer=*/true);
  teacher->market_appendix_mode_ = open_spiel::dune_imperium::MarketAppendixMode::kFullPublicInformationV3;
  teacher->semantic_descriptor_schema_ = open_spiel::dune_semantic::kDescriptorSchemaVersionV3;

  open_spiel::LoadModelCheckpointRobust(teacher, teacher_path, device);
  teacher->to(device);
  teacher->eval();
  for (auto& p : teacher->parameters()) {
    p.set_requires_grad(false);
  }
  std::cout << "Teacher U31400 loaded successfully and frozen.\n";

  // Step 3: Initialize Student Model
  std::cout << "--- Initializing Student Model ---\n";
  auto student = std::make_shared<open_spiel::SharedDunePolicyValueNetImpl>(
      /*input_dim=*/9182, /*hidden_dim=*/2048, /*action_dim=*/2391,
      /*num_blocks=*/12, /*use_nonlinear=*/false, /*with_aux_heads=*/false,
      /*head_init_seed=*/0, /*with_semantic_scorer=*/true);
  student->market_appendix_mode_ = open_spiel::dune_imperium::MarketAppendixMode::kFullPublicInformationV3;
  student->semantic_descriptor_schema_ = open_spiel::dune_semantic::kDescriptorSchemaVersionV3;
  student->to(device);

  std::vector<torch::Tensor> policy_params = student->policy_head->parameters();
  std::vector<torch::Tensor> other_params;
  for (auto& p : student->input_layer->parameters()) other_params.push_back(p);
  for (auto& b : student->res_blocks) {
    for (auto& p : b->parameters()) other_params.push_back(p);
  }
  for (auto& p : student->value_head->parameters()) other_params.push_back(p);
  if (student->semantic_scorer_) {
    for (auto& p : student->semantic_scorer_->parameters()) other_params.push_back(p);
  }

  std::vector<torch::optim::OptimizerParamGroup> groups;
  groups.emplace_back(policy_params);
  groups.emplace_back(other_params);
  auto optimizer = std::make_unique<torch::optim::AdamW>(
      groups, torch::optim::AdamWOptions(lr_start).eps(1e-5).weight_decay(1e-4));

  // Step 4: Resume State Check
  std::string progress_path = out_dir + "/distill_progress.json";
  int completed_chunks = flag_start_chunk;
  int phase_start_chunk = flag_start_chunk;
  int total_chunks = flag_total_chunks;
  int phase_chunks = flag_phase_chunks;
  uint64_t global_step = flag_initial_global_step;
  double best_combined_mean_kl = flag_initial_best_mean_kl;
  int best_chunk = flag_initial_best_chunk;
  std::string current_student_model_path = student_path;
  std::string current_student_optim_path = student_optim_path;

  if (std::filesystem::exists(progress_path)) {
    std::ifstream p_in(progress_path);
    std::string p_str((std::istreambuf_iterator<char>(p_in)), std::istreambuf_iterator<char>());
    auto p_json = open_spiel::json::FromString(p_str);
    if (p_json.has_value() && p_json->IsObject()) {
      const auto& p_obj = p_json->GetObject();
      if (p_obj.count("completed_chunks")) completed_chunks = p_obj.at("completed_chunks").GetInt();
      if (p_obj.count("phase_start_chunk")) phase_start_chunk = p_obj.at("phase_start_chunk").GetInt();
      if (p_obj.count("phase_chunks")) phase_chunks = p_obj.at("phase_chunks").GetInt();
      if (p_obj.count("total_chunks")) total_chunks = p_obj.at("total_chunks").GetInt();
      if (p_obj.count("global_step")) global_step = p_obj.at("global_step").GetInt();
      if (p_obj.count("best_combined_mean_kl")) best_combined_mean_kl = p_obj.at("best_combined_mean_kl").GetDouble();
      if (p_obj.count("best_chunk")) best_chunk = p_obj.at("best_chunk").GetInt();
      if (p_obj.count("latest_model_path")) current_student_model_path = p_obj.at("latest_model_path").GetString();
      if (p_obj.count("latest_optim_path")) current_student_optim_path = p_obj.at("latest_optim_path").GetString();
      std::cout << absl::StrFormat(
          "[RESUME] Found existing progress in output dir: completed_chunks=%d, global_step=%llu, best_mean_kl=%.4f (chunk %d)\n"
          "         Resuming student from: %s\n",
          completed_chunks, static_cast<unsigned long long>(global_step),
          best_combined_mean_kl, best_chunk, current_student_model_path.c_str());
    }
  } else {
    // Starting fresh for this phase:
    // If initial_best_model is provided, ensure ckpt_dir has initial ppo_model_best.*
    std::string best_model_dest = ckpt_dir + "/ppo_model_best.pt";
    std::string best_optim_dest = ckpt_dir + "/ppo_optimizer_best.pt";
    std::string best_manifest_dest = ckpt_dir + "/ppo_model_best.json";
    if (!initial_best_model.empty() && std::filesystem::exists(initial_best_model) && !std::filesystem::exists(best_model_dest)) {
      std::cout << "Copying initial best model from " << initial_best_model << " to " << best_model_dest << "...\n";
      std::filesystem::copy_file(initial_best_model, best_model_dest, std::filesystem::copy_options::overwrite_existing);
      if (!initial_best_optim.empty() && std::filesystem::exists(initial_best_optim)) {
        std::filesystem::copy_file(initial_best_optim, best_optim_dest, std::filesystem::copy_options::overwrite_existing);
      }
      std::string initial_manifest = initial_best_model.substr(0, initial_best_model.find_last_of('.')) + ".json";
      if (std::filesystem::exists(initial_manifest)) {
        std::filesystem::copy_file(initial_manifest, best_manifest_dest, std::filesystem::copy_options::overwrite_existing);
      }
    }
    // Also copy best_fidelity_metrics.json if present in parent dir
    std::string parent_metrics = "/home/warcr/dune_drl_runtime/round7/u31400_distill_streamed_4096_20261008/best_fidelity_metrics.json";
    std::string current_best_metrics = out_dir + "/best_fidelity_metrics.json";
    if (std::filesystem::exists(parent_metrics) && !std::filesystem::exists(current_best_metrics)) {
      std::filesystem::copy_file(parent_metrics, current_best_metrics, std::filesystem::copy_options::overwrite_existing);
    }
  }

  // Load student weights and optimizer (preserves learned weights and AdamW moments)
  open_spiel::LoadModelCheckpointRobust(student, current_student_model_path, device);
  try {
    torch::load(*optimizer, current_student_optim_path, device);
    std::cout << "[OPTIMIZER LOAD] Loaded optimizer state from " << current_student_optim_path << "\n";
  } catch (const std::exception& e) {
    std::cout << "[OPTIMIZER LOAD] Fallback CPU load: " << e.what() << "\n";
    torch::load(*optimizer, current_student_optim_path, torch::kCPU);
  }

  // Step 5: Load Preserved Exact Held-Out Dataset
  std::cout << "\n--- Fixed Held-Out Dataset Setup ---\n";
  std::string held_out_cache = out_dir + "/held_out_dataset.bin";
  std::vector<open_spiel::StateTransition> held_out_data;

  if (!held_out_path.empty() && std::filesystem::exists(held_out_path)) {
    if (!std::filesystem::exists(held_out_cache)) {
      std::cout << "Copying preserved held-out dataset from " << held_out_path << " to " << held_out_cache << "...\n";
      std::filesystem::copy_file(held_out_path, held_out_cache, std::filesystem::copy_options::overwrite_existing);
    }
  }

  if (open_spiel::LoadHeldOutDataset(held_out_cache, held_out_data)) {
    std::cout << "Preserved exact held-out dataset loaded from cache: " << held_out_cache << " (" << held_out_data.size() << " transitions)\n";
  } else {
    std::cerr << "Error: Preserved held-out dataset not found at " << held_out_cache << "!\n";
    std::exit(1);
  }

  std::string fidelity_history_path = out_dir + "/fidelity_history.log";
  std::ofstream fidelity_history(fidelity_history_path, std::ios::app);

  // If starting completely fresh from chunk 0, record initial baseline fidelity
  if (completed_chunks == 0) {
    std::cout << "\n--- Baseline Fidelity Evaluation on Initial Student (Chunk 0) ---\n";
    fidelity_history << "\n=================================================================\n";
    fidelity_history << "  Initial Baseline Student Fidelity (Before Distillation)\n";
    fidelity_history << "=================================================================\n";

    double kl_t = 0.0, kl_s = 0.0;
    int64_t n_t = 0, n_s = 0;
    auto f_t = open_spiel::EvaluateFidelityPartition(
        student, held_out_data, 0, device, "Chunk 0 Baseline: Teacher-Generated Held-Out", std::cout, &kl_t, &n_t);
    open_spiel::EvaluateFidelityPartition(
        student, held_out_data, 0, device, "Chunk 0 Baseline: Teacher-Generated Held-Out", fidelity_history, nullptr, nullptr);

    auto f_s = open_spiel::EvaluateFidelityPartition(
        student, held_out_data, 1, device, "Chunk 0 Baseline: Student-Generated Held-Out", std::cout, &kl_s, &n_s);
    open_spiel::EvaluateFidelityPartition(
        student, held_out_data, 1, device, "Chunk 0 Baseline: Student-Generated Held-Out", fidelity_history, nullptr, nullptr);

    double init_mean_kl = (n_t + n_s > 0) ? (kl_t + kl_s) / (n_t + n_s) : 0.0;
    best_combined_mean_kl = init_mean_kl;
    best_chunk = 0;

    open_spiel::SaveDistillCheckpointBundle(
        student, *optimizer, ckpt_dir, "best", 0, 0, 0, best_combined_mean_kl, teacher_path, distillation_provenance);
    open_spiel::AliasDistillCheckpointBundle(ckpt_dir, "best", "latest");

    std::cout << absl::StrFormat(
        "Initial Combined Held-Out Mean KL: %.4f (Nontrivial decisions: %lld)\n\n",
        init_mean_kl, static_cast<long long>(n_t + n_s));
  }

  // Step 6: Chunk Training Loop
  const int games_per_source_per_chunk = games_per_chunk / 2; // 128
  std::mt19937_64 shuffle_rng(seed + 777);

  for (int chunk_idx = completed_chunks; chunk_idx < total_chunks; ++chunk_idx) {
    int current_chunk_num = chunk_idx + 1;
    int chunk_base_game_id = 1000 + chunk_idx * games_per_chunk;

    std::cout << "\n=================================================================\n";
    std::cout << absl::StrFormat("  Executing Chunk %d / %d (Games %d to %d)\n",
                                 current_chunk_num, total_chunks,
                                 chunk_base_game_id, chunk_base_game_id + games_per_chunk - 1);
    std::cout << "=================================================================\n";

    // 1. Collect teacher self-play games
    std::cout << absl::StrFormat("Collecting %d teacher self-play games (IDs %d..%d)...\n",
                                 games_per_source_per_chunk, chunk_base_game_id, chunk_base_game_id + games_per_source_per_chunk - 1);
    std::shared_mutex sync_mutex;
    auto t_eval = std::make_shared<open_spiel::BatchedEvaluator>(
        teacher, /*target_batch_size=*/32, /*timeout_ms=*/2, device, &sync_mutex, /*logit_cap=*/0.0f);
    auto t_trans = open_spiel::CollectRolloutGames(
        *game, t_eval, games_per_source_per_chunk, num_threads, seed + chunk_idx * 17,
        /*game_id_offset=*/chunk_base_game_id, /*source_group=*/0);
    t_eval = nullptr;

    // 2. Collect current student self-play games (freeze acting student)
    std::cout << absl::StrFormat("Collecting %d student self-play games (IDs %d..%d) with frozen student...\n",
                                 games_per_source_per_chunk, chunk_base_game_id + games_per_source_per_chunk,
                                 chunk_base_game_id + games_per_chunk - 1);
    student->eval();
    auto s_eval = std::make_shared<open_spiel::BatchedEvaluator>(
        student, /*target_batch_size=*/32, /*timeout_ms=*/2, device, &sync_mutex, /*logit_cap=*/0.0f);
    auto s_trans = open_spiel::CollectRolloutGames(
        *game, s_eval, games_per_source_per_chunk, num_threads, seed + 100000 + chunk_idx * 19,
        /*game_id_offset=*/chunk_base_game_id + games_per_source_per_chunk, /*source_group=*/1);
    s_eval = nullptr;

    // 3. Move both sources into chunk_transitions (bound memory, no duplication)
    std::vector<open_spiel::StateTransition> chunk_transitions;
    chunk_transitions.reserve(t_trans.size() + s_trans.size());
    chunk_transitions.insert(chunk_transitions.end(),
                             std::make_move_iterator(t_trans.begin()),
                             std::make_move_iterator(t_trans.end()));
    chunk_transitions.insert(chunk_transitions.end(),
                             std::make_move_iterator(s_trans.begin()),
                             std::make_move_iterator(s_trans.end()));
    t_trans.clear(); t_trans.shrink_to_fit();
    s_trans.clear(); s_trans.shrink_to_fit();

    std::cout << absl::StrFormat("Collected %zu total transitions for Chunk %d.\n",
                                 chunk_transitions.size(), current_chunk_num);

    // 4. Label all transitions with frozen teacher
    open_spiel::LabelTransitionsWithTeacher(teacher, chunk_transitions, device, batch_size);

    // 5. Train student on chunk data for 2 epochs
    student->train();
    const size_t n_trans = chunk_transitions.size();
    std::vector<size_t> chunk_indices(n_trans);
    std::iota(chunk_indices.begin(), chunk_indices.end(), 0);

    const int64_t obs_dim = 9182;
    const int64_t action_dim = 2391;
    const size_t num_batches_per_epoch = (n_trans + batch_size - 1) / batch_size;
    const size_t total_batches_in_chunk = epochs_per_chunk * num_batches_per_epoch;
    size_t batch_in_chunk = 0;

    for (int ep = 1; ep <= epochs_per_chunk; ++ep) {
      std::shuffle(chunk_indices.begin(), chunk_indices.end(), shuffle_rng);
      double ep_kl = 0.0, ep_mse = 0.0;
      size_t ep_count = 0;

      for (size_t start = 0; start < n_trans; start += batch_size) {
        size_t end = std::min(n_trans, start + batch_size);
        size_t b_len = end - start;

        // Explicit progress for the new phase:
        // Progress across the phase spans from 0.0 (start of chunk 17) to 1.0 (end of chunk 48)
        double chunk_progress = static_cast<double>(batch_in_chunk) / static_cast<double>(total_batches_in_chunk);
        double phase_progress = std::min(1.0, (static_cast<double>(chunk_idx - phase_start_chunk) + chunk_progress) / static_cast<double>(phase_chunks));
        double current_lr = lr_end + 0.5 * (lr_start - lr_end) * (1.0 + std::cos(phase_progress * M_PI));

        for (auto& grp : optimizer->param_groups()) {
          static_cast<torch::optim::AdamWOptions&>(grp.options()).lr(current_lr);
        }

        std::vector<float> flat_obs(b_len * obs_dim, 0.0f);
        std::vector<uint8_t> flat_masks(b_len * action_dim, 0);
        std::vector<float> flat_t_logits(b_len * action_dim, -1e9f);
        std::vector<float> flat_t_vals(b_len, 0.0f);
        std::vector<const open_spiel::dune_semantic::CandidateActionData*> mb_cands;

        for (size_t i = 0; i < b_len; ++i) {
          const auto& item = chunk_transitions[chunk_indices[start + i]];
          std::memcpy(&flat_obs[i * obs_dim], item.obs.data(), obs_dim * sizeof(float));
          flat_t_vals[i] = item.teacher_value;
          mb_cands.push_back(&item.cand_data);

          for (size_t j = 0; j < item.legal_actions.size(); ++j) {
            open_spiel::Action a = item.legal_actions[j];
            if (a >= 0 && a < action_dim) {
              flat_masks[i * action_dim + a] = 1;
              flat_t_logits[i * action_dim + a] = item.teacher_legal_logits[j];
            }
          }
        }

        torch::Tensor mb_states = torch::from_blob(
            flat_obs.data(), {static_cast<int64_t>(b_len), obs_dim}, torch::kFloat32).to(device);
        torch::Tensor mb_masks = torch::from_blob(
            flat_masks.data(), {static_cast<int64_t>(b_len), action_dim}, torch::kUInt8).to(device).to(torch::kBool);
        torch::Tensor mb_teacher_logits = torch::from_blob(
            flat_t_logits.data(), {static_cast<int64_t>(b_len), action_dim}, torch::kFloat32).to(device);
        torch::Tensor mb_teacher_vals = torch::from_blob(
            flat_t_vals.data(), {static_cast<int64_t>(b_len), 1}, torch::kFloat32).to(device);

        auto student_out = student->forward(mb_states);
        if (student->with_semantic_scorer_ && student->semantic_scorer_) {
          open_spiel::dune_semantic::ApplySemanticScorerBatch(
              student->semantic_scorer_, student_out.trunk, mb_cands, student_out.logits, device);
        }

        torch::Tensor student_logits = open_spiel::CenterAndCapLogitsTensor(student_out.logits, mb_masks, 10.0f);
        torch::Tensor student_masked = student_logits.masked_fill(mb_masks.logical_not(), -1e9f);
        torch::Tensor student_log_probs = torch::log_softmax(student_masked, -1);

        torch::Tensor teacher_masked = mb_teacher_logits.masked_fill(mb_masks.logical_not(), -1e9f);
        torch::Tensor teacher_probs = torch::softmax(teacher_masked, -1);
        torch::Tensor teacher_log_probs = torch::log_softmax(teacher_masked, -1);

        torch::Tensor kl = (teacher_probs * (teacher_log_probs - student_log_probs)).sum(-1);
        torch::Tensor val_mse = torch::mse_loss(student_out.values, mb_teacher_vals);
        torch::Tensor total_loss = kl.mean() + val_mse;

        optimizer->zero_grad();
        total_loss.backward();
        torch::nn::utils::clip_grad_norm_(student->parameters(), 1.0);
        optimizer->step();

        ep_kl += kl.sum().item<double>();
        ep_mse += (val_mse * b_len).item<double>();
        ep_count += b_len;
        ++batch_in_chunk;
        ++global_step;
      }

      double ep_progress = std::min(1.0, (static_cast<double>(chunk_idx - phase_start_chunk) + static_cast<double>(batch_in_chunk) / total_batches_in_chunk) / static_cast<double>(phase_chunks));
      double ep_lr = lr_end + 0.5 * (lr_start - lr_end) * (1.0 + std::cos(ep_progress * M_PI));
      std::cout << absl::StrFormat(
          "Chunk %2d, Epoch %d/%d:  Mean KL = %.5f,  Value MSE = %.5f (lr = %.6e, phase_progress = %.4f, step = %llu)\n",
          current_chunk_num, ep, epochs_per_chunk, ep_kl / ep_count, ep_mse / ep_count,
          ep_lr, ep_progress, static_cast<unsigned long long>(global_step));
    }

    // 6. Bound memory: release chunk training transitions immediately
    chunk_transitions.clear();
    chunk_transitions.shrink_to_fit();

    // 7. Evaluate held-out fidelity
    std::cout << absl::StrFormat("\n--- Chunk %d Held-Out Fidelity Evaluation ---\n", current_chunk_num);
    fidelity_history << "\n=================================================================\n";
    fidelity_history << absl::StrFormat("  Chunk %d Fidelity Report (Completed %d Games)\n",
                                        current_chunk_num, 4096 + (current_chunk_num - phase_start_chunk) * games_per_chunk);
    fidelity_history << "=================================================================\n";

    double kl_t = 0.0, kl_s = 0.0;
    int64_t n_t = 0, n_s = 0;
    auto f_t = open_spiel::EvaluateFidelityPartition(
        student, held_out_data, 0, device,
        absl::StrFormat("Chunk %d: Teacher-Generated Held-Out", current_chunk_num),
        std::cout, &kl_t, &n_t);
    open_spiel::EvaluateFidelityPartition(
        student, held_out_data, 0, device,
        absl::StrFormat("Chunk %d: Teacher-Generated Held-Out", current_chunk_num),
        fidelity_history, nullptr, nullptr);

    auto f_s = open_spiel::EvaluateFidelityPartition(
        student, held_out_data, 1, device,
        absl::StrFormat("Chunk %d: Student-Generated Held-Out", current_chunk_num),
        std::cout, &kl_s, &n_s);
    open_spiel::EvaluateFidelityPartition(
        student, held_out_data, 1, device,
        absl::StrFormat("Chunk %d: Student-Generated Held-Out", current_chunk_num),
        fidelity_history, nullptr, nullptr);

    double combined_mean_kl = (n_t + n_s > 0) ? (kl_t + kl_s) / (n_t + n_s) : 0.0;
    std::cout << absl::StrFormat(
        "Chunk %d Combined Held-Out Mean KL: %.4f (Previous Best: %.4f from Chunk %d)\n",
        current_chunk_num, combined_mean_kl, best_combined_mean_kl, best_chunk);

    // 8. Save chunk checkpoint and update latest
    std::string chunk_tag = absl::StrFormat("chunk_%d", current_chunk_num);
    uint64_t cumulative_games = 4096 + (current_chunk_num - phase_start_chunk) * games_per_chunk;

    open_spiel::SaveDistillCheckpointBundle(
        student, *optimizer, ckpt_dir, chunk_tag, current_chunk_num,
        global_step, cumulative_games, combined_mean_kl, teacher_path, distillation_provenance);

    open_spiel::AliasDistillCheckpointBundle(ckpt_dir, chunk_tag, "latest");

    bool is_best = false;
    if (combined_mean_kl < best_combined_mean_kl) {
      best_combined_mean_kl = combined_mean_kl;
      best_chunk = current_chunk_num;
      is_best = true;

      open_spiel::AliasDistillCheckpointBundle(ckpt_dir, chunk_tag, "best");

      open_spiel::json::Object best_metrics;
      best_metrics["best_chunk"] = best_chunk;
      best_metrics["best_combined_mean_kl"] = best_combined_mean_kl;
      best_metrics["teacher_generated_held_out"] = f_t;
      best_metrics["student_generated_held_out"] = f_s;
      std::string best_m_path = out_dir + "/best_fidelity_metrics.json";
      std::ofstream b_out(best_m_path);
      b_out << open_spiel::json::ToString(best_metrics, true);

      std::cout << absl::StrFormat(">>> NEW BEST CHECKPOINT retained at Chunk %d with combined mean KL = %.4f <<<\n",
                                   best_chunk, best_combined_mean_kl);
    }

    // 9. Prune older intermediate chunk bundles to preserve latest two + best
    open_spiel::PruneIntermediateCheckpoints(ckpt_dir, current_chunk_num, phase_start_chunk, best_chunk);

    // 10. Atomically record progress for safe restart
    open_spiel::json::Object prog_obj;
    prog_obj["completed_chunks"] = current_chunk_num;
    prog_obj["next_chunk_index"] = current_chunk_num;
    prog_obj["phase_start_chunk"] = phase_start_chunk;
    prog_obj["phase_chunks"] = phase_chunks;
    prog_obj["phase_progress"] = static_cast<double>(current_chunk_num - phase_start_chunk) / static_cast<double>(phase_chunks);
    prog_obj["lr_start"] = lr_start;
    prog_obj["lr_end"] = lr_end;
    prog_obj["total_chunks"] = total_chunks;
    prog_obj["completed_games"] = static_cast<int64_t>(cumulative_games);
    prog_obj["next_game_id"] = chunk_base_game_id + games_per_chunk;
    prog_obj["global_step"] = static_cast<int64_t>(global_step);
    prog_obj["best_combined_mean_kl"] = best_combined_mean_kl;
    prog_obj["best_chunk"] = best_chunk;
    prog_obj["latest_model_path"] = ckpt_dir + "/ppo_model_latest.pt";
    prog_obj["latest_optim_path"] = ckpt_dir + "/ppo_optimizer_latest.pt";
    prog_obj["best_model_path"] = ckpt_dir + "/ppo_model_best.pt";
    prog_obj["best_optim_path"] = ckpt_dir + "/ppo_optimizer_best.pt";

    std::string prog_tmp = progress_path + ".tmp";
    {
      std::ofstream p_out(prog_tmp);
      p_out << open_spiel::json::ToString(prog_obj, true);
    }
    std::filesystem::rename(prog_tmp, progress_path);
  }

  std::cout << "\n=================================================================\n";
  std::cout << absl::StrFormat("Distillation Complete! %d Chunks (%d cumulative games) Finished.\n",
                               total_chunks, 4096 + (total_chunks - phase_start_chunk) * games_per_chunk);
  std::cout << absl::StrFormat("Selected Best Checkpoint: Chunk %d (Combined Held-Out Mean KL: %.4f)\n",
                               best_chunk, best_combined_mean_kl);
  std::cout << "Model:     " << ckpt_dir << "/ppo_model_best.pt\n";
  std::cout << "Optimizer: " << ckpt_dir << "/ppo_optimizer_best.pt\n";
  std::cout << "Manifest:  " << ckpt_dir << "/ppo_model_best.json\n";
  std::cout << "=================================================================\n";

  return 0;
}
