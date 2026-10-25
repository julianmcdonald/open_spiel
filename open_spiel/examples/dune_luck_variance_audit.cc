#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mutex>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include <c10/util/Half.h>

#include "open_spiel/abseil-cpp/absl/flags/flag.h"
#include "open_spiel/abseil-cpp/absl/flags/parse.h"
#include "open_spiel/abseil-cpp/absl/strings/str_format.h"
#include "open_spiel/abseil-cpp/absl/types/span.h"
#include "open_spiel/games/dune_imperium/dune_imperium.h"
#include "open_spiel/spiel.h"
#include "open_spiel/utils/json.h"

#include "dune_network.h"
#include "dune_ppo_training_flags.h"
#include "dune_ppo_training_utils.h"
#include "dune_search_routing.h"
#include "dune_seed_utils.h"
#include "dune_semantic_action_scorer.h"
#include "dune_specimen_conversion.h"

ABSL_FLAG(std::string, mode, "all",
          "Execution mode: all, step1, step2, collect_states_only, collect_r7_44_critic_data");
ABSL_FLAG(int, num_games, 200,
          "Total number of games (first half selection, second half confirmation)");
ABSL_FLAG(uint64_t, master_seed, 34350, "Master seed for RNG derivation");
ABSL_FLAG(uint64_t, episode_start, 0, "Starting episode ID");
ABSL_FLAG(int, threads, 16, "Worker threads for game simulation");
ABSL_FLAG(int, eval_batch_size, 64, "Batched evaluator batch size");
ABSL_FLAG(int, eval_timeout_ms, 1, "Batched evaluator timeout in ms");
ABSL_FLAG(std::string, output_dir,
          "/home/warcr/dune_drl_runtime/round7/r7_43_luck_plasticity_audit_20261019",
          "Output directory for JSON and reports");
ABSL_FLAG(std::string, states_file,
          "/home/warcr/dune_drl_runtime/round7/r7_43_luck_plasticity_audit_20261019/confirmation_10k_states.bin",
          "Path to binary states file for Step 2");
ABSL_FLAG(std::string, step2_device, "cpu",
          "Device for Step 2 plasticity audit (cpu or cuda)");
ABSL_FLAG(int, bootstrap_resamples, 10000,
          "Number of bootstrap resamples for Step 1");
ABSL_FLAG(std::string, checkpoint_u34350_live,
          "/run/media/warcr/Storage/dune_drl_runtime/round7/u33350_16block_emagnet_to_u34350_20261017/checkpoints/ppo_model_update_34350.pt",
          "U34350 live checkpoint");
ABSL_FLAG(std::string, checkpoint_u34350_ema,
          "/run/media/warcr/Storage/dune_drl_runtime/round7/u33350_16block_emagnet_to_u34350_20261017/checkpoints/ppo_model_ema_update_34350.pt",
          "U34350 EMA checkpoint");
ABSL_FLAG(std::string, checkpoint_u33350_16init,
          "/run/media/warcr/Storage/dune_drl_runtime/round7/u33350_16block_init/ppo_model_update_33350.pt",
          "U33350 16b init checkpoint");
ABSL_FLAG(std::string, checkpoint_u33350_12parent,
          "/run/media/warcr/Storage/dune_drl_runtime/round7/u33000_emagnet_coef002_to_u33350_20261015/checkpoints/ppo_model_update_33350.pt",
          "U33350 12b parent checkpoint");
ABSL_FLAG(std::string, checkpoint_u29700_12,
          "/run/media/warcr/Storage/dune_drl_runtime/round7/u29200_12block_continuation_20261004/checkpoints/ppo_model_update_29700.pt",
          "U29700 12b checkpoint");
ABSL_FLAG(int, r744_rows_per_game, 32,
          "Sampled rows per game for R7-44 critic dataset");
ABSL_FLAG(int, r744_game_id_offset, 0,
          "Game ID offset for R7-44 critic dataset");


namespace open_spiel {

constexpr uint64_t kStatesFileMagic = 0x44554E4553544154ULL; // "DUNESTAT"
constexpr uint32_t kStatesFileVersion = 1;
constexpr int64_t kObsDim = 9182;
constexpr int64_t kHiddenDim = 2048;
constexpr int64_t kActionDim = 2391;

// ---------------------------------------------------------------------------
// Step 1 Data Structures
// ---------------------------------------------------------------------------

struct Step1Row {
  int player = -1;
  int round = 1;
  int round_band = 0; // 0: R1-3, 1: R4-6, 2: R7+
  int decision_role = 0;
  bool policy_loss_eligible = false; // actions.size() > 1

  std::vector<open_spiel::Action> legal_actions;
  open_spiel::Action chosen_action = open_spiel::kInvalidAction;
  std::vector<float> obs; // 9182 floats
  dune_semantic::CandidateActionData cand_data;

  float value = 0.0f;
  float raw_row_reward = 0.0f;
  bool is_last_row = false;
  float terminal_return = 0.0f;
  float next_row_value = 0.0f;
  float clamped_reward = 0.0f;
  float mc_return = 0.0f;
  float X_real = 0.0f;

  std::unique_ptr<open_spiel::State> post_action_state;

  // Variant F data
  bool f_deterministic_skip = false;
  float X_alt[8] = {0.0f};
  int f_opp_policy_calls[8] = {0};
  int f_val_only_calls[8] = {0};

  // Variant C data:
  // index 0: k=2, index 1: k=4, index 2: k=8
  float luck_C_on[3] = {0.0f, 0.0f, 0.0f};
  float luck_C_off[3] = {0.0f, 0.0f, 0.0f};
  int c_on_extra_val_calls[3] = {0, 0, 0};
  int c_on_total_val_calls[3] = {0, 0, 0};
  int c_off_extra_val_calls[3] = {0, 0, 0};

  // Computed advantages during backward pass:
  // A[0: lambda=0.95, 1: lambda=0.98, 2: lambda=1.0]
  float A[3] = {0.0f, 0.0f, 0.0f};

  // Computed C for each configuration
  // C_val[config_id] (for F and C_on)
  // C_off_val[config_id] (for C_onoff)
  std::vector<float> C_val;
  std::vector<float> C_off_val;
};

struct Step1ChanceBlockEvent {
  int block_index = 0;
  int actor_before_block = -1;
  int most_recent_row[4] = {-1, -1, -1, -1};
  std::unique_ptr<open_spiel::State> chance_block_start;
  std::unique_ptr<open_spiel::State> real_endpoint;
};

struct Step1GameData {
  int game_index = 0;
  uint64_t episode_id = 0;
  int final_round = 1;
  std::vector<Step1Row> rows;
  std::vector<int> player_rows[4];
  std::vector<Step1ChanceBlockEvent> chance_events;

  // Cost tracking
  int f_total_opp_policy_calls = 0;
  int f_total_val_only_calls = 0;
  int c_total_val_only_calls = 0;
};

// ---------------------------------------------------------------------------
// Step 2 Saved State Binary I/O
// ---------------------------------------------------------------------------

struct SavedState {
  float mc_return = 0.0f;
  float terminal_return = 0.0f;
  std::vector<float> obs;
  std::vector<open_spiel::Action> legal_actions;
  dune_semantic::CandidateActionData cand_data;
};

bool SaveConfirmationStates(const std::string& path,
                            const std::vector<SavedState>& states) {
  std::filesystem::path p(path);
  if (p.has_parent_path()) {
    std::filesystem::create_directories(p.parent_path());
  }
  std::ofstream out(path, std::ios::binary);
  if (!out.is_open()) {
    std::cerr << "Error: failed to open states file for writing: " << path << "\n";
    return false;
  }

  uint64_t magic = kStatesFileMagic;
  uint32_t version = kStatesFileVersion;
  uint32_t num_states = static_cast<uint32_t>(states.size());
  uint32_t obs_dim = static_cast<uint32_t>(kObsDim);

  out.write(reinterpret_cast<const char*>(&magic), sizeof(magic));
  out.write(reinterpret_cast<const char*>(&version), sizeof(version));
  out.write(reinterpret_cast<const char*>(&num_states), sizeof(num_states));
  out.write(reinterpret_cast<const char*>(&obs_dim), sizeof(obs_dim));

  for (const auto& s : states) {
    out.write(reinterpret_cast<const char*>(&s.mc_return), sizeof(float));
    out.write(reinterpret_cast<const char*>(&s.terminal_return), sizeof(float));
    out.write(reinterpret_cast<const char*>(s.obs.data()), kObsDim * sizeof(float));

    uint32_t num_acts = static_cast<uint32_t>(s.legal_actions.size());
    out.write(reinterpret_cast<const char*>(&num_acts), sizeof(num_acts));
    out.write(reinterpret_cast<const char*>(s.legal_actions.data()),
              num_acts * sizeof(open_spiel::Action));

    out.write(reinterpret_cast<const char*>(s.cand_data.features.data()),
              num_acts * dune_semantic::kSemanticFeatDim * sizeof(float));
    out.write(reinterpret_cast<const char*>(s.cand_data.card_ids.data()),
              num_acts * sizeof(int64_t));
    out.write(reinterpret_cast<const char*>(s.cand_data.space_ids.data()),
              num_acts * sizeof(int64_t));
    out.write(reinterpret_cast<const char*>(s.cand_data.supported.data()),
              num_acts * sizeof(uint8_t));

    std::vector<int32_t> roles_i32(num_acts);
    for (size_t a = 0; a < num_acts; ++a) {
      roles_i32[a] = static_cast<int32_t>(s.cand_data.roles[a]);
    }
    out.write(reinterpret_cast<const char*>(roles_i32.data()),
              num_acts * sizeof(int32_t));
  }

  out.close();
  std::cout << "Successfully saved " << states.size()
            << " states to binary file: " << path << "\n";
  return true;
}

std::vector<SavedState> LoadConfirmationStates(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in.is_open()) {
    open_spiel::SpielFatalError("Failed to open states file for reading: " + path);
  }

  uint64_t magic = 0;
  uint32_t version = 0;
  uint32_t num_states = 0;
  uint32_t obs_dim = 0;

  in.read(reinterpret_cast<char*>(&magic), sizeof(magic));
  in.read(reinterpret_cast<char*>(&version), sizeof(version));
  in.read(reinterpret_cast<char*>(&num_states), sizeof(num_states));
  in.read(reinterpret_cast<char*>(&obs_dim), sizeof(obs_dim));

  if (magic != kStatesFileMagic || version != kStatesFileVersion ||
      obs_dim != kObsDim) {
    open_spiel::SpielFatalError("Invalid states file header in: " + path);
  }

  std::vector<SavedState> states(num_states);
  for (uint32_t i = 0; i < num_states; ++i) {
    SavedState& s = states[i];
    in.read(reinterpret_cast<char*>(&s.mc_return), sizeof(float));
    in.read(reinterpret_cast<char*>(&s.terminal_return), sizeof(float));
    s.obs.resize(kObsDim);
    in.read(reinterpret_cast<char*>(s.obs.data()), kObsDim * sizeof(float));

    uint32_t num_acts = 0;
    in.read(reinterpret_cast<char*>(&num_acts), sizeof(num_acts));
    s.legal_actions.resize(num_acts);
    in.read(reinterpret_cast<char*>(s.legal_actions.data()),
            num_acts * sizeof(open_spiel::Action));

    s.cand_data.actions = s.legal_actions;
    s.cand_data.features.resize(num_acts * dune_semantic::kSemanticFeatDim);
    s.cand_data.card_ids.resize(num_acts);
    s.cand_data.space_ids.resize(num_acts);
    s.cand_data.supported.resize(num_acts);
    s.cand_data.roles.resize(num_acts);

    in.read(reinterpret_cast<char*>(s.cand_data.features.data()),
            num_acts * dune_semantic::kSemanticFeatDim * sizeof(float));
    in.read(reinterpret_cast<char*>(s.cand_data.card_ids.data()),
            num_acts * sizeof(int64_t));
    in.read(reinterpret_cast<char*>(s.cand_data.space_ids.data()),
            num_acts * sizeof(int64_t));
    in.read(reinterpret_cast<char*>(s.cand_data.supported.data()),
            num_acts * sizeof(uint8_t));

    std::vector<int32_t> roles_i32(num_acts);
    in.read(reinterpret_cast<char*>(roles_i32.data()),
            num_acts * sizeof(int32_t));
    for (size_t a = 0; a < num_acts; ++a) {
      s.cand_data.roles[a] = static_cast<dune_semantic::ActionRole>(roles_i32[a]);
    }
  }

  std::cout << "Successfully loaded " << states.size()
            << " states from binary file: " << path << "\n";
  return states;
}

// ---------------------------------------------------------------------------
// Step 1 Simulation Worker
// ---------------------------------------------------------------------------

Step1GameData SimulateRealTrajectory(
    int game_index, uint64_t episode_id, uint64_t master_seed,
    const open_spiel::Game& game,
    const std::shared_ptr<open_spiel::BatchedEvaluator>& evaluator,
    bool keep_states_and_chance = true) {
  Step1GameData game_data;
  game_data.game_index = game_index;
  game_data.episode_id = episode_id;

  std::mt19937_64 chance_rng = dune_seed::MakeRng64(dune_seed::DeriveSeed(
      master_seed, dune_seed::kDomainTrain, episode_id, dune_seed::kStreamChance));
  std::mt19937_64 policy_rng[4];
  for (int p = 0; p < 4; ++p) {
    uint64_t seed = dune_seed::DeriveSeed(
        master_seed, dune_seed::kDomainTrain, episode_id,
        dune_seed::kStreamPolicyPlayer0 + p);
    policy_rng[p] = dune_seed::MakeRng64(seed);
  }

  std::unique_ptr<open_spiel::State> state = game.NewInitialState();
  const auto* dune_state =
      dynamic_cast<const dune_imperium::DuneImperiumState*>(state.get());

  int last_row_for_player[4] = {-1, -1, -1, -1};
  int chance_block_count = 0;
  int last_actor = -1;

  int game_length = 0;
  while (!state->IsTerminal()) {
    if (++game_length > 5000) {
      open_spiel::SpielFatalError("Game simulation infinite loop detected");
    }

    // 1. Chance node: contiguous chance block handling
    if (state->IsChanceNode()) {
      bool any_player_active = false;
      for (int p = 0; p < 4; ++p) {
        if (last_row_for_player[p] >= 0) any_player_active = true;
      }

      std::unique_ptr<open_spiel::State> chance_start = nullptr;
      if (any_player_active && keep_states_and_chance) {
        chance_start = state->Clone();
      }

      while (state->IsChanceNode()) {
        auto outcomes = state->ChanceOutcomes();
        open_spiel::Action action =
            open_spiel::SampleAction(outcomes, chance_rng).first;
        state->ApplyAction(action);
      }

      if (any_player_active && keep_states_and_chance) {
        Step1ChanceBlockEvent ev;
        ev.block_index = chance_block_count++;
        ev.actor_before_block = last_actor;
        ev.chance_block_start = std::move(chance_start);
        ev.real_endpoint = state->Clone();
        for (int p = 0; p < 4; ++p) {
          ev.most_recent_row[p] = last_row_for_player[p];
        }
        game_data.chance_events.push_back(std::move(ev));
      }
      continue;
    }

    if (state->CurrentPlayer() == open_spiel::kSimultaneousPlayerId) {
      std::vector<open_spiel::Action> joint_action;
      for (int p = 0; p < game.NumPlayers(); ++p) {
        std::vector<open_spiel::Action> actions = state->LegalActions(p);
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

    // 2. Player decision
    open_spiel::Player p = state->CurrentPlayer();
    std::vector<open_spiel::Action> actions = state->LegalActions();
    if (actions.empty()) {
      open_spiel::SpielFatalError("Non-terminal state has no legal actions");
    }

    std::vector<float> obs(kObsDim, 0.0f);
    dune_state->InformationStateTensorWithAppendix(
        p, dune_imperium::MarketAppendixMode::kFullPublicInformationV3,
        absl::MakeSpan(obs), evaluator->DeckPoolSemantics());

    dune_semantic::CandidateActionData cand_data;
    dune_semantic::ExtractCandidateDescriptors(*dune_state, actions, &cand_data,
                                              "semantic_action_v3");

    open_spiel::EvalResult res =
        evaluator->EvaluateWithActions(obs, &cand_data);
    std::vector<float> logits = std::move(res.logits);
    open_spiel::CenterAndCapLegalLogits(logits, actions, 10.0f);

    std::vector<double> legal_probs;
    open_spiel::PolicyDistributionSample sample =
        open_spiel::SampleTrainingPolicyDistribution(*state, &policy_rng[p],
                                                     logits, actions, &legal_probs);
    open_spiel::Action chosen_action = sample.action;

    // Pre-action shaping reward penalties
    bool is_specimen = dune_shaping::IsSpecimenConversionAction(chosen_action);
    bool is_atomics = (chosen_action == dune_imperium::kActionFamilyAtomics);
    bool atomics_exempt =
        is_atomics && dune_state->IsPlayerInRevealTurn(p) &&
        (dune_state->PlayerPersuasion(p) > 0);
    bool is_plot = open_spiel::IsPlotIntrigueAction(chosen_action);
    bool plot_exempt =
        is_plot && (static_cast<int>(
                        dune_state->GetIntrigueHandForTesting(p).size()) >= 3);

    float raw_row_rew = 0.0f;
    if (is_specimen) raw_row_rew -= 0.5f;
    if (is_atomics && !atomics_exempt) raw_row_rew -= 0.1f;
    if (is_plot && !plot_exempt) raw_row_rew -= 0.03f;

    int cur_round = dune_state->GetCurrentRound();
    int r_band = 0;
    if (cur_round >= 7) r_band = 2;
    else if (cur_round >= 4) r_band = 1;

    int role_id = static_cast<int>(
        open_spiel::ClassifyDuneDecisionRole(*state, p, false));

    Step1Row row;
    row.player = p;
    row.round = cur_round;
    row.round_band = r_band;
    row.decision_role = role_id;
    row.policy_loss_eligible = (actions.size() > 1);
    row.legal_actions = std::move(actions);
    row.chosen_action = chosen_action;
    row.obs = std::move(obs);
    row.cand_data = std::move(cand_data);
    row.value = res.value;
    row.raw_row_reward = raw_row_rew;

    last_actor = p;
    state->ApplyAction(chosen_action);
    if (keep_states_and_chance) {
      row.post_action_state = state->Clone();
    }

    int row_idx = static_cast<int>(game_data.rows.size());
    game_data.rows.push_back(std::move(row));
    game_data.player_rows[p].push_back(row_idx);
    last_row_for_player[p] = row_idx;
  }

  // Terminal state resolution
  game_data.final_round = dune_state->GetCurrentRound();
  std::vector<double> term_returns =
      open_spiel::ComputeTerminalReturns(*state, "placement", 0.0);

  for (int p = 0; p < 4; ++p) {
    int last_idx = last_row_for_player[p];
    if (last_idx >= 0) {
      game_data.rows[last_idx].is_last_row = true;
      game_data.rows[last_idx].terminal_return =
          static_cast<float>(term_returns[p]);
    }

    const auto& p_rows = game_data.player_rows[p];
    for (size_t i = 0; i < p_rows.size(); ++i) {
      int curr_r = p_rows[i];
      if (i + 1 < p_rows.size()) {
        int next_r = p_rows[i + 1];
        game_data.rows[curr_r].next_row_value = game_data.rows[next_r].value;
      } else {
        game_data.rows[curr_r].next_row_value = 0.0f;
      }

      float total_raw_rew =
          game_data.rows[curr_r].raw_row_reward +
          (game_data.rows[curr_r].is_last_row
               ? game_data.rows[curr_r].terminal_return
               : 0.0f);
      game_data.rows[curr_r].clamped_reward =
          std::clamp(total_raw_rew / 4.0f, -1.0f, 1.0f);
      game_data.rows[curr_r].X_real =
          game_data.rows[curr_r].clamped_reward +
          game_data.rows[curr_r].next_row_value;
    }

    // Running MC return
    float running_mc = 0.0f;
    for (int i = static_cast<int>(p_rows.size()) - 1; i >= 0; --i) {
      int curr_r = p_rows[i];
      running_mc += game_data.rows[curr_r].clamped_reward;
      game_data.rows[curr_r].mc_return = running_mc;
    }
  }

  return game_data;
}

void SimulateVariantFForRow(
    Step1Row& row, int game_idx, int row_idx, uint64_t master_seed,
    uint64_t episode_id,
    const std::shared_ptr<open_spiel::BatchedEvaluator>& evaluator) {
  if (row.post_action_state == nullptr) return;
  int p = row.player;

  // Exact deterministic skip
  std::unique_ptr<open_spiel::State> probe = row.post_action_state->Clone();
  while (probe->IsChanceNode()) {
    auto outcomes = probe->ChanceOutcomes();
    if (outcomes.size() != 1) break;
    probe->ApplyAction(outcomes[0].first);
  }
  if (probe->IsTerminal() ||
      (!probe->IsChanceNode() && probe->CurrentPlayer() == p)) {
    row.f_deterministic_skip = true;
    for (int a = 0; a < 8; ++a) {
      row.X_alt[a] = row.X_real;
      row.f_opp_policy_calls[a] = 0;
      row.f_val_only_calls[a] = 0;
    }
  } else {
    row.f_deterministic_skip = false;
    for (int alt_idx = 0; alt_idx < 8; ++alt_idx) {
      uint64_t alt_chance_seed = dune_seed::DeriveSeed(
          master_seed, 0x4C55434B46ULL, episode_id,
          (static_cast<uint64_t>(row_idx) << 8) | alt_idx, 0);
      std::mt19937_64 alt_chance_rng = dune_seed::MakeRng64(alt_chance_seed);
      std::mt19937_64 alt_policy_rng[4];
      for (int q = 0; q < 4; ++q) {
        uint64_t alt_pol_seed = dune_seed::DeriveSeed(
            master_seed, 0x4C55434B46ULL, episode_id,
            (static_cast<uint64_t>(row_idx) << 8) | alt_idx, 1 + q);
        alt_policy_rng[q] = dune_seed::MakeRng64(alt_pol_seed);
      }

      std::unique_ptr<open_spiel::State> alt_state =
          row.post_action_state->Clone();
      int opp_calls = 0;
      int val_calls = 0;
      int steps = 0;
      while (!alt_state->IsTerminal() &&
             (alt_state->IsChanceNode() || alt_state->CurrentPlayer() != p)) {
        if (++steps > 2000) {
          open_spiel::SpielFatalError("Variant F alt simulation exceeded 2000 steps");
        }
        if (alt_state->IsChanceNode()) {
          auto outcomes = alt_state->ChanceOutcomes();
          open_spiel::Action a =
              open_spiel::SampleAction(outcomes, alt_chance_rng).first;
          alt_state->ApplyAction(a);
        } else if (alt_state->CurrentPlayer() ==
                   open_spiel::kSimultaneousPlayerId) {
          std::vector<open_spiel::Action> joint_action;
          for (int pl = 0; pl < 4; ++pl) {
            std::vector<open_spiel::Action> legal = alt_state->LegalActions(pl);
            if (legal.empty()) {
              joint_action.push_back(0);
            } else {
              std::uniform_int_distribution<int> dis(0, legal.size() - 1);
              joint_action.push_back(legal[dis(alt_policy_rng[pl])]);
            }
          }
          alt_state->ApplyActions(joint_action);
        } else {
          open_spiel::Player q = alt_state->CurrentPlayer();
          std::vector<open_spiel::Action> actions_q = alt_state->LegalActions();
          const auto* dune_alt =
              dynamic_cast<const dune_imperium::DuneImperiumState*>(
                  alt_state.get());
          std::vector<float> obs_q(kObsDim, 0.0f);
          dune_alt->InformationStateTensorWithAppendix(
              q, dune_imperium::MarketAppendixMode::kFullPublicInformationV3,
              absl::MakeSpan(obs_q), evaluator->DeckPoolSemantics());
          dune_semantic::CandidateActionData cand_q;
          dune_semantic::ExtractCandidateDescriptors(
              *dune_alt, actions_q, &cand_q, "semantic_action_v3");

          open_spiel::EvalResult eval_q =
              evaluator->EvaluateWithActions(obs_q, &cand_q);
          opp_calls++;
          std::vector<float> logits_q = std::move(eval_q.logits);
          open_spiel::CenterAndCapLegalLogits(logits_q, actions_q, 10.0f);
          std::vector<double> probs_q;
          open_spiel::PolicyDistributionSample sample_q =
              open_spiel::SampleTrainingPolicyDistribution(
                  *alt_state, &alt_policy_rng[q], logits_q, actions_q, &probs_q);
          alt_state->ApplyAction(sample_q.action);
        }
      }

      if (alt_state->IsTerminal()) {
        float alt_term = static_cast<float>(
            open_spiel::ComputeTerminalReturns(*alt_state, "placement", 0.0)[p]);
        float alt_rew =
            std::clamp((row.raw_row_reward + alt_term) / 4.0f, -1.0f, 1.0f);
        row.X_alt[alt_idx] = alt_rew;
      } else {
        const auto* dune_alt =
            dynamic_cast<const dune_imperium::DuneImperiumState*>(
                alt_state.get());
        std::vector<float> obs_p(kObsDim, 0.0f);
        dune_alt->InformationStateTensorWithAppendix(
            p, dune_imperium::MarketAppendixMode::kFullPublicInformationV3,
            absl::MakeSpan(obs_p), evaluator->DeckPoolSemantics());
        open_spiel::EvalResult val_res = evaluator->EvaluateValues(obs_p);
        val_calls++;
        float alt_rew = std::clamp(row.raw_row_reward / 4.0f, -1.0f, 1.0f);
        row.X_alt[alt_idx] = alt_rew + val_res.value;
      }
      row.f_opp_policy_calls[alt_idx] = opp_calls;
      row.f_val_only_calls[alt_idx] = val_calls;
    }
  }
  row.post_action_state.reset();
}

void RunVariantCOnGame(
    Step1GameData& game_data, uint64_t master_seed,
    const std::shared_ptr<open_spiel::BatchedEvaluator>& evaluator) {
  for (size_t ev_idx = 0; ev_idx < game_data.chance_events.size(); ++ev_idx) {
    Step1ChanceBlockEvent& ev = game_data.chance_events[ev_idx];

    std::vector<std::unique_ptr<open_spiel::State>> alt_endpoints(8);
    for (int alt_idx = 0; alt_idx < 8; ++alt_idx) {
      uint64_t alt_seed = dune_seed::DeriveSeed(
          master_seed, 0x4C55434B43ULL, game_data.episode_id,
          (static_cast<uint64_t>(ev.block_index) << 8) | alt_idx, 0);
      std::mt19937_64 alt_rng = dune_seed::MakeRng64(alt_seed);
      std::unique_ptr<open_spiel::State> alt_state =
          ev.chance_block_start->Clone();
      while (alt_state->IsChanceNode()) {
        auto outcomes = alt_state->ChanceOutcomes();
        open_spiel::Action a =
            open_spiel::SampleAction(outcomes, alt_rng).first;
        alt_state->ApplyAction(a);
      }
      alt_endpoints[alt_idx] = std::move(alt_state);
    }

    for (int q = 0; q < 4; ++q) {
      int target_row = ev.most_recent_row[q];
      if (target_row < 0) continue;

      bool real_is_term = ev.real_endpoint->IsTerminal();
      std::vector<float> obs_real(kObsDim, 0.0f);
      if (!real_is_term) {
        const auto* dune_real =
            dynamic_cast<const dune_imperium::DuneImperiumState*>(
                ev.real_endpoint.get());
        dune_real->InformationStateTensorWithAppendix(
            q, dune_imperium::MarketAppendixMode::kFullPublicInformationV3,
            absl::MakeSpan(obs_real), evaluator->DeckPoolSemantics());
      }

      std::vector<bool> matches_real(8, false);
      std::vector<std::vector<float>> obs_alt(8, std::vector<float>(kObsDim, 0.0f));
      for (int a = 0; a < 8; ++a) {
        bool alt_is_term = alt_endpoints[a]->IsTerminal();
        if (real_is_term != alt_is_term) {
          matches_real[a] = false;
        } else if (real_is_term) {
          float r_real = open_spiel::ComputeTerminalReturns(
              *ev.real_endpoint, "placement", 0.0)[q];
          float r_alt = open_spiel::ComputeTerminalReturns(
              *alt_endpoints[a], "placement", 0.0)[q];
          matches_real[a] = (r_real == r_alt);
        } else {
          const auto* dune_alt =
              dynamic_cast<const dune_imperium::DuneImperiumState*>(
                  alt_endpoints[a].get());
          dune_alt->InformationStateTensorWithAppendix(
              q, dune_imperium::MarketAppendixMode::kFullPublicInformationV3,
              absl::MakeSpan(obs_alt[a]), evaluator->DeckPoolSemantics());
          matches_real[a] =
              (std::memcmp(obs_real.data(), obs_alt[a].data(),
                           kObsDim * sizeof(float)) == 0);
        }
      }

      bool is_on_turn = (q == ev.actor_before_block);
      bool v_real_free = (!real_is_term && ev.real_endpoint->CurrentPlayer() == q);

      bool v_real_computed = false;
      float V_real = 0.0f;
      std::vector<bool> v_alt_computed(8, false);
      std::vector<float> V_alt(8, 0.0f);

      auto get_v_real = [&]() -> float {
        if (!v_real_computed) {
          if (real_is_term) {
            V_real = std::clamp(
                static_cast<float>(open_spiel::ComputeTerminalReturns(
                    *ev.real_endpoint, "placement", 0.0)[q]) /
                    4.0f,
                -1.0f, 1.0f);
          } else {
            V_real = evaluator->EvaluateValues(obs_real).value;
            game_data.c_total_val_only_calls++;
          }
          v_real_computed = true;
        }
        return V_real;
      };

      auto get_v_alt = [&](int a) -> float {
        if (!v_alt_computed[a]) {
          if (alt_endpoints[a]->IsTerminal()) {
            V_alt[a] = std::clamp(
                static_cast<float>(open_spiel::ComputeTerminalReturns(
                    *alt_endpoints[a], "placement", 0.0)[q]) /
                    4.0f,
                -1.0f, 1.0f);
          } else {
            V_alt[a] = evaluator->EvaluateValues(obs_alt[a]).value;
            game_data.c_total_val_only_calls++;
          }
          v_alt_computed[a] = true;
        }
        return V_alt[a];
      };

      const int k_vals[3] = {2, 4, 8};
      for (int k_idx = 0; k_idx < 3; ++k_idx) {
        int k = k_vals[k_idx];
        bool all_match = true;
        for (int a = 0; a < k; ++a) {
          if (!matches_real[a]) {
            all_match = false;
            break;
          }
        }
        float luck_C_k = 0.0f;
        if (!all_match) {
          float vr = get_v_real();
          float sum_va = 0.0f;
          for (int a = 0; a < k; ++a) {
            sum_va += get_v_alt(a);
          }
          luck_C_k = vr - (1.0f / static_cast<float>(k)) * sum_va;
        }

        if (is_on_turn) {
          game_data.rows[target_row].luck_C_on[k_idx] += luck_C_k;
          game_data.rows[target_row].c_on_extra_val_calls[k_idx] +=
              (all_match ? 0 : (k + (v_real_free ? 0 : 1)));
          game_data.rows[target_row].c_on_total_val_calls[k_idx] +=
              (all_match ? 0 : (1 + k));
        } else {
          game_data.rows[target_row].luck_C_off[k_idx] += luck_C_k;
          game_data.rows[target_row].c_off_extra_val_calls[k_idx] +=
              (all_match ? 0 : (k + (v_real_free ? 0 : 1)));
        }
      }
    }
    ev.chance_block_start.reset();
    ev.real_endpoint.reset();
  }
  game_data.chance_events.clear();
}

// ---------------------------------------------------------------------------
// Step 1 Backward Pass, Betas, Bootstrap
// ---------------------------------------------------------------------------

struct ConfigDef {
  int config_id = 0;
  double lambda = 1.0;
  int k = 2;
  int k_idx = 0; // 0: 2, 1: 4, 2: 8
  std::string variant; // "F", "C_on", "C_onoff"
  std::string skip_rule; // "AllRounds", "R1_6_Only", "R1_3_Only"

  // Cost metrics
  double extra_opp_policy_calls_per_game = 0.0;
  double extra_val_only_calls_per_game = 0.0;
  double extra_ms_per_game = 0.0;
  double training_step_ms_per_game = 173.8;
  double training_step_slowdown_factor = 1.0;
  double pct_training_slowdown = 0.0;
};

std::vector<ConfigDef> GenerateConfigs() {
  std::vector<ConfigDef> configs;
  int id = 0;
  for (double lam : {0.95, 0.98, 1.0}) {
    for (int k_idx = 0; k_idx < 3; ++k_idx) {
      int k = (k_idx == 0) ? 2 : (k_idx == 1 ? 4 : 8);
      for (const std::string& var : {"F", "C_on", "C_onoff"}) {
        for (const std::string& skip : {"AllRounds", "R1_6_Only", "R1_3_Only"}) {
          ConfigDef c;
          c.config_id = id++;
          c.lambda = lam;
          c.k = k;
          c.k_idx = k_idx;
          c.variant = var;
          c.skip_rule = skip;
          configs.push_back(c);
        }
      }
    }
  }
  return configs;
}

struct FittedBeta {
  double global_beta = 0.0;
  double global_beta_off = 0.0; // for C_onoff
  double band_beta[3] = {0.0, 0.0, 0.0};
  double band_beta_off[3] = {0.0, 0.0, 0.0}; // for C_onoff
};

struct StratumDef {
  int stratum_id = 0;
  std::string name;
  std::function<bool(const Step1Row&)> predicate;
};

std::vector<StratumDef> GenerateStrata() {
  std::vector<StratumDef> strata;
  strata.push_back({0, "All", [](const Step1Row&) { return true; }});
  strata.push_back({1, "R1-3", [](const Step1Row& r) { return r.round_band == 0; }});
  strata.push_back({2, "R4-6", [](const Step1Row& r) { return r.round_band == 1; }});
  strata.push_back({3, "R7+", [](const Step1Row& r) { return r.round_band == 2; }});

  // Decision roles
  strata.push_back({4, "Role:AgentPrimary", [](const Step1Row& r) {
    return r.decision_role == static_cast<int>(open_spiel::DuneDecisionRole::kAgentPrimary);
  }});
  strata.push_back({5, "Role:AgentContinuation", [](const Step1Row& r) {
    return r.decision_role == static_cast<int>(open_spiel::DuneDecisionRole::kAgentContinuation);
  }});
  strata.push_back({6, "Role:Purchase", [](const Step1Row& r) {
    return r.decision_role == static_cast<int>(open_spiel::DuneDecisionRole::kPurchase);
  }});
  strata.push_back({7, "Role:CombatIntrigue", [](const Step1Row& r) {
    return r.decision_role == static_cast<int>(open_spiel::DuneDecisionRole::kCombatIntrigue);
  }});
  strata.push_back({8, "Role:OtherOptional", [](const Step1Row& r) {
    return r.decision_role == static_cast<int>(open_spiel::DuneDecisionRole::kOtherOptional);
  }});
  strata.push_back({9, "Role:LeaderSelection", [](const Step1Row& r) {
    return r.decision_role == static_cast<int>(open_spiel::DuneDecisionRole::kLeaderSelection);
  }});
  return strata;
}

// ---------------------------------------------------------------------------
// Step 2 Plasticity Audit Implementation
// ---------------------------------------------------------------------------

struct ModelGradGroupNorms {
  double block_group_0_7 = 0.0;
  double block_group_8_11 = 0.0;
  double block_group_12_15 = 0.0;
  double input_layer = 0.0;
  double policy_head = 0.0;
  double value_head = 0.0;
  double semantic_scorer = 0.0;
  double total_model = 0.0;
};

double ComputeGradNorm(const std::vector<torch::Tensor>& params) {
  double sum_sq = 0.0;
  for (const auto& p : params) {
    if (p.grad().defined()) {
      sum_sq += p.grad().detach().pow(2).sum().item<double>();
    }
  }
  return std::sqrt(sum_sq);
}

ModelGradGroupNorms ExtractGradNorms(
    const std::shared_ptr<open_spiel::SharedDunePolicyValueNetImpl>& model) {
  ModelGradGroupNorms norms;
  norms.input_layer = ComputeGradNorm(model->input_layer->parameters());
  norms.policy_head = ComputeGradNorm(model->policy_head->parameters());
  norms.value_head = ComputeGradNorm(model->value_head->parameters());
  if (model->semantic_scorer_) {
    norms.semantic_scorer = ComputeGradNorm(model->semantic_scorer_->parameters());
  }

  // Group 0-7
  std::vector<torch::Tensor> g0_7;
  for (size_t b = 0; b < std::min<size_t>(8, model->res_blocks.size()); ++b) {
    auto bp = model->res_blocks[b]->parameters();
    g0_7.insert(g0_7.end(), bp.begin(), bp.end());
  }
  norms.block_group_0_7 = ComputeGradNorm(g0_7);

  // Group 8-11
  std::vector<torch::Tensor> g8_11;
  for (size_t b = 8; b < std::min<size_t>(12, model->res_blocks.size()); ++b) {
    auto bp = model->res_blocks[b]->parameters();
    g8_11.insert(g8_11.end(), bp.begin(), bp.end());
  }
  norms.block_group_8_11 = ComputeGradNorm(g8_11);

  // Group 12-15
  if (model->res_blocks.size() >= 16) {
    std::vector<torch::Tensor> g12_15;
    for (size_t b = 12; b < std::min<size_t>(16, model->res_blocks.size()); ++b) {
      auto bp = model->res_blocks[b]->parameters();
      g12_15.insert(g12_15.end(), bp.begin(), bp.end());
    }
    norms.block_group_12_15 = ComputeGradNorm(g12_15);
  }

  norms.total_model = ComputeGradNorm(model->parameters());
  return norms;
}

struct DormantStats {
  double dormant_tau0 = 0.0;
  double dormant_tau001 = 0.0;
  double dormant_tau01 = 0.0;
};

struct CheckpointPlasticityResult {
  std::string name;
  std::string short_name;
  std::string path;
  int num_blocks = 16;

  // Dormant fractions: block groups (0-7, 8-11, 12-15)
  // For each group: relu1, relu2, combined
  DormantStats group_dormant[3][3]; // [group 0..2][0: relu1, 1: relu2, 2: combined]
  std::vector<double> block_relu1_tau001;
  std::vector<double> block_relu2_tau001;

  // Trunk SVD
  double effective_rank = 0.0;
  double stable_rank = 0.0;
  std::vector<double> top_singular_values;

  // Gradient norms
  ModelGradGroupNorms grad_pass_a_norm_adv;
  ModelGradGroupNorms grad_pass_a_raw_adv;
  ModelGradGroupNorms grad_pass_b_raw;
  ModelGradGroupNorms grad_pass_b_scaled;
  ModelGradGroupNorms grad_pass_c_total;
};

CheckpointPlasticityResult EvaluateCheckpointPlasticity(
    const std::string& name, const std::string& short_name,
    const std::string& path, int num_blocks,
    const std::vector<SavedState>& states, torch::Device device) {
  std::cout << "\n======================================================\n";
  std::cout << "[Step 2] Evaluating Plasticity: " << name
            << " (" << num_blocks << " blocks) on " << device << "\n";
  std::cout << "Path: " << path << "\n";

  CheckpointPlasticityResult result;
  result.name = name;
  result.short_name = short_name;
  result.path = path;
  result.num_blocks = num_blocks;

  auto model = std::make_shared<open_spiel::SharedDunePolicyValueNetImpl>(
      kObsDim, kHiddenDim, kActionDim, num_blocks, /*use_nonlinear=*/false,
      /*with_aux_heads=*/false, /*head_init_seed=*/0,
      /*with_semantic_scorer=*/true, /*enable_zone_encoder=*/false,
      "semantic_action_v3");

  open_spiel::LoadModelCheckpointRobust(model, path, device);
  model->to(device);
  model->eval();

  const int64_t N = static_cast<int64_t>(states.size());

  // -------------------------------------------------------------------------
  // Part 1: Dormant Activations and Trunk SVD
  // -------------------------------------------------------------------------
  std::vector<torch::Tensor> sum_abs_h1(num_blocks);
  std::vector<torch::Tensor> sum_abs_h2(num_blocks);
  for (int b = 0; b < num_blocks; ++b) {
    sum_abs_h1[b] = torch::zeros({kHiddenDim}, torch::kFloat64);
    sum_abs_h2[b] = torch::zeros({kHiddenDim}, torch::kFloat64);
  }

  torch::Tensor H_matrix = torch::empty({N, kHiddenDim}, torch::kFloat64);

  const int64_t chunk_size = 256;
  {
    torch::NoGradGuard no_grad;
    for (int64_t start = 0; start < N; start += chunk_size) {
      int64_t end = std::min(start + chunk_size, N);
      int64_t B = end - start;

      torch::Tensor obs_batch = torch::empty({B, kObsDim}, torch::kFloat32);
      float* obs_ptr = obs_batch.data_ptr<float>();
      for (int64_t i = 0; i < B; ++i) {
        std::memcpy(obs_ptr + i * kObsDim, states[start + i].obs.data(),
                    kObsDim * sizeof(float));
      }
      obs_batch = obs_batch.to(device);

      torch::Tensor x = torch::relu(model->input_layer->forward(obs_batch));
      for (int b = 0; b < num_blocks; ++b) {
        auto& block = model->res_blocks[b];
        torch::Tensor h1 = torch::relu(block->ln1->forward(block->fc1->forward(x)));
        torch::Tensor h2 = torch::relu(block->ln2->forward(block->fc2->forward(h1)) + x);

        sum_abs_h1[b] += h1.abs().to(torch::kFloat64).to(torch::kCPU).sum(0);
        sum_abs_h2[b] += h2.abs().to(torch::kFloat64).to(torch::kCPU).sum(0);
        x = h2;
      }
      H_matrix.narrow(0, start, B).copy_(x.to(torch::kFloat64).to(torch::kCPU));
    }
  }

  // Compute dormant counts
  std::vector<std::vector<double>> dormant_u_h1(num_blocks, std::vector<double>(3, 0.0));
  std::vector<std::vector<double>> dormant_u_h2(num_blocks, std::vector<double>(3, 0.0));

  result.block_relu1_tau001.resize(num_blocks);
  result.block_relu2_tau001.resize(num_blocks);

  for (int b = 0; b < num_blocks; ++b) {
    torch::Tensor m_u1 = sum_abs_h1[b] / static_cast<double>(N);
    double M1 = m_u1.mean().item<double>();
    const double* p1 = m_u1.data_ptr<double>();

    torch::Tensor m_u2 = sum_abs_h2[b] / static_cast<double>(N);
    double M2 = m_u2.mean().item<double>();
    const double* p2 = m_u2.data_ptr<double>();

    const double taus[3] = {0.0, 0.01, 0.1};
    for (int t = 0; t < 3; ++t) {
      double thresh1 = taus[t] * M1;
      double thresh2 = taus[t] * M2;
      int cnt1 = 0;
      int cnt2 = 0;
      for (int u = 0; u < kHiddenDim; ++u) {
        if (p1[u] <= thresh1) cnt1++;
        if (p2[u] <= thresh2) cnt2++;
      }
      dormant_u_h1[b][t] = cnt1;
      dormant_u_h2[b][t] = cnt2;
    }
    result.block_relu1_tau001[b] = dormant_u_h1[b][1] / static_cast<double>(kHiddenDim);
    result.block_relu2_tau001[b] = dormant_u_h2[b][1] / static_cast<double>(kHiddenDim);
  }

  // Aggregate across groups: 0-7, 8-11, 12-15
  auto aggregate_group = [&](int b_start, int b_end, int g_idx) {
    int n_blks = b_end - b_start;
    if (n_blks <= 0) return;
    for (int t = 0; t < 3; ++t) {
      double c1 = 0;
      double c2 = 0;
      for (int b = b_start; b < b_end; ++b) {
        c1 += dormant_u_h1[b][t];
        c2 += dormant_u_h2[b][t];
      }
      double d_units = static_cast<double>(n_blks * kHiddenDim);
      if (t == 0) {
        result.group_dormant[g_idx][0].dormant_tau0 = c1 / d_units;
        result.group_dormant[g_idx][1].dormant_tau0 = c2 / d_units;
        result.group_dormant[g_idx][2].dormant_tau0 = (c1 + c2) / (2.0 * d_units);
      } else if (t == 1) {
        result.group_dormant[g_idx][0].dormant_tau001 = c1 / d_units;
        result.group_dormant[g_idx][1].dormant_tau001 = c2 / d_units;
        result.group_dormant[g_idx][2].dormant_tau001 = (c1 + c2) / (2.0 * d_units);
      } else if (t == 2) {
        result.group_dormant[g_idx][0].dormant_tau01 = c1 / d_units;
        result.group_dormant[g_idx][1].dormant_tau01 = c2 / d_units;
        result.group_dormant[g_idx][2].dormant_tau01 = (c1 + c2) / (2.0 * d_units);
      }
    }
  };

  aggregate_group(0, std::min(8, num_blocks), 0);
  if (num_blocks > 8) {
    aggregate_group(8, std::min(12, num_blocks), 1);
  }
  if (num_blocks > 12) {
    aggregate_group(12, std::min(16, num_blocks), 2);
  }

  // Trunk SVD
  torch::Tensor H_c = H_matrix - H_matrix.mean(0, /*keepdim=*/true);
  torch::Tensor sigma = torch::linalg_svdvals(H_c);
  double sigma_sum = sigma.sum().item<double>();
  double sigma_sq_sum = sigma.pow(2).sum().item<double>();
  double sigma_0 = sigma[0].item<double>();

  double entropy_rank = 0.0;
  const double* s_ptr = sigma.data_ptr<double>();
  int64_t n_sv = sigma.size(0);
  for (int64_t i = 0; i < n_sv; ++i) {
    double p_i = s_ptr[i] / sigma_sum;
    if (p_i > 1e-12) {
      entropy_rank -= p_i * std::log(p_i);
    }
  }
  result.effective_rank = std::exp(entropy_rank);
  result.stable_rank = sigma_sq_sum / (sigma_0 * sigma_0);
  for (int64_t i = 0; i < std::min<int64_t>(10, n_sv); ++i) {
    result.top_singular_values.push_back(s_ptr[i]);
  }

  // -------------------------------------------------------------------------
  // Part 2: Sampling and Gradient Norms
  // -------------------------------------------------------------------------
  std::vector<open_spiel::Action> chosen_actions(N);
  std::vector<float> old_log_probs(N);
  std::vector<float> old_values(N);
  std::vector<float> raw_adv(N);

  {
    torch::NoGradGuard no_grad;
    for (int64_t start = 0; start < N; start += chunk_size) {
      int64_t end = std::min(start + chunk_size, N);
      int64_t B = end - start;

      torch::Tensor obs_batch = torch::empty({B, kObsDim}, torch::kFloat32);
      float* obs_ptr = obs_batch.data_ptr<float>();
      std::vector<const dune_semantic::CandidateActionData*> chunk_cands(B);

      for (int64_t i = 0; i < B; ++i) {
        std::memcpy(obs_ptr + i * kObsDim, states[start + i].obs.data(),
                    kObsDim * sizeof(float));
        chunk_cands[i] = &states[start + i].cand_data;
      }
      obs_batch = obs_batch.to(device);

      auto outputs = model->forward(obs_batch);
      if (model->semantic_scorer_) {
        dune_semantic::ApplySemanticScorerBatch(
            model->semantic_scorer_, outputs.trunk, chunk_cands, outputs.logits,
            device);
      }

      torch::Tensor logits_cpu = outputs.logits.to(torch::kCPU);
      torch::Tensor values_cpu = outputs.values.to(torch::kCPU);
      const float* val_ptr = values_cpu.data_ptr<float>();

      for (int64_t i = 0; i < B; ++i) {
        int64_t s_idx = start + i;
        const auto& legal_acts = states[s_idx].legal_actions;
        std::vector<float> l_vec(legal_acts.size());
        const float* full_l = logits_cpu[i].data_ptr<float>();
        for (size_t a = 0; a < legal_acts.size(); ++a) {
          l_vec[a] = full_l[legal_acts[a]];
        }
        open_spiel::CenterAndCapLegalLogits(l_vec, legal_acts, 10.0f);

        std::mt19937_64 rng_s = dune_seed::MakeRng64(dune_seed::DeriveSeed(
            34350, 0x504C415354ULL, static_cast<uint64_t>(s_idx), 0));
        std::vector<double> probs;
        auto sample = open_spiel::SamplePolicyDistribution(
            &rng_s, l_vec, legal_acts, &probs);

        chosen_actions[s_idx] = sample.action;
        old_log_probs[s_idx] = sample.chosen_log_probability;
        old_values[s_idx] = val_ptr[i];
        raw_adv[s_idx] = states[s_idx].mc_return - old_values[s_idx];
      }
    }
  }

  // Advantage normalization
  double adv_sum = 0.0;
  for (int64_t s = 0; s < N; ++s) adv_sum += raw_adv[s];
  double adv_mean = adv_sum / static_cast<double>(N);

  double adv_sq_sum = 0.0;
  for (int64_t s = 0; s < N; ++s) {
    adv_sq_sum += (raw_adv[s] - adv_mean) * (raw_adv[s] - adv_mean);
  }
  double adv_std = std::sqrt(adv_sq_sum / static_cast<double>(N)) + 1e-8;

  std::vector<float> norm_adv(N);
  for (int64_t s = 0; s < N; ++s) {
    norm_adv[s] = static_cast<float>((raw_adv[s] - adv_mean) / adv_std);
  }

  // Gradient accumulation passes in chunks of 250
  const int64_t grad_chunk_size = 250;

  auto run_accumulation_pass = [&](int pass_type) -> ModelGradGroupNorms {
    model->zero_grad();

    for (int64_t start = 0; start < N; start += grad_chunk_size) {
      int64_t end = std::min(start + grad_chunk_size, N);
      int64_t B = end - start;
      double scale = static_cast<double>(B) / static_cast<double>(N);

      torch::Tensor obs_batch = torch::empty({B, kObsDim}, torch::kFloat32);
      torch::Tensor mask_batch = torch::zeros({B, kActionDim}, torch::kBool);
      torch::Tensor act_batch = torch::empty({B}, torch::kInt64);
      torch::Tensor olp_batch = torch::empty({B}, torch::kFloat32);
      torch::Tensor adv_norm_batch = torch::empty({B}, torch::kFloat32);
      torch::Tensor adv_raw_batch = torch::empty({B}, torch::kFloat32);
      torch::Tensor g_batch = torch::empty({B}, torch::kFloat32);

      float* obs_ptr = obs_batch.data_ptr<float>();
      bool* mask_ptr = mask_batch.data_ptr<bool>();
      int64_t* act_ptr = act_batch.data_ptr<int64_t>();
      float* olp_ptr = olp_batch.data_ptr<float>();
      float* adv_n_ptr = adv_norm_batch.data_ptr<float>();
      float* adv_r_ptr = adv_raw_batch.data_ptr<float>();
      float* g_ptr = g_batch.data_ptr<float>();

      std::vector<const dune_semantic::CandidateActionData*> chunk_cands(B);

      for (int64_t i = 0; i < B; ++i) {
        int64_t s_idx = start + i;
        std::memcpy(obs_ptr + i * kObsDim, states[s_idx].obs.data(),
                    kObsDim * sizeof(float));
        chunk_cands[i] = &states[s_idx].cand_data;

        for (open_spiel::Action a : states[s_idx].legal_actions) {
          if (a >= 0 && a < kActionDim) {
            mask_ptr[i * kActionDim + a] = true;
          }
        }
        act_ptr[i] = chosen_actions[s_idx];
        olp_ptr[i] = old_log_probs[s_idx];
        adv_n_ptr[i] = norm_adv[s_idx];
        adv_r_ptr[i] = raw_adv[s_idx];
        g_ptr[i] = states[s_idx].mc_return;
      }

      obs_batch = obs_batch.to(device);
      mask_batch = mask_batch.to(device);
      act_batch = act_batch.to(device);
      olp_batch = olp_batch.to(device);
      adv_norm_batch = adv_norm_batch.to(device);
      adv_raw_batch = adv_raw_batch.to(device);
      g_batch = g_batch.to(device);

      auto outputs = model->forward(obs_batch);
      if (model->semantic_scorer_) {
        dune_semantic::ApplySemanticScorerBatch(
            model->semantic_scorer_, outputs.trunk, chunk_cands, outputs.logits,
            device);
      }

      torch::Tensor centered =
          open_spiel::CenterAndCapLogitsTensor(outputs.logits, mask_batch, 10.0f);
      torch::Tensor masked_logits =
          centered.masked_fill(mask_batch.logical_not(), -1e9f);
      torch::Tensor log_probs = torch::log_softmax(masked_logits, -1);
      torch::Tensor probs = torch::softmax(masked_logits, -1);
      torch::Tensor sel_lp =
          log_probs.gather(1, act_batch.unsqueeze(1)).squeeze(1);
      torch::Tensor ratio = torch::exp(sel_lp - olp_batch);
      torch::Tensor v = outputs.values.squeeze(1);

      torch::Tensor chunk_loss;
      if (pass_type == 0) {
        // Pass A1: policy_loss with normalized adv
        chunk_loss = -(adv_norm_batch * ratio).mean() * scale;
      } else if (pass_type == 1) {
        // Pass A2: policy_loss with raw unnormalized adv
        chunk_loss = -(adv_raw_batch * ratio).mean() * scale;
      } else if (pass_type == 2) {
        // Pass B: value_loss raw = 0.5 * (v - G)^2
        chunk_loss = (0.5f * (v - g_batch).pow(2)).mean() * scale;
      } else if (pass_type == 3) {
        // Pass C: total_loss = policy_loss + 0.5 * value_loss - 0.01 * entropy
        torch::Tensor p_loss = -(adv_norm_batch * ratio).mean();
        torch::Tensor v_loss = (0.5f * (v - g_batch).pow(2)).mean();
        torch::Tensor masked_p = torch::where(mask_batch, probs, torch::zeros_like(probs));
        torch::Tensor masked_lp = torch::where(mask_batch, log_probs, torch::zeros_like(log_probs));
        torch::Tensor entropy = -(masked_p * masked_lp).sum(-1).mean();
        chunk_loss = (p_loss + 0.5f * v_loss - 0.01f * entropy) * scale;
      }

      chunk_loss.backward();
    }

    return ExtractGradNorms(model);
  };

  result.grad_pass_a_norm_adv = run_accumulation_pass(0);
  result.grad_pass_a_raw_adv = run_accumulation_pass(1);
  result.grad_pass_b_raw = run_accumulation_pass(2);

  // Scaled value_loss grad norm is exactly 0.5 * raw
  result.grad_pass_b_scaled.block_group_0_7 = 0.5 * result.grad_pass_b_raw.block_group_0_7;
  result.grad_pass_b_scaled.block_group_8_11 = 0.5 * result.grad_pass_b_raw.block_group_8_11;
  result.grad_pass_b_scaled.block_group_12_15 = 0.5 * result.grad_pass_b_raw.block_group_12_15;
  result.grad_pass_b_scaled.input_layer = 0.5 * result.grad_pass_b_raw.input_layer;
  result.grad_pass_b_scaled.policy_head = 0.5 * result.grad_pass_b_raw.policy_head;
  result.grad_pass_b_scaled.value_head = 0.5 * result.grad_pass_b_raw.value_head;
  result.grad_pass_b_scaled.semantic_scorer = 0.5 * result.grad_pass_b_raw.semantic_scorer;
  result.grad_pass_b_scaled.total_model = 0.5 * result.grad_pass_b_raw.total_model;

  result.grad_pass_c_total = run_accumulation_pass(3);

  std::cout << "Done evaluating " << name << ".\n";
  return result;
}

// ---------------------------------------------------------------------------
// R7-44 Critic Architecture Dataset Collection
// ---------------------------------------------------------------------------

constexpr int kR744DefaultSampledRowsPerGame = 32;

#pragma pack(push, 1)
struct R744RowMeta {
  int32_t game_id = 0;
  int8_t player = 0;
  int8_t round = 0;
  int8_t round_band = 0;
  int8_t decision_role = 0;
  float u34350_value = 0.0f;
  float target = 0.0f;
};
#pragma pack(pop)
static_assert(sizeof(R744RowMeta) == 16, "R744RowMeta size must be exactly 16 bytes");

struct R744SampledRow {
  std::array<uint16_t, kObsDim> obs_f16;
  R744RowMeta meta;
  float gae095_target = 0.0f;
};

struct R744GameRecord {
  int32_t game_index = 0;
  uint64_t episode_id = 0;
  int32_t final_round = 1;
  int32_t total_rows = 0;
  int32_t eligible_rows = 0;
  std::vector<R744SampledRow> sampled_rows;
};

void RunCollectR744CriticData(
    const open_spiel::Game& game,
    int num_games,
    uint64_t master_seed,
    uint64_t episode_start,
    int num_threads,
    int eval_batch_size,
    int eval_timeout_ms,
    const std::string& output_dir,
    const std::string& ckpt_path,
    int rows_per_game = kR744DefaultSampledRowsPerGame,
    int game_id_offset = 0) {
  std::cout << "\n>>> Starting R7-44 Critic Architecture Dataset Collection <<<\n";
  std::cout << "Target games: " << num_games << " | Master Seed: " << master_seed
            << " | Episode Start: " << episode_start << " | Threads: " << num_threads << "\n";
  std::cout << "Rows / Game: " << rows_per_game << " | Game ID Offset: " << game_id_offset << "\n";
  std::cout << "Output Directory: " << output_dir << "\n";

  std::string resolved_ckpt = ckpt_path;
  if (!std::filesystem::exists(resolved_ckpt)) {
    size_t pos = resolved_ckpt.find("/Storage/");
    if (pos != std::string::npos) {
      std::string alt = resolved_ckpt;
      alt.replace(pos, 9, "/Storage1/");
      if (std::filesystem::exists(alt)) {
        resolved_ckpt = alt;
      }
    }
  }

  torch::Device device(torch::kCUDA);
  auto model = std::make_shared<open_spiel::SharedDunePolicyValueNetImpl>(
      kObsDim, kHiddenDim, kActionDim, 16, /*use_nonlinear=*/false,
      /*with_aux_heads=*/false, /*head_init_seed=*/0,
      /*with_semantic_scorer=*/true, /*enable_zone_encoder=*/false,
      "semantic_action_v3");

  std::cout << "Loading U34350 live checkpoint from: " << resolved_ckpt << "\n";
  open_spiel::LoadModelCheckpointRobust(model, resolved_ckpt, device);
  model->to(device);
  model->eval();

  std::shared_mutex sync_mutex;
  auto evaluator = std::make_shared<open_spiel::BatchedEvaluator>(
      model, eval_batch_size, eval_timeout_ms, device, &sync_mutex,
      /*logit_cap=*/10.0f, /*device_synchronize=*/true,
      /*high_priority_stream=*/false, /*emit_batch_membership=*/false,
      /*rollout_amp=*/false, /*allow_tf32=*/false);

  std::vector<R744GameRecord> records(num_games);
  std::atomic<int> next_game_idx{0};
  std::atomic<int> completed_games{0};

  auto wall_start = std::chrono::steady_clock::now();

  auto worker_fn = [&]() {
    while (true) {
      int g = next_game_idx.fetch_add(1);
      if (g >= num_games) break;
      uint64_t ep_id = episode_start + static_cast<uint64_t>(g);

      // Simulate trajectory without keeping intermediate clone states or chance events
      Step1GameData gd = SimulateRealTrajectory(
          g, ep_id, master_seed, game, evaluator, /*keep_states_and_chance=*/false);

      // Terminal placement returns without penalties
      float player_targets[4] = {0.0f, 0.0f, 0.0f, 0.0f};
      for (int p = 0; p < 4; ++p) {
        if (!gd.player_rows[p].empty()) {
          int last_idx = gd.player_rows[p].back();
          player_targets[p] = gd.rows[last_idx].terminal_return / 4.0f;
        }
      }

      // Find all policy-loss eligible rows (actions.size() > 1)
      std::vector<int> eligible_indices;
      eligible_indices.reserve(gd.rows.size());
      for (size_t r = 0; r < gd.rows.size(); ++r) {
        if (gd.rows[r].policy_loss_eligible) {
          eligible_indices.push_back(static_cast<int>(r));
        }
      }

      if (eligible_indices.size() < static_cast<size_t>(rows_per_game)) {
        open_spiel::SpielFatalError(absl::StrFormat(
            "Game %d (ep_id %llu) had only %zu eligible rows (< %d)",
            g, static_cast<unsigned long long>(ep_id),
            eligible_indices.size(), rows_per_game));
      }

      // Sample rows_per_game row indices uniformly at random without replacement
      std::mt19937_64 row_rng = dune_seed::MakeRng64(
          dune_seed::DeriveSeed(master_seed, 0x52373434ULL, ep_id, 0));
      std::shuffle(eligible_indices.begin(), eligible_indices.end(), row_rng);

      std::vector<int> sampled_indices(
          eligible_indices.begin(),
          eligible_indices.begin() + rows_per_game);
      std::sort(sampled_indices.begin(), sampled_indices.end());

      // Backward pass for GAE(gamma=1.0, lambda=0.95)
      std::vector<float> row_gae095(gd.rows.size(), 0.0f);
      const float gae_lambda = 0.95f;
      for (int p = 0; p < 4; ++p) {
        const auto& p_rows = gd.player_rows[p];
        int M = static_cast<int>(p_rows.size());
        if (M == 0) continue;

        float next_value = 0.0f;
        float next_gae = 0.0f;
        for (int i = M - 1; i >= 0; --i) {
          int r_idx = p_rows[i];
          float v_cur = gd.rows[r_idx].value;
          float reward = (i == M - 1) ? player_targets[p] : 0.0f;
          float delta = reward + next_value - v_cur;
          float adv = delta + gae_lambda * next_gae;
          row_gae095[r_idx] = v_cur + adv;
          next_value = v_cur;
          next_gae = adv;
        }
      }

      records[g].game_index = static_cast<int32_t>(g);
      records[g].episode_id = ep_id;
      records[g].final_round = static_cast<int32_t>(gd.final_round);
      records[g].total_rows = static_cast<int32_t>(gd.rows.size());
      records[g].eligible_rows = static_cast<int32_t>(eligible_indices.size());
      records[g].sampled_rows.resize(rows_per_game);

      for (int k = 0; k < rows_per_game; ++k) {
        int r = sampled_indices[k];
        const auto& row = gd.rows[r];
        int p = row.player;
        float target = (p >= 0 && p < 4) ? player_targets[p] : 0.0f;

        records[g].sampled_rows[k].meta.game_id = static_cast<int32_t>(game_id_offset + g);
        records[g].sampled_rows[k].meta.player = static_cast<int8_t>(row.player);
        records[g].sampled_rows[k].meta.round = static_cast<int8_t>(row.round);
        records[g].sampled_rows[k].meta.round_band = static_cast<int8_t>(row.round_band);
        records[g].sampled_rows[k].meta.decision_role = static_cast<int8_t>(row.decision_role);
        records[g].sampled_rows[k].meta.u34350_value = row.value;
        records[g].sampled_rows[k].meta.target = target;
        records[g].sampled_rows[k].gae095_target = row_gae095[r];

        for (int64_t i = 0; i < kObsDim; ++i) {
          records[g].sampled_rows[k].obs_f16[i] = c10::Half(row.obs[i]).x;
        }
      }

      int done = completed_games.fetch_add(1) + 1;
      if (done % std::max(1, num_games / 10) == 0 || done == num_games) {
        std::cout << "  [R7-44 Data Worker] Completed " << done << " / "
                  << num_games << " games\n";
      }
    }
  };

  std::vector<std::thread> workers;
  workers.reserve(num_threads);
  for (int t = 0; t < num_threads; ++t) {
    workers.emplace_back(worker_fn);
  }
  for (auto& w : workers) w.join();

  auto wall_end = std::chrono::steady_clock::now();
  double wall_seconds =
      std::chrono::duration<double>(wall_end - wall_start).count();

  std::cout << "\nWriting binary dataset and summary to " << output_dir << "...\n";

  // Write obs_f16.bin
  std::string obs_path = output_dir + "/obs_f16.bin";
  std::ofstream obs_out(obs_path, std::ios::binary);
  if (!obs_out) {
    open_spiel::SpielFatalError("Failed to open " + obs_path + " for writing");
  }
  for (int g = 0; g < num_games; ++g) {
    for (int k = 0; k < rows_per_game; ++k) {
      obs_out.write(
          reinterpret_cast<const char*>(records[g].sampled_rows[k].obs_f16.data()),
          kObsDim * sizeof(uint16_t));
    }
  }
  obs_out.close();

  // Write metadata.bin
  std::string meta_path = output_dir + "/metadata.bin";
  std::ofstream meta_out(meta_path, std::ios::binary);
  if (!meta_out) {
    open_spiel::SpielFatalError("Failed to open " + meta_path + " for writing");
  }
  for (int g = 0; g < num_games; ++g) {
    for (int k = 0; k < rows_per_game; ++k) {
      meta_out.write(
          reinterpret_cast<const char*>(&records[g].sampled_rows[k].meta),
          sizeof(R744RowMeta));
    }
  }
  meta_out.close();

  // Write gae095_targets.bin
  std::string gae_path = output_dir + "/gae095_targets.bin";
  std::ofstream gae_out(gae_path, std::ios::binary);
  if (!gae_out) {
    open_spiel::SpielFatalError("Failed to open " + gae_path + " for writing");
  }
  for (int g = 0; g < num_games; ++g) {
    for (int k = 0; k < rows_per_game; ++k) {
      gae_out.write(
          reinterpret_cast<const char*>(&records[g].sampled_rows[k].gae095_target),
          sizeof(float));
    }
  }
  gae_out.close();

  // Verify file sizes
  uintmax_t obs_size = std::filesystem::file_size(obs_path);
  uintmax_t expected_obs_size =
      static_cast<uintmax_t>(num_games) * rows_per_game * kObsDim * sizeof(uint16_t);
  if (obs_size != expected_obs_size) {
    open_spiel::SpielFatalError(absl::StrFormat(
        "obs_f16.bin size mismatch: got %llu bytes, expected %llu bytes",
        static_cast<unsigned long long>(obs_size),
        static_cast<unsigned long long>(expected_obs_size)));
  }

  uintmax_t meta_size = std::filesystem::file_size(meta_path);
  uintmax_t expected_meta_size =
      static_cast<uintmax_t>(num_games) * rows_per_game * sizeof(R744RowMeta);
  if (meta_size != expected_meta_size) {
    open_spiel::SpielFatalError(absl::StrFormat(
        "metadata.bin size mismatch: got %llu bytes, expected %llu bytes",
        static_cast<unsigned long long>(meta_size),
        static_cast<unsigned long long>(expected_meta_size)));
  }

  uintmax_t gae_size = std::filesystem::file_size(gae_path);
  uintmax_t expected_gae_size =
      static_cast<uintmax_t>(num_games) * rows_per_game * sizeof(float);
  if (gae_size != expected_gae_size) {
    open_spiel::SpielFatalError(absl::StrFormat(
        "gae095_targets.bin size mismatch: got %llu bytes, expected %llu bytes",
        static_cast<unsigned long long>(gae_size),
        static_cast<unsigned long long>(expected_gae_size)));
  }

  // Compute slice round averages
  auto compute_slice_avg_round = [&](int start_idx, int end_idx) -> double {
    int actual_start = std::min(start_idx, num_games);
    int actual_end = std::min(end_idx, num_games);
    if (actual_end <= actual_start) return 0.0;
    double sum = 0.0;
    for (int i = actual_start; i < actual_end; ++i) {
      sum += records[i].final_round;
    }
    return sum / (actual_end - actual_start);
  };

  double overall_avg_round = compute_slice_avg_round(0, num_games);
  double train_10k_avg_round = compute_slice_avg_round(0, 10000);
  double train_9k_avg_round = compute_slice_avg_round(0, 9000);
  double val_1k_avg_round = compute_slice_avg_round(9000, 10000);
  double test_2k_avg_round = compute_slice_avg_round(10000, 12000);
  double train_2500_avg_round = compute_slice_avg_round(0, 2500);

  int64_t total_rows_all = 0;
  int64_t eligible_rows_all = 0;
  for (int g = 0; g < num_games; ++g) {
    total_rows_all += records[g].total_rows;
    eligible_rows_all += records[g].eligible_rows;
  }
  double avg_total_rows = (num_games > 0)
      ? (static_cast<double>(total_rows_all) / num_games) : 0.0;
  double avg_eligible_rows = (num_games > 0)
      ? (static_cast<double>(eligible_rows_all) / num_games) : 0.0;

  open_spiel::json::Object summary_root;
  summary_root["num_games"] = static_cast<int64_t>(num_games);
  summary_root["master_seed"] = static_cast<int64_t>(master_seed);
  summary_root["episode_start"] = static_cast<int64_t>(episode_start);
  summary_root["rows_per_game"] = static_cast<int64_t>(rows_per_game);
  summary_root["game_id_offset"] = static_cast<int64_t>(game_id_offset);
  summary_root["overall_avg_round_finished"] = overall_avg_round;
  summary_root["train_10k_avg_round"] = train_10k_avg_round;
  summary_root["train_9k_avg_round"] = train_9k_avg_round;
  summary_root["val_1k_avg_round"] = val_1k_avg_round;
  summary_root["test_2k_avg_round"] = test_2k_avg_round;
  summary_root["train_2500_avg_round"] = train_2500_avg_round;
  summary_root["avg_total_rows_per_game"] = avg_total_rows;
  summary_root["avg_eligible_rows_per_game"] = avg_eligible_rows;
  summary_root["wall_seconds"] = wall_seconds;

  open_spiel::json::Array final_rounds_arr;
  final_rounds_arr.reserve(num_games);
  for (int g = 0; g < num_games; ++g) {
    final_rounds_arr.push_back(static_cast<int64_t>(records[g].final_round));
  }
  summary_root["final_rounds"] = std::move(final_rounds_arr);

  std::string summary_path = output_dir + "/games_summary.json";
  std::ofstream json_out(summary_path);
  json_out << open_spiel::json::ToString(open_spiel::json::Value(summary_root), true);
  json_out.close();

  std::cout << "\n========================================================================================\n";
  std::cout << "                 R7-44 CRITIC DATA COLLECTION COMPLETE\n";
  std::cout << "========================================================================================\n";
  std::cout << "Total Games Collected:       " << num_games << "\n";
  std::cout << "Total Rows Sampled:          " << num_games * rows_per_game << " (" << rows_per_game << " per game)\n";
  std::cout << "Game ID Range:               " << game_id_offset << " .. " << (game_id_offset + num_games - 1) << "\n";
  std::cout << "Overall Avg Round Finished:  " << std::fixed << std::setprecision(4) << overall_avg_round << "\n";
  std::cout << "Train 10k Avg Round:         " << train_10k_avg_round << "\n";
  std::cout << "Train 9k Avg Round:          " << train_9k_avg_round << "\n";
  std::cout << "Val 1k Avg Round:            " << val_1k_avg_round << "\n";
  std::cout << "Test 2k Avg Round:           " << test_2k_avg_round << "\n";
  std::cout << "Train 2500 Avg Round:        " << train_2500_avg_round << "\n";
  std::cout << "Avg Total Rows / Game:       " << std::setprecision(2) << avg_total_rows << "\n";
  std::cout << "Avg Eligible Rows / Game:    " << avg_eligible_rows << "\n";
  std::cout << "Wall Time:                   " << std::setprecision(2) << wall_seconds << " s ("
            << (num_games > 0 ? (wall_seconds * 1000.0 / num_games) : 0.0) << " ms/game)\n";
  std::cout << "Wrote obs_f16:               " << obs_path << " (" << obs_size << " bytes)\n";
  std::cout << "Wrote metadata:              " << meta_path << " (" << meta_size << " bytes)\n";
  std::cout << "Wrote gae095_targets:        " << gae_path << " (" << gae_size << " bytes)\n";
  std::cout << "Wrote summary:               " << summary_path << "\n";
  std::cout << "========================================================================================\n\n";
}

} // namespace open_spiel

int main(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);
  using namespace open_spiel;

  std::string mode = absl::GetFlag(FLAGS_mode);
  int num_games = absl::GetFlag(FLAGS_num_games);
  uint64_t master_seed = absl::GetFlag(FLAGS_master_seed);
  uint64_t episode_start = absl::GetFlag(FLAGS_episode_start);
  int num_threads = absl::GetFlag(FLAGS_threads);
  int eval_batch_size = absl::GetFlag(FLAGS_eval_batch_size);
  int eval_timeout_ms = absl::GetFlag(FLAGS_eval_timeout_ms);
  std::string output_dir = absl::GetFlag(FLAGS_output_dir);
  std::string states_file = absl::GetFlag(FLAGS_states_file);
  std::string step2_device_str = absl::GetFlag(FLAGS_step2_device);
  int num_boot_resamples = absl::GetFlag(FLAGS_bootstrap_resamples);
  int r744_rows_per_game = absl::GetFlag(FLAGS_r744_rows_per_game);
  int r744_game_id_offset = absl::GetFlag(FLAGS_r744_game_id_offset);

  std::filesystem::create_directories(output_dir);

  std::cout << "=== Dune Imperium Offline Luck Variance & Plasticity Audit ===\n";
  std::cout << "Mode: " << mode << " | Games: " << num_games
            << " | Master Seed: " << master_seed
            << " | Output Dir: " << output_dir << "\n";

  // Force global training flags to match recipe
  absl::SetFlag(&FLAGS_random_leader_draft, true);
  absl::SetFlag(&FLAGS_market_appendix_mode, "full_public_information_v3");
  absl::SetFlag(&FLAGS_enable_semantic_scorer, true);
  absl::SetFlag(&FLAGS_semantic_descriptor_schema, "semantic_action_v3");
  absl::SetFlag(&FLAGS_rollout_amp, false);
  absl::SetFlag(&FLAGS_allow_tf32, false);

  const auto configs = GenerateConfigs();
  const auto strata = GenerateStrata();

  std::shared_ptr<const open_spiel::Game> game =
      open_spiel::LoadGame("dune_imperium");

  if (mode == "collect_r7_44_critic_data") {
    if (output_dir == "/home/warcr/dune_drl_runtime/round7/r7_43_luck_plasticity_audit_20261019") {
      output_dir = "/home/warcr/dune_drl_runtime/round7/r7_44_critic_arch_test_20261019";
      std::filesystem::create_directories(output_dir);
    }
    open_spiel::RunCollectR744CriticData(
        *game, num_games, master_seed, episode_start, num_threads,
        eval_batch_size, eval_timeout_ms, output_dir,
        absl::GetFlag(FLAGS_checkpoint_u34350_live),
        r744_rows_per_game, r744_game_id_offset);
    return 0;
  }

  // Run Step 1 if mode is "all", "step1", or "collect_states_only"
  if (mode == "all" || mode == "step1" || mode == "collect_states_only") {
    std::cout << "\n>>> Starting Step 1: Real Trajectory & Luck Simulation <<<\n";

    torch::Device device(torch::kCUDA);
    auto model = std::make_shared<open_spiel::SharedDunePolicyValueNetImpl>(
        kObsDim, kHiddenDim, kActionDim, 16, /*use_nonlinear=*/false,
        /*with_aux_heads=*/false, /*head_init_seed=*/0,
        /*with_semantic_scorer=*/true, /*enable_zone_encoder=*/false,
        "semantic_action_v3");

    std::string ckpt_u34350_live = absl::GetFlag(FLAGS_checkpoint_u34350_live);
    std::cout << "Loading U34350 live checkpoint from: " << ckpt_u34350_live << "\n";
    open_spiel::LoadModelCheckpointRobust(model, ckpt_u34350_live, device);
    model->to(device);
    model->eval();

    std::shared_mutex sync_mutex;
    auto evaluator = std::make_shared<open_spiel::BatchedEvaluator>(
        model, eval_batch_size, eval_timeout_ms, device, &sync_mutex,
        /*logit_cap=*/10.0f, /*device_synchronize=*/true,
        /*high_priority_stream=*/false, /*emit_batch_membership=*/false,
        /*rollout_amp=*/false, /*allow_tf32=*/false);

    std::vector<Step1GameData> games(num_games);
    std::atomic<int> next_game_idx{0};
    std::atomic<int> completed_p1a{0};

    std::cout << "\n>>> Phase 1A: Real Trajectory Simulation (" << num_games << " games) <<<\n";
    auto p1a_start = std::chrono::steady_clock::now();

    auto p1a_worker_fn = [&]() {
      while (true) {
        int g = next_game_idx.fetch_add(1);
        if (g >= num_games) break;
        uint64_t ep_id = episode_start + static_cast<uint64_t>(g);
        bool keep = (mode != "collect_states_only");
        games[g] = SimulateRealTrajectory(g, ep_id, master_seed, *game, evaluator, keep);
        int done = completed_p1a.fetch_add(1) + 1;
        if (done % std::max(1, num_games / 10) == 0 || done == num_games) {
          std::cout << "  [Phase 1A Worker] Completed " << done << " / "
                    << num_games << " games\n";
        }
      }
    };

    std::vector<std::thread> workers;
    for (int t = 0; t < num_threads; ++t) {
      workers.emplace_back(p1a_worker_fn);
    }
    for (auto& w : workers) w.join();

    auto p1a_end = std::chrono::steady_clock::now();
    double phase1a_wall_ms =
        std::chrono::duration<double, std::milli>(p1a_end - p1a_start).count();

    int num_sel_games = num_games / 2;
    int num_conf_games = num_games - num_sel_games;

    double sum_all_rounds = 0.0, sum_sel_rounds = 0.0, sum_conf_rounds = 0.0;
    int64_t total_rows = 0, total_eligible_rows = 0;
    for (int g = 0; g < num_games; ++g) {
      sum_all_rounds += games[g].final_round;
      if (g < num_sel_games) {
        sum_sel_rounds += games[g].final_round;
      } else {
        sum_conf_rounds += games[g].final_round;
      }
      total_rows += games[g].rows.size();
      for (const auto& row : games[g].rows) {
        if (row.policy_loss_eligible) total_eligible_rows++;
      }
    }
    double avg_round_all = sum_all_rounds / num_games;
    double avg_round_sel = sum_sel_rounds / std::max(1, num_sel_games);
    double avg_round_conf = sum_conf_rounds / std::max(1, num_conf_games);
    double avg_rows_per_game = static_cast<double>(total_rows) / num_games;
    double avg_eligible_rows_per_game = static_cast<double>(total_eligible_rows) / num_games;
    double pct_eligible_rows = (total_rows > 0)
        ? (static_cast<double>(total_eligible_rows) / total_rows * 100.0) : 0.0;

    std::cout << absl::StrFormat("Phase 1A Complete in %.2f s (%.1f ms/game wall clock)\n",
                                 phase1a_wall_ms / 1000.0, phase1a_wall_ms / num_games);
    std::cout << absl::StrFormat("Overall Average Round Finished: %.2f (All %d), %.2f (Sel %d), %.2f (Conf %d)\n",
                                 avg_round_all, num_games, avg_round_sel, num_sel_games, avg_round_conf, num_conf_games);
    std::cout << absl::StrFormat("Average Rows / Game: %.1f | Policy-Loss Eligible: %.1f (%.1f%%)\n",
                                 avg_rows_per_game, avg_eligible_rows_per_game, pct_eligible_rows);

    // Extract confirmation states pool from confirmation half
    std::vector<SavedState> confirmation_states_pool;
    for (int g = num_sel_games; g < num_games; ++g) {
      for (const auto& row : games[g].rows) {
        if (row.policy_loss_eligible) {
          SavedState ss;
          ss.mc_return = row.mc_return;
          ss.terminal_return = row.terminal_return / 4.0f;
          ss.obs = row.obs;
          ss.legal_actions = row.legal_actions;
          ss.cand_data = row.cand_data;
          confirmation_states_pool.push_back(std::move(ss));
        }
      }
    }
    std::cout << "Total confirmation eligible states available: "
              << confirmation_states_pool.size() << "\n";

    size_t sample_n = std::min<size_t>(10000, confirmation_states_pool.size());
    std::mt19937_64 sample_rng(master_seed ^ 0x535441544553ULL);
    std::shuffle(confirmation_states_pool.begin(), confirmation_states_pool.end(), sample_rng);
    confirmation_states_pool.resize(sample_n);

    bool states_exist = false;
    {
      std::ifstream check_f(states_file, std::ios::binary);
      if (check_f.good()) states_exist = true;
    }
    if (states_exist) {
      std::cout << "Confirmation states file already exists: " << states_file
                << " (preserving existing file)\n";
    } else {
      SaveConfirmationStates(states_file, confirmation_states_pool);
    }

    if (mode == "collect_states_only") {
      std::cout << "Mode is collect_states_only. Exiting now.\n";
      return 0;
    }

    // Phase 1B: Variant C Contiguous Chance Rollouts
    std::cout << "\n>>> Phase 1B: Variant C Contiguous Chance Rollouts (" << num_games << " games) <<<\n";
    auto p1b_start = std::chrono::steady_clock::now();
    std::atomic<int> next_p1b_idx{0};
    std::atomic<int> completed_p1b{0};
    workers.clear();
    for (int t = 0; t < num_threads; ++t) {
      workers.emplace_back([&]() {
        while (true) {
          int g = next_p1b_idx.fetch_add(1);
          if (g >= num_games) break;
          RunVariantCOnGame(games[g], master_seed, evaluator);
          int done = completed_p1b.fetch_add(1) + 1;
          if (done % std::max(1, num_games / 10) == 0 || done == num_games) {
            std::cout << "  [Phase 1B Worker] Completed " << done << " / "
                      << num_games << " games\n";
          }
        }
      });
    }
    for (auto& w : workers) w.join();
    auto p1b_end = std::chrono::steady_clock::now();
    double phase1b_wall_ms =
        std::chrono::duration<double, std::milli>(p1b_end - p1b_start).count();
    std::cout << absl::StrFormat("Phase 1B Complete in %.2f s (%.1f ms/game wall clock)\n",
                                 phase1b_wall_ms / 1000.0, phase1b_wall_ms / num_games);

    // Phase 1C: Variant F Opponent Rollouts
    std::cout << "\n>>> Phase 1C: Variant F Opponent Rollouts <<<\n";
    auto p1c_start = std::chrono::steady_clock::now();
    struct RowWorkItem { int game_idx; int row_idx; };
    std::vector<RowWorkItem> all_work_rows;
    all_work_rows.reserve(total_rows);
    for (int g = 0; g < num_games; ++g) {
      for (int r = 0; r < static_cast<int>(games[g].rows.size()); ++r) {
        if (games[g].rows[r].post_action_state != nullptr) {
          all_work_rows.push_back({g, r});
        }
      }
    }
    std::cout << "Total rows queued for Variant F evaluation: " << all_work_rows.size() << "\n";

    std::atomic<size_t> next_work_idx{0};
    std::atomic<size_t> completed_p1c{0};
    workers.clear();
    for (int t = 0; t < num_threads; ++t) {
      workers.emplace_back([&]() {
        while (true) {
          size_t idx = next_work_idx.fetch_add(1);
          if (idx >= all_work_rows.size()) break;
          int g = all_work_rows[idx].game_idx;
          int r = all_work_rows[idx].row_idx;
          SimulateVariantFForRow(games[g].rows[r], g, r, master_seed,
                                 games[g].episode_id, evaluator);
          size_t done = completed_p1c.fetch_add(1) + 1;
          if (done % std::max<size_t>(1, all_work_rows.size() / 10) == 0 || done == all_work_rows.size()) {
            std::cout << "  [Phase 1C Worker] Completed " << done << " / "
                      << all_work_rows.size() << " rows\n";
          }
        }
      });
    }
    for (auto& w : workers) w.join();
    auto p1c_end = std::chrono::steady_clock::now();
    double phase1c_wall_ms =
        std::chrono::duration<double, std::milli>(p1c_end - p1c_start).count();
    double total_step1_wall_ms = phase1a_wall_ms + phase1b_wall_ms + phase1c_wall_ms;
    std::cout << absl::StrFormat("Phase 1C Complete in %.2f s (%.1f ms/game wall clock)\n",
                                 phase1c_wall_ms / 1000.0, phase1c_wall_ms / num_games);
    std::cout << absl::StrFormat("Total Step 1 Benchmark Time: %.2f s (%.1f ms/game wall clock)\n",
                                 total_step1_wall_ms / 1000.0, total_step1_wall_ms / num_games);

    // Call Counts & Unit Latencies
    int64_t total_c_val_calls_benchmark = 0;
    int64_t total_f_opp_calls_benchmark = 0;
    int64_t total_f_val_calls_benchmark = 0;
    for (int g = 0; g < num_games; ++g) {
      total_c_val_calls_benchmark += games[g].c_total_val_only_calls;
      for (const auto& row : games[g].rows) {
        for (int a = 0; a < 8; ++a) {
          total_f_opp_calls_benchmark += row.f_opp_policy_calls[a];
          total_f_val_calls_benchmark += row.f_val_only_calls[a];
        }
      }
    }
    double total_c_val_calls_per_game_k8_all =
        static_cast<double>(total_c_val_calls_benchmark) / num_games;
    double ms_per_c_val_call =
        (phase1b_wall_ms / num_games) / std::max(1.0, total_c_val_calls_per_game_k8_all);

    double total_f_evals_benchmark =
        static_cast<double>(total_f_opp_calls_benchmark + total_f_val_calls_benchmark);
    double total_f_evals_per_game_k8_all =
        total_f_evals_benchmark / num_games;
    double ms_per_f_eval =
        (phase1c_wall_ms / num_games) / std::max(1.0, total_f_evals_per_game_k8_all);

    // Backward Pass for A(lambda) and C(lambda, config)
    std::cout << "\nRunning Backward Pass for GAE advantages and luck adjustments...\n";
    const double lambdas[3] = {0.95, 0.98, 1.0};

    for (int g = 0; g < num_games; ++g) {
      for (int p = 0; p < 4; ++p) {
        const auto& p_rows = games[g].player_rows[p];
        int M = static_cast<int>(p_rows.size());
        if (M == 0) continue;

        for (int l_idx = 0; l_idx < 3; ++l_idx) {
          double lam = lambdas[l_idx];
          float next_A = 0.0f;
          for (int i = M - 1; i >= 0; --i) {
            int r_idx = p_rows[i];
            float delta = games[g].rows[r_idx].X_real - games[g].rows[r_idx].value;
            float A_i = delta + static_cast<float>(lam) * next_A;
            games[g].rows[r_idx].A[l_idx] = A_i;
            next_A = A_i;
          }
        }

        for (int i = 0; i < M; ++i) {
          int r_idx = p_rows[i];
          games[g].rows[r_idx].C_val.assign(configs.size(), 0.0f);
          games[g].rows[r_idx].C_off_val.assign(configs.size(), 0.0f);
        }

        for (const auto& cfg : configs) {
          float next_C = 0.0f;
          float next_C_off = 0.0f;

          for (int i = M - 1; i >= 0; --i) {
            int r_idx = p_rows[i];
            const auto& row = games[g].rows[r_idx];
            bool skip = (cfg.skip_rule == "R1_6_Only" && row.round > 6) ||
                        (cfg.skip_rule == "R1_3_Only" && row.round > 3);

            float shock = 0.0f;
            float shock_off = 0.0f;

            if (cfg.variant == "F") {
              if (!skip && !row.f_deterministic_skip) {
                float sum_x = 0.0f;
                for (int a = 0; a < cfg.k; ++a) sum_x += row.X_alt[a];
                shock = row.X_real - (1.0f / static_cast<float>(cfg.k)) * sum_x;
              }
            } else if (cfg.variant == "C_on") {
              if (!skip) {
                shock = row.luck_C_on[cfg.k_idx];
              }
            } else if (cfg.variant == "C_onoff") {
              if (!skip) {
                shock = row.luck_C_on[cfg.k_idx];
                shock_off = row.luck_C_off[cfg.k_idx];
              }
            }

            float C_i = shock + static_cast<float>(cfg.lambda) * next_C;
            float C_off_i = shock_off + static_cast<float>(cfg.lambda) * next_C_off;

            games[g].rows[r_idx].C_val[cfg.config_id] = C_i;
            if (cfg.variant == "C_onoff") {
              games[g].rows[r_idx].C_off_val[cfg.config_id] = C_off_i;
            }

            next_C = C_i;
            next_C_off = C_off_i;
          }
        }
      }
    }

    const size_t num_strata = strata.size();
    const size_t num_cfgs = configs.size();

    // Baseline Unadjusted Confirmation Variances
    std::vector<std::vector<double>> base_unadj_var(3, std::vector<double>(num_strata, 0.0));
    for (int l_idx = 0; l_idx < 3; ++l_idx) {
      for (size_t s = 0; s < num_strata; ++s) {
        int64_t n_c = 0;
        double sum_a = 0.0, sum_aa = 0.0;
        for (int cg = 0; cg < num_conf_games; ++cg) {
          int g = num_sel_games + cg;
          for (const auto& row : games[g].rows) {
            if (!row.policy_loss_eligible) continue;
            if (!strata[s].predicate(row)) continue;
            double a = row.A[l_idx];
            sum_a += a;
            sum_aa += a * a;
            n_c++;
          }
        }
        if (n_c > 1) {
          base_unadj_var[l_idx][s] = (sum_aa - (sum_a * sum_a) / n_c) / (n_c - 1);
        }
      }
    }
    double base_r13_var = base_unadj_var[0][1]; // lambda=0.95, stratum=1 (R1-3)
    std::cout << "Confirmation Baseline R1-3 Variance (lambda=0.95, beta=0): "
              << base_r13_var << "\n";

    // Per-Configuration Exact Call Counts & Training-Step Slowdown
    std::vector<ConfigDef> updated_configs = configs;
    for (auto& cfg : updated_configs) {
      int64_t cfg_extra_opp = 0;
      int64_t cfg_extra_val = 0;
      for (int g = 0; g < num_games; ++g) {
        for (const auto& row : games[g].rows) {
          bool active = true;
          if (cfg.skip_rule == "R1_6_Only" && row.round > 6) active = false;
          if (cfg.skip_rule == "R1_3_Only" && row.round > 3) active = false;
          if (!active) continue;

          if (cfg.variant == "F") {
            if (!row.f_deterministic_skip) {
              for (int a = 0; a < cfg.k; ++a) {
                cfg_extra_opp += row.f_opp_policy_calls[a];
                cfg_extra_val += row.f_val_only_calls[a];
              }
            }
          } else if (cfg.variant == "C_on") {
            cfg_extra_val += row.c_on_extra_val_calls[cfg.k_idx];
          } else if (cfg.variant == "C_onoff") {
            cfg_extra_val += (row.c_on_extra_val_calls[cfg.k_idx] +
                              row.c_off_extra_val_calls[cfg.k_idx]);
          }
        }
      }
      cfg.extra_opp_policy_calls_per_game = static_cast<double>(cfg_extra_opp) / num_games;
      cfg.extra_val_only_calls_per_game = static_cast<double>(cfg_extra_val) / num_games;
      if (cfg.variant == "F") {
        cfg.extra_ms_per_game = ms_per_f_eval *
            (cfg.extra_opp_policy_calls_per_game + cfg.extra_val_only_calls_per_game);
      } else {
        cfg.extra_ms_per_game = ms_per_c_val_call * cfg.extra_val_only_calls_per_game;
      }
      cfg.training_step_ms_per_game = 173.8 + cfg.extra_ms_per_game;
      cfg.training_step_slowdown_factor = cfg.training_step_ms_per_game / 173.8;
      cfg.pct_training_slowdown = (cfg.extra_ms_per_game / 173.8) * 100.0;
    }

    // Fit Betas on Selection Half (g < num_games / 2)
    std::cout << "Fitting OLS Betas on Selection Half...\n";
    std::vector<FittedBeta> fitted_betas(num_cfgs);

    for (size_t c = 0; c < num_cfgs; ++c) {
      const auto& cfg = updated_configs[c];
      int l_idx = (cfg.lambda == 0.95) ? 0 : (cfg.lambda == 0.98 ? 1 : 2);
      FittedBeta& fb = fitted_betas[c];

      if (cfg.variant == "F" || cfg.variant == "C_on") {
        double sum_a = 0, sum_c = 0, sum_ac = 0, sum_cc = 0;
        int64_t n_sel = 0;
        for (int g = 0; g < num_sel_games; ++g) {
          for (const auto& row : games[g].rows) {
            if (!row.policy_loss_eligible) continue;
            double a = row.A[l_idx];
            double c_val = row.C_val[c];
            sum_a += a;
            sum_c += c_val;
            sum_ac += a * c_val;
            sum_cc += c_val * c_val;
            n_sel++;
          }
        }
        if (n_sel > 1) {
          double var_c = (sum_cc - (sum_c * sum_c) / n_sel) / (n_sel - 1);
          double cov_ac = (sum_ac - (sum_a * sum_c) / n_sel) / (n_sel - 1);
          if (var_c > 1e-12) {
            fb.global_beta = cov_ac / var_c;
          }
        }
      } else {
        double sum_a = 0, sum_c1 = 0, sum_c2 = 0;
        double sum_c1c1 = 0, sum_c2c2 = 0, sum_c1c2 = 0;
        double sum_ac1 = 0, sum_ac2 = 0;
        int64_t n_sel = 0;
        for (int g = 0; g < num_sel_games; ++g) {
          for (const auto& row : games[g].rows) {
            if (!row.policy_loss_eligible) continue;
            double a = row.A[l_idx];
            double c1 = row.C_val[c];
            double c2 = row.C_off_val[c];
            sum_a += a;
            sum_c1 += c1;
            sum_c2 += c2;
            sum_c1c1 += c1 * c1;
            sum_c2c2 += c2 * c2;
            sum_c1c2 += c1 * c2;
            sum_ac1 += a * c1;
            sum_ac2 += a * c2;
            n_sel++;
          }
        }
        if (n_sel > 1) {
          double s11 = sum_c1c1 - (sum_c1 * sum_c1) / n_sel;
          double s22 = sum_c2c2 - (sum_c2 * sum_c2) / n_sel;
          double s12 = sum_c1c2 - (sum_c1 * sum_c2) / n_sel;
          double y1 = sum_ac1 - (sum_a * sum_c1) / n_sel;
          double y2 = sum_ac2 - (sum_a * sum_c2) / n_sel;
          double det = s11 * s22 - s12 * s12;
          if (det > 1e-12 && s11 > 1e-12 && s22 > 1e-12) {
            fb.global_beta = (s22 * y1 - s12 * y2) / det;
            fb.global_beta_off = (s11 * y2 - s12 * y1) / det;
          } else if (s11 > 1e-12) {
            fb.global_beta = y1 / s11;
          } else if (s22 > 1e-12) {
            fb.global_beta_off = y2 / s22;
          }
        }
      }

      for (int b = 0; b < 3; ++b) {
        if (cfg.variant == "F" || cfg.variant == "C_on") {
          double sum_a = 0, sum_c = 0, sum_ac = 0, sum_cc = 0;
          int64_t n_sel_b = 0;
          for (int g = 0; g < num_sel_games; ++g) {
            for (const auto& row : games[g].rows) {
              if (!row.policy_loss_eligible || row.round_band != b) continue;
              double a = row.A[l_idx];
              double c_val = row.C_val[c];
              sum_a += a;
              sum_c += c_val;
              sum_ac += a * c_val;
              sum_cc += c_val * c_val;
              n_sel_b++;
            }
          }
          if (n_sel_b > 1) {
            double var_c = (sum_cc - (sum_c * sum_c) / n_sel_b) / (n_sel_b - 1);
            double cov_ac = (sum_ac - (sum_a * sum_c) / n_sel_b) / (n_sel_b - 1);
            if (var_c > 1e-12) {
              fb.band_beta[b] = cov_ac / var_c;
            }
          }
        } else {
          double sum_a = 0, sum_c1 = 0, sum_c2 = 0;
          double sum_c1c1 = 0, sum_c2c2 = 0, sum_c1c2 = 0;
          double sum_ac1 = 0, sum_ac2 = 0;
          int64_t n_sel_b = 0;
          for (int g = 0; g < num_sel_games; ++g) {
            for (const auto& row : games[g].rows) {
              if (!row.policy_loss_eligible || row.round_band != b) continue;
              double a = row.A[l_idx];
              double c1 = row.C_val[c];
              double c2 = row.C_off_val[c];
              sum_a += a;
              sum_c1 += c1;
              sum_c2 += c2;
              sum_c1c1 += c1 * c1;
              sum_c2c2 += c2 * c2;
              sum_c1c2 += c1 * c2;
              sum_ac1 += a * c1;
              sum_ac2 += a * c2;
              n_sel_b++;
            }
          }
          if (n_sel_b > 1) {
            double s11 = sum_c1c1 - (sum_c1 * sum_c1) / n_sel_b;
            double s22 = sum_c2c2 - (sum_c2 * sum_c2) / n_sel_b;
            double s12 = sum_c1c2 - (sum_c1 * sum_c2) / n_sel_b;
            double y1 = sum_ac1 - (sum_a * sum_c1) / n_sel_b;
            double y2 = sum_ac2 - (sum_a * sum_c2) / n_sel_b;
            double det = s11 * s22 - s12 * s12;
            if (det > 1e-12 && s11 > 1e-12 && s22 > 1e-12) {
              fb.band_beta[b] = (s22 * y1 - s12 * y2) / det;
              fb.band_beta_off[b] = (s11 * y2 - s12 * y1) / det;
            } else if (s11 > 1e-12) {
              fb.band_beta[b] = y1 / s11;
            } else if (s22 > 1e-12) {
              fb.band_beta_off[b] = y2 / s22;
            }
          }
        }
      }
    }

    // Confirmation Half Evaluation & Bootstrap
    std::cout << "Evaluating Confirmation Half across " << num_boot_resamples
              << " whole-game bootstrap resamples...\n";
    struct GameStat {
      int64_t n = 0;
      double sum = 0.0;
      double sum_sq = 0.0;
    };

    const size_t num_modes = 4;
    std::vector<std::vector<std::vector<std::vector<GameStat>>>> game_stats(
        num_conf_games,
        std::vector<std::vector<std::vector<GameStat>>>(
            num_strata,
            std::vector<std::vector<GameStat>>(
                num_cfgs, std::vector<GameStat>(num_modes))));

    for (int cg = 0; cg < num_conf_games; ++cg) {
      int g = num_sel_games + cg;
      for (const auto& row : games[g].rows) {
        if (!row.policy_loss_eligible) continue;

        for (size_t s = 0; s < num_strata; ++s) {
          if (!strata[s].predicate(row)) continue;

          for (size_t c = 0; c < num_cfgs; ++c) {
            const auto& cfg = updated_configs[c];
            int l_idx = (cfg.lambda == 0.95) ? 0 : (cfg.lambda == 0.98 ? 1 : 2);
            float a = row.A[l_idx];
            float c1 = row.C_val[c];
            float c2 = (cfg.variant == "C_onoff") ? row.C_off_val[c] : 0.0f;
            const auto& fb = fitted_betas[c];

            float a_adj[4];
            a_adj[0] = a; // beta = 0
            a_adj[1] = (cfg.variant == "C_onoff") ? (a - 1.0f * (c1 + c2))
                                                  : (a - 1.0f * c1); // beta = 1
            a_adj[2] = (cfg.variant == "C_onoff")
                           ? (a - static_cast<float>(fb.global_beta) * c1 -
                              static_cast<float>(fb.global_beta_off) * c2)
                           : (a - static_cast<float>(fb.global_beta) * c1);
            int b = row.round_band;
            a_adj[3] = (cfg.variant == "C_onoff")
                           ? (a - static_cast<float>(fb.band_beta[b]) * c1 -
                              static_cast<float>(fb.band_beta_off[b]) * c2)
                           : (a - static_cast<float>(fb.band_beta[b]) * c1);

            for (size_t m = 0; m < num_modes; ++m) {
              GameStat& gs = game_stats[cg][s][c][m];
              gs.n++;
              gs.sum += a_adj[m];
              gs.sum_sq += a_adj[m] * a_adj[m];
            }
          }
        }
      }
    }

    // Direct Confirmation Variances and Shares (point estimates)
    std::vector<std::vector<std::vector<double>>> direct_vars(
        num_strata,
        std::vector<std::vector<double>>(
            num_cfgs, std::vector<double>(num_modes, 0.0)));
    std::vector<std::vector<std::vector<double>>> direct_shares(
        num_strata,
        std::vector<std::vector<double>>(
            num_cfgs, std::vector<double>(num_modes, 0.0)));

    for (size_t s = 0; s < num_strata; ++s) {
      for (size_t c = 0; c < num_cfgs; ++c) {
        for (size_t m = 0; m < num_modes; ++m) {
          int64_t tot_n = 0;
          double tot_sum = 0.0;
          double tot_sq = 0.0;
          for (int cg = 0; cg < num_conf_games; ++cg) {
            const GameStat& gs = game_stats[cg][s][c][m];
            tot_n += gs.n;
            tot_sum += gs.sum;
            tot_sq += gs.sum_sq;
          }
          double v = 0.0;
          if (tot_n > 1) {
            v = (tot_sq - (tot_sum * tot_sum) / tot_n) / (tot_n - 1);
          }
          direct_vars[s][c][m] = v;
        }
        double v0 = direct_vars[s][c][0];
        for (size_t m = 0; m < num_modes; ++m) {
          direct_shares[s][c][m] = (v0 > 1e-12) ? ((v0 - direct_vars[s][c][m]) / v0) : 0.0;
        }
      }
    }

    base_r13_var = direct_vars[1][0][0];
    std::cout << "Confirmation Direct Baseline R1-3 Variance (lambda=0.95, beta=0): "
              << base_r13_var << "\n";

    std::mt19937_64 boot_rng(master_seed ^ 0x424F4F5453545250ULL);
    std::uniform_int_distribution<int> game_dist(0, num_conf_games - 1);

    std::vector<std::vector<std::vector<std::vector<double>>>> boot_vars(
        num_strata,
        std::vector<std::vector<std::vector<double>>>(
            num_cfgs, std::vector<std::vector<double>>(num_modes)));
    std::vector<std::vector<std::vector<std::vector<double>>>> boot_shares(
        num_strata,
        std::vector<std::vector<std::vector<double>>>(
            num_cfgs, std::vector<std::vector<double>>(num_modes)));

    for (size_t s = 0; s < num_strata; ++s) {
      for (size_t c = 0; c < num_cfgs; ++c) {
        for (size_t m = 0; m < num_modes; ++m) {
          boot_vars[s][c][m].resize(num_boot_resamples);
          boot_shares[s][c][m].resize(num_boot_resamples);
        }
      }
    }

    std::vector<int> counts(num_conf_games, 0);
    for (int b = 0; b < num_boot_resamples; ++b) {
      std::fill(counts.begin(), counts.end(), 0);
      for (int i = 0; i < num_conf_games; ++i) {
        counts[game_dist(boot_rng)]++;
      }

      for (size_t s = 0; s < num_strata; ++s) {
        for (size_t c = 0; c < num_cfgs; ++c) {
          for (size_t m = 0; m < num_modes; ++m) {
            int64_t tot_n = 0;
            double tot_sum = 0.0;
            double tot_sq = 0.0;

            for (int cg = 0; cg < num_conf_games; ++cg) {
              if (counts[cg] == 0) continue;
              const GameStat& gs = game_stats[cg][s][c][m];
              tot_n += counts[cg] * gs.n;
              tot_sum += counts[cg] * gs.sum;
              tot_sq += counts[cg] * gs.sum_sq;
            }

            double v = 0.0;
            if (tot_n > 1) {
              v = (tot_sq - (tot_sum * tot_sum) / tot_n) / (tot_n - 1);
            }
            boot_vars[s][c][m][b] = v;
          }
          double b_v0 = boot_vars[s][c][0][b];
          for (size_t m = 0; m < num_modes; ++m) {
            boot_shares[s][c][m][b] =
                (b_v0 > 1e-12) ? ((b_v0 - boot_vars[s][c][m][b]) / b_v0 * 100.0) : 0.0;
          }
        }
      }
    }

    // Diagnostic Mean Luck with Game-Clustered SEs
    std::cout << "Computing Diagnostic Mean Luck with game-clustered SEs...\n";
    struct LuckDiag {
      std::string name;
      double mean = 0.0;
      double clustered_se = 0.0;
    };
    std::vector<std::vector<LuckDiag>> stratum_luck_diags(num_strata);

    for (size_t s = 0; s < num_strata; ++s) {
      for (const std::string& type : {"F_k4", "C_on_k4", "C_off_k4"}) {
        std::vector<double> game_l(num_conf_games, 0.0);
        std::vector<int64_t> game_n(num_conf_games, 0);
        int64_t total_n = 0;
        double total_l = 0;

        for (int cg = 0; cg < num_conf_games; ++cg) {
          int g = num_sel_games + cg;
          for (const auto& row : games[g].rows) {
            if (!row.policy_loss_eligible) continue;
            if (!strata[s].predicate(row)) continue;

            double l_val = 0.0;
            if (type == "F_k4") {
              if (!row.f_deterministic_skip) {
                double sx = 0;
                for (int a = 0; a < 4; ++a) sx += row.X_alt[a];
                l_val = row.X_real - 0.25 * sx;
              }
            } else if (type == "C_on_k4") {
              l_val = row.luck_C_on[1];
            } else if (type == "C_off_k4") {
              l_val = row.luck_C_off[1];
            }
            game_l[cg] += l_val;
            game_n[cg]++;
            total_l += l_val;
            total_n++;
          }
        }

        LuckDiag ld;
        ld.name = type;
        if (total_n > 0) {
          ld.mean = total_l / total_n;
          double u_sq_sum = 0.0;
          for (int cg = 0; cg < num_conf_games; ++cg) {
            double u = game_l[cg] - ld.mean * game_n[cg];
            u_sq_sum += u * u;
          }
          if (num_conf_games > 1) {
            double var_mu = (static_cast<double>(num_conf_games) /
                             (num_conf_games - 1.0)) *
                            (u_sq_sum / (static_cast<double>(total_n) * total_n));
            ld.clustered_se = std::sqrt(std::max(0.0, var_mu));
          }
        }
        stratum_luck_diags[s].push_back(ld);
      }
    }

    // Produce Step 1 JSON
    std::cout << "Writing step1_summary.json and step1_report.txt...\n";
    open_spiel::json::Object root;
    root["num_games"] = static_cast<int64_t>(num_games);
    root["selection_games"] = static_cast<int64_t>(num_sel_games);
    root["confirmation_games"] = static_cast<int64_t>(num_conf_games);
    root["master_seed"] = static_cast<int64_t>(master_seed);
    root["baseline_r13_var"] = base_r13_var;

    open_spiel::json::Object avg_round_obj;
    avg_round_obj["all"] = avg_round_all;
    avg_round_obj["selection"] = avg_round_sel;
    avg_round_obj["confirmation"] = avg_round_conf;
    root["overall_avg_round_finished"] = std::move(avg_round_obj);

    root["avg_rows_per_game"] = avg_rows_per_game;
    root["avg_policy_loss_eligible_rows_per_game"] = avg_eligible_rows_per_game;
    root["pct_policy_loss_eligible_rows"] = pct_eligible_rows;

    open_spiel::json::Object timing_obj;
    timing_obj["phase1a_wall_ms"] = phase1a_wall_ms;
    timing_obj["phase1b_wall_ms"] = phase1b_wall_ms;
    timing_obj["phase1c_wall_ms"] = phase1c_wall_ms;
    timing_obj["total_step1_wall_ms"] = total_step1_wall_ms;
    timing_obj["ms_per_f_eval"] = ms_per_f_eval;
    timing_obj["ms_per_c_val_call"] = ms_per_c_val_call;
    root["timings"] = std::move(timing_obj);

    open_spiel::json::Object base_var_obj;
    for (size_t s = 0; s < num_strata; ++s) {
      open_spiel::json::Object s_obj;
      s_obj["lambda_095"] = base_unadj_var[0][s];
      s_obj["lambda_098"] = base_unadj_var[1][s];
      s_obj["lambda_100"] = base_unadj_var[2][s];
      base_var_obj[strata[s].name] = std::move(s_obj);
    }
    root["baseline_unadjusted_variances"] = std::move(base_var_obj);

    auto calc_se = [](const std::vector<double>& vals) -> double {
      if (vals.size() <= 1) return 0.0;
      double mean = 0.0;
      for (double v : vals) mean += v;
      mean /= vals.size();
      double sq = 0;
      for (double v : vals) sq += (v - mean) * (v - mean);
      return std::sqrt(sq / (vals.size() - 1));
    };

    auto get_ci95 = [](std::vector<double> vals) -> std::pair<double, double> {
      if (vals.empty()) return {0.0, 0.0};
      std::sort(vals.begin(), vals.end());
      size_t idx_low = static_cast<size_t>(0.025 * vals.size());
      size_t idx_high = static_cast<size_t>(0.975 * vals.size());
      if (idx_high >= vals.size()) idx_high = vals.size() - 1;
      return {vals[idx_low], vals[idx_high]};
    };

    open_spiel::json::Array passing_gate_cfgs;
    open_spiel::json::Array cfgs_json;

    for (size_t c = 0; c < num_cfgs; ++c) {
      const auto& cfg = updated_configs[c];
      open_spiel::json::Object c_obj;
      c_obj["config_id"] = static_cast<int64_t>(cfg.config_id);
      c_obj["lambda"] = cfg.lambda;
      c_obj["k"] = static_cast<int64_t>(cfg.k);
      c_obj["variant"] = cfg.variant;
      c_obj["skip_rule"] = cfg.skip_rule;

      open_spiel::json::Object cost_obj;
      cost_obj["extra_opp_policy_calls_per_game"] = cfg.extra_opp_policy_calls_per_game;
      cost_obj["extra_val_only_calls_per_game"] = cfg.extra_val_only_calls_per_game;
      cost_obj["extra_ms_per_game"] = cfg.extra_ms_per_game;
      cost_obj["training_step_ms_per_game"] = cfg.training_step_ms_per_game;
      cost_obj["training_step_slowdown_factor"] = cfg.training_step_slowdown_factor;
      cost_obj["pct_training_slowdown"] = cfg.pct_training_slowdown;
      c_obj["cost"] = std::move(cost_obj);

      const auto& fb = fitted_betas[c];
      c_obj["global_beta"] = fb.global_beta;
      if (cfg.variant == "C_onoff") c_obj["global_beta_off"] = fb.global_beta_off;

      open_spiel::json::Array b_betas;
      for (int b = 0; b < 3; ++b) b_betas.push_back(fb.band_beta[b]);
      c_obj["band_betas"] = std::move(b_betas);

      open_spiel::json::Object strata_res;

      for (size_t s = 0; s < num_strata; ++s) {
        open_spiel::json::Object s_obj;
        s_obj["var_b0"] = direct_vars[s][c][0];
        s_obj["var_b1"] = direct_vars[s][c][1];
        s_obj["var_global_fit"] = direct_vars[s][c][2];
        s_obj["var_band_fit"] = direct_vars[s][c][3];
        s_obj["se_var_b0"] = calc_se(boot_vars[s][c][0]);
        s_obj["se_var_b1"] = calc_se(boot_vars[s][c][1]);
        s_obj["se_var_global_fit"] = calc_se(boot_vars[s][c][2]);
        s_obj["se_var_band_fit"] = calc_se(boot_vars[s][c][3]);
        s_obj["share_b1"] = direct_shares[s][c][1] * 100.0;
        s_obj["share_global_fit"] = direct_shares[s][c][2] * 100.0;
        s_obj["share_band_fit"] = direct_shares[s][c][3] * 100.0;
        s_obj["se_share_b1"] = calc_se(boot_shares[s][c][1]);
        s_obj["se_share_global_fit"] = calc_se(boot_shares[s][c][2]);
        s_obj["se_share_band_fit"] = calc_se(boot_shares[s][c][3]);

        auto ci_vg = get_ci95(boot_vars[s][c][2]);
        open_spiel::json::Array ci_vg_arr;
        ci_vg_arr.push_back(ci_vg.first);
        ci_vg_arr.push_back(ci_vg.second);
        s_obj["ci95_var_global_fit"] = std::move(ci_vg_arr);

        auto ci_sg = get_ci95(boot_shares[s][c][2]);
        open_spiel::json::Array ci_sg_arr;
        ci_sg_arr.push_back(ci_sg.first);
        ci_sg_arr.push_back(ci_sg.second);
        s_obj["ci95_share_global_fit"] = std::move(ci_sg_arr);

        strata_res[strata[s].name] = std::move(s_obj);
      }
      c_obj["strata_results"] = std::move(strata_res);

      double r13_gfit = direct_vars[1][c][2];
      bool gate_pass = (r13_gfit <= base_r13_var);
      c_obj["gate_pass"] = gate_pass;
      if (gate_pass) {
        passing_gate_cfgs.push_back(absl::StrFormat("config_%d|%s|lam=%.2f|k=%d|%s",
                                                    cfg.config_id, cfg.variant, cfg.lambda, cfg.k, cfg.skip_rule));
      }
      cfgs_json.push_back(std::move(c_obj));
    }
    root["configurations"] = std::move(cfgs_json);
    root["passing_gate_configurations"] = std::move(passing_gate_cfgs);

    std::ofstream json_out(output_dir + "/step1_summary.json");
    json_out << open_spiel::json::ToString(open_spiel::json::Value(root), true);
    json_out.close();

    // Text Report
    std::ofstream txt_out(output_dir + "/step1_report.txt");
    txt_out << "========================================================================================\n";
    txt_out << "          STEP 1: LUCK-ADJUSTED ADVANTAGE CONFIRMATION AUDIT REPORT\n";
    txt_out << "========================================================================================\n\n";

    txt_out << "Overall Average Round Finished:\n";
    txt_out << absl::StrFormat("  All %-3d Games:          %5.2f\n", num_games, avg_round_all);
    txt_out << absl::StrFormat("  Selection (%-3d Games):    %5.2f\n", num_sel_games, avg_round_sel);
    txt_out << absl::StrFormat("  Confirmation (%-3d Games): %5.2f\n\n", num_conf_games, avg_round_conf);

    txt_out << "Dataset Overview:\n";
    txt_out << absl::StrFormat("  Master Seed:            %lu\n", master_seed);
    txt_out << absl::StrFormat("  Average Rows / Game:    %5.1f\n", avg_rows_per_game);
    txt_out << absl::StrFormat("  Policy-Loss Eligible:   %5.1f (%5.1f%%)\n\n",
                               avg_eligible_rows_per_game, pct_eligible_rows);

    txt_out << "Wall-Clock Benchmark Runtimes:\n";
    txt_out << absl::StrFormat("  Phase 1A (Real Trajectories): %7.2f s (%6.1f ms/game wall clock)\n",
                               phase1a_wall_ms / 1000.0, phase1a_wall_ms / num_games);
    txt_out << absl::StrFormat("  Phase 1B (Variant C Rollouts): %6.2f s (%6.1f ms/game wall clock)\n",
                               phase1b_wall_ms / 1000.0, phase1b_wall_ms / num_games);
    txt_out << absl::StrFormat("  Phase 1C (Variant F Rollouts): %6.2f s (%6.1f ms/game wall clock)\n",
                               phase1c_wall_ms / 1000.0, phase1c_wall_ms / num_games);
    txt_out << absl::StrFormat("  Total Step 1 Benchmark Time:  %7.2f s (%6.1f ms/game wall clock)\n\n",
                               total_step1_wall_ms / 1000.0, total_step1_wall_ms / num_games);

    // Table 1: Baseline Unadjusted Variances
    txt_out << "--- BASELINE UNADJUSTED CONFIRMATION VARIANCES (Beta = 0) ---\n";
    txt_out << "+--------------------+----------------+----------------+----------------+\n";
    txt_out << "| Stratum            | Lambda = 0.95  | Lambda = 0.98  | Lambda = 1.00  |\n";
    txt_out << "+--------------------+----------------+----------------+----------------+\n";
    for (size_t s = 0; s < num_strata; ++s) {
      txt_out << absl::StrFormat("| %-18s | %14.6f | %14.6f | %14.6f |\n",
                                 strata[s].name.substr(0, 18),
                                 base_unadj_var[0][s], base_unadj_var[1][s], base_unadj_var[2][s]);
    }
    txt_out << "+--------------------+----------------+----------------+----------------+\n\n";

    // Table 2: Computational Cost (27 profiles)
    txt_out << "--- COMPUTATIONAL COST & ESTIMATED TRAINING-STEP SLOWDOWN ---\n";
    txt_out << "Baseline Training Step Time: 173.8 ms/game\n";
    txt_out << "+---------+---+------------+--------------+--------------+-------------+-------------+-----------+\n";
    txt_out << "| Variant | k | Skip Rule  | Ext Opp Calls| Ext Val Calls| Extra ms/gm | Step ms/gm  | Slowdown  |\n";
    txt_out << "+---------+---+------------+--------------+--------------+-------------+-------------+-----------+\n";
    for (const auto& var : {"F", "C_on", "C_onoff"}) {
      for (int k : {2, 4, 8}) {
        for (const auto& skip : {"AllRounds", "R1_6_Only", "R1_3_Only"}) {
          const ConfigDef* match_cfg = nullptr;
          for (const auto& c : updated_configs) {
            if (c.variant == var && c.k == k && c.skip_rule == skip) {
              match_cfg = &c;
              break;
            }
          }
          if (!match_cfg) continue;
          txt_out << absl::StrFormat("| %-7s | %-1d | %-10s | %12.1f | %12.1f | %11.2f | %11.2f | %8.1f%% |\n",
                                     var, k, skip,
                                     match_cfg->extra_opp_policy_calls_per_game,
                                     match_cfg->extra_val_only_calls_per_game,
                                     match_cfg->extra_ms_per_game,
                                     match_cfg->training_step_ms_per_game,
                                     match_cfg->pct_training_slowdown);
        }
      }
    }
    txt_out << "+---------+---+------------+--------------+--------------+-------------+-------------+-----------+\n\n";

    // Table 3: Diagnostic Mean Luck
    txt_out << "--- DIAGNOSTIC MEAN LUCK (with Game-Clustered SE, k = 4) ---\n";
    txt_out << "+--------------------+--------------------+--------------------+--------------------+\n";
    txt_out << "| Stratum            | F (k=4)            | C_on (k=4)         | C_off (k=4)        |\n";
    txt_out << "+--------------------+--------------------+--------------------+--------------------+\n";
    for (size_t s = 0; s < num_strata; ++s) {
      txt_out << absl::StrFormat("| %-18s | %8.5f +- %7.5f | %8.5f +- %7.5f | %8.5f +- %7.5f |\n",
                                 strata[s].name.substr(0, 18),
                                 stratum_luck_diags[s][0].mean, stratum_luck_diags[s][0].clustered_se,
                                 stratum_luck_diags[s][1].mean, stratum_luck_diags[s][1].clustered_se,
                                 stratum_luck_diags[s][2].mean, stratum_luck_diags[s][2].clustered_se);
    }
    txt_out << "+--------------------+--------------------+--------------------+--------------------+\n\n";

    // Table 4: Full Confirmation Variance & Variance Reduction (All 81 Configurations for R1-3 and All)
    txt_out << "--- CONFIRMATION VARIANCE & VARIANCE REDUCTION (All 81 Configurations) ---\n";
    txt_out << absl::StrFormat("Gate Baseline: Var(R1-3, lam=0.95, b=0) = %.6f\n", base_r13_var);
    txt_out << "+----+---------+------+---+------------+---------+----------+----------+----------+-------+\n";
    txt_out << "| ID | Variant | Lam  | k | Skip Rule  | Stratum | Var(b=0) | Var(fit) | Red (%)  | Gate  |\n";
    txt_out << "+----+---------+------+---+------------+---------+----------+----------+----------+-------+\n";
    for (size_t c = 0; c < num_cfgs; ++c) {
      const auto& cfg = updated_configs[c];
      for (size_t s : {1, 0}) {
        double v0 = direct_vars[s][c][0];
        double vf = direct_vars[s][c][2];
        double red_pct = direct_shares[s][c][2] * 100.0;
        std::string gate_str = (s == 1) ? ((vf <= base_r13_var) ? "PASS" : "FAIL") : "---";
        txt_out << absl::StrFormat("| %-2d | %-7s | %4.2f | %-1d | %-10s | %-7s | %8.5f | %8.5f | %7.2f%% | %-5s |\n",
                                   cfg.config_id, cfg.variant, cfg.lambda, cfg.k, cfg.skip_rule,
                                   strata[s].name, v0, vf, red_pct, gate_str);
      }
    }
    txt_out << "+----+---------+------+---+------------+---------+----------+----------+----------+-------+\n\n";

    // Table 5: Variance Reduction Breakdown by Round Band and Decision Role (AllRounds Configs)
    txt_out << "--- VARIANCE REDUCTION BY ROUND BAND (Representative AllRounds Configs) ---\n";
    txt_out << "+----------------------------+---------+----------+----------+----------+----------+\n";
    txt_out << "| Configuration              | Band    | Var(b=0) | Var(fit) | Red (%)  | Boot SE  |\n";
    txt_out << "+----------------------------+---------+----------+----------+----------+----------+\n";
    for (size_t c = 0; c < num_cfgs; ++c) {
      const auto& cfg = updated_configs[c];
      if (cfg.k != 4 || cfg.skip_rule != "AllRounds") continue;
      std::string cfg_title = absl::StrFormat("%s lam=%.2f k=%d", cfg.variant, cfg.lambda, cfg.k);
      for (size_t s : {1, 2, 3}) {
        double v0 = direct_vars[s][c][0];
        double vf = direct_vars[s][c][2];
        double red_pct = direct_shares[s][c][2] * 100.0;
        double se = calc_se(boot_vars[s][c][2]);
        txt_out << absl::StrFormat("| %-26s | %-7s | %8.5f | %8.5f | %7.2f%% | %8.5f |\n",
                                   cfg_title, strata[s].name, v0, vf, red_pct, se);
      }
    }
    txt_out << "+----------------------------+---------+----------+----------+----------+----------+\n\n";

    txt_out << "--- VARIANCE REDUCTION BY DECISION ROLE (Representative AllRounds Configs) ---\n";
    txt_out << "+----------------------------+--------------------+----------+----------+----------+\n";
    txt_out << "| Configuration              | Role               | Var(b=0) | Var(fit) | Red (%)  |\n";
    txt_out << "+----------------------------+--------------------+----------+----------+----------+\n";
    for (size_t c = 0; c < num_cfgs; ++c) {
      const auto& cfg = updated_configs[c];
      if (cfg.k != 4 || cfg.skip_rule != "AllRounds") continue;
      std::string cfg_title = absl::StrFormat("%s lam=%.2f k=%d", cfg.variant, cfg.lambda, cfg.k);
      for (size_t s = 4; s < num_strata; ++s) {
        double v0 = direct_vars[s][c][0];
        double vf = direct_vars[s][c][2];
        double red_pct = direct_shares[s][c][2] * 100.0;
        txt_out << absl::StrFormat("| %-26s | %-18s | %8.5f | %8.5f | %7.2f%% |\n",
                                   cfg_title, strata[s].name.substr(0, 18), v0, vf, red_pct);
      }
    }
    txt_out << "+----------------------------+--------------------+----------+----------+----------+\n\n";

    // Table 6: Passing Configurations Summary Table
    struct PassEntry {
      int id;
      std::string variant;
      double lambda;
      int k;
      std::string skip_rule;
      double r13_red;
      double slowdown;
      double efficiency;
    };
    std::vector<PassEntry> pass_list;
    for (size_t c = 0; c < num_cfgs; ++c) {
      const auto& cfg = updated_configs[c];
      double vf = direct_vars[1][c][2];
      if (vf <= base_r13_var) {
        double red_pct = direct_shares[1][c][2] * 100.0;
        double slow = std::max(0.01, cfg.pct_training_slowdown);
        pass_list.push_back({cfg.config_id, cfg.variant, cfg.lambda, cfg.k, cfg.skip_rule,
                             red_pct, cfg.pct_training_slowdown, red_pct / slow});
      }
    }

    std::sort(pass_list.begin(), pass_list.end(), [](const PassEntry& a, const PassEntry& b) {
      if (std::abs(a.lambda - b.lambda) > 1e-5) return a.lambda > b.lambda;
      return a.efficiency > b.efficiency;
    });

    txt_out << "--- PASSING CONFIGURATIONS (Sorted by Lambda Desc, then Efficiency Desc) ---\n";
    txt_out << "Gate Criterion: Var_R1_3(adjusted) <= Var_R1_3(lambda=0.95, beta=0)\n";
    txt_out << "Efficiency = (% R1-3 Variance Removed) / (% Training Step Slowdown)\n";
    txt_out << "+----+---------+------+---+------------+-------------+--------------+--------------+\n";
    txt_out << "| ID | Variant | Lam  | k | Skip Rule  | R1-3 Red (%)| Slowdown (%) | Efficiency   |\n";
    txt_out << "+----+---------+------+---+------------+-------------+--------------+--------------+\n";
    for (const auto& pe : pass_list) {
      txt_out << absl::StrFormat("| %-2d | %-7s | %4.2f | %-1d | %-10s | %10.2f%% | %11.2f%% | %12.4f |\n",
                                 pe.id, pe.variant, pe.lambda, pe.k, pe.skip_rule,
                                 pe.r13_red, pe.slowdown, pe.efficiency);
    }
    txt_out << "+----+---------+------+---+------------+-------------+--------------+--------------+\n\n";

    if (!pass_list.empty()) {
      txt_out << "Recommended Configuration:\n";
      txt_out << absl::StrFormat("  Config ID %d: %s | lambda=%.2f | k=%d | %s\n",
                                 pass_list[0].id, pass_list[0].variant, pass_list[0].lambda, pass_list[0].k, pass_list[0].skip_rule);
      txt_out << absl::StrFormat("  R1-3 Variance Reduction: %.2f%% | Slowdown: %.2f%% | Efficiency: %.4f\n\n",
                                 pass_list[0].r13_red, pass_list[0].slowdown, pass_list[0].efficiency);
    }

    txt_out.close();
    std::cout << "Step 1 artifacts written successfully.\n";
  }

  // -------------------------------------------------------------------------
  // Run Step 2 (Plasticity Audit)
  // -------------------------------------------------------------------------
  if (mode == "all" || mode == "step2") {
    std::cout << "\n>>> Starting Step 2: Plasticity Audit on Held-Out States <<<\n";
    torch::Device step2_dev(step2_device_str == "cuda" ? torch::kCUDA : torch::kCPU);

    std::vector<SavedState> states = LoadConfirmationStates(states_file);
    if (states.empty()) {
      open_spiel::SpielFatalError("No states loaded from: " + states_file);
    }

    struct CkptEntry {
      std::string name;
      std::string short_name;
      std::string path;
      int num_blocks;
    };

    std::vector<CkptEntry> ckpt_list = {
        {"U29700 (12b)", "U29700 (12b)", absl::GetFlag(FLAGS_checkpoint_u29700_12), 12},
        {"U33350 Parent (12b)", "U33350 Par(12b)", absl::GetFlag(FLAGS_checkpoint_u33350_12parent), 12},
        {"U33350 16b Init", "U33350-Init(16)", absl::GetFlag(FLAGS_checkpoint_u33350_16init), 16},
        {"U34350 Live (16b)", "U34350-Live(16)", absl::GetFlag(FLAGS_checkpoint_u34350_live), 16},
        {"U34350 EMA (16b)", "U34350-EMA (16)", absl::GetFlag(FLAGS_checkpoint_u34350_ema), 16}};

    std::vector<CheckpointPlasticityResult> results;
    for (const auto& ckpt : ckpt_list) {
      results.push_back(EvaluateCheckpointPlasticity(
          ckpt.name, ckpt.short_name, ckpt.path, ckpt.num_blocks, states, step2_dev));
    }

    // Write Step 2 JSON
    open_spiel::json::Object s2_root;
    s2_root["num_states"] = static_cast<int64_t>(states.size());
    s2_root["states_file"] = states_file;
    s2_root["device"] = step2_device_str;

    open_spiel::json::Array res_arr;
    for (const auto& res : results) {
      open_spiel::json::Object c_obj;
      c_obj["name"] = res.name;
      c_obj["path"] = res.path;
      c_obj["num_blocks"] = static_cast<int64_t>(res.num_blocks);
      c_obj["effective_rank"] = res.effective_rank;
      c_obj["stable_rank"] = res.stable_rank;

      open_spiel::json::Array sv_arr;
      for (double s : res.top_singular_values) sv_arr.push_back(s);
      c_obj["top_singular_values"] = std::move(sv_arr);

      // Dormant fractions
      open_spiel::json::Object dorm_obj;
      const char* g_names[3] = {"blocks_0_7", "blocks_8_11", "blocks_12_15"};
      for (int g = 0; g < 3; ++g) {
        if (g == 2 && res.num_blocks < 16) continue;
        open_spiel::json::Object g_obj;
        g_obj["relu1_tau0"] = res.group_dormant[g][0].dormant_tau0;
        g_obj["relu1_tau001"] = res.group_dormant[g][0].dormant_tau001;
        g_obj["relu1_tau01"] = res.group_dormant[g][0].dormant_tau01;
        g_obj["relu2_tau0"] = res.group_dormant[g][1].dormant_tau0;
        g_obj["relu2_tau001"] = res.group_dormant[g][1].dormant_tau001;
        g_obj["relu2_tau01"] = res.group_dormant[g][1].dormant_tau01;
        g_obj["combined_tau0"] = res.group_dormant[g][2].dormant_tau0;
        g_obj["combined_tau001"] = res.group_dormant[g][2].dormant_tau001;
        g_obj["combined_tau01"] = res.group_dormant[g][2].dormant_tau01;
        dorm_obj[g_names[g]] = std::move(g_obj);
      }
      c_obj["dormant_fractions"] = std::move(dorm_obj);

      // Gradient norms
      auto pack_norms = [](const ModelGradGroupNorms& n) {
        open_spiel::json::Object nobj;
        nobj["block_group_0_7"] = n.block_group_0_7;
        nobj["block_group_8_11"] = n.block_group_8_11;
        nobj["block_group_12_15"] = n.block_group_12_15;
        nobj["input_layer"] = n.input_layer;
        nobj["policy_head"] = n.policy_head;
        nobj["value_head"] = n.value_head;
        nobj["semantic_scorer"] = n.semantic_scorer;
        nobj["total_model"] = n.total_model;
        return nobj;
      };

      c_obj["grad_norm_pass_a_norm_adv"] = pack_norms(res.grad_pass_a_norm_adv);
      c_obj["grad_norm_pass_a_raw_adv"] = pack_norms(res.grad_pass_a_raw_adv);
      c_obj["grad_norm_pass_b_raw"] = pack_norms(res.grad_pass_b_raw);
      c_obj["grad_norm_pass_b_scaled"] = pack_norms(res.grad_pass_b_scaled);
      c_obj["grad_norm_pass_c_total"] = pack_norms(res.grad_pass_c_total);

      res_arr.push_back(std::move(c_obj));
    }
    s2_root["checkpoints"] = std::move(res_arr);

    std::ofstream json_out(output_dir + "/step2_summary.json");
    json_out << open_spiel::json::ToString(open_spiel::json::Value(s2_root), true);
    json_out.close();

    // Write Step 2 Report (strictly aligned tables, NO LaTeX math)
    std::ofstream txt_out(output_dir + "/step2_report.txt");
    txt_out << "========================================================================================\n";
    txt_out << "                 STEP 2: PLASTICITY AUDIT (CPU) SUMMARY REPORT\n";
    txt_out << "========================================================================================\n\n";
    txt_out << "Held-Out States Evaluated: " << states.size() << " | Device: " << step2_device_str << "\n\n";

    // Table 1: Trunk SVD & Rank Summary
    txt_out << "--- TRUNK SVD & RANK SUMMARY ---\n";
    txt_out << "+---------------------+------+----------+----------+-------------+----------+----------+\n";
    txt_out << "| Checkpoint          | Blks | Eff Rank | Stb Rank | s0 (Top SV) | s1       | s2       |\n";
    txt_out << "+---------------------+------+----------+----------+-------------+----------+----------+\n";
    for (const auto& r : results) {
      double s0 = (r.top_singular_values.size() > 0) ? r.top_singular_values[0] : 0.0;
      double s1 = (r.top_singular_values.size() > 1) ? r.top_singular_values[1] : 0.0;
      double s2 = (r.top_singular_values.size() > 2) ? r.top_singular_values[2] : 0.0;
      txt_out << absl::StrFormat("| %-19s | %-4d | %8.2f | %8.2f | %11.2f | %8.2f | %8.2f |\n",
                                 r.name, r.num_blocks, r.effective_rank, r.stable_rank, s0, s1, s2);
    }
    txt_out << "+---------------------+------+----------+----------+-------------+----------+----------+\n\n";

    // Table 2: Dormant Unit Fractions for tau in {0.00, 0.01, 0.10}
    struct TauDef {
      const char* name;
      const char* desc;
      int tau_idx;
    };
    std::vector<TauDef> tau_list = {
        {"tau = 0.00", "Completely Inactive / Dead Units", 0},
        {"tau = 0.01", "Standard Dormancy Threshold", 1},
        {"tau = 0.10", "Weakly Active Units", 2}};

    for (const auto& td : tau_list) {
      txt_out << absl::StrFormat("--- DORMANT UNIT FRACTIONS (%s: %s) ---\n", td.name, td.desc);
      txt_out << "+------------------+------------------------+------------------------+------------------------+\n";
      txt_out << "| Checkpoint       | Grp 0-7 Comb (r1/r2)   | Grp 8-11 Comb (r1/r2)  | Grp 12-15 Comb (r1/r2) |\n";
      txt_out << "+------------------+------------------------+------------------------+------------------------+\n";
      for (const auto& r : results) {
        auto get_dorm_str = [&](int g) -> std::string {
          if (g == 2 && r.num_blocks < 16) return "          N/A         ";
          double r1 = 0, r2 = 0, comb = 0;
          if (td.tau_idx == 0) {
            r1 = r.group_dormant[g][0].dormant_tau0 * 100.0;
            r2 = r.group_dormant[g][1].dormant_tau0 * 100.0;
            comb = r.group_dormant[g][2].dormant_tau0 * 100.0;
          } else if (td.tau_idx == 1) {
            r1 = r.group_dormant[g][0].dormant_tau001 * 100.0;
            r2 = r.group_dormant[g][1].dormant_tau001 * 100.0;
            comb = r.group_dormant[g][2].dormant_tau001 * 100.0;
          } else {
            r1 = r.group_dormant[g][0].dormant_tau01 * 100.0;
            r2 = r.group_dormant[g][1].dormant_tau01 * 100.0;
            comb = r.group_dormant[g][2].dormant_tau01 * 100.0;
          }
          return absl::StrFormat("%4.1f%% (%4.1f%%/%4.1f%%)", comb, r1, r2);
        };
        std::string s0 = get_dorm_str(0);
        std::string s1 = get_dorm_str(1);
        std::string s2 = get_dorm_str(2);
        txt_out << absl::StrFormat("| %-16s | %-22s | %-22s | %-22s |\n",
                                   r.short_name, s0, s1, s2);
      }
      txt_out << "+------------------+------------------------+------------------------+------------------------+\n\n";
    }

    // Table 3: Gradient Norms across all 5 passes
    struct PassDef {
      const char* title;
      std::function<const ModelGradGroupNorms&(const CheckpointPlasticityResult&)> get_norms;
    };
    std::vector<PassDef> pass_list = {
        {"PASS A (Policy Loss with Normalized Advantages)",
         [](const CheckpointPlasticityResult& r) -> const ModelGradGroupNorms& { return r.grad_pass_a_norm_adv; }},
        {"PASS A (Policy Loss with Raw Advantages)",
         [](const CheckpointPlasticityResult& r) -> const ModelGradGroupNorms& { return r.grad_pass_a_raw_adv; }},
        {"PASS B (Value Loss Raw, Coef = 1.0)",
         [](const CheckpointPlasticityResult& r) -> const ModelGradGroupNorms& { return r.grad_pass_b_raw; }},
        {"PASS B (Value Loss Scaled, Coef = 0.5)",
         [](const CheckpointPlasticityResult& r) -> const ModelGradGroupNorms& { return r.grad_pass_b_scaled; }},
        {"PASS C (Total PPO Loss: Policy + 0.5*Value - 0.01*Entropy)",
         [](const CheckpointPlasticityResult& r) -> const ModelGradGroupNorms& { return r.grad_pass_c_total; }}};

    for (const auto& p : pass_list) {
      txt_out << absl::StrFormat("--- GRADIENT NORMS: %s ---\n", p.title);
      txt_out << "+-----------------+---------+---------+---------+---------+---------+---------+---------+---------+\n";
      txt_out << "| Checkpoint      | G0-7    | G8-11   | G12-15  | Input   | PolHed  | ValHed  | Scorer  | Total   |\n";
      txt_out << "+-----------------+---------+---------+---------+---------+---------+---------+---------+---------+\n";
      for (const auto& r : results) {
        const auto& n = p.get_norms(r);
        std::string g12_str = (r.num_blocks >= 16)
            ? absl::StrFormat("%7.4f", n.block_group_12_15)
            : "  N/A  ";
        txt_out << absl::StrFormat("| %-15s | %7.4f | %7.4f | %7s | %7.4f | %7.4f | %7.4f | %7.4f | %7.4f |\n",
                                   r.short_name,
                                   n.block_group_0_7,
                                   n.block_group_8_11,
                                   g12_str,
                                   n.input_layer,
                                   n.policy_head,
                                   n.value_head,
                                   n.semantic_scorer,
                                   n.total_model);
      }
      txt_out << "+-----------------+---------+---------+---------+---------+---------+---------+---------+---------+\n\n";
    }

    txt_out.close();
    std::cout << "Step 2 artifacts written successfully.\n";
  }

  std::cout << "\n>>> Dune Luck Variance & Plasticity Audit Complete! <<<\n";
  return 0;
}
