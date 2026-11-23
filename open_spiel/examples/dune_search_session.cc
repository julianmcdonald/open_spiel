#include "dune_search_session_internal.h"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>

#include "dune_seed_utils.h"
#include "open_spiel/utils/json.h"

namespace open_spiel {
namespace dune_search_session_internal {

std::string LocalPhaseToString(dune_imperium::GamePhase phase) {
  switch (phase) {
    case dune_imperium::GamePhase::kLeaderOfferChance: return "kLeaderOfferChance";
    case dune_imperium::GamePhase::kLeaderDraft: return "kLeaderDraft";
    case dune_imperium::GamePhase::kDeal: return "kDeal";
    case dune_imperium::GamePhase::kRoundStart: return "kRoundStart";
    case dune_imperium::GamePhase::kAgentTurns: return "kAgentTurns";
    case dune_imperium::GamePhase::kRevealTurns: return "kRevealTurns";
    case dune_imperium::GamePhase::kCombat: return "kCombat";
    case dune_imperium::GamePhase::kMakers: return "kMakers";
    case dune_imperium::GamePhase::kRecall: return "kRecall";
    case dune_imperium::GamePhase::kTerminal: return "kTerminal";
    default: return "Unknown";
  }
}

// Greedy (argmax) when temperature == 0, else sample at r_val. Honors the eval
// semantics of FLAGS_temperature ("0.0 = greedy") on EVERY candidate selection
// path — in particular the low-coverage / starved-continuation fallback, where
// the search policy is the flat network prior rather than a one-hot
// GetFinalPolicy output. Sampling that flat prior (the Phase 18A Step 2
// regression) played the candidate stochastically against greedy opponents;
// taking the argmax restores parity. Training self-play uses temperature > 0 and
// is unaffected: the non-greedy branch delegates to SampleActionFromPrior, the
// session's own inverse-CDF sampler that the raw-reference path already used, so
// temperature > 0 reproduces the pre-existing sampling exactly (openspiel's
// SampleAction differs at cumulative boundaries and hard-asserts r_val < 1).
Action PickActionRespectingTemperature(const ActionsAndProbs& policy,
                                       double temperature, double r_val) {
  if (policy.empty()) return kInvalidAction;
  if (temperature == 0.0) {
    Action best = policy.front().first;
    double best_p = policy.front().second;
    for (const auto& ap : policy) {
      if (ap.second > best_p) {
        best_p = ap.second;
        best = ap.first;
      }
    }
    return best;
  }
  return SampleActionFromPrior(policy, r_val);
}

Action ArgmaxVisitAction(const SearchDiagnostics& diagnostics) {
  if (diagnostics.actions.empty() || diagnostics.visit_counts.empty() ||
      diagnostics.actions.size() != diagnostics.visit_counts.size()) {
    return kInvalidAction;
  }
  Action best_action = kInvalidAction;
  int max_visits = 0;
  for (size_t i = 0; i < diagnostics.actions.size(); ++i) {
    if (diagnostics.visit_counts[i] > max_visits) {
      max_visits = diagnostics.visit_counts[i];
      best_action = diagnostics.actions[i];
    }
  }
  return best_action;
}

int GetActionVisits(const SearchDiagnostics& diagnostics, Action action) {
  for (size_t i = 0; i < diagnostics.actions.size(); ++i) {
    if (diagnostics.actions[i] == action) {
      return diagnostics.visit_counts[i];
    }
  }
  return 0;
}

double GetActionQValue(const SearchDiagnostics& diagnostics, Action action) {
  for (size_t i = 0; i < diagnostics.actions.size(); ++i) {
    if (diagnostics.actions[i] == action) {
      return diagnostics.q_values[i];
    }
  }
  return 0.0;
}

LoadedCheckpointNet LoadDuneCheckpointNet(
    const std::string& checkpoint_path,
    const torch::Device& device,
    int64_t default_input_dim,
    int64_t action_size,
    int hidden_dim,
    int num_blocks,
    bool freeze_params,
    bool use_cuda_amp,
    bool enable_fp8_trunk) {
  int64_t ckpt_input_dim = default_input_dim;
  {
    torch::serialize::InputArchive archive;
    archive.load_from(checkpoint_path, torch::kCPU);
    torch::serialize::InputArchive in_archive;
    if (archive.try_read("input_layer", in_archive)) {
      torch::Tensor w;
      if (in_archive.try_read("weight", w)) {
        ckpt_input_dim = w.size(1);
      }
    }
  }

  dune_imperium::MarketAppendixMode detected_mode =
      dune_imperium::MarketAppendixMode::kNone;
  std::string descriptor_schema = "";
  bool has_semantic_scorer_flag = false;
  bool enable_zone_encoder = false;
  bool nonlinear_value_head = false;
  bool with_aux_heads = false;
  bool slot_indexed_out_layer_f = false;
  bool has_json_num_blocks = false;
  {
    std::filesystem::path p_ckpt(checkpoint_path);
    std::filesystem::path p_json = p_ckpt;
    p_json.replace_extension(".json");
    std::ifstream jf(p_json);
    if (!jf.good()) {
      jf.close();
      jf.open(p_ckpt.parent_path() / "manifest.json");
    }
    if (jf.good()) {
      std::string jstr((std::istreambuf_iterator<char>(jf)),
                       std::istreambuf_iterator<char>());
      auto jval = open_spiel::json::FromString(jstr);
      if (jval.has_value() && jval->IsObject()) {
        const auto& dict = jval->GetObject();
        auto it_mm = dict.find("market_appendix_mode");
        if (it_mm != dict.end() && it_mm->second.IsString()) {
          detected_mode = dune_imperium::ParseMarketAppendixMode(
              it_mm->second.GetString());
        }
        auto it_hd = dict.find("hidden_dim");
        if (it_hd != dict.end() && it_hd->second.IsInt()) {
          hidden_dim = static_cast<int>(it_hd->second.GetInt());
        }
        auto it_nb = dict.find("num_blocks");
        if (it_nb != dict.end() && it_nb->second.IsInt()) {
          num_blocks = static_cast<int>(it_nb->second.GetInt());
          has_json_num_blocks = true;
        }
        auto it_od = dict.find("observation_dim");
        if (it_od != dict.end() && it_od->second.IsInt()) {
          ckpt_input_dim = it_od->second.GetInt();
        }
        auto it_ss = dict.find("semantic_descriptor_schema");
        if (it_ss != dict.end() && it_ss->second.IsString()) {
          descriptor_schema = it_ss->second.GetString();
        }
        auto it_ws = dict.find("enable_semantic_scorer");
        if (it_ws != dict.end() && it_ws->second.IsBool()) {
          has_semantic_scorer_flag = it_ws->second.GetBool();
        }
        auto it_ze = dict.find("enable_zone_encoder");
        if (it_ze != dict.end() && it_ze->second.IsBool()) {
          enable_zone_encoder = it_ze->second.GetBool();
        }
        auto it_nl = dict.find("nonlinear_value_head");
        if (it_nl != dict.end() && it_nl->second.IsBool()) {
          nonlinear_value_head = it_nl->second.GetBool();
        }
        auto it_aux = dict.find("with_aux_heads");
        if (it_aux != dict.end() && it_aux->second.IsBool()) {
          with_aux_heads = it_aux->second.GetBool();
        }
        auto it_sif = dict.find("slot_indexed_out_layer_f");
        if (it_sif != dict.end()) {
          slot_indexed_out_layer_f =
              (it_sif->second.IsBool() && it_sif->second.GetBool()) ||
              (it_sif->second.IsString() && it_sif->second.GetString() == "true");
        }
      }
    }
  }

  const bool has_scorer = has_semantic_scorer_flag ||
                          CheckpointHasSemanticScorer(checkpoint_path, torch::kCPU);

  auto model = std::make_shared<SharedDunePolicyValueNetImpl>(
      ckpt_input_dim, hidden_dim, action_size, num_blocks,
      nonlinear_value_head, with_aux_heads, /*head_init_seed=*/0,
      /*with_semantic_scorer=*/has_scorer, enable_zone_encoder,
      descriptor_schema);
  if (!descriptor_schema.empty()) {
    model->SetSemanticDescriptorSchema(descriptor_schema);
  }
  if (slot_indexed_out_layer_f && model->semantic_scorer_ != nullptr) {
    model->semantic_scorer_->EnableF(true);
  }
  LoadModelCheckpointRobust(model, checkpoint_path, torch::Device(torch::kCPU));
  model->to(device);
  model->eval();
  if (freeze_params) {
    for (auto& parameter : model->parameters()) {
      parameter.set_requires_grad(false);
    }
  }
  if (use_cuda_amp) {
    if (enable_fp8_trunk) {
      model->PrepareFp8Cache();
    } else {
      model->RefreshBf16InferenceCache();
    }
  }
  return {std::move(model), detected_mode, hidden_dim, num_blocks, has_json_num_blocks};
}

std::vector<int> BuildSessionWarmupSizes(int target_batch_size,
                                         int max_inflight_worlds) {
  std::vector<int> warmup_sizes = {
      1, 4, std::min(target_batch_size, max_inflight_worlds),
      std::min(target_batch_size, 128), target_batch_size};
  std::sort(warmup_sizes.begin(), warmup_sizes.end());
  warmup_sizes.erase(std::unique(warmup_sizes.begin(), warmup_sizes.end()),
                     warmup_sizes.end());
  return warmup_sizes;
}

void InitCudaSuperhumanPriors(std::array<double, 12>& ms_per_round,
                              double& ms_ema,
                              int rollout_num_blocks,
                              bool& initialized_flag) {
  constexpr std::array<double, 12> kBase16bMs = {
      {8.5, 5.2, 4.6, 4.1, 3.6, 3.0, 2.5, 2.0, 1.6, 1.3, 1.1, 0.9}};
  const double block_scale =
      (rollout_num_blocks > 16)
          ? (static_cast<double>(rollout_num_blocks) / 16.0)
          : 1.0;
  for (size_t i = 0; i < 12; ++i) {
    ms_per_round[i] = kBase16bMs[i] * block_scale;
  }
  ms_ema = 4.0 * block_scale;
  initialized_flag = true;
}

}  // namespace dune_search_session_internal

uint64_t DeriveTrainingFullFastSeed(uint64_t config_seed, int round,
                                    Player seat, int episode_id,
                                    int decision_id) {
  // One position-tagged axis per coordinate. Signed inputs (round_ and
  // active_player_ are -1 / kInvalidPlayer before a placement activation) cast
  // injectively into the unsigned domain, so a negative coordinate is a
  // distinct stream rather than an aliased one.
  return dune_seed::DeriveSeed(config_seed, dune_seed::kStreamBlueprint,
                               static_cast<uint64_t>(round),
                               static_cast<uint64_t>(seat),
                               static_cast<uint64_t>(episode_id),
                               static_cast<uint64_t>(decision_id));
}

DuneSearchSession::DuneSearchSession(
    const DuneSearchConfig& config,
    std::shared_ptr<algorithms::Evaluator> evaluator,
    DuneSearchBudgetMode budget_mode)
    : DuneSearchSession(config, std::vector<std::shared_ptr<algorithms::Evaluator>>(4, evaluator), budget_mode) {}

DuneSearchSession::DuneSearchSession(
    const DuneSearchConfig& config,
    const std::vector<std::shared_ptr<algorithms::Evaluator>>& evaluators,
    DuneSearchBudgetMode budget_mode)
    : config_(config), evaluators_(evaluators), budget_mode_(budget_mode), rng_(config.seed) {
  placement_bot_ = std::make_shared<DunePUCTISMCTSBot>(config_, evaluators_);
  placement_bot_->in_session_ = true;

  DuneSearchConfig short_cfg = config_;
  short_bot_ = std::make_shared<DunePUCTISMCTSBot>(short_cfg, evaluators_);
  short_bot_->in_session_ = true;

  ResetSession("initialization");
}

void DuneSearchSession::ResetSession(const std::string& reason) {
  placement_bot_->Reset();
  short_bot_->Reset();
  has_active_session_ = false;
  session_id_ = "";
  active_player_ = kInvalidPlayer;
  round_ = -1;
  session_new_simulations_completed_ = 0;
  session_elapsed_time_ms_ = 0.0;
  long_agent_elapsed_time_ms_ = 0.0;
  cumulative_simulations_ = 0;
  is_full_session_ = false;
  training_full_fast_rolled_ = false;
  last_re_root_status_ = "none";
  post_chance_branch_miss_ = false;
  live_deadline_initialized_ = false;
  live_deadline_budget_ms_ = -1.0;
  last_role_ = DuneDecisionRole::kForcedOrBookkeeping;
  short_sims_completed_ = 0;
  short_cumulative_counter_ = 0;
  session_history_.clear();
  last_input_history_.clear();
  last_public_card_constraints_.clear();
  last_reset_reason_ = reason;

  has_pending_commit_ = false;
  last_search_state_.reset();
  last_requested_max_sims_ = 0;
}

// HandleReRootMismatch represents caller desynchronization safety reset boundary.
// When a history mismatch occurs during continuation/optional decision roles, the search
// forest cache cannot be reused because it is tied to the previous history prefix.
// Resetting the bot's state is the correct safety boundary to prevent stale state usage.
void DuneSearchSession::HandleReRootMismatch(const std::string& reason) {
  placement_bot_->Reset();
  short_bot_->Reset();
  last_re_root_status_ = "miss";
  post_chance_branch_miss_ = true;
  last_reset_reason_ = reason;
}

int DuneSearchSession::ConfiguredHardSimLimit(bool is_short_window_role,
                                             DuneDecisionRole role) const {
  if (is_short_window_role) return config_.purchase_combat_budget;
  switch (budget_mode_) {
    case DuneSearchBudgetMode::kTrainingFullFast:
      return is_full_session_ ? 64 : 8;
    case DuneSearchBudgetMode::kFixedSessionSimulations:
      return config_.fixed_session_limit;
    case DuneSearchBudgetMode::kTrainingPolicyIteration:
      // PER-ROLE, because in this mode the two agent-turn roles have genuinely
      // different configured caps. Reporting the primary's budget on a
      // continuation row would make every continuation look budget-starved to
      // any consumer computing `simulations_completed < hard_sim_limit` -- and
      // continuations are the majority of the rows this lane emits.
      return role == DuneDecisionRole::kAgentContinuation
                 ? config_.pi_continuation_simulations
                 : config_.pi_primary_simulations;
    default:
      return config_.max_simulations;
  }
}

double DuneSearchSession::ConfiguredHardTimeLimitMs(
    double resolved_live_deadline_ms) const {
  if (budget_mode_ != DuneSearchBudgetMode::kLiveDeadline) {
    return config_.relative_time_budget_ms;
  }
  // Report the deadline actually in force: the one the active session's clock
  // was built from once initialized, otherwise the value this call resolved.
  // Never a compile-time constant.
  return live_deadline_initialized_ ? live_deadline_budget_ms_
                                    : resolved_live_deadline_ms;
}

void DuneSearchSession::SetTrainingPolicyIterationBudgets(
    int primary_simulations, int continuation_simulations,
    int short_window_simulations) {
  if (budget_mode_ != DuneSearchBudgetMode::kTrainingPolicyIteration) {
    SpielFatalError(
        "SetTrainingPolicyIterationBudgets requires the policy-iteration "
        "budget mode.");
  }
  if (primary_simulations <= 0 || continuation_simulations <= 0 ||
      short_window_simulations <= 0) {
    SpielFatalError(
        "scratch policy-iteration budgets must all be strictly positive.");
  }
  config_.pi_primary_simulations = primary_simulations;
  config_.pi_continuation_simulations = continuation_simulations;
  config_.purchase_combat_budget = short_window_simulations;
}

std::shared_ptr<algorithms::Evaluator> MakeDuneNNEvaluator(
    const std::string& checkpoint_path,
    const std::string& device_str,
    int hidden_dim,
    int num_blocks) {
  auto game = open_spiel::LoadGame("dune_imperium");
  int64_t obs_size = game->GetType().provides_information_state_tensor
                         ? game->InformationStateTensorSize()
                         : game->ObservationTensorSize();
  int64_t action_size = game->NumDistinctActions();

  torch::Device device(device_str);
  auto loaded = dune_search_session_internal::LoadDuneCheckpointNet(
      checkpoint_path, device, obs_size, action_size, hidden_dim, num_blocks,
      /*freeze_params=*/false, /*use_cuda_amp=*/device.is_cuda(),
      /*enable_fp8_trunk=*/false);
  return std::make_shared<DuneNNEvaluator>(loaded.model, device, 10.0f,
                                           loaded.detected_mode);
}

void LoadCalibratedParameters(DuneSearchConfig& config) {
  std::ifstream f("calibration_results_v2/protocol_manifest.json");
  if (!f.good()) {
    f.close();
    f.open("calibration_results_v2/protocol_manifest_draft.json");
  }
  if (f.good()) {
    std::cout << "Loading calibrated search parameters from manifest...\n";
    std::string str((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    auto json_val = open_spiel::json::FromString(str);
    if (json_val.has_value() && json_val->IsObject()) {
      const auto& dict = json_val->GetObject();
      if (dict.find("winning_puct_c") != dict.end()) {
        const auto& val = dict.at("winning_puct_c");
        config.puct_c = val.IsDouble() ? val.GetDouble() : static_cast<double>(val.GetInt());
      }
      if (dict.find("winning_fixed_continuation_reserve") != dict.end()) {
        const auto& val = dict.at("winning_fixed_continuation_reserve");
        config.fixed_continuation_reserve = static_cast<int>(val.IsDouble() ? val.GetDouble() : static_cast<double>(val.GetInt()));
      }
      if (dict.find("winning_purchase_combat_budget") != dict.end()) {
        const auto& val = dict.at("winning_purchase_combat_budget");
        config.purchase_combat_budget = static_cast<int>(val.IsDouble() ? val.GetDouble() : static_cast<double>(val.GetInt()));
      }
      if (dict.find("winning_live_continuation_reserve_seconds") != dict.end()) {
        const auto& val = dict.at("winning_live_continuation_reserve_seconds");
        config.live_continuation_reserve_seconds = val.IsDouble() ? val.GetDouble() : static_cast<double>(val.GetInt());
      }
      if (dict.find("winning_root_prior_temperature") != dict.end()) {
        const auto& val = dict.at("winning_root_prior_temperature");
        config.root_prior_temperature = val.IsDouble() ? val.GetDouble() : static_cast<double>(val.GetInt());
      }
      if (dict.find("training_root_prior_temperature") != dict.end()) {
        const auto& val = dict.at("training_root_prior_temperature");
        config.training_root_prior_temperature = val.IsDouble() ? val.GetDouble() : static_cast<double>(val.GetInt());
      }
      if (dict.find("winning_conservative_override_enabled") != dict.end()) {
        config.conservative_override_enabled = dict.at("winning_conservative_override_enabled").GetBool();
      }
      if (dict.find("winning_conservative_covered_prior_threshold") != dict.end()) {
        const auto& val = dict.at("winning_conservative_covered_prior_threshold");
        config.conservative_covered_prior_threshold = val.IsDouble() ? val.GetDouble() : static_cast<double>(val.GetInt());
      }
      if (dict.find("winning_conservative_meaningful_visit_threshold") != dict.end()) {
        const auto& val = dict.at("winning_conservative_meaningful_visit_threshold");
        config.conservative_meaningful_visit_threshold = static_cast<int>(val.IsDouble() ? val.GetDouble() : static_cast<double>(val.GetInt()));
      }
      if (dict.find("winning_conservative_q_margin_threshold") != dict.end()) {
        const auto& val = dict.at("winning_conservative_q_margin_threshold");
        config.conservative_q_margin_threshold = val.IsDouble() ? val.GetDouble() : static_cast<double>(val.GetInt());
      }
      if (dict.find("winning_conservative_stability_checkpoint_fraction") != dict.end()) {
        const auto& val = dict.at("winning_conservative_stability_checkpoint_fraction");
        config.conservative_stability_checkpoint_fraction = val.IsDouble() ? val.GetDouble() : static_cast<double>(val.GetInt());
      }
      if (dict.find("winning_conservative_continuation_overrides_disabled") != dict.end()) {
        config.conservative_continuation_overrides_disabled = dict.at("winning_conservative_continuation_overrides_disabled").GetBool();
      }
    }
  } else {
    std::cout << "Calibration manifest not found. Using default parameters.\n";
  }
}

// ---------------------------------------------------------------------------
// Centralized Controller Selection and Lifecycle Commit Implementation
// ---------------------------------------------------------------------------

Action SampleActionFromPrior(const ActionsAndProbs& prior, double r_val) {
  if (prior.empty()) return kInvalidAction;
  double sum = 0.0;
  for (const auto& ap : prior) {
    sum += ap.second;
    if (r_val <= sum) return ap.first;
  }
  return prior.back().first;
}

DuneSearchResult DuneSearchSession::CommitAction(const ControllerDecision& decision) {
  SPIEL_CHECK_TRUE(has_pending_commit_);
  std::vector<Action> legal_acts = last_search_state_->LegalActions();
  bool is_legal = (std::find(legal_acts.begin(), legal_acts.end(), decision.selected_action) != legal_acts.end());
  SPIEL_CHECK_TRUE(is_legal);

  if (has_active_session_) {
    session_history_ = last_search_state_->History();
    session_history_.push_back(decision.selected_action);
  } else {
    session_history_.clear();
  }

  last_input_history_ = last_search_state_->History();

  last_search_state_.reset();
  has_pending_commit_ = false;

  // Propagate Decision Telemetry
  last_search_result_.diagnostics.raw_reference_action = decision.raw_reference_action;
  last_search_result_.diagnostics.mcts_proposed_action = decision.mcts_proposed_action;
  last_search_result_.diagnostics.selected_action = decision.selected_action;
  last_search_result_.diagnostics.confidence_fallback = decision.confidence_fallback;
  last_search_result_.diagnostics.mcts_overrode_raw = decision.mcts_overrode_raw;
  last_search_result_.diagnostics.stability_checkpoint_reached = decision.stability_checkpoint_reached;
  last_search_result_.diagnostics.stability_agreement = decision.stability_agreement;

  last_search_result_.diagnostics.pass_complete_search = decision.pass_complete_search;
  last_search_result_.diagnostics.pass_min_actions = decision.pass_min_actions;
  last_search_result_.diagnostics.pass_prior_mass = decision.pass_prior_mass;
  last_search_result_.diagnostics.pass_meaningful_visits = decision.pass_meaningful_visits;
  last_search_result_.diagnostics.pass_q_margin = decision.pass_q_margin;
  last_search_result_.diagnostics.pass_stability = decision.pass_stability;

  return last_search_result_;
}

void DuneSearchSession::DiscardPendingAction() {
  if (last_search_state_) {
    last_input_history_ = last_search_state_->History();
    session_history_ = last_input_history_;
  }
  last_search_state_.reset();
  has_pending_commit_ = false;
}

DuneSearchResult DuneSearchSession::SearchAndSelect(const State& state) {
  DuneSearchResult res = Search(state);
  std::uniform_real_distribution<double> dist(0.0, 1.0);
  double r_val = dist(rng_);
  ControllerDecision dec = SelectControllerAction(state, res, r_val);
  return CommitAction(dec);
}

DuneSearchResult DuneSearchSession::SearchAndSelect(const State& state, double r_val) {
  DuneSearchResult res = Search(state);
  ControllerDecision dec = SelectControllerAction(state, res, r_val);
  return CommitAction(dec);
}

DuneSearchResult DuneSearchSession::SearchAndSelectWithDeadline(
    const State& state, double remaining_time_ms) {
  DuneSearchResult res = Search(state, remaining_time_ms);
  std::uniform_real_distribution<double> dist(0.0, 1.0);
  double r_val = dist(rng_);
  ControllerDecision dec = SelectControllerAction(state, res, r_val);
  return CommitAction(dec);
}

}  // namespace open_spiel
