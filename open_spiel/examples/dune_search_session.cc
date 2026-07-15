#include "dune_search_session.h"
#include "open_spiel/games/dune_imperium/dune_imperium.h"
#include "open_spiel/abseil-cpp/absl/strings/str_cat.h"
#include "dune_seed_utils.h"
#include "open_spiel/spiel_bots.h"
#include "open_spiel/utils/json.h"
#include <fstream>
#include "open_spiel/abseil-cpp/absl/random/distributions.h"
#include <algorithm>
#include <iostream>
#include "dune_evaluator.h"
#include <torch/torch.h>

namespace open_spiel {

namespace {
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
} // namespace

DuneSearchSession::DuneSearchSession(
    const DuneSearchConfig& config,
    std::shared_ptr<algorithms::Evaluator> evaluator,
    DuneSearchBudgetMode budget_mode)
    : DuneSearchSession(config, std::vector<std::shared_ptr<algorithms::Evaluator>>(4, evaluator), budget_mode) {}

DuneSearchSession::DuneSearchSession(
    const DuneSearchConfig& config,
    const std::vector<std::shared_ptr<algorithms::Evaluator>>& evaluators,
    DuneSearchBudgetMode budget_mode)
    : config_(config), evaluators_(evaluators), budget_mode_(budget_mode) {
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
  last_role_ = DuneDecisionRole::kForcedOrBookkeeping;
  short_sims_completed_ = 0;
  session_history_.clear();
  last_input_history_.clear();
  last_reset_reason_ = reason;
}

DuneSearchResult DuneSearchSession::Search(const State& state, double remaining_time_ms) {
  DuneDecisionRole role = ClassifyDuneDecisionRole(state, state.CurrentPlayer(), has_active_session_);
  const dune_imperium::DuneImperiumState& dune_state =
      static_cast<const dune_imperium::DuneImperiumState&>(state);

  // Descendant re-root validation check
  if (has_active_session_ && state.History() != last_input_history_) {
    if (state.CurrentPlayer() != active_player_ || dune_state.GetCurrentRound() != round_) {
      ResetSession("seat_or_round_boundary");
    } else {
      const std::vector<Action>& history = state.History();
      bool is_descendant = false;
      if (history.size() >= session_history_.size()) {
        is_descendant = true;
        for (size_t i = 0; i < session_history_.size(); ++i) {
          if (history[i] != session_history_[i]) {
            is_descendant = false;
            break;
          }
        }
      }
      if (!is_descendant) {
        if (role != DuneDecisionRole::kAgentPrimary) {
          ResetSession("re_root_mismatch");
          bool is_short_window_role = (role == DuneDecisionRole::kPurchase ||
                                       role == DuneDecisionRole::kCombatIntrigue ||
                                       role == DuneDecisionRole::kOtherOptional);
          if (role == DuneDecisionRole::kAgentContinuation || role == DuneDecisionRole::kOtherOptional || is_short_window_role) {
            last_re_root_status_ = "miss";
            post_chance_branch_miss_ = true;
          }
        }
      }
    }
  }

  // Session start boundary
  if (role == DuneDecisionRole::kAgentPrimary) {
    if (!has_active_session_ || state.History() != last_input_history_) {
      std::string pending_reason = last_reset_reason_;
      ResetSession("new_placement_activation");
      if (pending_reason != "none" && pending_reason != "initialization") {
        last_reset_reason_ = pending_reason;
      }
      has_active_session_ = true;
      active_player_ = state.CurrentPlayer();
      round_ = dune_state.GetCurrentRound();
      session_id_ = absl::StrCat("session_", episode_id_, "_r", round_, "_p", active_player_, "_d", decision_id_++);
    }
  }

  // Short window transition budget sharing
  bool is_short_window_role = (role == DuneDecisionRole::kPurchase ||
                               role == DuneDecisionRole::kCombatIntrigue ||
                               role == DuneDecisionRole::kOtherOptional);
  if (is_short_window_role) {
    if (last_role_ != role) {
      short_bot_->Reset();
      short_window_start_time_ = std::chrono::steady_clock::now();
      short_sims_completed_ = 0;
    }
  }
  if (role != DuneDecisionRole::kForcedOrBookkeeping && role != DuneDecisionRole::kLeaderSelection) {
    last_role_ = role;
  }

  std::shared_ptr<DunePUCTISMCTSBot> active_bot = is_short_window_role ? short_bot_ : placement_bot_;

  // Standard policy fallback result initialization
  auto get_policy_only_result = [&](const std::string& fallback_reason) {
    DuneSearchResult res;
    if (state.IsChanceNode()) {
      res.policy = state.ChanceOutcomes();
    } else {
      res.policy = evaluators_[state.CurrentPlayer()]->Prior(state);
    }
    res.simulations_completed = 0;
    res.elapsed_time_ms = 0.0;
    res.timeout_status = false;
    res.used_fallback = true;
    res.fallback_reason = fallback_reason;
    res.inference_count = 1;

    // Fill telemetry
    res.diagnostics.protocol_version = "v2";
    res.diagnostics.session_id = session_id_;
    res.diagnostics.searched_seat = state.CurrentPlayer();
    res.diagnostics.round = dune_state.GetCurrentRound();
    res.diagnostics.phase = LocalPhaseToString(dune_state.phase());
    res.diagnostics.decision_role = absl::StrCat(static_cast<int>(role));
    res.diagnostics.budget_mode = absl::StrCat(static_cast<int>(budget_mode_));
    res.diagnostics.hard_sim_limit = 0;
    res.diagnostics.soft_sim_limit = 0;
    res.diagnostics.hard_time_limit_ms = 0.0;
    res.diagnostics.soft_time_limit_ms = 0.0;
    res.diagnostics.elapsed_search_time_ms = 0.0;
    res.diagnostics.inherited_root_visits = 0;
    res.diagnostics.newly_completed_simulations = 0;
    res.diagnostics.session_cumulative_simulations = session_new_simulations_completed_;
    res.diagnostics.session_cumulative_search_time_ms = session_elapsed_time_ms_;
    res.diagnostics.long_agent_session_cumulative_time_ms = long_agent_elapsed_time_ms_;
    res.diagnostics.re_root_status = last_re_root_status_;
    res.diagnostics.post_chance_branch_miss = post_chance_branch_miss_;
    res.diagnostics.root_coverage = 0.0;
    res.diagnostics.reset_reason = fallback_reason;
    res.diagnostics.tree_node_count = active_bot->nodes_.size();
    res.diagnostics.inference_count = 1;
    res.diagnostics.legality_result = true;
    res.diagnostics.fallback_reason = fallback_reason;

    Action selected = kInvalidAction;
    double max_prob = -1.0;
    res.diagnostics.actions.reserve(res.policy.size());
    res.diagnostics.priors.reserve(res.policy.size());
    res.diagnostics.visit_counts.assign(res.policy.size(), 0);
    res.diagnostics.q_values.assign(res.policy.size(), 0.0);
    for (const auto& ap : res.policy) {
      res.diagnostics.actions.push_back(ap.first);
      res.diagnostics.priors.push_back(ap.second);
      if (ap.second > max_prob) {
        max_prob = ap.second;
        selected = ap.first;
      }
    }
    res.diagnostics.selected_action = selected;
    return res;
  };

  // 1. Zero simulations for bookkeeping or draft
  if (role == DuneDecisionRole::kForcedOrBookkeeping || role == DuneDecisionRole::kLeaderSelection) {
    return get_policy_only_result("forced_or_bookkeeping");
  }

  // 2. Budget limits setup setup based on role and mode
  int max_sims = config_.max_simulations;
  double max_time_ms = config_.relative_time_budget_ms;

  bool limit_exceeded = false;
  std::string limit_reason = "";

  if (is_short_window_role) {
    if (config_.purchase_combat_budget <= 0) {
      return get_policy_only_result("policy_only_purchase_combat");
    }
    max_sims = config_.purchase_combat_budget - short_sims_completed_;
    auto now = std::chrono::steady_clock::now();
    double elapsed_ms = std::chrono::duration<double, std::milli>(now - short_window_start_time_).count();
    max_time_ms = (config_.relative_time_budget_ms == std::numeric_limits<double>::infinity())
        ? std::numeric_limits<double>::infinity()
        : (500.0 - elapsed_ms);

    if (max_sims <= 0 || max_time_ms <= 0.0) {
      max_sims = 0;
      max_time_ms = 0.0;
      limit_exceeded = true;
      limit_reason = "short_window_budget_exceeded";
    }
  } else if (role == DuneDecisionRole::kAgentPrimary || role == DuneDecisionRole::kAgentContinuation) {
    if (budget_mode_ == DuneSearchBudgetMode::kPolicyOnly) {
      return get_policy_only_result("policy_only_mode");
    }

    if (budget_mode_ == DuneSearchBudgetMode::kFixedSessionSimulations) {
      int reserve = (role == DuneDecisionRole::kAgentPrimary) ? config_.fixed_continuation_reserve : 0;
      max_sims = config_.fixed_session_limit - reserve - session_new_simulations_completed_;
      if (max_sims <= 0) {
        max_sims = 0;
        limit_exceeded = true;
        limit_reason = "fixed_session_limit_exceeded";
      }
    } else if (budget_mode_ == DuneSearchBudgetMode::kTrainingFullFast) {
      if (!training_full_fast_rolled_) {
        uint64_t session_seed = dune_seed::Combine(config_.seed, dune_seed::kStreamBlueprint, round_ + active_player_ + episode_id_ + decision_id_);
        std::mt19937 roll_rng(session_seed);
        is_full_session_ = (roll_rng() % 4 == 0); // 25% chance of full search
        training_full_fast_rolled_ = true;
      }
      
      // Dynamic config adjusting for full vs fast searches:
      if (is_full_session_) {
        placement_bot_->GetConfig().temperature = 1.0;
        placement_bot_->GetConfig().dirichlet_epsilon = 0.25;
        placement_bot_->GetConfig().root_prior_temperature = config_.training_root_prior_temperature;
      } else {
        placement_bot_->GetConfig().temperature = 0.0;
        placement_bot_->GetConfig().dirichlet_epsilon = 0.0;
        placement_bot_->GetConfig().root_prior_temperature = 1.0; // unflattened / raw prior
      }

      int reserve = 0;
      if (role == DuneDecisionRole::kAgentPrimary && is_full_session_) {
        reserve = config_.fixed_continuation_reserve;
      }
      int total_limit = is_full_session_ ? 64 : 8;
      max_sims = total_limit - reserve - session_new_simulations_completed_;
      if (max_sims <= 0) {
        max_sims = 0;
        limit_exceeded = true;
        limit_reason = "training_full_fast_limit_exceeded";
      }
    } else if (budget_mode_ == DuneSearchBudgetMode::kLiveDeadline) {
      if (!live_deadline_initialized_) {
        double init_time_ms = (remaining_time_ms >= 0.0) ? remaining_time_ms : 52000.0;
        absolute_live_deadline_ = std::chrono::steady_clock::now() + std::chrono::milliseconds(static_cast<int64_t>(init_time_ms));
        live_deadline_initialized_ = true;
      }
      auto now = std::chrono::steady_clock::now();
      double reserve_sec = (role == DuneDecisionRole::kAgentPrimary) ? config_.live_continuation_reserve_seconds : 0.0;
      double remaining_sec = std::chrono::duration<double>(absolute_live_deadline_ - now).count() - reserve_sec;
      max_time_ms = remaining_sec * 1000.0;
      if (max_time_ms <= 0.0) {
        max_time_ms = 0.0;
        limit_exceeded = true;
        limit_reason = "live_deadline_reached";
      }
      max_sims = config_.max_simulations;
    }
  }

  // 3. Re-routing key matching check
  int inherited_visits = 0;
  if (has_active_session_ && last_re_root_status_ != "miss") {
    auto root_key = active_bot->GetStateKey(state);
    auto it = active_bot->nodes_.find(root_key);
    if (it != active_bot->nodes_.end()) {
      last_re_root_status_ = "hit";
      inherited_visits = it->second->total_visits;
    } else {
      if (role == DuneDecisionRole::kAgentContinuation || role == DuneDecisionRole::kOtherOptional) {
        last_re_root_status_ = "miss";
        post_chance_branch_miss_ = true;
      } else {
        last_re_root_status_ = "none";
      }
    }
  } else if (!has_active_session_ && last_re_root_status_ != "miss") {
    last_re_root_status_ = "none";
  }

  // Run the MCTS search with limits and RNG seed continuity
  DuneSearchResult result = active_bot->RunSearch(state, max_sims, max_time_ms, cumulative_simulations_);

  if (limit_exceeded && result.used_fallback) {
    result.fallback_reason = limit_reason;
    result.diagnostics.fallback_reason = limit_reason;
  }

  // Update session counters
  if (is_short_window_role) {
    short_sims_completed_ += result.simulations_completed;
  } else {
    session_new_simulations_completed_ += result.simulations_completed;
    cumulative_simulations_ += result.simulations_completed;
  }
  session_elapsed_time_ms_ += result.elapsed_time_ms;
  if (!is_short_window_role) {
    long_agent_elapsed_time_ms_ += result.elapsed_time_ms;
  }

  // Sample selected action for telemetry
  Action selected_act = kInvalidAction;
  if (!result.policy.empty()) {
    active_bot->search_count_++;
    uint64_t step_seed = dune_seed::Combine(config_.seed, dune_seed::kStreamBlueprint, active_bot->search_count_);
    std::mt19937 step_rng(step_seed);
    double r_val = absl::Uniform(step_rng, 0.0, 1.0);
    selected_act = SampleAction(result.policy, r_val).first;
  }

  // Add rich telemetry and audit record fields
  result.diagnostics.protocol_version = "v2";
  result.diagnostics.session_id = session_id_;
  result.diagnostics.searched_seat = state.CurrentPlayer();
  result.diagnostics.round = dune_state.GetCurrentRound();
  result.diagnostics.phase = LocalPhaseToString(dune_state.phase());
  result.diagnostics.decision_role = absl::StrCat(static_cast<int>(role));
  result.diagnostics.budget_mode = absl::StrCat(static_cast<int>(budget_mode_));
  result.diagnostics.hard_sim_limit = is_short_window_role
      ? config_.purchase_combat_budget
      : ((budget_mode_ == DuneSearchBudgetMode::kTrainingFullFast)
         ? (is_full_session_ ? 64 : 8)
         : ((budget_mode_ == DuneSearchBudgetMode::kFixedSessionSimulations) ? config_.fixed_session_limit : config_.max_simulations));
  result.diagnostics.soft_sim_limit = max_sims;
  result.diagnostics.hard_time_limit_ms = (budget_mode_ == DuneSearchBudgetMode::kLiveDeadline) ? 52000.0 : config_.relative_time_budget_ms;
  result.diagnostics.soft_time_limit_ms = max_time_ms;
  result.diagnostics.elapsed_search_time_ms = result.elapsed_time_ms;
  result.diagnostics.inherited_root_visits = inherited_visits;
  result.diagnostics.newly_completed_simulations = result.simulations_completed;
  result.diagnostics.session_cumulative_simulations = session_new_simulations_completed_;
  result.diagnostics.short_window_cumulative_simulations = short_sims_completed_;
  result.diagnostics.session_cumulative_search_time_ms = session_elapsed_time_ms_;
  result.diagnostics.long_agent_session_cumulative_time_ms = long_agent_elapsed_time_ms_;
  result.diagnostics.re_root_status = last_re_root_status_;
  result.diagnostics.post_chance_branch_miss = post_chance_branch_miss_;
  result.diagnostics.root_coverage = result.diagnostics.covered_prior_mass;
  result.diagnostics.reset_reason = last_reset_reason_;
  last_reset_reason_ = "none"; // Clear it after use so it doesn't persist
  result.diagnostics.tree_node_count = active_bot->nodes_.size();
  result.diagnostics.inference_count = result.inference_count;
  result.diagnostics.selected_action = selected_act;

  bool is_legal = false;
  if (selected_act != kInvalidAction) {
    std::vector<Action> legal_acts = state.LegalActions();
    is_legal = (std::find(legal_acts.begin(), legal_acts.end(), selected_act) != legal_acts.end());
  }
  result.diagnostics.legality_result = is_legal;
  result.diagnostics.fallback_reason = result.fallback_reason;

  if (has_active_session_) {
    session_history_ = state.History();
    if (selected_act != kInvalidAction) {
      session_history_.push_back(selected_act);
    }
  } else {
    session_history_.clear();
  }
  last_input_history_ = state.History();

  return result;
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
  auto model = std::make_shared<SharedDunePolicyValueNetImpl>(
      obs_size, hidden_dim, action_size, num_blocks);
  torch::load(model, checkpoint_path, device);
  model->to(device);
  model->eval();
  return std::make_shared<DuneNNEvaluator>(model, device);
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
    }
  } else {
    std::cout << "Calibration manifest not found. Using default parameters.\n";
  }
}

} // namespace open_spiel
