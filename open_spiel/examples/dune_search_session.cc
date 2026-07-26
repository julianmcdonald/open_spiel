#include "dune_search_session.h"
#include "open_spiel/games/dune_imperium/dune_imperium.h"
#include "open_spiel/abseil-cpp/absl/strings/str_cat.h"
#include "dune_seed_utils.h"
#include "open_spiel/spiel_bots.h"
#include "open_spiel/utils/json.h"
#include <fstream>
#include "open_spiel/abseil-cpp/absl/random/distributions.h"
#include <algorithm>
#include <cmath>
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
} // namespace

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

int DuneSearchSession::ConfiguredHardSimLimit(bool is_short_window_role) const {
  if (is_short_window_role) return config_.purchase_combat_budget;
  switch (budget_mode_) {
    case DuneSearchBudgetMode::kTrainingFullFast:
      return is_full_session_ ? 64 : 8;
    case DuneSearchBudgetMode::kFixedSessionSimulations:
      return config_.fixed_session_limit;
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

DuneSearchResult DuneSearchSession::Search(const State& state, double remaining_time_ms) {
  if (has_pending_commit_) {
    SpielFatalError("DuneSearchSession::Search called while a commit is still pending!");
  }
  intermediate_re_root_status_ = "none";
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
            std::cerr << "DEBUG RE-ROOT MISMATCH at index " << i << ": history=" << history[i]
                      << ", session_history=" << session_history_[i] << std::endl;
            is_descendant = false;
            break;
          }
        }
      }
      if (is_descendant) {
        last_re_root_status_ = "hit";
      } else {
        if (role != DuneDecisionRole::kAgentPrimary) {
          HandleReRootMismatch("re_root_mismatch");
        }
      }
    }
    intermediate_re_root_status_ = last_re_root_status_;
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

  // Resolve the live-deadline budget up front, so that BOTH the deadline the
  // search actually runs against and the deadline telemetry reports come from
  // one place. This used to be a hardcoded 52000.0 in two independent spots
  // (deadline init and hard_time_limit_ms), which meant every kLiveDeadline run
  // silently measured a 52s deadline no matter what the caller configured, and
  // the telemetry confirmed the lie. A measurement instrument must not invent
  // its own deadline: if no budget is configured, that is a fatal
  // misconfiguration, not something to paper over with a default.
  //
  // Resolved BEFORE the policy-only fallback lambda below so that (a) the
  // fallback path can report the same configured limits as the searched path,
  // and (b) a misconfigured live run fails at its very first decision rather
  // than only once it reaches a strategic one.
  double resolved_live_deadline_ms = -1.0;
  if (budget_mode_ == DuneSearchBudgetMode::kLiveDeadline) {
    resolved_live_deadline_ms = (remaining_time_ms >= 0.0)
                                    ? remaining_time_ms
                                    : config_.relative_time_budget_ms;
    if (!std::isfinite(resolved_live_deadline_ms) ||
        resolved_live_deadline_ms <= 0.0) {
      SpielFatalError(absl::StrCat(
          "DuneSearchSession: kLiveDeadline budget mode entered with no usable "
          "time budget (resolved=",
          resolved_live_deadline_ms,
          " ms). Pass a positive, finite remaining_time_ms via "
          "SearchAndSelectWithDeadline(), or configure a positive finite "
          "DuneSearchConfig::relative_time_budget_ms. Note --disable_time_limit "
          "sets that budget to infinity, which is meaningless in live mode."));
    }
  }

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
    // HARD limits describe what this decision was CONFIGURED to get; SOFT
    // limits and elapsed describe what it actually got, which on this path is
    // nothing (no search ran). Before WO-1 the hard limits were emitted as 0
    // here, which made ~62% of live-mode search.jsonl rows indistinguishable
    // from "zero budget configured" -- see the README note on grouping by
    // hard_time_limit_ms in historical runs.
    res.diagnostics.hard_sim_limit = ConfiguredHardSimLimit(is_short_window_role);
    res.diagnostics.soft_sim_limit = 0;
    res.diagnostics.hard_time_limit_ms =
        ConfiguredHardTimeLimitMs(resolved_live_deadline_ms);
    res.diagnostics.soft_time_limit_ms = 0.0;
    res.diagnostics.elapsed_search_time_ms = 0.0;
    res.diagnostics.inherited_root_visits = 0;
    res.diagnostics.newly_completed_simulations = 0;
    res.diagnostics.session_cumulative_simulations = session_new_simulations_completed_;
    res.diagnostics.short_window_cumulative_simulations = short_sims_completed_;
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

    res.diagnostics.actions.reserve(res.policy.size());
    res.diagnostics.priors.reserve(res.policy.size());
    res.diagnostics.visit_counts.assign(res.policy.size(), 0);
    res.diagnostics.q_values.assign(res.policy.size(), 0.0);
    for (const auto& ap : res.policy) {
      res.diagnostics.actions.push_back(ap.first);
      res.diagnostics.priors.push_back(ap.second);
    }

    last_search_state_ = state.Clone();
    has_pending_commit_ = true;
    last_requested_max_sims_ = 0;
    last_search_result_ = res;
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
    // If purchase_combat_budget is configured to 0 or less, short-window decisions
    // are routed directly to the policy-only fallback without search.
    // Zero simulations are completed, making further budget deduction moot.
    if (config_.purchase_combat_budget <= 0) {
      return get_policy_only_result("policy_only_purchase_combat");
    }
    max_sims = config_.purchase_combat_budget - short_sims_completed_;
    if (budget_mode_ == DuneSearchBudgetMode::kFixedSessionSimulations) {
      int overall_remaining = config_.fixed_session_limit - session_new_simulations_completed_ - short_sims_completed_;
      if (overall_remaining < max_sims) {
        max_sims = overall_remaining;
      }
    } else if (budget_mode_ == DuneSearchBudgetMode::kTrainingFullFast) {
      int total_limit = is_full_session_ ? 64 : 8;
      int overall_remaining = total_limit - session_new_simulations_completed_ - short_sims_completed_;
      if (overall_remaining < max_sims) {
        max_sims = overall_remaining;
      }
    }

    auto now = std::chrono::steady_clock::now();
    double elapsed_ms = std::chrono::duration<double, std::milli>(now - short_window_start_time_).count();
    max_time_ms = (config_.relative_time_budget_ms == std::numeric_limits<double>::infinity())
        ? std::numeric_limits<double>::infinity()
        : (500.0 - elapsed_ms);

    if (max_sims <= 0 || max_time_ms <= 0.0) {
      // Budget exhausted for this short window (e.g. the primary decision
      // consumed the whole fixed session budget with reserve == 0). Do NOT
      // fall through to RunSearch with zeroed limits: max_time_ms == 0 trips
      // RunSearch's expired-deadline pre-loop check, which returned a UNIFORM
      // policy that then got played verbatim at ~32% of decisions (the
      // Phase 18A Step 2 regression). Degrade to the raw network prior on the
      // true current state instead — identical to the purchase_combat_budget
      // <= 0 routing whose diagnostic run validated this behavior.
      return get_policy_only_result("short_window_budget_exceeded");
    }
  } else if (role == DuneDecisionRole::kAgentPrimary || role == DuneDecisionRole::kAgentContinuation) {
    if (budget_mode_ == DuneSearchBudgetMode::kPolicyOnly) {
      return get_policy_only_result("policy_only_mode");
    }

    if (budget_mode_ == DuneSearchBudgetMode::kFixedSessionSimulations) {
      int reserve = (role == DuneDecisionRole::kAgentPrimary) ? config_.fixed_continuation_reserve : 0;
      max_sims = config_.fixed_session_limit - reserve - session_new_simulations_completed_ - short_sims_completed_;
      if (max_sims <= 0) {
        max_sims = 0;
        limit_exceeded = true;
        limit_reason = "fixed_session_limit_exceeded";
      }
    } else if (budget_mode_ == DuneSearchBudgetMode::kTrainingFullFast) {
      if (!training_full_fast_rolled_) {
        uint64_t session_seed = DeriveTrainingFullFastSeed(
            config_.seed, round_, active_player_, episode_id_, decision_id_);
        // MakeRng32 seeds through a seed_seq over both halves; the plain
        // mt19937(seed) ctor would truncate the derived seed to 32 bits.
        std::mt19937 roll_rng = dune_seed::MakeRng32(session_seed);
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
      max_sims = total_limit - reserve - session_new_simulations_completed_ - short_sims_completed_;
      if (max_sims <= 0) {
        max_sims = 0;
        limit_exceeded = true;
        limit_reason = "training_full_fast_limit_exceeded";
      }
    } else if (budget_mode_ == DuneSearchBudgetMode::kLiveDeadline) {
      if (!live_deadline_initialized_) {
        absolute_live_deadline_ =
            std::chrono::steady_clock::now() +
            std::chrono::milliseconds(
                static_cast<int64_t>(resolved_live_deadline_ms));
        live_deadline_budget_ms_ = resolved_live_deadline_ms;
        live_deadline_initialized_ = true;
      }
      auto now = std::chrono::steady_clock::now();
      double reserve_sec = (role == DuneDecisionRole::kAgentPrimary) ? config_.live_continuation_reserve_seconds : 0.0;
      double remaining_sec = std::chrono::duration<double>(absolute_live_deadline_ - now).count() - reserve_sec;
      max_time_ms = remaining_sec * 1000.0;
      if (max_time_ms <= 0.0) {
        // Live deadline reached: calling RunSearch with max_time_ms == 0 trips
        // the expired-deadline pre-loop check and returns a uniform policy (the
        // same poison as the short-window path, but in live play). Degrade to
        // the raw network prior on the true current state instead.
        return get_policy_only_result("live_deadline_reached");
      }
      max_sims = config_.max_simulations;
    }
  }

  // 3. Re-routing key matching check
  int inherited_visits = 0;
  if (has_active_session_) {
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
  } else {
    last_re_root_status_ = "none";
  }

  // Run the MCTS search with limits and RNG seed continuity
  int start_sim_index = is_short_window_role ? short_cumulative_counter_ : cumulative_simulations_;
  DuneSearchResult result = active_bot->RunSearch(state, max_sims, max_time_ms, start_sim_index);

  if (limit_exceeded) {
    // The session-level budget was exhausted, so max_sims was clamped to 0 and
    // RunSearch could only degrade to a low-coverage/zero-visit fallback.
    // Record WHICH session limit was hit in a separate field and mark the
    // fallback, but do NOT overwrite RunSearch's own fallback_reason (e.g.
    // "low_coverage"): clobbering it here hid the real search behavior for days.
    result.diagnostics.budget_limit_reason = limit_reason;
    result.used_fallback = true;
  }

  // Update session counters
  if (is_short_window_role) {
    short_sims_completed_ += result.simulations_completed;
    short_cumulative_counter_ += result.simulations_completed;
  } else {
    session_new_simulations_completed_ += result.simulations_completed;
    cumulative_simulations_ += result.simulations_completed;
  }
  session_elapsed_time_ms_ += result.elapsed_time_ms;
  if (!is_short_window_role) {
    long_agent_elapsed_time_ms_ += result.elapsed_time_ms;
  }

  // Add rich telemetry and audit record fields
  result.diagnostics.protocol_version = "v2";
  result.diagnostics.session_id = session_id_;
  result.diagnostics.searched_seat = state.CurrentPlayer();
  result.diagnostics.round = dune_state.GetCurrentRound();
  result.diagnostics.phase = LocalPhaseToString(dune_state.phase());
  result.diagnostics.decision_role = absl::StrCat(static_cast<int>(role));
  result.diagnostics.budget_mode = absl::StrCat(static_cast<int>(budget_mode_));
  result.diagnostics.hard_sim_limit = ConfiguredHardSimLimit(is_short_window_role);
  result.diagnostics.soft_sim_limit = max_sims;
  result.diagnostics.hard_time_limit_ms =
      ConfiguredHardTimeLimitMs(resolved_live_deadline_ms);
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
  result.diagnostics.fallback_reason = result.fallback_reason;

  last_search_state_ = state.Clone();
  has_pending_commit_ = true;
  last_requested_max_sims_ = max_sims;
  last_search_result_ = result;

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

ControllerDecision DuneSearchSession::SelectControllerAction(
    const State& state,
    const DuneSearchResult& search_result,
    double r_val) {
  ControllerDecision decision;

  // 1. Raw Reference Action
  ActionsAndProbs raw_prior;
  if (state.IsChanceNode()) {
    raw_prior = state.ChanceOutcomes();
  } else {
    Player p = state.CurrentPlayer();
    if (p >= 0 && p < evaluators_.size()) {
      raw_prior = evaluators_[p]->Prior(state);
    }
    if (raw_prior.empty()) {
      std::vector<Action> legals = state.LegalActions();
      for (Action a : legals) {
        raw_prior.push_back({a, 1.0 / legals.size()});
      }
    }
  }
  decision.raw_reference_action =
      PickActionRespectingTemperature(raw_prior, config_.temperature, r_val);

  // 2. MCTS Proposed Action
  decision.mcts_proposed_action = ArgmaxVisitAction(search_result.diagnostics);

  // Get decision role
  DuneDecisionRole role = ClassifyDuneDecisionRole(state, state.CurrentPlayer(), has_active_session_);

  // Per-role logic
  if (role == DuneDecisionRole::kAgentPrimary) {
    if (config_.conservative_override_enabled) {
      // Enforce the conservative override protocol
      decision.pass_complete_search = (search_result.simulations_completed == last_requested_max_sims_ &&
                                       !search_result.timeout_status &&
                                       search_result.fallback_reason == "none");

      std::vector<Action> legal_actions = state.LegalActions();
      decision.pass_min_actions = (search_result.diagnostics.num_covered_actions >= std::min<int>(3, legal_actions.size()));
      decision.pass_prior_mass = (search_result.diagnostics.covered_prior_mass >= config_.conservative_covered_prior_threshold);

      int raw_visits = GetActionVisits(search_result.diagnostics, decision.raw_reference_action);
      int mcts_visits = GetActionVisits(search_result.diagnostics, decision.mcts_proposed_action);
      decision.pass_meaningful_visits = (raw_visits >= config_.conservative_meaningful_visit_threshold &&
                                         mcts_visits >= config_.conservative_meaningful_visit_threshold);

      double q_proposed = GetActionQValue(search_result.diagnostics, decision.mcts_proposed_action);
      double q_raw = GetActionQValue(search_result.diagnostics, decision.raw_reference_action);
      decision.pass_q_margin = (decision.mcts_proposed_action == decision.raw_reference_action) ||
                               ((q_proposed - q_raw) >= config_.conservative_q_margin_threshold);

      decision.stability_checkpoint_reached = search_result.diagnostics.stability_checkpoint_reached;
      decision.stability_agreement = search_result.diagnostics.stability_agreement;
      decision.pass_stability = decision.stability_agreement;

      if (decision.pass_complete_search &&
          decision.pass_min_actions &&
          decision.pass_prior_mass &&
          decision.pass_meaningful_visits &&
          decision.pass_q_margin &&
          decision.pass_stability) {
        decision.selected_action = decision.mcts_proposed_action;
        decision.mcts_overrode_raw = (decision.mcts_proposed_action != decision.raw_reference_action);
        decision.confidence_fallback = false;
      } else {
        decision.selected_action = decision.raw_reference_action;
        decision.mcts_overrode_raw = false;
        decision.confidence_fallback = true;
      }
    } else {
      // Legacy primary role logic (non-conservative MCTS action selection)
      if (search_result.policy.empty()) {
        decision.selected_action = decision.raw_reference_action;
      } else {
        decision.selected_action = PickActionRespectingTemperature(
            search_result.policy, config_.temperature, r_val);
      }
      decision.mcts_overrode_raw = (decision.selected_action != decision.raw_reference_action);
      decision.confidence_fallback = search_result.used_fallback;
    }
  } else if (role == DuneDecisionRole::kAgentContinuation) {
    if (config_.conservative_override_enabled && config_.conservative_continuation_overrides_disabled) {
      // Disable continuation overrides: return raw_ref, confidence_fallback = true
      decision.selected_action = decision.raw_reference_action;
      decision.confidence_fallback = true;
      decision.mcts_overrode_raw = false;
    } else {
      // Legacy continuation routing
      if (search_result.policy.empty()) {
        decision.selected_action = decision.raw_reference_action;
      } else {
        decision.selected_action = PickActionRespectingTemperature(
            search_result.policy, config_.temperature, r_val);
      }
      decision.mcts_overrode_raw = (decision.selected_action != decision.raw_reference_action);
      decision.confidence_fallback = search_result.used_fallback;
    }
  } else if (role == DuneDecisionRole::kForcedOrBookkeeping || role == DuneDecisionRole::kLeaderSelection) {
    std::vector<Action> legal_actions = state.LegalActions();
    if (role == DuneDecisionRole::kForcedOrBookkeeping && legal_actions.size() == 1) {
      decision.selected_action = legal_actions[0];
    } else {
      decision.selected_action = decision.raw_reference_action;
    }
    decision.confidence_fallback = true;
    decision.mcts_overrode_raw = false;
  } else if (role == DuneDecisionRole::kPurchase ||
             role == DuneDecisionRole::kCombatIntrigue ||
             role == DuneDecisionRole::kOtherOptional) {
    // Preserve their configured routing (use search result directly)
    if (search_result.policy.empty()) {
      decision.selected_action = decision.raw_reference_action;
    } else {
      decision.selected_action = PickActionRespectingTemperature(
          search_result.policy, config_.temperature, r_val);
    }
    decision.mcts_overrode_raw = (decision.selected_action != decision.raw_reference_action);
    decision.confidence_fallback = search_result.used_fallback;
  } else {
    SpielFatalError("Unknown/unhandled DuneDecisionRole in SelectControllerAction");
  }

  return decision;
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

} // namespace open_spiel
