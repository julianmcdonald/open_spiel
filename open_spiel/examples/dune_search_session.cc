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
#include "dune_batched_evaluator.h"
#include "dune_compound_turn_search.h"
#include "dune_wide_rollout_search.h"
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

DuneSearchResult DuneSearchSession::Search(const State& state, double remaining_time_ms) {
  if (has_pending_commit_) {
    SpielFatalError("DuneSearchSession::Search called while a commit is still pending!");
  }
  intermediate_re_root_status_ = "none";
  DuneDecisionRole role = ClassifyDuneDecisionRole(state, state.CurrentPlayer(), has_active_session_);
  const dune_imperium::DuneImperiumState& dune_state =
      static_cast<const dune_imperium::DuneImperiumState&>(state);
  const std::string public_constraints = dune_state.PublicCardKnowledgeDigest();
  if (state.History() == last_input_history_ &&
      !last_public_card_constraints_.empty() &&
      public_constraints != last_public_card_constraints_) {
    // A capture can correct knowledge without applying an action. Invalidate
    // the cached worlds while retaining the current turn's clocks and budget.
    HandleReRootMismatch("public_card_knowledge_correction");
  }

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
  last_public_card_constraints_ = public_constraints;
  bool is_short_window_role = (role == DuneDecisionRole::kPurchase ||
                               role == DuneDecisionRole::kCombatIntrigue ||
                               role == DuneDecisionRole::kOtherOptional);
  if (is_short_window_role) {
    // Under exact per-root budgets the pool resets at EVERY in-scope root, not
    // only when the role changes -- otherwise two consecutive roots of the same
    // short-window role share one budget and the second receives zero.
    // The extra condition is deliberately ALSO gated on the budget being armed.
    // With a zero budget these roles return policy-only immediately below, so
    // resetting more often could only differ by a side effect of
    // short_bot_->Reset() -- and the narrow teacher must stay bit-identical for
    // the recorded parity chains to remain references.
    if (last_role_ != role ||
        (config_.exact_short_window_budgets &&
         config_.purchase_combat_budget > 0)) {
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
        resolved_live_deadline_ms < 0.0 ||
        (remaining_time_ms < 0.0 && resolved_live_deadline_ms <= 0.0)) {
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
    res.diagnostics.hard_sim_limit = ConfiguredHardSimLimit(is_short_window_role, role);
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

  // 1. Zero simulations for bookkeeping, and for a leader draft UNLESS the
  // WO-LEADER-1A flag makes it search-eligible. Splitting the two roles keeps
  // the default path (search_leader_draft == false) bit-for-bit identical:
  // both bookkeeping and leader nodes still return the "forced_or_bookkeeping"
  // policy-only result. When search_leader_draft is true, kLeaderSelection
  // falls through to the budget setup and search below.
  if (role == DuneDecisionRole::kForcedOrBookkeeping) {
    return get_policy_only_result("forced_or_bookkeeping");
  }
  if (role == DuneDecisionRole::kLeaderSelection && !config_.search_leader_draft) {
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
    // kPolicyOnly means policy-only at EVERY role, not only the agent ones.
    //
    // This check lived solely in the kAgentPrimary/kAgentContinuation branch
    // below, which was invisible for as long as the only raw control ever run
    // was the NARROW one: at purchase_combat_budget == 0 the short-window roles
    // took the early return immediately above and never reached a budget. The
    // moment the WIDE-MATCHED control (raw at budget 64) exists, the same
    // branch falls straight through to RunSearch -- so the control ran a real
    // 64-simulation search at every kPurchase / kCombatIntrigue /
    // kOtherOptional root, then discarded the result and played the raw-prior
    // argmax anyway. Behaviourally the arm was still a control; it simply paid
    // a searched arm's compute bill to be one, reported zero requested
    // simulations against nonzero completed ones, and would have failed the
    // registered "the control spent nothing" gate at hour N of the audit.
    //
    // Ordered AFTER the budget test on purpose, so the narrow control keeps
    // emitting "policy_only_purchase_combat" and nothing about the existing
    // instrument moves.
    if (budget_mode_ == DuneSearchBudgetMode::kPolicyOnly) {
      return get_policy_only_result("policy_only_mode");
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
    if (config_.exact_short_window_budgets) {
      // The registered watchdog is the ONLY time authority. The 500 ms window
      // below is a latency allowance for a live session; against an audit's
      // 64-simulation root it is a truncation nobody asked for and, because
      // these roles report through no role-stats bucket in the caller, one
      // nobody would see.
      max_time_ms = config_.relative_time_budget_ms;
    } else {
      max_time_ms = (config_.relative_time_budget_ms == std::numeric_limits<double>::infinity())
          ? std::numeric_limits<double>::infinity()
          : (500.0 - elapsed_ms);
    }

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
    } else if (budget_mode_ == DuneSearchBudgetMode::kTrainingPolicyIteration) {
      // INDEPENDENT per-role budgets. Deliberately no subtraction of
      // session_new_simulations_completed_ or short_sims_completed_: that
      // subtraction is what makes kFixedSessionSimulations a session POOL, and
      // pooling is exactly the property this mode exists to remove. The primary
      // spends pi_primary_simulations; every continuation then gets a full
      // pi_continuation_simulations of NEW simulations on the retained tree.
      max_sims = (role == DuneDecisionRole::kAgentPrimary)
                     ? config_.pi_primary_simulations
                     : config_.pi_continuation_simulations;
      if (max_sims <= 0) {
        max_sims = 0;
        limit_exceeded = true;
        limit_reason = "search_pi_budget_not_configured";
      }
      // A fixed-simulation search stops on the COUNT; the time budget is only a
      // runaway guard, as on the leader path.
      max_time_ms = config_.relative_time_budget_ms;
    } else if (budget_mode_ == DuneSearchBudgetMode::kLiveDeadline) {
      if (!live_deadline_initialized_ && resolved_live_deadline_ms > 0.0) {
        absolute_live_deadline_ =
            std::chrono::steady_clock::now() +
            std::chrono::milliseconds(
                static_cast<int64_t>(resolved_live_deadline_ms));
        live_deadline_budget_ms_ = resolved_live_deadline_ms;
        live_deadline_initialized_ = true;
      }
      if (resolved_live_deadline_ms == 0.0) {
        max_time_ms = 0.0;
      } else {
        auto now = std::chrono::steady_clock::now();
        double reserve_sec = 0.0;
        if (role == DuneDecisionRole::kAgentPrimary) {
          reserve_sec = config_.live_continuation_reserve_seconds;
          if (reserve_sec * 1000.0 >= live_deadline_budget_ms_) {
            reserve_sec = 0.5 * (live_deadline_budget_ms_ / 1000.0);
          }
        }
        double remaining_sec =
            std::chrono::duration<double>(absolute_live_deadline_ - now).count() -
            reserve_sec;
        max_time_ms = remaining_sec * 1000.0;
      }
      if (max_time_ms <= 0.0) {
        // Live deadline reached: calling RunSearch with max_time_ms == 0 trips
        // the expired-deadline pre-loop check and returns a uniform policy (the
        // same poison as the short-window path, but in live play). Degrade to
        // the raw network prior on the true current state instead.
        return get_policy_only_result("live_deadline_reached");
      }
      max_sims = config_.max_simulations;
    }
  } else if (role == DuneDecisionRole::kLeaderSelection) {
    // WO-LEADER-1A: search-eligible leader draft. Only reached when
    // config_.search_leader_draft is true (the default path returned the
    // "forced_or_bookkeeping" policy-only result above). A leader draft is a
    // once-per-seat decision that is neither a short-window role nor an
    // agent-turn role, so it runs on the placement bot with a dedicated,
    // fixed simulation budget independent of the session budgets.
    if (config_.leader_draft_simulations <= 0) {
      // Starved by configuration. Degrade to the raw network prior carrying
      // the leader node's OWN fallback reason -- never "forced_or_bookkeeping",
      // and never a uniform policy (get_policy_only_result returns the
      // evaluator prior, which SelectControllerAction argmaxes at temperature
      // 0). Obeys the search/eval parity invariant.
      return get_policy_only_result("leader_draft_no_budget");
    }
    max_sims = config_.leader_draft_simulations;
    // A fixed-simulation search stops on the COUNT; the time budget is only a
    // guard against a runaway clock. Reuse the configured relative budget so a
    // live/finite deadline still bounds the leader search.
    max_time_ms = config_.relative_time_budget_ms;
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
  result.diagnostics.hard_sim_limit = ConfiguredHardSimLimit(is_short_window_role, role);
  result.diagnostics.soft_sim_limit = max_sims;
  result.diagnostics.hard_time_limit_ms =
      ConfiguredHardTimeLimitMs(resolved_live_deadline_ms);
  result.diagnostics.soft_time_limit_ms = max_time_ms;
  result.diagnostics.elapsed_search_time_ms = result.elapsed_time_ms;
  // max_sim_duration_ms is populated by DunePUCTISMCTSBot::RunSearch and is
  // deliberately left intact here: it is the predictor margin the watchdog
  // actually used, not a session-level reconstruction.
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

  int64_t ckpt_input_dim = obs_size;
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
  {
    std::string json_path = checkpoint_path;
    if (json_path.size() >= 3 && json_path.substr(json_path.size() - 3) == ".pt") {
      json_path = json_path.substr(0, json_path.size() - 3) + ".json";
      std::ifstream jf(json_path);
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
        }
      }
    }
  }

  const bool has_scorer = CheckpointHasSemanticScorer(checkpoint_path, torch::kCPU);

  auto model = std::make_shared<SharedDunePolicyValueNetImpl>(
      ckpt_input_dim, hidden_dim, action_size, num_blocks,
      /*use_nonlinear=*/false, /*with_aux_heads=*/false, /*head_init_seed=*/0,
      /*with_semantic_scorer=*/has_scorer);
  LoadModelCheckpointRobust(model, checkpoint_path, device);
  model->to(device);
  model->eval();
  return std::make_shared<DuneNNEvaluator>(model, device, 10.0f, detected_mode);
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
  } else if (role == DuneDecisionRole::kLeaderSelection && config_.search_leader_draft) {
    // WO-LEADER-1A: consume the leader search result like an ordinary searched
    // role. On success search_result.policy is the visit-based final policy; on
    // a starved leader node Search() set it to the raw network prior, so this
    // reduces to the raw-prior argmax (never uniform), with an empty-policy
    // guard for defense in depth. Default-inert: when search_leader_draft is
    // false, kLeaderSelection falls through to the unchanged branch below.
    if (search_result.policy.empty()) {
      decision.selected_action = decision.raw_reference_action;
    } else {
      decision.selected_action = PickActionRespectingTemperature(
          search_result.policy, config_.temperature, r_val);
    }
    decision.mcts_overrode_raw = (decision.selected_action != decision.raw_reference_action);
    decision.confidence_fallback = search_result.used_fallback;
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

struct DuneAdaptiveRolloutSession::Impl {
  torch::Device device;
  std::shared_ptr<SharedDunePolicyValueNetImpl> model;
  std::unique_ptr<std::shared_mutex> mutex;
  std::vector<std::shared_ptr<BatchedEvaluator>> coordinators;
  std::vector<std::shared_ptr<BatchedNNEvaluator>> evaluators;
  dune_imperium::TurnSearchTree tree;

  explicit Impl(torch::Device dev) : device(dev) {}
};

DuneAdaptiveRolloutSession::DuneAdaptiveRolloutSession(
    const std::string& checkpoint_path,
    const std::string& device_str,
    int hidden_dim,
    int num_blocks,
    int reference_rollouts_per_action,
    int leader_rollouts,
    int target_batch_size,
    uint64_t seed,
    bool use_successive_halving,
    bool fixed_top5_candidates,
    int max_inflight_worlds,
    int num_evaluators,
    int top5_min_budget,
    bool all_buy_candidates)
    : checkpoint_path_(checkpoint_path),
      device_str_(device_str),
      hidden_dim_(hidden_dim),
      num_blocks_(num_blocks),
      reference_rollouts_per_action_(reference_rollouts_per_action),
      leader_rollouts_(leader_rollouts),
      target_batch_size_(target_batch_size),
      seed_(seed),
      use_successive_halving_(use_successive_halving),
      fixed_top5_candidates_(fixed_top5_candidates),
      max_inflight_worlds_(std::max(1, max_inflight_worlds)),
      num_evaluators_(std::max(1, num_evaluators)),
      top5_min_budget_(top5_min_budget),
      all_buy_candidates_(all_buy_candidates) {
  torch::Device device(torch::kCPU);
  if (device_str == "cuda" && torch::cuda::is_available()) {
    device = torch::Device(torch::kCUDA);
  } else if (device_str == "cpu") {
    device = torch::Device(torch::kCPU);
  } else {
    device = torch::Device(device_str);
  }

  impl_ = std::make_unique<Impl>(device);

  int64_t ckpt_input_dim = dune_imperium::kFullPublicInformationStateSize;
  {
    torch::serialize::InputArchive archive;
    archive.load_from(checkpoint_path_, torch::kCPU);
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
  {
    std::string json_path = checkpoint_path_;
    if (json_path.size() >= 3 && json_path.substr(json_path.size() - 3) == ".pt") {
      json_path = json_path.substr(0, json_path.size() - 3) + ".json";
      std::ifstream jf(json_path);
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
            hidden_dim_ = it_hd->second.GetInt();
          }
          auto it_nb = dict.find("num_blocks");
          if (it_nb != dict.end() && it_nb->second.IsInt()) {
            num_blocks_ = it_nb->second.GetInt();
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
        }
      }
    }
  }

  const bool has_scorer = has_semantic_scorer_flag ||
                          CheckpointHasSemanticScorer(checkpoint_path_, torch::kCPU);

  impl_->model = std::make_shared<SharedDunePolicyValueNetImpl>(
      ckpt_input_dim, hidden_dim_, dune_imperium::kNumDistinctPlayerActions,
      num_blocks_, /*use_nonlinear=*/false, /*with_aux_heads=*/false,
      /*head_init_seed=*/0, /*with_semantic_scorer=*/has_scorer,
      /*enable_zone_encoder=*/false, descriptor_schema);

  LoadModelCheckpointRobust(impl_->model, checkpoint_path_, torch::Device(torch::kCPU));
  impl_->model->to(impl_->device);
  impl_->model->eval();
  for (auto& parameter : impl_->model->parameters()) {
    parameter.set_requires_grad(false);
  }

  impl_->mutex = std::make_unique<std::shared_mutex>();
  impl_->coordinators.resize(num_evaluators_);
  impl_->evaluators.resize(num_evaluators_);
  for (int e = 0; e < num_evaluators_; ++e) {
    impl_->coordinators[e] = std::make_shared<BatchedEvaluator>(
        impl_->model, target_batch_size_, 1, impl_->device,
        impl_->mutex.get(), 10.0f, false, true, false,
        /*rollout_amp=*/false, /*allow_tf32=*/false);
    impl_->evaluators[e] = std::make_shared<BatchedNNEvaluator>(
        impl_->coordinators[e], 10.0f, detected_mode);
  }

  ResetSession("initialization");
}

DuneAdaptiveRolloutSession::~DuneAdaptiveRolloutSession() = default;

void DuneAdaptiveRolloutSession::DiscardPendingAction() {
  last_search_state_.reset();
  has_pending_commit_ = false;
}

void DuneAdaptiveRolloutSession::ResetSession(const std::string& reason) {
  if (impl_) {
    impl_->tree.Reset();
  }
  has_active_session_ = false;
  has_pending_commit_ = false;
  last_search_state_.reset();
  last_player_ = kInvalidPlayer;
  last_round_ = -1;
  last_hand_ = 0;
  last_intrigues_ = 0;
  last_history_.clear();
  session_new_simulations_completed_ = 0;
  session_elapsed_time_ms_ = 0.0;
  turn_elapsed_search_time_ms_ = 0.0;
  ms_per_rollout_by_round_ = {20.2, 18.5, 17.3, 15.2, 13.4, 10.6, 8.5, 6.1, 4.3, 3.2, 2.5, 2.0};
}

DuneSearchResult DuneAdaptiveRolloutSession::Search(const State& state, double remaining_time_ms) {
  if (has_pending_commit_) {
    SpielFatalError("DuneAdaptiveRolloutSession::Search called while a commit is still pending!");
  }
  const auto& dune_state = static_cast<const dune_imperium::DuneImperiumState&>(state);
  Player p = state.CurrentPlayer();

  if (!has_active_session_ || p != last_player_ || dune_state.GetCurrentRound() != last_round_) {
    impl_->tree.Reset();
    turn_elapsed_search_time_ms_ = 0.0;
  } else if (state.History().size() > last_history_.size() &&
             std::equal(last_history_.begin(), last_history_.end(), state.History().begin())) {
    if (dune_state.GetPlayerCardsInHand(p) > last_hand_ ||
        static_cast<int>(dune_state.GetPlayerIntrigues(p).size()) > last_intrigues_) {
      impl_->tree.Reset();
      turn_elapsed_search_time_ms_ = 0.0;
    } else {
      for (size_t i = last_history_.size(); i < state.History().size(); ++i) {
        Action a = state.History()[i];
        impl_->tree.Advance(a);
        if (a == dune_imperium::kActionEndTurn) {
          impl_->tree.Reset();
          turn_elapsed_search_time_ms_ = 0.0;
        }
      }
    }
  } else if (state.History() != last_history_) {
    impl_->tree.Reset();
    turn_elapsed_search_time_ms_ = 0.0;
  }
  impl_->tree.BindPublicCardKnowledge(state);

  has_active_session_ = true;
  last_player_ = p;
  last_round_ = dune_state.GetCurrentRound();
  last_hand_ = (p >= 0 && p < 4) ? dune_state.GetPlayerCardsInHand(p) : 0;
  last_intrigues_ = (p >= 0 && p < 4) ? static_cast<int>(dune_state.GetPlayerIntrigues(p).size()) : 0;
  last_history_ = state.History();

  const auto t_start = std::chrono::steady_clock::now();
  auto [raw_prior, values] = impl_->evaluators.front()->PriorAndEvaluate(state);
  std::vector<Action> legal = state.LegalActions();
  Action raw_action = legal.empty() ? kInvalidAction : dune_imperium::PickGreedyAction(raw_prior, legal);

  DuneSearchResult result;
  result.diagnostics.protocol_version = "adaptive_rollout";
  result.diagnostics.searched_seat = p;
  result.diagnostics.round = dune_state.GetCurrentRound();
  result.diagnostics.phase = LocalPhaseToString(dune_state.phase());
  result.diagnostics.actions = legal;
  result.diagnostics.priors.assign(legal.size(), 0.0);
  for (size_t i = 0; i < legal.size(); ++i) {
    Action a = legal[i];
    for (const auto& ap : raw_prior) {
      if (ap.first == a) {
        result.diagnostics.priors[i] = ap.second;
        break;
      }
    }
  }
  result.diagnostics.raw_priors = result.diagnostics.priors;
  result.diagnostics.root_value = (p >= 0 && p < static_cast<int>(values.size())) ? values[p] : 0.0;
  result.diagnostics.raw_reference_action = raw_action;
  result.diagnostics.visit_counts.assign(legal.size(), 0);
  result.diagnostics.q_values.assign(legal.size(), result.diagnostics.root_value);

  bool is_leader = (dune_state.phase() == dune_imperium::GamePhase::kLeaderDraft ||
                    dune_state.phase() == dune_imperium::GamePhase::kLeaderOfferChance);

  if (legal.size() <= 1) {
    result.diagnostics.selected_action = raw_action;
    result.diagnostics.mcts_proposed_action = raw_action;
    result.used_fallback = true;
    result.fallback_reason = (remaining_time_ms == 0.0) ? "live_deadline_reached" : "forced_or_bookkeeping";
    result.diagnostics.fallback_reason = result.fallback_reason;
    result.diagnostics.termination_reason = result.diagnostics.fallback_reason;
    result.policy = raw_prior;
  } else if (is_leader) {
    const double available_after_reserve_ms = (remaining_time_ms >= 0.0)
        ? (remaining_time_ms - reserve_ms_)
        : -1.0;

    int effective_leader_rollouts = leader_rollouts_;
    double target_ms = -1.0;
    bool deadline_reached = false;

    if (remaining_time_ms >= 0.0) {
      if (available_after_reserve_ms <= 0.0) {
        deadline_reached = true;
      } else {
        double usable_leader_ms = full_search_
            ? (0.965 * available_after_reserve_ms)
            : available_after_reserve_ms;
        target_ms = std::min(25000.0, usable_leader_ms);
        int num_cands = std::max(1, static_cast<int>(legal.size()));
        double leader_ms_est = full_search_
            ? std::max(2.0, ms_per_rollout_by_round_[0])
            : std::max(2.0, ms_per_rollout_ema_);
        double plan_ms = full_search_ ? (0.975 * target_ms) : target_ms;
        int total_leader_rollout_budget = static_cast<int>(plan_ms / leader_ms_est);
        effective_leader_rollouts = full_search_
            ? (total_leader_rollout_budget / num_cands)
            : std::min(2 * leader_rollouts_, total_leader_rollout_budget / num_cands);
        if (effective_leader_rollouts < 2) {
          deadline_reached = true;
        }
      }
    }
    result.diagnostics.hard_time_limit_ms = target_ms;
    result.diagnostics.soft_time_limit_ms = remaining_time_ms;

    if (deadline_reached) {
      result.diagnostics.selected_action = raw_action;
      result.diagnostics.mcts_proposed_action = raw_action;
      result.used_fallback = true;
      result.fallback_reason = "live_deadline_reached";
      result.diagnostics.fallback_reason = "live_deadline_reached";
      result.diagnostics.termination_reason = result.diagnostics.fallback_reason;
      result.policy = raw_prior;
    } else {
      int num_cands = std::max(1, static_cast<int>(legal.size()));
      int leader_inflight = full_search_
          ? std::max(max_inflight_worlds_, 384 / num_cands)
          : max_inflight_worlds_;
      double leader_deadline_ms = (full_search_ && target_ms > 0.0) ? (1.008 * target_ms) : target_ms;
      const auto c_res = dune_imperium::LeaderPickRolloutSearchDecision(
          state, p, impl_->evaluators, effective_leader_rollouts, 0.10,
          dune_seed::DeriveSeed(seed_, 0x4C454144ULL, static_cast<uint64_t>(p), search_seed_counter_++),
          dune_imperium::SearchRolloutCrnMode::kSequential,
          /*sample_opponent_leader_draft=*/false,
          /*max_time_ms=*/leader_deadline_ms,
          leader_inflight);
      result.diagnostics.selected_action = c_res.action;
      result.diagnostics.mcts_proposed_action = c_res.action;
      result.diagnostics.mcts_overrode_raw = c_res.overridden;
      result.simulations_completed = c_res.new_rollouts_executed;
      result.diagnostics.newly_completed_simulations = c_res.new_rollouts_executed;
      result.diagnostics.inherited_root_visits = c_res.prior_rollouts_reused;
      result.diagnostics.stage1_planned_worlds = c_res.stage1_planned_worlds;
      result.diagnostics.stage1_kept_worlds = c_res.stage1_kept_worlds;
      result.diagnostics.stage2_planned_worlds = c_res.stage2_planned_worlds;
      result.diagnostics.stage2_kept_worlds = c_res.stage2_kept_worlds;
      result.diagnostics.stage1_elapsed_ms = c_res.stage1_elapsed_ms;
      result.diagnostics.stage2_elapsed_ms = c_res.stage2_elapsed_ms;
      result.diagnostics.termination_reason = c_res.termination_reason;
      result.policy = raw_prior;

      for (const auto& cu : c_res.candidate_utilities) {
        for (size_t i = 0; i < legal.size(); ++i) {
          if (legal[i] == cu.first) {
            result.diagnostics.q_values[i] = cu.second;
            break;
          }
        }
      }
      for (const auto& cru : c_res.candidate_rollout_utilities) {
        for (size_t i = 0; i < legal.size(); ++i) {
          if (legal[i] == cru.first) {
            result.diagnostics.visit_counts[i] = static_cast<int>(cru.second.size());
            break;
          }
        }
      }
      const int retained_pairs = c_res.candidate_rollout_utilities.empty()
          ? 0
          : static_cast<int>(c_res.candidate_rollout_utilities[0].second.size());
      if (remaining_time_ms >= 0.0 && (retained_pairs < 2 || c_res.new_rollouts_executed < 2 * num_cands)) {
        result.used_fallback = true;
        result.fallback_reason = "live_deadline_reached";
        result.diagnostics.fallback_reason = "live_deadline_reached";
      }
    }
  } else {
    const double available_after_reserve_ms = (remaining_time_ms >= 0.0)
        ? (remaining_time_ms - reserve_ms_)
        : -1.0;

    bool ran_high_stakes = false;
    bool live_deadline_starved = false;

    bool should_search = dune_imperium::IsHighStakesStrategicState(state, p);
    if (full_search_ && !should_search && state.CurrentPlayer() == p && legal.size() >= 2) {
      if (dune_imperium::IsPurchaseDecisionState(state) ||
          dune_state.phase() == dune_imperium::GamePhase::kCombat) {
        should_search = true;
      } else {
        bool has_end_or_reveal = false;
        for (Action a : legal) {
          if (a == dune_imperium::kActionEndTurn || a == dune_imperium::kActionReveal) {
            has_end_or_reveal = true;
            break;
          }
        }
        bool has_tree_reuse = false;
        if (impl_->tree.current_node() && !impl_->tree.current_node()->children.empty()) {
          for (Action a : legal) {
            if (impl_->tree.HasPriorData(a)) {
              has_tree_reuse = true;
              break;
            }
          }
        }
        if (!has_end_or_reveal && !has_tree_reuse) {
          should_search = true;
        } else if (has_end_or_reveal &&
                   dune_imperium::IsFullStrategicState(state, p, raw_prior, 0.020)) {
          should_search = true;
        }
      }
    }

    if (should_search) {
      if (remaining_time_ms >= 0.0 && available_after_reserve_ms <= 0.0) {
        live_deadline_starved = true;
      } else {
        int ref_rollouts = reference_rollouts_per_action_;
        double target_ms = -1.0;
        const bool is_agent_turn_phase =
            (dune_state.phase() == dune_imperium::GamePhase::kAgentTurns &&
             !dune_state.IsPlayerInRevealTurn(p));

        int eff_r = dune_state.GetCurrentRound();
        if (dune_state.phase() == dune_imperium::GamePhase::kCombat ||
            dune_state.phase() == dune_imperium::GamePhase::kRevealTurns ||
            dune_state.IsPlayerInRevealTurn(p)) {
          eff_r += 1;
        }
        const int r_bucket = std::clamp(eff_r, 1, 11);
        const double round_ms_est = full_search_
            ? std::max(0.4, ms_per_rollout_by_round_[r_bucket])
            : std::max(2.0, ms_per_rollout_ema_);

        if (remaining_time_ms >= 0.0) {
          double usable_ms = full_search_
              ? ((is_agent_turn_phase ? 0.980 : 0.980) * available_after_reserve_ms)
              : available_after_reserve_ms;
          double frac = is_agent_turn_phase ? 0.60 : (full_search_ ? 1.0 : 0.75);
          double cap = is_agent_turn_phase ? 25000.0 : 30000.0;
          if (full_search_ && is_agent_turn_phase) {
            const int hand0 = dune_state.GetPlayerCardsInHand(p);
            const int int0 = static_cast<int>(dune_state.GetPlayerIntrigues(p).size());
            auto count_branch_searched = [&](Action first_action, int max_count) -> int {
              auto sim = state.Clone();
              sim->ApplyAction(first_action);
              bool tree_broken = false;
              int rem_searched = 0;
              for (int step = 0; step < 16 && !sim->IsTerminal(); ++step) {
                if (sim->IsChanceNode()) {
                  tree_broken = true;
                  auto outs = sim->ChanceOutcomes();
                  if (outs.empty()) break;
                  sim->ApplyAction(outs[0].first);
                  continue;
                }
                if (sim->CurrentPlayer() != p) break;
                const auto& sim_d = static_cast<const dune_imperium::DuneImperiumState&>(*sim);
                if (sim_d.phase() != dune_imperium::GamePhase::kAgentTurns ||
                    sim_d.IsPlayerInRevealTurn(p)) {
                  break;
                }
                if (sim_d.GetPlayerCardsInHand(p) > hand0 ||
                    static_cast<int>(sim_d.GetPlayerIntrigues(p).size()) > int0) {
                  tree_broken = true;
                }
                std::vector<Action> nl = sim->LegalActions();
                if (nl.empty()) break;
                if (nl.size() == 1) {
                  if (nl[0] == dune_imperium::kActionEndTurn ||
                      nl[0] == dune_imperium::kActionReveal) {
                    break;
                  }
                  sim->ApplyAction(nl[0]);
                  continue;
                }
                bool has_end_or_reveal = false;
                int non_spec = 0;
                for (Action na : nl) {
                  if (na == dune_imperium::kActionEndTurn ||
                      na == dune_imperium::kActionReveal) {
                    has_end_or_reveal = true;
                  }
                  if (!dune_imperium::IsConvertSpecimenAction(na)) {
                    non_spec++;
                  }
                }
                if (has_end_or_reveal && non_spec < 2) {
                  break;
                }
                bool hs_or_buy = dune_imperium::IsHighStakesStrategicState(*sim, p) ||
                                 dune_imperium::IsPurchaseDecisionState(*sim);
                ActionsAndProbs sub_prior = impl_->evaluators.front()->Prior(*sim);
                bool sub_search = hs_or_buy;
                if (!sub_search) {
                  if (!has_end_or_reveal && tree_broken) {
                    sub_search = true;
                  } else if (has_end_or_reveal &&
                             dune_imperium::IsFullStrategicState(*sim, p, sub_prior, 0.020)) {
                    sub_search = true;
                  }
                }
                if (sub_search) {
                  rem_searched++;
                  if (rem_searched >= max_count) break;
                }
                Action next_a = dune_imperium::PickGreedyAction(sub_prior, nl);
                if (next_a == dune_imperium::kActionEndTurn ||
                    next_a == dune_imperium::kActionReveal) {
                  break;
                }
                sim->ApplyAction(next_a);
              }
              return rem_searched;
            };

            int rem_0 = 0;
            int max_other_rem = 0;
            const auto sorted_cands = dune_imperium::GetSortedCandidatesByPrior(raw_prior, legal);
            if (!sorted_cands.empty()) {
              rem_0 = count_branch_searched(sorted_cands[0].first, 3);
            }
            const size_t k_check = std::min<size_t>(5, sorted_cands.size());
            for (size_t ci = 1; ci < k_check; ++ci) {
              int r_ci = count_branch_searched(sorted_cands[ci].first, 1);
              max_other_rem = std::max(max_other_rem, r_ci);
              if (max_other_rem >= 1) break;
            }

            cap = 49200.0;
            if (rem_0 == 0) {
              frac = (max_other_rem == 0) ? 1.0 : 0.94;
            } else if (rem_0 == 1) {
              if (usable_ms >= 38000.0) {
                frac = 0.50;
              } else {
                frac = (max_other_rem == 0) ? 0.68 : 0.65;
              }
            } else if (rem_0 == 2) {
              frac = (usable_ms >= 38000.0) ? 0.45 : 0.54;
            } else {
              frac = 0.40;
            }
          }
          target_ms = std::min(cap, frac * usable_ms);
          result.diagnostics.hard_time_limit_ms = target_ms;
          result.diagnostics.soft_time_limit_ms = remaining_time_ms;
          double plan_ms = full_search_ ? (0.94 * target_ms) : target_ms;
          int total_rollout_budget = static_cast<int>(plan_ms / round_ms_est);
          ref_rollouts = full_search_
              ? (total_rollout_budget / 3)
              : std::min(4 * reference_rollouts_per_action_, total_rollout_budget / 3);
          if (full_search_ && is_agent_turn_phase &&
              usable_ms < std::max(1800.0, 24.0 * round_ms_est)) {
            ref_rollouts = 0;
          }
        }

        if (remaining_time_ms >= 0.0 && ref_rollouts < 4) {
          live_deadline_starved = true;
        } else {
          ran_high_stakes = true;
          const uint64_t decision_seed =
              dune_seed::DeriveSeed(seed_, static_cast<uint64_t>(p), search_seed_counter_++);
          const int min_top5_budget = full_search_ ? std::min(top5_min_budget_, 96) : top5_min_budget_;
          const bool use_top5 = fixed_top5_candidates_ && 3 * ref_rollouts >= min_top5_budget;
          int effective_top_k = use_top5 ? 5 : 0;
          if (all_buy_candidates_ && dune_imperium::IsPurchaseDecisionState(state) &&
              (use_top5 || (full_search_ && 3 * ref_rollouts >= 48))) {
            effective_top_k = static_cast<int>(legal.size());
          }
          const auto c_res = use_successive_halving_
              ? dune_imperium::SuccessiveHalvingRolloutSearchDecision(
                    state, p, &(impl_->tree), impl_->evaluators, ref_rollouts, 0.15, decision_seed,
                    dune_imperium::SearchRolloutCrnMode::kSequential, /*unbiased_stage2_gate=*/false,
                    target_ms, effective_top_k, max_inflight_worlds_,
                    /*use_bounded_pool_when_unclocked=*/false,
                    /*full_clock_stage2=*/full_search_)
              : dune_imperium::AdaptiveRolloutSearchDecision(
                    state, p, &(impl_->tree), impl_->evaluators, ref_rollouts, 0.15, decision_seed,
                    dune_imperium::SearchRolloutCrnMode::kSequential, target_ms, max_inflight_worlds_);
          result.diagnostics.selected_action = c_res.action;
          result.diagnostics.mcts_proposed_action = c_res.action;
          result.diagnostics.mcts_overrode_raw = c_res.overridden;
          result.simulations_completed = c_res.new_rollouts_executed;
          result.diagnostics.newly_completed_simulations = c_res.new_rollouts_executed;
          result.diagnostics.inherited_root_visits = c_res.prior_rollouts_reused;
          result.diagnostics.stage1_planned_worlds = c_res.stage1_planned_worlds;
          result.diagnostics.stage1_kept_worlds = c_res.stage1_kept_worlds;
          result.diagnostics.stage2_planned_worlds = c_res.stage2_planned_worlds;
          result.diagnostics.stage2_kept_worlds = c_res.stage2_kept_worlds;
          result.diagnostics.stage1_elapsed_ms = c_res.stage1_elapsed_ms;
          result.diagnostics.stage2_elapsed_ms = c_res.stage2_elapsed_ms;
          result.diagnostics.termination_reason = c_res.termination_reason;
          result.policy = raw_prior;

          for (const auto& cu : c_res.candidate_utilities) {
            for (size_t i = 0; i < legal.size(); ++i) {
              if (legal[i] == cu.first) {
                result.diagnostics.q_values[i] = cu.second;
                break;
              }
            }
          }
          for (const auto& cru : c_res.candidate_rollout_utilities) {
            for (size_t i = 0; i < legal.size(); ++i) {
              if (legal[i] == cru.first) {
                result.diagnostics.visit_counts[i] = static_cast<int>(cru.second.size());
                break;
              }
            }
          }
          const int retained_pairs = c_res.candidate_rollout_utilities.empty()
              ? 0
              : static_cast<int>(c_res.candidate_rollout_utilities[0].second.size());
          if (remaining_time_ms >= 0.0 && retained_pairs < 2) {
            result.used_fallback = true;
            result.fallback_reason = "live_deadline_reached";
            result.diagnostics.fallback_reason = "live_deadline_reached";
          }
        }
      }
    }

    if (!ran_high_stakes) {
      Action chosen = raw_action;
      int inherited_visits = 0;
      int covered_in_tree = 0;
      if (impl_->tree.current_node() && !impl_->tree.current_node()->children.empty()) {
        double best = -1e9;
        for (size_t i = 0; i < legal.size(); ++i) {
          Action a = legal[i];
          if (impl_->tree.HasPriorData(a)) {
            int v = impl_->tree.GetVisits(a);
            double q = impl_->tree.GetMeanUtility(a);
            result.diagnostics.visit_counts[i] = v;
            result.diagnostics.q_values[i] = q;
            inherited_visits += v;
            covered_in_tree++;
            if (q > best) {
              best = q;
              chosen = a;
            }
          }
        }
      }

      if (covered_in_tree > 0) {
        result.diagnostics.selected_action = chosen;
        result.diagnostics.mcts_proposed_action = chosen;
        result.diagnostics.mcts_overrode_raw = (chosen != raw_action);
        result.used_fallback = false;
        result.fallback_reason = "none";
        result.diagnostics.fallback_reason = "none";
        result.diagnostics.termination_reason = "tree_reuse";
        result.diagnostics.inherited_root_visits = inherited_visits;
        result.policy = raw_prior;
      } else {
        result.diagnostics.selected_action = raw_action;
        result.diagnostics.mcts_proposed_action = raw_action;
        result.used_fallback = true;
        std::string reason = (live_deadline_starved || (remaining_time_ms >= 0.0 && available_after_reserve_ms <= 0.0))
            ? "live_deadline_reached"
            : "policy_only";
        result.fallback_reason = reason;
        result.diagnostics.fallback_reason = reason;
        result.diagnostics.termination_reason = reason;
        result.policy = raw_prior;
      }
    }
  }

  int covered = 0;
  int total_visits = 0;
  for (int v : result.diagnostics.visit_counts) {
    total_visits += v;
    if (v > 0) covered++;
  }
  result.diagnostics.total_root_visits = total_visits;
  result.diagnostics.num_covered_actions = covered;

  const auto t_end = std::chrono::steady_clock::now();
  double elapsed_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
  result.elapsed_time_ms = elapsed_ms;
  result.diagnostics.elapsed_search_time_ms = elapsed_ms;
  session_elapsed_time_ms_ += elapsed_ms;
  turn_elapsed_search_time_ms_ += elapsed_ms;
  session_new_simulations_completed_ += result.simulations_completed;
  if (remaining_time_ms >= 0.0 && result.simulations_completed >= 6 &&
      !result.used_fallback && elapsed_ms > 0.0) {
    double current_rate = elapsed_ms / result.simulations_completed;
    ms_per_rollout_ema_ = 0.8 * ms_per_rollout_ema_ + 0.2 * current_rate;
    if (result.simulations_completed >= 160) {
      int r_bucket = 0;
      if (!is_leader) {
        int eff_r = dune_state.GetCurrentRound();
        if (dune_state.phase() == dune_imperium::GamePhase::kCombat ||
            dune_state.phase() == dune_imperium::GamePhase::kRevealTurns ||
            dune_state.IsPlayerInRevealTurn(p)) {
          eff_r += 1;
        }
        r_bucket = std::clamp(eff_r, 1, 11);
      }
      double clamped_rate = std::clamp(current_rate, 0.4, 35.0);
      ms_per_rollout_by_round_[r_bucket] =
          0.8 * ms_per_rollout_by_round_[r_bucket] + 0.2 * clamped_rate;
    }
  }

  last_search_state_ = state.Clone();
  last_search_result_ = result;
  has_pending_commit_ = true;
  return result;
}

} // namespace open_spiel
