#include "dune_search_benchmark_internal.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <limits>
#include <sstream>

#include "dune_split_evaluator.h"
#include "open_spiel/abseil-cpp/absl/flags/flag.h"
#include "open_spiel/abseil-cpp/absl/random/distributions.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace {

class DuneGreedyBot : public Bot {
 public:
  DuneGreedyBot(std::shared_ptr<open_spiel::algorithms::Evaluator> evaluator,
                int seed, double temperature)
      : evaluator_(std::move(evaluator)), rng_(seed), temperature_(temperature) {}

  Action Step(const State& state) override {
    if (state.IsTerminal()) return kInvalidAction;
    ActionsAndProbs prior = evaluator_->Prior(state);
    return SelectBestAction(prior, state);
  }

  bool ProvidesPolicy() override { return true; }

  ActionsAndProbs GetPolicy(const State& state) override {
    return evaluator_->Prior(state);
  }

  std::pair<ActionsAndProbs, Action> StepWithPolicy(const State& state) override {
    ActionsAndProbs policy = GetPolicy(state);
    Action chosen = SelectBestAction(policy, state);
    return {policy, chosen};
  }

  void Restart() override {}
  void RestartAt(const State& state) override {}

 private:
  double RandomNumber() {
    return absl::Uniform(rng_, 0.0, 1.0);
  }

  Action SelectBestAction(const ActionsAndProbs& policy, const State& state) {
    if (policy.empty()) {
      auto legals = state.LegalActions();
      return legals.empty() ? kInvalidAction : legals.front();
    }
    double temp = temperature_;
    if (temp <= 0.0) {
      Action best_action = policy[0].first;
      double best_prob = policy[0].second;
      for (const auto& ap : policy) {
        if (ap.second > best_prob) {
          best_prob = ap.second;
          best_action = ap.first;
        }
      }
      return best_action;
    } else {
      ActionsAndProbs scaled_policy;
      scaled_policy.reserve(policy.size());
      double sum = 0.0;
      double inv_temp = 1.0 / temp;
      for (const auto& ap : policy) {
        double p = std::pow(std::max(ap.second, 1e-12), inv_temp);
        scaled_policy.push_back({ap.first, p});
        sum += p;
      }
      for (auto& ap : scaled_policy) {
        ap.second /= sum;
      }
      return SampleAction(scaled_policy, RandomNumber()).first;
    }
  }

  std::shared_ptr<open_spiel::algorithms::Evaluator> evaluator_;
  std::mt19937 rng_;
  double temperature_;
};

class DuneRandomBot : public Bot {
 public:
  explicit DuneRandomBot(int seed) : rng_(seed) {}
  Action Step(const State& state) override {
    if (state.IsTerminal()) return kInvalidAction;
    auto legals = state.LegalActions();
    if (legals.empty()) return kInvalidAction;
    return legals[absl::Uniform(rng_, 0u, legals.size())];
  }
  bool ProvidesPolicy() override { return false; }
  ActionsAndProbs GetPolicy(const State& state) override { return {}; }
  std::pair<ActionsAndProbs, Action> StepWithPolicy(const State& state) override {
    return {{}, Step(state)};
  }
  void Restart() override {}
  void RestartAt(const State& state) override {}

 private:
  std::mt19937 rng_;
};

}  // namespace

// Parses --fresh_search_roles (comma-separated role tokens) into a role set.
// Empty input -> empty set, which the driver treats as "search every role"
// (unchanged full-search behavior). Unknown tokens are a fatal misconfiguration.
std::set<DuneDecisionRole> ParseFreshSearchRoles(const std::string& csv) {
  std::set<DuneDecisionRole> roles;
  std::stringstream ss(csv);
  std::string tok;
  while (std::getline(ss, tok, ',')) {
    size_t b = tok.find_first_not_of(" \t");
    size_t e = tok.find_last_not_of(" \t");
    if (b == std::string::npos) continue;  // skip empty/whitespace token
    tok = tok.substr(b, e - b + 1);
    if (tok == "primary") roles.insert(DuneDecisionRole::kAgentPrimary);
    else if (tok == "continuation") roles.insert(DuneDecisionRole::kAgentContinuation);
    else if (tok == "purchase") roles.insert(DuneDecisionRole::kPurchase);
    else if (tok == "combat") roles.insert(DuneDecisionRole::kCombatIntrigue);
    else if (tok == "other") roles.insert(DuneDecisionRole::kOtherOptional);
    else if (tok == "leader") roles.insert(DuneDecisionRole::kLeaderSelection);
    else if (tok == "forced") roles.insert(DuneDecisionRole::kForcedOrBookkeeping);
    else SpielFatalError("Unrecognized --fresh_search_roles token: '" + tok + "'");
  }
  return roles;
}

SeatControllerMode ParseControllerMode(const std::string& s) {
  if (s == "single") return SeatControllerMode::kSingle;
  if (s == "homogeneous") return SeatControllerMode::kHomogeneous;
  if (s == "mixed") return SeatControllerMode::kMixed;
  SpielFatalError("Unrecognized --controller_mode: '" + s +
                  "' (expected single|homogeneous|mixed)");
}

MixedOpponentController ParseMixedOpponentController(const std::string& s) {
  if (s == "random_legal") return MixedOpponentController::kRandomLegal;
  if (s == "raw_policy") return MixedOpponentController::kRawPolicy;
  SpielFatalError("Unrecognized --mixed_opponent_controller: '" + s +
                  "' (expected random_legal|raw_policy)");
}

std::string BudgetModeName(DuneSearchBudgetMode m) {
  switch (m) {
    case DuneSearchBudgetMode::kPolicyOnly: return "policy_only";
    case DuneSearchBudgetMode::kFixedSessionSimulations:
      return "fixed_session_simulations";
    case DuneSearchBudgetMode::kTrainingFullFast: return "training_full_fast";
    case DuneSearchBudgetMode::kLiveDeadline: return "live_deadline";
    case DuneSearchBudgetMode::kTrainingPolicyIteration:
      return "training_policy_iteration";
  }
  return "unknown";
}

std::string DuneDecisionRoleToString(open_spiel::DuneDecisionRole role) {
  switch (role) {
    case open_spiel::DuneDecisionRole::kForcedOrBookkeeping: return "kForcedOrBookkeeping";
    case open_spiel::DuneDecisionRole::kLeaderSelection: return "kLeaderSelection";
    case open_spiel::DuneDecisionRole::kAgentPrimary: return "kAgentPrimary";
    case open_spiel::DuneDecisionRole::kAgentContinuation: return "kAgentContinuation";
    case open_spiel::DuneDecisionRole::kPurchase: return "kPurchase";
    case open_spiel::DuneDecisionRole::kCombatIntrigue: return "kCombatIntrigue";
    case open_spiel::DuneDecisionRole::kOtherOptional: return "kOtherOptional";
    default: return "kUnknown";
  }
}

std::shared_ptr<SharedDunePolicyValueNetImpl> LoadDuneBenchmarkModel(
    const std::string& ckpt_path,
    int64_t obs_size,
    int64_t action_size,
    int h_dim,
    int n_blocks,
    bool nonlinear,
    torch::Device device) {
  int64_t model_dim = obs_size;
  {
    torch::serialize::InputArchive archive;
    archive.load_from(ckpt_path, torch::kCPU);
    torch::serialize::InputArchive in_archive;
    if (archive.try_read("input_layer", in_archive)) {
      torch::Tensor w;
      if (in_archive.try_read("weight", w)) {
        model_dim = w.size(1);
      }
    }
  }
  const bool has_scorer = open_spiel::CheckpointHasSemanticScorer(ckpt_path, torch::kCPU);
  auto m = std::make_shared<open_spiel::SharedDunePolicyValueNetImpl>(
      model_dim, h_dim, action_size, n_blocks, nonlinear,
      /*with_aux_heads=*/false, /*head_init_seed=*/0,
      /*with_semantic_scorer=*/has_scorer);
  open_spiel::LoadModelCheckpointRobust(m, ckpt_path, device);
  m->to(device);
  m->eval();
  return m;
}

void ConfigureOutputPathsFromOutputDir(bool record_market_diagnostics) {
  std::string output_dir = absl::GetFlag(FLAGS_output_dir);
  if (output_dir.empty()) return;

  std::filesystem::create_directories(output_dir);
  if (absl::GetFlag(FLAGS_game_jsonl_path).empty()) {
    absl::SetFlag(&FLAGS_game_jsonl_path, output_dir + "/games.jsonl");
  }
  if (absl::GetFlag(FLAGS_aggregate_json_path).empty()) {
    absl::SetFlag(&FLAGS_aggregate_json_path, output_dir + "/aggregate.json");
  }
  if (absl::GetFlag(FLAGS_record_candidate_decision_traces) &&
      absl::GetFlag(FLAGS_candidate_decision_traces_path).empty()) {
    absl::SetFlag(&FLAGS_candidate_decision_traces_path,
                  output_dir + "/candidate_decision_traces.jsonl");
  }
  if (record_market_diagnostics) {
    if (absl::GetFlag(FLAGS_market_diag_json_path).empty()) {
      absl::SetFlag(&FLAGS_market_diag_json_path,
                    output_dir + "/market_diagnostics.json");
    }
    if (absl::GetFlag(FLAGS_market_diag_csv_path).empty()) {
      absl::SetFlag(&FLAGS_market_diag_csv_path,
                    output_dir + "/market_diagnostics.csv");
    }
    if (absl::GetFlag(FLAGS_market_diag_games_path).empty()) {
      absl::SetFlag(&FLAGS_market_diag_games_path,
                    output_dir + "/market_diagnostics_games.jsonl");
    }
  }
}

std::shared_ptr<open_spiel::algorithms::Evaluator> BuildSearchEvaluator(
    const std::shared_ptr<SharedDunePolicyValueNetImpl>& search_model,
    const std::shared_ptr<SharedDunePolicyValueNetImpl>& search_value_model,
    const std::shared_ptr<open_spiel::BatchedEvaluator>& search_coord,
    torch::Device device,
    float candidate_logit_cap) {
  if (search_coord != nullptr) {
    return std::make_shared<open_spiel::BatchedNNEvaluator>(
        search_coord, candidate_logit_cap);
  }
  if (g_using_split_evaluator) {
    auto policy_eval = std::make_shared<DuneNNEvaluator>(
        search_model, device, candidate_logit_cap);
    auto value_model_ptr = search_value_model ? search_value_model : search_model;
    auto value_eval = std::make_shared<DuneNNEvaluator>(
        value_model_ptr, device, candidate_logit_cap);
    return std::make_shared<SplitPolicyValueEvaluator>(policy_eval, value_eval);
  }
  return std::make_shared<DuneNNEvaluator>(
      search_model, device, candidate_logit_cap);
}

GameSeatSetup SetupGameSeats(
    std::mt19937& game_rng,
    SeatControllerMode controller_mode,
    MixedOpponentController mixed_opponent,
    int search_seat,
    float candidate_logit_cap,
    float opponent_logit_cap,
    torch::Device device,
    const std::shared_ptr<open_spiel::algorithms::Evaluator>& search_evaluator,
    const std::shared_ptr<SharedDunePolicyValueNetImpl>& opponent_model,
    const std::shared_ptr<open_spiel::BatchedEvaluator>& opp_coord) {
  GameSeatSetup setup;
  // Which seats run the search controller this game. In 'single' (legacy) and
  // 'mixed' that is exactly the candidate seat; in 'homogeneous' it is all
  // four. `search_seat` remains the CANDIDATE seat in every mode, so all the
  // candidate-relative statistics below keep their meaning.
  for (int p = 0; p < 4; ++p) {
    setup.seat_is_searched[p] = (controller_mode == SeatControllerMode::kHomogeneous)
                                    ? true
                                    : (p == search_seat);
  }

  // One session per searched seat. Index `search_seat` is the candidate's.
  setup.seat_sessions.resize(4);
  // Per-seat evaluators: searched seats share the candidate evaluator (all
  // four copies are the SAME controller in homogeneous mode, by definition).
  setup.seat_evaluators.resize(4);
  setup.bots.resize(4);

  // PF-2 Part B: remember each seat's DuneSearchConfig::seed. Path-B derives
  // its per-decision sampling seed from config_.seed, so the matched arm must
  // derive from the SAME value or the streams cannot agree. Recording it
  // rather than re-drawing is essential — the seat draw below is exactly one
  // game_rng() call per seat in seat order, and any extra draw reshuffles
  // every downstream chance realization.
  for (int p = 0; p < 4; ++p) {
    // Exactly ONE game_rng() draw per seat, in seat order, in EVERY mode.
    // The legacy code drew once for the searched seat's config.seed and once
    // per opponent bot, which is the same count and order; preserving that is
    // what keeps 'single' mode bitwise-identical to the pre-WO-1 binary.
    // Moving, adding or conditionalising this draw silently reshuffles every
    // downstream chance realization.
    const uint64_t seat_seed = game_rng();
    setup.seat_config_seed[p] = seat_seed;
    if (setup.seat_is_searched[p]) {
      // Designated initializers bind by NAME. This block used to be
      // positional, and Phase 18B's three inserted KataGo fields shifted
      // everything after dirichlet_alpha by three slots, so this binary
      // silently ran with dirichlet_alpha_total=1.0 and
      // root_noise_fpu_zero=true while DROPPING --check_strategic_state,
      // --verbose_diagnostics and --root_prior_temperature entirely
      // (bool<->double in a braced-init list is only a -Wnarrowing warning).
      // See WO-15 / search finding 2 — latent in the legacy constructor, live
      // here.
      DuneSearchConfig config{
          .max_simulations = absl::GetFlag(FLAGS_max_simulations),
          .relative_time_budget_ms =
              absl::GetFlag(FLAGS_disable_time_limit)
                  ? std::numeric_limits<double>::infinity()
                  : absl::GetFlag(FLAGS_relative_time_budget_ms),
          .max_nodes = absl::GetFlag(FLAGS_max_nodes),
          .puct_c = absl::GetFlag(FLAGS_puct_c),
          .opponent_mode = absl::GetFlag(FLAGS_use_opponent_model)
                               ? SearchOpponentMode::kPolicy
                               : SearchOpponentMode::kMaxN,
          .temperature = absl::GetFlag(FLAGS_temperature),
          .opponent_temperature =
              absl::GetFlag(FLAGS_simulated_opponent_temperature),
          .max_world_samples = absl::GetFlag(FLAGS_max_world_samples),
          .utility_divisor = absl::GetFlag(FLAGS_utility_divisor),
          .min_visit_threshold = 2,
          .covered_prior_threshold = 0.50,
          .seed = seat_seed,
          .final_policy_type = DuneISMCTSFinalPolicyType::kNormalizedVisitCount,
          .dirichlet_epsilon = absl::GetFlag(FLAGS_dirichlet_epsilon),
          .dirichlet_alpha = absl::GetFlag(FLAGS_dirichlet_alpha),
          .use_observation_string = true,
          .verbose_diagnostics = absl::GetFlag(FLAGS_verbose_diagnostics),
          .check_strategic_state = absl::GetFlag(FLAGS_check_strategic_state),
          .root_prior_temperature =
              absl::GetFlag(FLAGS_root_prior_temperature),
      };
      // WO-LEADER-1A: search-eligible leader drafts (default-inert).
      config.search_leader_draft = absl::GetFlag(FLAGS_search_leader_draft);
      config.leader_draft_simulations = absl::GetFlag(FLAGS_leader_draft_simulations);
      config.leader_mass_only_coverage =
          absl::GetFlag(FLAGS_leader_mass_only_coverage);
      config.fixed_session_limit = absl::GetFlag(FLAGS_max_simulations);
      config.fixed_continuation_reserve = absl::GetFlag(FLAGS_fixed_continuation_reserve);
      config.purchase_combat_budget = absl::GetFlag(FLAGS_purchase_combat_budget);
      if (absl::GetFlag(FLAGS_audit_training_policy_iteration)) {
        config.pi_primary_simulations =
            absl::GetFlag(FLAGS_pi_primary_simulations);
        config.pi_continuation_simulations =
            absl::GetFlag(FLAGS_pi_continuation_simulations);
        config.purchase_combat_budget =
            absl::GetFlag(FLAGS_pi_short_window_simulations);
        // The training teacher resets the short-window budget at every root;
        // it is not a 500 ms role window shared by consecutive roots.
        config.exact_short_window_budgets = true;
        config.policy_only_covers_short_window = false;
      }
      config.live_continuation_reserve_seconds = absl::GetFlag(FLAGS_live_continuation_reserve_seconds);
      config.model_checkpoint_path = absl::GetFlag(FLAGS_model_checkpoint);
      config.conservative_override_enabled = absl::GetFlag(FLAGS_conservative_override_enabled);
      config.conservative_covered_prior_threshold = absl::GetFlag(FLAGS_conservative_covered_prior_threshold);
      config.conservative_meaningful_visit_threshold = absl::GetFlag(FLAGS_conservative_meaningful_visit_threshold);
      config.conservative_q_margin_threshold = absl::GetFlag(FLAGS_conservative_q_margin_threshold);
      config.conservative_stability_checkpoint_fraction = absl::GetFlag(FLAGS_conservative_stability_checkpoint_fraction);
      config.conservative_continuation_overrides_disabled = absl::GetFlag(FLAGS_conservative_continuation_overrides_disabled);
      config.action_selection_mode =
          ParseDuneActionSelectionMode(absl::GetFlag(FLAGS_action_selection_mode));
      DuneSearchBudgetMode budget_mode =
          absl::GetFlag(FLAGS_policy_only)
              ? DuneSearchBudgetMode::kPolicyOnly
              : (absl::GetFlag(FLAGS_audit_training_policy_iteration)
                     ? DuneSearchBudgetMode::kTrainingPolicyIteration
                     : (absl::GetFlag(FLAGS_live_deadline)
                            ? DuneSearchBudgetMode::kLiveDeadline
                            : DuneSearchBudgetMode::kFixedSessionSimulations));
      setup.seat_evaluators[p] = search_evaluator;
      if (absl::GetFlag(FLAGS_use_session)) {
        setup.seat_sessions[p] = std::make_unique<DuneSearchSession>(config, search_evaluator, budget_mode);
      } else {
        // Reference protocol: fresh full search per decision, no session.
        // Path B is single-seat only (guarded in main()), so this is the
        // candidate's bot.
        setup.search_bot = std::make_unique<DunePUCTISMCTSBot>(config, search_evaluator);
      }
      setup.seat_provenance[p] = SeatControllerProvenance{
          /*controller=*/"search",
          /*checkpoint=*/absl::GetFlag(FLAGS_model_checkpoint),
          /*checkpoint_sha256=*/g_candidate_ckpt_sha256,
          /*max_simulations=*/absl::GetFlag(FLAGS_max_simulations),
          /*budget_mode=*/BudgetModeName(budget_mode),
          /*logit_cap=*/static_cast<double>(candidate_logit_cap)};
    } else if (controller_mode == SeatControllerMode::kMixed &&
               mixed_opponent == MixedOpponentController::kRandomLegal) {
      // Slice S1: candidate + 3x random-legal, stated explicitly rather than
      // inferred from --opponent_checkpoint=random.
      setup.bots[p] = std::make_unique<DuneRandomBot>(static_cast<int>(seat_seed));
      setup.seat_provenance[p] = SeatControllerProvenance{
          /*controller=*/"random_legal", /*checkpoint=*/"",
          /*checkpoint_sha256=*/"", /*max_simulations=*/-1,
          /*budget_mode=*/"none", /*logit_cap=*/-1.0};
    } else {
      // 'single' legacy behavior, and 'mixed' with raw_policy opponents: both
      // resolve to the --opponent_checkpoint-driven bot. opponent_model is
      // null exactly when --opponent_checkpoint=random.
      if (opp_coord != nullptr) {
        auto local_opp_eval = std::make_shared<open_spiel::BatchedNNEvaluator>(
            opp_coord, opponent_logit_cap);
        setup.bots[p] = std::make_unique<DuneGreedyBot>(
            local_opp_eval, static_cast<int>(seat_seed),
            absl::GetFlag(FLAGS_external_opponent_temperature));
        setup.seat_provenance[p] = SeatControllerProvenance{
            /*controller=*/"raw_policy",
            /*checkpoint=*/absl::GetFlag(FLAGS_opponent_checkpoint).empty()
                ? absl::GetFlag(FLAGS_model_checkpoint)
                : absl::GetFlag(FLAGS_opponent_checkpoint),
            /*checkpoint_sha256=*/g_opponent_ckpt_sha256,
            /*max_simulations=*/-1, /*budget_mode=*/"none",
            /*logit_cap=*/static_cast<double>(opponent_logit_cap)};
      } else if (opponent_model != nullptr) {
        auto local_opp_eval = std::make_shared<DuneNNEvaluator>(
            opponent_model, device, opponent_logit_cap);
        setup.bots[p] = std::make_unique<DuneGreedyBot>(
            local_opp_eval, static_cast<int>(seat_seed),
            absl::GetFlag(FLAGS_external_opponent_temperature));
        setup.seat_provenance[p] = SeatControllerProvenance{
            /*controller=*/"raw_policy",
            /*checkpoint=*/absl::GetFlag(FLAGS_opponent_checkpoint).empty()
                ? absl::GetFlag(FLAGS_model_checkpoint)
                : absl::GetFlag(FLAGS_opponent_checkpoint),
            /*checkpoint_sha256=*/g_opponent_ckpt_sha256,
            /*max_simulations=*/-1, /*budget_mode=*/"none",
            /*logit_cap=*/static_cast<double>(opponent_logit_cap)};
      } else {
        setup.bots[p] = std::make_unique<DuneRandomBot>(static_cast<int>(seat_seed));
        setup.seat_provenance[p] = SeatControllerProvenance{
            /*controller=*/"random_legal", /*checkpoint=*/"",
            /*checkpoint_sha256=*/"", /*max_simulations=*/-1,
            /*budget_mode=*/"none", /*logit_cap=*/-1.0};
      }
    }
  }
  return setup;
}

void InitSharedBatchedCoordinators(
    const std::shared_ptr<SharedDunePolicyValueNetImpl>& search_model,
    const std::shared_ptr<SharedDunePolicyValueNetImpl>& opponent_model,
    torch::Device device,
    std::shared_mutex& search_coord_mutex,
    std::shared_mutex& opp_coord_mutex,
    std::shared_ptr<open_spiel::BatchedEvaluator>* search_coord,
    std::shared_ptr<open_spiel::BatchedEvaluator>* opp_coord) {
  int batch_target = absl::GetFlag(FLAGS_batch_target_rows);
  if (batch_target <= 0) batch_target = 16;
  int timeout_ms = absl::GetFlag(FLAGS_batch_max_queue_wait_ms);
  if (timeout_ms <= 0) timeout_ms = 2;

  std::cout << "[Full Game] Initializing shared batched coordinators:\n"
            << "  target_batch_size: " << batch_target << "\n"
            << "  timeout_ms:        " << timeout_ms << " ms\n"
            << "  rollout_amp:       true (BF16)\n"
            << "  allow_tf32:        false\n";

  *search_coord = std::make_shared<open_spiel::BatchedEvaluator>(
      search_model, batch_target, timeout_ms, device, &search_coord_mutex,
      static_cast<float>(absl::GetFlag(FLAGS_candidate_logit_cap)),
      /*device_synchronize=*/true, /*high_priority_stream=*/false, /*emit_batch_membership=*/false,
      /*rollout_amp=*/true, /*allow_tf32=*/false);
  (*search_coord)->EnableBatcherTelemetry();

  if (opponent_model != nullptr) {
    *opp_coord = std::make_shared<open_spiel::BatchedEvaluator>(
        opponent_model, batch_target, timeout_ms, device, &opp_coord_mutex,
        static_cast<float>(absl::GetFlag(FLAGS_opponent_logit_cap)),
        /*device_synchronize=*/true, /*high_priority_stream=*/false, /*emit_batch_membership=*/false,
        /*rollout_amp=*/true, /*allow_tf32=*/false);
    (*opp_coord)->EnableBatcherTelemetry();
  }
}

}  // namespace open_spiel
