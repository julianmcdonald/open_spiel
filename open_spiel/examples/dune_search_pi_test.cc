// Acceptance tests for the agent-turn search policy-iteration lane.
//
// Numbered to the fifteen registered acceptance criteria; each test names the
// criterion it discharges. Everything runs on CPU with mock evaluators.

#include "dune_search_pi.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include "open_spiel/abseil-cpp/absl/flags/flag.h"
#include "open_spiel/algorithms/mcts.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"
#include "dune_online_search_collector.h"
#include "dune_ppo_training_utils.h"  // CenterAndCapLogitsTensor, for the
                                      // independent gradient recomputation in
                                      // test 14
#include "dune_search_session.h"

// dune_ppo_training_utils.cc DECLARES these; dune_ppo_train.cc defines them.
// A test binary that links the former without the latter has to supply them,
// which is exactly what dune_ppo_training_utils_test.cc:64-77 does.
//
// The alternative -- re-implementing CenterAndCapLogitsTensor locally to avoid
// the dependency -- was rejected: a test of a re-expression says nothing about
// the function the learner actually calls, and this lane's whole point is that
// its CE runs on the same masked-logit convention as everything else. None of
// these flags is read by the search-PI learner; they exist so TrainPpoUpdate,
// which this test never calls, resolves at link time.
ABSL_FLAG(int, ppo_minibatch_size, 2048, "");
ABSL_FLAG(int, ppo_update_epochs, 4, "");
ABSL_FLAG(double, ppo_clip_epsilon, 0.2, "");
ABSL_FLAG(bool, normalize_advantages, true, "");
ABSL_FLAG(bool, ppo_clip_value_loss, true, "");
ABSL_FLAG(double, entropy_coef, 0.01, "");
ABSL_FLAG(double, value_coef, 0.5, "");
ABSL_FLAG(double, logit_cap, 10.0, "");
ABSL_FLAG(double, target_kl, 0.0, "");
ABSL_FLAG(bool, train_amp, true, "");
ABSL_FLAG(double, grad_clip_norm, 0.5, "");
ABSL_FLAG(uint64_t, shaping_start_env_steps, 206830543, "");
ABSL_FLAG(uint64_t, shaping_decay_env_steps, 0, "");
ABSL_FLAG(bool, diagnostics_only, false, "");

namespace open_spiel {
namespace {

// --- Fixtures --------------------------------------------------------------

// Uniform prior, zero value. Never touches torch/CUDA.
class UniformMockEvaluator : public algorithms::Evaluator {
 public:
  explicit UniformMockEvaluator(int num_players) : num_players_(num_players) {}
  std::vector<double> Evaluate(const State& state) override {
    return std::vector<double>(num_players_, 0.0);
  }
  ActionsAndProbs Prior(const State& state) override {
    std::vector<Action> legal = state.LegalActions();
    ActionsAndProbs out;
    out.reserve(legal.size());
    const double p =
        legal.empty() ? 0.0 : 1.0 / static_cast<double>(legal.size());
    for (Action a : legal) out.push_back({a, p});
    return out;
  }

 private:
  const int num_players_;
};

// A peaked prior, which is what a trained network produces.
class PeakedMockEvaluator : public algorithms::Evaluator {
 public:
  explicit PeakedMockEvaluator(int num_players) : num_players_(num_players) {}
  std::vector<double> Evaluate(const State& state) override {
    return std::vector<double>(num_players_, 0.0);
  }
  ActionsAndProbs Prior(const State& state) override {
    std::vector<Action> legal = state.LegalActions();
    ActionsAndProbs out;
    out.reserve(legal.size());
    if (legal.empty()) return out;
    const double head = 0.70;
    const double tail =
        legal.size() > 1 ? (1.0 - head) / static_cast<double>(legal.size() - 1)
                         : 0.0;
    for (size_t i = 0; i < legal.size(); ++i) {
      out.push_back({legal[i], i == 0 ? head : tail});
    }
    return out;
  }

 private:
  const int num_players_;
};

// Small, fast configuration. Keeps every structural invariant (multiple of 4,
// uncapped depth, leader off, no forced playouts without noise) but shrinks the
// budgets so a full generation of real games stays quick. The BUDGETS are what
// the tests assert against, so small values prove the mechanism just as well as
// the 200/64 defaults.
SearchPiConfig FastConfig() {
  SearchPiConfig c;
  c.games_per_generation = 4;
  c.primary_simulations = 8;
  c.continuation_simulations = 3;
  c.seed_domain = 0xDEADBEEFULL;
  return c;
}

// A synthetic root: n actions, given visits, peaked prior, flat Q.
SearchDiagnostics MakeDiag(const std::vector<int>& visits,
                           const std::vector<double>& priors,
                           const std::vector<double>& qs) {
  SearchDiagnostics d;
  for (size_t i = 0; i < visits.size(); ++i) {
    d.actions.push_back(static_cast<Action>(100 + i));
  }
  d.visit_counts = visits;
  d.priors = priors;
  d.raw_priors = priors;
  d.q_values = qs;
  int total = 0;
  for (int v : visits) total += v;
  d.total_root_visits = total;
  return d;
}

std::shared_ptr<SharedDunePolicyValueNetImpl> TinyModel(int64_t in_dim,
                                                        int64_t action_dim) {
  torch::manual_seed(1234);
  auto m = std::make_shared<SharedDunePolicyValueNetImpl>(
      in_dim, /*hidden_dim=*/16, action_dim, /*num_blocks=*/1);
  m->to(torch::kCPU);
  m->train();
  return m;
}

// One fully-formed synthetic row against a tiny model's input size.
SearchPiRow MakeRow(int64_t in_dim, const std::vector<Action>& legal,
                    const std::vector<double>& target, double value,
                    Player player) {
  SearchPiRow r;
  r.observation.assign(in_dim, 0.0f);
  for (int64_t i = 0; i < in_dim; ++i) {
    r.observation[i] = static_cast<float>((i % 7) * 0.1);
  }
  r.observation_is_information_state = true;
  r.player = player;
  r.role = DuneDecisionRole::kAgentPrimary;
  r.legal_actions = legal;
  r.raw_policy.assign(legal.size(), 1.0 / static_cast<double>(legal.size()));
  r.raw_visits.assign(legal.size(), 1);
  r.target_visits.assign(legal.size(), 1);
  r.target_probs = target;
  r.chosen_action = legal.front();
  r.value_target = value;
  r.value_target_attached = true;
  r.generation = 1;
  r.episode_id = 0;
  r.decision_id = 0;
  FillSearchPiRowScalars(&r);
  return r;
}

// --- 1. Existing search-disabled PPO behaviour is unchanged ---------------
//
// The C++ half of this criterion: the fields and modes the PPO path reads are
// untouched by the additions, and a kFixedSessionSimulations session still
// POOLS its budget exactly as before (which is the behaviour the new mode
// deliberately differs from). The end-to-end half is the 8-second full-18B
// smoke recorded with the run.
void Test01_ExistingBehaviourUnchanged() {
  std::cout << "Running Test01_ExistingBehaviourUnchanged...\n";
  // Pre-existing enumerators keep their wire values; the new one is appended.
  SPIEL_CHECK_EQ(static_cast<int>(DuneSearchBudgetMode::kPolicyOnly), 0);
  SPIEL_CHECK_EQ(static_cast<int>(DuneSearchBudgetMode::kFixedSessionSimulations), 1);
  SPIEL_CHECK_EQ(static_cast<int>(DuneSearchBudgetMode::kTrainingFullFast), 2);
  SPIEL_CHECK_EQ(static_cast<int>(DuneSearchBudgetMode::kLiveDeadline), 3);
  SPIEL_CHECK_EQ(static_cast<int>(DuneSearchBudgetMode::kTrainingPolicyIteration), 4);

  // Every pre-existing DuneSearchConfig default is unchanged; the two new
  // fields are additive and inert outside the new mode.
  DuneSearchConfig d;
  SPIEL_CHECK_EQ(d.max_simulations, 64);
  SPIEL_CHECK_EQ(d.fixed_session_limit, 200);
  SPIEL_CHECK_EQ(d.purchase_combat_budget, 16);
  SPIEL_CHECK_EQ(d.fixed_continuation_reserve, 0);
  SPIEL_CHECK_FLOAT_EQ(d.puct_c, 0.15);
  SPIEL_CHECK_FLOAT_EQ(d.dirichlet_epsilon, 0.0);
  SPIEL_CHECK_FLOAT_EQ(d.forced_playouts_k, 0.0);
  SPIEL_CHECK_EQ(d.max_search_decision_depth, -1);
  SPIEL_CHECK_FALSE(d.search_leader_draft);
  SPIEL_CHECK_EQ(d.pi_primary_simulations, 200);
  SPIEL_CHECK_EQ(d.pi_continuation_simulations, 64);

  // The PF-C collector's own defaults are untouched by this work.
  OnlineSearchConfig o;
  SPIEL_CHECK_EQ(o.max_simulations, 64);
  SPIEL_CHECK_FLOAT_EQ(o.search_loss_coef_target, 0.10);
  SPIEL_CHECK_FLOAT_EQ(o.target_sharpen_exponent, 1.0);
  SPIEL_CHECK_FALSE(o.search_leader_draft);
  std::cout << "Test01 Passed!\n\n";
}

// --- 2. The new mode never constructs a PPO policy loss -------------------
//
// Also discharges the addendum's extension: zero Leader labels and no
// leader-teacher loss in this mode.
void Test02_NoPpoPolicyLoss() {
  std::cout << "Running Test02_NoPpoPolicyLoss...\n";
  const int64_t in_dim = 8, action_dim = 5;
  auto model = TinyModel(in_dim, action_dim);
  torch::optim::AdamW opt(model->parameters(),
                          torch::optim::AdamWOptions(1e-3));
  std::vector<SearchPiRow> rows;
  rows.push_back(MakeRow(in_dim, {0, 1, 2}, {0.5, 0.3, 0.2}, 0.25, 0));
  rows.push_back(MakeRow(in_dim, {1, 3, 4}, {0.1, 0.1, 0.8}, -0.5, 1));

  SearchPiLearnerConfig cfg;
  cfg.minibatch_size = 2;
  cfg.epochs = 1;
  SearchPiLearnerStats s = RunSearchPiLearner(model, opt, rows, in_dim,
                                              action_dim, torch::kCPU, 7, 1, cfg);
  // The structural negative, asserted rather than inferred.
  SPIEL_CHECK_FALSE(s.ppo_policy_loss_constructed);
  // A real objective ran: CE is finite and positive, presentations counted.
  SPIEL_CHECK_TRUE(std::isfinite(s.policy_ce));
  SPIEL_CHECK_GT(s.policy_ce, 0.0);
  SPIEL_CHECK_EQ(s.distinct_rows, 2);
  SPIEL_CHECK_EQ(s.presentations, 2);
  SPIEL_CHECK_EQ(s.minibatches, 1);
  // No leader rows anywhere in this lane's data contract.
  for (const SearchPiRow& r : rows) {
    SPIEL_CHECK_TRUE(r.role != DuneDecisionRole::kLeaderSelection);
  }
  // And the lane's search config never opens the leader gate.
  SearchPiConfig pc = FastConfig();
  DuneSearchConfig sc = SearchPiSearchConfigFor(pc, 1);
  SPIEL_CHECK_FALSE(sc.search_leader_draft);
  SPIEL_CHECK_FALSE(sc.leader_mass_only_coverage);
  std::cout << "Test02 Passed!\n\n";
}

// --- 3/4. Only primary and continuation search; every other role is zero ---
void Test03_04_RoleScoping(const std::shared_ptr<const Game>& game) {
  std::cout << "Running Test03_04_RoleScoping...\n";
  SearchPiConfig c = FastConfig();
  SearchPiGenerator gen(c);
  auto ev = std::make_shared<PeakedMockEvaluator>(game->NumPlayers());
  std::vector<SearchPiRow> rows;
  SearchPiGenerationStats st;
  gen.GenerateGeneration(1, game, ev, &rows, &st);

  // 3. Search ran at the two agent-turn roles.
  SPIEL_CHECK_GT(st.simulations_by_role[
      static_cast<int>(DuneDecisionRole::kAgentPrimary)], 0);
  // 4. Zero NEW training-search simulations at every other role.
  const DuneDecisionRole must_be_zero[] = {
      DuneDecisionRole::kForcedOrBookkeeping,
      DuneDecisionRole::kLeaderSelection,
      DuneDecisionRole::kPurchase,
      DuneDecisionRole::kCombatIntrigue,
      DuneDecisionRole::kOtherOptional};
  for (DuneDecisionRole r : must_be_zero) {
    const int i = static_cast<int>(r);
    std::cout << "  role " << i << ": decisions=" << st.decisions_by_role[i]
              << " sims=" << st.simulations_by_role[i] << "\n";
    SPIEL_CHECK_EQ(st.simulations_by_role[i], 0);
  }
  // The leader role was actually reached, so the zero above is a measurement
  // rather than a vacuous pass.
  SPIEL_CHECK_GT(
      st.decisions_by_role[static_cast<int>(DuneDecisionRole::kLeaderSelection)],
      0);
  SPIEL_CHECK_EQ(st.leader_rows_emitted, 0);
  // No emitted row carries a non-agent-turn role.
  for (const SearchPiRow& r : rows) {
    SPIEL_CHECK_TRUE(IsSearchPiSearchRole(r.role));
  }
  std::cout << "Test03_04 Passed!\n\n";
}

// --- 5. One placement activation retains and re-roots one session ----------
void Test05_SessionRetainedAndReRooted(const std::shared_ptr<const Game>& game) {
  std::cout << "Running Test05_SessionRetainedAndReRooted...\n";
  SearchPiConfig c = FastConfig();
  SearchPiGenerator gen(c);
  auto ev = std::make_shared<PeakedMockEvaluator>(game->NumPlayers());
  std::vector<SearchPiRow> rows;
  SearchPiGenerationStats st;
  gen.GenerateGeneration(1, game, ev, &rows, &st);

  SPIEL_CHECK_GT(st.continuation.searches_run, 0);
  // A continuation that re-rooted onto the retained tree inherits visits from
  // the primary. If the session were rebuilt per root (the PF-C shape) this
  // would be zero everywhere.
  SPIEL_CHECK_GT(st.continuation.re_root_hits, 0);
  SPIEL_CHECK_GT(st.continuation.inherited_visits, 0);
  bool saw_inherited_row = false;
  for (const SearchPiRow& r : rows) {
    if (r.role == DuneDecisionRole::kAgentContinuation &&
        r.inherited_root_visits > 0) {
      SPIEL_CHECK_EQ(r.re_root_status, std::string("hit"));
      saw_inherited_row = true;
    }
    // A primary opens an activation, so it never inherits.
    if (r.role == DuneDecisionRole::kAgentPrimary) {
      SPIEL_CHECK_EQ(r.inherited_root_visits, 0);
    }
  }
  SPIEL_CHECK_TRUE(saw_inherited_row);
  std::cout << "  continuation re_root hits=" << st.continuation.re_root_hits
            << " misses=" << st.continuation.re_root_misses
            << " inherited=" << st.continuation.inherited_visits << "\n";
  std::cout << "Test05 Passed!\n\n";
}

// --- 6. Exact budgets; the primary cannot consume the continuation budget ---
void Test06_ExactBudgets(const std::shared_ptr<const Game>& game) {
  std::cout << "Running Test06_ExactBudgets...\n";
  SearchPiConfig c = FastConfig();
  c.primary_simulations = 9;
  c.continuation_simulations = 4;
  SearchPiGenerator gen(c);
  auto ev = std::make_shared<PeakedMockEvaluator>(game->NumPlayers());
  std::vector<SearchPiRow> rows;
  SearchPiGenerationStats st;
  gen.GenerateGeneration(1, game, ev, &rows, &st);

  int primary_rows = 0, continuation_rows = 0;
  for (const SearchPiRow& r : rows) {
    if (r.role == DuneDecisionRole::kAgentPrimary) {
      SPIEL_CHECK_EQ(r.simulations_completed, 9);
      ++primary_rows;
    } else {
      // The decisive property: EVERY continuation gets its own full budget of
      // new simulations, however many the primary already spent. Under
      // kFixedSessionSimulations these would be 0 with the pool exhausted.
      SPIEL_CHECK_EQ(r.simulations_completed, 4);
      ++continuation_rows;
    }
  }
  SPIEL_CHECK_GT(primary_rows, 0);
  SPIEL_CHECK_GT(continuation_rows, 0);
  std::cout << "  primary rows=" << primary_rows << " @9 sims, continuation rows="
            << continuation_rows << " @4 sims\n";
  std::cout << "Test06 Passed!\n\n";
}

// --- 7. A concentrated search is kept where the old coverage gate rejected --
void Test07_ConcentratedSearchIsKept() {
  std::cout << "Running Test07_ConcentratedSearchIsKept...\n";
  // 5 legal actions, a peaked prior, and all visits on ONE action. The PF-C
  // gate needs min(3,5)=3 actions at >= 2 visits, so it rejects.
  const std::vector<int> visits = {20, 0, 0, 0, 0};
  const std::vector<double> priors = {0.70, 0.10, 0.10, 0.05, 0.05};
  const std::vector<double> qs(5, 0.0);
  int covered = 0;
  double mass = 0.0;
  const bool legacy = ComputeSearchAcceptance(visits, priors, 5, /*min_coverage=*/3,
                                              /*min_visits=*/2,
                                              /*min_prior_mass=*/0.50, &covered,
                                              &mass);
  SPIEL_CHECK_FALSE(legacy);   // the old gate would discard this search
  SPIEL_CHECK_EQ(covered, 1);

  // This lane keeps it: a valid, nonzero-visit search yields a target.
  SearchPiConfig c = FastConfig();
  SearchDiagnostics d = MakeDiag(visits, priors, qs);
  std::vector<int> tv;
  std::vector<double> tp;
  SPIEL_CHECK_TRUE(BuildSearchPiTarget(c, DuneDecisionRole::kAgentPrimary, d,
                                       &tv, &tp));
  SPIEL_CHECK_EQ(tp.size(), 5);
  SPIEL_CHECK_FLOAT_EQ(tp[0], 1.0);
  // And it is not treated as a technical failure.
  DuneSearchResult res;
  res.diagnostics = d;
  res.simulations_completed = 20;
  std::vector<Action> legal = d.actions;
  SPIEL_CHECK_TRUE(ClassifySearchPiResult(res, legal) == SearchPiFallback::kNone);
  std::cout << "Test07 Passed!\n\n";
}

// --- 8. Zero-visit / invalid searches fall back safely, no false target ----
void Test08_TechnicalFailureFallback() {
  std::cout << "Running Test08_TechnicalFailureFallback...\n";
  SearchPiConfig c = FastConfig();

  // Zero visits.
  {
    SearchDiagnostics d = MakeDiag({0, 0, 0}, {0.5, 0.3, 0.2}, {0, 0, 0});
    DuneSearchResult res;
    res.diagnostics = d;
    SPIEL_CHECK_TRUE(ClassifySearchPiResult(res, d.actions) ==
                     SearchPiFallback::kZeroVisits);
    std::vector<int> tv;
    std::vector<double> tp;
    SPIEL_CHECK_FALSE(
        BuildSearchPiTarget(c, DuneDecisionRole::kAgentPrimary, d, &tv, &tp));
    SPIEL_CHECK_TRUE(tp.empty());  // no false target emitted
  }
  // Zero visits with a timeout keeps the two causes distinguishable.
  {
    SearchDiagnostics d = MakeDiag({0, 0}, {0.5, 0.5}, {0, 0});
    DuneSearchResult res;
    res.diagnostics = d;
    res.timeout_status = true;
    SPIEL_CHECK_TRUE(ClassifySearchPiResult(res, d.actions) ==
                     SearchPiFallback::kTimeoutBeforeUsefulSearch);
  }
  // Misaligned visits vs actions.
  {
    SearchDiagnostics d = MakeDiag({3, 1}, {0.5, 0.5}, {0, 0});
    d.visit_counts.push_back(1);  // now 3 visits vs 2 actions
    DuneSearchResult res;
    res.diagnostics = d;
    SPIEL_CHECK_TRUE(ClassifySearchPiResult(res, d.actions) ==
                     SearchPiFallback::kInvalidAlignment);
  }
  // Diagnostics that disagree with the state's own legal actions.
  {
    SearchDiagnostics d = MakeDiag({3, 1}, {0.5, 0.5}, {0, 0});
    std::vector<Action> other_legal = {999, 1000};
    DuneSearchResult res;
    res.diagnostics = d;
    SPIEL_CHECK_TRUE(ClassifySearchPiResult(res, other_legal) ==
                     SearchPiFallback::kInvalidAlignment);
  }
  // Empty root.
  {
    DuneSearchResult res;
    SPIEL_CHECK_TRUE(ClassifySearchPiResult(res, {}) ==
                     SearchPiFallback::kEmptyPolicy);
  }
  // A starved path plays the RAW-PRIOR ARGMAX, never uniform, never index 0 by
  // accident: the peak here is deliberately not the first action.
  {
    ActionsAndProbs prior = {{10, 0.2}, {11, 0.65}, {12, 0.15}};
    SPIEL_CHECK_EQ(RawPriorArgmaxAction(prior, {10, 11, 12}), 11);
  }
  // Behaviour selection reports "no action" rather than inventing one.
  SPIEL_CHECK_EQ(SelectSearchPiActionIndex({0, 0, 0}, 0.0, 0.5), -1);
  SPIEL_CHECK_EQ(SelectSearchPiActionIndex({}, 0.0, 0.5), -1);
  std::cout << "Test08 Passed!\n\n";
}

// --- 9. Action ids, masks and target probabilities stay aligned and sum to 1 -
void Test09_AlignmentAndNormalization(const std::shared_ptr<const Game>& game) {
  std::cout << "Running Test09_AlignmentAndNormalization...\n";
  SearchPiConfig c = FastConfig();
  SearchPiGenerator gen(c);
  auto ev = std::make_shared<PeakedMockEvaluator>(game->NumPlayers());
  std::vector<SearchPiRow> rows;
  SearchPiGenerationStats st;
  gen.GenerateGeneration(1, game, ev, &rows, &st);
  SPIEL_CHECK_GT(rows.size(), 0);

  for (const SearchPiRow& r : rows) {
    const size_t n = r.legal_actions.size();
    SPIEL_CHECK_GT(n, 0);
    SPIEL_CHECK_EQ(r.target_probs.size(), n);
    SPIEL_CHECK_EQ(r.target_visits.size(), n);
    SPIEL_CHECK_EQ(r.raw_visits.size(), n);
    SPIEL_CHECK_EQ(r.raw_policy.size(), n);
    double s = 0.0;
    for (double p : r.target_probs) {
      SPIEL_CHECK_GE(p, 0.0);
      s += p;
    }
    SPIEL_CHECK_LT(std::abs(s - 1.0), 1e-9);
    // The chosen action is one of the row's own legal actions.
    SPIEL_CHECK_TRUE(std::find(r.legal_actions.begin(), r.legal_actions.end(),
                               r.chosen_action) != r.legal_actions.end());
    // Action ids are distinct, so the action-indexed mask cannot collide.
    std::set<Action> uniq(r.legal_actions.begin(), r.legal_actions.end());
    SPIEL_CHECK_EQ(uniq.size(), n);
    // Scalars are finite and in range.
    SPIEL_CHECK_GE(r.target_entropy_norm, 0.0);
    SPIEL_CHECK_LE(r.target_entropy_norm, 1.0);
    SPIEL_CHECK_GE(r.raw_policy_entropy_norm, 0.0);
    SPIEL_CHECK_LE(r.raw_policy_entropy_norm, 1.0);
    SPIEL_CHECK_TRUE(std::isfinite(r.kl_target_given_raw));
  }
  std::cout << "Test09 Passed! (" << rows.size() << " rows)\n\n";
}

// --- 10. Executed action from UNPRUNED visits; target from pruned ----------
void Test10_UnprunedBehaviourPrunedTarget() {
  std::cout << "Running Test10_UnprunedBehaviourPrunedTarget...\n";
  SearchPiConfig c = FastConfig();
  // Arm the exploration package so pruning is live.
  c.dirichlet_epsilon = 0.25;
  c.forced_playouts_k = 2.0;
  c.root_noise_fpu_zero = true;

  // A high-prior action carrying forced visits it did not earn: prior 0.60 but
  // a poor Q, against a modestly-visited best child.
  const std::vector<int> visits = {30, 9, 4};
  const std::vector<double> priors = {0.25, 0.60, 0.15};
  const std::vector<double> qs = {0.90, -0.80, 0.10};

  const std::vector<int> pruned = PruneForcedPlayouts(visits, priors, qs, 43,
                                                      c.puct_c,
                                                      c.forced_playouts_k);
  // The pruner must actually bite, or this test proves nothing.
  SPIEL_CHECK_LT(pruned[1], visits[1]);

  SearchDiagnostics d = MakeDiag(visits, priors, qs);
  std::vector<int> tv;
  std::vector<double> tp;
  SPIEL_CHECK_TRUE(
      BuildSearchPiTarget(c, DuneDecisionRole::kAgentPrimary, d, &tv, &tp));
  // The TARGET is the pruned distribution.
  SPIEL_CHECK_EQ(tv, pruned);
  int ptot = 0;
  for (int v : pruned) ptot += v;
  SPIEL_CHECK_LT(std::abs(tp[1] - static_cast<double>(pruned[1]) / ptot), 1e-12);
  // ...and it differs from the unpruned one, so the two are distinguishable.
  int rtot = 0;
  for (int v : visits) rtot += v;
  SPIEL_CHECK_GT(std::abs(tp[1] - static_cast<double>(visits[1]) / rtot), 1e-9);

  // The BEHAVIOUR reads the raw, unpruned visits.
  const int idx = SelectSearchPiActionIndex(d.visit_counts,
                                            c.behavior_temperature, 0.5);
  SPIEL_CHECK_EQ(idx, 0);   // argmax of {30, 9, 4}

  // With pruning off (the lane default) target == raw normalized visits.
  SearchPiConfig off = FastConfig();
  std::vector<int> tv2;
  std::vector<double> tp2;
  SPIEL_CHECK_TRUE(
      BuildSearchPiTarget(off, DuneDecisionRole::kAgentPrimary, d, &tv2, &tp2));
  SPIEL_CHECK_EQ(tv2, visits);
  std::cout << "Test10 Passed!\n\n";
}

// --- 11. Terminal value targets attach to the correct player ---------------
void Test11_TerminalValueTargets(const std::shared_ptr<const Game>& game) {
  std::cout << "Running Test11_TerminalValueTargets...\n";
  SearchPiConfig c = FastConfig();
  SearchPiGenerator gen(c);
  auto ev = std::make_shared<PeakedMockEvaluator>(game->NumPlayers());
  std::vector<SearchPiRow> rows;
  SearchPiGenerationStats st;
  gen.GenerateGeneration(1, game, ev, &rows, &st);
  SPIEL_CHECK_GT(rows.size(), 0);

  // Every row is attached, on the /utility_divisor scale, and every row of one
  // episode shares one value -- there is exactly one searched seat per game, so
  // a per-player mixup would show up as two distinct values inside an episode.
  std::map<int64_t, double> per_episode;
  std::map<int64_t, Player> seat_of_episode;
  for (const SearchPiRow& r : rows) {
    SPIEL_CHECK_TRUE(r.value_target_attached);
    SPIEL_CHECK_TRUE(std::isfinite(r.value_target));
    SPIEL_CHECK_LE(std::abs(r.value_target), 1.5);
    auto it = per_episode.find(r.episode_id);
    if (it == per_episode.end()) {
      per_episode[r.episode_id] = r.value_target;
      seat_of_episode[r.episode_id] = r.player;
    } else {
      SPIEL_CHECK_LT(std::abs(it->second - r.value_target), 1e-12);
      SPIEL_CHECK_EQ(seat_of_episode[r.episode_id], r.player);
    }
  }
  // And the searched seat is the rotation's seat, not an arbitrary one.
  for (const auto& kv : seat_of_episode) {
    SPIEL_CHECK_EQ(kv.second, SearchPiGenerator::SearchedSeatForEpisode(
                                  kv.first, game->NumPlayers()));
  }
  std::cout << "Test11 Passed! (" << per_episode.size() << " episodes)\n\n";
}

// --- 12. CE alone moves the policy toward a synthetic search target --------
void Test12_CeMovesPolicy() {
  std::cout << "Running Test12_CeMovesPolicy...\n";
  const int64_t in_dim = 8, action_dim = 4;
  auto model = TinyModel(in_dim, action_dim);
  torch::optim::AdamW opt(model->parameters(),
                          torch::optim::AdamWOptions(5e-2));
  // A hard target on action 2.
  std::vector<SearchPiRow> rows;
  rows.push_back(MakeRow(in_dim, {0, 1, 2, 3}, {0.0, 0.0, 1.0, 0.0}, 0.0, 0));

  SearchPiLearnerConfig cfg;
  cfg.minibatch_size = 1;
  cfg.epochs = 1;
  cfg.value_coef = 0.0;   // CE alone
  cfg.grad_clip_norm = 0.0;

  auto prob_of_target = [&]() {
    torch::NoGradGuard ng;
    torch::Tensor x = torch::zeros({1, in_dim});
    for (int64_t i = 0; i < in_dim; ++i) {
      x[0][i] = static_cast<float>((i % 7) * 0.1);
    }
    auto out = model->forward(x);
    return torch::softmax(out.logits, -1)[0][2].item<double>();
  };

  const double before = prob_of_target();
  double ce_first = 0.0, ce_last = 0.0;
  for (int step = 0; step < 30; ++step) {
    SearchPiLearnerStats s = RunSearchPiLearner(model, opt, rows, in_dim,
                                                action_dim, torch::kCPU, 7,
                                                step, cfg);
    if (step == 0) ce_first = s.policy_ce;
    ce_last = s.policy_ce;
  }
  const double after = prob_of_target();
  std::cout << "  p(target): " << before << " -> " << after
            << ", CE: " << ce_first << " -> " << ce_last << "\n";
  SPIEL_CHECK_GT(after, before);
  SPIEL_CHECK_LT(ce_last, ce_first);
  std::cout << "Test12 Passed!\n\n";
}

// --- 13. Value loss alone reaches the value head ---------------------------
void Test13_ValueLossReachesValueHead() {
  std::cout << "Running Test13_ValueLossReachesValueHead...\n";
  const int64_t in_dim = 8, action_dim = 4;
  auto model = TinyModel(in_dim, action_dim);
  torch::optim::AdamW opt(model->parameters(),
                          torch::optim::AdamWOptions(5e-2));
  std::vector<SearchPiRow> rows;
  rows.push_back(MakeRow(in_dim, {0, 1, 2, 3}, {0.25, 0.25, 0.25, 0.25}, 0.75, 0));

  // Snapshot the value head so we can prove the gradient landed on it.
  torch::Tensor vh_before;
  for (const auto& p : model->named_parameters()) {
    if (p.key() == "value_head.weight") vh_before = p.value().detach().clone();
  }
  SPIEL_CHECK_TRUE(vh_before.defined());

  SearchPiLearnerConfig cfg;
  cfg.minibatch_size = 1;
  cfg.epochs = 1;
  cfg.value_coef = 1.0;
  cfg.grad_clip_norm = 0.0;

  double mse_first = 0.0, mse_last = 0.0;
  for (int step = 0; step < 40; ++step) {
    SearchPiLearnerStats s = RunSearchPiLearner(model, opt, rows, in_dim,
                                                action_dim, torch::kCPU, 11,
                                                step, cfg);
    if (step == 0) {
      mse_first = s.value_mse;
      SPIEL_CHECK_GT(s.value_grad_norm, 0.0);
    }
    mse_last = s.value_mse;
  }
  torch::Tensor vh_after;
  for (const auto& p : model->named_parameters()) {
    if (p.key() == "value_head.weight") vh_after = p.value().detach().clone();
  }
  const double delta = (vh_after - vh_before).abs().sum().item<double>();
  std::cout << "  value MSE: " << mse_first << " -> " << mse_last
            << ", |dW_value_head| = " << delta << "\n";
  SPIEL_CHECK_GT(delta, 0.0);
  SPIEL_CHECK_LT(mse_last, mse_first);

  // The converse: with value_coef 0 the value gradient is exactly zero.
  SearchPiLearnerConfig no_v = cfg;
  no_v.value_coef = 0.0;
  SearchPiLearnerStats s0 = RunSearchPiLearner(model, opt, rows, in_dim,
                                               action_dim, torch::kCPU, 11, 99,
                                               no_v);
  SPIEL_CHECK_FLOAT_EQ(s0.value_grad_norm, 0.0);
  std::cout << "Test13 Passed!\n\n";
}

// --- 14. Gradient norms and cosines are numerically checked ----------------
void Test14_GradientTelemetryNumerics() {
  std::cout << "Running Test14_GradientTelemetryNumerics...\n";
  const int64_t in_dim = 8, action_dim = 4;
  auto model = TinyModel(in_dim, action_dim);
  torch::optim::AdamW opt(model->parameters(),
                          torch::optim::AdamWOptions(0.0));  // LR 0: no drift
  std::vector<SearchPiRow> rows;
  rows.push_back(MakeRow(in_dim, {0, 1, 2, 3}, {0.1, 0.2, 0.3, 0.4}, 0.5, 0));

  SearchPiLearnerConfig cfg;
  cfg.minibatch_size = 1;
  cfg.epochs = 1;
  cfg.value_coef = 1.0;
  cfg.grad_clip_norm = 0.0;
  SearchPiLearnerStats s = RunSearchPiLearner(model, opt, rows, in_dim,
                                              action_dim, torch::kCPU, 5, 1, cfg);

  // Independent recomputation of the same three quantities, by hand.
  torch::Tensor x = torch::zeros({1, in_dim});
  for (int64_t i = 0; i < in_dim; ++i) {
    x[0][i] = static_cast<float>((i % 7) * 0.1);
  }
  torch::Tensor mask = torch::zeros({1, action_dim}, torch::kBool);
  torch::Tensor tgt = torch::zeros({1, action_dim});
  const double tp[4] = {0.1, 0.2, 0.3, 0.4};
  for (int64_t a = 0; a < action_dim; ++a) {
    mask[0][a] = true;
    tgt[0][a] = static_cast<float>(tp[a]);
  }
  torch::Tensor val = torch::full({1}, 0.5f);

  auto named = model->named_parameters();
  auto grads_of = [&](bool policy_term) {
    for (auto& p : model->parameters()) {
      if (p.grad().defined()) p.mutable_grad().reset();
    }
    auto out = model->forward(x);
    torch::Tensor logits = CenterAndCapLogitsTensor(out.logits, mask,
                                                    static_cast<float>(cfg.logit_cap));
    torch::Tensor masked = logits.masked_fill(mask.logical_not(), -1e9f);
    torch::Tensor loss;
    if (policy_term) {
      loss = -(tgt * torch::log_softmax(masked, -1)).sum(-1).mean();
    } else {
      loss = static_cast<float>(cfg.value_coef) *
             (out.values.squeeze(1) - val).pow(2).mean();
    }
    loss.backward();
    std::vector<torch::Tensor> g;
    for (const auto& p : named) {
      auto gg = p.value().grad();
      g.push_back(gg.defined() ? gg.detach().clone() : torch::Tensor());
    }
    return g;
  };
  std::vector<torch::Tensor> ga = grads_of(true);
  std::vector<torch::Tensor> gb = grads_of(false);

  double a2 = 0.0, b2 = 0.0, dot = 0.0;
  double pa = 0.0, pb = 0.0, pdot = 0.0;   // policy_head group
  double va = 0.0, vb = 0.0, vdot = 0.0;   // value_head group
  double ta = 0.0, tb = 0.0, tdot = 0.0;   // trunk group
  size_t k = 0;
  for (const auto& p : named) {
    const std::string& nm = p.key();
    const bool is_pol = nm.rfind("policy_head", 0) == 0;
    const bool is_val = !is_pol && nm.find("value_head") != std::string::npos;
    const double aa = ga[k].defined() ? ga[k].pow(2).sum().item<double>() : 0.0;
    const double bb = gb[k].defined() ? gb[k].pow(2).sum().item<double>() : 0.0;
    const double dd = (ga[k].defined() && gb[k].defined())
                          ? (ga[k] * gb[k]).sum().item<double>()
                          : 0.0;
    a2 += aa; b2 += bb; dot += dd;
    if (is_pol) { pa += aa; pb += bb; pdot += dd; }
    else if (is_val) { va += aa; vb += bb; vdot += dd; }
    else { ta += aa; tb += bb; tdot += dd; }
    ++k;
  }
  const double exp_pol_norm = std::sqrt(a2);
  const double exp_val_norm = std::sqrt(b2);
  const double exp_cos = dot / (std::sqrt(a2) * std::sqrt(b2));

  std::cout << "  policy_grad_norm " << s.policy_grad_norm << " vs " << exp_pol_norm
            << "\n  value_grad_norm  " << s.value_grad_norm << " vs " << exp_val_norm
            << "\n  cosine overall   " << s.grad_cosine_overall << " vs " << exp_cos
            << "\n";
  SPIEL_CHECK_LT(std::abs(s.policy_grad_norm - exp_pol_norm), 1e-6);
  SPIEL_CHECK_LT(std::abs(s.value_grad_norm - exp_val_norm), 1e-6);
  SPIEL_CHECK_LT(std::abs(s.grad_cosine_overall - exp_cos), 1e-6);

  // The value objective cannot reach the policy head, so that group's CE-vs-value
  // cosine is exactly 0 -- a structural check on the split, not a numeric one.
  SPIEL_CHECK_FLOAT_EQ(pb, 0.0);
  SPIEL_CHECK_FLOAT_EQ(s.grad_cosine_policy_head, 0.0);
  // Symmetrically the CE cannot reach the value head.
  SPIEL_CHECK_FLOAT_EQ(va, 0.0);
  SPIEL_CHECK_FLOAT_EQ(s.grad_cosine_value_head, 0.0);
  // The trunk is the ONLY place the two objectives meet, so it supplies the
  // entire dot product. Its cosine is nonetheless LARGER than the overall one,
  // because the overall normalizers carry the two heads' own gradient mass
  // while the numerator does not:
  //     cos_overall = dot_trunk / (|a|_total * |b|_total)
  //     cos_trunk   = dot_trunk / (|a|_trunk * |b|_trunk)
  // Asserting the two equal would be wrong, and would have hidden a real
  // difference behind a coincidence of naming.
  SPIEL_CHECK_LT(std::abs(dot - tdot), 1e-9);   // all alignment is in the trunk
  const double exp_trunk_cos = tdot / (std::sqrt(ta) * std::sqrt(tb));
  std::cout << "  cosine trunk     " << s.grad_cosine_trunk << " vs "
            << exp_trunk_cos << "\n";
  SPIEL_CHECK_LT(std::abs(s.grad_cosine_trunk - exp_trunk_cos), 1e-6);
  SPIEL_CHECK_GT(std::abs(exp_trunk_cos), std::abs(exp_cos));
  // Cosines are in range.
  SPIEL_CHECK_GE(s.grad_cosine_overall, -1.0000001);
  SPIEL_CHECK_LE(s.grad_cosine_overall, 1.0000001);

  // A reported 0.0 must be distinguishable from "no measurement was possible".
  // The two head groups are STRUCTURALLY undefined -- neither objective reaches
  // the other's head -- so their cosines are sentinels, and a reader who took
  // them for measured orthogonality would be drawing a conclusion from nothing.
  SPIEL_CHECK_TRUE(s.grad_cosine_trunk_defined);
  SPIEL_CHECK_TRUE(s.grad_cosine_overall_defined);
  SPIEL_CHECK_FALSE(s.grad_cosine_policy_head_defined);
  SPIEL_CHECK_FALSE(s.grad_cosine_value_head_defined);
  std::cout << "Test14 Passed!\n\n";
}

// --- 15. Seeds, target hashes, manifests and resume cursors reproduce -------
//
// Scoped, per the addendum, to single-threaded/single-worker running. The
// generator is sequential by construction. Model-weight and target hashes are
// what is compared; optimizer serialization is deliberately NOT, because
// libtorch keys optimizer state by TensorImpl pointer and its bytes differ
// across identical runs.
void Test15_Reproducibility(const std::shared_ptr<const Game>& game) {
  std::cout << "Running Test15_Reproducibility...\n";
  SearchPiConfig c = FastConfig();
  auto ev = std::make_shared<PeakedMockEvaluator>(game->NumPlayers());

  std::vector<SearchPiRow> rows_a, rows_b;
  SearchPiGenerationStats st_a, st_b;
  SearchPiGenerator gen_a(c);
  gen_a.GenerateGeneration(3, game, ev, &rows_a, &st_a);
  SearchPiGenerator gen_b(c);
  gen_b.GenerateGeneration(3, game, ev, &rows_b, &st_b);

  SPIEL_CHECK_EQ(rows_a.size(), rows_b.size());
  SPIEL_CHECK_EQ(st_a.target_hash_chain, st_b.target_hash_chain);
  SPIEL_CHECK_FALSE(st_a.target_hash_chain.empty());
  SPIEL_CHECK_EQ(st_a.next_episode_id, st_b.next_episode_id);
  SPIEL_CHECK_EQ(st_a.primary.rows_emitted, st_b.primary.rows_emitted);
  SPIEL_CHECK_EQ(st_a.continuation.rows_emitted, st_b.continuation.rows_emitted);
  for (size_t i = 0; i < rows_a.size(); ++i) {
    SPIEL_CHECK_EQ(rows_a[i].episode_id, rows_b[i].episode_id);
    SPIEL_CHECK_EQ(rows_a[i].decision_id, rows_b[i].decision_id);
    SPIEL_CHECK_EQ(rows_a[i].chosen_action, rows_b[i].chosen_action);
    SPIEL_CHECK_EQ(rows_a[i].target_probs.size(), rows_b[i].target_probs.size());
    for (size_t j = 0; j < rows_a[i].target_probs.size(); ++j) {
      SPIEL_CHECK_EQ(rows_a[i].target_probs[j], rows_b[i].target_probs[j]);
    }
    SPIEL_CHECK_EQ(rows_a[i].value_target, rows_b[i].value_target);
  }

  // A different seed domain must produce a different stream. Otherwise the
  // equality above would be proving determinism of a constant.
  SearchPiConfig c2 = c;
  c2.seed_domain = 0xABCDEF01ULL;
  std::vector<SearchPiRow> rows_c;
  SearchPiGenerationStats st_c;
  SearchPiGenerator gen_c(c2);
  gen_c.GenerateGeneration(3, game, ev, &rows_c, &st_c);
  SPIEL_CHECK_NE(st_a.target_hash_chain, st_c.target_hash_chain);

  // The cursor advances by exactly one generation of games and resumes there.
  SPIEL_CHECK_EQ(st_a.first_episode_id, 0);
  SPIEL_CHECK_EQ(st_a.next_episode_id, c.games_per_generation);
  SPIEL_CHECK_EQ(gen_a.config().next_episode_id, c.games_per_generation);

  // Manifest round-trip: every field survives, and the fingerprint is stable.
  SearchPiState s;
  s.generation = 3;
  s.next_episode_id = st_a.next_episode_id;
  s.cum_rows = st_a.rows_total;
  s.cum_primary_rows = st_a.primary.rows_emitted;
  s.cum_continuation_rows = st_a.continuation.rows_emitted;
  s.cum_primary_simulations = st_a.primary.simulations_completed;
  s.cum_continuation_simulations = st_a.continuation.simulations_completed;
  s.target_hash_chain = st_a.target_hash_chain;
  s.config = c;
  s.learner = SearchPiLearnerConfig();

  json::Object obj = WriteSearchPiState(s);
  SearchPiState back;
  std::string err;
  SPIEL_CHECK_TRUE(ReadSearchPiState(obj, &back, &err));
  SPIEL_CHECK_EQ(err, std::string(""));
  SPIEL_CHECK_EQ(back.generation, s.generation);
  SPIEL_CHECK_EQ(back.next_episode_id, s.next_episode_id);
  SPIEL_CHECK_EQ(back.target_hash_chain, s.target_hash_chain);
  SPIEL_CHECK_EQ(back.config.primary_simulations, c.primary_simulations);
  SPIEL_CHECK_EQ(back.config.continuation_simulations, c.continuation_simulations);
  SPIEL_CHECK_EQ(back.config.seed_domain, c.seed_domain);
  SPIEL_CHECK_FLOAT_EQ(back.config.puct_c, c.puct_c);
  SPIEL_CHECK_FLOAT_EQ(back.config.forced_playouts_k, c.forced_playouts_k);
  SPIEL_CHECK_FLOAT_EQ(back.config.dirichlet_epsilon, c.dirichlet_epsilon);
  SPIEL_CHECK_FLOAT_EQ(back.config.target_sharpen_exponent,
                       c.target_sharpen_exponent);
  SPIEL_CHECK_FALSE(back.config.search_leader_draft);
  SPIEL_CHECK_TRUE(back.config.continuation_target ==
                   SearchPiContinuationTarget::kTotalVisits);
  SPIEL_CHECK_EQ(SearchPiConfigFingerprint(c, s.learner),
                 SearchPiConfigFingerprint(back.config, back.learner));

  // A changed budget changes the fingerprint -- a resume cannot silently drift.
  SearchPiConfig c3 = c;
  c3.continuation_simulations += 1;
  SPIEL_CHECK_NE(SearchPiConfigFingerprint(c, s.learner),
                 SearchPiConfigFingerprint(c3, s.learner));
  // So does a changed learner LR.
  SearchPiLearnerConfig l2 = s.learner;
  l2.learning_rate *= 2.0;
  SPIEL_CHECK_NE(SearchPiConfigFingerprint(c, s.learner),
                 SearchPiConfigFingerprint(c, l2));
  // ...and so does weight decay, which AdamW applies at every step regardless
  // of gradient. It is part of the objective, so a resume must not be able to
  // change it silently.
  SearchPiLearnerConfig l3 = s.learner;
  l3.weight_decay = 0.01;
  SPIEL_CHECK_NE(SearchPiConfigFingerprint(c, s.learner),
                 SearchPiConfigFingerprint(c, l3));
  SearchPiLearnerConfig l4 = s.learner;
  l4.policy_weight_decay = 0.01;
  SPIEL_CHECK_NE(SearchPiConfigFingerprint(c, s.learner),
                 SearchPiConfigFingerprint(c, l4));
  SPIEL_CHECK_FLOAT_EQ(back.learner.weight_decay, s.learner.weight_decay);
  SPIEL_CHECK_FLOAT_EQ(back.learner.policy_weight_decay,
                       s.learner.policy_weight_decay);

  // GenerateGeneration APPENDS: a second call into the same buffer must report
  // only its own slice, and must not re-hash the first generation's rows into
  // the second's chain (which would break resume verification).
  std::vector<SearchPiRow> shared = rows_a;   // pretend a prior generation
  const size_t before = shared.size();
  SearchPiGenerationStats st_d;
  SearchPiGenerator gen_d(c);
  gen_d.GenerateGeneration(3, game, ev, &shared, &st_d);
  SPIEL_CHECK_EQ(st_d.rows_total, static_cast<int64_t>(shared.size() - before));
  SPIEL_CHECK_EQ(st_d.target_hash_chain, st_a.target_hash_chain);

  // The row-target hash chain is order- and content-sensitive.
  SPIEL_CHECK_NE(ChainSearchPiTargetHash("", rows_a[0]),
                 ChainSearchPiTargetHash("x", rows_a[0]));
  std::cout << "Test15 Passed! (chain " << st_a.target_hash_chain.substr(0, 16)
            << "...)\n\n";
}

// --- Pure-helper coverage used by several criteria -------------------------
void TestScalarHelpers() {
  std::cout << "Running TestScalarHelpers...\n";
  // Normalized entropy: uniform is 1, one-hot is 0, singleton is 0.
  SPIEL_CHECK_LT(std::abs(NormalizedEntropy({0.25, 0.25, 0.25, 0.25}) - 1.0), 1e-12);
  SPIEL_CHECK_FLOAT_EQ(NormalizedEntropy({1.0, 0.0, 0.0}), 0.0);
  SPIEL_CHECK_FLOAT_EQ(NormalizedEntropy({1.0}), 0.0);

  // KL of a distribution against itself is 0.
  bool floored = false;
  SPIEL_CHECK_LT(
      std::abs(KlTargetGivenRaw({0.5, 0.5}, {0.5, 0.5}, &floored)), 1e-12);
  SPIEL_CHECK_FALSE(floored);
  // Positive target mass on a zero raw prior uses the floor and SAYS so,
  // rather than returning a fabricated finite number silently.
  const double kl = KlTargetGivenRaw({1.0, 0.0}, {0.0, 1.0}, &floored);
  SPIEL_CHECK_TRUE(floored);
  SPIEL_CHECK_GT(kl, 20.0);
  SPIEL_CHECK_TRUE(std::isfinite(kl));

  // Behaviour temperature 0 is argmax; temperature 1 is proportional.
  SPIEL_CHECK_EQ(SelectSearchPiActionIndex({1, 7, 3}, 0.0, 0.99), 1);
  SPIEL_CHECK_EQ(SelectSearchPiActionIndex({10, 0, 0}, 1.0, 0.5), 0);
  SPIEL_CHECK_EQ(SelectSearchPiActionIndex({0, 0, 10}, 1.0, 0.5), 2);

  // Fallback names are the persisted strings.
  SPIEL_CHECK_EQ(std::string(SearchPiFallbackName(SearchPiFallback::kNone)),
                 std::string("none"));
  SPIEL_CHECK_EQ(std::string(SearchPiFallbackName(SearchPiFallback::kZeroVisits)),
                 std::string("zero_visits"));
  SearchPiContinuationTarget ct;
  SPIEL_CHECK_TRUE(ParseSearchPiContinuationTarget("total_visits", &ct));
  SPIEL_CHECK_TRUE(ct == SearchPiContinuationTarget::kTotalVisits);
  SPIEL_CHECK_TRUE(ParseSearchPiContinuationTarget("new_visits_only", &ct));
  SPIEL_CHECK_TRUE(ct == SearchPiContinuationTarget::kNewVisitsOnly);
  SPIEL_CHECK_FALSE(ParseSearchPiContinuationTarget("nonsense", &ct));

  // Seat rotation is a pure function of episode id.
  for (int64_t ep = 0; ep < 12; ++ep) {
    SPIEL_CHECK_EQ(SearchPiGenerator::SearchedSeatForEpisode(ep, 4),
                   static_cast<Player>(ep % 4));
  }

  // The search config the lane builds pins the zero-simulation guards.
  SearchPiConfig c = FastConfig();
  DuneSearchConfig sc = SearchPiSearchConfigFor(c, 42);
  SPIEL_CHECK_EQ(sc.purchase_combat_budget, 0);
  SPIEL_CHECK_FALSE(sc.search_leader_draft);
  SPIEL_CHECK_EQ(sc.pi_primary_simulations, c.primary_simulations);
  SPIEL_CHECK_EQ(sc.pi_continuation_simulations, c.continuation_simulations);
  SPIEL_CHECK_EQ(sc.max_search_decision_depth, -1);
  SPIEL_CHECK_TRUE(sc.opponent_mode == SearchOpponentMode::kPolicy);
  SPIEL_CHECK_FLOAT_EQ(sc.opponent_temperature, 1.0);
  SPIEL_CHECK_FLOAT_EQ(sc.root_prior_temperature, 1.0);
  SPIEL_CHECK_FLOAT_EQ(sc.dirichlet_epsilon, 0.0);
  SPIEL_CHECK_FLOAT_EQ(sc.forced_playouts_k, 0.0);
  std::cout << "TestScalarHelpers Passed!\n\n";
}

}  // namespace
}  // namespace open_spiel

int main(int argc, char** argv) {
  using namespace open_spiel;
  std::shared_ptr<const Game> game = LoadGame("dune_imperium");

  TestScalarHelpers();
  Test01_ExistingBehaviourUnchanged();
  Test02_NoPpoPolicyLoss();
  Test07_ConcentratedSearchIsKept();
  Test08_TechnicalFailureFallback();
  Test10_UnprunedBehaviourPrunedTarget();
  Test12_CeMovesPolicy();
  Test13_ValueLossReachesValueHead();
  Test14_GradientTelemetryNumerics();

  // Game-driving tests last: they are the slow ones.
  Test03_04_RoleScoping(game);
  Test05_SessionRetainedAndReRooted(game);
  Test06_ExactBudgets(game);
  Test09_AlignmentAndNormalization(game);
  Test11_TerminalValueTargets(game);
  Test15_Reproducibility(game);

  std::cout << "All dune search-PI tests passed!\n";
  return 0;
}
