// Tests for the offline / legacy training utilities hardened by WO-19.
//
// Covers the four review findings this work order bundles:
//
//   PPO finding 4  | dune_search_teacher WriteLabel paired ppo_prior[i] with
//                  | teacher_prior[i] by index with no size or action-id check.
//   PPO finding 5  | RolloutSwordmasterRace sized its observation buffer to
//                  | InformationStateTensorSize but filled it via
//                  | ObservationTensor, overrunning the engine's offset check.
//   PPO finding 8  | dune_create_corpus's opportunity filter hardcoded
//                  | `solari >= 8`, excluding affordable Leto seats.
//   PPO finding 10 | dune_eval_1000 seeded from std::random_device, per thread,
//                  | so no run was reproducible.
//
// 12 tests:
//   1. Aligned priors            | Reorder is identity when orders already agree
//   2. Reordered teacher actions | PPO probabilities follow their own action id
//   3. Size mismatch             | Rejected, not truncated
//   4. Disjoint action sets      | Rejected, not silently paired
//   5. PriorsShareActionOrder    | Detects reorder that AlignPriorToActions fixes
//   6. NetworkObservation        | Matches InformationStateTensor exactly
//   7. Owner policy rollout      | Completes; evaluator sees model-sized obs
//   8. Opponent policy rollout   | Completes; evaluator sees model-sized obs
//   9. AgentSpaceSolariCost      | Leto pays 7 for Swordmaster, others 8
//  10. Leto affordability        | 7-solari Leto seat is affordable; old >= 8 was not
//  11. Legacy eval seeds         | Same seed reproduces; streams/games/seeds differ
//  12. Legacy eval seed domain   | Disjoint from dune_population_eval's domains

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <random>
#include <set>
#include <string>
#include <vector>

#include "open_spiel/spiel.h"
#include "open_spiel/games/dune_imperium/dune_imperium.h"
#include "open_spiel/games/dune_imperium/dune_imperium_board.h"
#include "open_spiel/games/dune_imperium/dune_imperium_common.h"
#include "open_spiel/games/dune_imperium/dune_imperium_test_utils.h"

#include "dune_network.h"
#include "dune_ppo_training_utils.h"
#include "dune_seed_utils.h"
#include "dune_warmstart_helpers.h"

using namespace open_spiel;
using namespace open_spiel::dune_imperium;

static int test_count = 0;
static int pass_count = 0;

#define TEST_BEGIN(name)                                              \
  do {                                                                \
    ++test_count;                                                     \
    const char* test_name_ = (name);                                  \
    std::cout << "Test " << test_count << ": " << test_name_ << "... ";

#define TEST_END()                                                    \
    ++pass_count;                                                     \
    std::cout << "PASSED\n";                                          \
  } while (0)

#define UTILS_CHECK(cond)                                             \
  do {                                                                \
    if (!(cond)) {                                                    \
      std::cerr << "FAILED\n  Assertion failed: " #cond               \
                << "\n  at " << __FILE__ << ":" << __LINE__ << "\n";  \
      std::abort();                                                   \
    }                                                                 \
  } while (0)

#define CHECK_EQ(a, b)                                                \
  do {                                                                \
    auto a_ = (a); auto b_ = (b);                                     \
    if (a_ != b_) {                                                   \
      std::cerr << "FAILED\n  Expected " #a " == " #b                 \
                << "\n  Got " << a_ << " vs " << b_                   \
                << "\n  at " << __FILE__ << ":" << __LINE__ << "\n";  \
      std::abort();                                                   \
    }                                                                 \
  } while (0)

namespace {

// ---------------------------------------------------------------------------
// Stub evaluator: records every observation length it is handed and returns a
// flat policy. Standing in for BatchedEvaluator lets the policy-guided rollout
// branches run without loading real weights, and lets the test assert the
// tensor the rollout builds is the one a real network would accept.
// ---------------------------------------------------------------------------
class RecordingEvaluator : public IGameEvaluator {
 public:
  explicit RecordingEvaluator(int64_t action_dim) : action_dim_(action_dim) {}

  EvalResult Evaluate(const std::vector<float>& obs) override {
    observed_sizes_.push_back(obs.size());
    // Mirror BatchedEvaluator's own contract check: the real evaluator rejects
    // any length that is neither the model input dim nor the 5580/5584 pair, so
    // a rollout that builds the wrong tensor must fail here too.
    CheckEvalObsSize(obs.size(), model_input_dim_);
    EvalResult result;
    result.logits.assign(action_dim_, 0.0f);
    result.value = 0.0f;
    return result;
  }

  EvaluatorStats GetStats() const override { return EvaluatorStats(); }

  void set_model_input_dim(int64_t dim) { model_input_dim_ = dim; }
  const std::vector<size_t>& observed_sizes() const { return observed_sizes_; }

 private:
  int64_t action_dim_;
  int64_t model_input_dim_ = 0;
  std::vector<size_t> observed_sizes_;
};

// Advances a fresh game to a P0 agent turn, the state shape the swordmaster
// planner rollout and the corpus opportunity filter both operate on.
std::unique_ptr<State> AgentTurnState(const Game& game, Player player) {
  std::unique_ptr<State> state = game.NewInitialState();
  int guard = 0;
  while (state->IsChanceNode() || state->CurrentPlayer() != player ||
         state->ToString().find("phase=agent_turns") == std::string::npos) {
    if (++guard > 5000) {
      std::cerr << "FAILED\n  Could not reach an agent turn for player "
                << player << "\n";
      std::abort();
    }
    if (state->IsChanceNode()) {
      state->ApplyAction(state->ChanceOutcomes()[0].first);
    } else {
      state->ApplyAction(DefaultProgressionAction(state->LegalActions()));
    }
  }
  return state;
}

}  // namespace

int main(int argc, char** argv) {
  std::cout << "Running WO-19 offline/legacy utility tests...\n\n";

  std::shared_ptr<const Game> game = LoadGame("dune_imperium");
  const int64_t istate_size = game->InformationStateTensorSize();
  const int64_t obs_tensor_size = game->ObservationTensorSize();

  // =========================================================================
  // PPO finding 4 — teacher labels relied on index alignment
  // =========================================================================

  // -----------------------------------------------------------------------
  // Test 1: orders already agree
  // -----------------------------------------------------------------------
  TEST_BEGIN("Aligned priors reorder to themselves") {
    ActionsAndProbs ppo = {{10, 0.5}, {20, 0.3}, {30, 0.2}};
    std::vector<Action> actions = {10, 20, 30};
    ActionsAndProbs aligned;
    UTILS_CHECK(AlignPriorToActions(ppo, actions, &aligned));
    CHECK_EQ(aligned.size(), ppo.size());
    for (size_t i = 0; i < ppo.size(); ++i) {
      CHECK_EQ(aligned[i].first, ppo[i].first);
      CHECK_EQ(aligned[i].second, ppo[i].second);
    }
    // This is the production case today, so the fix must be a byte-for-byte
    // no-op: label files generated before and after WO-19 are identical.
    UTILS_CHECK(PriorsShareActionOrder(aligned, ppo));
  } TEST_END();

  // -----------------------------------------------------------------------
  // Test 2: the actual defect — a reordered search root
  // -----------------------------------------------------------------------
  TEST_BEGIN("Reordered search actions keep each PPO probability on its own id") {
    ActionsAndProbs ppo = {{10, 0.5}, {20, 0.3}, {30, 0.2}};
    // Search returns the same actions in a different order.
    std::vector<Action> actions = {30, 10, 20};
    ActionsAndProbs aligned;
    UTILS_CHECK(AlignPriorToActions(ppo, actions, &aligned));
    CHECK_EQ(aligned.size(), size_t{3});
    CHECK_EQ(aligned[0].first, Action{30});
    CHECK_EQ(aligned[0].second, 0.2);
    CHECK_EQ(aligned[1].first, Action{10});
    CHECK_EQ(aligned[1].second, 0.5);
    CHECK_EQ(aligned[2].first, Action{20});
    CHECK_EQ(aligned[2].second, 0.3);

    // Index pairing — what the writer used to do — would have stamped action 30
    // with action 10's probability. Pin that this is genuinely a different
    // answer, so the test would fail if the alignment were dropped.
    UTILS_CHECK(aligned[0].second != ppo[0].second);
  } TEST_END();

  // -----------------------------------------------------------------------
  // Test 3: size mismatch (e.g. search pruned a root action)
  // -----------------------------------------------------------------------
  TEST_BEGIN("Pruned search root is rejected, not truncated") {
    ActionsAndProbs ppo = {{10, 0.5}, {20, 0.3}, {30, 0.2}};
    std::vector<Action> actions = {10, 20};
    ActionsAndProbs aligned = {{99, 1.0}};
    UTILS_CHECK(!AlignPriorToActions(ppo, actions, &aligned));
    // Output is left untouched on failure so a caller that ignores the return
    // value cannot pick up a half-built vector.
    CHECK_EQ(aligned.size(), size_t{1});
    CHECK_EQ(aligned[0].first, Action{99});
  } TEST_END();

  // -----------------------------------------------------------------------
  // Test 4: same size, different membership
  // -----------------------------------------------------------------------
  TEST_BEGIN("Disjoint action sets of equal size are rejected") {
    ActionsAndProbs ppo = {{10, 0.5}, {20, 0.5}};
    std::vector<Action> actions = {10, 40};
    ActionsAndProbs aligned;
    UTILS_CHECK(!AlignPriorToActions(ppo, actions, &aligned));

    // A duplicated id must not be able to consume one probability twice.
    ActionsAndProbs dup_ppo = {{10, 0.3}, {10, 0.7}};
    std::vector<Action> dup_actions = {10, 20};
    UTILS_CHECK(!AlignPriorToActions(dup_ppo, dup_actions, &aligned));
  } TEST_END();

  // -----------------------------------------------------------------------
  // Test 5: the WriteLabel guard itself
  // -----------------------------------------------------------------------
  TEST_BEGIN("PriorsShareActionOrder rejects exactly what alignment repairs") {
    ActionsAndProbs ppo = {{10, 0.5}, {20, 0.3}, {30, 0.2}};
    ActionsAndProbs teacher = {{30, 0.1}, {10, 0.6}, {20, 0.3}};
    // WriteLabel aborts on this pairing rather than writing mismatched ids.
    UTILS_CHECK(!PriorsShareActionOrder(ppo, teacher));

    ActionsAndProbs aligned;
    std::vector<Action> teacher_actions = {30, 10, 20};
    UTILS_CHECK(AlignPriorToActions(ppo, teacher_actions, &aligned));
    // After alignment the writer's precondition holds.
    UTILS_CHECK(PriorsShareActionOrder(aligned, teacher));

    // Size mismatch is caught too.
    ActionsAndProbs shorter = {{10, 1.0}};
    UTILS_CHECK(!PriorsShareActionOrder(ppo, shorter));
  } TEST_END();

  // =========================================================================
  // PPO finding 5 — warmstart observation-size mismatch
  // =========================================================================

  // -----------------------------------------------------------------------
  // Test 6: the tensor the rollout builds
  // -----------------------------------------------------------------------
  TEST_BEGIN("NetworkObservation matches InformationStateTensor exactly") {
    // The engine provides both tensors and they differ in length; that gap is
    // the whole defect, so assert the premise before relying on it.
    UTILS_CHECK(game->GetType().provides_information_state_tensor);
    UTILS_CHECK(istate_size != obs_tensor_size);

    std::unique_ptr<State> state = AgentTurnState(*game, 0);
    std::vector<float> obs = NetworkObservation(*state, 0);
    CHECK_EQ(static_cast<int64_t>(obs.size()), istate_size);

    std::vector<float> expected = state->InformationStateTensor(0);
    CHECK_EQ(obs.size(), expected.size());
    for (size_t i = 0; i < obs.size(); ++i) {
      CHECK_EQ(obs[i], expected[i]);
    }
  } TEST_END();

  // -----------------------------------------------------------------------
  // Test 7: owner policy-guided rollout
  // -----------------------------------------------------------------------
  TEST_BEGIN("Policy-guided owner rollout runs and feeds model-sized obs") {
    std::unique_ptr<State> state = AgentTurnState(*game, 0);
    RecordingEvaluator evaluator(game->NumDistinctActions());
    evaluator.set_model_input_dim(istate_size);

    SwordmasterPlannerConfig cfg;
    cfg.use_policy_for_owner = true;
    cfg.max_depth = 24;

    std::mt19937 rng(7);
    uint64_t eval_requests = 0;
    // Before WO-19 this aborted inside ObservationTensor: the buffer was
    // istate-sized but ObservationTensor writes obs_tensor_size values, so the
    // engine's `offset == values.size()` check failed.
    double score = RolloutSwordmasterRace(*state, 0, cfg, &rng, &evaluator,
                                          &eval_requests);
    UTILS_CHECK(std::isfinite(score));
    UTILS_CHECK(eval_requests > 0);
    CHECK_EQ(evaluator.observed_sizes().size(),
             static_cast<size_t>(eval_requests));
    for (size_t observed : evaluator.observed_sizes()) {
      CHECK_EQ(static_cast<int64_t>(observed), istate_size);
    }
  } TEST_END();

  // -----------------------------------------------------------------------
  // Test 8: opponent policy-guided rollout
  // -----------------------------------------------------------------------
  TEST_BEGIN("Policy-guided opponent rollout runs and feeds model-sized obs") {
    std::unique_ptr<State> state = AgentTurnState(*game, 0);
    RecordingEvaluator evaluator(game->NumDistinctActions());
    evaluator.set_model_input_dim(istate_size);

    SwordmasterPlannerConfig cfg;
    cfg.use_policy_for_opponents = true;
    cfg.max_depth = 24;

    std::mt19937 rng(11);
    uint64_t eval_requests = 0;
    double score = RolloutSwordmasterRace(*state, 0, cfg, &rng, &evaluator,
                                          &eval_requests);
    UTILS_CHECK(std::isfinite(score));
    UTILS_CHECK(eval_requests > 0);
    CHECK_EQ(evaluator.observed_sizes().size(),
             static_cast<size_t>(eval_requests));
    for (size_t observed : evaluator.observed_sizes()) {
      CHECK_EQ(static_cast<int64_t>(observed), istate_size);
    }
  } TEST_END();

  // =========================================================================
  // PPO finding 8 — Leto opportunity filter used a hard-coded cost
  // =========================================================================

  // -----------------------------------------------------------------------
  // Test 9: the engine's own cost table
  // -----------------------------------------------------------------------
  TEST_BEGIN("AgentSpaceSolariCost applies Leto's Landsraad discount") {
    const AgentSpace* sm = FindAgentSpace(kActionAgentSpaceSwordmaster);
    UTILS_CHECK(sm != nullptr);
    UTILS_CHECK(IsLandsraadSpaceAction(sm->action));
    CHECK_EQ(sm->cost_solari, 8);

    CHECK_EQ(AgentSpaceSolariCost(*sm, static_cast<int>(LeaderId::kLetoAtreides)), 7);
    CHECK_EQ(AgentSpaceSolariCost(*sm, static_cast<int>(LeaderId::kVladimirHarkonnen)), 8);
    CHECK_EQ(AgentSpaceSolariCost(*sm, static_cast<int>(LeaderId::kArmandEcaz)), 8);

    // The discount is Landsraad-only: a non-Landsraad space is unaffected.
    const AgentSpace* arrakeen = FindAgentSpace(kActionAgentSpaceArrakeen);
    UTILS_CHECK(arrakeen != nullptr);
    UTILS_CHECK(!IsLandsraadSpaceAction(arrakeen->action));
    CHECK_EQ(AgentSpaceSolariCost(*arrakeen, static_cast<int>(LeaderId::kLetoAtreides)),
             arrakeen->cost_solari);
  } TEST_END();

  // -----------------------------------------------------------------------
  // Test 10: the corpus filter's actual predicate
  // -----------------------------------------------------------------------
  TEST_BEGIN("Leto seat holding exactly 7 solari is Swordmaster-affordable") {
    std::unique_ptr<State> state = AgentTurnState(*game, 0);
    auto* dstate = static_cast<DuneImperiumState*>(state.get());

    dstate->SetLeaderForTesting(0, static_cast<int>(LeaderId::kLetoAtreides));
    dstate->SetPlayerSolariForTesting(0, 7);
    CHECK_EQ(SwordmasterSolariCost(*dstate, 0), 7);
    UTILS_CHECK(CanAffordSwordmaster(*dstate, 0));
    // The old corpus predicate. This is the state the gate-2 corpus was missing.
    UTILS_CHECK(!(dstate->GetPlayerSolari(0) >= 8));

    // Six is still short, for Leto too.
    dstate->SetPlayerSolariForTesting(0, 6);
    UTILS_CHECK(!CanAffordSwordmaster(*dstate, 0));

    // A non-Leto seat on 7 remains unaffordable — the fix widens the filter for
    // Leto only, it does not loosen it for everyone.
    dstate->SetLeaderForTesting(1, static_cast<int>(LeaderId::kVladimirHarkonnen));
    dstate->SetPlayerSolariForTesting(1, 7);
    CHECK_EQ(SwordmasterSolariCost(*dstate, 1), 8);
    UTILS_CHECK(!CanAffordSwordmaster(*dstate, 1));
    dstate->SetPlayerSolariForTesting(1, 8);
    UTILS_CHECK(CanAffordSwordmaster(*dstate, 1));
  } TEST_END();

  // =========================================================================
  // PPO finding 10 — legacy evaluator was unseeded
  // =========================================================================

  // -----------------------------------------------------------------------
  // Test 11: per-game stream derivation
  // -----------------------------------------------------------------------
  TEST_BEGIN("Legacy eval seeds reproduce per game and separate per stream") {
    auto game_seed = [](uint64_t base, uint64_t g, uint64_t stream) {
      return dune_seed::DeriveSeed(base, dune_seed::kDomainEvalLegacy, g, stream);
    };

    // Same (seed, game, stream) is reproducible — the acceptance criterion.
    CHECK_EQ(game_seed(12, 0, dune_seed::kStreamChance),
             game_seed(12, 0, dune_seed::kStreamChance));

    // Different games, streams, and base seeds are all distinct. Nothing here
    // depends on which worker thread claims the game, which is what made the
    // old per-thread random_device seeding irreproducible.
    std::set<uint64_t> seen;
    for (uint64_t g = 0; g < 64; ++g) {
      UTILS_CHECK(seen.insert(game_seed(12, g, dune_seed::kStreamChance)).second);
      for (uint64_t p = 0; p < 4; ++p) {
        UTILS_CHECK(seen.insert(game_seed(
            12, g, dune_seed::kStreamPolicyPlayer0 + p)).second);
      }
    }
    for (uint64_t g = 0; g < 64; ++g) {
      UTILS_CHECK(seen.insert(game_seed(13, g, dune_seed::kStreamChance)).second);
    }

    // And the derived RNGs actually produce matching draws, not just matching
    // seeds.
    std::mt19937 a = dune_seed::MakeRng32(game_seed(12, 5, dune_seed::kStreamChance));
    std::mt19937 b = dune_seed::MakeRng32(game_seed(12, 5, dune_seed::kStreamChance));
    for (int i = 0; i < 32; ++i) CHECK_EQ(a(), b());
  } TEST_END();

  // -----------------------------------------------------------------------
  // Test 12: domain separation from dune_population_eval
  // -----------------------------------------------------------------------
  TEST_BEGIN("Legacy eval domain is disjoint from the population-eval domains") {
    const uint64_t domains[] = {
        dune_seed::kDomainTrain,        dune_seed::kDomainEvalBaseline,
        dune_seed::kDomainEvalDev,      dune_seed::kDomainEvalFinal,
        dune_seed::kDomainSearchTeacher, dune_seed::kDomainFidelityGate,
        dune_seed::kDomainEvalLegacy};
    std::set<uint64_t> unique_domains(std::begin(domains), std::end(domains));
    CHECK_EQ(unique_domains.size(), sizeof(domains) / sizeof(domains[0]));

    // Same base seed + same game id under two domains must not collide.
    for (uint64_t d : domains) {
      if (d == dune_seed::kDomainEvalLegacy) continue;
      UTILS_CHECK(dune_seed::DeriveSeed(12, d, 0, dune_seed::kStreamChance) !=
                  dune_seed::DeriveSeed(12, dune_seed::kDomainEvalLegacy, 0,
                                        dune_seed::kStreamChance));
    }
  } TEST_END();

  std::cout << "\nAll " << pass_count << "/" << test_count << " tests PASSED!\n";
  return 0;
}
