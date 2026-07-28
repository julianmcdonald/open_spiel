// PWO-3 Amendment 2, battery check 9: root raw priors are IMMUTABLE after
// expansion, and the evaluator covers every legal action.
//
// Protocol: docs/PWO3_AMENDMENT_2_PRECISION_RECOVERY.md sections 3 and 4.
//
// ---------------------------------------------------------------------------
// THE CLAIM UNDER TEST
// ---------------------------------------------------------------------------
// dune_pwo3_prior_recovery replays the grid's emission chain with ZERO
// simulations. That is only a faithful replay of an 800-simulation row if the
// vector FilterAndNormalizeRawPriors emits at the end of a real search equals
// the one it emits straight after expansion. Reading the code says it does --
// the simulation loop writes `visits` and `return_sum`, never `raw_prior` -- but
// "I read the code" is not a check. This test runs both and compares BITWISE.
//
// Two assertions, matching the amendment's check 9:
//
//   A. IMMUTABILITY. A real search of several hundred simulations, then
//      GetRootDiagnostics().raw_priors, must equal the zero-simulation snapshot
//      bit for bit -- not to a tolerance. A tolerance would pass a search that
//      perturbed the priors by an ulp, which is exactly the failure mode that
//      would make a zero-simulation recovery inadmissible.
//
//   B. EVALUATOR COVERAGE. DuneNNEvaluator::PriorAndEvaluate must return an
//      entry for EVERY legal action, with a strictly positive probability. That
//      is what makes dune_puct_is_mcts.cc:285's pre-initialisation of every
//      child to `raw_prior = 0.0` unreachable at emission: :320 overwrites all
//      of them. If the evaluator ever returned a short vector, a genuine 0.0
//      would reach the emitted vector and the recovery premise would be false.
//
// B uses a RANDOMLY INITIALISED network, not the Branch-A checkpoint. The
// properties under test -- cardinality, ordering, strict positivity of a
// max-subtracted softmax over capped logits -- are properties of the evaluator,
// not of the weights, so pinning the test to a 1.4 GB artifact would buy nothing
// and make the test unrunnable without it.
//
// A uses a deterministic stub evaluator for the same reason: the immutability
// claim is about who WRITES child_info[a].raw_prior, and a stub makes any write
// show up as an exact, obvious difference rather than as inference noise.

#include <cassert>
#include <cmath>
#include <cstring>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "open_spiel/games/dune_imperium/dune_imperium.h"
#include "open_spiel/spiel.h"

#include <torch/torch.h>

#include "dune_evaluator.h"
#include "dune_network.h"
#include "dune_puct_is_mcts.h"
#include "dune_pwo3_common.h"

namespace open_spiel {
namespace {

// A deterministic stand-in for the network. Priors are an uneven, reproducible
// function of the action id -- deliberately NOT uniform, so that a normalising
// or overwriting bug cannot pass by accident -- and small enough in the tail to
// exercise the same magnitudes the six-decimal floor censored.
class DeterministicPriorEvaluator : public algorithms::Evaluator {
 public:
  std::vector<double> Evaluate(const State& state) override {
    return std::vector<double>(state.NumPlayers(), 0.125);
  }

  ActionsAndProbs Prior(const State& state) override {
    std::vector<Action> legal = state.LegalActions();
    ActionsAndProbs out;
    if (legal.empty()) return out;
    double sum = 0.0;
    std::vector<double> w;
    w.reserve(legal.size());
    for (size_t i = 0; i < legal.size(); ++i) {
      // exp of a bounded, action-dependent logit: same shape as the capped
      // softmax, with a genuinely tiny tail entry.
      const double logit =
          std::sin(static_cast<double>(legal[i] % 97)) * 3.0 - 4.0 * (i % 5);
      const double e = std::exp(logit);
      w.push_back(e);
      sum += e;
    }
    for (size_t i = 0; i < legal.size(); ++i) {
      out.push_back({legal[i], w[i] / sum});
    }
    return out;
  }

  std::pair<ActionsAndProbs, std::vector<double>> PriorAndEvaluate(
      const State& state) override {
    return {Prior(state), Evaluate(state)};
  }
};

DuneSearchConfig PinnedConfig(int max_sims) {
  // The registered controller pin (registration section 2), the values every
  // PWO-3 grid row was produced under.
  DuneSearchConfig c;
  c.max_simulations = max_sims;
  c.relative_time_budget_ms = std::numeric_limits<double>::infinity();
  c.max_nodes = 200000;
  c.puct_c = 0.3;
  c.opponent_mode = SearchOpponentMode::kPolicy;
  c.temperature = 0.0;
  c.opponent_temperature = 1.0;
  c.max_world_samples = -1;
  c.utility_divisor = 4.0;
  c.min_visit_threshold = 2;
  c.dirichlet_epsilon = 0.0;
  c.check_strategic_state = false;
  c.root_prior_temperature = 1.0;
  c.covered_prior_threshold = 0.50;
  c.conservative_stability_checkpoint_fraction = 0.5;
  c.seed = 1301;
  c.fixed_continuation_reserve = 0;
  c.purchase_combat_budget = 16;
  c.live_continuation_reserve_seconds = 0.0;
  c.fixed_session_limit = max_sims;
  return c;
}

// Plays forward from the initial state until a decision node with >= 3 legal
// actions is reached, so the search has something to spread visits over. The
// walk is seeded and takes the lowest legal action, so the test root is fixed.
std::unique_ptr<State> FindTestRoot(const std::shared_ptr<const Game>& game,
                                    int min_legal) {
  std::mt19937 rng(20260728);
  auto state = game->NewInitialState();
  for (int step = 0; step < 4000; ++step) {
    if (state->IsTerminal()) break;
    if (state->IsChanceNode()) {
      auto outcomes = state->ChanceOutcomes();
      std::uniform_int_distribution<size_t> d(0, outcomes.size() - 1);
      state->ApplyAction(outcomes[d(rng)].first);
      continue;
    }
    if (state->CurrentPlayer() == kSimultaneousPlayerId) {
      std::vector<Action> joint;
      for (int p = 0; p < state->NumPlayers(); ++p) {
        joint.push_back(state->LegalActions(p).front());
      }
      state->ApplyActions(joint);
      continue;
    }
    if (static_cast<int>(state->LegalActions().size()) >= min_legal) {
      return state;
    }
    state->ApplyAction(state->LegalActions().front());
  }
  return nullptr;
}

std::vector<double> EmitRawPriors(const State& state,
                                  std::shared_ptr<algorithms::Evaluator> ev,
                                  int max_sims, int* simulations_out) {
  DuneSearchConfig cfg = PinnedConfig(max_sims);
  DunePUCTISMCTSBot bot(cfg, ev);
  DuneSearchResult res =
      bot.RunSearch(state, max_sims,
                    std::numeric_limits<double>::infinity(), 0);
  if (simulations_out) *simulations_out = res.simulations_completed;
  SearchDiagnostics d =
      bot.GetRootDiagnostics(state, cfg.min_visit_threshold, kInvalidAction);
  // The `raw_priors` branch is the one every PWO-3 row took. If the root is
  // absent from the tree the field is EMPTY and `priors` is already raw -- a
  // different branch, and not what this test is about.
  SPIEL_CHECK_EQ(d.raw_priors.size(), d.actions.size());
  return d.raw_priors;
}

void TestRootRawPriorsAreImmutableAcrossARealSearch() {
  auto game = LoadGame("dune_imperium");
  auto state = FindTestRoot(game, /*min_legal=*/3);
  SPIEL_CHECK_TRUE(state != nullptr);
  const std::vector<Action> legal = state->LegalActions();
  SPIEL_CHECK_GE(legal.size(), 3u);
  SPIEL_CHECK_TRUE(pwo3::IsAscending(legal));

  auto ev = std::static_pointer_cast<algorithms::Evaluator>(
      std::make_shared<DeterministicPriorEvaluator>());

  int sims_zero = -1;
  const std::vector<double> snapshot =
      EmitRawPriors(*state, ev, /*max_sims=*/0, &sims_zero);
  SPIEL_CHECK_EQ(sims_zero, 0);
  SPIEL_CHECK_EQ(snapshot.size(), legal.size());

  int sims_real = -1;
  const std::vector<double> after =
      EmitRawPriors(*state, ev, /*max_sims=*/400, &sims_real);
  // "Hundreds of simulations" has to actually happen, or the comparison is
  // vacuous: a search that ran zero simulations trivially matches a snapshot.
  SPIEL_CHECK_GE(sims_real, 300);
  SPIEL_CHECK_EQ(after.size(), snapshot.size());

  for (size_t i = 0; i < snapshot.size(); ++i) {
    // BITWISE, not approximate: memcmp of the IEEE payloads.
    uint64_t a_bits = 0, b_bits = 0;
    std::memcpy(&a_bits, &snapshot[i], sizeof(a_bits));
    std::memcpy(&b_bits, &after[i], sizeof(b_bits));
    if (a_bits != b_bits) {
      std::cerr << "FAIL: root raw prior for action " << legal[i]
                << " changed across a " << sims_real << "-simulation search: "
                << snapshot[i] << " -> " << after[i] << "\n";
      std::abort();
    }
    // The recovery premise itself: a capped softmax cannot emit an exact zero,
    // so no legal action can carry 0.0 in memory.
    SPIEL_CHECK_GT(snapshot[i], 0.0);
  }
  std::cout << "  PASS immutability: " << snapshot.size()
            << " root raw priors identical bitwise after " << sims_real
            << " simulations\n";
}

void TestEvaluatorReturnsAnEntryForEveryLegalAction() {
  auto game = LoadGame("dune_imperium");
  auto state = FindTestRoot(game, /*min_legal=*/3);
  SPIEL_CHECK_TRUE(state != nullptr);
  const std::vector<Action> legal = state->LegalActions();

  const int64_t obs_size = game->GetType().provides_information_state_tensor
                               ? game->InformationStateTensorSize()
                               : game->ObservationTensorSize();
  const int64_t act_size = game->NumDistinctActions();
  torch::manual_seed(20260728);
  // A small RANDOM net: cardinality and strict positivity are properties of
  // DuneNNEvaluator, not of Branch-A's weights.
  auto model = std::make_shared<SharedDunePolicyValueNetImpl>(
      obs_size, /*hidden_dim=*/32, act_size, /*num_blocks=*/1,
      /*nonlinear_value_head=*/false);
  model->to(torch::kCPU);
  model->eval();
  // logit_cap 10.0 is the registered pin (--candidate_logit_cap=10.0).
  DuneNNEvaluator evaluator(model, torch::Device(torch::kCPU), 10.0f);

  auto pair = evaluator.PriorAndEvaluate(*state);
  const ActionsAndProbs& priors = pair.first;
  SPIEL_CHECK_EQ(priors.size(), legal.size());
  double sum = 0.0;
  for (size_t i = 0; i < legal.size(); ++i) {
    // Same order as LegalActions(): the expansion at :320 indexes child_info by
    // priors[i].first, so an out-of-order or short vector would leave some
    // child at the :285 pre-init 0.0.
    SPIEL_CHECK_EQ(priors[i].first, legal[i]);
    SPIEL_CHECK_GT(priors[i].second, 0.0);
    SPIEL_CHECK_TRUE(std::isfinite(priors[i].second));
    sum += priors[i].second;
  }
  SPIEL_CHECK_LT(std::abs(sum - 1.0), 1e-12);
  SPIEL_CHECK_EQ(pair.second.size(), static_cast<size_t>(state->NumPlayers()));
  std::cout << "  PASS evaluator coverage: " << priors.size()
            << " strictly-positive entries for " << legal.size()
            << " legal actions\n";
}

}  // namespace
}  // namespace open_spiel

int main(int argc, char** argv) {
  at::set_num_threads(1);
  std::cout << "dune_pwo3_immutability_test (Amendment 2, battery check 9)\n";
  open_spiel::TestRootRawPriorsAreImmutableAcrossARealSearch();
  open_spiel::TestEvaluatorReturnsAnEntryForEveryLegalAction();
  std::cout << "ALL PASS\n";
  return 0;
}
