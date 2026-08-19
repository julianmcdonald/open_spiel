// A deliberately UNFUSED MCTS evaluator: the policy prior comes from one model,
// the scalar leaf value from a different, frozen one.
//
// WHY THIS CLASS EXISTS
//
// open_spiel::algorithms::Evaluator (open_spiel/algorithms/mcts.h) has three
// entry points -- Prior(), Evaluate(), and PriorAndEvaluate(). The third is a
// convenience whose BASE implementation is literally {Prior(state),
// Evaluate(state)}: two independent calls. Both production evaluators in this
// directory override it and FUSE those two into one network forward pass.
// DuneNNEvaluator (dune_evaluator.h) and BatchedNNEvaluator
// (dune_batched_evaluator.h) each run the net once at a leaf and read the
// policy head and the value head off that single result. MCTS calls
// PriorAndEvaluate at every leaf precisely so that this fusion is available,
// and when both heads are supposed to come from the same weights the fusion is
// a pure win with no semantic content at all.
//
// That is exactly why "search with model X's policy but model Y's value" cannot
// be expressed by composing the existing evaluators. Hand a fused evaluator to
// MCTS and every leaf takes BOTH heads off whichever single network that
// evaluator owns. There is no seam to interpose on: the value is never
// requested separately, so there is nothing to redirect. Wire two models
// together naively -- hold the candidate because you want its priors, and
// assume the value is coming from somewhere else -- and the fused override
// silently wins. The candidate's own value head backs up through the tree, no
// call site looks wrong, no assertion fires, no diagnostic changes shape, and
// the run measures a different experiment than the registered one. When the
// value head is itself the object of study, that failure has the value head
// steering the search that produces its own training targets.
//
// SplitPolicyValueEvaluator makes that outcome impossible by construction. It
// holds two evaluators and routes each entry point to the source that owns it
// -- including, and especially, the fused entry point, which it overrides in
// order to UN-fuse. The second forward pass is the price of the guarantee, not
// an oversight awaiting an optimisation: any future change that collapses those
// two calls back into one reintroduces the precise defect this class was
// written to prevent, and it will do so silently.
//
// OWNERSHIP AND THREADING. Both sources are held by shared_ptr because that is
// how MCTS and this directory's callers already pass evaluators around, and
// because a frozen value model is normally ONE object shared across arms,
// threads, and searches rather than a per-search copy. This class adds no
// mutable state after construction and performs no synchronisation, so it is
// exactly as thread-safe as the two evaluators it wraps -- no more, no less.
//
// ALIASING IS LEGAL. Passing the same evaluator as both sources is well defined
// and degenerates to that evaluator's own Prior/Evaluate behaviour, at the cost
// of the fusion, since the two calls stay separate. That degenerate wiring is
// useful as a control: it isolates the effect of splitting the MODELS from the
// effect of splitting the CALLS, which are two different changes that a naive
// A/B would otherwise confound.

#pragma once

#include <memory>
#include <utility>
#include <vector>

#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"
#include "open_spiel/algorithms/mcts.h"

namespace open_spiel {

class SplitPolicyValueEvaluator : public open_spiel::algorithms::Evaluator {
 public:
  // Both sources are required. The checks are here, at wiring time, rather than
  // at the first leaf: a null source discovered inside a search would surface
  // as a crash thousands of simulations into a generation, with the run's
  // configuration already committed and the failure attributed to search rather
  // than to construction.
  SplitPolicyValueEvaluator(
      std::shared_ptr<open_spiel::algorithms::Evaluator> policy_source,
      std::shared_ptr<open_spiel::algorithms::Evaluator> value_source)
      : policy_source_(std::move(policy_source)),
        value_source_(std::move(value_source)) {
    SPIEL_CHECK_TRUE(policy_source_ != nullptr);
    SPIEL_CHECK_TRUE(value_source_ != nullptr);
  }

  // The prior is the POLICY source's, verbatim. No renormalisation, no
  // masking, no reordering: whatever transformation the source applies is the
  // transformation this class is supposed to preserve, and re-deriving any of
  // it here would mean the split evaluator's priors could differ from the ones
  // that same model produces when used alone.
  ActionsAndProbs Prior(const State& state) override {
    return policy_source_->Prior(state);
  }

  // The leaf value is the VALUE source's, verbatim, for every player. This is
  // the number MCTS backs up, so it is the one that decides which lines the
  // tree explores -- the entire reason the two models are separated.
  std::vector<double> Evaluate(const State& state) override {
    return value_source_->Evaluate(state);
  }

  // The load-bearing override.
  //
  // MCTS calls this at every leaf, so this -- not Prior() and not Evaluate() --
  // is the function that actually runs during a search. Fusing here, the way
  // DuneNNEvaluator and BatchedNNEvaluator both do, would take the policy AND
  // the value off a single forward pass of a single model, silently routing the
  // candidate's value head into search backups. That substitution is invisible:
  // it produces well-formed priors and well-formed values, changes no
  // signature, and trips no check. It is the exact failure this class exists to
  // prevent, so the two calls below are kept independent on purpose.
  //
  // Note also what is NOT written here: this does not delegate to either
  // source's own PriorAndEvaluate. Doing so would hand the fusion decision back
  // to a source that is very likely to fuse, and would additionally pull the
  // policy source's values (or the value source's priors) into existence for no
  // reason. Prior() and Evaluate() are the narrow entry points, and they are
  // the ones used.
  std::pair<ActionsAndProbs, std::vector<double>> PriorAndEvaluate(
      const State& state) override {
    // Braced-init-list elements are evaluated left to right, so the policy call
    // completes before the value call begins; neither can observe the other.
    return {policy_source_->Prior(state), value_source_->Evaluate(state)};
  }

  // Accessors so a caller can assert WHICH models it wired in -- the swap of
  // these two arguments is the other silent failure mode, and it is only
  // detectable by identity, since both sources have the same static type.
  const std::shared_ptr<open_spiel::algorithms::Evaluator>& policy_source() const {
    return policy_source_;
  }
  const std::shared_ptr<open_spiel::algorithms::Evaluator>& value_source() const {
    return value_source_;
  }

 private:
  // Supplies priors only. Its value head is never consulted through this class.
  std::shared_ptr<open_spiel::algorithms::Evaluator> policy_source_;
  // Supplies leaf values only. Its policy head is never consulted through this
  // class. In the intended configuration this is the frozen model.
  std::shared_ptr<open_spiel::algorithms::Evaluator> value_source_;
};

} // namespace open_spiel
