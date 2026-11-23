// Copyright 2026 DeepMind Technologies Limited.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef OPEN_SPIEL_EXAMPLES_DUNE_COMPOUND_TURN_TREE_H_
#define OPEN_SPIEL_EXAMPLES_DUNE_COMPOUND_TURN_TREE_H_

#include <algorithm>
#include <cmath>
#include <condition_variable>
#include <deque>
#include <functional>
#include <future>
#include <limits>
#include <memory>
#include <mutex>
#include <ostream>
#include <string>
#include <thread>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#include "open_spiel/examples/dune_batched_evaluator.h"
#include "open_spiel/games/dune_imperium/dune_imperium.h"
#include "open_spiel/spiel.h"

namespace open_spiel {
namespace dune_imperium {

// A node in the intra-turn search tree tracking visits and accumulated utility.
struct TurnSearchNode {
  int visits = 0;
  double utility_sum = 0.0;
  std::unordered_map<Action, std::unique_ptr<TurnSearchNode>> children;

  double MeanUtility() const {
    return visits > 0 ? (utility_sum / static_cast<double>(visits)) : 0.0;
  }

  TurnSearchNode* GetOrCreateChild(Action a) {
    auto it = children.find(a);
    if (it == children.end()) {
      auto new_node = std::make_unique<TurnSearchNode>();
      TurnSearchNode* ptr = new_node.get();
      children[a] = std::move(new_node);
      return ptr;
    }
    return it->second.get();
  }

  void RecordOutcome(double u) {
    visits++;
    utility_sum += u;
  }
};

// Search tree preserved across sequential actions within a single player turn.
// Clears when the player turn ends (kActionEndTurn).
class TurnSearchTree {
 public:
  TurnSearchTree() { Reset(); }

  void Reset() {
    root_ = std::make_unique<TurnSearchNode>();
    current_node_ = root_.get();
    constraint_history_.clear();
    public_card_constraints_.clear();
    public_card_source_state_.reset();
  }

  void BindPublicCardKnowledge(const State& state) {
    const auto* dune = dynamic_cast<const DuneImperiumState*>(&state);
    if (dune == nullptr) return;
    const std::string digest = dune->PublicCardKnowledgeDigest();
    const auto& history = state.History();
    if (!public_card_constraints_.empty()) {
      bool matches = false;
      if (constraint_history_ == history) {
        matches = public_card_constraints_ == digest;
      } else if (public_card_source_state_ != nullptr &&
                 history.size() > constraint_history_.size() &&
                 std::equal(constraint_history_.begin(), constraint_history_.end(),
                            history.begin())) {
        // Replay the authoritative source to distinguish ordinary action
        // effects from additional captures that arrived between searches.
        matches = true;
        for (size_t i = constraint_history_.size(); i < history.size(); ++i) {
          const auto legal = public_card_source_state_->LegalActions();
          if (std::find(legal.begin(), legal.end(), history[i]) == legal.end()) {
            matches = false;
            break;
          }
          public_card_source_state_->ApplyAction(history[i]);
        }
        if (matches) {
          const auto* expected = dynamic_cast<const DuneImperiumState*>(
              public_card_source_state_.get());
          matches = expected != nullptr && expected->PublicCardKnowledgeDigest() == digest;
        }
      }
      if (!matches) Reset();
    }
    constraint_history_ = history;
    public_card_constraints_ = digest;
    public_card_source_state_ = state.Clone();
  }

  // Advance the current root to child 'action' when that action is taken in the real game.
  void Advance(Action action) {
    if (!current_node_) {
      Reset();
      return;
    }
    current_node_ = current_node_->GetOrCreateChild(action);
  }

  // Record a simulated rollout path and its terminal payoff down the active subtree.
  void RecordRollout(Action first_action, const std::vector<Action>& subsequent_turn_actions, double terminal_u) {
    if (!current_node_) return;

    TurnSearchNode* node = current_node_->GetOrCreateChild(first_action);
    node->RecordOutcome(terminal_u);

    for (Action a : subsequent_turn_actions) {
      node = node->GetOrCreateChild(a);
      node->RecordOutcome(terminal_u);
    }
  }

  // Check if an action has preserved rollouts from earlier in the turn.
  bool HasPriorData(Action a) const {
    if (!current_node_) return false;
    auto it = current_node_->children.find(a);
    return (it != current_node_->children.end() && it->second->visits > 0);
  }

  int GetVisits(Action a) const {
    if (!current_node_) return 0;
    auto it = current_node_->children.find(a);
    return (it != current_node_->children.end()) ? it->second->visits : 0;
  }

  double GetUtilitySum(Action a) const {
    if (!current_node_) return 0.0;
    auto it = current_node_->children.find(a);
    return (it != current_node_->children.end()) ? it->second->utility_sum : 0.0;
  }

  double GetMeanUtility(Action a) const {
    if (!current_node_) return 0.0;
    auto it = current_node_->children.find(a);
    return (it != current_node_->children.end()) ? it->second->MeanUtility() : 0.0;
  }

  bool HasConfirmedSubtree(Action raw_action, int min_visits = 16) const {
    return GetVisits(raw_action) >= min_visits;
  }

  Action BestConfirmedAction(
      Action raw_action,
      const ActionsAndProbs& raw_prior,
      const std::vector<Action>& legal_actions,
      int min_visits = 16,
      double base_margin = 0.12) const {
    if (!current_node_ || current_node_->children.empty()) {
      return raw_action;
    }
    Action best_action = kInvalidAction;
    double best_mean = -std::numeric_limits<double>::infinity();
    if (!legal_actions.empty()) {
      for (Action a : legal_actions) {
        auto it = current_node_->children.find(a);
        if (it != current_node_->children.end() && it->second &&
            it->second->visits > 0) {
          const double q = it->second->MeanUtility();
          if (q > best_mean) {
            best_mean = q;
            best_action = a;
          }
        }
      }
    } else {
      for (const auto& [a, child] : current_node_->children) {
        if (child && child->visits > 0) {
          const double q = child->MeanUtility();
          if (q > best_mean) {
            best_mean = q;
            best_action = a;
          }
        }
      }
    }
    if (best_action == kInvalidAction || best_action == raw_action) {
      return raw_action;
    }
    const int best_visits = GetVisits(best_action);
    const int raw_visits = GetVisits(raw_action);
    if (best_visits < min_visits || raw_visits < min_visits) {
      return raw_action;
    }
    const double raw_mean = GetMeanUtility(raw_action);
    double p_raw = 0.0;
    double p_best = 0.0;
    for (const auto& [a, p] : raw_prior) {
      if (a == raw_action) p_raw = p;
      if (a == best_action) p_best = p;
    }
    const double eff_margin =
        std::min(base_margin, 0.5 * std::max(0.04, 2.25 - raw_mean));
    const double req_margin =
        eff_margin +
        0.02 * std::max(0.0, std::log((p_raw + 1e-6) / (p_best + 1e-6)));
    return (best_mean - raw_mean >= req_margin) ? best_action : raw_action;
  }

  Action BestConfirmedAction(
      Action raw_action,
      const ActionsAndProbs& raw_prior,
      int min_visits = 16,
      double base_margin = 0.12) const {
    return BestConfirmedAction(raw_action, raw_prior, std::vector<Action>{},
                               min_visits, base_margin);
  }

  TurnSearchNode* current_node() const { return current_node_; }

 private:
  std::unique_ptr<TurnSearchNode> root_;
  TurnSearchNode* current_node_ = nullptr;
  std::vector<Action> constraint_history_;
  std::string public_card_constraints_;
  std::unique_ptr<State> public_card_source_state_;
};

// Search candidate pairing mode across rollouts
enum class SearchPairingMode {
  kUnpaired,  // Candidate-independent world/rollout seeds (default today)
  kPaired     // Shared world/rollout seeds across candidates
};

// R7-55 search categories for hybrid on-policy search-in-training
enum class R755SearchCategory : int {
  kIneligible = 0,
  kAgentCardOrStartPlot = 1,
  kImmediateSpacePlacement = 2,
  kRevealPurchase = 3,
  kStandaloneOrPreRevealPlot = 4,
  kCombatIntrigue = 5,
};

inline const char* R755SearchCategoryName(R755SearchCategory cat) {
  switch (cat) {
    case R755SearchCategory::kIneligible: return "ineligible";
    case R755SearchCategory::kAgentCardOrStartPlot: return "agent_card_or_start_plot";
    case R755SearchCategory::kImmediateSpacePlacement: return "immediate_space_placement";
    case R755SearchCategory::kRevealPurchase: return "reveal_purchase";
    case R755SearchCategory::kStandaloneOrPreRevealPlot: return "standalone_or_pre_reveal_plot";
    case R755SearchCategory::kCombatIntrigue: return "combat_intrigue";
  }
  return "unknown";
}

inline std::ostream& operator<<(std::ostream& os, R755SearchCategory cat) {
  return os << R755SearchCategoryName(cat);
}

inline bool IsCardOrTechAcquisitionAction(Action a) {
  // 10..17: Imperium row slots (0..4 standard, up to +7 extended)
  if (a >= kActionBuyImperiumRow0 && a < kActionBuyImperiumRow0 + 8) return true;
  // 572..579: Helena reserve row slots
  if (a >= kActionBuyHelenaReserve0 && a < kActionBuyHelenaReserve0 + 8) return true;
  // 598, 599: Reserve cards (Arrakis Liaison, The Spice Must Flow)
  if (a == kActionBuyReserveArrakisLiaison || a == kActionBuyReserveTheSpiceMustFlow) return true;
  // 70..75: Tech acquisition (spice), strictly excluding 76 (Skip)
  if (a >= kActionTechAcquire0 && a < kActionTechAcquireSkip) return true;
  // 91..95: Tleilaxu row slots, strictly excluding 90 (Reclaimed Forces) and 96 (Skip)
  if (a >= kActionTleilaxuAcquire0 + 1 && a < kActionTleilaxuAcquireSkip) return true;
  // 554..556: Intrigue choice tech acquisition (e.g. Ixian Engineer), excluding 557 (Skip)
  if (a >= kActionIntrigueChoiceTech0 && a < kActionIntrigueChoiceTechSkip) return true;
  // 770..775: Tech acquisition with Solari (Sardaukar/Ix), excluding 776 (Skip)
  if (a >= kActionTechAcquireWithSolari0 && a < kActionTechAcquireWithSolariSkip) return true;
  return false;
}

inline bool IsR755PurchaseAction(Action a) {
  return IsCardOrTechAcquisitionAction(a) || a == kActionTleilaxuAcquire0;
}

inline R755SearchCategory ClassifyR755SearchCategory(
    const State& state,
    Player searched_player,
    bool space_placed_this_turn,
    int category_mask = 31,
    bool strict_cat3_card_tech_only = false) {
  if (state.IsChanceNode() || state.CurrentPlayer() != searched_player) {
    return R755SearchCategory::kIneligible;
  }
  std::vector<Action> legal_actions = state.LegalActions();
  if (legal_actions.size() <= 1) {
    return R755SearchCategory::kIneligible;
  }
  const auto* dune_state = dynamic_cast<const dune_imperium::DuneImperiumState*>(&state);
  if (!dune_state) {
    return R755SearchCategory::kIneligible;
  }
  if (dune_state->phase() == GamePhase::kLeaderOfferChance ||
      dune_state->phase() == GamePhase::kLeaderDraft) {
    return R755SearchCategory::kIneligible;
  }

  R755SearchCategory cat = R755SearchCategory::kIneligible;

  // Category 5: Combat intrigue / commit
  if (dune_state->phase() == GamePhase::kCombat) {
    bool has_combat_act = false;
    for (Action a : legal_actions) {
      if (a == kActionCombatPass || a == kActionFamilyAtomics ||
          (a >= kActionPlayIntrigueCombatCard0 && a < kActionPlayIntrigueCombatCard0 + kMaxIntrigueCards) ||
          (a >= kActionCombatCommit0 && a <= kActionCombatCommit0 + 47)) {
        has_combat_act = true;
        break;
      }
    }
    if (has_combat_act) {
      cat = R755SearchCategory::kCombatIntrigue;
    }
  }

  // Category 3: Reveal purchase
  if (cat == R755SearchCategory::kIneligible && dune_state->IsPlayerInRevealTurn(searched_player)) {
    if (strict_cat3_card_tech_only) {
      if (dune_state->GetCurrentRound() <= 6) {
        bool has_card_or_tech = false;
        for (Action a : legal_actions) {
          if (IsCardOrTechAcquisitionAction(a)) {
            has_card_or_tech = true;
            break;
          }
        }
        if (has_card_or_tech) {
          cat = R755SearchCategory::kRevealPurchase;
        }
      }
    } else {
      bool has_purchase = false;
      for (Action a : legal_actions) {
        if (IsR755PurchaseAction(a)) {
          has_purchase = true;
          break;
        }
      }
      if (has_purchase) {
        cat = R755SearchCategory::kRevealPurchase;
      }
    }
  }

  // Category 1 & Category 2: Agent turns
  if (cat == R755SearchCategory::kIneligible &&
      dune_state->phase() == GamePhase::kAgentTurns &&
      !dune_state->IsPlayerInRevealTurn(searched_player) &&
      dune_state->GetPlayerAgentsRemainingForTesting(searched_player) > 0) {
    bool has_agent_card = false;
    bool has_agent_space = false;
    for (Action a : legal_actions) {
      if ((a >= kActionSelectAgentCard0 && a <= kActionSelectAgentCard0 + 255) ||
          a == kActionPlayKwisatzHaderach) {
        has_agent_card = true;
      }
      if (a >= kActionAgentSpaceConspire && a <= kActionAgentSpaceSietchTabr) {
        has_agent_space = true;
      }
    }
    if (has_agent_card && !has_agent_space) {
      cat = R755SearchCategory::kAgentCardOrStartPlot;
    } else if (has_agent_space && !space_placed_this_turn) {
      cat = R755SearchCategory::kImmediateSpacePlacement;
    }
  }

  // Category 4: Standalone or pre-reveal plot
  if (cat == R755SearchCategory::kIneligible) {
    bool has_plot_card = false;
    bool all_other_allowed = true;
    for (Action a : legal_actions) {
      if (a >= kActionPlayIntriguePlotCard0 && a < kActionPlayIntriguePlotCard0 + kMaxIntrigueCards) {
        has_plot_card = true;
      } else if (a == kActionReveal || a == kActionEndTurn) {
        // Allowed
      } else {
        all_other_allowed = false;
      }
    }
    if (has_plot_card && all_other_allowed) {
      cat = R755SearchCategory::kStandaloneOrPreRevealPlot;
    }
  }

  if (cat != R755SearchCategory::kIneligible) {
    int bit = static_cast<int>(cat) - 1;
    if (((category_mask >> bit) & 1) == 0) {
      return R755SearchCategory::kIneligible;
    }
  }
  return cat;
}

// Search rollout truncation mode
enum class SearchTruncationMode {
  kTerminal,               // Full rollout until game is terminal (default today)
  kNextRoundSearchPlayer,  // Cut off at search player's first decision in subsequent round
  kPlus2Rounds             // Cut off at search player's first decision in start_round + 2
};

// Search rollout CRN mode:
// kDeckOrderCrn implements stream-and-reshuffle-epoch-indexed CRN
// (separating the 15 chance streams and resynchronizing draw indices whenever a deck reshuffles).
enum class SearchRolloutCrnMode {
  kSequential = 0,
  kDeckOrderCrn = 1,
  kEventKeyedCrn = 2,
};

// Conversion scale from value head predicted return to ladder game returns
inline constexpr double kValueToLadderScale = 4.0;

// Converts a value head prediction (scaled return [-0.25, +0.25]) to raw ladder utility units [-1.0, +1.0]
inline double TruncatedLeafUtility(double value_head_prediction) {
  return value_head_prediction * kValueToLadderScale;
}

struct ActiveRolloutReleaseGuard {
  BatchedNNEvaluator* evaluator = nullptr;
  IGameEvaluator* game_evaluator = nullptr;
  int count = 1;

  ActiveRolloutReleaseGuard() = default;
  ActiveRolloutReleaseGuard(BatchedNNEvaluator* ev, int c = 1)
      : evaluator(ev), game_evaluator(nullptr), count(c) {}
  ActiveRolloutReleaseGuard(IGameEvaluator* ev, int c = 1)
      : evaluator(nullptr), game_evaluator(ev), count(c) {}

  ~ActiveRolloutReleaseGuard() {
    if (evaluator != nullptr) {
      evaluator->AdjustActiveRollouts(-count);
    }
    if (game_evaluator != nullptr) {
      game_evaluator->AdjustActiveRollouts(-count);
    }
  }
};

class SearchRolloutThreadPool {
 public:
  static constexpr size_t kMaxWorkers = 512;

  static SearchRolloutThreadPool& Instance() {
    static SearchRolloutThreadPool pool;
    return pool;
  }

  ~SearchRolloutThreadPool() {
    {
      std::lock_guard<std::mutex> lock(mu_);
      stop_ = true;
    }
    cv_.notify_all();
    for (auto& w : workers_) {
      if (w.joinable()) w.join();
    }
  }

  template <typename Fn>
  std::future<std::invoke_result_t<Fn>> Submit(Fn&& fn) {
    using R = std::invoke_result_t<Fn>;
    auto task = std::make_shared<std::packaged_task<R()>>(std::forward<Fn>(fn));
    std::future<R> fut = task->get_future();
    {
      std::lock_guard<std::mutex> lock(mu_);
      if (!stop_) {
        tasks_.emplace_back([task]() { (*task)(); });
        if (idle_workers_ < static_cast<int>(tasks_.size()) &&
            workers_.size() < kMaxWorkers) {
          SpawnWorkerLocked();
        }
        if (idle_workers_ >= static_cast<int>(tasks_.size()) ||
            !workers_.empty()) {
          cv_.notify_one();
          return fut;
        }
      }
    }
    std::thread([task]() { (*task)(); }).detach();
    return fut;
  }

 private:
  SearchRolloutThreadPool() = default;

  void SpawnWorkerLocked() {
    ++idle_workers_;
    workers_.emplace_back([this]() {
      for (;;) {
        std::function<void()> job;
        {
          std::unique_lock<std::mutex> lock(mu_);
          cv_.wait(lock, [this]() { return stop_ || !tasks_.empty(); });
          if (stop_ && tasks_.empty()) {
            --idle_workers_;
            return;
          }
          job = std::move(tasks_.front());
          tasks_.pop_front();
          --idle_workers_;
        }
        job();
        {
          std::lock_guard<std::mutex> lock(mu_);
          ++idle_workers_;
        }
      }
    });
  }

  std::mutex mu_;
  std::condition_variable cv_;
  std::deque<std::function<void()>> tasks_;
  std::vector<std::thread> workers_;
  int idle_workers_ = 0;
  bool stop_ = false;
};

}  // namespace dune_imperium
}  // namespace open_spiel

#endif  // OPEN_SPIEL_EXAMPLES_DUNE_COMPOUND_TURN_TREE_H_
