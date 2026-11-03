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

#ifndef OPEN_SPIEL_EXAMPLES_DUNE_COMPOUND_TURN_SEARCH_H_
#define OPEN_SPIEL_EXAMPLES_DUNE_COMPOUND_TURN_SEARCH_H_

#include <functional>
#include <memory>
#include <random>
#include <unordered_map>
#include <vector>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <future>
#include <set>
#include <limits>
#include <numeric>

#include "open_spiel/examples/dune_batched_evaluator.h"
#include "open_spiel/examples/dune_online_search_collector.h"
#include "open_spiel/examples/dune_seed_utils.h"
#include "open_spiel/games/dune_imperium/dune_imperium.h"
#include "open_spiel/games/dune_imperium/dune_imperium_common.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"

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

inline bool IsR755PurchaseAction(Action a) {
  if (a >= kActionBuyImperiumRow0 && a <= kActionBuyImperiumRow0 + 7) return true;
  if (a >= kActionBuyHelenaReserve0 && a <= kActionBuyHelenaReserve0 + 7) return true;
  if (a == kActionBuyReserveArrakisLiaison || a == kActionBuyReserveTheSpiceMustFlow) return true;
  if (a >= kActionTechAcquire0 && a <= kActionTechAcquireSkip) return true;
  if (a >= kActionTleilaxuAcquire0 && a <= kActionTleilaxuAcquireSkip) return true;
  if (a >= kActionTechAcquireWithSolari0 && a <= kActionTechAcquireWithSolariSkip) return true;
  return false;
}

inline R755SearchCategory ClassifyR755SearchCategory(
    const State& state,
    Player searched_player,
    bool space_placed_this_turn,
    int category_mask = 31) {
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
  kNextRoundSearchPlayer   // Cut off at search player's first decision in subsequent round
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

struct EpochCardDrawCounter {
  std::array<std::pair<Action, int>, 32> inline_counts{};
  int inline_size = 0;
  std::vector<std::pair<Action, int>> overflow;

  void clear() {
    inline_size = 0;
    overflow.clear();
  }
  int get(Action a) const {
    for (int i = 0; i < inline_size; ++i) {
      if (inline_counts[i].first == a) return inline_counts[i].second;
    }
    for (const auto& kv : overflow) {
      if (kv.first == a) return kv.second;
    }
    return 0;
  }
  void increment(Action a) {
    for (int i = 0; i < inline_size; ++i) {
      if (inline_counts[i].first == a) {
        ++inline_counts[i].second;
        return;
      }
    }
    if (inline_size < 32) {
      inline_counts[inline_size++] = {a, 1};
      return;
    }
    for (auto& kv : overflow) {
      if (kv.first == a) {
        ++kv.second;
        return;
      }
    }
    overflow.push_back({a, 1});
  }
};

// Result of a rollout tracking both terminal utility and the actions taken within the active turn.
struct RolloutTrajectory {
  double terminal_utility = 0.0;
  std::vector<Action> intra_turn_actions;
  bool aborted_by_deadline = false;
  int steps = 0;
  int nn_evals = 0;
};

inline Action PickGreedyAction(const ActionsAndProbs& prior, const std::vector<Action>& legal_actions);

// Simulate a rollout to terminal or truncation cutoff while recording any actions
// taken by search_player in the active agent turn before kActionEndTurn.
inline RolloutTrajectory SimulateRolloutWithTurnTrajectory(
    State* sim_state,
    Player search_player,
    BatchedNNEvaluator* evaluator,
    uint64_t rollout_seed,
    SearchTruncationMode truncation_mode = SearchTruncationMode::kTerminal,
    SearchRolloutCrnMode crn_mode = SearchRolloutCrnMode::kSequential,
    bool sample_opponent_leader_draft = false,
    const std::chrono::steady_clock::time_point* deadline = nullptr,
    int* out_steps = nullptr,
    int* out_nn_evals = nullptr) {
  std::mt19937 rng(rollout_seed);
  int steps = 0;
  const int max_steps = 2500;
  std::vector<Action> intra_turn_actions;
  bool active_turn = true;
  int loc_steps = 0;
  int loc_nn_evals = 0;

  auto apply_action = [&](Action act) {
    sim_state->ApplyAction(act);
    ++loc_steps;
    if (out_steps) ++(*out_steps);
  };
  auto count_nn_eval = [&]() {
    ++loc_nn_evals;
    if (out_nn_evals) ++(*out_nn_evals);
  };

  const auto* cur_dune = dynamic_cast<const dune_imperium::DuneImperiumState*>(sim_state);
  int start_round = cur_dune ? cur_dune->GetCurrentRound() : 1;

  std::array<int, 15> epoch_draw_ordinal{};
  std::array<int, 15> last_deck_size;
  last_deck_size.fill(-1);
  std::array<int, 15> reshuffle_epoch{};
  std::array<bool, 15> just_reshuffled{};
  std::array<int, 15> last_effective_epoch;
  last_effective_epoch.fill(-1);
  std::array<EpochCardDrawCounter, 15> epoch_card_draws{};

  while (!sim_state->IsTerminal() && steps < max_steps) {
    if (deadline != nullptr && std::chrono::steady_clock::now() >= *deadline) {
      return {0.0, std::move(intra_turn_actions), /*aborted_by_deadline=*/true, loc_steps, loc_nn_evals};
    }
    steps++;
    if (sim_state->IsChanceNode()) {
      auto chance_outcomes = sim_state->ChanceOutcomes();
      if (crn_mode == SearchRolloutCrnMode::kSequential) {
        double u = std::generate_canonical<double, 53>(rng);
        Action ca = open_spiel::SampleAction(chance_outcomes, u).first;
        sim_state->ApplyAction(ca);
      } else if (crn_mode == SearchRolloutCrnMode::kDeckOrderCrn) {
        const int s_id = cur_dune ? cur_dune->CurrentChanceStreamId() : 14;
        if (cur_dune != nullptr) {
          const int rem_size = cur_dune->CurrentChanceDeckRemainingSize();
          const bool will_reshuffle = cur_dune->CurrentChanceWillReshuffle();
          const bool is_tracked_deck =
              (s_id == 0 || s_id == 1 || s_id == 2 || (s_id >= 3 && s_id <= 5) ||
               s_id == 8 || (s_id >= 10 && s_id <= 13));
          if (is_tracked_deck) {
            if (will_reshuffle) {
              reshuffle_epoch[s_id]++;
              epoch_draw_ordinal[s_id] = 0;
              just_reshuffled[s_id] = true;
            } else {
              if (!just_reshuffled[s_id] && last_deck_size[s_id] >= 0 &&
                  rem_size > last_deck_size[s_id]) {
                reshuffle_epoch[s_id]++;
                epoch_draw_ordinal[s_id] = 0;
              }
              just_reshuffled[s_id] = false;
            }
            last_deck_size[s_id] = rem_size;
          }
        }
        const uint64_t k = (s_id == 0 && cur_dune != nullptr)
                               ? static_cast<uint64_t>(cur_dune->GetCurrentRound())
                               : ((static_cast<uint64_t>(reshuffle_epoch[s_id]) << 32) |
                                  static_cast<uint64_t>(epoch_draw_ordinal[s_id]++));
        const uint64_t draw_seed = dune_seed::DeriveSeed(
            rollout_seed, 0x43524E00ULL + static_cast<uint64_t>(s_id), k);
        const double u =
            static_cast<double>(draw_seed >> 11) * (1.0 / 9007199254740992.0);
        Action ca = open_spiel::SampleAction(chance_outcomes, u).first;
        sim_state->ApplyAction(ca);
      } else {
        // SearchRolloutCrnMode::kEventKeyedCrn
        if (chance_outcomes.size() == 1) {
          sim_state->ApplyAction(chance_outcomes.front().first);
          continue;
        }
        const int s_id = cur_dune ? cur_dune->CurrentChanceStreamId() : 14;
        const bool is_tracked_deck =
            (s_id == 0 || s_id == 1 || s_id == 2 || (s_id >= 3 && s_id <= 5) ||
             s_id == 8 || (s_id >= 10 && s_id <= 13));
        if (cur_dune != nullptr && is_tracked_deck) {
          const int rem_size = cur_dune->CurrentChanceDeckRemainingSize();
          const bool will_reshuffle = cur_dune->CurrentChanceWillReshuffle();
          if (will_reshuffle) {
            reshuffle_epoch[s_id]++;
            epoch_draw_ordinal[s_id] = 0;
            just_reshuffled[s_id] = true;
          } else {
            if (!just_reshuffled[s_id] && last_deck_size[s_id] >= 0 &&
                rem_size > last_deck_size[s_id]) {
              reshuffle_epoch[s_id]++;
              epoch_draw_ordinal[s_id] = 0;
            }
            just_reshuffled[s_id] = false;
          }
          last_deck_size[s_id] = rem_size;
        }

        int effective_epoch = reshuffle_epoch[s_id];
        if (s_id == 0) {
          effective_epoch = (cur_dune != nullptr)
                                ? ((cur_dune->GetCurrentRound() <= 3)
                                       ? 0
                                       : ((cur_dune->GetCurrentRound() <= 6) ? 1 : 2))
                                : reshuffle_epoch[0];
        } else if (s_id >= 10 && s_id <= 13 && cur_dune != nullptr) {
          const int p = s_id - 10;
          const int engine_reshuffles =
              cur_dune->deck_reshuffle_count(p) +
              (cur_dune->CurrentChanceWillReshuffle() ? 1 : 0);
          effective_epoch = std::max(reshuffle_epoch[s_id], engine_reshuffles);
        }

        if (effective_epoch != last_effective_epoch[s_id]) {
          epoch_card_draws[s_id].clear();
          last_effective_epoch[s_id] = effective_epoch;
        }

        if (is_tracked_deck) {
          const int pool_size =
              (cur_dune != nullptr) ? cur_dune->CurrentChanceSourcePoolSize() : 0;
          Action chosen_action = chance_outcomes.front().first;
          uint64_t min_key = std::numeric_limits<uint64_t>::max();
          for (const auto& [cand_action, prob] : chance_outcomes) {
            const int m =
                (pool_size > 0)
                    ? std::max(1, static_cast<int>(std::llround(
                                      prob * static_cast<double>(pool_size))))
                    : 1;
            const int d = epoch_card_draws[s_id].get(cand_action);
            const int total_copies = d + m;
            const uint64_t StreamEpochTag =
                0x4F534343ULL + (static_cast<uint64_t>(s_id) << 16) +
                static_cast<uint64_t>(effective_epoch);
            std::array<uint64_t, 16> copy_keys;
            const int capped_total = std::min(total_copies, 16);
            const int capped_d = std::min(d, capped_total - 1);
            for (int j = 0; j < capped_total; ++j) {
              copy_keys[j] = dune_seed::DeriveSeed(
                  rollout_seed, StreamEpochTag,
                  (static_cast<uint64_t>(cand_action) << 8) +
                      static_cast<uint64_t>(j));
            }
            std::nth_element(copy_keys.begin(), copy_keys.begin() + capped_d,
                             copy_keys.begin() + capped_total);
            const uint64_t cand_key = copy_keys[capped_d];
            if (cand_key < min_key) {
              min_key = cand_key;
              chosen_action = cand_action;
            }
          }
          epoch_card_draws[s_id].increment(chosen_action);
          apply_action(chosen_action);
        } else {
          // Fall back to kDeckOrderCrn formula for non-deck chance streams (6, 7, 9, 14)
          const uint64_t k = (static_cast<uint64_t>(reshuffle_epoch[s_id]) << 32) |
                             static_cast<uint64_t>(epoch_draw_ordinal[s_id]++);
          const uint64_t draw_seed = dune_seed::DeriveSeed(
              rollout_seed, 0x43524E00ULL + static_cast<uint64_t>(s_id), k);
          const double u =
              static_cast<double>(draw_seed >> 11) * (1.0 / 9007199254740992.0);
          Action ca = open_spiel::SampleAction(chance_outcomes, u).first;
          apply_action(ca);
        }
      }
      continue;
    }

    Player cur = sim_state->CurrentPlayer();
    int cur_round = cur_dune ? cur_dune->GetCurrentRound() : 1;

    // Truncation check: search player's first decision in the subsequent round
    if (truncation_mode == SearchTruncationMode::kNextRoundSearchPlayer &&
        cur == search_player && cur_round > start_round) {
      if (evaluator) count_nn_eval();
      double raw_val = evaluator ? evaluator->EvaluatePlayerValue(*sim_state, search_player) : 0.0;
      return {TruncatedLeafUtility(raw_val), std::move(intra_turn_actions), /*aborted_by_deadline=*/false, loc_steps, loc_nn_evals};
    }

    std::vector<Action> legals = sim_state->LegalActions();
    if (legals.empty()) break;

    Action a = kInvalidAction;
    if (legals.size() == 1) {
      a = legals.front();
    } else if (sample_opponent_leader_draft && cur_dune != nullptr &&
               cur_dune->phase() == dune_imperium::GamePhase::kLeaderDraft &&
               cur != search_player) {
      if (evaluator) count_nn_eval();
      ActionsAndProbs prior = evaluator ? evaluator->PriorWithLegals(*sim_state, legals) : ActionsAndProbs{};
      std::vector<double> prior_probs(legals.size(), 0.0);
      double prior_sum = 0.0;
      if (prior.size() == legals.size()) {
        bool aligned = true;
        for (size_t i = 0; i < legals.size(); ++i) {
          if (prior[i].first != legals[i]) {
            aligned = false;
            break;
          }
          prior_probs[i] = std::max(0.0, prior[i].second);
          prior_sum += prior_probs[i];
        }
        if (!aligned) {
          std::fill(prior_probs.begin(), prior_probs.end(), 0.0);
          prior_sum = 0.0;
          for (size_t i = 0; i < legals.size(); ++i) {
            for (const auto& ap : prior) {
              if (ap.first == legals[i]) {
                prior_probs[i] = std::max(0.0, ap.second);
                prior_sum += prior_probs[i];
                break;
              }
            }
          }
        }
      } else {
        for (size_t i = 0; i < legals.size(); ++i) {
          for (const auto& ap : prior) {
            if (ap.first == legals[i]) {
              prior_probs[i] = std::max(0.0, ap.second);
              prior_sum += prior_probs[i];
              break;
            }
          }
        }
      }
      const double uniform_prob = 1.0 / static_cast<double>(legals.size());
      ActionsAndProbs mixture_outcomes;
      mixture_outcomes.reserve(legals.size());
      for (size_t i = 0; i < legals.size(); ++i) {
        double p_prior = (prior_sum > 1e-12) ? (prior_probs[i] / prior_sum) : uniform_prob;
        double p_mix = 0.5 * p_prior + 0.5 * uniform_prob;
        mixture_outcomes.push_back({legals[i], p_mix});
      }
      uint64_t lead_seed = dune_seed::DeriveSeed(rollout_seed, 0x4C445246ULL, static_cast<uint64_t>(cur));
      double u_lead = static_cast<double>(lead_seed >> 11) * (1.0 / 9007199254740992.0);
      a = open_spiel::SampleAction(mixture_outcomes, u_lead).first;
    } else {
      if (evaluator) count_nn_eval();
      a = evaluator ? evaluator->GreedyActionWithLegals(*sim_state, legals) : legals.front();
    }

    if (active_turn) {
      if (cur == search_player) {
        intra_turn_actions.push_back(a);
        if (a == dune_imperium::kActionEndTurn) {
          active_turn = false;
        }
      } else {
        // Another player acted, turn has completed
        active_turn = false;
      }
    }

    apply_action(a);
  }

  // In kNextRoundSearchPlayer mode, if loop exits without terminal, evaluate value head
  if (truncation_mode == SearchTruncationMode::kNextRoundSearchPlayer && !sim_state->IsTerminal()) {
    if (evaluator) count_nn_eval();
    double raw_val = evaluator ? evaluator->EvaluatePlayerValue(*sim_state, search_player) : 0.0;
    return {TruncatedLeafUtility(raw_val), std::move(intra_turn_actions), /*aborted_by_deadline=*/false, loc_steps, loc_nn_evals};
  }

  // Original fallback: exactly as before for kTerminal mode (and terminal rollouts)
  std::vector<double> ret = sim_state->Returns();
  double u = (static_cast<size_t>(search_player) < ret.size()) ? ret[search_player] : 0.0;
  return {u, std::move(intra_turn_actions), /*aborted_by_deadline=*/false, loc_steps, loc_nn_evals};
}

inline RolloutTrajectory SimulateRemainingTurn(
    State* sim_state,
    Player search_player,
    BatchedNNEvaluator* evaluator,
    uint64_t rollout_seed,
    SearchTruncationMode truncation_mode = SearchTruncationMode::kTerminal,
    SearchRolloutCrnMode crn_mode = SearchRolloutCrnMode::kSequential,
    bool sample_opponent_leader_draft = false,
    const std::chrono::steady_clock::time_point* deadline = nullptr,
    int* out_steps = nullptr,
    int* out_nn_evals = nullptr) {
  return SimulateRolloutWithTurnTrajectory(
      sim_state, search_player, evaluator, rollout_seed, truncation_mode,
      crn_mode, sample_opponent_leader_draft, deadline, out_steps, out_nn_evals);
}

// Pick greedy action from prior over legal actions
inline Action PickGreedyAction(const ActionsAndProbs& prior, const std::vector<Action>& legal_actions) {
  if (legal_actions.empty()) return kInvalidAction;
  if (prior.empty()) return legal_actions.front();
  if (prior.size() == legal_actions.size()) {
    bool aligned = true;
    double best_p = -1.0;
    Action best_a = legal_actions.front();
    for (size_t i = 0; i < legal_actions.size(); ++i) {
      if (prior[i].first != legal_actions[i]) {
        aligned = false;
        break;
      }
      if (prior[i].second > best_p) {
        best_p = prior[i].second;
        best_a = prior[i].first;
      }
    }
    if (aligned) return best_a;
  }
  double best_p = -1.0;
  Action best_a = legal_actions.front();
  for (Action a : legal_actions) {
    for (const auto& ap : prior) {
      if (ap.first == a) {
        if (ap.second > best_p) {
          best_p = ap.second;
          best_a = a;
        }
        break;
      }
    }
  }
  return best_a;
}

// Get top-K candidate actions sorted by prior probability
inline std::vector<Action> GetTopKActions(
    const ActionsAndProbs& prior,
    const std::vector<Action>& legal_actions,
    int k) {
  if (legal_actions.size() <= static_cast<size_t>(k)) {
    return legal_actions;
  }
  std::vector<std::pair<double, Action>> scored;
  scored.reserve(legal_actions.size());
  for (Action a : legal_actions) {
    double p = 0.0;
    for (const auto& ap : prior) {
      if (ap.first == a) {
        p = ap.second;
        break;
      }
    }
    scored.push_back({p, a});
  }
  std::sort(scored.rbegin(), scored.rend());
  std::vector<Action> top_actions;
  top_actions.reserve(k);
  for (int i = 0; i < k; ++i) {
    top_actions.push_back(scored[i].second);
  }
  return top_actions;
}

// Tree reuse strategies
enum class TreeReuseMode {
  kTargetFill,  // Top up candidate visits to reach target (maximizes GPU efficiency)
  kAdditive     // Always run target rollouts and add to existing tree visits (deepens search)
};

// Optional search configuration options for CompoundRolloutSearchDecision
struct CompoundSearchOptions {
  const ActionsAndProbs* precomputed_raw_prior = nullptr;
  bool dedup_paired_worlds = false;
  int stage1_rollouts = 0;
  double stage1_min_mean_diff = -1e9;
};

// Decision result from CompoundRolloutSearch
struct CompoundSearchDecisionResult {
  Action action = kInvalidAction;
  bool overridden = false;
  double raw_mean_utility = 0.0;
  double chosen_mean_utility = 0.0;
  int prior_rollouts_reused = 0;
  int new_rollouts_executed = 0;
  std::vector<std::pair<Action, double>> candidate_utilities = {};
  std::vector<std::pair<Action, std::vector<double>>> candidate_rollout_utilities = {};
  int stage1_planned_worlds = 0;
  int stage1_kept_worlds = 0;
  int stage2_planned_worlds = 0;
  int stage2_kept_worlds = 0;
  double stage1_elapsed_ms = 0.0;
  double stage2_elapsed_ms = 0.0;
  std::string termination_reason = "none";
  double mean_d_stage2 = 0.0;
  double se_d_stage2 = 0.0;
  // R7-55 hybrid search diagnostics
  Action raw_action = kInvalidAction;
  Action best_action = kInvalidAction;
  Action chosen_action = kInvalidAction;
  double best_vs_raw_mean_diff = 0.0;
  double best_vs_raw_t_stat = 0.0;
  int rollout_steps = 0;
  int rollout_nn_evals = 0;
  // R7-58 options and diagnostics
  std::vector<std::pair<Action, double>> candidate_priors = {};
  int resample_fallbacks = 0;
  bool stopped_stage1 = false;
};

using CompoundRolloutSearchResult = CompoundSearchDecisionResult;

// Optional source of search worlds, keyed by the world seed. Returning nullptr
// (or leaving it empty) uses ResampleFromInfostate as before. A sampler must be
// deterministic in its seed so paired candidates share worlds.
using SearchWorldSampler = std::function<std::unique_ptr<State>(uint64_t world_seed)>;

// Intra-turn compound rollout search decision with tree preservation
inline CompoundSearchDecisionResult CompoundRolloutSearchDecision(
    const State& state,
    Player search_player,
    TurnSearchTree* turn_tree,
    const std::vector<std::shared_ptr<BatchedNNEvaluator>>& evaluators,
    int top_k,
    int rollouts_per_action,
    double min_override_margin,
    TreeReuseMode reuse_mode,
    uint64_t decision_seed,
    SearchPairingMode pairing_mode = SearchPairingMode::kUnpaired,
    SearchTruncationMode truncation_mode = SearchTruncationMode::kTerminal,
    SearchRolloutCrnMode crn_mode = SearchRolloutCrnMode::kSequential,
    const SearchWorldSampler& world_sampler = nullptr,
    double min_paired_t_stat = 0.0,
    const CompoundSearchOptions& opts = {}) {
  if (turn_tree != nullptr) turn_tree->BindPublicCardKnowledge(state);
  std::vector<Action> legal_actions = state.LegalActions();
  if (legal_actions.empty()) return {kInvalidAction, false, 0.0, 0.0, 0, 0, {}, {}};
  if (legal_actions.size() == 1) {
    CompoundSearchDecisionResult r;
    r.action = legal_actions.front();
    r.chosen_action = r.action;
    r.raw_action = r.action;
    r.best_action = r.action;
    if (opts.precomputed_raw_prior != nullptr) {
      for (const auto& ap : *opts.precomputed_raw_prior) {
        if (ap.first == r.action) {
          r.candidate_priors.push_back(ap);
          break;
        }
      }
    }
    if (r.candidate_priors.empty()) {
      r.candidate_priors.push_back({r.action, 1.0});
    }
    return r;
  }

  int total_steps = 0;
  int total_nn_evals = 0;
  ActionsAndProbs raw_prior;
  if (opts.precomputed_raw_prior != nullptr) {
    raw_prior = *opts.precomputed_raw_prior;
  } else {
    if (!evaluators.empty()) ++total_nn_evals;
    raw_prior = evaluators.front()->Prior(state);
  }
  Action raw_action = PickGreedyAction(raw_prior, legal_actions);

  std::vector<Action> candidates = GetTopKActions(raw_prior, legal_actions, top_k);
  if (candidates.size() <= 1) {
    CompoundSearchDecisionResult r;
    r.action = raw_action;
    r.chosen_action = raw_action;
    r.raw_action = raw_action;
    r.best_action = raw_action;
    r.rollout_nn_evals = total_nn_evals;
    for (Action a : candidates) {
      double p = 0.0;
      for (const auto& ap : raw_prior) {
        if (ap.first == a) { p = ap.second; break; }
      }
      r.candidate_priors.push_back({a, p});
    }
    return r;
  }

  // Ensure candidate 0 is always the baseline raw_action
  if (candidates.front() != raw_action) {
    auto it = std::find(candidates.begin(), candidates.end(), raw_action);
    if (it != candidates.end()) {
      std::swap(candidates.front(), *it);
    } else {
      candidates.insert(candidates.begin(), raw_action);
      if (candidates.size() > static_cast<size_t>(top_k)) {
        candidates.pop_back();
      }
    }
  }

  std::vector<std::pair<Action, double>> candidate_priors;
  candidate_priors.reserve(candidates.size());
  for (Action cand_a : candidates) {
    double p = 0.0;
    for (const auto& ap : raw_prior) {
      if (ap.first == cand_a) {
        p = ap.second;
        break;
      }
    }
    candidate_priors.push_back({cand_a, p});
  }

  const auto* dune = dynamic_cast<const dune_imperium::DuneImperiumState*>(&state);

  int prior_rollouts_reused = 0;
  for (size_t c_idx = 0; c_idx < candidates.size(); ++c_idx) {
    Action cand_a = candidates[c_idx];
    int existing_visits = (turn_tree != nullptr) ? turn_tree->GetVisits(cand_a) : 0;
    if (turn_tree != nullptr && reuse_mode == TreeReuseMode::kTargetFill) {
      prior_rollouts_reused += std::min(existing_visits, rollouts_per_action);
    } else if (turn_tree != nullptr) {
      prior_rollouts_reused += existing_visits;
    }
  }

  const bool use_two_stage =
      (opts.stage1_rollouts > 0 &&
       opts.stage1_rollouts < rollouts_per_action &&
       pairing_mode == SearchPairingMode::kPaired &&
       prior_rollouts_reused == 0);

  const bool can_dedup =
      (opts.dedup_paired_worlds &&
       pairing_mode == SearchPairingMode::kPaired &&
       !world_sampler &&
       dune != nullptr &&
       prior_rollouts_reused == 0);

  struct RolloutTask {
    size_t c_idx;
    int rollout_idx;
    Action action;
    std::unique_ptr<State> sim_state;
    uint64_t wseed;
  };

  std::atomic<int> resample_fallbacks{0};
  int new_rollouts_executed = 0;

  TurnSearchTree local_tree;
  TurnSearchTree* active_tree = (turn_tree != nullptr) ? turn_tree : &local_tree;

  std::vector<std::vector<double>> per_candidate_rollouts(
      candidates.size(), std::vector<double>(rollouts_per_action, 0.0));
  std::vector<std::vector<double>> cand_rollout_returns(candidates.size());

  const State* root_state_ptr = &state;

  auto build_tasks = [&](const std::vector<size_t>& active_c_indices,
                         int r_begin, int r_end, bool use_dedup) -> std::vector<RolloutTask> {
    std::vector<RolloutTask> batch_tasks;
    if (r_begin >= r_end || active_c_indices.empty()) return batch_tasks;
    const int num_worlds = r_end - r_begin;

    std::vector<std::unique_ptr<State>> base_worlds;
    if (use_dedup) {
      base_worlds.resize(num_worlds);
      for (int w = 0; w < num_worlds; ++w) {
        uint64_t wseed = dune_seed::DeriveSeed(decision_seed, r_begin + w);
        std::mt19937 wrng(wseed);
        auto rng_func = [&wrng]() {
          return std::generate_canonical<double, 53>(wrng);
        };
        std::unique_ptr<State> base_state = dune->ResampleFromInfostate(search_player, rng_func);
        if (!base_state) {
          ++resample_fallbacks;
          base_state = root_state_ptr->Clone();
        }
        base_worlds[w] = std::move(base_state);
      }
    }

    batch_tasks.reserve(active_c_indices.size() * num_worlds);
    for (size_t c_idx : active_c_indices) {
      Action cand_a = candidates[c_idx];
      for (int r = r_begin; r < r_end; ++r) {
        uint64_t wseed = (pairing_mode == SearchPairingMode::kPaired)
            ? dune_seed::DeriveSeed(decision_seed, r)
            : dune_seed::DeriveSeed(decision_seed, c_idx, r);
        std::unique_ptr<State> sim_state = nullptr;
        if (use_dedup) {
          sim_state = base_worlds[r - r_begin]->Clone();
          sim_state->ApplyAction(cand_a);
        } else if (world_sampler || dune == nullptr) {
          if (world_sampler) {
            sim_state = world_sampler(wseed);
            if (sim_state) {
              const auto sim_legal = sim_state->LegalActions();
              if (std::find(sim_legal.begin(), sim_legal.end(), cand_a) == sim_legal.end()) {
                sim_state = nullptr;
              }
            }
          }
          if (!sim_state && dune) {
            std::mt19937 wrng(wseed);
            auto rng_func = [&wrng]() {
              return std::generate_canonical<double, 53>(wrng);
            };
            sim_state = dune->ResampleFromInfostate(search_player, rng_func);
            if (!sim_state) {
              ++resample_fallbacks;
            }
          }
          if (!sim_state) {
            sim_state = root_state_ptr->Clone();
          }
          sim_state->ApplyAction(cand_a);
        }
        batch_tasks.push_back({c_idx, r, cand_a, std::move(sim_state), wseed});
      }
    }
    return batch_tasks;
  };

  auto execute_tasks = [&](std::vector<RolloutTask>& batch_tasks) {
    if (batch_tasks.empty()) return;
    new_rollouts_executed += static_cast<int>(batch_tasks.size());

    std::vector<std::future<RolloutTrajectory>> futures;
    futures.reserve(batch_tasks.size());

    for (size_t i = 0; i < batch_tasks.size(); ++i) {
      auto* active_evaluator = evaluators[i % evaluators.size()].get();
      if (active_evaluator != nullptr) {
        active_evaluator->AdjustActiveRollouts(1);
      }
    }

    for (size_t i = 0; i < batch_tasks.size(); ++i) {
      auto* prebuilt_ptr = batch_tasks[i].sim_state.get();
      uint64_t wseed = batch_tasks[i].wseed;
      Action cand_a = batch_tasks[i].action;
      auto* active_evaluator = evaluators[i % evaluators.size()].get();
      futures.push_back(SearchRolloutThreadPool::Instance().Submit(
          [prebuilt_ptr, root_state_ptr, dune, cand_a, wseed, search_player,
           active_evaluator, truncation_mode, crn_mode, &resample_fallbacks]() {
            ActiveRolloutReleaseGuard guard{active_evaluator};
            std::unique_ptr<State> local_state;
            State* s_ptr = prebuilt_ptr;
            if (s_ptr == nullptr) {
              if (dune != nullptr) {
                std::mt19937 wrng(wseed);
                auto rng_func = [&wrng]() {
                  return std::generate_canonical<double, 53>(wrng);
                };
                local_state = dune->ResampleFromInfostate(search_player, rng_func);
              }
              if (!local_state) {
                ++resample_fallbacks;
                local_state = root_state_ptr->Clone();
              }
              local_state->ApplyAction(cand_a);
              s_ptr = local_state.get();
            }
            return SimulateRolloutWithTurnTrajectory(
                s_ptr, search_player, active_evaluator, wseed + 999, truncation_mode, crn_mode);
          }));
    }

    for (size_t i = 0; i < futures.size(); ++i) {
      RolloutTrajectory traj = futures[i].get();
      total_steps += (traj.steps + 1);
      total_nn_evals += traj.nn_evals;
      active_tree->RecordRollout(batch_tasks[i].action, traj.intra_turn_actions, traj.terminal_utility);
      if (batch_tasks[i].c_idx < per_candidate_rollouts.size() &&
          batch_tasks[i].rollout_idx >= 0 &&
          static_cast<size_t>(batch_tasks[i].rollout_idx) < per_candidate_rollouts[batch_tasks[i].c_idx].size()) {
        per_candidate_rollouts[batch_tasks[i].c_idx][batch_tasks[i].rollout_idx] = traj.terminal_utility;
      }
      if (batch_tasks[i].c_idx < cand_rollout_returns.size()) {
        cand_rollout_returns[batch_tasks[i].c_idx].push_back(traj.terminal_utility);
      }
    }
  };

  bool stopped_stage1 = false;
  size_t best_s1_c = 0;
  if (use_two_stage) {
    const int R1 = opts.stage1_rollouts;
    std::vector<size_t> all_c_indices(candidates.size());
    std::iota(all_c_indices.begin(), all_c_indices.end(), 0);
    auto s1_tasks = build_tasks(all_c_indices, 0, R1, can_dedup);
    execute_tasks(s1_tasks);

    best_s1_c = 1;
    double best_s1_mean_d = -1e9;
    for (size_t c_idx = 1; c_idx < candidates.size(); ++c_idx) {
      double sum_d = 0.0;
      for (int r = 0; r < R1; ++r) {
        sum_d += (per_candidate_rollouts[c_idx][r] - per_candidate_rollouts[0][r]);
      }
      double mean_d = sum_d / static_cast<double>(R1);
      if (c_idx == 1 || mean_d > best_s1_mean_d) {
        best_s1_mean_d = mean_d;
        best_s1_c = c_idx;
      }
    }

    if (best_s1_c == 0 || best_s1_mean_d < opts.stage1_min_mean_diff) {
      stopped_stage1 = true;
    } else {
      std::vector<size_t> stage2_c_indices = {0, best_s1_c};
      auto s2_tasks = build_tasks(stage2_c_indices, R1, rollouts_per_action, can_dedup);
      execute_tasks(s2_tasks);
    }

    for (size_t c_idx = 0; c_idx < candidates.size(); ++c_idx) {
      per_candidate_rollouts[c_idx].resize(cand_rollout_returns[c_idx].size());
    }
  } else if (prior_rollouts_reused == 0) {
    std::vector<size_t> all_c_indices(candidates.size());
    std::iota(all_c_indices.begin(), all_c_indices.end(), 0);
    auto tasks = build_tasks(all_c_indices, 0, rollouts_per_action, can_dedup);
    execute_tasks(tasks);
  } else {
    std::vector<RolloutTask> tasks;
    for (size_t c_idx = 0; c_idx < candidates.size(); ++c_idx) {
      Action cand_a = candidates[c_idx];
      int existing_visits = (turn_tree != nullptr) ? turn_tree->GetVisits(cand_a) : 0;
      int needed_rollouts = rollouts_per_action;
      if (turn_tree != nullptr && reuse_mode == TreeReuseMode::kTargetFill) {
        needed_rollouts = std::max(0, rollouts_per_action - existing_visits);
      }
      auto c_tasks = build_tasks({c_idx}, 0, needed_rollouts, /*use_dedup=*/false);
      tasks.insert(tasks.end(), std::make_move_iterator(c_tasks.begin()),
                   std::make_move_iterator(c_tasks.end()));
    }
    execute_tasks(tasks);
  }

  double raw_mean_u = active_tree->GetMeanUtility(raw_action);
  Action best_action = raw_action;
  double best_mean_u = raw_mean_u;
  size_t best_c_idx = 0;
  double best_vs_raw_mean_diff = 0.0;
  double best_vs_raw_t_stat = 0.0;
  bool overridden = false;

  if (use_two_stage) {
    if (!stopped_stage1) {
      double cand_mean_u = active_tree->GetMeanUtility(candidates[best_s1_c]);
      if (cand_mean_u > raw_mean_u) {
        best_action = candidates[best_s1_c];
        best_mean_u = cand_mean_u;
        best_c_idx = best_s1_c;
        best_vs_raw_mean_diff = best_mean_u - raw_mean_u;

        const auto& raw_returns = cand_rollout_returns[0];
        const auto& best_returns = cand_rollout_returns[best_c_idx];
        const size_t R = std::min(raw_returns.size(), best_returns.size());
        if (R >= 2) {
          double sum_d = 0.0;
          for (size_t r = 0; r < R; ++r) {
            sum_d += (best_returns[r] - raw_returns[r]);
          }
          double mean_d = sum_d / static_cast<double>(R);
          double sum_sq_d = 0.0;
          for (size_t r = 0; r < R; ++r) {
            double diff = (best_returns[r] - raw_returns[r]) - mean_d;
            sum_sq_d += diff * diff;
          }
          double var_d = sum_sq_d / static_cast<double>(R - 1);
          double se_d = std::sqrt(var_d / static_cast<double>(R));
          best_vs_raw_t_stat = (se_d > 1e-12) ? (mean_d / se_d) : (mean_d > 0.0 ? 1e9 : -1e9);
        } else {
          best_vs_raw_t_stat = (min_paired_t_stat <= 0.0) ? 1e9 : 0.0;
        }

        if (best_mean_u > raw_mean_u + min_override_margin &&
            best_vs_raw_t_stat >= min_paired_t_stat) {
          overridden = true;
        }
      }
    } else {
      double cand_mean_u = active_tree->GetMeanUtility(candidates[best_s1_c]);
      if (cand_mean_u > raw_mean_u) {
        best_action = candidates[best_s1_c];
        best_mean_u = cand_mean_u;
        best_c_idx = best_s1_c;
        best_vs_raw_mean_diff = best_mean_u - raw_mean_u;

        const auto& raw_returns = cand_rollout_returns[0];
        const auto& best_returns = cand_rollout_returns[best_c_idx];
        const size_t R = std::min(raw_returns.size(), best_returns.size());
        if (R >= 2) {
          double sum_d = 0.0;
          for (size_t r = 0; r < R; ++r) {
            sum_d += (best_returns[r] - raw_returns[r]);
          }
          double mean_d = sum_d / static_cast<double>(R);
          double sum_sq_d = 0.0;
          for (size_t r = 0; r < R; ++r) {
            double diff = (best_returns[r] - raw_returns[r]) - mean_d;
            sum_sq_d += diff * diff;
          }
          double var_d = sum_sq_d / static_cast<double>(R - 1);
          double se_d = std::sqrt(var_d / static_cast<double>(R));
          best_vs_raw_t_stat = (se_d > 1e-12) ? (mean_d / se_d) : (mean_d > 0.0 ? 1e9 : -1e9);
        }
      }
      overridden = false;
    }
  } else {
    for (size_t c_idx = 1; c_idx < candidates.size(); ++c_idx) {
      Action cand_a = candidates[c_idx];
      double mean_u = active_tree->GetMeanUtility(cand_a);
      if (mean_u > best_mean_u) {
        best_mean_u = mean_u;
        best_action = cand_a;
        best_c_idx = c_idx;
      }
    }

    if (best_action != raw_action) {
      best_vs_raw_mean_diff = best_mean_u - raw_mean_u;
      const auto& raw_returns = cand_rollout_returns[0];
      const auto& best_returns = cand_rollout_returns[best_c_idx];
      const size_t R = std::min(raw_returns.size(), best_returns.size());

      if (pairing_mode == SearchPairingMode::kPaired && R >= 2) {
        double sum_d = 0.0;
        for (size_t r = 0; r < R; ++r) {
          sum_d += (best_returns[r] - raw_returns[r]);
        }
        double mean_d = sum_d / static_cast<double>(R);
        double sum_sq_d = 0.0;
        for (size_t r = 0; r < R; ++r) {
          double diff = (best_returns[r] - raw_returns[r]) - mean_d;
          sum_sq_d += diff * diff;
        }
        double var_d = sum_sq_d / static_cast<double>(R - 1);
        double se_d = std::sqrt(var_d / static_cast<double>(R));
        best_vs_raw_t_stat = (se_d > 1e-12) ? (mean_d / se_d) : (mean_d > 0.0 ? 1e9 : -1e9);
      } else if (raw_returns.size() >= 2 && best_returns.size() >= 2) {
        double m0 = 0.0, m1 = 0.0;
        for (double v : raw_returns) m0 += v;
        m0 /= raw_returns.size();
        for (double v : best_returns) m1 += v;
        m1 /= best_returns.size();
        double v0 = 0.0, v1 = 0.0;
        for (double v : raw_returns) v0 += (v - m0) * (v - m0);
        v0 /= (raw_returns.size() - 1);
        for (double v : best_returns) v1 += (v - m1) * (v - m1);
        v1 /= (best_returns.size() - 1);
        double se = std::sqrt(v0 / raw_returns.size() + v1 / best_returns.size());
        best_vs_raw_t_stat = (se > 1e-12) ? ((m1 - m0) / se) : ((m1 - m0) > 0.0 ? 1e9 : -1e9);
      } else {
        best_vs_raw_t_stat = (min_paired_t_stat <= 0.0) ? 1e9 : 0.0;
      }
    }

    if (best_action != raw_action &&
        best_mean_u > raw_mean_u + min_override_margin &&
        best_vs_raw_t_stat >= min_paired_t_stat) {
      overridden = true;
    }
  }

  std::vector<std::pair<Action, double>> candidate_utilities;
  candidate_utilities.reserve(candidates.size());
  for (Action cand_a : candidates) {
    double mean_u = active_tree->GetMeanUtility(cand_a);
    candidate_utilities.push_back({cand_a, mean_u});
  }

  std::vector<std::pair<Action, std::vector<double>>> candidate_rollout_utilities;
  candidate_rollout_utilities.reserve(candidates.size());
  for (size_t c_idx = 0; c_idx < candidates.size(); ++c_idx) {
    candidate_rollout_utilities.push_back({candidates[c_idx], per_candidate_rollouts[c_idx]});
  }

  CompoundSearchDecisionResult result;
  result.action = overridden ? best_action : raw_action;
  result.chosen_action = result.action;
  result.best_action = best_action;
  result.raw_action = raw_action;
  result.overridden = overridden;
  result.stopped_stage1 = stopped_stage1;
  result.resample_fallbacks = resample_fallbacks.load();
  result.raw_mean_utility = raw_mean_u;
  result.chosen_mean_utility = overridden ? best_mean_u : raw_mean_u;
  result.prior_rollouts_reused = prior_rollouts_reused;
  result.new_rollouts_executed = new_rollouts_executed;
  result.candidate_priors = std::move(candidate_priors);
  result.candidate_utilities = std::move(candidate_utilities);
  result.candidate_rollout_utilities = std::move(candidate_rollout_utilities);
  result.best_vs_raw_mean_diff = best_vs_raw_mean_diff;
  result.best_vs_raw_t_stat = best_vs_raw_t_stat;
  result.rollout_steps = total_steps;
  result.rollout_nn_evals = total_nn_evals;

  return result;
}

inline CompoundSearchDecisionResult CompoundRolloutSearchDecision(
    const State& state,
    Player search_player,
    TurnSearchTree* turn_tree,
    const std::vector<std::shared_ptr<BatchedNNEvaluator>>& evaluators,
    int top_k,
    int rollouts_per_action,
    double min_override_margin,
    TreeReuseMode reuse_mode,
    uint64_t decision_seed,
    SearchPairingMode pairing_mode,
    SearchTruncationMode truncation_mode,
    bool /*full_turn_supervision*/,
    double min_paired_t_stat = 0.0) {
  return CompoundRolloutSearchDecision(
      state, search_player, turn_tree, evaluators, top_k, rollouts_per_action,
      min_override_margin, reuse_mode, decision_seed, pairing_mode,
      truncation_mode, SearchRolloutCrnMode::kSequential, nullptr, min_paired_t_stat);
}

// Compute softmax target distribution over evaluated actions at temperature T
inline std::vector<std::pair<Action, double>> ComputeSoftmaxTarget(
    const std::vector<std::pair<Action, double>>& action_utilities,
    const std::vector<Action>& legal_actions,
    double temperature) {
  std::vector<std::pair<Action, double>> probs;
  if (action_utilities.empty() || legal_actions.empty()) return probs;

  double max_u = -1e9;
  for (const auto& au : action_utilities) {
    if (au.second > max_u) max_u = au.second;
  }

  double sum_exp = 0.0;
  std::unordered_map<Action, double> exp_map;
  for (const auto& au : action_utilities) {
    double e = std::exp((au.second - max_u) / temperature);
    exp_map[au.first] = e;
    sum_exp += e;
  }

  probs.reserve(legal_actions.size());
  for (Action a : legal_actions) {
    auto it = exp_map.find(a);
    double p = (it != exp_map.end() && sum_exp > 0.0) ? (it->second / sum_exp) : 0.0;
    probs.push_back({a, p});
  }
  return probs;
}

// Blend blueprint prior mu and search target q_search: (1 - eta) * mu + eta * q_search
inline std::vector<std::pair<Action, double>> BlendTargets(
    const std::vector<std::pair<Action, double>>& blueprint_prior,
    const std::vector<std::pair<Action, double>>& search_target,
    double eta) {
  std::unordered_map<Action, double> q_map;
  for (const auto& st : search_target) q_map[st.first] = st.second;

  std::vector<std::pair<Action, double>> blended;
  blended.reserve(blueprint_prior.size());
  for (const auto& bp : blueprint_prior) {
    Action a = bp.first;
    double mu = bp.second;
    auto it = q_map.find(a);
    double q = (it != q_map.end()) ? it->second : 0.0;
    blended.push_back({a, (1.0 - eta) * mu + eta * q});
  }
  return blended;
}

// Override-gated target: (1 - eta) * prior + eta * onehot(best_candidate)
inline std::vector<double> ComputeOverrideGatedTarget(
    const std::vector<std::pair<Action, double>>& raw_policy,
    const std::vector<Action>& legal_actions,
    Action best_candidate,
    double override_eta) {
  std::vector<double> target(legal_actions.size(), 0.0);
  double prior_sum = 0.0;
  for (size_t i = 0; i < legal_actions.size(); ++i) {
    Action a = legal_actions[i];
    double p = 0.0;
    for (const auto& ap : raw_policy) {
      if (ap.first == a) { p = ap.second; break; }
    }
    target[i] = p;
    prior_sum += p;
  }
  if (prior_sum > 1e-12) {
    for (double& p : target) p /= prior_sum;
  } else {
    std::fill(target.begin(), target.end(), 1.0 / legal_actions.size());
  }

  for (size_t i = 0; i < legal_actions.size(); ++i) {
    Action a = legal_actions[i];
    double onehot = (a == best_candidate) ? 1.0 : 0.0;
    target[i] = (1.0 - override_eta) * target[i] + override_eta * onehot;
  }
  return target;
}

// Check if state_after gained new private information (card drawn or intrigue drawn)
// compared to state_before for player.
inline bool HasNewInformation(
    const DuneImperiumState& state_before,
    const DuneImperiumState& state_after,
    Player player) {
  // 1. Intrigue card drawn
  const auto& int_before = state_before.GetIntrigueHandForTesting(player);
  const auto& int_after = state_after.GetIntrigueHandForTesting(player);
  if (int_after.size() > int_before.size()) return true;

  // 2. Card drawn from deck
  const auto& draw_before = state_before.GetPlayerDrawDeckForTesting(player);
  const auto& draw_after = state_after.GetPlayerDrawDeckForTesting(player);
  if (draw_after.size() != draw_before.size()) return true;

  return false;
}

// Check for high-stakes decision points: Agent Placement, Combat Deployment, Reveal Purchase
inline bool IsHighStakesStrategicState(const State& state, Player searched_player) {
  if (state.CurrentPlayer() != searched_player) return false;
  std::vector<Action> legal_actions = state.LegalActions();
  if (legal_actions.size() < 2) return false;

  const auto* dune_state = dynamic_cast<const dune_imperium::DuneImperiumState*>(&state);
  if (!dune_state) return false;

  for (Action a : legal_actions) {
    std::string s = dune_state->ActionToString(searched_player, a);
    if (s.rfind("PlaceAgent", 0) == 0 ||
        s.rfind("SelectAgentCard", 0) == 0 ||
        s.rfind("Deploy ", 0) == 0 ||
        s.rfind("CombatCommit", 0) == 0 ||
        s.rfind("Buy", 0) == 0) {
      return true;
    }
  }
  return false;
}

inline bool IsPurchaseDecisionState(const State& state) {
  const std::vector<Action> legal_actions = state.LegalActions();
  if (legal_actions.size() < 2) return false;
  for (Action a : legal_actions) {
    if ((a >= kActionBuyImperiumRow0 && a < kActionBuyImperiumRow0 + 8) ||
        (a >= kActionBuyHelenaReserve0 && a < kActionBuyHelenaReserve0 + 8) ||
        (a == kActionBuyReserveArrakisLiaison) ||
        (a == kActionBuyReserveTheSpiceMustFlow) ||
        (a >= kActionTechAcquire0 && a <= kActionTechAcquireSkip) ||
        (a >= kActionTleilaxuAcquire0 && a <= kActionTleilaxuAcquireSkip) ||
        (a >= kActionTechAcquireWithSolari0 && a <= kActionTechAcquireWithSolariSkip)) {
      return true;
    }
  }
  return false;
}

// Owned immutable snapshot of a pre-action decision visited by the learner
struct SearchSnapshot {
  std::unique_ptr<State> sim_state;
  Player player = kInvalidPlayer;
  std::vector<float> observation;
  std::vector<Action> legal_actions;
  std::vector<std::pair<Action, double>> raw_policy;
  Action chosen_action = kInvalidAction;
  float behavior_log_prob = 0.0f;
  uint64_t root_id = 0;
  // Candidate action descriptors for the deployed semantic action scorer.
  dune_semantic::CandidateActionData candidate_data;
};

// Audited Fallback-Aligned Target Rule:
// q = (1 - eta) * mu + eta * q_search if overridden or if D(q_search) == a_mu;
// q = mu if not overridden and D(q_search) != a_mu.
inline std::vector<double> ComputeFallbackAlignedTarget(
    const std::vector<std::pair<Action, double>>& raw_policy,
    const std::vector<Action>& legal_actions,
    const CompoundSearchDecisionResult& search_res,
    Action raw_action,
    double softmax_temperature,
    double eta_blend,
    bool* fell_back_to_mu) {
  auto q_search_probs = ComputeSoftmaxTarget(search_res.candidate_utilities, legal_actions, softmax_temperature);

  Action search_argmax = raw_action;
  double max_q = -1.0;
  for (const auto& qp : q_search_probs) {
    if (qp.second > max_q) {
      max_q = qp.second;
      search_argmax = qp.first;
    }
  }

  bool fallback = (!search_res.overridden) && (search_argmax != raw_action);
  if (fell_back_to_mu != nullptr) {
    *fell_back_to_mu = fallback;
  }

  std::vector<double> target_distribution(legal_actions.size(), 0.0);
  if (fallback) {
    for (size_t i = 0; i < legal_actions.size(); ++i) {
      Action a = legal_actions[i];
      for (const auto& ap : raw_policy) {
        if (ap.first == a) {
          target_distribution[i] = ap.second;
          break;
        }
      }
    }
  } else {
    auto blended = BlendTargets(raw_policy, q_search_probs, eta_blend);
    for (size_t i = 0; i < legal_actions.size(); ++i) {
      Action a = legal_actions[i];
      for (const auto& ap : blended) {
        if (ap.first == a) {
          target_distribution[i] = ap.second;
          break;
        }
      }
    }
  }
  return target_distribution;
}

// MPO-Style Policy-Relative Reweighting with Temperature Floor and KL Ceiling:
// Outside candidate set C: q(a | s) = pi_ref(a | s).
// Inside candidate set C: q(a | s) = P_C * (pi_ref(a | s) * exp((Q(s, a) - Q_max) / tau)) / sum_{c in C} (pi_ref(c | s) * exp((Q(s, c) - Q_max) / tau)),
// where P_C = sum_{c in C} pi_ref(c | s).
// Base temperature tau_base acts as a floor. If KL(q || pi_ref) <= max_kl, tau_base is kept.
// If KL(q || pi_ref) > max_kl, tau is increased via bisection until KL(q || pi_ref) <= max_kl.
inline std::vector<double> ComputeMpoRelativeTarget(
    const std::vector<std::pair<Action, double>>& raw_policy,
    const std::vector<Action>& legal_actions,
    const std::vector<std::pair<Action, double>>& candidate_utilities,
    double base_temperature = 0.50,
    double max_target_kl = 0.05,
    double* out_realized_kl = nullptr,
    double* out_realized_temperature = nullptr) {
  std::unordered_map<Action, double> pi_map;
  for (const auto& ap : raw_policy) {
    pi_map[ap.first] = std::max(0.0, ap.second);
  }

  std::vector<double> fallback_dist(legal_actions.size(), 0.0);
  for (size_t i = 0; i < legal_actions.size(); ++i) {
    auto it = pi_map.find(legal_actions[i]);
    fallback_dist[i] = (it != pi_map.end()) ? it->second : 0.0;
  }

  if (candidate_utilities.size() <= 1 || legal_actions.empty()) {
    if (out_realized_kl) *out_realized_kl = 0.0;
    if (out_realized_temperature) *out_realized_temperature = base_temperature;
    return fallback_dist;
  }

  std::vector<std::pair<Action, double>> valid_cands;
  double p_c = 0.0;
  double q_max = -1e9;
  for (const auto& cu : candidate_utilities) {
    auto it = pi_map.find(cu.first);
    if (it != pi_map.end() && it->second > 0.0) {
      valid_cands.push_back(cu);
      p_c += it->second;
      if (cu.second > q_max) q_max = cu.second;
    }
  }

  if (valid_cands.size() <= 1 || p_c <= 1e-12) {
    if (out_realized_kl) *out_realized_kl = 0.0;
    if (out_realized_temperature) *out_realized_temperature = base_temperature;
    return fallback_dist;
  }

  auto eval_cand_dist_and_kl = [&](double tau, std::vector<double>& cand_q) -> double {
    double sum_w = 0.0;
    std::vector<double> weights(valid_cands.size(), 0.0);
    for (size_t i = 0; i < valid_cands.size(); ++i) {
      Action a = valid_cands[i].first;
      double pi_a = pi_map[a];
      double u = valid_cands[i].second;
      double w = pi_a * std::exp((u - q_max) / tau);
      weights[i] = w;
      sum_w += w;
    }
    if (sum_w <= 1e-12) {
      for (size_t i = 0; i < valid_cands.size(); ++i) {
        cand_q[i] = pi_map[valid_cands[i].first];
      }
      return 0.0;
    }
    double kl = 0.0;
    for (size_t i = 0; i < valid_cands.size(); ++i) {
      Action a = valid_cands[i].first;
      double pi_a = pi_map[a];
      double q_a = p_c * (weights[i] / sum_w);
      cand_q[i] = q_a;
      if (q_a > 1e-15 && pi_a > 1e-15) {
        kl += q_a * std::log(q_a / pi_a);
      }
    }
    return kl;
  };

  double tau = std::max(1e-4, base_temperature);
  std::vector<double> cand_q(valid_cands.size(), 0.0);
  double current_kl = eval_cand_dist_and_kl(tau, cand_q);

  if (current_kl > max_target_kl) {
    double tau_low = tau;
    double tau_high = 1000.0;
    for (int iter = 0; iter < 25; ++iter) {
      double tau_mid = 0.5 * (tau_low + tau_high);
      double kl_mid = eval_cand_dist_and_kl(tau_mid, cand_q);
      if (kl_mid > max_target_kl) {
        tau_low = tau_mid;
      } else {
        tau_high = tau_mid;
      }
    }
    tau = tau_high;
    current_kl = eval_cand_dist_and_kl(tau, cand_q);
  }

  if (out_realized_kl) *out_realized_kl = current_kl;
  if (out_realized_temperature) *out_realized_temperature = tau;

  std::unordered_map<Action, double> cand_q_map;
  for (size_t i = 0; i < valid_cands.size(); ++i) {
    cand_q_map[valid_cands[i].first] = cand_q[i];
  }

  std::vector<double> target_distribution(legal_actions.size(), 0.0);
  for (size_t i = 0; i < legal_actions.size(); ++i) {
    Action a = legal_actions[i];
    auto it = cand_q_map.find(a);
    if (it != cand_q_map.end()) {
      target_distribution[i] = it->second;
    } else {
      auto pit = pi_map.find(a);
      target_distribution[i] = (pit != pi_map.end()) ? pit->second : 0.0;
    }
  }

  return target_distribution;
}

inline std::vector<double> ComputePairwiseMpoTarget(
    const std::vector<std::pair<Action, double>>& raw_policy,
    const std::vector<Action>& legal_actions,
    Action raw_action, double raw_q,
    Action best_action, double best_q,
    double base_temperature = 0.15,
    double max_target_kl = 0.05) {
  std::vector<std::pair<Action, double>> pair_utilities = {
      {raw_action, raw_q},
      {best_action, best_q},
  };
  return ComputeMpoRelativeTarget(
      raw_policy, legal_actions, pair_utilities,
      base_temperature, max_target_kl);
}

inline std::vector<float> ExtractObservationTensor(
    const State& state,
    Player player,
    dune_imperium::MarketAppendixMode mode = dune_imperium::MarketAppendixMode::kFullPublicInformationV3,
    dune_imperium::DeckPoolSemantics deck_pool_semantics = dune_imperium::DeckPoolSemantics::kLegacy) {
  const auto* dune = dynamic_cast<const dune_imperium::DuneImperiumState*>(&state);
  if (dune) {
    return dune->InformationStateTensorWithAppendix(player, mode, deck_pool_semantics);
  }
  return state.InformationStateTensor(player);
}

struct SearchSupervisionResult {
  std::vector<open_spiel::SearchTrainingExample> examples;
  std::vector<open_spiel::SearchTrainingExample> agreement_examples;
  int roots_evaluated = 0;
  int overrides = 0;
  int fallbacks_to_mu = 0;
  double elapsed_seconds = 0.0;
  double mean_override_rate = 0.0;
  int total_decisions_evaluated = 0;
  double mean_target_prob_search_move = 0.0;
};

inline SearchSupervisionResult BatchCompoundSearchSupervision(
    std::vector<SearchSnapshot>& snapshots,
    const std::vector<std::shared_ptr<BatchedNNEvaluator>>& evaluators,
    int top_k = 3,
    int rollouts_per_action = 64,
    double min_override_margin = 0.15,
    double softmax_temperature = 0.50,
    double eta_blend = 0.80,
    uint64_t search_seed = 0,
    int num_workers = 1,
    bool full_turn_supervision = true,
    bool use_mpo_target = true,
    double max_target_kl = 0.05,
    SearchPairingMode pairing_mode = SearchPairingMode::kUnpaired,
    SearchTruncationMode truncation_mode = SearchTruncationMode::kTerminal,
    const std::string& target_mode = "mpo",
    double override_eta = 0.50) {
  auto start_time = std::chrono::steady_clock::now();
  SearchSupervisionResult result;
  if (snapshots.empty() || evaluators.empty()) return result;

  struct TurnResult {
    bool valid = false;
    int overrides = 0;
    int fallbacks = 0;
    int rollouts = 0;
    int decisions_evaluated = 0;
    double target_prob_search_move_sum = 0.0;
    std::vector<open_spiel::SearchTrainingExample> examples;
    std::vector<open_spiel::SearchTrainingExample> agreement_examples;
  };
  std::vector<TurnResult> turn_results(snapshots.size());

  auto process_root = [&](size_t r_idx) {
    auto& snap = snapshots[r_idx];
    if (!snap.sim_state) return;

    uint64_t root_seed = dune_seed::DeriveSeed(search_seed, r_idx);
    TurnResult& tr = turn_results[r_idx];
    tr.valid = true;

    if (!full_turn_supervision) {
      // Single-decision supervision at the root state
      auto current_legals = snap.sim_state->LegalActions();
      SPIEL_CHECK_EQ(current_legals.size(), snap.legal_actions.size());
      for (size_t i = 0; i < current_legals.size(); ++i) {
        SPIEL_CHECK_EQ(current_legals[i], snap.legal_actions[i]);
      }

      CompoundSearchDecisionResult search_res = CompoundRolloutSearchDecision(
          *snap.sim_state,
          snap.player,
          /*turn_tree=*/nullptr,
          evaluators,
          top_k,
          rollouts_per_action,
          min_override_margin,
          TreeReuseMode::kTargetFill,
          root_seed,
          pairing_mode,
          truncation_mode);

      std::vector<std::pair<Action, double>> mu;
      if (!use_mpo_target && !snap.raw_policy.empty()) {
        mu = snap.raw_policy;
      } else if (!evaluators.empty()) {
        ActionsAndProbs raw_prior = evaluators.front()->Prior(*snap.sim_state);
        mu.reserve(snap.legal_actions.size());
        for (Action a : snap.legal_actions) {
          double p = 0.0;
          for (const auto& ap : raw_prior) {
            if (ap.first == a) { p = ap.second; break; }
          }
          mu.push_back({a, p});
        }
      } else {
        mu = snap.raw_policy;
      }
      Action raw_action = PickGreedyAction(mu, snap.legal_actions);

      bool fell_back = false;
      std::vector<double> target_dist;
      bool keep_example = true;
      double target_prob_search_move = 0.0;

      if (target_mode == "override_gated") {
        if (search_res.overridden) {
          target_dist = ComputeOverrideGatedTarget(
              mu, snap.legal_actions, search_res.action, override_eta);
          for (size_t i = 0; i < snap.legal_actions.size(); ++i) {
            if (snap.legal_actions[i] == search_res.action) {
              target_prob_search_move = target_dist[i];
              break;
            }
          }
        } else {
          // Drop non-override states from auxiliary training set; collect as agreement state
          keep_example = false;
          open_spiel::SearchTrainingExample agr_ex;
          agr_ex.observation = snap.observation;
          agr_ex.player = snap.player;
          agr_ex.legal_actions = snap.legal_actions;
          agr_ex.candidate_data = snap.candidate_data;
          tr.agreement_examples.push_back(std::move(agr_ex));
        }
      } else if (use_mpo_target) {
        double r_kl = 0.0;
        double r_tau = softmax_temperature;
        target_dist = ComputeMpoRelativeTarget(
            mu, snap.legal_actions, search_res.candidate_utilities,
            softmax_temperature, max_target_kl, &r_kl, &r_tau);
        for (size_t i = 0; i < snap.legal_actions.size(); ++i) {
          if (snap.legal_actions[i] == search_res.action) {
            target_prob_search_move = target_dist[i];
            break;
          }
        }
      } else {
        target_dist = ComputeFallbackAlignedTarget(
            mu, snap.legal_actions, search_res, raw_action,
            softmax_temperature, eta_blend, &fell_back);
        for (size_t i = 0; i < snap.legal_actions.size(); ++i) {
          if (snap.legal_actions[i] == search_res.action) {
            target_prob_search_move = target_dist[i];
            break;
          }
        }
      }

      tr.decisions_evaluated = 1;
      tr.rollouts = search_res.new_rollouts_executed;
      if (search_res.overridden) tr.overrides = 1;
      if (fell_back) tr.fallbacks = 1;

      if (keep_example) {
        open_spiel::SearchTrainingExample ex;
        ex.observation = snap.observation;
        ex.player = snap.player;
        ex.legal_actions = snap.legal_actions;
        ex.normalized_visits = std::move(target_dist);
        ex.value_target = 0.0;
        ex.value_target_attached = false;
        ex.simulations_completed = search_res.new_rollouts_executed;
        ex.candidate_data = snap.candidate_data;
        ex.raw_action = raw_action;
        ex.best_action = search_res.action;
        if (snap.sim_state) {
          ex.state_snapshot = snap.sim_state->Clone();
        }

        tr.target_prob_search_move_sum += target_prob_search_move;
        tr.examples.push_back(std::move(ex));
      }
      return;
    }

    // Full-turn supervision: search takes over from turn start to kActionEndTurn
    TurnSearchTree turn_tree;
    int intra_step = 0;
    const int max_turn_steps = 20;

    // Use a persistent per-root RNG across consecutive chance outcomes
    uint64_t chance_root_seed = dune_seed::DeriveSeed(root_seed, /*stream=*/0x4348414EULL);
    std::mt19937_64 chance_rng(chance_root_seed);

    while (!snap.sim_state->IsTerminal() &&
           (snap.sim_state->CurrentPlayer() == snap.player || snap.sim_state->IsChanceNode()) &&
           intra_step < max_turn_steps) {
      if (snap.sim_state->IsChanceNode()) {
        auto chance_outcomes = snap.sim_state->ChanceOutcomes();
        double u = std::generate_canonical<double, 53>(chance_rng);
        Action ca = open_spiel::SampleAction(chance_outcomes, u).first;
        snap.sim_state->ApplyAction(ca);
        // Random chance event (card draw/market refill) invalidates cached search tree
        turn_tree.Reset();
        continue;
      }

      std::vector<Action> current_legals = snap.sim_state->LegalActions();
      if (current_legals.empty()) break;

      if (current_legals.size() == 1) {
        Action forced_a = current_legals.front();
        turn_tree.Advance(forced_a);
        const auto* dune_before = dynamic_cast<const dune_imperium::DuneImperiumState*>(snap.sim_state.get());
        std::unique_ptr<dune_imperium::DuneImperiumState> dune_snap_before =
            dune_before ? std::unique_ptr<dune_imperium::DuneImperiumState>(
                              dynamic_cast<dune_imperium::DuneImperiumState*>(dune_before->Clone().release()))
                        : nullptr;
        snap.sim_state->ApplyAction(forced_a);
        if (dune_snap_before) {
          const auto* dune_after = dynamic_cast<const dune_imperium::DuneImperiumState*>(snap.sim_state.get());
          if (dune_after && HasNewInformation(*dune_snap_before, *dune_after, snap.player)) {
            turn_tree.Reset();
          }
        }
        if (forced_a == dune_imperium::kActionEndTurn) {
          break;
        }
        continue;
      }

      uint64_t step_seed = dune_seed::DeriveSeed(root_seed, intra_step);
      intra_step++;

      CompoundSearchDecisionResult search_res = CompoundRolloutSearchDecision(
          *snap.sim_state,
          snap.player,
          &turn_tree,
          evaluators,
          top_k,
          rollouts_per_action,
          min_override_margin,
          TreeReuseMode::kTargetFill,
          step_seed,
          pairing_mode,
          truncation_mode);

      // Raw policy prior mu for fallback target / reference policy
      std::vector<std::pair<Action, double>> mu;
      if (!use_mpo_target && intra_step == 1 && !snap.raw_policy.empty() && current_legals.size() == snap.legal_actions.size()) {
        mu = snap.raw_policy;
      } else if (!evaluators.empty()) {
        ActionsAndProbs raw_prior = evaluators.front()->Prior(*snap.sim_state);
        mu.reserve(current_legals.size());
        for (Action a : current_legals) {
          double p = 0.0;
          for (const auto& ap : raw_prior) {
            if (ap.first == a) { p = ap.second; break; }
          }
          mu.push_back({a, p});
        }
      } else {
        mu = snap.raw_policy;
      }
      Action raw_action = PickGreedyAction(mu, current_legals);

      bool fell_back = false;
      std::vector<double> target_dist;
      bool keep_example = true;
      double target_prob_search_move = 0.0;

      if (target_mode == "override_gated") {
        if (search_res.overridden) {
          target_dist = ComputeOverrideGatedTarget(
              mu, current_legals, search_res.action, override_eta);
          for (size_t i = 0; i < current_legals.size(); ++i) {
            if (current_legals[i] == search_res.action) {
              target_prob_search_move = target_dist[i];
              break;
            }
          }
        } else {
          keep_example = false;
          open_spiel::SearchTrainingExample agr_ex;
          if (intra_step == 1 && !snap.observation.empty()) {
            agr_ex.observation = snap.observation;
            agr_ex.candidate_data = snap.candidate_data;
          } else {
            const auto active_mode = (!evaluators.empty() && evaluators.front())
                                         ? evaluators.front()->MarketMode()
                                         : dune_imperium::MarketAppendixMode::kFullPublicInformationV3;
            const auto active_semantics = (!evaluators.empty() && evaluators.front())
                                              ? evaluators.front()->DeckPoolSemantics()
                                              : dune_imperium::DeckPoolSemantics::kLegacy;
            agr_ex.observation = ExtractObservationTensor(*snap.sim_state, snap.player, active_mode, active_semantics);
            const auto* dune_s = dynamic_cast<const dune_imperium::DuneImperiumState*>(snap.sim_state.get());
            if (dune_s) {
              const std::string cand_schema = (!evaluators.empty() && evaluators.front())
                                                  ? evaluators.front()->SemanticDescriptorSchema()
                                                  : dune_semantic::kDescriptorSchemaVersionV3;
              dune_semantic::ExtractCandidateDescriptors(*dune_s, current_legals, &agr_ex.candidate_data, cand_schema);
            }
          }
          agr_ex.player = snap.player;
          agr_ex.legal_actions = current_legals;
          tr.agreement_examples.push_back(std::move(agr_ex));
        }
      } else if (use_mpo_target) {
        double r_kl = 0.0;
        double r_tau = softmax_temperature;
        target_dist = ComputeMpoRelativeTarget(
            mu, current_legals, search_res.candidate_utilities,
            softmax_temperature, max_target_kl, &r_kl, &r_tau);
        for (size_t i = 0; i < current_legals.size(); ++i) {
          if (current_legals[i] == search_res.action) {
            target_prob_search_move = target_dist[i];
            break;
          }
        }
      } else {
        target_dist = ComputeFallbackAlignedTarget(
            mu, current_legals, search_res, raw_action,
            softmax_temperature, eta_blend, &fell_back);
        for (size_t i = 0; i < current_legals.size(); ++i) {
          if (current_legals[i] == search_res.action) {
            target_prob_search_move = target_dist[i];
            break;
          }
        }
      }

      if (keep_example) {
        open_spiel::SearchTrainingExample ex;
        if (intra_step == 1 && !snap.observation.empty()) {
          ex.observation = snap.observation;
          ex.candidate_data = snap.candidate_data;
        } else {
          const auto active_mode = (!evaluators.empty() && evaluators.front())
                                       ? evaluators.front()->MarketMode()
                                       : dune_imperium::MarketAppendixMode::kFullPublicInformationV3;
          const auto active_semantics = (!evaluators.empty() && evaluators.front())
                                            ? evaluators.front()->DeckPoolSemantics()
                                            : dune_imperium::DeckPoolSemantics::kLegacy;
          ex.observation = ExtractObservationTensor(*snap.sim_state, snap.player, active_mode, active_semantics);
          const auto* dune_s = dynamic_cast<const dune_imperium::DuneImperiumState*>(snap.sim_state.get());
          if (dune_s) {
            const std::string cand_schema = (!evaluators.empty() && evaluators.front())
                                                ? evaluators.front()->SemanticDescriptorSchema()
                                                : dune_semantic::kDescriptorSchemaVersionV3;
            dune_semantic::ExtractCandidateDescriptors(*dune_s, current_legals, &ex.candidate_data, cand_schema);
          }
        }
        ex.player = snap.player;
        ex.legal_actions = current_legals;
        ex.normalized_visits = std::move(target_dist);
        ex.value_target = 0.0;
        ex.value_target_attached = false;
        ex.simulations_completed = search_res.new_rollouts_executed;
        ex.raw_action = raw_action;
        ex.best_action = search_res.action;
        if (snap.sim_state) {
          ex.state_snapshot = snap.sim_state->Clone();
        }

        tr.target_prob_search_move_sum += target_prob_search_move;
        tr.examples.push_back(std::move(ex));
      }

      tr.decisions_evaluated++;
      tr.rollouts += search_res.new_rollouts_executed;
      if (search_res.overridden) tr.overrides++;
      if (fell_back) tr.fallbacks++;

      Action chosen_action = search_res.action;
      turn_tree.Advance(chosen_action);
      const auto* dune_before = dynamic_cast<const dune_imperium::DuneImperiumState*>(snap.sim_state.get());
      std::unique_ptr<dune_imperium::DuneImperiumState> dune_snap_before =
          dune_before ? std::unique_ptr<dune_imperium::DuneImperiumState>(
                            dynamic_cast<dune_imperium::DuneImperiumState*>(dune_before->Clone().release()))
                      : nullptr;
      snap.sim_state->ApplyAction(chosen_action);
      if (dune_snap_before) {
        const auto* dune_after = dynamic_cast<const dune_imperium::DuneImperiumState*>(snap.sim_state.get());
        if (dune_after && HasNewInformation(*dune_snap_before, *dune_after, snap.player)) {
          turn_tree.Reset();
        }
      }

      if (chosen_action == dune_imperium::kActionEndTurn) {
        break;
      }
    }
  };

  int actual_workers = std::min(num_workers, static_cast<int>(snapshots.size()));
  if (actual_workers <= 1) {
    for (size_t r_idx = 0; r_idx < snapshots.size(); ++r_idx) {
      process_root(r_idx);
    }
  } else {
    std::atomic<size_t> next_root{0};
    std::vector<std::thread> workers;
    workers.reserve(actual_workers);
    for (int w = 0; w < actual_workers; ++w) {
      workers.emplace_back([&]() {
        while (true) {
          size_t r_idx = next_root.fetch_add(1);
          if (r_idx >= snapshots.size()) break;
          process_root(r_idx);
        }
      });
    }
    for (auto& worker : workers) {
      worker.join();
    }
  }

  double total_target_prob_search_move = 0.0;
  for (size_t r_idx = 0; r_idx < snapshots.size(); ++r_idx) {
    if (!turn_results[r_idx].valid) continue;
    result.roots_evaluated++;
    result.overrides += turn_results[r_idx].overrides;
    result.fallbacks_to_mu += turn_results[r_idx].fallbacks;
    result.total_decisions_evaluated += turn_results[r_idx].decisions_evaluated;
    total_target_prob_search_move += turn_results[r_idx].target_prob_search_move_sum;
    for (auto& ex : turn_results[r_idx].examples) {
      result.examples.push_back(std::move(ex));
    }
    for (auto& ex : turn_results[r_idx].agreement_examples) {
      result.agreement_examples.push_back(std::move(ex));
    }
  }

  auto end_time = std::chrono::steady_clock::now();
  result.elapsed_seconds = std::chrono::duration<double>(end_time - start_time).count();
  if (result.total_decisions_evaluated > 0) {
    result.mean_override_rate = static_cast<double>(result.overrides) / result.total_decisions_evaluated;
  }
  if (!result.examples.empty()) {
    result.mean_target_prob_search_move = total_target_prob_search_move / result.examples.size();
  }
  return result;
}
}  // namespace dune_imperium
}  // namespace open_spiel

#endif  // OPEN_SPIEL_EXAMPLES_DUNE_COMPOUND_TURN_SEARCH_H_
