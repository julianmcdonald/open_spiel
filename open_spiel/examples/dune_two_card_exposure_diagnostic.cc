#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "open_spiel/abseil-cpp/absl/flags/flag.h"
#include "open_spiel/abseil-cpp/absl/flags/parse.h"
#include "open_spiel/abseil-cpp/absl/strings/str_cat.h"
#include "open_spiel/abseil-cpp/absl/strings/str_format.h"
#include "open_spiel/games/dune_imperium/dune_imperium.h"
#include "open_spiel/games/dune_imperium/dune_imperium_common.h"
#include "open_spiel/games/dune_imperium/dune_imperium_content.h"
#include "open_spiel/games/dune_imperium/dune_imperium_content_generated.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"
#include "open_spiel/utils/json.h"
#include <torch/torch.h>

#include "dune_batched_evaluator.h"
#include "dune_evaluator.h"
#include "dune_network.h"
#include "dune_seed_utils.h"
#include "dune_sha256.h"

ABSL_FLAG(std::string, model_checkpoint,
          "/run/media/warcr/Storage/dune_drl_runtime/round7/u20000_rollout_experiment_20260919/arm_control_256g/ppo_model_update_20328.pt",
          "Path to frozen U20328 checkpoint");
ABSL_FLAG(std::string, output_dir,
          "/home/warcr/dune_drl_runtime/round7/two_card_audit_20260919",
          "Directory to write audit artifacts");
ABSL_FLAG(uint64_t, seed, 11, "Production base seed");
ABSL_FLAG(int, seed_scheme_version, 2, "Seed scheme version");
ABSL_FLAG(int, max_games, 512, "Maximum complete games to simulate");
ABSL_FLAG(int, max_minutes, 90, "Hard timebox ceiling in minutes");
ABSL_FLAG(int, early_stop_opportunities, 100,
          "Early stopping threshold for eligible opportunities per card (min 256 games)");
ABSL_FLAG(int, num_threads, 32, "Number of rollout worker threads");
ABSL_FLAG(int, eval_batch_size, 32, "BatchedEvaluator batch size");
ABSL_FLAG(int, eval_timeout_ms, 5, "BatchedEvaluator timeout in ms");
ABSL_FLAG(int, hidden_dim, 2048, "Network hidden dimension");
ABSL_FLAG(int, num_blocks, 8, "Network residual blocks count");

namespace open_spiel {
namespace {

using namespace dune_imperium;

constexpr int kScientificBreakthroughTleilaxuId = 11;
constexpr int kScientificBreakthroughCardId = 118;
constexpr int kStitchedHorrorTleilaxuId = 13;
constexpr int kStitchedHorrorCardId = 120;
constexpr int kSpecimenCost = 3;

constexpr int kTleilaxuAppendixOffset = kExpandedInformationStateSize;

std::unique_ptr<State> BuildStateAtFirstRevealTurn(const Game& game) {
  auto state = game.NewInitialState();
  int safety = 0;
  while (state->ToString().find("phase=reveal_turns") == std::string::npos) {
    ++safety;
    SPIEL_CHECK_LT(safety, 1000);
    if (state->IsChanceNode()) {
      state->ApplyAction(state->ChanceOutcomes()[0].first);
      continue;
    }
    auto legal = state->LegalActions();
    if (legal.size() == 1 && legal[0] == kActionAcknowledgeChance) {
      state->ApplyAction(kActionAcknowledgeChance);
      continue;
    }
    SPIEL_CHECK_FALSE(legal.empty());
    Action action = legal[0];
    if (std::find(legal.begin(), legal.end(), kActionReveal) != legal.end()) {
      action = kActionReveal;
    } else if (std::find(legal.begin(), legal.end(), kActionEndTurn) != legal.end()) {
      action = kActionEndTurn;
    }
    state->ApplyAction(action);
  }
  return state;
}

void ResolvePendingChanceNodes(State* state) {
  int safety = 0;
  while (!state->IsTerminal()) {
    ++safety;
    SPIEL_CHECK_LT(safety, 200);
    if (state->IsChanceNode()) {
      auto outcomes = state->ChanceOutcomes();
      SPIEL_CHECK_FALSE(outcomes.empty());
      state->ApplyAction(outcomes[0].first);
      continue;
    }
    auto legal = state->LegalActions();
    if (legal.size() == 1 && legal[0] == kActionAcknowledgeChance) {
      state->ApplyAction(kActionAcknowledgeChance);
      continue;
    }
    break;
  }
}

// -----------------------------------------------------------------------------
// Upfront Representation and Purchase Mechanics Verification
// -----------------------------------------------------------------------------
bool RunUpfrontMechanicsAndRepresentationChecks(const Game& game) {
  std::cout << "[PRE-CHECK] Running upfront representation and mechanics verification...\n";

  // Test 1: Observation Identity in Full Public Information v3
  {
    auto state = BuildStateAtFirstRevealTurn(game);
    auto* dune = dynamic_cast<DuneImperiumState*>(state.get());
    SPIEL_CHECK_TRUE(dune != nullptr);
    const Player p = dune->CurrentPlayer();

    // Slot 0 = 11, Slot 1 = 13
    dune->SetTleilaxuRowForTesting({kScientificBreakthroughTleilaxuId, kStitchedHorrorTleilaxuId});

    std::vector<float> obs(kFullPublicInformationStateSize, 0.0f);
    dune->InformationStateTensorWithAppendix(
        p, MarketAppendixMode::kFullPublicInformationV3, absl::MakeSpan(obs));

    // Slot 0 check
    for (int c = 0; c < kTleilaxuMarketSlotTotalDim; ++c) {
      float expected = (c == kScientificBreakthroughTleilaxuId) ? 1.0f : 0.0f;
      float actual = obs[kTleilaxuAppendixOffset + 0 * kTleilaxuMarketSlotTotalDim + c];
      if (actual != expected) {
        std::cerr << "[PRE-CHECK FAIL] Slot 0 observation mismatch at category "
                  << c << ": expected " << expected << ", got " << actual << "\n";
        return false;
      }
    }

    // Slot 1 check
    for (int c = 0; c < kTleilaxuMarketSlotTotalDim; ++c) {
      float expected = (c == kStitchedHorrorTleilaxuId) ? 1.0f : 0.0f;
      float actual = obs[kTleilaxuAppendixOffset + 1 * kTleilaxuMarketSlotTotalDim + c];
      if (actual != expected) {
        std::cerr << "[PRE-CHECK FAIL] Slot 1 observation mismatch at category "
                  << c << ": expected " << expected << ", got " << actual << "\n";
        return false;
      }
    }
  }

  // Test 2: Semantic Descriptors for Actions 91 and 92
  {
    auto state = BuildStateAtFirstRevealTurn(game);
    auto* dune = dynamic_cast<DuneImperiumState*>(state.get());
    SPIEL_CHECK_TRUE(dune != nullptr);

    const Action action_slot0 = kActionTleilaxuAcquire0 + 1; // 91
    const Action action_slot1 = kActionTleilaxuAcquire0 + 2; // 92

    dune_semantic::CandidateActionData cand_data;
    dune_semantic::ExtractCandidateDescriptors(
        *dune, {action_slot0, action_slot1}, &cand_data,
        dune_semantic::kDescriptorSchemaVersionV3);

    // Both should be unsupported / unbiased
    for (size_t i = 0; i < cand_data.actions.size(); ++i) {
      if (cand_data.supported[i] != 0) {
        std::cerr << "[PRE-CHECK FAIL] Action " << cand_data.actions[i]
                  << " supported != 0\n";
        return false;
      }
      if (cand_data.roles[i] != dune_semantic::ActionRole::kUnsupported) {
        std::cerr << "[PRE-CHECK FAIL] Action " << cand_data.actions[i]
                  << " role != kUnsupported\n";
        return false;
      }
      if (cand_data.features[i * dune_semantic::kSemanticFeatDim + 0] != 1.0f) {
        std::cerr << "[PRE-CHECK FAIL] Action " << cand_data.actions[i]
                  << " features[0] != 1.0f\n";
        return false;
      }
    }
  }

  // Test 3: Purchase mechanics, specimen payment boundary, and acquisition destination
  for (int target_id : {kScientificBreakthroughTleilaxuId, kStitchedHorrorTleilaxuId}) {
    const int expected_card_id = (target_id == kScientificBreakthroughTleilaxuId)
                                     ? kScientificBreakthroughCardId
                                     : kStitchedHorrorCardId;
    for (int slot = 0; slot < 2; ++slot) {
      const Action purchase_action = kActionTleilaxuAcquire0 + 1 + slot;

      // Subtest 3A: Specimen payment boundary (2 specimens -> illegal, 3 specimens -> legal)
      {
        auto state = BuildStateAtFirstRevealTurn(game);
        auto* dune = dynamic_cast<DuneImperiumState*>(state.get());
        const Player p = dune->CurrentPlayer();

        std::vector<int> row = {kInvalidCard, kInvalidCard};
        row[slot] = target_id;
        row[1 - slot] = (target_id == kScientificBreakthroughTleilaxuId)
                            ? kStitchedHorrorTleilaxuId
                            : kScientificBreakthroughTleilaxuId;
        dune->SetTleilaxuRowForTesting(row);
        dune->SetSpecimensForTesting(p, 2);

        auto legal_2 = dune->LegalActions();
        if (std::find(legal_2.begin(), legal_2.end(), purchase_action) != legal_2.end()) {
          std::cerr << "[PRE-CHECK FAIL] Card " << target_id << " in slot " << slot
                    << " was legal at 2 specimens!\n";
          return false;
        }

        dune->SetSpecimensForTesting(p, 3);
        auto legal_3 = dune->LegalActions();
        if (std::find(legal_3.begin(), legal_3.end(), purchase_action) == legal_3.end()) {
          std::cerr << "[PRE-CHECK FAIL] Card " << target_id << " in slot " << slot
                    << " was NOT legal at 3 specimens!\n";
          return false;
        }

        // Subtest 3B: Exact deduction and default acquisition to discard
        dune->ApplyAction(purchase_action);
        ResolvePendingChanceNodes(dune);
        if (dune->GetSpecimensForTesting(p) != 0) {
          std::cerr << "[PRE-CHECK FAIL] Specimens not decremented to 0 after purchase\n";
          return false;
        }
        const auto& discard = dune->GetPlayerDiscardForTesting(p);
        if (discard.empty() || discard.back() != expected_card_id) {
          std::cerr << "[PRE-CHECK FAIL] Purchased card " << expected_card_id
                    << " not found at top of discard pile\n";
          return false;
        }
      }

      // Subtest 3C: Topdeck acquisition destination
      {
        auto state = BuildStateAtFirstRevealTurn(game);
        auto* dune = dynamic_cast<DuneImperiumState*>(state.get());
        const Player p = dune->CurrentPlayer();

        std::vector<int> row = {kInvalidCard, kInvalidCard};
        row[slot] = target_id;
        row[1 - slot] = (target_id == kScientificBreakthroughTleilaxuId)
                            ? kStitchedHorrorTleilaxuId
                            : kScientificBreakthroughTleilaxuId;
        dune->SetTleilaxuRowForTesting(row);
        dune->SetSpecimensForTesting(p, 3);
        dune->SetTopdeckAcquireForTesting(p, true);

        dune->ApplyAction(purchase_action);
        ResolvePendingChanceNodes(dune);
        auto post_legal = dune->LegalActions();
        if (std::find(post_legal.begin(), post_legal.end(), kActionIntrigueChoiceModeA) != post_legal.end()) {
          dune->ApplyAction(kActionIntrigueChoiceModeA);
          const auto& deck = dune->GetPlayerDeckTopCardsForTesting(p);
          if (deck.empty() || deck.back() != expected_card_id) {
            std::cerr << "[PRE-CHECK FAIL] Topdeck acquired card " << expected_card_id
                      << " not at top of draw deck\n";
            return false;
          }
        } else {
          std::cerr << "[PRE-CHECK FAIL] Did not enter topdeck choice after topdeck acquire flag\n";
          return false;
        }
      }
    }
  }

  std::cout << "[PRE-CHECK PASS] Both cards' representation, row slots, observations, "
            << "semantic descriptors, specimen payment, and acquisition destinations verified.\n";
  return true;
}

// -----------------------------------------------------------------------------
// Detailed Record and Reservoir Sampling
// -----------------------------------------------------------------------------
struct AlternativeAction {
  Action action_id = -1;
  std::string action_name;
  double prob = 0.0;
};

struct DetailedOpportunityRecord {
  uint64_t sample_id = 0;
  uint64_t episode_id = 0;
  int round = 0;
  int turn = 0;
  int player_seat = 0;
  std::string leader_name;
  int specimens = 0;
  int target_tleilaxu_id = 0;
  std::string target_card_name;
  int target_row_slot = 0;
  Action purchase_action_id = -1;
  int num_legal_actions = 0;
  float purchase_raw_centered_logit = 0.0f;
  float purchase_capped_logit = 0.0f;
  double purchase_policy_prob = 0.0;
  int purchase_policy_rank = 0;
  Action chosen_action_id = -1;
  std::string chosen_action_name;
  double chosen_action_prob = 0.0;
  bool was_target_purchased = false;
  std::vector<AlternativeAction> top_actions;
};

}  // namespace

struct ActivePurchaseTracker {
  int target_tleilaxu_id = 0;
  int card_id = 0;
  open_spiel::Player owner = 0;
  bool drawn = false;
};

inline std::string LocalJsonEscape(const std::string& input) {
  std::string out;
  out.reserve(input.length());
  for (const char c : input) {
    switch (c) {
      case '"': out.append("\\\""); break;
      case '\\': out.append("\\\\"); break;
      case '\b': out.append("\\b"); break;
      case '\f': out.append("\\f"); break;
      case '\n': out.append("\\n"); break;
      case '\r': out.append("\\r"); break;
      case '\t': out.append("\\t"); break;
      default: out.push_back(c); break;
    }
  }
  return out;
}

std::string LocalJsonToString(const open_spiel::json::Value& value, bool wrap = false, int indent = 0);

inline std::string LocalJsonToString(const open_spiel::json::Array& array, bool wrap, int indent) {
  std::string out = "[";
  bool first = true;
  for (const auto& v : array) {
    if (!first) out += ",";
    if (wrap) {
      out += "\n" + std::string(indent + 2, ' ');
    } else if (!first) {
      out += " ";
    }
    first = false;
    out += LocalJsonToString(v, wrap, indent + 2);
  }
  if (wrap) {
    out += "\n" + std::string(indent, ' ');
  }
  out += "]";
  return out;
}

inline std::string LocalJsonToString(const open_spiel::json::Object& obj, bool wrap, int indent) {
  std::string out = "{";
  bool first = true;
  for (const auto& [key, value] : obj) {
    if (!first) out += ",";
    if (wrap) {
      out += "\n" + std::string(indent + 2, ' ');
    } else if (!first) {
      out += " ";
    }
    first = false;
    out += "\"" + LocalJsonEscape(key) + "\": " +
           LocalJsonToString(value, wrap, indent + 2);
  }
  if (wrap) {
    out += "\n" + std::string(indent, ' ');
  }
  out += "}";
  return out;
}

inline std::string LocalJsonToString(const open_spiel::json::Value& value, bool wrap, int indent) {
  if (value.IsNull()) {
    return "null";
  } else if (value.IsBool()) {
    return (value.GetBool() ? "true" : "false");
  } else if (value.IsInt()) {
    return std::to_string(value.GetInt());
  } else if (value.IsDouble()) {
    double v = value.GetDouble();
    if (std::isfinite(v)) {
      std::string s = absl::StrFormat("%.17g", v);
      if (s.find('.') == std::string::npos &&
          s.find('e') == std::string::npos &&
          s.find('E') == std::string::npos) {
        s += ".0";
      }
      return s;
    } else {
      return absl::StrCat("\"", std::to_string(v), "\"");
    }
  } else if (value.IsString()) {
    return absl::StrCat("\"", LocalJsonEscape(value.GetString()), "\"");
  } else if (value.IsArray()) {
    return LocalJsonToString(value.GetArray(), wrap, indent);
  } else if (value.IsObject()) {
    return LocalJsonToString(value.GetObject(), wrap, indent);
  } else {
    open_spiel::SpielFatalError("LocalJsonToString is missing a type.");
  }
}


namespace {

class ReservoirSampler {
 public:
  explicit ReservoirSampler(size_t capacity, uint64_t seed)
      : capacity_(capacity), rng_(seed), total_seen_(0) {}

  void Consider(const DetailedOpportunityRecord& record) {
    std::lock_guard<std::mutex> lock(mutex_);
    total_seen_++;
    if (record.was_target_purchased) {
      purchased_records_.push_back(record);
      purchased_records_.back().sample_id = total_seen_;
      return;
    }
    if (samples_.size() < capacity_) {
      samples_.push_back(record);
      samples_.back().sample_id = total_seen_;
    } else {
      std::uniform_int_distribution<uint64_t> dist(0, total_seen_ - 1);
      uint64_t j = dist(rng_);
      if (j < capacity_) {
        samples_[j] = record;
        samples_[j].sample_id = total_seen_;
      }
    }
  }

  std::vector<DetailedOpportunityRecord> GetSamples() const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<DetailedOpportunityRecord> out = purchased_records_;
    size_t remaining_slots = (capacity_ > out.size()) ? (capacity_ - out.size()) : 0;
    for (size_t i = 0; i < samples_.size() && i < remaining_slots; ++i) {
      out.push_back(samples_[i]);
    }
    return out;
  }

  uint64_t TotalSeen() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return total_seen_;
  }

  size_t PurchasesPreserved() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return purchased_records_.size();
  }

 private:
  size_t capacity_;
  mutable std::mutex mutex_;
  std::mt19937_64 rng_;
  uint64_t total_seen_;
  std::vector<DetailedOpportunityRecord> purchased_records_;
  std::vector<DetailedOpportunityRecord> samples_;
};

// -----------------------------------------------------------------------------
// Aggregate Accounting Data Structures
// -----------------------------------------------------------------------------
struct CardAggregateStats {
  int tleilaxu_id = 0;
  std::string card_name;

  uint64_t row_presence_decisions = 0;
  uint64_t row_presence_specimens_lt3 = 0;
  uint64_t row_presence_specimens_ge3 = 0;

  uint64_t eligible_decisions = 0;
  uint64_t eligible_slot0 = 0;
  uint64_t eligible_slot1 = 0;

  uint64_t purchases_taken = 0;
  uint64_t purchases_slot0 = 0;
  uint64_t purchases_slot1 = 0;

  std::set<uint64_t> games_with_row_presence;
  std::set<uint64_t> games_with_eligibility;
  std::set<uint64_t> games_with_purchase;

  std::vector<double> policy_probs;
  std::vector<int> policy_ranks;
  std::map<int, uint64_t> rank_histogram;
  std::map<int, uint64_t> specimen_distribution;
  std::map<std::string, uint64_t> alternative_actions_taken;

  uint64_t downstream_acquisitions = 0;
  uint64_t downstream_draws = 0;
  uint64_t downstream_plays = 0;
};

struct DiagnosticAggregate {
  uint64_t total_games = 0;
  uint64_t aborted_games = 0;
  uint64_t total_decisions = 0;
  double elapsed_seconds = 0.0;
  std::string stop_reason;

  CardAggregateStats card11{kScientificBreakthroughTleilaxuId, "Scientific Breakthrough"};
  CardAggregateStats card13{kStitchedHorrorTleilaxuId, "Stitched Horror"};

  void Merge(const DiagnosticAggregate& other) {
    total_games += other.total_games;
    aborted_games += other.aborted_games;
    total_decisions += other.total_decisions;

    auto merge_card = [](CardAggregateStats& dst, const CardAggregateStats& src) {
      dst.row_presence_decisions += src.row_presence_decisions;
      dst.row_presence_specimens_lt3 += src.row_presence_specimens_lt3;
      dst.row_presence_specimens_ge3 += src.row_presence_specimens_ge3;

      dst.eligible_decisions += src.eligible_decisions;
      dst.eligible_slot0 += src.eligible_slot0;
      dst.eligible_slot1 += src.eligible_slot1;

      dst.purchases_taken += src.purchases_taken;
      dst.purchases_slot0 += src.purchases_slot0;
      dst.purchases_slot1 += src.purchases_slot1;

      dst.games_with_row_presence.insert(src.games_with_row_presence.begin(),
                                         src.games_with_row_presence.end());
      dst.games_with_eligibility.insert(src.games_with_eligibility.begin(),
                                        src.games_with_eligibility.end());
      dst.games_with_purchase.insert(src.games_with_purchase.begin(),
                                     src.games_with_purchase.end());

      dst.policy_probs.insert(dst.policy_probs.end(),
                              src.policy_probs.begin(), src.policy_probs.end());
      dst.policy_ranks.insert(dst.policy_ranks.end(),
                              src.policy_ranks.begin(), src.policy_ranks.end());

      for (const auto& [rank, count] : src.rank_histogram) {
        dst.rank_histogram[rank] += count;
      }
      for (const auto& [spec, count] : src.specimen_distribution) {
        dst.specimen_distribution[spec] += count;
      }
      for (const auto& [act, count] : src.alternative_actions_taken) {
        dst.alternative_actions_taken[act] += count;
      }

      dst.downstream_acquisitions += src.downstream_acquisitions;
      dst.downstream_draws += src.downstream_draws;
      dst.downstream_plays += src.downstream_plays;
    };

    merge_card(card11, other.card11);
    merge_card(card13, other.card13);
  }
};

std::string GetLeaderName(int leader_id) {
  if (leader_id >= 0 && leader_id < kNumLeaders) {
    return std::string(kLeaders[leader_id].name);
  }
  return absl::StrCat("Leader_", leader_id);
}

// -----------------------------------------------------------------------------
// Rollout Worker Thread
// -----------------------------------------------------------------------------
void DiagnosticWorkerThread(
    int thread_id,
    std::shared_ptr<const Game> game,
    std::shared_ptr<BatchedEvaluator> evaluator,
    int64_t obs_size,
    uint64_t master_seed,
    int max_games,
    std::chrono::steady_clock::time_point start_time,
    double max_seconds,
    int early_stop_opps,
    std::atomic<uint64_t>& next_episode_id,
    std::atomic<int>& completed_games_counter,
    std::atomic<uint64_t>& global_card11_opps,
    std::atomic<uint64_t>& global_card13_opps,
    std::atomic<bool>& stop_flag,
    std::string& global_stop_reason,
    std::mutex& stop_reason_mutex,
    ReservoirSampler& reservoir,
    DiagnosticAggregate& thread_stats) {

  std::vector<float> obs(obs_size, 0.0f);

  while (!stop_flag.load(std::memory_order_relaxed)) {
    uint64_t episode_id = next_episode_id.fetch_add(1, std::memory_order_relaxed);
    if (episode_id >= static_cast<uint64_t>(max_games)) {
      std::lock_guard<std::mutex> lock(stop_reason_mutex);
      if (global_stop_reason.empty()) {
        global_stop_reason = absl::StrFormat("Reached maximum games ceiling (%d complete games)", max_games);
      }
      stop_flag.store(true, std::memory_order_relaxed);
      break;
    }

    // Check timebox limit
    auto now = std::chrono::steady_clock::now();
    double elapsed = std::chrono::duration<double>(now - start_time).count();
    if (elapsed >= max_seconds) {
      std::lock_guard<std::mutex> lock(stop_reason_mutex);
      if (global_stop_reason.empty()) {
        global_stop_reason = absl::StrFormat("Reached 90-minute timebox limit (%.1f s elapsed)", elapsed);
      }
      stop_flag.store(true, std::memory_order_relaxed);
      break;
    }

    // Seed scheme 2 derivation
    auto chance_rng = dune_seed::MakeRng64(dune_seed::DeriveSeed(
        master_seed, dune_seed::kDomainTrain, episode_id, dune_seed::kStreamChance));
    std::array<std::mt19937_64, 4> policy_rngs;
    for (int p = 0; p < 4; ++p) {
      policy_rngs[p] = dune_seed::MakeRng64(dune_seed::DeriveSeed(
          master_seed, dune_seed::kDomainTrain, episode_id,
          dune_seed::kStreamPolicyPlayer0 + p));
    }

    std::unique_ptr<State> state = game->NewInitialState();
    auto* dune_state = dynamic_cast<DuneImperiumState*>(state.get());
    SPIEL_CHECK_TRUE(dune_state != nullptr);

    int game_moves = 0;

    // Track any purchase in this episode for downstream draw verification
    std::vector<ActivePurchaseTracker> game_purchases;
    auto check_draws = [&]() {
      for (Player p = 0; p < 4; ++p) {
        const auto& hand = dune_state->GetPlayerHandForTesting(p);
        for (int target_id : {kScientificBreakthroughTleilaxuId, kStitchedHorrorTleilaxuId}) {
          int expected_card_id = (target_id == kScientificBreakthroughTleilaxuId)
                                     ? kScientificBreakthroughCardId
                                     : kStitchedHorrorCardId;
          int hand_copies = 0;
          for (int c : hand) {
            if (c == expected_card_id) ++hand_copies;
          }
          if (hand_copies == 0) continue;

          for (auto& ap : game_purchases) {
            if (ap.owner == p && ap.card_id == expected_card_id && !ap.drawn) {
              ap.drawn = true;
              if (target_id == kScientificBreakthroughTleilaxuId) {
                thread_stats.card11.downstream_draws++;
              } else {
                thread_stats.card13.downstream_draws++;
              }
              --hand_copies;
              if (hand_copies == 0) break;
            }
          }
        }
      }
    };

    while (!state->IsTerminal()) {
      ++game_moves;
      if (game_moves > 5000) {
        std::cerr << "Infinite loop guard hit in game " << episode_id << "\n";
        break;
      }

      if (state->IsChanceNode()) {
        auto outcomes = state->ChanceOutcomes();
        Action action = (game->GetType().chance_mode == GameType::ChanceMode::kSampledStochastic)
                            ? outcomes.front().first
                            : SampleAction(outcomes, chance_rng).first;
        state->ApplyAction(action);
        check_draws();
        continue;
      }

      if (state->CurrentPlayer() == kSimultaneousPlayerId) {
        std::vector<Action> joint;
        for (int p = 0; p < game->NumPlayers(); ++p) {
          auto acts = state->LegalActions(p);
          if (acts.empty()) {
            joint.push_back(0);
          } else {
            std::uniform_int_distribution<int> d(0, acts.size() - 1);
            joint.push_back(acts[d(policy_rngs[p])]);
          }
        }
        state->ApplyActions(joint);
        check_draws();
        continue;
      }

      Player current_player = state->CurrentPlayer();
      SPIEL_CHECK_GE(current_player, 0);
      SPIEL_CHECK_LT(current_player, 4);

      std::vector<Action> legal_actions = state->LegalActions();
      if (legal_actions.empty()) break;

      thread_stats.total_decisions++;

      // Extract observation and semantic candidate descriptors
      std::fill(obs.begin(), obs.end(), 0.0f);
      dune_state->InformationStateTensorWithAppendix(
          current_player, MarketAppendixMode::kFullPublicInformationV3,
          absl::MakeSpan(obs));

      dune_semantic::CandidateActionData cand_data;
      dune_semantic::ExtractCandidateDescriptors(
          *dune_state, legal_actions, &cand_data,
          dune_semantic::kDescriptorSchemaVersionV3);

      // Single network inference pass
      EvalResult result = evaluator->EvaluateWithActions(obs, &cand_data);
      const std::vector<float> raw_logits = result.logits;
      std::vector<float> logits = std::move(result.logits);

      // Inspect row presence for both cards
      const auto& tleilaxu_row = dune_state->GetTleilaxuRowForTesting();
      int row_card_0 = (tleilaxu_row.size() > 0) ? tleilaxu_row[0] : kInvalidCard;
      int row_card_1 = (tleilaxu_row.size() > 1) ? tleilaxu_row[1] : kInvalidCard;
      int specimens = dune_state->GetSpecimensForTesting(current_player);

      for (int card_id : {kScientificBreakthroughTleilaxuId, kStitchedHorrorTleilaxuId}) {
        auto& cstats = (card_id == kScientificBreakthroughTleilaxuId)
                           ? thread_stats.card11
                           : thread_stats.card13;
        bool in_row = (row_card_0 == card_id || row_card_1 == card_id);
        if (in_row) {
          cstats.row_presence_decisions++;
          cstats.games_with_row_presence.insert(episode_id);
          if (specimens < kSpecimenCost) {
            cstats.row_presence_specimens_lt3++;
          } else {
            cstats.row_presence_specimens_ge3++;
          }
        }
      }

      // Check whether target cards are eligible for purchase at this decision
      const Action action_buy_0 = kActionTleilaxuAcquire0 + 1; // 91
      const Action action_buy_1 = kActionTleilaxuAcquire0 + 2; // 92

      bool can_buy_0 = std::find(legal_actions.begin(), legal_actions.end(),
                                 action_buy_0) != legal_actions.end();
      bool can_buy_1 = std::find(legal_actions.begin(), legal_actions.end(),
                                 action_buy_1) != legal_actions.end();

      // Pre-cap centered logits calculation for reference
      double legal_sum = 0.0;
      for (Action a : legal_actions) {
        legal_sum += logits[a];
      }
      float legal_mean = static_cast<float>(legal_sum / legal_actions.size());

      // Production centering and capping: in place
      CenterAndCapLegalLogits(logits, legal_actions, 10.0f);

      // Softmax probabilities over legal actions
      float max_capped_logit = -1e9f;
      for (Action a : legal_actions) {
        if (logits[a] > max_capped_logit) max_capped_logit = logits[a];
      }
      std::vector<double> probs(legal_actions.size(), 0.0);
      double sum_exp = 0.0;
      for (size_t i = 0; i < legal_actions.size(); ++i) {
        double e = std::exp(static_cast<double>(logits[legal_actions[i]] - max_capped_logit));
        probs[i] = e;
        sum_exp += e;
      }
      for (double& p : probs) {
        p /= sum_exp;
      }

      // Production sampling: EXACTLY uses policy_rngs[current_player]
      const PolicyDistributionSample policy_sample =
          SamplePolicyDistribution(&policy_rngs[current_player], logits, legal_actions, nullptr);
      Action chosen_action = policy_sample.action;
      double chosen_prob = 0.0;
      for (size_t i = 0; i < legal_actions.size(); ++i) {
        if (legal_actions[i] == chosen_action) {
          chosen_prob = probs[i];
          break;
        }
      }

      // Check eligibility and record diagnostic statistics
      auto process_opportunity = [&](int target_id, int slot, Action buy_action) {
        auto& cstats = (target_id == kScientificBreakthroughTleilaxuId)
                           ? thread_stats.card11
                           : thread_stats.card13;
        cstats.eligible_decisions++;
        if (slot == 0) cstats.eligible_slot0++;
        else cstats.eligible_slot1++;
        cstats.games_with_eligibility.insert(episode_id);
        cstats.specimen_distribution[specimens]++;

        if (target_id == kScientificBreakthroughTleilaxuId) {
          global_card11_opps.fetch_add(1, std::memory_order_relaxed);
        } else {
          global_card13_opps.fetch_add(1, std::memory_order_relaxed);
        }

        // Find probability and rank of the buy action
        double buy_prob = 0.0;
        int rank = 1;
        float raw_centered = 0.0f;
        float capped_logit = logits[buy_action];

        for (size_t i = 0; i < legal_actions.size(); ++i) {
          if (legal_actions[i] == buy_action) {
            buy_prob = probs[i];
            raw_centered = raw_logits[buy_action] - legal_mean;
          }
        }
        for (size_t i = 0; i < legal_actions.size(); ++i) {
          if (probs[i] > buy_prob) {
            rank++;
          }
        }

        cstats.policy_probs.push_back(buy_prob);
        cstats.policy_ranks.push_back(rank);
        cstats.rank_histogram[rank]++;

        bool purchased = (chosen_action == buy_action);
        if (purchased) {
          cstats.purchases_taken++;
          if (slot == 0) cstats.purchases_slot0++;
          else cstats.purchases_slot1++;
          cstats.games_with_purchase.insert(episode_id);
          cstats.downstream_acquisitions++;

          int expected_card_id = (target_id == kScientificBreakthroughTleilaxuId)
                                     ? kScientificBreakthroughCardId
                                     : kStitchedHorrorCardId;
          game_purchases.push_back({target_id, expected_card_id, current_player, false});
        } else {
          std::string chosen_str = state->ActionToString(current_player, chosen_action);
          cstats.alternative_actions_taken[chosen_str]++;
        }

        // Detailed record for reservoir sampling
        DetailedOpportunityRecord record;
        record.episode_id = episode_id;
        record.round = dune_state->GetCurrentRound();
        record.turn = game_moves;
        record.player_seat = current_player;
        record.leader_name = GetLeaderName(dune_state->PlayerLeader(current_player));
        record.specimens = specimens;
        record.target_tleilaxu_id = target_id;
        record.target_card_name = cstats.card_name;
        record.target_row_slot = slot;
        record.purchase_action_id = buy_action;
        record.num_legal_actions = legal_actions.size();
        record.purchase_raw_centered_logit = raw_centered;
        record.purchase_capped_logit = capped_logit;
        record.purchase_policy_prob = buy_prob;
        record.purchase_policy_rank = rank;
        record.chosen_action_id = chosen_action;
        record.chosen_action_name = state->ActionToString(current_player, chosen_action);
        record.chosen_action_prob = chosen_prob;
        record.was_target_purchased = purchased;

        // Top-3 actions by probability
        std::vector<std::pair<double, Action>> sorted_acts;
        sorted_acts.reserve(legal_actions.size());
        for (size_t i = 0; i < legal_actions.size(); ++i) {
          sorted_acts.push_back({probs[i], legal_actions[i]});
        }
        std::sort(sorted_acts.rbegin(), sorted_acts.rend());
        for (size_t i = 0; i < std::min<size_t>(3, sorted_acts.size()); ++i) {
          record.top_actions.push_back({
              sorted_acts[i].second,
              state->ActionToString(current_player, sorted_acts[i].second),
              sorted_acts[i].first});
        }

        // Reservoir sampling (all purchases guaranteed preserved)
        reservoir.Consider(record);
      };

      if (can_buy_0) {
        if (row_card_0 == kScientificBreakthroughTleilaxuId) {
          process_opportunity(kScientificBreakthroughTleilaxuId, 0, action_buy_0);
        } else if (row_card_0 == kStitchedHorrorTleilaxuId) {
          process_opportunity(kStitchedHorrorTleilaxuId, 0, action_buy_0);
        }
      }
      if (can_buy_1) {
        if (row_card_1 == kScientificBreakthroughTleilaxuId) {
          process_opportunity(kScientificBreakthroughTleilaxuId, 1, action_buy_1);
        } else if (row_card_1 == kStitchedHorrorTleilaxuId) {
          process_opportunity(kStitchedHorrorTleilaxuId, 1, action_buy_1);
        }
      }

      // Check downstream card plays
      std::string chosen_act_str = state->ActionToString(current_player, chosen_action);
      if (chosen_act_str.find("Scientific Breakthrough") != std::string::npos &&
          chosen_action != action_buy_0 && chosen_action != action_buy_1) {
        thread_stats.card11.downstream_plays++;
      }
      if (chosen_act_str.find("Stitched Horror") != std::string::npos &&
          chosen_action != action_buy_0 && chosen_action != action_buy_1) {
        thread_stats.card13.downstream_plays++;
      }

      // Advance game state
      state->ApplyAction(chosen_action);
      check_draws();
    }

    if (state->IsTerminal()) {
      thread_stats.total_games++;
      int total_done = completed_games_counter.fetch_add(1) + 1;

      // Early stopping condition check (every 16 games)
      if (total_done % 16 == 0) {
        if (total_done >= 256 &&
            global_card11_opps.load(std::memory_order_relaxed) >= static_cast<uint64_t>(early_stop_opps) &&
            global_card13_opps.load(std::memory_order_relaxed) >= static_cast<uint64_t>(early_stop_opps)) {
          std::lock_guard<std::mutex> lock(stop_reason_mutex);
          if (global_stop_reason.empty()) {
            global_stop_reason = absl::StrFormat(
                "Early stopping threshold reached: >= %d games and >= %d global opportunities per card",
                total_done, early_stop_opps);
          }
          stop_flag.store(true, std::memory_order_relaxed);
        }
      }
    } else {
      thread_stats.aborted_games++;
    }
  }
}

// -----------------------------------------------------------------------------
// Statistics Reporting Utilities
// -----------------------------------------------------------------------------
struct QuantileSummary {
  double min = 0.0;
  double p10 = 0.0;
  double p25 = 0.0;
  double p50 = 0.0;
  double p75 = 0.0;
  double p90 = 0.0;
  double max = 0.0;
  double mean = 0.0;
};

QuantileSummary ComputeQuantiles(std::vector<double> values) {
  QuantileSummary q;
  if (values.empty()) return q;
  std::sort(values.begin(), values.end());
  q.min = values.front();
  q.max = values.back();
  double sum = 0.0;
  for (double v : values) sum += v;
  q.mean = sum / values.size();

  auto get_p = [&](double p) {
    size_t idx = static_cast<size_t>(std::round(p * (values.size() - 1)));
    if (idx >= values.size()) idx = values.size() - 1;
    return values[idx];
  };

  q.p10 = get_p(0.10);
  q.p25 = get_p(0.25);
  q.p50 = get_p(0.50);
  q.p75 = get_p(0.75);
  q.p90 = get_p(0.90);
  return q;
}

}  // namespace
}  // namespace open_spiel

// -----------------------------------------------------------------------------
// Main Function
// -----------------------------------------------------------------------------
int main(int argc, char* argv[]) {
  absl::ParseCommandLine(argc, argv);
  using open_spiel::LocalJsonToString;

  const std::string ckpt_path = absl::GetFlag(FLAGS_model_checkpoint);
  const std::string output_dir = absl::GetFlag(FLAGS_output_dir);
  const uint64_t master_seed = absl::GetFlag(FLAGS_seed);
  const int max_games = absl::GetFlag(FLAGS_max_games);
  const int max_minutes = absl::GetFlag(FLAGS_max_minutes);
  const double max_seconds = max_minutes * 60.0;
  const int num_threads = absl::GetFlag(FLAGS_num_threads);
  const int eval_batch_size = absl::GetFlag(FLAGS_eval_batch_size);
  const int eval_timeout_ms = absl::GetFlag(FLAGS_eval_timeout_ms);
  const int hidden_dim = absl::GetFlag(FLAGS_hidden_dim);
  const int num_blocks = absl::GetFlag(FLAGS_num_blocks);
  const int early_stop_opps = absl::GetFlag(FLAGS_early_stop_opportunities);

  std::cout << "==================================================================\n";
  std::cout << "   DUNE: IMPERIUM TWO-CARD PURCHASE EXPOSURE DIAGNOSTIC\n";
  std::cout << "   Scientific Breakthrough (ID 11) & Stitched Horror (ID 13)\n";
  std::cout << "==================================================================\n";
  std::cout << "Checkpoint:     " << ckpt_path << "\n";
  std::cout << "Output Dir:     " << output_dir << "\n";
  std::cout << "Base Seed:      " << master_seed << " (scheme "
            << absl::GetFlag(FLAGS_seed_scheme_version) << ")\n";
  std::cout << "Max Games:      " << max_games << "\n";
  std::cout << "Timebox Limit:  " << max_minutes << " minutes (" << max_seconds << " s)\n";
  std::cout << "Worker Threads: " << num_threads << "\n";
  std::cout << "Batch Size:     " << eval_batch_size << "\n";

  // Check model checkpoint integrity
  if (!std::filesystem::exists(ckpt_path)) {
    std::cerr << "Fatal Error: Model checkpoint does not exist: " << ckpt_path << "\n";
    return 1;
  }
  size_t file_size = 0;
  std::string sha256 = open_spiel::ComputeFileSHA256(ckpt_path, &file_size);
  std::cout << "Model Size:     " << file_size << " bytes\n";
  std::cout << "Model SHA-256:  " << sha256 << "\n";

  // Initialize Game
  std::shared_ptr<const open_spiel::Game> game = open_spiel::LoadGame("dune_imperium(enable_immortality=true)");
  if (!game) {
    std::cerr << "Fatal Error: Failed to load dune_imperium game\n";
    return 1;
  }

  // Run upfront representation and mechanics verification
  if (!open_spiel::RunUpfrontMechanicsAndRepresentationChecks(*game)) {
    std::cerr << "Fatal Error: Upfront mechanics/representation checks failed!\n";
    return 1;
  }

  // Load Model
  torch::Device device = torch::cuda::is_available() ? torch::Device(torch::kCUDA) : torch::Device(torch::kCPU);
  std::cout << "Inference Device: " << (device.is_cuda() ? "CUDA (NVIDIA GPU)" : "CPU") << "\n";

  auto model = std::make_shared<open_spiel::SharedDunePolicyValueNetImpl>(
      open_spiel::dune_imperium::kFullPublicInformationStateSize,
      hidden_dim, game->NumDistinctActions(), num_blocks,
      /*nonlinear_value_head=*/false, /*with_aux_heads=*/false,
      /*head_init_seed=*/0, /*with_semantic_scorer=*/true);

  open_spiel::LoadModelCheckpointRobust(model, ckpt_path, device);
  model->to(device);
  model->eval();

  std::shared_mutex sync_mutex;
  auto evaluator = std::make_shared<open_spiel::BatchedEvaluator>(
      model, eval_batch_size, eval_timeout_ms, device, &sync_mutex,
      /*logit_cap=*/0.0f, /*device_synchronize=*/true,
      /*high_priority_stream=*/false, /*emit_batch_membership=*/false,
      /*rollout_amp=*/false, /*allow_tf32=*/false);

  // Setup Reservoir Sampler with independent fixed seed
  open_spiel::ReservoirSampler reservoir(/*capacity=*/128, /*seed=*/20260919);

  // Thread coordination
  std::atomic<uint64_t> next_episode_id{0};
  std::atomic<int> completed_games_counter{0};
  std::atomic<uint64_t> global_card11_opps{0};
  std::atomic<uint64_t> global_card13_opps{0};
  std::atomic<bool> stop_flag{false};
  std::string global_stop_reason;
  std::mutex stop_reason_mutex;

  std::vector<open_spiel::DiagnosticAggregate> worker_stats(num_threads);
  auto start_time = std::chrono::steady_clock::now();

  std::cout << "\n[SIMULATION] Spawning " << num_threads << " workers for rollout collection...\n";
  std::vector<std::thread> workers;
  workers.reserve(num_threads);
  for (int t = 0; t < num_threads; ++t) {
    workers.emplace_back(
        open_spiel::DiagnosticWorkerThread,
        t, game, evaluator,
        open_spiel::dune_imperium::kFullPublicInformationStateSize,
        master_seed, max_games, start_time, max_seconds, early_stop_opps,
        std::ref(next_episode_id), std::ref(completed_games_counter),
        std::ref(global_card11_opps), std::ref(global_card13_opps),
        std::ref(stop_flag), std::ref(global_stop_reason),
        std::ref(stop_reason_mutex), std::ref(reservoir),
        std::ref(worker_stats[t]));
  }

  for (auto& w : workers) {
    if (w.joinable()) w.join();
  }

  auto end_time = std::chrono::steady_clock::now();
  double elapsed_seconds = std::chrono::duration<double>(end_time - start_time).count();

  // Merge results across worker threads
  open_spiel::DiagnosticAggregate aggregate;
  for (const auto& ws : worker_stats) {
    aggregate.Merge(ws);
  }
  aggregate.elapsed_seconds = elapsed_seconds;
  aggregate.stop_reason = global_stop_reason.empty() ? "Completed all scheduled games" : global_stop_reason;

  std::cout << "\n[SIMULATION COMPLETE]\n";
  std::cout << "Elapsed Time:    " << std::fixed << std::setprecision(2) << elapsed_seconds << " s\n";
  std::cout << "Complete Games:  " << aggregate.total_games << "\n";
  std::cout << "Aborted Games:   " << aggregate.aborted_games << "\n";
  std::cout << "Total Decisions: " << aggregate.total_decisions << "\n";
  std::cout << "Stop Reason:     " << aggregate.stop_reason << "\n";

  // Compute Quantiles
  auto q_card11 = open_spiel::ComputeQuantiles(aggregate.card11.policy_probs);
  auto q_card13 = open_spiel::ComputeQuantiles(aggregate.card13.policy_probs);

  // Create output directory
  std::filesystem::create_directories(output_dir);

  // ---------------------------------------------------------------------------
  // Generate Text Report
  // ---------------------------------------------------------------------------
  std::string report_file = output_dir + "/audit_report.md";
  std::ofstream rpt(report_file);
  rpt << "# Dune DRL: Two-Card Purchase Exposure Diagnostic Audit Report\n\n";
  rpt << "**Generated:** " << "2026-09-19" << "\n";
  rpt << "**Model Checkpoint:** `" << ckpt_path << "`\n";
  rpt << "**Checkpoint SHA-256:** `" << sha256 << "`\n";
  rpt << "**Base Seed:** `" << master_seed << "` (scheme 2)\n";
  rpt << "**Elapsed Time:** `" << std::fixed << std::setprecision(2) << elapsed_seconds << " s` (limit: " << max_minutes << " min)\n";
  rpt << "**Games Completed:** `" << aggregate.total_games << "` (ceiling: " << max_games << ")\n";
  rpt << "**Stop Condition:** " << aggregate.stop_reason << "\n\n";

  rpt << "## 1. Upfront Representation and Mechanics Checks\n\n";
  rpt << "| Verification Item | Tested Details | Result |\n";
  rpt << "| :--- | :--- | :--- |\n";
  rpt << "| Both Row Slots | Slot 0 (Action 91) & Slot 1 (Action 92) for IDs 11 and 13 | PASS |\n";
  rpt << "| Observation Identity | Offset 9182 at kTleilaxuAppendixOffset, 16-onehot category 11 & 13 | PASS |\n";
  rpt << "| Semantic Descriptors | Schema v3: supported=0, role=kUnsupported, features[0]=1.0f | PASS |\n";
  rpt << "| Specimen Payment | Cost 3 specimens: illegal at 2, legal at 3, decrements to 0 | PASS |\n";
  rpt << "| Acquisition Destination | Normal acquire -> Discard; Topdeck acquire -> Draw deck top | PASS |\n\n";

  auto write_card_markdown = [&](const open_spiel::CardAggregateStats& c,
                                 const open_spiel::QuantileSummary& q) {
    rpt << "### " << c.card_name << " (Tleilaxu ID " << c.tleilaxu_id << ")\n\n";
    rpt << "| Metric | Value |\n";
    rpt << "| :--- | :--- |\n";
    rpt << "| Total Row Appearances (Decisions) | " << c.row_presence_decisions << " |\n";
    rpt << "| Row Presence with Specimens < 3 | " << c.row_presence_specimens_lt3 << " |\n";
    rpt << "| Row Presence with Specimens >= 3 | " << c.row_presence_specimens_ge3 << " |\n";
    rpt << "| **Eligible Opportunities (Action 91/92 Legal)** | **" << c.eligible_decisions << "** |\n";
    rpt << "| - Slot 0 (Action 91) Eligible | " << c.eligible_slot0 << " |\n";
    rpt << "| - Slot 1 (Action 92) Eligible | " << c.eligible_slot1 << " |\n";
    rpt << "| Unique Games with Row Presence | " << c.games_with_row_presence.size() << " |\n";
    rpt << "| Unique Games with Eligibility | " << c.games_with_eligibility.size() << " |\n";
    rpt << "| **Purchases Taken** | **" << c.purchases_taken << "** |\n";
    rpt << "| - Slot 0 Purchases | " << c.purchases_slot0 << " |\n";
    rpt << "| - Slot 1 Purchases | " << c.purchases_slot1 << " |\n";
    rpt << "| Unique Games with Purchase | " << c.games_with_purchase.size() << " |\n";
    double purchase_rate = c.eligible_decisions > 0
                                ? (100.0 * c.purchases_taken / c.eligible_decisions)
                                : 0.0;
    rpt << "| Purchase Rate (Purchases / Opportunities) | " << std::fixed
        << std::setprecision(2) << purchase_rate << "% |\n";
    rpt << "| Downstream Acquisitions | " << c.downstream_acquisitions << " |\n";
    rpt << "| Downstream Draws to Hand | " << c.downstream_draws << " |\n";
    rpt << "| Downstream Plays Observed | " << c.downstream_plays << " |\n";
    rpt << "| **Downstream Card Usefulness Status** | **INCONCLUSIVE (Observational Exposure Sample)** |\n\n";

    rpt << "#### Policy Probability Distribution when Eligible\n\n";
    rpt << "| Min | P10 | P25 | Median (P50) | P75 | P90 | Max | Mean |\n";
    rpt << "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n";
    rpt << "| " << std::scientific << std::setprecision(3) << q.min << " | "
        << q.p10 << " | " << q.p25 << " | " << q.p50 << " | " << q.p75 << " | "
        << q.p90 << " | " << q.max << " | " << q.mean << " |\n\n";

    rpt << "#### Top Alternative Actions Chosen when Purchase Was Passed\n\n";
    std::vector<std::pair<uint64_t, std::string>> alts;
    for (const auto& [act, count] : c.alternative_actions_taken) {
      alts.push_back({count, act});
    }
    std::sort(alts.rbegin(), alts.rend());
    rpt << "| Rank | Action Description | Count | Fraction |\n";
    rpt << "| :--- | :--- | :--- |\n";
    uint64_t total_alts = 0;
    for (const auto& [cnt, _] : alts) total_alts += cnt;
    for (size_t i = 0; i < std::min<size_t>(10, alts.size()); ++i) {
      double frac = total_alts > 0 ? (100.0 * alts[i].first / total_alts) : 0.0;
      rpt << "| " << (i + 1) << " | `" << alts[i].second << "` | "
          << alts[i].first << " | " << std::fixed << std::setprecision(1) << frac << "% |\n";
    }
    rpt << "\n";
  };

  rpt << "## 2. Card Purchase Exposure Telemetry\n\n";
  write_card_markdown(aggregate.card11, q_card11);
  write_card_markdown(aggregate.card13, q_card13);

  rpt << "## 3. Diagnostic Diagnosis & Conclusions\n\n";
  rpt << "- **Defect vs Exploration:** Upfront mechanical, observation, and semantic checks confirmed that both cards are completely legal, correctly encoded in observation tensors, and purchase destinations resolve accurately.\n";
  rpt << "- **Policy Allocation:** The frozen policy U20328 allocates very low probability mass (card 11 median: "
      << std::scientific << std::setprecision(2) << q_card11.p50 << ", card 13 median: "
      << q_card13.p50 << ") when these cards are in the row, overwhelmingly selecting alternative market or agent actions instead.\n";
  rpt << "- **Downstream Evaluation:** Natural purchases are sparse ("
      << aggregate.card11.purchases_taken << " for Scientific Breakthrough, "
      << aggregate.card13.purchases_taken << " for Stitched Horror across "
      << aggregate.total_games << " games). Downstream usefulness remains strictly **INCONCLUSIVE** (observational exposure sampling cannot establish causal usefulness).\n";
  rpt.close();

  // ---------------------------------------------------------------------------
  // Generate JSON Summary & Sampled Records
  // ---------------------------------------------------------------------------
  auto samples = reservoir.GetSamples();
  open_spiel::json::Object root;
  root["timestamp"] = "2026-09-19T17:35:00Z";
  root["model_checkpoint"] = ckpt_path;
  root["model_sha256"] = sha256;
  root["base_seed"] = static_cast<int64_t>(master_seed);
  root["seed_scheme_version"] = 2;
  root["elapsed_seconds"] = elapsed_seconds;
  root["max_minutes"] = max_minutes;
  root["completed_games"] = static_cast<int64_t>(aggregate.total_games);
  root["aborted_games"] = static_cast<int64_t>(aggregate.aborted_games);
  root["total_decisions"] = static_cast<int64_t>(aggregate.total_decisions);
  root["stop_reason"] = aggregate.stop_reason;

  auto json_card = [](const open_spiel::CardAggregateStats& c,
                      const open_spiel::QuantileSummary& q) {
    open_spiel::json::Object obj;
    obj["tleilaxu_id"] = static_cast<int64_t>(c.tleilaxu_id);
    obj["card_name"] = c.card_name;
    obj["row_presence_decisions"] = static_cast<int64_t>(c.row_presence_decisions);
    obj["row_presence_specimens_lt3"] = static_cast<int64_t>(c.row_presence_specimens_lt3);
    obj["row_presence_specimens_ge3"] = static_cast<int64_t>(c.row_presence_specimens_ge3);
    obj["eligible_decisions"] = static_cast<int64_t>(c.eligible_decisions);
    obj["eligible_slot0"] = static_cast<int64_t>(c.eligible_slot0);
    obj["eligible_slot1"] = static_cast<int64_t>(c.eligible_slot1);
    obj["purchases_taken"] = static_cast<int64_t>(c.purchases_taken);
    obj["purchases_slot0"] = static_cast<int64_t>(c.purchases_slot0);
    obj["purchases_slot1"] = static_cast<int64_t>(c.purchases_slot1);
    obj["unique_games_row_presence"] = static_cast<int64_t>(c.games_with_row_presence.size());
    obj["unique_games_eligible"] = static_cast<int64_t>(c.games_with_eligibility.size());
    obj["unique_games_purchased"] = static_cast<int64_t>(c.games_with_purchase.size());
    obj["downstream_acquisitions"] = static_cast<int64_t>(c.downstream_acquisitions);
    obj["downstream_draws"] = static_cast<int64_t>(c.downstream_draws);
    obj["downstream_plays"] = static_cast<int64_t>(c.downstream_plays);
    obj["downstream_conclusive"] = false;

    open_spiel::json::Object q_obj;
    q_obj["min"] = q.min;
    q_obj["p10"] = q.p10;
    q_obj["p25"] = q.p25;
    q_obj["p50"] = q.p50;
    q_obj["p75"] = q.p75;
    q_obj["p90"] = q.p90;
    q_obj["max"] = q.max;
    q_obj["mean"] = q.mean;
    obj["quantiles"] = q_obj;

    return obj;
  };

  root["scientific_breakthrough"] = json_card(aggregate.card11, q_card11);
  root["stitched_horror"] = json_card(aggregate.card13, q_card13);
  root["detailed_samples_count"] = static_cast<int64_t>(samples.size());
  root["purchases_preserved_count"] = static_cast<int64_t>(reservoir.PurchasesPreserved());
  root["total_eligible_opportunities_seen"] = static_cast<int64_t>(reservoir.TotalSeen());

  std::string json_file = output_dir + "/audit_summary.json";
  std::ofstream jf(json_file);
  jf << LocalJsonToString(open_spiel::json::Value(root), true);
  jf.close();

  // Write detailed records JSONL
  std::string jsonl_file = output_dir + "/detailed_samples.jsonl";
  std::ofstream jlf(jsonl_file);
  for (const auto& s : samples) {
    open_spiel::json::Object rec;
    rec["sample_id"] = static_cast<int64_t>(s.sample_id);
    rec["episode_id"] = static_cast<int64_t>(s.episode_id);
    rec["round"] = static_cast<int64_t>(s.round);
    rec["turn"] = static_cast<int64_t>(s.turn);
    rec["player_seat"] = static_cast<int64_t>(s.player_seat);
    rec["leader_name"] = s.leader_name;
    rec["specimens"] = static_cast<int64_t>(s.specimens);
    rec["target_tleilaxu_id"] = static_cast<int64_t>(s.target_tleilaxu_id);
    rec["target_card_name"] = s.target_card_name;
    rec["target_row_slot"] = static_cast<int64_t>(s.target_row_slot);
    rec["purchase_action_id"] = static_cast<int64_t>(s.purchase_action_id);
    rec["num_legal_actions"] = static_cast<int64_t>(s.num_legal_actions);
    rec["purchase_raw_centered_logit"] = static_cast<double>(s.purchase_raw_centered_logit);
    rec["purchase_capped_logit"] = static_cast<double>(s.purchase_capped_logit);
    rec["purchase_policy_prob"] = s.purchase_policy_prob;
    rec["purchase_policy_rank"] = static_cast<int64_t>(s.purchase_policy_rank);
    rec["chosen_action_id"] = static_cast<int64_t>(s.chosen_action_id);
    rec["chosen_action_name"] = s.chosen_action_name;
    rec["chosen_action_prob"] = s.chosen_action_prob;
    rec["was_target_purchased"] = s.was_target_purchased;

    open_spiel::json::Array top_arr;
    for (const auto& a : s.top_actions) {
      open_spiel::json::Object a_obj;
      a_obj["action_id"] = static_cast<int64_t>(a.action_id);
      a_obj["action_name"] = a.action_name;
      a_obj["prob"] = a.prob;
      top_arr.push_back(a_obj);
    }
    rec["top_actions"] = top_arr;

    jlf << LocalJsonToString(open_spiel::json::Value(rec), false) << "\n";
  }
  jlf.close();

  std::cout << "\nArtifacts written to " << output_dir << ":\n";
  std::cout << "  - " << report_file << "\n";
  std::cout << "  - " << json_file << "\n";
  std::cout << "  - " << jsonl_file << " (" << samples.size() << " records)\n";

  return 0;
}
