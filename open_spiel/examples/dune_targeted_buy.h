#ifndef OPEN_SPIEL_EXAMPLES_DUNE_TARGETED_BUY_H_
#define OPEN_SPIEL_EXAMPLES_DUNE_TARGETED_BUY_H_

#include <atomic>
#include <cstdint>
#include <cmath>
#include <vector>
#include <string>
#include <random>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <mutex>
#include <filesystem>

#include "open_spiel/abseil-cpp/absl/strings/str_format.h"
#include "open_spiel/games/dune_imperium/dune_imperium.h"
#include "open_spiel/games/dune_imperium/dune_imperium_cards.h"
#include "open_spiel/games/dune_imperium/dune_imperium_common.h"
#include "open_spiel/games/dune_imperium/dune_imperium_content.h"
#include "open_spiel/games/dune_imperium/dune_imperium_content_generated.h"
#include "open_spiel/spiel.h"

namespace open_spiel {
namespace dune_targeted_buy {

inline constexpr int kScientificBreakthroughTleilaxuId = 11;
inline constexpr int kScientificBreakthroughCardId = 118;
inline constexpr int kStitchedHorrorTleilaxuId = 13;
inline constexpr int kStitchedHorrorCardId = 120;
inline constexpr int kDefaultMaxRound = 6;
inline constexpr int kDefaultQuotaPerCard = 8;

// Thread-safe tracker for targeted-buy exploration across an update's rollout games.
struct TargetedBuyTracker {
  bool enabled = false;
  int quota_card11 = kDefaultQuotaPerCard;
  int quota_card13 = kDefaultQuotaPerCard;
  int max_round = kDefaultMaxRound;

  // Achieved distinct-game forced purchases in this update
  std::atomic<int> achieved_games_card11{0};
  std::atomic<int> achieved_games_card13{0};

  // Available opportunities for designated exploration seat in eligible rounds (<= max_round)
  std::atomic<int> designated_opps_card11{0};
  std::atomic<int> designated_opps_card13{0};

  // Total opportunities across all seats / rounds
  std::atomic<int> total_opps_card11{0};
  std::atomic<int> total_opps_card13{0};

  void Reset(bool is_enabled, int q11 = kDefaultQuotaPerCard, int q13 = kDefaultQuotaPerCard, int max_r = kDefaultMaxRound) {
    enabled = is_enabled;
    quota_card11 = q11;
    quota_card13 = q13;
    max_round = max_r;
    achieved_games_card11.store(0, std::memory_order_relaxed);
    achieved_games_card13.store(0, std::memory_order_relaxed);
    designated_opps_card11.store(0, std::memory_order_relaxed);
    designated_opps_card13.store(0, std::memory_order_relaxed);
    total_opps_card11.store(0, std::memory_order_relaxed);
    total_opps_card13.store(0, std::memory_order_relaxed);
  }

  // Atomically try to reserve a quota slot for card 11 (Scientific Breakthrough).
  // Returns true if reserved, false if quota already reached.
  bool TryReserveForcedBuyCard11() {
    int cur = achieved_games_card11.load(std::memory_order_relaxed);
    while (cur < quota_card11) {
      if (achieved_games_card11.compare_exchange_weak(cur, cur + 1,
                                                      std::memory_order_acq_rel,
                                                      std::memory_order_relaxed)) {
        return true;
      }
    }
    return false;
  }

  // Atomically try to reserve a quota slot for card 13 (Stitched Horror).
  // Returns true if reserved, false if quota already reached.
  bool TryReserveForcedBuyCard13() {
    int cur = achieved_games_card13.load(std::memory_order_relaxed);
    while (cur < quota_card13) {
      if (achieved_games_card13.compare_exchange_weak(cur, cur + 1,
                                                      std::memory_order_acq_rel,
                                                      std::memory_order_relaxed)) {
        return true;
      }
    }
    return false;
  }
};

// Global tracker instance shared across worker threads within each training update.
inline TargetedBuyTracker g_targeted_buy_tracker;

// Record structure for native minibatch normalization diagnostic logging.
struct ForcedNormalizationRecord {
  int update = 0;
  int epoch = 0;
  int minibatch = 0;
  int card_id = 0;
  int round = 0;
  float raw_gae = 0.0f;
  float mc_return = 0.0f;
  float value = 0.0f;
  float full_return_minus_value = 0.0f;
  float mb_mean = 0.0f;
  float mb_std = 0.0f;
  float norm_gae = 0.0f;
  float norm_full_return = 0.0f;
  float raw_adv_used = 0.0f;
  float norm_adv_used = 0.0f;
};

inline std::mutex g_forced_norm_mutex;
inline std::vector<ForcedNormalizationRecord> g_forced_norm_records;
inline std::string g_forced_norm_csv_path;

inline void RecordForcedNormalization(
    int update, int epoch, int mb_index,
    int card_id, int round,
    float raw_gae, float raw_adv_used, float mc_return, float value,
    float mb_mean, float mb_std) {
  float full_return_minus_value = mc_return - value;
  float norm_gae = (raw_gae - mb_mean) / mb_std;
  float norm_full_return = (full_return_minus_value - mb_mean) / mb_std;
  float norm_adv_used = (raw_adv_used - mb_mean) / mb_std;

  ForcedNormalizationRecord rec{
      update, epoch, mb_index, card_id, round,
      raw_gae, mc_return, value, full_return_minus_value,
      mb_mean, mb_std, norm_gae, norm_full_return,
      raw_adv_used, norm_adv_used
  };

  {
    std::lock_guard<std::mutex> lock(g_forced_norm_mutex);
    g_forced_norm_records.push_back(rec);
  }

  std::cout << absl::StrFormat(
      "[TARGETED_BUY_FORCED_NORM] update=%d epoch=%d mb=%d card=%d (%s) round=%d | "
      "raw_gae=%+.4f full_ret=%+.4f val=%+.4f full_ret_minus_v=%+.4f | "
      "adv_used_raw=%+.4f adv_used_norm=%+.4f | "
      "mb_mean=%+.4f mb_std=%.4f | norm_gae=%+.4f norm_full_ret=%+.4f\n",
      update, epoch, mb_index, card_id,
      (card_id == kScientificBreakthroughTleilaxuId ? "Scientific Breakthrough" : "Stitched Horror"),
      round, raw_gae, mc_return, value, full_return_minus_value,
      raw_adv_used, norm_adv_used,
      mb_mean, mb_std, norm_gae, norm_full_return);

  if (!g_forced_norm_csv_path.empty()) {
    std::lock_guard<std::mutex> lock(g_forced_norm_mutex);
    bool exists = std::filesystem::exists(g_forced_norm_csv_path);
    std::ofstream ofs(g_forced_norm_csv_path, std::ios::app);
    if (ofs.is_open()) {
      if (!exists || std::filesystem::file_size(g_forced_norm_csv_path) == 0) {
        ofs << "update,epoch,minibatch,card_id,card_name,round,raw_gae,mc_return,value,full_return_minus_value,mb_mean,mb_std,norm_gae,norm_full_return,raw_adv_used,norm_adv_used\n";
      }
      ofs << absl::StrFormat("%d,%d,%d,%d,%s,%d,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n",
                             update, epoch, mb_index, card_id,
                             (card_id == kScientificBreakthroughTleilaxuId ? "Scientific Breakthrough" : "Stitched Horror"),
                             round, raw_gae, mc_return, value, full_return_minus_value,
                             mb_mean, mb_std, norm_gae, norm_full_return,
                             raw_adv_used, norm_adv_used);
    }
  }
}

// Record structure for update-path telemetry (Step 3).
struct ForcedUpdateTelemetryRecord {
  int update = 0;
  int card_id = 0;
  int round = 0;
  float log_pi_collect = 0.0f;
  float log_pi_before = 0.0f;
  float log_pi_after = 0.0f;
  float ratio_pre = 0.0f;
  float adv_used_raw = 0.0f;
  float adv_used_norm = 0.0f;
  float gae_adv = 0.0f;
  float full_return_minus_val = 0.0f;
};

inline std::mutex g_forced_telemetry_mutex;
inline std::vector<ForcedUpdateTelemetryRecord> g_forced_telemetry_records;
inline std::string g_forced_telemetry_csv_path;

inline void RecordForcedUpdateTelemetry(
    int update, int card_id, int round,
    float log_pi_collect, float log_pi_before, float log_pi_after,
    float ratio_pre, float adv_used_raw, float adv_used_norm,
    float gae_adv, float full_return_minus_val) {
  ForcedUpdateTelemetryRecord rec{
      update, card_id, round,
      log_pi_collect, log_pi_before, log_pi_after,
      ratio_pre, adv_used_raw, adv_used_norm,
      gae_adv, full_return_minus_val
  };

  {
    std::lock_guard<std::mutex> lock(g_forced_telemetry_mutex);
    g_forced_telemetry_records.push_back(rec);
  }

  std::cout << absl::StrFormat(
      "[TARGETED_BUY_UPDATE_TELEMETRY] update=%d card=%d (%s) round=%d | "
      "log_pi_collect=%+.6f log_pi_before=%+.6f log_pi_after=%+.6f ratio_pre=%.6f | "
      "adv_used_raw=%+.6f adv_used_norm=%+.6f gae_adv=%+.6f full_ret_minus_v=%+.6f\n",
      update, card_id,
      (card_id == kScientificBreakthroughTleilaxuId ? "Scientific Breakthrough" : "Stitched Horror"),
      round, log_pi_collect, log_pi_before, log_pi_after, ratio_pre,
      adv_used_raw, adv_used_norm, gae_adv, full_return_minus_val);

  std::string csv_path = g_forced_telemetry_csv_path;
  if (csv_path.empty() && !g_forced_norm_csv_path.empty()) {
    std::filesystem::path p(g_forced_norm_csv_path);
    csv_path = (p.parent_path() / "targeted_buy_update_telemetry.csv").string();
  }

  if (!csv_path.empty()) {
    std::lock_guard<std::mutex> lock(g_forced_telemetry_mutex);
    bool exists = std::filesystem::exists(csv_path);
    std::ofstream ofs(csv_path, std::ios::app);
    if (ofs.is_open()) {
      if (!exists || std::filesystem::file_size(csv_path) == 0) {
        ofs << "update,card_id,card_name,round,log_pi_collect,log_pi_before,log_pi_after,ratio_pre,adv_used_raw,adv_used_norm,gae_adv,full_return_minus_value\n";
      }
      ofs << absl::StrFormat("%d,%d,%s,%d,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n",
                             update, card_id,
                             (card_id == kScientificBreakthroughTleilaxuId ? "Scientific Breakthrough" : "Stitched Horror"),
                             round, log_pi_collect, log_pi_before, log_pi_after, ratio_pre,
                             adv_used_raw, adv_used_norm, gae_adv, full_return_minus_val);
    }
  }
}


// Helper to find a legal Tleilaxu acquisition action for a target Tleilaxu card ID in the current state.
inline bool FindLegalTleilaxuBuyAction(
    const dune_imperium::DuneImperiumState& state,
    const std::vector<Action>& legal_actions,
    int target_tleilaxu_id,
    int* out_slot = nullptr,
    Action* out_action = nullptr) {
  const auto& row = state.GetTleilaxuRowForTesting();
  for (size_t slot = 0; slot < row.size() && slot < 2; ++slot) {
    if (row[slot] == target_tleilaxu_id) {
      Action buy_act = dune_imperium::kActionTleilaxuAcquire0 + 1 + static_cast<Action>(slot);
      for (Action a : legal_actions) {
        if (a == buy_act) {
          if (out_slot) *out_slot = static_cast<int>(slot);
          if (out_action) *out_action = buy_act;
          return true;
        }
      }
    }
  }
  return false;
}

// Telemetry structure split by card (0=SB, 1=SH) and round (1..10)
struct TargetedBuyTelemetry {
  uint64_t unforced_opps[2][11] = {};
  double unforced_sum_prob[2][11] = {};
  uint64_t unforced_buys[2][11] = {};

  uint64_t forced_buys[2][11] = {};
  double forced_sum_adv[2][11] = {};
  double unforced_sum_adv[2][11] = {};

  TargetedBuyTelemetry& operator+=(const TargetedBuyTelemetry& o) {
    for (int c = 0; c < 2; ++c) {
      for (int r = 0; r <= 10; ++r) {
        unforced_opps[c][r] += o.unforced_opps[c][r];
        unforced_sum_prob[c][r] += o.unforced_sum_prob[c][r];
        unforced_buys[c][r] += o.unforced_buys[c][r];
        forced_buys[c][r] += o.forced_buys[c][r];
        forced_sum_adv[c][r] += o.forced_sum_adv[c][r];
        unforced_sum_adv[c][r] += o.unforced_sum_adv[c][r];
      }
    }
    return *this;
  }
};

struct ForcingDecision {
  bool forced = false;
  int forced_card_tleilaxu_id = -1;
  Action forced_action = kInvalidAction;
  float old_log_prob = 0.0f;
  std::string rejection_reason;
};

template <typename Rng>
inline ForcingDecision DecideTargetedForcedBuy(
    TargetedBuyTracker& tracker,
    const dune_imperium::DuneImperiumState& state,
    int current_player,
    int designated_explore_seat,
    int cur_round,
    const std::vector<Action>& actions,
    const std::vector<double>& legal_probabilities,
    bool& episode_forced_card11,
    bool& episode_forced_card13,
    Rng& rng) {
  ForcingDecision decision;

  Action buy_act_11 = kInvalidAction;
  int slot_11 = -1;
  bool legal_11 = FindLegalTleilaxuBuyAction(
      state, actions, kScientificBreakthroughTleilaxuId, &slot_11, &buy_act_11);

  Action buy_act_13 = kInvalidAction;
  int slot_13 = -1;
  bool legal_13 = FindLegalTleilaxuBuyAction(
      state, actions, kStitchedHorrorTleilaxuId, &slot_13, &buy_act_13);

  if (current_player == designated_explore_seat) {
    if (legal_11) {
      tracker.total_opps_card11.fetch_add(1, std::memory_order_relaxed);
      if (cur_round <= tracker.max_round) {
        tracker.designated_opps_card11.fetch_add(1, std::memory_order_relaxed);
      }
    }
    if (legal_13) {
      tracker.total_opps_card13.fetch_add(1, std::memory_order_relaxed);
      if (cur_round <= tracker.max_round) {
        tracker.designated_opps_card13.fetch_add(1, std::memory_order_relaxed);
      }
    }
  }

  if (!tracker.enabled) {
    decision.rejection_reason = "disabled";
    return decision;
  }
  if (current_player != designated_explore_seat) {
    decision.rejection_reason = "seat_not_designated";
    return decision;
  }
  if (cur_round > tracker.max_round) {
    decision.rejection_reason = "round_exceeds_max";
    return decision;
  }
  if (!legal_11 && !legal_13) {
    decision.rejection_reason = "neither_card_legal";
    return decision;
  }

  bool try_11_first = (rng() % 2 == 0);
  auto try_force_11 = [&]() -> bool {
    if (legal_11 && !episode_forced_card11) {
      if (tracker.TryReserveForcedBuyCard11()) {
        decision.forced = true;
        decision.forced_action = buy_act_11;
        decision.forced_card_tleilaxu_id = kScientificBreakthroughTleilaxuId;
        episode_forced_card11 = true;
        return true;
      }
    }
    return false;
  };
  auto try_force_13 = [&]() -> bool {
    if (legal_13 && !episode_forced_card13) {
      if (tracker.TryReserveForcedBuyCard13()) {
        decision.forced = true;
        decision.forced_action = buy_act_13;
        decision.forced_card_tleilaxu_id = kStitchedHorrorTleilaxuId;
        episode_forced_card13 = true;
        return true;
      }
    }
    return false;
  };

  if (try_11_first) {
    if (!try_force_11()) {
      try_force_13();
    }
  } else {
    if (!try_force_13()) {
      try_force_11();
    }
  }

  if (decision.forced) {
    for (size_t act_i = 0; act_i < actions.size(); ++act_i) {
      if (actions[act_i] == decision.forced_action) {
        if (act_i < legal_probabilities.size()) {
          decision.old_log_prob = static_cast<float>(
              std::log(std::max(1e-12, legal_probabilities[act_i])));
        }
        break;
      }
    }
  } else {
    if (legal_11 && episode_forced_card11 && legal_13 && episode_forced_card13) {
      decision.rejection_reason = "both_cards_already_forced_in_episode";
    } else if (legal_11 && episode_forced_card11 && !legal_13) {
      decision.rejection_reason = "card11_already_forced_in_episode";
    } else if (legal_13 && episode_forced_card13 && !legal_11) {
      decision.rejection_reason = "card13_already_forced_in_episode";
    } else {
      decision.rejection_reason = "quota_exhausted";
    }
  }

  return decision;
}

}  // namespace dune_targeted_buy
}  // namespace open_spiel

#endif  // OPEN_SPIEL_EXAMPLES_DUNE_TARGETED_BUY_H_
