#ifndef OPEN_SPIEL_EXAMPLES_DUNE_PPO_TRAINING_REPLAY_H_
#define OPEN_SPIEL_EXAMPLES_DUNE_PPO_TRAINING_REPLAY_H_

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <functional>
#include <iosfwd>
#include <map>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "open_spiel/abseil-cpp/absl/strings/str_format.h"
#include "open_spiel/spiel.h"

#ifdef OPEN_SPIEL_BUILD_WITH_LIBTORCH
#include "dune_online_search_collector.h"
#endif

namespace open_spiel {

#ifdef OPEN_SPIEL_BUILD_WITH_LIBTORCH
struct R765DiagProbeItem {
  uint64_t root_uid = 0;
  std::vector<float> observation;
  std::vector<Action> legal_actions;
  dune_semantic::CandidateActionData candidate_data;
  Action a_star = kInvalidAction;
  Action a_ref = kInvalidAction;
  double t0_log_odds = 0.0;
};

struct R765PostEvictReadout {
  int64_t active_train_cohorts = 0;
  int64_t active_diag_cohorts = 0;
  bool has_final_age_cohort_this_update = false;
  bool final_age_full_coverage = true;
  int64_t final_age_visited_rows = 0;
  int64_t final_age_total_rows = 0;
  int64_t post_evict_5u_roots = 0;
  double post_evict_5u_d_final_denom = 0.0;
  double post_evict_5u_d_cur = 0.0;
  double post_evict_5u_ret = 0.0;
  int64_t post_evict_10u_roots = 0;
  double post_evict_10u_d_final_denom = 0.0;
  double post_evict_10u_d_cur = 0.0;
  double post_evict_10u_ret = 0.0;
};

class R765PreserveWindowGuard;

class R765ReplayCohortManager {
 public:
  struct TrainCohort {
    int birth_update = 0;
    int age = 0;
    std::vector<SearchTrainingExample> examples;
  };

  struct DiagCohort {
    int birth_update = 0;
    int expected_final_age = 0;
    int last_presented_update = -1;
    bool final_age_full_coverage = false;
    std::vector<R765DiagProbeItem> probes;
    std::map<uint64_t, double> root_d_final_live;
  };

  struct BatchMappingItem {
    int birth_update = 0;
    size_t row_idx_in_cohort = 0;
    int age = 0;
  };

  R765ReplayCohortManager() = default;

  void AdvanceAndBuildUpdateBatch(
      int global_update,
      const std::vector<SearchTrainingExample>& fresh_examples,
      double fresh_coef,
      double replay_coef,
      int window_k,
      uint64_t shuffle_seed,
      bool do_shuffle,
      std::vector<SearchTrainingExample>* out_combined) {
    out_combined->clear();
    last_batch_map_.clear();
    const int eff_k = std::max(1, window_k);

    for (auto& cohort : train_cohorts_) {
      cohort.age += 1;
    }
    train_cohorts_.erase(
        std::remove_if(train_cohorts_.begin(), train_cohorts_.end(),
                       [eff_k](const TrainCohort& c) { return c.age >= eff_k; }),
        train_cohorts_.end());

    if (!fresh_examples.empty()) {
      train_cohorts_.push_back(TrainCohort{global_update, /*age=*/0, fresh_examples});
    }

    for (const auto& cohort : train_cohorts_) {
      for (size_t r = 0; r < cohort.examples.size(); ++r) {
        SearchTrainingExample ex = cohort.examples[r];
        ex.replay_age = cohort.age;
        ex.row_loss_weight = (cohort.age == 0) ? fresh_coef : replay_coef;
        out_combined->push_back(std::move(ex));
        last_batch_map_.push_back({cohort.birth_update, r, cohort.age});
      }
    }

    if (do_shuffle && out_combined->size() > 1) {
      std::mt19937_64 rng(shuffle_seed);
      for (size_t i = out_combined->size() - 1; i > 0; --i) {
        std::uniform_int_distribution<size_t> dist(0, i);
        size_t j = dist(rng);
        if (i != j) {
          std::swap((*out_combined)[i], (*out_combined)[j]);
          std::swap(last_batch_map_[i], last_batch_map_[j]);
        }
      }
    }
  }

  R765PostEvictReadout FinalizeUpdateAndEvaluateRetention(
      int global_update,
      int window_k,
      const std::vector<R765DiagProbeItem>& fresh_probes,
      const std::vector<uint8_t>& aux_visited_mask,
      const std::function<std::vector<std::vector<double>>(
          const std::vector<const std::vector<float>*>&,
          const std::vector<const dune_semantic::CandidateActionData*>&,
          const std::vector<const std::vector<Action>*>&)>& eval_probs_fn) {
    R765PostEvictReadout readout;
    const int eff_k = std::max(1, window_k);
    const int expected_final_age = eff_k - 1;

    if (!fresh_probes.empty()) {
      DiagCohort dc;
      dc.birth_update = global_update;
      dc.expected_final_age = expected_final_age;
      dc.last_presented_update = -1;
      dc.final_age_full_coverage = false;
      dc.probes = fresh_probes;
      diag_cohorts_.push_back(std::move(dc));
    }

    std::map<int, int64_t> visited_counts;
    for (size_t i = 0; i < last_batch_map_.size(); ++i) {
      if (i < aux_visited_mask.size() && aux_visited_mask[i] != 0) {
        visited_counts[last_batch_map_[i].birth_update]++;
      }
    }

    for (const auto& tc : train_cohorts_) {
      int64_t v_count = 0;
      auto it = visited_counts.find(tc.birth_update);
      if (it != visited_counts.end()) {
        v_count = it->second;
      }
      DiagCohort* matching_diag = nullptr;
      for (auto& dc : diag_cohorts_) {
        if (dc.birth_update == tc.birth_update) {
          matching_diag = &dc;
          break;
        }
      }
      if (v_count > 0 && matching_diag != nullptr) {
        matching_diag->last_presented_update = global_update;
      }
      if (tc.age == expected_final_age && !tc.examples.empty()) {
        readout.has_final_age_cohort_this_update = true;
        readout.final_age_visited_rows = v_count;
        readout.final_age_total_rows = static_cast<int64_t>(tc.examples.size());
        readout.final_age_full_coverage = (v_count == static_cast<int64_t>(tc.examples.size()));
        if (matching_diag != nullptr) {
          matching_diag->final_age_full_coverage = readout.final_age_full_coverage;
        }
      }
    }

    auto eval_root_delta = [&](const std::vector<R765DiagProbeItem>& probes) -> std::map<uint64_t, double> {
      std::map<uint64_t, double> per_root;
      if (probes.empty() || !eval_probs_fn) return per_root;
      std::vector<const std::vector<float>*> obs_ptrs;
      std::vector<const dune_semantic::CandidateActionData*> cads;
      std::vector<const std::vector<Action>*> legals;
      obs_ptrs.reserve(probes.size());
      cads.reserve(probes.size());
      legals.reserve(probes.size());
      for (const auto& pr : probes) {
        obs_ptrs.push_back(&pr.observation);
        cads.push_back(&pr.candidate_data);
        legals.push_back(&pr.legal_actions);
      }
      auto probs = eval_probs_fn(obs_ptrs, cads, legals);
      std::map<uint64_t, std::pair<double, int>> accum;
      for (size_t i = 0; i < probes.size(); ++i) {
        const auto& pr = probes[i];
        int s_idx = -1, r_idx = -1;
        for (size_t a = 0; a < pr.legal_actions.size(); ++a) {
          if (pr.legal_actions[a] == pr.a_star) s_idx = static_cast<int>(a);
          if (pr.legal_actions[a] == pr.a_ref) r_idx = static_cast<int>(a);
        }
        if (s_idx >= 0 && r_idx >= 0 && i < probs.size() &&
            s_idx < static_cast<int>(probs[i].size()) &&
            r_idx < static_cast<int>(probs[i].size())) {
          double p_star = probs[i][s_idx];
          double p_ref = probs[i][r_idx];
          double cur_lo = std::log(std::max(1e-12, p_star)) - std::log(std::max(1e-12, p_ref));
          accum[pr.root_uid].first += (cur_lo - pr.t0_log_odds);
          accum[pr.root_uid].second += 1;
        }
      }
      for (const auto& kv : accum) {
        if (kv.second.second > 0) {
          per_root[kv.first] = kv.second.first / static_cast<double>(kv.second.second);
        }
      }
      return per_root;
    };

    for (auto& dc : diag_cohorts_) {
      if (dc.last_presented_update == global_update) {
        dc.root_d_final_live = eval_root_delta(dc.probes);
      }
    }

    int64_t roots_5u_count = 0;
    double sum_5u_denom = 0.0;
    double sum_5u_cur = 0.0;
    for (auto& dc : diag_cohorts_) {
      if (dc.last_presented_update > 0 && global_update == dc.last_presented_update + 5) {
        auto cur_deltas = eval_root_delta(dc.probes);
        for (const auto& kv : cur_deltas) {
          uint64_t uid = kv.first;
          double d_cur = kv.second;
          auto fit = dc.root_d_final_live.find(uid);
          if (fit != dc.root_d_final_live.end()) {
            roots_5u_count++;
            sum_5u_denom += fit->second;
            sum_5u_cur += d_cur;
          }
        }
      }
    }
    readout.post_evict_5u_roots = roots_5u_count;
    if (roots_5u_count > 0) {
      readout.post_evict_5u_d_final_denom = sum_5u_denom / static_cast<double>(roots_5u_count);
      readout.post_evict_5u_d_cur = sum_5u_cur / static_cast<double>(roots_5u_count);
      if (std::abs(readout.post_evict_5u_d_final_denom) > 1e-8) {
        readout.post_evict_5u_ret = readout.post_evict_5u_d_cur / readout.post_evict_5u_d_final_denom;
      }
    }

    int64_t roots_10u_count = 0;
    double sum_10u_denom = 0.0;
    double sum_10u_cur = 0.0;
    for (auto& dc : diag_cohorts_) {
      if (dc.last_presented_update > 0 && global_update == dc.last_presented_update + 10) {
        auto cur_deltas = eval_root_delta(dc.probes);
        for (const auto& kv : cur_deltas) {
          uint64_t uid = kv.first;
          double d_cur = kv.second;
          auto fit = dc.root_d_final_live.find(uid);
          if (fit != dc.root_d_final_live.end()) {
            roots_10u_count++;
            sum_10u_denom += fit->second;
            sum_10u_cur += d_cur;
          }
        }
      }
    }
    readout.post_evict_10u_roots = roots_10u_count;
    if (roots_10u_count > 0) {
      readout.post_evict_10u_d_final_denom = sum_10u_denom / static_cast<double>(roots_10u_count);
      readout.post_evict_10u_d_cur = sum_10u_cur / static_cast<double>(roots_10u_count);
      if (std::abs(readout.post_evict_10u_d_final_denom) > 1e-8) {
        readout.post_evict_10u_ret = readout.post_evict_10u_d_cur / readout.post_evict_10u_d_final_denom;
      }
    }

    diag_cohorts_.erase(
        std::remove_if(diag_cohorts_.begin(), diag_cohorts_.end(),
                       [global_update](const DiagCohort& dc) {
                         if (dc.last_presented_update > 0) {
                           return global_update >= dc.last_presented_update + 10;
                         } else {
                           return global_update >= dc.birth_update + dc.expected_final_age + 10;
                         }
                       }),
        diag_cohorts_.end());

    readout.active_train_cohorts = static_cast<int64_t>(train_cohorts_.size());
    readout.active_diag_cohorts = static_cast<int64_t>(diag_cohorts_.size());
    return readout;
  }

  int64_t active_train_cohorts() const { return static_cast<int64_t>(train_cohorts_.size()); }
  int64_t active_diag_cohorts() const { return static_cast<int64_t>(diag_cohorts_.size()); }
  const std::vector<TrainCohort>& train_cohorts() const { return train_cohorts_; }
  const std::vector<DiagCohort>& diag_cohorts() const { return diag_cohorts_; }

  std::vector<uint8_t> FreshCohortVisitedMask(
      int global_update, size_t fresh_count,
      const std::vector<uint8_t>& batch_visited_mask) const;
  bool SaveState(const std::string& path,
                 const R765PreserveWindowGuard* preserve_guard = nullptr) const;
  bool LoadState(const std::string& path,
                 R765PreserveWindowGuard* preserve_guard = nullptr);

 private:
  std::vector<TrainCohort> train_cohorts_;
  std::vector<DiagCohort> diag_cohorts_;
  std::vector<BatchMappingItem> last_batch_map_;
};

class R765PreserveWindowGuard {
 public:
  struct WindowEntry {
    int64_t preserved = 0;
    int64_t eligible = 0;
  };

  void Reset() {
    window_.clear();
  }

  bool PushAndCheck(int64_t preserved, int64_t eligible, double min_rate,
                    bool all_metrics_finite, double* out_pooled_rate,
                    int64_t* out_pooled_preserved, int64_t* out_pooled_eligible,
                    std::string* out_abort_reason) {
    if (out_abort_reason) out_abort_reason->clear();
    if (out_pooled_rate) *out_pooled_rate = -1.0;
    if (out_pooled_preserved) *out_pooled_preserved = 0;
    if (out_pooled_eligible) *out_pooled_eligible = 0;

    if (!all_metrics_finite || preserved < 0 || eligible < 0 || preserved > eligible) {
      if (out_abort_reason) *out_abort_reason = "nonfinite_or_invalid_telemetry";
      return false;
    }

    window_.push_back({preserved, eligible});
    while (window_.size() > 5) {
      window_.pop_front();
    }

    int64_t pooled_p = 0;
    int64_t pooled_e = 0;
    for (const auto& w : window_) {
      pooled_p += w.preserved;
      pooled_e += w.eligible;
    }
    if (out_pooled_preserved) *out_pooled_preserved = pooled_p;
    if (out_pooled_eligible) *out_pooled_eligible = pooled_e;

    double pooled_rate = (pooled_e > 0) ? (static_cast<double>(pooled_p) / static_cast<double>(pooled_e)) : -1.0;
    if (out_pooled_rate) *out_pooled_rate = pooled_rate;

    if (min_rate > 0.0 && window_.size() >= 5) {
      if (pooled_e == 0) {
        if (out_abort_reason) *out_abort_reason = "empty_5u_preserve_window";
        return false;
      }
      if (pooled_rate < min_rate) {
        if (out_abort_reason) {
          *out_abort_reason = absl::StrFormat(
              "pooled_5u_preserve_rate_breach: rate %.4f < min_rate %.4f (%lld/%lld preserved)",
              pooled_rate, min_rate, static_cast<long long>(pooled_p), static_cast<long long>(pooled_e));
        }
        return false;
      }
    }
    return true;
  }

  const std::deque<WindowEntry>& window() const { return window_; }

  bool SaveState(std::ostream& out) const;
  bool LoadState(std::istream& in);
  bool SaveState(const std::string& path) const;
  bool LoadState(const std::string& path);

 private:
  std::deque<WindowEntry> window_;
};
#endif

class CombatCreditAccumulator {
public:
  struct Event {
    int transition_index;
    int strength_delta;
  };

  void Clear(int player) {
    events_[player].clear();
  }

  void ClearAll() {
    for (int p = 0; p < 4; ++p) {
      events_[p].clear();
    }
  }

  void RecordDeployment(int player, int transition_index, int delta) {
    if (delta > 0) {
      events_[player].push_back({transition_index, delta});
    }
  }

  const std::vector<Event>& GetEvents(int player) const {
    return events_[player];
  }

  int GetTotalInvestment(int player) const {
    int total = 0;
    for (const auto& ev : events_[player]) {
      total += ev.strength_delta;
    }
    return total;
  }

private:
  std::vector<Event> events_[4];
};

}  // namespace open_spiel

#endif  // OPEN_SPIEL_EXAMPLES_DUNE_PPO_TRAINING_REPLAY_H_
