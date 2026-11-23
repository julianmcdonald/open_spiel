#ifndef OPEN_SPIEL_EXAMPLES_DUNE_PPO_TRAINING_TYPES_H_
#define OPEN_SPIEL_EXAMPLES_DUNE_PPO_TRAINING_TYPES_H_

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "open_spiel/abseil-cpp/absl/strings/string_view.h"
#include "open_spiel/spiel.h"
#include "dune_search_routing.h"

#ifdef OPEN_SPIEL_BUILD_WITH_LIBTORCH
#include <torch/torch.h>
#include <c10/util/Half.h>
#include <c10/core/Event.h>
#include <c10/core/impl/VirtualGuardImpl.h>
#include <chrono>
#include <optional>
#include "dune_network.h"
#include "dune_online_search_collector.h"
#endif

namespace open_spiel {

inline constexpr int kPrivilegedCriticAppendixSize = 3432;
inline constexpr int kPrivilegedCriticInformationStateSize = 12614;

#ifdef OPEN_SPIEL_BUILD_WITH_LIBTORCH
PolicyDistributionSample SampleTrainingPolicyDistribution(
    const State& state, std::mt19937_64* rng,
    const std::vector<float>& logits, const std::vector<Action>& legal_actions,
    std::vector<double>* legal_probabilities);
#endif

// Define PpoTransition here so it is shared.
struct PpoTransition {
  std::vector<float> state;
  std::vector<open_spiel::Action> legal_actions;
  int64_t action;
  float old_log_prob;
  float reward;
  float value;
  float advantage;
  float return_value;
  int player_id;
  uint64_t episode_id;
  // Populated ONLY by the opt-in numerical-parity diagnostic. Normal PPO
  // rollout rows leave the vector empty and the role at -1, so the training
  // path pays no per-row distribution storage cost.
  std::vector<float> behavior_legal_log_probs;
  // V3 hard-validity input: evaluator logits for legal actions only, in exact
  // state.LegalActions() order, captured BEFORE CPU centering/capping. Empty on
  // every normal PPO path and every pre-v3 transition.
  std::vector<float> behavior_raw_legal_logits;
  int decision_role = -1;
  int64_t behavior_physical_batch_id = -1;
  int32_t behavior_physical_batch_size = -1;
  int32_t behavior_physical_batch_row = -1;
  // Candidate action descriptors for the semantic action scorer.
  dune_semantic::CandidateActionData candidate_data;
  // Excludes this row from policy and entropy losses (used for D2 purchase exploration)
  bool mask_policy_loss = false;
  // Compact fp16 storage for privileged central critic features (3,432 floats, A2)
  std::vector<uint16_t> privileged_appendix;
  // High stakes strategic state flag for rail/anchor partitioning
  bool is_high_stakes = false;
  // Monte Carlo return (lambda=1, gamma=1) and trajectory third (0=first, 1=mid, 2=last) for value calibration
  float mc_return = 0.0f;
  int trajectory_third = -1;
  // Targeted buy exploration
  int targeted_buy_card = 0;
  int targeted_buy_round = 0;
  bool targeted_buy_forced = false;
  bool off_policy_sampled = false;
  float gae_advantage = 0.0f;
  float norm_advantage = 0.0f;
  int round = 0;
};

inline bool IsAcquisitionAction(Action a) {
  if (a >= dune_imperium::kActionBuyImperiumRow0 && a < dune_imperium::kActionBuyImperiumRow0 + 8) return true;
  if (a >= dune_imperium::kActionBuyHelenaReserve0 && a < dune_imperium::kActionBuyHelenaReserve0 + 8) return true;
  if (a == dune_imperium::kActionBuyReserveArrakisLiaison) return true;
  if (a == dune_imperium::kActionBuyReserveTheSpiceMustFlow) return true;
  if (a >= dune_imperium::kActionTechAcquire0 && a < dune_imperium::kActionTechAcquireSkip) return true;
  if (a >= dune_imperium::kActionTleilaxuAcquire0 && a < dune_imperium::kActionTleilaxuAcquireSkip) return true;
  if (a >= dune_imperium::kActionTechAcquireWithSolari0 && a < dune_imperium::kActionTechAcquireWithSolariSkip) return true;
  return false;
}

DuneDecisionRole ClassifyTransitionDecisionRole(const PpoTransition& t);

void StandardizeAdvantagesPhaseStratified(
    std::vector<PpoTransition>* batch,
    const std::vector<int64_t>& mc_indices,
    int64_t min_stratum_count = 2);

#ifdef OPEN_SPIEL_BUILD_WITH_LIBTORCH

inline uint16_t FloatToFp16(float f) {
  return c10::Half(f).x;
}
inline float Fp16ToFloat(uint16_t h) {
  return static_cast<float>(c10::Half(h, c10::Half::from_bits()));
}

// Summary of PRE-cap legal-centered |z| over one update's rollout decisions.
//
// Per decision, z = raw_logit - mean(raw_logit over legal actions) and the
// decision's statistic is max|z| over its legal actions, measured BEFORE the
// tanh soft-cap and centered exactly as CenterAndCapLegalLogits centers. The
// percentiles/max/fraction below are then taken OVER DECISIONS of that
// per-decision value, which is the same decision-level convention the frozen
// scripts/eval/logit_cap_audit.py uses (its frac_ge(maxz, N)).
//
// Domain: decisions with >= 2 legal actions only, matching the trainer's own
// `nontrivial` definition (legal_actions.size() > 1). A forced decision has
// z == 0 identically -- the centered value of a single logit is always zero --
// so including them would only dilute the percentiles toward 0 while telling
// us nothing. `decisions` records the denominator so this stays auditable.
struct PrecapAbszStats {
  int64_t decisions = 0;
  double p95 = 0.0;
  double p99 = 0.0;
  double max = 0.0;
  double frac_ge10 = 0.0;
};

// Summarizes per-decision max|z| values. Takes the vector by value and sorts
// it, so the result depends only on the multiset and not on the order the
// worker threads happened to append in.
PrecapAbszStats SummarizePrecapAbsz(std::vector<float> values);

// --- WO-PERF-TIMING (2026-08-16): per-update PPO phase attribution. ---------
//
// Populated only under --phase_timing_mode=phases. At the default `off` no
// timer is armed, no event is created and no clock is read, so the default
// path is bit-for-bit and instruction-for-instruction today's behavior.
//
// `device_s` is CUDA-event time: the interval between the bracket's two events
// EXECUTING on the compute stream, read once after a single end-of-update
// synchronise. On CPU it falls back to steady_clock. `host_s` is host wall
// time for the same bracket on both devices.
//
// READ device_s CAREFULLY. It is NOT "GPU busy time". Stream events are
// markers in the stream's own order, so the interval between them includes any
// period the stream sat IDLE waiting on the host. A phase that is host-bound
// -- the tensor-packing loop, or `per_param` gradient telemetry with its ~70
// `.item()` readbacks per minibatch, each draining the stream -- will report a
// large device_s that is mostly idle, not execution. Neither number alone is
// an attribution:
//
//   * host_s around an async phase measures LAUNCH latency, and the real cost
//     lands on whichever later phase happens to synchronise first;
//   * device_s around a host-bound phase measures stream idle.
//
// Both are reported so a reader can tell those two cases apart. sum(host_s) is
// the quantity comparable to the trainer's `PPO:` figure; `unattributed_host_s`
// is the declared residual. See §3.3 of
// docs/work_orders/WO_PERF_TIMING_INSTRUMENTATION_2026_08_16.md.
inline constexpr int kPhaseTimingNumPhases = 8;

// Ordered exactly as the work order's table, and as the sidecar's field list.
extern const char* const kPhaseTimingNames[kPhaseTimingNumPhases];

struct PpoPhaseTimings {
  bool enabled = false;   // --phase_timing_mode=phases
  bool cuda = false;      // device_s came from CUDA events, not steady_clock
  double device_s[kPhaseTimingNumPhases] = {0, 0, 0, 0, 0, 0, 0, 0};
  double host_s[kPhaseTimingNumPhases] = {0, 0, 0, 0, 0, 0, 0, 0};
  double rollout_hash_host_s = 0.0;
  double total_attributed_s = 0.0;       // sum(device_s)
  double total_host_attributed_s = 0.0;  // sum(host_s) + rollout_hash_host_s
  // Cost of the single end-of-update synchronise, and NOTHING ELSE. It is not
  // the instrument's total perturbation: the per-bracket cudaEventRecord calls
  // execute INSIDE the brackets they delimit, and the elapsedTime readback loop
  // runs after this. `events_recorded` sizes both of those so a reader can
  // bound them rather than assume sync_s covers them.
  double sync_s = 0.0;
  int64_t bracket_count = 0;    // timed brackets recorded this update
  int64_t events_recorded = 0;  // 2 per bracket on CUDA, 0 on CPU
};

// Phase indices, named so call sites read as the work order's §3.2 table.
enum PhaseIndex {
  kPhaseTensorPackH2D = 0,
  kPhaseDiagPrepass = 1,
  kPhasePpoForwardLoss = 2,
  kPhaseBackward = 3,
  kPhaseGradTelemetry = 4,
  kPhaseGradClip = 5,
  kPhaseOptimizerStep = 6,
  kPhaseScalarReads = 7,
};

// Brackets the eight phases. Declared here rather than hidden in the .cc so the
// bracket pairing, the accumulation and the CPU fallback are reachable from the
// test TU -- an untestable timer is how a wrong attribution table survives.
//
// When disabled every method returns immediately having touched nothing: no
// clock read, no event, no allocation. That is what makes the default path
// instruction-for-instruction today's behavior.
class PhaseTimer {
 public:
  PhaseTimer(bool enabled, torch::Device device);

  void Begin(int phase);
  void End(int phase);
  // ONE synchronise, then read every pair. Safe to call when disabled.
  void Finalize(PpoPhaseTimings* out);

 private:
  static constexpr size_t kNoEvent = static_cast<size_t>(-1);
  struct Pair {
    int phase;
    size_t start;
  };

  size_t AcquirePair();

  const bool enabled_;
  const bool cuda_;
  const torch::Device device_;
  std::optional<c10::impl::VirtualGuardImpl> impl_;

  std::vector<Pair> pairs_;
  size_t next_free_ = 0;
  size_t last_recorded_ = kNoEvent;
  // kNoEvent until Begin opens the phase, so an unmatched End is detectable
  // instead of indexing element 1 of a possibly-empty pool.
  size_t open_event_[kPhaseTimingNumPhases];

  std::chrono::steady_clock::time_point host_begin_[kPhaseTimingNumPhases];
  double host_s_[kPhaseTimingNumPhases] = {0, 0, 0, 0, 0, 0, 0, 0};
  int64_t bracket_count_ = 0;
  int64_t events_recorded_ = 0;
};

// RAII bracket, so an early break/return inside a phase cannot leak a half-open
// bracket and silently drop the phase from the attribution.
class PhaseScope {
 public:
  PhaseScope(PhaseTimer* t, int phase) : t_(t), phase_(phase) {
    t_->Begin(phase_);
  }
  ~PhaseScope() { t_->End(phase_); }
  PhaseScope(const PhaseScope&) = delete;
  PhaseScope& operator=(const PhaseScope&) = delete;

 private:
  PhaseTimer* t_;
  int phase_;
};

inline float ComputeHybridAuxSumScale(
    double search_loss_coef, int64_t slice_rows, int64_t denom_mb_rows,
    absl::string_view norm_mode = "sum_over_mb",
    double max_slice_coef = 0.24) {
  if (slice_rows <= 0) return 0.0f;
  if (norm_mode == "per_row_capped") {
    const double total_w = std::min(
        max_slice_coef, search_loss_coef * static_cast<double>(slice_rows));
    return static_cast<float>(total_w / static_cast<double>(slice_rows));
  }
  const double denom = std::max<double>(1.0, static_cast<double>(denom_mb_rows));
  return static_cast<float>(search_loss_coef / denom);
}

inline std::vector<float> ComputeWeightedHybridAuxRowWeights(
    const std::vector<double>& raw_weights,
    int64_t denom_mb_rows,
    absl::string_view norm_mode = "sum_over_mb",
    double max_slice_coef = 0.24,
    bool* out_cap_active = nullptr) {
  if (out_cap_active != nullptr) {
    *out_cap_active = false;
  }
  const size_t n = raw_weights.size();
  std::vector<float> eff(n, 0.0f);
  if (n == 0) {
    return eff;
  }
  if (norm_mode == "per_row_capped") {
    double sum_w = 0.0;
    for (double w : raw_weights) {
      sum_w += std::max(0.0, w);
    }
    double scale = 1.0;
    if (sum_w > max_slice_coef && sum_w > 0.0) {
      scale = std::max(0.0, max_slice_coef) / sum_w;
      if (out_cap_active != nullptr) {
        *out_cap_active = true;
      }
    }
    for (size_t i = 0; i < n; ++i) {
      eff[i] = static_cast<float>(std::max(0.0, raw_weights[i]) * scale);
    }
    return eff;
  }
  const double denom = std::max<double>(1.0, static_cast<double>(denom_mb_rows));
  for (size_t i = 0; i < n; ++i) {
    eff[i] = static_cast<float>(std::max(0.0, raw_weights[i]) / denom);
  }
  return eff;
}

struct PpoUpdateStats {
  double policy_loss = 0.0;
  double value_loss = 0.0;
  double entropy = 0.0;
  double approx_kl = 0.0;
  double clip_fraction = 0.0;
  double explained_variance = 0.0;
  int minibatches = 0;
  bool early_stopped = false;
  double grad_norm_sum = 0.0;
  int grad_norm_count = 0;

  // --- Phase A durable PPO-side canaries (log-only; no behavior change) ---
  // grad_norm_max is the per-update maximum of the SAME unclipped quantity
  // grad_norm_sum accumulates: torch clip_grad_norm_ returns the total norm as
  // measured BEFORE clipping, so both are pre-clip.
  double grad_norm_max = 0.0;
  // Epoch at which target_kl tripped the early stop, or -1 if it never did.
  // `early_stopped` already carries the boolean; this adds the index.
  int kl_early_stop_epoch = -1;
  // Set immediately before the fatal-exit on a nonfinite gradient norm. NOTE:
  // that path calls std::exit from inside TrainPpoUpdate, so a run that
  // survives always reports 0 here and the durable evidence of a nonfinite
  // abort is the "Fatal PPO gradient norm" line in run.log plus a short row
  // count. The analyzer treats all three as equivalent evidence.
  bool nonfinite_abort = false;

  // Pre-cap legal-centered |z|, overall and by decision role. Sourced from the
  // rollout (the behavior policy that generated the batch), not from the
  // minibatch forward passes, and copied in by the caller before the
  // diagnostics row is written.
  PrecapAbszStats precap_absz_all;
  PrecapAbszStats precap_absz_primary;       // DuneDecisionRole::kAgentPrimary
  PrecapAbszStats precap_absz_continuation;  // DuneDecisionRole::kAgentContinuation
  PrecapAbszStats precap_absz_purchase;      // DuneDecisionRole::kPurchase
  int64_t forced_buys = 0;                   // Purchase exploration forced buy count (D2)
  double purchase_grad_cosine = 0.0;         // Cosine similarity between PPO policy loss grads and MPO aux loss grads on scorer output params
  double purchase_grad_cosine_sum = 0.0;
  int purchase_grad_cosine_count = 0;
  double action_head_grad_cosine = 0.0;
  double action_head_grad_cosine_sum = 0.0;
  int action_head_grad_cosine_count = 0;
  double aux_t1_delta_log_odds_sum = 0.0;
  int64_t aux_t1_measured_rows = 0;

  // Mean of the unclipped per-minibatch gradient norms over this update.
  // Distinct from the WO-17 column `ppo_grad_norm_mean`, which is the
  // aux-comparison ppo-only decomposition and is populated ONLY when online
  // collection fed examples -- it reads 0.0 with collection off, so it cannot
  // serve as a general grad-norm canary.
  double GradNormMean() const {
    return grad_norm_count > 0 ? grad_norm_sum / grad_norm_count : 0.0;
  }

  // --- WO-1 Phase 5 scratch-anomaly bundle (log-only; no behavior change) ---
  // Fraction of minibatch samples whose critic output moved further than
  // ppo_clip_epsilon from its rollout-time value, i.e. the fraction the clipped
  // value loss actually clipped. Reads 0.0 when --ppo_clip_value_loss=false,
  // where no value clipping happens at all.
  double value_clip_fraction = 0.0;
  // Per-module pre-clip gradient L2 norms, averaged over executed minibatches.
  // Measured BEFORE clip_grad_norm_, which rescales gradients in place.
  // Grouping is by parameter-name prefix: `policy_head.*`, anything containing
  // `value_head` (which correctly catches `value_head2` in the nonlinear
  // configuration), and everything else as trunk (`input_layer.*`, `resN.*`).
  // Under --train_value_only the trunk/policy groups are frozen and read 0.0
  // rather than being absent.
  double policy_head_grad_norm_sum = 0.0;
  double value_head_grad_norm_sum = 0.0;
  double trunk_grad_norm_sum = 0.0;
  // PWO-5 section 16 gate 3 item 6: per-AUXILIARY-head gradient norms. Zero in
  // head-off arms, where the loss terms are not constructed and `.grad()` is
  // never defined for these parameters -- which is itself the check that the
  // short-circuit worked.
  double final_vp_head_grad_norm_sum = 0.0;
  double terminal_round_head_grad_norm_sum = 0.0;
  double next_own_action_head_grad_norm_sum = 0.0;
  int head_grad_norm_count = 0;
  double PolicyHeadGradNormMean() const {
    return head_grad_norm_count > 0
               ? policy_head_grad_norm_sum / head_grad_norm_count : 0.0;
  }
  double ValueHeadGradNormMean() const {
    return head_grad_norm_count > 0
               ? value_head_grad_norm_sum / head_grad_norm_count : 0.0;
  }
  double TrunkGradNormMean() const {
    return head_grad_norm_count > 0
               ? trunk_grad_norm_sum / head_grad_norm_count : 0.0;
  }
  double FinalVpHeadGradNormMean() const {
    return head_grad_norm_count > 0
               ? final_vp_head_grad_norm_sum / head_grad_norm_count : 0.0;
  }
  double TerminalRoundHeadGradNormMean() const {
    return head_grad_norm_count > 0
               ? terminal_round_head_grad_norm_sum / head_grad_norm_count : 0.0;
  }
  double NextOwnActionHeadGradNormMean() const {
    return head_grad_norm_count > 0
               ? next_own_action_head_grad_norm_sum / head_grad_norm_count : 0.0;
  }

  // Diagnostics (Phase 5)
  bool episode_ids_unique = true;
  double policy_kl_before = 0.0;
  double return_min = 0.0;
  double return_max = 0.0;
  double return_p50 = 0.0;
  double return_p95 = 0.0;
  double return_p99 = 0.0;
  double abs_return_p99 = 0.0;
  double fraction_targets_outside_1 = 0.0;
  double fraction_critic_near_1 = 0.0;
  int64_t total_transitions = 0;
  int64_t nontrivial_transitions = 0;
  int64_t forced_transitions = 0;
  // WO-PERF-1: how many nontrivial transitions the policy_kl_before statistic
  // was actually measured over this update.
  //
  //   -1  = NOT MEASURED this update (a cadenced-mode off-interval update, or
  //         value-only mode).
  //    0  = measured, and there were zero nontrivial transitions.
  //
  // Either way an unmeasured update can never be read as KL == 0. In
  // --diag_prepass_mode=full every update measures, so this equals the
  // nontrivial count (or 0 under --train_value_only, which skips the KL).
  //
  // (Corrected 2026-08-16. This comment documented 0 as the not-measured
  // sentinel, which WO-PERF-R2 had already changed to -1 in the .cc without
  // updating it here. Anything gating on `measured_transitions == 0` to mean
  // "not measured" would misclassify every genuinely-measured-but-empty
  // update. The implementation is dune_ppo_training_utils.cc:716-717 and its
  // comment at :594-597; this header now agrees with both.)
  int64_t measured_transitions = 0;
  std::vector<double> epoch_kls;
  std::string rollout_hash;

  // --- Phase 18B combined optimization (online search aux loss) diagnostics ---
  // All zero / false unless online collection fed non-empty examples with a
  // positive search-loss coefficient this update — except aux_search_loss_coef,
  // which always records what was asked for.
  // --- PWO-5 section 14.1c: the seventeen canary columns 70-86. ----------
  //
  // Measured at section 14.1b's FROZEN site (the rollout behaviour-policy
  // decision) and carried here by AttachCanaryStats. These are NOT column 36
  // `entropy`, which is presentation-weighted over executed PPO minibatches on
  // an evolving model; these are per-decision on the frozen behaviour policy.
  // The two differ in population, weighting, model state and timing, and may
  // never be compared or substituted. Column 36 remains emitted and is read by
  // no rail.
  //
  // Role index = the integer value of DuneDecisionRole. Six of the seven can
  // carry a nontrivial row here; kForcedOrBookkeeping is identically empty at
  // this site by section 14.1c's proof, and is accumulated anyway so the
  // partition identity is CHECKED rather than asserted.
  //
  // Zero-support semantics (section 14.1c): when a role's count is 0 its mean
  // is emitted as 0.0, and 0.0 there means "no support this update", NEVER
  // "entropy collapsed to zero". Every consumer reads the count first. No rail
  // reads a role split at all -- the rails read the global mean -- so zero
  // support can never move a rail.
  double norm_entropy = 0.0;
  int64_t norm_entropy_n = 0;
  double norm_entropy_role[7] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  int64_t norm_entropy_n_role[7] = {0, 0, 0, 0, 0, 0, 0};

  double max_action_prob = 0.0;
  double frac_legal_absz_ge_cap = 0.0;
  int64_t frac_legal_absz_ge_cap_n = 0;

  int aux_examples_used = 0;          // # search examples folded in this update
  double aux_search_loss_coef = 0.0;  // coefficient REQUESTED this update (warmup
                                      // ramps it); >0 with aux_examples_used==0
                                      // means the collector produced nothing
  double aux_ce = 0.0;                // mean legal-action cross-entropy over aux slices
  double aux_value_mse = 0.0;         // mean critic MSE over aux slices
  double aux_grad_norm_mean = 0.0;    // per-update mean unclipped aux-only grad norm
  double ppo_grad_norm_mean = 0.0;    // per-update mean unclipped ppo-only grad norm
  double aux_ppo_norm_ratio = 0.0;    // aux_grad_norm_mean / ppo_grad_norm_mean
  bool aux_ratio_abort = false;       // ratio exceeded abort_grad_norm_ratio (caller aborts)
  int64_t aux_total_presentations = 0; // realized auxiliary presentations across minibatches
  int64_t aux_distinct_visited = 0;    // distinct search_examples rows presented >= 1 time
  std::vector<uint8_t> aux_visited_mask; // size == aux_examples_used when aux_active; 1 if visited
  int64_t aux_finish_pass_steps = 0;   // steps executed in opt-in unvisited aux finish pass
  int64_t aux_finish_pass_rows = 0;    // aux rows stepped in opt-in unvisited aux finish pass
  double search_aux_loss = 0.0;       // mean auxiliary search loss (aux_ce)

  static constexpr int kMaxR765ReplayAges = 16;
  int64_t aux_presentations_by_age[kMaxR765ReplayAges] = {0};
  double aux_eff_weight_sum_by_age[kMaxR765ReplayAges] = {0.0};
  int64_t aux_cap_active_slices = 0;
  int64_t aux_total_slices = 0;

  // --- Reverse KL penalty (Arm B: PPO with reverse-KL to rollout collection policy) ---
  double reverse_kl = 0.0;

  // --- EMAgnet forward KL regularization (Arm 2 / Treatment) ---
  double ema_magnet_kl = 0.0;
  double ema_magnet_kl_post = 0.0;

  // --- 12-block H/J study readouts ---
  int64_t single_action_transitions = 0;
  double single_action_fraction = 0.0;
  double single_action_value_mse_before = 0.0;
  double single_action_value_mse_after = 0.0;
  int64_t single_action_sample_size = 0;
  double unfiltered_kl = 0.0;
  int64_t unfiltered_kl_sample_size = 0;
  int64_t retained_transitions = 0;
  int64_t optimizer_minibatches = 0;

  // WO-PERF-TIMING. All zero and `enabled == false` unless
  // --phase_timing_mode=phases. No diagnostics column consumes this; it is
  // written only to the phase_timing.jsonl sidecar.
  PpoPhaseTimings phase_timings;
};

std::string ComputeRolloutHash(const std::vector<PpoTransition>& batch);
std::string ComputeRolloutSemanticDigest(const std::vector<PpoTransition>& batch);

bool VerifyRolloutCandidateAlignment(
    const std::vector<PpoTransition>& batch,
    std::string* error_msg = nullptr);

bool VerifyPackedMinibatchAlignment(
    const std::vector<PpoTransition>& batch,
    const int64_t* mb_indices,
    int64_t start_seq,
    int64_t mb_len,
    int64_t obs_size,
    int64_t action_dim,
    const float* staging_states_ptr,
    const bool* staging_masks_ptr,
    const int64_t* staging_actions_ptr,
    const std::vector<const dune_semantic::CandidateActionData*>& mb_cands,
    std::string* error_msg = nullptr);

torch::Tensor LegalLogitMean(const torch::Tensor& logits,
                             const torch::Tensor& legal_mask);

torch::Tensor ApplyLogitCapTensor(const torch::Tensor& logits,
                                  float logit_cap);

torch::Tensor CenterAndCapLogitsTensor(const torch::Tensor& logits,
                                       const torch::Tensor& legal_mask,
                                       float logit_cap);

float ComputeRewardLambda(uint64_t env_steps, uint64_t start_steps, uint64_t decay_steps);

// WO-PERF-1 (R2: per-model): re-arm the per-model "first TrainPpoUpdate in
// this process" markers that force a full policy_kl_before measurement at
// startup/resume when --diag_prepass_mode=cadenced. The marker is keyed on
// module identity, so a process training two models measures each model's
// first update. A resume is always a new process, so first-call-per-model IS
// the startup/resume condition; this hook exists only so tests can exercise
// the cadence deterministically.
void ResetDiagPrepassStateForTesting();

struct V5LegacySlotResidualState {
  bool active = false;
  int N = 0;
  int start_global_update = 0;
  // 7 multi-slot families
  std::vector<std::vector<int64_t>> families = {
      {10, 11, 12, 13, 14},
      {70, 71, 72},
      {91, 92},
      {538, 539, 540, 541, 542},
      {546, 547, 548, 549, 550},
      {554, 555, 556},
      {770, 771, 772}};
  std::vector<torch::Tensor> init_w_residuals;
  std::vector<torch::Tensor> init_b_residuals;

  void Initialize(const std::shared_ptr<SharedDunePolicyValueNetImpl>& model, int anneal_updates);
  void Apply(const std::shared_ptr<SharedDunePolicyValueNetImpl>& model, int completed_updates_in_run);
  bool SaveSidecar(const std::string& path) const;
  bool LoadSidecar(const std::string& path, torch::Device device);
};

PpoUpdateStats TrainPpoUpdate(
    std::shared_ptr<SharedDunePolicyValueNetImpl> model,
    torch::optim::AdamW& optimizer, std::vector<PpoTransition>& batch,
    int64_t obs_size, int64_t action_dim, torch::Device device,
    uint64_t master, int global_update,
    std::shared_ptr<SharedDunePolicyValueNetImpl> anchor_model = nullptr,
    // Phase 18B combined optimization. Default-empty / coef 0 => aux machinery is
    // skipped entirely and the update is numerically identical to today.
    const std::vector<SearchTrainingExample>& search_examples = {},
    double search_loss_coef = 0.0,
    double abort_grad_norm_ratio = 0.0,
    // Arm B: reverse-KL penalty to the policy that collected the rollout.
    // Detached collection snapshot; coefficient 0 => inert.
    std::shared_ptr<SharedDunePolicyValueNetImpl> collection_model = nullptr,
    double reverse_kl_coef = 0.0,
    bool compute_diagnostics = true,
    // Arm 2 (Treatment): EMAgnet forward KL regularization to moving parameter-EMA model.
    std::shared_ptr<SharedDunePolicyValueNetImpl> ema_model = nullptr,
    double ema_magnet_kl_coef = 0.0);

// Arm D: Fully separate actor and critic networks.
// Actor parameters produce policy logits and entropy loss; critic parameters produce
// scalar values and value loss. Global gradient clipping is applied over the union of
// active actor and critic parameters before stepping independent optimizers.
PpoUpdateStats TrainPpoUpdateSeparate(
    std::shared_ptr<SharedDunePolicyValueNetImpl> actor_model,
    torch::optim::AdamW& actor_optimizer,
    std::shared_ptr<SharedDunePolicyValueNetImpl> critic_model,
    torch::optim::AdamW& critic_optimizer,
    std::vector<PpoTransition>& batch,
    int64_t obs_size, int64_t action_dim, torch::Device device,
    uint64_t master, int global_update,
    bool compute_diagnostics = true,
    double gamma = 1.0,
    double gae_lambda = 1.0);

// Decoupled Standalone Auxiliary Search Supervision Phase (PPG-style)
// Explicit step budget, strict KL ceiling with rollback on violation,
// bitwise immutable value-head parameters, and critic probe shift tracking.
struct SearchAuxPhaseResult {
  int steps_budgeted = 0;
  int steps_accepted = 0;
  int accepted_override_presentations = 0;
  double initial_aux_ce = 0.0;
  double final_aux_ce = 0.0;
  double override_kl = 0.0;
  double cumulative_kl_searched = 0.0;
  double cumulative_kl_rollout = 0.0;
  double rail_kl_total = 0.0;
  double rail_kl_agreement = 0.0;
  double rail_kl_non_high_stakes = 0.0;
  double delta_entropy_rail = 0.0;
  double delta_entropy_high_stakes = 0.0;
  double mean_aux_grad_norm = 0.0;
  double mean_override_ce_loss = 0.0;
  double mean_anchor_kl_loss = 0.0;
  double mean_anchor_value_mse_loss = 0.0;
  bool step_rejected_on_kl = false;
  bool step_rejected_nonfinite = false;
  bool phase_rejected = false;
  bool null_supervision_skipped = false;
  bool value_head_immutable = true;
  double critic_probe_mse_shift = 0.0;
  double critic_probe_mse_shift_anchors = 0.0;
  double critic_probe_mse_shift_rail = 0.0;
  double max_prob_movement = 0.0;
  int64_t optimizer_states_before = 0;
  int64_t optimizer_states_after = 0;
};

struct SearchAuxAnchoringConfig {
  bool enabled = false;
  double beta = 1.0;
  double alpha = 1.0;
  double rail_kl_target = 0.001;
  int anchor_sample_size = 256;
  double logit_cap = 0.0;
};

SearchAuxPhaseResult TrainSearchAuxiliaryPhase(
    std::shared_ptr<SharedDunePolicyValueNetImpl> model,
    torch::optim::AdamW& optimizer,
    const std::vector<SearchTrainingExample>& search_examples,
    int64_t obs_size, int64_t action_dim,
    torch::Device device,
    int max_steps = 0,
    int batch_size = 64,
    double max_cumulative_kl = 0.01,
    double search_loss_coef = 0.10,
    const std::vector<PpoTransition>& reference_rollout_sample = {},
    uint64_t aux_seed = 0,
    int epochs = 1,
    const std::vector<SearchTrainingExample>& agreement_examples = {},
    const std::vector<PpoTransition>& all_rollout_transitions = {},
    const SearchAuxAnchoringConfig& anchoring_config = {});

struct SearchExamplesMetrics {
  double mean_ce = 0.0;
  double mean_p_search_move = 0.0;
  double greedy_agreement_pct = 0.0;
  int64_t count = 0;
};

// Evaluate cross-entropy, mean probability on search move, and greedy agreement
// on search training examples (for survival check, multi-update retention, and fresh-state check)
SearchExamplesMetrics EvaluateSearchExamplesMetrics(
    const std::shared_ptr<SharedDunePolicyValueNetImpl>& model,
    const std::vector<SearchTrainingExample>& examples,
    int64_t obs_size, int64_t action_dim,
    torch::Device device);

// Evaluate cross-entropy of policy on search training examples (for survival check)
double EvaluateSearchExamplesCrossEntropy(
    const std::shared_ptr<SharedDunePolicyValueNetImpl>& model,
    const std::vector<SearchTrainingExample>& examples,
    int64_t obs_size, int64_t action_dim,
    torch::Device device);

// --- WO-PERF-TIMING sidecar ------------------------------------------------
//
// Same shape as the PWO-5 head-telemetry sidecar and for the same reason: a
// new diagnostics.csv column is a hard fatal against an existing file
// (WriteDiagnostics' header gate) and would break the character-pinned v4/v5
// schema the schema tests enforce.
//
// The derived sidecar path, or "" when diagnostics_path is empty.
std::string PhaseTimingPath(const std::string& diagnostics_path);

// The fixed contract prefix of the header record. Run-agnostic, so it can be
// compared byte for byte across a resume.
std::string PhaseTimingContract();

// Appends one update's record, writing the header record first if the file is
// new. A header whose contract prefix differs from this binary's is fatal
// rather than an append. `ppo_elapsed_s` is the trainer's own `PPO:` figure,
// carried so a reader can see the attribution discrepancy directly instead of
// having to join this file against stdout.
void WritePhaseTiming(const std::string& diagnostics_path, int update,
                      const PpoUpdateStats& stats, double ppo_elapsed_s,
                      const std::string& run_uuid,
                      const std::string& run_prefix);

// Direct version of WritePhaseTiming that writes to a specified sidecar file path,
// independent of --diagnostics_path.
void WritePhaseTimingToPath(const std::string& sidecar_path, int update,
                            const PpoUpdateStats& stats, double ppo_elapsed_s,
                            const std::string& run_uuid,
                            const std::string& run_prefix);

void WriteDiagnostics(const std::string& filepath, int update, const PpoUpdateStats& stats,
                      double conflict_vp_generated, double conflict_vp_attributed, double conflict_vp_unattributed,
                      uint64_t seed, const std::string& run_uuid, const std::string& run_prefix, const std::string& config_fingerprint,
                      double raw_conflict_vp, double raw_noncombat_vp, double raw_total_vp,
                      double validation_kl = -1.0,
                      // PWO-5 section 14.1c. Defaults to false so every
                      // pre-PWO-5 caller keeps the v4 69-column schema
                      // byte-for-byte -- the header is compared literally on
                      // resume, so a silent widening would kill every existing
                      // run at its first WriteDiagnostics.
                      bool emit_canary_columns = false);

// The diagnostics CSV header for the requested schema. v4 = 69 columns; v5 =
// those 69 plus section 14.1c's seventeen canary columns 70-86, appended at the
// tail and changing no existing column. There is no numeric version constant in
// this format: the header string IS the version, stored as line 1 of the file
// and compared byte-for-byte on resume.
std::string DiagnosticsCsvHeader(bool emit_canary_columns);

#endif  // OPEN_SPIEL_BUILD_WITH_LIBTORCH

}  // namespace open_spiel

#endif  // OPEN_SPIEL_EXAMPLES_DUNE_PPO_TRAINING_TYPES_H_
