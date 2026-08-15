#ifndef OPEN_SPIEL_EXAMPLES_DUNE_PPO_TRAINING_UTILS_H_
#define OPEN_SPIEL_EXAMPLES_DUNE_PPO_TRAINING_UTILS_H_

#include <vector>
#include <string>
#include <memory>
#include <cstring>
#include <cstdint>

#include "open_spiel/spiel.h"       // Action, ActionsAndProbs
#include "open_spiel/utils/json.h"  // json::Object for manifest online-collection state
#include "dune_specimen_conversion.h"  // the single 741-752 conversion predicate



#ifdef OPEN_SPIEL_BUILD_WITH_LIBTORCH
#include <torch/torch.h>
#include "dune_network.h"
#include "dune_online_search_collector.h"  // SearchTrainingExample (18B combined opt)
#endif

namespace open_spiel {

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
};

#ifdef OPEN_SPIEL_BUILD_WITH_LIBTORCH

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
  // --- PWO-5 section 16 gate 3 item 6: per-head telemetry. ---------------
  // Each head's loss AND its denominator, plus the realized auxiliary exposure.
  // The denominators are the GAME counts (all three heads are
  // trajectory-weighted, section 8.6), summed over the update's slices. A row
  // with no later action contributes to neither the numerator nor the
  // denominator of next_own_action, so its game count is legitimately lower.
  double pwo5_final_vp_loss = 0.0;
  double pwo5_terminal_round_loss = 0.0;
  double pwo5_next_action_loss = 0.0;
  // The REGISTERED trajectory denominators: the count of DRAWN GAMES
  // contributing to each head this update (section 8.2's "sampled game
  // count"). Properties of the draw, independent of slicing and of how many
  // epochs --target_kl allowed.
  int64_t pwo5_sampled_games = 0;
  int64_t pwo5_final_vp_games = 0;
  int64_t pwo5_terminal_round_games = 0;
  int64_t pwo5_next_action_games = 0;
  // The per-slice game counts SUMMED over the update's slice-uses. This is the
  // denominator of the reported loss MEAN, not a trajectory denominator, and
  // the two are named differently so they cannot be confused: a slice-sum
  // counts a game once per epoch it was re-presented in.
  int64_t pwo5_slice_game_sum_final_vp = 0;
  int64_t pwo5_slice_game_sum_terminal_round = 0;
  int64_t pwo5_slice_game_sum_next_action = 0;
  int64_t pwo5_slice_uses = 0;
  int64_t pwo5_distinct_rows = 0;   // 1,024 by construction when active
  int64_t pwo5_presentations = 0;   // <= 4,096; "<=" because target_kl truncates

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
};

std::string ComputeRolloutHash(const std::vector<PpoTransition>& batch);

torch::Tensor LegalLogitMean(const torch::Tensor& logits,
                             const torch::Tensor& legal_mask);

torch::Tensor ApplyLogitCapTensor(const torch::Tensor& logits,
                                  float logit_cap);

torch::Tensor CenterAndCapLogitsTensor(const torch::Tensor& logits,
                                       const torch::Tensor& legal_mask,
                                       float logit_cap);

float ComputeRewardLambda(uint64_t env_steps, uint64_t start_steps, uint64_t decay_steps);

// PWO-5 section 8.6: one update's auxiliary-head batch, already materialized as
// device tensors by the caller.
//
// `game_id` maps each row to its 0-based index among the batch's sampled games,
// which is what makes the TRAJECTORY-WEIGHTED denominators of sections 8.2/8.6
// computable inside the loss: "mean over the batch's sampled games of the
// within-game mean loss", never a per-row mean. Flat row weighting would let
// long games dominate -- games carry 67 to 206 label rows -- and section 8.4
// item (i) mandates equal trajectory weight.
//
// `next_action` is -1 on a row with no later action in its trajectory (exactly
// the 400 final rows, one per game). Section 8.3 item (d): such a row is
// OMITTED, and "omitted" means removed from the NUMERATOR AND THE DENOMINATOR
// -- not zero-filled and not given a uniform target, either of which would
// train the head toward a fiction on 0.8% of rows.
struct Pwo5AuxBatch {
  torch::Tensor obs;             // [N, obs_size] float
  torch::Tensor final_vp;        // [N] float,  FinalScoredVp(seat)/20
  torch::Tensor terminal_round;  // [N] int64,  class 0/1/2
  torch::Tensor next_action;     // [N] int64,  -1 = omit
  torch::Tensor game_id;         // [N] int64,  0 .. num_games-1
  int64_t num_games = 0;
  bool valid = false;

  // The REGISTERED per-update trajectory denominators (sections 8.2, 8.6, and
  // amendment ruling 6): the count of DRAWN GAMES that contribute to each
  // head, computed once from the draw itself.
  //
  // These are NOT the same as summing each minibatch slice's game count. The
  // aux batch is partitioned across the update's PPO minibatches and the
  // slices are RE-CONSUMED once per executed epoch, so a slice-sum counts the
  // same game up to `ppo_update_epochs` times and its value depends on how
  // many epochs --target_kl happened to allow. Section 8.2 registers "the
  // sampled game count", which is a property of the DRAW and of nothing else.
  int64_t sampled_games = 0;             // = num_games, i.e. G = 64
  int64_t final_vp_games = 0;            // every drawn game (all rows have a target)
  int64_t terminal_round_games = 0;      // every drawn game
  int64_t next_own_action_games = 0;     // drawn games with >= 1 row that HAS a
                                         // later action; a game whose sampled
                                         // rows are all terminal contributes
                                         // nothing and is out of the denominator
};

// ---------------------------------------------------------------------------
// The three PWO-5 head loss forms, as SHARED definitions.
// ---------------------------------------------------------------------------
//
// Extracted so the trainer, the update-300 whole-dataset evaluator and the
// numeric tests all call ONE implementation. Three reimplementations of one
// arithmetic rule drifting apart is the exact defect the Memocorders finding
// was, and these are the same shape of rule.
namespace pwo5loss {

// Section 8.4 item i and section 8.6: the mean over CONTRIBUTING GAMES of the
// within-game mean. NOT a row mean.
//
// A per-row mean weights each game by its label count, and round-10 games are
// both over-represented among games AND carry more rows each -- the same
// imbalance twice over. `mask` selects contributing rows, so an omitted row is
// out of BOTH the numerator and the denominator, and a game with no
// contributing row is out of the game denominator too.
//
// `*out_games` receives the realized contributing-game count.
torch::Tensor TrajectoryMean(const torch::Tensor& per_row,
                             const torch::Tensor& mask,
                             const torch::Tensor& game_id, int64_t num_games,
                             int64_t* out_games);

// Huber, on the /20 scale. delta = 0.10 puts the quadratic/linear knee at
// 1.034 target standard deviations, so squared-error behaviour covers the
// central 68.25% of trajectories and the tails are linear. The framework
// default of 1.0 would be pure squared error over the whole reachable range
// [0.25, 0.75] and would make "Huber" a misnomer.
torch::Tensor HuberPerRow(const torch::Tensor& pred, const torch::Tensor& target,
                          double delta);

// Cross-entropy of `logits` against integer class targets, per row.
// Used by terminal_round_head (3 classes) and next_own_action_head (the FULL
// 2,391-action vocabulary, with NO mask -- the current state's legal mask is
// not valid for a FUTURE action, and the target is illegal at the predicting
// state on 74.57% of rows).
torch::Tensor CrossEntropyPerRow(const torch::Tensor& logits,
                                 const torch::Tensor& target_index);

}  // namespace pwo5loss

// The three coefficients plus the Huber delta, carried together so the
// short-circuit is decided in ONE place.
struct Pwo5AuxConfig {
  double final_vp_coef = 0.0;
  double terminal_round_coef = 0.0;
  double next_own_action_coef = 0.0;
  double huber_delta = 0.10;
  // Section 7.4: at exactly zero the loss terms are NOT CONSTRUCTED -- a
  // short-circuit, not a multiply-by-zero. Multiply-by-zero would still build
  // the graph, still populate .grad with exact zeros, and still expose the
  // parameters to weight decay and to any non-finite value in the head's
  // forward pass.
  bool AnyActive() const {
    return final_vp_coef != 0.0 || terminal_round_coef != 0.0 ||
           next_own_action_coef != 0.0;
  }
};

// WO-PERF-1 (R2: per-model): re-arm the per-model "first TrainPpoUpdate in
// this process" markers that force a full policy_kl_before measurement at
// startup/resume when --diag_prepass_mode=cadenced. The marker is keyed on
// module identity, so a process training two models measures each model's
// first update. A resume is always a new process, so first-call-per-model IS
// the startup/resume condition; this hook exists only so tests can exercise
// the cadence deterministically.
void ResetDiagPrepassStateForTesting();

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
    // PWO-5 section 8.6. Default-invalid batch + all-zero coefficients => the
    // head machinery is skipped ENTIRELY: no auxiliary forward, no head output
    // computed, no graph node, no gradient. That is what makes a head-off arm's
    // behaviour independent of the heads' numerics.
    const Pwo5AuxBatch& pwo5_aux = Pwo5AuxBatch(),
    const Pwo5AuxConfig& pwo5_cfg = Pwo5AuxConfig());

// ---------------------------------------------------------------------------
// PWO-5 head telemetry sidecar (amendment 1 ruling 6).
// ---------------------------------------------------------------------------
//
// Registration section 16 gate 3 item 6 requires per-head loss, per-head
// denominators and per-component gradient norms; sections 14.1c and 17.5 item
// 14 simultaneously pin `diagnostics.csv` at EXACTLY 86 columns and make any
// other header a STOP. Both cannot hold. Ruling 6 resolves it: the CSV stays
// v5/86, and the head telemetry goes to a separate sidecar with four registered
// properties --
//
//   * FIXED PATH, derived mechanically from --diagnostics_path: that path with
//     its filename replaced by `pwo5_head_telemetry.jsonl` in the same
//     directory. Deterministic, per-arm, no operator choice;
//   * NO NEW CLI FLAG. Appendix A.1's closing rule makes any gate-3 flag
//     outside its table a STOP, so the path is DERIVED, never PASSED, and
//     emission is gated by the already-registered --emit_canary_columns;
//   * SELF-DESCRIBING: line 1 is a header record declaring the schema, the run
//     identity and the exact ordered field names. A consumer never infers a
//     column position;
//   * ROUND-TRIP PRECISION: every float serialized at `%.17g`, and NOT routed
//     through open_spiel's json.cc writer, whose `%f` six-decimal floor turns
//     anything below 5e-7 into exactly 0.0 -- which a per-head loss approaching
//     zero would otherwise trip.

// Section 11.1: the base is GLOBAL update 2450 (Branch-A u2450, the frozen
// PWO-5 base), so pilot-local `n` is global `2450 + n`. A CONSTANT rather than
// a flag, because ruling 6 forbids the sidecar introducing one and because the
// base checkpoint is frozen by section 7.1 -- a flag here could drift from the
// checkpoint it describes.
inline constexpr int kPwo5GlobalUpdateBase = 2450;

// The derived sidecar path, or "" when diagnostics_path is empty.
std::string Pwo5HeadTelemetryPath(const std::string& diagnostics_path);

// The fixed contract prefix of the header record: schema, float format, global
// update base and the exact ordered field names. Everything a consumer must
// agree on, and nothing run-specific -- so it can be compared byte for byte
// across a resume.
std::string Pwo5HeadTelemetryContract();

// Appends one update's record, writing the header record first if the file is
// new. On resume, a header whose contract prefix differs from this binary's is
// a fatal error rather than an append.
void WritePwo5HeadTelemetry(const std::string& diagnostics_path,
                            int pilot_local_update,
                            const PpoUpdateStats& stats,
                            const std::string& run_uuid,
                            const std::string& config_fingerprint,
                            uint64_t base_seed);

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

struct CheckpointManifest {
  int schema_version;
  std::string checkpoint_uuid;
  int global_update;
  int target_end_update;
  uint64_t total_env_steps;
  uint64_t next_episode_id;
  uint64_t base_seed;
  int seed_scheme_version;
  std::string config_fingerprint;
  std::string search_label_fingerprint;
  std::string run_uuid;
  std::string model_filename;
  size_t model_file_size;
  std::string model_sha256;
  std::string optimizer_filename;
  size_t optimizer_file_size;
  std::string optimizer_sha256;
  int hidden_dim;
  int num_blocks;
};

// Deterministic contiguous partition of `num_examples` aux examples across
// `num_minibatches` PPO minibatch steps in one epoch: base each, remainder to
// the first slices. Same partition every epoch => each example used exactly once
// per epoch that fully runs. Returns {start, len} per minibatch. Exposed for
// unit testing the each-example-once schedule.
std::vector<std::pair<int64_t, int64_t>> ComputeAuxSlices(int64_t num_examples,
                                                          int64_t num_minibatches);

// Phase 18B online-collection state persisted in the checkpoint manifest for
// exact resume (plan §18C). All fields are OPTIONAL in the manifest (absent
// entirely when online collection is off), so reading tolerates their absence.
struct OnlineCollectionState {
  bool present = false;              // true once read from a manifest that had it
  // Effective config echo (audit).
  int auxiliary_games = 0;
  uint64_t auxiliary_search_seed_domain = 0;
  double collector_dirichlet_epsilon = 0.0;
  double swordmaster_grant_fraction = 0.0;
  int swordmaster_grant_round = 0;
  double search_loss_coef_target = 0.0;
  int search_loss_warmup_update = 0;   // warmup position (updates 1 -> this)
  double abort_grad_norm_ratio = 0.0;
  double target_sharpen_exponent = 1.0;  // CE-target peaking exponent (1.0 = inert)
  // Prior the collector's covered-mass acceptance rule was measured against
  // ("raw_network_prior" | "tree_prior"; see AcceptancePriorSource). The
  // cumulative counters and hash chain below are only meaningful within ONE
  // source, so this travels with them. EMPTY means the manifest predates WO-20,
  // i.e. the run accumulated them under the post-noise tree prior; the resume
  // path must treat that as "unknown contract" and refuse to extend it silently
  // rather than defaulting.
  std::string acceptance_prior_source;
  // Exact-resume cursor + cumulative counters + chained accepted-target hash.
  uint64_t next_auxiliary_episode_id = 0;
  int64_t cum_accepted = 0, cum_rejected = 0;
  int64_t cum_role_searches[3] = {0, 0, 0};
  int64_t cum_role_accepted[3] = {0, 0, 0};
  int64_t cum_granted = 0, cum_organic = 0;
  std::string accepted_hash_chain;
};

// Adds the online-collection fields to a manifest object under "online_collection".
void WriteOnlineCollectionState(json::Object& manifest_obj,
                                const OnlineCollectionState& st);
// Reads them back from a manifest file. Returns true and sets out.present=true
// iff the "online_collection" object was present and well-formed; returns true
// with out.present=false if simply absent (not an error); false on malformed.
bool ReadOnlineCollectionState(const std::string& manifest_path,
                               OnlineCollectionState& out,
                               std::string& error_msg);

// Returns true on success. Populates out_manifest.
// Populates error_msg and returns false on failure.
bool ParseAndValidateManifest(const std::string& manifest_path,
                              const std::string& model_path,
                              const std::string& optim_path,
                              uint64_t current_base_seed,
                              int current_target_end_update,
                              int current_seed_scheme_version,
                              const std::string& current_config_fingerprint,
                              const std::string& current_search_label_fingerprint,
                              int current_hidden_dim,
                              int current_num_blocks,
                              CheckpointManifest& out_manifest,
                              std::string& error_msg,
                              const std::string& current_legacy_config_fingerprint = "");

inline uint32_t Fnv1a(const uint8_t* data, size_t size) {
  uint32_t hash = 2166136261U;
  for (size_t i = 0; i < size; ++i) {
    hash ^= data[i];
    hash *= 16777619U;
  }
  return hash;
}

inline uint32_t ComputeLabelFnv1a(const std::vector<float>& observation,
                                  const std::vector<int32_t>& legal_actions,
                                  int32_t player_id) {
  size_t obs_bytes_len = observation.size() * sizeof(float);
  std::vector<uint8_t> bytes(obs_bytes_len + legal_actions.size() * sizeof(int32_t) + sizeof(int32_t));
  if (obs_bytes_len > 0) {
    std::memcpy(bytes.data(), observation.data(), obs_bytes_len);
  }
  size_t offset = obs_bytes_len;
  for (int32_t action : legal_actions) {
    std::memcpy(bytes.data() + offset, &action, sizeof(int32_t));
    offset += sizeof(int32_t);
  }
  std::memcpy(bytes.data() + offset, &player_id, sizeof(int32_t));
  return Fnv1a(bytes.data(), bytes.size());
}

inline bool IsValidationLabel(const std::vector<float>& observation,
                              const std::vector<int32_t>& legal_actions,
                              int32_t player_id) {
  return ComputeLabelFnv1a(observation, legal_actions, player_id) % 11 == 0;
}

// ---------------------------------------------------------------------------
// Teacher-label action pairing
// ---------------------------------------------------------------------------
// The offline search teacher derives its PPO prior from LegalActions() and its
// teacher prior from the search root's action vector. Those two orders agree
// today, so every label paired them by index — but nothing enforced it. If
// search ever prunes or reorders root actions, index pairing writes teacher
// probabilities beside the wrong action ids and produces a label file that
// still parses cleanly. Reorder explicitly instead, and refuse to guess when
// the two do not describe the same action set.

// Reorders `prior` into `actions` order. Returns false and leaves `out`
// untouched unless the two are a bijection over the same action ids; callers
// drop the label rather than emit a mispaired one.
inline bool AlignPriorToActions(const ActionsAndProbs& prior,
                                const std::vector<Action>& actions,
                                ActionsAndProbs* out) {
  if (out == nullptr || prior.size() != actions.size()) return false;
  ActionsAndProbs aligned;
  aligned.reserve(actions.size());
  // `consumed` makes this a bijection check rather than a membership check, so
  // duplicated ids on either side cannot map two slots onto one probability.
  std::vector<bool> consumed(prior.size(), false);
  for (Action action : actions) {
    size_t match = prior.size();
    for (size_t i = 0; i < prior.size(); ++i) {
      if (!consumed[i] && prior[i].first == action) {
        match = i;
        break;
      }
    }
    if (match == prior.size()) return false;
    consumed[match] = true;
    aligned.push_back(prior[match]);
  }
  *out = std::move(aligned);
  return true;
}

// Specimen-exchange anti-breadcrumb shaping, extracted so its SIGN is testable.
//
// PWO-5 gate 2 items (a)+(b) (docs/PWO5_PILOT_REGISTRATION.md section 16).
//
// `--specimen_exchange_penalty` is a MAGNITUDE THAT IS SUBTRACTED, so a POSITIVE
// value PENALIZES and a NEGATIVE value is a BONUS. That inversion is not
// hypothetical: the u175 lineage was trained with `-0.02`, i.e. a +0.02 bonus on
// exactly the breadcrumb behaviour the term exists to suppress, and the
// committed `calibration_results_v2/pilot300_search_seed12/launch.sh` still
// shows it. `dune_ppo_train` now rejects a negative value fatally, and this
// function exists so "positive DECREASES the reward" is a unit-tested property
// of the code the trainer actually runs rather than a comment above it.
//
// Returns the shaped reward. A non-conversion action, or a zero penalty, returns
// `reward` unchanged and bit-identically.
float ApplySpecimenExchangeShaping(float reward, Action action,
                                   double specimen_exchange_penalty,
                                   float reward_lambda);

// True when both vectors carry the same action ids in the same order — the
// invariant a label writer relies on when it stores prior[i]'s action id beside
// teacher[i]'s probability.
inline bool PriorsShareActionOrder(const ActionsAndProbs& a,
                                   const ActionsAndProbs& b) {
  if (a.size() != b.size()) return false;
  for (size_t i = 0; i < a.size(); ++i) {
    if (a[i].first != b[i].first) return false;
  }
  return true;
}

} // namespace open_spiel

#endif // OPEN_SPIEL_EXAMPLES_DUNE_PPO_TRAINING_UTILS_H_
