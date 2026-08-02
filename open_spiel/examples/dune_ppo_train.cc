// Native LibTorch PPO trainer for Dune Imperium.
//
// This executable is intentionally separate from dune_benchmark.cc. It follows
// the OpenSpiel PyTorch PPO structure: collect an on-policy rollout batch,
// freeze it, run shuffled PPO epochs/minibatches, then sync the inference model.

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <sstream>
#include <iomanip>

#include <random>
#include <shared_mutex>
#include <string>
#include <thread>
#include <vector>

#include "open_spiel/abseil-cpp/absl/flags/flag.h"
#include "open_spiel/abseil-cpp/absl/flags/parse.h"
#include "open_spiel/abseil-cpp/absl/strings/str_format.h"
#include "open_spiel/games/dune_imperium/dune_imperium.h"

#include "open_spiel/games/dune_imperium/dune_imperium_util.h"
#include "open_spiel/spiel.h"

#ifdef OPEN_SPIEL_BUILD_WITH_LIBTORCH
#include <torch/torch.h>

#include "dune_network.h"
#include "dune_seed_utils.h"
#include "dune_sha256.h"
#include "dune_ppo_training_utils.h"
#include "dune_pwo5_aux.h"
#include "dune_search_label_buffer.h"
#include "dune_search_routing.h"
#include "dune_evaluator.h"  // DuneNNEvaluator for online-collection snapshot inference
#include "open_spiel/utils/json.h"
#endif

ABSL_FLAG(std::string, game, "dune_imperium", "The OpenSpiel game to train.");
ABSL_FLAG(int, threads, 64, "Rollout worker threads.");
ABSL_FLAG(int, total_updates, 1000, "Number of PPO collect/update cycles.");
ABSL_FLAG(int, rollout_transitions, 32768,
          "Minimum on-policy transitions collected before each PPO update.");
ABSL_FLAG(int, rollout_games, 0,
          "Exact complete games collected per PPO update rollout batch. "
          "If > 0, overrides transition-threshold mode.");
ABSL_FLAG(int, ppo_minibatch_size, 2048, "PPO minibatch size.");
ABSL_FLAG(int, ppo_update_epochs, 4, "PPO epochs per rollout batch.");
ABSL_FLAG(double, learning_rate, 2.5e-4, "AdamW learning rate.");
ABSL_FLAG(bool, anneal_lr, true, "Linearly anneal learning rate over updates.");
ABSL_FLAG(double, gamma, 1.0, "Discount factor.");
ABSL_FLAG(double, gae_lambda, 1.0, "GAE lambda.");
ABSL_FLAG(bool, normalize_advantages, true,
          "Normalize advantages within each PPO minibatch.");
ABSL_FLAG(double, ppo_clip_epsilon, 0.2, "PPO surrogate/value clip epsilon.");
ABSL_FLAG(bool, ppo_clip_value_loss, true, "Use clipped value loss.");
ABSL_FLAG(double, entropy_coef, 0.01, "Entropy bonus coefficient.");
ABSL_FLAG(double, value_coef, 0.5, "Value loss coefficient.");
ABSL_FLAG(double, grad_clip_norm, 0.5, "Maximum gradient norm.");
ABSL_FLAG(double, target_kl, 0.0,
          "Stop PPO epochs early if approximate KL exceeds this value. <=0 disables.");
ABSL_FLAG(double, weight_decay, 0.0, "AdamW weight decay for torso/value params.");
ABSL_FLAG(double, policy_weight_decay, 0.0,
          "AdamW weight decay for policy-head params.");

ABSL_FLAG(int, eval_batch_size, 64, "Target inference batch size.");
ABSL_FLAG(int, eval_timeout_ms, 2, "Inference batch timeout in milliseconds.");
ABSL_FLAG(int, hidden_dim, 2048, "Network hidden dimension.");
ABSL_FLAG(int, num_blocks, 8, "Network residual block count.");
ABSL_FLAG(double, logit_cap, 10.0,
          "Smooth tanh cap on legal-centered policy logits. <=0 disables.");
ABSL_FLAG(bool, train_amp, true, "Use CUDA BF16 autocast for PPO updates.");
ABSL_FLAG(bool, evaluator_device_synchronize, true,
          "Use whole-device CUDA synchronize after evaluator D2H copies.");
ABSL_FLAG(bool, deterministic, true, "Enable strict PyTorch/LibTorch deterministic algorithms.");
ABSL_FLAG(bool, deterministic_rollout_eval, false, "Use deterministic batch-1 rollout evaluation.");
ABSL_FLAG(bool, diagnostics_only, false, "Collect one rollout, write diagnostics, and exit without optimization.");

ABSL_FLAG(double, shaped_reward_weight, 0.0,
          "Weight for intermediate VP shaped rewards.");
ABSL_FLAG(double, tleilaxu_breadcrumb_weight, 0.0,
          "Weight for Tleilaxu levels 5/6 breadcrumbs.");
ABSL_FLAG(double, tleilaxu_level7_breadcrumb_weight, 0.0,
          "Weight for Tleilaxu level 7 breadcrumb.");
// Sign convention: the value is SUBTRACTED from reward — pass a POSITIVE value to penalize. (The phase-18 pilots passed -0.02, which was a +0.02 bonus.)
// A NEGATIVE value is now rejected fatally in main() (PWO-5 gate 2 item (a)).
ABSL_FLAG(double, specimen_exchange_penalty, 0.0,
          "Magnitude of the negative shaping SUBTRACTED from a transition that "
          "takes a ConvertSpecimenToTroop action (IDs 741-752; 740 is an unused "
          "base constant and is never legal). MUST BE >= 0: the value is "
          "subtracted, so a POSITIVE value penalizes and a NEGATIVE value would "
          "be a BONUS on the very behaviour this term exists to suppress -- a "
          "negative value is rejected fatally at startup. Training-only "
          "anti-breadcrumb (never eval). Apply the SAME value to BOTH pilot and "
          "control arms so the search-distillation contrast stays a pure "
          "experiment. Typical 0.02 (terminal win utility is 2.25). Requires "
          "--allow_shaping.");
ABSL_FLAG(bool, allow_shaping, false,
          "Allow experimental reward shaping flags to be non-zero.");
ABSL_FLAG(double, reward_scale, 4.0,
          "Divide shaped plus terminal rewards by this value.");

ABSL_FLAG(std::string, model_checkpoint, "dune_ppo_model.pt",
          "Model checkpoint to load/save.");
ABSL_FLAG(std::string, optim_checkpoint, "dune_ppo_optimizer.pt",
          "Optimizer checkpoint to load/save.");
ABSL_FLAG(std::string, artifact_manifest, "",
          "Path to the Phase 1 baseline artifact manifest.json for validation.");
ABSL_FLAG(std::string, run_prefix, "dune_ppo",
          "Prefix for rotating checkpoints.");
ABSL_FLAG(int, checkpoint_interval, 10,
          "Save rotating checkpoints every N PPO updates. <=0 disables.");
ABSL_FLAG(bool, save_final_checkpoint, true,
          "Save final model and optimizer checkpoints.");
ABSL_FLAG(int, seed, 1, "Base random seed.");
ABSL_FLAG(bool, pipeline, false,
          "Overlap next rollout collection with current PPO training. "
          "Beneficial with separate inference/training GPUs. On a single GPU, "
          "both workloads compete for compute and pipelining may be slower.");

ABSL_FLAG(std::string, search_label_dir, "",
          "[LEGACY offline-distillation mode] Directory containing precomputed "
          "search teacher labels. Empty disables. Mutually exclusive with "
          "--online_search_collection (the Phase 18B online path).");
ABSL_FLAG(double, search_lambda, 0.5,
          "[LEGACY offline-distillation] Weight for the static search teacher KL "
          "distillation loss.");
ABSL_FLAG(int, search_minibatches_per_update, 2,
          "[LEGACY offline-distillation] Search distillation minibatches per PPO update.");
ABSL_FLAG(int, search_minibatch_size, 512,
          "[LEGACY offline-distillation] Size of each search distillation minibatch.");

// --- PF-6 / C: joint distillation. DEFAULT-INERT. -------------------------
// C's design sums the teacher KL INTO the PPO objective under PPO's own KL
// throttle, instead of PWO-5's SEPARATE optimizer steps. With this flag false
// the trainer's objective, RNG draws and step count are bit-for-bit what they
// were before the flag existed: no batch is materialized (so no RNG is
// drawn), Pf6JointDistillBatch stays default-constructed, and its Active() is
// false, so TrainPpoUpdate constructs no term at all.
//
// C IS CLOSED. This flag exists so the implementation is written, built and
// smoke-validated BEFORE any trigger fires -- not assembled in the moment by
// someone who wants the door open. Turning it on for a registered run
// requires the C amendment, ratified under section 0.1.
ABSL_FLAG(bool, search_distill_joint, false,
          "[PF-6 / C] Sum the search-teacher KL into the PPO objective "
          "(under PPO's KL throttle) instead of running separate optimizer "
          "steps. DEFAULT FALSE = bit-for-bit unchanged. Mutually exclusive "
          "with the legacy separate-step path: enabling this DISABLES those "
          "steps rather than running both.");

// --- Phase 18B online auxiliary-search collection (combined optimization) ---
// Treatment-arm switch. When true, each update collects search examples online
// from the frozen pre-update snapshot and folds them into TrainPpoUpdate as an
// auxiliary CE+value loss. Mutually exclusive with --search_label_dir. Every
// other OnlineSearchConfig field keeps its library default (the frozen pilot
// values) and is NOT restated as a flag.
ABSL_FLAG(bool, online_search_collection, false,
          "Enable Phase 18B online auxiliary-search collection + combined optimization.");
ABSL_FLAG(int, auxiliary_games, 16,
          "Online collector auxiliary games per update (must be a multiple of 4).");
ABSL_FLAG(uint64_t, auxiliary_search_seed_domain, 0,
          "Isolated seed domain for online collection. REQUIRED nonzero when "
          "--online_search_collection is set (no silent default).");
ABSL_FLAG(double, collector_dirichlet_epsilon, 0.25,
          "Root Dirichlet noise weight for online collection (pilot 0.25; probe used 0).");
ABSL_FLAG(std::string, collector_acceptance_prior, "",
          "Prior the collector's covered-mass acceptance rule is measured "
          "against: 'raw_network_prior' (default when unset) or 'tree_prior' "
          "(pre-WO-20 post-noise behavior). Must be stated EXPLICITLY when "
          "resuming a checkpoint whose manifest predates the field, because its "
          "cumulative counters were accumulated under tree_prior.");
ABSL_FLAG(double, search_loss_coef, 0.10,
          "Target auxiliary search-loss coefficient (warms 0 -> target).");
ABSL_FLAG(int, search_loss_warmup_update, 25,
          "Update at which the search-loss coefficient reaches its target.");
ABSL_FLAG(double, target_sharpen_exponent, 1.0,
          "CE-target sharpening exponent for online collection: the pruned root "
          "visit target is peaked as p_i = v_i^alpha / sum_j v_j^alpha before "
          "emission. 1.0 = inert (today's behavior).");
ABSL_FLAG(double, abort_grad_norm_ratio, 0.50,
          "Abort the run if the per-update aux/PPO grad-norm ratio exceeds this.");
ABSL_FLAG(double, swordmaster_grant_fraction, 0.0,
          "Endowment-curriculum: fraction of collector games granting the searched "
          "seat a free Swordmaster.");
ABSL_FLAG(int, swordmaster_grant_round, 2,
          "Round at which a selected game's Swordmaster grant fires.");

ABSL_FLAG(std::string, init_mode, "",
          "Initialization mode (required): random, checkpoint, bootstrap, validate_legacy.");
ABSL_FLAG(uint64_t, shaping_start_env_steps, 206830543,
          "Environment steps at which shaped reward decay starts.");
ABSL_FLAG(uint64_t, shaping_decay_env_steps, 0,
          "Environment steps duration for shaped reward decay (0 means no decay).");
ABSL_FLAG(int, seed_scheme_version, 2, "Seed scheme version.");
ABSL_FLAG(int, target_end_update, 2450, "Absolute target update number to train until.");
ABSL_FLAG(int, start_update, 1, "Fallback start update for bootstrap mode.");
ABSL_FLAG(uint64_t, start_env_steps, 0, "Fallback start environment steps for bootstrap mode.");
ABSL_FLAG(uint64_t, start_episode_id, 0, "Fallback start episode ID for bootstrap mode.");
ABSL_FLAG(std::string, diagnostics_path, "",
          "Path to write structured diagnostics JSON/CSV. Extension determines format.");
ABSL_DECLARE_FLAG(bool, train_value_only);
ABSL_FLAG(bool, unfreeze_trunk, false, "Unfreeze the shared trunk (input + res blocks) during training.");
ABSL_FLAG(bool, nonlinear_value_head, false, "Use a nonlinear value head.");
ABSL_FLAG(bool, sample_counterfactual_states, false, "Sample generic counterfactual successor states.");
ABSL_FLAG(int, counterfactual_samples_per_game, 4,
          "Maximum uniformly sampled alternate primary-action successors per game.");
ABSL_FLAG(int, counterfactual_replay_weight, 8,
          "Value-loss replay multiplicity for each sampled counterfactual successor.");
ABSL_DECLARE_FLAG(double, policy_kl_anchor_coeff);

// ===========================================================================
// PWO-5 gate 3 — Appendix A.1 of docs/PWO5_PILOT_REGISTRATION.md
// ===========================================================================
//
// These THIRTEEN flags are FROZEN by that appendix, which registers them
// BEFORE they exist precisely so no post-hoc flag can enter the experiment
// claiming it was always intended. **A flag below whose name, type or compiled
// default differs from the appendix is a STOP, as is any ADDITIONAL flag gate 3
// introduces that the appendix does not list.** So: exactly these thirteen,
// exactly these names, exactly these types, exactly these defaults.
//
// EVERY COMPILED DEFAULT IS INERT (0.0 / "" / 0 / false). That is a registered
// property, not a convenience: a build that has gained these flags but is
// launched without them trains exactly as the pre-gate-3 binary did, which is
// what keeps the section 7.3 legacy-inference parity gate and the section 6.4
// no-aux-leakage check meaningful AFTER gate 3 lands.
//
// The three coefficients are the H-vs-T TREATMENT (section 9.1) and are
// therefore expected to differ across arms; everything else here is MATCHED and
// must be byte-identical on all six.

ABSL_FLAG(double, final_vp_head_coef, 0.0,
          "PWO-5 section 8.1: coefficient on the final_vp_head Huber loss. "
          "0.05 in H arms, 0.0 in T and P. Exactly 0.0 short-circuits the head "
          "entirely -- the loss term is NOT CONSTRUCTED, so no graph node "
          "exists and the head parameters receive no gradient at all (section "
          "7.4 rejects multiply-by-zero as the mechanism).");
ABSL_FLAG(double, terminal_round_head_coef, 0.0,
          "PWO-5 section 8.1: coefficient on the terminal_round_head "
          "cross-entropy over the three classes <=8 / 9 / 10. 0.05 in H arms, "
          "0.0 in T and P.");
ABSL_FLAG(double, next_own_action_head_coef, 0.0,
          "PWO-5 section 8.1: coefficient on the next_own_action_head "
          "cross-entropy over the full 2391-action vocabulary, unmasked. 0.15 "
          "in H arms, 0.0 in T and P.");
ABSL_FLAG(std::string, aux_target_path, "",
          "PWO-5 section 6.1 item 1: the parallel, separately hashed auxiliary "
          "target artifact keyed by (game_index, decision_index). MATCHED on "
          "all six arms and digest-verified on all six; LOADED AND USED only "
          "when at least one head coefficient is nonzero. The verification is "
          "unconditional, the use is not -- so a corrupt artifact is caught on "
          "every arm at launch, not only on the two that would have read it.");
ABSL_FLAG(std::string, aux_target_sha256, "",
          "PWO-5 section 5.2: sha256 of --aux_target_path, asserted on all six "
          "arms. A mismatch is fatal (section 17.5 item 3): a materialized "
          "artifact that is not hashed is an unpinned replay input.");
ABSL_FLAG(int, aux_games_per_update, 0,
          "PWO-5 section 8.6: G, the number of games drawn uniformly WITHOUT "
          "replacement ONCE per update, then partitioned. Registered 64.");
ABSL_FLAG(int, aux_rows_per_game, 0,
          "PWO-5 section 8.6: R, rows drawn uniformly without replacement from "
          "each drawn game. Registered 16.");
ABSL_FLAG(int, aux_batches_per_update, 0,
          "PWO-5 section 8.6: the partition count. Registered 2, giving "
          "2 x 32 games x 16 rows = 1,024 DISTINCT rows per update.");
ABSL_FLAG(std::string, aux_heldout_games_path, "",
          "PWO-5 section 8.5: the 60-game whole-trajectory held-out membership "
          "list. Held-out games are excluded from ALL offline losses, "
          "INCLUDING distillation, because the trunk is shared.");
ABSL_FLAG(std::string, aux_heldout_sha256, "",
          "PWO-5 section 8.5: sha256 of the canonical serialization of the 60 "
          "held-out game indices, asserted on resume and on the u600 extension "
          "so the split cannot silently differ across arms or replicates.");
ABSL_FLAG(double, huber_delta_final_vp, 0.10,
          "PWO-5 section 8.2: Huber delta for final_vp_head, on the /20 scale. "
          "The target spans [0.25, 0.80], so 0.10 puts the quadratic/linear "
          "knee at ~18% of the target range. The framework default 1.0 would "
          "be pure squared error over the whole reachable range and would make "
          "'Huber' a misnomer.");
ABSL_FLAG(uint64_t, head_init_constant, 20260800,
          "PWO-5 section 7.2: kHeadInitConstant. NOT a run seed -- it is a "
          "fixed registered constant so that the initial head parameters are "
          "byte-identical across ALL SIX arms. Deriving from the triplet seed "
          "would give T1/P1/H1 one set and T2/P2/H2 another. Outside the "
          "reserved final-gate base-seed range by section 10.2's arithmetic.");
ABSL_FLAG(bool, emit_canary_columns, false,
          "PWO-5 section 14.1a: emit the seventeen collapse-canary columns "
          "70-86 (diagnostics schema v4 -> v5, 69 -> 86 columns), measured at "
          "section 14.1b's FROZEN site -- the rollout behaviour-policy "
          "decision -- and nowhere else. Launching a PWO-5 arm with this false "
          "is section 17.5 item 14: without the columns the section 14.2 halt "
          "rail cannot fire, and an unmonitored 300-update run is exactly the "
          "failure mode the u175 collapse lineage represents.");

namespace open_spiel {
namespace {

std::set<std::vector<open_spiel::Action>> prohibited_histories;

#ifdef OPEN_SPIEL_BUILD_WITH_LIBTORCH

class PpoRolloutBuffer {
 public:
  size_t PushTrajectory(std::vector<PpoTransition>&& trajectory) {
    std::lock_guard<std::mutex> lock(mu_);
    size_t size = trajectory.size();
    trajectories_.push_back(std::move(trajectory));
    num_transitions_ += size;
    return num_transitions_;
  }

  size_t Size() const {
    std::lock_guard<std::mutex> lock(mu_);
    return num_transitions_;
  }

  std::vector<PpoTransition> TakeAll(bool* out_episode_ids_unique = nullptr) {
    std::lock_guard<std::mutex> lock(mu_);
    if (out_episode_ids_unique) {
      *out_episode_ids_unique = true;
      std::unordered_set<uint64_t> seen_ids;
      for (const auto& traj : trajectories_) {
        if (!traj.empty()) {
          uint64_t ep_id = traj[0].episode_id;
          if (seen_ids.count(ep_id)) {
            *out_episode_ids_unique = false;
          }
          seen_ids.insert(ep_id);
        }
      }
    }
    std::sort(trajectories_.begin(), trajectories_.end(),
              [](const std::vector<PpoTransition>& a,
                 const std::vector<PpoTransition>& b) {
                uint64_t id_a = a.empty() ? 0 : a[0].episode_id;
                uint64_t id_b = b.empty() ? 0 : b[0].episode_id;
                return id_a < id_b;
              });
    std::vector<PpoTransition> out;
    out.reserve(num_transitions_);
    for (auto& traj : trajectories_) {
      for (auto& transition : traj) {
        out.push_back(std::move(transition));
      }
    }
    trajectories_.clear();
    num_transitions_ = 0;
    return out;
  }

 private:
  mutable std::mutex mu_;
  std::vector<std::vector<PpoTransition>> trajectories_;
  size_t num_transitions_ = 0;
};

// SearchLabel, SearchLabelRole, SearchLabelFileEntry and SearchLabelBuffer now
// live in dune_search_label_buffer.h so that the trainer and the ruling-5 role
// tests link ONE definition rather than two that can drift.
using open_spiel::SearchLabel;
using open_spiel::SearchLabelBuffer;
using open_spiel::SearchLabelFileEntry;
using open_spiel::SearchLabelRole;

struct WorkerStats {
  uint64_t games = 0;
  uint64_t moves = 0;

  // Natural SM Acquisitions
  uint64_t sm_acquisitions_by_seat[4] = {0};
  uint64_t sm_acquisitions_by_leader[14] = {0};
  uint64_t sm_acquisitions_by_round[11] = {0};

  double conflict_vp_generated = 0.0;
  double conflict_vp_attributed = 0.0;
  double conflict_vp_unattributed = 0.0;

  // Raw VPs for conservation checks
  double raw_conflict_vp = 0.0;
  double raw_noncombat_vp = 0.0;
  double raw_total_vp = 0.0;

  // Phase A canary: per-decision PRE-cap legal-centered max|z| for every
  // nontrivial rollout decision this worker made, overall and split by decision
  // role. Purely observational -- collected before CenterAndCapLegalLogits
  // mutates the logits, and never read back into the acting path.
  std::vector<float> precap_absz_all;
  std::vector<float> precap_absz_primary;
  std::vector<float> precap_absz_continuation;
  std::vector<float> precap_absz_purchase;

  // --- PWO-5 section 14.1b: the three collapse canaries. ------------------
  //
  // Measured at the ROLLOUT BEHAVIOUR-POLICY DECISION and nowhere else. The
  // registration considered three candidate sites and rejected the other two:
  // the executed-PPO-minibatch site (existing col 36 `entropy`) is
  // presentation-weighted over an EVOLVING model and is truncated by
  // --target_kl, so an arm that early-stops more often would show a different
  // entropy for reasons unrelated to collapse -- "a safety rail must not be a
  // function of its own optimizer's stopping behaviour"; and a pre-optimizer
  // snapshot would cost a second forward over 32,768 transitions and still
  // lack the role and legal-action data.
  //
  // Population: unique rollout decisions of the acting seat with n_legal >= 2,
  // one measurement each. No re-presentation, no epoch weighting, no dependence
  // on KL early stop.
  //
  // These are NOT column 36 and may never be compared to or substituted for it.
  //
  // Role index is the integer value of DuneDecisionRole (7 members). The
  // kForcedOrBookkeeping slot is identically empty at this site, by the
  // section 14.1c proof: the classifier returns it iff the searched player is
  // the chance player OR n_legal <= 1, chance nodes `continue` before ever
  // reaching the classifier here, and the population already requires
  // n_legal >= 2. It is accumulated anyway so the partition identity is
  // checked rather than assumed.
  double canary_ne_sum = 0.0;             // sum of H_row / log(n_legal)
  int64_t canary_ne_n = 0;                // qualifying decisions
  double canary_ne_sum_role[7] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  int64_t canary_ne_n_role[7] = {0, 0, 0, 0, 0, 0, 0};
  double canary_maxprob_sum = 0.0;        // denominator is canary_ne_n
  int64_t canary_sat_ge_cap = 0;          // legal PRE-cap |z| >= logit_cap
  int64_t canary_sat_logits = 0;          // legal logits examined (LOGIT unit)
};


// Copies BY NAME, not by position.
//
// PWO-5 makes the two models' module sets deliberately asymmetric: the training
// model carries the three auxiliary heads and the inference model does not
// (section 8.6 -- the heads must never be computed in a rollout/inference
// forward, and the cleanest way to guarantee that is for them not to exist
// there). The previous positional loop indexed `target_params[i]` for every `i`
// in the SOURCE's range, so a source with 6 extra tensors would have walked off
// the end of the target.
//
// Name-based copy is also strictly safer than positional in general: it cannot
// silently pair `policy_head.weight` with `value_head.weight` if a future
// registration order changes. When the module sets DO match it copies exactly
// the same tensors as before, so this is behaviour-preserving for every
// pre-PWO-5 caller.
void CopyModelWeights(std::shared_ptr<SharedDunePolicyValueNetImpl> source,
                      std::shared_ptr<SharedDunePolicyValueNetImpl> target) {
  torch::NoGradGuard no_grad;
  auto source_params = source->named_parameters();
  auto source_buffers = source->named_buffers();
  for (auto& item : target->named_parameters()) {
    auto* src = source_params.find(item.key());
    if (src == nullptr) {
      SpielFatalError(
          "CopyModelWeights: target parameter '" + item.key() +
          "' has no counterpart in the source model. The inference model may "
          "be a SUBSET of the training model (PWO-5 auxiliary heads), never a "
          "superset.");
    }
    item.value().copy_(src->to(item.value().device()));
  }
  for (auto& item : target->named_buffers()) {
    auto* src = source_buffers.find(item.key());
    if (src == nullptr) {
      SpielFatalError("CopyModelWeights: target buffer '" + item.key() +
                      "' has no counterpart in the source model.");
    }
    item.value().copy_(src->to(item.value().device()));
  }
}

void SyncModels(std::shared_ptr<SharedDunePolicyValueNetImpl> training_model,
                std::shared_ptr<SharedDunePolicyValueNetImpl> inference_model,
                std::shared_mutex* sync_mutex) {
  std::unique_lock<std::shared_mutex> lock(*sync_mutex);
  CopyModelWeights(training_model, inference_model);
}

std::string HashNonValueParameters(const std::shared_ptr<SharedDunePolicyValueNetImpl>& model) {
  std::stringstream ss;
  torch::NoGradGuard no_grad;
  bool unfreeze_trunk = absl::GetFlag(FLAGS_unfreeze_trunk);
  for (auto& name_param : model->named_parameters()) {
    std::string name = name_param.key();
    bool should_hash = false;
    if (unfreeze_trunk) {
      if (name.rfind("policy_head", 0) == 0) {
        should_hash = true;
      }
    } else {
      if (name.find("value_head") == std::string::npos) {
        should_hash = true;
      }
    }
    if (should_hash) {
      auto tensor = name_param.value().contiguous().cpu().to(torch::kFloat32);
      float* data = static_cast<float*>(tensor.data_ptr());
      size_t num_el = tensor.numel();
      ss.write(reinterpret_cast<const char*>(data), num_el * sizeof(float));
    }
  }
  return open_spiel::ComputeStringSHA256(ss.str());
}

// PWO-5 section 7.2 — THE ARCHITECTURE-VERSIONED CHECKPOINT MIGRATION.
//
// Loads trunk, policy head, value head "WITHOUT NUMERICAL CHANGE" from a
// checkpoint that may PREDATE the three auxiliary heads, and leaves the heads at
// their deterministic init (section 7.2) when the checkpoint does not carry
// them.
//
// The mechanism is the temp-model copy already proven in this file for the
// nonlinear-value-head case below, plus one probe. It relies on a documented
// property of torch: `Module::load(InputArchive&)` iterates the MODULE's own
// parameters/buffers/children and reads each by key, so
//
//   * loading a PRE-migration archive into a HEAD-LESS temp model succeeds, and
//   * a POST-migration archive's extra head entries are simply never visited.
//
// So the temp model reads both generations, and the only question is whether to
// then also pull the heads across -- which `try_read` on the head submodule
// answers without ever partially mutating `model`.
//
// Numerical exactness: the trunk/policy/value tensors are `copy_`d from the
// loaded temp model, which is a bitwise copy of the same dtype and shape. This
// is what makes section 7.3's bitwise legacy-inference parity gate passable.
bool LoadModelCheckpointMigrating(
    std::shared_ptr<SharedDunePolicyValueNetImpl> model,
    const std::string& path, torch::Device device) {
  int64_t input_dim = model->input_layer->weight.size(1);
  int64_t hidden_dim = absl::GetFlag(FLAGS_hidden_dim);
  int64_t action_dim = model->policy_head->weight.size(0);
  int num_blocks = absl::GetFlag(FLAGS_num_blocks);
  const bool use_nonlinear = absl::GetFlag(FLAGS_nonlinear_value_head);

  auto temp_model = std::make_shared<SharedDunePolicyValueNetImpl>(
      input_dim, hidden_dim, action_dim, num_blocks, use_nonlinear,
      /*with_aux_heads=*/false);
  temp_model->to(device);
  torch::load(temp_model, path, device);

  torch::NoGradGuard no_grad;
  // Trunk.
  model->input_layer->weight.copy_(temp_model->input_layer->weight);
  if (model->input_layer->bias.defined() &&
      temp_model->input_layer->bias.defined()) {
    model->input_layer->bias.copy_(temp_model->input_layer->bias);
  }
  for (size_t i = 0; i < model->res_blocks.size(); ++i) {
    auto source_params = temp_model->res_blocks[i]->parameters();
    auto target_params = model->res_blocks[i]->parameters();
    for (size_t j = 0; j < source_params.size(); ++j) {
      target_params[j].copy_(source_params[j]);
    }
    auto source_buffers = temp_model->res_blocks[i]->buffers();
    auto target_buffers = model->res_blocks[i]->buffers();
    for (size_t j = 0; j < source_buffers.size(); ++j) {
      target_buffers[j].copy_(source_buffers[j]);
    }
  }
  // Policy head.
  model->policy_head->weight.copy_(temp_model->policy_head->weight);
  if (model->policy_head->bias.defined() &&
      temp_model->policy_head->bias.defined()) {
    model->policy_head->bias.copy_(temp_model->policy_head->bias);
  }
  // Value head(s). Unlike the nonlinear partial-copy path below, the migration
  // DOES carry the value head across: section 7.2 requires the value head load
  // "without numerical change", and a fresh critic here would silently restart
  // the value function at update 0 of every arm.
  model->value_head->weight.copy_(temp_model->value_head->weight);
  if (model->value_head->bias.defined() &&
      temp_model->value_head->bias.defined()) {
    model->value_head->bias.copy_(temp_model->value_head->bias);
  }
  if (use_nonlinear) {
    model->value_head2->weight.copy_(temp_model->value_head2->weight);
    if (model->value_head2->bias.defined() &&
        temp_model->value_head2->bias.defined()) {
      model->value_head2->bias.copy_(temp_model->value_head2->bias);
    }
  }

  // Now the heads, if and only if this checkpoint already has them.
  bool migrated = true;
  if (model->has_aux_heads_) {
    torch::serialize::InputArchive archive;
    archive.load_from(path, device);
    torch::serialize::InputArchive head_archive;
    if (archive.try_read("final_vp_head", head_archive)) {
      // Post-migration checkpoint: load all three normally.
      model->final_vp_head->load(head_archive);
      torch::serialize::InputArchive a2, a3;
      if (!archive.try_read("terminal_round_head", a2) ||
          !archive.try_read("next_own_action_head", a3)) {
        SpielFatalError(
            "PWO-5 migration: the checkpoint at " + path +
            " carries final_vp_head but not all three auxiliary heads. A "
            "partially migrated checkpoint is a STOP, not something to repair "
            "silently -- section 17.5 item 4.");
      }
      model->terminal_round_head->load(a2);
      model->next_own_action_head->load(a3);
      migrated = false;  // it was ALREADY migrated
    } else {
      std::cout << "[PWO-5] MIGRATION: " << path
                << " predates the auxiliary heads. Trunk, policy head and "
                   "value head loaded without numerical change; the three "
                   "heads keep their deterministic init from "
                   "kHeadInitConstant=" << absl::GetFlag(FLAGS_head_init_constant)
                << "." << std::endl;
    }
  }
  return migrated;
}

void LoadModelCheckpoint(std::shared_ptr<SharedDunePolicyValueNetImpl> model,
                         const std::string& path, torch::Device device) {
  // PWO-5: when the model carries the auxiliary heads, the migrating loader is
  // the only correct path -- a plain torch::load would demand head tensors the
  // base checkpoint does not have and would abort the run.
  if (model->has_aux_heads_) {
    LoadModelCheckpointMigrating(model, path, device);
    return;
  }
  if (!absl::GetFlag(FLAGS_nonlinear_value_head)) {
    torch::load(model, path, device);
    return;
  }
  int64_t input_dim = model->input_layer->weight.size(1);
  int64_t hidden_dim = absl::GetFlag(FLAGS_hidden_dim);
  int64_t action_dim = model->policy_head->weight.size(0);
  int num_blocks = absl::GetFlag(FLAGS_num_blocks);

  auto temp_model = std::make_shared<SharedDunePolicyValueNetImpl>(
      input_dim, hidden_dim, action_dim, num_blocks, /*use_nonlinear=*/false);
  temp_model->to(device);

  std::cout << "[INFO] Loading standard checkpoint for partial weight copy (nonlinear model)..." << std::endl;
  torch::load(temp_model, path, device);

  torch::NoGradGuard no_grad;
  model->input_layer->weight.copy_(temp_model->input_layer->weight);
  if (model->input_layer->bias.defined() && temp_model->input_layer->bias.defined()) {
    model->input_layer->bias.copy_(temp_model->input_layer->bias);
  }
  for (size_t i = 0; i < model->res_blocks.size(); ++i) {
    auto& target_block = model->res_blocks[i];
    auto& source_block = temp_model->res_blocks[i];
    auto source_params = source_block->parameters();
    auto target_params = target_block->parameters();
    for (size_t j = 0; j < source_params.size(); ++j) {
      target_params[j].copy_(source_params[j]);
    }
    auto source_buffers = source_block->buffers();
    auto target_buffers = target_block->buffers();
    for (size_t j = 0; j < source_buffers.size(); ++j) {
      target_buffers[j].copy_(source_buffers[j]);
    }
  }
  model->policy_head->weight.copy_(temp_model->policy_head->weight);
  if (model->policy_head->bias.defined() && temp_model->policy_head->bias.defined()) {
    model->policy_head->bias.copy_(temp_model->policy_head->bias);
  }
  std::cout << "[INFO] Partial weight copy completed successfully." << std::endl;
}

void SetOptimizerLearningRate(torch::optim::Optimizer& optimizer, double lr) {
  for (auto& group : optimizer.param_groups()) {
    auto& options = static_cast<torch::optim::AdamWOptions&>(group.options());
    options.lr(lr);
  }
}

std::unique_ptr<torch::optim::AdamW> MakeOptimizer(
    std::shared_ptr<SharedDunePolicyValueNetImpl> model) {
  if (absl::GetFlag(FLAGS_train_value_only)) {
    std::vector<torch::Tensor> trainable_params;
    for (auto& param : model->parameters()) {
      if (param.requires_grad()) {
        trainable_params.push_back(param);
      }
    }
    std::vector<torch::optim::OptimizerParamGroup> groups;
    groups.emplace_back(trainable_params);
    auto optimizer = std::make_unique<torch::optim::AdamW>(
        groups, torch::optim::AdamWOptions(absl::GetFlag(FLAGS_learning_rate)).eps(1e-5));
    static_cast<torch::optim::AdamWOptions&>(
        optimizer->param_groups()[0].options())
        .weight_decay(absl::GetFlag(FLAGS_weight_decay));
    return optimizer;
  }

  std::vector<torch::Tensor> policy_params;
  std::vector<torch::Tensor> other_params;
  // PWO-5 section 7.4: the three auxiliary heads sit in their OWN parameter
  // group with weight_decay = 0.0, IN ALL SIX ARMS.
  //
  // This is not tidiness. In a head-off arm (T, P) the coefficients are exactly
  // zero and the loss terms are not constructed, so the head parameters receive
  // no gradient -- but AdamW's DECOUPLED weight decay does not need a gradient
  // to move a parameter. Left in the default group at --weight_decay, a
  // head-off arm's head tensors would shrink on every step and would silently
  // diverge from a head-on arm's initial values, breaking both the section 9
  // "same tensors, same keys, same order" match and section 15.2 gate 7's
  // bitwise-equal-to-initial check.
  std::vector<torch::Tensor> aux_head_params;
  auto policy_params_set = model->policy_head->parameters();
  std::vector<torch::Tensor> aux_params_set;
  if (model->has_aux_heads_) {
    for (auto& p : model->final_vp_head->parameters()) aux_params_set.push_back(p);
    for (auto& p : model->terminal_round_head->parameters()) aux_params_set.push_back(p);
    for (auto& p : model->next_own_action_head->parameters()) aux_params_set.push_back(p);
  }
  for (auto& param : model->parameters()) {
    bool is_policy = false;
    for (auto& policy_param : policy_params_set) {
      if (param.is_same(policy_param)) {
        is_policy = true;
        break;
      }
    }
    bool is_aux = false;
    for (auto& aux_param : aux_params_set) {
      if (param.is_same(aux_param)) {
        is_aux = true;
        break;
      }
    }
    if (is_policy) {
      policy_params.push_back(param);
    } else if (is_aux) {
      aux_head_params.push_back(param);
    } else {
      other_params.push_back(param);
    }
  }

  // Group order is part of the checkpoint contract (section 9: "same
  // param_groups, same order"), and the post-load re-assertion below indexes by
  // it. Order: 0 = policy, 1 = everything else, 2 = the auxiliary heads.
  std::vector<torch::optim::OptimizerParamGroup> groups;
  groups.emplace_back(policy_params);
  groups.emplace_back(other_params);
  if (model->has_aux_heads_) {
    groups.emplace_back(aux_head_params);
  }
  auto optimizer = std::make_unique<torch::optim::AdamW>(
      groups, torch::optim::AdamWOptions(absl::GetFlag(FLAGS_learning_rate))
                  .eps(1e-5));
  static_cast<torch::optim::AdamWOptions&>(
      optimizer->param_groups()[0].options())
      .weight_decay(absl::GetFlag(FLAGS_policy_weight_decay));
  static_cast<torch::optim::AdamWOptions&>(
      optimizer->param_groups()[1].options())
      .weight_decay(absl::GetFlag(FLAGS_weight_decay));
  if (model->has_aux_heads_) {
    static_cast<torch::optim::AdamWOptions&>(
        optimizer->param_groups()[2].options())
        .weight_decay(0.0);  // section 7.4 -- hard zero, never a flag
  }
  return optimizer;
}

// PWO-5 section 7.2 — the OPTIMIZER half of the migration.
//
// The base optimizer checkpoint was written by an AdamW over the PRE-head
// parameter set in TWO groups. The migrated optimizer has THREE groups and 6
// more tensors, so `torch::load` straight into it cannot work: the archive and
// the optimizer disagree about how many parameters exist.
//
// The migration mirrors the model's: build a TEMPORARY AdamW whose group
// structure is exactly the pre-migration one -- and, critically, whose tensors
// are THE REAL MODEL'S OWN non-head parameters, not copies. AdamW keys its state
// map by `TensorImpl*`, so after loading, the temp optimizer's state is already
// keyed by the identities the real optimizer uses, and the entries can simply be
// moved across. Nothing is re-derived and no state tensor is reconstructed, so
// "optimizer state loaded without numerical change" is exact.
//
// Returns true if it migrated (the archive was pre-head), false if the caller
// should do an ordinary load.
bool LoadOptimizerCheckpointMigrating(
    std::shared_ptr<SharedDunePolicyValueNetImpl> model,
    torch::optim::AdamW& optimizer, const std::string& path,
    torch::Device device) {
  if (!model->has_aux_heads_) return false;

  // Reproduce the pre-migration grouping over the real model's tensors.
  std::vector<torch::Tensor> policy_params;
  std::vector<torch::Tensor> other_params;
  auto policy_params_set = model->policy_head->parameters();
  std::vector<torch::Tensor> aux_params_set;
  for (auto& p : model->final_vp_head->parameters()) aux_params_set.push_back(p);
  for (auto& p : model->terminal_round_head->parameters()) aux_params_set.push_back(p);
  for (auto& p : model->next_own_action_head->parameters()) aux_params_set.push_back(p);
  for (auto& param : model->parameters()) {
    bool is_policy = false;
    for (auto& q : policy_params_set) {
      if (param.is_same(q)) { is_policy = true; break; }
    }
    bool is_aux = false;
    for (auto& q : aux_params_set) {
      if (param.is_same(q)) { is_aux = true; break; }
    }
    if (is_policy) policy_params.push_back(param);
    else if (!is_aux) other_params.push_back(param);
  }

  std::vector<torch::optim::OptimizerParamGroup> legacy_groups;
  legacy_groups.emplace_back(policy_params);
  legacy_groups.emplace_back(other_params);
  torch::optim::AdamW legacy_optimizer(
      legacy_groups,
      torch::optim::AdamWOptions(absl::GetFlag(FLAGS_learning_rate)).eps(1e-5));

  try {
    torch::load(legacy_optimizer, path, device);
  } catch (const c10::Error& e) {
    // Not a pre-migration archive (or a genuinely broken one). Let the caller's
    // ordinary load run and report its own error, so a real corruption is not
    // misreported as a migration failure.
    return false;
  }

  int moved = 0;
  for (auto& entry : legacy_optimizer.state()) {
    optimizer.state()[entry.first] = std::move(entry.second);
    ++moved;
  }
  std::cout << "[PWO-5] MIGRATION: optimizer state moved for " << moved
            << " pre-head parameters from " << path
            << " without numerical change; the three heads' state is "
               "materialized separately at zero." << std::endl;
  return true;
}

// PWO-5 section 7.4 — MATERIALIZE the auxiliary heads' optimizer state.
//
// "This is a registered requirement, not a framework default." With the
// coefficient short-circuit, a head-off arm's head parameters never receive a
// gradient, and AdamW creates per-parameter state LAZILY on the first gradient.
// A head-off arm's optimizer state dict would therefore be MISSING those keys
// while a head-on arm's has them -- breaking section 9's "same tensors, same
// keys, same order" and making section 15.2 gate 7's "auxiliary optimizer state
// all-zero" unverifiable, because ABSENT IS NOT ZERO.
//
// Called in all six arms, whether or not a gradient ever flows.
void MaterializeAuxOptimizerState(
    std::shared_ptr<SharedDunePolicyValueNetImpl> model,
    torch::optim::AdamW& optimizer) {
  if (!model->has_aux_heads_) return;
  if (optimizer.param_groups().size() < 3) {
    SpielFatalError(
        "PWO-5 section 7.4: the model has auxiliary heads but the optimizer has "
        "fewer than three parameter groups. The head group is missing.");
  }
  torch::NoGradGuard no_grad;
  int materialized = 0;
  int already_present = 0;
  for (auto& param : optimizer.param_groups()[2].params()) {
    auto key = param.unsafeGetTensorImpl();
    if (optimizer.state().count(key) > 0) {
      ++already_present;
      continue;
    }
    auto state = std::make_unique<torch::optim::AdamWParamState>();
    state->step(0);
    state->exp_avg(torch::zeros_like(param, torch::MemoryFormat::Preserve));
    state->exp_avg_sq(torch::zeros_like(param, torch::MemoryFormat::Preserve));
    optimizer.state()[key] = std::move(state);
    ++materialized;
  }
  std::cout << "[PWO-5] auxiliary optimizer state: materialized "
            << materialized << " entries (step=0, exp_avg=0, exp_avg_sq=0), "
            << already_present << " already present." << std::endl;
}

std::string GenerateUUID() {
  std::random_device rd;
  std::mt19937_64 generator(rd());
  uint64_t r1 = generator();
  uint64_t r2 = generator();
  // Set version 4 (random) and variant (RFC 4122)
  uint32_t a = (r1 >> 32);
  uint16_t b = (r1 >> 16) & 0xFFFF;
  uint16_t c = (r1 & 0x0FFF) | 0x4000;
  uint16_t d = ((r2 >> 48) & 0x3FFF) | 0x8000;
  uint64_t e = r2 & 0xFFFFFFFFFFFFULL;
  return absl::StrFormat("%08x-%04x-%04x-%04x-%012llx", a, b, c, d, e);
}

// Acceptance prior source this run will actually use: the flag when stated,
// otherwise the collector's library default. Unrecognized names are fatal —
// never silently fall back, since the whole point of the field is that the
// coverage contract is chosen explicitly (WO-20).
AcceptancePriorSource EffectiveAcceptancePriorSource() {
  const std::string name = absl::GetFlag(FLAGS_collector_acceptance_prior);
  if (name.empty()) return OnlineSearchConfig().acceptance_prior_source;
  AcceptancePriorSource source;
  if (!ParseAcceptancePriorSource(name, &source)) {
    SpielFatalError(absl::StrCat(
        "--collector_acceptance_prior=", name, " is not a known prior source. ",
        "Use '", AcceptancePriorSourceName(AcceptancePriorSource::kRawNetworkPrior),
        "' or '", AcceptancePriorSourceName(AcceptancePriorSource::kTreePrior),
        "'."));
  }
  return source;
}

std::string ComputeLegacyConfigFingerprint() {
  open_spiel::json::Object config_obj;
  config_obj["game"] = absl::GetFlag(FLAGS_game);
  config_obj["ppo_minibatch_size"] = absl::GetFlag(FLAGS_ppo_minibatch_size);
  config_obj["ppo_update_epochs"] = absl::GetFlag(FLAGS_ppo_update_epochs);
  config_obj["learning_rate"] = absl::GetFlag(FLAGS_learning_rate);
  config_obj["anneal_lr"] = absl::GetFlag(FLAGS_anneal_lr);
  config_obj["gamma"] = absl::GetFlag(FLAGS_gamma);
  config_obj["gae_lambda"] = absl::GetFlag(FLAGS_gae_lambda);
  config_obj["normalize_advantages"] = absl::GetFlag(FLAGS_normalize_advantages);
  config_obj["ppo_clip_epsilon"] = absl::GetFlag(FLAGS_ppo_clip_epsilon);
  config_obj["ppo_clip_value_loss"] = absl::GetFlag(FLAGS_ppo_clip_value_loss);
  config_obj["entropy_coef"] = absl::GetFlag(FLAGS_entropy_coef);
  config_obj["value_coef"] = absl::GetFlag(FLAGS_value_coef);
  config_obj["grad_clip_norm"] = absl::GetFlag(FLAGS_grad_clip_norm);
  config_obj["target_kl"] = absl::GetFlag(FLAGS_target_kl);
  config_obj["weight_decay"] = absl::GetFlag(FLAGS_weight_decay);
  config_obj["policy_weight_decay"] = absl::GetFlag(FLAGS_policy_weight_decay);
  config_obj["hidden_dim"] = absl::GetFlag(FLAGS_hidden_dim);
  config_obj["num_blocks"] = absl::GetFlag(FLAGS_num_blocks);
  config_obj["logit_cap"] = absl::GetFlag(FLAGS_logit_cap);
  config_obj["shaped_reward_weight"] = absl::GetFlag(FLAGS_shaped_reward_weight);
  config_obj["tleilaxu_breadcrumb_weight"] = absl::GetFlag(FLAGS_tleilaxu_breadcrumb_weight);
  config_obj["tleilaxu_level7_breadcrumb_weight"] = absl::GetFlag(FLAGS_tleilaxu_level7_breadcrumb_weight);
  config_obj["specimen_exchange_penalty"] = absl::GetFlag(FLAGS_specimen_exchange_penalty);
  config_obj["shaping_start_env_steps"] = static_cast<int64_t>(absl::GetFlag(FLAGS_shaping_start_env_steps));
  config_obj["shaping_decay_env_steps"] = static_cast<int64_t>(absl::GetFlag(FLAGS_shaping_decay_env_steps));
  config_obj["reward_scale"] = absl::GetFlag(FLAGS_reward_scale);
  config_obj["train_amp"] = absl::GetFlag(FLAGS_train_amp);
  config_obj["pipeline"] = absl::GetFlag(FLAGS_pipeline);
  config_obj["rollout_games"] = absl::GetFlag(FLAGS_rollout_games);
  config_obj["rollout_transitions"] = absl::GetFlag(FLAGS_rollout_transitions);
  config_obj["search_lambda"] = absl::GetFlag(FLAGS_search_lambda);
  config_obj["search_minibatches_per_update"] = absl::GetFlag(FLAGS_search_minibatches_per_update);
  config_obj["search_minibatch_size"] = absl::GetFlag(FLAGS_search_minibatch_size);

  // Phase 18B online auxiliary-search collection (combined optimization).
  config_obj["online_search_collection"] = absl::GetFlag(FLAGS_online_search_collection);
  config_obj["auxiliary_games"] = absl::GetFlag(FLAGS_auxiliary_games);
  config_obj["auxiliary_search_seed_domain"] = static_cast<int64_t>(absl::GetFlag(FLAGS_auxiliary_search_seed_domain));
  config_obj["collector_dirichlet_epsilon"] = absl::GetFlag(FLAGS_collector_dirichlet_epsilon);
  config_obj["search_loss_coef"] = absl::GetFlag(FLAGS_search_loss_coef);
  config_obj["search_loss_warmup_update"] = absl::GetFlag(FLAGS_search_loss_warmup_update);
  config_obj["abort_grad_norm_ratio"] = absl::GetFlag(FLAGS_abort_grad_norm_ratio);
  config_obj["target_sharpen_exponent"] = absl::GetFlag(FLAGS_target_sharpen_exponent);
  config_obj["swordmaster_grant_fraction"] = absl::GetFlag(FLAGS_swordmaster_grant_fraction);
  config_obj["swordmaster_grant_round"] = absl::GetFlag(FLAGS_swordmaster_grant_round);

  std::string json_str = open_spiel::json::ToString(config_obj);
  return open_spiel::ComputeStringSHA256(json_str);
}

std::string ComputeConfigFingerprint() {
  open_spiel::json::Object config_obj;
  config_obj["game"] = absl::GetFlag(FLAGS_game);
  config_obj["ppo_minibatch_size"] = absl::GetFlag(FLAGS_ppo_minibatch_size);
  config_obj["ppo_update_epochs"] = absl::GetFlag(FLAGS_ppo_update_epochs);
  config_obj["learning_rate"] = absl::GetFlag(FLAGS_learning_rate);
  config_obj["anneal_lr"] = absl::GetFlag(FLAGS_anneal_lr);
  config_obj["gamma"] = absl::GetFlag(FLAGS_gamma);
  config_obj["gae_lambda"] = absl::GetFlag(FLAGS_gae_lambda);
  config_obj["normalize_advantages"] = absl::GetFlag(FLAGS_normalize_advantages);
  config_obj["ppo_clip_epsilon"] = absl::GetFlag(FLAGS_ppo_clip_epsilon);
  config_obj["ppo_clip_value_loss"] = absl::GetFlag(FLAGS_ppo_clip_value_loss);
  config_obj["entropy_coef"] = absl::GetFlag(FLAGS_entropy_coef);
  config_obj["value_coef"] = absl::GetFlag(FLAGS_value_coef);
  config_obj["grad_clip_norm"] = absl::GetFlag(FLAGS_grad_clip_norm);
  config_obj["target_kl"] = absl::GetFlag(FLAGS_target_kl);
  config_obj["weight_decay"] = absl::GetFlag(FLAGS_weight_decay);
  config_obj["policy_weight_decay"] = absl::GetFlag(FLAGS_policy_weight_decay);
  config_obj["hidden_dim"] = absl::GetFlag(FLAGS_hidden_dim);
  config_obj["num_blocks"] = absl::GetFlag(FLAGS_num_blocks);
  config_obj["logit_cap"] = absl::GetFlag(FLAGS_logit_cap);
  config_obj["shaped_reward_weight"] = absl::GetFlag(FLAGS_shaped_reward_weight);
  config_obj["tleilaxu_breadcrumb_weight"] = absl::GetFlag(FLAGS_tleilaxu_breadcrumb_weight);
  config_obj["tleilaxu_level7_breadcrumb_weight"] = absl::GetFlag(FLAGS_tleilaxu_level7_breadcrumb_weight);
  config_obj["specimen_exchange_penalty"] = absl::GetFlag(FLAGS_specimen_exchange_penalty);
  config_obj["shaping_start_env_steps"] = static_cast<int64_t>(absl::GetFlag(FLAGS_shaping_start_env_steps));
  config_obj["shaping_decay_env_steps"] = static_cast<int64_t>(absl::GetFlag(FLAGS_shaping_decay_env_steps));
  config_obj["reward_scale"] = absl::GetFlag(FLAGS_reward_scale);
  config_obj["train_amp"] = absl::GetFlag(FLAGS_train_amp);
  config_obj["pipeline"] = absl::GetFlag(FLAGS_pipeline);
  config_obj["rollout_games"] = absl::GetFlag(FLAGS_rollout_games);
  config_obj["rollout_transitions"] = absl::GetFlag(FLAGS_rollout_transitions);
  config_obj["search_lambda"] = absl::GetFlag(FLAGS_search_lambda);
  config_obj["search_minibatches_per_update"] = absl::GetFlag(FLAGS_search_minibatches_per_update);
  config_obj["search_minibatch_size"] = absl::GetFlag(FLAGS_search_minibatch_size);

  // Phase 18B online auxiliary-search collection (combined optimization).
  config_obj["online_search_collection"] = absl::GetFlag(FLAGS_online_search_collection);
  config_obj["auxiliary_games"] = absl::GetFlag(FLAGS_auxiliary_games);
  config_obj["auxiliary_search_seed_domain"] = static_cast<int64_t>(absl::GetFlag(FLAGS_auxiliary_search_seed_domain));
  config_obj["collector_dirichlet_epsilon"] = absl::GetFlag(FLAGS_collector_dirichlet_epsilon);
  config_obj["search_loss_coef"] = absl::GetFlag(FLAGS_search_loss_coef);
  config_obj["search_loss_warmup_update"] = absl::GetFlag(FLAGS_search_loss_warmup_update);
  config_obj["abort_grad_norm_ratio"] = absl::GetFlag(FLAGS_abort_grad_norm_ratio);
  config_obj["target_sharpen_exponent"] = absl::GetFlag(FLAGS_target_sharpen_exponent);
  config_obj["swordmaster_grant_fraction"] = absl::GetFlag(FLAGS_swordmaster_grant_fraction);
  config_obj["swordmaster_grant_round"] = absl::GetFlag(FLAGS_swordmaster_grant_round);
  // Acceptance prior source is training semantics, not a launcher detail: it
  // decides which searches become CE targets. Fingerprinted so two runs that
  // measure coverage differently cannot stamp their examples with the same
  // provenance hash (WO-20). Deliberately NOT added to the legacy fingerprint
  // above — that one is a frozen field set whose job is to recognize older
  // manifests, so extending it would defeat its purpose.
  config_obj["collector_acceptance_prior"] =
      std::string(AcceptancePriorSourceName(EffectiveAcceptancePriorSource()));

  // New flags added for complete config fingerprint validation
  config_obj["deterministic"] = absl::GetFlag(FLAGS_deterministic);
  config_obj["deterministic_rollout_eval"] = absl::GetFlag(FLAGS_deterministic_rollout_eval);
  // Critic-remediation settings are training semantics, not launcher details.
  // Persist them in the fingerprint so a partially completed value-only run
  // cannot be resumed with a different set of trainable parameters or data.
  config_obj["train_value_only"] = absl::GetFlag(FLAGS_train_value_only);
  config_obj["unfreeze_trunk"] = absl::GetFlag(FLAGS_unfreeze_trunk);
  config_obj["nonlinear_value_head"] =
      absl::GetFlag(FLAGS_nonlinear_value_head);
  config_obj["sample_counterfactual_states"] =
      absl::GetFlag(FLAGS_sample_counterfactual_states);
  config_obj["counterfactual_samples_per_game"] =
      absl::GetFlag(FLAGS_counterfactual_samples_per_game);
  config_obj["counterfactual_replay_weight"] =
      absl::GetFlag(FLAGS_counterfactual_replay_weight);
  config_obj["policy_kl_anchor_coeff"] =
      absl::GetFlag(FLAGS_policy_kl_anchor_coeff);

  std::string json_str = open_spiel::json::ToString(config_obj);
  return open_spiel::ComputeStringSHA256(json_str);
}

// ---------------------------------------------------------------------------
// PWO-5 manifest / resume contract (registration Appendix A.1 note 3).
// ---------------------------------------------------------------------------
//
// Appendix A.1 registers that "the manifest gains matching fields, asserted on
// resume and on the u600 extension": both target flags, the held-out split
// digest, the three head coefficients, the three sampler-shape flags, the
// Huber delta, the head-init constant, --emit_canary_columns, and section
// 8.6's update-1 sampler digest.
//
// These are kept OUT of ComputeConfigFingerprint deliberately. That fingerprint
// is a frozen field set whose job is to recognize older manifests; extending it
// would make every pre-PWO-5 checkpoint fail to resume as an ANONYMOUS hash
// mismatch. A dedicated block instead fails with the field NAME that moved,
// and leaves every legacy fingerprint verifying exactly as before.
//
// Every double is serialized as a %.17g STRING, never as a JSON number:
// open_spiel's json.cc writer emits doubles as `%f` at six decimals, so a
// coefficient of 5e-8 would persist as "0.000000" and compare equal to zero.
std::string Pwo5Double(double v) {
  char buf[64];
  std::snprintf(buf, sizeof(buf), "%.17g", v);
  return std::string(buf);
}

// Populated once at startup by BuildPwo5ManifestFields(), then merged into
// every manifest this process writes. Empty when the run is not a PWO-5 arm
// (no --aux_target_path), in which case nothing is written and nothing is
// asserted -- a legacy run is untouched.
open_spiel::json::Object g_pwo5_manifest_fields;
bool g_pwo5_manifest_active = false;

// A canonical, order-stable digest over every field above, so a single
// comparison covers the whole block and a mismatch cannot be missed by a
// checker that forgot one key. json::Object is a std::map, so ToString emits
// keys in lexicographic order regardless of insertion order.
std::string ComputePwo5Fingerprint(const open_spiel::json::Object& fields) {
  return open_spiel::ComputeStringSHA256(open_spiel::json::ToString(fields));
}

// The head-initialization IDENTITY: a digest over the three auxiliary heads'
// parameter tensors as they stand immediately after construction/migration.
//
// Section 7.2 requires the initial head parameters be BYTE-IDENTICAL on all six
// arms (derived from kHeadInitConstant, not from any run seed). A digest makes
// that checkable rather than asserted, and it is what section 15.2 gate 7's
// "head parameters bitwise equal their initial values" is compared against.
std::string ComputeAuxHeadIdentity(
    const std::shared_ptr<SharedDunePolicyValueNetImpl>& model) {
  torch::NoGradGuard no_grad;
  std::string blob;
  auto append = [&blob](const torch::Tensor& t) {
    torch::Tensor c = t.detach().to(torch::kCPU).contiguous().to(torch::kFloat32);
    const auto n = c.numel();
    const char* p = reinterpret_cast<const char*>(c.data_ptr<float>());
    blob.append(p, static_cast<size_t>(n) * sizeof(float));
  };
  for (auto& p : model->final_vp_head->parameters()) append(p);
  for (auto& p : model->terminal_round_head->parameters()) append(p);
  for (auto& p : model->next_own_action_head->parameters()) append(p);
  return open_spiel::ComputeStringSHA256(blob);
}

std::string GetSearchLabelFingerprint(const std::string& search_label_dir) {
  if (search_label_dir.empty()) {
    return "";
  }
  std::filesystem::path manifest_path = std::filesystem::path(search_label_dir) / "manifest.json";
  if (!std::filesystem::exists(manifest_path)) {
    std::cout << "Warning: search_label_dir is specified but manifest.json not found at " << manifest_path << "\n";
    return "";
  }
  try {
    std::ifstream ifs(manifest_path);
    std::string content((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
    auto val_opt = open_spiel::json::FromString(content);
    if (val_opt.has_value() && val_opt->IsObject()) {
      const auto& obj = val_opt->GetObject();
      auto it = obj.find("search_label_fingerprint");
      if (it != obj.end() && it->second.IsString()) {
        return it->second.GetString();
      }
    }
  } catch (const std::exception& e) {
    std::cerr << "Error reading search label manifest at " << manifest_path << ": " << e.what() << "\n";
  }
  return "";
}

void SaveCheckpoint(std::shared_ptr<SharedDunePolicyValueNetImpl> model,
                    torch::optim::AdamW& optimizer,
                    const std::string& model_path,
                    const std::string& optim_path,
                    int global_update,
                    int target_end_update,
                    uint64_t total_env_steps,
                    uint64_t next_episode_id,
                    uint64_t base_seed,
                    int seed_scheme_version,
                    const std::string& config_fingerprint,
                    const std::string& search_label_fingerprint,
                    const std::string& run_uuid,
                    const open_spiel::OnlineCollectionState* aux_state = nullptr) {
  std::string model_tmp = model_path + ".tmp";
  std::string optim_tmp = optim_path + ".tmp";

  std::filesystem::path manifest_path = model_path;
  manifest_path.replace_extension(".json");
  std::string manifest_path_str = manifest_path.string();
  std::string manifest_tmp = manifest_path_str + ".tmp";

  try {
    torch::save(model, model_tmp);
    torch::save(optimizer, optim_tmp);

    size_t model_size = 0;
    std::string model_hash = ComputeFileSHA256(model_tmp, &model_size);

    size_t optim_size = 0;
    std::string optim_hash = ComputeFileSHA256(optim_tmp, &optim_size);

    std::string checkpoint_uuid = GenerateUUID();
    json::Object manifest_obj;
    manifest_obj["schema_version"] = static_cast<int64_t>(2);
    manifest_obj["checkpoint_uuid"] = checkpoint_uuid;
    manifest_obj["global_update"] = static_cast<int64_t>(global_update);
    manifest_obj["target_end_update"] = static_cast<int64_t>(target_end_update);
    manifest_obj["total_env_steps"] = static_cast<int64_t>(total_env_steps);
    manifest_obj["next_episode_id"] = static_cast<int64_t>(next_episode_id);
    manifest_obj["base_seed"] = static_cast<int64_t>(base_seed);
    manifest_obj["seed_scheme_version"] = static_cast<int64_t>(seed_scheme_version);
    manifest_obj["config_fingerprint"] = config_fingerprint;
    manifest_obj["search_label_fingerprint"] = search_label_fingerprint;
    manifest_obj["run_uuid"] = run_uuid;
    manifest_obj["model_filename"] = std::filesystem::path(model_path).filename().string();
    manifest_obj["model_file_size"] = static_cast<int64_t>(model_size);
    manifest_obj["model_sha256"] = model_hash;
    manifest_obj["optimizer_filename"] = std::filesystem::path(optim_path).filename().string();
    manifest_obj["optimizer_file_size"] = static_cast<int64_t>(optim_size);
    manifest_obj["optimizer_sha256"] = optim_hash;
    manifest_obj["hidden_dim"] = static_cast<int64_t>(absl::GetFlag(FLAGS_hidden_dim));
    manifest_obj["num_blocks"] = static_cast<int64_t>(absl::GetFlag(FLAGS_num_blocks));
    // PWO-5 Appendix A.1 note 3: the matching fields, asserted on resume.
    if (g_pwo5_manifest_active) {
      manifest_obj["pwo5"] = json::Value(g_pwo5_manifest_fields);
      manifest_obj["pwo5_fingerprint"] =
          json::Value(ComputePwo5Fingerprint(g_pwo5_manifest_fields));
    }
    // Phase 18B: persist online-collection state for exact resume (plan §18C).
    if (aux_state != nullptr) {
      open_spiel::WriteOnlineCollectionState(manifest_obj, *aux_state);
    }

    {
      std::ofstream ofs(manifest_tmp);
      if (!ofs) {
        throw std::runtime_error("Could not open manifest temp file for writing: " + manifest_tmp);
      }
      ofs << json::ToString(manifest_obj, true);
    }

    std::filesystem::rename(model_tmp, model_path);
    std::filesystem::rename(optim_tmp, optim_path);
    std::filesystem::rename(manifest_tmp, manifest_path);

    std::cout << "Saved checkpoint successfully:\n"
              << "  Model: " << model_path << " (" << model_size << " bytes, sha256=" << model_hash << ")\n"
              << "  Optimizer: " << optim_path << " (" << optim_size << " bytes, sha256=" << optim_hash << ")\n"
              << "  Manifest: " << manifest_path << "\n";
  } catch (const std::exception& e) {
    std::cerr << "CRITICAL ERROR: SaveCheckpoint failed: " << e.what() << "\n";
    if (std::filesystem::exists(model_tmp)) std::filesystem::remove(model_tmp);
    if (std::filesystem::exists(optim_tmp)) std::filesystem::remove(optim_tmp);
    if (std::filesystem::exists(manifest_tmp)) std::filesystem::remove(manifest_tmp);
    SpielFatalError("SaveCheckpoint failed, aborting training to prevent corrupted states.");
  }
}

struct CounterfactualPending {
  PpoTransition transition;
  int replay_weight;
};

int PpoSimulation(uint64_t master, uint64_t episode_id, const Game& game,
                  std::shared_ptr<IGameEvaluator> evaluator, int64_t obs_size,
                  std::vector<PpoTransition>* trajectory,
                  std::atomic<uint64_t>* total_env_steps,
                  float reward_lambda,
                  WorkerStats* local_stats) {

  auto chance_rng = dune_seed::MakeRng64(dune_seed::DeriveSeed(master, dune_seed::kDomainTrain, episode_id, dune_seed::kStreamChance));
  std::mt19937_64 policy_rng[4];
  for (int p = 0; p < 4; ++p) {
    uint64_t seed = dune_seed::DeriveSeed(master, dune_seed::kDomainTrain, episode_id, dune_seed::kStreamPolicyPlayer0 + p);
    policy_rng[p] = dune_seed::MakeRng64(seed);
  }

  while (true) {
    std::unique_ptr<State> state = game.NewInitialState();
    bool provides_info_state_tensor =
        game.GetType().provides_information_state_tensor;
    bool provides_observations_tensor =
        game.GetType().provides_observation_tensor;

    const auto* dune_state =
        dynamic_cast<const dune_imperium::DuneImperiumState*>(state.get());

    std::vector<int> current_tleilaxu(game.NumPlayers(), 0);
    std::vector<bool> had_swordmaster(game.NumPlayers(), false);
    std::vector<int> prev_conflict_vp(game.NumPlayers(), 0);
    std::vector<int> prev_total_vp(game.NumPlayers(), 0);
    if (dune_state != nullptr) {
      for (int p = 0; p < game.NumPlayers(); ++p) {
        current_tleilaxu[p] = dune_state->GetTleilaxuTrackForTesting(p);
        had_swordmaster[p] = dune_state->HasSwordmaster(p);
        prev_conflict_vp[p] = dune_state->ConflictVpDelta(p);
        prev_total_vp[p] = dune_state->GetPlayerVp(p);
      }
    }

    CombatCreditAccumulator combat_accumulator;
    std::vector<int> last_transition_index(game.NumPlayers(), -1);
    std::vector<int> pre_combat_strength(game.NumPlayers(), 0);
    dune_imperium::GamePhase pre_action_phase = dune_imperium::GamePhase::kDeal;

    float logit_cap = static_cast<float>(absl::GetFlag(FLAGS_logit_cap));
    double shaped_weight = absl::GetFlag(FLAGS_shaped_reward_weight);
    double tleilaxu_breadcrumb_weight =
        absl::GetFlag(FLAGS_tleilaxu_breadcrumb_weight);
    double tleilaxu_level7_breadcrumb_weight =
        absl::GetFlag(FLAGS_tleilaxu_level7_breadcrumb_weight);
    double specimen_exchange_penalty =
        absl::GetFlag(FLAGS_specimen_exchange_penalty);
    // PWO-5 section 14.1b. Hoisted per episode alongside the other flag reads,
    // so the canary measurement costs one bool test per decision when off.
    const bool emit_canary_columns =
        (local_stats != nullptr) && absl::GetFlag(FLAGS_emit_canary_columns);
    // Carried across the CenterAndCapLegalLogits call: the PRE-cap half (SAT,
    // and the role) is measured before it, the POST-cap half (NE,
    // max_action_prob) after it, and both halves must describe the SAME
    // decision.
    bool canary_measure_this_decision = false;
    int canary_role_this_decision = 0;

    torch::NoGradGuard no_grad;
    int game_length = 0;
    int counterfactual_samples = 0;
    std::vector<CounterfactualPending> pending_cf;
    while (!state->IsTerminal()) {
      ++game_length;
      if (game_length > 5000) {
        std::cerr << "Possible infinite loop detected. State:\n"
                  << state->ToString() << "\n";
        std::abort();
      }

      if (state->IsChanceNode()) {
        std::vector<std::pair<Action, double>> outcomes = state->ChanceOutcomes();
        Action action = game.GetType().chance_mode == GameType::ChanceMode::kSampledStochastic
                    ? outcomes.front().first
                    : SampleAction(outcomes, chance_rng).first;
        state->ApplyAction(action);
        continue;
      }

      if (state->CurrentPlayer() == kSimultaneousPlayerId) {
        std::vector<Action> joint_action;
        for (int p = 0; p < game.NumPlayers(); ++p) {
          std::vector<Action> actions = state->LegalActions(p);
          if (actions.empty()) {
            joint_action.push_back(0);
          } else {
            std::uniform_int_distribution<int> dis(0, actions.size() - 1);
            joint_action.push_back(actions[dis(policy_rng[p])]);
          }
        }
        state->ApplyActions(joint_action);
        continue;
      }

      if (absl::GetFlag(FLAGS_train_value_only)) {
        if (!prohibited_histories.empty() && prohibited_histories.count(state->History())) {
          SpielFatalError("CRITICAL PROTECTION BOUNDARY FAILURE: Attempted to process or train on a prohibited official-corpus state!");
        }
      }

      Player current_player = state->CurrentPlayer();
      std::vector<Action> actions = state->LegalActions();
      if (actions.empty()) {
        std::cerr << "Non-terminal state has no legal actions. State:\n"
                  << state->ToString() << "\n";
        std::abort();
      }

      std::vector<float> obs(obs_size, 0.0f);
      if (provides_info_state_tensor && current_player >= 0) {
        state->InformationStateTensor(current_player, absl::MakeSpan(obs));
      } else if (provides_observations_tensor && current_player >= 0) {
        state->ObservationTensor(current_player, absl::MakeSpan(obs));
      }

      EvalResult result = evaluator->Evaluate(obs);
      std::vector<float> logits = std::move(result.logits);

      // --- Phase A canary: pre-cap legal-centered |z| for this decision ---
      // Measured HERE because CenterAndCapLegalLogits below mutates `logits` in
      // place, destroying the raw values. The centering is a deliberate mirror
      // of that function's: double accumulator, mean cast to float, same
      // in-range guard -- so `z` is exactly what it would subtract, taken
      // BEFORE its tanh. Restricted to nontrivial decisions (>= 2 legal
      // actions) to match the trainer's own nontrivial definition; a forced
      // decision's centered logit is identically 0 and would only dilute the
      // percentiles. Observational only: no RNG draw, no tensor op, nothing
      // read back into the acting path, so the sampled action is unaffected.
      if (local_stats != nullptr && actions.size() >= 2) {
        double legal_sum = 0.0;
        int legal_count = 0;
        for (Action a : actions) {
          if (a >= 0 && static_cast<size_t>(a) < logits.size()) {
            legal_sum += logits[a];
            ++legal_count;
          }
        }
        if (legal_count > 0) {
          const float legal_mean = static_cast<float>(legal_sum / legal_count);
          float max_absz = 0.0f;
          for (Action a : actions) {
            if (a < 0 || static_cast<size_t>(a) >= logits.size()) continue;
            max_absz = std::max(max_absz, std::abs(logits[a] - legal_mean));
          }
          local_stats->precap_absz_all.push_back(max_absz);
          const DuneDecisionRole decision_role = ClassifyDuneDecisionRole(
              *state, current_player, /*has_active_session=*/false);
          switch (decision_role) {
            case DuneDecisionRole::kAgentPrimary:
              local_stats->precap_absz_primary.push_back(max_absz);
              break;
            case DuneDecisionRole::kAgentContinuation:
              local_stats->precap_absz_continuation.push_back(max_absz);
              break;
            case DuneDecisionRole::kPurchase:
              local_stats->precap_absz_purchase.push_back(max_absz);
              break;
            default:
              break;
          }

          // --- PWO-5 section 14.1b: SAT, the PRE-cap half. ----------------
          //
          // `SAT` is PER LOGIT, over the decision's legal actions, against the
          // CONFIGURED cap -- three properties that are each load-bearing and
          // none of which the existing `frac_decisions_absz_ge10` has:
          // that column is per DECISION (max|z| over the row), and hard-codes
          // 10.0 independently of --logit_cap. The two are reported side by
          // side and are never conflated.
          //
          // It must be PRE-cap. The cap is `c * tanh(z / c)`, not a clamp, so
          // in exact arithmetic |y| < c strictly for every finite z and a
          // post-cap definition would be IDENTICALLY ZERO -- a rail that can
          // never fire. In finite precision it is worse than zero: tanh rounds
          // to exactly 1.0 at |z| >= 90.1 in fp32 but at |z| >= 31.2 in bf16,
          // so the rail's sensitivity would become a function of the autocast
          // dtype. Captured here, before CenterAndCapLegalLogits mutates
          // `logits` in place.
          if (emit_canary_columns) {
            for (Action a : actions) {
              if (a < 0 || static_cast<size_t>(a) >= logits.size()) continue;
              ++local_stats->canary_sat_logits;
              if (std::abs(logits[a] - legal_mean) >= logit_cap) {
                ++local_stats->canary_sat_ge_cap;
              }
            }
            canary_role_this_decision = static_cast<int>(decision_role);
            canary_measure_this_decision = true;
          }
        }
      }

      CenterAndCapLegalLogits(logits, actions, logit_cap);

      // --- PWO-5 section 14.1b: NE and max_action_prob, the POST-cap half. --
      //
      // Taken from the distribution the behaviour policy ACTUALLY SAMPLES FROM
      // (SamplePolicyAction, immediately below), which is the legal softmax of
      // the capped logits. Post-cap is what section 14.2's positive-entropy-
      // floor argument assumes: because the cap is applied BEFORE the softmax,
      // the softmax argument range is bounded by (-c, +c), so post-cap entropy
      // has a strictly positive floor for every n_legal >= 2 and can never
      // reach 0 -- which is why an absolute entropy threshold is unanchorable
      // and the rails are decline-based rather than absolute.
      //
      // The normalizer is log(n_legal) -- not log(vocabulary_size) and not
      // log(len(visits)). n_legal >= 2 here by construction, so log(n_legal)
      // > 0 always and the repository's two incompatible conventions for the
      // n_legal == 1 case are both moot.
      if (canary_measure_this_decision) {
        float max_logit = -std::numeric_limits<float>::infinity();
        for (Action a : actions) {
          if (a >= 0 && static_cast<size_t>(a) < logits.size()) {
            max_logit = std::max(max_logit, logits[a]);
          }
        }
        double total = 0.0;
        std::vector<double> probs;
        probs.reserve(actions.size());
        for (Action a : actions) {
          double w = 0.0;
          if (a >= 0 && static_cast<size_t>(a) < logits.size() &&
              std::isfinite(max_logit)) {
            w = std::exp(static_cast<double>(logits[a]) -
                         static_cast<double>(max_logit));
          }
          probs.push_back(w);
          total += w;
        }
        if (total > 0.0) {
          double h = 0.0;
          double max_p = 0.0;
          for (double& w : probs) {
            w /= total;
            if (w > 0.0) h -= w * std::log(w);
            max_p = std::max(max_p, w);
          }
          const double ne = h / std::log(static_cast<double>(actions.size()));
          local_stats->canary_ne_sum += ne;
          ++local_stats->canary_ne_n;
          local_stats->canary_ne_sum_role[canary_role_this_decision] += ne;
          ++local_stats->canary_ne_n_role[canary_role_this_decision];
          local_stats->canary_maxprob_sum += max_p;
        }
        canary_measure_this_decision = false;
      }

      auto policy_sample = SamplePolicyAction(&policy_rng[current_player], logits, actions);
      Action action = policy_sample.first;
      float old_log_prob = policy_sample.second;

      if (absl::GetFlag(FLAGS_sample_counterfactual_states) &&
          counterfactual_samples <
              absl::GetFlag(FLAGS_counterfactual_samples_per_game) &&
          ClassifyDuneDecisionRole(*state, current_player, false) == DuneDecisionRole::kAgentPrimary) {
        // Clone independent RNGs to prevent perturbing the live trajectory
        std::mt19937_64 cf_chance_rng = chance_rng;
        std::mt19937_64 cf_policy_rng[4];
        for (int p = 0; p < 4; ++p) {
          cf_policy_rng[p] = policy_rng[p];
        }

        std::vector<Action> alternate_actions;
        alternate_actions.reserve(actions.size());
        for (Action candidate : actions) {
          if (candidate != action) alternate_actions.push_back(candidate);
        }
        if (!alternate_actions.empty()) {
          // One uniformly sampled alternative per encountered primary state.
          // Episode/player RNG streams make this deterministic while the full
          // raw-policy game set provides coverage across all legal actions.
          std::uniform_int_distribution<size_t> alternate_distribution(
              0, alternate_actions.size() - 1);
          Action alt_act = alternate_actions[
              alternate_distribution(cf_policy_rng[current_player])];

          auto succ_state = state->Clone();
          succ_state->ApplyAction(alt_act);

          while (succ_state->IsChanceNode()) {
            auto outcomes = succ_state->ChanceOutcomes();
            if (outcomes.empty()) break;
            Action chance_act = game.GetType().chance_mode == GameType::ChanceMode::kSampledStochastic
                        ? outcomes.front().first
                        : SampleAction(outcomes, cf_chance_rng).first;
            succ_state->ApplyAction(chance_act);
          }

          std::vector<float> succ_obs(obs_size, 0.0f);
          bool valid = false;
          float succ_val = 0.0f;
          float target_return = 0.0f;
          if (!succ_state->IsTerminal()) {
            if (provides_info_state_tensor) {
              succ_state->InformationStateTensor(current_player, absl::MakeSpan(succ_obs));
              valid = true;
            } else if (provides_observations_tensor) {
              succ_state->ObservationTensor(current_player, absl::MakeSpan(succ_obs));
              valid = true;
            }
            if (valid) {
              EvalResult succ_result = evaluator->Evaluate(succ_obs);
              succ_val = succ_result.value;

              // Run a counterfactual rollout to get a terminal-return learning signal
              auto roll_state = succ_state->Clone();
              while (!roll_state->IsTerminal()) {
                if (roll_state->IsChanceNode()) {
                  auto outcomes = roll_state->ChanceOutcomes();
                  if (outcomes.empty()) break;
                  Action choice = game.GetType().chance_mode == GameType::ChanceMode::kSampledStochastic
                              ? outcomes.front().first
                              : SampleAction(outcomes, cf_chance_rng).first;
                  roll_state->ApplyAction(choice);
                  continue;
                }

                if (roll_state->CurrentPlayer() == kSimultaneousPlayerId) {
                  std::vector<Action> joint_action;
                  for (int p = 0; p < game.NumPlayers(); ++p) {
                    std::vector<Action> actions = roll_state->LegalActions(p);
                    if (actions.empty()) {
                      joint_action.push_back(0);
                    } else {
                      std::uniform_int_distribution<int> dis(0, actions.size() - 1);
                      joint_action.push_back(actions[dis(cf_policy_rng[p])]);
                    }
                  }
                  roll_state->ApplyActions(joint_action);
                  continue;
                }

                Player curr_p = roll_state->CurrentPlayer();
                std::vector<Action> curr_acts = roll_state->LegalActions();
                std::vector<float> curr_obs(obs_size, 0.0f);
                if (provides_info_state_tensor) {
                  roll_state->InformationStateTensor(curr_p, absl::MakeSpan(curr_obs));
                } else {
                  roll_state->ObservationTensor(curr_p, absl::MakeSpan(curr_obs));
                }
                EvalResult roll_res = evaluator->Evaluate(curr_obs);
                CenterAndCapLegalLogits(roll_res.logits, curr_acts, logit_cap);
                Action roll_act = SamplePolicyAction(&cf_policy_rng[curr_p], roll_res.logits, curr_acts).first;
                roll_state->ApplyAction(roll_act);
              }
              target_return = static_cast<float>(roll_state->Returns()[current_player]) / 4.0f;
            }
          } else {
            succ_val = static_cast<float>(succ_state->Returns()[current_player]) / 4.0f;
            target_return = succ_val;
            valid = true;
          }

          if (valid) {
            if (absl::GetFlag(FLAGS_train_value_only)) {
              if (!prohibited_histories.empty() && prohibited_histories.count(succ_state->History())) {
                SpielFatalError("CRITICAL PROTECTION BOUNDARY FAILURE: Attempted to process or train on a prohibited official-corpus state via counterfactual generation!");
              }
            }
            PpoTransition cf_trans;
            cf_trans.state = std::move(succ_obs);
            cf_trans.legal_actions = succ_state->IsTerminal() ? std::vector<Action>{} : succ_state->LegalActions();
            cf_trans.action = kInvalidAction;
            cf_trans.old_log_prob = 0.0f;
            cf_trans.reward = 0.0f;
            cf_trans.value = succ_val;
            cf_trans.advantage = target_return - succ_val;
            cf_trans.return_value = target_return;
            cf_trans.player_id = current_player;
            cf_trans.episode_id = episode_id;

            CounterfactualPending pending;
            pending.replay_weight =
                absl::GetFlag(FLAGS_counterfactual_replay_weight);
            pending.transition = std::move(cf_trans);
            pending_cf.push_back(std::move(pending));
            ++counterfactual_samples;
          }
        }
      }

      if (dune_state != nullptr) {
        pre_action_phase = dune_state->phase();
        if (current_player >= 0 && current_player < game.NumPlayers()) {
          pre_combat_strength[current_player] =
              dune_imperium::CombatStrength(*dune_state, current_player);
        }
      }

      state->ApplyAction(action);
      total_env_steps->fetch_add(1, std::memory_order_relaxed);

      if (dune_state != nullptr) {
        for (int p = 0; p < game.NumPlayers(); ++p) {
          if (!had_swordmaster[p] && dune_state->HasSwordmaster(p)) {
            had_swordmaster[p] = true;
            int purchase_round = dune_state->GetCurrentRound();
            int leader = dune_state->PlayerLeader(p);
            local_stats->sm_acquisitions_by_seat[p]++;
            if (leader >= 0 && leader < 14) {
              local_stats->sm_acquisitions_by_leader[leader]++;
            }
            if (purchase_round >= 1 && purchase_round <= 10) {
              local_stats->sm_acquisitions_by_round[purchase_round]++;
            }
          }
        }
      }

      PpoTransition transition;
      transition.state = std::move(obs);
      transition.legal_actions = std::move(actions);
      transition.action = action;
      transition.old_log_prob = old_log_prob;
      transition.reward = 0.0f;
      transition.value = result.value;
      transition.advantage = 0.0f;
      transition.return_value = 0.0f;
      transition.player_id = current_player;
      transition.episode_id = episode_id;
      trajectory->push_back(std::move(transition));

      if (current_player >= 0 && current_player < game.NumPlayers()) {
        last_transition_index[current_player] = static_cast<int>(trajectory->size() - 1);
      }

      if (dune_state != nullptr) {
        if (current_player >= 0 && current_player < game.NumPlayers() &&
            action != dune_imperium::kActionCombatPass) {
          int post_strength = dune_imperium::CombatStrength(*dune_state, current_player);
          int strength_delta = post_strength - pre_combat_strength[current_player];
          if (strength_delta > 0) {
            combat_accumulator.RecordDeployment(current_player, trajectory->size() - 1, strength_delta);
          }
        }

        // Specimen-exchange anti-breadcrumb (Item 3): a small negative shaping on
        // the transition that takes a ConvertSpecimenToTroop action (IDs 741-752
        // -- 740 is an unused base constant and is never legal; see
        // dune_specimen_conversion.h), to discourage the over-used
        // specimen->troop breadcrumb behavior. Training-only (never eval). Must
        // be set to the SAME value on both the pilot and control arms so the
        // search-distillation contrast stays pure.
        //
        // PWO-5 gate 2 items (b)+(c): the range predicate and the subtraction
        // both moved into shared, unit-tested helpers. The sign is the point --
        // a POSITIVE penalty must DECREASE this reward.
        if (specimen_exchange_penalty != 0.0 &&
            current_player >= 0 && current_player < game.NumPlayers() &&
            dune_shaping::IsSpecimenConversionAction(action)) {
          int idx = last_transition_index[current_player];
          if (idx >= 0 && idx < static_cast<int>(trajectory->size())) {
            (*trajectory)[idx].reward = ApplySpecimenExchangeShaping(
                (*trajectory)[idx].reward, action, specimen_exchange_penalty,
                reward_lambda);
          }
        }

        for (int p = 0; p < game.NumPlayers(); ++p) {
          int new_tleilaxu = dune_state->GetTleilaxuTrackForTesting(p);
          int tleilaxu_delta = new_tleilaxu - current_tleilaxu[p];
          current_tleilaxu[p] = new_tleilaxu;
          if (tleilaxu_delta > 0 &&
              (tleilaxu_breadcrumb_weight > 0.0 ||
               tleilaxu_level7_breadcrumb_weight > 0.0)) {
            float tleilaxu_reward = 0.0f;
            int start_l = new_tleilaxu - tleilaxu_delta;
            for (int l = start_l + 1; l <= new_tleilaxu; ++l) {
              if (l >= 1 && l <= 6) {
                tleilaxu_reward += static_cast<float>(tleilaxu_breadcrumb_weight) * reward_lambda;
              } else if (l == 7) {
                tleilaxu_reward += static_cast<float>(tleilaxu_level7_breadcrumb_weight) * reward_lambda;
              }
            }
            int idx = last_transition_index[p];
            if (idx >= 0 && idx < static_cast<int>(trajectory->size())) {
              (*trajectory)[idx].reward += tleilaxu_reward;
            }
          }
        }

        for (int p = 0; p < game.NumPlayers(); ++p) {
          int raw_conflict_vp_delta = dune_state->ConflictVpDelta(p) - prev_conflict_vp[p];
          prev_conflict_vp[p] = dune_state->ConflictVpDelta(p);

          int raw_total_vp_delta = dune_state->GetPlayerVp(p) - prev_total_vp[p];
          prev_total_vp[p] = dune_state->GetPlayerVp(p);

          int raw_noncombat = raw_total_vp_delta - raw_conflict_vp_delta;

          local_stats->raw_conflict_vp += raw_conflict_vp_delta;
          local_stats->raw_noncombat_vp += raw_noncombat;
          local_stats->raw_total_vp += raw_total_vp_delta;

          float combat_shape = std::max(raw_conflict_vp_delta, 0)
                                * static_cast<float>(shaped_weight) * reward_lambda;
          float noncombat_shape = raw_noncombat
                                  * static_cast<float>(shaped_weight) * reward_lambda;

          if (noncombat_shape != 0.0f) {
            int idx = last_transition_index[p];
            if (idx >= 0 && idx < static_cast<int>(trajectory->size())) {
              (*trajectory)[idx].reward += noncombat_shape;
            }
          }

          if (combat_shape > 0.0f) {
            local_stats->conflict_vp_generated += combat_shape;
            int total_investment = combat_accumulator.GetTotalInvestment(p);
            if (total_investment > 0) {
              for (const auto& ev : combat_accumulator.GetEvents(p)) {
                if (ev.transition_index >= 0 && ev.transition_index < static_cast<int>(trajectory->size())) {
                  float attributed_portion = combat_shape * static_cast<float>(ev.strength_delta) /
                                             static_cast<float>(total_investment);
                  (*trajectory)[ev.transition_index].reward += attributed_portion;
                  local_stats->conflict_vp_attributed += attributed_portion;
                }
              }
            } else {
              local_stats->conflict_vp_unattributed += combat_shape;
            }
          }
        }

        if (pre_action_phase == dune_imperium::GamePhase::kCombat &&
            dune_state->phase() != dune_imperium::GamePhase::kCombat) {
          combat_accumulator.ClearAll();
        }
      }
    }

    std::vector<double> terminal_returns = state->Returns();
    double gamma = absl::GetFlag(FLAGS_train_value_only) ? 1.0 : absl::GetFlag(FLAGS_gamma);
    double gae_lambda = absl::GetFlag(FLAGS_train_value_only) ? 1.0 : absl::GetFlag(FLAGS_gae_lambda);
    float reward_scale = static_cast<float>(
        std::max(1e-6, absl::GetFlag(FLAGS_reward_scale)));
    std::vector<float> last_value(game.NumPlayers(), 0.0f);
    std::vector<float> last_gae(game.NumPlayers(), 0.0f);
    std::vector<bool> seen_last_action(game.NumPlayers(), false);

    for (auto it = trajectory->rbegin(); it != trajectory->rend(); ++it) {
      if (it->player_id < 0 || it->player_id >= game.NumPlayers()) continue;
      int p = it->player_id;
      float reward = 0.0f;
      if (absl::GetFlag(FLAGS_train_value_only)) {
        if (!seen_last_action[p]) {
          reward = static_cast<float>(terminal_returns[p]);
          seen_last_action[p] = true;
        }
      } else {
        reward = it->reward;
        if (!seen_last_action[p]) {
          reward += static_cast<float>(terminal_returns[p]);
          seen_last_action[p] = true;
        }
      }
      reward = std::clamp(reward / reward_scale, -1.0f, 1.0f);

      float delta = reward + static_cast<float>(gamma) * last_value[p] -
                    it->value;
      float advantage =
          delta + static_cast<float>(gamma * gae_lambda) * last_gae[p];
      it->advantage = advantage;
      it->return_value = advantage + it->value;
      last_value[p] = it->value;
      last_gae[p] = advantage;
    }

    for (auto& cf : pending_cf) {
      for (int repeat = 0; repeat < cf.replay_weight; ++repeat) {
        if (repeat + 1 == cf.replay_weight) {
          trajectory->push_back(std::move(cf.transition));
        } else {
          trajectory->push_back(cf.transition);
        }
      }
    }

    return game_length;
  }
}

void RolloutWorker(int thread_id, const Game* game,
                    std::shared_ptr<IGameEvaluator> evaluator, int64_t obs_size,
                    PpoRolloutBuffer* rollout_buffer,
                    std::atomic<bool>* stop_collection,
                    std::atomic<uint64_t>* total_env_steps,
                    std::vector<WorkerStats>* worker_stats,
                    std::atomic<uint64_t>* local_episode_id,
                    uint64_t start_episode_id,
                    int rollout_games,
                    float reward_lambda) {
  uint64_t master = absl::GetFlag(FLAGS_seed);
  WorkerStats local_stats;
  while (true) {
    if (rollout_games == 0 && stop_collection->load(std::memory_order_relaxed)) {
      break;
    }
    uint64_t episode_id = local_episode_id->fetch_add(1, std::memory_order_relaxed);
    if (rollout_games > 0 && episode_id >= start_episode_id + rollout_games) {
      break;
    }
    std::vector<PpoTransition> trajectory;
    int moves = PpoSimulation(master, episode_id, *game, evaluator, obs_size, &trajectory,
                              total_env_steps, reward_lambda, &local_stats);
    local_stats.games += 1;
    local_stats.moves += moves;
    size_t size = rollout_buffer->PushTrajectory(std::move(trajectory));
    if (rollout_games == 0) {
      if (size >= static_cast<size_t>(absl::GetFlag(FLAGS_rollout_transitions))) {
        stop_collection->store(true, std::memory_order_relaxed);
      }
    }
  }
  (*worker_stats)[thread_id] = local_stats;
}

struct CollectResult {
  std::vector<PpoTransition> rollout;
  uint64_t games = 0;
  uint64_t moves = 0;
  double elapsed_seconds = 0.0;

  uint64_t sm_acquisitions_by_seat[4] = {0};
  uint64_t sm_acquisitions_by_leader[14] = {0};
  uint64_t sm_acquisitions_by_round[11] = {0};

  double conflict_vp_generated = 0.0;
  double conflict_vp_attributed = 0.0;
  double conflict_vp_unattributed = 0.0;

  bool episode_ids_unique = true;
  double raw_conflict_vp = 0.0;
  double raw_noncombat_vp = 0.0;
  double raw_total_vp = 0.0;

  // Phase A canary: concatenation of every worker's per-decision max|z|, merged
  // in thread-index order. Only the multiset matters downstream (percentiles
  // sort), so the merge order does not affect the reported statistics.
  std::vector<float> precap_absz_all;
  std::vector<float> precap_absz_primary;
  std::vector<float> precap_absz_continuation;
  std::vector<float> precap_absz_purchase;

  // PWO-5 section 14.1b canaries, summed over workers. Summation is exact for
  // the integer counts; the double sums are order-dependent at the last ulp,
  // and the merge is in thread-index order, which is deterministic given a
  // thread count. That is the same guarantee every other double accumulator in
  // this struct carries, and it is why every bitwise gate runs at --threads=1.
  double canary_ne_sum = 0.0;
  int64_t canary_ne_n = 0;
  double canary_ne_sum_role[7] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  int64_t canary_ne_n_role[7] = {0, 0, 0, 0, 0, 0, 0};
  double canary_maxprob_sum = 0.0;
  int64_t canary_sat_ge_cap = 0;
  int64_t canary_sat_logits = 0;

  // Phase 18B: online auxiliary-search examples collected from the SAME frozen
  // snapshot as `rollout`, so they travel together and are consumed by the same
  // TrainPpoUpdate. Empty unless --online_search_collection.
  std::vector<open_spiel::SearchTrainingExample> aux_examples;
  open_spiel::OnlineSearchCollectionStats aux_stats;
};

// Phase A: folds the rollout-side pre-cap |z| canaries into the update's stats
// so every WriteDiagnostics site persists them. Called immediately after
// episode_ids_unique is copied, which is the existing seam for carrying
// collection-side facts into the diagnostics row.
// PWO-5 section 14.1c: fold the rollout-side canaries into the update's stats,
// and CHECK the partition identity rather than assuming it.
//
// Two halves, both registered:
//   Sum_r n_r == norm_entropy_n            EXACTLY (integer equality) -- the
//     stronger half; it proves no qualifying decision was dropped or
//     double-counted.
//   |Sum_r (n_r * NE_r) / Sum_r n_r - NE_global| <= 1e-9  -- float slack only.
//
// A violation is a gate-3 failure, not a rounding artifact, so it is fatal here
// rather than reported.
void AttachCanaryStats(open_spiel::PpoUpdateStats* stats,
                       const CollectResult& collect) {
  stats->norm_entropy_n = collect.canary_ne_n;
  stats->norm_entropy =
      collect.canary_ne_n > 0 ? collect.canary_ne_sum / collect.canary_ne_n : 0.0;
  stats->max_action_prob =
      collect.canary_ne_n > 0 ? collect.canary_maxprob_sum / collect.canary_ne_n
                              : 0.0;
  stats->frac_legal_absz_ge_cap_n = collect.canary_sat_logits;
  stats->frac_legal_absz_ge_cap =
      collect.canary_sat_logits > 0
          ? static_cast<double>(collect.canary_sat_ge_cap) /
                static_cast<double>(collect.canary_sat_logits)
          : 0.0;

  int64_t role_n_total = 0;
  double weighted_sum = 0.0;
  for (int r = 0; r < 7; ++r) {
    stats->norm_entropy_n_role[r] = collect.canary_ne_n_role[r];
    // Zero-support semantics: mean is 0.0 when the count is 0, and 0.0 there
    // means "no support this update", never "entropy collapsed to zero".
    stats->norm_entropy_role[r] =
        collect.canary_ne_n_role[r] > 0
            ? collect.canary_ne_sum_role[r] / collect.canary_ne_n_role[r]
            : 0.0;
    role_n_total += collect.canary_ne_n_role[r];
    weighted_sum += collect.canary_ne_sum_role[r];
  }

  if (role_n_total != collect.canary_ne_n) {
    SpielFatalError(absl::StrFormat(
        "PWO-5 section 14.1c partition identity FAILED (count half): "
        "sum of role denominators = %lld but norm_entropy_n = %lld. A "
        "qualifying decision was dropped or double-counted.",
        static_cast<long long>(role_n_total),
        static_cast<long long>(collect.canary_ne_n)));
  }
  if (role_n_total > 0) {
    const double reconstructed = weighted_sum / role_n_total;
    if (std::abs(reconstructed - stats->norm_entropy) > 1e-9) {
      SpielFatalError(absl::StrFormat(
          "PWO-5 section 14.1c partition identity FAILED (mean half): "
          "row-count-weighted role mean %.17g vs norm_entropy %.17g, "
          "difference %.3g exceeds 1e-9.",
          reconstructed, stats->norm_entropy,
          std::abs(reconstructed - stats->norm_entropy)));
    }
  }
  // The seventh role must be identically empty at this site (section 14.1c's
  // proof). Checked, because the proof depends on the measurement site and a
  // future move of that site would silently invalidate it.
  if (collect.canary_ne_n_role[static_cast<int>(
          DuneDecisionRole::kForcedOrBookkeeping)] != 0) {
    SpielFatalError(
        "PWO-5 section 14.1c: kForcedOrBookkeeping carried a canary decision. "
        "That role is provably empty at the registered rollout measurement "
        "site (chance nodes continue before classification, and the population "
        "requires n_legal >= 2), so a nonzero count means the canaries are no "
        "longer being measured where section 14.1b froze them.");
  }
}

void AttachPrecapAbszStats(open_spiel::PpoUpdateStats* stats,
                           const CollectResult& collect) {
  stats->precap_absz_all =
      open_spiel::SummarizePrecapAbsz(collect.precap_absz_all);
  stats->precap_absz_primary =
      open_spiel::SummarizePrecapAbsz(collect.precap_absz_primary);
  stats->precap_absz_continuation =
      open_spiel::SummarizePrecapAbsz(collect.precap_absz_continuation);
  stats->precap_absz_purchase =
      open_spiel::SummarizePrecapAbsz(collect.precap_absz_purchase);
}



CollectResult CollectRollout(const Game* game,
                             std::shared_ptr<IGameEvaluator> evaluator,
                             int64_t obs_size,
                             std::atomic<uint64_t>* total_env_steps,
                             int num_threads,
                             std::atomic<uint64_t>* next_episode_id,
                             int rollout_games,
                             float reward_lambda) {
  CollectResult result;
  PpoRolloutBuffer rollout_buffer;
  std::atomic<bool> stop_collection{false};
  std::vector<WorkerStats> worker_stats(num_threads);
  std::vector<std::thread> workers;
  workers.reserve(num_threads);

  uint64_t actual_start_ep = next_episode_id->load();
  std::atomic<uint64_t> local_episode_id{actual_start_ep};

  auto start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < num_threads; ++i) {
    workers.emplace_back(RolloutWorker, i, game, evaluator, obs_size,
                         &rollout_buffer, &stop_collection, total_env_steps,
                         &worker_stats, &local_episode_id, actual_start_ep, rollout_games, reward_lambda);
  }
  for (auto& worker : workers) worker.join();

  if (rollout_games > 0) {
    next_episode_id->store(actual_start_ep + rollout_games);
  } else {
    next_episode_id->store(local_episode_id.load());
  }
  auto end = std::chrono::high_resolution_clock::now();

  bool episode_ids_unique = true;
  result.rollout = rollout_buffer.TakeAll(&episode_ids_unique);
  result.episode_ids_unique = episode_ids_unique;

  for (const auto& stats : worker_stats) {
    result.games += stats.games;
    result.moves += stats.moves;

    for (int i = 0; i < 4; ++i) result.sm_acquisitions_by_seat[i] += stats.sm_acquisitions_by_seat[i];
    for (int i = 0; i < 14; ++i) result.sm_acquisitions_by_leader[i] += stats.sm_acquisitions_by_leader[i];
    for (int i = 0; i < 11; ++i) result.sm_acquisitions_by_round[i] += stats.sm_acquisitions_by_round[i];

    result.conflict_vp_generated += stats.conflict_vp_generated;
    result.conflict_vp_attributed += stats.conflict_vp_attributed;
    result.conflict_vp_unattributed += stats.conflict_vp_unattributed;

    result.raw_conflict_vp += stats.raw_conflict_vp;
    result.raw_noncombat_vp += stats.raw_noncombat_vp;
    result.raw_total_vp += stats.raw_total_vp;

    // Phase A canary: concatenate in thread-index order. Downstream only sorts,
    // so order is irrelevant to the reported percentiles.
    result.precap_absz_all.insert(result.precap_absz_all.end(),
                                  stats.precap_absz_all.begin(),
                                  stats.precap_absz_all.end());
    result.precap_absz_primary.insert(result.precap_absz_primary.end(),
                                      stats.precap_absz_primary.begin(),
                                      stats.precap_absz_primary.end());
    result.precap_absz_continuation.insert(result.precap_absz_continuation.end(),
                                           stats.precap_absz_continuation.begin(),
                                           stats.precap_absz_continuation.end());
    result.precap_absz_purchase.insert(result.precap_absz_purchase.end(),
                                       stats.precap_absz_purchase.begin(),
                                       stats.precap_absz_purchase.end());

    // PWO-5 section 14.1b canaries.
    result.canary_ne_sum += stats.canary_ne_sum;
    result.canary_ne_n += stats.canary_ne_n;
    for (int r = 0; r < 7; ++r) {
      result.canary_ne_sum_role[r] += stats.canary_ne_sum_role[r];
      result.canary_ne_n_role[r] += stats.canary_ne_n_role[r];
    }
    result.canary_maxprob_sum += stats.canary_maxprob_sum;
    result.canary_sat_ge_cap += stats.canary_sat_ge_cap;
    result.canary_sat_logits += stats.canary_sat_logits;
  }

  // Verify shaped reward conservation invariant
  if (std::abs(result.conflict_vp_generated - (result.conflict_vp_attributed + result.conflict_vp_unattributed)) > 1e-4) {
    open_spiel::SpielFatalError(absl::StrFormat("Shaped reward conservation violated: generated = %f, attributed = %f, unattributed = %f",
                                                result.conflict_vp_generated, result.conflict_vp_attributed, result.conflict_vp_unattributed));
  }

  // Verify signed raw VP conservation invariant
  if (std::abs(result.raw_conflict_vp + result.raw_noncombat_vp - result.raw_total_vp) > 1e-4) {
    open_spiel::SpielFatalError(absl::StrFormat("Signed raw VP conservation violated: raw_conflict = %f, raw_noncombat = %f, raw_total = %f",
                                                result.raw_conflict_vp, result.raw_noncombat_vp, result.raw_total_vp));
  }
  result.elapsed_seconds =
      std::chrono::duration<double>(end - start).count();
  return result;
}


#endif  // OPEN_SPIEL_BUILD_WITH_LIBTORCH

}  // namespace
}  // namespace open_spiel

int main(int argc, char** argv) {
  setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8", 1);
  absl::ParseCommandLine(argc, argv);
  // PWO-5 section 10.2: the reserved final-gate base-seed range, enforced at
  // the launcher so the exclusion is mechanical rather than a matter of
  // operator care. Checked before ANY other work.
  {
    const std::string stop = dune_seed::ReservedFinalGateSeedStop(
        "--seed", static_cast<long long>(absl::GetFlag(FLAGS_seed)));
    if (!stop.empty()) {
      std::cerr << stop << "\n";
      return 1;
    }
  }
  // PWO-5 gate 2 item (a): reject a NEGATIVE --specimen_exchange_penalty
  // fatally, after argument parsing and before anything else runs.
  //
  // The flag is a magnitude that is SUBTRACTED from the reward of a conversion
  // transition, so a negative value is a BONUS on exactly the breadcrumb
  // behaviour the term exists to suppress. This is not a hypothetical foot-gun:
  // the u175 lineage was trained with `--specimen_exchange_penalty=-0.02` and
  // the committed calibration_results_v2/pilot300_search_seed12/launch.sh still
  // carries it. Until now the ONLY validation of this flag was the
  // --allow_shaping gate below, which tests `!= 0.0` and accepts negatives.
  if (absl::GetFlag(FLAGS_specimen_exchange_penalty) < 0.0) {
    std::cerr << "Fatal: --specimen_exchange_penalty="
              << absl::GetFlag(FLAGS_specimen_exchange_penalty)
              << " is NEGATIVE. This flag is a MAGNITUDE THAT IS SUBTRACTED "
                 "from the reward of a ConvertSpecimenToTroop transition, so a "
                 "negative value is a BONUS on the breadcrumb behaviour the "
                 "term exists to penalize (this is the u175-lineage defect). "
                 "Pass a value >= 0; 0.0 disables the term."
              << std::endl;
    return -1;
  }
  if (absl::GetFlag(FLAGS_shaped_reward_weight) != 0.0 ||
      absl::GetFlag(FLAGS_tleilaxu_breadcrumb_weight) != 0.0 ||
      absl::GetFlag(FLAGS_tleilaxu_level7_breadcrumb_weight) != 0.0 ||
      absl::GetFlag(FLAGS_specimen_exchange_penalty) != 0.0) {
    if (!absl::GetFlag(FLAGS_allow_shaping)) {
      std::cerr << "Fatal: Reward shaping weights are non-zero but --allow_shaping is not set. "
                << "To run with reward shaping, pass --allow_shaping=true." << std::endl;
      return -1;
    } else {
      std::cout << "Warning: Running with non-zero experimental reward shaping weights!" << std::endl;
    }
  }
  if (absl::GetFlag(FLAGS_deterministic)) {
    at::globalContext().setDeterministicAlgorithms(true, /*silent=*/true);
  }
  using namespace open_spiel;

#ifndef OPEN_SPIEL_BUILD_WITH_LIBTORCH
  std::cerr << "dune_ppo_train requires OPEN_SPIEL_BUILD_WITH_LIBTORCH.\n";
  return 1;
#else
  at::set_num_threads(1);
  at::set_num_interop_threads(1);

  uint64_t master = static_cast<uint64_t>(absl::GetFlag(FLAGS_seed));
  torch::manual_seed(dune_seed::DeriveSeed(master, dune_seed::kStreamModelInit));

  auto game = open_spiel::LoadGame(absl::GetFlag(FLAGS_game));
  int64_t obs_size = game->GetType().provides_information_state_tensor
                         ? game->InformationStateTensorSize()
                         : game->ObservationTensorSize();
  int64_t action_size = game->NumDistinctActions();

  torch::Device device(torch::kCPU);
  if (torch::cuda::is_available()) {
    device = torch::Device(torch::kCUDA);
    at::globalContext().setAllowTF32CuBLAS(true);
    at::globalContext().setAllowTF32CuDNN(true);
    at::autocast::set_autocast_dtype(at::kCUDA, at::ScalarType::BFloat16);
    std::cout << "CUDA available. PPO training on GPU.\n";
  }

  // PWO-5 section 7.2 / Appendix A.1.
  //
  // The migrated layout is used by ALL SIX ARMS, P arms included -- "the arms
  // differ only in coefficients, never in architecture". A P arm running the
  // pre-migration architecture would differ from T in TWO ways at once and
  // would not be a control for it. So the heads are constructed whenever the
  // auxiliary configuration is present, NOT when a coefficient is nonzero.
  //
  // The trigger is --aux_target_path, which Appendix A.1 files under MATCHED:
  // all six arms are launched with the same target artifact and all six verify
  // its digest, while whether the targets are ever READ is decided by the
  // coefficients alone. That keeps H vs T at exactly three differences.
  //
  // The init seed is the fixed constant, never the run seed, so all six arms'
  // initial head parameters are byte-identical (section 9's matching table).
  const bool pwo5_aux_layout = !absl::GetFlag(FLAGS_aux_target_path).empty();
  const uint64_t pwo5_head_init_seed =
      dune_seed::DeriveSeed(absl::GetFlag(FLAGS_head_init_constant),
                            dune_seed::kDomainTrain, 0,
                            dune_seed::kStreamModelInit);
  if (pwo5_aux_layout) {
    std::cout << "[PWO-5] auxiliary head layout ENABLED (--aux_target_path set). "
              << "head_init_constant=" << absl::GetFlag(FLAGS_head_init_constant)
              << " -> derived init seed " << pwo5_head_init_seed << std::endl;
  }

  std::shared_ptr<open_spiel::SharedDunePolicyValueNetImpl> training_model =
      std::make_shared<open_spiel::SharedDunePolicyValueNetImpl>(
          obs_size, absl::GetFlag(FLAGS_hidden_dim), action_size,
          absl::GetFlag(FLAGS_num_blocks), absl::GetFlag(FLAGS_nonlinear_value_head),
          pwo5_aux_layout, pwo5_head_init_seed);
  // The INFERENCE model deliberately does NOT get the heads. Section 8.6
  // requires the three heads be computed "never in the rollout/inference
  // forward"; constructing them here would make that a matter of care instead
  // of a structural impossibility.
  std::shared_ptr<open_spiel::SharedDunePolicyValueNetImpl> inference_model =
      std::make_shared<open_spiel::SharedDunePolicyValueNetImpl>(
          obs_size, absl::GetFlag(FLAGS_hidden_dim), action_size,
          absl::GetFlag(FLAGS_num_blocks), absl::GetFlag(FLAGS_nonlinear_value_head));
  training_model->to(device);
  inference_model->to(device);

  if (absl::GetFlag(FLAGS_sample_counterfactual_states) && !absl::GetFlag(FLAGS_train_value_only)) {
    SpielFatalError("Counterfactual states sampling can only be enabled during value-only training (train_value_only=true).");
  }
  if (absl::GetFlag(FLAGS_sample_counterfactual_states) &&
      (absl::GetFlag(FLAGS_counterfactual_samples_per_game) <= 0 ||
       absl::GetFlag(FLAGS_counterfactual_replay_weight) <= 0)) {
    SpielFatalError("Counterfactual sample and replay counts must both be positive.");
  }

  if (absl::GetFlag(FLAGS_train_value_only)) {
    if (absl::GetFlag(FLAGS_reward_scale) != 4.0) {
      SpielFatalError("reward_scale must be exactly 4.0 when train_value_only is active.");
    }
    {
      std::string corpus_path = "data/dune_diagnostic_corpus.json";
      std::ifstream f(corpus_path);
      if (!f.good()) {
        SpielFatalError(absl::StrCat("CRITICAL EXCEPTION: Prohibited histories corpus file ", corpus_path, " is missing! Fail-closed boundary triggered."));
      }
      std::string content((std::istreambuf_iterator<char>(f)),
                           std::istreambuf_iterator<char>());
      auto json_parsed = open_spiel::json::FromString(content);
      if (!json_parsed) {
        SpielFatalError(absl::StrCat("CRITICAL EXCEPTION: Failed to parse prohibited histories corpus from ", corpus_path, "!"));
      }
      auto json_arr = json_parsed.value().GetArray();
      for (const auto& item_val : json_arr) {
        auto item = item_val.GetObject();
        auto history_arr = item.at("history").GetArray();
        std::vector<open_spiel::Action> hist;
        for (const auto& act_val : history_arr) {
          hist.push_back(static_cast<open_spiel::Action>(act_val.GetInt()));
        }
        prohibited_histories.insert(hist);
      }
      std::cout << "[INFO] Loaded " << prohibited_histories.size()
                << " prohibited histories from official corpus to prevent training leakage.\n";
    }
    std::cout << "[INFO] train_value_only is true: freezing trunk & policy head parameters.\n";
    if (absl::GetFlag(FLAGS_unfreeze_trunk)) {
      if (absl::GetFlag(FLAGS_policy_kl_anchor_coeff) <= 0.0) {
        SpielFatalError("CRITICAL SAFETY ERROR: policy_kl_anchor_coeff must be positive when unfreezing the trunk layer in value-only training mode.");
      }
    } else {
      for (auto& p : training_model->input_layer->parameters()) {
        p.set_requires_grad(false);
      }
      for (auto& block : training_model->res_blocks) {
        for (auto& p : block->parameters()) {
          p.set_requires_grad(false);
        }
      }
    }
    for (auto& p : training_model->policy_head->parameters()) {
      p.set_requires_grad(false);
    }
  }

  training_model->train();
  inference_model->eval();

  auto optimizer = open_spiel::MakeOptimizer(training_model);
  // Section 7.4: materialize BEFORE any load, so the entries exist in all six
  // arms; a subsequent load simply overwrites the ones the archive carries.
  open_spiel::MaterializeAuxOptimizerState(training_model, *optimizer);

  // --- PWO-5 sections 5.2, 8.5, 8.6: the auxiliary data path. -------------
  //
  // The target artifact and the held-out membership list are MATCHED fields
  // (Appendix A.1): all six arms are launched with the same paths and the same
  // digests, and ALL SIX VERIFY THEM. Verification is UNCONDITIONAL; USE is not
  // -- whether the targets are ever read is decided by the coefficients alone.
  // That keeps H vs T at exactly three differences while still catching a
  // corrupt artifact on every arm at launch rather than only on the two that
  // would have read it.
  open_spiel::pwo5::AuxTargetStore pwo5_store;
  open_spiel::Pwo5AuxConfig pwo5_cfg;
  pwo5_cfg.final_vp_coef = absl::GetFlag(FLAGS_final_vp_head_coef);
  pwo5_cfg.terminal_round_coef = absl::GetFlag(FLAGS_terminal_round_head_coef);
  pwo5_cfg.next_own_action_coef = absl::GetFlag(FLAGS_next_own_action_head_coef);
  pwo5_cfg.huber_delta = absl::GetFlag(FLAGS_huber_delta_final_vp);
  std::vector<int32_t> pwo5_heldout_games;
  // Carried out of the block below so the Appendix A.1 manifest fields can be
  // assembled once `search_label_fingerprint` also exists.
  std::string pwo5_target_digest, pwo5_heldout_digest,
      pwo5_update1_sampler_digest, pwo5_head_identity;
  if (pwo5_aux_layout) {
    // (a) the held-out membership list, and its registered digest.
    const std::string heldout_path = absl::GetFlag(FLAGS_aux_heldout_games_path);
    if (heldout_path.empty()) {
      SpielFatalError(
          "PWO-5: --aux_target_path is set but --aux_heldout_games_path is "
          "empty. Section 8.5 requires the whole-trajectory split membership "
          "list on every arm; without it the loader's per-row % 11 rule would "
          "silently determine the heads' held-out set, which plan section 9.6 "
          "forbids.");
    }
    std::ifstream hin(heldout_path);
    if (!hin) SpielFatalError("PWO-5: cannot open " + heldout_path);
    std::string heldout_text((std::istreambuf_iterator<char>(hin)),
                             std::istreambuf_iterator<char>());
    // Parse the ascending integer list out of "heldout_game_indices": [...].
    {
      const std::size_t key = heldout_text.find("heldout_game_indices");
      if (key == std::string::npos) {
        SpielFatalError("PWO-5: " + heldout_path +
                        " has no heldout_game_indices");
      }
      const std::size_t lb = heldout_text.find('[', key);
      const std::size_t rb = heldout_text.find(']', lb);
      std::string body = heldout_text.substr(lb + 1, rb - lb - 1);
      for (char& c : body) if (c == ',') c = ' ';
      std::istringstream bs(body);
      int32_t v;
      while (bs >> v) pwo5_heldout_games.push_back(v);
    }
    // The canonical serialization is the ASCENDING indices joined by "," with
    // no spaces -- computed over the LIST, not the file, so a reformatted JSON
    // still verifies.
    std::sort(pwo5_heldout_games.begin(), pwo5_heldout_games.end());
    std::string canonical;
    for (std::size_t i = 0; i < pwo5_heldout_games.size(); ++i) {
      if (i) canonical += ",";
      canonical += std::to_string(pwo5_heldout_games[i]);
    }
    const std::string heldout_digest =
        open_spiel::ComputeStringSHA256(canonical);
    const std::string expect_heldout = absl::GetFlag(FLAGS_aux_heldout_sha256);
    if (!expect_heldout.empty() && expect_heldout != heldout_digest) {
      SpielFatalError(
          "PWO-5 section 8.5: held-out split digest mismatch. expected " +
          expect_heldout + " computed " + heldout_digest +
          ". The split cannot silently differ across arms, replicates or the "
          "u600 extension.");
    }

    // (b) the target artifact, and its registered digest. Section 5.2: a
    // materialized artifact that is not hashed is an unpinned replay input,
    // which section 7's exact-resume gate would not catch.
    const std::string target_path = absl::GetFlag(FLAGS_aux_target_path);
    size_t target_size = 0;
    const std::string target_digest =
        open_spiel::ComputeFileSHA256(target_path, &target_size);
    const std::string expect_target = absl::GetFlag(FLAGS_aux_target_sha256);
    if (!expect_target.empty() && expect_target != target_digest) {
      SpielFatalError("PWO-5 section 5.2: --aux_target_sha256 mismatch for " +
                      target_path + ". expected " + expect_target +
                      " computed " + target_digest);
    }
    std::string load_error;
    if (!pwo5_store.Load(target_path, pwo5_heldout_games, &load_error)) {
      SpielFatalError("PWO-5: auxiliary target artifact rejected: " +
                      load_error);
    }
    if (pwo5_store.obs_size() != obs_size) {
      SpielFatalError(absl::StrFormat(
          "PWO-5: auxiliary target obs_size %lld != the game's %lld. The "
          "targets were prepared against a different observation encoder.",
          static_cast<long long>(pwo5_store.obs_size()),
          static_cast<long long>(obs_size)));
    }
    std::cout << "[PWO-5] auxiliary targets: " << target_path << " sha256="
              << target_digest << "\n"
              << "[PWO-5]   training games " << pwo5_store.training_games().size()
              << ", training rows " << pwo5_store.training_row_count()
              << ", min rows/game " << pwo5_store.MinRowsPerTrainingGame()
              << ", held out " << pwo5_heldout_games.size()
              << " (digest " << heldout_digest << ")\n"
              << "[PWO-5]   head coefficients: final_vp="
              << pwo5_cfg.final_vp_coef << " terminal_round="
              << pwo5_cfg.terminal_round_coef << " next_own_action="
              << pwo5_cfg.next_own_action_coef
              << (pwo5_cfg.AnyActive() ? "  (HEAD-ON)" : "  (HEAD-OFF: no "
                  "auxiliary forward, no head output computed)")
              << std::endl;
    // Section 17.4 floors (a2)/(a3), checked here rather than trusted.
    if (pwo5_cfg.AnyActive()) {
      const int min_rows = pwo5_store.MinRowsPerTrainingGame();
      if (min_rows < absl::GetFlag(FLAGS_aux_rows_per_game)) {
        SpielFatalError(absl::StrFormat(
            "PWO-5 section 8.6: a training game has only %d rows but "
            "--aux_rows_per_game=%d. The without-replacement draw would be "
            "infeasible.",
            min_rows, absl::GetFlag(FLAGS_aux_rows_per_game)));
      }
      if (static_cast<int>(pwo5_store.training_games().size()) <
          absl::GetFlag(FLAGS_aux_games_per_update)) {
        SpielFatalError(
            "PWO-5 section 8.6: fewer training games than "
            "--aux_games_per_update.");
      }
    }

    // Section 8.6's update-1 sampler digest, computed EAGERLY on ALL SIX arms
    // rather than captured when update 1 happens to run. Two reasons, both
    // registered: T and P arms have all-zero coefficients so they never draw,
    // yet Appendix A.1 note 3 requires the digest in EVERY arm's manifest so a
    // triplet can be checked to have drawn identically; and the bootstrap
    // manifest is written before any update executes, so a lazily captured
    // digest could not appear in it at all.
    const uint64_t update1_aux_seed = dune_seed::DeriveSeed(
        master, dune_seed::kDomainTrain, /*update=*/1, /*aux=*/0,
        dune_seed::kStreamAuxSampling);
    const auto update1_draw = pwo5_store.Draw(
        update1_aux_seed, absl::GetFlag(FLAGS_aux_games_per_update),
        absl::GetFlag(FLAGS_aux_rows_per_game),
        absl::GetFlag(FLAGS_aux_batches_per_update));
    pwo5_update1_sampler_digest =
        open_spiel::ComputeStringSHA256(update1_draw.digest);
    pwo5_target_digest = target_digest;
    pwo5_heldout_digest = heldout_digest;
    pwo5_head_identity = ComputeAuxHeadIdentity(training_model);
    std::cout << "[PWO-5] update-1 sampler digest (precomputed) "
              << pwo5_update1_sampler_digest << "\n"
              << "[PWO-5] head-init identity " << pwo5_head_identity
              << std::endl;
  }

  std::shared_ptr<open_spiel::SharedDunePolicyValueNetImpl> anchor_model = nullptr;
  if (absl::GetFlag(FLAGS_train_value_only) && absl::GetFlag(FLAGS_unfreeze_trunk)) {
    anchor_model = std::make_shared<open_spiel::SharedDunePolicyValueNetImpl>(
        obs_size, absl::GetFlag(FLAGS_hidden_dim), action_size,
        absl::GetFlag(FLAGS_num_blocks), absl::GetFlag(FLAGS_nonlinear_value_head));
    anchor_model->to(device);
  }

  std::string init_mode = absl::GetFlag(FLAGS_init_mode);
  if (init_mode.empty()) {
    SpielFatalError("Required flag --init_mode is missing. Must be 'random', 'checkpoint', 'bootstrap', or 'validate_legacy'.");
  }

  std::string config_fingerprint = open_spiel::ComputeConfigFingerprint();
  std::string search_label_fingerprint = open_spiel::GetSearchLabelFingerprint(absl::GetFlag(FLAGS_search_label_dir));
  std::string run_uuid = open_spiel::GenerateUUID();

  // -------------------------------------------------------------------------
  // PWO-5 Appendix A.1 note 3: assemble the manifest block, ONCE, before any
  // manifest is written or read. Both init_mode branches below consume it.
  // -------------------------------------------------------------------------
  if (pwo5_aux_layout) {
    open_spiel::json::Object& f = open_spiel::g_pwo5_manifest_fields;
    // --- the five TREATMENT fields (section 9.1). Differing ACROSS arms is
    // the point; differing from the arm's OWN registered value across a resume
    // is the STOP, which is what the within-arm check below enforces.
    f["final_vp_head_coef"] =
        open_spiel::Pwo5Double(absl::GetFlag(FLAGS_final_vp_head_coef));
    f["terminal_round_head_coef"] =
        open_spiel::Pwo5Double(absl::GetFlag(FLAGS_terminal_round_head_coef));
    f["next_own_action_head_coef"] =
        open_spiel::Pwo5Double(absl::GetFlag(FLAGS_next_own_action_head_coef));
    f["search_lambda"] =
        open_spiel::Pwo5Double(absl::GetFlag(FLAGS_search_lambda));
    f["search_label_dir"] = absl::GetFlag(FLAGS_search_label_dir);
    // --- the MATCHED fields: byte-identical on all six arms.
    f["aux_target_path"] = absl::GetFlag(FLAGS_aux_target_path);
    f["aux_target_sha256"] = pwo5_target_digest;
    f["aux_heldout_games_path"] = absl::GetFlag(FLAGS_aux_heldout_games_path);
    f["aux_heldout_sha256"] = pwo5_heldout_digest;
    f["aux_games_per_update"] =
        static_cast<int64_t>(absl::GetFlag(FLAGS_aux_games_per_update));
    f["aux_rows_per_game"] =
        static_cast<int64_t>(absl::GetFlag(FLAGS_aux_rows_per_game));
    f["aux_batches_per_update"] =
        static_cast<int64_t>(absl::GetFlag(FLAGS_aux_batches_per_update));
    f["huber_delta_final_vp"] =
        open_spiel::Pwo5Double(absl::GetFlag(FLAGS_huber_delta_final_vp));
    f["head_init_constant"] =
        static_cast<int64_t>(absl::GetFlag(FLAGS_head_init_constant));
    f["emit_canary_columns"] = absl::GetFlag(FLAGS_emit_canary_columns);
    // --- identities the flags alone do not pin. The label pack's FILE ROLES
    // ride `search_label_fingerprint`: a role-aware manifest binds every
    // (filename, sha256, role) triple into that digest, so persisting it here
    // persists the role mapping.
    f["head_init_identity"] = pwo5_head_identity;
    f["search_label_fingerprint"] = search_label_fingerprint;
    // --- section 8.6's update-1 sampler digest. Matched WITHIN a triplet
    // (T/P/H share a base seed) and expected to differ BETWEEN triplets, so it
    // is neither a treatment nor a global matched field.
    f["update1_sampler_digest"] = pwo5_update1_sampler_digest;
    open_spiel::g_pwo5_manifest_active = true;
    std::cout << "[PWO-5] manifest fingerprint "
              << open_spiel::ComputePwo5Fingerprint(f) << std::endl;
  }

  std::atomic<uint64_t> total_env_steps{0};
  std::atomic<uint64_t> next_episode_id{0};
  // Phase 18B online-collection persistent state (restored from the manifest on
  // resume, persisted at every checkpoint). aux_next_episode_id_persist TRAILS
  // the collector's live cursor by one pipeline step: it is the next-episode-id
  // as of the LAST CONSUMED update, which is exactly what a resume must continue
  // from so episode ids neither repeat nor skip.
  uint64_t aux_next_episode_id_persist = 0;
  int64_t aux_cum_accepted = 0, aux_cum_rejected = 0;
  int64_t aux_cum_role_searches[3] = {0, 0, 0};
  int64_t aux_cum_role_accepted[3] = {0, 0, 0};
  int64_t aux_cum_granted = 0, aux_cum_organic = 0;
  std::string aux_hash_chain = "";
  int start_update = 1;
  int target_end_update = absl::GetFlag(FLAGS_target_end_update);

  if (init_mode == "random") {
    std::string model_path = absl::GetFlag(FLAGS_model_checkpoint);
    std::string optim_path = absl::GetFlag(FLAGS_optim_checkpoint);
    std::filesystem::path m_path = model_path;
    m_path.replace_extension(".json");
    std::string manifest_path = m_path.string();
    if (std::filesystem::exists(model_path) || std::filesystem::exists(optim_path) || std::filesystem::exists(manifest_path)) {
      SpielFatalError("init_mode=random but checkpoint or manifest file already exists. Refusing to overwrite.");
    }
    std::cout << "Starting fresh training run (init_mode=random).\n";
  } else if (init_mode == "checkpoint") {
    if (absl::GetFlag(FLAGS_pipeline)) {
      SpielFatalError("init_mode=checkpoint with pipeline=true is rejected because prefetched rollout is not persisted.");
    }
    std::string model_path = absl::GetFlag(FLAGS_model_checkpoint);
    std::string optim_path = absl::GetFlag(FLAGS_optim_checkpoint);
    std::filesystem::path m_path = model_path;
    m_path.replace_extension(".json");
    std::string manifest_path = m_path.string();

    // WO-20 acceptance-contract guard, deliberately BEFORE the fingerprint
    // check: adding the source to the fingerprint already makes a pre-WO-20
    // 18B checkpoint fail to resume, but it would fail as an anonymous hash
    // mismatch. Diagnose the specific cause first, while we can still name it.
    //
    // The cumulative accepted/rejected counters and the accepted-target hash
    // chain are only meaningful within ONE acceptance contract. A manifest
    // written before WO-20 has no source field and accumulated its counters
    // against the post-noise tree prior, so silently extending them under the
    // new raw-prior default would blend two contracts into one total.
    if (absl::GetFlag(FLAGS_online_search_collection)) {
      OnlineCollectionState prior_state;
      std::string oc_probe_err;
      if (ReadOnlineCollectionState(manifest_path, prior_state, oc_probe_err) &&
          prior_state.present) {
        const std::string effective =
            AcceptancePriorSourceName(EffectiveAcceptancePriorSource());
        const bool stated = !absl::GetFlag(FLAGS_collector_acceptance_prior).empty();
        if (prior_state.acceptance_prior_source.empty() && !stated) {
          SpielFatalError(absl::StrCat(
              "Refusing to resume: ", manifest_path, " predates WO-20 and has "
              "no online_collection.acceptance_prior_source, so its "
              "cum_accepted=", prior_state.cum_accepted, " / cum_rejected=",
              prior_state.cum_rejected, " were accumulated against the "
              "POST-NOISE tree prior. Continuing under the new default ('",
              effective, "') would mix two coverage contracts in one counter "
              "set and one accepted-target hash chain.\n"
              "  To continue that run's original contract: "
              "--collector_acceptance_prior=tree_prior\n"
              "  To adopt the new contract knowingly (counters and hash chain "
              "then span both): --collector_acceptance_prior=", effective, "\n"
              "  Note either choice changes the config fingerprint relative to "
              "the stored one, which is intended — the semantics did change."));
        }
        if (!prior_state.acceptance_prior_source.empty() &&
            prior_state.acceptance_prior_source != effective) {
          SpielFatalError(absl::StrCat(
              "Refusing to resume: ", manifest_path, " collected under "
              "acceptance_prior_source='", prior_state.acceptance_prior_source,
              "' but this run resolves to '", effective, "'. Its counters and "
              "hash chain cannot be extended across a change of coverage "
              "contract. Pass --collector_acceptance_prior=",
              prior_state.acceptance_prior_source, " to continue it, or start "
              "a fresh run."));
        }
      }
    }

    CheckpointManifest manifest;
    std::string err;
    // Legacy fingerprints predate critic-remediation flags and therefore
    // cannot prove which parameters or data source were active. Never use
    // that compatibility escape hatch for value-only resumes.
    std::string legacy_fingerprint = absl::GetFlag(FLAGS_train_value_only)
        ? ""
        : ComputeLegacyConfigFingerprint();
    if (!ParseAndValidateManifest(manifest_path, model_path, optim_path,
                                  master, target_end_update,
                                  absl::GetFlag(FLAGS_seed_scheme_version),
                                  config_fingerprint, search_label_fingerprint,
                                  absl::GetFlag(FLAGS_hidden_dim),
                                  absl::GetFlag(FLAGS_num_blocks),
                                  manifest, err, legacy_fingerprint)) {
      SpielFatalError(err);
    }

    // ---------------------------------------------------------------------
    // PWO-5 Appendix A.1 note 3: the within-arm invariant. Every field of the
    // block -- both target flags, the split digest, the three head
    // coefficients, the three sampler-shape flags, the Huber delta, the
    // head-init constant AND identity, --emit_canary_columns, the label pack's
    // role-bearing fingerprint, and the update-1 sampler digest -- MUST NOT
    // change across a resume or the u600 extension. An arm's own configuration
    // is fixed for its lifetime.
    //
    // The check runs HERE: before the model is loaded, before the optimizer is
    // restored, and before a single update executes.
    // ---------------------------------------------------------------------
    {
      std::string stored_fp, stored_block;
      open_spiel::json::Object stored_fields;
      bool stored_present = false;
      {
        std::ifstream mfs(manifest_path);
        std::string content((std::istreambuf_iterator<char>(mfs)),
                            std::istreambuf_iterator<char>());
        auto parsed = open_spiel::json::FromString(content);
        if (parsed && parsed->IsObject()) {
          const auto& obj = parsed->GetObject();
          auto it = obj.find("pwo5");
          auto fp_it = obj.find("pwo5_fingerprint");
          if (it != obj.end() && it->second.IsObject()) {
            stored_fields = it->second.GetObject();
            stored_present = true;
            stored_block = open_spiel::json::ToString(stored_fields);
          }
          if (fp_it != obj.end() && fp_it->second.IsString()) {
            stored_fp = fp_it->second.GetString();
          }
        }
      }
      if (stored_present != open_spiel::g_pwo5_manifest_active) {
        SpielFatalError(
            std::string("PWO-5 resume contract: the manifest ") +
            (stored_present ? "CARRIES" : "does NOT carry") +
            " a pwo5 block but this run " +
            (open_spiel::g_pwo5_manifest_active ? "IS" : "is NOT") +
            " a PWO-5 arm (--aux_target_path " +
            (open_spiel::g_pwo5_manifest_active ? "set" : "empty") +
            "). Resuming across that boundary would silently change the "
            "architecture, the auxiliary data path, or both.");
      }
      if (stored_present) {
        const std::string current_fp =
            open_spiel::ComputePwo5Fingerprint(open_spiel::g_pwo5_manifest_fields);
        if (stored_fp != current_fp) {
          // Name the fields that moved rather than emitting an anonymous hash
          // mismatch: a resume failure a human cannot diagnose is a resume
          // failure a human will work around.
          std::string detail;
          for (const auto& kv : open_spiel::g_pwo5_manifest_fields) {
            auto sit = stored_fields.find(kv.first);
            const std::string now = open_spiel::json::ToString(kv.second);
            const std::string was = (sit == stored_fields.end())
                                        ? std::string("<absent>")
                                        : open_spiel::json::ToString(sit->second);
            if (now != was) {
              detail += "\n    " + kv.first + ": manifest=" + was +
                        "  this run=" + now;
            }
          }
          for (const auto& kv : stored_fields) {
            if (open_spiel::g_pwo5_manifest_fields.find(kv.first) ==
                open_spiel::g_pwo5_manifest_fields.end()) {
              detail += "\n    " + kv.first + ": manifest=" +
                        open_spiel::json::ToString(kv.second) +
                        "  this run=<absent>";
            }
          }
          SpielFatalError(
              "PWO-5 resume contract VIOLATED (Appendix A.1 note 3).\n"
              "  manifest pwo5_fingerprint: " + stored_fp + "\n"
              "  this run's fingerprint:    " + current_fp + "\n"
              "  fields that differ:" + detail +
              "\n  An arm's own configuration is fixed for its lifetime. "
              "Training is refused before it continues.");
        }
        std::cout << "[PWO-5] resume contract verified: pwo5_fingerprint "
                  << current_fp << " matches the manifest ("
                  << open_spiel::g_pwo5_manifest_fields.size() << " fields)."
                  << std::endl;
        (void)stored_block;
      }
    }

    int global_update = manifest.global_update;
    if (global_update >= target_end_update) {
      SpielFatalError(absl::StrFormat(
          "global_update (%d) >= target_end_update (%d). "
          "Resuming is not possible as training is already finished.",
          global_update, target_end_update));
    }
    uint64_t manifest_env_steps = manifest.total_env_steps;
    uint64_t manifest_episode_id = manifest.next_episode_id;
    std::string manifest_run_uuid = manifest.run_uuid;

    // Phase 18B exact resume: restore online-collection cursor + cumulative
    // counters + accepted-target hash chain from the manifest (if present).
    {
      open_spiel::OnlineCollectionState restored;
      std::string oc_err;
      if (!open_spiel::ReadOnlineCollectionState(manifest_path, restored, oc_err)) {
        SpielFatalError("Failed to read online_collection manifest state: " + oc_err);
      }
      if (restored.present) {
        aux_next_episode_id_persist = restored.next_auxiliary_episode_id;
        aux_cum_accepted = restored.cum_accepted;
        aux_cum_rejected = restored.cum_rejected;
        aux_cum_granted = restored.cum_granted;
        aux_cum_organic = restored.cum_organic;
        for (int r = 0; r < 3; ++r) {
          aux_cum_role_searches[r] = restored.cum_role_searches[r];
          aux_cum_role_accepted[r] = restored.cum_role_accepted[r];
        }
        aux_hash_chain = restored.accepted_hash_chain;
        std::cout << absl::StrFormat(
            "[18B] Restored online-collection state: next_aux_episode_id=%llu "
            "cum_accepted=%lld cum_granted=%lld cum_organic=%lld\n",
            (unsigned long long)aux_next_episode_id_persist,
            (long long)aux_cum_accepted, (long long)aux_cum_granted,
            (long long)aux_cum_organic);
      }
    }

    try {
      LoadModelCheckpoint(training_model, model_path, device);
      if (anchor_model) {
        LoadModelCheckpoint(anchor_model, model_path, device);
        anchor_model->eval();
        for (auto& p : anchor_model->parameters()) {
          p.set_requires_grad(false);
        }
        std::cout << "[INFO] Loaded anchor model for policy KL penalty.\n";
      }
      if (!absl::GetFlag(FLAGS_train_value_only)) {
        // PWO-5 section 7.2: a pre-head archive migrates; a post-head archive
        // loads normally.
        if (!open_spiel::LoadOptimizerCheckpointMigrating(
                training_model, *optimizer, optim_path, device)) {
          torch::load(*optimizer, optim_path, device);
        }
      } else {
        std::cout << "[INFO] train_value_only is true: keeping fresh value-only optimizer. Skipping optimizer checkpoint load.\n";
      }
    } catch (const c10::Error& e) {
      SpielFatalError("LibTorch load failed: " + std::string(e.msg()));
    }

    open_spiel::SetOptimizerLearningRate(*optimizer, absl::GetFlag(FLAGS_learning_rate));
    for (size_t g = 0; g < optimizer->param_groups().size(); ++g) {
      auto& options =
          static_cast<torch::optim::AdamWOptions&>(
              optimizer->param_groups()[g].options());
      // PWO-5 section 7.4: group 2 is the auxiliary-head group and must stay
      // at weight_decay = 0.0. The pre-PWO-5 form of this loop was
      // `g == 0 ? policy_wd : wd`, which would have applied --weight_decay to
      // the heads on every resume and silently drifted a head-off arm's head
      // parameters away from their initial values.
      options.weight_decay(g == 0 ? absl::GetFlag(FLAGS_policy_weight_decay)
                          : g == 1 ? absl::GetFlag(FLAGS_weight_decay)
                                   : 0.0);
    }
    for (auto& pair : optimizer->state()) {
      if (auto* state =
              dynamic_cast<torch::optim::AdamWParamState*>(pair.second.get())) {
        if (state->exp_avg().defined()) state->exp_avg() = state->exp_avg().to(device);
        if (state->exp_avg_sq().defined()) state->exp_avg_sq() = state->exp_avg_sq().to(device);
        if (state->max_exp_avg_sq().defined()) {
          state->max_exp_avg_sq() = state->max_exp_avg_sq().to(device);
        }
      }
    }

    start_update = global_update + 1;
    total_env_steps.store(manifest_env_steps);
    next_episode_id.store(manifest_episode_id);
    run_uuid = manifest_run_uuid;

    std::cout << "Successfully verified and loaded checkpoint manifest. Resuming from update "
              << start_update << " to " << target_end_update << ".\n";
  } else if (init_mode == "bootstrap") {
    std::string model_path = absl::GetFlag(FLAGS_model_checkpoint);
    std::string optim_path = absl::GetFlag(FLAGS_optim_checkpoint);
    if (model_path.find("artifacts/baselines") != std::string::npos ||
        model_path.find("artifacts/branch_a_frozen") != std::string::npos) {
      SpielFatalError("Refusing to bootstrap directly inside the frozen baseline or branch_a_frozen directory: " + model_path +
                      "\nRun bootstrap beside a BRANCH COPY in a separate output directory.");
    }
    if (!std::filesystem::exists(model_path)) {
      SpielFatalError("Model file not found for bootstrap: " + model_path);
    }
    if (!std::filesystem::exists(optim_path)) {
      SpielFatalError("Optimizer file not found for bootstrap: " + optim_path);
    }

    try {
      LoadModelCheckpoint(training_model, model_path, device);
      if (anchor_model) {
        LoadModelCheckpoint(anchor_model, model_path, device);
        anchor_model->eval();
        for (auto& p : anchor_model->parameters()) {
          p.set_requires_grad(false);
        }
        std::cout << "[INFO] Loaded anchor model for policy KL penalty.\n";
      }
      if (!absl::GetFlag(FLAGS_train_value_only)) {
        // PWO-5 section 7.2: the bootstrap path is where the Branch-A u2450
        // optimizer -- written before the three heads existed -- is migrated.
        if (!open_spiel::LoadOptimizerCheckpointMigrating(
                training_model, *optimizer, optim_path, device)) {
          torch::load(*optimizer, optim_path, device);
        }
      } else {
        std::cout << "[INFO] train_value_only is true: using fresh value-only optimizer (skipping baseline optimizer load).\n";
      }
    } catch (const c10::Error& e) {
      SpielFatalError("LibTorch load failed during bootstrap: " + std::string(e.msg()));
    }

    size_t model_size = 0;
    std::string model_hash = open_spiel::ComputeFileSHA256(model_path, &model_size);
    size_t optim_size = 0;
    std::string optim_hash = open_spiel::ComputeFileSHA256(optim_path, &optim_size);

    std::filesystem::path m_path = model_path;
    m_path.replace_extension(".json");
    std::string manifest_path = m_path.string();
    std::string manifest_tmp = manifest_path + ".tmp";

    json::Object manifest_obj;
    manifest_obj["schema_version"] = static_cast<int64_t>(2);
    manifest_obj["global_update"] = static_cast<int64_t>(absl::GetFlag(FLAGS_start_update));
    manifest_obj["target_end_update"] = static_cast<int64_t>(absl::GetFlag(FLAGS_target_end_update));
    manifest_obj["total_env_steps"] = static_cast<int64_t>(absl::GetFlag(FLAGS_start_env_steps));
    manifest_obj["next_episode_id"] = static_cast<int64_t>(absl::GetFlag(FLAGS_start_episode_id));
    manifest_obj["base_seed"] = static_cast<int64_t>(absl::GetFlag(FLAGS_seed));
    manifest_obj["seed_scheme_version"] = static_cast<int64_t>(absl::GetFlag(FLAGS_seed_scheme_version));
    manifest_obj["config_fingerprint"] = config_fingerprint;
    manifest_obj["search_label_fingerprint"] = search_label_fingerprint;
    manifest_obj["run_uuid"] = open_spiel::GenerateUUID();
    manifest_obj["checkpoint_uuid"] = open_spiel::GenerateUUID();
    manifest_obj["model_filename"] = std::filesystem::path(model_path).filename().string();
    manifest_obj["model_file_size"] = static_cast<int64_t>(model_size);
    manifest_obj["model_sha256"] = model_hash;
    manifest_obj["optimizer_filename"] = std::filesystem::path(optim_path).filename().string();
    manifest_obj["optimizer_file_size"] = static_cast<int64_t>(optim_size);
    manifest_obj["optimizer_sha256"] = optim_hash;
    manifest_obj["hidden_dim"] = static_cast<int64_t>(absl::GetFlag(FLAGS_hidden_dim));
    manifest_obj["num_blocks"] = static_cast<int64_t>(absl::GetFlag(FLAGS_num_blocks));
    manifest_obj["legacy_migration_provenance"] = "Synthesized via init_mode=bootstrap";
    // PWO-5 Appendix A.1 note 3: the same block the checkpoint writer emits, so
    // the bootstrap manifest an arm resumes FROM already carries the contract.
    if (g_pwo5_manifest_active) {
      manifest_obj["pwo5"] = json::Value(g_pwo5_manifest_fields);
      manifest_obj["pwo5_fingerprint"] =
          json::Value(ComputePwo5Fingerprint(g_pwo5_manifest_fields));
    }

    {
      std::ofstream ofs(manifest_tmp);
      if (!ofs) {
        SpielFatalError("Could not open manifest temp file for writing: " + manifest_tmp);
      }
      ofs << json::ToString(manifest_obj, true);
    }
    std::filesystem::rename(manifest_tmp, manifest_path);
    std::cout << "Successfully bootstrapped manifest at " << manifest_path << "\n";
    exit(0);
  } else if (init_mode == "validate_legacy") {
    std::string model_path = absl::GetFlag(FLAGS_model_checkpoint);
    std::string optim_path = absl::GetFlag(FLAGS_optim_checkpoint);
    std::filesystem::path manifest_path;
    std::string artifact_manifest_flag = absl::GetFlag(FLAGS_artifact_manifest);
    if (!artifact_manifest_flag.empty()) {
      manifest_path = artifact_manifest_flag;
    } else {
      manifest_path = std::filesystem::path(model_path).parent_path() / "manifest.json";
    }

    if (!std::filesystem::exists(manifest_path)) {
      SpielFatalError("init_mode=validate_legacy but Phase 1 manifest file not found at: " + manifest_path.string());
    }

    std::ifstream ifs(manifest_path.string());
    if (!ifs) {
      SpielFatalError("Could not open manifest file at: " + manifest_path.string());
    }
    std::string content((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
    auto val_opt = open_spiel::json::FromString(content);
    if (!val_opt.has_value() || !val_opt->IsObject()) {
      SpielFatalError("Manifest file is malformed JSON at: " + manifest_path.string());
    }
    const auto& manifest_obj = val_opt->GetObject();

    auto get_object_field = [&](const std::string& key) -> const json::Object& {
      auto it = manifest_obj.find(key);
      if (it == manifest_obj.end() || !it->second.IsObject()) {
        SpielFatalError("Manifest missing required object field: " + key);
      }
      return it->second.GetObject();
    };

    auto get_nested_string = [](const json::Object& obj, const std::string& key) -> std::string {
      auto it = obj.find(key);
      if (it == obj.end() || !it->second.IsString()) {
        SpielFatalError("Nested object missing required string field: " + key);
      }
      return it->second.GetString();
    };

    auto get_nested_int = [](const json::Object& obj, const std::string& key) -> int64_t {
      auto it = obj.find(key);
      if (it == obj.end() || !it->second.IsInt()) {
        SpielFatalError("Nested object missing required int field: " + key);
      }
      return it->second.GetInt();
    };

    const auto& model_obj = get_object_field("model");
    const auto& optim_obj = get_object_field("optimizer");
    const auto& arch_obj = get_object_field("architecture");

    std::string expected_model_sha256 = get_nested_string(model_obj, "sha256");
    size_t expected_model_size = get_nested_int(model_obj, "size_bytes");

    std::string expected_optim_sha256 = get_nested_string(optim_obj, "sha256");
    size_t expected_optim_size = get_nested_int(optim_obj, "size_bytes");

    int64_t manifest_hidden_dim = get_nested_int(arch_obj, "hidden_dim");
    int64_t manifest_num_blocks = get_nested_int(arch_obj, "num_blocks");
    int64_t manifest_obs_size = get_nested_int(arch_obj, "observation_size");
    int64_t manifest_action_size = get_nested_int(arch_obj, "action_size");

    if (!std::filesystem::exists(model_path)) {
      SpielFatalError("Model file not found: " + model_path);
    }
    if (!std::filesystem::exists(optim_path)) {
      SpielFatalError("Optimizer file not found: " + optim_path);
    }

    size_t actual_model_size = 0;
    std::string actual_model_hash = open_spiel::ComputeFileSHA256(model_path, &actual_model_size);
    if (actual_model_size != expected_model_size) {
      SpielFatalError(absl::StrFormat("Model file size mismatch. Manifest: %d, Actual: %d", expected_model_size, actual_model_size));
    }
    if (actual_model_hash != expected_model_sha256) {
      SpielFatalError("Model SHA-256 hash mismatch. Manifest: " + expected_model_sha256 + ", Actual: " + actual_model_hash);
    }

    size_t actual_optim_size = 0;
    std::string actual_optim_hash = open_spiel::ComputeFileSHA256(optim_path, &actual_optim_size);
    if (actual_optim_size != expected_optim_size) {
      SpielFatalError(absl::StrFormat("Optimizer file size mismatch. Manifest: %d, Actual: %d", expected_optim_size, actual_optim_size));
    }
    if (actual_optim_hash != expected_optim_sha256) {
      SpielFatalError("Optimizer SHA-256 hash mismatch. Manifest: " + expected_optim_sha256 + ", Actual: " + actual_optim_hash);
    }

    if (absl::GetFlag(FLAGS_hidden_dim) != manifest_hidden_dim) {
      SpielFatalError(absl::StrFormat("hidden_dim mismatch. Flags: %d, Manifest: %d", absl::GetFlag(FLAGS_hidden_dim), manifest_hidden_dim));
    }
    if (absl::GetFlag(FLAGS_num_blocks) != manifest_num_blocks) {
      SpielFatalError(absl::StrFormat("num_blocks mismatch. Flags: %d, Manifest: %d", absl::GetFlag(FLAGS_num_blocks), manifest_num_blocks));
    }
    if (obs_size != manifest_obs_size) {
      SpielFatalError(absl::StrFormat("observation_size mismatch. Game: %d, Manifest: %d", obs_size, manifest_obs_size));
    }
    if (action_size != manifest_action_size) {
      SpielFatalError(absl::StrFormat("action_size mismatch. Game: %d, Manifest: %d", action_size, manifest_action_size));
    }

    try {
      LoadModelCheckpoint(training_model, model_path, device);
    } catch (const c10::Error& e) {
      SpielFatalError("LibTorch load failed: " + std::string(e.msg()));
    }

    int64_t actual_param_count = 0;
    for (const auto& param : training_model->parameters()) {
      actual_param_count += param.numel();
    }

    auto dummy_model = std::make_shared<open_spiel::SharedDunePolicyValueNetImpl>(
        manifest_obs_size, manifest_hidden_dim, manifest_action_size, manifest_num_blocks, absl::GetFlag(FLAGS_nonlinear_value_head));
    int64_t expected_param_count = 0;
    for (const auto& param : dummy_model->parameters()) {
      expected_param_count += param.numel();
    }

    if (actual_param_count != expected_param_count) {
      SpielFatalError(absl::StrFormat("Parameter count mismatch. Expected: %d (calculated from architecture), Actual: %d", expected_param_count, actual_param_count));
    }

    std::cout << absl::StrFormat("Legacy checkpoint validation successful!\n"
                                 "  Model parameter count: %d\n"
                                 "  Model size and hash match Phase 1 manifest.\n"
                                 "  Optimizer size and hash match Phase 1 manifest.\n", actual_param_count);
    exit(0);
  } else {
    SpielFatalError("Unsupported init_mode: " + init_mode);
  }

  std::shared_mutex sync_mutex;
  std::mutex eval_mutex;
  open_spiel::SyncModels(training_model, inference_model, &sync_mutex);

  std::shared_ptr<open_spiel::IGameEvaluator> evaluator;
  if (absl::GetFlag(FLAGS_deterministic_rollout_eval)) {
    evaluator = std::make_shared<open_spiel::DeterministicEvaluator>(
        inference_model, device, &eval_mutex, &sync_mutex);
  } else {
    evaluator = std::make_shared<open_spiel::BatchedEvaluator>(
        inference_model, absl::GetFlag(FLAGS_eval_batch_size),
        absl::GetFlag(FLAGS_eval_timeout_ms), device, &sync_mutex, 0.0f,
        absl::GetFlag(FLAGS_evaluator_device_synchronize));
  }

  // --- Phase 18B online auxiliary-search collection setup ---
  // The collector runs its 64-sim searches through a DuneNNEvaluator over the
  // FROZEN inference_model snapshot (never the training model, never post-update
  // weights). Collection is placed alongside each rollout (below) so aux and PPO
  // examples always come from the same snapshot. SyncModels only runs on the main
  // thread after the bg collection joins, so the snapshot is stable during a
  // collection even without the evaluator's sync_mutex.
  const bool online_search_collection = absl::GetFlag(FLAGS_online_search_collection);
  if (online_search_collection && !absl::GetFlag(FLAGS_search_label_dir).empty()) {
    SpielFatalError(
        "--online_search_collection and --search_label_dir are mutually "
        "exclusive (online collection vs legacy offline distillation).");
  }
  std::shared_ptr<open_spiel::OnlineSearchCollector> aux_collector;
  std::shared_ptr<open_spiel::DuneNNEvaluator> aux_evaluator;
  float aux_logit_cap = 0.0f;  // cap actually handed to aux_evaluator
  double aux_search_loss_coef_target = absl::GetFlag(FLAGS_search_loss_coef);
  int aux_search_loss_warmup = absl::GetFlag(FLAGS_search_loss_warmup_update);
  double aux_abort_ratio = absl::GetFlag(FLAGS_abort_grad_norm_ratio);
  open_spiel::OnlineSearchConfig aux_config;  // effective echo (persisted in manifest)
  if (online_search_collection) {
    if (absl::GetFlag(FLAGS_auxiliary_search_seed_domain) == 0) {
      SpielFatalError(
          "--auxiliary_search_seed_domain must be set nonzero when "
          "--online_search_collection is enabled (no silent default).");
    }
    aux_config.auxiliary_games = absl::GetFlag(FLAGS_auxiliary_games);
    aux_config.auxiliary_search_seed_domain =
        absl::GetFlag(FLAGS_auxiliary_search_seed_domain);
    aux_config.dirichlet_epsilon = absl::GetFlag(FLAGS_collector_dirichlet_epsilon);
    aux_config.acceptance_prior_source = EffectiveAcceptancePriorSource();
    aux_config.target_sharpen_exponent =
        absl::GetFlag(FLAGS_target_sharpen_exponent);
    aux_config.swordmaster_grant_fraction =
        absl::GetFlag(FLAGS_swordmaster_grant_fraction);
    aux_config.swordmaster_grant_round =
        absl::GetFlag(FLAGS_swordmaster_grant_round);
    // Exact resume: continue the aux episode cursor from the manifest.
    aux_config.next_auxiliary_episode_id =
        static_cast<int64_t>(aux_next_episode_id_persist);
    // Same policy transform as the PPO rollout (which applies FLAGS_logit_cap
    // itself, below): the 18B teacher's priors and the student being trained
    // must not sit under different transforms. Echoed on the startup line so a
    // run log records the cap the aux evaluator actually received.
    aux_logit_cap = static_cast<float>(absl::GetFlag(FLAGS_logit_cap));
    aux_evaluator = std::make_shared<open_spiel::DuneNNEvaluator>(
        inference_model, device, aux_logit_cap);
    aux_collector = std::make_shared<open_spiel::OnlineSearchCollector>(
        aux_config, config_fingerprint);
    std::cout << absl::StrFormat(
        "[18B] Online collection ON | aux_games=%d seed_domain=%llu "
        "dirichlet_eps=%.3f grant_frac=%.3f grant_round=%d loss_coef=%.3f/warmup%d "
        "abort_ratio=%.3f sharpen=%.3f resume_ep=%llu logit_cap=%.3f "
        "accept_prior=%s\n",
        aux_config.auxiliary_games,
        (unsigned long long)aux_config.auxiliary_search_seed_domain,
        aux_config.dirichlet_epsilon, aux_config.swordmaster_grant_fraction,
        aux_config.swordmaster_grant_round, aux_search_loss_coef_target,
        aux_search_loss_warmup, aux_abort_ratio, aux_config.target_sharpen_exponent,
        (unsigned long long)aux_next_episode_id_persist, aux_logit_cap,
        // Which prior the acceptance rate in this run's diagnostics was measured
        // against — noise does not silently redefine it (WO-20).
        open_spiel::AcceptancePriorSourceName(aux_config.acceptance_prior_source));
  }

  // Collect one update's aux examples into `cr`, from the frozen snapshot. Runs
  // in whatever thread collects the paired PPO rollout (pre-loop main thread or
  // the bg pipeline thread), advancing the collector's private episode cursor.
  auto run_aux_collection = [&](int update_id, open_spiel::CollectResult& cr) {
    if (!online_search_collection) return;
    aux_collector->CollectUpdate(update_id, game, aux_evaluator, &cr.aux_examples,
                                 &cr.aux_stats);
  };
  // Fold a CONSUMED update's collection into the persisted cumulative counters,
  // the chained accepted-target hash, and the resume cursor (main thread only,
  // so no race with the bg collector).
  auto account_aux_consumed = [&](const open_spiel::CollectResult& cr) {
    if (!online_search_collection) return;
    const open_spiel::OnlineSearchCollectionStats& st = cr.aux_stats;
    aux_cum_accepted += st.accepted_targets;
    aux_cum_rejected += st.rejected_incomplete;
    aux_cum_granted += st.swordmaster_granted_games;
    aux_cum_organic += st.swordmaster_organic_games;
    for (int r = 0; r < 3; ++r) {
      aux_cum_role_searches[r] += st.by_role[r].searches;
      aux_cum_role_accepted[r] += st.by_role[r].accepted;
    }
    aux_next_episode_id_persist = static_cast<uint64_t>(st.next_episode_id);
    std::string digest;
    for (const auto& ex : cr.aux_examples) {
      digest.append(reinterpret_cast<const char*>(&ex.episode_id), sizeof(ex.episode_id));
      int64_t did = ex.decision_id;
      digest.append(reinterpret_cast<const char*>(&did), sizeof(did));
      int32_t pl = ex.player;
      digest.append(reinterpret_cast<const char*>(&pl), sizeof(pl));
      for (Action a : ex.legal_actions)
        digest.append(reinterpret_cast<const char*>(&a), sizeof(a));
      for (double v : ex.normalized_visits)
        digest.append(reinterpret_cast<const char*>(&v), sizeof(v));
      digest.append(reinterpret_cast<const char*>(&ex.value_target), sizeof(ex.value_target));
    }
    aux_hash_chain = open_spiel::ComputeStringSHA256(aux_hash_chain + digest);
  };
  // Snapshot the persisted online-collection state for a checkpoint manifest.
  auto build_aux_state = [&]() {
    open_spiel::OnlineCollectionState s;
    s.present = true;
    s.auxiliary_games = aux_config.auxiliary_games;
    s.auxiliary_search_seed_domain = aux_config.auxiliary_search_seed_domain;
    s.collector_dirichlet_epsilon = aux_config.dirichlet_epsilon;
    s.target_sharpen_exponent = aux_config.target_sharpen_exponent;
    // Travels with cum_accepted/cum_rejected/accepted_hash_chain below: those
    // totals mean nothing without the contract they were measured under.
    s.acceptance_prior_source =
        AcceptancePriorSourceName(aux_config.acceptance_prior_source);
    s.swordmaster_grant_fraction = aux_config.swordmaster_grant_fraction;
    s.swordmaster_grant_round = aux_config.swordmaster_grant_round;
    s.search_loss_coef_target = aux_search_loss_coef_target;
    s.search_loss_warmup_update = aux_search_loss_warmup;
    s.abort_grad_norm_ratio = aux_abort_ratio;
    s.next_auxiliary_episode_id = aux_next_episode_id_persist;
    s.cum_accepted = aux_cum_accepted;
    s.cum_rejected = aux_cum_rejected;
    for (int r = 0; r < 3; ++r) {
      s.cum_role_searches[r] = aux_cum_role_searches[r];
      s.cum_role_accepted[r] = aux_cum_role_accepted[r];
    }
    s.cum_granted = aux_cum_granted;
    s.cum_organic = aux_cum_organic;
    s.accepted_hash_chain = aux_hash_chain;
    return s;
  };

  std::cout << absl::StrFormat(
      "Initialized dune_ppo_train | obs=%d actions=%d hidden=%d blocks=%d "
      "rollout_transitions=%d rollout_games=%d minibatch=%d epochs=%d clip=%.3f vf_coef=%.3f "
      "ent_coef=%.4f gamma=%.3f gae_lambda=%.3f\n",
      obs_size, action_size, absl::GetFlag(FLAGS_hidden_dim),
      absl::GetFlag(FLAGS_num_blocks),
      absl::GetFlag(FLAGS_rollout_transitions),
      absl::GetFlag(FLAGS_rollout_games),
      absl::GetFlag(FLAGS_ppo_minibatch_size),
      absl::GetFlag(FLAGS_ppo_update_epochs),
      absl::GetFlag(FLAGS_ppo_clip_epsilon),
      absl::GetFlag(FLAGS_value_coef), absl::GetFlag(FLAGS_entropy_coef),
      absl::GetFlag(FLAGS_gamma), absl::GetFlag(FLAGS_gae_lambda));

  uint64_t total_games = 0;
  uint64_t total_moves = 0;
  auto training_start = std::chrono::high_resolution_clock::now();

  double base_lr = absl::GetFlag(FLAGS_learning_rate);
  bool pipeline = absl::GetFlag(FLAGS_pipeline);
  int num_threads = absl::GetFlag(FLAGS_threads);

  open_spiel::SearchLabelBuffer search_buffer;
  search_buffer.SetExpectedDimensions(obs_size, action_size);
  std::string search_label_dir = absl::GetFlag(FLAGS_search_label_dir);
  if (!search_label_dir.empty()) {
    search_buffer.LoadFromDirectory(search_label_dir);
  }

  // PF-6 / C: joint distillation. FAIL LOUDLY rather than silently inert.
  //
  // The precedent is PF-2's Part-B rule that the matched-fallback flag ON with
  // --check_strategic_state=false is a FATAL configuration error: a treatment
  // flag that is on but has nothing to act on produces an arm that LOOKS
  // treated and is not, which is exactly the ambiguity PWO-5 could not resolve
  // after the fact. So joint mode with no labels is fatal, not a warning.
  const bool pf6_joint_distill = absl::GetFlag(FLAGS_search_distill_joint);
  if (pf6_joint_distill) {
    if (search_label_dir.empty() || search_buffer.Size() == 0) {
      SpielFatalError(
          "--search_distill_joint=true requires a non-empty "
          "--search_label_dir: joint distillation with no labels would train "
          "a nominally-treated arm with no treatment.");
    }
    if (absl::GetFlag(FLAGS_search_lambda) <= 0.0) {
      SpielFatalError(
          "--search_distill_joint=true requires --search_lambda > 0.");
    }
    if (absl::GetFlag(FLAGS_online_search_collection)) {
      SpielFatalError(
          "--search_distill_joint is mutually exclusive with "
          "--online_search_collection.");
    }
    if (absl::GetFlag(FLAGS_diagnostics_only)) {
      // The diagnostics-only path deliberately calls TrainPpoUpdate with the
      // bare PPO signature -- no aux, no PWO-5 heads, and no joint batch. A
      // joint flag there would be SILENTLY IGNORED, producing a run that
      // reports itself as treated and is not. Fatal instead.
      SpielFatalError(
          "--search_distill_joint is not supported with --diagnostics_only: "
          "that path runs bare PPO and would silently ignore the flag.");
    }
    std::cout << "[PF-6] JOINT distillation ACTIVE: the teacher KL is summed "
                 "into the PPO objective under its KL throttle; the legacy "
                 "separate-step path is DISABLED for this run.\n";
  }

  // Materializes ONE distillation minibatch per update, on device, for the
  // joint path. Called ONLY when the flag is on, so with the flag off no RNG
  // is drawn here and `Sample()` is never reached -- which is what makes the
  // flag-off projection bit-for-bit rather than merely equivalent.
  //
  // It draws from the SAME seed stream as the legacy separate-step path
  // (kStreamSearchSampling, aux index 0). The two paths are mutually
  // exclusive, so there is no collision, and sharing the stream means a
  // joint-vs-separate comparison sees the same labels rather than confounding
  // the objective change with a different draw.
  auto make_pf6_joint_batch =
      [&](int update_idx) -> open_spiel::Pf6JointDistillBatch {
    open_spiel::Pf6JointDistillBatch jb;
    if (!pf6_joint_distill) return jb;  // inert: no draw, no tensors
    uint64_t s = dune_seed::DeriveSeed(master, dune_seed::kDomainTrain,
                                       update_idx, 0,
                                       dune_seed::kStreamSearchSampling);
    std::mt19937 rng = dune_seed::MakeRng32(s);
    std::vector<SearchLabel> lb =
        search_buffer.Sample(absl::GetFlag(FLAGS_search_minibatch_size), &rng);
    if (lb.empty()) return jb;
    int64_t m = static_cast<int64_t>(lb.size());
    auto cf = torch::TensorOptions().dtype(torch::kFloat32);
    auto cb = torch::TensorOptions().dtype(torch::kBool);
    torch::Tensor st = torch::empty({m, obs_size}, cf);
    torch::Tensor mk = torch::zeros({m, action_size}, cb);
    torch::Tensor tp = torch::zeros({m, action_size}, cf);
    float* sp = st.data_ptr<float>();
    bool* mp = mk.data_ptr<bool>();
    float* tpp = tp.data_ptr<float>();
    for (int64_t i = 0; i < m; ++i) {
      std::memcpy(sp + i * obs_size, lb[i].state.data(),
                  obs_size * sizeof(float));
      for (const auto& ap : lb[i].teacher_probs) {
        int64_t a = ap.first;
        if (a >= 0 && a < action_size) {
          mp[i * action_size + a] = true;
          tpp[i * action_size + a] = ap.second;
        }
      }
    }
    jb.states = st.to(device);
    jb.masks = mk.to(device);
    jb.teacher_probs = tp.to(device);
    jb.lambda = absl::GetFlag(FLAGS_search_lambda);
    return jb;
  };

  int rollout_games = absl::GetFlag(FLAGS_rollout_games);

  std::string initial_non_value_hash = "";
  if (absl::GetFlag(FLAGS_train_value_only)) {
    initial_non_value_hash = HashNonValueParameters(training_model);
    std::cout << "[INFO] Initial non-value parameters SHA256: " << initial_non_value_hash << "\n";
  }

  // Collect first rollout synchronously.
  float reward_lambda = ComputeRewardLambda(total_env_steps.load(),
                                            absl::GetFlag(FLAGS_shaping_start_env_steps),
                                            absl::GetFlag(FLAGS_shaping_decay_env_steps));
  open_spiel::CollectResult current_collect = open_spiel::CollectRollout(
      game.get(), evaluator, obs_size, &total_env_steps, num_threads, &next_episode_id,
      rollout_games, reward_lambda);
  if (absl::GetFlag(FLAGS_diagnostics_only)) {
    open_spiel::PpoUpdateStats stats =
        open_spiel::TrainPpoUpdate(training_model, *optimizer, current_collect.rollout,
                                   obs_size, action_size, device, master, start_update, anchor_model);
    stats.episode_ids_unique = current_collect.episode_ids_unique;
    AttachPrecapAbszStats(&stats, current_collect);
    AttachCanaryStats(&stats, current_collect);
    std::string diagnostics_path = absl::GetFlag(FLAGS_diagnostics_path);
    if (!diagnostics_path.empty()) {
      double val_kl = search_buffer.ComputeValidationKL(training_model, device, static_cast<float>(absl::GetFlag(FLAGS_logit_cap)));
      open_spiel::WriteDiagnostics(diagnostics_path, start_update, stats,
                                   current_collect.conflict_vp_generated,
                                   current_collect.conflict_vp_attributed,
                                   current_collect.conflict_vp_unattributed,
                                   master, run_uuid, absl::GetFlag(FLAGS_run_prefix), config_fingerprint,
                                   current_collect.raw_conflict_vp,
                                   current_collect.raw_noncombat_vp,
                                   current_collect.raw_total_vp,
                                   val_kl,
                                   absl::GetFlag(FLAGS_emit_canary_columns));
    }
    std::cout << "Diagnostics-only run complete. Exiting.\n";
    exit(0);
  }

  total_games += current_collect.games;
  total_moves += current_collect.moves;

  // Phase 18B: aux examples for the first update, from the same frozen snapshot
  // the pre-loop rollout used (sequential, main thread).
  run_aux_collection(start_update, current_collect);

  for (int update = start_update; update <= target_end_update; ++update) {
    if (absl::GetFlag(FLAGS_anneal_lr)) {
      double frac = 1.0 - static_cast<double>(update - 1) /
                              std::max(1, target_end_update);
      open_spiel::SetOptimizerLearningRate(*optimizer,
                                           std::max(1e-8, frac * base_lr));
    }

    auto wall_start = std::chrono::high_resolution_clock::now();

    // Launch background collection for the next update (pipelined).
    // The inference model is frozen during PPO training, so the background
    // collection safely uses the current policy snapshot.
    open_spiel::CollectResult next_collect;
    std::thread bg_collect_thread;
    bool have_bg = pipeline && update < target_end_update;
    if (have_bg) {
      float next_reward_lambda = ComputeRewardLambda(total_env_steps.load(),
                                                     absl::GetFlag(FLAGS_shaping_start_env_steps),
                                                     absl::GetFlag(FLAGS_shaping_decay_env_steps));
      bg_collect_thread = std::thread([&, next_reward_lambda]() {
        next_collect = open_spiel::CollectRollout(
            game.get(), evaluator, obs_size, &total_env_steps, num_threads, &next_episode_id,
            rollout_games, next_reward_lambda);
        // Phase 18B: aux for the NEXT update, from the same frozen snapshot, run
        // sequentially AFTER the rollout (design: collect PPO first, then aux).
        run_aux_collection(update + 1, next_collect);
      });
    }

    auto ppo_start = std::chrono::high_resolution_clock::now();
    const double this_search_coef =
        online_search_collection
            ? open_spiel::SearchLossCoefForUpdate(update, aux_search_loss_coef_target,
                                                  aux_search_loss_warmup)
            : 0.0;
    // --- PWO-5 section 8.6: draw this update's auxiliary batch. ----------
    //
    // ONE draw per update (aux = 0), on its own RNG stream. Reusing
    // kStreamSearchSampling is FORBIDDEN: the two samplers draw different
    // populations (41,132 rows vs 20,582) in different units (games vs rows),
    // so a shared stream would couple head and distillation row choice.
    open_spiel::Pwo5AuxBatch pwo5_batch;
    std::string pwo5_draw_digest;
    if (pwo5_aux_layout && pwo5_cfg.AnyActive()) {
      const uint64_t aux_seed = dune_seed::DeriveSeed(
          master, dune_seed::kDomainTrain, update, /*aux=*/0,
          dune_seed::kStreamAuxSampling);
      const auto draw = pwo5_store.Draw(
          aux_seed, absl::GetFlag(FLAGS_aux_games_per_update),
          absl::GetFlag(FLAGS_aux_rows_per_game),
          absl::GetFlag(FLAGS_aux_batches_per_update));
      pwo5_draw_digest = open_spiel::ComputeStringSHA256(draw.digest);
      if (update == 1) {
        // The manifest's update-1 digest was precomputed at startup. Assert
        // the REALIZED draw reproduces it, so the recorded value is a
        // measurement of this run rather than a prediction about it.
        if (pwo5_draw_digest != pwo5_update1_sampler_digest) {
          SpielFatalError(
              "PWO-5 section 8.6: the realized update-1 sampler digest " +
              pwo5_draw_digest + " does not match the precomputed digest " +
              pwo5_update1_sampler_digest +
              " recorded in the manifest. The sampler is not reproducible.");
        }
        std::cout << "[PWO-5] update-1 sampler digest " << pwo5_draw_digest
                  << " (matches the manifest)" << std::endl;
      }
      // Flatten batch -> game -> rows into one tensor set, with game_id a
      // 0-based index over the update's sampled games so the trajectory
      // weighting inside the loss is exact.
      std::vector<float> obs_flat;
      std::vector<float> fvp;
      std::vector<int64_t> tround, nact, gid;
      int64_t g_counter = 0;
      for (const auto& batch_games : draw.batches) {
        for (const auto& game_rows : batch_games) {
          for (int64_t r : game_rows) {
            const auto& row = pwo5_store.rows()[r];
            const float* o = pwo5_store.observation(r);
            obs_flat.insert(obs_flat.end(), o, o + pwo5_store.obs_size());
            fvp.push_back(row.final_vp_target);
            tround.push_back(row.terminal_round_class);
            nact.push_back(row.next_own_action);
            gid.push_back(g_counter);
          }
          ++g_counter;
        }
      }
      const int64_t nrows = static_cast<int64_t>(fvp.size());
      auto fopt = torch::TensorOptions().dtype(torch::kFloat32);
      auto iopt = torch::TensorOptions().dtype(torch::kInt64);
      pwo5_batch.obs = torch::from_blob(obs_flat.data(),
                                        {nrows, pwo5_store.obs_size()}, fopt)
                           .clone().to(device);
      pwo5_batch.final_vp =
          torch::from_blob(fvp.data(), {nrows}, fopt).clone().to(device);
      pwo5_batch.terminal_round =
          torch::from_blob(tround.data(), {nrows}, iopt).clone().to(device);
      pwo5_batch.next_action =
          torch::from_blob(nact.data(), {nrows}, iopt).clone().to(device);
      pwo5_batch.game_id =
          torch::from_blob(gid.data(), {nrows}, iopt).clone().to(device);
      pwo5_batch.num_games = g_counter;
      // Section 8.2's registered trajectory denominators, computed HERE from
      // the draw itself. Every drawn game contributes `aux_rows_per_game` rows,
      // and final_vp / terminal_round have a target on every row, so both are
      // the full drawn-game count. next_own_action can legitimately be lower:
      // a row whose decision is the game's LAST label row has no later action
      // (target -1), and a game all of whose sampled rows were terminal would
      // contribute nothing and drop out of the denominator entirely. That is
      // measured rather than assumed.
      {
        std::vector<char> na_present(static_cast<size_t>(g_counter), 0);
        for (std::size_t i = 0; i < nact.size(); ++i) {
          if (nact[i] >= 0) na_present[static_cast<size_t>(gid[i])] = 1;
        }
        int64_t na_games = 0;
        for (char c : na_present) na_games += c ? 1 : 0;
        pwo5_batch.sampled_games = g_counter;
        pwo5_batch.final_vp_games = g_counter;
        pwo5_batch.terminal_round_games = g_counter;
        pwo5_batch.next_own_action_games = na_games;
      }
      pwo5_batch.valid = true;
    }

    open_spiel::PpoUpdateStats stats =
        open_spiel::TrainPpoUpdate(training_model, *optimizer, current_collect.rollout,
                                   obs_size, action_size, device, master, update, anchor_model,
                                   current_collect.aux_examples, this_search_coef,
                                   online_search_collection ? aux_abort_ratio : 0.0,
                                   pwo5_batch, pwo5_cfg,
                                   make_pf6_joint_batch(update));
    stats.episode_ids_unique = current_collect.episode_ids_unique;
    AttachPrecapAbszStats(&stats, current_collect);
    AttachCanaryStats(&stats, current_collect);

    // Phase 18B clean abort at the latest valid checkpoint (plan §18C): a
    // per-update aux/PPO grad-norm ratio over threshold stops the run; this
    // update's mutated weights are discarded (never checkpointed) and we exit
    // nonzero with a distinct message.
    if (stats.aux_ratio_abort) {
      if (have_bg && bg_collect_thread.joinable()) bg_collect_thread.join();
      // Record the aborting update before exiting: its ratio is the one row a
      // post-mortem needs, and stderr is not a durable diagnostics channel.
      std::string abort_diagnostics_path = absl::GetFlag(FLAGS_diagnostics_path);
      if (!abort_diagnostics_path.empty()) {
        open_spiel::WriteDiagnostics(
            abort_diagnostics_path, update, stats,
            current_collect.conflict_vp_generated,
            current_collect.conflict_vp_attributed,
            current_collect.conflict_vp_unattributed, master, run_uuid,
            absl::GetFlag(FLAGS_run_prefix), config_fingerprint,
            current_collect.raw_conflict_vp, current_collect.raw_noncombat_vp,
            current_collect.raw_total_vp,
            search_buffer.ComputeValidationKL(
                training_model, device,
                static_cast<float>(absl::GetFlag(FLAGS_logit_cap))),
            absl::GetFlag(FLAGS_emit_canary_columns));
      }
      std::cerr << absl::StrFormat(
          "ABORT[18B grad-ratio]: update %d aux/PPO norm ratio %.4f > %.4f "
          "(aux_norm=%.4f ppo_norm=%.4f). Keeping last valid checkpoint; this "
          "update is NOT saved.\n",
          update, stats.aux_ppo_norm_ratio, aux_abort_ratio,
          stats.aux_grad_norm_mean, stats.ppo_grad_norm_mean);
      std::exit(3);
    }
    // Fold this consumed update's collection into the persisted cumulative
    // counters / hash chain / resume cursor (main thread only).
    account_aux_consumed(current_collect);

    if (absl::GetFlag(FLAGS_train_value_only)) {
      std::string current_non_value_hash = HashNonValueParameters(training_model);
      if (current_non_value_hash != initial_non_value_hash) {
        SpielFatalError(absl::StrFormat("Policy/trunk parameters changed during value-only training! Expected: %s, Got: %s",
                                        initial_non_value_hash, current_non_value_hash));
      }
    }

    double ppo_elapsed = std::chrono::duration<double>(
        std::chrono::high_resolution_clock::now() - ppo_start).count();

    // Log PPO updates to diagnostics CSV
    std::string diagnostics_path = absl::GetFlag(FLAGS_diagnostics_path);
    if (!diagnostics_path.empty()) {
      double val_kl = search_buffer.ComputeValidationKL(training_model, device, static_cast<float>(absl::GetFlag(FLAGS_logit_cap)));
      open_spiel::WriteDiagnostics(diagnostics_path, update, stats,
                                   current_collect.conflict_vp_generated,
                                   current_collect.conflict_vp_attributed,
                                   current_collect.conflict_vp_unattributed,
                                   master, run_uuid, absl::GetFlag(FLAGS_run_prefix), config_fingerprint,
                                   current_collect.raw_conflict_vp,
                                   current_collect.raw_noncombat_vp,
                                   current_collect.raw_total_vp,
                                   val_kl,
                                   absl::GetFlag(FLAGS_emit_canary_columns));
      // PWO-5 amendment 1 ruling 6: the head telemetry sidecar. Path DERIVED
      // from --diagnostics_path (no new flag), gated by the already-registered
      // --emit_canary_columns.
      if (absl::GetFlag(FLAGS_emit_canary_columns)) {
        open_spiel::WritePwo5HeadTelemetry(
            diagnostics_path, update, stats, run_uuid, config_fingerprint,
            static_cast<uint64_t>(absl::GetFlag(FLAGS_seed)));
      }
    }

    // --- Search auxiliary distillation steps ---
    double search_kl_sum = 0.0;
    double search_grad_sum = 0.0;
    double search_lambda = absl::GetFlag(FLAGS_search_lambda);
    int search_minibatches_per_update = absl::GetFlag(FLAGS_search_minibatches_per_update);
    int search_minibatch_size = absl::GetFlag(FLAGS_search_minibatch_size);
    float logit_cap = static_cast<float>(absl::GetFlag(FLAGS_logit_cap));

    // Per-update refresh, for the streaming case where a producer drops new
    // .bin packs into the directory while training runs.
    //
    // A ROLE-AWARE dataset is CLOSED at load time -- its membership is exactly
    // the manifest's list, and a rescan would reload those files without their
    // roles, which is the precise defect ruling 5 removes. So the refresh is
    // SKIPPED rather than attempted; LoadNewFiles itself still refuses a
    // role-aware directory, which is what makes this a guard and not a
    // convention.
    if (!search_label_dir.empty() && !search_buffer.role_aware()) {
      search_buffer.LoadNewFiles(search_label_dir);
    }

    // PF-6 / C: the legacy SEPARATE-STEP path. Disabled when joint mode is on
    // -- the two are alternatives, not layers. Running both would apply the
    // teacher twice per update and confound the very comparison C exists to
    // make. With --search_distill_joint=false this condition is exactly what
    // it was before the flag existed.
    if (!pf6_joint_distill && search_lambda > 0.0 &&
        search_buffer.Size() > 0) {
      auto cpu_float = torch::TensorOptions().dtype(torch::kFloat32);
      auto cpu_bool = torch::TensorOptions().dtype(torch::kBool);

      for (int aux = 0; aux < search_minibatches_per_update; ++aux) {
        uint64_t aux_seed = dune_seed::DeriveSeed(master, dune_seed::kDomainTrain, update, aux, dune_seed::kStreamSearchSampling);
        std::mt19937 aux_rng = dune_seed::MakeRng32(aux_seed);
        std::vector<SearchLabel> search_batch = search_buffer.Sample(search_minibatch_size, &aux_rng);
        if (search_batch.empty()) continue;

        int64_t sb_size = search_batch.size();
        torch::Tensor search_states_cpu = torch::empty({sb_size, obs_size}, cpu_float);
        torch::Tensor search_masks_cpu = torch::zeros({sb_size, action_size}, cpu_bool);
        torch::Tensor search_teacher_probs_cpu = torch::zeros({sb_size, action_size}, cpu_float);

        float* states_ptr = search_states_cpu.data_ptr<float>();
        bool* masks_ptr = search_masks_cpu.data_ptr<bool>();
        float* teacher_ptr = search_teacher_probs_cpu.data_ptr<float>();

        for (int64_t i = 0; i < sb_size; ++i) {
          const auto& label = search_batch[i];
          std::memcpy(states_ptr + i * obs_size, label.state.data(), obs_size * sizeof(float));
          for (const auto& ap : label.teacher_probs) {
            int64_t action_id = ap.first;
            float prob = ap.second;
            if (action_id >= 0 && action_id < action_size) {
              masks_ptr[i * action_size + action_id] = true;
              teacher_ptr[i * action_size + action_id] = prob;
            }
          }
        }

        torch::Tensor search_states = search_states_cpu.to(device);
        torch::Tensor search_masks = search_masks_cpu.to(device);
        torch::Tensor search_teacher_probs = search_teacher_probs_cpu.to(device);

        optimizer->zero_grad();

        torch::Tensor mean_kl;
        auto compute_distill_loss = [&]() {
          auto outputs = training_model->forward(search_states);
          torch::Tensor logits = CenterAndCapLogitsTensor(outputs.logits, search_masks, logit_cap);
          torch::Tensor masked_logits = logits.masked_fill(search_masks.logical_not(), -1e9f);
          torch::Tensor log_probs = torch::log_softmax(masked_logits, -1);

          torch::Tensor log_teacher = torch::log(search_teacher_probs.clamp_min(1e-12f));
          torch::Tensor kl_loss = search_teacher_probs * (log_teacher - log_probs);
          mean_kl = kl_loss.sum(-1).mean();
        };

        if (device.is_cuda() && absl::GetFlag(FLAGS_train_amp)) {
          AutocastGuard autocast_guard(c10::DeviceType::CUDA, true);
          compute_distill_loss();
        } else {
          compute_distill_loss();
        }

        torch::Tensor search_loss = search_lambda * mean_kl;
        search_loss.backward();

        double search_grad_norm = torch::nn::utils::clip_grad_norm_(
            training_model->parameters(), std::numeric_limits<double>::infinity());
        if (!std::isfinite(search_grad_norm) || !std::isfinite(mean_kl.item<double>())) {
          std::cerr << "Warning: Non-finite search distillation gradient (" << search_grad_norm
                    << ") or loss. Skipping optimizer step.\n";
          optimizer->zero_grad();
          continue;
        }
        torch::nn::utils::clip_grad_norm_(
            training_model->parameters(), absl::GetFlag(FLAGS_grad_clip_norm));
        optimizer->step();

        search_kl_sum += mean_kl.item<double>();
        search_grad_sum += search_grad_norm;
      }
    }

    // Join background collection before syncing models.
    if (have_bg) {
      bg_collect_thread.join();
      total_games += next_collect.games;
      total_moves += next_collect.moves;
    }

    open_spiel::SyncModels(training_model, inference_model, &sync_mutex);

    auto wall_end = std::chrono::high_resolution_clock::now();

    double collect_elapsed = current_collect.elapsed_seconds;
    double wall_elapsed =
        std::chrono::duration<double>(wall_end - wall_start).count();
    double sps = collect_elapsed > 0.0
                     ? current_collect.rollout.size() / collect_elapsed
                     : 0.0;

    std::cout << absl::StrFormat(
        "Update %5d/%d | Transitions: %6zu | Games: %llu | "
        "Collect: %.2fs (%.0f t/s) | PPO: %.2fs | Wall: %.2fs | "
        "PolicyLoss: %.6f | ValueLoss: %.6f | Entropy: %.4f | "
        "ApproxKL: %.6f | ClipFrac: %.3f | ExplVar: %.3f%s\n",
        update, target_end_update, current_collect.rollout.size(),
        static_cast<unsigned long long>(total_games),
        collect_elapsed, sps, ppo_elapsed, wall_elapsed,
        stats.policy_loss, stats.value_loss, stats.entropy, stats.approx_kl,
        stats.clip_fraction, stats.explained_variance,
        stats.early_stopped ? " | early-stop" : "");

    // Phase 18B combined-optimization + collector diagnostics.
    if (online_search_collection) {
      const open_spiel::OnlineSearchCollectionStats& cst = current_collect.aux_stats;
      auto acc = [](const open_spiel::PerRoleSearchStats& r) {
        return r.searches > 0 ? static_cast<double>(r.accepted) / r.searches : 0.0;
      };
      auto mkl = [](const open_spiel::PerRoleSearchStats& r) {
        return r.searches > 0 ? r.sum_kl / r.searches : 0.0;
      };
      std::cout << absl::StrFormat(
          "  [18B Aux] used=%d coef=%.4f ce=%.4f vmse=%.4f | aux_norm=%.4f "
          "ppo_norm=%.4f ratio=%.3f\n",
          stats.aux_examples_used, this_search_coef, stats.aux_ce,
          stats.aux_value_mse, stats.aux_grad_norm_mean, stats.ppo_grad_norm_mean,
          stats.aux_ppo_norm_ratio);
      std::cout << absl::StrFormat(
          "  [18B Collector] accepted=%d rejected=%d wall=%.1fs | acc/KL "
          "primary %.2f/%.2f cont %.2f/%.2f purch %.2f/%.2f | SM granted=%d organic=%d\n",
          cst.accepted_targets, cst.rejected_incomplete, cst.collection_wall_time_s,
          acc(cst.by_role[0]), mkl(cst.by_role[0]), acc(cst.by_role[1]),
          mkl(cst.by_role[1]), acc(cst.by_role[2]), mkl(cst.by_role[2]),
          static_cast<int>(cst.swordmaster_granted_games),
          static_cast<int>(cst.swordmaster_organic_games));
    }

    std::cout << absl::StrFormat(
        "  [Conflict VP Stats] Generated: %.2f | Attributed: %.2f | Unattributed: %.2f\n",
        current_collect.conflict_vp_generated,
        current_collect.conflict_vp_attributed,
        current_collect.conflict_vp_unattributed);

    {
      uint64_t total_acquisitions = 0;
      for (int i = 0; i < 4; ++i) total_acquisitions += current_collect.sm_acquisitions_by_seat[i];
      if (total_acquisitions > 0) {
        std::cout << absl::StrFormat(
            "  [Natural SM Acquisitions] Total: %d | Seat P0: %d | P1: %d | P2: %d | P3: %d\n",
            total_acquisitions, current_collect.sm_acquisitions_by_seat[0],
            current_collect.sm_acquisitions_by_seat[1], current_collect.sm_acquisitions_by_seat[2],
            current_collect.sm_acquisitions_by_seat[3]);
        std::cout << absl::StrFormat(
            "  [SM Acquisitions by Round] R1: %d | R2: %d | R3: %d | R4: %d | R5: %d | R6-10: %d\n",
            current_collect.sm_acquisitions_by_round[1], current_collect.sm_acquisitions_by_round[2],
            current_collect.sm_acquisitions_by_round[3], current_collect.sm_acquisitions_by_round[4],
            current_collect.sm_acquisitions_by_round[5],
            current_collect.sm_acquisitions_by_round[6] + current_collect.sm_acquisitions_by_round[7] +
            current_collect.sm_acquisitions_by_round[8] + current_collect.sm_acquisitions_by_round[9] +
            current_collect.sm_acquisitions_by_round[10]);
        std::cout << absl::StrFormat(
            "  [SM Acquisitions by Leader] Leto (4): %d | Rabban (6): %d | Other Leaders: %d\n",
            current_collect.sm_acquisitions_by_leader[4], current_collect.sm_acquisitions_by_leader[6],
            total_acquisitions - current_collect.sm_acquisitions_by_leader[4] - current_collect.sm_acquisitions_by_leader[6]);
      }
    }

    // Telemetry for the SEPARATE-STEP path only. In joint mode those
    // accumulators are never written, so printing them would report a
    // SearchKL of 0.000 for a run that is distilling -- a false readout, not
    // merely a missing one. Suppressed rather than printed misleadingly.
    if (!pf6_joint_distill && search_lambda > 0.0 &&
        search_buffer.Size() > 0) {
      double avg_search_kl = (search_minibatches_per_update > 0) ? (search_kl_sum / search_minibatches_per_update) : 0.0;
      double avg_search_grad = (search_minibatches_per_update > 0) ? (search_grad_sum / search_minibatches_per_update) : 0.0;
      double avg_ppo_grad = (stats.grad_norm_count > 0) ? (stats.grad_norm_sum / stats.grad_norm_count) : 0.0;
      double ratio = (avg_ppo_grad > 0.0) ? (avg_search_grad / avg_ppo_grad) : 0.0;
      std::cout << absl::StrFormat(
          "SearchKL: %.3f | SearchGrad: %.3f | PPOGrad: %.3f | Ratio: %.2f | EV: %.3f\n",
          avg_search_kl, avg_search_grad, avg_ppo_grad, ratio, stats.explained_variance);
    }

    int checkpoint_interval = absl::GetFlag(FLAGS_checkpoint_interval);
    int training_step = update - start_update + 1;
    bool is_pilot_update = (training_step == 10 || training_step == 25 || training_step == 50);
    if (is_pilot_update || (checkpoint_interval > 0 && update % checkpoint_interval == 0)) {
      std::string prefix = absl::GetFlag(FLAGS_run_prefix);
      std::string model_path =
          absl::StrCat(prefix, "_model_update_", update, ".pt");
      std::string optim_path =
          absl::StrCat(prefix, "_optimizer_update_", update, ".pt");
      open_spiel::OnlineCollectionState aux_ckpt = build_aux_state();
      open_spiel::SaveCheckpoint(training_model, *optimizer, model_path,
                                 optim_path, update, target_end_update,
                                 total_env_steps.load(), next_episode_id.load(),
                                 master, absl::GetFlag(FLAGS_seed_scheme_version),
                                 config_fingerprint, search_label_fingerprint,
                                 run_uuid,
                                 online_search_collection ? &aux_ckpt : nullptr);
    }

    // Advance to next rollout.
    if (have_bg) {
      current_collect = std::move(next_collect);
    } else if (update < target_end_update) {
      float next_reward_lambda = ComputeRewardLambda(total_env_steps.load(),
                                                     absl::GetFlag(FLAGS_shaping_start_env_steps),
                                                     absl::GetFlag(FLAGS_shaping_decay_env_steps));
      current_collect = open_spiel::CollectRollout(
          game.get(), evaluator, obs_size, &total_env_steps, num_threads, &next_episode_id,
          rollout_games, next_reward_lambda);
      total_games += current_collect.games;
      total_moves += current_collect.moves;
      // Phase 18B (non-pipelined path): aux for the next update, same snapshot.
      run_aux_collection(update + 1, current_collect);
    }
  }

  auto training_end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = training_end - training_start;
  auto eval_stats = evaluator->GetStats();
  std::cout << absl::StrFormat(
      "\n=== PPO Training Complete ===\nElapsed: %.2fs\nEnv Steps: %llu\n"
      "Games: %llu\nMoves: %llu\nEvaluator Requests: %llu | Batches: %llu | "
      "Avg Batch: %.2f | Max Batch: %llu\n",
      elapsed.count(),
      static_cast<unsigned long long>(total_env_steps.load()),
      static_cast<unsigned long long>(total_games),
      static_cast<unsigned long long>(total_moves),
      static_cast<unsigned long long>(eval_stats.requests),
      static_cast<unsigned long long>(eval_stats.batches),
      eval_stats.avg_batch_size,
      static_cast<unsigned long long>(eval_stats.max_batch_size));

  if (absl::GetFlag(FLAGS_save_final_checkpoint)) {
    open_spiel::OnlineCollectionState aux_final = build_aux_state();
    open_spiel::SaveCheckpoint(training_model, *optimizer,
                               absl::GetFlag(FLAGS_model_checkpoint),
                               absl::GetFlag(FLAGS_optim_checkpoint),
                               target_end_update, target_end_update,
                               total_env_steps.load(), next_episode_id.load(),
                               master, absl::GetFlag(FLAGS_seed_scheme_version),
                               config_fingerprint, search_label_fingerprint,
                               run_uuid,
                               online_search_collection ? &aux_final : nullptr);
  }
  return 0;
#endif
}
