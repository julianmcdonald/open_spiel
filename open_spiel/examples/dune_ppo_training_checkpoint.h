#ifndef OPEN_SPIEL_EXAMPLES_DUNE_PPO_TRAINING_CHECKPOINT_H_
#define OPEN_SPIEL_EXAMPLES_DUNE_PPO_TRAINING_CHECKPOINT_H_

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "open_spiel/abseil-cpp/absl/strings/str_cat.h"
#include "open_spiel/abseil-cpp/absl/strings/str_format.h"
#include "open_spiel/spiel.h"
#include "open_spiel/utils/json.h"
#include "dune_specimen_conversion.h"
#include "dune_search_routing.h"

#ifdef OPEN_SPIEL_BUILD_WITH_LIBTORCH
#include <torch/torch.h>
#include "dune_network.h"
#endif

namespace open_spiel {

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
  bool precision_fields_present = false;
  bool rollout_amp = true;
  bool allow_tf32 = true;
  bool legacy_precision_migration = false;
  std::string precision_migration_source;
  bool reward_transition = false;
  std::string reward_transition_source_fingerprint;
};

// Fingerprint the exact pre-precision JSON object, or a copy extended with the
// two precision controls. Keeping the former explicit prevents compatibility
// migration from depending on a mutable "legacy" approximation.
std::string ComputePrePrecisionConfigFingerprint(
    const json::Object& pre_precision_config);
std::string ComputePrecisionConfigFingerprint(
    const json::Object& pre_precision_config, bool rollout_amp,
    bool allow_tf32);

// Every newly written checkpoint/bootstrap manifest carries these fields.
void WritePpoPrecisionManifestFields(json::Object& manifest_obj,
                                     bool rollout_amp, bool allow_tf32);

struct PpoPrecisionManifestCompatibility {
  bool fields_present = false;
  bool rollout_amp = true;
  bool allow_tf32 = true;
  bool legacy_precision_migration = false;
  std::string migration_source;
  bool reward_transition = false;
  std::string reward_transition_source_fingerprint;
};

// Fail-closed precision/fingerprint compatibility gate. A field-absent
// manifest may migrate only under default true/true precision and an exact
// pre-precision (or explicitly enabled older-legacy) fingerprint.
bool ValidatePpoPrecisionManifestCompatibility(
    const json::Object& manifest_obj,
    const std::string& stored_config_fingerprint,
    const std::string& current_config_fingerprint,
    const std::string& current_pre_precision_config_fingerprint,
    const std::string& current_legacy_config_fingerprint,
    bool current_rollout_amp, bool current_allow_tf32,
    PpoPrecisionManifestCompatibility* out, std::string* error,
    const std::string& permitted_transition_source_fingerprint = "");

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
  // 0=primary, 1=continuation, 2=purchase, 3=leader_selection. Leader was added
  // when it became a searched, labelled decision; an older manifest carries
  // three entries and the reader leaves the leader slot at zero, so resuming a
  // pre-Leader run is unaffected.
  static constexpr int kNumSearchRoles = 4;
  int64_t cum_role_searches[kNumSearchRoles] = {0, 0, 0, 0};
  int64_t cum_role_accepted[kNumSearchRoles] = {0, 0, 0, 0};
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
                              const std::string& current_pre_precision_config_fingerprint,
                              bool current_rollout_amp,
                              bool current_allow_tf32,
                              const std::string& current_search_label_fingerprint,
                              int current_hidden_dim,
                              int current_num_blocks,
                              CheckpointManifest& out_manifest,
                              std::string& error_msg,
                              const std::string& current_legacy_config_fingerprint = "",
                              const std::string& permitted_transition_source_fingerprint = "");

// Reads the manifest's "deck_pool_semantics" field (defaulting to "legacy" when
// absent), validates both manifest and flag semantics, and calls SpielFatalError
// when they differ (unless allow_transition is true at source_update). Returns
// the validated semantics string.
std::string ValidateResumeDeckPoolSemanticsOrDie(
    const json::Object& manifest_obj,
    const std::string& flag_semantics,
    const std::string& context_label = "resume",
    bool allow_transition = false,
    int64_t source_update = -1);

// Verifies that manifest_path exists, is valid JSON, and that the referenced
// model_path (and optional optim_path) match the file size and SHA256 recorded
// in the manifest. Returns true if and only if the bundle is complete and intact.
bool CheckBundleIntegrity(const std::string& manifest_path,
                          const std::string& model_path,
                          const std::string& optim_path = "",
                          std::string* error_msg = nullptr);

// Checks if publication was interrupted. If .bak files exist:
// - If the active bundle is complete and verified, cleans up stale .bak files.
// - If the active bundle is incomplete/mixed and the backup bundle is intact,
//   restores the active bundle from .bak and cleans up any .tmp files.
// - Checks all filesystem error codes and throws SpielFatalError if restoration fails.
void RecoverInterruptedBundleIfNeeded(const std::string& model_path,
                                      const std::string& optim_path,
                                      const std::string& manifest_path);

struct DualCheckpointManifest {
  int schema_version = 2;
  std::string checkpoint_uuid;
  std::string architecture;
  bool separate_actor_critic = false;
  std::string value_head_status;
  std::string evaluation_role;
  std::string warning;
  int global_update = 0;
  int target_end_update = 0;
  uint64_t total_env_steps = 0;
  uint64_t next_episode_id = 0;
  uint64_t base_seed = 0;
  int seed_scheme_version = 0;
  std::string config_fingerprint;
  std::string search_label_fingerprint;
  std::string run_uuid;

  std::string actor_model_filename;
  size_t actor_model_file_size = 0;
  std::string actor_model_sha256;
  std::string actor_optimizer_filename;
  size_t actor_optimizer_file_size = 0;
  std::string actor_optimizer_sha256;

  std::string critic_model_filename;
  size_t critic_model_file_size = 0;
  std::string critic_model_sha256;
  std::string critic_optimizer_filename;
  size_t critic_optimizer_file_size = 0;
  std::string critic_optimizer_sha256;

  int hidden_dim = 2048;
  int num_blocks = 8;
  bool rollout_amp = false;
  bool allow_tf32 = false;
};

bool ParseAndValidateDualManifest(
    const std::string& manifest_path,
    std::string& actor_model_path,
    std::string& actor_optim_path,
    std::string& critic_model_path,
    std::string& critic_optim_path,
    uint64_t current_base_seed,
    int current_target_end_update,
    int current_seed_scheme_version,
    const std::string& current_config_fingerprint,
    bool current_rollout_amp,
    bool current_allow_tf32,
    int current_hidden_dim,
    int current_num_blocks,
    DualCheckpointManifest& out_manifest,
    std::string& error_msg);

inline std::string GenerateUUID() {
  std::random_device rd;
  std::mt19937_64 generator(rd());
  uint64_t r1 = generator();
  uint64_t r2 = generator();
  uint32_t a = (r1 >> 32);
  uint16_t b = (r1 >> 16) & 0xFFFF;
  uint16_t c = (r1 & 0x0FFF) | 0x4000;
  uint16_t d = ((r2 >> 48) & 0x3FFF) | 0x8000;
  uint64_t e = r2 & 0xFFFFFFFFFFFFULL;
  return absl::StrFormat("%08x-%04x-%04x-%04x-%012llx", a, b, c, d, e);
}

void SaveDualCheckpoint(
    std::shared_ptr<SharedDunePolicyValueNetImpl> actor_model,
    torch::optim::AdamW& actor_optimizer,
    std::shared_ptr<SharedDunePolicyValueNetImpl> critic_model,
    torch::optim::AdamW& critic_optimizer,
    const std::string& actor_model_path,
    const std::string& actor_optim_path,
    const std::string& critic_model_path,
    const std::string& critic_optim_path,
    int global_update,
    int target_end_update,
    uint64_t total_env_steps,
    uint64_t next_episode_id,
    uint64_t base_seed,
    int seed_scheme_version,
    const std::string& config_fingerprint,
    const std::string& search_label_fingerprint,
    const std::string& run_uuid,
    const std::string& input_model_to_protect = "",
    const std::string& input_optim_to_protect = "");

struct CheckpointMetadataSpec {
  std::string market_appendix_mode = "none";
  dune_imperium::DeckPoolSemantics deck_pool_semantics = dune_imperium::DeckPoolSemantics::kLegacy;
  int64_t observation_dim = 5580;
  int64_t hidden_dim = 2048;
  int64_t num_blocks = 8;
  std::string feature_schema_label;
  std::string feature_schema_sha256;
  bool enable_semantic_scorer = false;
  std::string semantic_descriptor_schema;
  bool enable_zone_encoder = false;
};

void WriteCheckpointMetadataFile(const std::string& json_path,
                                 const CheckpointMetadataSpec& spec);

dune_imperium::DeckPoolSemantics ReadCheckpointDeckPoolSemantics(
    const std::string& checkpoint_path);

dune_imperium::MarketAppendixMode ReadCheckpointInputMode(
    const std::string& checkpoint_path, int input_dim);

bool DetectCheckpointModelDimensions(
    const std::string& model_path, int* hidden_dim, int* num_blocks,
    int* input_dim = nullptr, std::string* market_mode = nullptr,
    dune_imperium::DeckPoolSemantics* deck_pool_semantics = nullptr,
    std::string* semantic_descriptor_schema = nullptr,
    bool* slot_indexed_out_layer_f = nullptr);

bool LoadModelCheckpointMigrating(
    std::shared_ptr<SharedDunePolicyValueNetImpl> model,
    const std::string& path, torch::Device device,
    bool allow_skip_semantic_scorer = false);

void LoadModelCheckpoint(
    std::shared_ptr<SharedDunePolicyValueNetImpl> model,
    const std::string& path, torch::Device device,
    bool allow_skip_semantic_scorer = false);

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

// Family Atomics shaping helper:
// Subtracts family_atomics_penalty * reward_lambda when action == kActionFamilyAtomics
// unless is_own_reveal_turn is true AND remaining_persuasion > 0.
float ApplyFamilyAtomicsShaping(float reward, Action action,
                                bool is_own_reveal_turn, int remaining_persuasion,
                                double family_atomics_penalty,
                                float reward_lambda);

// Returns true iff action is a plot intrigue card play action (IDs 1600-1699).
inline bool IsPlotIntrigueAction(Action action) {
  return action >= dune_imperium::kActionPlayIntriguePlotCard0 &&
         action < dune_imperium::kActionPlayIntriguePlotCard0 + dune_imperium::kMaxIntrigueCards;
}

// Plot intrigue shaping helper:
// Subtracts plot_intrigue_penalty * reward_lambda when action is a plot intrigue play action
// unless intrigue_hand_size_before_play >= exemption_threshold (default 3).
float ApplyPlotIntrigueShaping(float reward, Action action,
                               int intrigue_hand_size_before_play,
                               int exemption_threshold,
                               double plot_intrigue_penalty,
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

// Computes terminal returns for a completed game state, applying the selected
// terminal reward mode ("placement" or "first_place") and the optional
// round-7 speed bonus (only applied if actual rounds played <= 7, and strictly
// to the sole winner).
std::vector<double> ComputeTerminalReturns(
    const State& state,
    const std::string& reward_mode,
    double round7_speed_bonus,
    double win_speed_bonus_per_round = 0.0);

// Formats double with round-trip precision (17 significant decimal digits)
inline double ResolveOnlineTeacherWinSpeedEps(
    double explicit_teacher_eps,
    double ppo_win_speed_bonus_per_round) {
  if (explicit_teacher_eps >= 0.0) {
    return explicit_teacher_eps;
  }
  if (ppo_win_speed_bonus_per_round > 0.0) {
    return std::min(0.05, ppo_win_speed_bonus_per_round);
  }
  return 0.0;
}

inline std::string FormatDoubleRoundTrip(double v) {
  if (std::isnan(v)) return "\"nan\"";
  if (std::isinf(v)) return v > 0 ? "\"inf\"" : "\"-inf\"";
  return absl::StrFormat("%.17g", v);
}

// Verification sidecar serializer with IEEE-754 round-trip precision and epoch KLs.
inline std::string SerializeVerificationSidecarRecord(
    int64_t update,
    int64_t transitions,
    const std::string& semantic_digest,
    const std::string& rollout_hash,
    double policy_kl_before,
    const std::vector<double>& epoch_kls,
    int64_t measured_transitions,
    double fraction_critic_near_1,
    int minibatches,
    double policy_loss,
    double value_loss,
    double explained_variance,
    bool early_stopped,
    int64_t kl_early_stop_epoch) {
  std::string out = "{";
  absl::StrAppend(&out, "\"update\": ", update, ", ");
  absl::StrAppend(&out, "\"transitions\": ", transitions, ", ");
  absl::StrAppend(&out, "\"semantic_candidate_digest\": \"", semantic_digest, "\", ");
  absl::StrAppend(&out, "\"rollout_hash\": \"", rollout_hash, "\", ");
  absl::StrAppend(&out, "\"policy_kl_before\": ", FormatDoubleRoundTrip(policy_kl_before), ", ");
  absl::StrAppend(&out, "\"epoch_kls\": [");
  for (size_t i = 0; i < epoch_kls.size(); ++i) {
    if (i > 0) absl::StrAppend(&out, ", ");
    absl::StrAppend(&out, FormatDoubleRoundTrip(epoch_kls[i]));
  }
  absl::StrAppend(&out, "], ");
  absl::StrAppend(&out, "\"measured_transitions\": ", measured_transitions, ", ");
  absl::StrAppend(&out, "\"fraction_critic_near_1\": ", FormatDoubleRoundTrip(fraction_critic_near_1), ", ");
  absl::StrAppend(&out, "\"minibatches\": ", minibatches, ", ");
  absl::StrAppend(&out, "\"policy_loss\": ", FormatDoubleRoundTrip(policy_loss), ", ");
  absl::StrAppend(&out, "\"value_loss\": ", FormatDoubleRoundTrip(value_loss), ", ");
  absl::StrAppend(&out, "\"explained_variance\": ", FormatDoubleRoundTrip(explained_variance), ", ");
  absl::StrAppend(&out, "\"early_stopped\": ", early_stopped ? "true" : "false", ", ");
  absl::StrAppend(&out, "\"kl_early_stop_epoch\": ", kl_early_stop_epoch);
  absl::StrAppend(&out, "}");
  return out;
}

} // namespace open_spiel

#endif  // OPEN_SPIEL_EXAMPLES_DUNE_PPO_TRAINING_CHECKPOINT_H_
