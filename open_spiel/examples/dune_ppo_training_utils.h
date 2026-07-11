#ifndef OPEN_SPIEL_EXAMPLES_DUNE_PPO_TRAINING_UTILS_H_
#define OPEN_SPIEL_EXAMPLES_DUNE_PPO_TRAINING_UTILS_H_

#include <vector>
#include <string>
#include <memory>
#include <cstring>
#include <cstdint>



#ifdef OPEN_SPIEL_BUILD_WITH_LIBTORCH
#include <torch/torch.h>
#include "dune_network.h"
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
  std::vector<double> epoch_kls;
  std::string rollout_hash;
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

PpoUpdateStats TrainPpoUpdate(
    std::shared_ptr<SharedDunePolicyValueNetImpl> model,
    torch::optim::AdamW& optimizer, std::vector<PpoTransition>& batch,
    int64_t obs_size, int64_t action_dim, torch::Device device,
    uint64_t master, int global_update);

void WriteDiagnostics(const std::string& filepath, int update, const PpoUpdateStats& stats,
                      double conflict_vp_generated, double conflict_vp_attributed, double conflict_vp_unattributed,
                      uint64_t seed, const std::string& run_uuid, const std::string& run_prefix, const std::string& config_fingerprint,
                      double raw_conflict_vp, double raw_noncombat_vp, double raw_total_vp,
                      double validation_kl = -1.0);
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

} // namespace open_spiel

#endif // OPEN_SPIEL_EXAMPLES_DUNE_PPO_TRAINING_UTILS_H_
