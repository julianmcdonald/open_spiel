#ifndef OPEN_SPIEL_EXAMPLES_DUNE_SURGERY_12BLOCK_INTERNAL_H_
#define OPEN_SPIEL_EXAMPLES_DUNE_SURGERY_12BLOCK_INTERNAL_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

#include <torch/torch.h>

#include "open_spiel/games/dune_imperium/dune_imperium_common.h"
#include "open_spiel/utils/json.h"

#include "dune_network.h"

namespace open_spiel::dune_surgery_12block {

struct ManifestMetadata {
  int64_t observation_dim = open_spiel::dune_imperium::kFullPublicInformationStateSize;
  open_spiel::dune_imperium::MarketAppendixMode market_appendix_mode =
      open_spiel::dune_imperium::MarketAppendixMode::kNone;
  std::string market_appendix_mode_str = "";
  open_spiel::dune_imperium::DeckPoolSemantics deck_pool_semantics =
      open_spiel::dune_imperium::DeckPoolSemantics::kLegacy;
  std::string deck_pool_semantics_str = "";
  std::string semantic_descriptor_schema = "semantic_action_v3";
  int64_t num_blocks = 12;
  int64_t global_update = 31400;
  std::string feature_schema_sha256 = "";
  std::string feature_schema_label = "";
  bool rollout_amp = true;
  bool allow_tf32 = true;
  bool train_amp = true;
  double specimen_exchange_penalty = 0.5;
  int64_t rollout_games = 256;
  std::string training_opponent_pool = "";
  double pool_game_fraction = 0.0;
  int64_t pool_opponents_per_game = 0;
  double family_atomics_penalty = 0.0;
  double plot_intrigue_penalty = 0.0;
  int64_t plot_intrigue_exemption_threshold = 0;
  double ema_magnet_kl_coef = 0.0;
  std::string config_fingerprint = "";
};

struct LiveSurgeryResult {
  std::shared_ptr<open_spiel::SharedDunePolicyValueNetImpl> model_src;
  std::shared_ptr<open_spiel::SharedDunePolicyValueNetImpl> model_dst;
  std::string dst_model_path;
  size_t dst_model_size = 0;
  std::string dst_model_sha256;
};

struct EmaSurgeryResult {
  std::shared_ptr<open_spiel::SharedDunePolicyValueNetImpl> model_ema_src;
  std::shared_ptr<open_spiel::SharedDunePolicyValueNetImpl> model_ema_dst;
  std::string dst_ema_model_path;
  size_t dst_ema_model_size = 0;
  std::string dst_ema_model_sha256;
  open_spiel::json::Object j_ema_obj;
  bool has_ema_manifest = false;
};

struct OptimizerSurgeryResult {
  std::unique_ptr<torch::optim::AdamW> opt_src;
  std::unique_ptr<torch::optim::AdamW> opt_dst;
  std::string dst_opt_path;
  size_t dst_opt_size = 0;
  std::string dst_opt_sha256;
};

ManifestMetadata ParseManifestMetadata(const open_spiel::json::Object& obj);

ManifestMetadata LoadManifestForModel(const std::string& model_path,
                                      const std::string& fallback_manifest = "");

std::shared_ptr<open_spiel::SharedDunePolicyValueNetImpl> MakeModel(
    int64_t obs_dim, int blocks, const std::string& sem_schema,
    open_spiel::dune_imperium::MarketAppendixMode mkt_mode,
    open_spiel::dune_imperium::DeckPoolSemantics semantics);

std::string BuildPrePrecisionConfigJsonString(
    int num_blocks,
    double base_lr,
    const ManifestMetadata& meta,
    const std::string& target_opponent_pool = "");

std::string ComputeManifestConfigFingerprint(
    int num_blocks,
    double base_lr,
    const ManifestMetadata& meta,
    const std::string& target_opponent_pool = "");

void VerifyOptimizerGroups(
    const torch::optim::AdamW& opt_dst,
    int dst_blocks,
    int dst_nb_first_idx,
    double base_lr,
    double nb_lr,
    const std::string& semantic_descriptor_schema);

LiveSurgeryResult RunLiveModelSurgery(
    const std::string& src_m_path,
    const std::string& out_dir,
    int src_blocks,
    int dst_blocks,
    int64_t init_seed,
    int64_t global_update,
    const ManifestMetadata& manifest_meta,
    const torch::Device& device);

EmaSurgeryResult RunEmaModelSurgery(
    const std::string& src_ema_m_path,
    const std::string& src_ema_json_path,
    const std::string& out_dir,
    int src_blocks,
    int dst_blocks,
    int64_t global_update,
    const ManifestMetadata& manifest_meta,
    const std::shared_ptr<open_spiel::SharedDunePolicyValueNetImpl>& model_dst,
    const torch::Device& device);

OptimizerSurgeryResult RunOptimizerSurgery(
    const std::shared_ptr<open_spiel::SharedDunePolicyValueNetImpl>& model_src,
    const std::shared_ptr<open_spiel::SharedDunePolicyValueNetImpl>& model_dst,
    const std::string& src_opt_path,
    const std::string& out_dir,
    int src_blocks,
    int dst_blocks,
    int src_nb_first_idx,
    int dst_nb_first_idx,
    double base_lr,
    double nb_lr,
    int64_t global_update,
    const ManifestMetadata& manifest_meta,
    const torch::Device& device);

void UpdateAndSaveManifests(
    open_spiel::json::Object& j_obj,
    EmaSurgeryResult& ema_res,
    const ManifestMetadata& manifest_meta,
    int src_blocks,
    int dst_blocks,
    double base_lr,
    const std::string& target_opp_pool,
    bool update_config_fingerprint,
    int64_t global_update,
    const std::string& out_dir,
    const LiveSurgeryResult& live_res,
    const OptimizerSurgeryResult& opt_res);

void VerifyB2OptimizerMoments(
    const std::shared_ptr<open_spiel::SharedDunePolicyValueNetImpl>& model_src,
    torch::optim::AdamW& opt_src,
    const std::shared_ptr<open_spiel::SharedDunePolicyValueNetImpl>& model_dst,
    torch::optim::AdamW& opt_dst,
    int src_blocks = 12,
    int dst_blocks = 16);

void VerifyPostSerializationOptimizerReload(
    const std::shared_ptr<open_spiel::SharedDunePolicyValueNetImpl>& model_src,
    torch::optim::AdamW& opt_src,
    const std::string& dst_model_path,
    const std::string& opt_to_check,
    int src_blocks,
    int dst_blocks,
    int dst_nb_first_idx,
    double base_lr,
    double nb_lr,
    const ManifestMetadata& manifest_meta,
    const torch::Device& device);

void VerifyB3IdentityBin(
    const std::shared_ptr<open_spiel::SharedDunePolicyValueNetImpl>& model_src,
    const std::shared_ptr<open_spiel::SharedDunePolicyValueNetImpl>& model_dst,
    const std::string& obs_bin_path,
    int test_states_count,
    int64_t obs_size);

void VerifyB3IdentityJson(
    const std::shared_ptr<open_spiel::SharedDunePolicyValueNetImpl>& model_src,
    const std::shared_ptr<open_spiel::SharedDunePolicyValueNetImpl>& model_dst,
    const std::string& states_json_path,
    int test_states_count,
    int64_t obs_size,
    open_spiel::dune_imperium::MarketAppendixMode mkt_mode,
    open_spiel::dune_imperium::DeckPoolSemantics semantics);

void RunNativeEffectCheck(
    const std::string& model_path,
    const std::string& obs_bin_path,
    const std::string& meta_jsonl_path,
    const std::string& states_json_path,
    int test_states_count,
    int num_blocks = 16,
    int bypass_first_index = 12,
    const ManifestMetadata& meta = ManifestMetadata());

void RunOptimizerEqualityCheck(
    const std::string& src_opt_path,
    const std::string& check_opt_path,
    const std::string& model_dst_path,
    const std::string& model_src_path,
    int src_blocks = 12,
    int dst_blocks = 16,
    int src_nb_first_idx = 8,
    int dst_nb_first_idx = 16,
    double base_lr = 5e-6,
    double nb_lr = 5e-5,
    const ManifestMetadata& meta = ManifestMetadata());

}  // namespace open_spiel::dune_surgery_12block

#endif  // OPEN_SPIEL_EXAMPLES_DUNE_SURGERY_12BLOCK_INTERNAL_H_
