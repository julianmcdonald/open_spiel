#ifndef OPEN_SPIEL_EXAMPLES_DUNE_NETWORK_CHECKPOINT_H_
#define OPEN_SPIEL_EXAMPLES_DUNE_NETWORK_CHECKPOINT_H_

#include <c10/cuda/CUDAStream.h>
#include <torch/torch.h>

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <memory>
#include <string>
#include <system_error>
#include <vector>

#include "dune_network_model.h"
#include "open_spiel/spiel.h"
#include "open_spiel/utils/json.h"

namespace open_spiel {

inline void CopyModelWeights(std::shared_ptr<SharedDunePolicyValueNetImpl> source,
                             std::shared_ptr<SharedDunePolicyValueNetImpl> target) {
  torch::NoGradGuard no_grad;
  target->SetSemanticDescriptorSchema(source->semantic_descriptor_schema_);
  if (source->semantic_scorer_ && target->semantic_scorer_ &&
      source->semantic_scorer_->enable_f_ &&
      (!target->semantic_scorer_->enable_f_ ||
       target->semantic_scorer_->slot_indexed_f_ != source->semantic_scorer_->slot_indexed_f_)) {
    target->semantic_scorer_->EnableF(source->semantic_scorer_->slot_indexed_f_);
  }
  if (source->enable_zone_encoder_ && !target->enable_zone_encoder_) {
    target->EnableZoneEncoder();
  }
  if (!source->parameters().empty()) {
    target->to(source->parameters().front().device());
  }
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
  target->market_appendix_mode_ = source->market_appendix_mode_;
  target->deck_pool_semantics_ = source->deck_pool_semantics_;
  target->semantic_descriptor_schema_ = source->semantic_descriptor_schema_;
  target->enable_zone_encoder_ = source->enable_zone_encoder_;
  if (target->use_fp8_trunk()) {
    target->PrepareFp8Cache();
  } else if (target->bf16_inference_cache_ready()) {
    target->RefreshBf16InferenceCache();
  }
  if (!target->parameters().empty() && target->parameters().front().is_cuda()) {
    c10::cuda::getCurrentCUDAStream().synchronize();
  }
}

inline void CopyInputToOrderedCardSlots(
    torch::Tensor target, const torch::Tensor& source,
    dune_imperium::MarketAppendixMode source_mode) {
  const auto schema = dune_imperium::GetInformationStateSchema(source_mode);
  SPIEL_CHECK_EQ(source.dim(), 2);
  SPIEL_CHECK_EQ(target.dim(), 2);
  SPIEL_CHECK_EQ(source.size(0), target.size(0));
  SPIEL_CHECK_EQ(source.size(1), schema.size);
  SPIEL_CHECK_EQ(target.size(1),
                 dune_imperium::kOrderedCardSlotsInformationStateSize);
  // A zero-appendix model has never used its extra columns. Activating those
  // columns must not import arbitrary saved values into the migrated policy.
  const int copied_columns = source_mode == dune_imperium::MarketAppendixMode::kZeros
      ? dune_imperium::kLegacyInformationStateSize : schema.size;
  torch::NoGradGuard no_grad;
  target.zero_();
  target.slice(1, 0, copied_columns).copy_(source.slice(1, 0, copied_columns));
}

inline void CopyInputToFullPublicInformation(
    torch::Tensor target, const torch::Tensor& source,
    dune_imperium::MarketAppendixMode source_mode) {
  const auto schema = dune_imperium::GetInformationStateSchema(source_mode);
  SPIEL_CHECK_EQ(source.dim(), 2);
  SPIEL_CHECK_EQ(target.dim(), 2);
  SPIEL_CHECK_EQ(source.size(0), target.size(0));
  SPIEL_CHECK_EQ(source.size(1), schema.size);
  SPIEL_CHECK_EQ(target.size(1),
                 dune_imperium::kFullPublicInformationStateSize);
  const int copied_columns = source_mode == dune_imperium::MarketAppendixMode::kZeros
      ? dune_imperium::kLegacyInformationStateSize : schema.size;
  torch::NoGradGuard no_grad;
  target.zero_();
  target.slice(1, 0, copied_columns).copy_(source.slice(1, 0, copied_columns));
}

inline void CopyInputToFullPublicInformationV4(
    torch::Tensor target, const torch::Tensor& source,
    dune_imperium::MarketAppendixMode source_mode) {
  const auto schema = dune_imperium::GetInformationStateSchema(source_mode);
  SPIEL_CHECK_EQ(target.dim(), 2);
  SPIEL_CHECK_EQ(source.dim(), 2);
  SPIEL_CHECK_EQ(target.size(0), source.size(0));
  SPIEL_CHECK_EQ(target.size(1),
                 dune_imperium::kFullPublicInformationV4StateSize);
  SPIEL_CHECK_EQ(source.size(1), schema.size);
  const int64_t copied_columns =
      source_mode == dune_imperium::MarketAppendixMode::kZeros
          ? dune_imperium::kLegacyInformationStateSize
          : schema.size;
  torch::NoGradGuard no_grad;
  target.zero_();
  target.slice(1, 0, copied_columns)
      .copy_(source.slice(1, 0, copied_columns));
}

inline void CopyInputToFullPublicInformationV5(
    torch::Tensor target, const torch::Tensor& source,
    dune_imperium::MarketAppendixMode source_mode) {
  const auto schema = dune_imperium::GetInformationStateSchema(source_mode);
  SPIEL_CHECK_EQ(target.dim(), 2);
  SPIEL_CHECK_EQ(source.dim(), 2);
  SPIEL_CHECK_EQ(target.size(0), source.size(0));
  SPIEL_CHECK_EQ(target.size(1),
                 dune_imperium::kFullPublicInformationV5StateSize);
  SPIEL_CHECK_EQ(source.size(1), schema.size);
  const int64_t copied_columns =
      source_mode == dune_imperium::MarketAppendixMode::kZeros
          ? dune_imperium::kLegacyInformationStateSize
          : schema.size;
  torch::NoGradGuard no_grad;
  target.zero_();
  target.slice(1, 0, copied_columns)
      .copy_(source.slice(1, 0, copied_columns));
}

inline void LoadModelCheckpointRobust(
    std::shared_ptr<SharedDunePolicyValueNetImpl> model,
    const std::string& path,
    torch::Device device) {
  torch::serialize::InputArchive archive;
  archive.load_from(path, device);

  torch::NoGradGuard no_grad;
  torch::serialize::InputArchive in_arch;
  const int64_t expected_input_dim = model->input_layer->weight.size(1);
  if (!archive.try_read("input_layer", in_arch)) {
    SpielFatalError("LoadModelCheckpointRobust: Missing required module 'input_layer' in " + path);
  }
  model->input_layer->load(in_arch);
  if (model->input_layer->weight.size(1) != expected_input_dim) {
    SpielFatalError("LoadModelCheckpointRobust: Mismatched input dimension in " + path);
  }

  const size_t N = model->res_blocks.size();
  for (size_t i = 0; i < N; ++i) {
    torch::serialize::InputArchive b_arch;
    std::string b_name = "res" + std::to_string(i + 1);
    if (!archive.try_read(b_name, b_arch)) {
      SpielFatalError("LoadModelCheckpointRobust: Missing required residual block '" + b_name + "' in " + path);
    }
    model->res_blocks[i]->load(b_arch);
  }

  // Depth guard: raise error if archive contains any deeper residual block exceeding model capacity N
  {
    torch::serialize::InputArchive extra_arch;
    if (archive.try_read("res0", extra_arch)) {
      SpielFatalError("LoadModelCheckpointRobust: Checkpoint " + path +
                      " contains deeper residual block 'res0' exceeding model num_blocks (" +
                      std::to_string(N) + ")");
    }
    for (size_t k = N + 1; k <= N + 64; ++k) {
      std::string extra_name = "res" + std::to_string(k);
      if (archive.try_read(extra_name, extra_arch)) {
        SpielFatalError("LoadModelCheckpointRobust: Checkpoint " + path +
                        " contains deeper residual block '" + extra_name +
                        "' exceeding model num_blocks (" + std::to_string(N) + ")");
      }
    }
  }

  torch::serialize::InputArchive p_arch;
  if (!archive.try_read("policy_head", p_arch)) {
    SpielFatalError("LoadModelCheckpointRobust: Missing required module 'policy_head' in " + path);
  }
  model->policy_head->load(p_arch);

  torch::serialize::InputArchive v_arch;
  if (!archive.try_read("value_head", v_arch)) {
    SpielFatalError("LoadModelCheckpointRobust: Missing required module 'value_head' in " + path);
  }
  model->value_head->load(v_arch);

  if (model->use_nonlinear_value_head_) {
    torch::serialize::InputArchive v2_arch;
    if (!archive.try_read("value_head2", v2_arch)) {
      SpielFatalError("LoadModelCheckpointRobust: Missing required module 'value_head2' in " + path);
    }
    model->value_head2->load(v2_arch);
  }

  if (model->has_aux_heads_) {
    torch::serialize::InputArchive a1, a2, a3;
    if (archive.try_read("final_vp_head", a1)) model->final_vp_head->load(a1);
    if (archive.try_read("terminal_round_head", a2)) model->terminal_round_head->load(a2);
    if (archive.try_read("next_own_action_head", a3)) model->next_own_action_head->load(a3);
  }

  // Find sidecar JSON by replacing any file extension, including checking resolved symlink targets
  std::string json_path = std::filesystem::path(path).replace_extension(".json").string();
  std::vector<std::string> candidate_json_paths;
  candidate_json_paths.push_back(json_path);
  std::error_code ec_sym;
  if (std::filesystem::is_symlink(path, ec_sym)) {
    std::error_code ec;
    auto resolved = std::filesystem::canonical(path, ec);
    if (!ec) {
      std::string resolved_json = resolved.replace_extension(".json").string();
      if (resolved_json != json_path) {
        candidate_json_paths.push_back(resolved_json);
      }
    }
  }

  // Validate num_blocks in sidecar JSON if present
  for (const auto& jp : candidate_json_paths) {
    std::ifstream jf_nb(jp);
    if (jf_nb.good()) {
      std::string jstr((std::istreambuf_iterator<char>(jf_nb)),
                       std::istreambuf_iterator<char>());
      auto jval = open_spiel::json::FromString(jstr);
      if (jval.has_value() && jval->IsObject()) {
        const auto& dict = jval->GetObject();
        auto it_nb = dict.find("num_blocks");
        if (it_nb != dict.end()) {
          int64_t declared_blocks = -1;
          if (it_nb->second.IsInt()) {
            declared_blocks = it_nb->second.GetInt();
          } else if (it_nb->second.IsString()) {
            declared_blocks = std::stoll(it_nb->second.GetString());
          }
          if (declared_blocks != static_cast<int64_t>(N)) {
            SpielFatalError("LoadModelCheckpointRobust: Declared num_blocks (" +
                            std::to_string(declared_blocks) +
                            ") does not match model num_blocks (" +
                            std::to_string(N) + ") in " + jp);
          }
        }
      }
    }
  }

  bool sidecar_found = false;
  bool obs_dim_found = false;
  std::string sds_found = "";
  dune_imperium::MarketAppendixMode mode_found = dune_imperium::MarketAppendixMode::kNone;
  std::string feature_schema_sha = "";

  std::ifstream jf(json_path);
  if (!jf.good() && candidate_json_paths.size() > 1) {
    json_path = candidate_json_paths[1];
    jf.open(json_path);
  }
  if (!jf.good()) {
    SpielFatalError(absl::StrCat("Missing checkpoint sidecar JSON: ", json_path));
  }
  sidecar_found = true;
  std::string jstr((std::istreambuf_iterator<char>(jf)),
                   std::istreambuf_iterator<char>());
  auto jval = open_spiel::json::FromString(jstr);
  if (!jval.has_value() || !jval->IsObject()) {
    SpielFatalError("LoadModelCheckpointRobust: Malformed sidecar JSON in " + json_path);
  }
  const auto& dict = jval->GetObject();
  auto it_sds = dict.find("semantic_descriptor_schema");
  if (it_sds != dict.end() && it_sds->second.IsString()) {
    sds_found = it_sds->second.GetString();
    if (sds_found != dune_semantic::kDescriptorSchemaVersionV2 &&
        sds_found != dune_semantic::kDescriptorSchemaVersionV3 &&
        sds_found != dune_semantic::kDescriptorSchemaVersionV4 &&
        sds_found != dune_semantic::kDescriptorSchemaVersionV5) {
      SpielFatalError("LoadModelCheckpointRobust: Checkpoint " + path +
                      " has invalid semantic_descriptor_schema ('" + sds_found + "')");
    }
    if (SemanticSchemaRank(sds_found) >= SemanticSchemaRank(model->semantic_descriptor_schema_)) {
      model->SetSemanticDescriptorSchema(sds_found);
    } else {
      model->SetSemanticDescriptorSchema(model->semantic_descriptor_schema_);
    }
  }

  auto it_eze = dict.find("enable_zone_encoder");
  if (it_eze != dict.end()) {
    bool enabled = (it_eze->second.IsBool() && it_eze->second.GetBool()) ||
                   (it_eze->second.IsString() && it_eze->second.GetString() == "true");
    if (enabled) {
      model->EnableZoneEncoder();
    } else {
      model->enable_zone_encoder_ = false;
    }
  }

  auto it_ess = dict.find("enable_semantic_scorer");
  if (it_ess != dict.end()) {
    bool enabled = (it_ess->second.IsBool() && it_ess->second.GetBool()) ||
                   (it_ess->second.IsString() && it_ess->second.GetString() == "true");
    if (enabled && sds_found.empty()) {
      SpielFatalError("LoadModelCheckpointRobust: Checkpoint " + path +
                      " has enable_semantic_scorer=true but missing or invalid semantic_descriptor_schema");
    }
  }

  auto it_mode = dict.find("market_appendix_mode");
  if (it_mode != dict.end() && it_mode->second.IsString()) {
    mode_found = dune_imperium::ParseMarketAppendixMode(it_mode->second.GetString());
    model->market_appendix_mode_ = mode_found;
  }

  dune_imperium::DeckPoolSemantics resolved_semantics = dune_imperium::DeckPoolSemantics::kLegacy;
  auto it_dps = dict.find("deck_pool_semantics");
  if (it_dps != dict.end() && it_dps->second.IsString()) {
    resolved_semantics = dune_imperium::ParseDeckPoolSemanticsOrDie(it_dps->second.GetString());
  }
  model->deck_pool_semantics_ = resolved_semantics;

  auto it_hash = dict.find("feature_schema_sha256");
  if (it_hash != dict.end() && it_hash->second.IsString()) {
    feature_schema_sha = it_hash->second.GetString();
  }

  auto it_obs = dict.find("observation_dim");
  if (it_obs != dict.end()) {
    obs_dim_found = true;
    int64_t obs_dim = -1;
    if (it_obs->second.IsInt()) {
      obs_dim = it_obs->second.GetInt();
    } else if (it_obs->second.IsString()) {
      obs_dim = std::stoll(it_obs->second.GetString());
    }
    if (obs_dim != expected_input_dim) {
      SpielFatalError("LoadModelCheckpointRobust: Declared observation_dim (" +
                      std::to_string(obs_dim) + ") does not match model input dimension (" +
                      std::to_string(expected_input_dim) + ") in " + path);
    }
  }

  torch::serialize::InputArchive s_arch_check;
  const bool checkpoint_has_scorer = archive.try_read("semantic_scorer", s_arch_check);

  if (checkpoint_has_scorer) {
    if (!sidecar_found) {
      SpielFatalError("LoadModelCheckpointRobust: Checkpoint at " + path +
                      " contains trained semantic scorer, but sidecar JSON is missing: " + json_path);
    }
    if (sds_found.empty()) {
      SpielFatalError("LoadModelCheckpointRobust: Checkpoint " + path +
                      " contains semantic_scorer module but sidecar has missing or invalid semantic_descriptor_schema");
    }
    if (!model->with_semantic_scorer_ || !model->semantic_scorer_) {
      SpielFatalError("LoadModelCheckpointRobust: Checkpoint at " + path +
                      " contains a trained semantic scorer, but the model was "
                      "instantiated without with_semantic_scorer=true.");
    }
  }

  const auto schema = dune_imperium::GetInformationStateSchema(mode_found, resolved_semantics);
  if (dune_imperium::IsExtendedInformationStateSize(expected_input_dim)) {
    if (!sidecar_found) {
      SpielFatalError("LoadModelCheckpointRobust: Extended checkpoint requires valid sidecar metadata: " + path);
    }
    if (!obs_dim_found) {
      SpielFatalError("LoadModelCheckpointRobust: Extended checkpoint requires observation_dim in sidecar: " + path);
    }
    if (mode_found == dune_imperium::MarketAppendixMode::kNone) {
      SpielFatalError("LoadModelCheckpointRobust: Extended checkpoint requires explicit market_appendix_mode: " + path);
    }
    if (expected_input_dim != schema.size) {
      SpielFatalError("LoadModelCheckpointRobust: Input dimension mismatch with schema in " + path);
    }
    if (feature_schema_sha.empty() || feature_schema_sha != schema.sha256) {
      SpielFatalError("LoadModelCheckpointRobust: Extended checkpoint " + path +
                      " requires valid feature_schema_sha256 matching schema ('" +
                      schema.sha256 + "')");
    }
  } else if (!feature_schema_sha.empty() && feature_schema_sha != schema.sha256) {
    SpielFatalError("LoadModelCheckpointRobust: Checkpoint " + path +
                    " requires valid feature_schema_sha256 matching schema ('" +
                    schema.sha256 + "')");
  }
  auto it_label = dict.find("feature_schema_label");
  if (it_label != dict.end() && it_label->second.IsString()) {
    if (it_label->second.GetString() != schema.label) {
      SpielFatalError("LoadModelCheckpointRobust: Checkpoint " + path +
                      " feature_schema_label mismatch: got '" + it_label->second.GetString() +
                      "', expected '" + schema.label + "'");
    }
  }

  if (expected_input_dim == dune_imperium::kExpandedInformationStateSize) {
    if (mode_found != dune_imperium::MarketAppendixMode::kZeros &&
        mode_found != dune_imperium::MarketAppendixMode::kOrderedMarket) {
      SpielFatalError("LoadModelCheckpointRobust: Ambiguous 6,215 checkpoint requires explicit market_appendix_mode ('zeros' or 'ordered_market'): " + path);
    }
  }

  if (dune_imperium::IsFullPublicInformationStateSize(expected_input_dim)) {
    if (expected_input_dim == dune_imperium::kFullPublicInformationStateSize) {
      if (mode_found != dune_imperium::MarketAppendixMode::kFullPublicInformationV3) {
        SpielFatalError("LoadModelCheckpointRobust: 9182 checkpoint requires market_appendix_mode='full_public_information_v3': " + path);
      }
    } else if (expected_input_dim == dune_imperium::kFullPublicInformationV4StateSize) {
      if (mode_found != dune_imperium::MarketAppendixMode::kFullPublicInformationV4) {
        SpielFatalError("LoadModelCheckpointRobust: 9941 checkpoint requires market_appendix_mode='full_public_information_v4': " + path);
      }
    } else if (expected_input_dim == dune_imperium::kFullPublicInformationV5StateSize) {
      if (mode_found != dune_imperium::MarketAppendixMode::kFullPublicInformationV5) {
        SpielFatalError("LoadModelCheckpointRobust: 10445 checkpoint requires market_appendix_mode='full_public_information_v5': " + path);
      }
    }
    if (!checkpoint_has_scorer) {
      SpielFatalError("LoadModelCheckpointRobust: Full public information checkpoint requires trained semantic scorer: " + path);
    }
    if (sds_found != dune_semantic::kDescriptorSchemaVersionV3 &&
        sds_found != dune_semantic::kDescriptorSchemaVersionV4 &&
        sds_found != dune_semantic::kDescriptorSchemaVersionV5) {
      SpielFatalError("LoadModelCheckpointRobust: Full public information checkpoint requires semantic_descriptor_schema='semantic_action_v3', 'semantic_action_v4', or 'semantic_action_v5': " + path);
    }
    if (!model->with_semantic_scorer_ || !model->semantic_scorer_) {
      SpielFatalError("LoadModelCheckpointRobust: Full public information model requires with_semantic_scorer=true: " + path);
    }
  }

  if (model->semantic_descriptor_schema_ == dune_semantic::kDescriptorSchemaVersionV4) {
    model->SetSemanticDescriptorSchema(dune_semantic::kDescriptorSchemaVersionV4);
  } else if (model->semantic_descriptor_schema_ == dune_semantic::kDescriptorSchemaVersionV5) {
    model->SetSemanticDescriptorSchema(dune_semantic::kDescriptorSchemaVersionV5);
  }

  if (model->with_semantic_scorer_ && model->semantic_scorer_) {
    torch::serialize::InputArchive s_arch;
    if (archive.try_read("semantic_scorer", s_arch)) {
      model->semantic_scorer_->Load(s_arch, model->semantic_descriptor_schema_);
    }
  }

  if (model->enable_zone_encoder_ && model->zone_encoder_) {
    torch::serialize::InputArchive z_arch;
    if (archive.try_read("zone_encoder", z_arch)) {
      model->zone_encoder_->load(z_arch);
    }
  }

  if (model->bf16_inference_cache_ready()) {
    model->RefreshBf16InferenceCache();
  }
  if (!model->parameters().empty() && model->parameters().front().is_cuda()) {
    c10::cuda::getCurrentCUDAStream().synchronize();
  }
}

inline bool CheckpointHasSemanticScorer(const std::string& path, torch::Device device = torch::kCPU) {
  try {
    torch::serialize::InputArchive archive;
    archive.load_from(path, device);
    torch::serialize::InputArchive s_arch;
    return archive.try_read("semantic_scorer", s_arch);
  } catch (...) {
    return false;
  }
}

inline void RequireLegacy5580ManifestOrDie(const json::Object& object,
                                          const std::string& context) {
  auto it = object.find("deck_pool_semantics");
  if (it != object.end() && it->second.IsString()) {
    const auto sem = dune_imperium::ParseDeckPoolSemanticsOrDie(it->second.GetString());
    if (sem != dune_imperium::DeckPoolSemantics::kLegacy) {
      SpielFatalError("Legacy 5580 tool does not support fixed deck_pool_semantics checkpoint: " + context);
    }
  }
}

inline void RequireLegacy5580CheckpointOrDie(const std::string& checkpoint_path) {
  std::filesystem::path metadata_path(checkpoint_path);
  metadata_path.replace_extension(".json");
  if (!std::filesystem::exists(metadata_path)) {
    SpielFatalError("Missing checkpoint sidecar JSON: " + metadata_path.string());
  }
  std::ifstream input(metadata_path);
  if (!input) {
    SpielFatalError("Cannot open checkpoint sidecar JSON: " + metadata_path.string());
  }
  const std::string content((std::istreambuf_iterator<char>(input)), {});
  const auto metadata = open_spiel::json::FromString(content);
  if (!metadata || !metadata->IsObject()) {
    SpielFatalError("Invalid checkpoint sidecar JSON: " + metadata_path.string());
  }
  RequireLegacy5580ManifestOrDie(metadata->GetObject(), checkpoint_path);
}

}  // namespace open_spiel

#endif  // OPEN_SPIEL_EXAMPLES_DUNE_NETWORK_CHECKPOINT_H_
