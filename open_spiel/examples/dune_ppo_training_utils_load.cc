#include "dune_ppo_training_utils_internal.h"
#include "open_spiel/abseil-cpp/absl/flags/flag.h"
#include "open_spiel/abseil-cpp/absl/strings/str_format.h"
#include "open_spiel/games/dune_imperium/dune_imperium.h"
#include "open_spiel/utils/json.h"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>

namespace open_spiel {

bool LoadModelCheckpointMigrating(
    std::shared_ptr<SharedDunePolicyValueNetImpl> model,
    const std::string& path, torch::Device device,
    bool allow_skip_semantic_scorer) {
  if (model->semantic_descriptor_schema_ == dune_semantic::kDescriptorSchemaVersionV4) {
    model->SetSemanticDescriptorSchema(dune_semantic::kDescriptorSchemaVersionV4);
  } else if (model->semantic_descriptor_schema_ == dune_semantic::kDescriptorSchemaVersionV5) {
    model->SetSemanticDescriptorSchema(dune_semantic::kDescriptorSchemaVersionV5);
  }
  const int64_t input_dim = model->input_layer->weight.size(1);
  const int64_t hidden_dim = model->input_layer->weight.size(0);
  const int64_t action_dim = model->policy_head->weight.size(0);
  const int num_blocks = static_cast<int>(model->res_blocks.size());
  const bool use_nonlinear = absl::GetFlag(FLAGS_nonlinear_value_head);
  if (CheckpointHasSemanticScorer(path) && !model->with_semantic_scorer_ &&
      !allow_skip_semantic_scorer) {
    SpielFatalError("Model loading must preserve the source scorer; checkpoint at " + path +
                    " has a semantic scorer but model was not configured with --enable_semantic_scorer=true.");
  }

  std::filesystem::path metadata_path(path);
  metadata_path.replace_extension(".json");
  std::string source_schema;
  bool metadata_valid = false;
  if (std::filesystem::exists(metadata_path)) {
    std::ifstream input(metadata_path);
    if (input) {
      const std::string content((std::istreambuf_iterator<char>(input)), {});
      const auto metadata = open_spiel::json::FromString(content);
      if (metadata && metadata->IsObject()) {
        metadata_valid = true;
        const auto& object = metadata->GetObject();
        auto nb_entry = object.find("num_blocks");
        if (nb_entry != object.end()) {
          int json_blocks = -1;
          if (nb_entry->second.IsInt()) {
            json_blocks = nb_entry->second.GetInt();
          } else if (nb_entry->second.IsString()) {
            json_blocks = std::stoi(nb_entry->second.GetString());
          }
          if (json_blocks != -1 && json_blocks != num_blocks) {
            SpielFatalError(absl::StrFormat(
                "LoadModelCheckpointMigrating: Checkpoint %s metadata num_blocks (%d) does not match target model num_blocks (%d)",
                metadata_path.string(), json_blocks, num_blocks));
          }
        }
        auto hd_entry = object.find("hidden_dim");
        if (hd_entry != object.end()) {
          int json_hidden = -1;
          if (hd_entry->second.IsInt()) {
            json_hidden = hd_entry->second.GetInt();
          } else if (hd_entry->second.IsString()) {
            json_hidden = std::stoi(hd_entry->second.GetString());
          }
          if (json_hidden != -1 && json_hidden != hidden_dim) {
            SpielFatalError(absl::StrFormat(
                "LoadModelCheckpointMigrating: Checkpoint %s metadata hidden_dim (%d) does not match target model hidden_dim (%d)",
                metadata_path.string(), json_hidden, hidden_dim));
          }
        }
        auto schema_entry = object.find("semantic_descriptor_schema");
        if (schema_entry != object.end() && schema_entry->second.IsString()) {
          source_schema = schema_entry->second.GetString();
          if (source_schema != dune_semantic::kDescriptorSchemaVersionV2 &&
              source_schema != dune_semantic::kDescriptorSchemaVersionV3 &&
              source_schema != dune_semantic::kDescriptorSchemaVersionV4 &&
              source_schema != dune_semantic::kDescriptorSchemaVersionV5) {
            SpielFatalError("Mismatched or invalid semantic_descriptor_schema in metadata: " + metadata_path.string());
          }
        }
        auto eze_entry = object.find("enable_zone_encoder");
        if (eze_entry != object.end()) {
          bool eze = (eze_entry->second.IsBool() && eze_entry->second.GetBool()) ||
                     (eze_entry->second.IsString() && eze_entry->second.GetString() == "true");
          if (eze) {
            model->EnableZoneEncoder();
          }
        }
        auto sem_entry = object.find("deck_pool_semantics");
        if (sem_entry != object.end() && sem_entry->second.IsString()) {
          model->deck_pool_semantics_ = dune_imperium::ParseDeckPoolSemanticsOrDie(
              sem_entry->second.GetString());
        } else {
          model->deck_pool_semantics_ = dune_imperium::DeckPoolSemantics::kLegacy;
        }
        auto mode_entry = object.find("market_appendix_mode");
        auto mode = dune_imperium::MarketAppendixMode::kNone;
        if (mode_entry != object.end() && mode_entry->second.IsString()) {
          mode = dune_imperium::ParseMarketAppendixMode(mode_entry->second.GetString());
        }
        auto hash_entry = object.find("feature_schema_sha256");
        if (hash_entry != object.end() && hash_entry->second.IsString()) {
          const auto schema = dune_imperium::GetInformationStateSchema(mode, model->deck_pool_semantics_);
          if (hash_entry->second.GetString() != schema.sha256) {
            SpielFatalError("LoadModelCheckpointMigrating: Checkpoint " + path +
                            " feature_schema_sha256 (" + hash_entry->second.GetString() +
                            ") does not match schema (" + std::string(schema.sha256) + ")");
          }
        }
      }
    }
  } else {
    model->deck_pool_semantics_ = dune_imperium::DeckPoolSemantics::kLegacy;
  }

  auto update_semantic_schema = [&]() {
    if (!source_schema.empty()) {
      if (SemanticSchemaRank(source_schema) >= SemanticSchemaRank(model->semantic_descriptor_schema_)) {
        model->SetSemanticDescriptorSchema(source_schema);
      } else {
        model->SetSemanticDescriptorSchema(model->semantic_descriptor_schema_);
      }
    } else if (model->semantic_descriptor_schema_.empty()) {
      model->SetSemanticDescriptorSchema(dune_semantic::kDescriptorSchemaVersionV3);
    }
  };

  int64_t ckpt_input_dim = input_dim;
  {
    torch::serialize::InputArchive archive;
    archive.load_from(path, torch::kCPU);
    torch::serialize::InputArchive in_archive;
    if (archive.try_read("input_layer", in_archive)) {
      torch::Tensor w;
      if (in_archive.try_read("weight", w)) {
        if (w.size(0) != hidden_dim) {
          SpielFatalError(absl::StrFormat(
              "LoadModelCheckpointMigrating: Checkpoint %s input_layer hidden_dim (%d) does not match target model hidden_dim (%d)",
              path, w.size(0), hidden_dim));
        }
        ckpt_input_dim = w.size(1);
      }
    }
    for (int b = 1; b <= num_blocks; ++b) {
      torch::serialize::InputArchive block_arch;
      if (!archive.try_read("res" + std::to_string(b), block_arch)) {
        SpielFatalError(absl::StrFormat(
            "LoadModelCheckpointMigrating: Checkpoint %s is missing residual block 'res%d' required by target model num_blocks (%d)",
            path, b, num_blocks));
      }
    }
    for (int b = num_blocks + 1; b <= num_blocks + 64; ++b) {
      torch::serialize::InputArchive extra_arch;
      if (archive.try_read("res" + std::to_string(b), extra_arch)) {
        SpielFatalError(absl::StrFormat(
            "LoadModelCheckpointMigrating: Checkpoint %s contains deeper residual block 'res%d' exceeding target model num_blocks (%d)",
            path, b, num_blocks));
      }
    }
  }

  if (!std::filesystem::exists(metadata_path)) {
    SpielFatalError("LoadModelCheckpointMigrating: Missing checkpoint sidecar JSON: " + metadata_path.string());
  }
  if (!metadata_valid) {
    SpielFatalError("LoadModelCheckpointMigrating: Invalid checkpoint sidecar JSON: " + metadata_path.string());
  }

  auto temp_model = std::make_shared<SharedDunePolicyValueNetImpl>(
      ckpt_input_dim, hidden_dim, action_dim, num_blocks, use_nonlinear,
      /*with_aux_heads=*/false);
  temp_model->to(device);
  torch::load(temp_model, path, device);

  torch::NoGradGuard no_grad;
  // Trunk.
  if (input_dim == dune_imperium::kFullPublicInformationV5StateSize) {
    const auto source_mode = ReadCheckpointInputMode(path, ckpt_input_dim);
    CopyInputToFullPublicInformationV5(model->input_layer->weight,
                                       temp_model->input_layer->weight, source_mode);
    model->market_appendix_mode_ = dune_imperium::MarketAppendixMode::kFullPublicInformationV5;
    update_semantic_schema();
    std::cout << "[INFO] Migrated model checkpoint input_dim from "
              << ckpt_input_dim << " ("
              << dune_imperium::MarketAppendixModeToString(source_mode)
              << ") to " << input_dim
              << " (full_public_information_v5).\n";
  } else if (input_dim == dune_imperium::kFullPublicInformationV4StateSize) {
    const auto source_mode = ReadCheckpointInputMode(path, ckpt_input_dim);
    CopyInputToFullPublicInformationV4(model->input_layer->weight,
                                       temp_model->input_layer->weight, source_mode);
    model->market_appendix_mode_ = dune_imperium::MarketAppendixMode::kFullPublicInformationV4;
    update_semantic_schema();
    std::cout << "[INFO] Migrated model checkpoint input_dim from "
              << ckpt_input_dim << " ("
              << dune_imperium::MarketAppendixModeToString(source_mode)
              << ") to " << input_dim
              << " (full_public_information_v4).\n";
  } else if (input_dim == dune_imperium::kFullPublicInformationStateSize) {
    const auto source_mode = ReadCheckpointInputMode(path, ckpt_input_dim);
    CopyInputToFullPublicInformation(model->input_layer->weight,
                                     temp_model->input_layer->weight, source_mode);
    model->market_appendix_mode_ = dune_imperium::MarketAppendixMode::kFullPublicInformationV3;
    update_semantic_schema();
  } else if (input_dim == dune_imperium::kOrderedCardSlotsInformationStateSize) {
    const auto source_mode = ReadCheckpointInputMode(path, ckpt_input_dim);
    CopyInputToOrderedCardSlots(model->input_layer->weight,
                               temp_model->input_layer->weight, source_mode);
    model->market_appendix_mode_ = dune_imperium::MarketAppendixMode::kOrderedCardSlotsV2;
    update_semantic_schema();
  } else if (input_dim == dune_imperium::kExpandedInformationStateSize) {
    if (ckpt_input_dim == input_dim) {
      model->input_layer->weight.copy_(temp_model->input_layer->weight);
    } else if (ckpt_input_dim == 5580) {
      model->input_layer->weight.slice(1, 0, 5580).copy_(temp_model->input_layer->weight);
      model->input_layer->weight.slice(1, 5580, 6215).zero_();
    } else {
      SpielFatalError(absl::StrFormat("Unsupported model input migration: ckpt=%d -> target=%d",
                                      ckpt_input_dim, input_dim));
    }
    const std::string flag_mode_str = absl::GetFlag(FLAGS_market_appendix_mode);
    if (flag_mode_str != "none") {
      model->market_appendix_mode_ = dune_imperium::ParseMarketAppendixMode(flag_mode_str);
    } else {
      model->market_appendix_mode_ = ReadCheckpointInputMode(path, ckpt_input_dim);
    }
    if (model->market_appendix_mode_ != dune_imperium::MarketAppendixMode::kZeros &&
        model->market_appendix_mode_ != dune_imperium::MarketAppendixMode::kOrderedMarket) {
      SpielFatalError("Ambiguous 6,215 model requires explicit market_appendix_mode ('zeros' or 'ordered_market')");
    }
  } else if (input_dim == open_spiel::kPrivilegedCriticInformationStateSize) {
    if (ckpt_input_dim == input_dim) {
      model->input_layer->weight.copy_(temp_model->input_layer->weight);
    } else if (ckpt_input_dim == dune_imperium::kFullPublicInformationStateSize) {
      // Step 3b (A2): Privileged critic: copy all 9,182 weights and zero the 3,432 new features.
      model->input_layer->weight.slice(1, 0, dune_imperium::kFullPublicInformationStateSize)
          .copy_(temp_model->input_layer->weight);
      model->input_layer->weight.slice(1, dune_imperium::kFullPublicInformationStateSize,
                                       open_spiel::kPrivilegedCriticInformationStateSize).zero_();
    } else {
      SpielFatalError(absl::StrFormat("Unsupported privileged critic input migration: ckpt=%d -> target=%d",
                                      ckpt_input_dim, input_dim));
    }
    model->market_appendix_mode_ = dune_imperium::MarketAppendixMode::kFullPublicInformationV3;
    update_semantic_schema();
  } else {
    SpielFatalError(absl::StrFormat("Unsupported model input migration: ckpt=%d -> target=%d",
                                    ckpt_input_dim, input_dim));
  }
  if (model->with_semantic_scorer_) {
    update_semantic_schema();
  }
  if (model->input_layer->bias.defined() &&
      temp_model->input_layer->bias.defined()) {
    model->input_layer->bias.copy_(temp_model->input_layer->bias);
  }
  SPIEL_CHECK_EQ(model->res_blocks.size(), temp_model->res_blocks.size());
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
  // Value head(s).
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

  // Auxiliary heads.
  bool migrated = true;
  if (model->has_aux_heads_) {
    torch::serialize::InputArchive archive;
    archive.load_from(path, device);
    torch::serialize::InputArchive head_archive;
    if (archive.try_read("final_vp_head", head_archive)) {
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
      migrated = false;
    } else {
      std::cout << "[PWO-5] MIGRATION: " << path
                << " predates the auxiliary heads. Trunk, policy head and "
                   "value head loaded without numerical change; the three "
                   "heads keep their deterministic init from "
                   "kHeadInitConstant=" << absl::GetFlag(FLAGS_head_init_constant)
                << "." << std::endl;
    }
  }
  if (model->with_semantic_scorer_) {
    torch::serialize::InputArchive archive;
    archive.load_from(path, device);
    torch::serialize::InputArchive scorer_archive;
    if (archive.try_read("semantic_scorer", scorer_archive)) {
      const std::string effective_schema = model->semantic_descriptor_schema_;
      model->semantic_scorer_->Load(scorer_archive, effective_schema);
      std::cout << "[SEMANTIC SCORER] Loaded semantic_scorer submodule from " << path
                << " (schema: " << model->semantic_descriptor_schema_ << ")" << std::endl;
    } else {
      std::cout << "[SEMANTIC SCORER] Checkpoint does not contain semantic_scorer; keeping zero-init scorer." << std::endl;
    }
  }
  if (model->enable_zone_encoder_ && model->zone_encoder_) {
    torch::serialize::InputArchive archive;
    archive.load_from(path, device);
    torch::serialize::InputArchive ze_archive;
    if (archive.try_read("zone_encoder", ze_archive)) {
      model->zone_encoder_->load(ze_archive);
      std::cout << "[ZONE ENCODER] Loaded zone_encoder submodule from " << path << std::endl;
    } else {
      std::cout << "[ZONE ENCODER] Checkpoint does not contain zone_encoder; keeping zero-init zone encoder." << std::endl;
    }
  }
  return migrated;
}

bool DetectCheckpointModelDimensions(
    const std::string& model_path, int* hidden_dim, int* num_blocks,
    int* input_dim, std::string* market_mode,
    dune_imperium::DeckPoolSemantics* deck_pool_semantics,
    std::string* semantic_descriptor_schema,
    bool* slot_indexed_out_layer_f) {
  std::string json_path = "";
  std::filesystem::path p(model_path);
  std::filesystem::path p_json = p;
  p_json.replace_extension(".json");
  if (std::filesystem::exists(p_json)) {
    json_path = p_json.string();
  } else {
    std::filesystem::path p_parent_manifest = p.parent_path() / "manifest.json";
    if (std::filesystem::exists(p_parent_manifest)) {
      json_path = p_parent_manifest.string();
    }
  }

  if (json_path.empty()) {
    SpielFatalError(
        "DetectCheckpointModelDimensions: Missing required checkpoint sidecar JSON for " +
        model_path);
  }

  try {
    std::ifstream f(json_path);
    if (f) {
      std::string content((std::istreambuf_iterator<char>(f)),
                          std::istreambuf_iterator<char>());
      auto val_opt = json::FromString(content);
      if (val_opt.has_value() && val_opt->IsObject()) {
        const auto& obj = val_opt->GetObject();
        if (input_dim != nullptr) {
          auto it_od = obj.find("observation_dim");
          if (it_od != obj.end() && it_od->second.IsInt()) {
            *input_dim = static_cast<int>(it_od->second.GetInt());
          }
        }
        if (market_mode != nullptr) {
          auto it_mm = obj.find("market_appendix_mode");
          if (it_mm != obj.end() && it_mm->second.IsString()) {
            *market_mode = it_mm->second.GetString();
          }
        }
        dune_imperium::DeckPoolSemantics sem =
            dune_imperium::DeckPoolSemantics::kLegacy;
        auto it_sem = obj.find("deck_pool_semantics");
        if (it_sem != obj.end() && it_sem->second.IsString()) {
          sem = dune_imperium::ParseDeckPoolSemanticsOrDie(
              it_sem->second.GetString());
        }
        if (deck_pool_semantics != nullptr) {
          *deck_pool_semantics = sem;
        }
        if (input_dim != nullptr &&
            dune_imperium::IsExtendedInformationStateSize(*input_dim)) {
          auto it_mode = obj.find("market_appendix_mode");
          if (it_mode == obj.end() || !it_mode->second.IsString()) {
            SpielFatalError("Missing card-slot input mode for " + model_path);
          }
          const auto mode = dune_imperium::ParseMarketAppendixMode(
              it_mode->second.GetString());
          const auto schema =
              dune_imperium::GetInformationStateSchema(mode, sem);
          SPIEL_CHECK_EQ(*input_dim, schema.size);
          auto it_schema = obj.find("feature_schema_sha256");
          std::string schema_hash =
              (it_schema != obj.end() && it_schema->second.IsString())
                  ? it_schema->second.GetString()
                  : "";
          if (schema_hash != schema.sha256) {
            SpielFatalError(absl::StrFormat(
                "Expanded model checkpoint %s has invalid or missing feature_schema_sha256 ('%s', expected '%s')",
                model_path, schema_hash, schema.sha256));
          }
        }
        auto it_sds = obj.find("semantic_descriptor_schema");
        if (it_sds != obj.end() && it_sds->second.IsString()) {
          std::string schema = it_sds->second.GetString();
          if (schema != dune_semantic::kDescriptorSchemaVersionV2 &&
              schema != dune_semantic::kDescriptorSchemaVersionV3 &&
              schema != dune_semantic::kDescriptorSchemaVersionV4 &&
              schema != dune_semantic::kDescriptorSchemaVersionV5) {
            SpielFatalError(absl::StrFormat(
                "Model checkpoint %s has invalid semantic_descriptor_schema ('%s')",
                model_path, schema));
          }
          if (semantic_descriptor_schema != nullptr) {
            *semantic_descriptor_schema = schema;
          }
        }
        if (slot_indexed_out_layer_f != nullptr) {
          auto it_sif = obj.find("slot_indexed_out_layer_f");
          if (it_sif != obj.end()) {
            *slot_indexed_out_layer_f =
                (it_sif->second.IsBool() && it_sif->second.GetBool()) ||
                (it_sif->second.IsString() &&
                 it_sif->second.GetString() == "true");
          } else {
            *slot_indexed_out_layer_f = false;
          }
        }
        auto it_ess = obj.find("enable_semantic_scorer");
        if (it_ess != obj.end()) {
          bool enabled =
              (it_ess->second.IsBool() && it_ess->second.GetBool()) ||
              (it_ess->second.IsString() &&
               it_ess->second.GetString() == "true");
          if (enabled) {
            if (it_sds == obj.end() || !it_sds->second.IsString()) {
              SpielFatalError(absl::StrFormat(
                  "Model checkpoint %s has enable_semantic_scorer=true but missing or invalid semantic_descriptor_schema",
                  model_path));
            }
          }
        }
        if (input_dim != nullptr &&
            dune_imperium::IsFullPublicInformationStateSize(*input_dim)) {
          auto it_ess_9182 = obj.find("enable_semantic_scorer");
          bool enabled =
              it_ess_9182 != obj.end() &&
              ((it_ess_9182->second.IsBool() && it_ess_9182->second.GetBool()) ||
               (it_ess_9182->second.IsString() &&
                it_ess_9182->second.GetString() == "true"));
          if (!enabled) {
            SpielFatalError(absl::StrFormat(
                "Model checkpoint %s with full public information input dim requires enable_semantic_scorer=true",
                model_path));
          }
          if (it_sds == obj.end() || !it_sds->second.IsString() ||
              (it_sds->second.GetString() !=
                   dune_semantic::kDescriptorSchemaVersionV3 &&
               it_sds->second.GetString() !=
                   dune_semantic::kDescriptorSchemaVersionV4 &&
               it_sds->second.GetString() !=
                   dune_semantic::kDescriptorSchemaVersionV5)) {
            SpielFatalError(absl::StrFormat(
                "Model checkpoint %s with 9182 input dim requires semantic_descriptor_schema='%s', '%s', or '%s'",
                model_path, dune_semantic::kDescriptorSchemaVersionV3,
                dune_semantic::kDescriptorSchemaVersionV4,
                dune_semantic::kDescriptorSchemaVersionV5));
          }
        }
        auto it_hd = obj.find("hidden_dim");
        auto it_nb = obj.find("num_blocks");
        if (it_hd != obj.end() && it_hd->second.IsInt() &&
            it_nb != obj.end() && it_nb->second.IsInt()) {
          *hidden_dim = static_cast<int>(it_hd->second.GetInt());
          *num_blocks = static_cast<int>(it_nb->second.GetInt());
          return true;
        }
        auto it_arch = obj.find("architecture");
        if (it_arch != obj.end() && it_arch->second.IsObject()) {
          const auto& arch_obj = it_arch->second.GetObject();
          auto it_arch_hd = arch_obj.find("hidden_dim");
          auto it_arch_nb = arch_obj.find("num_blocks");
          if (it_arch_hd != arch_obj.end() && it_arch_hd->second.IsInt() &&
              it_arch_nb != arch_obj.end() && it_arch_nb->second.IsInt()) {
            *hidden_dim = static_cast<int>(it_arch_hd->second.GetInt());
            *num_blocks = static_cast<int>(it_arch_nb->second.GetInt());
            return true;
          }
        }
      }
    }
  } catch (const std::exception& e) {
    SpielFatalError(absl::StrFormat(
        "Error reading or parsing sidecar JSON for %s: %s", model_path,
        e.what()));
  }

  try {
    torch::serialize::InputArchive archive;
    archive.load_from(model_path, torch::kCPU);
    torch::serialize::InputArchive input_layer_archive;
    archive.read("input_layer", input_layer_archive);
    torch::Tensor weight;
    input_layer_archive.read("weight", weight);
    if (dune_imperium::IsExtendedInformationStateSize(weight.size(1))) {
      SpielFatalError(
          "Extended information state checkpoint requires valid schema metadata: " +
          model_path);
    }
    torch::serialize::InputArchive s_arch;
    if (archive.try_read("semantic_scorer", s_arch)) {
      SpielFatalError(
          "Checkpoint contains semantic_scorer module but sidecar metadata is missing or invalid: " +
          model_path);
    }
    *hidden_dim = static_cast<int>(weight.size(0));
    if (input_dim != nullptr) {
      *input_dim = static_cast<int>(weight.size(1));
    }
    int blocks = 0;
    while (true) {
      torch::serialize::InputArchive res_archive;
      std::string block_name = "res" + std::to_string(blocks + 1);
      try {
        archive.read(block_name, res_archive);
        blocks++;
      } catch (...) {
        break;
      }
    }
    *num_blocks = blocks;
    return true;
  } catch (const std::exception& e) {
    SpielFatalError(absl::StrFormat(
        "DetectCheckpointModelDimensions failed for %s: %s", model_path,
        e.what()));
  }
  return false;
}

void LoadModelCheckpoint(std::shared_ptr<SharedDunePolicyValueNetImpl> model,
                         const std::string& path, torch::Device device,
                         bool allow_skip_semantic_scorer) {
  if (model->has_aux_heads_ || model->with_semantic_scorer_ ||
      model->enable_zone_encoder_ || CheckpointHasSemanticScorer(path) ||
      dune_imperium::IsExtendedInformationStateSize(model->input_layer->weight.size(1)) ||
      model->input_layer->weight.size(1) == open_spiel::kPrivilegedCriticInformationStateSize ||
      allow_skip_semantic_scorer) {
    LoadModelCheckpointMigrating(model, path, device, allow_skip_semantic_scorer);
    return;
  }
  model->deck_pool_semantics_ = ReadCheckpointDeckPoolSemantics(path);
  if (!absl::GetFlag(FLAGS_nonlinear_value_head)) {
    const int64_t expected_input_dim = model->input_layer->weight.size(1);
    torch::load(model, path, device);
    SPIEL_CHECK_EQ(model->input_layer->weight.size(1), expected_input_dim);
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

}  // namespace open_spiel
