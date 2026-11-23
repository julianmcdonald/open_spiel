#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "open_spiel/spiel_utils.h"
#include "open_spiel/utils/json.h"

#include <torch/torch.h>

#include "dune_surgery_12block_internal.h"

ABSL_FLAG(std::string, src_model,
          "/run/media/warcr/Storage/dune_drl_runtime/round7/u30650_12block_random_draft_overnight_20261006/checkpoints/ppo_model_update_31400.pt",
          "Source model path.");
ABSL_FLAG(std::string, src_optimizer,
          "/run/media/warcr/Storage/dune_drl_runtime/round7/u30650_12block_random_draft_overnight_20261006/checkpoints/ppo_optimizer_update_31400.pt",
          "Source optimizer path.");
ABSL_FLAG(std::string, src_manifest,
          "/run/media/warcr/Storage/dune_drl_runtime/round7/u30650_12block_random_draft_overnight_20261006/checkpoints/ppo_model_update_31400.json",
          "Source JSON manifest path.");
ABSL_FLAG(std::string, src_ema_model, "",
          "Optional source EMA model path (if empty, auto-detects in src_model directory).");
ABSL_FLAG(std::string, src_ema_manifest, "",
          "Optional source EMA JSON manifest path (if empty, auto-detects in src_model directory).");
ABSL_FLAG(std::string, out_dir,
          "/home/warcr/dune_drl_runtime/round7/u31400_16block_init",
          "Destination directory for output model, optimizer and manifest.");
ABSL_FLAG(int, src_blocks, 12, "Number of blocks in source model.");
ABSL_FLAG(int, dst_blocks, 16, "Number of blocks in destination model.");
ABSL_FLAG(int, src_new_block_first_index, 8,
          "First block index for higher learning rate group in SOURCE optimizer.");
ABSL_FLAG(int, new_block_first_index, 8,
          "First block index for higher learning rate group in DESTINATION optimizer.");
ABSL_FLAG(double, base_lr, 5e-6, "Base learning rate.");
ABSL_FLAG(double, new_block_lr, 5e-5, "Learning rate for new blocks group.");
ABSL_FLAG(int64_t, init_seed, 20261115,
          "Deterministic torch manual seed before constructing destination model.");
ABSL_FLAG(std::string, test_states_json, "",
          "Optional JSON array of stored states with 'serialized_state' and 'legal_actions' for v5 B3 identity & 3c native effect verification.");
ABSL_FLAG(std::string, target_training_opponent_pool, "",
          "If non-empty, updates training_opponent_pool in output manifests and recomputes config_fingerprint.");
ABSL_FLAG(bool, update_config_fingerprint, true,
          "Recompute config_fingerprint for dst_blocks (and target_training_opponent_pool) after verifying exact match on source manifest config_fingerprint.");
ABSL_FLAG(std::string, test_obs_bin,
          "/home/warcr/dune_drl_runtime/round7/held_out_u28700/trajectory_obs.bin",
          "Path to stored observation binary for B3 identity verification.");
ABSL_FLAG(int, test_states_count, 1024,
          "Number of stored states to verify in B3 identity check (>= 1000).");
ABSL_FLAG(std::string, check_optimizer, "",
          "Optional path to optimizer file to verify against surgery output.");
ABSL_FLAG(std::string, eval_model, "",
          "Optional path to model checkpoint to run native effect check on.");
ABSL_FLAG(std::string, test_meta_jsonl,
          "/home/warcr/dune_drl_runtime/round7/held_out_u28700/trajectory_meta.jsonl",
          "Path to trajectory metadata jsonl for legal action and game state reconstruction.");

namespace surgery = open_spiel::dune_surgery_12block;

int main(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);

  int src_blocks = absl::GetFlag(FLAGS_src_blocks);
  int dst_blocks = absl::GetFlag(FLAGS_dst_blocks);
  int src_nb_first_idx = absl::GetFlag(FLAGS_src_new_block_first_index);
  int dst_nb_first_idx = absl::GetFlag(FLAGS_new_block_first_index);
  double base_lr = absl::GetFlag(FLAGS_base_lr);
  double nb_lr = absl::GetFlag(FLAGS_new_block_lr);
  int64_t init_seed = absl::GetFlag(FLAGS_init_seed);
  std::string target_opp_pool = absl::GetFlag(FLAGS_target_training_opponent_pool);
  std::string test_states_json_path = absl::GetFlag(FLAGS_test_states_json);

  std::string src_m_path = absl::GetFlag(FLAGS_src_model);
  std::string src_opt_path = absl::GetFlag(FLAGS_src_optimizer);
  std::string src_json_path = absl::GetFlag(FLAGS_src_manifest);
  std::string src_ema_m_path = absl::GetFlag(FLAGS_src_ema_model);
  std::string src_ema_json_path = absl::GetFlag(FLAGS_src_ema_manifest);
  std::string out_dir = absl::GetFlag(FLAGS_out_dir);

  if (!absl::GetFlag(FLAGS_check_optimizer).empty()) {
    std::string check_opt = absl::GetFlag(FLAGS_check_optimizer);
    std::string eval_m = absl::GetFlag(FLAGS_eval_model);
    if (eval_m.empty()) {
      eval_m = out_dir + "/ppo_model_update_41760.pt";
      if (!std::filesystem::exists(eval_m)) {
        eval_m = out_dir + "/ppo_model_update_31400.pt";
      }
    }
    surgery::ManifestMetadata meta =
        surgery::LoadManifestForModel(eval_m, src_json_path);
    surgery::RunOptimizerEqualityCheck(
        src_opt_path, check_opt, eval_m, src_m_path, src_blocks, dst_blocks,
        src_nb_first_idx, dst_nb_first_idx, base_lr, nb_lr, meta);
    return 0;
  }

  if (!absl::GetFlag(FLAGS_eval_model).empty()) {
    std::string eval_m = absl::GetFlag(FLAGS_eval_model);
    std::string obs_bin = absl::GetFlag(FLAGS_test_obs_bin);
    std::string meta_jsonl = absl::GetFlag(FLAGS_test_meta_jsonl);
    int test_count = absl::GetFlag(FLAGS_test_states_count);
    surgery::ManifestMetadata meta =
        surgery::LoadManifestForModel(eval_m, src_json_path);
    surgery::RunNativeEffectCheck(eval_m, obs_bin, meta_jsonl, test_states_json_path,
                                  test_count, dst_blocks, src_blocks, meta);
    return 0;
  }

  std::cout << "======================================================================\n";
  std::cout << "DUNE DRL CAPACITY PROBE: " << src_blocks << " -> " << dst_blocks
            << " RESIDUAL BLOCKS SURGERY TOOL\n";
  std::cout << "======================================================================\n";
  std::cout << "Source model:     " << src_m_path << "\n";
  std::cout << "Source optimizer: " << src_opt_path << "\n";
  std::cout << "Source manifest:  " << src_json_path << "\n";
  std::cout << "Destination dir:  " << out_dir << "\n";
  std::cout << "Source new block first index:      " << src_nb_first_idx << "\n";
  std::cout << "Destination new block first index: " << dst_nb_first_idx << "\n";
  std::cout << "Base LR: " << base_lr << ", New Block LR: " << nb_lr << "\n";
  std::cout << "Init seed: " << init_seed << "\n";

  std::filesystem::create_directories(out_dir);

  torch::Device device(torch::kCPU);

  // Parse source manifest
  std::ifstream src_jf(src_json_path);
  if (!src_jf.is_open()) {
    open_spiel::SpielFatalError("Cannot open src_manifest: " + src_json_path);
  }
  std::string j_content((std::istreambuf_iterator<char>(src_jf)),
                        std::istreambuf_iterator<char>());
  src_jf.close();

  auto j_opt = open_spiel::json::FromString(j_content);
  if (!j_opt) {
    open_spiel::SpielFatalError("Failed parsing source manifest JSON");
  }
  auto j_obj = j_opt->GetObject();
  surgery::ManifestMetadata manifest_meta = surgery::ParseManifestMetadata(j_obj);

  int64_t global_update = manifest_meta.global_update;
  std::cout << "Global update from manifest: " << global_update << "\n";
  std::cout << "Observation dim:             " << manifest_meta.observation_dim << "\n";
  std::cout << "Market appendix mode:        " << manifest_meta.market_appendix_mode_str << "\n";
  std::cout << "Deck pool semantics:         " << manifest_meta.deck_pool_semantics_str << "\n";
  std::cout << "Semantic descriptor schema:  " << manifest_meta.semantic_descriptor_schema << "\n";

  // Auto-detect sibling EMA model and manifest if not explicitly given
  std::filesystem::path src_p(src_m_path);
  std::filesystem::path parent_p = src_p.parent_path();
  std::string auto_ema_m =
      (parent_p / ("ppo_model_ema_update_" + std::to_string(global_update) + ".pt")).string();
  std::string auto_ema_j =
      (parent_p / ("ppo_model_ema_update_" + std::to_string(global_update) + ".json")).string();

  if (src_ema_m_path.empty() && std::filesystem::exists(auto_ema_m)) {
    src_ema_m_path = auto_ema_m;
  }
  if (src_ema_json_path.empty() && std::filesystem::exists(auto_ema_j)) {
    src_ema_json_path = auto_ema_j;
  }
  if (!src_ema_m_path.empty()) {
    std::cout << "Source EMA model:    " << src_ema_m_path << "\n";
  }
  if (!src_ema_json_path.empty()) {
    std::cout << "Source EMA manifest: " << src_ema_json_path << "\n";
  }

  // 1-4. B1 Live Model Surgery
  surgery::LiveSurgeryResult live_res = surgery::RunLiveModelSurgery(
      src_m_path, out_dir, src_blocks, dst_blocks, init_seed, global_update,
      manifest_meta, device);

  // 4b. EMA Model Surgery (if present)
  surgery::EmaSurgeryResult ema_res = surgery::RunEmaModelSurgery(
      src_ema_m_path, src_ema_json_path, out_dir, src_blocks, dst_blocks,
      global_update, manifest_meta, live_res.model_dst, device);

  // 5. Optimizer Migration using MakeDuneOptimizer
  surgery::OptimizerSurgeryResult opt_res = surgery::RunOptimizerSurgery(
      live_res.model_src, live_res.model_dst, src_opt_path, out_dir, src_blocks,
      dst_blocks, src_nb_first_idx, dst_nb_first_idx, base_lr, nb_lr,
      global_update, manifest_meta, device);

  // 6 & 6b. Config Fingerprint Verification, Recomputation, and JSON Manifest Saving
  surgery::UpdateAndSaveManifests(
      j_obj, ema_res, manifest_meta, src_blocks, dst_blocks, base_lr,
      target_opp_pool, absl::GetFlag(FLAGS_update_config_fingerprint),
      global_update, out_dir, live_res, opt_res);

  // 7. Verify In-Memory Moments
  std::cout << "\n[B2 TEST STAGE (a)] Verifying in-memory optimizer moments...\n";
  surgery::VerifyB2OptimizerMoments(live_res.model_src, *opt_res.opt_src,
                                    live_res.model_dst, *opt_res.opt_dst,
                                    src_blocks, dst_blocks);

  // 7b. Verify Post-Serialization Reload
  std::string opt_to_check = opt_res.dst_opt_path;
  std::string flag_check_opt = absl::GetFlag(FLAGS_check_optimizer);
  if (!flag_check_opt.empty()) {
    opt_to_check = flag_check_opt;
  }
  surgery::VerifyPostSerializationOptimizerReload(
      live_res.model_src, *opt_res.opt_src, live_res.dst_model_path, opt_to_check,
      src_blocks, dst_blocks, dst_nb_first_idx, base_lr, nb_lr, manifest_meta,
      device);

  // 8. Verify B3 Identity Checks
  int test_count = absl::GetFlag(FLAGS_test_states_count);
  if (!test_states_json_path.empty()) {
    std::cout << "\n[B3 LIVE TEST] Verifying Forward Identity on Live models ("
              << test_count << " states via JSON)\n";
    surgery::VerifyB3IdentityJson(
        live_res.model_src, live_res.model_dst, test_states_json_path,
        test_count, manifest_meta.observation_dim,
        manifest_meta.market_appendix_mode, manifest_meta.deck_pool_semantics);
    if (ema_res.model_ema_src && ema_res.model_ema_dst) {
      std::cout << "\n[B3 EMA TEST] Verifying Forward Identity on EMA models ("
                << test_count << " states via JSON)\n";
      surgery::VerifyB3IdentityJson(
          ema_res.model_ema_src, ema_res.model_ema_dst, test_states_json_path,
          test_count, manifest_meta.observation_dim,
          manifest_meta.market_appendix_mode, manifest_meta.deck_pool_semantics);
    }
  } else {
    std::string obs_bin = absl::GetFlag(FLAGS_test_obs_bin);
    std::cout << "\n[B3 LIVE TEST] Verifying Forward Identity on Live models ("
              << test_count << " states via bin)\n";
    surgery::VerifyB3IdentityBin(live_res.model_src, live_res.model_dst, obs_bin,
                                 test_count, manifest_meta.observation_dim);
    if (ema_res.model_ema_src && ema_res.model_ema_dst) {
      std::cout << "\n[B3 EMA TEST] Verifying Forward Identity on EMA models ("
                << test_count << " states via bin)\n";
      surgery::VerifyB3IdentityBin(ema_res.model_ema_src, ema_res.model_ema_dst,
                                   obs_bin, test_count, manifest_meta.observation_dim);
    }
  }

  // 9. Verify 3c Native Effect Check (Self-test on step-0 checkpoint: all zeros expected)
  std::string meta_jsonl = absl::GetFlag(FLAGS_test_meta_jsonl);
  std::string obs_bin = absl::GetFlag(FLAGS_test_obs_bin);
  std::cout << "\n[3c LIVE NATIVE EFFECT CHECK] Auditing destination live model\n";
  surgery::RunNativeEffectCheck(
      live_res.dst_model_path, obs_bin, meta_jsonl, test_states_json_path,
      test_count, dst_blocks, src_blocks, manifest_meta);
  if (!ema_res.dst_ema_model_path.empty()) {
    std::cout << "\n[3c EMA NATIVE EFFECT CHECK] Auditing destination EMA model\n";
    surgery::RunNativeEffectCheck(
        ema_res.dst_ema_model_path, obs_bin, meta_jsonl, test_states_json_path,
        test_count, dst_blocks, src_blocks, manifest_meta);
  }

  std::cout << "\n======================================================================\n";
  std::cout << "ALL B1, B2, AND B3 SURGERY & IDENTITY VERIFICATIONS PASSED ("
            << src_blocks << " -> " << dst_blocks << ")!\n";
  std::cout << "======================================================================\n";
  return 0;
}
