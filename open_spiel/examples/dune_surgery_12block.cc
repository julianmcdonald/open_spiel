#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <memory>
#include <cmath>
#include <filesystem>
#include <algorithm>

#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"
#include "open_spiel/utils/json.h"
#include "open_spiel/games/dune_imperium/dune_imperium.h"
#include "open_spiel/games/dune_imperium/dune_imperium_common.h"
#include "absl/flags/flag.h"
#include "absl/flags/parse.h"

#include <torch/torch.h>

#include "dune_network.h"
#include "dune_evaluator.h"
#include "dune_semantic_action_scorer.h"
#include "dune_sha256.h"

ABSL_FLAG(std::string, src_model,
          "/run/media/warcr/Storage/dune_drl_runtime/round7/u27900_continuation_20260930/checkpoints/ppo_model_update_28700.pt",
          "Source 8-block model path.");
ABSL_FLAG(std::string, src_optimizer,
          "/run/media/warcr/Storage/dune_drl_runtime/round7/u28700_screen_control_20261001/checkpoints/ppo_optimizer_update_28700.pt",
          "Source 8-block optimizer path.");
ABSL_FLAG(std::string, src_manifest,
          "/run/media/warcr/Storage/dune_drl_runtime/round7/u27900_continuation_20260930/checkpoints/ppo_model_update_28700.json",
          "Source 8-block JSON manifest path.");
ABSL_FLAG(std::string, out_dir,
          "/home/warcr/dune_drl_runtime/round7/u28700_12block_init",
          "Destination directory for 12-block model, optimizer and manifest.");
ABSL_FLAG(std::string, test_obs_bin,
          "/home/warcr/dune_drl_runtime/round7/held_out_u28700/trajectory_obs.bin",
          "Path to stored observation binary for B3 identity verification.");
ABSL_FLAG(int, test_states_count, 1024,
          "Number of stored states to verify in B3 identity check (>= 1000).");
ABSL_FLAG(std::string, check_optimizer, "",
          "Optional path to post-bootstrap optimizer file to verify against surgery output.");

namespace {

void VerifyB2OptimizerMoments(
    const std::shared_ptr<open_spiel::SharedDunePolicyValueNetImpl>& model8,
    torch::optim::AdamW& opt8,
    const std::shared_ptr<open_spiel::SharedDunePolicyValueNetImpl>& model12,
    torch::optim::AdamW& opt12) {
  std::cout << "\n======================================================================\n";
  std::cout << "[B2 TEST] Verifying Optimizer Moments & Step Counters (8 -> 12)\n";
  std::cout << "======================================================================\n";

  auto check_pair = [&](const std::string& name, const torch::Tensor& t8, const torch::Tensor& t12) {
    auto k8 = t8.unsafeGetTensorImpl();
    auto k12 = t12.unsafeGetTensorImpl();
    auto it8 = opt8.state().find(k8);
    auto it12 = opt12.state().find(k12);

    if (it8 == opt8.state().end()) {
      open_spiel::SpielFatalError("Param " + name + " missing in source optimizer state!");
    }
    if (it12 == opt12.state().end()) {
      open_spiel::SpielFatalError("Param " + name + " missing in destination optimizer state!");
    }

    auto* s8 = dynamic_cast<torch::optim::AdamWParamState*>(it8->second.get());
    auto* s12 = dynamic_cast<torch::optim::AdamWParamState*>(it12->second.get());

    if (s8->step() != s12->step()) {
      open_spiel::SpielFatalError("Step mismatch on " + name + ": src=" +
                                  std::to_string(s8->step()) + ", dst=" + std::to_string(s12->step()));
    }
    if (!torch::equal(s8->exp_avg(), s12->exp_avg())) {
      open_spiel::SpielFatalError("exp_avg mismatch on " + name);
    }
    if (!torch::equal(s8->exp_avg_sq(), s12->exp_avg_sq())) {
      open_spiel::SpielFatalError("exp_avg_sq mismatch on " + name);
    }
  };

  // 1. Check Policy Head
  check_pair("policy_head.weight", model8->policy_head->weight, model12->policy_head->weight);
  if (model8->policy_head->bias.defined()) {
    check_pair("policy_head.bias", model8->policy_head->bias, model12->policy_head->bias);
  }

  // 2. Check Input Layer
  check_pair("input_layer.weight", model8->input_layer->weight, model12->input_layer->weight);
  if (model8->input_layer->bias.defined()) {
    check_pair("input_layer.bias", model8->input_layer->bias, model12->input_layer->bias);
  }

  // 3. Check res1 .. res8
  for (int b = 0; b < 8; ++b) {
    std::string prefix = "res" + std::to_string(b + 1) + ".";
    check_pair(prefix + "fc1.weight", model8->res_blocks[b]->fc1->weight, model12->res_blocks[b]->fc1->weight);
    check_pair(prefix + "fc1.bias", model8->res_blocks[b]->fc1->bias, model12->res_blocks[b]->fc1->bias);
    check_pair(prefix + "fc2.weight", model8->res_blocks[b]->fc2->weight, model12->res_blocks[b]->fc2->weight);
    check_pair(prefix + "fc2.bias", model8->res_blocks[b]->fc2->bias, model12->res_blocks[b]->fc2->bias);
    check_pair(prefix + "ln1.weight", model8->res_blocks[b]->ln1->weight, model12->res_blocks[b]->ln1->weight);
    check_pair(prefix + "ln1.bias", model8->res_blocks[b]->ln1->bias, model12->res_blocks[b]->ln1->bias);
    check_pair(prefix + "ln2.weight", model8->res_blocks[b]->ln2->weight, model12->res_blocks[b]->ln2->weight);
    check_pair(prefix + "ln2.bias", model8->res_blocks[b]->ln2->bias, model12->res_blocks[b]->ln2->bias);
  }

  // 4. Check Value Head
  check_pair("value_head.weight", model8->value_head->weight, model12->value_head->weight);
  if (model8->value_head->bias.defined()) {
    check_pair("value_head.bias", model8->value_head->bias, model12->value_head->bias);
  }

  // 5. Check Semantic Scorer
  if (model8->semantic_scorer_ && model12->semantic_scorer_) {
    const auto& params8 = model8->semantic_scorer_->named_parameters();
    const auto& params12 = model12->semantic_scorer_->named_parameters();
    for (const auto& pair : params8) {
      const std::string& key = pair.key();
      check_pair("semantic_scorer." + key, pair.value(), params12[key]);
    }
  }

  std::cout << "[B2 TEST PASS] All 88 source parameters (input, res1-8, value, policy, scorer) matched EXACTLY in exp_avg, exp_avg_sq, and step!\n";

  // 6. Check res9 .. res12 have fresh state
  int fresh_params_checked = 0;
  for (int b = 8; b < 12; ++b) {
    for (const auto& p : model12->res_blocks[b]->parameters()) {
      auto key = p.unsafeGetTensorImpl();
      auto it = opt12.state().find(key);
      if (it == opt12.state().end()) {
        open_spiel::SpielFatalError("New param in res block " + std::to_string(b + 1) + " missing optimizer state!");
      }
      auto* s = dynamic_cast<torch::optim::AdamWParamState*>(it->second.get());
      if (s->step() != 0) {
        open_spiel::SpielFatalError("New param step != 0 (got " + std::to_string(s->step()) + ")");
      }
      if (s->exp_avg().abs().max().item<float>() != 0.0f) {
        open_spiel::SpielFatalError("New param exp_avg is not zero!");
      }
      if (s->exp_avg_sq().abs().max().item<float>() != 0.0f) {
        open_spiel::SpielFatalError("New param exp_avg_sq is not zero!");
      }
      ++fresh_params_checked;
    }
  }
  std::cout << "[B2 TEST PASS] All " << fresh_params_checked
            << " new parameters in res9-res12 verified to have fresh state (step=0, exp_avg=0, exp_avg_sq=0).\n";
}

void VerifyB3Identity(
    const std::shared_ptr<open_spiel::SharedDunePolicyValueNetImpl>& model8,
    const std::shared_ptr<open_spiel::SharedDunePolicyValueNetImpl>& model12,
    const std::string& obs_bin_path,
    int test_states_count) {
  std::cout << "\n======================================================================\n";
  std::cout << "[B3 TEST] Verifying Forward Identity on " << test_states_count << " Stored States\n";
  std::cout << "======================================================================\n";

  std::ifstream obs_file(obs_bin_path, std::ios::binary);
  if (!obs_file.is_open()) {
    open_spiel::SpielFatalError("Cannot open test_obs_bin: " + obs_bin_path);
  }

  const size_t obs_size = open_spiel::dune_imperium::kFullPublicInformationStateSize;
  const size_t byte_stride = obs_size * sizeof(float);
  std::vector<float> obs_buffer(obs_size);

  torch::NoGradGuard no_grad;
  model8->eval();
  model12->eval();

  double max_abs_diff_logits = 0.0;
  double max_abs_diff_value = 0.0;
  double sum_abs_diff_logits = 0.0;
  int64_t total_logit_elements = 0;

  for (int i = 0; i < test_states_count; ++i) {
    obs_file.seekg(i * byte_stride);
    obs_file.read(reinterpret_cast<char*>(obs_buffer.data()), byte_stride);
    if (!obs_file.good()) {
      open_spiel::SpielFatalError("Failed reading observation row " + std::to_string(i));
    }

    torch::Tensor x = torch::from_blob(obs_buffer.data(), {1, static_cast<int64_t>(obs_size)}, torch::kFloat32);

    auto out8 = model8->forward(x);
    auto out12 = model12->forward(x);

    torch::Tensor diff_logits = (out12.logits - out8.logits).abs();
    torch::Tensor diff_val = (out12.values - out8.values).abs();

    double cur_max_l = diff_logits.max().item<double>();
    double cur_max_v = diff_val.max().item<double>();

    if (cur_max_l > max_abs_diff_logits) max_abs_diff_logits = cur_max_l;
    if (cur_max_v > max_abs_diff_value) max_abs_diff_value = cur_max_v;

    sum_abs_diff_logits += diff_logits.sum().item<double>();
    total_logit_elements += diff_logits.numel();
  }

  std::cout << "Tested stored states:     " << test_states_count << "\n";
  std::cout << "Max absolute diff logits: " << max_abs_diff_logits << " (exact 0 in fp32 expected)\n";
  std::cout << "Max absolute diff values: " << max_abs_diff_value << " (exact 0 in fp32 expected)\n";
  std::cout << "Mean absolute diff logits: " << (sum_abs_diff_logits / total_logit_elements) << "\n";

  if (max_abs_diff_logits > 0.0f || max_abs_diff_value > 0.0f) {
    open_spiel::SpielFatalError(absl::StrFormat(
        "B3 Identity check FAILED! Expected bit-exact 0 in fp32, got max_l=%g, max_v=%g",
        max_abs_diff_logits, max_abs_diff_value));
  }
  std::cout << "[B3 TEST PASS] Exact numerical identity proven! max_abs_diff = 0.00000000 in fp32.\n";
}

} // namespace

int main(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);

  std::string src_m_path = absl::GetFlag(FLAGS_src_model);
  std::string src_opt_path = absl::GetFlag(FLAGS_src_optimizer);
  std::string src_json_path = absl::GetFlag(FLAGS_src_manifest);
  std::string out_dir = absl::GetFlag(FLAGS_out_dir);

  std::cout << "======================================================================\n";
  std::cout << "DUNE DRL CAPACITY PROBE: U28700 8 -> 12 RESIDUAL BLOCKS SURGERY TOOL\n";
  std::cout << "======================================================================\n";
  std::cout << "Source model:     " << src_m_path << "\n";
  std::cout << "Source optimizer: " << src_opt_path << "\n";
  std::cout << "Source manifest:  " << src_json_path << "\n";
  std::cout << "Destination dir:  " << out_dir << "\n";

  std::filesystem::create_directories(out_dir);

  torch::Device device(torch::kCPU);

  // 1. Build Source 8-block Model and Load
  std::cout << "\n[B1 SURGERY] Loading source 8-block model...\n";
  auto model8 = std::make_shared<open_spiel::SharedDunePolicyValueNetImpl>(
      open_spiel::dune_imperium::kFullPublicInformationStateSize, 2048, 2391, 8,
      /*nonlinear=*/false, /*aux=*/false, /*seed=*/0, /*scorer=*/true);
  open_spiel::LoadModelCheckpointRobust(model8, src_m_path, device);
  std::cout << "Source 8-block model loaded successfully.\n";

  // 2. Build Destination 12-block Model
  std::cout << "[B1 SURGERY] Constructing destination 12-block model...\n";
  auto model12 = std::make_shared<open_spiel::SharedDunePolicyValueNetImpl>(
      open_spiel::dune_imperium::kFullPublicInformationStateSize, 2048, 2391, 12,
      /*nonlinear=*/false, /*aux=*/false, /*seed=*/0, /*scorer=*/true);

  // 3. Copy layers exactly
  {
    torch::NoGradGuard no_grad;
    // Input layer
    model12->input_layer->weight.copy_(model8->input_layer->weight);
    if (model8->input_layer->bias.defined()) {
      model12->input_layer->bias.copy_(model8->input_layer->bias);
    }

    // res1 .. res8
    for (int b = 0; b < 8; ++b) {
      model12->res_blocks[b]->fc1->weight.copy_(model8->res_blocks[b]->fc1->weight);
      model12->res_blocks[b]->fc1->bias.copy_(model8->res_blocks[b]->fc1->bias);
      model12->res_blocks[b]->fc2->weight.copy_(model8->res_blocks[b]->fc2->weight);
      model12->res_blocks[b]->fc2->bias.copy_(model8->res_blocks[b]->fc2->bias);
      model12->res_blocks[b]->ln1->weight.copy_(model8->res_blocks[b]->ln1->weight);
      model12->res_blocks[b]->ln1->bias.copy_(model8->res_blocks[b]->ln1->bias);
      model12->res_blocks[b]->ln2->weight.copy_(model8->res_blocks[b]->ln2->weight);
      model12->res_blocks[b]->ln2->bias.copy_(model8->res_blocks[b]->ln2->bias);
    }

    // Append res9 .. res12:
    // fc1, fc2, ln1 remain at trainer's normal initialization.
    // ln2 weight = 0, ln2 bias = 0.
    for (int b = 8; b < 12; ++b) {
      model12->res_blocks[b]->ln2->weight.zero_();
      model12->res_blocks[b]->ln2->bias.zero_();
    }

    // Policy head
    model12->policy_head->weight.copy_(model8->policy_head->weight);
    if (model8->policy_head->bias.defined()) {
      model12->policy_head->bias.copy_(model8->policy_head->bias);
    }

    // Value head
    model12->value_head->weight.copy_(model8->value_head->weight);
    if (model8->value_head->bias.defined()) {
      model12->value_head->bias.copy_(model8->value_head->bias);
    }

    // Semantic scorer
    if (model8->semantic_scorer_ && model12->semantic_scorer_) {
      for (const auto& pair : model8->semantic_scorer_->named_parameters()) {
        model12->semantic_scorer_->named_parameters()[pair.key()].copy_(pair.value());
      }
      for (const auto& pair : model8->semantic_scorer_->named_buffers()) {
        model12->semantic_scorer_->named_buffers()[pair.key()].copy_(pair.value());
      }
    }
  }
  std::cout << "[B1 SURGERY] Layers copied: input_layer, res1-8, policy_head, value_head, semantic_scorer.\n";
  std::cout << "[B1 SURGERY] res9-res12 appended with normal init on fc1/fc2/ln1, and ln2 weight=0, bias=0.\n";

  // 4. Save 12-block Model
  std::string dst_model_path = out_dir + "/ppo_model_update_28700.pt";
  torch::save(model12, dst_model_path);
  size_t dst_m_size = 0;
  std::string dst_m_sha = open_spiel::ComputeFileSHA256(dst_model_path, &dst_m_size);
  std::cout << "[B1 SURGERY] Saved 12-block model: " << dst_model_path << " (" << dst_m_size << " bytes, SHA=" << dst_m_sha << ")\n";

  // 5. Optimizer Migration (8 -> 12)
  std::cout << "\n[B2 OPTIMIZER] Setting up 12-block AdamW optimizer...\n";
  std::vector<torch::Tensor> opt_policy_params;
  opt_policy_params.push_back(model12->policy_head->weight);
  if (model12->policy_head->bias.defined()) opt_policy_params.push_back(model12->policy_head->bias);

  std::vector<torch::Tensor> opt_torso_params;
  opt_torso_params.push_back(model12->input_layer->weight);
  if (model12->input_layer->bias.defined()) opt_torso_params.push_back(model12->input_layer->bias);
  for (int b = 0; b < 12; ++b) {
    for (auto& p : model12->res_blocks[b]->parameters()) opt_torso_params.push_back(p);
  }
  for (auto& p : model12->value_head->parameters()) opt_torso_params.push_back(p);
  if (model12->with_semantic_scorer_ && model12->semantic_scorer_) {
    auto add_mod_params = [&](auto& mod) {
      for (auto& p : mod->parameters()) opt_torso_params.push_back(p);
    };
    add_mod_params(model12->semantic_scorer_->card_embedding);
    add_mod_params(model12->semantic_scorer_->space_embedding);
    add_mod_params(model12->semantic_scorer_->desc_proj);
    add_mod_params(model12->semantic_scorer_->desc_ln);
    add_mod_params(model12->semantic_scorer_->trunk_proj);
    add_mod_params(model12->semantic_scorer_->trunk_ln);
    add_mod_params(model12->semantic_scorer_->mlp1);
    add_mod_params(model12->semantic_scorer_->mlp1_ln);
    add_mod_params(model12->semantic_scorer_->out_layer);
    add_mod_params(model12->semantic_scorer_->out_layer_ext);
  }

  std::vector<torch::optim::OptimizerParamGroup> opt_groups;
  opt_groups.emplace_back(opt_policy_params);
  opt_groups.emplace_back(opt_torso_params);

  torch::optim::AdamW opt12(opt_groups, torch::optim::AdamWOptions(5e-6).eps(1e-5));

  bool migrated = open_spiel::LoadOptimizerCheckpointMigrating(model12, opt12, src_opt_path, device, 5e-6);
  if (!migrated) {
    open_spiel::SpielFatalError("LoadOptimizerCheckpointMigrating failed!");
  }

  std::string dst_opt_path = out_dir + "/ppo_optimizer_update_28700.pt";
  torch::save(opt12, dst_opt_path);
  size_t dst_opt_size = 0;
  std::string dst_opt_sha = open_spiel::ComputeFileSHA256(dst_opt_path, &dst_opt_size);
  std::cout << "[B2 OPTIMIZER] Saved migrated 12-block optimizer: " << dst_opt_path
            << " (" << dst_opt_size << " bytes, SHA=" << dst_opt_sha << ")\n";

  // 6. Update JSON Manifest
  std::cout << "\n[B1 MANIFEST] Updating JSON manifest...\n";
  std::ifstream src_jf(src_json_path);
  if (!src_jf.is_open()) {
    open_spiel::SpielFatalError("Cannot open src_manifest: " + src_json_path);
  }
  std::string j_content((std::istreambuf_iterator<char>(src_jf)), std::istreambuf_iterator<char>());
  src_jf.close();

  auto j_opt = open_spiel::json::FromString(j_content);
  if (!j_opt) open_spiel::SpielFatalError("Failed parsing source manifest JSON");
  auto j_obj = j_opt->GetObject();

  j_obj["num_blocks"] = open_spiel::json::Value(static_cast<int64_t>(12));
  j_obj["model_file_size"] = open_spiel::json::Value(static_cast<int64_t>(dst_m_size));
  j_obj["model_sha256"] = open_spiel::json::Value(dst_m_sha);
  j_obj["optimizer_file_size"] = open_spiel::json::Value(static_cast<int64_t>(dst_opt_size));
  j_obj["optimizer_sha256"] = open_spiel::json::Value(dst_opt_sha);

  std::string dst_json_path = out_dir + "/ppo_model_update_28700.json";
  std::ofstream out_jf(dst_json_path);
  out_jf << open_spiel::json::ToString(open_spiel::json::Value(j_obj));
  out_jf.close();
  std::cout << "[B1 MANIFEST] Saved updated manifest with num_blocks=12: " << dst_json_path << "\n";

  // 7. Verify B2 Optimizer Moments
  // To verify B2, also load the source optimizer into an 8-block AdamW
  std::vector<torch::Tensor> src_pol_params;
  src_pol_params.push_back(model8->policy_head->weight);
  if (model8->policy_head->bias.defined()) src_pol_params.push_back(model8->policy_head->bias);

  std::vector<torch::Tensor> src_tor_params;
  src_tor_params.push_back(model8->input_layer->weight);
  if (model8->input_layer->bias.defined()) src_tor_params.push_back(model8->input_layer->bias);
  for (int b = 0; b < 8; ++b) {
    for (auto& p : model8->res_blocks[b]->parameters()) src_tor_params.push_back(p);
  }
  for (auto& p : model8->value_head->parameters()) src_tor_params.push_back(p);
  if (model8->with_semantic_scorer_ && model8->semantic_scorer_) {
    auto add_mod_params8 = [&](auto& mod) {
      for (auto& p : mod->parameters()) src_tor_params.push_back(p);
    };
    add_mod_params8(model8->semantic_scorer_->card_embedding);
    add_mod_params8(model8->semantic_scorer_->space_embedding);
    add_mod_params8(model8->semantic_scorer_->desc_proj);
    add_mod_params8(model8->semantic_scorer_->desc_ln);
    add_mod_params8(model8->semantic_scorer_->trunk_proj);
    add_mod_params8(model8->semantic_scorer_->trunk_ln);
    add_mod_params8(model8->semantic_scorer_->mlp1);
    add_mod_params8(model8->semantic_scorer_->mlp1_ln);
    add_mod_params8(model8->semantic_scorer_->out_layer);
    add_mod_params8(model8->semantic_scorer_->out_layer_ext);
  }
  std::vector<torch::optim::OptimizerParamGroup> src_groups;
  src_groups.emplace_back(src_pol_params);
  src_groups.emplace_back(src_tor_params);
  torch::optim::AdamW opt8(src_groups, torch::optim::AdamWOptions(5e-6).eps(1e-5));
  torch::load(opt8, src_opt_path, device);

  std::cout << "\n[B2 TEST STAGE (a)] Verifying in-memory optimizer moments...\n";
  VerifyB2OptimizerMoments(model8, opt8, model12, opt12);

  // 7b. Verify B2 Optimizer Moments Post-Serialization (Bootstrap Path)
  std::cout << "\n[B2 TEST STAGE (b)] Verifying post-serialization optimizer reload via bootstrap path...\n";
  auto model12_loaded = std::make_shared<open_spiel::SharedDunePolicyValueNetImpl>(
      open_spiel::dune_imperium::kFullPublicInformationStateSize, 2048, 2391, 12,
      /*nonlinear=*/false, /*aux=*/false, /*seed=*/0, /*scorer=*/true);
  open_spiel::LoadModelCheckpointRobust(model12_loaded, dst_model_path, device);

  std::vector<torch::Tensor> loaded_pol_params;
  loaded_pol_params.push_back(model12_loaded->policy_head->weight);
  if (model12_loaded->policy_head->bias.defined()) loaded_pol_params.push_back(model12_loaded->policy_head->bias);

  std::vector<torch::Tensor> loaded_tor_params;
  loaded_tor_params.push_back(model12_loaded->input_layer->weight);
  if (model12_loaded->input_layer->bias.defined()) loaded_tor_params.push_back(model12_loaded->input_layer->bias);
  for (int b = 0; b < 12; ++b) {
    for (auto& p : model12_loaded->res_blocks[b]->parameters()) loaded_tor_params.push_back(p);
  }
  for (auto& p : model12_loaded->value_head->parameters()) loaded_tor_params.push_back(p);
  if (model12_loaded->with_semantic_scorer_ && model12_loaded->semantic_scorer_) {
    auto add_mod_params_loaded = [&](auto& mod) {
      for (auto& p : mod->parameters()) loaded_tor_params.push_back(p);
    };
    add_mod_params_loaded(model12_loaded->semantic_scorer_->card_embedding);
    add_mod_params_loaded(model12_loaded->semantic_scorer_->space_embedding);
    add_mod_params_loaded(model12_loaded->semantic_scorer_->desc_proj);
    add_mod_params_loaded(model12_loaded->semantic_scorer_->desc_ln);
    add_mod_params_loaded(model12_loaded->semantic_scorer_->trunk_proj);
    add_mod_params_loaded(model12_loaded->semantic_scorer_->trunk_ln);
    add_mod_params_loaded(model12_loaded->semantic_scorer_->mlp1);
    add_mod_params_loaded(model12_loaded->semantic_scorer_->mlp1_ln);
    add_mod_params_loaded(model12_loaded->semantic_scorer_->out_layer);
    add_mod_params_loaded(model12_loaded->semantic_scorer_->out_layer_ext);
  }

  std::vector<torch::optim::OptimizerParamGroup> loaded_groups;
  loaded_groups.emplace_back(loaded_pol_params);
  loaded_groups.emplace_back(loaded_tor_params);
  torch::optim::AdamW opt12_loaded(loaded_groups, torch::optim::AdamWOptions(5e-6).eps(1e-5));

  std::string opt_to_check = dst_opt_path;
  std::string flag_check_opt = absl::GetFlag(FLAGS_check_optimizer);
  if (!flag_check_opt.empty()) {
    opt_to_check = flag_check_opt;
  }
  std::cout << "[B2 TEST STAGE (b)] Loading optimizer from: " << opt_to_check << "\n";

  // Replicate exact bootstrap startup logic in dune_ppo_train.cc:
  if (!open_spiel::LoadOptimizerCheckpointMigrating(model12_loaded, opt12_loaded, opt_to_check, device, 5e-6)) {
    torch::load(opt12_loaded, opt_to_check, device);
  }

  VerifyB2OptimizerMoments(model8, opt8, model12_loaded, opt12_loaded);
  std::cout << "[B2 TEST PASS] Post-serialization bootstrap reload verified bit-exact on " << opt_to_check << "!\n";

  // 8. Verify B3 Identity Checks
  std::string obs_bin = absl::GetFlag(FLAGS_test_obs_bin);
  int test_count = absl::GetFlag(FLAGS_test_states_count);
  VerifyB3Identity(model8, model12, obs_bin, test_count);

  std::cout << "\n======================================================================\n";
  std::cout << "ALL B1, B2, AND B3 SURGERY & IDENTITY VERIFICATIONS PASSED!\n";
  std::cout << "======================================================================\n";
  return 0;
}
