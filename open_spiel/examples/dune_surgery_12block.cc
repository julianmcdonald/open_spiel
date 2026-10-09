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
          "/run/media/warcr/Storage/dune_drl_runtime/round7/u30650_12block_random_draft_overnight_20261006/checkpoints/ppo_model_update_31400.pt",
          "Source model path.");
ABSL_FLAG(std::string, src_optimizer,
          "/run/media/warcr/Storage/dune_drl_runtime/round7/u30650_12block_random_draft_overnight_20261006/checkpoints/ppo_optimizer_update_31400.pt",
          "Source optimizer path.");
ABSL_FLAG(std::string, src_manifest,
          "/run/media/warcr/Storage/dune_drl_runtime/round7/u30650_12block_random_draft_overnight_20261006/checkpoints/ppo_model_update_31400.json",
          "Source JSON manifest path.");
ABSL_FLAG(std::string, out_dir,
          "/home/warcr/dune_drl_runtime/round7/u31400_16block_init",
          "Destination directory for 16-block model, optimizer and manifest.");
ABSL_FLAG(int, src_blocks, 12, "Number of blocks in source model.");
ABSL_FLAG(int, dst_blocks, 16, "Number of blocks in destination model.");
ABSL_FLAG(int, new_block_first_index, 8, "First block index for higher learning rate group.");
ABSL_FLAG(double, base_lr, 5e-6, "Base learning rate.");
ABSL_FLAG(double, new_block_lr, 5e-5, "Learning rate for new blocks group.");
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

namespace {

void VerifyB2OptimizerMoments(
    const std::shared_ptr<open_spiel::SharedDunePolicyValueNetImpl>& model_src,
    torch::optim::AdamW& opt_src,
    const std::shared_ptr<open_spiel::SharedDunePolicyValueNetImpl>& model_dst,
    torch::optim::AdamW& opt_dst,
    int src_blocks = 12,
    int dst_blocks = 16) {
  std::cout << "\n======================================================================\n";
  std::cout << "[B2 TEST] Verifying Optimizer Moments & Step Counters ("
            << src_blocks << " -> " << dst_blocks << ")\n";
  std::cout << "======================================================================\n";

  auto check_pair = [&](const std::string& name, const torch::Tensor& t_src, const torch::Tensor& t_dst) {
    auto k_src = t_src.unsafeGetTensorImpl();
    auto k_dst = t_dst.unsafeGetTensorImpl();
    auto it_src = opt_src.state().find(k_src);
    auto it_dst = opt_dst.state().find(k_dst);

    if (it_src == opt_src.state().end()) {
      open_spiel::SpielFatalError("Param " + name + " missing in source optimizer state!");
    }
    if (it_dst == opt_dst.state().end()) {
      open_spiel::SpielFatalError("Param " + name + " missing in destination optimizer state!");
    }

    auto* s_src = dynamic_cast<torch::optim::AdamWParamState*>(it_src->second.get());
    auto* s_dst = dynamic_cast<torch::optim::AdamWParamState*>(it_dst->second.get());

    if (s_src->step() != s_dst->step()) {
      open_spiel::SpielFatalError("Step mismatch on " + name + ": src=" +
                                  std::to_string(s_src->step()) + ", dst=" + std::to_string(s_dst->step()));
    }
    if (!torch::equal(s_src->exp_avg(), s_dst->exp_avg())) {
      open_spiel::SpielFatalError("exp_avg mismatch on " + name);
    }
    if (!torch::equal(s_src->exp_avg_sq(), s_dst->exp_avg_sq())) {
      open_spiel::SpielFatalError("exp_avg_sq mismatch on " + name);
    }
  };

  int inherited_params_checked = 0;
  // 1. Check Policy Head
  check_pair("policy_head.weight", model_src->policy_head->weight, model_dst->policy_head->weight);
  ++inherited_params_checked;
  if (model_src->policy_head->bias.defined()) {
    check_pair("policy_head.bias", model_src->policy_head->bias, model_dst->policy_head->bias);
    ++inherited_params_checked;
  }

  // 2. Check Input Layer
  check_pair("input_layer.weight", model_src->input_layer->weight, model_dst->input_layer->weight);
  ++inherited_params_checked;
  if (model_src->input_layer->bias.defined()) {
    check_pair("input_layer.bias", model_src->input_layer->bias, model_dst->input_layer->bias);
    ++inherited_params_checked;
  }

  // 3. Check res1 .. res[src_blocks]
  for (int b = 0; b < src_blocks; ++b) {
    std::string prefix = "res" + std::to_string(b + 1) + ".";
    check_pair(prefix + "fc1.weight", model_src->res_blocks[b]->fc1->weight, model_dst->res_blocks[b]->fc1->weight);
    check_pair(prefix + "fc1.bias", model_src->res_blocks[b]->fc1->bias, model_dst->res_blocks[b]->fc1->bias);
    check_pair(prefix + "fc2.weight", model_src->res_blocks[b]->fc2->weight, model_dst->res_blocks[b]->fc2->weight);
    check_pair(prefix + "fc2.bias", model_src->res_blocks[b]->fc2->bias, model_dst->res_blocks[b]->fc2->bias);
    check_pair(prefix + "ln1.weight", model_src->res_blocks[b]->ln1->weight, model_dst->res_blocks[b]->ln1->weight);
    check_pair(prefix + "ln1.bias", model_src->res_blocks[b]->ln1->bias, model_dst->res_blocks[b]->ln1->bias);
    check_pair(prefix + "ln2.weight", model_src->res_blocks[b]->ln2->weight, model_dst->res_blocks[b]->ln2->weight);
    check_pair(prefix + "ln2.bias", model_src->res_blocks[b]->ln2->bias, model_dst->res_blocks[b]->ln2->bias);
    inherited_params_checked += 8;
  }

  // 4. Check Value Head
  check_pair("value_head.weight", model_src->value_head->weight, model_dst->value_head->weight);
  ++inherited_params_checked;
  if (model_src->value_head->bias.defined()) {
    check_pair("value_head.bias", model_src->value_head->bias, model_dst->value_head->bias);
    ++inherited_params_checked;
  }

  // 5. Check Semantic Scorer
  if (model_src->semantic_scorer_ && model_dst->semantic_scorer_) {
    const auto& params_src = model_src->semantic_scorer_->named_parameters();
    const auto& params_dst = model_dst->semantic_scorer_->named_parameters();
    for (const auto& pair : params_src) {
      const std::string& key = pair.key();
      check_pair("semantic_scorer." + key, pair.value(), params_dst[key]);
      ++inherited_params_checked;
    }
  }

  std::cout << "[B2 TEST PASS] All " << inherited_params_checked
            << " source parameters (input, res1-" << src_blocks
            << ", value, policy, scorer) matched EXACTLY in exp_avg, exp_avg_sq, and step!\n";

  // 6. Check res[src_blocks+1] .. res[dst_blocks] have fresh state
  int fresh_params_checked = 0;
  for (int b = src_blocks; b < dst_blocks; ++b) {
    for (const auto& p : model_dst->res_blocks[b]->parameters()) {
      auto key = p.unsafeGetTensorImpl();
      auto it = opt_dst.state().find(key);
      if (it == opt_dst.state().end()) {
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
            << " new parameters in res" << (src_blocks + 1) << "-res" << dst_blocks
            << " verified to have fresh state (step=0, exp_avg=0, exp_avg_sq=0).\n";
}

void VerifyB3Identity(
    const std::shared_ptr<open_spiel::SharedDunePolicyValueNetImpl>& model_src,
    const std::shared_ptr<open_spiel::SharedDunePolicyValueNetImpl>& model_dst,
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
  model_src->eval();
  model_dst->eval();

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

    auto out_src = model_src->forward(x);
    auto out_dst = model_dst->forward(x);

    torch::Tensor diff_logits = (out_dst.logits - out_src.logits).abs();
    torch::Tensor diff_val = (out_dst.values - out_src.values).abs();

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

void RunNativeEffectCheck(
    const std::string& model_path,
    const std::string& obs_bin_path,
    const std::string& meta_jsonl_path,
    int test_states_count,
    int num_blocks = 16,
    int bypass_first_index = 12) {
  std::cout << "\n======================================================================\n";
  std::cout << "[3c NATIVE EFFECT CHECK] Auditing " << num_blocks << "-block model on "
            << test_states_count << " Stored States\n";
  std::cout << "Bypassing new residual blocks indices [" << bypass_first_index << ".." << (num_blocks - 1)
            << "] (res" << (bypass_first_index + 1) << "-res" << num_blocks << ")\n";
  std::cout << "Model: " << model_path << "\n";
  std::cout << "Obs:   " << obs_bin_path << "\n";
  std::cout << "Meta:  " << meta_jsonl_path << "\n";
  std::cout << "======================================================================\n";

  torch::Device device(torch::kCPU);

  // 1. Load active model
  auto model_active = std::make_shared<open_spiel::SharedDunePolicyValueNetImpl>(
      open_spiel::dune_imperium::kFullPublicInformationStateSize, 2048, 2391, num_blocks,
      /*nonlinear=*/false, /*aux=*/false, /*seed=*/0, /*scorer=*/true);
  open_spiel::LoadModelCheckpointRobust(model_active, model_path, device);
  model_active->eval();

  // 2. Clone active model and zero ln2 weight/bias ONLY on bypassed new blocks [bypass_first_index..num_blocks-1]
  auto model_zeroed = std::make_shared<open_spiel::SharedDunePolicyValueNetImpl>(
      open_spiel::dune_imperium::kFullPublicInformationStateSize, 2048, 2391, num_blocks,
      /*nonlinear=*/false, /*aux=*/false, /*seed=*/0, /*scorer=*/true);
  {
    torch::NoGradGuard no_grad;
    for (const auto& pair : model_active->named_parameters()) {
      model_zeroed->named_parameters()[pair.key()].copy_(pair.value());
    }
    for (const auto& pair : model_active->named_buffers()) {
      model_zeroed->named_buffers()[pair.key()].copy_(pair.value());
    }
    for (int b = bypass_first_index; b < num_blocks; ++b) {
      model_zeroed->res_blocks[b]->ln2->weight.zero_();
      model_zeroed->res_blocks[b]->ln2->bias.zero_();
    }
  }
  model_zeroed->eval();

  // 3. Open observation binary and meta jsonl
  std::ifstream obs_file(obs_bin_path, std::ios::binary);
  if (!obs_file.is_open()) {
    open_spiel::SpielFatalError("Cannot open test_obs_bin: " + obs_bin_path);
  }
  std::ifstream meta_file(meta_jsonl_path);
  if (!meta_file.is_open()) {
    open_spiel::SpielFatalError("Cannot open test_meta_jsonl: " + meta_jsonl_path);
  }

  auto game = open_spiel::LoadGame("dune_imperium");

  const size_t obs_size = open_spiel::dune_imperium::kFullPublicInformationStateSize;
  const size_t byte_stride = obs_size * sizeof(float);
  std::vector<float> obs_buffer(obs_size);

  torch::NoGradGuard no_grad;

  double max_noise_legal_logits = 0.0;
  double sum_noise_legal_logits = 0.0;
  double max_noise_val = 0.0;
  double sum_noise_val = 0.0;

  double max_diff_legal_logits = 0.0;
  double sum_diff_legal_logits = 0.0;
  double max_diff_val = 0.0;
  double sum_diff_val = 0.0;

  int64_t total_legal_decisions = 0;
  int states_evaluated = 0;

  for (int i = 0; i < 20000 && states_evaluated < test_states_count; ++i) {
    obs_file.seekg(i * byte_stride);
    obs_file.read(reinterpret_cast<char*>(obs_buffer.data()), byte_stride);
    if (!obs_file.good()) {
      break;
    }

    std::string meta_line;
    if (!std::getline(meta_file, meta_line) || meta_line.empty()) {
      break;
    }

    auto meta_json = open_spiel::json::FromString(meta_line);
    if (!meta_json.has_value() || !meta_json->IsObject()) {
      continue;
    }
    const auto& m_obj = meta_json->GetObject();

    std::vector<open_spiel::Action> action_history;
    auto it_act = m_obj.find("action_history");
    if (it_act != m_obj.end() && it_act->second.IsArray()) {
      for (const auto& v : it_act->second.GetArray()) {
        action_history.push_back(static_cast<open_spiel::Action>(v.GetInt()));
      }
    }

    std::vector<open_spiel::Action> legal_actions;
    auto it_leg = m_obj.find("legal_actions");
    if (it_leg != m_obj.end() && it_leg->second.IsArray()) {
      for (const auto& v : it_leg->second.GetArray()) {
        legal_actions.push_back(static_cast<open_spiel::Action>(v.GetInt()));
      }
    }

    // Replay action history to reconstruct game state
    auto state = game->NewInitialState();
    bool replay_ok = true;
    for (open_spiel::Action a : action_history) {
      auto leg = state->LegalActions();
      if (std::find(leg.begin(), leg.end(), a) == leg.end()) {
        replay_ok = false;
        break;
      }
      state->ApplyAction(a);
    }
    if (!replay_ok) {
      continue;
    }
    const auto* dune_state = dynamic_cast<const open_spiel::dune_imperium::DuneImperiumState*>(state.get());
    if (!dune_state) {
      continue;
    }

    torch::Tensor x = torch::from_blob(obs_buffer.data(), {1, static_cast<int64_t>(obs_size)}, torch::kFloat32);

    // Pass 1: active model
    auto out_act = model_active->forward(x);
    torch::Tensor logits_act = out_act.logits.clone();
    if (model_active->with_semantic_scorer_ && model_active->semantic_scorer_) {
      open_spiel::dune_semantic::CandidateActionData cand_data;
      open_spiel::dune_semantic::ExtractCandidateDescriptors(
          *dune_state, legal_actions, &cand_data, model_active->semantic_descriptor_schema_);
      std::vector<const open_spiel::dune_semantic::CandidateActionData*> batch_cands = {&cand_data};
      open_spiel::dune_semantic::ApplySemanticScorerBatch(
          model_active->semantic_scorer_, out_act.trunk, batch_cands, logits_act, device);
    }

    // Pass 2: repeat active model (noise floor)
    auto out_rep = model_active->forward(x);
    torch::Tensor logits_rep = out_rep.logits.clone();
    if (model_active->with_semantic_scorer_ && model_active->semantic_scorer_) {
      open_spiel::dune_semantic::CandidateActionData cand_data;
      open_spiel::dune_semantic::ExtractCandidateDescriptors(
          *dune_state, legal_actions, &cand_data, model_active->semantic_descriptor_schema_);
      std::vector<const open_spiel::dune_semantic::CandidateActionData*> batch_cands = {&cand_data};
      open_spiel::dune_semantic::ApplySemanticScorerBatch(
          model_active->semantic_scorer_, out_rep.trunk, batch_cands, logits_rep, device);
    }

    // Pass 3: zeroed model (ln2 reset to 0)
    auto out_zero = model_zeroed->forward(x);
    torch::Tensor logits_zero = out_zero.logits.clone();
    if (model_zeroed->with_semantic_scorer_ && model_zeroed->semantic_scorer_) {
      open_spiel::dune_semantic::CandidateActionData cand_data;
      open_spiel::dune_semantic::ExtractCandidateDescriptors(
          *dune_state, legal_actions, &cand_data, model_zeroed->semantic_descriptor_schema_);
      std::vector<const open_spiel::dune_semantic::CandidateActionData*> batch_cands = {&cand_data};
      open_spiel::dune_semantic::ApplySemanticScorerBatch(
          model_zeroed->semantic_scorer_, out_zero.trunk, batch_cands, logits_zero, device);
    }

    // Measure noise floor
    double val_noise = (out_act.values - out_rep.values).abs().item<double>();
    if (val_noise > max_noise_val) max_noise_val = val_noise;
    sum_noise_val += val_noise;

    for (open_spiel::Action a : legal_actions) {
      double l_noise = std::abs(logits_act[0][a].item<double>() - logits_rep[0][a].item<double>());
      if (l_noise > max_noise_legal_logits) max_noise_legal_logits = l_noise;
      sum_noise_legal_logits += l_noise;
    }

    // Measure native effect
    double val_diff = (out_act.values - out_zero.values).abs().item<double>();
    if (val_diff > max_diff_val) max_diff_val = val_diff;
    sum_diff_val += val_diff;

    for (open_spiel::Action a : legal_actions) {
      double l_diff = std::abs(logits_act[0][a].item<double>() - logits_zero[0][a].item<double>());
      if (l_diff > max_diff_legal_logits) max_diff_legal_logits = l_diff;
      sum_diff_legal_logits += l_diff;
    }

    total_legal_decisions += legal_actions.size();
    ++states_evaluated;
  }

  std::cout << absl::StrFormat("Tested stored states:           %d\n", states_evaluated);
  std::cout << absl::StrFormat("Total legal decisions:          %lld\n", total_legal_decisions);
  std::cout << absl::StrFormat("Noise floor (repeat pass):\n");
  std::cout << absl::StrFormat("  Max |Δ legal logits|:         %.6e\n", max_noise_legal_logits);
  std::cout << absl::StrFormat("  Mean |Δ legal logits|:        %.6e\n", total_legal_decisions > 0 ? (sum_noise_legal_logits / total_legal_decisions) : 0.0);
  std::cout << absl::StrFormat("  Max |Δ value|:                %.6e\n", max_noise_val);
  std::cout << absl::StrFormat("  Mean |Δ value|:               %.6e\n", states_evaluated > 0 ? (sum_noise_val / states_evaluated) : 0.0);
  std::cout << absl::StrFormat("Native effect (checkpoint vs res%d-res%d ln2 zeroed):\n", bypass_first_index + 1, num_blocks);
  std::cout << absl::StrFormat("  Max |Δ legal logits|:         %.6e\n", max_diff_legal_logits);
  std::cout << absl::StrFormat("  Mean |Δ legal logits|:        %.6e\n", total_legal_decisions > 0 ? (sum_diff_legal_logits / total_legal_decisions) : 0.0);
  std::cout << absl::StrFormat("  Max |Δ value|:                %.6e\n", max_diff_val);
  std::cout << absl::StrFormat("  Mean |Δ value|:               %.6e\n", states_evaluated > 0 ? (sum_diff_val / states_evaluated) : 0.0);
  std::cout << "======================================================================\n";
}

void RunOptimizerEqualityCheck(
    const std::string& src_opt_path,
    const std::string& check_opt_path,
    const std::string& model_dst_path,
    const std::string& model_src_path,
    int src_blocks = 12,
    int dst_blocks = 16,
    int nb_first_idx = 8,
    double base_lr = 5e-6,
    double nb_lr = 5e-5) {
  std::cout << "======================================================================\n";
  std::cout << "[3b OPTIMIZER EQUALITY CHECK] Name-Based State Verification ("
            << src_blocks << " -> " << dst_blocks << ")\n";
  std::cout << "Source optimizer:    " << src_opt_path << "\n";
  std::cout << "Candidate optimizer: " << check_opt_path << "\n";
  std::cout << "Destination model:   " << model_dst_path << "\n";
  std::cout << "Source model:        " << model_src_path << "\n";
  std::cout << "======================================================================\n";

  torch::Device device(torch::kCPU);

  // 1. Load source model and optimizer
  auto model_src = std::make_shared<open_spiel::SharedDunePolicyValueNetImpl>(
      open_spiel::dune_imperium::kFullPublicInformationStateSize, 2048, 2391, src_blocks,
      /*nonlinear=*/false, /*aux=*/false, /*seed=*/0, /*scorer=*/true);
  open_spiel::LoadModelCheckpointRobust(model_src, model_src_path, device);

  std::vector<torch::Tensor> src_pol_params;
  src_pol_params.push_back(model_src->policy_head->weight);
  if (model_src->policy_head->bias.defined()) src_pol_params.push_back(model_src->policy_head->bias);

  std::vector<torch::Tensor> src_tor_params;
  src_tor_params.push_back(model_src->input_layer->weight);
  if (model_src->input_layer->bias.defined()) src_tor_params.push_back(model_src->input_layer->bias);
  for (int b = 0; b < nb_first_idx; ++b) {
    for (auto& p : model_src->res_blocks[b]->parameters()) src_tor_params.push_back(p);
  }
  for (auto& p : model_src->value_head->parameters()) src_tor_params.push_back(p);
  if (model_src->with_semantic_scorer_ && model_src->semantic_scorer_) {
    auto add_mod = [&](auto& mod) { for (auto& p : mod->parameters()) src_tor_params.push_back(p); };
    add_mod(model_src->semantic_scorer_->card_embedding);
    add_mod(model_src->semantic_scorer_->space_embedding);
    add_mod(model_src->semantic_scorer_->desc_proj);
    add_mod(model_src->semantic_scorer_->desc_ln);
    add_mod(model_src->semantic_scorer_->trunk_proj);
    add_mod(model_src->semantic_scorer_->trunk_ln);
    add_mod(model_src->semantic_scorer_->mlp1);
    add_mod(model_src->semantic_scorer_->mlp1_ln);
    add_mod(model_src->semantic_scorer_->out_layer);
    add_mod(model_src->semantic_scorer_->out_layer_ext);
  }

  std::vector<torch::Tensor> src_nb_params;
  for (int b = nb_first_idx; b < src_blocks; ++b) {
    for (auto& p : model_src->res_blocks[b]->parameters()) src_nb_params.push_back(p);
  }

  std::vector<torch::optim::OptimizerParamGroup> src_groups;
  src_groups.emplace_back(src_pol_params, std::make_unique<torch::optim::AdamWOptions>(torch::optim::AdamWOptions(base_lr).eps(1e-5).weight_decay(0.0)));
  src_groups.emplace_back(src_tor_params, std::make_unique<torch::optim::AdamWOptions>(torch::optim::AdamWOptions(base_lr).eps(1e-5).weight_decay(0.0)));
  if (!src_nb_params.empty()) {
    src_groups.emplace_back(src_nb_params, std::make_unique<torch::optim::AdamWOptions>(torch::optim::AdamWOptions(nb_lr).eps(1e-5).weight_decay(0.0)));
  }
  torch::optim::AdamW opt_src(src_groups, torch::optim::AdamWOptions(base_lr).eps(1e-5));
  torch::load(opt_src, src_opt_path, device);

  // 2. Load destination model
  auto model_dst = std::make_shared<open_spiel::SharedDunePolicyValueNetImpl>(
      open_spiel::dune_imperium::kFullPublicInformationStateSize, 2048, 2391, dst_blocks,
      /*nonlinear=*/false, /*aux=*/false, /*seed=*/0, /*scorer=*/true);
  open_spiel::LoadModelCheckpointRobust(model_dst, model_dst_path, device);

  // 3. Construct destination optimizer
  std::vector<torch::Tensor> dst_pol_params;
  dst_pol_params.push_back(model_dst->policy_head->weight);
  if (model_dst->policy_head->bias.defined()) dst_pol_params.push_back(model_dst->policy_head->bias);

  std::vector<torch::Tensor> dst_tor_params;
  dst_tor_params.push_back(model_dst->input_layer->weight);
  if (model_dst->input_layer->bias.defined()) dst_tor_params.push_back(model_dst->input_layer->bias);
  for (int b = 0; b < nb_first_idx; ++b) {
    for (auto& p : model_dst->res_blocks[b]->parameters()) dst_tor_params.push_back(p);
  }
  for (auto& p : model_dst->value_head->parameters()) dst_tor_params.push_back(p);
  if (model_dst->with_semantic_scorer_ && model_dst->semantic_scorer_) {
    auto add_mod = [&](auto& mod) { for (auto& p : mod->parameters()) dst_tor_params.push_back(p); };
    add_mod(model_dst->semantic_scorer_->card_embedding);
    add_mod(model_dst->semantic_scorer_->space_embedding);
    add_mod(model_dst->semantic_scorer_->desc_proj);
    add_mod(model_dst->semantic_scorer_->desc_ln);
    add_mod(model_dst->semantic_scorer_->trunk_proj);
    add_mod(model_dst->semantic_scorer_->trunk_ln);
    add_mod(model_dst->semantic_scorer_->mlp1);
    add_mod(model_dst->semantic_scorer_->mlp1_ln);
    add_mod(model_dst->semantic_scorer_->out_layer);
    add_mod(model_dst->semantic_scorer_->out_layer_ext);
  }

  std::vector<torch::Tensor> dst_nb_params;
  for (int b = nb_first_idx; b < dst_blocks; ++b) {
    for (auto& p : model_dst->res_blocks[b]->parameters()) dst_nb_params.push_back(p);
  }

  std::vector<torch::optim::OptimizerParamGroup> dst_groups;
  dst_groups.emplace_back(dst_pol_params, std::make_unique<torch::optim::AdamWOptions>(torch::optim::AdamWOptions(base_lr).eps(1e-5).weight_decay(0.0)));
  dst_groups.emplace_back(dst_tor_params, std::make_unique<torch::optim::AdamWOptions>(torch::optim::AdamWOptions(base_lr).eps(1e-5).weight_decay(0.0)));
  dst_groups.emplace_back(dst_nb_params, std::make_unique<torch::optim::AdamWOptions>(torch::optim::AdamWOptions(nb_lr).eps(1e-5).weight_decay(0.0)));
  torch::optim::AdamW opt_dst(dst_groups, torch::optim::AdamWOptions(base_lr).eps(1e-5));

  // Load candidate optimizer
  torch::load(opt_dst, check_opt_path, device);

  // 4. Verify groups configuration
  if (opt_dst.param_groups().size() != 3) {
    open_spiel::SpielFatalError("Expected 3 optimizer groups, found " + std::to_string(opt_dst.param_groups().size()));
  }
  const auto& opt0 = static_cast<const torch::optim::AdamWOptions&>(opt_dst.param_groups()[0].options());
  const auto& opt1 = static_cast<const torch::optim::AdamWOptions&>(opt_dst.param_groups()[1].options());
  const auto& opt2 = static_cast<const torch::optim::AdamWOptions&>(opt_dst.param_groups()[2].options());

  std::cout << absl::StrFormat("Group 0: lr=%.6e, weight_decay=%.6e, tensors=%zu  [policy_head.* (%zu)]\n",
      opt0.lr(), opt0.weight_decay(), opt_dst.param_groups()[0].params().size(), opt_dst.param_groups()[0].params().size());
  std::cout << absl::StrFormat("Group 1: lr=%.6e, weight_decay=%.6e, tensors=%zu [input_layer.* (2), res1-res8.* (64), value_head.* (2), semantic_scorer.* (18)]\n",
      opt1.lr(), opt1.weight_decay(), opt_dst.param_groups()[1].params().size());
  std::cout << absl::StrFormat("Group 2: lr=%.6e, weight_decay=%.6e, tensors=%zu [res9-res%d.* (%zu)]\n",
      opt2.lr(), opt2.weight_decay(), opt_dst.param_groups()[2].params().size(), dst_blocks, opt_dst.param_groups()[2].params().size());

  if (std::abs(opt0.lr() - base_lr) > 1e-12) open_spiel::SpielFatalError("Group 0 lr mismatch!");
  if (std::abs(opt1.lr() - base_lr) > 1e-12) open_spiel::SpielFatalError("Group 1 lr mismatch!");
  if (std::abs(opt2.lr() - nb_lr) > 1e-12) open_spiel::SpielFatalError("Group 2 lr mismatch!");
  if (opt_dst.param_groups()[0].params().size() != 2) open_spiel::SpielFatalError("Group 0 size != 2");
  if (opt_dst.param_groups()[1].params().size() != 86) open_spiel::SpielFatalError("Group 1 size != 86");
  size_t expected_g2_size = (dst_blocks - nb_first_idx) * 8;
  if (opt_dst.param_groups()[2].params().size() != expected_g2_size) {
    open_spiel::SpielFatalError(absl::StrFormat("Group 2 size mismatch: got %zu, expected %zu",
                                opt_dst.param_groups()[2].params().size(), expected_g2_size));
  }

  // 5. Verify Moments using VerifyB2OptimizerMoments
  VerifyB2OptimizerMoments(model_src, opt_src, model_dst, opt_dst, src_blocks, dst_blocks);

  std::cout << "======================================================================\n";
  std::cout << "[3b OPTIMIZER EQUALITY CHECK PASSED]\n";
  std::cout << "  - Every inherited parameter's exp_avg, exp_avg_sq, and step equal source\n";
  std::cout << "  - All " << ((dst_blocks - src_blocks) * 8) << " new tensors are fresh (step=0, exp_avg=0, exp_avg_sq=0)\n";
  std::cout << "  - All new tensors are in Group 2 with lr = " << nb_lr << "\n";
  std::cout << "======================================================================\n";
}

} // namespace

int main(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);

  int src_blocks = absl::GetFlag(FLAGS_src_blocks);
  int dst_blocks = absl::GetFlag(FLAGS_dst_blocks);
  int nb_first_idx = absl::GetFlag(FLAGS_new_block_first_index);
  double base_lr = absl::GetFlag(FLAGS_base_lr);
  double nb_lr = absl::GetFlag(FLAGS_new_block_lr);

  if (!absl::GetFlag(FLAGS_check_optimizer).empty()) {
    std::string check_opt = absl::GetFlag(FLAGS_check_optimizer);
    std::string src_opt = absl::GetFlag(FLAGS_src_optimizer);
    std::string src_m = absl::GetFlag(FLAGS_src_model);
    std::string eval_m = absl::GetFlag(FLAGS_eval_model);
    if (eval_m.empty()) {
      eval_m = absl::GetFlag(FLAGS_out_dir) + "/ppo_model_update_31400.pt";
    }
    RunOptimizerEqualityCheck(src_opt, check_opt, eval_m, src_m, src_blocks, dst_blocks, nb_first_idx, base_lr, nb_lr);
    return 0;
  }

  if (!absl::GetFlag(FLAGS_eval_model).empty()) {
    std::string eval_m = absl::GetFlag(FLAGS_eval_model);
    std::string obs_bin = absl::GetFlag(FLAGS_test_obs_bin);
    std::string meta_jsonl = absl::GetFlag(FLAGS_test_meta_jsonl);
    int test_count = absl::GetFlag(FLAGS_test_states_count);
    RunNativeEffectCheck(eval_m, obs_bin, meta_jsonl, test_count, dst_blocks, src_blocks);
    return 0;
  }

  std::string src_m_path = absl::GetFlag(FLAGS_src_model);
  std::string src_opt_path = absl::GetFlag(FLAGS_src_optimizer);
  std::string src_json_path = absl::GetFlag(FLAGS_src_manifest);
  std::string out_dir = absl::GetFlag(FLAGS_out_dir);

  std::cout << "======================================================================\n";
  std::cout << "DUNE DRL CAPACITY PROBE: U31400 " << src_blocks << " -> " << dst_blocks << " RESIDUAL BLOCKS SURGERY TOOL\n";
  std::cout << "======================================================================\n";
  std::cout << "Source model:     " << src_m_path << "\n";
  std::cout << "Source optimizer: " << src_opt_path << "\n";
  std::cout << "Source manifest:  " << src_json_path << "\n";
  std::cout << "Destination dir:  " << out_dir << "\n";

  std::filesystem::create_directories(out_dir);

  torch::Device device(torch::kCPU);

  // Parse source manifest to get update number
  std::ifstream src_jf(src_json_path);
  if (!src_jf.is_open()) {
    open_spiel::SpielFatalError("Cannot open src_manifest: " + src_json_path);
  }
  std::string j_content((std::istreambuf_iterator<char>(src_jf)), std::istreambuf_iterator<char>());
  src_jf.close();

  auto j_opt = open_spiel::json::FromString(j_content);
  if (!j_opt) open_spiel::SpielFatalError("Failed parsing source manifest JSON");
  auto j_obj = j_opt->GetObject();

  int64_t global_update = 31400;
  auto it_up = j_obj.find("global_update");
  if (it_up != j_obj.end() && it_up->second.IsInt()) {
    global_update = it_up->second.GetInt();
  }

  // 1. Build Source Model and Load
  std::cout << "\n[B1 SURGERY] Loading source " << src_blocks << "-block model...\n";
  auto model_src = std::make_shared<open_spiel::SharedDunePolicyValueNetImpl>(
      open_spiel::dune_imperium::kFullPublicInformationStateSize, 2048, 2391, src_blocks,
      /*nonlinear=*/false, /*aux=*/false, /*seed=*/0, /*scorer=*/true);
  open_spiel::LoadModelCheckpointRobust(model_src, src_m_path, device);
  std::cout << "Source " << src_blocks << "-block model loaded successfully.\n";

  // 2. Build Destination Model
  std::cout << "[B1 SURGERY] Constructing destination " << dst_blocks << "-block model...\n";
  auto model_dst = std::make_shared<open_spiel::SharedDunePolicyValueNetImpl>(
      open_spiel::dune_imperium::kFullPublicInformationStateSize, 2048, 2391, dst_blocks,
      /*nonlinear=*/false, /*aux=*/false, /*seed=*/0, /*scorer=*/true);

  // 3. Copy layers exactly
  {
    torch::NoGradGuard no_grad;
    // Input layer
    model_dst->input_layer->weight.copy_(model_src->input_layer->weight);
    if (model_src->input_layer->bias.defined()) {
      model_dst->input_layer->bias.copy_(model_src->input_layer->bias);
    }

    // Inherited residual blocks (0 .. src_blocks - 1)
    for (int b = 0; b < src_blocks; ++b) {
      model_dst->res_blocks[b]->fc1->weight.copy_(model_src->res_blocks[b]->fc1->weight);
      model_dst->res_blocks[b]->fc1->bias.copy_(model_src->res_blocks[b]->fc1->bias);
      model_dst->res_blocks[b]->fc2->weight.copy_(model_src->res_blocks[b]->fc2->weight);
      model_dst->res_blocks[b]->fc2->bias.copy_(model_src->res_blocks[b]->fc2->bias);
      model_dst->res_blocks[b]->ln1->weight.copy_(model_src->res_blocks[b]->ln1->weight);
      model_dst->res_blocks[b]->ln1->bias.copy_(model_src->res_blocks[b]->ln1->bias);
      model_dst->res_blocks[b]->ln2->weight.copy_(model_src->res_blocks[b]->ln2->weight);
      model_dst->res_blocks[b]->ln2->bias.copy_(model_src->res_blocks[b]->ln2->bias);
    }

    // Append new blocks (src_blocks .. dst_blocks - 1):
    // fc1, fc2, ln1 remain at trainer's normal initialization.
    // ln2 weight = 0, ln2 bias = 0.
    for (int b = src_blocks; b < dst_blocks; ++b) {
      model_dst->res_blocks[b]->ln2->weight.zero_();
      model_dst->res_blocks[b]->ln2->bias.zero_();
    }

    // Policy head
    model_dst->policy_head->weight.copy_(model_src->policy_head->weight);
    if (model_src->policy_head->bias.defined()) {
      model_dst->policy_head->bias.copy_(model_src->policy_head->bias);
    }

    // Value head
    model_dst->value_head->weight.copy_(model_src->value_head->weight);
    if (model_src->value_head->bias.defined()) {
      model_dst->value_head->bias.copy_(model_src->value_head->bias);
    }

    // Semantic scorer
    if (model_src->semantic_scorer_ && model_dst->semantic_scorer_) {
      for (const auto& pair : model_src->semantic_scorer_->named_parameters()) {
        model_dst->semantic_scorer_->named_parameters()[pair.key()].copy_(pair.value());
      }
      for (const auto& pair : model_src->semantic_scorer_->named_buffers()) {
        model_dst->semantic_scorer_->named_buffers()[pair.key()].copy_(pair.value());
      }
    }
  }
  std::cout << "[B1 SURGERY] Layers copied: input_layer, res1-" << src_blocks
            << ", policy_head, value_head, semantic_scorer.\n";
  std::cout << "[B1 SURGERY] res" << (src_blocks + 1) << "-res" << dst_blocks
            << " appended with normal init on fc1/fc2/ln1, and ln2 weight=0, bias=0.\n";

  // 4. Save destination model
  std::string dst_model_path = out_dir + "/ppo_model_update_" + std::to_string(global_update) + ".pt";
  torch::save(model_dst, dst_model_path);
  size_t dst_m_size = 0;
  std::string dst_m_sha = open_spiel::ComputeFileSHA256(dst_model_path, &dst_m_size);
  std::cout << "[B1 SURGERY] Saved " << dst_blocks << "-block model: " << dst_model_path
            << " (" << dst_m_size << " bytes, SHA=" << dst_m_sha << ")\n";

  // 5. Optimizer Migration
  std::cout << "\n[B2 OPTIMIZER] Setting up source and destination AdamW optimizers...\n";
  // Setup source optimizer
  std::vector<torch::Tensor> src_pol_params;
  src_pol_params.push_back(model_src->policy_head->weight);
  if (model_src->policy_head->bias.defined()) src_pol_params.push_back(model_src->policy_head->bias);

  std::vector<torch::Tensor> src_tor_params;
  src_tor_params.push_back(model_src->input_layer->weight);
  if (model_src->input_layer->bias.defined()) src_tor_params.push_back(model_src->input_layer->bias);
  for (int b = 0; b < nb_first_idx; ++b) {
    for (auto& p : model_src->res_blocks[b]->parameters()) src_tor_params.push_back(p);
  }
  for (auto& p : model_src->value_head->parameters()) src_tor_params.push_back(p);
  if (model_src->with_semantic_scorer_ && model_src->semantic_scorer_) {
    auto add_mod = [&](auto& mod) { for (auto& p : mod->parameters()) src_tor_params.push_back(p); };
    add_mod(model_src->semantic_scorer_->card_embedding);
    add_mod(model_src->semantic_scorer_->space_embedding);
    add_mod(model_src->semantic_scorer_->desc_proj);
    add_mod(model_src->semantic_scorer_->desc_ln);
    add_mod(model_src->semantic_scorer_->trunk_proj);
    add_mod(model_src->semantic_scorer_->trunk_ln);
    add_mod(model_src->semantic_scorer_->mlp1);
    add_mod(model_src->semantic_scorer_->mlp1_ln);
    add_mod(model_src->semantic_scorer_->out_layer);
    add_mod(model_src->semantic_scorer_->out_layer_ext);
  }

  std::vector<torch::Tensor> src_nb_params;
  for (int b = nb_first_idx; b < src_blocks; ++b) {
    for (auto& p : model_src->res_blocks[b]->parameters()) src_nb_params.push_back(p);
  }

  std::vector<torch::optim::OptimizerParamGroup> src_opt_groups;
  src_opt_groups.emplace_back(src_pol_params, std::make_unique<torch::optim::AdamWOptions>(torch::optim::AdamWOptions(base_lr).eps(1e-5).weight_decay(0.0)));
  src_opt_groups.emplace_back(src_tor_params, std::make_unique<torch::optim::AdamWOptions>(torch::optim::AdamWOptions(base_lr).eps(1e-5).weight_decay(0.0)));
  if (!src_nb_params.empty()) {
    src_opt_groups.emplace_back(src_nb_params, std::make_unique<torch::optim::AdamWOptions>(torch::optim::AdamWOptions(nb_lr).eps(1e-5).weight_decay(0.0)));
  }
  torch::optim::AdamW opt_src(src_opt_groups, torch::optim::AdamWOptions(base_lr).eps(1e-5));
  torch::load(opt_src, src_opt_path, device);
  std::cout << "[B2 OPTIMIZER] Source optimizer loaded successfully from " << src_opt_path << "\n";

  // Setup destination optimizer
  std::vector<torch::Tensor> dst_pol_params;
  dst_pol_params.push_back(model_dst->policy_head->weight);
  if (model_dst->policy_head->bias.defined()) dst_pol_params.push_back(model_dst->policy_head->bias);

  std::vector<torch::Tensor> dst_tor_params;
  dst_tor_params.push_back(model_dst->input_layer->weight);
  if (model_dst->input_layer->bias.defined()) dst_tor_params.push_back(model_dst->input_layer->bias);
  for (int b = 0; b < nb_first_idx; ++b) {
    for (auto& p : model_dst->res_blocks[b]->parameters()) dst_tor_params.push_back(p);
  }
  for (auto& p : model_dst->value_head->parameters()) dst_tor_params.push_back(p);
  if (model_dst->with_semantic_scorer_ && model_dst->semantic_scorer_) {
    auto add_mod = [&](auto& mod) { for (auto& p : mod->parameters()) dst_tor_params.push_back(p); };
    add_mod(model_dst->semantic_scorer_->card_embedding);
    add_mod(model_dst->semantic_scorer_->space_embedding);
    add_mod(model_dst->semantic_scorer_->desc_proj);
    add_mod(model_dst->semantic_scorer_->desc_ln);
    add_mod(model_dst->semantic_scorer_->trunk_proj);
    add_mod(model_dst->semantic_scorer_->trunk_ln);
    add_mod(model_dst->semantic_scorer_->mlp1);
    add_mod(model_dst->semantic_scorer_->mlp1_ln);
    add_mod(model_dst->semantic_scorer_->out_layer);
    add_mod(model_dst->semantic_scorer_->out_layer_ext);
  }

  std::vector<torch::Tensor> dst_nb_params;
  for (int b = nb_first_idx; b < dst_blocks; ++b) {
    for (auto& p : model_dst->res_blocks[b]->parameters()) dst_nb_params.push_back(p);
  }

  std::vector<torch::optim::OptimizerParamGroup> dst_opt_groups;
  dst_opt_groups.emplace_back(dst_pol_params, std::make_unique<torch::optim::AdamWOptions>(torch::optim::AdamWOptions(base_lr).eps(1e-5).weight_decay(0.0)));
  dst_opt_groups.emplace_back(dst_tor_params, std::make_unique<torch::optim::AdamWOptions>(torch::optim::AdamWOptions(base_lr).eps(1e-5).weight_decay(0.0)));
  dst_opt_groups.emplace_back(dst_nb_params, std::make_unique<torch::optim::AdamWOptions>(torch::optim::AdamWOptions(nb_lr).eps(1e-5).weight_decay(0.0)));
  torch::optim::AdamW opt_dst(dst_opt_groups, torch::optim::AdamWOptions(base_lr).eps(1e-5));

  // Migrate state from opt_src to opt_dst
  auto copy_state = [&](const torch::Tensor& t_src, const torch::Tensor& t_dst) {
    auto it = opt_src.state().find(t_src.unsafeGetTensorImpl());
    if (it == opt_src.state().end()) {
      open_spiel::SpielFatalError("Source param missing from source optimizer state!");
    }
    auto* s_src = dynamic_cast<torch::optim::AdamWParamState*>(it->second.get());
    auto s_dst = std::make_unique<torch::optim::AdamWParamState>();
    s_dst->step(s_src->step());
    s_dst->exp_avg(s_src->exp_avg().clone());
    s_dst->exp_avg_sq(s_src->exp_avg_sq().clone());
    opt_dst.state()[t_dst.unsafeGetTensorImpl()] = std::move(s_dst);
  };

  // Policy head
  copy_state(model_src->policy_head->weight, model_dst->policy_head->weight);
  if (model_src->policy_head->bias.defined()) {
    copy_state(model_src->policy_head->bias, model_dst->policy_head->bias);
  }
  // Input layer
  copy_state(model_src->input_layer->weight, model_dst->input_layer->weight);
  if (model_src->input_layer->bias.defined()) {
    copy_state(model_src->input_layer->bias, model_dst->input_layer->bias);
  }
  // Inherited res blocks
  for (int b = 0; b < src_blocks; ++b) {
    copy_state(model_src->res_blocks[b]->fc1->weight, model_dst->res_blocks[b]->fc1->weight);
    copy_state(model_src->res_blocks[b]->fc1->bias, model_dst->res_blocks[b]->fc1->bias);
    copy_state(model_src->res_blocks[b]->fc2->weight, model_dst->res_blocks[b]->fc2->weight);
    copy_state(model_src->res_blocks[b]->fc2->bias, model_dst->res_blocks[b]->fc2->bias);
    copy_state(model_src->res_blocks[b]->ln1->weight, model_dst->res_blocks[b]->ln1->weight);
    copy_state(model_src->res_blocks[b]->ln1->bias, model_dst->res_blocks[b]->ln1->bias);
    copy_state(model_src->res_blocks[b]->ln2->weight, model_dst->res_blocks[b]->ln2->weight);
    copy_state(model_src->res_blocks[b]->ln2->bias, model_dst->res_blocks[b]->ln2->bias);
  }
  // Value head
  copy_state(model_src->value_head->weight, model_dst->value_head->weight);
  if (model_src->value_head->bias.defined()) {
    copy_state(model_src->value_head->bias, model_dst->value_head->bias);
  }
  // Semantic scorer
  if (model_src->semantic_scorer_ && model_dst->semantic_scorer_) {
    const auto& p_src = model_src->semantic_scorer_->named_parameters();
    const auto& p_dst = model_dst->semantic_scorer_->named_parameters();
    for (const auto& pair : p_src) {
      copy_state(pair.value(), p_dst[pair.key()]);
    }
  }

  // Initialize fresh state for new blocks
  for (int b = src_blocks; b < dst_blocks; ++b) {
    for (const auto& p : model_dst->res_blocks[b]->parameters()) {
      auto s_new = std::make_unique<torch::optim::AdamWParamState>();
      s_new->step(0);
      s_new->exp_avg(torch::zeros_like(p));
      s_new->exp_avg_sq(torch::zeros_like(p));
      opt_dst.state()[p.unsafeGetTensorImpl()] = std::move(s_new);
    }
  }

  std::string dst_opt_path = out_dir + "/ppo_optimizer_update_" + std::to_string(global_update) + ".pt";
  torch::save(opt_dst, dst_opt_path);
  size_t dst_opt_size = 0;
  std::string dst_opt_sha = open_spiel::ComputeFileSHA256(dst_opt_path, &dst_opt_size);
  std::cout << "[B2 OPTIMIZER] Saved migrated " << dst_blocks << "-block optimizer: " << dst_opt_path
            << " (" << dst_opt_size << " bytes, SHA=" << dst_opt_sha << ")\n";

  // 6. Update JSON Manifest
  std::cout << "\n[B1 MANIFEST] Updating JSON manifest...\n";
  j_obj["num_blocks"] = open_spiel::json::Value(static_cast<int64_t>(dst_blocks));
  j_obj["model_filename"] = open_spiel::json::Value("ppo_model_update_" + std::to_string(global_update) + ".pt");
  j_obj["model_file_size"] = open_spiel::json::Value(static_cast<int64_t>(dst_m_size));
  j_obj["model_sha256"] = open_spiel::json::Value(dst_m_sha);
  j_obj["optimizer_filename"] = open_spiel::json::Value("ppo_optimizer_update_" + std::to_string(global_update) + ".pt");
  j_obj["optimizer_file_size"] = open_spiel::json::Value(static_cast<int64_t>(dst_opt_size));
  j_obj["optimizer_sha256"] = open_spiel::json::Value(dst_opt_sha);

  std::string dst_json_path = out_dir + "/ppo_model_update_" + std::to_string(global_update) + ".json";
  std::ofstream out_jf(dst_json_path);
  out_jf << open_spiel::json::ToString(open_spiel::json::Value(j_obj));
  out_jf.close();
  std::cout << "[B1 MANIFEST] Saved updated manifest with num_blocks=" << dst_blocks << ": " << dst_json_path << "\n";

  // 7. Verify In-Memory Moments
  std::cout << "\n[B2 TEST STAGE (a)] Verifying in-memory optimizer moments...\n";
  VerifyB2OptimizerMoments(model_src, opt_src, model_dst, opt_dst, src_blocks, dst_blocks);

  // 7b. Verify Post-Serialization Reload
  std::cout << "\n[B2 TEST STAGE (b)] Verifying post-serialization optimizer reload...\n";
  auto model_dst_loaded = std::make_shared<open_spiel::SharedDunePolicyValueNetImpl>(
      open_spiel::dune_imperium::kFullPublicInformationStateSize, 2048, 2391, dst_blocks,
      /*nonlinear=*/false, /*aux=*/false, /*seed=*/0, /*scorer=*/true);
  open_spiel::LoadModelCheckpointRobust(model_dst_loaded, dst_model_path, device);

  std::vector<torch::Tensor> loaded_pol_params;
  loaded_pol_params.push_back(model_dst_loaded->policy_head->weight);
  if (model_dst_loaded->policy_head->bias.defined()) loaded_pol_params.push_back(model_dst_loaded->policy_head->bias);

  std::vector<torch::Tensor> loaded_tor_params;
  loaded_tor_params.push_back(model_dst_loaded->input_layer->weight);
  if (model_dst_loaded->input_layer->bias.defined()) loaded_tor_params.push_back(model_dst_loaded->input_layer->bias);
  for (int b = 0; b < nb_first_idx; ++b) {
    for (auto& p : model_dst_loaded->res_blocks[b]->parameters()) loaded_tor_params.push_back(p);
  }
  for (auto& p : model_dst_loaded->value_head->parameters()) loaded_tor_params.push_back(p);
  if (model_dst_loaded->with_semantic_scorer_ && model_dst_loaded->semantic_scorer_) {
    auto add_mod = [&](auto& mod) { for (auto& p : mod->parameters()) loaded_tor_params.push_back(p); };
    add_mod(model_dst_loaded->semantic_scorer_->card_embedding);
    add_mod(model_dst_loaded->semantic_scorer_->space_embedding);
    add_mod(model_dst_loaded->semantic_scorer_->desc_proj);
    add_mod(model_dst_loaded->semantic_scorer_->desc_ln);
    add_mod(model_dst_loaded->semantic_scorer_->trunk_proj);
    add_mod(model_dst_loaded->semantic_scorer_->trunk_ln);
    add_mod(model_dst_loaded->semantic_scorer_->mlp1);
    add_mod(model_dst_loaded->semantic_scorer_->mlp1_ln);
    add_mod(model_dst_loaded->semantic_scorer_->out_layer);
    add_mod(model_dst_loaded->semantic_scorer_->out_layer_ext);
  }

  std::vector<torch::Tensor> loaded_nb_params;
  for (int b = nb_first_idx; b < dst_blocks; ++b) {
    for (auto& p : model_dst_loaded->res_blocks[b]->parameters()) loaded_nb_params.push_back(p);
  }

  std::vector<torch::optim::OptimizerParamGroup> loaded_groups;
  loaded_groups.emplace_back(loaded_pol_params, std::make_unique<torch::optim::AdamWOptions>(torch::optim::AdamWOptions(base_lr).eps(1e-5).weight_decay(0.0)));
  loaded_groups.emplace_back(loaded_tor_params, std::make_unique<torch::optim::AdamWOptions>(torch::optim::AdamWOptions(base_lr).eps(1e-5).weight_decay(0.0)));
  loaded_groups.emplace_back(loaded_nb_params, std::make_unique<torch::optim::AdamWOptions>(torch::optim::AdamWOptions(nb_lr).eps(1e-5).weight_decay(0.0)));
  torch::optim::AdamW opt_dst_loaded(loaded_groups, torch::optim::AdamWOptions(base_lr).eps(1e-5));

  std::string opt_to_check = dst_opt_path;
  std::string flag_check_opt = absl::GetFlag(FLAGS_check_optimizer);
  if (!flag_check_opt.empty()) {
    opt_to_check = flag_check_opt;
  }
  std::cout << "[B2 TEST STAGE (b)] Loading optimizer from: " << opt_to_check << "\n";

  torch::load(opt_dst_loaded, opt_to_check, device);

  VerifyB2OptimizerMoments(model_src, opt_src, model_dst_loaded, opt_dst_loaded, src_blocks, dst_blocks);
  std::cout << "[B2 TEST PASS] Post-serialization reload verified bit-exact on " << opt_to_check << "!\n";

  // 8. Verify B3 Identity Checks
  std::string obs_bin = absl::GetFlag(FLAGS_test_obs_bin);
  int test_count = absl::GetFlag(FLAGS_test_states_count);
  VerifyB3Identity(model_src, model_dst, obs_bin, test_count);

  // 9. Verify 3c Native Effect Check (Self-test on step-0 checkpoint: all zeros expected)
  std::string meta_jsonl = absl::GetFlag(FLAGS_test_meta_jsonl);
  RunNativeEffectCheck(dst_model_path, obs_bin, meta_jsonl, test_count, dst_blocks, src_blocks);

  std::cout << "\n======================================================================\n";
  std::cout << "ALL B1, B2, AND B3 SURGERY & IDENTITY VERIFICATIONS PASSED ("
            << src_blocks << " -> " << dst_blocks << ")!\n";
  std::cout << "======================================================================\n";
  return 0;
}
