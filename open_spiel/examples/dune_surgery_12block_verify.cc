#include "dune_surgery_12block_internal.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <iterator>
#include <memory>
#include <string>
#include <vector>

#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "open_spiel/games/dune_imperium/dune_imperium.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"

#include "dune_evaluator.h"
#include "dune_semantic_action_scorer.h"

namespace open_spiel::dune_surgery_12block {
namespace {

struct ScoredForwardOutput {
  open_spiel::SharedDunePolicyValueNetImpl::ModelOutputs raw;
  torch::Tensor scored_logits;
};

ScoredForwardOutput ForwardWithSemanticScorer(
    const std::shared_ptr<open_spiel::SharedDunePolicyValueNetImpl>& model,
    const open_spiel::dune_imperium::DuneImperiumState& dune_state,
    const std::vector<open_spiel::Action>& legal_actions,
    const torch::Tensor& x,
    const torch::Device& device) {
  auto out = model->forward(x);
  torch::Tensor logits = out.logits.clone();
  if (model->with_semantic_scorer_ && model->semantic_scorer_) {
    open_spiel::dune_semantic::CandidateActionData cand_data;
    open_spiel::dune_semantic::ExtractCandidateDescriptors(
        dune_state, legal_actions, &cand_data, model->semantic_descriptor_schema_);
    std::vector<const open_spiel::dune_semantic::CandidateActionData*> batch_cands = {&cand_data};
    open_spiel::dune_semantic::ApplySemanticScorerBatch(
        model->semantic_scorer_, out.trunk, batch_cands, logits, device);
  }
  return {std::move(out), std::move(logits)};
}

struct NativeEffectMetrics {
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
};

void EvaluateNativeEffectSample(
    const std::shared_ptr<open_spiel::SharedDunePolicyValueNetImpl>& model_active,
    const std::shared_ptr<open_spiel::SharedDunePolicyValueNetImpl>& model_zeroed,
    const open_spiel::dune_imperium::DuneImperiumState& dune_state,
    const std::vector<open_spiel::Action>& legal_actions,
    const torch::Tensor& x,
    const torch::Device& device,
    NativeEffectMetrics* metrics) {
  // Pass 1: active model
  auto act = ForwardWithSemanticScorer(model_active, dune_state, legal_actions, x, device);

  // Pass 2: repeat active model (noise floor)
  auto rep = ForwardWithSemanticScorer(model_active, dune_state, legal_actions, x, device);

  // Pass 3: zeroed model (ln2 reset to 0)
  auto zero = ForwardWithSemanticScorer(model_zeroed, dune_state, legal_actions, x, device);

  // Noise floor
  double val_noise = (act.raw.values - rep.raw.values).abs().item<double>();
  if (val_noise > metrics->max_noise_val) metrics->max_noise_val = val_noise;
  metrics->sum_noise_val += val_noise;

  for (open_spiel::Action a : legal_actions) {
    double l_noise = std::abs(act.scored_logits[0][a].item<double>() -
                              rep.scored_logits[0][a].item<double>());
    if (l_noise > metrics->max_noise_legal_logits) metrics->max_noise_legal_logits = l_noise;
    metrics->sum_noise_legal_logits += l_noise;
  }

  // Native effect
  double val_diff = (act.raw.values - zero.raw.values).abs().item<double>();
  if (val_diff > metrics->max_diff_val) metrics->max_diff_val = val_diff;
  metrics->sum_diff_val += val_diff;

  for (open_spiel::Action a : legal_actions) {
    double l_diff = std::abs(act.scored_logits[0][a].item<double>() -
                             zero.scored_logits[0][a].item<double>());
    if (l_diff > metrics->max_diff_legal_logits) metrics->max_diff_legal_logits = l_diff;
    metrics->sum_diff_legal_logits += l_diff;
  }

  metrics->total_legal_decisions += legal_actions.size();
  ++metrics->states_evaluated;
}

}  // namespace

void VerifyB2OptimizerMoments(
    const std::shared_ptr<open_spiel::SharedDunePolicyValueNetImpl>& model_src,
    torch::optim::AdamW& opt_src,
    const std::shared_ptr<open_spiel::SharedDunePolicyValueNetImpl>& model_dst,
    torch::optim::AdamW& opt_dst,
    int src_blocks,
    int dst_blocks) {
  std::cout << "\n======================================================================\n";
  std::cout << "[B2 TEST] Verifying Optimizer Moments & Step Counters ("
            << src_blocks << " -> " << dst_blocks << ")\n";
  std::cout << "======================================================================\n";

  auto check_pair = [&](const std::string& name, const torch::Tensor& t_src,
                        const torch::Tensor& t_dst) {
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
                                  std::to_string(s_src->step()) +
                                  ", dst=" + std::to_string(s_dst->step()));
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
  check_pair("policy_head.weight", model_src->policy_head->weight,
             model_dst->policy_head->weight);
  ++inherited_params_checked;
  if (model_src->policy_head->bias.defined()) {
    check_pair("policy_head.bias", model_src->policy_head->bias,
               model_dst->policy_head->bias);
    ++inherited_params_checked;
  }

  // 2. Check Input Layer
  check_pair("input_layer.weight", model_src->input_layer->weight,
             model_dst->input_layer->weight);
  ++inherited_params_checked;
  if (model_src->input_layer->bias.defined()) {
    check_pair("input_layer.bias", model_src->input_layer->bias,
               model_dst->input_layer->bias);
    ++inherited_params_checked;
  }

  // 3. Check res1 .. res[src_blocks]
  for (int b = 0; b < src_blocks; ++b) {
    std::string prefix = "res" + std::to_string(b + 1) + ".";
    check_pair(prefix + "fc1.weight", model_src->res_blocks[b]->fc1->weight,
               model_dst->res_blocks[b]->fc1->weight);
    check_pair(prefix + "fc1.bias", model_src->res_blocks[b]->fc1->bias,
               model_dst->res_blocks[b]->fc1->bias);
    check_pair(prefix + "fc2.weight", model_src->res_blocks[b]->fc2->weight,
               model_dst->res_blocks[b]->fc2->weight);
    check_pair(prefix + "fc2.bias", model_src->res_blocks[b]->fc2->bias,
               model_dst->res_blocks[b]->fc2->bias);
    check_pair(prefix + "ln1.weight", model_src->res_blocks[b]->ln1->weight,
               model_dst->res_blocks[b]->ln1->weight);
    check_pair(prefix + "ln1.bias", model_src->res_blocks[b]->ln1->bias,
               model_dst->res_blocks[b]->ln1->bias);
    check_pair(prefix + "ln2.weight", model_src->res_blocks[b]->ln2->weight,
               model_dst->res_blocks[b]->ln2->weight);
    check_pair(prefix + "ln2.bias", model_src->res_blocks[b]->ln2->bias,
               model_dst->res_blocks[b]->ln2->bias);
    inherited_params_checked += 8;
  }

  // 4. Check Value Head
  check_pair("value_head.weight", model_src->value_head->weight,
             model_dst->value_head->weight);
  ++inherited_params_checked;
  if (model_src->value_head->bias.defined()) {
    check_pair("value_head.bias", model_src->value_head->bias,
               model_dst->value_head->bias);
    ++inherited_params_checked;
  }

  // 5. Check Semantic Scorer (covers all named parameters including 5 v5_* params)
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
        open_spiel::SpielFatalError("New param in res block " + std::to_string(b + 1) +
                                    " missing optimizer state!");
      }
      auto* s = dynamic_cast<torch::optim::AdamWParamState*>(it->second.get());
      if (s->step() != 0) {
        open_spiel::SpielFatalError("New param step != 0 (got " +
                                    std::to_string(s->step()) + ")");
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
    const torch::Device& device) {
  std::cout << "\n[B2 TEST STAGE (b)] Verifying post-serialization optimizer reload...\n";
  auto model_dst_loaded = MakeModel(manifest_meta.observation_dim, dst_blocks,
                                    manifest_meta.semantic_descriptor_schema,
                                    manifest_meta.market_appendix_mode,
                                    manifest_meta.deck_pool_semantics);
  open_spiel::LoadModelCheckpointRobust(model_dst_loaded, dst_model_path, device);

  auto opt_dst_loaded_ptr = open_spiel::MakeDuneOptimizer(
      model_dst_loaded, base_lr, /*weight_decay=*/0.0, /*policy_weight_decay=*/0.0,
      nb_lr, dst_nb_first_idx);
  torch::optim::AdamW& opt_dst_loaded = *opt_dst_loaded_ptr;

  std::cout << "[B2 TEST STAGE (b)] Loading optimizer from: " << opt_to_check << "\n";
  torch::load(opt_dst_loaded, opt_to_check, device);

  VerifyB2OptimizerMoments(model_src, opt_src, model_dst_loaded, opt_dst_loaded,
                           src_blocks, dst_blocks);
  std::cout << "[B2 TEST PASS] Post-serialization reload verified bit-exact on "
            << opt_to_check << "!\n";
}

void VerifyB3IdentityBin(
    const std::shared_ptr<open_spiel::SharedDunePolicyValueNetImpl>& model_src,
    const std::shared_ptr<open_spiel::SharedDunePolicyValueNetImpl>& model_dst,
    const std::string& obs_bin_path,
    int test_states_count,
    int64_t obs_size) {
  std::cout << "\n======================================================================\n";
  std::cout << "[B3 TEST BIN] Verifying Forward Identity on " << test_states_count
            << " Stored States\n";
  std::cout << "======================================================================\n";

  std::ifstream obs_file(obs_bin_path, std::ios::binary);
  if (!obs_file.is_open()) {
    open_spiel::SpielFatalError("Cannot open test_obs_bin: " + obs_bin_path);
  }

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

    torch::Tensor x = torch::from_blob(obs_buffer.data(), {1, obs_size}, torch::kFloat32);

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
  std::cout << "Max absolute diff logits: " << max_abs_diff_logits
            << " (exact 0 in fp32 expected)\n";
  std::cout << "Max absolute diff values: " << max_abs_diff_value
            << " (exact 0 in fp32 expected)\n";
  std::cout << "Mean absolute diff logits: "
            << (sum_abs_diff_logits / total_logit_elements) << "\n";

  if (max_abs_diff_logits > 0.0 || max_abs_diff_value > 0.0) {
    open_spiel::SpielFatalError(absl::StrFormat(
        "B3 Identity check FAILED! Expected bit-exact 0 in fp32, got max_l=%g, max_v=%g",
        max_abs_diff_logits, max_abs_diff_value));
  }
  std::cout << "[B3 TEST PASS] Exact numerical identity proven! max_abs_diff = 0.00000000 in fp32.\n";
}

void VerifyB3IdentityJson(
    const std::shared_ptr<open_spiel::SharedDunePolicyValueNetImpl>& model_src,
    const std::shared_ptr<open_spiel::SharedDunePolicyValueNetImpl>& model_dst,
    const std::string& states_json_path,
    int test_states_count,
    int64_t obs_size,
    open_spiel::dune_imperium::MarketAppendixMode mkt_mode,
    open_spiel::dune_imperium::DeckPoolSemantics semantics) {
  std::cout << "\n======================================================================\n";
  std::cout << "[B3 TEST JSON] Verifying Forward Identity on " << test_states_count
            << " Stored States (" << states_json_path << ")\n";
  std::cout << "======================================================================\n";

  std::ifstream json_f(states_json_path);
  if (!json_f.is_open()) {
    open_spiel::SpielFatalError("Cannot open test_states_json: " + states_json_path);
  }
  std::string json_content((std::istreambuf_iterator<char>(json_f)),
                           std::istreambuf_iterator<char>());
  json_f.close();

  auto parsed = open_spiel::json::FromString(json_content);
  if (!parsed || !parsed->IsArray()) {
    open_spiel::SpielFatalError("test_states_json must be a JSON array");
  }
  const auto& arr = parsed->GetArray();
  int total_to_test = std::min(test_states_count, static_cast<int>(arr.size()));

  auto game = open_spiel::LoadGame("dune_imperium");
  torch::Device device(torch::kCPU);

  torch::NoGradGuard no_grad;
  model_src->eval();
  model_dst->eval();

  std::vector<float> obs_buffer(obs_size);

  double max_diff_raw_logits = 0.0;
  double max_diff_legal_logits = 0.0;
  double max_diff_values = 0.0;
  int64_t total_legal_checked = 0;
  int v5_states_count = 0;

  for (int i = 0; i < total_to_test; ++i) {
    const auto& entry = arr[i].GetObject();
    auto it_state = entry.find("serialized_state");
    if (it_state == entry.end() || !it_state->second.IsString()) {
      open_spiel::SpielFatalError("Missing serialized_state in entry " + std::to_string(i));
    }
    std::string ser_state = it_state->second.GetString();
    auto state = game->DeserializeState(ser_state);
    const auto* dune_state =
        dynamic_cast<const open_spiel::dune_imperium::DuneImperiumState*>(state.get());
    if (!dune_state) {
      open_spiel::SpielFatalError("Failed casting deserialized state in entry " +
                                  std::to_string(i));
    }

    auto legal_actions = state->LegalActions();
    auto it_v5 = entry.find("has_v5_legal");
    if (it_v5 != entry.end() && it_v5->second.IsBool() && it_v5->second.GetBool()) {
      ++v5_states_count;
    }

    dune_state->InformationStateTensorWithAppendix(
        state->CurrentPlayer(), mkt_mode, absl::MakeSpan(obs_buffer), semantics);
    torch::Tensor x = torch::from_blob(obs_buffer.data(), {1, obs_size}, torch::kFloat32);

    // Pass src
    auto src_out = ForwardWithSemanticScorer(model_src, *dune_state, legal_actions, x, device);

    // Pass dst
    auto dst_out = ForwardWithSemanticScorer(model_dst, *dune_state, legal_actions, x, device);

    // Raw logits diff
    double raw_l_diff = (dst_out.raw.logits - src_out.raw.logits).abs().max().item<double>();
    if (raw_l_diff > max_diff_raw_logits) max_diff_raw_logits = raw_l_diff;

    // Value diff
    double v_diff = (dst_out.raw.values - src_out.raw.values).abs().max().item<double>();
    if (v_diff > max_diff_values) max_diff_values = v_diff;

    // Legal scored logits diff
    for (open_spiel::Action a : legal_actions) {
      double l_diff = std::abs(dst_out.scored_logits[0][a].item<double>() -
                               src_out.scored_logits[0][a].item<double>());
      if (l_diff > max_diff_legal_logits) max_diff_legal_logits = l_diff;
      ++total_legal_checked;
    }
  }

  std::cout << absl::StrFormat(
      "Tested stored JSON states:       %d (with %d has_v5_legal=true)\n",
      total_to_test, v5_states_count);
  std::cout << absl::StrFormat("Total legal decisions:           %lld\n",
                               total_legal_checked);
  std::cout << absl::StrFormat("Max absolute diff raw logits:    %.8f (exact 0 expected)\n",
                               max_diff_raw_logits);
  std::cout << absl::StrFormat("Max absolute diff legal logits:  %.8f (exact 0 expected)\n",
                               max_diff_legal_logits);
  std::cout << absl::StrFormat("Max absolute diff values:        %.8f (exact 0 expected)\n",
                               max_diff_values);

  if (max_diff_raw_logits > 0.0 || max_diff_legal_logits > 0.0 || max_diff_values > 0.0) {
    open_spiel::SpielFatalError(absl::StrFormat(
        "B3 JSON Identity check FAILED! Expected bit-exact 0 in fp32, got raw_l=%g, legal_l=%g, v=%g",
        max_diff_raw_logits, max_diff_legal_logits, max_diff_values));
  }
  std::cout << "[B3 TEST PASS] Exact numerical identity proven on JSON corpus! max_abs_diff = 0.00000000 in fp32.\n";
}

void RunNativeEffectCheck(
    const std::string& model_path,
    const std::string& obs_bin_path,
    const std::string& meta_jsonl_path,
    const std::string& states_json_path,
    int test_states_count,
    int num_blocks,
    int bypass_first_index,
    const ManifestMetadata& meta) {
  std::cout << "\n======================================================================\n";
  std::cout << "[3c NATIVE EFFECT CHECK] Auditing " << num_blocks << "-block model on "
            << test_states_count << " Stored States\n";
  std::cout << "Bypassing new residual blocks indices [" << bypass_first_index << ".."
            << (num_blocks - 1) << "] (res" << (bypass_first_index + 1) << "-res"
            << num_blocks << ")\n";
  std::cout << "Model: " << model_path << "\n";
  if (!states_json_path.empty()) {
    std::cout << "States JSON: " << states_json_path << "\n";
  } else {
    std::cout << "Obs:   " << obs_bin_path << "\n";
    std::cout << "Meta:  " << meta_jsonl_path << "\n";
  }
  std::cout << "======================================================================\n";

  torch::Device device(torch::kCPU);

  // 1. Load active model
  auto model_active = MakeModel(meta.observation_dim, num_blocks,
                                meta.semantic_descriptor_schema,
                                meta.market_appendix_mode,
                                meta.deck_pool_semantics);
  open_spiel::LoadModelCheckpointRobust(model_active, model_path, device);
  model_active->eval();

  // 2. Clone active model and zero ln2 weight/bias ONLY on bypassed new blocks [bypass_first_index..num_blocks-1]
  auto model_zeroed = MakeModel(meta.observation_dim, num_blocks,
                                meta.semantic_descriptor_schema,
                                meta.market_appendix_mode,
                                meta.deck_pool_semantics);
  {
    torch::NoGradGuard no_grad;
    for (const auto& pair : model_active->named_parameters()) {
      model_zeroed->named_parameters()[pair.key()].copy_(pair.value());
    }
    for (const auto& pair : model_active->named_buffers()) {
      model_zeroed->named_buffers()[pair.key()].copy_(pair.value());
    }
    if (model_active->v5_dynamic_slot_mask_.defined()) {
      model_zeroed->v5_dynamic_slot_mask_ = model_active->v5_dynamic_slot_mask_.clone();
    }
    for (int b = bypass_first_index; b < num_blocks; ++b) {
      model_zeroed->res_blocks[b]->ln2->weight.zero_();
      model_zeroed->res_blocks[b]->ln2->bias.zero_();
    }
  }
  model_zeroed->eval();

  auto game = open_spiel::LoadGame("dune_imperium");
  torch::NoGradGuard no_grad;

  NativeEffectMetrics metrics;

  if (!states_json_path.empty()) {
    std::ifstream json_f(states_json_path);
    if (!json_f.is_open()) {
      open_spiel::SpielFatalError("Cannot open test_states_json: " + states_json_path);
    }
    std::string json_content((std::istreambuf_iterator<char>(json_f)),
                             std::istreambuf_iterator<char>());
    json_f.close();

    auto parsed = open_spiel::json::FromString(json_content);
    if (!parsed || !parsed->IsArray()) {
      open_spiel::SpielFatalError("test_states_json must be a JSON array");
    }
    const auto& arr = parsed->GetArray();
    int total_to_test = std::min(test_states_count, static_cast<int>(arr.size()));
    std::vector<float> obs_buffer(meta.observation_dim);

    for (int i = 0; i < total_to_test; ++i) {
      const auto& entry = arr[i].GetObject();
      auto it_state = entry.find("serialized_state");
      if (it_state == entry.end() || !it_state->second.IsString()) {
        continue;
      }
      auto state = game->DeserializeState(it_state->second.GetString());
      const auto* dune_state =
          dynamic_cast<const open_spiel::dune_imperium::DuneImperiumState*>(state.get());
      if (!dune_state) {
        continue;
      }

      auto legal_actions = state->LegalActions();
      dune_state->InformationStateTensorWithAppendix(
          state->CurrentPlayer(), meta.market_appendix_mode,
          absl::MakeSpan(obs_buffer), meta.deck_pool_semantics);
      torch::Tensor x =
          torch::from_blob(obs_buffer.data(), {1, meta.observation_dim}, torch::kFloat32);

      EvaluateNativeEffectSample(model_active, model_zeroed, *dune_state,
                                 legal_actions, x, device, &metrics);
    }
  } else {
    // Binary obs and meta jsonl fallback
    std::ifstream obs_file(obs_bin_path, std::ios::binary);
    if (!obs_file.is_open()) {
      open_spiel::SpielFatalError("Cannot open test_obs_bin: " + obs_bin_path);
    }
    std::ifstream meta_file(meta_jsonl_path);
    if (!meta_file.is_open()) {
      open_spiel::SpielFatalError("Cannot open test_meta_jsonl: " + meta_jsonl_path);
    }

    const size_t obs_size = meta.observation_dim;
    const size_t byte_stride = obs_size * sizeof(float);
    std::vector<float> obs_buffer(obs_size);

    for (int i = 0; i < 20000 && metrics.states_evaluated < test_states_count; ++i) {
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
      const auto* dune_state =
          dynamic_cast<const open_spiel::dune_imperium::DuneImperiumState*>(state.get());
      if (!dune_state) {
        continue;
      }

      torch::Tensor x = torch::from_blob(
          obs_buffer.data(), {1, static_cast<int64_t>(obs_size)}, torch::kFloat32);

      EvaluateNativeEffectSample(model_active, model_zeroed, *dune_state,
                                 legal_actions, x, device, &metrics);
    }
  }

  std::cout << absl::StrFormat("Tested stored states:           %d\n",
                               metrics.states_evaluated);
  std::cout << absl::StrFormat("Total legal decisions:          %lld\n",
                               metrics.total_legal_decisions);
  std::cout << absl::StrFormat("Noise floor (repeat pass):\n");
  std::cout << absl::StrFormat("  Max |Δ legal logits|:         %.6e\n",
                               metrics.max_noise_legal_logits);
  std::cout << absl::StrFormat(
      "  Mean |Δ legal logits|:        %.6e\n",
      metrics.total_legal_decisions > 0
          ? (metrics.sum_noise_legal_logits / metrics.total_legal_decisions)
          : 0.0);
  std::cout << absl::StrFormat("  Max |Δ value|:                %.6e\n",
                               metrics.max_noise_val);
  std::cout << absl::StrFormat(
      "  Mean |Δ value|:               %.6e\n",
      metrics.states_evaluated > 0
          ? (metrics.sum_noise_val / metrics.states_evaluated)
          : 0.0);
  std::cout << absl::StrFormat(
      "Native effect (checkpoint vs res%d-res%d ln2 zeroed):\n",
      bypass_first_index + 1, num_blocks);
  std::cout << absl::StrFormat("  Max |Δ legal logits|:         %.6e\n",
                               metrics.max_diff_legal_logits);
  std::cout << absl::StrFormat(
      "  Mean |Δ legal logits|:        %.6e\n",
      metrics.total_legal_decisions > 0
          ? (metrics.sum_diff_legal_logits / metrics.total_legal_decisions)
          : 0.0);
  std::cout << absl::StrFormat("  Max |Δ value|:                %.6e\n",
                               metrics.max_diff_val);
  std::cout << absl::StrFormat(
      "  Mean |Δ value|:               %.6e\n",
      metrics.states_evaluated > 0
          ? (metrics.sum_diff_val / metrics.states_evaluated)
          : 0.0);
  std::cout << "======================================================================\n";
}

void RunOptimizerEqualityCheck(
    const std::string& src_opt_path,
    const std::string& check_opt_path,
    const std::string& model_dst_path,
    const std::string& model_src_path,
    int src_blocks,
    int dst_blocks,
    int src_nb_first_idx,
    int dst_nb_first_idx,
    double base_lr,
    double nb_lr,
    const ManifestMetadata& meta) {
  std::cout << "======================================================================\n";
  std::cout << "[3b OPTIMIZER EQUALITY CHECK] Optimizer State Verification ("
            << src_blocks << " -> " << dst_blocks << ")\n";
  std::cout << "Source optimizer:    " << src_opt_path << "\n";
  std::cout << "Candidate optimizer: " << check_opt_path << "\n";
  std::cout << "Destination model:   " << model_dst_path << "\n";
  std::cout << "Source model:        " << model_src_path << "\n";
  std::cout << "Source new_block_first_index:      " << src_nb_first_idx << "\n";
  std::cout << "Destination new_block_first_index: " << dst_nb_first_idx << "\n";
  std::cout << "======================================================================\n";

  torch::Device device(torch::kCPU);

  // 1. Load source model and optimizer
  auto model_src = MakeModel(meta.observation_dim, src_blocks,
                             meta.semantic_descriptor_schema,
                             meta.market_appendix_mode,
                             meta.deck_pool_semantics);
  open_spiel::LoadModelCheckpointRobust(model_src, model_src_path, device);

  auto opt_src_ptr = open_spiel::MakeDuneOptimizer(
      model_src, base_lr, /*weight_decay=*/0.0, /*policy_weight_decay=*/0.0,
      nb_lr, src_nb_first_idx);
  torch::optim::AdamW& opt_src = *opt_src_ptr;
  torch::load(opt_src, src_opt_path, device);

  // 2. Load destination model
  auto model_dst = MakeModel(meta.observation_dim, dst_blocks,
                             meta.semantic_descriptor_schema,
                             meta.market_appendix_mode,
                             meta.deck_pool_semantics);
  open_spiel::LoadModelCheckpointRobust(model_dst, model_dst_path, device);

  // 3. Construct destination optimizer
  auto opt_dst_ptr = open_spiel::MakeDuneOptimizer(
      model_dst, base_lr, /*weight_decay=*/0.0, /*policy_weight_decay=*/0.0,
      nb_lr, dst_nb_first_idx);
  torch::optim::AdamW& opt_dst = *opt_dst_ptr;

  // Load candidate optimizer
  torch::load(opt_dst, check_opt_path, device);

  // 4. Verify groups configuration
  VerifyOptimizerGroups(opt_dst, dst_blocks, dst_nb_first_idx, base_lr, nb_lr,
                        meta.semantic_descriptor_schema);

  // 5. Verify Moments using VerifyB2OptimizerMoments
  VerifyB2OptimizerMoments(model_src, opt_src, model_dst, opt_dst, src_blocks,
                           dst_blocks);

  std::cout << "======================================================================\n";
  std::cout << "[3b OPTIMIZER EQUALITY CHECK PASSED]\n";
  std::cout << "  - Every inherited parameter's exp_avg, exp_avg_sq, and step equal source\n";
  std::cout << "  - All " << ((dst_blocks - src_blocks) * 8)
            << " new tensors are fresh (step=0, exp_avg=0, exp_avg_sq=0)\n";
  std::cout << "  - All new tensors are in Group 2 with lr = " << nb_lr << "\n";
  std::cout << "======================================================================\n";
}

}  // namespace open_spiel::dune_surgery_12block
