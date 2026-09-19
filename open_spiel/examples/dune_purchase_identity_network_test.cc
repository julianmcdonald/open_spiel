#include <algorithm>
#include <filesystem>
#include <fstream>
#include <future>
#include <iostream>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <vector>

#include "dune_network.h"
#include "dune_evaluator.h"
#include "dune_batched_evaluator.h"
#include "dune_sha256.h"
#include "open_spiel/games/dune_imperium/dune_imperium.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace {
using namespace dune_imperium;

void TestInputMigration() {
  for (auto mode : {MarketAppendixMode::kNone, MarketAppendixMode::kZeros,
                    MarketAppendixMode::kOrderedMarket,
                    MarketAppendixMode::kOrderedCardSlotsV2}) {
    const auto schema = GetInformationStateSchema(mode);
    auto source = torch::randn({8, schema.size});
    auto target = torch::empty({8, kOrderedCardSlotsInformationStateSize});
    CopyInputToOrderedCardSlots(target, source, mode);
    const int prefix = mode == MarketAppendixMode::kZeros
                           ? kLegacyInformationStateSize : schema.size;
    SPIEL_CHECK_TRUE(torch::equal(target.slice(1, 0, prefix), source.slice(1, 0, prefix)));
    if (prefix < kOrderedCardSlotsInformationStateSize) {
      SPIEL_CHECK_EQ(target.slice(1, prefix).count_nonzero().item<int64_t>(), 0);
    }

    auto target_v3 = torch::empty({8, kFullPublicInformationStateSize});
    CopyInputToFullPublicInformation(target_v3, source, mode);
    SPIEL_CHECK_TRUE(torch::equal(target_v3.slice(1, 0, prefix), source.slice(1, 0, prefix)));
    if (prefix < kFullPublicInformationStateSize) {
      SPIEL_CHECK_EQ(target_v3.slice(1, prefix).count_nonzero().item<int64_t>(), 0);
    }
  }
}

void TestIncumbentMigration(const std::filesystem::path& output_dir) {
  const std::string incumbent =
      "/home/warcr/projects/dune_drl/calibration_results_v2/pf_c_run2/s1_ctl_b/ppo_model_update_15828.pt";
  auto source = std::make_shared<SharedDunePolicyValueNetImpl>(5580, 2048, 2391, 8);
  LoadModelCheckpointRobust(source, incumbent, torch::kCPU);
  auto target = std::make_shared<SharedDunePolicyValueNetImpl>(
      kOrderedCardSlotsInformationStateSize, 2048, 2391, 8);
  {
    torch::NoGradGuard guard;
    auto source_params = source->named_parameters();
    for (auto& item : target->named_parameters()) {
      if (item.key() != "input_layer.weight") {
        item.value().copy_(source_params[item.key()]);
      }
    }
    CopyInputToOrderedCardSlots(target->input_layer->weight,
                               source->input_layer->weight, MarketAppendixMode::kNone);
  }
  source->eval();
  target->eval();
  auto game = LoadGame("dune_imperium");
  auto state = game->NewInitialState();
  std::vector<std::vector<float>> old_inputs, new_inputs;
  std::vector<std::vector<Action>> legal_sets;
  std::vector<std::unique_ptr<State>> test_states;
  for (int step = 0; step < 2000 && old_inputs.size() < 12 && !state->IsTerminal(); ++step) {
    if (state->IsChanceNode()) {
      state->ApplyAction(state->ChanceOutcomes().front().first);
      continue;
    }
    auto* dune = dynamic_cast<DuneImperiumState*>(state.get());
    SPIEL_CHECK_TRUE(dune != nullptr);
    const auto legal = state->LegalActions();
    old_inputs.push_back(state->InformationStateTensor(state->CurrentPlayer()));
    new_inputs.push_back(dune->InformationStateTensorWithAppendix(
        state->CurrentPlayer(), MarketAppendixMode::kOrderedCardSlotsV2));
    legal_sets.push_back(legal);
    test_states.push_back(state->Clone());
    state->ApplyAction(legal.front());
  }
  SPIEL_CHECK_EQ(old_inputs.size(), 12);
  std::mutex source_mutex, target_mutex;
  std::shared_mutex sync_mutex;
  DeterministicEvaluator old_eval(source, torch::kCPU, &source_mutex, nullptr, nullptr, false);
  DeterministicEvaluator new_eval(target, torch::kCPU, &target_mutex, nullptr, nullptr, false);
  BatchedEvaluator batch_eval(target, 4, 5, torch::kCPU, &sync_mutex,
                              0.0f, true, false, false, false, false);
  DuneNNEvaluator old_adapter(source, torch::kCPU);
  DuneNNEvaluator new_adapter(target, torch::kCPU);
  double max_raw_delta = 0.0;
  for (size_t i = 0; i < old_inputs.size(); ++i) {
    auto old_result = old_eval.Evaluate(old_inputs[i]);
    auto new_result = new_eval.Evaluate(new_inputs[i]);
    for (Action a : legal_sets[i]) {
      max_raw_delta = std::max(max_raw_delta,
          std::abs(static_cast<double>(old_result.logits[a] - new_result.logits[a])));
    }
    torch::NoGradGuard guard;
    auto old_tensor = torch::from_blob(old_inputs[i].data(), {1, kLegacyInformationStateSize}, torch::kFloat32);
    auto new_tensor = torch::from_blob(new_inputs[i].data(), {1, kOrderedCardSlotsInformationStateSize}, torch::kFloat32);
    SPIEL_CHECK_TRUE(torch::allclose(source->forward(old_tensor).values,
                                    target->forward(new_tensor).values, 1e-5, 1e-5));
    const auto old_prior = old_adapter.Prior(*test_states[i]);
    const auto new_prior = new_adapter.Prior(*test_states[i]);
    SPIEL_CHECK_EQ(old_prior.size(), new_prior.size());
    for (size_t a = 0; a < old_prior.size(); ++a) {
      SPIEL_CHECK_EQ(old_prior[a].first, new_prior[a].first);
      SPIEL_CHECK_LT(std::abs(old_prior[a].second - new_prior[a].second), 1e-5);
    }
  }
  SPIEL_CHECK_LT(max_raw_delta, 1e-4);
  std::vector<std::future<EvalResult>> futures;
  for (size_t i = 0; i < new_inputs.size(); ++i) {
    futures.push_back(std::async(std::launch::async, [&, i] {
      return batch_eval.Evaluate(new_inputs[i]);
    }));
  }
  for (size_t i = 0; i < futures.size(); ++i) {
    auto batched = futures[i].get();
    auto direct = new_eval.Evaluate(new_inputs[i]);
    for (Action a : legal_sets[i]) {
      SPIEL_CHECK_LT(std::abs(batched.logits[a] - direct.logits[a]), 1e-4);
    }
  }
  target->zero_grad();
  auto learning_input = torch::from_blob(new_inputs.front().data(), {1, kOrderedCardSlotsInformationStateSize}, torch::kFloat32);
  auto legal_indices = torch::tensor(legal_sets.front(), torch::kInt64);
  SPIEL_CHECK_GT(legal_sets.front().size(), 1);
  auto legal_logits = target->forward(learning_input).logits.index_select(1, legal_indices);
  (-torch::log_softmax(legal_logits, 1)[0][0]).backward();
  SPIEL_CHECK_GT(target->input_layer->weight.grad()
                     .slice(1, kExpandedInformationStateSize).abs().sum().item<double>(), 0.0);
  target->zero_grad();
  std::filesystem::create_directories(output_dir);
  const auto model_path = output_dir / "identity_initial.pt";
  torch::save(target, model_path.string());
  std::ofstream meta(output_dir / "identity_initial.json");
  const auto schema = GetInformationStateSchema(MarketAppendixMode::kOrderedCardSlotsV2);
  meta << "{\"observation_dim\":" << kOrderedCardSlotsInformationStateSize
       << ",\"hidden_dim\":2048,\"num_blocks\":8,"
       << "\"market_appendix_mode\":\"ordered_card_slots_v2\","
       << "\"feature_schema_sha256\":\"" << schema.sha256 << "\","
       << "\"feature_schema_label\":\"" << schema.label << "\"}\n";
  meta.close();
  auto restored = std::make_shared<SharedDunePolicyValueNetImpl>(
      kOrderedCardSlotsInformationStateSize, 2048, 2391, 8);
  LoadModelCheckpointRobust(restored, model_path.string(), torch::kCPU);
  auto saved = target->named_parameters();
  for (const auto& item : restored->named_parameters()) {
    SPIEL_CHECK_TRUE(torch::equal(item.value(), saved[item.key()]));
  }
  std::error_code ec;
  std::filesystem::remove(model_path, ec);
  std::filesystem::remove(output_dir / "identity_initial.json", ec);
  std::cout << "Incumbent migration, concurrent inference, and checkpoint reload passed; max raw-logit delta="
            << max_raw_delta << "\n";
}

void TestEvaluatorSemanticScoring() {
  auto game = LoadGame("dune_imperium");
  auto state = game->NewInitialState();
  while (!state->IsTerminal()) {
    if (state->IsChanceNode()) {
      state->ApplyAction(state->ChanceOutcomes().front().first);
      continue;
    }
    const auto legal = state->LegalActions();
    const auto* dune = dynamic_cast<const DuneImperiumState*>(state.get());
    dune_semantic::CandidateActionData cand_data;
    dune_semantic::ExtractCandidateDescriptors(*dune, legal, &cand_data);
    bool has_supported = false;
    for (uint8_t supp : cand_data.supported) {
      if (supp != 0) {
        has_supported = true;
        break;
      }
    }
    if (has_supported) {
      break;
    }
    state->ApplyAction(legal.front());
  }

  auto base_model = std::make_shared<SharedDunePolicyValueNetImpl>(
      5580, 2048, 2391, 8, /*use_nonlinear=*/false, /*with_aux_heads=*/false,
      /*head_init_seed=*/0, /*with_semantic_scorer=*/false);
  auto semantic_model = std::make_shared<SharedDunePolicyValueNetImpl>(
      5580, 2048, 2391, 8, /*use_nonlinear=*/false, /*with_aux_heads=*/false,
      /*head_init_seed=*/0, /*with_semantic_scorer=*/true);

  {
    torch::NoGradGuard guard;
    auto base_params = base_model->named_parameters();
    for (auto& item : semantic_model->named_parameters()) {
      if (base_params.contains(item.key())) {
        item.value().copy_(base_params[item.key()]);
      }
    }
    semantic_model->semantic_scorer_->out_layer->weight.fill_(0.5f);
  }

  base_model->eval();
  semantic_model->eval();

  DuneNNEvaluator base_eval(base_model, torch::kCPU);
  DuneNNEvaluator semantic_eval(semantic_model, torch::kCPU);

  const auto base_prior = base_eval.Prior(*state);
  const auto semantic_prior = semantic_eval.Prior(*state);

  SPIEL_CHECK_EQ(base_prior.size(), semantic_prior.size());
  bool any_diff = false;
  for (size_t i = 0; i < base_prior.size(); ++i) {
    if (std::abs(base_prior[i].second - semantic_prior[i].second) > 1e-4) {
      any_diff = true;
      break;
    }
  }
  SPIEL_CHECK_TRUE(any_diff);

  const auto base_pe = base_eval.PriorAndEvaluate(*state);
  const auto semantic_pe = semantic_eval.PriorAndEvaluate(*state);
  SPIEL_CHECK_EQ(base_pe.first.size(), semantic_pe.first.size());
  any_diff = false;
  for (size_t i = 0; i < base_pe.first.size(); ++i) {
    if (std::abs(base_pe.first[i].second - semantic_pe.first[i].second) > 1e-4) {
      any_diff = true;
      break;
    }
  }
  SPIEL_CHECK_TRUE(any_diff);

  std::cout << "Evaluator semantic scoring verification passed.\n";
}

void TestSemanticDualHeadMigrationAndStagedGradients(const std::filesystem::path& output_dir) {
  // 1. Source model: 5580, with semantic scorer (v2).
  auto source = std::make_shared<SharedDunePolicyValueNetImpl>(
      5580, 2048, 2391, 8, /*use_nonlinear=*/false, /*with_aux_heads=*/false,
      /*head_init_seed=*/0, /*with_semantic_scorer=*/true);
  {
    torch::NoGradGuard guard;
    source->semantic_scorer_->out_layer->weight.fill_(0.25f);
    if (source->semantic_scorer_->out_layer->bias.defined()) {
      source->semantic_scorer_->out_layer->bias.fill_(0.1f);
    }
  }

  // 2. Target model: 9182 (kFullPublicInformationStateSize), with semantic scorer (v3).
  auto target = std::make_shared<SharedDunePolicyValueNetImpl>(
      kFullPublicInformationStateSize, 2048, 2391, 8, /*use_nonlinear=*/false, /*with_aux_heads=*/false,
      /*head_init_seed=*/0, /*with_semantic_scorer=*/true);

  // Copy weights: input layer via CopyInputToFullPublicInformation; matching named parameters bitwise.
  {
    torch::NoGradGuard guard;
    CopyInputToFullPublicInformation(target->input_layer->weight, source->input_layer->weight, MarketAppendixMode::kNone);
    if (target->input_layer->bias.defined() && source->input_layer->bias.defined()) {
      target->input_layer->bias.copy_(source->input_layer->bias);
    }
    auto source_params = source->named_parameters();
    for (auto& item : target->named_parameters()) {
      if (item.key() != "input_layer.weight" && item.key() != "input_layer.bias" &&
          item.key() != "semantic_scorer.out_layer_ext.weight" && item.key() != "semantic_scorer.out_layer_ext.bias") {
        if (source_params.contains(item.key())) {
          item.value().copy_(source_params[item.key()]);
        }
      }
    }
  }

  // A. Verify out_layer_ext is initialized exactly to zero
  SPIEL_CHECK_EQ(target->semantic_scorer_->out_layer_ext->weight.abs().sum().item<double>(), 0.0);
  if (target->semantic_scorer_->out_layer_ext->bias.defined()) {
    SPIEL_CHECK_EQ(target->semantic_scorer_->out_layer_ext->bias.abs().sum().item<double>(), 0.0);
  }

  source->eval();
  target->eval();

  // B. Forward parity verification on a simulated input and candidate actions
  auto game = LoadGame("dune_imperium");
  auto state = game->NewInitialState();
  while (!state->IsTerminal()) {
    if (state->IsChanceNode()) {
      state->ApplyAction(state->ChanceOutcomes().front().first);
      continue;
    }
    const auto legal = state->LegalActions();
    const auto* dune = dynamic_cast<const DuneImperiumState*>(state.get());
    dune_semantic::CandidateActionData cand_data;
    dune_semantic::ExtractCandidateDescriptors(*dune, legal, &cand_data);
    bool has_supported = false;
    for (uint8_t s : cand_data.supported) {
      if (s != 0) { has_supported = true; break; }
    }
    if (has_supported) break;
    state->ApplyAction(legal.front());
  }

  auto* dune_st = dynamic_cast<const DuneImperiumState*>(state.get());
  SPIEL_CHECK_TRUE(dune_st != nullptr);
  const auto legal_actions = state->LegalActions();

  std::vector<float> old_obs(5580, 0.0f);
  state->InformationStateTensor(state->CurrentPlayer(), absl::MakeSpan(old_obs));
  std::vector<float> new_obs(kFullPublicInformationStateSize, 0.0f);
  dune_st->InformationStateTensorWithAppendix(
      state->CurrentPlayer(), MarketAppendixMode::kFullPublicInformationV3, absl::MakeSpan(new_obs));

  std::mutex src_m, tgt_m;
  DeterministicEvaluator src_eval(source, torch::kCPU, &src_m, nullptr, nullptr, false);
  DeterministicEvaluator tgt_eval(target, torch::kCPU, &tgt_m, nullptr, nullptr, false);

  dune_semantic::CandidateActionData cdata_v2, cdata_v3;
  dune_semantic::ExtractCandidateDescriptors(*dune_st, legal_actions, &cdata_v2, dune_semantic::kDescriptorSchemaVersionV2);
  dune_semantic::ExtractCandidateDescriptors(*dune_st, legal_actions, &cdata_v3, dune_semantic::kDescriptorSchemaVersionV3);

  auto res_src = src_eval.EvaluateWithActions(old_obs, &cdata_v2);
  auto res_tgt = tgt_eval.EvaluateWithActions(new_obs, &cdata_v3);

  double max_delta = 0.0;
  for (Action a : legal_actions) {
    double d = std::abs(static_cast<double>(res_src.logits[a] - res_tgt.logits[a]));
    max_delta = std::max(max_delta, d);
  }
  std::cout << "Dual-head migration forward parity on 5580->9182 expansion: max logit delta = "
            << max_delta << " (tolerance 1e-4)\n";
  SPIEL_CHECK_LT(max_delta, 1e-4);

  // C. Staged Gradient Verification:
  // Zero-initialized out_layer_ext blocks embedding gradients on the first backward pass.
  // First update out_layer_ext; then verify embedding gradients on a subsequent pass.
  target->train();
  target->zero_grad();

  int batch_size = 1;
  torch::Tensor trunk = torch::randn({batch_size, 2048});
  torch::Tensor inout_logits = torch::zeros({batch_size, 2391});
  torch::Tensor batch_indices = torch::tensor({0}, torch::kLong);
  torch::Tensor action_indices = torch::tensor({705}, torch::kLong);
  torch::Tensor feat_tensor = torch::zeros({1, 56});
  torch::Tensor card_ids = torch::tensor({dune_semantic::kIntrigueCardVocabOffset + 5}, torch::kLong);
  torch::Tensor space_ids = torch::tensor({0}, torch::kLong);
  torch::Tensor supported_mask = torch::tensor({1.0f}, torch::kFloat32);
  torch::Tensor is_ext_mask = torch::tensor({true}, torch::kBool);

  target->semantic_scorer_->ComputeAndAddCorrections(
      trunk, batch_indices, action_indices, feat_tensor, card_ids, space_ids,
      supported_mask, inout_logits, is_ext_mask);

  torch::Tensor loss1 = inout_logits.index_select(1, action_indices).sum();
  loss1.backward();

  // Pass 1:
  SPIEL_CHECK_GT(target->semantic_scorer_->out_layer_ext->weight.grad().abs().sum().item<double>(), 0.0);
  double card_embed_grad_pass1 = target->semantic_scorer_->card_embedding->weight.grad()
                                     .slice(0, 128, 192).abs().sum().item<double>();
  SPIEL_CHECK_EQ(card_embed_grad_pass1, 0.0);

  // Update out_layer_ext
  {
    torch::NoGradGuard guard;
    target->semantic_scorer_->out_layer_ext->weight.add_(
        target->semantic_scorer_->out_layer_ext->weight.grad() * 0.1f);
  }
  target->zero_grad();

  // Pass 2:
  torch::Tensor inout_logits2 = torch::zeros({batch_size, 2391});
  target->semantic_scorer_->ComputeAndAddCorrections(
      trunk, batch_indices, action_indices, feat_tensor, card_ids, space_ids,
      supported_mask, inout_logits2, is_ext_mask);

  torch::Tensor loss2 = inout_logits2.index_select(1, action_indices).sum();
  loss2.backward();

  double card_embed_grad_pass2 = target->semantic_scorer_->card_embedding->weight.grad()
                                     .slice(0, 128, 192).abs().sum().item<double>();
  std::cout << "Staged gradient test passed: pass 1 embed grad = " << card_embed_grad_pass1
            << ", pass 2 embed grad = " << card_embed_grad_pass2 << "\n";
  SPIEL_CHECK_GT(card_embed_grad_pass2, 0.0);
}

void ThrowingErrorHandler(const std::string& msg) {
  throw std::runtime_error(msg);
}

void ExitingErrorHandler(const std::string& msg) {
  std::cerr << "Spiel Fatal Error: " << msg << std::endl;
  std::exit(1);
}

void TestVersionAwareScorerDeserialization(const std::filesystem::path& temp_dir) {
  std::filesystem::create_directories(temp_dir);

  auto scorer_v2 = std::make_shared<dune_semantic::SemanticActionScorerImpl>();
  const std::string v2_path = (temp_dir / "scorer_v2.pt").string();
  {
    torch::serialize::OutputArchive out_arch;
    auto save_child = [&](const std::string& name, auto& mod) {
      torch::serialize::OutputArchive c;
      mod->save(c);
      out_arch.write(name, c);
    };
    save_child("card_embedding", scorer_v2->card_embedding);
    save_child("space_embedding", scorer_v2->space_embedding);
    save_child("desc_proj", scorer_v2->desc_proj);
    save_child("desc_ln", scorer_v2->desc_ln);
    save_child("trunk_proj", scorer_v2->trunk_proj);
    save_child("trunk_ln", scorer_v2->trunk_ln);
    save_child("mlp1", scorer_v2->mlp1);
    save_child("mlp1_ln", scorer_v2->mlp1_ln);
    save_child("out_layer", scorer_v2->out_layer);
    out_arch.save_to(v2_path);
  }

  auto target_scorer_a = std::make_shared<dune_semantic::SemanticActionScorerImpl>();
  {
    torch::NoGradGuard guard;
    target_scorer_a->out_layer_ext->weight.fill_(1.0f);
  }
  {
    torch::serialize::InputArchive in_arch;
    in_arch.load_from(v2_path);
    target_scorer_a->Load(in_arch, dune_semantic::kDescriptorSchemaVersionV2);
  }
  SPIEL_CHECK_EQ(target_scorer_a->out_layer_ext->weight.abs().sum().item<double>(), 0.0);
  std::cout << "Version-aware scorer deserialization: v2-to-v3 explicit migration succeeded.\n";

  bool caught_rejection = false;
  open_spiel::SetErrorHandler(ThrowingErrorHandler);
  try {
    torch::serialize::InputArchive in_arch;
    in_arch.load_from(v2_path);
    target_scorer_a->Load(in_arch, dune_semantic::kDescriptorSchemaVersionV3);
  } catch (const std::exception& e) {
    caught_rejection = true;
    std::cout << "Version-aware scorer deserialization: missing out_layer_ext rejected as expected: "
              << e.what() << "\n";
  }
  open_spiel::SetErrorHandler(ExitingErrorHandler);
  SPIEL_CHECK_TRUE(caught_rejection);

  const std::string v3_path = (temp_dir / "scorer_v3.pt").string();
  {
    torch::NoGradGuard guard;
    target_scorer_a->out_layer_ext->weight.fill_(0.42f);
  }
  {
    torch::serialize::OutputArchive out_arch;
    target_scorer_a->save(out_arch);
    out_arch.save_to(v3_path);
  }
  auto target_scorer_c = std::make_shared<dune_semantic::SemanticActionScorerImpl>();
  {
    torch::serialize::InputArchive in_arch;
    in_arch.load_from(v3_path);
    target_scorer_c->Load(in_arch, dune_semantic::kDescriptorSchemaVersionV3);
  }
  SPIEL_CHECK_TRUE(torch::equal(target_scorer_c->out_layer_ext->weight,
                               target_scorer_a->out_layer_ext->weight));
  std::error_code ec;
  std::filesystem::remove_all(temp_dir, ec);
  std::cout << "Version-aware scorer deserialization: v3 full load succeeded.\n";
}

void TestFailClosedValidation(const std::filesystem::path& temp_dir) {
  std::filesystem::create_directories(temp_dir);

  // 1. Ambiguous 6215 checkpoint with kNone mode
  {
    auto m6215 = std::make_shared<SharedDunePolicyValueNetImpl>(
        kExpandedInformationStateSize, 32, 2391, 1);
    const std::string p6215 = (temp_dir / "ambiguous_6215.pt").string();
    torch::save(m6215, p6215);
    std::ofstream jf((temp_dir / "ambiguous_6215.json").string());
    jf << "{\"market_appendix_mode\": \"none\"}\n";
    jf.close();

    bool caught = false;
    open_spiel::SetErrorHandler(ThrowingErrorHandler);
    try {
      auto load_target = std::make_shared<SharedDunePolicyValueNetImpl>(
          kExpandedInformationStateSize, 32, 2391, 1);
      LoadModelCheckpointRobust(load_target, p6215, torch::kCPU);
    } catch (const std::exception& e) {
      caught = true;
      std::cout << "Fail-closed check passed: ambiguous 6215 rejected: " << e.what() << "\n";
    }
    open_spiel::SetErrorHandler(ExitingErrorHandler);
    SPIEL_CHECK_TRUE(caught);
  }

  // 2. 9182 checkpoint without sidecar
  {
    auto m9182 = std::make_shared<SharedDunePolicyValueNetImpl>(
        kFullPublicInformationStateSize, 32, 2391, 1, false, false, 0, true);
    const std::string p9182 = (temp_dir / "no_sidecar_9182.pt").string();
    torch::save(m9182, p9182);

    bool caught = false;
    open_spiel::SetErrorHandler(ThrowingErrorHandler);
    try {
      auto load_target = std::make_shared<SharedDunePolicyValueNetImpl>(
          kFullPublicInformationStateSize, 32, 2391, 1, false, false, 0, true);
      LoadModelCheckpointRobust(load_target, p9182, torch::kCPU);
    } catch (const std::exception& e) {
      caught = true;
      std::cout << "Fail-closed check passed: 9182 without sidecar rejected: " << e.what() << "\n";
    }
    open_spiel::SetErrorHandler(ExitingErrorHandler);
    SPIEL_CHECK_TRUE(caught);
  }

  // 3a. In-memory 9182 constructor without scorer is permitted for value-only critic / internal use,
  // but rejected when passed to an actor evaluator.
  {
    auto m = std::make_shared<SharedDunePolicyValueNetImpl>(
        kFullPublicInformationStateSize, 32, 2391, 1, false, false, 0, false);
    SPIEL_CHECK_FALSE(m->with_semantic_scorer_);
    torch::Tensor x = torch::zeros({1, kFullPublicInformationStateSize});
    auto out = m->forward(x);
    SPIEL_CHECK_EQ(out.values.numel(), 1);

    // But rejected when passed to an actor evaluator:
    bool caught_actor = false;
    open_spiel::SetErrorHandler(ThrowingErrorHandler);
    try {
      DuneNNEvaluator actor_eval(m, torch::kCPU);
    } catch (const std::exception& e) {
      caught_actor = true;
      std::cout << "Fail-closed check passed: 9182 without scorer rejected as actor: " << e.what() << "\n";
    }
    open_spiel::SetErrorHandler(ExitingErrorHandler);
    SPIEL_CHECK_TRUE(caught_actor);
  }

  // 3b. 9182 checkpoint without scorer in archive rejected
  {
    const std::string p9182_ns = (temp_dir / "no_scorer_9182.pt").string();
    {
      auto lin = torch::nn::Linear(kFullPublicInformationStateSize, 32);
      torch::save(lin, p9182_ns);
    }
    std::ofstream jf((temp_dir / "no_scorer_9182.json").string());
    jf << "{\"market_appendix_mode\": \"full_public_information_v3\","
       << "\"observation_dim\": 9182,"
       << "\"feature_schema_sha256\": \""
       << GetInformationStateSchema(dune_imperium::MarketAppendixMode::kFullPublicInformationV3).sha256 << "\"}\n";
    jf.close();

    bool caught = false;
    open_spiel::SetErrorHandler(ThrowingErrorHandler);
    try {
      auto load_target = std::make_shared<SharedDunePolicyValueNetImpl>(
          kFullPublicInformationStateSize, 32, 2391, 1, false, false, 0, true);
      LoadModelCheckpointRobust(load_target, p9182_ns, torch::kCPU);
    } catch (const std::exception& e) {
      caught = true;
      std::cout << "Fail-closed check passed: 9182 without scorer archive rejected: " << e.what() << "\n";
    }
    open_spiel::SetErrorHandler(ExitingErrorHandler);
    SPIEL_CHECK_TRUE(caught);
  }

  // 4. 9182 checkpoint with v2 scorer schema rejected
  {
    auto m9182_v2 = std::make_shared<SharedDunePolicyValueNetImpl>(
        kFullPublicInformationStateSize, 32, 2391, 1, false, false, 0, true);
    const std::string p9182_v2 = (temp_dir / "scorer_v2_9182.pt").string();
    torch::save(m9182_v2, p9182_v2);
    std::ofstream jf((temp_dir / "scorer_v2_9182.json").string());
    jf << "{\"market_appendix_mode\": \"full_public_information_v3\","
       << "\"enable_semantic_scorer\": true,"
       << "\"feature_schema_sha256\": \""
       << GetInformationStateSchema(dune_imperium::MarketAppendixMode::kFullPublicInformationV3).sha256 << "\","
       << "\"semantic_descriptor_schema\": \"semantic_action_v2\"}\n";
    jf.close();

    bool caught = false;
    open_spiel::SetErrorHandler(ThrowingErrorHandler);
    try {
      auto load_target = std::make_shared<SharedDunePolicyValueNetImpl>(
          kFullPublicInformationStateSize, 32, 2391, 1, false, false, 0, true);
      LoadModelCheckpointRobust(load_target, p9182_v2, torch::kCPU);
    } catch (const std::exception& e) {
      caught = true;
      std::cout << "Fail-closed check passed: 9182 with v2 scorer rejected: " << e.what() << "\n";
    }
    open_spiel::SetErrorHandler(ExitingErrorHandler);
    SPIEL_CHECK_TRUE(caught);
  }

  // 5. 6215 checkpoint missing feature_schema_sha256
  {
    auto m6215 = std::make_shared<SharedDunePolicyValueNetImpl>(
        kExpandedInformationStateSize, 32, 2391, 1);
    const std::string p6215_nohash = (temp_dir / "nohash_6215.pt").string();
    torch::save(m6215, p6215_nohash);
    std::ofstream jf((temp_dir / "nohash_6215.json").string());
    jf << "{\"market_appendix_mode\": \"ordered_market\"}\n";
    jf.close();

    bool caught = false;
    open_spiel::SetErrorHandler(ThrowingErrorHandler);
    try {
      auto load_target = std::make_shared<SharedDunePolicyValueNetImpl>(
          kExpandedInformationStateSize, 32, 2391, 1);
      LoadModelCheckpointRobust(load_target, p6215_nohash, torch::kCPU);
    } catch (const std::exception& e) {
      caught = true;
      std::cout << "Fail-closed check passed: 6215 missing feature_schema_sha256 rejected: " << e.what() << "\n";
    }
    open_spiel::SetErrorHandler(ExitingErrorHandler);
    SPIEL_CHECK_TRUE(caught);
  }

  // 6. Checkpoint with contradictory observation_dim
  {
    auto m6215 = std::make_shared<SharedDunePolicyValueNetImpl>(
        kExpandedInformationStateSize, 32, 2391, 1);
    const std::string p6215_baddim = (temp_dir / "baddim_6215.pt").string();
    torch::save(m6215, p6215_baddim);
    std::ofstream jf((temp_dir / "baddim_6215.json").string());
    jf << "{\"market_appendix_mode\": \"ordered_market\","
       << "\"feature_schema_sha256\": \""
       << GetInformationStateSchema(dune_imperium::MarketAppendixMode::kOrderedMarket).sha256 << "\","
       << "\"observation_dim\": 123}\n";
    jf.close();

    bool caught = false;
    open_spiel::SetErrorHandler(ThrowingErrorHandler);
    try {
      auto load_target = std::make_shared<SharedDunePolicyValueNetImpl>(
          kExpandedInformationStateSize, 32, 2391, 1);
      LoadModelCheckpointRobust(load_target, p6215_baddim, torch::kCPU);
    } catch (const std::exception& e) {
      caught = true;
      std::cout << "Fail-closed check passed: contradictory observation_dim rejected: " << e.what() << "\n";
    }
    open_spiel::SetErrorHandler(ExitingErrorHandler);
    SPIEL_CHECK_TRUE(caught);
  }

  // 7. Bare relative checkpoint filename finds bare .json sidecar
  {
    const std::filesystem::path bare_dir = temp_dir / "bare_test";
    std::filesystem::create_directories(bare_dir);
    auto m6215 = std::make_shared<SharedDunePolicyValueNetImpl>(
        kExpandedInformationStateSize, 32, 2391, 1);
    torch::save(m6215, (bare_dir / "model.pt").string());
    std::ofstream jf((bare_dir / "model.json").string());
    jf << "{\"market_appendix_mode\": \"ordered_market\","
       << "\"feature_schema_sha256\": \""
       << GetInformationStateSchema(dune_imperium::MarketAppendixMode::kOrderedMarket).sha256 << "\","
       << "\"observation_dim\": 6215}\n";
    jf.close();

    auto old_cwd = std::filesystem::current_path();
    std::filesystem::current_path(bare_dir);
    try {
      auto load_target = std::make_shared<SharedDunePolicyValueNetImpl>(
          kExpandedInformationStateSize, 32, 2391, 1);
      LoadModelCheckpointRobust(load_target, "model.pt", torch::kCPU);
      SPIEL_CHECK_TRUE(load_target->market_appendix_mode_ ==
                       dune_imperium::MarketAppendixMode::kOrderedMarket);
      std::cout << "Bare relative checkpoint filename test passed: resolved model.json correctly.\n";
    } catch (...) {
      std::filesystem::current_path(old_cwd);
      throw;
    }
    std::filesystem::current_path(old_cwd);
  }

  // 8. Evaluators reject 9182 model without active v3 scorer
  {
    auto m9182 = std::make_shared<SharedDunePolicyValueNetImpl>(
        kFullPublicInformationStateSize, 32, 2391, 1, false, false, 0, true);
    m9182->with_semantic_scorer_ = false;

    // Test DuneNNEvaluator
    bool caught_eval = false;
    open_spiel::SetErrorHandler(ThrowingErrorHandler);
    try {
      DuneNNEvaluator eval(m9182, torch::kCPU);
    } catch (const std::exception& e) {
      caught_eval = true;
      std::cout << "DuneNNEvaluator 9182 without scorer rejected: " << e.what() << "\n";
    }
    open_spiel::SetErrorHandler(ExitingErrorHandler);
    SPIEL_CHECK_TRUE(caught_eval);

    // Test BatchedEvaluator
    bool caught_beval = false;
    open_spiel::SetErrorHandler(ThrowingErrorHandler);
    try {
      BatchedEvaluator beval(m9182, 1, 10, torch::kCPU, nullptr);
    } catch (const std::exception& e) {
      caught_beval = true;
      std::cout << "BatchedEvaluator 9182 without scorer rejected: " << e.what() << "\n";
    }
    open_spiel::SetErrorHandler(ExitingErrorHandler);
    SPIEL_CHECK_TRUE(caught_beval);

    // Test DeterministicEvaluator
    bool caught_deval = false;
    open_spiel::SetErrorHandler(ThrowingErrorHandler);
    try {
      std::mutex m;
      DeterministicEvaluator deval(m9182, torch::kCPU, &m);
    } catch (const std::exception& e) {
      caught_deval = true;
      std::cout << "DeterministicEvaluator 9182 without scorer rejected: " << e.what() << "\n";
    }
    open_spiel::SetErrorHandler(ExitingErrorHandler);
    SPIEL_CHECK_TRUE(caught_deval);
  }

  std::error_code ec;
  std::filesystem::remove_all(temp_dir, ec);
}

void TestPopulatedOptimizerMigration(const std::filesystem::path& temp_dir) {
  std::filesystem::create_directories(temp_dir);

  auto src_model = std::make_shared<SharedDunePolicyValueNetImpl>(
      5580, 64, 2391, 1, /*use_nonlinear=*/false, /*with_aux_heads=*/true);
  auto src_opt = MakeDuneOptimizer(src_model);

  torch::Tensor dummy_input = torch::randn({2, 5580});
  auto out = src_model->forward(dummy_input);
  torch::Tensor loss = out.logits.sum() + out.values.sum();
  loss.backward();
  src_opt->step();

  const std::string optim_path = (temp_dir / "source_populated.optim.pt").string();
  torch::save(*src_opt, optim_path);

  auto target_model = std::make_shared<SharedDunePolicyValueNetImpl>(
      kFullPublicInformationStateSize, 64, 2391, 1, /*use_nonlinear=*/false,
      /*with_aux_heads=*/true, /*head_init_seed=*/0, /*with_semantic_scorer=*/true);
  auto target_opt = MakeDuneOptimizer(target_model);

  bool migrated = LoadOptimizerCheckpointMigrating(
      target_model, *target_opt, optim_path, torch::kCPU);
  SPIEL_CHECK_TRUE(migrated);

  auto input_w_key = target_model->input_layer->weight.unsafeGetTensorImpl();
  SPIEL_CHECK_GT(target_opt->state().count(input_w_key), 0);
  auto* adam_state = dynamic_cast<torch::optim::AdamWParamState*>(
      target_opt->state()[input_w_key].get());
  SPIEL_CHECK_TRUE(adam_state != nullptr);
  SPIEL_CHECK_TRUE(adam_state->exp_avg().defined());
  SPIEL_CHECK_EQ(adam_state->exp_avg().size(0), 64);
  SPIEL_CHECK_EQ(adam_state->exp_avg().size(1), kFullPublicInformationStateSize);

  auto* src_adam_state = dynamic_cast<torch::optim::AdamWParamState*>(
      src_opt->state()[src_model->input_layer->weight.unsafeGetTensorImpl()].get());
  SPIEL_CHECK_TRUE(torch::equal(
      adam_state->exp_avg().slice(1, 0, 5580),
      src_adam_state->exp_avg()));

  SPIEL_CHECK_EQ(
      adam_state->exp_avg().slice(1, 5580, kFullPublicInformationStateSize).abs().sum().item<double>(), 0.0);
  SPIEL_CHECK_EQ(
      adam_state->exp_avg_sq().slice(1, 5580, kFullPublicInformationStateSize).abs().sum().item<double>(), 0.0);

  auto ext_w_key = target_model->semantic_scorer_->out_layer_ext->weight.unsafeGetTensorImpl();
  SPIEL_CHECK_GT(target_opt->state().count(ext_w_key), 0);
  auto* ext_state = dynamic_cast<torch::optim::AdamWParamState*>(
      target_opt->state()[ext_w_key].get());
  SPIEL_CHECK_TRUE(ext_state != nullptr);
  SPIEL_CHECK_EQ(ext_state->step(), 0);
  SPIEL_CHECK_EQ(ext_state->exp_avg().abs().sum().item<double>(), 0.0);
  SPIEL_CHECK_EQ(ext_state->exp_avg_sq().abs().sum().item<double>(), 0.0);

  target_model->train();
  target_opt->zero_grad();
  torch::Tensor target_input = torch::randn({2, kFullPublicInformationStateSize});
  auto target_out = target_model->forward(target_input);
  torch::Tensor target_loss = target_out.logits.sum() + target_out.values.sum();
  target_loss.backward();
  target_opt->step();

  std::error_code ec;
  std::filesystem::remove_all(temp_dir, ec);
  std::cout << "Populated optimizer migration verification passed: moments expanded and target step succeeded.\n";
}

void TestOptimizerMigration6255To9182(const std::filesystem::path& temp_dir) {
  std::filesystem::create_directories(temp_dir);

  auto src_model = std::make_shared<SharedDunePolicyValueNetImpl>(
      6255, 64, 2391, 1, /*use_nonlinear=*/false, /*with_aux_heads=*/true,
      /*head_init_seed=*/0, /*with_semantic_scorer=*/true);
  auto src_opt = MakeDuneOptimizer(src_model);

  torch::Tensor dummy_input = torch::randn({2, 6255});
  auto out = src_model->forward(dummy_input);
  torch::Tensor loss = out.logits.sum() + out.values.sum();
  loss.backward();
  src_opt->step();

  const std::string optim_path = (temp_dir / "src_6255.optim.pt").string();
  torch::save(*src_opt, optim_path);

  auto target_model = std::make_shared<SharedDunePolicyValueNetImpl>(
      kFullPublicInformationStateSize, 64, 2391, 1, /*use_nonlinear=*/false,
      /*with_aux_heads=*/true, /*head_init_seed=*/0, /*with_semantic_scorer=*/true);
  auto target_opt = MakeDuneOptimizer(target_model);

  bool migrated = LoadOptimizerCheckpointMigrating(
      target_model, *target_opt, optim_path, torch::kCPU);
  SPIEL_CHECK_TRUE(migrated);

  auto input_w_key = target_model->input_layer->weight.unsafeGetTensorImpl();
  SPIEL_CHECK_GT(target_opt->state().count(input_w_key), 0);
  auto* adam_state = dynamic_cast<torch::optim::AdamWParamState*>(
      target_opt->state()[input_w_key].get());
  SPIEL_CHECK_TRUE(adam_state != nullptr);
  SPIEL_CHECK_TRUE(adam_state->exp_avg().defined());
  SPIEL_CHECK_EQ(adam_state->exp_avg().size(0), 64);
  SPIEL_CHECK_EQ(adam_state->exp_avg().size(1), kFullPublicInformationStateSize);

  auto* src_adam_state = dynamic_cast<torch::optim::AdamWParamState*>(
      src_opt->state()[src_model->input_layer->weight.unsafeGetTensorImpl()].get());
  SPIEL_CHECK_TRUE(torch::equal(
      adam_state->exp_avg().slice(1, 0, 6255),
      src_adam_state->exp_avg()));

  SPIEL_CHECK_EQ(
      adam_state->exp_avg().slice(1, 6255, kFullPublicInformationStateSize).abs().sum().item<double>(), 0.0);
  SPIEL_CHECK_EQ(
      adam_state->exp_avg_sq().slice(1, 6255, kFullPublicInformationStateSize).abs().sum().item<double>(), 0.0);

  // A full AdamW update on target_model succeeds without any shape mismatch error
  target_model->train();
  target_opt->zero_grad();
  torch::Tensor target_input = torch::randn({2, kFullPublicInformationStateSize});
  auto target_out = target_model->forward(target_input);
  torch::Tensor target_loss = target_out.logits.sum() + target_out.values.sum();
  target_loss.backward();
  target_opt->step();

  std::error_code ec;
  std::filesystem::remove_all(temp_dir, ec);
  std::cout << "6255-to-9182 optimizer migration verification passed: moments expanded and target step succeeded.\n";
}

void TestBootstrappedBundleVerification(const std::filesystem::path& bundle_dir) {
  const std::filesystem::path model_path = bundle_dir / "ppo_model_update_400.pt";
  const std::filesystem::path optim_path = bundle_dir / "ppo_optimizer_update_400.pt";
  SPIEL_CHECK_TRUE(std::filesystem::exists(model_path));
  SPIEL_CHECK_TRUE(std::filesystem::exists(optim_path));

  auto model = std::make_shared<SharedDunePolicyValueNetImpl>(
      kFullPublicInformationStateSize, 2048, 2391, 8, /*use_nonlinear=*/false,
      /*with_aux_heads=*/false, /*head_init_seed=*/0, /*with_semantic_scorer=*/true);
  LoadModelCheckpointRobust(model, model_path.string(), torch::kCPU);
  SPIEL_CHECK_TRUE(model->market_appendix_mode_ == dune_imperium::MarketAppendixMode::kFullPublicInformationV3);
  SPIEL_CHECK_TRUE(model->with_semantic_scorer_);
  SPIEL_CHECK_TRUE(model->semantic_scorer_->out_layer_ext->weight.defined());

  // Verify actor evaluator accepts the loaded model
  DuneNNEvaluator evaluator(model, torch::kCPU);

  // Verify optimizer reload: already 9182, LoadOptimizerCheckpointMigrating returns false
  auto opt = MakeDuneOptimizer(model);
  bool needs_migration = LoadOptimizerCheckpointMigrating(
      model, *opt, optim_path.string(), torch::kCPU);
  SPIEL_CHECK_FALSE(needs_migration);

  torch::load(*opt, optim_path.string(), torch::kCPU);
  SPIEL_CHECK_EQ(opt->state().size(), 88);

  // Verify step succeeds
  model->train();
  opt->zero_grad();
  torch::Tensor x = torch::randn({2, kFullPublicInformationStateSize});
  auto out = model->forward(x);
  torch::Tensor loss = out.logits.sum() + out.values.sum();
  loss.backward();
  opt->step();
  std::cout << "Real bootstrapped B400 bundle reload, evaluation, and update step passed successfully.\n";
}

void TestScientificBreakthroughAndStitchedHorrorSemanticDescriptors() {
  auto game = LoadGame("dune_imperium(enable_immortality=true)");
  auto state = game->NewInitialState();
  while (state->IsChanceNode()) {
    state->ApplyAction(state->ChanceOutcomes().front().first);
  }
  auto* impl = dynamic_cast<DuneImperiumState*>(state.get());
  SPIEL_CHECK_TRUE(impl != nullptr);
  impl->SetPhaseForTesting(GamePhase::kRevealTurns);
  impl->SetCurrentPlayerForTesting(0);
  impl->SetSpecimensForTesting(0, 10);

  for (bool swap_slots : {false, true}) {
    if (!swap_slots) {
      impl->SetTleilaxuRowForTesting({11, 13});
    } else {
      impl->SetTleilaxuRowForTesting({13, 11});
    }

    const auto legal = impl->LegalActions();
    SPIEL_CHECK_TRUE(std::find(legal.begin(), legal.end(), kActionTleilaxuAcquire0 + 1) != legal.end());
    SPIEL_CHECK_TRUE(std::find(legal.begin(), legal.end(), kActionTleilaxuAcquire0 + 2) != legal.end());

    dune_semantic::CandidateActionData cand_data;
    dune_semantic::ExtractCandidateDescriptors(*impl, legal, &cand_data);

    for (size_t i = 0; i < cand_data.actions.size(); ++i) {
      if (cand_data.actions[i] == kActionTleilaxuAcquire0 + 1 ||
          cand_data.actions[i] == kActionTleilaxuAcquire0 + 2) {
        SPIEL_CHECK_EQ(cand_data.supported[i], 0);
        SPIEL_CHECK_TRUE(cand_data.roles[i] == dune_semantic::ActionRole::kUnsupported);
        SPIEL_CHECK_FLOAT_EQ(cand_data.features[i * dune_semantic::kSemanticFeatDim + 0], 1.0f);
      }
    }
  }
  std::cout << "PASS: TestScientificBreakthroughAndStitchedHorrorSemanticDescriptors\n";
}

}  // namespace
}  // namespace open_spiel

int main(int argc, char** argv) {
  torch::set_num_threads(2);
  torch::manual_seed(901);
  const std::filesystem::path artifacts_dir = argc > 1 ? argv[1] : "identity_test_artifacts";
  open_spiel::TestInputMigration();
  open_spiel::TestIncumbentMigration(artifacts_dir);
  open_spiel::TestEvaluatorSemanticScoring();
  open_spiel::TestScientificBreakthroughAndStitchedHorrorSemanticDescriptors();
  open_spiel::TestSemanticDualHeadMigrationAndStagedGradients(artifacts_dir);
  open_spiel::TestVersionAwareScorerDeserialization(artifacts_dir / "version_aware");
  open_spiel::TestFailClosedValidation(artifacts_dir / "fail_closed");
  open_spiel::TestPopulatedOptimizerMigration(artifacts_dir / "optim_mig");
  open_spiel::TestOptimizerMigration6255To9182(artifacts_dir / "optim_mig_6255");
  if (argc > 2) {
    const std::filesystem::path b400_dir(argv[2]);
    open_spiel::TestBootstrappedBundleVerification(b400_dir);
  }
  std::error_code ec;
  std::filesystem::remove_all(artifacts_dir, ec);
  return 0;
}

