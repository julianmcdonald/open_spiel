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
#include "open_spiel/games/dune_imperium/dune_imperium.h"
#include "open_spiel/spiel.h"

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

}  // namespace
}  // namespace open_spiel

int main(int argc, char** argv) {
  torch::set_num_threads(2);
  torch::manual_seed(901);
  open_spiel::TestInputMigration();
  open_spiel::TestIncumbentMigration(argc > 1 ? argv[1] : "identity_test_artifacts");
  open_spiel::TestEvaluatorSemanticScoring();
  return 0;
}
