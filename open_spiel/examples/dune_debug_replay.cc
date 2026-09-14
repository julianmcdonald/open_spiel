#include <iostream>
#include <fstream>
#include <vector>
#include <iomanip>
#include "open_spiel/spiel.h"
#include "open_spiel/utils/json.h"
#include "open_spiel/games/dune_imperium/dune_imperium.h"
#include "dune_network.h"
#include "dune_batched_evaluator.h"
#include "dune_search_session.h"

using namespace open_spiel;

int main() {
  std::ifstream f("/tmp/ep678_d171.json");
  std::string str((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
  auto j = json::FromString(str);
  auto obj = j->GetObject();
  auto hist = obj.at("state_history").GetArray();

  std::shared_ptr<const Game> game = LoadGame("dune_imperium");
  auto state = game->NewInitialState();
  for (size_t i = 0; i < hist.size(); ++i) {
    Action a = static_cast<Action>(hist[i].GetInt());
    state->ApplyAction(a);
  }

  std::cout << "State reached. Current player: " << state->CurrentPlayer() << "\n";
  std::cout << "Legal actions: ";
  for (Action a : state->LegalActions()) std::cout << a << " ";
  std::cout << "\n";

  // Load model
  std::string ckpt = "/run/media/warcr/Storage/dune_drl_runtime/round7/gemini_semantic_v3_continuation_b2400_to_b4400_20260913_141123/checkpoints/ppo_model_update_5300.pt";
  torch::Device device(torch::kCUDA);
  std::shared_ptr<SharedDunePolicyValueNetImpl> model =
      std::make_shared<SharedDunePolicyValueNetImpl>(9182, 2048, 8, 2408);
  torch::load(model, ckpt, device);
  model->eval();
  model->to(device);

  std::shared_mutex mtx;
  auto coord = std::make_shared<BatchedEvaluator>(
      model, 16, 2, device, &mtx, 10.0f, true, false, false, true, false);
  auto eval = std::make_shared<BatchedNNEvaluator>(coord, 10.0f);

  DuneSearchConfig cfg;
  cfg.temperature = 0.0;
  auto session = std::make_unique<DuneSearchSession>(cfg, eval, DuneSearchBudgetMode::kPolicyOnly);

  DuneSearchResult res = session->SearchAndSelectWithDeadline(*state, std::numeric_limits<double>::infinity());
  std::cout << "res.diagnostics.selected_action: " << res.diagnostics.selected_action << "\n";
  std::cout << "res.diagnostics.raw_reference_action: " << res.diagnostics.raw_reference_action << "\n";
  std::cout << "res.policy size: " << res.policy.size() << "\n";
  for (const auto& ap : res.policy) {
    std::cout << "Action " << ap.first << ": " << std::setprecision(12) << ap.second << "\n";
  }

  return 0;
}
